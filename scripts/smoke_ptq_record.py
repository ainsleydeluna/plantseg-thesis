#!/usr/bin/env python3
"""LQP on synthetic data: scripts/run_ptq.py's bound path (src/quant/ptq_record.py), the official PTQ record gate
(src/eval/stage_artifacts.py int8_official_error) and scripts/ptq_calibration_range.py.

No PlantSeg data, no real checkpoint, no GPU, no TEST path. A synthetic TRAIN tree of 5,367 image/mask pairs (the
count for which scripts/build_calibration_lists.py marks its lists "registered", its :137 and :157), a synthetic
records repository (git: the four lists, the λ and α selections and the KD launch log, committed once) and
synthetic E1, E2 and E3 run directories stand in for the real inputs. ptq_record.RECORDS_ROOT points at the records
repository and src.quant.ptq.code_provenance answers a fixed identity (a clean checkout at a known commit), in this
process and in each driver process; the identity cases change that answer. The real PTQ core runs, each in its own
process, for E4 s42, E7 s42, the E4 d3 replicate, s10's unbound run and s11's E7-E2 s42; every other bound case
runs a stub core that writes small fake artifacts and the run-provenance fields the record reads. No TEST folder is
ever created (SL-1): a TEST surface exists only through os.path.lexists, patched for one path and restored. The
repository's git status is the same after the run as before it.

  s0 fixtures     the TRAIN tree, the registered lists, the committed records
  s1 old runners  run_e4.py and run_e7.py refuse every run (ruling Q1-R2)
  s2 E4 s42       the real core: files, the record of record (parent, list, code, environment, isolation),
                  its manifest entry, the official gate, VAL and TEST never opened
  s3 E7 s42       the real core: the selections, teacher_arch, the launch log
  s4 seeds 43     stub core: E7 s43 bound by the launch log (no α run_id rule); E4 s43
  s5 refusals     every refusal of the bound path and of the new run_ptq.py gates, each reached with every earlier
                  gate passing: exit 2, its code, nothing created, the core never called; the two STOPs
  s6 d3           the real core: the replicate's role, and run_ptq.py --compare against E4 s42
  s7 sensitivity  stub core: lists 43-45 at seed 42 make calibration_sensitivity records
  s8 official     int8_official_error on stub records; scripts/evaluate_model.py calls only it
  s9 range        scripts/ptq_calibration_range.py: written, and each refusal
  s10 hygiene     an unbound run, start to finish (the real core), never imports the record module; no test-named
                  path; repository untouched
  s11 E7-E2       AM-23 item 2 (L-AM23Q B-2), on records repositories of its own: the real core (the record of
                  role quant_descriptive, its parent block, manifest and planned entries, the official gate),
                  read_record's role coupling, each refusal of E7-E2's rule (stub core), and each single
                  condition of bind_e2's multi-condition refusals, alone (stub core; DL-24 F4-2)

    python -B scripts/smoke_ptq_record.py [--sections s0,s5] [--only CASE,CASE] [--cache-dir DIR]
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import make_tree, refusable, repo_state, safe_tmpdir, state_verdict  # noqa: E402

STATE_BEFORE = repo_state(REPO)          # the repository as the run found it (repository_untouched)
BASE = safe_tmpdir("smoke_ptq_record_")
TRAIN_IMAGES = 5367                      # configs/data.py SPLIT_SIZES["train"]: a registered build's count
TREE_SPEC = {"n_train": TRAIN_IMAGES, "seed": 5, "format": "smoke_ptq_record train tree/1"}
FAKE_HEAD = "c0de" * 10                  # the commit the patched identity reader answers
GIT_HEAD, RECORDS_COMMIT = "1" * 40, "2" * 40     # the synthetic E3 runs' launch commits
LAMBDA, ALPHA, E1_BEST, E3_BEST = 0.5, 25.0, 0.5, 0.41
E3_S42, E3_S43 = "e3_s42_lambda0.5_alpha25_80k_a1", "e3_s43_lambda0.5_alpha25_80k_a1"
E3_S43_A2, E3_S43_A3, E3_S43_A4 = (f"e3_s43_lambda0.5_alpha25_80k_a{n}" for n in (2, 3, 4))
WALL = {42: 1760000000.25, 43: 1760100000.5}
CK1, CK3 = "e1_student_best_iter80000.pt", "e3_student_best_iter80000.pt"
LAMBDA_REL, ALPHA_REL = "reports/derived/lambda_selection.json", "reports/derived/alpha_selection.json"
OTHER_SEL_REL, UNTRACKED_SEL_REL = "reports/derived/other_selection.json", "reports/derived/untracked_selection.json"
LOG_REL = "reports/kd_launch_log.jsonl"
# DL-24 D-3: one E3 s43 attempt per field of R8's set; its launch and launched lines differ from run_meta in that
# field alone ({launch line override}, {launched line override}); a changed stage, seed, horizon, lambda or alpha
# starts its own attempt group (attempt 1), the others continue the s43 group
OTHER_COMMIT = "3" * 40
LOG_FIELD_LINES = {"git_head": ({"code_pin": OTHER_COMMIT}, {"git_head": OTHER_COMMIT}),
                   "code_pin": ({"code_pin": OTHER_COMMIT}, {}),
                   "records_commit": ({}, {"records_commit": OTHER_COMMIT}),
                   "swept_key": ({"sweep": "alpha_cwd", "value": 10.0}, {}), "stage": ({"stage": "E2"}, {}),
                   "seed": ({"seed": 44}, {}), "horizon": ({"horizon": 160000}, {}),
                   "lambda_logit": ({"lambda_logit": 1.0}, {}), "alpha_cwd": ({"alpha_cwd": 10.0}, {})}
E3_S43_FIELD = {f: f"e3_s43_lambda0.5_alpha25_80k_{f}" for f in LOG_FIELD_LINES}
CLEAN = {"type": "clean", "name": None, "severity": None}
SECTIONS = ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11")

results: list[tuple[str, bool, str]] = []
FX: dict = {}                             # fixtures, built once per process
STUB = {"calls": 0, "mode": "plain"}
IDENT = {"commit": FAKE_HEAD, "clean": True}
AUDIT = {"on": False, "paths": []}
REFUSALS: list[dict] = []


def _audit(event, args):
    if AUDIT["on"] and event in ("open", "os.listdir", "os.scandir") and args:
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            AUDIT["paths"].append(os.fsdecode(p))


sys.addaudithook(_audit)


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), str(detail)))


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    return sha_bytes(Path(p).read_bytes())


def code_of(log: str) -> str | None:
    m = re.findall(r"RESULT: (?:REFUSED|STOP) \[([a-z0-9_]+)\]", log)
    return m[-1] if m else None


def last_result(log: str) -> str:
    lines = [ln for ln in log.splitlines() if ln.startswith(("RESULT", "REFUSING"))]
    return lines[-1][:150] if lines else log.strip()[-150:]


def cli(fn, argv) -> tuple[int, str]:
    """(exit code, output). An exception is reported, never raised: exit -1 and its text, so a mutation that
    crashes the command fails its check rather than skipping it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rc = fn(argv)
        except SystemExit as e:
            rc = int(e.code) if isinstance(e.code, int) else 2
        except Exception as e:                              # noqa: BLE001
            rc = -1
            print(f"RESULT: CRASHED -- {type(e).__name__}: {e}")
    return rc, buf.getvalue()


def run_main(argv) -> tuple[int, str]:
    from scripts.run_ptq import main
    return cli(main, argv)


def fresh_out(name: str) -> Path:
    """A never-created out-dir named `name` in its own new parent directory."""
    parent = BASE / "out" / f"case{len(list((BASE / 'out').iterdir())) if (BASE / 'out').exists() else 0:04d}"
    while parent.exists():
        parent = parent.with_name(parent.name + "x")
    parent.mkdir(parents=True)
    return parent / name


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
           "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
           "GIT_AUTHOR_DATE": "2026-10-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-10-01T00:00:00Z",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout


# ------------------------------------------------------------------ fixtures
def train_root(cache: Path | None) -> Path:
    """The 5,367-pair TRAIN tree: built once into the cache (or BASE). An incomplete tree is left alone."""
    parent = cache if cache is not None else BASE
    for i in range(100):
        d = parent / f"train_tree_{i}"
        marker = d / "complete.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == TREE_SPEC:
            return d / "data"
        if not d.exists():
            make_tree(d / "data", n_train=TRAIN_IMAGES, seed=TREE_SPEC["seed"])
            marker.write_text(json.dumps(TREE_SPEC), encoding="utf-8")
            return d / "data"
    raise RuntimeError(f"no usable TRAIN tree under {parent}")


def student_state(seed: int) -> dict:
    import torch
    from src.models.student import build_student
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}


def teacher_provenance(**over) -> dict:
    import scripts.preflight_distill as pd
    v = pd.TEACHER_VALUES
    tp = {"builder": pd.TEACHER_BUILDER, "ckpt_path": f"/teacher/{pd.TEACHER_FILE}", "ckpt_sha256": pd.TEACHER_SHA256,
          "ckpt_bytes": pd.TEACHER_BYTES, "expected_sha256": pd.TEACHER_SHA256,
          "config_path": "configs/teacher/segnext.py", "config_sha256": v["config_sha256"],
          "ham_kwargs": v["ham_kwargs"], "architecture_signature": v["architecture_signature"],
          "teacher_components_sha256": v["teacher_components_sha256"],
          "reused_module_hashes": v["reused_module_hashes"], "model_cfg_sha256": v["model_cfg_sha256"]}
    tp.update(over)
    return tp


def real_ckpt(kind: str) -> Path:
    """A random-weight student checkpoint in the E1 or E3 schema (the real loaders' input), built once."""
    import torch
    parent = FX["cache"] if FX["cache"] is not None else BASE
    p = parent / f"{kind}_student_v1.pt"
    marker = parent / f"{kind}_student_v1.done"
    if marker.is_file() and p.is_file():
        return p
    if kind == "e1":
        payload = {"iter": 80000, "model_state_dict": student_state(7), "optimizer_state_dict": {},
                   "scheduler_state_dict": {}, "scheduler": "poly", "best_val_miou_all_class": E1_BEST,
                   "num_classes": 116}
    else:
        payload = {"iter": 80000, "stage": "E3", "model_state_dict": student_state(11),
                   "cwd_projection_state_dict": {"weight": torch.randn(320, 160, 1, 1)},
                   "best_val_miou_all_class": E3_BEST, "num_classes": 116, "teacher_provenance": teacher_provenance()}
    tmp = p.with_name(p.name + f".{os.getpid()}.part")
    torch.save(payload, tmp)
    os.replace(tmp, p)
    marker.write_text("1", encoding="utf-8")
    return p


def e1_run(name: str, *, seed=42, max_iters=80000, poly_horizon=80000, val_iter=None, rows=1, mode="real",
           best=E1_BEST, payload=None, real=False, parent_dir=None, best_json=None) -> tuple[Path, str]:
    import torch
    d = (parent_dir or BASE / "runs") / name
    d.mkdir(parents=True)
    meta = {"event": "run_meta", "mode": mode, "seed": seed, "max_iters": max_iters, "batch_size": 16}
    if poly_horizon is not None:
        meta["poly_horizon"] = poly_horizon
    lines = [json.dumps(meta)] * rows + [json.dumps({"event": "val", "iter": max_iters if val_iter is None
                                                     else val_iter, "miou_all_class": best})]
    (d / "e1_telemetry.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "best.json").write_text(best_json if best_json is not None else
                                 json.dumps({"best_ckpt": CK1, "best_val_miou_all_class": best}), encoding="utf-8")
    if real:
        os.symlink(real_ckpt("e1"), d / CK1)
    else:
        torch.save(payload if payload is not None else
                   {"iter": 80000, "model_state_dict": {}, "best_val_miou_all_class": best, "num_classes": 116},
                   d / CK1)
    return d, sha_file(d / CK1)


def e3_meta(seed: int, **over) -> dict:
    meta = {"event": "run_meta", "mode": "real", "seed": seed, "stage": "E3", "max_iters": 80000,
            "poly_horizon": 80000, "lambda_logit": LAMBDA, "alpha_cwd": ALPHA, "wall_clock": WALL[seed],
            "git_head": GIT_HEAD, "teacher_provenance": teacher_provenance(), "teacher_mock": False, "arm": None,
            "descriptive": False, "parent_of_e4_e7": True, "records_commit": RECORDS_COMMIT}
    meta.update(over)
    return meta


def e3_run(name: str, *, seed: int, meta=None, run_end=True, checks=True, abort=False, best=E3_BEST, payload=None,
           real=False, parent_dir=None) -> tuple[Path, str]:
    import torch
    d = (parent_dir or BASE / "runs") / name
    d.mkdir(parents=True)
    (d / "e3_run_meta.jsonl").write_text(json.dumps(meta if meta is not None else e3_meta(seed)) + "\n",
                                         encoding="utf-8")
    tel = [{"event": "train", "iter": 1, "loss": 1.0}]
    if abort:
        tel.append({"event": "run_abort", "iter": 2})
    if run_end:
        tel.append({"event": "run_end", "iter": 80000, "best_val_miou_all_class": best, "best_ckpt": CK3,
                    "checks_passed": checks, "wall_clock_start": WALL.get(seed, 0.0),
                    "wall_clock_end": WALL.get(seed, 0.0) + 3600.0})
    (d / "e3_telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tel), encoding="utf-8")
    (d / "best.json").write_text(json.dumps({"best_ckpt": CK3, "best_val_miou_all_class": best}), encoding="utf-8")
    if real:
        os.symlink(real_ckpt("e3"), d / CK3)
    else:
        torch.save(payload if payload is not None else
                   {"iter": 80000, "stage": "E3", "model_state_dict": {}, "best_val_miou_all_class": best,
                    "num_classes": 116, "teacher_provenance": teacher_provenance()}, d / CK3)
    return d, sha_file(d / CK3)


def launch_line(rid: str, seed: int, *, sweep=None, value=None, attempt=1) -> dict:
    return {"format": "kd_launch_log/1", "event": "launch", "run_id": rid, "stage": "E3", "seed": seed,
            "horizon": 80000, "sweep": sweep, "value": value, "lambda_logit": LAMBDA, "alpha_cwd": ALPHA,
            "attempt": attempt, "schedule": "T", "code_pin": GIT_HEAD, "ckpt_dir": f"/runs/{rid}",
            "am8a_report": None}


def launched_line(rid: str, run_meta_sha256: str, launch_time: float) -> dict:
    return {"event": "launched", "run_id": rid, "run_meta_sha256": run_meta_sha256, "launch_time_utc": launch_time,
            "git_head": GIT_HEAD, "records_commit": RECORDS_COMMIT, "decision_date": None,
            "launch_order_position": None}


def setup(cache: Path | None) -> None:
    """The shared fixtures: TRAIN tree, parents E1 s42 / E3 s42 / E3 s43 / E1 s43, the records repository."""
    if FX:
        return
    FX["cache"] = cache
    FX["root"] = train_root(cache)
    os.environ["PLANTSEG_DATA_ROOT"] = str(FX["root"])
    FX["e1_s42"], FX["e1_sha"] = e1_run("e1_s42", real=True)
    FX["e1_s43"], FX["e1_s43_sha"] = e1_run("e1_s43", seed=43)
    FX["e3_s42"], FX["e3_sha"] = e3_run(E3_S42, seed=42, real=True)
    FX["e3_s43"], FX["e3_s43_sha"] = e3_run(E3_S43, seed=43)
    R = BASE / "records"
    (R / "configs" / "calibration").mkdir(parents=True)
    from scripts.build_calibration_lists import main as build_lists
    rc, log = cli(build_lists, ["--data-root", str(FX["root"]), "--out-dir", str(R / "configs" / "calibration"),
                                "--generated-utc", "2026-10-01T00:00:00Z"])
    if rc != 0:
        raise RuntimeError(f"the registered lists were not built: {last_result(log)}")
    d = R / "reports" / "derived"
    d.mkdir(parents=True)
    (d / "lambda_selection.json").write_text(json.dumps(
        {"format": "lambda_selection/1", "winner": {"lambda": LAMBDA, "run_id": "e2_s42_lambda0.5_80k_a1"}}),
        encoding="utf-8")
    (d / "alpha_selection.json").write_text(json.dumps(
        {"format": "alpha_selection/1", "lambda_logit": LAMBDA,
         "winner": {"alpha": ALPHA, "run_id": E3_S42, "best_val": E3_BEST, "ckpt_sha256": FX["e3_sha"],
                    "checkpoint_present": True}}), encoding="utf-8")
    (d / "other_selection.json").write_text(json.dumps({"format": "something_else/1", "winner": {}}),
                                            encoding="utf-8")
    m42, m43 = sha_file(FX["e3_s42"] / "e3_run_meta.jsonl"), sha_file(FX["e3_s43"] / "e3_run_meta.jsonl")
    log_lines = [launch_line(E3_S42, 42, sweep="alpha_cwd", value=ALPHA), launched_line(E3_S42, m42, WALL[42]),
                 launch_line(E3_S43, 43), launched_line(E3_S43, m43, WALL[43]),
                 launch_line(E3_S43_A2, 43, attempt=2),
                 launch_line(E3_S43_A3, 43, attempt=3), launched_line(E3_S43_A3, m43, WALL[43] + 60.0),
                 launch_line(E3_S43_A4, 43, attempt=4), launched_line(E3_S43_A4, m43, WALL[43]),
                 {"event": "stopped", "run_id": E3_S43_A4, "report": {"path": "reports/stop.md", "sha256": "0" * 64}}]
    n = 4                                               # E3_S43, A2, A3 and A4: attempts 1-4 of the s43 group
    for f, (line_over, launched_over) in LOG_FIELD_LINES.items():
        regroup = bool({"stage", "seed", "horizon", "lambda_logit", "alpha_cwd"} & set(line_over))
        n += 0 if regroup else 1
        log_lines += [{**launch_line(E3_S43_FIELD[f], 43, attempt=1 if regroup else n), **line_over},
                      {**launched_line(E3_S43_FIELD[f], m43, WALL[43]), **launched_over}]
    (R / LOG_REL).write_bytes("".join(json.dumps(x) + "\n" for x in log_lines).encode("utf-8"))
    tracked = [f"configs/calibration/ptq_calibration_seed{s}.json" for s in (42, 43, 44, 45)] + [
        LAMBDA_REL, ALPHA_REL, OTHER_SEL_REL, LOG_REL]
    git(R, "init", "-q")
    git(R, "add", "--", *tracked)
    git(R, "commit", "-q", "-m", "synthetic records")
    (R / UNTRACKED_SEL_REL).write_text((d / "alpha_selection.json").read_text(encoding="utf-8"), encoding="utf-8")
    FX.update(records=R, tracked=tracked, sel_sha={k: sha_file(R / rel) for k, rel in
                                                   (("lambda", LAMBDA_REL), ("alpha", ALPHA_REL),
                                                    ("other", OTHER_SEL_REL), ("untracked", UNTRACKED_SEL_REL))})
    from src.quant import ptq
    from src.quant.ptq_record import record_code_paths
    FX["file_shas"] = {p: sha_file(REPO / p) for p in record_code_paths() if (REPO / p).is_file()}
    from src.quant.qat import IMAGE_DIGEST_OF_RECORD
    FX["digest"] = IMAGE_DIGEST_OF_RECORD
    smoke = json.loads(list_path(42).read_text(encoding="utf-8"))
    smoke["artifact_status"] = "smoke"
    (BASE / "smoke_list").mkdir()
    (BASE / "smoke_list" / "ptq_calibration_seed42.json").write_text(json.dumps(smoke), encoding="utf-8")
    FX["smoke_list"] = BASE / "smoke_list" / "ptq_calibration_seed42.json"
    FX["names"] = ptq.artifact_names


def list_path(seed: int) -> Path:
    return BASE / "records" / "configs" / "calibration" / f"ptq_calibration_seed{seed}.json"


def checksum(seed: int) -> str:
    return json.loads(list_path(seed).read_text(encoding="utf-8"))["checksum_sha256"]


def parent_of(stage: str, seed: int) -> tuple[Path, str]:
    return {("E4", 42): (FX["e1_s42"], FX["e1_sha"]), ("E4", 43): (FX["e1_s43"], FX["e1_s43_sha"]),
            ("E7", 42): (FX["e3_s42"], FX["e3_sha"]), ("E7", 43): (FX["e3_s43"], FX["e3_s43_sha"])}[(stage, seed)]


def argv(stage: str = "E4", *, seed: int = 42, list_seed: int = 42, out=None, **flags) -> list[str]:
    """A passing bound-path command; each keyword names a flag (underscores for dashes), None drops it."""
    run_dir, sha = parent_of(stage, seed) if (stage, seed) in (("E4", 42), ("E4", 43), ("E7", 42), ("E7", 43)) \
        else (FX["e1_s42"], FX["e1_sha"])
    a = {"--stage": stage, "--seed": str(seed), "--source-run-dir": str(run_dir), "--expect-source-sha256": sha,
         "--calibration-list": str(list_path(list_seed)), "--expect-list-checksum": checksum(list_seed),
         "--expect-head": FAKE_HEAD, "--data-root": str(FX["root"]),
         "--out-dir": str(out or fresh_out(f"{stage.lower()}_s{seed}_cal{list_seed}_a1"))}
    if stage == "E7":
        a.update({"--lambda-selection": LAMBDA_REL, "--lambda-selection-sha256": FX["sel_sha"]["lambda"],
                  "--alpha-selection": ALPHA_REL, "--alpha-selection-sha256": FX["sel_sha"]["alpha"]})
    for k, v in flags.items():
        flag = "--" + k.replace("_", "-")
        if v is None:
            a.pop(flag, None)
        else:
            a[flag] = str(v)
    return ["--real-run", "--confirm-real-run"] + [x for kv in a.items() for x in kv]


def out_of(av: list[str]) -> Path:
    return Path(av[av.index("--out-dir") + 1])


# ------------------------------------------------------------------ the patched environment
def fake_identity(paths=None) -> dict:
    from src.quant.ptq import CODE_PATHS
    ps = CODE_PATHS if paths is None else paths
    return {"commit": IDENT["commit"], "clean_at_commit": IDENT["clean"],
            "files_sha256": {p: FX["file_shas"][p] for p in ps if p in FX["file_shas"]}}


def stub_core(*, stage, source_ckpt, expect_source_sha256, calibration_list, expect_list_checksum, data_root,
              out_dir, threads=None, log=print) -> dict:
    """Stands in for src.quant.ptq.run_ptq: small fake artifacts and the run-provenance fields the record, the
    evaluator's validator and the official gate read."""
    from src.quant import ptq
    from src.quant.stages import resolve_quant_stage
    STUB["calls"] += 1
    st = resolve_quant_stage(stage)
    names = ptq.artifact_names(st["key"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc, list_sha = ptq.load_calibration_list(calibration_list, expected_checksum=expect_list_checksum)
    src_sha = sha_file(source_ckpt)
    blobs = {k: f"{k}|{st['name']}|{src_sha}|{doc['checksum_sha256']}".encode("utf-8")
             for k in ("torchscript", "state_dict", "x86_copy")}
    for k, b in blobs.items():
        (out / names[k]).write_bytes(b)
    sh = {k: sha_bytes(b) for k, b in blobs.items()}
    code = ptq.code_provenance()
    if STUB["mode"] == "commit_changed":
        code = {**code, "commit": "f" * 40}
    prov = {"stage": st["name"], "source_stage": st["source_stage"], "source_checkpoint": str(source_ckpt),
            "source_checkpoint_sha256": src_sha, "random_init": False, "num_classes": 116,
            "quantization_method": "ptq", "backend": "qnnpack", "schema": ptq.RUN_META_SCHEMA,
            "qconfig": {"fingerprint": "f" * 16}, "artifact_format": "torchscript",
            "ptq_run_role": (ptq.PTQ_RUN_ROLES[doc["role"]] if doc.get("artifact_status") == "registered"
                             else "smoke"),
            "converted_artifact": names["torchscript"], "converted_artifact_sha256": sh["torchscript"],
            "state_dict_artifact": {"path": names["state_dict"], "sha256": sh["state_dict"],
                                    "bytes": len(blobs["state_dict"])},
            "x86_latency_copy": {"path": names["x86_copy"], "sha256": sh["x86_copy"], "bytes": len(blobs["x86_copy"]),
                                 "engine": "x86"},
            "engine": "qnnpack", "observers": {"activation": "HistogramObserver",
                                               "weight": "PerChannelMinMaxObserver"},
            "calibration": {"list_path": str(calibration_list), "list_file_sha256": list_sha,
                            "checksum_sha256": doc["checksum_sha256"], "seed": doc["seed"], "role": doc["role"],
                            "list_artifact_status": doc.get("artifact_status"), "batches_seen": 128,
                            "batch_size": 1, "n_calib": 128},
            "verification": {"checks": {"stub": True}, "census": {"sigmoid": {"quantized": True, "u6_values": 1}}},
            "code": code, "environment": ptq.environment(), "created_utc": "2026-10-01T00:00:00Z"}
    if st["source_stage"] == "E3":
        prov["cwd_projection_loaded"] = False
    (out / names["run_meta"]).write_bytes(ptq._json_bytes(prov))
    if STUB["mode"] == "record_exists":
        (out / f"{st['key']}_ptq_record.json").write_text("{}", encoding="utf-8")
    return prov


@contextlib.contextmanager
def bound_env(*, stub: bool = True):
    from src.quant import ptq, ptq_record
    saved = (ptq.code_provenance, ptq.run_ptq, ptq_record.RECORDS_ROOT, os.environ.get("PLANTSEG_IMAGE_DIGEST"))
    ptq.code_provenance = fake_identity
    if stub:
        ptq.run_ptq = stub_core
    ptq_record.RECORDS_ROOT = FX["records"]
    os.environ["PLANTSEG_IMAGE_DIGEST"] = FX["digest"]
    try:
        yield
    finally:
        ptq.code_provenance, ptq.run_ptq, ptq_record.RECORDS_ROOT = saved[:3]
        if saved[3] is None:
            os.environ.pop("PLANTSEG_IMAGE_DIGEST", None)
        else:
            os.environ["PLANTSEG_IMAGE_DIGEST"] = saved[3]
        IDENT.update(commit=FAKE_HEAD, clean=True)
        STUB["mode"] = "plain"


def snapshot(out: Path | None):
    if out is None or not out.exists():
        return None
    return sorted(p.name for p in out.iterdir()) if out.is_dir() else ["<file>"]


def under(path: str, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def refused(name: str, av: list[str], want: str, *, stub: bool = True, env=None, lead: str | None = None) -> None:
    """A refused command: exit 2 with its code, nothing created, the core never called, no data opened; with
    `lead`, the refusal's message starts with it (ruling R9)."""
    out = out_of(av) if "--out-dir" in av else None
    before = snapshot(out)
    STUB["calls"] = 0
    AUDIT["paths"] = []
    with bound_env(stub=stub):
        if env is not None:
            env()
        AUDIT["on"] = True
        try:
            rc, log = run_main(av)
        finally:
            AUDIT["on"] = False
    code = code_of(log)
    nothing = snapshot(out) == before
    data = [p for p in AUDIT["paths"] if under(p, FX["root"])]
    led = lead is None or f"[{want}] -- {lead}" in log
    REFUSALS.append({"name": name, "core": STUB["calls"], "nothing": nothing, "data": data})
    check(name, rc == 2 and code == want and STUB["calls"] == 0 and nothing and not data and led,
          f"exit {rc} [{code}]" + ("" if nothing else " (something was written)") + (f" data {data[:2]}" if data
                                                                                      else "")
          + ("" if led else f" (message not led by {lead!r})"))


def written(av: list[str]) -> tuple[int, str, dict]:
    """A bound run with the stub core: (exit, log, record or {})."""
    STUB["calls"] = 0
    with bound_env():
        rc, log = run_main(av)
    out, stage = out_of(av), av[av.index("--stage") + 1].lower()
    p = out / f"{stage}_ptq_record.json"
    return rc, log, (json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {})


def stub_record(key: str, av: list[str]) -> Path:
    """A stub-core record, made once per process (s7, s8, s9)."""
    if key not in FX:
        rc, log, rec = written(av)
        if rc != 0 or not rec:
            raise RuntimeError(f"stub run {key} failed: {last_result(log)}")
        FX[key] = out_of(av)
    return FX[key]


# ------------------------------------------------------------------ real-core runs (each its own process)
DRIVER = r"""
import json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
repo, records, ident_json, audit_out = sys.argv[1:5]
sys.path.insert(0, repo)
paths = []
def _audit(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
        paths.append(os.fsdecode(args[0]))
sys.addaudithook(_audit)
from src.quant import ptq, ptq_record
ptq_record.RECORDS_ROOT = Path(records)
IDENT = json.loads(Path(ident_json).read_text(encoding="utf-8"))
def _identity(paths=None):
    ps = ptq.CODE_PATHS if paths is None else paths
    return {"commit": IDENT["commit"], "clean_at_commit": True,
            "files_sha256": {p: IDENT["files"][p] for p in ps if p in IDENT["files"]}}
ptq.code_provenance = _identity
from scripts.run_ptq import main
try:
    rc = main(sys.argv[5:])
finally:
    Path(audit_out).write_text(json.dumps(paths), encoding="utf-8")
raise SystemExit(rc)
"""


def run_real(label: str, av: list[str]) -> tuple[int, str, list]:
    ident = BASE / f"identity_{label}.json"
    ident.write_text(json.dumps({"commit": FAKE_HEAD, "files": FX["file_shas"]}), encoding="utf-8")
    audit = BASE / f"audit_{label}.json"
    p = subprocess.run([sys.executable, "-B", "-c", DRIVER, str(REPO), str(FX["records"]), str(ident), str(audit),
                        *av], capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PLANTSEG_IMAGE_DIGEST": FX["digest"],
                            "PLANTSEG_DATA_ROOT": str(FX["root"])})
    paths = json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else []
    return p.returncode, p.stdout + p.stderr, paths


def eval_split_touched(paths) -> list[str]:
    bad = []
    for p in paths:
        parts = Path(os.path.realpath(p)).parts
        if any(parts[i] in ("images", "annotations") and parts[i + 1] in ("val", "test")
               for i in range(len(parts) - 1)):
            bad.append(p)
    return bad


def real_e4() -> Path:
    if "real_e4" not in FX:
        av = argv("E4", out=BASE / "real" / "e4_s42_cal42_a1")
        (BASE / "real").mkdir(exist_ok=True)
        FX["real_e4_run"] = run_real("e4_s42", av)
        FX["real_e4"] = out_of(av)
    return FX["real_e4"]


# ------------------------------------------------------------------ s0 fixtures
def case_s0() -> None:
    root = FX["root"]
    n_img = sum(1 for e in os.scandir(root / "images" / "train") if e.name.endswith(".jpg"))
    n_mask = sum(1 for e in os.scandir(root / "annotations" / "train") if e.name.endswith(".png"))
    check("fixture_train_tree_5367_pairs", n_img == n_mask == TRAIN_IMAGES, f"{n_img} images, {n_mask} masks")
    from src.quant.ptq import load_calibration_list
    docs = [load_calibration_list(list_path(s))[0] for s in (42, 43, 44, 45)]
    check("fixture_lists_registered", [d["artifact_status"] for d in docs] == ["registered"] * 4
          and [d["seed"] for d in docs] == [42, 43, 44, 45], str([d["artifact_status"] for d in docs]))
    files = git(FX["records"], "ls-files").split()
    status = git(FX["records"], "status", "--porcelain", "--untracked-files=no")
    check("fixture_records_committed", sorted(files) == sorted(FX["tracked"]) and status == "",
          f"{len(files)} tracked, status {status!r}")


# ------------------------------------------------------------------ s1 old runners
def case_s1() -> None:
    from scripts.run_e4 import main as run_e4
    from scripts.run_e7 import main as run_e7
    rows = []
    for fn, src in ((run_e4, FX["e1_s42"] / CK1), (run_e7, FX["e3_s42"] / CK3)):
        out = fresh_out("legacy_out")
        rc, log = cli(fn, ["--real-run", "--confirm-real-run", "--source-ckpt", str(src), "--out-dir", str(out),
                           "--calibration-index", str(list_path(42)), "--data-root", str(FX["root"])])
        rows.append((rc, code_of(log), out.exists()))
    check("old_runners_write_nothing", rows == [(2, "old_runner_real_run", False)] * 2, str(rows))


# ------------------------------------------------------------------ s2 E4 s42, the real core
def case_s2() -> None:
    from src.eval.stage_artifacts import int8_official_error, validate_int8_artifact
    from src.quant.ptq import CODE_PATHS
    from src.quant.ptq_record import ISOLATION_METHOD, manifest_entry, read_record, record_code_paths
    out = real_e4()
    rc, log, touched = FX["real_e4_run"]
    names = FX["names"]("e4")
    rec_p = out / "e4_ptq_record.json"
    rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.is_file() else {}
    meta = json.loads((out / names["run_meta"]).read_text(encoding="utf-8")) if (out / names["run_meta"]).is_file() \
        else {}
    check("e4_s42_exit_0", rc == 0 and last_result(log).startswith(
        "RESULT: PTQ OF RECORD WRITTEN (E4 s42, role of_record, parent e1_s42, record "), last_result(log))
    present = sorted(p.name for p in out.iterdir()) if out.is_dir() else []
    want = sorted([names["torchscript"], names["state_dict"], names["x86_copy"], names["run_meta"],
                   "e4_ptq_record.json"])
    check("e4_files_and_record_last", present == want and rec.get("ptq_run", {}).get("sha256")
          == sha_file(out / names["run_meta"]) and rec_p.stat().st_mtime_ns >= (out / names["run_meta"]).stat()
          .st_mtime_ns, str(present))
    check("e4_record_format_role", rec.get("format") == "ptq_record/1" and rec.get("role") == "of_record"
          and rec.get("descriptive") is False and rec.get("replicate_of") is None and rec.get("stage") == "E4"
          and rec.get("seed") == 42, f"{rec.get('format')} {rec.get('role')}")
    par = rec.get("parent", {})
    check("e4_record_parent_e1", par.get("rule") == "e1_run_of_record/1" and par.get("stage") == "E1"
          and par.get("run_id") == "e1_s42" and par.get("checkpoint_sha256") == FX["e1_sha"]
          and par.get("run_meta") == {"file": "e1_telemetry.jsonl",
                                      "sha256": sha_file(FX["e1_s42"] / "e1_telemetry.jsonl")}
          and par.get("best_json_sha256") == sha_file(FX["e1_s42"] / "best.json")
          and par.get("max_iters") == 80000 and par.get("poly_horizon") == 80000
          and par.get("best_val_miou_all_class") == E1_BEST, str({k: par.get(k) for k in ("rule", "run_id")}))
    cal = rec.get("calibration", {})
    check("e4_record_list", cal == {"list_path": "configs/calibration/ptq_calibration_seed42.json",
                                    "list_file_sha256": sha_file(list_path(42)), "checksum_sha256": checksum(42),
                                    "seed": 42, "role": "am10_list_of_record", "list_artifact_status": "registered"},
          str(cal)[:120])
    ok_files = False
    try:
        read_record(rec_p)
        ok_files = all(sha_file(out / v["path"]) == v["sha256"] and (out / v["path"]).stat().st_size == v["bytes"]
                       for v in rec["artifacts"].values()) and rec["artifacts"]["torchscript"]["sha256"] == \
            meta.get("converted_artifact_sha256")
    except (OSError, ValueError, KeyError) as e:
        ok_files = f"{type(e).__name__}: {e}"
    check("e4_record_artifacts_match_files", ok_files is True, str(ok_files))
    w = rec.get("writer", {})
    want_files = {p: FX["file_shas"][p] for p in record_code_paths() if p in FX["file_shas"]}
    check("e4_record_code_identity", w.get("commit") == FAKE_HEAD == meta.get("code", {}).get("commit")
          and w.get("clean_at_commit") is True and w.get("code_sha256") == want_files
          and w.get("script") == {"path": "scripts/run_ptq.py", "sha256": FX["file_shas"]["scripts/run_ptq.py"]}
          and w.get("module") == {"path": "src/quant/ptq_record.py",
                                  "sha256": FX["file_shas"]["src/quant/ptq_record.py"]}
          and set(CODE_PATHS) <= set(w.get("code_sha256", {})), str(w.get("commit")))
    env = rec.get("environment", {})
    check("e4_record_environment", env.get("image_digest") == FX["digest"] and env == {
        k: meta.get("environment", {}).get(k) for k in ("image_digest", "torch", "torchvision", "numpy", "pillow",
                                                        "torch_num_threads")}, str(env)[:120])
    iso = rec.get("isolation", {})
    check("e4_record_isolation", iso.get("method") == ISOLATION_METHOD and iso.get("data_root") == str(FX["root"])
          and iso.get("result") == {"images/test": "absent", "annotations/test": "absent",
                                    "annotation_test.json": "absent"}, str(iso)[:120])
    entry = manifest_entry(rec) if rec else {}
    check("e4_manifest_entry_am19_4c", entry.get("parent_checkpoint_sha256") == FX["e1_sha"]
          and entry.get("list_file_sha256") == sha_file(list_path(42)) and entry.get("ptq_commit") == FAKE_HEAD
          and entry.get("ptq_script") == "scripts/run_ptq.py" and entry.get("image_digest") == FX["digest"]
          and entry.get("qconfig_fingerprint") == meta.get("qconfig", {}).get("fingerprint")
          and entry.get("artifacts", {}).get("torchscript", {}).get("sha256") == meta.get("converted_artifact_sha256"),
          str(sorted(entry))[:120])
    from src.quant.ptq_record import planned_entry
    planned = planned_entry("E4", 42, FX["e1_sha"], sha_file(list_path(42)), checksum(42), FAKE_HEAD, FX["digest"])
    check("e4_planned_entry_matches_record", planned == {**entry, "status": "planned", "artifacts": None},
          str(sorted(k for k in planned if planned.get(k) != entry.get(k))))
    err = "not run"
    try:
        err = int8_official_error(validate_int8_artifact("E4", out / names["run_meta"]), repo_root=FX["records"])
    except Exception as e:                                  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    check("e4_real_record_official", err is None, str(err)[:120])
    check("e4_run_never_touched_val_or_test", bool(touched) and not eval_split_touched(touched),
          f"{len(touched)} paths audited")


def record_fields(rec: dict, out: Path, parent_dir: Path, run_id: str, list_seed: int) -> list[str]:
    """The ptq_record/1 fields that bind a record to its run (empty when every one holds)."""
    from src.quant.ptq import CODE_PATHS
    from src.quant.ptq_record import ISOLATION_METHOD, read_record, record_code_paths
    names = FX["names"](rec.get("stage", "e4").lower())
    meta = json.loads((out / names["run_meta"]).read_text(encoding="utf-8"))
    par, cal, w = rec.get("parent", {}), rec.get("calibration", {}), rec.get("writer", {})
    want_files = {p: FX["file_shas"][p] for p in record_code_paths() if p in FX["file_shas"]}
    bad = [k for k, ok in (
        ("parent run_meta", par.get("run_meta", {}).get("sha256") == sha_file(parent_dir / par.get("run_meta", {})
                                                                               .get("file", "absent"))),
        ("parent run_id", par.get("run_id") == run_id and par.get("checkpoint_sha256") == meta.get(
            "source_checkpoint_sha256")),
        ("list", cal.get("list_path") == f"configs/calibration/ptq_calibration_seed{list_seed}.json"
         and cal.get("list_file_sha256") == sha_file(list_path(list_seed)) and cal.get("checksum_sha256")
         == checksum(list_seed)),
        ("run provenance", rec.get("ptq_run", {}).get("sha256") == sha_file(out / names["run_meta"])),
        ("artifacts", all(sha_file(out / v["path"]) == v["sha256"] for v in rec.get("artifacts", {}).values())
         and len(rec.get("artifacts", {})) == 3),
        ("writer", w.get("commit") == meta.get("code", {}).get("commit") == FAKE_HEAD
         and w.get("clean_at_commit") is True and w.get("code_sha256") == want_files
         and set(CODE_PATHS) <= set(w.get("code_sha256", {}))),
        ("isolation", (rec.get("isolation") or {}).get("method") == ISOLATION_METHOD
         and (rec.get("isolation") or {}).get("data_root") == str(FX["root"])),
        ("descriptive", rec.get("descriptive") is (rec.get("role") != "of_record"))) if not ok]
    try:
        read_record(out / f"{rec.get('stage', 'e4').lower()}_ptq_record.json")
    except (OSError, ValueError) as e:
        bad.append(f"read_record: {e}")
    return bad


def safely(fn, *a) -> list[str]:
    """A field check that reports, never raises: a mutated record may lack a key the check reads."""
    try:
        return fn(*a)
    except Exception as e:                                  # noqa: BLE001
        return [f"the check raised {type(e).__name__}: {e}"]


def manifest_ok(rec: dict, out: Path) -> list[str]:
    from src.quant.ptq_record import manifest_entry
    names = FX["names"](rec["stage"].lower())
    meta = json.loads((out / names["run_meta"]).read_text(encoding="utf-8"))
    e = manifest_entry(rec)
    want = {"status": "made", "stage": rec["stage"], "seed": rec["seed"], "role": rec["role"],
            "parent_checkpoint_sha256": meta["source_checkpoint_sha256"],
            "list_file_sha256": meta["calibration"]["list_file_sha256"],
            "list_checksum_sha256": meta["calibration"]["checksum_sha256"], "ptq_commit": FAKE_HEAD,
            "ptq_script": "scripts/run_ptq.py", "image_digest": FX["digest"],
            "qconfig_fingerprint": meta["qconfig"]["fingerprint"],
            "artifacts": {"torchscript": {"path": names["torchscript"], "sha256": meta["converted_artifact_sha256"]},
                          "state_dict": {k: meta["state_dict_artifact"][k] for k in ("path", "sha256")},
                          "x86_copy": {k: meta["x86_latency_copy"][k] for k in ("path", "sha256")}}}
    return sorted(k for k in set(want) | set(e) if e.get(k) != want.get(k))


# ------------------------------------------------------------------ s3 E7 s42, the real core
def case_s3() -> None:
    out = BASE / "real" / "e7_s42_cal42_a1"
    (BASE / "real").mkdir(exist_ok=True)
    rc, log, touched = run_real("e7_s42", argv("E7", out=out))
    rec_p = out / "e7_ptq_record.json"
    rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.is_file() else {}
    par = rec.get("parent", {})
    check("e7_s42_exit_0", rc == 0 and last_result(log).startswith(
        f"RESULT: PTQ OF RECORD WRITTEN (E7 s42, role of_record, parent {E3_S42}, record ")
        and not eval_split_touched(touched), last_result(log))
    sel = par.get("selections", {})
    check("e7_record_selections", sel.get("lambda", {}).get("sha256") == FX["sel_sha"]["lambda"]
          and sel.get("alpha", {}).get("sha256") == FX["sel_sha"]["alpha"]
          and sel.get("alpha", {}).get("winner", {}).get("ckpt_sha256") == par.get("checkpoint_sha256") == FX["e3_sha"]
          and par.get("lambda_logit") == LAMBDA and par.get("alpha_cwd") == ALPHA, str(sorted(sel)))
    import scripts.preflight_distill as pd
    check("e7_record_teacher_arch", par.get("teacher_arch") == "segnext_mscan-b"
          and par.get("teacher_ckpt_sha256") == pd.TEACHER_SHA256 and par.get("parent_of_e4_e7") is True
          and par.get("rule") == "e3_run_of_record/1" and par.get("run_end", {}).get("checks_passed") is True,
          str(par.get("teacher_arch")))
    ll = par.get("launch_log", {})
    check("e7_record_launch_log", ll.get("path") == LOG_REL and ll.get("sha256") == sha_file(FX["records"] / LOG_REL)
          and ll.get("launched", {}).get("run_meta_sha256") == par.get("run_meta", {}).get("sha256")
          and ll.get("launch", {}).get("run_id") == E3_S42, str(ll.get("launched", {}).get("run_id")))


# ------------------------------------------------------------------ s4 seeds 43
def case_s4() -> None:
    rc, log, rec = written(argv("E7", seed=43))
    check("e7_s43_exit_0", rc == 0 and STUB["calls"] == 1 and rec.get("role") == "of_record"
          and last_result(log).startswith(f"RESULT: PTQ OF RECORD WRITTEN (E7 s43, role of_record, parent {E3_S43}"),
          last_result(log))
    ll = rec.get("parent", {}).get("launch_log", {})
    check("e7_s43_bound_by_launch_log", ll.get("launch", {}).get("run_id") == E3_S43
          and ll.get("launch", {}).get("sweep") is None and ll.get("launched", {}).get("run_meta_sha256")
          == sha_file(FX["e3_s43"] / "e3_run_meta.jsonl"), str(ll.get("launch", {}).get("run_id")))
    av = argv("E4", seed=43)
    rc, log, rec = written(av)
    check("e4_s43_exit_0", rc == 0 and rec.get("parent", {}).get("run_id") == "e1_s43" and rec.get("seed") == 43,
          last_result(log))
    bad = safely(record_fields, rec, out_of(av), FX["e1_s43"], "e1_s43", 42) if rec else ["no record"]
    check("e4_s43_record_fields", not bad, str(bad))
    bad = safely(manifest_ok, rec, out_of(av)) if rec else ["no record"]
    check("e4_s43_manifest_entry", not bad, str(bad))


# ------------------------------------------------------------------ s5 refusals
def variant_e1(name: str, **kw) -> tuple[Path, str]:
    return e1_run(name, parent_dir=BASE / "variants", **kw)


def variant_e3(name: str, seed: int, **kw) -> tuple[Path, str]:
    FX["n_variants"] = FX.get("n_variants", 0) + 1
    return e3_run(name, seed=seed, parent_dir=BASE / "variants3" / f"v{FX['n_variants']:03d}", **kw)


def e3_payload(**over) -> dict:
    p = {"iter": 80000, "stage": "E3", "model_state_dict": {}, "best_val_miou_all_class": E3_BEST,
         "num_classes": 116, "teacher_provenance": teacher_provenance()}
    p.update(over)
    return p


@contextlib.contextmanager
def edited(path: Path, data: bytes):
    """A records-repository file changed in the working tree for one case, its bytes restored after."""
    saved = path.read_bytes()
    path.write_bytes(data)
    try:
        yield
    finally:
        path.write_bytes(saved)


def case_s5_flags() -> None:
    refused("refuses_source_exactly_one", argv(source_ckpt=FX["e1_s42"] / CK1), "source_exactly_one")
    refused("refuses_test_path_run_dir", argv(source_run_dir=BASE / "latest" / "e1_s42"), "test_path")
    refused("refuses_test_path_out_dir", argv(out=BASE / "latest" / "e4_s42_cal42_a1", expect_source_sha256="0" * 64),
            "test_path")
    refused("refuses_binding_flags_unregistered_list",
            argv(source_run_dir=None, source_ckpt=FX["e1_s42"] / CK1, calibration_list=FX["smoke_list"]),
            "binding_needs_registered_list")
    copy = BASE / "list_copy" / "configs" / "calibration" / "ptq_calibration_seed42.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(list_path(42).read_bytes())
    refused("refuses_list_not_committed", argv(calibration_list=copy), "calibration_list_not_committed")
    p42 = list_path(42)
    with edited(p42, p42.read_bytes() + b"\n"):
        refused("refuses_list_changed_since_head", argv(), "calibration_list_not_committed")
    refused("refuses_seed_invalid", argv(seed=7), "seed_invalid")
    refused("refuses_sensitivity_list_at_seed_43", argv(seed=43, list_seed=43), "sensitivity_seed")
    refused("refuses_source_run_dir_required", argv(source_run_dir=None, source_ckpt=FX["e1_s42"] / CK1),
            "source_run_dir_required")
    refused("refuses_expect_head_format", argv(expect_head="abc"), "expect_head_format")
    refused("refuses_e4_selection_flags", argv(lambda_selection=LAMBDA_REL), "selection_not_applicable")
    av = argv()
    with bound_env():                                   # every gate would pass: only argparse can refuse (DL-24 D-1)
        rc, log = run_main(av + ["--arm", "E7-L"])
    check("refuses_arm_flag", rc == 2 and "unrecognized arguments: --arm" in log and not out_of(av).exists(),
          f"exit {rc}: {last_result(log)}")


def case_s5_out_dir() -> None:
    # a later gate (G8: a wrong source sha256) refuses these too, so a run without the out-dir refusal writes nothing
    refused("refuses_out_dir_relative", argv(out="relative_dir/e4_s42_cal42_a1", expect_source_sha256="0" * 64),
            "out_dir_relative")
    refused("refuses_out_dir_inside_repo", argv(out=REPO / "ptq_record_smoke_out" / "e4_s42_cal42_a1",
                                                expect_source_sha256="0" * 64), "output_dir_inside_repo")
    refused("refuses_out_dir_name_stage", argv(out=fresh_out("e7_s42_cal42_a1")), "out_dir_name")
    refused("refuses_out_dir_name_list", argv(out=fresh_out("e4_s42_cal43_a1")), "out_dir_name")
    refused("refuses_d3_name_without_replicate", argv(out=fresh_out("e4_s42_cal42_a1_d3")), "out_dir_name")
    refused("refuses_replicate_without_d3_name",
            argv(out=fresh_out("e4_s42_cal42_a2"), replicate_of=s8_records()["e4"] / "e4_ptq_record.json"),
            "out_dir_name")
    full = fresh_out("e4_s42_cal42_a1")
    full.mkdir()
    (full / "notes.txt").write_text("x", encoding="utf-8")
    refused("refuses_out_dir_not_empty_bound", argv(out=full), "out_dir_not_empty")
    full_u = fresh_out("e4_unbound")
    full_u.mkdir()
    (full_u / "notes.txt").write_text("x", encoding="utf-8")
    # the unbound path's core (C2): the real src.quant.ptq.run_ptq, refused before anything is loaded; a later
    # gate (the wrong source sha256) would refuse too, so the case stays fast if the out-dir refusal is removed
    refused("refuses_out_dir_not_empty_unbound",
            ["--real-run", "--confirm-real-run", "--stage", "E4", "--source-ckpt", str(FX["e1_s42"] / CK1),
             "--expect-source-sha256", "0" * 64, "--calibration-list", str(FX["smoke_list"]),
             "--expect-list-checksum", checksum(42), "--data-root", str(FX["root"]), "--out-dir", str(full_u)],
            "out_dir_not_empty", stub=False)


def case_s5_identity() -> None:
    refused("refuses_code_identity_unavailable", argv(), "code_identity_unavailable",
            env=lambda: IDENT.update(commit=None))
    refused("refuses_expect_head_mismatch", argv(expect_head="d" * 40), "expect_head_mismatch")
    refused("refuses_code_not_clean", argv(), "code_not_clean", env=lambda: IDENT.update(clean=False))
    refused("refuses_image_digest_unset", argv(), "image_digest_not_of_record",
            env=lambda: os.environ.pop("PLANTSEG_IMAGE_DIGEST", None))
    refused("refuses_image_digest_not_of_record", argv(), "image_digest_not_of_record",
            env=lambda: os.environ.__setitem__("PLANTSEG_IMAGE_DIGEST", "sha256:" + "0" * 64))
    # PL-4 (SL-1): no TEST folder is created; os.path.lexists answers True for the data root's first TEST surface
    from src.data.isolation import TEST_SURFACES
    surface = os.path.join(str(FX["root"]), *TEST_SURFACES[0].split("/"))
    real_lexists = os.path.lexists

    def lexists(path, _real=real_lexists):
        return True if os.fspath(path) == surface else _real(path)
    os.path.lexists = lexists
    try:
        refused("refuses_data_root_with_test_surface", argv(), "data_root_has_test")
    finally:
        os.path.lexists = real_lexists


def case_s5_selections() -> None:
    refused("refuses_e7_without_lambda_selection", argv("E7", lambda_selection=None), "lambda_selection_missing")
    refused("refuses_selection_sha256_mismatch", argv("E7", alpha_selection_sha256="0" * 64),
            "selection_sha256_mismatch", lead="E7 parent rule (e3_run_of_record/1): ")
    a = FX["records"] / ALPHA_REL
    with edited(a, a.read_bytes() + b" "):
        refused("refuses_selection_changed_since_head", argv("E7"), "selection_changed_since_head")
    refused("refuses_selection_not_tracked",
            argv("E7", alpha_selection=UNTRACKED_SEL_REL, alpha_selection_sha256=FX["sel_sha"]["untracked"]),
            "selection_not_tracked")
    refused("refuses_selection_format",
            argv("E7", alpha_selection=OTHER_SEL_REL, alpha_selection_sha256=FX["sel_sha"]["other"]),
            "selection_format")


def case_s5_parent() -> None:
    refused("refuses_parent_dir_missing", argv(source_run_dir=BASE / "absent_run"), "parent_dir_missing")
    refused("refuses_parent_sha256_mismatch", argv(expect_source_sha256="0" * 64), "parent_sha256_mismatch",
            lead="E4 parent rule (e1_run_of_record/1): ")
    d, s = variant_e1("e1_best_json_list", best_json="[]")
    refused("refuses_parent_best_json_format", argv(source_run_dir=d, expect_source_sha256=s),
            "parent_best_json_format")
    d, s = variant_e1("e1_two_rows", rows=2)
    refused("refuses_parent_two_run_meta_rows", argv(source_run_dir=d, expect_source_sha256=s),
            "parent_run_meta_rows")
    d, s = variant_e1("e1_mode_smoke", mode="smoke")
    refused("refuses_parent_mode_dry", argv(source_run_dir=d, expect_source_sha256=s), "parent_mode_not_real")
    d, s = variant_e1("e1_seed43_meta", seed=43)
    refused("refuses_parent_seed_mismatch", argv(source_run_dir=d, expect_source_sha256=s), "parent_seed_mismatch")


def case_s5_e1_rule() -> None:
    d, s = variant_e1("e1_160k", max_iters=160000, poly_horizon=160000)
    refused("refuses_e4_parent_160k", argv(source_run_dir=d, expect_source_sha256=s), "parent_horizon")
    d, s = variant_e1("e1_horizon_160k", poly_horizon=160000)
    refused("refuses_e4_parent_poly_horizon_160k", argv(source_run_dir=d, expect_source_sha256=s), "parent_horizon")
    d, s = variant_e1("e1_160k_no_horizon", max_iters=160000, poly_horizon=None)     # max_iters alone (DL-24 D-2)
    refused("refuses_e4_parent_max_iters_160k", argv(source_run_dir=d, expect_source_sha256=s), "parent_horizon")
    d, s = variant_e1("e1_no_horizon", poly_horizon=None)
    rc, log, rec = written(argv(source_run_dir=d, expect_source_sha256=s))
    check("accepts_e4_parent_without_poly_horizon", rc == 0 and rec.get("parent", {}).get("poly_horizon") is None
          and rec.get("parent", {}).get("max_iters") == 80000, last_result(log))
    d, s = variant_e1("e1_no_final_val", val_iter=78000)
    refused("refuses_e4_parent_without_final_val", argv(source_run_dir=d, expect_source_sha256=s),
            "parent_incomplete")
    d, s = variant_e1("e1_payload_best", payload={"iter": 80000, "model_state_dict": {},
                                                   "best_val_miou_all_class": 0.25, "num_classes": 116})
    refused("refuses_e4_parent_payload_best_mismatch", argv(source_run_dir=d, expect_source_sha256=s),
            "parent_payload")
    d, s = variant_e1("e1_payload_e3", payload={"iter": 80000, "stage": "E3", "model_state_dict": {},
                                                 "best_val_miou_all_class": E1_BEST, "num_classes": 116})
    refused("refuses_e4_parent_payload_stage_e3", argv(source_run_dir=d, expect_source_sha256=s), "parent_payload")


def e7_variant(name: str, seed: int = 43, **kw) -> list[str]:
    d, s = variant_e3(name, seed, **kw)
    return argv("E7", seed=seed, source_run_dir=d, expect_source_sha256=s)


def case_s5_e3_rule() -> None:
    refused("refuses_e7_parent_stage_e2", e7_variant(E3_S43, meta=e3_meta(43, stage="E2")), "parent_stage_mismatch")
    refused("refuses_e7_parent_run_abort", e7_variant(E3_S43, abort=True), "parent_incomplete")
    refused("refuses_e7_parent_without_run_end", e7_variant(E3_S43, run_end=False), "parent_incomplete")
    refused("refuses_e7_parent_checks_failed", e7_variant(E3_S43, checks=False), "parent_incomplete")
    refused("refuses_e7_parent_lambda_mismatch", e7_variant(E3_S43, meta=e3_meta(43, lambda_logit=1.0)),
            "parent_lambda_mismatch")
    refused("refuses_e7_parent_alpha_mismatch", e7_variant(E3_S43, meta=e3_meta(43, alpha_cwd=10.0)),
            "parent_alpha_mismatch")
    refused("refuses_e7_s42_non_winner_run", e7_variant("e3_s42_lambda0.5_alpha25_80k_a2", seed=42),
            "parent_run_id_mismatch")
    refused("refuses_e7_s42_checkpoint_not_selected", e7_variant(E3_S42, seed=42), "parent_checkpoint_not_selected")
    refused("refuses_e7_parent_arm_run",
            e7_variant(E3_S43, meta=e3_meta(43, arm="E3-R1", descriptive=True, parent_of_e4_e7=False,
                                            target_construction={"K": 2, "views": 2})), "parent_not_of_record")
    refused("refuses_e7_parent_160k",
            e7_variant(E3_S43, meta=e3_meta(43, max_iters=160000, poly_horizon=160000, descriptive=True,
                                            parent_of_e4_e7=False)), "parent_not_of_record")
    tp_r1 = teacher_provenance(ckpt_sha256="4c54aa1b06e787d93b04ef5469398bb19e34f4660166124d37cff69273eb9f20")
    refused("refuses_e7_parent_teacher_r1", e7_variant(E3_S43, meta=e3_meta(43, teacher_provenance=tp_r1)),
            "parent_teacher_not_of_record")
    refused("refuses_e7_parent_teacher_mock", e7_variant(E3_S43, meta=e3_meta(43, teacher_mock=True)),
            "parent_teacher_not_of_record")
    # DL-24 D-6: each condition of parent_not_of_record and parent_teacher_not_of_record, alone
    for key, over in (("arm", {"arm": "E3-R1"}), ("descriptive", {"descriptive": True}),
                      ("parent_of_e4_e7", {"parent_of_e4_e7": False}), ("max_iters", {"max_iters": 160000}),
                      ("poly_horizon", {"poly_horizon": 160000}),
                      ("target_construction", {"target_construction": {"K": 2, "views": 2}})):
        refused(f"refuses_e7_parent_not_of_record_{key}", e7_variant(E3_S43, meta=e3_meta(43, **over)),
                "parent_not_of_record")
    for key, tp in (("builder", teacher_provenance(builder="another_builder")),
                    ("expected_sha256", teacher_provenance(expected_sha256="0" * 64)), ("not_an_object", None)):
        refused(f"refuses_e7_parent_teacher_{key}", e7_variant(E3_S43, meta=e3_meta(43, teacher_provenance=tp)),
                "parent_teacher_not_of_record")
    refused("refuses_e7_s43_not_launched", e7_variant(E3_S43_A2), "parent_launch_log")
    refused("refuses_e7_s43_run_meta_not_launched", e7_variant(E3_S43, meta=e3_meta(43, note="another run_meta")),
            "parent_launch_log")
    refused("refuses_e7_s43_launch_line_fields", e7_variant(E3_S43_A3), "parent_launch_log")
    refused("refuses_e7_s43_stopped_line", e7_variant(E3_S43_A4), "parent_launch_log")
    for f in LOG_FIELD_LINES:                           # DL-24 D-3: each field of R8's set, alone
        refused(f"refuses_e7_s43_launch_field_{f}", e7_variant(E3_S43_FIELD[f]), "parent_launch_log")
    log = FX["records"] / LOG_REL
    with edited(log, log.read_bytes() + log.read_bytes().splitlines(keepends=True)[0]):
        refused("refuses_e7_launch_log_uncommitted", argv("E7", seed=43), "parent_launch_log")
    refused("refuses_e7_payload_arm_key", e7_variant(E3_S43, payload=e3_payload(arm="E3-R1")), "parent_payload")
    refused("refuses_e7_payload_teacher_mismatch",
            e7_variant(E3_S43, payload=e3_payload(teacher_provenance=teacher_provenance(config_path="other.py"))),
            "parent_payload")
    refused("refuses_e7_payload_best_mismatch", e7_variant(E3_S43, payload=e3_payload(best_val_miou_all_class=0.3)),
            "parent_payload")
    refused("refuses_e7_payload_target_construction_key",                   # DL-24 D-6
            e7_variant(E3_S43, payload=e3_payload(target_construction={"K": 2, "views": 2})), "parent_payload")


def case_s5_unreadable() -> None:
    """DL-24 A-1: an input a gate cannot read refuses (bound_input_unreadable, exit 2), never a traceback."""
    d, s = variant_e3(E3_S43, 43)
    tel = d / "e3_telemetry.jsonl"
    rows = tel.read_text(encoding="utf-8").splitlines(keepends=True)
    tel.write_text("".join(rows[:-1]) + '{"event": "train", "iter": 2\n' + rows[-1], encoding="utf-8")
    refused("refuses_e7_parent_torn_telemetry_line", argv("E7", seed=43, source_run_dir=d, expect_source_sha256=s),
            "bound_input_unreadable")
    d, _s = variant_e1("e1_ckpt_unloadable")
    (d / CK1).write_bytes(b"not a checkpoint\n")
    refused("refuses_e4_parent_checkpoint_unloadable",
            argv(source_run_dir=d, expect_source_sha256=sha_file(d / CK1)), "bound_input_unreadable")
    empty = BASE / "path_without_git"                   # G2 runs git first: a PATH without it
    empty.mkdir(exist_ok=True)
    saved = os.environ.get("PATH")
    try:
        refused("refuses_without_git_binary", argv(), "bound_input_unreadable",
                env=lambda: os.environ.__setitem__("PATH", str(empty)))
    finally:
        if saved is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = saved


def case_s5_replicate() -> None:
    recs = s8_records()
    refused("refuses_replicate_not_a_record",
            argv(out=fresh_out("e4_s42_cal42_a1_d3"), replicate_of=recs["e4"] / "e4_ptq_run_meta.json"),
            "replicate_mismatch")
    refused("refuses_replicate_other_list",
            argv(out=fresh_out("e4_s42_cal42_a1_d3"), replicate_of=recs["e4_cal43"] / "e4_ptq_record.json"),
            "replicate_mismatch")
    refused("refuses_replicate_other_parent",
            argv(out=fresh_out("e4_s42_cal42_a1_d3"), replicate_of=recs["e4_alt"] / "e4_ptq_record.json"),
            "replicate_mismatch")
    refused("refuses_replicate_of_replicate",
            argv(out=fresh_out("e4_s42_cal42_a1_d3"), replicate_of=recs["e4_d3"] / "e4_ptq_record.json"),
            "replicate_mismatch")


def case_s5_stops() -> None:
    for name, mode, code in (("stop_identity_changed", "commit_changed", "identity_changed"),
                             ("stop_record_write_failed", "record_exists", "record_write_failed")):
        av = argv()
        STUB["calls"] = 0
        with bound_env():
            STUB["mode"] = mode
            rc, log = run_main(av)
        rec = out_of(av) / "e4_ptq_record.json"
        kept = rec.is_file() and rec.read_text(encoding="utf-8") == "{}" if mode == "record_exists" else not rec.exists()
        check(name, rc == 1 and code_of(log) == code and STUB["calls"] == 1 and kept, f"exit {rc} [{code_of(log)}]")


def case_s5_order() -> None:
    check("refused_runs_wrote_nothing", REFUSALS and all(r["nothing"] for r in REFUSALS),
          str([r["name"] for r in REFUSALS if not r["nothing"]][:3]))
    check("refused_runs_never_reached_the_core", REFUSALS and all(r["core"] == 0 for r in REFUSALS),
          str([r["name"] for r in REFUSALS if r["core"]][:3]))
    check("no_data_opened_before_gates", REFUSALS and not any(r["data"] for r in REFUSALS),
          f"{len(REFUSALS)} refused runs audited")


# ------------------------------------------------------------------ s6 d3, the real core
def case_s6() -> None:
    orig = real_e4()
    out = BASE / "real" / "e4_s42_cal42_a1_d3"
    rc, log, _ = run_real("e4_s42_d3", argv("E4", out=out, replicate_of=orig / "e4_ptq_record.json"))
    rec = json.loads((out / "e4_ptq_record.json").read_text(encoding="utf-8")) if (out / "e4_ptq_record.json") \
        .is_file() else {}
    check("e4_d3_replicate_role", rc == 0 and rec.get("role") == "d3_replicate" and rec.get("descriptive") is True
          and rec.get("replicate_of", {}).get("sha256") == sha_file(orig / "e4_ptq_record.json")
          and last_result(log).startswith("RESULT: PTQ RECORD WRITTEN (E4 s42, role d3_replicate"), last_result(log))
    names = FX["names"]("e4")
    rc, log = run_main(["--compare", str(orig / names["run_meta"]), str(out / names["run_meta"])])
    check("e4_d3_compare_pass", rc == 0 and "RESULT: PTQ d3 PASS (bitwise identical)" in log, last_result(log))


# ------------------------------------------------------------------ s7 sensitivity
def case_s7() -> None:
    rc, log, rec = written(argv("E4", list_seed=43))
    check("e4_s42_cal43_role", rc == 0 and rec.get("role") == "calibration_sensitivity"
          and rec.get("descriptive") is True and rec.get("calibration", {}).get("seed") == 43
          and last_result(log).startswith("RESULT: PTQ RECORD WRITTEN (E4 s42, role calibration_sensitivity"),
          last_result(log))
    rc, log, rec = written(argv("E7", list_seed=44))
    check("e7_s42_cal44_role", rc == 0 and rec.get("role") == "calibration_sensitivity"
          and rec.get("calibration", {}).get("role") == "am16_calibration_sensitivity", last_result(log))
    orig = s8_records()["e4"] / "e4_ptq_record.json"
    d3 = s8_records()["e4_d3"]
    rec = json.loads((d3 / "e4_ptq_record.json").read_text(encoding="utf-8"))
    check("e4_d3_stub_replicate_role", rec.get("role") == "d3_replicate" and rec.get("descriptive") is True
          and rec.get("replicate_of") == {"path": str(orig.resolve()), "sha256": sha_file(orig)}
          and not safely(record_fields, rec, d3, FX["e1_s42"], "e1_s42", 42), str(rec.get("replicate_of"))[:120])


# ------------------------------------------------------------------ s8 official gate (stub records)
def s8_records() -> dict:
    if "s8" not in FX:
        root = BASE / "s8"
        recs = {"e4": stub_record("s8_e4", argv("E4", out=root / "e4" / "e4_s42_cal42_a1")),
                "e4_cal43": stub_record("s8_e4_cal43", argv("E4", list_seed=43, out=root / "c43" / "e4_s42_cal43_a1")),
                "e7": stub_record("s8_e7", argv("E7", out=root / "e7" / "e7_s42_cal42_a1"))}
        recs["e4_d3"] = stub_record("s8_e4_d3", argv("E4", out=root / "e4" / "e4_s42_cal42_a1_d3",
                                                     replicate_of=recs["e4"] / "e4_ptq_record.json"))
        alt, alt_sha = e1_run("e1_s42_alt", parent_dir=BASE / "alt_parent", best=E1_BEST,
                              payload={"iter": 80000, "model_state_dict": {"x": 1}, "best_val_miou_all_class": E1_BEST,
                                       "num_classes": 116})
        recs["e4_alt"] = stub_record("s8_e4_alt", argv("E4", source_run_dir=alt, expect_source_sha256=alt_sha,
                                                       out=root / "alt" / "e4_s42_cal42_a1"))
        FX["s8"] = recs
    return FX["s8"]


def official(stage: str, run_meta: Path, repo_root=None):
    from src.eval.stage_artifacts import int8_official_error, validate_int8_artifact
    return int8_official_error(validate_int8_artifact(stage, run_meta), repo_root=repo_root or FX["records"])


def copy_run(src: Path, name: str, *, without=()) -> Path:
    dst = BASE / "s8_copies" / f"{name}"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*without) if without else None)
    return dst


def case_s8_gate() -> None:
    recs = s8_records()
    meta4 = "e4_ptq_run_meta.json"
    check("official_accepts_the_record_of_record", official("E4", recs["e4"] / meta4) is None,
          str(official("E4", recs["e4"] / meta4))[:120])
    legacy = copy_run(recs["e4"], "legacy")
    m = json.loads((legacy / meta4).read_text(encoding="utf-8"))
    m.pop("artifact_format")
    m.update(converted_artifact=m["state_dict_artifact"]["path"],
             converted_artifact_sha256=m["state_dict_artifact"]["sha256"])
    (legacy / meta4).write_text(json.dumps(m), encoding="utf-8")
    err = official("E4", legacy / meta4)
    check("official_refuses_legacy_state_dict_ptq", "[legacy_ptq_artifact]" in str(err), str(err)[:120])
    missing = copy_run(recs["e4"], "missing", without=("e4_ptq_record.json",))
    err = official("E4", missing / meta4)
    check("official_refuses_missing_record", "[ptq_record_missing]" in str(err), str(err)[:120])
    arm = copy_run(recs["e7"], "e7_s42_a1", without=("e7_ptq_record.json",))
    (arm / "e7_arm_record.json").write_text(json.dumps({"format": "e7_arm_record/1"}), encoding="utf-8")
    err = official("E7", arm / "e7_ptq_run_meta.json")
    check("official_refuses_arm_directory", "[ptq_record_missing]" in str(err), str(err)[:120])
    bad = copy_run(recs["e4"], "malformed")
    (bad / "e4_ptq_record.json").write_text("{", encoding="utf-8")
    err = official("E4", bad / meta4)
    check("official_refuses_malformed_record", "[ptq_record_invalid]" in str(err), str(err)[:120])
    err = official("E4", recs["e4_d3"] / meta4)
    check("official_refuses_replicate_record", "[ptq_record_role]" in str(err), str(err)[:120])
    tampered = copy_run(recs["e4"], "tampered_x86")
    (tampered / "e4_int8_x86_latency.torchscript.pt").write_bytes(b"another x86 copy")
    err = official("E4", tampered / meta4)
    check("official_refuses_tampered_artifact", "[ptq_record_invalid]" in str(err), str(err)[:120])
    other = copy_run(recs["e4"], "another_run")
    rec = json.loads((other / "e4_ptq_record.json").read_text(encoding="utf-8"))
    rec["parent"]["checkpoint_sha256"] = "0" * 64
    (other / "e4_ptq_record.json").write_text(json.dumps(rec), encoding="utf-8")
    err = official("E4", other / meta4)
    check("official_refuses_record_of_another_run", "[ptq_record_mismatch]" in str(err), str(err)[:120])
    # DL-24 D-6: the TorchScript and list-checksum comparisons, each alone (the record re-hashes, so it names
    # another existing file)
    named = copy_run(recs["e4"], "record_names_another_file")
    (named / "e4_other_artifact.pt").write_bytes(b"another TorchScript file")
    rec = json.loads((named / "e4_ptq_record.json").read_text(encoding="utf-8"))
    other_ts = named / "e4_other_artifact.pt"
    rec["artifacts"]["torchscript"] = {"path": other_ts.name, "sha256": sha_file(other_ts),
                                       "bytes": other_ts.stat().st_size}
    (named / "e4_ptq_record.json").write_text(json.dumps(rec), encoding="utf-8")
    err = official("E4", named / meta4)
    check("official_refuses_record_of_another_torchscript", "[ptq_record_mismatch]" in str(err)
          and "['torchscript'] differ" in str(err), str(err)[:120])
    listed = copy_run(recs["e4"], "record_of_another_list")
    rec = json.loads((listed / "e4_ptq_record.json").read_text(encoding="utf-8"))
    rec["calibration"]["checksum_sha256"] = "0" * 64
    (listed / "e4_ptq_record.json").write_text(json.dumps(rec), encoding="utf-8")
    err = official("E4", listed / meta4)
    check("official_refuses_record_of_another_list", "[ptq_record_mismatch]" in str(err)
          and "['list checksum'] differ" in str(err), str(err)[:120])
    alt = copy_run(recs["e4"], "another_provenance")
    m = json.loads((alt / meta4).read_text(encoding="utf-8"))
    m["created_utc"] = "2026-10-02T00:00:00Z"
    (alt / "e4_alt_run_meta.json").write_text(json.dumps(m), encoding="utf-8")
    err = official("E4", alt / "e4_alt_run_meta.json")
    check("official_refuses_another_provenance", "[ptq_record_mismatch]" in str(err), str(err)[:120])
    err = official("E4", recs["e4_cal43"] / meta4)
    check("official_refuses_sensitivity_record", err is not None and "seed 43" in str(err), str(err)[:120])
    first = copy_run(recs["e4_cal43"], "cal43_without_record", without=("e4_ptq_record.json",))
    err = official("E4", first / meta4)
    check("official_calibration_gate_runs_first", err is not None and "seed 43" in str(err)
          and "[ptq_record" not in str(err), str(err)[:120])
    from src.eval.stage_artifacts import int8_official_error, resolve_stage_artifact
    qat = {"spec": resolve_stage_artifact("E5"), "provenance": {"stage": "E5"}, "artifact_path": BASE / "absent.pt",
           "artifact_sha256": "0" * 64, "artifact_format": "state_dict"}
    check("official_record_gate_passes_qat", int8_official_error(qat) is None, str(int8_official_error(qat))[:120])


def case_s8_e2e() -> None:
    """End to end through scripts/evaluate_model.py, refused before any dataset: a run provenance that claims the
    repository's committed list of record (so the calibration gate passes) in a directory without a PTQ record."""
    recs = s8_records()
    meta4 = "e4_ptq_run_meta.json"
    real_list = json.loads((REPO / "configs" / "calibration" / "ptq_calibration_seed42.json").read_text(encoding="utf-8"))
    e2e = copy_run(recs["e4"], "e2e_without_record", without=("e4_ptq_record.json",))
    m = json.loads((e2e / meta4).read_text(encoding="utf-8"))
    m["calibration"].update(checksum_sha256=real_list["checksum_sha256"], seed=42, role="am10_list_of_record",
                            list_artifact_status="registered")
    (e2e / meta4).write_text(json.dumps(m), encoding="utf-8")
    from scripts.evaluate_model import CliError, Counters
    from scripts.evaluate_model import build_parser as eval_parser, run as eval_run
    out = BASE / "s8_eval_out"
    args = eval_parser().parse_args(["--stage", "E4", "--precision", "int8_ptq", "--provenance", str(e2e / meta4),
                                     "--split", "test", "--confirm-test-split", "--artifact-status", "official",
                                     "--batch-size", "1", "--device", "cpu", "--out-dir", str(out)])
    counters = Counters(dataset=[], model=[])
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            eval_run(args, counters=counters)
        check("official_record_gate_before_any_dataset", False, "accepted")
    except CliError as e:
        check("official_record_gate_before_any_dataset", "[ptq_record_missing]" in str(e) and counters.dataset == []
              and counters.model == [] and not out.exists(), str(e)[:120])


def case_s8_source() -> None:
    tree = ast.parse((REPO / "scripts" / "evaluate_model.py").read_text(encoding="utf-8"))
    used = [n.id if isinstance(n, ast.Name) else n.attr for n in ast.walk(tree)
            if isinstance(n, (ast.Name, ast.Attribute))]
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    text = (REPO / "scripts" / "evaluate_model.py").read_text(encoding="utf-8")
    check("evaluate_model_calls_only_the_wrapper", "int8_official_error" in used and "int8_official_error" in imported
          and "int8_official_calibration_error" not in used + imported
          and "official_error = int8_official_error(resolved, repo_root=REPO)\n            if official_error:\n"
              "                raise CliError(official_error)" in text,
          f"wrapper {used.count('int8_official_error')}x, calibration gate {used.count('int8_official_calibration_error')}x")


# ------------------------------------------------------------------ s9 the calibration range (stub records)
def s9_records() -> dict:
    if "s9" not in FX:
        root = BASE / "s9"
        recs = {s: stub_record(f"s9_e4_cal{s}", argv("E4", list_seed=s, out=root / f"r{s}" / f"e4_s42_cal{s}_a1"))
                for s in (42, 43, 44, 45)}
        recs["d3"] = stub_record("s9_e4_d3", argv("E4", out=root / "r42" / "e4_s42_cal42_a1_d3",
                                                  replicate_of=recs[42] / "e4_ptq_record.json"))
        alt, alt_sha = e1_run("e1_s42_alt9", parent_dir=BASE / "alt_parent9", best=E1_BEST,
                              payload={"iter": 80000, "model_state_dict": {"y": 2}, "best_val_miou_all_class": E1_BEST,
                                       "num_classes": 116})
        recs["alt43"] = stub_record("s9_e4_alt43", argv("E4", list_seed=43, source_run_dir=alt,
                                                        expect_source_sha256=alt_sha,
                                                        out=root / "alt" / "e4_s42_cal43_a1"))
        FX["s9"] = recs
    return FX["s9"]


def summary(rec_dir: Path, seed: int, name: str, **over) -> Path:
    rec = json.loads((rec_dir / "e4_ptq_record.json").read_text(encoding="utf-8"))
    doc = {"run": {"artifact_status": "official" if seed == 42 else "provisional", "stage": "E4",
                   "checkpoint_sha256": rec["artifacts"]["torchscript"]["sha256"]},
           "dataset": {"split": "test", "condition": dict(CLEAN), "preprocess_protocol": "core_preprocess/1.0.0",
                       "expected_rows": 1561, "actual_rows": 1561},
           "dataset_level": {"all_class_miou": 0.30 + 0.01 * (seed - 42), "disease_only_miou": 0.20 + 0.02 * (seed - 42)}}
    for k, v in over.items():
        part, key = k.split("__")
        if v is None:
            doc[part].pop(key, None)
        else:
            doc[part][key] = v
    p = BASE / "s9" / "summaries" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def range_cmd(pairs, out: Path) -> tuple[int, str]:
    from scripts.ptq_calibration_range import main
    av = ["--stage", "E4"]
    for r, s in pairs:
        av += ["--pair", str(r), str(s)]
    return cli(main, av + ["--out-dir", str(out)])


def good_pairs() -> list:
    if "s9_pairs" not in FX:
        recs = s9_records()
        FX["s9_pairs"] = [(recs[s] / "e4_ptq_record.json", summary(recs[s], s, f"good_{s}")) for s in (42, 43, 44, 45)]
    return FX["s9_pairs"]


def range_refused(name: str, pairs, want: str, out: Path | None = None) -> None:
    out = out or BASE / "s9" / "outs" / name
    rc, log = range_cmd(pairs, out)
    check(name, rc == 2 and code_of(log) == want and not (out / "ptq_calibration_range_E4.json").exists(),
          f"exit {rc} [{code_of(log)}]")


def case_s9() -> None:
    recs = s9_records()
    out = BASE / "s9" / "outs" / "written"
    pairs = good_pairs()
    rc, log = range_cmd(pairs, out)
    p = out / "ptq_calibration_range_E4.json"
    doc = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    vals = [0.30 + 0.01 * (s - 42) for s in (42, 43, 44, 45)]
    dis = [0.20 + 0.02 * (s - 42) for s in (42, 43, 44, 45)]
    per = doc.get("per_subset", [])
    check("range_written", rc == 0 and doc.get("format") == "ptq_calibration_range/1" and doc.get("descriptive") is True
          and doc.get("metric") == "dataset_level.all_class_miou" and [x["list_seed"] for x in per] == [42, 43, 44, 45]
          and abs(doc["all_class_miou"]["range"] - (max(vals) - min(vals))) < 1e-12
          and abs(doc["disease_only_miou"]["range"] - (max(dis) - min(dis))) < 1e-12
          and [x["summary_sha256"] for x in per] == [sha_file(s) for _, s in pairs]
          and [x["record_sha256"] for x in per] == [sha_file(r) for r, _ in pairs]
          and [x["artifact_status"] for x in per] == ["official"] + ["provisional"] * 3, last_result(log))
    range_refused("range_refuses_input_invalid",
                  pairs[:3] + [(pairs[3][0], summary(recs[45], 45, "no_split", dataset__split=None))],
                  "range_input_invalid")
    # DL-24 B-1: a field of the wrong type, or a record field read_record does not check, refuses too
    null_dl = summary(recs[45], 45, "dataset_level_null")
    doc_null = json.loads(null_dl.read_text(encoding="utf-8"))
    doc_null["dataset_level"] = None
    null_dl.write_text(json.dumps(doc_null), encoding="utf-8")
    range_refused("range_refuses_dataset_level_null", pairs[:3] + [(pairs[3][0], null_dl)], "range_input_invalid")
    range_refused("range_refuses_rows_not_integer",
                  pairs[:3] + [(pairs[3][0], summary(recs[45], 45, "rows_list", dataset__expected_rows=[1561]))],
                  "range_input_invalid")
    no_seed = copy_run(recs[45], "s9_record_without_list_seed")
    rec45 = json.loads((no_seed / "e4_ptq_record.json").read_text(encoding="utf-8"))
    rec45["calibration"].pop("seed")
    (no_seed / "e4_ptq_record.json").write_text(json.dumps(rec45), encoding="utf-8")
    range_refused("range_refuses_record_without_list_seed", pairs[:3] + [(no_seed / "e4_ptq_record.json", pairs[3][1])],
                  "range_input_invalid")
    range_refused("range_refuses_stage", pairs[:3] + [(pairs[3][0], summary(recs[45], 45, "stage_e7", run__stage="E7"))],
                  "range_stage")
    range_refused("range_refuses_duplicate_list", [pairs[0], pairs[0], pairs[1], pairs[2]], "range_lists")
    range_refused("range_refuses_missing_list", pairs[:3], "range_lists")
    range_refused("range_refuses_replicate_as_list_of_record",
                  [(recs["d3"] / "e4_ptq_record.json", summary(recs["d3"], 42, "d3_42"))] + pairs[1:], "range_lists")
    range_refused("range_refuses_two_parents",
                  [pairs[0], (recs["alt43"] / "e4_ptq_record.json", summary(recs["alt43"], 43, "alt_43"))] + pairs[2:],
                  "range_parent")
    other_sha = json.loads((recs[44] / "e4_ptq_record.json").read_text(encoding="utf-8"))["artifacts"]["torchscript"]["sha256"]
    range_refused("range_refuses_summary_of_another_artifact",
                  [pairs[0], (pairs[1][0], summary(recs[43], 43, "other_art", run__checkpoint_sha256=other_sha))]
                  + pairs[2:], "range_artifact")
    range_refused("range_refuses_status",
                  [pairs[0], (pairs[1][0], summary(recs[43], 43, "status_off", run__artifact_status="official"))]
                  + pairs[2:], "range_status")
    range_refused("range_refuses_mixed_split",
                  pairs[:2] + [(pairs[2][0], summary(recs[44], 44, "split_val", dataset__split="val"))] + pairs[3:],
                  "range_protocol")
    range_refused("range_refuses_rows",
                  pairs[:3] + [(pairs[3][0], summary(recs[45], 45, "rows", dataset__actual_rows=1560))], "range_rows")
    range_refused("range_refuses_metric",
                  pairs[:3] + [(pairs[3][0], summary(recs[45], 45, "metric", dataset_level__all_class_miou=None))],
                  "range_metric")
    range_refused("range_refuses_out_dir_inside_repo", pairs[:3], "output_dir_inside_repo",
                  out=REPO / "ptq_range_smoke_out")
    kept = p.read_bytes() if p.is_file() else None
    rc2, log2 = range_cmd(pairs, out)
    check("range_refuses_existing_output", rc2 == 2 and code_of(log2) == "range_output_exists" and kept is not None
          and p.read_bytes() == kept, f"exit {rc2} [{code_of(log2)}]")


# ------------------------------------------------------------------ s11 E7-E2 (AM-23 item 2; L-AM23Q B-2)
E2_S42, CK2, E2_BEST = "e2_s42_lambda0.5_80k_a1", "e2_student_best_iter64000.pt", 0.38
E2_TERMS = {"logit_kd": True, "cwd_feat": False, "cwd_logit": False}


def e2_meta(**over) -> dict:
    """E2 seed 42's run_meta row (8487551 train_distill.py :963-1003, launch_fields :678-699)."""
    meta = {"event": "run_meta", "mode": "real", "seed": 42, "stage": "E2", "terms": dict(E2_TERMS),
            "max_iters": 80000, "poly_horizon": 80000, "lambda_logit": LAMBDA, "wall_clock": WALL[42],
            "git_head": GIT_HEAD, "teacher_provenance": teacher_provenance(), "teacher_mock": False, "arm": None,
            "descriptive": False, "parent_of_e4_e7": False, "records_commit": RECORDS_COMMIT}
    meta.update(over)
    return meta


def e2_payload(**over) -> dict:
    p = {"iter": 64000, "stage": "E2", "model_state_dict": {}, "best_val_miou_all_class": E2_BEST, "num_classes": 116,
         "teacher_provenance": teacher_provenance(), "terms": dict(E2_TERMS), "lambda_logit": LAMBDA}
    p.update(over)
    return p


def e2_ckpt_real() -> Path:
    """A random-weight student checkpoint in E2's schema (load_student_from_e2's input), built once."""
    import torch
    parent = FX["cache"] if FX["cache"] is not None else BASE
    p = parent / "e2_student_v1.pt"
    marker = parent / "e2_student_v1.done"
    if marker.is_file() and p.is_file():
        return p
    tmp = p.with_name(p.name + f".{os.getpid()}.part")
    torch.save(e2_payload(model_state_dict=student_state(13)), tmp)
    os.replace(tmp, p)
    marker.write_text("1", encoding="utf-8")
    return p


def e2_run(variant: str, *, meta=None, payload=None, real=False, files=None) -> tuple[Path, str]:
    """An E2 run directory named E2_S42 (the λ winner's run_id) under its own parent folder."""
    import torch
    d = BASE / "s11" / "runs" / variant / E2_S42
    d.mkdir(parents=True)
    (d / "e2_run_meta.jsonl").write_text(json.dumps(meta if meta is not None else e2_meta()) + "\n",
                                         encoding="utf-8")
    tel = [{"event": "train", "iter": 1, "loss": 1.0},
           {"event": "run_end", "iter": 80000, "best_val_miou_all_class": E2_BEST, "best_ckpt": CK2,
            "checks_passed": True, "wall_clock_start": WALL[42], "wall_clock_end": WALL[42] + 3600.0}]
    (d / "e2_telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tel), encoding="utf-8")
    (d / "best.json").write_text(json.dumps({"best_ckpt": CK2, "best_val_miou_all_class": E2_BEST}), encoding="utf-8")
    if real:
        os.symlink(e2_ckpt_real(), d / CK2)
    else:
        torch.save(payload if payload is not None else e2_payload(), d / CK2)
    for name, data in (files or {}).items():
        (d / name).write_bytes(data)
    return d, sha_file(d / CK2)


def s11_records(name: str, run: Path, *, winner=None, inputs=None, launch=None, launched=None) -> Path:
    """A records repository of its own for one E2 parent (the plan's §0.3): the four registered lists, a λ
    selection whose winner and inputs entry are that run's, and its launch and launched lines; committed once."""
    root = BASE / "s11" / "records" / name
    (root / "configs" / "calibration").mkdir(parents=True)
    for s in (42, 43, 44, 45):
        (root / "configs" / "calibration" / f"ptq_calibration_seed{s}.json").write_bytes(list_path(s).read_bytes())
    entry = {"run_dir": f"/runs/{E2_S42}", "checkpoint": CK2, "best_json_sha256": sha_file(run / "best.json"),
             "run_meta_sha256": sha_file(run / "e2_run_meta.jsonl")}
    lam = {"format": "lambda_selection/1",
           "winner": winner or {"lambda": LAMBDA, "run_id": E2_S42, "best_val": E2_BEST,
                                "ckpt_sha256": sha_file(run / CK2)},
           "inputs": inputs if inputs is not None else [
               {"run_dir": "/runs/e2_s42_lambda0.25_80k_a1", "checkpoint": CK2, "best_json_sha256": "1" * 64,
                "run_meta_sha256": "2" * 64}, entry]}
    (root / "reports" / "derived").mkdir(parents=True)
    (root / LAMBDA_REL).write_text(json.dumps(lam, indent=2), encoding="utf-8")
    rm = sha_file(run / "e2_run_meta.jsonl")
    lines = [{**launch_line(E2_S42, 42, sweep="lambda_logit", value=LAMBDA), "stage": "E2", "alpha_cwd": None,
              **(launch or {})},
             {**launched_line(E2_S42, rm, WALL[42]), **(launched or {})}]
    (root / LOG_REL).write_bytes("".join(json.dumps(x) + "\n" for x in lines).encode("utf-8"))
    tracked = [f"configs/calibration/ptq_calibration_seed{s}.json" for s in (42, 43, 44, 45)] + [LAMBDA_REL, LOG_REL]
    git(root, "init", "-q")
    git(root, "add", "--", *tracked)
    git(root, "commit", "-q", "-m", "synthetic E2 records")
    return root


@contextlib.contextmanager
def records_of(root: Path):
    """s11's records repository in place of the shared one for one case (bound_env and run_real read
    FX["records"]); the shared one is restored after."""
    saved = FX["records"]
    FX["records"] = root
    try:
        yield
    finally:
        FX["records"] = saved


def argv_e2(root: Path, run: Path, sha: str, *, seed: int = 42, list_seed: int = 42, out=None, **flags) -> list[str]:
    """A passing E7-E2 command against records repository `root` (argv's form; None drops a flag)."""
    a = {"--stage": "E7-E2", "--seed": str(seed), "--source-run-dir": str(run), "--expect-source-sha256": sha,
         "--calibration-list": str(root / "configs" / "calibration" / f"ptq_calibration_seed{list_seed}.json"),
         "--expect-list-checksum": checksum(list_seed), "--expect-head": FAKE_HEAD, "--data-root": str(FX["root"]),
         "--out-dir": str(out or fresh_out(f"e7-e2_s{seed}_cal{list_seed}_a1")),
         "--lambda-selection": LAMBDA_REL, "--lambda-selection-sha256": sha_file(root / LAMBDA_REL)}
    for k, v in flags.items():
        flag = "--" + k.replace("_", "-")
        if v is None:
            a.pop(flag, None)
        else:
            a[flag] = str(v)
    return ["--real-run", "--confirm-real-run"] + [x for kv in a.items() for x in kv]


def s11_fx() -> dict:
    """The winner run (a real E2 checkpoint), its records repository and one real-core E7-E2 run, made once."""
    if "s11" not in FX:
        run, sha = e2_run("winner", real=True)
        root = s11_records("winner", run)
        out = BASE / "real" / "e7-e2_s42_cal42_a1"
        (BASE / "real").mkdir(exist_ok=True)
        with records_of(root):
            rc, log, touched = run_real("e7_e2_s42", argv_e2(root, run, sha, out=out))
        FX["s11"] = {"run": run, "sha": sha, "root": root, "out": out, "rc": rc, "log": log, "touched": touched}
    return FX["s11"]


def e2_refused(name: str, want: str, variant: str, *, meta=None, payload=None, files=None, records=None,
               **flags) -> None:
    """A stub-core E7-E2 command refused with `want`, every earlier gate passing: the variant run and a records
    repository whose λ selection and launch lines are that run's (or `records`, a builder of another)."""
    run, sha = e2_run(variant, meta=meta, payload=payload, files=files)
    root = (records or s11_records)(variant, run)
    with records_of(root):
        refused(name, argv_e2(root, run, sha, **flags), want)


def case_s11_real() -> None:
    from src.eval.stage_artifacts import int8_official_error, validate_int8_artifact
    from src.quant.ptq_record import manifest_entry, planned_entry, read_record
    fx = s11_fx()
    out, names = fx["out"], FX["names"]("e7-e2")
    rec_p = out / "e7-e2_ptq_record.json"
    rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.is_file() else {}
    check("e7_e2_s42_exit_0", fx["rc"] == 0 and last_result(fx["log"]).startswith(
        f"RESULT: PTQ RECORD WRITTEN (E7-E2 s42, role quant_descriptive, parent {E2_S42}, record ")
        and not eval_split_touched(fx["touched"]), last_result(fx["log"]))
    present = sorted(p.name for p in out.iterdir()) if out.is_dir() else []
    check("e7_e2_files", present == sorted([names["torchscript"], names["state_dict"], names["x86_copy"],
                                            names["run_meta"], "e7-e2_ptq_record.json"]), str(present))
    check("e7_e2_record_role_descriptive", rec.get("role") == "quant_descriptive" and rec.get("descriptive") is True
          and rec.get("stage") == "E7-E2" and rec.get("seed") == 42 and rec.get("replicate_of") is None,
          f"{rec.get('role')} {rec.get('descriptive')}")
    par, root = rec.get("parent", {}), fx["root"]
    sel = par.get("selections", {})
    ll = par.get("launch_log", {})
    check("e7_e2_record_parent", par.get("rule") == "e2_run_of_record/1" and par.get("stage") == "E2"
          and par.get("run_id") == E2_S42 and par.get("checkpoint_sha256") == fx["sha"]
          and par.get("lambda_logit") == LAMBDA and par.get("parent_of_e4_e7") is False
          and list(sel) == ["lambda"] and sel["lambda"].get("sha256") == sha_file(root / LAMBDA_REL)
          and par.get("selection_input", {}).get("run_meta_sha256") == sha_file(fx["run"] / "e2_run_meta.jsonl")
          and ll.get("sha256") == sha_file(root / LOG_REL) and ll.get("launch", {}).get("stage") == "E2"
          and par.get("teacher_arch") == "segnext_mscan-b", str({k: par.get(k) for k in ("rule", "run_id")}))
    check("e7_e2_manifest_entry", manifest_ok(rec, out) == [] if rec else False, str(rec.get("role")))
    planned = planned_entry("E7-E2", 42, fx["sha"], sha_file(list_path(42)), checksum(42), FAKE_HEAD, FX["digest"])
    check("e7_e2_planned_entry_role", planned.get("role") == "quant_descriptive" and planned.get("stage") == "E7-E2",
          str(planned.get("role")))
    others = [planned_entry(s, 42, "0" * 64, "1" * 64, "2" * 64, FAKE_HEAD, FX["digest"])["role"] for s in ("E4", "E7")]
    check("e4_e7_planned_entry_unchanged", others == ["of_record", "of_record"], str(others))
    err = "not run"
    try:
        err = int8_official_error(validate_int8_artifact("E7-E2", out / names["run_meta"]), repo_root=root)
    except Exception as e:                                  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    check("e7_e2_record_official", err is None, str(err)[:120])
    other = BASE / "s11_copies" / "another_parent"
    shutil.copytree(out, other)
    r2 = json.loads((other / "e7-e2_ptq_record.json").read_text(encoding="utf-8"))
    r2["parent"]["checkpoint_sha256"] = "0" * 64
    (other / "e7-e2_ptq_record.json").write_text(json.dumps(r2), encoding="utf-8")
    err = official("E7-E2", other / names["run_meta"], repo_root=root)
    check("e7_e2_official_refuses_record_of_another_run", "[ptq_record_mismatch]" in str(err), str(err)[:120])
    for name, stage, role in (("read_record_refuses_e7_quant_descriptive", "E7", "quant_descriptive"),
                              ("read_record_refuses_e7_e2_of_record", "E7-E2", "of_record")):
        d = BASE / "s11_copies" / f"coupling_{stage.lower()}_{role}"
        shutil.copytree(out, d)
        r3 = json.loads((d / "e7-e2_ptq_record.json").read_text(encoding="utf-8"))
        r3.update(stage=stage, role=role, descriptive=role != "of_record")
        (d / "e7-e2_ptq_record.json").write_text(json.dumps(r3), encoding="utf-8")
        try:
            read_record(d / "e7-e2_ptq_record.json")
            got = "accepted"
        except ValueError as e:
            got = str(e)
        check(name, "quant_descriptive is E7-E2's" in got, got[-160:])
    d = BASE / "s11_copies" / "coupling_stage_not_a_string"           # DL-24 F4-3: RecordInvalid, not TypeError
    shutil.copytree(out, d)
    r4 = json.loads((d / "e7-e2_ptq_record.json").read_text(encoding="utf-8"))
    r4["stage"] = ["E7-E2"]
    (d / "e7-e2_ptq_record.json").write_text(json.dumps(r4), encoding="utf-8")
    try:
        read_record(d / "e7-e2_ptq_record.json")
        got = "accepted"
    except ValueError as e:
        got = str(e)
    except Exception as e:                                  # noqa: BLE001 -- the defect this check pins
        got = f"{type(e).__name__}: {e}"
    check("read_record_refuses_a_stage_not_a_string", "quant_descriptive is E7-E2's" in got, got[-160:])


def case_s11_refusals() -> None:
    fx = s11_fx()
    root, run, sha = fx["root"], fx["run"], fx["sha"]
    with records_of(root):
        refused("refuses_e7_e2_alpha_flags", argv_e2(root, run, sha, alpha_selection=ALPHA_REL,
                                                     alpha_selection_sha256="0" * 64), "selection_not_applicable")
        refused("refuses_e7_e2_sensitivity_list", argv_e2(root, run, sha, list_seed=43),
                "calibration_list_not_of_record")
        refused("refuses_e7_e2_seed_43", argv_e2(root, run, sha, seed=43), "am23_seed")
        refused("refuses_e7_e2_out_dir_name", argv_e2(root, run, sha, out=fresh_out("e7_s42_cal42_a1")),
                "out_dir_name")
        refused("refuses_e7_e2_replicate",
                argv_e2(root, run, sha, out=fresh_out("e7-e2_s42_cal42_a1_d3"),
                        replicate_of=fx["out"] / "e7-e2_ptq_record.json"), "replicate_mismatch")
    other_winner = "e2_s42_lambda0.5_80k_a2"                # the λ selection names another E2 run
    e2_refused("refuses_e7_e2_parent_not_the_winner", "parent_run_id_mismatch", "not_winner",
               records=lambda name, r: s11_records(name, r, winner={"lambda": LAMBDA, "run_id": other_winner,
                                                                    "ckpt_sha256": sha_file(r / CK2)}))
    e2_refused("refuses_e7_e2_parent_lambda", "parent_lambda_mismatch", "lambda", meta=e2_meta(lambda_logit=1.0))
    e2_refused("refuses_e7_e2_checkpoint_not_selected", "parent_checkpoint_not_selected", "checkpoint",
               records=lambda name, r: s11_records(name, r, winner={"lambda": LAMBDA, "run_id": E2_S42,
                                                                    "ckpt_sha256": "0" * 64}))
    e2_refused("refuses_e7_e2_selection_input", "parent_not_selection_input", "input",
               records=lambda name, r: s11_records(name, r, inputs=[
                   {"run_dir": f"/runs/{E2_S42}", "checkpoint": CK2, "best_json_sha256": "3" * 64,
                    "run_meta_sha256": sha_file(r / "e2_run_meta.jsonl")}]))
    e2_refused("refuses_e7_e2_projection_pt", "parent_projection_present", "projection",
               files={"projection.pt": b"a projection file"})
    e2_refused("refuses_e7_e2_parent_stage_e3", "parent_stage_mismatch", "stage_e3", meta=e2_meta(stage="E3"))
    e2_refused("refuses_e7_e2_parent_not_of_record", "parent_not_of_record", "not_of_record",
               meta=e2_meta(descriptive=True))
    e2_refused("refuses_e7_e2_parent_teacher", "parent_teacher_not_of_record", "teacher",
               meta=e2_meta(teacher_mock=True))
    e2_refused("refuses_e7_e2_launch_line", "parent_launch_log", "launch",
               records=lambda name, r: s11_records(name, r, launch={"code_pin": OTHER_COMMIT}))
    e2_refused("refuses_e7_e2_payload_stage", "parent_payload", "payload",
               payload=e2_payload(stage="E3"))


def case_s11_conditions() -> None:
    """DL-24 F4-2: each single condition of bind_e2's parent_not_of_record, parent_teacher_not_of_record and
    parent_payload, failed alone with every earlier gate passing, as the E3 rule's in s5 (DL-24 D-6); stub core
    and no real run. descriptive and teacher_mock alone are s11_e7_e2_refusals' own checks."""
    for key, over in (("arm", {"arm": "E3-R1"}), ("parent_of_e4_e7", {"parent_of_e4_e7": True}),
                      ("max_iters", {"max_iters": 160000}), ("poly_horizon", {"poly_horizon": 160000})):
        e2_refused(f"refuses_e7_e2_parent_not_of_record_{key}", "parent_not_of_record", f"not_of_record_{key}",
                   meta=e2_meta(**over))
    for key in ("max_iters", "poly_horizon"):                 # AQ13: strict keys, never tolerated when absent
        meta = e2_meta()
        del meta[key]
        e2_refused(f"refuses_e7_e2_parent_not_of_record_{key}_absent", "parent_not_of_record", f"absent_{key}",
                   meta=meta)
    tp_r1 = teacher_provenance(ckpt_sha256="4c54aa1b06e787d93b04ef5469398bb19e34f4660166124d37cff69273eb9f20")
    for key, tp in (("ckpt_sha256", tp_r1), ("builder", teacher_provenance(builder="another_builder")),
                    ("expected_sha256", teacher_provenance(expected_sha256="0" * 64)), ("not_an_object", None)):
        e2_refused(f"refuses_e7_e2_parent_teacher_{key}", "parent_teacher_not_of_record", f"teacher_{key}",
                   meta=e2_meta(teacher_provenance=tp))
    for key, over in (("arm_key", {"arm": "E3-R1"}),
                      ("target_construction_key", {"target_construction": {"K": 2, "views": 2}}),
                      ("teacher_mismatch", {"teacher_provenance": teacher_provenance(config_path="other.py")}),
                      ("best_mismatch", {"best_val_miou_all_class": 0.3})):
        e2_refused(f"refuses_e7_e2_payload_{key}", "parent_payload", f"payload_{key}", payload=e2_payload(**over))


# ------------------------------------------------------------------ s10 hygiene
IMPORT_PROBE = r"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from scripts.run_ptq import main
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    rc = main(sys.argv[2:])
print(f"RC={rc} LOADED={'src.quant.ptq_record' in sys.modules}")
"""


def case_s10() -> None:
    def probe(av):
        p = subprocess.run([sys.executable, "-B", "-c", IMPORT_PROBE, str(REPO), *av], capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return (p.stdout.strip().splitlines() or [p.stderr.strip()[-120:]])[-1]
    # DL-24 D-4: the unbound run goes start to finish (the real core, its own process), and never loads the module
    out_u = fresh_out("e4_unbound_probe")
    unbound = probe(["--real-run", "--confirm-real-run", "--stage", "E4", "--source-ckpt", str(FX["e1_s42"] / CK1),
                     "--expect-source-sha256", FX["e1_sha"], "--calibration-list", str(FX["smoke_list"]),
                     "--expect-list-checksum", checksum(42), "--data-root", str(FX["root"]), "--out-dir", str(out_u)])
    inside = REPO / "ptq_record_smoke_probe" / "e4_s42_cal42_a1"
    bound = probe(argv(out=inside))
    check("unbound_run_never_imports_record_module", unbound == "RC=0 LOADED=False" and bound == "RC=2 LOADED=True"
          and (out_u / "e4_ptq_run_meta.json").is_file() and not (out_u / "e4_ptq_record.json").exists()
          and not inside.exists(), f"unbound {unbound}; bound {bound}")
    named = []
    for top in [BASE] + ([FX["cache"]] if FX.get("cache") else []):
        for dirpath, dirnames, filenames in os.walk(top):
            named += [os.path.join(dirpath, n) for n in dirnames + filenames if "test" in n.lower()]
    check("no_test_named_path_created", not named and not refusable(BASE), str(named[:3]))
    verdict = state_verdict(STATE_BEFORE, repo_state(REPO))
    check("repository_untouched", verdict == "unchanged", f"the repository's git status: {verdict}")


CASES = [
    ("s0_fixtures", "s0", case_s0),
    ("s1_old_runners", "s1", case_s1),
    ("s2_e4_s42_real", "s2", case_s2),
    ("s3_e7_s42_real", "s3", case_s3),
    ("s4_seeds_43", "s4", case_s4),
    ("s5_flags", "s5", case_s5_flags),
    ("s5_out_dir", "s5", case_s5_out_dir),
    ("s5_identity", "s5", case_s5_identity),
    ("s5_selections", "s5", case_s5_selections),
    ("s5_parent", "s5", case_s5_parent),
    ("s5_e1_rule", "s5", case_s5_e1_rule),
    ("s5_e3_rule", "s5", case_s5_e3_rule),
    ("s5_unreadable", "s5", case_s5_unreadable),
    ("s5_replicate", "s5", case_s5_replicate),
    ("s5_stops", "s5", case_s5_stops),
    ("s5_order", "s5", case_s5_order),
    ("s6_d3_real", "s6", case_s6),
    ("s7_sensitivity", "s7", case_s7),
    ("s8_gate", "s8", case_s8_gate),
    ("s8_e2e", "s8", case_s8_e2e),
    ("s8_source", "s8", case_s8_source),
    ("s9_range", "s9", case_s9),
    ("s11_e7_e2_real", "s11", case_s11_real),
    ("s11_e7_e2_refusals", "s11", case_s11_refusals),
    ("s11_e7_e2_conditions", "s11", case_s11_conditions),
    ("s10_hygiene", "s10", case_s10),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--only", default=None, help="comma-separated case names (scripts/smoke_ptq_record_mutations.py)")
    ap.add_argument("--cache-dir", default=None, help="keep (or reuse) the TRAIN tree and the two student checkpoints")
    args = ap.parse_args()
    want = set(args.sections.split(","))
    names = [n for n, _, _ in CASES]
    if want - set(SECTIONS):
        print(f"RESULT: ERROR unknown sections {sorted(want - set(SECTIONS))}")
        return 4
    if args.only:
        only = args.only.split(",")
        if set(only) - set(names):
            print(f"RESULT: ERROR unknown cases {sorted(set(only) - set(names))}")
            return 4
        todo = [c for c in CASES if c[0] in only]
    else:
        todo = [c for c in CASES if c[1] in want]
    cache = Path(args.cache_dir) if args.cache_dir else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        if refusable(cache):
            print(f"RESULT: ERROR --cache-dir {cache} names TEST or VAL")
            return 4
    t0 = time.time()
    print("=" * 78)
    print("PTQ RECORD SMOKE (LQP) — synthetic; scripts/run_ptq.py's bound path, the official gate, the range")
    print(f"{len(todo)} cases | sections {sorted({c[1] for c in todo})} | temp {BASE}")
    print("=" * 78)
    setup(cache)
    for name, _section, fn in todo:
        print(f"\n--- {name} ---", flush=True)
        before = len(results)
        try:
            fn()
        except Exception as e:                               # noqa: BLE001 -- the case fails, the smoke goes on
            check(name, False, f"the case raised {type(e).__name__}: {e}")
        if len(results) == before:
            check(name, False, "the case reported no check")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:50}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n[time] {time.time() - t0:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
