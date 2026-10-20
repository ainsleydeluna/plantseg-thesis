#!/usr/bin/env python3
"""Lane L-AM3, B-2: E6-KD's registration, conversion, scoring and epoch selection, on synthetic inputs (LB18, LB19,
LB31, LB35).

E6-KD's runs are real runs of the trainer (src/quant/qat_kd.py run_qat_kd, mode "smoke") on Q2's colour task: a
trained student (scripts/synthetic_qat_fixtures.py confident_student), the harness's stub SegNeXt teacher
(scripts/invariance_harness.py) and a fixed projection; a clean run and two whose state turns non-finite (the
trainer's state_nan injection: at step 27, and at step 1). They are kept raw with --cache-dir.

- rules: configs/e6kd_selection_rules.json against AM-23's file, and its RULES_FILES routing (LB18);
- evaluator: the evaluator's stage tables and contract for E6-KD (LB19, LB20);
- convert, score: scripts/qat_epoch_eval.py convert and score on the clean run (Q2's code, unchanged), the artifact
  loaded through src/eval/model_loading.py and validated as E6-KD (LB35); finalize refused (LB31);
- select: scripts/select_qat_epoch.py on the clean run re-labelled mode "real", over eval directories fabricated as
  scripts/smoke_qat_selection.py fabricates them (summaries holding chosen values, every hash consistent);
- unaffected: E5's and E6's rules, and select_clip.

No PlantSeg data, no checkpoint of record; a synthetic data tree and every file in a temp directory outside the
repository. No check name, case name, file or folder holds the token SL-1 forbids.

    python -B scripts/smoke_e6kd_selection.py [--sections S,...] [--only CASE,...] [--cache-dir DIR]

Ends with one RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_qat_fixtures import (confident_student, evaluator_driver, make_tree, safe_tmpdir,  # noqa: E402
                                            task_loaders)

TMP = safe_tmpdir("smoke_e6kd_selection_")
ROOT = TMP / "data"
make_tree(ROOT, n_train=16, n_val=846, seed=3)              # TRAIN and VAL only
os.environ["PLANTSEG_DATA_ROOT"] = str(ROOT)                 # before configs/data.py is first imported
os.environ["SMOKE_REPO"] = str(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import scripts.evaluate_model as EM  # noqa: E402
import scripts.qat_epoch_eval as QEE  # noqa: E402
import scripts.select_clip as SC  # noqa: E402
import scripts.select_qat_epoch as SE  # noqa: E402
from scripts.invariance_harness import build_stub_segnext  # noqa: E402
from src.distill import teacher as TT  # noqa: E402
from src.distill.segnext_teacher import SegNeXtTeacherAdapter  # noqa: E402
from src.eval import artifacts as ART  # noqa: E402
from src.eval import stage_artifacts as SA  # noqa: E402
from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME  # noqa: E402
from src.eval.model_loading import load_int8_torchscript, rebuild_int8_from_state_dict  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_kd as KD  # noqa: E402
from src.quant import qat_select as S  # noqa: E402
from src.quant.qconfig import select_qnnpack_backend  # noqa: E402
from src.quant.stages import QUANT_STAGES  # noqa: E402

results: list[tuple[str, bool, str]] = []
CASES: list[tuple[str, str, object]] = []
SECTIONS = ("rules", "evaluator", "convert", "score", "select", "unaffected")
OPTS: dict = {"cache": None}
HOST = "cloud-smoke"
DRIVER = evaluator_driver(TMP / "evaluator_driver.py")
KD_RULES = "configs/e6kd_selection_rules.json"
AM23_RULES = "configs/qat_selection_rules_am23.json"
Q2_RULES = "configs/qat_selection_rules.json"
Q2_RULES_SHA256 = "76addc57d0367dda9a1163f749ec32c9c240b20736104dc12aa77488344f9568"     # Q2's lane report §17.6
AM23_RULES_SHA256 = "e299941524968f143359d154566930819cd064152691395fb3647f4de0d98f3d"   # at O (L-AM23Q B-2)
W = {"logit_kd": 0.25, "cwd_feat": 12.5, "cwd_logit": 1.5}
E3REC = {"e3_weights": {"logit_kd": 0.5, "cwd_feat": 25.0, "cwd_logit": 3.0},
         "grids": {"logit_kd_grid": "smoke", "cwd_feat_grid": "smoke", "cwd_logit_grid": "smoke"}}
GT_SUM = 159_279_104


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def case(section: str):
    def deco(fn):
        CASES.append((fn.__name__, section, fn))
        return fn
    return deco


def quiet(*_a, **_k) -> None:
    return None


def sha_file(p) -> str:
    return Q.sha256_file(Path(p))


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def run_cli(main, argv) -> tuple[int, str, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT:")]
    return rc, (lines[-1] if lines else out[-300:]), out


def lane_cli(script: str, *argv: str) -> tuple[int, str, str]:
    """A lane script in a fresh process (convert-epoch, select and finalize trace or convert), from the root."""
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / script), *argv], cwd=str(REPO),
                       capture_output=True, text=True, timeout=3000)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT:")]
    return p.returncode, (lines[-1] if lines else p.stderr.strip()[-300:]), p.stdout


# ---------------------------------------------------------------- E6-KD runs (kept raw with --cache-dir)
def stub_teacher():
    return TT.FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext()))


def projection_fixture(seed: int = 7) -> nn.Conv2d:
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.utils.skip_init(nn.Conv2d, 160, 320, 1, bias=False)
    with torch.no_grad():
        p.weight.copy_(torch.randn(320, 160, 1, 1, generator=g) * 0.05)
    return p


def clone_run(src: Path, dst: Path) -> Path:
    """A copy of a run: checkpoints hard-linked (never rewritten), the record copied."""
    (dst / Q.EPOCH_DIR).mkdir(parents=True)
    for p in (src / Q.EPOCH_DIR).iterdir():
        os.link(p, dst / Q.EPOCH_DIR / p.name)
    shutil.copy2(src / Q.TELEMETRY_NAME, dst / Q.TELEMETRY_NAME)
    return dst


@functools.lru_cache(maxsize=None)
def source_student_state() -> dict:
    return {k: v.clone() for k, v in confident_student(seed=0, steps=30).state_dict().items()}


def train_kd(dst: Path, inject: dict | None) -> None:
    from src.models.student import build_student
    model = build_student(pretrained=False)
    model.load_state_dict(source_student_state())
    KD.run_qat_kd(mode="smoke", model=model, teacher=stub_teacher(), projection=projection_fixture(),
                  weights=dict(W), e3=E3REC, source_meta={"sha256": "0" * 64, "path": None, "bytes": None},
                  out_dir=dst, seed=42, clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0,
                  loaders=task_loaders(42), inject=inject, log=quiet)


RAW = {"run_e6kd_clean": None, "run_e6kd_state_nan27": {"state_nan": [27]}, "run_e6kd_state_nan1": {"state_nan": [1]}}


@functools.lru_cache(maxsize=None)
def raw_run(name: str) -> Path:
    """A 15-epoch E6-KD smoke run, trained once (or cloned from --cache-dir, where it is kept raw)."""
    cache = OPTS["cache"]
    if cache is not None and (cache / name / "READY").is_file():
        return clone_run(cache / name, TMP / name)
    train_kd(TMP / name, RAW[name])
    if cache is not None:
        clone_run(TMP / name, cache / name)
        (cache / name / "READY").write_text("raw E6-KD smoke run\n", encoding="utf-8")
    return TMP / name


def fill_cache() -> None:
    """--cache-dir: the three raw runs, all at once and before any case, so that jobs started later (the mutation
    harness's pooled baselines) only clone them (B-1's fill_cache; DL-24 SM2)."""
    cache = OPTS["cache"]
    if cache is None or all((cache / name / "READY").is_file() for name in RAW):
        return
    for name in RAW:
        raw_run(name)


def relabel(run: Path, *, meta: dict | None = None) -> str:
    """Re-label a smoke run's record mode "real", a launch bound to clip_selection.json (the checkpoints are
    untouched). Returns the new telemetry sha256."""
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    rows[0].update({"mode": "real", "u4_pilot": False, "clip_source": "clip_selection", **(meta or {})})
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Q.sha256_file(tel)


def real_run(raw: str, name: str) -> tuple[Path, str]:
    run = clone_run(raw_run(raw), TMP / name)
    return run, relabel(run)


# ---------------------------------------------------------------- fabricated eval directories (Q2's form)
def summary(value: float, run_id: str, ts_sha: str, stage: str, *, commit: str = "a" * 40) -> dict:
    gt = [1000] * 116
    gt[0] = GT_SUM - 115_000
    return {"schema_version": "plantseg-eval/1.0.0",
            "run": {"run_id": run_id, "artifact_status": "provisional", "stage": stage, "model_role": "student",
                    "precision": "int8_qat", "quant_backend": "qnnpack", "checkpoint_sha256": ts_sha,
                    "random_init": False, "repo_commit": commit, "governed_paths_clean": True,
                    "metric_impl_sha256": "b" * 64, "config_sha256": "c" * 64,
                    "env": {"torch": "2.1.0+cpu", "device": "cpu"},
                    "eval_runtime": {"batch_size": 1, "forward_batches": 846, "num_workers": 0,
                                     "torch_num_threads": 4, "image_digest": "sha256:" + "6" * 64}},
            "dataset": {"split": "val", "expected_rows": 846, "actual_rows": 846,
                        "preprocess_protocol": "core_preprocess/1.0.0", "split_manifest_sha256": "d" * 64,
                        "class_map_sha256": "e" * 64},
            "dataset_level": {"all_class_miou": value}, "per_class": {"gt_support": gt}}


def write_artifact(d: Path, s: dict) -> None:
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(s, sort_keys=True), encoding="utf-8")
    (d / "per_image.jsonl").write_text("", encoding="utf-8")
    with open(d / "sufficient_stats.npz", "wb") as fh:
        np.savez(fh, tp=np.zeros(116, dtype=np.int64))
    lines = [f"{Q.sha256_file(d / n)}  {n}" for n in ARTIFACT_FILES]
    (d / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fabricate(run: Path, ev: Path, values: dict, *, excluded=()) -> None:
    """converted/, scores/, qat_convert.json and qat_epoch_eval.json for `run`, scored with `values`
    (scripts/smoke_qat_selection.py fabricate, for E6-KD)."""
    commit, host_label = "a" * 40, "fab-host"
    rec = A.read_run_record(run)
    meta = rec["run_meta"]
    conv, scores = ev / A.CONVERTED_DIR, ev / A.SCORES_DIR
    conv.mkdir(parents=True)
    outcomes, rows = [], []
    for e in range(1, 16):
        names = A.epoch_names(e)
        ck_sha = rec["ends"][e]["checkpoint_sha256"]
        if e in excluded:
            (conv / names["not_convertible"]).write_text(json.dumps(
                {"status": "not convertible", "rule": A.NOT_CONVERTIBLE_RULE, "qat_checkpoint_sha256": ck_sha,
                 "failing": ["fabricated"], "epoch": e, "run_id": rec["run_id"],
                 "telemetry_sha256": rec["telemetry_sha256"], "purpose": "record", "host_label": host_label,
                 "cpu_model": "fab-cpu", "git_head": commit}), encoding="utf-8")
            outcomes.append({"epoch": e, "status": "not convertible"})
            rows.append({"epoch": e, "checkpoint_sha256": ck_sha, "status": "excluded", "rule": A.NOT_CONVERTIBLE_RULE,
                         "not_convertible": f"{A.CONVERTED_DIR}/{names['not_convertible']}"})
            continue
        ident = {"schema": "plantseg-int8-torchscript/1.0.0", "stage": meta["stage"], "quantization": "qat",
                 "num_classes": 116, "engine": "qnnpack", "artifact_role": "accuracy",
                 "source_checkpoint_sha256": meta["source_checkpoint_sha256"], "qat_checkpoint_sha256": ck_sha,
                 "epoch": e, "run_id": rec["run_id"], "qconfig_fingerprint": "f" * 16,
                 "state_dict_companion": names["state_dict"]}
        ts = conv / names["torchscript"]
        with zipfile.ZipFile(ts, "w") as z:
            z.writestr(f"e{e:02d}/extra/plantseg_int8.json", json.dumps(ident))
            z.writestr(f"e{e:02d}/data.pkl", f"fabricated {rec['run_id']} e{e:02d}")
        ts_sha = Q.sha256_file(ts)
        sd = conv / names["state_dict"]
        sd.write_bytes(f"fabricated state_dict {rec['run_id']} e{e:02d}".encode())
        prov = {"stage": meta["stage"], "epoch": e, "run_id": rec["run_id"],
                "state_dict_artifact": {"path": names["state_dict"], "sha256": Q.sha256_file(sd)},
                "git_head": commit, "host_label": host_label, "cpu_model": "fab-cpu", "purpose": "record",
                "qat_checkpoint_sha256": ck_sha, "converted_artifact": names["torchscript"],
                "converted_artifact_sha256": ts_sha,
                "verification": {"checks": {"weights_per_channel_int8": True, "no_undeclared_float_module": True,
                                            "torchscript_float_boundary": True,
                                            "parity_torchscript_state_dict_eager": True}}}
        pp = conv / names["run_meta"]
        pp.write_text(json.dumps(prov), encoding="utf-8")
        run_id = f"{rec['run_id']}_e{e:02d}_{ck_sha[:12]}"
        write_artifact(scores / f"e{e:02d}", summary(values[e], run_id, ts_sha, meta["stage"], commit=commit))
        outcomes.append({"epoch": e, "status": "converted"})
        rows.append({"epoch": e, "checkpoint_sha256": ck_sha, "status": "scored", "run_id": run_id,
                     "provenance": {"path": f"{A.CONVERTED_DIR}/{names['run_meta']}", "sha256": Q.sha256_file(pp)},
                     "converted_artifact_sha256": ts_sha, "score_dir": f"{A.SCORES_DIR}/e{e:02d}",
                     "summary": {"path": f"{A.SCORES_DIR}/e{e:02d}/summary.json",
                                 "sha256": Q.sha256_file(scores / f"e{e:02d}" / "summary.json")},
                     "all_class_miou": values[e],
                     "evaluator_command": ["-B", "scripts/evaluate_model.py", "--stage", meta["stage"]],
                     "torch_num_threads": 4, "affinity_cpus": 4})
    head = {"run_dir": rec["run_dir"], "run_id": rec["run_id"], "stage": meta["stage"],
            "telemetry_sha256": rec["telemetry_sha256"], "purpose": "record", "host_label": host_label,
            "cpu_model": "fab-cpu", "git_head": commit, "smoke_inputs": False}
    (ev / "qat_convert.json").write_text(json.dumps({"format": "qat_convert/1", **head, "outcomes": outcomes}),
                                         encoding="utf-8")
    (ev / "qat_epoch_eval.json").write_text(json.dumps({"format": "qat_epoch_eval/1", **head, "epochs": rows}),
                                            encoding="utf-8")


def vals(**kw) -> dict:
    v = {e: 0.30 for e in range(1, 16)}
    for k, x in kw.items():
        v[int(k[1:])] = x
    return v


def select(run: Path, ev: Path, sha: str) -> tuple[int, str, dict]:
    rc, res, _out = run_cli(SE.main, ["--run-dir", str(run), "--eval-dir", str(ev), "--expect-telemetry-sha256", sha])
    p = ev / "qat_selection.json"
    return rc, res, (json.loads(p.read_text(encoding="utf-8")) if rc == 0 and p.is_file() else {})


def fabricated_select(tag: str, values: dict, *, raw: str = "run_e6kd_clean", excluded=()) -> tuple[int, str, dict]:
    run, sha = real_run(raw, f"real_{tag}")
    ev = TMP / f"ev_{tag}"
    fabricate(run, ev, values, excluded=excluded)
    return select(run, ev, sha)


# ---------------------------------------------------------------- the real conversion and scoring (LB35, LB31)
@functools.lru_cache(maxsize=None)
def converted() -> dict:
    """scripts/qat_epoch_eval.py convert (record, every epoch in a fresh process) on the clean smoke run."""
    run = clone_run(raw_run("run_e6kd_clean"), TMP / "conv_run")
    ev = TMP / "conv_eval"
    rc, res, out = run_cli(QEE.main, ["convert", "--run-dir", str(run), "--eval-dir", str(ev), "--purpose", "record",
                                      "--host-label", HOST, "--allow-smoke-inputs", "--expect-telemetry-sha256",
                                      Q.sha256_file(run / Q.TELEMETRY_NAME)])
    return {"run": run, "ev": ev, "rc": rc, "res": res, "out": out}


@functools.lru_cache(maxsize=None)
def scored() -> dict:
    """score (record, the evaluator capped at one VAL row: Q2's stand-in driver, git calls stubbed), then
    select_qat_epoch with smoke inputs allowed, on the converted eval directory."""
    c = converted()
    old = QEE.EVALUATOR_SCRIPT
    QEE.EVALUATOR_SCRIPT = str(DRIVER)
    env_before = {k: os.environ.get(k) for k in ("SMOKE_MAX_SAMPLES", "SMOKE_GIT_HEAD", "PLANTSEG_IMAGE_DIGEST")}
    os.environ["SMOKE_MAX_SAMPLES"] = "1"
    os.environ["SMOKE_GIT_HEAD"] = Q.code_identity()["git_head"] or "0" * 40
    os.environ.setdefault("PLANTSEG_IMAGE_DIGEST", "sha256:" + "6" * 64)
    try:
        rc, res, out = run_cli(QEE.main, ["score", "--run-dir", str(c["run"]), "--eval-dir", str(c["ev"]),
                                          "--purpose", "record", "--host-label", HOST, "--allow-smoke-inputs"])
    finally:
        QEE.EVALUATOR_SCRIPT = old
        for k, v in env_before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rc_s, res_s, _ = lane_cli("select_qat_epoch.py", "--run-dir", str(c["run"]), "--eval-dir", str(c["ev"]),
                              "--expect-telemetry-sha256", Q.sha256_file(c["run"] / Q.TELEMETRY_NAME),
                              "--allow-smoke-inputs")
    return {**c, "score_rc": rc, "score_res": res, "select_rc": rc_s, "select_res": res_s}


def epoch_paths(e: int) -> dict:
    c = converted()
    names = A.epoch_names(e)
    return {k: c["ev"] / A.CONVERTED_DIR / names[k] for k in ("torchscript", "state_dict", "run_meta")}


# ---------------------------------------------------------------- rules (LB18)
def read_json(rel: str) -> dict:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


@case("rules")
def e6kd_rules_equal_am23_but_stages_source_note():
    kd, am = read_json(KD_RULES), read_json(AM23_RULES)

    def strip(d):
        return {k: v for k, v in d["qat_epoch"].items() if k not in ("stages", "source")}
    ok = (sorted(kd) == sorted(am) == ["format", "note", "qat_epoch"] and kd["format"] == S.RULES_FORMAT
          and strip(kd) == strip(am) and sorted(kd["qat_epoch"]) == sorted(am["qat_epoch"])
          and kd["qat_epoch"]["stages"] == ["E6-KD"] and kd["note"] != am["note"]
          and kd["qat_epoch"]["source"] != am["qat_epoch"]["source"])
    diff = sorted(k for k in set(kd["qat_epoch"]) | set(am["qat_epoch"])
                  if kd["qat_epoch"].get(k) != am["qat_epoch"].get(k))
    check("e6kd_rules_equal_am23_but_stages_source_note", ok, f"qat_epoch keys that differ: {diff}")


CHECKED_EDITS = {"format": ("format", "qat_selection_rules/0"), "epochs": ("qat_epoch.epochs", 14),
                 "stages": ("qat_epoch.stages", ["E6-KD", "E6"]), "compare": ("qat_epoch.compare", "greater_or_equal"),
                 "tie": ("qat_epoch.tie", "last_epoch"),
                 "equal_across_evaluations": ("qat_epoch.equal_across_evaluations", ["git_head"]),
                 "purpose": ("qat_epoch.purpose", "timing"), "qat_clip": ("qat_clip", {"candidates": [1.0, 5.0]})}


@case("rules")
def e6kd_rules_mismatch_refused():
    """Each key load_rules checks (qat_select.py load_rules), changed in a copy that sits at the E6-KD file's path
    under a patched repository root, so RULES_FILES gives it E6-KD's entry: refused rules_mismatch; the copy as
    committed loads."""
    root = TMP / "rules_root"
    (root / "configs").mkdir(parents=True)
    path = root / KD_RULES
    got = {}
    with patched(S, "REPO", root):
        path.write_bytes((REPO / KD_RULES).read_bytes())
        try:
            S.load_rules(path)
            got["as_committed"] = None
        except Q.QATRefused as e:
            got["as_committed"] = e.code
        for key, (dotted, value) in CHECKED_EDITS.items():
            doc = read_json(KD_RULES)
            node = doc
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            try:
                S.load_rules(path)
                got[key] = None
            except Q.QATRefused as e:
                got[key] = e.code
    ok = got.pop("as_committed") is None and all(v == "rules_mismatch" for v in got.values()) and len(got) == 8
    check("e6kd_rules_mismatch_refused", ok, f"{got}")


@case("rules")
def e6kd_rules_routed_by_stage():
    kd_path = REPO / KD_RULES
    ok = (S.RULES_FILES.get(KD_RULES) == {"stages": ["E6-KD"], "qat_clip": False}
          and S.rules_path_for_stage("E6-KD").resolve() == kd_path.resolve()
          and S.rules_path_for_stage("e6-kd").resolve() == kd_path.resolve()
          and S.rules_path_for_stage("E6").resolve() == (REPO / Q2_RULES).resolve()
          and S.rules_path_for_stage("E6-R1").resolve() == (REPO / AM23_RULES).resolve()
          and S.rules_spec(kd_path) == {"stages": ["E6-KD"], "qat_clip": False}
          and S.rules_rel_of(S.load_rules(kd_path)[0]) == KD_RULES)
    check("e6kd_rules_routed_by_stage", ok, f"{S.RULES_FILES}")


@case("rules")
def e6kd_rules_sha_recorded():
    rc, res, sel = fabricated_select("rules_sha", vals(e07=0.40))
    trace = sel.get("rule_trace") or [""]
    ok = (rc == 0 and sel.get("rules_sha256") == sha_file(REPO / KD_RULES)
          and trace[0].startswith(f"rule {KD_RULES} qat_epoch") and sel.get("winner", {}).get("epoch") == 7)
    check("e6kd_rules_sha_recorded", ok, f"{res}; trace {trace[0][:80]!r}")


@case("rules")
def q2_and_am23_rules_unchanged():
    got = (sha_file(REPO / Q2_RULES), sha_file(REPO / AM23_RULES))
    check("q2_and_am23_rules_unchanged", got == (Q2_RULES_SHA256, AM23_RULES_SHA256), f"{got[0][:12]} {got[1][:12]}")


# ---------------------------------------------------------------- evaluator: the stage tables and contract (LB19, LB20)
@case("evaluator")
def em_cli_accepts_e6kd():
    with contextlib.redirect_stderr(io.StringIO()):
        args = EM.build_parser().parse_args(["--stage", "E6-KD", "--out-dir", str(TMP / "em_never_run")])
    check("em_cli_accepts_e6kd", args.stage == "E6-KD" and "E6-KD" in ART.STAGES and ART.STAGES[-1] == "E6-KD",
          f"{list(ART.STAGES)[-3:]}")


@case("evaluator")
def schema_admits_e6kd():
    from src.eval.artifacts import ArtifactRequestError, validate_artifact_request
    import inspect
    # the request gate refuses an unknown stage before anything else; E6-KD passes that check
    src = inspect.getsource(validate_artifact_request)
    check("schema_admits_e6kd", "E6-KD" in ART.STAGES and "req.run.stage not in STAGES" in src
          and issubclass(ArtifactRequestError, Exception), "the schema's stages admit E6-KD")


@case("evaluator")
def stage_contract_e6kd():
    spec = SA.STAGE_ARTIFACTS.get("E6-KD")
    ok = (spec == {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E3", "method": "qat"}
          and SA.resolve_stage_artifact("E6-KD")["stage"] == "E6-KD" and "E6-KD" in SA.PROJECTION_FREE_STAGES
          and SA.PROJECTION_FREE_STAGES[-1] == "E6-KD" and QUANT_STAGES["e6-kd"]["method"] == "qat_kd")
    check("stage_contract_e6kd", ok, f"{spec}")


@case("evaluator")
def e6kd_descriptive_only():
    check("e6kd_descriptive_only", "E6-KD" in SA.DESCRIPTIVE_ONLY_STAGES and SA.DESCRIPTIVE_ONLY_STAGES[-1] == "E6-KD",
          f"{SA.DESCRIPTIVE_ONLY_STAGES}")


@case("evaluator")
def e6kd_provenance_needs_projection_flag_false():
    p = epoch_paths(15)["run_meta"]
    prov = json.loads(p.read_text(encoding="utf-8"))
    got = {}
    for tag, flag in (("true", True), ("missing", None)):
        q = TMP / f"prov_flag_{tag}.json"
        doc = dict(prov)
        if flag is None:
            doc.pop("cwd_projection_loaded", None)
        else:
            doc["cwd_projection_loaded"] = flag
        doc["converted_artifact"] = str(epoch_paths(15)["torchscript"])
        q.write_text(json.dumps(doc), encoding="utf-8")
        try:
            SA.validate_int8_artifact("E6-KD", q)
            got[tag] = None
        except SA.StageArtifactError as e:
            got[tag] = getattr(e, "code", type(e).__name__)
    check("e6kd_provenance_needs_projection_flag_false",
          got == {"true": "projection_flag_missing", "missing": "projection_flag_missing"}, f"{got}")


@case("evaluator")
def e6kd_artifact_refused_as_e6():
    paths = epoch_paths(15)
    select_qnnpack_backend()
    got = {}
    for stage in ("E6", "E5", "E6-R1"):
        try:
            load_int8_torchscript(paths["torchscript"], stage=stage, method="qat", engine="qnnpack")
            got[stage] = "loaded"
        except Exception as e:                                   # noqa: BLE001 -- the refusal is the check
            got[stage] = type(e).__name__
        try:
            SA.validate_int8_artifact(stage, paths["run_meta"])
            got[f"{stage} validate"] = "valid"
        except SA.StageArtifactError as e:
            got[f"{stage} validate"] = getattr(e, "code", type(e).__name__)
    ok = all(v not in ("loaded", "valid") for v in got.values())
    check("e6kd_artifact_refused_as_e6", ok, f"{got}")


@case("evaluator")
def e6_artifact_refused_as_e6kd():
    """An artifact whose embedded identity names E6 (the converted E6-KD file with its identity edited) is refused as
    E6-KD by the loader, and an E6 provenance as E6-KD by the stage contract."""
    paths = epoch_paths(15)
    src = paths["torchscript"]
    dst = TMP / "as_e6.torchscript.pt"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith("extra/plantseg_int8.json"):
                ident = json.loads(data)
                ident["stage"] = "E6"
                data = json.dumps(ident).encode()
            zout.writestr(item, data)
    select_qnnpack_backend()
    try:
        load_int8_torchscript(dst, stage="E6-KD", method="qat", engine="qnnpack")
        loaded = "loaded"
    except Exception as e:                                       # noqa: BLE001 -- the refusal is the check
        loaded = type(e).__name__
    prov = json.loads(paths["run_meta"].read_text(encoding="utf-8"))
    q = TMP / "prov_as_e6.json"
    q.write_text(json.dumps({**prov, "stage": "E6", "converted_artifact": str(src)}), encoding="utf-8")
    try:
        SA.validate_int8_artifact("E6-KD", q)
        valid = "valid"
    except SA.StageArtifactError as e:
        valid = getattr(e, "code", type(e).__name__)
    check("e6_artifact_refused_as_e6kd", loaded != "loaded" and valid == "stage_mismatch", f"{loaded}; {valid}")


# ---------------------------------------------------------------- convert (LB35) and finalize (LB31)
@case("convert")
def convert_e6kd_run_converted_15_of_15():
    c = converted()
    check("convert_e6kd_run_converted_15_of_15", c["rc"] == 0 and c["res"].startswith("RESULT: CONVERTED 15/15"),
          c["res"])


@case("convert")
def converted_identity_stage_e6kd():
    p = epoch_paths(15)["torchscript"]
    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if n.endswith("extra/plantseg_int8.json"))
        ident = json.loads(z.read(name))
    ok = ident.get("stage") == "E6-KD" and ident.get("quantization") == "qat" and ident.get("epoch") == 15
    check("converted_identity_stage_e6kd", ok, f"{ {k: ident.get(k) for k in ('stage', 'quantization', 'epoch')} }")


@case("convert")
def converted_provenance_stage_and_source():
    prov = json.loads(epoch_paths(15)["run_meta"].read_text(encoding="utf-8"))
    ok = (prov.get("stage") == "E6-KD" and prov.get("source_stage") == "E3" and prov.get("cwd_projection_loaded") is False
          and prov.get("quantization_method") == "qat")
    check("converted_provenance_stage_and_source", ok,
          f"{ {k: prov.get(k) for k in ('stage', 'source_stage', 'cwd_projection_loaded', 'quantization_method')} }")


@case("convert")
def convert_freeze_cross_check():
    """The freezes took in the E6-KD run (src/quant/qat_artifacts.py freeze_cross_check, which conversion and selection
    read): BN statistics of e11-e15 equal e10's, observer buffers of e13-e15 equal e12's."""
    fc = A.freeze_cross_check(A.read_run_record(raw_run("run_e6kd_clean")))
    check("convert_freeze_cross_check", fc["ok"] is True and fc["bn_keys"] > 0 and fc["observer_keys"] > 0
          and all(fc["bn_equal_to_e10"].values()) and all(fc["observers_equal_to_e12"].values()),
          f"first differences {fc['first_differences']}")


@case("convert")
def convert_refuses_projection_key():
    """An eNN.pt whose state carries a projection key (its record re-hashed to match): convert-epoch refuses it."""
    run = clone_run(raw_run("run_e6kd_clean"), TMP / "proj_key_run")
    ck = run / Q.EPOCH_DIR / "e15.pt"
    payload = torch.load(ck, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["cwd_projection.weight"] = torch.zeros(320, 160, 1, 1)
    tmp = ck.with_name("e15.pt.new")
    torch.save(payload, tmp)
    os.replace(tmp, ck)                                         # a new file: the cached checkpoint is never rewritten
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r.get("event") == "epoch_end" and r["epoch"] == 15:
            r["checkpoint_sha256"] = Q.sha256_file(ck)
            r["checkpoint_bytes"] = ck.stat().st_size
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ev = TMP / "proj_key_eval"
    rc, res, out = lane_cli("qat_epoch_eval.py", "convert", "--run-dir", str(run), "--eval-dir", str(ev), "--purpose",
                            "timing", "--host-label", HOST, "--allow-smoke-inputs", "--epochs", "15")
    check("convert_refuses_projection_key", "state_key_mismatch" in out and rc != 0, f"{rc} {res}")


@case("convert")
def finalize_refuses_e6kd():
    s = scored()
    rc, res, _ = lane_cli("qat_epoch_eval.py", "finalize", "--run-dir", str(s["run"]), "--eval-dir", str(s["ev"]),
                          "--allow-smoke-inputs")
    want = ("RESULT: ERROR [X86LatencyCopyError] -- E6-KD is a QAT_KD stage; use the PTQ path with the frozen "
            "calibration subset instead")
    nothing = not (s["ev"] / QEE.FINAL_DIR).exists() and not (s["ev"] / QEE.FINALIZE_RECORD).exists()
    check("finalize_refuses_e6kd", s["select_rc"] == 0 and rc == 4 and res == want and nothing,
          f"select {s['select_res']}; finalize {rc} {res}; nothing written {nothing}")


@case("convert")
def artifact_loads_through_model_loading_as_e6kd():
    paths = epoch_paths(15)
    select_qnnpack_backend()
    ts, _ident = load_int8_torchscript(paths["torchscript"], stage="E6-KD", method="qat", engine="qnnpack")
    sd = rebuild_int8_from_state_dict(paths["state_dict"], stage="E6-KD", method="qat")
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        same = torch.equal(ts(x).argmax(1), sd(x).argmax(1))
    check("artifact_loads_through_model_loading_as_e6kd", same, "TorchScript and state_dict forms agree on argmax")


@case("convert")
def artifact_passes_validate_int8_artifact_as_e6kd():
    out = SA.validate_int8_artifact("E6-KD", epoch_paths(15)["run_meta"])
    prov = out.get("provenance") or {}
    check("artifact_passes_validate_int8_artifact_as_e6kd", prov.get("stage") == "E6-KD"
          and prov.get("cwd_projection_loaded") is False, f"{sorted(out)[:5]}")


# ---------------------------------------------------------------- score (the evaluator through Q2's driver)
@case("score")
def score_through_evaluator_driver():
    s = scored()
    check("score_through_evaluator_driver", s["score_rc"] == 0 and s["score_res"].startswith("RESULT: SCORED"),
          s["score_res"])


@case("score")
def summary_stage_e6kd():
    s = scored()
    doc = json.loads((s["ev"] / A.SCORES_DIR / "e15" / "summary.json").read_text(encoding="utf-8"))
    run = doc.get("run") or {}
    check("summary_stage_e6kd", run.get("stage") == "E6-KD" and run.get("precision") == "int8_qat"
          and run.get("quant_backend") == "qnnpack" and run.get("model_role") == "student", f"{run.get('stage')}")


@case("score")
def scores_dir_contents():
    s = scored()
    d = s["ev"] / A.SCORES_DIR
    names = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    files_ok = all(sorted(p.name for p in (d / f"e{e:02d}").iterdir()) == sorted([*ARTIFACT_FILES, MANIFEST_NAME])
                   for e in range(1, 16)) if names else False
    check("scores_dir_contents", names == [f"e{e:02d}" for e in range(1, 16)] and files_ok, f"{names[:3]}")


# ---------------------------------------------------------------- select: select_qat_epoch on E6-KD runs
@case("select")
def select_strictly_greater():
    rc, res, sel = fabricated_select("greater", vals(e05=0.41, e09=0.42))
    check("select_strictly_greater", rc == 0 and sel["winner"]["epoch"] == 9
          and res == "RESULT: SELECTED epoch 09 (E6-KD, seed 42, clip 5.0)", res)


@case("select")
def select_tie_goes_to_earlier_epoch():
    rc, res, sel = fabricated_select("tie", vals(e05=0.41, e09=0.41))
    check("select_tie_goes_to_earlier_epoch", rc == 0 and sel["winner"]["epoch"] == 5, res)


@case("select")
def select_fake_quant_never_selects():
    run, _sha = real_run("run_e6kd_clean", "real_fq")
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r.get("event") == "epoch_end":
            r["fake_quant_val_all_class_miou"] = 0.9 if r["epoch"] == 11 else 0.1
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ev = TMP / "ev_fq"
    fabricate(run, ev, vals(e04=0.45))
    rc, res, sel = select(run, ev, Q.sha256_file(tel))
    check("select_fake_quant_never_selects", rc == 0 and sel["winner"]["epoch"] == 4
          and sel.get("fake_quant_val_argmax_epoch") == 11 and sel.get("fake_quant_val_used") is False, res)


@case("select")
def select_nonfinite_epoch_excluded():
    rc, res, sel = fabricated_select("nan27", vals(e02=0.33, e15=0.9), raw="run_e6kd_state_nan27",
                                     excluded=range(14, 16))
    check("select_nonfinite_epoch_excluded", rc == 0 and sel["winner"]["epoch"] == 2
          and sel.get("excluded_epochs") == [14, 15]
          and any("excluded: non-finite state (AM-21 item 2(a))" in t for t in sel.get("rule_trace", [])), res)


@case("select")
def select_no_convertible_epoch_no_model():
    rc, res, _sel = fabricated_select("nan1", vals(), raw="run_e6kd_state_nan1", excluded=range(1, 16))
    check("select_no_convertible_epoch_no_model", rc == 2 and "[no_convertible_epoch]" in res
          and S.NO_MODEL_ENTRY in res, res)


@case("select")
def select_stop_file_refused():
    run, sha = real_run("run_e6kd_clean", "real_stop")
    ev = TMP / "ev_stop"
    fabricate(run, ev, vals(e07=0.4))
    (ev / "score_STOP.json").write_text(json.dumps({"stop": "smoke"}), encoding="utf-8")
    rc, res, _ = select(run, ev, sha)
    check("select_stop_file_refused", rc == 2 and "[stop_present]" in res, res)


@case("select")
def select_telemetry_mismatch_refused():
    run, _sha = real_run("run_e6kd_clean", "real_telemetry")
    ev = TMP / "ev_telemetry"
    fabricate(run, ev, vals(e07=0.4))
    rc, res, _ = select(run, ev, "f" * 64)
    check("select_telemetry_mismatch_refused", rc == 2 and "telemetry" in res, res)


@case("select")
def select_written_once():
    run, sha = real_run("run_e6kd_clean", "real_once")
    ev = TMP / "ev_once"
    fabricate(run, ev, vals(e07=0.4))
    first = select(run, ev, sha)
    rc, res, _ = select(run, ev, sha)
    check("select_written_once", first[0] == 0 and rc == 2 and "[output_exists]" in res, res)


@case("select")
def select_recomputation_equal():
    run, sha = real_run("run_e6kd_clean", "real_recompute")
    ev = TMP / "ev_recompute"
    fabricate(run, ev, vals(e06=0.37))
    rc, _res, stored = select(run, ev, sha)
    rules, rsha = S.load_rules(S.rules_path_for_run(run))
    recomputed = S.epoch_selection(run, ev, expect_telemetry_sha256=sha, rules=rules, rules_sha256=rsha)
    diff = S.selection_differences(stored, recomputed)
    check("select_recomputation_equal", rc == 0 and not diff and rsha == sha_file(REPO / KD_RULES), f"{diff[:3]}")


@case("select")
def select_trace_names_the_rules_file():
    rc, res, sel = fabricated_select("trace", vals(e03=0.5))
    first = (sel.get("rule_trace") or [""])[0]
    check("select_trace_names_the_rules_file", rc == 0 and first.startswith(f"rule {KD_RULES} qat_epoch")
          and Q2_RULES not in first, first[:90])


# ---------------------------------------------------------------- unaffected: E5/E6 and select_clip
@case("unaffected")
def e5_e6_selections_use_q2_rules():
    ok = all(S.rules_path_for_stage(s).resolve() == (REPO / Q2_RULES).resolve() for s in ("E5", "E6"))
    rules, rsha = S.load_rules(REPO / Q2_RULES)
    check("e5_e6_selections_use_q2_rules", ok and rsha == Q2_RULES_SHA256 and rules["qat_epoch"]["stages"] == ["E5", "E6"]
          and S.RULES_FILES.get(Q2_RULES) == {"stages": ["E5", "E6"], "qat_clip": True}, rsha[:12])


@case("unaffected")
def e6kd_run_refused_under_q2_rules():
    run, sha = real_run("run_e6kd_clean", "real_q2rules")
    ev = TMP / "ev_q2rules"
    fabricate(run, ev, vals(e07=0.4))
    rules, rsha = S.load_rules(REPO / Q2_RULES)
    try:
        S.epoch_selection(run, ev, expect_telemetry_sha256=sha, rules=rules, rules_sha256=rsha)
        got = None
    except Q.QATRefused as e:
        got = e.code
    check("e6kd_run_refused_under_q2_rules", got == "stage", f"[{got}]")     # the stage rule (DL-24 RG2)


@case("unaffected")
def select_clip_unaffected():
    """select_clip takes the two E5 seed-42 pilot runs only: an E6-KD run as a candidate is refused."""
    run, sha = real_run("run_e6kd_clean", "real_clip")
    ev = TMP / "ev_clip"
    fabricate(run, ev, vals(e07=0.4))
    rc, res, _ = run_cli(SC.main, ["--candidate", str(run), str(ev), "--expect-telemetry-sha256", sha,
                                   "--candidate", str(run), str(ev), "--expect-telemetry-sha256", sha,
                                   "--out", str(TMP / "clip_never_written.json")])
    check("select_clip_unaffected", rc == 2 and "[not_a_pilot_run]" in res                  # its own rule (DL-24 RG2)
          and not (TMP / "clip_never_written.json").exists() and QUANT_STAGES["e5"]["method"] == "qat", res)


# ---------------------------------------------------------------- main
def forbidden_names_created() -> list[str]:
    """Every path under this smoke's own temp directory whose name holds the token SL-1 forbids, or ':'."""
    token = "".join(("t", "e", "s", "t"))
    bad = []
    for d, dirs, files in os.walk(TMP):
        for n in dirs + files:
            if token in n.lower() or ":" in n:
                bad.append(os.path.join(d, n))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--only", default=None, help="comma-separated case names (scripts/smoke_e6kd_mutations.py)")
    ap.add_argument("--cache-dir", default=None, help="keep (or reuse) the raw E6-KD runs here")
    args = ap.parse_args()
    want = set(args.sections.split(","))
    names = [n for n, _, _ in CASES]
    token = "".join(("t", "e", "s", "t"))
    if want - set(SECTIONS) or len(set(names)) != len(names) or any(token in n.lower() for n in names):
        print(f"RESULT: ERROR unknown sections {sorted(want - set(SECTIONS))}, or a duplicate or forbidden case name")
        return 4
    if args.only:
        only = args.only.split(",")
        if set(only) - set(names):
            print(f"RESULT: ERROR unknown cases {sorted(set(only) - set(names))}")
            return 4
        todo = [c for c in CASES if c[0] in only]
    else:
        todo = [c for c in CASES if c[1] in want]
    OPTS["cache"] = Path(args.cache_dir) if args.cache_dir else None
    t_all = time.time()
    print("=" * 78)
    print("E6-KD SELECTION SMOKE (lane L-AM3, B-2) -- registration, conversion, scoring and selection, synthetic")
    print(f"{len(todo)} cases | sections {sorted({c[1] for c in todo})} | temp {TMP}")
    print("=" * 78)
    fill_cache()
    for name, _section, fn in todo:
        before = len(results)
        t0 = time.time()
        try:
            fn()
        except Exception as e:                               # noqa: BLE001 -- the case fails, the smoke goes on
            check(name, False, f"the case raised {type(e).__name__}: {str(e)[:300]}")
        if len(results) == before:
            check(name, False, "the case reported no check")
        print(f"[case] {name} ({time.time() - t0:.1f}s)", flush=True)
    bad = forbidden_names_created()
    check("no_forbidden_named_path_created", not bad, f"{len(bad)} path(s), e.g. {bad[:2]}")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        detail = detail.replace("\n", " | ")                  # one line per check: no detail line reads as a RESULT
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n[time] {time.time() - t_all:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
