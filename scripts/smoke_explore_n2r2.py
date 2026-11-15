#!/usr/bin/env python3
"""Smoke for N2R2 (addendum L-AM24-ADD-2; AM-29's E6-R2-M): src/explore/qat_explore.py n2r2_parent,
records_root_record and run_n2r2, scripts/explore_qat.py's N2R2 gates and scripts/explore_qat_blockb.py's N2R2 route
and card fix 1. Synthetic students, loaders, run folders and a synthetic records checkout (git, in the scratch folder;
the condition record made by the pinned writer, as scripts/smoke_am23_quant.py makes it); no dataset, no GPU, no TEST.

  * the constants against the records: E3R2_PARENT_OF_RECORD is DL-136's row and the addendum's da70d82f...; E6-R2's
    stage, parent arm and AM-23 rules file; the committed λ and α selections read through the pinned reader with the
    sha256s the launch command gives; the condition record's path of record, absent at this checkout;
  * flags and seeds: N2's flag set exactly (the same object), seed 42 only; N2's own unchanged;
  * the parent check: it accepts the listed hash and identity through E6-R2's loader (the same resolve_parent record
    and AM-23 binding as E6-R2's own gates give) and refuses other hashes (N2's E2 parent included), a wrong identity
    (run ID, best.json, run_meta, telemetry, value), D's flag, another seed and an E2 folder; the pinned binding refuses
    a condition record that is missing, changed after its commit or of another sha256, a decision-log row that is not
    RECORDED or absent, another arm, a wrong records commit or folder, and this checkout as the records root (no record
    here); N2 refuses E3-R2's parent;
  * through the pinned run_qat on a synthetic student: all ten sites fire, TF32 off is in run_meta, the substitution is
    recorded and the pinned name restored, run_meta is E6-R2's; a real run refuses the change removed and injected
    loaders;
  * with the substitution removed, N2R2's first-step loss (and CE, Dice, gradient norm, batch, epoch state, recipe
    fields) equals E6-R2's path, the pinned loader and run_qat, on one synthetic batch; with it, the loss differs;
  * N2's launch gate passes on a calibrated copy of the parent E6-R2's loader built;
  * block B: an N2R2 run is accepted and runs the pinned scripts (convert refuses the 2-epoch smoke run, card fix 1
    recorded as not run), an unknown item and an N2R2 item whose run is not E6-R2 are refused; card fix 1's command
    passes the pinned evaluator's own CLI checks for N2 (stage E6-E2) and is refused for N2R2 (E6-R2 is not in
    src/eval/artifacts.py FQOFF_STAGES), in process and by the script itself; the record of a scored, a failed and a
    mismatched step;
  * the CLI: N2R2 refuses another seed, a missing records flag, another run ID, TEST and a data root that fails the
    TRAIN/VAL-only check, writing nothing; N2 refuses N2R2's flags; block B's CLI accepts N2R2 up to the guard.
Mutations: a permissive identity check, N2R2 on E6-E2's stage, E6-R2's binding skipped, run_n2r2 without the sites
and a block B without card fix 1 each fail a case.

    python -I scripts/smoke_explore_n2r2.py        (exit 0 PASS, 1 FAIL)
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
from torch.utils.data import DataLoader, TensorDataset              # noqa: E402

from src.explore import guard as G                                   # noqa: E402
from src.explore import qat_explore as X                             # noqa: E402
from src.explore.sites import SITE_NAMES, tf32_flags                 # noqa: E402
from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402
from src.explore.synthetic import random_student                    # noqa: E402
from src.quant import e6x_condition as XC                            # noqa: E402
from src.quant import qat as Q                                       # noqa: E402
from src.quant.prepare import prepare_qat_model as PINNED_PREPARE   # noqa: E402

LAM_REL, ALPHA_REL, DL_REL = ("reports/derived/lambda_selection.json", "reports/derived/alpha_selection.json",
                              "docs/DECISION_LOG.md")
COND_REL = XC.RECORD_REL.format(arm="R2")
# The λ and α selections of record: whole-file sha256 of the committed files (the same blobs at fe93a24 and at master
# aa8cac2), the values the launch command passes.
LAM_SHA = "719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145"
ALPHA_SHA = "1399b897278db17a8876c724b89c8d8e7a7b42ffe1d9c4d67002578a889f2dd8"
REC = X.E3R2_PARENT_OF_RECORD
DL_ID = "DL-950"
N2_FLAGS = {"sites_on_input_grid", "tf32_off"}
_CACHE: dict = {}


def quiet(*_a, **_k):
    return None


def _sha(p) -> str:
    return G.sha256_file(p)


def _jl(rows) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _module(path: Path):
    """A script imported as a module: its top level only (the scripts used here run nothing on import)."""
    key = ("module", str(path))
    if key not in _CACHE:
        spec = importlib.util.spec_from_file_location(f"x_smoke_{path.stem}", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m                 # a dataclass in the module looks itself up there
        spec.loader.exec_module(m)
        _CACHE[key] = m
    return _CACHE[key]


def blockb():
    return _module(REPO / "scripts" / "explore_qat_blockb.py")


def evaluator():
    return _module(REPO / "scripts" / "evaluate_model.py")


def _git(repo: Path, *args: str) -> str:
    """git in a synthetic records checkout, as scripts/smoke_am23_quant.py runs it: a fixed identity and date, no global
    or system configuration."""
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env.update({"GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
                "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
                "GIT_AUTHOR_DATE": "2026-11-15T00:00:00Z", "GIT_COMMITTER_DATE": "2026-11-15T00:00:00Z",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"})
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


# ------------------------------------------------------------------ fixtures
def student_state(seed: int = 7) -> dict:
    key = ("state", seed)
    if key not in _CACHE:
        _CACHE[key] = {k: v.detach().clone() for k, v in random_student(seed=seed).state_dict().items()}
    return _CACHE[key]


def e3r2_meta(**over) -> dict:
    """E3-R2's run_meta row with the fields AM-23 item 4's binding reads (scripts/smoke_am23_quant.py e3_meta)."""
    row = {"event": "run_meta", "stage": "E3", "mode": "real", "seed": 42,
           "terms": {"logit_kd": True, "cwd_feat": True, "cwd_logit": True}, "lambda_logit": 0.5, "alpha_cwd": 25.0,
           "max_iters": 80000, "num_classes": 116, "poly_horizon": 80000, "teacher_mock": False, "arm": "E3-R2",
           "descriptive": True, "parent_of_e4_e7": False, "records_commit": "b" * 40,
           "target_construction": {"K": 2, "views": 2}}
    row.update(over)
    return row


def e3r2_run(base: Path, *, name: str = REC["run_id"], meta: dict | None = None) -> Path:
    """A fixture E3-R2 run folder: e3_run_meta.jsonl, e3_telemetry.jsonl ending in run_end, best.json (DL-136's value)
    and a checkpoint holding a full random student (stage E3), so the strict E3 loader builds it."""
    v, ck = REC["best_val"], REC["checkpoint"]
    d = base / name
    d.mkdir(parents=True)
    (d / "e3_run_meta.jsonl").write_text(_jl([meta or e3r2_meta()]), encoding="utf-8")
    end = {"event": "run_end", "iter": 80000, "best_val_miou_all_class": v, "best_ckpt": ck, "checks_passed": True,
           "wall_clock_end": 1792665391.5086315}
    (d / "e3_telemetry.jsonl").write_text(_jl([{"event": "val", "iter": 80000, "all_class_miou": v}, end]),
                                          encoding="utf-8")
    (d / "best.json").write_text(json.dumps({"best_ckpt": f"/workspace/kd_runs/{name}/{ck}",
                                             "best_val_miou_all_class": v}, indent=2), encoding="utf-8")
    torch.save({"stage": "E3", "model_state_dict": student_state(), "num_classes": 116, "best_val_miou_all_class": v,
                "arm": "E3-R2", "target_construction": {"K": 2, "views": 2}}, d / ck)
    return d


def e2_folder(base: Path) -> Path:
    """An E2-layout folder (e2_ records, an E2 checkpoint name) named as E3-R2's run."""
    d = base / REC["run_id"]
    d.mkdir(parents=True)
    ck = "e2_student_best_iter64000.pt"
    (d / "e2_run_meta.jsonl").write_text(_jl([{"event": "run_meta", "stage": "E2", "mode": "real", "seed": 42}]))
    (d / "e2_telemetry.jsonl").write_text(_jl([{"event": "run_end", "iter": 80000, "checks_passed": True}]))
    (d / "best.json").write_text(json.dumps({"best_ckpt": ck, "best_val_miou_all_class": 0.38}))
    (d / ck).write_bytes(b"e2 fixture checkpoint")
    return d


def table_of(d: Path, **over) -> dict:
    """E3R2_PARENT_OF_RECORD with a fixture folder's hashes (the real values name files that exist only on the laptop
    and a pod)."""
    t = {**REC, "sha256": _sha(d / REC["checkpoint"]), "best_json_sha256": _sha(d / "best.json"),
         "run_meta_sha256": _sha(d / "e3_run_meta.jsonl"), "telemetry_sha256": _sha(d / "e3_telemetry.jsonl")}
    t.update(over)
    return t


def selection(rel: str, sha: str) -> dict:
    doc = json.loads((REPO / rel).read_bytes().decode("utf-8"))
    return {"path": rel, "sha256": sha, "format": doc["format"], "winner": doc["winner"]}


def condition_record(d: Path) -> dict:
    """AM-23 item 3(b)'s record of a fixture run, by the pinned writer's build_record from the folder's own files (as
    scripts/smoke_am23_quant.py condition() makes it)."""
    meta_data, tel_data = (d / XC.META_NAME).read_bytes(), (d / XC.TELEMETRY_NAME).read_bytes()
    best_doc = json.loads((d / XC.BEST_JSON_NAME).read_text(encoding="utf-8"))
    ck = d / REC["checkpoint"]
    best = {"value": best_doc["best_val_miou_all_class"], "best_json_sha256": _sha(d / XC.BEST_JSON_NAME),
            "checkpoint": REC["checkpoint"], "checkpoint_sha256": _sha(ck), "checkpoint_bytes": ck.stat().st_size}
    end = json.loads(tel_data.decode("utf-8").splitlines()[-1])
    return XC.build_record(arm="R2", spec=SimpleNamespace(run_stage="E3-R2"), run_id=d.name, best=best,
                           meta_data=meta_data, tel_data=tel_data, end=end, lam=selection(LAM_REL, LAM_SHA),
                           alpha=selection(ALPHA_REL, ALPHA_SHA),
                           teacher={"path": "reports/arms/arm_teacher_record_R2.json", "sha256": "8" * 64,
                                    "checkpoint_sha256": "9" * 64},
                           launch={"path": "reports/kd_launch_log.jsonl", "sha256": "a" * 64},
                           tool={"script": "scripts/e6x_condition.py", "module": "src/quant/e6x_condition.py",
                                 "commit": "6" * 40, "clean_at_commit": True})


def records_checkout(root: Path, d: Path, *, record: bool = True, status: str = "RECORDED (synthetic)",
                     after: bytes | None = None) -> tuple[Path, str, str]:
    """A synthetic records checkout committed once: the λ and α selections of record (their committed bytes), E3-R2's
    condition record at its path of record and a decision log whose DL-950 carries the record's text. `record` False
    leaves the record out; `after` rewrites it after the commit. Returns (root, HEAD, the record's sha256)."""
    root.mkdir(parents=True)
    for rel in (LAM_REL, ALPHA_REL):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((REPO / rel).read_bytes())
    rec = condition_record(d)
    data = XC.record_bytes(rec)
    rec_sha = hashlib.sha256(data).hexdigest()
    files = [LAM_REL, ALPHA_REL, DL_REL]
    if record:
        (root / COND_REL).parent.mkdir(parents=True, exist_ok=True)
        XC.write_record(root / COND_REL, data)
        files.append(COND_REL)
    (root / DL_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / DL_REL).write_text("# Decision log (synthetic)\n\n| ID | Status | Entry |\n|---|---|---|\n"
                               f"| {DL_ID} | {status} | {XC.dl_text(rec, rec_sha)} |\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "--", *files)
    _git(root, "commit", "-q", "-m", "synthetic records")
    if after is not None:
        (root / COND_REL).write_bytes(after)
    return root, _git(root, "rev-parse", "HEAD").strip(), rec_sha


def fixtures() -> dict:
    """Built once per process; the mutation passes reuse them."""
    if "fx" in _CACHE:
        return _CACHE["fx"]
    base = scratch_dir("x_n2r2_fx_")
    fx = {"base": base, "d": e3r2_run(base / "runs")}
    fx["root"], fx["head"], fx["rec_sha"] = records_checkout(base / "recs", fx["d"])
    fx["root_norec"], fx["head_norec"], _ = records_checkout(base / "recs_norec", fx["d"], record=False)
    fx["root_decided"], fx["head_decided"], _ = records_checkout(base / "recs_decided", fx["d"],
                                                                 status="DECIDED (synthetic)")
    fx["root_changed"], fx["head_changed"], _ = records_checkout(base / "recs_changed", fx["d"], after=b"{}\n")
    fx["d_a1"] = e3r2_run(base / "runs_a1", name="e3-r2_s42_lambda0.5_alpha25_80k_a1")
    fx["d_arm"] = e3r2_run(base / "runs_arm", meta=e3r2_meta(arm="E3-R1"))
    fx["d_e2"] = e2_folder(base / "runs_e2")
    fx["repo_head"] = _git(REPO, "rev-parse", "HEAD").strip()
    _CACHE["fx"] = fx
    return fx


def n2r2_kw(fx: dict, **over) -> dict:
    kw = {"seed": 42, "expect_source_sha256": _sha(fx["d"] / REC["checkpoint"]), "records_root": str(fx["root"]),
          "records_commit": fx["head"], "lambda_selection": LAM_REL, "lambda_selection_sha256": LAM_SHA,
          "alpha_selection": ALPHA_REL, "alpha_selection_sha256": ALPHA_SHA, "condition_record": COND_REL,
          "condition_sha256": fx["rec_sha"], "condition_dl": DL_ID}
    kw.update(over)
    return kw


def one_batch(n: int = 2, size: int = 64, seed: int = 0):
    """One synthetic TRAIN batch (and the same images as the VAL loader)."""
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randn(n, 3, size, size, generator=g)
    masks = torch.randint(0, 116, (n, size, size), generator=g)
    masks[:, :4, :] = 255
    ds = TensorDataset(imgs, masks)
    return (DataLoader(ds, batch_size=n, shuffle=False, drop_last=True),
            DataLoader(ds, batch_size=n, shuffle=False, drop_last=False))


def rows_of(out: Path) -> list[dict]:
    return [json.loads(ln) for ln in (out / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()]


def first(rows: list[dict], event: str) -> dict:
    return next((r for r in rows if r.get("event") == event), {})


def tf32_restore(before: dict) -> None:
    torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]


def smoke_qat(model, out: Path, *, stage: str = "e6-r2", parent=None, extra=None, steps: int = 1, meta=None):
    """The pinned run_qat in smoke mode on one batch (E6-R2's path when stage is e6-r2)."""
    return Q.run_qat(stage=stage, mode="smoke", model=model, source_meta=meta or {"sha256": "0" * 64}, out_dir=out,
                     seed=42, clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0, loaders=one_batch(),
                     parent=parent, extra_meta=extra, max_steps=steps, max_val_batches=1, log=quiet)


def n2r2_smoke(model, out: Path, *, sites: bool = True, parent=None, extra=None, steps: int = 1, meta=None):
    return X.run_n2r2(mode="smoke", model=model, source_meta=meta or {"sha256": "0" * 64}, out_dir=out, seed=42,
                      clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0, parent=parent,
                      extra_meta=extra, sites=sites, loaders=one_batch(), max_steps=steps, max_val_batches=1,
                      log=quiet)


class Done:
    """What subprocess.run returns, for the injected runners."""

    def __init__(self, argv, code: int, out: str = "", err: str = ""):
        self.args, self.returncode, self.stdout, self.stderr = argv, code, out, err


def evaluator_cli_runner(argv, **_kw):
    """The pinned evaluator's own CLI checks (build_parser, validate_cli_args) on block B's command, in process; when
    they pass, a summary.json with the fields block B reads, as the evaluator's artifact holds them (synthetic: no
    dataset here, so no score is made)."""
    em = evaluator()
    if argv[1:3] != ["-B", "scripts/evaluate_model.py"]:
        return Done(argv, 9, "", f"not the evaluator: {argv[1:3]}")
    a = em.build_parser().parse_args(argv[3:])
    try:
        em.validate_cli_args(a)
    except em.CliError as e:
        return Done(argv, 1, "", f"CliError: {e}")
    out = Path(a.out_dir)
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({
        "run": {"stage": a.stage, "precision": a.precision, "checkpoint_sha256": a.expected_checkpoint_sha256,
                "eval_runtime": {"torch_num_threads": 8}},
        "dataset": {"split": a.split, "expected_rows": 846, "actual_rows": 846},
        "dataset_level": {"all_class_miou": 0.3412}}), encoding="utf-8")
    return Done(argv, 0, f"artifact written: {out}")


def script_runner(argv, **kw):
    """The script itself, with no data root in its environment."""
    env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
    return subprocess.run(argv, env=env, **kw)


def selection_doc(run: Path, epoch: int) -> dict:
    """A synthetic qat_selection.json naming `epoch` of a smoke run (no rules_sha256: no rules selected it)."""
    from src.quant.qat_select import EPOCH_SELECTION_FORMAT
    ends = {r["epoch"]: r for r in rows_of(run) if r.get("event") == "epoch_end"}
    return {"format": EPOCH_SELECTION_FORMAT, "run_id": run.name,
            "winner": {"epoch": epoch, "checkpoint": f"epoch_ckpts/e{epoch:02d}.pt",
                       "checkpoint_sha256": ends[epoch]["checkpoint_sha256"], "value": 0.35}}


def blockb_main(argv) -> tuple[int, str]:
    """Block B's main in process with the guard's data-root step given a fixture record (this session has no data
    root; the guard smoke covers that step), stdout captured."""
    iso = {"checked": True, "data_root": "<synthetic>", "counts": {"train": 5367, "val": 846},
           "reason": "smoke: the TRAIN/VAL-only check is smoke_explore_guard's"}
    buf = io.StringIO()
    with patched(G, "check_data_root", lambda records_only=False: dict(iso)), contextlib.redirect_stdout(buf):
        code = blockb().main(argv)
    return code, buf.getvalue()


# ------------------------------------------------------------------ the cases
def run_cases(verbose: bool = False, full: bool = True) -> Cases:
    c = Cases("SMOKE_EXPLORE_N2R2", verbose=verbose)
    fx = fixtures()
    work = scratch_dir("x_n2r2_run_")
    d, d_ck = fx["d"], fx["d"] / REC["checkpoint"]
    st = Q.resolve_quant_stage("e6-r2")
    # ---- 1. the constants against the records
    dl = (REPO / DL_REL).read_text(encoding="utf-8")
    row136 = next((ln for ln in dl.splitlines() if ln.startswith("| DL-136 |")), "")
    c.check("E3-R2 seed 42's checkpoint sha256 is the addendum's da70d82f...",
            REC["sha256"] == "da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4")
    vals = [REC[k] for k in ("run_id", "checkpoint", "sha256", "best_json_sha256", "run_meta_sha256",
                             "telemetry_sha256")] + [repr(REC["best_val"])]
    c.check("the parent record is DL-136's: run ID, checkpoint, its sha256, best.json, run_meta, telemetry, value",
            bool(row136) and all(v in row136 for v in vals) and "E6-R2 condition (AM-23 item 3(b)): yes" in row136,
            str([v for v in vals if v not in row136]))
    from src.quant import qat_select as S
    c.check("N2R2's stage is E6-R2: source E3, parent arm E3-R2, an AM-23 stage, scripts/run_e6_r2.py's STAGE",
            X.N2R2_STAGE == "e6-r2" and st["name"] == "E6-R2" and st["source_stage"] == "E3"
            and st.get("parent_arm") == "E3-R2" and "E6-R2" in Q.AM23_STAGES
            and _module(REPO / "scripts" / "run_e6_r2.py").STAGE == X.N2R2_STAGE)
    c.check("E6-R2's epoch rules are AM-23's file (qat_select.rules_path_for_stage)",
            Path(S.rules_path_for_stage("E6-R2")).resolve()
            == (REPO / "configs" / "qat_selection_rules_am23.json").resolve())
    lam = c.passes("the λ selection of record reads through the pinned reader with the command's sha256",
                   Q.read_tracked_selection, LAM_REL, LAM_SHA, Q.LAMBDA_SELECTION_FORMAT, "lambda", REPO)
    alpha = c.passes("the α selection of record reads through the pinned reader with the command's sha256",
                     Q.read_tracked_selection, ALPHA_REL, ALPHA_SHA, Q.ALPHA_SELECTION_FORMAT, "alpha", REPO)
    c.check("their winners: λ 0.5; α 25.0 with best_val 0.36337557435035706 (AM-23 item 3(b)'s threshold)",
            bool(lam) and bool(alpha) and lam["winner"]["lambda"] == 0.5 and alpha["winner"]["alpha"] == 25.0
            and repr(alpha["winner"]["best_val"]) == XC.THRESHOLD)
    c.check("the condition record's path of record is reports/arms/e6x_condition_R2.json; it is absent here",
            COND_REL == "reports/arms/e6x_condition_R2.json" and not (REPO / COND_REL).exists())
    c.check("N2R2's run ID is x-n2r2-s42 and its records flags are E6-R2's flag names",
            X.N2R2_RUN_ID == "x-n2r2-s42"
            and {"lambda_selection", "alpha_selection", "condition_record", "condition_sha256",
                 "condition_dl"} <= set(X.N2R2_RECORD_FLAGS))
    # ---- 2. flags and seeds
    c.check("N2R2's flag set is N2's (the same object)", X.ITEM_FLAGS["N2R2"] is X.ITEM_FLAGS["N2"])
    c.check("N2R2 with N2's two flags at seed 42 accepted", X.item_flag_error("N2R2", N2_FLAGS, 42) is None)
    c.check("N2R2 with one flag, none, an added flag or N5's refused (item_flags)",
            all((X.item_flag_error("N2R2", f, 42) or ("",))[0] == "item_flags"
                for f in ({"tf32_off"}, {"sites_on_input_grid"}, set(), N2_FLAGS | {"kd_logit"}, {"fp32_head"})))
    c.check("N2R2 at seed 43 and 44 refused (item_seed)",
            all((X.item_flag_error("N2R2", N2_FLAGS, s) or ("",))[0] == "item_seed" for s in (43, 44)))
    c.check("N2 unchanged: its two flags at seed 42 accepted, seed 43 refused",
            X.item_flag_error("N2", N2_FLAGS, 42) is None
            and (X.item_flag_error("N2", N2_FLAGS, 43) or ("",))[0] == "item_seed")
    # ---- 3. the parent check
    got = parent_e = binding_e = None
    with patched(X, "E3R2_PARENT_OF_RECORD", table_of(d)):
        got = c.passes("the parent check accepts the listed hash and identity, through E6-R2's loader",
                       X.n2r2_parent, d, **n2r2_kw(fx))
        b = (got or {}).get("am23_binding") or {}
        c.check("... stage E6-R2, binding arm E3-R2, condition yes from DL-950, the records commit recorded",
                bool(got) and got["stage"]["name"] == "E6-R2" and b.get("arm") == "E3-R2"
                and (b.get("condition") or {}).get("verdict") == "yes" and (b.get("condition") or {}).get("dl") == DL_ID
                and got["records_root"]["commit"] == fx["head"]
                and got["parent"]["of_record"]["run_id"] == REC["run_id"])
        parent_e = Q.resolve_parent(st, d, _sha(d_ck), 42)
        lam_r = Q.read_tracked_selection(LAM_REL, LAM_SHA, Q.LAMBDA_SELECTION_FORMAT, "lambda", fx["root"])
        alpha_r = Q.read_tracked_selection(ALPHA_REL, ALPHA_SHA, Q.ALPHA_SELECTION_FORMAT, "alpha", fx["root"])
        binding_e = Q.am23_parent_binding(st, parent_e, SimpleNamespace(seed=42, condition_record=COND_REL,
                                                                        condition_sha256=fx["rec_sha"],
                                                                        condition_dl=DL_ID),
                                          (lam_r, alpha_r), repo_root=fx["root"])
        c.check("N2R2's parent record and binding are E6-R2's own (resolve_parent, am23_parent_binding)",
                bool(got) and {k: v for k, v in got["parent"].items() if k != "of_record"} == parent_e
                and got["am23_binding"] == binding_e)
        for case, code, over in (
                ("another hash refused", "parent_not_of_record", {"expect_source_sha256": "0" * 64}),
                ("N2's E2 parent (73dbc32d...) refused", "parent_not_of_record",
                 {"expect_source_sha256": X.E2_PARENTS_OF_RECORD[42]["sha256"]}),
                ("D's --parent-sha256 refused", "parent_sha256_not_d", {"parent_sha256": _sha(d_ck)}),
                ("seed 43 refused", "item_seed", {"seed": 43}),
                ("a records commit that is not the checkout's HEAD refused", "records_commit_mismatch",
                 {"records_commit": "0" * 40}),
                ("a malformed records commit refused", "records_commit_format", {"records_commit": "abc"}),
                ("a records folder that is not a checkout refused", "records_root_not_checkout",
                 {"records_root": str(fx["base"] / "runs")}),
                ("a missing records folder refused", "records_root_missing",
                 {"records_root": str(fx["base"] / "absent")}),
                ("the λ selection of another sha256 refused (pinned reader)", "selection_sha256_mismatch",
                 {"lambda_selection_sha256": "0" * 64}),
                ("a condition record of another sha256 refused (pinned binding)", "condition_sha256_mismatch",
                 {"condition_sha256": "0" * 64}),
                ("a decision-log row that does not exist refused (pinned binding)", "condition_dl_row",
                 {"condition_dl": "DL-951"}),
                ("an uncommitted condition record refused: none in the checkout (pinned binding)",
                 "condition_missing_file", {"records_root": str(fx["root_norec"]), "records_commit": fx["head_norec"]}),
                ("a decision-log row that is not RECORDED refused (pinned binding)", "condition_dl_row",
                 {"records_root": str(fx["root_decided"]), "records_commit": fx["head_decided"]}),
                ("a condition record changed after its commit refused (pinned binding)",
                 "condition_changed_since_head",
                 {"records_root": str(fx["root_changed"]), "records_commit": fx["head_changed"]}),
                ("this checkout as the records root refused: no condition record at this HEAD",
                 "condition_missing_file", {"records_root": str(REPO), "records_commit": fx["repo_head"]})):
            c.refuses(f"N2R2 parent: {case}", code, X.n2r2_parent, d, **n2r2_kw(fx, **over))
    for case, folder, table in (
            ("another run's folder (attempt a1) under the record's hash", fx["d_a1"],
             table_of(fx["d_a1"], run_id=REC["run_id"])),
            ("a best.json other than DL-136's", d, table_of(d, best_json_sha256="0" * 64)),
            ("a run_meta other than DL-136's", d, table_of(d, run_meta_sha256="0" * 64)),
            ("a telemetry other than DL-136's", d, table_of(d, telemetry_sha256="0" * 64)),
            ("a best value other than DL-136's", d, table_of(d, best_val=0.38))):
        with patched(X, "E3R2_PARENT_OF_RECORD", table):
            c.refuses(f"N2R2 parent: {case} refused", "parent_identity", X.n2r2_parent, folder,
                      **n2r2_kw(fx, expect_source_sha256=table["sha256"]))
    with patched(X, "E3R2_PARENT_OF_RECORD", table_of(fx["d_arm"])):
        c.refuses("N2R2 parent: a run of arm E3-R1 refused (pinned binding)", "parent_arm_mismatch", X.n2r2_parent,
                  fx["d_arm"], **n2r2_kw(fx, expect_source_sha256=_sha(fx["d_arm"] / REC["checkpoint"])))
    e2ck = fx["d_e2"] / "e2_student_best_iter64000.pt"
    with patched(X, "E3R2_PARENT_OF_RECORD", {**REC, "sha256": _sha(e2ck)}):
        c.refuses("N2R2 parent: an E2 folder refused by E6-R2's loader (no e3 records)", "parent_records_missing",
                  X.n2r2_parent, fx["d_e2"], **n2r2_kw(fx, expect_source_sha256=_sha(e2ck)))
    c.refuses("N2R2 parent with the real table: the fixture's hash refused", "parent_not_of_record", X.n2r2_parent, d,
              **n2r2_kw(fx))
    c.refuses("N2 refuses E3-R2's parent: da70d82f... is not E2 seed 42's", "parent_not_of_record",
              X.exploratory_parent, "N2", d, seed=42, expect_source_sha256=REC["sha256"])
    with patched(X, "E2_PARENTS_OF_RECORD", {**X.E2_PARENTS_OF_RECORD,
                                             42: {**X.E2_PARENTS_OF_RECORD[42], "sha256": _sha(d_ck)}}):
        c.refuses("N2 refuses an E3-R2 folder even under its hash: E6-E2's loader needs e2 records",
                  "parent_records_missing", X.exploratory_parent, "N2", d, seed=42, expect_source_sha256=_sha(d_ck))
    # ---- 4. through the pinned run_qat
    item = work / "x-n2r2-s42"
    out_s = item / "x-n2r2-s42"
    item.mkdir()
    res = None
    if got:
        model, meta = Q.load_source(got["stage"], got["parent"])
        before = tf32_flags()
        try:
            res = c.passes("run_n2r2 runs the pinned run_qat on E6-R2's parent for 2 smoke steps", n2r2_smoke, model,
                           out_s, parent=got["parent"], extra={"am23_parent": got["am23_binding"], "x_item": "N2R2"},
                           steps=2, meta=meta)
        finally:
            tf32_restore(before)
    _summary, sub, matcher = res or (None, None, None)
    c.check("all ten sites were installed and fired", matcher is not None and tuple(matcher.calls) == SITE_NAMES
            and all(v > 0 for v in matcher.calls.values()), str(None if matcher is None else matcher.calls))
    m0 = rows_of(out_s)[0] if (out_s / Q.TELEMETRY_NAME).is_file() else {}
    c.check("run_meta records TF32 off (cuDNN and matmul), as read and as set",
            (m0.get("tf32") or {}).get("cudnn_allow_tf32") is False
            and (m0.get("tf32") or {}).get("cuda_matmul_allow_tf32") is False
            and (m0.get("x_tf32_set") or {}).get("cudnn_allow_tf32") is False, str(m0.get("tf32")))
    c.check("run_meta records the substitution: qat.py's prepare_qat_model, the file's sha256, item N2R2",
            sub is not None and m0.get("x_substitution") == sub and sub["pinned_file"] == "src/quant/qat.py"
            and sub["attribute"] == "prepare_qat_model" and sub["item"] == "N2R2"
            and sub["pinned_file_sha256"] == _sha(REPO / "src" / "quant" / "qat.py"))
    c.check("the pinned prepare_qat_model is restored afterwards", Q.prepare_qat_model is PINNED_PREPARE)
    c.check("run_meta is E6-R2's: stage, source stage, no projection, the parent's sha256, run ID and AM-23 binding",
            bool(got) and m0.get("stage") == "E6-R2" and m0.get("source_stage") == "E3"
            and m0.get("cwd_projection_loaded") is False
            and m0.get("source_checkpoint_sha256") == got["parent"]["checkpoint_sha256"]
            and (m0.get("parent") or {}).get("run_id") == REC["run_id"] and m0.get("x_item") == "N2R2"
            and json.dumps(m0.get("am23_parent"), sort_keys=True) == json.dumps(got["am23_binding"], sort_keys=True))
    before = tf32_flags()
    try:
        c.refuses("a real N2R2 run refuses the change removed", "n2r2_change_required", X.run_n2r2, mode="real",
                  model=random_student(seed=5), source_meta={"sha256": "0" * 64}, out_dir=work / "r1" / "x-n2r2-r",
                  seed=42, clip_norm=5.0, clip_source="clip_selection", device="cpu", num_workers=1, parent=None,
                  sites=False)
        c.refuses("a real N2R2 run refuses injected loaders (run_qat's own refusal)", "real_run_test_hooks",
                  X.run_n2r2, mode="real", model=random_student(seed=5), source_meta={"sha256": "0" * 64},
                  out_dir=work / "r2" / "x-n2r2-r", seed=42, clip_norm=5.0, clip_source="clip_selection",
                  device="cpu", num_workers=1, parent=None, loaders=one_batch())
    finally:
        tf32_restore(before)
    c.check("... and the pinned name is restored after the refusal", Q.prepare_qat_model is PINNED_PREPARE)
    # ---- 5. the first step: N2R2 with the substitution removed = E6-R2's path
    te = tr = tx = {}
    if got and parent_e is not None:
        model_e, meta_e = Q.load_source(st, parent_e)                 # E6-R2's own loader and parent record
        model_r, meta_r = Q.load_source(got["stage"], got["parent"])   # N2R2's
        c.check("the same source: both loads read one checkpoint into equal states",
                meta_e["sha256"] == meta_r["sha256"] == _sha(d_ck)
                and Q.state_digest(model_e) == Q.state_digest(model_r))
        out_e, out_r, out_x = work / "e6r2" / "x-e6r2-path", work / "nosites" / "x-n2r2-nosites", \
            work / "sites1" / "x-n2r2-sites1"
        smoke_qat(model_e, out_e, parent=parent_e, extra={"am23_parent": binding_e}, meta=meta_e)
        n2r2_smoke(model_r, out_r, sites=False, parent=got["parent"], extra={"am23_parent": got["am23_binding"]},
                   meta=meta_r)
        re_, rr_ = rows_of(out_e), rows_of(out_r)
        te, tr = first(re_, "train"), first(rr_, "train")
        c.check("with the substitution removed, N2R2's first-step loss equals E6-R2's path (one synthetic batch)",
                isinstance(te.get("loss"), float) and te.get("loss") == tr.get("loss"),
                f"{te.get('loss')!r} vs {tr.get('loss')!r}")
        c.check("... and its CE, Dice, gradient norm, lr and batch fingerprint, and the state after the step",
                all(te.get(k) == tr.get(k) for k in ("ce", "dice", "grad_norm", "lr", "batch_sha256"))
                and first(re_, "epoch_end").get("state_sha256") == first(rr_, "epoch_end").get("state_sha256"))
        keys = ("stage", "source_stage", "seed", "clip_norm", "epochs", "batch_size", "drop_last", "steps_per_epoch",
                "total_steps", "scheduler", "optimizer", "lr", "momentum", "weight_decay", "nesterov", "dampening",
                "bn_freeze_after_step", "obs_freeze_after_step", "objective", "class_weights_sha256", "engine",
                "qconfig", "observer_census", "fake_quant_modules", "never_observed_modules", "fused_convbn",
                "unfused_batchnorm", "tf32", "source_checkpoint_sha256", "cwd_projection_loaded", "am23_parent",
                "code_files_sha256")
        c.check("... and run_meta's recipe, census, TF32, source and binding fields",
                all(re_[0].get(k) == rr_[0].get(k) for k in keys),
                str([k for k in keys if re_[0].get(k) != rr_[0].get(k)]))
        before = tf32_flags()
        try:
            model_x, meta_x = Q.load_source(st, parent_e)
            n2r2_smoke(model_x, out_x, sites=True, parent=got["parent"], meta=meta_x)
        finally:
            tf32_restore(before)
        tx = first(rows_of(out_x), "train")
    c.check("with the substitution, the first-step loss differs: the comparison discriminates",
            isinstance(tx.get("loss"), float) and tx.get("loss") != te.get("loss"),
            f"{tx.get('loss')!r} vs {te.get('loss')!r}")
    if full and parent_e is not None:
        from src.explore.sites import calibrated_copy, require_site_set
        c.passes("N2's launch gate on N2R2's parent: a calibrated copy of the loaded E3-R2 student gives the ten sites",
                 require_site_set, calibrated_copy(Q.load_source(st, parent_e)[0]))
    # ---- 6. block B
    BB = blockb()
    (item / G.RUN_META_NAME).write_text(json.dumps({
        "exploratory": True, "item": "N2R2", "run_id": "x-n2r2-s42", "seeds": [42], "amendment": G.AMENDMENT,
        "precision_label": "int8 QNNPACK (as E6-R2)", "claim_line": G.CLAIM_LINE.format(seeds=42)}), encoding="utf-8")
    doc = c.passes("block B accepts an N2R2 run (run_item)", BB.run_item, out_s)
    c.check("... a registered item with card fix 1, its run of stage E6-R2",
            bool(doc) and doc["item"] == "N2R2" and "N2R2" in BB.REGISTERED_ITEMS and "N2R2" in BB.FQOFF_ITEMS
            and BB.ITEM_STAGE.get("N2R2") == "E6-R2" and (out_s / Q.TELEMETRY_NAME).is_file()
            and BB.run_stage(out_s) == "E6-R2")
    c.check("block B gives card fix 1 and the stage check to N2 and N2R2 only (C1, N1, D as at 60dc294)",
            tuple(BB.FQOFF_ITEMS) == ("N2", "N2R2") and all(i not in BB.FQOFF_ITEMS and i not in BB.ITEM_STAGE
                                                             for i in ("C1", "N1", "D", "N5")))
    unknown = work / "x-z9"
    (unknown / "r").mkdir(parents=True)
    (unknown / G.RUN_META_NAME).write_text(json.dumps({"exploratory": True, "item": "Z9"}), encoding="utf-8")
    c.refuses("block B refuses an unknown item (run_item)", "run_item", BB.run_item, unknown / "r")
    tel_sha = _sha(out_s / Q.TELEMETRY_NAME) if (out_s / Q.TELEMETRY_NAME).is_file() else "0" * 64
    bbo = work / "bbo" / "x-n2r2-bb"
    code, text = blockb_main(["--run-dir", str(out_s), "--expect-telemetry-sha256", tel_sha, "--host-label", "smoke",
                              "--run-id", "x-n2r2-bb", "--out-dir", str(bbo)])
    xb = json.loads((bbo / "x_blockb.json").read_text(encoding="utf-8")) if (bbo / "x_blockb.json").is_file() else {}
    steps = xb.get("steps") or [{}]
    c.check("block B on an N2R2 run runs the pinned scripts: convert, from the checkout, refuses the 2-epoch smoke run "
            "as incomplete (exit 3); card fix 1 recorded as not run; INCOMPLETE",
            code == 3 and xb.get("item") == "N2R2" and steps[0].get("step") == "convert"
            and steps[0].get("argv", ["", "", ""])[1:3] == ["scripts/qat_epoch_eval.py", "convert"]
            and steps[0].get("exit") == 3
            and (xb.get("selected_epoch_fake_quant_off_val") or {}).get("status") == "not run"
            and "RESULT: X BLOCK B INCOMPLETE" in text, text.strip()[-200:])
    bad_item = work / "bbs" / "x-n2r2-s42"
    (bad_item / "r").mkdir(parents=True)
    (bad_item / G.RUN_META_NAME).write_text(json.dumps({"exploratory": True, "item": "N2R2", "seeds": [42]}))
    (bad_item / "r" / Q.TELEMETRY_NAME).write_text(json.dumps({"event": "run_meta", "stage": "E6-E2"}) + "\n")
    bso = work / "bbs_o" / "x-n2r2-bbs"
    code2, text2 = blockb_main(["--run-dir", str(bad_item / "r"), "--expect-telemetry-sha256",
                                _sha(bad_item / "r" / Q.TELEMETRY_NAME), "--host-label", "smoke", "--run-id",
                                "x-n2r2-bbs", "--out-dir", str(bso)])
    c.check("block B refuses an N2R2 item whose run is stage E6-E2 (run_stage), before any step",
            code2 == 2 and "[run_stage]" in text2 and not (bso / "eval").exists()
            and not (bso / "x_blockb.json").exists(), text2.strip()[-200:])
    # card fix 1's step: the command, judged by the pinned evaluator's own CLI checks
    ev_r2 = work / "fq_eval_r2"
    ev_r2.mkdir()
    if (out_s / Q.TELEMETRY_NAME).is_file():
        (ev_r2 / "qat_selection.json").write_text(json.dumps(selection_doc(out_s, 2)), encoding="utf-8")
    r_r2 = c.passes("card fix 1 runs for N2R2's selection", BB.selected_epoch_vals, out_s, ev_r2, work / "fq_o_r2",
                    SimpleNamespace(expect_telemetry_sha256=tel_sha), stage="E6-R2", runner=evaluator_cli_runner) or {}
    fo = r_r2.get("fake_quant_off") or {}
    c.check("card fix 1 for N2R2: the pinned evaluator's CLI refuses stage E6-R2 (FQOFF_STAGES); recorded, not ok",
            r_r2.get("ok") is False and fo.get("status") == "failed" and fo.get("exit") == 1
            and "fqoff_stage_not_in_scope" in str(fo.get("message")), str(fo.get("message"))[:160])
    ends_s = {r["epoch"]: r for r in rows_of(out_s) if r.get("event") == "epoch_end"} if out_s.is_dir() else {}
    c.check("... the selected epoch's fake-quant VAL is recorded from its epoch_end row all the same",
            (r_r2.get("fake_quant") or {}).get("epoch") == 2 and 2 in ends_s
            and r_r2["fake_quant"]["value"] == ends_s[2]["fake_quant_val_all_class_miou"]
            and r_r2["fake_quant"]["telemetry"]["sha256"] == tel_sha)
    from src.eval.artifacts import FQOFF_STAGES
    c.check("the evaluator's stage list, read here: E6-E2 in FQOFF_STAGES, E6-R2 not",
            "E6-E2" in FQOFF_STAGES and "E6-R2" not in FQOFF_STAGES, str(FQOFF_STAGES))
    out_n2 = work / "n2run" / "x-n2-s42"
    smoke_qat(random_student(seed=4), out_n2, stage="e6-e2", steps=2)
    ev_n2 = work / "fq_eval_n2"
    ev_n2.mkdir()
    (ev_n2 / "qat_selection.json").write_text(json.dumps(selection_doc(out_n2, 2)), encoding="utf-8")
    tel_n2 = _sha(out_n2 / Q.TELEMETRY_NAME)
    r_n2 = c.passes("card fix 1 runs for N2's selection", BB.selected_epoch_vals, out_n2, ev_n2, work / "fq_o_n2",
                    SimpleNamespace(expect_telemetry_sha256=tel_n2), stage="E6-E2", runner=evaluator_cli_runner) or {}
    fo2 = r_n2.get("fake_quant_off") or {}
    c.check("card fix 1 for N2: the command passes the pinned evaluator's CLI checks; the value recorded with its "
            "summary's sha256, the checks all true",
            r_n2.get("ok") is True and fo2.get("status") == "scored" and fo2.get("value") == 0.3412
            and len((fo2.get("summary") or {}).get("sha256", "")) == 64
            and all((fo2.get("checks") or {"x": 0}).values()))
    a = evaluator().build_parser().parse_args(fo2.get("argv", ["-B", "x"])[2:]) if fo2.get("argv") else None
    sel_sha = _sha(ev_n2 / "qat_selection.json")
    win = selection_doc(out_n2, 2)["winner"]
    c.check("... the command: qat_fakequant_off on VAL, canvas, CPU, batch 1, provisional; the winner's eNN.pt and "
            "sha256; the selection's and the telemetry's sha256",
            a is not None and a.precision == "qat_fakequant_off" and a.split == "val" and a.protocol == "canvas"
            and a.device == "cpu" and a.batch_size == 1 and a.artifact_status == "provisional" and a.stage == "E6-E2"
            and Path(a.checkpoint) == out_n2 / "epoch_ckpts" / "e02.pt" and a.qat_run_dir == str(out_n2)
            and a.expected_checkpoint_sha256 == win["checkpoint_sha256"] and a.qat_selection_sha256 == sel_sha
            and a.qat_telemetry_sha256 == tel_n2 and a.max_samples is None and a.threads is None)

    def wrong_stage_runner(argv, **kw):
        p = evaluator_cli_runner(argv, **kw)
        out = Path(argv[argv.index("--out-dir") + 1]) / "summary.json"
        s = json.loads(out.read_text(encoding="utf-8"))
        s["run"]["stage"] = "E6"
        out.write_text(json.dumps(s), encoding="utf-8")
        return p
    r_w = BB.selected_epoch_vals(out_n2, ev_n2, work / "fq_o_n2w", SimpleNamespace(expect_telemetry_sha256=tel_n2),
                                 stage="E6-E2", runner=wrong_stage_runner)
    c.check("card fix 1: a summary of another stage is refused, not recorded as the value",
            r_w["ok"] is False and r_w["fake_quant_off"]["status"] == "refused"
            and r_w["fake_quant_off"]["checks"]["stage"] is False)
    r_f = BB.selected_epoch_vals(out_n2, ev_n2, work / "fq_o_n2f", SimpleNamespace(expect_telemetry_sha256=tel_n2),
                                 stage="E6-E2", runner=lambda argv, **kw: Done(argv, 1, "", "CliError: boom"))
    c.check("card fix 1: an evaluator that exits 1 is recorded as failed with its message",
            r_f["ok"] is False and r_f["fake_quant_off"]["status"] == "failed"
            and r_f["fake_quant_off"]["message"] == "CliError: boom")
    if full:
        r_s = BB.selected_epoch_vals(out_s, ev_r2, work / "fq_o_r2s", SimpleNamespace(expect_telemetry_sha256=tel_sha),
                                     stage="E6-R2", runner=script_runner)
        fs = r_s["fake_quant_off"]
        c.check("card fix 1 for N2R2 through the script itself: scripts/evaluate_model.py exits 1, "
                "[fqoff_stage_not_in_scope], writes nothing",
                fs.get("exit") == 1 and "fqoff_stage_not_in_scope" in str(fs.get("message"))
                and not Path(fs["out_dir"]).exists(), str(fs.get("message"))[:200])
        r_s2 = BB.selected_epoch_vals(out_n2, ev_n2, work / "fq_o_n2s", SimpleNamespace(expect_telemetry_sha256=tel_n2),
                                      stage="E6-E2", runner=script_runner)
        fs2 = r_s2["fake_quant_off"]
        c.check("card fix 1 for N2 through the script itself: past its CLI checks (no CliError), refused at the source "
                "binding on the synthetic selection (StageArtifactError on qat_selection.json; no dataset read)",
                fs2.get("exit") == 1 and str(fs2.get("message")).startswith("StageArtifactError:")
                and "qat_selection.json" in str(fs2.get("message")) and "CliError" not in str(fs2.get("message"))
                and not Path(fs2["out_dir"]).exists(), str(fs2.get("message"))[:200])
    # ---- 7. the CLI (subprocesses: no data root, no CUDA in this session)
    if full:
        py, cli = sys.executable, str(REPO / "scripts" / "explore_qat.py")
        env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
        (work / "clipdir").mkdir()
        common = ["--real-run", "--confirm-real-run", "--source-run-dir", str(d), "--expect-source-sha256",
                  REC["sha256"], "--clip-selection", str(work / "clipdir" / "clip.json"), "--clip-selection-sha256",
                  X.CLIP_SELECTION_SHA256, "--num-workers", "1", "--expect-head", "0" * 40, "--sites-on-input-grid",
                  "--tf32-off"]
        recs = ["--records-root", str(fx["root"]), "--records-commit", fx["head"], "--lambda-selection", LAM_REL,
                "--lambda-selection-sha256", LAM_SHA, "--alpha-selection", ALPHA_REL, "--alpha-selection-sha256",
                ALPHA_SHA, "--condition-record", COND_REL, "--condition-sha256", fx["rec_sha"], "--condition-dl",
                DL_ID]

        def cli_run(*extra):
            return subprocess.run([py, "-I", cli, *common, *extra], capture_output=True, text=True, env=env)
        r = cli_run("--item", "N2R2", "--seed", "43", *recs, "--run-id", "x-n2r2-s42",
                    "--out-dir", str(work / "o1" / "x-n2r2-s42"))
        c.check("CLI: N2R2 at seed 43 refused (exit 2)", r.returncode == 2 and "item_seed" in r.stdout)
        r = cli_run("--item", "N2R2", "--seed", "42", "--run-id", "x-n2r2-s42",
                    "--out-dir", str(work / "o2" / "x-n2r2-s42"))
        c.check("CLI: N2R2 without its records flags refused (exit 2), each named",
                r.returncode == 2 and "n2r2_records_required" in r.stdout and "--condition-record" in r.stdout
                and "--records-root" in r.stdout, r.stdout.strip()[-160:])
        r = cli_run("--item", "N2R2", "--seed", "42", *recs[:-2], "--run-id", "x-n2r2-s42",
                    "--out-dir", str(work / "o3" / "x-n2r2-s42"))
        c.check("CLI: N2R2 without --condition-dl refused (exit 2)", r.returncode == 2
                and "n2r2_records_required" in r.stdout and "--condition-dl" in r.stdout)
        r = cli_run("--item", "N2R2", "--seed", "42", *recs, "--run-id", "x-n2r2-s42-b",
                    "--out-dir", str(work / "o4" / "x-n2r2-s42-b"))
        c.check("CLI: N2R2 under another run ID refused (exit 2)", r.returncode == 2 and "n2r2_run_id" in r.stdout)
        r = cli_run("--item", "N2", "--seed", "42", "--records-root", str(fx["root"]), "--run-id", "x-n2-s42",
                    "--out-dir", str(work / "o5" / "x-n2-s42"))
        c.check("CLI: N2 given N2R2's records flag refused (exit 2)",
                r.returncode == 2 and "n2r2_flags_not_applicable" in r.stdout)
        r = cli_run("--item", "N2R2", "--seed", "42", *recs, "--split", "TE" + "ST", "--run-id", "x-n2r2-s42",
                    "--out-dir", str(work / "o6" / "x-n2r2-s42"))
        c.check("CLI: N2R2 with --split TEST refused first, by name (exit 2)",
                r.returncode == 2 and "split_test" in r.stdout)
        r = cli_run("--item", "N2R2", "--seed", "42", *recs, "--run-id", "x-n2r2-s42",
                    "--out-dir", str(work / "o7" / "x-n2r2-s42"))
        c.check("CLI: N2R2 with a data root that fails the TRAIN/VAL-only check refused, nothing written",
                r.returncode == 2 and "data_root_not_trainval_only" in r.stdout
                and not (work / "o7" / "x-n2r2-s42").exists(), r.stdout.strip()[-160:])
        bbs = str(REPO / "scripts" / "explore_qat_blockb.py")
        r = subprocess.run([py, "-I", bbs, "--run-dir", str(out_s), "--expect-telemetry-sha256", tel_sha,
                            "--host-label", "smoke", "--run-id", "x-n2r2-bbc", "--out-dir",
                            str(work / "bbc" / "x-n2r2-bbc")], capture_output=True, text=True, env=env)
        c.check("block B CLI: an N2R2 run passes run_item and reaches the guard's data-root check (exit 2 there)",
                r.returncode == 2 and "data_root_not_trainval_only" in r.stdout and "run_item" not in r.stdout)
        r = subprocess.run([py, "-I", bbs, "--run-dir", str(unknown / "r"), "--expect-telemetry-sha256", "0" * 64,
                            "--host-label", "smoke", "--run-id", "x-z9-bb", "--out-dir", str(work / "bbz" / "x-z9-bb")],
                           capture_output=True, text=True, env=env)
        c.check("block B CLI: an unknown item refused (exit 2, run_item)", r.returncode == 2 and "run_item" in r.stdout)
    return c


def main() -> int:
    print("[smoke_explore_n2r2] cases")
    c = run_cases(verbose=True, full=True)
    print("[smoke_explore_n2r2] mutations")
    bb = blockb()
    orig_run = X.run_n2r2
    for name, mut in {
        "a permissive identity check": lambda: patched(X, "n2r2_identity_differ", lambda parent, rec: []),
        "N2R2 on E6-E2's stage": lambda: patched(X, "N2R2_STAGE", "e6-e2"),
        "E6-R2's binding skipped": lambda: patched(Q, "am23_parent_binding", lambda *a, **k: {"stage": "E3"}),
        "run_n2r2 without the sites": lambda: patched(X, "run_n2r2", lambda **kw: orig_run(**{**kw, "sites": False})),
        "block B without card fix 1": lambda: patched(bb, "FQOFF_ITEMS", ()),
    }.items():
        killed, failed = mutation_killed(lambda: run_cases(False, full=False), mut)
        c.check(f"mutation KILLED: {name}", killed, f"e.g. {failed[:2]}")
    return c.result()


if __name__ == "__main__":
    raise SystemExit(main())
