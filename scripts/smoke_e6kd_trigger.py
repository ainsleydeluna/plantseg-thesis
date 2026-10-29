#!/usr/bin/env python3
"""Lane L-AM3, B-1: the AM-3 trigger of E6-KD (src/quant/e6kd_trigger.py, scripts/e6kd_trigger.py) on synthetic inputs.

Every fixture is synthetic and lives in a temp directory outside the repository; no PlantSeg data, no checkpoint of
record:
- a data root of tiny images with the locked TRAIN/VAL counts (5,367 / 846) and no TEST surface (SL-1);
- E3 seed 42's parent run: a random-weight E3 checkpoint, best.json, its run_meta row and its run_end record;
- a temp git repository holding the λ and α selection files (the α winner names that run and that checkpoint) and the
  trigger records the launch gate reads (src/quant/e6kd_trigger.py reads them with git show HEAD:<rel>);
- E6 seed 42's run: a real QAT run of the trainer in mode "smoke" (kept raw with --cache-dir), re-labelled mode
  "real" with its clip_selection, its λ/α selections and its parent, as scripts/smoke_qat_selection.py re-labels;
  a second run made non-finite at step 1, so that every epoch is excluded;
- block B's eval directories, fabricated as scripts/smoke_qat_selection.py fabricates them (summaries holding chosen
  values, every hash consistent) and selected by src/quant/qat_select.py; for the end-to-end case, epoch 1 converted
  for real (scripts/qat_epoch_eval.py convert-epoch) and scored by the evaluator (Q2's driver: git calls stubbed,
  VAL capped at E2E_SAMPLES rows);
- trigger eval dirs written by `score`, or fabricated with chosen values.

Sections: inputs (steps 1-8 of the module), outputs (the trigger eval dir and --out), d1 (the decision), artifacts
(the two summaries), gate (B-2's launch gate), e2e (score, then decide, on the real evaluator). --only runs named cases
(scripts/smoke_e6kd_mutations.py). No check name, file or folder holds the token SL-1 forbids; the name checks use
paths that are never created.

Ends with one RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
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

from scripts.synthetic_ptq_fixtures import make_e3_checkpoint  # noqa: E402
from scripts.synthetic_qat_fixtures import (confident_student, evaluator_driver, make_qat_run, make_tree,  # noqa: E402
                                            safe_tmpdir)

TMP = safe_tmpdir("smoke_e6kd_trigger_")
ROOT = TMP / "data"
make_tree(ROOT, n_train=5367, n_val=846, seed=5)            # the locked counts (configs/data.py); no TEST surface
DIGEST = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"   # the student image (LB7)
os.environ["PLANTSEG_DATA_ROOT"] = str(ROOT)                 # before configs/data.py is first imported
os.environ["PLANTSEG_IMAGE_DIGEST"] = DIGEST                 # as the docker lines set it; the evaluator records it
os.environ["SMOKE_REPO"] = str(REPO)

import numpy as np  # noqa: E402

import scripts.e6kd_trigger as CLI  # noqa: E402
import scripts.qat_epoch_eval as QEE  # noqa: E402
from configs.data import DATA  # noqa: E402
from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME  # noqa: E402
from src.quant import e6kd_trigger as T  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_select as S  # noqa: E402
from src.quant.stages import resolve_quant_stage  # noqa: E402

results: list[tuple[str, bool, str]] = []
SECTIONS = ("inputs", "outputs", "d1", "artifacts", "gate", "e2e")
HOST = "cloud-smoke"
GT_SUM = 159_279_104
METRIC = "b" * 64
E3_RUN_ID = "e3_s42_lambda0.5_alpha25_80k_a1"
E3_CKPT = "e3_student_best_iter76000.pt"
E3_BEST = 0.36337557435035706              # best.json's value: recorded, never used by the trigger
LAM_WINNER = {"lambda": 0.5, "run_id": "e2_s42_lambda0.5_80k_a1"}
CLIP_SHA = "c" * 64                         # E6's clip_selection.json as its run_meta records it (winner 1.0)
E6_WINNER, E6_VALUE = 7, 0.39                # block B's winner epoch and value in the fabricated eval dirs
E2E_SAMPLES = "8"                            # the VAL rows of the end-to-end evaluator runs (driver cap)
E6_RAW, E6_NAN1_RAW = "e6_s42_clip1.0", "e6_s42_nan1"
DRIVER = evaluator_driver(TMP / "evaluator_driver.py")
FAILING = TMP / "failing_evaluator.py"
FAILING.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
SELREPO = TMP / "selrepo"
OPTS = {"cache": None}
FX: dict = {}
TRIGGER_KEYS = sorted([
    "format", "lane", "rule", "seed", "e3_val", "e6_val", "e3_val_decimal", "e6_val_decimal", "drop", "drop_pp",
    "drop_pp_float", "threshold", "threshold_decimal", "threshold_unit", "decision_rule", "arithmetic", "triggered",
    "consequence", "descriptive", "holm_family", "e3", "e6", "selections", "same_code_path", "input_sha256s",
    "trigger_eval", "git_head", "code_clean_at_head", "code_files_sha256", "host_label", "cpu_model", "smoke_inputs",
    "rule_trace"])
E3_KEYS = sorted(["run_id", "checkpoint", "checkpoint_sha256", "best_json_sha256", "run_meta_sha256",
                  "telemetry_sha256", "best_json_value_not_used", "summary", "eval_run_id", "evaluator_command"])
E6_KEYS = sorted(["run_id", "telemetry_sha256", "clip_norm", "clip_selection_sha256", "qat_selection", "rules_sha256",
                  "epoch", "checkpoint_sha256", "artifact_of_record", "provenance", "block_b_summary",
                  "selection_value", "excluded_epochs", "summary", "eval_run_id", "evaluator_command",
                  "rescore_equals_selection"])
INPUT_KEYS = sorted(["e3_checkpoint", "e3_best_json", "e3_run_meta", "e3_telemetry", "e6_telemetry",
                     "e6_clip_selection", "e6_qat_selection", "qat_selection_rules", "e6_epoch_checkpoint",
                     "e6_artifact_of_record", "e6_provenance", "e6_block_b_summary", "lambda_selection",
                     "alpha_selection", "e3_trigger_summary", "e6_trigger_summary", "trigger_eval"])


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def sha_file(p) -> str:
    return Q.sha256_file(p)


def code_of(fn, *a, **kw) -> str | None:
    """fn's refusal or incomplete code; None when it returns; "raised:<type>" for any other exception."""
    try:
        fn(*a, **kw)
    except (Q.QATRefused, A.QATIncomplete) as e:
        return e.code
    except Exception as e:                                      # noqa: BLE001 -- reported in the check's detail
        return f"raised:{type(e).__name__}: {str(e)[:200]}"
    return None


@contextlib.contextmanager
def patched_attr(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def patched_env(**values):
    before = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def probe_recorder(path: Path):
    """os.stat, os.lstat, os.listdir, os.scandir and Path.iterdir record every call on `path` (or below it) while the
    block runs, and are restored after it: a name SL-1 forbids is refused before it is probed or listed."""
    root = os.path.abspath(path)
    seen: list[str] = []
    real = (os.stat, os.lstat, os.listdir, os.scandir, Path.iterdir)

    def hit(p) -> None:
        try:
            a = os.path.abspath(os.fsdecode(p))
        except TypeError:                                       # a file descriptor
            return
        if a == root or a.startswith(root + os.sep):
            seen.append(a)

    def stat(p, *a, **kw):
        hit(p)
        return real[0](p, *a, **kw)

    def lstat(p, *a, **kw):
        hit(p)
        return real[1](p, *a, **kw)

    def listdir(p=".", *a, **kw):
        hit(p)
        return real[2](p, *a, **kw)

    def scandir(p=".", *a, **kw):
        hit(p)
        return real[3](p, *a, **kw)

    def iterdir(self):
        hit(self)
        return real[4](self)
    os.stat, os.lstat, os.listdir, os.scandir, Path.iterdir = stat, lstat, listdir, scandir, iterdir
    try:
        yield seen
    finally:
        os.stat, os.lstat, os.listdir, os.scandir, Path.iterdir = real


def git(repo: Path, *args) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
           "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                   check=True, capture_output=True, env=env)


def run_cli(argv: list[str]) -> tuple[int, str]:
    """scripts/e6kd_trigger.py main() in this process: its exit code and its RESULT line (prefixed "RESULT NOT LAST:"
    when another line follows it)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CLI.main(argv)
    out = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    lines = [ln for ln in out if ln.startswith("RESULT:")]
    res = lines[-1] if lines else buf.getvalue()[-300:]
    return rc, (res if lines and out[-1] == lines[-1] else f"RESULT NOT LAST: {res}")


def argv_of(inp: T.TriggerInputs, cmd: str) -> list[str]:
    argv = [cmd, "--e3-run-dir", inp.e3_run_dir, "--e3-checkpoint-sha256", inp.e3_checkpoint_sha256,
            "--e6-run-dir", inp.e6_run_dir, "--e6-eval-dir", inp.e6_eval_dir,
            "--e6-telemetry-sha256", inp.e6_telemetry_sha256, "--e6-selection-sha256", inp.e6_selection_sha256,
            "--lambda-selection", inp.lambda_selection, "--lambda-selection-sha256", inp.lambda_selection_sha256,
            "--alpha-selection", inp.alpha_selection, "--alpha-selection-sha256", inp.alpha_selection_sha256,
            "--trigger-eval-dir", inp.trigger_eval_dir, "--host-label", inp.host_label]
    if cmd == "decide":
        argv += ["--out", inp.out]
    return argv + (["--allow-smoke-inputs"] if inp.allow_smoke_inputs else [])


# ---------------------------------------------------------------- evaluator artifacts
def write_artifact(d: Path, s: dict) -> None:
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(s, sort_keys=True), encoding="utf-8")
    (d / "per_image.jsonl").write_text("", encoding="utf-8")
    with open(d / "sufficient_stats.npz", "wb") as fh:
        np.savez(fh, tp=np.zeros(116, dtype=np.int64))
    lines = [f"{sha_file(d / n)}  {n}" for n in ARTIFACT_FILES]
    (d / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_doc(*, value: float, run_id: str, ck_sha: str, stage: str, precision: str, backend, rows: int = 846,
                commit: str = "a" * 40, digest=DIGEST, metric: str = METRIC, threads: int = 4, device: str = "cpu",
                protocol: str = "core_preprocess/1.0.0", status: str = "provisional", governed: bool = True,
                batch: int = 1, workers: int = 0, split: str = "val", gt_sum: int | None = None,
                manifest: str = "d" * 64, class_map: str = "e" * 64, config: str = "c" * 64, torch_v="2.1.0+cpu",
                metric_protocol: str = "plantseg-metrics/1.0.0", random_init: bool = False,
                role: str = "student") -> dict:
    gt = [1000] * 116
    gt[0] = (gt_sum if gt_sum is not None else GT_SUM if rows == 846 else rows * 100_000) - 115_000
    return {"schema_version": "plantseg-eval/1.0.0", "metric_protocol": metric_protocol,
            "run": {"run_id": run_id, "artifact_status": status, "stage": stage, "model_role": role,
                    "precision": precision, "quant_backend": backend, "checkpoint_sha256": ck_sha,
                    "random_init": random_init, "repo_commit": commit, "governed_paths_clean": governed,
                    "metric_impl_sha256": metric, "config_sha256": config,
                    "env": {"torch": torch_v, "device": device},
                    "eval_runtime": {"batch_size": batch, "forward_batches": rows, "num_workers": workers,
                                     "torch_num_threads": threads, "image_digest": digest}},
            "dataset": {"split": split, "expected_rows": rows, "actual_rows": rows, "preprocess_protocol": protocol,
                        "split_manifest_sha256": manifest, "class_map_sha256": class_map},
            "dataset_level": {"all_class_miou": value}, "per_class": {"gt_support": gt}}


# ---------------------------------------------------------------- fixtures: E3, the repository, E6
def cache_dir() -> Path | None:
    return OPTS["cache"]


def cache_ready() -> bool:
    c = cache_dir()
    return c is not None and (c / "READY").is_file()


def clone_run(src: Path, dst: Path) -> Path:
    """A copy of a run: checkpoints hard-linked (never rewritten), the record copied."""
    (dst / Q.EPOCH_DIR).mkdir(parents=True)
    for p in (src / Q.EPOCH_DIR).iterdir():
        os.link(p, dst / Q.EPOCH_DIR / p.name)
    shutil.copy2(src / Q.TELEMETRY_NAME, dst / Q.TELEMETRY_NAME)
    return dst


@functools.cache
def fx_e3() -> dict:
    """E3 seed 42's parent run directory: the α winner's run (AM-19 item 4(a))."""
    d = TMP / E3_RUN_ID
    d.mkdir()
    if cache_ready():
        shutil.copy2(cache_dir() / "e3" / E3_CKPT, d / E3_CKPT)
    else:
        make_e3_checkpoint(d / E3_CKPT)
    (d / "best.json").write_text(json.dumps({"best_ckpt": f"/workspace/e3_ckpts/{E3_CKPT}",
                                             "best_val_miou_all_class": E3_BEST}), encoding="utf-8")
    (d / "e3_run_meta.jsonl").write_text(json.dumps({"event": "run_meta", "stage": "E3", "mode": "real", "seed": 42,
                                                     "lambda_logit": 0.5, "alpha_cwd": 25.0}) + "\n", encoding="utf-8")
    (d / "e3_telemetry.jsonl").write_text(json.dumps({"event": "run_end", "iter": 80000, "checks_passed": True})
                                          + "\n", encoding="utf-8")
    return {"dir": d, "sha": sha_file(d / E3_CKPT)}


def trigger_doc(e3_val: float, e6_val: float, drop_pp: str, triggered: bool, **over) -> dict:
    """A synthetic e6kd_trigger/1 record, with the fields the launch gate reads (values written out, not computed)."""
    rp = FX["repo_shas"]
    doc = {"format": "e6kd_trigger/1", "seed": 42, "e3_val": e3_val, "e6_val": e6_val, "drop_pp": drop_pp,
           "triggered": triggered, "smoke_inputs": False,
           "e3": {"run_id": E3_RUN_ID, "checkpoint_sha256": fx_e3()["sha"]},
           "e6": {"clip_norm": 1.0, "clip_selection_sha256": CLIP_SHA},
           "selections": {"lambda": {"sha256": rp["lambda"]}, "alpha": {"sha256": rp["alpha"]}}}
    doc.update(over)
    return doc


@functools.cache
def fx_repo() -> dict:
    """The temp git repository: the λ and α selections and their variants, and the trigger records of the gate."""
    e3 = fx_e3()
    (SELREPO / "sel").mkdir(parents=True)
    (SELREPO / "trig").mkdir()
    lam = {"format": "lambda_selection/1", "winner": LAM_WINNER}
    alp = {"format": "alpha_selection/1", "winner": {"alpha": 25.0, "run_id": E3_RUN_ID, "ckpt_sha256": e3["sha"]}}
    sel = {"sel/lambda.json": lam, "sel/alpha.json": alp, "sel/lambda_changed.json": lam,
           "sel/alpha_other_ckpt.json": {**alp, "winner": {**alp["winner"], "ckpt_sha256": "9" * 64}},
           "sel/alpha_other_run.json": {**alp, "winner": {**alp["winner"], "run_id": "e3_s42_lambda0.5_alpha50_80k_a1"}}}
    for rel, doc in sel.items():
        (SELREPO / rel).write_text(json.dumps(doc), encoding="utf-8")
    FX["repo_shas"] = {"lambda": sha_file(SELREPO / "sel/lambda.json"), "alpha": sha_file(SELREPO / "sel/alpha.json")}
    trig = {"trig/ok.json": trigger_doc(0.4001, 0.39, "1.0100", True),
            "trig/not.json": trigger_doc(0.395, 0.39, "0.500", False),
            "trig/smoke.json": trigger_doc(0.4001, 0.39, "1.0100", True, smoke_inputs=True),
            "trig/seed43.json": trigger_doc(0.4001, 0.39, "1.0100", True, seed=43),
            "trig/fmt.json": trigger_doc(0.4001, 0.39, "1.0100", True, format="e6kd_trigger/0"),
            "trig/inconsistent.json": trigger_doc(0.395, 0.39, "0.500", True),
            "trig/changed.json": trigger_doc(0.4001, 0.39, "1.0100", True)}
    for rel, doc in trig.items():
        (SELREPO / rel).write_text(json.dumps(doc), encoding="utf-8")
    (SELREPO / "trig/strict.json").write_text('{"format": "e6kd_trigger/1", "e3_val": NaN}', encoding="utf-8")
    files = [*sel, *trig, "trig/strict.json"]
    git(SELREPO, "init", "-q")
    git(SELREPO, "add", "--", *files)
    git(SELREPO, "commit", "-q", "-m", "selections and trigger records")
    (SELREPO / "sel/lambda_changed.json").write_text(json.dumps({**lam, "note": "edited"}), encoding="utf-8")
    (SELREPO / "trig/changed.json").write_text(json.dumps(trigger_doc(0.4001, 0.38, "2.0100", True)),
                                              encoding="utf-8")
    (SELREPO / "sel/alpha_untracked.json").write_text(json.dumps(alp), encoding="utf-8")
    (SELREPO / "trig/untracked.json").write_text(json.dumps(trig["trig/ok.json"]), encoding="utf-8")
    return {"repo": SELREPO, **FX["repo_shas"]}


def selection_record(rel: str, what: str) -> dict:
    fx_repo()                                           # a case run alone (--only) builds the repository here
    fmt = Q.LAMBDA_SELECTION_FORMAT if what == "lambda" else Q.ALPHA_SELECTION_FORMAT
    return Q.read_tracked_selection(rel, sha_file(SELREPO / rel), fmt, what, SELREPO)


@functools.cache
def fx_parent() -> dict:
    e3 = fx_e3()
    return Q.resolve_parent(resolve_quant_stage("e6"), e3["dir"], e3["sha"], 42)


@functools.cache
def student_src():
    return confident_student(seed=0, steps=30)          # any trained source: block B's scores are fabricated


def raw_run(name: str) -> Path:
    """A raw smoke-mode E6 run, trained once (or cloned from --cache-dir): clip 1.0, and non-finite at step 1."""
    if cache_ready():
        return clone_run(cache_dir() / name, TMP / name)
    out = TMP / name
    make_qat_run(out, model=student_src(), stage="e6", seed=42, clip=1.0, source_sha256=fx_e3()["sha"],
                 nan_at=(1,) if name == E6_NAN1_RAW else ())
    return out


@functools.cache
def fx_raw(name: str) -> Path:
    return raw_run(name)


def fill_cache() -> None:
    """--cache-dir: the E3 checkpoint and both raw runs, all at once, so that jobs started later only clone them."""
    c = cache_dir()
    if c is None or cache_ready():
        return
    (c / "e3").mkdir(parents=True, exist_ok=True)
    shutil.copy2(fx_e3()["dir"] / E3_CKPT, c / "e3" / E3_CKPT)
    for name in (E6_RAW, E6_NAN1_RAW):
        clone_run(fx_raw(name), c / name)
    (c / "READY").write_text("raw smoke runs\n", encoding="utf-8")


def relabel(run: Path, **edits) -> str:
    """Re-label a smoke run's record as E6 seed 42 of record (+ edits). Returns the new telemetry sha256."""
    fx_repo()
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    meta = {"mode": "real", "clip_source": "clip_selection", "u4_pilot": False,
            "clip_selection": {"path": "/sel/clip_selection.json", "sha256": CLIP_SHA, "winner": 1.0},
            "e6_selections": {"lambda_selection": selection_record("sel/lambda.json", "lambda"),
                              "alpha_selection": selection_record("sel/alpha.json", "alpha")},
            "parent": fx_parent()}
    meta.update(edits)
    rows[0].update(meta)
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return sha_file(tel)


def e6_variant(dirname: str, raw: str = E6_RAW, **edits) -> tuple[Path, str]:
    run = clone_run(fx_raw(raw), TMP / dirname)
    return run, relabel(run, **edits)


@functools.cache
def fx_e6() -> dict:
    run, sha = e6_variant("e6_s42")
    return {"dir": run, "sha": sha}


# ---------------------------------------------------------------- fixtures: block B (E6's eval directories)
def vals(**kw) -> dict:
    v = {e: 0.30 for e in range(1, 16)}
    for k, x in kw.items():
        v[int(k[1:])] = x
    return v


def fab_block_b(run: Path, ev: Path, values: dict, *, excluded=(), real: dict | None = None) -> None:
    """converted/, scores/, qat_convert.json and qat_epoch_eval.json for `run`, scored with `values` (as
    scripts/smoke_qat_selection.py fabricates them). With `real`, epoch real["epoch"] is the one converted and scored
    for real; the others copy its evaluation identity and its summary's shared fields."""
    rec = A.read_run_record(run)
    meta = rec["run_meta"]
    conv, scores = ev / A.CONVERTED_DIR, ev / A.SCORES_DIR
    conv.mkdir(parents=True, exist_ok=True)
    if real:
        prov0, s0 = real["provenance"], real["summary"]
        ident = {k: prov0[k] for k in S.EVAL_IDENTITY_KEYS}
        shared = dict(commit=s0["run"]["repo_commit"], digest=s0["run"]["eval_runtime"]["image_digest"],
                      metric=s0["run"]["metric_impl_sha256"], threads=s0["run"]["eval_runtime"]["torch_num_threads"],
                      rows=s0["dataset"]["expected_rows"], manifest=s0["dataset"]["split_manifest_sha256"],
                      class_map=s0["dataset"]["class_map_sha256"], config=s0["run"]["config_sha256"],
                      torch_v=s0["run"]["env"]["torch"], gt_sum=sum(s0["per_class"]["gt_support"]))
        smoke = True
    else:
        ident = {"git_head": "a" * 40, "host_label": "fab-host", "cpu_model": "fab-cpu", "purpose": "record"}
        shared, smoke = dict(commit="a" * 40), False
    outcomes, rows = [], []
    for e in range(1, 16):
        names = A.epoch_names(e)
        ck_sha = rec["ends"][e]["checkpoint_sha256"]
        if real and e == real["epoch"]:
            prov_p = conv / names["run_meta"]
            summ = scores / f"e{e:02d}" / "summary.json"
            outcomes.append({"epoch": e, "status": "converted"})
            rows.append({"epoch": e, "checkpoint_sha256": ck_sha, "status": "scored", "run_id": real["run_id"],
                         "provenance": {"path": f"{A.CONVERTED_DIR}/{names['run_meta']}", "sha256": sha_file(prov_p)},
                         "converted_artifact_sha256": real["provenance"]["converted_artifact_sha256"],
                         "score_dir": f"{A.SCORES_DIR}/e{e:02d}",
                         "summary": {"path": f"{A.SCORES_DIR}/e{e:02d}/summary.json", "sha256": sha_file(summ)},
                         "all_class_miou": values[e], "evaluator_command": real["argv"][1:],
                         "torch_num_threads": shared["threads"], "affinity_cpus": 4})
            continue
        if e in excluded:
            (conv / names["not_convertible"]).write_text(json.dumps(
                {"status": "not convertible", "rule": A.NOT_CONVERTIBLE_RULE, "qat_checkpoint_sha256": ck_sha,
                 "failing": ["fabricated"], "epoch": e, "run_id": rec["run_id"],
                 "telemetry_sha256": rec["telemetry_sha256"], **ident}), encoding="utf-8")
            outcomes.append({"epoch": e, "status": "not convertible"})
            rows.append({"epoch": e, "checkpoint_sha256": ck_sha, "status": "excluded", "rule": A.NOT_CONVERTIBLE_RULE,
                         "not_convertible": f"{A.CONVERTED_DIR}/{names['not_convertible']}"})
            continue
        tsid = {"schema": "plantseg-int8-torchscript/1.0.0", "stage": meta["stage"], "quantization": "qat",
                "num_classes": 116, "engine": "qnnpack", "artifact_role": "accuracy",
                "source_checkpoint_sha256": meta["source_checkpoint_sha256"], "qat_checkpoint_sha256": ck_sha,
                "epoch": e, "run_id": rec["run_id"], "qconfig_fingerprint": "f" * 16,
                "state_dict_companion": names["state_dict"]}
        ts = conv / names["torchscript"]
        with zipfile.ZipFile(ts, "w") as z:
            z.writestr(f"e{e:02d}/extra/plantseg_int8.json", json.dumps(tsid))
            z.writestr(f"e{e:02d}/data.pkl", f"fabricated {rec['run_id']} e{e:02d}")
        ts_sha = sha_file(ts)
        sd = conv / names["state_dict"]
        sd.write_bytes(f"fabricated state_dict {rec['run_id']} e{e:02d}".encode())
        prov = {"stage": meta["stage"], "epoch": e, "run_id": rec["run_id"],
                "state_dict_artifact": {"path": names["state_dict"], "sha256": sha_file(sd)}, **ident,
                "qat_checkpoint_sha256": ck_sha, "converted_artifact": names["torchscript"],
                "converted_artifact_sha256": ts_sha,
                "verification": {"checks": {"weights_per_channel_int8": True, "no_undeclared_float_module": True,
                                            "torchscript_float_boundary": True,
                                            "parity_torchscript_state_dict_eager": True}}}
        pp = conv / names["run_meta"]
        pp.write_text(json.dumps(prov), encoding="utf-8")
        run_id = f"{rec['run_id']}_e{e:02d}_{ck_sha[:12]}"
        write_artifact(scores / f"e{e:02d}", summary_doc(value=values[e], run_id=run_id, ck_sha=ts_sha,
                                                         stage=meta["stage"], precision="int8_qat",
                                                         backend="qnnpack", **shared))
        outcomes.append({"epoch": e, "status": "converted"})
        rows.append({"epoch": e, "checkpoint_sha256": ck_sha, "status": "scored", "run_id": run_id,
                     "provenance": {"path": f"{A.CONVERTED_DIR}/{names['run_meta']}", "sha256": sha_file(pp)},
                     "converted_artifact_sha256": ts_sha, "score_dir": f"{A.SCORES_DIR}/e{e:02d}",
                     "summary": {"path": f"{A.SCORES_DIR}/e{e:02d}/summary.json",
                                 "sha256": sha_file(scores / f"e{e:02d}" / "summary.json")},
                     "all_class_miou": values[e],
                     "evaluator_command": ["-B", "scripts/evaluate_model.py", "--stage", meta["stage"]],
                     "torch_num_threads": 4, "affinity_cpus": 4})
    head = {"run_dir": rec["run_dir"], "run_id": rec["run_id"], "stage": meta["stage"],
            "telemetry_sha256": rec["telemetry_sha256"], **ident, "smoke_inputs": smoke}
    (ev / "qat_convert.json").write_text(json.dumps({"format": "qat_convert/1", **head, "outcomes": outcomes}),
                                         encoding="utf-8")
    (ev / "qat_epoch_eval.json").write_text(json.dumps({"format": "qat_epoch_eval/1", **head, "epochs": rows}),
                                            encoding="utf-8")


def select(run: Path, ev: Path, sha: str, *, smoke: bool = False) -> str:
    """Block B's selection (src/quant/qat_select.py epoch_selection), written as select_qat_epoch.py writes it."""
    rules, rules_sha = S.load_rules()
    sel = S.epoch_selection(run, ev, expect_telemetry_sha256=sha, rules=rules, rules_sha256=rules_sha,
                            allow_smoke_inputs=smoke)
    (ev / T.SELECTION_RECORD).write_bytes(A.json_bytes(sel))
    return sha_file(ev / T.SELECTION_RECORD)


@functools.cache
def fx_block_b() -> dict:
    """E6 seed 42's block B: the winner is epoch 7, value 0.39 (every other epoch 0.30)."""
    run, sha = fx_e6()["dir"], fx_e6()["sha"]
    ev = TMP / "ev_s42"
    fab_block_b(run, ev, vals(**{f"e{E6_WINNER:02d}": E6_VALUE}))
    return {"ev": ev, "sel_sha": select(run, ev, sha)}


def block_b_copy(name: str, edit=None) -> tuple[Path, str]:
    """A copy of block B's eval dir, its qat_selection.json edited by `edit` (a function of the record)."""
    ev = TMP / name
    shutil.copytree(fx_block_b()["ev"], ev)
    if edit is not None:
        p = ev / T.SELECTION_RECORD
        doc = json.loads(p.read_text(encoding="utf-8"))
        edit(doc)
        p.write_bytes(A.json_bytes(doc))
    return ev, sha_file(ev / T.SELECTION_RECORD)


# ---------------------------------------------------------------- inputs and trigger eval dirs
def inputs(tdir, *, out=None, run: Path | None = None, run_sha: str | None = None, ev: Path | None = None,
           sel_sha: str | None = None, e3_dir=None, e3_sha: str | None = None, lam: str = "sel/lambda.json",
           alpha: str = "sel/alpha.json", host: str = HOST, smoke: bool = False) -> T.TriggerInputs:
    """Valid inputs (the fixtures of record), with the given overrides."""
    fx_repo()
    e3 = fx_e3()
    if run is None:
        run, run_sha = fx_e6()["dir"], fx_e6()["sha"]
    if ev is None:
        ev, sel_sha = fx_block_b()["ev"], fx_block_b()["sel_sha"]
    if sel_sha is None:
        p = Path(ev) / T.SELECTION_RECORD
        sel_sha = sha_file(p) if p.is_file() else "f" * 64
    return T.TriggerInputs(
        e3_run_dir=str(e3_dir or e3["dir"]), e3_checkpoint_sha256=e3_sha or e3["sha"], e6_run_dir=str(run),
        e6_eval_dir=str(ev), e6_telemetry_sha256=run_sha, e6_selection_sha256=sel_sha,
        lambda_selection=lam, lambda_selection_sha256=sha_file(SELREPO / lam) if (SELREPO / lam).is_file() else "f" * 64,
        alpha_selection=alpha, alpha_selection_sha256=sha_file(SELREPO / alpha) if (SELREPO / alpha).is_file()
        else "f" * 64,
        trigger_eval_dir=str(tdir), host_label=host, allow_smoke_inputs=smoke, out=None if out is None else str(out))


def score_code(inp: T.TriggerInputs) -> str | None:
    """score's refusal code. The evaluator is a stub that fails at once: if a refusal under test were missing, the
    run would end at evaluator_failed instead of scoring."""
    with patched_attr(T, "EVALUATOR_SCRIPT", str(FAILING)):
        return code_of(T.score, inp, repo_root=SELREPO)


def decide_code(inp: T.TriggerInputs) -> str | None:
    return code_of(T.decide, inp, repo_root=SELREPO)


def fab_trigger(name: str, *, e3: float = 0.4001, e6: float = E6_VALUE, ev: Path | None = None,
                sel_sha: str | None = None, smoke: bool = False, s3: dict | None = None, s6: dict | None = None,
                edit_doc=None, after=None) -> T.TriggerInputs:
    """A trigger eval dir as `score` writes it, its two summaries holding chosen values (and edits); returns decide's
    inputs, its --out a new file. The commands, run ids and recorded inputs are the module's own for these inputs."""
    tdir = TMP / name
    inp = inputs(tdir, out=TMP / f"{name}_record.json", ev=ev, sel_sha=sel_sha, smoke=smoke)
    ctx = T.check_inputs(inp, purpose="score", repo_root=SELREPO)
    cmds, ident = T.commands(ctx), T.code_identity()
    head = ident["git_head"]
    base = {"e3": dict(value=e3, run_id=ctx["run_ids"]["e3"], ck_sha=ctx["parent"]["checkpoint_sha256"], stage="E3",
                       precision="fp32", backend=None, commit=head),
            "e6": dict(value=e6, run_id=ctx["run_ids"]["e6"], ck_sha=ctx["winner"]["artifact_of_record"]["sha256"],
                       stage="E6", precision="int8_qat", backend="qnnpack", commit=head)}
    base["e3"].update(s3 or {})
    base["e6"].update(s6 or {})
    scores = {}
    for which in ("e3", "e6"):
        d = tdir / T.SCORE_DIRS[which]
        write_artifact(d, summary_doc(**base[which]))
        scores[which] = {"evaluator_command": cmds[which][1:], "evaluator_python": cmds[which][0], "exit_code": 0,
                         "seconds": 1.0, "run_id": ctx["run_ids"][which], "score_dir": T.SCORE_DIRS[which],
                         "summary": {"path": f"{T.SCORE_DIRS[which]}/summary.json", "sha256": sha_file(d / "summary.json")},
                         "all_class_miou": base[which]["value"], "torch_num_threads": 4}
    doc = {"format": T.TRIGGER_EVAL_FORMAT, "lane": T.LANE, "inputs": T.recorded_inputs(ctx), "host_label": HOST,
           "cpu_model": Q.cpu_model(), "git_head": head, "code_clean_at_head": ident["code_clean_at_head"],
           "code_files_sha256": ident["code_files_sha256"], "smoke_inputs": smoke, "threads_note": T.THREADS_NOTE,
           "affinity_cpus": 4, "data_root": ctx["data_root"], "isolation_counts": ctx["isolation_counts"],
           "status": "scored", "scores": scores, "created_wall_clock": 0.0}
    if edit_doc is not None:
        edit_doc(doc)
    (tdir / T.TRIGGER_EVAL_RECORD).write_bytes(A.json_bytes(doc))
    if after is not None:
        after(tdir)
    return inp


def decided(name: str, **kw) -> tuple[str | None, dict, T.TriggerInputs]:
    """decide on a fabricated trigger eval dir: (its refusal code or None, its record, its inputs)."""
    inp = fab_trigger(name, **kw)
    code = decide_code(inp)
    out = Path(inp.out)
    return code, (json.loads(out.read_text(encoding="utf-8")) if code is None and out.is_file() else {}), inp


def minimal_trigger_dir(name: str, *, stop: bool = False, record: bool = True) -> Path:
    """A trigger eval dir for the checks decide makes before it reads one: a STOP file, trigger_eval.json present."""
    d = TMP / name
    d.mkdir()
    if record:
        (d / T.TRIGGER_EVAL_RECORD).write_text("{}\n", encoding="utf-8")
    if stop:
        (d / T.TRIGGER_STOP).write_text("{}\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------- the cases
CASES: list[tuple[str, str, object]] = []          # (name, section, fn); each case reports its own check


def case(section: str):
    def deco(fn):
        CASES.append((fn.__name__, section, fn))
        return fn
    return deco


def refusal_case(section: str, name: str, fn, want: str) -> None:
    """A case whose check is one refusal or incomplete code, reached with every earlier gate passing."""
    def run():
        got = fn()
        check(name, got == want, f"[{got}]")
    CASES.append((name, section, run))


# ---- inputs: steps 1-8, in order (score unless said otherwise)
@case("inputs")
def g1_forbidden_names_refused_before_any_probe():
    # DL-24 S-F5: a later gate fails too (e6_selection_sha256_mismatch, step 7), so that with the name check removed
    # (the mutation harness's edit) no gate lets score create the forbidden-named directory
    base = dataclasses.replace(inputs(TMP / "g1_trigger", out=TMP / "g1_record.json"), e6_selection_sha256="f" * 64)
    bad = TMP / "nowhere_latest_dir"                     # never created; its name holds the token SL-1 forbids
    seen_all, codes = [], {}
    for field in ("e3_run_dir", "e6_run_dir", "e6_eval_dir", "trigger_eval_dir"):
        with probe_recorder(bad) as seen:
            codes[field] = score_code(dataclasses.replace(base, **{field: str(bad)}))
        seen_all += seen
    with probe_recorder(bad) as seen:
        codes["out"] = decide_code(dataclasses.replace(base, out=str(bad / "e6kd_trigger.json")))
    seen_all += seen
    with probe_recorder(bad) as seen:
        rc, res = run_cli(argv_of(dataclasses.replace(base, e6_eval_dir=str(bad)), "score"))
    seen_all += seen
    alias = TMP / "alias_run"                            # its own name is clean; its target is never created
    os.symlink(TMP / "attestations" / "run", alias)
    codes["resolved"] = score_code(dataclasses.replace(base, e3_run_dir=str(alias)))
    check("g1_forbidden_names_refused_before_any_probe",
          all(c == "forbidden_path" for c in codes.values()) and not seen_all and rc == 2
          and res.startswith("RESULT: REFUSED [forbidden_path]"),
          f"{codes}; probed {seen_all[:3]}; CLI exit {rc} {res[:60]!r}")


@case("inputs")
def refuses_sha256_format():
    inp = dataclasses.replace(inputs(TMP / "fmt_trigger"), e6_telemetry_sha256="abc")
    rc, res = run_cli(argv_of(inp, "score"))
    check("refuses_sha256_format", rc == 2 and res.startswith("RESULT: REFUSED [sha256_format]")
          and not (TMP / "fmt_trigger").exists(), f"exit {rc} {res[:80]!r}")


refusal_case("inputs", "refuses_host_label_empty",
             lambda: score_code(inputs(TMP / "host_trigger", host="  ")), "host_label")


@case("inputs")
def refuses_image_not_of_record():
    with patched_env(PLANTSEG_IMAGE_DIGEST="sha256:" + "7" * 64):
        got = score_code(inputs(TMP / "image_trigger"))
    check("refuses_image_not_of_record", got == "image_not_of_record" and DIGEST == Q.IMAGE_DIGEST_OF_RECORD,
          f"[{got}]")


@case("inputs")
def refuses_trigger_eval_dir_inside_repo():
    target = REPO / "trigger_eval_scratch"               # never created
    got = score_code(inputs(target))
    check("refuses_trigger_eval_dir_inside_repo", got == "trigger_eval_dir_not_outside_repo" and not target.exists(),
          f"[{got}]")


@case("inputs")
def refuses_trigger_eval_dir_not_fresh():
    used = TMP / "used_trigger"
    used.mkdir()
    (used / "x.txt").write_text("x", encoding="utf-8")
    got = score_code(inputs(used))
    check("refuses_trigger_eval_dir_not_fresh", got == "trigger_eval_dir_not_fresh", f"[{got}]")


refusal_case("inputs", "refuses_selection_untracked",
             lambda: score_code(inputs(TMP / "untr_trigger", alpha="sel/alpha_untracked.json")),
             "selection_not_tracked")
refusal_case("inputs", "refuses_selection_changed_since_head",
             lambda: score_code(inputs(TMP / "chg_trigger", lam="sel/lambda_changed.json")),
             "selection_changed_since_head")
refusal_case("inputs", "refuses_missing_e3_run_dir",
             lambda: score_code(inputs(TMP / "e3miss_trigger", e3_dir=TMP / "e3_never_created")),
             "parent_dir_missing")
refusal_case("inputs", "refuses_e3_not_alpha_winner_run",
             lambda: score_code(inputs(TMP / "e3run_trigger", alpha="sel/alpha_other_run.json")),
             "parent_run_id_mismatch")
refusal_case("inputs", "refuses_e3_checkpoint_not_alpha_winner_sha",
             lambda: score_code(inputs(TMP / "e3sha_trigger", alpha="sel/alpha_other_ckpt.json")),
             "e3_checkpoint_not_alpha_winner")
refusal_case("inputs", "refuses_e6_telemetry_sha256_mismatch",
             lambda: score_code(inputs(TMP / "tel_trigger", run=fx_e6()["dir"], run_sha="f" * 64)),
             "e6_telemetry_sha256_mismatch")


def variant_code(tname: str, dirname: str, **edits) -> str | None:
    run, sha = e6_variant(dirname, **edits)
    return score_code(inputs(TMP / tname, run=run, run_sha=sha))


refusal_case("inputs", "refuses_e6_stage_not_e6",
             lambda: variant_code("stage_trigger", "e6_stage_e5", stage="E5"), "e6_stage_not_e6")
refusal_case("inputs", "refuses_e6_not_seed_42",
             lambda: variant_code("seed_trigger", "e6_seed43", seed=43), "e6_seed_not_42")
refusal_case("inputs", "refuses_e6_mode_not_real",
             lambda: variant_code("mode_trigger", "e6_mode_smoke", mode="smoke"), "e6_mode_not_real")
refusal_case("inputs", "refuses_e6_parent_not_the_e3_checkpoint",
             lambda: variant_code("parent_trigger", "e6_parent_other",
                                  parent={**fx_parent(), "checkpoint_sha256": "9" * 64}),
             "e6_parent_not_e3_checkpoint")
refusal_case("inputs", "refuses_e6_selections_mismatch",
             lambda: variant_code("sels_trigger", "e6_sels_other",
                                  e6_selections={"lambda_selection": {"sha256": "9" * 64},
                                                 "alpha_selection": selection_record("sel/alpha.json", "alpha")}),
             "e6_selections_mismatch")
refusal_case("inputs", "refuses_e6_clip_binding",
             lambda: variant_code("clip_trigger", "e6_clip_pilot", clip_source="u4_pilot", u4_pilot=True,
                                  clip_selection=None), "e6_clip_binding")


@case("inputs")
def refuses_e6_no_model():
    run, sha = e6_variant("e6_s42_no_model", raw=E6_NAN1_RAW)
    ev = TMP / "ev_s42_no_model"
    fab_block_b(run, ev, vals(), excluded=range(1, 16))
    got = score_code(inputs(TMP / "nomodel_trigger", run=run, run_sha=sha, ev=ev))
    check("refuses_e6_no_model", got == "e6_no_model", f"[{got}]")


@case("inputs")
def refuses_e6_selection_missing_incomplete():
    ev = TMP / "ev_s42_unselected"
    fab_block_b(fx_e6()["dir"], ev, vals(**{f"e{E6_WINNER:02d}": E6_VALUE}))
    got = score_code(inputs(TMP / "unsel_trigger", ev=ev, sel_sha="f" * 64))
    check("refuses_e6_selection_missing_incomplete", got == "e6_selection_missing", f"[{got}]")


refusal_case("inputs", "refuses_e6_selection_sha256_mismatch",
             lambda: score_code(inputs(TMP / "selsha_trigger", ev=fx_block_b()["ev"], sel_sha="f" * 64)),
             "e6_selection_sha256_mismatch")


def _rules_edit(doc: dict) -> None:
    doc["rules_sha256"] = "0" * 64


def _value_edit(doc: dict) -> None:
    doc["winner"]["value"] = 0.41


refusal_case("inputs", "refuses_e6_selection_format",
             lambda: score_code(inputs(TMP / "selfmt_trigger", ev=block_b_copy("ev_s42_rules", _rules_edit)[0])),
             "e6_selection_format")
refusal_case("inputs", "refuses_e6_selection_recompute_differs",
             lambda: score_code(inputs(TMP / "seldiff_trigger", ev=block_b_copy("ev_s42_value", _value_edit)[0])),
             "e6_selection_differs")


@case("inputs")
def refuses_data_root_not_trainval():
    small = TMP / "data_small"
    make_tree(small, n_train=16, n_val=8, seed=6)
    old = DATA["root"]
    DATA["root"] = str(small)
    try:
        got = score_code(inputs(TMP / "root_trigger"))
    finally:
        DATA["root"] = old
    check("refuses_data_root_not_trainval", got == "data_root_not_trainval_only", f"[{got}]")


# ---- outputs: the trigger eval dir and --out
@case("outputs")
def refuses_decide_before_score_incomplete():
    d = minimal_trigger_dir("empty_trigger", record=False)
    inp = inputs(d, out=TMP / "empty_record.json")
    rc, res = run_cli(argv_of(inp, "decide"))
    check("refuses_decide_before_score_incomplete", rc == 3
          and res.startswith("RESULT: INCOMPLETE [trigger_eval_missing]") and not (TMP / "empty_record.json").exists(),
          f"exit {rc} {res[:80]!r}")


refusal_case("outputs", "refuses_stop_file_present",
             lambda: decide_code(inputs(minimal_trigger_dir("stop_trigger", stop=True, record=False),
                                        out=TMP / "stop_record.json")), "stop_present")


@case("outputs")
def refuses_out_inside_repo():
    target = REPO / "e6kd_trigger_scratch.json"          # never created
    got = decide_code(inputs(minimal_trigger_dir("outrepo_trigger"), out=target))
    check("refuses_out_inside_repo", got == "out_not_outside_repo" and not target.exists(), f"[{got}]")


@case("outputs")
def refuses_out_exists():
    taken = TMP / "taken_record.json"
    taken.write_text("{}\n", encoding="utf-8")
    got = decide_code(inputs(minimal_trigger_dir("outexists_trigger"), out=taken))
    check("refuses_out_exists", got == "out_exists" and taken.read_text(encoding="utf-8") == "{}\n", f"[{got}]")


def _fmt(doc):
    doc["format"] = "e6kd_trigger_eval/0"


def _smoke_flag(doc):
    doc["smoke_inputs"] = True


def _other_head(doc):
    doc["git_head"] = "b" * 40


def _other_code(doc):                                    # score ran other code at the same commit (DL-24 F1)
    doc["code_files_sha256"]["src/quant/e6kd_trigger.py"] = "0" * 64


def _batch16(doc):
    cmd = doc["scores"]["e3"]["evaluator_command"]
    cmd[cmd.index("--batch-size") + 1] = "16"


def _summary_sha(doc):
    doc["scores"]["e3"]["summary"]["sha256"] = "0" * 64


def _extra_score_dir(tdir: Path):
    (tdir / T.SCORES_DIR / "e9").mkdir()


def _manifest_broken(tdir: Path):
    with open(tdir / T.SCORE_DIRS["e6"] / "per_image.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{}\n")


for _i, (_name, _kw, _want) in enumerate([
    ("refuses_trigger_eval_format", dict(edit_doc=_fmt), "trigger_eval_format"),
    ("smoke_inputs_refused_for_record", dict(edit_doc=_smoke_flag), "trigger_eval_smoke_inputs"),
    ("refuses_eval_dir_mixed_commit", dict(edit_doc=_other_head), "trigger_eval_mismatch"),
    ("refuses_code_changed_since_score", dict(edit_doc=_other_code), "trigger_eval_mismatch"),
    ("refuses_evaluator_command_not_literal", dict(edit_doc=_batch16), "evaluator_command_not_literal"),
    ("refuses_scores_dir_contents", dict(after=_extra_score_dir), "scores_dir_contents"),
    ("refuses_summary_artifact_invalid", dict(after=_manifest_broken), "summary_artifact_invalid"),
    ("refuses_summary_sha256_mismatch", dict(edit_doc=_summary_sha), "summary_sha256_mismatch"),
]):
    # LB2: the trigger eval dir is named by its index in this list, never after its check
    refusal_case("outputs", _name, functools.partial(lambda i, kw: decided(f"out_{i:02d}", **kw)[0], _i, _kw), _want)


@case("outputs")
def refuses_evaluator_failure_spends_the_dir():
    tdir = TMP / "failing_trigger"
    got = score_code(inputs(tdir))
    stop = tdir / T.TRIGGER_STOP
    doc = json.loads(stop.read_text(encoding="utf-8")) if stop.is_file() else {}
    check("refuses_evaluator_failure_spends_the_dir", got == "evaluator_failed" and doc.get("status") == "STOP"
          and doc.get("failed") == "e3" and not (tdir / T.TRIGGER_EVAL_RECORD).exists()
          and decide_code(inputs(tdir, out=TMP / "failing_record.json")) == "stop_present",
          f"[{got}] STOP {doc.get('status')} failed {doc.get('failed')}")


@case("outputs")
def refusal_writes_nothing():
    t1 = TMP / "nothing_trigger_a"
    c1 = variant_code("nothing_trigger_a", "e6_nothing_clip", clip_source="u4_pilot", u4_pilot=True,
                      clip_selection=None)
    c2, _doc, inp = decided("nothing_trigger_b", e6=0.3899)
    check("refusal_writes_nothing", c1 == "e6_clip_binding" and not t1.exists() and c2 == "e6_rescore_differs"
          and not Path(inp.out).exists(), f"[{c1}] [{c2}]")


# ---- d1: the decision (decimal on the recorded values, strictly greater than 1.0 pp)
@case("d1")
def d1_drop_1_01_triggered():
    code, doc, _ = decided("dec_101", e3=0.4001)
    check("d1_drop_1_01_triggered", code is None and doc.get("triggered") is True and doc.get("drop_pp") == "1.0100",
          f"[{code}] triggered {doc.get('triggered')} drop_pp {doc.get('drop_pp')}")


@functools.cache
def d1_100() -> tuple:
    return decided("dec_100", e3=0.40)


@case("d1")
def d1_drop_1_00_not_triggered():
    code, doc, _ = d1_100()
    check("d1_drop_1_00_not_triggered", code is None and doc.get("triggered") is False
          and doc.get("drop_pp") == "1.00", f"[{code}] triggered {doc.get('triggered')} drop_pp {doc.get('drop_pp')}")


@case("d1")
def d1_drop_0_50_not_triggered():
    code, doc, _ = decided("dec_050", e3=0.395)
    check("d1_drop_0_50_not_triggered", code is None and doc.get("triggered") is False
          and doc.get("drop_pp") == "0.500", f"[{code}] triggered {doc.get('triggered')} drop_pp {doc.get('drop_pp')}")


@case("d1")
def d1_negative_drop_not_triggered():
    code, doc, _ = decided("dec_neg", e3=0.38)
    check("d1_negative_drop_not_triggered", code is None and doc.get("triggered") is False
          and doc.get("drop_pp") == "-1.00", f"[{code}] triggered {doc.get('triggered')} drop_pp {doc.get('drop_pp')}")


@case("d1")
def d1_decimal_not_float():
    code, doc, _ = d1_100()
    as_float = 100 * (0.40 - 0.39)                      # 1.0000000000000009 in binary64: the float rule would fire
    check("d1_decimal_not_float", code is None and as_float > 1.0 and doc.get("triggered") is False
          and doc.get("drop") == "0.01" and doc.get("e3_val_decimal") == "0.4" and doc.get("e6_val_decimal") == "0.39"
          and doc.get("drop_pp_float") == 1.0, f"float {as_float!r}; record {doc.get('drop')} {doc.get('drop_pp')}")


@case("d1")
def d1_threshold_rule_recorded():
    code, doc, _ = d1_100()
    check("d1_threshold_rule_recorded", code is None and doc.get("threshold") == 1.0
          and doc.get("threshold_decimal") == "1.0" and doc.get("threshold_unit") == "percentage points"
          and doc.get("decision_rule") == "strictly_greater" and "Decimal(repr(e3))" in str(doc.get("arithmetic"))
          and "seed 42" in str(doc.get("rule")) and doc.get("descriptive") is True and doc.get("holm_family") is False
          and doc.get("consequence") == T.CONSEQUENCE[False] and doc.get("seed") == 42,
          f"[{code}] {doc.get('threshold')} {doc.get('decision_rule')}")


@case("d1")
def d1_result_line_marks_smoke_inputs():
    code, doc, _ = d1_100()
    tail = "drop 1.00 pp <= 1.0 pp (E3 0.4, E6 0.39; seed 42)"
    record = T.result_line(doc) if doc else ""
    smoke = T.result_line({**doc, "smoke_inputs": True}) if doc else ""
    fired = T.result_line({**doc, "smoke_inputs": True, "triggered": True}) if doc else ""
    check("d1_result_line_marks_smoke_inputs", code is None and record == f"RESULT: NOT TRIGGERED -- {tail}"
          and smoke == f"RESULT: NOT TRIGGERED (smoke inputs, not a record) -- {tail}"
          and fired == "RESULT: TRIGGERED (smoke inputs, not a record) -- drop 1.00 pp > 1.0 pp (E3 0.4, E6 0.39; "
                       "seed 42)", f"[{code}] {record!r} / {smoke!r}")


@case("d1")
def d1_e3_value_is_the_evaluator_score_not_best_json():
    code, doc, _ = decided("dec_best", e3=0.4001)
    check("d1_e3_value_is_the_evaluator_score_not_best_json", code is None and doc.get("e3_val") == 0.4001
          and (doc.get("e3") or {}).get("best_json_value_not_used") == E3_BEST,
          f"[{code}] e3_val {doc.get('e3_val')}")


@case("d1")
def d1_json_keys_exact():
    code, doc, _ = d1_100()
    check("d1_json_keys_exact", code is None and sorted(doc) == TRIGGER_KEYS
          and sorted(doc.get("e3") or {}) == E3_KEYS and sorted(doc.get("e6") or {}) == E6_KEYS,
          f"[{code}] extra {sorted(set(doc) - set(TRIGGER_KEYS))} missing {sorted(set(TRIGGER_KEYS) - set(doc))}")


@case("d1")
def d1_input_sha256s_complete():
    code, doc, inp = d1_100()
    e3, ev, tdir = fx_e3()["dir"], Path(inp.e6_eval_dir), Path(inp.trigger_eval_dir)
    sel = json.loads((ev / T.SELECTION_RECORD).read_text(encoding="utf-8"))
    w = sel["winner"]
    want = {"e3_checkpoint": sha_file(e3 / E3_CKPT), "e3_best_json": sha_file(e3 / "best.json"),
            "e3_run_meta": sha_file(e3 / "e3_run_meta.jsonl"), "e3_telemetry": sha_file(e3 / "e3_telemetry.jsonl"),
            "e6_telemetry": sha_file(Path(inp.e6_run_dir) / Q.TELEMETRY_NAME), "e6_clip_selection": CLIP_SHA,
            "e6_qat_selection": sha_file(ev / T.SELECTION_RECORD), "qat_selection_rules": S.load_rules()[1],
            "e6_epoch_checkpoint": sha_file(Path(inp.e6_run_dir) / w["checkpoint"]),
            "e6_artifact_of_record": sha_file(ev / w["artifact_of_record"]["path"]),
            "e6_provenance": sha_file(ev / w["provenance"]["path"]), "e6_block_b_summary": sha_file(ev / w["summary"]["path"]),
            "lambda_selection": sha_file(SELREPO / "sel/lambda.json"), "alpha_selection": sha_file(SELREPO / "sel/alpha.json"),
            "e3_trigger_summary": sha_file(tdir / T.SCORE_DIRS["e3"] / "summary.json"),
            "e6_trigger_summary": sha_file(tdir / T.SCORE_DIRS["e6"] / "summary.json"),
            "trigger_eval": sha_file(tdir / T.TRIGGER_EVAL_RECORD)}
    got = doc.get("input_sha256s") or {}
    check("d1_input_sha256s_complete", code is None and sorted(got) == INPUT_KEYS and got == want,
          f"[{code}] differ {sorted(k for k in want if got.get(k) != want[k])}")


@case("d1")
def decide_twice_identical_bytes():
    inp = fab_trigger("twice_trigger", e3=0.4001)
    c1 = decide_code(inp)
    second = dataclasses.replace(inp, out=str(TMP / "twice_record_2.json"))
    c2 = decide_code(second)
    b1 = Path(inp.out).read_bytes() if c1 is None else b"1"
    b2 = Path(second.out).read_bytes() if c2 is None else b"2"
    check("decide_twice_identical_bytes", c1 is None and c2 is None and b1 == b2, f"[{c1}] [{c2}] equal {b1 == b2}")


# ---- artifacts: the two summaries against the evaluator values of record
for _i, (_name, _kw, _want) in enumerate([
    ("refuses_e3_summary_not_val", dict(s3={"split": "train"}), "summary_values"),
    ("refuses_e6_summary_not_val", dict(s6={"split": "train"}), "summary_values"),
    ("refuses_e3_summary_not_fp32", dict(s3={"precision": "int8_qat"}), "summary_values"),
    ("refuses_e6_summary_not_int8_qat_qnnpack", dict(s6={"backend": "fbgemm"}), "summary_values"),
    ("refuses_summary_not_cpu", dict(s3={"device": "cuda"}), "summary_values"),
    ("refuses_summary_not_canvas", dict(s3={"protocol": "upstream/1.0.0"}), "summary_values"),
    ("refuses_summary_batch_not_1", dict(s6={"batch": 16}), "summary_values"),
    ("refuses_rows_not_846_for_record", dict(s3={"rows": 845, "gt_sum": GT_SUM}), "summary_values"),
    ("refuses_summaries_not_same_code_path", dict(s6={"threads": 2}), "summaries_not_same_code_path"),
    ("refuses_e6_summary_not_the_winner_artifact", dict(s6={"ck_sha": "9" * 64}), "summary_values"),
    ("refuses_e3_summary_not_the_checkpoint", dict(s3={"ck_sha": "9" * 64}), "summary_values"),
    ("refuses_e6_rescore_differs", dict(e6=0.3899), "e6_rescore_differs"),
    ("refuses_score_invalid", dict(s3={"value": 1.5}), "score_invalid"),
    ("refuses_summary_not_the_student_image", dict(s3={"digest": "sha256:" + "7" * 64}), "summary_values"),
    # DL-24 S-F2: each remaining condition of the two summary gates, alone
    ("refuses_summary_governed_paths_dirty", dict(s3={"governed": False}), "summary_values"),
    ("refuses_summary_not_provisional", dict(s6={"status": "smoke"}), "summary_values"),
    ("refuses_summary_workers_not_0", dict(s3={"workers": 1}), "summary_values"),
    ("refuses_summary_run_id_not_the_trigger_run", dict(s6={"run_id": "e6kd_trigger_e6_e07_000000000000"}),
     "summary_values"),
    ("refuses_summary_stage_mismatch", dict(s3={"stage": "E2"}), "summary_values"),
    ("refuses_summary_role_not_student", dict(s6={"role": "teacher"}), "summary_values"),
    ("refuses_summary_random_init", dict(s3={"random_init": True}), "summary_values"),
    ("refuses_e3_summary_backend_not_null", dict(s3={"backend": "qnnpack"}), "summary_values"),
    ("refuses_e6_summary_not_int8_qat", dict(s6={"precision": "int8_ptq"}), "summary_values"),
    ("refuses_summary_gt_support_sum", dict(s3={"gt_sum": GT_SUM + 1}), "summary_values"),
    ("refuses_summaries_scored_at_another_commit", dict(s3={"commit": "b" * 40}, s6={"commit": "b" * 40}),
     "summary_values"),
    ("refuses_summaries_differ_in_metric_code", dict(s6={"metric": "c" * 64}), "summaries_not_same_code_path"),
    ("refuses_summaries_differ_in_split_manifest", dict(s6={"manifest": "f" * 64}), "summaries_not_same_code_path"),
    ("refuses_summaries_differ_in_class_map", dict(s6={"class_map": "f" * 64}), "summaries_not_same_code_path"),
    ("refuses_summaries_differ_in_torch", dict(s6={"torch_v": "2.1.1+cpu"}), "summaries_not_same_code_path"),
    ("refuses_summaries_differ_in_metric_protocol", dict(s6={"metric_protocol": "plantseg-metrics/0.9.0"}),
     "summaries_not_same_code_path"),
]):
    # LB2: the trigger eval dir is named by its index in this list, never after its check
    refusal_case("artifacts", _name, functools.partial(lambda i, kw: decided(f"art_{i:02d}", **kw)[0], _i, _kw),
                 _want)


# ---- gate: B-2's launch gate (read_trigger_record, trigger_binding)
def gate_code(rel, sha=None) -> str | None:
    fx_repo()
    if sha is None:
        p = SELREPO / str(rel)
        sha = sha_file(p) if p.is_file() else "f" * 64
    return code_of(T.read_trigger_record, rel, sha, repo_root=SELREPO)


@case("gate")
def trigger_record_accepted_when_committed():
    fx_repo()
    ok = T.read_trigger_record("trig/ok.json", sha_file(SELREPO / "trig/ok.json"), repo_root=SELREPO)
    code, doc, inp = decided("gate_decided", e3=0.4001)          # a record decide wrote, committed alone
    shutil.copy2(inp.out, SELREPO / "trig/decided.json")
    git(SELREPO, "add", "--", "trig/decided.json")
    git(SELREPO, "commit", "-q", "-m", "the trigger record, alone")
    got = T.read_trigger_record("trig/decided.json", sha_file(SELREPO / "trig/decided.json"), repo_root=SELREPO)
    check("trigger_record_accepted_when_committed", ok["record"]["triggered"] is True and code is None
          and got["record"] == doc and got["path"] == "trig/decided.json", f"[{code}]")


for _name, _args, _want in [
    ("refuses_trigger_flag_missing", (None, "a" * 64), "trigger_missing"),
    ("refuses_trigger_sha256_format", ("trig/ok.json", "abc"), "trigger_sha256_format"),
    ("refuses_trigger_path_not_repo_relative", (str(SELREPO / "trig/ok.json"),), "trigger_path_not_repo_relative"),
    ("refuses_trigger_forbidden_path", ("trig/latest.json", "a" * 64), "trigger_forbidden_path"),
    ("refuses_trigger_file_missing", ("trig/none.json", "a" * 64), "trigger_missing"),
    ("refuses_trigger_untracked", ("trig/untracked.json",), "trigger_not_tracked"),
    ("refuses_trigger_changed_since_head", ("trig/changed.json",), "trigger_changed_since_head"),
    ("refuses_trigger_sha256_mismatch", ("trig/ok.json", "f" * 64), "trigger_sha256_mismatch"),
    ("refuses_trigger_not_json", ("trig/strict.json",), "trigger_format"),
    ("refuses_trigger_wrong_format", ("trig/fmt.json",), "trigger_format"),
    ("refuses_trigger_smoke_inputs", ("trig/smoke.json",), "trigger_smoke_inputs"),
    ("refuses_trigger_seed", ("trig/seed43.json",), "trigger_seed"),
    ("refuses_trigger_inconsistent", ("trig/inconsistent.json",), "trigger_inconsistent"),
    ("refuses_trigger_not_triggered", ("trig/not.json",), "trigger_not_triggered"),
]:
    refusal_case("gate", _name, functools.partial(lambda a: gate_code(*a), _args), _want)


def binding_code(**over) -> str | None:
    fx_repo()
    trig = T.read_trigger_record("trig/ok.json", sha_file(SELREPO / "trig/ok.json"), repo_root=SELREPO)
    kw = dict(parent=fx_parent(), clip_norm=1.0, clip_selection_sha256=CLIP_SHA,
              lam=selection_record("sel/lambda.json", "lambda"), alpha=selection_record("sel/alpha.json", "alpha"))
    kw.update(over)
    return code_of(T.trigger_binding, trig, **kw)


refusal_case("gate", "trigger_binding_accepted", lambda: binding_code(), None)
refusal_case("gate", "refuses_trigger_e3_mismatch",
             lambda: binding_code(parent={**fx_parent(), "checkpoint_sha256": "9" * 64}), "trigger_e3_mismatch")
refusal_case("gate", "refuses_trigger_clip_mismatch", lambda: binding_code(clip_norm=5.0), "trigger_clip_mismatch")
refusal_case("gate", "refuses_trigger_selection_mismatch",
             lambda: binding_code(alpha={"sha256": "9" * 64}), "trigger_selection_mismatch")


# ---- e2e: score, then decide, through the CLI, on the real evaluator
@contextlib.contextmanager
def driver_env():
    with patched_env(SMOKE_MAX_SAMPLES=E2E_SAMPLES, SMOKE_GIT_HEAD=T.code_identity()["git_head"] or "0" * 40):
        yield


@functools.cache
def fx_e2e() -> dict:
    """Block B with epoch 1 converted for real and scored by the evaluator; then score and decide through the CLI."""
    run, sha = fx_e6()["dir"], fx_e6()["sha"]
    rec = A.read_run_record(run)
    ev = TMP / "ev_s42_e2e"
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "qat_epoch_eval.py"), "convert-epoch",
                        "--run-dir", str(run), "--eval-dir", str(ev), "--epoch", "1", "--purpose", "record",
                        "--host-label", HOST, "--expect-telemetry-sha256", sha], cwd=str(REPO), capture_output=True,
                       text=True, timeout=2400)
    if "RESULT: QAT EPOCH 01 CONVERTED" not in p.stdout:
        raise RuntimeError(f"convert-epoch: {p.stdout[-300:]} {p.stderr[-500:]}")
    names = A.epoch_names(1)
    prov_p = ev / A.CONVERTED_DIR / names["run_meta"]
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    ck_sha = rec["ends"][1]["checkpoint_sha256"]
    run_id = f"{rec['run_id']}_e01_{ck_sha[:12]}"
    (ev / A.SCORES_DIR).mkdir()
    with patched_attr(QEE, "EVALUATOR_SCRIPT", str(DRIVER)), driver_env():
        argv = QEE.evaluator_command("E6", prov_p, ev / A.SCORES_DIR / "e01", run_id, "record")
        p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True, timeout=2400)
    summ = ev / A.SCORES_DIR / "e01" / "summary.json"
    if p.returncode != 0 or not summ.is_file():
        raise RuntimeError(f"block B evaluator exited {p.returncode}: {p.stderr[-500:]}")
    s0 = json.loads(summ.read_text(encoding="utf-8"))
    v1 = s0["dataset_level"]["all_class_miou"]
    values = {e: (v1 if e == 1 else v1 / 2) for e in range(1, 16)}
    fab_block_b(run, ev, values, real={"epoch": 1, "summary": s0, "provenance": prov, "argv": argv, "run_id": run_id})
    sel_sha = select(run, ev, sha, smoke=True)
    tdir = TMP / "trig_e2e"
    inp = inputs(tdir, out=tdir / "e6kd_trigger.json", ev=ev, sel_sha=sel_sha, smoke=True)
    real_score, real_decide = T.score, T.decide
    with patched_attr(T, "score", functools.partial(real_score, repo_root=SELREPO)), \
            patched_attr(T, "decide", functools.partial(real_decide, repo_root=SELREPO)), \
            patched_attr(T, "EVALUATOR_SCRIPT", str(DRIVER)), driver_env():
        rc_s, res_s = run_cli(argv_of(inp, "score"))
        rc_d, res_d = run_cli(argv_of(inp, "decide"))
    return {"inp": inp, "tdir": tdir, "ev": ev, "e6_value": v1, "ck_sha": ck_sha, "rc_s": rc_s, "res_s": res_s,
            "rc_d": rc_d, "res_d": res_d}


@case("e2e")
def score_runs_the_literal_evaluator_commands():
    fx = fx_e2e()
    tev_p = fx["tdir"] / T.TRIGGER_EVAL_RECORD
    tev = json.loads(tev_p.read_text(encoding="utf-8")) if tev_p.is_file() else {}
    sc = tev.get("scores") or {}
    e3 = fx_e3()
    tail = ["--artifact-status", "provisional", "--device", "cpu", "--batch-size", "1"]
    want3 = ["-B", str(DRIVER), "--stage", "E3", "--model-role", "student", "--precision", "fp32", "--split", "val",
             "--protocol", "canvas", "--checkpoint", str((e3["dir"] / E3_CKPT).resolve()), *tail,
             "--out-dir", str(fx["tdir"].resolve() / "scores" / "e3"), "--run-id", f"e6kd_trigger_e3_{e3['sha'][:12]}"]
    want6 = ["-B", str(DRIVER), "--stage", "E6", "--model-role", "student", "--precision", "int8_qat", "--split",
             "val", "--protocol", "canvas", "--provenance",
             str(fx["ev"].resolve() / "converted" / "e01_run_meta.json"), *tail,
             "--out-dir", str(fx["tdir"].resolve() / "scores" / "e6"),
             "--run-id", f"e6kd_trigger_e6_e01_{fx['ck_sha'][:12]}"]
    got3, got6 = (sc.get("e3") or {}).get("evaluator_command"), (sc.get("e6") or {}).get("evaluator_command")
    check("score_runs_the_literal_evaluator_commands", got3 == want3 and got6 == want6
          and T.EVALUATOR_SCRIPT == "scripts/evaluate_model.py" and (sc.get("e3") or {}).get("exit_code") == 0
          and (sc.get("e6") or {}).get("exit_code") == 0,
          f"e3 {got3 == want3}, e6 {got6 == want6}; {fx['res_s'][:80]!r}")


@case("e2e")
def score_then_decide_end_to_end():
    fx = fx_e2e()
    out = Path(fx["inp"].out)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    s3_p = fx["tdir"] / T.SCORE_DIRS["e3"] / "summary.json"
    e3 = json.loads(s3_p.read_text(encoding="utf-8"))["dataset_level"]["all_class_miou"] if s3_p.is_file() else None
    dec = T.drop_decision(doc["e3_val"], doc["e6_val"]) if doc else {}
    word = ("RESULT: TRIGGERED (smoke inputs, not a record) -- " if dec.get("triggered")
            else "RESULT: NOT TRIGGERED (smoke inputs, not a record) -- ")
    check("score_then_decide_end_to_end", fx["rc_s"] == 0 and fx["res_s"] == "RESULT: SCORED 2/2 (smoke inputs)"
          and fx["rc_d"] == 0 and fx["res_d"].startswith(word) and doc.get("e6_val") == fx["e6_value"]
          and doc.get("e3_val") == e3 and doc.get("triggered") is dec.get("triggered")
          and doc.get("smoke_inputs") is True and (doc.get("e6") or {}).get("rescore_equals_selection") is True
          and (doc.get("e6") or {}).get("epoch") == 1,
          f"score exit {fx['rc_s']} {fx['res_s'][:60]!r}; decide exit {fx['rc_d']} {fx['res_d'][:90]!r}")


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
    ap.add_argument("--cache-dir", default=None, help="keep (or reuse) the E3 checkpoint and the raw E6 runs here")
    args = ap.parse_args()
    want = set(args.sections.split(","))
    names = [n for n, _, _ in CASES]
    if want - set(SECTIONS) or len(set(names)) != len(names):
        print(f"RESULT: ERROR unknown sections {sorted(want - set(SECTIONS))} or duplicate case names")
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
    print("E6-KD TRIGGER SMOKE (lane L-AM3, B-1) -- synthetic runs, fabricated block B and trigger eval dirs")
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
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n[time] {time.time() - t_all:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
