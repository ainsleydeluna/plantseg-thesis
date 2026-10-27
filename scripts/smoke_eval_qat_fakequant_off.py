#!/usr/bin/env python3
"""Lane L-FQOFF: AM-4's fake-quant-off score (precision qat_fakequant_off), on synthetic inputs.

Phase 1 (rulings FQR1-FQR3, FQR9, FQR12, FQR14): the precision's vocabulary and scope (src/eval/artifacts.py
PRECISIONS, FQOFF_STAGES and its request rule), the binding (src/eval/stage_artifacts.py
validate_qat_fakequant_off_source, and resolve_evaluation_source's route to it), the loader
(src/eval/model_loading.py load_qat_fakequant_off_student) and the FP32 reference (fp32_reference_from_qat_state).
Every refusal code of the binding and the loader (FQR12's 6-22) is reached with every earlier gate passing.
Phase 2 (FQR11, FQR13 and LEFF B-1b's flags): scripts/evaluate_model.py's path for the precision, its CLI rules
(FQR12's 1-5), the run end to end on a synthetic VAL split (make_tree, through scripts/synthetic_qat_fixtures.py),
the summary's fields, the order of the refusals around the dataset adapter, and the modules a run loads (T-FQ8).

The QAT runs are real runs of the trainers in mode "smoke" on Q2's colour task (scripts/synthetic_qat_fixtures.py):
E5 and AM-23's E6-E2 through src/quant/qat.py run_qat (make_qat_run), E6-KD through src/quant/qat_kd.py run_qat_kd
with the invariance harness's stub teacher, as scripts/smoke_e6kd_selection.py trains it. Each run is re-labelled
mode "real" (its checkpoints untouched); its eval directory is fabricated as scripts/smoke_qat_selection.py
fabricates it (summaries holding chosen values, every hash consistent; mirrored here, since that module makes a
temp folder and sets PLANTSEG_IMAGE_DIGEST when imported); and Q2's scripts/select_qat_epoch.py makes the selection
of record under the committed rules file of the stage. A refusal case edits copies: a changed file is rebound (its
epoch_end row's sha256, the telemetry's and the selection's, and their pins) wherever an earlier gate would
otherwise refuse first.

Sections: stages, binding, loader, parity, finder, never_selects (phase 1); cli, run, summary, order, imports
(phase 2); hygiene.

No PlantSeg data, no checkpoint of record, no GPU; every file in a temp directory outside the repository. No check
name, case name, code, file or folder holds the token SL-1 forbids, and no path is derived from a check name; the
forbidden-path cases name paths that are never created.

    python -B scripts/smoke_eval_qat_fakequant_off.py [--only NAME,...] [--cache-dir DIR]

--cache-dir keeps the three raw QAT runs, trained at the start when it lacks them and cloned after. Ends with one
RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import repo_state, state_verdict  # noqa: E402
from scripts.synthetic_qat_fixtures import (color_task, confident_student, make_qat_run, make_tree,  # noqa: E402
                                            safe_tmpdir, task_loaders)

TMP = safe_tmpdir("smoke_fqoff_")
os.environ["PLANTSEG_IMAGE_DIGEST"] = "sha256:" + "5" * 64       # the runs record a non-null digest
DATA = TMP / "data"                                             # the run section's synthetic VAL split
os.environ["PLANTSEG_DATA_ROOT"] = str(DATA)                    # before configs/data.py is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import scripts.evaluate_model as EM  # noqa: E402
import scripts.select_qat_epoch as SE  # noqa: E402
from src.eval import artifacts as ART  # noqa: E402
from src.eval import model_loading as ML  # noqa: E402
from src.eval import stage_artifacts as SA  # noqa: E402
from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME  # noqa: E402
from src.eval.eval_runtime import EvalRuntimeError  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_select as S  # noqa: E402

results: list[tuple[str, bool, str]] = []
CASES: list[tuple[str, str, object]] = []          # (name, section, fn): each case reports one check, of its name
SECTIONS = ("stages", "binding", "loader", "parity", "finder", "never_selects", "cli", "run", "summary", "order",
            "imports", "hygiene")
OPTS: dict = {"cache": None, "before": None}
FQOFF = "qat_fakequant_off"
CARRIER_KEYS = {"spec", "checkpoint_path", "checkpoint_sha256", "epoch", "run_id", "selection_sha256",
                "telemetry_sha256", "finalize_sha256"}
# FQR12's codes 6-22, in order: the binding's (stage_artifacts.py) and the loader's (model_loading.py)
PHASE1_CODES = ("fqoff_forbidden_path", "fqoff_selection_sha256_mismatch", "fqoff_selection_not_of_record",
                "fqoff_stage_mismatch", "fqoff_run_record", "fqoff_run_incomplete", "fqoff_run_not_real",
                "fqoff_telemetry_sha256_mismatch", "fqoff_finalize_disagrees", "fqoff_not_the_winner",
                "fqoff_checkpoint_missing", "fqoff_checkpoint_sha256_mismatch", "fqoff_not_qat_train_state",
                "fqoff_checkpoint_identity", "fqoff_state_not_finite", "fqoff_load_not_exact",
                "fqoff_fake_quant_enabled")
# FQR12's codes 1-5, the CLI's (scripts/evaluate_model.py, phase 2); with phase 1's, all 22 in FQR12's order
PHASE2_CODES = ("fqoff_stage_not_in_scope", "fqoff_provenance_forbidden", "fqoff_source_flags", "fqoff_not_cpu",
                "fqoff_protocol")
ALL_CODES = PHASE2_CODES + PHASE1_CODES
# FQR14: the parity tolerance, pinned once from the first measurement: the smallest of 1e-6, 1e-5, 1e-4 and 1e-3 at
# least four times the measured maximum |logit difference| (8.583e-06 on lane/fqoff's first run, four times that
# 3.43e-05); docs/lane_reports/fqoff.md gives both values
PARITY_TOL = 1e-4
GT_SUM = 159_279_104
TOKEN = "".join(("t", "e", "s", "t"))                         # the token SL-1 forbids, never written as a name


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def case(section: str):
    def deco(fn):
        CASES.append((fn.__name__, section, fn))
        return fn
    return deco


def quiet(*_a, **_k) -> None:
    return None


def run_cli(main, argv) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    return rc, (lines[-1] if lines else buf.getvalue()[-300:])


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# ---------------------------------------------------------------- every file opened, while recording
OPENS: dict = {"on": False, "paths": []}


def _audit(event, args) -> None:
    if event == "open" and OPENS["on"] and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENS["paths"].append(os.fsdecode(args[0]))


sys.addaudithook(_audit)


@contextlib.contextmanager
def recording():
    OPENS["paths"] = []
    OPENS["on"] = True
    try:
        yield OPENS["paths"]
    finally:
        OPENS["on"] = False


def fixture_opens(paths) -> list[str]:
    """The opened paths that are this smoke's fixtures or a rules file (imports and the like left out)."""
    rules = {str(REPO / rel) for rel in S.RULES_FILES}
    return sorted({p for p in paths if p.startswith(str(TMP)) or p in rules})


# ---------------------------------------------------------------- the raw QAT runs (kept raw with --cache-dir)
RAW = {"run_e5": "E5", "run_e6e2": "E6-E2", "run_e6kd": "E6-KD"}
SOURCE_META = {"sha256": "0" * 64, "path": None, "bytes": None}
# E6-KD's smoke inputs, as scripts/smoke_e6kd_selection.py gives them
KD_WEIGHTS = {"logit_kd": 0.25, "cwd_feat": 12.5, "cwd_logit": 1.5}
KD_E3 = {"e3_weights": {"logit_kd": 0.5, "cwd_feat": 25.0, "cwd_logit": 3.0},
         "grids": {"logit_kd_grid": "smoke", "cwd_feat_grid": "smoke", "cwd_logit_grid": "smoke"}}


def clone_run(src: Path, dst: Path, *, skip=()) -> Path:
    """A copy of a run: checkpoints hard-linked (a forged one replaces its link, never writes through it), the record
    copied."""
    (dst / Q.EPOCH_DIR).mkdir(parents=True)
    for p in sorted((src / Q.EPOCH_DIR).iterdir()):
        if p.name not in skip:
            os.link(p, dst / Q.EPOCH_DIR / p.name)
    shutil.copy2(src / Q.TELEMETRY_NAME, dst / Q.TELEMETRY_NAME)
    return dst


@functools.lru_cache(maxsize=None)
def source_state() -> dict:
    return {k: v.clone() for k, v in confident_student(seed=0, steps=30).state_dict().items()}


def source_student() -> nn.Module:
    from src.models.student import build_student
    m = build_student(pretrained=False)
    m.load_state_dict(source_state())
    return m.eval()


def projection_fixture(seed: int = 7) -> nn.Conv2d:
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.utils.skip_init(nn.Conv2d, 160, 320, 1, bias=False)
    with torch.no_grad():
        p.weight.copy_(torch.randn(320, 160, 1, 1, generator=g) * 0.05)
    return p


def train(name: str, dst: Path) -> None:
    if name == "run_e6kd":
        from scripts.invariance_harness import build_stub_segnext
        from src.distill import teacher as TT
        from src.distill.segnext_teacher import SegNeXtTeacherAdapter
        from src.quant import qat_kd as KD
        teacher = TT.FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext()))
        KD.run_qat_kd(mode="smoke", model=source_student(), teacher=teacher, projection=projection_fixture(),
                      weights=dict(KD_WEIGHTS), e3=KD_E3, source_meta=SOURCE_META, out_dir=dst, seed=42,
                      clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0, loaders=task_loaders(42),
                      log=quiet)
    else:
        make_qat_run(dst, model=source_student(), stage={"run_e5": "e5", "run_e6e2": "e6-e2"}[name], seed=42,
                     clip=1.0)


@functools.lru_cache(maxsize=None)
def raw_run(name: str) -> Path:
    """A raw 15-epoch smoke run: trained once, or kept raw in --cache-dir and never edited there."""
    cache = OPTS["cache"]
    if cache is not None and (cache / name / "READY").is_file():
        return cache / name
    dst = TMP / "raw" / name
    train(name, dst)
    if cache is not None:
        clone_run(dst, cache / name)
        (cache / name / "READY").write_text("raw smoke run\n", encoding="utf-8")
    return dst


def fill_cache() -> None:
    """--cache-dir: every raw run at once and before any case, so that later jobs (the mutation harness's) only
    clone them."""
    for name in RAW:
        raw_run(name)


def relabel(run: Path) -> str:
    """The run's record re-labelled mode "real", a launch bound to clip_selection.json (the checkpoints untouched).
    Returns the new telemetry sha256."""
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    rows[0].update({"mode": "real", "u4_pilot": False, "clip_source": "clip_selection"})
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Q.sha256_file(tel)


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


def fabricate(run: Path, ev: Path, values: dict) -> None:
    """converted/, scores/, qat_convert.json and qat_epoch_eval.json for `run`, scored with `values`
    (scripts/smoke_qat_selection.py fabricate, :164-234, without its tampering options)."""
    commit, host_label = "a" * 40, "fab-host"
    rec = A.read_run_record(run)
    meta = rec["run_meta"]
    conv, scores = ev / A.CONVERTED_DIR, ev / A.SCORES_DIR
    conv.mkdir(parents=True)
    outcomes, rows = [], []
    for e in range(1, 16):
        names = A.epoch_names(e)
        ck_sha = rec["ends"][e]["checkpoint_sha256"]
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


# ---------------------------------------------------------------- the bindings
@dataclass(frozen=True)
class Bound:
    """One binding: a run, its eval directory with the selection of record, and their pins."""
    stage: str
    run: Path
    ev: Path
    sel: Path
    sel_sha: str
    tel_sha: str
    epoch: int

    @property
    def ck(self) -> Path:
        return self.run / Q.EPOCH_DIR / f"e{self.epoch:02d}.pt"

    def kw(self, **over) -> dict:
        d = {"run_dir": self.run, "selection": self.sel, "selection_sha256": self.sel_sha,
             "telemetry_sha256": self.tel_sha}
        d.update(over)
        return d


@functools.lru_cache(maxsize=None)
def bound(raw: str = "run_e5", winner: int = 7, smoke: bool = False) -> Bound:
    """A run re-labelled real (or left in smoke mode), its eval directory fabricated with `winner` the best epoch, and
    Q2's scripts/select_qat_epoch.py's selection of it (with --allow-smoke-inputs on a smoke run)."""
    tag = f"{raw}_w{winner:02d}" + ("_smoke" if smoke else "")
    run = clone_run(raw_run(raw), TMP / "bound" / tag / "run")
    tel_sha = Q.sha256_file(run / Q.TELEMETRY_NAME) if smoke else relabel(run)
    ev = TMP / "bound" / tag / "eval"
    values = {e: 0.30 for e in range(1, 16)}
    values[winner] = 0.40
    fabricate(run, ev, values)
    argv = ["--run-dir", str(run), "--eval-dir", str(ev), "--expect-telemetry-sha256", tel_sha]
    rc, res = run_cli(SE.main, argv + (["--allow-smoke-inputs"] if smoke else []))
    sel = ev / "qat_selection.json"
    if rc != 0 or json.loads(sel.read_text(encoding="utf-8"))["winner"]["epoch"] != winner:
        raise RuntimeError(f"scripts/select_qat_epoch.py on {tag}: rc {rc}, {res}")
    return Bound(RAW[raw], run, ev, sel, Q.sha256_file(sel), tel_sha, winner)


def payload_bytes(b: Bound, edit) -> bytes:
    """The winner's eNN.pt payload, loaded weights_only, edited and saved again."""
    p = torch.load(b.ck, map_location="cpu", weights_only=True)
    edit(p)
    buf = io.BytesIO()
    torch.save(p, buf)
    return buf.getvalue()


def forge(b: Bound, tag: str, *, ck_bytes: bytes | None = None, rebind_ck: bool = True, rows=None, sel=None,
          sel_name: str = "qat_selection.json", finalize=None, skip_ck: bool = False) -> Bound:
    """A copy of b's binding with one change, every earlier link rebound: a new eNN.pt's sha256 in its epoch_end row
    and the selection's winner (unless rebind_ck is False), the telemetry's sha256 in the selection and the pin, the
    selection's sha256 in the pin and the finalize record. `rows` and `sel` edit the telemetry rows and the selection;
    `finalize` writes a qat_finalize.json from (selection, its sha256)."""
    d = TMP / "forged" / tag
    run = clone_run(b.run, d / "run", skip=(b.ck.name,) if skip_ck else ())
    ck = run / Q.EPOCH_DIR / b.ck.name
    new_sha = None
    if ck_bytes is not None:
        tmp = ck.with_name(ck.name + ".new")
        tmp.write_bytes(ck_bytes)
        os.replace(tmp, ck)                                 # the hard link is replaced; its target is never written
        new_sha = Q.sha256_bytes(ck_bytes) if rebind_ck else None
    tel = run / Q.TELEMETRY_NAME
    if new_sha is not None or rows is not None:
        rws = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
        for r in rws:
            if new_sha is not None and r.get("event") == "epoch_end" and r.get("epoch") == b.epoch:
                r["checkpoint_sha256"] = new_sha
        if rows is not None:
            rows(rws)
        tel.write_text("".join(json.dumps(r) + "\n" for r in rws), encoding="utf-8")
    tel_sha = Q.sha256_file(tel)
    ev = d / "eval"
    ev.mkdir(parents=True)
    doc = json.loads(b.sel.read_text(encoding="utf-8"))
    doc["telemetry_sha256"] = tel_sha
    if new_sha is not None:
        doc["winner"]["checkpoint_sha256"] = new_sha
    if sel is not None:
        sel(doc)
    sp = ev / sel_name
    sp.write_bytes(A.json_bytes(doc))
    sel_sha = Q.sha256_file(sp)
    if finalize is not None:
        (ev / SA.QAT_FINALIZE_NAME).write_bytes(finalize(doc, sel_sha))
    return Bound(b.stage, run, ev, sp, sel_sha, tel_sha, b.epoch)


def finalize_record(doc: dict, sel_sha: str, **over) -> bytes:
    """A qat_finalize/1 record of the selection `doc` (scripts/qat_epoch_eval.py cmd_finalize's keys, :485-499, the x86
    copy left out), with `over` replacing top-level keys or, under "checkpoint", the artifact's checkpoint."""
    w = doc["winner"]
    fin = {"format": "qat_finalize/1", "stage": doc["stage"], "run_id": doc["run_id"], "seed": doc["seed"],
           "clip_norm": doc["clip_norm"], "telemetry_sha256": doc["telemetry_sha256"],
           "qat_selection": {"path": "qat_selection.json", "sha256": sel_sha}, "clip_selection": None,
           "artifact_of_record": {"epoch": w["epoch"], "value": w["value"], "torchscript": w["artifact_of_record"],
                                  "provenance": w["provenance"], "summary": w["summary"],
                                  "checkpoint": {"path": w["checkpoint"], "sha256": w["checkpoint_sha256"]}},
           "smoke_inputs": False, "status": "finalized"}
    if "checkpoint" in over:
        fin["artifact_of_record"]["checkpoint"] = over.pop("checkpoint")
    fin.update(over)
    return A.json_bytes(fin)


def bind(b: Bound, *, stage: str | None = None, checkpoint=None, smoke: bool = False, **over) -> dict:
    return SA.validate_qat_fakequant_off_source(stage or b.stage, b.ck if checkpoint is None else checkpoint,
                                                allow_smoke_inputs=smoke, **b.kw(**over))


def code_of(fn, *a, **kw) -> tuple[str | None, str]:
    try:
        fn(*a, **kw)
    except SA.StageArtifactError as e:
        return e.code, str(e)
    except EvalRuntimeError as e:
        return e.code, str(e)
    return None, "accepted"


def refused(code: str, fn, *a, **kw) -> tuple[bool, str]:
    got, msg = code_of(fn, *a, **kw)
    return got == code, f"{got}: {msg[:240]}"


def other_epoch_bytes(b: Bound, epoch: int) -> bytes:
    return (b.run / Q.EPOCH_DIR / f"e{epoch:02d}.pt").read_bytes()


def torchscript_bytes() -> bytes:
    """A TorchScript archive (torch.jit.save) carrying a plantseg_int8.json, as a converted artifact does."""
    m = torch.jit.trace(nn.Sequential(nn.Conv2d(3, 4, 1)).eval(), torch.zeros(1, 3, 8, 8))
    buf = io.BytesIO()
    torch.jit.save(m, buf, _extra_files={"plantseg_int8.json": json.dumps({"stage": "E5", "quantization": "qat"})})
    return buf.getvalue()


def never(*parts: str) -> Path:
    """A path under this smoke's temp directory that is never created."""
    return TMP.joinpath("never_created", *parts)


# ================================================================ stages (FQR1, FQR3)
def request(**run_over):
    """An artifact request of precision qat_fakequant_off (E5, student, no backend, trained); class_map empty, so a
    request the precision rule passes is refused next by the class-map check, before any I/O."""
    from src.eval import Condition, DatasetMeta, RunMeta
    run = dict(stage="E5", model_role="student", precision=FQOFF, quant_backend=None, checkpoint_path="e07.pt",
               checkpoint_sha256="0" * 64, random_init=False, device="cpu")
    run.update(run_over)
    return ART.prepare_artifact_request(
        out_dir=never("artifact"), artifact_status="provisional", run_id="r", run=RunMeta(**run),
        dataset=DatasetMeta(name="PlantSeg", doi="10.5281/zenodo.13958858", split="val", condition=Condition(),
                            preprocess_protocol="core_preprocess/1.0.0", expected_rows=0),
        expected_manifest=[], class_map=[])


def request_error(**run_over) -> str:
    try:
        ART.validate_artifact_request(request(**run_over))
    except ART.ArtifactRequestError as e:
        return str(e)
    return "accepted"


RULE_TEXT = f"precision {FQOFF!r} requires a stage in"


@case("stages")
def stages_fqoff_stages_defined_once():
    """FQOFF_STAGES is artifacts.py's tuple, assigned in no other evaluator file (FQR1)."""
    others = []
    for rel in ("src/eval/stage_artifacts.py", "src/eval/model_loading.py", "scripts/evaluate_model.py"):
        for n in ast.walk(ast.parse((REPO / rel).read_text(encoding="utf-8"))):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
            others += [rel for t in targets if isinstance(t, ast.Name) and t.id == "FQOFF_STAGES"]
    return ART.FQOFF_STAGES == ("E5", "E6", "E6-E2", "E6-KD") and not others, f"{ART.FQOFF_STAGES}; elsewhere {others}"


@case("stages")
def stages_fqoff_stages_are_qat_int8():
    rows = {s: SA.STAGE_ARTIFACTS.get(s) for s in ART.FQOFF_STAGES}
    ok = all(r is not None and (r["kind"], r["precision"], r["method"]) == ("int8_artifact", "int8_qat", "qat")
             for r in rows.values()) and set(ART.FQOFF_STAGES) <= set(ART.STAGES)
    return ok, str(rows)


@case("stages")
def stages_cut_stages_left_out():
    qat = sorted(s for s, r in SA.STAGE_ARTIFACTS.items() if r["method"] == "qat")
    left = sorted(set(qat) - set(ART.FQOFF_STAGES))
    return left == ["E6-R1", "E6-R2"] and set(ART.FQOFF_STAGES) <= set(qat), f"QAT stages {qat}; left out {left}"


@case("stages")
def stages_precision_in_schema():
    p = ART.PRECISIONS
    return p[:3] == ("fp32", "int8_ptq", "int8_qat") and p.count(FQOFF) == 1, str(p)


@case("stages")
def stages_rule_refuses_cut_stage():
    msg = request_error(stage="E6-R1")
    return RULE_TEXT in msg, msg[:240]


@case("stages")
def stages_rule_refuses_fp32_stage():
    msg = request_error(stage="E1")
    return RULE_TEXT in msg, msg[:240]


@case("stages")
def stages_rule_refuses_teacher_role():
    msg = request_error(stage="E5", model_role="teacher")
    return RULE_TEXT in msg, msg[:240]


@case("stages")
def stages_rule_refuses_backend():
    msg = request_error(quant_backend="qnnpack")
    return RULE_TEXT in msg, msg[:240]


@case("stages")
def stages_rule_refuses_random_init():
    msg = request_error(random_init=True, checkpoint_path=None, checkpoint_sha256=None)
    return RULE_TEXT in msg, msg[:240]


@case("stages")
def stages_rule_accepts_each_fqoff_stage():
    msgs = {s: request_error(stage=s) for s in ART.FQOFF_STAGES}
    ok = all(RULE_TEXT not in m and "class_map must describe all 116 classes" in m for m in msgs.values())
    return ok, str({s: m[:60] for s, m in msgs.items()})


@case("stages")
def stages_config_payload_hashes_precision():
    a = ART.build_config_payload(request())
    b = ART.build_config_payload(request(precision="int8_qat", quant_backend="qnnpack"))
    diff = sorted(k for k in a if a[k] != b[k])
    sha = [ART._sha256_bytes(ART.canonical_json_bytes(x)) for x in (a, b)]
    return diff == ["precision", "quant_backend"] and sha[0] != sha[1], f"differ in {diff}; {sha[0][:12]} {sha[1][:12]}"


# ================================================================ binding (FQR2, FQR12 codes 6-20)
def rules_sha(stage: str) -> str:
    return Q.sha256_file(S.rules_path_for_stage(stage))


def carrier_ok(b: Bound, c: dict) -> tuple[bool, str]:
    sel = json.loads(b.sel.read_text(encoding="utf-8"))
    ok = (set(c) == CARRIER_KEYS and c["spec"]["stage"] == b.stage and c["checkpoint_path"] == b.ck
          and c["checkpoint_sha256"] == sel["winner"]["checkpoint_sha256"] == Q.sha256_file(b.ck)
          and c["epoch"] == b.epoch and c["run_id"] == sel["run_id"] and c["selection_sha256"] == b.sel_sha
          and c["telemetry_sha256"] == b.tel_sha and sel["rules_sha256"] == rules_sha(b.stage))
    return ok, (f"{b.stage} e{c['epoch']:02d} {c['checkpoint_sha256'][:12]}; rules "
                f"{S.rules_path_for_stage(b.stage).name} {sel['rules_sha256'][:12]}; finalize {c['finalize_sha256']}")


@case("binding")
def bind_e5_selection_of_record():
    b = bound("run_e5")
    c = bind(b)
    ok, detail = carrier_ok(b, c)
    return ok and c["finalize_sha256"] is None, detail


@case("binding")
def bind_e6e2_rules_am23():
    b = bound("run_e6e2")
    c = bind(b)
    ok, detail = carrier_ok(b, c)
    return ok and S.rules_path_for_stage("E6-E2").name == "qat_selection_rules_am23.json", detail


@case("binding")
def bind_e6kd_rules_e6kd():
    b = bound("run_e6kd")
    c = bind(b)
    ok, detail = carrier_ok(b, c)
    return ok and S.rules_path_for_stage("E6-KD").name == "e6kd_selection_rules.json" \
        and c["finalize_sha256"] is None, detail


@case("binding")
def bind_finalize_agreeing():
    b = forge(bound("run_e5"), "fin_agreeing", finalize=finalize_record)
    c = bind(b)
    fin = Q.sha256_file(b.ev / SA.QAT_FINALIZE_NAME)
    return c["finalize_sha256"] == fin and c["selection_sha256"] == b.sel_sha, f"finalize {c['finalize_sha256']}"


@case("binding")
def bind_reads_only_its_files():
    """The telemetry, the selection, the rules file of its stage, the finalize record and the winner's eNN.pt; no other
    epoch's file."""
    b = forge(bound("run_e5"), "reads", finalize=finalize_record)
    with recording() as paths:
        bind(b)
    got = fixture_opens(paths)
    want = sorted({str(b.run / Q.TELEMETRY_NAME), str(b.sel), str(S.rules_path_for_stage("E5")),
                   str(b.ev / SA.QAT_FINALIZE_NAME), str(b.ck)})
    return got == want, f"opened {[Path(p).name for p in got]}"


@case("binding")
def bind_smoke_status_accepts_smoke_inputs():
    b = bound("run_e5", smoke=True)
    c = bind(b, smoke=True)
    sel = json.loads(b.sel.read_text(encoding="utf-8"))
    return sel["smoke_inputs"] is True and c["epoch"] == b.epoch, f"smoke_inputs {sel['smoke_inputs']}"


@case("binding")
def bind_resolve_route_same_carrier():
    b = bound("run_e5")
    c = SA.resolve_evaluation_source("E5", checkpoint=b.ck, qat_run_dir=b.run, qat_selection=b.sel,
                                     qat_selection_sha256=b.sel_sha, qat_telemetry_sha256=b.tel_sha)
    return c == bind(b), f"{sorted(c)}"


@case("binding")
def bind_resolve_old_routes_unchanged():
    got = {"E5 without provenance": code_of(SA.resolve_evaluation_source, "E5")[0],
           "E1 without checkpoint": code_of(SA.resolve_evaluation_source, "E1")[0],
           "teacher without checkpoint": code_of(SA.resolve_evaluation_source, "teacher")[0]}
    want = {"E5 without provenance": "provenance_required", "E1 without checkpoint": "checkpoint_required",
            "teacher without checkpoint": "checkpoint_required"}
    return got == want, str(got)


@case("binding")
def bind_fp32_route_refuses_qat_state():
    return refused("checkpoint_is_quantized", SA.validate_fp32_artifact, "E1", bound("run_e5").ck)


# ---- 6 fqoff_forbidden_path
def forbidden(**over) -> tuple[bool, str]:
    b = bound("run_e5")
    with recording() as paths:
        ok, detail = refused("fqoff_forbidden_path", bind, b, **over)
    opened = fixture_opens(paths)
    return ok and not opened, f"{detail}; opened {opened}"


@case("binding")
def c06_forbidden_checkpoint():
    return forbidden(checkpoint=never("nowhere_latest_ck", "e07.pt"))


@case("binding")
def c06_forbidden_run_dir():
    return forbidden(run_dir=never("nowhere_latest_run"))


@case("binding")
def c06_forbidden_selection():
    return forbidden(selection=never("nowhere_latest_eval", "qat_selection.json"))


# ---- 7 fqoff_selection_sha256_mismatch
@case("binding")
def c07_selection_absent():
    return refused("fqoff_selection_sha256_mismatch", bind, bound("run_e5"),
                   selection=never("eval", "qat_selection.json"))


@case("binding")
def c07_selection_pin_differs():
    return refused("fqoff_selection_sha256_mismatch", bind, bound("run_e5"), selection_sha256="0" * 64)


# ---- 8 fqoff_selection_not_of_record
@case("binding")
def c08_selection_not_named_of_record():
    b = forge(bound("run_e5"), "sel_renamed", sel_name="qat_selection_copy.json")
    return refused("fqoff_selection_not_of_record", bind, b)


@case("binding")
def c08_selection_not_json():
    b = bound("run_e5")
    d = TMP / "forged" / "sel_not_json"
    d.mkdir(parents=True)
    (d / "qat_selection.json").write_bytes(b"not a selection\n")
    return refused("fqoff_selection_not_of_record", bind, b, selection=d / "qat_selection.json",
                   selection_sha256=Q.sha256_file(d / "qat_selection.json"))


def sel_edit(**kv):
    def edit(doc: dict) -> None:
        for k, v in kv.items():
            node, parts = doc, k.split("__")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = v
    return edit


def forged_refusal(code: str, tag: str, *, b: Bound | None = None, call=None, **forge_kw) -> tuple[bool, str]:
    fb = forge(b or bound("run_e5"), tag, **forge_kw)
    return refused(code, call or bind, fb)


@case("binding")
def c08_selection_format():
    return forged_refusal("fqoff_selection_not_of_record", "sel_format", sel=sel_edit(format="qat_selection/0"))


@case("binding")
def c08_selection_rules_sha256():
    return forged_refusal("fqoff_selection_not_of_record", "sel_rules", sel=sel_edit(rules_sha256="0" * 64))


@case("binding")
def c08_selection_smoke_inputs():
    return refused("fqoff_selection_not_of_record", bind, bound("run_e5", smoke=True))


@case("binding")
def c08_winner_epoch_not_int():
    return forged_refusal("fqoff_selection_not_of_record", "win_epoch_str", sel=sel_edit(winner__epoch="07"))


@case("binding")
def c08_winner_epoch_out_of_range():
    return forged_refusal("fqoff_selection_not_of_record", "win_epoch_16", sel=sel_edit(winner__epoch=16))


@case("binding")
def c08_winner_sha_malformed():
    return forged_refusal("fqoff_selection_not_of_record", "win_sha", sel=sel_edit(winner__checkpoint_sha256="XYZ"))


@case("binding")
def c08_selection_run_id():
    return forged_refusal("fqoff_selection_not_of_record", "sel_run_id", sel=sel_edit(run_id="another_run"))


# ---- 9 fqoff_stage_mismatch
@case("binding")
def c09_selection_stage():
    """The selection alone names another stage (E6, whose rules file is E5's) while the run and the payload are E5's;
    and an E5 binding requested as E6 is refused all the same."""
    ok1, d1 = forged_refusal("fqoff_stage_mismatch", "sel_stage", sel=sel_edit(stage="E6"))
    ok2, d2 = refused("fqoff_stage_mismatch", bind, bound("run_e5"), stage="E6")
    return ok1 and ok2, f"{d1}; {d2}"


def first_row(**kv):
    def edit(rows: list) -> None:
        rows[0].update(kv)
    return edit


@case("binding")
def c09_run_meta_stage():
    return forged_refusal("fqoff_stage_mismatch", "meta_stage", rows=first_row(stage="E6"))


def payload_edit(**kv):
    def edit(p: dict) -> None:
        p.update(kv)
    return edit


@case("binding")
def c09_payload_stage():
    b = bound("run_e5")
    return forged_refusal("fqoff_stage_mismatch", "payload_stage", b=b,
                          ck_bytes=payload_bytes(b, payload_edit(stage="E6")))


# ---- 10 fqoff_run_record
@case("binding")
def c10_run_record_missing():
    d = TMP / "forged" / "empty_run"
    d.mkdir(parents=True)
    return refused("fqoff_run_record", bind, bound("run_e5"), run_dir=d)


@case("binding")
def c10_run_dir_none():
    return refused("fqoff_run_record", bind, bound("run_e5"), run_dir=None)


# ---- 11 fqoff_run_incomplete
def drop_last_epoch(rows: list) -> None:
    cut = next(i for i, r in enumerate(rows) if r.get("event") == "epoch_end" and r.get("epoch") == Q.EPOCHS)
    del rows[cut:]


@case("binding")
def c11_run_incomplete():
    return forged_refusal("fqoff_run_incomplete", "incomplete", rows=drop_last_epoch)


# ---- 12 fqoff_run_not_real
@case("binding")
def c12_run_not_real():
    return forged_refusal("fqoff_run_not_real", "mode_smoke", rows=first_row(mode="smoke"))


# ---- 13 fqoff_telemetry_sha256_mismatch
@case("binding")
def c13_telemetry_pin():
    return refused("fqoff_telemetry_sha256_mismatch", bind, bound("run_e5"), telemetry_sha256="0" * 64)


@case("binding")
def c13_telemetry_selection():
    return forged_refusal("fqoff_telemetry_sha256_mismatch", "sel_tel", sel=sel_edit(telemetry_sha256="0" * 64))


# ---- 14 fqoff_finalize_disagrees
def finalize_with(**over):
    return lambda doc, sel_sha: finalize_record(doc, sel_sha, **over)


@case("binding")
def c14_finalize_unreadable():
    return forged_refusal("fqoff_finalize_disagrees", "fin_unreadable", finalize=lambda doc, sha: b"not json\n")


@case("binding")
def c14_finalize_format():
    return forged_refusal("fqoff_finalize_disagrees", "fin_format", finalize=finalize_with(format="qat_finalize/0"))


@case("binding")
def c14_finalize_status():
    return forged_refusal("fqoff_finalize_disagrees", "fin_status", finalize=finalize_with(status="STOP"))


@case("binding")
def c14_finalize_selection_sha256():
    return forged_refusal("fqoff_finalize_disagrees", "fin_sel",
                          finalize=finalize_with(qat_selection={"path": "qat_selection.json", "sha256": "0" * 64}))


@case("binding")
def c14_finalize_checkpoint():
    return forged_refusal("fqoff_finalize_disagrees", "fin_ck",
                          finalize=finalize_with(checkpoint={"path": "epoch_ckpts/e03.pt", "sha256": "0" * 64}))


@case("binding")
def c14_finalize_run_id():
    return forged_refusal("fqoff_finalize_disagrees", "fin_run", finalize=finalize_with(run_id="another_run"))


@case("binding")
def c14_finalize_telemetry():
    return forged_refusal("fqoff_finalize_disagrees", "fin_tel", finalize=finalize_with(telemetry_sha256="0" * 64))


@case("binding")
def c14_finalize_stage():
    return forged_refusal("fqoff_finalize_disagrees", "fin_stage", finalize=finalize_with(stage="E6"))


# ---- 15 fqoff_not_the_winner
@case("binding")
def c15_winner_checkpoint_not_its_epoch():
    return forged_refusal("fqoff_not_the_winner", "win_ck", sel=sel_edit(winner__checkpoint="epoch_ckpts/e01.pt"))


@case("binding")
def c15_checkpoint_other_epoch():
    b = bound("run_e5")
    return refused("fqoff_not_the_winner", bind, b, checkpoint=b.run / Q.EPOCH_DIR / "e03.pt")


def winner_row(**kv):
    def edit(rows: list) -> None:
        for r in rows:
            if r.get("event") == "epoch_end" and r.get("epoch") == 7:
                r.update(kv)
    return edit


@case("binding")
def c15_row_names_other_file():
    return forged_refusal("fqoff_not_the_winner", "row_ck", rows=winner_row(checkpoint="epoch_ckpts/e07_copy.pt"))


# ---- 16 fqoff_checkpoint_missing
@case("binding")
def c16_checkpoint_missing():
    return forged_refusal("fqoff_checkpoint_missing", "no_ck", skip_ck=True)


# ---- 17 fqoff_checkpoint_sha256_mismatch
@case("binding")
def c17_bytes_differ_from_row():
    b = bound("run_e5")
    return forged_refusal("fqoff_checkpoint_sha256_mismatch", "ck_bytes", b=b, ck_bytes=other_epoch_bytes(b, 3),
                          rebind_ck=False)


@case("binding")
def c17_winner_sha_differs():
    return forged_refusal("fqoff_checkpoint_sha256_mismatch", "win_sha_other",
                          sel=sel_edit(winner__checkpoint_sha256="0" * 64))


# ---- 18 fqoff_not_qat_train_state
@case("binding")
def c18_torchscript_archive():
    fb = forge(bound("run_e5"), "torchscript", ck_bytes=torchscript_bytes())
    ok, detail = refused("fqoff_not_qat_train_state", bind, fb)
    return ok and "TorchScript archive" in detail, detail


@case("binding")
def c18_wrong_kind():
    b = bound("run_e5")
    return forged_refusal("fqoff_not_qat_train_state", "kind", b=b,
                          ck_bytes=payload_bytes(b, payload_edit(quantization="qat")))


@case("binding")
def c18_unloadable():
    return forged_refusal("fqoff_not_qat_train_state", "unloadable", ck_bytes=b"not a checkpoint\n")


@case("binding")
def c18_wrong_role():
    b = bound("run_e5")
    return forged_refusal("fqoff_not_qat_train_state", "role", b=b,
                          ck_bytes=payload_bytes(b, payload_edit(artifact_role="accuracy")))


# ---- 19 fqoff_checkpoint_identity
def identity_refusal(tag: str, **kv) -> tuple[bool, str]:
    b = bound("run_e5")
    return forged_refusal("fqoff_checkpoint_identity", tag, b=b, ck_bytes=payload_bytes(b, payload_edit(**kv)))


@case("binding")
def c19_identity_epoch():
    return identity_refusal("id_epoch", epoch=6)


@case("binding")
def c19_identity_run_id():
    return identity_refusal("id_run", run_id="another_run")


@case("binding")
def c19_identity_seed():
    return identity_refusal("id_seed", seed=43)


@case("binding")
def c19_identity_source():
    return identity_refusal("id_source", source_checkpoint_sha256="f" * 64)


# ---- 20 fqoff_state_not_finite
@case("binding")
def c20_state_not_finite():
    b = bound("run_e5")
    return forged_refusal("fqoff_state_not_finite", "nonfinite", b=b,
                          ck_bytes=payload_bytes(b, payload_edit(state_finite=False)))


@case("binding")
def c20_state_flag_missing():
    b = bound("run_e5")
    return forged_refusal("fqoff_state_not_finite", "flag_none", b=b,
                          ck_bytes=payload_bytes(b, payload_edit(state_finite=None)))


# ================================================================ loader (FQR9, FQR12 codes 21-22)
def census(model: nn.Module) -> tuple[bool, str]:
    flags = Q.flag_summary(model)
    fq = Q.fake_quant_modules(model)
    devices = {str(t.device) for t in model.state_dict().values()}
    training = [n for n, m in model.named_modules() if m.training]
    ok = flags == {"observer_enabled": [0], "fake_quant_enabled": [0]} and fq and not training and devices == {"cpu"}
    return ok, f"{flags} over {len(fq)} fake-quant modules; training {len(training)}; devices {sorted(devices)}"


def stored_flags(b: Bound) -> dict:
    return Q.flag_summary(torch.load(b.ck, map_location="cpu", weights_only=True)["model_state_dict"])


@case("loader")
def load_fake_quant_off_model():
    """An epoch-7 state (observers on in the state as stored) loads with every fake-quant and observer off."""
    b = bound("run_e5")
    ok, detail = census(ML.load_qat_fakequant_off_student(bind(b)))
    stored = stored_flags(b)
    return ok and stored == {"observer_enabled": [1], "fake_quant_enabled": [1]}, f"{detail}; stored {stored}"


@case("loader")
def load_late_epoch_state():
    b = bound("run_e5", winner=14)
    ok, detail = census(ML.load_qat_fakequant_off_student(bind(b)))
    stored = stored_flags(b)
    return ok and stored == {"observer_enabled": [0], "fake_quant_enabled": [1]}, f"{detail}; stored {stored}"


@case("loader")
def load_e6e2_state():
    return census(ML.load_qat_fakequant_off_student(bind(bound("run_e6e2"))))


@case("loader")
def load_e6kd_state():
    """E6-KD's eNN.pt holds the prepared student's state only (its projection is not in it)."""
    return census(ML.load_qat_fakequant_off_student(bind(bound("run_e6kd"))))


@case("loader")
def load_reads_the_file_once():
    b = bound("run_e5")
    c = bind(b)
    with recording() as paths:
        ML.load_qat_fakequant_off_student(c)
    got = fixture_opens(paths)
    return got == [str(b.ck)] and paths.count(str(b.ck)) == 1, f"opened {got}"


@case("loader")
def load_file_changed_after_validation():
    """The file replaced between validation and load: the loader's own hash of the bytes it loads refuses it."""
    b = forge(bound("run_e5"), "changed_after")
    c = bind(b)
    tmp = b.ck.with_name(b.ck.name + ".new")
    tmp.write_bytes(other_epoch_bytes(b, 3))
    os.replace(tmp, b.ck)
    return refused("checkpoint_changed_after_validation", ML.load_qat_fakequant_off_student, c)


def state_edit(fn):
    def edit(p: dict) -> None:
        fn(p["model_state_dict"])
    return edit


def own_key(s: dict, suffix: str) -> str:
    """The first entry of a state that is a module's own parameter (no fake-quant or observer entry)."""
    return next(k for k in sorted(s) if k.endswith(suffix) and "fake_quant" not in k
                and "activation_post_process" not in k)


@case("loader")
def c21_state_key_missing():
    b = bound("run_e5")
    fb = forge(b, "state_missing_key", ck_bytes=payload_bytes(b, state_edit(lambda s: s.pop(own_key(s, ".weight")))))
    return refused("fqoff_load_not_exact", ML.load_qat_fakequant_off_student, bind(fb))


@case("loader")
def c21_state_shape():
    b = bound("run_e5")

    def widen(s: dict) -> None:
        k = own_key(s, ".bias")
        s[k] = torch.zeros(s[k].shape[0] + 1)
    fb = forge(b, "state_shape", ck_bytes=payload_bytes(b, state_edit(widen)))
    return refused("fqoff_load_not_exact", ML.load_qat_fakequant_off_student, bind(fb))


@case("loader")
def c22_toggle_not_taken():
    """torch's disable_fake_quant made a no-op: the census refuses the model."""
    import torch.ao.quantization as tq
    c = bind(bound("run_e5"))
    with patched(tq, "disable_fake_quant", lambda mod: None):
        return refused("fqoff_fake_quant_enabled", ML.load_qat_fakequant_off_student, c)


# ================================================================ parity (FQR14)
def parity_inputs() -> torch.Tensor:
    return color_task(4, 64, seed=4242)[0]


@functools.lru_cache(maxsize=None)
def parity_pair() -> tuple:
    b = bound("run_e5")
    c = bind(b)
    off = ML.load_qat_fakequant_off_student(c)
    state = torch.load(b.ck, map_location="cpu", weights_only=True)["model_state_dict"]
    ref, report = ML.fp32_reference_from_qat_state(state)
    return off, ref, report, state


def margins(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


@case("parity")
def parity_fp32_reference_logits():
    off, ref, _report, _state = parity_pair()
    x = parity_inputs()
    with torch.no_grad():
        a, r = off(x), ref(x)
    diff = float((a - r).abs().max())
    confident = margins(r) > 2 * PARITY_TOL
    agree = (a.argmax(1) == r.argmax(1))[confident]
    share = float(confident.float().mean())
    ok = diff <= PARITY_TOL and bool(agree.all())
    return ok, (f"max |logit difference| {diff:.3e} <= {PARITY_TOL:g}; argmax agreement "
                f"{float(agree.float().mean()):.4f} on the {share:.4f} of pixels whose top-two margin exceeds "
                f"{2 * PARITY_TOL:g}")


@case("parity")
def parity_mapping_accounts_every_key():
    _off, _ref, report, state = parity_pair()
    mapped, left = sum(report["mapped"].values()), sum(report["left_out"].values())
    ok = (mapped + left == report["state_keys"] == len(state) and mapped == report["fp32_keys"]
          and report["left_out"].get("fixedqparams_alias") == 18)
    return ok, json.dumps(report, sort_keys=True)


@case("parity")
def parity_toggle_bites():
    """The same state with fake quantization on (Q2's fake_quant_model: observers off, eval) gives other logits."""
    off, _ref, _report, state = parity_pair()
    fq = A.fake_quant_model(state)
    x = parity_inputs()
    with torch.no_grad():
        diff = float((fq(x) - off(x)).abs().max())
    return diff > 10 * PARITY_TOL, f"max |logit difference| fake quant on vs off {diff:.3e} (> {10 * PARITY_TOL:g})"


# ================================================================ finder (FQR12, FQR15a)
@case("finder")
def finder_finds_phase1_sites():
    """The mutation harness's site finder, over its SCOPE files, finds every fqoff raise form of the files (counted
    as text), each with a literal code; their codes are phase 1's, first met in FQR12's order; and its refusal table
    names exactly the sites found."""
    import scripts.smoke_eval_qat_fakequant_off_mutations as MH
    forms = re.compile(r"\b(_fail|_fqoff_refuse|_fqoff_cli)\(\s*\"fqoff_")
    sites, raw = [], 0
    for rel in MH.SCOPE:
        text = (REPO / rel).read_text(encoding="utf-8")
        sites += MH.fqoff_refusal_sites(text, rel)
        raw += len(forms.findall(text))
    codes = [s["code"] for s in sites]
    first = sorted(set(codes), key=codes.index)
    keys = {(s["rel"], s["func"], s["code"], s["k"]) for s in sites}
    ok = (set(codes) == set(ALL_CODES) and all(s["literal"] for s in sites) and len(sites) == raw
          and first == list(ALL_CODES) and keys == set(MH.KILLS))
    return ok, f"{len(sites)} sites of {len(set(codes))} codes; {raw} raise forms in the files"


# ================================================================ never_selects (FQR4, F4)
@case("never_selects")
def never_selects_rules_files():
    """Each rules file's evaluator block (read as JSON) requires int8_qat, and each names the fake-quant VAL in
    never_selects."""
    got = {}
    for rel in S.RULES_FILES:
        doc = json.loads((REPO / rel).read_text(encoding="utf-8"))["qat_epoch"]
        got[rel] = (doc["evaluator"]["precision"], "fake_quant_val_all_class_miou" in doc["never_selects"])
    return all(v == ("int8_qat", True) for v in got.values()) and len(got) == 3, str(got)


@case("never_selects")
def never_selects_stats_precision():
    """src/stats/driver.py's STAGE_PRECISION, read with ast (never imported), holds no qat_fakequant_off."""
    tree = ast.parse((REPO / "src/stats/driver.py").read_text(encoding="utf-8"))
    value = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "STAGE_PRECISION" for t in n.targets))
    return FQOFF not in value.values() and value["E5"] == value["E6"] == "int8_qat", str(value)


@case("never_selects")
def never_selects_selection_refuses_fqoff_summary():
    """A selection's summary check refuses a qat_fakequant_off summary (src/quant/qat_select.py _check_summary)."""
    ev = S.load_rules(S.rules_path_for_stage("E5"))[0]["qat_epoch"]["evaluator"]
    s = summary(0.4, "r", "0" * 64, "E5")
    s["run"].update(precision=FQOFF, quant_backend=None)
    bad = S._check_summary(s, stage="E5", ev=ev, smoke=False)
    return any(x.startswith("run.precision") for x in bad), str(bad)


# ================================================================ cli (phase 2: FQR11, FQR12 codes 1-5, FQR13)
CLI = TMP / "cli"                       # never created: validate_cli_args is pure and opens nothing
CONFIRM = f"confirm_{TOKEN}_split"       # the confirm flag's argparse name, the token never written


def cli_base(**kw) -> dict:
    """An fqoff run's arguments, every flag valid, `kw` replacing (None drops a flag)."""
    d = {"stage": "E5", "precision": FQOFF, "split": "val", "out_dir": str(TMP / "out_cli"),
         "checkpoint": str(CLI / "e07.pt"), "qat_run_dir": str(CLI / "run"),
         "qat_selection": str(CLI / "eval" / "qat_selection.json"), "qat_selection_sha256": "1" * 64,
         "qat_telemetry_sha256": "2" * 64, "artifact_status": "provisional", "device": "cpu", "batch_size": 1}
    d.update(kw)
    return d


def argv_of(d: dict) -> list[str]:
    argv = []
    for k, v in d.items():
        if v is True:
            argv.append(f"--{k.replace('_', '-')}")
        elif v not in (None, False):
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def em_args(d: dict):
    return EM.build_parser().parse_args(argv_of(d))


def cli_code_of(err: Exception) -> str:
    m = re.match(r"\[(\w+)\]", str(err))
    return m.group(1) if m else "uncoded"


def cli_code(args) -> tuple[str | None, str]:
    """validate_cli_args' refusal code ("[code] ..."), "uncoded" for a refusal without one, None when accepted."""
    try:
        EM.validate_cli_args(args)
    except EM.CliError as e:
        return cli_code_of(e), str(e)
    return None, "accepted"


def cli_refused(code: str, **kw) -> tuple[bool, str]:
    got, msg = cli_code(em_args(cli_base(**kw)))
    return got == code, f"{got}: {msg[:200]}"


@case("cli")
def cli_precision_and_flags():
    """The precision is a --precision choice and a supported precision, never an INT8 one; the four binding flags
    default to None; an fqoff run's arguments pass validate_cli_args."""
    plain = EM.build_parser().parse_args(["--out-dir", str(TMP / "out_cli")])
    ok = (FQOFF in EM.SUPPORTED_PRECISIONS and FQOFF not in EM.INT8_PRECISIONS
          and all(getattr(plain, n) is None for n in EM.QAT_FLAGS) and cli_code(em_args(cli_base()))[0] is None)
    return ok, f"SUPPORTED_PRECISIONS {EM.SUPPORTED_PRECISIONS}; flags {EM.QAT_FLAGS}"


# ---- 1 fqoff_stage_not_in_scope
@case("cli")
def c01_stage_not_in_scope():
    oks = [cli_refused("fqoff_stage_not_in_scope", stage=s) for s in ("E1", "E4", "E6-R1", "E6-R2", "E7")]
    return all(ok for ok, _ in oks), oks[2][1]


@case("cli")
def c01_role_not_student():
    return cli_refused("fqoff_stage_not_in_scope", model_role="teacher")


# ---- 2 fqoff_provenance_forbidden
@case("cli")
def c02_provenance_forbidden():
    return cli_refused("fqoff_provenance_forbidden", provenance=str(CLI / "prov.json"))


# ---- 3 fqoff_source_flags
@case("cli")
def c03_qat_flag_off_precision():
    """For every other precision a --qat-* flag is refused: an fp32 checkpoint run, an INT8 provenance run."""
    fp32 = cli_base(stage="E1", precision="fp32", qat_selection=None, qat_selection_sha256=None,
                    qat_telemetry_sha256=None)
    int8 = cli_base(precision="int8_qat", checkpoint=None, provenance=str(CLI / "prov.json"), qat_run_dir=None,
                    qat_selection_sha256=None, qat_telemetry_sha256=None)
    a, b = cli_code(em_args(fp32)), cli_code(em_args(int8))
    return a[0] == b[0] == "fqoff_source_flags", f"{a[1][:100]}; {b[1][:100]}"


@case("cli")
def c03_checkpoint_missing():
    return cli_refused("fqoff_source_flags", checkpoint=None)


@case("cli")
def c03_binding_flag_missing():
    oks = [cli_refused("fqoff_source_flags", **{n: None}) for n in EM.QAT_FLAGS]
    return all(ok for ok, _ in oks), "; ".join(d[:60] for _ok, d in oks)


@case("cli")
def c03_random_init():
    return cli_refused("fqoff_source_flags", random_init=True)


@case("cli")
def c03_sha256_malformed():
    oks = [cli_refused("fqoff_source_flags", **{n: v}) for n in ("qat_selection_sha256", "qat_telemetry_sha256")
           for v in ("A" * 64, "1" * 63)]
    return all(ok for ok, _ in oks), oks[0][1]


# ---- 4 fqoff_not_cpu and 5 fqoff_protocol
@case("cli")
def c04_not_cpu():
    return cli_refused("fqoff_not_cpu", device="cuda")


@case("cli")
def c05_protocol_upstream():
    return cli_refused("fqoff_protocol", protocol="upstream")


@case("cli")
def cli_guards_of_record_kept():
    """The split of record (the token, never a path here) keeps its guards for this precision, the confirm flag and
    the cap among them, and LEFF B-1b's rules under R (official): H and T required; with both the arguments pass.
    validate_cli_args is pure: nothing is opened."""
    r = dict(split=TOKEN, artifact_status="official")
    no_confirm = cli_code(em_args(cli_base(**r)))
    no_h = cli_code(em_args(cli_base(**r, **{CONFIRM: True})))
    no_t = cli_code(em_args(cli_base(**r, **{CONFIRM: True}, expected_checkpoint_sha256="3" * 64)))
    full = dict(r, **{CONFIRM: True}, expected_checkpoint_sha256="3" * 64, threads=2)
    ok_r, capped = cli_code(em_args(cli_base(**full))), cli_code(em_args(cli_base(**full, max_samples=4)))
    ok = (no_confirm[0] == "uncoded" and "--confirm-" in no_confirm[1]
          and no_h[0] == "expected_checkpoint_sha256_required" and no_t[0] == "threads_required"
          and ok_r[0] is None and capped[0] == "uncoded" and "sample cap" in capped[1])
    return ok, f"{no_confirm[0]}, {no_h[0]}, {no_t[0]}, {ok_r[0]}, {capped[0]}"


@case("cli")
def cli_h2_refused():
    """LEFF B-1b's own refusal of --expected-converted-sha256 off INT8 stands for this precision, unchanged."""
    return cli_refused("expected_converted_sha256_not_int8", expected_converted_sha256="4" * 64)


@case("cli")
def cli_callers_without_the_flags():
    """An args object without the four attributes (a caller that predates them) validates as before."""
    a = EM.build_parser().parse_args(["--stage", "E1", "--checkpoint", str(CLI / "e1.pt"), "--out-dir",
                                      str(TMP / "out_cli")])
    for n in EM.QAT_FLAGS:
        delattr(a, n)
    got = cli_code(a)
    return got[0] is None, got[1][:120]


# ================================================================ run (phase 2: end to end, FQR11)
@functools.lru_cache(maxsize=None)
def val_tree() -> dict:
    """The synthetic VAL split under PLANTSEG_DATA_ROOT (set before configs/data.py is imported): all 846 stems, so
    the evaluator's cap picks its rows as it always does; no TRAIN and no TEST folder."""
    return make_tree(DATA, n_train=0, n_val=846, seed=5)


@contextlib.contextmanager
def writer_without_git():
    """The artifact writer's whole-repository git calls stubbed, as scripts/synthetic_qat_fixtures.py's evaluator
    driver stubs them: a cloud lane never lists the whole repository."""
    with patched(ART, "git_porcelain_bytes", lambda repo: b""), patched(ART, "git_commit", lambda repo: "0" * 40):
        yield


def run_args(b: Bound, tag: str, **kw):
    d = cli_base(stage=b.stage, out_dir=str(TMP / "runs" / tag), checkpoint=str(b.ck), qat_run_dir=str(b.run),
                 qat_selection=str(b.sel), qat_selection_sha256=b.sel_sha, qat_telemetry_sha256=b.tel_sha,
                 max_samples=4, run_id=f"fqoff_{tag}")
    d.update(kw)
    return em_args(d)


def em_run(args) -> tuple[object, str, object]:
    """run() with counters, stdout captured, the writer's git calls stubbed and torch's thread count restored.
    Returns (the artifact directory, or the exception raised; stdout; the counters)."""
    val_tree()
    c = EM.Counters(dataset=[], model=[])
    buf = io.StringIO()
    threads = torch.get_num_threads()
    try:
        with writer_without_git(), contextlib.redirect_stdout(buf):
            out = EM.run(args, counters=c)
    except Exception as e:                                # noqa: BLE001 -- the case reads it
        out = e
    finally:
        torch.set_num_threads(threads)
    return out, buf.getvalue(), c


def run_code(out) -> str | None:
    """The refusal code of a run's exception, None for an artifact directory."""
    if isinstance(out, (SA.StageArtifactError, EvalRuntimeError)):
        return out.code
    if isinstance(out, EM.CliError):
        return cli_code_of(out)
    return None if isinstance(out, Path) else f"{type(out).__name__}: {str(out)[:160]}"


def summary_of(out) -> dict:
    return json.loads((out / "summary.json").read_text(encoding="utf-8")) if isinstance(out, Path) else {}


@functools.lru_cache(maxsize=None)
def e2e(raw: str = "run_e5", tag: str = "e5") -> tuple:
    b = bound(raw)
    out, stdout, c = em_run(run_args(b, tag))
    return b, out, stdout, c, summary_of(out)


@case("run")
def run_e5_end_to_end():
    """E5's selected epoch scored end to end on the synthetic VAL rows: an artifact, the adapter and the model built
    after validation."""
    _b, out, _stdout, c, summ = e2e()
    ok = isinstance(out, Path) and bool(summ) and c.dataset and c.model
    return bool(ok), str(out) if isinstance(out, Path) else str(run_code(out))


@case("run")
def run_binding_line():
    """One stdout line after validation names the binding (FQR11)."""
    b, _out, stdout, _c, _summ = e2e()
    run_id = json.loads(b.sel.read_text(encoding="utf-8"))["run_id"]
    want = (f"fqoff binding: stage E5, run {run_id}, epoch {b.epoch:02d}, checkpoint {Q.sha256_file(b.ck)}, "
            f"selection {b.sel_sha}, telemetry {b.tel_sha}, finalize absent")
    return want in stdout.splitlines(), want[:150]


@case("run")
def run_loads_through_fqoff_loader():
    """The load dispatch takes this precision before INT8's and FP32's: load_qat_fakequant_off_student once,
    load_student_checkpoint never (FQ11)."""
    b = bound("run_e5")
    calls = {"fqoff": 0, "fp32": 0}
    real_fq, real_fp = ML.load_qat_fakequant_off_student, ML.load_student_checkpoint

    def fq(carrier):
        calls["fqoff"] += 1
        return real_fq(carrier)

    def fp(*a, **k):
        calls["fp32"] += 1
        return real_fp(*a, **k)
    with patched(ML, "load_qat_fakequant_off_student", fq), patched(ML, "load_student_checkpoint", fp):
        out, _stdout, _c = em_run(run_args(b, "dispatch"))
    return isinstance(out, Path) and calls == {"fqoff": 1, "fp32": 0}, f"{calls}; {run_code(out)}"


@case("run")
def run_finalize_line():
    """With an agreeing finalize record beside the selection, the binding line names its sha256."""
    b = forge(bound("run_e5"), "run_finalize", finalize=lambda doc, sha: finalize_record(doc, sha))
    out, stdout, _c = em_run(run_args(b, "finalize"))
    fin = Q.sha256_file(b.ev / SA.QAT_FINALIZE_NAME)
    line = next((ln for ln in stdout.splitlines() if ln.startswith("fqoff binding:")), "")
    return isinstance(out, Path) and line.endswith(f"finalize {fin}"), line[-90:] or str(run_code(out))


@case("run")
def run_e6kd_end_to_end():
    """E6-KD's binding (its own rules file, no finalize record) scores end to end."""
    _b, out, _stdout, _c, summ = e2e("run_e6kd", "e6kd")
    run = summ.get("run", {})
    ok = isinstance(out, Path) and run.get("stage") == "E6-KD" and run.get("precision") == FQOFF
    return ok, f"{run.get('stage')} {run.get('precision')}" if summ else str(run_code(out))


@case("run")
def run_h_matching_accepted():
    """--expected-checkpoint-sha256 equal to the eNN.pt's: P1 and P2 pass and the run scores."""
    b = bound("run_e5")
    out, _stdout, _c = em_run(run_args(b, "h_match", expected_checkpoint_sha256=Q.sha256_file(b.ck)))
    return isinstance(out, Path), str(run_code(out))


@case("run")
def run_h_mismatch_refused_before_parse():
    """H differing from the eNN.pt: LEFF B-1b's P1 refuses with its code, on the eNN.pt this precision's handed-file
    case names, before the validator opens any binding file (FQ14a)."""
    b = bound("run_e5")
    with recording() as opened:
        out, _stdout, c = em_run(run_args(b, "h_mismatch", expected_checkpoint_sha256="7" * 64))
    binding = [p for p in fixture_opens(opened) if p in {str(b.sel), str(b.run / Q.TELEMETRY_NAME)}]
    ok = run_code(out) == "expected_checkpoint_sha256_mismatch" and not binding and not c.dataset and not c.model
    return ok, f"{run_code(out)}; binding files opened {binding}"


@case("run")
def run_h_changed_during_validation():
    """The eNN.pt equal to H at P1 and replaced by the bound file before the validator hashes it: this precision's
    P2 comparison (the carrier's sha256 against H) refuses with LEFF B-1b's code (FQ14b)."""
    b = forge(bound("run_e5"), "h_swap")
    true_bytes, other = b.ck.read_bytes(), other_epoch_bytes(b, 3)
    swap = b.ck.with_name(b.ck.name + ".new")
    swap.write_bytes(other)
    os.replace(swap, b.ck)                               # the hard link is replaced; the raw run is never written
    real = A.read_run_record

    def restore_then_read(run_dir):
        if b.ck.read_bytes() != true_bytes:
            back = b.ck.with_name(b.ck.name + ".back")
            back.write_bytes(true_bytes)
            os.replace(back, b.ck)
        return real(run_dir)
    with patched(A, "read_run_record", restore_then_read):
        out, _stdout, c = em_run(run_args(b, "h_swap", expected_checkpoint_sha256=Q.sha256_bytes(other)))
    ok = run_code(out) == "source_changed_during_validation" and not c.dataset and not c.model
    return ok, str(run_code(out))


@case("run")
def run_threads_recorded():
    """LEFF B-1b's --threads holds for this precision, set right after validate_cli_args, before any source check:
    the summary records it."""
    b = bound("run_e5")
    want = 3 if torch.get_num_threads() == 2 else 2
    out, _stdout, _c = em_run(run_args(b, "threads", threads=want))
    got = summary_of(out).get("run", {}).get("eval_runtime", {}).get("torch_num_threads")
    return got == want, f"torch_num_threads {got}, --threads {want}; {run_code(out)}"


# ================================================================ summary (phase 2: FQR3)
@case("summary")
def summary_precision_and_backend():
    """run.precision qat_fakequant_off with run.quant_backend null (FQ8)."""
    _b, _out, _stdout, _c, summ = e2e()
    run = summ.get("run", {})
    return run.get("precision") == FQOFF and "quant_backend" in run and run["quant_backend"] is None, \
        f"{run.get('precision')}, {run.get('quant_backend')!r}"


@case("summary")
def summary_checkpoint_fields():
    """run.checkpoint_path is the eNN.pt and run.checkpoint_sha256 its sha256; a trained student, never random."""
    b, _out, _stdout, _c, summ = e2e()
    run = summ.get("run", {})
    ok = (run.get("checkpoint_path") == str(b.ck) and run.get("checkpoint_sha256") == Q.sha256_file(b.ck)
          and run.get("random_init") is False and run.get("stage") == "E5" and run.get("model_role") == "student")
    return ok, str(run.get("checkpoint_path"))


@case("summary")
def summary_runtime_record():
    """CPU, no determinism policy, no checkpoint metadata (ckpt_info stays None), the cap's rows, canvas."""
    _b, _out, _stdout, _c, summ = e2e()
    rt, ds = summ.get("run", {}).get("eval_runtime", {}), summ.get("dataset", {})
    ok = (rt.get("checkpoint_iteration") is None and rt.get("checkpoint_best_val_miou_all_class") is None
          and rt.get("determinism_policy_applied") is False and str(rt.get("model_device")) == "cpu"
          and ds.get("expected_rows") == ds.get("actual_rows") == 4
          and ds.get("preprocess_protocol") == "core_preprocess/1.0.0")
    keys = ("model_device", "determinism_policy_applied", "checkpoint_iteration", "torch_num_threads")
    return ok, json.dumps({k: rt.get(k) for k in keys})


@case("summary")
def summary_config_sha256():
    """config_sha256 hashes the request's config payload, precision included: the same request at int8_qat hashes
    otherwise (FQR3)."""
    import dataclasses
    b = bound("run_e5")
    seen = {}
    real = ART.prepare_artifact_request

    def capture(**kw):
        seen["req"] = real(**kw)
        return seen["req"]
    with patched(ART, "prepare_artifact_request", capture):
        out, _stdout, _c = em_run(run_args(b, "config"))
    req = seen["req"]
    other = dataclasses.replace(req, run=dataclasses.replace(req.run, precision="int8_qat"))
    mine = Q.sha256_bytes(ART.canonical_json_bytes(ART.build_config_payload(req)))
    theirs = Q.sha256_bytes(ART.canonical_json_bytes(ART.build_config_payload(other)))
    got = summary_of(out).get("run", {}).get("config_sha256")
    return got == mine and mine != theirs, f"config_sha256 {str(got)[:16]}..; at int8_qat {theirs[:16]}.."


@case("summary")
def summary_never_selected():
    """A selection's summary check refuses this run's own summary (src/quant/qat_select.py _check_summary)."""
    _b, _out, _stdout, _c, summ = e2e()
    ev = S.load_rules(S.rules_path_for_stage("E5"))[0]["qat_epoch"]["evaluator"]
    bad = S._check_summary(summ, stage="E5", ev=ev, smoke=False)
    return any(x.startswith("run.precision") for x in bad), str(bad)[:200]


# ================================================================ order (phase 2: FQR12's places)
@case("order")
def order_cli_refusals_construct_nothing():
    """Codes 1-5 through run(): validate_cli_args refuses each before anything is constructed."""
    kws = [dict(stage="E1"), dict(model_role="teacher"), dict(provenance=str(CLI / "prov.json")),
           dict(checkpoint=None), dict(device="cuda"), dict(protocol="upstream")]
    want = ["fqoff_stage_not_in_scope", "fqoff_stage_not_in_scope", "fqoff_provenance_forbidden",
            "fqoff_source_flags", "fqoff_not_cpu", "fqoff_protocol"]
    got = []
    for kw in kws:
        out, _stdout, c = em_run(em_args(cli_base(**kw)))
        got.append((run_code(out), bool(c.dataset or c.model)))
    return [g[0] for g in got] == want and not any(g[1] for g in got), str(got)


def source_refusals() -> list[tuple[str, Bound, dict]]:
    """One binding per source code (6-20) for run(): (code, the binding, run arguments replaced)."""
    b = bound("run_e5")
    other_ck = str(b.run / Q.EPOCH_DIR / "e03.pt")
    return [
        ("fqoff_forbidden_path", b, dict(checkpoint=str(never("nowhere_latest_run", "e07.pt")))),
        ("fqoff_selection_sha256_mismatch", b, dict(qat_selection_sha256="8" * 64)),
        ("fqoff_selection_not_of_record", forge(b, "o08", sel=sel_edit(rules_sha256="0" * 64)), {}),
        ("fqoff_stage_mismatch", forge(b, "o09", sel=sel_edit(stage="E6")), {}),
        ("fqoff_run_record", b, dict(qat_run_dir=str(never("run")))),
        ("fqoff_run_incomplete", forge(b, "o11", rows=drop_last_epoch), {}),
        ("fqoff_run_not_real", forge(b, "o12", rows=first_row(mode="smoke")), {}),
        ("fqoff_telemetry_sha256_mismatch", b, dict(qat_telemetry_sha256="9" * 64)),
        ("fqoff_finalize_disagrees", forge(b, "o14", finalize=lambda doc, sha: finalize_record(doc, sha,
                                                                                               status="pending")), {}),
        ("fqoff_not_the_winner", b, dict(checkpoint=other_ck)),
        ("fqoff_checkpoint_missing", forge(b, "o16", skip_ck=True), {}),
        ("fqoff_checkpoint_sha256_mismatch", forge(b, "o17", ck_bytes=other_epoch_bytes(b, 3), rebind_ck=False), {}),
        ("fqoff_not_qat_train_state", forge(b, "o18", ck_bytes=torchscript_bytes()), {}),
        ("fqoff_checkpoint_identity", forge(b, "o19", ck_bytes=payload_bytes(b, payload_edit(epoch=3))), {}),
        ("fqoff_state_not_finite", forge(b, "o20", ck_bytes=payload_bytes(b, payload_edit(state_finite=False))), {}),
    ]


@case("order")
def order_source_refusals_before_adapter():
    """Codes 6-20 through run(): each refused in the source branch, before the dataset adapter or a model exists."""
    got = []
    for code, b, over in source_refusals():
        out, _stdout, c = em_run(run_args(b, f"order_{code}", **over))
        got.append((code, run_code(out), bool(c.dataset or c.model)))
    bad = [g for g in got if g[0] != g[1] or g[2]]
    return not bad and len(got) == 15, f"{len(got) - len(bad)}/{len(got)} refused before the adapter; {bad[:2]}"


@case("order")
def order_loader_refusal_after_adapter():
    """Codes 21-22 are the loader's: after validate_artifact_request and the adapter, and no artifact is written."""
    b = bound("run_e5")
    fb = forge(b, "o21", ck_bytes=payload_bytes(b, state_edit(lambda s: s.pop(own_key(s, ".weight")))))
    out, _stdout, c = em_run(run_args(fb, "order_loader"))
    written = [os.path.join(d, n) for d, _dirs, files in os.walk(TMP / "runs" / "order_loader") for n in files
               if n == "summary.json"]
    ok = run_code(out) == "fqoff_load_not_exact" and bool(c.dataset) and bool(c.model) and not written
    return ok, f"{run_code(out)}; adapter {bool(c.dataset)}, model {bool(c.model)}, written {len(written)}"


# ================================================================ imports (phase 2: T-FQ8)
IMPORTS_DRIVER = r'''"""One fqoff run of scripts/evaluate_model.py's main() in a fresh interpreter: the repository
modules it loads.

Its git calls stubbed as scripts/synthetic_qat_fixtures.py's evaluator driver stubs them. Prints one IMPORTS line.
"""
import json
import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ["SMOKE_REPO"])
import scripts.evaluate_model as EM
before = set(sys.modules)
import src.eval.artifacts as artifacts
artifacts.git_porcelain_bytes = lambda r: b""
artifacts.git_commit = lambda r: "0" * 40
rc = EM.main(sys.argv[1:])
print("IMPORTS " + json.dumps(sorted(set(sys.modules) - before)))
raise SystemExit(rc)
'''
IMPORTS_REQUIRED = {"src.quant.qat", "src.quant.qat_artifacts", "src.quant.qat_select", "src.quant.x86_latency",
                    "src.quant.prepare"}


@case("imports")
def imports_fqoff_run_modules():
    """The repository modules an fqoff run loads at run time: a sys.modules difference over one end-to-end run in a
    fresh interpreter (the report's list for TMAN's evaluator file set, T-FQ8). src/quant's five are among them, and
    no src.stats module is."""
    import subprocess
    b = bound("run_e5")
    val_tree()
    drv = TMP / "imports_driver.py"
    drv.write_text(IMPORTS_DRIVER, encoding="utf-8")
    d = cli_base(stage=b.stage, out_dir=str(TMP / "runs" / "imports"), checkpoint=str(b.ck), qat_run_dir=str(b.run),
                 qat_selection=str(b.sel), qat_selection_sha256=b.sel_sha, qat_telemetry_sha256=b.tel_sha,
                 max_samples=4, run_id="fqoff_imports")
    env = {**os.environ, "SMOKE_REPO": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run([sys.executable, "-B", str(drv), *argv_of(d)], capture_output=True, text=True, env=env,
                       cwd=str(REPO), timeout=1800, stdin=subprocess.DEVNULL)
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("IMPORTS ")), None)
    mods = json.loads(line[len("IMPORTS "):]) if line else []
    repo_mods = sorted(m for m in mods if m.split(".")[0] in ("src", "scripts", "configs"))
    ok = (p.returncode == 0 and IMPORTS_REQUIRED <= set(repo_mods)
          and not any(m == "src.stats" or m.startswith("src.stats.") for m in mods))
    return ok, (f"rc {p.returncode}; " + " ".join(repo_mods)) if line else f"rc {p.returncode}: {p.stderr[-200:]}"


# ================================================================ hygiene
def forbidden_names_created() -> list[str]:
    """Every path under this smoke's temp directory and its cache whose name holds the token SL-1 forbids, or ':'."""
    roots = [TMP] + ([OPTS["cache"]] if OPTS["cache"] is not None else [])
    bad = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            bad += [os.path.join(dirpath, n) for n in dirnames + filenames if TOKEN in n.lower() or ":" in n]
    return bad


@case("hygiene")
def hygiene_no_forbidden_named_path_created():
    bad = forbidden_names_created()
    return not bad, f"{len(bad)} found" + (f": {bad[:3]}" if bad else "")


@case("hygiene")
def hygiene_repository_untouched():
    verdict = state_verdict(OPTS["before"], repo_state(REPO))
    return verdict == "unchanged", verdict


# ================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated case names")
    ap.add_argument("--cache-dir", default=None, help="keeps the raw QAT runs (trained when absent, cloned after)")
    a = ap.parse_args(argv)
    names = [n for n, _s, _f in CASES]
    wanted = names if not a.only else a.only.split(",")
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        print(f"RESULT: ERROR unknown cases {unknown}")
        return 2
    if a.cache_dir:
        cache = Path(a.cache_dir).resolve()
        if Q.names_test(cache):
            print("RESULT: ERROR --cache-dir names the token SL-1 forbids")
            return 2
        OPTS["cache"] = cache
    OPTS["before"] = repo_state(REPO)
    t0 = time.time()
    print("=" * 78)
    print("L-FQOFF FAKE-QUANT-OFF SMOKE (phases 1 and 2) -- the binding, the loader, the FP32 reference, the CLI and "
          "the run, synthetic only")
    print(f"temp: {TMP}" + (f" | cache: {OPTS['cache']}" if OPTS["cache"] else ""))
    print("=" * 78, flush=True)
    if OPTS["cache"] is not None:
        fill_cache()
    for name, section, fn in CASES:
        if name not in wanted:
            continue
        try:
            ok, detail = fn()
        except Exception as e:                           # noqa: BLE001 -- a case that raises fails
            ok, detail = False, f"raised {type(e).__name__}: {str(e)[:300]}"
        check(name, ok, detail)
        print(f"  [{section}] {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:58}: {'PASS' if ok else 'FAIL'}  {detail}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n[time] {time.time() - t0:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) and results else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
