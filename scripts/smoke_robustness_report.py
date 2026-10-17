#!/usr/bin/env python3
"""S5 — the descriptive robustness outputs and the VAL rehearsal script (L-CORR B-4). Synthetic; no
PlantSeg data; no GPU.

Fixtures. Every artifact is written by the real writer (src/eval/artifacts.write_artifact) from a
hand-built EvalResult, in a throwaway git repository so that 'official' artifacts can be written;
every cell record by src/eval/corrupted.build_cell_record and write_cell_record. Nothing this smoke
creates has "test" in its name; TEST-named refusals use never-created paths.
  A  smoke policy, split val, 2 images: E1 and E6 (int8_qat) at seed 42 and the teacher on all 24
     cells, E1 at seed 43 on the 15 inferential cells; hand-set totals (class 0: tp = k of 100;
     class 1: tp = j of 10), so every value has a closed form, checked with Fractions
  B  E1b (seed 42) whose brightness cells beat its clean value: E1's degradation <= 0 (LE7)
  O  official policy, split test: E1 at seed 42, 25 artifacts of 1,561 rows with a 26-key runtime
     record, at the fixture repository's HEAD
  H  rehearsal: E1 at seed 42 and the stand-in (E1 at seed 44), clean + 15 inferential cells, 846
     rows, provisional, one row without disease ground truth (K_val = 1)

  Q   the quantities Q1-Q9 (plan section 4, LE4, LE6-LE8)
  V   the checks and refusals (V1-V14; LE1, LE2, LE7)
  O   the outputs: strict canonical JSON, the md banner and no significance value, SHA256SUMS, the
      CLI (smoke and official), descriptive code only
  H   scripts/rehearse_robustness_val.py (R-G5; the stand-in mapping)
  RC  the cell-record codes of src/eval/corrupted.py that B-4 reads
  M   mutations: the plan's five (M1-M5), LE2/LE4/LE7/LE13 (M6-M10), the rehearsal's two, and one
      per refusal code, each killed by its named case. A case that raised is no kill (its checks
      never ran); a case that sees an outcome other than its expectation has run its checks.

Run:  python -B scripts/smoke_robustness_report.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import scripts.rehearse_robustness_val as RH  # noqa: E402
import scripts.report_robustness as RP  # noqa: E402
import scripts.run_corruption_grid as RG  # noqa: E402
import src.eval.corrupted as CO  # noqa: E402
import src.eval.robustness_report as RR  # noqa: E402
import src.stats.report as SR  # noqa: E402
from src import corruption_roles  # noqa: E402
from src.corruption_cache import _strict_json  # noqa: E402
from src.eval.artifacts import (EVAL_RUNTIME_VERSION, DatasetMeta, RunMeta,  # noqa: E402
                                prepare_artifact_request, validate_artifact_request,
                                write_artifact)
from src.eval.eval_runtime import RECORD_KEYS  # noqa: E402
from src.eval.evaluate import Condition, EvalResult, ManifestEntry, PerImageRow  # noqa: E402
from src.eval.protocols import CANVAS_PROTOCOL_ID  # noqa: E402

C = 116
CASES: dict = {}
S: dict = {}
results: list = []
_SEQ = itertools.count()
CELLS = corruption_roles.CELL_IDS
INF_CELLS = corruption_roles.INFERENTIAL_CELL_IDS
CORR = corruption_roles.CANONICAL_IDS
INT8 = ("int8_ptq", "int8_qat")
#: The case's own copy of LE6's two series (never read from the module under check).
SERIES_EXPECTED = (("non_noise_s1_5", ("motion_blur", "jpeg_compression", "brightness", "fog"),
                    (1, 2, 3, 4, 5)),
                   ("all_five_s1_4", CORR, (1, 2, 3, 4)))
DIGEST = "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b"
CACHE = hashlib.sha256(b"s5 cache").hexdigest()
CACHE_OTHER = hashlib.sha256(b"s5 another cache").hexdigest()
INF_SHA = hashlib.sha256(b"s5 inferential cells").hexdigest()
VREC = hashlib.sha256(b"s5 verification record").hexdigest()
SHA = {k: hashlib.sha256(f"s5 model {k}".encode()).hexdigest()
       for k in ("e1s42", "e6src", "e6conv", "tch", "e1s43", "e1b", "e1s44", "e1o", "e1v", "e1s44v")}
CLASS_MAP = ([{"class_id": 0, "name": "", "role": "background_or_non_disease"}]
             + [{"class_id": c, "name": f"syn_{c:03d}", "role": "disease"} for c in range(1, C)])
CTX = {"model_device": "cpu", "input_devices": ["cpu"], "determinism_policy_applied": False,
       "batch_size": 1, "num_workers": 0, "torch_num_threads": 1, "image_digest": DIGEST}
A_IDS = ["img_a", "img_b"]
O_IDS = [f"o_{i:05d}" for i in range(1561)]
H_IDS = [f"v_{i:05d}" for i in range(846)]
H_AM5 = "v_00845"
#: Fixture A models: stage, seed, precision, role, the identity the artifacts carry, the export's
#: source sha (INT8: the provenance), the clean base k, the cells scored and the cells written.
A_MODELS = {
    "e1s42": dict(stage="E1", seed=42, sha=SHA["e1s42"], base=90, cells=CELLS, threads=1),
    "e6s42": dict(stage="E6", seed=42, precision="int8_qat", sha=SHA["e6conv"], src=SHA["e6src"],
                  base=86, cells=CELLS, threads=2),
    "tch": dict(stage="teacher", seed=None, role="teacher", sha=SHA["tch"], base=95, cells=CELLS,
                threads=1),
    "e1s43": dict(stage="E1", seed=43, sha=SHA["e1s43"], base=88, cells=INF_CELLS, threads=1),
    "e1b": dict(stage="E1", seed=42, sha=SHA["e1b"], base=70, cells=CELLS, better=("brightness",),
                threads=1),
    "e1s44": dict(stage="E1", seed=44, sha=SHA["e1s44"], base=87, cells=INF_CELLS,
                  written=tuple(c for c in INF_CELLS if c != "fog_s3"), threads=1),
}
O_MODEL = dict(stage="E1", seed=42, sha=SHA["e1o"], base=90, cells=CELLS, threads=1)
H_MODELS = {"e1v": dict(stage="E1", seed=42, sha=SHA["e1v"], base=90, cells=INF_CELLS, threads=1),
            "e1s44v": dict(stage="E1", seed=44, sha=SHA["e1s44v"], base=84, cells=INF_CELLS,
                           threads=1)}


# ---------------------------------------------------------------- harness (S4's)
def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def run(name) -> tuple[bool, str]:
    try:
        ok, detail = CASES[name]()
        return bool(ok), str(detail)
    except Exception as e:                                   # noqa: BLE001
        return False, f"raised {type(e).__name__}: {e}"


@contextlib.contextmanager
def patched(obj, attr, value):
    saved = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, saved)


def mutant(label, targets, *patches):
    """Killed only when every named case fails by its own checks: a case that raised is no kill."""
    with contextlib.ExitStack() as stack:
        for obj, attr, value in patches:
            stack.enter_context(patched(obj, attr, value))
        outcomes = {t: run(t) for t in targets}
    killed = all(not ok and not d.startswith("raised ") for ok, d in outcomes.values())
    why = "; ".join(f"{t}: {d[:90]}" for t, (_, d) in outcomes.items())
    print(f"MUTANT {label} {'killed' if killed else 'SURVIVED'} by {', '.join(targets)} [{why}]")
    results.append((f"{label} killed by {', '.join(targets)}", killed, why))


def skip(mod, helper: str, code: str):
    """A patch that disables one refusal code at its raising helper (the gate removed)."""
    real = getattr(mod, helper)

    def gate(c, *a, **k):
        if c == code:
            return None
        return real(c, *a, **k)
    return (mod, helper, gate)


def outcome(fn, *a, **kw) -> tuple[str, object]:
    """('ok', value), or (the refusal code, else the exception type name, exception)."""
    try:
        return "ok", fn(*a, **kw)
    except Exception as e:                                   # noqa: BLE001 -- recorded
        return (getattr(e, "code", None) or type(e).__name__), e


def cli(main, argv) -> tuple[str, object, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code, value = outcome(main, [str(a) for a in argv])
    return code, value, out.getvalue(), err.getvalue()


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fresh(tag: str) -> Path:
    """A never-yet-created path under the work folder (no 'test' in it)."""
    return S["work"] / "out" / f"{tag}_{next(_SEQ)}"


def never(tag: str) -> Path:
    return S["work"] / "never_created" / f"{tag}_{next(_SEQ)}"


def refused(got: tuple, code: str, *needles) -> tuple[bool, str]:
    c, v = got
    msg = str(v)
    ok = c == code and all(n in msg for n in needles)
    return ok, f"outcome {c}: {msg[:160]}"


# ---------------------------------------------------------------- closed forms
def kj(base: int, label: str, better=()) -> tuple[int, int]:
    """The fixture's totals: class 0 tp = k of gt = pred = ... (union 100), class 1 tp = j (union 10)."""
    if label == "clean":
        return base, 8
    c, s = corruption_roles.parse_cell_id(label)
    if c in better:
        return min(100, base + 5 + s), 8
    ci = CORR.index(c)
    return base - 6 * s - 2 * ci, 7 - s - ci % 2


def exact_m(k: int, j: int) -> Fraction:
    return (Fraction(k, 100) + Fraction(j, 10)) / 2


def exact_table(spec) -> dict:
    return {corruption_roles.parse_cell_id(c): exact_m(*kj(spec["base"], c, spec.get("better", ())))
            for c in spec["cells"]}


def exact_clean(spec) -> Fraction:
    return exact_m(*kj(spec["base"], "clean"))


def exact_miou_c(spec) -> Fraction:
    m = exact_table(spec)
    return sum(sum(m[(c, s)] for s in (1, 2, 3)) / 3 for c in CORR) / len(CORR)


def exact_series(spec) -> tuple[dict, dict]:
    m = exact_table(spec)
    q2 = {name: {str(s): sum(m[(c, s)] for c in cs) / len(cs) for s in ss}
          for name, cs, ss in SERIES_EXPECTED}
    q8 = {name: sum(sum(m[(c, s)] for s in ss) / len(ss) for c in cs) / len(cs)
          for name, cs, ss in SERIES_EXPECTED}
    flat = {name: sum(m[(c, s)] for c in cs for s in ss) / (len(cs) * len(ss))
            for name, cs, ss in SERIES_EXPECTED}
    return q2, {"nested": q8, "flat": flat}


def exact_degradation(spec, corruption) -> Fraction:
    m, clean = exact_table(spec), exact_clean(spec)
    return sum(clean - m[(corruption, s)] for s in (1, 2, 3))


def close(got, want, tol=1e-12) -> bool:
    return got is not None and abs(float(got) - float(want)) <= tol


# ---------------------------------------------------------------- fixtures
def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=s5-fixture", "-c", "user.email=s5-fixture@invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), capture_output=True, text=True, check=False,
        env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(repo.parent)))
    if proc.returncode != 0:
        raise AssertionError(f"fixture git {args} failed ({proc.returncode}): {proc.stderr[:200]}")
    return proc.stdout


def make_fixture_repo(root: Path) -> Path:
    """A throwaway repository (never the project), clean, so official artifacts can be written."""
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "seed.txt").write_text("s5 fixture\n", encoding="utf-8")
    _git(root, "add", "--", "seed.txt")
    _git(root, "commit", "-q", "-m", "fixture baseline")
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root.resolve():
        raise AssertionError("the fixture is not its own repository")
    return root


def runtime(threads: int = 1) -> dict:
    rt = {k: None for k in RECORD_KEYS}
    rt.update({"eval_runtime_version": EVAL_RUNTIME_VERSION, "model_device": "cpu",
               "model_tensor_devices": ["cpu"], "input_devices": ["cpu"], "forward_batches": 1,
               "batch_size": 1, "num_workers": 0, "determinism_policy_applied": False,
               "determinism": {}, "cudnn_enabled": True, "torch_num_threads": threads,
               "pillow": "12.3.0", "image_digest": DIGEST, "eval_warnings": [],
               "eval_warnings_truncated": False, "nondeterministic_alert_count": 0})
    return rt


def condition_of(label: str) -> Condition:
    if label == "clean":
        return Condition()
    name, severity = corruption_roles.parse_cell_id(label)
    return Condition("corruption", name, severity)


def make_result(ids, cond, totals, *, all_vals, dis_vals, am5=()) -> EvalResult:
    rows = []
    for i, iid in enumerate(ids):
        ex = iid in am5
        rows.append(PerImageRow(
            image_id=iid, clean_image_id=iid, manifest_index=i, condition=cond,
            all_class_miou=float(all_vals[i]), all_class_miou_status="ok",
            disease_only_miou=None if ex else float(dis_vals[i]),
            disease_only_miou_status="undefined_no_eligible_class" if ex else "ok",
            n_eligible_all_class=1 if ex else 2, n_eligible_disease_only=0 if ex else 1,
            gt_disease_classes=[] if ex else [1]))
    s_img, s_cls, s_tp, s_gt, s_pr = [], [], [], [], []
    d_tp, d_gt, d_pr = (np.zeros(C, np.int64) for _ in range(3))
    for pos, cid in enumerate(sorted(totals)):
        tp, gt, pr = totals[cid]
        s_img.append(pos % len(ids)); s_cls.append(cid); s_tp.append(tp); s_gt.append(gt)
        s_pr.append(pr)
        d_tp[cid] += tp; d_gt[cid] += gt; d_pr[cid] += pr
    un = d_gt + d_pr - d_tp
    el_u, el_g = un > 0, d_gt > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(el_u, d_tp / np.maximum(un, 1), np.nan)
        dice = np.where((d_gt + d_pr) > 0, 2 * d_tp / np.maximum(d_gt + d_pr, 1), np.nan)
        acc = np.where(el_g, d_tp / np.maximum(d_gt, 1), np.nan)
    return EvalResult(
        rows=rows, manifest_ids=list(ids),
        sparse_image_index=np.array(s_img, np.int64), sparse_class_id=np.array(s_cls, np.int64),
        sparse_tp=np.array(s_tp, np.int64), sparse_gt=np.array(s_gt, np.int64),
        sparse_pred=np.array(s_pr, np.int64), dataset_tp=d_tp, dataset_gt=d_gt, dataset_pred=d_pr,
        dataset_level={"all_class_miou": float(np.nanmean(iou[el_u])),
                       "all_class_miou_n_eligible": int(el_u.sum()),
                       "all_class_macro_dice": float(np.nanmean(dice[el_u])),
                       "all_class_dice_n_eligible": int(el_u.sum()),
                       "all_class_macc": float(np.nanmean(acc[el_g])),
                       "all_class_macc_n_eligible": int(el_g.sum()),
                       "disease_only_miou": float(iou[1]), "disease_only_miou_n_eligible": 1,
                       "aacc_diagnostic": float(d_tp.sum() / d_gt.sum())},
        per_class={"class_ids": list(range(C)), "gt_support": d_gt.tolist(),
                   "pred_support": d_pr.tolist(), "intersection": d_tp.tolist(),
                   "union": un.tolist(),
                   "iou": [None if not el_u[c] else float(iou[c]) for c in range(C)],
                   "dice": [None if not el_u[c] else float(dice[c]) for c in range(C)],
                   "acc": [None if not el_g[c] else float(acc[c]) for c in range(C)],
                   "iou_status": ["ok" if el_u[c] else "undefined_absent_from_gt_and_pred"
                                  for c in range(C)]},
        integrity={"row_count_ok": True, "ids_unique": True, "ids_match_manifest": True,
                   "order_canonical": True, "duplicate_ids": [], "missing_ids": [],
                   "invalid_pred_labels": 0, "undefined_per_image_scores": 0,
                   "overwrite_policy": "refuse_existing"},
        num_classes=C, background_index=0, ignore_index=255, condition=cond)


def dir_name(split: str, file_id: str, label: str) -> str:
    return (CO.clean_dir_name(split, file_id) if label == "clean"
            else CO.corrupted_dir_name(split, file_id, label))


def write_art(root: Path, name: str, *, stage, sha, label, ids, totals, role="student",
              precision="fp32", status="smoke", split="val", protocol=CANVAS_PROTOCOL_ID,
              all_vals=None, dis_vals=None, am5=(), rt=None) -> Path:
    cond = condition_of(label)
    n = len(ids)
    res = make_result(ids, cond, totals, all_vals=all_vals if all_vals is not None else [0.5] * n,
                      dis_vals=dis_vals if dis_vals is not None else [0.4] * n, am5=am5)
    req = prepare_artifact_request(
        out_dir=root / name, artifact_status=status, run_id=f"run_{name}",
        run=RunMeta(stage=stage, model_role=role, precision=precision,
                    quant_backend="qnnpack" if precision in INT8 else None,
                    checkpoint_path=f"/models/{name}.bin", checkpoint_sha256=sha,
                    random_init=False, device="cpu"),
        dataset=DatasetMeta(name="SYNTHETIC S5 fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split=split, condition=cond,
                            preprocess_protocol=protocol, expected_rows=n),
        expected_manifest=[ManifestEntry(i, iid, iid) for i, iid in enumerate(ids)],
        class_map=CLASS_MAP, repo_root=S["repo"])
    prov = validate_artifact_request(req)
    return write_artifact(res, req, prov, timestamp_utc="2026-10-17T00:00:00Z", eval_runtime=rt)


def write_record(d: Path, *, cache=CACHE, mutate=None) -> Path:
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    run, ds = summary["run"], summary["dataset"]
    cid = f"{ds['condition']['name']}_s{ds['condition']['severity']}"
    rows = ds["actual_rows"]
    record = CO.build_cell_record(
        out_dir=d, run_id=run["run_id"], stage=run["stage"], condition=ds["condition"],
        role=RR.cell_role(cid),
        manifest={"canonical_manifest_sha256": cache, "inferential_cells_sha256": INF_SHA,
                  "status": run["artifact_status"], "source_image_count": rows},
        cell_sha256=hashlib.sha256(cid.encode()).hexdigest(), verification_record_sha256=VREC,
        rows=rows, items_verified=rows, source_digests_verified=rows,
        evaluator_commit=run["repo_commit"], started_utc="2026-10-17T00:00:00.000000Z",
        split=ds["split"])
    if mutate is not None:
        mutate(record)
    return CO.write_cell_record(CO.record_path(d), record)


def totals_for(spec, label) -> dict:
    k, j = kj(spec["base"], label, spec.get("better", ()))
    return {0: (k, 100, k), 1: (j, 10, j)}


def write_model(root: Path, file_id: str, spec: dict, *, ids, split="val", status="smoke",
                values=None, am5=()) -> None:
    """The clean artifact and every written cell (artifact and record) of one model."""
    for label in ("clean", *spec.get("written", spec["cells"])):
        kw = values(file_id, label) if values is not None else {}
        d = write_art(root, dir_name(split, file_id, label), stage=spec["stage"], sha=spec["sha"],
                      label=label, ids=ids, totals=totals_for(spec, label),
                      role=spec.get("role", "student"), precision=spec.get("precision", "fp32"),
                      status=status, split=split, am5=am5, rt=runtime(spec["threads"]), **kw)
        if label != "clean":
            write_record(d)


def row_for(file_id: str, spec: dict, **over) -> dict:
    prec = spec.get("precision", "fp32")
    row = {"file_id": file_id, "stage": spec["stage"], "model_role": spec.get("role", "student"),
           "precision": prec, "seed": spec["seed"],
           "source": {"kind": "provenance" if prec in INT8 else "checkpoint",
                      "path": f"models/{file_id}.bin", "sha256": spec.get("src", spec["sha"])},
           "converted_artifact_sha256": spec["sha"] if prec in INT8 else None,
           "cells": list(spec["cells"])}
    row.update(over)
    return row


def export(rows, *, commit=None, ctx=None) -> dict:
    return {"schema": RG.INVENTORY_SCHEMA, "evaluator_commit": commit or S["commit"],
            "execution_context": dict(ctx or CTX), "manifest_sha256": "ab" * 32,
            "models": rows}


def a_rows(*fids, **over) -> list:
    return [row_for(f, A_MODELS[f], **over.get(f, {})) for f in fids]


def variant(tag: str, src: Path, file_id: str, spec: dict, *, split="val", skip_dirs=(),
            skip_records=(), substitute=None) -> Path:
    """A new root with copies of one model's artifacts and records, minus the named ones;
    `substitute` maps a label to the label whose directory and record are copied in its place."""
    root = S["work"] / f"v_{tag}"
    root.mkdir()
    for label in ("clean", *spec["cells"]):
        if label in skip_dirs:
            continue
        source = (substitute or {}).get(label, label)
        shutil.copytree(src / dir_name(split, file_id, source), root / dir_name(split, file_id, label))
        if label != "clean" and label not in skip_records:
            shutil.copyfile(CO.record_path(src / dir_name(split, file_id, source)),
                            CO.record_path(root / dir_name(split, file_id, label)))
    return root


def h_values(file_id: str, label: str) -> dict:
    """Per-image values of the H fixture, seeded per model and cell, kept for R-G5's own check."""
    seed = 1000 + 100 * list(H_MODELS).index(file_id) + ("clean", *INF_CELLS).index(label)
    rs = np.random.RandomState(seed)
    dis, alls = rs.uniform(0.05, 0.95, len(H_IDS)), rs.uniform(0.05, 0.95, len(H_IDS))
    S["h_dis"][(file_id, label)] = dis
    return {"dis_vals": dis.tolist(), "all_vals": alls.tolist()}


def write_inventory(tag: str, doc: dict) -> Path:
    p = S["work"] / "inventories" / f"{tag}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return p


def setup(work: Path) -> None:
    S["work"] = work
    S["repo"] = make_fixture_repo(work / "fixture_repo")
    S["commit"] = _git(S["repo"], "rev-parse", "HEAD").strip()
    S["h_dis"] = {}
    a = S["a"] = work / "a_root"
    a.mkdir()
    for fid, spec in A_MODELS.items():
        write_model(a, fid, spec, ids=A_IDS)
    S["export_a"] = export(a_rows("e1s42", "e6s42", "tch", "e1s43"))
    S["export_b"] = export(a_rows("e1b", "e6s42", "tch"))
    o = S["o"] = work / "o_root"
    o.mkdir()
    write_model(o, "e1o", O_MODEL, ids=O_IDS, split="test", status="official")
    S["export_o"] = export([row_for("e1o", O_MODEL)])
    h = S["h"] = work / "h_root"
    h.mkdir()
    for fid, spec in H_MODELS.items():
        write_model(h, fid, spec, ids=H_IDS, status="provisional", values=h_values, am5=(H_AM5,))
    S["export_h"] = export([row_for(fid, spec) for fid, spec in H_MODELS.items()])


def doc_a(**kw):
    return outcome(RR.build_report, kw.pop("export", S["export_a"]), kw.pop("root", S["a"]),
                   split=kw.pop("split", "val"), policy=kw.pop("policy", "smoke"),
                   expected_cache=kw.pop("expected_cache", CACHE),
                   inventory_sha256=kw.pop("inventory_sha256", "0" * 64))


def views_for(models: dict):
    loaded = {}
    for stage, model in models.items():
        for cid, run_ in model.cells.items():
            name, sev = corruption_roles.parse_cell_id(cid)
            loaded[(stage, ("corruption", name, sev))] = types.SimpleNamespace(run=run_)
    return types.SimpleNamespace(loaded=loaded)


# ---------------------------------------------------------------- Q: the quantities
@case("Q1")
def q1():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    for fid in ("e1s42", "e6s42", "tch", "e1s43"):
        spec, got = A_MODELS[fid], doc["q1_per_cell"][fid]
        if list(got) != list(spec["cells"]):
            bad.append((fid, "cell set"))
            continue
        for cid in spec["cells"]:
            k, j = kj(spec["base"], cid)
            if not close(got[cid]["all_class_miou"], exact_m(k, j)) or \
                    not close(got[cid]["disease_only_miou"], Fraction(j, 10)):
                bad.append((fid, cid))
    n = sum(len(v) for v in doc["q1_per_cell"].values())
    return not bad and n == 24 * 3 + 15, f"{n} cells; bad {bad[:3]}"


@case("Q2")
def q2():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    if tuple((n, tuple(c), tuple(s)) for n, c, s in RR.SERIES) != SERIES_EXPECTED:
        bad.append("RR.SERIES is not LE6's pair")
    if sorted(doc["q2_per_severity"]) != ["e1s42", "e6s42", "tch"]:
        bad.append(("models", sorted(doc["q2_per_severity"])))
    for fid in ("e1s42", "e6s42", "tch"):
        want, _ = exact_series(A_MODELS[fid])
        got = doc["q2_per_severity"].get(fid, {})
        for name, per in want.items():
            if sorted(got.get(name, {})) != sorted(per):
                bad.append((fid, name, "severities"))
            for s, v in per.items():
                if not close(got.get(name, {}).get(s), v):
                    bad.append((fid, name, s))
    return not bad, f"bad {bad[:4]}"


@case("Q3")
def q3():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = [fid for fid in ("e1s42", "e6s42", "tch", "e1s43")
           if not close(doc["q3_miou_c"][fid], exact_miou_c(A_MODELS[fid]))]
    views = views_for({"E1": S["model_a"]["e1s42"], "E6": S["model_a"]["e6s42"]})
    for stage, fid in (("E1", "e1s42"), ("E6", "e6s42")):
        if doc["q3_miou_c"][fid] != SR.dataset_miou_c(views, stage):
            bad.append(("P28 bitwise", fid))
    return not bad, f"E1 {doc['q3_miou_c']['e1s42']!r}; seed 43 {doc['q3_miou_c']['e1s43']!r}; bad {bad}"


@case("Q3b")
def q3b():
    code, doc = outcome(RR.build_report, S["export_h"], S["h"], split="val", policy="rehearsal",
                        expected_cache=CACHE)
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    models = {fid: RR.load_model(S["h"], "val", row_for(fid, spec), "rehearsal")
              for fid, spec in H_MODELS.items()}
    bitwise = all(doc["q3_miou_c"][fid] == SR.dataset_miou_c(views_for({"E1": models[fid]}), "E1")
                  for fid in H_MODELS)
    ones = all(doc["q5_rcd"]["e1v"][c]["value"] == 1.0 for c in CORR)
    only42 = sorted(doc["q5_rcd"]) == ["e1v"] and doc["q6_rcd_mean"]["e1v"]["value"] == 1.0
    return bitwise and ones and only42 and doc["status"] == "nonofficial", \
        f"bitwise {bitwise}; rCD(E1) == 1: {ones}; rCD subjects {sorted(doc['q5_rcd'])}"


@case("Q4")
def q4():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    for fid in ("e1s42", "e6s42", "tch", "e1s43"):
        spec = A_MODELS[fid]
        clean, mc = exact_clean(spec), exact_miou_c(spec)
        want = (clean - mc) / clean * 100
        got = doc["q4_rpd"][fid]
        if got["reason"] is not None or not close(got["value"], want, 1e-10):
            bad.append((fid, got))
    return not bad, f"teacher {doc['q4_rpd']['tch']}; bad {bad[:2]}"


@case("Q5")
def q5():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    if sorted(doc["q5_rcd"]) != ["e1s42", "e6s42"]:
        bad.append(("subjects", sorted(doc["q5_rcd"])))
    for c in CORR:
        if doc["q5_rcd"].get("e1s42", {}).get(c, {}).get("value") != 1.0:
            bad.append(("E1 not exactly 1", c))
        want = exact_degradation(A_MODELS["e6s42"], c) / exact_degradation(A_MODELS["e1s42"], c)
        if not close(doc["q5_rcd"].get("e6s42", {}).get(c, {}).get("value"), want):
            bad.append(("E6", c))
    return not bad, f"E6 {[round(doc['q5_rcd'].get('e6s42', {}).get(c, {}).get('value') or 0, 6) for c in CORR]}; bad {bad[:3]}"


@case("Q6")
def q6():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    want = sum(exact_degradation(A_MODELS["e6s42"], c) / exact_degradation(A_MODELS["e1s42"], c)
               for c in CORR) / len(CORR)
    e1, e6 = doc["q6_rcd_mean"]["e1s42"], doc["q6_rcd_mean"]["e6s42"]
    ok = e1 == {"value": 1.0, "reason": None} and close(e6["value"], want) and e6["reason"] is None
    return ok and sorted(doc["q6_rcd_mean"]) == ["e1s42", "e6s42"], f"E1 {e1}; E6 {e6}"


@case("Q7")
def q7():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    if sorted(doc["q7_per_corruption"]) != ["e1s42", "e6s42", "tch"]:
        bad.append(("models", sorted(doc["q7_per_corruption"])))
    for fid in ("e1s42", "e6s42", "tch"):
        m = exact_table(A_MODELS[fid])
        for c in CORR:
            got = doc["q7_per_corruption"].get(fid, {}).get(c, {})
            if not close(got.get("mean_s1_3"), sum(m[(c, s)] for s in (1, 2, 3)) / 3):
                bad.append((fid, c))
            if ("rcd" in got) != (fid != "tch"):
                bad.append((fid, c, "rcd column"))
            elif fid != "tch" and got["rcd"] != doc["q5_rcd"][fid][c]["value"]:
                bad.append((fid, c, "rcd value"))
    return not bad, f"bad {bad[:3]}"


@case("Q8")
def q8():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    for fid in ("e1s42", "e6s42", "tch"):
        _, want = exact_series(A_MODELS[fid])
        for name, _, _ in SERIES_EXPECTED:
            got = doc["q8_kamann_rother"].get(fid, {}).get(name)
            if want["nested"][name] != want["flat"][name] or not close(got, want["nested"][name]):
                bad.append((fid, name, got))
    if sorted(doc["q8_kamann_rother"]) != ["e1s42", "e6s42", "tch"]:
        bad.append(("models", sorted(doc["q8_kamann_rother"])))
    return not bad, f"E1 {doc['q8_kamann_rother'].get('e1s42')}; bad {bad[:2]}"


@case("Q9a")
def q9a():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    present = (doc["q3_miou_c"].get("tch") is not None and doc["q4_rpd"]["tch"]["value"] is not None
               and all(v is not None for v in doc["q8_kamann_rother"]["tch"].values()))
    absent = "tch" not in doc["q5_rcd"] and "tch" not in doc["q6_rcd_mean"] and \
        all("rcd" not in v for v in doc["q7_per_corruption"]["tch"].values())
    return present and absent, f"teacher in Q3/Q4/Q8 {present}; absent from Q5/Q6 {absent}"


@case("Q9b")
def q9b():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    for m in doc["models"]:
        for label, art in m["artifacts"].items():
            d = S["a"] / art["dir"]
            if art["file_sha256"] != {n: sha256_file(d / n) for n in CO.ARTIFACT_FOUR_FILES}:
                bad.append((art["dir"], "files"))
            want_rec = None if label == "clean" else sha256_file(CO.record_path(d))
            if art["record_sha256"] != want_rec:
                bad.append((art["dir"], "record"))
            if art["repo_commit"] != S["commit"] or \
                    art["runtime"]["torch_num_threads"] != A_MODELS[m["file_id"]]["threads"]:
                bad.append((art["dir"], "runtime"))
    code_prov = doc["provenance"]["code"]
    ok = (not bad and doc["cache"]["canonical_manifest_sha256"] == CACHE
          and doc["cache"]["verification_record_sha256"] == [VREC]
          and doc["inventory"]["sha256"] == "0" * 64
          and re.fullmatch(r"[0-9a-f]{40}", str(code_prov["code_commit"])) is not None
          and sorted(code_prov["code_files"]) == sorted(RR.CODE_FILES)
          and sum(len(m["artifacts"]) for m in doc["models"]) == 25 * 3 + 16)
    return ok, f"{sum(len(m['artifacts']) for m in doc['models'])} inputs; bad {bad[:3]}"


# ---------------------------------------------------------------- V: checks and refusals
@case("V1a")
def v1a():
    root = S.get("v1a_root") or variant("o_missing", S["o"], "e1o", O_MODEL, split="test",
                                        skip_dirs=("fog_s5",))
    S["v1a_root"] = root
    return refused(outcome(RR.build_report, S["export_o"], root, split="test", policy="official",
                           expected_cache=CACHE), "artifact_missing", "e1o:fog_s5")


@case("V1b")
def v1b():
    code, doc = doc_a(export=export(a_rows("e1s42", "e1s44")))
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    m = {x["file_id"]: x for x in doc["models"]}
    ok = (m["e1s44"]["missing"] == ["fog_s3"] and doc["q3_miou_c"]["e1s44"] is None
          and doc["q4_rpd"]["e1s44"] == {"value": None, "reason": "cell_missing"}
          and m["e1s42"]["missing"] == [] and doc["q3_miou_c"]["e1s42"] is not None)
    return ok, f"missing {m['e1s44']['missing']}; Q4 {doc['q4_rpd']['e1s44']}"


@case("V2")
def v2():
    root = S.get("v2_root") or variant("dup", S["a"], "e1s42", A_MODELS["e1s42"],
                                       substitute={"fog_s1": "fog_s2"})
    S["v2_root"] = root
    got = outcome(RR.build_report, export(a_rows("e1s42")), root, split="val", policy="smoke",
                  expected_cache=CACHE)
    ok, detail = refused(got, "condition_mismatch", "fog")
    inv = S["work"] / "inventories" / "dup_cell.json"
    if not inv.exists():
        write_inventory("dup_cell", export(a_rows("e1s42", e1s42={"cells": ["fog_s1", "fog_s1"]})))
    listed = outcome(RG.load_inventory, inv)
    return ok and listed[0] == "inventory_invalid", f"{detail}; a cell listed twice: {listed[0]}"


def rewrite_one(tag: str, label: str, **kw) -> Path:
    """Model E1 (fixture A) copied, with one cell rewritten by write_art(**kw) and recorded."""
    root = variant(tag, S["a"], "e1s42", A_MODELS["e1s42"], skip_dirs=(label,))
    spec = A_MODELS["e1s42"]
    args = dict(stage=spec["stage"], sha=spec["sha"], label=label, ids=A_IDS,
                totals=totals_for(spec, label), rt=runtime(1))
    args.update(kw)
    d = write_art(root, dir_name("val", "e1s42", label), **args)
    write_record(d)
    return root


@case("V3")
def v3():
    root = S.get("v3_root") or rewrite_one("digest", "fog_s2", ids=["img_a", "img_c"])
    S["v3_root"] = root
    return refused(outcome(RR.build_report, export(a_rows("e1s42")), root, split="val",
                           policy="smoke", expected_cache=CACHE), "split_digest_mismatch")


@case("V4")
def v4():
    root = S.get("v4_root") or rewrite_one("canvas", "fog_s2", protocol="synthetic-stats/1.0.0")
    S["v4_root"] = root
    return refused(outcome(RR.build_report, export(a_rows("e1s42")), root, split="val",
                           policy="smoke", expected_cache=CACHE), "non_canvas_input",
                   "synthetic-stats")


@case("V5")
def v5():
    root = S.get("v5_root") or rewrite_one("ckpt", "fog_s2", sha="f" * 64)
    S["v5_root"] = root
    return refused(outcome(RR.build_report, export(a_rows("e1s42")), root, split="val",
                           policy="smoke", expected_cache=CACHE), "checkpoint_identity_mismatch",
                   "source.sha256")


@case("V6a")
def v6a():
    root = S.get("v6_root")
    if root is None:
        root = S["v6_root"] = variant("cache", S["a"], "e1s42", A_MODELS["e1s42"],
                                      skip_records=("fog_s2",))
        write_record(root / dir_name("val", "e1s42", "fog_s2"), cache=CACHE_OTHER)
    return refused(outcome(RR.build_report, export(a_rows("e1s42")), root, split="val",
                           policy="smoke", expected_cache=None), "cache_checksum_mismatch")


@case("V6b")
def v6b():
    return refused(doc_a(expected_cache=CACHE_OTHER), "cache_checksum_mismatch", CACHE_OTHER)


@case("V6c")
def v6c():
    return refused(outcome(RR.build_report, S["export_o"], S["o"], split="test", policy="official",
                           expected_cache=None), "expected_cache_sha256_required")


@case("V6d")
def v6d():
    return refused(doc_a(expected_cache="ABC"), "expected_cache_sha256_format")


@case("V7")
def v7():
    code, doc = doc_a(export=S["export_b"])
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    bad = []
    null = {"value": None, "reason": "rcd_reference_nonpositive"}
    if exact_degradation(A_MODELS["e1b"], "brightness") > 0:
        bad.append("fixture: E1b's brightness degradation is positive")
    for fid in ("e1b", "e6s42"):
        if doc["q5_rcd"][fid]["brightness"] != null:
            bad.append((fid, doc["q5_rcd"][fid]["brightness"]))
        if doc["q6_rcd_mean"][fid] != null:
            bad.append((fid, "mean", doc["q6_rcd_mean"][fid]))
    for c in CORR:
        if c == "brightness":
            continue
        if doc["q5_rcd"]["e1b"][c]["value"] != 1.0:
            bad.append(("e1b", c))
        want = exact_degradation(A_MODELS["e6s42"], c) / exact_degradation(A_MODELS["e1b"], c)
        if not close(doc["q5_rcd"]["e6s42"][c]["value"], want):
            bad.append(("e6s42", c))
    return not bad, f"E6 brightness {doc['q5_rcd']['e6s42']['brightness']}; bad {bad[:3]}"


@case("V8a")
def v8a():
    target = REPO / f"s5_never_created_out_{next(_SEQ)}"
    got = outcome(RR.check_out_dir, target)
    ok, detail = refused(got, "out_dir_inside_repo")
    return ok and not target.exists(), detail


@case("V8b")
def v8b():
    target = S["work"] / "a_root"
    return refused(outcome(RR.check_out_dir, target), "out_dir_exists")


@case("V8c")
def v8c():
    target = never("tday_TeSt_out")
    ok, detail = refused(outcome(RR.check_out_dir, target), "out_dir_name_invalid")
    return ok and not target.exists(), detail


@case("V9a")
def v9a():
    root = S.get("v9_root")
    if root is None:
        root = S["v9_root"] = variant("o_het", S["o"], "e1o", O_MODEL, split="test",
                                      skip_dirs=("fog_s5",))
        d = write_art(root, dir_name("test", "e1o", "fog_s5"), stage="E1", sha=O_MODEL["sha"],
                      label="fog_s5", ids=O_IDS, totals=totals_for(O_MODEL, "fog_s5"),
                      status="official", split="test", rt=runtime(2))
        write_record(d)
    return refused(outcome(RR.build_report, S["export_o"], root, split="test", policy="official",
                           expected_cache=CACHE), "heterogeneous_runtime", "fog_s5",
                   "'torch_num_threads': 2")


@case("V9b")
def v9b():
    return refused(outcome(RR.build_report, export([row_for("e1o", O_MODEL)], commit="0" * 40),
                           S["o"], split="test", policy="official", expected_cache=CACHE),
                   "heterogeneous_runtime", "0" * 40)


@case("V9c")
def v9c():
    ok_code, doc = doc_a(export=export(a_rows("e6s42")))
    accepted = ok_code == "ok" and doc["models"][0]["identity_sha256"] == SHA["e6conv"]
    wrong_conv = outcome(RR.build_report, export(a_rows("e6s42", e6s42={
        "converted_artifact_sha256": "d" * 64})), S["a"], split="val", policy="smoke",
        expected_cache=CACHE)
    wrong_src = outcome(RR.build_report, export(a_rows("e1s42", e1s42={
        "source": {"kind": "checkpoint", "path": "models/e1s42.bin", "sha256": "e" * 64}})),
        S["a"], split="val", policy="smoke", expected_cache=CACHE)
    r1 = refused(wrong_conv, "checkpoint_identity_mismatch", "converted_artifact_sha256")
    r2 = refused(wrong_src, "checkpoint_identity_mismatch", "source.sha256")
    return accepted and r1[0] and r2[0], f"INT8 accepted {accepted}; {r1[1][:70]}; {r2[1][:70]}"


@case("V9d")
def v9d():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    threads = {m["file_id"]: {a["runtime"]["torch_num_threads"] for a in m["artifacts"].values()}
               for m in doc["models"]}
    return threads == {"e1s42": {1}, "e6s42": {2}, "tch": {1}, "e1s43": {1}} and \
        doc["status"] == "nonofficial", f"recorded thread counts {threads}"


@case("V9e")
def v9e():
    code, doc = outcome(RR.build_report, S["export_o"], S["o"], split="test", policy="official",
                        expected_cache=CACHE)
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    rts = [a["runtime"] for a in doc["models"][0]["artifacts"].values()]
    ok = (doc["status"] == "official" and doc["banner"] is None and doc["nonofficial_inputs"] == []
          and all(rt == CTX for rt in rts) and len(rts) == 25
          and all(doc["q5_rcd"]["e1o"][c]["value"] == 1.0 for c in CORR)
          and RR.RUNTIME_KEYS == RG.CONTEXT_KEYS)
    return ok, f"status {doc['status']}; {len(rts)} runtime tuples equal to the export's: " \
               f"{all(rt == CTX for rt in rts)}"


@case("V10a")
def v10a():
    return refused(outcome(RR.build_report, S["export_a"], never("root"), split="val",
                           policy="final"), "policy_unknown")


@case("V10b")
def v10b():
    r1 = refused(outcome(RR.build_report, S["export_o"], never("root"), split="val",
                         policy="official"), "policy_split_mismatch")
    r2 = refused(outcome(RR.build_report, S["export_a"], never("root"), split="test",
                         policy="smoke"), "policy_split_mismatch")
    return r1[0] and r2[0], f"{r1[1][:80]}; {r2[1][:80]}"


@case("V11")
def v11():
    return refused(doc_a(policy="rehearsal"), "input_refused", "real-run artifact")


@case("V12")
def v12():
    return refused(doc_a(export=export(a_rows("e1s42", e1s42={"stage": "E2"}))),
                   "artifact_row_mismatch", "E2")


@case("V13a")
def v13a():
    root = S.get("v13a_root") or variant("norec", S["a"], "e1s42", A_MODELS["e1s42"],
                                         skip_records=("fog_s2",))
    S["v13a_root"] = root
    return refused(outcome(RR.build_report, export(a_rows("e1s42")), root, split="val",
                           policy="smoke", expected_cache=CACHE), "record_mismatch",
                   "no cell record")


@case("V14")
def v14():
    root = S.get("v14_root")
    if root is None:
        spec = A_MODELS["e1s42"]
        root = S["v14_root"] = variant("split", S["a"], "e1s42", spec, skip_dirs=("clean",))
        write_art(root, dir_name("val", "e1s42", "clean"), stage="E1", sha=spec["sha"],
                  label="clean", ids=A_IDS, totals=totals_for(spec, "clean"), split="test",
                  rt=runtime(1))
    return refused(outcome(RR.build_report, export(a_rows("e1s42")), root, split="val",
                           policy="smoke", expected_cache=CACHE), "artifact_split_mismatch",
                   "'test'")


def record_variant(tag: str, mutate=None, raw: bytes | None = None) -> Path:
    root = variant(tag, S["a"], "e1s42", A_MODELS["e1s42"], skip_records=("fog_s2",))
    d = root / dir_name("val", "e1s42", "fog_s2")
    if raw is not None:
        CO.record_path(d).write_bytes(raw)
    else:
        write_record(d, mutate=mutate)
    return root


def v13_run(key: str, tag: str, needles, **kw):
    root = S.get(key) or record_variant(tag, **kw)
    S[key] = root
    return refused(outcome(RR.build_report, export(a_rows("e1s42")), root, split="val",
                           policy="smoke", expected_cache=CACHE), "record_mismatch", *needles)


@case("V13b")
def v13b():
    return v13_run("v13b_root", "recrun", ("run_id",),
                   mutate=lambda r: r.update(run_id="run_another"))


@case("V13c")
def v13c():
    return v13_run("v13c_root", "recbad", ("[record_unreadable]",), raw=b"{not json")


@case("V13d")
def v13d():
    return v13_run("v13d_root", "recschema", ("[record_schema]",),
                   mutate=lambda r: r.update(extra_key=1))


@case("V13e")
def v13e():
    return v13_run("v13e_root", "recfiles", ("[record_artifact_mismatch]",),
                   mutate=lambda r: r["artifact_sha256"].update({"summary.json": "0" * 64}))


# ---------------------------------------------------------------- RC: corrupted.py's record codes
def rc_record() -> tuple[Path, dict]:
    d = S["a"] / dir_name("val", "e1s42", "fog_s2")
    return d, CO.read_cell_record(CO.record_path(d))


@case("RC1")
def rc1():
    p = S["work"] / "rc" / f"bad_{next(_SEQ)}.corruption.json"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"{not json")
    return refused(outcome(CO.read_cell_record, p), "record_unreadable")


@case("RC2")
def rc2():
    _, rec = rc_record()
    p1 = S["work"] / "rc" / f"schema_{next(_SEQ)}.corruption.json"
    p2 = S["work"] / "rc" / f"keys_{next(_SEQ)}.corruption.json"
    p1.parent.mkdir(exist_ok=True)
    p1.write_bytes(CO.record_bytes({**rec, "schema": "plantseg-corruption-eval-record/0.9.0"}))
    p2.write_bytes(CO.record_bytes({**rec, "extra_key": 1}))
    a = refused(outcome(CO.read_cell_record, p1), "record_schema")
    b = refused(outcome(CO.read_cell_record, p2), "record_schema")
    return a[0] and b[0], f"{a[1][:80]}; {b[1][:80]}"


@case("RC3")
def rc3():
    d, rec = rc_record()
    files = {**rec, "artifact_sha256": {**rec["artifact_sha256"], "summary.json": "0" * 64}}
    other = {**rec, "artifact_dir": "val_corr_e1s42_fog_s3"}
    a = refused(outcome(CO.verify_cell_record, d, files), "record_artifact_mismatch",
                "no longer match")
    b = refused(outcome(CO.verify_cell_record, d, other), "record_artifact_mismatch",
                "val_corr_e1s42_fog_s3")
    good = outcome(CO.verify_cell_record, d, rec)[0] == "ok"
    return a[0] and b[0] and good, f"{a[1][:70]}; {b[1][:70]}; the true record verifies {good}"


# ---------------------------------------------------------------- O: outputs
def written_a() -> tuple[Path, dict]:
    if "out_a" not in S:
        code, doc = doc_a()
        if code != "ok":
            raise AssertionError(f"build_report: {code}: {doc}")
        S["out_a"] = (RR.write_outputs(fresh("robustness_a"), doc), doc)
    return S["out_a"]


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_keys(v)


@case("O1")
def o1():
    code, doc = doc_a()
    if code != "ok":
        return False, f"build_report: {code}: {doc}"
    raw = RR.json_bytes(doc)
    parsed = _strict_json(raw.decode("utf-8"))
    canonical = (json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
                 + "\n").encode("utf-8") == raw
    text = raw.decode("utf-8")
    ok = (canonical and parsed["class_space"] == SR.CLASS_SPACE and parsed["schema"] == RR.SCHEMA
          and "dataset_miou_c_delta" not in text and parsed["status"] == "nonofficial")
    return ok, f"canonical {canonical}; class space {parsed['class_space'][:40]}"


@case("O2")
def o2():
    out, doc = written_a()
    md = (out / RR.MD_NAME).read_text(encoding="utf-8")
    banner = f"> **{doc['banner']}**" in md and doc["banner"].startswith("NONOFFICIAL")
    listed = all(f"> - {name}" in md for name in doc["nonofficial_inputs"]) and \
        len(doc["nonofficial_inputs"]) == 25 * 3 + 16
    pval = re.search(r"\bp[-_ ]?values?\b|\bp\s*[<=>]", md, re.IGNORECASE)
    keys = [k for k in walk_keys(json.loads((out / RR.JSON_NAME).read_text(encoding="utf-8")))
            if "p_value" in k or k in ("p", "pvalue")]
    return banner and listed and pval is None and not keys, \
        f"banner {banner}; inputs listed {listed}; p-value text {pval}; keys {keys[:3]}"


@case("O3")
def o3():
    out, doc = written_a()
    names = sorted(os.listdir(out))
    sums = (out / RR.SUMS_NAME).read_text(encoding="utf-8").splitlines()
    verified = sorted(line.split("  ", 1)[1] for line in sums
                      if sha256_file(out / line.split("  ", 1)[1]) == line.split("  ", 1)[0])
    leftovers = [n for n in os.listdir(out.parent) if ".tmp-" in n]
    again = outcome(RR.write_outputs, out, doc)
    return (names == sorted([RR.JSON_NAME, RR.MD_NAME, RR.SUMS_NAME])
            and verified == sorted([RR.JSON_NAME, RR.MD_NAME]) and not leftovers
            and again[0] == "out_dir_exists"), \
        f"files {names}; verified {verified}; temp leftovers {leftovers}; rewrite {again[0]}"


@case("O4")
def o4():
    inv = write_inventory(f"cli_a_{next(_SEQ)}", S["export_a"])
    out = fresh("cli_a")
    code, rc, stdout, stderr = cli(RP.main, ["--inventory", inv, "--artifacts-root", S["a"],
                                             "--policy", "smoke", "--split", "val", "--out", out,
                                             "--expected-cache-sha256", CACHE])
    doc = json.loads((out / RR.JSON_NAME).read_text(encoding="utf-8")) if out.exists() else {}
    bad_out = fresh("cli_refused")
    code2, rc2, _, stderr2 = cli(RP.main, ["--inventory", inv, "--artifacts-root", S["a"],
                                           "--policy", "official", "--split", "val", "--out",
                                           bad_out])
    ok = (code == "ok" and rc == 0 and doc.get("inventory", {}).get("sha256") == sha256_file(inv)
          and code2 == "ok" and rc2 == 2 and "[policy_split_mismatch]" in stderr2
          and not bad_out.exists())
    return ok, f"rc {rc}; refused rc {rc2}: {stderr2.strip()[:90]}"


@case("O5")
def o5():
    inv = write_inventory(f"cli_o_{next(_SEQ)}", S["export_o"])
    out = fresh("cli_o")
    code, rc, stdout, stderr = cli(RP.main, ["--inventory", inv, "--artifacts-root", S["o"],
                                             "--policy", "official", "--split", "test", "--out",
                                             out, "--expected-cache-sha256", CACHE])
    doc = json.loads((out / RR.JSON_NAME).read_text(encoding="utf-8")) if out.exists() else {}
    md = (out / RR.MD_NAME).read_text(encoding="utf-8") if out.exists() else ""
    ok = (code == "ok" and rc == 0 and doc.get("status") == "official"
          and doc.get("banner") is None and "NONOFFICIAL" not in md)
    return ok, f"rc {rc}; status {doc.get('status')}; {stderr.strip()[:90]}"


@case("O6")
def o6():
    src = (REPO / "src" / "eval" / "robustness_report.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]
    banned = [t for t in ("run_comparison", "wilcoxon", "holm", "bootstrap_", "dataset_miou_c_delta",
                          "copyfile", "copytree") if t in code]
    return not banned, f"inferential or copying tokens in the code: {banned}"


# ---------------------------------------------------------------- H: the rehearsal script
def rehearse(root, e1="e1v", stand_in="e1s44v", out=None):
    out = out or fresh("rehearsal")
    code, rc, stdout, stderr = cli(RH.main, ["--artifacts-root", root, "--e1", e1, "--stand-in",
                                             stand_in, "--out", out])
    doc = (json.loads((out / RH.OUT_NAME).read_text(encoding="utf-8"))
           if out.exists() and (out / RH.OUT_NAME).exists() else None)
    return code, rc, stdout, stderr, out, doc


def h_flat_mean(file_id: str) -> float:
    keep = [i for i, iid in enumerate(H_IDS) if iid != H_AM5]
    stack = np.stack([S["h_dis"][(file_id, c)][keep] for c in INF_CELLS])
    return float(np.mean(stack.mean(axis=0)))


@case("H1")
def h1():
    code, rc, stdout, stderr, out, doc = rehearse(S["h"])
    if code != "ok" or rc != 0 or doc is None:
        return False, f"rc {rc} {code}: {stderr.strip()[:160]}"
    g5, cmp_ = doc["r_g5"], doc["comparison"]
    sums = (out / RH.SUMS_NAME).read_text(encoding="utf-8").split("  ")
    ok = (g5["holds"] and all(g5[s]["n"] == 845 and g5[s]["k_val"] == 1 for s in ("E1", "E6"))
          and all(g5[s]["max_abs_difference"] <= 1e-12 for s in ("E1", "E6"))
          and abs(doc["miou_c_per_image_mean"]["E1"] - h_flat_mean("e1v")) <= 1e-12
          and abs(doc["miou_c_per_image_mean"]["E6"] - h_flat_mean("e1s44v")) <= 1e-12
          and cmp_["comparison_id"] == "robustness_e1_e6" and cmp_["baseline_stage"] == "E1"
          and cmp_["candidate_stage"] == "E6" and cmp_["n_paired"] == 845
          and sums[0] == sha256_file(out / RH.OUT_NAME)
          and sorted(os.listdir(out)) == sorted([RH.OUT_NAME, RH.SUMS_NAME])
          and doc["status"] == "nonofficial")
    S["h1_doc"] = doc
    return ok, f"R-G5 E1 n {g5['E1']['n']} max {g5['E1']['max_abs_difference']!r}; n_paired " \
               f"{cmp_['n_paired']}"


def h_bad_root(tag: str, **kw) -> Path:
    """E1's clean VAL artifact and a first cell that the REHEARSAL reader must refuse."""
    root = S["work"] / f"h_{tag}"
    root.mkdir()
    shutil.copytree(S["h"] / dir_name("val", "e1v", "clean"), root / dir_name("val", "e1v", "clean"))
    spec = H_MODELS["e1v"]
    write_art(root, dir_name("val", "e1v", INF_CELLS[0]), stage="E1", sha=spec["sha"],
              label=INF_CELLS[0], ids=A_IDS, totals=totals_for(spec, INF_CELLS[0]),
              rt=runtime(1), **kw)
    return root


@case("H2")
def h2():
    root = S.get("h2_root") or h_bad_root("split", split="test", status="provisional")
    S["h2_root"] = root
    code, rc, _, stderr, out, _ = rehearse(root)
    ok = rc == 2 and "[rehearsal_input_refused]" in stderr and "VAL artifacts only" in stderr
    return ok and not out.exists(), f"rc {rc}: {stderr.strip()[:140]}"


@case("H3")
def h3():
    root = S.get("h3_root") or h_bad_root("status", status="smoke")
    S["h3_root"] = root
    code, rc, _, stderr, out, _ = rehearse(root)
    ok = rc == 2 and "[rehearsal_input_refused]" in stderr and "real-run artifact" in stderr
    return ok and not out.exists(), f"rc {rc}: {stderr.strip()[:140]}"


@case("H4")
def h4():
    code, rc, stdout, stderr, out, doc = rehearse(S["h"])
    if doc is None:
        return False, f"rc {rc} {code}: {stderr.strip()[:160]}"
    si = doc["stand_in"]["E6"]
    ok = (si["file_id"] == "e1s44v" and si["scored_as"] == "E1" and si["relabelled_to"] == "E6"
          and doc["inputs"]["E6:clean"]["dir"] == "val_clean_e1s44v"
          and doc["inputs"]["E1:clean"]["dir"] == "val_clean_e1v"
          and doc["inputs"]["E6:clean"]["checkpoint_sha256"] == SHA["e1s44v"]
          and len(doc["inputs"]) == 32 and "stand-in" in doc["banner"])
    return ok, f"stand-in {si}"


@case("H5")
def h5():
    root = never("h_same")
    code, rc, _, stderr, out, _ = rehearse(root, e1="e1v", stand_in="e1v")
    first = rc == 2 and "[stand_in_same_as_e1]" in stderr and not root.exists()
    twin = S.get("h5_root")
    if twin is None:
        twin = S["h5_root"] = S["work"] / "h_twin"
        twin.mkdir()
        for label in ("clean", *INF_CELLS):
            shutil.copytree(S["h"] / dir_name("val", "e1v", label), twin / dir_name("val", "e1v", label))
            shutil.copytree(S["h"] / dir_name("val", "e1v", label),
                            twin / dir_name("val", "e1vtwin", label))
    code2, rc2, _, stderr2, out2, _ = rehearse(twin, e1="e1v", stand_in="e1vtwin")
    second = rc2 == 2 and "[stand_in_same_as_e1]" in stderr2 and "one checkpoint" in stderr2
    return first and second, f"same id rc {rc}; same checkpoint rc {rc2}: {stderr2.strip()[:80]}"


@case("H6")
def h6():
    target = REPO / f"s5_never_created_rehearsal_{next(_SEQ)}"
    code, rc, _, stderr, out, _ = rehearse(never("root"), out=target)
    return rc == 2 and "[out_dir_inside_repo]" in stderr and not target.exists(), \
        f"rc {rc}: {stderr.strip()[:100]}"


LABELS = {
    "Q1": "Q1 per cell: the all-class pooled values and the disease-only column equal their closed "
          "forms for every cell of E1 and E6 (seed 42), the teacher and E1 seed 43",
    "Q2": "Q2 per severity: both LE6 series equal their closed forms for the teacher and the seed-42 "
          "students only; RR.SERIES is LE6's pair",
    "Q3": "Q3 mIoU-C equals its closed form for every model holding the 15 inferential cells, seed 43 "
          "included (LE4), and src/stats/report.dataset_miou_c bitwise for E1 and E6 (P28)",
    "Q3b": "Q3 under REHEARSAL on 846-row VAL artifacts: report.dataset_miou_c bitwise and rCD(E1) = 1 "
           "exactly; seed 44 carries no rCD",
    "Q4": "Q4 RPD from the pooled clean value and mIoU-C, teacher included",
    "Q5": "Q5 rCD per corruption: E1 = 1 exactly, E6 its closed form; the seed-42 students only",
    "Q6": "Q6 the rCD mean over the five corruptions",
    "Q7": "Q7 per corruption: the mean over severities 1-3 for the teacher and the seed-42 students; "
          "rCD for the students only",
    "Q8": "Q8 Kamann-Rother for both series: nested (LE8) and, within a series, equal to the flat mean",
    "Q9a": "Q9 the teacher is in Q3, Q4 and Q8 and absent from Q5 and Q6",
    "Q9b": "Q9 provenance: each input's four-file sha256, record sha256, runtime tuple and commit, one "
           "cache checksum, the inventory and the code provenance",
    "V1a": "V1 a missing cell in an official set is refused (artifact_missing)",
    "V1b": "V1 a missing cell in a smoke set is reported; the quantities that need it are null",
    "V2": "V2 a duplicate: a cell directory holding another cell's artifact is refused "
          "(condition_mismatch); a cell listed twice is refused by the inventory reader",
    "V3": "V3 split digests differ across the inputs (split_digest_mismatch)",
    "V4": "V4 a non-canvas input is refused (non_canvas_input)",
    "V5": "V5 a checkpoint sha differing within a model is refused (checkpoint_identity_mismatch)",
    "V6a": "V6 the cache checksum differs across the cell records (cache_checksum_mismatch)",
    "V6b": "V6 the records' cache differs from --expected-cache-sha256 (cache_checksum_mismatch)",
    "V6c": "V6 an official set requires --expected-cache-sha256 (expected_cache_sha256_required)",
    "V6d": "V6 a malformed --expected-cache-sha256 is refused (expected_cache_sha256_format)",
    "V7": "V7 (LE7) E1's degradation <= 0 for brightness: that rCD is null with reason "
          "rcd_reference_nonpositive for every stage, the others defined, the mean null",
    "V8a": "V8 an out-dir inside the repository is refused and not created (out_dir_inside_repo)",
    "V8b": "V8 an existing out-dir is refused (out_dir_exists)",
    "V8c": "V8 an out-dir holding 'test' in any case is refused before any filesystem call "
           "(out_dir_name_invalid)",
    "V9a": "V9 (LE1) one official artifact whose runtime tuple differs from the export's context is "
           "refused (heterogeneous_runtime)",
    "V9b": "V9 (LE1) official artifacts at a commit other than the export's evaluator_commit are "
           "refused (heterogeneous_runtime)",
    "V9c": "V9 (LE2) identity: INT8 is compared with converted_artifact_sha256, fp32 with "
           "source.sha256 (checkpoint_identity_mismatch otherwise)",
    "V9d": "V9 (LE1) a non-official set with differing runtime tuples is accepted and the values "
           "recorded",
    "V9e": "V9 (LE1) an official set at the export's context and commit is accepted: status official, "
           "no banner, rCD(E1) = 1; RUNTIME_KEYS equal the runner's CONTEXT_KEYS",
    "V10a": "V10 an unknown policy is refused (policy_unknown)",
    "V10b": "V10 a policy read at another split is refused before any input is read "
            "(policy_split_mismatch)",
    "V11": "V11 an input the policy's reader refuses is refused (input_refused)",
    "V12": "V12 an artifact whose stage differs from its export row is refused "
           "(artifact_row_mismatch)",
    "V13a": "V13 a corrupted artifact without its cell record is refused (record_mismatch)",
    "V13b": "V13 a record naming another run is refused (record_mismatch)",
    "V13c": "V13 an unreadable record is refused, naming record_unreadable (record_mismatch)",
    "V13d": "V13 a record with other keys is refused, naming record_schema (record_mismatch)",
    "V13e": "V13 a record whose file hashes differ is refused, naming record_artifact_mismatch "
            "(record_mismatch)",
    "V14": "V14 an artifact of another split than the policy's is refused (artifact_split_mismatch)",
    "RC1": "RC read_cell_record refuses a record that is not strict JSON (record_unreadable)",
    "RC2": "RC read_cell_record refuses another schema or other keys (record_schema)",
    "RC3": "RC verify_cell_record refuses changed files or another directory "
           "(record_artifact_mismatch); the true record verifies",
    "O1": "O1 strict canonical JSON stating the class space, never the rejected alias (LE13)",
    "O2": "O2 the md carries the banner and lists every non-official input; no significance value "
          "anywhere",
    "O3": "O3 SHA256SUMS verifies; exactly three files; no temporary leftover; a second write is "
          "refused (out_dir_exists)",
    "O4": "O4 the CLI writes the smoke outputs with the inventory's sha256, and refuses a policy at "
          "another split with exit 2 and nothing written",
    "O5": "O5 the CLI writes official outputs from an official set (scopes checked): status official, "
          "no banner",
    "O6": "O6 robustness_report.py holds no inferential call, no alias and no copying",
    "H1": "H1 the frozen path runs on 846-row REHEARSAL artifacts: R-G5 (n = 846 - K_val, nested = "
          "flat within 1e-12), the comparison serialised, SHA256SUMS",
    "H2": "H2 a TEST-split input is refused (rehearsal_input_refused)",
    "H3": "H3 a smoke-status input is refused (rehearsal_input_refused)",
    "H4": "H4 the stand-in mapping is recorded (E1 seed 44 relabelled to E6)",
    "H5": "H5 one model on both sides is refused, by file_id or by checkpoint (stand_in_same_as_e1)",
    "H6": "H6 a rehearsal out-dir inside the repository is refused",
}


def mutations() -> None:
    mutant("M1 severity 4 in mIoU-C", ["Q3"], (RR, "MIOU_C_SEVERITIES", (1, 2, 3, 4)))
    mutant("M2 teacher in rCD", ["Q9a"],
           (RR, "in_rcd", lambda row: row.get("seed") == 42 or row["stage"] == "teacher"))
    mutant("M3 flat K&R over all the model's cells", ["Q8"],
           (RR, "kamann_rother", lambda m, cs, ss: RR._mean(list(m.values()))))
    mutant("M4 RPD from a per-image mean", ["Q4"],
           (RR, "clean_value",
            lambda run_: float(np.mean([r.all_class_miou for r in run_.records]))))
    mutant("M5 gaussian_noise in the s1-5 series", ["Q2"],
           (RR, "SERIES", ((RR.SERIES[0][0], CORR, (1, 2, 3, 4, 5)), RR.SERIES[1])))
    mutant("M6 E1's reference guard removed (LE7)", ["V7"],
           (RR, "rcd", lambda stage, a, b: float(a / b)))
    mutant("M7 rCD mean over the defined corruptions (LE7)", ["V7"],
           (RR, "rcd_mean", lambda per: {"value": RR._mean([v["value"] for v in per.values()
                                                            if v["value"] is not None]),
                                         "reason": None}))
    mutant("M8a seeds 43/44 in rCD (LE4)", ["Q5"],
           (RR, "in_rcd", lambda row: row["stage"] != "teacher"))
    mutant("M8b seeds 43/44 in the series (LE4)", ["Q2"], (RR, "in_series", lambda row: True))
    mutant("M9 INT8 identity from source.sha256 (LE2)", ["V9c"],
           (RR, "expected_identity", lambda row: row["source"]["sha256"]))
    mutant("M10 class space not stated (LE13)", ["O1"], (RR, "class_space", lambda: "disease-only"))

    def flat14(cells):
        per = {}
        for cid in INF_CELLS[:14]:
            for rec in cells[cid].records:
                per.setdefault(rec.clean_image_id, []).append(float(rec.disease_only_miou))
        return {k: float(np.mean(v)) for k, v in per.items()}
    mutant("MH1 R-G5's flat mean over 14 cells", ["H1"], (RH, "flat_means", flat14))
    mutant("MH4 the stand-in not relabelled", ["H1", "H4"], (RH, "as_stand_in", lambda run_: run_))
    for code, targets in (
            ("artifact_split_mismatch", ["V14"]),
            ("condition_mismatch", ["V2"]), ("non_canvas_input", ["V4"]),
            ("artifact_row_mismatch", ["V12"]), ("checkpoint_identity_mismatch", ["V5", "V9c"]),
            ("input_refused", ["V11"]), ("record_mismatch", ["V13a", "V13b"]),
            ("artifact_missing", ["V1a"]), ("split_digest_mismatch", ["V3"]),
            ("expected_cache_sha256_required", ["V6c"]),
            ("expected_cache_sha256_format", ["V6d"]),
            ("cache_checksum_mismatch", ["V6a", "V6b"]),
            ("heterogeneous_runtime", ["V9a", "V9b"]), ("policy_unknown", ["V10a"]),
            ("policy_split_mismatch", ["V10b"]), ("out_dir_name_invalid", ["V8c"]),
            ("out_dir_inside_repo", ["V8a"]), ("out_dir_exists", ["V8b"])):
        mutant(f"M-{code}", targets, skip(RR, "_refuse", code))
    for code, targets in (("stand_in_same_as_e1", ["H5"]),
                          ("rehearsal_input_refused", ["H2", "H3"])):
        mutant(f"M-{code}", targets, skip(RH, "_refuse", code))
    for code, targets in (("record_unreadable", ["RC1", "V13c"]), ("record_schema", ["RC2", "V13d"]),
                          ("record_artifact_mismatch", ["RC3", "V13e"])):
        mutant(f"M-{code}", targets, skip(CO, "_fail", code))


def make_work() -> Path:
    for _ in range(20):
        work = Path(tempfile.mkdtemp(prefix="robustness_report_"))
        if "test" not in str(work).lower():
            return work
        os.rmdir(work)
    raise AssertionError("no temporary folder free of 'test'")


def main() -> int:
    print("=" * 100)
    print("S5 DESCRIPTIVE ROBUSTNESS OUTPUTS SMOKE — synthetic; no PlantSeg data; no GPU")
    print("=" * 100)
    work = make_work()
    try:
        setup(work)
        S["model_a"] = {fid: RR.load_model(S["a"], "val", row_for(fid, A_MODELS[fid]), "smoke")
                        for fid in ("e1s42", "e6s42")}
        missing = sorted(set(CASES) - set(LABELS))
        if missing:
            raise AssertionError(f"cases without labels: {missing}")
        for name in CASES:
            ok, detail = run(name)
            results.append((LABELS[name], ok, detail))
        mutations()
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        results.append(("FATAL", False, traceback.format_exc(limit=3)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print()
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail and not ok:
            print(f"         {detail[:240]}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
