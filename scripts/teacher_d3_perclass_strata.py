#!/usr/bin/env python3
"""D3: the teacher's per-class TRAIN and VAL IoU, grouped by the AM-17 item 8 strata (AM-18 item 1(c);
lane L-TEACHER-DIAG).

No model is loaded. Inputs:
  --train-scores   teacher_train_scores_<UTC>.json (scripts/score_teacher_train.py): the teacher of record's
                   per-class TRAIN table under the R3 procedure; a smoke output is refused (P2)
  --perclass-val   S3's perclass_gap_val_<UTC>.json (scripts/perclass_gap_table.py, lane L-AM17-PERCLASS):
                   its iou_teacher column is the VAL part
  --strata         the strata file both name (reports/strata/train_strata_v1.json, the registered build)
Refused unless both tables name the teacher of record, S3's table names this strata file's sha256, and the
TRAIN scores were computed on the strata file's split list (both hashes). The TRAIN table must reproduce its
dataset-level all-class and disease-only mIoU bitwise from its own supports (src.stats.perclass's
model_integrity, the evaluator's float32 arithmetic); otherwise STOP.

Per disease class (1..115): stratum, rare flag, TRAIN and VAL ground-truth support, IoU on each split and
TRAIN - VAL. A class with no ground truth on a split is not evaluable there (class 69 on VAL) and enters no
summary of that split nor of the difference. D3 writes its own group summaries (P20): per stratum T1, T2,
T3, for the rare classes and overall, the classes evaluable on each split, the math.fsum mean and the median
IoU on TRAIN and on VAL, and of TRAIN - VAL over the classes evaluable on both. The background (class 0) is
one separate line and enters no summary.

Outputs (outside the repository): teacher_d3_perclass_<UTC>.json and .csv, serialized first and created
exclusively. Every run is a run of record: --script-commit and --script-commit-dl-id are required and
--generated-utc is refused (C2). Exit codes: 0 written, 1 STOP, 2 refusal or usage, 4 unexpected exception.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval import teacher_diag as td  # noqa: E402

SCRIPT = "scripts/teacher_d3_perclass_strata.py"
TRAIN_SCRIPT = "scripts/score_teacher_train.py"
TRAIN_SCHEMA = "plantseg-teacher-train-scores/1.0.0"
PERCLASS_LANE = "L-AM17-PERCLASS"
SCHEMA = "plantseg-teacher-d3/1.0.0"
KIND = "teacher_d3_perclass"
NUM_CLASSES, BACKGROUND = 116, 0
DISEASES = tuple(range(1, NUM_CLASSES))
GROUPS = ("T1", "T2", "T3", "rare", "overall")
COLUMNS = ("class_id", "stratum", "rare", "train_share", "train_rank", "train_gt_support", "iou_train",
           "train_status", "val_gt_support", "iou_val", "val_status", "train_minus_val")
OK, NOT_EVALUABLE = "ok", "not_evaluable"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D3: teacher per-class TRAIN and VAL IoU by stratum (AM-18 item 1(c)).")
    td.add_common_flags(p)
    p.add_argument("--train-scores", required=True, help="teacher_train_scores_<UTC>.json")
    p.add_argument("--perclass-val", required=True, help="S3's perclass_gap_val_<UTC>.json")
    p.add_argument("--strata", required=True, help="the AM-17 item 8 strata file")
    return p


def read_inputs(args) -> tuple[dict, dict, dict, dict]:
    """(train scores, S3's table, strata document, the inputs block). Refuses anything not of record."""
    from src.stats.perclass import PerclassError, load_strata
    train, train_sha = td.read_diag_output(args.train_scores, script=TRAIN_SCRIPT, stub=False, what="--train-scores")
    t = train.get("teacher") or {}
    if train.get("schema") != TRAIN_SCHEMA or (train.get("pass") or {}).get("split") != "train":
        raise td.Refused("--train-scores is not a TRAIN scores output")
    if (train.get("pass") or {}).get("rows") != td.TRAIN_ROWS:
        raise td.Refused(f"--train-scores scored {(train.get('pass') or {}).get('rows')} TRAIN images, not {td.TRAIN_ROWS}")
    if t.get("role") != "record" or (t.get("checkpoint") or {}).get("sha256_verified") != td.RECORD_SHA256:
        raise td.Refused("--train-scores is not the teacher of record's")

    p = Path(args.perclass_val)
    if not p.is_file():
        raise td.Refused(f"--perclass-val not found: {p}")
    val_sha = td.file_sha256(p)
    val = td.read_json_strict(p)
    vt = (val.get("inputs") or {}).get("teacher") or {}
    if val.get("lane") != PERCLASS_LANE or val.get("artifact_status") != "provisional" or val.get("split") != "val":
        raise td.Refused(f"--perclass-val is not a provisional VAL table of {PERCLASS_LANE}")
    if vt.get("checkpoint_sha256") != td.RECORD_SHA256 or vt.get("stage") != "teacher":
        raise td.Refused("--perclass-val's teacher column is not the teacher of record's")

    sp = Path(args.strata)
    if not sp.is_file():
        raise td.Refused(f"--strata not found: {sp}")
    try:
        strata_doc, strata_sha = load_strata(sp)
    except (PerclassError, ValueError, OSError) as e:
        raise td.Refused(f"--strata cannot be read: {type(e).__name__}: {e}") from e
    if strata_doc.get("artifact_status") != "provisional":
        raise td.Refused(f"--strata has artifact_status {strata_doc.get('artifact_status')!r}; the registered "
                         "build is provisional")
    if ((val.get("inputs") or {}).get("strata") or {}).get("sha256") != strata_sha:
        raise td.Refused("--perclass-val was built on another strata file (sha256)")
    ts = (train.get("inputs") or {}).get("strata") or {}
    if ts.get("sha256") != strata_sha or ts.get("split_list_sha256") != strata_doc.get("split_list_sha256"):
        raise td.Refused("--train-scores was computed on another strata file or TRAIN split list")
    block = {"train_scores": {"path": str(args.train_scores), "sha256": train_sha,
                              "artifact_files_sha256": (train.get("artifact") or {}).get("files_sha256"),
                              "teacher_checkpoint_sha256": td.RECORD_SHA256,
                              "loaded_state_sha256": t.get("loaded_state_sha256")},
             "perclass_val": {"path": str(p), "sha256": val_sha, "csv_sha256": val.get("csv_sha256"),
                              "teacher_checkpoint_sha256": vt.get("checkpoint_sha256")},
             "strata": {"path": str(sp), "sha256": strata_sha, "split_list_sha256": strata_doc.get("split_list_sha256")}}
    return train, val, strata_doc, block


def train_table(train: dict) -> dict:
    pc = train.get("per_class") or {}
    for k in ("class_ids", "gt_support", "pred_support", "intersection", "union", "iou", "iou_status"):
        if len(pc.get(k) or []) != NUM_CLASSES:
            raise td.Refused(f"--train-scores per_class.{k} does not list {NUM_CLASSES} classes")
    if pc["class_ids"] != list(range(NUM_CLASSES)):
        raise td.Refused("--train-scores per_class must list classes 0..115 in order")
    return pc


def val_rows(val: dict, strata: dict) -> tuple[dict, dict]:
    rows = val.get("rows") or []
    by_class = {r.get("class_id"): r for r in rows}
    if sorted(by_class) != list(DISEASES) or len(rows) != len(DISEASES):
        raise td.Refused("--perclass-val rows are not one per disease class 1..115")
    for c, r in by_class.items():
        if (r.get("stratum"), r.get("rare")) != (strata[c]["tercile"], strata[c]["rare"]):
            raise td.Refused(f"--perclass-val class {c}: stratum/rare differ from the strata file")
        if r.get("status") not in (OK, NOT_EVALUABLE):
            raise td.Refused(f"--perclass-val class {c}: status {r.get('status')!r}")
    bg = val.get("background") or {}
    if bg.get("class_id") != BACKGROUND:
        raise td.Refused("--perclass-val has no background line")
    return by_class, bg


def build_rows(pc: dict, vrows: dict, vbg: dict, strata: dict) -> tuple[list[dict], dict]:
    def one(c, stratum, rare, share, rank, vrow):
        tgt = int(pc["gt_support"][c])
        t_ok = tgt > 0
        iou_t = pc["iou"][c]
        if t_ok and iou_t is None:
            raise td.Stop(f"TRAIN class {c} has ground truth but an undefined IoU")
        v_ok = vrow.get("status") == OK if c != BACKGROUND else int(vrow.get("gt_support") or 0) > 0
        iou_v = vrow.get("iou_teacher")
        if v_ok and iou_v is None:
            raise td.Stop(f"VAL class {c} is evaluable but its teacher IoU is undefined")
        return {"class_id": c, "stratum": stratum, "rare": rare, "train_share": share, "train_rank": rank,
                "train_gt_support": tgt, "iou_train": iou_t if t_ok else None,
                "train_status": OK if t_ok else NOT_EVALUABLE,
                "val_gt_support": int(vrow.get("gt_support") or 0), "iou_val": iou_v if v_ok else None,
                "val_status": OK if v_ok else NOT_EVALUABLE,
                "train_minus_val": (iou_t - iou_v) if (t_ok and v_ok) else None}
    rows = [one(c, strata[c]["tercile"], strata[c]["rare"], strata[c]["share"], strata[c]["rank"], vrows[c])
            for c in DISEASES]
    return rows, one(BACKGROUND, "background", None, None, None, vbg)


def _stats(values: list[float]) -> dict:
    import numpy as np
    n = len(values)
    return {"n": n, "mean": (math.fsum(values) / n) if n else None,
            "median": float(np.median(np.asarray(values, dtype=np.float64))) if n else None}


def summaries(rows: list[dict]) -> dict:
    """P20: written here, not perclass._summary."""
    members = {"T1": lambda r: r["stratum"] == "T1", "T2": lambda r: r["stratum"] == "T2",
               "T3": lambda r: r["stratum"] == "T3", "rare": lambda r: r["rare"] is True, "overall": lambda r: True}
    out = {}
    for g in GROUPS:
        sel = [r for r in rows if members[g](r)]
        tr = [r for r in sel if r["train_status"] == OK]
        va = [r for r in sel if r["val_status"] == OK]
        both = [r for r in sel if r["train_minus_val"] is not None]
        out[g] = {"n_classes": len(sel),
                  "train": {**_stats([r["iou_train"] for r in tr]), "classes": [r["class_id"] for r in tr]},
                  "val": {**_stats([r["iou_val"] for r in va]), "classes": [r["class_id"] for r in va]},
                  "train_minus_val": {**_stats([r["train_minus_val"] for r in both]),
                                      "classes": [r["class_id"] for r in both]},
                  "not_evaluable_train": [r["class_id"] for r in sel if r["train_status"] != OK],
                  "not_evaluable_val": [r["class_id"] for r in sel if r["val_status"] != OK]}
    return out


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def csv_bytes(rows: list[dict], background: dict) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in [*rows, background]:
        w.writerow([_cell(r[k]) for k in COLUMNS])
    return buf.getvalue().encode("utf-8")


def write_pair(json_path: Path, doc: dict, csv_path: Path, csv_b: bytes) -> None:
    """Both serialized first; both created exclusively; a failure leaves neither behind."""
    td.write_files_exclusive([(csv_path, csv_b), (json_path, td.json_bytes(doc))])


def run(args) -> int:
    return td.run_with_exit_codes(_run, args)


def _run(args) -> int:
    start, t0 = td.utc_now(), time.monotonic()
    td.check_common_flags(args, real=True)
    out_dir = td.require_outside_repo(args.out_dir, "--out-dir")
    code = td.require_commit_binding(args.script_commit, args.script_commit_dl_id)
    repeat = td.require_single_output(out_dir, KIND, args)
    stamp = td.output_stamp(args, start)
    json_path, csv_path = out_dir / f"{KIND}_{stamp}.json", out_dir / f"{KIND}_{stamp}.csv"

    from src.stats.perclass import PerclassError, model_integrity, strata_index
    train, val, strata_doc, inputs = read_inputs(args)
    try:
        strata = strata_index(strata_doc, DISEASES)
    except PerclassError as e:
        raise td.Refused(f"--strata: {e}") from e
    pc = train_table(train)
    vrows, vbg = val_rows(val, strata)
    try:
        integrity, problems = model_integrity({"per_class": pc, "dataset_level": train.get("dataset_level") or {}},
                                              pc["iou"], background_index=BACKGROUND)
    except PerclassError as e:
        raise td.Stop(f"TRAIN per_class: {e}") from e
    if problems:
        raise td.Stop("the TRAIN table does not reproduce its dataset-level mIoU: " + "; ".join(problems))
    rows, background = build_rows(pc, vrows, vbg, strata)
    csv_b = csv_bytes(rows, background)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = td.base_document(SCRIPT, args, stub=False, start_utc=start, code=code, extra={
        "schema": SCHEMA, "repeat": repeat, "gates": "nothing (descriptive)", "inputs": inputs,
        "direction": "train_minus_val = iou_train - iou_val (the teacher of record on both splits)",
        "not_evaluable_rule": "no ground truth on a split (gt_support == 0): that split's IoU and the "
                              "difference are null and the class enters no summary of them",
        "train_integrity": integrity,
        "rows": rows, "background": background, "summaries": summaries(rows),
        "csv": {"name": csv_path.name, "sha256": hashlib.sha256(csv_b).hexdigest(), "columns": list(COLUMNS)},
        "status": "written",
    })
    doc["environment"] = td.environment_block(start, td.utc_now())
    doc["wall_seconds"] = round(time.monotonic() - t0, 3)
    write_pair(json_path, doc, csv_path, csv_b)
    print(f"written: {json_path.name} sha256 {td.file_sha256(json_path)}; {csv_path.name} sha256 "
          f"{doc['csv']['sha256']}")
    return td.EXIT_OK


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
