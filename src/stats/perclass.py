"""Per-class VAL IoU, teacher of record minus E1 seed 42, by TRAIN stratum (lane L-AM17-PERCLASS).

AM-17 item 1(f) (DRAFT, adviser approval pending); docs/lane_specs/part1.md lane 4. A descriptive
diagnostic from existing artifacts: no run, no test, no decision, and it selects nothing. Inputs are
the two artifacts' `summary.per_class` blocks (the R3 teacher artifact and the DL-17 run1 E1 artifact)
and the AM-17 item 8 strata file (scripts/build_train_strata.py).

For every disease class (1-115): stratum (T1/T2/T3), rare flag, VAL `gt_support`, `iou_teacher`,
`iou_e1`, `delta = iou_teacher - iou_e1` and both `iou_status`. A class with no VAL ground truth
(`gt_support == 0`; class 69 on VAL) is `not_evaluable`: its delta is null and it enters no summary.
Otherwise both IoUs are defined (union >= gt > 0). Per stratum, for the rare classes and overall: the
number of evaluable classes, the mean (exactly rounded, math.fsum) and median delta, and the counts
teacher better (delta > 0), E1 better (delta < 0) and ties (delta == 0, exactly). The background
(class 0) is reported as one separate line and enters no summary.

Integrity (lane 4 d2), per model: the per-class supports carried by the table reproduce the model's
dataset-level mIoU BITWISE under the evaluator's own arithmetic (`src.stats.eligibility.dataset_miou`,
float32 as src/eval/metrics.py), all-class and disease-only, with the same eligible-class counts; the
table's IoU equals `summary.per_class.iou` and the exact quotient intersection / union. So the mean of
each model's table IoU over its union-present classes, in the evaluator's arithmetic, IS its recorded
mIoU. The float64 mean of the same IoUs is recorded beside it (it differs from the float32 value by
float32 rounding only). Both artifacts must carry identical `gt_support` (one VAL ground truth).

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .eligibility import UNION_PRESENT, ClassTotals, dataset_miou

STATUS_OK = "ok"
NOT_EVALUABLE = "not_evaluable"
TERCILES = ("T1", "T2", "T3")
GROUPS = ("T1", "T2", "T3", "rare", "overall")
STRATA_SCHEMA = "plantseg-train-strata/1.0.0"          # scripts/build_train_strata.py SCHEMA
PER_CLASS_KEYS = ("class_ids", "gt_support", "pred_support", "intersection", "union", "iou",
                  "iou_status")
ROW_COLUMNS = ("class_id", "stratum", "rare", "train_share", "train_rank", "gt_support", "iou_teacher",
               "iou_e1", "delta", "iou_status_teacher", "iou_status_e1", "status")


class PerclassError(RuntimeError):
    """Malformed or inconsistent input. Always fatal."""


# --------------------------------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------------------------------
def load_strata(path) -> tuple[dict, str]:
    """(the strata document, the sha256 of the file's bytes)."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(
        PerclassError(f"non-finite JSON constant {c!r} in the strata file")))
    return doc, hashlib.sha256(raw).hexdigest()


def strata_index(doc: dict, disease_classes) -> dict[int, dict]:
    """{class_id: {tercile, rare, share, rank}}; every disease class exactly once, nothing else."""
    if doc.get("schema") != STRATA_SCHEMA:
        raise PerclassError(f"strata schema {doc.get('schema')!r} is not {STRATA_SCHEMA!r}")
    rows = doc.get("per_class")
    if not isinstance(rows, list):
        raise PerclassError("strata file has no per_class list")
    index: dict[int, dict] = {}
    for r in rows:
        c = r.get("class_id")
        if not isinstance(c, int) or isinstance(c, bool) or c in index:
            raise PerclassError(f"strata class_id {c!r} is not an integer or appears twice")
        if r.get("tercile") not in TERCILES or not isinstance(r.get("rare"), bool):
            raise PerclassError(f"strata class {c}: tercile {r.get('tercile')!r} / rare "
                                f"{r.get('rare')!r} invalid")
        index[c] = {"tercile": r["tercile"], "rare": r["rare"], "share": r.get("share"),
                    "rank": r.get("rank")}
    want = sorted(int(c) for c in disease_classes)
    if sorted(index) != want:
        missing, extra = sorted(set(want) - set(index)), sorted(set(index) - set(want))
        raise PerclassError(f"strata classes differ from the disease classes: missing {missing[:5]}, "
                            f"extra {extra[:5]}")
    return index


# --------------------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------------------
def _per_class(pc: dict, label: str) -> dict:
    for k in PER_CLASS_KEYS:
        if k not in pc:
            raise PerclassError(f"{label}: summary.per_class lacks {k!r}")
    n = len(pc["class_ids"])
    if pc["class_ids"] != list(range(n)) or any(len(pc[k]) != n for k in PER_CLASS_KEYS):
        raise PerclassError(f"{label}: summary.per_class must list classes 0..{n - 1} in order, with "
                            "every array the same length")
    return pc


def _row(pc_t: dict, pc_e: dict, c: int) -> dict:
    gt = int(pc_t["gt_support"][c])
    it, ie = pc_t["iou"][c], pc_e["iou"][c]
    evaluable = gt > 0
    return {"class_id": c, "gt_support": gt, "iou_teacher": it, "iou_e1": ie,
            "delta": (it - ie) if evaluable else None,
            "iou_status_teacher": pc_t["iou_status"][c], "iou_status_e1": pc_e["iou_status"][c],
            "status": STATUS_OK if evaluable else NOT_EVALUABLE}


def perclass_rows(pc_teacher: dict, pc_e1: dict, strata: dict[int, dict], *,
                  background_index: int = 0) -> tuple[list[dict], dict]:
    """(one row per disease class, ascending; the background line)."""
    pc_t, pc_e = _per_class(pc_teacher, "teacher"), _per_class(pc_e1, "E1")
    n = len(pc_t["class_ids"])
    if len(pc_e["class_ids"]) != n:
        raise PerclassError(f"class counts differ: teacher {n}, E1 {len(pc_e['class_ids'])}")
    if [int(v) for v in pc_t["gt_support"]] != [int(v) for v in pc_e["gt_support"]]:
        raise PerclassError("the two artifacts' per-class gt_support differ: they do not score the same "
                            "VAL ground truth")
    diseases = [c for c in range(n) if c != background_index]
    strata_index_classes = sorted(strata)
    if strata_index_classes != diseases:
        raise PerclassError("the strata index does not cover exactly the disease classes")
    rows = []
    for c in diseases:
        r = _row(pc_t, pc_e, c)
        if r["status"] == STATUS_OK and (r["iou_teacher"] is None or r["iou_e1"] is None):
            raise PerclassError(f"class {c} has ground truth but an undefined IoU: malformed per_class")
        s = strata[c]
        rows.append({"class_id": c, "stratum": s["tercile"], "rare": s["rare"], "train_share": s["share"],
                     "train_rank": s["rank"], **{k: v for k, v in r.items() if k != "class_id"}})
    bg = _row(pc_t, pc_e, background_index)
    background = {"class_id": background_index, "stratum": "background", "rare": None, "train_share": None,
                  "train_rank": None, **{k: v for k, v in bg.items() if k != "class_id"}}
    return rows, background


def _summary(deltas: list[float], n_not_evaluable: int) -> dict:
    n = len(deltas)
    return {"n_classes": n, "n_not_evaluable": n_not_evaluable,
            "mean_delta": (math.fsum(deltas) / n) if n else None,
            "median_delta": float(np.median(np.asarray(deltas, dtype=np.float64))) if n else None,
            "teacher_better": sum(1 for d in deltas if d > 0),
            "e1_better": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0)}


def summarize(rows: list[dict]) -> dict:
    """Per stratum (T1, T2, T3), the rare classes and overall; not-evaluable classes are only counted."""
    members = {"T1": lambda r: r["stratum"] == "T1", "T2": lambda r: r["stratum"] == "T2",
               "T3": lambda r: r["stratum"] == "T3", "rare": lambda r: r["rare"] is True,
               "overall": lambda r: True}
    out = {}
    for g in GROUPS:
        sel = [r for r in rows if members[g](r)]
        out[g] = _summary([r["delta"] for r in sel if r["status"] == STATUS_OK],
                          sum(1 for r in sel if r["status"] == NOT_EVALUABLE))
    return out


# --------------------------------------------------------------------------------------------------
# integrity (lane 4 d2)
# --------------------------------------------------------------------------------------------------
def _totals(pc: dict) -> ClassTotals:
    tp = np.asarray(pc["intersection"], dtype=np.int64)
    gt = np.asarray(pc["gt_support"], dtype=np.int64)
    pred = np.asarray(pc["pred_support"], dtype=np.int64)
    if not np.array_equal(np.asarray(pc["union"], dtype=np.int64), gt + pred - tp):
        raise PerclassError("summary.per_class union != gt_support + pred_support - intersection")
    return ClassTotals(tp=tp, gt=gt, pred=pred)


def model_integrity(summary: dict, table_iou: list, *, background_index: int = 0) -> tuple[dict, list[str]]:
    """The table's IoU column against one model's summary.json (the lane 4 d2 proof)."""
    pc, dl = summary["per_class"], summary["dataset_level"]
    problems: list[str] = []
    tot = _totals(pc)
    n = tot.num_classes
    out: dict = {}
    for scope, key, classes in (("all_class", "all_class_miou", range(n)),
                                ("disease_only", "disease_only_miou",
                                 [c for c in range(n) if c != background_index])):
        mine = dataset_miou(tot, classes, UNION_PRESENT)
        want, want_n = dl.get(key), dl.get(f"{key}_n_eligible")
        same = mine.value is not None and isinstance(want, float) and mine.value == want
        out[scope] = {"summary": want, "reproduced": mine.value, "bitwise_equal": same,
                      "n_union_present": mine.n_classes, "n_eligible_summary": want_n}
        if not same or mine.n_classes != want_n:
            problems.append(f"{scope}: the table's union-present mean {mine.value!r} (n={mine.n_classes}) "
                            f"!= summary {key} {want!r} (n={want_n})")
        if scope == "all_class":
            eligible = list(mine.eligible)
            f64 = math.fsum(table_iou[c] for c in eligible) / len(eligible) if eligible else None
            out[scope]["float64_mean_of_table_iou"] = f64
            out[scope]["abs_diff_float64_vs_summary"] = (abs(f64 - want) if f64 is not None
                                                         and isinstance(want, float) else None)
    exact = [(int(tp) / int(un)) if un > 0 else None for tp, un in zip(pc["intersection"], pc["union"])]
    same_iou = list(table_iou) == list(pc["iou"]) == exact
    out["table_iou_equals_summary_and_quotient"] = same_iou
    if not same_iou:
        problems.append("the table's IoU differs from summary.per_class.iou or intersection / union")
    return out, problems


def integrity(summary_teacher: dict, summary_e1: dict, rows: list[dict], background: dict,
              strata: dict[int, dict], *, background_index: int = 0) -> tuple[dict, list[str]]:
    """The lane 4 d2 checks. Returns (record, problems); empty problems = integrity holds."""
    n = len(summary_teacher["per_class"]["class_ids"])
    col_t = [None] * n
    col_e = [None] * n
    for r in [*rows, background]:
        col_t[r["class_id"]], col_e[r["class_id"]] = r["iou_teacher"], r["iou_e1"]
    rec_t, prob_t = model_integrity(summary_teacher, col_t, background_index=background_index)
    rec_e, prob_e = model_integrity(summary_e1, col_e, background_index=background_index)
    diseases = [c for c in range(n) if c != background_index]
    one_stratum = [r["class_id"] for r in rows] == diseases and sorted(strata) == diseases \
        and all(r["stratum"] in TERCILES for r in rows)
    problems = [f"teacher {p}" for p in prob_t] + [f"E1 {p}" for p in prob_e]
    if not one_stratum:
        problems.append("rows are not one per disease class, each in exactly one stratum")
    if background.get("class_id") != background_index:
        problems.append("the background line is missing")
    record = {"teacher": rec_t, "e1": rec_e, "n_rows": len(rows), "background_line": True,
              "classes_in_exactly_one_stratum": one_stratum,
              "not_evaluable_classes": [r["class_id"] for r in rows if r["status"] == NOT_EVALUABLE]}
    return record, problems


__all__ = ["STATUS_OK", "NOT_EVALUABLE", "TERCILES", "GROUPS", "STRATA_SCHEMA", "ROW_COLUMNS",
           "PerclassError", "load_strata", "strata_index", "perclass_rows", "summarize",
           "model_integrity", "integrity"]
