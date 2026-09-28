"""Dataset-level mIoU under the union-present and GT-present eligibility rules (L-AM17-GTPRESENT).

AM-17 item 1(b), descriptive (it changes no decision): for every model on VAL and TEST, dataset-level
mIoU is also reported over the classes with ground truth in that split ("GT-present"), beside the
union-present headline of docs/EVALUATION_CONTRACT.md section 3.1 (AM-6). Both are recomputed here
from an artifact's `sufficient_stats.npz`; the evaluator is unchanged and src/eval/metrics.py is not
touched.

    TP_c, GT_c, PRED_c   per-class totals, summed over the per-image sparse triplets
    union_c              GT_c + PRED_c - TP_c
    IoU_c                TP_c / union_c
    union-present        {c : union_c > 0}    the frozen headline rule
    GT-present           {c : GT_c > 0}       the descriptive sensitivity

Every mean carries two arithmetics, kept apart on purpose:

  * `value` -- the production arithmetic of `src.eval.metrics.miou_from_confusion` (float32 counts,
    float32 IoU, torch float32 mean over the eligible classes in class order). Under the union-present
    rule it reproduces summary.json's dataset-level mIoU BITWISE, which is the proof that the npz
    reconstruction is faithful; the GT-present value uses the same arithmetic with the other
    eligibility mask, so the two are like for like.
  * `value_float64` -- the same mean over float64 per-class IoU with an exactly rounded sum
    (`math.fsum`): independent of platform and summation order, for exact relations at 1e-12 (for
    example GT-present = union-present x n_union / n_gt, which holds because a class that is
    union-present but not GT-present has TP = 0) and for float64 consumers such as
    `src.stats.noninferiority`.

Import-time behaviour is side-effect free; torch is imported only when a float32 value is computed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable

import numpy as np

UNION_PRESENT = "union_present"
GT_PRESENT = "gt_present"
RULES = (UNION_PRESENT, GT_PRESENT)

#: Per-class status vocabulary of summary.json `per_class.iou_status` (src/eval/evaluate.py).
CLASS_STATUS_OK = "ok"
CLASS_STATUS_UNDEFINED = "undefined_absent_from_gt_and_pred"


class EligibilityError(RuntimeError):
    """Malformed sufficient statistics or an invalid request. Always fatal."""


def _field(stats, name: str):
    if isinstance(stats, Mapping):                    # dict, or an open np.load(...) NpzFile
        return stats[name] if name in stats else None
    return getattr(stats, name, None)                 # src.stats.ingest.PerImageSufficientStats


def _int_vector(stats, name: str) -> np.ndarray:
    v = _field(stats, name)
    if v is None:
        raise EligibilityError(f"sufficient statistics lack {name!r}")
    a = np.asarray(v)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise EligibilityError(f"{name} must be a 1-D integer array, got {a.dtype} {a.shape}")
    return a.astype(np.int64, copy=False)


# --------------------------------------------------------------------------------------------------
# class totals
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassTotals:
    """Dataset-level per-class int64 totals, reconstructed from the per-image sparse triplets."""
    tp: np.ndarray
    gt: np.ndarray
    pred: np.ndarray

    @property
    def num_classes(self) -> int:
        return int(self.tp.shape[0])

    @property
    def union(self) -> np.ndarray:
        return self.gt + self.pred - self.tp


def class_totals(stats, num_classes: int | None = None) -> ClassTotals:
    """Sum the sparse `(class_id, tp, gt, pred)` triplets per class, exactly, in int64.

    `stats` is a `src.stats.ingest.PerImageSufficientStats`, an open `np.load` of
    sufficient_stats.npz, or a mapping with those arrays. When it carries `dataset_tp/gt/pred`, the
    reconstruction must equal them exactly (EVALUATION_CONTRACT section 5.5 invariant), and
    `num_classes` defaults to their length.
    """
    if isinstance(stats, ClassTotals):
        return stats
    cls = _int_vector(stats, "class_id")
    parts = {k: _int_vector(stats, k) for k in ("tp", "gt", "pred")}
    for k, v in parts.items():
        if v.shape != cls.shape:
            raise EligibilityError(f"{k} has {v.shape[0]} entries, class_id has {cls.shape[0]}")
        if v.size and int(v.min()) < 0:
            raise EligibilityError(f"negative {k} count")
    if cls.size and (bool(np.any(parts["tp"] > parts["gt"])) or bool(np.any(parts["tp"] > parts["pred"]))):
        raise EligibilityError("a triplet has tp > gt or tp > pred: malformed sufficient statistics")

    declared = {k: _field(stats, f"dataset_{k}") for k in ("tp", "gt", "pred")}
    if num_classes is None:
        if declared["tp"] is None:
            raise EligibilityError("num_classes is required when dataset_tp is absent")
        num_classes = int(np.asarray(declared["tp"]).shape[0])
    if num_classes < 1:
        raise EligibilityError(f"num_classes must be >= 1, got {num_classes}")
    if cls.size and (int(cls.min()) < 0 or int(cls.max()) >= num_classes):
        raise EligibilityError(f"class_id outside 0..{num_classes - 1}")

    totals = {}
    for k, v in parts.items():
        acc = np.zeros(num_classes, dtype=np.int64)
        np.add.at(acc, cls, v)
        if declared[k] is not None:
            d = np.asarray(declared[k])
            if d.shape != (num_classes,) or not np.array_equal(acc, d.astype(np.int64)):
                raise EligibilityError(
                    f"the sparse {k} triplets do not sum to dataset_{k}: the npz is not self-consistent")
        acc.flags.writeable = False
        totals[k] = acc
    return ClassTotals(tp=totals["tp"], gt=totals["gt"], pred=totals["pred"])


# --------------------------------------------------------------------------------------------------
# the dataset-level mean under one eligibility rule
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetMiou:
    rule: str
    classes: tuple[int, ...]           # the requested class subset, ascending
    eligible: tuple[int, ...]          # the classes of that subset that the rule admits, ascending
    value: float | None                # production float32 arithmetic; None when nothing is eligible
    value_float64: float | None        # exactly rounded float64 mean; None when nothing is eligible

    @property
    def n_classes(self) -> int:
        return len(self.eligible)


def _production_float32(totals: ClassTotals, selected: np.ndarray, rule: str):
    """Mirror of src/eval/metrics.py `_cm_parts` + `miou_from_confusion` on the class totals.

    Returns (value or None, eligibility mask). Only the eligibility mask differs between rules.
    """
    import torch

    tp = torch.from_numpy(np.array(totals.tp, dtype=np.int64)).float()
    gt = torch.from_numpy(np.array(totals.gt, dtype=np.int64)).float()
    pr = torch.from_numpy(np.array(totals.pred, dtype=np.int64)).float()
    un = gt + pr - tp
    present = (un > 0) if rule == UNION_PRESENT else (gt > 0)
    eligible = present & torch.from_numpy(np.array(selected, dtype=bool))
    if not bool(eligible.any()):
        return None, eligible.numpy()
    iou = tp / un.clamp_min(1e-9)
    return float(iou[eligible].mean().item()), eligible.numpy()


def dataset_miou(stats, classes: Iterable[int], rule: str) -> DatasetMiou:
    """Dataset-level mIoU over `classes` under `rule` ("union_present" or "gt_present").

    `stats` is a `ClassTotals` or anything `class_totals` accepts. A class subset with no eligible
    class gives `value = value_float64 = None` (the contract's undefined; never NaN).
    """
    if rule not in RULES:
        raise EligibilityError(f"rule must be one of {RULES}, got {rule!r}")
    totals = class_totals(stats)
    n = totals.num_classes
    subset = sorted({int(c) for c in classes})
    if not subset or subset[0] < 0 or subset[-1] >= n:
        raise EligibilityError(f"classes must be a non-empty subset of 0..{n - 1}")
    selected = np.zeros(n, dtype=bool)
    selected[subset] = True

    union = totals.union
    if bool(np.any(union < 0)):
        raise EligibilityError("negative union: malformed class totals")
    present = (union > 0) if rule == UNION_PRESENT else (totals.gt > 0)
    eligible = present & selected
    value32, mask32 = _production_float32(totals, selected, rule)
    if not np.array_equal(mask32, eligible):
        raise EligibilityError("float32 eligibility disagrees with the exact int64 eligibility")

    idx = [int(c) for c in np.flatnonzero(eligible)]
    value64 = (math.fsum(int(totals.tp[c]) / int(union[c]) for c in idx) / len(idx)) if idx else None
    return DatasetMiou(rule=rule, classes=tuple(subset), eligible=tuple(idx), value=value32,
                       value_float64=value64)


# --------------------------------------------------------------------------------------------------
# the four numbers and the per-class table
# --------------------------------------------------------------------------------------------------
def rule_variants(stats, *, background_index: int = 0) -> dict:
    """All-class (0..C-1) and disease-only (every class except the background) means under both rules.

    Returns {rule: {"all_class": DatasetMiou, "disease_only": DatasetMiou}}.
    """
    totals = class_totals(stats)
    n = totals.num_classes
    if not 0 <= background_index < n:
        raise EligibilityError(f"background_index {background_index} outside 0..{n - 1}")
    everything = range(n)
    diseases = [c for c in range(n) if c != background_index]
    return {rule: {"all_class": dataset_miou(totals, everything, rule),
                   "disease_only": dataset_miou(totals, diseases, rule)} for rule in RULES}


def per_class_table(stats) -> list[dict]:
    """One row per class: totals, float64 IoU (None when union is 0) and the two eligibility flags.

    `iou` is the correctly rounded quotient TP_c / union_c, so it equals summary.json
    `per_class.iou` exactly.
    """
    totals = class_totals(stats)
    rows = []
    for c in range(totals.num_classes):
        tp, gt, pr = int(totals.tp[c]), int(totals.gt[c]), int(totals.pred[c])
        un = gt + pr - tp
        rows.append({"class_id": c, "tp": tp, "gt": gt, "pred": pr, "union": un,
                     "iou": (tp / un) if un > 0 else None,
                     "union_present": un > 0, "gt_present": gt > 0,
                     "iou_status": CLASS_STATUS_OK if un > 0 else CLASS_STATUS_UNDEFINED})
    return rows


__all__ = [
    "UNION_PRESENT", "GT_PRESENT", "RULES", "EligibilityError", "ClassTotals", "class_totals",
    "DatasetMiou", "dataset_miou", "rule_variants", "per_class_table",
]
