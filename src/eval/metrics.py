"""Segmentation metrics. Confusion-matrix based; ignore_index=255 excluded everywhere.

Semantics are frozen by docs/EVALUATION_CONTRACT.md sections 3.1/3.2 (decisions D3/D3b in
docs/open_questions.md). THREE ELIGIBILITY RULES COEXIST BY DESIGN -- do not "harmonise" them:

  * dataset-level mIoU   -> UNION-present : eligible iff UN_c = GT_c + PR_c - TP_c > 0.
                            A class with GT_c = 0 but PR_c > 0 scores IoU 0 and IS COUNTED;
                            a class absent from GT and prediction alike is omitted.
                            Matches the official PlantSeg evaluator config
                            (IoUMetric, iou_metrics=['mIoU'], nan_to_num unset -> np.nanmean).
  * dataset-level mAcc   -> GT-present    : eligible iff GT_c > 0. Prediction-only classes are
                            omitted. The asymmetry vs mIoU is genuine benchmark behaviour.
  * dataset-level Dice   -> UNION-present : project-defined DESCRIPTIVE metric. The official
                            benchmark never computes mDice; this is a thesis-internal extension
                            that reuses the mIoU eligibility rule for internal consistency only.
  * per-image mIoU       -> GT-present    : preregistered by ch3 section (f) "per-image
                            absent-class exclusion". `_miou_from_cm` / `per_image_miou` are the
                            per-image path and are DELIBERATELY left on the GT-present rule.

Class space: num_classes = 116, background = 0, diseases = 1..115, ignore = 255 (pad only;
absent from released masks). reduce_zero_label is NOT used. Reduction is float32 (see the A1a
report case C4); a float64 reduction is a separate follow-up, deliberately not done here.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = 255


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                     ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """[num_classes, num_classes] confusion matrix (rows=GT, cols=pred); ignore_index pixels dropped.

    Labels are validated first (EVALUATION_CONTRACT section 5.3 `invalid_pred_labels` must be 0):
    an out-of-range label would otherwise alias through `target * num_classes + pred` into a
    neighbouring row and silently fabricate counts. Nothing is clamped, remapped, or discarded.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: pred {tuple(pred.shape)} vs target "
            f"{tuple(target.shape)}; they must be identical")
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # Targets: valid class ids, or the ignore label. Checked before masking.
    bad_t = (target != ignore_index) & ((target < 0) | (target >= num_classes))
    if bool(bad_t.any()):
        raise ValueError(
            f"target label out of range: found {int(target[bad_t][0])} "
            f"({int(bad_t.sum())} offending pixel(s)); valid targets are 0..{num_classes - 1} "
            f"or the ignore label {ignore_index}")

    valid = target != ignore_index
    pred = pred[valid].long()
    target = target[valid].long()

    # Predictions: only those at non-ignored positions enter the matrix, so only those are
    # validated -- this keeps ignore-invariance intact (predictions under ignored GT are irrelevant).
    if pred.numel():
        lo, hi = int(pred.min()), int(pred.max())
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"prediction label out of range: observed [{lo}, {hi}] at non-ignored positions; "
                f"valid predictions are 0..{num_classes - 1}")

    k = target * num_classes + pred
    cm = torch.bincount(k, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def _miou_from_cm(cm: torch.Tensor, class_indices: torch.Tensor | None = None):
    diag = torch.diag(cm).float()
    row = cm.sum(1).float()                 # GT pixels per class
    col = cm.sum(0).float()                 # predicted pixels per class
    iou = diag / (row + col - diag).clamp_min(1e-9)
    present = row > 0                        # classes present in ground truth
    if class_indices is not None:
        sel = torch.zeros_like(present)
        sel[class_indices] = True
        present = present & sel
    if present.any():
        return float(iou[present].mean().item()), iou
    return float("nan"), iou


# --------------------------------------------------------------------------------------------------
# dataset-level reducers (EVALUATION_CONTRACT section 3.1)
# --------------------------------------------------------------------------------------------------
def _cm_parts(cm: torch.Tensor):
    """Per-class counts from an accumulated CM. Carries NO eligibility policy -- each public
    dataset-level reducer applies its own rule, so the intentional asymmetries stay explicit."""
    tp = torch.diag(cm).float()          # intersection
    gt = cm.sum(1).float()               # ground-truth support   (MMSeg area_label)
    pr = cm.sum(0).float()               # prediction support     (MMSeg area_pred_label)
    return tp, gt, pr, gt + pr - tp      # ... and the union


def _restrict(eligible: torch.Tensor, class_indices: torch.Tensor | None) -> torch.Tensor:
    """Intersect an eligibility mask with an optional class subset (e.g. disease-only 1..115)."""
    if class_indices is None:
        return eligible
    sel = torch.zeros_like(eligible)
    sel[class_indices] = True
    return eligible & sel


def _macro(values: torch.Tensor, eligible: torch.Tensor) -> float:
    """Unweighted mean over eligible classes; NaN when nothing is eligible."""
    if not bool(eligible.any()):
        return float("nan")
    return float(values[eligible].mean().item())


def miou_from_confusion(cm: torch.Tensor, class_indices: torch.Tensor | None = None) -> float:
    """DATASET-LEVEL mIoU -- UNION-present (contract section 3.1).

    Eligible iff `UN_c = GT_c + PR_c - TP_c > 0`, so a prediction-only class contributes IoU 0 and
    is counted, while a class absent from GT and prediction alike is omitted. `class_indices=None`
    -> all-class; otherwise restrict (disease-only = every class except background).

    This is the reducer the E1 validation loop uses for its accumulate-ONE-confusion-matrix-then-
    compute-once protocol, so checkpoint-selection mIoU follows the contract automatically.
    NOT the per-image rule -- per-image metrics stay GT-present via `_miou_from_cm`.
    Returns NaN if no selected class is eligible.
    """
    tp, gt, pr, un = _cm_parts(cm)
    eligible = _restrict(un > 0, class_indices)
    return _macro(tp / un.clamp_min(1e-9), eligible)


def macc_from_confusion(cm: torch.Tensor, class_indices: torch.Tensor | None = None) -> float:
    """DATASET-LEVEL mAcc -- GT-present (contract section 3.1). `Acc_c = TP_c / GT_c`.

    Eligible iff `GT_c > 0`, so prediction-only classes are EXCLUDED. The different denominator
    from `miou_from_confusion` is deliberate and mirrors MMSeg (`acc = intersect / label`).
    """
    tp, gt, _pr, _un = _cm_parts(cm)
    eligible = _restrict(gt > 0, class_indices)
    return _macro(tp / gt.clamp_min(1e-9), eligible)


def dice_from_confusion(cm: torch.Tensor, class_indices: torch.Tensor | None = None) -> float:
    """DATASET-LEVEL macro per-class Dice (DSC) -- UNION-present (contract section 3.1).

    `Dice_c = 2*TP_c / (GT_c + PR_c)`, eligible iff `GT_c + PR_c > 0` (equivalently `UN_c > 0`).
    PROJECT-DEFINED DESCRIPTIVE METRIC: the official PlantSeg evaluator requests
    `iou_metrics=['mIoU']` and never computes mDice, so this is a thesis-internal extension and is
    NOT evidence of comparability with published PlantSeg results. Not binary foreground Dice.
    """
    tp, gt, pr, _un = _cm_parts(cm)
    eligible = _restrict((gt + pr) > 0, class_indices)
    return _macro(2.0 * tp / (gt + pr).clamp_min(1e-9), eligible)


def all_class_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                   ignore_index: int = IGNORE_INDEX) -> float:
    """Dataset-level all-class mIoU (union-present) over one pred/target pair."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    return miou_from_confusion(cm)


def disease_only_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                      background_index: int = 0, ignore_index: int = IGNORE_INDEX) -> float:
    """Dataset-level disease-only mIoU (union-present), excluding `background_index` (= 0).

    The class mapping is confirmed by the official PlantSeg source (METAINFO index 0 is the
    non-disease slot, 1..115 are the diseases; num_classes=116; reduce_zero_label=False), so this
    is no longer PROVISIONAL -- see open_questions #2.
    """
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    idx = torch.tensor([c for c in range(num_classes) if c != background_index], dtype=torch.long)
    return miou_from_confusion(cm, idx)


def per_image_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                   class_indices: torch.Tensor | None = None,
                   ignore_index: int = IGNORE_INDEX) -> list[float]:
    """One mIoU per image (paired-test unit). class_indices=None -> all-class; else restrict (e.g. disease-only)."""
    out = []
    for p, t in zip(pred, target):
        cm = confusion_matrix(p, t, num_classes, ignore_index)
        out.append(_miou_from_cm(cm, class_indices)[0])
    return out
