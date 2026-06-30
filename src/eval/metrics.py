"""Segmentation metric scaffolds (Blocker B11). Confusion-matrix based; ignore_index=255 excluded.

Per docs/IMPLEMENTATION_CONTRACT.md (f):
  - all-class mIoU: macro IoU over all classes (0-115; background 0 + 115 diseases) present in GT.
  - disease-only mIoU: macro IoU over disease classes only (background index excluded). The background
    index is empirically 0 but the formal disease-only convention is NEED_TO_CONFIRM (open_questions #2),
    so disease-only results are labelled PROVISIONAL.
  - reduce_zero_label is NOT used (NEED_TO_CONFIRM).
  - 255 (pad/ignore) excluded from every computation.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = 255


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                     ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """[num_classes, num_classes] confusion matrix (rows=GT, cols=pred); ignore_index pixels dropped."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = target != ignore_index
    pred = pred[valid].long()
    target = target[valid].long()
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


def miou_from_confusion(cm: torch.Tensor, class_indices: torch.Tensor | None = None) -> float:
    """Public mIoU from an (already accumulated) confusion matrix.

    class_indices=None -> macro IoU over all GT-present classes; otherwise restrict to those indices
    (e.g. disease-only = every class except background). This is the helper the E1 validation loop uses
    for the accumulate-ONE-confusion-matrix-then-compute-once protocol (see
    reports/e1_training_loop_readiness.md); it delegates to the same `_miou_from_cm` core used by the
    per-tensor metrics, so behaviour is identical. Returns NaN if no selected class is present in GT.
    """
    return _miou_from_cm(cm, class_indices)[0]


def all_class_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                   ignore_index: int = IGNORE_INDEX) -> float:
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    return _miou_from_cm(cm)[0]


def disease_only_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                      background_index: int = 0, ignore_index: int = IGNORE_INDEX) -> float:
    """PROVISIONAL / NEED_TO_CONFIRM: excludes `background_index` (empirically 0). Not a settled convention."""
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    idx = torch.tensor([c for c in range(num_classes) if c != background_index], dtype=torch.long)
    return _miou_from_cm(cm, idx)[0]


def per_image_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                   class_indices: torch.Tensor | None = None,
                   ignore_index: int = IGNORE_INDEX) -> list[float]:
    """One mIoU per image (paired-test unit). class_indices=None -> all-class; else restrict (e.g. disease-only)."""
    out = []
    for p, t in zip(pred, target):
        cm = confusion_matrix(p, t, num_classes, ignore_index)
        out.append(_miou_from_cm(cm, class_indices)[0])
    return out
