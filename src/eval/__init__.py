"""Segmentation metrics + the stage-neutral evaluation core.

Metric semantics are frozen by docs/EVALUATION_CONTRACT.md sections 3.1/3.2: dataset-level mIoU
and macro Dice are union-present, dataset-level mAcc is GT-present, and both per-image vectors are
GT-present. See `metrics.py` for the full statement.

Import-time behaviour is side-effect free: no Git, no filesystem inspection, no directory
creation, no model construction. The artifact writer (`src.eval.artifacts`) is deliberately NOT
re-exported here -- A2b imports it explicitly.
"""

from .evaluate import (
    Condition,
    DatasetMeta,
    EvalBatch,
    EvalResult,
    EvaluationIntegrityError,
    ManifestEntry,
    PerImageRow,
    RunMeta,
    evaluate_model,
)
from .metrics import (
    all_class_miou,
    confusion_matrix,
    dice_from_confusion,
    disease_only_miou,
    macc_from_confusion,
    miou_from_confusion,
    per_image_miou,
)

__all__ = [
    # metrics (production reducers -- the only source of aggregate values)
    "confusion_matrix",
    "all_class_miou",
    "disease_only_miou",
    "per_image_miou",
    "miou_from_confusion",
    "macc_from_confusion",
    "dice_from_confusion",
    # evaluation core
    "Condition",
    "EvalBatch",
    "EvalResult",
    "EvaluationIntegrityError",
    "ManifestEntry",
    "PerImageRow",
    "evaluate_model",
    # request metadata (pure dataclasses; the artifact WRITER stays in src.eval.artifacts)
    "RunMeta",
    "DatasetMeta",
]
