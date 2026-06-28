"""Segmentation metric scaffolds (confusion-matrix based, ignore_index=255)."""

from .metrics import (
    all_class_miou,
    confusion_matrix,
    disease_only_miou,
    per_image_miou,
)

__all__ = [
    "confusion_matrix",
    "all_class_miou",
    "disease_only_miou",
    "per_image_miou",
]
