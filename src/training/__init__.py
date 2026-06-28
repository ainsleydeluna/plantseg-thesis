"""Loss scaffolds for the PlantSeg student (E1 supervised; E2/E3 distillation)."""

from .losses import (
    CombinedCEDiceLoss,
    SoftDiceLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
    cwd_channelwise_kl,
    logit_kd_kl,
)

__all__ = [
    "WeightedCrossEntropyLoss",
    "SoftDiceLoss",
    "CombinedCEDiceLoss",
    "compute_class_weights",
    "logit_kd_kl",
    "cwd_channelwise_kl",
]
