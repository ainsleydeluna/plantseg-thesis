"""PlantSeg student model scaffold (MobileNetV3-Large quantizable + LR-ASPP head)."""

from .student import LRASPPHead, PlantSegStudent, build_student, probe_backbone_stages

__all__ = ["build_student", "PlantSegStudent", "LRASPPHead", "probe_backbone_stages"]
