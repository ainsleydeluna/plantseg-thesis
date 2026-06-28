"""Distillation scaffolds (training-only CWD projection; full CWD loss deferred to E3)."""

from .cwd_projection import STUDENT_C5_CH, TEACHER_STRIDE16_CH, build_cwd_projection

__all__ = ["build_cwd_projection", "STUDENT_C5_CH", "TEACHER_STRIDE16_CH"]
