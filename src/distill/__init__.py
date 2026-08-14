"""Distillation components for E2 (Logit KD) and E3 (Logit KD + Channel-Wise KD)."""

from .cwd_projection import STUDENT_C5_CH, TEACHER_STRIDE16_CH, build_cwd_projection
from .export import (CWD_PROJECTION_KEY, CWDProjectionLeak, assert_clean_student_state,
                     deployment_student_state, find_projection_keys, strip_cwd_projection)
from .features import StudentTaps
from .teacher import (FrozenTeacher, MockTeacher, TeacherCheckpointMissing, TeacherOutput,
                      TeacherProvenance, TeacherStackMissing, load_frozen_teacher,
                      require_teacher_checkpoint)

__all__ = [
    "build_cwd_projection", "STUDENT_C5_CH", "TEACHER_STRIDE16_CH",
    "StudentTaps",
    "FrozenTeacher", "MockTeacher", "TeacherOutput", "TeacherProvenance",
    "TeacherCheckpointMissing", "TeacherStackMissing",
    "load_frozen_teacher", "require_teacher_checkpoint",
    "CWD_PROJECTION_KEY", "CWDProjectionLeak", "assert_clean_student_state",
    "deployment_student_state", "find_projection_keys", "strip_cwd_projection",
]
