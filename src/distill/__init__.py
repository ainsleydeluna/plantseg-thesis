"""Distillation components for E2 (Logit KD) and E3 (Logit KD + Channel-Wise KD)."""

from .cwd_projection import STUDENT_C5_CH, TEACHER_STRIDE16_CH, build_cwd_projection
from .export import (CWD_PROJECTION_KEY, CWDProjectionLeak, assert_clean_student_state,
                     deployment_student_state, find_projection_keys, strip_cwd_projection)
from .features import StudentTaps
from .segnext_teacher import (PLANTSEG_CONFIG_STEM, STAGE3_CHANNELS, STAGE3_STRIDE,
                              STOCK_INIT_CONFIG_STEM, SegNeXtTeacherAdapter,
                              TeacherArchitectureMismatch, TeacherCheckpointInvalid,
                              build_segnext_teacher, load_teacher_state_dict,
                              segnext_builder, select_stride16_feature)
from .teacher import (FrozenTeacher, MockTeacher, TeacherCheckpointMissing, TeacherOutput,
                      TeacherProvenance, TeacherStackMissing, build_mmseg_teacher,
                      load_frozen_teacher, require_teacher_checkpoint)

__all__ = [
    "build_cwd_projection", "STUDENT_C5_CH", "TEACHER_STRIDE16_CH",
    "StudentTaps",
    "FrozenTeacher", "MockTeacher", "TeacherOutput", "TeacherProvenance",
    "TeacherCheckpointMissing", "TeacherStackMissing",
    "build_mmseg_teacher", "load_frozen_teacher", "require_teacher_checkpoint",
    "SegNeXtTeacherAdapter", "TeacherArchitectureMismatch", "TeacherCheckpointInvalid",
    "build_segnext_teacher", "load_teacher_state_dict", "segnext_builder",
    "select_stride16_feature", "STAGE3_CHANNELS", "STAGE3_STRIDE",
    "STOCK_INIT_CONFIG_STEM", "PLANTSEG_CONFIG_STEM",
    "CWD_PROJECTION_KEY", "CWDProjectionLeak", "assert_clean_student_state",
    "deployment_student_state", "find_projection_keys", "strip_cwd_projection",
]
