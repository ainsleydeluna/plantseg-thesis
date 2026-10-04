"""Distillation components for E2 (Logit KD) and E3 (Logit KD + Channel-Wise KD)."""

from .cwd_projection import STUDENT_C5_CH, TEACHER_STRIDE16_CH, build_cwd_projection
from .export import (CWD_PROJECTION_KEY, CWDProjectionLeak, assert_clean_student_state,
                     deployment_student_state, find_projection_keys, strip_cwd_projection)
from .features import StudentTaps
from .nmf_stream import M4_NMF_SEED, NMFStream, NMFStreamError, attach_nmf_stream
from .segnext_teacher import (PLANTSEG_CONFIG_STEM, STAGE3_CHANNELS, STAGE3_STRIDE,
                              STOCK_INIT_CONFIG_STEM, SegNeXtTeacherAdapter,
                              TeacherArchitectureMismatch, TeacherCheckpointInvalid,
                              TeacherStateDictMismatch, architecture_signature,
                              build_segnext_teacher, load_teacher_state_dict,
                              segnext_builder, select_stride16_feature, strict_load_teacher_state)
from .teacher import (FrozenTeacher, MockTeacher, TeacherCheckpointMissing,
                      TeacherChecksumFormatError, TeacherChecksumMismatch, TeacherOutput,
                      TeacherProvenance, TeacherStackMissing, build_mmseg_teacher,
                      load_frozen_teacher, require_teacher_checkpoint, sha256_format_error)

__all__ = [
    "build_cwd_projection", "STUDENT_C5_CH", "TEACHER_STRIDE16_CH",
    "StudentTaps",
    "M4_NMF_SEED", "NMFStream", "NMFStreamError", "attach_nmf_stream",
    "FrozenTeacher", "MockTeacher", "TeacherOutput", "TeacherProvenance",
    "TeacherCheckpointMissing", "TeacherStackMissing", "TeacherChecksumFormatError",
    "TeacherChecksumMismatch", "sha256_format_error",
    "build_mmseg_teacher", "load_frozen_teacher", "require_teacher_checkpoint",
    "SegNeXtTeacherAdapter", "TeacherArchitectureMismatch", "TeacherCheckpointInvalid",
    "TeacherStateDictMismatch", "architecture_signature", "strict_load_teacher_state",
    "build_segnext_teacher", "load_teacher_state_dict", "segnext_builder",
    "select_stride16_feature", "STAGE3_CHANNELS", "STAGE3_STRIDE",
    "STOCK_INIT_CONFIG_STEM", "PLANTSEG_CONFIG_STEM",
    "CWD_PROJECTION_KEY", "CWDProjectionLeak", "assert_clean_student_state",
    "deployment_student_state", "find_projection_keys", "strip_cwd_projection",
]
