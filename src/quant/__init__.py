"""INT8 quantization core for E4 (PTQ from E1) and E5 (QAT from E1).

Frozen to the registered PyTorch 2.1 eager-mode `torch.ao.quantization` API — deliberately NOT
migrated to torchao / PT2E. E6/E7 (which derive from E3 with the CWD head removed) are not
implemented here.
"""

from .calibration import (CALIBRATION_COUNT, CALIBRATION_SCHEMA, CALIBRATION_SEED,
                          CALIBRATION_SPLIT, CalibrationIndexError, build_calibration_index,
                          load_calibration_index, save_calibration_index,
                          verify_calibration_index)
from .checkpoint import (SourceCheckpointInvalid, load_e1_source_checkpoint,
                         load_e3_source_checkpoint, load_student_from_e1, load_student_from_e3)
from .stages import (QUANT_STAGES, SHARED_CALIBRATION_STAGES, QuantStageError,
                     load_source_for_stage, prepare_for_stage, require_shared_calibration_index,
                     resolve_quant_stage)
from .prepare import (BN_FREEZE_PCT_RANGE, QuantPreparationError, bn_freeze_iteration, calibrate,
                      convert_model, disable_observers, enable_observers, freeze_bn_stats,
                      prepare_ptq, prepare_qat_model, qat_grad_clip_gate_error,
                      quantization_coverage, try_converted_forward)
from .qconfig import (QUANT_BACKEND, X86_ACT_REDUCE_RANGE, X86_BACKENDS, QuantBackendUnavailable,
                      describe_qconfig, ptq_qconfig, qat_qconfig, select_qnnpack_backend,
                      select_x86_backend, x86_ptq_qconfig, x86_qat_qconfig)
from .x86_latency import (ACCURACY_ARTIFACT_ROLE, QAT_SIDECAR_KIND, QAT_SIDECAR_ROLE,
                          QAT_SIDECAR_SUFFIX, X86_LATENCY_ARTIFACT_ROLE, X86_PTQ_STAGES,
                          X86_QAT_STAGES, X86_RECONSTRUCTIBLE_STAGES, X86LatencyCopyError,
                          assert_trained_state_preserved, build_x86_latency_copy, load_qat_sidecar,
                          prepare_x86_ptq, prepare_x86_qat, qat_sidecar_missing_error,
                          require_x86_calibration_identity, translate_qat_state_to_x86,
                          x86_artifact_provenance)

__all__ = [
    "QUANT_BACKEND", "QuantBackendUnavailable", "select_qnnpack_backend",
    "ptq_qconfig", "qat_qconfig", "describe_qconfig",
    # x86/fbgemm CPU-proxy LATENCY copy (efficiency only; never the accuracy artifact)
    "X86_BACKENDS", "X86_ACT_REDUCE_RANGE", "select_x86_backend", "x86_ptq_qconfig",
    "x86_qat_qconfig", "ACCURACY_ARTIFACT_ROLE", "X86_LATENCY_ARTIFACT_ROLE",
    "X86_RECONSTRUCTIBLE_STAGES", "X86_PTQ_STAGES", "X86_QAT_STAGES", "X86LatencyCopyError",
    "QAT_SIDECAR_KIND", "QAT_SIDECAR_ROLE", "QAT_SIDECAR_SUFFIX", "load_qat_sidecar",
    "translate_qat_state_to_x86", "assert_trained_state_preserved", "prepare_x86_qat",
    "build_x86_latency_copy", "prepare_x86_ptq", "qat_sidecar_missing_error",
    "require_x86_calibration_identity", "x86_artifact_provenance",
    "SourceCheckpointInvalid", "load_e1_source_checkpoint", "load_student_from_e1",
    "load_e3_source_checkpoint", "load_student_from_e3",
    "QUANT_STAGES", "SHARED_CALIBRATION_STAGES", "QuantStageError", "resolve_quant_stage",
    "load_source_for_stage", "prepare_for_stage", "require_shared_calibration_index",
    "CALIBRATION_SCHEMA", "CALIBRATION_COUNT", "CALIBRATION_SEED", "CALIBRATION_SPLIT",
    "CalibrationIndexError", "build_calibration_index", "verify_calibration_index",
    "save_calibration_index", "load_calibration_index",
    "QuantPreparationError", "prepare_ptq", "calibrate", "prepare_qat_model", "convert_model",
    "freeze_bn_stats", "disable_observers", "enable_observers", "bn_freeze_iteration",
    "qat_grad_clip_gate_error", "BN_FREEZE_PCT_RANGE",
    "quantization_coverage", "try_converted_forward",
]
