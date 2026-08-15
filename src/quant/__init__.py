"""INT8 quantization core for E4 (PTQ from E1) and E5 (QAT from E1).

Frozen to the registered PyTorch 2.1 eager-mode `torch.ao.quantization` API — deliberately NOT
migrated to torchao / PT2E. E6/E7 (which derive from E3 with the CWD head removed) are not
implemented here.
"""

from .calibration import (CALIBRATION_COUNT, CALIBRATION_SCHEMA, CALIBRATION_SEED,
                          CALIBRATION_SPLIT, CalibrationIndexError, build_calibration_index,
                          load_calibration_index, save_calibration_index,
                          verify_calibration_index)
from .checkpoint import SourceCheckpointInvalid, load_e1_source_checkpoint, load_student_from_e1
from .prepare import (BN_FREEZE_PCT_RANGE, QuantPreparationError, bn_freeze_iteration, calibrate,
                      convert_model, disable_observers, enable_observers, freeze_bn_stats,
                      prepare_ptq, prepare_qat_model, qat_grad_clip_gate_error,
                      quantization_coverage, try_converted_forward)
from .qconfig import (QUANT_BACKEND, QuantBackendUnavailable, describe_qconfig, ptq_qconfig,
                      qat_qconfig, select_qnnpack_backend)

__all__ = [
    "QUANT_BACKEND", "QuantBackendUnavailable", "select_qnnpack_backend",
    "ptq_qconfig", "qat_qconfig", "describe_qconfig",
    "SourceCheckpointInvalid", "load_e1_source_checkpoint", "load_student_from_e1",
    "CALIBRATION_SCHEMA", "CALIBRATION_COUNT", "CALIBRATION_SEED", "CALIBRATION_SPLIT",
    "CalibrationIndexError", "build_calibration_index", "verify_calibration_index",
    "save_calibration_index", "load_calibration_index",
    "QuantPreparationError", "prepare_ptq", "calibrate", "prepare_qat_model", "convert_model",
    "freeze_bn_stats", "disable_observers", "enable_observers", "bn_freeze_iteration",
    "qat_grad_clip_gate_error", "BN_FREEZE_PCT_RANGE",
    "quantization_coverage", "try_converted_forward",
]
