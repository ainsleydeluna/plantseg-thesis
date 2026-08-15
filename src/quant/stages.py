"""Thin stage wrappers binding each quantization experiment to its source and method.

The four quantization stages are the SAME two mechanisms applied to two different FP32 sources
(contract B4 stage table) — so there is one PTQ implementation and one QAT implementation, and this
module only routes:

    E4 : E1 FP32 checkpoint                  -> PTQ
    E5 : E1 FP32 checkpoint                  -> QAT
    E6 : E3 projection-free student          -> QAT   ("identical config as E5")
    E7 : E3 projection-free student          -> PTQ   (EXACT same calibration subset as E4)

No stage reintroduces the teacher, Logit KD, CWD loss or the training-only CWD projection: E6/E7
start from the completed E3 *weights* and quantize them. `configs/quant.py` sets
`distillation_during_qat = False`, so QAT is supervised-only for both E5 and E6.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from .calibration import CalibrationIndexError, load_calibration_index
from .checkpoint import load_student_from_e1, load_student_from_e3
from .prepare import prepare_ptq, prepare_qat_model

QUANT_STAGES: dict[str, dict] = {
    "e4": {"name": "E4", "method": "ptq", "source_stage": "E1",
           "objective": "static INT8 PTQ from the E1 FP32 checkpoint"},
    "e5": {"name": "E5", "method": "qat", "source_stage": "E1",
           "objective": "INT8 QAT from the E1 FP32 checkpoint"},
    "e6": {"name": "E6", "method": "qat", "source_stage": "E3",
           "objective": "INT8 QAT from the E3 projection-free student (identical config to E5)"},
    "e7": {"name": "E7", "method": "ptq", "source_stage": "E3",
           "objective": "static INT8 PTQ from the E3 projection-free student "
                        "(same calibration subset as E4)"},
}

# Stages that must consume the SAME persisted calibration artifact (contract B4 /
# configs/quant.py `calibration_subset_shared`).
SHARED_CALIBRATION_STAGES = ("E4", "E7")


class QuantStageError(ValueError):
    """Raised for an unknown stage or a stage/source mismatch."""


def resolve_quant_stage(stage: str) -> dict:
    key = str(stage).lower()
    if key not in QUANT_STAGES:
        raise QuantStageError(f"unknown quantization stage {stage!r}; expected one of "
                              f"{sorted(QUANT_STAGES)}")
    return {"key": key, **QUANT_STAGES[key]}


def load_source_for_stage(stage: str, ckpt_path: str | Path, **kw):
    """Load and validate the FP32 source checkpoint required by `stage`.

    E4/E5 route to the strict E1 contract; E6/E7 to the strict E3 contract, which accepts only the
    projection-free student and never loads the training-only CWD projection.
    """
    st = resolve_quant_stage(stage)
    if st["source_stage"] == "E1":
        return load_student_from_e1(ckpt_path, **kw)
    return load_student_from_e3(ckpt_path, **kw)


def prepare_for_stage(stage: str, model: nn.Module, *, select_backend: bool = True) -> nn.Module:
    """Prepare `model` with the stage's method — the same PTQ/QAT code paths as E4/E5."""
    st = resolve_quant_stage(stage)
    if st["method"] == "ptq":
        return prepare_ptq(model, select_backend=select_backend)
    return prepare_qat_model(model, select_backend=select_backend)


def require_shared_calibration_index(path: str | Path,
                                     expected_checksum: str | None = None) -> dict:
    """Load the E4/E7 calibration artifact, verifying it is the SHARED one.

    E7 must reuse the exact artifact E4 used — never an independently resampled list. Passing the
    E4 index's `checksum_sha256` as `expected_checksum` makes that binding enforceable: a
    differently-sampled index is rejected even though it would be internally valid on its own.
    """
    index = load_calibration_index(path)             # validates schema, count, seed and checksum
    shared = [s.upper() for s in index.get("shared_by", [])]
    for required in SHARED_CALIBRATION_STAGES:
        if required not in shared:
            raise CalibrationIndexError(
                f"calibration index at {path} is not marked as shared by {required} "
                f"(shared_by={index.get('shared_by')}); E4 and E7 must consume the same subset")
    if expected_checksum is not None and index["checksum_sha256"] != expected_checksum:
        raise CalibrationIndexError(
            f"calibration index checksum {index['checksum_sha256'][:16]}… does not match the "
            f"expected {expected_checksum[:16]}… — E7 must reuse the EXACT artifact E4 used, not a "
            "separately generated list.")
    return index
