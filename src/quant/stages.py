"""Thin stage wrappers binding each quantization experiment to its source and method.

The four quantization stages are the SAME two mechanisms applied to two different FP32 sources
(contract B4 stage table) — so there is one PTQ implementation and one QAT implementation, and this
module only routes:

    E4 : E1 FP32 checkpoint                  -> PTQ
    E5 : E1 FP32 checkpoint                  -> QAT
    E6 : E3 projection-free student          -> QAT   ("identical config as E5")
    E7 : E3 projection-free student          -> PTQ   (EXACT same calibration subset as E4)
    E6-KD : E3 projection-free student       -> QAT_KD (AM-3: E6's QAT with the teacher of record, E3's terms at 0.5x)
    E6-E2 : E2 seed-42 student (λ winner)    -> QAT   (AM-23 item 1: E6's recipe; only the parent differs)
    E7-E2 : E2 seed-42 student (λ winner)    -> PTQ   (AM-23 item 2: the AM-10 list of record)
    E6-R1 : E3-R1 projection-free student    -> QAT   (AM-23 items 3(a), 4: when item 3(b) gives yes)
    E6-R2 : E3-R2 projection-free student    -> QAT   (likewise)

AM-23's four stages (AM23_STAGES) are descriptive, run at seed 42 only and are never a parent.

No stage except E6-KD reintroduces the teacher, Logit KD, CWD loss or the training-only CWD projection: E6/E7
start from the completed E3 *weights* and quantize them. `configs/quant.py` sets
`distillation_during_qat = False`, so QAT is supervised-only for both E5 and E6. E6-KD (AM-3; DL-39, DL-51; lane
L-AM3, B-2), descriptive and at seed 42 only, is E6's QAT with the frozen teacher of record and E3's three
distillation terms at 0.5x, its CWD projection E3's trained one; method "qat_kd" runs it in src/quant/qat_kd.py.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from .calibration import CalibrationIndexError, load_calibration_index
from .checkpoint import load_student_from_e1, load_student_from_e2, load_student_from_e3
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
    # AM-3; DL-39, DL-51 (lane L-AM3, B-2): E6's QAT with the teacher of record in the loop and E3's three distillation
    # terms at 0.5x, seed 42 only, descriptive. Method "qat_kd": src/quant/runner.py hands it to src/quant/qat_kd.py.
    "e6-kd": {"name": "E6-KD", "method": "qat_kd", "source_stage": "E3",
              "objective": "INT8 QAT from the E3 projection-free student with the teacher of record and E3's "
                           "distillation terms at 0.5x (AM-3; DL-39, DL-51; descriptive)"},
    # AM-23 items 1-4: descriptive stages at seed 42 only, never a parent. An E6-X stage names its parent's arm.
    "e6-e2": {"name": "E6-E2", "method": "qat", "source_stage": "E2",
              "objective": "INT8 QAT from the E2 seed-42 student, the λ winner (AM-23 item 1; descriptive)"},
    "e7-e2": {"name": "E7-E2", "method": "ptq", "source_stage": "E2",
              "objective": "static INT8 PTQ from the E2 seed-42 student on the AM-10 list of record "
                           "(AM-23 item 2; descriptive)"},
    "e6-r1": {"name": "E6-R1", "method": "qat", "source_stage": "E3", "parent_arm": "E3-R1",
              "objective": "INT8 QAT from the E3-R1 projection-free student (AM-23 items 3(a), 4; descriptive)"},
    "e6-r2": {"name": "E6-R2", "method": "qat", "source_stage": "E3", "parent_arm": "E3-R2",
              "objective": "INT8 QAT from the E3-R2 projection-free student (AM-23 items 3(a), 4; descriptive)"},
}

# Stages that must consume the SAME persisted calibration artifact (contract B4 /
# configs/quant.py `calibration_subset_shared`).
SHARED_CALIBRATION_STAGES = ("E4", "E7")
# AM-23's stages, in the amendment's order: descriptive, seed 42 only, never a parent of a run (items 1-4).
AM23_STAGES = ("E6-E2", "E7-E2", "E6-R1", "E6-R2")
# E6-X -> its parent's arm (AM-23 item 4: only the run directory of E3-X for the same X).
E6X_PARENT_ARM = {st["name"]: st["parent_arm"] for st in QUANT_STAGES.values() if "parent_arm" in st}


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

    E4/E5 route to the strict E1 contract; E6/E7 and E6-R1/E6-R2 to the strict E3 contract, which accepts
    only the projection-free student and never loads the training-only CWD projection; E6-E2/E7-E2 to the
    strict E2 contract (AM-23 item 4). A source stage without a loader is refused, never routed to another.
    """
    st = resolve_quant_stage(stage)
    if st["source_stage"] == "E1":
        return load_student_from_e1(ckpt_path, **kw)
    if st["source_stage"] == "E2":
        return load_student_from_e2(ckpt_path, **kw)
    if st["source_stage"] == "E3":
        return load_student_from_e3(ckpt_path, **kw)
    raise QuantStageError(f"{st['name']} names source stage {st['source_stage']!r}, which has no loader")


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
