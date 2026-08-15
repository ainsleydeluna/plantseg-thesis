"""Strict E1-source checkpoint contract for E4 (PTQ) and E5 (QAT).

Contract B4 stage table: **E4 <- E1** and **E5 <- E1**. E6/E7 derive from E3 with the CWD head
removed and are NOT implemented here. This loader therefore refuses anything that is not a clean
FP32 E1 student:

  * no `model_state_dict`                       -> reject
  * structurally incompatible with PlantSegStudent -> reject
  * stage metadata says E2/E3 (or anything else) -> reject
  * training-only CWD projection keys present    -> reject
  * a separate `cwd_projection_state_dict`       -> reject
  * teacher-only state                           -> reject
  * already quantized / fake-quant / observer state -> reject (never accept as FP32 E1)

The E1 schema comes from `train_e1.py :: save_checkpoint`: keys `iter`, `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, `scheduler`, `best_val_miou_all_class`,
`num_classes`. The distillation checkpoints add `stage` and `cwd_projection_state_dict`, which is
exactly what makes an E2/E3 source detectable here.
"""

from __future__ import annotations

from pathlib import Path

import torch

CWD_PROJECTION_PREFIXES = ("cwd_projection", "cwd_proj", "projection_head")
CWD_PROJECTION_KEY = "cwd_projection_state_dict"
TEACHER_PREFIXES = ("teacher.", "backbone.block", "decode_head.")
QUANT_STATE_MARKERS = ("activation_post_process", "fake_quant", ".observer", "_packed_params",
                       "scale", "zero_point")
ALLOWED_SOURCE_STAGES = ("E1",)


class SourceCheckpointInvalid(ValueError):
    """Raised when a checkpoint is not an acceptable FP32 E1 source for E4/E5."""


def _reject(msg: str) -> None:
    raise SourceCheckpointInvalid(msg)


def load_e1_source_checkpoint(path: str | Path, *, expected_num_classes: int = 116,
                              map_location: str = "cpu") -> dict:
    """Load and validate an E1 FP32 checkpoint for E4/E5. Returns the checkpoint dict.

    Fails loudly on every violation; there is no permissive mode.
    """
    p = Path(path)
    if not p.is_file():
        _reject(f"E1 source checkpoint not found: {p}")
    try:
        ckpt = torch.load(str(p), map_location=map_location, weights_only=False)
    except Exception as e:  # noqa: BLE001
        _reject(f"could not read E1 source checkpoint {p}: {type(e).__name__}: {e}")

    if not isinstance(ckpt, dict):
        _reject(f"checkpoint {p} holds {type(ckpt).__name__}, expected a dict")
    if "model_state_dict" not in ckpt:
        _reject(f"checkpoint {p} has no 'model_state_dict' — not a student training checkpoint")

    state = ckpt["model_state_dict"]
    if not isinstance(state, dict) or not state:
        _reject(f"checkpoint {p} has an empty or non-dict model_state_dict")
    if not any(torch.is_tensor(v) for v in state.values()):
        _reject(f"checkpoint {p} model_state_dict holds no tensors")

    # --- stage provenance: E4/E5 must come from E1, never E2/E3 ---
    stage = ckpt.get("stage")
    if stage is not None and str(stage).upper() not in ALLOWED_SOURCE_STAGES:
        _reject(f"checkpoint {p} is stage {stage!r}; E4/E5 take the E1 FP32 checkpoint only "
                "(E6/E7 are the stages that derive from E3)")

    # --- distillation artefacts must be absent ---
    if CWD_PROJECTION_KEY in ckpt and ckpt[CWD_PROJECTION_KEY] is not None:
        _reject(f"checkpoint {p} carries '{CWD_PROJECTION_KEY}' — that is an E3 checkpoint; the "
                "training-only CWD projection must never enter a quantized model")
    leaked = [k for k in state
              if any(k == pre or k.startswith(pre + ".") for pre in CWD_PROJECTION_PREFIXES)]
    if leaked:
        _reject(f"checkpoint {p} model_state_dict contains CWD projection keys {leaked}")

    # --- teacher contamination ---
    teacher_keys = [k for k in state if k.startswith("teacher.")]
    if teacher_keys:
        _reject(f"checkpoint {p} contains teacher state {teacher_keys[:3]}; the teacher is never "
                "part of a student checkpoint")

    # --- must be FP32, not an already-quantized or QAT-instrumented model ---
    quant_keys = [k for k in state
                  if "activation_post_process" in k or "fake_quant" in k or "_packed_params" in k]
    if quant_keys:
        _reject(f"checkpoint {p} already carries quantization state {quant_keys[:3]} — E4/E5 must "
                "start from a clean FP32 E1 model, not a prepared/converted one")
    non_float = [k for k, v in state.items() if torch.is_tensor(v) and v.dtype in
                 (torch.qint8, torch.quint8, torch.qint32)]
    if non_float:
        _reject(f"checkpoint {p} holds quantized tensors {non_float[:3]} — expected FP32")

    if ckpt.get("num_classes") is not None and int(ckpt["num_classes"]) != expected_num_classes:
        _reject(f"checkpoint {p} declares num_classes={ckpt['num_classes']}, expected "
                f"{expected_num_classes}")
    return ckpt


def load_student_from_e1(path: str | Path, *, expected_num_classes: int = 116,
                         map_location: str = "cpu"):
    """Validate the checkpoint and load it into a fresh FP32 `PlantSegStudent` (strict)."""
    ckpt = load_e1_source_checkpoint(path, expected_num_classes=expected_num_classes,
                                     map_location=map_location)
    return _build_strict("E1", ckpt["model_state_dict"], path), ckpt


def _build_strict(label: str, state: dict, path: str | Path):
    """Load a validated state dict into a fresh FP32 student, requiring an exact structural match."""
    from src.models.student import build_student

    model = build_student(pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    if missing or unexpected:
        _reject(f"{label} state_dict is not structurally compatible with PlantSegStudent — "
                f"missing={missing[:5]} unexpected={unexpected[:5]} (from {path})")
    return model


def load_e3_source_checkpoint(path: str | Path, *, expected_num_classes: int = 116,
                              map_location: str = "cpu") -> dict:
    """Load and validate an **E3** checkpoint as the E6/E7 source.

    Contract B4 stage table: **E6 <- E3 (CWD head removed)** and **E7 <- E3 (CWD head removed)**;
    IMPLEMENTATION_CONTRACT also states the training-only projection is "absent from every evaluated
    model". `src/distill/export.py` is the authority for that interface: E3 writes the student under
    `model_state_dict` (already projection-free) and the training-only projection under the separate
    `cwd_projection_state_dict` field.

    This loader therefore takes the FULL E3 training checkpoint, validates via
    `deployment_student_state` that the projection is genuinely absent from `model_state_dict`, and
    then **explicitly ignores** the separate projection field — it is never loaded into the student.

    Stricter than the literal contract wording in one respect, deliberately: `stage` metadata is
    REQUIRED and must be `E3`. Our own `train_distill.py` always writes it, so a missing or
    different stage means the checkpoint is not a distillation product and must not silently become
    an E6/E7 source.
    """
    from src.distill.export import CWDProjectionLeak, deployment_student_state

    p = Path(path)
    if not p.is_file():
        _reject(f"E3 source checkpoint not found: {p}")
    try:
        ckpt = torch.load(str(p), map_location=map_location, weights_only=False)
    except Exception as e:  # noqa: BLE001
        _reject(f"could not read E3 source checkpoint {p}: {type(e).__name__}: {e}")
    if not isinstance(ckpt, dict):
        _reject(f"checkpoint {p} holds {type(ckpt).__name__}, expected a dict")

    stage = ckpt.get("stage")
    if stage is None:
        _reject(f"checkpoint {p} carries no 'stage' metadata; E6/E7 require an identified E3 "
                "source (E1/E2 checkpoints must never be quantized as if they were E3)")
    if str(stage).upper() != "E3":
        _reject(f"checkpoint {p} is stage {stage!r}; E6/E7 take the E3 checkpoint only")

    if "model_state_dict" not in ckpt:
        _reject(f"checkpoint {p} has no 'model_state_dict'")
    try:
        state = deployment_student_state(ckpt)      # asserts the projection is NOT in the student
    except CWDProjectionLeak as e:
        _reject(f"{e}")
    if not any(torch.is_tensor(v) for v in state.values()):
        _reject(f"checkpoint {p} model_state_dict holds no tensors")

    teacher_keys = [k for k in state if k.startswith("teacher.")]
    if teacher_keys:
        _reject(f"checkpoint {p} contains teacher state {teacher_keys[:3]} in the deployment "
                "student; the frozen teacher is never part of a quantized model")
    quant_keys = [k for k in state
                  if "activation_post_process" in k or "fake_quant" in k or "_packed_params" in k]
    if quant_keys:
        _reject(f"checkpoint {p} already carries quantization state {quant_keys[:3]} — E6/E7 must "
                "start from the clean FP32 E3 student")
    non_float = [k for k, v in state.items() if torch.is_tensor(v) and v.dtype in
                 (torch.qint8, torch.quint8, torch.qint32)]
    if non_float:
        _reject(f"checkpoint {p} holds quantized tensors {non_float[:3]} — expected FP32")
    if ckpt.get("num_classes") is not None and int(ckpt["num_classes"]) != expected_num_classes:
        _reject(f"checkpoint {p} declares num_classes={ckpt['num_classes']}, expected "
                f"{expected_num_classes}")
    return ckpt


def load_student_from_e3(path: str | Path, *, expected_num_classes: int = 116,
                         map_location: str = "cpu"):
    """Validate an E3 checkpoint and load ONLY its projection-free student into a fresh model."""
    from src.distill.export import deployment_student_state

    ckpt = load_e3_source_checkpoint(path, expected_num_classes=expected_num_classes,
                                     map_location=map_location)
    # deployment_student_state re-asserts cleanliness; the separate training-only projection field
    # is deliberately not touched.
    return _build_strict("E3", deployment_student_state(ckpt), path), ckpt
