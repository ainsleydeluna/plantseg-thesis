"""Keep the training-only CWD projection out of the deployable E3 student.

Contract B3/B4: the 1x1 CWD projection head exists ONLY during E3 training and is "removed before
E6/E7 via state_dict edit prior to observer insertion"; IMPLEMENTATION_CONTRACT.md also states the
projection is "absent from every evaluated model".

This module enforces that structurally rather than by convention: `train_distill.py` never merges the
projection into the student module, and writes it to a SEPARATE checkpoint key, so
`checkpoint["model_state_dict"]` is already the clean deployment student that E6/E7 consume.
`assert_clean_student_state` is the guard that proves it, and `strip_cwd_projection` repairs a
state_dict that was produced some other way.
"""

from __future__ import annotations

from typing import Mapping

CWD_PROJECTION_KEY = "cwd_projection_state_dict"
CWD_PROJECTION_PREFIXES = ("cwd_projection", "cwd_proj", "projection_head")


class CWDProjectionLeak(RuntimeError):
    """Raised when a training-only CWD projection tensor is found in a deployment student state."""


def find_projection_keys(state: Mapping[str, object]) -> list[str]:
    """Return any keys that belong to the training-only CWD projection."""
    return [k for k in state
            if any(k == p or k.startswith(p + ".") for p in CWD_PROJECTION_PREFIXES)]


def assert_clean_student_state(state: Mapping[str, object]) -> None:
    """Raise `CWDProjectionLeak` if a deployment student state still carries the projection."""
    leaked = find_projection_keys(state)
    if leaked:
        raise CWDProjectionLeak(
            f"the training-only CWD projection is present in the student state_dict: {leaked}. "
            "It must be absent from every evaluated/quantized model (contract B3/B4); E6/E7 load "
            f"checkpoint['model_state_dict'], and the projection belongs under '{CWD_PROJECTION_KEY}'.")


def strip_cwd_projection(state: Mapping[str, object]) -> dict:
    """Return a copy of `state` with all training-only CWD projection entries removed."""
    drop = set(find_projection_keys(state))
    return {k: v for k, v in state.items() if k not in drop}


def deployment_student_state(checkpoint: Mapping[str, object]) -> dict:
    """Extract the E6/E7-ready student state from an E2/E3 checkpoint, verifying it is clean."""
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint has no 'model_state_dict'")
    state = dict(checkpoint["model_state_dict"])  # type: ignore[arg-type]
    assert_clean_student_state(state)
    return state
