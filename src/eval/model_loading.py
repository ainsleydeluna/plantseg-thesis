"""FP32 student construction and strict repository-checkpoint loading (A2b).

Separated from `adapters.py` so dataset identity and model/checkpoint concerns stay independent:
a future teacher / PTQ / QAT adapter replaces this module without touching the dataset layer, and
vice versa.

Only the ONE documented repository checkpoint schema is accepted -- the nested dict written by
`src/training/train_e1.py::save_checkpoint`:

    {"iter", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
     "scheduler", "best_val_miou_all_class", "num_classes"}

Bare state dicts and any other undocumented shape are rejected loudly rather than silently
supported.

Import-time behaviour is side-effect free: no model is constructed and no file is read on import.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from ..models.student import build_student

FROZEN_NUM_CLASSES = 116

# `torch.load` weights_only policy (torch 2.9.1 reports a default of None, i.e. version-dependent
# resolution). We therefore pass it EXPLICITLY rather than inheriting a shifting library default.
#
# weights_only=True is both the safest option and fully compatible with the repository schema:
# every value in the nested checkpoint is a plain int/float/str, a dict/list thereof, or a
# torch.Tensor -- all of which the weights_only unpickler allows. No custom classes, no
# numpy scalars, no lambdas are stored. If a future checkpoint needs a richer type, that is a
# deliberate schema change and must be handled explicitly, not by relaxing this flag.
TORCH_LOAD_WEIGHTS_ONLY = True

REQUIRED_KEYS = ("model_state_dict", "num_classes")


class CheckpointError(RuntimeError):
    """Any rejected checkpoint. Always fatal -- never degrade to a partial load."""


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata the artifact request needs. `path` is caller-supplied; normalise before recording."""
    path: str
    sha256: str
    num_classes: int
    iteration: int | None
    best_val_miou_all_class: float | None


def sha256_file(path: Path) -> str:
    """SHA-256 over the exact raw checkpoint bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_fp32_student(num_classes: int = FROZEN_NUM_CLASSES, device: str = "cpu"):
    """FP32 student via the repository factory. No download, no architecture duplication.

    `pretrained=False` is forced: ImageNet init belongs to training, not evaluation, and would
    otherwise reach for the torch-hub cache or the network.
    """
    if num_classes != FROZEN_NUM_CLASSES:
        raise CheckpointError(
            f"frozen contract requires num_classes={FROZEN_NUM_CLASSES}, got {num_classes}")
    model = build_student(num_classes=num_classes, pretrained=False)
    if getattr(model, "used_pretrained", False):
        raise CheckpointError("build_student unexpectedly loaded pretrained weights")
    return model.to(torch.device(device)).eval()


def _looks_like_bare_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(k, str) for k in obj) and \
        all(isinstance(v, torch.Tensor) for v in obj.values())


def load_student_checkpoint(path, *, map_location: str = "cpu",
                            num_classes: int = FROZEN_NUM_CLASSES):
    """Strictly load the repository's nested FP32-student checkpoint.

    Returns `(model, CheckpointInfo)`. Raises CheckpointError on any deviation.
    """
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"checkpoint file not found: {p}")
    digest = sha256_file(p)

    obj = torch.load(p, map_location=map_location, weights_only=TORCH_LOAD_WEIGHTS_ONLY)

    if _looks_like_bare_state_dict(obj):
        raise CheckpointError(
            "this is a BARE state_dict (all values are tensors). The repository schema is a "
            f"nested dict containing {list(REQUIRED_KEYS)}; bare state dicts are rejected so an "
            "unverifiable class count can never be assumed.")
    if not isinstance(obj, dict):
        raise CheckpointError(
            f"checkpoint must be a nested dict, got {type(obj).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise CheckpointError(
            f"checkpoint is missing required key(s) {missing}; present keys: {sorted(obj)}")

    ck_classes = obj["num_classes"]
    if not isinstance(ck_classes, int) or ck_classes != FROZEN_NUM_CLASSES:
        raise CheckpointError(
            f"checkpoint num_classes={ck_classes!r} != required {FROZEN_NUM_CLASSES}")
    if num_classes != FROZEN_NUM_CLASSES:
        raise CheckpointError(f"requested num_classes={num_classes} != {FROZEN_NUM_CLASSES}")

    model = build_fp32_student(num_classes=FROZEN_NUM_CLASSES, device=map_location)
    try:
        model.load_state_dict(obj["model_state_dict"], strict=True)
    except Exception as e:                                    # noqa: BLE001 -- surfaced verbatim
        raise CheckpointError(
            f"strict state_dict load failed: {type(e).__name__}: {str(e)[:300]}") from e

    info = CheckpointInfo(
        path=str(p),
        sha256=digest,
        num_classes=FROZEN_NUM_CLASSES,
        iteration=obj.get("iter") if isinstance(obj.get("iter"), int) else None,
        best_val_miou_all_class=(float(obj["best_val_miou_all_class"])
                                 if isinstance(obj.get("best_val_miou_all_class"), (int, float))
                                 else None),
    )
    return model.eval(), info
