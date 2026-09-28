"""Training-only CWD projection head for the feature-map CWD term (Blocker B12; L-AM17B-FG).

Channel-Wise KD (Shu 2021) aligns the student's stride-16 C5 feature to the teacher's matching
channel dimension via a single 1x1 convolution applied to the student's C5 during training.

  student C5 (OS16) : 160 channels  -- B10-verified tap (features index 15); see
                                        reports/student_forward_smoke.md.
  teacher stride-16 : 320 channels  -- LOCKED per docs/IMPLEMENTATION_CONTRACT.md B3 / ch3
                                        (MSCAN-B Stage-3, C=320). Empirical teacher-side confirmation
                                        is the Table 3.1 forward-hook check, run when the SegNeXt-B
                                        teacher is built (out of scope here — no teacher code yet).

*** TRAINING-ONLY. *** The projection exists only while a stage with the feature-map CWD term
(`cwd_feat`: E3, A, F) trains; `build_cwd_projection_for` instantiates it behind that switch, so E2
and G never build one. It never enters the student module or its state_dict, and it is not written
into the student checkpoint either: `save_projection` writes it, with its optimizer param group, to
`projection.pt` beside the checkpoint, so the E3 checkpoint E6/E7 consume carries nothing to strip.
`restore_projection` is the only way back in. This module is not the CWD loss.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

STUDENT_C5_CH = 160       # B10-verified (student OS16/C5 tap, features index 15)
TEACHER_STRIDE16_CH = 320  # LOCKED: contract B3 / ch3 (MSCAN-B stride-16 Stage-3 channel count)
PROJECTION_PARAMS = STUDENT_C5_CH * TEACHER_STRIDE16_CH   # 51,200: bias-free 1x1 conv
PROJECTION_FILE = "projection.pt"                         # beside the student checkpoint
PROJECTION_GROUP = "cwd_projection"                       # its optimizer param-group name
PROJECTION_FORMAT = "cwd_projection/1"


def build_cwd_projection(in_ch: int = STUDENT_C5_CH, out_ch: int = TEACHER_STRIDE16_CH) -> nn.Conv2d:
    """Single bias-free 1x1 conv mapping student C5 (in_ch) -> teacher channel dim (out_ch).

    Training-only; never part of the student — see module docstring.
    """
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def build_cwd_projection_for(stage) -> nn.Conv2d | None:
    """The projection, instantiated only when the stage's `cwd_feat` term is on (L-AM17B-FG).

    `stage` is any mapping carrying the boolean `cwd_feat`. A stage without the feature-map term
    (E2, G) gets None: no module, hence no parameters, no optimizer param group and no projection.pt.
    """
    return build_cwd_projection() if stage["cwd_feat"] else None


def split_optimizer_state(state: dict, group: str = PROJECTION_GROUP) -> tuple[dict, dict]:
    """Split an `optimizer.state_dict()` into (the other groups, the named group).

    Parameter indices keep their global numbering, so `merge_optimizer_state` rebuilds exactly the
    original dict from the two halves.
    """
    groups = state["param_groups"]
    moved = [g for g in groups if g.get("name") == group]
    kept = [g for g in groups if g.get("name") != group]
    moved_ids = {i for g in moved for i in g["params"]}
    kept_ids = {i for g in kept for i in g["params"]}
    return ({"state": {i: s for i, s in state["state"].items() if i in kept_ids}, "param_groups": kept},
            {"state": {i: s for i, s in state["state"].items() if i in moved_ids},
             "param_groups": moved})


def merge_optimizer_state(rest: dict, part: dict) -> dict:
    """Inverse of `split_optimizer_state` (the named group was built last, so it goes last)."""
    return {"state": {**rest["state"], **part["state"]},
            "param_groups": list(rest["param_groups"]) + list(part["param_groups"])}


def save_projection(path, projection: nn.Module, optimizer_part: dict, *, stage: str, it: int,
                    checkpoint: str) -> str:
    """Write projection.pt: the projection's weights and its optimizer group (hyperparameters and
    momentum buffer), paired by name and iteration with the student checkpoint of the same save.

    Written atomically (same-directory temp file + os.replace), so a kill mid-write keeps the previous
    pair's file intact; `restore_projection` refuses a file that does not pair with its checkpoint.
    """
    path = Path(path)
    payload = {"format": PROJECTION_FORMAT, "stage": stage, "iter": int(it), "checkpoint": checkpoint,
               "projection_state_dict": projection.state_dict(), "optimizer_state": optimizer_part,
               "in_channels": projection.in_channels, "out_channels": projection.out_channels,
               "params": sum(p.numel() for p in projection.parameters()), "training_only": True}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return str(path)


def restore_projection(path, projection: nn.Module, optimizer, *, student_optimizer_state: dict,
                       checkpoint=None) -> dict:
    """Load projection.pt back into `projection` (strict) and, merged with the student checkpoint's
    optimizer state, into `optimizer` (built with the same groups as the run that saved it).

    `checkpoint`, when given, is the student checkpoint path or name the file must pair with.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != PROJECTION_FORMAT:
        raise ValueError(f"{path}: format {payload.get('format')!r} != {PROJECTION_FORMAT!r}")
    if checkpoint is not None and payload.get("checkpoint") != Path(checkpoint).name:
        raise ValueError(f"{path} pairs with {payload.get('checkpoint')!r}, not "
                         f"{Path(checkpoint).name!r}")
    projection.load_state_dict(payload["projection_state_dict"], strict=True)
    optimizer.load_state_dict(merge_optimizer_state(student_optimizer_state,
                                                    payload["optimizer_state"]))
    return {"file": str(path), "stage": payload["stage"], "iter": payload["iter"],
            "checkpoint": payload["checkpoint"]}
