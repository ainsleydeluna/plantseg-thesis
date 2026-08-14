"""Non-invasive feature taps on the E1 student, for E3 channel-wise distillation.

The student (`src/models/student.py`) is shared with E1 and MUST NOT change. This module therefore
captures the two tensors E3 needs via forward hooks instead of editing the model:

  * `c5`          — the OS16 backbone tap (`features[high_tap]`, 160 ch) fed to the CWD projection.
  * `head_logits` — the LR-ASPP head output BEFORE the final float upsample (OS8, num_classes ch),
                    which is the map the CWD logit term is computed on by default (Shu 2021 applies
                    channel-wise KD to the head's native output map, not to an upsampled copy).

Both are plain forward hooks: registering them does not alter the student's numerics, and removing
them restores the exact E1 forward path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StudentTaps:
    """Context manager capturing the student's OS16 C5 tensor and native head logits.

    Usage:
        with StudentTaps(student) as taps:
            logits = student(img)          # ordinary E1 forward, unchanged
            c5 = taps.c5                   # [B, 160, H/16, W/16]
            head_logits = taps.head_logits # [B, num_classes, H/8, W/8]

    The captured tensors stay attached to the autograd graph, so gradients flow back into the
    student exactly as they would through `logits`.
    """

    def __init__(self, student: nn.Module):
        self._student = student
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.c5: torch.Tensor | None = None
        self.head_logits: torch.Tensor | None = None

    def __enter__(self) -> "StudentTaps":
        def _c5_hook(_module, _inp, out):
            self.c5 = out

        def _head_hook(_module, _inp, out):
            self.head_logits = out

        high_tap = getattr(self._student, "high_tap")
        self._handles.append(self._student.features[high_tap].register_forward_hook(_c5_hook))
        self._handles.append(self._student.head.register_forward_hook(_head_hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return None

    def require(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (c5, head_logits), raising if the student was not run inside this context."""
        if self.c5 is None or self.head_logits is None:
            raise RuntimeError(
                "StudentTaps captured nothing — call the student INSIDE the `with StudentTaps(...)` "
                "block before reading .c5 / .head_logits")
        return self.c5, self.head_logits
