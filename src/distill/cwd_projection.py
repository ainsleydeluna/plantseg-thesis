"""Training-only CWD projection head for E3 feature distillation (Blocker B12 — stub).

Channel-Wise KD (Shu 2021) aligns the student's stride-16 C5 feature to the teacher's matching
channel dimension via a single 1x1 convolution applied to the student's C5 during E3 training.

  student C5 (OS16) : 160 channels  -- B10-verified tap (features index 15); see
                                        reports/student_forward_smoke.md.
  teacher stride-16 : 320 channels  -- LOCKED per docs/IMPLEMENTATION_CONTRACT.md B3 / ch3
                                        (MSCAN-B Stage-3, C=320). Empirical teacher-side confirmation
                                        is the Table 3.1 forward-hook check, run when the SegNeXt-B
                                        teacher is built (out of scope here — no teacher code yet).

*** TRAINING-ONLY. *** This projection exists ONLY during E3 training. It MUST be removed from the
E3 checkpoint via state_dict editing BEFORE E6/E7 (quantization observer insertion / PTQ / QAT export),
so the evaluated and quantized models never contain it. This module is NOT the CWD loss — the full
channel-wise feature distillation is deferred to the E3 blocker.
"""

from __future__ import annotations

import torch.nn as nn

STUDENT_C5_CH = 160       # B10-verified (student OS16/C5 tap, features index 15)
TEACHER_STRIDE16_CH = 320  # LOCKED: contract B3 / ch3 (MSCAN-B stride-16 Stage-3 channel count)


def build_cwd_projection(in_ch: int = STUDENT_C5_CH, out_ch: int = TEACHER_STRIDE16_CH) -> nn.Conv2d:
    """Single bias-free 1x1 conv mapping student C5 (in_ch) -> teacher channel dim (out_ch).

    Training-only (E3); removed from the checkpoint before E6/E7 — see module docstring.
    """
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
