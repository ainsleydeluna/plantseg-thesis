#!/usr/bin/env python3
"""E3 entry point — FP32 student + Logit KD + Channel-Wise KD (the proposed FP32 student).

E3 = the E1 recipe plus the SAME Logit KD as E2 (reused unchanged, contract B3) plus Shu-2021
Channel-Wise KD on the stride-16 C5 feature map and the head logit map:

    L = L_CE + L_Dice + lambda_logit*L_LogitKD + 50*L_CWD_feat + 3*L_CWD_logit

Like E2 it starts from **E1 init, trained independently** — it does NOT resume from the trained E2
checkpoint. The 1x1 CWD projection (student 160-ch C5 -> teacher 320-ch) is TRAINING-ONLY: it is
stored under a separate checkpoint key so `model_state_dict` is already the projection-free student
that E6 (QAT) and E7 (PTQ) consume. E3 itself stays FP32 with no QAT/PTQ.

    python src/training/train_e3.py --dry-run
    python src/training/train_e3.py --real-run --confirm-real-run \
        --teacher-ckpt /workspace/teacher/segnext_b_plantseg.pth \
        --lambda-logit <same value selected for E2> \
        --ckpt-dir /workspace/e3_ckpts
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training.train_distill import main as _main  # noqa: E402

STAGE = "e3"


def main(argv=None) -> int:
    return _main(argv, stage_default=STAGE)


if __name__ == "__main__":
    raise SystemExit(main())
