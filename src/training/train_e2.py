#!/usr/bin/env python3
"""E2 entry point — FP32 student + Logit KD from the frozen SegNeXt-B teacher.

E2 = the E1 recipe (MobileNetV3-Large + LR-ASPP, FP32, ImageNet init, same preprocessing/splits,
all-class validation mIoU checkpoint selection) plus response-level Logit KD. It adds NO CWD and NO
quantization, and it starts from **E1 init, trained independently** — it does not resume from a
trained E1 checkpoint (IMPLEMENTATION_CONTRACT.md stage table).

The implementation lives in `train_distill.py`; this file exists so E2 stays an independently
identifiable, reproducible stage on the command line and in logs.

    python src/training/train_e2.py --dry-run
    python src/training/train_e2.py --real-run --confirm-real-run \
        --teacher-ckpt /workspace/teacher/segnext_b_plantseg.pth \
        --lambda-logit <value from the validation sweep> \
        --ckpt-dir /workspace/e2_ckpts
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training.train_distill import main as _main  # noqa: E402

STAGE = "e2"


def main(argv=None) -> int:
    return _main(argv, stage_default=STAGE)


if __name__ == "__main__":
    raise SystemExit(main())
