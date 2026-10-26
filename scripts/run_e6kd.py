#!/usr/bin/env python3
"""E6-KD — INT8 QAT from the projection-free E3 student with the teacher of record in the loop (AM-3; DL-39, DL-51).

The stage is PINNED below and is deliberately NOT a command-line option, so this entry point can
never be redirected to another experiment. `src/quant/runner.py` main hands method qat_kd to
`src/quant/qat_kd.py`, which holds E6-KD's recipe and every launch gate (lane L-AM3, B-2).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant.runner import main as _main  # noqa: E402

STAGE = "e6-kd"


def main(argv=None) -> int:
    return _main(argv, stage_key=STAGE)


if __name__ == "__main__":
    raise SystemExit(main())
