#!/usr/bin/env python3
"""E6-R1 — INT8 QAT from the projection-free E3-R1 student (AM-23 items 3(a), 3(b) and 4; descriptive).

It launches only when the condition record of AM-23 item 3(b) says yes for E3-R1's run and its decision-log
row is RECORDED. The stage is PINNED below and is deliberately NOT a command-line option, so this entry
point can never be redirected to another experiment. `src/quant/runner.py` main hands the QAT stages to
`src/quant/qat.py`, which holds the AM-4/AM-4a recipe and every launch gate, AM-23's parent binding included.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant.runner import main as _main  # noqa: E402

STAGE = "e6-r1"


def main(argv=None) -> int:
    return _main(argv, stage_key=STAGE)


if __name__ == "__main__":
    raise SystemExit(main())
