#!/usr/bin/env python3
"""E4 — static INT8 PTQ from the official E1 FP32 checkpoint. This entry point makes no E4 of record.

Ruling Q1-R2 (L-CKPT-GUARD Q-part): `src/quant/runner.py` refuses every E4 run started here, whatever its
flags (`old_runner_real_run`, exit 2), before anything is read, loaded or written. The E4 of record is made
by `scripts/run_ptq.py`, which binds the run to its E1 parent and writes `e4_ptq_record.json`.

The stage is PINNED below and is deliberately NOT a command-line option, so this entry point can
never be redirected to another experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant.runner import main as _main  # noqa: E402

STAGE = "e4"


def main(argv=None) -> int:
    return _main(argv, stage_key=STAGE)


if __name__ == "__main__":
    raise SystemExit(main())
