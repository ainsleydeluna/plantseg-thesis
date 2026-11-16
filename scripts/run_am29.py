#!/usr/bin/env python3
"""AM-29's QAT runner (lane L-AM29, P1): E6's QAT recipe of record with the allow-list's factor, rate and seed.

The arm is a flag, but only the nine arms of src/quant/am29.py ARMS exist, each with exactly its seed, factor and
rate; any other value is refused before anything is read. src/quant/am29.py holds the recipe and every gate.

    python -B scripts/run_am29.py --arm T1 --seed 42 --factor 0.5 --lr 3e-4 --real-run --confirm-real-run \\
        --source-run-dir <E3-R2 s42 run dir> --expect-source-sha256 da70d82f... --teacher-sha256 da70d82f... \\
        --expect-head <40 hex> --grad-clip-norm 5.0 --clip-selection <clip_selection.json> \\
        --clip-selection-sha256 c433145a... --lambda-selection <rel> --lambda-selection-sha256 <hex> \\
        --alpha-selection <rel> --alpha-selection-sha256 <hex> --num-workers 12 --out-dir <fresh dir>

Exit codes: 0 done, 1 STOP, 2 refused, 3 aborted or incomplete, 4 error.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant.am29 import main as _main  # noqa: E402


def main(argv=None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
