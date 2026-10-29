#!/usr/bin/env python3
"""Lane L-AM3, B-1: the AM-3 trigger of E6-KD (src/quant/e6kd_trigger.py; docs/lane_specs/part2.md lane 7).

    python -B scripts/e6kd_trigger.py score  IDENTITY --trigger-eval-dir NEWDIR --host-label TEXT
    python -B scripts/e6kd_trigger.py decide IDENTITY --trigger-eval-dir DIR --host-label TEXT --out NEWFILE

    IDENTITY = --e3-run-dir DIR --e3-checkpoint-sha256 H
               --e6-run-dir DIR --e6-eval-dir DIR --e6-telemetry-sha256 H --e6-selection-sha256 H
               --lambda-selection REL --lambda-selection-sha256 H --alpha-selection REL --alpha-selection-sha256 H

`score` checks every input, then runs the two literal evaluator commands (E3's FP32 checkpoint, E6's QNNPACK
artifact of record; VAL, canvas protocol, CPU, batch size 1) and writes trigger_eval.json into the trigger eval dir.
`decide` checks every input and that directory again, then writes the trigger record (format e6kd_trigger/1) to --out
once: TRIGGERED when the decimal drop is strictly greater than 1.0 percentage point. Both refuse a path that holds a
name SL-1 forbids before any filesystem call. Every subcommand ends with one RESULT line. Exit codes: 0 done, 1 STOP,
2 refused (nothing written), 3 incomplete (an evaluator failure writes trigger_STOP.json, which spends the trigger
eval dir), 4 error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant import e6kd_trigger as T  # noqa: E402
from src.quant.qat import (EXIT_ABORTED, EXIT_ERROR, EXIT_OK, EXIT_REFUSED, EXIT_STOP, QATRefused,  # noqa: E402
                           QATStop)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("score", "check every input, then run the two literal evaluator commands"),
                        ("decide", "check every input and the trigger eval dir, then write the trigger record")):
        p = sub.add_parser(name, help=help_)
        for flag in ("--e3-run-dir", "--e3-checkpoint-sha256", "--e6-run-dir", "--e6-eval-dir",
                     "--e6-telemetry-sha256", "--e6-selection-sha256", "--lambda-selection",
                     "--lambda-selection-sha256", "--alpha-selection", "--alpha-selection-sha256",
                     "--trigger-eval-dir", "--host-label"):
            p.add_argument(flag, required=True)
        if name == "decide":
            p.add_argument("--out", required=True)
        p.add_argument("--allow-smoke-inputs", action="store_true",
                       help="smokes only: accept a smoke run and capped evaluator rows; recorded in every output")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    inp = T.inputs_from_args(args)
    try:
        if args.cmd == "score":
            out = T.score(inp)
        else:
            doc = T.decide(inp)
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return EXIT_REFUSED
    except T.QATIncomplete as e:
        print(f"RESULT: INCOMPLETE [{e.code}] -- {e}")
        return EXIT_ABORTED
    except QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    except Exception as e:                                   # noqa: BLE001 -- reported, exit 4
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}")
        return EXIT_ERROR
    if args.cmd == "score":
        what = "smoke inputs" if out["smoke_inputs"] else "record"
        print(f"[score] {out['trigger_eval_dir']}")
        print(f"RESULT: SCORED 2/2 ({what})")
        return EXIT_OK
    for line in doc["rule_trace"]:
        print(f"  {line}")
    print(f"[decide] {args.out}")
    print(T.result_line(doc))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
