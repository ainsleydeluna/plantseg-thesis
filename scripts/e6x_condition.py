#!/usr/bin/env python3
"""L-AM23Q B-1: the E6-X condition of AM-23 item 3(b). Writes e6x_condition/1 once and prints its decision-log text.

  python -B scripts/e6x_condition.py record --arm R1|R2 --run-dir DIR \\
      --lambda-selection reports/derived/lambda_selection.json --lambda-selection-sha256 HEX \\
      --alpha-selection reports/derived/alpha_selection.json --alpha-selection-sha256 HEX \\
      --expect-head COMMIT --out OUT/e6x_condition_<arm>.json
      E3-X's run directory and the records at HEAD of this clone, through gates G0-G10; the record is created
      exclusively at --out (outside the repository) and read back; then its sha256, the decision-log text (one line)
      and RESULT: E6-<X> CONDITION yes|no.
  python -B scripts/e6x_condition.py print-dl --record PATH --record-sha256 HEX
      The same decision-log text, from a record and its expected sha256 (the record re-checked, its verdict
      recomputed from its value).

Exit: 0 written (yes or no) or printed; 1 STOP; 2 refused; 3 E3-X incomplete (no run_end); 4 error. Nothing is
written but the record, last. The gates, the record and the readers: src/quant/e6x_condition.py.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant import e6x_condition as C  # noqa: E402
from src.quant.qat import QATRefused  # noqa: E402

RECORD_OPTIONS = ("--arm", "--run-dir", "--lambda-selection", "--lambda-selection-sha256", "--alpha-selection",
                  "--alpha-selection-sha256", "--expect-head", "--out")
DL_HEADER = "decision-log text (one line; the orchestrator enters it verbatim):"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="e6x_condition.py", description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="command")
    rec = sub.add_parser("record", allow_abbrev=False, help="decide the condition and write e6x_condition/1")
    for option in RECORD_OPTIONS:
        rec.add_argument(option, default=None)
    pdl = sub.add_parser("print-dl", allow_abbrev=False, help="print a record's decision-log text")
    pdl.add_argument("--record", default=None)
    pdl.add_argument("--record-sha256", default=None)
    return ap


def cmd_record(args) -> int:
    rec, sha, out, text = C.record_condition(args)
    print(f"{C.FORMAT} written: {out}")
    print(f"sha256: {sha}")
    print(DL_HEADER)
    print(text)
    print(C.result_line(rec))
    return C.EXIT_WRITTEN


def cmd_print_dl(args) -> int:
    C.check_flags(args, C.PRINT_FLAGS)
    C.check_names(args, ("record",))
    rec, sha = C.read_record_file(args.record, args.record_sha256)
    print(DL_HEADER)
    print(C.dl_text(rec, sha))
    print(f"RESULT: DL TEXT ({rec['stage']}, verdict {rec['verdict']}, record {sha[:12]})")
    return C.EXIT_WRITTEN


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        print("RESULT: REFUSED [missing_flags] -- a command is required: record or print-dl. Nothing was written.")
        return C.EXIT_REFUSED
    try:
        return cmd_record(args) if args.command == "record" else cmd_print_dl(args)
    except C.ConditionIncomplete as e:
        print(f"RESULT: INCOMPLETE [{e.code}] -- {e}. Nothing was written.")
        return C.EXIT_INCOMPLETE
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return C.EXIT_REFUSED
    except C.ConditionStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return C.EXIT_STOP
    except Exception as e:  # noqa: BLE001 -- reported, exit 4
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. Nothing was written.")
        return C.EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
