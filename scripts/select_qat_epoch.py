#!/usr/bin/env python3
"""Lane 6 (L-AM4 + L-AM1q): select one QAT run's epoch of record (AM-4a item 2).

    python -B scripts/select_qat_epoch.py --run-dir R --eval-dir E --expect-telemetry-sha256 H [--out PATH]

The highest converted QNNPACK VAL all-class mIoU, strict >, a tie keeps the earlier epoch; an epoch whose
state is non-finite is excluded (AM-19 item 3(a)); the fake-quant VAL score never selects. The rules are
configs/qat_selection_rules.json (src/quant/qat_select.py). The output defaults to E/qat_selection.json and
is written once, on exit 0 only. Exit codes: 0 selected, 1 STOP, 2 refused, 3 incomplete, 4 error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_select as S  # noqa: E402
from src.quant.qat import (EXIT_ABORTED, EXIT_ERROR, EXIT_OK, EXIT_REFUSED, EXIT_STOP, QATRefused,  # noqa: E402
                           QATStop)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--expect-telemetry-sha256", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-smoke-inputs", action="store_true",
                    help="smokes only: accept smoke runs and capped scores; recorded in the output")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.eval_dir) / "qat_selection.json"
    try:
        A.refuse_test_path(args.eval_dir, "eval_dir")
        A.refuse_test_path(out, "out")
        if out.exists():
            raise QATRefused("output_exists", f"{out} exists; a selection is written once")
        rules, rules_sha = S.load_rules()
        sel = S.epoch_selection(args.run_dir, args.eval_dir, expect_telemetry_sha256=args.expect_telemetry_sha256,
                                rules=rules, rules_sha256=rules_sha, allow_smoke_inputs=args.allow_smoke_inputs)
        from src.quant.ptq import write_exclusive
        write_exclusive({out: A.json_bytes(sel)})
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return EXIT_REFUSED
    except A.QATIncomplete as e:
        print(f"RESULT: INCOMPLETE [{e.code}] -- {e}. Nothing was written.")
        return EXIT_ABORTED
    except QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    except Exception as e:                                   # noqa: BLE001 -- reported, exit 4
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. Nothing was written.")
        return EXIT_ERROR
    for line in sel["rule_trace"]:
        print(f"  {line}")
    w = sel["winner"]
    excluded = sel["excluded_epochs"]
    tail = f"; excluded: {' '.join(f'e{e:02d}' for e in excluded)}" if excluded else ""
    print(f"[select] e{w['epoch']:02d}: converted VAL all-class mIoU {w['value']!r} -> {out}")
    print(f"RESULT: SELECTED epoch {w['epoch']:02d} ({sel['stage']}, seed {sel['seed']}, "
          f"clip {sel['clip_norm']}{tail})")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
