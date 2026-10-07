#!/usr/bin/env python3
"""Lane 6 (L-AM4 + L-AM1q): select the U4 QAT clipping threshold (AM-4a item 3, AM-21 item 3).

    python -B scripts/select_clip.py --candidate R1 E1 --expect-telemetry-sha256 S1 \
        --candidate R2 E2 --expect-telemetry-sha256 S2 --out PATH

The two E5 seed-42 pilot runs (clip 1.0 and 5.0), each with its telemetry sha256 as recorded on the pod, in
the order of the candidates. A run rejected under AM-21 item 3(a), read from its telemetry and its epoch
checkpoints, loses: its conversion and scoring of record are reported (present, missing, stopped or failed;
its scores select nothing) and never required (item 3(d)); with both rejected there is no winner (item 3(c)).
A run not rejected needs its stored qat_selection.json, which the recomputed epoch selection must equal; the two
runs' records must match in every run_meta key but the clip and the run and host identity, and in every batch
fingerprint. The higher selected value wins, and a tie within 0.1 pp, exact (|Fraction(a) - Fraction(b)| <=
1/1000), goes to 5.0. The loser is retained. Once both candidates are read and compared, each rejected run gets
one REPORT line before the RESULT line, on a selection and on the no-winner refusal.
Written once, on exit 0 only. Exit codes: 0 selected, 1 STOP, 2 refused, 3 incomplete, 4 error.
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


def _print_report(report: list[str]) -> None:
    for line in report:
        print(f"REPORT: {line}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", nargs=2, action="append", metavar=("RUN_DIR", "EVAL_DIR"), required=True)
    ap.add_argument("--expect-telemetry-sha256", action="append", default=[], metavar="SHA256",
                    help="one per --candidate, in their order: the telemetry sha256 recorded on the pod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-smoke-inputs", action="store_true",
                    help="smokes only: accept smoke runs and capped scores; recorded in the output")
    args = ap.parse_args(argv)
    out = Path(args.out)
    report: list[str] = []                                   # one line per rejected run (AM-21 item 3(d))
    try:
        A.refuse_test_path(out, "out")
        if out.exists():
            raise QATRefused("output_exists", f"{out} exists; a selection is written once")
        rules, rules_sha = S.load_rules()
        sel = S.clip_selection([tuple(c) for c in args.candidate], expect_telemetry_sha256=args.expect_telemetry_sha256,
                               rules=rules, rules_sha256=rules_sha, allow_smoke_inputs=args.allow_smoke_inputs,
                               report=report)
        from src.quant.ptq import write_exclusive
        out.parent.mkdir(parents=True, exist_ok=True)
        write_exclusive({out: A.json_bytes(sel)})
    except QATRefused as e:
        _print_report(report)
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return EXIT_REFUSED
    except A.QATIncomplete as e:
        _print_report(report)
        print(f"RESULT: INCOMPLETE [{e.code}] -- {e}. Nothing was written.")
        return EXIT_ABORTED
    except QATStop as e:
        _print_report(report)
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    except Exception as e:                                   # noqa: BLE001 -- reported, exit 4
        _print_report(report)
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. Nothing was written.")
        return EXIT_ERROR
    for line in sel["rule_trace"]:
        print(f"  {line}")
    w, lo = sel["winner"], sel["loser"]
    tail = f"; clip {lo['clip_norm']} {S.REJECTED_NONFINITE}" if lo["rejected"] else ""
    print(f"[select_clip] winner clip {w['clip_norm']} (e{w['epoch']:02d}, {w['value']!r}); loser retained -> {out}")
    _print_report(report)
    print(f"RESULT: CLIP SELECTED {w['clip_norm']} (tie {str(sel['tie']).lower()}{tail})")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
