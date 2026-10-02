#!/usr/bin/env python3
"""Select lambda_logit from the five E2 seed-42 sweep runs (AM-2 / DL-06 with AM-7a; lanes L-AM16-ALPHA
and L-KD-HARDEN).

    python scripts/select_lambda.py --runs <E2 run dir> [<E2 run dir> ...]
                                    [--out reports/derived/lambda_selection.json]

Rule (configs/sweep_rules.json 'lambda_logit', checked against configs/distill.py): each candidate's
value is its best-checkpoint VAL all-class mIoU (strict >, earliest tie, as E1); the highest wins;
candidates within 0.5 pp of the best are tied and the tie goes to the smallest lambda; a winner at
0.25 or 4 is flagged as a boundary result and the grid is not extended. The winning run is E2 seed 42.
A candidate is a real run of the sweep (src/training/sweep_select.py lists the files read); nothing
else is opened, and TEST never is.

AM-7a: a non-default candidate (lambda 0.25, 0.5, 2 or 4) whose run ends in the trainer's own run_abort
record for a student divergence (AM-7 (a)/(b)) is excluded from the selection and never relaunched; its
directory is still a required input. The rule runs over the finished candidates (a sole finished
candidate wins with no band and no tie); a winner at an edge of the finished set next to a diverged
value is a boundary result too. A diverged default candidate (lambda 1) keeps AM-7 in full: refused.

Exit 0: the selection file is written. Exit 2: refused (a malformed, unreadable or inconsistent input,
a finished run whose checkpoint is absent, an existing selection file, an AM-7a refusal:
default_candidate_diverged, run_aborted_other, abort_record_invalid, or item 10b's recipe_mismatch). Exit 3:
shortfall, a grid value with neither a finished nor a diverged run (a run directory absent, its
run_meta/telemetry not written yet, a finished run without best.json, or neither a run_end nor a
run_abort record). AM-17 item 9 (at the TEST freeze, select among the finished candidates by the AM-2
rule and disclose the shortfall) is NOT implemented: the refusal names it, and no selection is made.
Exit 4: an unexpected error. Only exit 0 writes a file.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training.sweep_select import (RULES_PATH, UNEXPECTED_ERROR_EXIT,  # noqa: E402
                                       SelectionRefused, apply_rule, collect_candidates,
                                       load_rules, missing_grid_values, sha256_file,
                                       write_selection)

KEY = "lambda_logit"


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if REPO in p.parents else str(p)


def select(runs, out: Path, rules_path: Path = RULES_PATH) -> dict:
    rules = load_rules(rules_path)
    sweep = rules[KEY]
    finished, diverged, shortfall = collect_candidates(runs, sweep, KEY)
    missing = missing_grid_values(finished + diverged, sweep)
    if shortfall or missing:
        absent = ["{} ({})".format(s["run_dir"], s["code"]) for s in shortfall]
        raise SelectionRefused(
            "shortfall_am17_item9",
            f"lambda shortfall: {len(finished)} of {len(sweep['grid'])} candidates finished and "
            f"{len(diverged)} diverged; grid values without a finished or diverged run: {missing}; "
            f"absent or unfinished: {absent} (diverged candidates are not shortfalls: pass their "
            "directories). AM-17 item 9 (at the TEST freeze, select among the finished candidates by the "
            "AM-2 rule and disclose the shortfall) is not implemented here: no selection was made.")
    band = float(sweep["band"]["value"])
    res = apply_rule(finished, sweep, band, "lambda", diverged=diverged)
    rec = lambda c: {"lambda": c["value"], "run_id": c["run_id"], "best_val": c["best_val"],  # noqa: E731
                     "ckpt_sha256": c["ckpt_sha256"]}
    doc = {"format": "lambda_selection/1", "rule": sweep["source"], "amendments": ["AM-2", "AM-7a"],
           "rules_file": {"path": _rel(rules_path), "sha256": sha256_file(Path(rules_path))},
           "candidates": [rec(c) for c in sorted(finished, key=lambda c: c["value"])],
           "band": band, "rule_trace": res["rule_trace"], "winner": rec(res["winner"]),
           "tie": res["tie"], "tied": res["tied"], "boundary": res["boundary"],
           "boundary_kind": res["boundary_kind"], "winner_is": sweep["winner_is"],
           "excluded_am7a": res["excluded_am7a"], "n_finished": res["n_finished"],
           "n_diverged": res["n_diverged"],
           "inputs": [{"run_dir": c["run_dir"], "checkpoint": c["checkpoint"],
                       "best_json_sha256": c["best_json_sha256"],
                       "run_meta_sha256": c["run_meta_sha256"]} for c in finished],
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script": {"path": "scripts/select_lambda.py", "sha256": sha256_file(Path(__file__))}}
    write_selection(out, doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select lambda_logit (AM-2 / DL-06) from the E2 sweep.")
    ap.add_argument("--runs", nargs="+", required=True, help="the E2 seed-42 run (checkpoint) dirs")
    ap.add_argument("--out", default=str(REPO / "reports" / "derived" / "lambda_selection.json"))
    ap.add_argument("--rules", default=str(RULES_PATH))
    a = ap.parse_args(argv)
    try:
        doc = select(a.runs, Path(a.out), Path(a.rules))
    except SelectionRefused as e:
        print(f"REFUSED [{e.code}]: {e}", file=sys.stderr)
        print(f"RESULT: REFUSED ({e.code})")
        return 3 if e.code.startswith("shortfall") else 2
    except Exception as e:  # noqa: BLE001 — never a selection, never a bare traceback
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT: ERROR (no selection was made)")
        return UNEXPECTED_ERROR_EXIT
    for line in doc["rule_trace"]:
        print(f"  {line}")
    w = doc["winner"]
    div = ", ".join(f"{x['value']:g}" for x in doc["excluded_am7a"])
    print(f"RESULT: SELECTED lambda_logit = {w['lambda']:g} (run {w['run_id']}, tie={doc['tie']}, "
          f"boundary={doc['boundary']}/{doc['boundary_kind']}, diverged=[{div}]) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
