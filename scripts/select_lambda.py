#!/usr/bin/env python3
"""Select lambda_logit from the five E2 seed-42 sweep runs (AM-2 / DL-06 with AM-7a and AM-19 item 2; lanes
L-AM16-ALPHA, L-KD-HARDEN and K2).

    python scripts/select_lambda.py --runs <E2 run dir> [<E2 run dir> ...]
                                    [--out reports/derived/lambda_selection.json]
                                    [--rules configs/sweep_rules.json]
                                    [--decision-record reports/derived/lambda_decision_record.json]
                                    [--write-decision-record <path>]

Rule (configs/sweep_rules.json 'lambda_logit', checked against configs/distill.py and the rule pins of
src/training/sweep_select.py): each candidate's value is the best VAL all-class mIoU of its run_end record
(strict >, earliest tie, as E1); the highest wins; candidates within 0.5 pp of the best are tied and the
tie goes to the smallest lambda; a winner at 0.25 or 4 is flagged as a boundary result and the grid is not
extended. The winning run is E2 seed 42. A candidate is a real run of the sweep, launched through the
committed launch log (src/training/sweep_select.py lists the files read); nothing else is opened, and
TEST never is. Committed inputs are read at HEAD of the repository: the rules, the schedule
(configs/kd_schedule.json), the launch log (reports/kd_launch_log.jsonl), the decision log and the
decision record; the code that runs is committed and equals the sweep's code pin.

AM-7a: a non-default candidate (lambda 0.25, 0.5, 2 or 4) whose run ends in the trainer's own run_abort
record for a student divergence (AM-7 (a)/(b)) is excluded from the selection and never relaunched; its
directory is still a required input. A diverged default candidate (lambda 1) keeps AM-7 in full: refused.

AM-19 item 2: the lambda decision date is 2026-10-19 under Schedule T (2026-12-07 under R); it ends at
24:00 Asia/Manila. Before its end the selection waits until every candidate has finished or diverged,
and then runs as before (item 2(f)). After its end a non-default candidate that has neither is cut, with
its reason (one repeat of a stop that was on course is waited for); the default is never cut and is
waited for (item 2(d)). A selection with a cut candidate needs the committed decision record that
--write-decision-record derives (reviewed, committed and pushed, its sha256 entered in the decision log);
the selection derives every status again and refuses a record that differs. The rule then runs among the
finished candidates; a winner at an edge of the finished set next to a diverged or cut value is a boundary
result, and edge_removed names each such value with its rule.

Exit 0: the selection file (or, with --write-decision-record, the record) is written. Exit 2: refused (a
malformed, unreadable, uncommitted or inconsistent input, a launch-log disagreement, a repeat rule, an
existing output file, an AM-7a refusal: default_candidate_diverged, run_aborted_other,
abort_record_invalid, or item 10b's recipe_mismatch; default_aborted_twice; before the end of the decision
date a single run_end with failed checks or other abort, unless it waits as below, after it for the
default). Exit 3: shortfall_am19_item2: a grid value with neither a finished nor a diverged run while the
selection waits (before the end of the decision date; the default candidate; an on-course repeat), a
non-default value whose last two attempts ended the same way, before the end of the date (it is cut at the
date; K2 interpretation 13), a non-default value whose single failed check or other abort follows an
attempt that is incomplete and could still end the same way (before the end of the date; K2
interpretation 15), a value with an attempt whose launched directory is not supplied or whose launch line
has no outcome yet (before the end of the date, once no refusal applies; PL-14), or a cut without a
committed decision record. Exit 4: an unexpected error. Only exit 0 writes a file.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.training.sweep_select as ss  # noqa: E402
from src.training.sweep_select import (LAMBDA_SELECTION_REL, SHORTFALL_CODES,  # noqa: E402
                                       UNEXPECTED_ERROR_EXIT, SelectionRefused, apply_rule,
                                       derive_sweep, selection_fields, settle, sha256_file,
                                       write_decision_record, write_selection)

KEY = "lambda_logit"


def select(runs, out: Path, rules_path=None, decision_record=None, write_record=None) -> dict:
    state = derive_sweep(KEY, runs, script_path=Path(__file__), rules=rules_path, decision_record=decision_record)
    if write_record is not None:
        sha, rec = write_decision_record(state, Path(write_record))
        return {"decision_record_written": str(write_record), "sha256": sha, "record": rec}
    settled = settle(state, decision_record)
    sweep = state["sweep"]
    band = float(sweep["band"]["value"])
    res = apply_rule(settled["finished"], sweep, band, "lambda", diverged=settled["diverged"], cut=settled["cut"])
    am19 = selection_fields(state, settled, res)
    rec = lambda c: {"lambda": c["value"], "run_id": c["run_id"], "best_val": c["best_val"],  # noqa: E731
                     "ckpt_sha256": c["ckpt_sha256"], "checkpoint_present": c["checkpoint_present"]}
    fin = sorted(settled["finished"], key=lambda c: c["value"])
    doc = {"format": "lambda_selection/1", "rule": sweep["source"],
           "amendments": ["AM-2", "AM-7a"] + (["AM-19 item 2"] if am19["n_cut"] else []),
           "rules_file": {"path": state["rules_rel"], "sha256": sha256_file(state["src"].root / state["rules_rel"])},
           "candidates": [rec(c) for c in fin],
           "band": band, "rule_trace": res["rule_trace"], "winner": rec(res["winner"]),
           "tie": res["tie"], "tied": res["tied"], "boundary": res["boundary"],
           "boundary_kind": res["boundary_kind"], "winner_is": sweep["winner_is"],
           "excluded_am7a": res["excluded_am7a"], "n_finished": res["n_finished"],
           "n_diverged": res["n_diverged"],
           "inputs": [{"run_dir": c["run_dir"], "checkpoint": c["checkpoint"],
                       "best_json_sha256": c["best_json_sha256"],
                       "run_meta_sha256": c["run_meta_sha256"]} for c in fin],
           "generated_utc": datetime.fromtimestamp(ss.now_utc(), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script": {"path": "scripts/select_lambda.py", "sha256": sha256_file(Path(__file__))},
           **am19}
    write_selection(out, doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select lambda_logit (AM-2 / DL-06; AM-19 item 2) from the E2 sweep.")
    ap.add_argument("--runs", nargs="+", required=True, help="the E2 seed-42 run (checkpoint) dirs")
    ap.add_argument("--out", default=None, help=f"default: {LAMBDA_SELECTION_REL} in the repository")
    ap.add_argument("--rules", default=None, help="a committed rule file (default configs/sweep_rules.json)")
    ap.add_argument("--decision-record", default=None,
                    help="a committed decision record (default reports/derived/lambda_decision_record.json, "
                         "read when present)")
    ap.add_argument("--write-decision-record", default=None, metavar="PATH",
                    help="derive the decision record after the end of the decision date and write it to PATH "
                         "(no selection)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else ss.GIT_ROOT / LAMBDA_SELECTION_REL
    try:
        doc = select(a.runs, out, a.rules, a.decision_record, a.write_decision_record)
    except SelectionRefused as e:
        print(f"REFUSED [{e.code}]: {e}", file=sys.stderr)
        print(f"RESULT: REFUSED ({e.code})")
        return 3 if e.code.startswith("shortfall") or e.code in SHORTFALL_CODES else 2
    except Exception as e:  # noqa: BLE001 — never a selection, never a bare traceback
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT: ERROR (no selection was made)")
        return UNEXPECTED_ERROR_EXIT
    if "decision_record_written" in doc:
        cut = [f"{v['value']:g}" for v in doc["record"]["values"] if v["status"] == "cut"]
        print(f"RESULT: DECISION RECORD written {doc['decision_record_written']} sha256 {doc['sha256']} (cut "
              f"[{', '.join(cut)}]; review it, commit and push it within 24 hours and enter its sha256 in the "
              "decision log: AM-19 item 2(h))")
        return 0
    for line in doc["rule_trace"]:
        print(f"  {line}")
    w = doc["winner"]
    div = ", ".join(f"{x['value']:g}" for x in doc["excluded_am7a"])
    cut = ", ".join(f"{x['value']:g}" for x in doc["cut_am19"])
    note = "" if w["checkpoint_present"] else "; the winner's checkpoint is absent (ckpt_sha256 null)"
    print(f"RESULT: SELECTED lambda_logit = {w['lambda']:g} (run {w['run_id']}, tie={doc['tie']}, "
          f"boundary={doc['boundary']}/{doc['boundary_kind']}, diverged=[{div}]) cut=[{cut}]{note} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
