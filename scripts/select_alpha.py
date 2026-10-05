#!/usr/bin/env python3
"""Select alpha_cwd from the three E3 seed-42 sweep runs (AM-16 item 2 / DL-27 with AM-7a and AM-19 item 2;
lanes L-AM16-ALPHA, L-KD-HARDEN and K2).

    python scripts/select_alpha.py --runs <E3 run dir> <E3 run dir> <E3 run dir>
                                   [--band reports/derived/dl27_band.json]
                                   [--lambda-selection reports/derived/lambda_selection.json]
                                   [--out reports/derived/alpha_selection.json]
                                   [--rules configs/sweep_rules.json]
                                   [--decision-record reports/derived/alpha_decision_record.json]
                                   [--write-decision-record <path>]

Rule (configs/sweep_rules.json 'alpha_cwd', checked against configs/distill.py and the rule pins of
src/training/sweep_select.py): each candidate's value is the best VAL all-class mIoU of its run_end record;
the highest wins; candidates within the band of the best are tied (inclusive); a tie that contains 50 goes
to 50 (the Chapter 3 / Shu et al. default), otherwise to the smallest alpha; a winner at 25 or 100 is
flagged as a boundary result and the grid is not extended. The winning run is E3 seed 42. The runs share
one lambda_logit, equal to the winner of the committed lambda selection file (AM-16 item 2: the alpha sweep
runs after lambda is fixed). Committed inputs are read at HEAD of the repository, as select_lambda reads
them; --band and --lambda-selection name only the committed files of record.

AM-7a: a non-default candidate (alpha 25 or 100) whose run ends in the trainer's own run_abort record for
a student divergence (AM-7 (a)/(b)) is excluded from the selection and never relaunched; its directory is
still a required input. A diverged default candidate (alpha 50) keeps AM-7 in full: refused.

AM-19 item 2: the alpha decision date is the later of 2026-10-22 under Schedule T (2026-12-10 under R) and
the third day after the Asia/Manila day on which the commit that added reports/derived/lambda_selection.json
reached the remote (PL-17: from the lambda selection's last input timestamp and the alpha default's launch,
and from the committed GitHub activity record reports/derived/lambda_selection_push.json only when those
two straddle a Manila midnight). The statuses, cuts and the decision record follow select_lambda. If no
non-default alpha candidate was launched before the end of the alpha decision date, the sweep is cut:
alpha = 50 (AM-16 item 2; AM-19 item 2(g)), no selection file, and the committed decision record states the
cut: exit 5, printing the record's sha256 (exit 3 while no such record exists).

Order of the refusals (K8-2(a)): the candidates' recipe (recipe_mismatch_across_candidates), then the
shared lambda over every supplied run and every alpha launch line, finished, diverged, cut or running
(lambda_mismatch), then a diverged default (default_candidate_diverged), then the shortfall, then the rule.

Band file (the DL-27 decision-log entry, written after B66 and before the sweep):
    {"e1_best_val": {"42": <float>, "43": <float>, "44": <float>}, "s": <float>, "band": <float, optional>}
s is the n = 3 sample SD of E1's best VAL all-class mIoU; band = max(0.005, sqrt(2) * s). s is recomputed
from the three values and a recorded band is recomputed; any disagreement refuses.

Exit 0: the selection file (or, with --write-decision-record, the record) is written. Exit 2: refused (as
select_lambda; also the band or lambda selection file missing, uncommitted, malformed or changed since
the alpha default's launch, and alpha_selection.json beside a record that cuts the sweep). Exit 3:
shortfall_am19_item2. Exit 4: an unexpected error. Exit 5: the alpha sweep is cut (no selection file).
Only exit 0 writes a file.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.training.sweep_select as ss  # noqa: E402
from src.training.sweep_select import (ALPHA_SELECTION_REL, SHORTFALL_CODES, SWEEP_CUT_EXIT,  # noqa: E402
                                       UNEXPECTED_ERROR_EXIT, SelectionRefused, SweepCut, apply_rule,
                                       derive_sweep, selection_fields, settle, sha256_file,
                                       write_decision_record, write_selection)

KEY = "alpha_cwd"


def select(runs, band_path=None, out: Path | None = None, rules_path=None, lambda_selection=None,
           decision_record=None, write_record=None) -> dict:
    state = derive_sweep(KEY, runs, script_path=Path(__file__), rules=rules_path, band=band_path,
                         lambda_selection=lambda_selection, decision_record=decision_record)
    if write_record is not None:
        sha, rec = write_decision_record(state, Path(write_record))
        return {"decision_record_written": str(write_record), "sha256": sha, "record": rec}
    settled = settle(state, decision_record)
    sweep, alpha = state["sweep"], state["alpha"]
    res = apply_rule(settled["finished"], sweep, alpha["band"], "alpha", diverged=settled["diverged"],
                     cut=settled["cut"])
    am19 = selection_fields(state, settled, res)
    rec = lambda c: {"alpha": c["value"], "run_id": c["run_id"], "best_val": c["best_val"],  # noqa: E731
                     "ckpt_sha256": c["ckpt_sha256"], "checkpoint_present": c["checkpoint_present"]}
    fin = sorted(settled["finished"], key=lambda c: c["value"])
    doc = {"format": "alpha_selection/1", "rule": sweep["source"],
           "amendments": ["AM-16 item 2", "AM-7a"] + (["AM-19 item 2"] if am19["n_cut"] else []),
           "rules_file": {"path": state["rules_rel"], "sha256": sha256_file(state["src"].root / state["rules_rel"])},
           "band_file": {"path": alpha["band_rel"], "sha256": alpha["band_sha256"]},
           "lambda_selection_file": {"path": alpha["lambda_rel"], "sha256": alpha["lambda_sha256"],
                                     "lambda": alpha["lambda"]},
           "candidates": [rec(c) for c in fin],
           "s": alpha["s"], "band": alpha["band"], "rule_trace": alpha["band_trace"] + res["rule_trace"],
           "winner": rec(res["winner"]), "tie": res["tie"], "tied": res["tied"],
           "boundary": res["boundary"], "boundary_kind": res["boundary_kind"],
           "winner_is": sweep["winner_is"], "excluded_am7a": res["excluded_am7a"],
           "n_finished": res["n_finished"], "n_diverged": res["n_diverged"],
           "lambda_logit": alpha["lambda"],
           "inputs": [{"run_dir": c["run_dir"], "checkpoint": c["checkpoint"],
                       "best_json_sha256": c["best_json_sha256"],
                       "run_meta_sha256": c["run_meta_sha256"]} for c in fin],
           "generated_utc": datetime.fromtimestamp(ss.now_utc(), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script": {"path": "scripts/select_alpha.py", "sha256": sha256_file(Path(__file__))},
           **am19}
    write_selection(out, doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select alpha_cwd (AM-16 item 2 / DL-27; AM-19 item 2) from the E3 "
                                             "sweep.")
    ap.add_argument("--runs", nargs="+", required=True, help="the three E3 seed-42 run (checkpoint) dirs")
    ap.add_argument("--band", default=None, help="the committed band file of record (the rules' band.file)")
    ap.add_argument("--lambda-selection", default=None,
                    help="the committed lambda selection file (reports/derived/lambda_selection.json)")
    ap.add_argument("--out", default=None, help=f"default: {ALPHA_SELECTION_REL} in the repository")
    ap.add_argument("--rules", default=None, help="a committed rule file (default configs/sweep_rules.json)")
    ap.add_argument("--decision-record", default=None,
                    help="a committed decision record (default reports/derived/alpha_decision_record.json, "
                         "read when present)")
    ap.add_argument("--write-decision-record", default=None, metavar="PATH",
                    help="derive the decision record after the end of the decision date and write it to PATH "
                         "(no selection)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else ss.GIT_ROOT / ALPHA_SELECTION_REL
    try:
        doc = select(a.runs, a.band, out, a.rules, a.lambda_selection, a.decision_record, a.write_decision_record)
    except SweepCut as e:
        print(f"[cut] {e}", file=sys.stderr)
        print(f"RESULT: CUT (AM-16 item 2; AM-19 item 2(g)) alpha_cwd = 50; no selection file; decision record "
              f"{e.record_sha256}")
        return SWEEP_CUT_EXIT
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
              f"[{', '.join(cut)}]; alpha_sweep_cut={doc['record']['alpha_sweep_cut']}; review it, commit and push it "
              "within 24 hours and enter its sha256 in the decision log: AM-19 item 2(h))")
        return 0
    for line in doc["rule_trace"]:
        print(f"  {line}")
    w = doc["winner"]
    div = ", ".join(f"{x['value']:g}" for x in doc["excluded_am7a"])
    cut = ", ".join(f"{x['value']:g}" for x in doc["cut_am19"])
    note = "" if w["checkpoint_present"] else "; the winner's checkpoint is absent (ckpt_sha256 null)"
    print(f"RESULT: SELECTED alpha_cwd = {w['alpha']:g} (run {w['run_id']}, tie={doc['tie']}, "
          f"boundary={doc['boundary']}/{doc['boundary_kind']}, diverged=[{div}]) cut=[{cut}]{note} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
