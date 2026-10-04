#!/usr/bin/env python3
"""Select alpha_cwd from the three E3 seed-42 sweep runs (AM-16 item 2 / DL-27 with AM-7a; lanes
L-AM16-ALPHA and L-KD-HARDEN).

    python scripts/select_alpha.py --runs <E3 run dir> <E3 run dir> <E3 run dir>
                                   [--band reports/derived/dl27_band.json]
                                   [--lambda-selection reports/derived/lambda_selection.json]
                                   [--out reports/derived/alpha_selection.json]

Rule (configs/sweep_rules.json 'alpha_cwd', checked against configs/distill.py): each candidate's value
is its best-checkpoint VAL all-class mIoU; the highest wins; candidates within the band of the best are
tied (inclusive); a tie that contains 50 goes to 50 (the Chapter 3 / Shu et al. default), otherwise to
the smallest alpha; a winner at 25 or 100 is flagged as a boundary result and the grid is not extended.
The winning run is E3 seed 42. The three runs must share one lambda_logit, equal to the winner of the
lambda selection file (AM-16 item 2: the alpha sweep runs after lambda is fixed).

AM-7a: a non-default candidate (alpha 25 or 100) whose run ends in the trainer's own run_abort record for
a student divergence (AM-7 (a)/(b)) is excluded from the selection and never relaunched; its directory is
still a required input. The rule runs over the finished candidates (a sole finished candidate wins with
no band and no tie); a winner at an edge of the finished set next to a diverged value is a boundary
result too. A diverged default candidate (alpha 50) keeps AM-7 in full: refused.

Order of the refusals (K8-2(a)): the candidates' recipe (recipe_mismatch), then the shared lambda over
every loaded run, finished or diverged (lambda_mismatch; skipped when no run loaded, so a sweep still
training stays a shortfall), then a diverged default (default_candidate_diverged), then the shortfall,
then the rule. A diverged default run at a lambda other than the selection's is a lambda_mismatch: the
run is not a candidate of this sweep.

Band file (the DL-27 decision-log entry, written after B66 and before the sweep):
    {"e1_best_val": {"42": <float>, "43": <float>, "44": <float>}, "s": <float>, "band": <float, optional>}
s is the n = 3 sample SD of E1's best VAL all-class mIoU; band = max(0.005, sqrt(2) * s). s is recomputed
from the three values and a recorded band is recomputed; any disagreement refuses.

Exit 0: the selection file is written. Exit 2: refused (the band or lambda selection file missing or
malformed, a malformed, unreadable or inconsistent input, a finished run whose checkpoint is absent,
an existing selection file, an AM-7a refusal: default_candidate_diverged, run_aborted_other,
abort_record_invalid, or item 10b's recipe_mismatch). Exit 3: a grid value with neither a finished nor a
diverged run (a run directory absent, its run_meta/telemetry not written yet, a finished run without
best.json, or neither a run_end nor a run_abort record): lane 2 STOP; a cut sweep means alpha = 50 per
AM-16 and no selection file. Exit 4: an unexpected error. Only exit 0 writes a file.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training.sweep_select import (RULES_PATH, UNEXPECTED_ERROR_EXIT,  # noqa: E402
                                       SelectionRefused, apply_rule, collect_candidates, dl27_band,
                                       load_rules, missing_grid_values, read_json,
                                       refuse_diverged_default, refuse_test_path, sha256_file,
                                       write_selection)

KEY = "alpha_cwd"


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if REPO in p.parents else str(p)


def lambda_of_record(path: Path) -> float:
    """The lambda_logit winner of the committed lambda selection (scripts/select_lambda.py output)."""
    path = Path(path)
    refuse_test_path(path)
    if not path.is_file():
        raise SelectionRefused("lambda_selection_missing", f"the lambda selection file is missing: "
                                                           f"{path}. AM-16 item 2 runs the alpha sweep "
                                                           "after lambda is fixed.")
    doc = read_json(path)
    winner = doc.get("winner") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or doc.get("format") != "lambda_selection/1" \
            or not isinstance(winner, dict) or not isinstance(winner.get("lambda"), (int, float)):
        raise SelectionRefused("lambda_selection_format", f"{path} is not a lambda_selection/1 file "
                                                          "with a winner")
    return float(winner["lambda"])


def select(runs, band_path: Path, out: Path, rules_path: Path = RULES_PATH,
           lambda_selection: Path | None = None) -> dict:
    rules = load_rules(rules_path)
    sweep = rules[KEY]
    band_path = Path(band_path)
    refuse_test_path(band_path)
    if not band_path.is_file():
        raise SelectionRefused("band_missing", f"the DL-27 band file is missing: {band_path}. The alpha "
                                               "sweep is selected only after the DL-27 entry exists.")
    s, band, band_trace = dl27_band(read_json(band_path, "band_format"), float(sweep["band"]["floor"]))
    lambda_path = Path(lambda_selection) if lambda_selection is not None else \
        REPO / "reports" / "derived" / "lambda_selection.json"
    lam_record = lambda_of_record(lambda_path)
    finished, diverged, shortfall = collect_candidates(runs, sweep, KEY, check_default=False)
    loaded = finished + diverged
    lambdas = sorted({c["lambda_logit"] for c in loaded}, key=repr)
    if loaded and (len(lambdas) != 1 or lambdas[0] is None or float(lambdas[0]) != lam_record):
        raise SelectionRefused("lambda_mismatch", f"the alpha runs must share one lambda_logit equal to "
                                                  f"the lambda selection's winner {lam_record!r}; got "
                                                  f"{lambdas}")
    refuse_diverged_default(diverged, sweep, KEY)
    missing = missing_grid_values(loaded, sweep)
    if shortfall or missing:
        absent = ["{} ({})".format(x["run_dir"], x["code"]) for x in shortfall]
        raise SelectionRefused(
            "shortfall_lane2_stop",
            f"alpha selection needs all three runs, finished or diverged; finished: "
            f"{sorted(c['value'] for c in finished)}; diverged: {sorted(c['value'] for c in diverged)}; "
            f"grid values without a finished or diverged run: {missing}; absent or unfinished: {absent} "
            "(diverged candidates are not shortfalls: pass their directories). Lane 2 STOP; a cut sweep "
            "means alpha = 50 (AM-16 item 2) and no selection file.")
    res = apply_rule(finished, sweep, band, "alpha", diverged=diverged)
    rec = lambda c: {"alpha": c["value"], "run_id": c["run_id"], "best_val": c["best_val"],  # noqa: E731
                     "ckpt_sha256": c["ckpt_sha256"]}
    doc = {"format": "alpha_selection/1", "rule": sweep["source"],
           "amendments": ["AM-16 item 2", "AM-7a"],
           "rules_file": {"path": _rel(rules_path), "sha256": sha256_file(Path(rules_path))},
           "band_file": {"path": _rel(band_path), "sha256": sha256_file(band_path)},
           "lambda_selection_file": {"path": _rel(lambda_path), "sha256": sha256_file(lambda_path),
                                     "lambda": lam_record},
           "candidates": [rec(c) for c in sorted(finished, key=lambda c: c["value"])],
           "s": s, "band": band, "rule_trace": band_trace + res["rule_trace"],
           "winner": rec(res["winner"]), "tie": res["tie"], "tied": res["tied"],
           "boundary": res["boundary"], "boundary_kind": res["boundary_kind"],
           "winner_is": sweep["winner_is"], "excluded_am7a": res["excluded_am7a"],
           "n_finished": res["n_finished"], "n_diverged": res["n_diverged"],
           "lambda_logit": lambdas[0],
           "inputs": [{"run_dir": c["run_dir"], "checkpoint": c["checkpoint"],
                       "best_json_sha256": c["best_json_sha256"],
                       "run_meta_sha256": c["run_meta_sha256"]} for c in finished],
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script": {"path": "scripts/select_alpha.py", "sha256": sha256_file(Path(__file__))}}
    write_selection(out, doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select alpha_cwd (AM-16 item 2 / DL-27) from the E3 sweep.")
    ap.add_argument("--runs", nargs="+", required=True, help="the three E3 seed-42 run (checkpoint) dirs")
    ap.add_argument("--band", default=str(REPO / "reports" / "derived" / "dl27_band.json"))
    ap.add_argument("--lambda-selection",
                    default=str(REPO / "reports" / "derived" / "lambda_selection.json"))
    ap.add_argument("--out", default=str(REPO / "reports" / "derived" / "alpha_selection.json"))
    ap.add_argument("--rules", default=str(RULES_PATH))
    a = ap.parse_args(argv)
    try:
        doc = select(a.runs, Path(a.band), Path(a.out), Path(a.rules), Path(a.lambda_selection))
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
    print(f"RESULT: SELECTED alpha_cwd = {w['alpha']:g} (run {w['run_id']}, tie={doc['tie']}, "
          f"boundary={doc['boundary']}/{doc['boundary_kind']}, diverged=[{div}]) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
