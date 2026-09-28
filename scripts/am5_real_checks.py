#!/usr/bin/env python3
"""L-AM5 real-data checks d4 and d5 (docs/lane_specs/part1.md lane 1 (d)). LOCAL ONLY: they read
real evaluation artifacts produced on the local machine; nothing here runs a model or opens a dataset.

    python -B scripts/am5_real_checks.py d4 PRE_LANE_ARTIFACT LANE_ARTIFACT
    python -B scripts/am5_real_checks.py d5 RUN_A RUN_B

d4 (regression): one checkpoint scored by the evaluator at the pre-lane commit (PRE) and at the lane
commit (LANE), same machine, `--device cpu --batch-size 16`. PASS iff
  * every PRE per-image field is unchanged in LANE -- each LANE row is the PRE row with
    `"am5_excluded": <bool>` appended last, byte for byte, so disease_only_miou,
    n_eligible_disease_only and gt_disease_classes are bitwise equal -- and the flag equals
    n_eligible_disease_only == 0;
  * every sufficient_stats.npz array is bitwise equal (same keys, dtype, shape and bytes);
  * summary.json is equal once the run provenance keys (PROVENANCE_RUN_KEYS) and run.eval_runtime are
    set aside and the two new keys (am5, artifact_schema_version) are removed from LANE; LANE's am5
    block matches its rows. An eval_runtime difference is printed as a NOTE, not a failure.
Anything else is a STOP (part1 lane 1 (f): "d4 differs in any old field").

d5 (ingest): both VAL artifacts ingested under the REHEARSAL policy (AM-5 applied). PASS iff each run's
excluded count equals its number of rows with an empty gt_disease_classes (an independent count), and
the pair's excluded id sets -- hence excluded_ids_sha256 -- are identical. Prints K_val.

Exit codes: 0 PASS, 1 FAIL (STOP and report), 2 usage error or an artifact that cannot be read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_PASS, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
# Run provenance: where and when a run happened, never what it computed. checkpoint_path is a path
# string; the checkpoint's identity is run.checkpoint_sha256, which IS compared.
PROVENANCE_RUN_KEYS = ("run_id", "timestamp_utc", "repo_commit", "worktree_state_sha256",
                       "governed_paths_clean", "dirty_allowlisted", "checkpoint_path")
NEW_SUMMARY_KEYS = ("am5", "artifact_schema_version")


class Unreadable(RuntimeError):
    """An artifact that cannot be verified or read: not a verdict."""


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _diff_keys(a, b, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ (lists compared whole, dicts recursed)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            out += _diff_keys(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    return [] if _canonical(a) == _canonical(b) else [prefix.rstrip(".")]


def _verified_summary(d: Path) -> dict:
    from src.eval.artifacts import verify_artifact
    try:
        return verify_artifact(d)
    except Exception as e:                                # noqa: BLE001 -- reported verbatim
        raise Unreadable(f"verify_artifact({d}) failed: {type(e).__name__}: {e}") from e


def d4(pre: Path, lane: Path) -> int:
    import numpy as np

    from src.eval.artifacts import AM5_RULE, ARTIFACT_SCHEMA_VERSION, am5_excluded_ids_sha256

    sp, sl = _verified_summary(pre), _verified_summary(lane)
    problems: list[str] = []
    notes: list[str] = []
    if any(k in sp for k in NEW_SUMMARY_KEYS):
        problems.append("PRE carries the L-AM5 summary keys: it is not a pre-lane artifact")
    if sl.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION or "am5" not in sl:
        problems.append(f"LANE lacks the L-AM5 layout (artifact_schema_version "
                        f"{sl.get('artifact_schema_version')!r}, expected {ARTIFACT_SCHEMA_VERSION!r})")

    # ---- per_image.jsonl: LANE row == PRE row + '"am5_excluded": <bool>' appended last ----
    pre_lines = (pre / "per_image.jsonl").read_text(encoding="utf-8").splitlines()
    lane_lines = (lane / "per_image.jsonl").read_text(encoding="utf-8").splitlines()
    if len(pre_lines) != len(lane_lines):
        problems.append(f"row count {len(pre_lines)} (PRE) != {len(lane_lines)} (LANE)")
    bad_rows, excluded = [], []
    for a, b in zip(pre_lines, lane_lines):
        ra, rb = json.loads(a), json.loads(b)
        flag = rb.get("am5_excluded")
        tail = ', "am5_excluded": ' + ("true" if flag is True else "false") + "}"
        if "am5_excluded" in ra or not isinstance(flag, bool) or b != a[:-1] + tail \
                or flag is not (ra["n_eligible_disease_only"] == 0):
            bad_rows.append(ra.get("image_id"))
        if flag is True:
            excluded.append(rb["image_id"])
    if bad_rows:
        problems.append(f"{len(bad_rows)} per-image row(s) differ beyond the appended am5_excluded, "
                        f"e.g. {bad_rows[:3]}")
    spec_fields = ("disease_only_miou", "n_eligible_disease_only", "gt_disease_classes")
    same_fields = len(pre_lines) == len(lane_lines) and all(
        _canonical(json.loads(a)[k]) == _canonical(json.loads(b)[k])
        for a, b in zip(pre_lines, lane_lines) for k in spec_fields)
    print(f"per-image {', '.join(spec_fields)} bitwise equal: {same_fields}")
    if not same_fields:
        problems.append(f"per-image {spec_fields} are not bitwise equal")

    # ---- sufficient_stats.npz: bitwise ----
    with np.load(pre / "sufficient_stats.npz", allow_pickle=False) as za, \
            np.load(lane / "sufficient_stats.npz", allow_pickle=False) as zb:
        if sorted(za.files) != sorted(zb.files):
            problems.append(f"NPZ key sets differ: {sorted(za.files)} vs {sorted(zb.files)}")
        else:
            for k in sorted(za.files):
                xa, xb = za[k], zb[k]
                if xa.dtype != xb.dtype or xa.shape != xb.shape or xa.tobytes() != xb.tobytes():
                    problems.append(f"NPZ {k} differs (dtype/shape/bytes)")
    print(f"sufficient_stats.npz arrays bitwise equal: "
          f"{not any(p.startswith('NPZ') for p in problems)}")

    # ---- summary.json: equal apart from provenance, eval_runtime and the two new keys ----
    def strip(s: dict, drop_new: bool) -> dict:
        s = json.loads(json.dumps(s))
        for k in PROVENANCE_RUN_KEYS + ("eval_runtime",):
            s.get("run", {}).pop(k, None)
        if drop_new:
            for k in NEW_SUMMARY_KEYS:
                s.pop(k, None)
        return s
    diff = _diff_keys(strip(sp, False), strip(sl, True))
    if diff:
        problems.append(f"summary.json differs outside provenance and the new keys: {diff[:10]}")
    rt_a, rt_b = sp.get("run", {}).get("eval_runtime"), sl.get("run", {}).get("eval_runtime")
    if _canonical(rt_a) != _canonical(rt_b):
        notes.append("run.eval_runtime differs (runtime record, not a result value)")
    for k in PROVENANCE_RUN_KEYS:
        va, vb = sp.get("run", {}).get(k), sl.get("run", {}).get(k)
        if va != vb:
            notes.append(f"run.{k}: {va!r} -> {vb!r}")

    # ---- LANE am5 block matches its rows ----
    expect = {"rule": AM5_RULE, "excluded_count": len(excluded),
              "included_count": len(lane_lines) - len(excluded),
              "excluded_ids_sha256": am5_excluded_ids_sha256(excluded)}
    if sl.get("am5") != expect:
        problems.append(f"LANE summary.am5 {sl.get('am5')!r} != derivation from its rows {expect!r}")

    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(f"D4: {p}")
    print(f"am5 (LANE): excluded_count={expect['excluded_count']} "
          f"included_count={expect['included_count']} "
          f"excluded_ids_sha256={expect['excluded_ids_sha256']}")
    if problems:
        print("RESULT: D4 FAIL -- an old field changed (STOP and report)")
        return EXIT_FAIL
    print("RESULT: D4 PASS -- only the new fields differ")
    return EXIT_PASS


def d5(a_dir: Path, b_dir: Path) -> int:
    from src.stats.ingest import IngestError, Policy, am5_pair, load_run

    runs = []
    for d in (a_dir, b_dir):
        try:
            runs.append(load_run(d, Policy.REHEARSAL))
        except IngestError as e:
            raise Unreadable(f"REHEARSAL ingest of {d} refused: {e}") from e
    problems: list[str] = []
    for run, d in zip(runs, (a_dir, b_dir)):
        empty = sum(1 for ln in (d / "per_image.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(ln)["gt_disease_classes"] == [])
        am5 = run.am5
        print(f"{run.identity.run_id} (stage {run.identity.stage}, {run.identity.split}, "
              f"{run.identity.artifact_status}): am5 source={am5.source} excluded_count="
              f"{am5.excluded_count} included_count={am5.included_count} records={len(run.records)} "
              f"rows_with_empty_gt_disease_classes={empty} excluded_ids_sha256="
              f"{am5.excluded_ids_sha256}")
        if am5.excluded_count != empty or len(run.records) != am5.included_count:
            problems.append(f"{run.identity.run_id}: excluded_count {am5.excluded_count} != rows with "
                            f"empty gt_disease_classes {empty} (or records != included_count)")
    try:
        pair = am5_pair(runs[0], runs[1])
        print(f"pair: n_excluded_am5={pair.n_excluded_am5} n_included={pair.n_included} "
              f"excluded_ids_sha256={pair.excluded_ids_sha256}")
        print(f"K_val = {pair.n_excluded_am5}")
    except IngestError as e:
        problems.append(str(e))
    for p in problems:
        print(f"D5: {p}")
    if problems:
        print("RESULT: D5 FAIL (STOP and report)")
        return EXIT_FAIL
    print("RESULT: D5 PASS -- identical excluded sets; excluded_count == rows with empty "
          "gt_disease_classes")
    return EXIT_PASS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="L-AM5 real-data checks d4/d5 (local only).")
    sub = p.add_subparsers(dest="check", required=True)
    for name, a, b in (("d4", "pre_lane", "lane"), ("d5", "run_a", "run_b")):
        s = sub.add_parser(name)
        s.add_argument(a, type=Path)
        s.add_argument(b, type=Path)
    args = p.parse_args(argv)
    try:
        if args.check == "d4":
            return d4(args.pre_lane, args.lane)
        return d5(args.run_a, args.run_b)
    except Unreadable as e:
        print(f"RESULT: ERROR -- {e}")
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
