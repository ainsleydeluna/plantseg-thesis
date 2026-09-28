#!/usr/bin/env python3
"""L-AM17-GTPRESENT: dataset-level mIoU under the union-present and GT-present rules, from one
evaluation artifact's sufficient_stats.npz (AM-17 item 1(b); docs/lane_specs/part1.md lane 3).

    python -B scripts/eligibility_variants.py ARTIFACT_DIR [ARTIFACT_DIR ...] --out OUT_DIR
        [--expect-union-all-class VALUE]

For each artifact directory: verify it (MANIFEST.sha256 plus the stats-ingest integrity checks), rebuild
the per-class totals from the per-image sparse triplets (src/stats/eligibility.py), and compute the four
numbers (all-class and disease-only under each rule), the per-class table and three faithfulness proofs:

  R1  the union-present all-class and disease-only values equal summary.json's dataset-level values
      BITWISE (the float32 arithmetic of src/eval/metrics.py), with the same eligible-class counts;
  R2  the per-class table equals summary.json per_class (supports, intersection, union, IoU exactly,
      iou_status);
  R3  GT-present all-class == union-present all-class x n_union / n_gt within 1e-12 (float64): a class
      that is union-present but not GT-present has TP = 0, so both means share one sum. This is the
      part1 d2 check (class 69 is the only such class on VAL) in a form that stays valid if a second
      GT-absent class appears; the direct recomputation is `rules_float64`.

Writes <OUT_DIR>/<run_id>_eligibility_variants.json only when R1-R3 hold (and --expect-union-all-class,
if given, matches exactly). An existing output file is never overwritten. Reads the four artifact files
only: no dataset, no model, no TEST image or mask.

Exit codes: 0 every artifact written and faithful; 1 a faithfulness proof failed or the expected value
differs (STOP: the npz is not faithful to the summary); 2 usage, verification or integrity error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_UNFAITHFUL, EXIT_ERROR = 0, 1, 2
RELATION_TOL = 1e-12
OUTPUT_SUFFIX = "_eligibility_variants.json"
RUN_ID_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
PER_CLASS_KEYS = (("gt_support", "gt"), ("pred_support", "pred"), ("intersection", "tp"),
                  ("union", "union"), ("iou", "iou"), ("iou_status", "iou_status"))


class UsageError(RuntimeError):
    """Not a verdict: a bad argument, an unreadable artifact or an existing output file."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _same_float(a, b) -> bool:
    """Bitwise equality for the JSON float/None values compared here (NaN never occurs: strict JSON)."""
    if a is None or b is None:
        return a is None and b is None
    return isinstance(a, float) and isinstance(b, float) and a == b


def analyse(run_dir: Path) -> tuple[dict, list[str]]:
    """Report for one artifact and the list of failed faithfulness proofs (empty when faithful)."""
    import numpy as np
    import torch

    from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME
    from src.stats.eligibility import (GT_PRESENT, RULES, UNION_PRESENT, class_totals,
                                       per_class_table, rule_variants)
    from src.stats.ingest import Policy, load_run, verify_run_manifest

    summary = verify_run_manifest(run_dir)
    run = load_run(run_dir, Policy.NONOFFICIAL_SMOKE)      # integrity checks only; no officiality claim
    ident = run.identity
    if not RUN_ID_SAFE.match(ident.run_id):
        raise UsageError(f"run_id {ident.run_id!r} is not safe as a file name")
    totals = class_totals(run.stats, ident.num_classes)
    variants = rule_variants(totals, background_index=ident.background_index)
    table = per_class_table(totals)
    problems: list[str] = []

    # R1 -- bitwise reproduction of the summary's union-present values and eligible counts
    dl = summary["dataset_level"]
    r1 = {}
    for scope, key in (("all_class", "all_class_miou"), ("disease_only", "disease_only_miou")):
        mine = variants[UNION_PRESENT][scope]
        same = _same_float(mine.value, dl.get(key))
        same_n = mine.n_classes == dl.get(f"{key}_n_eligible")
        r1[key] = {"summary": dl.get(key), "reproduced": mine.value, "bitwise_equal": same,
                   "n_eligible_summary": dl.get(f"{key}_n_eligible"),
                   "n_eligible_reproduced": mine.n_classes}
        if not same:
            problems.append(f"R1 union-present {scope} {mine.value!r} != summary {key} {dl.get(key)!r}")
        if not same_n:
            problems.append(f"R1 union-present {scope} n={mine.n_classes} != summary "
                            f"{key}_n_eligible={dl.get(f'{key}_n_eligible')!r}")

    # R2 -- the per-class table equals summary.per_class
    pc = summary["per_class"]
    r2 = {}
    for key, col in PER_CLASS_KEYS:
        mine = [row[col] for row in table]
        theirs = pc.get(key)
        same = (isinstance(theirs, list) and len(theirs) == len(mine) and all(
            _same_float(a, b) if col == "iou" else a == b for a, b in zip(mine, theirs)))
        r2[key] = same
        if not same:
            problems.append(f"R2 summary per_class.{key} differs from the npz reconstruction")

    # R3 -- GT-present = union-present x n_union / n_gt (float64, exact up to rounding)
    up, gp = variants[UNION_PRESENT]["all_class"], variants[GT_PRESENT]["all_class"]
    extra = sorted(set(up.eligible) - set(gp.eligible))
    r3 = {"n_union_present": up.n_classes, "n_gt_present": gp.n_classes,
          "union_present_not_gt_present": extra,
          "their_tp": [int(totals.tp[c]) for c in extra], "tolerance": RELATION_TOL}
    if up.value_float64 is None or gp.value_float64 is None or not set(gp.eligible) <= set(up.eligible):
        r3["holds"] = False
        problems.append("R3 undefined: no eligible class, or a GT-present class is not union-present")
    else:
        scaled = up.value_float64 * up.n_classes / gp.n_classes
        diff = abs(gp.value_float64 - scaled)
        r3.update({"gt_present_all_class_float64": gp.value_float64,
                   "union_present_all_class_float64_x_n_union_over_n_gt": scaled,
                   "abs_diff": diff, "holds": diff <= RELATION_TOL and not any(r3["their_tp"])})
        if not r3["holds"]:
            problems.append(f"R3 GT-present relation fails: |diff|={diff!r}, extra-class tp={r3['their_tp']}")

    files = sorted([*ARTIFACT_FILES, MANIFEST_NAME])
    run_blk, ds = summary["run"], summary["dataset"]
    report = {
        "lane": "L-AM17-GTPRESENT",
        "artifact_sha256s": {name: _sha256(run_dir / name) for name in files},
        "split": ds["split"],
        "rules": {rule: {"all_class": variants[rule]["all_class"].value,
                         "disease_only": variants[rule]["disease_only"].value,
                         "n_classes": variants[rule]["all_class"].n_classes,
                         "n_classes_disease_only": variants[rule]["disease_only"].n_classes}
                  for rule in RULES},
        "rules_float64": {rule: {"all_class": variants[rule]["all_class"].value_float64,
                                 "disease_only": variants[rule]["disease_only"].value_float64}
                          for rule in RULES},
        "faithfulness": {"R1_summary_bitwise": r1, "R2_per_class_equal": r2,
                         "R3_gt_present_relation": r3},
        "artifact": {"dir_name": run_dir.name, "run_id": run_blk["run_id"], "stage": run_blk["stage"],
                     "artifact_status": run_blk["artifact_status"], "condition": ds["condition"],
                     "schema_version": summary["schema_version"],
                     "artifact_schema_version": summary.get("artifact_schema_version"),
                     "split_manifest_sha256": ds["split_manifest_sha256"],
                     "metric_impl_sha256": run_blk["metric_impl_sha256"],
                     "repo_commit": run_blk["repo_commit"], "num_classes": ds["num_classes"],
                     "background_index": ds["background_index"]},
        "arithmetic": {
            "rules": "float32 counts, IoU and torch float32 mean over the eligible classes: the "
                     "arithmetic of src/eval/metrics.py miou_from_confusion",
            "rules_float64": "float64 per-class IoU, exactly rounded sum (math.fsum), one division",
            "per_class.iou": "float64 TP/union, correctly rounded (equals summary per_class.iou)"},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": torch.__version__,
                        "cpu_capability": torch.backends.cpu.get_cpu_capability()},
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "per_class": table,
    }
    return report, problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="L-AM17-GTPRESENT eligibility variants (AM-17 item 1(b)).")
    p.add_argument("artifact_dirs", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path, help="output directory (created if absent)")
    p.add_argument("--expect-union-all-class", type=float, default=None,
                   help="exact expected union-present all-class value (one artifact only)")
    args = p.parse_args(argv)
    if args.expect_union_all_class is not None and len(args.artifact_dirs) != 1:
        print("error: --expect-union-all-class applies to exactly one ARTIFACT_DIR", file=sys.stderr)
        return EXIT_ERROR

    worst = EXIT_OK
    for run_dir in args.artifact_dirs:
        try:
            report, problems = analyse(run_dir)
            expected = args.expect_union_all_class
            got = report["rules"]["union_present"]["all_class"]
            if expected is not None and not _same_float(got, expected):
                problems.append(f"union-present all-class {got!r} != expected {expected!r}")
            rid, up, gp = (report["artifact"]["run_id"], report["rules"]["union_present"],
                           report["rules"]["gt_present"])
            extra = report["faithfulness"]["R3_gt_present_relation"]["union_present_not_gt_present"]
            print(f"{rid}: union-present all={up['all_class']!r} dis={up['disease_only']!r} "
                  f"(n={up['n_classes']}/{up['n_classes_disease_only']}) | GT-present "
                  f"all={gp['all_class']!r} dis={gp['disease_only']!r} "
                  f"(n={gp['n_classes']}/{gp['n_classes_disease_only']}) | union-present-only "
                  f"classes {extra}")
            if problems:
                for msg in problems:
                    print(f"  UNFAITHFUL: {msg}")
                print(f"  {rid}: nothing written")
                worst = max(worst, EXIT_UNFAITHFUL)
                continue
            args.out.mkdir(parents=True, exist_ok=True)
            target = args.out / f"{rid}{OUTPUT_SUFFIX}"
            try:
                with open(target, "x", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            except FileExistsError as e:
                raise UsageError(f"refusing to overwrite {target}") from e
            print(f"  R1 bitwise, R2 per-class and R3 relation hold -> {target} "
                  f"(sha256 {_sha256(target)})")
        except Exception as e:                            # noqa: BLE001 -- reported, never a verdict
            print(f"  ERROR {run_dir}: {type(e).__name__}: {e}")
            worst = max(worst, EXIT_ERROR)
    verdict = {EXIT_OK: "ELIGIBILITY VARIANTS WRITTEN (faithful)",
               EXIT_UNFAITHFUL: "STOP -- npz reconstruction not faithful to summary.json",
               EXIT_ERROR: "ERROR -- see above"}[worst]
    print(f"RESULT: {verdict}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
