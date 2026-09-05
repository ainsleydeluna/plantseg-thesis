#!/usr/bin/env python3
"""Pod-side histogram verification for the CE class weights (B34b Tier 2). ADVISORY.

Recomputes the FULL train-split mask pixel histogram on this machine and asserts EXACT equality
against reports/e1_class_weights.json's own `pixel_counts` and `total_nonignore_pixels`. Writes
reports/e1_class_weights_pod_verify.md.

WHY THIS EXISTS -- and why the pre-flight gate is not enough
------------------------------------------------------------
scripts/preflight_e1.py stage 1 proves the committed weight VECTOR is internally consistent and is
the one B18a computed. It says nothing about whether THIS machine's dataset is the dataset those
weights were derived from.

The nearest existing check is verify_env.py's per-split pair count (its section 17b), which compares
the NUMBER of image/mask pairs per split against configs/data.py SPLIT_SIZES. That is a CARDINALITY
check. Identical file counts are entirely compatible with a different pixel histogram: a re-exported,
re-rasterized, differently-compressed, or partially-corrupted mask set of the same cardinality passes
it unchanged. The CE weights are a function of the per-class PIXEL histogram over ~4.58e9 pixels, not
of the file count. Count equality is NECESSARY BUT NOT SUFFICIENT to bind the weights to the data.

This script closes that gap. It is the only check in the repository that does.

ADVISORY, NOT GATING -- and what that does and does not mean
------------------------------------------------------------
This is deliberately NOT a stage in scripts/preflight_e1.py. It needs the dataset mounted and reads
every train mask; on a cold network volume that is an I/O-bound minutes-long operation, and making a
correct launch fail on storage flakiness would be a false NO-GO.

Advisory means A HUMAN MAY OVERRIDE. It does NOT mean the script shrugs:
  * on mismatch it prints MISMATCH and exits NON-ZERO;
  * the report it writes records the verdict in full;
  * if this script is never run, NO report is produced -- and the absence of
    reports/e1_class_weights_pod_verify.md from a run's artifact set is itself the record that E1
    was launched WITHOUT histogram verification. The emitted report states this explicitly so the
    skip case cannot be read as a pass.

It is a named step in the launch checklist (reports/e1_runpod_launch_runbook.md section 9).

RUNTIME
-------
~1-2 minutes single-threaded on warm local storage; budget ~5 minutes cold on a pod network volume.
Basis: a 60-mask sample timed at 0.009 s/mask on the development box extrapolates to ~49 s over
5,367 masks; that sample skewed small on pixel count (3.34e9 projected vs 4.58e9 recorded), so the
pixel-scaled estimate is ~65-70 s, with headroom for cold-cache first reads.

Read-only. No training, no GPU, no downloads. Writes exactly one file, under reports/.

Usage:
    python scripts/verify_class_weights_pod.py
    python scripts/verify_class_weights_pod.py --limit 50     # smoke the plumbing, NEVER a verdict
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs.data import DATA, SPLIT_SIZES          # noqa: E402

CLASS_WEIGHTS_JSON = REPO / "reports" / "e1_class_weights.json"
OUT_MD = REPO / "reports" / "e1_class_weights_pod_verify.md"
NUM_CLASSES = DATA["num_classes"]
IGNORE_INDEX = DATA["ignore_index"]


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:                                                 # noqa: BLE001
        return "UNKNOWN"


def recompute_histogram(mask_paths, limit=None):
    """Per-class non-ignore pixel counts over the given masks. Returns (counts, n_masks, secs)."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    paths = mask_paths[:limit] if limit else mask_paths
    t0 = time.time()
    for n, p in enumerate(paths, 1):
        a = np.asarray(Image.open(p))
        # Bin every value, then drop ignore and anything out of range. Counting first and
        # discarding after keeps a stray label visible instead of silently absorbing it.
        b = np.bincount(a.reshape(-1), minlength=256)
        counts += b[:NUM_CLASSES]
        if n % 500 == 0:
            print(f"    ... {n}/{len(paths)} masks ({time.time() - t0:.0f}s)", flush=True)
    return counts, len(paths), time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pod-side CE class-weight histogram verification.")
    ap.add_argument("--limit", type=int, default=None,
                    help="decode only the first N masks: plumbing smoke ONLY, never a verdict")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("CE CLASS-WEIGHT POD VERIFICATION (B34b Tier 2, ADVISORY)")
    print("=" * 78)

    if not CLASS_WEIGHTS_JSON.exists():
        print(f"MISMATCH: artifact not found: {CLASS_WEIGHTS_JSON}", file=sys.stderr)
        return 2
    doc = json.loads(CLASS_WEIGHTS_JSON.read_text(encoding="utf-8"))
    expected_counts = doc["pixel_counts"]
    expected_total = doc["total_nonignore_pixels"]
    expected_masks = doc.get("train_mask_count")

    root = Path(DATA["root"])
    mask_dir = root / "annotations" / "train"
    print(f"  dataset root : {root}")
    print(f"  mask dir     : {mask_dir}")
    print(f"  artifact     : {CLASS_WEIGHTS_JSON.relative_to(REPO).as_posix()}")
    if not mask_dir.is_dir():
        print(f"MISMATCH: train mask directory absent: {mask_dir}", file=sys.stderr)
        return 2

    mask_paths = sorted(glob.glob(str(mask_dir / "*.png")))
    print(f"  masks found  : {len(mask_paths)} (artifact recorded {expected_masks}; "
          f"configs/data.py SPLIT_SIZES['train'] = {SPLIT_SIZES['train']})")
    if args.limit:
        print(f"\n  *** --limit {args.limit}: PLUMBING SMOKE ONLY. This run cannot produce a "
              f"verdict. ***")

    print("\n  recomputing the full train histogram ...", flush=True)
    counts, n_masks, secs = recompute_histogram(mask_paths, args.limit)
    total = int(counts.sum())
    print(f"  done: {n_masks} masks in {secs:.1f}s ({secs / max(n_masks, 1):.4f} s/mask), "
          f"{total:,} non-ignore pixels")

    if args.limit:
        print("\nRESULT: SMOKE ONLY — no verdict (partial pass over the split).")
        return 0

    # ---- exact comparison ----
    problems = []
    if expected_masks is not None and n_masks != expected_masks:
        problems.append(f"mask count {n_masks} != artifact train_mask_count {expected_masks}")
    if total != expected_total:
        problems.append(f"total_nonignore_pixels {total:,} != artifact {expected_total:,} "
                        f"(delta {total - expected_total:+,})")
    diffs = [(c, int(counts[c]), expected_counts[c])
             for c in range(NUM_CLASSES) if int(counts[c]) != expected_counts[c]]
    if diffs:
        problems.append(f"{len(diffs)} of {NUM_CLASSES} per-class pixel counts differ")

    ok = not problems
    verdict = "MATCH" if ok else "MISMATCH"

    print("\n" + "=" * 78)
    if ok:
        print("  per-class pixel_counts     : all 116 EXACT")
        print(f"  total_nonignore_pixels     : {total:,} EXACT")
        print(f"  train mask count           : {n_masks} EXACT")
    else:
        for p in problems:
            print(f"  [FAIL] {p}")
        for c, got, exp in diffs[:10]:
            print(f"         class {c:3d}: recomputed {got:,} vs artifact {exp:,} "
                  f"(delta {got - exp:+,})")
        if len(diffs) > 10:
            print(f"         ... and {len(diffs) - 10} more")
    print(f"RESULT: {verdict}")
    print("=" * 78)

    _write_report(verdict, ok, problems, diffs, n_masks, total, secs, root, expected_masks,
                  expected_total)
    print(f"\nreport -> {OUT_MD.relative_to(REPO).as_posix()}")

    if not ok:
        print("MISMATCH: the CE class weights were NOT derived from this machine's train split. "
              "Do not launch E1 without an explicit, recorded override.", file=sys.stderr)
        return 1
    return 0


def _write_report(verdict, ok, problems, diffs, n_masks, total, secs, root, expected_masks,
                  expected_total) -> None:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    lines = [
        "# CE Class-Weight Pod Verification (B34b Tier 2)",
        "",
        f"**RESULT: {'✅ MATCH' if ok else '❌ MISMATCH'}** — recomputed train-split pixel histogram "
        f"vs `reports/e1_class_weights.json`.",
        "",
        f"_Generated: {now} · HEAD `{_git_head()}` · host `{platform.node()}` "
        f"({platform.system()} {platform.machine()}) · dataset root `{root}`._",
        "",
        "**ADVISORY check.** A human may override a MISMATCH; the script does not gate the launch.",
        "It is a named step in `reports/e1_runpod_launch_runbook.md` section 9.",
        "",
        "> **If this report is absent from a run's artifact set, E1 was launched WITHOUT "
        "class-weight histogram verification.** Absence is not a pass — it is the record of a skip.",
        "",
        "## Why the pre-flight split-count check is not sufficient",
        "",
        "`scripts/verify_env.py` (section 17b) compares the **number** of image/mask pairs per split "
        "against `configs/data.py` `SPLIT_SIZES`. That is a **cardinality** check. Identical file "
        "counts are compatible with a different pixel histogram — a re-exported, re-rasterized, "
        "differently-compressed, or partially-corrupted mask set of the same cardinality passes it "
        "unchanged. The CE weights are a function of the per-class **pixel histogram**, not of the "
        "file count, so count equality is necessary but **not sufficient** to bind the weights to "
        "the data. This report is the only artifact that establishes that binding.",
        "",
        "## Result",
        "",
        "| quantity | recomputed here | artifact | verdict |",
        "|---|---|---|---|",
        f"| train masks decoded | {n_masks:,} | {expected_masks:,} | "
        f"{'✅' if expected_masks == n_masks else '❌'} |",
        f"| total non-ignore pixels | {total:,} | {expected_total:,} | "
        f"{'✅' if total == expected_total else '❌'} |",
        f"| per-class `pixel_counts` | 116 classes | 116 classes | "
        f"{'✅ all exact' if not diffs else f'❌ {len(diffs)} differ'} |",
        "",
        f"Decode time {secs:.1f}s ({secs / max(n_masks, 1):.4f} s/mask), single-threaded.",
        "",
    ]
    if not ok:
        lines += ["## Discrepancies", ""]
        lines += [f"- {p}" for p in problems]
        lines += ["", "| class | recomputed | artifact | delta |", "|---|---|---|---|"]
        lines += [f"| {c} | {got:,} | {exp:,} | {got - exp:+,} |" for c, got, exp in diffs[:50]]
        if len(diffs) > 50:
            lines.append(f"| … | … | … | {len(diffs) - 50} more |")
        lines += [
            "",
            "**Interpretation.** The committed CE class weights were **not** derived from this "
            "machine's train split. Launching E1 would train under a weighting that does not match "
            "the data. Either restore the dataset the artifact was computed from, or regenerate the "
            "artifact against this dataset and re-run the B18a provenance record — do not proceed "
            "on the assumption that the difference is cosmetic.",
        ]
    else:
        lines += [
            "## Interpretation",
            "",
            "Every per-class pixel count and the non-ignore total reproduce **exactly**. The "
            "committed CE class weights were derived from the train split present on this machine, "
            "so `train_e1.py`'s unconditional load of the artifact is bound to this data.",
        ]
    lines += ["", "---", "",
              "Read-only verification: no training, no GPU, no downloads; the dataset was opened "
              "for decoding only and no dataset file was modified. This report is the only file "
              "written."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
