#!/usr/bin/env python3
"""DL-17 comparator: identity and PASS for the two E1 seed-42 VAL re-scores (EVALUATION_CONTRACT section 10).

    python -B scripts/compare_eval_artifacts.py RUN_A RUN_B            # full DL-17 verdict
    python -B scripts/compare_eval_artifacts.py --identity-only A B    # identity rules alone

Exit codes (full mode):
    0  PASS      -- both artifacts are valid DL-17 runs, identical, and within the DL-17 band
    1  FAIL      -- not identical, or |all_class_miou - DL17_REFERENCE_MIOU| > DL17_BAND
    2  VALIDITY  -- a validity precondition is violated: not a DL-17 run at all (STOP and report)
Validity is judged before identity whenever both artifacts pass verify_artifact; an artifact that
fails verify_artifact fails identity (exit 1).
--identity-only exits 0 (identical) or 1 (not identical) and prints both all_class_miou values.
Exit 3 is reserved for anything that is not a verdict: a malformed command line, --help, the same
directory given twice, two artifacts carrying the same run.run_id (a copy of one run is not two runs;
"RESULT: NO VERDICT"), or an unexpected error ("RESULT: ERROR").

Identity (A == B): both pass verify_artifact; per_image.jsonl byte-equal; the NPZ key sets are equal
and every array has the same dtype and shape and the same raw bytes; summary.json equal as canonical
JSON (sort_keys, compact separators, allow_nan=False) after removing only run.run_id and
run.timestamp_utc.

Validity preconditions (each artifact): run.eval_runtime present; determinism_policy_applied; model
device == input devices == cuda:N; batch_size 16, num_workers 0, forward_batches 53, actual_rows 846;
checkpoint iteration DL17_CHECKPOINT_ITER and stored best DL17_REFERENCE_MIOU; image digest
DL17_IMAGE_DIGEST; sum(per_class.gt_support) == DL17_GT_SUPPORT; run.checkpoint_sha256 ==
DL17_CHECKPOINT_SHA256; eval_runtime.gpu_name == DL17_GPU_NAME (exact).

PASS (DL-17 as DECIDED): identical, and |all_class_miou - DL17_REFERENCE_MIOU| <= DL17_BAND; a
difference in (DL17_RECORD_ABOVE, DL17_BAND] is recorded, not a failure.

Reads only the two artifact directories. Writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DL17_REFERENCE_MIOU = 0.36314016580581665
DL17_CHECKPOINT_ITER = 80000
DL17_CHECKPOINT_SHA256 = "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03"
DL17_IMAGE_DIGEST = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
DL17_GT_SUPPORT = 159279104
DL17_GPU_NAME = "NVIDIA A40"          # exact gpu_name of seed 42's run_meta (a substring would admit "RTX A4000")
DL17_BATCH_SIZE = 16
DL17_NUM_WORKERS = 0
DL17_FORWARD_BATCHES = 53
DL17_VAL_ROWS = 846
DL17_BAND = 1e-4
DL17_RECORD_ABOVE = 1e-6

EXIT_PASS, EXIT_FAIL, EXIT_VALIDITY, EXIT_USAGE = 0, 1, 2, 3
EPHEMERAL_RUN_KEYS = ("run_id", "timestamp_utc")


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strip_ephemeral(summary: dict) -> dict:
    s = json.loads(json.dumps(summary))
    for k in EPHEMERAL_RUN_KEYS:
        s.get("run", {}).pop(k, None)
    return s


def identity_problems(a: Path, b: Path) -> tuple[list[str], dict, dict]:
    """Identity rules. Returns (problems, summary_a, summary_b); summaries are {} when unreadable."""
    import numpy as np

    from src.eval.artifacts import verify_artifact

    problems: list[str] = []
    summaries = []
    for label, d in (("A", a), ("B", b)):
        try:
            summaries.append(verify_artifact(d))
        except Exception as e:                            # noqa: BLE001 -- reported verbatim
            problems.append(f"verify_artifact({label}) failed: {type(e).__name__}: {e}")
            summaries.append({})
    sa, sb = summaries
    if problems:
        return problems, sa, sb

    if (a / "per_image.jsonl").read_bytes() != (b / "per_image.jsonl").read_bytes():
        problems.append("per_image.jsonl differs byte-wise")

    with np.load(a / "sufficient_stats.npz", allow_pickle=False) as za, \
            np.load(b / "sufficient_stats.npz", allow_pickle=False) as zb:
        if sorted(za.files) != sorted(zb.files):
            problems.append(f"NPZ key sets differ: {sorted(za.files)} vs {sorted(zb.files)}")
        else:
            for k in sorted(za.files):
                xa, xb = za[k], zb[k]
                if xa.dtype != xb.dtype:
                    problems.append(f"NPZ {k}: dtype {xa.dtype} vs {xb.dtype}")
                elif xa.shape != xb.shape:
                    problems.append(f"NPZ {k}: shape {xa.shape} vs {xb.shape}")
                elif xa.tobytes() != xb.tobytes():
                    problems.append(f"NPZ {k}: values differ")

    if _canonical(_strip_ephemeral(sa)) != _canonical(_strip_ephemeral(sb)):
        problems.append("summary.json differs (canonical JSON, run.run_id/run.timestamp_utc removed)")
    return problems, sa, sb


def validity_problems(label: str, s: dict) -> list[str]:
    """DL-17 validity preconditions for one artifact's summary."""
    out: list[str] = []
    run = s.get("run", {})
    rt = run.get("eval_runtime")
    if not isinstance(rt, dict):
        return [f"{label}: run.eval_runtime is missing"]

    def need(ok: bool, what: str) -> None:
        if not ok:
            out.append(f"{label}: {what}")

    need(rt.get("determinism_policy_applied") is True, "determinism_policy_applied is not true")
    md = rt.get("model_device")
    need(isinstance(md, str) and md.startswith("cuda:"), f"model_device {md!r} is not cuda:N")
    need(rt.get("input_devices") == [md], f"input_devices {rt.get('input_devices')!r} != [{md!r}]")
    need(rt.get("batch_size") == DL17_BATCH_SIZE, f"batch_size {rt.get('batch_size')!r} != {DL17_BATCH_SIZE}")
    need(rt.get("num_workers") == DL17_NUM_WORKERS,
         f"num_workers {rt.get('num_workers')!r} != {DL17_NUM_WORKERS}")
    need(rt.get("forward_batches") == DL17_FORWARD_BATCHES,
         f"forward_batches {rt.get('forward_batches')!r} != {DL17_FORWARD_BATCHES}")
    rows = s.get("dataset", {}).get("actual_rows")
    need(rows == DL17_VAL_ROWS, f"actual_rows {rows!r} != {DL17_VAL_ROWS}")
    need(rt.get("checkpoint_iteration") == DL17_CHECKPOINT_ITER,
         f"checkpoint_iteration {rt.get('checkpoint_iteration')!r} != {DL17_CHECKPOINT_ITER}")
    need(rt.get("checkpoint_best_val_miou_all_class") == DL17_REFERENCE_MIOU,
         f"checkpoint_best_val_miou_all_class {rt.get('checkpoint_best_val_miou_all_class')!r} "
         f"!= {DL17_REFERENCE_MIOU!r}")
    need(rt.get("image_digest") == DL17_IMAGE_DIGEST,
         f"image_digest {rt.get('image_digest')!r} != {DL17_IMAGE_DIGEST}")
    gt = s.get("per_class", {}).get("gt_support")
    total = sum(gt) if isinstance(gt, list) and all(isinstance(x, int) for x in gt) else None
    need(total == DL17_GT_SUPPORT, f"sum(per_class.gt_support) {total!r} != {DL17_GT_SUPPORT}")
    need(run.get("checkpoint_sha256") == DL17_CHECKPOINT_SHA256,
         f"run.checkpoint_sha256 {run.get('checkpoint_sha256')!r} != {DL17_CHECKPOINT_SHA256}")
    gpu = rt.get("gpu_name")
    need(gpu == DL17_GPU_NAME, f"gpu_name {gpu!r} != {DL17_GPU_NAME!r} (exact match)")
    return out


def _miou(s: dict):
    return s.get("dataset_level", {}).get("all_class_miou")


def compare(a: Path, b: Path, *, identity_only: bool = False) -> int:
    problems, sa, sb = identity_problems(a, b)
    rid_a, rid_b = sa.get("run", {}).get("run_id"), sb.get("run", {}).get("run_id")
    if rid_a is not None and rid_a == rid_b:
        print(f"RESULT: NO VERDICT -- A and B carry the same run.run_id {rid_a!r}: a copy of one run is not "
              f"two runs; DL-17 compares two fresh runs")
        return EXIT_USAGE
    print(f"all_class_miou A = {_miou(sa)!r}")
    print(f"all_class_miou B = {_miou(sb)!r}")
    if identity_only:
        for p in problems:
            print(f"IDENTITY: {p}")
        print("RESULT: IDENTICAL" if not problems else "RESULT: NOT IDENTICAL")
        return EXIT_PASS if not problems else EXIT_FAIL

    # Validity is judged first whenever both artifacts verified: a run that is not a DL-17 run is
    # never reported as a DL-17 FAIL. An artifact that fails verify_artifact fails identity.
    if not any(p.startswith("verify_artifact(") for p in problems):
        invalid = validity_problems("A", sa) + validity_problems("B", sb)
        if invalid:
            for p in invalid:
                print(f"VALIDITY: {p}")
            print("RESULT: VALIDITY VIOLATION -- not a DL-17 run (STOP and report)")
            return EXIT_VALIDITY
    for p in problems:
        print(f"IDENTITY: {p}")
    if problems:
        print("RESULT: FAIL (the two runs are not identical)")
        return EXIT_FAIL

    miou = _miou(sa)
    delta = abs(miou - DL17_REFERENCE_MIOU)
    print(f"|all_class_miou - {DL17_REFERENCE_MIOU!r}| = {delta!r} (band {DL17_BAND!r})")
    if delta > DL17_BAND:
        print("RESULT: FAIL (outside the DL-17 band)")
        return EXIT_FAIL
    if delta > DL17_RECORD_ABOVE:
        print(f"RECORDED: difference {delta!r} lies in ({DL17_RECORD_ABOVE!r}, {DL17_BAND!r}]")
    print("RESULT: PASS (identical and within the DL-17 band)")
    return EXIT_PASS


class _Parser(argparse.ArgumentParser):
    """Every argparse-driven exit (usage error or --help) is EXIT_USAGE: never 0 (PASS) or 2 (VALIDITY)."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)

    def exit(self, status=0, message=None):
        if message:
            sys.stderr.write(message)
        raise SystemExit(EXIT_USAGE)


def main(argv=None) -> int:
    p = _Parser(description="DL-17 comparator (EVALUATION_CONTRACT section 10).")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--identity-only", action="store_true",
                   help="apply the identity rules alone (exit 0/1); used by the Q12 rehearsal and the smoke")
    args = p.parse_args(argv)
    a, b = Path(args.run_a), Path(args.run_b)
    try:
        if a.resolve() == b.resolve():
            print(f"error: RUN_A and RUN_B are the same directory ({a.resolve()}); DL-17 compares two runs",
                  file=sys.stderr)
            return EXIT_USAGE
        return compare(a, b, identity_only=args.identity_only)
    except Exception as e:                                # noqa: BLE001 -- a crash is never PASS/FAIL/VALIDITY
        print(f"RESULT: ERROR ({type(e).__name__}: {e})")
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
