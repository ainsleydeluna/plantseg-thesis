#!/usr/bin/env python3
"""Corrupted-condition evaluation entry point for teacher and E1-E7. Validates, then would run.

Reuses the clean evaluator's completed model resolution (`src.eval.stage_artifacts`) — there is no
second checkpoint-loading system here — and the frozen corruption vocabulary from
`src.stats.corruption_protocol`. Metric reducers and artifact finalisation are untouched.

The corruption IMPLEMENTATION is deliberately not vendored yet (see `src/eval/robustness.py`), so a
real corrupted run refuses at the transform gate rather than improvising a definition. Everything
before that gate — authorization, model source, vocabulary, severity role, test-split protection,
output location — is complete and enforced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.robustness import (RobustnessError, real_corruption_transform,  # noqa: E402
                                 registered_corruptions, validate_condition)
from src.eval.stage_artifacts import (OFFICIAL_ROWS, OFFICIAL_SPLIT,  # noqa: E402
                                      official_launch_error, resolve_evaluation_source,
                                      resolve_stage_artifact)

STAGES = ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7")


class CorruptionCliError(RuntimeError):
    """Argument/mode violation, raised before any model or dataset construction."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Corrupted-condition PlantSeg evaluation runner.")
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--corruption", required=True)
    p.add_argument("--severity", required=True, type=int)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--provenance", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--artifact-status", default="smoke",
                   choices=["official", "provisional", "smoke"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--confirm-test-split", action="store_true")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    return p


def validate_args(args) -> dict:
    """Pure validation: no filesystem, no dataset, no model. Returns the condition payload."""
    resolve_stage_artifact(args.stage)
    official = args.artifact_status == "official"
    condition = validate_condition(args.corruption, args.severity, official=official)

    if args.split == OFFICIAL_SPLIT:
        if not args.confirm_test_split:
            raise CorruptionCliError(
                "--split test requires --confirm-test-split (the test split is not touched casually)")
        if args.artifact_status == "smoke":
            raise CorruptionCliError("--split test refuses artifact_status=smoke")
    if official:
        if args.max_samples is not None:
            raise CorruptionCliError(
                "an official robustness artifact forbids any sample cap; the full test split is "
                f"required ({OFFICIAL_ROWS} rows)")
        err = official_launch_error(split=args.split, random_init=False, stage=args.stage,
                                    expected_manifest_rows=OFFICIAL_ROWS,
                                    max_samples=args.max_samples)
        if err:
            raise CorruptionCliError(err)
    return condition


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        condition = validate_args(args)

        # ---- model SOURCE validation, before any dataset work ----
        resolved = resolve_evaluation_source(args.stage, checkpoint=args.checkpoint,
                                             provenance=args.provenance)

        out_dir = Path(args.out_dir).resolve()
        if REPO == out_dir or REPO in out_dir.parents:
            raise CorruptionCliError("artifacts are never written inside the repository")

        # ---- corruption transform gate: not vendored, so this refuses loudly ----
        real_corruption_transform(args.corruption, args.severity)
    except (CorruptionCliError, RobustnessError, Exception) as e:   # reported, not swallowed
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print(f"validated {args.stage} under {condition}")             # unreachable until vendored
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
