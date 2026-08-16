#!/usr/bin/env python3
"""Corrupted-condition evaluation entry point for teacher and E1-E7. Validates, then would run.

Reuses the clean evaluator's completed model resolution (`src.eval.stage_artifacts`) — there is no
second checkpoint-loading system here — and the frozen corruption vocabulary from
`src.stats.corruption_protocol`. Metric reducers and artifact finalisation are untouched.

Corrupted inputs are READ from the frozen shared corruption cache (PATH B) and are never regenerated
here. Direct deterministic generation (PATH A) belongs to `scripts/build_corruption_cache.py` alone.
That split is what makes the teacher and E1-E7 score byte-identical corrupted pixels rather than
per-model stochastic realisations, so this module deliberately does not import
`real_corruption_transform` at all.

The cached payload is already resized, corrupted, uint8 RGB. Evaluation therefore continues with
mean-value padding -> tensor -> ImageNet normalization -> model: no second resize, no second
corruption, and no mask ever enters the corruption/cache path.

Everything before inference — authorization, model source, vocabulary, severity role, test-split
protection, output location, cache provenance — is complete and enforced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.corruption_cache import CorruptionCacheError, load_manifest  # noqa: E402
from src.eval.robustness import (RobustnessError, load_cached_corruption,  # noqa: E402
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
    # The ONE cache input. Everything else about the cache (grid, seed policy, vendor identity,
    # per-item digests) is read from the committed manifest rather than duplicated as CLI flags.
    p.add_argument("--corruption-cache", required=True,
                   help="root of the frozen shared corruption cache (must contain the manifest)")
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


def resolve_corruption_cache(cache_root, corruption: str, severity: int, *, split: str,
                             official: bool) -> dict:
    """Validate the shared cache BEFORE any corrupted-image inference. Never regenerates.

    `load_manifest` already enforces the manifest schema, the canonical manifest checksum, the seed
    policy identity and the patched vendor-runtime SHA; anything it rejects raises here. On top of
    that this adds the request-level coverage checks the runner is responsible for: the cache must
    actually be the split being evaluated and must contain the requested corruption and severity.
    """
    root = Path(cache_root).resolve()
    if not root.is_dir():
        raise CorruptionCacheError("cache_root_missing", f"corruption cache not found: {root}")

    manifest = load_manifest(root)          # schema + checksum + seed policy + vendor SHA

    if manifest.get("split") != split:
        raise CorruptionCacheError(
            "cache_split_mismatch",
            f"cache was built for split {manifest.get('split')!r}, evaluating {split!r}")
    if corruption not in manifest.get("corruption_order", []):
        raise CorruptionCacheError(
            "cache_corruption_absent",
            f"{corruption!r} is not in the cached grid {manifest.get('corruption_order')}")
    if severity not in manifest.get("severities", []):
        raise CorruptionCacheError(
            "cache_severity_absent",
            f"severity {severity} is not in the cached severities {manifest.get('severities')}")
    if official and manifest.get("source_image_count") != OFFICIAL_ROWS:
        raise CorruptionCacheError(
            "cache_incomplete_for_official",
            f"an official robustness run requires the full cached test split ({OFFICIAL_ROWS} "
            f"source images), the cache holds {manifest.get('source_image_count')}")
    return manifest


def cached_image_loader(cache_root, manifest: dict, corruption: str, severity: int):
    """PATH B accessor: `image_id -> uint8 RGB`, hash-verified on every read.

    This is the ONLY way this runner obtains a corrupted image, and it is identical for the teacher
    and E1-E7, so every stage scores the same frozen realisation. It takes no mask: ground truth
    never enters the corruption/cache path.
    """
    root = Path(cache_root).resolve()

    def _load(image_id: str):
        return load_cached_corruption(root, manifest, image_id, corruption, severity)

    return _load


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

        # ---- PATH B: the shared cache is proven valid BEFORE any corrupted-image inference ----
        manifest = resolve_corruption_cache(args.corruption_cache, args.corruption, args.severity,
                                            split=args.split,
                                            official=args.artifact_status == "official")
        load_corrupted = cached_image_loader(args.corruption_cache, manifest, args.corruption,
                                             args.severity)
    except (CorruptionCliError, RobustnessError, CorruptionCacheError, Exception) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)   # reported, not swallowed
        return 2

    print(f"validated {args.stage} under {condition} against cache "
          f"{manifest['canonical_manifest_sha256'][:16]}… "
          f"({manifest['source_image_count']} source images, loader={load_corrupted.__name__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
