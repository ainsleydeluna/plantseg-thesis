#!/usr/bin/env python3
"""Build the frozen corruption cache. Generation only — never model evaluation.

Guarded like every real-effect entry point, and additionally gated on the corruption dependency
environment in TWO separate phases:

  1. versions REGISTERED  -- the exact python/numpy/Pillow/scikit-image pins are recorded in
     `src.corruption_cache` (values taken from `requirements.lock`). DONE.
  2. environment EXECUTABLY VALIDATED -- that exact stack has been installed and the deterministic
     corruption verification re-run under it. NOT DONE.

Generation refuses until BOTH hold, and the refusal names which one is missing. Writing version
numbers down is not evidence that the corrupted bytes reproduce under them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.corruption_cache import (MANIFEST_NAME, OFFICIAL_EXPECTED_IMAGES,  # noqa: E402
                                  CorruptionCacheError, generate, official_dependency_pins,
                                  require_official_dependency_environment, runtime_versions)
from src.vendor.imagecorruptions import CORRUPTION_IDS, GENERATED_SEVERITIES  # noqa: E402
from src.vendor.imagecorruptions.provenance import verify_vendor_integrity  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the frozen PlantSeg corruption cache.")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--split", default="test", choices=["test"])
    args = p.parse_args(argv)

    if not (args.real_run and args.confirm_real_run):
        print("REFUSING: cache generation requires --real-run --confirm-real-run. Nothing was "
              "read or written.", file=sys.stderr)
        return 2
    try:
        verify_vendor_integrity()
        if not args.cache_dir:
            raise CorruptionCacheError("cache_dir_unset", "--cache-dir is required")
        out = Path(args.cache_dir).resolve()
        if REPO == out or REPO in out.parents:
            raise CorruptionCacheError("cache_inside_repo",
                                       "the corruption cache is never written inside the repository")
        # Two-phase gate: versions are REGISTERED, but the environment is not yet EXECUTABLY
        # VALIDATED, so generation still refuses — and says which of the two is missing.
        require_official_dependency_environment()
    except CorruptionCacheError as e:
        print(f"CACHE GENERATION FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2

    print(json.dumps({"grid": {"corruptions": list(CORRUPTION_IDS),
                               "severities": list(GENERATED_SEVERITIES),
                               "expected_images": OFFICIAL_EXPECTED_IMAGES,
                               "manifest": MANIFEST_NAME},
                      "dependency_pins": official_dependency_pins(),
                      "runtime": runtime_versions()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
