#!/usr/bin/env python3
"""Build the frozen corruption cache. Generation only — never model evaluation.

Guarded like every real-effect entry point. It is NOT run against PlantSeg in the vendoring task:
official cache generation additionally requires the Pillow and scikit-image versions to be frozen in
the official evaluation environment (see the dependency-gap note below), which has not happened yet.
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
                                  CorruptionCacheError, generate, runtime_versions)
from src.vendor.imagecorruptions import CORRUPTION_IDS, GENERATED_SEVERITIES  # noqa: E402
from src.vendor.imagecorruptions.provenance import verify_vendor_integrity  # noqa: E402

# Official generation prerequisite: the corruption bytes depend on Pillow (jpeg_compression) and
# scikit-image (brightness), neither of which is version-frozen in the official stack yet.
OFFICIAL_DEPENDENCY_PINS_FROZEN = False


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
        if not OFFICIAL_DEPENDENCY_PINS_FROZEN:
            raise CorruptionCacheError(
                "official_dependency_pins_unfrozen",
                "OFFICIAL CORRUPTION CACHE GENERATION BLOCKED: the corrupted bytes depend on "
                f"Pillow and scikit-image (current runtime {runtime_versions()}), and neither is "
                "pinned in the official evaluation environment. Freeze those versions before "
                "generating the cache every stage will be scored against.")
    except CorruptionCacheError as e:
        print(f"CACHE GENERATION FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2

    print(json.dumps({"grid": {"corruptions": list(CORRUPTION_IDS),
                               "severities": list(GENERATED_SEVERITIES),
                               "expected_images": OFFICIAL_EXPECTED_IMAGES,
                               "manifest": MANIFEST_NAME}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
