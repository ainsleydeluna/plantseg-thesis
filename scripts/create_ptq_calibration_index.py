#!/usr/bin/env python3
"""Create the frozen PTQ calibration-index artifact shared UNCHANGED by E4 and E7.

Contract B4: 128 images, seed 42, from the **training partition**, identifiers persisted as a fixed
list, same subset for E4 and E7. This utility enumerates TRAIN identifiers only, sorts them into the
canonical order before sampling (so filesystem enumeration order can never influence the subset),
delegates the selection to `src.quant.calibration`, writes the artifact OUTSIDE the repository and
prints its checksum — which E7 later pins via `--calibration-sha256`.

Guarded like every other real-effect entry point: it refuses without `--real-run --confirm-real-run`
before touching the dataset, and it never reads the validation or test splits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant.calibration import build_calibration_index, save_calibration_index  # noqa: E402
from src.quant.runner import QuantRunError, check_output_dir  # noqa: E402

DATA_ROOT_ENV = "PLANTSEG_DATA_ROOT"
IMAGE_SUFFIXES = (".jpg", ".jpeg")
EXPECTED_TRAIN = 5367


def enumerate_train_ids(root: Path) -> list[str]:
    """Return TRAIN image stems only. The val/test directories are never listed."""
    img_dir = root / "images" / "train"
    if not img_dir.is_dir():
        raise QuantRunError("train_dir_missing", f"TRAIN image directory not found: {img_dir}")
    ids = [os.path.splitext(e.name)[0] for e in os.scandir(img_dir)
           if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_SUFFIXES]
    if len(ids) != EXPECTED_TRAIN:
        raise QuantRunError("train_count_mismatch",
                            f"TRAIN split holds {len(ids)} images, expected {EXPECTED_TRAIN}")
    return sorted(ids)          # canonical order; build_calibration_index re-sorts defensively


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Create the frozen E4/E7 PTQ calibration index.")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--data-root", default=None, help=f"overrides ${DATA_ROOT_ENV}")
    p.add_argument("--out-dir", default=None, help="destination outside the repository")
    p.add_argument("--filename", default="e4_e7_ptq_calibration_index.json")
    args = p.parse_args(argv)

    if not args.real_run:
        print("REFUSING: creating the frozen calibration artifact requires --real-run "
              "--confirm-real-run. The dataset was not read.", file=sys.stderr)
        return 2
    if not args.confirm_real_run:
        print("REFUSING: --real-run requires --confirm-real-run.", file=sys.stderr)
        return 2

    try:
        out_dir = check_output_dir(args.out_dir, create=True)
        raw = args.data_root or os.environ.get(DATA_ROOT_ENV, "")
        if not raw:
            raise QuantRunError("data_root_unset", f"{DATA_ROOT_ENV} is not set")
        root = Path(raw)
        if not root.is_dir():
            raise QuantRunError("data_root_missing", f"{DATA_ROOT_ENV} does not resolve: {root}")
        ids = enumerate_train_ids(root)
        index = build_calibration_index(ids)
        path = save_calibration_index(index, out_dir / args.filename)
    except QuantRunError as e:
        print(f"CALIBRATION INDEX FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2

    print(json.dumps({"artifact": str(path), "count": index["count"], "seed": index["seed"],
                      "split": index["split"], "shared_by": index["shared_by"],
                      "candidate_pool_size": index["candidate_pool_size"],
                      "checksum_sha256": index["checksum_sha256"]}, indent=2))
    print(f"\nFrozen calibration artifact written to {path}")
    print(f"Pin this checksum for E7:  --calibration-sha256 {index['checksum_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
