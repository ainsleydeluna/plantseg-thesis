"""Frozen PTQ calibration-subset identity, shared by E4 and E7.

Contract B4 (PTQ): "**~128 images**, sampled with **seed 42** from the training partition; no
augmentation; same preprocessing as clean test; identifiers persisted as a fixed list; **same subset
for E4 and E7**." `configs/quant.py` pins `calibration_num_images=128`, `calibration_seed=42`,
`calibration_source="training partition"`, `calibration_subset_shared=("E4", "E7")`.

REPRODUCIBILITY RULE enforced here: candidate identifiers are placed into a **canonical sorted
order before sampling**, so the selected subset can never depend on filesystem enumeration order,
glob order, or the order a caller happened to build its list in. Two callers that see the same set
of train identifiers always produce the same 128.

This module never touches the dataset — it operates on identifier strings only.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

CALIBRATION_SCHEMA = "plantseg-ptq-calibration-index/1.0.0"
CALIBRATION_COUNT = 128
CALIBRATION_SEED = 42
CALIBRATION_SPLIT = "train"
SHARED_BY = ("E4", "E7")


class CalibrationIndexError(ValueError):
    """Raised when a calibration index cannot be built or verified."""


def _checksum(selected: Sequence[str]) -> str:
    """SHA-256 over the canonical newline-joined selected-ID representation."""
    return hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()


def build_calibration_index(candidate_ids: Iterable[str], *, split: str = CALIBRATION_SPLIT,
                            count: int = CALIBRATION_COUNT,
                            seed: int = CALIBRATION_SEED) -> dict:
    """Deterministically select `count` calibration identifiers from the TRAIN split.

    Rejects a non-train split, duplicates, empty/blank ids and an insufficient candidate pool.
    """
    if split != CALIBRATION_SPLIT:
        raise CalibrationIndexError(
            f"calibration must come from the {CALIBRATION_SPLIT!r} partition, got {split!r}. "
            "Validation and test identifiers are never used for PTQ calibration.")
    ids = list(candidate_ids)
    if any((not isinstance(i, str)) or not i.strip() for i in ids):
        raise CalibrationIndexError("candidate identifiers must be non-empty strings")
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise CalibrationIndexError(f"duplicate candidate identifiers: {dupes[:5]}")
    if len(ids) < count:
        raise CalibrationIndexError(
            f"need at least {count} train identifiers to calibrate, got {len(ids)}")

    canonical = sorted(ids)                      # canonical order BEFORE sampling
    selected = random.Random(seed).sample(canonical, count)
    return {
        "schema": CALIBRATION_SCHEMA,
        "split": split,
        "seed": seed,
        "count": count,
        "shared_by": list(SHARED_BY),
        "candidate_pool_size": len(canonical),
        "selected_ids": selected,
        "checksum_sha256": _checksum(selected),
    }


def verify_calibration_index(index: dict, *, expected_count: int = CALIBRATION_COUNT) -> None:
    """Validate a loaded calibration index; raises `CalibrationIndexError` on any mismatch."""
    for key in ("schema", "split", "seed", "count", "selected_ids", "checksum_sha256"):
        if key not in index:
            raise CalibrationIndexError(f"calibration index missing {key!r}")
    if index["schema"] != CALIBRATION_SCHEMA:
        raise CalibrationIndexError(f"unknown calibration schema {index['schema']!r}")
    if index["split"] != CALIBRATION_SPLIT:
        raise CalibrationIndexError(
            f"calibration index split is {index['split']!r}, expected {CALIBRATION_SPLIT!r}")
    if index["seed"] != CALIBRATION_SEED:
        raise CalibrationIndexError(f"calibration seed is {index['seed']}, expected "
                                    f"{CALIBRATION_SEED}")
    selected = index["selected_ids"]
    if len(selected) != expected_count or index["count"] != expected_count:
        raise CalibrationIndexError(
            f"calibration index holds {len(selected)} ids (count={index['count']}), expected "
            f"{expected_count}")
    if len(set(selected)) != len(selected):
        raise CalibrationIndexError("calibration index contains duplicate identifiers")
    if _checksum(selected) != index["checksum_sha256"]:
        raise CalibrationIndexError("calibration index checksum does not match its selected ids")


def save_calibration_index(index: dict, path: str | Path) -> Path:
    """Persist the index as JSON (outside the repository, alongside the run artifacts)."""
    verify_calibration_index(index)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return p


def load_calibration_index(path: str | Path) -> dict:
    """Load and validate a persisted calibration index — the E4/E7 shared-subset entry point."""
    p = Path(path)
    if not p.is_file():
        raise CalibrationIndexError(f"calibration index not found: {p}")
    index = json.loads(p.read_text(encoding="utf-8"))
    verify_calibration_index(index)
    return index
