"""TRAIN/VAL-only staged data root check (M11, B60 §5; B62).

The official teacher, E2 and E3 run on a data root staged with TRAIN and VAL only. This check
fails closed if any TEST surface exists in that root, and verifies the TRAIN and VAL pair counts.

TEST surfaces are tested for EXISTENCE ONLY (`os.path.lexists` on three exact paths). Nothing under
them is opened, read, listed, counted or hashed. TRAIN and VAL are counted by file name only; no
image or mask is opened. Standard library only.
"""

from __future__ import annotations

import os
from pathlib import Path

TEST_SURFACES = ("images/test", "annotations/test", "annotation_test.json")
TRAINVAL_SPLITS = ("train", "val")
DEFAULT_EXPECTED_COUNTS = {"train": 5367, "val": 846}      # configs/data.py SPLIT_SIZES [counted]
IMAGE_SUFFIXES = (".jpg", ".jpeg")
MASK_SUFFIX = ".png"


class TrainValIsolationError(RuntimeError):
    """A named isolation failure; `code` makes each guard independently testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def assert_no_test_surfaces(root: Path) -> dict:
    """Fail closed if any exact TEST surface exists under `root`. Existence check only."""
    present = [rel for rel in TEST_SURFACES if os.path.lexists(os.path.join(root, *rel.split("/")))]
    if present:
        raise TrainValIsolationError(
            "test_split_present",
            f"the staged data root {root} contains TEST surface(s) {present}. Official teacher, E2 and "
            "E3 roots must be staged with TRAIN and VAL only (M11). Nothing under them was opened, "
            "listed or counted.")
    return {rel: "absent" for rel in TEST_SURFACES}


def count_trainval(root: Path, expected: dict | None = None) -> dict:
    """Count TRAIN and VAL images and masks by name and require the locked counts."""
    expected = DEFAULT_EXPECTED_COUNTS if expected is None else expected
    counts: dict[str, dict[str, int]] = {}
    for split in TRAINVAL_SPLITS:
        img_dir, mask_dir = root / "images" / split, root / "annotations" / split
        for d in (img_dir, mask_dir):
            if not d.is_dir():
                raise TrainValIsolationError("split_dir_missing", f"missing split directory: {d}")
        n_img = sum(1 for e in os.scandir(img_dir)
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_SUFFIXES)
        n_mask = sum(1 for e in os.scandir(mask_dir)
                     if e.is_file() and os.path.splitext(e.name)[1] == MASK_SUFFIX)
        if n_img != expected[split] or n_mask != expected[split]:
            raise TrainValIsolationError(
                "split_count_mismatch",
                f"split {split}: {n_img} images / {n_mask} masks, expected {expected[split]} of each")
        counts[split] = {"images": n_img, "masks": n_mask}
    return counts


def assert_trainval_only_root(root: str | Path, expected: dict | None = None) -> dict:
    """The M11 gate: TEST surfaces absent first, then TRAIN/VAL counts.

    No TEST content is opened, read, listed, counted or hashed. The helper performs only fail-closed
    existence checks on the three approved TEST surface paths.
    """
    root = Path(root)
    if not root.is_dir():
        raise TrainValIsolationError("data_root_missing", f"data root does not resolve: {root}")
    surfaces = assert_no_test_surfaces(root)
    counts = count_trainval(root, expected)
    return {"counts": counts, "test_surfaces": surfaces,
            "method": "os.path.lexists on exact TEST paths (no enumeration); TRAIN/VAL counted by name"}
