"""PlantSegDataset + build_dataloader for the student.

Reads JPG image + grayscale-PNG mask pairs from the on-disk, pre-partitioned folders
(images/<split>/ + annotations/<split>/) — the folder layout is the split source of
truth; annotation_*.json is NOT parsed for split membership. Config (root, num_classes)
comes from configs/data.py; augmentation params from configs/augment.py.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Make the repo root importable (configs/ and src/ are resolvable when run from anywhere).
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.augment import AUGMENT          # noqa: E402
from configs.data import DATA                # noqa: E402
from src.seeds import SEED                   # noqa: E402

from .transforms import core_preprocess, finalize, train_preprocess  # noqa: E402

NUM_CLASSES = DATA["num_classes"]
_SPLITS = ("train", "val", "test")


class PlantSegDataset(Dataset):
    def __init__(self, split: str):
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        root = Path(DATA["root"])
        if not root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        self.split = split
        img_dir, mask_dir = root / "images" / split, root / "annotations" / split
        self.pairs = []
        for img in sorted(img_dir.glob("*.jpg")):
            mask = mask_dir / f"{img.stem}.png"
            if mask.exists():
                self.pairs.append((img, mask))
        if not self.pairs:
            raise RuntimeError(f"no image/mask pairs found for split={split} under {root}")
        self.aug_params = AUGMENT

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            if self.split == "train":
                rng = np.random.RandomState((SEED + idx) % (2 ** 32))  # deterministic per-sample aug
                # true train-time multi-scale RRC on the original-resolution image (pre-pad)
                img_np, mask_np = train_preprocess(im, mk, rng, self.aug_params)
            else:
                img_np, mask_np = core_preprocess(im, mk)    # val/test: unchanged deterministic core
        return finalize(img_np, mask_np)                      # float32 CHW, int64 HW


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def build_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42)."""
    dataset = PlantSegDataset(split)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
        drop_last=False,
    )
