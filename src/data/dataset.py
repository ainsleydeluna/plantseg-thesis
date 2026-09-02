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
from configs.data import DATA, SPLIT_SIZES, SPLIT_TOTAL  # noqa: E402
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
        # Collect images case-insensitively for .jpg/.jpeg (consistent with
        # scripts/verify_plantseg_dataset.py); deterministic order via name sort.
        imgs = sorted((p for p in img_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
                      key=lambda p: p.name)
        self.pairs, missing = [], []
        for img in imgs:
            mask = mask_dir / f"{img.stem}.png"
            if mask.exists():
                self.pairs.append((img, mask))
            else:
                missing.append(img.name)                       # do NOT silently drop
        # Missing masks must FAIL LOUD, not silently reduce the sample count.
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) in split={split} have no matching mask under {mask_dir} "
                f"(expected <stem>.png), e.g. {missing[:5]}")
        if not self.pairs:
            raise RuntimeError(f"no image/mask pairs found for split={split} under {root}")
        # Guard against silent under/over-count vs the locked PlantSeg split sizes (configs/data.py).
        # An unregistered split is a HARD error: a missing/renamed key must never silently disable
        # this guard (the previous `.get()` chain defaulted to None and skipped the check entirely).
        if split not in SPLIT_SIZES:
            raise RuntimeError(
                f"split={split!r} has no registered expected count in configs/data.py SPLIT_SIZES "
                f"(registered: {sorted(SPLIT_SIZES)}); refusing to load an unverified split")
        expected = SPLIT_SIZES[split]
        if len(self.pairs) != expected:
            raise RuntimeError(
                f"split={split} pair count MISMATCH: expected {expected}, actual {len(self.pairs)} "
                f"(delta {len(self.pairs) - expected:+d}). All splits expected {dict(SPLIT_SIZES)}, "
                f"total {SPLIT_TOTAL}. Check the dataset upload/extraction under {root}.\n"
                f"NOTE: the COUNTED values in configs/data.py SPLIT_SIZES are AUTHORITATIVE. ch3's "
                f"5,442/778/1,554 is a known arithmetic artifact (nominal 70/10/20 applied to 7,774) "
                f"and is under correction in the manuscript — do NOT edit SPLIT_SIZES to match it.")
        self.aug_params = AUGMENT

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            if self.split == "train":
                # Per-sample augmentation RNG seeded from the worker/global RNG stream — reseeded per
                # worker per epoch by worker_init_fn (num_workers>0) or advancing in the seeded main
                # process (num_workers=0). Augmentation VARIES across epochs/repeats yet stays
                # reproducible from the global seed (previously seeded by idx alone -> frozen per image).
                rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
                # true train-time multi-scale RRC on the original-resolution image (pre-pad)
                img_np, mask_np = train_preprocess(im, mk, rng, self.aug_params)
            else:
                img_np, mask_np = core_preprocess(im, mk)    # val/test: unchanged deterministic core
        return finalize(img_np, mask_np)                      # float32 CHW, int64 HW


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


PREFETCH_FACTOR = 4          # batches pre-staged per worker (B31-5)


def build_dataloader(split: str, batch_size: int, num_workers: int = 0,
                     persistent_workers: bool = False) -> DataLoader:
    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42).

    `persistent_workers` is opt-in and intended for the TRAIN loader only: it keeps the worker pool
    (and its decoded-image buffers) alive for the whole run, which is worth it across 80k iterations
    but not across the 20 validation passes. `prefetch_factor`/`persistent_workers` are only legal
    when num_workers > 0 — PyTorch raises otherwise, and the dry-run path uses num_workers=0 — so
    both are passed conditionally. `pin_memory` is gated on CUDA: it is a no-op without a device and
    emits a warning, so gating keeps the CPU dry-run output clean.
    """
    dataset = PlantSegDataset(split)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    extra = {}
    if num_workers > 0:
        extra["prefetch_factor"] = PREFETCH_FACTOR
        extra["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
        # TRAIN only: 5367 % 16 = 7, so every epoch would otherwise end on a ragged 7-sample batch,
        # perturbing BatchNorm statistics and the Dice term's per-batch class-presence set.
        # val/test keep drop_last=False — dropping evaluation samples would corrupt the metric.
        drop_last=(split == "train"),
        pin_memory=torch.cuda.is_available(),
        **extra,
    )
