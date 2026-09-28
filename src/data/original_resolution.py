"""Original-resolution PlantSeg pairs for the upstream evaluation protocol (lane L-AM13).

`PlantSegDataset` is not modified (docs/EVALUATION_CONTRACT.md section 6). This class reuses its pairing
by composition -- the name-sorted image list, the missing-mask guard and the locked split-count guard --
and never calls its `__getitem__`. Each pair is returned at its ORIGINAL resolution: the image
EXIF-transposed and converted to RGB exactly as `core_preprocess` does, the mask as raw class indices,
never resized, remapped or padded. Evaluation splits only; there is no augmentation path here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .dataset import PlantSegDataset

_EVAL_SPLITS = ("val", "test")


class PlantSegOriginalResolutionDataset(Dataset):
    """(uint8 HxWx3 RGB image, int64 HxW mask) per pair, at the original resolution."""

    def __init__(self, split: str):
        if split not in _EVAL_SPLITS:
            raise ValueError(f"split must be one of {_EVAL_SPLITS}, got {split!r}")
        self.base = PlantSegDataset(split)       # listing and guards only; never indexed
        self.split = split
        self.pairs = self.base.pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_path, mask_path = self.pairs[idx]
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            image = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(mk).astype(np.int64)     # raw class indices (no palette expansion)
        if mask.ndim != 2:
            raise ValueError(f"{mask_path.name}: mask must be single-channel, got shape {mask.shape}")
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"{img_path.stem}: image {image.shape[:2]} (after EXIF transpose) and mask "
                f"{mask.shape} differ; the upstream protocol scores the original mask and never "
                "resizes it")
        return image, mask
