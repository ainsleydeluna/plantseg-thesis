#!/usr/bin/env python3
"""Smoke test for the PlantSeg student dataloaders (no training).

Pulls one TRAIN batch and one VAL batch (batch_size=2) and asserts shapes/dtypes/label
range. Prints the sorted unique non-255 labels + count per batch. This FEEDS — but does
NOT replace — the full np.unique() audit in Table 3.1 (reports/dataset_report.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.seeds import set_seed  # noqa: E402
from src.data import NUM_CLASSES, build_dataloader  # noqa: E402


def check_batch(name: str, img: torch.Tensor, mask: torch.Tensor) -> None:
    print(f"\n[{name}] image: shape={tuple(img.shape)} dtype={img.dtype} | "
          f"mask: shape={tuple(mask.shape)} dtype={mask.dtype}")

    assert tuple(img.shape) == (2, 3, 512, 512), f"{name} image shape {tuple(img.shape)}"
    assert img.dtype == torch.float32, f"{name} image dtype {img.dtype}"
    assert tuple(mask.shape) == (2, 512, 512), f"{name} mask shape {tuple(mask.shape)}"
    assert mask.dtype == torch.int64, f"{name} mask dtype {mask.dtype}"
    assert img.shape[-2:] == mask.shape[-2:] == (512, 512), f"{name} H,W mismatch"

    uniq = torch.unique(mask)
    non_ignore = uniq[uniq != 255]
    bad = non_ignore[(non_ignore < 0) | (non_ignore > NUM_CLASSES - 1)]
    assert bad.numel() == 0, f"{name} mask has out-of-range labels: {bad.tolist()}"

    labels = sorted(int(v) for v in non_ignore.tolist())
    print(f"[{name}] unique non-255 labels ({len(labels)}): {labels}")
    print(f"[{name}] contains 255 (ignore/pad): {bool((uniq == 255).any())}")


def main() -> int:
    set_seed()
    print(f"num_classes (configs/data.py) = {NUM_CLASSES}  -> valid label range [0, {NUM_CLASSES - 1}] U {{255}}")

    train_img, train_mask = next(iter(build_dataloader("train", batch_size=2)))
    check_batch("TRAIN", train_img, train_mask)

    val_img, val_mask = next(iter(build_dataloader("val", batch_size=2)))
    check_batch("VAL", val_img, val_mask)

    print("\n[PASS] dataloader smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
