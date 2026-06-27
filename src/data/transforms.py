"""Core preprocessing + train-only augmentation for the PlantSeg student dataloader.

Core (configs/data.py) — applied IDENTICALLY to train/val/test:
  EXIF-transpose image (never mask) -> aspect-ratio-preserving resize long side->512
  (bilinear image / nearest mask) -> symmetric pad to 512x512 (image [124,116,104],
  mask 255) -> scale [0,1] -> ImageNet normalize -> float32 CHW image / int64 HW mask.
Masks are read as RAW class-index arrays (PIL 'L'/'P', no RGB/palette expansion).

Train augmentation (configs/augment.py) on top of core, in uint8 space before normalize:
joint hflip / vflip / rotation (image fill = mean, mask fill = 255) + image-only hue/sat.
(The multi-scale random-resized-crop in augment.py is out of scope for this loader per the
task spec, which lists "joint geometric flips/rotation + image-only hue/sat".)

No Albumentations dependency — params come from configs/augment.py; ops use numpy/PIL/torch
under a per-sample seeded RNG for determinism.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps

SIZE = 512
MASK_IGNORE = 255
IMAGENET_MEAN_8BIT = (124, 116, 104)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_keep_ratio(img: Image.Image, mask: Image.Image):
    w, h = img.size
    if w >= h:
        nw, nh = SIZE, max(1, round(h * SIZE / w))
    else:
        nw, nh = max(1, round(w * SIZE / h)), SIZE
    return img.resize((nw, nh), Image.BILINEAR), mask.resize((nw, nh), Image.NEAREST)


def _pad_to_square(img_np: np.ndarray, mask_np: np.ndarray):
    h, w = img_np.shape[:2]
    top, left = (SIZE - h) // 2, (SIZE - w) // 2
    canvas = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    canvas[:] = IMAGENET_MEAN_8BIT
    canvas[top:top + h, left:left + w] = img_np
    mcanvas = np.full((SIZE, SIZE), MASK_IGNORE, dtype=np.int64)
    mcanvas[top:top + h, left:left + w] = mask_np
    return canvas, mcanvas


def core_preprocess(img_pil: Image.Image, mask_pil: Image.Image):
    """Return (uint8 HWC 512x512 image, int64 HW 512x512 mask) — geometric core, pre-normalize."""
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")  # EXIF on image only; RGB 3ch
    img_r, mask_r = _resize_keep_ratio(img_pil, mask_pil)
    img_np = np.asarray(img_r, dtype=np.uint8)                 # HWC uint8
    mask_np = np.asarray(mask_r).astype(np.int64)              # HW raw class indices (no expansion)
    return _pad_to_square(img_np, mask_np)


def _jitter_hue_sat(img_np: np.ndarray, hue_delta: float, sat_factor: float) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(img_np).convert("HSV"), dtype=np.float32)  # H,S,V in 0-255
    hsv[..., 0] = (hsv[..., 0] + hue_delta * 255.0) % 256.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0.0, 255.0)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"), dtype=np.uint8)


def augment(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
    """Train-only aug on uint8 image + int64 mask (both 512x512). Params from configs/augment.py."""
    if rng.rand() < p["horizontal_flip_p"]:
        img_np, mask_np = img_np[:, ::-1, :].copy(), mask_np[:, ::-1].copy()
    if rng.rand() < p["vertical_flip_p"]:
        img_np, mask_np = img_np[::-1, :, :].copy(), mask_np[::-1, :].copy()

    rot = p["rotation"]
    if rng.rand() < rot["p"]:
        angle = float(rng.uniform(-rot["degrees"], rot["degrees"]))
        img_np = np.asarray(
            Image.fromarray(img_np).rotate(angle, resample=Image.BILINEAR, fillcolor=IMAGENET_MEAN_8BIT),
            dtype=np.uint8)
        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L").rotate(
            angle, resample=Image.NEAREST, fillcolor=MASK_IGNORE)
        mask_np = np.asarray(mask_pil).astype(np.int64)

    photo = p["photometric"]
    if rng.rand() < photo["p"]:
        hue_delta = float(rng.uniform(-photo["hue"], photo["hue"]))
        lo, hi = photo["saturation_factor"]
        sat_factor = float(rng.uniform(lo, hi))
        img_np = _jitter_hue_sat(img_np, hue_delta, sat_factor)

    return img_np, mask_np


def finalize(img_np: np.ndarray, mask_np: np.ndarray):
    """uint8 HWC image + int64 HW mask -> (float32 CHW image tensor, int64 HW mask tensor)."""
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD          # per-channel, broadcast over HxW
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # CHW
    img_t = torch.from_numpy(img).float()
    mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).long()
    return img_t, mask_t
