"""Core preprocessing + train-only augmentation for the PlantSeg student dataloader.

Core (configs/data.py) — applied IDENTICALLY to train/val/test:
  EXIF-transpose image (never mask) -> aspect-ratio-preserving resize long side->512
  (bilinear image / nearest mask) -> symmetric pad to 512x512 (image [124,116,104],
  mask 255) -> scale [0,1] -> ImageNet normalize -> float32 CHW image / int64 HW mask.
Masks are read as RAW class-index arrays (PIL 'L'/'P', no RGB/palette expansion).

Train path (train_preprocess, configs/augment.py) — true multi-scale RRC on the ORIGINAL-
resolution image, BEFORE the final pad/normalize: EXIF(image) -> RGB -> aspect-preserving
multi-scale resize -> rotation (before crop) -> random 512x512 crop+pad with cat_max_ratio
-> joint flips -> image-only hue/sat. val/test use core_preprocess (unchanged). augment() is
retained as a LEGACY 512x512-canvas helper (flips/rotation/photometric; no RRC crop).

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


# ---- shared train-aug primitives (used by train_preprocess and the legacy augment) ----

def _apply_flips(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
    """Joint horizontal/vertical flips (works on any HxW)."""
    if rng.rand() < p["horizontal_flip_p"]:
        img_np, mask_np = img_np[:, ::-1, :].copy(), mask_np[:, ::-1].copy()
    if rng.rand() < p["vertical_flip_p"]:
        img_np, mask_np = img_np[::-1, :, :].copy(), mask_np[::-1, :].copy()
    return img_np, mask_np


def _apply_rotation(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
    """Joint rotation: image bilinear (fill mean), mask nearest (fill 255). Works on any HxW."""
    rot = p["rotation"]
    if rng.rand() < rot["p"]:
        angle = float(rng.uniform(-rot["degrees"], rot["degrees"]))
        img_np = np.asarray(
            Image.fromarray(img_np).rotate(angle, resample=Image.BILINEAR, fillcolor=IMAGENET_MEAN_8BIT),
            dtype=np.uint8)
        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L").rotate(
            angle, resample=Image.NEAREST, fillcolor=MASK_IGNORE)
        mask_np = np.asarray(mask_pil).astype(np.int64)
    return img_np, mask_np


def _apply_photometric(img_np: np.ndarray, rng: np.random.RandomState, p: dict) -> np.ndarray:
    """Image-only hue/saturation jitter (mask untouched)."""
    photo = p["photometric"]
    if rng.rand() < photo["p"]:
        hue_delta = float(rng.uniform(-photo["hue"], photo["hue"]))
        lo, hi = photo["saturation_factor"]
        sat_factor = float(rng.uniform(lo, hi))
        img_np = _jitter_hue_sat(img_np, hue_delta, sat_factor)
    return img_np


def _resize_long_side(img_pil: Image.Image, mask_pil: Image.Image, target_long: int):
    """Aspect-preserving resize so the LONG side == target_long (bilinear image / nearest mask)."""
    w, h = img_pil.size
    if w >= h:
        nw, nh = target_long, max(1, round(h * target_long / w))
    else:
        nw, nh = max(1, round(w * target_long / h)), target_long
    return img_pil.resize((nw, nh), Image.BILINEAR), mask_pil.resize((nw, nh), Image.NEAREST)


def _dom_nonignore_ratio(mask_np: np.ndarray) -> float:
    """Dominant NON-ignore class pixel fraction (255 excluded); 1.0 if the crop is all-ignore."""
    vals, cnts = np.unique(mask_np, return_counts=True)
    cnts = cnts[vals != MASK_IGNORE]
    tot = int(cnts.sum())
    return float(cnts.max() / tot) if tot > 0 else 1.0


def _random_crop_pad_512(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState,
                         cat_max_ratio: float, size: int = SIZE, max_attempts: int = 10):
    """Random size x size crop. Per dimension: if the source side >= size take a random window;
    else random-place and pad (image [124,116,104], mask 255). cat_max_ratio: accept iff the
    dominant non-255 class fraction <= cat_max_ratio (255 excluded; background counts); up to
    max_attempts tries, else fall back to the best (lowest-dominance) candidate seen."""
    h, w = img_np.shape[:2]
    best = None  # (dom, img_crop, mask_crop)
    for k in range(1, max_attempts + 1):
        canvas_img = np.empty((size, size, 3), dtype=np.uint8)
        canvas_img[:] = IMAGENET_MEAN_8BIT
        canvas_mask = np.full((size, size), MASK_IGNORE, dtype=np.int64)
        if h >= size:
            sy, dy, ch = int(rng.randint(0, h - size + 1)), 0, size
        else:
            sy, dy, ch = 0, int(rng.randint(0, size - h + 1)), h
        if w >= size:
            sx, dx, cw = int(rng.randint(0, w - size + 1)), 0, size
        else:
            sx, dx, cw = 0, int(rng.randint(0, size - w + 1)), w
        canvas_img[dy:dy + ch, dx:dx + cw] = img_np[sy:sy + ch, sx:sx + cw]
        canvas_mask[dy:dy + ch, dx:dx + cw] = mask_np[sy:sy + ch, sx:sx + cw]
        dom = _dom_nonignore_ratio(canvas_mask)
        if best is None or dom < best[0]:
            best = (dom, canvas_img, canvas_mask)
        if dom <= cat_max_ratio:
            return canvas_img, canvas_mask, {"attempts": k, "fallback": False, "dom_ratio": dom}
    return best[1], best[2], {"attempts": max_attempts, "fallback": True, "dom_ratio": best[0]}


def train_preprocess(img_pil: Image.Image, mask_pil: Image.Image,
                     rng: np.random.RandomState, p: dict):
    """TRAIN-ONLY true RRC / multi-scale pipeline on the ORIGINAL-resolution image (pre-pad):
    EXIF(image)->RGB -> aspect-preserving multi-scale resize (long side = round(512*r),
    r~U(scale_range)) -> rotation (before crop) -> random 512x512 crop+pad (cat_max_ratio) ->
    joint flips -> image-only photometric. Returns (uint8 HWC 512x512, int64 HW 512x512).
    bilinear image / nearest mask; image pad [124,116,104] / mask pad 255; labels {0..115,255};
    no label remap. val/test use core_preprocess (unchanged)."""
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")     # EXIF image only; RGB 3ch
    rrc = p["random_resized_crop"]
    lo, hi = rrc["scale_range"]
    r = float(rng.uniform(lo, hi))                                 # multi-scale factor
    target_long = max(1, int(round(SIZE * r)))
    img_r, mask_r = _resize_long_side(img_pil, mask_pil, target_long)
    img_np = np.asarray(img_r, dtype=np.uint8)
    mask_np = np.asarray(mask_r).astype(np.int64)
    img_np, mask_np = _apply_rotation(img_np, mask_np, rng, p)     # rotation BEFORE crop
    img_np, mask_np, _ = _random_crop_pad_512(img_np, mask_np, rng, rrc["cat_max_ratio"])
    img_np, mask_np = _apply_flips(img_np, mask_np, rng, p)
    img_np = _apply_photometric(img_np, rng, p)
    return img_np, mask_np


def augment(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
    """LEGACY (compatibility) train aug on a 512x512 canvas: flips -> rotation -> image-only
    photometric. The ACTIVE train path is train_preprocess() (true original-resolution multi-scale
    RRC). Retained for backward compatibility / direct 512x512-canvas use; it does NOT perform the
    multi-scale RRC crop."""
    img_np, mask_np = _apply_flips(img_np, mask_np, rng, p)
    img_np, mask_np = _apply_rotation(img_np, mask_np, rng, p)
    img_np = _apply_photometric(img_np, rng, p)
    return img_np, mask_np


def finalize(img_np: np.ndarray, mask_np: np.ndarray):
    """uint8 HWC image + int64 HW mask -> (float32 CHW image tensor, int64 HW mask tensor)."""
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD          # per-channel, broadcast over HxW
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # CHW
    img_t = torch.from_numpy(img).float()
    mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).long()
    return img_t, mask_t
