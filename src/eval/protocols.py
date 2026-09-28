"""Evaluation protocols: the 512-canvas protocol and the upstream PlantSeg test protocol (lane L-AM13).

Implements docs/EVALUATION_CONTRACT.md section 11 (AM-13, AM-16 item 4, AM-17 item 1(e);
docs/lane_specs/part1.md lane 2).

  canvas    `core_preprocess/1.0.0` -- the evaluator's default and the only protocol of the
            inferential analyses: src/data/transforms.core_preprocess (long side 512, symmetric pad to
            512x512), scored on the canvas. Nothing in this module changes it.
  upstream  `upstream/1.0.0` -- the test geometry of `tqwei05/PlantSeg`, for descriptive scores only:
     1. rescale, mmcv `Resize(scale=(2048, 512), keep_ratio=True)`: s = min(2048 / long side,
        512 / short side); each side becomes int(side * s + 0.5), mmcv `rescale_size`'s rounding
        (half up: 512.5 -> 513). The image is resized bilinearly; the mask is never resized.
     2. pad, `SegDataPreProcessor` with `test_cfg=dict(size_divisor=32)` and `pad_val=0`: the
        NORMALIZED image, bottom and right, with 0.0 up to the next multiple of 32 (mmseg
        `stack_batch`).
     3. whole-image inference, one image per batch.
     4. logits bilinear to the padded input shape (`predict_by_feat`), cropped to the rescaled shape
        and bilinear to the original shape (`postprocess_result`), align_corners=False; then argmax.
     5. original-resolution accumulation: the evaluation core takes the argmax of the logits this
        module returns and accumulates the confusion against the ORIGINAL mask (ignore 255) with the
        frozen src/eval/metrics.py -- the same functions, on a different confusion.

Only the geometry changes. Pixel numerics are the evaluator's own and identical for the student and
the teacher, since the R3 teacher path feeds the teacher the student's tensors: EXIF transpose of the
image, RGB, the repository's PIL bilinear resize, then [0, 1] scaling and ImageNet mean/std through
`transforms.finalize`. mmcv's own image resize (cv2) is not used (contract section 11).

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .evaluate import EvaluationIntegrityError

CANVAS = "canvas"
UPSTREAM = "upstream"
PROTOCOL_NAMES = (CANVAS, UPSTREAM)
CANVAS_PROTOCOL_ID = "core_preprocess/1.0.0"          # src/data/transforms.py core_preprocess
UPSTREAM_VERSION = "1.0.0"
UPSTREAM_PROTOCOL_ID = f"{UPSTREAM}/{UPSTREAM_VERSION}"
PROTOCOL_IDS = {CANVAS: CANVAS_PROTOCOL_ID, UPSTREAM: UPSTREAM_PROTOCOL_ID}

UPSTREAM_SCALE = (2048, 512)       # mmseg test pipeline: Resize(scale=(2048, 512), keep_ratio=True)
SIZE_DIVISOR = 32                  # SegDataPreProcessor test_cfg=dict(size_divisor=32)
PAD_VALUE_NORMALIZED = 0.0         # SegDataPreProcessor pad_val=0, applied after normalisation
ALIGN_CORNERS = False              # decode head and segmentor align_corners
UPSTREAM_BATCH_SIZE = 1            # whole-image inference
SHAPE_FIELDS = ("ori_shape", "rescaled_shape", "padded_shape")   # per-image, [h, w], in this order

UPSTREAM_NOTE = (
    "Descriptive protocol (AM-13, AM-16 item 4, AM-17 item 1(e)); never an inferential comparator. "
    "The upstream MSCAN-B class count behind the published 42.05% is NOT DETERMINABLE: the upstream "
    "PlantSeg MSCAN-L config uses 116 classes and the MSCAN-T config 115.")


class ProtocolError(ValueError):
    """An input or record that the upstream protocol cannot evaluate faithfully."""


# --------------------------------------------------------------------------------------------------
# artifact record
# --------------------------------------------------------------------------------------------------
def upstream_block() -> dict:
    """summary.json `protocol` of an upstream artifact (contract section 11). `name/version` equals the
    artifact's `dataset.preprocess_protocol`."""
    return {
        "name": UPSTREAM,
        "version": UPSTREAM_VERSION,
        "resize": {"short": min(UPSTREAM_SCALE), "long_max": max(UPSTREAM_SCALE),
                   "interp": "bilinear"},
        "size_divisor": SIZE_DIVISOR,
        "pad_value_normalized": PAD_VALUE_NORMALIZED,
        "logit_upsample": ["bilinear_to_padded", "crop", "bilinear_to_original"],
        "align_corners": ALIGN_CORNERS,
        "scored_resolution": "original",
        "note": UPSTREAM_NOTE,
    }


def summary_block(preprocess_protocol: str) -> dict | None:
    """summary.json `protocol` for an artifact scored under `preprocess_protocol`, or None.

    Only the upstream protocol writes the block and the per-image shape fields; a canvas artifact (and
    a synthetic fixture protocol) carries neither, and readers derive "canvas" from the absence.
    """
    return upstream_block() if preprocess_protocol == UPSTREAM_PROTOCOL_ID else None


# --------------------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------------------
def _check_hw(h, w) -> None:
    if not (type(h) is int and type(w) is int and h > 0 and w > 0):
        raise ProtocolError(f"image shape must be positive integers, got ({h!r}, {w!r})")


def upstream_scale_factor(h: int, w: int) -> float:
    """mmcv `rescale_size`'s factor for scale=(2048, 512): min(2048 / long side, 512 / short side)."""
    _check_hw(h, w)
    return min(max(UPSTREAM_SCALE) / max(h, w), min(UPSTREAM_SCALE) / min(h, w))


def upstream_rescaled_shape(h: int, w: int) -> tuple[int, int]:
    """(h, w) after the upstream rescale, exactly as mmcv `_scale_size`: int(side * s + 0.5)."""
    s = float(upstream_scale_factor(h, w))
    return int(h * s + 0.5), int(w * s + 0.5)


def padded_shape(h: int, w: int, divisor: int = SIZE_DIVISOR) -> tuple[int, int]:
    """(h, w) rounded up to multiples of `divisor`, as mmseg `stack_batch` does for `size_divisor`."""
    _check_hw(h, w)
    return (h + divisor - 1) // divisor * divisor, (w + divisor - 1) // divisor * divisor


def upstream_shapes(ori_h: int, ori_w: int) -> dict:
    """The per-image shape record: original, rescaled and padded (h, w), as JSON-ready int lists."""
    rh, rw = upstream_rescaled_shape(ori_h, ori_w)
    ph, pw = padded_shape(rh, rw)
    return {"ori_shape": [ori_h, ori_w], "rescaled_shape": [rh, rw], "padded_shape": [ph, pw]}


def validate_shapes(record: Mapping) -> dict:
    """Check a per-image shape record: exactly SHAPE_FIELDS, each [h, w] positive ints, and the
    rescaled and padded shapes are the upstream geometry of `ori_shape`. Returns a normalised copy."""
    if set(record) != set(SHAPE_FIELDS):
        raise ProtocolError(f"shape record keys {sorted(record)} != {sorted(SHAPE_FIELDS)}")
    out = {}
    for key in SHAPE_FIELDS:
        v = record[key]
        if not (isinstance(v, (list, tuple)) and len(v) == 2
                and all(type(x) is int and x > 0 for x in v)):
            raise ProtocolError(f"{key} must be [h, w] positive integers, got {v!r}")
        out[key] = [v[0], v[1]]
    expect = upstream_shapes(*out["ori_shape"])
    if out != expect:
        raise ProtocolError(f"shape record {out} is not the upstream geometry of ori_shape "
                            f"{out['ori_shape']} (expected {expect})")
    return out


# --------------------------------------------------------------------------------------------------
# input side: rescale, normalise, pad
# --------------------------------------------------------------------------------------------------
def rescale_image(image: np.ndarray, size_hw) -> np.ndarray:
    """uint8 HxWx3 RGB -> uint8 at `size_hw` (h, w) with the repository's resize, PIL bilinear, exactly
    as `transforms.core_preprocess` resizes (PIL returns an unchanged copy when the size already fits)."""
    h, w = (int(v) for v in size_hw)
    resized = Image.fromarray(np.ascontiguousarray(image)).resize((w, h), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def pad_normalized(image: torch.Tensor, divisor: int = SIZE_DIVISOR,
                   value: float = PAD_VALUE_NORMALIZED) -> torch.Tensor:
    """[C, h, w] normalised image -> [C, ph, pw], padded bottom and right with `value` to multiples of
    `divisor` (mmseg `stack_batch`: F.pad(t, (0, pw - w, 0, ph - h), value=pad_val))."""
    h, w = (int(v) for v in image.shape[-2:])
    ph, pw = padded_shape(h, w, divisor)
    return F.pad(image, (0, pw - w, 0, ph - h), value=value)


@dataclass(frozen=True)
class UpstreamSample:
    """One image prepared for the upstream protocol."""
    image: torch.Tensor      # float32 [3, ph, pw]: rescaled, normalised, then padded with 0.0
    target: torch.Tensor     # int64 [H, W]: the original-resolution mask, never resized
    shapes: dict             # SHAPE_FIELDS -> [h, w]


def prepare_upstream_sample(image: np.ndarray, mask: np.ndarray) -> UpstreamSample:
    """Original-resolution image (uint8 HxWx3 RGB, EXIF applied) and mask (HxW class indices) ->
    the model input and the scoring target of the upstream protocol."""
    # Imported here so the artifact writer, which imports this module, stays free of the dataset stack.
    from ..data.transforms import finalize

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ProtocolError(f"image must be uint8 HxWx3 RGB, got {image.dtype} {image.shape}")
    if mask.ndim != 2:
        raise ProtocolError(f"mask must be a 2-D class-index array, got shape {mask.shape}")
    if image.shape[:2] != mask.shape:
        raise ProtocolError(f"image {image.shape[:2]} and mask {mask.shape} differ; the upstream "
                            "protocol scores the original mask and never resizes it")
    shapes = upstream_shapes(int(mask.shape[0]), int(mask.shape[1]))
    img_t, target = finalize(rescale_image(image, shapes["rescaled_shape"]), mask)
    padded = pad_normalized(img_t)
    if list(padded.shape[-2:]) != shapes["padded_shape"]:
        raise ProtocolError(f"padded input {tuple(padded.shape[-2:])} != {shapes['padded_shape']}")
    return UpstreamSample(image=padded, target=target, shapes=shapes)


# --------------------------------------------------------------------------------------------------
# output side: two-step logit resize
# --------------------------------------------------------------------------------------------------
def logits_to_original(logits: torch.Tensor, shapes: Mapping) -> torch.Tensor:
    """MMSeg `predict_by_feat` -> `postprocess_result` for one image: [1, C, h, w] -> [1, C, H, W].

      bilinear_to_padded    logits not already at the padded input shape are resized there. Both
                            evaluator models already return logits at their input shape -- the
                            student's final F.interpolate, FrozenTeacher's `logits_size` resize, both
                            bilinear with align_corners=False -- so for them the step has been done;
      crop                  to the rescaled shape (the padding is bottom and right);
      bilinear_to_original  to the original shape (align_corners=False).
    """
    ph, pw = shapes["padded_shape"]
    rh, rw = shapes["rescaled_shape"]
    oh, ow = shapes["ori_shape"]
    if logits.dim() != 4 or logits.shape[0] != 1:
        raise ProtocolError(f"upstream logits must be [1, C, h, w], got {tuple(logits.shape)}")
    if tuple(logits.shape[-2:]) != (ph, pw):
        logits = F.interpolate(logits, size=(ph, pw), mode="bilinear", align_corners=ALIGN_CORNERS)
    logits = logits[:, :, :rh, :rw]
    return F.interpolate(logits, size=(oh, ow), mode="bilinear", align_corners=ALIGN_CORNERS)


class UpstreamForward:
    """`evaluate_model(batch_forward=...)` hook: one image in, logits at its original resolution out.

    `inner` is the evaluator's ordinary forward hook (`eval_runtime.ModelDeviceForward`), so inputs
    still move to the model's device and the runtime record and post-evaluation checks see every call;
    None calls the model directly. The core then takes the argmax and accumulates the confusion against
    the original-resolution target, so every prediction's shape equals its `ori_shape`.
    """

    def __init__(self, inner=None):
        self.inner = inner

    def __call__(self, model, batch) -> torch.Tensor:
        if len(batch.image_ids) != UPSTREAM_BATCH_SIZE:
            raise EvaluationIntegrityError(
                f"the upstream protocol evaluates {UPSTREAM_BATCH_SIZE} image per batch; got "
                f"{len(batch.image_ids)}")
        image_id = batch.image_ids[0]
        if batch.sample_meta is None:
            raise EvaluationIntegrityError(f"{image_id!r}: upstream batch carries no shape record")
        try:
            shapes = validate_shapes(batch.sample_meta[0])
        except ProtocolError as e:
            raise EvaluationIntegrityError(f"{image_id!r}: {e}") from e
        if list(batch.images.shape[-2:]) != shapes["padded_shape"]:
            raise EvaluationIntegrityError(
                f"{image_id!r}: input {tuple(batch.images.shape[-2:])} != padded_shape "
                f"{shapes['padded_shape']}")
        if list(batch.targets.shape[-2:]) != shapes["ori_shape"]:
            raise EvaluationIntegrityError(
                f"{image_id!r}: target {tuple(batch.targets.shape[-2:])} != ori_shape "
                f"{shapes['ori_shape']}; the upstream protocol scores the original mask")
        x = batch.images
        logits = self.inner(model, x) if self.inner is not None else model(x)
        out = logits_to_original(logits, shapes)
        if list(out.shape[-2:]) != shapes["ori_shape"]:
            raise EvaluationIntegrityError(
                f"{image_id!r}: prediction {tuple(out.shape[-2:])} != ori_shape {shapes['ori_shape']}")
        return out
