"""The demo's only preprocessing: the evaluator's canvas protocol, called, never re-implemented.

core_preprocess/1.0.0 is src/data/transforms.py `core_preprocess` followed by `finalize`, exactly as the
evaluator's dataset applies them to a VAL or TEST image (src/data/dataset.py:96-97, reached through
src/eval/adapters.py PlantSegEvalDataset): EXIF transpose, RGB, aspect-preserving bilinear resize of
the long side to 512, centred pad to 512x512 with (124, 116, 104), scale to [0, 1], ImageNet
normalisation, float32 CHW. The batch dimension is added the way eval_collate's torch.stack adds it.

The evaluator always holds a ground-truth mask; an uploaded photo has none, so a constant placeholder
mask is passed. The image and the mask are resized and padded separately inside core_preprocess, so the
placeholder cannot change the image tensor (the smoke checks this), and its padded canvas (255 outside
the image) marks where the photo sits on the 512x512 canvas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

import demo.common  # noqa: F401  (puts the checkout on sys.path)
from src.data.transforms import MASK_IGNORE, SIZE, core_preprocess, finalize

PROTOCOL_ID = "core_preprocess/1.0.0"     # = src.eval.adapters.PREPROCESS_PROTOCOL (smoke-checked)
CANVAS = SIZE                             # 512


@dataclass(frozen=True)
class Preprocessed:
    tensor: torch.Tensor                  # float32 [1, 3, 512, 512]: the model input
    canvas: np.ndarray                    # uint8 [512, 512, 3]: the padded canvas before normalising
    valid: np.ndarray                     # bool [512, 512]: True where the photo is
    box: tuple[int, int, int, int]        # top, left, height, width of the photo on the canvas


def placeholder_mask(img: Image.Image, value: int = 0) -> Image.Image:
    return Image.new("L", img.size, value)


def evaluator_preprocess(img: Image.Image, placeholder_value: int = 0) -> Preprocessed:
    canvas, mask_canvas = core_preprocess(img, placeholder_mask(img, placeholder_value))
    x, m = finalize(canvas, mask_canvas)
    valid = (m != MASK_IGNORE).numpy()
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    box = (int(rows[0]), int(cols[0]), int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1))
    return Preprocessed(torch.stack([x]), canvas, valid, box)
