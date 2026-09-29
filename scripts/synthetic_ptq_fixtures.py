"""Synthetic fixtures for the lane 8 (L-AM10) smokes. No PlantSeg data, no checkpoint of record.

Importing this module imports no repository code, so a smoke can create its synthetic dataset and set
PLANTSEG_DATA_ROOT before configs/data.py is first imported (it reads the root once, at import).

    make_tree(root, n_train=..., n_val=..., n_test=...)   tiny JPEG images + PNG masks per split
    make_e1_checkpoint(path) / make_e3_checkpoint(path)    random-weight students in the E1 / E3 schema
    safe_tmpdir(prefix)                                    a temp dir no VAL/TEST path rule can refuse
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np

IMAGE_SIZE = (48, 36)                  # (w, h): tiny; core_preprocess resizes the long side to 512
NUM_CLASSES = 116
_REFUSED = re.compile(r"(test)|(^|[^a-z0-9])(val|valid|validation)([^a-z0-9]|$)")


def refusable(path) -> bool:
    """True when a VAL/TEST path rule of src/quant/ptq.py would refuse this path."""
    return bool(_REFUSED.search(str(Path(path).resolve()).lower()))


def safe_tmpdir(prefix: str) -> Path:
    """A fresh temp directory whose resolved path names neither TEST nor VAL."""
    for _ in range(100):
        d = Path(tempfile.mkdtemp(prefix=prefix))
        if not refusable(d):
            return d
    raise RuntimeError("could not create a temp directory free of 'test'/'val' components")


def stems(split: str, n: int) -> list[str]:
    """Split-specific, mutually disjoint stems (names that no VAL/TEST path rule refuses)."""
    tag = {"train": "leaf", "val": "vleaf", "test": "tleaf"}[split]
    return [f"plant_{tag}_{i:04d}" for i in range(n)]


def _write_pair(img_path: Path, mask_path: Path, rng: np.random.RandomState) -> None:
    from PIL import Image
    w, h = IMAGE_SIZE
    img = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = rng.randint(1, NUM_CLASSES)
    Image.fromarray(img, "RGB").save(img_path, format="JPEG", quality=90)
    Image.fromarray(mask, "L").save(mask_path, format="PNG")


def make_tree(root, *, n_train: int, n_val: int = 0, n_test: int = 0, seed: int = 0) -> dict:
    """images/<split>/<stem>.jpg + annotations/<split>/<stem>.png. Returns {split: [stems]}."""
    root = Path(root)
    rng = np.random.RandomState(seed)
    out = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        if n <= 0:
            continue
        img_dir, mask_dir = root / "images" / split, root / "annotations" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        names = stems(split, n)
        for s in names:
            _write_pair(img_dir / f"{s}.jpg", mask_dir / f"{s}.png", rng)
        out[split] = names
    return out


def _student_state(seed: int) -> dict:
    import torch
    from src.models.student import build_student
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}


def make_e1_checkpoint(path, *, seed: int = 7) -> Path:
    """A random-weight student in the train_e1.py checkpoint schema (no `stage`: E1)."""
    import torch
    p = Path(path)
    torch.save({"iter": 80000, "model_state_dict": _student_state(seed), "optimizer_state_dict": {},
                "scheduler_state_dict": {}, "scheduler": "poly", "best_val_miou_all_class": 0.0,
                "num_classes": NUM_CLASSES}, p)
    return p


def make_e3_checkpoint(path, *, seed: int = 11) -> Path:
    """A random-weight E3 checkpoint: projection-free student plus the separate projection field."""
    import torch
    p = Path(path)
    torch.save({"iter": 80000, "stage": "E3", "model_state_dict": _student_state(seed),
                "cwd_projection_state_dict": {"weight": torch.randn(320, 160, 1, 1)},
                "best_val_miou_all_class": 0.0, "num_classes": NUM_CLASSES}, p)
    return p
