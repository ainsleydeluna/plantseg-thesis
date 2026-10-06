"""Synthetic fixtures for the lane 8 (L-AM10) smokes. No PlantSeg data, no checkpoint of record.

Importing this module imports no repository code, so a smoke can create its synthetic dataset and set
PLANTSEG_DATA_ROOT before configs/data.py is first imported (it reads the root once, at import).

    make_tree(root, n_train=..., n_val=..., n_test=...)   tiny JPEG images + PNG masks for TRAIN and VAL; the
                                                          TEST stems only (SL-1: no TEST folder is ever created)
    make_e1_checkpoint(path) / make_e3_checkpoint(path)    random-weight students in the E1 / E3 schema
    safe_tmpdir(prefix)                                    a temp dir no VAL/TEST path rule can refuse
    repo_state(repo) / state_verdict(before, after)        the repository's git status, before and after a run
"""
from __future__ import annotations

import os
import re
import subprocess
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


# The exclude form every git command of a smoke carries: docs/reference/ is never listed (DL-37) and no path whose
# name contains "test" is named (SL-1).
GIT_EXCLUDES = (".", ":(exclude)docs/reference/reference.pdf", ":(exclude)docs/reference", ":(exclude,icase)*test*")


def repo_state(repo) -> bytes | None:
    """The repository's state for a before/after check, or None when git cannot report it. Never printed.

    `git status --porcelain=v1` (untracked files one by one, ignored entries included), then the size and
    modification time of every file that status names: a write anywhere in the repository changes it, a second
    write to a file that was already dirty included, while files tracked at HEAD and left alone (such as
    configs/calibration/, tracked since c689634) change nothing. GIT_OPTIONAL_LOCKS=0: no index refresh.
    """
    argv = ["git", "-C", str(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching",
            "--", *GIT_EXCLUDES]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=300, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    state = [p.stdout]
    for entry in p.stdout.split(b"\0"):
        f = Path(repo) / os.fsdecode(entry[3:]) if len(entry) > 3 else None
        if f is not None and f.is_file():
            st = f.stat()
            state.append(f"{st.st_size} {st.st_mtime_ns} ".encode("ascii") + entry[3:])
    return b"\0".join(state)


def state_verdict(before: bytes | None, after: bytes | None) -> str:
    """The check's verdict: "unchanged" only when both states were read and are equal."""
    if before is None or after is None:
        return "git status unavailable"
    return "unchanged" if after == before else "changed during the run"


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
    """images/<split>/<stem>.jpg + annotations/<split>/<stem>.png for TRAIN and VAL. Returns {split: [stems]}.

    SL-1: no TEST folder is ever created. With n_test > 0 the TEST stems are returned (for disjointness checks)
    and images/test and annotations/test stay never-created paths; an attempt to open or list them still raises
    the audit events the smokes record.
    """
    root = Path(root)
    rng = np.random.RandomState(seed)
    out = {}
    for split, n in (("train", n_train), ("val", n_val)):
        if n <= 0:
            continue
        img_dir, mask_dir = root / "images" / split, root / "annotations" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        names = stems(split, n)
        for s in names:
            _write_pair(img_dir / f"{s}.jpg", mask_dir / f"{s}.png", rng)
        out[split] = names
    if n_test > 0:
        out["test"] = stems("test", n_test)
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
