"""Synthetic fixtures for the lane 6 (L-AM4 + L-AM1q) QAT smokes. No PlantSeg data, no checkpoint of record.

    color_task(n, size, seed, band)  2x4 grids of the eight task classes' colours plus noise: (x, mask)
    confident_student(seed)          the real FP32 student trained briefly on color_task until confident
    task_loaders(seed)               TRAIN (32 images, batch 16, drop_last) and VAL (8) loaders of color_task
    make_qat_run(out_dir, ...)       a 15-epoch QAT run directory written by src/quant/qat.py in mode "smoke"
    evaluator_driver(path)           a stand-in for scripts/evaluate_model.py that stubs only its git calls
    make_tree / safe_tmpdir          re-exported from scripts/synthetic_ptq_fixtures.py

Importing this module imports numpy and scripts/synthetic_ptq_fixtures.py only (torch and the repository's
modules load inside the functions), so a smoke can create a synthetic dataset and set PLANTSEG_DATA_ROOT
before configs/data.py is first imported.
"""
from __future__ import annotations

from pathlib import Path

from scripts.synthetic_ptq_fixtures import make_tree, safe_tmpdir  # noqa: F401  (re-exported)

NUM_CLASSES = 116
TASK_CLASSES = (0, 13, 29, 44, 57, 72, 101, 115)
# one colour per class, the corners of the cube {-1.5, 1.5}^3 in input space, plus an offset that puts the
# BN statistics far from their (0, 1) defaults (so a conversion that drops them cannot agree by accident)
TASK_COLORS = {c: (a, b, d) for c, (a, b, d) in zip(TASK_CLASSES, [(a, b, d) for a in (-1.5, 1.5)
                                                                   for b in (-1.5, 1.5) for d in (-1.5, 1.5)])}
GRID = (2, 4)
NOISE = 0.3
OFFSET = 3.0


def color_task(n: int, size: int = 64, seed: int = 0, band: int = 0):
    """n images, each a 2x4 grid of the eight TASK_CLASSES colours in a random order, plus noise and OFFSET.

    The mask is each cell's class; pixels within `band` of a cell border are 255 (ignore). Every image holds
    every class, so batch statistics and running statistics describe the same mixture.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    rows, cols = GRID
    ch, cw = size // rows, size // cols
    x = torch.empty(n, 3, size, size)
    mask = torch.empty(n, size, size, dtype=torch.long)
    for i in range(n):
        perm = torch.randperm(len(TASK_CLASSES), generator=g)
        for q in range(rows * cols):
            c = TASK_CLASSES[int(perm[q])]
            r0, c0 = (q // cols) * ch, (q % cols) * cw
            x[i, :, r0:r0 + ch, c0:c0 + cw] = torch.tensor(TASK_COLORS[c])[:, None, None]
            mask[i, r0:r0 + ch, c0:c0 + cw] = c
    x = x + NOISE * torch.randn(n, 3, size, size, generator=g) + OFFSET
    if band:
        for r in range(1, rows):
            mask[:, r * ch - band:r * ch + band, :] = 255
        for c in range(1, cols):
            mask[:, :, c * cw - band:c * cw + band] = 255
    return x.contiguous(), mask


def confident_student(seed: int = 0, steps: int = 250, batch: int = 8, lr: float = 2e-3):
    """The real FP32 student (random init, no download), trained briefly on color_task in train mode.

    Its BN statistics are then re-estimated on the trained network (a cumulative average over 40 batches),
    so eval mode normalises as train mode did and QAT's train-mode batch statistics match them. Measured on
    this host: eval and train-mode accuracy 0.995; after the 15-epoch QAT run the epoch-15 fake-quant model
    is 100% accurate on valid pixels with a top-1/top-2 margin of at least 8 LSB on all of them.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.models.student import build_student
    torch.manual_seed(seed)
    m = build_student(pretrained=False).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for i in range(steps):
        x, y = color_task(batch, 64, seed=10_000 + i)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    bns = [b for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None                                # a cumulative average
    with torch.no_grad():
        for i in range(40):
            m(color_task(16, 64, seed=30_000 + i)[0])
    for b in bns:
        b.momentum = 0.1
    return m.eval()


def task_loaders(seed: int = 42, n_train: int = 32, n_val: int = 8):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    x, y = color_task(n_train, 64, seed=1)
    vx, vy = color_task(n_val, 64, seed=2)
    train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(vx, vy), batch_size=16, shuffle=False)
    return train, val


def make_qat_run(out_dir, *, model, stage: str = "e5", seed: int = 42, clip: float = 1.0,
                 source_sha256: str = "0" * 64, nan_at=(), extra_meta: dict | None = None, log=None) -> dict:
    """A 15-epoch QAT run directory, written by src/quant/qat.py run_qat in mode "smoke"."""
    from src.quant import qat as Q
    return Q.run_qat(stage=stage, mode="smoke", model=model, source_meta={"sha256": source_sha256, "path": None,
                                                                           "bytes": None},
                     out_dir=Path(out_dir), seed=seed, clip_norm=clip, clip_source="smoke", device="cpu",
                     num_workers=0, loaders=task_loaders(seed), inject_nan_at_steps=nan_at,
                     extra_meta=extra_meta, log=log or (lambda *a: None))


EVALUATOR_DRIVER = r'''"""Smoke stand-in for scripts/evaluate_model.py: the same main(), only its git calls stubbed.

A cloud lane session never lists the whole repository (docs/reference/ stays unlisted), so the artifact
writer's whole-repository `git status` is replaced by an empty porcelain and a given commit, as
scripts/smoke_run_ptq.py does in-process: SMOKE_GIT_HEAD (the HEAD the eval records name, so the summaries
can be selected), else a fixed one. SMOKE_MAX_SAMPLES, when set, caps the VAL rows.
"""
import os
import sys
sys.dont_write_bytecode = True
repo = os.environ["SMOKE_REPO"]
sys.path.insert(0, repo)
import src.eval.artifacts as artifacts
artifacts.git_porcelain_bytes = lambda r: b""
artifacts.git_commit = lambda r: os.environ.get("SMOKE_GIT_HEAD") or "0" * 40
from scripts.evaluate_model import main
argv = sys.argv[1:]
cap = os.environ.get("SMOKE_MAX_SAMPLES")
if cap and "--max-samples" not in argv:
    argv += ["--max-samples", cap]
raise SystemExit(main(argv))
'''


def evaluator_driver(path) -> Path:
    p = Path(path)
    p.write_text(EVALUATOR_DRIVER, encoding="utf-8")
    return p
