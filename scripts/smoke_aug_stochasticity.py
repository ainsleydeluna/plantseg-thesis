#!/usr/bin/env python3
"""Standing pre-launch gate: train-augmentation stochasticity + reproducibility (B31-10).

This is the regression test whose ABSENCE let finding F1 ("every image gets identical augmentation
every epoch") look plausible against reports/pre_e1_launch_audit.md. F1 was refuted empirically, but
nothing in the repo would have caught a real regression of that kind. This script is that guard.

Three properties, all of which the E1 recipe depends on and ch3 asserts:

  (a) TRAIN augmentation VARIES across two passes over the same samples.
      A regression here (e.g. re-seeding the per-sample RNG from the index) silently collapses
      augmentation to one fixed view per image for the whole 80k run.

  (b) VAL is BIT-IDENTICAL across two passes.
      The control. If (a) and (b) both changed, the harness is broken rather than the loader.

  (c) The seed sequence and the resulting tensors REPRODUCE across separate PROCESSES.
      ch3: "All training runs use random seed 42 across torch, numpy, and python's random module."
      Cross-epoch variation without cross-process reproducibility would trade one violation of that
      sentence for another.

Property (c) is the one that persistent_workers=True (B31-5) put at risk: _base_seed is drawn once
per DataLoader iterator rather than per epoch, so this must be re-checked whenever the loader
changes. CPU-only, no GPU, no downloads, 12 image decodes per process.

Usage:
    python scripts/smoke_aug_stochasticity.py
Exit code 0 == PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import src.data.dataset as DS  # noqa: E402
from src.data.dataset import PlantSegDataset, _seed_worker  # noqa: E402
from src.seeds import SEED, set_seed  # noqa: E402

MARKER = "###PAYLOAD###"
IDX = [100, 500]          # two fixed samples; enough to detect a frozen-augmentation regression


def _loader(dataset, num_workers: int, persistent: bool) -> DataLoader:
    """Mirror build_dataloader's kwargs over a Subset, with shuffle=False.

    shuffle=False is deliberate: the question is whether the augmentation RNG advances and
    reproduces, so fixing the sample order removes order variation as a confound. It cannot mask a
    change in the RNG properties under test.
    """
    generator = torch.Generator()
    generator.manual_seed(SEED)
    extra = {}
    if num_workers > 0:
        extra["prefetch_factor"] = DS.PREFETCH_FACTOR
        extra["persistent_workers"] = persistent
    return DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers,
                      generator=generator, worker_init_fn=_seed_worker, drop_last=False,
                      pin_memory=False, **extra)


def _digests(loader) -> list:
    out = []
    for img, _mask in loader:
        for i in range(img.shape[0]):
            out.append(hashlib.sha256(img[i].numpy().tobytes()).hexdigest()[:16])
    return out


def collect() -> dict:
    """Two passes each over train (workers=0), train (workers=2, persistent), and val."""
    set_seed(42)
    train_ds, val_ds = PlantSegDataset("train"), PlantSegDataset("val")

    original = np.random.RandomState
    seen: list = []

    def logging_rs(seed=None, *a, **k):
        seen.append(seed)
        return original(seed, *a, **k)

    np.random.RandomState = logging_rs
    try:
        set_seed(42)
        loader = _loader(Subset(train_ds, IDX), 0, False)
        train_w0, seeds = [], []
        for _ in range(2):
            seen.clear()
            train_w0.append(_digests(loader))
            seeds.append(list(seen))
    finally:
        np.random.RandomState = original

    set_seed(42)
    loader = _loader(Subset(train_ds, IDX), 2, True)
    train_w2 = [_digests(loader) for _ in range(2)]

    set_seed(42)
    loader = _loader(Subset(val_ds, IDX), 2, False)
    val_w2 = [_digests(loader) for _ in range(2)]

    return {"seeds": seeds, "train_w0": train_w0, "train_w2": train_w2, "val_w2": val_w2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child:                       # second process for the cross-process comparison
        print(MARKER + json.dumps(collect()))
        return 0

    print("=" * 78)
    print("B31-10 — augmentation stochasticity + reproducibility gate")
    print(f"torch {torch.__version__} | numpy {np.__version__} | "
          f"PREFETCH_FACTOR={DS.PREFETCH_FACTOR} | indices {IDX}")
    print("=" * 78)

    a = collect()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"],
                           cwd=REPO, capture_output=True, text=True, timeout=900)
    if child.returncode != 0:
        print("child process FAILED:")
        print(child.stderr[-2000:])
        return 1
    line = next(x for x in child.stdout.splitlines() if x.startswith(MARKER))
    b = json.loads(line[len(MARKER):])

    checks = {}

    checks["(a) train varies across passes [workers=0]"] = a["train_w0"][0] != a["train_w0"][1]
    checks["(a) train varies across passes [workers=2, persistent]"] = (
        a["train_w2"][0] != a["train_w2"][1])
    checks["(a) per-sample seeds differ across passes"] = a["seeds"][0] != a["seeds"][1]
    checks["(b) val bit-identical across passes [control]"] = a["val_w2"][0] == a["val_w2"][1]
    checks["(c) seeds reproduce across processes"] = a["seeds"] == b["seeds"]
    checks["(c) train digests reproduce across processes [workers=0]"] = (
        a["train_w0"] == b["train_w0"])
    checks["(c) train digests reproduce across processes [workers=2]"] = (
        a["train_w2"] == b["train_w2"])
    checks["(c) val digests reproduce across processes"] = a["val_w2"] == b["val_w2"]

    print(f"\nprocess A seeds : {a['seeds']}")
    print(f"process B seeds : {b['seeds']}")
    print(f"train workers=0 : pass1={a['train_w0'][0]}\n                  pass2={a['train_w0'][1]}")
    print(f"train workers=2 : pass1={a['train_w2'][0]}\n                  pass2={a['train_w2'][1]}")
    print(f"val   workers=2 : pass1={a['val_w2'][0]}\n                  pass2={a['val_w2'][1]}")

    print("\n[CHECKS]")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    passed = all(checks.values())
    if not passed:
        print("\nIf (a) failed: the per-sample augmentation RNG has stopped advancing — check "
              "src/data/dataset.py __getitem__ and build_dataloader's worker seeding.")
        print("If (c) failed: augmentation is stochastic but the run is NOT reproducible from "
              "seed 42, which contradicts ch3. Do NOT launch E1 until this is resolved.")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
