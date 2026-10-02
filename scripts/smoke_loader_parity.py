#!/usr/bin/env python3
"""The KD TRAIN loader is E1's (lane L-KD-HARDEN item 6; A1 finding G-F2). Synthetic, data-free, CPU.

The TRAIN-loader call is captured INSIDE `train_e1.run` and `train_distill.run` by a spy that stops each
run at that call (student built, nothing trained, nothing written). Both captured calls are then
replayed through the repository's own `build_dataloader` over a synthetic stand-in for PlantSegDataset
(real split class replaced; no file is opened), with num_workers 2, persistent workers and seed 42,
for 3 epochs of 4 batches:

  P1  the two trainers make the same TRAIN-loader call (split, batch, num_workers, persistent_workers,
      seed)
  P2  the realized stream is identical in every epoch: per-epoch digests of the sample order, the
      worker seeds, the per-sample augmentation seeds (the dataset.py:92 draw) and one further draw
      from each sample's RandomState
  N1  negative control: the 44c05dc KD call (train_distill.py:545 there, no persistent_workers) matches
      E1 in epoch 1 and differs in epochs 2 and 3, so the digest can tell the streams apart; this is
      the change of the realized stream from epoch 2 on that item 6 makes
  Z1  num_workers 0 (the invariance harness): the TRAIN-loader call captured inside train_distill.run at
      num_workers 0 passes persistent_workers False and gives the 44c05dc call's stream over 3 epochs
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

import src.data.dataset as ds_mod  # noqa: E402
import src.training.train_distill as td  # noqa: E402
import src.training.train_e1 as e1  # noqa: E402
from src.distill import FrozenTeacher, MockTeacher  # noqa: E402
from src.seeds import set_seed  # noqa: E402

N_SAMPLES, BATCH, WORKERS, SEED, EPOCHS = 64, 16, 2, 42, 3
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


class SyntheticTrain(Dataset):
    """Stand-in for PlantSegDataset(split): N_SAMPLES items, no files. Makes the dataset.py:92 draw."""

    def __init__(self, split: str):
        self.split = split

    def __len__(self) -> int:
        return N_SAMPLES

    def __getitem__(self, idx):
        aug_seed = int(np.random.randint(0, 2 ** 31 - 1))          # exactly dataset.py:92
        follow = int(np.random.RandomState(aug_seed).randint(0, 2 ** 31 - 1))
        return torch.tensor([idx, torch.initial_seed(), aug_seed, follow], dtype=torch.int64)


class _Stop(Exception):
    pass


def captured_train_call(module, **run_kwargs) -> dict:
    """Run `module.run` until its first build_dataloader call (TRAIN) and return that call."""
    calls, real = [], module.build_dataloader

    def spy(split, batch_size, **kw):
        calls.append({"split": split, "batch_size": batch_size, **kw})
        raise _Stop()

    module.build_dataloader = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module.run(**run_kwargs)
    except _Stop:
        pass
    finally:
        module.build_dataloader = real
    return calls[0] if calls else {}


def epoch_digests(call: dict) -> list[str]:
    """Per-epoch sha256 of the realized stream of `call`; EPOCHS empty strings when nothing was captured,
    so a failed capture FAILs its checks instead of crashing the smoke."""
    if not call:
        return [""] * EPOCHS
    set_seed(SEED)                                   # run() seeds before building its loaders
    kw = {k: call[k] for k in ("num_workers", "persistent_workers", "seed") if k in call}
    loader = ds_mod.build_dataloader(call["split"], call["batch_size"], **kw)
    out = []
    for _ in range(EPOCHS):                          # cycle() iterates the same loader object
        h = hashlib.sha256()
        for batch in loader:
            h.update(batch.numpy().tobytes())
        out.append(h.hexdigest())
    del loader
    return out


def main() -> int:
    print("=" * 78)
    print("LOADER PARITY SMOKE (L-KD-HARDEN item 6) - synthetic, CPU, no dataset file opened")
    print(f"torch {torch.__version__}")
    print("=" * 78)
    ds_mod.PlantSegDataset = SyntheticTrain
    common = dict(mode="dry", device="cpu", pretrained=False, batch_size=BATCH, max_iters=4,
                  val_interval=2, max_val_batches=1, num_workers=WORKERS, ckpt_dir_arg=None,
                  grad_clip_norm=None, log_every=1, seed=SEED)
    e1_call = captured_train_call(e1, **common)
    kd_call = captured_train_call(td, stage=td.resolve_stage("e2"), teacher=FrozenTeacher(MockTeacher(116)),
                                  lambda_logit=1.0, **common)
    check("P1_same_train_loader_call", bool(e1_call) and e1_call == kd_call
          and e1_call.get("persistent_workers") is True and e1_call.get("seed") == SEED,
          f"E1 {e1_call} | KD {kd_call}")

    e1_d = epoch_digests(e1_call)
    kd_d = epoch_digests(kd_call)
    for ep in range(EPOCHS):
        check(f"P2_epoch{ep + 1}_identical_stream", bool(e1_d[ep]) and e1_d[ep] == kd_d[ep],
              f"{e1_d[ep][:16]} vs {kd_d[ep][:16]}")

    old_kd_call = {"split": "train", "batch_size": BATCH, "num_workers": WORKERS, "seed": SEED}
    old_d = epoch_digests(old_kd_call)
    check("N1_44c05dc_call_equal_in_epoch1", old_d[0] == e1_d[0], f"{old_d[0][:16]} vs {e1_d[0][:16]}")
    check("N1_44c05dc_call_differs_from_epoch2_on", old_d[1] != e1_d[1] and old_d[2] != e1_d[2],
          f"epoch2 {old_d[1][:16]} vs {e1_d[1][:16]}; epoch3 {old_d[2][:16]} vs {e1_d[2][:16]}")

    kd_call0 = captured_train_call(td, stage=td.resolve_stage("e2"), teacher=FrozenTeacher(MockTeacher(116)),
                                   lambda_logit=1.0, **{**common, "num_workers": 0})
    zero_new = epoch_digests(kd_call0)
    zero_old = epoch_digests({**old_kd_call, "num_workers": 0})
    check("Z1_num_workers0_stream_unchanged", kd_call0.get("num_workers") == 0
          and kd_call0.get("persistent_workers") is False and zero_new == zero_old and all(zero_new),
          f"{kd_call0} {zero_new[0][:16]} vs {zero_old[0][:16]}")

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:46}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
