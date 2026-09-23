#!/usr/bin/env python3
"""B64 C1 — the student DataLoader follows the run seed. Synthetic, data-free, CPU-only.

`build_dataloader` used to seed its generator from the constant `src.seeds.SEED`, so `--seed` reached
the global RNGs but not the shuffle order, the worker base seeds or the per-sample augmentation stream.
After B64 the generator takes the caller's seed; the default stays 42.

No dataset file is read: `src.data.dataset.PlantSegDataset` is replaced by a synthetic stand-in with
the real split sizes, and the repository's own `build_dataloader` is driven unchanged.

  A   seed 42 through build_dataloader(seed=42) == the pre-fix construction replicated below
      (generator seeded from src.seeds.SEED, identical kwargs), for num_workers 0 and 2: the first
      200 sampled indices, the per-worker base seeds, and the per-sample augmentation seeds.
  A'  the same record == GOLDEN, captured with --capture from the UNMODIFIED 3c43f89 code.
      Reported SKIP (never PASS) when the running torch differs from the capture's.
  C   build_dataloader() with no seed == seed 42 (unchanged callers are unaffected).
  B   seeds 43 and 44 differ from 42 and from each other (indices, worker seeds, augmentation seeds).
  W   train_e1.run and train_distill.run pass `seed=seed` to both build_dataloader calls.
  G   total_grad_norm == clip_grad_norm_'s total norm, and leaves every gradient bitwise unchanged.

Usage:  python scripts/smoke_loader_seed.py             # verify
        python scripts/smoke_loader_seed.py --capture   # print the seed-42 digest (pre-fix tree)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import src.data.dataset as ds_mod  # noqa: E402
from configs.data import SPLIT_SIZES  # noqa: E402
from src.seeds import SEED, set_seed  # noqa: E402

N_INDICES = 200
BATCH = 16
WORKER_COUNTS = (0, 2)
# Captured with --capture from the UNMODIFIED 3c43f89 loader (B64 Phase B, 2026-09-23), twice, identical.
GOLDEN = {"torch": "2.9.1+cpu",
          "sha256": "8672bdc42952fa6f60d0eab6495d4d958e8a04f4de08e023998154c6ebd723f6"}


class SyntheticSplit(Dataset):
    """Stand-in for PlantSegDataset: the real split size, no files. Draws one np.random.randint per
    sample exactly as the train __getitem__ does (dataset.py:92) and reports the process torch seed."""

    def __init__(self, split: str):
        self.split = split
        self.n = SPLIT_SIZES[split]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        aug_seed = int(np.random.randint(0, 2 ** 31 - 1))
        return torch.tensor([idx, torch.initial_seed(), aug_seed], dtype=torch.int64)


def _record(loader) -> dict:
    idx, seeds, aug = [], set(), []
    it = iter(loader)
    while len(idx) < N_INDICES:
        for row in next(it).tolist():
            idx.append(row[0]); seeds.add(row[1]); aug.append(row[2])
    del it
    return {"indices": idx[:N_INDICES], "worker_seeds": sorted(seeds), "aug_seeds": aug[:N_INDICES]}


def _build(seed: int, nw: int, pass_seed: bool = True) -> dict:
    set_seed(seed)                                   # train_e1.run: set_seed(seed) precedes the loaders
    kw = {"seed": seed} if pass_seed else {}
    return _record(ds_mod.build_dataloader("train", BATCH, num_workers=nw,
                                           persistent_workers=nw > 0, **kw))


def _prefix_replica(nw: int) -> dict:
    """The pre-B64 build_dataloader body, verbatim except for the dataset class."""
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    extra = ({"prefetch_factor": ds_mod.PREFETCH_FACTOR, "persistent_workers": True}
             if nw > 0 else {})
    return _record(DataLoader(SyntheticSplit("train"), batch_size=BATCH, shuffle=True,
                              num_workers=nw, generator=generator,
                              worker_init_fn=ds_mod._seed_worker, drop_last=True,
                              pin_memory=torch.cuda.is_available(), **extra))


def _digest(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode("utf-8")).hexdigest()


def _loader_calls(path: Path, func: str) -> list[bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    return [any(k.arg == "seed" and isinstance(k.value, ast.Name) and k.value.id == "seed"
                for k in n.keywords)
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "build_dataloader"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args(argv)
    ds_mod.PlantSegDataset = SyntheticSplit          # no dataset file is ever opened

    if args.capture:
        rec = {f"nw{nw}": _build(42, nw, pass_seed=False) for nw in WORKER_COUNTS}
        print(json.dumps({"torch": torch.__version__, "sha256": _digest(rec)}))
        return 0

    results: list[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    r42 = {f"nw{nw}": _build(42, nw) for nw in WORKER_COUNTS}
    d42 = _digest(r42)
    check("A_seed42_equals_prefix_replica",
          d42 == _digest({f"nw{nw}": _prefix_replica(nw) for nw in WORKER_COUNTS}), d42)
    if GOLDEN is None:
        check("A'_golden_present", False, "GOLDEN is not filled in")
    elif GOLDEN["torch"] != torch.__version__:
        print(f"[SKIP] A'_seed42_equals_golden  torch {torch.__version__} != capture {GOLDEN['torch']}")
    else:
        check("A'_seed42_equals_golden", d42 == GOLDEN["sha256"], GOLDEN["sha256"])
    check("C_default_equals_seed42",
          _digest({f"nw{nw}": _build(42, nw, pass_seed=False) for nw in WORKER_COUNTS}) == d42)

    r43 = {f"nw{nw}": _build(43, nw) for nw in WORKER_COUNTS}
    r44 = {f"nw{nw}": _build(44, nw) for nw in WORKER_COUNTS}
    for nw in WORKER_COUNTS:
        k = f"nw{nw}"
        for field in ("indices", "worker_seeds", "aug_seeds"):
            distinct = {tuple(r[k][field]) for r in (r42, r43, r44)}
            check(f"B_{field}_differ_{k}", len(distinct) == 3)

    for path, func in ((REPO / "src/training/train_e1.py", "run"),
                       (REPO / "src/training/train_distill.py", "run")):
        calls = _loader_calls(path, func)
        check(f"W_{path.stem}_passes_seed", len(calls) == 2 and all(calls), str(calls))

    from src.training.train_e1 import total_grad_norm
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                            torch.nn.Conv2d(8, 4, 1))
    m(torch.randn(2, 3, 16, 16)).square().mean().backward()
    before = [p.grad.clone() for p in m.parameters()]
    g = total_grad_norm(m.parameters())
    check("G_gradients_bitwise_unchanged",
          all(torch.equal(b, p.grad) for b, p in zip(before, m.parameters())))
    ref = float(torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")))
    check("G_equals_clip_grad_norm_total", abs(g - ref) <= 1e-6 * max(1.0, ref), f"{g} vs {ref}")

    n_fail = results.count(False)
    print(f"\nRESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({len(results) - n_fail}/{len(results)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
