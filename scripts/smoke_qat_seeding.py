#!/usr/bin/env python3
"""d2 (lane 6, L-AM1q / AM-4a item 4): QAT seeding through E1's real loaders, in fresh processes.

A synthetic PlantSeg tree of the registered split sizes (5,367 TRAIN and 846 VAL pairs of tiny images,
no TEST surface) stands in for the dataset, so the trainer's own `build_qat_loaders` (E1's
build_dataloader calls with seed=--seed) and the geometry of record (335 steps per epoch, T_max 5,025)
run unchanged. Each training is a fresh process running run_qat in mode "smoke" for 4 steps with a
one-batch VAL: seed 42 twice and seed 43 at num_workers 0, then seed 42 twice at num_workers 2.

Checks: the loaders take the run's seed, set_seed is applied, the same seed repeats the batch order and
the four losses exactly, another seed changes the order, and each batch fingerprint is the exact sha256
of the integer masks the loader returns. No real data, no checkpoint, no download; temp files only.

Each worker sets the canvas to 64x64 in its own process (src.data.transforms.SIZE and the default size of
_random_crop_pad_512; the file is not touched): one QAT step on a 16x3x512x512 batch needs about 15 GB
on a CPU (measured: 1.9 GB at batch 2, 3.6 GB at batch 4). The dataset, sampler, loader generator,
worker seeding and augmentation streams are the real ones; only the canvas they produce is smaller.

Ends with one RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

N_TRAIN, N_VAL, NC = 5367, 846, 116
CANVAS = 64
RUNS = (("A", 42, 0), ("B", 42, 0), ("C", 43, 0), ("D", 42, 2), ("E", 42, 2))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------- the synthetic tree (parent process)
def build_tree(root: Path) -> None:
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(20261005)
    for split, n, stem in (("train", N_TRAIN, "train_{:05d}"), ("val", N_VAL, "val_{:04d}")):
        img_dir, mask_dir = root / "images" / split, root / "annotations" / split
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for i in range(n):
            h, w = 24 + int(rng.randint(0, 9)), 32 + int(rng.randint(0, 9))
            Image.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8)).save(img_dir / (stem.format(i) + ".jpg"),
                                                                              quality=90)
            m = rng.randint(0, NC, (h, w)).astype(np.uint8)
            m[:2, :] = 255
            Image.fromarray(m, mode="L").save(mask_dir / (stem.format(i) + ".png"))


# ---------------------------------------------------------------- one training (fresh worker process)
def worker(seed: int, num_workers: int, out: Path) -> int:
    import torch
    import src.data.transforms as T
    from src.models.student import build_student
    from src.quant import qat as Q
    from src.seeds import set_seed

    T.SIZE = CANVAS                                                    # this process only
    T._random_crop_pad_512.__defaults__ = (CANVAS,) + T._random_crop_pad_512.__defaults__[1:]

    torch.manual_seed(1234)
    model = build_student(pretrained=False)
    t0 = time.time()
    summary = Q.run_qat(stage="e5", mode="smoke", model=model, source_meta={"sha256": "0" * 64, "path": None},
                        out_dir=out / "run", seed=seed, clip_norm=1.0, clip_source="smoke", device="cpu",
                        num_workers=num_workers, max_steps=4, max_val_batches=1, log=lambda *a: None)
    seconds = time.time() - t0
    rows = [json.loads(line) for line in (out / "run" / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()]
    meta = rows[0]
    train = [r for r in rows if r.get("event") == "train"]
    pythonhashseed = os.environ.get("PYTHONHASHSEED")
    # the fingerprint replay: a fresh loader of the same seed, the RNG reset as run_qat resets it
    train_loader, val_loader = Q.build_qat_loaders(seed, num_workers)
    set_seed(seed)
    replay = []
    for i, (_img, mask) in enumerate(train_loader):
        replay.append(hashlib.sha256(mask.contiguous().numpy().tobytes()).hexdigest())
        if i == 3:
            break
    doc = {"seed": seed, "num_workers": num_workers, "seconds": seconds, "summary": summary,
           "losses": [r["loss"] for r in train], "batch_sha256": [r["batch_sha256"] for r in train],
           "replay_sha256": replay, "mask_dtype": str(mask.dtype), "mask_shape": list(mask.shape),
           "train_loader_seed": int(train_loader.generator.initial_seed()),
           "val_loader_seed": int(val_loader.generator.initial_seed()),
           "pythonhashseed": pythonhashseed,
           "run_meta": {k: meta.get(k) for k in ("loader", "steps_per_epoch", "total_steps", "train_images",
                                                  "val_images", "batch_size", "drop_last", "bn_freeze_after_step",
                                                  "obs_freeze_after_step", "determinism", "seed", "num_workers",
                                                  "persistent_workers", "scheduler")}}
    (out / "result.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.worker:
        return worker(args.seed, args.num_workers, Path(args.out))

    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    tmp = safe_tmpdir("smoke_qat_seeding_")                 # a temp path naming neither TEST nor VAL
    print("=" * 78)
    print("QAT SEEDING SMOKE (d2) — synthetic tree of the registered split sizes; fresh process per run")
    print(f"temp {tmp}")
    print("=" * 78)
    t0 = time.time()
    root = tmp / "plantseg"
    build_tree(root)
    print(f"[tree] {N_TRAIN} TRAIN + {N_VAL} VAL synthetic pairs in {time.time() - t0:.1f}s")
    docs = {}
    for label, seed, nw in RUNS:
        out = tmp / f"run_{label}"
        out.mkdir()
        env = {**os.environ, "PLANTSEG_DATA_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONHASHSEED", None)
        p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--seed", str(seed),
                            "--num-workers", str(nw), "--out", str(out)], env=env, capture_output=True, text=True,
                           timeout=3600)
        res = out / "result.json"
        docs[label] = json.loads(res.read_text(encoding="utf-8")) if p.returncode == 0 and res.is_file() else None
        print(f"[run {label}] seed {seed} num_workers {nw}: exit {p.returncode}"
              + (f", {docs[label]['seconds']:.1f}s" if docs[label] else f"\n{p.stderr[-1500:]}"))
    ok = all(docs.values())
    check("d2_all_runs_completed", ok, str({k: v is not None for k, v in docs.items()}))
    A, B, C, D, E = (docs.get(k) or {} for k in "ABCDE")
    check("d2_loader_seeded_with_run_seed",
          ok and all(d["train_loader_seed"] == d["seed"] == d["val_loader_seed"] for d in docs.values())
          and all(d["run_meta"]["seed"] == d["seed"] for d in docs.values()),
          str({k: (v or {}).get("train_loader_seed") for k, v in docs.items()}))
    det = {"deterministic_algorithms": True, "deterministic_algorithms_warn_only": True,
           "cudnn_deterministic": True, "cudnn_benchmark": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    check("d2_set_seed_applied",
          ok and all(d["run_meta"]["determinism"] == det and d["pythonhashseed"] == str(d["seed"])
                     for d in docs.values()),
          str((A.get("run_meta") or {}).get("determinism")))
    rm = A.get("run_meta") or {}
    check("d2_trainer_builds_the_loaders_of_record",
          ok and rm.get("loader", "").startswith("E1 build_dataloader") and rm.get("steps_per_epoch") == 335
          and rm.get("total_steps") == 5025 and rm.get("train_images") == N_TRAIN and rm.get("val_images") == N_VAL
          and rm.get("batch_size") == 16 and rm.get("drop_last") is True
          and (rm.get("bn_freeze_after_step"), rm.get("obs_freeze_after_step")) == (3350, 4020)
          and (rm.get("scheduler") or {}).get("T_max") == 5025
          and D.get("run_meta", {}).get("persistent_workers") is True and rm.get("persistent_workers") is False,
          f"{rm.get('steps_per_epoch')} steps/epoch, T_max {(rm.get('scheduler') or {}).get('T_max')}")
    check("d2_seed42_identical_batch_order",
          ok and len(A["batch_sha256"]) == 4 and A["batch_sha256"] == B["batch_sha256"], "num_workers 0")
    check("d2_seed42_identical_4step_losses",
          ok and len(A["losses"]) == 4 and A["losses"] == B["losses"],
          f"{A.get('losses')} vs {B.get('losses')}")
    check("d2_seed43_different_order",
          ok and all(a != c for a, c in zip(A["batch_sha256"], C["batch_sha256"])),
          "every one of the 4 batches differs")
    check("d2_workers2_identical_order_and_losses",
          ok and len(D["batch_sha256"]) == 4 and D["batch_sha256"] == E["batch_sha256"] and D["losses"] == E["losses"],
          "num_workers 2, persistent TRAIN workers")
    check("d2_fingerprint_is_sha256_of_loader_masks",
          ok and all(d["batch_sha256"] == d["replay_sha256"] and d["mask_dtype"] == "torch.int64"
                     and d["mask_shape"] == [16, CANVAS, CANVAS] for d in docs.values()),
          "exact, on the CPU, from the int64 masks as the loader returns them")
    print("\n[CHECKS]")
    for name, good, detail in results:
        print(f"  {name:46}: {'PASS' if good else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, good, _ in results if good)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
