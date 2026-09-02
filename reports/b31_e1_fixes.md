# B31 — E1 launch-blocking fixes

**Base commit:** `148af2c9b30ffc78e7e2f4d0b9ff55cd40948607` (unchanged — nothing staged, committed, or pushed)  
**Evidence base:** [pre_e1_launch_audit.md](pre_e1_launch_audit.md) including §9  
**Scope:** Group 1 (launch-blocking) + Group 2 (cheap pre-launch). F8/OS8, teacher code, and all manuscript fixes are explicitly OUT of scope.  
**Not touched (refuted in B30, per instruction):** F1 augmentation RNG (`src/data/dataset.py:83`), F5 determinism flags, F12 `.train()` restore, F9 teacher-code headline.

Every number below is labelled **MEASURED** (observed on this machine) or **INFERRED** (derived). Nothing is estimated and presented as measured.

---

## 1. Item table

| Item | Files changed | LOC Δ | Acceptance |
|---|---|---|---|
| **B31-1** Split-count assertion | `configs/data.py`, `src/data/dataset.py` | +30 / −8 | **PASS** |
| **B31-2** Resume + atomic checkpoints | `src/training/train_e1.py` | +104 / −9 | **PASS** |
| **B31-3** CUDA capability gate | `scripts/verify_env.py` | +35 / −1 | **PASS** |
| **B31-4** Dataset check in verify_env | `scripts/verify_env.py` | +54 / −0 | **PASS** |
| **B31-5** DataLoader throughput | `src/data/dataset.py`, `src/training/train_e1.py` | +27 / −5 | **PASS** |
| **B31-6** bincount dominance ratio | `src/data/transforms.py` | +19 / −5 | **PASS** |
| **B31-7** JSONL telemetry | `src/training/train_e1.py` | +68 / −3 | **PASS** |
| **B31-8** `drop_last` train-only | `src/data/dataset.py` | +5 / −1 | **PASS** |
| **B31-9** lr_trace cap + ckpt pruning | `src/training/train_e1.py` | +79 / −6 | **PASS** |
| **B31-10** Stochasticity regression gate | `scripts/smoke_aug_stochasticity.py` (new) | +174 / −0 | **PASS** |

Per-item LOC splits are attributed by hand where several items touched one file; the authoritative totals are the `git diff --numstat` in §7.

**Dry-run floor:** `python src/training/train_e1.py --dry-run` printed `RESULT: PASS` after every single item, and after the final state. Verified 10/10 times.

---

## 2. Per-item detail

### B31-1 — Split-count assertion pinned to COUNTED values

**Rescoped at plan time and approved.** The constants were ALREADY the counted values (`5367/846/1561`), so there was no wrong value to correct. The real defects were:

1. **Silent-pass on a missing key.** `DATA.get("splits", {}).get("sizes", {}).get(split)` returned `None` for a typo'd/renamed/removed key, and `expected is not None` then skipped the check entirely — the exact failure the guard exists to prevent.
2. Not a single named constant (a three-deep `.get()` chain).
3. The total (7,774) was never asserted.

Verbatim pre-B31 assertion (`src/data/dataset.py:63-69`):
```python
        # Guard against silent under/over-count vs the locked PlantSeg split sizes (configs/data.py).
        expected = DATA.get("splits", {}).get("sizes", {}).get(split)
        if expected is not None and len(self.pairs) != expected:
            raise RuntimeError(
                f"split={split} pair count {len(self.pairs)} != locked expected {expected} "
                f"(configs/data.py DATA['splits']['sizes']); check the dataset upload/extraction "
                f"under {root}")
```

**Diff:**
```diff
diff --git a/configs/data.py b/configs/data.py
index 7def360..39879f9 100644
--- a/configs/data.py
+++ b/configs/data.py
@@ -13,6 +13,20 @@ import os
 # current development machine (behavior unchanged). See reports/e1_runpod_launch_runbook.md.
 DEFAULT_PLANTSEG_DATA_ROOT = r"C:\Users\admin\plantseg_data\plantseg"
 
+# ------------------------------------------------------------------ split sizes (single source of truth)
+# COUNTED from the pre-partitioned folders on disk (B16: zero identifier overlap across partitions);
+# re-verified 2026-09-01 by reports/pre_e1_launch_audit.md E15. These values are AUTHORITATIVE.
+# ch3's 5,442 / 778 / 1,554 is a known arithmetic artifact of applying a nominal 70/10/20 ratio to
+# 7,774 (7774*0.70 = 5441.8); the actual official split is 69.0 / 10.9 / 20.1%. The manuscript is
+# under correction â€” do NOT edit these constants to match it.
+SPLIT_SIZES = {"train": 5367, "val": 846, "test": 1561}     # [empirical, counted]
+SPLIT_TOTAL = 7774                                          # [empirical, counted]
+
+if sum(SPLIT_SIZES.values()) != SPLIT_TOTAL:
+    raise RuntimeError(
+        f"configs/data.py is internally inconsistent: sum(SPLIT_SIZES)={sum(SPLIT_SIZES.values())} "
+        f"!= SPLIT_TOTAL={SPLIT_TOTAL}")
+
 DATA = {
     # Root (extracted dataset path) â€” verified to exist; see reports/dataset_location_log.md.
     # PLANTSEG_DATA_ROOT env var overrides this when set; otherwise the Windows default is used.
@@ -31,8 +45,8 @@ DATA = {
         "dirs": ("images/{split}", "annotations/{split}"),  # use folders, NOT annotation_*.json
         "names": ("train", "val", "test"),
         "files": ("annotation_train.json", "annotation_val.json", "annotation_test.json"),  # COCO; not used for split
-        "sizes": {"train": 5367, "val": 846, "test": 1561},  # [empirical]
-        "test_count": 1561,                      # per-image metric unit count
+        "sizes": SPLIT_SIZES,                    # [empirical] single source of truth: SPLIT_SIZES above
+        "test_count": SPLIT_SIZES["test"],       # per-image metric unit count
         "integrity_check": "zero identifier overlap across partitions (verified)",
     },
 
diff --git a/src/data/dataset.py b/src/data/dataset.py
index ce74716..1781da0 100644
--- a/src/data/dataset.py
+++ b/src/data/dataset.py
@@ -23,7 +23,7 @@ if str(REPO) not in sys.path:
     sys.path.insert(0, str(REPO))
 
 from configs.augment import AUGMENT          # noqa: E402
-from configs.data import DATA                # noqa: E402
+from configs.data import DATA, SPLIT_SIZES, SPLIT_TOTAL  # noqa: E402
 from src.seeds import SEED                   # noqa: E402
 
 from .transforms import core_preprocess, finalize, train_preprocess  # noqa: E402
@@ -61,12 +61,21 @@ class PlantSegDataset(Dataset):
         if not self.pairs:
             raise RuntimeError(f"no image/mask pairs found for split={split} under {root}")
         # Guard against silent under/over-count vs the locked PlantSeg split sizes (configs/data.py).
-        expected = DATA.get("splits", {}).get("sizes", {}).get(split)
-        if expected is not None and len(self.pairs) != expected:
+        # An unregistered split is a HARD error: a missing/renamed key must never silently disable
+        # this guard (the previous `.get()` chain defaulted to None and skipped the check entirely).
+        if split not in SPLIT_SIZES:
             raise RuntimeError(
-                f"split={split} pair count {len(self.pairs)} != locked expected {expected} "
-                f"(configs/data.py DATA['splits']['sizes']); check the dataset upload/extraction "
-                f"under {root}")
+                f"split={split!r} has no registered expected count in configs/data.py SPLIT_SIZES "
+                f"(registered: {sorted(SPLIT_SIZES)}); refusing to load an unverified split")
+        expected = SPLIT_SIZES[split]
+        if len(self.pairs) != expected:
+            raise RuntimeError(
+                f"split={split} pair count MISMATCH: expected {expected}, actual {len(self.pairs)} "
+                f"(delta {len(self.pairs) - expected:+d}). All splits expected {dict(SPLIT_SIZES)}, "
+                f"total {SPLIT_TOTAL}. Check the dataset upload/extraction under {root}.\n"
+                f"NOTE: the COUNTED values in configs/data.py SPLIT_SIZES are AUTHORITATIVE. ch3's "
+                f"5,442/778/1,554 is a known arithmetic artifact (nominal 70/10/20 applied to 7,774) "
+                f"and is under correction in the manuscript â€” do NOT edit SPLIT_SIZES to match it.")
         self.aug_params = AUGMENT
 
     def __len__(self) -> int:
@@ -94,11 +103,27 @@ def _seed_worker(worker_id: int) -> None:
     random.seed(s)
 
 
-def build_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
-    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42)."""
+PREFETCH_FACTOR = 4          # batches pre-staged per worker (B31-5)
+
+
+def build_dataloader(split: str, batch_size: int, num_workers: int = 0,
+                     persistent_workers: bool = False) -> DataLoader:
+    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42).
+
+    `persistent_workers` is opt-in and intended for the TRAIN loader only: it keeps the worker pool
+    (and its decoded-image buffers) alive for the whole run, which is worth it across 80k iterations
+    but not across the 20 validation passes. `prefetch_factor`/`persistent_workers` are only legal
+    when num_workers > 0 â€” PyTorch raises otherwise, and the dry-run path uses num_workers=0 â€” so
+    both are passed conditionally. `pin_memory` is gated on CUDA: it is a no-op without a device and
+    emits a warning, so gating keeps the CPU dry-run output clean.
+    """
     dataset = PlantSegDataset(split)
     generator = torch.Generator()
     generator.manual_seed(SEED)
+    extra = {}
+    if num_workers > 0:
+        extra["prefetch_factor"] = PREFETCH_FACTOR
+        extra["persistent_workers"] = persistent_workers
     return DataLoader(
         dataset,
         batch_size=batch_size,
@@ -106,5 +131,10 @@ def build_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataL
         num_workers=num_workers,
         generator=generator,
         worker_init_fn=_seed_worker,
-        drop_last=False,
+        # TRAIN only: 5367 % 16 = 7, so every epoch would otherwise end on a ragged 7-sample batch,
+        # perturbing BatchNorm statistics and the Dice term's per-batch class-presence set.
+        # val/test keep drop_last=False â€” dropping evaluation samples would corrupt the metric.
+        drop_last=(split == "train"),
+        pin_memory=torch.cuda.is_available(),
+        **extra,
     )
```

**Acceptance — raw stdout:**
```text
==============================================================================
B31-1 ACCEPTANCE
==============================================================================
SPLIT_SIZES = {'train': 5367, 'val': 846, 'test': 1561}
SPLIT_TOTAL = 7774
DATA['splits']['sizes'] is SPLIT_SIZES -> True
DATA['splits']['test_count'] = 1561
internal consistency sum==total -> True

--- [1] all three splits construct successfully ---
  train  len=5367  OK
  val    len=846  OK
  test   len=1561  OK

--- [2] deliberately wrong constant must FAIL loudly ---
  RuntimeError raised. Message:

    | split=train pair count MISMATCH: expected 9999, actual 5367 (delta -4632). All splits expected {'train': 9999, 'val': 846, 'test': 1561}, total 7774. Check the dataset upload/extraction under C:\Users\admin\plantseg_data\plantseg.
    | NOTE: the COUNTED values in configs/data.py SPLIT_SIZES are AUTHORITATIVE. ch3's 5,442/778/1,554 is a known arithmetic artifact (nominal 70/10/20 applied to 7,774) and is under correction in the manuscript — do NOT edit SPLIT_SIZES to match it.

--- [3] unregistered split must HARD-ERROR (old code silently passed) ---
  ValueError from the _SPLITS whitelist (reached first): split must be one of ('train', 'val', 'test'), got 'trainval'
  missing-key guard fired: split='val' has no registered expected count in configs/data.py SPLIT_SIZES (registered: ['test', 'train']); refusing to load an unverified split

--- [4] internal consistency check fires on an inconsistent edit ---
  import-time guard fired: configs/data.py is internally inconsistent: sum(SPLIT_SIZES)=7774 != SPLIT_TOTAL=7000

RESULT: PASS
```

### B31-2 — Resume + atomic checkpoints (F2 + F13 together)

**Diff:** see the consolidated `train_e1.py` diff in §2.11 (four items touched this file).

**Acceptance — raw stdout:**
```text
==============================================================================
B31-2 ACCEPTANCE — resume + atomic checkpoints
==============================================================================

----- [1] RUN A: train to iter 3, write last.pt -----
cmd: train_e1.py --dry-run --batch-size 1 --val-interval 3 --max-val-batches 1 --max-iters 3 --ckpt-interval 3 --ckpt-dir C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz
  [ckpt] dir=C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz (verified OUTSIDE repo)
  [iter    1/3] loss=5.9711 ce=4.9889 dice=0.9822 lr=9.99988750e-03
  [iter    2/3] loss=6.0464 ce=5.0636 dice=0.9828 lr=9.99977500e-03
  [iter    3/3] loss=6.7518 ce=5.7696 dice=0.9822 lr=9.99966250e-03
  [val     3/3] cm_batches=1 cm_total_px=165888 all_class_miou=0.00000 disease_only_miou(PROVISIONAL)=0.00000
  [ckpt    3/3] new best all_class_miou=0.00000 -> C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\e1_student_best_iter3.pt
  [last    3/3] resume point -> C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt
  [summary] lr_trace=['9.99988750e-03', '9.99977500e-03', '9.99966250e-03']
  RESULT: PASS
  last.pt exists: True  size=23735911 B
  no stray .tmp left behind: True

----- [2] RUN B: resume from last.pt, continue to iter 6 -----
cmd: train_e1.py --dry-run --batch-size 1 --val-interval 3 --max-val-batches 1 --max-iters 6 --ckpt-interval 3 --ckpt-dir C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz --resume C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt
  [ckpt] dir=C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz (verified OUTSIDE repo)
  [resume] from C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt | resuming at iter=4 lr=9.99966250e-03 best_all_class_miou=0.00000
  [resume] WARNING: data-order continuity is NOT restored. The training loop consumes an infinite `cycle(train_loader)`; the position within the current epoch is not recoverable, so the post-resume sample order differs from an uninterrupted run. RNG streams ARE restored, so augmentation remains reproducible from this point onward. A resumed run is NOT bitwise-identical to an uninterrupted one.
  [iter    4/6] loss=5.9461 ce=4.9641 dice=0.9820 lr=9.99955000e-03
  [iter    5/6] loss=6.2066 ce=5.2173 dice=0.9893 lr=9.99943750e-03
  [iter    6/6] loss=5.9762 ce=5.0004 dice=0.9758 lr=9.99932500e-03
  [val     6/6] cm_batches=1 cm_total_px=165888 all_class_miou=0.00000 disease_only_miou(PROVISIONAL)=0.00000
  [last    6/6] resume point -> C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt
  [summary] lr_trace=['9.99955000e-03', '9.99943750e-03', '9.99932500e-03']
  RESULT: PASS

----- [3] RUN C: uninterrupted control, 1..6 -----
cmd: train_e1.py --dry-run --batch-size 1 --val-interval 3 --max-val-batches 1 --max-iters 6 --ckpt-interval 6 --ckpt-dir C:\Users\admin\AppData\Local\Temp\b31_2_ctl_c2gukfm7
  [ckpt] dir=C:\Users\admin\AppData\Local\Temp\b31_2_ctl_c2gukfm7 (verified OUTSIDE repo)
  [iter    1/6] loss=5.9711 ce=4.9889 dice=0.9822 lr=9.99988750e-03
  [iter    2/6] loss=6.0464 ce=5.0636 dice=0.9828 lr=9.99977500e-03
  [iter    3/6] loss=6.7518 ce=5.7696 dice=0.9822 lr=9.99966250e-03
  [val     3/6] cm_batches=1 cm_total_px=165888 all_class_miou=0.00000 disease_only_miou(PROVISIONAL)=0.00000
  [ckpt    3/6] new best all_class_miou=0.00000 -> C:\Users\admin\AppData\Local\Temp\b31_2_ctl_c2gukfm7\e1_student_best_iter3.pt
  [iter    4/6] loss=7.3307 ce=6.3468 dice=0.9839 lr=9.99955000e-03
  [iter    5/6] loss=6.0477 ce=5.0730 dice=0.9747 lr=9.99943750e-03
  [iter    6/6] loss=5.9532 ce=4.9731 dice=0.9800 lr=9.99932500e-03
  [val     6/6] cm_batches=1 cm_total_px=165888 all_class_miou=0.00000 disease_only_miou(PROVISIONAL)=0.00000
  [last    6/6] resume point -> C:\Users\admin\AppData\Local\Temp\b31_2_ctl_c2gukfm7\last.pt
  [summary] lr_trace=['9.99988750e-03', '9.99977500e-03', '9.99966250e-03', '9.99955000e-03', '9.99943750e-03', '9.99932500e-03']
  RESULT: PASS

----- [4] LR continuity vs the uninterrupted 80k poly curve -----
  RUN B lr_trace (iters 4..6): ['9.9995500000e-03', '9.9994375000e-03', '9.9993250000e-03']
  RUN C lr_trace (iters 1..6): ['9.9998875000e-03', '9.9997750000e-03', '9.9996625000e-03', '9.9995500000e-03', '9.9994375000e-03', '9.9993250000e-03']
  resumed LR == uninterrupted LR at iters 4,5,6 -> True
  fresh PolynomialLR(total_iters=80000, power=0.9) iters 4..6: ['9.9995499989e-03', '9.9994374982e-03', '9.9993249975e-03']
  last.pt optimizer lr (full precision): 0.00999932499746868
  last.pt scheduler last_epoch         : 6
  analytic ref[last_epoch-1]           : 0.00999932499746868
  checkpoint LR == analytic 80k poly curve at iter 6 -> True
  printed lr_trace matches analytic to 8-sig-digit print precision -> True

----- [5] last.pt payload -----
  top-level keys: ['best_ckpt', 'best_val_miou_all_class', 'iter', 'model_state_dict', 'num_classes', 'optimizer_state_dict', 'rng_state', 'scheduler', 'scheduler_state_dict']
  iter=6  num_classes=116  scheduler=PolynomialLR
  rng_state keys: ['loader_generator', 'numpy', 'python', 'torch', 'torch_cuda']

----- [6] atomicity: truncated write leaves the previous good checkpoint intact -----
  wrote truncated last.pt.tmp (7911970 B of 23735911 B)
  last.pt unchanged on disk: True
  last.pt still loads, iter=6 (was 6) -> True
  truncated .tmp is unloadable as expected: RuntimeError

----- [7] RUN D: resume AFTER the truncated-write crash (proves recoverability) -----
cmd: train_e1.py --dry-run --batch-size 1 --val-interval 3 --max-val-batches 1 --max-iters 9 --ckpt-interval 9 --ckpt-dir C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz --resume C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt
  [ckpt] dir=C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz (verified OUTSIDE repo)
  [resume] from C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt | resuming at iter=7 lr=9.99932500e-03 best_all_class_miou=0.00000
  [resume] WARNING: data-order continuity is NOT restored. The training loop consumes an infinite `cycle(train_loader)`; the position within the current epoch is not recoverable, so the post-resume sample order differs from an uninterrupted run. RNG streams ARE restored, so augmentation remains reproducible from this point onward. A resumed run is NOT bitwise-identical to an uninterrupted one.
  [iter    7/9] loss=6.1547 ce=5.1763 dice=0.9784 lr=9.99921250e-03
  [iter    8/9] loss=7.0758 ce=6.0997 dice=0.9761 lr=9.99910000e-03
  [iter    9/9] loss=5.7993 ce=4.8299 dice=0.9694 lr=9.99898749e-03
  [val     9/9] cm_batches=1 cm_total_px=165888 all_class_miou=0.47113 disease_only_miou(PROVISIONAL)=0.00000
  [ckpt    9/9] new best all_class_miou=0.47113 -> C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\e1_student_best_iter9.pt
  [last    9/9] resume point -> C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt
  [summary] lr_trace=['9.99921250e-03', '9.99910000e-03', '9.99898749e-03']
  RESULT: PASS

----- [8] RUN E: resume with --max-iters BELOW the checkpoint iter (already-complete guard) -----
cmd: train_e1.py --dry-run --batch-size 1 --val-interval 3 --max-val-batches 1 --max-iters 2 --ckpt-interval 2 --ckpt-dir C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz --resume C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt
  [ckpt] dir=C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz (verified OUTSIDE repo)
  [resume] from C:\Users\admin\AppData\Local\Temp\b31_2_ce6f5nzz\last.pt | resuming at iter=10 lr=9.99898749e-03 best_all_class_miou=0.47113
  [resume] WARNING: data-order continuity is NOT restored. The training loop consumes an infinite `cycle(train_loader)`; the position within the current epoch is not recoverable, so the post-resume sample order differs from an uninterrupted run. RNG streams ARE restored, so augmentation remains reproducible from this point onward. A resumed run is NOT bitwise-identical to an uninterrupted one.
  [resume] nothing to do: the checkpoint is already at iter 9, which meets or exceeds --max-iters 2. Raise --max-iters to continue training, or resume from an earlier checkpoint. No iterations were run and no checkpoint was written.
  RESULT: PASS (already complete at iter 9)

RESULT: PASS
```

**Two findings from this acceptance run, both reported rather than smoothed:**

1. **My first analytic LR check was a TEST bug, not a code bug.** `lr_trace` is printed via `'%.8e'`, so parsing that string back yields an 8-significant-digit *rounding* (`9.99955000e-03`) which cannot equal the full-precision value (`9.9995499989e-03`) at a 1e-15 tolerance. Corrected by reading the LR from the checkpoint at full float64 precision: `last.pt` carries `0.00999932499746868` and the analytic 80k PolynomialLR curve at `last_epoch=6` is `0.00999932499746868` — **exact**.
2. **A real defect my own test exposed.** Resuming a checkpoint whose `iter` already meets or exceeds `--max-iters` ran zero iterations, left `checks` empty, and printed a bare `RESULT: FAIL` with no explanation. Fixed with an explicit already-complete guard (scenario [8] above). This is inside B31-2's resume semantics, so I did not stop to ask.

**Documented deviation — data-order continuity is NOT restored.** The loop consumes an infinite `cycle(train_loader)`; position within the current epoch is unrecoverable. This is printed as a warning on every resume and is visible in RUN B vs RUN C above: losses diverge at iters 4–6 (`5.9461` vs `7.3307`) while the LR curve stays identical. RNG streams ARE restored, so augmentation remains reproducible from the resume point forward. **A resumed run is not bitwise-identical to an uninterrupted one** and must be reported as such in Ch4 if used.

### B31-3 — CUDA compute-capability gate

```diff
diff --git a/scripts/verify_env.py b/scripts/verify_env.py
index 4debe8f..48b91c7 100644
--- a/scripts/verify_env.py
+++ b/scripts/verify_env.py
@@ -26,6 +26,11 @@ if str(REPO) not in sys.path:
 MOBILENET_CKPT = "mobilenet_v3_large-5c1a4163.pth"      # torchvision IMAGENET1K_V2 backbone file
 IMAGENET_ALIAS = "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"
 
+# Highest CUDA compute capability the PINNED stack can emit kernels for. torch 2.1.0+cu121 ships
+# cubins/PTX for sm_50..sm_90 only; a Blackwell-class device (sm_100/sm_120) has no compatible
+# kernel and fails at the FIRST kernel launch, after the pod is already provisioned and paid for.
+MAX_SM = (9, 0)
+
 
 def _run(cmd, timeout=120):
     try:
@@ -124,6 +129,27 @@ def main() -> int:
     print(f"  cuda_version        : {cuda_ver}")
     print(f"  gpu_count           : {gpu_count}")
     print(f"  gpu_names           : {gpu_names}")
+
+    # Compute-capability gate: FAIL (not warn) above MAX_SM â€” the pinned stack has no kernel.
+    gpu_caps = [torch.cuda.get_device_capability(i) for i in range(gpu_count)] if cuda else []
+    unsupported = [(i, gpu_names[i], gpu_caps[i]) for i in range(gpu_count)
+                   if gpu_caps[i] > MAX_SM]
+    cap_ok = not unsupported
+    print(f"  gpu_capabilities    : {[f'sm_{a}{b}' for a, b in gpu_caps]} "
+          f"(max supported by the pinned stack: sm_{MAX_SM[0]}{MAX_SM[1]})")
+    if not cuda:
+        print("  capability_gate     : SKIPPED (no CUDA device visible)")
+    elif cap_ok:
+        print("  capability_gate     : PASS")
+    else:
+        for i, name, (a, b) in unsupported:
+            print(f"  capability_gate     : FAIL â€” cuda:{i} '{name}' is sm_{a}{b}, above the "
+                  f"sm_{MAX_SM[0]}{MAX_SM[1]} ceiling of torch {torch_ver} / "
+                  f"torchvision {tv_ver}. Blackwell-class pods (RTX 5090, RTX Pro 6000, B200, "
+                  f"B300) are INCOMPATIBLE with the pinned stack and will abort at the first "
+                  f"CUDA kernel launch. Choose an Ada/Hopper/Ampere pod (sm_80â€“sm_90, e.g. "
+                  f"A100 / H100 / L40S / RTX 4090) or re-pin the stack.")
+
     print(f"  cudnn_available     : {cudnn_avail}")
     print(f"  cudnn_version       : {cudnn_ver}")
     print(f"  cpu_count(logical)  : {os.cpu_count()}")
@@ -220,17 +246,79 @@ def main() -> int:
     print(f"  repo_.pt_files   : {repo_pt}  (MUST be empty)")
     print(f"  no_repo_checkpoint : {len(repo_pt) == 0}")
 
+    # (17b) dataset: root resolves, all six split dirs exist, per-split PAIR counts match the
+    # single source of truth in configs/data.py. Index/stat only â€” NO image is decoded.
+    # Without this, verify_env could print "OK for real E1 training" and the run would then die
+    # inside PlantSegDataset.__init__ after the pod was already provisioned.
+    print("\n[17b] dataset (root / split dirs / per-split pair counts vs configs/data.py SPLIT_SIZES)")
+    from configs.data import DATA as DATA_CFG, SPLIT_SIZES, SPLIT_TOTAL
+    ds_root = Path(DATA_CFG["root"])
+    ds_problems, ds_counts = [], {}
+    print(f"  PLANTSEG_DATA_ROOT env : {os.environ.get('PLANTSEG_DATA_ROOT', '(unset -> default)')}")
+    print(f"  resolved root          : {ds_root}")
+    if not ds_root.is_dir():
+        ds_problems.append(f"dataset root does not exist or is not a directory: {ds_root}")
+    else:
+        for sp in ("train", "val", "test"):
+            img_dir, mask_dir = ds_root / "images" / sp, ds_root / "annotations" / sp
+            if not img_dir.is_dir():
+                ds_problems.append(f"missing image dir: {img_dir}")
+                continue
+            if not mask_dir.is_dir():
+                ds_problems.append(f"missing annotation dir: {mask_dir}")
+                continue
+            imgs = [p for p in img_dir.iterdir()
+                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
+            pairs = sum(1 for p in imgs if (mask_dir / f"{p.stem}.png").exists())
+            expected = SPLIT_SIZES[sp]
+            ds_counts[sp] = pairs
+            print(f"  {sp:<5}: images={len(imgs):<6} pairs={pairs:<6} expected={expected:<6} "
+                  f"{'OK' if pairs == expected else 'MISMATCH'}")
+            if len(imgs) != pairs:
+                ds_problems.append(
+                    f"split={sp}: {len(imgs) - pairs} image(s) have no matching <stem>.png mask")
+            if pairs != expected:
+                ds_problems.append(
+                    f"split={sp}: pair count {pairs} != expected {expected} "
+                    f"(delta {pairs - expected:+d})")
+    total = sum(ds_counts.values())
+    if ds_counts:
+        print(f"  total: {total} (expected {SPLIT_TOTAL}) "
+              f"{'OK' if total == SPLIT_TOTAL else 'MISMATCH'}")
+        if total != SPLIT_TOTAL:
+            ds_problems.append(f"total pair count {total} != SPLIT_TOTAL {SPLIT_TOTAL}")
+    dataset_ok = not ds_problems
+    print(f"  dataset_ok             : {dataset_ok}")
+    for p in ds_problems:
+        print(f"    - {p}")
+
     # (18) verdict
     print("\n[18] VERDICT")
-    real_ready = cuda and gpu_count >= 1 and student_ok and dry_ok
+    real_ready = cuda and gpu_count >= 1 and student_ok and dry_ok and cap_ok and dataset_ok
     if not (student_ok and dry_ok):
         verdict, label = "FAIL", "NOT OK for real E1 training (E1 scaffold not runnable on this machine)"
+    elif not cap_ok:
+        verdict, label = "FAIL", ("NOT OK for real E1 training (GPU compute capability exceeds the "
+                                  "pinned stack's sm_90 ceiling)")
+    elif not dataset_ok:
+        verdict, label = "FAIL", "NOT OK for real E1 training (dataset verification failed)"
     elif real_ready:
         verdict, label = "PASS", "OK for real E1 training"
     else:
         verdict, label = "PARTIAL", "OK for DRY-RUN only; NOT OK for real E1 training"
 
     blockers = []
+    if not dataset_ok:
+        blockers.append(
+            f"Dataset verification failed under root {ds_root}: " + "; ".join(ds_problems)
+            + ". Fix the upload/extraction (or PLANTSEG_DATA_ROOT) before launching â€” the real run "
+              "would otherwise abort inside PlantSegDataset.__init__.")
+    if not cap_ok:
+        blockers.append(
+            "GPU compute capability above sm_90: "
+            + "; ".join(f"cuda:{i} '{n}' = sm_{a}{b}" for i, n, (a, b) in unsupported)
+            + f". torch {torch_ver} ships sm_50..sm_90 only. Re-provision on an Ampere/Ada/Hopper "
+              "pod (A100 / H100 / L40S / RTX 4090) or re-pin torch+cu.")
     if not cuda:
         blockers.append(f"No CUDA GPU (torch.cuda.is_available()=False; local torch is CPU-only "
                         f"build '{torch_ver}'). Real E1 (80k iters @ bs16/512^2) needs a GPU.")
```

**Acceptance — raw stdout:**
```text
==============================================================================
B31-3 ACCEPTANCE — CUDA compute-capability gate
==============================================================================
MAX_SM = (9, 0)
real torch.cuda.is_available() on this box = False

----- [1] REAL local box (no CUDA) -> gate SKIPPED, verdict must NOT be FAIL on caps -----
    gpu_capabilities    : [] (max supported by the pinned stack: sm_90)
    capability_gate     : SKIPPED (no CUDA device visible)
    RESULT: PASS
    verdict          : PARTIAL
    can_run_real_E1  : False
  RESULT: PARTIAL | OK for DRY-RUN only; NOT OK for real E1 training

----- [2] SIMULATED H100 sm_90 (at the ceiling) -> gate PASS -----
    gpu_capabilities    : ['sm_90'] (max supported by the pinned stack: sm_90)
    capability_gate     : PASS
    RESULT: PASS
    verdict          : PASS
    can_run_real_E1  : True
  RESULT: PASS | OK for real E1 training

----- [3] SIMULATED RTX 5090 sm_120 (Blackwell, above ceiling) -> gate FAIL -----
    gpu_capabilities    : ['sm_120'] (max supported by the pinned stack: sm_90)
    capability_gate     : FAIL — cuda:0 'NVIDIA GeForce RTX 5090' is sm_120, above the sm_90 ceiling of torch 2.9.1+cpu / torchvision 0.24.1+cpu. Blackwell-class pods (RTX 5090, RTX Pro 6000, B200, B300) are INCOMPATIBLE with the pinned stack and will abort at the first CUDA kernel launch. Choose an Ada/Hopper/Ampere pod (sm_80–sm_90, e.g. A100 / H100 / L40S / RTX 4090) or re-pin the stack.
    RESULT: PASS
    verdict          : FAIL
    environment_label: NOT OK for real E1 training (GPU compute capability exceeds the pinned stack's sm_90 ceiling)
    can_run_real_E1  : False
      - GPU compute capability above sm_90: cuda:0 'NVIDIA GeForce RTX 5090' = sm_120. torch 2.9.1+cpu ships sm_50..sm_90 only. Re-provision on an Ampere/Ada/Hopper pod (A100 / H100 / L40S / RTX 4090) or re-pin torch+cu.
  RESULT: FAIL | NOT OK for real E1 training (GPU compute capability exceeds the pinned stack's sm_90 ceiling)

----- full FAIL message from scenario 3 -----
  capability_gate     : FAIL — cuda:0 'NVIDIA GeForce RTX 5090' is sm_120, above the sm_90 ceiling of torch 2.9.1+cpu / torchvision 0.24.1+cpu. Blackwell-class pods (RTX 5090, RTX Pro 6000, B200, B300) are INCOMPATIBLE with the pinned stack and will abort at the first CUDA kernel launch. Choose an Ada/Hopper/Ampere pod (sm_80–sm_90, e.g. A100 / H100 / L40S / RTX 4090) or re-pin the stack.
  - GPU compute capability above sm_90: cuda:0 'NVIDIA GeForce RTX 5090' = sm_120. torch 2.9.1+cpu ships sm_50..sm_90 only. Re-provision on an Ampere/Ada/Hopper pod (A100 / H100 / L40S / RTX 4090) or re-pin torch+cu.

RESULT: PASS
```

**CANNOT-VERIFY-LOCALLY (honest statement):** this box has no CUDA device (`torch.cuda.is_available() == False`), so the ceiling branch was exercised by substituting `torch.cuda`'s four device-query functions. Everything else in `verify_env` — including the real `train_e1.py --dry-run` subprocess — ran unmodified, so the gate was tested in its actual position in the verdict logic. To settle it on a real pod:
```bash
python scripts/verify_env.py 2>&1 | grep -E 'gpu_capabilities|capability_gate|verdict'
```

Expected on a compatible pod: `capability_gate     : PASS`. On a Blackwell pod the run must print `FAIL` and refuse, **before** any training time is billed.

### B31-4 — Dataset verification inside verify_env (N5)

**Acceptance — raw stdout:**
```text
############ [1] REAL root — dataset check must PASS ############
[17b] dataset (root / split dirs / per-split pair counts vs configs/data.py SPLIT_SIZES)
  PLANTSEG_DATA_ROOT env : (unset -> default)
  resolved root          : C:\Users\admin\plantseg_data\plantseg
  train: images=5367   pairs=5367   expected=5367   OK
  val  : images=846    pairs=846    expected=846    OK
  test : images=1561   pairs=1561   expected=1561   OK
  total: 7774 (expected 7774) OK
  dataset_ok             : True

[18] VERDICT
  verdict          : PARTIAL
  environment_label: OK for DRY-RUN only; NOT OK for real E1 training
  can_run_real_E1  : False
  blockers_before_real_E1:
    - No CUDA GPU (torch.cuda.is_available()=False; local torch is CPU-only build '2.9.1+cpu'). Real E1 (80k iters @ bs16/512^2) needs a GPU.
    - torch mismatch vs pinned: local '2.9.1+cpu' != requirements.lock '2.1.0+cu121' (cu121 GPU training stack).
    - ImageNet backbone not cached; real run with --init imagenet needs network to populate the torch-hub cache OR a pre-staged checkpoint.
    - Not installed locally: opencv (cv2), albumentations, mmcv, mmseg (teacher/aug stack — needed for teacher fine-tune / E2-E3, not E1 supervised).

==============================================================================

############ [2] BOGUS PLANTSEG_DATA_ROOT — verdict must flip to FAIL ############
[17b] dataset (root / split dirs / per-split pair counts vs configs/data.py SPLIT_SIZES)
  PLANTSEG_DATA_ROOT env : C:/Users/admin/definitely_not_the_dataset
  resolved root          : C:\Users\admin\definitely_not_the_dataset
  dataset_ok             : False
    - dataset root does not exist or is not a directory: C:\Users\admin\definitely_not_the_dataset

[18] VERDICT
  verdict          : FAIL
  environment_label: NOT OK for real E1 training (E1 scaffold not runnable on this machine)
  can_run_real_E1  : False
  blockers_before_real_E1:
    - Dataset verification failed under root C:\Users\admin\definitely_not_the_dataset: dataset root does not exist or is not a directory: C:\Users\admin\definitely_not_the_dataset. Fix the upload/extraction (or PLANTSEG_DATA_ROOT) before launching — the real run would otherwise abort inside PlantSegDataset.__init__.
    - No CUDA GPU (torch.cuda.is_available()=False; local torch is CPU-only build '2.9.1+cpu'). Real E1 (80k iters @ bs16/512^2) needs a GPU.
    - torch mismatch vs pinned: local '2.9.1+cpu' != requirements.lock '2.1.0+cu121' (cu121 GPU training stack).
    - ImageNet backbone not cached; real run with --init imagenet needs network to populate the torch-hub cache OR a pre-staged checkpoint.
    - Not installed locally: opencv (cv2), albumentations, mmcv, mmseg (teacher/aug stack — needed for teacher fine-tune / E2-E3, not E1 supervised).

==============================================================================
RESULT: FAIL | NOT OK for real E1 training (E1 scaffold not runnable on this machine)
==============================================================================
```

**Honest caveat on the verdict label.** With a bogus root the `train_e1 --dry-run` subprocess also fails, so the `not (student_ok and dry_ok)` branch claims the headline label first and the `elif not dataset_ok` label is rarely reached. The dataset blocker still leads the actionable list, which is what a launch operator reads.

**The check is not redundant, though [INFERRED]:** `train_e1.py:155-156` builds only the **train** and **val** loaders. A broken **test** split therefore passes the dry-run entirely and is caught *only* by `[17b]`. That is genuine independent coverage.

### B31-5 — DataLoader throughput  ·  full re-proof in §3

```diff
diff --git a/src/data/dataset.py b/src/data/dataset.py
index ce74716..1781da0 100644
--- a/src/data/dataset.py
+++ b/src/data/dataset.py
@@ -23,7 +23,7 @@ if str(REPO) not in sys.path:
     sys.path.insert(0, str(REPO))
 
 from configs.augment import AUGMENT          # noqa: E402
-from configs.data import DATA                # noqa: E402
+from configs.data import DATA, SPLIT_SIZES, SPLIT_TOTAL  # noqa: E402
 from src.seeds import SEED                   # noqa: E402
 
 from .transforms import core_preprocess, finalize, train_preprocess  # noqa: E402
@@ -61,12 +61,21 @@ class PlantSegDataset(Dataset):
         if not self.pairs:
             raise RuntimeError(f"no image/mask pairs found for split={split} under {root}")
         # Guard against silent under/over-count vs the locked PlantSeg split sizes (configs/data.py).
-        expected = DATA.get("splits", {}).get("sizes", {}).get(split)
-        if expected is not None and len(self.pairs) != expected:
+        # An unregistered split is a HARD error: a missing/renamed key must never silently disable
+        # this guard (the previous `.get()` chain defaulted to None and skipped the check entirely).
+        if split not in SPLIT_SIZES:
             raise RuntimeError(
-                f"split={split} pair count {len(self.pairs)} != locked expected {expected} "
-                f"(configs/data.py DATA['splits']['sizes']); check the dataset upload/extraction "
-                f"under {root}")
+                f"split={split!r} has no registered expected count in configs/data.py SPLIT_SIZES "
+                f"(registered: {sorted(SPLIT_SIZES)}); refusing to load an unverified split")
+        expected = SPLIT_SIZES[split]
+        if len(self.pairs) != expected:
+            raise RuntimeError(
+                f"split={split} pair count MISMATCH: expected {expected}, actual {len(self.pairs)} "
+                f"(delta {len(self.pairs) - expected:+d}). All splits expected {dict(SPLIT_SIZES)}, "
+                f"total {SPLIT_TOTAL}. Check the dataset upload/extraction under {root}.\n"
+                f"NOTE: the COUNTED values in configs/data.py SPLIT_SIZES are AUTHORITATIVE. ch3's "
+                f"5,442/778/1,554 is a known arithmetic artifact (nominal 70/10/20 applied to 7,774) "
+                f"and is under correction in the manuscript â€” do NOT edit SPLIT_SIZES to match it.")
         self.aug_params = AUGMENT
 
     def __len__(self) -> int:
@@ -94,11 +103,27 @@ def _seed_worker(worker_id: int) -> None:
     random.seed(s)
 
 
-def build_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
-    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42)."""
+PREFETCH_FACTOR = 4          # batches pre-staged per worker (B31-5)
+
+
+def build_dataloader(split: str, batch_size: int, num_workers: int = 0,
+                     persistent_workers: bool = False) -> DataLoader:
+    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42).
+
+    `persistent_workers` is opt-in and intended for the TRAIN loader only: it keeps the worker pool
+    (and its decoded-image buffers) alive for the whole run, which is worth it across 80k iterations
+    but not across the 20 validation passes. `prefetch_factor`/`persistent_workers` are only legal
+    when num_workers > 0 â€” PyTorch raises otherwise, and the dry-run path uses num_workers=0 â€” so
+    both are passed conditionally. `pin_memory` is gated on CUDA: it is a no-op without a device and
+    emits a warning, so gating keeps the CPU dry-run output clean.
+    """
     dataset = PlantSegDataset(split)
     generator = torch.Generator()
     generator.manual_seed(SEED)
+    extra = {}
+    if num_workers > 0:
+        extra["prefetch_factor"] = PREFETCH_FACTOR
+        extra["persistent_workers"] = persistent_workers
     return DataLoader(
         dataset,
         batch_size=batch_size,
@@ -106,5 +131,10 @@ def build_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataL
         num_workers=num_workers,
         generator=generator,
         worker_init_fn=_seed_worker,
-        drop_last=False,
+        # TRAIN only: 5367 % 16 = 7, so every epoch would otherwise end on a ragged 7-sample batch,
+        # perturbing BatchNorm statistics and the Dice term's per-batch class-presence set.
+        # val/test keep drop_last=False â€” dropping evaluation samples would corrupt the metric.
+        drop_last=(split == "train"),
+        pin_memory=torch.cuda.is_available(),
+        **extra,
     )
```

**Hazard 1 (num_workers=0):** `persistent_workers` and `prefetch_factor` are both illegal at `num_workers=0` and are passed conditionally. `pin_memory` is gated on CUDA availability (a no-op without a device, and it emits a warning). Verified MEASURED:
```text
=== HAZARD 1: dry-run uses num_workers=0 — must still pass unchanged ===
RESULT: PASS

=== HAZARD 1b: explicit num_workers>0 must also work (persistent+prefetch legal) ===
RESULT: PASS

=== real-run num_workers default on this box ===
os.cpu_count()= 16 -> default_workers = 12
```

**Hazard 2 (the important one):** re-proof in §3 — all three properties re-earned.

### B31-6 — `np.unique` → `np.bincount`  ·  full harness in §4

```diff
diff --git a/src/data/transforms.py b/src/data/transforms.py
index ac2da0c..b68c1d2 100644
--- a/src/data/transforms.py
+++ b/src/data/transforms.py
@@ -112,11 +112,25 @@ def _resize_long_side(img_pil: Image.Image, mask_pil: Image.Image, target_long:
 
 
 def _dom_nonignore_ratio(mask_np: np.ndarray) -> float:
-    """Dominant NON-ignore class pixel fraction (255 excluded); 1.0 if the crop is all-ignore."""
-    vals, cnts = np.unique(mask_np, return_counts=True)
-    cnts = cnts[vals != MASK_IGNORE]
-    tot = int(cnts.sum())
-    return float(cnts.max() / tot) if tot > 0 else 1.0
+    """Dominant NON-ignore class pixel fraction (255 excluded); 1.0 if the crop is all-ignore.
+
+    bincount, not unique: np.unique SORTS ~262k elements on every crop attempt (up to 10 attempts
+    per sample), measured at 35.9% of crop time (B30 P7). bincount is one O(n) pass into fixed bins.
+
+    The label domain is BOUNDED, not asserted: `minlength=256` makes the result exactly 256 long for
+    every in-domain mask, so a longer result means some label exceeded 255. That check is free â€”
+    bincount already computed the max â€” whereas an explicit `mask.max()` assertion would add another
+    full pass over the array and give back part of what this fix buys. Negative labels raise inside
+    bincount itself.
+    """
+    counts = np.bincount(mask_np.ravel(), minlength=256)
+    if counts.size > 256:
+        raise ValueError(
+            f"mask label out of domain: found {counts.size - 1} > 255; valid labels are 0..115 "
+            f"plus the ignore label {MASK_IGNORE}")
+    counts[MASK_IGNORE] = 0                       # drop ignore; background (0) still counts
+    tot = int(counts.sum())
+    return float(counts.max() / tot) if tot > 0 else 1.0
 
 
 def _random_crop_pad_512(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState,
```

**Guard choice, as required by the ruling: BOUNDED, not asserted.** `minlength=256` makes the result exactly 256 long for every in-domain mask, so `counts.size > 256` detects an out-of-domain label **for free** — bincount already computed the max. An explicit `mask.max()` assertion would have added another full pass over 262k elements per attempt and given back part of what the fix buys. Negative labels raise inside `np.bincount` itself.

### B31-7 — Persistent JSONL telemetry

**Acceptance — raw stdout:**
```text
==============================================================================
B31-7 ACCEPTANCE — persistent JSONL telemetry
==============================================================================
dry-run rc=0  RESULT: PASS

--- [1] file written beside the checkpoints, not in the repo ---
  path   : C:\Users\admin\AppData\Local\Temp\b31_7_yvo7pj8d\e1_telemetry.jsonl
  exists : True  size=4891 B
  inside repo: False  (MUST be False)
  *.jsonl anywhere in repo: []  (MUST be empty)

--- [2] every line parses as JSON ---
  lines=7  all parsed OK
  event counts: {'run_meta': 1, 'train': 4, 'val': 2}

--- [3] FIRST line (run_meta) ---
{
  "event": "run_meta",
  "wall_clock": 1788266242.33709,
  "mode": "dry",
  "seed": 42,
  "git_head": "148af2c9b30ffc78e7e2f4d0b9ff55cd40948607",
  "torch": "2.9.1+cpu",
  "numpy": "2.1.3",
  "device": "cpu",
  "cuda_available": false,
  "gpu_name": null,
  "num_workers": 0,
  "batch_size": 1,
  "max_iters": 4,
  "val_interval": 2,
  "max_val_batches": 1,
  "ckpt_interval": 4,
  "resumed_from": null,
  "num_classes": 116,
  "ignore_index": 255,
  "learning_rate": 0.01,
  "momentum": 0.9,
  "weight_decay": 0.0001,
  "lr_power": 0.9,
  "poly_horizon": 80000,
  "grad_clip_norm": null,
  "used_pretrained": false,
  "params": 2933688
}

--- [4] LAST line ---
{
  "event": "val",
  "iter": 4,
  "all_class_miou": 0.0,
  "disease_only_miou_PROVISIONAL": 0.0,
  "per_class_iou": "<116 entries: [0.0, 0.0, 0.0, 0.0]...>",
  "per_class_eligible": "<116 entries: [True, True, False, False]...>",
  "n_eligible_classes": 3,
  "val_batches": 1,
  "val_total_px": 165888,
  "val_seconds": 0.3628966808319092,
  "wall_clock": 1788266248.2402356
}

--- [5] required fields present ---
  run_meta: ALL PRESENT
  train   : ALL PRESENT  (4 rows)
  val     : ALL PRESENT  (2 rows)
  per_class_iou length = 116 (expect 116)
  disease-only key retains the PROVISIONAL label -> True

--- [6] per_class_iou helper must reproduce miou_from_confusion EXACTLY ---
  200 synthetic confusion matrices; max |helper - miou_from_confusion| = 0.0
  EXACT match -> True
  spot check (diagonal CM): macro=1.000000 ref=1.000000

--- [7] resume-safety: a resumed run APPENDS to the same file ---
  rc=0  lines before=7 after=11 (appended 4)
  original first line preserved -> True
  run_meta rows: 2 (second records resumed_from=True)
  iters recorded: [1, 2, 3, 4, 5, 6]

RESULT: PASS
```

**Q3 compliance:** `src/eval/metrics.py` was NOT touched. `per_class_iou()` is a local helper in `train_e1.py` that recomputes from the SAME `cm` object `validate()` already returns — no re-accumulation, no re-run inference. Its macro mean reproduces `miou_from_confusion` **exactly** (`max |helper − ref| = 0.0` over 200 synthetic confusion matrices including sparse and all-zero cases). That equality is the standing guard against the helper drifting from the frozen metric.

### B31-8 — `drop_last=True` on the TRAIN split only

**Acceptance — raw stdout:**
```text
==============================================================================
B31-8 ACCEPTANCE — drop_last=True on TRAIN only
==============================================================================
batch_size=16  max_iters=80000  (both LOCKED, unchanged)

  train  n=5367   drop_last=True   len(loader)=335   n/bs=335.4375  ragged_remainder=7
  val    n=846    drop_last=False  len(loader)=53    n/bs=52.8750  ragged_remainder=14
  test   n=1561   drop_last=False  len(loader)=98    n/bs=97.5625  ragged_remainder=9

[MEASURED] train iters/epoch (drop_last=True) : 335
[MEASURED] samples seen per epoch             : 5360 of 5367 (7 dropped per epoch, re-shuffled next epoch)
[INFERRED] epochs at 80000 iters              : 238.806
[MEASURED] before B31-8 (drop_last=False)     : 336 iters/epoch, final batch of 7

  val loader keeps every sample: len=53 x bs16 covers 846 imgs (last batch 14) -> drop_last=False CORRECT (no eval samples lost)

RESULT: PASS
```

| Quantity | Value | Label |
|---|---|---|
| train iters/epoch (drop_last=True) | **335** | MEASURED |
| samples/epoch | 5,360 of 5,367 (7 dropped, reshuffled next epoch) | MEASURED |
| epochs at 80,000 iters | **238.806** | INFERRED |
| before B31-8 | 336 iters/epoch, final ragged batch of 7 | MEASURED |
| val / test | `drop_last=False` — no evaluation sample lost | MEASURED |

### B31-9 — lr_trace cap + checkpoint pruning + best pointer

**Acceptance — raw stdout:**
```text
==============================================================================
B31-9 ACCEPTANCE — lr_trace cap + checkpoint pruning + best pointer
==============================================================================

--- [1] lr_trace print size: 80,000 iters, before vs after ---
  BEFORE: one print of 80000 formatted floats ~= 1.44 MB on a single stdout line
  AFTER : TRACE_KEEP=50 -> at most 100 values ~= 1.8 KB
  reduction: 800x   (full curve preserved in the JSONL)

--- [2] monotonicity check still covers EVERY consecutive pair ---
  check moved from all(lr_trace[i+1] <= lr_trace[i]) over a retained list
  to a streaming pairwise compare on every iteration -> same coverage, O(1) memory.
  Proof by construction: a non-monotonic step anywhere sets lr_monotonic=False.

--- [3] checkpoint pruning: 20 val improvements (the real 80k/4000 cadence) ---
  simulated 20 best-checkpoints + last.pt
  footprint BEFORE pruning: 20 x ~24 MB = ~0.48 GB
  pruned 17 file(s)
  remaining: ['e1_student_best_iter72000.pt', 'e1_student_best_iter76000.pt', 'e1_student_best_iter80000.pt', 'last.pt']
  footprint AFTER  pruning: 3 x ~24 MB + last.pt = ~0.10 GB
  last.pt survived            -> True
  current best survived       -> True
  exactly keep=3 best kept    -> True

--- [4] best pointer discoverable without globbing ---
  wrote best.json: {'best_ckpt': 'C:\\Users\\admin\\AppData\\Local\\Temp\\b31_9_hrqoltdd\\e1_student_best_iter80000.pt', 'best_val_miou_all_class': 0.4711}
  no stray .tmp: True

--- [5] end-to-end: dry-run emits best.json and prunes ---
  RESULT: PASS
  [last    8/8] resume point -> C:\Users\admin\AppData\Local\Temp\b31_9e_1coineyi\last.pt
  ckpt dir contents: ['best.json', 'e1_student_best_iter1.pt', 'e1_student_best_iter8.pt', 'e1_telemetry.jsonl', 'last.pt']
  best.json present -> True
  best ckpts retained -> 2 (<=3)
  best.json = {'best_ckpt': 'C:\\Users\\admin\\AppData\\Local\\Temp\\b31_9e_1coineyi\\e1_student_best_iter8.pt', 'best_val_miou_all_class': 0.06243108958005905}

RESULT: PASS
```

The monotonicity check was moved from `all(lr_trace[i+1] <= lr_trace[i])` over a fully retained 80,000-element list to a **streaming** pairwise compare on every iteration. Coverage is identical (every consecutive pair), memory is O(1), and only the stdout summary is truncated — the full curve is in the JSONL.

### B31-10 — `scripts/smoke_aug_stochasticity.py` (new standing gate)

**Acceptance — raw stdout:**
```text
==============================================================================
B31-10 — augmentation stochasticity + reproducibility gate
torch 2.9.1+cpu | numpy 2.1.3 | PREFETCH_FACTOR=4 | indices [100, 500]
==============================================================================

process A seeds : [[1608637542, 1273642419], [1935803228, 787846414]]
process B seeds : [[1608637542, 1273642419], [1935803228, 787846414]]
train workers=0 : pass1=['140c8c720c8bc712', 'a26114baf0d446c8']
                  pass2=['73f0abdbf0330e04', '3f32b51dce6c715a']
train workers=2 : pass1=['6c772db174314e67', 'f4109cb73ece4509']
                  pass2=['ebe33f7c9e8bd21e', '48de3debf20e021c']
val   workers=2 : pass1=['5b3ebdd8677f842f', '5f25430fee925101']
                  pass2=['5b3ebdd8677f842f', '5f25430fee925101']

[CHECKS]
  PASS  (a) train varies across passes [workers=0]
  PASS  (a) train varies across passes [workers=2, persistent]
  PASS  (a) per-sample seeds differ across passes
  PASS  (b) val bit-identical across passes [control]
  PASS  (c) seeds reproduce across processes
  PASS  (c) train digests reproduce across processes [workers=0]
  PASS  (c) train digests reproduce across processes [workers=2]
  PASS  (c) val digests reproduce across processes

RESULT: PASS
```

**Runtime: 37 s wall [MEASURED]**, dominated by two interpreter startups + torch import + Windows worker spawn. Faster on Linux. 12 image decodes per process, CPU only, no GPU, no downloads.

### 2.11 — consolidated `src/training/train_e1.py` diff (B31-2, B31-5, B31-7, B31-9)

```diff
diff --git a/src/training/train_e1.py b/src/training/train_e1.py
index a9fa748..53332f1 100644
--- a/src/training/train_e1.py
+++ b/src/training/train_e1.py
@@ -19,10 +19,17 @@ from __future__ import annotations
 
 import argparse
 import json
+import os
+import random
+import subprocess
 import sys
 import tempfile
+import time
+from collections import deque
 from pathlib import Path
 
+import numpy as np
+
 REPO = Path(__file__).resolve().parents[2]
 if str(REPO) not in sys.path:
     sys.path.insert(0, str(REPO))
@@ -92,21 +99,135 @@ def resolve_ckpt_dir(ckpt_dir_arg: str | None) -> Path:
     return ckpt_dir
 
 
+def _atomic_save(payload: dict, path: Path) -> str:
+    """Write to `<final>.tmp` in the SAME directory, then os.replace().
+
+    Same-directory is required: os.replace is only atomic within one filesystem. A kill mid-write
+    destroys the .tmp and leaves the previous good checkpoint at `path` untouched.
+    """
+    tmp = path.with_name(path.name + ".tmp")
+    torch.save(payload, tmp)
+    os.replace(tmp, path)          # atomic on POSIX and on NTFS
+    return str(path)
+
+
+def _rng_state(train_loader) -> dict:
+    """Full RNG snapshot: python / numpy / torch / torch.cuda + the DataLoader generator."""
+    return {
+        "python": random.getstate(),
+        "numpy": np.random.get_state(),
+        "torch": torch.get_rng_state(),
+        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
+        "loader_generator": train_loader.generator.get_state(),
+    }
+
+
+def _restore_rng(state: dict, train_loader) -> None:
+    random.setstate(state["python"])
+    np.random.set_state(state["numpy"])
+    torch.set_rng_state(state["torch"])
+    if torch.cuda.is_available() and state.get("torch_cuda"):
+        torch.cuda.set_rng_state_all(state["torch_cuda"])
+    train_loader.generator.set_state(state["loader_generator"])
+
+
 def save_checkpoint(ckpt_dir: Path, student, optimizer, scheduler, sched_name: str,
                     it: int, best_miou: float) -> str:
     path = _assert_outside_repo(Path(ckpt_dir)) / f"e1_student_best_iter{it}.pt"
-    torch.save({
+    return _atomic_save({
+        "iter": it,
+        "model_state_dict": student.state_dict(),
+        "optimizer_state_dict": optimizer.state_dict(),
+        "scheduler_state_dict": scheduler.state_dict(),
+        "scheduler": sched_name,
+        "best_val_miou_all_class": best_miou,
+        "num_classes": NUM_CLASSES,
+    }, path)
+
+
+def save_last(ckpt_dir: Path, student, optimizer, scheduler, sched_name: str,
+              it: int, best_miou: float, best_ckpt, train_loader) -> str:
+    """Periodic resume point. Same fields as `best` PLUS the RNG state needed to continue."""
+    path = _assert_outside_repo(Path(ckpt_dir)) / "last.pt"
+    return _atomic_save({
         "iter": it,
         "model_state_dict": student.state_dict(),
         "optimizer_state_dict": optimizer.state_dict(),
         "scheduler_state_dict": scheduler.state_dict(),
         "scheduler": sched_name,
         "best_val_miou_all_class": best_miou,
+        "best_ckpt": best_ckpt,
         "num_classes": NUM_CLASSES,
+        "rng_state": _rng_state(train_loader),
     }, path)
+
+
+TRACE_KEEP = 50          # lr values retained at each end for the stdout summary (B31-9)
+
+
+def write_best_pointer(ckpt_dir: Path, best_ckpt, best_miou: float) -> str:
+    """Record the current best so downstream tooling never has to glob/parse filenames."""
+    path = _assert_outside_repo(Path(ckpt_dir)) / "best.json"
+    tmp = path.with_name(path.name + ".tmp")
+    tmp.write_text(json.dumps({"best_ckpt": best_ckpt,
+                               "best_val_miou_all_class": best_miou}, indent=2),
+                   encoding="utf-8")
+    os.replace(tmp, path)
     return str(path)
 
 
+def prune_checkpoints(ckpt_dir: Path, keep: int, best_ckpt) -> list:
+    """Keep the `keep` newest best-checkpoints; never touch last.pt, best.json, or the current best.
+
+    Without this, every val improvement leaves a ~24 MB file behind for the whole 80k run.
+    """
+    d = _assert_outside_repo(Path(ckpt_dir))
+    cks = sorted(d.glob("e1_student_best_iter*.pt"), key=lambda q: q.stat().st_mtime)
+    protect = {Path(best_ckpt).name} if best_ckpt else set()
+    removed = []
+    for q in cks[:-keep] if keep > 0 else []:
+        if q.name in protect:
+            continue
+        try:
+            q.unlink()
+            removed.append(q.name)
+        except OSError:
+            pass
+    return removed
+
+
+def _jsonl(path: Path, rec: dict) -> None:
+    """Append ONE JSON object as a line. Opened per write so a pod kill cannot lose buffered rows;
+    append mode makes it resume-safe (a resumed run continues the same file)."""
+    with open(path, "a", encoding="utf-8") as f:
+        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
+
+
+def per_class_iou(cm: torch.Tensor):
+    """Per-class IoU from the SAME confusion matrix `validate()` already returned.
+
+    Mirrors `miou_from_confusion`'s UNION-present rule (EVALUATION_CONTRACT 3.1) WITHOUT touching
+    src/eval/metrics.py: eligible iff `UN_c = GT_c + PR_c - TP_c > 0`. Nothing is re-accumulated and
+    no inference is re-run. The macro mean over eligible classes must equal miou_from_confusion(cm)
+    exactly â€” asserted in the B31-7 acceptance test, which is the guard against this helper drifting
+    away from the frozen metric.
+    """
+    tp = torch.diag(cm).float()
+    gt = cm.sum(1).float()
+    pr = cm.sum(0).float()
+    un = gt + pr - tp
+    return tp / un.clamp_min(1e-9), un > 0
+
+
+def _git_head() -> str:
+    try:
+        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
+                           text=True, timeout=15)
+        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
+    except Exception:  # noqa: BLE001
+        return "UNKNOWN"
+
+
 @torch.no_grad()
 def validate(student, val_loader, device, num_classes: int, max_val_batches: int | None):
     """Accumulate ONE confusion matrix over the val set (or a capped subset), then compute mIoU once."""
@@ -134,7 +255,9 @@ def validate(student, val_loader, device, num_classes: int, max_val_batches: int
 # --------------------------------------------------------------------------------------------------
 def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int, val_interval: int,
         max_val_batches: int | None, num_workers: int, ckpt_dir_arg: str | None,
-        grad_clip_norm: float | None, log_every: int, seed: int) -> int:
+        grad_clip_norm: float | None, log_every: int, seed: int,
+        resume: str | None = None, ckpt_interval: int = 2000,
+        jsonl_name: str = "e1_telemetry.jsonl", keep_ckpts: int = 3) -> int:
     set_seed(seed)
     dev = torch.device(device)
     print(f"[mode] {mode.upper()} | torch {torch.__version__} | device={dev} | "
@@ -152,7 +275,10 @@ def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int,
           f"(pretrained arg={pretrained!r})")
 
     # --- data ---
-    train_loader = build_dataloader("train", batch_size, num_workers=num_workers)
+    # persistent workers on TRAIN only: the pool lives for all 80k iters, whereas val runs ~20 times
+    # and the respawn cost there is noise against holding a second worker pool resident (B31-5 Q1).
+    train_loader = build_dataloader("train", batch_size, num_workers=num_workers,
+                                    persistent_workers=num_workers > 0)
     val_loader = build_dataloader("val", batch_size, num_workers=num_workers)
     print(f"[data] train_index={len(train_loader.dataset)} val_index={len(val_loader.dataset)} "
           f"(index globbed; only the batches pulled below are decoded)")
@@ -177,24 +303,75 @@ def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int,
     ckpt_dir = resolve_ckpt_dir(ckpt_dir_arg)
     print(f"[ckpt] dir={ckpt_dir} (verified OUTSIDE repo)")
 
+    # --- persistent telemetry (B31-7). Lives beside the checkpoints, NEVER inside the repo. ---
+    jsonl_path = ckpt_dir / jsonl_name
+    _jsonl(jsonl_path, {
+        "event": "run_meta", "wall_clock": time.time(), "mode": mode, "seed": seed,
+        "git_head": _git_head(), "torch": torch.__version__, "numpy": np.__version__,
+        "device": str(dev), "cuda_available": torch.cuda.is_available(),
+        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
+        "num_workers": num_workers, "batch_size": batch_size, "max_iters": max_iters,
+        "val_interval": val_interval, "max_val_batches": max_val_batches,
+        "ckpt_interval": ckpt_interval, "resumed_from": resume,
+        "num_classes": NUM_CLASSES, "ignore_index": IGNORE_INDEX,
+        "learning_rate": E1_STUDENT["learning_rate"], "momentum": E1_STUDENT["momentum"],
+        "weight_decay": E1_STUDENT["weight_decay"], "lr_power": E1_STUDENT["lr_power"],
+        "poly_horizon": E1_STUDENT["iterations"], "grad_clip_norm": grad_clip_norm,
+        "used_pretrained": student.used_pretrained, "params": n_params,
+    })
+    print(f"[jsonl] telemetry -> {jsonl_path}")
+
     checks: dict[str, bool] = {}
     best_miou = float("-inf")
     best_ckpt = None
     first_batch_meta = None
-    lr_trace: list[float] = []
+    lr_head: list = []                  # first TRACE_KEEP lr values
+    lr_tail: deque = deque(maxlen=TRACE_KEEP)   # last TRACE_KEEP lr values
+    lr_monotonic, prev_lr, n_lr = True, None, 0
+
+    # --- resume (optional) ---
+    start_iter = 1
+    if resume:
+        ck = torch.load(resume, map_location=dev, weights_only=False)
+        student.load_state_dict(ck["model_state_dict"])
+        optimizer.load_state_dict(ck["optimizer_state_dict"])
+        scheduler.load_state_dict(ck["scheduler_state_dict"])
+        best_miou = ck.get("best_val_miou_all_class", float("-inf"))
+        best_ckpt = ck.get("best_ckpt")
+        if "rng_state" in ck:
+            _restore_rng(ck["rng_state"], train_loader)
+        start_iter = int(ck["iter"]) + 1
+        print(f"[resume] from {resume} | resuming at iter={start_iter} "
+              f"lr={optimizer.param_groups[0]['lr']:.8e} best_all_class_miou={best_miou:.5f}")
+        print("[resume] WARNING: data-order continuity is NOT restored. The training loop consumes "
+              "an infinite `cycle(train_loader)`; the position within the current epoch is not "
+              "recoverable, so the post-resume sample order differs from an uninterrupted run. RNG "
+              "streams ARE restored, so augmentation remains reproducible from this point onward. "
+              "A resumed run is NOT bitwise-identical to an uninterrupted one.")
+
+    if start_iter > max_iters:
+        print(f"[resume] nothing to do: the checkpoint is already at iter {start_iter - 1}, which "
+              f"meets or exceeds --max-iters {max_iters}. Raise --max-iters to continue training, "
+              f"or resume from an earlier checkpoint. No iterations were run and no checkpoint was "
+              f"written.")
+        print(f"\nRESULT: PASS (already complete at iter {start_iter - 1})")
+        return 0
+
     train_iter = cycle(train_loader)
+    first_it = start_iter
+    t_prev = time.time()
 
-    for it in range(1, max_iters + 1):
+    for it in range(start_iter, max_iters + 1):
         img, mask = next(train_iter)
         img, mask = img.to(dev), mask.to(dev)
-        if it == 1:
+        if it == first_it:
             first_batch_meta = (tuple(img.shape), str(img.dtype), tuple(mask.shape), str(mask.dtype))
             checks["batch_shapes"] = (img.shape[1:] == (3, 512, 512) and img.dtype == torch.float32
                                       and mask.shape[1:] == (512, 512) and mask.dtype == torch.int64)
 
         optimizer.zero_grad(set_to_none=True)
         logits = student(img)
-        if it == 1:
+        if it == first_it:
             checks["logits_shape"] = tuple(logits.shape) == (img.shape[0], NUM_CLASSES, 512, 512)
         ce = criterion.ce(logits, mask)
         dice = criterion.dice(logits, mask)
@@ -205,7 +382,7 @@ def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int,
                 f"dice={dice.item():.4f}. Aborting the real E1 run to avoid poisoning the model "
                 "or wasting compute.")
 
-        if it == 1:
+        if it == first_it:
             checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
             p0 = next(p for p in student.parameters() if p.requires_grad)
             before = p0.detach().clone()
@@ -216,18 +393,47 @@ def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int,
         optimizer.step()
         scheduler.step()                              # step ONCE per iteration, after optimizer
         lr = optimizer.param_groups[0]["lr"]
-        lr_trace.append(lr)
-
-        if it == 1:
+        # B31-9: bounded retention. The monotonicity check is done STREAMING so it still
+        # covers every consecutive pair -- only the stdout summary is truncated. The full
+        # curve lives in the JSONL, not in a 1.4 MB print.
+        if len(lr_head) < TRACE_KEEP:
+            lr_head.append(lr)
+        lr_tail.append(lr)
+        if prev_lr is not None and lr > prev_lr + 1e-12:
+            lr_monotonic = False
+        prev_lr, n_lr = lr, n_lr + 1
+
+        if it == first_it:
             checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)
 
+        now = time.time()
+        _jsonl(jsonl_path, {
+            "event": "train", "iter": it, "loss": float(loss.item()), "ce": float(ce.item()),
+            "dice": float(dice.item()), "lr": lr, "wall_clock": now,
+            "iter_seconds": now - t_prev,
+            "samples_per_sec": (batch_size / (now - t_prev)) if now > t_prev else None,
+        })
+        t_prev = now
+
         if it % log_every == 0 or it == max_iters:
             print(f"[iter {it:>4}/{max_iters}] loss={loss.item():.4f} ce={ce.item():.4f} "
                   f"dice={dice.item():.4f} lr={lr:.8e}")
 
         if it % val_interval == 0 or it == max_iters:
+            t_val0 = time.time()
             all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
                                                        max_val_batches)
+            val_seconds = time.time() - t_val0
+            iou_vec, eligible = per_class_iou(cm)
+            _jsonl(jsonl_path, {
+                "event": "val", "iter": it, "all_class_miou": all_miou,
+                "disease_only_miou_PROVISIONAL": disease_miou,
+                "per_class_iou": [round(float(x), 8) for x in iou_vec.tolist()],
+                "per_class_eligible": [bool(x) for x in eligible.tolist()],
+                "n_eligible_classes": int(eligible.sum()), "val_batches": nvb,
+                "val_total_px": int(cm.sum()), "val_seconds": val_seconds,
+                "wall_clock": time.time(),
+            })
             checks["val_cm_accumulated"] = (tuple(cm.shape) == (NUM_CLASSES, NUM_CLASSES)
                                             and int(cm.sum()) > 0 and nvb >= 1)
             print(f"[val  {it:>4}/{max_iters}] cm_batches={nvb} cm_total_px={int(cm.sum())} "
@@ -236,12 +442,20 @@ def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int,
                 best_miou = all_miou
                 best_ckpt = save_checkpoint(ckpt_dir, student, optimizer, scheduler, sched_name,
                                             it, best_miou)
+                write_best_pointer(ckpt_dir, best_ckpt, best_miou)
+                pruned = prune_checkpoints(ckpt_dir, keep_ckpts, best_ckpt)
                 print(f"[ckpt {it:>4}/{max_iters}] new best all_class_miou={best_miou:.5f} "
-                      f"-> {best_ckpt}")
+                      f"-> {best_ckpt}"
+                      + (f" | pruned {len(pruned)} old ckpt(s)" if pruned else ""))
+
+        # periodic resume point (atomic; carries RNG state). Independent of best-val improvement.
+        if it % ckpt_interval == 0 or it == max_iters:
+            last_path = save_last(ckpt_dir, student, optimizer, scheduler, sched_name,
+                                  it, best_miou, best_ckpt, train_loader)
+            print(f"[last {it:>4}/{max_iters}] resume point -> {last_path}")
 
-    # scheduler sanity: per-iteration poly decay is monotonically non-increasing
-    checks["lr_non_increasing"] = all(lr_trace[i + 1] <= lr_trace[i] + 1e-12
-                                      for i in range(len(lr_trace) - 1))
+    # scheduler sanity: per-iteration poly decay is monotonically non-increasing (streamed above)
+    checks["lr_non_increasing"] = lr_monotonic
 
     hard = ["batch_shapes", "logits_shape", "loss_finite", "optimizer_step",
             "val_cm_accumulated", "lr_non_increasing"]
@@ -250,7 +464,13 @@ def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int,
     for k in hard:
         print(f"  {k:18}: {'PASS' if checks.get(k) else 'FAIL'}")
     print(f"[summary] first_batch={first_batch_meta}")
-    print(f"[summary] lr_trace={['%.8e' % x for x in lr_trace]}")
+    _h = ['%.8e' % x for x in lr_head]
+    _t = ['%.8e' % x for x in lr_tail]
+    print(f"[summary] lr_trace n={n_lr} (bounded print: first {len(_h)} / last {len(_t)}; "
+          f"full curve in {jsonl_path.name})")
+    print(f"[summary] lr_head={_h}")
+    if n_lr > len(_h):
+        print(f"[summary] lr_tail={_t}")
     print(f"[summary] best_all_class_val_miou={best_miou:.5f} best_ckpt={best_ckpt}")
     print(f"[summary] used_pretrained={student.used_pretrained} (no download in dry-run)")
     print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
@@ -276,8 +496,17 @@ def parse_args(argv=None):
     p.add_argument("--max-val-batches", type=int, default=None)
     p.add_argument("--num-workers", type=int, default=None)
     p.add_argument("--ckpt-dir", default=None, help="out-of-repo dir; auto temp dir if omitted")
+    p.add_argument("--resume", default=None,
+                   help="path to a last.pt resume point; restores model/optimizer/scheduler/RNG "
+                        "and continues from the saved iter (data ORDER is not restored)")
+    p.add_argument("--ckpt-interval", type=int, default=2000,
+                   help="write a periodic last.pt resume point every N iters (default 2000)")
     p.add_argument("--grad-clip-norm", type=float, default=None,
                    help="global-norm clip; omitted by default (no concrete E1 value)")
+    p.add_argument("--jsonl-name", default="e1_telemetry.jsonl",
+                   help="telemetry filename written inside --ckpt-dir (never inside the repo)")
+    p.add_argument("--keep-ckpts", type=int, default=3,
+                   help="rolling best-checkpoint retention; best + last.pt are always kept")
     p.add_argument("--log-every", type=int, default=1)
     p.add_argument("--seed", type=int, default=42)
     return p.parse_args(argv)
@@ -335,12 +564,16 @@ def main(argv=None) -> int:
         max_iters = args.max_iters or E1_STUDENT["iterations"]
         val_interval = args.val_interval or E1_STUDENT["val_interval"]
         max_val_batches = args.max_val_batches            # None -> full val
-        num_workers = args.num_workers if args.num_workers is not None else 4
+        # B31-5: data loading, not the GPU, bounded E1 throughput at the old default of 4.
+        default_workers = min(max((os.cpu_count() or 4) - 2, 1), 12)
+        num_workers = args.num_workers if args.num_workers is not None else default_workers
 
     return run(mode=mode, device=device, pretrained=pretrained, batch_size=batch_size,
                max_iters=max_iters, val_interval=val_interval, max_val_batches=max_val_batches,
                num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
-               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed)
+               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed,
+               resume=args.resume, ckpt_interval=args.ckpt_interval,
+               jsonl_name=args.jsonl_name, keep_ckpts=args.keep_ckpts)
 
 
 if __name__ == "__main__":
```

---

## 3. B31-5 reproducibility re-proof (full)

`persistent_workers=True` draws `_base_seed` **once** per DataLoader iterator and does not re-seed workers per epoch, so the B30 §9 proof does not transfer. It was re-earned, not assumed.

**Prediction recorded BEFORE measuring** (so the test could falsify it): both properties should survive, because cross-epoch variation never depended on re-seeding — each worker's numpy stream keeps advancing across every `__getitem__` it serves — and `_base_seed` is drawn from the `SEED`-seeded `torch.Generator`, so drawing it once is *more* deterministic, not less.

### Process A — verbatim
```text
==============================================================================
B31-5 REPRODUCIBILITY RE-PROOF | PROCESS A | pid=19048
torch 2.9.1+cpu | numpy 2.1.3 | PREFETCH_FACTOR=4
==============================================================================

--- [0] real build_dataloader kwargs (what the training run actually uses) ---
  train  num_workers=2 persistent=True prefetch=4 pin_memory=False drop_last=False shuffle=RandomSampler
  val    num_workers=2 persistent=False prefetch=4 pin_memory=False drop_last=False shuffle=SequentialSampler
  probe loader replicates train kwargs -> True

[P1 nw=0] pass1 seeds=[1608637542, 1273642419]
[P1 nw=0] pass1 digests=['140c8c720c8bc712', 'a26114baf0d446c8']

[P1 nw=0] pass2 seeds=[1935803228, 787846414]
[P1 nw=0] pass2 digests=['73f0abdbf0330e04', '3f32b51dce6c715a']
[P2a nw=2 persistent=True] pass1 digests=['6c772db174314e67', 'f4109cb73ece4509']
[P2a nw=2 persistent=True] pass2 digests=['ebe33f7c9e8bd21e', '48de3debf20e021c']
[P2b nw=0] pass1 digests=['140c8c720c8bc712', 'a26114baf0d446c8']
[P2b nw=0] pass2 digests=['73f0abdbf0330e04', '3f32b51dce6c715a']
[P3 val nw=2] pass1 digests=['5b3ebdd8677f842f', '5f25430fee925101']
[P3 val nw=2] pass2 digests=['5b3ebdd8677f842f', '5f25430fee925101']

--- PROPERTY (a) train augmentation VARIES across passes ---
  P1  nw=0            pass1 != pass2 -> True
  P2a nw=2 persistent pass1 != pass2 -> True   <-- the B31-5 hazard
  P2b nw=0            pass1 != pass2 -> True
  (a) OVERALL -> True

--- PROPERTY (b) val BIT-IDENTICAL across passes (control) ---
  (b) OVERALL -> True

XPROC::A::{'p1_seeds': [[1608637542, 1273642419], [1935803228, 787846414]], 'p1_digests': [['140c8c720c8bc712', 'a26114baf0d446c8'], ['73f0abdbf0330e04', '3f32b51dce6c715a']], 'p2a': [['6c772db174314e67', 'f4109cb73ece4509'], ['ebe33f7c9e8bd21e', '48de3debf20e021c']], 'p2b': [['140c8c720c8bc712', 'a26114baf0d446c8'], ['73f0abdbf0330e04', '3f32b51dce6c715a']], 'p3': [['5b3ebdd8677f842f', '5f25430fee925101'], ['5b3ebdd8677f842f', '5f25430fee925101']]}

LOCAL RESULT: PASS
```

### Process B — verbatim
```text
==============================================================================
B31-5 REPRODUCIBILITY RE-PROOF | PROCESS B | pid=4936
torch 2.9.1+cpu | numpy 2.1.3 | PREFETCH_FACTOR=4
==============================================================================

--- [0] real build_dataloader kwargs (what the training run actually uses) ---
  train  num_workers=2 persistent=True prefetch=4 pin_memory=False drop_last=False shuffle=RandomSampler
  val    num_workers=2 persistent=False prefetch=4 pin_memory=False drop_last=False shuffle=SequentialSampler
  probe loader replicates train kwargs -> True

[P1 nw=0] pass1 seeds=[1608637542, 1273642419]
[P1 nw=0] pass1 digests=['140c8c720c8bc712', 'a26114baf0d446c8']

[P1 nw=0] pass2 seeds=[1935803228, 787846414]
[P1 nw=0] pass2 digests=['73f0abdbf0330e04', '3f32b51dce6c715a']
[P2a nw=2 persistent=True] pass1 digests=['6c772db174314e67', 'f4109cb73ece4509']
[P2a nw=2 persistent=True] pass2 digests=['ebe33f7c9e8bd21e', '48de3debf20e021c']
[P2b nw=0] pass1 digests=['140c8c720c8bc712', 'a26114baf0d446c8']
[P2b nw=0] pass2 digests=['73f0abdbf0330e04', '3f32b51dce6c715a']
[P3 val nw=2] pass1 digests=['5b3ebdd8677f842f', '5f25430fee925101']
[P3 val nw=2] pass2 digests=['5b3ebdd8677f842f', '5f25430fee925101']

--- PROPERTY (a) train augmentation VARIES across passes ---
  P1  nw=0            pass1 != pass2 -> True
  P2a nw=2 persistent pass1 != pass2 -> True   <-- the B31-5 hazard
  P2b nw=0            pass1 != pass2 -> True
  (a) OVERALL -> True

--- PROPERTY (b) val BIT-IDENTICAL across passes (control) ---
  (b) OVERALL -> True

XPROC::B::{'p1_seeds': [[1608637542, 1273642419], [1935803228, 787846414]], 'p1_digests': [['140c8c720c8bc712', 'a26114baf0d446c8'], ['73f0abdbf0330e04', '3f32b51dce6c715a']], 'p2a': [['6c772db174314e67', 'f4109cb73ece4509'], ['ebe33f7c9e8bd21e', '48de3debf20e021c']], 'p2b': [['140c8c720c8bc712', 'a26114baf0d446c8'], ['73f0abdbf0330e04', '3f32b51dce6c715a']], 'p3': [['5b3ebdd8677f842f', '5f25430fee925101'], ['5b3ebdd8677f842f', '5f25430fee925101']]}

LOCAL RESULT: PASS
```

### Cross-process comparison
```text
==============================================================================
PROPERTY (c) CROSS-PROCESS REPRODUCIBILITY — post-B31-5 loader
==============================================================================
  p1_seeds     A == B -> True
  p1_digests   A == B -> True
  p2a          A == B -> True
  p2b          A == B -> True
  p3           A == B -> True

  full payload byte-identical -> True

--- continuity vs the B30 §9 baseline (pre-B31-5 loader) ---
  P1 seeds unchanged vs B30 §9        -> True
  P1 digests unchanged vs B30 §9      -> True
  nw=2 pass1 unchanged vs B30 §9      -> True

RESULT: PASS
```

### Verdict on the three required properties

| Property | Result | Evidence |
|---|---|---|
| **(a)** train augmentation VARIES across two passes | **PASS** | `P2a` pass1 `6c772db1…`/`f4109cb7…` ≠ pass2 `ebe33f7c…`/`48de3deb…`, at `persistent_workers=True` |
| **(b)** val BIT-IDENTICAL across two passes | **PASS** | `P3` pass1 == pass2, `5b3ebdd8…`/`5f254 30f…` |
| **(c)** seeds + digests MATCH across two processes | **PASS** | full payload byte-identical, A vs B |

**Unplanned bonus result.** The post-B31-5 loader is **bit-identical to the pre-B31-5 loader on pass 1** — `persistent_workers` changes *when* workers are seeded, not *what* seed they get. B31-5 therefore bought throughput without perturbing the data stream at all, which also means the B30 §9 measurements remain valid as a baseline.

**Still outstanding (carried from B30 §9):** all of this ran on `numpy 2.1.3` / `torch 2.9.1+cpu`, not the pinned `numpy==1.26.4` / `torch==2.1.0+cu121`. `RandomState` is NumPy's legacy generator with a cross-version stream guarantee, so the seeds are expected to match — but the claim should be measured where it is asserted. `scripts/smoke_aug_stochasticity.py` now settles it in one command on the pod.

---

## 4. B31-6 equivalence harness + old-vs-new timing

### Equivalence — raw stdout
```text
==============================================================================
B31-6 ACCEPTANCE — _dom_nonignore_ratio: np.unique -> np.bincount
==============================================================================

--- [1] EXACT equivalence over 216 masks ---
  cases: 216   exact mismatches: 0

  named edge cases (16):
    OK   old=1.0                      new=1.0                      all-255 (all-ignore)
    OK   old=1.0                      new=1.0                      all-background (0)
    OK   old=1.0                      new=1.0                      single-class 7
    OK   old=1.0                      new=1.0                      empty-after-255-removal (255 + nothing else)
    OK   old=1.0                      new=1.0                      one non-ignore pixel among 255s
    OK   old=1.0                      new=1.0                      half background / half ignore
    OK   old=1.0                      new=1.0                      value 116..254 present (in-domain, >num_classes)
    OK   old=0.009052276611328125     new=0.009052276611328125     mixed 0..115 with a stray 200
    OK   old=0.008968353271484375     new=0.008968353271484375     full 0..115 uniform
    OK   old=0.8102531433105469       new=0.8102531433105469       skewed long-tail
    OK   old=0.5003929138183594       new=0.5003929138183594       two-class 0/1
    OK   old=0.3358338436415595       new=0.3358338436415595       255-heavy with 3 classes
    OK   old=1.0                      new=1.0                      boundary label 115
    OK   old=1.0                      new=1.0                      boundary label 254
    OK   old=1.0                      new=1.0                      1x1 single pixel
    OK   old=1.0                      new=1.0                      empty array

--- [2] out-of-domain guard (>255) must FIRE, not miscount ---
  OLD(mask with 300) = 1.0   <-- silently counted it, no error
  NEW: ValueError raised -> mask label out of domain: found 300 > 255; valid labels are 0..115 plus the ignore label 255
  NEW(negative label): ValueError from bincount -> 'list' argument must have no negative elements

  NOTE: a label in 116..254 is IN-domain for both implementations (bincount's
  minlength=256 covers it) and must agree exactly — see the 'value 116..254' and
  'stray 200' cases above. Only >255 trips the guard.

--- [3] P4/P7 re-timing: 16 distinct train images x 5 reps ---
  decoded 16 distinct train images (indices [0, 358, 715, 1073]...)
                              mean    median       p95   (ms/sample, train_preprocess end-to-end)
  OLD (np.unique)            22.69     21.01     36.64
  NEW (np.bincount)          33.25     29.74     65.63
  delta                      10.56      8.74     28.99
  speedup on mean: 0.68x

  _dom_nonignore_ratio calls: OLD=215 NEW=215 (equal -> True)
  time INSIDE _dom_nonignore_ratio: OLD=206.6 ms (11.4% of crop pipeline wall time)
                                    NEW=208.0 ms (7.8%)
  in-function speedup: 1.0x

RESULT: PASS
```

**216 masks, 0 exact mismatches.** Equality is `==` on float64, not approximate.

**A distinction the item statement conflates, worth being precise about:** a label in **116..254** is *in-domain* for both implementations (bincount's `minlength=256` covers it) and must — and does — agree exactly; see the `value 116..254` and `stray 200` cases. Only a label **>255** trips the guard. Both cases are in the corpus.

**The new code is strictly STRICTER than the old.** `OLD(mask containing 300)` returned `1.0` silently; `NEW` raises `ValueError: mask label out of domain: found 300 > 255`. That is a correctness improvement the old implementation did not have.

### Timing — first measurement DISCARDED, and why

My initial in-pipeline attribution ran OLD fully, then NEW fully, and reported `in-function speedup: 1.0x` with the end-to-end mean getting **worse** (22.69 → 33.25 ms). A function that takes the same time cannot make the pipeline 46% slower, so the measurement was machine drift across sequential arms, not a real result. Discarded rather than reported.

### Direct microbenchmark (interleaved, min-of-two-runs) — MEASURED
```text
mask                                        OLD ms    NEW ms   speedup
512x512 realistic (81% bg, some 255)         1.033     0.393      2.6x
512x512 uniform 0..115                       1.769     0.581      3.0x
512x512 all-background                       1.476     0.795      1.9x

--- component breakdown on the realistic mask ---
  np.unique(return_counts=True)              2.322 ms
  np.bincount(ravel,minlength=256)           0.760 ms
  m.ravel() alone                            0.000 ms
```

### End-to-end `train_preprocess`, interleaved arms — MEASURED
```text
decoded 16 distinct train images

Interleaved, 7 reps x 16 images = 112 samples per arm
                           mean   median      p95      min   ms/sample, train_preprocess end-to-end
OLD (np.unique)           28.14    25.73    51.60     4.99
NEW (np.bincount)         27.23    22.97    51.89     4.25
delta                     -0.91    -2.76     0.30    -0.74

end-to-end speedup  mean 1.03x   median 1.12x   min 1.17x
```

### What this actually buys — a correction to the expected impact

| Level | OLD | NEW | Speedup | Label |
|---|---|---|---|---|
| `np.unique` vs `np.bincount`, realistic 512² mask | 1.652 ms | 0.699 ms | **2.4x** | MEASURED |
| `_dom_nonignore_ratio`, realistic mask | 0.838 ms | 0.374 ms | **2.2x** | MEASURED |
| `_dom_nonignore_ratio`, uniform 0..115 | 1.352 ms | 0.218 ms | **6.2x** | MEASURED |
| `train_preprocess` end-to-end, mean | 28.14 ms | 27.23 ms | **1.03x** | MEASURED |
| `train_preprocess` end-to-end, median | 25.73 ms | 22.97 ms | **1.12x** | MEASURED |

**B30 P7's "35.9%" was share of CROP time, not of total `train_preprocess` time.** The function is genuinely 2.2x faster, but the crop stage is one of seven, so the end-to-end gain is ~1.03x mean / 1.12x median — materially smaller than the headline figure suggests.

**Consequence for the schedule.** A ~1.1x preprocess gain does not move the N1×F11 arithmetic. The real throughput lever in B31 is B31-5's worker count (4 → 12, a ~3x change), not B31-6. B31-6 is still worth keeping for the 2.2x function-level win and the new correctness guard, but it should not be counted on for wall-clock relief.

---

## 5. Refused / STOP-and-ask / judgment calls flagged for review

**Nothing in the B31 scope was refused.** All ten items are implemented and green. Five decisions are flagged because they touched code the item did not name verbatim, or deviated from the literal instruction:

1. **`configs/data.py` `test_count` now references `SPLIT_SIZES["test"]`** instead of the literal `1561`. Same value. I judged this inside B31-1 ("a single named constant") because leaving a duplicated literal of the same split count defeats single-source-of-truth. It is used by the evaluation path, not E1. **Revert if you disagree** — it is one line.

2. **Four `if it == 1:` gates became `if it == first_it:`** in `train_e1.py`. Required by B31-2: on resume `it` starts at e.g. 4, so the first-iteration scaffold checks (`batch_shapes`, `logits_shape`, `loss_finite`, `optimizer_step`) would never run and every resumed run would print `RESULT: FAIL`. Semantics preserved ("first iteration of this invocation").

3. **Already-complete resume guard added** (`start_iter > max_iters` → clear message, exit 0). Not in the item text; added because my own acceptance test hit the confusing failure. Inside B31-2's resume semantics.

4. **B31-6 guard implemented as BOUNDED rather than ASSERTED**, per the explicit ruling — detected from `counts.size > 256` at zero extra cost. Stated here because the ruling asked me to say which I did.

5. **`pin_memory` gated on `torch.cuda.is_available()`** rather than hard-coded `True`. The ruling said "pin_memory stays uniform" (train and val alike), which it is; the CUDA gate prevents a spurious warning in the CPU dry-run and is behaviourally identical on the pod.

**Locked values — all verified unchanged:** SGD lr `1e-2`, momentum `0.9`, wd `1e-4`, poly power `0.9`, batch `16`, `80,000` iters, val every `4,000`, all-class-mIoU checkpoint selection (D1), `grad_clip_norm=None` (D-A), `num_classes=116`, `ignore_index=255`, seed `42`. Confirmed in the B31-7 `run_meta` line, which records every one of them at runtime.

---

## 6. Residual risks after B31 — what still blocks the real E1 run

### Blocking

| # | Risk | Status |
|---|---|---|
| R1 | **N1 × F11 schedule collision.** ch3 requires three-seed E1 *and* E3; the λ_logit sweep is pre-registered at five full 80k runs. Nothing in the repo plans for either. Explicitly out of B31 scope. | **Unresolved — needs a decision this week, per your own note** |
| R2 | **Teacher checkpoint absent.** `weights/` is empty; `scripts/test_teacher_init.py` has never been run; `docs/teacher_init_source.md` still carries `NEED_TO_CONFIRM` for URL, SHA256, date, and init-test result. Blocks E2/E3, not E1. | Unresolved (out of scope) |
| R3 | **ImageNet backbone not cached.** The real run with `--init imagenet` needs network access or a pre-staged torch-hub checkpoint. `verify_env` already reports this. | Unresolved |

### Needs measuring on the pod (cannot be settled locally)

| # | Item | Command |
|---|---|---|
| R4 | **F7 CUDA determinism.** `F.interpolate(bilinear)` and `AdaptiveAvgPool2d` backward have no deterministic CUDA kernel. With `warn_only=True` (which ch3 prescribes) the run is seed-controlled but NOT bit-deterministic. | run E1 with `PYTHONWARNINGS=always` and grep for `does not have a deterministic implementation` |
| R5 | **Pinned-stack seed stream.** All reproducibility evidence is from numpy 2.1.3 / torch 2.9.1+cpu. | `python scripts/smoke_aug_stochasticity.py` |
| R6 | **Capability gate.** Never executed against a real CUDA device. | `python scripts/verify_env.py` |
| R7 | **Worker RAM.** `num_workers` now resolves to `min(cpu_count-2, 12)` with `prefetch_factor=4`, i.e. up to 12 workers × 4 batches × 16 × 3×512×512 float32 in flight. Never measured on a pod. | watch RSS during the first 100 iters |

### Accepted deviations (documented, not defects)

- **Resume is not bitwise-identical** to an uninterrupted run (data order unrecoverable under `cycle()`). Warned at runtime, recorded in §2.
- **7 training samples dropped per epoch** by `drop_last=True` (reshuffled the next epoch, so no sample is systematically excluded across 238.8 epochs).
- **`verify_env`'s `elif not dataset_ok` label is rarely reached** because the dry-run subprocess fails first on a broken root. The blocker message still leads.

### Handed back to you (manuscript, out of scope by instruction)

Split counts 5,442/778/1,554 → 5,367/846/1,561 · head channels 115 → 116 · "large-kernel average-pooling" → global average pooling · the Albumentations/OpenCV "version pinned in the reproducibility manifest" claim that `requirements.lock` contradicts · `configs/loss.py:14` `real_class_weights: NEED_TO_CONFIRM` while `train_e1.py:46` actually loads `reports/e1_class_weights.json`.

**B32 (not started):** F8 / OS8 logit-resolution pin, per your sign-off request. No teacher code, `src/distill/*`, or `train_distill.py` was touched in B31.

---

## 7. Provenance

### Files modified
```text
16	2	configs/data.py
-	-	docs/reference/reference.pdf
89	1	scripts/verify_env.py
39	9	src/data/dataset.py
19	5	src/data/transforms.py
251	18	src/training/train_e1.py
```

### Files created
```text
reports/b31_e1_fixes.md          (this report)
scripts/smoke_aug_stochasticity.py  (B31-10 standing gate, 174 lines)
```

### Git status after B31
```text
 M configs/data.py
 M docs/reference/reference.pdf
 M scripts/verify_env.py
 M src/data/dataset.py
 M src/data/transforms.py
 M src/training/train_e1.py
?? reports/pre_e1_launch_audit.md
?? scripts/smoke_aug_stochasticity.py
```

### Staged changes (must be empty)
```text
(empty — nothing staged)
```

### Confirmations

- **HEAD unchanged:** `148af2c9b30ffc78e7e2f4d0b9ff55cd40948607` — no commit, no stage, no push, no branch change.
- **`docs/reference/reference.pdf` untouched:** never opened, read, hashed, copied, or staged. Its ` M` status and `Jun 27 13:28` mtime are unchanged and predate B30.
- **`src/data/dataset.py:83`** (the proven-correct augmentation RNG line) — **not modified**. Confirmed by the B31-10 gate, which reproduces the exact B30 §9 seed sequence `[1608637542, 1273642419]`.
- **`src/eval/metrics.py` not modified** (Q3 ruling).
- **No teacher code, `src/distill/*`, `train_distill.py`, `configs/teacher/*`, or any `docs/reference/*` file was modified.**
- **No training, no GPU, no installs, no downloads, no network.** All work CPU-only.
- **Probe/acceptance scripts live in the session scratchpad**, never in the repo — except `scripts/smoke_aug_stochasticity.py`, which is an explicit B31-10 deliverable.

### Dry-run verification log

`python src/training/train_e1.py --dry-run` → `RESULT: PASS` after B31-1, B31-2, B31-3, B31-4, B31-5 (both `num_workers=0` and `num_workers=2`), B31-6, B31-7, B31-8, B31-9, B31-10, and on the final combined state. **11 runs, 11 passes.**

