# Pre-B19 E1 Training-Loop-Scaffold Readiness Audit

**RESULT: ✅ PASS — B19 (E1 training-loop scaffold, dry-run) MAY PROCEED.**
**This is a readiness audit — NOT training.** No model was trained, no real validation ran, the
dataset was not scanned, no weights were downloaded, no GPU was used, and nothing was committed.

_Generated: 2026-06-30. Verification: `C:\Users\admin\anaconda3\python.exe` (torch **2.9.1+cpu**,
CUDA **unavailable**), CPU, dummy tensors + the real `reports/e1_class_weights.json`. Verification
script lived in the session scratchpad only (NOT added to the repo)._

---

## 0. B18d commit status (precondition)

**B18d (ImageNet student-init wiring) IS committed** — B19 is unblocked on this precondition.

| Field | Value |
|---|---|
| Full hash | **`895e60819ba75c677a1d4bfbc71b3f7667eb538f`** |
| Short hash | **`895e608`** |
| Subject | `wire ImageNet student init (B18d)` |
| Date | 2026-06-30T13:39:36+08:00 |
| Files in commit | `src/models/student.py` (+54/-11), `reports/imagenet_init_wiring.md` (+65) |
| Working-tree state of `src/models/student.py` | clean (committed; not dirty) |

`git status --porcelain` shows only ` M docs/reference/reference.pdf` (pre-existing deferred file —
untouched). B18d's `student.py` changes are fully committed, so the B19 scaffold can rely on
`build_student(pretrained=...)` as the audited HEAD defines it.

---

## 1. Verification evidence (scratchpad, read-only) — 16/16 PASS

| Check | Result | Evidence |
|---|---|---|
| weights length 116 | ✅ | `len=116` |
| weights all finite | ✅ | min=0.0370, median≈1.00*, max=9.5915 |
| index 0 = background | ✅ | `w[0]=0.036954` |
| `CombinedCEDiceLoss(weight=w)` runs, scalar finite | ✅ | `loss=6.3040`, `dim()==0` |
| CE weight is a registered buffer | ✅ | `named_buffers()` contains `ce.weight` |
| CE weight buffer len-116 finite | ✅ | — |
| CE weight **moves with `.to(device)`** | ✅ | registered buffer ⇒ `nn.Module.to()` relocates it (host is CPU-only) |
| accumulated all-class mIoU | ✅ | **0.55556** (one CM over both batches) |
| mean-of-per-batch mIoU | ✅ | **0.45833** |
| accumulation ≠ mean-of-per-batch | ✅ | 0.55556 vs 0.45833 ⇒ **accumulation REQUIRED** |
| absent-GT class skipped | ✅ | `present = row>0` excludes GT-absent class (mirrors val-missing class 68 / test-missing class 41) |
| disease-only excludes background | ✅ | disease-only mIoU=**0.50000** (class 0 dropped) |
| `confusion_matrix` scales to C=116 | ✅ | shape `(116,116)` |
| `torch.optim.lr_scheduler.PolynomialLR` available | ✅ | present in torch 2.9.1 |
| PolynomialLR matches manual poly formula | ✅ | sched lr@it5 `9.99943750e-03` == manual |
| SGD constructs from `configs/e1_student.py` | ✅ | lr=0.01, momentum=0.9, weight_decay=1e-4 |
| ImageNet torch-hub cache present | ℹ️ **False** (informational, not a gate) | `…/.cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth` absent |

\* The artifact is median-normalized over present classes (all 116 present) so the construction median
is 1.0; `torch.median` on an even count (116) returns the lower of the two middle values (0.9963).
Cosmetic — the weights are unchanged and correct.

Poly-LR sanity (base 1e-2, power 0.9, 80 000 iters): it=0 → `1.00e-2`, it=40 000 → `5.36e-3`,
it=79 999 → `3.87e-7`.

---

## 2. Exact B19 call signatures (audited)

```python
# --- seed (src/seeds.py) ---
from src.seeds import set_seed
set_seed(42)                                   # returns 42; seeds python/numpy/torch(+cuda), cuDNN deterministic

# --- model (src/models/student.py @ 895e608) ---
from src.models.student import build_student
# DRY-RUN (B19): random init, no download, CPU/offline-safe
student = build_student(pretrained=False).to(device)            # num_classes defaults to 116
# REAL E1 RUN: ImageNet backbone per configs/e1_student.py["init_weights"]
# student = build_student(pretrained="torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2").to(device)
#   (also accepts True / "IMAGENET1K_V2"); student.used_pretrained reports which path was taken

# --- dataloaders (src/data/__init__.py -> dataset.py) ---
from src.data import build_dataloader, NUM_CLASSES            # NUM_CLASSES == 116
train_loader = build_dataloader("train", batch_size=E1_STUDENT["batch_size"], num_workers=N)  # shuffles
val_loader   = build_dataloader("val",   batch_size=VAL_BS,                  num_workers=N)  # deterministic core_preprocess, no shuffle

# --- loss (src/training/losses.py) ---
import json, torch
w = torch.tensor(json.load(open("reports/e1_class_weights.json"))["weights"], dtype=torch.float32)  # len-116, idx 0..115
from src.training.losses import CombinedCEDiceLoss
criterion = CombinedCEDiceLoss(weight=w, ignore_index=255).to(device)   # weight buffer follows .to(device)
loss = criterion(logits, mask)     # logits [B,116,512,512] float, mask [B,512,512] int64 -> scalar

# --- optimizer (configs/e1_student.py) ---
optimizer = torch.optim.SGD(student.parameters(),
                            lr=E1_STUDENT["learning_rate"],      # 1e-2
                            momentum=E1_STUDENT["momentum"],     # 0.9
                            weight_decay=E1_STUDENT["weight_decay"])  # 1e-4

# --- scheduler (poly, per-iteration) ---
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=E1_STUDENT["iterations"], power=E1_STUDENT["lr_power"])  # 80000, 0.9
# Fallback if PolynomialLR absent (older torch):
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: (1 - it/80000)**0.9)

# --- metrics / val-mIoU (src/eval/metrics.py) — accumulate ONE confusion matrix ---
from src.eval.metrics import confusion_matrix, _miou_from_cm   # consider adding a public miou_from_confusion wrapper
cm = torch.zeros(116, 116, dtype=torch.long)
for img, mask in val_loader:
    pred = student(img.to(device)).argmax(1).cpu()             # logits -> class indices (REQUIRED)
    cm += confusion_matrix(pred, mask, 116, ignore_index=255)
all_class_miou_val, per_class_iou = _miou_from_cm(cm)                                  # selection metric
disease_idx = torch.tensor([c for c in range(116) if c != 0])
disease_miou_val, _ = _miou_from_cm(cm, disease_idx)                                   # PROVISIONAL secondary
```

Notes carried from the audit:
- `confusion_matrix(pred, target, …)` — argument order is **(pred, target)**; pass argmaxed predictions.
- `_miou_from_cm` is underscore-private; B19 may import it as-is **or** add a 1-line public
  `miou_from_confusion(cm, class_indices=None)` to `src/eval/metrics.py` (that small source edit is
  B19's, gated — not done in this read-only audit).
- The CE `weight` is a registered buffer, so a single `criterion.to(device)` is sufficient — do **not**
  separately re-create the tensor on device.

---

## 3. Required conclusions

1. **May B19 proceed?** **Yes** — as a plan-gated **dry-run** scaffold. B18d is committed (`895e608`);
   every loss/metrics/optimizer/scheduler/seed/checkpoint API the loop needs is verified above.

2. **Exact B19 call signatures** — see §2 (model, dataloader, loss, metrics, optimizer, scheduler, seed).

3. **Validation mIoU from one accumulated confusion matrix over the whole val set?** **Yes.** Proven:
   accumulating a single CM gives 0.55556, whereas averaging per-batch mIoUs gives 0.45833 — they are
   not equal, and the accumulated value is the correct dataset-level metric. Accumulate `cm` over all
   val batches, compute IoU once. GT-absent classes (val lacks class 68) are auto-skipped by
   `present = row>0`, satisfying the absent-class NaN/skip requirement.

4. **Checkpoint selection on all-class val mIoU, disease-only logged secondary/provisional?** **Yes.**
   Select `checkpoint_selection="best_val_miou"` on **all-class** val mIoU (the settled metric). Log
   **disease-only** mIoU as a **PROVISIONAL** secondary (background-exclusion convention is
   open_questions #2; `reduce_zero_label=False` is settled on the data side, but the reporting
   exclusion is a reporting choice).

5. **Iteration-based with a cycling dataloader (not epoch-based)?** **Yes.** The recipe is iteration-
   budgeted (80 000 iters, validate every 4 000). Use an infinite `cycle(train_loader)` generator and
   loop over iterations. (~5 367/16 ≈ **335 iters/epoch** ⇒ 80 000 iters ≈ **239 epochs**, **20**
   validations.)

6. **Poly-LR per-iteration?** **Yes.** Call `scheduler.step()` once per training iteration (after
   `optimizer.step()`), with `PolynomialLR(total_iters=80000, power=0.9)` — verified to match the
   closed-form `lr = 1e-2·(1 − it/80000)^0.9`.

7. **Grad clipping omitted for E1 dry-run unless a concrete E1 value exists?** **Yes — omit it.**
   `configs/e1_student.py` lists `gradient_clipping="global_norm"` but **under "Distillation stages
   (E2/E3) shared mechanics"**, and **no numeric clip value exists anywhere** in the configs/contract.
   Omit clipping for the E1 dry-run; flag "does E1 apply global-norm clipping, and at what max_norm?"
   as an open item for the real E1 run. If later confirmed:
   `torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=<value>)` before `optimizer.step()`.

8. **What still blocks the real E1 run?** (none block the B19 dry-run)
   - **Compute/GPU** — host is **CPU-only** (`torch 2.9.1+cpu`, `cuda_available=False`); 80 000 iters at
     bs 16 / 512² is infeasible on CPU. The real run needs a GPU.
   - **ImageNet cache not populated** — `…/checkpoints/mobilenet_v3_large-5c1a4163.pth` is **absent**;
     the real run's `pretrained="…IMAGENET1K_V2"` must populate the torch-hub cache (download) or have
     network. (Dry-run uses `pretrained=False`, so unaffected.)
   - **Grad-clip value for E1** — unresolved (only matters if E1 uses clipping; see #7).
   - **Disease-only reporting convention** — open_questions #2 (reporting choice, non-blocking).
   - **Strict-G0, non-blocking for E1** — platform verify (`scripts/verify_env.py` +
     `reports/platform_verify.md` absent), teacher-init weights (env-gated; needed for E2/E3, not E1),
     GitHub remote/push.

9. **Exact next safe B19 prompt** — see §4.

---

## 4. Exact next safe B19 prompt (ready to paste)

> We are proceeding to **B19 — E1 training-loop scaffold (dry-run only)**, plan-gated. Read-first:
> `reports/e1_training_loop_readiness.md`, `configs/e1_student.py`, `src/training/losses.py`,
> `src/eval/metrics.py`, `src/data/dataset.py`, `src/models/student.py`, `reports/e1_class_weights.json`.
> Then **show a PLAN and wait for my `go`** before writing/running anything.
>
> Scope of B19 (scaffold, NOT a real run): create `src/training/train_e1.py` (and, if you choose, a
> 1-line public `miou_from_confusion(cm, class_indices=None)` in `src/eval/metrics.py`) implementing:
> `set_seed(42)`; `build_student(pretrained=False)` for the dry-run (keep the ImageNet path from
> `configs/e1_student.py["init_weights"]` reachable via a flag for the real run); `build_dataloader`
> for train (bs from config) + val; `CombinedCEDiceLoss(weight=<loaded from reports/e1_class_weights.json>,
> ignore_index=255).to(device)`; `SGD` from `configs/e1_student.py`; `PolynomialLR(total_iters=80000,
> power=0.9)` stepped **per iteration**; an **iteration-based** loop with a **cycling** train loader;
> periodic validation that **accumulates ONE confusion matrix** over the val set and reports
> **all-class** val mIoU (selection metric) + **disease-only** val mIoU (PROVISIONAL secondary);
> **best-all-class-val-mIoU** checkpoint selection; and logging.
>
> Dry-run constraints: CPU only; `pretrained=False`; cap at a few iterations with a tiny
> `val_interval` override; write any checkpoint to the **session scratchpad only — never into the repo**;
> do **not** scan the full dataset beyond what a few iterations pull; do **not** download ImageNet
> weights; do **not** train for real. Approved repo outputs for B19: `src/training/train_e1.py`
> (+ optional `src/eval/metrics.py` helper) and a `reports/e1_train_scaffold_dryrun.md` report.
>
> Guardrails: do not touch `docs/reference/reference.pdf` or `docs/reference/context.md` or teacher-init
> files; explicit-path staging only; **commit only when I ask**; use `C:\Users\admin\anaconda3\python.exe`
> with `PYTHONIOENCODING=utf-8` for any torch run.

**If B19 were NOT ready** (it is), the smallest correction would be: commit B18d first — but B18d is
already committed at `895e608`, so no correction is required.

---

## Provenance / guardrails honored
Read-only audit. Only this one file (`reports/e1_training_loop_readiness.md`) was written to the repo;
the verification script stayed in the session scratchpad. `docs/reference/reference.pdf` untouched
(still the pre-existing deferred ` M`); `docs/reference/context.md` and teacher-init files untouched;
no source/config/model/data/loss files modified; no dataset scan; no download; no GPU; no checkpoint in
the repo; nothing staged or committed; Claude memory and `~/.claude` untouched.
