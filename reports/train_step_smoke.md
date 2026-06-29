# E1 Train-Step Smoke (B17, Week 2 · Task 1)

**RESULT: ✅ PASS** — wiring / gradient-flow only. **NOT real E1 training** (no ImageNet init, no real
class weights, no multi-scale aug, no checkpoint, no validation, no epoch).

_Generated: 2026-06-28 by `scripts/smoke_train_step.py` · Anaconda Python (torch 2.9.1+cpu) · CPU ·
`PYTHONIOENCODING=utf-8` · no downloads/installs · seed 42._

---

## What this proves
The E1 components compose into one working train step on CPU: `build_student()` (weights=None) →
`build_dataloader("train", 2)` → forward → `CombinedCEDiceLoss(weight=None, ignore_index=255)` →
backward → SGD step, with finite loss, full gradient flow, and parameter updates. Loss decreases on a
fixed batch (sanity signal). This is a pre-flight wiring check, **not** the E1 experiment.

## Results
| Item | Value |
|---|---|
| Student | `PlantSegStudent`, **2,933,688** params (all trainable), `used_pretrained=False` |
| Optimizer | SGD `lr=0.01`, `momentum=0.9`, `weight_decay=1e-4` (from `configs/e1_student.py`) |
| Batch — image | `(2, 3, 512, 512)` `float32` ✅ |
| Batch — mask | `(2, 512, 512)` `int64` ✅; unique values `[0, 87, 109, 255]` (bg 0 + 2 diseases + 255 pad) |
| Forward — logits | `(2, 116, 512, 512)` `float32`, finite ✅ |
| Loss (step 0) | **CE = 5.1344** · **Dice = 0.9825** · **CE+Dice = 6.1169** (finite scalar) |
| Gradient flow | **176/176** trainable param-tensors received grads · **0** None · **0** non-finite ✅ |
| Optimizer update | `features.0.0.weight` changed after `step()` → **params update** ✅ |
| 3-step loss trajectory | `[6.1169, 5.7689, 5.2175]` → post-step `4.5258` |
| Trajectory sanity | **non-increasing = True**, decreased overall = True (sanity only, **non-gating**) |

### Gating checks (all PASS)
`img_shape` ✅ · `mask_shape` ✅ · `logits_shape` ✅ · `loss_finite` ✅ · `grad_flow` ✅ · `opt_updates` ✅.
Trajectory is a **non-gating** sanity signal (here it strictly decreased — gradients + optimizer confirmed
effective; no diagnosis needed).

> Note: "176" counts trainable **parameter tensors** (every one received a finite grad), not the 2.93M
> scalar parameters. The mask's `255` is the preprocessing **pad/ignore** region (added by
> `transforms.core_preprocess`), correctly excluded by CE (`ignore_index=255`) and Dice (validity mask).

## Scope caveats (per B17 spec) — what is acceptable here vs required for real E1
- **`weight=None` is acceptable for B17 only.** The E1 supervised CE is **class-weighted** (√ inverse-
  frequency); the **real CE class weights** (computed over the **full train set**, median-normalized) are
  **still required before the real E1 training run** — currently `NEED_TO_CONFIRM` (`configs/loss.py`).
- **ImageNet initialization is a carry-forward.** The student was built with `weights=None` (no download);
  real E1 uses `MobileNet_V3_Large_Weights.IMAGENET1K_V2` (wire in the pinned/connected env before the run).
- **Full training augmentation is a carry-forward.** The loader applies joint flips/rotation + image-only
  hue/sat; ch3's **multi-scale random-resized-crop** is declared in `configs/augment.py` but not yet wired
  into `transforms.py` — add before the real E1 run.
- **Not real training:** no checkpoint saved, no validation loop, no epoch, single fixed batch, 3 steps.

## Alignment
Consistent with the Pre-B17 Ch3 alignment audit (PASS): ch3 E1 = FP32 MobileNetV3-L + LR-ASPP baseline,
CE+Dice, `ignore_index=255`, `num_classes=116` (bg 0 + diseases 1–115), no distillation/quantization in E1.
B17 exercises exactly these mechanics at smoke scale.

## Files
- `scripts/smoke_train_step.py` (new) · `reports/train_step_smoke.md` (this).
- No existing code/config/dataset modified. Nothing committed.

## Recommended next step
**E1 training-loop implementation** (Week 2, Task 2): build the actual training loop/engine
(`src/training/`) wiring student + dataloader + CombinedCEDiceLoss + SGD/poly schedule + val@4000 +
best-val-mIoU checkpointing per `configs/e1_student.py` — **first resolving the real-E1 carry-forwards**
above (real CE class weights, ImageNet init, multi-scale RRC aug). The real run itself is compute/env-gated
(80,000 iters @ 512×512, batch 16) and out of local-CPU scope.

_`docs/reference/reference.pdf` untouched (deferred). No training beyond this smoke._
