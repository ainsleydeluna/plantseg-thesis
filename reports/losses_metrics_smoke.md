# Losses & metrics smoke test — B11

**RESULT: PASS**  
_Generated: 2026-06-27 by `scripts/smoke_losses_metrics.py` (dummy + one real val batch; no training, no backward)._

PASS is gated ONLY on: finite losses, valid all-class mIoU, correct `255` handling, valid output
shapes/dtypes, no NaN/Inf. Provisional disease-only mIoU does **not** gate PASS.

## Loss formulas (implemented / scaffolded — `src/training/losses.py`)
- **Weighted CE** — `F.cross_entropy(weight, ignore_index=255)`. Class weights via
  `compute_class_weights`: `sqrt(total / (num_classes · count_c))` over non-255 pixels, median-normalized
  (**SMOKE-ONLY batch-derived; real train-set weights NEED_TO_CONFIRM**).
- **Soft Dice** — on `softmax(logits)`, per-class macro over classes **present in the batch** (absent
  excluded), smoothing **1e-5** (num+denom), validity-masked (255 excluded), unweighted.
- **Combined** — `L_sup = L_CE + L_Dice` (equal weight) — E1 supervised objective.
- **Logit KD KL** — `KL(softmax(teacher/T) ‖ log_softmax(student/T)) · T²`, averaged over valid (non-255)
  pixels, **T=4**. `λ_logit` (term weight) = **NEED_TO_CONFIRM** (validation sweep).
- **CWD** — `cwd_channelwise_kl` is a **PLACEHOLDER/scaffold only** (raises `NotImplementedError`); full
  channel-wise feature distillation + 160→320 projection + E3 wiring deferred to the E3 blocker.

## Metrics (confusion-matrix based — `src/eval/metrics.py`)
| Item | Status |
|---|---|
| num_classes used | **116** (from `configs/data.py`) |
| ignore_index handling | **255 excluded** from every computation — verified: CM total = non-255 pixel count (345088 = 345088) |
| all-class mIoU | macro IoU over classes 0–115 present in GT — **implemented** |
| per-image mIoU | implemented (paired-test unit; `class_indices` selectable) |
| `reduce_zero_label` | **NEED_TO_CONFIRM — NOT used** |
| disease-only mIoU | **PROVISIONAL / NEED_TO_CONFIRM** — excludes index **0** (empirical background); convention not yet settled (open_questions #2) |

## Smoke-test output
```
num_classes (configs/data.py) = 116 | ignore_index = 255
[forward] logits (2, 116, 512, 512) dtype=torch.float32 finite=True
[class weights] SMOKE-ONLY batch-derived: present=2 min=1.000 median=1.000 max=4.016
[loss] CE=4.7600  Dice=0.9836  CE+Dice=5.7436  logitKD-KL(T=4, dummy teacher)=0.5005
[metric] all-class mIoU = 0.000000  (valid range [0,1]: True)
[ignore] non-255 pixels=345088  CM total=345088  -> 255 excluded: True
[metric] disease-only mIoU = 0.000000  ** PROVISIONAL / NEED_TO_CONFIRM (excludes index 0) **
[metric] per-image disease-only mIoU (PROVISIONAL) = [0.0, 0.0]
checks: all 8 PASS
```
_(mIoU = 0 is expected: the student is randomly initialized — `weights=None` — so argmax predictions are
near-random over 116 classes. The smoke only checks range validity, not segmentation quality.)_

## Loss values
- CE = **4.7600** · Dice = **0.9836** · CE+Dice = **5.7436** · logit-KD KL (T=4, dummy teacher) = **0.5005** — all finite.

## Metric values
- all-class mIoU = **0.000000** (valid) · disease-only mIoU = **0.000000** (PROVISIONAL) · per-image disease-only = **[0.0, 0.0]** (PROVISIONAL).

## `NEED_TO_CONFIRM` items
1. **`reduce_zero_label` / disease-only background convention** — empirically background = index 0, but not
   explicitly named in the dataset files; `reduce_zero_label` not used. Disease-only metric is PROVISIONAL.
2. **`λ_logit`** (Logit-KD term weight) — validation sweep over {0.25, 0.5, 1, 2, 4}, reported in Ch4.
3. **Real CE class weights** — computed over the FULL train set later; the batch-derived weights here are
   smoke-only and are **not** the thesis/train weights.
4. **CWD full feature loss + E3 wiring** — placeholder/scaffold only in B11.
