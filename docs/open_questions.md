# Open Questions — every OPEN / NEED_TO_CONFIRM and how it gets resolved

Companion to [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) and [conflicts.md](conflicts.md).
No value here is guessed. Each item lists **how** it will be settled. Nothing below blocks Week-1
analysis/docs/smoke-test work; items flagged "before E1" are pre-training gates.

Source tags as in the contract.

---

## OPEN conflicts (two authoritative sources disagree — resolve empirically, no winner)

### 1. `num_classes` — 115 (Wei, indices 0–114) vs 116 (ch3, provisional bg+115)
- **Resolution:** run `np.unique()` over **all** real annotation PNGs (train+val+test) from the
  downloaded Zenodo dataset; the verified distinct label set (excluding 255) defines the classifier
  output size. `[ch3 Table 3.1; ctx]`
- **When:** mandatory **before E1** training. Do **not** hardcode 115 or 116 until then.
- **Status:** unresolved by design (dataset not downloaded in Week 1).

### 2. `reduce_zero_label` / background-index encoding
- **Question:** does a distinct background index exist (e.g., a separate value outside 0–114), or is
  background folded into the 0–114 range? This drives `reduce_zero_label` and the disease-only vs
  all-class metric split.
- **Resolution:** same `np.unique()` inspection as #1 — inspect whether index 0 is a disease class or
  background, and whether the label set is exactly {0..114} ∪ {255} or includes another value. `[ch3; ctx]`
- **When:** before E1.

---

## NEED_TO_CONFIRM (not stated in any source; filled by selection/measurement, never guessed)

### 3. `λ_logit` (Logit-KD weight)
- **Resolution:** pre-registered validation grid sweep over **{0.25, 0.5, 1, 2, 4}** at seed 42; select
  the candidate maximizing dataset-level **validation** mIoU (ties → smaller weight; boundary value
  reported as such). Selected value reported in Ch4 and **reused unchanged in E3**. `[ch3 §C "E2"]`
- **When:** during E2 (training phase — out of Week-1 scope).

### 4. E6-KD reduced distillation weights
- **Resolution:** only relevant **if** the contingency triggers (E3→E6 clean mIoU drop > 1.0 pp); weights
  set **below** their E3 values so soft targets do not override INT8 adaptation. Exact reduced values
  `NEED_TO_CONFIRM`. `[ch3 §C "E6"]`
- **When:** conditional, post-E6.

### 5. Library versions without pinned numbers
- **Albumentations:** "version pinned in the reproducibility manifest" but no number given → confirm from
  the actual `requirements.lock` / environment manifest once present. `[ch3 §D; ctx]`
- **statsmodels:** no version stated → confirm from the manifest. `[ch3 §D; ctx]`
- **Resolution:** read the pinned `requirements.lock` / Dockerfile when committed to the repo.

### 6. Additional seed values (multi-seed runs)
- **Resolution:** three-seed validation is planned for **E1 and E3**; the two seeds beyond **42** are not
  specified ("all seeds used are reported in the appendix") → confirm at run time. `[ch3 §F]`
- **When:** if/when multi-seed runs execute (compute-permitting; out of Week-1 scope).

### 7. RunPod hardware specifics
- **Resolution:** final pod type, GPU model, VRAM, CPU model + thread count, CUDA/container image, storage,
  and per-stage GPU-hours/cost are "reported in Chapter 4". `[ctx]` names RunPod **RTX 4090 (~$0.34/hr)** as
  the working assumption; treat exact values as `NEED_TO_CONFIRM` until logged. `[ch3 §D; ctx]`

### 8. Citation details (DOIs / venues / years) in reference.pdf
- **Resolution:** `[ctx]` warns two divergent citation lists exist historically — **never fabricate** a
  DOI/venue/author-year. Verify each against the primary PDF before use. Confirmed-good anchors:
  PlantSeg DOI **10.5281/zenodo.17719108**, Wei et al. *Scientific Data* (2026) 13:205,
  https://doi.org/10.1038/s41597-025-06513-4 `[Wei]`.

### 9. All experimental result numbers (teacher recovered mIoU; E1–E7 accuracy/efficiency/robustness)
- **Resolution:** produced by training/evaluation runs — **out of Week-1 scope** (analysis/setup/docs/smoke
  tests only). Teacher success criterion is recovery of 42.05% within ±1.5–2.0 pp. `[ch3]`

---

## Pre-training verification gates (mandatory before ANY E1 training) `[ch3 Table 3.1; ctx]`

These are not "open questions" so much as **blocking checks**; logged here for completeness. Store each
result as a JSON artifact next to the run.

1. **Mask class count** — `np.unique()` over all annotation PNGs (settles #1/#2 above).
2. **Image↔mask pairing** — shared filename/ID, both readable, identical dimensions; zero unmatched/
   unreadable/dimension-mismatched pairs.
3. **Split integrity** — zero identifier overlap across train/val/test (overlap count = 0).
4. **Mask value range** — every mask pixel ∈ (verified class set) ∪ {255}; no out-of-range values.
5. **Ignore-label unit tests** — CE / Dice / Logit-KD / CWD losses unchanged when padded (255) pixel
   values are altered; CWD channel-wise spatial softmax sums to 1 over valid locations only; teacher
   forward hook returns the stride-16 MSCAN-B Stage-3 feature (**320 channels**) for a synthetic input.
6. **QNNPACK operator support** — confirm **INT8 Sigmoid** in the LR-ASPP global-pool branch is supported
   (gate before E4); any unsupported op → FP32 fallback, reported.
