# Open Questions — every OPEN / NEED_TO_CONFIRM and how it gets resolved

Companion to [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) and [conflicts.md](conflicts.md).
No value here is guessed. Each item lists **how** it will be settled. Nothing below blocks Week-1
analysis/docs/smoke-test work; items flagged "before E1" are pre-training gates.

Source tags as in the contract.

---

## Empirically resolved (2026-06-27) — formerly OPEN conflicts

### 1. ✅ RESOLVED — `num_classes` = 116
- **Result:** `np.unique()` over **all 7,774** annotation PNGs → contiguous **0–115** (max non-ignore 115).
  Background = **0**, diseases = **1–115** (`mask = category_id + 1`, 99.85% consistent). Wei's "0–114" are
  the COCO `category_id`s. All-class classifier output = **116**.
- **Status:** **RESOLVED.** Evidence: [reports/dataset_report.md](../reports/dataset_report.md). Do **not**
  change masks; apply `ImageOps.exif_transpose` to images (never masks) before pairing/transform.

### 2. ◐ PARTIALLY RESOLVED (D1) — disease-only / background convention (`reduce_zero_label`)
> **D1 (2026-07-01) — metric policy:** **all-class mIoU is the checkpoint-selection metric and the headline
> E1/E-stage validation metric** (`src/training/train_e1.py` already selects the best-val checkpoint on
> all-class mIoU and logs disease-only mIoU as a **secondary/provisional** metric). Confirming the exact
> official disease-only exclusion convention is still useful for **Ch4** lesion-focused interpretation/
> reporting, **but it does NOT block E1 training, checkpointing, or all-class reporting.** A future switch to
> disease-only headline/checkpointing requires a separate explicit decision. This is **separate from — and
> does not change — the manuscript's per-image disease-only mIoU _primary inferential_ unit** for the Ch4
> hypothesis tests (see `IMPLEMENTATION_CONTRACT.md` §(f)).

- **Empirical finding:** background = **index 0** (in 7773/7774 masks, 80.6% of pixels); diseases = 1–115.
  So disease-only mIoU should **exclude index 0**.
- **Why still NEED_TO_CONFIRM:** no literal "background" category is named in the dataset files (the COCO
  `categories` list is **empty**), so the background label is *derived*, not declared. `reduce_zero_label`
  (keep background as class 0 vs MMSeg-style shift) is an implementation choice not fixed by any source.
- **Resolution:** confirm against the official PlantSeg repo / MMSeg config convention
  (`https://github.com/tqwei05/PlantSeg`) before finalizing disease-only reporting (Ch4) — this does **not**
  block E1 (D1); until then, exclude index 0 for disease-only metrics
  as the empirically-supported default. `[empirical; ch3; ctx]`
- **Minor data-quality note:** 12/7,774 masks (0.15%) have a disease value off by one from their
  `Metadata` `Index` (logged in the report) — does not affect the num_classes/background conclusion.

### D2 — E1 gradient clipping `max_norm` — ✅ RESOLVED (D-A: unclipped-E1 deviation, 2026-07-01)
**D2 / D-A RESOLVED:** E1 will proceed unclipped as an explicit documented deviation from Chapter 3's
value-less "global-norm, throughout" wording. The existing `src/training/train_e1.py` hook remains
available through `--grad-clip-norm` if divergence or instability is observed, but no numeric `max_norm` is
chosen by default. This resolves the pre-real-E1 decision gate without changing training code.
`configs/e1_student.py` records `grad_clip_max_norm=None`; `gradient_clipping="global_norm"` is the method
family, not a numeric value. `[ch3 §C; D2; D-A]`

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
