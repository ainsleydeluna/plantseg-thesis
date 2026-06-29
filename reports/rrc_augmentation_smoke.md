# RRC / Multi-Scale Augmentation Smoke (B18c)

**RESULT: ✅ PASS** — true train-time multi-scale random-resized-crop implemented and verified.
**This is the ACCEPTED implementation (RRC on the original-resolution image, before pad/normalize) — NOT
the rejected "RRC on the already-letterboxed 512² canvas" version.** No training was run.

_Generated: 2026-06-28 (read-only verification via scratch runner; Anaconda Python, CPU). HEAD `d6b0f1a`._

---

## What was implemented
**New active train path `train_preprocess()` (`src/data/transforms.py`)** — runs on the **original-resolution**
image (post-EXIF, **before** the final pad/normalize), in this order:
1. **EXIF-transpose image only** → 2. **RGB** convert →
3. **Multi-scale aspect-preserving resize**: `r ~ U(0.75, 2.0)`, long side → `round(512·r)` (bilinear image / **nearest** mask) →
4. **Rotation ±10° (p0.5) BEFORE crop** (image bilinear fill `[124,116,104]`, mask nearest fill `255`) →
5. **Random 512×512 crop + pad** with **`cat_max_ratio = 0.95`** (per-dim: crop if side ≥512, else random-place + pad image `[124,116,104]` / mask `255`) →
6. **Joint flips** (h/v, p0.5) →
7. **Image-only photometric** (hue ±0.015 / sat [0.8,1.2], p0.5) →
8. **`finalize`** (ImageNet normalize) — **unchanged**.

**Routing (`src/data/dataset.py`):** train → `train_preprocess`; **val/test → `core_preprocess` (unchanged)**.
**`augment()` retained** as a LEGACY 512²-canvas helper (flips/rotation/photometric; **no** RRC), now a thin
wrapper over shared primitives — kept for backward compatibility (only `dataset.py` imported it; it no longer
does). **Public dataloader API unchanged.** No new dependencies (numpy/PIL/torch).

### `cat_max_ratio` enforcement
For each candidate crop, accept iff the **dominant non-255 class fraction ≤ 0.95** (255 excluded; background
class 0 counts). Up to **10 attempts**; on exhaustion, **fall back** to the best (lowest-dominance) candidate.
This rejects mostly-background / single-class crops while always yielding a valid 512².

## Smoke results
**1. Train dataloader batch (RRC active):** image `(2,3,512,512)` float32 ✅ · mask `(2,512,512)` int64 ✅ ·
labels `{0, 87, 109, 255}` ⊆ `{0..115,255}` ✅.

**2. Val dataloader batch (core_preprocess):** image `(2,3,512,512)` float32 ✅ · mask `(2,512,512)` int64 ✅ ·
labels `{0,1,255}` ✅ · **deterministic across two passes = True** → val/test preprocessing **unchanged**.

**3. Per-sample RRC over 40 train samples:**
| Metric | Value |
|---|---|
| Label out-of-range violations (`out ⊄ {0..115,255}`) | **0** |
| **New-label violations** (`out ⊄ raw ∪ {255}`) | **0** → nearest introduced **no new/fractional labels** |
| `cat_max_ratio` **pass** | **34 / 40** (accepted on attempt 1) |
| `cat_max_ratio` **fallback** | **6 / 40** (hit 10 attempts → best crop kept) |
| Attempts histogram | `{1: 34, 10: 6}` |
| Scale `r` | min 0.772 · mean 1.316 · max 1.986 |
| Dominant non-255 ratio | min 0.526 · mean 0.833 · max 0.979 |

Examples: `idx0 r=1.218 fallback dom=0.951` · `idx3 r=1.986 pass dom=0.660` · `idx5 r=0.892 fallback dom=0.979`.
(The 6 fallbacks are images dominated by background/one class at most scales — fallback keeps the
least-dominant crop; working as designed.)

## Label safety (Q-required)
- Output masks are **int64**, shape `(512,512)`; every value ∈ `{0..115, 255}`.
- For all 40 samples, `unique(out_mask) ⊆ unique(in_mask) ∪ {255}` — **nearest mask interpolation + 255 pad
  introduced no new or fractional labels**. `reduce_zero_label = False`; no label remapping.

## Accepted (this) vs rejected (512-canvas) implementation
| | **Accepted — B18c** | Rejected — earlier 512-canvas plan |
|---|---|---|
| Operates on | **original-resolution** image (post-EXIF, pre-pad) | already resized+letterboxed **512² canvas** |
| Uses original detail | **yes** (resize from source) | no (capped at long-side-512) |
| Crops padding? | only pads when a scaled crop dim < 512 | could crop the letterbox **255/mean padding** |
| `cat_max_ratio` over | real content | content + letterbox borders |
| Verdict | **implemented** | rejected (methodologically weaker) |

## Invariants preserved
val/test deterministic & unchanged ✅ · public dataloader API unchanged ✅ · output `[3,512,512]` f32 /
`[512,512]` i64 ✅ · bilinear image / nearest mask ✅ · image pad `[124,116,104]` / mask pad `255` ✅ ·
labels `{0..115,255}` ✅ · no label remap, `reduce_zero_label=False` ✅.

## Files changed / created
- `src/data/transforms.py` — added `train_preprocess` + RRC helpers (`_resize_long_side`,
  `_dom_nonignore_ratio`, `_random_crop_pad_512`) + shared `_apply_flips/_apply_rotation/_apply_photometric`;
  `augment()` refactored into a thin legacy wrapper; docstring updated. `core_preprocess`/`finalize` unchanged.
- `src/data/dataset.py` — `__getitem__` routes train → `train_preprocess`, val/test → `core_preprocess`;
  import updated. Public API unchanged.
- `reports/rrc_augmentation_smoke.md` — this report.

## Post-RRC train-step regression (B17 re-run)
Re-ran the committed B17 train-step smoke **after** the B18c RRC changes to confirm no regression (the
train batch now flows through `train_preprocess` / RRC).

- **Command:** `python scripts/smoke_train_step.py` (Anaconda Python, torch 2.9.1+cpu, CPU, `PYTHONIOENCODING=utf-8`).
- **RESULT: ✅ PASS** (all 6 gating checks).
- **Shapes:** image `(2,3,512,512)` float32 · mask `(2,512,512)` int64 · logits `(2,116,512,512)` float32, finite.
- **Loss finite:** CE 5.1293 · Dice 0.9820 · CE+Dice **6.1112** (finite scalar). _(Values shifted slightly
  vs the pre-RRC run — expected, since the train batch now passes through multi-scale RRC; shapes and all
  gating checks are unchanged.)_
- **Gradients finite:** 176/176 trainable param-tensors received grads · 0 None · 0 non-finite.
- **Optimizer step changed ≥1 parameter:** yes (`features.0.0.weight` changed after `step()`).
- **Loss trajectory (sanity, non-gating):** `[6.1112, 5.8383, 5.3888] -> 4.7847` (decreasing).
- This is still a **wiring smoke, not training**.

_Not training. `docs/reference/reference.pdf` untouched (deferred). Nothing committed._
