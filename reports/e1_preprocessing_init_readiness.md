# E1 Preprocessing / Augmentation / ImageNet-Init Readiness (B18b)

**Verdict: ◑ PARTIAL** — core **preprocessing is fully aligned** and **B18a class weights are ready**;
**augmentation is PARTIAL** (multi-scale RRC configured but not implemented) and **ImageNet init is PARTIAL**
(config requests it; code forces `weights=None`). **Nothing blocks the B19 *scaffold***; RRC + ImageNet
init **must** be resolved before the **real E1 training run**.

_Synthesis/readiness audit (read-only) from already-committed files. No implementation. Basis: contract B2/§(e)
(ch3-traced), `updated_ch3_sync_audit.md`, configs, `src/data/*`, `src/models/student.py`, B18a artifacts.
Generated 2026-06-28 · HEAD `4eb5644`._

---

## 1. Implemented AND aligned (ready)
**Core preprocessing — identical for train/val/test** (`src/data/transforms.py: core_preprocess` + `finalize`;
matches `configs/data.py`, contract §(e), and B16):
- ✅ **EXIF transpose — image only** (never the mask).
- ✅ **Keep-ratio resize**, long side → **512**.
- ✅ **Image interpolation: bilinear**; **mask interpolation: nearest**.
- ✅ **Center pad** to **512×512** (symmetric).
- ✅ **Image pad value `[124, 116, 104]`** (ImageNet mean, 8-bit).
- ✅ **Mask pad / ignore value `255`**.
- ✅ **ImageNet normalization** (mean `[.485,.456,.406]`, std `[.229,.224,.225]`), scale `/255`.
- ✅ **Raw mask convention preserved** — masks read as raw `L` indices 0–115 (no RGB/palette expansion, no
  label shift); `reduce_zero_label = False`.
- ✅ Output dtypes/layout: **float32 CHW** image, **int64 HW** mask.

**Partially-implemented train augmentation that IS aligned:**
- ✅ Horizontal flip p=0.5, vertical flip p=0.5, rotation ±10° p=0.5 (image fill = mean, mask fill = 255),
  image-only hue ±0.015 / saturation [0.8, 1.2] p=0.5 — match contract B2.
- ✅ Correct exclusions (no brightness/contrast/blur/noise/jpeg) — no eval-corruption leakage.

**B18a class weights — ready** (see §4 note).

### Preprocessing table (current vs ch3/contract/B16)
| Aspect | Implemented (`transforms.py`) | ch3 / contract / B16 | Match |
|---|---|---|---|
| EXIF | `exif_transpose` image only | image only, never mask | ✅ |
| Resize | keep-ratio long-side 512 | keep-ratio → 512 | ✅ |
| Image interp | bilinear | bilinear | ✅ |
| Mask interp | nearest | nearest | ✅ |
| Pad | center to 512² | symmetric pad to 512² | ✅ |
| Image pad value | `[124,116,104]` | `[124,116,104]` | ✅ |
| Mask pad/ignore | 255 | 255 (pad-only; absent from raw) | ✅ |
| Normalize | ImageNet mean/std, /255 | ImageNet mean/std, [0,1] | ✅ |
| Mask labels | raw 0–115, `reduce_zero_label=False` | 0=bg, 1–115 diseases, no shift | ✅ |
| dtype/layout | f32 CHW / i64 HW | f32 CHW / i64 HW | ✅ |

## 2. Configured but NOT implemented (gaps)
### Augmentation — config vs implementation
| Aug component | `configs/augment.py` | `src/data/transforms.augment` | Status |
|---|---|---|---|
| Horizontal flip 0.5 | yes | yes | ✅ implemented |
| Vertical flip 0.5 | yes | yes | ✅ implemented |
| Rotation ±10° p0.5 | yes | yes (fill mean / 255) | ✅ implemented |
| Photometric hue/sat | yes | yes (image-only) | ✅ implemented |
| **Multi-scale RRC** (size 512, scale **[0.75, 2.0]**, aspect-preserved) | yes | **no** | ❌ **not implemented** |
| **`cat_max_ratio` 0.95** (crop-rejection) | yes | **no** | ❌ **not implemented** |
| Excluded (brightness/contrast/blur/noise/jpeg) | excluded | not present | ✅ consistent |
| **Library = Albumentations** | declared | **numpy/PIL** (no Albumentations) | ⚠️ declared-not-used |

- **AUGMENTATION = PARTIAL:** flips / rotation / hue-saturation are implemented; **RRC / multi-scale /
  `cat_max_ratio` are configured but not implemented**; **Albumentations is declared in config but the
  implementation is numpy/PIL** (documented in `transforms.py` docstring + `updated_ch3_sync_audit.md` #6).

### ImageNet initialization — config vs implementation
| Item | State |
|---|---|
| `configs/e1_student.py` `init_weights` | **requests** `torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2` |
| `build_student(pretrained=…)` | `pretrained` arg is **plumbed but ignored** |
| `_build_backbone()` | **forces `weights=None`** (NOTE: "wire IMAGENET1K_V2 when available"); `used_pretrained=False` |
| Effect | student is **randomly initialized**; no download performed |

- **IMAGENET INIT = PARTIAL:** config requests ImageNet init; current `build_student()` effectively uses
  `weights=None`. **Fine for B17 / B19 scaffold** (wiring/dry-run), **not** for the real E1 run.

## 3. What blocks the B19 scaffold
**Nothing blocks B19** (the E1 training-loop *scaffold*: optimizer + poly schedule + val loop + mIoU +
best-val-checkpoint selection + logging). It can be built and dry-run on existing pieces:
- student (`build_student()`, weights=None) ✅, dataloader ✅, CE+Dice loss ✅, metrics (`src/eval/metrics.py`) ✅,
  class weights (B18a) ✅, SGD/poly settings (`configs/e1_student.py`) ✅.
- **B19 may proceed only if it keeps RRC and ImageNet-init as documented carry-forwards** (the loop must
  load the B18a weights and call `build_dataloader`; both RRC and ImageNet init can be slotted in later
  **without changing the loop API**).

## 4. What blocks the real E1 training run
**The real E1 run must NOT start until these are resolved:**
| Carry-forward | Why it blocks the real run | Type |
|---|---|---|
| **Multi-scale RRC / `cat_max_ratio`** | ch3 B2 training augmentation; missing → not ch3-faithful | local code (edit `transforms.py` + wire into `dataset.py` train path) |
| **ImageNet student init** | ch3/contract require `IMAGENET1K_V2`; random init ≠ E1 recipe | code (honor `pretrained` in `_build_backbone`) + **download-gated** |
| Compute / environment | 80,000 iters @ 512×512, batch 16 | GPU/RunPod (env-gated) |
| (Ready) real CE class weights | — | **DONE — B18a** (`reports/e1_class_weights.json`) |

**B18a class-weights readiness note:** `reports/e1_class_weights.json` exists (length-116, all finite,
median-normalized √inv-freq, no cap; bg=class 0 weighted normally) and is the **future CE weight artifact** —
the E1 loop loads it as `weight=torch.tensor(json["weights"])`. ✅ ready.

## 5. Separation summary
- **Implemented & aligned:** all core preprocessing; flips/rotation/hue-sat; B18a class weights.
- **Configured but not implemented:** multi-scale RRC + `cat_max_ratio`; Albumentations (impl is numpy/PIL).
- **Blocks B19 scaffold:** nothing (proceed with documented carry-forwards).
- **Blocks real E1 run:** RRC/multi-scale aug; ImageNet init; compute/env.

## 6. Recommended next exact step
**Proceed to B19 — E1 training-loop scaffold — with RRC and ImageNet-init as documented carry-forwards**
(it is unblocked and is the structural backbone; both gaps slot in later without changing the loop API).
The next *local code* carry-forward after the scaffold is **RRC** (edit `transforms.py`, wire into the train
path, re-verify via the dataloader smoke); **ImageNet init** is a small code change but **download-gated**.

```
B19 — E1 training-loop scaffold. Read-only first. Plan-gated. Dry-run only (no real training).

Goal: implement a minimal, correct E1 training-loop scaffold in src/training/ that wires:
build_student() + build_dataloader("train"/"val") + CombinedCEDiceLoss(weight=<B18a weights>) +
SGD(poly, from configs/e1_student.py) + periodic val mIoU (src/eval/metrics.py) + best-val-mIoU
checkpoint selection + logging. Carry-forwards kept explicit: RRC augmentation and ImageNet init
are NOT required for the scaffold (document them); weights=None and current aug are acceptable for a
short CPU dry-run.

Use Anaconda Python, PYTHONIOENCODING=utf-8, CPU. No real training run (cap the dry-run at a few
iters / tiny subset), no GPU, no downloads, no checkpoints written to the repo (or write only to a
gitignored/scratch path). Load B18a weights from reports/e1_class_weights.json.

Before writing/running anything, show the implementation plan and wait for `go`.

Approved new files only (proposed): src/training/train_e1.py (or engine module) + reports/e1_loop_dryrun.md.
Do NOT modify existing model/data/loss/config code; if an API needs changing, report it instead.

Rules: no real training, no GPU, no download/install, no RRC/ImageNet impl in this step, no edits to
transforms/model/config, no dataset/reference.pdf/context.md/teacher-init touch, no stage/commit, no
memory/~/.claude writes.

After: scaffold summary; dry-run result (loss/val mIoU over a few iters); what's stubbed/carry-forward;
files created/changed; git status; exact next prompt (RRC impl, then ImageNet init). Stop after B19.
```
(Alternative if you prefer to clear gaps first: do **B18c — RRC augmentation** before B19. Either order works.)

---

_Readiness audit only. No code/config/dataset changed; `docs/reference/reference.pdf` untouched (deferred);
nothing committed._
