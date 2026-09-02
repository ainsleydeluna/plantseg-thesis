# Implementation Contract — Resource-Constrained Plant Lesion Segmentation (KD + INT8 QAT)

> **Status:** Week-1 analysis artifact. No training, no model code, no dataset download.
> This document records the *locked* facts and configs extracted from the authoritative
> sources, with every value traced to a source file. Where a value is not stated in any
> pointed-to source, it is recorded as the literal string `NEED_TO_CONFIRM` (never guessed).
> Where two authoritative sources disagree and the disagreement is not yet resolvable, it is
> recorded as **OPEN** and cross-listed in [conflicts.md](conflicts.md) / [open_questions.md](open_questions.md).

## Source legend & hierarchy

| Tag | File | Authority |
|---|---|---|
| `[ch3]` | `docs/reference/ch3.pdf` | **Authoritative** — single source of truth for method/protocol. When ch3 conflicts with anything, ch3 wins. |
| `[ctx]` | `docs/reference/context.md` | **Authoritative** — reconciliation / source-of-truth `.md` (the "CLAUDE.md"). |
| `[Wei]` | `docs/reference/Wei (2026) - PlantSeg dataset paper (2).pdf` | **Authoritative for dataset facts only.** |
| `[ch2]` | `docs/reference/ch2.pdf` | Secondary (background/rationale). |
| `[ch1]` | `docs/reference/ch1.pdf` | Secondary (intro/hypotheses). |
| `[ref]` | `docs/reference/reference.pdf` | Secondary citation list (treat as needs-verification; never fabricate DOIs). |

Primary-source papers in `docs/reference/` (used to justify individual mechanisms, not as config
authorities): Howard 2019 (MobileNetV3), Shu 2021 (CWD), Hinton "Distilling…" (logit KD),
Jacob 2018 + Krishnamoorthi 2018 (quantization), Guo 2022 (SegNeXt), Hendrycks 2019 + Kamann 2020
(robustness).

---

## (a) Project summary

A **controlled computer-vision evaluation** (not a deployment study) that compresses a
**SegNeXt-B / MSCAN-B teacher** into a **mobile-deployable MobileNetV3-Large + LR-ASPP student**
via knowledge distillation (Logit KD + Channel-Wise KD) and **INT8 quantization** (QAT + PTQ),
evaluated on the **PlantSeg** in-the-wild plant-disease segmentation benchmark `[ch3; ctx]`.

- Goal: a more favorable **accuracy ⊕ efficiency ⊕ robustness** trade-off than the lightweight
  student baseline — judged by balance across all three dimensions, not peak on any single one `[ch3 §A]`.
- Scope guardrails: **no on-device / ARM claims**; CPU-proxy efficiency only. Single-seed vertical
  slice (E1→E7 + full eval) before any multi-seed scaling `[ch3 §D; ctx]`.
- The teacher is a **descriptive upper-bound reference only** — never an inferential comparator `[ch3 §A]`.
- Published Wei et al. (2026) numbers are **contextual reference only**, never inferential comparators
  (no per-image scores available; preprocessing/training/eval mismatched) `[ch3 §A]`.

---

## (b) Pipeline + the seven experiments (E1–E7) + Teacher

`[ch3 Table 3.2; ctx]`

| Stage | Starts from | Adds / does | Main comparison |
|---|---|---|---|
| **Teacher** | ADE20K-pretrained SegNeXt-B / MSCAN-B (MMSeg zoo) | Fine-tune on PlantSeg → recover ~42.05% mIoU ref. **Not deployed.** | Teacher vs E6 (descriptive only) |
| **E1** | ImageNet-pretrained quantizable MobileNetV3-L | FP32 baseline — no KD, no quant (anchor baseline) | E1 vs E2; E1 vs E3; E1 vs E6 |
| **E2** | E1 init (trained independently) | + **Logit KD** | E1 vs E2 |
| **E3** | E1 init (trained independently) | + **Logit KD + CWD** (**proposed FP32 student**) | E2 vs E3 |
| **E4** | E1 FP32 checkpoint | **INT8 PTQ** | E4 vs E5; E4 vs E7 |
| **E5** | E1 FP32 checkpoint | **INT8 QAT** | E4 vs E5; E5 vs E6 |
| **E6** | E3 checkpoint (**CWD head removed**) | **INT8 QAT — full proposed pipeline** | E3 vs E6; E7 vs E6; E1 vs E6; Teacher vs E6 |
| **E7** | E3 checkpoint (**CWD head removed**) | **INT8 PTQ** | E7 vs E6; E4 vs E7 |

- E4–E7 form a **2×2 quantization × distillation matrix** (PTQ vs QAT) × (undistilled E1 vs distilled E3) `[ch3 §A,C]`.
- **E6-KD** is a *pre-registered contingency arm*, not a locked stage: run only if the E3→E6
  clean-test mIoU drop exceeds **1.0 pp**; reinstates frozen teacher with **reduced** distillation
  weights during QAT `[ch3 §C, §F]`.
- All stages share the **same official 70/10/20 split, preprocessing, 512×512 resolution, normalization,
  mask formatting, metrics, and corruption settings**; only the compression component varies `[ch3 §C]`.
- All training runs are **iteration-matched (80,000 iters)**, not wall-clock- or FLOPs-matched `[ch3 §C]`.

---

## (c) Dataset facts — PlantSeg (Wei et al., 2026)

| Fact | Value | Source |
|---|---|---|
| Total images | **7,774** diseased images (with masks) | `[Wei]` |
| Disease classes | **115** unique plant-disease combinations | `[Wei]` |
| Plant hosts | **34** | `[Wei]` |
| Distinct disease types | **69** (combine with 34 hosts → 115 combos) | `[Wei]` |
| Split ratio | **70 / 10 / 20** (train / val / test) | `[Wei]` |
| Split sizes (**empirical, verified**) | **train 5,367 / val 846 / test 1,561** (total 7,774); folder + JSON + CSV encodings all agree, **zero overlap** | `[empirical]` |
| Split files | `annotation_train.json`, `annotation_val.json`, `annotation_test.json` (COCO format) + `train/val/test` subfolders + `Metadata.csv` `Split` column | `[ch3 §E; empirical]` |
| Test images used everywhere | **1,561** (per-image metric unit count; verified) | `[empirical]` |
| Image format | **JPEG** (`images/`) | `[Wei]` |
| Mask format | **PNG, single-channel indexed** (mode `L`, 7774/7774; no color masks) | `[Wei; empirical]` |
| Mask pixel values | **0–115** (contiguous): **0 = background**, **1–115 = the 115 diseases** (`mask value = COCO category_id + 1`, 99.85% per-image consistent). Wei's "0–114" describe the `category_id`s, not the mask pixel values. | `[Wei; empirical]` |
| EXIF orientation (images) | **`ImageOps.exif_transpose` is MANDATORY** on images before pairing/transforming with masks; **never** on masks (PNG, no EXIF). 9 images affected (8× orient-6 dim-swap + 1× orient-3 180°). | `[empirical]` |
| Ignore index | **255** (padding regions; excluded from all loss & metrics). **Absent from raw masks** — introduced only at preprocessing. | `[ch3; empirical]` |
| `num_classes` (all-class) | **116** — empirically supported (mask values 0–115; max non-ignore = 115; background 0 + 115 diseases). | `[empirical]` |
| `reduce_zero_label` / background | **background = index 0** (empirical: present in 7773/7774 masks, 80.6% of pixels). `reduce_zero_label` itself = `NEED_TO_CONFIRM` (implementation choice; no literal "background" category named in dataset files). | `[empirical]` |
| Zenodo DOI | **10.5281/zenodo.17719108** | `[Wei; ch3]` |
| License (dataset) | **CC BY-NC 4.0** | `[Wei; ch3]` |
| License (the article itself) | CC BY-NC-**ND** 4.0 (Nature) — distinct from dataset license; do not conflate | `[Wei]` |
| Imbalance note | mask ratios of ~80% of images are below 36% (small-lesion regime) | `[Wei; ch3]` |

> `[empirical]` = verified by `scripts/verify_plantseg_dataset.py` → [reports/dataset_report.md](../reports/dataset_report.md) (full-dataset, read-only, 2026-06-27).

**Benchmark reference numbers (contextual only, `[Wei]` Table 3):**
SegNeXt-B/MSCAN-B = 28M params, **42.05%** mIoU / 56.30% mAcc · MobileNetV3 = 5M, 10.22% / 17.15% ·
SegNeXt-L/MSCAN-L = 49M, 44.52% mIoU · ConvNeXt-L = 121M, 46.24% / 59.97% (highest).

### ✅ RESOLVED (empirically, 2026-06-27) — `num_classes` = 116
`np.unique()` over **all 7,774** real annotation PNGs returns contiguous values **0–115** (116 distinct;
max non-ignore = **115**), so the earlier 115-vs-116 conflict is **resolved toward 116** for all-class
segmentation. The two source claims were a **frame mismatch**, not a true contradiction:
- `[Wei]`'s "indices **0–114**" describe the COCO **`category_id`s** (verified: `category_id` ∈ 0–114).
- The **rasterized mask pixel values** are **0–115**: **0 = background**, **1–115 = the 115 diseases**,
  i.e. `mask value = category_id + 1` (verified **99.85%** per-image consistent, 7762/7774; 12 minor
  off-by-one exceptions logged in the report). `[ch3]`'s `num_classes = 116` matches this mask space.

**Locked outcomes:** classifier output channels = **116**; **do not change masks**; apply
`ImageOps.exif_transpose` to images (not masks) before any pairing/resizing.
**Residual `NEED_TO_CONFIRM`:** the *disease-only* metric must exclude background = **index 0**
(empirically), but no literal "background" category is named in the dataset files (COCO `categories`
list is empty), so `reduce_zero_label` and the formal disease-only exclusion stay `NEED_TO_CONFIRM`
pending the PlantSeg repo's official convention. `[empirical; ch3 Table 3.1; ctx]`

---

## (d) Locked configs (every value traced)

### B1 — Teacher fine-tune `[ch3 §C, Table 3.2/3.5]`
| Param | Value | Source |
|---|---|---|
| Base / init | SegNeXt-B / MSCAN-B, ADE20K-pretrained; MMSeg zoo ckpt `segnext_mscan-b_512x512_160k_ade20k` (or equivalent) | `[ch3 §D]` |
| Framework | MMSegmentation 1.2.2 + mmcv 2.1.0 | `[ch3 §D; ctx]` |
| Optimizer | AdamW | `[ch3]` |
| Learning rate | **6e-5** | `[ch3]` |
| Weight decay | **0.01** | `[ch3]` |
| Betas | **(0.9, 0.999)** | `[ch3]` |
| Decode-head LR multiplier | **lr_mult = 10** | `[ch3]` |
| LR schedule | poly | `[ch3]` |
| Iterations | **40,000** | `[ch3]` |
| Batch size | **16** | `[ch3]` |
| Crop | **512×512** | `[ch3]` |
| Loss | cross-entropy | `[ch3]` |
| Success criterion | recover 42.05% mIoU within **±1.5–2.0 pp** (protocol match, not max accuracy) | `[ch3]` |

### B2 — Student training (E1 / E2 / E3 shared recipe) `[ch3 §C "E1"]`
| Param | Value | Source |
|---|---|---|
| Init | ImageNet-pretrained quantizable MobileNetV3-Large; TorchVision `MobileNet_V3_Large_Weights.IMAGENET1K_V2` (top-1 75.27%) | `[ch3 §D, Table 3.5]` |
| Optimizer | SGD, momentum **0.9** | `[ch3]` |
| Learning rate | **1e-2** | `[ch3]` |
| LR schedule | polynomial decay, **power 0.9** | `[ch3]` |
| Weight decay | **1e-4** | `[ch3]` |
| Batch size | **16** | `[ch3]` |
| Iterations | **80,000** @ 512×512 | `[ch3]` |
| Validation interval | every **4,000** iters | `[ch3]` |
| Checkpoint selection | **best validation-mIoU — all-class val mIoU (D1)** | `[ch3; D1]` |
| Distillation-weight ramp (E2/E3) | linear **0 → target over first epoch** | `[ch3]` |
| Gradient clipping | global-norm, throughout | `[ch3]` |
| Teacher in loop (E2/E3) | eval mode, online, consumes **identical augmented input** as student | `[ch3]` |
| Optional control | extended-schedule E2 (~160,000 iters, 1 seed) — optional, future work if not run | `[ch3 §C]` |

> **[D-A/D2 RESOLVED — unclipped-E1 deviation]** Chapter 3 specifies global-norm clipping "throughout"
> (the row above) but does not specify a numeric `max_norm`. To avoid inventing an unsupported value, E1
> proceeds with `grad_clip_max_norm=None` as an **explicit documented deviation**. The
> `src/training/train_e1.py` hook remains config/CLI-available through `--grad-clip-norm` if divergence is
> observed. This does **not** alter the `[ch3]`-traced method row; it records the accepted implementation
> deviation for E1. See `docs/open_questions.md` D2. `[D2; D-A]`

### B2 — Augmentation (train split only) `[ch3 §E.2.c]`
| Component | Value | Source |
|---|---|---|
| Horizontal flip | p = **0.5** | `[ch3]` |
| Vertical flip | p = **0.5** | `[ch3]` |
| Rotation | **±10°**, p = 0.5, applied **before crop**; image fill = ImageNet mean, mask fill = **255** | `[ch3]` |
| Multi-scale random resized crop | final **512×512**, scale range **[0.75, 2.0]**, aspect-ratio preserved | `[ch3]` |
| Crop-rejection threshold | `cat_max_ratio` = **0.95** | `[ch3]` |
| Mask interpolation (in crop) | nearest-neighbor | `[ch3]` |
| Photometric (image-only) | hue **±0.015**, saturation factor **[0.8, 1.2]**, p = 0.5 | `[ch3]` |
| Brightness / contrast | **excluded** (diagnostic color + overlap with brightness/fog corruptions) | `[ch3]` |
| Blur / noise / JPEG aug | **excluded** (data-contamination prohibition — overlap with eval corruptions) | `[ch3]` |
| Library | Albumentations (joint image+mask) | `[ch3 §D]` |
| Class weighting | class-aware √ inverse-frequency (see B5) | `[ch3]` |

### B3 — Distillation
**Logit KD (E2)** `[ch3 §C "E2"]`
| Param | Value | Source |
|---|---|---|
| Loss | CE + KL on temperature-softened outputs | `[ch3]` |
| Temperature `T_Logit` | **4** | `[ch3]` |
| Weight `λ_logit` | **`NEED_TO_CONFIRM`** — selected via validation sweep over **{0.25, 0.5, 1, 2, 4}** at seed 42; reported in Ch4; reused **unchanged** in E3 | `[ch3]` |
| KL averaging | over valid (non-255) pixels only | `[ch3]` |

**CWD (E3, Shu 2021)** `[ch3 §C "E3"]`
| Param | Value | Source |
|---|---|---|
| Temperature `T_CWD` | **4** | `[ch3]` |
| Feature-map weight `α_CWD` | **50** (stride-16 C5 map) | `[ch3]` |
| Logit-map weight `β_CWD` | **3** | `[ch3]` |
| Normalization | **T²/C**, with **C = 320** (MSCAN-B stride-16 Stage-3 channel count) | `[ch3]` |
| Projection head | training-only 1×1 conv: student **160-ch C5 → teacher 320-ch**; removed before E6/E7 via state_dict edit prior to observer insertion | `[ch3]` |
| Ignore handling | validity mask downsampled to stride-16; channel-wise spatial softmax + KL restricted to valid locations | `[ch3]` |

#### B3 — spatial grid of each distillation term `[project; B32/F8, 2026-09-01]`
ch3 pins *which pixels* enter the Logit-KD KL ("averaged over valid pixels only") but **not which
grid**. B32 pins the grid. This is an implementation resolution of a genuine ch3 gap, not a change
to a `[ch3]`-traced value.

| Term | Grid | Validity mask | Note |
|---|---|---|---|
| Logit-KD KL | **OS8 64×64** | 64×64, min-pooled | **changed by B32** — was an upsampled 512×512 |
| `L_CWD_logit` | **OS8 64×64** | 64×64, **shared** with Logit-KD | already OS8 pre-B32; unchanged |
| `L_CWD_feat` | stride-16 32×32 | 32×32, min-pooled | unchanged |
| `L_CE`, `L_Dice` | full 512×512 | full-resolution target | **unchanged — the supervised path is not touched** |

Both the student head and the SegNeXt LightHamHead emit logits natively at 64×64 for a 512×512
input, so neither side is resampled. The old path upsampled both, putting **98.4%** of the KL on
interpolants; because interpolation happens in *logit* space before the softmax
(`softmax(interp(z)) ≠ interp(softmax(z))`), the soft targets were distorted rather than repeated.

- **Valid-population cost [empirical, measured, 60 train samples]:** the Logit-KD valid population
  is a strict subset of CE's and is **1.29% smaller in relative terms** (mean valid fraction 0.7096
  at 64×64 vs 0.7189 at 512×512). `L_CWD_feat` at 32×32 loses 3.14%. Conservative all-valid
  min-pooling penalises letterboxed images ~3× more than near-square ones in relative terms (1.94%
  vs 0.62% of valid locations), but the absolute effect is under 2% everywhere; the large absolute
  gap between those groups is **padding**, which CE sees identically.
- **Zero-valid cells are unreachable by construction**, not merely unobserved: 0 of 60 samples, and
  it would require aspect ratio > 64 (widest observed 2.42). No guard was added.
- **λ_logit magnitude shift [empirical, measured — UPPER BOUND]:** the KD term's magnitude changed
  by **1.95×** and its gradient contribution by **1.76×**. Measured against a *random synthetic*
  teacher, the worst case for spatial smoothness, so this is an **upper bound** — a real trained
  SegNeXt-B should shift it less. `CANNOT-VERIFY-LOCALLY`; re-measure on the pod.
- **Consequence for the λ sweep:** the preregistered grid `{0.25, 0.5, 1, 2, 4}` is geometric with
  ratio 2, so ~1.95× is about **one grid step**. The grid still brackets a sensible optimum and
  needs **no re-centring** — F11 is intact — but the expected optimum sits roughly one step lower,
  and **a λ selected under the old semantics is not transferable**. Ch4 must state that the sweep
  ran under OS8 semantics.
- **Machine-checkable guard `[project; B32c-2]`:** `configs/distill.py` defines
  `LOGIT_KD_SEMANTICS = "logitkd@os8-64x64-of-512"`. Every E2/E3 checkpoint and a
  `<stage>_run_meta.jsonl` record it. `--lambda-semantics` is optional but checked when supplied;
  a mismatch refuses unless `--allow-semantics-mismatch` is passed, and that override is stamped
  into both artifacts so a non-comparable run stays identifiable without the terminal.

#### E2/E3 peak memory `[project; B32c-3, INFERRED from a MEASURED activation set]`
Retained-activation bytes MEASURED via `saved_tensors_hooks` at batch 2 and scaled ×8 (the graph is
fixed-shape, so retained bytes are exactly linear in batch); parameters/gradients/momentum exact.

| | E2 | E3 |
|---|---|---|
| activations + static, **pre-B32** | 13.63 GB | 13.78 GB |
| activations + static, **post-B32** | **10.04 GB** | **10.20 GB** |

The **frozen teacher retains nothing** — `FrozenTeacher.forward` is `@torch.no_grad()` — so its cost
is a transient working set bounded at **~0.3–0.55 GB**, not an activation graph. The dominant costs
are the student's own retained activations (4.47 GB) and the CE+Dice term (5.62 GB), both at full
512×512, both unchanged by B32 and inherent to the locked batch-16/512² recipe.

**E2/E3 therefore cost only marginally more than E1 itself.** Adding allocator overhead (+5–15%),
the teacher transient, cuDNN workspace and backward temporaries gives an **INFERRED 11–14 GB** peak
on CUDA at batch 16. This is **not a GO** — settle it on the pod with
`torch.cuda.max_memory_allocated()`.
| E3 total loss | `L_CE + L_Dice + λ_logit·L_LogitKD + 50·L_CWD_feat + 3·L_CWD_logit` | `[ch3]` |
| Optional control | α_CWD sensitivity sweep {25, 50, 100} at 1 seed — optional, else fixed 50 per Shu 2021 | `[ch3 §C]` |

### B4 — Quantization
**INT8 QAT (E5 / E6)** `[ch3 §C "E5"/"E6", §D]`
| Param | Value | Source |
|---|---|---|
| Backend / toolchain | **QNNPACK**, eager-mode `torch.ao.quantization` | `[ch3]` |
| Optimizer | SGD, momentum **0.9** | `[ch3]` |
| Learning rate | **3e-4** (3×10⁻⁴), **cosine** decay | `[ch3]` |
| Epochs | **~15**, early-stop on val mIoU | `[ch3]` |
| Activation quant start | from **step 0**, moving-average range observers | `[ch3]` |
| BN-stat freeze | after **~65–70%** of training | `[ch3]` |
| Observer freeze | shortly after BN freeze | `[ch3]` |
| Gradient clipping | global-norm | `[ch3]` |
| Weight EMA | **none** (instantaneous weights quantize better) | `[ch3]` |
| Checkpoint selection | best val mIoU | `[ch3]` |
| Distillation during QAT | **none** (E5/E6 supervised-only) | `[ch3]` |
| Weight quant | **per-channel symmetric INT8**, all conv layers | `[ch3]` |
| Activation quant | **per-tensor asymmetric UINT8** (full 8-bit range for QNNPACK/ARM) | `[ch3]` |
| Fusion | Conv-BN-ReLU via `fuse_modules` before observer insertion | `[ch3]` |
| Hard-Swish / Hardsigmoid | quantized as **standalone** ops | `[ch3]` |
| Source ckpt | E5 ← E1; **E6 ← E3 (CWD head removed)**; E6 uses identical config as E5 | `[ch3]` |
| E6-KD reduced weights | `NEED_TO_CONFIRM` (reduced relative to E3 values; only if drop > 1.0 pp) | `[ch3]` |

**INT8 PTQ (E4 / E7)** `[ch3 §C "E4"/"E7", §E.1]`
| Param | Value | Source |
|---|---|---|
| Method | static INT8 (standard) | `[ch3]` |
| Calibration set | **~128 images**, sampled with **seed 42** from training partition; no augmentation; same preprocessing as clean test; identifiers persisted as fixed list; **same subset for E4 and E7** | `[ch3]` |
| Activation observer | histogram (minimizes quantization error) | `[ch3]` |
| Weight observer | per-channel min/max | `[ch3]` |
| Weight quant | per-channel symmetric INT8, all conv (per-channel depthwise essential) | `[ch3]` |
| Activation quant | per-tensor asymmetric UINT8 | `[ch3]` |
| Source ckpt | E4 ← E1; **E7 ← E3 (CWD head removed)** | `[ch3]` |
| Calibration selection | configuration selected on **validation**, PTQ results reported on **test** | `[ch3]` |

### B5 — Loss `[ch3 §C "E1", §C.1]`
- **Supervised:** `L_sup = L_CE + L_Dice` (equal weight).
- **Cross-entropy:** class-weighted with **√ inverse-frequency** weights; weight =
  `sqrt( total_train_pixels / (num_classes × pixels_in_class) )`; computed over **non-255** pixels only;
  **normalized by the median weight**; report realized min/median/max (no fixed cap); `ignore_index = 255` (native).
- **Dice:** soft, on predicted probabilities; per-class macro over classes **present in the batch**
  (absent classes excluded); smoothing **1e-5** added to numerator & denominator; **unweighted**;
  validity mask applied before intersection/union.
- **ignore_index = 255** excluded from CE, Dice, Logit KD, CWD, and **all** metrics `[ch3; ctx]`.

### B6 — Seed + library versions `[ch3 §D, Table 3.4; ctx]`
- **Seed = 42** across `torch`, `numpy`, python `random` `[ch3]`.
- Determinism set **before CUDA init**: `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` `[ch3; ctx]`.
- Multi-seed plan: three-seed validation planned for **E1 and E3** (the two extra seed values =
  `NEED_TO_CONFIRM`); E4/E7 recomputed per seed; E5/E6 fine-tuned per seed where compute permits;
  Teacher trained once; E2 repetition optional `[ch3 §F]`.
- **`num_workers` is a REPRODUCIBILITY-RELEVANT parameter and must be reported alongside seed 42**
  `[project; empirical, measured 2026-09-01 — B31-5/A1]`. Seed 42 alone does **not** determine the
  realized augmentation sequence. Measured on a fixed 8-sample / 4-batch train subset
  (`shuffle=False`, identical seed), three distinct augmentation streams were observed:

  | `num_workers` | first two sample digests | stream |
  |---|---|---|
  | 0 | `140c8c720c8bc712`, `a26114baf0d446c8` | A |
  | 2 | `6c772db174314e67`, `f4109cb73ece4509` | B |
  | 4, 8 | `6c772db174314e67`, `f4109cb73ece4509` | B (first 2 batches) → C (later batches) |

  Mechanism: at `num_workers=0` the per-sample augmentation seed is drawn from the main-process
  NumPy stream; at `num_workers>0` each worker is seeded from `torch.initial_seed()` and batch *b*
  is served by worker *b* mod `num_workers`, so the seed a given sample receives depends on the
  worker count. The `{4, 8}` agreement above is a **probe artifact** — with `num_workers >= n_batches`
  every batch draws a distinct fresh worker. That condition never holds at real scale
  (**335 batches/epoch vs 12 workers**), so in the real run each `num_workers` value yields a
  distinct realized sequence.
- **Consequence:** the B31-5 change of the real-run default from `4` to `min(cpu_count-2, 12)`
  changes the realized augmentation *sequence*. It does **not** change the augmentation
  *distribution*, the recipe, or any locked hyperparameter, and cross-process reproducibility at a
  fixed `num_workers` is preserved (measured byte-identical across two independent processes).
  Runs are comparable in distribution; they are not bitwise-comparable across different
  `num_workers`. Record the value in Ch4 with the seed.
- **Manuscript gap (for correction outside this repo):** ch3 §D's reproducibility paragraph pins
  seed 42, the cuDNN flags, `use_deterministic_algorithms(True, warn_only=True)` and
  `CUBLAS_WORKSPACE_CONFIG`, but is silent on `num_workers`. As written it is **under-specified**:
  two runs satisfying every stated condition can still differ. `[project]`
- **Resume is NOT bitwise-identical to an uninterrupted run** — documented deviation, same register
  as D-A `[project; B31-2]`. `--resume` restores model / optimizer / scheduler / RNG state
  (`torch`, `torch.cuda`, `numpy`, python `random`, and the DataLoader generator) and the LR curve
  continues **exactly** (checkpoint-recorded LR matches the analytic 80,000-iteration
  `PolynomialLR` curve at full float64 precision). Data **order** is not recoverable: the loop
  consumes an infinite `cycle(train_loader)` and the position within the current epoch is not
  persisted. A resumed run is therefore a valid E1 run but not a byte-reproduction of an
  uninterrupted one, and must be reported as resumed if used for a headline result.
- **LR-monotonicity coverage across resume** `[project; B31c V1]`: `last.pt` carries `prev_lr`, so a
  *k*-segment run leaves **zero** unverified LR transitions (measured: a 3-segment 9-iteration run
  compared 2+3+3 = 8 of 8 transitions). When **no** transition is compared (a one-iteration fresh
  run) the check reports `SKIPPED`, never a vacuous `PASS`. The scaffold's final line carries
  `RESULT: <PASS|FAIL> (n/6 checks exercised, m skipped)`, and a resume with nothing left to do
  prints the distinct token `RESULT: NOOP`.

#### E1 runtime defaults introduced by B31 `[project; authorised 2026-09-01]`
These are **operational** parameters. None of them touches a `[ch3]`-traced method value.

| Parameter | Value | Note |
|---|---|---|
| `--ckpt-interval` | **2000** | periodic atomic `last.pt` resume point |
| `--keep-ckpts` | **3** | rolling best-checkpoint retention (best + `last.pt` always kept) |
| `prefetch_factor` | **4** | train and val, only when `num_workers > 0` |
| `persistent_workers` | **True, TRAIN only** | val respawns per validation (~20 times); avoids a second resident worker pool |
| `pin_memory` | gated on `torch.cuda.is_available()` | uniform across train/val |
| `num_workers` (real run) | **`min(cpu_count-2, 12)`** | reproducibility-relevant — see above |
| `drop_last` | **True, TRAIN only** | val/test keep every sample; dropping eval samples would corrupt the metric |
| telemetry | `e1_telemetry.jsonl` in `--ckpt-dir` | append-mode, resume-safe, never inside the repo |

- **Iterations per epoch = 335** under `drop_last=True` (5,367 train / batch 16; 7 samples dropped
  per epoch and reshuffled into the next), giving **238.806 epochs** at the locked 80,000 iterations
  `[empirical, measured]`.
- **`_dom_nonignore_ratio` uses `np.bincount`, not `np.unique`** `[project; B31-6]`. This is an
  **exact-equivalence optimisation, not a behaviour change**: verified bit-identical on 216 masks
  including all-ignore, single-class, all-background, empty, and in-domain labels above
  `num_classes`. It is **2.2x faster in-function** (0.838 → 0.374 ms on a realistic 512² mask) but
  only **1.03x mean / 1.12x median end-to-end** on `train_preprocess`, because the crop stage is one
  of seven. It is **not a throughput lever** — the throughput lever is `num_workers`. The new code
  is additionally *stricter* than the old: a label > 255 now raises instead of being silently
  counted.

| Library | Version | Source |
|---|---|---|
| Python | 3.11 | `[ch3; ctx]` |
| torch | 2.1.0+cu121 (**hard ceiling** — mmcv prebuilt-wheel constraint) | `[ch3; ctx]` |
| torchvision | 0.16.0 | `[ch3; ctx]` |
| MMSegmentation | 1.2.2 | `[ch3; ctx]` |
| mmcv | 2.1.0 | `[ch3; ctx]` |
| numpy | 1.26.4 | `[ch3; ctx]` |
| scipy | 1.11.4 | `[ch3; ctx]` |
| Pillow | 12.3.0 (corruption `jpeg_compression`) — security-maintenance release superseding 12.2.0 | `[requirements.lock; requirements-e1.txt]` |
| scikit-image | 0.23.2 (corruption `brightness`) — **corrected** from a stale `0.20.0` row, which contradicted `requirements.lock` and additionally hard-requires `PyWavelets`, a package the lock does not contain | `[requirements.lock; PyPI metadata]` |
| OpenCV | 4.8.1 | `[ch3; ctx]` |
| fvcore | 0.1.5.post20221221 | `[ch3; ctx]` |
| imagecorruptions | vendored **1.1.2** (`np.float_`→`np.float64` patched) | `[ctx]` |
| Albumentations | `NEED_TO_CONFIRM` (pinned in manifest, no number given) | `[ch3; ctx]` |
| statsmodels | `NEED_TO_CONFIRM` (no version given) | `[ch3; ctx]` |
| Quant backend | eager-mode `torch.ao.quantization`, **QNNPACK** | `[ch3; ctx]` |
| Compute | RunPod RTX 4090 (~$0.34/hr); exact pod/CPU = `NEED_TO_CONFIRM`, reported in Ch4 | `[ctx; ch3 §D]` |

---

## (e) Architecture — student + LR-ASPP head `[ch3 §C "E1", §D Table 3.4; ctx]`

- **Backbone:** MobileNetV3-Large via `torchvision.models.quantization.mobilenet_v3_large`
  (**quantizable** variant); **native Hard-Swish / ReLU retained** — the global ReLU6 substitution is
  **rejected** `[ch3; ch2]`. ~5M params `[Wei]`.
- **Output stride 16:** atrous (dilation rate **2**) in the final-stage **C4–C5 depthwise** convs, that
  stage's stride set to **1**; the head itself contains **no parallel atrous pyramid** `[ch3]`.
- **LR-ASPP head** (additive fusion at class-logit level, not concatenation) `[ch3]`:
  1. **High-level branch (C5, OS16):** two parallel sub-branches —
     (a) 1×1 **Conv-BN-ReLU → 256 channels**;
     (b) large-kernel **average-pool → 1×1 conv → sigmoid → bilinear upsample**.
     The two are **multiplied element-wise**, then bilinearly upsampled to **OS8**.
  2. **Low-level skip (OS8):** projected by a **1×1 conv → `num_classes`**.
  3. The **upsampled high-level branch** is projected by a **separate 1×1 conv → `num_classes`**.
  4. The two `num_classes` maps are **summed** and bilinearly upsampled to **full resolution**.
- **Output channels = 116** (empirically verified: background 0 + 115 diseases; mask values 0–115) `[ch3; ctx; empirical]`.
- The CWD training-only 1×1 projection head (B3) is **absent from every evaluated model** `[ch3; ctx]`.

---

## (f) Evaluation metrics + statistics `[ch3 §F]`

> **Exact computation is frozen in [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md) (A0, 2026-07-26).**
> This section states *which* metrics the thesis reports; the evaluation contract states *how* each is
> computed — class-eligibility rules (union-present for dataset-level IoU/Dice, GT-present for mAcc and for
> both per-image vectors), ignore-255 masking, undefined-value handling, the bootstrap resampling unit, and
> the result-artifact schema. Decision records: `open_questions.md` D3/D3b/D3c/D4/D5.

**Accuracy**
- **Primary inferential unit:** per-image **disease-only mIoU** (115 disease classes = mask values **1–115**;
  **background = index 0 excluded** [empirical; residual `NEED_TO_CONFIRM` on the formal convention];
  per-image absent-class exclusion; 255 always excluded).
- **Dataset-level all-class mIoU:** TP/FP/FN accumulated per class over the whole test set, macro-averaged;
  basis for **non-inferiority + bootstrap**; matches PlantSeg benchmark reporting.
- **[D1] Checkpoint selection & headline validation metric = all-class mIoU.** Training-time best-val
  checkpoint selection and headline E1/E-stage validation reporting use **all-class validation mIoU**
  (`configs/e1_student.py` `checkpoint_selection`; implemented in `src/training/train_e1.py`). Disease-only
  mIoU is a **secondary/provisional** validation/report metric there. This is **distinct from and does not
  change** the **per-image disease-only mIoU _primary inferential_ unit** above (used for the Ch4 hypothesis
  tests); a future switch of the headline/checkpoint convention to disease-only requires a separate explicit
  decision.
- **Dice:** dataset-level **secondary, descriptive only** — no separate test (monotonic with IoU).
- **mAcc:** descriptive only (benchmark against Wei).

**Robustness**
> **Canonical corruption identifiers are frozen in `configs/corruption_protocol.json`**
> (`plantseg-corruptions/1.0.0`, A3b-0): `motion_blur` · `gaussian_noise` · `jpeg_compression` ·
> **`brightness`** · `fog`, in that order — the vendored `imagecorruptions` reference function
> names. Inferential severities **1–3**; severity **4 descriptive-only**; severity **5 excluded**.
> "brightness variation" is a display label, never an identifier; `brightness_variation` and
> `motion-blur` are rejected. The future corruption generator and cache manifest must **consume
> that file** rather than duplicate the vocabulary. Rationale:
> [STATISTICAL_ANALYSIS_CONTRACT.md](STATISTICAL_ANALYSIS_CONTRACT.md) §2.1.

> **Observed-value provenance and result schema frozen (A3b-2, 2026-07-30).** **Amendment C** corrects
> where each bootstrap task's observed scalar comes from — **24** from typed A3a `ComparisonResult`
> objects, **3** descriptive E1→E3 values from the public A3a estimator primitives, **8** from A3b
> pooled re-accumulation — recorded per task in a required `observed_source` field. E1→E3 stays outside
> the canonical eight-member family and is never routed through `run_comparison`. Separately, the
> complete top-level `family.json` schema is **newly frozen**: nineteen ordered keys, the A+
> `a3a_family` envelope carrying the full finalized `HolmFamily`, the literal A3a dataclass field
> tuples, the `relative_retention` ratio, the 37-record official input set, and the integrity block.
> See [STATISTICAL_ANALYSIS_CONTRACT.md](STATISTICAL_ANALYSIS_CONTRACT.md) **§12.3.3, §12.3.5 and
> §12.4**. Decision records: `open_questions.md` **D23** (amendment) and **D24** (newly frozen). Not
> duplicated here.

- **5 corruptions:** motion blur, Gaussian noise, JPEG compression, brightness, fog. Vendored from
  Hendrycks 2019 reference impl (not the installed package), applied to **uint8 RGB before padding &
  normalization**, **never to masks**; byte-identical cached + checksummed set across stages.
- **mIoU-C:** mean of per-corruption-type mIoU, each averaged over **severities 1–3**.
- Severity **4** = descriptive degradation profile only; severity **5** = **excluded**.
- **RPD** = (mIoU_clean − mIoU_C)/mIoU_clean × 100 — **descriptive only**.
- **rCD** (Kamann 2020) — descriptive only; E1 = internal reference (rCD(E1)=1); teacher excluded.

> **The inferential protocol is frozen in [STATISTICAL_ANALYSIS_CONTRACT.md](STATISTICAL_ANALYSIS_CONTRACT.md)
> (A3-0, 2026-07-28).** This section states *which* analyses the thesis reports; that contract states
> *how* each is computed — test configuration, tie and degenerate policy, effect-size definitions,
> the BCa seed/`z0`/acceleration/fallback rules, and the statistics result artifact. Decision
> records: `open_questions.md` D7-series.

> **Bootstrap reproducibility protocol frozen (A3b-1, 2026-07-29).** The seed namespaces, the exact
> 35 bootstrap tasks and their canonical order, the row-wise draw call and per-task RNG streams, the
> BCa adjusted-probability transform and nominal tails, the deterministic first-trigger P1–P8 fallback
> ladder, int64 sufficient-statistic accumulation, and the `bootstrap.npz` / `family.json` schemas with
> exact cross-representation verification live in
> [STATISTICAL_ANALYSIS_CONTRACT.md](STATISTICAL_ANALYSIS_CONTRACT.md) **§8.7** and **§12.3**. Two rules
> were **amended** there before any implementation or official use — the §8.1 seed bytes now render
> `root_seed`, and the artifact field `confidence_type` became `interval_type`. A pooled dataset-level
> *robustness* estimand is **deferred**, not omitted. Decision records: `open_questions.md` **D21**,
> **D22**; amendment history in **D12**. Not duplicated here.

**Inferential tests (8 total, Holm-Bonferroni family, α = 0.05)**
- Primary: one-tailed **Wilcoxon signed-rank**, `scipy.stats.wilcoxon(d, zero_method='pratt',
  correction=True, alternative='greater', **method='approx'**)`, `d = candidate − baseline`.
- Sensitivity: paired t-test `scipy.stats.ttest_rel(..., alternative='greater')` (CLT at n = **1,561**).
  Sensitivity p-values **never** enter the Holm family.
- Correction: `statsmodels.stats.multitest.multipletests(method='holm')`, strict boundary
  (equality does not reject).
- **The 8 comparison IDs** (candidate − baseline, positive favours the candidate):
  `accuracy_e1_e2` · `accuracy_e2_e3` · `accuracy_e4_e5` · `accuracy_e7_e6` · `accuracy_e4_e7` ·
  `accuracy_e5_e6` · `accuracy_e1_e6` — all on **clean per-image disease-only mIoU** — plus
  `robustness_e1_e6` on **per-image mIoU-C**.

> **Chapter III internal inconsistency, reconciled.** A §B summary sentence lists E1→E3 and E3→E6 as
> though they complete the eight-test family, but Table 3.6 marks **E1→E3 descriptive and "not
> included in the eight-test Holm-Bonferroni family"**, and ch3 states three separate times that the
> **E3→E6 non-inferiority check is reported separately** from that family. Admitting both would give
> ten members against ch3's own explicit eight-test, "seven clean + one robustness" structure. The
> comparison table, formal hypotheses and repeated exclusions therefore govern the summary sentence.
> Full ledger: STATISTICAL_ANALYSIS_CONTRACT.md §0.1.

**Excluded from the family (reported, but not Holm-corrected)**
- **E1→E3** — descriptive total-FP32-distillation gain: mean ΔmIoU and Hodges-Lehmann shift with
  BCa 95 % CIs, **no family p-value, no superiority claim**.
- **E3→E6** — dataset-level non-inferiority (below).
- Teacher comparisons, Dice, mAcc, aAcc, RPD, rCD and efficiency remain descriptive.

**Non-inferiority (separate from the 8)**
- E6 non-inferior to E3 iff the **one-sided 95 % BCa lower bound** on
  `ΔmIoU = mIoU(E6) − mIoU(E3)` (dataset-level, all-class, union-present) is **strictly greater than
  −2.0 pp**; equality fails. **B = 10,000**. Sensitivity decisions at 1.0/1.5/2.0/2.5 pp are read
  from the **same** bootstrap distribution.
- **E6-KD contingency trigger** (stricter, distinct): run if the **observed** clean E3→E6 mIoU drop
  is **> 1.0 pp**; equality does not trigger. No p-value, not a Holm test.

**Effect sizes (reported with each test)**
- matched-pairs **rank-biserial r_rb** = `(R₊ − R₋)/(R₊ + R₋)` with Pratt ranking (zeros are ranked
  but their ranks enter neither signed sum) [−1,1]; **Cohen's dz** (small 0.20 / med 0.50 / large
  0.80); **exact Hodges-Lehmann** shift (median of Walsh averages, i ≤ j); **mean Δ mIoU (pp)**;
  **BCa 95 % CI, B = 10,000** with a single frozen **percentile** fallback when acceleration is
  degenerate. Measured cost: exact HL BCa ≈ 5.3 min per comparison — feasible, not approximated.

**Statistics environment**
- Official p-values and confidence intervals may be produced **only on the pinned stack**
  (`requirements.lock`). Any version mismatch forces non-official status.

**Efficiency (descriptive only)**
- params · model size (deterministic **parameter-byte footprint**: INT8=1B/weight, FP32=4B/weight,
  bias+per-channel scale/zero-point=4B each) **+** serialized on-disk size, measured per the
  **which-artifact rule** below.
- **FLOPs = 2 × MACs** via fvcore @ 512×512, profiled from the **FP32** architecture (QAT does not cut FLOPs).
- **CPU-proxy latency** via `torch.utils.benchmark` (CPU, batch 1, **20 warm-up + 100 measured**,
  median / IQR / p95, `eval()` + `inference_mode()`, AMP off, fixed thread count).
- **Peak memory** via `resource.getrusage` / psutil RSS delta.
- No ARM / on-device latency claimed. All runtime/memory results are **CPU PROXY**.

**Which artifact each efficiency number is measured from** (resolves the earlier ambiguity between
"the same artifact used for latency" and the INT8 backend split — the INT8 rule is the specific one
and governs):

| Precision | accuracy + robustness | serialized size | CPU-proxy latency |
|---|---|---|---|
| **FP32** (teacher, E1–E3) | the model artifact | **same** artifact | **same** artifact |
| **INT8** (E4–E7) | **QNNPACK** artifact | **QNNPACK** artifact | separate **fbgemm/x86** copy (`reduce_range=True`) |

- For FP32 stages one artifact serves all three, so size and latency do refer to the same file.
- For INT8 stages the **QNNPACK** copy is authoritative for accuracy, robustness **and reported
  serialized size**; the fbgemm/x86 copy exists **only** for CPU-proxy latency, is separately
  identified (`artifact_role = x86_cpu_proxy_latency`), and is never substituted into an accuracy,
  robustness or size result. Its activation qparams legitimately differ from the QNNPACK copy's
  because `reduce_range` differs.

**INT8 x86 latency copy — how it is obtained** (no second training run):
- **E4/E7 (PTQ):** rebuilt from the same FP32 source checkpoint and the **same frozen 128-image
  seed-42 shared calibration subset**; calibration image identity is never resampled.
- **E5/E6 (QAT):** the QAT runner writes an **auxiliary companion artifact** alongside the official
  converted model — the best **pre-convert QAT state** (`*_qat_state.pt`,
  `quantization = "qat-train-state"`, `artifact_role = preconvert_qat_state`). It is explicitly
  **not** an accuracy, robustness, size or deployment artifact, and does not alter the official
  converted-artifact schema or the best-validation selection rule. The x86 copy is produced from it
  by **translation only**: the same QAT-trained weights and the same learned activation-range
  evidence are carried over, x86 quantizer parameters are recomputed under the reduced range, and
  **zero optimizer steps / gradient updates** occur. QNNPACK activation qparams are never reused as
  x86 qparams. E5/E6 are **not** re-trained under x86 to obtain a latency artifact, and the x86
  artifact must never be described as x86-QAT-trained.

**Remaining real-run values — what is LOCKED and what stays PILOT-SELECTED**

*E5/E6 QAT controls — LOCKED* (`configs/quant.py['qat_real_run']`, `status = LOCKED`). These are
**thesis implementation choices**, authorized as such; they are **not** values specified by
Krishnamoorthi, Jacob or any other source and must never be cited as though they were. Every entry
applies **identically to E5 and E6**, so the stages differ only in source checkpoint and distillation
history. A real launch still supplies each explicitly — the runner defaults none of them.

| control | value | rationale |
|---|---|---|
| **physical** batch size | **16** | preserves the registered student-training scale rather than adding another E5/E6 difference |
| weight decay | **1e-4** | preserves the registered student regularization scale |
| max epochs | **15** | converts the methodology's "~15 epochs" into a reproducible **maximum** budget |
| early-stop patience | **3** validation checks | makes the already-required early stopping executable; best-val-mIoU checkpoint selection stays a separate mechanism |
| BN-statistics freeze | **0.65** of planned optimizer steps | lower registered endpoint of the methodology's 65–70% window |
| observer freeze | **0.70** of planned optimizer steps | places observer freezing shortly afterwards |

**Batch semantics.** 16 is a **physical** batch. Gradient accumulation is deliberately **not**
introduced: fake-quant observers and BatchNorm statistics are batch-sensitive, so accumulated
micro-batches are not equivalent to one true batch of 16. The GPU must accommodate the registered
batch; if it genuinely cannot, that is an explicit experiment-design issue, not a silent change of
batch semantics.

**Freeze points and early-stop safety.** Freezes are **optimizer-step fractions** of the planned
budget, rounded as `round(total_iters * pct)`, floored at step 1, with the observer freeze clamped
never to precede the BN freeze. Fake quantization runs from step 0. A QAT run must not end before its
quantization schedule has executed, so **early-stop patience accrues only after observer freeze**;
best-checkpoint tracking still begins at the first validation. Without that guard a patience of 3 with
per-epoch validation could terminate around epoch 3–4, before the 65%/70% freezes.

*Gradient clipping — TWO SEPARATE, STILL-UNRESOLVED DECISIONS.* Chapter 3 requires global-norm
clipping throughout distillation training **and** during QAT, and the launchers require a positive
finite `max_norm`, so **"no clipping" is not a candidate** for these stages. E1's separately resolved
unclipped status (D2/D-A) is **not** permission to leave the distilled stages unclipped, and E1 is not
reopened here. Distillation and QAT are different optimization regimes and no source establishes that
one numeric norm should serve both, so they are selected independently:

| decision | applies to | candidates | selection |
|---|---|---|---|
| `DISTILLATION_GRAD_CLIP_NORM` (`configs/distill.py`) | **E2, E3** (same value) | {1.0, 5.0} | pilot on **E2**, λ_logit held at **1.0**, 8,000 iters (10% of official) with validation every 1,000 |
| `QAT_GRAD_CLIP_NORM` (`configs/quant.py`) | **E5, E6** (same value) | {1.0, 5.0} | pilot on **E5** from the official E1 FP32 checkpoint, 5 epochs (official max stays 15) |

Both: seed **42**, **train + validation only, TEST prohibited**, labelled
**HYPERPARAMETER SELECTION / PILOT**, excluded from test evaluation, robustness, hypothesis testing
and the statistics family, and frozen before the corresponding official runs. Selection rule for each:
reject a candidate whose training becomes non-finite or unstable; otherwise take the higher
dataset-level validation mIoU under the identical pilot budget; on a tie within the repository's
existing validation tie/noise rule prefer **5.0** as the less intrusive threshold. No new significance
test is invented. The QAT decision never inherits the distillation result.

**Decision order for E2/E3:** select `DISTILLATION_GRAD_CLIP_NORM` → freeze it → run the existing
λ_logit sweep → freeze λ_logit → official E2/E3 runs. Fixing λ at the grid centre during the clipping
pilot avoids a 2 × 5 Cartesian search while staying inside the registered grid.

`λ_logit` is unchanged — {0.25, 0.5, 1, 2, 4}, seed 42, validation-only, boundary winner reported
rather than extending the grid. It is not decided here.

**Carry forward (manuscript, not resolved here).** Chapter 3 describes E1/E2/E3 as sharing an identical
recipe and attributes their differences to the distillation objectives, yet applies clipping only to
the distillation stages. That wording needs reconciling; execution is governed by the repository
decisions above.

**Official experiment environment (Chapter III reproducibility materials)**
Four artifacts, with distinct and non-interchangeable roles:

| artifact | role |
|---|---|
| `requirements.lock` | human-readable **exact-version registry** — the version authority |
| `requirements-runpod.in` | resolver **input**, derived verbatim from the registry minus Windows-only wheels |
| `requirements-runpod.lock` | **hash-verified** Linux x86-64 / CPython 3.11 install lock (78 distributions, 78 `--hash=sha256:` constraints) |
| `Dockerfile` + `.dockerignore` | the **container specification**; base pinned by immutable digest |

- **Interpreter/base:** `python:3.11-slim-bookworm` @
  `sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91` (Python 3.11.16, glibc
  2.36). The historical `pytorch/pytorch:2.1.0-cuda12.1-*` images are **rejected** — the upstream
  v2.1.0 Dockerfile defaults to `ARG PYTHON_VERSION=3.8` and the published tags do not ship Python
  3.11, so they would violate the registered interpreter.
- **Binary reproducibility, never source builds:** the cu121 torch/torchvision wheels and
  `mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl` (OpenMMLab `cu121/torch2.1.0` index) exist for
  exactly this combination, so **MMCV is never compiled**. `pip install --require-hashes` makes a
  CPU-only torch, a newer torch, a NumPy upgrade, an MMCV source build or a silent "latest"
  resolution a hard failure rather than a wrong experiment.
- **Hash provenance:** every hash came from the actual resolved artifact — pip's own resolution
  report on Linux, plus two artifacts whose index publishes no hash fragment which were downloaded
  and hashed directly (and confirmed identical to the PyPI copies). No hash is hand-written.
- **Preflight:** `scripts/preflight_environment.py` is the single executable authority, with
  `--mode image` (software identity; CPU-only, runs in `docker build`) and `--mode gpu` (the official
  preflight; fails unless a real CUDA device is present and records GPU/CPU/driver metadata). A
  CPU-only host can never satisfy the official mode.
- **Image vs run:** the image defines **software**; the run record defines **hardware**. No GPU model
  is baked in. Dataset and checkpoints are mounted, never baked — `.dockerignore` denies by default
  and re-excludes datasets, checkpoints, archives, keys and the protected `docs/reference/`.
- **Two states, never conflated:** *container specification complete* (current) versus *full official
  experiment environment executably validated* — the latter requires an actual build plus a passing
  `--mode gpu` run, and until then `full_experiment_environment_validated = false`. Residual gap:
  apt package versions follow the pinned base digest rather than being individually pinned.

Full procedure: [runpod_environment.md](runpod_environment.md).

**Corruption dependency pins — REGISTERED and EXECUTABLY VALIDATED (corruption closure only)**
The corrupted bytes depend on three libraries: **numpy** (all corruptions), **Pillow**
(`jpeg_compression`), **scikit-image** (`brightness`). The registered pins, taken from
`requirements.lock` and verified against primary PyPI metadata for the recorded official
**Python 3.11**, are:

| package | version | evidence |
|---|---|---|
| numpy | **1.26.4** | unchanged |
| Pillow | **12.3.0** | `requires_python ">=3.10"`, cp311 wheels published; released 2026-07-01 |
| scikit-image | **0.23.2** | `requires_python ">=3.10"`; `numpy>=1.23`, `scipy>=1.9`, `pillow>=9.1`; cp311 wheels published |

**Why 12.3.0 supersedes 12.2.0.** 12.3.0 is a later upstream security-maintenance release
(CVE-2026-55798; the CVE-2026-54059 / 54060 / 55379 / 55380 decompression-bomb set; out-of-bounds
fixes in TGA, RankFilter, `Image.paste()` and ImageCmsTransform). Its release notes document **no**
JPEG encode/decode change, and that was confirmed executably rather than assumed: the registered
`jpeg_compression` grid produces **byte-identical** output under 12.2.0 and 12.3.0, and the full
40/40 zero-tolerance reference-equivalence grid passes under 12.3.0. **No official thesis artifact
or corruption cache had been produced under 12.2.0**, so moving the pin costs nothing and gains the
security fixes.

`0.23.x` dropped scikit-image's hard `PyWavelets` dependency, which is why `requirements.lock`
correctly contains no PyWavelets entry — the decisive evidence that the lock, not the old `0.20.0`
contract row, is the coherent artifact. The corruption runtime closure imports only
numpy/Pillow/scikit-image; **cv2 and scipy are not corruption-runtime dependencies** (scipy remains
in the stack for statistics and as a scikit-image transitive dependency).

**Two phases, never collapsed into one Boolean:**
- `dependency_versions_registered = true` — the exact versions above are recorded and enforced at
  runtime by `src.corruption_cache.require_official_dependency_environment()`.
- `official_dependency_environment_validated = true` — that exact stack was installed in an
  isolated **Python 3.11.15** environment (numpy 1.26.4 · Pillow 12.3.0 · scikit-image 0.23.2 ·
  scipy 1.11.4) and `scripts/smoke_corruption_vendor.py` passed there, including the **40/40
  zero-tolerance** upstream reference-equivalence grid. `byte_reproducibility_proven_under_these_pins`
  is therefore true **for the corruption closure**.

That flag is a **historical record**, not a claim about whatever interpreter happens to be running:
`runtime_pin_mismatches()` independently checks the live versions, and cache generation requires
**both** — a validated stack *and* a matching runtime. Running the corruption smoke on any other
interpreter remains development evidence only.

**Scope limit — two different claims, never interchangeable:**
- **Validated:** the corruption dependency closure (Python, numpy, Pillow, scikit-image and
  scikit-image's own runtime dependencies). `validation_scope = corruption_dependency_closure_only`.
- **NOT validated:** the full official experiment environment —
  `full_experiment_environment_validated = false`. torch 2.1.0+cu121, torchvision 0.16.0,
  mmsegmentation 1.2.2, mmcv 2.1.0, fvcore, CUDA and the RunPod container are now **specified**
  (see *Official experiment environment* below) but have not been executably validated together.

Generating the official 31,220-item cache remains a **separate** operation: it reads the PlantSeg
test split and produces a large governed artifact. This validation only removes the
dependency-version blocker.

**Descriptive backend-accuracy parity (ch3: "any accuracy difference between the two quantization
configurations is documented")**
- Because the x86/fbgemm copy uses a backend-specific quantization configuration, its clean
  segmentation metrics may differ from the authoritative QNNPACK artifact's. For **E4–E7** those
  metrics are therefore evaluated **once** as a descriptive **backend-parity check** on the same
  governed clean test split (**1,561 rows**, same manifest, preprocessing, `ignore_index` and metric
  reducers as the normal clean evaluator — no second evaluator is built), and the difference from
  QNNPACK is reported for all-class mIoU, macro Dice, mAcc and disease-only mIoU.
- Metrics stay on the repository scale (**fractions in [0, 1]**); any percentage-point figure is the
  same value × 100 and is labelled as such. The delta is defined as **x86 − QNNPACK**.
- **No acceptable-difference threshold is defined.** The requirement is to DOCUMENT the difference,
  not to gate on it.
- These x86 metrics **do not** enter model selection, robustness evaluation, hypothesis testing, or
  the official E-stage accuracy results. The record is written separately
  (`evaluation_role = descriptive_x86_backend_parity`) and never into an official clean-evaluation
  artifact directory, so statistics and robustness consumers cannot discover it as a stage score.
  The **QNNPACK** artifact remains authoritative for accuracy, robustness and serialized size.

---

## (g) Open / NEED_TO_CONFIRM list

Full detail + resolution mechanism in [open_questions.md](open_questions.md); full disagreement log in
[conflicts.md](conflicts.md). Summary:

**✅ RESOLVED empirically (2026-06-27, see [reports/dataset_report.md](../reports/dataset_report.md)):**
1. **`num_classes` = 116** — masks contain contiguous **0–115** (max non-ignore 115). Background = **0**,
   diseases = **1–115** (`mask = category_id + 1`, 99.85% consistent). Wei's "0–114" = COCO `category_id`s.
   Earlier 115-vs-116 conflict resolved toward **116** for all-class segmentation. Output layer = 116.
- Empirical split sizes **5,367 / 846 / 1,561** (replacing the contract's earlier ≈ guesses).
- **EXIF**: `ImageOps.exif_transpose` on images (never masks) is mandatory before pairing/transform.

**OPEN / NEED_TO_CONFIRM remaining (do not guess):**
2. **Disease-only / background convention** — empirically background = **index 0** (exclude for disease-only
   mIoU; diseases 1–115), but no literal "background" category is named in the dataset files (COCO
   `categories` empty), so `reduce_zero_label` and the formal disease-only exclusion stay **NEED_TO_CONFIRM**
   pending the PlantSeg repo's official convention.

**NEED_TO_CONFIRM (not stated in any source; filled by experiment/selection, never guessed):**
- `λ_logit` — validation sweep over {0.25, 0.5, 1, 2, 4}, reported Ch4.
- E6-KD reduced distillation weights (only if contingency triggers).
- Albumentations version; statsmodels version.
- Multi-seed values beyond seed 42 (for E1/E3 three-seed runs).
- Final RunPod pod type / GPU / CPU model / CUDA image (reported Ch4).
- Teacher recovered mIoU and all student result numbers (E1–E7 outcomes) — produced by training, out of Week-1 scope.

**Pre-training verification gates (mandatory, Table 3.1, before any E1 training) `[ch3; ctx]`:**
mask class count (np.unique) · image↔mask pairing (name/readability/dims) · split integrity (zero ID
overlap) · mask value range ⊆ class set ∪ {255} · ignore-label unit tests (CE/Dice/LogitKD/CWD unchanged
when padded values altered; CWD channel-wise softmax sums to 1 over valid locations; teacher hook returns
stride-16 MSCAN-B Stage-3 320-ch feature). Store verification logs as JSON next to each run.
**Plus** QNNPACK INT8-Sigmoid support in the LR-ASPP global-pool branch must be verified before E4.

### (g.1) Tool/environment precondition-verification phase `[ch3 ~p.112, ~p.113, ~p.11–12]`

Distinct from the Table 3.1 *data* gates above, Chapter III also requires **tool-specific preconditions**
to be verified through pilot/smoke checks before the pipeline proceeds:

> *"Tool-specific preconditions (e.g., QNNPACK backend operator support for INT8 Sigmoid in the LR-ASPP
> head, MMSegmentation model-zoo checkpoint availability for SegNeXt-B, and RunPod GPU pod availability)
> are verified during pilot runs before E1 training begins."* `[ch3 ~p.112]`

**Stage-specific applicability.** Chapter III's ~p.112 sentence groups all three preconditions under
"before E1 training begins", while ~p.113 and ~p.11–12 specifically tie QNNPACK to *pilot quantization,
before the quantization stages are run*. That source tension is recorded openly; the **stage-specific
reading is adopted**:

| Precondition | Governs | Evidence / status |
|---|---|---|
| **RunPod GPU/pod availability** | **before real E1 execution** | verified on the pod (`scripts/verify_env.py` → PASS, `train_e1.py --dry-run` → PASS) |
| **SegNeXt-B / MMSeg checkpoint availability** | **before teacher-dependent execution** (teacher fine-tune, E2/E3 prep) | `docs/B8_checkpoint.md` — none publicly released; in-house fine-tune planned |
| **QNNPACK INT8 operator support** | **before the quantization stages (E4–E7)** | `docs/b7_result.md` — onednn proxy `PASS_CLEAN`; authoritative run env-gated |

**QNNPACK is a pre-quantization gate and is NOT an independent blocker to FP32 E1.**

**No training-pilot requirement.** Chapter III specifies **no** short-training-run pilot and **no**
iteration count for this phase. The "2,000-iteration E1 pilot" figure appears **only** in
`docs/reference/context.md` `[ctx]`; it is operational, is **not** part of the locked methodology, and does
**not** modify the **80,000-iteration / validate-every-4,000** recipe in §(d) B2. An optional short training
rehearsal is tracked as `[operational] RECOMMENDED_NOT_BLOCKING` — see
[open_questions.md](open_questions.md) **D6** and
[reports/e1_runpod_launch_runbook.md](../reports/e1_runpod_launch_runbook.md).
