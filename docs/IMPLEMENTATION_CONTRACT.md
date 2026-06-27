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
| Split sizes (computed) | ≈ **5,442 / 778 / 1,554** from 7,774 total | `[ch3 §E]` |
| Split files | `annotation_train.json`, `annotation_val.json`, `annotation_test.json` | `[ch3 §E]` |
| Test images used everywhere | **1,554** (per-image metric unit count) | `[ch3 §E]` |
| Image format | **JPEG** (`images/`) | `[Wei]` |
| Mask format | **PNG grayscale** (`annotations/`) | `[Wei]` |
| Mask class indices | **0–114** for the 115 plant-disease combinations | `[Wei]` |
| Ignore index | **255** (padding regions; excluded from all loss & metrics) | `[ch3]` |
| `num_classes` | **OPEN — DO NOT HARDCODE.** See conflict below. | `[Wei]` vs `[ch3]` |
| `reduce_zero_label` | `NEED_TO_CONFIRM` (not stated in any source; tied to the num_classes/background question) | — |
| Zenodo DOI | **10.5281/zenodo.17719108** | `[Wei; ch3]` |
| License (dataset) | **CC BY-NC 4.0** | `[Wei; ch3]` |
| License (the article itself) | CC BY-NC-**ND** 4.0 (Nature) — distinct from dataset license; do not conflate | `[Wei]` |
| Imbalance note | mask ratios of ~80% of images are below 36% (small-lesion regime) | `[Wei; ch3]` |

**Benchmark reference numbers (contextual only, `[Wei]` Table 3):**
SegNeXt-B/MSCAN-B = 28M params, **42.05%** mIoU / 56.30% mAcc · MobileNetV3 = 5M, 10.22% / 17.15% ·
SegNeXt-L/MSCAN-L = 49M, 44.52% mIoU · ConvNeXt-L = 121M, 46.24% / 59.97% (highest).

### ⚠ Critical OPEN conflict — `num_classes`
- `[Wei]`: "we use indices **0 to 114** to represent the corresponding plant-disease combinations" → **115** disease classes.
- `[ch3]`: provisionally states **`num_classes = 116`** ("the background class and 115 disease classes"),
  but is **internally inconsistent** (elsewhere writes "115 output channels", and itself calls 116
  "provisional until confirmed by the mask-value verification in Table 3.1").
- **Resolution (locked rule):** size the final classifier to the **empirically verified** count from
  `np.unique()` over the real annotation PNGs. **Do not pick a winner now.** `[ch3 Table 3.1; ctx]`

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
| Checkpoint selection | **best validation-mIoU** | `[ch3]` |
| Distillation-weight ramp (E2/E3) | linear **0 → target over first epoch** | `[ch3]` |
| Gradient clipping | global-norm, throughout | `[ch3]` |
| Teacher in loop (E2/E3) | eval mode, online, consumes **identical augmented input** as student | `[ch3]` |
| Optional control | extended-schedule E2 (~160,000 iters, 1 seed) — optional, future work if not run | `[ch3 §C]` |

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

| Library | Version | Source |
|---|---|---|
| Python | 3.11 | `[ch3; ctx]` |
| torch | 2.1.0+cu121 (**hard ceiling** — mmcv prebuilt-wheel constraint) | `[ch3; ctx]` |
| torchvision | 0.16.0 | `[ch3; ctx]` |
| MMSegmentation | 1.2.2 | `[ch3; ctx]` |
| mmcv | 2.1.0 | `[ch3; ctx]` |
| numpy | 1.26.4 | `[ch3; ctx]` |
| scipy | 1.11.4 | `[ch3; ctx]` |
| scikit-image | 0.20.0 | `[ctx]` |
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
- **Output channels = empirically verified class count** (see the OPEN num_classes conflict) `[ch3; ctx]`.
- The CWD training-only 1×1 projection head (B3) is **absent from every evaluated model** `[ch3; ctx]`.

---

## (f) Evaluation metrics + statistics `[ch3 §F]`

**Accuracy**
- **Primary inferential unit:** per-image **disease-only mIoU** (115 disease classes; background excluded;
  per-image absent-class exclusion; 255 always excluded).
- **Dataset-level all-class mIoU:** TP/FP/FN accumulated per class over the whole test set, macro-averaged;
  basis for **non-inferiority + bootstrap**; matches PlantSeg benchmark reporting.
- **Dice:** dataset-level **secondary, descriptive only** — no separate test (monotonic with IoU).
- **mAcc:** descriptive only (benchmark against Wei).

**Robustness**
- **5 corruptions:** motion blur, Gaussian noise, JPEG compression, brightness, fog. Vendored from
  Hendrycks 2019 reference impl (not the installed package), applied to **uint8 RGB before padding &
  normalization**, **never to masks**; byte-identical cached + checksummed set across stages.
- **mIoU-C:** mean of per-corruption-type mIoU, each averaged over **severities 1–3**.
- Severity **4** = descriptive degradation profile only; severity **5** = **excluded**.
- **RPD** = (mIoU_clean − mIoU_C)/mIoU_clean × 100 — **descriptive only**.
- **rCD** (Kamann 2020) — descriptive only; E1 = internal reference (rCD(E1)=1); teacher excluded.

**Inferential tests (8 total, Holm-Bonferroni family, α = 0.05)**
- Primary: one-tailed **Wilcoxon signed-rank**,
  `scipy.stats.wilcoxon(d, zero_method='pratt', correction=True, alternative='greater')`.
- Sensitivity: paired t-test `scipy.stats.ttest_rel(..., alternative='greater')` (CLT at n≈1,554).
- Correction: `statsmodels.stats.multitest.multipletests(method='holm')`.
- The 8 tests: **E1vE2, E2vE3, E4vE5, E7vE6, E4vE7, E5vE6, E1vE6 (mIoU), E1vE6 (mIoU-C)**.

**Non-inferiority (separate from the 8)**
- E6 non-inferior to E3 iff lower bound of **paired BCa 95% CI** on `ΔmIoU = mIoU(E6) − mIoU(E3)` > **−2.0 pp**,
  **B = 10,000** (`scipy.stats.bootstrap(method='BCa')`).
- **E6-KD contingency trigger** (stricter, distinct): run if E3→E6 clean mIoU drop > **1.0 pp**.

**Effect sizes (reported with each test)**
- matched-pairs **rank-biserial r_rb** [−1,1]; **Cohen's dz** (small 0.20 / med 0.50 / large 0.80);
  **Hodges-Lehmann** shift (Walsh averages); **mean Δ mIoU (pp)**; **BCa 95% CI, B = 10,000**
  (documented percentile/basic fallback for degenerate zero-inflated medians; HL BCa ≈ 7–8 min, O(n²)).

**Efficiency (descriptive only)**
- params · model size (deterministic **parameter-byte footprint**: INT8=1B/weight, FP32=4B/weight,
  bias+per-channel scale/zero-point=4B each) **+** serialized on-disk size of the same artifact used for latency.
- **FLOPs = 2 × MACs** via fvcore @ 512×512, profiled from the **FP32** architecture (QAT does not cut FLOPs).
- **CPU-proxy latency** via `torch.utils.benchmark` (CPU, batch 1, **20 warm-up + 100 measured**,
  median / IQR / p95, `eval()` + `inference_mode()`, AMP off, fixed thread count).
- **Peak memory** via `resource.getrusage` / psutil RSS delta.
- x86 CPU proxy = separate **fbgemm/x86 INT8** copy (`reduce_range=True`); the **QNNPACK** copy stays the
  reported accuracy/size artifact. No ARM / on-device latency claimed.

---

## (g) Open / NEED_TO_CONFIRM list

Full detail + resolution mechanism in [open_questions.md](open_questions.md); full disagreement log in
[conflicts.md](conflicts.md). Summary:

**OPEN (authoritative sources disagree / unresolved — resolve empirically, do not guess):**
1. **`num_classes`** — `[Wei]` 0–114 = 115 disease classes vs `[ch3]` provisional 116 (bg + 115).
   Resolve via `np.unique()` on real masks (Table 3.1) **before E1**. Do not pick a winner.
2. **`reduce_zero_label` / background index** — whether a distinct background index exists or background
   is encoded inside 0–114. Same `np.unique()` resolution.

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
