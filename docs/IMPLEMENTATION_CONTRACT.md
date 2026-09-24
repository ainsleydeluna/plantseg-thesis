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
- Scope guardrails: ~~**no on-device / ARM claims**; CPU-proxy efficiency only. Single-seed vertical
  slice (E1→E7 + full eval) before any multi-seed scaling~~ **[UPDATED 2026-09-24 — B65 CP-006]** no
  on-device claims (Chapter 3 p. 141 stands); the x86 fbgemm-copy CPU-proxy path stays official, and a
  supplementary server-class ARM latency measurement is descriptive (AM-16 item 7). Multi-seed and
  longer-schedule runs follow the `docs/DECISION_LOG.md` Current plan (AM-1, AM-16) `[ch3 §D; ctx; AM-16]`.
- The teacher is a **descriptive upper-bound reference only** — never an inferential comparator `[ch3 §A]`.
- Published Wei et al. (2026) numbers are **contextual reference only**, never inferential comparators
  (no per-image scores available; preprocessing/training/eval mismatched) `[ch3 §A]`.

---

## (b) Pipeline + the seven experiments (E1–E7) + Teacher

`[ch3 Table 3.2; ctx]`

| Stage | Starts from | Adds / does | Main comparison |
|---|---|---|---|
| **Teacher** | ADE20K-pretrained SegNeXt-B / MSCAN-B (MMSeg zoo) | Fine-tune on PlantSeg — **thesis-derived SegNeXt-B teacher configuration**; Wei (2026) 42.05% mIoU is a contextual published value only (see B1). **Not deployed.** | Teacher vs E6 (descriptive only) |
| **E1** | ImageNet-pretrained quantizable MobileNetV3-L | FP32 baseline — no KD, no quant (anchor baseline) | E1 vs E2; E1 vs E3; E1 vs E6 |
| **E2** | E1 init (trained independently) | + **Logit KD** | E1 vs E2 |
| **E3** | E1 init (trained independently) | + **Logit KD + CWD** (**proposed FP32 student**) | E2 vs E3 |
| **E4** | E1 FP32 checkpoint | **INT8 PTQ** | E4 vs E5; E4 vs E7 |
| **E5** | E1 FP32 checkpoint | **INT8 QAT** | E4 vs E5; E5 vs E6 |
| **E6** | E3 checkpoint (**CWD head removed**) | **INT8 QAT — full proposed pipeline** | E3 vs E6; E7 vs E6; E1 vs E6; Teacher vs E6 |
| **E7** | E3 checkpoint (**CWD head removed**) | **INT8 PTQ** | E7 vs E6; E4 vs E7 |

- E4–E7 form a **2×2 quantization × distillation matrix** (PTQ vs QAT) × (undistilled E1 vs distilled E3) `[ch3 §A,C]`.
- **E6-KD** is a *pre-registered contingency arm*, not a locked stage `[ch3 §C, §F]`.
  **[LOCKED 2026-09-23 — AM-3]** It runs only if the E3→E6 drop in dataset-level **VAL** all-class
  mIoU, with E6 scored on the converted INT8 model, exceeds **1.0 pp at seed 42**. E6-KD then runs at
  seed 42 with the Logit-KD and CWD weights at **0.5×** their E3 values, T unchanged; it is descriptive
  and outside the Holm family. The clean-TEST trigger is withdrawn. *(Was: "run only if the E3→E6
  clean-test mIoU drop exceeds 1.0 pp; reinstates frozen teacher with reduced distillation weights
  during QAT", registered as a METHODOLOGY DECISION OPEN `[B59 C4]`.)*
- All stages share the **same official 70/10/20 split, preprocessing, 512×512 resolution, normalization,
  mask formatting, metrics, and corruption settings**; only the compression component varies `[ch3 §C]`.
- All training runs are **iteration-matched (80,000 iters)**, not wall-clock- or FLOPs-matched `[ch3 §C]`.
  **[UPDATED 2026-09-24 — B65 CP-006]** Exception: the AM-16 item-3 longer-schedule controls (E1, E2 and
  E3 at seed 42 with 160,000 iterations, descriptive, with measured GPU-hours) `[AM-16]`.

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
| `reduce_zero_label` / background | **background = index 0** (empirical: present in 7773/7774 masks, 80.6% of pixels). ~~`reduce_zero_label` itself = `NEED_TO_CONFIRM` (implementation choice; no literal "background" category named in dataset files).~~ **[UPDATED 2026-09-23 — B64 C5]** open_questions #2 RESOLVED (D1 + A0-FIX 2026-07-26): `reduce_zero_label = False`; index 0 is the non-disease slot and 1–115 are the diseases (official PlantSeg METAINFO). | `[empirical]` |
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
~~**Residual `NEED_TO_CONFIRM`:** the *disease-only* metric must exclude background = **index 0**
(empirically), but no literal "background" category is named in the dataset files (COCO `categories`
list is empty), so `reduce_zero_label` and the formal disease-only exclusion stay `NEED_TO_CONFIRM`
pending the PlantSeg repo's official convention.~~ `[empirical; ch3 Table 3.1; ctx]`
**[UPDATED 2026-09-23 — B64 C5]** open_questions #2 RESOLVED (D1 + A0-FIX 2026-07-26): `reduce_zero_label = False`; index 0 is the non-disease slot and 1–115 are the diseases (official PlantSeg METAINFO).

---

## (d) Locked configs (every value traced)

### B1 — Teacher fine-tune `[ch3 §C, Table 3.2/3.5]`
| Param | Value | Source |
|---|---|---|
| Base / init | SegNeXt-B / MSCAN-B, ADE20K-pretrained; MMSeg zoo ckpt ~~`segnext_mscan-b_512x512_160k_ade20k` (or equivalent)~~ **[G10 closed, B62]** MMSeg 1.x `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512`, file `segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth`, SHA-256 `647a0cda…40ef1` (readiness PASS, B61 §1); ch3's name is the 0.x spelling of the same model | `[ch3 §D; B61 §1]` |
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
| Success criterion | **LOCKED (M5, B60, 2026-09-22): readiness rule R1–R4, not a published-value band.** Wei 42.05 mIoU / 56.30 mAcc / ~28M params are **contextual published values only**: not a reproduction target, acceptance band, protocol-match criterion or retraining trigger. The ch3 "±1.5–2.0 pp — protocol matching" rule has no methodological authority from 2026-09-22. **R1** integrity: official preflight, first-train/first-val attestations, finite loss, all 40,000 iterations, no resume. **R2** TRAIN/VAL only; TEST never consulted. **R3** after M4: a *controlled deterministic re-evaluation under the M4-locked evaluation rule*, by the thesis VAL evaluator (EVALUATION_CONTRACT §7.2); VAL all-class mIoU must be **strictly greater than 0.36314016580581665** (E1 run of record), with no margin; report the gap and disease-only mIoU. An operational competence floor on the same VAL used for selection, **not** an independent estimate or an inferential comparison. **R4** Stage-3 320 ch @ stride 16 and the input-equivalence guard pass on the trained teacher. **R3 failure → STOP and escalate** (no retraining, search, band relaxation or TEST). **[AM-13]** The comparison with Wei's 42.05% is descriptive and made at TEST evaluation only, on the teacher's upstream-protocol score (lane L-AM13) | `[B60 §2; ch3 roles; MEASURED E1; AM-13]` |
| Warmup (LOCKED, M5) | LinearLR, 1,500 iterations, `start_factor` 1e-6 | `[REPO-UP; B60 §2.2]` |
| Poly (LOCKED, M5) | PolyLR power **1.0**, **end 40,000** (full decay over the run) | `[REPO-UP power; THESIS-DERIVED end; B60]` |
| Validation (LOCKED, M5) | every **4,000** iterations, **VAL only**; no TEST consultation | `[REPO-UP MMSeg schedule_40k; MS by analogy; B60]` |
| Checkpoint selection (LOCKED, M12) | validations at 4k, 8k, …, 40k, each under **M4-V**. Select the **numerically highest VAL all-class mIoU**; exactly equal stored values → **earliest iteration**; no tolerance/noise-band tie rule; no TEST; no training extension. Persist per validation: iteration, all-class mIoU (full precision — not MMSeg's 2-decimal summary), disease-only where reported, NMF seed, VAL manifest hash/order identity; plus the selected iteration and checkpoint SHA-256. R3 follows under M4-V on the selected checkpoint; it need not be byte-identical to the MMSeg selection metric | `[B61 §6; B60 §2.3]` |
| NMF / Hamburger (LOCKED, M4) | **`rand_init = True`**, randomness preserved but isolated. **M4-T** training: upstream fresh bases from the run's seeded global CPU stream. **M4-V** every complete evaluation pass (VAL selection, R3, final evaluation): save caller CPU RNG → dedicated NMF stream from **seed 42** → whole split in a frozen deterministic order, **batch 1**, fresh basis per image → restore caller RNG exactly; manifest/order persisted or hash-attested. **M4-KD** frozen teacher in E2/E3: private NMF stream initialised **once** from seed 42; per forward save caller → install private → forward → capture advanced → restore caller; never reset per batch. Not adopted: `rand_init=False`, a fixed basis, multi-draw averaging, image-keyed bases | `[PRIMARY Hamburger App. F; UPSTREAM; MEASURED B61 §2; THESIS-DERIVED; B61 §4]` |
| Loss weighting (LOCKED, M5) | **unweighted** CE; student class weights are not imported | `[ch3; B60]` |
| CE ignore normalisation (LOCKED, M13) | teacher decode-head CE **`avg_non_ignore = True`**, `ignore_index = 255`: ignore/padded pixels enter neither the numerator nor the mean denominator (mean over valid pixels). Thesis-derived correction of the MMSeg default; implemented before the official teacher run | `[MS ch3 p.128; MEASURED B61 §7; REPO E1 losses.py:53; THESIS-DERIVED]` |
| Train augmentation (LOCKED, M2) | **semantic parity with the E1–E3 recipe.** Rotation ±10° with p 0.5, before the crop (image fill ImageNet mean, mask 255); 512 crop with cat_max_ratio 0.95; independent horizontal and vertical flips, each p 0.5; image-only hue ±0.015 and saturation [0.8, 1.2], jointly p 0.5. **No** brightness, contrast, blur, noise or JPEG (no `PhotoMetricDistortion`). Parity is semantic; code, draws and RNG need not be identical | `[MS student recipe; PRIMARY only for flip/scale/crop; THESIS-DERIVED for the teacher; B60 §3]` |
| Train scale (LOCKED, M3) | long side = 512·r, r ~ U[0.75, 2.0], applied to the unpadded image, then crop/pad to 512. Clean VAL evaluation stays long side 512 + pad (thesis evaluator) | `[MS student recipe; THESIS-DERIVED for the teacher; B60 §4]` |
| Data isolation (LOCKED, M11) | configured data root staged with **TRAIN + VAL only**; `images/test` and `annotations/test` absent; preflight checks 5,367 / 846 and **fails closed** if TEST paths exist. No active `test_dataloader`, `test_evaluator` or `test_cfg`. Scope = the configured data root, not the host. Applies to the teacher, E2 and E3 **[UPDATED 2026-09-24 — B66-prep S2, DL-21]** and, from B66, every real E1 run (seeds 43/44 and the longer-schedule E1): `train_e1.py` refuses a real run on a root failing `assert_trainval_only_root` (5,367/846), and `scripts/preflight_e1_trainval.py` runs it in place of `verify_env.py`'s dataset loop | `[MS TEST policy; B60 §5]` |

> **Runtime non-conformance, intentional until the unified implementation (B60 §9 step H, §10).** The
> current runtime config (`510b212b…`), launcher (`58575276…`), smokes and `configs/teacher_finetune.py`
> still carry the old teacher pipeline:
> - the upstream short-side pipeline with `PhotoMetricDistortion`;
> - `VAL_INTERVAL` 10,000;
> - active TEST surfaces;
> - a TEST filename count;
> - **[B61]** no M4 NMF RNG isolation (`NMF_SEED_CONTROL = 'NEED_TO_CONFIRM'`), the default
>   `avg_non_ignore=False` (M13), and a checkpoint interval of 10,000 with `save_best='mIoU'` on MMSeg's
>   rounded summary value (M12).
>
> The locks above govern. **The current runtime must not be used for an official teacher run.**
> Implementation is the unified governed change of B61 §9 step 2.
>
> **[UPDATED 2026-09-22 — B62]** Implemented in the B62 working tree (config, launcher, KD adapter,
> evaluator, smokes; `reports/b62_teacher_runtime_reconciliation.md`), CPU-validated, ~~**pending review and
> commit, the new runtime hash freeze and a G20-style CUDA re-canary**~~. The pre-B62 runtime
> (`510b212b…`, `58575276…`) is superseded and must not be used for an official run.
> **[UPDATED 2026-09-23 — B64]** Committed in `3c43f89` and pushed; runtime hashes frozen at that commit
> (`docs/teacher_prep_runbook.md` §4a step 8). Pending: the G20-style CUDA re-canary, G21 (teacher
> batch-16 VRAM), the official TRAIN/VAL preflight and an explicit teacher-training GO.

> **Provenance — THESIS-DERIVED SEGNeXt-B TEACHER CONFIGURATION `[B59 B1–B4, 2026-09-21]`.**
> ch3 attributes the AdamW 6e-5 / wd 0.01 / head lr_mult 10 / poly / 40k recipe to "the Wei et al.
> (2026) PlantSeg repository configuration that produced the published 42.05% benchmark". That
> attribution is **incorrect**:
> - The Wei paper reports its benchmark trained with **SGD, lr 0.001, momentum 0.9, weight decay
>   0.0005, cross-entropy, batch 16**.
> - The public PlantSeg repository (pinned commit `1a3dd4d9…`) ships **no MSCAN-B PlantSeg config**.
>   Its SegNeXt T/L PlantSeg configs use an **ImageNet-pretrained** MSCAN backbone with AdamW 6e-5.
>
> The rows above are therefore a thesis-derived configuration, informed by Guo et al. (2022) and the
> public PlantSeg SegNeXt-family conventions. It must never be described as an exact reproduction of
> the published 42.05. **ADE20K initialization is unchanged** — it is ch3's explicit design choice.
>
> Implementation details ch3 does not state (source-derived; `configs/teacher/…plantseg116-512x512.py`):
> - LinearLR warmup of 1,500 iterations;
> - PolyLR power 1.0, with its end horizon-corrected from the public 160k to 40k;
> - validation every 10,000 iterations.
>
> **[UPDATED 2026-09-22 — B60]** The schedule details above are now **LOCKED (M5)**. The one change is
> validation: **every 4,000 iterations**, replacing 10,000. PlantSeg's 10,000 came from an upstream
> configuration whose validation ran on the TEST split; MMSeg's own 40k schedule uses 4,000.
>
> - **LOCKED:** teacher train augmentation (M2) and teacher train/eval scaling (M3), in the rows above.
> - ~~**Still METHODOLOGY DECISION OPEN:** NMF/Hamburger RNG control (**M4**, `NEED_TO_CONFIRM`) and the
>   operational checkpoint-selection rule (**M12**).~~ **[UPDATED 2026-09-22 — B61]** **LOCKED:** NMF/Hamburger
>   control (**M4**), checkpoint selection (**M12**) and the new teacher CE ignore normalisation (**M13**),
>   in the rows above. ADE20K checkpoint readiness **PASS** (SHA-256 `647a0cda…40ef1`; B61 §1;
>   `docs/teacher_init_source.md`).
>
> See `reports/b59_pre_runpod_reconciliation.md`, `reports/b60_teacher_methodology_lock.md` and
> `reports/b61_teacher_nmf_checkpoint_selection_lock.md`.

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
| Gradient clipping | ch3 p.104 places it in the distillation-stage sentence: "During the distillation stages … global-norm gradient clipping is applied throughout". **LOCKED 2026-09-23 (AM-7): E1, E2 and E3 share one rule — no clipping** (E1 seed 42 logged no gradient norm). A NaN/divergence (as defined in AM-7) in any E2/E3 run stops the stage; one clipping rule is then adopted for all three and the FP32 stages are rerun. The E2/E3 launcher gate changes in lane L-AM7. *(Was: E1 none; E2/E3 METHODOLOGY DECISION OPEN, B59 C2.)* | `[ch3; B59 A5/C2; AM-7]` |
| Teacher in loop (E2/E3) | eval mode, online, consumes **identical augmented input** as student | `[ch3]` |
| ~~Optional control~~ Longer-schedule controls **[UPDATED 2026-09-24 — B65 CP-006]** | ~~extended-schedule E2 (~160,000 iters, 1 seed) — **not run; recorded as future work (AM-11)**~~ **Run, descriptive (AM-16 item 3):** E1, E2 and E3 at seed 42 with 160,000 iterations each (poly schedule over the 160,000-iteration horizon; VAL every 4,000 iterations; best-checkpoint selection as in the 80,000-iteration runs; E2/E3 use the selected λ and α). Compared on clean TEST mIoU, with each run's measured GPU-hours reported alongside: each 160,000-iteration run against its 80,000-iteration run; E3 at 80,000 against E2 at 160,000 (the Chapter 3 sanity check); E2 and E3 at 80,000 against E1 at 160,000 (a longer-trained-baseline control; not compute-matched). Code: lanes L-AM16-ITERS and L-AM16-GPUH **[UPDATED 2026-09-24 — B66-prep S3]** E1: `train_e1.py --iterations 160000` sets the poly horizon and the run length (L-AM16-ITERS, E1 part; default 80000 = the locked recipe). A run refuses `--max-iters` above its horizon; a real run refuses any other `--max-iters`; `--resume` refuses a checkpoint whose scheduler `total_iters` differs. E2/E3: KD part pending (required before any E2/E3 160,000-iteration launch). GPU-hours: L-AM16-GPUH | `[ch3 §C; AM-11; AM-16]` |

> **[D-A/D2 RESOLVED — E1 unclipped; RECLASSIFIED 2026-09-21 as consistent with ch3, not a deviation]**
> E1 runs with `grad_clip_max_norm=None`, which the completed E1 seed-42 run used. The
> `src/training/train_e1.py` hook remains config/CLI-available through `--grad-clip-norm`.
>
> This note originally (2026-07-01) called unclipped E1 an "explicit documented deviation" from a
> clipping "throughout" requirement. On re-reading, ch3 p.104 attaches clipping to **the
> distillation stages**, not to E1. Unclipped E1 is therefore not a deviation, and no E1 rerun follows
> from it `[B59 A5]`.
>
> ~~Whether E2/E3 are clipped remains a **METHODOLOGY DECISION OPEN** (§(f)).~~ See
> `docs/open_questions.md` D2. `[D2; D-A; B59]`
> **[UPDATED 2026-09-23 — B64 C5]** Resolved by AM-7 (M1): no clipping in E1, E2 or E3; a NaN or divergence (as defined in AM-7) in any E2/E3 run stops the stage, after which one clipping rule is adopted for all three and the FP32 stages are rerun. Code: lane L-AM7.

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
| Library | **Hand-written NumPy/PIL transforms** (`src/data/transforms.py`; `configs/augment.py` `"library"`). ch3 §D names Albumentations; E1 seed 42 trained without it. The augmentation *semantics* above are implemented; only the library differs. Manuscript wording to become behaviour-specific `[B52 §9.1; B59 A3]` | `[ch3 §D; project]` |
| Implemented order (train only) | EXIF(image) → aspect-preserving resize of the **unpadded** image to long side 512·r, r~U[0.75, 2.0] → rotation ±10°, p 0.5 → random 512×512 crop, padding (random placement; image fill [124,116,104], mask 255) where the scaled image is smaller → cat_max_ratio 0.95 (≤10 attempts, lowest-dominance fallback) → flips → hue/sat, p 0.5. ch3 §E.2.b says training augmentations are "applied after padding"; E1 instead pads inside the crop. Recorded as a **non-material implementation difference** — scale, rotation-before-crop, crop size, cat_max_ratio, flips and photometric semantics all match `[B59 A2]` | `[project; src/data/transforms.py:166-186]` |
| Class weighting | class-aware √ inverse-frequency (see B5) | `[ch3]` |

### B3 — Distillation
**Logit KD (E2)** `[ch3 §C "E2"]`
| Param | Value | Source |
|---|---|---|
| Loss | CE + KL on temperature-softened outputs | `[ch3]` |
| Temperature `T_Logit` | **4** | `[ch3]` |
| Weight `λ_logit` | **`NEED_TO_CONFIRM`** — selected via validation sweep over **{0.25, 0.5, 1, 2, 4}** at seed 42; reported in Ch4; reused **unchanged** in E3. **LOCKED 2026-09-23 (AM-2):** 80,000 iterations per candidate; the highest VAL all-class mIoU (each candidate's best-checkpoint value) wins; candidates within 0.5 pp of the best are tied → smallest λ; a boundary winner is reported and the grid is not extended; the winner is E2 seed 42 `[B59 C3; AM-2]` | `[ch3]` |
| KL averaging | over valid (non-255) pixels only | `[ch3]` |

**CWD (E3, Shu 2021)** `[ch3 §C "E3"]`
| Param | Value | Source |
|---|---|---|
| Temperature `T_CWD` | **4** | `[ch3]` |
| Feature-map weight `α_CWD` | ~~**50**~~ **[UPDATED 2026-09-24 — B65 CP-006]** selected on VAL from {25, 50, 100} by the AM-16 item-2 sweep; 50 (the Chapter 3 and Shu et al. (2021) default) wins ties and applies if item 2 is cut (stride-16 C5 map). `configs/distill.py` keeps 50 until lane L-AM16-ALPHA adds the override | `[ch3; AM-16]` |
| Logit-map weight `β_CWD` | **3** | `[ch3]` |
| Normalization | **T²/C**, with **C = the channel count of the map being distilled** (Shu et al. 2021, Eq. 4): **C = 320** for the stride-16 feature term (MSCAN-B Stage-3); **C = 116** for the logit-map term. Implemented at `src/training/train_distill.py:239` (`channels_norm=320`) and `:249` (default = map's own 116). ch3 names only the 320 case. | `[ch3; Shu 2021; B59 C1]` |
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

**Measured for E1 (B48 §2; closes D30):** 20.667 GiB peak at batch 16, 512², 116 classes under
`use_deterministic_algorithms(True)` (17.014 GiB without). The determinism reroute of `F.interpolate`
adds a 12.658 GiB forward transient that scales with num_classes × H × W × batch, so every student
stage needs a 48 GB-class card. Whether E2/E3 add materially on top of E1 stays INFERRED until measured
on the pod. *(Was: "E2/E3 therefore cost only marginally more than E1 itself … an INFERRED 11–14 GB
peak on CUDA at batch 16", which omitted that transient.)*
| E3 total loss | ~~`L_CE + L_Dice + λ_logit·L_LogitKD + 50·L_CWD_feat + 3·L_CWD_logit`~~ **[UPDATED 2026-09-24 — B65 CP-006]** `L_CE + L_Dice + λ_logit·L_LogitKD + α_CWD·L_CWD_feat + 3·L_CWD_logit`, α_CWD per AM-16 item 2 | `[ch3; AM-16]` |
| ~~Optional control~~ α_CWD sweep **[UPDATED 2026-09-24 — B65 CP-006]** | ~~α_CWD sensitivity sweep {25, 50, 100} — **not run; α_CWD fixed at 50 per Shu 2021; recorded as future work (AM-11)**~~ **Run (AM-16 item 2):** after λ is fixed, E3 runs at seed 42 with α_CWD in {25, 50, 100} (β and T unchanged), 80,000 iterations each. The highest best-checkpoint VAL all-class mIoU wins. Tie band: the larger of 0.5 pp and √2·s, where s is the sample standard deviation (n = 3) of E1's best-checkpoint VAL all-class mIoU over seeds 42, 43 and 44; s, the band and the three E1 values are recorded in the decision log after B66 and before the sweep launches, and the sweep does not launch before that entry exists. A tie goes to 50 when 50 is tied, otherwise to the smallest α; a winner at 25 or 100 is reported as a boundary result, and the grid is not extended. The winning run is E3 seed 42, and its α is used for E3 seeds 43 and 44 (E6 and E7 inherit it through the E3 checkpoint). All three runs are reported as the Chapter 3 neighborhood-stability check. A pre-registered departure from Chapter 3 p. 98 (α_CWD fixed at 50); TEST is never consulted. If item 2 is cut, E3 runs at α_CWD = 50 and no sweep is reported. Code: lane L-AM16-ALPHA | `[ch3 §C; AM-11; AM-16]` |

### B4 — Quantization
**INT8 QAT (E5 / E6)** `[ch3 §C "E5"/"E6", §D]`
| Param | Value | Source |
|---|---|---|
| Backend / toolchain | **QNNPACK**, eager-mode `torch.ao.quantization` | `[ch3]` |
| Optimizer | SGD, momentum **0.9** | `[ch3]` |
| Learning rate | **3e-4** (3×10⁻⁴), **cosine** decay | `[ch3]` |
| Epochs | **15, fixed; no early stopping** (AM-4; was "~15, early-stop on val mIoU") | `[ch3; AM-4]` |
| Activation quant start | from **step 0**, moving-average range observers | `[ch3]` |
| BN-stat freeze | after **epoch 10** (AM-4; was "~65–70% of training") | `[ch3; AM-4]` |
| Observer freeze | after **epoch 12** (AM-4; was "shortly after BN freeze") | `[ch3; AM-4]` |
| Gradient clipping | global-norm | `[ch3]` |
| Weight EMA | **none** (instantaneous weights quantize better) | `[ch3]` |
| Checkpoint selection | **LOCKED 2026-09-23 (AM-4):** a checkpoint every epoch; after training each is converted (QNNPACK) and scored on VAL on CPU; the epoch with the highest converted VAL all-class mIoU is evaluated (ties → earlier). Code: lane L-AM4 (today `configs/quant.py` / `src/quant/runner.py` select the best fake-quant val mIoU with early stopping). *(Was: METHODOLOGY DECISION OPEN, B59 D5.)* | `[ch3; AM-4]` |
| Supplementary diagnostic (AM-4) | each selected QAT model is also scored with fake quantization disabled (its FP32 weights after QAT), **descriptively** — separates extra fine-tuning from INT8 adaptation; lane L-AM4 | `[AM-4]` |
| Distillation during QAT | **none** (E5/E6 supervised-only) | `[ch3]` |
| Weight quant | **per-channel symmetric INT8**, all conv layers | `[ch3]` |
| Activation quant | **per-tensor asymmetric UINT8** (full 8-bit range for QNNPACK/ARM) | `[ch3]` |
| Fusion | Conv-BN-ReLU via `fuse_modules` before observer insertion | `[ch3]` |
| Hard-Swish / Hardsigmoid | quantized as **standalone** ops | `[ch3]` |
| Source ckpt | E5 ← E1; **E6 ← E3 (CWD head removed)**; E6 uses identical config as E5 | `[ch3]` |
| E6-KD reduced weights | **0.5×** the E3 Logit-KD and CWD weights, T unchanged; triggered on VAL (AM-3) *(was: `NEED_TO_CONFIRM`; trigger METHODOLOGY DECISION OPEN, B59 C4)* | `[ch3; AM-3]` |

**INT8 PTQ (E4 / E7)** `[ch3 §C "E4"/"E7", §E.1]`
| Param | Value | Source |
|---|---|---|
| Method | static INT8 (standard) | `[ch3]` |
| Calibration set | ~~**~128 images**~~ **[UPDATED 2026-09-23 — B64 C5]** exactly **128 images**, one per mini-batch (AM-10), sampled with **seed 42** from training partition; no augmentation; same preprocessing as clean test; identifiers persisted as fixed list; **same subset for E4 and E7**; **one image per mini-batch** (AM-10). **[UPDATED 2026-09-24 — B65 CP-006]** Descriptive sensitivity (AM-16 item 5): E4 and E7 at seed 42 are also calibrated on three further 128-image TRAIN subsets, drawn by the AM-10 procedure with seeds 43, 44 and 45, and reported per subset with the range over all four calibration sets; the official E4/E7 models keep this list. Code: lane L-AM16-CALIB | `[ch3; AM-10; AM-16]` |
| Activation observer | histogram (minimizes quantization error) | `[ch3]` |
| Weight observer | per-channel min/max | `[ch3]` |
| Weight quant | per-channel symmetric INT8, all conv (per-channel depthwise essential) | `[ch3]` |
| Activation quant | per-tensor asymmetric UINT8 | `[ch3]` |
| Source ckpt | E4 ← E1; **E7 ← E3 (CWD head removed)** | `[ch3]` |
| Calibration selection | **fixed configuration (the registered qconfig); no validation-based calibration choice** (AM-10); PTQ results reported on **test** *(was: "configuration selected on validation")* | `[ch3; AM-10]` |

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
- **Seed = 42** across `torch`, `numpy`, python `random` `[ch3]`. **[UPDATED 2026-09-23 — B64 C5]** Seed 42 is the primary seed; seeds 43 and 44 per AM-1 (below).
- Determinism set **before CUDA init**: `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` `[ch3; ctx]`.
- **Seeds — LOCKED 2026-09-23 (AM-1; resolves M8):** seeds **42 (primary), 43, 44** for **E1 and
  E3**; E4/E7 recomputed per seed; E5/E6 once per seed, from their FP32 parent's seed; ~~**E2 at seed 42;
  seeds 43/44 optional, budget permitting, with λ fixed from the seed-42 sweep**~~
  **[UPDATED 2026-09-24 — B65 CP-006]** **E2 seeds 43 and 44 are planned runs, with λ fixed from the
  seed-42 sweep (AM-16 item 1)**; teacher trained once.
  All training randomness, including data order and augmentation, derives from the run seed (B64 C1).
  The Holm family and the E3-vs-E6 non-inferiority check use the seed-42 models only. *(Was: three-seed
  validation planned for E1 and E3 with the extra values `NEED_TO_CONFIRM`; E5/E6 "where compute
  permits"; E2 repetition optional; the obligation a METHODOLOGY DECISION OPEN `[B59 D4]`.)*
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
  *distribution*, the recipe, or any locked hyperparameter. **Measured:** at a fixed
  `num_workers` the *augmentation stream* is byte-identical across two independent processes —
  8-sample / 4-batch train subset, `shuffle=False`, CPU `[B31-5/A1, 2026-09-01]`. **Not
  measured:** no cross-process comparison of a CUDA training run exists, and that measurement
  predates the project's first GPU execution (2026-09-09, B42). Runs are comparable in
  distribution; they are not bitwise-comparable across different `num_workers`. Record the value
  in Ch4 with the seed. Compare ~~`:825-837`~~ §(f), **Corruption dependency pins — REGISTERED and EXECUTABLY VALIDATED (corruption closure only)**, the "Two phases, never collapsed into one Boolean" flag block **[UPDATED 2026-09-23 — B64 C6]**, which scopes and dates its own byte-identity flag.
- **Bitwise identity of training *results* is not claimed, and ch3 does not claim it** `[ch3 §D]`.
  ch3 §D states that "floating-point variation may remain across GPU classes and compiled CUDA
  kernels", and specifies "mean ± SD reported across completed seeds" rather than bitwise
  agreement. A live instance: `nll_loss2d_forward_out_cuda_template` has no deterministic CUDA
  implementation and runs under `warn_only=True` — see
  [B48](../reports/b48_e1_oom_investigation.md) §5. Whether that reaches gradients through the
  weighted mean-reduction's `total_weight`, or only the logged scalar, is `[UNRESOLVED]` pending
  an on-pod gradient comparison.
- **Manuscript gap (for correction outside this repo):** ch3 §D's reproducibility paragraph pins
  seed 42, the cuDNN flags, `use_deterministic_algorithms(True, warn_only=True)` and
  `CUBLAS_WORKSPACE_CONFIG`, but is silent on `num_workers`. As written it is **under-specified**:
  two runs satisfying every stated condition can still differ. `[project]`
- **Resume is NOT bitwise-identical to an uninterrupted run** — documented deviation, same register
  as D-A `[project; B31-2]`. `--resume` restores model / optimizer / scheduler / RNG state
  (`torch`, `torch.cuda`, `numpy`, python `random`, and the DataLoader generator) and the LR curve
  continues **exactly** ~~(checkpoint-recorded LR matches the analytic 80,000-iteration
  `PolynomialLR` curve at full float64 precision)~~ **[UPDATED 2026-09-24 — B66-prep S3]** (the restored scheduler continues the chained `PolynomialLR` recursion over the checkpoint's own horizon, `total_iters`; the seed-42 curve equals that recursion bitwise and differs from the closed form in 79,684 of 80,000 values, B66-prep P1 MEASURED; a `--resume` whose horizon differs from `--iterations` is refused). Data **order** is not recoverable: the loop
  consumes an infinite `cycle(train_loader)` and the position within the current epoch is not
  persisted. A resumed run is therefore a valid E1 run but not a byte-reproduction of an
  uninterrupted one~~, and must be reported as resumed if used for a headline result~~.
  **[UPDATED 2026-09-23 — B64 C5]** Official (headline) runs are never launched or continued with `--resume` — see the no-resume rule below (D29); a non-official resumed run discloses it (D26).
- **No-resume rule for official runs** `[project; B44; closes D29]`: official runs of any stage
  (teacher, E1, E2, E3, E5, E6) are never launched or continued with `--resume`. An interrupted
  official run is discarded and relaunched from iteration 0 with the same seed into a fresh
  `--ckpt-dir`. `--resume` stays available for debugging and rehearsals; a non-official resumed run
  discloses it (D26).
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
| `--iterations` **[UPDATED 2026-09-24 — B66-prep S3]** | default `E1_STUDENT["iterations"]` (**80000**); registered 80000 and 160000 | sets the poly horizon and the real-run length; recorded as run_meta `poly_horizon`. Not an operational default like the rows above: the default is the `[ch3]` value (80,000, B2), and 160000 is the AM-16 item-3 exception |

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
| Albumentations | **not applicable** — no Albumentations dependency; E1 uses NumPy/PIL transforms (`requirements-e1.txt:10-11`) `[B52 §9.1; B59 A3]` | `[ch3; ctx]` |
| statsmodels | **0.14.6** (`requirements.lock:63`; confirmed on the E1 pod by `verify_env`) | `[requirements.lock; open_questions #5]` |
| mmengine | **0.10.7** — teacher/MMSeg runtime (`requirements.lock:30`, `requirements-runpod.lock:96`, `requirements-runpod.in:31`); excluded from `requirements-e1.txt` as teacher-only. ch3's version list omits it `[G19]` | `[requirements.lock; requirements-runpod.lock]` |
| ftfy | **6.3.0** — teacher image only (`requirements-teacher.lock:23`; also `requirements-runpod.in:16`, `requirements-runpod.lock:66`); `mmseg.models` cannot register without it (B55). ch3's version list omits it | `[requirements-teacher.lock; B55]` |
| Quant backend | eager-mode `torch.ao.quantization`, **QNNPACK** | `[ch3; ctx]` |
| Compute | RunPod; **ch3 locks no GPU model** ("the exact hardware used for each stage is reported"). E1 seed 42 ran on an **NVIDIA A40 (48 GB), Secure Cloud** (B52). An **RTX 4090 (24 GB) was measured unable to run E1** under the registered determinism policy: 20.667 GiB peak, OOM at iteration 2 (B48; D30). The teacher GPU is `NEED_TO_CONFIRM` and is chosen only after a measured batch-16 deterministic-policy VRAM reading (G21). Per-stage hardware and cost are reported in Ch4 | `[ch3 §D; B48; B52; B59 A4]` |

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
  **background = index 0 excluded** [confirmed by the official PlantSeg METAINFO; open_questions #2 RESOLVED];
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
> names. Inferential severities **1–3**; severity **4 descriptive-only**; ~~severity **5 excluded**~~.
> **[UPDATED 2026-09-24 — B65 CP-006]** The four non-noise corruptions (motion blur, JPEG compression,
> brightness, fog) are also scored at severities 4 and 5, descriptively (AM-16 item 6); the protocol
> file still lists severity 5 as excluded until lane L-AM16-SEV lands.
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
- ~~Severity **4** = descriptive degradation profile only; severity **5** = **excluded**.~~
  **[UPDATED 2026-09-24 — B65 CP-006]** Severity **4** = descriptive degradation profile only. The four
  non-noise corruptions (motion blur, JPEG compression, brightness, fog) are also scored at severities 4
  and 5 (descriptive), following Kamann & Rother (2020), who average non-noise corruptions over
  severities 1–5 and noise over severities 1–3; reported per severity and as that average (AM-16 item 6).
  The mIoU-C, RPD and rCD definitions (inferential and descriptive) stay on severities 1–3. Code: lane
  L-AM16-SEV.
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
- ~~**E6-KD contingency trigger** (stricter, distinct): run if the **observed** clean E3→E6 mIoU drop
  is **> 1.0 pp**; equality does not trigger. No p-value, not a Holm test.
  **METHODOLOGY DECISION OPEN `[B59 C4]`:** this reads the clean TEST split to decide whether
  to train another model. It must be amended before E6 — validation-based trigger, or a fixed
  pre-registered arm. The rule above is recorded as written, not endorsed.~~
  **[UPDATED 2026-09-23 — B64 C5]** Resolved by AM-3 (M7): E6-KD runs if the E3 → E6 drop in dataset-level VAL all-class mIoU, with E6 scored on the converted INT8 model, is > 1.0 pp at seed 42; the clean-TEST block is descriptive only and never launches a model. Code: lane L-AM3.

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
- ~~No ARM / on-device latency claimed. All runtime/memory results are **CPU PROXY**.~~
  **[UPDATED 2026-09-24 — B65 CP-006]** No on-device latency is claimed (Chapter 3 p. 141 stands). The
  official runtime/memory results are **CPU PROXY** (x86). Supplementary, descriptive (AM-16 item 7): the
  INT8 models E4–E7 (the QNNPACK artifacts of record) and their FP32 parents E1 and E3 are also timed on
  an AWS c6g.xlarge (Graviton2, Arm Neoverse N1; c6g.2xlarge if the x86 thread count exceeds 4),
  on-demand, with the QNNPACK engine for INT8 and this latency protocol at the x86 path's fixed thread
  count, on a recorded aarch64 runtime (torch 2.1.0 and torchvision 0.16.0
  aarch64 CPU wheels with their hashes, OS image, kernel, CPU model and RAM). ARM timings are reported only
  if the ARM INT8 outputs on the first 16 VAL images (sorted by file name) agree with the accuracy outputs
  of record (the QNNPACK-configured models as evaluated on x86 with the QNNPACK engine, never the
  fbgemm/x86 latency copies) on at least 99% of valid (non-255) pixels; the agreement is reported either
  way. ARM and x86 latencies are not compared with each other. This is a server-class ARM measurement, not
  an on-device one. Code: lane L-AM16-ARM.

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
- **[UPDATED 2026-09-24 — B65 CP-006]** The supplementary ARM latency (AM-16 item 7) times the QNNPACK
  artifact for E4–E7 and the model artifact for E1 and E3 on ARM. It is descriptive, is never pooled
  with or compared against the x86 CPU-proxy latency, and leaves this table's x86 rule official.

**INT8 x86 latency copy — how it is obtained** (no second training run):
- **E4/E7 (PTQ):** rebuilt from the same FP32 source checkpoint and the **same frozen 128-image
  seed-42 shared calibration subset**; calibration image identity is never resampled.
- **E5/E6 (QAT):** the QAT runner writes an **auxiliary companion artifact** alongside the official
  converted model — the best **pre-convert QAT state** (`*_qat_state.pt`,
  `quantization = "qat-train-state"`, `artifact_role = preconvert_qat_state`). It is explicitly
  **not** an accuracy, robustness, size or deployment artifact, and does not alter the official
  converted-artifact schema or the AM-4 selection rule. The x86 copy is produced from it
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

> **[AMENDED 2026-09-23 — AM-4]** The schedule and selection rows below follow AM-4
> ([PREREGISTRATION_AMENDMENTS.md](PREREGISTRATION_AMENDMENTS.md)). `configs/quant.py['qat_real_run']` and
> `src/quant/runner.py` keep the pre-amendment values (patience 3, freezes at 0.65/0.70 of steps, best
> fake-quant val mIoU) until lane L-AM4 changes the runner, the config and their smokes together.

| control | value | rationale |
|---|---|---|
| **physical** batch size | **16** | preserves the registered student-training scale rather than adding another E5/E6 difference |
| weight decay | **1e-4** | preserves the registered student regularization scale |
| epochs | **15, fixed** (AM-4; was "max epochs 15") | no early stopping, so the full quantization schedule always runs |
| early-stop patience | **none** (AM-4; was 3 validation checks) | withdrawn with early stopping |
| BN-statistics freeze | **after epoch 10** (AM-4; was 0.65 of planned optimizer steps) | epoch boundary |
| observer freeze | **after epoch 12** (AM-4; was 0.70 of planned optimizer steps) | epoch boundary |
| checkpoint selection | **AM-4** — every epoch saved; each converted (QNNPACK) and scored on VAL on CPU; highest converted VAL all-class mIoU, ties → earlier epoch | selection on the deployed INT8 model |

**Batch semantics.** 16 is a **physical** batch. Gradient accumulation is deliberately **not**
introduced: fake-quant observers and BatchNorm statistics are batch-sensitive, so accumulated
micro-batches are not equivalent to one true batch of 16. The GPU must accommodate the registered
batch; if it genuinely cannot, that is an explicit experiment-design issue, not a silent change of
batch semantics.

**Freeze points [AMENDED — AM-4].** Freezes occur at epoch boundaries (BN statistics after epoch 10,
observers after epoch 12). Fake quantization runs from step 0, and there is no early stopping, so the
full 15-epoch schedule always runs. *(Was: optimizer-step fractions of the planned budget, rounded as
`round(total_iters * pct)` and floored at step 1, with the observer freeze clamped never to precede the
BN freeze, and early-stop patience accruing only after the observer freeze.)* Until lane L-AM4 lands,
`src/quant/runner.py` still implements the pre-amendment fractions and patience.

> **[AMENDED 2026-09-23 — AM-7]** E1, E2 and E3 share one rule: **no clipping** (E1 seed 42 logged no
> gradient norm). A NaN/divergence (as defined in AM-7) in any E2/E3 run stops the stage; one clipping
> rule is then adopted for all three and the FP32 stages are rerun. The `DISTILLATION_GRAD_CLIP_NORM`
> pilot is **withdrawn**; the E5/E6 (QAT) row below is unchanged. The quote and the table that follow
> are the pre-amendment record; the E2/E3 launcher gate changes in lane L-AM7.

> **METHODOLOGY DECISION OPEN `[B59 C2, 2026-09-21]`.** The paragraph below records the
> preregistered reading (PREREGISTRATION U3/U4): clipping is mandatory for E2/E3 and E5/E6, and
> "no clipping" is excluded. That reading is **not settled** for E2/E3. ch3 places clipping in the
> distillation stages, but also states that E1/E2/E3 share an identical recipe that differs only in
> the distillation terms. Clipping only E2/E3 would therefore change the recipe that E1 comparisons
> depend on. Whether E2/E3 are clipped at all — and, if so, the `max_norm` — must be decided
> explicitly before the first E2 run. Nothing here selects an option, and the committed launcher
> gate is unchanged. The QAT half (E5/E6) is tracked separately.

~~*Gradient clipping — TWO SEPARATE, STILL-UNRESOLVED DECISIONS.* Chapter 3 requires global-norm
clipping throughout distillation training **and** during QAT, and the launchers require a positive
finite `max_norm`, so **"no clipping" is not a candidate** for these stages. E1's separately resolved
unclipped status (D2/D-A) is **not** permission to leave the distilled stages unclipped, and E1 is not
reopened here.~~ **[UPDATED 2026-09-23 — B64 C5]** Resolved by AM-7 (M1): no clipping in E1, E2 or E3; a NaN or divergence (as defined in AM-7) in any E2/E3 run stops the stage, after which one clipping rule is adopted for all three and the FP32 stages are rerun. Code: lane L-AM7. The QAT (E5/E6) decision below is unchanged. Distillation and QAT are different optimization regimes and no source establishes that
one numeric norm should serve both, so they are selected independently:

| decision | applies to | candidates | selection |
|---|---|---|---|
| ~~`DISTILLATION_GRAD_CLIP_NORM` (`configs/distill.py`)~~ | ~~**E2, E3** (same value)~~ | ~~{1.0, 5.0}~~ | **WITHDRAWN (AM-7)** — ~~pilot on **E2**, λ_logit held at **1.0**, 8,000 iters (10% of official) with validation every 1,000~~ |
| `QAT_GRAD_CLIP_NORM` (`configs/quant.py`) | **E5, E6** (same value) | {1.0, 5.0} | pilot on **E5** from the official E1 FP32 checkpoint, 5 epochs (official max stays 15); once L-AM4 lands, the pilot compares candidates on converted-model VAL mIoU (AM-4) |

Both: seed **42**, **train + validation only, TEST prohibited**, labelled
**HYPERPARAMETER SELECTION / PILOT**, excluded from test evaluation, robustness, hypothesis testing
and the statistics family, and frozen before the corresponding official runs. Selection rule for each:
reject a candidate whose training becomes non-finite or unstable; otherwise take the higher
dataset-level validation mIoU under the identical pilot budget; on a tie within the repository's
existing validation tie/noise rule prefer **5.0** as the less intrusive threshold. No new significance
test is invented. The QAT decision never inherits the distillation result.

**Decision order for E2/E3:** ~~select `DISTILLATION_GRAD_CLIP_NORM` → freeze it →~~ run the existing
λ_logit sweep → freeze λ_logit → official E2/E3 runs. **[AM-7]** The clipping pilot is withdrawn;
**[AM-2]** fixes the sweep's budget and tie rule. ~~Fixing λ at the grid centre during the clipping
pilot avoids a 2 × 5 Cartesian search while staying inside the registered grid.~~
**[UPDATED 2026-09-24 — B65 CP-006]** For E3 (AM-16 item 2): after λ is fixed and the α tie band is
recorded in the decision log (after B66), the α_CWD sweep {25, 50, 100} runs at seed 42; the winning run is
E3 seed 42, and its α is used for E3 seeds 43 and 44. If item 2 is cut, E3 runs at α_CWD = 50.

`λ_logit` is unchanged — {0.25, 0.5, 1, 2, 4}, seed 42, validation-only, boundary winner reported
rather than extending the grid. It is not decided here. **[AM-2]** Budget 80,000 iterations per
candidate; the highest VAL all-class mIoU (each candidate's best-checkpoint value) wins; candidates
within 0.5 pp of the best are tied → smallest λ; the winner is E2 seed 42.

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
dependency-version blocker. **[UPDATED 2026-09-24 — B65 CP-006]** AM-16 item 6 adds 6,244 severity-5
items for the four non-noise corruptions (1,561 × 4; 37,464 in all) once lane L-AM16-SEV lands; until
then the generator and cache refuse severity 5, and severity-5 reference equivalence is not validated.

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
2. ~~**Disease-only / background convention** — empirically background = **index 0** (exclude for disease-only
   mIoU; diseases 1–115), but no literal "background" category is named in the dataset files (COCO
   `categories` empty), so `reduce_zero_label` and the formal disease-only exclusion stay **NEED_TO_CONFIRM**
   pending the PlantSeg repo's official convention.~~ **RESOLVED (D1 + A0-FIX 2026-07-26)** — index 0 is the
   non-disease slot and 1–115 are the diseases (official PlantSeg METAINFO); see open_questions #2.

**NEED_TO_CONFIRM (not stated in any source; filled by experiment/selection, never guessed):**
- `λ_logit` — validation sweep over {0.25, 0.5, 1, 2, 4}, reported Ch4. Budget and tie rule fixed by AM-2.
- ~~E6-KD reduced distillation weights (the trigger itself is a METHODOLOGY DECISION OPEN — B59 C4).~~ **Resolved by AM-3:** 0.5× the E3 weights; VAL trigger.
- ~~Albumentations version; statsmodels version.~~ Resolved: statsmodels **0.14.6**; Albumentations
  **not applicable** (no dependency) — see §(d) B6 `[B52 §9; B59 A3]`.
- ~~Multi-seed values beyond seed 42 (for E1/E3 three-seed runs); whether the extra seeds are
  obligatory is itself a METHODOLOGY DECISION OPEN (B59 D4).~~ **Resolved by AM-1:** seeds 43 and 44.
- Final RunPod pod type / GPU / CPU model / CUDA image (reported Ch4). E1 seed 42 = A40 Secure (B52);
  teacher GPU pending a measured batch-16 VRAM reading (G21).
- **METHODOLOGY DECISIONS registered by B59 (2026-09-21):**
  - **LOCKED by B60 (2026-09-22); runtime implemented by B62 (committed `3c43f89`; frozen, runbook §4a; CUDA re-canary pending):**
    - teacher augmentation (M2);
    - teacher train/eval scaling (M3);
    - teacher acceptance band / readiness and schedule (M5);
    - TEST filenames in the teacher preflight (M11), replaced by TRAIN/VAL-only data roots.
  - **LOCKED by B61 (2026-09-22); runtime implemented by B62 (committed `3c43f89`; frozen, runbook §4a; CUDA re-canary pending):**
    - NMF/Hamburger RNG control (M4): `rand_init=True`, isolated streams M4-T / M4-V / M4-KD;
    - teacher checkpoint selection under M4 (M12);
    - teacher CE ignore normalisation, `avg_non_ignore=True` (M13, new).
  - **LOCKED by B64 amendments (2026-09-23; [PREREGISTRATION_AMENDMENTS.md](PREREGISTRATION_AMENDMENTS.md)):**
    - E1/E2/E3 clipping — one rule, no clipping (M1, AM-7);
    - λ sweep run length and tie band — 80,000 iterations per candidate, 0.5 pp tie band (M6, AM-2);
    - E6-KD trigger and weights — VAL trigger on converted E6, 0.5× weights (M7, AM-3);
    - seeds — 42, 43, 44 (M8, AM-1);
    - QAT schedule and checkpoint rule — 15 fixed epochs, converted-VAL selection (M9, AM-4);
    - TEST images with zero disease pixels — excluded from per-image analyses, counted (M10, AM-5).
  - **Still OPEN:** none.

  Detail: `reports/b59_pre_runpod_reconciliation.md`; `reports/b61_teacher_nmf_checkpoint_selection_lock.md`;
  `docs/open_questions.md`.
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
| **RunPod GPU/pod availability** | **before real E1 execution** | verified on the pod (`scripts/verify_env.py` → PASS, `train_e1.py --dry-run` → PASS) **[UPDATED 2026-09-24 — B66-prep S2, DL-21]** seed 42: `scripts/preflight_e1.py` → GO. TRAIN/VAL-only pods from B66: `scripts/preflight_e1_trainval.py gate` → GO; verify_env.py and preflight_e1.py are not run there. |
| **SegNeXt-B / MMSeg checkpoint availability** | **before teacher-dependent execution** (teacher fine-tune, E2/E3 prep) | `docs/B8_checkpoint.md` — none publicly released; in-house fine-tune planned |
| **QNNPACK INT8 operator support** | **before the quantization stages (E4–E7)** | `docs/b7_result.md` — onednn proxy `PASS_CLEAN`; authoritative run env-gated |

**QNNPACK is a pre-quantization gate and is NOT an independent blocker to FP32 E1.**

**No training-pilot requirement.** Chapter III specifies **no** short-training-run pilot and **no**
iteration count for this phase. The "2,000-iteration E1 pilot" figure appears **only** in
`docs/reference/context.md` `[ctx]`; it is operational, is **not** part of the locked methodology, and does
**not** modify the **80,000-iteration / validate-every-4,000** recipe in §(d) B2. An optional short training
rehearsal is tracked as `[operational] RECOMMENDED_NOT_BLOCKING` — see
[open_questions.md](open_questions.md) **D6** and
~~[reports/e1_runpod_launch_runbook.md](../reports/e1_runpod_launch_runbook.md)~~ [reports/e1_launch_runbook_v2.md](../reports/e1_launch_runbook_v2.md) **[UPDATED 2026-09-23 — B64 C5]** (v1 superseded).
