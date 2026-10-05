# Evaluation Contract — metric semantics + result-artifact schema (A0)

> **Status: FROZEN (2026-07-26).** This is the authoritative definition of *how every evaluation number in
> this thesis is computed* and *what every evaluation run must emit*. A1 (metric tests) and A2 (evaluator)
> implement this document and must not invent additional metric or schema decisions. **[UPDATED 2026-09-24 — B66-prep L-EVAL-DET, DL-17]** §10 adds one optional, separately versioned block (`run.eval_runtime`) and the evaluation determinism rule; `schema_version` is unchanged. **[UPDATED 2026-09-28 — L-AM13]** §11 adds the descriptive upstream evaluation protocol (`upstream/1.0.0`) and artifact layout `plantseg-eval-artifact/1.2.0`; `schema_version`, `metric_protocol` and the canvas protocol are unchanged. **[UPDATED 2026-10-02 — L-CKPT-GUARD]** §7.4 adds the exploratory arms A, F and G as descriptive FP32 stages, the declared checkpoint stage and the teacher checkpoint's identity; `schema_version` and `artifact_schema_version` (`plantseg-eval-artifact/1.2.0`) are unchanged.
>
> Companion to [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §(f) (which states *which* metrics
> the thesis reports) and [open_questions.md](open_questions.md) D3/D4 (the decision records).
> Source hierarchy unchanged: `[ch3]` > `[ctx]` > `[empirical]` overrides both on dataset facts.

**Protocol identifiers** (embedded in every artifact):
`schema_version = "plantseg-eval/1.0.0"` · `metric_protocol = "plantseg-metrics/1.0.0"`

---

## 0. Verified empirical basis

Every number below is `[empirical]`, established by the committed B16 audit set — not by ch3, not by
`context.md`, and not by ratio arithmetic.

| Fact | Value | Source |
|---|---|---|
| Split counts | train **5,367** · val **846** · test **1,561** (total 7,774) **[AM-18 item 8, 2026-10-01]** "Zero overlap" is over image identifiers. Issue tqwei05/PlantSeg #11 reports byte-identical images across splits (70 test ↔ train, 32 train ↔ val and 11 test ↔ val groups); AM-18 items 1(d) and 8 measure and disclose them. | `dataset_report.md` §"Split mechanism" (folder == JSON == CSV, pairwise overlap 0); `dataset_audit_summary.md` §1 |
| `num_classes` | **116** (background 0 + diseases 1–115) | `dataset_report.md` §"num_classes critical conflict"; `configs/data.py` `num_classes` |
| Background index | **0** | `dataset_report.md` §"Background / disease determination" |
| Ignore index | **255** — pad/rotation-fill only; **absent from all released masks** | `dataset_report.md` §"Mask label analysis"; `test_mask_value_audit.md` §2; `trainval_mask_value_audit.md` §1/§2 |
| Mask encoding | `mask value = COCO category_id + 1`, **100 % exact across all 7,774 images** | `trainval_mask_value_audit.md` §0/§4 |
| **One disease class per image** | **holds dataset-wide** — `distinct_category_ids_per_image = {1: N}` for every split | `trainval_mask_value_audit.md` §1/§2/§3; `test_mask_value_audit.md` §3 |
| Background present per image | test **1,561/1,561** · val **846/846** · train **5,366/5,367** | `trainval_mask_value_audit.md` §4 |
| Class absent from **val** GT | class 68 → **mask value 69** | `trainval_mask_value_audit.md` §2/§3 |
| Class absent from **test** GT | class 41 → **mask value 42** | `test_mask_value_audit.md` §1; `trainval_mask_value_audit.md` §3 |
| FLAG-F image | `apple_black_rot_google_0001` has 0 COCO polygons but a **valid mask `{0, 1}`** | `test_mask_value_audit.md` §4 |
| Stem uniqueness | duplicate stems **0/0**; zero cross-split overlap **[AM-18 item 8, 2026-10-01]** "Zero overlap" is over image identifiers. Issue tqwei05/PlantSeg #11 reports byte-identical images across splits (70 test ↔ train, 32 train ↔ val and 11 test ↔ val groups); AM-18 items 1(d) and 8 measure and disclose them. | `dataset_report.md` §"Counts & identifiers" |

> **Evidence scope `[B59 A12, 2026-09-21]`.** The "one disease class per image" and "background
> present" rows above are **raw-mask** audits at native resolution. The per-image metrics in §3.2
> are computed on the **final 512×512 canvas** (`core_preprocess`: nearest-neighbour resize, long
> side 512, then pad 255). Nearest-neighbour downsampling could in principle remove a tiny lesion.
>
> - **VAL — verified on the final canvas, 2026-09-21:**
>   - 846/846 images through `core_preprocess`: **0/846** canvases with zero disease pixels.
>   - Smallest canvas lesion 259 px; 3 images under 500 px.
>   - Exactly one disease class per canvas, unchanged from the raw mask; background present 846/846.
>   - Scratch instrument; results embedded in `reports/b59_pre_runpod_reconciliation.md`.
> - **TEST — NOT verified on the canvas.** TEST was deliberately not accessed.
>
> ~~The handling of a TEST image whose final canvas holds zero disease pixels is a **METHODOLOGY
> DECISION OPEN**. Candidates are the current abort rule (§3.3), or a pre-registered exclusion using
> an identical eligibility set across models, reporting *k* excluded and *n*_eff. It must be
> registered before the single test campaign. Until amended, **§3.3 stands unchanged**.~~
> **[UPDATED 2026-09-23 — B64 C5]** Resolved by AM-5 (M10): TEST images with zero disease pixels are excluded from per-image disease-only analyses, including per-image mIoU-C; their count is reported; they stay in dataset-level metrics; the expected per-image disease-only count becomes 1,561 minus that count; any other `null` still aborts. Code: lane L-AM5.

### The `1,554` value is NOT a count — never use it

`context.md:107` says "official 70/10/20 (~1,554 test)". That is **7,774 × 0.20 = 1,554.8**, i.e. nominal
ratio arithmetic. `dataset_report.md` §"Inconsistencies" lists all three as *contract approx* values
(5,442 / 778 / 1,554) against the empirical 5,367 / 846 / 1,561, and `dataset_report.json` records the
resolution verbatim: *"empirical split is authoritative"*. The realised ratio is 69.04 / 10.88 / 20.08.

**The authoritative test count is 1,561.** Any document, prompt, or code path stating ≈1,554 is stale.

---

## 1. Notation

All metrics derive from one accumulated **confusion matrix** `CM ∈ ℤ^{116×116}`, rows = ground truth,
columns = prediction, built over **non-ignore pixels only**.

For class `c`:

```
TP_c   = CM[c, c]                       (intersection)
GT_c   = Σ_j CM[c, j]                   (ground-truth support;  MMSeg area_label)
PR_c   = Σ_i CM[i, c]                   (prediction support;    MMSeg area_pred_label)
UN_c   = GT_c + PR_c − TP_c             (union)
```

Ignore handling is **masking on the ground-truth label only**, before accumulation:
`valid = (target != 255)`; both `pred` and `target` are subset by `valid`. Predictions on ignore pixels are
discarded and never counted as false positives. This matches MMSeg exactly.

Two class-eligibility rules are used, and they are **deliberately different**:

```
GT-present     E_gt(S)    = { c ∈ S : GT_c > 0 }
Union-present  E_union(S) = { c ∈ S : UN_c > 0 }
```

where `S = {0..115}` (all-class) or `S = {1..115}` (disease-only).

---

## 2. Benchmark-convention evidence (why union-present at dataset level)

`IMPLEMENTATION_CONTRACT.md` §(f) requires dataset-level all-class mIoU to **"match PlantSeg benchmark
reporting"**. Four distinct sources establish what that means. They are kept separate on purpose — the
benchmark's *configuration* and the evaluator's *implementation* are different authorities, and the
thesis's own version pin is neither.

### 2.1 Authority chain

| # | Authority | Source | What it establishes |
|---|---|---|---|
| 1 | **Official PlantSeg benchmark configuration** | `tqwei05/PlantSeg` → `configs/_base_/datasets/plantseg115.py` | `reduce_zero_label=False` (in `LoadAnnotations` for train *and* test, and in both dataloader dataset dicts); `val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])`; **`nan_to_num` is not configured**; `test_dataloader = val_dataloader`, pointing at the official test images |
| 2 | **Official PlantSeg dataset metadata** | `tqwei05/PlantSeg` → `mmseg/datasets/plantseg115.py` | `METAINFO['classes'] = ('', <115 disease names>)` — index **0** is the non-disease slot (named with the **empty string**), indices **1–115** are the diseases; palette sliced `[:116]`; no zero-label remap |
| 3 | **Official PlantSeg model configuration** | `tqwei05/PlantSeg` → `configs/segnext/segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py` | `decode_head.num_classes = **116**` |
| 4 | **Shared evaluator implementation** | MMSegmentation → `mmseg/evaluation/metrics/iou_metric.py` | the arithmetic executed by `IoUMetric` (quoted in §2.2) |

**Version qualification — do not overstate this.** The PlantSeg README says only *"Follow MMSegmentation to
build the environment"* and **pins no MMSegmentation version**. Therefore:

- **PlantSeg does NOT pin MMSegmentation v1.2.2.** v1.2.2 must never be described as the benchmark's
  governing version.
- v1.2.2 is the **thesis's own** implementation pin (`requirements.lock`, contract §(d) B1, for the
  teacher/MMSeg workflow) and may be identified as such where relevant.
- The metric denominator behaviour is established by **authority 1 (the official evaluator configuration)
  together with authority 4 (the applicable `IoUMetric` implementation)** — not by any single version tag.
- This document **freezes that verified behaviour** as the thesis convention. Nothing here is justified by
  citing this document; every rule traces to authorities 1–4 or to `[ch3]`/`[empirical]`.

### 2.2 The arithmetic (authority 4, quoted verbatim)

```python
area_union = area_pred_label + area_label - area_intersect
iou  = total_area_intersect / total_area_union
acc  = total_area_intersect / total_area_label
dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
ret_metrics_summary = OrderedDict({
    ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2) ...})
```

### 2.3 `nan_to_num` is NOT configured — this is the decisive finding

`IoUMetric` accepts a `nan_to_num` argument; when set, undefined per-class values are replaced by that
constant **before** averaging, which would change absent-class treatment entirely. **The official PlantSeg
evaluator config does not set it** (authority 1), so it defaults to unset, no replacement occurs, and the
reduction is genuinely NaN-aware (`np.nanmean`).

This must stay visible in the decision record: had `nan_to_num` been configured, the union-present
conclusion below would not follow.

Because `intersect = pred_label[pred_label == label]`, a class with **no** ground truth can never
accumulate intersection. Combined with §2.3, the edge-case behaviour is therefore determined, not assumed:

| Case | `TP_c` | `UN_c` | IoU | Acc |
|---|---|---|---|---|
| `GT_c > 0`, `PR_c > 0` | ≥0 | >0 | defined | defined |
| **`GT_c = 0`, `PR_c > 0`** (false-positive-only) | 0 | `PR_c` | `0/PR_c` = **0.0 — included in mIoU** | `0/0` → **NaN — omitted from mAcc** |
| `GT_c > 0`, `PR_c = 0` | 0 | `GT_c` | 0.0 — included | 0.0 — included |
| `GT_c = 0`, `PR_c = 0` | 0 | 0 | `0/0` → **NaN — omitted** | NaN — omitted |

**Consequence:** the official evaluator's `mIoU` is **union-present** and its `mAcc` is **GT-present**. That
asymmetry is real benchmark behaviour and is adopted deliberately. Do not "harmonise" it.

### 2.4 Scope limit — metric arithmetic matches; preprocessing does NOT

The official benchmark test pipeline uses `Resize(scale=(2048, 512), keep_ratio=True)` and **no
pad-to-512×512**. The thesis pipeline resizes long-side→512 and **symmetrically pads to 512²** with image
pad `[124,116,104]` and mask pad `255` (`configs/data.py`, contract §(e)). These are different
preprocessing protocols.

Therefore:
- the **metric formula** in §3.1 is aligned with the official PlantSeg evaluator, **and**
- **identical metric arithmetic does not establish absolute numerical comparability** with published
  PlantSeg scores. Wei et al.'s numbers — including SegNeXt-B **42.05 %** — remain **contextual references
  only**, exactly as `[ch3 §A]` already requires, and must never be used as inferential comparators.

The thesis preprocessing protocol is **not** changed by this document.

> **[UPDATED 2026-09-28 — L-AM13]** For the descriptive scores of AM-13, AM-16 item 4 and AM-17 item
> 1(e), the upstream test geometry is available as the `upstream` protocol (§11), and that score, not
> the 512-canvas score, is the one compared with 42.05% (AM-13). The canvas protocol, and everything
> this section says about it, is unchanged.

---

## 3. DECISION D3 — frozen metric semantics

### 3.1 Dataset-level metrics (accumulate ONE `CM`, compute once)

| Metric | Classes `S` | Eligibility | Formula (per eligible `c`, then unweighted mean) | Status |
|---|---|---|---|---|
| **all-class mIoU** | `{0..115}` | **union-present** | `TP_c / UN_c` | **HEADLINE / official** — benchmark-aligned formula |
| all-class mIoU, GT-present **sensitivity** | `{0..115}` | **GT-present** (classes present in TEST ground truth) | `TP_c / UN_c` | **descriptive sensitivity only (AM-6)** — never headline; lane L-AM6 |
| **all-class mAcc** | `{0..115}` | **GT-present** | `TP_c / GT_c` | descriptive — benchmark-aligned formula |
| **all-class macro Dice (DSC)** | `{0..115}` | **union-present** (`GT_c + PR_c > 0` ⟺ `UN_c > 0`) | `2·TP_c / (GT_c + PR_c)` | descriptive — **thesis-internal extension, NOT a benchmark metric** |
| **disease-only mIoU** | `{1..115}` | **union-present** | `TP_c / UN_c` | secondary, descriptive |
| **disease-only macro Dice** | `{1..115}` | union-present | `2·TP_c / (GT_c + PR_c)` | optional, thesis-internal |
| **disease-only mAcc** | `{1..115}` | GT-present | `TP_c / GT_c` | optional |
| **aAcc** (micro pixel accuracy) | `{0..115}` | n/a | `Σ_c TP_c / Σ_c GT_c` | **diagnostic only — never a thesis metric** |

- **Candidate A (GT-present) is rejected at dataset level.** It omits false-positive-only classes from the
  denominator, so a model that hallucinates absent classes is not penalised, and the number would not match
  PlantSeg benchmark reporting as §(f) requires.
- **Candidate C (fixed 116-class macro with an imputed value) is rejected.** It makes the headline number
  depend on an arbitrary constant for classes absent from both GT and prediction, and no source requires it.
  Note this is precisely what configuring `nan_to_num` would do — and §2.3 confirms the benchmark does not.
- **Macro Dice is a project-defined descriptive metric, not a benchmark metric.** The official evaluator
  requests `iou_metrics=['mIoU']` only (§2.1 authority 1), so it computes **mIoU, mAcc and aAcc** and
  **never computes `mDice`**. Dice is retained here as a thesis-internal descriptive extension and is
  assigned the **same union-present eligibility rule as dataset-level mIoU purely for internal
  consistency**. It is **not** evidence of comparability with published PlantSeg results, and no claim that
  PlantSeg reports or validates Dice may be made anywhere in this project.
- **Disease-only metrics are NOT provisional.** The class-index mapping they depend on is now confirmed by
  the official source: index 0 is the non-disease slot and 1–115 are the diseases (§2.1 authority 2), with
  `num_classes = 116` (authority 3) and `reduce_zero_label = False` (authority 1). The only residual is
  cosmetic — the official METAINFO names index 0 with the **empty string** rather than the literal token
  `"background"`. See `open_questions.md` #2.
- **"macro per-class Dice (DSC)"** is the mandated term. Never write bare "Dice" — it must never be confused
  with binary foreground Dice.

**Practical note (test split):** class 42 has zero test GT. If a model predicts it anywhere, it contributes
`IoU = 0` to the union-present mean; if no model predicts it, it is dropped. **The eligible-class count is
therefore model-dependent and MUST be recorded per run** (`n_eligible_*` fields, §5.3).
**[AM-6, 2026-09-23]** The headline stays union-present; the GT-present sensitivity row above is reported
beside it, descriptively ([PREREGISTRATION_AMENDMENTS.md](PREREGISTRATION_AMENDMENTS.md); resolves G7).

### 3.2 Per-image metrics (one `CM_i` per image)

| Metric | Classes `S` | Eligibility | Status |
|---|---|---|---|
| **per-image disease-only mIoU** | `{1..115}` | **GT-present** | **PRIMARY INFERENTIAL UNIT** |
| **per-image all-class mIoU** | `{0..115}` | **GT-present** | secondary / retained |

**GT-present per-image is preregistered, not chosen here.** `IMPLEMENTATION_CONTRACT.md` §(f) specifies
*"per-image absent-class exclusion"* and `context.md:118` repeats *"Absent classes excluded per-image"*.
A0 preserves it and records the two consequences below rather than silently upgrading the protocol.

> **CONSEQUENCE 1 — the per-image disease-only unit is a single-class IoU.**
> Because one-disease-per-image holds **dataset-wide** (`trainval_mask_value_audit.md` §3), `E_gt({1..115})`
> contains **exactly one** class for every image. Per-image disease-only mIoU is therefore *identically*
> the IoU of that image's single true disease class — the "mean over classes" is a mean over one element.
> Per-image all-class mIoU is likewise the mean of exactly **two** IoUs (background + that disease class)
> for every val/test image, since background is present in 846/846 val and 1,561/1,561 test masks.

> **CONSEQUENCE 2 — wrong-class predictions are penalised asymmetrically.**
> If the model labels lesion pixels with the *wrong* disease `d ≠ c`, those pixels become false negatives
> for `c` (lowering `IoU_c`, and hence the score), but `d` contributes **no** additional `IoU = 0` term
> because `d ∉ E_gt`. A union-present per-image rule would add that term and score substantially harsher.
> This is a *deliberate, preregistered* property of the inferential unit.

**Mitigation (mandatory, not optional):** every run persists per-image classwise sufficient statistics
(§5.4). Union-present per-image variants — and any other per-image aggregation — are therefore fully
recomputable **without re-running inference on any stage**. This is the hedge that makes preserving the
preregistered rule safe.

> **[UPDATED 2026-09-28 — L-AM5]** AM-5 fields (the §3.3 AM-5 block; lane L-AM5). Every
> `per_image.jsonl` row gains `am5_excluded: bool`, appended as the last key: `true` iff
> `n_eligible_disease_only == 0`, i.e. the evaluated mask holds no disease pixel. It depends on ground
> truth only, so it is identical across models. No existing value changes: such a row keeps its `null`
> + `undefined_no_eligible_class` disease-only value. `summary.json` gains
> `am5 = {rule: "no_disease_gt", excluded_count, included_count, excluded_ids_sha256}`, where the hash is
> the SHA-256 of the sorted excluded `image_id`s joined by `\n` (UTF-8, no trailing newline), and
> `artifact_schema_version = "plantseg-eval-artifact/1.1.0"`. `schema_version` and `config_sha256` are
> unchanged, so a pre-lane artifact (neither field; layout 1.0.0) stays pairable with a lane artifact:
> readers derive the flag from `n_eligible_disease_only == 0` and tolerate the extra fields. The writer
> refuses an `am5` block or a row flag that disagrees with the rows. The statistics ingest applies the
> exclusion (STATISTICAL_ANALYSIS_CONTRACT §10), and `scripts/evaluate_model.py` prints the count with
> every artifact.
>
> **[UPDATED 2026-09-28 — L-AM13]** The writer now records `plantseg-eval-artifact/1.2.0`, which keeps
> these fields unchanged (§11).

### 3.3 Undefined cases

| Situation | Result | Expected on val/test? |
|---|---|---|
| `E_gt` empty for the requested class set | `null` + `status = "undefined_no_eligible_class"` | **NO — never** |
| Row excluded by an approved data-integrity record | `null` + `status = "excluded_data_integrity"` | **NO — none preregistered** |
| Inference/metric raised | `null` + `status = "evaluation_error"` | **NO — fail the run** |
| Normal | numeric + `status = "ok"` | **YES — all rows** |

~~**Expected defined-score counts on the official test split: 1,561 / 1,561 for BOTH per-image vectors.**
Every test mask contains background *and* exactly one disease class, so neither vector can be undefined.
Any `null` on test is a **data-integrity failure that must abort the run**, not a row to skip quietly.~~
**[UPDATED 2026-09-23 — B64 C5]** Resolved by AM-5 (M10): TEST images with zero disease pixels are excluded from per-image disease-only analyses, including per-image mIoU-C; their count is reported; they stay in dataset-level metrics; the expected per-image disease-only count becomes 1,561 minus that count; any other `null` still aborts. Code: lane L-AM5.

> *Scope note `[B59 A12]`:* the "every test mask contains … exactly one disease class" premise is a
> raw-mask fact. Survival on the final 512 canvas is verified for VAL only (0/846 lost). ~~The rule above
> is unchanged. Whether to replace it with a pre-registered exclusion rule is a **METHODOLOGY DECISION
> OPEN** (see §0 evidence-scope note).~~
> **[UPDATED 2026-09-23 — B64 C5]** Replaced by AM-5 (M10); see the AM-5 block below. Code: lane L-AM5.

> **[AMENDED 2026-09-23 — AM-5; resolves M10]** The one permitted exception is this. A TEST image whose
> evaluated mask has **no disease pixels** gets a `null` + `undefined_no_eligible_class` per-image
> **disease-only** value. That row is excluded from per-image disease-only analyses (paired tests,
> per-image effect sizes). The exclusion depends on ground truth only, so it is identical across models.
> Such rows are counted and reported, and they stay in every dataset-level metric. Every other `null` on
> TEST still aborts. Per-image mIoU-C is the mean of the image's per-image disease-class mIoU over its
> 15 corrupted variants (five corruptions × severities 1–3), and the same exclusion applies. That mean
> equals the nested severities-then-corruptions mean of STATISTICAL_ANALYSIS_CONTRACT §2. Code: lane L-AM5.

---

## 4. DECISION D3b — statistical input + resampling

> **The inferential protocol built on these inputs is frozen in
> [STATISTICAL_ANALYSIS_CONTRACT.md](STATISTICAL_ANALYSIS_CONTRACT.md) (A3-0, 2026-07-28)** — the
> eight-comparison family, tie/degenerate policy, effect-size definitions, BCa seed and acceleration
> rules, and the statistics result artifact. **Nothing in this section changes**: the metric
> definitions, artifact field names, class-eligibility rules, dataset identity and evaluator
> behaviour frozen in §§3, 5–7 remain exactly as they are.

### 4.1 Inferential unit and pairing
- The **8 Holm-corrected paired tests** consume the **per-image disease-only mIoU** vector.
- Pairing is by **canonical image ID** (§6), never by position. Both vectors must have identical ID
  *sets* and be sorted into identical ID *order* before differencing; a mismatch is a hard error.

### 4.2 Non-inferiority bootstrap — the resampling unit is the IMAGE, and `CM` is re-accumulated

`IMPLEMENTATION_CONTRACT.md` §(f) requires a **paired BCa CI on `ΔmIoU = mIoU(E6) − mIoU(E3)`** where the
named basis is **dataset-level all-class mIoU**. A dataset-level metric is a single aggregate, so the
resampling unit and recomputation rule must be stated explicitly:

```
For each of B = 10,000 bootstrap replicates:
  1. Resample image IDs with replacement (n = 1,561), ONE shared index vector applied to BOTH stages
     (paired / matched resampling — the same images for E3 and E6 in every replicate).
  2. For each stage: CM* = Σ over the resampled image multiset of that image's per-class
     sufficient statistics (TP_c, GT_c, PR_c), duplicates counted with multiplicity.
  3. Recompute dataset-level all-class mIoU from CM* using the §3.1 union-present rule.
  4. Δ* = mIoU*(E6) − mIoU*(E3).
BCa interval over the B values of Δ*; non-inferiority passes iff lower bound > −2.0 pp.
```

**Why per-image scalar averaging is NOT sufficient.** Dataset-level mIoU is a *ratio of sums*
(`Σ TP_c / Σ UN_c`, then macro over classes); the mean of per-image scalar mIoUs is a *mean of ratios*.
These are not equal and do not converge to each other. Bootstrapping per-image scalars and averaging them
would produce a CI for a **different estimand** than the one §(f) names. Re-accumulating classwise
sufficient statistics reproduces the intended dataset-level metric exactly.

**This is the binding architectural requirement of A0:** per-image scalars alone are insufficient, so
per-image **classwise sufficient statistics are mandatory** in every artifact (§5.4).

---

## 5. DECISION D4 — frozen artifact contract

### 5.1 Layout (chosen: hybrid; Options "one monolithic JSON" and "summary + CSV only" are rejected)

Every evaluation run writes **one directory** containing exactly:

| File | Purpose | Format |
|---|---|---|
| `summary.json` | run + dataset identity, dataset-level metrics, per-class arrays, integrity block | strict JSON (UTF-8) |
| `per_image.jsonl` | one object per image: scalars + status | JSON Lines, one row per line |
| `sufficient_stats.npz` | **sparse** per-image classwise stats + dataset-level per-class arrays | `numpy.savez_compressed`, int64 |
| `MANIFEST.sha256` | SHA-256 of the three files above | text |

Rationale: strict JSON gives schema evolution and human inspection but cannot hold large numeric arrays
losslessly or compactly; JSONL streams to corruption scale (5 corruptions × 3 severities × 1,561 ≈ 23k rows
per stage) and diffs deterministically; NPZ stores exact int64 counts with no float rounding. **NumPy is
already a declared dependency** (`requirements-e1.txt`) — Parquet/pyarrow is rejected as a new dependency.

**Sparsity:** only classes with `TP_c > 0 ∨ GT_c > 0 ∨ PR_c > 0` are stored per image. With
one-disease-per-image, GT support is ≤2 classes per image, so the artifact stays small.

### 5.1.1 `condition` semantics `[project-decision, A3b-0 2026-07-28]`

The **type** of `condition.name` is unchanged — it remains a string. A3b-0 adds only a *semantic*
restriction on the values an **official** artifact may carry. No metric rule and no existing
artifact field name changes.

| Artifact | `condition.type` | `condition.name` | `condition.severity` |
|---|---|---|---|
| clean | `"clean"` | **`null`** | **`null`** |
| official inferential corruption | `"corruption"` | one of the **five exact IDs** below | **1, 2 or 3** |
| descriptive-only corruption | `"corruption"` | one of the five exact IDs | **4** |
| ~~—~~ | ~~—~~ | ~~—~~ | ~~**5 must never be produced** by the official study pipeline~~ |
| descriptive-only corruption (AM-16 item 6) **[UPDATED 2026-09-24 — B65 CP-006]** | `"corruption"` | `motion_blur`, `jpeg_compression`, `brightness` or `fog` | **5** (never for `gaussian_noise`) |

**[UPDATED 2026-09-24 — B65 CP-006]** Severity 5 of the four non-noise corruptions is descriptive (AM-16
item 6) and never enters the inferential grid; `configs/corruption_protocol.json` and the evaluator still
refuse severity 5 until lane L-AM16-SEV lands.

Five exact IDs, frozen in `configs/corruption_protocol.json`:
`motion_blur` · `gaussian_noise` · `jpeg_compression` · **`brightness`** · `fog`.
Aliases such as `brightness_variation` or `motion-blur` are rejected;
"brightness variation" is a display label only. Full rationale:
[STATISTICAL_ANALYSIS_CONTRACT.md](STATISTICAL_ANALYSIS_CONTRACT.md) §2.1.

`NONOFFICIAL_SMOKE` fixtures may use synthetic identifiers, but such artifacts can never enter
official statistics.

### 5.2 JSON validity rules (binding)
- Undefined values are **`null`**, always paired with a `*_status` string from the §3.3 enum.
- **Never** emit `NaN`, `Infinity`, `-Infinity`, `""`, `-1`, or any other sentinel for "undefined".
- Floats serialise with full `repr` precision (≥17 significant digits); no pre-rounding. Rounding is a
  presentation concern for Chapter 4, never an artifact concern.
- Row order in `per_image.jsonl` is **ascending `manifest_index`**, always.

### 5.3 `summary.json` — required fields

```jsonc
{
  "schema_version": "plantseg-eval/1.0.0",     // string, fixed
  "metric_protocol": "plantseg-metrics/1.0.0", // string, fixed
  "run": {
    "run_id":            "string",   // unique, e.g. e1_test_clean_20260726T101500Z
    "artifact_status":   "official | provisional | smoke",
    "stage":             "teacher | E1 | E2 | E3 | A | F | G | E4 | E5 | E6 | E7",   // A, F, G [UPDATED 2026-10-02 — L-CKPT-GUARD]: §7.4
    "model_role":        "teacher | student",
    "precision":         "fp32 | int8_ptq | int8_qat",
    "quant_backend":     "string | null",      // e.g. qnnpack, fbgemm/x86
    "checkpoint_path":   "string | null",      // null only when random_init is true
    "checkpoint_sha256": "string | null",
    "random_init":       false,                // true ⇒ artifact_status MUST be "smoke"
    "repo_commit":       "string",             // 40-hex
    "governed_paths_clean": true,              // §7 scoped rule; false ⇒ MUST NOT be "official"
    "dirty_allowlisted": ["docs/reference/reference.pdf"],  // observed non-governed dirty/untracked paths
    "worktree_state_sha256": "string",         // sha256 of full `git status --porcelain` output
    "metric_impl_sha256":    "string",         // sha256 of the metric module actually imported
    "timestamp_utc":     "ISO-8601 Z",
    "env": { "python": "s", "torch": "s", "torchvision": "s", "numpy": "s", "device": "s" },
    "config_sha256":     "string"              // hash of the resolved config snapshot below; optional sibling "eval_runtime": {…} per §10 [UPDATED 2026-09-24 — B66-prep L-EVAL-DET]
  },
  "dataset": {
    "name": "PlantSeg", "doi": "10.5281/zenodo.17719108",
    "split": "val | test",
    "split_manifest_sha256": "string",         // hash of the ordered canonical ID list
    "expected_rows": 1561, "actual_rows": 1561,
    "condition": { "type": "clean | corruption", "name": "string | null", "severity": "int | null" },
    "preprocess_protocol": "core_preprocess/1.0.0",
    "num_classes": 116, "background_index": 0, "ignore_index": 255,
    "class_map_sha256": "string"
  },
  "dataset_level": {
    "all_class_miou":        0.0,   "all_class_miou_n_eligible":  116,
    "all_class_macro_dice":  0.0,   "all_class_dice_n_eligible":  116,
    "all_class_macc":        0.0,   "all_class_macc_n_eligible":  116,
    "disease_only_miou":     0.0,   "disease_only_miou_n_eligible": 115,
    "aacc_diagnostic":       0.0
  },
  "per_class": {
    "class_ids": [0],  "gt_support": [0], "pred_support": [0],
    "intersection": [0], "union": [0],
    "iou":  [0.0],  "dice": [0.0], "acc": [0.0],   // null where undefined
    "iou_status":  ["ok | undefined_absent_from_gt_and_pred"]
  },
  "integrity": {
    "row_count_ok": true, "ids_unique": true, "ids_match_manifest": true,
    "order_canonical": true, "duplicate_ids": [], "missing_ids": [],
    "invalid_pred_labels": 0,     // predictions outside [0, 115] — MUST be 0
    "undefined_per_image_scores": 0,
    "overwrite_policy": "refuse_existing"
  }
}
```

**[UPDATED 2026-09-28 — L-AM5]** A lane artifact also carries `am5` and `artifact_schema_version`
(§3.2).

**[UPDATED 2026-09-28 — L-AM13]** An artifact scored under the upstream protocol also carries
`protocol`, and its `dataset.preprocess_protocol` is `upstream/1.0.0` (§11).

### 5.4 `per_image.jsonl` — required fields (one object per line)

```jsonc
{
  "image_id": "apple_black_rot_google_0001",   // canonical: file stem
  "clean_image_id": "apple_black_rot_google_0001", // == image_id for clean; the source stem for corrupted
  "manifest_index": 0,                          // 0-based position in the canonical ordered manifest
  "condition": { "type": "clean", "name": null, "severity": null },
  "all_class_miou": 0.0,            "all_class_miou_status": "ok",
  "disease_only_miou": 0.0,         "disease_only_miou_status": "ok",
  "n_eligible_all_class": 2,        "n_eligible_disease_only": 1,
  "gt_disease_classes": [1]         // GT-present disease class ids; exactly one, dataset-wide
}
```

**[UPDATED 2026-09-28 — L-AM5]** A lane artifact's rows also carry `am5_excluded`, as the last key
(§3.2).

**[UPDATED 2026-09-28 — L-AM13]** An upstream artifact's rows append `ori_shape`, `rescaled_shape` and
`padded_shape` after `am5_excluded` (§11); a canvas row is unchanged.

### 5.5 `sufficient_stats.npz` — required arrays

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `image_index` | int64 | `[nnz]` | row index into the ordered manifest |
| `class_id` | int64 | `[nnz]` | class `c` |
| `tp` / `gt` / `pred` | int64 | `[nnz]` | `TP_c`, `GT_c`, `PR_c` for that (image, class) |
| `dataset_tp` / `dataset_gt` / `dataset_pred` | int64 | `[116]` | dataset-level per-class totals |
| `manifest_ids` | unicode | `[n_rows]` | canonical IDs, in manifest order |

**Invariant (must be asserted):** summing the sparse triplets over all images reproduces
`dataset_tp/gt/pred` exactly. This is what makes §4.2's bootstrap valid.

Full per-image 116×116 confusion matrices are **rejected** (1,561 × 13,456 int64 ≈ 168 MB per stage per
condition, ×8 stages ×16 conditions) — the sparse triplets are sufficient statistics for every metric in
§3 and for §4.2, at a tiny fraction of the size.

---

## 6. DECISION D4b — image identity and pairing

**Canonical image ID = the image file stem** (e.g. `apple_black_rot_google_0001`), the same key
`src/data/dataset.py` already pairs on. Proven globally unique: `dataset_report.md` reports duplicate stems
**0/0** and zero cross-split overlap.

**Required mechanism — the wrapper emits canonical identity DIRECTLY with each sample.** A2 wraps
`PlantSegDataset` in a thin evaluation-only `Dataset`. Each sample it yields must carry, at minimum:

| Element | Purpose |
|---|---|
| image tensor | model input |
| mask tensor | ground truth |
| **canonical image ID** (file stem) | the pairing key, resolved **inside** `__getitem__` |
| **stable manifest index** | canonical ordering + integrity checks |

The exact Python return structure (tuple, dict, or dataclass) is an **A2 implementation decision**; the
binding requirement is that identity travels *with the sample* and is **never reconstructed after
batching**.

**Rejected — resolving the ID after batching.** Returning only `(image, mask, index)` and later computing
`base.pairs[index][0].stem` re-introduces a read of a mutable dataset attribute at exactly the point this
design exists to eliminate, and it breaks if the underlying dataset is wrapped, filtered, or rebuilt.
Resolve the stem once, inside `__getitem__`, and carry it.

**Rejected — deriving IDs from batch position** via `loader.dataset.pairs[i]`. Correct only under the
current `shuffle=False, drop_last=False` configuration, and it fails **silently** — the worst failure mode
available, because misalignment corrupts all 8 paired tests without raising.

**Rejected — modifying `PlantSegDataset.__getitem__`.** It would change the training data path to serve an
evaluation-only need; the wrapper is strictly less invasive.

**Rejected:** modifying `PlantSegDataset.__getitem__` to return IDs. It changes the training data path for
an evaluation-only need. The wrapper is strictly less invasive and requires no core-dataset edit.

**Asserted before any artifact is written** (all hard failures):
`actual_rows == expected_rows` · IDs unique · ID set == split-manifest set · rows ordered by ascending
`manifest_index` · zero invalid prediction labels · undefined-score count == 0 on val/test.

**Corrupted conditions** reuse this contract unchanged: `image_id` becomes
`<stem>__<corruption>_s<severity>`, `clean_image_id` retains the bare stem, and `condition` carries the
type/name/severity. Clean↔corrupted joins and mIoU-C aggregation therefore need no second schema.

---

## 7. Official vs provisional artifacts

| Guard | Rule |
|---|---|
| Default split | `val`. Evaluating `test` requires an explicit confirmation flag. |
| Batch capping | Any row-limiting option is **forbidden** on `test`. |
| Random init | **Forbidden** on `test`; elsewhere forces `artifact_status = "smoke"`. |
| Worktree | **Scoped rule — see §7.1.** A whole-worktree-clean requirement is explicitly **NOT** used. |
| Row count | `official` requires `actual_rows == expected_rows` (1,561 on test). |
| Overwrite | Refuse to overwrite an existing artifact directory. |
| Namespacing | `smoke` artifacts must not be written to the official results location. |

Development smoke runs use **validation or synthetic data only**.

### 7.1 Official-run worktree rule (scoped, not whole-repo)

A blanket "repository must be clean" requirement is **unsatisfiable in this repo and must never be
introduced.** `CLAUDE.md` mandates that `docs/reference/reference.pdf` stays modified and unstaged
permanently, so a whole-worktree-clean gate would make `official` unreachable forever.

An artifact may be marked **`official`** only when **all** of the following hold:

1. **Governed paths are clean** — no dirty *or* untracked files under any of:
   `src/**` · `configs/**` · `scripts/**` · `requirements*` · `docs/EVALUATION_CONTRACT.md` ·
   `docs/IMPLEMENTATION_CONTRACT.md`.
   Record as `governed_paths_clean = true`.
2. **Every remaining dirty/untracked path is allowlisted** — enumerated verbatim in
   `dirty_allowlisted[]`. A path outside the allowlist blocks `official` even if it is not governed.
3. **The full worktree state is recorded** — `worktree_state_sha256` = SHA-256 of the complete
   `git status --porcelain` output, recorded for **every** run regardless of status.
4. **The exact implementation and configuration are identified** — `repo_commit`,
   `metric_impl_sha256` (hash of the metric module actually imported at runtime), `config_sha256`,
   `checkpoint_sha256`, and `split_manifest_sha256` all present and non-null.

**Allowlist status (observation, not methodology).** For the repository's current state the allowlist is
`docs/reference/reference.pdf` — permanently dirty by standing policy. `AGENTS.md` was previously
recorded here as an observed non-governed untracked path; that pending decision is now resolved — it is
tracked, so it is no longer dirty or untracked and has been removed from the allowlist. This paragraph
remains an observation of repository state, never methodology: any future non-governed dirty path must
be enumerated verbatim in `dirty_allowlisted[]` to permit `official`.

**[UPDATED 2026-10-05 — CP-007e]** The data session's six evaluator artifacts (2026-10-03, afd2d33: the four upstream-protocol runs
and the two AM-17b item 2(c) re-scores) recorded two `dirty_allowlisted` paths: `docs/reference/reference.pdf`
(DL-26; the sanctioned exception, modified and unstaged by standing policy) and the untracked
`.claude/settings.local.json` (Claude Code's local settings file). From this commit `.gitignore` lists
`.claude/settings.local.json`, so later artifacts no longer list it; the six keep their recorded
paths.

**Rejected — requiring a pristine clone or detached worktree.** Cleanest in principle, but it conflicts
with the standing `reference.pdf` policy and adds RunPod friction for no methodological gain; the scoped
rule above already guarantees that no code, config, or governing contract differs from `repo_commit`.

### 7.2 Teacher readiness evaluation (M5-R3) and development data isolation (M11) `[B60, 2026-09-22]`

**Decision locked; runtime implemented by B62 (committed `3c43f89`; frozen, runbook §4a; CUDA re-canary pending).** Recorded by
[reports/b60_teacher_methodology_lock.md](../reports/b60_teacher_methodology_lock.md).

**R3 evaluation semantics.** The teacher readiness number is the **dataset-level all-class mIoU of §3.1**:
- union-present eligibility, one accumulated `CM`, `ignore_index = 255`;
- computed by this contract's evaluator on the **VAL** split (846 rows), using the `core_preprocess` canvas;
- for the teacher checkpoint selected on VAL.

Disease-only mIoU (§3.1) is reported alongside it.

- **How it is produced.** A **controlled deterministic re-evaluation under the M4-locked evaluation
  rule.** ~~The NMF/Hamburger control is `NEED_TO_CONFIRM` until M4 is locked, and R3 cannot run before
  then.~~ **[UPDATED 2026-09-22 — B61]** M4 is locked: R3 uses the **M4-V** rule of §7.3 on the checkpoint
  selected under M12. **[B62]** Implemented as `scripts/teacher_readiness_r3.py` (evaluator artifact
  `artifact_status = provisional`; readiness record `teacher_r3_readiness.json`). R3 runs only after the
  official teacher run.
- **Comparator.** E1's run-of-record VAL all-class mIoU, **0.36314016580581665**. It was produced by the
  same metric implementation (`miou_from_confusion`, union-present) on the same VAL canvas.
- **Rule.** The teacher value must be strictly greater, with no margin.
- **What it is not.** The same VAL partition took part in checkpoint selection, so R3 is an
  **operational competence floor**. It is **not** an independent or fresh unbiased estimate, **not** an
  inferential comparison, and **not** a thesis result. It enters no Holm family, no bootstrap and no
  official statistics.
- **Artifact.** Any R3 artifact is a VAL readiness record, never an official TEST artifact.
- **[AM-13, 2026-09-23] Published comparison.** At the single TEST evaluation the teacher is additionally
  scored, descriptively, under the upstream PlantSeg protocol: the repository's aspect-ratio-preserving
  resize, scored against original-resolution ground truth. That score, not the 512-canvas score, is the
  one compared with the published 42.05%. Code: lane L-AM13 (TEST-time only). **[UPDATED 2026-09-28 —
  L-AM13]** Implemented as the `upstream` protocol (§11); under AM-17 item 1(e) the teacher is also scored
  with it on VAL.
- **[AM-16, 2026-09-24] Students.** Every student is also scored under the upstream PlantSeg protocol at
  the single TEST evaluation (descriptive), as the teacher is under AM-13. E1 is also scored this way on
  VAL, on the existing seed-42 best checkpoint, before the first KD run. Code: lane L-AM16-UP.
  **[UPDATED 2026-09-28 — L-AM13]** The code is §11, delivered by lane L-AM13 (docs/lane_specs/part1.md
  lane 2 covers AM-16 item 4); it applies to every student precision.

**M11 development data isolation.**
- **Which runs:** the official teacher, E2 and E3. Development and training data roots are staged with
  **TRAIN and VAL only**.
- **Where:** in the **configured data root**, `images/test` and `annotations/test` must be absent.
  Preflight verifies TRAIN 5,367 and VAL 846 and fails closed if the TEST paths exist.
- **Never during development:** TEST is not enumerated, counted or inspected.
- **[B62] Runtime check:** `src/data/isolation.py` fails closed if `images/test`, `annotations/test` or
  `annotation_test.json` exists in the staged root (existence only; the third is an approved B62 safeguard).
- **Scope:** the configured, staged data root only, not the whole host.
- **Official TEST integrity** (the 1,561-row manifest and row-count guards of §5–§7) is verified **only
  after the final TEST unlock**.
- **Over time:** forward-looking. E1 is unaffected, because its training path constructed TRAIN and VAL
  datasets only. **[UPDATED 2026-09-24 — B66-prep S2, DL-21]** From B66, real E1 runs also use a TRAIN/VAL-only staged root (`train_e1.py` real-run gate; `scripts/preflight_e1_trainval.py`).
- **[AM-19 item 1(c), 2026-10-02]** The TEST-day manifest carries a generator pin for every entry that can only be computed from TEST files; the values are computed in the TEST session.

### 7.3 Teacher NMF evaluation rule (M4-V) and checkpoint selection (M12) `[B61, 2026-09-22]`

**Decision locked; runtime implemented by B62 (committed `3c43f89`; frozen, runbook §4a; CUDA re-canary pending).** Recorded by
[reports/b61_teacher_nmf_checkpoint_selection_lock.md](../reports/b61_teacher_nmf_checkpoint_selection_lock.md).

**Why a rule is needed.** The teacher's LightHamHead keeps upstream `rand_init=True`, so every forward
draws fresh NMF bases from the CPU generator. On the real ADE20K checkpoint, 20 repeated forwards gave 20
distinct outputs, with the predicted class changing on 26.0% of valid pixels on average and ~20% relative
change at the pre-classifier feature (MEASURED, B61 §2). An evaluation must not depend on accidental
global RNG state.

**M4-V — applies to every complete teacher evaluation pass:** the VAL checkpoint-selection validations,
R3, and the final descriptive teacher evaluation.
1. Save the caller/global CPU RNG state.
2. Initialise the dedicated NMF sequence from **seed 42**.
3. Evaluate the entire split in a **frozen deterministic sample order**.
4. **Batch size 1.**
5. `rand_init=True` draws a fresh basis for each image, as upstream.
6. After the complete pass, restore the caller/global CPU RNG state exactly.

The manifest/order is persisted or hash-attested. Changing the manifest/order, batch size or seed
invalidates comparability unless explicitly disclosed and rerun consistently. The dedicated stream is
consumed by NMF only, and is seeded without touching the CUDA generators (B61 §4).

**M12 — teacher checkpoint selection.**
- Validations at 4,000, 8,000, …, 40,000 iterations, each under M4-V, with the same frozen VAL manifest
  and order, no shuffle, the same preprocessing and the same all-class dataset-level mIoU computation.
- Select the **numerically highest** VAL all-class mIoU. Exactly equal stored values → the **earliest
  iteration**. No tolerance/noise-band tie rule, no TEST, no training extension because of the result.
- The compared value is the full-precision all-class mIoU, not MMSeg's two-decimal summary.
- Persist per validation: iteration, all-class mIoU, disease-only mIoU where reported, NMF seed, VAL
  manifest hash/order identity; and the selected iteration and checkpoint SHA-256.

**R3 under M4-V.** R3 (§7.2) re-evaluates the selected checkpoint once under M4-V with this contract's
evaluator. Its value is **not required to be byte-identical** to the MMSeg selection metric, unless both
are proven to use the exact same evaluation implementation.

**[AM-3, AM-4, 2026-09-23] INT8 VAL scoring.** QAT checkpoint selection (AM-4) and the E6-KD trigger
(AM-3) score VAL on the **converted INT8** (QNNPACK) model on CPU, not on the fake-quant model.
Code: lanes L-AM4 and L-AM3.

### 7.4 Exploratory arms A, F, G; declared checkpoint stage; teacher checkpoint identity [added 2026-10-02, lane L-CKPT-GUARD; DL-52, DL-59]

**Arms.** The evaluator scores the exploratory arms A (both channel-wise terms; AM-17 item 7), F (feature-map term only; AM-17b item 1(a)) and G (logit-map term only; AM-17b item 1(b)) as stages of their own: `scripts/evaluate_model.py --stage A|F|G` with `--model-role student --precision fp32 --checkpoint <the arm's best checkpoint>`. In `src/eval/stage_artifacts.py` they are `fp32_checkpoint` stages and, with the teacher, members of `DESCRIPTIVE_ONLY_STAGES`. An arm's checkpoint passes the same projection-free check as E3 (`assert_clean_student_state`: no training-only CWD projection key in `model_state_dict`). §5.3's `run.stage` admits A, F and G; `schema_version` (`plantseg-eval/1.0.0`) and `artifact_schema_version` (`plantseg-eval-artifact/1.2.0`) are unchanged: the field gains values and no field changes.

**Descriptive only.** No evaluator code enforces it: an artifact of the teacher, A, F or G is an ordinary artifact. The statistics driver must refuse the teacher, A, F and G as comparators in any inferential test.

**Declared stage.** Every checkpoint `src/training/train_distill.py` writes records its stage (`E2`, `E3`, `A`, `F` or `G`). Before any dataset or model exists, the evaluator refuses an E2, E3, A, F or G checkpoint that records no stage (`stage_undeclared`) or another stage (`stage_mismatch`). E1 checkpoints record none (`train_e1.save_checkpoint`): `--stage E1` accepts a checkpoint without a stage and refuses one that records another, and an E1 checkpoint is refused under `--stage E2`, `E3`, `A`, `F` or `G`.

**Teacher checkpoint identity.** The teacher stage requires `--teacher-ckpt-sha256`, the checkpoint's SHA-256 as 64 lowercase hex characters (`[teacher_ckpt_sha256_required]`, `[teacher_ckpt_sha256_format]`); any other stage refuses the flag (`[teacher_ckpt_sha256_not_teacher]`). The file is hashed and compared before it is parsed (`teacher_hash_mismatch`), hashed again before the model is built, and its state must match the built teacher exactly at load (IMPLEMENTATION_CONTRACT B1, the L-CKPT-GUARD note). R3 (`scripts/teacher_readiness_r3.py`) passes the checkpoint's SHA-256 after verifying that it equals the `checkpoint_sha256` of `teacher_selection.json`. The rule is the same under both protocols (`canvas` and `upstream`, §11). The file evaluated is the M12-selected `iter_<N>.pth` (the teacher of record is `iter_24000.pth`); `best_mIoU_full_iter_<N>.pth` is another file with another hash and is refused. Every evaluator refusal exits 1.

---

## 8. Implementation record — E1 validation correction (CLOSED, A1b 2026-07-26)

**Status: COMPLETE. No implementation blocker remains.** This section previously specified a required
correction; it is now the record of what shipped. Metric definitions are unchanged — §3.1 and §3.2 remain
the single statement of the rules and are **not** restated here.

**Test evidence:** all **21/21** frozen metric-contract cases PASS, **exit code 0**
(`scripts/smoke_metrics_contract.py`). The same cases stood at 16 PASS / 5 EXPECTED_RED in the A1a RED
phase, so the RED→GREEN chain demonstrates the change in production behaviour rather than in the tests.
Reports: [a1a_metric_contract_red.md](../reports/a1a_metric_contract_red.md) →
[a1b_metric_contract_green.md](../reports/a1b_metric_contract_green.md).

**Delivered in `src/eval/metrics.py`:**

| Interface | Eligibility | Per |
|---|---|---|
| `miou_from_confusion` | **union-present** | §3.1 dataset-level mIoU |
| `macc_from_confusion` *(new)* | **GT-present** | §3.1 dataset-level mAcc |
| `dice_from_confusion` *(new)* | **union-present** | §3.1 dataset-level macro Dice (project-defined) |
| `_miou_from_cm`, `per_image_miou` | **GT-present — unchanged** | §3.2 per-image |

The dataset-level ↔ per-image eligibility separation required by §3.1/§3.2 is therefore enforced by
construction: the per-image core was not touched, and the three dataset-level reducers each apply their own
rule.

**Label-integrity guard.** `confusion_matrix` validates shape equality, target values
(`0..num_classes−1` ∪ `{ignore_index}`), and prediction values at non-ignored positions, raising a clear
`ValueError` instead of permitting an out-of-range label to alias through `target * num_classes + pred`.
Satisfies §5.3 `invalid_pred_labels == 0`. Nothing is clamped, remapped, or discarded, and the
zero-valid-pixel NaN signal of §3.3 is preserved.

**E1 validation inheritance.** `src/training/train_e1.py` was **not modified**: `validate` already called
`miou_from_confusion`, so checkpoint-selection arithmetic follows §3.1 automatically. `configs/e1_student.py`
`checkpoint_selection = "best_val_miou"` is unchanged — D1 stands; only the arithmetic was corrected.

**Float32 reduction unchanged.** Counts are exact `int64`; the reduction remains `float32` (~7 significant
digits), pinned by contract case C4. A float64 reduction is a **separate optional follow-up**, deliberately
out of scope here.

**Pilot note.** This section previously referred to a "2,000-iteration E1 pilot". That value is `[ctx]`-only
and is **not** a Chapter III requirement — see `open_questions.md` **D6** for the resolved
`[ch3]` / `[ctx]` / `[operational]` split.

---

## 9. What A1 and A2 must not re-decide

Frozen here: dataset-level eligibility (union-present for IoU/Dice, GT-present for mAcc); per-image
eligibility (GT-present, both vectors); ignore-255 masking on ground truth only; background inclusion;
undefined-value representation (`null` + status, never `NaN`); the test count (**1,561**); ~~expected defined
scores (**1,561/1,561**)~~ (**[UPDATED 2026-09-23 — B64 C5]** expected defined per-image disease-only scores = 1,561 minus the AM-5 zero-disease count; code: lane L-AM5); the bootstrap resampling unit (images, with `CM` re-accumulation); canonical
identity (file stem via index-emitting wrapper); and the four-file artifact layout with its required fields.

A1 implements metric tests against §3 and §8.4. A2 implements the evaluator against §5–§7. **[UPDATED 2026-09-24 — B66-prep L-EVAL-DET]** The evaluation runtime record and the evaluation determinism rule are implemented against §10.

---

## 10. Evaluation runtime record and determinism (DL-17, L-EVAL-DET) [added 2026-09-24, B66-prep]

**(a) Record.** `summary.run.eval_runtime`, versioned `plantseg-eval-runtime/1.0.0`
(`EVAL_RUNTIME_VERSION`, `src/eval/artifacts.py`). `scripts/evaluate_model.py` writes it on every run, and
`write_artifact` refuses an `official` artifact without it. Core-only fixture writers and artifacts written
before L-EVAL-DET may lack it, so readers must tolerate its absence. It is outside `config_sha256`;
`schema_version` stays `plantseg-eval/1.0.0`. §5.2's ban on `""` applies at every depth, and `null` means
not applicable. The record has exactly these 26 keys (`src/eval/eval_runtime.py`, `RECORD_KEYS`):

| Key | Meaning |
|---|---|
| `eval_runtime_version` | `plantseg-eval-runtime/1.0.0` |
| `model_device` | the model's construction device (`cpu`, `cuda:N`); `env.device` keeps recording the requested device |
| `model_tensor_devices` | sorted devices of every parameter and buffer (`[]` if none) |
| `input_devices` | sorted devices the input batches were moved to |
| `forward_batches`, `batch_size`, `num_workers` | loader facts (`num_workers` is always 0) |
| `determinism_policy_applied` | true only for an FP32 student on CUDA (b) |
| `cuda_initialized_before_policy` | false when the policy was applied; null otherwise |
| `inherited_cublas_workspace_config` | `CUBLAS_WORKSPACE_CONFIG` before the policy (null if unset) |
| `determinism` | the live five-state after evaluation (`deterministic_algorithms`, `deterministic_algorithms_warn_only`, `cudnn_deterministic`, `cudnn_benchmark`, `CUBLAS_WORKSPACE_CONFIG`) |
| `fill_uninitialized_memory` | `torch.utils.deterministic.fill_uninitialized_memory`; null where torch lacks it (torch 2.1.0) |
| `cudnn_enabled`, `tf32` | cuDNN switch; TF32 state (two torch flags, `float32_matmul_precision`, two env overrides), recorded and never set |
| `gpu_name`, `gpu_capability`, `cudnn_version` | non-null only when the model is on CUDA |
| `torch_cuda`, `torch_num_threads`, `pillow` | runtime versions and CPU threads |
| `checkpoint_iteration`, `checkpoint_best_val_miou_all_class` | the student checkpoint's stored `iter` and best VAL mIoU (null for random-init, teacher and INT8, and when the stored value is missing or non-finite) |
| `image_digest` | `PLANTSEG_IMAGE_DIGEST`, stripped; null if unset or blank |
| `eval_warnings`, `eval_warnings_truncated`, `nondeterministic_alert_count` | unique warnings captured around the evaluation core (first line, at most 50 entries), and the number of emitted "does not have a deterministic implementation" alerts; each unique warning is also echoed to stderr as `[eval-warning xN] …` |

**(b) Policy.** For `model_role=student`, `precision=fp32` on a CUDA device only, the evaluator applies
IMPLEMENTATION_CONTRACT B6's four settings exactly as `src/seeds.py` applies them for E1
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`, `cudnn.benchmark=False`,
`use_deterministic_algorithms(True, warn_only=True)`), with no reseed. It runs after the pre-inference gate
(`validate_artifact_request`) and before anything can initialise CUDA, and it is refused
(`cuda_initialized_before_policy`) if CUDA is already initialised: one CUDA evaluation per process. TF32 is
recorded, not set. Teacher, INT8 and CPU evaluations initialise no CUDA, apply nothing and record
`determinism_policy_applied=false`.

**(c) Inputs.** Every input batch moves to the model's construction device before the forward, as
`train_e1.validate` does; predictions and targets return to the CPU for the confusion matrix, so §3's
arithmetic is unchanged. A run is refused, before anything is written, on `model_device_mismatch`,
`input_device_mismatch`, `policy_drift` or `checkpoint_changed_after_validation`.

**(d) Re-scoring identity.** Artifacts A and B are identical when both pass `verify_artifact`,
`per_image.jsonl` is byte-equal, the NPZ key sets are equal and every array has the same dtype, shape and
raw bytes, and `summary.json` is equal as canonical JSON (sorted keys, compact separators, no non-finite
values) after removing only `run.run_id` and `run.timestamp_utc`. NPZ container bytes need not match.

**[UPDATED 2026-10-05 — CP-007e]** `summary.json` also carries `worktree_state_sha256` and `dirty_allowlisted` (§7.1), so the two
runs of a re-scoring pair run on one unchanged checkout: any checkout change between them fails (d). In the
data session (2026-10-03) the worktree probe gave one state in all eight runs (`4bbc54ef…`).

**(e) DL-17.** `scripts/compare_eval_artifacts.py` is the DL-17 identity and PASS implementation (exit 0
PASS, 1 FAIL, 2 validity violation; 3 is never a verdict: a usage error or `--help`, the same directory
twice, two artifacts carrying the same `run.run_id` (a copy of one run is not two runs), or an unexpected error, and it releases nothing until DL-17 is re-run to a verdict). Validity
preconditions, per artifact: `run.eval_runtime` present;
`determinism_policy_applied`; `model_device` == `input_devices` == `cuda:N`; batch size 16, 0 workers, 53
forward batches, 846 rows; checkpoint iteration 80000 and stored best 0.36314016580581665; the DL-21 image
digest; Σ `gt_support` = 159,279,104; `run.checkpoint_sha256` = cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03;
`gpu_name` == "NVIDIA A40" exactly (the string seed 42's run_meta recorded; `DL17_GPU_NAME`, ruling R-GPU). A violation is not a DL-17 FAIL: the run is not a DL-17 run (STOP and report). PASS:
the two runs are identical under (d), and |`all_class_miou` − 0.36314016580581665| ≤ 1e-4; a difference in
(1e-6, 1e-4] is recorded, not a failure. DL-17 artifacts are `provisional`, and each run writes to a fresh
out-dir outside the clone. A FAIL or a validity STOP blocks every step that relies on this evaluator (E4–E7
and every official evaluation) until diagnosed; neither blocks the E1 training launches, which do not use it.

---

## 11. Upstream PlantSeg protocol (L-AM13) [added 2026-09-28, lane L-AM13]

Authority: AM-13, AM-16 item 4 and AM-17 item 1(e) ([PREREGISTRATION_AMENDMENTS.md](PREREGISTRATION_AMENDMENTS.md));
docs/lane_specs/part1.md lane 2. Code: `src/eval/protocols.py`, `src/data/original_resolution.py`, the
protocol switch in `src/eval/adapters.py`, and `scripts/evaluate_model.py --protocol {canvas,upstream}`.

**(a) Scope.** The evaluator has two protocols. `canvas` (`core_preprocess/1.0.0`, the default) is the
protocol of §§2–10: every inferential, checkpoint-selection and readiness number, including R3 (§7.2).
`upstream` (`upstream/1.0.0`) reproduces the test geometry of `tqwei05/PlantSeg` for descriptive scores
only: the teacher at TEST (AM-13), every student at TEST and E1 on VAL before the first KD run (AM-16
item 4), and the teacher on VAL for the protocol effect, upstream minus canvas (AM-17 item 1(e)). An
upstream score never enters the Holm family, a bootstrap, checkpoint selection or R3, and is never an
inferential comparator; it is the score compared with the published 42.05% (AM-13). Enforcement in code
is partial (open): `src/stats/align.py` refuses to pair a canvas and an upstream artifact, but the
frozen OFFICIAL and REHEARSAL ingest (`src/stats/ingest.py`) does not check the protocol, so the guard
`dataset.preprocess_protocol == "core_preprocess/1.0.0"` belongs in the statistics entry points that
consume TEST or VAL artifacts. **[UPDATED 2026-10-05 — CP-007f]** Done in the statistics driver (DL-71; STATS §10 note, 0c69abf).

**(b) Geometry** (source-proven from the upstream configs; lane spec (a)):
1. Rescale as mmcv `Resize(scale=(2048, 512), keep_ratio=True)`: s = min(2048 / long side, 512 / short
   side), and each side becomes int(side · s + 0.5), mmcv `rescale_size`'s half-up rounding
   (1025 × 1024 → 513 × 512). The image is resized bilinearly; the mask is never resized.
2. Pad the normalised tensor, bottom and right, with 0.0 to a multiple of 32 (`SegDataPreProcessor`
   with `test_cfg=dict(size_divisor=32)` and `pad_val=0`; mmseg `stack_batch`).
3. Whole-image inference at batch size 1; the CLI refuses any other batch size.
4. Logits bilinear (align_corners=False) to the padded input shape (`predict_by_feat`), cropped to the
   rescaled shape, bilinear (align_corners=False) to the original shape (`postprocess_result`), then
   argmax. Both bilinear steps are kept. Both evaluator models already end with the first step: the
   student's final interpolate to its input size, and `FrozenTeacher`'s `logits_size` resize.
5. The evaluation core accumulates the confusion against the original-resolution mask (ignore 255,
   116 classes, background 0) with the frozen `src/eval/metrics.py`; dataset-level mIoU is
   `miou_from_confusion` (union-present, §3.1). The functions are the same and the confusion is
   different; `metric_impl_sha256` is unchanged.

**(c) Numerics.** Only the geometry changes. The image numerics are the evaluator's own and identical
for both models, because the R3 teacher path feeds the teacher the student's tensors: EXIF transpose of
the image only, RGB, the repository's PIL bilinear resize (as `core_preprocess`), [0, 1] scaling and
ImageNet mean/std (`transforms.finalize`). mmcv's cv2 decoding and resize are not used. PIL's bilinear
filter widens with the reduction factor and cv2's `INTER_LINEAR` does not, so rescaled pixels differ
from the upstream pipeline's, increasingly with the downscale factor (on a synthetic texture, a mean
absolute difference of about 3/255 at 2× and about 10/255 at about 6×; within 1/255 when enlarging).
Every upstream artifact's `note` states this, and it is disclosed with every upstream score. The teacher
keeps M4-V (§7.3): batch size 1, the frozen manifest order, the NMF stream seeded 42 once per pass. The
NMF basis draw does not depend on the image size, so both protocols consume the stream identically.

**(d) Data.** `PlantSegDataset` is not modified (§6). `src/data/original_resolution.py` reuses its pairing,
missing-mask guard and split-count guard, and returns each pair at the original resolution. An image
whose EXIF-transposed shape differs from its mask's shape is refused by name, because the upstream
protocol never resizes the mask. The expected manifest, the identity rules and every §7 guard, including
the TEST confirmation flag, are identical under both protocols.

**(e) Artifact** (layout `plantseg-eval-artifact/1.2.0`). The four files of §5.1; `schema_version` and
`metric_protocol` are unchanged; `dataset.preprocess_protocol` is `upstream/1.0.0` and enters
`config_sha256`. summary.json adds:

```jsonc
"protocol": {
  "name": "upstream", "version": "1.0.0",            // name/version == dataset.preprocess_protocol
  "resize": {"short": 512, "long_max": 2048, "interp": "bilinear"},
  "size_divisor": 32, "pad_value_normalized": 0.0,
  "logit_upsample": ["bilinear_to_padded", "crop", "bilinear_to_original"],
  "align_corners": false, "scored_resolution": "original",
  "note": "..."                                       // descriptive use; the class-count note of (f)
}
```

Every per_image.jsonl row appends `ori_shape`, `rescaled_shape` and `padded_shape` (`[h, w]` each) after
`am5_excluded`. Every prediction is argmaxed at its row's `ori_shape` (the evaluator refuses otherwise),
so for each image the sufficient statistics hold Σ_c pred = Σ_c gt = its valid-pixel count. The writer
refuses a `protocol` block other than the one above and any row whose shape fields are not the (b)
geometry of its `ori_shape`. A canvas artifact carries neither the block nor the fields: under 1.2.0 its
files are laid out exactly as under 1.1.0, and readers derive "canvas" from the block's absence (every
artifact older than 1.2.0 is a canvas artifact).

**(f) Class-count note.** The upstream MSCAN-B class count behind the published 42.05% is NOT DETERMINABLE:
the upstream PlantSeg MSCAN-L config uses 116 classes and the MSCAN-T config 115. Every upstream artifact
carries this note, and Chapter 4 states it with the comparison.

**(g) Tests.** `scripts/smoke_am13_upstream.py` (d1, d2, d3 for the student, d4, d6, the layout and the CLI
end to end) and `scripts/smoke_am13_teacher.py` (d3 for a random-init teacher and the teacher CLI path;
teacher image). d5, the real VAL runs of E1 seed 42 and the teacher of record, is local and checked by
`scripts/am13_real_checks.py d5`.
