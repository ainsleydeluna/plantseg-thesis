# Evaluation Contract — metric semantics + result-artifact schema (A0)

> **Status: FROZEN (2026-07-26).** This is the authoritative definition of *how every evaluation number in
> this thesis is computed* and *what every evaluation run must emit*. A1 (metric tests) and A2 (evaluator)
> implement this document and must not invent additional metric or schema decisions.
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
| Split counts | train **5,367** · val **846** · test **1,561** (total 7,774) | `dataset_report.md` §"Split mechanism" (folder == JSON == CSV, pairwise overlap 0); `dataset_audit_summary.md` §1 |
| `num_classes` | **116** (background 0 + diseases 1–115) | `dataset_report.md` §"num_classes critical conflict"; `configs/data.py` `num_classes` |
| Background index | **0** | `dataset_report.md` §"Background / disease determination" |
| Ignore index | **255** — pad/rotation-fill only; **absent from all released masks** | `dataset_report.md` §"Mask label analysis"; `test_mask_value_audit.md` §2; `trainval_mask_value_audit.md` §1/§2 |
| Mask encoding | `mask value = COCO category_id + 1`, **100 % exact across all 7,774 images** | `trainval_mask_value_audit.md` §0/§4 |
| **One disease class per image** | **holds dataset-wide** — `distinct_category_ids_per_image = {1: N}` for every split | `trainval_mask_value_audit.md` §1/§2/§3; `test_mask_value_audit.md` §3 |
| Background present per image | test **1,561/1,561** · val **846/846** · train **5,366/5,367** | `trainval_mask_value_audit.md` §4 |
| Class absent from **val** GT | class 68 → **mask value 69** | `trainval_mask_value_audit.md` §2/§3 |
| Class absent from **test** GT | class 41 → **mask value 42** | `test_mask_value_audit.md` §1; `trainval_mask_value_audit.md` §3 |
| FLAG-F image | `apple_black_rot_google_0001` has 0 COCO polygons but a **valid mask `{0, 1}`** | `test_mask_value_audit.md` §4 |
| Stem uniqueness | duplicate stems **0/0**; zero cross-split overlap | `dataset_report.md` §"Counts & identifiers" |

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

---

## 3. DECISION D3 — frozen metric semantics

### 3.1 Dataset-level metrics (accumulate ONE `CM`, compute once)

| Metric | Classes `S` | Eligibility | Formula (per eligible `c`, then unweighted mean) | Status |
|---|---|---|---|---|
| **all-class mIoU** | `{0..115}` | **union-present** | `TP_c / UN_c` | **HEADLINE / official** — benchmark-aligned formula |
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

### 3.3 Undefined cases

| Situation | Result | Expected on val/test? |
|---|---|---|
| `E_gt` empty for the requested class set | `null` + `status = "undefined_no_eligible_class"` | **NO — never** |
| Row excluded by an approved data-integrity record | `null` + `status = "excluded_data_integrity"` | **NO — none preregistered** |
| Inference/metric raised | `null` + `status = "evaluation_error"` | **NO — fail the run** |
| Normal | numeric + `status = "ok"` | **YES — all rows** |

**Expected defined-score counts on the official test split: 1,561 / 1,561 for BOTH per-image vectors.**
Every test mask contains background *and* exactly one disease class, so neither vector can be undefined.
Any `null` on test is a **data-integrity failure that must abort the run**, not a row to skip quietly.

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
| — | — | — | **5 must never be produced** by the official study pipeline |

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
    "stage":             "teacher | E1 | E2 | E3 | E4 | E5 | E6 | E7",
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
    "config_sha256":     "string"              // hash of the resolved config snapshot below
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
`docs/reference/reference.pdf` — permanently dirty by standing policy — and, while it remains untracked,
`AGENTS.md`. **`AGENTS.md` is NOT permanently exempted:** its long-term tracking status is a separate
pending user decision, and this contract records it only as an observed non-governed path. If it is
committed or gitignored, remove it from the allowlist.

**Rejected — requiring a pristine clone or detached worktree.** Cleanest in principle, but it conflicts
with the standing `reference.pdf` policy and adds RunPod friction for no methodological gain; the scoped
rule above already guarantees that no code, config, or governing contract differs from `repo_commit`.

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
undefined-value representation (`null` + status, never `NaN`); the test count (**1,561**); expected defined
scores (**1,561/1,561**); the bootstrap resampling unit (images, with `CM` re-accumulation); canonical
identity (file stem via index-emitting wrapper); and the four-file artifact layout with its required fields.

A1 implements metric tests against §3 and §8.4. A2 implements the evaluator against §5–§7.
