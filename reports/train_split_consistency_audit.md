# Train Split Consistency Check (B16-4)

**Scope:** read-only consolidation of the **train** split. Directly reconciles the three train artifacts and
simulates the dataloader's folder/stem pairing rule. Three reads only: directory listings of `images\train`
and `annotations\train` (via `os.scandir`, **no decode**) and `annotation_train.json` (`file_name` stems
only). **No image/mask pixels decoded; no label values inspected.** No val/test artifacts opened. No files
modified.

**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c` · Dataset root:
`C:\Users\admin\plantseg_data\plantseg`.

---

## 0. PASS / FLAG verdict — **PASS (train split signed off at filename level; 0 new flags)**

The train image index (`annotation_train.json`), the train images (`images/train`), and the train masks
(`annotations/train`) are **mutually identical at the stem level** — one set of **5,367** stems, no
member exclusive to any artifact. A simulation of the actual dataloader pairing rule yields **exactly
5,367 image↔mask pairs with zero silent drops**. One pre-existing, non-blocking item carries forward from
B16-1 (FLAG-A: the JSON's `annotations[]` array is unfiltered) — it does not affect folder/stem loading.

---

## 1. THREE-WAY RECONCILIATION RESULT

| Stem set | Size |
|---|---|
| `images/train` `.jpg` stems | **5,367** |
| `annotations/train` `.png` stems | **5,367** |
| `annotation_train.json` `file_name` stems | **5,367** |
| **Union of all three** | **5,367** |

| Set difference | Count |
|---|---|
| image-only (not in mask/json) | **0** |
| mask-only (not in image/json) | **0** |
| json-only (not in image/mask) | **0** |
| every pairwise diff (img\mask, mask\img, img\json, json\img, mask\json, json\mask) | **0** |

- **All three sets are equal** and each contains exactly **5,367** stems → the train split is internally
  consistent across index, images, and masks. (B16-3's transitive conclusion is now **directly** proven.)

## 2. DATALOADER PAIRING SIMULATION RESULT

Replicates `src/data/dataset.py`: for each `images/train/<stem>.jpg`, keep the pair iff
`annotations/train/<stem>.png` exists (filename existence only — no decode).

| Metric | Value | Expected | Verdict |
|---|---|---|---|
| Simulated pairs (image with matching mask) | **5,367** | 5,367 | **MATCH** |
| Images silently dropped (no mask) | **0** | 0 | **PASS** |
| Masks with no image (unreached by loader) | **0** | 0 | **PASS** |

- The loader would construct **exactly 5,367 training pairs** and **drop nothing** — confirming B16-0
  FLAG-5 (silent-drop risk) does **not** materialize for the train split.

## 3. CONSOLIDATED TRAIN-SPLIT EVIDENCE (B16-1 / B16-2 / B16-3 / B16-4)

| Step | Artifact | Count | Ext | Dups | Cross-check | Verdict |
|---|---|---|---|---|---|---|
| B16-1 | `annotation_train.json` `images[]` | 5,367 | — | 0 (ids, names) | valid COCO; 115 cats (0–114) | PASS |
| B16-2 | `images/train` | 5,367 | `.jpg` only | 0 | == JSON stems (exact, case-sensitive) | PASS |
| B16-3 | `annotations/train` | 5,367 | `.png` only | 0 | == JSON stems (exact) | PASS |
| B16-4 | three-way + pairing | 5,367 | — | 0 | all equal; 5,367 pairs; 0 drops | PASS |

**Carried-forward FLAG (not a train-split blocker):**
- **FLAG-A (B16-1):** the JSON's `annotations[]` array is **not filtered to train** — 14,647/52,898 (27.7%)
  reference non-train image ids. **No loader impact** (loader ignores JSON). Matters only for future
  COCO-based use (filter `annotations` to `image_id ∈ images[].id`).

## 4. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source |
|---|---|
| Folder split is authoritative; train = 5,367; zero cross-split id overlap | `dataset_report.json` `split` |
| Whole dataset: 7,774 images / 7,774 masks, 0 unpaired, 0 dup stems | `dataset_report.json` `counts` |
| Masks single-channel indexed (`L`), labels contiguous **0–115**, **255 absent** | `dataset_report` `masks` |
| `mask = category_id + 1` consistent 99.85% (12/7,774 exceptions) | `dataset_report` `mask_value_encoding` |
| Loader pairs by stem; mask read as raw indices; `num_classes=116`, `ignore=255` | `dataset_code_expectation_audit.md` |
| Dataset root exists, outside repo, `.gitignore` safe | `dataset_location_log.md` |

## 5. REMAINING `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | **Inferred** — needs a decode-gated annotation↔mask **value** check (B16-5). Prior: 99.85% at pixel level, not 100%. |
| **NTC-2** | Per-train-mask label range (0–115, 255-absent, single-channel `L`) | Verified **full-dataset** in `dataset_report`; not re-derived per-split (no decode). |
| **NTC-3** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; not checked. |
| **NTC-4** | Annotation id space (1…7,916) vs 7,774 image files | Cross-split reconciliation (needs val/test). |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0/B16-1. |
| **NTC-6** | **Val and test splits not yet audited** | B16-1…4 covered **train only**; val/test need the same index↔image↔mask consistency pass. |

## 6. EXACT NEXT RECOMMENDED PROMPT (B16-5)

Two reasonable next moves; the prompt below proposes the **decode-gated mask-value proof** (closes NTC-1/2
for train). Alternatively, repeat B16-1…4 on the **val** split first (filename-level, no new permission).

```
B16-5 — Train mask VALUE check (decode-gated). Read-only. Plan-gated.

NOTE: This is the FIRST step that DECODES mask pixels. It opens and reads the pixel
values of annotations/train PNG masks (read-only, no modification). It does NOT decode
images. Approve the mask-decode permission explicitly before running.

Goal: prove, at the pixel level for the train split, (a) every mask is single-channel
indexed with values in {0..115} and no 255; (b) the mask=category_id+1 mapping holds per
image (mask nonzero values == {category_id+1} for that image's annotation category ids),
quantifying any exceptions (prior full-dataset report saw ~0.15%).

Allowed to open: annotations/train PNG masks (pixel read only); annotation_train.json
(image_id->category_id and file_name); reuse prior reports. Do NOT decode images. Do NOT
open val/test or Metadata.csv. Do NOT modify any file. Do NOT commit.

Before writing the report, show me your plan and wait for `go`.

Checks:
1. For each annotations/train/<stem>.png: mode/ndim (expect L/2D), unique values subset
   of {0..115}, no 255, contiguity not required per-image.
2. Aggregate label histogram across train masks; min/max non-ignore; presence of 255.
3. mask=category_id+1: for each image, compare mask nonzero index set to
   {cat_id+1 for its annotations}; count consistent vs exceptions; list <=20 exceptions.
4. Cross-check counts vs dataset_report (full-dataset) values.

Create or update only: reports/train_mask_value_audit.md
Separate: (1) per-mask label findings; (2) mask=category_id+1 result (now proven or
quantified); (3) prior report agreement; (4) NEED_TO_CONFIRM; (5) next step (val split).

After finishing show: PASS/FLAG; label-range result; 255 result; mask=category_id+1
consistency %; files created/changed; git status; exact next prompt (val split). Stop.
```

---

## 7. Required separations (recap)
**(1) Three-way reconciliation** → §1. **(2) Pairing simulation** → §2. **(3) Consolidated B16-1/2/3** →
§3. **(4) Prior reports** → §4. **(5) NEED_TO_CONFIRM** → §5. **(6) Next prompt** → §6.

_End of B16-4. Only `images/train` + `annotations/train` (listings) and `annotation_train.json` (stems)
were read; only this report was written; nothing committed._
