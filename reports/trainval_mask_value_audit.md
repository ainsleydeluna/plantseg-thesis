# Train & Val Mask VALUE Check (B16-13b, decode-gated)

**Scope:** read-only, **decode-gated** validation of **`annotations/train` + `annotations/val` mask pixels
only** (explicitly approved). Decoded all 5,367 train + 846 val `*.png` via `PIL.Image` + `numpy` (raw
indices, no RGB/palette expansion); read `annotation_train.json` / `annotation_val.json` for
`file_name → image_id → category_id`. **Test masks NOT re-decoded; no RGB images; `Metadata.csv` not
opened.** Per the unfiltered-`annotations[]` finding, **orphan annotations** (image_id ∉ that split's
`images[]`) were **excluded** from the per-image mapping check by construction (train 14,647 excluded; val
2,467 excluded). No files modified.

**Interpreter:** project Anaconda Python (`C:\Users\admin\anaconda3\python.exe`; numpy 2.1.3, Pillow 11.1.0),
`PYTHONIOENCODING=utf-8`. No install. **Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

---

## 0. PASS / FLAG verdict — **PASS (train + val)** — `mask=category_id+1` **100% exact**, mapping now **proven dataset-wide**

Every train and val mask is single-channel `L`/2-D, values **0–115**, **255 absent**, 0 unreadable.
`mask = category_id + 1` holds **exactly for 100%** of images in **both** splits (train 5,367/5,367; val
846/846), with **0 extra and 0 missing** labels. Combined with B16-13a (test), the mapping is now
**empirically proven against COCO `category_id` across all 7,774 dataset images** (0 exceptions among
annotated images). No FLAG-F-type 0-annotation images exist in train or val.

---

## 1. TRAIN — label range, 255, mapping

| Property | Result |
|---|---|
| Masks decoded / expected | **5,367 / 5,367** (0 unreadable) |
| Mode / ndim | `{'L': 5367}` / `{2: 5367}` (single-channel, 2-D) |
| Distinct values | **116** · min/max non-ignore **0 / 115** · all within 0–115 ✅ (0 violations) |
| **255** | **absent** (0 masks) |
| Labels 0–115 present | **116 of 116 — none absent** (train covers **all** 115 disease classes + bg) |
| Label 0 (background) | in **5,366 / 5,367** masks (**one** train mask has no background pixel) |
| Label 115 | in **39** masks |
| `mask=category_id+1` exact | **5,367 / 5,367 (100.0000%)** — extra 0, missing 0 |
| Distinct category_ids per image | `{1: 5367}` (every image single-disease; **no** 0-annotation images) |

## 2. VAL — label range, 255, mapping

| Property | Result |
|---|---|
| Masks decoded / expected | **846 / 846** (0 unreadable) |
| Mode / ndim | `{'L': 846}` / `{2: 846}` (single-channel, 2-D) |
| Distinct values | **115** · min/max non-ignore **0 / 115** · all within 0–115 ✅ (0 violations) |
| **255** | **absent** (0 masks) |
| Labels 0–115 present | **115 of 116 — absent: `[69]`** (val lacks disease class 68 = value 69) |
| Label 0 (background) | in **846 / 846** masks |
| Label 115 | in **6** masks |
| `mask=category_id+1` exact | **846 / 846 (100.0000%)** — extra 0, missing 0 |
| Distinct category_ids per image | `{1: 846}` (every image single-disease; no 0-annotation images) |

## 3. PER-SPLIT CLASS / LABEL COVERAGE

| Split | Distinct mask values | Absent (of 0–115) | Disease classes covered | Note |
|---|---|---|---|---|
| **train** | 116 | none | **all 115** (+ bg) | full coverage |
| **val** | 115 | `[69]` → class **68** | 114 | class 68 has **no val** support |
| **test** (B16-13a) | 115 | `[42]` → class **41** | 114 | class 41 has **no test** support |

- **Every disease class appears in train** (good for training). **Val** lacks class 68; **test** lacks
  class 41 → their **per-class val/test IoU is undefined** (no support); the metrics step must handle
  absent-class NaN (carry-forward note from B16-13a, now extended to val class 68).
- **One-disease-per-image** holds dataset-wide (`distinct_category_ids_per_image = {1: N}` for every split's
  annotated images).

## 4. AGREEMENT WITH B16-13a & `dataset_report`

| Quantity | This audit | Prior | Agreement |
|---|---|---|---|
| Masks single-channel `L`, 2-D, 0–115, 255 absent | train+val ✅ | `dataset_report` (full) | ✅ |
| Label-115 mask count — train | **39** | `dataset_report` `splits_present.train = 39` | ✅ **exact** |
| Label-115 mask count — val | **6** | `dataset_report` `splits_present.val = 6` | ✅ **exact** |
| Label-115 mask count — test | 11 (B16-13a) | `dataset_report test = 11` | ✅ (sum 39+6+11 = **56** = report total) |
| Label-0 absent in exactly 1 mask | **train 5,366/5,367**, val 846/846, test 1,561/1,561 → 7,773/7,774 | `dataset_report label_0 = 7,773/7,774` | ✅ **exact** (the 1 all-disease mask is in train) |
| `mask=category_id+1` vs **COCO** | **100%** exact (train+val+test annotated) | `dataset_report` 99.85% vs **Metadata.csv Index** | ⚠️ different reference → NTC-7 |

> The mask↔COCO agreement is **perfect** (0 exceptions across all 7,774 images), while `dataset_report`
> found 12 exceptions **against Metadata.csv `Index`**. This strongly implies the **masks are correct** and
> the 12 cases are a **Metadata `Index` ↔ COCO `category_id` disagreement** — to be confirmed when
> `Metadata.csv` is validated (B16-14).

### Whole-dataset mask-value roll-up (all splits, now decoded)
| | train | val | test | total |
|---|---|---|---|---|
| masks | 5,367 | 846 | 1,561 | **7,774** |
| single-channel `L`, 0–115, no 255 | ✅ | ✅ | ✅ | ✅ |
| `mask=category_id+1` exact (annotated) | 5,367 | 846 | 1,560 | **7,773 / 7,773 = 100%** |
| 0-annotation images (mask still valid) | 0 | 0 | 1 (FLAG-F) | 1 |
| label-115 masks | 39 | 6 | 11 | 56 |

## 5. REMAINING `NEED_TO_CONFIRM`

| ID | Item | Status |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | **RESOLVED / PROVEN** vs COCO `category_id` across all 7,774 images (0 exceptions). |
| **NTC-7** | The 12 `dataset_report` exceptions = Metadata `Index` ↔ COCO `category_id` disagreement? | **Open** — needs `Metadata.csv` (B16-14). Masks shown correct vs COCO, so Metadata is the likely odd source. |
| **NTC-8** | Absent-class support: class 41 (no test), class 68 (no val) | **Open** — per-class val/test IoU undefined for these; metrics step must handle NaN. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | **Open** — background = value 0 empirically; Metadata names Index 0 as a disease (mask=Index+1 shifts it off 0). Revisit in B16-14. |
| **NTC-F** | FLAG-F mask content | **RESOLVED** in B16-13a (apple black rot; COCO polygons absent only). |

## 6. EXACT NEXT RECOMMENDED PROMPT (B16-14 — `Metadata.csv` validation + reconciliation)

```
B16-14 — Validate Metadata.csv + reconcile Metadata Index vs COCO category_id. Read-only. Plan-gated.

Goal: validate Metadata.csv on its own terms and reconcile the 12 dataset_report mask-shift
exceptions: are they a Metadata `Index` vs COCO `category_id` disagreement (masks already proven
correct vs COCO in B16-13a/b)? Also address the background/reduce_zero_label convention (NTC-5).

Allowed to open/read:
- C:\Users\admin\plantseg_data\plantseg\Metadata.csv
- annotation_train.json, annotation_val.json, annotation_test.json  (image file_name -> category_id)
Reuse prior reports (dataset_report.{md,json}, test_mask_value_audit.md, trainval_mask_value_audit.md,
the annotation_*_audit.md). Do NOT decode images or masks (masks already audited). Do NOT modify
anything. Do NOT commit.

Before writing the report, show me your plan and wait for `go`.

Checks:
1. Parse Metadata.csv: row count == 7774; columns; Index range/distinct (expect 0..114, 115);
   Split column values/counts (expect train 5367 / val 846 / test 1561); Name uniqueness.
2. Per image (by Name/stem), compare Metadata Index vs that image's COCO category_id (from the
   split JSONs). Count agreements/disagreements; list <=20 disagreements. Confirm whether the 12
   dataset_report shift exceptions fall in this disagreement set.
3. Split-column vs folder split agreement (reuse prior split counts; no re-listing needed).
4. Background/naming: confirm Metadata Index is 0-based disease id (0=apple black rot), so mask
   value 0 = background is implicit (not a named Metadata class) -> inform reduce_zero_label (NTC-5).
5. License/Plant/Disease sanity (nulls, blank, distinct counts) — stats only.

Create or update only: reports/metadata_csv_audit.md
Separate: (1) Metadata.csv structure/counts; (2) Index vs COCO category_id reconciliation (NTC-7);
(3) split-column agreement; (4) background/reduce_zero_label finding (NTC-5); (5) NEED_TO_CONFIRM;
(6) next step (close out B16 dataset audit with a consolidated summary, or proceed to modeling).

After finishing show: PASS/FLAG; row count + Index range; split-column counts; Index-vs-COCO
disagreement count (and whether it explains the 12); background/reduce_zero_label finding;
files created/changed; git status; exact next prompt. Stop after B16-14.
```

---

## 7. Required separations (recap)
**(1) Train** → §1. **(2) Val** → §2. **(3) Per-split coverage** → §3. **(4) `dataset_report` agreement**
→ §4. **(5) NEED_TO_CONFIRM** → §5. **(6) Next prompt** → §6.

_End of B16-13b. Only `annotations/train` + `annotations/val` masks (pixel-decoded) and the two split JSONs
were read; only this report was written; nothing committed; no memory/`~/.claude` writes._
