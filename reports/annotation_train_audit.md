# annotation_train.json Validation (B16-1)

**Scope:** read-only validation of **`annotation_train.json` only**, via programmatic parsing (stdlib
`json`). No other dataset file was opened (`annotation_val/test.json`, `Metadata.csv`, and all
`images/`·`annotations/` folders were **not** touched). No image/mask was decoded. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\annotation_train.json` — 45.26 MB (47,461,502 bytes),
mtime 2025-11-26. **Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **B16-0 context:** the student dataloader does **not** read this JSON (it loads from folders + stem
> pairing). So this is an **independent dataset-index integrity check**, not a loader dependency check.
> Any anomaly here has **no effect on training/loading** as currently coded.

---

## 0. PASS / FLAG verdict — **PASS (train image index sound) · 1 notable FLAG · minor notes**

`annotation_train.json` is **valid COCO**, the **train image list is clean and complete** (5,367 images,
all referenced, no duplicates, no nulls, no malformed records), and **all 115 disease categories are
present**. One notable hygiene issue: the **`annotations` array is not filtered to the train split** (it
carries the full-dataset annotation list). This is harmless to the current pipeline but matters for any
future COCO-based use.

| ID | Finding | Severity |
|---|---|---|
| **FLAG-A** | `annotations[]` not filtered to train: **14,647 / 52,898 (27.7%)** annotations reference an `image_id` absent from this file's `images[]` (they belong to non-train images). | Medium (index hygiene; **no loader impact**) |
| **NOTE-B** | `categories[]` is **empty** → no class names and **no explicit background** category in the JSON (names live in `Metadata.csv`, not opened here). | Low |
| **NOTE-C** | `info`/`licenses` are placeholder/empty (`version 1.0`, `year 2024`, all other strings `""`); image `date_captured` empty, `license` = 1 for all. | Info |
| **INFER-D** | `num_classes=116` is **not proven** by this JSON. The JSON proves **115 categories (id 0–114)**; reaching 116 requires the **`mask = category_id + 1`** background convention, which remains **inferred** (to be confirmed only at the annotation↔mask check). | — |

---

## 1. WHAT `annotation_train.json` CONTAINS

### 1.1 Validity & top-level schema
- Valid JSON, loaded in 0.38 s. Root = `dict`. Standard COCO layout:

| Key | Type | Length |
|---|---|---|
| `info` | dict | 6 |
| `licenses` | list | 1 |
| `images` | list | **5,367** |
| `annotations` | list | **52,898** |
| `categories` | list | **0 (empty)** |

- `info` = `{"description":"","url":"","version":"1.0","year":2024,"contributor":"","date_created":""}`
- `licenses` = `[{"id":1,"name":"","url":""}]`

### 1.2 Counts
- **images = 5,367** · **annotations = 52,898** · **categories = 0** · other list: `licenses=1`.

### 1.3 Image count vs prior train split
- **5,367 == 5,367** (prior folder-authoritative train count) → **MATCH**.

### 1.4 Duplicates
| Field | Total | Distinct | Duplicates | Nulls |
|---|---|---|---|---|
| `image.id` | 5,367 | 5,367 | **0** | 0 |
| `image.file_name` | 5,367 | 5,367 | **0** | 0 |
| `annotation.id` | 52,898 | 52,898 | **0** | 0 |

### 1.5 Record fields & types (all values present in 100% of records; all ids are `int`)
- **Image record:** `id`(int), `license`(int), `file_name`(str), `height`(int), `width`(int),
  `date_captured`(str, empty). Required `{id,file_name,width,height}` present in all 5,367. Extra:
  `license`, `date_captured`. Example: `{id:1, file_name:"apple_black_rot_1.jpg", height:480, width:640}`.
- **Annotation record:** `id`, `image_id`, `category_id`, `segmentation`, `area`, `bbox`, `iscrowd` — all
  present in all 52,898. `segmentation` = **polygon (list)** for 100%; `iscrowd` = **0** for 100%.
  Example: `{id:1, image_id:1, category_id:0, area:22568.5, bbox:[571,267,162,182], iscrowd:0}`.

### 1.6 Null / malformed / suspicious
| Check | Result |
|---|---|
| image `width`/`height` null | 0 |
| image `width`/`height` ≤0 or non-int | 0 |
| `category_id` null | 0 |
| `bbox` malformed (len≠4 / negative / non-numeric / w≤0 / h≤0) | **0** |
| `area` ≤0 or non-numeric | 0 |
| images with 0 annotations | **0** (all 5,367 covered) |
| annotations per image | min 1 · max 135 · mean 7.13 |
| **orphan annotations** (`image_id` ∉ this file's `images[]`) | **14,647 (27.7%)** → see §1.8 |

### 1.7 Category IDs & label-index implication
- `annotation.category_id`: **distinct 115, min 0, max 114, contiguous** ✓.
- All 115 categories (0–114) are covered **even when restricted to train-image annotations** (none missing).
- Most common: `38(3044), 3(1801), 36(1778), 86(1663), 110(1481)`; rarest: `…95(34), 65(28), 68(19)`.
- `categories[]` is **empty** → the JSON supplies **no names and no id→name map** (those are in
  `Metadata.csv`).
- **num_classes implication:** the JSON establishes **115 disease classes (id 0–114)**. It does **not** by
  itself prove `num_classes=116`. 116 follows only under **`mask = category_id + 1`** (disease mask
  indices 1–115) **plus background = mask index 0** — an **inferred** convention, not provable from this
  instance-annotation file (it contains no masks). Kept **INFERRED** per instruction.

### 1.8 FLAG-A — `annotations` not filtered to the train split (characterized)
| Quantity | Value |
|---|---|
| `images[].id` | 5,367 distinct, **min 1, max 7915, NON-contiguous** (2,548 gaps in span) |
| First id gaps | 7, 12, 16, 20, 27, 28, 31, 43, 50, 51, 52, 53, 54, 57, 70 |
| Distinct `image_id` referenced by `annotations` | **7,916** (contiguous 1…7,916) |
| Train images referenced by ≥1 annotation | **5,367 / 5,367** (all) |
| Train images never referenced | 0 |
| Distinct **orphan** image_ids (referenced, not a train image) | **2,549** (range 7…7,916) |
| **Orphan annotations** | **14,647 / 52,898 (27.7%)** |

**Interpretation:** `images[]` is correctly filtered to the **train** subset, but each image keeps its
**global** id from a dataset-wide 1…7,916 numbering — hence the sparse, gappy id space. The
**`annotations` array is the full-dataset list** (references every id 1…7,916), so annotations belonging
to the ~2,549 non-train (val/test/other) images appear as "orphans" relative to this file's `images[]`.
The gaps in `images[].id` line up exactly with the orphan ids (e.g., 7, 12, 16…).

**Impact:** none on the current pipeline (loader ignores JSON; folder split is authoritative). It **does**
matter for any future COCO-based use (regenerating masks, detection/instance baselines, COCO eval): such
code must **filter annotations to `image_id ∈ images[].id`** or it would pull in non-train labels.

### 1.9 Background / ignore convention
- **Explicit background category:** **none** (categories list empty).
- **Ignore / 255:** not present — `category_id == 255`? **No**; no top-level `ignore` key; `iscrowd` is the
  COCO crowd flag (all 0), **not** an ignore-index. Consistent with prior finding that 255 is a
  **preprocessing pad**, never in released annotations.

---

## 2. WHAT PRIOR REPORTS ALREADY CONFIRMED (cross-reference)

From `reports/dataset_report.{md,json}` and `reports/dataset_code_expectation_audit.md`:

| This audit (JSON) | Prior report | Agreement |
|---|---|---|
| 5,367 train images | folder/json/csv train = 5,367 | ✅ exact |
| category_id 0–114 (115 distinct, contiguous) | COCO scan: cat_id 0–114, distinct 115 | ✅ exact |
| `categories[]` empty | report `categories_list_len = 0` | ✅ exact |
| No 255 / ignore in JSON | report: 255 absent in raw masks (pad-only) | ✅ consistent |
| 115 disease classes; bg via +1 → 116 | report `num_classes` RESOLVED_TOWARD_116 (mask max 115) | ✅ consistent (still inferred here) |
| Annotation id space up to 7,916 | report total **image files = 7,774** | ⚠️ id space (7,916) **exceeds** file count (7,774) by 142 → see NTC-3 |

---

## 3. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why it can't be closed here |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | This JSON has no masks; mapping is **inferred**. Confirm only at the annotation↔mask check (prior report saw 99.85% at pixel level, still <100%). |
| **NTC-2** | `file_name` ↔ actual `images/train/` files | Folder not opened (B16-2). Need exact stem-set equality between the 5,367 `file_name`s and `images/train/*.jpg`. |
| **NTC-3** | Id space (1…7,916, 7,916 distinct) vs 7,774 image files | Annotation id space exceeds the released file count by 142; reconciling needs val/test JSONs + folders (later cross-split step). |
| **NTC-4** | JSON `width`/`height` vs real (EXIF-corrected) image dims | No decode allowed here; prior report verified readability + image↔mask dim parity, but not vs JSON fields. |
| **NTC-5** | `reduce_zero_label` / background naming | Still open from B16-0 (NTC-1 there); JSON has no explicit background, so unchanged. |
| **NOTE** | FLAG-A orphan annotations | Confirmed as full-dataset annotation list; only matters if COCO annotations are used downstream. |

---

## 4. WHAT B16-2 SHOULD CHECK NEXT (`images/train`)

1. **Inventory (filename-only, no decode):** list `images/train/*.jpg`; count == **5,367**; extensions all
   `.jpg`; no duplicate stems; no stray/non-jpg files.
2. **Cross-ref to this JSON (NTC-2):** set of `images/train/` stems **==** set of the 5,367 `file_name`
   stems from `annotation_train.json` (report any folder-only or JSON-only stems).
3. **Mask-folder pairing preview (filename-only):** every `images/train/<stem>.jpg` has
   `annotations/train/<stem>.png` (this is the loader's actual pairing rule) — filename check only.
4. **Defer decode:** dimension verification (JSON `width`/`height` vs real EXIF-corrected size) is a
   separate **decode-gated** sub-step; prior `dataset_report` already verified readability + image/mask dim
   parity, so B16-2 can stay filename-level unless you want the decode pass.
5. **Keep INFER-D / NTC-1 open:** a folder check cannot prove `mask = category_id + 1`; leave inferred.
6. Output only `reports/images_train_audit.md`; no decode of pixels unless explicitly approved.

---

## 5. Required separations (recap)
**(1) What the JSON contains** → §1. **(2) What prior reports confirmed** → §2. **(3) NEED_TO_CONFIRM**
→ §3. **(4) What B16-2 checks next in `images/train`** → §4.

_End of B16-1. Only `annotation_train.json` was read; only this report was written; nothing committed._
