# annotation_test.json Validation (B16-9)

**Scope:** read-only validation of **`annotation_test.json` only**, via programmatic parsing (stdlib
`json`). No other dataset file was opened. No image/mask was decoded. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\annotation_test.json` — 13.32 MB (13,964,698 bytes),
mtime 2025-11-26. **Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** the dataloader does **not** read this JSON (folder/stem pairing per B16-0). This is an
> independent test-index integrity check, with **no effect on training/loading** as currently coded. It
> begins the test-split checks (train signed off B16-1…4; val signed off B16-5…8).

---

## 0. PASS / FLAG verdict — **PASS (test image index sound) · 2 recurring FLAGs · 1 NEW flag · minor notes**

`annotation_test.json` is **valid COCO**, the **test image list is clean and complete** (1,561 images, no
dups/nulls/malformed), and **all 115 disease categories (0–114) appear in test**. The two split-array
flags recur (FLAG-A unfiltered `annotations[]`; FLAG-E per-file local id space). **New for test:** exactly
**one** image has **zero** COCO annotations (FLAG-F).

| ID | Finding | Severity |
|---|---|---|
| **FLAG-F (new)** | **1 test image has 0 annotations** — `apple_black_rot_google_0001.jpg` (id=8, 499×332). 1,560/1,561 images referenced; `annotations_per_image` min=0. Train/val had none. **No expected loader impact** (loader pairs by folder, not JSON), but the image's mask content is unknown until the decode-gated mask check. | Medium |
| **FLAG-A** | `annotations[]` **not filtered to test**: **4,466 / 15,569 (28.7%)** annotations reference an `image_id` absent from `images[]` (734 distinct orphan ids). Matches train 27.7% / val 27.6%. **No loader impact.** | Medium |
| **FLAG-E** | **Per-file local id space**: test ids run **1…2295**, vs val 1…1247 and train 1…7916 → image ids **not comparable across split JSONs**; cross-split reconciliation must use `file_name`. | Medium |
| NOTE-B | `categories[]` **empty** → no names, no explicit background (names in `Metadata.csv`). | Low |
| NOTE-C | `info`/`licenses` placeholder/empty; `date_captured` empty; `license`=1 for all. | Info |
| INFER-D | `num_classes=116` **not proven here** — 115 categories (0–114); 116 needs `mask=category_id+1` background (kept **inferred**). | — |

---

## 1. WHAT `annotation_test.json` CONTAINS

### 1.1 Validity & schema
Valid JSON, loaded in 0.12 s. Root = `dict`, standard COCO:

| Key | Type | Length |
|---|---|---|
| `info` | dict | 6 |
| `licenses` | list | 1 |
| `images` | list | **1,561** |
| `annotations` | list | **15,569** |
| `categories` | list | **0 (empty)** |

`info` = `{"description":"","url":"","version":"1.0","year":2024,...}` · `licenses` =
`[{"id":1,"name":"","url":""}]` (identical placeholder shape to train/val).

### 1.2 Counts & image-count check
- **images = 1,561** · **annotations = 15,569** · **categories = 0**.
- **1,561 == 1,561** (prior folder-authoritative test count) → **MATCH**.

### 1.3 Duplicates
| Field | Total | Distinct | Duplicates | Nulls |
|---|---|---|---|---|
| `image.id` | 1,561 | 1,561 | **0** | 0 |
| `image.file_name` | 1,561 | 1,561 | **0** | 0 |
| `annotation.id` | 15,569 | 15,569 | **0** | 0 |

### 1.4 Record fields & types (100% present; all ids `int`)
- **Image record:** `id`(int), `license`(int), `file_name`(str), `height`(int), `width`(int),
  `date_captured`(str, empty). Required `{id,file_name,width,height}` present in all 1,561. Extra:
  `license`, `date_captured`. Example: `{id:1, file_name:"apple_black_rot_143.jpg", height:172, width:268}`.
- **Annotation record:** `id, image_id, category_id, segmentation, area, bbox, iscrowd` — all present in all
  15,569. `segmentation` = **polygon (list)** 100%; `iscrowd` = **0** 100%.
  Example: `{id:1, image_id:1, category_id:0, area:6634.5, bbox:[67,51,100,114], iscrowd:0}`.

### 1.5 Null / malformed / suspicious
| Check | Result |
|---|---|
| image `width`/`height` null / ≤0 / non-int | 0 / 0 |
| `category_id` null | 0 |
| `bbox` malformed | **0** |
| `area` ≤0 / non-numeric | 0 |
| **images with 0 annotations** | **1** → `apple_black_rot_google_0001.jpg` (id=8) — FLAG-F |
| annotations per image | min **0** · max 112 · mean 7.11 |
| **orphan annotations** | **4,466 (28.7%)** → §1.8 |

### 1.6 Category IDs & coverage (#9/#10)
- `category_id`: **distinct 115, min 0, max 114, contiguous** ✓.
- **ALL 115 categories (0–114) present in test** (absent count = 0). Most common `38(961), 110(479),
  48(464), 86(437), 96(424)`; rarest down to `68(5), 41(5)`.
- `categories[]` **empty** → no names / id→name map.
- **num_classes:** establishes **115 disease classes (0–114)**; reaching **116** needs `mask=category_id+1`
  + background=0 — **INFERRED**, not provable from this instance-annotation file.

### 1.7 Background / ignore convention
- **Explicit background category:** **none** (categories empty).
- **Ignore / 255:** absent — no `category_id==255`, no `ignore` key; `iscrowd` (all 0) is the COCO crowd
  flag, not an ignore-index.

### 1.8 FLAG-A & FLAG-E — `annotations[]` unfiltered, local id space
| Quantity | **test** | val (B16-5) | train (B16-1) |
|---|---|---|---|
| `images[].id` | 1,561 distinct, **1…2295, non-contiguous** | 846, 1…1247 | 5,367, 1…7915 |
| Distinct `image_id` referenced | **2,294** (1…2295) | 1,247 | 7,916 |
| Split images referenced ≥1 ann | **1,560 / 1,561** (one not referenced → FLAG-F) | 846/846 | 5,367/5,367 |
| Distinct orphan image_ids | **734** (11…2294) | 401 | 2,549 |
| **Orphan annotations** | **4,466 / 15,569 (28.7%)** | 2,467 (27.6%) | 14,647 (27.7%) |
| Union (images[] ∪ referenced) | **2,295** | 1,247 | 7,916 |

**Interpretation:** `images[]` is filtered to the **test** subset, but `annotations[]` carries a larger,
contiguous **local** id space (1…2295) → ~28.7% orphan annotations. The id numbering is **per-file** (test
1…2295 ≠ val 1…1247 ≠ train 1…7916 ≠ dataset 7,774), so image ids are **not comparable across split
JSONs**; orphan images are identifiable only by `file_name` (deferred). **Impact:** none on the current
pipeline; matters only for COCO-based downstream use (filter `annotations` to `image_id ∈ images[].id`
within each file).

---

## 2. WHAT PRIOR REPORTS ALREADY CONFIRMED

| This audit (test JSON) | Prior report | Agreement |
|---|---|---|
| 1,561 test images | folder/json/csv test = 1,561 | ✅ exact |
| category_id 0–114, all 115 present | report COCO scan: cat 0–114 distinct 115 | ✅ consistent |
| `categories[]` empty | report `categories_list_len = 0` | ✅ exact |
| No 255 / ignore in JSON | report: 255 absent in raw masks (pad-only) | ✅ consistent |
| `annotations[]` unfiltered (28.7%) | matches train/val pattern | ✅ same pattern |
| test file_name e.g. `apple_black_rot_143.jpg` | report: zero cross-split overlap | ✅ consistent (distinct images) |

---

## 3. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why it can't be closed here |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | This JSON has no masks; **inferred**. Confirm at the annotation↔mask value check. |
| **NTC-2** | `file_name` ↔ actual `images/test/` files (incl. id=8 image) | Folder not opened (B16-10). Need exact stem-set equality (1,561). |
| **NTC-F** | What the 0-annotation image (`apple_black_rot_google_0001.jpg`) is | Confirm its image+mask files exist (B16-10/B16-11); whether its mask is all-background or holds a disease region needs the decode-gated mask check. |
| **NTC-3** | What the orphan images actually are | Identifiable only by `file_name` across files; ids are local. |
| **NTC-4** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; not checked. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0. |

---

## 4. WHAT B16-10 SHOULD CHECK NEXT (`images/test`)

1. **Inventory (filename-only, no decode):** list `images/test/*.jpg`; count == **1,561**; extensions all
   `.jpg`; no duplicate stems; no stray/non-jpg/hidden/subdir.
2. **Cross-ref to this JSON (NTC-2):** set of `images/test/` stems **==** set of the 1,561 `file_name`
   stems from `annotation_test.json` (report folder-only / JSON-only).
3. **Confirm FLAG-F image present:** verify `apple_black_rot_google_0001.jpg` exists in `images/test`
   (its lack of COCO annotations should not remove the image/mask pair).
4. **Mask-pairing preview (filename-only):** every `images/test/<stem>.jpg` has
   `annotations/test/<stem>.png` (loader pairing rule) — or defer to B16-11.
5. **Keep INFER-D / NTC-1 open;** defer dimension/EXIF (covered by `dataset_report`).
6. Output only `reports/images_test_audit.md`; no pixel decode.

---

## 5. Required separations (recap)
**(1) Contains** → §1. **(2) Prior reports** → §2. **(3) NEED_TO_CONFIRM** → §3. **(4) B16-10 next
(`images/test`)** → §4.

_End of B16-9. Only `annotation_test.json` was read; only this report was written; nothing committed._
