# annotation_val.json Validation (B16-5)

**Scope:** read-only validation of **`annotation_val.json` only**, via programmatic parsing (stdlib
`json`). No other dataset file was opened. No image/mask was decoded. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\annotation_val.json` — 7.58 MB (7,943,765 bytes),
mtime 2025-11-26. **Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** the dataloader does **not** read this JSON (folder/stem pairing per B16-0). This is an
> independent val-index integrity check, with **no effect on training/loading** as currently coded. It
> begins the validation-split checks (train was signed off in B16-1…4).

---

## 0. PASS / FLAG verdict — **PASS (val image index sound) · 1 FLAG (same pattern as train) · minor notes**

`annotation_val.json` is **valid COCO**, the **val image list is clean and complete** (846 images, all
referenced, no dups/nulls/malformed), and **all 115 disease categories (0–114) appear in val**. The same
hygiene issue as train recurs (FLAG-A: `annotations[]` not filtered to the split), and a **new structural
finding**: each split JSON uses its **own local image-id space** (train 1…7916, val 1…1247) — they are
**not** a shared global numbering.

| ID | Finding | Severity |
|---|---|---|
| **FLAG-A** | `annotations[]` **not filtered to val**: **2,467 / 8,927 (27.6%)** annotations reference an `image_id` absent from this file's `images[]` (401 distinct orphan ids). Matches train's 27.7%. **No loader impact.** | Medium (index hygiene) |
| **FLAG-E (new)** | **Per-file local id space.** `images[].id` and `image_id` here run **1…1247**, vs train's **1…7916**. Image ids are **not comparable across split JSONs** → cross-split reconciliation must use `file_name`/stem, **not** ids. Refines B16-1 NTC-4. | Medium (cross-split) |
| NOTE-B | `categories[]` **empty** → no names, no explicit background (names in `Metadata.csv`). | Low |
| NOTE-C | `info`/`licenses` placeholder/empty; `date_captured` empty; `license`=1 for all. | Info |
| INFER-D | `num_classes=116` **not proven here** — JSON proves 115 categories (0–114); 116 needs `mask=category_id+1` background (kept **inferred**). | — |

---

## 1. WHAT `annotation_val.json` CONTAINS

### 1.1 Validity & schema
Valid JSON, loaded in 0.06 s. Root = `dict`, standard COCO:

| Key | Type | Length |
|---|---|---|
| `info` | dict | 6 |
| `licenses` | list | 1 |
| `images` | list | **846** |
| `annotations` | list | **8,927** |
| `categories` | list | **0 (empty)** |

`info` = `{"description":"","url":"","version":"1.0","year":2024,"contributor":"","date_created":""}` ·
`licenses` = `[{"id":1,"name":"","url":""}]` (identical placeholder shape to train).

### 1.2 Counts & image-count check
- **images = 846** · **annotations = 8,927** · **categories = 0**.
- **846 == 846** (prior folder-authoritative val count) → **MATCH**.

### 1.3 Duplicates
| Field | Total | Distinct | Duplicates | Nulls |
|---|---|---|---|---|
| `image.id` | 846 | 846 | **0** | 0 |
| `image.file_name` | 846 | 846 | **0** | 0 |
| `annotation.id` | 8,927 | 8,927 | **0** | 0 |

### 1.4 Record fields & types (100% present; all ids `int`)
- **Image record:** `id`(int), `license`(int), `file_name`(str), `height`(int), `width`(int),
  `date_captured`(str, empty). Required `{id,file_name,width,height}` present in all 846. Extra: `license`,
  `date_captured`. Example: `{id:1, file_name:"apple_black_rot_28.jpg", height:500, width:789}`.
- **Annotation record:** `id, image_id, category_id, segmentation, area, bbox, iscrowd` — all present in all
  8,927. `segmentation` = **polygon (list)** 100%; `iscrowd` = **0** 100%.
  Example: `{id:1, image_id:1, category_id:0, area:1315.0, bbox:[511,454,48,44], iscrowd:0}`.

### 1.5 Null / malformed / suspicious
| Check | Result |
|---|---|
| image `width`/`height` null / ≤0 / non-int | 0 / 0 |
| `category_id` null | 0 |
| `bbox` malformed | **0** |
| `area` ≤0 / non-numeric | 0 |
| images with 0 annotations | **0** (all 846 covered) |
| annotations per image | min 1 · max 129 · mean 7.64 |
| **orphan annotations** | **2,467 (27.6%)** → §1.8 |

### 1.6 Category IDs & label-index implication
- `category_id`: **distinct 115, min 0, max 114, contiguous** ✓; **all 115 categories present in val**
  (0 missing from 0–114). Most common `38(516), 86(417), 3(283), 97(256), 48(244)`; rarest down to `68(2)`.
- `categories[]` **empty** → no names / id→name map.
- **num_classes:** establishes **115 disease classes (0–114)**; reaching **116** needs `mask=category_id+1`
  + background=0 — **INFERRED**, not provable from this instance-annotation file.

### 1.7 Background / ignore convention
- **Explicit background category:** **none** (categories empty).
- **Ignore / 255:** absent — no `category_id==255`, no `ignore` key; `iscrowd` (all 0) is the COCO crowd
  flag, not an ignore-index. Consistent with "255 = preprocessing pad only."

### 1.8 FLAG-A & FLAG-E — `annotations[]` unfiltered, local id space (characterized)
| Quantity | **val** | (train, B16-1) |
|---|---|---|
| `images[].id` | 846 distinct, **min 1, max 1247, non-contiguous** | 5,367 distinct, 1…7915, non-contiguous |
| Distinct `image_id` referenced by `annotations` | **1,247** (contiguous 1…1247) | 7,916 (contiguous 1…7916) |
| Split images referenced by ≥1 ann | **846 / 846** (all) | 5,367 / 5,367 |
| Distinct **orphan** image_ids | **401** (range 9…1245) | 2,549 |
| **Orphan annotations** | **2,467 / 8,927 (27.6%)** | 14,647 / 52,898 (27.7%) |
| Union (images[] ∪ referenced) | **1,247** | 7,916 |

**Interpretation:** `images[]` is correctly filtered to the **val** subset, but `annotations[]` carries a
larger, **contiguous local** id space (1…1247) — so ~27.6% of annotations reference non-val images. The
**id numbering is per-file** (val 1…1247 ≠ train 1…7916 ≠ dataset total 7,774), so **image ids are not
comparable across split JSONs**. The orphan images therefore **cannot be identified by id across files** —
only by `file_name` (deferred). **Impact:** none on the current pipeline; matters only for COCO-based
downstream use, which must filter `annotations` to `image_id ∈ images[].id` **within each file**.

---

## 2. WHAT PRIOR REPORTS ALREADY CONFIRMED

| This audit (val JSON) | Prior report | Agreement |
|---|---|---|
| 846 val images | folder/json/csv val = 846 | ✅ exact |
| category_id 0–114 (115 distinct, contiguous), all present | report COCO scan: cat 0–114 distinct 115 | ✅ consistent |
| `categories[]` empty | report `categories_list_len = 0` | ✅ exact |
| No 255 / ignore in JSON | report: 255 absent in raw masks (pad-only) | ✅ consistent |
| `annotations[]` unfiltered (27.6%) | matches train pattern (B16-1 FLAG-A) | ✅ same pattern |
| val file_name e.g. `apple_black_rot_28.jpg` (≠ train's `_1/_11/…`) | report: zero cross-split id/stem overlap | ✅ consistent (different images) |

---

## 3. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why it can't be closed here |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | This JSON has no masks; **inferred**. Confirm at the annotation↔mask value check. |
| **NTC-2** | `file_name` ↔ actual `images/val/` files | Folder not opened (B16-6). Need exact stem-set equality (846). |
| **NTC-3** | What the orphan images actually are | Identifiable only by `file_name` across the other split files/folders (cross-split step); ids are local, not global. |
| **NTC-4** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; not checked. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0; JSON has no explicit background. |

---

## 4. WHAT B16-6 SHOULD CHECK NEXT (`images/val`)

1. **Inventory (filename-only, no decode):** list `images/val/*.jpg`; count == **846**; extensions all
   `.jpg`; no duplicate stems; no stray/non-jpg/hidden/subdir.
2. **Cross-ref to this JSON (NTC-2):** set of `images/val/` stems **==** set of the 846 `file_name` stems
   from `annotation_val.json` (report folder-only / JSON-only).
3. **Mask-pairing preview (filename-only):** every `images/val/<stem>.jpg` has
   `annotations/val/<stem>.png` (loader pairing rule) — filename check only (or defer to B16-7).
4. **Defer decode:** dimension verification is decode-gated; prior `dataset_report` already verified
   readability + image/mask dim parity.
5. **Keep INFER-D / NTC-1 open:** a folder check cannot prove `mask = category_id + 1`.
6. Output only `reports/images_val_audit.md`; no pixel decode.

---

## 5. Required separations (recap)
**(1) Contains** → §1. **(2) Prior reports** → §2. **(3) NEED_TO_CONFIRM** → §3. **(4) B16-6 next
(`images/val`)** → §4.

_End of B16-5. Only `annotation_val.json` was read; only this report was written; nothing committed._
