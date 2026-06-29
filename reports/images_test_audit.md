# images/test Validation (B16-10)

**Scope:** read-only, **filename-level** validation of `images/test` cross-checked against the 1,561
`file_name` entries in `annotation_test.json`. Two reads only: a **directory listing** of `images\test`
(via `os.scandir`, **no pixel decode**) and `json.load` of `annotation_test.json` **solely** to extract
`file_name`s. No image/mask decoded. No other dataset file opened. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\images\test`.
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** the dataloader pairs `images/test/<stem>.jpg` ↔ `annotations/test/<stem>.png` by stem
> (B16-0); the JSON is not used for loading. This validates the **test image inventory**, the folder↔JSON
> name agreement (closes B16-9 NTC-2), and the presence of the B16-9 FLAG-F image. Masks
> (`annotations/test`) are validated in B16-11.

---

## 0. PASS / FLAG verdict — **PASS (clean, 0 flags)**

`images/test` contains **exactly 1,561 `.jpg` files**, nothing else — no subdirectories, hidden, stray, or
alternate-extension files, no case anomalies, no duplicate stems. It is an **exact match** to
`annotation_test.json`'s `file_name` set at **both** full-filename (case-sensitive) and stem level (0
folder-only, 0 JSON-only). The B16-9 FLAG-F image **`apple_black_rot_google_0001.jpg` is present**.

---

## 1. WHAT `images/test` CONTAINS

| Property | Value |
|---|---|
| Directory exists | ✅ True |
| Total entries | **1,561** |
| Regular files | **1,561** |
| Subdirectories / hidden / other | 0 / 0 / 0 |

### Extension breakdown
| Extension | Count |
|---|---|
| `.jpg` (lowercase, exact case) | **1,561** |
| `.jpeg` / `.JPG` / `.png` / any non-`.jpg` | 0 / 0 / 0 / 0 |

- `.jpg` count **1,561 == 1,561 expected** → **MATCH**. All lowercase `.jpg` → loader `glob("*.jpg")`
  matches every file with no case/extension gap.

### Duplicates
| Check | Result |
|---|---|
| Case-insensitive duplicate full filenames | **0** |
| `.jpg` total / distinct stems | 1,561 / 1,561 |
| Duplicate `.jpg` stems | **0** |
| Duplicate stems across all extensions (ci) | **0** |

---

## 2. WHAT THE `annotation_test.json` CROSS-CHECK CONFIRMS

| Comparison | Folder | JSON | JSON-only | Folder-only | Verdict |
|---|---|---|---|---|---|
| Full filename (case-sensitive) | 1,561 | 1,561 | **0** | **0** | exact match |
| Stem (loader pairing key) | 1,561 | 1,561 | **0** | **0** | exact match |
| Stem intersection | — | — | — | — | **1,561** |
| Case-only diff | — | — | — | — | **0** |

- The 1,561 `file_name`s in `annotation_test.json` (validated clean in B16-9) correspond **1:1, exact
  case**, to the 1,561 files in `images/test`. **B16-9 NTC-2 (file_name ↔ folder) is CLOSED.**

---

## 3. `apple_black_rot_google_0001.jpg` PRESENCE (B16-9 FLAG-F)

| Check | Result |
|---|---|
| Present in `images/test` (exact case) | ✅ **True** |
| Present (case-insensitive) | ✅ True |
| Present in `annotation_test.json` `file_name`s | ✅ True |

- The image that has **zero COCO annotations** in `annotation_test.json` (FLAG-F) **does have an image
  file** in `images/test`. Because the dataloader pairs by image+mask file (not JSON annotations), this
  image will **not** be dropped — provided its mask exists (to be confirmed in B16-11). What its mask
  *contains* (all-background vs a disease region absent from the COCO polygons) remains a decode-gated
  question (NTC-F). The missing COCO annotations matter only for COCO-based instance evaluation, not for
  the semantic-segmentation loader.

---

## 4. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source | Relation |
|---|---|---|
| Test split = 1,561 (folder authoritative) | `dataset_report.json` `split.counts` | ✅ matches the 1,561 files here |
| Whole-dataset images `.jpg` only (7,774) | `dataset_report.json` `extensions` | ✅ consistent — test slice all `.jpg` |
| Zero cross-split id/stem overlap | `dataset_report.json` `overlaps_counts` | ✅ test images distinct from train/val |
| All images readable; image↔mask EXIF-aware dim parity (0 mismatch) | `dataset_report` | covers decode-level checks → **not repeated** |
| Loader pairs by stem; `.jpg`-only glob | `dataset_code_expectation_audit.md` | test shows no case/ext gap |

---

## 5. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `annotations/test` mask inventory & image↔mask pairing | Mask folder not opened (B16-11). Need 1,561 `.png`, stems == test stems, no stray/dupes. |
| **NTC-F** | The 0-annotation image's mask (`apple_black_rot_google_0001.png`) | Confirm the mask file exists (B16-11); its pixel content needs the decode-gated mask check. |
| **NTC-2** | `mask = category_id + 1` mapping | Still **inferred** — needs annotation↔mask value check (decode). |
| **NTC-3** | Test JSON orphan images (B16-9 FLAG-A/E) | Identifiable only by `file_name` across files; ids are local. |
| **NTC-4** | JSON `width`/`height` vs real dims | Decode-gated; prior report verified readability + dim parity. |
| **NTC-5** | `reduce_zero_label` / explicit background | Open since B16-0. |

---

## 6. WHAT B16-11 SHOULD CHECK NEXT (`annotations/test`)

1. **Inventory (filename-only, no decode):** count `annotations/test` files; confirm **1,561**; all `.png`;
   flag `.PNG`/other ext, hidden, subdirs, stray.
2. **Duplicates:** no duplicate `.png` filenames or stems.
3. **Mask stems vs test stems:** set of `annotations/test` `.png` stems == the 1,561 `annotation_test.json`
   `file_name` stems (images/test == JSON proven here, so JSON set is canonical — no need to re-list
   `images/test`); report any stem without a mask and any orphan mask.
4. **Confirm FLAG-F mask present:** verify `apple_black_rot_google_0001.png` exists in `annotations/test`
   (closes NTC-F at the filename level; pixel content stays deferred).
5. **Defer pixel/label decode:** keep `mask = category_id + 1` **inferred**; mask label values from
   `dataset_report` as prior evidence.
6. Output only `reports/annotations_test_audit.md`; no decode.

---

## 7. Required separations (recap)
**(1) Contains** → §1. **(2) JSON cross-check** → §2. **(3) FLAG-F presence** → §3. **(4) Prior reports**
→ §4. **(5) NEED_TO_CONFIRM** → §5. **(6) B16-11 next (`annotations/test`)** → §6.

_End of B16-10. Only `images/test` (listing) and `annotation_test.json` (file_name extract) were read; only
this report was written; nothing committed._
