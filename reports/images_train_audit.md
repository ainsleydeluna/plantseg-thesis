# images/train Validation (B16-2)

**Scope:** read-only, **filename-level** validation of `images/train` cross-checked against the 5,367
`file_name` entries in `annotation_train.json`. Two reads only: a **directory listing** of `images\train`
(via `os.scandir`, **no pixel decode**) and `json.load` of `annotation_train.json` **solely** to extract
`file_name`s. No image/mask was decoded. No other dataset file opened. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\images\train`.
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** the dataloader pairs `images/train/<stem>.jpg` ↔ `annotations/train/<stem>.png` by stem
> (B16-0); the JSON is not used for loading. This step validates the **train image inventory** and the
> **folder↔JSON name agreement** (closes B16-1 NTC-2). Masks (`annotations/train`) are **not** validated
> here — that is B16-3.

---

## 0. PASS / FLAG verdict — **PASS (clean, 0 flags)**

`images/train` contains **exactly 5,367 `.jpg` files**, nothing else — no subdirectories, hidden files,
stray files, alternate extensions, case anomalies, or duplicate stems. The folder is an **exact match** to
`annotation_train.json`'s `file_name` set at **both** the full-filename (case-sensitive) and stem level:
zero folder-only, zero JSON-only.

---

## 1. WHAT `images/train` CONTAINS

| Property | Value |
|---|---|
| Directory exists | ✅ True |
| Total entries | **5,367** |
| Regular files | **5,367** |
| Subdirectories | **0** |
| Hidden / dot-files | **0** |
| Other (symlink/junction/etc.) | **0** |

### Extension breakdown
| Extension | Count |
|---|---|
| `.jpg` (lowercase, exact case) | **5,367** |
| `.jpeg` | 0 |
| `.JPG` (uppercase) | 0 |
| `.png` | 0 |
| any non-`.jpg` | 0 |

- `.jpg` count **5,367 == 5,367 expected** → **MATCH**.
- All extensions are lowercase `.jpg` (case-sensitive check agrees) → the loader's `glob("*.jpg")` matches
  **every** file with no case/extension gaps (resolves B16-0 FLAG-4 *for the train split*).

### Duplicates
| Check | Result |
|---|---|
| Case-insensitive duplicate full filenames | **0** |
| `.jpg` total / distinct stems | 5,367 / 5,367 |
| Duplicate `.jpg` stems | **0** |
| Duplicate stems across all extensions (case-insensitive) | **0** |

---

## 2. WHAT THE `annotation_train.json` CROSS-CHECK CONFIRMS

| Comparison | Folder | JSON | JSON-only | Folder-only | Verdict |
|---|---|---|---|---|---|
| Full filename (case-sensitive) | 5,367 | 5,367 | **0** | **0** | exact match |
| Stem (loader pairing key) | 5,367 | 5,367 | **0** | **0** | exact match |
| Stem intersection | — | — | — | — | **5,367** |
| Case-insensitive-but-not-exact (case-only diff) | — | — | — | — | **0** |

- The 5,367 `file_name`s in `annotation_train.json` (validated clean in B16-1) correspond **1:1, exact
  case**, to the 5,367 files in `images/train`.
- **B16-1 NTC-2 (file_name ↔ folder) is now CLOSED.** Every train image record points to a real file, and
  every train file is accounted for in the JSON.

---

## 3. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source | Relation to this step |
|---|---|---|
| Train split = 5,367 (folder authoritative) | `dataset_report.json` `split.counts` | ✅ matches the 5,367 files here |
| Whole-dataset images are `.jpg` only (7,774) | `dataset_report.json` `extensions` | ✅ consistent — train slice is all `.jpg` |
| All images readable; image↔mask EXIF-aware dim parity (0 mismatch) | `dataset_report` readability/dimensions | covers decode-level checks → **not repeated here** |
| 8 EXIF orientation transposes (images only) | `dataset_report` `exif` | decode-level; out of scope for this filename pass |
| Loader pairs by stem; `.jpg`-only glob (FLAG-4) | `dataset_code_expectation_audit.md` | train split shows **no** case/ext gap → FLAG-4 moot for train |

---

## 4. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `annotations/train` mask inventory & image↔mask pairing | Mask folder not opened (B16-3). Need: 5,367 `.png`, stems == `images/train` stems, no stray/dupes. |
| **NTC-2** | `mask = category_id + 1` mapping | Still **inferred** — needs an annotation↔mask **value** check (decode step; prior report saw 99.85% at pixel level). |
| **NTC-3** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; prior `dataset_report` verified readability + image/mask dim parity but not vs JSON fields. Revisit only if needed. |
| **NTC-4** | Annotation id space (1…7,916) vs 7,774 image files | Cross-split reconciliation (later step). Unaffected by this filename pass. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0/B16-1; not addressable from image filenames. |

---

## 5. WHAT B16-3 SHOULD CHECK NEXT (`annotations/train`)

1. **Inventory (filename-only, no decode):** count `annotations/train` files; confirm **5,367**; all
   `.png`; flag `.PNG`/other extensions, hidden, subdirs, stray files.
2. **Duplicates:** no duplicate `.png` filenames or stems.
3. **Image↔mask pairing (the loader's actual rule):** for every `images/train/<stem>.jpg` there is exactly
   one `annotations/train/<stem>.png`, and vice-versa — report images-without-mask and masks-without-image
   (expect 0 each; prior whole-dataset report already showed 0/0).
4. **Optional cross-ref:** mask stems vs the 5,367 `annotation_train.json` `file_name` stems (expect exact).
5. **Defer pixel/label decode:** mask label values (0–115, 255-absence, single-channel `L`) were already
   verified full-dataset in `dataset_report`; keep them out of B16-3 unless you explicitly request a
   decode-gated sub-step. Keep `mask = category_id + 1` **inferred** (NTC-2).

---

## 6. Required separations (recap)
**(1) Contains** → §1. **(2) JSON cross-check** → §2. **(3) Prior reports** → §3. **(4) NEED_TO_CONFIRM**
→ §4. **(5) B16-3 next (`annotations/train`)** → §5.

_End of B16-2. Only `images/train` (listing) and `annotation_train.json` (file_name extract) were read;
only this report was written; nothing committed._
