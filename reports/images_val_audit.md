# images/val Validation (B16-6)

**Scope:** read-only, **filename-level** validation of `images/val` cross-checked against the 846
`file_name` entries in `annotation_val.json`. Two reads only: a **directory listing** of `images\val`
(via `os.scandir`, **no pixel decode**) and `json.load` of `annotation_val.json` **solely** to extract
`file_name`s. No image/mask decoded. No other dataset file opened. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\images\val`.
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** the dataloader pairs `images/val/<stem>.jpg` ↔ `annotations/val/<stem>.png` by stem
> (B16-0); the JSON is not used for loading. This validates the **val image inventory** and the folder↔JSON
> name agreement (closes B16-5 NTC-2). Masks (`annotations/val`) are validated in B16-7.

---

## 0. PASS / FLAG verdict — **PASS (clean, 0 flags)**

`images/val` contains **exactly 846 `.jpg` files**, nothing else — no subdirectories, hidden, stray, or
alternate-extension files, no case anomalies, no duplicate stems. It is an **exact match** to
`annotation_val.json`'s `file_name` set at **both** full-filename (case-sensitive) and stem level: zero
folder-only, zero JSON-only.

---

## 1. WHAT `images/val` CONTAINS

| Property | Value |
|---|---|
| Directory exists | ✅ True |
| Total entries | **846** |
| Regular files | **846** |
| Subdirectories / hidden / other | 0 / 0 / 0 |

### Extension breakdown
| Extension | Count |
|---|---|
| `.jpg` (lowercase, exact case) | **846** |
| `.jpeg` / `.JPG` / `.png` / any non-`.jpg` | 0 / 0 / 0 / 0 |

- `.jpg` count **846 == 846 expected** → **MATCH**. All lowercase `.jpg` → loader `glob("*.jpg")` matches
  every file with no case/extension gap.

### Duplicates
| Check | Result |
|---|---|
| Case-insensitive duplicate full filenames | **0** |
| `.jpg` total / distinct stems | 846 / 846 |
| Duplicate `.jpg` stems | **0** |
| Duplicate stems across all extensions (ci) | **0** |

---

## 2. WHAT THE `annotation_val.json` CROSS-CHECK CONFIRMS

| Comparison | Folder | JSON | JSON-only | Folder-only | Verdict |
|---|---|---|---|---|---|
| Full filename (case-sensitive) | 846 | 846 | **0** | **0** | exact match |
| Stem (loader pairing key) | 846 | 846 | **0** | **0** | exact match |
| Stem intersection | — | — | — | — | **846** |
| Case-only diff | — | — | — | — | **0** |

- The 846 `file_name`s in `annotation_val.json` (validated clean in B16-5) correspond **1:1, exact case**,
  to the 846 files in `images/val`. **B16-5 NTC-2 (file_name ↔ folder) is CLOSED.**

---

## 3. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source | Relation |
|---|---|---|
| Val split = 846 (folder authoritative) | `dataset_report.json` `split.counts` | ✅ matches the 846 files here |
| Whole-dataset images `.jpg` only (7,774) | `dataset_report.json` `extensions` | ✅ consistent — val slice all `.jpg` |
| Zero cross-split id/stem overlap | `dataset_report.json` `overlaps_counts` | ✅ val images distinct from train |
| All images readable; image↔mask EXIF-aware dim parity (0 mismatch) | `dataset_report` | covers decode-level checks → **not repeated** |
| Loader pairs by stem; `.jpg`-only glob | `dataset_code_expectation_audit.md` | val shows no case/ext gap |

---

## 4. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `annotations/val` mask inventory & image↔mask pairing | Mask folder not opened (B16-7). Need 846 `.png`, stems == val stems, no stray/dupes. |
| **NTC-2** | `mask = category_id + 1` mapping | Still **inferred** — needs annotation↔mask value check (decode). |
| **NTC-3** | What the val JSON orphan images are | B16-5 FLAG-A/E: identifiable only by `file_name` across files (cross-split); ids are local. |
| **NTC-4** | JSON `width`/`height` vs real dims | Decode-gated; prior report verified readability + dim parity. |
| **NTC-5** | `reduce_zero_label` / explicit background | Open since B16-0. |

---

## 5. WHAT B16-7 SHOULD CHECK NEXT (`annotations/val`)

1. **Inventory (filename-only, no decode):** count `annotations/val` files; confirm **846**; all `.png`;
   flag `.PNG`/other ext, hidden, subdirs, stray.
2. **Duplicates:** no duplicate `.png` filenames or stems.
3. **Mask stems vs val stems:** set of `annotations/val` `.png` stems == the 846 `annotation_val.json`
   `file_name` stems (B16-2/B16-3 proved images/val == JSON exactly, so the JSON set is canonical — no need
   to re-list `images/val`); report any stem without a mask and any orphan mask.
4. **Defer pixel/label decode:** mask label values already verified full-dataset in `dataset_report`; keep
   `mask = category_id + 1` **inferred**.
5. Output only `reports/annotations_val_audit.md`; no decode.

---

## 6. Required separations (recap)
**(1) Contains** → §1. **(2) JSON cross-check** → §2. **(3) Prior reports** → §3. **(4) NEED_TO_CONFIRM**
→ §4. **(5) B16-7 next (`annotations/val`)** → §5.

_End of B16-6. Only `images/val` (listing) and `annotation_val.json` (file_name extract) were read; only
this report was written; nothing committed._
