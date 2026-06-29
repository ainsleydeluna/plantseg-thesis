# annotations/test Validation (B16-11)

**Scope:** read-only, **filename-level** validation of the `annotations/test` mask folder, cross-checked
against the 1,561 test `file_name` stems in `annotation_test.json`. Two reads only: a **directory listing**
of `annotations\test` (via `os.scandir`, **no mask decode**) and `json.load` of `annotation_test.json`
**solely** to extract stems. **No mask pixels decoded; no label values inspected.** No other dataset file
opened. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\annotations\test`.
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** loader pairs `images/test/<stem>.jpg` ↔ `annotations/test/<stem>.png` by stem (B16-0).
> B16-10 proved `images/test` == the JSON's 1,561 stems exactly, so the JSON stem set is canonical; this
> step validates `annotations/test` against it **without re-listing `images/test`**, and confirms the
> B16-9 FLAG-F mask.

---

## 0. PASS / FLAG verdict — **PASS (clean, 0 flags)**

`annotations/test` contains **exactly 1,561 `.png` files**, nothing else — no subdirectories, hidden,
stray, alternate/uppercase-extension files, or duplicate stems. Its stem set is an **exact match** to the
1,561 canonical test stems: **0** test stems without a mask, **0** orphan masks. The FLAG-F mask
**`apple_black_rot_google_0001.png` is present**.

---

## 1. WHAT `annotations/test` CONTAINS

| Property | Value |
|---|---|
| Directory exists | ✅ True |
| Total entries | **1,561** |
| Regular files | **1,561** |
| Subdirectories / hidden / other | 0 / 0 / 0 |

### Extension breakdown
| Extension | Count |
|---|---|
| `.png` (lowercase, exact case) | **1,561** |
| `.PNG` (uppercase) / `.jpg`·`.jpeg` / any non-`.png` | 0 / 0 / 0 |

- `.png` count **1,561 == 1,561 expected** → **MATCH**. All lowercase `.png` → loader `f"{stem}.png"`
  lookup resolves for every stem.

### Duplicates
| Check | Result |
|---|---|
| Case-insensitive duplicate full filenames | **0** |
| `.png` total / distinct stems | 1,561 / 1,561 |
| Duplicate `.png` stems | **0** |
| Duplicate stems across all extensions (ci) | **0** |

---

## 2. WHAT THE TEST STEM CROSS-CHECK CONFIRMS

| Comparison | Masks | JSON stems | JSON-only (no mask) | Mask-only (orphan) | Intersection |
|---|---|---|---|---|---|
| Stem (loader pairing key) | 1,561 | 1,561 | **0** | **0** | **1,561** |

- Every one of the 1,561 canonical test stems has **exactly one** `.png` mask, with **no** masks outside
  that set.
- **Transitive test consistency:** B16-10 proved `images/test` stems == JSON stems (exact); B16-11 proves
  `annotations/test` stems == JSON stems (exact). Therefore
  **`images/test` stems == `annotations/test` stems == `annotation_test.json` stems == 1,561**. A *direct*
  image↔mask listing is consolidated in B16-12.

---

## 3. `apple_black_rot_google_0001.png` PRESENCE (B16-9 FLAG-F)

| Check | Result |
|---|---|
| Present in `annotations/test` (exact case) | ✅ **True** |
| Present (case-insensitive) | ✅ True |
| Stem in `annotation_test.json` test stems | ✅ True |

- The image with **zero COCO annotations** (FLAG-F) has **both** an image file (B16-10) **and** a mask file
  (B16-11). It therefore forms a **complete image↔mask pair** and **will load normally** despite its
  missing COCO polygons. **NTC-F is closed at the filename level** — only the mask's *pixel content*
  (all-background vs an unannotated disease region) remains a decode-gated question for the mask-value
  check.

---

## 4. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source |
|---|---|
| Whole-dataset masks: 7,774 `.png`, 0 unpaired, 0 dup stems | `dataset_report.json` `counts` |
| Masks single-channel indexed (`L`), labels contiguous 0–115, 255 absent | `dataset_report` `masks` (decode-level; not re-derived) |
| `mask = category_id + 1` consistent 99.85% (12 exceptions) | `dataset_report` `mask_value_encoding` (still **inferred** here) |
| Test split = 1,561; zero cross-split overlap | `dataset_report.json` `split` |
| Loader pairs by stem; mask read as raw `L` indices | `dataset_code_expectation_audit.md` |

---

## 5. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | **Inferred** — needs decode-gated annotation↔mask value check. |
| **NTC-F** | Pixel content of `apple_black_rot_google_0001.png` | Pair exists; whether the mask is all-background or holds a disease region needs the decode-gated mask check. |
| **NTC-2** | Per-test-mask label range (0–115, 255-absent, single-channel `L`) | Verified full-dataset in `dataset_report`; not re-derived per-split (no decode). |
| **NTC-3** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; not checked. |
| **NTC-4** | Test JSON orphan images (FLAG-A/E) | Identifiable only by `file_name` across files; ids are local. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0. |

---

## 6. WHAT B16-12 SHOULD CHECK NEXT (test split consistency check)

Mirror B16-4/B16-8 for the test split:
1. **Direct three-way reconciliation (filename-level):** list `images/test` **and** `annotations/test`
   (plus JSON stems); prove **`images/test` stems == `annotations/test` stems == JSON stems == 1,561**
   directly.
2. **Simulate loader pairing:** confirm exactly **1,561** image↔mask pairs with **0** silent drops and 0
   unreached masks — including the FLAG-F pair `apple_black_rot_google_0001`.
3. **Consolidated test-split PASS table** folding B16-9/10/11 + this check.
4. Keep `mask = category_id + 1` **inferred**; mask label values from `dataset_report` as prior evidence.
5. Output only `reports/test_split_consistency_audit.md`; no decode.

(After B16-12, all three splits are signed off at filename level. The remaining outstanding work is the
**decode-gated mask-value proof** — `mask=category_id+1` + per-mask label range + the FLAG-F image's mask
content — which would be the natural B16-13.)

---

## 7. Required separations (recap)
**(1) Contains** → §1. **(2) Test stem cross-check** → §2. **(3) FLAG-F presence** → §3. **(4) Prior
reports** → §4. **(5) NEED_TO_CONFIRM** → §5. **(6) B16-12 next (test split consistency)** → §6.

_End of B16-11. Only `annotations/test` (listing) and `annotation_test.json` (stem extract) were read; only
this report was written; nothing committed._
