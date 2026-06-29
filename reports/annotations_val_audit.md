# annotations/val Validation (B16-7)

**Scope:** read-only, **filename-level** validation of the `annotations/val` mask folder, cross-checked
against the 846 val `file_name` stems in `annotation_val.json`. Two reads only: a **directory listing** of
`annotations\val` (via `os.scandir`, **no mask decode**) and `json.load` of `annotation_val.json` **solely**
to extract stems. **No mask pixels decoded; no label values inspected.** No other dataset file opened. No
files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\annotations\val`.
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** loader pairs `images/val/<stem>.jpg` ↔ `annotations/val/<stem>.png` by stem (B16-0). B16-6
> proved `images/val` == the JSON's 846 stems exactly, so the JSON stem set is canonical; this step
> validates `annotations/val` against it **without re-listing `images/val`**.

---

## 0. PASS / FLAG verdict — **PASS (clean, 0 flags)**

`annotations/val` contains **exactly 846 `.png` files**, nothing else — no subdirectories, hidden, stray,
alternate/uppercase-extension files, or duplicate stems. Its stem set is an **exact match** to the 846
canonical val stems: **0** val stems without a mask, **0** orphan masks.

---

## 1. WHAT `annotations/val` CONTAINS

| Property | Value |
|---|---|
| Directory exists | ✅ True |
| Total entries | **846** |
| Regular files | **846** |
| Subdirectories / hidden / other | 0 / 0 / 0 |

### Extension breakdown
| Extension | Count |
|---|---|
| `.png` (lowercase, exact case) | **846** |
| `.PNG` (uppercase) / `.jpg`·`.jpeg` / any non-`.png` | 0 / 0 / 0 |

- `.png` count **846 == 846 expected** → **MATCH**. All lowercase `.png` → loader `f"{stem}.png"` lookup
  resolves for every stem.

### Duplicates
| Check | Result |
|---|---|
| Case-insensitive duplicate full filenames | **0** |
| `.png` total / distinct stems | 846 / 846 |
| Duplicate `.png` stems | **0** |
| Duplicate stems across all extensions (ci) | **0** |

---

## 2. WHAT THE VAL STEM CROSS-CHECK CONFIRMS

| Comparison | Masks | JSON stems | JSON-only (no mask) | Mask-only (orphan) | Intersection |
|---|---|---|---|---|---|
| Stem (loader pairing key) | 846 | 846 | **0** | **0** | **846** |

- Every one of the 846 canonical val stems has **exactly one** `.png` mask, with **no** masks outside that
  set.
- **Transitive val consistency:** B16-6 proved `images/val` stems == JSON stems (exact); B16-7 proves
  `annotations/val` stems == JSON stems (exact). Therefore
  **`images/val` stems == `annotations/val` stems == `annotation_val.json` stems == 846**. A *direct*
  image↔mask listing is consolidated in B16-8.

---

## 3. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source |
|---|---|
| Whole-dataset masks: 7,774 `.png`, 0 unpaired, 0 dup stems | `dataset_report.json` `counts` |
| Masks single-channel indexed (`L`), labels contiguous 0–115, 255 absent | `dataset_report` `masks` (decode-level; not re-derived) |
| `mask = category_id + 1` consistent 99.85% (12 exceptions) | `dataset_report` `mask_value_encoding` (still **inferred** here) |
| Val split = 846; zero cross-split overlap | `dataset_report.json` `split` |
| Loader pairs by stem; mask read as raw `L` indices | `dataset_code_expectation_audit.md` |

---

## 4. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | **Inferred** — needs decode-gated annotation↔mask value check. |
| **NTC-2** | Per-val-mask label range (0–115, 255-absent, single-channel `L`) | Verified full-dataset in `dataset_report`; not re-derived per-split (no decode). |
| **NTC-3** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; not checked. |
| **NTC-4** | Val JSON orphan images (B16-5 FLAG-A/E) | Identifiable only by `file_name` across files (cross-split); ids are local. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0. |

---

## 5. WHAT B16-8 SHOULD CHECK NEXT (val split consistency check)

Mirror B16-4 for the val split:
1. **Direct three-way reconciliation (filename-level):** list `images/val` **and** `annotations/val`
   (plus JSON stems); prove **`images/val` stems == `annotations/val` stems == JSON stems == 846**
   directly (not just transitively).
2. **Simulate loader pairing:** confirm exactly **846** image↔mask pairs with **0** silent drops and 0
   unreached masks.
3. **Consolidated val-split PASS table** folding B16-5/6/7 + this check.
4. Keep `mask = category_id + 1` **inferred**; mask label values from `dataset_report` as prior evidence.
5. Output only `reports/val_split_consistency_audit.md`; no decode.

(After B16-8, the **test** split would follow the same B16-1…4 pattern; and the decode-gated mask-value
proof — `mask=category_id+1` + per-mask label range — remains an outstanding cross-split step.)

---

## 6. Required separations (recap)
**(1) Contains** → §1. **(2) Val stem cross-check** → §2. **(3) Prior reports** → §3. **(4)
NEED_TO_CONFIRM** → §4. **(5) B16-8 next (val split consistency)** → §5.

_End of B16-7. Only `annotations/val` (listing) and `annotation_val.json` (stem extract) were read; only
this report was written; nothing committed._
