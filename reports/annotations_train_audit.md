# annotations/train Validation (B16-3)

**Scope:** read-only, **filename-level** validation of the `annotations/train` mask folder, cross-checked
against the 5,367 train `file_name` stems in `annotation_train.json`. Two reads only: a **directory
listing** of `annotations\train` (via `os.scandir`, **no mask decode**) and `json.load` of
`annotation_train.json` **solely** to extract stems. **No mask pixels decoded; no label values inspected.**
No other dataset file opened. No files modified.

**Target:** `C:\Users\admin\plantseg_data\plantseg\annotations\train`.
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** the dataloader pairs `images/train/<stem>.jpg` ↔ `annotations/train/<stem>.png` by stem
> (B16-0). B16-2 already proved `images/train` == the JSON's 5,367 stems **exactly**, so the JSON stem set
> is the canonical train stem set; this step validates `annotations/train` against it **without re-listing
> `images/train`**.

---

## 0. PASS / FLAG verdict — **PASS (clean, 0 flags)**

`annotations/train` contains **exactly 5,367 `.png` files**, nothing else — no subdirectories, hidden
files, stray files, alternate/uppercase extensions, or duplicate stems. Its stem set is an **exact match**
to the 5,367 canonical train stems: **0** train stems without a mask, **0** orphan masks.

---

## 1. WHAT `annotations/train` CONTAINS

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
| `.png` (lowercase, exact case) | **5,367** |
| `.PNG` (uppercase) | 0 |
| `.jpg` / `.jpeg` | 0 |
| any non-`.png` | 0 |

- `.png` count **5,367 == 5,367 expected** → **MATCH**.
- All masks are lowercase `.png` (case-sensitive check agrees) → the loader's `f"{stem}.png"` lookup
  resolves for every stem with no case/extension gap.

### Duplicates
| Check | Result |
|---|---|
| Case-insensitive duplicate full filenames | **0** |
| `.png` total / distinct stems | 5,367 / 5,367 |
| Duplicate `.png` stems | **0** |
| Duplicate stems across all extensions (case-insensitive) | **0** |

---

## 2. WHAT THE TRAIN STEM CROSS-CHECK CONFIRMS

| Comparison | Masks | JSON stems | JSON-only (no mask) | Mask-only (orphan) | Intersection |
|---|---|---|---|---|---|
| Stem (loader pairing key) | 5,367 | 5,367 | **0** | **0** | **5,367** |

- Every one of the 5,367 canonical train stems has **exactly one** `.png` mask, and there are **no** masks
  outside that set.
- **Transitive train consistency:** B16-2 proved `images/train` stems == JSON stems (exact); B16-3 proves
  `annotations/train` stems == JSON stems (exact). Therefore
  **`images/train` stems == `annotations/train` stems == `annotation_train.json` stems == 5,367**, i.e. the
  loader's `images/train/<stem>.jpg` ↔ `annotations/train/<stem>.png` pairing is **complete with zero
  unpaired files on either side** (filename level). A *direct* image↔mask listing is deferred to B16-4 to
  make this airtight rather than transitive.

---

## 3. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source | Relation to this step |
|---|---|---|
| Whole-dataset masks: 7,774 `.png`, 0 images-without-mask, 0 masks-without-image, 0 dup stems | `dataset_report.json` `counts`/`extensions` | ✅ consistent — train slice is 5,367 `.png`, fully paired |
| Masks single-channel indexed, mode `L`, ndim 2 | `dataset_report` `masks` | label-format evidence (decode-level) → **not re-derived here** |
| Mask labels contiguous **0–115**, max 115, min 0; **255 absent** in raw masks | `dataset_report` `masks` | prior label evidence (decode-level); kept as prior only |
| `mask = category_id + 1` consistent 99.85% (12 exceptions) | `dataset_report` `mask_value_encoding` | still **inferred** here (no decode) → NTC-1 |
| Loader pairs by stem; mask read as raw `L` indices | `dataset_code_expectation_audit.md` | ✅ filename pairing now confirmed complete for train |

---

## 4. WHAT REMAINS `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | Still **inferred** — requires an annotation↔mask **value** check (decode step). Prior report: 99.85% at pixel level (12/7,774 exceptions), not 100%. |
| **NTC-2** | Per-train-mask label values (0–115, 255-absent, single-channel `L`) | Verified **full-dataset** in `dataset_report` but not re-derived per-split here (no decode). Carried as prior evidence. |
| **NTC-3** | JSON `width`/`height` vs real (EXIF-corrected) mask/image dims | Decode-gated; prior report verified image↔mask dim parity, not vs JSON fields. |
| **NTC-4** | Annotation id space (1…7,916) vs 7,774 image files | Cross-split reconciliation (later step). Unaffected by this filename pass. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0/B16-1; not addressable from mask filenames. |

---

## 5. WHAT B16-4 SHOULD CHECK NEXT (train split consistency check)

1. **Direct three-way reconciliation (filename-level):** list `images/train` **and** `annotations/train`
   together (plus the JSON stems) and prove **`images/train` stems == `annotations/train` stems == JSON
   stems == 5,367** directly (not just transitively) — i.e. confirm the loader would build exactly 5,367
   pairs with **0** dropped (B16-0 noted the loader silently drops an image whose `.png` is missing; this
   proves none are dropped).
2. **Consolidated train-split integrity summary:** fold together B16-1/2/3 into a single PASS table (counts,
   extensions, duplicates, pairing) as the train-split sign-off.
3. **Decide the decode-gated value check (propose as B16-5):** proving `mask = category_id + 1` and the
   per-train-mask label range (0–115, 255-absent) requires decoding mask pixels — schedule it explicitly as
   its own gated step; keep the mapping **INFERRED** until then.
4. Output only `reports/train_split_consistency_audit.md`; still **no decode** unless B16-5 is approved.

---

## 6. Required separations (recap)
**(1) Contains** → §1. **(2) Train stem cross-check** → §2. **(3) Prior reports** → §3. **(4)
NEED_TO_CONFIRM** → §4. **(5) B16-4 next (train split consistency)** → §5.

_End of B16-3. Only `annotations/train` (listing) and `annotation_train.json` (stem extract) were read;
only this report was written; nothing committed._
