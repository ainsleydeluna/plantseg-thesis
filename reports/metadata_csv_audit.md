# Metadata.csv Validation + Index↔COCO Reconciliation (B16-14)

**Scope:** read-only validation of **`Metadata.csv`** plus reconciliation of Metadata `Index` vs COCO
`category_id`, using the three split JSONs for `file_name → image_id → category_id` (**orphan annotations
excluded** — only annotations whose `image_id` ∈ that split's `images[]`). **No image/mask pixels decoded.**
No files modified.

**Interpreter:** project Anaconda Python (numpy/PIL not needed here; stdlib `csv`/`json`).
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

---

## 0. PASS / FLAG verdict — **PASS** · NTC-7 **RESOLVED** · reduce_zero_label **RESOLVED (data side)**

`Metadata.csv` is structurally clean (7,774 rows, 10 columns, 0 nulls, `Index` 0–114, unique names, split
counts match). The **12 prior `dataset_report` mask-shift exceptions are conclusively explained**: they are
**exactly** the 12 images where Metadata `Index` disagrees with COCO `category_id`. Since the masks were
proven to equal `category_id + 1` (B16-13a/b), the **masks and COCO are authoritative and correct**, and
**`Metadata.csv` `Index` is the erroneous field for those 12/7,774 images (0.15%)** — with **zero pipeline
impact** (nothing uses Metadata `Index` as a label source). The background convention is resolved:
**`reduce_zero_label = False`** (masks already encode background = 0 as a real class, diseases 1–115).

---

## 1. METADATA.CSV STRUCTURE & COUNTS

| Property | Result |
|---|---|
| Row count | **7,774** ✓ |
| Columns (10) | `Name, Index, Plant, Disease, Resolution, Label file, Mask ratio, URL, License, Split` |
| Null / blank cells | **0** in every column |
| `Index` range / distinct | **0–114**, distinct **115**, contiguous ✓ → 0-based 115-class disease id |
| `Name` | 7,774 total, **7,774 distinct**, 0 blank, 0 duplicate stems |
| `Plant` | 34 distinct, 0 blank (e.g. Apple, Banana, Basil, Bean, Bell pepper …) |
| `Disease` | **115 distinct**, 0 blank (matches the 115 classes) |
| `License` | 2 distinct: `CC-BY-NC`, `CC0` (0 blank) |
| Metadata stems vs JSON stems | **0** Metadata-only, **0** JSON-only (perfect 7,774 overlap) |

## 2. SPLIT-COLUMN COUNTS & AGREEMENT

| Split (Metadata value) | Count | Canonical | Folder/JSON (prior) | Match |
|---|---|---|---|---|
| `training` | 5,367 | train | 5,367 | ✅ |
| `validation` | 846 | val | 846 | ✅ |
| `test` | 1,561 | test | 1,561 | ✅ |

- **Per-image split agreement: 0 real mismatches** after canonicalizing `training→train`,
  `validation→val`. (A naïve string compare flags 6,213 = all train+val, but that is purely the
  `training/validation` vs `train/val` spelling, not a reassignment.) The Metadata `Split` column thus
  agrees image-for-image with the folder/JSON split — consistent with `dataset_report` (folder/json/csv
  mechanisms all matched, zero overlap).

## 3. METADATA `Index` vs COCO `category_id` RECONCILIATION

| Outcome | Count |
|---|---|
| Images compared (single COCO category) | 7,773 |
| **Agree** (`Index == category_id`) | **7,761** |
| **Disagree** | **12** |
| No COCO category (0-annotation image = FLAG-F) | 1 |
| Multiple COCO categories per image | 0 (one-disease-per-image confirmed) |

**Total: 7,761 + 12 + 1 = 7,774** ✓.

The 12 disagreements (`delta = category_id − Index`):

| Name | Split | Meta Index | COCO category_id | Δ |
|---|---|---|---|---|
| bean_mosaic_virus_23 | train | 12 | 81 | **+69** |
| cucumber_powdery_mildew_google_0067 | train | 50 | 48 | **−2** |
| blueberry_mummy_berry_Bing_0058 | train | 20 | 21 | +1 |
| broccoli_alternaria_leaf_spot_Bing_0012 / _0035 / _0039 | train | 23 | 24 | +1 |
| eggplant_phomopsis_fruit_rot_Bing_0015 | train | 52 | 53 | +1 |
| ginger_leaf_spot_17 | train | 56 | 57 | +1 |
| broccoli_alternaria_leaf_spot_Bing_0021 / _0045 / _0054 | test | 23 | 24 | +1 |
| wheat_head_scab_Bing_0287 | test | 104 | 105 | +1 |

- Δ distribution: **+1 ×10, −2 ×1, +69 ×1** (mostly off-by-one; two larger). All in train (8) or test (4); none in val.

## 4. ARE THE 12 PRIOR EXCEPTIONS EXPLAINED? — **YES, EXACTLY**

| Check | Result |
|---|---|
| Total Metadata↔COCO disagreement stems | **12** |
| `dataset_report` listed exception stems | 12 |
| DR exceptions that ARE disagreements | **12 / 12** |
| DR exceptions missing from disagreement set | **0** |
| Disagreement stems not in DR list | **0** |

**Conclusion:** the 12 `dataset_report` "mask-shift exceptions" (its 99.85% figure) are **one-to-one** the
12 Metadata `Index` ↔ COCO `category_id` disagreements. Because B16-13a/b proved every mask equals
`category_id + 1` (100% vs COCO), the **mask + COCO pair is correct** and **`Metadata.csv` `Index` is wrong
for these 12 images**. `dataset_report`'s 99.85% was an artifact of comparing masks to the (slightly buggy)
Metadata `Index`; against the authoritative COCO `category_id`, consistency is **100%**.

> **Impact: none.** The loader derives labels from the **masks**; `num_classes`/label space come from masks
> (0–115). Nothing consumes `Metadata.csv` `Index`. The 12 errors are a dataset-metadata cosmetic issue
> (worth reporting upstream, not fixing here).

## 5. BACKGROUND / `reduce_zero_label` FINDING — **RESOLVED (data side)**

- Metadata `Index` is a **0-based disease id** (Index 0 = *Apple / apple black rot*; Index 114 = *Zucchini /
  zucchini yellow mosaic virus*); **115 diseases**, no background row.
- Mask encoding (proven): `mask = category_id + 1 = Index + 1` (where Metadata is correct) → **disease mask
  values 1–115**; **mask value 0 = background**, which is **implicit/derived** (no named background class in
  Metadata or COCO `categories`).
- Therefore **`reduce_zero_label = False`** is correct for this dataset: the masks are already in the final
  **116-class** space (background = 0 as a real, kept class; diseases 1–115). `reduce_zero_label=True`
  (drop/shift 0) would be **wrong** here. `255` is introduced only by preprocessing pad/rotation fill.
- This matches the loader (`configs/data.py` `num_classes=116`, passes raw 0–115, `ignore_index=255`).

## 6. REMAINING `NEED_TO_CONFIRM`

| ID | Item | Status |
|---|---|---|
| **NTC-7** | 12 exceptions = Metadata↔COCO disagreement? | **RESOLVED** — yes, exactly 12/12; Metadata `Index` is the wrong field; masks/COCO authoritative. |
| **NTC-5** | `reduce_zero_label` / background | **RESOLVED (data):** `reduce_zero_label=False`, background=0 real implicit class. **Residual (contract):** whether to report **disease-only** mIoU (exclude class 0) is a thesis/metric-convention decision, not a data fact. |
| **NTC-8** | Absent-class eval support: class 41 (no test), class 68 (no val) | **Open (metrics):** per-class val/test IoU undefined for these; metric code must handle NaN. Both present in train. |
| **DATA-12** | 12 erroneous `Metadata.csv` `Index` values | **Documented** (cosmetic; no pipeline impact; candidate upstream bug report). |
| **NTC-1, NTC-F** | mask=category_id+1; FLAG-F | **RESOLVED** earlier (B16-13a/b). |

## 7. EXACT NEXT RECOMMENDED PROMPT (B16-15 — consolidated dataset-audit closeout)

```
B16-15 — Consolidated B16 dataset-audit closeout. Read-only. Plan-gated.

Goal: produce a single consolidated closeout report tying together B16-0..B16-14 with the final
dataset verdict and the residual contract-level decisions. No new dataset reads — synthesize from
the existing reports only.

Allowed to read: all reports/*.md and reports/dataset_report.json. Do NOT open dataset files,
decode anything, or re-scan. Do NOT modify code/dataset. Do NOT commit.

Before writing the report, show me your plan and wait for `go`.

Contents:
1. End-to-end verdict per split (filename pairing + mask values) — train/val/test all PASS; 7,774
   complete pairs, 0 drops.
2. Proven facts: num_classes=116 (bg 0 + diseases 1..115); mask=category_id+1 (100% vs COCO);
   255 = pad/ignore only; one-disease-per-image; reduce_zero_label=False.
3. Known dataset-metadata issues (no pipeline impact): annotations[] unfiltered per split (~27-28%
   orphan, local id spaces); 12 Metadata.csv Index errors; FLAG-F image (0 COCO annotations, valid
   mask).
4. Residual decisions for the contract/modeling: disease-only mIoU convention (exclude class 0?);
   absent-class eval support (class 41 no test, class 68 no val) -> NaN handling; the unresolved
   items from the original transfer summary (teacher-init weights, QNNPACK authoritative run,
   reference.pdf decision).
5. Pointer index: which report answers which question.

Create or update only: reports/dataset_audit_summary.md
After finishing show: final PASS/FLAG verdict; the proven-facts list; the known-issues list; the
residual-decisions list; files created/changed; git status. Then ask whether to (a) commit the
B16 report set, or (b) proceed to modeling blockers. Stop after B16-15.
```

---

## 8. Required separations (recap)
**(1) Structure/counts** → §1. **(2) Split column** → §2. **(3) Index↔COCO** → §3. **(4) 12 explained** →
§4. **(5) background/reduce_zero_label** → §5. **(6) NEED_TO_CONFIRM** → §6. **(7) Next prompt** → §7.

_End of B16-14. Only `Metadata.csv` + the three split JSONs were read (no decode); only this report was
written; nothing committed; no memory/`~/.claude` writes._
