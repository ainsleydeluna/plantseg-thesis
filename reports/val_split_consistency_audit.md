# Val Split Consistency Check (B16-8)

**Scope:** read-only consolidation of the **val** split. Directly reconciles the three val artifacts and
simulates the dataloader's folder/stem pairing rule. Three reads only: directory listings of `images\val`
and `annotations\val` (via `os.scandir`, **no decode**) and `annotation_val.json` (`file_name` stems only).
**No pixels decoded; no label values inspected.** No train/test artifacts opened. No files modified.

**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c` · Dataset root:
`C:\Users\admin\plantseg_data\plantseg`.

---

## 0. PASS / FLAG verdict — **PASS (val split signed off at filename level; 0 new flags)**

The val image index (`annotation_val.json`), the val images (`images/val`), and the val masks
(`annotations/val`) are **mutually identical at the stem level** — one set of **846** stems, none exclusive
to any artifact. A simulation of the dataloader pairing rule yields **exactly 846 image↔mask pairs with
zero silent drops**. The only carried-forward items are the JSON-array hygiene flags from B16-5 (FLAG-A
unfiltered `annotations[]`; FLAG-E per-file local id space) — neither affects folder/stem loading.

---

## 1. THREE-WAY RECONCILIATION RESULT

| Stem set | Size |
|---|---|
| `images/val` `.jpg` stems | **846** |
| `annotations/val` `.png` stems | **846** |
| `annotation_val.json` `file_name` stems | **846** |
| **Union of all three** | **846** |

| Set difference | Count |
|---|---|
| image-only / mask-only / json-only | **0 / 0 / 0** |
| every pairwise diff (img\mask, mask\img, img\json, json\img, mask\json, json\mask) | **0** |

- **All three sets are equal** and each contains exactly **846** stems → the val split is internally
  consistent across index, images, and masks (B16-7's transitive conclusion now **directly** proven).

## 2. DATALOADER PAIRING SIMULATION RESULT

Replicates `src/data/dataset.py`: for each `images/val/<stem>.jpg`, keep the pair iff
`annotations/val/<stem>.png` exists (filename existence only — no decode).

| Metric | Value | Expected | Verdict |
|---|---|---|---|
| Simulated pairs | **846** | 846 | **MATCH** |
| Images silently dropped (no mask) | **0** | 0 | **PASS** |
| Masks with no image (unreached) | **0** | 0 | **PASS** |

- The loader would construct **exactly 846 validation pairs** and **drop nothing** — B16-0 FLAG-5
  (silent-drop risk) does not materialize for val.

## 3. CONSOLIDATED VAL-SPLIT EVIDENCE (B16-5 / B16-6 / B16-7 / B16-8)

| Step | Artifact | Count | Ext | Dups | Cross-check | Verdict |
|---|---|---|---|---|---|---|
| B16-5 | `annotation_val.json` `images[]` | 846 | — | 0 (ids, names) | valid COCO; 115 cats (0–114) all present | PASS |
| B16-6 | `images/val` | 846 | `.jpg` only | 0 | == JSON stems (exact, case-sensitive) | PASS |
| B16-7 | `annotations/val` | 846 | `.png` only | 0 | == JSON stems (exact) | PASS |
| B16-8 | three-way + pairing | 846 | — | 0 | all equal; 846 pairs; 0 drops | PASS |

**Carried-forward FLAGs (not val-split blockers):**
- **FLAG-A (B16-5):** `annotations[]` not filtered to val — 2,467/8,927 (27.6%) orphan. No loader impact.
- **FLAG-E (B16-5):** per-file local id space (val 1…1247 ≠ train 1…7916) → cross-split id reconciliation
  must use `file_name`, not ids.

## 4. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source |
|---|---|
| Folder split authoritative; val = 846; zero cross-split id/stem overlap | `dataset_report.json` `split` |
| Whole dataset: 7,774 images / 7,774 masks, 0 unpaired, 0 dup stems | `dataset_report.json` `counts` |
| Masks single-channel indexed (`L`), labels contiguous 0–115, 255 absent | `dataset_report` `masks` |
| `mask = category_id + 1` consistent 99.85% (12 exceptions) | `dataset_report` `mask_value_encoding` |
| Loader pairs by stem; `num_classes=116`, `ignore=255` | `dataset_code_expectation_audit.md` |

## 5. REMAINING `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | **Inferred** — needs decode-gated annotation↔mask value check. |
| **NTC-2** | Per-val-mask label range (0–115, 255-absent, single-channel `L`) | Verified full-dataset in `dataset_report`; not re-derived per-split (no decode). |
| **NTC-3** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; not checked. |
| **NTC-4** | Val/train JSON orphan images (FLAG-A/E) | Identifiable only by `file_name` across files; ids are local per file. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0. |
| **NTC-6** | **Test split not yet audited** | B16-1…8 covered train + val; test needs the same index↔image↔mask pass. |

## 6. EXACT NEXT RECOMMENDED PROMPT (test split — `annotation_test.json`)

```
B16-9 — Validate annotation_test.json only. Read-only first. Plan-gated.

Goal: validate annotation_test.json on its own terms (mirrors B16-1/B16-5). Expected test
image count = 1,561. Independent dataset-index integrity check (loader ignores JSON).

Target: C:\Users\admin\plantseg_data\plantseg\annotation_test.json

For this step do NOT open: annotation_train.json, annotation_val.json, Metadata.csv, and any
images/* or annotations/* folder.
Use prior reports for evidence: reports/annotation_train_audit.md, reports/annotation_val_audit.md,
reports/dataset_report.json, reports/dataset_code_expectation_audit.md.

Before writing the report, show me your plan and wait for `go`.
Use programmatic validation only; do not dump thousands of JSON lines.

Checks: 1 valid JSON; 2 top-level schema/keys; 3 counts (images/annotations/categories);
4 image count == 1,561; 5 duplicate image ids / annotation ids / file_names;
6 image record required fields + types; 7 annotation record required fields + types;
8 null/malformed/suspicious (bbox/area/orphan image_id/0-ann images/anns-per-image);
9 category_id range (expect 0-114, contiguous, all 115 present?); 10 explicit background?;
11 num_classes=116 implication (keep mask=category_id+1 INFERRED); 12 ignore-255/convention;
13 is annotations[] filtered to test or unfiltered (expect unfiltered, local id space);
14 deferred items (file_name<->images/test, dims, Metadata.csv).

Create or update only: reports/annotation_test_audit.md
Separate: (1) what annotation_test.json contains; (2) prior reports; (3) NEED_TO_CONFIRM;
(4) what to check next in images/test.

Rules: no dataset-file edits, no JSON/CSV rewrite, no image/mask decode, no train/download/
install, no model/loss/metric edits, do not touch reference.pdf / context.md / teacher-init
files, no commit.

After finishing show: PASS/FLAG; image/annotation/category counts; top-level schema; duplicate
result; malformed/null result; category/label-index findings; background/ignore finding;
annotations[] filtered-or-unfiltered; files created/changed; git status; exact B16-10 prompt
for images/test. Stop after B16-9.
```

---

## 7. Required separations (recap)
**(1) Three-way reconciliation** → §1. **(2) Pairing simulation** → §2. **(3) Consolidated B16-5/6/7** →
§3. **(4) Prior reports** → §4. **(5) NEED_TO_CONFIRM** → §5. **(6) Next prompt (test split)** → §6.

_End of B16-8. Only `images/val` + `annotations/val` (listings) and `annotation_val.json` (stems) were
read; only this report was written; nothing committed._
