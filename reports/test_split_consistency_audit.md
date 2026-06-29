# Test Split Consistency Check (B16-12)

**Scope:** read-only consolidation of the **test** split. Directly reconciles the three test artifacts,
simulates the dataloader's folder/stem pairing rule, and confirms the FLAG-F pair. Three reads only:
directory listings of `images\test` and `annotations\test` (via `os.scandir`, **no decode**) and
`annotation_test.json` (`file_name` stems only). **No pixels decoded; no label values inspected.** No
train/val artifacts opened. No files modified.

**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c` · Dataset root:
`C:\Users\admin\plantseg_data\plantseg`.

---

## 0. PASS / FLAG verdict — **PASS (test split signed off at filename level; 0 new flags)**

The test image index (`annotation_test.json`), the test images (`images/test`), and the test masks
(`annotations/test`) are **mutually identical at the stem level** — one set of **1,561** stems, none
exclusive to any artifact. A simulation of the dataloader pairing rule yields **exactly 1,561 image↔mask
pairs with zero silent drops**, and the **FLAG-F pair `apple_black_rot_google_0001` is included**. The only
carried-forward items are the JSON-array flags from B16-9 (FLAG-A unfiltered `annotations[]`; FLAG-E
per-file local id space; FLAG-F one image with no COCO annotations) — none affect folder/stem loading.

---

## 1. THREE-WAY RECONCILIATION RESULT

| Stem set | Size |
|---|---|
| `images/test` `.jpg` stems | **1,561** |
| `annotations/test` `.png` stems | **1,561** |
| `annotation_test.json` `file_name` stems | **1,561** |
| **Union of all three** | **1,561** |

| Set difference | Count |
|---|---|
| image-only / mask-only / json-only | **0 / 0 / 0** |
| every pairwise diff | **0** |

- **All three sets are equal** and each contains exactly **1,561** stems → the test split is internally
  consistent across index, images, and masks (B16-11's transitive conclusion now **directly** proven).

## 2. DATALOADER PAIRING SIMULATION RESULT

Replicates `src/data/dataset.py`: for each `images/test/<stem>.jpg`, keep the pair iff
`annotations/test/<stem>.png` exists (filename existence only — no decode).

| Metric | Value | Expected | Verdict |
|---|---|---|---|
| Simulated pairs | **1,561** | 1,561 | **MATCH** |
| Images silently dropped (no mask) | **0** | 0 | **PASS** |
| Masks with no image (unreached) | **0** | 0 | **PASS** |

- The loader would construct **exactly 1,561 test pairs** and **drop nothing** — B16-0 FLAG-5 does not
  materialize for test.

## 3. FLAG-F PAIR CONFIRMATION (`apple_black_rot_google_0001`)

| Check | Result |
|---|---|
| Image present (`images/test/apple_black_rot_google_0001.jpg`) | ✅ True |
| Mask present (`annotations/test/apple_black_rot_google_0001.png`) | ✅ True |
| Included in simulated loader pairs | ✅ **True** |

- The image with **zero COCO annotations** in `annotation_test.json` forms a **complete image↔mask pair**
  and **is loaded normally**. Its missing COCO polygons are irrelevant to the semantic-segmentation loader.
  Only the mask's *pixel content* (NTC-F) remains for the decode-gated check.

## 4. CONSOLIDATED TEST-SPLIT EVIDENCE (B16-9 / B16-10 / B16-11 / B16-12)

| Step | Artifact | Count | Ext | Dups | Cross-check | Verdict |
|---|---|---|---|---|---|---|
| B16-9 | `annotation_test.json` `images[]` | 1,561 | — | 0 | valid COCO; 115 cats (0–114) all present | PASS |
| B16-10 | `images/test` | 1,561 | `.jpg` only | 0 | == JSON stems (exact, case-sensitive); FLAG-F image present | PASS |
| B16-11 | `annotations/test` | 1,561 | `.png` only | 0 | == JSON stems (exact); FLAG-F mask present | PASS |
| B16-12 | three-way + pairing | 1,561 | — | 0 | all equal; 1,561 pairs; 0 drops; FLAG-F pair included | PASS |

**Carried-forward FLAGs (not test-split blockers):**
- **FLAG-A:** `annotations[]` unfiltered (4,466/15,569 = 28.7% orphan). No loader impact.
- **FLAG-E:** per-file local id space (test 1…2295). Cross-split reconciliation must use `file_name`.
- **FLAG-F:** one image (`apple_black_rot_google_0001`) has 0 COCO annotations but a complete image↔mask
  pair → loads normally; mask pixel content deferred (NTC-F).

### Whole-dataset filename-level roll-up (all three splits)
| Split | Images | Masks | JSON | Pairs (0 drops) | Status |
|---|---|---|---|---|---|
| train (B16-1…4) | 5,367 | 5,367 | 5,367 | 5,367 | ✅ PASS |
| val (B16-5…8) | 846 | 846 | 846 | 846 | ✅ PASS |
| test (B16-9…12) | 1,561 | 1,561 | 1,561 | 1,561 | ✅ PASS |
| **Total** | **7,774** | **7,774** | **7,774** | **7,774** | ✅ matches `dataset_report` |

## 5. WHAT PRIOR REPORTS ALREADY CONFIRMED

| Item | Source |
|---|---|
| Folder split authoritative; test = 1,561; zero cross-split overlap | `dataset_report.json` `split` |
| Whole dataset: 7,774 images / 7,774 masks, 0 unpaired, 0 dup stems | `dataset_report.json` `counts` |
| Masks single-channel indexed (`L`), labels contiguous 0–115, 255 absent | `dataset_report` `masks` |
| `mask = category_id + 1` consistent 99.85% (12 exceptions) | `dataset_report` `mask_value_encoding` |
| Loader pairs by stem; `num_classes=116`, `ignore=255` | `dataset_code_expectation_audit.md` |

## 6. REMAINING `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `mask = category_id + 1` mapping | **Inferred** across all splits — needs the decode-gated annotation↔mask value check (B16-13). |
| **NTC-F** | Pixel content of `apple_black_rot_google_0001.png` | Pair exists; all-background vs unannotated disease region needs decode (B16-13). |
| **NTC-2** | Per-mask label range (0–115, 255-absent, single-channel `L`) per split | Verified full-dataset in `dataset_report`; not re-derived per-split (no decode). |
| **NTC-3** | JSON `width`/`height` vs real (EXIF-corrected) dims | Decode-gated; not checked. |
| **NTC-4** | Orphan images across split JSONs (FLAG-A/E) | Identifiable only by `file_name`; ids are per-file local. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0. |

## 7. EXACT NEXT RECOMMENDED PROMPT (B16-13 — decode-gated mask-value proof)

```
B16-13 — Mask VALUE check, all splits (decode-gated). Read-only. Plan-gated.

NOTE: FIRST step that DECODES mask pixels (read-only, no modification). It opens and reads
pixel values of annotations/{train,val,test} PNG masks. It does NOT decode images. Approve
the mask-decode permission explicitly before running. (Optionally scope to test-only first.)

Goal: prove at the pixel level, per split, (a) every mask is single-channel indexed with
values in {0..115} and no 255; (b) mask=category_id+1 holds per image (mask nonzero index
set == {category_id+1 for that image's annotations}), quantifying exceptions (prior
full-dataset report: ~0.15%, 12 masks); (c) what apple_black_rot_google_0001.png contains
(all-background vs disease region) -> closes NTC-F.

Allowed to open: annotations/{train,val,test} PNG masks (pixel read only); the three
annotation_*.json (image_id->category_id + file_name). Reuse prior reports. Do NOT decode
images. Do NOT open Metadata.csv. Do NOT modify any file. Do NOT commit.

Before writing the report, show me your plan and wait for `go`.

Checks per split:
1. mode/ndim (expect L/2D); unique values subset of {0..115}; 255 absent.
2. aggregate label histogram; min/max non-ignore; 255 presence.
3. mask=category_id+1 per image: consistent vs exceptions; list <=20 exceptions per split.
4. FLAG-F image apple_black_rot_google_0001.png: report its unique mask values.
5. cross-check vs dataset_report full-dataset numbers.

Create or update only: reports/mask_value_audit.md
Separate: (1) per-split label findings; (2) mask=category_id+1 result (proven/quantified);
(3) FLAG-F image mask content; (4) prior report agreement; (5) NEED_TO_CONFIRM;
(6) next step (e.g. Metadata.csv validation, or close out the B16 dataset audit).

After finishing show: PASS/FLAG; label-range result; 255 result; mask=category_id+1
consistency % per split; FLAG-F mask content; files created/changed; git status; exact next
prompt. Stop after B16-13.
```

---

## 8. Required separations (recap)
**(1) Three-way reconciliation** → §1. **(2) Pairing simulation** → §2. **(3) FLAG-F pair** → §3.
**(4) Consolidated B16-9/10/11** → §4. **(5) Prior reports** → §5. **(6) NEED_TO_CONFIRM** → §6.
**(7) Next prompt (B16-13)** → §7.

_End of B16-12. Only `images/test` + `annotations/test` (listings) and `annotation_test.json` (stems) were
read; only this report was written; nothing committed._
