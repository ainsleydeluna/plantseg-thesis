# PlantSeg Dataset Audit — Consolidated Closeout (B16-15)

**Synthesis only.** Consolidates B16-0 … B16-14 from the existing `reports/*.md` (all produced this session)
plus `reports/dataset_report.json`. **No new dataset/folder/JSON/CSV reads, no image/mask decode.** No files
modified other than this report. No commit.

**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c` · Dataset root:
`C:\Users\admin\plantseg_data\plantseg`.

---

## 0. FINAL VERDICT — **PASS (dataset validated end-to-end)**

The PlantSeg dataset is validated from folder structure through pixel-level mask labels. All three splits
form **complete image↔mask pairs with zero silent drops**, masks are clean indexed labels (0–115, no 255),
and the previously-inferred label mapping is **empirically proven**. Remaining items are either cosmetic
dataset-metadata issues (no pipeline impact) or contract/modeling decisions — none block training.

---

## 1. END-TO-END VERDICT PER SPLIT

| Split | `images/<split>` `.jpg` | `annotations/<split>` `.png` | JSON `images[]` | Complete pairs | Silent drops | Verdict |
|---|---|---|---|---|---|---|
| **train** | 5,367 | 5,367 | 5,367 | **5,367** | **0** | ✅ PASS |
| **val** | 846 | 846 | 846 | **846** | **0** | ✅ PASS |
| **test** | 1,561 | 1,561 | 1,561 | **1,561** | **0** | ✅ PASS |
| **Total** | **7,774** | **7,774** | **7,774** | **7,774** | **0** | ✅ matches `dataset_report` |

- Per split, the three stem sets (images / masks / JSON `file_name`) are **identical** (0 image-only,
  mask-only, or json-only). The loader's `images/<split>/<stem>.jpg` ↔ `annotations/<split>/<stem>.png`
  pairing builds exactly 7,774 pairs and drops nothing.
- Cross-split: zero identifier/stem overlap (per `dataset_report`); train/val/test are disjoint.

## 2. PROVEN DATASET FACTS

| Fact | Status / evidence |
|---|---|
| `num_classes = 116` | background (0) + 115 diseases (1–115); mask max label 115 |
| background = raw mask label **0** | ubiquitous (in 7,773/7,774 masks; the 1 exception is an all-disease train mask), 80.6% of pixels |
| disease labels = raw mask labels **1–115** | contiguous; Metadata `Index` 0–114 shifted by +1 |
| `mask = category_id + 1` | **PROVEN 100%** vs COCO `category_id` across all 7,773 annotated images (train 5,367 + val 846 + test 1,560) |
| `reduce_zero_label = False` | masks already in final 116-class space; bg=0 is a real kept class, not ignore |
| raw masks contain **0–115 only** | single-channel `L`, 2-D; no value >115; 0 range violations |
| raw masks do **not** contain 255 | 0/7,774 masks contain 255 |
| 255 = preprocessing pad/ignore only | introduced at resize-pad + rotation fill; never in released masks |
| one-disease-per-image | every annotated image has exactly 1 COCO category (`distinct_category_ids_per_image = {1: N}`) |

## 3. KNOWN NON-BLOCKING DATASET ISSUES

| Issue | Detail |
|---|---|
| Split JSON `annotations[]` **unfiltered** | each split JSON's annotation array contains full-dataset annotations: train 27.7% (14,647), val 27.6% (2,467), test 28.7% (4,466) reference non-split image_ids (orphans). `images[]` is correctly filtered. |
| Split JSON image IDs are **local** | id spaces are per-file (train 1…7916, val 1…1247, test 1…2295) — **not comparable across split files**; cross-split joins must use `file_name`, not id. |
| **12 wrong `Metadata.csv` `Index`** values | 12/7,774 rows (0.15%) where `Index` ≠ COCO `category_id` (Δ = +1 ×10, −2 ×1, +69 ×1). Masks/COCO are correct; Metadata `Index` is the buggy field. These are exactly `dataset_report`'s 12 "shift exceptions." |
| **FLAG-F** `apple_black_rot_google_0001.jpg` | has **0 COCO annotations** but a valid image+mask pair; mask = `{0, 1}` → correctly labeled apple black rot (class 1). Loads normally. |
| class **41** has **no test** mask support | mask value 42 absent from all test masks (its `category_id` 41 appears only in orphan annotations). |
| class **68** has **no val** mask support | mask value 69 absent from all val masks. Both class 41 and 68 **are** present in train. |

## 4. PIPELINE IMPACT

- **Folder/stem loader is safe.** `src/data/dataset.py` pairs by folder + stem and reads masks as raw `L`
  indices; it never reads the JSONs or CSV. All 7,774 pairs load with 0 drops.
- **JSON/CSV anomalies have no loader impact.** Unfiltered `annotations[]`, local id spaces, and the 12
  Metadata `Index` errors are invisible to the training path.
- **COCO `category_id` + masks are authoritative** for training/evaluation labels.
- **`Metadata.csv` `Index` must NOT be used as the training label source** (12 known errors). It remains
  fine for human-readable Plant/Disease names and the Split column (which agrees image-for-image with the
  folders after `training→train`, `validation→val` canonicalization).

## 5. RESIDUAL CONTRACT / MODELING DECISIONS

| Decision | Recommendation / note |
|---|---|
| Disease-only mIoU | **Exclude background class 0** (report bg separately or alongside). This is a metric-convention choice; data supports both (`reduce_zero_label=False`). |
| Absent-class IoU | Class 41 (no test) and class 68 (no val) have no eval support → use **NaN/skip**, do **not** force 0 (forcing 0 would bias mIoU down). |
| Teacher-init weights | Still needs authoritative setup (`mim download` in the pinned env; fill URL/SHA/date + run `scripts/test_teacher_init.py`). From the original transfer summary. |
| Authoritative QNNPACK run | Still needs the pinned env (Python 3.11 / torch 2.1.0+cu121 with QNNPACK); local proxy was onednn. |
| `docs/reference/reference.pdf` | **Pre-existing modified (` M`), untouched** by this audit — commit/restore decision deferred to you. |

## 6. POINTER INDEX (which report answers what)

| Step | Report | Answers |
|---|---|---|
| B16-0 | [dataset_code_expectation_audit.md](dataset_code_expectation_audit.md) | What the loader code expects (root, folders, num_classes, ignore, EXIF, preprocessing) vs reality |
| B16-1 | [annotation_train_audit.md](annotation_train_audit.md) | train JSON validity, schema, counts, dup/null, categories, unfiltered-array finding |
| B16-2 | [images_train_audit.md](images_train_audit.md) | train image inventory + folder↔JSON stem match |
| B16-3 | [annotations_train_audit.md](annotations_train_audit.md) | train mask inventory + stem cross-check |
| B16-4 | [train_split_consistency_audit.md](train_split_consistency_audit.md) | train 3-way reconciliation + loader pairing sim |
| B16-5 | [annotation_val_audit.md](annotation_val_audit.md) | val JSON validity + unfiltered/local-id findings |
| B16-6 | [images_val_audit.md](images_val_audit.md) | val image inventory + folder↔JSON match |
| B16-7 | [annotations_val_audit.md](annotations_val_audit.md) | val mask inventory + stem cross-check |
| B16-8 | [val_split_consistency_audit.md](val_split_consistency_audit.md) | val 3-way reconciliation + pairing sim |
| B16-9 | [annotation_test_audit.md](annotation_test_audit.md) | test JSON validity + FLAG-F (0-annotation image) |
| B16-10 | [images_test_audit.md](images_test_audit.md) | test image inventory + FLAG-F image presence |
| B16-11 | [annotations_test_audit.md](annotations_test_audit.md) | test mask inventory + FLAG-F mask presence |
| B16-12 | [test_split_consistency_audit.md](test_split_consistency_audit.md) | test 3-way reconciliation + pairing sim (FLAG-F pair) |
| B16-13a | [test_mask_value_audit.md](test_mask_value_audit.md) | test mask pixel values; mask=category_id+1 (test); FLAG-F mask content |
| B16-13b | [trainval_mask_value_audit.md](trainval_mask_value_audit.md) | train+val mask pixel values; mask=category_id+1 (train/val); class coverage |
| B16-14 | [metadata_csv_audit.md](metadata_csv_audit.md) | Metadata.csv validity; Index↔COCO reconciliation (the 12); reduce_zero_label resolution |
| B16-15 | dataset_audit_summary.md (this) | consolidated closeout + residual decisions |
| (base) | [dataset_report.md](dataset_report.md) / [.json](dataset_report.json) | original full read-only EXIF-aware scan (counts, labels, encoding) |

## 7. RECOMMENDED NEXT ACTION

The dataset audit is complete and PASS. Two clean options:

- **(a) Commit the B16 report set only** (the 17 `reports/*audit*.md` + `dataset_audit_summary.md`),
  explicitly **excluding** `docs/reference/reference.pdf` (pre-existing modified, decision pending). See the
  commit prompt in the chat summary.
- **(b) Proceed to modeling blockers** (teacher-init weights, authoritative QNNPACK run, training loop)
  after committing the reports.

**Recommendation:** do (a) first (capture the audit), then (b). The `reference.pdf` modification and the
disease-only/absent-class metric conventions are separate decisions to make when convenient.

---

## Status note
The B16 report files are currently **untracked** (`git status` shows `??`), **not staged**.
`docs/reference/reference.pdf` is **modified (` M`)** and was **not touched** by this audit.

_End of B16-15. Only existing `reports/*` were read; only this report was written; nothing committed; no
memory/`~/.claude` writes._
