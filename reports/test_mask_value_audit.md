# Test Mask VALUE Check (B16-13a, decode-gated)

**Scope:** read-only, **decode-gated** validation of **`annotations/test` mask pixels only** (explicitly
approved). Decoded all 1,561 `annotations/test/*.png` via `PIL.Image` + `numpy` (raw indices, no RGB/palette
expansion); read `annotation_test.json` for `file_name → image_id → category_id`. **No RGB images, no
train/val masks, no other JSON/CSV decoded.** No files modified.

**Interpreter:** project Anaconda Python (`C:\Users\admin\anaconda3\python.exe`; numpy 2.1.3, Pillow 11.1.0)
— the env the smokes use; the bare `Programs\Python\Python313` lacks numpy. No install performed.
**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c`.

> **Context:** first step to decode mask pixels. The loader reads masks as raw `L` indices (B16-0); this
> verifies the actual label content for the test split and tests the (previously inferred)
> `mask = category_id + 1` mapping.

---

## 0. PASS / FLAG verdict — **PASS** — `mask=category_id+1` **proven for test**; FLAG-F resolved (mask valid)

Every test mask is single-channel indexed (`L`, 2-D), all values in **0–115**, **255 absent**, all readable.
`mask = category_id + 1` holds **exactly for 100% of annotated test images (1,560/1,560)**. The one
non-annotated image (FLAG-F) has a **valid disease mask** (label 1 = apple black rot) — its COCO polygons
are simply missing, which does not affect the segmentation label. No mask-value violations.

---

## 1. TEST MASK LABEL-RANGE FINDINGS

| Property | Result |
|---|---|
| Masks decoded / expected | **1,561 / 1,561** (0 unreadable) |
| Mode distribution | `{'L': 1561}` (single-channel) |
| ndim distribution | `{2: 1561}` (all 2-D) |
| Aggregate distinct values | **115** |
| Min / max non-ignore | **0 / 115** |
| All values within 0–115 | ✅ **True** (0 range violations) |
| Labels 0–115 present | **115 of 116** — **absent: `[42]`** |
| Label 0 (background) | present in **1,561 / 1,561** masks (ubiquitous) |
| Label 115 (= category 114) | present in **11** masks |

**Absent value 42 (= category_id 41 + 1):** no test **image** carries disease class 41. (B16-9 saw
`category_id 41` in the *annotations array*, but that array is unfiltered — those entries are **orphan**,
referencing non-test images.) So the **test images cover 114 of the 115 disease classes** (all but class
41), plus background. This is normal for a held-out split, but means **per-class test IoU for class 41
(mask value 42) is undefined** (no support) — relevant to the metrics step.

## 2. TEST `255` PRESENCE/ABSENCE

- **255 absent** from every test mask (0 masks contain 255). Confirms 255 is introduced only at
  preprocessing (pad/rotation fill), never in released masks — consistent with `dataset_report`.

## 3. TEST `mask = category_id + 1` CONSISTENCY

Per image: `expected = {cat_id+1 for its COCO annotations}` vs `observed = unique mask values − {0,255}`.

| Metric | Count | % |
|---|---|---|
| Images checked | 1,561 | — |
| **Exact** (`observed == expected`) | **1,560** | 99.94% |
| **Consistent** (`observed ⊆ expected`) | **1,560** | 99.94% |
| Has **extra** mask labels (`observed − expected ≠ ∅`) | **1** | (the FLAG-F image) |
| Has **missing** JSON labels (`expected − observed ≠ ∅`) | **0** | 0% |
| Distinct category_ids per image | `{0: 1, 1: 1560}` | 1 image w/ 0 cats; 1,560 single-disease |

- **Among the 1,560 annotated test images, the mapping holds EXACTLY (100%)** — 0 missing, 0 extra.
- The single non-exact image is **FLAG-F** (`apple_black_rot_google_0001`): `expected = ∅` (no COCO
  annotations) but `observed = {1}`. This is **not a mapping violation** — the mask correctly labels the
  image as disease 1; the JSON simply lacks polygons. Counting it as consistent (mask matches its true
  class), test consistency is **1,561/1,561**.
- **One-disease-per-image confirmed:** every annotated test image has exactly one category_id, so each mask
  has exactly one disease label (+ background).

## 4. FLAG-F MASK CONTENT (`apple_black_rot_google_0001.png`)

| Field | Value |
|---|---|
| image_id | 8 |
| COCO category_ids (JSON) | `[]` (zero annotations) |
| Mask unique values | **`[0, 1]`** |
| Mask nonzero | `[1]` |
| Classification | **CONTAINS disease label 1** (= category_id 0 + 1 = "apple black rot") — **not** all-background |

**NTC-F RESOLVED:** the image is a legitimate diseased sample — its mask labels it apple black rot (label 1),
matching its filename. The image↔mask pair is valid and **loads with the correct GT label**. The only gap
is the absent COCO instance annotations (irrelevant to semantic-segmentation training/eval; matters only
for COCO-based instance tasks).

## 5. AGREEMENT WITH PRIOR `dataset_report`

| This audit (test) | `dataset_report` (full dataset) | Agreement |
|---|---|---|
| Masks single-channel `L`, 2-D | `channel_modes {'L':...}`, ndim 2 | ✅ |
| Values 0–115, 255 absent | contiguous 0–115, `contains_255=false` | ✅ |
| Label 115 in **11** test masks | `label_115.splits_present test = 11` | ✅ **exact** |
| `mask=category_id+1` exact 100% (annotated, vs **COCO category_id**) | 99.85% (12 exceptions, vs **Metadata.csv Index**) | ⚠️ different reference — see NTC |

> **Note on the 99.85% vs 100%:** `dataset_report` tested mask-vs-**Metadata.csv `Index`** and found 12
> full-dataset exceptions (off-by-one type). B16-13a tested mask-vs-**COCO `category_id`** and found **0**
> exceptions among annotated test images. The two label sources (Metadata `Index` vs COCO `category_id`)
> may themselves disagree for those 12 images — i.e. the masks may be correct and a Metadata/COCO
> mismatch is the real source. Confirming that needs `Metadata.csv` (not opened here) → NTC.

## 6. REMAINING `NEED_TO_CONFIRM`

| ID | Item | Why still open |
|---|---|---|
| **NTC-1** | `mask=category_id+1` for **train & val** masks | Only test decoded here. Train/val decode = B16-13b (separate approval). |
| **NTC-7** | The 12 `dataset_report` exceptions = Metadata↔COCO disagreement? | Needs `Metadata.csv` (not opened). Test (vs COCO) is clean, suggesting the masks are fine and Metadata is the odd source — to confirm. |
| **NTC-8** | Class 41 (mask 42) has **no test support** | Per-class test IoU for class 41 undefined; metrics step must handle absent-class NaN. |
| **NTC-5** | `reduce_zero_label` / explicit background naming | Open since B16-0 (background = value 0 empirically; not affected). |
| **NTC-F** | ~~FLAG-F mask content~~ | **RESOLVED** (mask = apple black rot; COCO polygons absent only). |

## 7. EXACT NEXT RECOMMENDED PROMPT (B16-13b — train & val mask-value check)

```
B16-13b — Train & Val mask VALUE check (decode-gated). Read-only. Plan-gated.

NOTE: decodes annotations/train and annotations/val PNG mask pixels (read-only). Does NOT
decode RGB images or test masks (test done in B16-13a). Approve mask-decode for train+val.
Use the project Anaconda python (C:\Users\admin\anaconda3\python.exe; numpy+Pillow present);
set PYTHONIOENCODING=utf-8. No install.

Goal: replicate B16-13a for train (5,367) and val (846): single-channel L/2D; values 0..115;
255 absent; mask=category_id+1 per image vs each image's COCO annotations (annotated images
100%? quantify exceptions); per-split label coverage (which of 0..115 absent); cross-check
label-115 per-split counts vs dataset_report (train=39, val=6).

Allowed to open: annotations/train/*.png, annotations/val/*.png (pixel read only);
annotation_train.json, annotation_val.json (image_id->category_id, file_name). Reuse prior
reports. Do NOT decode RGB images or test masks. Do NOT open Metadata.csv. No modify, no commit.

Before writing the report, show me your plan and wait for `go`.

Create or update only: reports/trainval_mask_value_audit.md
Separate: (1) train label/255/mapping result; (2) val label/255/mapping result; (3) per-split
class coverage + dataset_report agreement (label-115 train=39/val=6); (4) NEED_TO_CONFIRM
(incl. the 12 Metadata-vs-COCO exceptions -> Metadata.csv step); (5) next step (Metadata.csv
validation B16-14, or close out B16).

After finishing show: PASS/FLAG (train, val); label-range + 255 per split; mask=category_id+1
% per split; any absent classes per split; files created/changed; git status; exact next
prompt. Stop after B16-13b.
```

---

## 8. Required separations (recap)
**(1) Label range** → §1. **(2) 255** → §2. **(3) `mask=category_id+1`** → §3. **(4) FLAG-F** → §4.
**(5) `dataset_report` agreement** → §5. **(6) NEED_TO_CONFIRM** → §6. **(7) Next prompt** → §7.

_End of B16-13a. Only `annotations/test` masks (pixel-decoded) and `annotation_test.json` were read; only
this report was written; nothing committed._
