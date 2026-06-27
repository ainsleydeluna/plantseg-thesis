# PlantSeg Dataset Verification Report

**STATUS: PASS**  
_Generated: 2026-06-27 by scripts/verify_plantseg_dataset.py (read-only; full dataset; EXIF-aware)._

- Dataset root: `C:\Users\admin\plantseg_data\plantseg`

## ⚠️ Warnings (non-fatal)
- 8 raw image/mask dimension transpose(s) due to EXIF orientation (resolved by exif_transpose; not a failure)

## Extracted structure (top level)
- `annotations` (DIR)
- `images` (DIR)
- `annotation_test.json` (FILE)
- `annotation_train.json` (FILE)
- `annotation_val.json` (FILE)
- `Metadata.csv` (FILE)

## File extensions
- images: `{'.jpg': 7774}` | masks: `{'.png': 7774}`
- metadata files: `['annotation_test.json', 'annotation_train.json', 'annotation_val.json', 'Metadata.csv']`

## Counts & identifiers
- total images: **7774** | total masks: **7774** | shared: **7774**
- images without mask: 0 | masks without image: 0
- duplicate stems (img/mask): 0/0

## Split mechanism
- mechanisms found: **['csv', 'folder', 'json']** | authoritative (folder→json→csv): **folder**
- counts (authoritative): `{'train': 5367, 'val': 846, 'test': 1561}`
- counts per mechanism: `{'folder': {'train': 5367, 'val': 846, 'test': 1561}, 'json': {'train': 5367, 'val': 846, 'test': 1561}, 'csv': {'train': 5367, 'val': 846, 'test': 1561}}`
- CSV metadata: `{'file': 'Metadata.csv', 'headers': ['Name', 'Index', 'Plant', 'Disease', 'Resolution', 'Label file', 'Mask ratio', 'URL', 'License', 'Split'], 'name_col': 'Name', 'split_col': 'Split', 'rows': 7774}`
- pairwise zero-overlap (must be 0): `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

## EXIF orientation (raw)
- orientation distribution: `{'None': 7366, '1': 391, '0': 8, '6': 8, '3': 1}`
- raw-dimension transpose warnings: **8** (resolved by `exif_transpose`)
- examples: `['squash_powdery_mildew_google_0105: raw image(981, 736) vs mask(736, 981) (resolved by EXIF)', 'tomato_septoria_leaf_spot_google_0033: raw image(450, 600) vs mask(600, 450) (resolved by EXIF)', 'tomato_early_blight_google_0056: raw image(450, 600) vs mask(600, 450) (resolved by EXIF)', 'tomato_early_blight_google_0000: raw image(450, 600) vs mask(600, 450) (resolved by EXIF)', 'tomato_early_blight_google_0043: raw image(600, 450) vs mask(450, 600) (resolved by EXIF)', 'apple_rust_google_0065: raw image(848, 636) vs mask(636, 848) (resolved by EXIF)', 'grape_leaf_spot_google_0041: raw image(800, 600) vs mask(600, 800) (resolved by EXIF)', 'tomato_bacterial_leaf_spot_42: raw image(438, 364) vs mask(364, 438) (resolved by EXIF)']`
- raw .size transposes are EXIF orientation; masks match after ImageOps.exif_transpose on images. Treated as WARNING, not failure.

## Readability & dimensions (EXIF-aware)
- unreadable images: 0 | unreadable masks: 0
- EXIF-corrected dimension mismatches: **0**
- raw transpose warnings: 8 | distinct image sizes: 3895

## Mask label analysis (full dataset)
- channel modes: `{'L': 7774}` | ndim: `{2: 7774}`
- single-channel indexed: **True** | color masks: **False**
- unique values: **116** | contiguous 0..max: True
- contains 255: **False** — 255 absent in raw masks — expected; the ignore label is introduced during preprocessing padding, not present in the released annotations
- max non-ignore: **115** | min non-ignore: **0**
- values: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]`

## Label-index diagnostics (index 0 and 115)
- **index 115**: pixels=17,713,518 (0.0027 of all pixels), masks containing=56/7774, splits={'train': 39, 'val': 6, 'test': 11}
- **index 0**: pixels=5,348,416,376 (0.8064), masks containing=7773/7774
- Metadata.csv: `{'rows': 7774, 'index_min': 0, 'index_max': 114, 'index_distinct': 115, 'index_115_present': False, 'name_index_0': ('Apple', 'apple black rot'), 'name_index_114': ('Zucchini', 'zucchini yellow mosaic virus'), 'name_index_115': None}`
- COCO annotations: `{'category_id_min': 0, 'category_id_max': 114, 'category_id_distinct': 115, 'categories_list_len': 0, 'category_115_used': False}`
- background scan (most-present value): `{'background_index': 0, 'masks_containing_background': 7773, 'background_pixel_fraction': 0.806409, 'top5_by_mask_presence': [[0, 7773], [39, 323], [105, 229], [111, 228], [60, 211]]}`
- mask-value encoding: `{'nonzero_label_min': 1, 'nonzero_label_max': 115, 'model': 'mask_value = category_id + 1', 'per_image_shift_checked': 7774, 'per_image_shift_consistent': 7762, 'consistent_fraction': 0.998456, 'shift_exceptions_examples': ['broccoli_alternaria_leaf_spot_Bing_0021: metaIndex=23 expected nonzero {24} got [25]', 'broccoli_alternaria_leaf_spot_Bing_0045: metaIndex=23 expected nonzero {24} got [25]', 'broccoli_alternaria_leaf_spot_Bing_0054: metaIndex=23 expected nonzero {24} got [25]', 'wheat_head_scab_Bing_0287: metaIndex=104 expected nonzero {105} got [106]', 'bean_mosaic_virus_23: metaIndex=12 expected nonzero {13} got [82]', 'blueberry_mummy_berry_Bing_0058: metaIndex=20 expected nonzero {21} got [22]', 'broccoli_alternaria_leaf_spot_Bing_0012: metaIndex=23 expected nonzero {24} got [25]', 'broccoli_alternaria_leaf_spot_Bing_0035: metaIndex=23 expected nonzero {24} got [25]', 'broccoli_alternaria_leaf_spot_Bing_0039: metaIndex=23 expected nonzero {24} got [25]', 'cucumber_powdery_mildew_google_0067: metaIndex=50 expected nonzero {51} got [49]', 'eggplant_phomopsis_fruit_rot_Bing_0015: metaIndex=52 expected nonzero {53} got [54]', 'ginger_leaf_spot_17: metaIndex=56 expected nonzero {57} got [58]']}`

## Background / disease determination
- empirical background index: **0** (ubiquitous & dominant: True)
- mask-value encoding: **mask_value = category_id + 1** (consistent fraction 0.998456)
- disease classes named 0–114 in Metadata: **True**
- index 0 → **BACKGROUND — mask value 0 is in 7773/7774 masks (80.6% of pixels). NOTE: Metadata Index 0 names a disease (('Apple', 'apple black rot')), but mask pixel value 0 is background because the mask encoding is category_id+1.**
- index 115 → **DISEASE class = category_id 114 + 1 (Metadata 114 = ('Zucchini', 'zucchini yellow mosaic virus')); appears in 56 masks only (class-specific, NOT background)**
- background explicitly named in files: **False**
- disease-only mapping (empirical): **background = mask value 0 (exclude); diseases = mask values 1-115 (= category_id + 1)**
- disease-only fully explicit: **False** (False → retain NEED_TO_CONFIRM)
- note: Determined empirically with high confidence, but no literal 'background' category is named in the released metadata/annotations (COCO categories list is empty), so a residual NEED_TO_CONFIRM is retained per protocol.

## num_classes critical conflict (no silent winner)
- status: **RESOLVED_TOWARD_116 (empirical max=115)**
- finding: empirical max label 115; num_classes=116 appears consistent if labels include 0-115; do not change masks.

## Inconsistencies vs docs/IMPLEMENTATION_CONTRACT.md
- train split 5367 != contract approx 5442 (empirical split is authoritative)
- val split 846 != contract approx 778 (empirical split is authoritative)
- test split 1561 != contract approx 1554 (empirical split is authoritative)

## Final summary

`TOTAL_IMAGES = 7774 | TEST_SIZE = 1561 | MAX_NON_IGNORE_LABEL = 115 | NUM_CLASSES_DECISION = num_classes=116 supported (labels 0-115)`
