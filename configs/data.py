# Dataset config — PlantSeg (Wei et al., 2026)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (c) "Dataset facts" + (e)/(f) preprocessing.
# Dataset facts traced to Wei (2026); preprocessing traced to ch3.pdf.
# num_classes/root resolved empirically 2026-06-27 (see reports/dataset_report.md). reduce_zero_label is
# set False (PROVEN no-remap: masks already 0-115, bg=0 kept); the separate disease-only metric convention
# (exclude index 0 in reporting) remains open — see open_questions #2.
# Analysis/config artifact only — contains NO training logic.

import os

# Dataset root is portable via the PLANTSEG_DATA_ROOT environment variable. Set PLANTSEG_DATA_ROOT on
# RunPod/Linux to avoid editing this file; if it is unset, the local Windows default below is used for the
# current development machine (behavior unchanged). See reports/e1_runpod_launch_runbook.md.
DEFAULT_PLANTSEG_DATA_ROOT = r"C:\Users\admin\plantseg_data\plantseg"

# ------------------------------------------------------------------ split sizes (single source of truth)
# COUNTED from the pre-partitioned folders on disk (B16: zero identifier overlap across partitions);
# re-verified 2026-09-01 by reports/pre_e1_launch_audit.md E15. These values are AUTHORITATIVE.
# ch3's 5,442 / 778 / 1,554 is a known arithmetic artifact of applying a nominal 70/10/20 ratio to
# 7,774 (7774*0.70 = 5441.8); the actual official split is 69.0 / 10.9 / 20.1%. The manuscript is
# under correction — do NOT edit these constants to match it.
SPLIT_SIZES = {"train": 5367, "val": 846, "test": 1561}     # [empirical, counted]
SPLIT_TOTAL = 7774                                          # [empirical, counted]

if sum(SPLIT_SIZES.values()) != SPLIT_TOTAL:
    raise RuntimeError(
        f"configs/data.py is internally inconsistent: sum(SPLIT_SIZES)={sum(SPLIT_SIZES.values())} "
        f"!= SPLIT_TOTAL={SPLIT_TOTAL}")

DATA = {
    # Root (extracted dataset path) — verified to exist; see reports/dataset_location_log.md.
    # PLANTSEG_DATA_ROOT env var overrides this when set; otherwise the Windows default is used.
    "root": os.environ.get("PLANTSEG_DATA_ROOT", DEFAULT_PLANTSEG_DATA_ROOT),

    # Class space — empirically verified: mask values 0-115 (background 0 + 115 diseases 1-115).
    "num_classes": 116,                          # all-class; output layer = 116  [empirical]
    "reduce_zero_label": False,                  # PROVEN no-remap: masks are already 0-115 (bg=0 kept). See reports/dataloader_smoke.md and reports/dataset_audit_summary.md. NOTE: the separate disease-only metric convention (exclude index 0 in reporting) is NOT controlled by this flag and remains open; see docs/open_questions.md #2.
    "ignore_index": 255,                         # padding/ignore; ABSENT from raw masks, added at preprocessing
    "background_index": 0,                       # [empirical] disease-only metrics exclude this
    "mask_indices_documented": "verified 0-115: 0=background, 1-115=diseases (mask=category_id+1)",

    # Splits (official 70/10/20) — pre-partitioned on disk as images/<split>/ + annotations/<split>/
    "splits": {
        "ratio": "70/10/20",
        "dirs": ("images/{split}", "annotations/{split}"),  # use folders, NOT annotation_*.json
        "names": ("train", "val", "test"),
        "files": ("annotation_train.json", "annotation_val.json", "annotation_test.json"),  # COCO; not used for split
        "sizes": SPLIT_SIZES,                    # [empirical] single source of truth: SPLIT_SIZES above
        "test_count": SPLIT_SIZES["test"],       # per-image metric unit count
        "integrity_check": "zero identifier overlap across partitions (verified)",
    },

    # Formats
    "image_format": "JPEG",
    "mask_format": "PNG grayscale",

    # Core preprocessing applied to ALL partitions (val / clean-test use exactly this, no augmentation)
    "preprocess_val_test": {
        "resize": "aspect-ratio preserving, long side -> 512",
        "pad_to": (512, 512),
        "image_pad_value": (124, 116, 104),      # ImageNet mean in 8-bit terms
        "mask_pad_value": 255,
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "scale": (0.0, 1.0),
        "normalize_mean": (0.485, 0.456, 0.406),
        "normalize_std": (0.229, 0.224, 0.225),
        "image_dtype_layout": "float32 CHW",
        "mask_dtype_layout": "int64 HW",
    },

    # Train partition = core preprocessing + train-only augmentation
    "preprocess_train": "same core preprocessing + augmentation (see configs/augment.py)",

    # Provenance
    "doi": "10.5281/zenodo.17719108",
    "license": "CC BY-NC 4.0",
}
