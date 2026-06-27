# Dataset config — PlantSeg (Wei et al., 2026)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (c) "Dataset facts" + (e)/(f) preprocessing.
# Dataset facts traced to Wei (2026); preprocessing traced to ch3.pdf.
# num_classes/root resolved empirically 2026-06-27 (see reports/dataset_report.md). reduce_zero_label
# stays NEED_TO_CONFIRM (background convention not explicitly named in dataset files; see open_questions #2).
# Analysis/config artifact only — contains NO training logic.

DATA = {
    # Root (extracted dataset path) — verified to exist; see reports/dataset_location_log.md
    "root": r"C:\Users\admin\plantseg_data\plantseg",

    # Class space — empirically verified: mask values 0-115 (background 0 + 115 diseases 1-115).
    "num_classes": 116,                          # all-class; output layer = 116  [empirical]
    "reduce_zero_label": "NEED_TO_CONFIRM",      # background=index 0 empirically, but not explicitly named
    "ignore_index": 255,                         # padding/ignore; ABSENT from raw masks, added at preprocessing
    "background_index": 0,                       # [empirical] disease-only metrics exclude this
    "mask_indices_documented": "verified 0-115: 0=background, 1-115=diseases (mask=category_id+1)",

    # Splits (official 70/10/20) — pre-partitioned on disk as images/<split>/ + annotations/<split>/
    "splits": {
        "ratio": "70/10/20",
        "dirs": ("images/{split}", "annotations/{split}"),  # use folders, NOT annotation_*.json
        "names": ("train", "val", "test"),
        "files": ("annotation_train.json", "annotation_val.json", "annotation_test.json"),  # COCO; not used for split
        "sizes": {"train": 5367, "val": 846, "test": 1561},  # [empirical]
        "test_count": 1561,                      # per-image metric unit count
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
