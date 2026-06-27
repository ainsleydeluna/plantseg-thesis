# Dataset config — PlantSeg (Wei et al., 2026)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (c) "Dataset facts" + (e)/(f) preprocessing.
# Dataset facts traced to Wei (2026); preprocessing traced to ch3.pdf.
# num_classes / reduce_zero_label are OPEN (resolved by np.unique before E1) and root is not a
# locked value, so all three stay the literal NEED_TO_CONFIRM string (rules 5 + golden rule).
# Analysis/config artifact only — contains NO training logic.

DATA = {
    # Root (extracted dataset path) — not a locked value anywhere; dataset not downloaded
    "root": "NEED_TO_CONFIRM",

    # Class space — OPEN conflict (Wei: 0-114 = 115 disease; ch3: provisional 116). DO NOT HARDCODE.
    "num_classes": "NEED_TO_CONFIRM",            # size to verified np.unique() count before E1
    "reduce_zero_label": "NEED_TO_CONFIRM",      # background-index encoding unresolved
    "ignore_index": 255,
    "mask_indices_documented": "0-114 per Wei (2026); verify empirically via np.unique()",

    # Splits (official 70/10/20)
    "splits": {
        "ratio": "70/10/20",
        "files": ("annotation_train.json", "annotation_val.json", "annotation_test.json"),
        "sizes_approx": {"train": 5442, "val": 778, "test": 1554},
        "test_count": 1554,                      # per-image metric unit count
        "integrity_check": "zero identifier overlap across partitions",
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
