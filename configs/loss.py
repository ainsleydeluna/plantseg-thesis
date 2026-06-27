# Supervised loss config — Blocker B5
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B5 - Loss"
# Every value traced to ch3.pdf sections C "E1" and C.1.
# Analysis/config artifact only — contains NO training logic.

LOSS = {
    "supervised": "L_sup = L_CE + L_Dice (equal weight)",

    "cross_entropy": {
        "class_weighted": True,
        "weighting": "sqrt inverse-frequency",
        "weight_formula": "sqrt( total_train_pixels / (num_classes * pixels_in_class) )",
        "computed_over": "non-255 pixels only",
        "normalization": "divide by median weight",
        "reporting": "realized min / median / max (no fixed cap)",
        "ignore_index": 255,
    },

    "dice": {
        "type": "soft, on predicted probabilities",
        "aggregation": "per-class macro over classes present in batch",
        "absent_classes": "excluded",
        "smoothing": 1e-5,           # added to numerator and denominator
        "weighted": False,
        "validity_mask": "applied before intersection / union",
    },

    # ignore_index 255 excluded from CE, Dice, Logit KD, CWD, and ALL metrics
    "ignore_index": 255,
    "ignore_excludes": ("CE", "Dice", "LogitKD", "CWD", "metrics"),
}
