# Supervised loss config — Blockers B5 + B11b
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B5 - Loss".
# Every value traced to ch3.pdf sections C "E1" and C.1. Locked / smoke-safe scalars only —
# unresolved values stay the literal NEED_TO_CONFIRM string. NO training logic.

from configs.data import DATA

LOSS = {
    # --- locked / smoke-safe scalars (B11b) ---
    "num_classes": DATA["num_classes"],          # 116 (empirical; reports/dataset_report.md)
    "objective": "CE + Dice (E1, equal weight)",
    "ce_dice_enabled_E1": True,
    # --- unresolved (never guessed) ---
    "real_class_weights": "NEED_TO_CONFIRM",     # sqrt inv-freq over FULL train set (not yet computed)
    "logit_kd_weight": "NEED_TO_CONFIRM",        # lambda_logit; validation sweep {0.25,0.5,1,2,4}
    "reduce_zero_label": "NEED_TO_CONFIRM",      # background convention unresolved (open_questions #2)

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
