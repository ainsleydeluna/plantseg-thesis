# Train-only augmentation config — Blocker B2 (augmentation)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B2 - Augmentation (train split only)"
# Every value traced to ch3.pdf section E.2.c (the `library` label is reconciled to the NumPy/PIL
# implementation — see the D3 note below). Applied to the TRAIN split only;
# val / clean-test / corrupted-test are never augmented.
# Analysis/config artifact only — contains NO training logic.

AUGMENT = {
    # D3 (2026-07-01): ch3 §D names Albumentations as the library *target*; the implementation uses
    # project-local NumPy/PIL-style transforms in src/data/transforms.py rather than Albumentations —
    # behavior is the methodological target, and no Albumentations dependency is required (not in
    # requirements.lock). Only this label is reconciled to the actual implementation; behavior unchanged.
    "library": "hand-written NumPy/PIL transforms",  # joint image+mask; src/data/transforms.py (numpy/PIL/torch)
    "split": "train_only",

    # Geometric (applied jointly to image + mask)
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.5,
    "rotation": {
        "degrees": 10,            # +/-10
        "p": 0.5,
        "apply": "before_crop",
        "image_fill": "imagenet_mean",
        "mask_fill": 255,
    },
    "random_resized_crop": {
        "size": (512, 512),
        "scale_range": (0.75, 2.0),
        "aspect_ratio_preserved": True,
        "cat_max_ratio": 0.95,
        "mask_interpolation": "nearest",
    },

    # Photometric (image only)
    "photometric": {
        "hue": 0.015,             # +/-0.015
        "saturation_factor": (0.8, 1.2),
        "p": 0.5,
    },

    # Deliberately excluded (diagnostic color + data-contamination prohibition vs eval corruptions)
    "excluded": ("brightness", "contrast", "blur", "noise", "jpeg"),
}
