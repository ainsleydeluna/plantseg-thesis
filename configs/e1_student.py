# Student training config — Blocker B2 (shared E1 / E2 / E3 recipe)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B2 - Student training"
# Every value traced to ch3.pdf; init checkpoint per ch3 section D / Table 3.5.
# Analysis/config artifact only — contains NO training logic.

E1_STUDENT = {
    # Initialization
    "init_weights": "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2",  # quantizable variant, top-1 75.27%

    # Optimizer
    "optimizer": "SGD",
    "momentum": 0.9,
    "learning_rate": 1e-2,
    "lr_schedule": "polynomial",
    "lr_power": 0.9,
    "weight_decay": 1e-4,

    # Budget
    "batch_size": 16,
    "iterations": 80000,
    "input": (512, 512),

    # Validation / checkpoint
    "val_interval": 4000,
    "checkpoint_selection": "best_val_miou",  # D1: ALL-CLASS validation mIoU (headline + checkpoint); disease-only mIoU is secondary/provisional only; switching headline/checkpoint to disease-only needs a separate explicit decision. See docs/open_questions.md #2.

    # Shared training mechanics — applied per ch3 to the WHOLE E1/E2/E3 recipe (see `shared_by`); the
    # distill-specific items only take effect once E2/E3 add distillation.
    "distill_weight_ramp": "linear 0 -> target over first epoch",
    "gradient_clipping": "global_norm",   # D2: METHOD FAMILY only (global-norm) — NOT a numeric value and NOT proof clipping is active.
    "grad_clip_max_norm": None,           # D2 Option E: the global-norm hook exists in src/training/train_e1.py, but E1 clipping stays DISABLED (None) until a numeric max_norm is explicitly decided before the real E1 run. See docs/open_questions.md D2.
    "teacher_in_loop": "eval mode, online, identical augmented input as student",

    # Scope note: E1=this recipe with no distillation; E2/E3 add distill terms on top (see configs/distill.py).
    "shared_by": ("E1", "E2", "E3"),
}
