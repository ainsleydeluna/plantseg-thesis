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
    "grad_clip_max_norm": None,           # D-A/D2 resolved: E1 is intentionally unclipped by default as a documented deviation because Chapter 3 gives no numeric max_norm. The optional train_e1.py hook remains available via --grad-clip-norm if instability occurs. See docs/open_questions.md D2.
    "teacher_in_loop": "eval mode, online, identical augmented input as student",

    # Scope note: E1=this recipe with no distillation; E2/E3 add distill terms on top (see configs/distill.py).
    "shared_by": ("E1", "E2", "E3"),

    # ---------------------------------------------------------------- clipping decision (PREREGISTERED)
    # THE PROBLEM. Chapter 3 says "global-norm, throughout" and names no threshold, so D2/D-A resolved
    # E1 as intentionally UNCLIPPED rather than inventing a number. The E2/E3 and E5/E6 launchers
    # meanwhile REQUIRE an explicit positive max_norm. Left as-is that produces a real confound: E1
    # would run unclipped while every distilled/quantized stage ran clipped, so E1-vs-E2, E1-vs-E3 and
    # E1-vs-E6 would differ in TWO ways instead of one.
    #
    # WHY NOT JUST PICK A NUMBER. No authoritative source fixes one — not Chapter 3 (D2), not the
    # contract (method family only), and not the KD/CWD/quantization primary papers, whose reported
    # recipes cover optimizer/schedule/temperature/loss weights rather than a clipping threshold.
    # Krishnamoorthi's ImageNet-scale step counts must not be transplanted literally onto a ~5.3k-image
    # PlantSeg fine-tune (source-hierarchy rule), and a guessed threshold would silently become an
    # unregistered experimental variable.
    #
    # THE DECISION. One preregistered pilot selects ONE clipping rule, which is then frozen and applied
    # IDENTICALLY to every stage that trains (E2, E3, E5, E6). "No clipping" is a first-class candidate
    # precisely so the rule can match E1's already-resolved deviation and remove the confound.
    "grad_clip_pilot": {
        "status": "PILOT_REQUIRED",          # no value is selected yet; launchers must keep refusing
        "question": "one global-norm clipping rule shared by every training stage",
        "candidates": ("none", 1.0, 5.0),    # 'none' = unclipped, matching E1 (D2/D-A)
        "splits_used": ("train", "val"),     # TEST is never touched by any selection procedure
        "test_used_for_selection": False,
        "seed": 42,
        "selection_metric": "dataset-level validation mIoU",
        "tie_rule": "prefer 'none' (matches E1, minimises confounding), then the smaller max_norm",
        "run_on": "E2",                      # cheapest stage that actually exercises distillation grads
        "applies_to": ("E2", "E3", "E5", "E6"),
        "budget": "one shortened E2 run per candidate, identical iteration budget and seed",
        "label": "HYPERPARAMETER SELECTION / PILOT",   # never an official E2 result
        "freeze_before": ("E2", "E3", "E5", "E6"),
        "excluded_from": ("test evaluation", "robustness", "hypothesis testing", "statistics family"),
    },
}
