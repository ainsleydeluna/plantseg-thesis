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

    # ---------------------------------------------------------------- clipping scope note
    # AM-7 (docs/PREREGISTRATION_AMENDMENTS.md; DL-04): E1, E2 and E3 share one rule, NO clipping. E1
    # was already unclipped (open_questions D2/D-A); train_distill's real-run gate refuses
    # --grad-clip-norm for E2/E3. Chapter 3 asks for global-norm clipping in the distillation stages and
    # in QAT; for the distillation stages AM-7 replaces that, and QAT keeps its own threshold:
    #   * E2/E3  -> none (AM-7). configs/distill.py DISTILL["distillation_grad_clip_pilot"] is the
    #               WITHDRAWN_AM7 record of the pilot that would have selected one; the
    #               grad_clip_scope["e2_e3"] value below still names that record.
    #   * E5/E6  -> QUANT["qat_grad_clip_pilot"]             (configs/quant.py)
    #
    # CARRY FORWARD (manuscript, not resolved here): Chapter 3 describes E1/E2/E3 as sharing an
    # identical recipe and attributes their differences to the distillation objectives, yet applies
    # clipping only to the distillation stages. That wording needs reconciling; execution is governed
    # by AM-7 (E1-E3 unclipped).
    "grad_clip_scope": {
        "e1": "unclipped (D2/D-A)",
        "e2_e3": "DISTILL['distillation_grad_clip_pilot']",
        "e5_e6": "QUANT['qat_grad_clip_pilot']",
        "shared_numeric_threshold_required": False,
        "manuscript_reconciliation_pending": True,
    },
}
