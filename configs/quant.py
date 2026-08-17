# Quantization config — Blocker B4 (INT8 QAT + PTQ)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B4 - Quantization"
# Every value traced to ch3.pdf sections C "E5"/"E6"/"E4"/"E7", D, E.1.
# The Sigmoid FixedQParams fix encodes the contract's gate; the contract gives NO numeric
# qparams, so scale/zero_point stay the literal NEED_TO_CONFIRM string (rule 5).
# Analysis/config artifact only — contains NO training logic.

QUANT = {
    # INT8 Quantization-Aware Training (E5 from E1; E6 from E3 head-removed, identical config to E5)
    "qat": {
        "backend": "QNNPACK",
        "toolchain": "eager-mode torch.ao.quantization",
        "optimizer": "SGD",
        "momentum": 0.9,
        "learning_rate": 3e-4,
        "lr_schedule": "cosine",
        "epochs_approx": 15,                         # "~15 epochs" per ch3
        "early_stop": "val_miou",
        "activation_quant_start": "step 0",
        "observer": "moving_average",
        "bn_freeze_pct": "65-70",
        "observer_freeze": "shortly after BN freeze",
        "gradient_clipping": "global_norm",
        "weight_ema": False,
        "checkpoint_selection": "best_val_miou",
        "distillation_during_qat": False,            # E5/E6 supervised-only
        "weight_quant": "per-channel symmetric INT8 (all conv)",
        "activation_quant": "per-tensor asymmetric UINT8 (full 8-bit range)",
        "fusion": "Conv-BN-ReLU via fuse_modules before observer insertion",
        "standalone_ops": ("Hard-Swish", "Hardsigmoid"),
        "e5_from": "E1",
        "e6_from": "E3 (CWD head removed)",
        "e6kd_reduced_weights": "NEED_TO_CONFIRM",   # contingency only if E3->E6 clean mIoU drop > 1.0 pp
    },

    # ---------------------------------------------------------------- E5/E6 real-run controls
    # LOCKED. These are THESIS IMPLEMENTATION CHOICES, explicitly authorized as such — they are NOT
    # values specified by Krishnamoorthi, Jacob, or any other source, and must never be cited as
    # though they were. Every entry applies IDENTICALLY to E5 and E6 (`shared_by`), so the two stages
    # differ only in source checkpoint and distillation history rather than in QAT tuning.
    #
    # A real launch still supplies each value explicitly; the runner defaults none of them.
    "qat_real_run": {
        "status": "LOCKED",
        "basis": "thesis implementation choice (authorized); not specified by the source papers",

        # PHYSICAL batch size — not an effective/accumulated one. Gradient accumulation is deliberately
        # NOT introduced: fake-quant observers and BatchNorm statistics are batch-sensitive, so N
        # accumulated micro-batches are not equivalent to one true batch of N. Preserves the registered
        # student-training scale instead of adding another E5/E6 difference.
        "batch_size_physical": 16,
        "gradient_accumulation": None,        # never silently enabled; see docs note
        "batch_semantics": "physical batch of 16; if it cannot fit, that is an experiment-design issue",

        # Preserves the registered student regularization scale rather than introducing a new one.
        "weight_decay": 1e-4,

        # Converts the methodology's "~15 epochs" into a reproducible MAXIMUM budget. Early stopping
        # decides the actual length; best-val-mIoU selection remains a separate mechanism.
        "max_epochs": 15,

        # Makes the already-required early stopping executable. Counted in validation checks.
        "early_stop_patience": 3,

        # Optimizer-step FRACTIONS of the planned budget (not rounded epoch prose). 0.65 is the lower
        # registered endpoint of the methodology's 65-70% BN-freeze window; 0.70 places observer
        # freezing shortly afterwards. Rounding convention: round(total_iters * pct), floored at step 1,
        # with the observer freeze clamped never to precede the BN freeze.
        "bn_freeze_pct": 0.65,
        "observer_freeze_pct": 0.70,
        "freeze_rounding": "round(total_iters * pct), min step 1, observer >= bn",

        # A QAT run must not end before its quantization schedule has run. Patience therefore accrues
        # only after observer freeze; best-checkpoint tracking still starts at the first validation.
        "fake_quant_start": "step 0",
        "early_stop_eligible_after": "observer_freeze",

        # The QAT clipping threshold is NOT part of this locked set — see `qat_grad_clip_pilot`.
        "grad_clip": "PILOT_REQUIRED",

        "shared_by": ("E5", "E6"),
    },

    # ---------------------------------------------------- E5/E6 gradient clipping (PREREGISTERED)
    # INDEPENDENT of the distillation decision (DISTILL["distillation_grad_clip_pilot"]). QAT and
    # distillation are different optimization regimes and no source establishes that one numeric norm
    # should serve both, so the two are selected separately. E5 and E6 must share the SAME selected
    # value. "No clipping" is not a candidate: the methodology mandates global-norm clipping for QAT.
    "qat_grad_clip_pilot": {
        "name": "QAT_GRAD_CLIP_NORM",
        "status": "PILOT_REQUIRED",              # unresolved; official E5/E6 launches must refuse
        "selected_value": None,
        "candidates": (1.0, 5.0),                # both positive finite
        "applies_to": ("E5", "E6"),
        "run_on": "E5",
        "source_checkpoint": "official E1 FP32 checkpoint",
        "seed": 42,
        "splits_used": ("train", "val"),
        "test_used_for_selection": False,
        "pilot_budget_epochs": 5,                # shortened; the official maximum stays 15
        "official_max_epochs": 15,
        "selection_rule": (
            "1) reject a candidate whose training becomes non-finite or numerically unstable; "
            "2) otherwise pick the higher dataset-level validation mIoU under the identical pilot "
            "budget; 3) on a tie within the repository's existing validation tie/noise rule prefer "
            "5.0 as the LESS INTRUSIVE clipping threshold."),
        "no_new_significance_test": True,
        "inherits_distillation_value": False,    # never auto-reused from the E2/E3 decision
        "label": "HYPERPARAMETER SELECTION / PILOT",
        "freeze_before": ("E5", "E6"),
        "excluded_from": ("test evaluation", "robustness", "hypothesis testing", "statistics family"),
    },

    # INT8 Post-Training Quantization (E4 from E1; E7 from E3 head-removed; same calib subset)
    "ptq": {
        "method": "static INT8",
        "calibration_num_images": 128,
        "calibration_seed": 42,
        "calibration_source": "training partition",
        "calibration_no_augmentation": True,
        "calibration_preprocessing": "same as clean test",
        "calibration_subset_shared": ("E4", "E7"),
        "activation_observer": "histogram (minimizes quantization error)",
        "weight_observer": "per-channel min/max",
        "weight_quant": "per-channel symmetric INT8 (all conv)",
        "activation_quant": "per-tensor asymmetric UINT8",
        "e4_from": "E1",
        "e7_from": "E3 (CWD head removed)",
        "selection": "config selected on validation, reported on test",
    },

    # Sigmoid FixedQParams fix for the LR-ASPP global-pool branch (gate before E4)
    "sigmoid_fixedqparams_fix": {
        "observer": "FixedQParamsObserver",
        "fake_quant": "FixedQParamsFakeQuantize",
        "scale": "NEED_TO_CONFIRM",                  # not specified in contract
        "zero_point": "NEED_TO_CONFIRM",             # not specified in contract
        "gate": "verify QNNPACK INT8 Sigmoid support in LR-ASPP global-pool branch before E4",
        "fallback": "unsupported op -> dequantize to FP32 (mixed-precision), reported",
    },
}
