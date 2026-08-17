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
    # RECOMMENDATION, NOT YET AUTHORITATIVE. A real QAT launch must supply these explicitly, and the
    # runner deliberately refuses to default them (src/quant/runner.py::unresolved_qat_values). What
    # is recorded here is the analysis of WHICH value each control should take and, critically, that
    # every one of them must be IDENTICAL for E5 and E6 — the two stages may differ only in source
    # checkpoint and distillation history, never in arbitrary QAT tuning, or E5-vs-E6 stops being a
    # controlled comparison (`shared_by`).
    #
    # STATUS is "RECOMMENDED_PENDING_AUTHORIZATION" rather than locked because two entries would
    # overturn a decision already recorded in the runner: B4's QAT table has NO batch-size and NO
    # weight-decay row, and the runner explicitly notes that the 1e-4/16 belong to B2's student SGD
    # recipe, accepting "any finite >= 0" weight decay including 0. Inheriting them here is a
    # defensible consistency argument, but it is a CHANGE of that recorded position, so it is proposed
    # rather than silently applied.
    "qat_real_run_recommendation": {
        "status": "RECOMMENDED_PENDING_AUTHORIZATION",

        # Consistency with the registered E1/E2/E3 recipe. NOTE the conflict described above.
        "batch_size": 16,                    # = E1_STUDENT["batch_size"]
        "weight_decay": 1e-4,                # = E1_STUDENT["weight_decay"]; B4 has no such row
        "overturns_recorded_position": ("batch_size", "weight_decay"),

        # The methodology's own "~15 epochs" read as a MAXIMUM budget, not a target: early stopping on
        # validation mIoU decides the actual length and selection is best-val-mIoU regardless, so 15
        # bounds the run rather than tuning it.
        "max_epochs": 15,

        # Bounded impact by construction — the reported model is the best-val-mIoU checkpoint, so
        # patience only decides how long to keep looking, never which weights win.
        "early_stop_patience": 3,            # validations (= epochs at this cadence)

        # The literal endpoints of the contract's registered "~65-70%" window, so nothing is invented
        # inside the range and "observer freeze shortly after BN freeze" becomes an explicit
        # 5%-of-budget gap with the required ordering.
        "bn_freeze_pct": 0.65,
        "observer_freeze_pct": 0.70,

        # See `grad_clip_pilot` in configs/e1_student.py: no authoritative source fixes a numeric
        # max_norm and E1 is already resolved as unclipped (open_questions D2/D-A), so the rule is
        # pilot-selected and then applied identically to every training stage.
        "grad_clip": "PILOT_REQUIRED",

        "shared_by": ("E5", "E6"),
        "enforced_at_launch": False,         # the runner still requires each value explicitly
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
