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
