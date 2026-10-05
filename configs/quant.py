# Quantization config — Blocker B4 (INT8 QAT + PTQ)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B4 - Quantization"
# Every value traced to ch3.pdf sections C "E5"/"E6"/"E4"/"E7", D, E.1.
# The Sigmoid FixedQParams fix encodes the contract's gate; the contract gives NO numeric
# qparams, so scale/zero_point stay the literal NEED_TO_CONFIRM string (rule 5).
# Analysis/config artifact only — contains NO training logic.

QUANT = {
    # AM-4 and AM-4a (docs/PREREGISTRATION_AMENDMENTS.md), implemented by lanes L-AM4 + L-AM1q in
    # src/quant/qat.py: 15 fixed epochs and no early stopping; BN statistics frozen at the top of epoch 11 and
    # every observer, weight and activation, at the top of epoch 13; one checkpoint per epoch; the epoch selected
    # on its converted (QNNPACK) VAL all-class mIoU on CPU (scripts/select_qat_epoch.py). src/quant/qat.py holds
    # the AM-4a values as constants and refuses to run when a value below differs.
    # INT8 Quantization-Aware Training (E5 from E1; E6 from E3 head-removed, identical config to E5)
    "qat": {
        "backend": "QNNPACK",
        "toolchain": "eager-mode torch.ao.quantization",
        "optimizer": "SGD",
        "momentum": 0.9,
        "learning_rate": 3e-4,
        "lr_schedule": "cosine",
        "lr_t_max": "15 x steps_per_epoch optimizer steps (5,025 at 335 per epoch), eta_min 0",
        "epochs": 15,                                # AM-4: a fixed budget; no early stopping
        "activation_quant_start": "step 0",
        "observer": "moving_average",
        "bn_freeze_epoch": 10,                       # AM-4a item 1: BN statistics frozen at the top of epoch 11
        "obs_freeze_epoch": 12,                      # AM-4a item 1: every observer disabled at the top of epoch 13
        "gradient_clipping": "global_norm",
        "weight_ema": False,
        "checkpoint_selection": "converted QNNPACK VAL all-class mIoU of each epoch, on CPU (AM-4a item 2)",
        "distillation_during_qat": False,            # E5/E6 supervised-only
        "weight_quant": "per-channel symmetric INT8 (all conv)",
        "activation_quant": "per-tensor asymmetric UINT8 (full 8-bit range)",
        "fusion": "Conv-BN-ReLU via fuse_modules before observer insertion",
        "standalone_ops": ("Hard-Swish", "Hardsigmoid"),
        "e5_from": "E1",
        "e6_from": "E3 (CWD head removed)",
        # AM-3: triggered by the E3->E6 VAL drop (E6 converted INT8) > 1.0 pp at seed 42.
        "e6kd_reduced_weights": "0.5x the E3 lambda_logit, alpha_cwd and beta_cwd; T unchanged (AM-3)",
    },

    # ---------------------------------------------------------------- E5/E6 real-run controls
    # LOCKED. These are THESIS IMPLEMENTATION CHOICES, explicitly authorized as such — they are NOT
    # values specified by Krishnamoorthi, Jacob, or any other source, and must never be cited as
    # though they were. Every entry applies IDENTICALLY to E5 and E6 (`shared_by`), so the two stages
    # differ only in source checkpoint and distillation history rather than in QAT tuning.
    #
    # AM-4a item 1 pins every value here; no launch flag sets one, and src/quant/qat.py refuses to run
    # when a value differs from its constants.
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
        "drop_last_train": True,              # E1's TRAIN loader: 5,367 images -> 335 steps per epoch

        # Preserves the registered student regularization scale rather than introducing a new one.
        "weight_decay": 1e-4,

        # AM-4: a fixed budget of 15 epochs; every epoch is checkpointed and the epoch is selected after
        # training on its converted VAL score (AM-4a item 2).
        "epochs": 15,

        # AM-4a item 1: whole-epoch boundaries, applied at the top of the epoch loop.
        "bn_freeze_epoch": 10,
        "obs_freeze_epoch": 12,
        "freeze_rule": ("epoch boundaries: BN statistics at the top of epoch 11, every observer at the top of "
                        "epoch 13 (src/quant/prepare.py qat_freeze_steps: 3,350 and 4,020 completed steps)"),
        "fake_quant_start": "step 0",

        # AM-4a item 4 (L-AM1q): set_seed(--seed) before CUDA initialises; both loaders take seed=--seed.
        "seeding": "set_seed(--seed) before CUDA initialises; TRAIN and VAL loaders built with seed=--seed",

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
        "source_checkpoint": "the E1 seed-42 best.json checkpoint (the E5 seed-42 parent)",
        "seed": 42,
        "splits_used": ("train", "val"),
        "test_used_for_selection": False,
        "pilot_budget_epochs": 15,               # AM-4a item 3: two full 15-epoch E5 seed-42 runs
        "official_epochs": 15,
        "selection_rule": (
            "1) reject a candidate whose run state became non-finite (AM-21 item 3); with both rejected "
            "there is no winner and a new amendment decides the value; 2) otherwise pick the higher "
            "converted QNNPACK VAL all-class mIoU of each run's selected epoch (AM-4a items 2-3); 3) on a "
            "tie within 0.1 pp, exact (|Fraction(a) - Fraction(b)| <= 1/1000), prefer 5.0 as the LESS "
            "INTRUSIVE clipping threshold; the loser is retained and reported."),
        "tie_band": "1/1000",
        "rules_file": "configs/qat_selection_rules.json",
        "inherited_by": ("E5 seed 43", "E5 seed 44", "E6 (all seeds)", "E6-KD"),
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
        "selection": "fixed registered qconfig; no validation-based calibration choice (AM-10); reported on test",
        "calibration_batch_size": 1,                 # AM-10: one image per mini-batch; src/quant/prepare.py calibrate() refuses any other (L-AM10)
        # AM-16 item 5: three further 128-image TRAIN subsets, drawn by the AM-10 procedure, for the
        # descriptive calibration-sensitivity runs of E4/E7 at seed 42. The official models keep the
        # seed-42 list (AM-10). All four lists: configs/calibration/ (scripts/build_calibration_lists.py).
        "calibration_sensitivity_seeds": (43, 44, 45),
        "calibration_list_dir": "configs/calibration",
        # Lane 8 (L-AM10): scripts/run_ptq.py writes the QNNPACK TorchScript artifact of record, the
        # converted state_dict alongside it, and the x86 latency copy (src/quant/ptq.py).
        "artifact_format": "torchscript",
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
