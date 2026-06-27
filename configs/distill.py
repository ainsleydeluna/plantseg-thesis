# Distillation config — Blocker B3 (Logit KD + Channel-Wise KD)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B3 - Distillation"
# Every value traced to ch3.pdf sections C "E2"/"E3". λ_logit is selection-determined and
# kept as the literal NEED_TO_CONFIRM string per source hierarchy (rule 5).
# Analysis/config artifact only — contains NO training logic.

DISTILL = {
    # E2: response-level Logit KD
    "logit_kd": {
        "loss": "CE + KL on temperature-softened outputs",
        "T_logit": 4,
        "lambda_logit": "NEED_TO_CONFIRM",          # selected via validation sweep; reported in Ch4
        "lambda_logit_sweep_grid": (0.25, 0.5, 1, 2, 4),
        "sweep_seed": 42,
        "kl_averaging": "valid (non-255) pixels only",
        "reused_unchanged_in_e3": True,
    },

    # E3: + Channel-Wise KD (Shu 2021)
    "cwd": {
        "T_cwd": 4,
        "alpha_cwd_feature_map": 50,                # stride-16 C5 map
        "beta_cwd_logit_map": 3,
        "normalization": "T^2 / C",
        "C": 320,                                   # MSCAN-B stride-16 Stage-3 channel count
        "projection_head": {
            "type": "1x1 conv, training-only",
            "maps": "student 160-ch C5 -> teacher 320-ch",
            "removed_before": ("E6", "E7"),
            "removal": "state_dict edit before quant observer insertion",
        },
        "ignore_handling": "validity mask downsampled to stride-16; softmax + KL over valid locations only",
    },

    # Combined E3 objective
    "e3_total_loss": "L_CE + L_Dice + lambda_logit*L_LogitKD + 50*L_CWD_feat + 3*L_CWD_logit",
}
