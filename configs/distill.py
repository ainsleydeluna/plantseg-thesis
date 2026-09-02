# Distillation config — Blocker B3 (Logit KD + Channel-Wise KD)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B3 - Distillation"
# Every value traced to ch3.pdf sections C "E2"/"E3". λ_logit is selection-determined and
# kept as the literal NEED_TO_CONFIRM string per source hierarchy (rule 5).
# Analysis/config artifact only — contains NO training logic.

# ---------------------------------------------------------------- lambda_logit semantics tag (B32)
# lambda_logit weights the Logit-KD term, so its numeric value is only meaningful RELATIVE TO THE
# SPATIAL GRID that term is computed on. B32/F8 moved that grid from an upsampled 512x512 to the
# head's native OS8 64x64, and the term's magnitude changed by ~1.95x (MEASURED over 20 real samples
# against a SYNTHETIC teacher, so an UPPER BOUND — a real smooth SegNeXt-B should shift it less).
# The preregistered grid {0.25, 0.5, 1, 2, 4} is geometric with ratio 2, so ~1.95x is about ONE grid
# step: the grid still brackets a sensible optimum, but the optimum sits roughly one step lower and
# a lambda selected under the OLD semantics is NOT transferable to the new one.
#
# The tag is deliberately self-describing — "the Logit-KD KL was computed at output stride 8, on a
# 64x64 grid, from a 512x512 input" — so a reader six months from now needs no decoder ring.
LOGIT_KD_SEMANTICS = "logitkd@os8-64x64-of-512"

# Superseded semantics. A lambda measured under any of these must NEVER be consumed under the
# current tag without an explicit, recorded override.
LOGIT_KD_SEMANTICS_SUPERSEDED = ("logitkd@full-512x512-upsampled",)   # pre-B32

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

    # ---------------------------------------------------- E2/E3 gradient clipping (PREREGISTERED)
    # Chapter 3 requires global-norm clipping THROUGHOUT distillation training but names no threshold,
    # and no primary source supplies one (the KD/CWD papers report optimizer, schedule, temperature and
    # loss weights, not a clipping norm). "No clipping" is NOT a candidate: the methodology mandates
    # clipping for these stages, and the committed E2/E3 launcher requires a positive finite max_norm.
    # E1's separately resolved unclipped status (D2/D-A) is not permission to leave E2/E3 unclipped.
    #
    # This decision is INDEPENDENT of the QAT one (QUANT["qat_grad_clip_pilot"]): distillation and QAT
    # are different optimization regimes, and nothing establishes that one numeric norm should serve
    # both. E2 and E3 must nevertheless use the SAME selected value, or E2-vs-E3 is confounded.
    "distillation_grad_clip_pilot": {
        "name": "DISTILLATION_GRAD_CLIP_NORM",
        "status": "PILOT_REQUIRED",              # unresolved; official E2/E3 launches must refuse
        "selected_value": None,
        "candidates": (1.0, 5.0),                # both positive finite; no 'none' option
        "applies_to": ("E2", "E3"),
        "run_on": "E2",                          # cheapest stage exercising distillation gradients
        "seed": 42,
        "splits_used": ("train", "val"),
        "test_used_for_selection": False,
        # λ is HELD FIXED at the centre of the preregistered λ grid during this pilot, which avoids a
        # 2 x 5 Cartesian clipping-by-λ search while keeping the pilot inside the registered grid.
        "lambda_logit_during_pilot": 1.0,
        # Shortened budget, identical for both candidates and clearly distinct from the official
        # 80,000-iteration run, so a pilot artifact can never be mistaken for an official E2 result.
        "pilot_budget_iters": 8000,              # 10% of the official 80,000
        "pilot_val_interval": 1000,              # 8 validation points per candidate
        "official_budget_iters": 80000,          # for contrast only; the pilot never runs this long
        "selection_rule": (
            "1) reject a candidate whose training becomes non-finite or numerically unstable; "
            "2) otherwise pick the higher dataset-level validation mIoU under the identical pilot "
            "budget; 3) on a tie within the repository's existing validation tie/noise rule prefer "
            "5.0 as the LESS INTRUSIVE clipping threshold."),
        "no_new_significance_test": True,        # reuses the existing tie/noise rule; invents no test
        "decision_order": ("select DISTILLATION_GRAD_CLIP_NORM", "freeze it",
                           "run the existing lambda_logit sweep", "freeze lambda_logit",
                           "official E2/E3 runs"),
        "label": "HYPERPARAMETER SELECTION / PILOT",
        "freeze_before": ("E2", "E3"),
        "excluded_from": ("test evaluation", "robustness", "hypothesis testing", "statistics family"),
    },
}
