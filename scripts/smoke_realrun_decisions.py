#!/usr/bin/env python3
"""Verification of the preregistered real-run decision surface. No dataset, no training, no GPU.

Proves two things that are easy to get wrong:
  * every value that IS now fixed is machine-readable and identical across the stages that must
    share it (E5/E6 common controls; one clipping rule for E2/E3/E5/E6);
  * every value that is NOT fixed stays visibly unresolved and keeps blocking an official launch.

Nothing here relaxes a gate: the launchers still require each control explicitly, and the clipping
rule is still unselected.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs.distill import DISTILL  # noqa: E402
from configs.e1_student import E1_STUDENT  # noqa: E402
from configs.quant import QUANT  # noqa: E402
from src.quant.prepare import qat_grad_clip_gate_error  # noqa: E402
from src.training.train_distill import grad_clip_gate_error  # noqa: E402

results: list[tuple[str, bool, str]] = []
QAT_RUN = QUANT["qat_real_run_recommendation"]
PILOT = E1_STUDENT["grad_clip_pilot"]


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------- 1. recommended values
def test_locked_values() -> None:
    check("recommendation_status_is_explicit",
          QAT_RUN["status"] == "RECOMMENDED_PENDING_AUTHORIZATION"
          and QAT_RUN["enforced_at_launch"] is False,
          "analysed, not silently applied")
    check("conflict_with_recorded_position_declared",
          tuple(QAT_RUN["overturns_recorded_position"]) == ("batch_size", "weight_decay"),
          "B4 has no batch-size/weight-decay row; the runner accepts any finite wd >= 0")
    for key, expected in (("batch_size", 16), ("weight_decay", 1e-4), ("max_epochs", 15),
                          ("early_stop_patience", 3), ("bn_freeze_pct", 0.65),
                          ("observer_freeze_pct", 0.70)):
        check(f"recommended_{key}", QAT_RUN.get(key) == expected, str(QAT_RUN.get(key)))

    # the two inherited controls must equal the registered FP32 recipe, not a second recipe
    check("qat_batch_inherits_student_recipe",
          QAT_RUN["batch_size"] == E1_STUDENT["batch_size"], str(E1_STUDENT["batch_size"]))
    check("qat_weight_decay_inherits_student_recipe",
          QAT_RUN["weight_decay"] == E1_STUDENT["weight_decay"], str(E1_STUDENT["weight_decay"]))

    # BN/observer freeze must sit inside the contract's registered window, in the required order
    from src.quant.prepare import BN_FREEZE_PCT_RANGE
    lo, hi = BN_FREEZE_PCT_RANGE
    check("bn_freeze_inside_locked_range", lo <= QAT_RUN["bn_freeze_pct"] <= hi, f"[{lo}, {hi}]")
    check("observer_freeze_after_bn_freeze",
          QAT_RUN["observer_freeze_pct"] > QAT_RUN["bn_freeze_pct"],
          f"{QAT_RUN['bn_freeze_pct']} -> {QAT_RUN['observer_freeze_pct']} "
          "('shortly after', 5% of budget)")
    check("epoch_budget_is_a_cap_not_a_target",
          QAT_RUN["max_epochs"] == QUANT["qat"]["epochs_approx"]
          and QUANT["qat"]["early_stop"] == "val_miou"
          and QUANT["qat"]["checkpoint_selection"] == "best_val_miou",
          "early stop + best-val-mIoU selection make 15 a maximum")


# ---------------------------------------------------------------- 2. E5/E6 controlled comparison
def test_e5_e6_controlled() -> None:
    check("controls_declared_shared_by_e5_e6",
          tuple(QAT_RUN["shared_by"]) == ("E5", "E6"), str(QAT_RUN["shared_by"]))
    check("shared_control_set_is_complete",
          {"batch_size", "weight_decay", "max_epochs", "early_stop_patience", "bn_freeze_pct",
           "observer_freeze_pct", "grad_clip"} <= set(QAT_RUN),
          "every non-stage-specific control is covered")

    # The launcher still demands each control explicitly for BOTH stages — nothing is defaulted, so
    # neither E5 nor E6 can start from an implicit value.
    from src.quant.runner import unresolved_qat_values

    class Empty:
        grad_clip_norm = batch_size = weight_decay = epochs = None
        early_stop_patience = bn_freeze_pct = observer_freeze_pct = None

    missing = unresolved_qat_values(Empty())
    check("launcher_still_requires_every_control", len(missing) >= 7,
          f"{len(missing)} values refused when unspecified")
    for flag in ("--grad-clip-norm", "--batch-size", "--weight-decay", "--epochs",
                 "--early-stop-patience", "--bn-freeze-pct", "--observer-freeze-pct"):
        check(f"gate_lists_{flag.strip('-')}", any(flag in m for m in missing), flag)


# ---------------------------------------------------------------- 3. clipping stays pilot-selected
def test_clipping_unresolved() -> None:
    check("clipping_visibly_unresolved",
          PILOT["status"] == "PILOT_REQUIRED" and QAT_RUN["grad_clip"] == "PILOT_REQUIRED",
          "no value is frozen yet")
    check("e1_remains_unclipped",
          E1_STUDENT["grad_clip_max_norm"] is None
          and E1_STUDENT["gradient_clipping"] == "global_norm",
          "D2/D-A deviation untouched; 'global_norm' is the method family")
    check("unclipped_is_a_candidate", "none" in PILOT["candidates"],
          f"candidates={PILOT['candidates']} — 'none' can match E1 and remove the confound")
    check("one_rule_applies_to_every_training_stage",
          tuple(PILOT["applies_to"]) == ("E2", "E3", "E5", "E6"), str(PILOT["applies_to"]))
    check("pilot_never_touches_test",
          PILOT["test_used_for_selection"] is False
          and tuple(PILOT["splits_used"]) == ("train", "val")
          and "test evaluation" in PILOT["excluded_from"])
    check("pilot_labelled_not_official",
          PILOT["label"] == "HYPERPARAMETER SELECTION / PILOT"
          and "hypothesis testing" in PILOT["excluded_from"]
          and "robustness" in PILOT["excluded_from"])
    check("pilot_is_small",
          len(PILOT["candidates"]) <= 3 and PILOT["seed"] == 42
          and PILOT["selection_metric"] == "dataset-level validation mIoU",
          f"{len(PILOT['candidates'])} candidates, seed 42, validation mIoU")
    check("value_frozen_before_official_stages",
          tuple(PILOT["freeze_before"]) == ("E2", "E3", "E5", "E6"))


# ---------------------------------------------------------------- 4. gates still refuse
def test_gates_still_refuse() -> None:
    # absent value: both launchers still refuse, exactly as before
    check("e2e3_refuses_absent_clip", grad_clip_gate_error(None) is not None)
    check("e5e6_refuses_absent_clip", qat_grad_clip_gate_error(None) is not None)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        check(f"e2e3_refuses_{bad}", grad_clip_gate_error(bad) is not None, str(bad))
        check(f"e5e6_refuses_{bad}", qat_grad_clip_gate_error(bad) is not None, str(bad))

    # Committed gate behaviour is UNCHANGED: an explicit positive finite value is still accepted, so
    # the pilot itself can run its candidates and no existing coverage was weakened.
    numeric = [c for c in PILOT["candidates"] if not isinstance(c, str)]
    for candidate in numeric:
        check(f"e2e3_accepts_candidate_{candidate}", grad_clip_gate_error(candidate) is None,
              str(candidate))
        check(f"e5e6_accepts_candidate_{candidate}", qat_grad_clip_gate_error(candidate) is None,
              str(candidate))
    check("both_gates_agree_on_the_same_input",
          (grad_clip_gate_error(numeric[0]) is None)
          == (qat_grad_clip_gate_error(numeric[0]) is None),
          "E2/E3 and E5/E6 apply the same accept/reject rule")
    check("clipping_value_not_written_into_any_config",
          E1_STUDENT["grad_clip_max_norm"] is None
          and QAT_RUN["grad_clip"] == "PILOT_REQUIRED",
          "no number was invented to satisfy a launcher")


# ---------------------------------------------------------------- 5. untouched decisions
def test_untouched() -> None:
    check("lambda_logit_still_sweep_selected",
          DISTILL["logit_kd"]["lambda_logit"] == "NEED_TO_CONFIRM"
          and tuple(DISTILL["logit_kd"]["lambda_logit_sweep_grid"]) == (0.25, 0.5, 1, 2, 4)
          and DISTILL["logit_kd"]["sweep_seed"] == 42,
          "existing preregistered sweep untouched")
    check("governed_fp32_recipe_untouched",
          E1_STUDENT["learning_rate"] == 1e-2 and E1_STUDENT["iterations"] == 80000
          and E1_STUDENT["batch_size"] == 16 and E1_STUDENT["val_interval"] == 4000
          and E1_STUDENT["lr_power"] == 0.9)
    check("governed_qat_recipe_untouched",
          QUANT["qat"]["learning_rate"] == 3e-4 and QUANT["qat"]["momentum"] == 0.9
          and QUANT["qat"]["lr_schedule"] == "cosine"
          and QUANT["qat"]["distillation_during_qat"] is False
          and QUANT["qat"]["weight_ema"] is False)
    check("cwd_values_untouched",
          DISTILL["cwd"]["T_cwd"] == 4 and DISTILL["cwd"]["alpha_cwd_feature_map"] == 50
          and DISTILL["cwd"]["beta_cwd_logit_map"] == 3)
    check("no_test_loader_in_selection_surface",
          all("test" not in str(v).lower() or k in ("splits_used", "excluded_from",
                                                    "test_used_for_selection")
              for k, v in PILOT.items()),
          "the pilot spec references TEST only to exclude it")


def main() -> int:
    print("=" * 78)
    print("REAL-RUN DECISION SURFACE SMOKE — configs + launch gates only; no data, no training")
    print("=" * 78)
    for fn in (test_locked_values, test_e5_e6_controlled, test_clipping_unresolved,
               test_gates_still_refuse, test_untouched):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:46}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    print("\nNOTE: the shared gradient-clipping rule is still UNSELECTED. Official E2/E3/E5/E6 runs "
          "remain blocked until the preregistered train/validation pilot freezes it.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
