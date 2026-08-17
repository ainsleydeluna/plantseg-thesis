#!/usr/bin/env python3
"""Verification of the preregistered real-run decision surface. No dataset, no training, no GPU.

Proves the surface is internally coherent and honestly labelled:
  * the QAT controls that ARE locked are machine-readable and identical for E5/E6;
  * the two gradient-clipping decisions are SEPARATE, both unresolved, both clipping-only;
  * early stopping cannot fire before the quantization schedule has executed;
  * official launches still refuse every unresolved value, and no selection path can see TEST.

Nothing here relaxes a gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs.distill import DISTILL  # noqa: E402
from configs.e1_student import E1_STUDENT  # noqa: E402
from configs.quant import QUANT  # noqa: E402
from src.quant.prepare import BN_FREEZE_PCT_RANGE, qat_grad_clip_gate_error  # noqa: E402
from src.quant.runner import EarlyStopper  # noqa: E402
from src.training.train_distill import grad_clip_gate_error  # noqa: E402

results: list[tuple[str, bool, str]] = []
QAT_RUN = QUANT["qat_real_run"]
DISTILL_PILOT = DISTILL["distillation_grad_clip_pilot"]
QAT_PILOT = QUANT["qat_grad_clip_pilot"]


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------- 1. locked QAT controls
def test_locked_qat_controls() -> None:
    check("qat_controls_locked", QAT_RUN["status"] == "LOCKED", QAT_RUN["status"])
    for key, expected in (("batch_size_physical", 16), ("weight_decay", 1e-4), ("max_epochs", 15),
                          ("early_stop_patience", 3), ("bn_freeze_pct", 0.65),
                          ("observer_freeze_pct", 0.70)):
        check(f"locked_{key}", QAT_RUN.get(key) == expected, str(QAT_RUN.get(key)))
    check("controls_identical_for_e5_e6", tuple(QAT_RUN["shared_by"]) == ("E5", "E6"),
          str(QAT_RUN["shared_by"]))
    check("basis_labelled_implementation_choice",
          "implementation choice" in QAT_RUN["basis"]
          and "not specified by the source papers" in QAT_RUN["basis"],
          "never attributed to Krishnamoorthi/Jacob")

    # batch size is PHYSICAL and accumulation is not silently enabled
    check("batch_size_is_physical",
          "batch_size_physical" in QAT_RUN and "physical batch" in QAT_RUN["batch_semantics"])
    check("no_gradient_accumulation",
          QAT_RUN["gradient_accumulation"] is None
          and not any("accum" in str(k).lower() and QAT_RUN[k] for k in QAT_RUN),
          "observers and BatchNorm are batch-sensitive, so accumulation is not equivalent")

    lo, hi = BN_FREEZE_PCT_RANGE
    check("bn_freeze_inside_registered_window", lo <= QAT_RUN["bn_freeze_pct"] <= hi, f"[{lo}, {hi}]")
    check("bn_freeze_precedes_observer_freeze",
          QAT_RUN["bn_freeze_pct"] < QAT_RUN["observer_freeze_pct"],
          f"{QAT_RUN['bn_freeze_pct']} < {QAT_RUN['observer_freeze_pct']}")
    check("freeze_points_are_step_fractions",
          "round(total_iters * pct)" in QAT_RUN["freeze_rounding"]
          and "observer >= bn" in QAT_RUN["freeze_rounding"],
          QAT_RUN["freeze_rounding"])
    check("fake_quant_from_first_step", QAT_RUN["fake_quant_start"] == "step 0")


# ---------------------------------------------------------------- 2. early-stop safety
def test_early_stop_after_observer_freeze() -> None:
    check("early_stop_eligibility_registered",
          QAT_RUN["early_stop_eligible_after"] == "observer_freeze")

    # patience must NOT accrue while ineligible, even across a long plateau
    s = EarlyStopper(QAT_RUN["early_stop_patience"])
    s.update(0.50, 1, stop_eligible=True)                       # establish a best
    for step in range(2, 12):
        s.update(0.10, step, stop_eligible=False)               # plateau before observer freeze
    check("patience_does_not_accrue_before_observer_freeze",
          s.num_bad == 0 and s.triggered is False,
          f"{10} non-improving validations ignored while ineligible")

    # once eligible, exactly `patience` non-improvements trigger it
    for step in range(12, 12 + QAT_RUN["early_stop_patience"]):
        s.update(0.10, step, stop_eligible=True)
    check("patience_accrues_once_eligible", s.triggered is True,
          f"triggered after {QAT_RUN['early_stop_patience']} non-improvements")

    # best-checkpoint tracking still works while ineligible
    s2 = EarlyStopper(3)
    improved = s2.update(0.80, 5, stop_eligible=False)
    check("best_checkpoint_tracked_while_ineligible",
          improved is True and s2.best == 0.80 and s2.triggered is False,
          "selection is unaffected by eligibility")

    # the runner wires eligibility to the observer-freeze step
    src = (REPO / "src/quant/runner.py").read_text(encoding="utf-8")
    check("runner_gates_early_stop_on_observer_freeze",
          "stop_eligible=it >= obs_freeze_at" in src, "wired in the QAT loop")


# ---------------------------------------------------------------- 3. two separate clipping decisions
def test_two_clipping_decisions() -> None:
    check("distillation_decision_named",
          DISTILL_PILOT["name"] == "DISTILLATION_GRAD_CLIP_NORM"
          and tuple(DISTILL_PILOT["applies_to"]) == ("E2", "E3"))
    check("qat_decision_named",
          QAT_PILOT["name"] == "QAT_GRAD_CLIP_NORM"
          and tuple(QAT_PILOT["applies_to"]) == ("E5", "E6"))
    check("decisions_are_independent",
          QAT_PILOT["inherits_distillation_value"] is False
          and E1_STUDENT["grad_clip_scope"]["shared_numeric_threshold_required"] is False,
          "distillation and QAT may select different numeric thresholds")
    check("both_still_unresolved",
          DISTILL_PILOT["status"] == "PILOT_REQUIRED" and QAT_PILOT["status"] == "PILOT_REQUIRED"
          and DISTILL_PILOT["selected_value"] is None and QAT_PILOT["selected_value"] is None,
          "not described as locked")

    for label, pilot in (("distillation", DISTILL_PILOT), ("qat", QAT_PILOT)):
        cands = pilot["candidates"]
        check(f"{label}_no_none_candidate",
              all(not isinstance(c, str) for c in cands) and "none" not in cands,
              f"candidates={cands}")
        check(f"{label}_candidates_positive_finite",
              all(isinstance(c, float) and c > 0.0 and c == c and c != float("inf") for c in cands),
              str(cands))
        check(f"{label}_candidates_are_1_and_5", tuple(cands) == (1.0, 5.0), str(cands))
        check(f"{label}_tie_prefers_5", "prefer" in pilot["selection_rule"]
              and "5.0" in pilot["selection_rule"]
              and "less intrusive" in pilot["selection_rule"].lower())
        check(f"{label}_stability_checked_first",
              pilot["selection_rule"].strip().startswith("1) reject"),
              "non-finite/unstable candidate rejected before comparing mIoU")
        check(f"{label}_no_new_significance_test", pilot["no_new_significance_test"] is True)
        check(f"{label}_pilot_labelled", pilot["label"] == "HYPERPARAMETER SELECTION / PILOT")
        check(f"{label}_seed_42", pilot["seed"] == 42)


# ---------------------------------------------------------------- 4. test-set firewall
def test_test_firewall() -> None:
    for label, pilot in (("distillation", DISTILL_PILOT), ("qat", QAT_PILOT)):
        check(f"{label}_pilot_train_val_only",
              tuple(pilot["splits_used"]) == ("train", "val")
              and pilot["test_used_for_selection"] is False)
        check(f"{label}_pilot_excluded_from_official_families",
              {"test evaluation", "robustness", "hypothesis testing", "statistics family"}
              <= set(pilot["excluded_from"]))
    check("lambda_sweep_validation_only",
          DISTILL["logit_kd"]["lambda_logit"] == "NEED_TO_CONFIRM"
          and DISTILL["logit_kd"]["sweep_seed"] == 42)
    # no pilot spec smuggles a test split in
    for label, pilot in (("distillation", DISTILL_PILOT), ("qat", QAT_PILOT)):
        leaks = [k for k, v in pilot.items()
                 if k not in ("splits_used", "excluded_from", "test_used_for_selection")
                 and "test" in str(v).lower()]
        check(f"{label}_pilot_mentions_test_only_to_exclude_it", not leaks, str(leaks))


# ---------------------------------------------------------------- 5. pilot budgets are separate
def test_pilot_budgets() -> None:
    check("distillation_pilot_budget_shortened",
          DISTILL_PILOT["pilot_budget_iters"] < DISTILL_PILOT["official_budget_iters"]
          and DISTILL_PILOT["official_budget_iters"] == 80000,
          f"{DISTILL_PILOT['pilot_budget_iters']} vs {DISTILL_PILOT['official_budget_iters']} iters")
    check("distillation_pilot_has_validation_cadence",
          DISTILL_PILOT["pilot_val_interval"] > 0
          and DISTILL_PILOT["pilot_budget_iters"] % DISTILL_PILOT["pilot_val_interval"] == 0,
          f"{DISTILL_PILOT['pilot_budget_iters'] // DISTILL_PILOT['pilot_val_interval']} checks")
    check("qat_pilot_budget_shortened",
          QAT_PILOT["pilot_budget_epochs"] < QAT_PILOT["official_max_epochs"]
          and QAT_PILOT["official_max_epochs"] == QAT_RUN["max_epochs"],
          f"{QAT_PILOT['pilot_budget_epochs']} vs {QAT_PILOT['official_max_epochs']} epochs")
    check("lambda_pilot_uses_grid_centre",
          DISTILL_PILOT["lambda_logit_during_pilot"] == 1.0
          and 1 in DISTILL["logit_kd"]["lambda_logit_sweep_grid"],
          "avoids a 2 x 5 Cartesian search while staying inside the registered grid")
    check("decision_order_clip_then_lambda",
          DISTILL_PILOT["decision_order"][0].startswith("select DISTILLATION_GRAD_CLIP_NORM")
          and "lambda_logit sweep" in DISTILL_PILOT["decision_order"][2]
          and DISTILL_PILOT["decision_order"][-1] == "official E2/E3 runs")


# ---------------------------------------------------------------- 6. gates still refuse
def test_gates_still_refuse() -> None:
    check("e2e3_refuses_absent_clip", grad_clip_gate_error(None) is not None)
    check("e5e6_refuses_absent_clip", qat_grad_clip_gate_error(None) is not None)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        check(f"e2e3_refuses_{bad}", grad_clip_gate_error(bad) is not None, str(bad))
        check(f"e5e6_refuses_{bad}", qat_grad_clip_gate_error(bad) is not None, str(bad))

    # the low-level primitive stays reusable: it accepts any positive finite norm, so PILOT runs can
    # use their candidates. Candidate membership belongs to the decision layer above, not here.
    for candidate in DISTILL_PILOT["candidates"]:
        check(f"primitive_accepts_candidate_{candidate}",
              grad_clip_gate_error(candidate) is None
              and qat_grad_clip_gate_error(candidate) is None, str(candidate))

    # every QAT control is still demanded explicitly
    from src.quant.runner import unresolved_qat_values

    class Empty:
        grad_clip_norm = batch_size = weight_decay = epochs = None
        early_stop_patience = bn_freeze_pct = observer_freeze_pct = None

    missing = unresolved_qat_values(Empty())
    for flag in ("--grad-clip-norm", "--batch-size", "--weight-decay", "--epochs",
                 "--early-stop-patience", "--bn-freeze-pct", "--observer-freeze-pct"):
        check(f"gate_still_requires_{flag.strip('-')}", any(flag in m for m in missing), flag)
    check("no_clip_value_written_into_configs",
          E1_STUDENT["grad_clip_max_norm"] is None
          and QAT_RUN["grad_clip"] == "PILOT_REQUIRED"
          and DISTILL_PILOT["selected_value"] is None and QAT_PILOT["selected_value"] is None,
          "no number invented to satisfy a launcher")


# ---------------------------------------------------------------- 7. untouched decisions
def test_untouched() -> None:
    check("e1_remains_unclipped",
          E1_STUDENT["grad_clip_max_norm"] is None
          and E1_STUDENT["grad_clip_scope"]["e1"].startswith("unclipped"),
          "D2/D-A governs execution; not reopened here")
    check("manuscript_reconciliation_flagged",
          E1_STUDENT["grad_clip_scope"]["manuscript_reconciliation_pending"] is True,
          "'identical recipe' wording vs E1-unclipped/E2-E3-clipped carried forward")
    check("lambda_grid_unchanged",
          tuple(DISTILL["logit_kd"]["lambda_logit_sweep_grid"]) == (0.25, 0.5, 1, 2, 4))
    check("governed_fp32_recipe_untouched",
          E1_STUDENT["learning_rate"] == 1e-2 and E1_STUDENT["iterations"] == 80000
          and E1_STUDENT["batch_size"] == 16 and E1_STUDENT["val_interval"] == 4000)
    check("governed_qat_recipe_untouched",
          QUANT["qat"]["learning_rate"] == 3e-4 and QUANT["qat"]["lr_schedule"] == "cosine"
          and QUANT["qat"]["distillation_during_qat"] is False
          and QUANT["qat"]["weight_ema"] is False)
    check("cwd_values_untouched",
          DISTILL["cwd"]["T_cwd"] == 4 and DISTILL["cwd"]["alpha_cwd_feature_map"] == 50
          and DISTILL["cwd"]["beta_cwd_logit_map"] == 3)


def main() -> int:
    print("=" * 78)
    print("REAL-RUN DECISION SURFACE SMOKE — configs + launch gates only; no data, no training")
    print("=" * 78)
    for fn in (test_locked_qat_controls, test_early_stop_after_observer_freeze,
               test_two_clipping_decisions, test_test_firewall, test_pilot_budgets,
               test_gates_still_refuse, test_untouched):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:50}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    print("\nNOTE: DISTILLATION_GRAD_CLIP_NORM and QAT_GRAD_CLIP_NORM are both UNSELECTED, and "
          "lambda_logit is unselected. Official E2/E3 and E5/E6 runs stay blocked until their "
          "validation-only pilots freeze those values.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
