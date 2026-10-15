#!/usr/bin/env python3
"""Verification of the preregistered real-run decision surface. No dataset, no training, no GPU.

Proves the surface is internally coherent and honestly labelled:
  * the QAT controls that ARE locked are machine-readable and identical for E5/E6;
  * the two gradient-clipping decisions are SEPARATE and clipping-only; the distillation one is
    withdrawn by AM-7 (E1-E3 unclipped: the E2/E3 gate refuses any value, lane L-AM7 carried out by
    L-KD-HARDEN) and kept as a readable record, the QAT one is still unresolved;
  * the QAT budget is 15 fixed epochs with epoch-boundary freezes and no early stopping (AM-4/AM-4a);
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
from src.quant.prepare import qat_freeze_steps, qat_grad_clip_gate_error  # noqa: E402
from src.quant import qat as QAT_TRAINER  # noqa: E402
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
    for key, expected in (("batch_size_physical", 16), ("weight_decay", 1e-4), ("epochs", 15),
                          ("bn_freeze_epoch", 10), ("obs_freeze_epoch", 12)):
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

    check("bn_freeze_precedes_observer_freeze",
          QAT_RUN["bn_freeze_epoch"] < QAT_RUN["obs_freeze_epoch"],
          f"{QAT_RUN['bn_freeze_epoch']} < {QAT_RUN['obs_freeze_epoch']}")
    check("freeze_points_are_epoch_boundaries",
          qat_freeze_steps(335) == (3350, 4020) and "epoch boundaries" in QAT_RUN["freeze_rule"],
          QAT_RUN["freeze_rule"])
    check("trainer_constants_equal_config", QAT_TRAINER.config_pins_error() is None,
          "src/quant/qat.py refuses a configs/quant.py that differs from AM-4a")
    check("fake_quant_from_first_step", QAT_RUN["fake_quant_start"] == "step 0")


# ---------------------------------------------------------------- 2. no early stopping (AM-4)
def test_early_stop_after_observer_freeze() -> None:
    """AM-4 removed early stopping: the section keeps its name and checks that none exists."""
    keys = list(QUANT["qat"]) + list(QAT_RUN)
    check("no_early_stop_controls_registered",
          not any("early_stop" in k or "patience" in k or "max_epochs" in k for k in keys),
          "15 fixed epochs; the epoch is selected after training")
    check("qat_epochs_fixed_at_15",
          QUANT["qat"]["epochs"] == QAT_RUN["epochs"] == QAT_TRAINER.EPOCHS == 15)
    check("selection_on_converted_val",
          "converted" in QUANT["qat"]["checkpoint_selection"]
          and "VAL" in QUANT["qat"]["checkpoint_selection"], QUANT["qat"]["checkpoint_selection"])
    src = (REPO / "src/quant/qat.py").read_text(encoding="utf-8")
    check("trainer_runs_every_epoch",
          "for epoch in range(1, EPOCHS + 1):" in src and "EarlyStopper" not in src and "patience" not in src,
          "no early-stop path in src/quant/qat.py")


# ---------------------------------------------------------------- 3. two separate clipping decisions
def test_two_clipping_decisions() -> None:
    # AM-7 withdrew the distillation pilot: the distillation_record_* checks guard the WITHDRAWN_AM7
    # record's values (kept unchanged as the readable record), not a live requirement.
    check("distillation_record_named",
          DISTILL_PILOT["name"] == "DISTILLATION_GRAD_CLIP_NORM"
          and tuple(DISTILL_PILOT["applies_to"]) == ("E2", "E3"))
    check("qat_decision_named",
          QAT_PILOT["name"] == "QAT_GRAD_CLIP_NORM"
          and tuple(QAT_PILOT["applies_to"]) == ("E5", "E6"))
    check("decisions_are_independent",
          QAT_PILOT["inherits_distillation_value"] is False
          and E1_STUDENT["grad_clip_scope"]["shared_numeric_threshold_required"] is False,
          "distillation and QAT may select different numeric thresholds")
    check("distillation_withdrawn_qat_unresolved",
          DISTILL_PILOT["status"] == "WITHDRAWN_AM7"
          and QAT_PILOT["status"] == "PILOT_REQUIRED"
          and DISTILL_PILOT["selected_value"] is None and QAT_PILOT["selected_value"] is None,
          "AM-7 withdraws the distillation pilot; neither is described as locked")

    for label, pilot in (("distillation_record", DISTILL_PILOT), ("qat", QAT_PILOT)):
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
    for label, pilot in (("distillation_record", DISTILL_PILOT), ("qat", QAT_PILOT)):
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
    for label, pilot in (("distillation_record", DISTILL_PILOT), ("qat", QAT_PILOT)):
        leaks = [k for k, v in pilot.items()
                 if k not in ("splits_used", "excluded_from", "test_used_for_selection")
                 and "test" in str(v).lower()]
        check(f"{label}_pilot_mentions_test_only_to_exclude_it", not leaks, str(leaks))


# ---------------------------------------------------------------- 5. pilot budgets are separate
def test_pilot_budgets() -> None:
    check("distillation_record_budget_shortened",
          DISTILL_PILOT["pilot_budget_iters"] < DISTILL_PILOT["official_budget_iters"]
          and DISTILL_PILOT["official_budget_iters"] == 80000,
          f"{DISTILL_PILOT['pilot_budget_iters']} vs {DISTILL_PILOT['official_budget_iters']} iters")
    check("distillation_record_has_validation_cadence",
          DISTILL_PILOT["pilot_val_interval"] > 0
          and DISTILL_PILOT["pilot_budget_iters"] % DISTILL_PILOT["pilot_val_interval"] == 0,
          f"{DISTILL_PILOT['pilot_budget_iters'] // DISTILL_PILOT['pilot_val_interval']} checks")
    check("qat_pilot_budget_full_15_epochs",
          QAT_PILOT["pilot_budget_epochs"] == QAT_PILOT["official_epochs"] == QAT_RUN["epochs"] == 15,
          f"{QAT_PILOT['pilot_budget_epochs']} vs {QAT_PILOT['official_epochs']} epochs (AM-4a item 3)")
    check("distillation_record_lambda_at_grid_centre",
          DISTILL_PILOT["lambda_logit_during_pilot"] == 1.0
          and 1 in DISTILL["logit_kd"]["lambda_logit_sweep_grid"],
          "avoids a 2 x 5 Cartesian search while staying inside the registered grid")
    check("distillation_record_decision_order_unchanged",
          DISTILL_PILOT["decision_order"][0].startswith("select DISTILLATION_GRAD_CLIP_NORM")
          and "lambda_logit sweep" in DISTILL_PILOT["decision_order"][2]
          and DISTILL_PILOT["decision_order"][-1] == "official E2/E3 runs",
          "the withdrawn pilot's order as written; under AM-7 E2/E3 wait on the lambda sweep only")


# ---------------------------------------------------------------- 6. gates still refuse
def test_gates_still_refuse() -> None:
    # AM-7: the E2/E3 real-run gate takes NO clipping value (None is the only admissible input).
    check("e2e3_accepts_absent_clip_am7", grad_clip_gate_error(None) is None)
    check("e5e6_refuses_absent_clip", qat_grad_clip_gate_error(None) is not None)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        check(f"e2e3_refuses_{bad}", grad_clip_gate_error(bad) is not None, str(bad))
        check(f"e5e6_refuses_{bad}", qat_grad_clip_gate_error(bad) is not None, str(bad))

    # the QAT primitive stays reusable: it accepts any positive finite norm, so PILOT runs can use
    # their candidates. Candidate membership belongs to the decision layer above, not here. The E2/E3
    # gate refuses the withdrawn pilot's candidates like any other value (AM-7).
    for candidate in DISTILL_PILOT["candidates"]:
        check(f"e2e3_refuses_withdrawn_candidate_{candidate}",
              grad_clip_gate_error(candidate) is not None, str(candidate))
        check(f"qat_primitive_accepts_candidate_{candidate}",
              qat_grad_clip_gate_error(candidate) is None, str(candidate))

    # every real E5/E6 launch passes src/quant/qat.py's gates: AM-4a pins the recipe, the clip binds to U4
    from src.quant.stages import resolve_quant_stage
    e5 = resolve_quant_stage("e5")

    def refused(argv: list[str]) -> str | None:
        try:
            QAT_TRAINER.real_run_gates(QAT_TRAINER.build_parser(e5).parse_args(argv), e5)
        except QAT_TRAINER.QATRefused as e:
            return e.code
        return None

    launch = ["--real-run", "--confirm-real-run", "--expect-head", "0" * 40, "--seed", "42",
              "--num-workers", "12"]
    check("gate_requires_expect_head", refused(["--real-run", "--confirm-real-run"]) == "expect_head_format")
    check("gate_still_requires_grad_clip_norm", refused(launch) == "grad_clip_norm_invalid")
    check("gate_requires_u4_candidate", refused(launch + ["--grad-clip-norm", "2.0"]) == "grad_clip_norm_not_candidate")
    check("gate_binds_e5_seed42_to_u4_pilot", refused(launch + ["--grad-clip-norm", "1.0"]) == "u4_pilot_required")
    flags = {a.option_strings[0] for a in QAT_TRAINER.build_parser(e5)._actions if a.option_strings}
    check("gate_takes_no_recipe_flag",
          not flags & {"--epochs", "--batch-size", "--weight-decay", "--bn-freeze-pct", "--observer-freeze-pct",
                       "--early-stop-patience", "--lr", "--momentum"}, str(sorted(flags)))
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
          "'identical recipe' wording carried forward; AM-7 makes E1-E3 unclipped")
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
    print("\nNOTE: AM-7 withdraws DISTILLATION_GRAD_CLIP_NORM (E1-E3 unclipped; the E2/E3 gate "
          "refuses any value). QAT_GRAD_CLIP_NORM and lambda_logit are unselected: official E2/E3 "
          "runs wait on the lambda_logit sweep, and E5/E6 runs on the QAT validation-only pilot.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
