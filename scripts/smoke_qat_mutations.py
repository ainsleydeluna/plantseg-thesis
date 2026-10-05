#!/usr/bin/env python3
"""(h) The mutation smoke (lane 6, L-AM4 + L-AM1q): M1-M24 and P36's refusal table, each killed by a named check.

The explicit paths of SHADOW_PATHS are copied into a temporary shadow of the repository and committed there
(one fixed commit, so the trainer records a git_head as in the checkout); the repository itself is never
written. One edit is applied at a time: a mutation replaces an exact string (which must occur exactly once);
a refusal is removed by replacing its statement (a `_refuse(...)` call or a `raise QAT...(...)`) with `pass`.
The smoke holding the killing check then runs in a subprocess inside the shadow (that check's section, or
that case alone), and the check must FAIL there, having PASSED on the unmutated shadow. The refusals are
those of P8-P10 (the launch gates, the parent, the clip binding), O2 (E6's λ/α binding) and P23-P27 (the
selections), with the record checks of convert, score, finalize and check-run-meta that P14, P24, P26 and
P28 rely on.

    python -B scripts/smoke_qat_mutations.py [--only ID,ID,...] [--no-refusals] [--keep-shadow]

Ends with one RESULT line; exit 0 only when every baseline check passes and every edit is killed.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import safe_tmpdir  # noqa: E402  (imports no repository code)

TIMEOUT = 2400
THREADS = {"value": str(os.cpu_count() or 1)}           # per smoke process; set from --workers in main()
CHECK_LINE = re.compile(r"^  (\S+)\s*: (PASS|FAIL)\b")
results: list[tuple[str, bool, str]] = []

# ------------------------------------------------------------------ the shadow: explicit paths, no listing
# Every file the smokes below open under the repository (recorded once with an audit hook on `open`, in
# every process they start); a file missing here fails the baseline, never a mutation.
SHADOW_PATHS: tuple[str, ...] = (
    "configs/augment.py",
    "configs/data.py",
    "configs/e1_student.py",
    "configs/model.py",
    "configs/qat_selection_rules.json",
    "configs/quant.py",
    "reports/e1_class_weights.json",
    "scripts/qat_epoch_eval.py",
    "scripts/run_e5.py",
    "scripts/run_e6.py",
    "scripts/select_clip.py",
    "scripts/select_qat_epoch.py",
    "scripts/smoke_qat_artifacts.py",
    "scripts/smoke_qat_runner.py",
    "scripts/smoke_qat_seeding.py",
    "scripts/smoke_qat_selection.py",
    "scripts/synthetic_ptq_fixtures.py",
    "scripts/synthetic_qat_fixtures.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/isolation.py",
    "src/data/transforms.py",
    "src/distill/__init__.py",
    "src/distill/cwd_projection.py",
    "src/distill/export.py",
    "src/distill/features.py",
    "src/distill/nmf_stream.py",
    "src/distill/segnext_teacher.py",
    "src/distill/teacher.py",
    "src/eval/__init__.py",
    "src/eval/artifacts.py",
    "src/eval/evaluate.py",
    "src/eval/metrics.py",
    "src/eval/model_loading.py",
    "src/eval/protocols.py",
    "src/eval/stage_artifacts.py",
    "src/models/__init__.py",
    "src/models/student.py",
    "src/quant/__init__.py",
    "src/quant/calibration.py",
    "src/quant/checkpoint.py",
    "src/quant/prepare.py",
    "src/quant/ptq.py",
    "src/quant/qat.py",
    "src/quant/qat_artifacts.py",
    "src/quant/qat_select.py",
    "src/quant/qconfig.py",
    "src/quant/runner.py",
    "src/quant/stages.py",
    "src/quant/x86_latency.py",
    "src/seeds.py",
    "src/training/__init__.py",
    "src/training/losses.py",
    "src/training/train_e1.py",
)

QAT, PREP, SEL, ART, QEE = ("src/quant/qat.py", "src/quant/prepare.py", "src/quant/qat_select.py",
                            "src/quant/qat_artifacts.py", "scripts/qat_epoch_eval.py")
RUNNER, SEEDING, SELECTION, ARTIFACTS = ("smoke_qat_runner", "smoke_qat_seeding", "smoke_qat_selection",
                                         "smoke_qat_artifacts")


def runner(section: str) -> tuple:
    return (RUNNER, ("--sections", section))


def case(*names: str) -> tuple:
    return (SELECTION, ("--only", ",".join(names), "--cache-dir", "{cache}/selection"))


def records() -> tuple:
    return (ARTIFACTS, ("--sections", "records", "--cache-dir", "{cache}/artifacts"))


SEEDING_ALL = (SEEDING, ())
UNITS = (ARTIFACTS, ("--sections", "units"))


def M(mid: str, what: str, file: str, old: str, new: str, run: tuple, *kills: str) -> dict:
    return {"id": mid, "what": what, "file": file, "old": old, "new": new, "run": run, "kills": kills}


MUTATIONS = [
    M("M1", "BN freeze one epoch late (epoch 11 -> 12)", QAT,
      "            if epoch == BN_FREEZE_EPOCH + 1:\n", "            if epoch == BN_FREEZE_EPOCH + 2:\n",
      runner("d1"), "d1_bn_frozen_from_step_21"),
    M("M2", "observer freeze one epoch late (epoch 13 -> 14)", QAT,
      "            if epoch == OBS_FREEZE_EPOCH + 1:\n", "            if epoch == OBS_FREEZE_EPOCH + 2:\n",
      runner("d1"), "d1_observers_off_from_step_25"),
    M("M3", "freeze steps as fractions round(0.65 T), round(0.70 T)", PREP,
      "    return bn_freeze_epoch * steps_per_epoch, obs_freeze_epoch * steps_per_epoch\n",
      "    return round(0.65 * 15 * steps_per_epoch), round(0.70 * 15 * steps_per_epoch)\n",
      runner("d1"), "d1_freeze_helper_335_gives_3350_4020"),
    M("M4", "epoch tie goes to the later epoch (>=)", SEL,
      '        elif v > scored[best]["value"]:\n', '        elif v >= scored[best]["value"]:\n',
      case("d3_tie_goes_to_earlier_epoch"), "d3_tie_goes_to_earlier_epoch"),
    M("M5", "clip tie goes to 1.0", SEL,
      '            win = cr["tie"]\n', "            win = lo\n",
      case("d5_boundary_pair_ties_to_5"), "d5_boundary_pair_ties_to_5"),
    M("M6", "float arithmetic instead of Fraction", SEL,
      "    d = abs(Fraction(a) - Fraction(b))\n    return d <= within, d\n",
      "    d = Fraction(abs(a - b))\n    return float(d) <= float(within), d\n",
      case("d5_exact_not_naive_float"), "d5_exact_not_naive_float"),
    M("M7", "band 1/1000 -> 1/10000", SEL,
      '    return Fraction(b["numerator"], b["denominator"])\n',
      '    return Fraction(b["numerator"], b["denominator"] * 10)\n',
      case("d5_boundary_pair_ties_to_5"), "d5_boundary_pair_ties_to_5"),
    M("M8", "select on the fake-quant VAL score", SEL,
      '        v = _get(s, "dataset_level.all_class_miou")\n',
      '        v = rec["ends"][e]["fake_quant_val_all_class_miou"]\n',
      case("d3_decoy_converted_wins"), "d3_decoy_converted_wins"),
    M("M9", "TRAIN loader built without seed=--seed", QAT,
      'persistent_workers=num_workers > 0, seed=seed)\n', "persistent_workers=num_workers > 0)\n",
      SEEDING_ALL, "d2_seed43_different_order", "d2_loader_seeded_with_run_seed"),
    M("M10", "set_seed removed", QAT,
      "    set_seed(seed)                                           # AM-4a item 4: before CUDA initialises\n",
      "    pass\n", SEEDING_ALL, "d2_set_seed_applied", "d2_seed42_identical_4step_losses"),
    M("M11", "observers left on during VAL", QAT,
      "            flags = observer_flags(prepared)\n            disable_observers(prepared)\n",
      "            flags = observer_flags(prepared)\n", runner("d1"), "val_leaves_state_unchanged"),
    M("M12", "T_max = steps_per_epoch", QAT,
      "    t_max = EPOCHS * steps_per_epoch\n", "    t_max = steps_per_epoch\n",
      runner("d1"), "d1_lr_all_steps_match_closed_form"),
    M("M13", "abort on a non-finite loss", QAT,
      '                if not math.isfinite(loss_v):\n                    ep["nonfinite_loss"] += 1\n',
      '                if not math.isfinite(loss_v):\n'
      '                    raise FloatingPointError("non-finite loss")\n',
      runner("nan"), "nonfinite_loss_recorded_run_reaches_epoch_15_step27"),
    M("M14", "a patience path added", QAT,
      "    step = 0\n    nonfinite_since = None\n",
      "    step = 0\n    patience = 3\n    nonfinite_since = None\n", runner("d1"), "d1_no_patience_path_grep"),
    M("M15", "the VAL restore re-enables observers from epoch 13", QAT,
      "        mods[name].observer_enabled[0] = value\n", "        mods[name].observer_enabled[0] = 1\n",
      runner("d1"), "d1_observers_off_from_step_25"),
    M("M16", "the epoch checkpoint taken inside the VAL bracket", QAT,
      "            state = cpu_state(prepared)\n            state_sha = state_digest(state)\n",
      "            disable_observers(prepared)\n            state = cpu_state(prepared)\n"
      "            state_sha = state_digest(state)\n", runner("d1"), "d1_checkpoint_observer_flags"),
    M("M17a", "gradient clipping removed", QAT,
      "gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,",
      "gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=float('inf'),",
      runner("d1"), "d1_clip_applied_at_the_step"),
    M("M17b", "gradient clipping after optimizer.step", QAT,
      "                    gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,\n"
      "                                                           norm_type=2.0, error_if_nonfinite=False)\n"
      "                    optimizer.step()\n",
      "                    optimizer.step()\n"
      "                    gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,\n"
      "                                                           norm_type=2.0, error_if_nonfinite=False)\n",
      runner("d1"), "d1_clip_applied_at_the_step"),
    M("M18a", "momentum 0", QAT,
      "lr=LEARNING_RATE, momentum=MOMENTUM,", "lr=LEARNING_RATE, momentum=0.0,",
      runner("d1"), "d1_optimizer_of_record"),
    M("M18b", "weight decay 0", QAT,
      "weight_decay=WEIGHT_DECAY, dampening=DAMPENING", "weight_decay=0.0, dampening=DAMPENING",
      runner("d1"), "d1_optimizer_of_record"),
    M("M19", "the checkpoint-to-provenance link dropped", SEL,
      '"provenance qat_checkpoint_sha256": prov.get("qat_checkpoint_sha256") == ck_sha,',
      '"provenance qat_checkpoint_sha256": True,',
      case("d3_provenance_checkpoint_mismatch_refused"), "d3_provenance_checkpoint_mismatch_refused"),
    M("M20a", "a 64-sample summary accepted (a record summary's row-count and gt_support guards dropped)", SEL,
      "    if not smoke:\n        if not (exp == act == rows == fwd):\n",
      "    if False:\n        if not (exp == act == rows == fwd):\n",
      case("d3_capped_summary_refused"), "d3_capped_summary_refused"),
    M("M20b", "an upstream-protocol summary accepted", SEL,
      '"dataset.preprocess_protocol": ev["preprocess_protocol"],', "",
      case("d3_upstream_protocol_refused"), "d3_upstream_protocol_refused"),
    M("M20c", "a record summary's row counts not compared with the rules (one guard of M20a)", SEL,
      "        if not (exp == act == rows == fwd):\n", "        if not (exp == act == fwd):\n",
      case("d3_capped_rows_alone_refused"), "d3_capped_rows_alone_refused"),
    M("M20d", "the gt_support sum not compared with the rules (the other guard of M20a)", SEL,
      '        if not isinstance(gt, list) or sum(gt) != ev["gt_support_sum"]:\n',
      "        if not isinstance(gt, list):\n",
      case("d3_gt_support_sum_refused"), "d3_gt_support_sum_refused"),
    M("M21", "the key-set comparison dropped", ART,
      "    if missing or extra:\n", "    if False:\n", UNITS, "d4_key_set_missing_key_refused"),
    M("M22a", "the predicate reduced to 'all tensors finite'", QAT,
      "    never = set(never_observed)\n    fails: list[str] = []\n",
      "    return all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()), []\n",
      UNITS, "d4_predicate_accepts_an_observed_state"),
    M("M22b", "the predicate reduced to parameters only", QAT,
      '        if kind in ("parameter", "bn_buffer", "fq_scale") and not bool(torch.isfinite(t).all()):\n',
      '        if kind == "parameter" and not bool(torch.isfinite(t).all()):\n',
      UNITS, "d4_predicate_flags_nonfinite_bn_with_finite_params"),
    M("M23", "the AM-21 rejection removed", SEL,
      "        rejected = run_rejected(rec)", "        rejected = False",
      case("d5_nonfinite_candidate_rejected_a"), "d5_nonfinite_candidate_rejected_a"),
    M("M24a", "E6's λ/α binding removed at launch", QAT,
      "        binding = e6_parent_binding(parent, args.seed, *sel)\n", "        binding = None\n",
      runner("gates"), "gate_refuses_e6_parent_not_the_selected_run"),
    M("M24b", "E6's λ/α binding removed from check-run-meta", QEE,
      '            Q.e6_parent_binding(meta.get("parent") or {}, args.seed, lam, alpha)\n', "",
      runner("profile"), "profile_stops_on_e6_parent_lambda"),
]

# ------------------------------------------------------------------ the refusal table (P36)
# file -> the functions whose refusals P8-P10, O2, P14 and P23-P28 cover
SCOPE = {
    QAT: {"resolve_parent", "read_clip_selection", "clip_binding", "read_tracked_selection", "e6_parent_binding",
          "real_run_gates", "load_source", "run_qat"},
    SEL: None,                                                   # every function
    QEE: {"_epochs", "cmd_convert", "cmd_score", "_read_json", "_pilot_clip_winner", "cmd_finalize",
          "cmd_check_run_meta"},
}
GATES, PROFILE = runner("gates"), runner("profile")
RECORDS = records()


def K(rel: str, func: str, code: str, run: tuple, *checks: str, k: int = 0) -> tuple:
    return (rel, func, code, k), (run, checks)


KILLS = dict([
    # P9: the parent run (AM-19 item 4(a))
    K(QAT, "resolve_parent", "parent_dir_test_path", GATES, "refuses_parent_dir_test_path"),
    K(QAT, "resolve_parent", "parent_dir_missing", GATES, "refuses_parent_dir_missing"),
    K(QAT, "resolve_parent", "expect_source_sha256_format", GATES, "refuses_expect_source_sha256_format"),
    K(QAT, "resolve_parent", "parent_best_json_missing", GATES, "refuses_parent_best_json_missing"),
    K(QAT, "resolve_parent", "parent_best_json_format", GATES, "refuses_parent_best_json_not_json"),
    K(QAT, "resolve_parent", "parent_best_json_format", GATES, "refuses_parent_best_json_format", k=1),
    K(QAT, "resolve_parent", "parent_checkpoint_missing", GATES, "refuses_parent_checkpoint_missing"),
    K(QAT, "resolve_parent", "parent_sha256_mismatch", GATES, "refuses_parent_sha256_mismatch"),
    K(QAT, "resolve_parent", "parent_records_missing", GATES, "refuses_parent_records_missing",
      "refuses_e1_layout_for_e6"),
    K(QAT, "resolve_parent", "parent_run_meta_rows", GATES, "refuses_parent_run_meta_rows"),
    K(QAT, "resolve_parent", "parent_run_meta_key", GATES, "refuses_parent_run_meta_without_mode"),
    K(QAT, "resolve_parent", "parent_mode_not_real", GATES, "refuses_parent_mode_not_real"),
    K(QAT, "resolve_parent", "parent_seed_mismatch", GATES, "refuses_parent_seed_mismatch"),
    K(QAT, "resolve_parent", "parent_stage_mismatch", GATES, "refuses_parent_stage_mismatch"),
    K(QAT, "resolve_parent", "parent_incomplete", GATES, "refuses_parent_incomplete_e3_abort"),
    K(QAT, "resolve_parent", "parent_incomplete", GATES, "refuses_parent_incomplete_e3_no_run_end", k=1),
    K(QAT, "resolve_parent", "parent_stage_mismatch", GATES, "refuses_e1_parent_of_another_stage", k=1),
    K(QAT, "resolve_parent", "parent_run_meta_key", GATES, "refuses_parent_run_meta_key", k=1),
    K(QAT, "resolve_parent", "parent_incomplete", GATES, "refuses_parent_incomplete_e1", k=2),
    # P10: the clip and its source
    K(QAT, "read_clip_selection", "clip_selection_sha256_format", GATES, "refuses_clip_selection_sha_format"),
    K(QAT, "read_clip_selection", "clip_selection_test_path", GATES, "refuses_clip_selection_test_path"),
    K(QAT, "read_clip_selection", "clip_selection_missing", GATES, "refuses_clip_selection_file_missing"),
    K(QAT, "read_clip_selection", "clip_selection_sha256_mismatch", GATES, "refuses_clip_selection_sha_mismatch"),
    K(QAT, "read_clip_selection", "clip_selection_format", GATES, "refuses_clip_selection_not_json"),
    K(QAT, "read_clip_selection", "clip_selection_format", GATES, "refuses_clip_selection_format", k=1),
    K(QAT, "read_clip_selection", "clip_selection_format", GATES, "refuses_clip_selection_winner_not_a_candidate",
      k=2),
    K(QAT, "clip_binding", "grad_clip_norm_invalid", GATES, "refuses_clip_absent"),
    K(QAT, "clip_binding", "grad_clip_norm_not_candidate", GATES, "refuses_clip_not_candidate"),
    K(QAT, "clip_binding", "u4_pilot_not_e5_s42", GATES, "refuses_u4_pilot_outside_e5_s42"),
    K(QAT, "clip_binding", "u4_pilot_with_clip_selection", GATES, "refuses_u4_pilot_with_selection"),
    K(QAT, "clip_binding", "u4_pilot_required", GATES, "refuses_e5_s42_without_u4_pilot"),
    K(QAT, "clip_binding", "clip_selection_required", GATES, "refuses_missing_clip_selection"),
    K(QAT, "clip_binding", "clip_selection_winner_mismatch", GATES, "refuses_clip_not_the_winner"),
    # O2: E6's λ and α selections and the parent they name
    K(QAT, "read_tracked_selection", "f'{what}_selection_missing'", GATES, "refuses_lambda_selection_missing",
      "refuses_alpha_selection_missing"),
    K(QAT, "read_tracked_selection", "selection_sha256_format", GATES, "refuses_selection_sha_format"),
    K(QAT, "read_tracked_selection", "selection_path_not_repo_relative", GATES, "refuses_selection_absolute_path"),
    K(QAT, "read_tracked_selection", "selection_test_path", GATES, "refuses_selection_test_path"),
    K(QAT, "read_tracked_selection", "selection_missing_file", GATES, "refuses_selection_missing_file"),
    K(QAT, "read_tracked_selection", "selection_not_tracked", GATES, "refuses_selection_untracked"),
    K(QAT, "read_tracked_selection", "selection_changed_since_head", GATES, "refuses_selection_changed_since_head"),
    K(QAT, "read_tracked_selection", "selection_sha256_mismatch", GATES, "refuses_selection_sha_mismatch"),
    K(QAT, "read_tracked_selection", "selection_format", GATES, "refuses_selection_not_json"),
    K(QAT, "read_tracked_selection", "selection_format", GATES, "refuses_selection_format",
      "refuses_alpha_file_as_lambda", k=1),
    K(QAT, "e6_parent_binding", "parent_stage_mismatch", GATES, "refuses_e6_parent_not_e3"),
    K(QAT, "e6_parent_binding", "parent_seed_mismatch", GATES, "refuses_e6_parent_seed"),
    K(QAT, "e6_parent_binding", "parent_lambda_mismatch", GATES, "refuses_e6_parent_lambda",
      "gate_refuses_e6_parent_not_the_selected_run"),
    K(QAT, "e6_parent_binding", "parent_alpha_mismatch", GATES, "refuses_e6_parent_alpha"),
    K(QAT, "e6_parent_binding", "parent_run_id_mismatch", GATES, "refuses_e6_s42_parent_not_alpha_winner"),
    # P8: the launch gates, in order
    K(QAT, "real_run_gates", "real_run_flag", GATES, "gate_refuses_without_real_run"),
    K(QAT, "real_run_gates", "confirm_real_run_flag", GATES, "gate_refuses_without_confirm"),
    K(QAT, "real_run_gates", "config_pins", GATES, "gate_refuses_config_pins"),
    K(QAT, "real_run_gates", "expect_head_format", GATES, "gate_refuses_head_format"),
    K(QAT, "real_run_gates", "seed", GATES, "gate_refuses_seed"),
    K(QAT, "real_run_gates", "num_workers", GATES, "gate_refuses_num_workers_0"),
    K(QAT, "real_run_gates", "selection_not_applicable", GATES, "gate_refuses_selection_flags_for_e5"),
    K(QAT, "real_run_gates", "out_dir", GATES, "gate_refuses_out_dir_in_repo", "gate_refuses_out_dir_not_empty",
      "gate_refuses_out_dir_test_path"),
    K(QAT, "real_run_gates", "cuda_required", GATES, "gate_refuses_cpu"),
    K(QAT, "real_run_gates", "cuda_initialized_before_seed", GATES, "gate_refuses_cuda_initialised"),
    K(QAT, "real_run_gates", "tf32_not_default", GATES, "gate_refuses_tf32_not_default"),
    K(QAT, "real_run_gates", "backend_unavailable", GATES, "gate_refuses_backend_unavailable"),
    K(QAT, "real_run_gates", "data_root_not_trainval_only", GATES, "gate_refuses_data_root_not_trainval_only"),
    K(QAT, "real_run_gates", "expect_head_mismatch", GATES, "gate_refuses_head_mismatch"),
    K(QAT, "real_run_gates", "code_not_clean", GATES, "gate_refuses_code_not_clean"),
    K(QAT, "load_source", "e.code", GATES, "load_source_refuses_projection_in_e6_student"),
    K(QAT, "load_source", "source_changed", GATES, "load_source_refuses_changed_source"),
    K(QAT, "load_source", "source_projection_keys", GATES, "load_source_refuses_projection_keys"),
    # P8 inside run_qat, and its two STOPs (P2's optimizer of record, P5's VAL bracket)
    K(QAT, "run_qat", "stage_not_qat", GATES, "run_refuses_ptq_stage"),
    K(QAT, "run_qat", "mode_invalid", GATES, "run_refuses_mode"),
    K(QAT, "run_qat", "real_run_test_hooks", GATES, "run_refuses_test_hooks_in_real"),
    K(QAT, "run_qat", "clip_source", GATES, "run_refuses_clip_source_in_real"),
    K(QAT, "run_qat", "config_pins", GATES, "run_refuses_config_pins"),
    K(QAT, "run_qat", "grad_clip_norm_invalid", GATES, "run_refuses_bad_clip"),
    K(QAT, "run_qat", "cuda_required", GATES, "run_refuses_cpu_in_real"),
    K(QAT, "run_qat", "cuda_initialized_before_seed", GATES, "run_refuses_cuda_initialised_in_real"),
    K(QAT, "run_qat", "out_dir_not_empty", GATES, "run_refuses_nonempty_out_dir"),
    K(QAT, "run_qat", "steps_per_epoch", GATES, "run_refuses_steps_per_epoch_in_real"),
    K(QAT, "run_qat", "loader_of_record", GATES, "run_refuses_loader_not_of_record_in_real"),
    K(QAT, "run_qat", "empty_loader", GATES, "run_refuses_empty_loader"),
    K(QAT, "run_qat", "backend_unavailable", GATES, "run_refuses_unavailable_backend"),
    K(QAT, "run_qat", "unfused_batchnorm", GATES, "run_refuses_unfused_bn_in_real"),
    K(QAT, "run_qat", "optimizer_of_record", GATES, "run_stops_on_an_optimizer_not_of_record"),
    K(QAT, "run_qat", "val_bracket_changed_state", GATES, "run_stops_when_val_changes_the_state"),
    # P23-P27: the epoch selection
    K(SEL, "load_rules", "rules_mismatch", case("d3_rules_disagreeing_with_config_refused"),
      "d3_rules_disagreeing_with_config_refused"),
    K(SEL, "_json", "record_unreadable", case("d3_unreadable_record_refused"), "d3_unreadable_record_refused"),
    K(SEL, "epoch_selection", "telemetry_sha256_mismatch", case("d3_telemetry_sha_mismatch_refused"),
      "d3_telemetry_sha_mismatch_refused"),
    K(SEL, "epoch_selection", "run_not_real", case("d3_smoke_run_refused"), "d3_smoke_run_refused"),
    K(SEL, "epoch_selection", "stage", case("d3_non_qat_stage_refused"), "d3_non_qat_stage_refused"),
    K(SEL, "epoch_selection", "stop_present", case("d3_stop_file_refused"), "d3_stop_file_refused"),
    K(SEL, "epoch_selection", "eval_record_missing", case("d3_missing_eval_record_exit_3"),
      "d3_missing_eval_record_exit_3"),
    K(SEL, "epoch_selection", "eval_record_mismatch", case("d3_swapped_eval_dir_refused"),
      "d3_swapped_eval_dir_refused"),
    K(SEL, "epoch_selection", "purpose_not_record", case("d3_timing_purpose_refused"), "d3_timing_purpose_refused"),
    K(SEL, "epoch_selection", "smoke_inputs", case("d3_smoke_input_records_refused"), "d3_smoke_input_records_refused"),
    K(SEL, "epoch_selection", "eval_dir_mixed", case("d3_two_commits_refused"), "d3_two_commits_refused"),
    K(SEL, "epoch_selection", "eval_incomplete", case("d3_partial_eval_incomplete"), "d3_partial_eval_incomplete"),
    K(SEL, "epoch_selection", "chain_checkpoint", case("d3_row_checkpoint_mismatch_refused"),
      "d3_row_checkpoint_mismatch_refused"),
    K(SEL, "epoch_selection", "exclusion_inconsistent", case("d3_excluding_a_finite_epoch_refused"),
      "d3_excluding_a_finite_epoch_refused"),
    K(SEL, "epoch_selection", "score_missing", case("d3_unscored_epoch_exit_3"), "d3_unscored_epoch_exit_3"),
    K(SEL, "epoch_selection", "scored_nonfinite_state", case("d3_scored_nonfinite_epoch_refused"),
      "d3_scored_nonfinite_epoch_refused"),
    K(SEL, "epoch_selection", "conversion_missing", case("d3_missing_provenance_exit_3"),
      "d3_missing_provenance_exit_3"),
    K(SEL, "epoch_selection", "chain_provenance", case("d3_provenance_not_scored_one_refused"),
      "d3_provenance_not_scored_one_refused"),
    K(SEL, "epoch_selection", "conversion_missing", case("d3_missing_torchscript_exit_3"),
      "d3_missing_torchscript_exit_3", k=1),
    K(SEL, "epoch_selection", "eval_dir_mixed", case("d3_provenance_other_commit_refused"),
      "d3_provenance_other_commit_refused", k=1),
    K(SEL, "epoch_selection", "chain_broken", case("d3_identity_sha_mismatch_refused",
                                                   "d3_provenance_checkpoint_mismatch_refused"),
      "d3_identity_sha_mismatch_refused", "d3_provenance_checkpoint_mismatch_refused"),
    K(SEL, "epoch_selection", "score_missing", case("d3_missing_score_exit_3"), "d3_missing_score_exit_3", k=1),
    K(SEL, "epoch_selection", "score_artifact_invalid", case("d3_sha_mismatch_refused"), "d3_sha_mismatch_refused"),
    K(SEL, "epoch_selection", "chain_summary", case("d3_summary_not_scored_one_refused"),
      "d3_summary_not_scored_one_refused"),
    K(SEL, "epoch_selection", "eval_dir_mixed", case("d3_summary_commit_not_eval_commit_refused"),
      "d3_summary_commit_not_eval_commit_refused", k=2),
    K(SEL, "epoch_selection", "chain_summary", case("d3_summary_identity_refused"), "d3_summary_identity_refused", k=1),
    K(SEL, "epoch_selection", "evaluator_not_literal", case("d3_evaluator_not_literal_refused"),
      "d3_evaluator_not_literal_refused"),
    K(SEL, "epoch_selection", "summary_values", case("d3_capped_summary_refused", "d3_non_val_split_refused",
                                                     "d3_upstream_protocol_refused", "d3_smoke_status_refused"),
      "d3_capped_summary_refused", "d3_non_val_split_refused", "d3_upstream_protocol_refused",
      "d3_smoke_status_refused"),
    K(SEL, "epoch_selection", "score_invalid", case("d3_score_outside_unit_interval_refused",
                                                    "d3_score_not_float_refused"),
      "d3_score_outside_unit_interval_refused", "d3_score_not_float_refused"),
    K(SEL, "epoch_selection", "scores_dir_contents", case("d3_extra_scores_entry_refused"),
      "d3_extra_scores_entry_refused"),
    K(SEL, "epoch_selection", "no_convertible_epoch", case("d3_no_convertible_epoch_refused_exit_2"),
      "d3_no_convertible_epoch_refused_exit_2"),
    K(SEL, "epoch_selection", "summaries_differ", case("d3_metric_impl_differs_across_summaries",
                                                       "d3_null_image_digest_refused"),
      "d3_metric_impl_differs_across_summaries", "d3_null_image_digest_refused"),
    K(SEL, "epoch_selection", "winner_changed", case("d3_winner_changed_during_selection_refused"),
      "d3_winner_changed_during_selection_refused"),
    # P27 and O1: the U4 clip
    K(SEL, "clip_selection", "candidates", case("d5_one_candidate_refused"), "d5_one_candidate_refused"),
    K(SEL, "clip_selection", "not_a_pilot_run", case("d5_non_pilot_run_refused"), "d5_non_pilot_run_refused"),
    K(SEL, "clip_selection", "epoch_selection_missing", case("d5_missing_run_refused"), "d5_missing_run_refused"),
    K(SEL, "clip_selection", "epoch_selection_format", case("d5_selection_of_other_rules_refused"),
      "d5_selection_of_other_rules_refused"),
    K(SEL, "clip_selection", "selection_differs", case("d5_edited_epoch_selection_refused"),
      "d5_edited_epoch_selection_refused"),
    K(SEL, "clip_selection", "clip_values", case("d5_two_clip_1_runs_refused"), "d5_two_clip_1_runs_refused"),
    K(SEL, "clip_selection", "recipe_mismatch", case("d5_recipe_mismatch_refused", "d5_cpu_model_difference_refused"),
      "d5_recipe_mismatch_refused", "d5_cpu_model_difference_refused"),
    K(SEL, "clip_selection", "recipe_identity_null", case("d5_null_git_head_refused"), "d5_null_git_head_refused"),
    K(SEL, "clip_selection", "batch_order_differs", case("d5_batch_order_mismatch_refused"),
      "d5_batch_order_mismatch_refused"),
    K(SEL, "clip_selection", "pilot_evals_differ", case("d5_pilot_summary_commits_differ_refused",
                                                        "d5_pilot_host_labels_differ_refused"),
      "d5_pilot_summary_commits_differ_refused", "d5_pilot_host_labels_differ_refused"),
    K(SEL, "clip_selection", "no_winner", case("d5_both_rejected_no_winner_exit_2_a"),
      "d5_both_rejected_no_winner_exit_2_a"),
    # P24, P26: convert's and score's own record checks; P16's freeze cross-check
    K(QEE, "_epochs", "epochs_with_record", RECORDS, "d4_convert_record_takes_no_epochs",
      "d4_score_record_takes_no_epochs"),
    K(QEE, "_epochs", "epochs_range", RECORDS, "d4_convert_epochs_out_of_range"),
    K(QEE, "cmd_convert", "expect_telemetry_sha256_required", RECORDS, "d4_convert_record_requires_telemetry_sha"),
    K(QEE, "cmd_convert", "eval_dir_not_fresh", RECORDS, "d4_convert_eval_dir_not_fresh"),
    K(QEE, "cmd_convert", "telemetry_sha256_mismatch", RECORDS, "d4_convert_telemetry_sha_mismatch"),
    K(QEE, "cmd_convert", "run_not_real", RECORDS, "d4_convert_smoke_run_refused"),
    K(QEE, "cmd_convert", "freeze_cross_check", RECORDS, "d4_convert_stops_when_a_freeze_did_not_take"),
    K(QEE, "cmd_score", "stop_present", RECORDS, "d4_score_refuses_a_stop_file"),
    K(QEE, "cmd_score", "convert_record_missing", RECORDS, "d4_score_without_convert_record_exit_3"),
    K(QEE, "cmd_score", "convert_record_mismatch", RECORDS, "d4_score_convert_record_of_another_run"),
    K(QEE, "cmd_score", "purpose_mismatch", RECORDS, "d4_score_purpose_mismatch"),
    K(QEE, "cmd_score", "convert_record_epochs", RECORDS, "d4_score_record_needs_all_15_epochs"),
    K(QEE, "cmd_score", "epoch_not_converted", RECORDS, "d4_score_unconverted_epoch_refused"),
    K(QEE, "cmd_score", "scores_not_fresh", RECORDS, "d4_score_scores_dir_not_fresh"),
    K(QEE, "cmd_score", "output_exists", RECORDS, "d4_score_runs_once"),
    # P28: finalize
    K(QEE, "_read_json", "code", case("finalize_selection_not_json"), "finalize_selection_not_json"),
    K(QEE, "_read_json", "code", case("finalize_selection_not_an_object"), "finalize_selection_not_an_object", k=1),
    K(QEE, "_pilot_clip_winner", "clip_selection_missing", case("finalize_clip_selection_file_missing"),
      "finalize_clip_selection_file_missing"),
    K(QEE, "_pilot_clip_winner", "clip_selection_sha256_mismatch", case("finalize_checks_a_pinned_clip_selection_sha"),
      "finalize_checks_a_pinned_clip_selection_sha"),
    K(QEE, "_pilot_clip_winner", "clip_selection_format", case("finalize_clip_selection_of_another_format"),
      "finalize_clip_selection_of_another_format"),
    K(QEE, "_pilot_clip_winner", "not_the_clip_winner", case("finalize_refuses_the_retained_loser"),
      "finalize_refuses_the_retained_loser"),
    K(QEE, "_pilot_clip_winner", "clip_selection_stale", case("finalize_refuses_a_stale_clip_selection"),
      "finalize_refuses_a_stale_clip_selection"),
    K(QEE, "cmd_finalize", "fresh_process_required", case("finalize_fresh_process_guard"),
      "finalize_fresh_process_guard"),
    K(QEE, "cmd_finalize", "run_not_real", case("finalize_smoke_run_refused"), "finalize_smoke_run_refused"),
    K(QEE, "cmd_finalize", "output_exists", case("finalize_serves_the_winner"), "finalize_runs_once"),
    K(QEE, "cmd_finalize", "epoch_selection_missing", case("finalize_without_epoch_selection_exit_3"),
      "finalize_without_epoch_selection_exit_3"),
    K(QEE, "cmd_finalize", "epoch_selection_format", case("finalize_selection_of_other_rules_refused"),
      "finalize_selection_of_other_rules_refused"),
    K(QEE, "cmd_finalize", "epoch_selection_mismatch", case("finalize_selection_of_another_run_refused"),
      "finalize_selection_of_another_run_refused"),
    K(QEE, "cmd_finalize", "smoke_inputs", case("finalize_smoke_input_selection_refused"),
      "finalize_smoke_input_selection_refused"),
    K(QEE, "cmd_finalize", "clip_selection_required", case("finalize_pilot_requires_clip_selection"),
      "finalize_pilot_requires_clip_selection"),
    K(QEE, "cmd_finalize", "clip_selection_not_applicable", case("finalize_non_pilot_takes_no_clip_selection"),
      "finalize_non_pilot_takes_no_clip_selection"),
    K(QEE, "cmd_finalize", "winner_checkpoint_changed", case("finalize_winner_checkpoint_changed_refused"),
      "finalize_winner_checkpoint_changed_refused"),
    K(QEE, "cmd_finalize", "winner_files_changed", case("finalize_winner_artifact_changed_refused"),
      "finalize_winner_artifact_changed_refused"),
    K(QEE, "cmd_finalize", "winner_files_changed", case("finalize_state_dict_companion_changed_refused"),
      "finalize_state_dict_companion_changed_refused", k=1),
    K(QEE, "cmd_finalize", "x86_backend_unavailable", case("finalize_refuses_a_non_x86_engine"),
      "finalize_refuses_a_non_x86_engine"),
    K(QEE, "cmd_finalize", "x86_copy_checks", case("finalize_stops_on_a_failed_x86_check"),
      "finalize_stops_on_a_failed_x86_check"),
    # P14 / P10 / O2: check-run-meta
    K(QEE, "cmd_check_run_meta", "telemetry_missing", PROFILE, "profile_run_without_telemetry_exit_3"),
    K(QEE, "cmd_check_run_meta", "run_meta_rows", PROFILE, "profile_first_row_not_json"),
    K(QEE, "cmd_check_run_meta", "run_meta_rows", PROFILE, "profile_first_row_not_run_meta", k=1),
    K(QEE, "cmd_check_run_meta", "run_meta_profile", PROFILE, "profile_stops_on_num_workers", "profile_stops_on_lr"),
])


# ------------------------------------------------------------------ machinery
def build_shadow(dst: Path) -> str:
    """Copy SHADOW_PATHS and commit exactly them, so the trainer records a git_head as it does in the checkout.

    The commit's author, committer, date and message are fixed, so every shadow has the same HEAD and the runs
    kept in the cache (recorded in one shadow) match the runs of every other shadow. Returns that HEAD.
    """
    for rel in SHADOW_PATHS:
        src = REPO / rel
        if not src.is_file():
            raise SystemExit(f"RESULT: ERROR shadow path {rel} is missing from the checkout")
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / rel)
    env = {**os.environ, "GIT_AUTHOR_NAME": "shadow", "GIT_AUTHOR_EMAIL": "shadow@example.invalid",
           "GIT_COMMITTER_NAME": "shadow", "GIT_COMMITTER_EMAIL": "shadow@example.invalid",
           "GIT_AUTHOR_DATE": "2026-10-05T00:00:00+00:00", "GIT_COMMITTER_DATE": "2026-10-05T00:00:00+00:00"}
    for argv in (["init", "-q"], ["add", "--", *SHADOW_PATHS],
                 ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "shadow"]):
        subprocess.run(["git", "-C", str(dst), *argv], check=True, capture_output=True, env=env)
    return subprocess.run(["git", "-C", str(dst), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def run_smoke(shadow: Path, cache: Path, spec: tuple) -> tuple[dict, str, float]:
    smoke, args = spec
    argv = [sys.executable, "-B", f"scripts/{smoke}.py", *[a.replace("{cache}", str(cache)) for a in args]]
    t0 = time.time()
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": THREADS["value"],
               "MKL_NUM_THREADS": THREADS["value"]}
        p = subprocess.run(argv, cwd=str(shadow), capture_output=True, text=True, timeout=TIMEOUT, env=env)
        out = p.stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += "\nRESULT: TIMEOUT"
    statuses = {}
    in_checks = False
    for line in out.splitlines():
        if line.startswith("[CHECKS]"):
            in_checks = True
            continue
        if in_checks:
            m = CHECK_LINE.match(line)
            if m:
                statuses[m.group(1)] = m.group(2)
    tail = [ln for ln in out.splitlines() if ln.startswith("RESULT:")]
    return statuses, (tail[-1] if tail else out.strip()[-200:]), time.time() - t0


def _char_offset(line: str, col: int) -> int:
    return len(line.encode("utf-8")[:col].decode("utf-8"))


def refusal_sites(text: str, rel: str) -> list[dict]:
    tree = ast.parse(text)
    owner = {}
    for fn in sorted((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                     key=lambda n: n.lineno):
        for n in ast.walk(fn):
            if n is not fn:
                owner[id(n)] = fn.name
    out, seen = [], {}
    for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
        call = None
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            call = node.exc
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        if call is None:
            continue
        f = call.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if isinstance(node, ast.Expr) and name != "_refuse":
            continue
        if name not in ("_refuse", "QATRefused", "QATStop", "QATIncomplete") or not call.args:
            continue
        func = owner.get(id(node))
        scope = SCOPE.get(rel)
        if func is None or func == "_refuse" or (scope is not None and func not in scope):
            continue
        a0 = call.args[0]
        if not (isinstance(a0, ast.Constant) and isinstance(a0.value, str)):
            code = ast.unparse(a0)
            k = seen.get((func, code), 0)
            seen[(func, code)] = k + 1
            out.append({"rel": rel, "func": func, "code": code, "k": k, "node": node, "literal": False})
            continue
        k = seen.get((func, a0.value), 0)
        seen[(func, a0.value)] = k + 1
        out.append({"rel": rel, "func": func, "code": a0.value, "k": k, "node": node, "literal": True})
    return out


def removed(text: str, node) -> str:
    lines = text.splitlines(keepends=True)
    start = sum(len(x) for x in lines[:node.lineno - 1]) + _char_offset(lines[node.lineno - 1], node.col_offset)
    end = sum(len(x) for x in lines[:node.end_lineno - 1]) + _char_offset(lines[node.end_lineno - 1],
                                                                            node.end_col_offset)
    return text[:start] + "pass" + text[end:]


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def jobs_of(shadow: Path, refusals: bool) -> list[dict]:
    jobs = [{"id": m["id"], "rel": m["file"], "what": m["what"], "run": m["run"], "kills": m["kills"],
             "edit": ("replace", m["old"], m["new"])} for m in MUTATIONS]
    if not refusals:
        return jobs
    found = set()
    for rel in SCOPE:
        text = (shadow / rel).read_text(encoding="utf-8")
        for site in refusal_sites(text, rel):
            key = (rel, site["func"], site["code"], site["k"])
            found.add(key)
            sid = f"{Path(rel).name}:{site['func']}:{site['code']}#{site['k']}"
            if key not in KILLS:
                check(f"{sid}_has_a_killing_check", False, "no entry in the refusal table")
                continue
            run, kills = KILLS[key]
            jobs.append({"id": sid, "rel": rel, "what": f"refusal removed (line {site['node'].lineno})", "run": run,
                         "kills": kills, "edit": ("site", key)})
    stale = sorted(set(KILLS) - found)
    check("refusal_table_names_only_real_sites", not stale, str(stale[:5]))
    return jobs


def edited(text: str, rel: str, edit: tuple) -> str:
    if edit[0] == "replace":
        _, old, new = edit
        if text.count(old) != 1:
            raise ValueError(f"the target occurs {text.count(old)} times in {rel}")
        out = text.replace(old, new)
    else:
        site = next(x for x in refusal_sites(text, rel) if (rel, x["func"], x["code"], x["k"]) == edit[1])
        out = removed(text, site["node"])
    ast.parse(out)                                             # every edit still compiles
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated job ids (M1, ..., or file:function:code#k)")
    ap.add_argument("--no-refusals", action="store_true", help="the M1-M24 mutations only")
    ap.add_argument("--workers", type=int, default=2, help="shadows edited and run in parallel")
    ap.add_argument("--keep-shadow", action="store_true")
    args = ap.parse_args()
    t_all = time.time()
    root = safe_tmpdir("smoke_qat_mutations_")
    cache = root / "cache"
    shadows = [root / f"shadow{i}" for i in range(max(1, args.workers))]
    # two torch processes spinning on the same cores slow each other many times over: share the CPUs out
    THREADS["value"] = str(max(1, (os.cpu_count() or 1) // len(shadows)))
    heads = {build_shadow(sh) for sh in shadows}
    if len(heads) != 1:
        raise SystemExit(f"RESULT: ERROR the shadows' commits differ: {sorted(heads)}")
    jobs = jobs_of(shadows[0], refusals=not args.no_refusals)
    if args.only:
        only = set(args.only.split(","))
        jobs = [j for j in jobs if j["id"] in only]
    print("=" * 78)
    print("QAT MUTATION SMOKE (h) -- M1-M24 and P36's refusal table, each killed by a named check")
    print(f"{len(jobs)} edits | {len(SHADOW_PATHS)} shadow files x {len(shadows)} at commit {heads.pop()[:12]} | "
          f"temp {root}")
    print("=" * 78)
    # the baselines: every killing check passes on the unmutated shadow
    specs = []
    for j in jobs:
        spec = (SELECTION, ("--cache-dir", "{cache}/selection")) if j["run"][0] == SELECTION else j["run"]
        if spec not in specs:
            specs.append(spec)
    base = {}
    for spec in specs:
        statuses, res, secs = run_smoke(shadows[0], cache, spec)
        base[spec] = statuses
        print(f"[baseline] {spec[0]} {' '.join(spec[1]).replace('{cache}', 'CACHE')}: {res} ({secs:.0f}s)")
    for j in jobs:
        spec = (SELECTION, ("--cache-dir", "{cache}/selection")) if j["run"][0] == SELECTION else j["run"]
        bad = [c for c in j["kills"] if base[spec].get(c) != "PASS"]
        if j["run"][0] == SELECTION:
            bad += [c for c in j["run"][1][1].split(",") if base[spec].get(c) != "PASS"]
        j["baseline_ok"] = not bad
        if bad:
            check(f"{j['id']}_baseline", False, f"killing checks not passing unmutated: {sorted(set(bad))}")

    # one edit per job, in its own shadow
    import queue
    import threading
    free: "queue.Queue[Path]" = queue.Queue()
    for sh in shadows:
        free.put(sh)
    lock = threading.Lock()
    todo = [j for j in jobs if j["baseline_ok"]]
    done = {"n": 0}

    def work(j: dict) -> None:
        sh = free.get()
        try:
            target = sh / j["rel"]
            original = target.read_text(encoding="utf-8")
            try:
                target.write_text(edited(original, j["rel"], j["edit"]), encoding="utf-8")
            except Exception as e:                       # noqa: BLE001 -- the edit is reported, never run
                with lock:
                    check(f"{j['id']}_killed", False, f"the edit could not be applied: {type(e).__name__}: {e}")
                return
            try:
                statuses, res, secs = run_smoke(sh, cache, j["run"])
            finally:
                target.write_text(original, encoding="utf-8")
            failed = [c for c in j["kills"] if statuses.get(c) == "FAIL"]
            unreached = [c for c in j["kills"] if c not in statuses]
            usage_error = res.startswith("RESULT: ERROR unknown")
            killed = j["baseline_ok"] and not usage_error and bool(failed or unreached)
            how = (f"by {', '.join(failed)}" if failed else
                   f"the smoke did not reach {', '.join(unreached)} ({res})" if unreached else "SURVIVED")
            with lock:
                done["n"] += 1
                print(f"  [{done['n']}/{len(todo)}] {j['id']:58} {'KILLED' if killed else 'SURVIVED'}  {how} "
                      f"({secs:.0f}s)", flush=True)
                check(f"{j['id']}_killed", killed, f"{j['what']}: {how}")
        finally:
            free.put(sh)

    print("\n[edits]")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(shadows)) as pool:
        list(pool.map(work, todo))
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:64}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    if not args.keep_shadow:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n[time] {time.time() - t_all:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) and results else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
