#!/usr/bin/env python3
"""Lane L-AM3, B-1: the mutation smoke of the AM-3 trigger. MT1-MT19 and the refusal table of
src/quant/e6kd_trigger.py, each killed by a named check of scripts/smoke_e6kd_trigger.py.
Lane L-AM3, B-2: E6-KD. MK1-MK57 and the refusal tables of src/quant/qat_kd.py and scripts/e6kd_eval.py, each killed
by a named check of scripts/smoke_e6kd_runner.py or scripts/smoke_e6kd_selection.py.

Q2's harness, scripts/smoke_qat_mutations.py, runs it unchanged:
- its self-check (ruling Q2-F/2's job handling, measured, with harness processes of its own);
- its shadow: SHADOW_PATHS copied into a temporary directory and committed there, so the repository is never written;
- its jobs: each smoke in a session of its own, the named check reported FAIL after it PASSED on the unmutated
  shadow, with the same argv;
- its refusal finder: a `_refuse(...)`, a `raise QATRefused(...)` or a `raise QATIncomplete(...)` replaced by `pass`.

This file gives the harness the lane's shadow paths, mutations, refusal scope and refusal table, through the
harness module's globals in this process only. Nothing in Q2's file changes.

    python -B scripts/smoke_e6kd_mutations.py [--only ID,ID,...] [--no-refusals] [--workers N] [--keep-shadow]
                                              [--self-check]

Ends with one RESULT line; exit 0 only when every baseline check passes and every edit is killed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.smoke_qat_mutations as H  # noqa: E402

TRG = "src/quant/e6kd_trigger.py"
SMOKE = "smoke_e6kd_trigger"
# B-2: E6-KD's files, the smokes that kill its edits, and what they load that Q2's shadow does not hold
QKD, EVL, KDRULES = "src/quant/qat_kd.py", "scripts/e6kd_eval.py", "configs/e6kd_selection_rules.json"
STG, RUN, SEL, SAF, TDS = ("src/quant/stages.py", "src/quant/runner.py", "src/quant/qat_select.py",
                           "src/eval/stage_artifacts.py", "src/training/train_distill.py")
KD_RUNNER, KD_SELECTION = "smoke_e6kd_runner", "smoke_e6kd_selection"
SHADOW_PATHS = (*H.SHADOW_PATHS, TRG, "scripts/e6kd_trigger.py", "scripts/smoke_e6kd_trigger.py",
                QKD, "scripts/run_e6kd.py", EVL, KDRULES, "configs/qat_selection_rules_am23.json",
                "scripts/smoke_e6kd_runner.py", "scripts/smoke_e6kd_selection.py", TDS, "configs/distill.py",
                "scripts/preflight_distill.py", "scripts/preflight_e1_trainval.py", "scripts/preflight_e1.py",
                "scripts/invariance_harness.py", "src/training/teacher_components.py")


def case(*names: str) -> tuple:
    return (SMOKE, ("--only", ",".join(names), "--cache-dir", "{cache}/trigger"))


def kcase(*names: str) -> tuple:
    """B-2: cases of scripts/smoke_e6kd_runner.py (each trains what it reads: no cache)."""
    return (KD_RUNNER, ("--only", ",".join(names)))


def scase(*names: str) -> tuple:
    """B-2: cases of scripts/smoke_e6kd_selection.py (its raw E6-KD runs kept in the cache)."""
    return (KD_SELECTION, ("--only", ",".join(names), "--cache-dir", "{cache}/e6kd_selection"))


M, MM = H.M, H.MM
MUTATIONS = [
    M("MT1", "the threshold compared with >= (strict > undone)", TRG,
      "        triggered = drop_pp > THRESHOLD_PP\n", "        triggered = drop_pp >= THRESHOLD_PP\n",
      case("d1_drop_1_00_not_triggered"), "d1_drop_1_00_not_triggered"),
    M("MT2", "the drop taken in float, then made decimal", TRG,
      "        drop = d3 - d6\n", "        drop = Decimal(repr(e3 - e6))\n",
      case("d1_decimal_not_float"), "d1_decimal_not_float"),
    M("MT3", "the drop taken as E6 - E3", TRG,
      "        drop = d3 - d6\n", "        drop = d6 - d3\n",
      case("d1_drop_1_01_triggered"), "d1_drop_1_01_triggered"),
    M("MT4", "the threshold read as a fraction (0.01) against the drop in pp", TRG,
      "        triggered = drop_pp > THRESHOLD_PP\n", "        triggered = drop_pp > THRESHOLD_PP / 100\n",
      case("d1_drop_0_50_not_triggered"), "d1_drop_0_50_not_triggered"),
    M("MT5", "E3's value taken from best.json", TRG,
      '    e3 = _get(summaries["e3"], "dataset_level.all_class_miou")\n',
      '    e3 = ctx["parent"]["best_val_miou_all_class"]\n',
      case("d1_e3_value_is_the_evaluator_score_not_best_json"), "d1_e3_value_is_the_evaluator_score_not_best_json"),
    M("MT6", "E6's re-score no longer compared with block B's value", TRG,
      '    if not (type(w["value"]) is float and e6 == w["value"]):\n', '    if not type(w["value"]) is float:\n',
      case("refuses_e6_rescore_differs"), "refuses_e6_rescore_differs"),
    M("MT7", "the summaries' split unchecked", TRG,
      '    want = {"dataset.split": ev["split"], "run.env.device": ev["device"],\n',
      '    want = {"run.env.device": ev["device"],\n',
      case("refuses_e6_summary_not_val"), "refuses_e6_summary_not_val"),
    M("MT8", "the thread setting dropped from the same-code-path keys", TRG,
      '                       "run.eval_runtime.torch_num_threads", "run.eval_runtime.image_digest", '
      '"metric_protocol")\n',
      '                       "run.eval_runtime.image_digest", "metric_protocol")\n',
      case("refuses_summaries_not_same_code_path"), "refuses_summaries_not_same_code_path"),
    M("MT9", "E6's stored selection not compared with its recomputation", TRG,
      "    diff = S.selection_differences(stored, recomputed)\n", "    diff = []\n",
      case("refuses_e6_selection_recompute_differs"), "refuses_e6_selection_recompute_differs"),
    M("MT10", "the α winner's ckpt_sha256 checked for presence only", TRG,
      '    if parent["checkpoint_sha256"] != alpha["winner"].get("ckpt_sha256"):\n',
      '    if alpha["winner"].get("ckpt_sha256") is None:\n',
      case("refuses_e3_checkpoint_not_alpha_winner_sha"), "refuses_e3_checkpoint_not_alpha_winner_sha"),
    M("MT11", "E6 with no model read as a missing selection (incomplete)", TRG,
      '            if e.code == "no_convertible_epoch":\n', '            if e.code == "no_model":\n',
      case("refuses_e6_no_model"), "refuses_e6_no_model"),
    M("MT12", "a wall-clock time written into the trigger record", TRG,
      '    doc = {"format": TRIGGER_FORMAT, "lane": LANE, "rule": RULE, "seed": SEED,\n',
      '    doc = {"format": TRIGGER_FORMAT, "lane": LANE, "rule": RULE, "seed": SEED, "created_wall_clock": time.time(),\n',
      case("decide_twice_identical_bytes"), "decide_twice_identical_bytes"),
    M("MT13", "the evaluator commands run at batch size 16", TRG,
      '            "--artifact-status", "provisional", "--device", "cpu", "--batch-size", "1", "--out-dir", '
      'str(out_dir),\n',
      '            "--artifact-status", "provisional", "--device", "cpu", "--batch-size", "16", "--out-dir", '
      'str(out_dir),\n',
      case("score_runs_the_literal_evaluator_commands"), "score_runs_the_literal_evaluator_commands"),
    M("MT14", "the SL-1 name check made after a probe of the path (is_dir)", TRG,
      "        if Q.names_test(value):\n", "        if Path(value).is_dir() or Q.names_test(value):\n",
      case("g1_forbidden_names_refused_before_any_probe"), "g1_forbidden_names_refused_before_any_probe"),
    # DL-24 F1: decide binds the code files to score's
    M("MT15", "decide no longer compares its code files with score's", TRG,
      '    now = {"git_head": ident["git_head"], "code_files_sha256": ident["code_files_sha256"],\n',
      '    now = {"git_head": ident["git_head"],\n',
      case("refuses_code_changed_since_score"), "refuses_code_changed_since_score"),
    # DL-24 R-F2: a record made from smoke inputs says so on decide's RESULT line
    M("MT16", "decide's RESULT line no longer marks smoke inputs", TRG,
      '    smoke = " (smoke inputs, not a record)" if doc.get("smoke_inputs") is not False else ""\n',
      '    smoke = ""\n',
      case("d1_result_line_marks_smoke_inputs"), "d1_result_line_marks_smoke_inputs"),
    # DL-24 S-F2: conditions of the summary gates that no case varied alone
    M("MT17", "the summaries' governed_paths_clean unchecked", TRG,
      '            "run.governed_paths_clean": ev["governed_paths_clean"], "run.stage": stage, '
      '"run.model_role": "student",\n',
      '            "run.stage": stage, "run.model_role": "student",\n',
      case("refuses_summary_governed_paths_dirty"), "refuses_summary_governed_paths_dirty"),
    M("MT18", "a summary scored at another commit than trigger_eval.json's accepted", TRG,
      '        if _get(s, "run.repo_commit") != tev.get("git_head"):\n', "        if False:\n",
      case("refuses_summaries_scored_at_another_commit"), "refuses_summaries_scored_at_another_commit"),
    M("MT19", "the sum of per_class.gt_support unchecked", TRG,
      '        if not isinstance(gt, list) or sum(gt) != ev["gt_support_sum"]:\n',
      "        if not isinstance(gt, list):\n",
      case("refuses_summary_gt_support_sum"), "refuses_summary_gt_support_sum"),
    # ---- B-2: E6-KD (plan §4.6 MK1-MK34 as the rulings amend them; LB16's two; LB17; check-run-meta's)
    M("MK1", "the KD weights at 1.0x E3's", QKD, "WEIGHT_FACTOR = 0.5 ", "WEIGHT_FACTOR = 1.0 ",
      kcase("d2_weights_exactly_half"), "d2_weights_exactly_half"),
    M("MK2", "β left at E3's 3", QKD, "    return {t: WEIGHT_FACTOR * float(e3_weights[t]) for t in TERMS}\n",
      '    return {t: (1.0 if t == "cwd_logit" else WEIGHT_FACTOR) * float(e3_weights[t]) for t in TERMS}\n',
      kcase("d2_weights_exactly_half"), "d2_weights_exactly_half"),
    M("MK3", "T_logit recorded as 2", QKD, '        "T_logit": TD.T_LOGIT, "T_cwd": TD.T_CWD, "cwd_C": TD.CWD_C_FEAT,\n',
      '        "T_logit": 2, "T_cwd": TD.T_CWD, "cwd_C": TD.CWD_C_FEAT,\n', kcase("d2_T_unchanged"), "d2_T_unchanged"),
    M("MK4", "every ramp at 1 from step 1", QKD, "                ramp_by_term = {t: r(step) for t, r in ramps.items()}\n",
      "                ramp_by_term = {t: 1.0 for t in ramps}\n", kcase("d2_distill_zero_at_step1"),
      "d2_distill_zero_at_step1"),
    M("MK5", "the ramp over one step less than the epoch", QKD, "    ramp_iters = steps_per_epoch ",
      "    ramp_iters = steps_per_epoch - 1 ", kcase("d2_ramp_335_unit"), "d2_ramp_335_unit"),
    M("MK6", "a trigger record with triggered false accepted", TRG, '    if doc.get("triggered") is not True:\n',
      '    if doc.get("triggered") is None:\n', kcase("gate_refuses_trigger_false"), "gate_refuses_trigger_false"),
    M("MK7", "the trigger record read from the worktree, not HEAD", TRG,
      "    at_head = Q._git_head_bytes(Path(repo_root), rel.as_posix())\n", "    at_head = data\n",
      kcase("gate_refuses_trigger_changed_since_head"), "gate_refuses_trigger_changed_since_head"),
    M("MK8", "the trigger's E3 binding dropped", TRG,
      '    if e3.get("checkpoint_sha256") != parent.get("checkpoint_sha256") or e3.get("run_id") != parent.get("run_id"):\n',
      "    if False:\n", kcase("gate_refuses_trigger_e3_mismatch"), "gate_refuses_trigger_e3_mismatch"),
    M("MK9", "the teacher's NMF stream begun again every epoch", QKD, "            if epoch == Q.BN_FREEZE_EPOCH + 1:\n",
      "            teacher.begin_nmf_stream(NMF_POLICY, M4_NMF_SEED)\n            if epoch == Q.BN_FREEZE_EPOCH + 1:\n",
      kcase("d3_stream_begun_once"), "d3_stream_begun_once"),
    M("MK10", "two teacher forwards per step", QKD,
      "                    teacher_out = teacher(img_d)             # one frozen forward per step, after the student's\n",
      "                    teacher(img_d)\n                    teacher_out = teacher(img_d)\n",
      kcase("d3_one_teacher_forward_per_step"), "d3_one_teacher_forward_per_step"),
    M("MK11", "a teacher forward inside VAL", QKD, "            t_val = time.time()\n",
      "            teacher(next(iter(val_loader))[0].to(dev))\n            t_val = time.time()\n",
      kcase("d3_no_teacher_forward_in_val"), "d3_no_teacher_forward_in_val"),
    M("MK12", "the projection group's optimizer state loaded", QKD,
      "    opt_rec = kd_optimizer_record(optimizer, prepared, projection)\n",
      '    optimizer.state[projection.weight]["momentum_buffer"] = torch.zeros_like(projection.weight)\n'
      "    opt_rec = kd_optimizer_record(optimizer, prepared, projection)\n",
      kcase("d2_projection_group_fresh"), "d2_projection_group_fresh"),
    M("MK13", "the projection given a qconfig", QKD, "    proj_fq = [n for n, m in projection.named_modules()",
      '    projection.qconfig = "mutant"\n    proj_fq = [n for n, m in projection.named_modules()',
      kcase("d2_projection_not_fake_quantized"), "d2_projection_not_fake_quantized"),
    M("MK14", "projection.pt's checkpoint pairing unchecked", QKD, '    if doc.get("checkpoint") != parent["checkpoint"]:\n',
      '    if doc.get("checkpoint") is None:\n', kcase("gate_refuses_projection_unpaired_checkpoint"),
      "gate_refuses_projection_unpaired_checkpoint"),
    M("MK15", "the projection built by default init (a global RNG draw)", QKD,
      "    projection = torch.nn.utils.skip_init(nn.Conv2d, STUDENT_C5_CH, TEACHER_STRIDE16_CH, 1, bias=False)\n",
      "    projection = nn.Conv2d(STUDENT_C5_CH, TEACHER_STRIDE16_CH, 1, bias=False)\n",
      kcase("d2_projection_no_global_rng_draw"), "d2_projection_no_global_rng_draw"),
    M("MK16", "the clip's norm over the student only", QKD,
      "    return torch.linalg.vector_norm(torch.stack([n_s, n_p.to(n_s.device)]), 2.0), n_s, n_p\n",
      "    return n_s, n_s, n_p\n", kcase("clip_includes_projection"), "clip_includes_projection"),
    M("MK17", "one joint clip_grad_norm_ over both groups", QKD,
      "                total, n_s, n_p = grad_norms(student_params, projection_params)\n",
      "                total = torch.nn.utils.clip_grad_norm_(student_params + projection_params, max_norm=1e30)\n"
      "                n_s, n_p = group_norm(student_params), group_norm(projection_params)\n",
      kcase("clip_structure_spy"), "clip_structure_spy"),
    M("MK18", "a non-finite loss continued past, as E6 does", QKD, "                if not math.isfinite(loss_v):\n",
      "                if False:\n", kcase("abort_nonfinite_loss_before_backward"), "abort_nonfinite_loss_before_backward"),
    MM("MK19", "the non-finite grad_norm abort moved after optimizer.step", QKD,
       [("                if not math.isfinite(norms[0]):\n", "                if False:\n"),
        ("                clipped = bool(norms[0] > clip_norm)\n",
         "                if not math.isfinite(norms[0]):\n"
         '                    abort(nonfinite_rule(step), quantity="grad_norm", detail={"grad_norm": norms[0]})\n'
         "                clipped = bool(norms[0] > clip_norm)\n")],
       kcase("abort_nonfinite_grad_before_step"), "abort_nonfinite_grad_before_step"),
    M("MK20", "the projection written into eNN.pt", QKD, "            state = Q.cpu_state(prepared)\n",
      '            state = {**Q.cpu_state(prepared), "cwd_projection.weight": projection.weight.detach().cpu().clone()}\n',
      kcase("d2_projection_never_in_checkpoint"), "d2_projection_never_in_checkpoint"),
    M("MK21", "the logit terms on the upsampled logits", QKD, "head_logits=head_logits, c5=c5",
      "head_logits=logits, c5=c5", kcase("kd_inputs_are_fake_quantized_taps"), "kd_inputs_are_fake_quantized_taps"),
    M("MK22", "the CWD weights crossed (α to the logit map, β to the feature map)", QKD,
      'alpha=weights["cwd_feat"], beta=weights["cwd_logit"])', 'alpha=weights["cwd_logit"], beta=weights["cwd_feat"])',
      kcase("kd_terms_equal_direct_loss_calls", "kd_addition_order"), "kd_terms_equal_direct_loss_calls",
      "kd_addition_order"),
    M("MK23", "the CWD feature term normalised by C 116 (train_distill, reused at O)", TDS,
      "l_feat = cwd_channelwise_kl(s_feat, t_feat, valid_s16, T=T_CWD, channels_norm=CWD_C_FEAT)",
      "l_feat = cwd_channelwise_kl(s_feat, t_feat, valid_s16, T=T_CWD, channels_norm=116)",
      kcase("kd_scaling_T2_and_C"), "kd_scaling_T2_and_C"),
    M("MK24", "seed 43 accepted", QKD, "    if args.seed != SEED:\n", "    if args.seed not in (SEED, 43):\n",
      kcase("gate_refuses_seed_43"), "gate_refuses_seed_43"),
    M("MK25", "the student image accepted for training", QKD, "    if digest != pd.KD_IMAGE_DIGEST:\n",
      "    if digest not in (pd.KD_IMAGE_DIGEST, Q.IMAGE_DIGEST_OF_RECORD):\n", kcase("gate_refuses_image_not_kd"),
      "gate_refuses_image_not_kd"),
    M("MK26", "the teacher's sha256 not pinned to the teacher of record", QKD,
      "    if expect_sha256 != pd.TEACHER_SHA256:\n", "    if not expect_sha256:\n",
      kcase("gate_refuses_teacher_not_of_record"), "gate_refuses_teacher_not_of_record"),
    M("MK27", "e6-kd routed to src/quant/qat.py main", RUN, "        from .qat_kd import main as qat_kd_main\n",
      "        from .qat import main as qat_kd_main\n", kcase("runner_routes_e6kd_to_qat_kd"),
      "runner_routes_e6kd_to_qat_kd"),
    M("MK28", "E6-KD registered with method qat", STG, '"name": "E6-KD", "method": "qat_kd"',
      '"name": "E6-KD", "method": "qat"', kcase("run_qat_refuses_e6kd"), "run_qat_refuses_e6kd"),
    M("MK29", "the E6-KD rules file differing from AM-23's (tie)", KDRULES, '    "tie": "earliest_epoch",\n',
      '    "tie": "last_epoch",\n', scase("e6kd_rules_equal_am23_but_stages_source_note"),
      "e6kd_rules_equal_am23_but_stages_source_note"),
    M("MK30", "E6-KD's RULES_FILES line dropped (Q2's rules file used)", SEL,
      '    "configs/e6kd_selection_rules.json": {"stages": ["E6-KD"], "qat_clip": False},\n', "",
      scase("e6kd_rules_sha_recorded"), "e6kd_rules_sha_recorded"),
    M("MK31", "E6-KD dropped from PROJECTION_FREE_STAGES", SAF, ', "E6-R2", "E6-KD")\n# FP32 checkpoints',
      ', "E6-R2")\n# FP32 checkpoints', scase("e6kd_provenance_needs_projection_flag_false"),
      "e6kd_provenance_needs_projection_flag_false"),
    MM("MK32", "e6kd, weight_factor and trigger_sha256 dropped from run_meta", QKD,
       [('        "e6kd": True, "weight_factor": WEIGHT_FACTOR, "descriptive": True, "holm_family": False,\n',
         '        "descriptive": True, "holm_family": False,\n'),
        ('             "trigger": trigger_meta(gates["trigger"]), "trigger_sha256": gates["trigger"]["sha256"],\n',
         '             "trigger": trigger_meta(gates["trigger"]),\n')],
       kcase("rec_run_meta_kd_keys_exact"), "rec_run_meta_kd_keys_exact"),
    M("MK33", "a failed step-1 check not stopping the run", QKD,
      "                    if failed:\n                        abort(STEP1_RULE,",
      "                    if False:\n                        abort(STEP1_RULE,", kcase("abort_step1_rule"), "abort_step1_rule"),
    M("MK34", "projection_sha256 dropped from the epoch_end row", QKD,
      '                       "projection_sha256": Q.state_digest(projection),\n', "",
      kcase("rec_epoch_end_projection_sha256"), "rec_epoch_end_projection_sha256"),
    # LB16: the two mutations the ruling names
    M("MK35", "the cause routed through train_distill's abort_cause (AM-7's rule)", QKD,
      "abort_cause(rule, input_finite=bool(input_finite), teacher_finite=bool(teacher_finite))",
      "TD.abort_cause(TD.nonfinite_rule(step), bool(input_finite), bool(teacher_finite))",
      kcase("abort_never_student_divergence"), "abort_never_student_divergence"),
    M("MK36", "AM-7(a) written as the generic abort's rule", QKD, 'GENERIC_RULE = "generic_abort (DL-04 (a); DL-51)"',
      'GENERIC_RULE = "AM-7(a)"', kcase("abort_never_student_divergence"), "abort_never_student_divergence"),
    # LB17: a non-finite state with a finite loss and norm is recorded, and the run continues
    M("MK37", "a non-finite state at an epoch end aborting the run", QKD,
      "            if not finite and nonfinite_since is None:\n                _mark_nonfinite(step, fails)\n",
      "            if not finite and nonfinite_since is None:\n                _mark_nonfinite(step, fails)\n"
      '                abort(GENERIC_RULE, quantity="state", detail={})\n',
      kcase("abort_nonfinite_state_continues"), "abort_nonfinite_state_continues"),
    # scripts/e6kd_eval.py check-run-meta: the profile's E6-KD rules
    M("MK38", "check-run-meta's KD image rule dropped (DL-87)", EVL, '    if meta.get("image_digest") != pd.KD_IMAGE_DIGEST:\n',
      "    if False:\n", kcase("profile_stops_on_student_image"), "profile_stops_on_student_image"),
    M("MK39", "check-run-meta's teacher provenance rule dropped (DL-88)", EVL,
      "    if wrong_t or sorted(prov) != sorted(rules) or args.teacher_ckpt_sha256 != pd.TEACHER_SHA256:\n",
      "    if False:\n", kcase("profile_stops_on_teacher_provenance"), "profile_stops_on_teacher_provenance"),
    M("MK40", "check-run-meta's two-group optimizer rule dropped", EVL, "    if KD.kd_optimizer_error(opt):\n",
      "    if False:\n", kcase("profile_stops_on_one_optimizer_group"), "profile_stops_on_one_optimizer_group"),
    M("MK41", "check-run-meta's E6 parent binding dropped", EVL,
      "        Q.e6_parent_binding(parent, args.seed, lam, alpha)\n", "        pass\n",
      kcase("profile_stops_on_parent_not_alpha_winner"), "profile_stops_on_parent_not_alpha_winner"),
    M("MK42", "check-run-meta's E3 weights compared by type, not value", EVL, "float(got_e3w[t]) == float(e3w[t])",
      "got_e3w[t] == e3w[t] and type(got_e3w[t]) is type(e3w[t])", kcase("profile_e3_weights_by_value"),
      "profile_e3_weights_by_value"),
    M("MK43", "check-run-meta's trigger binding rule dropped", EVL,
      '        if not _same(meta.get("trigger_binding"), want_bound):\n', "        if False:\n",
      kcase("profile_stops_on_trigger_binding"), "profile_stops_on_trigger_binding"),
    # DL-24's fixes (after C10): each killed by its new case
    M("MK44", "the run_abort record's readings unguarded on the exception path (DL-24 T1)", QKD,
      "        try:\n            return fn()\n"
      "        except Exception:                                    "
      "# noqa: BLE001 -- a device that cannot answer: None\n"
      "            if not guarded:\n                raise\n            return None\n", "        return fn()\n",
      kcase("abort_exception_record_survives_failed_readings"), "abort_exception_record_survives_failed_readings"),
    M("MK45", "a generic abort's readings guarded too, a failed reading read as non-finite (DL-24 T1)", QKD,
      "        guarded = rule == EXCEPTION_RULE\n", "        guarded = True\n",
      kcase("abort_generic_reading_failure_is_exception"), "abort_generic_reading_failure_is_exception"),
    M("MK46", "a step or VAL cap of 0 accepted in a real run (DL-24 T2)", QKD,
      "or max_steps is not None\n                           or max_val_batches is not None):",
      "or max_steps\n                           or max_val_batches):",
      kcase("run_refuses_zero_step_cap_in_real", "run_refuses_zero_val_cap_in_real"),
      "run_refuses_zero_step_cap_in_real", "run_refuses_zero_val_cap_in_real"),
    M("MK47", "E6KD_CODE_PATHS without src/eval/protocols.py (DL-24 G1)", QKD,
      '    "src/eval/artifacts.py", "src/eval/protocols.py",\n', '    "src/eval/artifacts.py",\n',
      kcase("code_paths_cover_the_launch_imports"), "code_paths_cover_the_launch_imports"),
    M("MK48", "the teacher's expanded path not name-checked (DL-24 G3)", QKD,
      "    if path and (Q.names_test(path) or Q.names_test(Path(path).expanduser())):\n",
      "    if path and Q.names_test(path):\n",
      kcase("sl1_forbidden_expanded_teacher_refused_before_any_probe"),
      "sl1_forbidden_expanded_teacher_refused_before_any_probe"),
    M("MK49", "check-run-meta's expanded teacher path not name-checked (DL-24 G3)", EVL,
      "    if Q.names_test(args.teacher_ckpt) or Q.names_test(Path(args.teacher_ckpt).expanduser()):\n",
      "    if Q.names_test(args.teacher_ckpt):\n",
      kcase("sl1_forbidden_expanded_paths_refused_by_check_run_meta"),
      "sl1_forbidden_expanded_paths_refused_by_check_run_meta"),
    M("MK50", "E-31: a relative teacher path accepted (DL-24 G2)", QKD, "not os.path.isabs(path) or ", "",
      kcase("gate_refuses_teacher_ckpt_relative"), "gate_refuses_teacher_ckpt_relative"),
    M("MK51", "E-31: a teacher path inside the repository accepted (DL-24 G2)", QKD, " or pd._under(p, REPO)\n", "\n",
      kcase("gate_refuses_teacher_ckpt_in_repo"), "gate_refuses_teacher_ckpt_in_repo"),
    M("MK52", "E-31: the given file name unchecked (DL-24 G2)", QKD, "            or p.name != pd.TEACHER_FILE or ",
      "            or ", kcase("gate_refuses_teacher_ckpt_given_name"), "gate_refuses_teacher_ckpt_given_name"),
    M("MK53", "E-31: the resolved file name unchecked (DL-24 G2)", QKD, " or resolved.name != pd.TEACHER_FILE):", "):",
      kcase("gate_refuses_teacher_ckpt_resolved_name"), "gate_refuses_teacher_ckpt_resolved_name"),
    M("MK54", "gate 14 takes the first row that names run_meta, whatever its event (DL-24 T3)", QKD,
      '        if row.get("event") == "run_meta":\n            rows.append(row)\n', "        rows.append(row)\n",
      kcase("gate_e3_record_reads_the_run_meta_row"), "gate_e3_record_reads_the_run_meta_row"),
    M("MK55", "gate 14 parses a torn line that names run_meta (DL-24 T3)", QKD,
      "        try:\n            row = json.loads(ln)\n        except json.JSONDecodeError:\n            continue\n",
      "        row = json.loads(ln)\n",
      kcase("gate_e3_record_reads_the_run_meta_row"), "gate_e3_record_reads_the_run_meta_row"),
    M("MK56", "gate 3 takes any --num-workers of at least 1 (DL-24 G4)", QKD,
      "    if args.num_workers != preflight_distill().NUM_WORKERS:     "
      "# the value check-run-meta requires (DL-24 G4)\n",
      "    if args.num_workers is None or args.num_workers < 1:\n",
      kcase("gate_refuses_num_workers_8"), "gate_refuses_num_workers_8"),
    M("MK57", "trigger_binding without the clip_selection.json comparison (DL-24 SM3)", TRG,
      ' or e6.get("clip_selection_sha256") != clip_selection_sha256:', ":",
      kcase("gate_refuses_trigger_clip_selection_mismatch"), "gate_refuses_trigger_clip_selection_mismatch"),
]

# the refusal tables: every refusal, STOP and incomplete site of each module, each with the case that reaches it
SCOPE = {TRG: None, QKD: None, EVL: None}                        # every function
SUMMARY_CASES = ("refuses_e3_summary_not_val", "refuses_e6_summary_not_val", "refuses_e3_summary_not_fp32",
                 "refuses_e6_summary_not_int8_qat_qnnpack", "refuses_summary_not_cpu", "refuses_summary_not_canvas",
                 "refuses_summary_batch_not_1", "refuses_rows_not_846_for_record",
                 "refuses_e6_summary_not_the_winner_artifact", "refuses_e3_summary_not_the_checkpoint",
                 "refuses_summary_not_the_student_image", "refuses_summary_governed_paths_dirty",
                 "refuses_summary_not_provisional", "refuses_summary_workers_not_0",
                 "refuses_summary_run_id_not_the_trigger_run", "refuses_summary_stage_mismatch",
                 "refuses_summary_role_not_student", "refuses_summary_random_init",
                 "refuses_e3_summary_backend_not_null", "refuses_e6_summary_not_int8_qat",
                 "refuses_summary_gt_support_sum", "refuses_summaries_scored_at_another_commit")
SAME_CASES = ("refuses_summaries_not_same_code_path", "refuses_summaries_differ_in_metric_code",
              "refuses_summaries_differ_in_split_manifest", "refuses_summaries_differ_in_class_map",
              "refuses_summaries_differ_in_torch", "refuses_summaries_differ_in_metric_protocol")


def K(func: str, code: str, *checks: str, k: int = 0) -> tuple:
    return H.K(TRG, func, code, case(*checks), *checks, k=k)


KILLS = dict([
    # steps 1-3: the names, the sha256 flags, the trigger eval dir and --out
    K("check_inputs", "forbidden_path", "g1_forbidden_names_refused_before_any_probe"),
    K("check_inputs", "sha256_format", "refuses_sha256_format"),
    K("check_inputs", "host_label", "refuses_host_label_empty"),
    K("check_inputs", "image_not_of_record", "refuses_image_not_of_record"),
    K("check_inputs", "trigger_eval_dir_not_outside_repo", "refuses_trigger_eval_dir_inside_repo"),
    K("check_inputs", "trigger_eval_dir_not_fresh", "refuses_trigger_eval_dir_not_fresh"),
    K("check_inputs", "stop_present", "refuses_stop_file_present"),
    K("check_inputs", "trigger_eval_missing", "refuses_decide_before_score_incomplete"),
    K("check_inputs", "out_not_outside_repo", "refuses_out_inside_repo"),
    K("check_inputs", "out_exists", "refuses_out_exists"),
    # step 5: the E3 parent is the α winner's run and checkpoint
    K("check_inputs", "e3_checkpoint_not_alpha_winner", "refuses_e3_checkpoint_not_alpha_winner_sha"),
    # step 6: the E6 run
    K("check_inputs", "e6_telemetry_sha256_mismatch", "refuses_e6_telemetry_sha256_mismatch"),
    K("check_inputs", "e6_stage_not_e6", "refuses_e6_stage_not_e6"),
    K("check_inputs", "e6_seed_not_42", "refuses_e6_not_seed_42"),
    K("check_inputs", "e6_mode_not_real", "refuses_e6_mode_not_real"),
    K("check_inputs", "e6_parent_not_e3_checkpoint", "refuses_e6_parent_not_the_e3_checkpoint"),
    K("check_inputs", "e6_selections_mismatch", "refuses_e6_selections_mismatch"),
    K("check_inputs", "e6_clip_binding", "refuses_e6_clip_binding"),
    # step 7: E6's selection (block B)
    K("check_inputs", "e6_no_model", "refuses_e6_no_model"),
    K("check_inputs", "e6_selection_missing", "refuses_e6_selection_missing_incomplete"),
    K("check_inputs", "e6_selection_sha256_mismatch", "refuses_e6_selection_sha256_mismatch"),
    K("check_inputs", "e6_selection_format", "refuses_e6_selection_format"),
    K("check_inputs", "e6_selection_differs", "refuses_e6_selection_recompute_differs"),
    # step 8: the data root
    K("check_inputs", "data_root_not_trainval_only", "refuses_data_root_not_trainval"),
    # score
    K("score", "evaluator_failed", "refuses_evaluator_failure_spends_the_dir"),
    # decide
    K("decide", "trigger_eval_format", "refuses_trigger_eval_format"),
    K("decide", "trigger_eval_smoke_inputs", "smoke_inputs_refused_for_record"),
    K("decide", "trigger_eval_mismatch", "refuses_eval_dir_mixed_commit", "refuses_code_changed_since_score"),
    K("decide", "evaluator_command_not_literal", "refuses_evaluator_command_not_literal"),
    K("decide", "scores_dir_contents", "refuses_scores_dir_contents"),
    K("decide", "summary_artifact_invalid", "refuses_summary_artifact_invalid"),
    K("decide", "summary_sha256_mismatch", "refuses_summary_sha256_mismatch"),
    K("decide", "summary_values", *SUMMARY_CASES),
    K("decide", "summaries_not_same_code_path", *SAME_CASES),
    K("decide", "score_invalid", "refuses_score_invalid"),
    K("decide", "e6_rescore_differs", "refuses_e6_rescore_differs"),
    # B-2's launch gate
    K("read_trigger_record", "trigger_missing", "refuses_trigger_flag_missing"),
    K("read_trigger_record", "trigger_sha256_format", "refuses_trigger_sha256_format"),
    K("read_trigger_record", "trigger_path_not_repo_relative", "refuses_trigger_path_not_repo_relative"),
    K("read_trigger_record", "trigger_forbidden_path", "refuses_trigger_forbidden_path"),
    K("read_trigger_record", "trigger_missing", "refuses_trigger_file_missing", k=1),
    K("read_trigger_record", "trigger_not_tracked", "refuses_trigger_untracked"),
    K("read_trigger_record", "trigger_changed_since_head", "refuses_trigger_changed_since_head"),
    K("read_trigger_record", "trigger_sha256_mismatch", "refuses_trigger_sha256_mismatch"),
    K("read_trigger_record", "trigger_format", "refuses_trigger_not_json"),
    K("read_trigger_record", "trigger_format", "refuses_trigger_wrong_format", k=1),
    K("read_trigger_record", "trigger_smoke_inputs", "refuses_trigger_smoke_inputs"),
    K("read_trigger_record", "trigger_seed", "refuses_trigger_seed"),
    K("read_trigger_record", "trigger_inconsistent", "refuses_trigger_inconsistent"),
    K("read_trigger_record", "trigger_not_triggered", "refuses_trigger_not_triggered"),
    K("trigger_binding", "trigger_e3_mismatch", "refuses_trigger_e3_mismatch"),
    K("trigger_binding", "trigger_clip_mismatch", "refuses_trigger_clip_mismatch"),
    K("trigger_binding", "trigger_selection_mismatch", "refuses_trigger_selection_mismatch"),
])


def KK(rel: str, func: str, code: str, *checks: str, k: int = 0) -> tuple:
    """B-2: a refusal site of src/quant/qat_kd.py or scripts/e6kd_eval.py, killed by the named runner cases."""
    return H.K(rel, func, code, kcase(*checks), *checks, k=k)


KILLS.update(dict([
    # gate 14: E3's terms and weights (e3_record)
    KK(QKD, "e3_record", "parent_run_meta_changed", "gate_refuses_parent_run_meta_changed"),
    KK(QKD, "e3_record", "parent_terms", "gate_refuses_parent_terms"),
    KK(QKD, "e3_record", "parent_weights", "gate_refuses_parent_weights_mismatch", "gate_refuses_parent_T_logit"),
    # gate 15: the projection of record (DL-51; LB14)
    KK(QKD, "projection_of_record", "projection_sha256_format", "gate_refuses_projection_sha256_format"),
    KK(QKD, "projection_of_record", "projection_missing", "gate_refuses_projection_missing"),
    KK(QKD, "projection_of_record", "projection_sha256_mismatch", "gate_refuses_projection_sha256_mismatch"),
    KK(QKD, "projection_of_record", "projection_unreadable", "gate_refuses_projection_unreadable"),
    KK(QKD, "projection_of_record", "projection_format", "gate_refuses_projection_format"),
    KK(QKD, "projection_of_record", "projection_stage", "gate_refuses_projection_stage"),
    KK(QKD, "projection_of_record", "projection_unpaired_checkpoint", "gate_refuses_projection_unpaired_checkpoint"),
    KK(QKD, "projection_of_record", "parent_checkpoint_invalid", "gate_refuses_parent_checkpoint_invalid"),
    KK(QKD, "projection_of_record", "projection_unpaired_iter", "gate_refuses_projection_unpaired_iter"),
    KK(QKD, "projection_of_record", "projection_shape", "gate_refuses_projection_shape"),
    # gate 16: the teacher of record (DL-88)
    KK(QKD, "teacher_of_record", "teacher_forbidden_path", "sl1_forbidden_teacher_refused_before_any_probe"),
    KK(QKD, "teacher_of_record", "teacher_sha256_format", "gate_refuses_teacher_sha256_format"),
    KK(QKD, "teacher_of_record", "teacher_not_of_record", "gate_refuses_teacher_not_of_record"),
    KK(QKD, "teacher_of_record", "teacher_ckpt_path", "gate_refuses_teacher_ckpt_relative",
       "gate_refuses_teacher_ckpt_in_repo", "gate_refuses_teacher_ckpt_given_name",
       "gate_refuses_teacher_ckpt_resolved_name"),
    KK(QKD, "teacher_of_record", "teacher_missing", "gate_refuses_teacher_missing"),
    KK(QKD, "teacher_of_record", "teacher_sha256_mismatch", "gate_refuses_teacher_sha256_mismatch"),
    KK(QKD, "teacher_of_record", "teacher_state_dict_mismatch", "gate_refuses_teacher_state_dict_mismatch"),
    KK(QKD, "teacher_of_record", "teacher_stack_missing", "gate_refuses_teacher_stack_missing"),
    KK(QKD, "teacher_of_record", "teacher_identity", "gate_refuses_teacher_identity"),
    # the trainer's own refusals and STOPs (nothing written before any of them)
    KK(QKD, "run_qat_kd", "stage_not_qat_kd", "run_refuses_stage_not_qat_kd"),
    KK(QKD, "run_qat_kd", "mode_invalid", "run_refuses_mode"),
    KK(QKD, "run_qat_kd", "real_run_overrides", "run_refuses_loaders_in_real", "run_refuses_hooks_in_real",
       "run_refuses_injection_in_real", "run_refuses_step_cap_in_real", "run_refuses_val_cap_in_real",
       "run_refuses_zero_step_cap_in_real", "run_refuses_zero_val_cap_in_real"),
    KK(QKD, "run_qat_kd", "clip_source", "run_refuses_clip_source_in_real"),
    KK(QKD, "run_qat_kd", "config_pins", "run_refuses_config_pins"),
    KK(QKD, "run_qat_kd", "grad_clip_norm_invalid", "run_refuses_bad_clip"),
    KK(QKD, "run_qat_kd", "weights_invalid", "run_refuses_weights_invalid"),
    KK(QKD, "run_qat_kd", "weights_not_half", "run_refuses_weights_not_half_in_real"),
    KK(QKD, "run_qat_kd", "teacher_not_frozen", "run_refuses_teacher_not_frozen"),
    KK(QKD, "run_qat_kd", "projection_not_of_record", "run_refuses_projection_not_of_record"),
    KK(QKD, "run_qat_kd", "cuda_required", "run_refuses_cpu_in_real"),
    KK(QKD, "run_qat_kd", "cuda_initialized_before_seed", "run_refuses_cuda_initialised_in_real"),
    KK(QKD, "run_qat_kd", "out_dir_not_empty", "run_refuses_nonempty_out_dir"),
    KK(QKD, "run_qat_kd", "steps_per_epoch", "run_refuses_steps_per_epoch_in_real"),
    KK(QKD, "run_qat_kd", "loader_of_record", "run_refuses_loader_of_record_in_real"),
    KK(QKD, "run_qat_kd", "empty_loader", "run_refuses_empty_loader"),
    KK(QKD, "run_qat_kd", "backend_unavailable", "run_refuses_backend_unavailable"),
    KK(QKD, "run_qat_kd", "unfused_batchnorm", "run_refuses_unfused_batchnorm_in_real"),
    KK(QKD, "run_qat_kd", "teacher_nmf_missing", "run_refuses_teacher_nmf_missing_in_real"),
    KK(QKD, "run_qat_kd", "optimizer_of_record", "run_stops_optimizer_of_record"),
    KK(QKD, "run_qat_kd", "val_bracket_changed_state", "run_stops_val_bracket_changed_state"),
    # gates 1-11 (real_run_gates' own; gates 4-6, 12 and 13 are src/quant/qat.py's and B-1's, killed in their harnesses)
    KK(QKD, "real_run_gates", "real_run_flag", "gate_refuses_without_real_run"),
    KK(QKD, "real_run_gates", "confirm_real_run_flag", "gate_refuses_without_confirm"),
    KK(QKD, "real_run_gates", "config_pins", "gate_refuses_config_pins"),
    KK(QKD, "real_run_gates", "expect_head_format", "gate_refuses_head_format"),
    KK(QKD, "real_run_gates", "seed_not_42", "gate_refuses_seed_43", "gate_refuses_seed_44"),
    KK(QKD, "real_run_gates", "num_workers", "gate_refuses_num_workers_0", "gate_refuses_num_workers_8"),
    KK(QKD, "real_run_gates", "out_dir", "gate_refuses_out_dir_in_repo", "gate_refuses_out_dir_not_fresh",
       "sl1_forbidden_out_dir_refused_before_any_probe"),
    KK(QKD, "real_run_gates", "cuda_required", "gate_refuses_cpu", "gate_refuses_device_flag_cpu"),
    KK(QKD, "real_run_gates", "cuda_initialized_before_seed", "gate_refuses_cuda_initialised"),
    KK(QKD, "real_run_gates", "tf32_not_default", "gate_refuses_tf32_not_default"),
    KK(QKD, "real_run_gates", "backend_unavailable", "gate_refuses_backend_unavailable"),
    KK(QKD, "real_run_gates", "image_not_kd", "gate_refuses_image_not_kd", "gate_refuses_image_digest_unset"),
    KK(QKD, "real_run_gates", "fingerprint_mismatch", "gate_refuses_fingerprint_mismatch",
       "gate_refuses_fingerprint_child_failed"),
    KK(QKD, "real_run_gates", "data_root_not_trainval_only", "gate_refuses_data_root_not_trainval_only"),
    KK(QKD, "real_run_gates", "expect_head_mismatch", "gate_refuses_head_mismatch"),
    KK(QKD, "real_run_gates", "code_not_clean", "gate_refuses_code_not_clean"),
    # scripts/e6kd_eval.py check-run-meta
    KK(EVL, "cmd_check_run_meta", "run_dir_forbidden_path", "sl1_forbidden_paths_refused_by_check_run_meta"),
    KK(EVL, "cmd_check_run_meta", "teacher_forbidden_path", "sl1_forbidden_paths_refused_by_check_run_meta"),
    KK(EVL, "cmd_check_run_meta", "telemetry_missing", "profile_run_without_telemetry_exit_3"),
    KK(EVL, "cmd_check_run_meta", "run_meta_rows", "profile_first_row_not_json"),
    KK(EVL, "cmd_check_run_meta", "run_meta_rows", "profile_first_row_not_run_meta", k=1),
    KK(EVL, "cmd_check_run_meta", "run_meta_profile", "profile_mismatch_stops_exit_1"),
]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated job ids (MT1, ..., or file:function:code#k)")
    ap.add_argument("--no-refusals", action="store_true", help="MT1-MT19 and MK1-MK57 only")
    ap.add_argument("--workers", type=int, default=2, help="shadows edited and run in parallel")
    ap.add_argument("--keep-shadow", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="Q2's harness self-check only (ruling Q2-F/2's job handling): no shadow and no smoke")
    args = ap.parse_args()
    H.install_stop_handlers()                        # first: the jobs run in sessions of their own
    H.SHADOW_PATHS, H.MUTATIONS, H.SCOPE, H.KILLS = SHADOW_PATHS, MUTATIONS, SCOPE, KILLS
    t_all = time.time()
    H.self_check()
    measured = {name for name, _ok, _detail in H.results}
    H.check("self_check_measured", set(H.SELF_CHECKS) <= measured,
            f"{len(H.SELF_CHECKS)} checks" if set(H.SELF_CHECKS) <= measured
            else f"not measured: {sorted(set(H.SELF_CHECKS) - measured)}")
    if args.self_check:
        return H.finish(t_all)
    root = H.safe_tmpdir("smoke_e6kd_mutations_")
    shadows = [root / f"shadow{i}" for i in range(max(1, args.workers))]
    H.THREADS["value"] = str(max(1, (os.cpu_count() or 1) // len(shadows)))
    heads = {H.build_shadow(sh) for sh in shadows}
    if len(heads) != 1:
        raise SystemExit(f"RESULT: ERROR the shadows' commits differ: {sorted(heads)}")
    head = f"commit {heads.pop()[:12]}"
    all_jobs = H.jobs_of(shadows[0], refusals=not args.no_refusals)
    try:
        jobs = H.select_jobs(all_jobs, args.only)
    except SystemExit:
        if not args.keep_shadow:
            shutil.rmtree(root, ignore_errors=True)
        raise
    print("=" * 78)
    print("E6-KD MUTATION SMOKE (lane L-AM3, B-1 and B-2) -- MT1-MT19, MK1-MK57 and the refusal tables, each killed "
          "by a named check")
    print(f"{len(jobs)} edits | {len(SHADOW_PATHS)} shadow files x {len(shadows)} at {head} | temp {root}")
    print("=" * 78)
    H.run_jobs(jobs, shadows, root / "cache")
    verdicts = {name for name, _ok, _detail in H.results}
    missing = [j["id"] for j in jobs if f"{j['id']}_killed" not in verdicts and f"{j['id']}_baseline" not in verdicts]
    H.check("every_job_has_a_verdict", not missing,
            f"no verdict for {missing[:5]}" if missing else f"{len(jobs)} jobs, each killed or with its baseline failing")
    if not args.keep_shadow:
        shutil.rmtree(root, ignore_errors=True)
    return H.finish(t_all)


if __name__ == "__main__":
    raise SystemExit(main())
