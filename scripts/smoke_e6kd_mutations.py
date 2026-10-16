#!/usr/bin/env python3
"""Lane L-AM3, B-1: the mutation smoke of the AM-3 trigger. MT1-MT14 and the refusal table of
src/quant/e6kd_trigger.py, each killed by a named check of scripts/smoke_e6kd_trigger.py.

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
SHADOW_PATHS = (*H.SHADOW_PATHS, TRG, "scripts/e6kd_trigger.py", "scripts/smoke_e6kd_trigger.py")


def case(*names: str) -> tuple:
    return (SMOKE, ("--only", ",".join(names), "--cache-dir", "{cache}/trigger"))


M = H.M
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
]

# the refusal table: every refusal and incomplete site of the module, each with the case that reaches it
SCOPE = {TRG: None}                                              # every function
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated job ids (MT1, ..., or file:function:code#k)")
    ap.add_argument("--no-refusals", action="store_true", help="MT1-MT14 only")
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
    print("E6-KD TRIGGER MUTATION SMOKE (lane L-AM3, B-1) -- MT1-MT14 and the refusal table, each killed by a named "
          "check")
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
