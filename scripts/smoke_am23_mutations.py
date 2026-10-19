#!/usr/bin/env python3
"""Smoke: the mutation harness of L-AM23Q B-2, AM-23's stages and bindings (the Phase A plan §4.4, as the B-2 GO
amends it: every new refusal code has a case reached with every earlier gate passing and a mutation a named case
kills; AQ15 drops §4.4's threshold and void edits, which are B-1's parse_record's).

Each code mutation rewrites one function from its own source: exact text replacements, each occurring exactly once in
that function, compiled into the module's namespace and restored afterwards (scripts/smoke_teacher_diag_mutations.py's
mutate, :56-81, as scripts/smoke_e6x_condition_mutations.py copies it). Each data mutation replaces one module-level
table (a stage tuple, a stage row, RULES_FILES) for its cases and restores it. The killing cases are cases of
scripts/smoke_am23_quant.py, run in this process on that smoke's synthetic fixtures (built here, in this harness's own
work dir). Three checks per mutation:
  <id>_anchor   every replacement occurs exactly once in the function's source (data: the table differs)
  <id>_control  each killing case passes unmutated
  <id>_killed   each killing case fails (or raises) under the mutation
and two more: every_code_killed (every new refusal code of B-2 is the code of a killing case) and every_case_named
(every killing case exists). Each kill is printed with its kind: accepted (the mutant accepted what the case refuses),
code (another code), text (the same code, another site's message), error (an exception) or result (a non-refusal case
saw another result). Last, the smoke's three hygiene cases on this run.

    python -B scripts/smoke_am23_mutations.py [--work-dir DIR] [--only M001,M002]
"""
from __future__ import annotations

import __future__ as _future
import argparse
import contextlib
import copy
import importlib
import inspect
import os
import re
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True    # no __pycache__ in the repository, with or without -B

S = importlib.import_module("scripts.smoke_am23_quant")
RUNNER = importlib.import_module("src.quant.runner")
SE = importlib.import_module("scripts.select_qat_epoch")
OWNERS = {"ST": S.ST, "CK": S.CK, "Q": S.Q, "SEL": S.S, "QEE": S.QEE, "EM": S.EM, "SA": S.SA, "ART": S.ART,
          "RUNNER": RUNNER, "SE": SE}
RESULTS: list = []

DESC_CASES = ("s5_refuses_not_descriptive",)
NOT_ARM = ("s5_refuses_not_descriptive", "s5_refuses_parent_of_e4_e7", "s5_refuses_max_iters",
           "s5_refuses_poly_horizon", "s5_refuses_float_horizon", "s5_refuses_no_target_construction")


def F(owner, func, pairs, cases, what):
    return ("fn", owner, func, pairs, tuple(cases), what)


def D(owner, attr, make, cases, what):
    """A data mutation: owner.attr replaced by make(the current value) for the cases."""
    return ("data", owner, attr, make, tuple(cases), what)


def _without(t, item):
    return tuple(x for x in t if x != item)


RAW = [
    # ---- the stage routing (stages.py)
    F("ST", "load_source_for_stage", [("        return load_student_from_e2(ckpt_path, **kw)",
                                       "        return load_student_from_e3(ckpt_path, **kw)")],
      ["s1_routes_e2_to_e2_loader"], "E2 sources routed to the E3 loader"),
    F("ST", "load_source_for_stage", [('    if st["source_stage"] == "E3":', "    if True:")],
      ["s1_unknown_source_refused"], "an unknown source falls through to the E3 loader"),
    # ---- the E2 loader (checkpoint.py)
    F("CK", "load_e2_source_checkpoint", [("    if not p.is_file():", "    if False:")],
      ["s2_refuses_missing_file"], "the missing-file check removed"),
    F("CK", "load_e2_source_checkpoint", [("    if not isinstance(ckpt, dict):", "    if False:")],
      ["s2_refuses_not_a_dict"], "the dict check removed"),
    F("CK", "load_e2_source_checkpoint", [("    if stage is None:", "    if False:")],
      ["s2_refuses_stage_none"], "the stage-None refusal removed"),
    F("CK", "load_e2_source_checkpoint", [('    if str(stage).upper() != "E2":',
                                           '    if str(stage).upper() not in ("E2", "E3"):')],
      ["s2_refuses_stage_e3"], "an E3 payload accepted as E2"),
    F("CK", "load_e2_source_checkpoint", [('    if str(stage).upper() != "E2":',
                                           '    if str(stage).upper() not in ("E1", "E2"):')],
      ["s2_refuses_stage_e1"], "an E1 payload accepted as E2"),
    F("CK", "load_e2_source_checkpoint", [('    if "model_state_dict" not in ckpt:', "    if False:")],
      ["s2_refuses_no_model_state_dict"], "the model_state_dict check removed"),
    F("CK", "load_e2_source_checkpoint",
      [("    if CWD_PROJECTION_KEY in ckpt and ckpt[CWD_PROJECTION_KEY] is not None:", "    if False:")],
      ["s2_refuses_cwd_projection_state_dict"], "the cwd_projection_state_dict check removed"),
    F("CK", "load_e2_source_checkpoint", [("        state = deployment_student_state(ckpt)",
                                           '        state = dict(ckpt["model_state_dict"])')],
      ["s2_refuses_projection_key_in_student"], "the projection assert (AM-4a item 5) skipped"),
    F("CK", "load_e2_source_checkpoint",
      [("    if not any(torch.is_tensor(v) for v in state.values()):", "    if False:")],
      ["s2_refuses_no_tensors"], "the tensors check removed"),
    F("CK", "load_e2_source_checkpoint", [("    if teacher_keys:", "    if False:")],
      ["s2_refuses_teacher_keys"], "the teacher-key check removed"),
    F("CK", "load_e2_source_checkpoint", [("    if quant_keys:", "    if False:")],
      ["s2_refuses_quant_keys"], "the quantization-key check removed"),
    F("CK", "load_e2_source_checkpoint", [("    if non_float:", "    if False:")],
      ["s2_refuses_quantized_dtype"], "the quantized-dtype check removed"),
    F("CK", "load_e2_source_checkpoint", [('    if ckpt.get("num_classes") is not None', "    if False")],
      ["s2_refuses_num_classes"], "the num_classes check removed"),
    F("CK", "load_e2_source_checkpoint", [('    if ckpt.get("terms") != E2_TERMS:', "    if False:")],
      ["s2_refuses_terms_not_e2", "s2_refuses_terms_missing"], "the E2 terms check removed"),
    F("CK", "load_student_from_e2", [('    return _build_strict("E2", deployment_student_state(ckpt), path), ckpt',
                                      '    return _build_strict("E3", deployment_student_state(ckpt), path), ckpt')],
      ["s2_refuses_structure"], "the strict build labelled E3"),
    # ---- resolve_parent's E2 branch (qat.py)
    F("Q", "resolve_parent", [('        key = stage["source_stage"].lower()', '        key = "e3"')],
      ["s3_e2_parent_accepted", "s3_e7_e2_takes_the_e2_branch"], "an E2 parent read from e3_ files"),
    F("Q", "resolve_parent",
      [('    if stage["source_stage"] in ("E2", "E3"):', '    if stage["source_stage"] == "E3":')],
      ["s3_e2_parent_accepted", "s3_run_abort_refused", "s3_no_run_end_refused", "s3_checks_failed_refused"],
      "an E2 parent checked as an E1 run"),
    F("Q", "resolve_parent", [('        if meta.get("stage") != stage["source_stage"]:',
                               '        if meta.get("stage") not in ("E2", "E3"):')],
      ["s3_stage_e3_row_refused"], "an E3 run_meta row accepted for an E2 stage"),
    F("Q", "load_source", [('    if stage["source_stage"] in ("E2", "E3") and any(',
                            '    if stage["source_stage"] == "E3" and any(')],
      ["s6_load_source_refuses_e2_projection_keys"], "the projection-key check skipped for E2 sources"),
    # ---- E6-E2's binding
    F("Q", "e2_selection", [("    if args.alpha_selection is not None or args.alpha_selection_sha256 is not None:",
                             "    if False:")],
      ["s6_e6_e2_refuses_alpha_flags"], "E6-E2 takes α flags"),
    F("Q", "lambda_selection_input", [('    if sha256_bytes(data) != lam["sha256"]:', "    if False:")],
      ["s4_refuses_selection_changed_since_read"], "the re-read selection not compared with the one read"),
    F("Q", "lambda_selection_input", [("    if len(entries) != 1:", "    if not entries:")],
      ["s4_refuses_input_twice"], "two inputs entries for the run accepted"),
    F("Q", "lambda_selection_input", [("    if len(entries) != 1:", "    if False:")],
      ["s4_refuses_input_missing"], "a missing inputs entry not refused"),
    F("Q", "lambda_selection_input", [('               and Path(str(e.get("run_dir", ""))).name == run_id]', "]")],
      ["s4_binding_accepts"], "the inputs entry not matched by run directory"),
    F("Q", "e2_parent_binding", [('    if parent.get("stage") != "E2":', "    if False:")],
      ["s4_refuses_stage"], "the parent's stage not checked"),
    F("Q", "e2_parent_binding", [('    if seed != AM23_SEED or parent.get("seed") != AM23_SEED:',
                                  '    if parent.get("seed") != AM23_SEED:')],
      ["s4_refuses_seed_43"], "--seed not checked (seed 42 only)"),
    F("Q", "e2_parent_binding", [('    if seed != AM23_SEED or parent.get("seed") != AM23_SEED:',
                                  "    if seed != AM23_SEED:")],
      ["s4_refuses_s43_parent_dict_at_seed_42"], "the parent's seed not checked"),
    F("Q", "e2_parent_binding", [('    if parent.get("run_id") != win.get("run_id"):', "    if False:")],
      ["s4_refuses_160k_directory"], "the λ winner's run not required"),
    F("Q", "e2_parent_binding", [('    if parent.get("lambda_logit") != win.get("lambda"):', "    if False:")],
      ["s4_refuses_lambda"], "the λ not compared"),
    F("Q", "e2_parent_binding",
      [('    if parent.get("checkpoint_sha256") != win.get("ckpt_sha256"):', "    if False:")],
      ["s4_refuses_checkpoint_not_selected"], "the selected checkpoint not compared"),
    F("Q", "e2_parent_binding", [('("best_json_sha256", "run_meta_sha256", "checkpoint")',
                                  '("run_meta_sha256", "checkpoint")')],
      ["s4_refuses_best_json_not_input"], "best.json not bound to the selection's inputs entry"),
    F("Q", "e2_parent_binding", [('("best_json_sha256", "run_meta_sha256", "checkpoint")',
                                  '("best_json_sha256", "checkpoint")')],
      ["s4_refuses_run_meta_not_input"], "run_meta not bound to the selection's inputs entry"),
    F("Q", "e2_parent_binding", [('("best_json_sha256", "run_meta_sha256", "checkpoint")',
                                  '("best_json_sha256", "run_meta_sha256")')],
      ["s4_refuses_checkpoint_name_not_input"], "the checkpoint name not bound to the inputs entry"),
    F("Q", "e2_parent_binding", [('    if (Path(parent["run_dir"]) / PROJECTION_FILE).exists():', "    if False:")],
      ["s4_refuses_projection_pt"], "a projection.pt beside the E2 checkpoint accepted"),
    # ---- E6-X's binding
    F("Q", "e6x_parent_binding", [('    if args.seed != AM23_SEED or parent.get("seed") != AM23_SEED:',
                                   '    if parent.get("seed") != AM23_SEED:')],
      ["s5_refuses_seed_43"], "--seed not checked (seed 42 only)"),
    F("Q", "e6x_parent_binding", [('    if meta.get("arm") != want_arm:', "    if False:")],
      ["s5_refuses_e3_r2_run_for_e6_r1", "s5_refuses_alpha_winner_for_e6_r1"], "the parent's arm not checked"),
    F("Q", "e6x_parent_binding", [('("descriptive", meta.get("descriptive") is True)', '("descriptive", True)')],
      ["s5_refuses_not_descriptive"], "descriptive not checked"),
    F("Q", "e6x_parent_binding", [('("parent_of_e4_e7", meta.get("parent_of_e4_e7") is False)',
                                   '("parent_of_e4_e7", True)')],
      ["s5_refuses_parent_of_e4_e7"], "parent_of_e4_e7 not checked"),
    F("Q", "e6x_parent_binding", [('("max_iters", type(meta.get("max_iters")) is int\n', '("max_iters", True\n')],
      ["s5_refuses_float_horizon"], "max_iters' type not checked"),
    F("Q", "e6x_parent_binding", [('                                and meta.get("max_iters") == AM23_HORIZON)',
                                   "                                and True)")],
      ["s5_refuses_max_iters"], "max_iters' value not checked"),
    F("Q", "e6x_parent_binding", [('                                and meta.get("poly_horizon") == AM23_HORIZON)',
                                   "                                and True)")],
      ["s5_refuses_poly_horizon"], "poly_horizon not checked"),
    F("Q", "e6x_parent_binding", [('("target_construction", meta.get("target_construction") is not None)',
                                   '("target_construction", True)')],
      ["s5_refuses_no_target_construction"], "target_construction not checked"),
    F("Q", "e6x_parent_binding",
      [('    if parent.get("lambda_logit") != lam["winner"].get("lambda"):', "    if False:")],
      ["s5_refuses_lambda"], "the λ not compared"),
    F("Q", "e6x_parent_binding", [('    if parent.get("alpha_cwd") != alpha["winner"].get("alpha"):', "    if False:")],
      ["s5_refuses_alpha"], "the α not compared"),
    F("Q", "e6x_parent_binding", [("    if not rel or not want_sha or not dl_id:", "    if not rel or not want_sha:")],
      ["s5_refuses_no_condition_dl"], "--condition-dl not required"),
    F("Q", "e6x_parent_binding", [("    if not rel or not want_sha or not dl_id:", "    if not rel or not dl_id:")],
      ["s5_refuses_no_condition_sha256"],
      "--condition-sha256 not required (B-1's reader refuses its form instead)"),
    F("Q", "e6x_parent_binding",
      [("    if not rel or not want_sha or not dl_id:", "    if not want_sha or not dl_id:")],
      ["s5_refuses_no_condition_record"], "--condition-record not required"),
    F("Q", "e6x_parent_binding", [('    if Path(str(rel)).as_posix() != X.RECORD_REL.format(arm=rec["arm"]):',
                                   "    if False:")],
      ["s5_condition_path_not_of_record"], "the path of record not enforced (AQ15)"),
    F("Q", "e6x_parent_binding", [('    if rec["arm"] != arm:', "    if False:")],
      ["s5_condition_arm_mismatch"], "the record's arm not compared"),
    F("Q", "e6x_parent_binding", [('("run_id", rec["run_id"] == parent["run_id"])', '("run_id", True)')],
      ["s5_condition_other_run_id"], "the record's run_id not compared"),
    F("Q", "e6x_parent_binding", [('("best_json", rec["best_json"]["sha256"] == parent["best_json_sha256"])',
                                   '("best_json", True)')],
      ["s5_condition_other_best_json"], "the record's best.json sha256 not compared"),
    F("Q", "e6x_parent_binding", [('("checkpoint", rec["checkpoint"]["file"] == parent["checkpoint"]\n',
                                   '("checkpoint", True\n')],
      ["s5_condition_other_checkpoint_name"], "the record's checkpoint name not compared"),
    F("Q", "e6x_parent_binding",
      [('                               and rec["checkpoint"]["sha256"] == parent["checkpoint_sha256"])',
        "                               and True)")],
      ["s5_condition_other_checkpoint"], "the record's checkpoint sha256 not compared"),
    F("Q", "e6x_parent_binding", [('("run_meta", rec["run_meta"]["sha256"] == parent["run_meta_sha256"])',
                                   '("run_meta", True)')],
      ["s5_condition_other_run_meta"], "the record's run_meta sha256 not compared"),
    F("Q", "e6x_parent_binding", [('("telemetry", rec["telemetry"]["sha256"] == parent["telemetry_sha256"])',
                                   '("telemetry", True)')],
      ["s5_condition_other_telemetry"], "the record's telemetry sha256 not compared"),
    F("Q", "e6x_parent_binding", [('("value", rec["value"]["repr"] == repr(parent["best_val_miou_all_class"]))',
                                   '("value", True)')],
      ["s5_condition_other_value"], "the record's value not compared with best.json's"),
    F("Q", "e6x_parent_binding", [('    if X.verdict(parent["best_val_miou_all_class"]) != "yes":', "    if False:")],
      ["s5_condition_no"], "a 'no' does not rule E6-X out"),
    F("Q", "e6x_parent_binding", [('    condition_dl_row(dl_id, stage["name"], rec, rec_sha, repo_root)', "    pass")],
      ["s5_dl_row_missing", "s5_dl_row_not_recorded"], "the decision-log row not read"),
    # ---- the decision-log row (AQ3)
    F("Q", "condition_dl_row", [("    if not isinstance(dl_id, str) or not _DL_ID.fullmatch(dl_id):", "    if False:")],
      ["s5_dl_id_malformed"], "--condition-dl's form not checked"),
    F("Q", "condition_dl_row", [("    if data is None:", "    if False:")],
      ["s5_dl_log_untracked"], "an untracked decision log not refused"),
    F("Q", "condition_dl_row", [("    if row is None:", "    if False:")],
      ["s5_dl_row_missing"], "a missing row not refused"),
    F("Q", "condition_dl_row", [('    if status != "RECORDED":', "    if False:")],
      ["s5_dl_row_not_recorded"], "a row not RECORDED accepted"),
    F("Q", "condition_dl_row", [("    need = (rec_sha256, ", "    need = (")],
      ["s5_dl_row_without_record_sha256", "s5_dl_row_of_another_record"], "the record's sha256 not required"),
    F("Q", "condition_dl_row", [('f"{stage_name} condition (AM-23 item 3(b)): yes", ', "")],
      ["s5_dl_row_without_yes"], "': yes' not required"),
    F("Q", "condition_dl_row", [('rec["value"]["repr"],\n', "\n")],
      ["s5_dl_row_without_value"], "the value's repr not required"),
    F("Q", "condition_dl_row", [('            rec["best_json"]["sha256"], ', "            ")],
      ["s5_dl_row_without_best_json_sha256"], "best.json's sha256 not required"),
    F("Q", "condition_dl_row", [(', rec["checkpoint"]["sha256"])', ")")],
      ["s5_dl_row_without_checkpoint_sha256"], "the checkpoint's sha256 not required"),
    # ---- the gates, main and the parsers
    F("Q", "am23_parent_binding", [('    if stage["source_stage"] == "E2":', "    if False:")],
      ["s6_e6_e2_gates_pass"], "E6-E2 bound as an E6-X stage"),
    F("Q", "real_run_gates", [('    if stage["name"] in AM23_STAGES:', "    if False:")],
      ["s6_e6_e2_gates_pass", "s6_e6_r1_gates_pass"], "AM-23 stages bound with E6's binding"),
    F("Q", "real_run_gates", [('    if stage["source_stage"] == "E2":', "    if False:")],
      ["s6_e6_e2_gates_pass"], "E6-E2's λ-only selection not read"),
    F("Q", "main", [('        extra["am23_parent"] = gates["am23_binding"]', "        pass")],
      ["s6_main_extra_e6_e2", "s6_main_extra_e6_r1"], "the binding not on run_meta"),
    F("Q", "main", [('            extra["cwd_projection_loaded"] = False', "            pass")],
      ["s6_main_extra_e6_e2"], "E6-E2's run_meta without the projection flag"),
    F("Q", "build_parser", [('    if stage["name"] in E6X_PARENT_ARM:', "    if True:")],
      ["s6_e5_e6_parsers_take_no_condition_flags", "s6_e6_e2_parser_refuses_condition_flags"],
      "the condition flags on every QAT stage's parser"),
    F("RUNNER", "build_run_provenance", [('    if stage["source_stage"] in ("E2", "E3"):',
                                          '    if stage["source_stage"] == "E3":')],
      ["s9_run_provenance_cwd_flag"], "AQ23 undone: no projection flag for E2 sources"),
    # ---- Block B (AQ24)
    F("SEL", "load_rules", [('("qat_epoch", "stages"): spec["stages"]', '("qat_epoch", "stages"): ["E5", "E6"]')],
      ["s8_am23_rules_load"], "every rules file held to Q2's stages"),
    F("SEL", "load_rules", [('    if spec["qat_clip"]:', "    if True:")],
      ["s8_am23_rules_load"], "every rules file held to the clip block"),
    F("SEL", "load_rules", [('        want[("qat_clip",)] = None', "        pass")],
      ["s8_am23_rules_with_qat_clip_refused"], "a qat_clip block in AM-23's file accepted"),
    F("SEL", "rules_spec",
      [("    return next((spec for rel, spec in RULES_FILES.items() if (REPO / rel).resolve() == p),",
        "    return next((spec for rel, spec in RULES_FILES.items() if False),")],
      ["s8_am23_rules_load"], "every file checked as Q2's"),
    F("SEL", "rules_path_for_stage",
      [("    return next((REPO / rel for rel, spec in RULES_FILES.items() if name in spec[\"stages\"]), "
        "RULES_PATH)", "    return RULES_PATH")],
      ["s8_rules_path_for_stage"], "every stage given Q2's rules"),
    F("SEL", "rules_path_for_run",
      [('        return rules_path_for_stage(meta.get("stage")) if isinstance(meta, dict) else RULES_PATH',
        "        return RULES_PATH")],
      ["s8_rules_path_for_run"], "every run given Q2's rules"),
    F("SEL", "rules_rel_of",
      [('    return next((rel for rel, spec in RULES_FILES.items() if spec["stages"] == stages),',
        "    return next((rel for rel, spec in RULES_FILES.items() if False),")],
      ["s8_am23_rules_load"], "the trace names Q2's file for AM-23's rules"),
    F("SE", "main", [("        rules, rules_sha = S.load_rules(S.rules_path_for_run(args.run_dir))",
                      "        rules, rules_sha = S.load_rules()")],
      ["s9_select_in_process_takes_am23_rules"], "select_qat_epoch.py reads Q2's rules for every run"),
    D("SEL", "RULES_FILES", lambda v: {k: {**s, "stages": ["E5", "E6"]} for k, s in v.items()},
      ["s8_rules_path_for_stage"], "RULES_FILES maps AM-23's stages to no file"),
    # ---- check-run-meta (qee)
    F("QEE", "profile_mismatches", [("    elif stage in Q.AM23_STAGES:", "    elif False:")],
      ["s7_e6_e2_profile_passes", "s7_e6_r1_profile_passes"], "the AM-23 profile branch removed"),
    F("Q", "am23_profile", [('            bad["am23_parent"] = (got, want)', "            pass")],
      ["s7_stops_on_am23_parent_missing", "s7_stops_on_am23_parent_other"], "the recorded binding not compared"),
    F("Q", "am23_profile", [('        bad["e6_selections"] = (meta.get("e6_selections"), None)', "        pass")],
      ["s7_stops_on_e6_selections"], "E6 selections on an AM-23 run accepted"),
    F("Q", "am23_profile",
      [('        bad["cwd_projection_loaded"] = (meta.get("cwd_projection_loaded"), False)', "        pass")],
      ["s7_stops_on_cwd_flag"], "the projection flag not required"),
    F("QEE", "build_parser", [('"E6-E2", "E6-R1", "E6-R2", "e6-e2",\n', '"e6-e2",\n')],
      ["s7_cli_accepts_the_am23_stages"], "check-run-meta's --stage without the upper-case AM-23 names"),
    # ---- evaluation (AQ25; the tables)
    F("EM", "build_parser", [('    p.add_argument("--stage", default="E1", choices=list(STAGES))',
                              '    p.add_argument("--stage", default="E1", choices=list(STAGES)[:11])')],
      ["s10_em_choices_are_the_stages", "s10_em_parses_each"], "em's --stage choices without the four"),
    D("ART", "STAGES", lambda v: _without(v, "E6-R2"), ["s10_stages_appended", "s10_em_choices_are_the_stages"],
      "artifacts.STAGES without E6-R2"),
    D("SA", "STAGE_ARTIFACTS", lambda v: {**copy.deepcopy(v), "E6-E2": {**v["E6-E2"], "source_stage": "E3"}},
      ["s10_stage_artifact_rows", "s9_validate_int8_accepts_e6_e2"], "E6-E2's row derives from E3"),
    D("SA", "PROJECTION_FREE_STAGES", lambda v: _without(v, "E6-E2"),
      ["s10_projection_free_tuple", "s9_e6_e2_provenance_without_cwd_flag_refused"],
      "E6-E2 not projection-free"),
    D("SA", "DESCRIPTIVE_ONLY_STAGES", lambda v: _without(v, "E7-E2"), ["s10_descriptive_only_tuple"],
      "E7-E2 not descriptive-only"),
    # ---- DL-24's fixes (appended, so the earlier ids stay)
    F("Q", "condition_dl_row", [("    missing = [n for n in need if not _whole_token(n, row)]",
                                 "    missing = [n for n in need if n not in row]")],
      ["s5_dl_row_value_with_an_extra_digit", "s5_dl_row_value_with_a_sign",
       "s5_dl_row_sha256_with_an_extra_hex"],
      "AQ3's tokens matched inside a longer number, sign or hex string (DL-24 B1-01)"),
]
MUTATIONS = [(f"M{i:03d}", *m) for i, m in enumerate(RAW, 1)]


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def anchor_counts(owner, name: str, pairs) -> list:
    src = textwrap.dedent(inspect.getsource(getattr(owner, name)))
    return [src.count(old) for old, _ in pairs]


def kill_kind(case_name: str, detail: str) -> str:
    """How a killing case saw its mutant, from the case's own detail line."""
    want = S.CASES[case_name][1]
    if want is None:
        return "result"
    if detail in ("[None]", "accepted"):
        return "accepted"
    if detail.startswith("raised:") or re.match(r"[A-Za-z_.]*(Error|Exception|Exit)\b", detail):
        return "error"
    got = re.search(r"\[([a-z0-9_]+)\]", detail)
    return "text" if got is not None and got.group(1) == want else "code"


@contextlib.contextmanager
def mutate(owner, name: str, *pairs):
    """Replace `owner.name` by its own source with each (old, new) replaced exactly once, compiled into its module's
    namespace; restore it afterwards. (scripts/smoke_teacher_diag_mutations.py :56-81, copied.)"""
    original = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    module = sys.modules[original.__module__]
    src = textwrap.dedent(inspect.getsource(original))
    for old, new in pairs:
        if src.count(old) != 1:
            raise AssertionError(f"mutation anchor occurs {src.count(old)} times in {name}: {old!r}")
        src = src.replace(old, new)
    ns = module.__dict__
    had, prev = name in ns, ns.get(name)
    exec(compile(src, f"<mutant {module.__name__}.{name}>", "exec", flags=_future.annotations.compiler_flag,
                 dont_inherit=True), ns)
    mutant = ns[name]
    if isinstance(owner, type):
        if had:
            ns[name] = prev
        else:
            del ns[name]
    setattr(owner, name, mutant)
    try:
        yield
    finally:
        setattr(owner, name, original)


@contextlib.contextmanager
def replaced(owner, attr: str, make):
    old = getattr(owner, attr)
    setattr(owner, attr, make(old))
    try:
        yield
    finally:
        setattr(owner, attr, old)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated mutation ids")
    a = ap.parse_args(argv)
    ids = [m[0] for m in MUTATIONS]
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else []
    if [x for x in only if x not in ids]:
        print(f"RESULT: ERROR unknown mutation ids: {[x for x in only if x not in ids]}")
        return 2
    work = S.choose_work_dir(a.work_dir)
    work.mkdir(parents=True)
    (work / "scratch").mkdir()
    (work / "relative_base").mkdir()
    tempfile.tempdir = str(work / "scratch")
    os.chdir(work / "relative_base")             # a relative path resolves here, never in the repository
    S.AUDIT_BASES[:] = [str(work), str(S.REPO)]
    sys.addaudithook(S.audit)
    t0 = time.time()
    print("=" * 78)
    print("AM-23 QUANT MUTATIONS (L-AM23Q B-2) -- each mutant against its killing cases")
    print(f"work dir: {work}")
    print("=" * 78, flush=True)
    S.setup(work)
    chosen = [m for m in MUTATIONS if not only or m[0] in only]
    lines, kinds = [], {}
    for mid, kind, owner_key, target, change, cases, what in chosen:
        owner = OWNERS[owner_key]
        unknown = [c for c in cases if c not in S.CASES]
        if kind == "fn":
            counts = anchor_counts(owner, target, change)
            anchored = all(n == 1 for n in counts)
            check(f"{mid}_anchor", anchored, f"{owner_key}.{target}: occurrences {counts}")
        else:
            current = getattr(owner, target)
            anchored = change(current) != current
            check(f"{mid}_anchor", anchored, f"{owner_key}.{target}: the replacement equals the table")
        if not anchored or unknown:
            check(f"{mid}_control", False, f"not run: anchor {anchored}, unknown cases {unknown}")
            check(f"{mid}_killed", False, "not run")
            lines.append(f"  {mid} {owner_key}.{target}: {what} -- NOT RUN")
            continue
        control = [(c, *S.run_case(c)) for c in cases]
        check(f"{mid}_control", all(ok for _, ok, _ in control),
              "; ".join(f"{c} unmutated: {d}" for c, ok, d in control if not ok))
        outcome = []
        for c in cases:
            try:
                ctx = mutate(owner, target, *change) if kind == "fn" else replaced(owner, target, change)
                with ctx:
                    ok_m, detail_m = S.run_case(c)
            except Exception as e:  # noqa: BLE001 -- a mutant that cannot be built is not a kill
                ok_m, detail_m = True, f"mutant not built ({type(e).__name__}: {e})"
            outcome.append((c, ok_m, detail_m))
        check(f"{mid}_killed", all(not ok for _, ok, _ in outcome),
              "; ".join(f"{c} PASSES under {mid}: {d}" for c, ok, d in outcome if ok))
        lines.append(f"  {mid} {owner_key}.{target}: {what}")
        for c, ok_m, detail_m in outcome:
            k = "SURVIVED" if ok_m else kill_kind(c, detail_m)
            kinds[k] = kinds.get(k, 0) + 1
            lines.append(f"        {'KILLED' if not ok_m else 'SURVIVED'} by {c} ({k}): {detail_m[:160]}")
    if not only:
        killed = {S.CASES[c][1] for m in MUTATIONS for c in m[5] if c in S.CASES}
        check("every_code_killed", set(S.NEW_CODES) <= killed,
              f"codes without a killing mutation: {sorted(set(S.NEW_CODES) - killed)}")
        missing = sorted({c for m in MUTATIONS for c in m[5]} - set(S.CASES))
        check("every_case_named", not missing, f"killing cases that do not exist: {missing}")
    for h in ("h_no_forbidden_named_path_created", "h_no_forbidden_path_opened", "h_repository_untouched"):
        ok, detail = S.run_case(h)
        check("hygiene_" + h[2:], ok, detail)
    print("\n[MUTANTS]")
    print("\n".join(lines))
    print("\n[CHECKS]")
    for name, ok, detail in RESULTS:
        print(f"  {name:40}: {'PASS' if ok else 'FAIL'}  {detail[:300] if not ok else ''}".rstrip())
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    killed_n = sum(1 for name, ok, _ in RESULTS if name.startswith("M") and name.endswith("_killed") and ok)
    pairs_n = sum(len(m[5]) for m in chosen)
    print(f"\nmutations killed: {killed_n}/{len(chosen)}; (mutation, case) pairs {pairs_n}, by kind "
          f"{dict(sorted(kinds.items()))} ({round(time.time() - t0, 1)} s)")
    print(f"RESULT: {'PASS' if RESULTS and passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)})")
    return 0 if RESULTS and passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
