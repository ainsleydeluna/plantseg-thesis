#!/usr/bin/env python3
"""Smoke for the lane's mutation tests (lane TMAN, B-1, C6): every refusal code has a mutation killed by a named
case (the GO's STEP 5 rule), and plan §5's sample mutations are each killed by the case the plan names.

A mutation replaces exact texts (each must occur once) in a scratch copy of the lane's code -- src/tman/*.py,
scripts/tday_manifest.py and configs/tday_catalogue.json, and, for the reused arithmetic (decision (xvii)),
scripts/preflight_arms.py -- and a fresh interpreter imports the copy as src.tman, scripts.tday_manifest and
scripts.preflight_arms (the catalogue's bytes are served for the repository's path) and runs one case of one TMAN
smoke up to its named check. The mutation is killed when a check the case records fails or the case raises.
Each killing case first passes alone on the unmutated copy, its named checks among those it records. The
repository's files are read, never written; the scratch copies live under the system temp dir and are removed.

Run:  python -B scripts/smoke_tman_mutations.py          (a worker is the same file with --worker <spec>)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VAL, MAN, ENT, SCH = "src/tman/validate.py", "src/tman/manifest.py", "src/tman/entries.py", "src/tman/schema.py"
REC, COM, CLI, CAT = "src/tman/records.py", "src/tman/completion.py", "scripts/tday_manifest.py", \
    "configs/tday_catalogue.json"
PRE = "scripts/preflight_arms.py"                     # reused, not the lane's: mutated in the scratch copy only
SMOKES = {k: f"scripts.smoke_tman_{k}" for k in ("schema", "paths", "complete", "cuts", "parent", "derived", "pins",
                                                  "validate", "unlock", "noscore", "determinism")}
RESULT_TAG = "MUTATION-RESULT "

# (id, file, [(old, new), ...], smoke, case, the check the case runs to, the refusal codes it guards, plan sample)
M = [
    # ------------------------------------------------------------------------------------------- input gates
    ("test_path_gate", VAL, [("if p is not None and S.test_path_error(p):",
                              "if False and p is not None and S.test_path_error(p):")],
     "validate", "case_input_gates", "test_path_manifest_named_test", ["test_path"], None),
    ("manifest_strict_json", VAL, [('x.doc = S.strict_json(x.data, x.path, code="manifest_unreadable")',
                                    'x.doc = __import__("json").loads(x.data)')],
     "validate", "case_input_gates", "manifest_duplicate_key", ["manifest_unreadable"], None),
    ("manifest_parts", VAL, [("sorted(d) != list(S.MANIFEST_PARTS) or ", "")],
     "validate", "case_input_gates", "manifest_without_session_part", ["manifest_format"], None),
    ("catalogue_from_the_checkout", VAL, [('x.cat = S.load_catalogue(f.read_bytes() if f.is_file() else b"", '
                                           'S.CATALOGUE_REL)',
                                           'x.cat = S.load_catalogue((M.CODE_ROOT / S.CATALOGUE_REL).read_bytes(), '
                                           'S.CATALOGUE_REL)')],
     "validate", "case_input_gates", "catalogue_invalid", ["catalogue_invalid"], None),
    ("expected_rows", VAL, [("if v != S.EXPECTED_ROWS:", "if v is None:")],
     "validate", "case_input_gates", "expected_rows_1560_TB14", ["expected_rows_mismatch"], None),
    ("entry_kind", VAL, [('if not isinstance(e, dict) or e.get("kind") not in S.ENTRY_KINDS]',
                          "if not isinstance(e, dict)]")],
     "validate", "case_input_gates", "entry_kind_unknown", ["entry_kind_unknown"], None),
    ("entry_fields", VAL, [("return _err([err for e in x.entries for err in S.entry_errors(e)])", "return None")],
     "validate", "case_input_gates", "entry_field_extra_key", ["entry_field"], None),
    ("slot_known", VAL, [('bad = [e["slot"] for e in x.entries if e["slot"] not in x.cat.slots]', "bad = []")],
     "validate", "case_input_gates", "slot_unknown", ["slot_unknown"], None),
    ("slot_once", VAL, [('        if e["slot"] in seen:\n            dup.append',
                         "        if False:\n            dup.append")],
     "validate", "case_input_gates", "slot_duplicate", ["slot_duplicate"], None),
    ("class_of_the_slot", VAL, [('if e["kind"] not in S.KINDS_BY_CLASS[s["class"]] or e["class"] != s["class"]:',
                                 'if e["kind"] not in S.KINDS_BY_CLASS[s["class"]]:')],
     "validate", "case_input_gates", "derived_slot_in_a_run_kind", ["derived_counted_as_run"], None),
    # ------------------------------------------------------------------------------------------ record gates
    ("records_read", VAL, [("    for rel in rels:\n        S.refuse_test_path(rel)",
                            "    for rel in rels[:1]:\n        S.refuse_test_path(rel)")],
     "validate", "case_record_gates", "record_modified_in_the_tree", ["record_uncommitted"], None),
    ("dl_rows_exist", VAL, [('for e in x.entries for r in e["dl_rows"] if not x.dl.has(r)]',
                             'for e in x.entries for r in e["dl_rows"] if False]')],
     "validate", "case_record_gates", "dl_row_absent", ["dl_row_missing"], None),
    ("entry_file_sha", VAL, [('if ref["sha256"] not in (None, sha):', "if False:")],
     "validate", "case_record_gates", "cut_entry_cited_with_another_sha", ["entry_file_rewritten"], None),
    ("evaluator_commit_exists", VAL, [("if not (isinstance(e, str) and S.COMMIT_RE.fullmatch(e) and "
                                       "src.commit_exists(e)\n            and src.is_ancestor(e, src.records_ref)):",
                                       "if not (isinstance(e, str) and S.COMMIT_RE.fullmatch(e)):")],
     "validate", "case_record_gates", "evaluator_commit_not_a_commit", ["evaluator_commit_invalid"], None),
    ("evaluator_in_the_records_history", VAL, [("and src.is_ancestor(e, src.records_ref)):",
                                                'and src.is_ancestor(e, "HEAD")):')],
     "unlock", "case_unlock_gates", "records_commit_without_e_in_its_history", ["evaluator_commit_invalid"], None),
    ("s_equals_e", VAL, [("if x.mode in U and src.head() != e:", "if False:")],
     "unlock", "case_unlock_gates", "session_checkout_not_e_TB16", ["evaluator_commit_invalid"], None),
    ("execution_tuple", VAL, [("bad = [k for k, v in TUPLE_FIXED.items() if ec[k] != v]", "bad = []")],
     "validate", "case_record_gates", "execution_context_on_a_gpu_TB1", ["execution_context_split"], None),
    ("calibration_index", VAL, [("if ci != M.calibration_index(x.records):", "if False:")],
     "validate", "case_record_gates", "calibration_index_other_checksum", ["calibration_index_mismatch"], None),
    ("references", VAL, [("if refs != M.references(x.records):", "if False:")],
     "validate", "case_record_gates", "reference_other_strata_sha", ["reference_mismatch"], None),
    ("teacher_values", VAL, [('if e is None or e["kind"] != "scored_run":\n        return None\n'
                              '    ck = e["checkpoint"]',
                              'if True:\n        return None\n    ck = e["checkpoint"]')],
     "validate", "case_record_gates", "teacher_entry_not_DL88_s", ["teacher_values_mismatch"], None),
    ("derived_image_digest", VAL, [('if e["kind"] == "derived" and e["inputs"].get("image_digest") not in '
                                    "S.IMAGE_DIGESTS_OF_RECORD:", "if False:")],
     "validate", "case_record_gates", "derived_image_not_of_record", ["image_digest_not_of_record"], None),
    ("schedule_block", VAL, [('if x.body["schedule"] != x.freeze.as_block():', "if False:")],
     "validate", "case_record_gates", "schedule_block_other_freeze_date", ["schedule_invalid"], None),
    ("new_freeze_first_complete_day", VAL, [("if want != x.freeze.moved_date:", "if False:")],
     "validate", "case_moved_freeze", "new_freeze_not_the_first_complete_day", ["freeze_source_conflict"], None),
    ("content_class", VAL, [('if not (isinstance(gp.get(k), dict) and gp[k].get("pin")):',
                             "if not isinstance(gp.get(k), dict):")],
     "validate", "case_record_gates", "content_class_missing_pin", ["content_class_missing"], None),
    ("S11_pin_without_library_versions", MAN, [
        ('PIN_FIELDS = ("schema", "script", "commit", "library_versions", "parameters", "pin_sha256")',
         'PIN_FIELDS = ("schema", "script", "commit", "parameters", "pin_sha256")'),
        ("    for block, keys in PIN_KEYS.items():\n        got = pin[block]",
         "    for block, keys in list(PIN_KEYS.items())[1:]:\n        got = pin[block]")],
     "pins", "case_pin_fields", "pin_without_library_versions_incomplete", ["generator_pin_incomplete"], 11),
    ("pin_sha256", MAN, [('return None if pin.get("pin_sha256") == got else', "return None if True else")],
     "pins", "case_pin_sha256", "pin_sha256_mismatch_after_an_edit", ["generator_pin_sha256_mismatch"], None),
    ("pin_cells_order", MAN, [("return [] if cells == reg else",
                               "return [] if sorted(cells or []) == sorted(reg) else")],
     "pins", "case_cells", "cells_reordered_refused", ["generator_cells_mismatch"], None),
    ("S10_test_value_before_the_unlock", MAN, [('        if block.get("value") is not None:\n            errs.append',
                                                "        if False:\n            errs.append")],
     "pins", "case_values_and_hasher", "corrupted_cache_value_before_the_unlock_refused",
     ["test_value_before_unlock"], 10),
    ("session_part_before_the_session", VAL, [('if x.doc["session"] is not None and x.mode != "session":',
                                               "if False:")],
     "validate", "case_record_gates", "session_part_before_the_session", ["test_value_before_unlock"], None),
    ("trigger_e6kd", VAL, [('if t["triggered"] and "E6-KD:s42:15ep" not in x.by_slot:', "if False:")],
     "validate", "case_record_gates", "triggered_without_e6kd", ["trigger_inconsistent"], None),
    # -------------------------------------------------------------------------------------- completion gates
    ("attempt_directories", VAL, [("missing = sorted(r for r in ids if not x.run_dir(r).is_dir())", "missing = []")],
     "validate", "case_completion_gates", "attempt_without_directory", ["launched_attempt_without_directory"], None),
    ("runs_needed_before_the_runs_gate", VAL, [("        if self.runs_root is None:                    # a gate",
                                                "        if False:                    # a gate")],
     "validate", "case_moved_freeze", "delay_entry_without_runs_refused_not_raised",
     ["launched_attempt_without_directory"], None),
    ("e1_max_iters", COM, [('if not metas or not S.TYPES["int"](metas[-1].get("max_iters")):', "if not metas:")],
     "validate", "case_completion_gates", "completion_record_unreadable", ["completion_record_invalid"], None),
    ("completion_block", VAL, [('if st.completion is None or e["completion"] != st.completion.as_entry():',
                                "if st.completion is None:")],
     "validate", "case_completion_gates", "completion_block_other_line", ["completion_source_mismatch"], None),
    ("telemetry_copy_back", MAN, [('if recorded is not None and (last is None or last["sha256"] != recorded):',
                                   "if False:")],
     "validate", "case_inventory", "telemetry_copy_back_differs_IF11_3", ["completion_source_mismatch"], None),
    ("qat_checkpoint_flags", VAL, [('errs += C.qat_flag_errors(x.run_dir(e["run_id"]), rows)', "errs += []")],
     "validate", "case_completion_gates", "qat_checkpoint_flag_disagrees_TB13", ["checkpoint_state_flag"], None),
    ("S01_meets_flipped", COM, [("        return ss.meets(self.at, cutoff)", "        return self.at <= cutoff")],
     "validate", "case_completion_gates", "complete_at_the_freeze_instant_is_late", ["entry_not_complete"], 1),
    ("S02_completeness_loop_removed", VAL, [("    for rid in sorted(log.launched if log else []):\n"
                                             "        slot = slot_of_launch(log.launch[rid])",
                                             "    for rid in []:\n        slot = slot_of_launch(log.launch[rid])")],
     "validate", "case_completion_gates", "complete_kd_run_omitted_AM19_3d", ["complete_run_omitted"], 2),
    ("checkpoint_sources", VAL, [('if src["sha256"] != ck["sha256"] or (src["dl"] and ck["sha256"] not in '
                                  '(x.dl.row_text(src["dl"]) or "")):', "if False:")],
     "validate", "case_completion_gates", "checkpoint_source_holds_another_sha", ["checkpoint_hash_mismatch"], None),
    ("checkpoint_bytes", VAL, [('if not f.is_file() or S.sha256_bytes(f.read_bytes()) != ck["sha256"]:',
                                "if not f.is_file():")],
     "validate", "case_completion_gates", "checkpoint_file_differs_DL110", ["checkpoint_missing"], None),
    ("restored_checkpoint", VAL, [('for src in e["checkpoint"]["sources"] if src["kind"] == "restored" and '
                                   'not src["dl"]]', 'for src in e["checkpoint"]["sources"] if False]')],
     "validate", "case_completion_gates", "restored_checkpoint_without_its_report",
     ["restored_checkpoint_unrecorded"], None),
    # ------------------------------------------------------------------------------------------- cut checks
    ("S03_rule_operator_allowed", ENT, [('return None if doc["rule"] in S.CUT_RULES else',
                                         'return None if doc["rule"] in S.CUT_RULES + ("operator",) else')],
     "cuts", "case_gate_order", "rule_operator_refused", ["cut_rule_unknown"], 3),
    ("S04_never_cut_test_skipped", ENT, [('bad = [s["slot"] for s in _slots(doc, ctx) if s["never_cut"]]',
                                          "bad = []")],
     "cuts", "case_gate_order", "never_cut_slot_refused", ["cut_of_never_cut"], 4),
    ("complete_run_cut", ENT, [('done = [s["slot"] for s in _slots(doc, ctx) if ctx.status(s["slot"]) == '
                                "C.COMPLETE]", "done = []")],
     "cuts", "case_gate_order", "complete_run_cut_refused", ["cut_of_complete_run"], None),
    ("never_launched_reason", ENT, [('if doc["reason"] not in S.NEVER_LAUNCHED_REASONS:', "if False:")],
     "cuts", "case_gate_order", "never_launched_reason_outside_list", ["cut_reason_invalid"], None),
    ("cut_day", ENT, [('return None if want == doc["date"] else', "return None if True else")],
     "cuts", "case_gate_order", "freeze_cut_dated_off_its_day", ["cut_date_mismatch"], None),
    ("S05_arithmetic_ge", PRE, [("    cut = needed > left\n", "    cut = needed >= left\n")],
     "cuts", "case_arithmetic", "arithmetic_equality_fits", ["cut_arithmetic_invalid"], 5),
    ("arithmetic_recomputed", ENT, [("        elif x != y:\n            return",
                                     "        elif False:\n            return")],
     "cuts", "case_arithmetic", "arithmetic_recomputation_catches_cut", ["cut_arithmetic_invalid"], None),
    ("launched_cut_records", ENT, [('if missing or doc["comparator"] is None:', "if False:")],
     "cuts", "case_gate_order", "launched_cut_without_telemetry", ["cut_val_record_missing"], None),
    ("arm_results_known", ENT, [('if _arm_group(doc, ctx) and not doc["results_known"]:', "if False:")],
     "cuts", "case_gate_order", "arm_cut_without_results", ["arm_cut_results_missing"], None),
    ("arm_running", ENT, [('if _arm_group(doc, ctx) and doc["launched"] and doc["rule"] != "freeze":', "if False:")],
     "cuts", "case_gate_order", "launched_arm_cut_by_named_rule_refused", ["arm_run_cut_while_running"], None),
    ("arm_partial", ENT, [("if not cut or cut != runs[first:] or any(", "if not cut or any(")],
     "cuts", "case_gate_order", "arm_cut_missing_its_later_run", ["arm_cut_partial"], None),
    ("cut_order_position", ENT, [('return None if pos == want else f"cut_order_position',
                                  'return None if True else f"cut_order_position')],
     "cuts", "case_gate_order", "freeze_cut_wrong_position", ["cut_order_position_mismatch"], None),
    ("decision_record_cuts", ENT, [("if rec is None or not vals <= cut_values:", "if rec is None:")],
     "cuts", "case_gate_order", "decision_date_cut_not_in_the_record", ["decision_date_cut_mismatch"], None),
    ("stopped_run", ENT, [('if stopped and (doc["rule"] != "freeze" or doc["date"] != ctx.freeze.original_date):',
                           "if False:")],
     "cuts", "case_gate_order", "stopped_run_cut_by_named_rule_refused", ["stopped_run_cut_early"], None),
    ("arm_registry", ENT, [("missing = [st for st in stages if st not in ctx.registry]", "missing = []")],
     "cuts", "case_gate_order", "arm_without_registry_file_refused", ["unregistered_arm"], None),
    ("condition_no", ENT, [('return None if verdict == "no" else', 'return None if verdict in ("no", "yes") else')],
     "cuts", "case_gate_order", "e6x_cut_citing_a_yes_condition_refused", ["condition_unmet"], None),
    # -------------------------------------------------------------------------------------------- parent rule
    ("parent_without_a_model", ENT, [('if pe is not None and pe["kind"] in NO_MODEL_KINDS and e["kind"] not in '
                                      "S.TERMINAL_KINDS:", "if False:")],
     "parent", "case_refusals", "derived_under_a_cut_parent_refused", ["parent_rule_violation"], None),
    ("S06_parent_map_E7_to_E1", CAT, [('"file_id": "E7_s43_list42",\n      "gates": [],\n      "model_role": '
                                       '"student",\n      "never_cut": true,\n      "params": {\n        "list": 42\n'
                                       '      },\n      "parent": "E3:s43:lambda0.5:alpha25:80k",',
                                       '"file_id": "E7_s43_list42",\n      "gates": [],\n      "model_role": '
                                       '"student",\n      "never_cut": true,\n      "params": {\n        "list": 42\n'
                                       '      },\n      "parent": "E1:s43:80k",')],
     "validate", "case_parent_and_derived_gates", "derived_under_a_cut_parent", ["parent_rule_violation"], 6),
    ("descendants_entered", ENT, [('if ce.get("kind") != "not_run" and not same_cut:', "if False:")],
     "parent", "case_refusals", "cut_parent_without_its_descendants_refused", ["descendant_entry_missing"], None),
    ("not_run_parent_record", ENT, [('                or e["parent_record"]["sha256"] != parent_record_sha256(pe):',
                                     "                or False:")],
     "parent", "case_refusals", "not_run_citing_another_record_refused", ["not_run_without_parent_record"], None),
    # ------------------------------------------------------------------------------------------ derived gates
    ("derived_parent", MAN, [('if entry["parent"] not in allowed or pe is None or pe["kind"] not in COMPLETE_KINDS:',
                              "if pe is None:")],
     "validate", "case_parent_and_derived_gates", "derived_parent_other_than_catalogue",
     ["derived_parent_incomplete"], None),
    ("derived_inputs", MAN, [("errs = S.value_errors(entry[\"inputs\"], (\"obj\", inputs_spec(entry[\"method\"])), "
                              "f\"{entry['slot']}.inputs\")", "errs = []")],
     "validate", "case_parent_and_derived_gates", "derived_inputs_without_qconfig", ["derived_inputs_incomplete"],
     None),
    ("ptq_record_seed", MAN, [('want = {"stage": s["evaluator_stage"], "seed": s["seed"], "role": want_role, '
                               '"descriptive": not of_record}',
                               'want = {"stage": s["evaluator_stage"], "role": want_role, '
                               '"descriptive": not of_record}')],
     "derived", "case_ptq_record", "ptq_record_other_seed_refused_A2", ["ptq_record_invalid"], None),
    ("append_ptq_record_read", VAL, [('    for a in x.doc["appends"]:\n        frozen = x.by_slot.get(a.get("slot"))',
                                      '    for a in []:\n        frozen = x.by_slot.get(a.get("slot"))')],
     "validate", "case_appends", "post_freeze_append_of_record_without_its_ptq_record_A2", ["ptq_record_invalid"],
     None),
    ("required_derived", MAN, [('        if kind == "derived" or (kind == "not_made" and code_gated(k, cat)):\n'
                                "            continue", "        if True:\n            continue")],
     "validate", "case_parent_and_derived_gates", "required_derived_absent", ["derived_missing"], None),
    ("e6s42_hashed", MAN, [('if not (isinstance(h, dict) and h.get("appended") == "freeze"):', "if False:")],
     "validate", "case_parent_and_derived_gates", "e6s42_by_inputs_at_the_freeze", ["e6s42_not_hashed_at_freeze"],
     None),
    ("fault_report", MAN, [('if not ev or any(e["fault_report"] is None for e in ev[1:]):', "if not ev:")],
     "validate", "case_parent_and_derived_gates", "second_evaluation_without_report", ["derived_selection_conflict"],
     None),
    ("non_finite_flags", MAN, [('elif len(ends) != 15 or any(r.get("state_finite") is not False for r in ends):',
                                "elif len(ends) != 15:")],
     "validate", "case_non_finite_gate", "non_finite_with_a_convertible_epoch", ["non_finite_invalid"], None),
    ("S07_isolation_test_dropped", MAN, [('if not D.meets(made, freeze_utc) and not (ptq_record_ok and '
                                          'frozen["method"] == "ptq"):', "if False:")],
     "validate", "case_appends", "post_freeze_append_without_an_isolation_record_xi", ["append_invalid"], 7),
    ("isolation_cited", MAN, [('        if S.sha256_bytes(data) != h["isolation"]["sha256"]:', "        if False:")],
     "validate", "case_appends", "isolation_record_not_the_cited_one", ["append_invalid"], None),
    ("cli_ptq_stand_in", CLI, [("ptq_record_ok=ok)", "ptq_record_ok=False)")],
     "paths", "case_cli_outputs", "cli_post_freeze_ptq_append_with_its_record_TB19_Q9", ["append_invalid"], None),
    ("S08_appends_after_the_unlock", VAL, [('if u is not None and S.canonical_sha256(x.doc["appends"]) != '
                                            'u["manifest"]["appends_sha256"]:', "if False:")],
     "unlock", "case_unlock_gates", "appends_changed_after_the_unlock_record", ["append_after_unlock"], 8),
    ("hashed_at_the_unlock", VAL, [('bad = [e["slot"] for e in _derived(x) if e["hashed"] is None and e["slot"] '
                                    "not in hashed]", "bad = []")],
     "unlock", "case_unlock_gates", "derived_by_inputs_at_the_unlock", ["derived_not_hashed_at_unlock"], None),
    # -------------------------------------------------------------------------------------------- scope gates
    ("S12_scope_rule_relaxed", SCH, [('        (req if pair_has_models else forb).add("corrupted")',
                                      '        (req if pair_has_models else set()).add("corrupted")')],
     "validate", "case_scope_gates", "scope_corrupted_for_a_per_seed_run", ["scope_forbidden"], 12),
    ("scope_required", VAL, [('if e["kind"] in MODEL_KINDS and scope_rule_of(x, e["slot"])[0] - set(e["scope"])]',
                              "if False]")],
     "validate", "case_scope_gates", "scope_upstream_missing_TB3", ["scope_required_missing"], None),
    ("S13_default_candidate_rule_dropped", VAL, [
        ('bad = [e["slot"] for e in x.entries if x.slot(e["slot"])["sweep"] and e["kind"] in MODEL_KINDS',
         'bad = [e["slot"] for e in x.entries if False and x.slot(e["slot"])["sweep"] and e["kind"] in MODEL_KINDS')],
     "validate", "case_scope_gates", "non_default_candidate_scored", ["candidate_scored"], 13),
    ("default_candidate_kind", VAL, [('.get("kind") != "default_candidate"]',
                                      '.get("kind") not in ("default_candidate", "listed_run")]')],
     "validate", "case_scope_gates", "default_candidate_entered_as_listed", ["default_candidate_missing"], None),
    ("listed_scope", VAL, [('bad = [e["slot"] for e in x.entries if e["kind"] not in MODEL_KINDS and e["scope"]]',
                            "bad = []")],
     "validate", "case_scope_gates", "listed_run_with_a_scope", ["listed_or_terminal_scored"], None),
    # ------------------------------------------------------------------------------------------- unlock gates
    ("S15_body_hash_not_recomputed", VAL, [('if fr["body_sha256"] != S.canonical_sha256(x.body):', "if False:")],
     "unlock", "case_unlock_gates", "body_changed_after_the_freeze_record", ["run_added_after_freeze"], 15),
    ("S09_original_instant_for_never_cut", VAL, [
        ('return self.freeze.moved_utc if self.slot(slot)["never_cut"] else self.freeze.original_utc',
         "return self.freeze.original_utc")],
     "validate", "case_moved_freeze", "delay_and_new_freeze_on_the_first_complete_day_AM19_1d",
     ["entry_not_complete"], 9),
    ("unlock_cites_the_freeze_record", VAL, [('if u is not None and (u.get("kind") != "unlock" or '
                                              'u["freeze_record"]["path"] != x.freeze_rel',
                                              'if False and (u.get("kind") != "unlock" or '
                                              'u["freeze_record"]["path"] != x.freeze_rel')],
     "unlock", "case_unlock_gates", "unlock_entry_citing_another_freeze_record", ["run_added_after_freeze"], None),
    ("unlock_records_the_body", VAL, [('m["body_sha256"] != S.canonical_sha256(x.body)\n', "False\n")],
     "unlock", "case_unlock_gates", "unlock_entry_of_another_body", ["run_added_after_freeze"], None),
    ("unlock_records_the_file", VAL, [('or (x.mode == "unlock" and m["sha256"] != S.sha256_bytes(x.data))):',
                                       "or False):")],
     "unlock", "case_unlock_gates", "manifest_file_not_the_unlock_entry_s", ["run_added_after_freeze"], None),
    ("every_slot_entered", VAL, [('bad = [k for k, s in x.cat.slots.items() if s["class"] in ("run", "measurement") '
                                  "and k not in x.by_slot",
                                  'bad = [k for k, s in x.cat.slots.items() if False and s["class"] in ("run", '
                                  '"measurement") and k not in x.by_slot')],
     "unlock", "case_unlock_gates", "slot_without_entry_at_the_unlock", ["cut_unentered_at_unlock"], None),
    ("lane_implemented", VAL, [('(x.dl.status(st["dl"]) or "").startswith("IMPLEMENTED")', 'x.dl.has(st["dl"])')],
     "unlock", "case_unlock_gates", "lane_neither_implemented_nor_not_required", ["lane_unmet"], None),
    ("lanes_the_unlock_cites", VAL, [('if u is not None and (u["lanes"]["path"], u["lanes"]["sha256"]) != '
                                      "(S.LANES_REL, sha):", "if False:")],
     "unlock", "case_unlock_gates", "lanes_not_the_unlock_entry_s", ["lane_unmet"], None),
    ("not_required_covered", VAL, [('if set(lane["serves"]) - covered:', "if False:")],
     "unlock", "case_unlock_gates", "not_required_lane_serving_an_uncut_slot", ["not_required_invalid"], None),
    ("S16_file_set_comparison_dropped", VAL, [('    for rel, blob in sorted(ev["files"].items()):\n'
                                               "        f = x.repo_root / rel",
                                               "    for rel, blob in []:\n        f = x.repo_root / rel")],
     "unlock", "case_unlock_gates", "evaluator_file_edited_at_S", ["evaluator_files_changed"], 16),
    ("hash_only_compared", VAL, [("if S.sha256_bytes(f.read_bytes()) != S.sha256_bytes(src.blob_bytes(blob)):",
                                  "if False:")],
     "unlock", "case_unlock_gates", "hash_only_evaluator_file_edited_at_S", ["evaluator_files_changed"], None),
    ("hasher_script", VAL, [('if not hp.is_file() or S.sha256_bytes(hp.read_bytes()) != hasher["script"]["sha256"]:',
                             "if not hp.is_file():")],
     "unlock", "case_unlock_gates", "hasher_script_not_its_pin", ["hasher_pin_mismatch"], None),
    ("one_unlock_row", VAL, [("if len(rows) > 1:", "if len(rows) > 2:")],
     "unlock", "case_unlock_gates", "second_unlock_entry_in_the_decision_log", ["second_session"], None),
    ("binding_written_once", VAL, [("if x.records.has(BINDING_REL):", "if False:")],
     "unlock", "case_session", "binding_already_written", ["second_session"], None),
    ("test_deadline", VAL, [("if t is not None and not D.meets(t, cutoff):", "if False:")],
     "unlock", "case_unlock_gates", "unlock_after_the_test_deadline_TB14", ["session_after_deadline"], None),
    ("clock_offset_limit", VAL, [('abs(doc["offset_seconds"]) > S.CLOCK_OFFSET_LIMIT_S',
                                  'abs(doc["offset_seconds"]) > 100')],
     "unlock", "case_session", "clock_offset_over_60_s_AM19a_r21", ["clock_offset_invalid"], None),
    ("push_before_the_session", VAL, [('and D.parse_utc_any(p.get("timestamp")) < started', "and True")],
     "unlock", "case_session", "unlock_pushed_after_the_session_started_TB11", ["push_evidence_invalid"], None),
    # -------------------------------------------------------------------------------------- the inventory
    ("rows_unique", VAL, [("            if key in seen:\n                errs.append",
                           "            if False:\n                errs.append")],
     "validate", "case_inventory", "rows_unique_by_slot_and_precision_IF11_1", ["inventory_invalid"], None),
    ("inventory_sha256", VAL, [("if expect_sha256 is not None and S.sha256_bytes(data) != expect_sha256:",
                                "if False:")],
     "validate", "case_inventory", "inventory_sha256_mismatch", ["inventory_sha256_mismatch"], None),
    ("binding_sha256", VAL, [('if S.sha256_bytes(bdata) != doc["binding"]["sha256"]:', "if False:")],
     "validate", "case_inventory", "bound_binding_other_sha256", ["binding_invalid"], None),
    # ------------------------------------------------------------------ score-blindness, paths, determinism
    ("S14_reader_reads_best_val", REC, [('"candidates": [{"lambda": True, "alpha": True, "run_id": True, '
                                         '"ckpt_sha256": True,',
                                         '"candidates": [{"lambda": True, "alpha": True, "run_id": True, '
                                         '"ckpt_sha256": True, "best_val": True,')],
     "noscore", "case_sentinel", "sentinel_quiet_through_every_mode_builder_export_and_printer", [], 14),
    ("decision_record_keeps_directories", REC, [('"values": [{"value": True, "status": True}]}',
                                                 '"values": [{"value": True, "status": True, "directories": True}]}')],
     "noscore", "case_sentinel", "decision_record_keeps_grid_values_and_statuses_only", [], None),
    ("score_literal_in_the_code", VAL, [('MODES = ("draft", "freeze", "unlock", "session")\n',
                                         'MODES = ("draft", "freeze", "unlock", "session")\n'
                                         'PROBE = "best_val_miou_all_class"\n')],
     "noscore", "case_static_scan", "no_score_key_literal_in_the_lane_s_code", [], None),
    ("records_from_the_folder_only", REC, [('_, data = self.src.read(rel, missing=missing, '
                                            'uncommitted="record_uncommitted")',
                                            '_, data = self.src.read(rel, missing=missing, '
                                            'uncommitted="record_uncommitted", record=True)')],
     "unlock", "case_unlock_world", "unlock_world_passes_every_unlock_gate_S_equals_E", ["record_uncommitted"],
     None),
    ("unreadable_run_file", REC, [("    except OSError as e:\n        raise S.TmanRefused(code, f\"{path}: ",
                                   "    except KeyError as e:\n        raise S.TmanRefused(code, f\"{path}: ")],
     "noscore", "case_removal_changes_the_verdict", "telemetry_removed_refused", ["completion_record_invalid"],
     None),
    ("test_named_module_parsed", MAN, [('if not rel.endswith(".py") or hash_only(rel):',
                                        'if not rel.endswith(".py"):')],
     "paths", "case_evaluator_closure_never_reads_a_test_named_module",
     "closure_records_a_test_named_module_by_blob_id_unread", [], None),
    ("entry_written_over", ENT, [('        with open(out, "xb") as fh:', '        with open(out, "wb") as fh:')],
     "paths", "case_cli_outputs", "a_second_write_is_refused_and_the_first_kept", ["entry_file_rewritten"], None),
    ("out_inside_the_repository", CLI, [("if p == repo or repo in p.parents:", "if p == repo:")],
     "paths", "case_cli_outputs", "out_in_the_repository_at_root_refused", ["test_path"], None),
    ("teacher_logs_unsorted", COM, [("logs = [log_rel] if log_rel else sorted(",
                                     "logs = [log_rel] if log_rel else list(")],
     "determinism", "case_listing_order", "shuffled_listing_4_byte_identical", [], None),
]
REFUSAL_EXTRA = ("inventory_invalid", "inventory_sha256_mismatch", "binding_invalid")       # read_inventory's codes
MUTATED_FILES = (VAL, MAN, ENT, SCH, REC, COM, CLI, CAT, PRE)


# ================================================================================================ the worker
def worker(spec: dict) -> dict:
    """Run in a fresh interpreter: import the scratch copy at spec["root"] as src.tman (and scripts.tday_manifest,
    scripts.preflight_arms), serve its catalogue bytes for the repository's path, run spec["case"] of spec["smoke"]
    up to spec["check"] (None: the whole case) and return the checks it recorded and any exception."""
    import importlib
    import importlib.util
    import src
    root = Path(spec["root"])

    def load(name: str, path: Path, package: bool = False):
        kw = {"submodule_search_locations": [str(path.parent)]} if package else {}
        sp = importlib.util.spec_from_file_location(name, path, **kw)
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
        return mod
    src.tman = load("src.tman", root / "src/tman/__init__.py", package=True)
    load("scripts.tday_manifest", root / CLI).REPO = REPO
    if (root / PRE).is_file():
        load("scripts.preflight_arms", root / PRE).REPO = REPO
    cat, real = (root / CAT).read_bytes(), (REPO / CAT).resolve()
    if cat != real.read_bytes():
        orig = Path.read_bytes
        Path.read_bytes = lambda self: cat if self.resolve() == real else orig(self)
    import scripts.smoke_tman_schema as FX
    first, target = FX.check, spec["check"]

    class Stop(Exception):
        pass

    def check(name, ok, detail=""):
        first(name, ok, detail)
        if name == target:
            raise Stop
    FX.check = check
    mod = importlib.import_module(SMOKES[spec["smoke"]])
    for m in list(sys.modules.values()):
        if getattr(m, "check", None) is first:
            m.check = check
    raised = None
    try:
        if spec["smoke"] == "validate":
            mod.WORLD = mod.World()
        elif spec["smoke"] == "unlock":
            mod.UW = mod.Unlock()
        elif spec["smoke"] == "determinism":
            mod.WORLDS.extend([mod.VW.World(), mod.VW.World()])
        getattr(mod, spec["case"])()
    except Stop:
        pass
    except Exception as e:  # noqa: BLE001 -- a case that raises fails by name
        raised = f"{type(e).__name__}: {e}"[:300]
    FX.remove_scratch()
    return {"results": [(n, ok, d[:200]) for n, ok, d in FX.RESULTS], "raised": raised}


# ================================================================================================ the parent
def scratch_root(edits_by_file: dict, FX) -> Path:
    """A scratch copy of the lane's code (and preflight_arms when it is mutated) with the edits applied; each old
    text must occur exactly once (ValueError otherwise)."""
    from src.tman import manifest as MM
    root = FX.scratch_dir("tmanmut")
    for rel in tuple(MM.LANE_CODE_FILES) + ((PRE,) if PRE in edits_by_file else ()):
        text = (REPO / rel).read_bytes().decode("utf-8")
        for old, new in edits_by_file.get(rel, ()):
            if text.count(old) != 1:
                raise ValueError(f"{rel}: {text.count(old)} occurrences of {old[:70]!r}")
            text = text.replace(old, new)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(text.encode("utf-8"))
    return root


def run_worker(spec: dict) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    try:
        r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
                           capture_output=True, text=True, env=env, cwd=str(REPO), timeout=900)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(RESULT_TAG)]
    if not lines:
        return {"error": (r.stderr or r.stdout)[-400:]}
    return json.loads(lines[-1][len(RESULT_TAG):])


def files_sha256() -> dict:
    import hashlib
    from src.tman import manifest as MM
    return {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in tuple(MM.LANE_CODE_FILES) + (PRE,)}


def case_mutations():
    import scripts.smoke_tman_schema as FX
    from src.tman import validate as V
    check = FX.check
    before = files_sha256()
    clean = scratch_root({}, FX)
    cases = sorted({(m[3], m[4]) for m in M})
    jobs = [("baseline", (smoke, case), {"root": str(clean), "smoke": smoke, "case": case, "check": None})
            for smoke, case in cases]
    for mid, rel, edits, smoke, case, target, codes, sample in M:
        try:
            root = scratch_root({rel: edits}, FX)
        except ValueError as e:
            check(f"mut_{mid}_text_occurs_once", False, str(e))
            continue
        jobs.append(("mutation", mid, {"root": str(root), "smoke": smoke, "case": case, "check": target}))
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as pool:
        outs = list(pool.map(lambda job: run_worker(job[2]), jobs))
    targets = {}
    for m in M:
        targets.setdefault((m[3], m[4]), set()).add(m[5])
    killed_codes = set()
    for (kind, key, spec), out in zip(jobs, outs):
        if kind == "baseline":
            smoke, case = key
            names = {n for n, _, _ in out.get("results", [])}
            bad = [n for n, ok, _ in out.get("results", []) if not ok]
            check(f"baseline_{smoke}_{case}_passes_alone", "error" not in out and out.get("raised") is None
                  and not bad and targets[key] <= names,
                  str(out.get("error") or out.get("raised") or bad or sorted(targets[key] - names))[:300])
            continue
        m = next(x for x in M if x[0] == key)
        fails = [n for n, ok, _ in out.get("results", []) if not ok]
        killed = "error" not in out and (out.get("raised") is not None or bool(fails))
        how = out.get("error") or (f"raised {out['raised']}" if out.get("raised") else f"failed {fails[:3]}")
        print(f"MUTATION {key:42} {m[1]:30} {'KILLED' if killed else 'SURVIVED'} by {m[3]}:{m[4]} -- {how[:160]}")
        check(f"mut_{key}", killed, f"survived: {m[3]}:{m[4]} ran to {m[5]} with every check passing" if not killed
              and "error" not in out else str(out.get("error"))[:300])
        if killed:
            killed_codes |= set(m[6])
    codes = {c for c, _, _ in V.GATES} | set(REFUSAL_EXTRA)
    check("every_refusal_code_has_a_killed_mutation", codes <= killed_codes and len(codes) == 82,
          str(sorted(codes - killed_codes)))
    check("plan_s_sixteen_sample_mutations_present", sorted(m[7] for m in M if m[7]) == list(range(1, 17)))
    check("the_repository_files_are_unchanged", files_sha256() == before)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(RESULT_TAG + json.dumps(worker(json.loads(sys.argv[2]))))
        return 0
    import scripts.smoke_tman_schema as FX
    return FX.run_cases("TMAN MUTATIONS SMOKE (lane TMAN, B-1, C6)", (case_mutations,))


if __name__ == "__main__":
    raise SystemExit(main())
