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

Run:  python -B scripts/smoke_tman_mutations.py          (a worker is the same file with --worker <spec>; --only
      <id>,<id> runs those mutations and their cases alone, a development aid whose RESULT is not the record)
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
    ("entry_file_sha", VAL, [('if c["sha256"] not in (None, sha):', "if False:")],
     "validate", "case_record_gates", "cut_entry_cited_with_another_sha", ["entry_file_rewritten"], None),
    # ------------------------------------------- the cited entry files (decision (iii); DL-24 #17 #38 #87-#89 #92)
    ("entry_file_dl_row", VAL, [("        if not x.dl.recorded_rows_with(sha):\n            errs.append(f\"{rel}: sha256",
                                 "        if False:\n            errs.append(f\"{rel}: sha256")],
     "validate", "case_entry_files", "entry_file_whose_row_is_only_DECIDED_B92", ["dl_row_missing"], None),
    ("entry_file_written_once", VAL, [("        err = x.records.written_once_error(rel)\n        if err:",
                                       "        err = None\n        if err:")],
     "validate", "case_entry_files", "entry_file_rewritten_after_its_commit_B92", ["entry_file_rewritten"], None),
    ("entry_file_kind", ENT, [('    if doc["kind"] != kind:\n        return [f"{rel}: a',
                               '    if False:\n        return [f"{rel}: a')],
     "validate", "case_entry_files", "not_run_citing_a_cut_entry_file_B38", ["entry_file_invalid"], None),
    ("entry_file_name", ENT, [("    err = entry_name_error(rel, doc)\n    return [err] if err else []",
                               "    err = None\n    return [err] if err else []")],
     "validate", "case_entry_files", "cut_entry_file_outside_its_folder_B89", ["entry_file_invalid"], None),
    ("entry_file_fields", ENT, [('        require_fields(doc)\n    except S.TmanRefused as e:\n        return [f"{rel}: ',
                                 '        pass\n    except S.TmanRefused as e:\n        return [f"{rel}: ')],
     "validate", "case_moved_freeze", "freeze_delay_file_without_runs_B88", ["entry_file_invalid"], None),
    ("entry_file_slots_known", VAL, [('unknown = [k for k in doc["slots"] if k not in x.cat.slots]', "unknown = []")],
     "validate", "case_entry_files", "cut_file_naming_an_unknown_slot", ["entry_file_invalid"], None),
    ("entry_file_slot_class", VAL, [('and c["kind"] not in S.KINDS_BY_CLASS[x.slot(k)["class"]]]', "and False]")],
     "validate", "case_entry_files", "cut_file_naming_a_derived_slot", ["entry_file_invalid"], None),
    ("entry_file_of_its_slot", VAL, [('    if e["slot"] not in doc["slots"]:\n        return f"{e[\'slot\']}: its',
                                      '    if False:\n        return f"{e[\'slot\']}: its')],
     "validate", "case_entry_files", "not_run_citing_another_slot_s_not_run_file_B17", ["entry_file_invalid"], None),
    ("entry_file_cut_label", VAL, [('if (e["cut_date"], e["rule"], e["comparator"]) != (doc["date"], doc["rule"], '
                                    "cmp):", "if False:")],
     "validate", "case_entry_files", "cut_label_against_its_entry_file", ["entry_file_invalid"], None),
    ("entry_file_not_run_fields", VAL, [('    elif e["kind"] == "not_run" and (e["reason"], e["parent"], ',
                                         '    elif False and (e["reason"], e["parent"], ')],
     "validate", "case_parent_and_derived_gates", "not_run_entry_against_its_file", ["entry_file_invalid"], None),
    ("cited_condition", VAL, [('                ref(doc.get("rule_record"), "condition")', "                pass")],
     "validate", "case_cut_gates", "e6x_cut_citing_a_cut_entry_as_its_condition", ["entry_file_invalid"], None),
    ("cited_freeze_source", VAL, [('            rel = ref(src, src["provider"])', '            rel = src.get("path")')],
     "validate", "case_moved_freeze", "freeze_delay_file_without_runs_B88", ["entry_file_invalid"], None),
    ("cited_freeze_record", VAL, [('        ref({"path": self.freeze_rel} if self.freeze_rel else None, "freeze")',
                                   "        pass")],
     "unlock", "case_unlock_gates", "freeze_record_without_body_sha256_B89", ["entry_file_invalid"], None),
    ("cited_unlock_entry", VAL, [('        ref({"path": self.unlock_rel} if self.unlock_rel else None, "unlock")',
                                  "        pass")],
     "unlock", "case_unlock_gates", "unlock_entry_whose_manifest_is_a_string_B89", ["entry_file_invalid"], None),
    ("cited_clock_offset", VAL, [('        ref(s.get("clock_offset") if isinstance(s, dict) else None, "clock_offset")',
                                  "        pass")],
     "unlock", "case_session", "clock_offset_entry_not_DL_recorded_B102", ["dl_row_missing"], None),
    ("cited_not_required", VAL, [('st.get("kind") == "not_required" else None,', 'False else None,')],
     "unlock", "case_unlock_gates", "not_required_entry_not_DL_recorded_B55c", ["dl_row_missing"], None),
    ("cited_not_required_cuts", VAL, [('                ref(c, "cut")', "                pass")],
     "unlock", "case_unlock_gates", "not_required_cut_entry_not_DL_recorded_B55c", ["dl_row_missing"], None),
    ("results_known_committed", VAL, [("    cited += x.results_known()", "    cited += []")],
     "validate", "case_entry_files", "results_known_at_another_sha256_B69", ["record_uncommitted"], None),
    ("results_known_dl_row", VAL, [('if isinstance(r.get("dl"), str) and r["sha256"] not in (x.dl.row_text(r["dl"]) '
                                    'or ""):', "if False:")],
     "validate", "case_entry_files", "results_known_row_not_holding_its_sha256_B69", ["dl_row_missing"], None),
    ("not_made_lane_merged", VAL, [("        if lane in merged:", "        if False:")],
     "validate", "case_parent_and_derived_gates", "not_made_whose_lane_merged_by_the_freeze_B23_B76",
     ["derived_missing"], None),
    ("lane_merged_by_the_freeze", VAL, [("if t is not None and D.meets(t, x.freeze.original_utc) and",
                                         "if t is not None and")],
     "validate", "case_parent_and_derived_gates", "not_made_whose_lane_merged_after_the_freeze_passes", [], None),
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
     "validate", "case_record_gates", "calibration_index_other_split_list", ["calibration_index_mismatch"], None),
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
    ("trigger_sha256_compared", VAL, [('        cited.append(x.body["trigger"])', "        pass")],
     "validate", "case_record_gates", "trigger_record_other_sha256_B1", ["record_uncommitted"], None),
    ("trigger_reader_required", VAL, [("        return f\"{t.get('path')}: no reader of the AM-3 trigger record",
                                       "        return None and f\"{t.get('path')}: no reader of the AM-3 trigger "
                                       "record")],
     "validate", "case_record_gates", "trigger_format_unknown_without_a_reader_O8", ["trigger_format_unknown"], None),
    ("trigger_verdict_compared", VAL, [('if x.trigger_verdict is not None and t["triggered"] != x.trigger_verdict:',
                                        "if False:")],
     "validate", "case_record_gates", "trigger_block_against_the_record_s_verdict_B1", ["trigger_inconsistent"], None),
    ("kind_of_the_slot_s_role", SCH, [('    return model.get(slot["scope_class"], ("scored_run",)) + ends',
                                       '    return model.get(slot["scope_class"], ("scored_run", "listed_run")) + ends')],
     "validate", "case_input_gates", "family_run_entered_as_listed_run_B2", ["derived_counted_as_run"], None),
    ("sweep_of_record_strict", SCH, [('    if sweep and not sweep["of_record"]:', "    if sweep:")],
     "validate", "case_input_gates", "lambda_of_record_entered_as_listed_run_B2", ["derived_counted_as_run"], None),
    ("non_finite_qat_only", SCH, [('ends = RUN_ENDINGS + (("non_finite",) if slot["record_kind"] == "qat_epoch15" '
                                   "else ())", 'ends = RUN_ENDINGS + ("non_finite",)')],
     "validate", "case_input_gates", "non_finite_entry_on_a_kd_slot", ["derived_counted_as_run"], None),
    ("evaluator_stage_of_the_catalogue", VAL, [('                and e["evaluator_stage"] != s["evaluator_stage"]:',
                                                "                and False:")],
     "validate", "case_input_gates", "evaluator_stage_other_than_the_catalogue_s_B91", ["derived_counted_as_run"], None),
    ("derived_method_of_the_slot", VAL, [('elif e["kind"] == "derived" and e["method"] != s["record_kind"]:',
                                          "elif False:")],
     "validate", "case_input_gates", "derived_method_other_than_the_slot_s_B21", ["derived_counted_as_run"], None),
    # -------------------------------------------------------------------------------------- completion gates
    ("attempt_directories", VAL, [("missing = sorted(r for r in ids if not x.run_dir(r).is_dir())", "missing = []")],
     "validate", "case_completion_gates", "attempt_without_directory", ["launched_attempt_without_directory"], None),
    ("cut_run_directories", VAL, [("for a in e.get(\"attempts\", [])} | _cut_run_ids(x)",
                                   "for a in e.get(\"attempts\", [])}")],
     "validate", "case_cut_attribution", "cut_run_withheld_from_runs_B44", ["launched_attempt_without_directory"],
     None),
    ("runs_needed_in_freeze_mode", VAL, [("    if x.runs_root is None:                      # draft: the absent",
                                          "    if False:                      # draft: the absent")],
     "validate", "case_moved_freeze", "delay_entry_without_runs_refused_at_its_gate",
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
    ("S02_completeness_loop_removed", VAL, [("        for slot, att in sorted(idx.by_slot.items()):",
                                             "        for slot, att in []:")],
     "validate", "case_completion_gates", "complete_kd_run_omitted_AM19_3d", ["complete_run_omitted"], 2),
    ("completeness_over_kd_slots_only", VAL, [("        for slot, att in sorted(idx.by_slot.items()):",
                                               "        for slot, att in sorted((k, v) for k, v in idx.by_slot.items() "
                                               "if x.slot(k)[\"record_kind\"] == \"kd_run_end\"):")],
     "validate", "case_cut_attribution", "complete_e1_run_omitted_B4", ["complete_run_omitted"], None),
    ("launched_run_outside_the_catalogue", VAL, [('bad = [f"{rid} (its records name {slot}, no catalogue run slot)" '
                                                  "for rid, slot in idx.foreign]", "bad = []")],
     "validate", "case_cut_attribution", "launched_run_outside_the_catalogue_B94", ["complete_run_omitted"], None),
    ("entry_runs_of_its_own_slot", VAL, [('wrong = sorted({r for r in _entry_run_ids(e) if idx.slot_of.get(r) != '
                                          'e["slot"]})', "wrong = []")],
     "validate", "case_completion_gates", "run_entry_citing_another_slot_s_run_B52", ["completion_source_mismatch"],
     None),
    ("late_never_cut_run_needs_the_delay", VAL, [("        return None if not late else f\"the never-cut runs",
                                                  "        return None if True else f\"the never-cut runs")],
     "validate", "case_cut_attribution", "late_never_cut_run_absent_without_delay_B3", ["freeze_source_conflict"],
     None),
    ("delay_keeps_the_manifest_a_draft", VAL, [("    if x.freeze.moved_date is None:\n        return f\"the delay",
                                                "    if False:\n        return f\"the delay")],
     "validate", "case_moved_freeze", "delay_without_new_freeze_stays_a_draft_B3", ["freeze_source_conflict"], None),
    ("late_run_entered_at_the_new_freeze", VAL, [("    if absent:\n        return f\"the late never-cut runs",
                                                  "    if False:\n        return f\"the late never-cut runs")],
     "validate", "case_moved_freeze", "late_never_cut_run_absent_at_the_new_freeze_B3", ["freeze_source_conflict"],
     None),
    ("draft_lists_absent_never_cut_runs", VAL, [("absent = [k for k in _never_cut_slots(x) if k not in x.by_slot] "
                                                 "if x.mode == \"draft\" else []", "absent = []")],
     "validate", "case_cut_attribution", "draft_lists_absent_never_cut_and_cuttable_slots_B3",
     ["freeze_source_conflict"], None),
    ("every_slot_entered_at_the_freeze", VAL, [('("cut_unentered_at_unlock", FU, g_cut_unentered_at_unlock)',
                                                '("cut_unentered_at_unlock", U, g_cut_unentered_at_unlock)')],
     "validate", "case_cut_attribution", "cuttable_slot_without_entry_at_the_freeze_B4", ["cut_unentered_at_unlock"],
     None),
    ("kd_attempts_from_the_launch_log", COM, [("                    self._add(found, rid, slot_of_launch(log.launch[rid]),",
                                               "                    if False: self._add(found, rid, "
                                               "slot_of_launch(log.launch[rid]),")],
     "validate", "case_cut_attribution", "launched_kd_run_cut_as_never_launched_B13", ["cut_reason_invalid"], None),
    ("qat_attempts_from_run_meta", COM, [('for rel, allow, key in ((QAT_TELEMETRY, R.QAT_ROWS, "qat"), '
                                          '(E1_TELEMETRY, R.E1_ROWS, "e1")):',
                                          'for rel, allow, key in ((E1_TELEMETRY, R.E1_ROWS, "e1"),):')],
     "validate", "case_cut_attribution", "launched_qat_run_cut_as_never_launched_B12", ["cut_reason_invalid"], None),
    ("teacher_arm_attempts_from_the_work_dir", COM, [('            if prov.get("launch") is not True:\n'
                                                      "                return None",
                                                      "            if True:\n                return None")],
     "validate", "case_cut_attribution", "cut_of_a_complete_arm_teacher_AM18_7b", ["cut_of_complete_run"], None),
    ("slot_status_from_its_own_attempts", COM, [("return list(self.by_slot.get(slot, [])) if self.known(slot) else None",
                                                 "return [] if self.known(slot) else None")],
     "validate", "case_cut_attribution", "group_cut_naming_a_complete_run_first_B0", ["cut_of_complete_run"], None),
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
    ("S03_rule_operator_allowed", ENT, [('    if doc["rule"] not in S.CUT_RULES:\n        return f"rule {doc',
                                         '    if doc["rule"] not in S.CUT_RULES + ["operator"]:\n        return f"rule {doc')],
     "cuts", "case_gate_order", "rule_operator_refused", ["cut_rule_unknown"], 3),
    ("S04_never_cut_test_skipped", ENT, [('bad = [s["slot"] for s in _slots(doc, ctx) if s["never_cut"]]',
                                          "bad = []")],
     "cuts", "case_gate_order", "never_cut_slot_refused", ["cut_of_never_cut"], 4),
    ("complete_run_cut", ENT, [('done = [s["slot"] for s in _slots(doc, ctx) if ctx.status(s["slot"]) == '
                                "C.COMPLETE]", "done = []")],
     "cuts", "case_gate_order", "complete_run_cut_refused", ["cut_of_complete_run"], None),
    ("never_launched_reason", ENT, [('if doc["reason"] not in S.NEVER_LAUNCHED_REASONS:', "if False:")],
     "cuts", "case_gate_order", "never_launched_reason_outside_list", ["cut_reason_invalid"], None),
    ("cut_day", ENT, [('    if want != doc["date"]:\n        return f"the cut is dated',
                       '    if False:\n        return f"the cut is dated')],
     "cuts", "case_gate_order", "freeze_cut_dated_off_its_day", ["cut_date_mismatch"], None),
    ("cut_entered_early", ENT, [("    if entered is not None and entered < want:", "    if False:")],
     "validate", "case_cut_gates", "cut_entered_before_its_rule_s_day_B75", ["cut_date_mismatch"], None),
    ("cut_results_known_by_its_day", ENT, [('if ctx.known_by(r["path"], r["sha256"], doc["date"]) is False]',
                                            "if False]")],
     "validate", "case_cut_gates", "results_known_committed_after_the_cut_s_day_B69", ["cut_results_unknown"], None),
    ("results_held_by_the_day", REC, [("if ss.meets(int(t), until) else None", "if True else None")],
     "validate", "case_cut_gates", "results_known_committed_after_the_cut_s_day_B69", ["cut_results_unknown"], None),
    ("arm_records_listed", ENT, [("        missing = [rel for rel in arm_records(g, ctx) if rel not in listed and "
                                  'ctx.known_by(rel, None, doc["date"])]', "        missing = []")],
     "validate", "case_cut_gates", "arm_cut_without_its_teacher_record_known_by_its_day_B69",
     ["arm_cut_results_missing"], None),
    ("arm_checks_per_group", ENT, [('    return [g for g in ctx.cat.groups if g in ARM_GROUPS and any(s["cut_group"] == g '
                                    "for s in _slots(doc, ctx))]",
                                    '    return [g for g in ctx.cat.groups if g in ARM_GROUPS and all(s["cut_group"] == g '
                                    "for s in _slots(doc, ctx))]")],
     "cuts", "case_gate_order", "two_group_named_rule_cut_of_a_running_arm_run_B15", ["arm_run_cut_while_running"],
     None),
    ("S05_arithmetic_ge", PRE, [("    cut = needed > left\n", "    cut = needed >= left\n")],
     "cuts", "case_arithmetic", "arithmetic_equality_fits", ["cut_arithmetic_invalid"], 5),
    ("arithmetic_recomputed", ENT, [("        elif x != y:\n            return",
                                     "        elif False:\n            return")],
     "cuts", "case_arithmetic", "arithmetic_recomputation_catches_cut", ["cut_arithmetic_invalid"], None),
    # --------------------------------------------- the arithmetic's inputs are the slot's (DL-24 #11 #111 #68)
    ("arithmetic_iterations_of_the_slot", ENT, [('    if want != [a["iterations"]]:', "    if False:")],
     "validate", "case_moved_freeze", "arithmetic_iterations_not_the_slot_s_B11", ["cut_arithmetic_invalid"], None),
    ("arithmetic_reference_read_again", VAL, [('        if any(got[k] != ref[k] for k in ("path", "sha256", '
                                               '"median", "detail")):', "        if False:")],
     "validate", "case_moved_freeze", "arithmetic_reference_read_again_differs_B11", ["cut_arithmetic_invalid"],
     None),
    ("arithmetic_reference_the_most_recent", VAL, [("                if rid not in latest:\n                    return",
                                                    "                if False:\n                    return")],
     "validate", "case_moved_freeze", "arithmetic_reference_not_the_most_recent_run_B11", ["cut_arithmetic_invalid"],
     None),
    ("latest_finished_the_latest", VAL, [("    return {r for r, t in done.items() if t == last}",
                                          "    return set(done)")],
     "validate", "case_moved_freeze", "arithmetic_reference_not_the_most_recent_run_B11", ["cut_arithmetic_invalid"],
     None),
    ("printer_reference_the_most_recent", CLI, [("            if rid not in latest:\n                raise",
                                                 "            if False:\n                raise")],
     "validate", "case_moved_freeze", "printer_arithmetic_reference_not_the_most_recent_B11",
     ["cut_arithmetic_invalid"], None),
    ("printer_iterations_of_the_slot", CLI, [("    if iterations is None or args.iterations not in (None, iterations):",
                                              "    if iterations is None:")],
     "validate", "case_moved_freeze", "printer_arithmetic_iterations_not_the_horizon_B11", ["cut_arithmetic_invalid"],
     None),
    ("reference_read_as_its_slot", ENT, [("    st = C.run_state(slot, d)\n    if st.status != C.COMPLETE:",
                                          '    st = C.run_state(dict(slot, stage="E3"), d)\n    if st.status != '
                                          "C.COMPLETE:")],
     "cuts", "case_references", "reference_of_another_stage_unread_B111", ["completion_record_invalid"], None),
    ("reference_smoke_not_an_object", ENT, [("except (pa.GateError, OSError, ValueError, AttributeError, TypeError) "
                                             "as e:", "except (pa.GateError, OSError, ValueError) as e:")],
     "cuts", "case_references", "reference_smoke_holding_a_list_B68", ["cut_arithmetic_invalid"], None),
    ("launched_cut_records", ENT, [('if missing or doc["comparator"] is None:', "if False:")],
     "cuts", "case_gate_order", "launched_cut_without_telemetry", ["cut_val_record_missing"], None),
    ("arm_results_known", ENT, [("        if not listed:\n            return f\"arm {g}'s cut lists",
                                 "        if False:\n            return f\"arm {g}'s cut lists")],
     "cuts", "case_gate_order", "arm_cut_without_results", ["arm_cut_results_missing"], None),
    ("arm_running", ENT, [('    if running:\n        return (f"{running}: a launched arm run',
                           '    if False:\n        return (f"{running}: a launched arm run')],
     "cuts", "case_gate_order", "launched_arm_cut_by_named_rule_refused", ["arm_run_cut_while_running"], None),
    ("arm_partial", ENT, [("if not cut or cut != runs[first:] or any(", "if not cut or any(")],
     "cuts", "case_gate_order", "arm_cut_missing_its_later_run", ["arm_cut_partial"], None),
    ("cut_order_position", ENT, [('return None if pos == want else f"cut_order_position',
                                  'return None if True else f"cut_order_position')],
     "cuts", "case_gate_order", "freeze_cut_wrong_position", ["cut_order_position_mismatch"], None),
    ("decision_record_cuts", ENT, [("if rec is None or not vals <= cut_values:", "if rec is None:")],
     "cuts", "case_gate_order", "decision_date_cut_not_in_the_record", ["decision_date_cut_mismatch"], None),
    ("stopped_run", ENT, [('if not stopped or (doc["rule"] == "freeze" and doc["date"] == ctx.freeze.original_date)',
                           'if True or (doc["rule"] == "freeze" and doc["date"] == ctx.freeze.original_date)')],
     "cuts", "case_gate_order", "stopped_run_cut_by_named_rule_refused", ["stopped_run_cut_early"], None),
    ("stopped_candidate_at_its_decision_date", ENT, [('            or doc["rule"] == "decision_date":\n        return None',
                                                      "            or False:\n        return None")],
     "cuts", "case_gate_order", "stopped_candidate_cut_at_its_decision_date_AM19_2b", ["stopped_run_cut_early"], None),
    ("launched_attempts_named", ENT, [("    if omitted:\n        return f\"the slots' launched attempts",
                                       "    if False:\n        return f\"the slots' launched attempts")],
     "validate", "case_cut_attribution", "launched_kd_run_cut_as_never_launched_B13", ["cut_reason_invalid"], None),
    ("run_ids_of_the_cut_slots", ENT, [('if owner is not None and owner not in doc["slots"]:', "if False:")],
     "validate", "case_cut_attribution", "cut_naming_another_slot_s_run_B5", ["cut_reason_invalid"], None),
    ("run_ids_launched", ENT, [("        if owner is None and not unread:", "        if False:")],
     "cuts", "case_gate_order", "run_id_no_record_launched", ["cut_reason_invalid"], None),
    ("arm_registry", ENT, [("missing = [st for st in stages if st not in ctx.registry]", "missing = []")],
     "cuts", "case_gate_order", "arm_without_registry_file_refused", ["unregistered_arm"], None),
    ("condition_no", ENT, [('or rec["verdict"] != "no":', 'or rec["verdict"] not in ("no", "yes"):')],
     "cuts", "case_gate_order", "e6x_cut_citing_a_yes_condition_refused", ["condition_unmet"], None),
    # ------------------------------------------------------------------ named rules (AM-19 item 3(b); DL-24 #16)
    ("named_rule_record_kind", ENT, [('    if rec is not None and rec["kind"] not in NAMED_RULE_KINDS:',
                                      "    if False:")],
     "validate", "case_cut_gates", "named_rule_citing_an_unknown_record_B16", ["cut_rule_unknown"], None),
    ("named_rule_record_date", ENT, [('    if rr is not None and rr["date"] != want:', "    if False:")],
     "validate", "case_cut_gates", "named_rule_record_dated_otherwise_B16", ["cut_date_mismatch"], None),
    ("rule_day_from_the_record", ENT, [('        return rec["date"] if rec else (rr["date"] if rr else None)',
                                        '        return rr["date"] if rr else None')],
     "cuts", "case_gate_order", "named_rule_cut_dated_by_its_entry_not_its_record_B16", ["cut_date_mismatch"], None),
    ("condition_of_its_arm", ENT, [('if [s["stage"] for s in _slots(doc, ctx)] != [f"E6-{arm}"] * len(doc["slots"]) '
                                    'or rec["verdict"] != "no":', 'if rec["verdict"] != "no":')],
     "cuts", "case_gate_order", "e6x_cut_citing_another_arm_s_no_condition_B16", ["condition_unmet"], None),
    ("condition_e3x_run_of_record", ENT, [("        if got is False or (got and (", "        if False and (got and (")],
     "validate", "case_cut_gates", "e6x_cut_without_an_e3x_run_of_record_B16", ["condition_unmet"], None),
    ("e3x_from_the_manifest", VAL, [('if e is not None and e["kind"] == "scored_run" else False',
                                     'if e is not None and e["kind"] == "scored_run" else None')],
     "validate", "case_cut_gates", "e6x_cut_without_an_e3x_run_of_record_B16", ["condition_unmet"], None),
    ("go_rule_of_its_arm", ENT, [('if rec["verdict"] != "no_go" or groups != [rec["arm"]]:',
                                  'if rec["verdict"] != "no_go":')],
     "validate", "case_cut_gates", "go_rule_no_go_of_another_arm_B16", ["condition_unmet"], None),
    ("go_rule_no_go", ENT, [('if rec["verdict"] != "no_go" or groups != [rec["arm"]]:', 'if groups != [rec["arm"]]:')],
     "cuts", "case_gate_order", "go_rule_reading_go_refused_B16", ["condition_unmet"], None),
    ("go_rule_record_read", ENT, [('if isinstance(doc, dict) and doc.get("format") == _pa().GO_RULE_FORMAT:',
                                   "if False:")],
     "validate", "case_cut_gates", "launched_arm_cut_by_a_named_rule", ["arm_run_cut_while_running"], None),
    ("condition_entry_read", ENT, [('and doc.get("kind") == "condition" \\\n', 'and doc.get("kind") == "conditions" \\\n')],
     "validate", "case_cut_gates", "e6x_cut_citing_a_yes_condition_AM23_3b", ["condition_unmet"], None),
    ("rule_records_committed", VAL, [("    cited += x.rule_records()", "    cited += []")],
     "validate", "case_cut_gates", "named_rule_record_at_another_sha256_B16", ["record_uncommitted"], None),
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
    ("fault_report", MAN, [('if not ev or any(e.get("fault_report") is None for e in ev[1:]):', "if not ev:")],
     "validate", "case_parent_and_derived_gates", "second_evaluation_without_report", ["derived_selection_conflict"],
     None),
    # ------------------- the rules file, DL-10, appended selections, fault reports (DL-24 #18-#20 #24 #30 #40 #63 #77)
    ("qat_rules_of_record", VAL, [('                for e in qat if e["inputs"]["rules_sha256"] != rules]',
                                   "                for e in qat if False]")],
     "validate", "case_record_gates", "qat_rules_sha256_not_the_committed_file_B18", ["derived_inputs_incomplete"],
     None),
    ("builder_rules_of_record", MAN, [('        if x["inputs"].setdefault("rules_sha256", rules) != rules:',
                                       "        if False:")],
     "validate", "case_record_gates", "builder_refuses_another_rules_sha256_B19", ["derived_inputs_incomplete"], None),
    ("builder_reads_the_rules", MAN, [('x["inputs"].setdefault("rules_sha256", rules)',
                                       'x["inputs"].setdefault("rules_sha256", None)')],
     "validate", "case_record_gates", "builder_reads_the_rules_sha256_TB13", [], None),
    ("calibration_lists_in_dl10", MAN, [('        if item["file_sha256"] not in (dl10_text or ""):', "        if False:")],
     "validate", "case_record_gates", "calibration_list_not_dl10_s_B30", ["calibration_index_mismatch"], None),
    ("calibration_checksum_in_dl10", MAN, [('    if str(ci["of_record"]["checksum_sha256"]) not in (dl10_text or ""):',
                                            "    if False:")],
     "validate", "case_record_gates", "calibration_checksum_not_dl10_s_B30", ["calibration_index_mismatch"], None),
    ("calibration_seed_its_key", MAN, [('if doc.get("seed") != seed or doc.get("role")',
                                        'if False or doc.get("role")')],
     "validate", "case_record_gates", "calibration_list_seed_not_its_key_B40", ["calibration_index_mismatch"], None),
    ("calibration_role_of_its_list", MAN, [('or doc.get("role") != CALIBRATION_ROLES.get(seed):', ":")],
     "validate", "case_record_gates", "calibration_list_role_not_am16_s_B40", ["calibration_index_mismatch"], None),
    ("appended_selection_conflict", VAL, [("            errs += M.selection_conflict_errors(M.appended_entry(app, listed))",
                                           "            pass")],
     "validate", "case_parent_and_derived_gates", "appended_qat_selection_second_evaluation_without_report_B24",
     ["derived_selection_conflict"], None),
    ("fault_reports_committed", VAL, [("    cited += x.fault_reports()", "    cited += []")],
     "validate", "case_parent_and_derived_gates", "qat_fault_report_not_committed_B77", ["record_uncommitted"], None),
    ("cli_second_append", CLI, [('    if any(isinstance(a, dict) and a.get("slot") == args.slot for a in doc["appends"]):',
                                 "    if False:")],
     "paths", "case_cli_outputs", "cli_append_refuses_a_second_append_of_its_slot_B63", ["append_invalid"], None),
    ("cli_append_reads_a_manifest", CLI, [("            and isinstance(doc.get(\"appends\"), list)):\n        raise "
                                           "S.TmanRefused(\"manifest_format\"",
                                           "            and isinstance(doc.get(\"appends\"), list)) and False:\n        "
                                           "raise S.TmanRefused(\"manifest_format\"")],
     "paths", "case_cli_outputs", "cli_append_refuses_a_file_that_is_no_manifest_B63", ["manifest_format"], None),
    ("non_finite_flags", MAN, [('elif len(ends) != 15 or any(r.get("state_finite") is not False for r in ends):',
                                "elif len(ends) != 15:")],
     "validate", "case_non_finite_gate", "non_finite_with_a_convertible_epoch", ["non_finite_invalid"], None),
    ("S07_isolation_test_dropped", MAN, [('elif not structural and not (ptq_record_ok and '
                                          'frozen["method"] == "ptq"):', "elif False:")],
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
    ("lane_register_lists_a_lane", VAL, [('if not doc["lanes"]:\n        return f"{S.LANES_REL} lists no lane',
                                          'if False:\n        return f"{S.LANES_REL} lists no lane')],
     "unlock", "case_unlock_gates", "lane_register_listing_no_lane_B55a", ["lane_unmet"], None),
    ("lane_register_dl_recorded", VAL, [('if not x.dl.recorded_rows_with(sha):\n        errs.append(f"{S.LANES_REL}',
                                         'if False:\n        errs.append(f"{S.LANES_REL}')],
     "unlock", "case_unlock_gates", "lane_register_in_no_decision_log_row", ["lane_unmet"], None),
    ("lane_register_row_recorded", VAL, [('if not x.dl.recorded_rows_with(sha):\n        errs.append(f"{S.LANES_REL}',
                                          'if sha not in x.dl.text:\n        errs.append(f"{S.LANES_REL}')],
     "unlock", "case_unlock_gates", "lane_register_in_a_DECIDED_row", ["lane_unmet"], None),
    ("implemented_row_names_the_lane", VAL, [('for n in (lane["lane"], st["merge_commit"][:7])',
                                              'for n in (st["merge_commit"][:7],)')],
     "unlock", "case_unlock_gates", "implemented_row_of_another_lane_B55b", ["lane_unmet"], None),
    ("implemented_row_names_the_merge", VAL, [('for n in (lane["lane"], st["merge_commit"][:7])',
                                               'for n in (lane["lane"],)')],
     "unlock", "case_unlock_gates", "implemented_row_naming_another_merge_commit", ["lane_unmet"], None),
    ("not_required_slots_entered_cut", VAL, [("        if scored:\n            errs.append(f\"lane {lane['lane']}: "
                                              "serves {scored}, not entered cut",
                                              "        if False:\n            errs.append(f\"lane {lane['lane']}: "
                                              "serves {scored}, not entered cut")],
     "unlock", "case_unlock_gates", "not_required_lane_serving_a_scored_slot", ["not_required_invalid"], None),
    ("not_required_cut_cited", VAL, [('"cut"\n                        or x.by_slot[k]["entry"]["path"] not in cuts)',
                                      '"cut")')],
     "unlock", "case_unlock_gates", "not_required_lane_citing_another_cut_of_its_slot", ["not_required_invalid"],
     None),
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
     "paths", "case_cli_outputs", "a_second_write_is_refused_and_the_first_kept", ["output_exists"], None),
    ("out_inside_the_repository", CLI, [("if p == repo or repo in p.parents:", "if p == repo:")],
     "paths", "case_cli_outputs", "out_in_the_repository_at_root_refused", ["output_location"], None),
    ("output_exists_code_of_the_cli", CLI, [('raise S.TmanRefused("output_exists", f"{out} exists: outputs are',
                                             'raise S.TmanRefused("entry_file_rewritten", f"{out} exists: outputs are')],
     "paths", "case_cli_outputs", "a_second_build_is_refused_and_the_first_kept_B110", ["output_exists"], None),
    ("output_exists_code_of_the_printer", ENT, [('raise S.TmanRefused("output_exists", f"{out} exists: an output',
                                                 'raise S.TmanRefused("entry_file_rewritten", f"{out} exists: an output')],
     "cuts", "case_printers_and_committed_files", "printer_writes_once_B110", ["output_exists"], None),
    ("output_location_code", CLI, [('raise S.TmanRefused("output_location", f"--out {p} is inside the repository',
                                    'raise S.TmanRefused("test_path", f"--out {p} is inside the repository')],
     "paths", "case_cli_outputs", "out_in_the_repository_at_root_refused", ["output_location"], None),
    ("write_entry_checks_the_created_name_only", ENT, [("    err = S.created_name_error(name)\n    if err:\n        "
                                                        'raise S.TmanRefused("test_path", f"refusing to create {name}',
                                                        "    err = S.created_name_error(Path(out_dir) / name)\n    "
                                                        'if err:\n        raise S.TmanRefused("test_path", '
                                                        'f"refusing to create {name}')],
     "cuts", "case_printers_and_committed_files", "printer_writes_into_a_folder_whose_path_holds_a_colon_B72",
     ["test_path"], None),
    ("make_entry_common_keys", ENT, [("    if common or not isinstance(fields, dict):",
                                      "    if not isinstance(fields, dict):")],
     "validate", "case_printers", "printer_fields_naming_a_common_key_B71", ["entry_field"], None),
    # -------------------------------------------- the printers run their kind's checks (DL-24 #8 #9 #68 #74)
    ("printer_checks_run", CLI, [('    ENTRY_CHECKS[doc["kind"]](args, records, cat, freeze, doc)', "    pass")],
     "validate", "case_moved_freeze", "printer_freeze_delay_listing_a_complete_run_B8", ["freeze_source_conflict"],
     None),
    ("printer_late_runs_from_the_records", CLI, [("errs = E.freeze_delay_errors(doc, freeze, "
                                                  "V.late_never_cut(inst, freeze.original_utc))",
                                                  'errs = E.freeze_delay_errors(doc, freeze, sorted(doc["runs"]))')],
     "validate", "case_moved_freeze", "printer_freeze_delay_listing_a_complete_run_B8", ["freeze_source_conflict"],
     None),
    ("freeze_delay_runs", ENT, [('    if not late or sorted(doc["runs"]) != late:', "    if False:")],
     "validate", "case_moved_freeze", "delay_listing_a_complete_run_too_B93", ["freeze_source_conflict"], None),
    ("new_freeze_first_day_of_the_printer", ENT, [('if want is None or doc["new_date"] != want or doc["date"] != '
                                                   'doc["new_date"]:', "if want is None:")],
     "validate", "case_moved_freeze", "printer_new_freeze_a_day_late_B8", ["freeze_source_conflict"], None),
    ("printer_not_run_slots", CLI, [('    E.check_slots(doc, cat, kinds_of_class=("run", "derived"))', "    pass")],
     "validate", "case_printers", "printer_not_run_naming_an_unknown_slot_B68", ["slot_unknown"], None),
    ("printer_not_required_coverage", CLI, [('    if set(doc["slots"]) - covered:', "    if False:")],
     "validate", "case_printers", "printer_not_required_slot_without_a_cut_entry_B9", ["not_required_invalid"], None),
    ("printer_not_required_cut_kind", CLI, [('cut, sha = E.read_entry_file(records, ref["path"], "cut")',
                                             'cut, sha = E.read_entry_file(records, ref["path"])')],
     "validate", "case_printers", "printer_not_required_citing_a_not_run_entry_as_a_cut", ["entry_file_invalid"],
     None),
    ("printer_unlock_manifest_file", CLI, [('or S.sha256_bytes(data) != m["sha256"] \\', "or False \\")],
     "unlock", "case_unlock_printer", "printer_unlock_of_another_manifest_file_B9", ["run_added_after_freeze"], None),
    ("printer_unlock_freeze_record_body", CLI, [('    if fsha != doc["freeze_record"]["sha256"] or fr["body_sha256"] '
                                                 '!= m["body_sha256"]:',
                                                 '    if fsha != doc["freeze_record"]["sha256"]:')],
     "unlock", "case_unlock_printer", "printer_unlock_citing_the_freeze_record_of_another_body_B9",
     ["run_added_after_freeze"], None),
    ("printer_unlock_lanes", CLI, [('    if doc["lanes"] != {"path": S.LANES_REL, "sha256": '
                                    "S.sha256_bytes(records.read(S.LANES_REL))}:", "    if False:")],
     "unlock", "case_unlock_printer", "printer_unlock_citing_another_lane_register_B9", ["lane_unmet"], None),
    ("printer_unlock_second", CLI, [('    if rows:\n        raise S.TmanRefused("second_session"',
                                     '    if False:\n        raise S.TmanRefused("second_session"')],
     "unlock", "case_unlock_printer", "printer_unlock_after_an_unlock_entry_B9", ["second_session"], None),
    ("printer_condition_slot", CLI, [("    if doc[\"slots\"] != [f\"E6-{doc['arm']}:s42:15ep\"]:", "    if False:")],
     "validate", "case_printers", "printer_condition_naming_another_arm_s_slot_B16", ["entry_field"], None),
    ("printer_not_made_gated", CLI, [('    bad = [k for k in doc["slots"] if not M.code_gated(k, cat)]', "    bad = []")],
     "validate", "case_printers", "printer_not_made_for_a_slot_without_a_code_gate_B9", ["derived_missing"], None),
    ("printer_not_made_class", CLI, [('    E.check_slots(doc, cat, kinds_of_class=("derived",))', "    pass")],
     "validate", "case_printers", "printer_not_made_naming_a_run_slot", ["derived_counted_as_run"], None),
    ("printer_freeze_record_body", CLI, [('    if doc["body_sha256"] != S.canonical_sha256(manifest_body(args, '
                                          '"run_added_after_freeze")):', "    if False:")],
     "validate", "case_printers", "printer_freeze_record_of_another_body_B9", ["run_added_after_freeze"], None),
    ("printer_clock_offset", CLI, [('    if abs(doc["offset_seconds"]) > S.CLOCK_OFFSET_LIMIT_S:', "    if False:")],
     "validate", "case_printers", "printer_clock_offset_over_60_s_B9", ["clock_offset_invalid"], None),
    ("printer_named_rule_record", CLI, [("rule_record=E.rule_record_reader(records), ", "")],
     "validate", "case_printers", "printer_named_rule_cut_citing_an_unknown_record_B16", ["cut_rule_unknown"], None),
    ("teacher_logs_unsorted", COM, [("logs = [log_rel] if log_rel else sorted(",
                                     "logs = [log_rel] if log_rel else list(")],
     "determinism", "case_listing_order", "shuffled_listing_4_byte_identical", [], None),
    # ------ derived inputs against their sources; hashed forms against their records (DL-24 #22 #27 #34 #49 #50 #53)
    ("inputs_parent_checkpoint", MAN, [('        if inp["parent_checkpoint_sha256"] != (pe.get("checkpoint") or {})'
                                        '.get("sha256"):', "        if False:")],
     "validate", "case_derived_sources", "derived_parent_checkpoint_not_its_parent_s_B22",
     ["derived_inputs_mismatch"], None),
    ("inputs_calibration_list", MAN, [('if ci is not None and inp["calibration_list"] != {"path": item.get("path"),',
                                       'if False and inp["calibration_list"] != {"path": item.get("path"),')],
     "validate", "case_derived_sources", "derived_calibration_list_not_the_index_s_B22", ["derived_inputs_mismatch"],
     None),
    ("inputs_script_path", MAN, [('if script is not None and inp["script"]["path"] != script:', "if False:")],
     "validate", "case_derived_sources", "derived_script_not_its_ptq_script_B53", ["derived_inputs_mismatch"], None),
    ("inputs_commit_in_history", MAN, [("    if not (src.commit_exists(commit) and src.is_ancestor(commit, "
                                        "src.records_ref)):", "    if False:")],
     "validate", "case_derived_sources", "derived_script_commit_outside_the_records_history",
     ["derived_inputs_mismatch"], None),
    ("inputs_qat_telemetry", MAN, [('        if inp["telemetry_sha256"] != (pe.get("completion") or {}).get("sha256"):',
                                    "        if False:")],
     "validate", "case_derived_sources", "qat_telemetry_not_its_run_s_completion_B22", ["derived_inputs_mismatch"],
     None),
    ("ptq_script_of_an_arm", MAN, [('return PTQ_ARM_SCRIPT if "arm" in cat.slots[slot]["roles"] else PTQ_SCRIPT',
                                    "return PTQ_SCRIPT")],
     "derived", "case_inputs_and_parents", "ptq_script_of_an_arm_s_e7_is_run_ptq_arm_B53",
     ["derived_inputs_mismatch"], None),
    ("ptq_script_of_a_lane_s_slot", MAN, [("    if code_gated(slot, cat):\n        return None",
                                           "    if False:\n        return None")],
     "derived", "case_inputs_and_parents", "ptq_script_of_a_slot_whose_code_is_not_merged_is_its_lane_s", [], None),
    ("hashed_files_cited", MAN, [('        if sha != h[k]["sha256"]:\n            bad.append(k)',
                                  "        if False:\n            bad.append(k)")],
     "validate", "case_derived_sources", "ptq_provenance_not_the_cited_sha256_B50", ["derived_hash_mismatch"], None),
    ("hashed_selected_checkpoint_read", MAN, [('        reads["selected_checkpoint"] = None', "        pass")],
     "validate", "case_derived_sources", "qat_selected_checkpoint_absent_B49", ["derived_hash_mismatch"], None),
    ("ptq_provenance_parent", MAN, [('        want = {"source_checkpoint_sha256": inp["parent_checkpoint_sha256"],\n',
                                     "        want = {\n")],
     "validate", "case_derived_sources", "ptq_provenance_naming_other_parent_B27", ["derived_hash_mismatch"], None),
    ("ptq_provenance_index_fields", MAN, [("        if ci is not None:\n            item = calibration_item(ci, s)",
                                           "        if False:\n            item = calibration_item(ci, s)")],
     "validate", "case_derived_sources", "ptq_provenance_naming_other_checksum_B27", ["derived_hash_mismatch"], None),
    ("ptq_provenance_artifact", MAN, [('"code.commit": inp["script"]["commit"], "converted_artifact_sha256": '
                                       'h["artifact_sha256"],', '"code.commit": inp["script"]["commit"],')],
     "validate", "case_derived_sources", "ptq_provenance_naming_other_artifact_B27", ["derived_hash_mismatch"], None),
    ("qat_record_winner_checkpoint", MAN, [('            "winner.checkpoint_sha256": h["selected_checkpoint"]'
                                            '["sha256"],\n', "")],
     "validate", "case_derived_sources", "qat_record_naming_winner_of_another_checkpoint_B34",
     ["derived_hash_mismatch"], None),
    ("qat_record_telemetry", MAN, [('"telemetry_sha256": inp["telemetry_sha256"], "clip_norm": meta.get("clip_norm"), ',
                                    '"clip_norm": meta.get("clip_norm"), ')],
     "validate", "case_derived_sources", "qat_record_naming_other_telemetry_B34", ["derived_hash_mismatch"], None),
    ("qat_record_clip_norm_of_its_run", MAN, [('"clip_norm": meta.get("clip_norm"), "winner.run_id": rid,',
                                               '"winner.run_id": rid,')],
     "validate", "case_derived_sources", "qat_record_naming_other_clip_norm_B34", ["derived_hash_mismatch"], None),
    ("qat_selected_checkpoint_row", MAN, [('    if row is None or h["selected_checkpoint"] != {"path": ',
                                           '    if False and h["selected_checkpoint"] != {"path": ')],
     "validate", "case_derived_sources", "qat_record_naming_winner_of_another_epoch_B34", ["derived_hash_mismatch"],
     None),
    ("clip_selection_cited", MAN, [('    if csha != inp["clip_selection"]["sha256"] or ', "    if False or ")],
     "validate", "case_derived_sources", "clip_selection_not_the_cited_sha256_B34", ["derived_hash_mismatch"], None),
    ("clip_selection_rules", MAN, [('            or _dotted(clip, "rules_sha256") != inp["rules_sha256"]:',
                                    "            or False:")],
     "validate", "case_derived_sources", "clip_selection_of_other_rules_B34", ["derived_hash_mismatch"], None),
    ("pilot_winner_TB9", MAN, [('    elif cat.slots[slot]["pilot"] and won != (rid, inp["telemetry_sha256"], '
                                'h["record"]["sha256"]):', "    elif False:")],
     "validate", "case_derived_sources", "pilot_selection_of_the_retained_loser_TB9_B34", ["derived_hash_mismatch"],
     None),
    ("pilot_winner_record_TB9", MAN, [('           _dotted(clip, "winner.qat_selection.sha256"))',
                                       '           h["record"]["sha256"])')],
     "validate", "case_derived_sources", "pilot_winner_naming_another_record_TB9", ["derived_hash_mismatch"], None),
    ("clip_norm_of_the_winner", MAN, [('    elif _dotted(clip, "winner.clip_norm") != _dotted(rec, "clip_norm"):',
                                       "    elif False:")],
     "validate", "case_derived_sources", "run_clip_not_the_clip_winner_s_B34", ["derived_hash_mismatch"], None),
    ("derived_hash_of_appends", VAL, [("            made.append(M.appended_entry(app, listed))", "            pass")],
     "validate", "case_appends", "appended_model_whose_provenance_names_another_parent_B27",
     ["derived_hash_mismatch"], None),
    # ------------------- the made instant, the structural first pass, the appends' history (DL-24 #25 #26 #28 #54)
    ("append_made_by_the_freeze", MAN, [("    if D.meets(made, freeze_utc):\n        errs.append(",
                                         "    if False:\n        errs.append(")],
     "validate", "case_appends", "append_made_by_the_freeze_B25", ["append_invalid"], None),
    # the first pass as it was (#26): the ptq_record stand-in for a PTQ append, none for a QAT selection
    ("append_first_pass_structural", VAL, [("first = M.append_errors(a, frozen, x.freeze.original_utc, "
                                            "structural=True)",
                                            "first = M.append_errors(a, frozen, x.freeze.original_utc, "
                                            "ptq_record_ok=True)")],
     "validate", "case_appends", "post_freeze_qat_append_with_its_isolation_record_passes_B26", ["append_invalid"],
     None),
    ("cli_append_first_pass_structural", CLI, [("errs = M.append_errors(append, frozen, freeze.original_utc, "
                                                "structural=True)",
                                                "errs = M.append_errors(append, frozen, freeze.original_utc, "
                                                "ptq_record_ok=True)")],
     "paths", "case_cli_outputs", "cli_append_of_a_qat_selection_made_after_the_freeze_B26", ["append_invalid"],
     None),
    ("ptq_made_instant", MAN, [('        if not _same_second(_dotted(docs["provenance"], "created_utc"), '
                                'h["made_utc"]):\n            bad.append(',
                                "        if False:\n            bad.append(")],
     "validate", "case_appends", "appended_made_utc_not_its_provenance_s_B25", ["derived_hash_mismatch"], None),
    ("qat_made_instant", MAN, [('    if not _same_second(_dotted(docs["provenance"], "created_utc"), '
                                'h["made_utc"]):\n        bad.append(', "    if False:\n        bad.append(")],
     "validate", "case_derived_sources", "qat_provenance_made_at_another_time_B25_B50", ["derived_hash_mismatch"],
     None),
    ("qat_provenance_artifact", MAN, [('    pwant = {"converted_artifact_sha256": h["artifact_sha256"], ',
                                       "    pwant = {")],
     "validate", "case_derived_sources", "qat_provenance_of_another_artifact_B25_B50", ["derived_hash_mismatch"],
     None),
    ("qat_provenance_epoch", MAN, [('"training.run_id": rid, "training.epoch": epoch}', '"training.run_id": rid}')],
     "validate", "case_derived_sources", "qat_provenance_of_another_epoch_B25_B50", ["derived_hash_mismatch"], None),
    ("appends_history_pairs", MAN, [("            err = None if before is None else appends_prefix_error(before, "
                                     "now or [])", "            err = None")],
     "validate", "case_appends_history", "committed_append_rewritten_in_a_later_commit_B28", ["append_invalid"],
     None),
    ("appends_history_newest", MAN, [("    err = None if newest is None else appends_prefix_error(newest, appends)",
                                      "    err = None")],
     "validate", "case_appends_history", "candidate_appends_behind_the_committed_B54", ["append_invalid"], None),
    ("validator_reads_the_appends_history", VAL, [('    errs, slots = M.appends_history_errors(x.records, '
                                                   'x.doc["appends"]), set()', "    errs, slots = [], set()")],
     "validate", "case_appends_history", "candidate_rewriting_a_committed_append_B54", ["append_invalid"], None),
    ("cli_append_history", CLI, [('    errs = M.appends_history_errors(records, doc["appends"])', "    errs = []")],
     "paths", "case_cli_outputs", "cli_append_to_a_manifest_behind_the_committed_appends_B54", ["append_invalid"],
     None),
    ("cli_append_derived_hash", CLI, [("    if args.runs is not None:\n        errs = M.derived_hash_errors(",
                                       "    if False:\n        errs = M.derived_hash_errors(")],
     "paths", "case_cli_outputs", "cli_append_made_utc_not_its_provenance_s_B25", ["derived_hash_mismatch"], None),
    # ------------------------- a checkpoint's committed source; an arm teacher's records (DL-24 #35 #39 #51)
    ("checkpoint_held_by_a_committed_source", VAL, [("        if not held:\n            errs.append(f\"{e['slot']}: no "
                                                     "committed source", "        if False:\n            errs.append("
                                                     "f\"{e['slot']}: no committed source")],
     "validate", "case_checkpoint_sources", "checkpoint_without_a_committed_source_B35",
     ["checkpoint_hash_mismatch"], None),
    ("checkpoint_source_row_recorded", VAL, [('        held = any(src["dl"] in rows for src in ck["sources"])',
                                              '        held = any(src["dl"] for src in ck["sources"])')],
     "validate", "case_checkpoint_sources", "checkpoint_source_row_not_recorded_B51", ["checkpoint_hash_mismatch"],
     None),
    ("checkpoint_held_by_its_selection_file", VAL, [("                held = held or bool(cand)",
                                                     "                pass")],
     "validate", "case_checkpoint_sources", "default_candidate_held_by_its_selection_file_passes", [], None),
    ("arm_teacher_records_read", VAL, [('        if e["kind"] == "arm_teacher":\n            docs = ',
                                        "        if False:\n            docs = ")],
     "validate", "case_arm_teacher", "arm_teacher_go_rule_verdict_not_the_record_s_B39", ["checkpoint_hash_mismatch"],
     None),
    ("arm_record_selection_checkpoint", MAN, [('            "selection.checkpoint.sha256": ck["sha256"], '
                                               '"selection.recorded_checkpoint_sha256": ck["sha256"],\n', "")],
     "validate", "case_arm_teacher", "arm_teacher_checkpoint_not_the_arm_record_s_selection_B39",
     ["checkpoint_hash_mismatch"], None),
    ("arm_record_complete", MAN, [('"arm": arm, "status": "complete", ', '"arm": arm, ')],
     "validate", "case_arm_teacher", "arm_teacher_record_incomplete_B39", ["checkpoint_hash_mismatch"], None),
    ("arm_record_final_selection", MAN, [('"status": "complete", "selection.status": "final",',
                                          '"status": "complete",')],
     "validate", "case_arm_teacher", "arm_teacher_without_a_final_selection_B39", ["checkpoint_hash_mismatch"], None),
    ("arm_record_attempt", MAN, [('"work_dir.name": rid, "work_dir.attempt": attempt,', '"work_dir.name": rid,')],
     "validate", "case_arm_teacher", "arm_teacher_record_of_another_attempt_B39", ["checkpoint_hash_mismatch"],
     None),
    ("arm_launch_provenance_arm", MAN, [('lwant = {"format": LAUNCH_FORMAT, "arm": arm, ',
                                         'lwant = {"format": LAUNCH_FORMAT, ')],
     "validate", "case_arm_teacher", "arm_teacher_launch_provenance_of_another_arm_B39", ["checkpoint_hash_mismatch"],
     None),
    ("arm_go_rule_verdict", MAN, [('"arm": arm, "verdict": entry["go_rule"]["verdict"],', '"arm": arm,')],
     "validate", "case_arm_teacher", "arm_teacher_go_rule_verdict_not_the_record_s_B39", ["checkpoint_hash_mismatch"],
     None),
    # --------------- the ARM measurement, aborted entries, a run's identity (DL-24 #36 #43 #45 #46 #52)
    ("measurement_files_present", VAL, [("    if absent:\n        return f\"{absent}: the measurement's",
                                         "    if False:\n        return f\"{absent}: the measurement's")],
     "validate", "case_measurement", "arm_measurement_files_absent_B36", ["completion_record_invalid"], None),
    ("measurement_files_cited", VAL, [('            if any(shas[k] != e[k]["sha256"] for k in shas) or e["completion"] '
                                       '!= st.completion.as_entry():',
                                       '            if e["completion"] != st.completion.as_entry():')],
     "validate", "case_measurement", "arm_measurement_output_not_the_cited_B36", ["completion_source_mismatch"],
     None),
    ("measurement_completion_block", VAL, [('            if any(shas[k] != e[k]["sha256"] for k in shas) or '
                                            'e["completion"] != st.completion.as_entry():',
                                            '            if any(shas[k] != e[k]["sha256"] for k in shas):')],
     "validate", "case_measurement", "arm_measurement_completion_block_not_its_record_B36",
     ["completion_source_mismatch"], None),
    ("measurement_complete_at_its_instant", VAL, [('    bad += [e["slot"] for e in x.entries if e["kind"] == '
                                                   '"measurement"', '    bad += [e["slot"] for e in [] if '
                                                   'e["kind"] == "measurement"')],
     "validate", "case_measurement", "arm_measurement_after_its_instant_B36", ["entry_not_complete"], None),
    ("aborted_names_the_last_two_attempts", VAL, [('            if len(e["run_ids"]) != 2 or set(e["run_ids"]) != '
                                                   'set(last):', "            if False:")],
     "validate", "case_aborted", "aborted_entry_naming_one_attempt_B43", ["completion_source_mismatch"], None),
    ("aborted_runs_not_complete", VAL, [("                if st.status in (C.COMPLETE, C.DIVERGED):",
                                         "                if False:")],
     "validate", "case_aborted", "complete_run_entered_as_aborted_B43", ["completion_source_mismatch"], None),
    ("aborted_rule_and_cause", VAL, [('                elif st.status == C.ABORTED and (st.rule, st.cause) != '
                                      '(e["rule"], e["cause"]):', "                elif False:")],
     "validate", "case_aborted", "aborted_entry_of_another_rule_B43", ["completion_source_mismatch"], None),
    ("kd_run_meta_of_the_launched_line", VAL, [('        if sha != log.launched[rid]["run_meta_sha256"]:',
                                                "        if False:")],
     "validate", "case_run_identity", "kd_run_meta_not_the_launched_line_s_B46", ["completion_source_mismatch"],
     None),
    ("qat_run_of_its_parent_checkpoint", VAL, [('            if pck is not None and meta.get('
                                                '"source_checkpoint_sha256") != pck:', "            if False:")],
     "validate", "case_run_identity", "qat_run_of_another_parent_checkpoint_B52", ["completion_source_mismatch"],
     None),
    # ---------------------------- the session's host, one session over history, the part read first (#37 #56 #101)
    ("session_on_the_frozen_host", VAL, [('    if s["host"] != x.body["execution_context"]["host"]:', "    if False:")],
     "unlock", "case_session", "session_host_not_the_frozen_host_B37", ["clock_offset_invalid"], None),
    ("clock_offset_record_of_the_session_host", VAL, [('doc["host"] != s["host"]["name"]', "False")],
     "unlock", "case_session", "clock_offset_record_of_another_host_B37", ["clock_offset_invalid"], None),
    ("second_session_over_history", VAL, [("    for commit, part in _committed_sessions(x):",
                                           "    for commit, part in []:")],
     "unlock", "case_session", "second_session_part_over_the_first_B56", ["second_session"], None),
    ("second_unlock_after_a_session", VAL, [('        if x.mode != "session":\n            return f"commit '
                                             '{commit[:12]} '
                                             'already holds', '        if False:\n            return f"commit '
                                             '{commit[:12]} already holds')],
     "unlock", "case_session", "unlock_after_a_committed_session_part_refused_B56", ["second_session"], None),
    ("session_values_only_added", VAL, [(" and names[0] <= names[1]):", " and True):")],
     "unlock", "case_session", "session_values_removed_B56", ["second_session"], None),
    ("session_identity_fields", VAL, [("        same = all(old.get(k) == cur.get(k) for k in SESSION_IDENTITY)",
                                       "        same = True")],
     "unlock", "case_session", "second_session_part_over_the_first_B56", ["second_session"], None),
    ("session_part_read_first", VAL, [('    ("second_session", U, g_second_session), ("clock_offset_invalid", SES, '
                                       'g_clock_offset_invalid),\n    ("session_after_deadline", U, '
                                       'g_session_after_deadline), ',
                                       '    ("second_session", U, g_second_session), ("session_after_deadline", U, '
                                       'g_session_after_deadline),\n    ("clock_offset_invalid", SES, '
                                       'g_clock_offset_invalid), ')],
     "unlock", "case_session", "session_part_null_refused_B101", ["clock_offset_invalid"], None),
]
REFUSAL_EXTRA = ("inventory_invalid", "inventory_sha256_mismatch", "binding_invalid",       # read_inventory's codes
                 "output_location", "output_exists")                                       # the CLI's outputs
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


ONLY: set = set()        # --only id,id: a development run of those mutations and their cases; never the record


def case_mutations():
    import scripts.smoke_tman_schema as FX
    from src.tman import validate as V
    check = FX.check
    before = files_sha256()
    clean = scratch_root({}, FX)
    mutations = [m for m in M if not ONLY or m[0] in ONLY]
    cases = sorted({(m[3], m[4]) for m in mutations})
    jobs = [("baseline", (smoke, case), {"root": str(clean), "smoke": smoke, "case": case, "check": None})
            for smoke, case in cases]
    for mid, rel, edits, smoke, case, target, codes, sample in mutations:
        try:
            root = scratch_root({rel: edits}, FX)
        except ValueError as e:
            check(f"mut_{mid}_text_occurs_once", False, str(e))
            continue
        jobs.append(("mutation", mid, {"root": str(root), "smoke": smoke, "case": case, "check": target}))
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as pool:
        outs = list(pool.map(lambda job: run_worker(job[2]), jobs))
    targets = {}
    for m in mutations:
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
        # killed: the named check itself fails (DL-24 #6) -- a crash, or another check failing, is no kill
        killed = "error" not in out and m[5] in fails
        how = out.get("error") or (f"raised {out['raised']}" if out.get("raised") else f"failed {fails[:3]}")
        print(f"MUTATION {key:42} {m[1]:30} {'KILLED' if killed else 'SURVIVED'} by {m[3]}:{m[4]} -- {how[:160]}")
        check(f"mut_{key}", killed, f"survived: {m[3]}:{m[4]}'s check {m[5]} did not fail ({how[:200]})"
              if "error" not in out else str(out.get("error"))[:300])
        if killed:
            killed_codes |= set(m[6])
    if not ONLY:
        codes = {c for c, _, _ in V.GATES} | set(REFUSAL_EXTRA)
        check("every_refusal_code_has_a_killed_mutation", codes <= killed_codes and len(codes) == 89,
              str(sorted(codes - killed_codes)))
        check("plan_s_sixteen_sample_mutations_present", sorted(m[7] for m in M if m[7]) == list(range(1, 17)))
    check("the_repository_files_are_unchanged", files_sha256() == before)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(RESULT_TAG + json.dumps(worker(json.loads(sys.argv[2]))))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        ONLY.update(sys.argv[2].split(","))
        unknown = ONLY - {m[0] for m in M}
        if unknown:
            raise SystemExit(f"--only names no mutation: {sorted(unknown)}")
    import scripts.smoke_tman_schema as FX
    return FX.run_cases("TMAN MUTATIONS SMOKE (lane TMAN, B-1, C6)", (case_mutations,))


if __name__ == "__main__":
    raise SystemExit(main())
