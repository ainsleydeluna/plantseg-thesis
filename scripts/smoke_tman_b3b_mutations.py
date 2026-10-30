#!/usr/bin/env python3
"""Smoke for B-3b's mutation tests (lane TMAN, B-3b; B3R12): every refusal and rule of src/tman/binding.py and of
scripts/tday_manifest.py's bind has a mutation killed by a named case of smoke_tman_binding.

smoke_tman_b3_mutations.py's form: a mutation replaces exact texts (each must occur once) in a scratch copy of the
two files, under the system temp dir; a fresh interpreter imports the copy as src.tman.binding and
scripts.tday_manifest (the lane's other modules, and src.stats, stay the checkout's; src.stats.artifact's import of
BINDING_REL then reaches the copy) and runs one case of smoke_tman_binding up to its named check. The mutation is
killed when that named check fails; a crash, or another check failing, is no kill. Each killing case first passes
alone on the unmutated copy, its named checks among those it records. The repository's files are read, never
written; the scratch copies are removed.

Run with TMPDIR set to a new folder:
  TMPDIR=<new folder> python -B scripts/smoke_tman_b3b_mutations.py
  (a worker is this file with --worker <spec>; --only <id>,<id> runs those mutations and their cases alone, a
  development aid whose RESULT is not the record)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BIN, CLI = "src/tman/binding.py", "scripts/tday_manifest.py"
SMOKES = {"binding": "scripts.smoke_tman_binding"}
RESULT_TAG = "MUTATION-RESULT "
WORD = "te" + "st"                                   # project rule 4's word, never written whole here
INV = "binding_invalid"

# (id, file, [(old, new), ...], smoke, case, the check the case runs to, the refusal codes it guards)
M = [
    # -------------------------------------------------------------------------- binding.py: names and rules (B3R2-B3R5)
    ("binding_rel_other", BIN, [('BINDING_REL: Final[str] = "reports/derived/tday_binding.json"',
                                 'BINDING_REL: Final[str] = "reports/derived/tday_binding_v2.json"')],
     "binding", "case_one_definition", "binding_path_is_reports_derived_tday_binding_json", []),
    ("cell_rule_other", BIN, [('else f"{name}_s{severity}"', 'else f"{name}_{severity}"')],
     "binding", "case_stats_restated", "cell_rule_agrees_with_corruption_roles_on_the_24_cells_B3R5", [INV]),
    ("identities_e6_first", BIN, [("for stage in STATS_CELL_STAGES for cond in corr",
                                   "for stage in reversed(STATS_CELL_STAGES) for cond in corr")],
     "binding", "case_stats_restated", "stats_identities_are_official_inventory_s_37_in_order_B3R5", [INV]),
    ("identities_clean_last", BIN, [("    return tuple((stage, CLEAN) for stage in STATS_STAGES) + tuple(\n"
                                     "        (stage, cond) for stage in STATS_CELL_STAGES for cond in corr)",
                                     "    return tuple(\n        (stage, cond) for stage in STATS_CELL_STAGES "
                                     "for cond in corr) + tuple((stage, CLEAN) for stage in STATS_STAGES)")],
     "binding", "case_stats_restated", "stats_identities_are_official_inventory_s_37_in_order_B3R5", [INV]),
    ("identities_from_the_anchors", BIN, [("for name in roles.order for sev", "for name in CR.CANONICAL_IDS for sev")],
     "binding", "case_stats_restated", "stats_identities_read_the_protocol_through_corruption_roles_CLAR1", [INV]),
    ("identities_every_severity", BIN, [("for sev in roles.inferential_severities]", "for sev in CR.SEVERITIES]")],
     "binding", "case_stats_restated", "stats_identities_are_official_inventory_s_37_in_order_B3R5", [INV]),
    ("precision_e4_other", BIN, [('"E4": "int8_ptq", "E5"', '"E4": "int8_qat", "E5"')],
     "binding", "case_stats_restated", "stats_precision_stages_and_input_list_schema_are_the_driver_s", [INV]),
    # (dropping sorted() alone is equivalent: MANIFEST_NAME sorts before the lower-case ARTIFACT_FILES, already sorted)
    ("file_names_reversed", BIN, [("    return tuple(sorted((MANIFEST_NAME, *ARTIFACT_FILES)))",
                                   "    return tuple(sorted((MANIFEST_NAME, *ARTIFACT_FILES), reverse=True))")],
     "binding", "case_stats_restated", "files_sha256_keys_are_the_driver_s_input_files_and_the_evaluator_s_B3R3",
     [INV]),
    ("artifact_rel_without_its_cell", BIN, [('+ ("" if cell is None else f"/{cell}")', '+ ""')],
     "binding", "case_stats_restated", "artifact_rel_layout_B3R9", []),
    ("artifact_rel_word_guard_dropped", BIN, [("    S.refuse_test_path(rel)\n", "")],
     "binding", "case_stats_restated", "artifact_rel_refuses_the_word_B3R9", [WORD + "_path"]),
    ("read_not_binding_invalid", BIN, [('code="binding_invalid")', 'code="manifest_unreadable")')],
     "binding", "case_file_check", "binding_duplicate_key_read_strictly_B3R3", [INV]),
    # -------------------------------------------------------------------------------- binding.py: the file check
    ("file_spec_unchecked", BIN, [('    errs = S.field_errors(doc, S.BINDING_FILE_SPEC, "binding")\n',
                                   "    errs = []\n")],
     "binding", "case_file_check", "format_other_B3R3", [INV]),
    ("record_twice_allowed", BIN, [("        if key_of(r) in recs:", "        if False:")],
     "binding", "case_file_check", "record_twice_B3R3", [INV]),
    ("files_keys_unchecked", BIN, [('        if sorted(r["files_sha256"]) != names:', "        if False:")],
     "binding", "case_file_check", "files_sha256_three_files_B3R3", [INV]),
    ("cell_rule_unchecked", BIN, [('        if (r["cell"] is not None) != (r["scope"] in CELL_SCOPES):',
                                   "        if False:")],
     "binding", "case_file_check", "corrupted_record_naming_no_cell", [INV]),
    ("protocol_rule_unchecked", BIN, [('        if (r["protocol"] == UPSTREAM) != (r["scope"] == UPSTREAM):',
                                       "        if False:")],
     "binding", "case_file_check", "clean_record_under_the_upstream_protocol", [INV]),
    ("record_path_form_unchecked", BIN, [('        err = S.relative_path_error(r["repo_relative_path"])\n',
                                          "        err = None\n")],
     "binding", "case_file_check", "record_path_with_a_dotdot_P2R5", [INV]),
    ("record_commit_unchecked", BIN, [('        if r["repo_commit"] != commit:', "        if False:")],
     "binding", "case_file_check", "record_repo_commit_not_the_evaluator_s_B3R5", [INV]),
    ("split_digests_unchecked", BIN, [("        if len(found) > 1:", "        if False:")],
     "binding", "case_file_check", "canvas_records_two_split_digests_DL74", [INV]),
    ("split_digest_groups_merged", BIN, [('digests[UPSTREAM if r["scope"] == UPSTREAM else "canvas"]',
                                          'digests["canvas"]')],
     "binding", "case_file_check", "fixture_binding_passes_the_file_check_B3R5", [INV]),
    ("upstream_digests_dropped", BIN, [('digests[UPSTREAM if r["scope"] == UPSTREAM else "canvas"].add(',
                                        '(set() if r["scope"] == UPSTREAM else digests["canvas"]).add(')],
     "binding", "case_file_check", "upstream_records_two_split_digests_B3R5", [INV]),
    ("identities_unchecked", BIN, [("    if not same(got, list(identities)):", "    if False:")],
     "binding", "case_file_check", "inputs_36_B3R5", [INV]),
    ("identities_order_free", BIN, [("    if not same(got, list(identities)):",
                                     "    if not same(sorted(got, key=repr), sorted(identities, key=repr)):")],
     "binding", "case_file_check", "inputs_out_of_order_B3R5", [INV]),
    ("input_key_shape_unchecked", BIN, [(" and all(k is None or isinstance(k, str) for k in it[\"key\"])", "")],
     "binding", "case_file_check", "input_key_holding_a_list_B3R5", [INV]),
    ("input_key_record_unchecked", BIN, [("        if key not in recs:", "        if False:")],
     "binding", "case_file_check", "input_key_naming_no_record_B3R5", [INV]),
    ("input_key_rule_unchecked", BIN, [("        elif key[1:] != want:", "        elif False:")],
     "binding", "case_file_check", "input_key_upstream_for_clean_B3R5", [INV]),
    ("input_precision_unchecked", BIN, [("want = (precision.get(stage),", "want = (key[1] if key else None,")],
     "binding", "case_file_check", "input_key_precision_not_the_stage_s_B3R5", [INV]),
    ("input_scope_unchecked", BIN, [('"clean" if cond == CLEAN else "corrupted", cell_of(cond))',
                                    'key[2] if key else None, cell_of(cond))')],
     "binding", "case_file_check", "input_key_sev45_scope_for_a_cell_B3R5", [INV]),
    ("input_cell_unchecked", BIN, [('"clean" if cond == CLEAN else "corrupted", cell_of(cond))',
                                   '"clean" if cond == CLEAN else "corrupted", key[3] if key else None)')],
     "binding", "case_file_check", "input_key_cell_not_the_condition_s_B3R5", [INV]),
    ("input_named_twice_allowed", BIN, [("        elif key in named:", "        elif False:")],
     "binding", "case_file_check", "two_inputs_naming_one_record_B3R5", [INV]),
    # ------------------------------------------------------------------------------ binding.py: the export check
    ("export_without_the_file_check", BIN, [("    errs = file_errors(doc, identities, precision)\n    if errs:\n"
                                             "        return errs\n    rows =", "    errs = []\n    rows =")],
     "binding", "case_export_check", "export_check_carries_the_file_check_B3R5", [INV]),
    ("export_row_unchecked", BIN, [("        if row is None:\n            errs.append(",
                                    "        if False:\n            errs.append(")],
     "binding", "case_export_check", "record_of_no_export_row_B3R5", [INV]),
    ("export_cell_unchecked", BIN, [('        elif r["cell"] is not None and r["cell"] not in row["cells"]:',
                                     "        elif False:")],
     "binding", "case_export_check", "record_cell_not_its_row_s_B3R5", [INV]),
    ("export_commit_unchecked", BIN, [('        if r["repo_commit"] != export["evaluator_commit"]:',
                                       "        if False:")],
     "binding", "case_export_check", "record_repo_commit_not_the_export_s_B3R5", [INV]),
    ("export_body_unchecked", BIN, [('    if refs["manifest"]["body_sha256"] != export["manifest_sha256"]:',
                                     "    if False:")],
     "binding", "case_export_check", "references_of_another_manifest_body_B3R5", [INV]),
    ("export_context_unchecked", BIN, [('    if not same(refs["execution_context"], export["execution_context"]):',
                                        "    if False:")],
     "binding", "case_export_check", "references_of_another_execution_context_B3R5", [INV]),
    # ---------------------------------------------------- binding.py: types, the references and the export (DL-24)
    ("identities_compared_loosely", BIN, [("    if not same(got, list(identities)):",
                                           "    if got != list(identities):")],
     "binding", "case_file_check", "input_severity_float_B3R5", [INV]),
    ("same_loose", BIN, [("    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)",
                          "    return a == b")],
     "binding", "case_file_check", "input_severity_true_B3R5", [INV]),
    ("references_paths_unchecked", BIN, [('        err = S.relative_path_error(doc["references"][part]["path"])\n',
                                          "        err = None\n")],
     "binding", "case_file_check", "manifest_path_absolute_B3R3", [INV]),
    ("export_manifest_file_unchecked", BIN, [('    if refs["manifest"]["sha256"] != export["manifest_file_sha256"]:',
                                              "    if False:")],
     "binding", "case_export_check", "references_of_another_manifest_file_B3R7", [INV]),
    ("export_dedup_unchecked", BIN, [('    if export.get("dedup") is not None and refs["session"]["dedup"]["sha256"] '
                                      '!= export["dedup"].get("sha256"):', "    if False:")],
     "binding", "case_export_check", "references_of_another_dedup_list_B3R7", [INV]),
    ("export_context_compared_loosely", BIN, [
        ('    if not same(refs["execution_context"], export["execution_context"]):',
         '    if refs["execution_context"] != export["execution_context"]:')],
     "binding", "case_export_check", "references_execution_context_not_an_exact_copy_B3R3", [INV]),
    # ------------------------------------------------- binding.py: the statistics code's check (B3R6(a), (d), (e))
    ("inputs_identities_unchecked", BIN, [("    if sorted(got, key=repr) != sorted(want, key=repr):", "    if False:")],
     "binding", "case_inputs_reason", "inputs_of_other_identities_reason_B3R6d", [INV]),
    ("inputs_fields_unchecked", BIN, [("        bad = [k for k in INPUT_FIELDS if fields.get(k) != rec[k]]",
                                       "        bad = []")],
     "binding", "case_inputs_reason", "input_checkpoint_sha256_differing_reason_B3R6d", [INV]),
    ("inputs_repo_commit_dropped", BIN, [('INPUT_FIELDS: Final = ("repo_relative_path", "checkpoint_sha256", '
                                          '"repo_commit", "split_manifest_sha256")',
                                          'INPUT_FIELDS: Final = ("repo_relative_path", "checkpoint_sha256", '
                                          '"split_manifest_sha256")')],
     "binding", "case_inputs_reason", "input_repo_commit_differing_reason_B3R6d", [INV]),
    ("inputs_files_unchecked", BIN, [('        if "files_sha256" in fields and fields["files_sha256"] != '
                                      'rec["files_sha256"]:', "        if False:")],
     "binding", "case_stats_loader", "one_byte_changed_in_one_per_image_jsonl_listed_B3R6d", [INV]),
    ("inputs_files_always_compared", BIN, [('        if "files_sha256" in fields and fields["files_sha256"] != '
                                            'rec["files_sha256"]:',
                                            '        if fields.get("files_sha256") != rec["files_sha256"]:')],
     "binding", "case_inputs_reason", "inputs_equal_on_the_fields_and_the_files_give_no_reason_B3R6d", [INV]),
    ("bound_absent_not_unbound", BIN, [("    if not os.path.isfile(path):", "    if False:")],
     "binding", "case_stats_loader", "no_binding_file_lists_the_binding_unbound_B3R6a", [INV]),
    ("bound_strict_read_uncaught", BIN, [("    except (OSError, S.TmanRefused) as e:", "    except OSError as e:")],
     "binding", "case_stats_loader", "binding_not_strict_json_listed_with_its_reason_B3R3", [INV]),
    ("bound_file_check_skipped", BIN, [("    errs = file_errors(doc, identities, precision)\n    if errs:\n"
                                        '        return f"the TEST-manifest binding',
                                        '    errs = []\n    if errs:\n        return f"the TEST-manifest binding')],
     "binding", "case_stats_loader", "binding_the_file_check_refuses_listed_with_its_reason_B3R6a", [INV]),
    ("bound_inputs_unchecked", BIN, [("    return inputs_reason(doc, got)", "    return None")],
     "binding", "case_stats_loader", "binding_naming_another_record_for_one_identity_listed_B3R6d", [INV]),
    # ------------------------------------------------------------------------------- binding.py: the full check (B3R5)
    ("full_checks_order_other", BIN, [('FULL_CHECKS: Final = ("binding_invalid", "binding_manifest_mismatch", '
                                       '"binding_incomplete",',
                                       'FULL_CHECKS: Final = ("binding_invalid", "binding_incomplete", '
                                       '"binding_manifest_mismatch",')],
     "binding", "case_full_check", "full_check_order_is_b3r5_s_B3R5", []),
    ("full_check_items_swapped", BIN, [("_identity_item, _context_item,\n" + " " * 40 + "_stem_item,",
                                        "_identity_item, _stem_item,\n" + " " * 40 + "_context_item,")],
     "binding", "case_full_check", "artifact_tuple_key_other_TB1", ["binding_context_split"]),
    ("full_check_passed_not_carried", BIN, [("            refusal.passed = list(passed)\n",
                                             "            refusal.passed = []\n")],
     "binding", "case_full_check", "export_of_another_manifest_body_B3R8", ["binding_manifest_mismatch"]),
    ("full_check_refusal_not_raised", BIN, [("            raise refusal\n", "            continue\n")],
     "binding", "case_full_check", "full_check_carries_the_export_check_B3R5", [INV]),
    ("full_export_item_dropped", BIN, [('    errs = export_errors(doc, plan["export"], plan["identities"], '
                                        'plan["precision"])\n', "    errs = []\n")],
     "binding", "case_full_check", "full_check_carries_the_export_check_B3R5", [INV]),
    ("manifest_body_unchecked", BIN, [('export["manifest_sha256"] == m["body_sha256"])', "True)")],
     "binding", "case_full_check", "export_of_another_manifest_body_B3R8", ["binding_manifest_mismatch"]),
    ("manifest_file_unchecked", BIN, [('export["manifest_file_sha256"] == m["sha256"])', "True)")],
     "binding", "case_full_check", "export_of_another_manifest_file_B3R8", ["binding_manifest_mismatch"]),
    ("manifest_references_unchecked", BIN, [('refs["manifest"] == {k: m[k] for k in MANIFEST_REFERENCE})', "True)")],
     "binding", "case_full_check", "references_freeze_row_not_the_manifest_s_B3R3", ["binding_manifest_mismatch"]),
    ("manifest_mde_entry_unchecked", BIN, [('refs["mde_entry"] == m["mde_entry"]),', "True),")],
     "binding", "case_full_check", "references_mde_entry_not_the_manifest_s_TB11", ["binding_manifest_mismatch"]),
    ("manifest_context_unchecked", BIN, [('same(refs["execution_context"], m["execution_context"])),', "True),")],
     "binding", "case_full_check", "references_execution_context_not_the_manifest_s_B3R3",
     ["binding_manifest_mismatch"]),
    ("manifest_evaluator_unchecked", BIN, [('refs["evaluator"] == {"commit": m["evaluator_commit"]})) if not ok]',
                                           "True)) if not ok]")],
     "binding", "case_full_check", "references_evaluator_not_the_manifest_s_B3R3", ["binding_manifest_mismatch"]),

    ("missing_records_allowed", BIN, [("    missing, extra = sorted(want - have, key=repr), sorted(have - want, "
                                       "key=repr)", "    missing, extra = [], sorted(have - want, key=repr)")],
     "binding", "case_full_check", "a_required_record_missing_B3R5", ["binding_incomplete"]),
    ("extra_records_allowed", BIN, [("    missing, extra = sorted(want - have, key=repr), sorted(have - want, "
                                     "key=repr)", "    missing, extra = sorted(want - have, key=repr), []")],
     "binding", "case_full_check", "an_extra_record_fq_off_of_an_int8_row_T_FQ7", ["binding_incomplete"]),
    ("required_fq_off_scope_other", BIN, [('out.add((m["slot"], m["precision"], "fq_off", None))',
                                           'out.add((m["slot"], m["precision"], "clean", None))')],
     "binding", "case_full_check", "fixture_passes_the_full_check_B3R5", ["binding_incomplete"]),
    ("required_upstream_dropped", BIN, [('            if scope in ("clean", UPSTREAM):',
                                         '            if scope == "clean":')],
     "binding", "case_full_check", "fixture_passes_the_full_check_B3R5", ["binding_incomplete"]),
    ("required_cells_unfiltered", BIN, [('for c in m["cells"] if c in cells[scope]}', 'for c in m["cells"]}')],
     "binding", "case_full_check", "fixture_passes_the_full_check_B3R5", ["binding_incomplete"]),
    ("identity_unread_record_allowed", BIN, [('in doc["artifacts"] if key_of(r) not in plan["reads"]\n',
                                              'in doc["artifacts"] if False\n')],
     "binding", "case_full_check", "record_whose_artifact_was_not_read_B3R8", ["binding_identity_mismatch"]),
    ("identity_record_uncompared", BIN, [('\n           or r != record_of(key_of(r), plan["reads"][key_of(r)])', "")],
     "binding", "case_full_check", "record_files_not_its_artifact_s_B3R8", ["binding_identity_mismatch"]),
    ("identity_row_uncompared", BIN, [('\n           or r["checkpoint_sha256"] != identity(rows[(r["slot"], '
                                       'r["precision"])])]', "]")],
     "binding", "case_full_check", "artifact_and_record_not_the_row_s_identity_TB17", ["binding_identity_mismatch"]),
    ("identity_int8_rule_dropped", BIN, [('    return row["converted_artifact_sha256"] if row["precision"] in INT8 '
                                          'else row["source"]["sha256"]', '    return row["source"]["sha256"]')],
     "binding", "case_full_check", "fixture_passes_the_full_check_B3R5", ["binding_identity_mismatch"]),
    ("record_protocol_canvas_always", BIN, [('"protocol": UPSTREAM if scope == UPSTREAM else "canvas",',
                                             '"protocol": "canvas",')],
     "binding", "case_full_check", "fixture_passes_the_full_check_B3R5", ["binding_identity_mismatch"]),
    ("context_runtime_uncompared", BIN, [('           if any(plan["reads"][key_of(r)]["runtime"].get(k) != ec.get(k) '
                                          'for k in plan["runtime_keys"])\n           or ', "           if ")],
     "binding", "case_full_check", "artifact_tuple_key_other_TB1", ["binding_context_split"]),
    ("context_commit_uncompared", BIN, [('\n           or plan["reads"][key_of(r)]["repo_commit"] != '
                                         'ec.get("evaluator_commit")]', "]")],
     "binding", "case_full_check", "artifact_commit_not_the_context_s_evaluator_TB1", ["binding_context_split"]),
    ("stems_unchecked", BIN, [('           and any(image_id != clean_id for image_id, clean_id in '
                               'plan["reads"][key_of(r)]["ids"])]', "           and False]")],
     "binding", "case_full_check", "clean_artifact_stem_differs_DL74", ["binding_stem_mismatch"]),
    ("stem_scopes_without_fq_off", BIN, [('STEM_SCOPES: Final = ("clean", "corrupted", "corrupted_sev45", "fq_off")',
                                          'STEM_SCOPES: Final = ("clean", "corrupted", "corrupted_sev45")')],
     "binding", "case_full_check", "fq_off_artifact_stem_differs_DL74", ["binding_stem_mismatch"]),
    ("stem_scopes_with_upstream", BIN, [('STEM_SCOPES: Final = ("clean", "corrupted", "corrupted_sev45", "fq_off")',
                                         'STEM_SCOPES: Final = ("clean", "corrupted", "corrupted_sev45", "fq_off", '
                                         '"upstream")')],
     "binding", "case_full_check", "upstream_artifact_stems_exempt_DL74", ["binding_stem_mismatch"]),
    ("inputs_slots_unchecked", BIN, [('if it["key"][0] != plan["slots_of_record"].get(it["stage"])})', "if False})")],
     "binding", "case_full_check", "input_naming_another_slot_than_the_slot_of_record_R065",
     ["binding_inputs_mismatch"]),
    ("session_unchecked", BIN, [('    return None if got == plan["dedup_list_sha256"] else \\',
                                 "    return None if True else \\")],
     "binding", "case_full_check", "session_dedup_list_not_the_session_s_TB11", ["binding_session_mismatch"]),
    ("stats_inputs_path_other", BIN, [('"path": recs[tuple(it["key"])]["repo_relative_path"]}',
                                       '"path": recs[tuple(it["key"])]["slot"]}')],
     "binding", "case_bind", "stats_inputs_load_through_the_driver_B3R8", []),
    ("stats_inputs_condition_other", BIN, [('dict(zip(("type", "name", "severity"), it["condition"]))',
                                            'dict(zip(("name", "type", "severity"), it["condition"]))')],
     "binding", "case_bind", "stats_inputs_load_through_the_driver_B3R8", []),
    # ------------------------------------------------------------- tday_manifest.py: bind (B3R8, B3R9; decision (vi))
    ("bind_outputs_not_prechecked", CLI, [("        if os.path.lexists(folder / name):", "        if False:")],
     "binding", "case_bind", "bind_refuses_when_either_output_exists_both_or_neither_B3R8", ["output_exists"]),
    ("bind_records_commit_unchecked", CLI, [("    S.refuse_test_path(args.records)\n    if not S.COMMIT_RE.fullmatch("
                                             "args.records_commit):\n        raise S.TmanRefused("
                                             "\"record_uncommitted\", "
                                             "f\"--records-commit {args.records_commit!r} is not 40 hex\")\n",
                                             "    S.refuse_test_path(args.records)\n")],
     "binding", "case_bind", "bind_records_commit_not_40_hex", ["record_uncommitted"]),
    # (dropping V.validate leaves x.body unset, so bind ends before the named check: its refusal is swallowed instead)
    ("bind_session_refusal_ignored", CLI, [("    V.validate(x)\n    export = V.read_inventory(args.inventory)\n",
                                            "    try:\n        V.validate(x)\n    except S.TmanRefused:\n        pass\n"
                                            "    export = V.read_inventory(args.inventory)\n")],
     "binding", "case_bind", "bind_validates_the_session_its_dedup_read_strictly_B3R8", []),
    ("bind_dedup_not_passed", CLI, [("records=records, dedup=args.dedup, runs_root=args.runs,",
                                     "records=records, runs_root=args.runs,")],
     "binding", "case_bind", "bind_writes_the_binding_and_the_stats_inputs_B3R8", []),
    ("bind_inventory_sha_other", CLI, [("{S.sha256_bytes(read_bytes(args.inventory, 'inventory_invalid'))}",
                                        "{S.sha256_bytes(data)}")],
     "binding", "case_bind", "bind_writes_the_binding_and_the_stats_inputs_B3R8", []),
    ("bind_full_check_skipped", CLI, [("    B.full_check(doc, {", "    (lambda *a: None)(doc, {")],
     "binding", "case_bind", "bind_required_artifact_absent_B3R5", ["binding_incomplete"]),
    ("bind_absent_folder_read", CLI, [("    if not folder.is_dir():\n        return None\n", "")],
     "binding", "case_bind", "bind_required_artifact_absent_B3R5", ["binding_incomplete"]),
    ("bind_unreadable_uncaught", CLI, [('        except OSError as e:\n            raise S.TmanRefused('
                                        '"binding_artifact_invalid", f"{rel}/{name}: unreadable',
                                        '        except KeyError as e:\n            raise S.TmanRefused('
                                        '"binding_artifact_invalid", f"{rel}/{name}: unreadable')],
     "binding", "case_bind", "bind_artifact_file_absent_B3R8", ["binding_artifact_invalid"]),
    ("bind_summary_values_unchecked", CLI, [('    if not (hasattr(rt, "keys") and hasattr(ds, "keys")',
                                             '    if False and (hasattr(rt, "keys") and hasattr(ds, "keys")')],
     "binding", "case_bind", "bind_artifact_without_its_runtime_record_B3R8", ["binding_artifact_invalid"]),
    ("bind_summary_strict_code_other", CLI, [('code="binding_artifact_invalid",\n' + " " * 22
                                              + "object_type=blind(SUMMARY_KEEP))",
                                              'code="inventory_invalid",\n' + " " * 22
                                              + "object_type=blind(SUMMARY_KEEP))")],
     "binding", "case_bind", "bind_artifact_summary_not_strict_json_B3R8", ["binding_artifact_invalid"]),
    ("bind_row_strict_code_other", CLI, [('code="binding_artifact_invalid",\n' + " " * 28
                                          + "object_type=blind(ROW_KEEP))",
                                          'code="inventory_invalid",\n' + " " * 28 + "object_type=blind(ROW_KEEP))")],
     "binding", "case_bind", "bind_artifact_row_not_strict_json_B3R8", ["binding_artifact_invalid"]),
    ("bind_row_ids_unchecked", CLI, [('\n                and isinstance(row.get("clean_image_id"), str)):', "):")],
     "binding", "case_bind", "bind_artifact_row_without_its_clean_id_B3R8", ["binding_artifact_invalid"]),
    ("bind_inputs_scope_sev45", CLI, [('["clean", None] if cond == B.CLEAN else ["corrupted", B.cell_of(cond)])',
                                       '["clean", None] if cond == B.CLEAN else ["corrupted_sev45", '
                                       'B.cell_of(cond)])')],
     "binding", "case_bind", "bind_writes_the_binding_and_the_stats_inputs_B3R8", [INV]),
    ("bind_family_seed_other", CLI, [('if "family" in s["roles"] and s["seed"] == 42}',
                                      'if "family" in s["roles"] and s["seed"] == 43}')],
     "binding", "case_bind", "bind_writes_the_binding_and_the_stats_inputs_B3R8", [INV]),
    ("bind_session_dedup_other", CLI, [('"session": {"dedup": {"sha256": S.sha256_bytes(read_bytes(args.dedup, '
                                        '"dedup_invalid"))}},',
                                        '"session": {"dedup": {"sha256": S.sha256_bytes(data)}},')],
     "binding", "case_bind", "bind_writes_the_binding_and_the_stats_inputs_B3R8", ["binding_session_mismatch"]),
    ("bind_written_utc_wall_clock", CLI, [('"written_utc": C.utc_text(int(clock(args)))},',
                                           '"written_utc": C.utc_text(int(time.time()))},')],
     "binding", "case_bind", "references_of_the_session_the_freeze_row_and_the_clock_B3R3", []),
    ("bind_binding_not_canonical", CLI, [("(BIND_NAMES[0], S.canonical_json_bytes(doc))",
                                          "(BIND_NAMES[0], S.json_file_bytes(doc))")],
     "binding", "case_bind", "binding_is_canonical_json_B3R3", []),
    ("bind_blind_keeps_every_key", CLI, [("    return lambda pairs: R.JSON_OBJECT([(k, v) for k, v in pairs if k in "
                                          "keep])", "    return lambda pairs: R.JSON_OBJECT(pairs)")],
     "binding", "case_score_blind_reads", "no_score_key_enters_an_object_bind_reads_B3R8", []),
    ("bind_summary_keep_a_score_key", CLI, [('"eval_runtime", *R.EVAL_RUNTIME))',
                                            '"eval_runtime", "all_class_miou", *R.EVAL_RUNTIME))')],
     "binding", "case_score_blind_reads", "no_score_key_enters_an_object_bind_reads_B3R8", []),
    ("bind_row_keep_a_score_key", CLI, [('ROW_KEEP = frozenset(("image_id", "clean_image_id"))',
                                        'ROW_KEEP = frozenset(("image_id", "clean_image_id", "all_class_miou"))')],
     "binding", "case_score_blind_reads", "no_score_key_enters_an_object_bind_reads_B3R8", []),
    ("bind_artifact_word_guard_dropped", CLI, [("    S.refuse_test_path(rel, root)\n    folder = Path(root) / rel\n",
                                                "    folder = Path(root) / rel\n")],
     "binding", "case_score_blind_reads", "bind_artifact_path_resolved_under_the_root_B3R9", [WORD + "_path"]),
]
# The refusal codes of the binding's code (schema.TmanRefused, and strict_json's code=) -- binding.py's, the full
# check's eight (B3R5) and bind's own (B3R8) -- and B-1's guard's: each has a killed mutation (a static case ties
# this list to the files)
CODES = (INV, "binding_manifest_mismatch", "binding_incomplete", "binding_identity_mismatch", "binding_context_split",
         "binding_stem_mismatch", "binding_inputs_mismatch", "binding_session_mismatch",
         "binding_artifact_invalid", "output_exists", "record_uncommitted")
GUARD_CODES = (WORD + "_path",)                     # raised by B-1's guard from artifact_rel and bind's read


# ================================================================================================ the worker
def worker(spec: dict) -> dict:
    """Run in a fresh interpreter: import the scratch copy at spec["root"] as src.tman.binding and
    scripts.tday_manifest (the lane's other modules stay the checkout's), run spec["case"] of spec["smoke"] up to
    spec["check"] (None: the whole case) and return the checks it recorded and any exception."""
    import importlib
    import importlib.util
    import src.tman
    root = Path(spec["root"])

    def load(name: str, path: Path):
        sp = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
        return mod
    src.tman.binding = load("src.tman.binding", root / BIN)
    load("scripts.tday_manifest", root / CLI).REPO = REPO
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
        getattr(mod, spec["case"])()
    except Stop:
        pass
    except Exception as e:  # noqa: BLE001 -- a case that raises fails by name
        raised = f"{type(e).__name__}: {e}"[:300]
    FX.remove_scratch()
    return {"results": [(n, ok, d[:200]) for n, ok, d in FX.RESULTS], "raised": raised}


# ================================================================================================ the parent
def scratch_root(edits_by_file: dict, FX) -> Path:
    """A scratch copy of the two files with the edits applied; each old text must occur exactly once (ValueError
    otherwise)."""
    root = FX.scratch_dir("tmanbbmut")
    for rel in (BIN, CLI):
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
    return {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in (BIN, CLI)}


def raised_codes(path: Path, functions=None) -> set:
    """The literal codes a file refuses with -- schema.TmanRefused's first argument and strict_json's code= -- in the
    named `functions` (all of the file when None)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scopes = [tree] if functions is None else [n for n in ast.walk(tree)
                                               if isinstance(n, ast.FunctionDef) and n.name in functions]
    out = set()
    for scope in scopes:
        for n in ast.walk(scope):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "TmanRefused" and n.args and isinstance(n.args[0], ast.Constant):
                    out.add(n.args[0].value)
                if n.func.attr == "strict_json":
                    out |= {k.value.value for k in n.keywords if k.arg == "code" and isinstance(k.value, ast.Constant)}
    return out


ONLY: set = set()        # --only id,id: a development run of those mutations and their cases; never the record


def case_mutations():
    import scripts.smoke_tman_schema as FX
    from src.tman import binding as B
    check = FX.check
    files_codes = raised_codes(REPO / BIN) | set(B.FULL_CHECKS) | raised_codes(
        REPO / CLI, functions={"cmd_bind", "read_artifact"})
    check("codes_listed_are_the_files_codes", set(CODES) == files_codes, str(sorted(files_codes)))
    before = files_sha256()
    clean = scratch_root({}, FX)
    mutations = [m for m in M if not ONLY or m[0] in ONLY]
    cases = sorted({(m[3], m[4]) for m in mutations})
    jobs = [("baseline", (smoke, case), {"root": str(clean), "smoke": smoke, "case": case, "check": None})
            for smoke, case in cases]
    for mid, rel, edits, smoke, case, target, codes in mutations:
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
        killed = "error" not in out and m[5] in fails          # the named check itself fails
        how = out.get("error") or (f"raised {out['raised']}" if out.get("raised") else f"failed {fails[:3]}")
        print(f"MUTATION {key:34} {m[1]:28} {'KILLED' if killed else 'SURVIVED'} by {m[3]}:{m[4]} -- {how[:150]}")
        check(f"mut_{key}", killed, f"survived: {m[3]}:{m[4]}'s check {m[5]} did not fail ({how[:200]})"
              if "error" not in out else str(out.get("error"))[:300])
        if killed:
            killed_codes |= set(m[6])
    if not ONLY:
        need = set(CODES) | set(GUARD_CODES)
        check("every_refusal_code_has_a_killed_mutation", need <= killed_codes and len(need) == 12,   # B3R8: 2 -> 12
              str(sorted(need - killed_codes)))
        check("every_mutation_has_its_own_id", len({m[0] for m in M}) == len(M))
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
    return FX.run_cases("TMAN B-3b MUTATIONS SMOKE (lane TMAN, B-3b; B3R12)", (case_mutations,))


if __name__ == "__main__":
    raise SystemExit(main())
