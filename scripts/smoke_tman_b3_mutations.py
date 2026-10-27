#!/usr/bin/env python3
"""Smoke for B-3a part 1's mutation tests (lane TMAN, B-3a part 1, B3A-3; plan :648 and ruling H2(c): its mutations
live in this smoke of their own): every new refusal and branch of src/tman/dedup.py and scripts/tday_hash_images.py
has a mutation killed by a named case -- the precedence swapped, the last instead of the first, a reversed sort, the
partition flag, the order of sha256-before-parse, the path guards, "xb" changed to "wb", the pin and module checks,
expected_images among them.

smoke_tman_mutations.py's form (:2-14, :1178-1261): a mutation replaces exact texts (each must occur once) in a scratch
copy of the two files, under the system temp dir; a fresh interpreter imports the copy as src.tman.dedup and
scripts.tday_hash_images (B-1's modules stay the checkout's) and runs one case of smoke_tman_dedup or smoke_tman_hasher
up to its named check. The mutation is killed when that named check fails; a crash, or another check failing, is no
kill. Each killing case first passes alone on the unmutated copy, its named checks among those it records. The
repository's files are read, never written; the scratch copies are removed.

Run with TMPDIR set to a new folder:
  TMPDIR=<new folder> python -B scripts/smoke_tman_b3_mutations.py
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

DED, HSH = "src/tman/dedup.py", "scripts/tday_hash_images.py"
SMOKES = {"dedup": "scripts.smoke_tman_dedup", "hasher": "scripts.smoke_tman_hasher"}
RESULT_TAG = "MUTATION-RESULT "
WORD = "te" + "st"                                   # project rule 4's word, never written whole here
GUARD = "refuse_" + WORD + "_path"                   # B-1's guard (schema.py :248)
LISTED = "        if len(f) != len(TRAINVAL_HEADER) or f[0] not in sets or not DIGITS_RE.fullmatch(f[3]) \\\n" \
         "                or not S.SHA256_RE.fullmatch(f[4]) or not SHA1_RE.fullmatch(f[5]):"
LIST_SHA = ('    got = S.sha256_bytes(data)\n    if got != sha256:\n        raise S.TmanRefused("train_val_list_mismatch", '
            'f"the TRAIN/VAL list\'s sha256 is {got}, not the pin\'s "\n'
            '                                                       f"{sha256}: it is not parsed")\n')
LIST_UTF8 = ('    try:\n        text = data.decode("utf-8")\n    except UnicodeDecodeError:\n        raise S.TmanRefused('
             '"train_val_list_invalid", "the TRAIN/VAL list is not UTF-8") from None\n')
READ_SHA = ('    if expect_sha256 is not None and S.sha256_bytes(data) != expect_sha256:\n        raise S.TmanRefused('
            '"dedup_invalid", f"{DEDUP_NAME}: sha256 {S.sha256_bytes(data)}, not {expect_sha256}: it "\n'
            '                                             "is not parsed")\n')
READ_PARSE = '    doc = S.strict_json(data, DEDUP_NAME, code="dedup_invalid")\n'

# (id, file, [(old, new), ...], smoke, case, the check the case runs to, the refusal codes it guards)
M = [
    # ------------------------------------------------------------------------------------- dedup.py: the rule (H3)
    ("precedence_swapped", DED, [('        if h in train_val["train"]:\n            removed["equal_train"].append(stem)\n'
                                  '            both += h in train_val["val"]\n        elif h in train_val["val"]:\n'
                                  '            removed["equal_val"].append(stem)\n',
                                  '        if h in train_val["val"]:\n            removed["equal_val"].append(stem)\n'
                                  '            both += h in train_val["train"]\n        elif h in train_val["train"]:\n'
                                  '            removed["equal_train"].append(stem)\n')],
     "dedup", "case_rule", "both_match_removed_under_equal_train_and_also_counted_plan577", ["dedup_invalid"]),
    ("last_instead_of_first", DED, [("        first, *rest = sorted(members)\n",
                                     "        *rest, first = sorted(members)\n")],
     "dedup", "case_rule", "kept_ids_are_each_group_s_first_sorted_id", ["dedup_invalid"]),
    ("reversed_sort", DED, [("first, *rest = sorted(members)", "first, *rest = sorted(members, reverse=True)")],
     "dedup", "case_rule", "code_point_upper_case_before_lower_case_keeps_kx_B1", ["dedup_invalid"]),
    ("both_match_not_counted", DED, [('            both += h in train_val["val"]\n', "            both += 0\n")],
     "dedup", "case_rule", "both_match_removed_under_equal_train_and_also_counted_plan577", ["dedup_invalid"]),
    ("partition_flag_false", DED, [('"reasons_partition": True,', '"reasons_partition": False,')],
     "dedup", "case_rule", "reasons_partition_true_in_the_built_doc", ["dedup_invalid"]),
    ("hash_list_unsorted", DED, [('for r in sorted(rows, key=lambda r: r["stem"])', "for r in rows")],
     "dedup", "case_canonical_and_hash_list", "hash_list_header_sorted_rows_lf_utf8_H6", []),
    # -------------------------------------------------------------------------------- dedup.py: the invariants (H5)
    ("builder_unchecked", DED, [('    errs = doc_errors(doc)\n    if errs:\n        raise S.TmanRefused("dedup_invalid", '
                                 '"; ".join(errs[:4]))', '    errs = []\n    if errs:\n        raise S.TmanRefused('
                                                         '"dedup_invalid", "; ".join(errs[:4]))')],
     "dedup", "case_invariants", "invariant_kept_not_empty_build_refused", ["dedup_invalid"]),
    ("partition_flag_unchecked", DED, [('    if doc["reasons_partition"] is not True:', "    if False:")],
     "dedup", "case_invariants", "invariant_partition_flag_false_reported", ["dedup_invalid"]),
    ("kept_may_be_empty", DED, [('    if not kept:\n        errs.append("kept_ids is empty")',
                                 '    if False:\n        errs.append("kept_ids is empty")')],
     "dedup", "case_invariants", "invariant_kept_not_empty_build_refused", ["dedup_invalid"]),
    ("counts_not_lengths", DED, [("    if any(counts[r] != len(rid[r]) for r in REASONS):", "    if False:")],
     "dedup", "case_invariants", "invariant_counts_are_the_lists_lengths", ["dedup_invalid"]),
    ("partition_sum_unchecked", DED, [('    if sum(counts.values()) != doc["n_images"] - len(kept):', "    if False:")],
     "dedup", "case_invariants", "invariant_sum_broken_reported", ["dedup_invalid"]),
    ("disjoint_unchecked", DED, [("    if len(set(every)) != len(every):", "    if False:")],
     "dedup", "case_invariants", "invariant_disjoint_broken_reported", ["dedup_invalid"]),
    ("union_unchecked", DED, [('    if len(every) != doc["n_images"] or ids_sha256(every) != doc["image_ids_sha256"]:',
                               "    if False:")],
     "dedup", "case_invariants", "invariant_union_broken_reported", ["dedup_invalid"]),
    ("kept_fingerprint_unchecked", DED, [('    if ids_sha256(kept) != doc["kept_ids_sha256"]:', "    if False:")],
     "dedup", "case_invariants", "invariant_kept_fingerprint_reported", ["dedup_invalid"]),
    ("also_counted_unbounded", DED, [('    if ac[ALSO_COUNTED] > counts["equal_train"]:', "    if False:")],
     "dedup", "case_invariants", "invariant_also_counted_at_most_equal_train", ["dedup_invalid"]),
    ("lists_unsorted_allowed", DED, [("    if ids != sorted(set(ids)):", "    if False:")],
     "dedup", "case_invariants", "invariant_kept_ids_sorted_and_unique", ["dedup_invalid"]),
    ("ids_shape_unchecked", DED, [("    if not isinstance(ids, list) or not all(_is_id(i) for i in ids):",
                                   "    if not isinstance(ids, list):")],
     "dedup", "case_invariants", "invariant_ids_without_surrounding_whitespace", ["dedup_invalid"]),
    # ------------------------------------------------------------------------- dedup.py: the TRAIN/VAL list (H9 4)
    ("list_parsed_before_its_sha256", DED, [(LIST_SHA + LIST_UTF8, LIST_UTF8 + LIST_SHA)],
     "dedup", "case_list_parser", "list_wrong_sha256_refused_before_parsing", ["train_val_list_mismatch"]),
    ("list_sha256_unchecked", DED, [('    if got != sha256:\n        raise S.TmanRefused("train_val_list_mismatch"',
                                     '    if False:\n        raise S.TmanRefused("train_val_list_mismatch"')],
     "dedup", "case_list_parser", "list_wrong_sha256_refused_before_parsing", ["train_val_list_mismatch"]),
    ("list_final_lf_unchecked", DED, [('    if not text.endswith("\\n"):', "    if False:")],
     "dedup", "case_list_parser", "list_without_final_lf_refused", ["train_val_list_invalid"]),
    ("list_header_unchecked", DED, [('    if lines[0] != "\\t".join(TRAINVAL_HEADER):', "    if False:")],
     "dedup", "case_list_parser", "list_wrong_header_refused", ["train_val_list_invalid"]),
    ("list_rows_unchecked", DED, [("    if len(body) != rows:", "    if False:")],
     "dedup", "case_list_parser", "list_row_count_refused", ["train_val_list_invalid"]),
    ("list_row_form_unchecked", DED, [(LISTED, "        if False:")],
     "dedup", "case_list_parser", "list_row_of_five_fields_refused", ["train_val_list_invalid"]),
    ("list_split_counts_unchecked", DED, [('    if counts != {"train": train, "val": val}:', "    if False:")],
     "dedup", "case_list_parser", "list_split_counts_refused", ["train_val_list_invalid"]),
    # ----------------------------------------------------------------------------------- dedup.py: the listed ids
    ("ids_whitespace_allowed", DED, [("    if spaced:", "    if False:")],
     "dedup", "case_image_ids", "ids_surrounding_whitespace_refused", ["image_ids_invalid"]),
    ("ids_tab_cr_lf_allowed", DED, [("    if unsafe:", "    if False:")],
     "dedup", "case_image_ids", "ids_tab_in_a_name_refused", ["image_ids_invalid"]),
    ("ids_not_utf8_allowed", DED, [("    if raw:", "    if False:")],
     "dedup", "case_image_ids", "ids_not_utf8_refused", ["image_ids_invalid"]),
    ("ids_shared_stem_allowed", DED, [("    if shared:", "    if False:")],
     "dedup", "case_image_ids", "ids_shared_stem_refused", ["image_ids_invalid"]),
    ("ids_case_fold_allowed", DED, [("    if clash:", "    if False:")],
     "dedup", "case_image_ids", "ids_case_folded_collision_refused", ["image_ids_invalid"]),
    # ------------------------------------------------------------------------------- dedup.py: the strict reader
    ("reader_parses_before_its_sha256", DED, [(READ_SHA + READ_PARSE, READ_PARSE + READ_SHA)],
     "dedup", "case_reader", "reader_sha256_compared_before_parsing", ["dedup_invalid"]),
    ("reader_accepts_non_canonical", DED, [("    if S.canonical_json_bytes(doc) != data:", "    if False:")],
     "dedup", "case_reader", "reader_non_canonical_bytes_refused", ["dedup_invalid"]),
    ("reader_skips_the_invariants", DED, [('    errs = doc_errors(doc)\n    if errs:\n        raise S.TmanRefused('
                                           '"dedup_invalid", f"{DEDUP_NAME}: "', '    errs = []\n    if errs:\n'
                                                                                 '        raise S.TmanRefused('
                                                                                 '"dedup_invalid", f"{DEDUP_NAME}: "')],
     "dedup", "case_reader", "reader_broken_invariant_refused", ["dedup_invalid"]),
    # ----------------------------------------------------------------- dedup.py: fingerprints, values, the issue
    ("ids_sha256_trailing_lf", DED, [('    return S.sha256_bytes("\\n".join(sorted(ids)).encode("utf-8"))',
                                      '    return S.sha256_bytes(("\\n".join(sorted(ids)) + "\\n").encode("utf-8"))')],
     "dedup", "case_lrep_rules", "ids_sha256_is_lrep_s_definition_no_trailing_lf", []),
    ("value_keys_reordered", DED, [('VALUE_KEYS = ("hash_list_sha256", "dedup_list_sha256", "counts_by_reason")',
                                    'VALUE_KEYS = ("dedup_list_sha256", "hash_list_sha256", "counts_by_reason")')],
     "dedup", "case_hasher_values", "hasher_values_keys_are_manifest_s_544", []),
    ("values_of_another_hash_list", DED, [('    if doc["hash_list_sha256"] != h:', "    if False:")],
     "dedup", "case_hasher_values", "hasher_values_refuse_a_dedup_of_another_hash_list", ["dedup_invalid"]),
    ("issue_count_changed", DED, [(f'("{WORD} \u2194 train", 70)', f'("{WORD} \u2194 train", 71)')],
     "dedup", "case_issue_rows_and_keys", "issue_counts_pinned_70_11_13", []),
    # -------------------------------------------------------------------- tday_hash_images.py: steps 1 and 2 (H9)
    ("path_guards_dropped", HSH, [(f"    for p in (args.repo, args.root, args.train_val_list, args.out):\n"
                                   f"        S.{GUARD}(p)", f"    for p in ():\n        S.{GUARD}(p)")],
     "hasher", "case_step1_paths", "step1_root_holding_the_word_refused_with_no_call", [WORD + "_path"]),
    ("unlock_guard_dropped", HSH, [(f"    S.{GUARD}(args.unlock_entry, args.repo)", "    pass")],
     "hasher", "case_step1_paths", "step1_unlock_entry_resolved_under_repo", [WORD + "_path"]),
    ("data_call_before_the_records", HSH, [('    check_paths(args)\n    params = check_records(args)["parameters"]',
                                            "    check_paths(args)\n    os.scandir(image_folder(args.root, "
                                            'split_of_record())).close()\n    params = check_records(args)'
                                            '["parameters"]')],
     "hasher", "case_step2_records", "step2_unlock_entry_absent_refused_before_any_data_call", []),
    ("shallow_clone_allowed", HSH, [("        records.src.require_not_shallow()\n", "        pass\n")],
     "hasher", "case_step2_repository", "step2_shallow_clone_refused_before_any_data_call", ["repository_shallow"]),
    ("no_head_commit_unnamed", HSH, [('        raise S.TmanRefused("record_uncommitted", f"--repo '
                                      '{records.src.root} is no git repository with a HEAD "\n'
                                      '                                                  "commit: the records are '
                                      'read at HEAD") from None\n', "        pass\n")],
     "hasher", "case_step2_repository", "step2_repository_without_a_commit_refused_before_any_data_call",
     ["record_uncommitted"]),
    ("unlock_kind_unchecked", HSH, [('    doc, _ = E.read_entry_file(records, args.unlock_entry, "unlock")',
                                     "    doc, _ = E.read_entry_file(records, args.unlock_entry)")],
     "hasher", "case_step2_records", "step2_unlock_entry_of_another_kind_refused_before_any_data_call",
     ["entry_file_invalid"]),
    ("manifest_path_unchecked", HSH, [('    if m["path"] != S.MANIFEST_REL:', "    if False:")],
     "hasher", "case_step2_records", "step2_manifest_path_other_refused_before_any_data_call",
     ["unlock_manifest_mismatch"]),
    ("manifest_sha256_unchecked", HSH, [('    if S.sha256_bytes(data) != m["sha256"]:', "    if False:")],
     "hasher", "case_step2_records", "step2_manifest_sha256_other_refused_before_any_data_call",
     ["unlock_manifest_mismatch"]),
    ("manifest_not_strict_json", HSH, [('    man = S.strict_json(data, m["path"], code="manifest_unreadable")',
                                        '    man = __import__("json").loads(data)')],
     "hasher", "case_step2_records", "step2_manifest_not_json_refused_before_any_data_call", ["manifest_unreadable"]),
    ("manifest_format_unchecked", HSH, [('    if not (isinstance(man, dict) and man.get("format") == '
                                         'S.MANIFEST_FORMAT', '    if False and (isinstance(man, dict) and '
                                                              'man.get("format") == S.MANIFEST_FORMAT')],
     "hasher", "case_step2_records", "step2_manifest_without_body_refused_before_any_data_call", ["manifest_format"]),
    ("manifest_body_unchecked", HSH, [('    if S.canonical_sha256(man["body"]) != m["body_sha256"]:', "    if False:")],
     "hasher", "case_step2_records", "step2_manifest_body_sha256_other_refused_before_any_data_call",
     ["unlock_manifest_mismatch"]),
    ("manifest_appends_unchecked", HSH, [('    if S.canonical_sha256(man["appends"]) != m["appends_sha256"]:',
                                          "    if False:")],
     "hasher", "case_step2_records", "step2_manifest_appends_sha256_other_refused_before_any_data_call",
     ["unlock_manifest_mismatch"]),
    ("pin_errors_ignored", HSH, [("    errs = pin_errors(pin)\n    if errs:", "    errs = []\n    if errs:")],
     "hasher", "case_step2_records", "step2_pin_script_sha256_other_refused_before_any_data_call",
     ["hasher_pin_mismatch"]),
    ("pin_form_unchecked", HSH, [('    errs = S.field_errors(pin, PIN_SPEC, "image_hasher.pin")\n    if errs:\n'
                                  '        return errs', "    errs = []\n    if errs:\n        return errs")],
     "hasher", "case_step2_records", "step2_pin_absent_refused_before_any_data_call", ["hasher_pin_mismatch"]),
    ("pin_script_unchecked", HSH, [('    if pin["script"]["path"] != SCRIPT_REL or pin["script"]["sha256"] != script:',
                                    "    if False:")],
     "hasher", "case_step2_records", "step2_pin_script_sha256_other_refused_before_any_data_call",
     ["hasher_pin_mismatch"]),
    ("pin_script_path_unchecked", HSH, [('pin["script"]["path"] != SCRIPT_REL or ', "")],
     "hasher", "case_step2_records", "step2_pin_script_path_other_refused_before_any_data_call",
     ["hasher_pin_mismatch"]),
    ("pin_module_unchecked", HSH, [('    if pin["parameters"]["modules"] != {DEDUP_REL: module}:', "    if False:")],
     "hasher", "case_step2_records", "step2_pin_module_sha256_other_refused_before_any_data_call",
     ["hasher_pin_mismatch"]),
    ("pin_python_unchecked", HSH, [('    if pin["python"] != running_python():', "    if False:")],
     "hasher", "case_step2_records", "step2_pin_python_other_refused_before_any_data_call", ["hasher_pin_mismatch"]),
    ("pin_parameters_unchecked", HSH, [('    if {k: pin["parameters"][k] for k in DESCRIPTIVE} != '
                                        '{k: own[k] for k in DESCRIPTIVE}:', "    if False:")],
     "hasher", "case_step2_records", "step2_pin_parameters_other_refused_before_any_data_call",
     ["hasher_pin_mismatch"]),
    # ----------------------------------------------------------------------- tday_hash_images.py: steps 3 to 6
    ("split_written_as_a_literal", HSH, [("    from src.eval.stage_artifacts import OFFICIAL_SPLIT\n"
                                          "    return OFFICIAL_SPLIT", '    return "tday_fixture"')],
     "hasher", "case_static", "h0_split_read_by_import_inside_the_resolver", []),
    ("a_second_listing", HSH, [("            names = [e.name for e in it if e.is_file()]\n",
                                "            names = [e.name for e in it if e.is_file()]\n        os.listdir(folder)\n")],
     "hasher", "case_pass_and_trace", "trace_no_listdir", []),
    ("folders_listed_as_images", HSH, [("[e.name for e in it if e.is_file()]", "[e.name for e in it]")],
     "hasher", "case_pass_and_trace", "hash_writes_both_outputs_exit_0", ["image_count_mismatch"]),
    ("suffix_case_sensitive", HSH, [("os.path.splitext(n)[1].lower() in SUFFIXES", "os.path.splitext(n)[1] in SUFFIXES")],
     "hasher", "case_pass_and_trace", "hash_writes_both_outputs_exit_0", ["image_count_mismatch"]),
    ("ids_unchecked", HSH, [("    errs = D.image_id_errors(pairs)\n    if errs:", "    errs = []\n    if errs:")],
     "hasher", "case_step3_listing", "step3_two_images_one_stem_refused_before_any_image_or_list_read",
     ["image_ids_invalid"]),
    ("expected_images_one_sided", HSH, [("    if len(pairs) != expected:", "    if len(pairs) > expected:")],
     "hasher", "case_step3_listing", "step3_image_count_other_refused_before_any_image_or_list_read",
     ["image_count_mismatch"]),
    ("folder_error_uncaught", HSH, [('    except OSError as e:\n        raise S.TmanRefused("images_unreadable", f"the '
                                     'image folder', '    except ValueError as e:\n        raise S.TmanRefused('
                                                     '"images_unreadable", f"the image folder')],
     "hasher", "case_step3_listing", "step3_image_folder_absent_refused_before_any_image_or_list_read",
     ["images_unreadable"]),
    ("list_error_uncaught", HSH, [('    except OSError as e:\n        raise S.TmanRefused("train_val_list_mismatch"',
                                   '    except ValueError as e:\n        raise S.TmanRefused("train_val_list_mismatch"')],
     "hasher", "case_step4_list", "step4_list_absent_refused_before_any_image_read", ["train_val_list_mismatch"]),
    ("list_rows_not_the_pin_s", HSH, [('rows=tv["rows"]', 'rows=len(data.split(b"\\n")) - 2')],
     "hasher", "case_step4_list", "step4_list_rows_not_the_pin_s_refused_before_any_image_read",
     ["train_val_list_invalid"]),
    ("image_error_uncaught", HSH, [('        except OSError as e:\n            raise S.TmanRefused("images_unreadable", '
                                    'f"an image', '        except ValueError as e:\n            raise S.TmanRefused('
                                                  '"images_unreadable", f"an image')],
     "hasher", "case_step5_hashing", "step5_unreadable_image_refused_nothing_written", ["images_unreadable"]),
    ("images_read_as_text", HSH, [('            with open(os.path.join(folder, name), "rb") as fh:',
                                   '            with open(os.path.join(folder, name), "r") as fh:')],
     "hasher", "case_pass_and_trace", "trace_one_binary_read_per_image", []),
    ("digest_not_sha256", HSH, [("        h, n = hashlib.sha256(), 0", "        h, n = hashlib.sha1(), 0")],
     "hasher", "case_pass_and_trace", "hash_list_is_the_images_sorted_by_stem_H6", []),
    ("create_not_exclusive", HSH, [('                fh = open(t, "xb")', '                fh = open(t, "wb")')],
     "hasher", "case_step6_outputs", "step6_exclusive_create_refuses_an_existing_file_xb", ["output_exists"]),
    ("partial_outputs_kept", HSH, [("        for t in made:\n            t.unlink(missing_ok=True)\n",
                                    "        for t in made:\n            pass\n")],
     "hasher", "case_step6_outputs", "step6_second_create_refused_first_output_removed", []),
    ("created_file_registered_after_its_write", HSH,
     [("                made.append(t)\n                fh.write(data)\n",
       "                fh.write(data)\n                made.append(t)\n")],
     "hasher", "case_step6_outputs", "step6_write_failure_exit_4_leaves_neither_output", []),
    ("error_exit_dropped", HSH, [("    except Exception as e:  # noqa: BLE001",
                                  "    except KeyError as e:  # noqa: BLE001")],
     "hasher", "case_step6_outputs", "step6_write_failure_exit_4_leaves_neither_output", []),
    ("outputs_not_checked_first", HSH, [("    if any(os.path.lexists(t) for t in targets):", "    if False:")],
     "hasher", "case_step6_outputs", "step6_one_output_present_refused_none_created", ["output_exists"]),
    ("out_inside_allowed", HSH, [("        if p == r or r in p.parents:", "        if False:")],
     "hasher", "case_step6_outputs", "step6_out_inside_the_repository_refused", ["output_location"]),
    ("this_checkout_not_checked", HSH, [("    for r in (REPO.resolve(), Path(repo).resolve()):",
                                         "    for r in (Path(repo).resolve(),):")],
     "hasher", "case_step6_outputs", "step6_out_inside_this_checkout_refused", ["output_location"]),
    ("out_folder_not_required", HSH, [("    if not p.is_dir():", "    if False:")],
     "hasher", "case_step6_outputs", "step6_out_not_an_existing_folder_refused", ["output_location"]),
    ("ids_printed", HSH, [("    print(D.issue_counts_line())", '    print(D.issue_counts_line(), doc["kept_ids"])')],
     "hasher", "case_pass_and_trace", "prints_no_id_H7", []),
    # ------------------------------------------------------------------------- tday_hash_images.py: the pin (H8)
    ("print_pin_unchecked", HSH, [("    errs = M.hasher_pin_errors(pin)\n    if errs:", "    errs = []\n    if errs:")],
     "hasher", "case_print_pin", "print_pin_commit_upper_case_refused", ["generator_pin_incomplete"]),
    ("pin_expected_images_not_1561", HSH, [('"expected_images": S.EXPECTED_ROWS,',
                                            '"expected_images": S.EXPECTED_ROWS - 1,')],
     "hasher", "case_print_pin", "pin_expected_images_is_schema_s_and_official_rows", []),
    ("pin_without_the_module", HSH, [('"modules": {DEDUP_REL: module}}}', '"modules": {}}}')],
     "hasher", "case_print_pin", "pin_modules_hold_the_dedup_module_s_sha256", []),
    ("pin_python_other", HSH, [('PYTHON = "3.11"', 'PYTHON = "3.12"')],
     "hasher", "case_print_pin", "pin_python_3_11", []),
    ("pin_counts_swapped", HSH, [("train=TRAIN_ROWS, val=VAL_ROWS", "train=VAL_ROWS, val=TRAIN_ROWS")],
     "hasher", "case_print_pin", "pin_train_val_list_is_b1_s_with_the_split_counts", []),
]
# The refusal codes the two files raise (schema.TmanRefused, and strict_json's code=), and B-1's guard's: each has a
# killed mutation (a static case ties this list to the files)
CODES = ("dedup_invalid", "generator_pin_incomplete", "hasher_pin_mismatch", "image_count_mismatch",
         "image_ids_invalid", "images_unreadable", "manifest_format", "manifest_unreadable", "output_exists",
         "output_location", "record_uncommitted", "repository_shallow", "train_val_list_invalid",
         "train_val_list_mismatch", "unlock_manifest_mismatch")
GUARD_CODES = (WORD + "_path",)                     # raised by B-1's guard from the hasher's step 1


# ================================================================================================ the worker
def worker(spec: dict) -> dict:
    """Run in a fresh interpreter: import the scratch copy at spec["root"] as src.tman.dedup and
    scripts.tday_hash_images (B-1's modules stay the checkout's), run spec["case"] of spec["smoke"] up to
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
    src.tman.dedup = load("src.tman.dedup", root / DED)
    load("scripts.tday_hash_images", root / HSH).REPO = REPO
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
    root = FX.scratch_dir("tmanbmut")
    for rel in (DED, HSH):
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
    return {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in (DED, HSH)}


def raised_codes(path: Path) -> set:
    """The literal codes a file refuses with: schema.TmanRefused's first argument, and strict_json's code=."""
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr == "TmanRefused" and n.args and isinstance(n.args[0], ast.Constant):
                out.add(n.args[0].value)
            if n.func.attr == "strict_json":
                out |= {k.value.value for k in n.keywords if k.arg == "code" and isinstance(k.value, ast.Constant)}
    return out


ONLY: set = set()        # --only id,id: a development run of those mutations and their cases; never the record


def case_mutations():
    import scripts.smoke_tman_schema as FX
    check = FX.check
    check("codes_listed_are_the_files_codes", set(CODES) == raised_codes(REPO / DED) | raised_codes(REPO / HSH),
          str(sorted(raised_codes(REPO / DED) | raised_codes(REPO / HSH))))
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
        check("every_refusal_code_has_a_killed_mutation", need <= killed_codes and len(need) == 16,
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
    return FX.run_cases("TMAN B-3a MUTATIONS SMOKE (lane TMAN, B-3a part 1, B3A-3)", (case_mutations,))


if __name__ == "__main__":
    raise SystemExit(main())
