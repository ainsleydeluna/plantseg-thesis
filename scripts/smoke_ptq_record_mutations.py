#!/usr/bin/env python3
"""LQP mutation smoke: every refusal the lane adds, and the record fields it relies on, each killed by a named check.

The explicit paths of SHADOW_PATHS (every repository file scripts/smoke_ptq_record.py opens, in every process it
starts, recorded once with an audit hook on `open`; bytecode caches left out) are copied into temporary shadows of the
repository and committed there once; the repository itself is never written. One edit is applied at a time, in a
free shadow: an exact string, which must occur exactly once in its file, is replaced (for a refusal, its condition
becomes `False` or its raise is replaced), and the edited file must still parse. scripts/smoke_ptq_record.py then runs inside that
shadow in a subprocess, for the one case that holds the killing check (--only), in a session and process group of
its own (a timeout kills the whole group), and the named check must be reported FAIL there, having PASSED on every
unmutated shadow in the same run with the same argv. A check the smoke never reports, a timeout included, is not a
kill. The edited file is restored byte for byte after its run. The TRAIN tree and the two student checkpoints are
built once into a cache shared by every run (--cache-dir of the smoke).

The edits (MUTATIONS): the refusals of scripts/run_ptq.py's new gates (G0 source_exactly_one, G1 test_path, G2
binding_needs_registered_list and the bound routing, the lazy import), src/quant/runner.py's old_runner_real_run,
src/quant/ptq.py's out_dir_not_empty (C2), each refusal and STOP of src/quant/ptq_record.py with its Q2 wrapper and
record reader, the record fields the lane names (rulings R6, R7), src/eval/stage_artifacts.py's record gate and its
order (ruling R2), scripts/evaluate_model.py's call, and each refusal of scripts/ptq_calibration_range.py (R13).

SIGTERM, SIGINT, SIGHUP or SIGQUIT to the harness kills every live job's group, starts nothing after it and ends the
run FAIL; run it detached with setsid and an outer timeout that keeps TERM (timeout -k 60 ...).

    python -B scripts/smoke_ptq_record_mutations.py [--only ID,ID] [--workers N] [--check-edits]

Ends with one RESULT line; exit 0 only when every baseline check passes and every edit is killed.
"""
from __future__ import annotations

import argparse
import ast
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import safe_tmpdir  # noqa: E402  (imports no repository code)

SMOKE = "scripts/smoke_ptq_record.py"
TIMEOUT = 1800                                   # per run: the slowest case (s5_e3_rule) takes well under a minute
SHADOW_PATHS: tuple[str, ...] = (
    "configs/augment.py",
    "configs/calibration/ptq_calibration_seed42.json",
    "configs/data.py",
    "configs/e1_student.py",
    "configs/model.py",
    "configs/plantseg_class_map.json",
    "configs/quant.py",
    "scripts/build_calibration_lists.py",
    "scripts/evaluate_model.py",
    "scripts/preflight_distill.py",
    "scripts/preflight_e1.py",
    "scripts/preflight_e1_trainval.py",
    "scripts/ptq_calibration_range.py",
    "scripts/run_e4.py",
    "scripts/run_e7.py",
    "scripts/run_ptq.py",
    "scripts/smoke_ptq_record.py",
    "scripts/synthetic_ptq_fixtures.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/isolation.py",
    "src/data/original_resolution.py",
    "src/data/transforms.py",
    "src/distill/__init__.py",
    "src/distill/cwd_projection.py",
    "src/distill/export.py",
    "src/distill/features.py",
    "src/distill/nmf_stream.py",
    "src/distill/segnext_teacher.py",
    "src/distill/teacher.py",
    "src/eval/__init__.py",
    "src/eval/adapters.py",
    "src/eval/artifacts.py",
    "src/eval/eval_runtime.py",
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
    "src/quant/ptq_record.py",
    "src/quant/qat.py",
    "src/quant/qconfig.py",
    "src/quant/runner.py",
    "src/quant/stages.py",
    "src/quant/x86_latency.py",
    "src/seeds.py",
    "src/training/__init__.py",
    "src/training/losses.py",
    "src/training/sweep_select.py",
    "src/training/train_e1.py",
)

RUNNER, PTQ, RECORD, RUN_PTQ = "src/quant/runner.py", "src/quant/ptq.py", "src/quant/ptq_record.py", "scripts/run_ptq.py"
GATE, EVAL, RANGE = "src/eval/stage_artifacts.py", "scripts/evaluate_model.py", "scripts/ptq_calibration_range.py"


def M(mid: str, rel: str, what: str, old: str, new: str, case: str, *kills: str) -> dict:
    return {"id": mid, "rel": rel, "what": what, "old": old, "new": new, "case": case, "kills": list(kills)}


MUTATIONS = [
    # ---- C1 (ruling Q1-R2, R11): the old runners
    M("LQ01", RUNNER, "the old runner's refusal skipped: E4/E7 return 0 once parsed",
      "    parser.parse_args(argv)\n    # Ruling Q1-R2", "    parser.parse_args(argv)\n    return 0\n    # Ruling Q1-R2",
      "s1_old_runners", "old_runners_write_nothing"),
    # ---- C2: the core's non-empty out-dir refusal (every caller of ptq.run_ptq)
    M("LQ02", PTQ, "ptq.run_ptq's out_dir_not_empty refusal removed",
      '    if out.exists() and (not out.is_dir() or any(out.iterdir())):\n        raise PTQRefused("out_dir_not_empty"',
      '    if False:\n        raise PTQRefused("out_dir_not_empty"', "s5_out_dir", "refuses_out_dir_not_empty_unbound"),
    # ---- C3: scripts/run_ptq.py's gates before the split (R4, G1) and the split itself
    M("LQ03", RUN_PTQ, "G0 source_exactly_one removed", "    if bool(args.source_ckpt) == bool(args.source_run_dir):\n",
      "    if False:\n", "s5_flags", "refuses_source_exactly_one"),
    M("LQ04", RUN_PTQ, "G1 test_path removed", "        if value and names_test(value):\n", "        if False:\n",
      "s5_flags", "refuses_test_path_run_dir", "refuses_test_path_out_dir"),
    M("LQ05", RUN_PTQ, "binding_needs_registered_list removed", "            if given:\n", "            if False:\n",
      "s5_flags", "refuses_binding_flags_unregistered_list"),
    M("LQ06", RUN_PTQ, "a registered list routed to the unbound path",
      '        if doc.get("artifact_status") == LIST_STATUS_REGISTERED:\n', "        if False:\n",
      "s4_seeds_43", "e7_s43_exit_0", "e4_s43_exit_0"),
    M("LQ07", RUN_PTQ, "the record module imported at the top of run_ptq.py",
      "EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2\n",
      "EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2\nimport src.quant.ptq_record  # noqa: E402,F401\n",
      "s10_hygiene", "unbound_run_never_imports_record_module"),
    # ---- C3: src/quant/ptq_record.py, G2-G6
    M("LQ08", RECORD, "G2 calibration_list_not_committed removed", "    if problem is not None:\n", "    if False:\n",
      "s5_flags", "refuses_list_not_committed", "refuses_list_changed_since_head"),
    M("LQ09", RECORD, "G3 seed_invalid removed", "    if args.seed not in SEEDS:", "    if False:", "s5_flags",
      "refuses_seed_invalid"),
    M("LQ10", RECORD, "G3 sensitivity_seed removed",
      '    if doc["seed"] != SEED_OF_RECORD and args.seed != SEED_OF_RECORD:\n', "    if False:\n", "s5_flags",
      "refuses_sensitivity_list_at_seed_43"),
    M("LQ11", RECORD, "G3 source_run_dir_required removed", "    if not args.source_run_dir:\n", "    if False:\n",
      "s5_flags", "refuses_source_run_dir_required"),
    M("LQ12", RECORD, "G3 expect_head_format removed",
      "    if not args.expect_head or not _HEX40.match(args.expect_head):\n", "    if False:\n", "s5_flags",
      "refuses_expect_head_format"),
    M("LQ13", RECORD, "G3 selection_not_applicable removed",
      '    if not rule["selections"] and any(v is not None for v in selection_flags):\n', "    if False:\n",
      "s5_flags", "refuses_e4_selection_flags"),
    M("LQ14", RECORD, "G4 out_dir_relative removed", "    if not Path(value).is_absolute():\n", "    if False:\n",
      "s5_out_dir", "refuses_out_dir_relative"),
    M("LQ15", RECORD, "G4 output_dir_inside_repo swallowed",
      "        raise _as_refusal(e.code, str(e)) from e\n    m = re.fullmatch",
      "        out = Path(value).resolve()\n    m = re.fullmatch", "s5_out_dir", "refuses_out_dir_inside_repo"),
    M("LQ16", RECORD, "G4 out_dir_name removed", "    if m is None or (m.group(2) is not None) != replicate:\n",
      "    if False:\n", "s5_out_dir", "refuses_out_dir_name_stage", "refuses_out_dir_name_list",
      "refuses_d3_name_without_replicate", "refuses_replicate_without_d3_name"),
    M("LQ17", RECORD, "G4 out_dir_not_empty removed (bound path)",
      '    if out.exists() and (not out.is_dir() or any(out.iterdir())):\n        _refuse("out_dir_not_empty"',
      '    if False:\n        _refuse("out_dir_not_empty"', "s5_out_dir", "refuses_out_dir_not_empty_bound"),
    M("LQ18", RECORD, "G5 code_identity_unavailable removed (R7)", '    if ident["commit"] is None:\n',
      "    if False:\n", "s5_identity", "refuses_code_identity_unavailable"),
    M("LQ19", RECORD, "G5 expect_head_mismatch removed (R7)", '    if ident["commit"] != expect_head:\n',
      "    if False:\n", "s5_identity", "refuses_expect_head_mismatch"),
    M("LQ20", RECORD, "G5 code_not_clean removed (R7)", '    if ident["clean_at_commit"] is not True:\n',
      "    if False:\n", "s5_identity", "refuses_code_not_clean"),
    M("LQ21", RECORD, "G5 image_digest_not_of_record removed", "    if digest != IMAGE_DIGEST_OF_RECORD:\n",
      "    if False:\n", "s5_identity", "refuses_image_digest_unset", "refuses_image_digest_not_of_record"),
    M("LQ22", RECORD, "G6 data_root_has_test swallowed (R6)",
      '        raise _as_refusal("data_root_has_test", f"[{e.code}] {e}") from e\n', "        result = {}\n",
      "s5_identity", "refuses_data_root_with_test_surface"),
    # ---- G7-G10: the Q2 wrapper (R9) and the parent rules
    M("LQ23", RECORD, "a Q2 refusal swallowed by the wrapper", '        raise _as_refusal(e.code, f"{prefix}{e}") from e\n',
      "        return None\n", "s5_selections", "refuses_selection_sha256_mismatch"),
    M("LQ24", RECORD, "a Q2 refusal's message not led by the stage and rule (R9)",
      '        raise _as_refusal(e.code, f"{prefix}{e}") from e\n', "        raise _as_refusal(e.code, str(e)) from e\n",
      "s5_selections", "refuses_selection_sha256_mismatch"),
    M("LQ25", RECORD, "G10 qat.e6_parent_binding dropped",
      "    _q2(prefix, qat.e6_parent_binding, parent, args.seed, lam, alpha)\n", "    pass\n", "s5_e3_rule",
      "refuses_e7_parent_lambda_mismatch", "refuses_e7_parent_alpha_mismatch", "refuses_e7_s42_non_winner_run"),
    M("LQ26", RECORD, "G9 parent_horizon removed",
      "    if not (_is_int(mi) and mi == h) or not (ph is None or (_is_int(ph) and ph == h)):\n", "    if False:\n",
      "s5_e1_rule", "refuses_e4_parent_160k", "refuses_e4_parent_poly_horizon_160k"),
    M("LQ27", RECORD, "G9 an absent poly_horizon refused (R3)", "(ph is None or (_is_int(ph) and ph == h))",
      "(_is_int(ph) and ph == h)", "s5_e1_rule", "accepts_e4_parent_without_poly_horizon"),
    M("LQ28", RECORD, "G9 a present poly_horizon not compared (R3)",
      " or not (ph is None or (_is_int(ph) and ph == h)):", ":", "s5_e1_rule", "refuses_e4_parent_poly_horizon_160k"),
    M("LQ29", RECORD, "G9 parent_payload removed (E1)",
      '    if bad:\n        _refuse("parent_payload", f"{prefix}the checkpoint best.json names is not that E1 run',
      '    if False:\n        _refuse("parent_payload", f"{prefix}the checkpoint best.json names is not that E1 run',
      "s5_e1_rule", "refuses_e4_parent_payload_best_mismatch", "refuses_e4_parent_payload_stage_e3"),
    M("LQ30", RECORD, "G10 parent_not_of_record removed", "    if not_of_record:\n", "    if False:\n", "s5_e3_rule",
      "refuses_e7_parent_arm_run", "refuses_e7_parent_160k"),
    M("LQ31", RECORD, "G10 parent_teacher_not_of_record removed (R17)",
      '    if (meta.get("teacher_mock") is not False or not isinstance(tp, dict) or tp.get("builder") != '
      'pd.TEACHER_BUILDER\n            or tp.get("ckpt_sha256") != pd.TEACHER_SHA256 or tp.get("expected_sha256") != '
      'pd.TEACHER_SHA256):\n', "    if False:\n", "s5_e3_rule", "refuses_e7_parent_teacher_r1",
      "refuses_e7_parent_teacher_mock"),
    M("LQ32", RECORD, "G10 parent_checkpoint_not_selected removed",
      '    if args.seed == 42 and alpha["winner"].get("ckpt_sha256") != parent["checkpoint_sha256"]:\n',
      "    if False:\n", "s5_e3_rule", "refuses_e7_s42_checkpoint_not_selected"),
    M("LQ33", RECORD, "G10 an unreadable launch log swallowed (R8)",
      '        raise _as_refusal("parent_launch_log", f"{prefix}[{e.code}] {e}") from e\n', "        return {}\n",
      "s5_e3_rule", "refuses_e7_launch_log_uncommitted"),
    M("LQ34", RECORD, "G10 no launched line accepted (R8)", "    if line is None or ld is None:\n", "    if False:\n",
      "s5_e3_rule", "refuses_e7_s43_not_launched"),
    M("LQ35", RECORD, "G10 a stopped line accepted (R8)", "    if rid in log.stopped:\n", "    if False:\n",
      "s5_e3_rule", "refuses_e7_s43_stopped_line"),
    M("LQ36", RECORD, "G10 the launched run_meta sha256 not compared (R8)",
      '    if ld["run_meta_sha256"] != parent["run_meta_sha256"]:\n', "    if False:\n", "s5_e3_rule",
      "refuses_e7_s43_run_meta_not_launched"),
    M("LQ37", RECORD, "G10 the launch-log fields not compared (R8)",
      '    if bad:\n        _refuse("parent_launch_log"', '    if False:\n        _refuse("parent_launch_log"',
      "s5_e3_rule", "refuses_e7_s43_launch_line_fields"),
    M("LQ38", RECORD, "G10 launch_time_utc dropped from the field set (R8, sweep_select.py:1810)",
      '("alpha_cwd", meta.get("alpha_cwd"), line["alpha_cwd"]),\n               ("launch_time_utc", meta.get('
      '"wall_clock"), ld["launch_time_utc"])]', '("alpha_cwd", meta.get("alpha_cwd"), line["alpha_cwd"])]',
      "s5_e3_rule", "refuses_e7_s43_launch_line_fields"),
    M("LQ39", RECORD, "G10 parent_payload removed (E3)",
      '    if bad:\n        _refuse("parent_payload", f"{prefix}the checkpoint best.json names is not that E3 run',
      '    if False:\n        _refuse("parent_payload", f"{prefix}the checkpoint best.json names is not that E3 run',
      "s5_e3_rule", "refuses_e7_payload_arm_key", "refuses_e7_payload_teacher_mismatch",
      "refuses_e7_payload_best_mismatch"),
    # ---- G11, the STOPs, the record reader
    M("LQ40", RECORD, "G11 an unreadable --replicate-of swallowed",
      '        raise _as_refusal("replicate_mismatch", f"--replicate-of {path}: {e}") from e\n',
      "        rec, sha = {}, None\n", "s5_replicate", "refuses_replicate_not_a_record"),
    M("LQ41", RECORD, "G11 replicate_mismatch removed", '    if bad:\n        _refuse("replicate_mismatch"',
      '    if False:\n        _refuse("replicate_mismatch"', "s5_replicate", "refuses_replicate_other_list",
      "refuses_replicate_other_parent", "refuses_replicate_of_replicate"),
    M("LQ42", RECORD, "STOP identity_changed removed (R7)", "    if changed:\n", "    if False:\n", "s5_stops",
      "stop_identity_changed"),
    M("LQ43", RECORD, "STOP record_write_failed removed",
      '    except OSError as e:\n        _stop("record_write_failed",',
      '    except OSError as e:\n        if False:\n            _stop("record_write_failed",', "s5_stops",
      "stop_record_write_failed"),
    M("LQ44", RECORD, "read_record's format and keys not checked",
      '    if not isinstance(rec, dict) or rec.get("format") != RECORD_FORMAT or sorted(rec) != sorted(RECORD_KEYS):\n',
      "    if False:\n", "s5_replicate", "refuses_replicate_not_a_record"),
    M("LQ45", RECORD, "read_record's re-hash of the listed files dropped",
      "        if not f.is_file() or _sha256_bytes(f.read_bytes()) != want:\n", "        if False:\n", "s8_gate",
      "official_refuses_tampered_artifact"),
    # ---- the record's fields (plan §1; R6, R7)
    M("LQ46", RECORD, "the parent's run_meta sha256 omitted",
      '"run_meta": {"file": parent["run_meta_file"], "sha256": parent["run_meta_sha256"]},',
      '"run_meta": {"file": parent["run_meta_file"]},', "s4_seeds_43", "e4_s43_record_fields"),
    M("LQ47", RECORD, "the list file sha256 omitted",
      '"calibration": {"list_path": g["list_rel"], "list_file_sha256": g["list_sha256"],',
      '"calibration": {"list_path": g["list_rel"],', "s4_seeds_43", "e4_s43_record_fields"),
    M("LQ48", RECORD, "another file's sha256 recorded for the run provenance",
      '"sha256": _ptq.sha256_file(out / names["run_meta"]),', '"sha256": _ptq.sha256_file(out / names["torchscript"]),',
      "s4_seeds_43", "e4_s43_record_fields"),
    M("LQ49", RECORD, "a sensitivity run labelled of_record",
      '            else ROLE_OF_RECORD if doc["seed"] == SEED_OF_RECORD else ROLE_SENSITIVITY)',
      "            else ROLE_OF_RECORD)", "s7_sensitivity", "e4_s42_cal43_role", "e7_s42_cal44_role"),
    M("LQ50", RECORD, "replicate_of omitted", '"replicate_of": g["replicate"],', '"replicate_of": None,',
      "s7_sensitivity", "e4_d3_stub_replicate_role"),
    M("LQ51", RECORD, "descriptive false for every role (R6)", '"descriptive": role != ROLE_OF_RECORD,',
      '"descriptive": False,', "s7_sensitivity", "e4_s42_cal43_role"),
    M("LQ52", RECORD, "the isolation block omitted (R6)", '"isolation": g["isolation"],', '"isolation": None,',
      "s4_seeds_43", "e4_s43_record_fields"),
    M("LQ53", RECORD, "the writer's commit not the identity reader's (R7)",
      '"commit": ident["commit"], "clean_at_commit": ident["clean_at_commit"],',
      '"commit": None, "clean_at_commit": ident["clean_at_commit"],', "s4_seeds_43", "e4_s43_record_fields"),
    M("LQ54", RECORD, "manifest_entry's ptq_script dropped (R6)", '"ptq_script": record["writer"]["script"]["path"],',
      '"ptq_script": None,', "s4_seeds_43", "e4_s43_manifest_entry"),
    # ---- C7: the official record gate (R2) and its one caller
    M("LQ55", GATE, "legacy_ptq_artifact removed",
      '    if prov.get("artifact_format", "state_dict") != "torchscript":\n        return (lead + "[legacy_ptq_artifact]',
      '    if False:\n        return (lead + "[legacy_ptq_artifact]', "s8_gate", "official_refuses_legacy_state_dict_ptq"),
    M("LQ56", GATE, "ptq_record_missing removed",
      '    if not path.is_file():\n        return (lead + f"[ptq_record_missing]',
      '    if False:\n        return (lead + f"[ptq_record_missing]', "s8_gate", "official_refuses_missing_record",
      "official_refuses_arm_directory"),
    M("LQ57", GATE, "ptq_record_invalid swallowed", 'return lead + f"[ptq_record_invalid] {path}: {e}"', "return None",
      "s8_gate", "official_refuses_malformed_record", "official_refuses_tampered_artifact"),
    M("LQ58", GATE, "ptq_record_role removed", '    if record["role"] != ROLE_OF_RECORD:\n', "    if False:\n", "s8_gate",
      "official_refuses_replicate_record"),
    M("LQ59", GATE, "ptq_record_mismatch removed", "    if differ:\n", "    if False:\n", "s8_gate",
      "official_refuses_record_of_another_run", "official_refuses_another_provenance"),
    M("LQ60", GATE, "the run provenance not compared with the record's", '("run provenance", run_meta == prov),', "",
      "s8_gate", "official_refuses_another_provenance"),
    M("LQ61", GATE, "the record gate called before the calibration gate",
      "return int8_official_calibration_error(resolved, repo_root) or _ptq_record_error(resolved)",
      "return _ptq_record_error(resolved) or int8_official_calibration_error(resolved, repo_root)", "s8_gate",
      "official_calibration_gate_runs_first"),
    M("LQ62", GATE, "QAT artifacts sent through the PTQ record gate",
      '    if spec["method"] != "ptq":\n        return None\n    stage = spec["stage"]\n', '    stage = spec["stage"]\n',
      "s8_gate", "official_record_gate_passes_qat"),
    M("LQ63", EVAL, "evaluate_model.py calls the calibration gate directly",
      "            from src.eval.stage_artifacts import int8_official_error\n"
      "            official_error = int8_official_error(resolved, repo_root=REPO)",
      "            from src.eval.stage_artifacts import int8_official_calibration_error\n"
      "            official_error = int8_official_calibration_error(resolved, repo_root=REPO)",
      "s8_source", "evaluate_model_calls_only_the_wrapper"),
    M("LQ64", EVAL, "evaluate_model.py ignores the wrapper's refusal",
      "            if official_error:\n                raise CliError(official_error)",
      "            if False:\n                raise CliError(official_error)", "s8_source",
      "evaluate_model_calls_only_the_wrapper"),
    # ---- C8: scripts/ptq_calibration_range.py (R13)
    M("LQ65", RANGE, "output_dir_inside_repo swallowed",
      "    except QuantRunError as e:\n        raise RangeRefused(e.code, str(e)) from e\n    return out / OUT_NAME",
      "    except QuantRunError as e:\n        out = Path(out_dir)\n    return out / OUT_NAME", "s9_range",
      "range_refuses_out_dir_inside_repo"),
    M("LQ66", RANGE, "range_input_invalid not raised as a refusal",
      '        raise RangeRefused("range_input_invalid", f"{record_path} / {summary_path}: {type(e).__name__}: {e}") from e',
      "        raise", "s9_range", "range_refuses_input_invalid"),
    M("LQ67", RANGE, "range_stage removed", "    if wrong:\n", "    if False:\n", "s9_range", "range_refuses_stage"),
    M("LQ68", RANGE, "range_lists removed",
      "    if seeds != list(LIST_SEEDS) or any(r != (ROLE_OF_RECORD if s == SEED else ROLE_SENSITIVITY) for s, r in "
      "roles):\n", "    if False:\n", "s9_range", "range_refuses_duplicate_list", "range_refuses_missing_list",
      "range_refuses_replicate_as_list_of_record"),
    M("LQ69", RANGE, "range_parent removed",
      '    if len(parents) != 1 or any(p["record"]["seed"] != SEED for p in rows):\n', "    if False:\n", "s9_range",
      "range_refuses_two_parents"),
    M("LQ70", RANGE, "range_artifact removed", "    if foreign:\n", "    if False:\n", "s9_range",
      "range_refuses_summary_of_another_artifact"),
    M("LQ71", RANGE, "range_status removed", "    if status:\n", "    if False:\n", "s9_range", "range_refuses_status"),
    M("LQ72", RANGE, "range_protocol removed", "    if off:\n", "    if False:\n", "s9_range", "range_refuses_mixed_split"),
    M("LQ73", RANGE, "range_rows removed", "    if len(counts) != 1 or any(e != a for e, a in counts):\n",
      "    if False:\n", "s9_range", "range_refuses_rows"),
    M("LQ74", RANGE, "range_metric removed", "    if nonfinite:\n", "    if False:\n", "s9_range", "range_refuses_metric"),
    M("LQ75", RANGE, "range_output_exists swallowed",
      '        raise RangeRefused("range_output_exists", f"{path} exists; the range is written once") from e',
      "        return path", "s9_range", "range_refuses_existing_output"),
    M("LQ76", RANGE, "a summary's sha256 not recorded (R13)", '"summary_sha256": p["summary_sha256"],',
      '"summary_sha256": None,', "s9_range", "range_written"),
    M("LQ77", RANGE, "the disease-only range dropped (R13)", '"disease_only_miou": span("disease_only_miou"),', "",
      "s9_range", "range_written"),
    # ---- the DL-24 fixes: an unreadable gate input (A-1)
    M("LQ78", RECORD, "an input a gate cannot read escapes as a traceback (DL-24 A-1)",
      "    except Exception as e:                      # noqa: BLE001 -- before the core: nothing was written (exit 2)\n",
      "    except ArithmeticError as e:                # noqa: BLE001 -- before the core: nothing was written (exit 2)\n",
      "s5_unreadable", "refuses_e7_parent_torn_telemetry_line", "refuses_e4_parent_checkpoint_unloadable",
      "refuses_without_git_binary"),
    # ---- the range reporter's input checks (B-1)
    M("LQ79", RANGE, "a summary's run, dataset and dataset_level not checked to be objects (DL-24 B-1)",
      '        run, ds, dl = (_typed(summary[k], dict, k) for k in ("run", "dataset", "dataset_level"))\n',
      '        run, ds, dl = (summary[k] for k in ("run", "dataset", "dataset_level"))\n', "s9_range",
      "range_refuses_dataset_level_null"),
    M("LQ80", RANGE, "a summary's row counts not checked to be integers (DL-24 B-1)",
      '"expected_rows": _typed(ds["expected_rows"], int, "dataset.expected_rows"),',
      '"expected_rows": ds["expected_rows"],', "s9_range", "range_refuses_rows_not_integer"),
    M("LQ81", RANGE, "a record's list seed read leniently (DL-24 B-1)",
      '"list_seed": _typed(rec["calibration"]["seed"], int, "calibration.seed"),',
      '"list_seed": rec["calibration"].get("seed"),', "s9_range", "range_refuses_record_without_list_seed"),
]

results: list[tuple[str, bool, str]] = []
STOP = threading.Event()
LIVE: dict[int, subprocess.Popen] = {}
LIVE_LOCK = threading.RLock()


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), str(detail)))


# ------------------------------------------------------------------ the shadow: explicit paths, no listing
def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "lqp", "GIT_AUTHOR_EMAIL": "lqp@example.invalid",
           "GIT_COMMITTER_NAME": "lqp", "GIT_COMMITTER_EMAIL": "lqp@example.invalid",
           "GIT_AUTHOR_DATE": "2026-10-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-10-01T00:00:00Z",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout


def build_shadow(dst: Path) -> Path:
    for rel in SHADOW_PATHS:
        src = REPO / rel
        if not src.is_file():
            raise RuntimeError(f"shadow path {rel} is not a file in the repository")
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / rel)
    git(dst, "init", "-q")
    git(dst, "add", "--", *SHADOW_PATHS)
    git(dst, "commit", "-q", "-m", "LQP mutation shadow")
    return dst


def shadow_status(shadow: Path) -> str:
    """The shadow's git status, untracked files included: empty unless a run wrote into the shadow or an edit
    was not restored."""
    return git(shadow, "status", "--porcelain=v1", "--untracked-files=all")


# ------------------------------------------------------------------ running the smoke (its own process group)
def run_case(shadow: Path, cache: Path, case: str) -> tuple[dict, str, float]:
    """{check: PASS|FAIL} for one case of the smoke in `shadow`, its RESULT line, its seconds."""
    t0 = time.time()
    if STOP.is_set():
        return {}, "skipped (stopped)", 0.0
    argv = [sys.executable, "-B", SMOKE, "--only", case, "--cache-dir", str(cache)]
    p = subprocess.Popen(argv, cwd=str(shadow), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, start_new_session=True)
    with LIVE_LOCK:
        LIVE[p.pid] = p
    try:
        out, _ = p.communicate(timeout=TIMEOUT)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_group(p)
        out, _ = p.communicate()
        timed_out = True
    finally:
        _kill_group(p)
        with LIVE_LOCK:
            LIVE.pop(p.pid, None)
    statuses = {}
    in_checks = False
    for line in out.splitlines():
        if line.strip() == "[CHECKS]":
            in_checks = True
            continue
        m = re.match(r"^  (\S+)\s*: (PASS|FAIL)", line) if in_checks else None
        if m:
            statuses[m.group(1)] = m.group(2)
    tail = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    res = ("TIMEOUT " if timed_out else "") + (tail[-1] if tail else out.strip()[-200:])
    return statuses, res, time.time() - t0


def _kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _on_signal(signum, _frame) -> None:
    STOP.set()
    with LIVE_LOCK:
        for p in list(LIVE.values()):
            _kill_group(p)


# ------------------------------------------------------------------ the edits
def edited(text: str, m: dict) -> str:
    if text.count(m["old"]) != 1:
        raise ValueError(f"{m['id']}: the target occurs {text.count(m['old'])} times in {m['rel']}")
    out = text.replace(m["old"], m["new"])
    ast.parse(out)                                             # every edit still compiles
    return out


def select(only: str | None) -> list[dict]:
    if not only:
        return list(MUTATIONS)
    wanted = set(only.split(","))
    unknown = sorted(wanted - {m["id"] for m in MUTATIONS})
    if unknown:
        raise SystemExit(f"RESULT: ERROR unknown mutation ids {unknown}")
    return [m for m in MUTATIONS if m["id"] in wanted]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated mutation ids")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--check-edits", action="store_true",
                    help="only check that every edit applies once to the repository's files and still parses")
    args = ap.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(sig, _on_signal)
    jobs = select(args.only)
    ids = [m["id"] for m in MUTATIONS]
    check("mutation_ids_unique", len(ids) == len(set(ids)), str(len(ids)))
    for m in jobs:
        try:
            edited((REPO / m["rel"]).read_text(encoding="utf-8"), m)
            ok, detail = True, ""
        except (ValueError, SyntaxError, OSError) as e:
            ok, detail = False, str(e)
        check(f"{m['id']}_edit_applies", ok, detail)
    if args.check_edits or not all(ok for _, ok, _ in results):
        return report(time.time())
    t0 = time.time()
    base = safe_tmpdir("lqp_mutations_")
    cache = base / "cache"
    cache.mkdir()
    shadows = [build_shadow(base / f"shadow{i}") for i in range(max(1, args.workers))]
    print(f"[harness] {len(jobs)} edits | {len(shadows)} shadow(s) under {base} | cache {cache}", flush=True)
    cases = sorted({m["case"] for m in jobs})
    baseline: dict[tuple[str, int], dict] = {}
    # the first case fills the cache (TRAIN tree, checkpoints) on shadow 0; then every case on every shadow
    for i, sh in enumerate(shadows):
        for case in cases:
            statuses, res, secs = run_case(sh, cache, case)
            baseline[(case, i)] = statuses
            print(f"[baseline] shadow{i} {case}: {res} ({secs:.0f}s)", flush=True)
    for m in jobs:
        for i in range(len(shadows)):
            st = baseline[(m["case"], i)]
            missing = [k for k in m["kills"] if st.get(k) != "PASS"]
            check(f"{m['id']}_baseline_passes_shadow{i}", not missing, f"{m['case']}: {missing}")
    free: queue.Queue = queue.Queue()
    for i in range(len(shadows)):
        free.put(i)
    verdicts: dict[str, tuple[bool, str]] = {}

    def job(m: dict) -> None:
        if STOP.is_set():
            return
        i = free.get()
        sh = shadows[i]
        f = sh / m["rel"]
        original = f.read_bytes()
        statuses, res, secs = {}, "not run", 0.0
        try:
            f.write_text(edited(original.decode("utf-8"), m), encoding="utf-8")
            statuses, res, secs = run_case(sh, cache, m["case"])
        finally:
            f.write_bytes(original)
            restored = f.read_bytes() == original and shadow_status(sh) == ""
            free.put(i)
        killed = [k for k in m["kills"] if statuses.get(k) == "FAIL"]
        verdicts[m["id"]] = (bool(killed) and restored, f"{m['case']}: killed by {killed or 'nothing'}; {res} "
                                                         f"({secs:.0f}s)" + ("" if restored else " NOT RESTORED"))
        print(f"[edit] {m['id']} {m['what']}: {verdicts[m['id']][1]}", flush=True)

    with ThreadPoolExecutor(max_workers=len(shadows)) as pool:
        list(pool.map(job, jobs))
    for m in jobs:
        ok, detail = verdicts.get(m["id"], (False, "no verdict (stopped)"))
        check(f"{m['id']}_killed", ok, f"{m['what']} -> {detail}")
    check("harness_ran_unsignalled", not STOP.is_set(), "a stop signal ended the run" if STOP.is_set() else "")
    check("every_shadow_clean_at_the_end", all(shadow_status(sh) == "" for sh in shadows))
    return report(t0)


def report(t0: float) -> int:
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    killed = sum(1 for n, ok, _ in results if n.endswith("_killed") and ok)
    edits = sum(1 for n, _, _ in results if n.endswith("_killed"))
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n[time] {time.time() - t0:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)} checks; "
          f"{killed}/{edits} edits killed)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
