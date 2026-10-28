#!/usr/bin/env python3
"""Lane L-FQOFF: the mutation smoke of AM-4's fake-quant-off path (precision qat_fakequant_off). The FQ mutations
(phase 1: FQ1-FQ7, FQ9, FQ10, FQ12 and FQ15; phase 2: FQ8, FQ11, FQ14a and FQ14b; DL-24's fixes: FQ16-FQ19) and the
refusal table of the fqoff codes (FQR12's 1-22: phase 2's 1-5 in scripts/evaluate_model.py, phase 1's 6-22 in
src/eval/stage_artifacts.py and src/eval/model_loading.py), each killed by a named check of
scripts/smoke_eval_qat_fakequant_off.py. FQ13 has no row:
LEFF B-1b sets --threads right after validate_cli_args, before any precision branch, so this precision's path never
applies it itself (the GO's rule for FQ13 and FQ14); the smoke's run_threads_recorded shows it holds.

Q2's harness, scripts/smoke_qat_mutations.py, runs it unchanged, as scripts/smoke_e6kd_mutations.py has it run:
- its self-check (ruling Q2-F/2's job handling, measured, with harness processes of its own);
- its shadow: SHADOW_PATHS copied into a temporary directory and committed there, so the repository is never written;
- its jobs: each smoke in a session of its own, the named check reported FAIL after it PASSED on the unmutated
  shadow, with the same argv.
Q2's refusal finder sees `_refuse(...)` and raises of QATRefused, QATStop and QATIncomplete only, so this file gives
the harness its own (fqoff_refusal_sites): in each of the four evaluator files, a statement calling that file's raise
form with a literal fqoff code first (stage_artifacts.py's `_fail("fqoff_...", ...)`, model_loading.py's
`_fqoff_refuse("fqoff_...", ...)`, evaluate_model.py's `_fqoff_cli("fqoff_...", ...)` from phase 2), inside a SCOPE
function; a removed refusal is that statement replaced by `pass`, as Q2's are.

This file gives the harness the lane's shadow paths, mutations, refusal scope, refusal table and site finder, through
the harness module's globals in this process only. Nothing in Q2's file changes.

    python -B scripts/smoke_eval_qat_fakequant_off_mutations.py [--only ID,ID,...] [--no-refusals] [--workers N]
                                                                [--keep-shadow] [--self-check]

Ends with one RESULT line; exit 0 only when every baseline check passes and every edit is killed.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.smoke_qat_mutations as H  # noqa: E402

SAF, ML, ART, EM = ("src/eval/stage_artifacts.py", "src/eval/model_loading.py", "src/eval/artifacts.py",
                    "scripts/evaluate_model.py")
SMOKE = "smoke_eval_qat_fakequant_off"
SMOKE_PATH = "scripts/smoke_eval_qat_fakequant_off.py"
# the two lane files, and every other file a cold run of the smoke opens that Q2's shadow does not hold (measured with
# an audit hook on open, imports included): among them AM-23's and E6-KD's rules files, src/stats/driver.py (read as
# text, never imported) and E6-KD's trainer, src/quant/qat_kd.py
SHADOW_PATHS = (*H.SHADOW_PATHS, "scripts/smoke_eval_qat_fakequant_off.py",
                "scripts/smoke_eval_qat_fakequant_off_mutations.py", "configs/qat_selection_rules_am23.json",
                "configs/e6kd_selection_rules.json", "src/stats/driver.py", "configs/distill.py",
                "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py", "scripts/e6kd_eval.py",
                "scripts/invariance_harness.py", "scripts/preflight_distill.py", "scripts/preflight_e1.py",
                "scripts/preflight_e1_trainval.py", "scripts/run_e6_e2.py", "scripts/run_e6_r1.py",
                "scripts/run_e6_r2.py", "scripts/run_e6kd.py", "src/quant/e6kd_trigger.py",
                "src/quant/e6x_condition.py", "src/quant/qat_kd.py", "src/training/teacher_components.py",
                "src/training/train_distill.py")


def case(*names: str) -> tuple:
    """Cases of scripts/smoke_eval_qat_fakequant_off.py, its raw QAT runs kept in the cache."""
    return (SMOKE, ("--only", ",".join(names), "--cache-dir", "{cache}/fqoff"))


M, MM = H.M, H.MM
MUTATIONS = [
    M("FQ1", "fake quantization left on (disable_fake_quant not applied)", ML,
      "    model.apply(disable_fake_quant)\n", "", case("load_fake_quant_off_model"), "load_fake_quant_off_model"),
    M("FQ2", "observers left on (disable_observer not applied)", ML,
      "    model.apply(disable_observer)\n", "", case("load_fake_quant_off_model"), "load_fake_quant_off_model"),
    M("FQ3", "a non-strict load: Q2's copy by name without load_state_by_name's exactness checks", ML,
      '        load_state_by_name(model, payload["model_state_dict"])\n',
      "        from ..quant.x86_latency import copy_qat_state_by_name\n"
      '        copy_qat_state_by_name(model, payload["model_state_dict"])\n',
      case("c21_state_key_missing"), "c21_state_key_missing"),
    M("FQ4", "the selection's sha256 not compared with its pin", SAF,
      "    if sel_sha is None or sel_sha != selection_sha256:\n", "    if sel_sha is None:\n",
      case("c07_selection_pin_differs"), "c07_selection_pin_differs"),
    M("FQ5", "the winner's epoch file unchecked: any checkpoint of the run accepted", SAF,
      '    if ck is None or end.get("checkpoint") != want_ck or ck.resolve() != winner_file:\n',
      '    if ck is None or end.get("checkpoint") != want_ck:\n',
      case("c15_checkpoint_other_epoch"), "c15_checkpoint_other_epoch"),
    M("FQ6", "a converted TorchScript archive passed to the payload load as if it were the eNN.pt", SAF,
      "    if _is_torchscript_archive(ck_raw):\n", "    if False:\n",
      case("c18_torchscript_archive"), "c18_torchscript_archive"),
    M("FQ7", "the stored state_finite unchecked", SAF,
      '    if payload.get("state_finite") is not True:\n', "    if False:\n",
      case("c20_state_not_finite"), "c20_state_not_finite"),
    M("FQ9", "the census skipped", ML,
      '    if flags != {"observer_enabled": [0], "fake_quant_enabled": [0]} or not fq or training:\n',
      "    if False:\n", case("c22_toggle_not_taken"), "c22_toggle_not_taken"),
    M("FQ10", "artifacts.py's precision rule removed", ART,
      '    if req.run.precision == "qat_fakequant_off" and not (\n', "    if False and not (\n",
      case("stages_rule_refuses_cut_stage"), "stages_rule_refuses_cut_stage"),
    M("FQ12", "the finalize record ignored when present", SAF,
      "    if fin_p.exists() or fin_p.is_symlink():\n", "    if False:\n",
      case("c14_finalize_selection_sha256"), "c14_finalize_selection_sha256"),
    M("FQ15", "the loader's re-hash of the eNN.pt's bytes before it loads them removed", ML,
      '    if digest != fqoff["checkpoint_sha256"]:\n', "    if False:\n",
      case("load_file_changed_after_validation"), "load_file_changed_after_validation"),
    # ---- phase 2: scripts/evaluate_model.py
    M("FQ8", "a quantization backend recorded for this precision", EM,
      '        ckpt_sha = fqoff["checkpoint_sha256"]\n',
      '        ckpt_sha = fqoff["checkpoint_sha256"]\n        quant_backend = "x86"\n',
      case("summary_precision_and_backend"), "summary_precision_and_backend"),
    M("FQ11", "the load dispatch's branch of this precision removed: the eNN.pt reaches load_student_checkpoint", EM,
      '    elif args.precision == "qat_fakequant_off":\n'
      "        from src.eval.model_loading import load_qat_fakequant_off_student\n"
      "        model = load_qat_fakequant_off_student(fqoff)        # lane L-FQOFF (FQR9, FQR11): ck_info stays None\n",
      "", case("run_loads_through_fqoff_loader"), "run_loads_through_fqoff_loader"),
    M("FQ14a", "LEFF B-1b's identity flag skipped at P1 for this precision: its handed-file case names the run dir",
      EM, "        return Path(args.checkpoint)\n", "        return Path(args.qat_run_dir)\n",
      case("run_h_mismatch_refused_before_parse"), "run_h_mismatch_refused_before_parse"),
    M("FQ14b", "LEFF B-1b's identity flag skipped at P2 for this precision: neither the carrier's sha256 nor the "
      "re-hashed file compared", EM,
      '            after = fqoff["checkpoint_sha256"]\n            if after == expected:\n',
      "            after = expected\n            if False:\n",
      case("run_h_changed_during_validation"), "run_h_changed_during_validation"),
    # ---- DL-24's confirmed findings (A5): each fix's mutation, killed by its case
    M("FQ16", "the payload loaded from the path again, not from the bytes the validator hashed (DL-24 R3-F1)", SAF,
      '        payload = validate_qat_sidecar(torch.load(io.BytesIO(ck_raw), map_location="cpu", weights_only=True),\n',
      '        payload = validate_qat_sidecar(torch.load(ck, map_location="cpu", weights_only=True),\n',
      case("bind_reads_the_checkpoint_once"), "bind_reads_the_checkpoint_once"),
    M("FQ17", "an unreadable finalize record's OSError escapes code 14 (DL-24 R1-F2)", SAF,
      "        try:\n            fin_raw = fin_p.read_bytes() if fin_p.is_file() else None\n"
      "        except OSError:\n            fin_raw = None\n",
      "        fin_raw = fin_p.read_bytes() if fin_p.is_file() else None\n",
      case("c14_finalize_unreadable_file"), "c14_finalize_unreadable_file"),
    M("FQ18", "the smoke leaves its temp directory behind (DL-24 R4-1)", SMOKE_PATH,
      "    shutil.rmtree(TMP, ignore_errors=True)               # its temp directory (DL-24 R4-1); a cache stays\n",
      "", case("hygiene_temp_removed"), "hygiene_temp_removed"),
    M("FQ19", "P2's re-hash of the handed file skipped for this precision: a file replaced after the validator returns "
      "passes P2 (DL-24 R3-F1, LFQOFF-RULE-3)", EM,
      "                after = identity_sha256(handed)\n", "                pass\n",
      case("run_h_changed_after_validation"), "run_h_changed_after_validation"),
]

# ------------------------------------------------------------------ the refusal table and its finder (FQR12)
# file -> the raise form of its fqoff refusals, and the functions that hold them, in FQR12's order of their codes
# (artifacts.py's rule has no code of its own, FQR3)
FORMS = {SAF: "_fail", ML: "_fqoff_refuse", EM: "_fqoff_cli"}
SCOPE = {EM: {"validate_qat_fakequant_off_cli"}, SAF: {"validate_qat_fakequant_off_source"},
         ML: {"load_qat_fakequant_off_student"}, ART: set()}


def fqoff_refusal_sites(text: str, rel: str) -> list[dict]:
    """The fqoff refusal sites of one of the four evaluator files, keyed as Q2's refusal_sites keys them (function,
    code, k). A site is a statement calling the file's raise form inside a SCOPE function, with a literal fqoff code
    first; a call of the form whose first argument is not a literal is a site too, so that the refusal table names it
    or the harness reports it."""
    form, scope = FORMS.get(rel), SCOPE.get(rel) or set()
    if form is None:
        return []
    tree = ast.parse(text)
    owner = {}
    for fn in sorted((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                     key=lambda n: n.lineno):
        for n in ast.walk(fn):
            if n is not fn:
                owner[id(n)] = fn.name
    out, seen = [], {}
    for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.args):
            continue
        f = node.value.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        func = owner.get(id(node))
        if name != form or func not in scope:
            continue
        a0 = node.value.args[0]
        literal = isinstance(a0, ast.Constant) and isinstance(a0.value, str)
        code = a0.value if literal else ast.unparse(a0)
        if literal and not code.startswith("fqoff_"):
            continue
        k = seen.get((func, code), 0)
        seen[(func, code)] = k + 1
        out.append({"rel": rel, "func": func, "code": code, "k": k, "node": node, "literal": literal})
    return out


def K(rel: str, func: str, code: str, *checks: str, k: int = 0) -> tuple:
    return H.K(rel, func, code, case(*checks), *checks, k=k)


def V(code: str, *checks: str, k: int = 0) -> tuple:
    """A site of validate_qat_fakequant_off_source (codes 6-20)."""
    return K(SAF, "validate_qat_fakequant_off_source", code, *checks, k=k)


def L(code: str, *checks: str, k: int = 0) -> tuple:
    """A site of load_qat_fakequant_off_student (codes 21-22)."""
    return K(ML, "load_qat_fakequant_off_student", code, *checks, k=k)


def C(code: str, *checks: str, k: int = 0) -> tuple:
    """A site of scripts/evaluate_model.py's validate_qat_fakequant_off_cli (codes 1-5)."""
    return K(EM, "validate_qat_fakequant_off_cli", code, *checks, k=k)


FINALIZE_CASES = ("c14_finalize_unreadable", "c14_finalize_format", "c14_finalize_status",
                  "c14_finalize_selection_sha256", "c14_finalize_checkpoint", "c14_finalize_run_id",
                  "c14_finalize_telemetry", "c14_finalize_stage")
KILLS = dict([
    C("fqoff_stage_not_in_scope", "c01_stage_not_in_scope"),
    C("fqoff_stage_not_in_scope", "c01_role_not_student", k=1),
    C("fqoff_provenance_forbidden", "c02_provenance_forbidden"),
    C("fqoff_source_flags", "c03_checkpoint_missing"),
    C("fqoff_source_flags", "c03_binding_flag_missing", k=1),
    C("fqoff_source_flags", "c03_random_init", k=2),
    C("fqoff_source_flags", "c03_sha256_malformed", k=3),
    C("fqoff_not_cpu", "c04_not_cpu"),
    C("fqoff_protocol", "c05_protocol_upstream"),
    C("fqoff_source_flags", "c03_qat_flag_off_precision", k=4),
    V("fqoff_forbidden_path", "c06_forbidden_checkpoint", "c06_forbidden_run_dir", "c06_forbidden_selection"),
    V("fqoff_selection_sha256_mismatch", "c07_selection_absent", "c07_selection_pin_differs"),
    V("fqoff_selection_not_of_record", "c08_selection_not_named_of_record"),
    V("fqoff_selection_not_of_record", "c08_selection_not_json", "c08_selection_format", k=1),
    V("fqoff_selection_not_of_record", "c08_selection_rules_sha256", k=2),
    V("fqoff_selection_not_of_record", "c08_selection_smoke_inputs", k=3),
    V("fqoff_selection_not_of_record", "c08_winner_epoch_not_int", "c08_winner_epoch_out_of_range",
      "c08_winner_sha_malformed", k=4),
    V("fqoff_stage_mismatch", "c09_selection_stage"),
    V("fqoff_run_record", "c10_run_record_missing", "c10_run_dir_none"),
    V("fqoff_run_incomplete", "c11_run_incomplete"),
    V("fqoff_run_not_real", "c12_run_not_real"),
    V("fqoff_stage_mismatch", "c09_run_meta_stage", k=1),
    V("fqoff_telemetry_sha256_mismatch", "c13_telemetry_pin", "c13_telemetry_selection"),
    V("fqoff_selection_not_of_record", "c08_selection_run_id", k=5),
    V("fqoff_finalize_disagrees", *FINALIZE_CASES),
    V("fqoff_not_the_winner", "c15_winner_checkpoint_not_its_epoch"),
    V("fqoff_not_the_winner", "c15_checkpoint_other_epoch", "c15_row_names_other_file", k=1),
    V("fqoff_checkpoint_missing", "c16_checkpoint_missing"),
    V("fqoff_checkpoint_sha256_mismatch", "c17_bytes_differ_from_row", "c17_winner_sha_differs"),
    V("fqoff_not_qat_train_state", "c18_torchscript_archive"),
    V("fqoff_not_qat_train_state", "c18_wrong_kind", "c18_unloadable", k=1),
    V("fqoff_not_qat_train_state", "c18_wrong_role", k=2),
    V("fqoff_stage_mismatch", "c09_payload_stage", k=2),
    V("fqoff_checkpoint_identity", "c19_identity_epoch", "c19_identity_run_id", "c19_identity_seed",
      "c19_identity_source"),
    V("fqoff_state_not_finite", "c20_state_not_finite", "c20_state_flag_missing"),
    L("fqoff_load_not_exact", "c21_state_key_missing", "c21_state_shape"),
    L("fqoff_fake_quant_enabled", "c22_toggle_not_taken"),
])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated job ids (FQ1, ..., or file:function:code#k)")
    ap.add_argument("--no-refusals", action="store_true", help="the FQ mutations only")
    ap.add_argument("--workers", type=int, default=2, help="shadows edited and run in parallel")
    ap.add_argument("--keep-shadow", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="Q2's harness self-check only (ruling Q2-F/2's job handling): no shadow and no smoke")
    args = ap.parse_args()
    H.install_stop_handlers()                        # first: the jobs run in sessions of their own
    H.SHADOW_PATHS, H.MUTATIONS, H.SCOPE, H.KILLS = SHADOW_PATHS, MUTATIONS, SCOPE, KILLS
    H.refusal_sites = fqoff_refusal_sites            # jobs_of and edited call it as a module global
    t_all = time.time()
    H.self_check()
    measured = {name for name, _ok, _detail in H.results}
    H.check("self_check_measured", set(H.SELF_CHECKS) <= measured,
            f"{len(H.SELF_CHECKS)} checks" if set(H.SELF_CHECKS) <= measured
            else f"not measured: {sorted(set(H.SELF_CHECKS) - measured)}")
    if args.self_check:
        return H.finish(t_all)
    root = H.safe_tmpdir("smoke_fqoff_mutations_")
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
    print("L-FQOFF MUTATION SMOKE (phases 1 and 2) -- the FQ mutations and the fqoff refusal table, each killed by a "
          "named check")
    print(f"{len(jobs)} edits | {len(SHADOW_PATHS)} shadow files x {len(shadows)} at {head} | temp {root}")
    print("=" * 78)
    H.run_jobs(jobs, shadows, root / "cache")
    verdicts = {name for name, _ok, _detail in H.results}
    missing = [j["id"] for j in jobs if f"{j['id']}_killed" not in verdicts and f"{j['id']}_baseline" not in verdicts]
    H.check("every_job_has_a_verdict", not missing,
            f"no verdict for {missing[:5]}" if missing
            else f"{len(jobs)} jobs, each killed or with its baseline failing")
    if not args.keep_shadow:
        shutil.rmtree(root, ignore_errors=True)
    return H.finish(t_all)


if __name__ == "__main__":
    raise SystemExit(main())
