#!/usr/bin/env python3
"""Smoke: the mutation harness of L-AM23Q B-1, the E6-X condition tool (the Phase A plan §4.2; the B-1 GO: every new
refusal code has a case reached with every earlier gate passing and a mutation that a named case kills).

Each mutation rewrites one function of src/quant/e6x_condition.py (or scripts/e6x_condition.py's main) from its own
source: exact text replacements, each occurring exactly once in that function, compiled into the module's namespace
and restored afterwards (scripts/smoke_teacher_diag_mutations.py's mutate, :56-81, copied below). Its killing case is
a case of scripts/smoke_e6x_condition.py, run on that smoke's synthetic fixtures (built here, in this harness's own
work dir). Three checks per mutation:
  <id>_anchor   every replacement occurs exactly once in the function's source
  <id>_control  the killing case passes unmutated
  <id>_killed   the killing case fails (or raises) under the mutation
and two more: every_code_killed (every refusal, incomplete and STOP code the tool names is the code of a killing
case) and every_case_named (every killing case exists). No mutant writes inside the repository: a mutant that drops
an outside-the-repository guard meets a never-created folder there, and the harness runs from a folder of its work
dir, so a relative path never names the repository. Last, the smoke's three hygiene cases on this run: no path
named "test" or with ':' created, none opened, the repository untouched.

    python -B scripts/smoke_e6x_condition_mutations.py [--work-dir DIR] [--only M01,M02]
"""
from __future__ import annotations

import __future__ as _future
import argparse
import contextlib
import importlib
import inspect
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

S = importlib.import_module("scripts.smoke_e6x_condition")
C = S.C
OWNERS = {"C": S.C, "CLI": S.CLI}
RESULTS: list = []

RUN_META_KEYS = ("mode", "stage", "seed", "arm", "descriptive", "parent_of_e4_e7", "max_iters", "poly_horizon",
                 "target_construction", "lambda_logit", "alpha_cwd", "teacher_mock", "teacher_provenance.ckpt_sha256")
PAYLOAD_KEYS = ("stage", "arm", "target_construction", "best_val_miou_all_class")
LOG_KEYS = ("run_meta_sha256", "git_head", "code_pin", "records_commit", "stage", "seed", "horizon", "sweep",
            "lambda_logit", "alpha_cwd", "launch_time_utc")
SAME_META = "        if not SS._same(got, want):"
SAME_PAYLOAD = "        if not SS._same(payload.get(key), want):"
SAME_END = "        if not SS._same(end.get(key), want):"
TEACHER_READ = "SS.HeadSource(root).read(TEACHER_RECORD_REL.format(arm=arm), missing="
WORKING_TREE_READ = "(lambda rel, **k: (rel, (root / rel).read_bytes()))(TEACHER_RECORD_REL.format(arm=arm), missing="
SHA_FORM = "    if not isinstance(expect_sha256, str) or not _HEX64.fullmatch(expect_sha256):"
END_FORMAT = "    if not (type(it) is int and it == HORIZON and SS._finite_number(t)):"


def _skip(key: str, anchor: str) -> list:
    """The comparison of one key dropped from a loop over (key, ...) pairs."""
    return [(anchor, anchor.replace("if not ", f"if key != {key!r} and not ", 1))]


# (id, owner, function, [(old, new), ...], killing case, what the mutant does)
MUTATIONS = [
    # ---- the verdict (AM-23 item 3(b): Decimal(repr(value)) > Decimal(threshold), strictly)
    ("M01", "C", "verdict", [('> Decimal(THRESHOLD) else "no"', '>= Decimal(THRESHOLD) else "no"')],
     "w_equal_is_no", "the strict > made >="),
    ("M02", "C", "verdict", [("Decimal(repr(float(value))) > Decimal(THRESHOLD)",
                              "round(float(value), 6) > round(float(THRESHOLD), 6)")],
     "w_one_ulp_above_is_yes", "the decimal comparison made one after rounding to six places"),
    ("M03", "C", "build_record", [('"verdict": verdict(v),', '"verdict": "yes",')],
     "w_no_written", "the verdict not derived from the value"),
    # ---- AQ4: a value that cannot be compared
    ("M04", "C", "value_error", [("    if not isinstance(x, float):", "    if isinstance(x, str):")],
     "r_value_invalid_bool_run_end", "AQ4's type rule reduced to 'not a string': a bool or an int passes"),
    ("M05", "C", "best_checkpoint", [("        err = value_error(x)", "        err = None")],
     "r_value_invalid_nan_best_json", "best.json's and run_end's values not checked"),
    ("M06", "C", "best_checkpoint", [('    err = value_error(payload.get("best_val_miou_all_class"))', "    err = None")],
     "r_value_invalid_int_payload", "the payload's value not checked"),
    # ---- G0, G1
    ("M07", "CLI", "main", [("    if args.command is None:", "    if False:")],
     "r_no_command", "the command not required"),
    ("M08", "C", "check_flags", [("    if missing:", "    if False:")],
     "r_missing_flags", "G0 removed"),
    ("M09", "C", "check_names", [("        if Q.names_test(value):", "        if False:")],
     "r_forbidden_path_run_dir", "G1 removed (SL-1, AQ6)"),
    # ---- G2
    ("M10", "C", "check_out", [("    if not Path(value).is_absolute():", "    if False:")],
     "r_out_relative", "--out may be relative"),
    ("M11", "C", "check_out", [("        out = R.check_output_dir(value)", "        out = Path(value)")],
     "r_out_inside_repo", "E1's outside-the-repository guard dropped for --out"),
    ("M12", "C", "check_out", [("    if not ARM_RE.fullmatch(arm) or out.name != want:", "    if False:")],
     "r_out_name", "--out's name not checked"),
    ("M13", "C", "check_out", [("    if os.path.lexists(out):", "    if False:")],
     "r_out_exists", "G2's existence check removed"),
    ("M14", "C", "check_out", [("    if not out.parent.is_dir():", "    if False:")],
     "r_out_parent_missing", "--out's folder not checked"),
    ("M15", "C", "write_record", [('with open(out, "xb") as fh:', 'with open(out, "wb") as fh:')],
     "r_out_created_before_write", "the exclusive creation made an overwrite"),
    # ---- G3 (AQ5)
    ("M16", "C", "tool_identity", [("    if not _HEX40.fullmatch(expect_head):", "    if False:")],
     "r_expect_head_format", "--expect-head's form not checked"),
    ("M17", "C", "tool_identity", [('    if prov["commit"] is None:', "    if False:")],
     "r_code_identity_unavailable", "a missing commit not refused by name"),
    ("M18", "C", "tool_identity", [('    if prov["commit"] != expect_head:', "    if False:")],
     "r_expect_head_mismatch", "--expect-head not compared"),
    ("M19", "C", "tool_identity", [('    if prov["clean_at_commit"] is not True:', "    if False:")],
     "aq5_untracked_arm_file_not_clean", "cleanliness not required: an untracked arm file passes (AQ5)"),
    ("M20", "C", "tool_identity", [('    if prov["clean_at_commit"] is not True:', "    if False:")],
     "r_code_not_clean", "cleanliness not required"),
    # ---- G4
    ("M21", "C", "arm_spec", [("    except K.ArmSpecError as e:", "    except KeyError as e:")],
     "r_arm_spec", "a broken arm registry not refused by name"),
    ("M22", "C", "arm_spec", [("    if len(found) != 1:", "    if False:")],
     "r_arm_unknown", "an unregistered arm not refused"),
    ("M23", "C", "arm_spec", [('s.teacher == "arm" and s.arm_teacher == arm', 's.teacher == "arm"')],
     "w_yes_written", "the arm teacher not matched (both arms found)"),
    # ---- G5
    ("M24", "C", "selections",
     [("    if args.lambda_selection != LAMBDA_SELECTION_REL or args.alpha_selection != ALPHA_SELECTION_REL:",
       "    if False:")],
     "r_selection_not_of_record", "any committed selection file accepted"),
    ("M25", "C", "selections",
     [('    if not (SS._finite_number(lam["winner"].get("lambda")) and SS._finite_number(alpha["winner"].get("alpha"))):',
       "    if False:")],
     "r_selection_format_winner", "the winners' lambda and alpha not checked"),
    ("M26", "C", "selections", [("    if not (isinstance(best, float) and repr(best) == THRESHOLD):", "    if False:")],
     "r_threshold_mismatch", "the threshold not checked against the α winner"),
    # ---- G6
    ("M27", "C", "teacher_record", [(TEACHER_READ, WORKING_TREE_READ)],
     "r_teacher_record_uncommitted", "the teacher record read from the working tree, not at HEAD"),
    ("M28", "C", "teacher_record", [(TEACHER_READ, WORKING_TREE_READ)],
     "r_teacher_record_missing", "the teacher record read from the working tree, not at HEAD"),
    ("M29", "C", "teacher_record", [(' and doc["status"] == "complete"', "")],
     "r_teacher_record_invalid", "the teacher record's status not checked"),
    # ---- G7
    ("M30", "C", "run_dir", [("    if not Path(value).is_absolute():", "    if False:")],
     "r_run_dir_relative", "--run-dir may be relative"),
    ("M31", "C", "run_dir", [("        d = R.check_output_dir(value)", "        d = Path(value)")],
     "r_run_dir_inside_repo", "E1's outside-the-repository guard dropped for --run-dir"),
    ("M32", "C", "run_dir",
     [("    if not (d.name.startswith(prefix) and RUN_TAIL_RE.fullmatch(d.name[len(prefix):])):", "    if False:")],
     "r_run_dir_name", "the run directory's name not checked"),
    ("M33", "C", "run_dir", [("    if missing:", "    if False:")],
     "r_run_records_missing", "a missing record file not refused by name"),
    ("M34", "C", "run_meta_row", [("    if torn or len(rows) != 1 or", "    if torn or len(rows) < 1 or")],
     "r_run_meta_rows", "an appended run_meta row accepted"),
    *[(f"M{35 + i}", "C", "run_meta_checks", _skip(k, SAME_META), f"r_run_meta_{k.replace('.', '_')}",
       f"run_meta {k} not compared") for i, k in enumerate(RUN_META_KEYS)],                          # M35-M47
    # ---- G8 (AM-19 Definitions: complete)
    ("M48", "C", "completion", [("    except ValueError as e:", "    except KeyError as e:")],
     "r_telemetry_format_line", "a malformed telemetry line not refused by name"),
    ("M49", "C", "completion", [("    if not all(isinstance(r, dict) for r in rows):", "    if False:")],
     "r_telemetry_format_row", "a telemetry row that is not an object accepted"),
    ("M50", "C", "completion", [("    if torn:", "    if False:")],
     "r_telemetry_torn", "a torn last line ignored"),
    ("M51", "C", "completion", [('    if any(r.get("event") == "run_abort" for r in rows):', "    if False:")],
     "r_parent_aborted", "a run_abort row ignored"),
    ("M52", "C", "completion", [("    if not ends:", "    if False:")],
     "i_run_end_missing", "a run without run_end not reported incomplete"),
    ("M53", "C", "completion", [("    if len(ends) != 1 or rows[-1] is not ends[0]:", "    if len(ends) != 1:")],
     "r_run_end_not_last", "rows after run_end accepted"),
    ("M54", "C", "completion", [('    if end.get("checks_passed") is not True:', "    if False:")],
     "r_parent_checks_failed", "checks_passed ignored"),
    ("M55", "C", "completion", [(END_FORMAT, "    if not SS._finite_number(t):")],
     "r_run_end_format_iter", "run_end's iter not checked"),
    ("M56", "C", "completion", [(END_FORMAT, "    if not (type(it) is int and it == HORIZON):")],
     "r_run_end_format_clock", "run_end's wall_clock_end not checked"),
    # ---- G9
    ("M57", "C", "best_checkpoint", [("sorted(best) != sorted(SS.BEST_JSON_KEYS)", '"best_ckpt" not in best')],
     "r_best_json_format_keys", "best.json's keys not checked"),
    ("M58", "C", "best_checkpoint", [("    if not CKPT_RE.fullmatch(name):", "    if False:")],
     "r_best_json_format_name", "best.json's checkpoint name not checked"),
    ("M59", "C", "best_checkpoint", [("    if not ck.is_file():", "    if False:")],
     "r_checkpoint_missing", "a missing checkpoint not refused by name"),
    ("M60", "C", "best_checkpoint",
     [("    except Exception as e:  # noqa: BLE001 -- any unreadable checkpoint is refused by name",
       "    except KeyError as e:  # noqa: BLE001 -- any unreadable checkpoint is refused by name")],
     "r_payload_unreadable", "an unreadable checkpoint not refused by name"),
    ("M61", "C", "best_checkpoint", [("    if not isinstance(payload, dict):", "    if False:")],
     "r_payload_not_dict", "a payload that is not a dict accepted"),
    *[(f"M{62 + i}", "C", "best_checkpoint", _skip(k, SAME_PAYLOAD), f"r_payload_{k}",
       f"the payload's {k} not compared") for i, k in enumerate(PAYLOAD_KEYS)],                       # M62-M65
    ("M66", "C", "best_checkpoint", _skip("best_val_miou_all_class", SAME_END),
     "r_run_end_best", "run_end's best not compared"),
    ("M67", "C", "best_checkpoint", _skip("best_ckpt", SAME_END),
     "r_run_end_ckpt", "run_end's best_ckpt not compared"),
    ("M68", "C", "best_checkpoint", [('"checkpoint_sha256": Q.sha256_bytes(raw),',
                                      '"checkpoint_sha256": Q.sha256_bytes(best_data),')],
     "w_record_fields", "the checkpoint's sha256 taken of best.json"),
    # ---- G10 (PL-13, PL-14)
    ("M69", "C", "launch_binding", [("    if line is None:", "    if False:")],
     "r_launch_log_launch", "a run without a launch line accepted"),
    ("M70", "C", "launch_binding", [("    if ld is None:", "    if False:")],
     "r_launch_log_launched", "a run without a launched line accepted"),
    ("M71", "C", "launch_binding", [("    if rid in log.stopped:", "    if False:")],
     "r_launch_log_stopped", "a stopped line ignored"),
    *[(f"M{72 + i}", "C", "launch_binding", _skip(k, SAME_META), f"r_launch_log_{k}",
       f"PL-14's {k} not compared") for i, k in enumerate(LOG_KEYS)],                                  # M72-M82
    # ---- the record, its deadline and its row (AQ3, AQ7)
    ("M83", "C", "deadline", [("    enter_by = day + timedelta(days=1)", "    enter_by = day")],
     "w_deadline_before_manila_midnight", "the deadline the day of run_end, not the next day"),
    ("M84", "C", "deadline", [("    day = end.astimezone(SS.MANILA).date()", "    day = end.date()")],
     "w_deadline_after_manila_midnight", "run_end's day taken in UTC, not in Manila"),
    ("M85", "C", "build_record", [('"format": FORMAT,', '"format": FORMAT, "written_at": __import__("time").time(),')],
     "w_deterministic", "a clock time in the record"),
    ("M86", "C", "build_record", [("Q.sha256_bytes(meta_data)", "Q.sha256_bytes(tel_data)")],
     "w_record_fields", "run_meta's sha256 taken of the telemetry"),
    ("M87", "C", "dl_text", [(" sha256 {record_sha256} ", " ")],
     "w_dl_text_one_line_aq3", "the record's sha256 left out of the decision-log text"),
    ("M88", "C", "dl_text", [('    if "|" in text or "\\n" in text or "\\r" in text or not text.isascii():', "    if False:"),
                             ('); best VAL all-class mIoU "', ') | best VAL all-class mIoU "')],
     "w_dl_text_one_line_aq3", "a '|' in the decision-log text, its guard removed"),
    ("M89", "C", "write_record", [("    if _read_back(out) != data:", "    if False:")],
     "s_write_mismatch", "the read-back check removed"),
    # ---- the readers (print-dl's read_record_file; B-2's read_condition)
    ("M90", "C", "parse_record", [('    if verdict(x) != rec["verdict"]:', "    if False:")],
     "d_condition_void_yes", "a void record accepted (yes at the threshold)"),
    ("M91", "C", "parse_record", [('    if verdict(x) != rec["verdict"]:', "    if False:")],
     "d_condition_void_no", "a void record accepted (no one ulp above it)"),
    ("M92", "C", "parse_record", [(" or sorted(rec) != sorted(RECORD_KEYS)", "")],
     "d_condition_format_keys", "a record's keys not checked"),
    ("M93", "C", "parse_record", [(", parse_constant=_no_constant)", ")")],
     "d_condition_format_nan", "NaN accepted in a record"),
    ("M94", "C", "parse_record", [(' and rec["threshold"]["repr"] == THRESHOLD', "")],
     "d_condition_format_threshold", "a record's threshold not checked"),
    ("M95", "C", "read_record_file", [(SHA_FORM, "    if False:")],
     "d_condition_sha256_format", "print-dl's sha256 form not checked"),
    ("M96", "C", "read_record_file", [("    if Q.names_test(p):", "    if False:")],
     "d_read_record_file_forbidden_path", "print-dl's reader takes a forbidden name"),
    ("M97", "C", "read_record_file", [("    if not p.is_file():", "    if False:")],
     "d_condition_missing_file", "print-dl's missing record not refused by name"),
    ("M98", "C", "read_record_file", [("    if sha != expect_sha256:", "    if False:")],
     "d_condition_sha256_mismatch", "print-dl's sha256 not compared"),
    ("M99", "C", "read_condition", [(SHA_FORM, "    if False:")],
     "d_read_condition_sha256_format", "read_condition's sha256 form not checked"),
    ("M100", "C", "read_condition", [('    if rel.is_absolute() or ".." in rel.parts or not rel.parts:', "    if False:")],
     "d_read_condition_not_repo_relative", "read_condition takes an absolute path"),
    ("M101", "C", "read_condition", [("    if Q.names_test(rel) or Q.names_test(full):", "    if False:")],
     "d_read_condition_forbidden_path", "read_condition takes a forbidden name"),
    ("M102", "C", "read_condition", [("    if not full.is_file():", "    if False:")],
     "d_read_condition_missing_file", "read_condition's missing record not refused by name"),
    ("M103", "C", "read_condition", [("    if head is None:", "    if False:")],
     "d_read_condition_not_tracked", "an untracked record not refused by name"),
    ("M104", "C", "read_condition", [("    if head != data:", "    if False:")],
     "d_read_condition_changed_since_head", "a record changed since HEAD accepted"),
    ("M105", "C", "read_condition", [("    if sha != expect_sha256:", "    if False:")],
     "d_read_condition_sha256_mismatch", "read_condition's sha256 not compared"),
    # ---- the order of the gates (the plan §4.2: the forbidden-name check before any file call)
    ("M106", "C", "record_condition", [("    check_names(args, RECORD_PATH_FLAGS)", "    pass"),
                                       ("    meta, meta_data = run_meta_row(d)\n",
                                        "    meta, meta_data = run_meta_row(d)\n"
                                        "    check_names(args, RECORD_PATH_FLAGS)\n")],
     "r_forbidden_path_run_dir", "G1 run after the run directory's files are read"),
    # ---- the readers, continued (AQ4 in a record)
    ("M107", "C", "parse_record", [("value_error(x) is None and ", "")],
     "d_condition_format_value", "a record's value not checked (a float, finite, in [0, 1])"),
]


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def anchor_counts(owner, name: str, pairs) -> list:
    src = textwrap.dedent(inspect.getsource(getattr(owner, name)))
    return [src.count(old) for old, _ in pairs]


@contextlib.contextmanager
def mutate(owner, name: str, *pairs):
    """Replace `owner.name` (a module function or a class method) by its own source with each (old, new)
    replaced exactly once, compiled into its module's namespace; restore it afterwards.
    (scripts/smoke_teacher_diag_mutations.py :56-81, copied.)"""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated mutation ids")
    a = ap.parse_args(argv)
    ids = [m[0] for m in MUTATIONS]
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else []
    if len(set(ids)) != len(ids) or [x for x in only if x not in ids]:
        print(f"RESULT: ERROR mutation ids repeated or unknown: {[x for x in only if x not in ids]}")
        return 2
    work = S.choose_work_dir(a.work_dir)
    work.mkdir(parents=True)
    (work / "scratch").mkdir()
    (work / "relative_base").mkdir()
    tempfile.tempdir = str(work / "scratch")
    os.chdir(work / "relative_base")             # a relative --out or --run-dir resolves here, never in the repository
    S.AUDIT_BASES[:] = [str(work), str(S.REPO)]
    sys.addaudithook(S.audit)
    t0 = time.time()
    print("=" * 78)
    print("E6-X CONDITION MUTATIONS (L-AM23Q B-1) -- each mutant against its killing case")
    print(f"work dir: {work}")
    print("=" * 78, flush=True)
    S.setup(work)
    chosen = [m for m in MUTATIONS if not only or m[0] in only]
    lines = []
    for mid, owner_key, func, pairs, case_name, what in chosen:
        owner = OWNERS[owner_key]
        counts = anchor_counts(owner, func, pairs)
        check(f"{mid}_anchor", all(n == 1 for n in counts), f"{owner_key}.{func}: occurrences {counts}")
        if not all(n == 1 for n in counts) or case_name not in S.CASES:
            check(f"{mid}_control", False, f"not run: anchor {counts}, case {case_name!r} known "
                                            f"{case_name in S.CASES}")
            check(f"{mid}_killed", False, "not run")
            lines.append(f"  {mid:5} {owner_key}.{func}: NOT RUN ({what})")
            continue
        ok, detail = S.run_case(case_name)
        check(f"{mid}_control", ok, f"{case_name} unmutated: {detail}")
        try:
            with mutate(owner, func, *pairs):
                ok_m, detail_m = S.run_case(case_name)
        except Exception as e:  # noqa: BLE001 -- a mutant that cannot be built is not a kill
            ok_m, detail_m = True, f"mutant not built ({type(e).__name__}: {e})"
        check(f"{mid}_killed", not ok_m, f"{what}: {case_name} under {mid}: "
                                         f"{'PASS (not killed)' if ok_m else 'FAIL (killed)'}: {detail_m}")
        lines.append(f"  {mid:5} {owner_key}.{func}: {what} -- {'KILLED' if not ok_m else 'SURVIVED'} by {case_name}"
                     f"\n          {detail_m[:200]}")
    if not only:
        codes = S.module_codes()
        killed = {S.CASES[m[4]][1] for m in MUTATIONS if m[4] in S.CASES}
        check("every_code_killed", codes <= killed, f"codes without a killing mutation: {sorted(codes - killed)}")
        missing = sorted({m[4] for m in MUTATIONS} - set(S.CASES))
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
    print(f"\nmutations killed: {killed_n}/{len(chosen)} ({round(time.time() - t0, 1)} s)")
    print(f"RESULT: {'PASS' if RESULTS and passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)})")
    return 0 if RESULTS and passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
