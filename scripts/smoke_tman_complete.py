#!/usr/bin/env python3
"""Smoke for "complete at an instant" (AM-19 Definitions; AM-19 item 3(d)) and the freeze resolver (lane TMAN, B-1,
C2): the completion kinds at instant - 1 s, the instant and + 1 s; failed checks, short runs and misplaced end
records refused; divergence split as sweep_select splits it; a QAT run with a non-finite state still complete
(AM-21 item 1(e)); stops and aborts; the M12 "(final)" line read for its timestamp only (TB19 Q4; TB14); the QAT
epoch checkpoints' stored flags against their rows (TB13); the resolver's providers, the moved freeze and TEST
deadline (AM-19 item 1(d)), AM-22's dates from its text (TB12) and rule-aware dating (T-15).

Synthetic run directories under the system temp dir (removed at the end); nothing test-named is created; TEST is
never read. The committed schedule, amendments and scripts/preflight_arms.py are read for the cross-checks.

Run:  python -B scripts/smoke_tman_complete.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check, expect_refused = FX.check, FX.expect_refused
SCHEDULE_T = {"in_force": "T", "file": "configs/kd_schedule.json", "file_sha256": "a" * 64,
              "decision_log_entry": "DL-70"}
# AM-22's item 2(a) phrasing, as a fixture until the amendment is committed (TB12)
AM22_FIXTURE = ("## AM-22 — Early TEST freeze (fixture)\n\n2. Launch cutoff.\n   (a) A run launches only before its "
                "cutoff. Its cutoff date is 2026-10-20, or a later date under (b), and in no case later than "
                "2026-10-29.\n\n## AM-23 — next\n")


def rewrite_rows(path: Path, fn) -> None:
    """Apply fn to each row of a JSONL fixture and write it back."""
    rows = [json.loads(x) for x in path.read_bytes().split(b"\n") if x.strip()]
    FX.write_jsonl(path, [fn(r) for r in rows])


def case_instants():
    from src.tman import schedule as D
    c = D.cutoff_instant("2026-11-03")
    check("instant_is_1600_utc", FX.utc("2026-11-03T16:00:00Z") == c)
    check("meets_is_strict", D.meets(c - 1, c) and not D.meets(c, c) and not D.meets(c + 1, c))
    check("manila_day_of_instant", D.manila_day(c - 1) == "2026-11-03" and D.manila_day(c) == "2026-11-04")


def case_kd():
    from src.tman import completion as C
    from src.tman import schedule as D
    root = FX.scratch_dir()
    cut = D.cutoff_instant("2026-10-20")
    for label, end, want in (("minus_1s", "2026-10-20T15:59:59Z", True), ("at_instant", "2026-10-20T16:00:00Z", False),
                             ("plus_1s", "2026-10-20T16:00:01Z", False)):
        st = C.kd_state(FX.kd_run(root, f"kd_{label}", end=end), "E2")
        check(f"kd_complete_{label}", st.status == C.COMPLETE and st.complete_at(cut) == want
              and st.completion.record == "e2_telemetry.jsonl" and st.completion.line == 5, f"{st}")
    st = C.kd_state(FX.kd_run(root, "kd_frac", end="2026-10-20T15:59:59.9Z"), "E2")
    check("kd_fractional_second_meets", st.complete_at(cut) and st.completion.at_utc == "2026-10-20T15:59:59Z")
    check("kd_completion_entry_block", st.completion.as_entry() == {
        "record": "e2_telemetry.jsonl", "at_utc": "2026-10-20T15:59:59Z", "sha256": st.telemetry_sha256, "line": 5})
    check("kd_telemetry_name_of_arm_stages", (C.kd_telemetry_name("E3-R1"), C.kd_telemetry_name("E2-L"),
                                              C.kd_telemetry_name("A")) == ("e3_telemetry.jsonl", "e2_telemetry.jsonl",
                                                                            "a_telemetry.jsonl"))
    expect_refused("kd_failed_checks_refused", "completion_record_invalid", C.kd_state,
                   FX.kd_run(root, "kd_fail", checks=False), "E2")
    d = FX.kd_run(root, "kd_late_row")
    with open(d / "e2_telemetry.jsonl", "ab") as fh:
        fh.write(b'{"event": "train", "iter": 9, "wall_clock": 1}\n')
    expect_refused("kd_run_end_not_last_row_refused", "completion_record_invalid", C.kd_state, d, "E2")
    d = FX.kd_run(root, "kd_short")
    rewrite_rows(d / "e2_run_meta.jsonl", lambda r: dict(r, max_iters=80000))
    expect_refused("kd_run_end_short_of_max_iters_refused", "completion_record_invalid", C.kd_state, d, "E2")
    d = FX.kd_run(root, "kd_two_ends", abort={"rule": "AM-7(b)", "cause": "student_divergence"})
    with open(d / "e2_telemetry.jsonl", "ab") as fh:
        fh.write(b'{"event": "run_end", "iter": 3, "checks_passed": true, "wall_clock_end": 5}\n')
    expect_refused("kd_run_end_beside_run_abort_refused", "completion_record_invalid", C.kd_state, d, "E2")
    finite = {"input_finite": True, "teacher_finite": True}
    st = C.kd_state(FX.kd_run(root, "kd_div", abort={"rule": "AM-7(b)", "cause": "student_divergence", **finite}),
                    "E2")
    check("kd_divergence_record_is_diverged", st.status == C.DIVERGED and st.rule == "AM-7(b)" and st.iteration == 3
          and st.ended_at == FX.utc("2026-10-15T00:00:00Z"))
    st = C.kd_state(FX.kd_run(root, "kd_div_teacher", abort={"rule": "AM-7(a)", "cause": "student_divergence",
                                                             "input_finite": True, "teacher_finite": False}), "E2")
    check("kd_divergence_with_nonfinite_teacher_is_aborted", st.status == C.ABORTED)
    st = C.kd_state(FX.kd_run(root, "kd_abort", abort={"rule": "step1", "cause": "input_nonfinite", **finite}), "E2")
    check("kd_other_abort_is_aborted", st.status == C.ABORTED and st.cause == "input_nonfinite")
    st = C.kd_state(FX.kd_run(root, "kd_running", end=None), "E2")
    check("kd_no_end_is_running", st.status == C.RUNNING and st.completion is None)
    d = FX.kd_run(root, "kd_noclock")
    rewrite_rows(d / "e2_telemetry.jsonl", lambda r: dict(r, wall_clock_end=None) if r["event"] == "run_end" else r)
    expect_refused("kd_run_end_without_time_refused", "completion_record_invalid", C.kd_state, d, "E2")


def case_e1():
    from src.tman import completion as C
    from src.tman import schedule as D
    root = FX.scratch_dir()
    cut = D.cutoff_instant("2026-09-20")
    st = C.e1_state(FX.e1_run(root, "e1_ok", end="2026-09-20T15:59:59Z"))
    check("e1_final_val_with_result_pass_complete", st.status == C.COMPLETE and st.complete_at(cut)
          and st.completion.record == "e1_telemetry.jsonl")
    st = C.e1_state(FX.e1_run(root, "e1_at", end="2026-09-20T16:00:00Z"))
    check("e1_at_instant_not_complete_at_date", st.status == C.COMPLETE and not st.complete_at(cut))
    expect_refused("e1_result_fail_refused", "completion_record_invalid", C.e1_state,
                   FX.e1_run(root, "e1_fail", result="FAIL"))
    expect_refused("e1_without_log_refused", "completion_record_invalid", C.e1_state,
                   FX.e1_run(root, "e1_nolog", result=None))
    st = C.e1_state(FX.e1_run(root, "e1_running", end=None))
    check("e1_without_final_val_running", st.status == C.RUNNING)
    d = FX.e1_run(root, "e1_nometa")
    rewrite_rows(d / "e1_telemetry.jsonl", lambda r: {k: v for k, v in r.items() if k != "max_iters"})
    expect_refused("e1_without_max_iters_refused", "completion_record_invalid", C.e1_state, d)


def case_qat():
    from src.tman import completion as C
    from src.tman import schedule as D
    root = FX.scratch_dir()
    cut = D.cutoff_instant("2026-10-25")
    st = C.qat_state(FX.qat_run(root, "q_ok", end="2026-10-25T15:59:59Z"))
    check("qat_epoch15_complete", st.status == C.COMPLETE and st.complete_at(cut) and st.completion.line == 46)
    st = C.qat_state(FX.qat_run(root, "q_nonfinite", nonfinite_from=7))
    check("qat_nonfinite_state_still_complete_AM21", st.status == C.COMPLETE)
    st = C.qat_state(FX.qat_run(root, "q_14", epochs=14))
    check("qat_14_epochs_running", st.status == C.RUNNING)
    st = C.qat_state(FX.qat_run(root, "q_stop", epochs=9, stop=True))
    check("qat_run_stop_is_stopped", st.status == C.STOPPED and st.rule == "x" and st.epoch == 9)
    d = FX.qat_run(root, "q_abort", epochs=4)
    with open(d / "qat_telemetry.jsonl", "ab") as fh:
        fh.write(b'{"event": "run_abort", "wall_clock": 1.5e9, "epoch": 4, "step": 40, "cause": "exception"}\n')
    st = C.qat_state(d)
    check("qat_run_abort_is_aborted", st.status == C.ABORTED and st.cause == "exception")
    st = C.qat_state(FX.qat_run(root, "q_torn", torn=True))
    check("qat_torn_last_line_not_complete", st.status == C.RUNNING)
    expect_refused("qat_epoch15_without_run_complete_refused", "completion_record_invalid", C.qat_state,
                   FX.qat_run(root, "q_flag", complete_flag=False))
    d = FX.qat_run(root, "q_dup", epochs=3)
    with open(d / "qat_telemetry.jsonl", "ab") as fh:
        fh.write(b'{"event": "epoch_end", "wall_clock": 1.5e9, "epoch": 3, "step": 30}\n')
    expect_refused("qat_repeated_epoch_end_refused", "completion_record_invalid", C.qat_state, d)
    d = FX.qat_run(root, "q_meta2", epochs=2)
    with open(d / "qat_telemetry.jsonl", "ab") as fh:
        fh.write(b'{"event": "run_meta", "wall_clock": 1.5e9}\n')
    expect_refused("qat_second_run_meta_refused", "completion_record_invalid", C.qat_state, d)


def case_qat_checkpoint_flags():
    import torch
    from src.tman import completion as C
    from src.tman import schema as S
    root = FX.scratch_dir()
    d = FX.qat_run(root, "q_flags", epochs=15)
    ck = d / "epoch_ckpts"
    ck.mkdir()
    stored = {1: {"state_finite": True}, 2: {"state_finite": False}, 3: {"state_finite": 1},
              4: {"state_nonfinite": None}, 5: {}}
    for e, flags in stored.items():
        payload = {"stage": "E6", "epoch": e, "step": e * 10, "seed": 42, "run_id": "q_flags",
                   "state_sha256": f"{e + 100:064x}", "source_checkpoint_sha256": "c" * 64,
                   "model_state_dict": {"w": torch.zeros(2)}, "state_finite": True, "state_nonfinite": False,
                   "nonfinite_since_step": None, "clip_norm": 5.0}
        payload.update(flags)
        if e == 4:
            del payload["state_nonfinite"]
        torch.save(payload, ck / f"e{e:02d}.pt")
    shas = {e: S.sha256_bytes((ck / f"e{e:02d}.pt").read_bytes()) for e in stored}
    shas[5] = "0" * 64                                       # the row records another file's hash
    rewrite_rows(d / "qat_telemetry.jsonl", lambda r: dict(r, checkpoint_sha256=shas[r["epoch"]])
                 if r["event"] == "epoch_end" and r["epoch"] in shas else r)
    rows, _, _ = C.qat_rows(d)
    errs = C.qat_flag_errors(d, rows)
    joined = " | ".join(errs)
    check("qat_checkpoint_flags_agreeing_pass", not any("e01.pt" in e for e in errs), joined)
    check("qat_checkpoint_flag_disagreeing_reported_TB13", any("e02.pt" in e and "state_finite" in e for e in errs),
          joined)
    check("qat_checkpoint_flag_type_differs_reported", any("e03.pt" in e and "state_finite 1" in e for e in errs),
          joined)
    check("qat_checkpoint_flag_missing_reported", any("e04.pt" in e and "missing" in e for e in errs), joined)
    check("qat_checkpoint_sha256_differs_reported", any("e05.pt" in e and "sha256" in e for e in errs), joined)
    check("qat_checkpoint_absent_skipped", len(errs) == 4, joined)
    flags = C.qat_checkpoint_flags(ck / "e01.pt")
    check("qat_checkpoint_projection", sorted(flags) == sorted(
        ["stage", "epoch", "step", "seed", "run_id", "state_sha256", "source_checkpoint_sha256", "state_finite",
         "state_nonfinite", "nonfinite_since_step"]), str(sorted(flags)))
    rewrite_rows(d / "qat_telemetry.jsonl", lambda r: dict(r, checkpoint="epoch_ckpts/x.pt")
                 if r["event"] == "epoch_end" and r["epoch"] == 6 else r)
    rows, _, _ = C.qat_rows(d)
    check("qat_checkpoint_misnamed_row_reported", any("epoch 6" in e for e in C.qat_flag_errors(d, rows)))


def case_teacher():
    from src.tman import completion as C
    root = FX.scratch_dir()
    st = C.teacher_state(FX.teacher_run(root, "teacher-r1_s42_40k_a1"))
    check("m12_console_line_read_as_utc_2026_Q4", st.status == C.COMPLETE
          and st.completion.at_utc == "2026-10-13T05:26:26Z" and st.completion.line == 2
          and st.completion.record == "20261012_010203/20261012_010203.log", str(st.completion))
    st = C.teacher_state(FX.teacher_run(root, "teacher-r2_s42_40k_a1", with_year=True, final="10/16 10:20:30"))
    check("m12_file_line_with_year", st.completion is not None and st.completion.at_utc == "2026-10-16T10:20:30Z")
    st = C.teacher_state(FX.teacher_run(root, "teacher-x_s42_40k_a1", final=None))
    check("m12_absent_is_running", st.status == C.RUNNING)
    two = FX.teacher_run(root, "teacher-z_s42_40k_a1")
    FX.teacher_run(root, "teacher-z_s42_40k_a1", folder="20261014_010203", final="10/15 01:00:00")
    st = C.teacher_state(two)
    check("teacher_complete_at_its_earliest_log_s_final_line_R3_15", st.completion is not None
          and st.completion.record == "20261012_010203/20261012_010203.log"
          and st.completion.at_utc == "2026-10-13T05:26:26Z", str(st.completion))
    later = FX.teacher_run(root, "teacher-w_s42_40k_a1")
    log = later / "20261012_010203" / "20261012_010203.log"
    log.write_text(log.read_text() + "02/30 01:02:03 - mmengine - INFO - Iter(train) [   60/40000]\n")
    try:
        status = C.teacher_state(later).status
    except Exception as e:  # noqa: BLE001 -- a refusal fails the check by name
        status = getattr(e, "code", type(e).__name__)
    check("teacher_complete_with_a_later_malformed_stamp_reads_complete_N2", status == C.COMPLETE, str(status))
    gi = C.M12_FINAL_RE.groupindex
    check("m12_regex_captures_the_timestamp_only_TB14", sorted(gi) == ["day", "hh", "mm", "month", "ss", "year"]
          and C.M12_FINAL_RE.pattern.endswith(r"M12 selection \(final\): "))
    colored = "10/13 05:26:26 - mmengine - \x1b[32mINFO\x1b[0m - M12 selection (final): iteration 20000 x=0.1"
    check("m12_colored_console_level_read", C.m12_final_instant(colored) == (FX.utc("2026-10-13T05:26:26Z"), 1))
    check("m12_partial_line_not_read", C.m12_final_instant("10/13 05:26:26 - mmengine - INFO - M12 selection: x")
          is None)
    expect_refused("m12_impossible_date_refused", "completion_record_invalid", C.m12_final_instant,
                   "02/30 05:26:26 - mmengine - INFO - M12 selection (final): x")
    work = FX.teacher_run(root, "teacher-y_s42_40k_a1")
    calls, saved_iterdir, saved_is_dir = [], Path.iterdir, Path.is_dir
    probe = work / ("te" + "st_folder")                       # never created
    try:
        Path.iterdir = lambda self: iter([probe, work / "20261012_010203"])
        Path.is_dir = lambda self: calls.append(self.name) or saved_is_dir(self)
        st = C.teacher_state(work)
    finally:
        Path.iterdir, Path.is_dir = saved_iterdir, saved_is_dir
    check("teacher_listing_matches_names_before_any_call", st.status == C.COMPLETE
          and probe.name not in calls, str(calls))
    calls.clear()
    other = work / "notes"                                    # never created; no log folder's name
    try:
        Path.iterdir = lambda self: iter([other, work / "20261012_010203"])
        Path.is_dir = lambda self: calls.append(self.name) or saved_is_dir(self)
        last = C.teacher_last_row(work)
    finally:
        Path.iterdir, Path.is_dir = saved_iterdir, saved_is_dir
    check("teacher_last_row_listing_matches_names_before_any_call_B2", last.at == int(FX.utc("2026-10-13T05:26:26Z"))
          and other.name not in calls, f"{last} {calls}")


def case_arm_and_dispatch():
    from src.tman import completion as C
    from src.tman import schema as S
    root = FX.scratch_dir()
    (root / "arm_out.json").write_text("{}\n", encoding="utf-8")
    st = C.arm_state(root, "arm_out.json", "arm_agree.json", "2026-10-28T01:00:00Z")
    check("arm_without_agreement_running", st.status == C.RUNNING)
    (root / "arm_agree.json").write_text("{}\n", encoding="utf-8")
    st = C.arm_state(root, "arm_out.json", "arm_agree.json", "2026-10-28T01:00:00Z")
    check("arm_with_output_and_agreement_complete", st.status == C.COMPLETE
          and st.completion.at_utc == "2026-10-28T01:00:00Z" and st.completion.line is None)
    cat = S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())
    d = FX.kd_run(root, "kd_dispatch", stage_key="e3")
    check("run_state_dispatches_kd_arm_stage", C.run_state(cat.slots["E3-R1:s42:lambda0.5:alpha25:80k"], d).status
          == C.COMPLETE)
    check("run_state_dispatches_qat", C.run_state(cat.slots["E6:s42:15ep"], FX.qat_run(root, "q_dispatch")).status
          == C.COMPLETE)
    check("run_state_dispatches_e1", C.run_state(cat.slots["E1:s42:80k"], FX.e1_run(root, "e1_dispatch")).status
          == C.COMPLETE)
    check("run_state_dispatches_teacher", C.run_state(cat.slots["teacher-R1:s42:40k"],
                                                      FX.teacher_run(root, "teacher-d")).status == C.COMPLETE)
    expect_refused("run_state_of_derived_slot_refused", "completion_record_invalid", C.run_state,
                   cat.slots["E4:s42:list42"], root)
    st = C.kd_state(FX.kd_run(root, "kd_status", end="2026-10-20T10:00:00Z"), "E2")
    t = FX.utc("2026-10-20T10:00:00Z")
    check("status_at_before_and_after_the_ending", C.status_at(st, int(t)) == C.RUNNING
          and C.status_at(st, int(t) + 1) == C.COMPLETE)
    dv = C.kd_state(FX.kd_run(root, "kd_status_div", abort={"rule": "AM-7(a)", "cause": "student_divergence",
                                                            "input_finite": True, "teacher_finite": True}), "E2")
    check("status_at_divergence", C.status_at(dv, dv.ended_at) == C.RUNNING
          and C.status_at(dv, dv.ended_at + 1) == C.DIVERGED)


def _entry(path: str, sha: str, **doc) -> dict:
    return {"doc": doc, "path": path, "sha256": sha}


def case_resolver():
    from src.tman import schedule as D
    fi = D.freeze_instant(SCHEDULE_T)
    check("schedule_T_freeze_and_deadline", (fi.original_date, fi.moved_date, fi.test_deadline)
          == ("2026-11-03", None, "2026-11-08") and fi.source["provider"] == "schedule")
    r = D.freeze_instant(dict(SCHEDULE_T, in_force="R"))
    check("schedule_R_freeze_and_deadline", (r.original_date, r.test_deadline) == ("2026-12-18", "2026-12-21"))
    expect_refused("schedule_unknown_refused", "schedule_invalid", D.freeze_instant, dict(SCHEDULE_T, in_force="S"))
    delay = _entry("reports/derived/tday_entries/freeze_delay__session__20261103__01.json", "d" * 64,
                   date="2026-11-03", original_date="2026-11-03", runs=["E6:s42:15ep"])
    new = _entry("reports/derived/tday_entries/new_freeze__session__20261105__01.json", "e" * 64,
                 date="2026-11-05", new_date="2026-11-05", delay_entry={"path": delay["path"], "sha256": "d" * 64})
    fi = D.freeze_instant(SCHEDULE_T, delays=[delay], news=[new])
    check("new_freeze_moves_freeze_and_deadline_AM19_1d", (fi.original_date, fi.moved_date, fi.test_deadline)
          == ("2026-11-03", "2026-11-05", "2026-11-10") and fi.original_utc < fi.moved_utc
          and fi.source == {"provider": "new_freeze", "path": new["path"], "sha256": "e" * 64})
    fi = D.freeze_instant(SCHEDULE_T, delays=[delay])
    check("delay_without_new_freeze_keeps_original", fi.moved_date is None and fi.source["provider"] == "freeze_delay"
          and fi.moved_utc == fi.original_utc)
    block = D.freeze_instant(SCHEDULE_T, delays=[delay], news=[new]).as_block()
    check("schedule_block_fields_plan_1_2", sorted(block) == sorted(
        ["in_force", "file", "file_sha256", "decision_log_entry", "original_freeze_date", "original_freeze_utc",
         "moved_freeze_date", "freeze_source", "test_deadline"]) and block["original_freeze_utc"]
          == "2026-11-03T16:00:00Z")
    late = dict(delay, doc=dict(delay["doc"], date="2026-11-04"))
    expect_refused("delay_not_on_original_date_refused", "freeze_source_conflict", D.freeze_instant, SCHEDULE_T,
                   delays=[late], news=[new])
    other = dict(delay, doc=dict(delay["doc"], original_date="2026-12-18"))
    expect_refused("delay_for_another_freeze_refused", "freeze_source_conflict", D.freeze_instant, SCHEDULE_T,
                   delays=[other])
    expect_refused("new_freeze_without_delay_refused", "freeze_source_conflict", D.freeze_instant, SCHEDULE_T,
                   news=[new])
    expect_refused("two_delays_refused", "freeze_source_conflict", D.freeze_instant, SCHEDULE_T,
                   delays=[delay, delay], news=[new])
    expect_refused("two_new_freezes_refused", "freeze_source_conflict", D.freeze_instant, SCHEDULE_T,
                   delays=[delay], news=[new, new])
    early = dict(new, doc=dict(new["doc"], new_date="2026-11-03", date="2026-11-03"))
    expect_refused("new_freeze_not_later_refused", "freeze_source_conflict", D.freeze_instant, SCHEDULE_T,
                   delays=[delay], news=[early])
    off_day = dict(new, doc=dict(new["doc"], date="2026-11-06"))
    expect_refused("new_freeze_entered_on_another_day_refused", "freeze_source_conflict", D.freeze_instant,
                   SCHEDULE_T, delays=[delay], news=[off_day])
    uncited = dict(new, doc=dict(new["doc"], delay_entry={"path": delay["path"], "sha256": "f" * 64}))
    expect_refused("new_freeze_not_citing_delay_refused", "freeze_source_conflict", D.freeze_instant, SCHEDULE_T,
                   delays=[delay], news=[uncited])
    front = D.freeze_instant(SCHEDULE_T, delays=[delay], news=[new],
                             providers=[lambda base: D.FreezeInstant(base.schedule, base.original_date, None,
                                                                     {"provider": "am22"}, base.test_deadline)])
    check("provider_in_front_answers_first_B2", front.source == {"provider": "am22"})
    check("first_complete_day_end_of_manila_day", D.first_complete_day(
        [FX.utc("2026-11-04T15:59:59Z"), FX.utc("2026-11-02T01:00:00Z")], "2026-11-03") == "2026-11-04")
    check("first_complete_day_next_day_at_1600", D.first_complete_day(
        [FX.utc("2026-11-04T16:00:00Z")], "2026-11-03") == "2026-11-05")
    check("first_complete_day_none_while_incomplete", D.first_complete_day(
        [FX.utc("2026-11-04T10:00:00Z"), None], "2026-11-03") is None)


def case_committed_cross_checks():
    import scripts.preflight_arms as pa
    from src.tman import schedule as D
    from src.training import sweep_select as ss
    check("freeze_dates_equal_preflight_arms_TB14", D.FREEZE_DATES == pa.FREEZE_DATES)
    text = (REPO / "docs/PREREGISTRATION_AMENDMENTS.md").read_text(encoding="utf-8")
    t_ok = ("TEST manifest freeze: 2026-11-03;" in text and "single TEST evaluation, no later than: 2026-11-08." in text
            and "TEST manifest freeze: 2026-12-18;" in text and "single TEST evaluation, no later than: 2026-12-21."
            in text)
    check("freeze_and_test_dates_equal_am19_1a_text", t_ok)
    blk = D.schedule_block(ss.HeadSource(REPO))
    check("committed_schedule_block", blk["in_force"] in D.FREEZE_DATES and blk["file"] == "configs/kd_schedule.json"
          and len(blk["file_sha256"]) == 64 and blk["decision_log_entry"].startswith("DL-"), str(blk))
    repo = FX.new_repo()
    FX.commit_files(repo, {"configs/kd_schedule.json": {"format": "kd_schedule/1", "in_force": "T",
                                                        "decision_log_entry": "DL-9"},
                           "docs/DECISION_LOG.md": FX.dl_table({"DL-70": ("RECORDED", "x")})})
    expect_refused("schedule_naming_no_dl_row_refused", "schedule_invalid", D.schedule_block, ss.HeadSource(repo))
    sec = D.am22_text(text)
    check("am22_committed_text_absent_or_parsable", sec is None or bool(D.am22_dates(sec, None).cutoff))


def case_am22_dates():
    from src.tman import schedule as D
    sec = D.am22_text(AM22_FIXTURE)
    check("am22_section_found_in_text", sec is not None and "AM-23" not in sec and sec.startswith("## AM-22 — "))
    d0 = D.am22_dates(sec, "2026-10-17")
    check("am22_dates_unmoved_on_1017_TB12", (d0.cutoff, d0.ceiling, d0.shift_days, d0.test_deadline)
          == ("2026-10-20", "2026-10-29", 0, "2026-11-08"))
    check("am22_dates_unmoved_before_1017", D.am22_dates(sec, "2026-10-15").shift_days == 0
          and D.am22_dates(sec, None).shift_days == 0)
    d2 = D.am22_dates(sec, "2026-10-19")
    check("am22_dates_move_by_push_days_TB12", (d2.cutoff, d2.ceiling, d2.shift_days, d2.base_cutoff)
          == ("2026-10-22", "2026-10-31", 2, "2026-10-20"))
    expect_refused("am22_text_without_dates_refused", "schedule_invalid", D.am22_dates, "## AM-22 — x\n", None)
    check("cutoff_cut_entered_day_after_T15", D.expected_entered_day("cutoff", "2026-10-20") == "2026-10-21")
    check("other_cuts_entered_on_their_day", all(D.expected_entered_day(r, "2026-11-03") == "2026-11-03"
                                                 for r in ("freeze", "decision_date", "named_rule", "arithmetic")))
    check("parse_utc_any_forms", D.parse_utc_any("2026-11-03T16:00:00Z") == D.cutoff_instant("2026-11-03")
          and D.parse_utc_any("2026-11-03T16:00:00.500000Z") == D.cutoff_instant("2026-11-03") + 0.5)


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN COMPLETION SMOKE (lane TMAN, B-1, C2)",
                                  (case_instants, case_kd, case_e1, case_qat, case_qat_checkpoint_flags, case_teacher,
                                   case_arm_and_dispatch, case_resolver, case_committed_cross_checks,
                                   case_am22_dates)))
