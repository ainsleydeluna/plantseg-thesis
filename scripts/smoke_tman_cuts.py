#!/usr/bin/env python3
"""Smoke for the cut records (AM-19 Code: "the cut records"; AM-19 items 1(d), 3(b), 3(c); AM-18 item 7; AM-23 item
3(b)) and the entry printers (lane TMAN, B-1, C3).

Each cut code is reached with every earlier check passing: a case starts from a cut that passes every check,
changes what the code is about, and requires that code and the list of checks passed before it. Also: the four
rules; reason and since (decision (xiv)); AM-19 item 3(b)'s arithmetic through scripts/preflight_arms.py's
cut_arithmetic (DL-117's 60,099 s vs 1,914,676 s and DL-119's 60,660 s vs 1,835,047 s both fit; equality fits; no
measurement, no cut); the reference figures; dates by rule; cut-order positions only for rule freeze (TB10); arm
results and partial arm cuts; decision-date cuts against the decision record; the freeze delay and new freeze; the
printer's exclusive output, its decision-log row, the committed file read back (written once, in a RECORDED row) and
the late-entry REPORT (decision (xv); T-15).

Synthetic records in scratch git repositories under the system temp dir (removed at the end); TEST is never read.

Run:  python -B scripts/smoke_tman_cuts.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check, expect_refused = FX.check, FX.expect_refused
REF = {"path": "reports/derived/tday_entries/x.json", "sha256": "f" * 64}
SCHEDULE_T = {"in_force": "T", "file": "configs/kd_schedule.json", "file_sha256": "a" * 64,
              "decision_log_entry": "DL-70"}
E3S42 = "E3:s42:lambda0.5:alpha25:80k"
ARM_R1 = ["teacher-R1:s42:40k", "E3-R1:s42:lambda0.5:alpha25:80k"]
E3X = {"run_id": "e3-r1_s42_80k_a1", "checkpoint_sha256": "6" * 64, "best_json_sha256": "7" * 64}
RULE_RECORDS = {   # named rules' records by sha256: AM-23 item 3(b) condition entries, AM-18 go-rule records
    "c" * 64: {"kind": "condition", "arm": "R1", "verdict": "yes", "date": "2026-10-29", "e3x": E3X},
    "d" * 64: {"kind": "condition", "arm": "R1", "verdict": "no", "date": "2026-10-29", "e3x": E3X},
    "e" * 64: {"kind": "condition", "arm": "R2", "verdict": "no", "date": "2026-10-29", "e3x": E3X},
    "f" * 64: {"kind": "go_rule", "arm": "R1", "verdict": "no_go", "date": "2026-10-25"},
    "9" * 64: {"kind": "go_rule", "arm": "R1", "verdict": "go", "date": "2026-10-25"},
    "b" * 64: {"kind": "unknown", "date": None}}


def catalogue():
    from src.tman import schema as S
    return S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())


def ctx(runs: dict | None = None, unread: tuple = (), **kw):
    """A cut context whose records give each slot's launched attempts: `runs` {slot: [run_id, ...]}, every other
    slot never launched, the slots in `unread` not read (a non-KD slot without --runs)."""
    from src.tman import entries as E
    from src.tman import schedule as D
    from src.training import kd_arms
    runs = dict(runs or {})
    owner = {r: k for k, rs in runs.items() for r in rs}
    reg = {spec.run_stage: spec for spec in kd_arms.load_arms().values()}
    base = dict(registry=reg, decision_records={}, stopped=set(), status=lambda s: None,
                attempts=lambda k: None if k in unread else list(runs.get(k, [])), run_slot=owner.get,
                rule_record=lambda r: {"kind": None, "date": None} if r is None else RULE_RECORDS.get(r["sha256"]))
    base.update(kw)
    return E.CutContext(catalogue(), D.freeze_instant(SCHEDULE_T), **base)


def freeze_cut(slots=("G:s42:80k",), pos=10, **kw) -> dict:
    """A never-launched cut at the original freeze date that passes every check."""
    doc = {"format": "tday_entry/1", "kind": "cut", "slots": list(slots), "date": "2026-11-03",
           "amendment": "AM-19 item 3(b)", "printed_by": {"script": "scripts/tday_manifest.py", "commit": "0" * 40},
           "corrects": None, "push_evidence": None, "rule": "freeze", "reason": "gate not met", "launched": False,
           "run_ids": [], "since": {"date": "2026-10-01", "record": dict(REF)}, "arithmetic": None, "telemetry": [],
           "comparator": None, "results_known": [], "cut_order_position": pos, "rule_record": None}
    doc.update(kw)
    return doc


def launched_fields(run_id: str, comparator: str = E3S42) -> dict:
    return {"launched": True, "run_ids": [run_id], "reason": "not complete at the freeze", "since": None,
            "telemetry": [{"run_id": run_id, "path": f"{run_id}/e3_telemetry.jsonl", "sha256": "1" * 64}],
            "comparator": {"slot": comparator, "run_id": "e3_s42", "path": "e3_s42/e3_telemetry.jsonl",
                           "sha256": "2" * 64}}


def run_checks(doc, c):
    from src.tman import entries as E
    passed = []
    try:
        E.check_cut(doc, c, passed)
        return None, passed
    except Exception as e:  # noqa: BLE001 -- a case reports the refusal code
        return getattr(e, "code", type(e).__name__), passed


def expect_gate(name: str, doc: dict, c, want: str | None) -> None:
    from src.tman import entries as E
    order = [k for k, _ in E.CUT_CHECKS]
    code, passed = run_checks(doc, c)
    want_passed = order if want is None else order[:order.index(want)] if want in order else []
    check(name, code == want and passed == want_passed, f"code {code}, {len(passed)} passed (want {want}, "
                                                        f"{len(want_passed)})")


def arithmetic(iterations: int, median: float, now: str, cut_expected: bool | None = None) -> dict:
    from src.tman import entries as E
    fig = [{"source": "telemetry", "path": "e3_s42/e3_telemetry.jsonl", "sha256": "3" * 64,
            "field": "every train row's iter_seconds", "median": median, "detail": {"rows": 10}}]
    a = E.cut_arithmetic(iterations, fig, "T", FX.utc(now))
    assert cut_expected is None or a["cut"] == cut_expected, a
    return a


# ======================================================================================================= cases
def case_gate_order():
    """One case per cut code, each reached with every earlier check passing."""
    c = ctx()
    expect_gate("base_freeze_cut_passes_every_check", freeze_cut(), c, None)
    expect_gate("rule_operator_refused", freeze_cut(rule="operator"), c, "cut_rule_unknown")
    expect_gate("never_cut_slot_refused", freeze_cut(slots=["E1:s42:80k"]), c, "cut_of_never_cut")
    expect_gate("complete_run_cut_refused", freeze_cut(), ctx(status=lambda s: "complete"), "cut_of_complete_run")
    expect_gate("never_launched_reason_outside_list", freeze_cut(reason="cards busy"), c, "cut_reason_invalid")
    expect_gate("never_launched_without_since", freeze_cut(since=None), c, "cut_reason_invalid")
    expect_gate("since_after_cut_date", freeze_cut(since={"date": "2026-11-04", "record": dict(REF)}), c,
                "cut_reason_invalid")
    expect_gate("launched_flag_against_launch_log", freeze_cut(run_ids=["g_s42_80k_a1"]),
                ctx(runs={"G:s42:80k": ["g_s42_80k_a1"]}), "cut_reason_invalid")
    expect_gate("launched_attempt_left_out_of_run_ids_R099", freeze_cut(), ctx(runs={"G:s42:80k": ["g_s42_80k_a1"]}),
                "cut_reason_invalid")
    expect_gate("run_id_of_another_slot", freeze_cut(**launched_fields("f_s42_alpha25_80k_a1")),
                ctx(runs={"F:s42:alpha25:80k": ["f_s42_alpha25_80k_a1"]}), "cut_reason_invalid")
    expect_gate("run_id_no_record_launched", freeze_cut(run_ids=["g_s42_80k_a9"]), c, "cut_reason_invalid")
    qat = freeze_cut(slots=["E5:s43:15ep"], pos=19, **launched_fields("e5_s43_15ep_a1"))
    expect_gate("launched_qat_cut_passes_AM19_definitions", qat, ctx(runs={"E5:s43:15ep": ["e5_s43_15ep_a1"]}), None)
    expect_gate("qat_cut_unread_without_runs_passes_as_named", qat, ctx(unread=("E5:s43:15ep",)), None)
    expect_gate("qat_cut_unread_naming_runs_never_launched", dict(qat, launched=False, reason="gate not met",
                                                                 since={"date": "2026-10-01", "record": dict(REF)}),
                ctx(unread=("E5:s43:15ep",)), "cut_reason_invalid")
    expect_gate("freeze_cut_dated_off_its_day", freeze_cut(date="2026-11-02"), c, "cut_date_mismatch")
    a = arithmetic(80000, 30.0, "2026-10-20T03:00:00Z", True)
    arith = freeze_cut(rule="arithmetic", reason="does not fit", since=None, date="2026-10-20", arithmetic=a,
                       cut_order_position=None)
    expect_gate("arithmetic_cut_passes", arith, c, None)
    expect_gate("arithmetic_recomputation_disagrees", dict(arith, arithmetic=dict(a, seconds_needed=1.0)), c,
                "cut_arithmetic_invalid")
    fits = arithmetic(10, 1.0, "2026-10-20T03:00:00Z", False)
    expect_gate("arithmetic_rule_that_fits_refused", dict(arith, arithmetic=fits), c, "cut_arithmetic_invalid")
    expect_gate("launched_cut_without_telemetry", freeze_cut(**dict(launched_fields("g_s42_80k_a1"), telemetry=[])),
                ctx(runs={"G:s42:80k": ["g_s42_80k_a1"]}), "cut_val_record_missing")
    expect_gate("launched_cut_without_comparator",
                freeze_cut(**dict(launched_fields("g_s42_80k_a1"), comparator=None)),
                ctx(runs={"G:s42:80k": ["g_s42_80k_a1"]}), "cut_val_record_missing")
    expect_gate("launched_freeze_cut_passes", freeze_cut(**launched_fields("g_s42_80k_a1")),
                ctx(runs={"G:s42:80k": ["g_s42_80k_a1"]}), None)
    arm = freeze_cut(slots=ARM_R1, pos=7, results_known=[{"dl": "DL-118", "path": "reports/x.json",
                                                          "sha256": "4" * 64}])
    expect_gate("arm_freeze_cut_passes", arm, c, None)
    expect_gate("arm_cut_without_results", dict(arm, results_known=[]), c, "arm_cut_results_missing")
    rr = {"path": "reports/derived/arm_go_rule_r1.json", "sha256": "5" * 64, "date": "2026-10-25"}
    running = dict(arm, **launched_fields("e3-r1_s42_80k_a1"), rule="named_rule", rule_record=rr, date="2026-10-25",
                   cut_order_position=None)
    running["run_ids"] = ["teacher-r1_s42_40k_a1", "e3-r1_s42_80k_a1"]
    running["telemetry"] = running["telemetry"] + [{"run_id": "teacher-r1_s42_40k_a1", "path": "t", "sha256": "6" * 64}]
    expect_gate("launched_arm_cut_by_named_rule_refused", running,
                ctx(runs={"teacher-R1:s42:40k": ["teacher-r1_s42_40k_a1"], ARM_R1[1]: ["e3-r1_s42_80k_a1"]}),
                "arm_run_cut_while_running")
    known = {"dl": "DL-99", "path": "reports/derived/mde.json", "sha256": "8" * 64}
    teacher, go = "reports/arms/arm_teacher_record_R1.json", "reports/arms/arm_go_rule_R1.json"
    expect_gate("results_committed_after_the_cut_s_day_B69", dict(arm, results_known=[known]),
                ctx(known_by=lambda rel, sha, day: False), "cut_results_unknown")
    expect_gate("results_committed_by_the_cut_s_day_pass", dict(arm, results_known=[known]),
                ctx(known_by=lambda rel, sha, day: rel == known["path"]), None)
    on_record = ctx(known_by=lambda rel, sha, day: rel in (known["path"], teacher, go))
    expect_gate("arm_cut_without_its_teacher_and_go_rule_records_B69", dict(arm, results_known=[known]), on_record,
                "arm_cut_results_missing")
    expect_gate("arm_cut_listing_its_teacher_and_go_rule_records_passes", dict(arm, results_known=[
        known, {"dl": None, "path": teacher, "sha256": "9" * 64}, {"dl": None, "path": go, "sha256": "a" * 64}]),
                on_record, None)
    expect_gate("cut_entered_before_its_rule_s_day_B75", freeze_cut(), ctx(entered=lambda d: "2026-11-02"),
                "cut_date_mismatch")
    expect_gate("cut_entered_on_its_day_passes", freeze_cut(), ctx(entered=lambda d: "2026-11-03"), None)
    expect_gate("cut_entered_after_its_day_passes_its_REPORT_xv", freeze_cut(), ctx(entered=lambda d: "2026-11-05"),
                None)
    flip = "E3-flip:s42:lambda0.5:alpha25:80k"
    two = freeze_cut(slots=[flip, "G:s42:80k"], pos=None, rule="named_rule", since=None, date="2026-10-28",
                     rule_record={"path": "reports/x.json", "sha256": "7" * 64, "date": "2026-10-28"},
                     results_known=[known])
    expect_gate("two_group_named_rule_cut_passes", two, c, None)
    expect_gate("two_group_cut_without_arm_results_B15", dict(two, results_known=[]), c, "arm_cut_results_missing")
    expect_gate("two_group_named_rule_cut_of_a_running_arm_run_B15", dict(two, **launched_fields("e3-flip_s42_80k_a1")),
                ctx(runs={flip: ["e3-flip_s42_80k_a1"]}), "arm_run_cut_while_running")
    expect_gate("two_group_cut_of_teacher_R1_and_G_partial_B15", dict(two, slots=["teacher-R1:s42:40k", "G:s42:80k"]),
                c, "arm_cut_partial")
    expect_gate("arm_cut_missing_its_later_run", dict(arm, slots=ARM_R1[:1]), c, "arm_cut_partial")
    expect_gate("arm_cut_skipping_an_incomplete_teacher", dict(arm, slots=ARM_R1[1:]),
                ctx(status=lambda s: "running" if s.startswith("teacher") else None), "arm_cut_partial")
    expect_gate("arm_cut_after_a_complete_teacher_passes", dict(arm, slots=ARM_R1[1:]),
                ctx(status=lambda s: "complete" if s.startswith("teacher") else None), None)
    expect_gate("freeze_cut_wrong_position", freeze_cut(pos=11), c, "cut_order_position_mismatch")
    expect_gate("freeze_cut_without_position", freeze_cut(pos=None), c, "cut_order_position_mismatch")
    expect_gate("freeze_cut_over_two_groups", freeze_cut(slots=["G:s42:80k", "F:s42:alpha25:80k"]), c,
                "cut_order_position_mismatch")
    expect_gate("arithmetic_cut_with_a_wrong_position", dict(arith, cut_order_position=3), c,
                "cut_order_position_mismatch")
    expect_gate("arithmetic_cut_with_its_position_passes_TB10", dict(arith, cut_order_position=10), c, None)
    a100 = "E3:s42:lambda0.5:alpha100:80k"
    rec = {"decision_date": "2026-10-22", "values": [{"value": 25.0, "status": "finished"},
                                                     {"value": 50.0, "status": "finished"},
                                                     {"value": 100.0, "status": "finished"}]}
    dd = freeze_cut(slots=[a100], rule="decision_date", reason="cut at its decision date", since=None,
                    date="2026-10-22", cut_order_position=None)
    expect_gate("decision_date_cut_not_in_the_record", dd, ctx(decision_records={"alpha_cwd": rec}),
                "decision_date_cut_mismatch")
    cut_rec = dict(rec, values=[dict(v, status="cut") if v["value"] == 100.0 else v for v in rec["values"]])
    expect_gate("decision_date_cut_in_the_record_passes", dd, ctx(decision_records={"alpha_cwd": cut_rec}), None)
    expect_gate("decision_date_cut_of_a_complete_run_exempt_TB10", dd,
                ctx(decision_records={"alpha_cwd": cut_rec}, status=lambda s: "complete"), None)
    expect_gate("decision_date_cut_without_record", dd, c, "cut_date_mismatch")
    named = freeze_cut(**launched_fields("g_s42_80k_a2"), rule="named_rule",
                       rule_record={"path": "reports/x.json", "sha256": "7" * 64, "date": "2026-10-28"},
                       date="2026-10-28", cut_order_position=None)
    expect_gate("stopped_run_cut_by_named_rule_refused", named,
                ctx(runs={"G:s42:80k": ["g_s42_80k_a2"]}, stopped={"g_s42_80k_a2"}), "stopped_run_cut_early")
    expect_gate("stopped_run_cut_at_the_freeze_passes", freeze_cut(**launched_fields("g_s42_80k_a2")),
                ctx(runs={"G:s42:80k": ["g_s42_80k_a2"]}, stopped={"g_s42_80k_a2"}), None)
    stopped_dd = {**dd, **launched_fields("e3_s42_lambda0.5_alpha100_80k_a1")}
    expect_gate("stopped_candidate_cut_at_its_decision_date_AM19_2b", stopped_dd,
                ctx(runs={a100: ["e3_s42_lambda0.5_alpha100_80k_a1"]}, stopped={"e3_s42_lambda0.5_alpha100_80k_a1"},
                    decision_records={"alpha_cwd": cut_rec}), None)
    reg = {k: v for k, v in ctx().registry.items() if k != "E3-R1"}
    expect_gate("arm_without_registry_file_refused", arm, ctx(registry=reg), "unregistered_arm")
    cond = freeze_cut(slots=["E6-R1:s42:15ep"], pos=None, rule="named_rule", reason="condition not met",
                      since=None, date="2026-10-29",
                      rule_record={"path": "reports/derived/tday_entries/c.json", "sha256": "c" * 64,
                                   "date": "2026-10-29"})
    expect_gate("e6x_cut_citing_a_yes_condition_refused", cond, c, "condition_unmet")
    expect_gate("e6x_cut_citing_a_no_condition_passes", dict(cond, rule_record=dict(cond["rule_record"],
                                                                                     sha256="d" * 64)), c, None)
    no = dict(cond, rule_record=dict(cond["rule_record"], sha256="d" * 64))
    expect_gate("e6x_cut_citing_another_arm_s_no_condition_B16", dict(no, rule_record=dict(no["rule_record"],
                                                                                         sha256="e" * 64)), c,
                "condition_unmet")
    expect_gate("e6x_cuts_of_both_arms_under_one_condition_B16", dict(no, slots=["E6-R1:s42:15ep", "E6-R2:s42:15ep"]),
                c, "condition_unmet")
    expect_gate("e6x_condition_with_its_e3x_run_of_record_passes", no,
                ctx(e3x=lambda arm: {"run_id": E3X["run_id"], "checkpoint_sha256": E3X["checkpoint_sha256"]}), None)
    expect_gate("e6x_condition_naming_another_e3x_run_B16", no,
                ctx(e3x=lambda arm: {"run_id": "e3-r1_s42_80k_a2", "checkpoint_sha256": "6" * 64}), "condition_unmet")
    expect_gate("e6x_condition_without_an_e3x_run_of_record_B16", no, ctx(e3x=lambda arm: False), "condition_unmet")
    go = freeze_cut(slots=[ARM_R1[1]], pos=None, rule="named_rule", since=None, date="2026-10-25",
                    rule_record={"path": "reports/arms/arm_go_rule_R1.json", "sha256": "f" * 64, "date": "2026-10-25"},
                    results_known=[known])
    expect_gate("go_rule_no_go_cuts_its_arm_s_run_passes", go, c, None)
    expect_gate("go_rule_reading_go_refused_B16", dict(go, rule_record=dict(go["rule_record"], sha256="9" * 64)), c,
                "condition_unmet")
    expect_gate("go_rule_of_another_arm_refused_B16", dict(go, slots=["E3-R2:s42:lambda0.5:alpha25:80k"]), c,
                "condition_unmet")
    expect_gate("e6x_cut_by_a_go_rule_refused_B16", dict(go, slots=["E6-R1:s42:15ep"], results_known=[]), c,
                "condition_unmet")
    expect_gate("named_rule_citing_an_unknown_record_B16", dict(go, rule_record=dict(go["rule_record"],
                                                                                   sha256="b" * 64)), c,
                "cut_rule_unknown")
    expect_gate("named_rule_without_its_record_B16", dict(go, rule_record=None), c, "cut_rule_unknown")
    expect_gate("named_rule_record_dated_otherwise_B16", dict(go, rule_record=dict(go["rule_record"],
                                                                                 date="2026-10-24")), c,
                "cut_date_mismatch")
    expect_gate("named_rule_cut_dated_by_its_entry_not_its_record_B16",
                dict(go, date="2026-10-24", rule_record=dict(go["rule_record"], date="2026-10-24")), c,
                "cut_date_mismatch")


def case_input_gates():
    from src.tman import entries as E
    c = ctx()
    expect_refused("cut_naming_unknown_slot", "slot_unknown", E.check_cut, freeze_cut(slots=["G:s42:90k"]), c)
    expect_refused("cut_naming_derived_slot", "derived_counted_as_run", E.check_cut,
                   freeze_cut(slots=["E7-R1:s42:list42"]), c)
    expect_refused("cut_with_a_missing_field", "entry_field", E.check_cut,
                   {k: v for k, v in freeze_cut().items() if k != "since"}, c)
    expect_refused("cut_with_an_extra_field", "entry_field", E.check_cut, dict(freeze_cut(), value=0.3), c)
    expect_refused("entry_of_unknown_kind", "entry_kind_unknown", E.check_cut, dict(freeze_cut(), kind="launch"), c)
    common = {k: v for k, v in freeze_cut().items() if k in E.ENTRY_FILE_COMMON}
    delay = {**common, "kind": "freeze_delay", "original_date": "2026-11-03", "runs": []}
    check("session_entry_without_slot_well_formed", not E.entry_field_errors(dict(delay, slots=[])))
    expect_refused("session_entry_naming_a_slot", "entry_field", E.require_fields, delay)
    expect_refused("slot_entry_naming_no_slot", "entry_field", E.require_fields, dict(freeze_cut(), slots=[]))


def case_arithmetic():
    import scripts.preflight_arms as pa
    from src.tman import entries as E
    from src.training import sweep_select as ss
    for dl, med, now, needed, left in (("DL-117", 1.5024688363075256, "2026-10-12T12:08:44Z", 60099, 1914676),
                                       ("DL-119", 1.5164975285530091, "2026-10-13T10:15:53Z", 60660, 1835047)):
        a = arithmetic(40000, med, now)
        b = pa.cut_arithmetic(ss, 40000, a["references"], "T", FX.utc(now))
        check(f"arithmetic_{dl}_fits", not a["cut"] and round(a["seconds_needed"]) == needed
              and round(a["seconds_left"]) == left, f"{a['seconds_needed']} {a['seconds_left']}")
        check(f"arithmetic_{dl}_is_preflight_arms", a == b)
    now = "2026-11-03T15:00:00Z"                               # 3,600 s before the end of the freeze date
    eq = arithmetic(3600, 1.0, now)
    check("arithmetic_equality_fits", eq["seconds_needed"] == eq["seconds_left"] == 3600.0 and not eq["cut"])
    check("arithmetic_one_iteration_over_cuts", arithmetic(3601, 1.0, now)["cut"])
    none = E.cut_arithmetic(80000, [], "T", FX.utc(now))
    check("arithmetic_no_measurement_no_cut", not none["measured"] and not none["cut"])
    two = E.cut_arithmetic(10, eq["references"] + [dict(eq["references"][0], median=2.0)], "T", FX.utc(now))
    check("arithmetic_applies_the_larger_reference", two["applied_s_per_iter"] == 2.0)
    check("arithmetic_recomputation_agrees", E.arithmetic_error(eq, "T") is None)
    for k, v in (("seconds_needed", 1.0), ("cut", True), ("freeze_date", "2026-12-18"), ("text", "fits"),
                 ("seconds_left", 3601.0)):
        check(f"arithmetic_recomputation_catches_{k}", E.arithmetic_error(dict(eq, **{k: v}), "T") is not None)
    check("arithmetic_unreadable_time", E.arithmetic_error(dict(eq, now_utc="later"), "T") is not None)


def case_references():
    """A reference is read as the cut slot's runs are read (its record kind and stage; DL-24 #111), and a smoke is
    kept under the run directories, its path recorded there (#11)."""
    from src.tman import entries as E
    cat = catalogue()
    root = FX.scratch_dir()
    e3 = cat.slots[E3S42]
    FX.kd_run(root, "e3_s42", stage_key="e3")
    fig = E.reference_telemetry(root, "e3_s42", e3)
    check("reference_telemetry_median_of_train_rows", fig["median"] == 0.7 + 0.02
          and fig["detail"] == {"rows": 3, "end_utc": "2026-10-20T10:00:00Z"}
          and fig["path"] == "e3_s42/e3_telemetry.jsonl" and len(fig["sha256"]) == 64, str(fig))
    FX.kd_run(root, "e3_s43", stage_key="e3", end=None)
    expect_refused("reference_unfinished_run_refused", "cut_arithmetic_invalid", E.reference_telemetry, root,
                   "e3_s43", e3)
    expect_refused("reference_of_another_stage_unread_B111", "completion_record_invalid", E.reference_telemetry, root,
                   "e3_s42", cat.slots["E2:s42:lambda0.5:80k"])
    FX.qat_run(root, "e6_s42_15ep")
    q = E.reference_telemetry(root, "e6_s42_15ep", cat.slots["E6:s42:15ep"])
    check("reference_qat_telemetry_and_its_steps", q["median"] == 0.3 and q["detail"]["steps"] == 150, str(q))
    expect_refused("reference_teacher_gives_no_figure", "cut_arithmetic_invalid", E.reference_telemetry, root,
                   "teacher_s42", cat.slots["teacher:s42:40k"])
    smoke = root / "kd_smoke.json"
    smoke.write_text(json.dumps({"format": "kd_smoke/1", "throughput": {"e3": {"iter_seconds": {"median": 1.5}},
                                                                       "loss": 2.0}}), encoding="utf-8")
    fig = E.reference_smoke(root, "kd_smoke.json", "e3")
    check("reference_smoke_figure_and_its_path_under_the_runs", fig["median"] == 1.5 and fig["path"] == "kd_smoke.json"
          and fig["detail"] == {"format": "kd_smoke/1", "key": "e3"})
    expect_refused("reference_smoke_without_the_key_refused", "cut_arithmetic_invalid", E.reference_smoke, root,
                   "kd_smoke.json", "a")
    (root / "list_smoke.json").write_text("[1, 2]\n", encoding="utf-8")
    try:
        got = FX.refused_code(E.reference_smoke, root, "list_smoke.json", "e3")
    except Exception as e:  # noqa: BLE001 -- a crash where a refusal is due fails the check
        got = type(e).__name__
    check("reference_smoke_holding_a_list_B68", got == "cut_arithmetic_invalid", str(got))
    check("iterations_are_the_slot_s_horizon_B11", [E.slot_iterations(cat.slots[k]) for k in (
        "G:s42:80k", "E1:s42:160k", "teacher:s42:40k", "E6:s42:15ep")] == [80000, 160000, 40000, None]
          and E.slot_iterations(cat.slots["E6:s42:15ep"], 150) == 150)


def case_freeze_delay_and_new_freeze():
    from src.tman import entries as E
    from src.tman import schedule as D
    c = ctx()
    late = ["E6:s42:15ep"]                    # validate.late_never_cut's reading, the one the printer shares
    delay = {"format": "tday_entry/1", "kind": "freeze_delay", "slots": [], "date": "2026-11-03",
             "amendment": "AM-19 item 1(d)", "printed_by": {"script": "scripts/tday_manifest.py", "commit": None},
             "corrects": None, "push_evidence": None, "original_date": "2026-11-03", "runs": list(late)}
    check("freeze_delay_fields", not E.entry_field_errors(delay))
    check("freeze_delay_lists_the_late_never_cut_runs", not E.freeze_delay_errors(delay, c.freeze, late))
    check("freeze_delay_missing_a_late_run", bool(E.freeze_delay_errors(dict(delay, runs=[]), c.freeze, late)))
    check("freeze_delay_off_the_original_date", bool(E.freeze_delay_errors(dict(delay, date="2026-11-04"), c.freeze,
                                                                            late)))
    check("freeze_delay_without_late_runs_refused", bool(E.freeze_delay_errors(delay, c.freeze, [])))
    ref = {"path": "reports/derived/tday_entries/freeze_delay__session__20261103__01.json", "sha256": "8" * 64}
    new = dict(delay, kind="new_freeze", date="2026-11-05", new_date="2026-11-05", delay_entry=dict(ref))
    del new["original_date"], new["runs"]
    inst = [FX.utc("2026-11-05T09:00:00Z")]
    check("new_freeze_fields", not E.entry_field_errors(new))
    check("new_freeze_first_complete_day", not E.new_freeze_errors(new, ref, inst, "2026-11-03"))
    check("new_freeze_not_the_first_complete_day", bool(E.new_freeze_errors(dict(new, new_date="2026-11-06",
                                                                                  date="2026-11-06"), ref, inst,
                                                                             "2026-11-03")))
    check("new_freeze_while_a_run_is_incomplete", bool(E.new_freeze_errors(new, ref, inst + [None], "2026-11-03")))
    check("new_freeze_not_citing_the_delay", bool(E.new_freeze_errors(new, dict(ref, sha256="9" * 64), inst,
                                                                      "2026-11-03")))
    fi = D.freeze_instant(SCHEDULE_T, delays=[{"doc": delay, "path": ref["path"], "sha256": ref["sha256"]}],
                          news=[{"doc": new, "path": "n.json", "sha256": "e" * 64}])
    check("printed_delay_and_new_freeze_move_the_freeze", (fi.moved_date, fi.test_deadline)
          == ("2026-11-05", "2026-11-10"))


def case_printers_and_committed_files():
    from src.tman import entries as E
    from src.tman import records as R
    from src.tman import schema as S
    out = FX.scratch_dir("tmanout")
    doc = E.make_entry("cut", ["G:s42:80k"], "2026-11-03", {k: v for k, v in freeze_cut().items()
                                                            if k in E.ENTRY_FILE_FIELDS["cut"]},
                       amendment="AM-19 item 3(b)", script="scripts/tday_manifest.py", commit="0" * 40)
    E.check_cut(doc, ctx())
    path, rel, sha = E.write_entry(doc, out)
    check("printer_names_the_file_IF2", path.name == "cut__G_s42_80k__20261103__01.json"
          and rel == "reports/derived/tday_entries/cut__G_s42_80k__20261103__01.json"
          and sha == S.sha256_bytes(path.read_bytes()) and S.created_name_error(rel) is None)
    check("printer_file_is_record_bytes", path.read_bytes() == S.json_file_bytes(doc))
    expect_refused("printer_writes_once_B110", "output_exists", E.write_entry, doc, out)
    path2, _, _ = E.write_entry(doc, out, nn=2)
    check("printer_next_sequence", path2.name.endswith("__02.json"))
    drive = out / "drive:x"                     # a folder named like a Windows drive: the operator's (DL-24 #72)
    drive.mkdir()
    try:
        path3, _, _ = E.write_entry(doc, drive)
        got = path3.parent == drive and path3.is_file()
    except S.TmanRefused as e:
        got = f"refused: {e.line()}"
    check("printer_writes_into_a_folder_whose_path_holds_a_colon_B72", got is True, str(got))
    expect_refused("printer_fields_naming_a_common_key_B71", "entry_field", E.make_entry, "unlock", [], "2026-11-04",
                   {"kind": "cut", "rule": "operator"}, amendment="x", script="s", commit=None)
    row = E.dl_row(doc, rel, sha, "G is cut at the freeze (gate not met since 2026-10-01).")
    check("printer_row_names_path_and_sha", rel in row and sha in row and "| DL-<next> | RECORDED (2026-11-03) |"
          in row)
    session = E.make_entry("unlock", [], "2026-11-04", {"manifest": {"path": S.MANIFEST_REL, "sha256": "1" * 64,
                                                                     "body_sha256": "2" * 64,
                                                                     "appends_sha256": "4" * 64},
                                                        "lanes": {"path": S.LANES_REL, "sha256": "3" * 64},
                                                        "freeze_record": dict(REF)},
                           amendment="AM-17 item 9", script="scripts/tday_manifest.py", commit=None)
    check("session_entry_file_id", E.entry_name(session) == "unlock__session__20261104__01.json")
    expect_refused("printer_refuses_unknown_kind", "entry_kind_unknown", E.make_entry, "launch", [], "2026-11-03",
                   {}, amendment="x", script="s", commit=None)
    repo = FX.new_repo()
    dl_ok = FX.dl_table({"DL-130": ("RECORDED (2026-11-03)", row.split("|")[3].strip())})
    FX.commit_files(repo, {rel: path.read_bytes(), S.DECISION_LOG_REL: dl_ok}, date="2026-11-03T02:00:00+00:00")
    rec = R.Records.at_head(repo)
    got, gsha = E.read_entry_file(rec, rel)
    check("committed_entry_read_back", got == doc and gsha == sha)
    check("committed_entry_read_as_its_kind", E.read_entry_file(rec, rel, doc["kind"]) == (got, gsha))
    expect_refused("committed_entry_read_as_another_kind_refused", "entry_file_invalid", E.read_entry_file, rec, rel,
                   "not_run")
    check("entered_on_its_day_no_report", E.late_entry_report(got, E.entered_day(rec, sha)) is None
          and E.entered_day(rec, sha) == "2026-11-03")
    late_repo = FX.new_repo()
    FX.commit_files(late_repo, {rel: path.read_bytes()}, date="2026-11-03T02:00:00+00:00")
    FX.commit_files(late_repo, {S.DECISION_LOG_REL: dl_ok}, date="2026-11-05T02:00:00+00:00")
    lrec = R.Records.at_head(late_repo)
    rep = E.late_entry_report(doc, E.entered_day(lrec, sha))
    check("late_entry_is_a_report_xv", rep is not None and rep.startswith("deviation: entry later than its day")
          and "entered 2026-11-05" in rep, str(rep))
    check("cutoff_cut_due_the_day_after_T15", E.late_entry_report(dict(doc, rule="cutoff"), "2026-11-04") is None
          and E.late_entry_report(dict(doc, rule="cutoff"), "2026-11-05") is not None)
    decided = FX.new_repo()
    FX.commit_files(decided, {rel: path.read_bytes(),
                              S.DECISION_LOG_REL: FX.dl_table({"DL-130": ("DECIDED", f"x {sha}")})})
    expect_refused("committed_entry_needs_a_recorded_row", "dl_row_missing", E.read_entry_file,
                   R.Records.at_head(decided), rel)
    FX.commit_files(repo, {rel: S.json_file_bytes(dict(doc, reason="no card"))}, msg="rewrite")
    expect_refused("committed_entry_rewritten_refused", "entry_file_rewritten", E.read_entry_file,
                   R.Records.at_head(repo), rel)
    renamed = FX.new_repo()
    wrong = rel.replace("__20261103__", "__20261104__")
    FX.commit_files(renamed, {wrong: path.read_bytes(), S.DECISION_LOG_REL: dl_ok})
    expect_refused("committed_entry_name_states_its_date", "entry_field", E.read_entry_file,
                   R.Records.at_head(renamed), wrong)
    (renamed / wrong).write_bytes(b"{}\n")
    expect_refused("committed_entry_modified_in_the_tree", "record_uncommitted", E.read_entry_file,
                   R.Records.at_head(renamed), wrong)


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN CUTS SMOKE (lane TMAN, B-1, C3)",
                                  (case_gate_order, case_input_gates, case_arithmetic, case_references,
                                   case_freeze_delay_and_new_freeze, case_printers_and_committed_files)))
