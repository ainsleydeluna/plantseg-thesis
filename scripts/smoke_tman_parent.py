#!/usr/bin/env python3
"""Smoke for the parent rule (AM-19 Code: "the parent rule"; AM-19 item 4(a): "When a parent is cut or diverged, its
descendants at that seed are not run") (lane TMAN, B-1, C3).

Manifest entries by slot, synthetic: E3 seed 43 cut or diverged leaves E6 and E7 seed 43 not run; E1 seed 44 cut
leaves E4 and E5 seed 44 not run (E4 is never cut: a not-run entry is its exemption); a QAT selection has no entry
when its run yields no model (IF§11.2 via ADDENDUM 1); a not-run chain passes its root cause on; an arm cut from its
first incomplete run onward is one entry (AM-18 item 7(b)); and the refusals: a not-run entry with a live parent, a
wrong reason or another record; a scored or derived entry under a parent without a model; a cut or diverged parent
whose descendants are not entered as not run. The not-run printer's own check reads the parent's entry.

Run:  python -B scripts/smoke_tman_parent.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check = FX.check
E3S43 = "E3:s43:lambda0.5:alpha25:80k"
CUT_SHA, TEL_SHA = "c" * 64, "d" * 64


def catalogue():
    from src.tman import schema as S
    return S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())


def entry(kind: str, slot: str, cat, **kw) -> dict:
    """The fields of a manifest entry the parent rule reads: kind, the cut's entry file, the divergence's telemetry,
    a not-run's reason, parent, parent record and own entry file."""
    e = {"slot": slot, "kind": kind}
    if kind == "cut":
        e["entry"] = {"path": f"reports/derived/tday_entries/cut__{slot}.json", "sha256": CUT_SHA}
    if kind == "diverged":
        e["telemetry_sha256"] = TEL_SHA
    if kind == "not_run":
        e.update(reason="parent cut", parent=cat.slots[slot]["parent"],
                 parent_record={"path": "x", "sha256": CUT_SHA},
                 entry={"path": f"reports/derived/tday_entries/not_run__{slot}.json", "sha256": "e" * 64})
    e.update(kw)
    return e


def not_run_under(parent_kind: str, slot: str, cat, chain_sha: str | None = None) -> dict:
    """A not-run entry citing its parent's record: a cut's entry file, a divergence's telemetry, or (chain) the
    parent's not-run entry file."""
    reason = "parent diverged" if parent_kind == "diverged" else "parent cut"
    sha = chain_sha or (TEL_SHA if parent_kind == "diverged" else CUT_SHA)
    return entry("not_run", slot, cat, reason=reason, parent_record={"path": "x", "sha256": sha})


def clean(world: dict, cat) -> list:
    from src.tman import entries as E
    return E.parent_rule_errors(world, cat) + E.not_run_errors(world, cat) + E.descendant_errors(world, cat)


def case_valid_worlds():
    c = catalogue()
    w = {E3S43: entry("cut", E3S43, c), "E6:s43:15ep": not_run_under("cut", "E6:s43:15ep", c),
         "E7:s43:list42": not_run_under("cut", "E7:s43:list42", c)}
    check("e3_s43_cut_e6_e7_not_run_sel_without_entry", not clean(w, c), str(clean(w, c)))
    d = {E3S43: entry("diverged", E3S43, c), "E6:s43:15ep": not_run_under("diverged", "E6:s43:15ep", c),
         "E7:s43:list42": not_run_under("diverged", "E7:s43:list42", c)}
    check("e3_s43_diverged_e6_e7_not_run", not clean(d, c), str(clean(d, c)))
    e1 = "E1:s44:80k"
    w2 = {e1: entry("cut", e1, c), "E4:s44:list42": not_run_under("cut", "E4:s44:list42", c),
          "E5:s44:15ep": not_run_under("cut", "E5:s44:15ep", c)}
    check("e1_s44_cut_e4_e5_not_run_never_cut_exempt", not clean(w2, c) and c.slots["E4:s44:list42"]["never_cut"],
          str(clean(w2, c)))
    t = "teacher-R2:s42:40k"
    x = "E3-R2:s42:lambda0.5:alpha25:80k"
    chain = entry("not_run", x, c, reason="parent diverged", parent_record={"path": "x", "sha256": TEL_SHA})
    w3 = {t: entry("diverged", t, c), x: chain,
          "E7-R2:s42:list42": not_run_under("cut", "E7-R2:s42:list42", c, chain["entry"]["sha256"])
          | {"reason": "parent diverged"},
          "E6-R2:s42:15ep": not_run_under("cut", "E6-R2:s42:15ep", c, chain["entry"]["sha256"])
          | {"reason": "parent diverged"}}
    check("diverged_arm_teacher_chain_passes_its_cause_on", not clean(w3, c), str(clean(w3, c)))
    r1 = ["teacher-R1:s42:40k", "E3-R1:s42:lambda0.5:alpha25:80k"]
    w4 = {r1[0]: entry("cut", r1[0], c), r1[1]: entry("cut", r1[1], c),
          "E7-R1:s42:list42": not_run_under("cut", "E7-R1:s42:list42", c),
          "E6-R1:s42:15ep": not_run_under("cut", "E6-R1:s42:15ep", c)}
    check("arm_cut_in_one_entry_AM18_7b", not clean(w4, c), str(clean(w4, c)))
    q = "E6:s44:15ep"
    w5 = {q: entry("non_finite", q, c)}
    check("qat_run_without_model_its_sel_without_entry_A1", not clean(w5, c), str(clean(w5, c)))
    w6 = {"E1:s43:80k": entry("scored_run", "E1:s43:80k", c), "E4:s43:list42": entry("derived", "E4:s43:list42", c)}
    check("live_parent_live_child", not clean(w6, c))


def case_refusals():
    from src.tman import entries as E
    c = catalogue()
    e1 = "E1:s44:80k"
    live = {e1: entry("scored_run", e1, c), "E4:s44:list42": not_run_under("cut", "E4:s44:list42", c)}
    got = E.not_run_errors(live, c)
    check("not_run_with_a_live_parent_refused", len(got) == 1 and "E4:s44:list42" in got[0], str(got))
    absent = {"E4:s44:list42": not_run_under("cut", "E4:s44:list42", c)}
    check("not_run_with_an_absent_parent_refused", len(E.not_run_errors(absent, c)) == 1)
    wrong = {E3S43: entry("cut", E3S43, c), "E6:s43:15ep": not_run_under("diverged", "E6:s43:15ep", c),
             "E7:s43:list42": not_run_under("cut", "E7:s43:list42", c)}
    got = E.not_run_errors(wrong, c)
    check("not_run_reason_differs_from_the_parent_refused", len(got) == 1 and "E6:s43:15ep" in got[0], str(got))
    other = {E3S43: entry("cut", E3S43, c), "E6:s43:15ep": not_run_under("cut", "E6:s43:15ep", c, "9" * 64),
             "E7:s43:list42": not_run_under("cut", "E7:s43:list42", c)}
    check("not_run_citing_another_record_refused", len(E.not_run_errors(other, c)) == 1)
    moved = {E3S43: entry("cut", E3S43, c), "E7:s43:list42": not_run_under("cut", "E7:s43:list42", c)
             | {"parent": "E3:s42:lambda0.5:alpha25:80k"}, "E6:s43:15ep": not_run_under("cut", "E6:s43:15ep", c)}
    check("not_run_naming_another_parent_refused", len(E.not_run_errors(moved, c)) == 1)
    scored = {E3S43: entry("cut", E3S43, c), "E7:s43:list42": entry("derived", "E7:s43:list42", c),
              "E6:s43:15ep": not_run_under("cut", "E6:s43:15ep", c)}
    got = E.parent_rule_errors(scored, c)
    check("derived_under_a_cut_parent_refused", len(got) == 1 and "E7:s43:list42" in got[0], str(got))
    for kind in ("diverged", "aborted", "non_finite", "superseded", "not_run"):
        w = {E3S43: entry(kind, E3S43, c), "E7:s43:list42": entry("derived", "E7:s43:list42", c)}
        check(f"derived_under_a_{kind}_parent_refused", any("E7:s43:list42" in x for x in E.parent_rule_errors(w, c)))
    q = "E6:s44:15ep"
    sel = {q: entry("non_finite", q, c), "E6:s44:sel": entry("derived", "E6:s44:sel", c)}
    check("qat_sel_entered_under_a_run_without_model_refused_A1", any("E6:s44:sel" in x
                                                                     for x in E.parent_rule_errors(sel, c)))
    missing = {E3S43: entry("cut", E3S43, c)}
    got = E.descendant_errors(missing, c)
    check("cut_parent_without_its_descendants_refused", sorted(x.split()[0] for x in got)
          == ["E6:s43:15ep", "E7:s43:list42"], str(got))
    sep = {E3S43: entry("cut", E3S43, c),
           "E6:s43:15ep": entry("cut", "E6:s43:15ep", c) | {"entry": {"path": "y", "sha256": "8" * 64}},
           "E7:s43:list42": not_run_under("cut", "E7:s43:list42", c)}
    got = E.descendant_errors(sep, c)
    check("descendant_cut_by_its_own_entry_refused", len(got) == 1 and "E6:s43:15ep" in got[0], str(got))
    div = {E3S43: entry("diverged", E3S43, c), "E7:s43:list42": not_run_under("diverged", "E7:s43:list42", c)}
    check("diverged_parent_with_an_unentered_child_refused", len(E.descendant_errors(div, c)) == 1)


def case_not_run_printer():
    from src.tman import entries as E
    c = catalogue()
    doc = {"slots": ["E6:s43:15ep", "E7:s43:list42"], "reason": "parent cut", "parent": E3S43,
           "parent_record": {"path": "reports/derived/tday_entries/cut__E3_s43.json", "sha256": CUT_SHA}}
    cut = entry("cut", E3S43, c)
    check("not_run_printer_accepts_a_cut_parent", not E.not_run_entry_errors(doc, c, cut))
    check("not_run_printer_refuses_a_live_parent", bool(E.not_run_entry_errors(doc, c, entry("scored_run", E3S43,
                                                                                              c))))
    check("not_run_printer_refuses_no_parent_entry", bool(E.not_run_entry_errors(doc, c, None)))
    check("not_run_printer_refuses_wrong_reason", bool(E.not_run_entry_errors(dict(doc, reason="parent diverged"), c,
                                                                              cut)))
    check("not_run_printer_refuses_another_record", bool(E.not_run_entry_errors(
        dict(doc, parent_record={"path": "x", "sha256": "9" * 64}), c, cut)))
    check("not_run_printer_refuses_a_slot_of_another_parent", bool(E.not_run_entry_errors(
        dict(doc, slots=["E6:s44:15ep"]), c, cut)))
    chain = entry("not_run", "E6:s43:15ep", c)
    check("not_run_printer_follows_a_not_run_parent", not E.not_run_entry_errors(
        {"slots": ["E6:s43:sel"], "reason": "parent cut", "parent": "E6:s43:15ep",
         "parent_record": {"path": "x", "sha256": chain["entry"]["sha256"]}}, c, chain))
    dv = entry("diverged", E3S43, c)
    check("not_run_printer_cites_the_divergence_telemetry", not E.not_run_entry_errors(
        dict(doc, reason="parent diverged", parent_record={"path": "x", "sha256": TEL_SHA}), c, dv))
    check("root_cause_through_a_chain", E.root_cause("E6:s43:sel", {E3S43: entry("cut", E3S43, c),
                                                                    "E6:s43:15ep": chain}, c) == "parent cut"
          and E.root_cause("E6:s43:sel", {}, c) is None)
    check("catalogue_parents_of_record_AM19_4b", c.slots["E4:s43:list42"]["parent"] == "E1:s43:80k"
          and c.slots["E7:s43:list42"]["parent"] == E3S43 and c.slots["E5:s43:15ep"]["parent"] == "E1:s43:80k"
          and c.slots["E6:s43:15ep"]["parent"] == E3S43)


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN PARENT-RULE SMOKE (lane TMAN, B-1, C3)",
                                  (case_valid_worlds, case_refusals, case_not_run_printer)))
