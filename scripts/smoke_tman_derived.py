#!/usr/bin/env python3
"""Smoke for derived artifacts (AM-19 Code: "item 4(c)'s derived-artifact entries"; AM-19 Definitions and items
4(b), 4(c); AM-21 items 2(b), 2(c); AM-23 items 2 and 4; ADDENDUM 1 A2) (lane TMAN, B-1, C4).

By inputs at the freeze and hashed by the unlock; the derived slots required at the freeze (E4 at every seed whose
E1 is complete, E7 at every seed whose E3 is complete, every complete QAT run's selection, E7-X and E7-flip, the
sensitivity E4/E7; E7-E2 entered as not made while its code is not merged); E3 seed 43 cut leaves E7 seed 43 not
required; E6 seed 42 hashed at the freeze; a second evaluation directory without its fault report; a non-finite
entry from the telemetry's flags and its decision-log row; the ptq_record of a model of record, read only through
lane LQP's reader (a stand-in here) and projected (A2); appends (an isolation record of the same root and digest made
before the artifact, an image digest of record, unrelated or already-hashed slots refused, the prefix rule); the
isolation record itself (src/data/isolation.assert_trainval_only_root on a fixture root; a TEST surface shown by
os.path.lexists patched for one surface and restored, never created).

Run:  python -B scripts/smoke_tman_derived.py
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

check, expect_refused = FX.check, FX.expect_refused
KD = "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b"
E3S42, E3S43 = "E3:s42:lambda0.5:alpha25:80k", "E3:s43:lambda0.5:alpha25:80k"
FREEZE = "2026-11-03"


def catalogue():
    from src.tman import schema as S
    return S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())


def ptq_inputs(**kw) -> dict:
    d = {"parent_checkpoint_sha256": "1" * 64,
         "calibration_list": {"path": "configs/calibration/ptq_calibration_seed42.json", "file_sha256": "2" * 64},
         "script": {"path": "scripts/run_ptq.py", "commit": "3" * 40}, "image_digest": KD,
         "qconfig_fingerprint": "0123456789abcdef"}
    d.update(kw)
    return d


def qat_inputs(run_slot: str = "E6:s42:15ep", **kw) -> dict:
    d = {"run_slot": run_slot, "telemetry_sha256": "4" * 64, "scripts_commit": "5" * 40, "image_digest": KD,
         "rules_sha256": "76addc57d0367dda9a1163f749ec32c9c240b20736104dc12aa77488344f9568",
         "clip_selection_sha256": "6" * 64}
    d.update(kw)
    return d


def hashed(method: str = "ptq", appended: str = "freeze", made: str = "2026-11-02T10:00:00Z", **kw) -> dict:
    h = {"artifact_sha256": "5" * 64, "state_dict_sha256": "6" * 64, "x86_copy_sha256": "7" * 64,
         "provenance": {"path": "e4/e4_ptq_run_meta.json", "sha256": "8" * 64},
         "record": {"path": "e4/e4_ptq_record.json", "sha256": "9" * 64}, "made_utc": made, "image_digest": KD,
         "isolation": None, "appended": appended}
    if method == "qat_selection":
        h["evaluations"] = [{"dir": "eval_e6_s42", "fault_report": None}]
    h.update(kw)
    return h


def derived(slot: str, parent: str, method: str = "ptq", **kw) -> dict:
    e = {"slot": slot, "kind": "derived", "method": method, "parent": parent,
         "inputs": ptq_inputs() if method == "ptq" else qat_inputs(parent), "hashed": None, "evaluator_stage": "E4",
         "ptq_record": None, "dl_rows": []}
    e.update(kw)
    return e


def run(kind: str, slot: str, sha: str = "1" * 64) -> dict:
    return {"slot": slot, "kind": kind, "checkpoint": {"sha256": sha}, "dl_rows": []}


def world_s42() -> dict:
    """The seed-42 family with its parents complete and every derived slot entered by inputs."""
    w = {k: run("scored_run", k) for k in ("E1:s42:80k", "E2:s42:lambda0.5:80k", E3S42)}
    w["E6:s42:15ep"] = run("listed_run", "E6:s42:15ep")
    w["E5:s42:clip5:15ep"] = run("listed_run", "E5:s42:clip5:15ep")
    w["E5:s42:clip1:15ep"] = run("listed_run", "E5:s42:clip1:15ep")
    for k, p in (("E4:s42:list42", "E1:s42:80k"), ("E4:s42:list43", "E1:s42:80k"), ("E4:s42:list44", "E1:s42:80k"),
                 ("E4:s42:list45", "E1:s42:80k"), ("E7:s42:list42", E3S42), ("E7:s42:list43", E3S42),
                 ("E7:s42:list44", E3S42), ("E7:s42:list45", E3S42)):
        w[k] = derived(k, p)
    w["E6:s42:sel"] = derived("E6:s42:sel", "E6:s42:15ep", "qat_selection", hashed=hashed("qat_selection"))
    w["E5:s42:sel"] = derived("E5:s42:sel", "E5:s42:clip5:15ep", "qat_selection")
    w["E7-E2:s42:list42"] = {"slot": "E7-E2:s42:list42", "kind": "not_made", "reason": "code not merged"}
    return w


def case_inputs_and_parents():
    from src.tman import manifest as M
    c = catalogue()
    e = derived("E4:s42:list42", "E1:s42:80k")
    check("ptq_by_inputs_complete_4c", not M.derived_inputs_errors(e))
    for k in ("parent_checkpoint_sha256", "calibration_list", "script", "image_digest", "qconfig_fingerprint"):
        bad = dict(e, inputs={x: v for x, v in e["inputs"].items() if x != k})
        check(f"ptq_inputs_without_{k}_refused", bool(M.derived_inputs_errors(bad)))
    check("ptq_inputs_script_without_commit_refused", bool(M.derived_inputs_errors(
        dict(e, inputs=ptq_inputs(script={"path": "scripts/run_ptq.py", "commit": "x"})))))
    q = derived("E6:s42:sel", "E6:s42:15ep", "qat_selection")
    check("qat_by_inputs_complete", not M.derived_inputs_errors(q))
    check("qat_inputs_without_rules_sha256_refused_TB13", bool(M.derived_inputs_errors(
        dict(q, inputs={x: v for x, v in q["inputs"].items() if x != "rules_sha256"}))))
    check("hashed_complete", not M.derived_inputs_errors(dict(e, hashed=hashed())))
    check("hashed_without_x86_copy_refused", bool(M.derived_inputs_errors(
        dict(e, hashed={k: v for k, v in hashed().items() if k != "x86_copy_sha256"}))))
    check("qat_hashed_needs_its_evaluations", bool(M.derived_inputs_errors(dict(q, hashed=hashed()))))
    w = world_s42()
    check("parent_complete_accepted_4b", not M.derived_parent_errors(w["E4:s42:list42"], w, c))
    for kind in ("cut", "diverged", "not_run", "aborted", "non_finite"):
        w2 = dict(w, **{"E1:s42:80k": run(kind, "E1:s42:80k")})
        check(f"parent_{kind}_refused", bool(M.derived_parent_errors(w["E4:s42:list42"], w2, c)))
    check("parent_without_entry_refused", bool(M.derived_parent_errors(w["E4:s42:list42"], {}, c)))
    check("parent_other_than_catalogue_refused", bool(M.derived_parent_errors(
        dict(w["E4:s42:list42"], parent="E1:s43:80k"), dict(w, **{"E1:s43:80k": run("scored_run", "E1:s43:80k")}),
        c)))
    check("pilot_selection_parent_is_a_pilot_run_TB9", not M.derived_parent_errors(w["E5:s42:sel"], w, c))
    check("pilot_selection_other_parent_refused", bool(M.derived_parent_errors(
        dict(w["E5:s42:sel"], parent="E6:s42:15ep", inputs=qat_inputs("E6:s42:15ep")), w, c)))
    check("qat_run_slot_not_the_parent_refused", bool(M.derived_parent_errors(
        dict(w["E6:s42:sel"], inputs=qat_inputs("E5:s42:clip5:15ep")), w, c)))


def case_required_and_hashed():
    from src.tman import manifest as M
    c = catalogue()
    w = world_s42()
    req = M.required_derived(c, w)
    check("required_at_the_freeze_seed42_family_4b", sorted(req) == sorted(
        ["E4:s42:list42", "E4:s42:list43", "E4:s42:list44", "E4:s42:list45", "E7:s42:list42", "E7:s42:list43",
         "E7:s42:list44", "E7:s42:list45", "E5:s42:sel", "E6:s42:sel", "E7-E2:s42:list42"]), str(sorted(req)))
    check("required_all_entered", not M.derived_missing_errors(c, w), str(M.derived_missing_errors(c, w)))
    w2 = {k: v for k, v in w.items() if k != "E7:s42:list45"}
    check("sensitivity_e7_missing_refused_AM16_5", any("E7:s42:list45" in x for x in M.derived_missing_errors(c, w2)))
    w3 = dict(w, **{"E4:s42:list42": {"slot": "E4:s42:list42", "kind": "not_made", "reason": "code not merged"}})
    check("not_made_for_merged_code_refused", any("E4:s42:list42" in x for x in M.derived_missing_errors(c, w3)))
    check("e7e2_not_made_while_code_unmerged_TB8", M.code_gated("E7-E2:s42:list42", c)
          and not M.code_gated("E4:s42:list42", c))
    w4 = dict(w, **{"E7-E2:s42:list42": derived("E7-E2:s42:list42", "E2:s42:lambda0.5:80k")})
    check("e7e2_by_inputs_once_made", not M.derived_missing_errors(c, w4))
    w5 = dict(w, **{E3S43: run("cut", E3S43)})
    check("e3_s43_cut_leaves_e7_s43_not_required", "E7:s43:list42" not in M.required_derived(c, w5))
    w6 = dict(w, **{E3S43: run("scored_run", E3S43)})
    check("e3_s43_complete_requires_e7_s43", any("E7:s43:list42" in x for x in M.derived_missing_errors(c, w6)))
    w7 = dict(w, **{"E6:s42:15ep": run("non_finite", "E6:s42:15ep")})
    w7.pop("E6:s42:sel")
    check("non_finite_run_requires_no_selection_A1", "E6:s42:sel" not in M.required_derived(c, w7)
          and not M.e6s42_errors(w7))
    check("e6s42_hashed_at_the_freeze", not M.e6s42_errors(w))
    w8 = dict(w, **{"E6:s42:sel": derived("E6:s42:sel", "E6:s42:15ep", "qat_selection")})
    check("e6s42_by_inputs_at_the_freeze_refused", bool(M.e6s42_errors(w8)))
    check("not_hashed_at_unlock_listed", len(M.not_hashed_errors(w)) == 9, str(M.not_hashed_errors(w)))
    check("all_hashed_at_unlock", not M.not_hashed_errors({"E6:s42:sel": w["E6:s42:sel"]}))
    q = w["E6:s42:sel"]
    check("one_evaluation_directory", not M.selection_conflict_errors(q))
    two = dict(q, hashed=hashed("qat_selection", evaluations=[{"dir": "a", "fault_report": None},
                                                              {"dir": "b", "fault_report": None}]))
    check("second_evaluation_without_fault_report_refused_AM21_2c", bool(M.selection_conflict_errors(two)))
    ok = dict(q, hashed=hashed("qat_selection", evaluations=[{"dir": "a", "fault_report": None},
                                                             {"dir": "b", "fault_report": {"path": "r.md",
                                                                                           "sha256": "a" * 64}}]))
    check("second_evaluation_with_fault_report", not M.selection_conflict_errors(ok))


def case_non_finite():
    from src.tman import completion as C
    from src.tman import manifest as M
    from src.tman import records as R
    root = FX.scratch_dir()
    d = FX.qat_run(root, "e6_s44_15ep", nonfinite_from=1)
    rows, _, sha = C.qat_rows(d)
    dl = R.DecisionLog(FX.dl_table({"DL-140": ("RECORDED", "E6 seed 44: non-finite: no model (AM-21 item 2(b))"),
                                    "DL-141": ("RECORDED", "other")}))
    e = {"slot": "E6:s44:15ep", "kind": "non_finite", "telemetry_sha256": sha, "dl_rows": ["DL-140"]}
    check("non_finite_with_row_and_flags_AM21_2b", not M.non_finite_errors(e, rows, sha, dl))
    check("non_finite_without_runs_reads_its_row", not M.non_finite_errors(e, None, None, dl))
    check("non_finite_without_its_row_refused", bool(M.non_finite_errors(dict(e, dl_rows=["DL-141"]), None, None, dl)))
    check("non_finite_other_telemetry_refused", bool(M.non_finite_errors(e, rows, "0" * 64, dl)))
    d2 = FX.qat_run(root, "e6_s43_15ep", nonfinite_from=3)
    rows2, _, sha2 = C.qat_rows(d2)
    check("non_finite_with_a_convertible_epoch_refused", bool(M.non_finite_errors(dict(e, telemetry_sha256=sha2),
                                                                                  rows2, sha2, dl)))
    d3 = FX.qat_run(root, "e6_s45_15ep", epochs=9, nonfinite_from=1)
    rows3, _, sha3 = C.qat_rows(d3)
    check("non_finite_of_an_incomplete_run_refused", bool(M.non_finite_errors(dict(e, telemetry_sha256=sha3), rows3,
                                                                              sha3, dl)))


def case_ptq_record():
    from src.tman import manifest as M
    c = catalogue()
    root = FX.scratch_dir()
    parent = run("scored_run", "E1:s42:80k", "1" * 64)
    read, sha = FX.ptq_reader_stub(FX.ptq_record_doc())
    made = derived("E4:s42:list42", "E1:s42:80k", hashed=hashed(), ptq_record={"path": "e4/r.json", "sha256": sha})
    check("ptq_record_of_record_accepted_A2", not M.ptq_record_errors(made, c, parent, "3" * 64, runs_root=root,
                                                                      reader=read))
    check("ptq_record_read_only_under_runs", not M.ptq_record_errors(made, c, parent, "3" * 64, runs_root=None))
    check("by_inputs_without_ptq_record_accepted", not M.ptq_record_errors(
        derived("E4:s42:list42", "E1:s42:80k"), c, parent, "3" * 64, runs_root=root, reader=read))
    check("made_without_ptq_record_refused", bool(M.ptq_record_errors(dict(made, ptq_record=None), c, parent,
                                                                      "3" * 64, runs_root=root, reader=read)))
    for name, doc in (("stage", FX.ptq_record_doc(stage="E7")), ("seed", FX.ptq_record_doc(seed=43)),
                      ("role_d3_replicate", FX.ptq_record_doc(role="d3_replicate")),
                      ("role_calibration_sensitivity", FX.ptq_record_doc(role="calibration_sensitivity"))):
        r2, s2 = FX.ptq_reader_stub(doc)
        check(f"ptq_record_other_{name}_refused_A2", bool(M.ptq_record_errors(
            dict(made, ptq_record={"path": "e4/r.json", "sha256": s2}), c, parent, "3" * 64, runs_root=root,
            reader=r2)))
    check("ptq_record_other_parent_checkpoint_refused", bool(M.ptq_record_errors(
        made, c, run("scored_run", "E1:s42:80k", "0" * 64), "3" * 64, runs_root=root, reader=read)))
    check("ptq_record_other_calibration_refused", bool(M.ptq_record_errors(made, c, parent, "0" * 64, runs_root=root,
                                                                           reader=read)))
    check("ptq_record_other_artifact_refused", bool(M.ptq_record_errors(
        dict(made, hashed=hashed(state_dict_sha256="0" * 64)), c, parent, "3" * 64, runs_root=root, reader=read)))
    expect_refused("ptq_record_sha256_mismatch_refused", "ptq_record_invalid", M.ptq_record_errors,
                   dict(made, ptq_record={"path": "e4/r.json", "sha256": "0" * 64}), c, parent, "3" * 64,
                   runs_root=root, reader=read)
    rs, ss_ = FX.ptq_reader_stub(FX.ptq_record_doc(role="calibration_sensitivity"))
    sens = derived("E4:s42:list43", "E1:s42:80k", hashed=hashed(), ptq_record={"path": "e4/s.json", "sha256": ss_})
    check("sensitivity_record_is_descriptive", not M.ptq_record_errors(sens, c, parent, "3" * 64, runs_root=root,
                                                                      reader=rs))
    check("sensitivity_without_record_accepted", not M.ptq_record_errors(dict(sens, ptq_record=None), c, parent,
                                                                         "3" * 64, runs_root=root, reader=rs))
    check("sensitivity_with_a_record_of_record_refused", bool(M.ptq_record_errors(
        dict(sens, ptq_record={"path": "e4/r.json", "sha256": sha}), c, parent, "3" * 64, runs_root=root,
        reader=read)))
    check("qat_selection_carries_no_ptq_record", not M.ptq_record_errors(
        derived("E6:s42:sel", "E6:s42:15ep", "qat_selection", hashed=hashed("qat_selection")), c,
        run("listed_run", "E6:s42:15ep"), "3" * 64, runs_root=root, reader=read))


def iso(root: str, utc: str = "2026-11-04T09:00:00Z", digest: str = KD) -> dict:
    return {"format": "tday_isolation/1", "root": root, "image_digest": digest, "utc": utc,
            "result": {"counts": {}, "test_surfaces": {}, "method": "x"}}


def case_appends():
    from src.tman import manifest as M
    from src.tman import schedule as D
    fz = D.cutoff_instant(FREEZE)
    frozen = derived("E4:s43:list42", "E1:s43:80k")
    before = {"slot": "E4:s43:list42", "hashed": hashed(appended="append")}
    check("append_made_before_the_freeze", not M.append_errors(before, frozen, fz))
    after = {"slot": "E4:s43:list42", "hashed": hashed(appended="append", made="2026-11-04T10:00:00Z",
                                                       isolation={"path": "iso.json", "sha256": "a" * 64})}
    prov = {"calibration": {"data_root": "/workspace/trainval_root"}}
    check("append_after_the_freeze_with_isolation_xi", not M.append_errors(
        after, frozen, fz, isolation=iso("/workspace/trainval_root"), provenance=prov))
    check("append_after_the_freeze_without_isolation_refused", bool(M.append_errors(
        dict(after, hashed=dict(after["hashed"], isolation=None)), frozen, fz, provenance=prov)))
    check("isolation_after_the_artifact_refused", bool(M.append_errors(
        after, frozen, fz, isolation=iso("/workspace/trainval_root", utc="2026-11-04T11:00:00Z"), provenance=prov)))
    check("isolation_of_another_root_refused", bool(M.append_errors(
        after, frozen, fz, isolation=iso("/workspace/other"), provenance=prov)))
    check("isolation_of_another_digest_refused", bool(M.append_errors(
        after, frozen, fz, isolation=iso("/workspace/trainval_root", digest="sha256:" + "0" * 64), provenance=prov)))
    check("ptq_after_the_freeze_with_intact_record_TB19_Q9", not M.append_errors(
        dict(after, hashed=dict(after["hashed"], isolation=None)), frozen, fz, ptq_record_ok=True))
    qfrozen = derived("E6:s43:sel", "E6:s43:15ep", "qat_selection")
    qafter = {"slot": "E6:s43:sel", "hashed": hashed("qat_selection", appended="append", made="2026-11-04T10:00:00Z",
                                                     isolation={"path": "iso.json", "sha256": "a" * 64})}
    check("qat_after_the_freeze_with_isolation", not M.append_errors(qafter, qfrozen, fz,
                                                                      isolation=iso("/somewhere")))
    check("qat_after_the_freeze_record_does_not_stand_in", bool(M.append_errors(
        dict(qafter, hashed=dict(qafter["hashed"], isolation=None)), qfrozen, fz, ptq_record_ok=True)))
    check("append_unrelated_slot_refused", bool(M.append_errors(dict(before, slot="E4:s44:list42"), None, fz)))
    check("append_to_a_hashed_entry_refused", bool(M.append_errors(before, dict(frozen, hashed=hashed()), fz)))
    check("append_to_a_run_refused", bool(M.append_errors(before, run("scored_run", "E4:s43:list42"), fz)))
    check("append_digest_not_of_record_refused", bool(M.append_errors(
        {"slot": "E4:s43:list42", "hashed": hashed(appended="append", image_digest="sha256:" + "0" * 64)}, frozen, fz)))
    check("append_digest_other_than_declared_refused", bool(M.append_errors(
        before, dict(frozen, inputs=ptq_inputs(image_digest=FX_STUDENT)), fz)))
    check("append_marked_freeze_refused", bool(M.append_errors(
        {"slot": "E4:s43:list42", "hashed": hashed(appended="freeze")}, frozen, fz)))
    check("append_with_inputs_refused", bool(M.append_errors(dict(before, inputs=ptq_inputs()), frozen, fz)))
    check("append_incomplete_hashed_refused", bool(M.append_errors(
        {"slot": "E4:s43:list42", "hashed": {k: v for k, v in hashed(appended="append").items() if k != "record"}},
        frozen, fz)))
    a, b = [before], [before, after]
    check("appends_grow", M.appends_prefix_error(a, b) is None)
    check("appends_changed_refused", M.appends_prefix_error(b, [after, before]) is not None
          and M.appends_prefix_error(b, a) is not None)


FX_STUDENT = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"


def case_isolation_record():
    from src.tman import manifest as M
    from src.tman import schema as S
    root = FX.scratch_dir("tmanroot")
    for split in ("train", "val"):
        for sub, ext in (("images", ".jpg"), ("annotations", ".png")):
            p = root / sub / split
            p.mkdir(parents=True)
            (p / f"a{ext}").write_bytes(b"x")
    rec = M.isolation_record(root, utc="2026-11-04T09:00:00Z", image_digest=KD, expected={"train": 1, "val": 1})
    check("isolation_record_on_a_trainval_root_xi", rec["format"] == "tday_isolation/1" and rec["root"] == str(root)
          and rec["result"]["counts"] == {"train": {"images": 1, "masks": 1}, "val": {"images": 1, "masks": 1}}
          and set(rec["result"]["test_surfaces"].values()) == {"absent"})
    check("isolation_record_reads_back", M.read_isolation(S.json_file_bytes(rec), "iso.json") == rec)
    expect_refused("isolation_record_malformed_refused", "append_invalid", M.read_isolation, b'{"format": "x"}',
                   "iso.json")
    saved = os.path.lexists
    surface = os.path.join(str(root), "annotation_" + "te" + "st.json")
    try:
        os.path.lexists = lambda p: True if str(p) == surface else saved(p)
        code = FX.refused_code(M.isolation_record, root, utc="2026-11-04T09:00:00Z", image_digest=KD,
                               expected={"train": 1, "val": 1})
    finally:
        os.path.lexists = saved
    check("isolation_refuses_a_test_surface_present", code == "append_invalid", str(code))
    expect_refused("isolation_refuses_a_wrong_count", "append_invalid", M.isolation_record, root,
                   utc="2026-11-04T09:00:00Z", image_digest=KD, expected={"train": 2, "val": 1})


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN DERIVED SMOKE (lane TMAN, B-1, C4)",
                                  (case_inputs_and_parents, case_required_and_hashed, case_non_finite,
                                   case_ptq_record, case_appends, case_isolation_record)))
