#!/usr/bin/env python3
"""The default candidates on TEST (lane LREP, B-2): src/stats/default_candidates.py against AM-17
item 6, the plan of record's §2.2 and D-10 (LRB16).

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world, whose E2 and E3 seed-42 inputs carry DL-109's
and DL-121's selected checkpoints and whose two default_candidate slots carry their defaults. Created
paths live under a new directory in TMPDIR (left in place); none contains "test".

  V  values: m, differences, per-image n / mean / median / HL from the A3a primitives, the rows of
     record equal to the statistics artifact's, the coincidence rule, the GT-present and disease-only
     fields equal to the AM-6 table's
  R  refusals: another candidate, a default that is not complete, a missing default, a selection
     record that differs, a record that is not canvas
  D  no p-value or interval key, determinism, the record sentence, no 'test' path

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_default_candidates.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import lrep_fixtures as LF                                               # noqa: E402
from src.stats import am6_tables as AM6                                  # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import default_candidates as DC                           # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, align_runs              # noqa: E402
from src.stats.bootstrap import descriptive_scalars                      # noqa: E402
from src.stats.noninferiority import pooled_miou_from_totals             # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, paths={})


def stats_artifact():
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def doc_of(inventory=None, record=DC.SELECTION_RECORD):
    st = stats_artifact()
    inv = H.open_inventory(inventory or CTX.world.inventory_path, policy=st.policy)
    return DC.build_default_candidates(st, inv, H.planned_cells(inv), input_root=CTX.world.input_root,
                                       record=record)


def variant(name, edit):
    if name not in CTX.paths:
        export, binding = copy.deepcopy(CTX.world.export), copy.deepcopy(CTX.world.binding)
        edit(export, binding)
        CTX.paths[name] = LF.write_inventory(CTX.world.fresh(name), export, binding)
    return CTX.paths[name]


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:300]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def m(stats):
    return pooled_miou_from_totals(stats.dataset_tp, stats.dataset_gt, stats.dataset_pred)


def default_run(stage):
    tail = "lambda1" if stage == "E2" else "alpha50"
    w = CTX.world.written[(f"{stage}:s42:{tail}", "fp32", "clean", None)]
    from src.stats import driver as D
    from src.stats.ingest import Policy
    return D.load_input(stage, D.CLEAN, w.rel, CTX.world.input_root, Policy.NONOFFICIAL_SMOKE).run


def rows():
    return {r["row"]: r for r in doc_of()["rows"]}


# --------------------------------------------------------------------------------------------------
def v_m(rid, base, cand):
    def fn():
        r = rows()[rid]
        v = r["values"]
        return (v["m_baseline"] == m(stats_artifact().clean(base).run.stats)
                and v["m_candidate"] == m(default_run(cand).stats)), str(v)[:200]
    return fn


def v_difference(rid):
    def fn():
        v = rows()[rid]["values"]
        return v["dataset_level_difference"] == v["m_candidate"] - v["m_baseline"], ""
    return fn


def _primitives(rid, base, cand):
    st = stats_artifact()
    pv = align_runs(st.clean(base).run, default_run(cand), policy=st.policy, metric=METRIC_DISEASE_ONLY)
    return rows()[rid]["values"]["per_image"], descriptive_scalars(pv.delta, metric=pv.metric,
                                                                   policy=pv.policy.value), pv


def v_n():
    got, _, pv = _primitives("D1", "E1", "E2")
    return got["n_paired"] == pv.n == CTX.world.n, str(got)


def v_mean():
    got, ds, _ = _primitives("D1", "E1", "E2")
    return got["mean_delta"] == ds.mean_delta, f"{got['mean_delta']} {ds.mean_delta}"


def v_median():
    got, ds, _ = _primitives("D2", "E2", "E3")
    return got["median_delta"] == ds.median_delta, f"{got['median_delta']} {ds.median_delta}"


def v_hl():
    got, ds, _ = _primitives("D2", "E2", "E3")
    return got["hodges_lehmann_shift"] == ds.hodges_lehmann_shift, ""


def v_record(rid, cid):
    def fn():
        fam = stats_artifact().family
        comp = next(c for c in fam["a3a_family"]["comparisons"] if c["comparison_id"] == cid)
        task = next(t for t in fam["bootstrap_tasks"] if t["task_id"] == f"{cid}__dataset_miou_delta")
        r = rows()[rid]["run_of_record"]
        return (r["comparison_id"] == cid and r["dataset_level_difference"] == task["observed"]
                and r["per_image"]["mean_delta"] == comp["shifts"]["mean_delta"]
                and r["per_image"]["hodges_lehmann_shift"] == comp["hodges_lehmann_shift"]), ""
    return fn


#: Texts written out here rather than read from the module under test (R4-03).
COINCIDES = "coincides with the run of record (AM-17 item 6)"
RECORD_SENTENCE = ("The selected runs remain the runs of record whatever these comparisons show "
                   "(AM-17 item 6).")


def v_coincide():
    """A default that is its selection is not scored a second time; its markdown rows have the
    header's 12 cells (R3-10)."""
    rec = copy.deepcopy(DC.SELECTION_RECORD)
    rec["E3"]["default_checkpoint"] = rec["E3"]["selected_checkpoint"]

    def edit(export, binding):
        export["entries"] = [e for e in export["entries"] if e["slot"] != "E3:s42:alpha50"]
    d = doc_of(variant("no_e3_default", edit), record=rec)
    r = {x["row"]: x for x in d["rows"]}
    md = DC.render_markdown(d)
    width = {ln.count("|") for ln in md if ln.startswith("| ")}
    return (r["D2"]["coincides"] is True and r["D2"]["values"] is None and r["D2"]["note"] == COINCIDES
            and r["D1"]["coincides"] is False and width == {13}), f"{str(r['D2'])[:150]} {width}"


def v_gt_present():
    v = rows()["D1"]["values"]
    run = default_run("E2")
    w = CTX.world.written[("E2:s42:lambda1", "fp32", "clean", None)]
    summary = json.loads((CTX.world.input_root / w.rel / "summary.json").read_text(encoding="utf-8"))
    g = AM6.gt_present_row(run.stats, summary, with_classes=False)
    return v["candidate_gt_present_all_class"] == g["gt_present_all_class_float64"], ""


def v_disease_only():
    v = rows()["D2"]["values"]
    run = default_run("E3")
    w = CTX.world.written[("E3:s42:alpha50", "fp32", "clean", None)]
    summary = json.loads((CTX.world.input_root / w.rel / "summary.json").read_text(encoding="utf-8"))
    g = AM6.gt_present_row(run.stats, summary, with_classes=False)
    return v["candidate_union_present_disease_only"] == g["union_present_disease_only_float64"], ""


def v_candidate_rules():
    """On hand totals where class 3 is predicted but absent from ground truth, the candidate's
    GT-present fields are the GT-present rule's and differ from the union rule's (the world has no such
    class, k = 0; R4-01)."""
    import numpy as np
    from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, class_totals, rule_variants
    cls, tp, gt, pred = (np.array(v, np.int64) for v in ([0, 1, 3], [90, 8, 0], [100, 10, 0], [95, 9, 5]))
    t = class_totals({"class_id": cls, "tp": tp, "gt": gt, "pred": pred}, 116)
    rv = rule_variants(t)
    got = DC.candidate_values(t)
    want = {"candidate_gt_present_all_class": rv[GT_PRESENT]["all_class"].value_float64,
            "candidate_gt_present_disease_only": rv[GT_PRESENT]["disease_only"].value_float64,
            "candidate_union_present_disease_only": rv[UNION_PRESENT]["disease_only"].value_float64}
    differ = all(rv[GT_PRESENT][s].value_float64 != rv[UNION_PRESENT][s].value_float64
                 for s in ("all_class", "disease_only"))
    return got == want and differ, f"{got} vs {want}"


# --------------------------------------------------------------------------------------------------
def r_other_candidate():
    def edit(export, binding):
        row = next(r for r in export["models"] if r["slot"] == "E2:s42:lambda1")
        row["source"]["sha256"] = DC.SELECTION_RECORD["E2"]["other_checkpoints"]["0.25"]
    return refused("default_candidate_invalid", lambda: doc_of(variant("other_lambda", edit)))


def r_not_complete():
    def edit(export, binding):
        e = next(e for e in export["entries"] if e["slot"] == "E3:s42:alpha50")
        e.update(kind="cut", cut_date="2026-10-22", rule="AM-19 item 2", comparator=None)
    return refused("default_candidate_invalid", lambda: doc_of(variant("default_cut", edit)))


def r_missing():
    def edit(export, binding):
        export["entries"] = [e for e in export["entries"] if e["slot"] != "E3:s42:alpha50"]
    return refused("default_candidate_missing", lambda: doc_of(variant("no_e3_default", edit)))


def r_selection():
    rec = copy.deepcopy(DC.SELECTION_RECORD)
    rec["E2"]["selected_checkpoint"] = rec["E2"]["other_checkpoints"]["2"]
    return refused("selection_record_mismatch", lambda: doc_of(record=rec))


def r_not_canvas():
    def edit(export, binding):
        r = next(r for r in binding["artifacts"] if r["slot"] == "E2:s42:lambda1")
        r["protocol"] = "upstream"
    return refused("not_canvas", lambda: doc_of(variant("default_upstream", edit)))


# --------------------------------------------------------------------------------------------------
def d_no_inference_keys():
    keys = []

    def walk(v):
        if isinstance(v, dict):
            keys.extend(v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(doc_of())
    bad = [k for k in keys if any(s in k.lower() for s in ("p_value", "interval", "bound", "holm",
                                                           "reject", "bca"))]
    return not bad, str(bad)


def d_deterministic():
    return A.canonical_json(doc_of()) == A.canonical_json(doc_of()), ""


def d_deterministic_md():
    a, b = doc_of(), doc_of()
    return DC.render_markdown(a) == DC.render_markdown(b), ""


def d_sentence():
    d = doc_of()
    md = "\n".join(DC.render_markdown(d))
    return d["record_sentence"] == RECORD_SENTENCE and RECORD_SENTENCE in md, ""


def d_no_test_path():
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower()]
    return not bad, str(bad[:3])


def setup(root: Path, world=None) -> None:
    CTX.world = world or LF.build_world(root, strata=False)
    CTX.stats, CTX.paths = None, {}


def cases():
    return [("V1", "D1 m(E1 of record) and m(E2 at lambda 1)", v_m("D1", "E1", "E2")),
            ("V2", "D2 m(E2 of record) and m(E3 at alpha 50)", v_m("D2", "E2", "E3")),
            ("V3", "D1 difference is m(cand) - m(base)", v_difference("D1")),
            ("V4", "D2 difference is m(cand) - m(base)", v_difference("D2")),
            ("V5", "per-image n_paired", v_n),
            ("V6", "per-image mean delta (the A3a primitives)", v_mean),
            ("V7", "per-image median delta", v_median),
            ("V8", "per-image Hodges-Lehmann shift", v_hl),
            ("V9", "D1's run-of-record row equals the statistics artifact's", v_record("D1",
                                                                                     "accuracy_e1_e2")),
            ("V10", "D2's run-of-record row equals the statistics artifact's", v_record("D2",
                                                                                      "accuracy_e2_e3")),
            ("V11", "a default that is its selection coincides: no second artifact; 12 cells a row",
             v_coincide),
            ("V12", "the candidate's GT-present value equals the AM-6 table's", v_gt_present),
            ("V13", "the candidate's disease-only value equals the AM-6 table's", v_disease_only),
            ("V14", "the candidate's GT-present fields follow the GT-present rule (a union-only class)",
             v_candidate_rules),
            ("R1", "another lambda candidate under default_candidate is refused", r_other_candidate),
            ("R2", "a default that is not complete is refused", r_not_complete),
            ("R3", "a missing default is refused", r_missing),
            ("R4", "a selected checkpoint that is not DL-109's is refused", r_selection),
            ("R5", "a default whose record is not canvas is refused", r_not_canvas),
            ("D1", "no p-value, interval or Holm key", d_no_inference_keys),
            ("D2", "byte-identical JSON on repeated runs", d_deterministic),
            ("D3", "identical markdown on repeated runs", d_deterministic_md),
            ("D4", "the record sentence is present", d_sentence),
            ("D5", "no created path contains 'test'", d_no_test_path)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_defaults_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_default_candidates: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'DEFAULT CANDIDATES OK' if good == len(results) else 'DEFAULT CANDIDATES FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; no table here is a thesis result.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
