#!/usr/bin/env python3
"""L-AM8 seed stability and the per-seed tables (lane LREP, B-1): src/stats/seed_stability.py and its
input layer src/stats/heldout_inputs.py, against the plan of record's §1 and rulings LRB2-LRB15.

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py writes the world (the 37 seed-42 inputs and their
statistics artifact, seeds 43/44, an inventory export and a bound binding in the IF§11 v2 layout,
telemetry) under a new directory in TMPDIR, which is left in place. No PlantSeg data, checkpoint,
model or GPU; no created path contains "test".

  F  formulas against hand values: m(S,k) on hand totals, the effect directions, the nested mC (and
     its equality with report.dataset_miou_c), g and d, the mean gain by math.fsum, statistics.stdev
     (n - 1), condition 2's larger SD, condition 3's 0.020; the world's seed-42 cross-checks
  S  the condition states and the verdict on hand values (AM-19 item 6(b)-(e); D-7)
  L  the cell labels (IF§11.2's map; AM-19 item 6(e); D-6; D-15), from world variants
  W  the world end to end: ten labelled rows, robustness at every seed, ignored slots never refused,
     cut-run VAL curves (iterations for KD, epochs for QAT, the cut entry's comparator, never
     interpolated), divergences and the verdict sentence
  R  refusals: one case per code, each reached with every earlier gate passing (R-15 is an assertion),
     and one per further raise site the DL-24 review added (R2h-R2j, R4c-R4d, R12b-R12e, R13b, R14b,
     R17b, R18b)
  D  determinism
  X  hygiene: no test or p-value object imported, repr floats, pp only in markdown, no 'test' path

Cases are importable (setup / cases) so that scripts/smoke_lrep_mutations.py can run them mutated.

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_seed_stability.py
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import lrep_fixtures as LF                                               # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import report as RP                                       # noqa: E402
from src.stats import seed_stability as SS                               # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, paths={}, tel={}, copies={})
CUT_RULE = "AM-19 item 3(b): original freeze"
CUT_DATE = "2026-11-02"
E3_CUT_ITERS = tuple(range(4000, 40001, 4000))


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def stats_artifact() -> H.StatsArtifact:
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def inventory(path, *, policy=None, reader=None, expect=None) -> H.Inventory:
    return H.open_inventory(path, policy=policy or stats_artifact().policy, reader=reader,
                            expect_sha256=expect)


def seed_doc(path=None, **kw) -> dict:
    return SS.build_seed_stability(stats_artifact(), inventory(path or CTX.world.inventory_path, **kw),
                                   input_root=CTX.world.input_root)


def entry(export, slot):
    hits = [e for e in export["entries"] if e["slot"] == slot]
    if len(hits) != 1:
        raise KeyError(slot)
    return hits[0]


def set_kind(export, slot, kind, **fields):
    """Set a slot's kind and its record fields; a run that never ran lists no telemetry (IF§11.3)."""
    e = entry(export, slot)
    e["kind"] = kind
    e.update(fields)
    if kind in ("not_run", "not_made"):
        e.pop("telemetry", None)


def drop(export, binding, slot):
    export["models"] = [r for r in export["models"] if r["slot"] != slot]
    binding["artifacts"] = [r for r in binding["artifacts"] if r["slot"] != slot]


def record(binding, slot, precision, scope="clean", cell=None):
    hits = [r for r in binding["artifacts"]
            if (r["slot"], r["precision"], r["scope"], r.get("cell")) == (slot, precision, scope, cell)]
    if len(hits) != 1:
        raise KeyError((slot, precision, scope, cell))
    return hits[0]


def row(export, slot, precision):
    hits = [r for r in export["models"] if (r["slot"], r["precision"]) == (slot, precision)]
    if len(hits) != 1:
        raise KeyError((slot, precision))
    return hits[0]


def variant(name, edit, *, with_binding=True) -> Path:
    """The base export and binding edited by `edit(export, binding)`, written once into a new case
    directory and cached by name."""
    if name not in CTX.paths:
        export, binding = copy.deepcopy(CTX.world.export), copy.deepcopy(CTX.world.binding)
        edit(export, binding)
        CTX.paths[name] = LF.write_inventory(CTX.world.fresh(name), export,
                                             binding if with_binding else None)
    return CTX.paths[name]


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:300]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def cell(doc, cid, seed):
    return next(r for r in doc["rows"] if r["comparison_id"] == cid)["cells"][str(seed)]


def ns(**kw):
    return types.SimpleNamespace(**kw)


def totals(tp, gt, pred):
    return ns(dataset_tp=np.array(tp, np.int64), dataset_gt=np.array(gt, np.int64),
              dataset_pred=np.array(pred, np.int64))


def standin_reader(path, *, expect_sha256=None):
    """A reader that is not the stub (TMAN's stands here only to reach the OFFICIAL gates)."""
    return H.read_inventory(path, expect_sha256=expect_sha256)


# --------------------------------------------------------------------------------------------------
# world variants
# --------------------------------------------------------------------------------------------------
def v_cut_e2(export, binding):
    set_kind(export, "E2:s43", "cut", cut_date=CUT_DATE, rule=CUT_RULE, comparator="E1:s43")
    entry(export, "E2:s43")["telemetry"] = [CTX.tel["e2_s43_cut"]]
    drop(export, binding, "E2:s43")


def v_cut_e6(export, binding):
    set_kind(export, "E6:s43", "cut", cut_date=CUT_DATE, rule=CUT_RULE, comparator=None)
    entry(export, "E6:s43")["telemetry"] = [CTX.tel["e6_s43_cut"]]
    set_kind(export, "E6:s43:sel", None)
    drop(export, binding, "E6:s43:sel")


def v_div_e3(export, binding):
    set_kind(export, "E3:s44", "diverged", iteration=41000, rule="AM-7a item 1")
    drop(export, binding, "E3:s44")
    set_kind(export, "E7:s44", None)
    drop(export, binding, "E7:s44")
    set_kind(export, "E6:s44", "not_run", reason="parent diverged")
    set_kind(export, "E6:s44:sel", None)
    drop(export, binding, "E6:s44:sel")


def v_nonfinite_e6(export, binding):
    set_kind(export, "E6:s43", "non_finite")
    set_kind(export, "E6:s43:sel", None)
    drop(export, binding, "E6:s43:sel")


def v_div_e2(export, binding):
    set_kind(export, "E2:s43", "diverged", iteration=30000, rule="AM-7a item 1")
    drop(export, binding, "E2:s43")


def v_labels(export, binding):
    """An aborted E2 at 44, a not-made E7 at 44, a cut E3 at 43 with both descendants not run, and a
    QAT run diverged at an epoch (E5 at 44)."""
    set_kind(export, "E2:s44", "aborted", rule="AM-19 item 3(a)", cause="non-finite VAL score")
    drop(export, binding, "E2:s44")
    set_kind(export, "E7:s44", "not_made", reason="code not merged")
    drop(export, binding, "E7:s44")
    set_kind(export, "E3:s43", "cut", cut_date=CUT_DATE, rule=CUT_RULE, comparator=None)
    drop(export, binding, "E3:s43")
    set_kind(export, "E7:s43", None)
    drop(export, binding, "E7:s43")
    set_kind(export, "E6:s43", "not_run", reason="parent cut")
    set_kind(export, "E6:s43:sel", None)
    drop(export, binding, "E6:s43:sel")
    set_kind(export, "E5:s44", "diverged", epoch=7, rule="AM-21 item 1")
    set_kind(export, "E5:s44:sel", None)
    drop(export, binding, "E5:s44:sel")


def v_ghosts(export, binding):
    """Every slot no seed-stability table reads gets a kind LREP would refuse on a planned cell."""
    for slot in ("E2:s42:lambda0.25", "U4:pilot", "E2:s42:160k", "E4:s42:cal43", "E3-R1:s42", "R1",
                 "teacher", "E6-E2:s42:sel", "E5:s42", "E6-KD:s42:sel", "E2:s42:lambda1"):
        set_kind(export, slot, "superseded")


def v_swap_e1(export, binding):
    a, b = "E1:s43", "E1:s44"
    for r in binding["artifacts"]:
        if r["slot"] in (a, b):
            r["slot"] = b if r["slot"] == a else a
    ra, rb = row(export, a, "fp32"), row(export, b, "fp32")
    ra["source"]["sha256"], rb["source"]["sha256"] = rb["source"]["sha256"], ra["source"]["sha256"]


# --------------------------------------------------------------------------------------------------
# F: formulas
# --------------------------------------------------------------------------------------------------
def f_m_hand():
    a = SS.m_of(totals([1, 1, 0], [2, 1, 0], [1, 1, 0]))         # IoU 1/2 and 1, class 2 absent
    b = SS.m_of(totals([1, 0], [2, 1], [1, 1]))                   # IoU 1/2 and 0
    c = SS.m_of(totals([2, 0], [2, 0], [2, 5]))                   # a prediction-only class counts, 0
    return (a, b, c) == (0.75, 0.25, 0.5), f"{a!r} {b!r} {c!r}"


def f_effect_sign():
    """The table's clean cell (seed_stability._cell) is m(candidate) - m(baseline) (STATS §1)."""
    base, cand = totals([1, 0], [2, 1], [1, 1]), totals([1, 1, 0], [2, 1, 0], [1, 1, 0])
    c = SS._cell({("E1", 43): SS.m_of(base), ("E2", 43): SS.m_of(cand)}, {}, "E1", "E2", 43)
    return c["value"] == 0.5 and c["missing"] == [], str(c)


def f_ni_direction():
    """The table's NI cell (E3 -> E6) is the negative of the drop d = m(E3) - m(E6), and condition 3
    reads d."""
    c = SS._cell({("E3", 43): 0.75, ("E6", 43): 0.5}, {}, "E3", "E6", 43)
    c3 = SS.condition_3({42: 0.0, 43: 0.75 - 0.5, 44: 0.0})
    return c["value"] == -0.25 and c3.values[43] == 0.25 == -c["value"] and c3.failing_seeds == (43,), \
        f"{c} {c3.values}"


def f_miou_c_hand():
    """Nested mC on cells whose unions differ by severity, with a prediction-only class at severity 3:
    the mean of the per-corruption means of the cells' m is 37/96 (0.38541666666666663 in float64),
    while pooling the totals over the severities gives 0.19642857142857145 (R5-03). A flat mean of
    the 15 cells is equal in exact arithmetic; F5 checks report.dataset_miou_c bitwise."""
    names = H.corruption_names()
    cells = {}
    for i, n in enumerate(names):
        for s in (1, 2, 3):
            g = 2 ** (s + 1)
            tp = (i + s) % (g + 1)
            cells[(n, s)] = totals([tp, 0], [g, 0], [tp, 5 if s == 3 else 0])
    got = SS.miou_c(cells, names)
    return got == 0.38541666666666663, repr(got)


def f_miou_c_report():
    """mC over an explicit cell map equals report.dataset_miou_c on the same cells, bitwise."""
    names = H.corruption_names()
    rng = np.random.default_rng(7)
    loaded, cells = {}, {}
    for n in names:
        for s in (1, 2, 3):
            tp = rng.integers(0, 50, 9)
            st = totals(tp, tp + rng.integers(0, 50, 9), tp + rng.integers(0, 50, 9))
            cells[(n, s)] = st
            loaded[("E1", ("corruption", n, s))] = ns(run=ns(stats=st))
    a, b = SS.miou_c(cells, names), RP.dataset_miou_c(ns(loaded=loaded), "E1")
    return a == b, f"{a!r} vs {b!r}"


def f_g_and_d():
    doc = seed_doc()
    m = {(r["stage"], r["seed"]): r["m"] for r in doc["models"]}
    g = {k: m[("E3", k)] - m[("E1", k)] for k in SS.SEEDS}
    d = {k: m[("E3", k)] - m[("E6", k)] for k in SS.SEEDS}
    c1, c3 = doc["conditions"]["1"]["values"], doc["conditions"]["3"]["values"]
    return (all(c1[str(k)] == g[k] for k in SS.SEEDS) and all(c3[str(k)] == d[k] for k in SS.SEEDS)
            and all(cell(doc, D.DESCRIPTIVE_E1_E3, k)["value"] == g[k] for k in SS.SEEDS)
            and all(cell(doc, D.NONINFERIORITY_E3_E6, k)["value"] == -d[k] for k in SS.SEEDS)), ""


def f_fsum_mean():
    mean, _, _ = SS.mean_and_sds([0.0, 0.0, 0.0], [0.1, 0.2, 0.3])
    naive = (0.1 + 0.2 + 0.3) / 3
    return mean == math.fsum([0.1, 0.2, 0.3]) / 3 and mean != naive, f"{mean!r} vs naive {naive!r}"


def f_stdev_exact():
    _, s1, s3 = SS.mean_and_sds([0.5, 0.5, 0.5], [0.5, 0.75, 1.0])
    return (s1, s3) == (0.0, 0.25), f"{s1!r} {s3!r}"


def f_stdev_sample():
    _, s, _ = SS.mean_and_sds([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    return s == statistics.stdev([1.0, 2.0, 3.0]) == 1.0 and s != statistics.pstdev([1.0, 2.0, 3.0]), \
        repr(s)


def f_c2_larger_sd():
    """Exact binary fractions: SD_E1 0.5 > mean gain 0.375 > SD_E3 0.125, so condition 2 fails against
    the larger SD and would hold against the smaller (R4-10)."""
    c2 = SS.condition_2({42: 0.0, 43: 0.5, 44: 1.0}, {42: 0.75, 43: 0.875, 44: 1.0})
    d = c2.detail
    return (c2.status == SS.FAILS and (d["sd_e1"], d["mean_gain"], d["sd_e3"]) == (0.5, 0.375, 0.125)
            and d["max_sd"] == 0.5), str(d)


def f_c3_fraction():
    c3 = SS.condition_3({42: 0.5, 43: 0.5, 44: 0.5})               # would hold against 2.0
    return c3.status == SS.FAILS and SS.DROP_THRESHOLD == 0.020, c3.status


def f_seed42_world():
    doc = seed_doc()
    want = {f"{cid} effect" for cid in D.CLEAN_COMPARISON_IDS} | {
        "m(E3,42)", "m(E6,42)", "noninferiority_e3_e6 effect", "d_42", "d_42 (E6-KD block)",
        "mC(E1,42)", "mC(E6,42)"}
    got = set(doc["seed42_cross_checks"]["checked"])
    return got == want and doc["seed42_cross_checks"]["bitwise_equal"] is True, sorted(want ^ got)


def f_robust42_report():
    doc = seed_doc()
    con = RP.dataset_miou_c_contrast(stats_artifact().views)
    return cell(doc, D.ROBUSTNESS_ID, 42)["value"] == con["difference_e6_minus_e1"], ""


# --------------------------------------------------------------------------------------------------
# S: states and the verdict (hand values)
# --------------------------------------------------------------------------------------------------
E1V, E3V = {42: 0.30, 43: 0.31, 44: 0.29}, {42: 0.40, 43: 0.42, 44: 0.41}


def vd(c1, c2, c3, e3=(), missing=(), e2=()):
    return SS.verdict(c1, c2, c3, e3_divergences=list(e3), missing=list(missing),
                      e2_divergences=list(e2))


def holds3():
    return (SS.condition_1({42: 0.1, 43: 0.11, 44: 0.12}), SS.condition_2(E1V, E3V),
            SS.condition_3({42: 0.01, 43: 0.005, 44: 0.0}))


def s_all_hold():
    c1, c2, c3 = holds3()
    v = vd(c1, c2, c3)
    return (c1.status, c2.status, c3.status, v["verdict"], v["reasons"]) == (
        SS.HOLDS, SS.HOLDS, SS.HOLDS, SS.STABLE, []), str(v)


def s_c1_zero():
    c = SS.condition_1({42: 0.1, 43: 0.0, 44: 0.1})
    return c.status == SS.FAILS and c.failing_seeds == (43,), str(c)


def s_c1_negative():
    c = SS.condition_1({42: 0.1, 43: 0.1, 44: -1e-12})
    return c.status == SS.FAILS and c.failing_seeds == (44,), str(c)


def s_c1_fails_missing():
    c = SS.condition_1({42: 0.1, 44: -0.1})
    return c.status == SS.FAILS and c.failing_seeds == (44,), str(c)


def s_c1_not_evaluable():
    c = SS.condition_1({42: 0.1, 44: 0.1})
    return c.status == SS.NOT_EVALUABLE, str(c)


def s_c2_holds():
    c = SS.condition_2(E1V, E3V)
    return c.status == SS.HOLDS and c.detail["mean_gain"] > c.detail["max_sd"], str(c.detail)


def s_c2_equal_fails():
    c = SS.condition_2({42: 0.5, 43: 0.5, 44: 0.5}, {42: 0.5, 43: 0.75, 44: 1.0})
    return (c.status == SS.FAILS and c.detail["mean_gain"] == c.detail["max_sd"] == 0.25), str(c.detail)


def s_c2_two_seeds():
    c = SS.condition_2({42: 0.3, 44: 0.3}, E3V)
    return c.status == SS.NOT_EVALUABLE and c.detail["mean_gain"] is None, str(c)


def s_c3_at_threshold():
    c = SS.condition_3({42: 0.01, 43: 0.020, 44: 0.01})
    return c.status == SS.FAILS and c.failing_seeds == (43,), str(c)


def s_c3_below():
    c = SS.condition_3({k: math.nextafter(0.020, 0.0) for k in SS.SEEDS})
    return c.status == SS.HOLDS, str(c)


def s_c3_fails_missing():
    c = SS.condition_3({42: 0.025, 43: 0.01})
    return c.status == SS.FAILS and c.failing_seeds == (42,), str(c)


def s_c3_not_evaluable():
    c = SS.condition_3({42: 0.01, 43: 0.01})
    return c.status == SS.NOT_EVALUABLE, str(c)


def s_nonfinite_reason():
    doc = seed_doc(variant("nonfinite_e6", v_nonfinite_e6))
    want = f"not evaluable: runs missing — E6 seed 43: {H.LABEL_NON_FINITE}"
    return (doc["conditions"]["3"]["status"] == SS.NOT_EVALUABLE and doc["verdict"] == SS.NOT_STABLE
            and doc["reasons"] == [want]), str(doc["reasons"])


def s_e3_override():
    c1, c2, c3 = holds3()
    v = vd(c1, c2, c3, e3=[SS.Divergence("E3", 44, "iteration", 41000, "AM-7a item 1")])
    return v["verdict"] == SS.NOT_STABLE and v["reasons"] == ["E3 diverged at seed 44"], str(v)


def s_divergence_first():
    c1 = SS.condition_1({42: 0.1, 43: -0.1})
    v = vd(c1, SS.condition_2({42: 0.3}, E3V), SS.condition_3({42: 0.03, 43: 0.0}),
           e3=[SS.Divergence("E3", 44, "iteration", 41000, "AM-7a item 1")],
           missing=[SS.Missing("E3", 44, "diverged (iteration 41000, AM-7a item 1)")])
    return v["reasons"][0] == "E3 diverged at seed 44", str(v["reasons"])


def s_reason_order():
    """D-7's fixed order, in the two combinations AM-19 item 6(b) allows: (a) a divergence, failing
    conditions 1 and 3 and a not-evaluable condition 2 naming its missing run (a run missing for a
    condition that already fails is no reason); (b) conditions 1, 2 and 3 failing, in that order."""
    div = SS.Divergence("E3", 44, "iteration", 41000, "AM-7a item 1")
    a = vd(SS.condition_1({42: 0.0, 43: 0.1}), SS.condition_2(E1V, {42: 0.4, 43: 0.42}),
           SS.condition_3({42: 0.03, 43: 0.0}), e3=[div],
           missing=[SS.Missing("E3", 44, "diverged (iteration 41000, AM-7a item 1)"),
                    SS.Missing("E6", 44, H.LABEL_PARENT_DIVERGED)])
    want_a = ["E3 diverged at seed 44", "condition 1 fails at seed 42", "condition 3 fails at seed 42",
              "not evaluable: runs missing — E3 seed 44: diverged (iteration 41000, AM-7a item 1)"]
    b = vd(SS.condition_1({42: 0.0, 43: -0.1, 44: 0.1}),
           SS.condition_2({42: 0.5, 43: 0.5, 44: 0.5}, {42: 0.5, 43: 0.75, 44: 1.0}),
           SS.condition_3({42: 0.03, 43: 0.01, 44: 0.0}))
    want_b = ["condition 1 fails at seed 42", "condition 1 fails at seed 43", SS.C2_FAILS,
              "condition 3 fails at seed 42"]
    return a["reasons"] == want_a and b["reasons"] == want_b, f"{a['reasons']} | {b['reasons']}"


def s_e2_sentence():
    c1, c2, c3 = holds3()
    v = vd(c1, c2, c3, e2=[SS.Divergence("E2", 43, "iteration", 30000, "AM-7a item 1")])
    return (v["verdict"] == SS.STABLE and v["verdict_sentence"] ==
            "Seed stability (AM-8; AM-19 item 6): stable; E2 diverged at seed 43 (iteration 30000, "
            "AM-7a item 1)."), v["verdict_sentence"]


def s_missing_never_stable():
    c1, c2, _ = holds3()
    v = vd(c1, c2, SS.condition_3({42: 0.0, 43: 0.0}), missing=[SS.Missing("E6", 44, "x")])
    return v["verdict"] == SS.NOT_STABLE, str(v)


def s_values_reported():
    c = SS.condition_1({42: 0.1, 44: 0.3})
    j = c.as_json()
    return j["values"] == {"42": 0.1, "44": 0.3} and j["status"] == SS.NOT_EVALUABLE, str(j)


def s_c2_no_seed():
    c2 = SS.condition_2({42: 0.5, 43: 0.5, 44: 0.5}, {42: 0.5, 43: 0.75, 44: 1.0})
    v = vd(SS.condition_1({42: 0.1, 43: 0.1, 44: 0.1}), c2, SS.condition_3({k: 0.0 for k in SS.SEEDS}))
    return v["reasons"] == [SS.C2_FAILS] and "seed" not in SS.C2_FAILS, str(v["reasons"])


def s_reason_entry():
    """D-6: the reason line names the run and its entry: the decision-log rows of the entry that
    decided it."""
    def edit(export, binding):
        v_nonfinite_e6(export, binding)
        entry(export, "E6:s43")["dl_rows"] = ["DL-140"]
    doc = seed_doc(variant("nonfinite_dl", edit))
    want = f"not evaluable: runs missing — E6 seed 43: {H.LABEL_NON_FINITE} [DL-140]"
    return doc["reasons"] == [want], str(doc["reasons"])


# --------------------------------------------------------------------------------------------------
# L: labels (IF§11.2; AM-19 item 6(e); D-6; D-15)
# --------------------------------------------------------------------------------------------------
def l_cut():
    c = cell(seed_doc(variant("cut_e2", v_cut_e2)), "accuracy_e1_e2", 43)
    return c["missing"] == [f"E2: cut ({CUT_DATE}, {CUT_RULE})"], str(c)


def l_diverged():
    c = cell(seed_doc(variant("div_e2", v_div_e2)), "accuracy_e2_e3", 43)
    return c["missing"] == ["E2: diverged (iteration 30000, AM-7a item 1)"], str(c)


def l_aborted():
    c = cell(seed_doc(variant("labels", v_labels)), "accuracy_e1_e2", 44)
    return c["missing"] == ["E2: aborted (AM-19 item 3(a), non-finite VAL score)"], str(c)


def l_parent_cut():
    doc = seed_doc(variant("labels", v_labels))
    a, b = cell(doc, "accuracy_e4_e7", 43), cell(doc, "accuracy_e7_e6", 43)
    return (a["missing"] == [f"E7: {H.LABEL_PARENT_CUT}"]
            and b["missing"] == [f"E7: {H.LABEL_PARENT_CUT}", f"E6: {H.LABEL_PARENT_CUT}"]), f"{a} {b}"


def l_parent_diverged():
    doc = seed_doc(variant("div_e3", v_div_e3))
    a, b = cell(doc, "accuracy_e4_e7", 44), cell(doc, "accuracy_e5_e6", 44)
    return (a["missing"] == [f"E7: {H.LABEL_PARENT_DIVERGED}"]
            and b["missing"] == [f"E6: {H.LABEL_PARENT_DIVERGED}"]), f"{a} {b}"


def l_nonfinite():
    c = cell(seed_doc(variant("nonfinite_e6", v_nonfinite_e6)), "accuracy_e1_e6", 43)
    return c["missing"] == [f"E6: {H.LABEL_NON_FINITE}"], str(c)


def l_two_missing():
    c = cell(seed_doc(variant("labels", v_labels)), "noninferiority_e3_e6", 43)
    return c["missing"] == [f"E3: cut ({CUT_DATE}, {CUT_RULE})", f"E6: {H.LABEL_PARENT_CUT}"], str(c)


def l_date():
    def bad_date(export, binding):
        set_kind(export, "E2:s43", "cut", cut_date="2026-11-2", rule=CUT_RULE, comparator=None)
        drop(export, binding, "E2:s43")
    ok_label = cell(seed_doc(variant("cut_e2", v_cut_e2)), "accuracy_e2_e3", 43)["missing"][0]
    ok, det = refused("status_invalid", lambda: seed_doc(variant("bad_date", bad_date)))
    return ok and f"({CUT_DATE}," in ok_label, det


def l_stage_prefix():
    doc = seed_doc(variant("labels", v_labels))
    cells = [c for r in doc["rows"] for c in r["cells"].values()]
    texts = [t for c in cells for t in c["missing"]]
    return bool(texts) and all(t.split(":", 1)[0] in D.STAGES for t in texts), str(texts[:4])


def l_not_made():
    c = cell(seed_doc(variant("labels", v_labels)), "accuracy_e4_e7", 44)
    return c["missing"] == ["E7: not made: code not merged"], str(c)


def l_epoch():
    c = cell(seed_doc(variant("labels", v_labels)), "accuracy_e4_e5", 44)
    return c["missing"] == ["E5: diverged (epoch 7, AM-21 item 1)"], str(c)


# --------------------------------------------------------------------------------------------------
# W: the world end to end
# --------------------------------------------------------------------------------------------------
def w_base():
    doc = seed_doc()
    ids = [r["comparison_id"] for r in doc["rows"]]
    labels = [r["label"] for r in doc["rows"]]
    return (ids == list(D.COMPARISON_TABLE) and labels == ["family"] * 8 + ["descriptive",
                                                                          "non-inferiority"]
            and doc["verdict"] == SS.STABLE and doc["reasons"] == [] and doc["cut_runs"] == []
            and all(c["value"] is not None for r in doc["rows"] for c in r["cells"].values())), \
        f"{doc['verdict_sentence']} {labels}"


def w_robustness_seeds():
    doc = seed_doc()
    vals = [cell(doc, D.ROBUSTNESS_ID, k)["value"] for k in SS.SEEDS]
    mc = {(r["stage"], r["seed"]): r["miou_c"] for r in doc["models"]}
    return (all(isinstance(v, float) for v in vals)
            and all(vals[i] == mc[("E6", k)] - mc[("E1", k)] for i, k in enumerate(SS.SEEDS))), str(vals)


def w_ghosts_ignored():
    doc = seed_doc(variant("ghosts", v_ghosts))
    return doc["verdict"] == SS.STABLE, doc["verdict_sentence"]


def w_cut_kd_curve():
    doc = seed_doc(variant("cut_e2", v_cut_e2))
    b = doc["cut_runs"]
    if len(b) != 1:
        return False, str(b)
    b = b[0]
    comps = {c["stage"]: c for c in b["comparators"]}
    pts = [p["iter"] for p in b["attempts"][0]["points"]]
    e1 = comps.get("E1", {})
    vals = {v["iter"]: v["all_class_miou"] for v in e1.get("values", [])}
    tel = CTX.world.telemetry["E1:s43"][0]
    want = dict(LF.kd_curve(0.36 + 0.001))
    return (b["stage"] == "E2" and b["unit"] == "iter" and sorted(comps) == ["E1", "E3"]
            and e1["cut_entry_comparator"] is True and comps["E3"]["cut_entry_comparator"] is False
            and pts[-1] == 50000 and vals[50000] is None and e1["shared_points"] == len(pts) - 1
            and all(vals[u] == want[u] for u in pts[:-1]) and e1["run_id"] == tel["run_id"]
            and doc["verdict"] == SS.STABLE), str(b)[:400]


def w_cut_qat_curve():
    doc = seed_doc(variant("cut_e6", v_cut_e6))
    b = [x for x in doc["cut_runs"] if x["stage"] == "E6"]
    if len(b) != 1:
        return False, str(doc["cut_runs"])
    b = b[0]
    c = b["comparators"]
    return (b["unit"] == "epoch" and b["cut_slot"] == "E6:s43" and len(c) == 1
            and (c[0]["stage"], c[0]["seed"], c[0]["run_slot"]) == ("E6", 42, "E6:s42")
            and c[0]["shared_points"] == 9 and doc["conditions"]["3"]["status"] == SS.NOT_EVALUABLE
            and cell(doc, D.ROBUSTNESS_ID, 43)["missing"] == [f"E6: cut ({CUT_DATE}, {CUT_RULE})"]), \
        str(b)[:400]


def w_div_e3():
    doc = seed_doc(variant("div_e3", v_div_e3))
    r = doc["reasons"]
    want_missing = ("not evaluable: runs missing — E3 seed 44: diverged (iteration 41000, AM-7a item "
                    f"1); E6 seed 44: {H.LABEL_PARENT_DIVERGED}")
    d = doc["divergences_43_44"]
    return (doc["verdict"] == SS.NOT_STABLE and r == ["E3 diverged at seed 44", want_missing]
            and [(x["stage"], x["seed"]) for x in d] == [("E3", 44)] and d[0]["limitation"]
            and doc["limitation_divergences"] == ["E3 diverged at seed 44 (iteration 41000, AM-7a item 1)"]
            and cell(doc, D.ROBUSTNESS_ID, 44)["missing"] == [f"E6: {H.LABEL_PARENT_DIVERGED}"]), str(r)


def w_div_e2_sentence():
    doc = seed_doc(variant("div_e2", v_div_e2))
    return (doc["verdict"] == SS.STABLE and doc["verdict_sentence"].endswith(
        "; E2 diverged at seed 43 (iteration 30000, AM-7a item 1).")), doc["verdict_sentence"]


def w_cut_e3_comparators():
    doc = seed_doc(variant("labels", v_labels))
    b = [x for x in doc["cut_runs"] if x["stage"] == "E3"][0]
    comps = {c["stage"]: c for c in b["comparators"]}
    return (sorted(comps) == ["E1", "E2", "E6"]
            and comps["E6"]["note"] == f"no comparator values: {H.LABEL_PARENT_CUT}"
            and comps["E2"]["shared_points"] == len(LF.KD_VAL_ITERS)), str(b)[:300]


def v_cut_e2_e3(export, binding):
    """E2 and E3 both cut at seed 43, each with its own VAL rows; E3's descendants not run."""
    v_cut_e2(export, binding)
    entry(export, "E2:s43")["comparator"] = None
    set_kind(export, "E3:s43", "cut", cut_date=CUT_DATE, rule=CUT_RULE, comparator=None)
    entry(export, "E3:s43")["telemetry"] = [CTX.tel["e3_s43_cut"]]
    drop(export, binding, "E3:s43")
    set_kind(export, "E7:s43", None)
    drop(export, binding, "E7:s43")
    set_kind(export, "E6:s43", "not_run", reason="parent cut")
    set_kind(export, "E6:s43:sel", None)
    drop(export, binding, "E6:s43:sel")


def w_cut_pair():
    """Two paired cut KD runs at one seed: each block shows the other's VAL rows at its own points
    (AM-19 item 3(c); R2-03); nothing is interpolated."""
    doc = seed_doc(variant("cut_e2_e3", v_cut_e2_e3))
    blocks = {b["stage"]: b for b in doc["cut_runs"] if b["seed"] == 43}
    if sorted(blocks) != ["E2", "E3"]:
        return False, str(doc["cut_runs"])[:300]
    e3_in_e2 = next(c for c in blocks["E2"]["comparators"] if c["stage"] == "E3")
    e2_in_e3 = next(c for c in blocks["E3"]["comparators"] if c["stage"] == "E2")
    want = dict(LF.kd_curve(0.38, iters=E3_CUT_ITERS))
    a = e3_in_e2["attempts"]
    vals = {v["iter"]: v["all_class_miou"] for v in a[0]["values"]} if len(a) == 1 else {}
    md = "\n".join(SS.render_markdown(doc))
    return (len(a) == 1 and a[0]["shared_points"] == len(E3_CUT_ITERS)
            and all(vals[u] == want[u] for u in E3_CUT_ITERS)
            and all(vals[u] is None for u in vals if u not in want)
            and e3_in_e2["note"].startswith("no model (cut (")
            and len(e2_in_e3["attempts"]) == 1 and e2_in_e3["attempts"][0]["shared_points"] == len(E3_CUT_ITERS)
            and f"attempt `{a[0]['run_id']}`" in md), str(e3_in_e2)[:300]


def v_cut_e2_odd(export, binding):
    v_cut_e2(export, binding)
    entry(export, "E2:s43")["comparator"] = None
    entry(export, "E2:s43")["telemetry"] = [CTX.tel["e2_s43_odd"]]


def w_no_shared_point():
    """A comparator curve that shares no point with the cut run is stated as such in the markdown, beside
    its empty values (LRB6; R2-02)."""
    doc = seed_doc(variant("cut_e2_odd", v_cut_e2_odd))
    b = next(x for x in doc["cut_runs"] if x["stage"] == "E2")
    e1 = next(c for c in b["comparators"] if c["stage"] == "E1")
    note = "no VAL point shared with the cut run (never interpolated)"
    line = next((ln for ln in SS.render_markdown(doc) if ln.startswith("  - comparator E1 seed 43")), "")
    return (e1["shared_points"] == 0 and e1["note"] == note and "2000: —" in line and note in line), line


# --------------------------------------------------------------------------------------------------
# R: refusals, one case per code, every earlier gate passing
# --------------------------------------------------------------------------------------------------
def r_stats_unverified():
    empty = CTX.world.fresh("no_inputs")
    return refused("stats_artifact_unverified", lambda: H.load_stats_artifact(CTX.world.stats_path, empty))


def r_stub_official():
    return refused("stub_reader_official",
                   lambda: H.open_inventory(CTX.world.inventory_path, policy=Policy.OFFICIAL))


def r_inventory_unreadable():
    return refused("inventory_unreadable",
                   lambda: inventory(CTX.world.fresh("missing_export") / LF.EXPORT_NAME))


def r_inventory_sha():
    return refused("inventory_sha256_mismatch",
                   lambda: inventory(CTX.world.inventory_path, expect="0" * 64))


def r_inventory_invalid():
    def edit(export, binding):
        export["schema"] = "tday-inventory/0"
    return refused("inventory_invalid", lambda: inventory(variant("bad_schema", edit)))


def r_binding_invalid():
    def edit(export, binding):
        export["binding"]["sha256"] = "0" * 64
    return refused("binding_invalid", lambda: inventory(variant("bad_binding_sha", edit)))


def r_binding_unbound():
    def edit(export, binding):
        export["binding"] = {"state": "unbound", "path": None, "sha256": None}
    p = variant("unbound", edit, with_binding=False)
    return refused("binding_unbound", lambda: inventory(p, policy=Policy.OFFICIAL, reader=standin_reader))


def r_freeze():
    def edit(export, binding):
        export["freeze"]["manifest_sha256"] = LF.syn_sha("another manifest")
    return refused("freeze_mismatch", lambda: inventory(variant("freeze", edit)))


def r_slot_missing():
    def edit(export, binding):
        entry(export, "E2:s43")["role"] = []
    return refused("planned_slot_missing", lambda: seed_doc(variant("slot_missing", edit)))


def r_slot_duplicate():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "E2:s43"))
        e["slot"] = "E2:s43:b"
        export["entries"].append(e)
    return refused("planned_slot_duplicate", lambda: seed_doc(variant("slot_dup", edit)))


def r_unplanned():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "E2:s43"))
        e.update(slot="E2:s45", seed=45)
        export["entries"].append(e)
    return refused("unplanned_slot", lambda: seed_doc(variant("unplanned", edit)))


def r_status_vocab():
    def edit(export, binding):
        set_kind(export, "E2:s43", "complete")
    return refused("status_invalid", lambda: seed_doc(variant("vocab", edit)))


def r_status_field():
    def edit(export, binding):
        set_kind(export, "E2:s43", "cut", cut_date=CUT_DATE, comparator=None)
    return refused("status_invalid", lambda: seed_doc(variant("no_rule", edit)))


def r_superseded():
    def edit(export, binding):
        set_kind(export, "E2:s43", "superseded")
    return refused("superseded", lambda: seed_doc(variant("superseded", edit)))


def r_artifact_missing():
    def edit(export, binding):
        binding["artifacts"].remove(record(binding, "E2:s43", "fp32"))
    return refused("artifact_missing", lambda: seed_doc(variant("no_record", edit)))


def r_artifact_for_missing():
    def edit(export, binding):
        set_kind(export, "E2:s43", "cut", cut_date=CUT_DATE, rule=CUT_RULE, comparator=None)
    return refused("artifact_for_missing_run", lambda: seed_doc(variant("cut_kept_record", edit)))


def r_record_files():
    def edit(export, binding):
        record(binding, "E2:s43", "fp32")["files_sha256"]["summary.json"] = "0" * 64
    return refused("artifact_record_mismatch", lambda: seed_doc(variant("rec_files", edit)))


def r_record_protocol():
    up = CTX.copies["upstream"]

    def edit(export, binding):
        rec = record(binding, "E2:s43", "fp32")
        rec["repo_relative_path"], rec["files_sha256"] = up.rel, dict(up.files_sha256)
    return refused("artifact_record_mismatch", lambda: seed_doc(variant("rec_protocol", edit)))


def r_record_identity():
    def edit(export, binding):
        row(export, "E2:s43", "fp32")["source"]["sha256"] = LF.syn_sha("another checkpoint")
    return refused("artifact_record_mismatch", lambda: seed_doc(variant("row_identity", edit)))


def r_not_canvas():
    def edit(export, binding):
        record(binding, "E2:s43", "fp32")["protocol"] = "upstream"
    return refused("not_canvas", lambda: seed_doc(variant("rec_upstream", edit)))


def r_unreadable():
    def edit(export, binding):
        record(binding, "E2:s43", "fp32")["repo_relative_path"] = "runs/heldout/never_written_case"
    return refused("artifact_unreadable", lambda: seed_doc(variant("rec_nodir", edit)))


def r_cross():
    om = CTX.copies["other_metric"]

    def edit(export, binding):
        rec = record(binding, "E3:s43", "fp32")
        rec["repo_relative_path"], rec["files_sha256"] = om.rel, dict(om.files_sha256)
    return refused("cross_artifact_mismatch", lambda: seed_doc(variant("rec_metric", edit)))


def r_e1_record():
    return refused("e1_checkpoint_record", lambda: seed_doc(variant("swap_e1", v_swap_e1)))


def r_seed42():
    def edit(export, binding):
        row(export, "E2:s42", "fp32")["source"]["sha256"] = LF.syn_sha("another checkpoint")
    return refused("seed42_mismatch", lambda: seed_doc(variant("s42_identity", edit)))


def r_parent():
    def edit(export, binding):
        set_kind(export, "E6:s43", "not_run", reason="parent cut")
        set_kind(export, "E6:s43:sel", None)
        drop(export, binding, "E6:s43:sel")
    return refused("parent_rule", lambda: seed_doc(variant("parent_complete", edit)))


def r_seed42_complete():
    def edit(export, binding):
        set_kind(export, "E2:s42", "cut", cut_date="2026-10-19", rule="AM-19 item 2", comparator=None)
    return refused("seed42_not_complete", lambda: seed_doc(variant("s42_cut", edit)))


def r_seed_inconsistent():
    def edit(export, binding):
        row(export, "E2:s43", "fp32")["seed"] = 44
    return refused("seed_inconsistent", lambda: seed_doc(variant("row_seed", edit)))


def r_assert_two_seeds():
    try:
        SS.mean_and_sds([0.3, 0.31], [0.4, 0.42])
    except AssertionError as e:
        return "R-15" in str(e), str(e)
    return False, "no assertion"


def r_telemetry_sha():
    def edit(export, binding):
        v_cut_e2(export, binding)
        entry(export, "E2:s43")["telemetry"] = [dict(CTX.tel["e2_s43_cut"], sha256="0" * 64)]
    return refused("telemetry_sha256_mismatch", lambda: seed_doc(variant("tel_sha", edit)))


def r_telemetry_invalid():
    def edit(export, binding):
        v_cut_e2(export, binding)
        entry(export, "E1:s43")["telemetry"][0]["run_id"] = "e1_s43_80k_a9"
    return refused("telemetry_invalid", lambda: seed_doc(variant("tel_attempt", edit)))


def r_precision():
    def edit(export, binding):
        rec, fq = record(binding, "E5:s43:sel", "int8_qat"), record(binding, "E5:s43:sel",
                                                                     H.PRECISION_FQOFF, "fq_off")
        rec["repo_relative_path"], rec["files_sha256"] = fq["repo_relative_path"], dict(fq["files_sha256"])
    return refused("precision_mismatch", lambda: seed_doc(variant("fqoff_as_int8", edit)))


def r_path():
    def edit(export, binding):
        record(binding, "E2:s43", "fp32")["repo_relative_path"] = "runs/heldout/../heldout/e2_s43_clean"
    return refused("path_unsafe", lambda: seed_doc(variant("dotdot", edit)))


def r_robustness_cells():
    def edit(export, binding):
        binding["artifacts"].remove(record(binding, "E6:s43:sel", "int8_qat", "corrupted", "fog_s3"))
    return refused("robustness_cells_missing", lambda: seed_doc(variant("no_cell", edit)))


def r_binding_no_artifacts():
    """A reader (not the stub) that returns a bound export without binding.artifacts (IF§11.6)."""
    def reader(path, *, expect_sha256=None):
        doc = H.read_inventory(path, expect_sha256=expect_sha256)
        doc["binding"] = {k: v for k, v in doc["binding"].items() if k != "artifacts"}
        return doc
    return refused("binding_invalid", lambda: inventory(CTX.world.inventory_path, reader=reader))


def r_row_entry_kind():
    def edit(export, binding):
        row(export, "E2:s43", "fp32")["entry_kind"] = "cut"
    return refused("inventory_invalid", lambda: seed_doc(variant("row_entry_kind", edit)))


def r_duplicate_key():
    """The export's text repeats a key; the strict parse refuses it (a plain parse would keep the
    later, valid value)."""
    d = CTX.world.fresh("dup_key")
    src = CTX.world.inventory_path
    (d / LF.BINDING_NAME).write_bytes((src.parent / LF.BINDING_NAME).read_bytes())
    p = d / LF.EXPORT_NAME
    p.write_text(src.read_text(encoding="utf-8").replace("{", '{"schema": "tday-inventory/0", ', 1),
                 encoding="utf-8", newline="\n")
    return refused("inventory_unreadable", lambda: inventory(p))


def r_two_entries():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "E2:s43"))
        e.update(kind="cut", cut_date=CUT_DATE, rule=CUT_RULE, comparator=None)
        export["entries"].append(e)
    return refused("status_invalid", lambda: seed_doc(variant("two_entries", edit)))


def r_divergence_unit():
    """E3 at seed 44 diverged, recorded at an epoch: a KD run diverges at an iteration (IF§11.2)."""
    def edit(export, binding):
        v_div_e3(export, binding)
        e = entry(export, "E3:s44")
        del e["iteration"]
        e["epoch"] = 9
    return refused("status_invalid", lambda: seed_doc(variant("div_unit", edit)))


def r_parent_run():
    def edit(export, binding):
        entry(export, "E4:s43")["catalogue"]["parent"] = "E2:s43"
    return refused("parent_rule", lambda: seed_doc(variant("parent_e2", edit)))


def r_parent_selection():
    def edit(export, binding):
        set_kind(export, "E6:s43:sel", None)
        entry(export, "E6:s43:sel")["catalogue"]["parent"] = "E3:s43"
        drop(export, binding, "E6:s43:sel")
    return refused("parent_rule", lambda: seed_doc(variant("sel_parent_e3", edit)))


def r_parent_not_run():
    def edit(export, binding):
        set_kind(export, "E3:s43", "cut", cut_date=CUT_DATE, rule=CUT_RULE, comparator=None)
        drop(export, binding, "E3:s43")
        set_kind(export, "E7:s43", None)
        drop(export, binding, "E7:s43")
        set_kind(export, "E6:s43", "aborted", rule="AM-19 item 3(a)", cause="non-finite loss")
        set_kind(export, "E6:s43:sel", None)
        drop(export, binding, "E6:s43:sel")
    return refused("parent_rule", lambda: seed_doc(variant("abort_under_cut", edit)))


def r_ptq_parent_stage():
    def edit(export, binding):
        set_kind(export, "E7:s43", None)
        entry(export, "E7:s43")["catalogue"]["parent"] = "E2:s43"
        drop(export, binding, "E7:s43")
    return refused("parent_rule", lambda: seed_doc(variant("ptq_parent_e2", edit)))


def r_seed42_before_parentage():
    def edit(export, binding):
        set_kind(export, "E3:s42", "cut", cut_date="2026-10-19", rule="AM-19 item 2", comparator=None)
    return refused("seed42_not_complete", lambda: seed_doc(variant("s42_e3_cut", edit)))


def r_ptq_parent_seed():
    def edit(export, binding):
        set_kind(export, "E7:s43", None)
        entry(export, "E7:s43")["catalogue"]["parent"] = "E3:s44"
        drop(export, binding, "E7:s43")
    return refused("seed_inconsistent", lambda: seed_doc(variant("ptq_parent_s44", edit)))


def r_telemetry_path():
    def edit(export, binding):
        v_cut_e2(export, binding)
        entry(export, "E2:s43")["telemetry"] = [{k: v for k, v in CTX.tel["e2_s43_cut"].items()
                                                 if k != "path"}]
    return refused("telemetry_invalid", lambda: seed_doc(variant("tel_no_path", edit)))


def r_row_role():
    def edit(export, binding):
        row(export, "E2:s43", "fp32")["model_role"] = "teacher"
    return refused("precision_mismatch", lambda: seed_doc(variant("row_teacher", edit)))


# --------------------------------------------------------------------------------------------------
# D and X
# --------------------------------------------------------------------------------------------------
def d_identical():
    a = A.canonical_json(seed_doc())
    b = A.canonical_json(seed_doc(variant("cut_e2", v_cut_e2)))
    c = A.canonical_json(seed_doc(variant("cut_e2", v_cut_e2)))
    return b == c and a != b and A.canonical_json(seed_doc()) == a, ""


def d_key_order():
    doc = seed_doc()
    keys = list(doc)
    return keys[:4] == ["part", "title", "seeds", "robustness_seeds"] and keys[-1] == "footnote" and \
        json.loads(A.canonical_json(doc)) == doc, str(keys)


#: X1 scans the six LREP modules of src/stats. LRB19 L-14: they may import the statistics module's
#: engineering_shifts and hodges_lehmann, as bootstrap.py does, and nothing else from it.
LREP_SRC = ("heldout_inputs", "seed_stability", "strata_tables", "default_candidates", "am6_tables",
            "report_extensions")
L14_ALLOWED = {"engineering_shifts", "hodges_lehmann"}
STATS_PACKAGES = {"scipy", "statsmodels"}
FORBIDDEN_NAMES = {"run_comparison", "holm_family", "wilcoxon", "paired_t_test", "bca_interval",
                   "run_bootstrap_task"}


def x_no_test_objects():
    """No test or p-value object is imported or reached: no import of the statistics module but L-14's
    two names (`from .tests import ...`, `from . import tests`, `import src.stats.tests`), no scipy or
    statsmodels import, and no attribute named after a test object (e.g. D.run_comparison)."""
    bad = []
    for name in LREP_SRC:
        path = REPO / "src" / "stats" / f"{name}.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = {a.name for a in node.names}
                if mod.split(".")[-1] == "tests":
                    bad += [(path.name, f"{mod}.{n}") for n in sorted(names - L14_ALLOWED)]
                elif mod.split(".")[0] in STATS_PACKAGES:
                    bad.append((path.name, mod))
                bad += [(path.name, n) for n in sorted(names & ({"tests"} | FORBIDDEN_NAMES))]
            elif isinstance(node, ast.Import):
                bad += [(path.name, a.name) for a in node.names
                        if a.name.split(".")[-1] == "tests" or a.name.split(".")[0] in STATS_PACKAGES]
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES | {"tests"}:
                bad.append((path.name, f".{node.attr}"))
    return not bad, str(bad)


def x_repr_floats():
    """Every JSON float round-trips by repr, and the seed-42 m values equal a float64 recomputation from
    the 37's dataset totals in this smoke (so no value is rounded on its way out; R4-07)."""
    doc = seed_doc(variant("cut_e2", v_cut_e2))
    floats = []

    def walk(v):
        if isinstance(v, float):
            floats.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(doc)
    back = json.loads(json.dumps(doc, allow_nan=False))
    exact = {}
    for stage in D.STAGES:
        s = stats_artifact().clean(stage).run.stats
        tp = np.asarray(s.dataset_tp, np.int64)
        union = np.asarray(s.dataset_gt, np.int64) + np.asarray(s.dataset_pred, np.int64) - tp
        keep = union > 0
        exact[stage] = float(np.mean(tp[keep].astype(np.float64) / union[keep].astype(np.float64)))
    m42 = {r["stage"]: r["m"] for r in doc["models"] if r["seed"] == 42}
    return (back == doc and all(float(repr(f)) == f for f in floats) and len(floats) > 50
            and m42 == exact), f"{len(floats)} floats; m(S,42) equal: {m42 == exact}"


def x_pp_markdown_only():
    doc = seed_doc()
    text = json.dumps(doc)
    md = "\n".join(SS.render_markdown(doc))
    return ("_pp" not in text and "(pp)" not in text and " pp)" in md
            and "m (pp)" in md and doc["verdict_sentence"] in md), ""


def x_no_test_path():
    root = CTX.world.root
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(root) for n in dirs + files
           if "test" in n.lower()]
    return not bad and "test" not in str(root).lower(), str(bad[:3])


def x_path_rule():
    probs = [H.path_problem(p) for p in ("runs/heldout/TeSt_x", "/abs", "a/../b", "a\\b", "ok/path")]
    return all(probs[:4]) and probs[4] is None and "test" in probs[0], str(probs)


def x_no_slot_in_markdown():
    """IF§2: slot keys with ':' stay inside the JSON; the markdown names runs by stage and seed."""
    slots = {e["slot"] for e in CTX.world.export["entries"] if ":" in e["slot"]}
    doc = seed_doc(variant("cut_e2", v_cut_e2))
    md = "\n".join(SS.render_markdown(doc))
    bad = sorted(s for s in slots if s in md)
    return not bad and any(s in A.canonical_json(doc) for s in slots), str(bad[:5])


# --------------------------------------------------------------------------------------------------
# setup and run
# --------------------------------------------------------------------------------------------------
def setup(root: Path, world=None) -> None:
    """Build the world once (or take a shared one): the base fixtures, two partial telemetry files and
    two edited copies."""
    w = world or LF.build_world(root)
    CTX.world, CTX.stats, CTX.paths = w, None, {}
    CTX.tel["e2_s43_cut"] = LF.write_telemetry(
        w, "e2_s43_cut", run_id="e2_s43_lambda0.5_80k_a1", unit="iter",
        points=LF.kd_curve(0.37, iters=list(range(4000, 48001, 4000)) + [50000]))
    CTX.tel["e6_s43_cut"] = LF.write_telemetry(w, "e6_s43_cut", run_id="e6_s43_qat_a1", unit="epoch",
                                               points=LF.qat_curve(0.36, epochs=range(1, 10)))
    CTX.tel["e3_s43_cut"] = LF.write_telemetry(w, "e3_s43_cut", run_id="e3_s43_lambda0.5_alpha25_80k_a1",
                                               unit="iter", points=LF.kd_curve(0.38, iters=E3_CUT_ITERS))
    CTX.tel["e2_s43_odd"] = LF.write_telemetry(w, "e2_s43_odd", run_id="e2_s43_lambda0.5_80k_a1",
                                               unit="iter", points=LF.kd_curve(0.37, iters=[2000, 6000,
                                                                                          10000]))
    CTX.copies["upstream"] = LF.edited_copy(
        w, w.art("E2", 43), "e2_s43_clean_upstream",
        lambda s: {**s, "dataset": {**s["dataset"], "preprocess_protocol": "upstream/1.0.0"}})
    CTX.copies["other_metric"] = LF.edited_copy(
        w, w.art("E3", 43), "e3_s43_clean_othermetric",
        lambda s: {**s, "run": {**s["run"], "metric_impl_sha256": LF.syn_sha("another metric")}})


def cases():
    """(id, description, function) in report order; each function returns (ok, detail)."""
    return [
        ("F1", "m(S,k) on three hand totals (0.75, 0.25, 0.5; prediction-only class counts)", f_m_hand),
        ("F2", "a clean effect is m(candidate) - m(baseline)", f_effect_sign),
        ("F3", "d = m(E3) - m(E6) is the NI effect's negative", f_ni_direction),
        ("F4", "nested mC on hand cells (0.25)", f_miou_c_hand),
        ("F5", "mC over an explicit cell map == report.dataset_miou_c bitwise", f_miou_c_report),
        ("F6", "g_k and d_k from the model table; the descriptive and NI cells equal g and -d", f_g_and_d),
        ("F7", "the mean gain is math.fsum(g)/3, not a naive sum", f_fsum_mean),
        ("F8", "statistics.stdev exact on hand values (0.0, 0.25)", f_stdev_exact),
        ("F9", "the SD is the sample SD (n - 1), not pstdev", f_stdev_sample),
        ("F10", "condition 2 compares with the larger of the two SDs", f_c2_larger_sd),
        ("F11", "condition 3 compares d with the fraction 0.020, not 2.0", f_c3_fraction),
        ("F12", "the world's seed-42 values equal the statistics artifact's (14 bitwise checks)",
         f_seed42_world),
        ("F13", "seed-42 robustness effect == report.dataset_miou_c_contrast's difference",
         f_robust42_report),
        ("S1", "all three hold -> stable", s_all_hold),
        ("S2", "g_43 = 0 -> condition 1 fails at seed 43", s_c1_zero),
        ("S3", "g_44 < 0 -> condition 1 fails at seed 44", s_c1_negative),
        ("S4", "condition 1 fails even with seed 43 missing", s_c1_fails_missing),
        ("S5", "condition 1 not evaluable with a seed missing", s_c1_not_evaluable),
        ("S6", "condition 2 holds", s_c2_holds),
        ("S7", "condition 2 fails at exact equality (0.25 = 0.25)", s_c2_equal_fails),
        ("S8", "condition 2 not evaluable with two seeds (no mean or SD taken)", s_c2_two_seeds),
        ("S9", "condition 3 fails at d = 0.020 exactly", s_c3_at_threshold),
        ("S10", "condition 3 holds at nextafter(0.020, 0)", s_c3_below),
        ("S11", "condition 3 fails while another seed is missing", s_c3_fails_missing),
        ("S12", "condition 3 not evaluable with a seed missing", s_c3_not_evaluable),
        ("S13", "E6 non-finite at seed 43: condition 3 not evaluable, the reason names it (D-6)",
         s_nonfinite_reason),
        ("S14", "an E3 divergence makes the verdict not stable whatever the conditions", s_e3_override),
        ("S15", "the divergence reason comes first", s_divergence_first),
        ("S16", "several reasons in the fixed order (D-7)", s_reason_order),
        ("S17", "an E2 divergence leaves the verdict and is in the sentence", s_e2_sentence),
        ("S18", "a missing run never gives stable", s_missing_never_stable),
        ("S19", "a condition's values at every seed where they exist", s_values_reported),
        ("S20", "condition 2's failure has no seed", s_c2_no_seed),
        ("S21", "a missing run's reason names its entry's decision-log rows (D-6)", s_reason_entry),
        ("L1", "cut (YYYY-MM-DD, rule)", l_cut),
        ("L2", "diverged (iteration n, rule)", l_diverged),
        ("L3", "aborted (rule, cause)", l_aborted),
        ("L4", "not run: parent cut (PTQ kind null; QAT run not_run)", l_parent_cut),
        ("L5", "not run: parent diverged", l_parent_diverged),
        ("L6", "not evaluable: non-finite: no model (AM-21 item 2(b))", l_nonfinite),
        ("L7", "two missing runs: both labels, baseline first (D-15)", l_two_missing),
        ("L8", "the cut date is YYYY-MM-DD; another form is refused", l_date),
        ("L9", "every missing-cell text starts with its stage", l_stage_prefix),
        ("L10", "not made: <reason> (IF§11.2)", l_not_made),
        ("L11", "a QAT run diverged at an epoch: diverged (epoch n, rule)", l_epoch),
        ("W1", "the complete world: ten labelled rows, stable, every cell a value (LRB13)", w_base),
        ("W2", "robustness_e1_e6 at seeds 42, 43 and 44 (LRB12)", w_robustness_seeds),
        ("W3", "slots no table reads are never refused, even superseded (LRB4)", w_ghosts_ignored),
        ("W4", "a cut KD run: curve by iteration, paired comparators, the entry's comparator, no "
               "interpolation (LRB6)", w_cut_kd_curve),
        ("W5", "a cut QAT run: curve by epoch against the seed-42 model-of-record run (LRB6)",
         w_cut_qat_curve),
        ("W6", "E3 diverged at 44: descendants not run, reasons in order, limitation, the pair's "
               "status", w_div_e3),
        ("W7", "E2 diverged at 43: stated in the verdict sentence (AM-19 item 6(e))", w_div_e2_sentence),
        ("W8", "a cut E3: comparators E2, E1 and E6 (no model, stated)", w_cut_e3_comparators),
        ("W9", "two paired cut KD runs show each other's VAL rows (AM-19 item 3(c))", w_cut_pair),
        ("W10", "a comparator sharing no point is stated as such in the markdown (LRB6)",
         w_no_shared_point),
        ("R1", "R-1 stats_artifact_unverified (its inputs missing)", r_stats_unverified),
        ("R2a", "stub_reader_official", r_stub_official),
        ("R2b", "R-2 inventory_unreadable", r_inventory_unreadable),
        ("R2c", "R-2 inventory_sha256_mismatch", r_inventory_sha),
        ("R2d", "R-2 inventory_invalid (schema)", r_inventory_invalid),
        ("R2e", "R-2 binding_invalid (sha256)", r_binding_invalid),
        ("R2f", "binding_unbound under Policy.OFFICIAL (a non-stub reader)", r_binding_unbound),
        ("R2g", "freeze_mismatch", r_freeze),
        ("R2h", "R-2 binding_invalid (a bound export without binding.artifacts; IF§11.6)",
         r_binding_no_artifacts),
        ("R2i", "R-2 inventory_invalid (a model row's entry_kind is not its slot's; IF§11.1)",
         r_row_entry_kind),
        ("R2j", "R-2 inventory_unreadable (a repeated JSON key; the strict parse)", r_duplicate_key),
        ("R3a", "R-3 planned_slot_missing", r_slot_missing),
        ("R3b", "R-3 planned_slot_duplicate", r_slot_duplicate),
        ("R3c", "R-3 unplanned_slot (a per_seed slot at seed 45)", r_unplanned),
        ("R4a", "R-4 status_invalid (a kind outside the vocabulary)", r_status_vocab),
        ("R4b", "R-4 status_invalid (a cut without its rule)", r_status_field),
        ("R4c", "R-4 status_invalid (two entries for one planned slot)", r_two_entries),
        ("R4d", "R-4 status_invalid (a KD run diverged at an epoch; IF§11.2)", r_divergence_unit),
        ("R5", "R-5 superseded", r_superseded),
        ("R6a", "R-6 artifact_missing (a complete run without its record)", r_artifact_missing),
        ("R6b", "R-6 artifact_for_missing_run", r_artifact_for_missing),
        ("R7a", "R-7 artifact_record_mismatch (files_sha256)", r_record_files),
        ("R7b", "R-7 artifact_record_mismatch (protocol; LRB3)", r_record_protocol),
        ("R7c", "R-7 artifact_record_mismatch (the model row's identity)", r_record_identity),
        ("R8", "R-8 not_canvas", r_not_canvas),
        ("R9", "R-9 artifact_unreadable", r_unreadable),
        ("R10a", "R-10 cross_artifact_mismatch (metric_impl_sha256)", r_cross),
        ("R10b", "R-10 e1_checkpoint_record (E1 seeds 43 and 44 swapped)", r_e1_record),
        ("R11", "R-11 seed42_mismatch (the 37's identity against the family slot)", r_seed42),
        ("R12", "R-12 parent_rule (E6 not run while E3 is complete)", r_parent),
        ("R12b", "R-12 parent_rule (E4 descends from E2's run; LRB5)", r_parent_run),
        ("R12c", "R-12 parent_rule (a selection without a model whose parent is the E3 run; LRB5)",
         r_parent_selection),
        ("R12d", "R-12 parent_rule (E6's QAT run aborted under a cut E3; AM-19 item 4(a))",
         r_parent_not_run),
        ("R12e", "R-12 parent_rule (a PTQ model with kind null whose parent is E2's run)",
         r_ptq_parent_stage),
        ("R13", "R-13 seed42_not_complete", r_seed42_complete),
        ("R13b", "R-13 seed42_not_complete before R-12 (E3 cut at seed 42, E6/E7 with models)",
         r_seed42_before_parentage),
        ("R14", "R-14 seed_inconsistent", r_seed_inconsistent),
        ("R14b", "R-14 seed_inconsistent (a PTQ model with kind null whose parent is at seed 44)",
         r_ptq_parent_seed),
        ("R15", "R-15 an assertion: no mean or SD over two seeds", r_assert_two_seeds),
        ("R16", "telemetry_sha256_mismatch (LRB2)", r_telemetry_sha),
        ("R17", "telemetry_invalid: no attempt of the run of record (LRB2)", r_telemetry_invalid),
        ("R17b", "telemetry_invalid: a telemetry item without its path (LRB2)", r_telemetry_path),
        ("R18", "precision_mismatch (LRB7)", r_precision),
        ("R18b", "precision_mismatch: a model row whose model_role is teacher (LRB7)", r_row_role),
        ("R19", "path_unsafe", r_path),
        ("R20", "robustness_cells_missing (LRB11)", r_robustness_cells),
        ("D1", "byte-identical JSON on repeated runs", d_identical),
        ("D2", "fixed key order", d_key_order),
        ("X1", "no test or p-value object imported or reached in the six LREP modules (AST; L-14)",
         x_no_test_objects),
        ("X2", "JSON floats round-trip by repr; m(S,42) equals a float64 recomputation", x_repr_floats),
        ("X3", "pp appears in the markdown only", x_pp_markdown_only),
        ("X4", "no created path contains 'test'", x_no_test_path),
        ("X5", "path_problem: 'test' in any case, absolute, '..' and backslash", x_path_rule),
        ("X6", "no slot key with ':' in the markdown; the JSON keeps them (IF§2)", x_no_slot_in_markdown),
    ]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_seed_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_seed_stability: temporary root {root} (left in place)", flush=True)
    setup(root)
    print(f"  (world built in {time.time() - t0:.0f} s)", flush=True)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'L-AM8 SEED STABILITY OK' if good == len(results) else 'L-AM8 SEED STABILITY FAILED'}"
          f" ({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; no table here is a thesis result.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
