#!/usr/bin/env python3
"""Rare-class strata TEST tables (lane LREP, B-2): src/stats/strata_tables.py against AM-17 item 8,
the plan of record's §2.1, D-8 and D-9 (LRB16).

SYNTHETIC ONLY (DL-41). Two fixtures: a hand fixture of eight stages over 115 disease classes built
from integer counts (every IoU an exact quotient, dataset-level values from src.stats.eligibility's
own arithmetic), and scripts/lrep_fixtures.py's world for the end-to-end part. Created paths live
under a new directory in TMPDIR (left in place); none contains "test".

  H  hand values: deltas, the five group summaries, not-evaluable classes, the background line, the
     INT8 zero-IoU lists of the four direct conversions (D-8), ties, row counts, pair order
  E  end to end on the world's 37 inputs: integrity, determinism, repr floats, the pp column
  R  refusals: strata_invalid, strata_not_of_record, strata_gt_mismatch, strata_integrity, and a
     statistics-artifact input that is not its family slot's model (seed42_mismatch)

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_strata_tables.py
"""
from __future__ import annotations

import copy
import json
import math
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

import numpy as np                                                       # noqa: E402

import lrep_fixtures as LF                                               # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import strata_tables as ST                                # noqa: E402
from src.stats.eligibility import UNION_PRESENT, ClassTotals, dataset_miou  # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, hand=None, strata=None, paths={})
C = 116


# --------------------------------------------------------------------------------------------------
# the hand fixture
# --------------------------------------------------------------------------------------------------
def hand_summary(tp, gt, pred) -> dict:
    """A summary.json-shaped dict from integer per-class counts, with the evaluator's arithmetic."""
    tp, gt, pred = (np.asarray(x, np.int64) for x in (tp, gt, pred))
    un = gt + pred - tp
    tot = ClassTotals(tp=tp, gt=gt, pred=pred)
    allc, dis = dataset_miou(tot, range(C), UNION_PRESENT), dataset_miou(tot, range(1, C), UNION_PRESENT)
    return {"per_class": {"class_ids": list(range(C)), "gt_support": gt.tolist(),
                          "pred_support": pred.tolist(), "intersection": tp.tolist(), "union": un.tolist(),
                          "iou": [int(tp[c]) / int(un[c]) if un[c] > 0 else None for c in range(C)],
                          "iou_status": ["ok" if un[c] > 0 else "undefined_absent_from_gt_and_pred"
                                         for c in range(C)]},
            "dataset_level": {"all_class_miou": allc.value, "all_class_miou_n_eligible": allc.n_classes,
                              "disease_only_miou": dis.value,
                              "disease_only_miou_n_eligible": dis.n_classes}}


def hand_fixture():
    """The seven stages over classes 0..115. Every class has gt 8, tp 4 and pred 4 (IoU 4/8), except:
    class 42 has no ground truth; E2 moves class 1 to 6/8 and class 2 to 2/8 (class 3 stays tied);
    E4-E7 have tp 0 on class 5 (IoU 0 after conversion); class 7 has tp 0 everywhere but E2 (already
    0 before conversion)."""
    base_tp = np.full(C, 4, np.int64)
    gt = np.full(C, 8, np.int64)
    pred = np.full(C, 4, np.int64)
    gt[42], base_tp[42], pred[42] = 0, 0, 0
    stages = {}
    for st in D.STAGES:
        tp, pr = base_tp.copy(), pred.copy()
        if st == "E2":
            tp[1], pr[1] = 6, 6
            tp[2], pr[2] = 2, 2
        else:
            tp[7] = 0
        if st in ("E4", "E5", "E6", "E7"):
            tp[5] = 0
        stages[st] = hand_summary(tp, gt, pr)
    return stages


def hand_strata():
    """Terciles by class id (1-38 T1, 39-76 T2, 77-115 T3); classes 1-28 rare."""
    return {c: {"tercile": "T1" if c <= 38 else ("T2" if c <= 76 else "T3"), "rare": c <= 28,
                "share": 1.0 / 115, "rank": c} for c in range(1, 116)}


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:300]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def pair(b, c):
    return ST.pair_table(b, c, CTX.hand, CTX.strata)


def row_of(t, cls):
    return next(r for r in t["rows"] if r["class_id"] == cls)


def stats_artifact():
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def world_doc(inventory=None, *, strata=None, official=False):
    st = stats_artifact()
    inv = H.open_inventory(inventory or CTX.world.inventory_path, policy=st.policy)
    return ST.build_strata_tables(st, inv, H.planned_cells(inv), strata_path=strata or CTX.world.strata_path,
                                  official=official)


# --------------------------------------------------------------------------------------------------
# H: hand values
# --------------------------------------------------------------------------------------------------
def h_delta_better():
    r = row_of(pair("E1", "E2"), 1)
    return r["delta"] == 6 / 8 - 4 / 8 == 0.25 and r["status"] == "ok", str(r)


def h_delta_worse():
    r = row_of(pair("E1", "E2"), 2)
    return r["delta"] == 2 / 8 - 4 / 8 == -0.25, str(r)


def h_delta_tie():
    r = row_of(pair("E1", "E2"), 3)
    return r["delta"] == 0.0, str(r)


def h_delta_direction():
    a, b = row_of(pair("E1", "E2"), 1)["delta"], row_of(pair("E2", "E1"), 1)["delta"]
    return a == -b and a > 0, f"{a} {b}"


def _group(t, g):
    return t["groups"][g]


def h_group(g):
    def fn():
        t = pair("E1", "E2")
        rows = [r for r in t["rows"] if r["status"] == "ok" and
                (r["stratum"] == g if g in ST.TERCILES else (r["rare"] if g == "rare" else True))]
        d = [r["delta"] for r in rows]
        s = _group(t, g)
        want = {"n_classes": len(d), "mean_delta": math.fsum(d) / len(d),
                "median_delta": float(np.median(np.asarray(d, np.float64))),
                "candidate_better": sum(x > 0 for x in d), "baseline_better": sum(x < 0 for x in d),
                "ties": sum(x == 0 for x in d)}
        return all(s[k] == v for k, v in want.items()), f"{s} vs {want}"
    return fn


def h_not_evaluable():
    t = pair("E1", "E2")
    r = row_of(t, 42)
    return (r["status"] == "not_evaluable" and r["delta"] is None
            and t["groups"]["T2"]["n_not_evaluable"] == 1 and t["groups"]["overall"]["n_classes"] == 114), \
        str(r)


def h_background():
    t = pair("E1", "E2")
    return (t["background"]["class_id"] == 0 and all(r["class_id"] != 0 for r in t["rows"])), \
        str(t["background"])


def h_zero(b, c):
    def fn():
        z = pair(b, c)["zero_iou_after_int8"]
        return z is not None and [x["class_id"] for x in z["classes"]] == [5], str(z)
    return fn


def h_no_list_e1_e6():
    return pair("E1", "E6")["zero_iou_after_int8"] is None, ""


def h_already_zero():
    z = pair("E3", "E7")["zero_iou_after_int8"]
    return 7 not in [x["class_id"] for x in z["classes"]], str(z)


def h_ties_counted():
    t = pair("E1", "E3")
    return t["groups"]["overall"]["ties"] == 114 and t["groups"]["overall"]["candidate_better"] == 0, \
        str(t["groups"]["overall"])


def h_rows():
    t = pair("E1", "E2")
    return [r["class_id"] for r in t["rows"]] == list(range(1, 116)) and t["background"] is not None, ""


def h_pair_order():
    doc = world_doc()
    return [(p["baseline_stage"], p["candidate_stage"]) for p in doc["pairs"]] == list(ST.PAIRS), ""


# --------------------------------------------------------------------------------------------------
# E: end to end
# --------------------------------------------------------------------------------------------------
def e_integrity():
    doc = world_doc()
    rec = doc["integrity"]
    return (sorted(rec) == sorted(D.STAGES)
            and all(rec[s]["all_class"]["bitwise_equal"] and rec[s]["disease_only"]["bitwise_equal"]
                    and rec[s]["table_iou_equals_summary_and_quotient"] for s in rec)), ""


def e_iou_from_summary():
    doc, st = world_doc(), stats_artifact()
    p = doc["pairs"][0]
    pc = st.clean("E2").summary["per_class"]["iou"]
    return all(r["iou_candidate"] == pc[r["class_id"]] for r in p["rows"]), ""


def e_not_evaluable_world():
    doc, st = world_doc(), stats_artifact()
    gt = st.clean("E1").summary["per_class"]["gt_support"]
    return doc["not_evaluable_classes"] == [c for c in range(1, 116) if gt[c] == 0], \
        str(doc["not_evaluable_classes"][:5])


def e_strata_record():
    doc = world_doc()
    rec = doc["strata_file"]
    return (rec["of_record"] is False and rec["sha256"] == H.sha256_file(CTX.world.strata_path)
            and doc["tercile_sizes"] == {"T1": 38, "T2": 38, "T3": 39} and doc["rare_classes"] == 28), \
        str(rec)


def e_committed_strata():
    """The committed DL-63 file is read and recorded as of record (nonofficial profile)."""
    rec = ST.load_strata_checked(REPO / ST.STRATA_REL, official=True)[1]
    return rec["of_record"] is True and rec["sha256"] == ST.STRATA_SHA256_OF_RECORD, str(rec)


def e_deterministic():
    return A.canonical_json(world_doc()) == A.canonical_json(world_doc()), ""


def e_deterministic_hand():
    return json.dumps(pair("E1", "E5"), sort_keys=True) == json.dumps(pair("E1", "E5"), sort_keys=True), ""


def e_repr_floats():
    doc = world_doc()
    back = json.loads(A.canonical_json(doc))
    return back == doc, ""


def e_pp_markdown():
    doc = world_doc()
    md = "\n".join(ST.render_markdown(doc))
    return "mean Δ (pp)" in md and "(pp)" not in json.dumps(doc), ""


def e_no_test_path():
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower()]
    return not bad, str(bad[:3])


# --------------------------------------------------------------------------------------------------
# R: refusals
# --------------------------------------------------------------------------------------------------
def r_gt_differs():
    hand = copy.deepcopy(CTX.hand)
    tp = np.asarray(hand["E4"]["per_class"]["intersection"])
    gt = np.asarray(hand["E4"]["per_class"]["gt_support"])
    pred = np.asarray(hand["E4"]["per_class"]["pred_support"])
    gt[9] += 1
    hand["E4"] = hand_summary(tp, gt, pred)
    return refused("strata_gt_mismatch", lambda: ST.model_checks(hand))


def r_integrity():
    hand = copy.deepcopy(CTX.hand)
    hand["E3"]["dataset_level"]["all_class_miou"] = math.nextafter(
        hand["E3"]["dataset_level"]["all_class_miou"], 1.0)
    return refused("strata_integrity", lambda: ST.model_checks(hand))


def r_not_of_record():
    return refused("strata_not_of_record", lambda: world_doc(official=True))


def r_class_missing():
    if "short" not in CTX.paths:
        doc = json.loads(CTX.world.strata_path.read_text(encoding="utf-8"))
        doc["per_class"] = doc["per_class"][:-1]
        d = CTX.world.fresh("strata_short")
        (d / "strata_short.json").write_text(json.dumps(doc), encoding="utf-8")
        CTX.paths["short"] = d / "strata_short.json"
    return refused("strata_invalid", lambda: world_doc(strata=CTX.paths["short"]))


def r_seed43_model():
    """A family slot whose identity is another model (a seed-43 checkpoint) is refused (R-11)."""
    if "s43" not in CTX.paths:
        export, binding = copy.deepcopy(CTX.world.export), copy.deepcopy(CTX.world.binding)
        row = next(r for r in export["models"] if (r["slot"], r["precision"]) == ("E3:s42", "fp32"))
        row["source"]["sha256"] = LF.fp32_checkpoint("E3", 43)
        CTX.paths["s43"] = LF.write_inventory(CTX.world.fresh("strata_s43"), export, binding)
    return refused("seed42_mismatch", lambda: world_doc(CTX.paths["s43"]))


# --------------------------------------------------------------------------------------------------
def setup(root: Path, world=None) -> None:
    CTX.world = world or LF.build_world(root)
    CTX.stats, CTX.paths = None, {}
    CTX.hand, CTX.strata = hand_fixture(), hand_strata()


def cases():
    out = [("H1", "delta of a class the candidate improves (6/10 - 4/12)", h_delta_better),
           ("H2", "delta of a class the candidate worsens", h_delta_worse),
           ("H3", "an exact tie has delta 0.0", h_delta_tie),
           ("H4", "delta is candidate - baseline", h_delta_direction)]
    out += [(f"H{5 + i}", f"group summary {g}: n, fsum mean, float64 median, counts", h_group(g))
            for i, g in enumerate(ST.GROUPS)]
    out += [("H10", "a class with no TEST ground truth is not evaluable and enters no summary",
             h_not_evaluable),
            ("H11", "the background line is separate", h_background)]
    out += [(f"H{12 + i}", f"zero-IoU list for {b} -> {c} (D-8)", h_zero(b, c))
            for i, (b, c) in enumerate(ST.ZERO_IOU_PAIRS)]
    out += [("H16", "E1 -> E6 has no zero-IoU list (D-8)", h_no_list_e1_e6),
            ("H17", "a class already at 0 is not listed", h_already_zero),
            ("H18", "exact ties are counted", h_ties_counted),
            ("H19", "115 rows plus the background line", h_rows),
            ("H20", "the eight pairs in the fixed order", h_pair_order),
            ("E1", "per model, the per-class supports reproduce the summary bitwise (perclass)",
             e_integrity),
            ("E2", "IoU columns are summary.per_class.iou exactly", e_iou_from_summary),
            ("E3", "not-evaluable classes are those with no TEST gt_support", e_not_evaluable_world),
            ("E4", "the strata file's sha256 and state; terciles 38/38/39, 28 rare", e_strata_record),
            ("E5", "the committed DL-63 file is of record", e_committed_strata),
            ("E6", "byte-identical JSON on repeated runs", e_deterministic),
            ("E7", "the hand table is deterministic", e_deterministic_hand),
            ("E8", "JSON floats unrounded (repr round trip)", e_repr_floats),
            ("E9", "the pp column is in the markdown only", e_pp_markdown),
            ("E10", "no created path contains 'test'", e_no_test_path),
            ("R1", "strata_gt_mismatch (one model's gt_support differs)", r_gt_differs),
            ("R2", "strata_integrity (a summary not reproduced bitwise)", r_integrity),
            ("R3", "strata_not_of_record (official profile, synthetic file)", r_not_of_record),
            ("R4", "strata_invalid (a class missing from the strata file)", r_class_missing),
            ("R5", "seed42_mismatch (a family slot that is not the 37's model)", r_seed43_model)]
    return out


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_strata_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_strata_tables: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'STRATA TABLES OK' if good == len(results) else 'STRATA TABLES FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; no table here is a thesis result.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
