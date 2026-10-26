#!/usr/bin/env python3
"""GT-present on TEST for every model of record (lane LREP, B-3: L-AM6): src/stats/am6_tables.py
against AM-6, AM-17 item 1(b), STATS §9.5 and ERR E-24, the plan of record's §3 and LRB7 / LRB10 /
LRB17.

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world, plus two edited or re-scored copies of one
seed-43 artifact for the refusals. Created paths live under a new directory in TMPDIR (left in
place); none contains "test".

  N  the four numbers, both arithmetics, against eligibility.rule_variants
  F  R1 bitwise and R3, the union-present-only classes, the per-class table
  T  the table: roles and seeds, 21 rows from the 37 and the binding (LRB7), no qat_fakequant_off
     row, a run without a model listed with its label, the seed-42 rows equal to report.py's
  R  refusals: an unfaithful summary (gt_present_unfaithful), another ground truth
     (cross_artifact_mismatch), a record that is not canvas, a model of record without its record,
     E1's seed-43/44 records swapped (e1_checkpoint_record)
  D  determinism, repr floats, no 'test' path

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_am6_tables.py
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

import lrep_fixtures as LF                                               # noqa: E402
import stats_fixtures as F                                               # noqa: E402
from src.eval.evaluate import Condition                                  # noqa: E402
from src.stats import am6_tables as AM6                                  # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import report as RP                                       # noqa: E402
from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, rule_variants  # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, paths={}, copies={})


def stats_artifact():
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def doc_of(inventory=None):
    st = stats_artifact()
    inv = H.open_inventory(inventory or CTX.world.inventory_path, policy=st.policy)
    return AM6.build_am6_tables(st, inv, H.planned_cells(inv), input_root=CTX.world.input_root)


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


def row(doc, stage, seed):
    return next(r for r in doc["rows"] if (r["stage"], r["seed"]) == (stage, seed))


def bound_stats(stage, seed):
    w = CTX.world.art(stage, seed)
    li = D.load_input(stage, D.CLEAN, w.rel, CTX.world.input_root, stats_artifact().policy)
    return li.run.stats, li.summary


def point_record(binding, slot, precision, copy_w):
    r = next(r for r in binding["artifacts"]
             if (r["slot"], r["precision"], r["scope"], r.get("cell")) == (slot, precision, "clean", None))
    r["repo_relative_path"], r["files_sha256"] = copy_w.rel, dict(copy_w.files_sha256)


# --------------------------------------------------------------------------------------------------
def n_number(rule, space):
    def fn():
        doc = doc_of()
        r = row(doc, "E4", 43)
        stats, _ = bound_stats("E4", 43)
        rv = rule_variants(stats)[rule][space]
        return (r[f"{rule}_{space}"] == rv.value and r[f"{rule}_{space}_float64"] == rv.value_float64
                and r[f"{rule}_{space}_n_classes"] == rv.n_classes), str(rv)
    return fn


def f_r1(key):
    def fn():
        doc = doc_of()
        return all(r["R1"][key]["bitwise_equal"] and r["R1"][key]["n_equal"] for r in doc["rows"]), ""
    return fn


def f_r3():
    doc = doc_of()
    return all(r["R3"]["holds"] for r in doc["rows"]), str([r["R3"] for r in doc["rows"]][:1])


def f_extra_classes():
    """A hand artifact whose class 3 is predicted but absent from ground truth: it is listed, R3 holds
    (its TP is 0), every one of the row's twelve numbers (value, value_float64 and n_classes for both
    rules and both scopes) equals rule_variants', the GT-present ones differing from the union-present
    ones (R4-01); and the world's rows list exactly rule_variants' difference (empty there: k = 0)."""
    import numpy as np
    from src.stats.eligibility import class_totals, dataset_miou
    cls, tp, gt, pred = (np.array(v, np.int64) for v in ([0, 1, 3], [90, 8, 0], [100, 10, 0], [95, 9, 5]))
    tot = {"class_id": cls, "tp": tp, "gt": gt, "pred": pred}
    t = class_totals(tot, 116)
    allc, dis = dataset_miou(t, range(116), UNION_PRESENT), dataset_miou(t, range(1, 116), UNION_PRESENT)
    summary = {"dataset": {"num_classes": 116, "background_index": 0},
               "dataset_level": {"all_class_miou": allc.value, "all_class_miou_n_eligible": allc.n_classes,
                                 "disease_only_miou": dis.value,
                                 "disease_only_miou_n_eligible": dis.n_classes}}
    hand = AM6.gt_present_row(tot, summary, with_classes=False)
    hrv = rule_variants(t)
    fields = (("", "value"), ("_float64", "value_float64"), ("_n_classes", "n_classes"))
    every = all(hand[f"{rule}_{s}{sfx}"] == getattr(hrv[rule][s], attr)
                for rule in (GT_PRESENT, UNION_PRESENT) for s in ("all_class", "disease_only")
                for sfx, attr in fields)
    rules_differ = every and all(getattr(hrv[GT_PRESENT][s], attr) != getattr(hrv[UNION_PRESENT][s], attr)
                                 for s in ("all_class", "disease_only") for _, attr in fields)
    doc = doc_of()
    r = row(doc, "E2", 44)
    stats, _ = bound_stats("E2", 44)
    rv = rule_variants(stats)
    want = sorted(set(rv[UNION_PRESENT]["all_class"].eligible) - set(rv[GT_PRESENT]["all_class"].eligible))
    return (hand["union_present_not_gt_present"] == [3] and hand["R3"]["holds"] is True and rules_differ
            and r["union_present_not_gt_present"] == want), f"{hand['union_present_not_gt_present']} {want}"


def f_per_class():
    doc = doc_of()
    return all(len(r["per_class"]) == 116 and r["per_class"][0]["class_id"] == 0 for r in doc["rows"]), ""


def t_roles():
    doc = doc_of()
    return all(r["role"] == ("family" if r["seed"] == 42 else "per_seed") for r in doc["rows"]), ""


def t_seeds_slots():
    doc = doc_of()
    return all(r["slot"] == LF.record_slot(r["stage"], r["seed"]) for r in doc["rows"]), ""


def t_rows_21():
    doc = doc_of()
    keys = [(r["seed"], r["stage"]) for r in doc["rows"]]
    return keys == [(k, st) for k in H.SEEDS for st in D.STAGES] and doc["missing"] == [], str(keys[:3])


def t_no_fqoff():
    doc = doc_of()
    ids = {(r["stage"], r["seed"]): r["run_id"] for r in doc["rows"]}
    return all(ids[(st, k)] == f"lrep-{st.lower()}-s{k}-clean" for st in H.QAT_STAGES for k in H.SEEDS), \
        str({k: v for k, v in ids.items() if k[0] in H.QAT_STAGES})


def t_seed42_equals_report():
    doc = doc_of()
    ref = {g["stage"]: g for g in RP.gt_present_rows(stats_artifact().views)}
    keys = [k for k in ref["E1"] if k != "stage"]
    return all({k: (r[k] if k.endswith("_n_classes") else r[f"{k}_float64"]) for k in keys}
               == {k: ref[r["stage"]][k] for k in keys} for r in doc["rows"] if r["seed"] == 42), ""


def t_missing_listed():
    def edit(export, binding):
        e = next(e for e in export["entries"] if e["slot"] == "E2:s44")
        e.update(kind="cut", cut_date="2026-11-02", rule="AM-19 item 3(b): original freeze", comparator=None)
        export["models"] = [r for r in export["models"] if r["slot"] != "E2:s44"]
        binding["artifacts"] = [r for r in binding["artifacts"] if r["slot"] != "E2:s44"]
    doc = doc_of(variant("e2_cut", edit))
    return (doc["missing"] == [{"slot": "E2:s44", "stage": "E2", "seed": 44,
                                "label": "cut (2026-11-02, AM-19 item 3(b): original freeze)"}]
            and len(doc["rows"]) == 20), str(doc["missing"])


# --------------------------------------------------------------------------------------------------
def r_unfaithful():
    def edit(export, binding):
        point_record(binding, "E2:s43", "fp32", CTX.copies["nudged"])
    return refused("gt_present_unfaithful", lambda: doc_of(variant("nudged", edit)))


def r_other_gt():
    def edit(export, binding):
        point_record(binding, "E2:s43", "fp32", CTX.copies["other_gt"])
    return refused("cross_artifact_mismatch", lambda: doc_of(variant("other_gt", edit)))


def r_upstream():
    def edit(export, binding):
        r = next(r for r in binding["artifacts"]
                 if (r["slot"], r["precision"], r["scope"]) == ("E7:s44", "int8_ptq", "clean"))
        r["protocol"] = "upstream"
    return refused("not_canvas", lambda: doc_of(variant("upstream", edit)))


def r_no_record():
    def edit(export, binding):
        binding["artifacts"] = [r for r in binding["artifacts"]
                                if (r["slot"], r["scope"]) != ("E4:s43", "clean")]
    return refused("artifact_missing", lambda: doc_of(variant("no_record", edit)))


def d_deterministic():
    return A.canonical_json(doc_of()) == A.canonical_json(doc_of()), ""


def d_deterministic_md():
    return AM6.render_markdown(doc_of()) == AM6.render_markdown(doc_of()), ""


def d_repr():
    """The JSON round-trips, and every seed-42 row's four float64 values equal rule_variants on the 37's
    statistics, called here (so nothing is rounded on its way out; R4-07)."""
    doc = doc_of()
    st = stats_artifact()
    rv = {stage: rule_variants(st.clean(stage).run.stats) for stage in D.STAGES}
    exact = all(r[f"{rule}_{s}_float64"] == rv[r["stage"]][rule][s].value_float64
                for r in doc["rows"] if r["seed"] == 42 for rule in (GT_PRESENT, UNION_PRESENT)
                for s in ("all_class", "disease_only"))
    return json.loads(A.canonical_json(doc)) == doc and exact, ""


def r_e1_swap():
    """E1's seed-43 and seed-44 records and identities exchanged: every other check passes, and DL-99's
    record names the swap (R-10; R1-13)."""
    def edit(export, binding):
        a, b = "E1:s43", "E1:s44"
        for r in binding["artifacts"]:
            if r["slot"] in (a, b):
                r["slot"] = b if r["slot"] == a else a
        ra = next(r for r in export["models"] if (r["slot"], r["precision"]) == (a, "fp32"))
        rb = next(r for r in export["models"] if (r["slot"], r["precision"]) == (b, "fp32"))
        ra["source"]["sha256"], rb["source"]["sha256"] = rb["source"]["sha256"], ra["source"]["sha256"]
    return refused("e1_checkpoint_record", lambda: doc_of(variant("e1_swap", edit)))


def d_no_test_path():
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower()]
    return not bad, str(bad[:3])


def setup(root: Path, world=None) -> None:
    w = world or LF.build_world(root, strata=False)
    CTX.world, CTX.stats, CTX.paths = w, None, {}
    src = w.art("E2", 43)
    CTX.copies["nudged"] = LF.edited_copy(
        w, src, "e2_s43_clean_nudged",
        lambda s: {**s, "dataset_level": {**s["dataset_level"], "all_class_miou": math.nextafter(
            s["dataset_level"]["all_class_miou"], 1.0)}})
    other = F.make_truth(w.n, 0, F.int_seed(LF.WORLD_SEED, "another truth"))
    rel = "runs/heldout/e2_s43_clean_othergt"
    scored = F.score(other, F.predict(other, quality=0.7, seed=F.int_seed(LF.WORLD_SEED, rel)),
                     Condition(*D.CLEAN))
    F.write_fixture(w.input_root / rel, scored, w.n, stage="E2", condition=Condition(*D.CLEAN),
                    split="val", status="provisional", repo_root=w.input_root,
                    run_id="lrep-e2-s43-othergt", checkpoint_sha256=src.checkpoint)
    summary = json.loads((w.input_root / rel / "summary.json").read_text(encoding="utf-8"))
    CTX.copies["other_gt"] = LF.Written(
        slot=src.slot, stage="E2", seed=43, precision="fp32", scope="clean", cell=None, rel=rel,
        checkpoint=src.checkpoint, files_sha256=D.input_file_sha256(w.input_root / rel),
        run_id="lrep-e2-s43-othergt",
        split_manifest_sha256=summary["dataset"]["split_manifest_sha256"])


def cases():
    out = [(f"N{i + 1}", f"{rule} {space}: value, value_float64 and n equal rule_variants",
            n_number(rule, space)) for i, (rule, space) in enumerate(
        (r, s) for r in (UNION_PRESENT, GT_PRESENT) for s in ("all_class", "disease_only"))]
    out += [("F1", "R1: union-present all-class equals summary bitwise, same n", f_r1("all_class_miou")),
            ("F2", "R1: union-present disease-only equals summary bitwise, same n",
             f_r1("disease_only_miou")),
            ("F3", "R3: GT-present = union-present x n_union / n_gt (1e-12), extra classes TP 0", f_r3),
            ("F4", "a union-only class: listed, and all twelve numbers follow their rules", f_extra_classes),
            ("F5", "the per-class table (116 rows) is carried", f_per_class),
            ("T1", "role family at seed 42, per_seed at 43/44", t_roles),
            ("T2", "each row's slot is its planned cell's model of record", t_seeds_slots),
            ("T3", "21 rows: the 37's seven and the binding's fourteen (LRB7), seed then stage order",
             t_rows_21),
            ("T4", "E5/E6 rows are the int8_qat artifacts, never qat_fakequant_off", t_no_fqoff),
            ("T5", "the seed-42 rows equal report.gt_present_rows", t_seed42_equals_report),
            ("T6", "a run without a model is listed with its label, not read", t_missing_listed),
            ("R1", "gt_present_unfaithful: a summary value one ulp off", r_unfaithful),
            ("R2", "cross_artifact_mismatch: an artifact scored on another ground truth", r_other_gt),
            ("R3", "not_canvas: an upstream record", r_upstream),
            ("R4", "artifact_missing: a model of record whose record the binding lacks", r_no_record),
            ("R5", "e1_checkpoint_record: E1's seed-43/44 records swapped (DL-99)", r_e1_swap),
            ("D1", "byte-identical JSON on repeated runs", d_deterministic),
            ("D2", "identical markdown on repeated runs", d_deterministic_md),
            ("D3", "JSON floats unrounded: the seed-42 float64 values equal rule_variants'", d_repr),
            ("D4", "no created path contains 'test'", d_no_test_path)]
    return out


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_am6_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_am6_tables: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'L-AM6 TABLES OK' if good == len(results) else 'L-AM6 TABLES FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; no table here is a thesis result.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
