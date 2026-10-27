#!/usr/bin/env python3
"""AM-23's descriptive rows on TEST (lane LREP, B-5 (c)): src/stats/quant_rows.py against AM-23 items 5-6
(PA :2607-2615), LRB22 and ruling B5-S1.

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world (its quant_descriptive slots E6-E2, E7-E2,
E6-R1 and E6-R2) and variants of its export under a new directory in TMPDIR (left in place); none
contains "test". Expected values are recomputed here from the artifacts.

  Q  the rows: E6-E2 − E6 and E7-E2 − E7 (all-class from m, GT-present and disease-only), E2 − E3 beside
     them as the negative of the official accuracy_e2_e3 delta, B5-S1's evaluator stages, the E6-X rows
     named in the arms part, the converted VAL scores "not on record"
  L  labels: a cut QAT run, "not made: code not merged", a stage the export has no slot for; a slot
     without a model whose clean artifact is still bound (artifact_for_missing_run)
  R  quant_slot_invalid (a stage outside the four; a seed other than 42; two slots for one stage);
     R-11 on the family slots; an E2 parent that is not the family slot (parent_rule, R-12)
  D  no interval or test key, determinism, no test object imported, no 'test' path, the markdown

Cases are importable (setup / cases) so that scripts/smoke_lrep_mutations.py can run them mutated.

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_quant_rows.py
"""
from __future__ import annotations

import ast
import copy
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
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import quant_rows as QR                                   # noqa: E402
from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, rule_variants  # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402
from src.stats.seed_stability import m_of                                # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, paths={}, doc=None)
CUT = {"cut_date": "2026-10-24", "rule": "AM-22 item 2(a): the launch cutoff", "comparator": None}


def stats_artifact():
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def doc_of(inventory=None):
    st = stats_artifact()
    inv = H.open_inventory(inventory or CTX.world.inventory_path, policy=st.policy)
    return QR.build_quant_rows(st, inv, H.planned_cells(inv), input_root=CTX.world.input_root)


def the_doc():
    if CTX.doc is None:
        CTX.doc = doc_of()
    return CTX.doc


def variant(name, edit):
    if name not in CTX.paths:
        export, binding = copy.deepcopy(CTX.world.export), copy.deepcopy(CTX.world.binding)
        edit(export, binding)
        CTX.paths[name] = LF.write_inventory(CTX.world.fresh(name), export, binding, CTX.world.dedup)
    return CTX.paths[name]


def entry(export, slot):
    return next(e for e in export["entries"] if e["slot"] == slot)


def drop_rows(export, binding, slot):
    export["models"] = [r for r in export["models"] if r["slot"] != slot]
    binding["artifacts"] = [r for r in binding["artifacts"] if r["slot"] != slot]


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:300]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def row(doc, cand):
    return next(r for r in doc["rows"] if r["candidate_stage"] == cand)


def values_of(stats):
    rv = rule_variants(stats)
    return (m_of(stats), rv[GT_PRESENT]["all_class"].value_float64,
            rv[UNION_PRESENT]["disease_only"].value_float64)


def load(stage, slot, precision):
    w = CTX.world
    rec = next(r for r in w.binding["artifacts"]
               if (r["slot"], r["precision"], r["scope"]) == (slot, precision, "clean"))
    return D.load_input(stage, D.CLEAN, rec["repo_relative_path"], w.input_root,
                        Policy.NONOFFICIAL_SMOKE)


# --------------------------------------------------------------------------------------------------
def q_rows():
    """E6-E2 − E6 and E7-E2 − E7: candidate − baseline on all-class m, GT-present and disease-only."""
    doc = the_doc()
    st = stats_artifact()
    ok = []
    for cand, base, slot, prec in (("E6-E2", "E6", "E6-E2:s42:sel", "int8_qat"),
                                   ("E7-E2", "E7", "E7-E2:s42", "int8_ptq")):
        c = values_of(load(cand, slot, prec).run.stats)
        b = values_of(st.clean(base).run.stats)
        d = row(doc, cand)["differences"]
        ok.append((d["all_class"], d["gt_present"], d["disease_only"]) == tuple(x - y for x, y in zip(c, b)))
    return all(ok) and [r["candidate_stage"] for r in doc["rows"]] == ["E6-E2", "E7-E2"], str(ok)


def q_reference():
    """E2 − E3 = m(E2) − m(E3), bitwise the negative of the official accuracy_e2_e3 delta (E3 − E2)."""
    doc = the_doc()
    st = stats_artifact()
    off = next(t["observed"] for t in st.family["bootstrap_tasks"]
               if t["task_id"] == "accuracy_e2_e3__dataset_miou_delta")
    d = doc["reference"]["differences"]["all_class"]
    want = m_of(st.clean("E2").run.stats) - m_of(st.clean("E3").run.stats)
    ref = doc["reference"]
    return (d == want == -off and (ref["candidate_stage"], ref["baseline_stage"]) == ("E2", "E3")
            and "negative of the official accuracy_e2_e3 delta" in ref["note"]), f"{d} {off}"


def q_stage_map():
    """B5-S1: E6-E2 read at evaluator stage E6-E2 (int8_qat), E7-E2 at E7-E2 (int8_ptq)."""
    w = CTX.world
    st = stats_artifact()
    inv = H.open_inventory(w.inventory_path, policy=st.policy)
    got = {}
    for slot, stage in (("E6-E2:s42:sel", "E6-E2"), ("E7-E2:s42", "E7-E2")):
        ba = H.load_bound(inv, slot=slot, stage=stage, precision=H.stage_spec(stage)[2], scope="clean",
                          input_root=w.input_root, policy=st.policy)
        got[stage] = (ba.summary["run"]["stage"], ba.summary["run"]["precision"])
    return got == {"E6-E2": ("E6-E2", "int8_qat"), "E7-E2": ("E7-E2", "int8_ptq")}, str(got)


def q_e6x():
    doc = the_doc()
    return [(e["stage"], e["arms_row"], e["label"]) for e in doc["e6_x"]] == [
        ("E6-R1", "arm__r1__e6", None), ("E6-R2", "arm__r2__e6", None)], str(doc["e6_x"])


def q_converted_val():
    doc = the_doc()
    return doc["converted_val_scores"].startswith("not on record"), doc["converted_val_scores"]


def l_cut():
    def edit(export, binding):
        entry(export, "E6-E2:s42").update(kind="cut", **CUT)
        entry(export, "E6-E2:s42:sel")["kind"] = None
        drop_rows(export, binding, "E6-E2:s42:sel")
    r = row(doc_of(variant("e6e2_cut", edit)), "E6-E2")
    return (r["label"] == f"cut ({CUT['cut_date']}, {CUT['rule']})" and r["candidate"] is None
            and r["differences"] is None), str(r)[:300]


def l_not_made():
    def edit(export, binding):
        entry(export, "E7-E2:s42").update(kind="not_made", reason="code not merged")
        drop_rows(export, binding, "E7-E2:s42")
    r = row(doc_of(variant("e7e2_not_made", edit)), "E7-E2")
    return r["label"] == "not made: code not merged" and r["candidate"] is None, str(r)[:300]


def l_no_slot():
    def edit(export, binding):
        export["entries"] = [e for e in export["entries"] if e["slot"] != "E7-E2:s42"]
        drop_rows(export, binding, "E7-E2:s42")
    r = row(doc_of(variant("no_e7e2", edit)), "E7-E2")
    return r["label"] == QR.NO_SLOT and r["slot"] is None, str(r)[:300]


def r_outside():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "E7-E2:s42"))
        e.update(slot="E7-E3:s42", stage="E7-E3")
        export["entries"].append(e)
    return refused("quant_slot_invalid", lambda: doc_of(variant("quant_outside", edit)))


def r_two():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "E6-E2:s42:sel"))
        e["slot"] = "E6-E2:s42:sel2"
        export["entries"].append(e)
    return refused("quant_slot_invalid", lambda: doc_of(variant("quant_two", edit)))


def r_seed():
    """quant_slot_invalid: E7-E2's slot at seed 43 (AM-23's rows are seed 42's)."""
    def edit(export, binding):
        entry(export, "E7-E2:s42")["seed"] = 43
    return refused("quant_slot_invalid", lambda: doc_of(variant("e7e2_s43", edit)))


def l_bound_without_model():
    """R-6: E7-E2 reads 'not made' while the binding still lists its clean artifact."""
    def edit(export, binding):
        entry(export, "E7-E2:s42").update(kind="not_made", reason="code not merged")
        export["models"] = [r for r in export["models"] if r["slot"] != "E7-E2:s42"]
    return refused("artifact_for_missing_run", lambda: doc_of(variant("e7e2_bound", edit)))


def d_markdown():
    """The markdown: the two rows and E2 − E3 under the banner, the E6-X line naming the arms rows, the
    converted VAL scores' note."""
    md = QR.render_markdown(the_doc(), "BANNER")
    text = "\n".join(md)
    return (md[2] == "BANNER" and sum(1 for line in md if line.startswith("| E")) == 3
            and "| E2 − E3 |" in text and "`arm__r1__e6`" in text and "`arm__r2__e6`" in text
            and "not on record" in text), str(md[:6])


def r_identity():
    def edit(export, binding):
        r = next(r for r in export["models"] if (r["slot"], r["precision"]) == ("E2:s42", "fp32"))
        r["source"]["sha256"] = LF.syn_sha("another E2")
    return refused("seed42_mismatch", lambda: doc_of(variant("e2_identity", edit)))


def r_parent_not_family():
    """parent_rule (R-12): E7-E2 descends from the seed-42 E2 default candidate (E2:s42:lambda1), not
    the family slot, the E2 of record (AM-23 item 2; IF§11.1)."""
    def edit(export, binding):
        entry(export, "E7-E2:s42")["catalogue"]["parent"] = "E2:s42:lambda1"
    return refused("parent_rule", lambda: doc_of(variant("e7e2_parent_lambda1", edit)))


def d_no_interval():
    keys = set()

    def walk(v):
        if isinstance(v, dict):
            keys.update(v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(the_doc())
    bad = sorted(k for k in keys if "__" in k
                 or k in ("lower_bound", "upper_bound", "p", "p_value", "interval", "wilcoxon"))
    return not bad, str(bad)


def d_deterministic():
    return A.canonical_json(doc_of()) == A.canonical_json(doc_of()), ""


def d_no_test_import():
    tree = ast.parse((REPO / "src" / "stats" / "quant_rows.py").read_text(encoding="utf-8"))
    bad = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
           and ((node.module or "").split(".")[-1] in ("tests", "bootstrap")
                or (node.module or "").split(".")[0] in ("scipy", "statsmodels"))]
    return not bad, str(bad)


def d_no_test_path():
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower()]
    return not bad, str(bad[:3])


def setup(root: Path, world=None) -> None:
    CTX.world = world or LF.build_world(root, strata=False)
    CTX.stats, CTX.paths, CTX.doc = None, {}, None


def cases():
    return [("Q1", "E6-E2 − E6 and E7-E2 − E7: all-class (m), GT-present and disease-only", q_rows),
            ("Q2", "E2 − E3 = m(E2) − m(E3), the negative of the official accuracy_e2_e3 delta", q_reference),
            ("Q3", "B5-S1: E6-E2 and E7-E2 read at their own evaluator stages", q_stage_map),
            ("Q4", "E6-X − E6 named as the arms part's rows", q_e6x),
            ("Q5", "the converted VAL scores read 'not on record'", q_converted_val),
            ("L1", "E6-E2's QAT run cut: the cut label, no value", l_cut),
            ("L2", "E7-E2 'not made: code not merged' (AM-23 item 4)", l_not_made),
            ("L3", "a stage the export has no slot for", l_no_slot),
            ("L4", "artifact_for_missing_run: not made, its clean artifact still bound",
             l_bound_without_model),
            ("R1", "quant_slot_invalid: a quant_descriptive slot outside the four stages", r_outside),
            ("R2", "quant_slot_invalid: two slots for one stage", r_two),
            ("R4", "quant_slot_invalid: a slot at seed 43", r_seed),
            ("R3", "R-11: the E2 family identity is checked", r_identity),
            ("R5", "parent_rule: E7-E2 under a seed-42 E2 that is not the family slot (R-12)",
             r_parent_not_family),
            ("D1", "no interval, test or '__' key", d_no_interval),
            ("D2", "byte-identical JSON on repeated runs", d_deterministic),
            ("D3", "quant_rows imports no test object and no bootstrap (AST)", d_no_test_import),
            ("D4", "no created path contains 'test'", d_no_test_path),
            ("D5", "the markdown's rows, E6-X line and VAL note", d_markdown)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_quant_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_quant_rows: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'QUANT ROWS OK' if good == len(results) else 'QUANT ROWS FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; descriptive.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
