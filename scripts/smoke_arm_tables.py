#!/usr/bin/env python3
"""AM-18's arm tables on TEST (lane LREP, B-5 (a)): src/stats/arm_tables.py against AM-18 items 3(d) and
11, AM-23 item 6, rulings B5-D1, B5-D2 and B5-S1, and the teacher-values rule.

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world, plus edited copies of the committed DL-99,
DL-124 and DL-125 records under a new directory for the record refusals (the committed records are read
by exact path and never changed). Created paths live under a new directory in TMPDIR (left in place);
none contains "test".

  S  B5-D1: √2·s from DL-99's record, equal to its band and 0.015621041685101825 bitwise; the inclusive
     boundary in float64
  T  the teacher values: R1 and R2 from DL-124's and DL-125's records, flip "not on record"; the record
     refusals (arm_record_invalid)
  P  placement: the eight rows in order and no E6-flip row; the refusals of arm_slot_invalid; a model
     under a cut arm run (parent_rule, R-12)
  V  values: both sides' m and rule_variants' values, Δ and its reading, the 37 as comparators (R-11),
     the per-image mean Δ and its interval (B5-D2), B5-S1's evaluator stages
  L  a candidate without a model: its IF§11.2 label, no value, never a refusal (unless the binding lists
     its clean artifact anyway)
  D  determinism, no test or p-value key, the banner, no test object imported, the markdown's cells, no
     'test' path

Cases are importable (setup / cases) so that scripts/smoke_lrep_mutations.py can run them mutated.

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_arm_tables.py
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import shutil
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
from src.stats import arm_tables as AT                                   # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import lrep_descriptive as LD                             # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, align_runs              # noqa: E402
from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, rule_variants  # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, paths={}, doc=None)
SMOKE_B = 200
#: Values typed out from the records and rulings (never read from the module under test; R4-03).
BAND = 0.015621041685101825
S_DL99 = 0.011045744504733232
R1_VALUES = (0.38849374651908875, 0.051708213712213985, "go")
R2_VALUES = (0.39719322323799133, 0.0573499952011995, "go")
ORDER = [("R1", "E3"), ("R1", "E7"), ("R1", "E6"), ("R2", "E3"), ("R2", "E7"), ("R2", "E6"), ("flip", "E3"),
         ("flip", "E7")]
CUT = {"cut_date": "2026-10-24", "rule": "AM-22 item 2(a): the launch cutoff", "comparator": None}


def stats_artifact():
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def doc_of(inventory=None, **kw):
    st = stats_artifact()
    inv = H.open_inventory(inventory or CTX.world.inventory_path, policy=st.policy)
    return AT.build_arm_tables(st, inv, H.planned_cells(inv), input_root=CTX.world.input_root,
                               B=kw.pop("B", SMOKE_B), **kw)


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


def row(doc, x, stage):
    return next(r for r in doc["rows"] if (r["x"], r["stage"]) == (x, stage))


def load(slot_stage, slot, precision):
    """An artifact read in the smoke itself (D.load_input at the evaluator stage)."""
    w = CTX.world
    rec = next(r for r in w.binding["artifacts"]
               if (r["slot"], r["precision"], r["scope"]) == (slot, precision, "clean"))
    return D.load_input(slot_stage, D.CLEAN, rec["repo_relative_path"], w.input_root,
                        Policy.NONOFFICIAL_SMOKE)


def repo_copy(name, rel, edit=None):
    """A temporary repository root holding one record copied from the repository (optionally edited)."""
    root = CTX.world.fresh(name)
    dst = root / rel
    dst.parent.mkdir(parents=True)
    shutil.copyfile(REPO / rel, dst)
    if edit is not None:
        dst.write_bytes(edit(dst.read_bytes()))
    return root, dst


# --------------------------------------------------------------------------------------------------
def s_scale():
    n = AT.seed_noise()
    return (n["s"] == S_DL99 and n["scale"] == math.sqrt(2) * S_DL99 == n["band"] == BAND
            and n["path"] == "reports/derived/dl27_band.json" and n["dl"] == "DL-99"), str(n)


def s_boundary():
    up, down = math.nextafter(BAND, math.inf), math.nextafter(-BAND, -math.inf)
    got = [AT.reading(v, BAND) for v in (BAND, -BAND, 0.0, up, down)]
    want = [AT.READING_INSIDE] * 3 + [AT.READING_OUTSIDE] * 2
    return got == want and AT.READING_INSIDE == "not distinguishable from seed noise", str(got)


def s_rows_read():
    """Every row with a model is read beside the record's scale, with the reading B5-D1 gives (|Δ| <= √2·s
    computed here); the world's rows give both readings."""
    doc = the_doc()
    rows = [r for r in doc["rows"] if r["candidate"] is not None]
    want = [("not distinguishable from seed noise" if abs(r["delta_all_class"]) <= BAND
             else "larger than the seed-noise scale") for r in rows]
    return (len(rows) == 8 and all(r["seed_noise_scale"] == BAND for r in rows)
            and [r["reading"] for r in rows] == want and len(set(want)) == 2), \
        str([(r["contrast_id"], r["delta_all_class"], r["reading"]) for r in rows])


def t_r1_r2():
    doc = the_doc()
    t = {x["x"]: x for x in doc["teachers"]}
    ok = all((t[x]["val_x"], t[x]["ece_x"], t[x]["go"]) == v
             for x, v in (("R1", R1_VALUES), ("R2", R2_VALUES)))
    rows = all(row(doc, x, st)["teacher"]["val_x"] == t[x]["val_x"] for x, st in ORDER)
    return (ok and rows and t["R1"]["record"]["dl"] == "DL-124"
            and t["R2"]["record"]["dl"] == "DL-125"), str(t)[:300]


def t_flip():
    doc = the_doc()
    t = next(x for x in doc["teachers"] if x["x"] == "flip")
    return (t["val_x"] is None and t["ece_x"] is None and t["go"] is None
            and t["note"].startswith("not on record")
            and row(doc, "flip", "E3")["teacher"]["note"] == t["note"]), str(t)


def t_record_sha():
    """arm_record_invalid: R1's record with one byte of its generation time changed -- every field it
    reads still valid -- so only its sha256 (not DL-124's) refuses it."""
    rel = AT.TEACHER_RECORDS["R1"]["path"]
    root, _ = repo_copy("r1_record", rel,
                        lambda b: b.replace(b'"2026-10-16T06:54:28Z"', b'"2026-10-16T06:54:29Z"'))
    return refused("arm_record_invalid", lambda: AT.teacher_values("R1", repo=root))


def t_band():
    """arm_record_invalid: a seed-noise record whose band is not √2·s (read at its own sha256)."""
    rel = AT.SEED_NOISE_RECORD["path"]
    root, dst = repo_copy("band_record", rel, lambda b: b.replace(b'"band": 0.015621041685101825',
                                                                  b'"band": 0.015621041685101826'))
    rec = {**AT.SEED_NOISE_RECORD, "sha256": H.sha256_file(dst)}
    return refused("arm_record_invalid", lambda: AT.seed_noise(repo=root, record=rec))


def t_missing():
    """arm_record_invalid: a repository root without R2's record."""
    root = CTX.world.fresh("no_records")
    return refused("arm_record_invalid", lambda: AT.teacher_values("R2", repo=root))


def t_malformed():
    """arm_record_invalid: R1's record at its own (edited) sha256 whose arm is R2: read and parsed, then
    refused on its fields."""
    rel = AT.TEACHER_RECORDS["R1"]["path"]
    root, dst = repo_copy("r1_arm_r2", rel, lambda b: b.replace(b'"arm": "R1"', b'"arm": "R2"'))
    recs = {"R1": {**AT.TEACHER_RECORDS["R1"], "sha256": H.sha256_file(dst)}}
    return refused("arm_record_invalid", lambda: AT.teacher_values("R1", repo=root, records=recs))


def p_order():
    doc = the_doc()
    keys = [(r["x"], r["stage"]) for r in doc["rows"]]
    ids = [r["contrast_id"] for r in doc["rows"]]
    want = ["arm__r1__e3", "arm__r1__e7", "arm__r1__e6", "arm__r2__e3", "arm__r2__e7", "arm__r2__e6",
            "arm__flip__e3", "arm__flip__e7"]
    return keys == ORDER and ids == want, str(keys)


def p_k8():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "E3-flip:s42"))
        e.update(slot="E3-K8:s42", stage="E3-K8")
        export["entries"].append(e)
    return refused("arm_slot_invalid", lambda: doc_of(variant("k8", edit)))


def p_two():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "E3-R1:s42"))
        e["slot"] = "E3-R1:s42:b"
        export["entries"].append(e)
    return refused("arm_slot_invalid", lambda: doc_of(variant("two_r1", edit)))


def p_teacher_avg():
    def edit(export, binding):
        e = copy.deepcopy(entry(export, "R1"))
        e.update(slot="Bavg", stage="B-avg")
        export["entries"].append(e)
    return refused("arm_slot_invalid", lambda: doc_of(variant("avg_teacher", edit)))


def p_missing():
    def edit(export, binding):
        export["entries"] = [e for e in export["entries"] if e["slot"] != "E7-flip:s42"]
        drop_rows(export, binding, "E7-flip:s42")
    return refused("arm_slot_invalid", lambda: doc_of(variant("no_e7flip", edit)))


def p_other_quant():
    """A quant_descriptive slot at E6-E2 or E7-E2 is not an arm row and is not refused here."""
    doc = the_doc()
    return all(r["candidate_stage"] not in ("E6-E2", "E7-E2") for r in doc["rows"]), ""


def p_seed():
    """arm_slot_invalid: E3-R1's slot at seed 43 (the arm tables place seed 42 only)."""
    def edit(export, binding):
        entry(export, "E3-R1:s42")["seed"] = 43
    return refused("arm_slot_invalid", lambda: doc_of(variant("e3r1_s43", edit)))


def p_parent_without_model():
    """parent_rule (R-12): E3-R1 cut while a model descends from it -- E7-R1 still a derived PTQ model, or
    (E7-R1 not run) E6-R1's selection still a derived QAT model, its artifact bound."""
    def cut_e3r1(export, binding):
        entry(export, "E3-R1:s42").update(kind="cut", **CUT)
        drop_rows(export, binding, "E3-R1:s42")

    def cut_e3r1_e7r1(export, binding):
        cut_e3r1(export, binding)
        entry(export, "E7-R1:s42")["kind"] = None
        drop_rows(export, binding, "E7-R1:s42")
    out = [refused("parent_rule", lambda n=n, f=f: doc_of(variant(n, f)))
           for n, f in (("e3r1_cut_e7r1_model", cut_e3r1), ("e3r1_cut_e6r1_model", cut_e3r1_e7r1))]
    return all(ok for ok, _ in out), str([d for _, d in out])


def v_m():
    doc = the_doc()
    from src.stats.seed_stability import m_of
    r = row(doc, "R1", "E3")
    c = load("E3", "E3-R1:s42", "fp32")
    b = stats_artifact().clean("E3")
    return (r["candidate"]["m"] == m_of(c.run.stats)
            and r["baseline"]["m"] == m_of(b.run.stats)), str(r["candidate"])


def v_rule_variants():
    doc = the_doc()
    r = row(doc, "R2", "E7")
    c = load("E7", "E7-R2:s42", "int8_ptq")
    rv = rule_variants(c.run.stats)
    want = {"all_class": rv[UNION_PRESENT]["all_class"], "gt_present": rv[GT_PRESENT]["all_class"],
            "disease_only": rv[UNION_PRESENT]["disease_only"]}
    return all(r["candidate"][k]["value"] == d.value and r["candidate"][k]["value_float64"] == d.value_float64
               and r["candidate"][k]["n_classes"] == d.n_classes for k, d in want.items()), ""


def v_delta():
    doc = the_doc()
    rows = [r for r in doc["rows"] if r["candidate"] is not None]
    return all(r["delta_all_class"] == r["candidate"]["m"] - r["baseline"]["m"] for r in rows) \
        and len(rows) == 8, ""


def v_baselines():
    doc = the_doc()
    st = stats_artifact()
    return all(r["baseline"]["run_id"] == st.clean(r["stage"]).summary["run"]["run_id"]
               for r in doc["rows"]), ""


def v_per_image():
    """B5-D2: the mean on the disease-only per-image vector, its interval recomputed here."""
    doc = the_doc()
    r = row(doc, "flip", "E7")
    c = load("E7", "E7-flip:s42", "int8_ptq")
    pv = align_runs(stats_artifact().clean("E7").run, c.run, policy=Policy.NONOFFICIAL_SMOKE,
                    metric=METRIC_DISEASE_ONLY)
    ref = LD.mean_delta_task("arm__flip__e7", pv.delta, B=SMOKE_B)
    p = r["per_image"]
    return (p["metric"] == METRIC_DISEASE_ONLY and p["observed"] == ref.observed == float(np.mean(pv.delta))
            and [p["lower_bound"], p["upper_bound"]] == list(ref.bca.bounds) and p["n_paired"] == pv.n
            and p["interval_type"] == "two_sided" and p["B"] == SMOKE_B), str(p)[:300]


def v_ids():
    doc = the_doc()
    return all(r["per_image"]["analysis_id"] == "lrep_descriptive_v1"
               and r["per_image"]["contrast_id"] == r["contrast_id"]
               and r["per_image"]["statistic_name"] == "mean_delta"
               for r in doc["rows"] if r["per_image"]), ""


def v_stage_map():
    """B5-S1: E3-X is read at evaluator stage E3 (fp32), E7-X at E7 (int8_ptq), E6-X at itself (int8_qat)."""
    w = CTX.world
    st = stats_artifact()
    inv = H.open_inventory(w.inventory_path, policy=st.policy)
    got = {}
    for slot, stage in (("E3-R1:s42", "E3-R1"), ("E7-R2:s42", "E7-R2"), ("E6-R1:s42:sel", "E6-R1")):
        ba = H.load_bound(inv, slot=slot, stage=stage, precision=H.stage_spec(stage)[2], scope="clean",
                          input_root=w.input_root, policy=st.policy)
        run = ba.summary["run"]
        got[stage] = (run["stage"], run["precision"], run["model_role"])
    return got == {"E3-R1": ("E3", "fp32", "student"), "E7-R2": ("E7", "int8_ptq", "student"),
                   "E6-R1": ("E6-R1", "int8_qat", "student")}, str(got)


def v_identity():
    """R-11: an E3 family slot whose identity is not the 37's E3 is refused."""
    def edit(export, binding):
        r = next(r for r in export["models"] if (r["slot"], r["precision"]) == ("E3:s42", "fp32"))
        r["source"]["sha256"] = LF.syn_sha("another E3")
    return refused("seed42_mismatch", lambda: doc_of(variant("e3_identity", edit)))


def _cut_r2(export, binding):
    """E3-R2 cut (and so E7-R2 not run: parent cut), E6-R2's QAT run cut (DL-138's shape)."""
    e = entry(export, "E3-R2:s42")
    e.update(kind="cut", **CUT)
    drop_rows(export, binding, "E3-R2:s42")
    entry(export, "E7-R2:s42")["kind"] = None
    drop_rows(export, binding, "E7-R2:s42")
    entry(export, "E6-R2:s42").update(kind="cut", **CUT)
    entry(export, "E6-R2:s42:sel")["kind"] = None
    drop_rows(export, binding, "E6-R2:s42:sel")


def l_cut():
    doc = doc_of(variant("cut_r2", _cut_r2))
    labels = {st: row(doc, "R2", st)["label"] for st in ("E3", "E7", "E6")}
    want = {"E3": f"cut ({CUT['cut_date']}, {CUT['rule']})", "E7": "not run: parent cut",
            "E6": f"cut ({CUT['cut_date']}, {CUT['rule']})"}
    return labels == want, str(labels)


def l_not_made():
    def edit(export, binding):
        e = entry(export, "E7-flip:s42")
        e.update(kind="not_made", reason="code not merged")
        drop_rows(export, binding, "E7-flip:s42")
    doc = doc_of(variant("e7flip_not_made", edit))
    return row(doc, "flip", "E7")["label"] == "not made: code not merged", row(doc, "flip", "E7")["label"]


def l_no_values():
    doc = doc_of(variant("cut_r2", _cut_r2))
    r = row(doc, "R2", "E3")
    return (r["candidate"] is None and r["delta_all_class"] is None and r["per_image"] is None
            and r["reading"] is None and r["baseline"]["m"] is not None), str(r)[:300]


def l_teacher_status():
    doc = the_doc()
    t = {x["x"]: x for x in doc["teachers"]}
    return (t["R1"]["slot"] == "R1" and t["R1"]["kind"] == "arm_teacher" and t["R1"]["label"] is None
            and t["flip"]["slot"] is None), str(t)[:300]


def l_record_of_missing():
    """R-6: E3-flip reads 'cut' while the binding still lists its clean artifact
    (artifact_for_missing_run)."""
    def edit(export, binding):
        entry(export, "E3-flip:s42").update(kind="cut", **CUT)
        export["models"] = [r for r in export["models"] if r["slot"] != "E3-flip:s42"]
        entry(export, "E7-flip:s42")["kind"] = None
        drop_rows(export, binding, "E7-flip:s42")
    return refused("artifact_for_missing_run", lambda: doc_of(variant("flip_cut_bound", edit)))


def d_deterministic():
    return A.canonical_json(doc_of()) == A.canonical_json(doc_of()), ""


def d_no_test_keys():
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
    bad = sorted(k for k in keys
                 if k in ("p_value", "p", "wilcoxon", "holm", "reject", "t_test") or "__" in k)
    return not bad, str(bad)


def d_banner():
    doc = the_doc()
    md = AT.render_markdown(doc, "")
    return doc["banner"] == "descriptive; outside protocol 1.0.0" and doc["banner"] in "\n".join(md), ""


def d_no_test_import():
    """arm_tables imports no test or p-value object (it reads the A3a mean through lrep_descriptive)."""
    tree = ast.parse((REPO / "src" / "stats" / "arm_tables.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        mod = (node.module or "") if isinstance(node, ast.ImportFrom) else ""
        if mod and (mod.split(".")[-1] == "tests" or mod.split(".")[0] in ("scipy", "statsmodels")):
            bad.append(node.module)
        if isinstance(node, ast.Attribute) and node.attr in ("run_comparison", "holm_family", "wilcoxon"):
            bad.append(node.attr)
    return not bad, str(bad)


def d_no_test_path():
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower()]
    return not bad, str(bad[:3])


def d_cells():
    """Every line of the arm table has the header's 13 cells, with and without a model (the cut variant);
    the teachers' table has 6."""
    bad = []
    for doc in (the_doc(), doc_of(variant("cut_r2", _cut_r2))):
        md = AT.render_markdown(doc, "")
        i = md.index("| " + " | ".join(AT.HEADER) + " |")
        for line in md[i:i + 2 + 8]:
            if line.count("|") != 14:
                bad.append(line)
        j = md.index("| arm teacher | VAL_X | ECE_X | go/no-go | record | TEST slot |")
        bad += [line for line in md[j:j + 5] if line.count("|") != 7]
    return not bad and len(AT.HEADER) == 13, str(bad[:2])


def setup(root: Path, world=None) -> None:
    CTX.world = world or LF.build_world(root, strata=False)
    CTX.stats, CTX.paths, CTX.doc = None, {}, None


def cases():
    return [("S1", "√2·s from DL-99's record equals its band and 0.015621041685101825 bitwise", s_scale),
            ("S2", "B5-D1: |Δ| <= √2·s inside (inclusive), one ulp beyond outside", s_boundary),
            ("S3", "every row with a model is read beside the record's √2·s", s_rows_read),
            ("T1", "R1's and R2's VAL_X, ECE_X and go/no-go from DL-124's and DL-125's records", t_r1_r2),
            ("T2", "flip: 'not on record'", t_flip),
            ("T3", "arm_record_invalid: a go-rule record not at its sha256", t_record_sha),
            ("T4", "arm_record_invalid: a band that is not √2·s", t_band),
            ("T5", "arm_record_invalid: a record that is missing", t_missing),
            ("T6", "arm_record_invalid: a record at its sha256 whose fields are not R1's", t_malformed),
            ("P1", "the eight rows in order, no E6-flip row, the arm__ ids", p_order),
            ("P2", "arm_slot_invalid: an arm slot at E3-K8 (DL-96)", p_k8),
            ("P3", "arm_slot_invalid: two slots for one (X, stage)", p_two),
            ("P4", "arm_slot_invalid: an arm teacher outside R1 and R2", p_teacher_avg),
            ("P5", "arm_slot_invalid: a row the export has no slot for", p_missing),
            ("P6", "E6-E2 and E7-E2 are not arm rows and are not refused here", p_other_quant),
            ("P7", "arm_slot_invalid: an arm candidate at seed 43", p_seed),
            ("P8", "parent_rule: a model under a cut arm run (E7-R1, E6-R1's selection; R-12)",
             p_parent_without_model),
            ("V1", "both sides' m equal seed_stability.m_of on the artifacts", v_m),
            ("V2", "rule_variants' all-class, GT-present and disease-only values", v_rule_variants),
            ("V3", "Δ = m(candidate) - m(baseline) on all eight rows", v_delta),
            ("V4", "the comparators are the 37's E3, E7 and E6", v_baselines),
            ("V5", "B5-D2: the per-image mean Δ and its BCa interval", v_per_image),
            ("V6", "every interval in lrep_descriptive_v1 under its row's contrast id", v_ids),
            ("V7", "B5-S1: E3-X at E3, E7-X at E7, E6-X at itself", v_stage_map),
            ("V8", "R-11: the E3 family identity is checked", v_identity),
            ("L1", "a cut arm: 'cut', 'not run: parent cut', and E6-X's QAT run cut", l_cut),
            ("L2", "'not made: code not merged'", l_not_made),
            ("L3", "a row without a model has no value and no interval, and is not refused", l_no_values),
            ("L4", "the arm teachers' slots and status", l_teacher_status),
            ("L5", "artifact_for_missing_run: a cut arm whose clean artifact is still bound",
             l_record_of_missing),
            ("D1", "byte-identical JSON on repeated runs", d_deterministic),
            ("D2", "no test or p-value key, no '__' key", d_no_test_keys),
            ("D3", "the banner 'descriptive; outside protocol 1.0.0'", d_banner),
            ("D4", "arm_tables imports no test object (AST)", d_no_test_import),
            ("D5", "no created path contains 'test'", d_no_test_path),
            ("D6", "the markdown's lines have the header's cells, with and without a model", d_cells)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_arms_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_arm_tables: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'ARM TABLES OK' if good == len(results) else 'ARM TABLES FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; descriptive; outside protocol 1.0.0.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
