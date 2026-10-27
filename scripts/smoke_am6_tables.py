#!/usr/bin/env python3
"""GT-present on TEST for every model of record (lane LREP, B-3: L-AM6; B-5 (d) and the subset mode):
src/stats/am6_tables.py against AM-6, AM-17 item 1(b), STATS §9.5 and ERR E-24, the plan of record's §3,
LRB7 / LRB10 / LRB17 and rulings B5-S1 and (d) (DL-24 R3-14).

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world, plus edited or re-scored copies of one
seed-43 artifact and of the teacher's for the refusals. Created paths live under a new directory in
TMPDIR (left in place); none contains "test".

  N  the four numbers, both arithmetics, against eligibility.rule_variants
  F  R1 bitwise and R3, the union-present-only classes, the per-class table
  T  the table: roles and seeds, 21 rows from the 37 and the binding (LRB7), no qat_fakequant_off
     row, a run without a model listed with its label, the seed-42 rows equal to report.py's
  R  refusals: an unfaithful summary (gt_present_unfaithful), another ground truth
     (cross_artifact_mismatch), a record that is not canvas, a model of record without its record,
     E1's seed-43/44 records swapped (e1_checkpoint_record); and, with coverage, another ground truth
     under the teacher's record, a teacher scored as a student (precision_mismatch), a model row at
     another seed (seed_inconsistent), a stage outside the map (stage_unmapped), a teacher's model row
     that says student (precision_mismatch)
  M  B5-S1's stage map, static, against STAGE_ARTIFACTS and the evaluator's STAGES
  V  (d): every other clean canvas record as a row (23 here), each with its slot's roles, kind, stage,
     seed and label, its twelve numbers and R1/R3, the stages it is scored at, the sort, the markdown
  S  the subset mode: the four numbers recomputed on the kept rows, the eligibility changes, the
     fingerprint; zero ids removed gives the full values bitwise; a kept id outside the artifact
  D  determinism, repr floats, no 'test' path

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_am6_tables.py
"""
from __future__ import annotations

import copy
import hashlib
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
import stats_fixtures as F                                               # noqa: E402
from src.eval import artifacts as EA                                     # noqa: E402
from src.eval.evaluate import Condition                                  # noqa: E402
from src.eval.stage_artifacts import STAGE_ARTIFACTS                     # noqa: E402
from src.stats import am6_tables as AM6                                  # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import report as RP                                       # noqa: E402
from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, class_totals, rule_variants  # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, paths={}, copies={}, cov=None)
#: B5-S1, typed out from the ruling: the TMAN stages in the map's order and the evaluator stages that
#: differ from their own name (E3-X, E7-X, the teachers).
TMAN_STAGES = ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E3-R1", "E3-R2", "E3-flip", "E7-R1", "E7-R2",
               "E7-flip", "E6-E2", "E7-E2", "E6-R1", "E6-R2", "E6-KD", "A", "F", "G", "teacher", "R1", "R2")
EVALUATED_AS = {"E3-R1": "E3", "E3-R2": "E3", "E3-flip": "E3", "E7-R1": "E7", "E7-R2": "E7", "E7-flip": "E7",
                "R1": "teacher", "R2": "teacher"}
#: (d)'s rows beyond B-5's slots: the default candidates and E6-KD (Part 1's world).
PART1_COVERAGE = (("E2:s42:lambda1", "fp32"), ("E3:s42:alpha50", "fp32"), ("E6-KD:s42:sel", "int8_qat"))
SPACES = ("all_class", "disease_only")
FIELDS = (("", "value"), ("_float64", "value_float64"), ("_n_classes", "n_classes"))


def stats_artifact():
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def doc_of(inventory=None, *, coverage=False, subset=None):
    st = stats_artifact()
    inv = H.open_inventory(inventory or CTX.world.inventory_path, policy=st.policy)
    return AM6.build_am6_tables(st, inv, H.planned_cells(inv), input_root=CTX.world.input_root,
                                coverage=coverage, subset=subset)


def cov_doc():
    if CTX.cov is None:
        CTX.cov = doc_of(coverage=True)
    return CTX.cov


def summary_of(w):
    return json.loads((CTX.world.input_root / w.rel / "summary.json").read_text(encoding="utf-8"))


def stats_at(w):
    """An artifact's statistics, loaded at the stage its own summary records (independent of B5-S1)."""
    li = D.load_input(summary_of(w)["run"]["stage"], D.CLEAN, w.rel, CTX.world.input_root,
                      stats_artifact().policy)
    return li.run.stats, li.summary


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


# --------------------------------------------------------------------------------------------------
# B-5: the stage map (B5-S1), the coverage rows ((d)) and the subset mode
# --------------------------------------------------------------------------------------------------
def m_stage_map():
    """B5-S1, static: the map holds B-5's stages in order; every stage is scored at its evaluator stage
    (E3-X at E3, E7-X at E7, the teachers at "teacher", E6-KD at E6-KD, the others at their own); the
    map reaches every stage of src/eval/artifacts.STAGES and no other; each precision is
    STAGE_ARTIFACTS' (and driver.STAGE_PRECISION's for E1-E7); the teachers alone are model_role
    teacher; a stage outside the map (K8, avg, an E6-flip) is stage_unmapped."""
    m = H.STAGE_MAP
    evaluated = {st: EVALUATED_AS.get(st, st) for st in TMAN_STAGES}
    checks = {
        "keys": tuple(m) == TMAN_STAGES == H.STAGE_ORDER,
        "evaluator": all(m[st][0] == evaluated[st] for st in TMAN_STAGES),
        "reach": {spec[0] for spec in m.values()} == set(EA.STAGES),
        "precision": all(spec[2] == STAGE_ARTIFACTS["TEACHER" if spec[0] == "teacher" else spec[0]][
            "precision"] for spec in m.values()),
        "E1-E7": all(m[st] == (st, "student", D.STAGE_PRECISION[st]) for st in D.STAGES),
        "roles": all(spec[1] == ("teacher" if spec[0] == "teacher" else "student") for spec in m.values()),
        "E6-KD": m["E6-KD"] == ("E6-KD", "student", "int8_qat"),
        "unmapped": all(refused("stage_unmapped", lambda s=s: H.stage_spec(s))[0]
                        for s in ("E3-K8", "E3-avg", "E6-flip", "e3-r1", None))}
    return all(checks.values()), str({k: v for k, v in checks.items() if not v})


def v_coverage_rows():
    """(d): the rows extend to the 23 other clean canvas records (B-5's slots, the default candidates,
    E6-KD), none an fq_off or upstream row; the 21 planned rows and the missing list are those of the
    table without coverage, value for value."""
    doc, base = cov_doc(), doc_of()
    planned = [r for r in doc["rows"] if AM6.is_planned(r)]
    others = sorted((r["slot"], r["precision"]) for r in doc["rows"] if not AM6.is_planned(r))
    want = sorted([(slot, p) for slot, _, _, precs, *_ in LF.B5_SLOTS for p in precs
                   if p != H.PRECISION_FQOFF] + list(PART1_COVERAGE))
    return (others == want and len(others) == 23 and doc["missing"] == base["missing"]
            and A.canonical_json(planned) == A.canonical_json(base["rows"])), \
        str(sorted(set(others) ^ set(want))[:5])


def v_identity():
    """Each coverage row carries its slot's roles, kind, stage and seed (its entry's), the IF§11.2 label
    None (a scored slot carries its value), its precision, run_id and checkpoint (its artifact's)."""
    entries = {e["slot"]: e for e in CTX.world.export["entries"]}
    bad = []
    for r in cov_doc()["rows"]:
        if AM6.is_planned(r):
            continue
        e, w = entries[r["slot"]], CTX.world.written[(r["slot"], r["precision"], "clean", None)]
        if not (r["roles"] == e["role"] and r["kind"] == e["kind"] and r["stage"] == e["stage"]
                and r["seed"] == e["seed"] and r["label"] is None and r["run_id"] == w.run_id
                and r["checkpoint_sha256"] == w.checkpoint and r["source"] == AM6.COVERAGE_SOURCE
                and "role" not in r):
            bad.append(r["slot"])
    return not bad, str(bad)


def v_values():
    """Each coverage row's twelve numbers equal rule_variants on its artifact's statistics, loaded here at
    the stage its summary records; R1 (float32 equal to summary.json's, the default candidates' among
    them) and R3 hold."""
    bad, n = [], 0
    for r in cov_doc()["rows"]:
        if AM6.is_planned(r):
            continue
        n += 1
        stats, summary = stats_at(CTX.world.written[(r["slot"], r["precision"], "clean", None)])
        rv, dl = rule_variants(stats), summary["dataset_level"]
        same = all(r[f"{rule}_{s}{sfx}"] == getattr(rv[rule][s], attr) for rule in (GT_PRESENT, UNION_PRESENT)
                   for s in SPACES for sfx, attr in FIELDS)
        r1 = (r["union_present_all_class"] == dl["all_class_miou"]
              and r["union_present_disease_only"] == dl["disease_only_miou"])
        if not (same and r1 and r["R3"]["holds"]):
            bad.append(r["slot"])
    return not bad and n == 23, f"{bad} of {n}"


def v_stages():
    """B5-S1 in the rows: E6-KD's artifact is scored at E6-KD, the arms' at E3 and E7, the teachers' at
    "teacher" with model_role teacher and a null seed; the row's stage is TMAN's."""
    rows = {r["slot"]: r for r in cov_doc()["rows"] if not AM6.is_planned(r)}
    run = {slot: summary_of(CTX.world.written[(slot, rows[slot]["precision"], "clean", None)])["run"]
           for slot in rows}
    want = {"E6-KD:s42:sel": ("E6-KD", "student"), "E3-R1:s42": ("E3", "student"),
            "E3-flip:s42": ("E3", "student"), "E7-R2:s42": ("E7", "student"), "A:s42": ("A", "student"),
            "E7-E2:s42": ("E7-E2", "student"), "teacher": ("teacher", "teacher"),
            "R1": ("teacher", "teacher"), "R2": ("teacher", "teacher")}
    got = {slot: (run[slot]["stage"], run[slot]["model_role"]) for slot in want}
    nulls = sorted(slot for slot, r in rows.items() if r["seed"] is None)
    return (got == want and nulls == ["R1", "R2", "teacher"] and rows["E6-KD:s42:sel"]["stage"] == "E6-KD"
            and rows["R1"]["stage"] == "R1"), str(got)


def v_sort():
    """The sort covers every stage and a null seed: seed (null last), then the map's stage order, then
    the slot."""
    rows = cov_doc()["rows"]
    keys = [(r["seed"] is None, r["seed"] or 0, TMAN_STAGES.index(r["stage"]), r["slot"]) for r in rows]
    tail = [r["seed"] for r in rows[-3:]]
    return keys == sorted(keys) and tail == [None] * 3 and rows[0]["slot"] == "E1:s42", \
        str([r["slot"] for r in rows[:4]])


def v_markdown():
    """The planned table renders as without coverage; a second table, under the banner, lists the 23
    rows with their roles and precision (a null seed as —); no slot key with ':' is printed (IF§2)."""
    md = AM6.render_markdown(cov_doc(), "BANNER")
    head = AM6.render_markdown(doc_of(), "BANNER")[:-3]
    i = md.index(f"### {AM6.COVERAGE_TITLE}")
    rows = [line for line in md[i:] if line.startswith("| ") and not line.startswith("| stage")]
    slots = {e["slot"] for e in CTX.world.export["entries"] if ":" in e["slot"]}
    text = "\n".join(md)
    return (md[:len(head)] == head and md[i + 1:i + 3] == ["", "BANNER"] and len(rows) == 23
            and "| teacher | — | teacher_reference | fp32 |" in text
            and "| R2 | — | arm_teacher | fp32 |" in text
            and "| E6-E2 | 42 | quant_descriptive | int8_qat |" in text
            and not any(s in text for s in slots)), str(rows[:2])


def r_cov_other_gt():
    """With coverage, the teacher's record pointed at an artifact scored on another ground truth (its
    checkpoint, stage and model_role the teacher's): one ground truth holds across every row (R-10)."""
    def edit(export, binding):
        point_record(binding, "teacher", "fp32", CTX.copies["teacher_other_gt"])
    return refused("cross_artifact_mismatch",
                   lambda: doc_of(variant("teacher_other_gt", edit), coverage=True))


def r_cov_role():
    """B5-S1: the teacher's artifact recording model_role student is refused (precision_mismatch)."""
    def edit(export, binding):
        point_record(binding, "teacher", "fp32", CTX.copies["teacher_as_student"])
    return refused("precision_mismatch", lambda: doc_of(variant("teacher_as_student", edit), coverage=True))


def r_cov_seed():
    """R-14 for a coverage row: A's model row at seed 43 while its entry is at 42."""
    def edit(export, binding):
        next(r for r in export["models"] if (r["slot"], r["precision"]) == ("A:s42", "fp32"))["seed"] = 43
    return refused("seed_inconsistent", lambda: doc_of(variant("a_row_s43", edit), coverage=True))


def r_cov_unmapped():
    """B5-S1 through (d): a slot whose stage is outside the map (E3-K8) is refused, never skipped."""
    def edit(export, binding):
        next(e for e in export["entries"] if e["slot"] == "E3-R1:s42")["stage"] = "E3-K8"
        next(r for r in export["models"] if r["slot"] == "E3-R1:s42")["stage"] = "E3-K8"
    return refused("stage_unmapped", lambda: doc_of(variant("e3r1_k8", edit), coverage=True))


def r_cov_row_role():
    """B5-S1 for the model row: the teacher's fp32 row says model_role student while its artifact is the
    teacher's (precision_mismatch; DL-24 B5R1-05)."""
    def edit(export, binding):
        row = next(r for r in export["models"] if (r["slot"], r["precision"]) == ("teacher", "fp32"))
        row["model_role"] = "student"
    return refused("precision_mismatch", lambda: doc_of(variant("teacher_row_student", edit), coverage=True))


def own_subset(stats, kept):
    """The kept rows' totals, rebuilt here: image_index -> manifest_ids, then class_totals."""
    keep = {i for i, m in enumerate(stats.manifest_ids) if m in set(kept)}
    mask = np.array([int(ix) in keep for ix in np.asarray(stats.image_index)], dtype=bool)
    return class_totals({k: np.asarray(getattr(stats, k))[mask] for k in ("class_id", "tp", "gt", "pred")},
                        num_classes=116)


def s_subset_values():
    """The subset mode on the world's kept ids: each row's twelve numbers equal rule_variants on the kept
    rows' totals rebuilt here, the eligibility changes are the eligible sets' differences against the full
    split (some class drops out here), 26 kept and 4 removed, and the fingerprint is the sha256 of the
    sorted ids joined by LF."""
    kept = CTX.world.dedup["kept_ids"]
    doc = doc_of(subset=kept)
    bad, dropped = [], 0
    for r in doc["rows"]:
        if r["seed"] == 42:
            stats = stats_artifact().clean(r["stage"]).run.stats
        else:
            stats, _ = stats_at(CTX.world.art(r["stage"], r["seed"]))
        sub, full = rule_variants(own_subset(stats, kept)), rule_variants(stats)
        s = r["subset"]
        same = all(s[f"{rule}_{sp}{sfx}"] == getattr(sub[rule][sp], attr)
                   for rule in (GT_PRESENT, UNION_PRESENT) for sp in SPACES for sfx, attr in FIELDS)
        changes = {}
        for rule in (UNION_PRESENT, GT_PRESENT):
            f_el, s_el = set(full[rule]["all_class"].eligible), set(sub[rule]["all_class"].eligible)
            changes[rule] = {"dropped": sorted(f_el - s_el), "added": sorted(s_el - f_el)}
        fp = hashlib.sha256("\n".join(sorted(stats.manifest_ids)).encode("utf-8")).hexdigest()
        dropped += len(changes[GT_PRESENT]["dropped"])
        if not (same and s["eligibility_changes"] == changes and s["n_images_kept"] == 26
                and s["n_images_removed"] == 4 and s["image_ids_sha256"] == fp):
            bad.append((r["stage"], r["seed"]))
    return not bad and dropped > 0 and len(doc["rows"]) == 21, f"{bad} dropped={dropped}"


def s_zero_removed():
    """Guard 1, (i)'s half: with every id kept, each row's subset values equal its full values bitwise and
    nothing changes eligibility."""
    doc = doc_of(coverage=True, subset=list(CTX.world.truth.ids))
    keys = [f"{rule}_{sp}{sfx}" for rule in (GT_PRESENT, UNION_PRESENT) for sp in SPACES for sfx, _ in FIELDS]
    bad = [r["slot"] for r in doc["rows"]
           if any(r["subset"][k] != r[k] for k in keys) or r["subset"]["n_images_removed"] != 0
           or any(v["dropped"] or v["added"] for v in r["subset"]["eligibility_changes"].values())]
    return not bad and len(doc["rows"]) == 44, str(bad[:3])


def s_outside():
    """A kept id that is not an image id of the artifact is refused (dedup_invalid; B5-S2)."""
    return refused("dedup_invalid", lambda: doc_of(subset=[*CTX.world.dedup["kept_ids"], "img_99999"]))


def setup(root: Path, world=None) -> None:
    w = world or LF.build_world(root, strata=False)
    CTX.world, CTX.stats, CTX.paths, CTX.cov = w, None, {}, None
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
    # B-5 (d): the teacher's artifact on another ground truth, and the teacher scored as a student
    tw = w.written[("teacher", "fp32", "clean", None)]
    rel_t = "runs/heldout/teacher_clean_othergt"
    scored = F.score(other, F.predict(other, quality=0.8, seed=F.int_seed(LF.WORLD_SEED, rel_t)),
                     Condition(*D.CLEAN))
    F.write_fixture(w.input_root / rel_t, scored, w.n, stage="teacher", condition=Condition(*D.CLEAN),
                    split="val", status="provisional", repo_root=w.input_root,
                    run_id="lrep-teacher-othergt", precision="fp32", checkpoint_sha256=tw.checkpoint)
    F.edit_summary(w.input_root / rel_t, lambda s: {**s, "run": {**s["run"], "model_role": "teacher"}})
    summary = json.loads((w.input_root / rel_t / "summary.json").read_text(encoding="utf-8"))
    CTX.copies["teacher_other_gt"] = LF.Written(
        slot="teacher", stage="teacher", seed=None, precision="fp32", scope="clean", cell=None, rel=rel_t,
        checkpoint=tw.checkpoint, files_sha256=D.input_file_sha256(w.input_root / rel_t),
        run_id="lrep-teacher-othergt", split_manifest_sha256=summary["dataset"]["split_manifest_sha256"])
    CTX.copies["teacher_as_student"] = LF.edited_copy(
        w, tw, "teacher_clean_asstudent", lambda s: {**s, "run": {**s["run"], "model_role": "student"}})


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
            ("M1", "B5-S1: the stage map against STAGE_ARTIFACTS and the evaluator's STAGES", m_stage_map),
            ("V1", "(d): the 23 other clean canvas records are rows; the 21 planned rows unchanged",
             v_coverage_rows),
            ("V2", "(d): each row's roles, kind, stage, seed, label, run_id and checkpoint", v_identity),
            ("V3", "(d): each row's twelve numbers equal rule_variants'; R1 and R3 hold", v_values),
            ("V4", "(d): E6-KD scored at E6-KD, the arms at E3/E7, the teachers at teacher (B5-S1)",
             v_stages),
            ("V5", "(d): the sort covers every stage and a null seed", v_sort),
            ("V6", "(d): the second table, its banner, a null seed as —, no slot key", v_markdown),
            ("R6", "cross_artifact_mismatch: the teacher's record on another ground truth", r_cov_other_gt),
            ("R7", "precision_mismatch: the teacher scored as a student (B5-S1)", r_cov_role),
            ("R8", "seed_inconsistent: a coverage row's model row at another seed (R-14)", r_cov_seed),
            ("R9", "stage_unmapped: a coverage slot at a stage outside the map", r_cov_unmapped),
            ("R10", "precision_mismatch: the teacher's model row says student (B5-S1)", r_cov_row_role),
            ("S1", "subset: the four numbers recomputed on the kept rows, the changes, the fingerprint",
             s_subset_values),
            ("S2", "subset, guard 1 (i): nothing removed gives the full values bitwise", s_zero_removed),
            ("S3", "dedup_invalid: a kept id outside the artifact", s_outside),
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
