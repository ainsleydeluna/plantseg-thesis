#!/usr/bin/env python3
"""Duplicate-free TEST (lane LREP, B-5 (b)): src/stats/dedup_subset.py and B5-S2's reader against AM-18
item 8(c) (PA :1215-1231), TL-13, LRB17, rulings B5-D3, B5-D4 and B5-S2 and LREP-B5-RULE-2 item 4.

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world with its dedup file (four of thirty images
removed, the per-reason counts overlapping by one), variants of the export and the dedup file under new
directories in TMPDIR, and one pair of artifacts with three AM-5 images for the Am5Pair (all left in
place); none contains "test" or ":". Every expected value is recomputed here from the artifacts with the
frozen functions (never read from the module under test), except that T4 applies dedup_subset's label_of
and compare to the part's rows, rules that T2 and T3 pin on typed hand rows.

  I  B5-S2: the reader (path, sha256 first, strict JSON, T4's fields) and dedup_invalid on each failure;
     the invariants against the scored artifacts (one id set, kept inside it, |all| - |kept| <= the sum)
  V  (i): the rows read (every model of record and every arm, E6-KD), their four numbers on the subset
     with eligibility recomputed, the eligibility changes, the counts; no other slot is read (LRB4)
  C  (ii): the eight contrasts on the subset: the restricted vectors and run_comparison, robustness
     through assemble_miou_c and align_miou_c, the dataset-level differences, the Am5Pair on the kept
     rows, the 31 intervals beside the official ones (lrep_descriptive_v1)
  G  guard 1 (nothing removed: (i) and (ii) equal the full results bitwise) and guard 2 (Policy.OFFICIAL)
  N  (iii): the E3-vs-E6 check on the kept rows
  T  (iv) B5-D3: the strict unadjusted decision, the labels, "not evaluable", the disagreements, the
     official Holm decision and label unchanged
  R  (v): removed_by_reason beside the issue's counts, as string values
  D  determinism, no Holm object, the banner, no 'test' path, the measured time at B = PRODUCTION_B
  O  B-5b (ruling LREP-B5-RULE-4 R4-1): item 5, a DT run given to (ii) and (iii) is refused
     (descriptive_input_in_official); items 2 and 6, the E7-X rows' read and DT label

Cases are importable (setup / cases) so that scripts/smoke_lrep_mutations.py can run them mutated.

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_dedup_subset.py
"""
from __future__ import annotations

import ast
import copy
import dataclasses
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
from src.eval.artifacts import am5_excluded_ids_sha256                   # noqa: E402
from src.eval.evaluate import Condition                                  # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import bootstrap as BS                                    # noqa: E402
from src.stats import dedup_subset as DS                                 # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import lrep_descriptive as LD                             # noqa: E402
from src.stats import report as RP                                       # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, align_runs              # noqa: E402
from src.stats.corruption_protocol import official_corruption_grid       # noqa: E402
from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, class_totals, rule_variants  # noqa: E402
from src.stats.ingest import Policy, am5_pair                            # noqa: E402
from src.stats.noninferiority import (PooledStages, build_non_inferiority,  # noqa: E402
                                      pooled_miou_from_totals)
from src.stats.robustness import INFERENTIAL_SEVERITIES, align_miou_c, assemble_miou_c  # noqa: E402

CTX = types.SimpleNamespace(world=None, stats=None, paths={}, doc=None, am5=None, notes={})
SMOKE_B = 60
#: The issue's rows and totals, typed out from PA :1196-1204 (never read from the module under test).
ISSUE = [("train ↔ train", 124), ("test ↔ train", 70), ("train ↔ val", 32), ("test ↔ test", 13),
         ("test ↔ val", 11), ("3-way and more", 15)]
SPACES = ("all_class", "disease_only")
FIELDS = (("", "value"), ("_float64", "value_float64"), ("_n_classes", "n_classes"))
#: Cases whose detail is printed even when they pass (the guard's outcome and the measured time).
NOTE_CASES = ("G2", "D5")
#: R4-1 item 6, typed from the ruling: what every row built from a DT run prints.
DT_LABEL = "descriptive TEST; provisional artifact (the evaluator scores this stage descriptively only)"


def stats_artifact():
    if CTX.stats is None:
        CTX.stats = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    return CTX.stats


def inventory(path=None):
    return H.open_inventory(path or CTX.world.inventory_path, policy=stats_artifact().policy)


def doc_of(path=None, B=SMOKE_B, **kw):
    inv = inventory(path)
    return DS.build_dedup(stats_artifact(), inv, H.planned_cells(inv), input_root=CTX.world.input_root,
                          B=B, **kw)


def the_doc():
    if CTX.doc is None:
        CTX.doc = doc_of()
    return CTX.doc


def kept():
    return sorted(set(CTX.world.truth.ids) - set(LF.DEDUP_REMOVED))


def variant(name, edit=None, dedup=None, dedup_raw=None):
    """A variant export: `edit` on copies of the world's export and binding; `dedup` (a document) or
    `dedup_raw` (bytes) as its dedup file, the export's dedup.sha256 set from those bytes unless `edit`
    pinned it."""
    if name not in CTX.paths:
        export, binding = copy.deepcopy(CTX.world.export), copy.deepcopy(CTX.world.binding)
        if edit is not None:
            edit(export, binding)
        d = CTX.world.fresh(name)
        doc = CTX.world.dedup if dedup is None else dedup
        if dedup_raw is not None:
            (d / LF.DEDUP_NAME).write_bytes(dedup_raw)
            if isinstance(export.get("dedup"), dict) and export["dedup"].get("sha256") is None:
                export["dedup"]["sha256"] = hashlib.sha256(dedup_raw).hexdigest()
            doc = None
        CTX.paths[name] = LF.write_inventory(d, export, binding, doc)
    return CTX.paths[name]


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:300]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def clean_run(stage):
    return stats_artifact().clean(stage).run


def own_totals(stats, ids):
    """The kept rows' class totals, rebuilt here: image_index -> manifest_ids, then class_totals."""
    keep = {i for i, m in enumerate(stats.manifest_ids) if m in set(ids)}
    mask = np.array([int(ix) in keep for ix in np.asarray(stats.image_index)], dtype=bool)
    return class_totals({k: np.asarray(getattr(stats, k))[mask] for k in ("class_id", "tp", "gt", "pred")},
                        num_classes=116)


def own_pair(cid, ids):
    """The clean contrast's full pair (align_runs here), restricted here by id."""
    b, c, metric = D.COMPARISON_TABLE[cid]
    pv = align_runs(clean_run(b), clean_run(c), policy=stats_artifact().policy, metric=metric)
    keep = [i in set(ids) for i in pv.image_ids]
    return pv, [i for i, k in zip(pv.image_ids, keep) if k], np.asarray(pv.delta)[np.asarray(keep)]


def own_dense(cid, ids):
    """The contrast's PooledStages on the kept rows, built here from each run's densified rows."""
    from src.stats.noninferiority import densify
    b, c, _ = D.COMPARISON_TABLE[cid]
    rb, rc = clean_run(b), clean_run(c)
    rows = [i for i, m in enumerate(rb.stats.manifest_ids) if m in set(ids)]
    db = densify(rb.stats, len(rb.stats.manifest_ids), 116)
    dc = densify(rc.stats, len(rc.stats.manifest_ids), 116)
    return PooledStages.from_dense(tuple(a[rows] for a in db), tuple(a[rows] for a in dc))


def own_miou_c(stage, ids):
    """mC on the kept rows (DL-73/P28's nested mean of pooled cell values), rebuilt here."""
    st = stats_artifact()
    per = []
    for name in official_corruption_grid().names:
        cells = []
        for sev in INFERENTIAL_SEVERITIES:
            t = own_totals(st.cell(stage, name, sev).run.stats, ids)
            cells.append(pooled_miou_from_totals(t.tp, t.gt, t.pred))
        per.append(float(np.mean(np.array(cells, dtype=np.float64))))
    return float(np.mean(np.array(per, dtype=np.float64)))


def contrast(doc, cid):
    return next(c for c in doc["contrasts"] if c["comparison_id"] == cid)


# --------------------------------------------------------------------------------------------------
# I: B5-S2's reader and the invariants
# --------------------------------------------------------------------------------------------------
def i_read():
    """The world's record: path, the file's sha256 (checked before parsing), the 26 kept ids, the
    per-reason counts, their sum 5 and no other field."""
    rec = H.read_dedup(inventory())
    raw = (CTX.world.inventory_path.parent / LF.DEDUP_NAME).read_bytes()
    return (rec.path == "tday_dedup.json" and rec.sha256 == hashlib.sha256(raw).hexdigest()
            and list(rec.kept_ids) == kept() and len(rec.kept_ids) == 26
            and rec.removed_by_reason == (("equals_train_image", 2), ("equals_val_image", 2),
                                          ("repeats_an_earlier_image", 1))
            and rec.removed_total == 5 and rec.other_fields == ()), str(rec)[:300]


def i_null():
    def edit(export, binding):
        export["dedup"] = None
    return refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_null", edit))))


def i_fields():
    def edit(export, binding):
        del export["dedup"]["removed_by_reason"]
    return refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_no_reasons", edit))))


def i_path():
    """A dedup.path that is not plain relative POSIX is refused before anything is read."""
    def edit(export, binding):
        export["dedup"]["path"] = "../tday_dedup.json"
        export["dedup"]["sha256"] = LF.syn_sha("never read")
    return refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_dotdot", edit))))


def i_missing():
    def edit(export, binding):
        export["dedup"]["path"] = "tday_dedup_absent.json"
        export["dedup"]["sha256"] = LF.syn_sha("absent")
    return refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_absent", edit))))


def i_sha():
    def edit(export, binding):
        export["dedup"]["sha256"] = LF.syn_sha("another dedup file")
    return refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_sha", edit))))


def i_utf8():
    raw = b'{"kept_ids": ["img\xff"]}'
    return refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_latin1", dedup_raw=raw))))


def i_strict():
    """A repeated key, and a NaN, are refused by the strict parse."""
    raw1 = b'{"kept_ids": ["img_00000"], "kept_ids": ["img_00001"], "removed_by_reason": {}}'
    raw2 = b'{"kept_ids": ["img_00000"], "removed_by_reason": {"x": NaN}}'
    a = refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_dupkey", dedup_raw=raw1))))
    b = refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_nan", dedup_raw=raw2))))
    return a[0] and b[0], f"{a[1]} | {b[1]}"


def i_ids():
    """kept_ids unsorted, repeated or empty are refused (T4: sorted and unique)."""
    reasons = dict(LF.DEDUP_REASONS)
    out = []
    for name, ids in (("dedup_unsorted", ["img_00001", "img_00000"]), ("dedup_repeat", ["img_00000"] * 2),
                      ("dedup_empty", [])):
        out.append(refused("dedup_invalid", lambda n=name, i=ids: H.read_dedup(inventory(
            variant(n, dedup={"kept_ids": i, "removed_by_reason": reasons})))))
    return all(ok for ok, _ in out), str([d for _, d in out])


def i_reasons():
    """The file's removed_by_reason must be the export's."""
    doc = {"kept_ids": kept(), "removed_by_reason": {"equals_train_image": 3}}
    return refused("dedup_invalid", lambda: H.read_dedup(inventory(variant("dedup_reasons", dedup=doc))))


def i_extra():
    """A field LREP does not read is reported, never refused."""
    doc = {**CTX.world.dedup, "groups": [["img_00003", "img_00011"]]}
    rec = H.read_dedup(inventory(variant("dedup_extra", dedup=doc)))
    return rec.other_fields == ("groups",) and list(rec.kept_ids) == kept(), str(rec.other_fields)


def i_invariants():
    """check_ids: a fingerprint that is not the 37's, a kept id outside them, and fewer counted than
    removed are each dedup_invalid; the world's counts and overlap are reported."""
    rec = H.read_dedup(inventory())
    ids = CTX.world.truth.ids
    good = {"E1 clean": DS.AM6.ids_sha256(ids)}
    counts = DS.check_ids(rec, good, ids)
    a = refused("dedup_invalid", lambda: DS.check_ids(rec, {**good, "X": DS.AM6.ids_sha256(ids[:-1])}, ids))
    b = refused("dedup_invalid", lambda: DS.check_ids(dataclasses.replace(rec, kept_ids=(*rec.kept_ids,
                                                                                         "img_00099")),
                                                      good, ids))
    c = refused("dedup_invalid", lambda: DS.check_ids(dataclasses.replace(rec, removed_by_reason=(("x", 3),)),
                                                      good, ids))
    ok = (counts["n_all"], counts["n_kept"], counts["n_removed"], counts["sum_removed_by_reason"],
          counts["overlap"], counts["partition_stated"]) == (30, 26, 4, 5, 1, False)
    return ok and a[0] and b[0] and c[0], f"{counts} | {a[1]} | {b[1]} | {c[1]}"


def i_short_count():
    """Through the part: a dedup file whose reasons count fewer images than it removes."""
    doc = {"kept_ids": kept(), "removed_by_reason": {"equals_train_image": 3}}

    def edit(export, binding):
        export["dedup"]["removed_by_reason"] = {"equals_train_image": 3}
    return refused("dedup_invalid", lambda: doc_of(variant("dedup_short", edit, dedup=doc)))


# --------------------------------------------------------------------------------------------------
# V: (i)
# --------------------------------------------------------------------------------------------------
def v_rows():
    """The rows read: the 21 planned, the arms' students and teachers (A, F and G among them: their
    catalogue role is arm, R4-5(b)), the teacher of record, AM-23's quant_descriptive slots and E6-KD (role
    e6kd) -- 38 in this world; never a default candidate, a control or a calibration-sensitivity model, and
    never a not-built L slot (no model)."""
    roles = ("family", "per_seed", "arm", "arm_teacher", "quant_descriptive", "teacher_reference")
    want = sorted(e["slot"] for e in CTX.world.export["entries"]
                  if (set(e["role"]) & set(roles) or e["stage"] == "E6-KD")
                  and e["kind"] in ("scored_run", "derived", "arm_teacher"))
    got = sorted(m["slot"] for m in the_doc()["models"])
    return (got == want and len(got) == 38
            and {"A:s42:alpha25:80k", "F:s42:alpha25:80k", "G:s42:80k"} <= set(got)), \
        str(sorted(set(got) ^ set(want)))


def v_values():
    """Each row's twelve numbers on the subset equal rule_variants on the kept rows' totals rebuilt here,
    and its full values the artifact's own."""
    st = stats_artifact()
    bad = []
    for m in the_doc()["models"]:
        if m["stage"] in D.STAGES and m["seed"] == 42 and m["slot"] == LF.record_slot(m["stage"], 42):
            stats = st.clean(m["stage"]).run.stats
        else:
            w = next(v for (slot, prec, scope, cell), v in CTX.world.written.items()
                     if slot == m["slot"] and scope == "clean" and prec != H.PRECISION_FQOFF)
            summary = json.loads((CTX.world.input_root / w.rel / "summary.json").read_text(encoding="utf-8"))
            stats = D.load_input(summary["run"]["stage"], D.CLEAN, w.rel, CTX.world.input_root,
                                 st.policy).run.stats
        sub, full = rule_variants(own_totals(stats, kept())), rule_variants(stats)
        for rule in (GT_PRESENT, UNION_PRESENT):
            for sp in SPACES:
                for sfx, attr in FIELDS:
                    if (m["subset"][f"{rule}_{sp}{sfx}"] != getattr(sub[rule][sp], attr)
                            or m["full"][f"{rule}_{sp}{sfx}"] != getattr(full[rule][sp], attr)):
                        bad.append(m["slot"])
    return not bad, str(sorted(set(bad))[:5])


def v_changes():
    """The eligibility changes are dedup_val_scores.py:159's: per rule, the all-class eligible classes
    dropped and added; the removed images take some classes out here."""
    bad, dropped = [], 0
    for m in the_doc()["models"]:
        ch = m["subset"]["eligibility_changes"]
        dropped += len(ch[GT_PRESENT]["dropped"])
        if (set(ch) != {UNION_PRESENT, GT_PRESENT}
                or any(sorted(v["dropped"]) != v["dropped"] for v in ch.values())):
            bad.append(m["slot"])
        if m["subset"]["n_images_kept"] != 26 or m["subset"]["n_images_removed"] != 4:
            bad.append(m["slot"])
    return not bad and dropped > 0, f"{bad[:3]} dropped={dropped}"


def v_counts():
    d = the_doc()["dedup"]
    every = hashlib.sha256("\n".join(sorted(CTX.world.truth.ids)).encode()).hexdigest()
    return ((d["n_all"], d["n_kept"], d["n_removed"], d["sum_removed_by_reason"], d["overlap"])
            == (30, 26, 4, 5, 1) and d["image_ids_sha256"] == every
            and d["kept_ids_sha256"] == hashlib.sha256("\n".join(kept()).encode()).hexdigest()), str(d)[:300]


def v_unreported_slot():
    """LRB4: the part reads only the rows it reports. The 160k E2 control's binding record off its files
    (an R-7 defect in a slot (b) does not report) refuses the am6 part with coverage
    (artifact_record_mismatch), while the duplicate-free part is built with its 38 rows (DL-24 B5R3-02)."""
    def edit(export, binding):
        r = next(r for r in binding["artifacts"]
                 if (r["slot"], r["precision"], r["scope"]) == ("E2:s42:lambda0.5:160k", "fp32", "clean"))
        r["files_sha256"][sorted(r["files_sha256"])[0]] = LF.syn_sha("not the control's file")
    path = variant("control_off_record", edit)
    inv = inventory(path)
    am6 = refused("artifact_record_mismatch", lambda: DS.AM6.build_am6_tables(
        stats_artifact(), inv, H.planned_cells(inv), input_root=CTX.world.input_root, coverage=True))
    doc = doc_of(path)
    return am6[0] and len(doc["models"]) == 38, am6[1]


# --------------------------------------------------------------------------------------------------
# C: (ii)
# --------------------------------------------------------------------------------------------------
def c_order():
    doc = the_doc()
    ids = [c["comparison_id"] for c in doc["contrasts"]]
    return ids == list(D.CLEAN_COMPARISON_IDS) + [D.ROBUSTNESS_ID] and len(ids) == 8, str(ids)


def c_vectors():
    """Each clean contrast's subset vector is the full align_runs vector restricted by id here, and its
    comparison is run_comparison on that vector (serialized equal)."""
    from src.stats.tests import run_comparison
    bad = []
    for cid in D.CLEAN_COMPARISON_IDS:
        b, c, _ = D.COMPARISON_TABLE[cid]
        pv, ids, delta = own_pair(cid, kept())
        sub = dataclasses.replace(pv, image_ids=tuple(ids), baseline=pv.baseline[np.isin(pv.image_ids, ids)],
                                  candidate=pv.candidate[np.isin(pv.image_ids, ids)], delta=delta)
        want = A.serialize_a3a(run_comparison(cid, sub, baseline_stage=b, candidate_stage=c))
        got = contrast(the_doc(), cid)["subset"]
        if got["comparison"] != want or got["n_paired"] != len(ids) or got["am5"] is not None:
            bad.append(cid)
    return not bad, str(bad)


def c_pooled():
    """Each clean contrast's dataset-level difference is the pooled delta of its kept rows, rebuilt here
    (cand - base of pooled mIoU on the kept totals)."""
    bad = []
    for cid in D.CLEAN_COMPARISON_IDS:
        b, c, _ = D.COMPARISON_TABLE[cid]
        tb, tc = own_totals(clean_run(b).stats, kept()), own_totals(clean_run(c).stats, kept())
        want = pooled_miou_from_totals(tc.tp, tc.gt, tc.pred) - pooled_miou_from_totals(tb.tp, tb.gt, tb.pred)
        if contrast(the_doc(), cid)["subset"]["dataset_level_difference"] != want:
            bad.append(cid)
    return not bad, str(bad)


def c_robustness():
    """robustness_e1_e6: the mIoU-C vector of assemble_miou_c and align_miou_c restricted by clean id
    here, run_comparison on it, and the dataset-level difference mC(E6) - mC(E1) on the kept rows."""
    from src.stats.tests import run_comparison
    st, grid = stats_artifact(), official_corruption_grid()
    mc = {s: assemble_miou_c([st.cell(s, n, v).run for n in grid.names for v in INFERENTIAL_SEVERITIES], grid,
                             policy=st.policy, stage=s, clean=clean_run(s)) for s in ("E1", "E6")}
    pv = align_miou_c(mc["E1"], mc["E6"], policy=st.policy)
    keep = np.isin(pv.image_ids, kept())
    sub = dataclasses.replace(pv, image_ids=tuple(np.asarray(pv.image_ids)[keep]), baseline=pv.baseline[keep],
                              candidate=pv.candidate[keep], delta=pv.delta[keep])
    want = A.serialize_a3a(run_comparison(D.ROBUSTNESS_ID, sub, baseline_stage="E1", candidate_stage="E6"))
    got = contrast(the_doc(), D.ROBUSTNESS_ID)
    diff = own_miou_c("E6", kept()) - own_miou_c("E1", kept())
    return (got["subset"]["comparison"] == want and got["subset"]["dataset_level_difference"] == diff
            and got["metric"] == "per_image_miou_c"), f"{got['subset']['dataset_level_difference']} {diff}"


def c_intervals():
    """The 31 tasks of the eight contrasts on the subset, ids dedup__<comparison_id> with the official
    statistic names and interval types, B as given; one of each kind recomputed here; the official
    intervals beside them are the family's."""
    doc = the_doc()
    fam = stats_artifact().family
    tasks = [(c["comparison_id"], i) for c in doc["contrasts"] for i in c["subset"]["intervals"]]
    spec = {(t.comparison_id, t.statistic_name): t.interval_type for t in BS.FROZEN_TASK_MATRIX}
    names_ok = (len(tasks) == 31 and all(i["contrast_id"] == f"dedup__{cid}" and i["B"] == SMOKE_B
                                         and spec[(cid, i["statistic_name"])] == i["interval_type"]
                                         and i["analysis_id"] == "lrep_descriptive_v1" for cid, i in tasks))
    cid = "accuracy_e7_e6"
    _, _, delta = own_pair(cid, kept())
    rf, jf = BS.scalar_callables(BS.HL_SHIFT, delta)
    got = next(i for i in contrast(doc, cid)["subset"]["intervals"] if i["statistic_name"] == BS.HL_SHIFT)
    ref = LD.descriptive_task(f"dedup__{cid}", BS.HL_SHIFT, BS.TWO_SIDED, delta.size, SMOKE_B,
                              contrast(doc, cid)["subset"]["comparison"]["hodges_lehmann_shift"], rf, jf)
    ps = own_dense(cid, kept())
    pref = LD.descriptive_task(f"dedup__{cid}", BS.DATASET_MIOU_DELTA, BS.TWO_SIDED, ps.n_images, SMOKE_B,
                               ps.observed(), ps.replicate, ps.jackknife)
    pgot = next(i for i in contrast(doc, cid)["subset"]["intervals"]
                if i["statistic_name"] == BS.DATASET_MIOU_DELTA)
    off = {t["statistic_name"]: t for t in fam["bootstrap_tasks"] if t["comparison_id"] == cid}
    official_ok = all(contrast(doc, cid)["official"]["intervals"][k]["lower_bound"] == off[k]["lower_bound"]
                      for k in off)
    return (names_ok and got == LD.task_json(ref) and pgot == LD.task_json(pref) and official_ok), \
        f"{len(tasks)} {names_ok} {official_ok}"


def c_am5():
    """The Am5Pair on the kept rows, by am5_pair's method: on a pair with three AM-5 images, every id kept
    gives ingest.am5_pair's Am5Pair exactly; dropping one excluded and one included image gives 2
    excluded, 26 included and the evaluator's sha256 of the two; restrict_pair carries it."""
    rb, rc = CTX.am5
    every = list(rb.stats.manifest_ids)
    full = DS.am5_on_kept(rb, rc, set(every))
    exc = sorted(rb.am5.excluded_ids)
    inc = next(i for i in every if i not in exc)
    some = set(every) - {exc[0], inc}
    part = DS.am5_on_kept(rb, rc, some)
    pv = dataclasses.replace(align_runs(rb, rc, policy=Policy.NONOFFICIAL_SMOKE, metric=METRIC_DISEASE_ONLY),
                             am5=am5_pair(rb, rc))
    restricted = DS.restrict_pair(pv, some, (rb, rc))
    return (len(exc) == 3 and full == am5_pair(rb, rc)
            and (part.n_excluded_am5, part.n_included) == (2, 26)
            and part.excluded_ids_sha256 == am5_excluded_ids_sha256(tuple(exc[1:]))
            and restricted.am5 == part and restricted.n == 28), f"{full} {part}"


# --------------------------------------------------------------------------------------------------
# G: the two guard cases
# --------------------------------------------------------------------------------------------------
def g_nothing_removed():
    """Guard 1: with every id kept (removed_by_reason empty), (i)'s subset values equal the full ones and
    (ii)'s run_comparison results equal the family's stored comparisons bitwise; the clean dataset-level
    differences equal the stored pooled observed values, robustness's equals report's mC contrast, and
    the NI's stage mIoUs and observed delta equal the official ones."""
    every = sorted(CTX.world.truth.ids)

    def edit(export, binding):
        export["dedup"]["removed_by_reason"] = {}
    doc = doc_of(variant("dedup_none", edit, dedup={"kept_ids": every, "removed_by_reason": {}}))
    fam = stats_artifact().family
    stored = {c["comparison_id"]: c for c in fam["a3a_family"]["comparisons"]}
    tasks = {t["task_id"]: t for t in fam["bootstrap_tasks"]}
    keys = [f"{r}_{s}{x}" for r in (GT_PRESENT, UNION_PRESENT) for s in SPACES for x, _ in FIELDS]
    i_ok = all(m["subset"][k] == m["full"][k] for m in doc["models"] for k in keys)
    ii_ok = all(json.loads(A.canonical_json(c["subset"]["comparison"])) == stored[c["comparison_id"]]
                for c in doc["contrasts"])
    robust = RP.dataset_miou_c_contrast(stats_artifact().views)["difference_e6_minus_e1"]
    diffs = all(c["subset"]["dataset_level_difference"] == (
        robust if c["comparison_id"] == D.ROBUSTNESS_ID
        else tasks[f"{c['comparison_id']}__dataset_miou_delta"]["observed"]) for c in doc["contrasts"])
    ni, off = doc["noninferiority"]["subset"], fam["noninferiority_e3_e6"]
    ni_ok = (ni["observed_delta"] == off["observed_delta"]
             and ni["baseline_dataset_miou"] == off["baseline_dataset_miou"]
             and ni["candidate_dataset_miou"] == off["candidate_dataset_miou"])
    agree = all(c["disagrees"] is False for c in doc["contrasts"])
    return i_ok and ii_ok and diffs and ni_ok and agree and doc["dedup"]["n_removed"] == 0, \
        f"(i) {i_ok} (ii) {ii_ok} diffs {diffs} NI {ni_ok} agree {agree}"


def g_official_policy():
    """Guard 2: (ii) on a subset PairedVector whose policy is Policy.OFFICIAL (dataclasses.replace of a
    fixture pair, accuracy_e1_e2 and robustness_e1_e6), beside the control -- the same pair unrestricted,
    its policy replaced the same way; (iii) on the same kept rows. The case fails if the restricted vector
    alone is refused (the frozen code's refusal of the subset; never investigated by opening
    src/stats/tests.py); a refusal of the control is the fixture's, recorded here, and the guard is then
    run under NONOFFICIAL_SMOKE."""
    from src.stats.tests import run_comparison
    st, grid = stats_artifact(), official_corruption_grid()
    mc = {s: assemble_miou_c([st.cell(s, n, v).run for n in grid.names for v in INFERENTIAL_SEVERITIES], grid,
                             policy=st.policy, stage=s, clean=clean_run(s)) for s in ("E1", "E6")}
    pairs = {"accuracy_e1_e2": ("E1", "E2", align_runs(clean_run("E1"), clean_run("E2"), policy=st.policy,
                                                       metric=METRIC_DISEASE_ONLY)),
             D.ROBUSTNESS_ID: ("E1", "E6", align_miou_c(mc["E1"], mc["E6"], policy=st.policy))}
    notes, ok = [], True
    for cid, (b, c, pv) in pairs.items():
        outcome = {}
        for label, vec in (("control", pv), ("restricted", DS.restrict_pair(pv, set(kept())))):
            for pol in (Policy.OFFICIAL, Policy.NONOFFICIAL_SMOKE):
                try:
                    run_comparison(cid, dataclasses.replace(vec, policy=pol), baseline_stage=b,
                                   candidate_stage=c)
                    outcome[(label, pol.name)] = "ran"
                except Exception as e:                                  # noqa: BLE001 -- recorded
                    outcome[(label, pol.name)] = f"{type(e).__name__}: {str(e)[:160]}"
        ctl, res = outcome[("control", "OFFICIAL")], outcome[("restricted", "OFFICIAL")]
        if ctl == "ran" and res != "ran":
            ok = False                                    # the frozen code refuses the subset alone
        if ctl != "ran" and outcome[("restricted", "NONOFFICIAL_SMOKE")] != "ran":
            ok = False
        notes.append(f"{cid}: OFFICIAL control {ctl}; OFFICIAL restricted {res}; NONOFFICIAL_SMOKE "
                     f"restricted {outcome[('restricted', 'NONOFFICIAL_SMOKE')]}")
    ni, _ = DS.subset_noninferiority({s: clean_run(s) for s in D.STAGES}, set(kept()), B=SMOKE_B)
    notes.append(f"(iii) NI on the kept rows: passed {ni.passed}, lower bound {ni.lower_bound!r}")
    CTX.notes["guard2"] = notes
    return ok, " || ".join(notes)


# --------------------------------------------------------------------------------------------------
# N: (iii)
# --------------------------------------------------------------------------------------------------
def n_noninferiority():
    """The E3-vs-E6 check on the kept rows: the one-sided bound recomputed here (dedup__noninferiority_e3_e6,
    run_job's pooled callables), build_non_inferiority on it, the official block copied."""
    ps = own_dense("noninferiority_e3_e6", kept())
    r = LD.descriptive_task("dedup__noninferiority_e3_e6", BS.DATASET_MIOU_DELTA, BS.ONE_SIDED_LOWER,
                            ps.n_images, SMOKE_B, ps.observed(), ps.replicate, ps.jackknife)
    b, c = ps.stage_mious()
    ni = build_non_inferiority(b, c, r.bca.bounds[0])
    n = the_doc()["noninferiority"]
    off = stats_artifact().family["noninferiority_e3_e6"]
    return (n["subset"]["passed"] is ni.passed and n["subset"]["lower_bound"] == ni.lower_bound
            and n["subset"]["observed_delta"] == ni.observed_delta
            and n["subset"]["interval"] == LD.task_json(r)
            and n["official"]["passed"] is off["passed"]
            and n["official"]["lower_bound"] == off["lower_bound"]
            and n["disagrees"] is (ni.passed is not off["passed"])
            and n["subset"]["interval"]["interval_type"] == "one_sided_lower"), str(n["subset"])[:200]


# --------------------------------------------------------------------------------------------------
# T: (iv) B5-D3
# --------------------------------------------------------------------------------------------------
def t_unadjusted():
    """p < 0.05, strictly; None is not evaluable (never False)."""
    got = [DS.unadjusted(p) for p in (0.05, math.nextafter(0.05, 0.0), 0.2, 0.0, None)]
    return got == [False, True, False, True, None], str(got)


def t_labels():
    """report.contrast_label on Python bool and float; a pending caveat named as family_rows names it;
    caveat True gives 'inconclusive'; not evaluable for a None decision."""
    got = [DS.label_of(True, np.float64(0.02), "pending"), DS.label_of(True, 0.005, False),
           DS.label_of(False, -0.01, "pending"), DS.label_of(False, 0.0, True), DS.label_of(None, 0.1, True)]
    want = [RP.LABEL_SIZABLE, RP.LABEL_BELOW_SESOI, f"{RP.LABEL_NOT_REJECTED}; {RP.CAVEAT_PENDING}",
            RP.LABEL_INCONCLUSIVE, "not evaluable"]
    return got == want, str(got)


def t_compare():
    """Hand rows: a different decision, a different label, opposite signs (product -1), a zero difference
    (no sign disagreement), agreement, and a subset that is not evaluable: with the same sign (no
    verdict) and with the opposite sign (it disagrees on the sign; DL-24 B5R3-01)."""
    o = {"unadjusted_reject": True, "label": "sizable", "dataset_level_difference": 0.02}
    cases = [({**o, "unadjusted_reject": False, "label": "not rejected"}, (True, ["decision", "label"])),
             ({**o, "label": "detected, below"}, (True, ["label"])),
             ({**o, "dataset_level_difference": -0.001}, (True, ["sign of the dataset-level difference"])),
             ({**o, "dataset_level_difference": 0.0}, (False, [])),
             (dict(o), (False, [])),
             ({**o, "unadjusted_reject": None, "label": "not evaluable"},
              (None, ["not evaluable on the subset"])),
             ({**o, "unadjusted_reject": None, "label": "not evaluable", "dataset_level_difference": -0.001},
              (True, ["not evaluable on the subset", "sign of the dataset-level difference"]))]
    got = [DS.compare(o, s) for s, _ in cases]
    return got == [w for _, w in cases], str(got)


def t_table():
    """In the part: each contrast's official decision is p_raw < 0.05 (the family's holm members), its
    labels and disagreement recomputed here, and the official Holm decision and label are
    report.family_rows' unchanged."""
    doc = the_doc()
    fam = stats_artifact().family
    holm = {m["comparison_id"]: m for m in fam["a3a_family"]["holm"]["members"]}
    robust = RP.dataset_miou_c_contrast(stats_artifact().views)["difference_e6_minus_e1"]
    rows = {r["comparison_id"]: r for r in RP.family_rows(fam, robust, power_caveat="pending")}
    bad = []
    for c in doc["contrasts"]:
        o, s, cid = c["official"], c["subset"], c["comparison_id"]
        off_reject = holm[cid]["p_raw"] < 0.05
        sub_reject = None if s["p"] is None else s["p"] < 0.05
        ok = (o["p_raw"] == holm[cid]["p_raw"] and o["unadjusted_reject"] is off_reject
              and s["unadjusted_reject"] is sub_reject
              and o["label"] == DS.label_of(off_reject, o["dataset_level_difference"], "pending")
              and s["label"] == DS.label_of(sub_reject, s["dataset_level_difference"], "pending")
              and (c["disagrees"], c["why"]) == DS.compare(o, s)
              and o["holm_reject"] is rows[cid]["holm_reject"] and o["holm_label"] == rows[cid]["label"]
              and o["dataset_level_difference"] == rows[cid]["dataset_level_difference"])
        if not ok:
            bad.append(cid)
    return not bad, str(bad)


# --------------------------------------------------------------------------------------------------
# R: (v)
# --------------------------------------------------------------------------------------------------
def r_issue():
    """removed_by_reason beside the issue's rows and totals (PA :1196-1204), typed out here; the issue's
    labels are values only: no key of the part is, or holds, one of them (the frozen A3a serializer's own
    keys, such as t_test, are family.json's)."""
    doc = the_doc()
    keys = set()

    def walk(v):
        if isinstance(v, dict):
            keys.update(v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(doc)
    md = "\n".join(DS.render_markdown(doc, ""))
    return (doc["issue"]["rows"] == [{"row": r, "count": n} for r, n in ISSUE]
            and (doc["issue"]["duplicate_groups"], doc["issue"]["redundant_files"], doc["issue"]["files"])
            == (265, 277, 7774)
            and doc["removed_by_reason"] == [{"reason": "equals_train_image", "count": 2},
                                             {"reason": "equals_val_image", "count": 2},
                                             {"reason": "repeats_an_earlier_image", "count": 1}]
            and not any("↔" in str(k) or str(k) in dict(ISSUE) for k in keys)
            and all(f"| {r} | {n} |" in md for r, n in ISSUE)), \
        str(sorted(k for k in keys if "↔" in str(k)))


# --------------------------------------------------------------------------------------------------
# D
# --------------------------------------------------------------------------------------------------
def d_deterministic():
    return A.canonical_json(doc_of()) == A.canonical_json(the_doc()), ""


def d_no_holm():
    """No Holm object on the subset side: no subset key names holm, and the module imports no
    holm_family; the official Holm fields are copied, never recomputed."""
    doc = the_doc()
    sub_keys = {k for c in doc["contrasts"] for k in c["subset"]}
    tree = ast.parse((REPO / "src" / "stats" / "dedup_subset.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    return not any("holm" in k for k in sub_keys) and "holm_family" not in names, str(sorted(names))


def d_banner():
    doc = the_doc()
    md = "\n".join(DS.render_markdown(doc, "BANNER"))
    return (doc["banner"] == "descriptive; outside protocol 1.0.0" and doc["banner"] in md
            and doc["analysis_id"] == "lrep_descriptive_v1" and md.count("BANNER") >= 5
            and doc["footnote"] in md), ""


def d_no_test_path():
    """LRB20, with ':' (R4-6): no created path contains 'test' or ':'."""
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower() or ":" in n]
    return not bad, str(bad[:3])


def d_production_time():
    """The part at B = bootstrap.PRODUCTION_B on the fixture (26 kept images): every interval carries
    B = 10,000; the measured time is printed (Part 2 states it; TEST's own cost is n-dependent)."""
    t0 = time.time()
    doc = doc_of(B=BS.PRODUCTION_B)
    secs = time.time() - t0
    n = sum(len(c["subset"]["intervals"]) for c in doc["contrasts"]) + 1
    ok = all(i["B"] == 10_000 for c in doc["contrasts"] for i in c["subset"]["intervals"]) and \
        doc["noninferiority"]["subset"]["interval"]["B"] == 10_000 and n == 32
    CTX.notes["production_seconds"] = secs
    return ok, f"{n} tasks at B = {BS.PRODUCTION_B}: {secs:.1f} s on the fixture"


class _Injected(H.StatsArtifact):
    """The verified statistics artifact (its family, views and the 37 unchanged), except that its clean
    input of one stage is another run: the injection R4-1 item 5 guards against."""
    def clean(self, stage):
        return CTX.notes["injected"].get(stage) or H.StatsArtifact.clean(self, stage)


def o_dt_in_official():
    """R4-1 item 5: a DT run -- E7-R1's (role arm at int8_ptq), read by load_slot -- given to (ii) and (iii)
    as E7's clean input is refused where the eight contrasts and NI take their inputs
    (descriptive_input_in_official), naming it; the 37's own inputs pass the guard (the part is built)."""
    st, inv = stats_artifact(), inventory()
    ba = H.load_slot(st, inv, slot="E7-R1:s42:list42", stage="E7-R1", precision="int8_ptq",
                     input_root=CTX.world.input_root)
    CTX.notes["injected"] = {"E7": ba.loaded}
    fake = _Injected(**{f.name: getattr(st, f.name) for f in dataclasses.fields(st)})
    ok, detail = refused("descriptive_input_in_official", lambda: DS.build_dedup(
        fake, inv, H.planned_cells(inv), input_root=CTX.world.input_root, B=SMOKE_B))
    return (ok and "['E7 clean']" in detail and ba.read == "descriptive_test"
            and fake.clean("E1") is st.clean("E1") and the_doc()["part"] == "dedup"), detail


def o_read():
    """R4-1 items 2 and 6: (i)'s rows of the arms' E7-R1, E7-R2 and E7-flip read descriptive_test and print
    DT_LABEL in their stage cell; every other row reads official and prints none."""
    doc = the_doc()
    md = "\n".join(DS.render_markdown(doc, ""))
    reads = {m["slot"]: (m["read"], m["read_label"]) for m in doc["models"]}
    dt = sorted(s for s, v in reads.items() if v != ("official", None))
    return (dt == ["E7-R1:s42:list42", "E7-R2:s42:list42", "E7-flip:s42:list42"]
            and all(reads[s] == ("descriptive_test", DT_LABEL) for s in dt)
            and all(f"| {st} ({DT_LABEL}) | 42 | arm |" in md for st in ("E7-R1", "E7-R2", "E7-flip"))
            and md.count(DT_LABEL) == 3), str(dt)


def write_am5_pair(w) -> tuple:
    """Two clean artifacts (E1, E2) on a truth with three AM-5 images, loaded here."""
    truth = F.make_truth(w.n, 3, F.int_seed(LF.WORLD_SEED, "am5 truth"))
    runs = []
    for stage, q in (("E1", 0.70), ("E2", 0.72)):
        rel = f"runs/heldout/am5pair_{stage.lower()}_clean"
        scored = F.score(truth, F.predict(truth, quality=q, seed=F.int_seed(LF.WORLD_SEED, rel)),
                         Condition(*D.CLEAN))
        F.write_fixture(w.input_root / rel, scored, w.n, stage=stage, condition=Condition(*D.CLEAN),
                        split="val", status="provisional", repo_root=w.input_root,
                        run_id=f"lrep-am5pair-{stage.lower()}", precision="fp32",
                        checkpoint_sha256=LF.syn_sha("am5 pair", stage))
        runs.append(D.load_input(stage, D.CLEAN, rel, w.input_root, Policy.NONOFFICIAL_SMOKE).run)
    return tuple(runs)


def setup(root: Path, world=None) -> None:
    CTX.world = world or LF.build_world(root, strata=False)
    CTX.stats, CTX.paths, CTX.doc, CTX.notes = None, {}, None, {"injected": {}}
    CTX.am5 = write_am5_pair(CTX.world)


def cases():
    return [("I1", "B5-S2: the record read (sha256 first, T4's fields)", i_read),
            ("I2", "dedup_invalid: dedup null", i_null),
            ("I3", "dedup_invalid: dedup without removed_by_reason", i_fields),
            ("I4", "dedup_invalid: a path that is not plain relative POSIX", i_path),
            ("I5", "dedup_invalid: the file is missing", i_missing),
            ("I6", "dedup_invalid: the file is not at the export's sha256", i_sha),
            ("I7", "dedup_invalid: the file is not UTF-8", i_utf8),
            ("I8", "dedup_invalid: a repeated key or NaN (strict JSON)", i_strict),
            ("I9", "dedup_invalid: kept_ids unsorted, repeated or empty", i_ids),
            ("I10", "dedup_invalid: the file's removed_by_reason is not the export's", i_reasons),
            ("I11", "a field LREP does not read is reported, not refused", i_extra),
            ("I12", "the invariants: one id set, kept inside it, |all| - |kept| <= the sum", i_invariants),
            ("I13", "dedup_invalid through the part: fewer counted than removed", i_short_count),
            ("V1", "(i): the 38 rows read (models of record, arms with A, F, G, quant_descriptive, E6-KD)",
             v_rows),
            ("V2", "(i): twelve numbers on the subset recomputed here; full values the artifact's", v_values),
            ("V3", "(i): the eligibility changes (:159's form); 26 kept, 4 removed", v_changes),
            ("V4", "(i): the counts, the overlap and the fingerprints", v_counts),
            ("V5", "(i): a defect in a slot (b) does not report leaves the part unrefused (LRB4)",
             v_unreported_slot),
            ("C1", "(ii): the eight contrasts in order", c_order),
            ("C2", "(ii): the clean vectors restricted by id, run_comparison on them", c_vectors),
            ("C3", "(ii): the pooled dataset-level differences on the kept rows", c_pooled),
            ("C4", "(ii): robustness through assemble_miou_c/align_miou_c; mC on the kept rows",
             c_robustness),
            ("C5", "(ii): the 31 intervals, ids and types; recomputed here; the official beside",
             c_intervals),
            ("C6", "(ii): the Am5Pair on the kept rows by am5_pair's method", c_am5),
            ("G1", "guard 1: nothing removed gives (i) and (ii) bitwise equal to the full results",
             g_nothing_removed),
            ("G2", "guard 2: Policy.OFFICIAL on the restricted vector beside its control; (iii) on its rows",
             g_official_policy),
            ("N1", "(iii): the NI check on the kept rows, recomputed here", n_noninferiority),
            ("T1", "B5-D3: p < 0.05 strictly; None is not evaluable", t_unadjusted),
            ("T2", "B5-D3: contrast_label on Python types; pending and inconclusive", t_labels),
            ("T3", "B5-D3: the disagreements of hand rows", t_compare),
            ("T4", "B5-D3: the part's table recomputed here; Holm decision and label unchanged", t_table),
            ("R1", "(v): removed_by_reason beside the issue's counts; labels values only", r_issue),
            ("D1", "byte-identical JSON on repeated runs", d_deterministic),
            ("D2", "no Holm object on the subset side", d_no_holm),
            ("D3", "the banner, the analysis id and the footnote", d_banner),
            ("D4", "no created path contains 'test' or ':' (LRB20; R4-6)", d_no_test_path),
            ("D5", "the part at B = PRODUCTION_B, timed", d_production_time),
            ("O1", "R4-1 item 5: descriptive_input_in_official: a DT run given to (ii)-(iii) as E7's input",
             o_dt_in_official),
            ("O2", "R4-1 items 2, 6: the E7-X rows read descriptive_test and print the DT label", o_read)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_dedup_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_dedup_subset: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok or cid in NOTE_CASES:
            print(f"         {detail[:900]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'DEDUP SUBSET OK' if good == len(results) else 'DEDUP SUBSET FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; descriptive; outside protocol 1.0.0.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
