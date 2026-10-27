"""Duplicate-free TEST (lane LREP, B-5 (b): AM-18 item 8(c); TL-13; LRB17): the subset's dataset-level
values, and the eight family contrasts and the E3-vs-E6 check recomputed on it, unadjusted, beside the
official results.

PA :1221-1231: "Duplicate-free TEST = TEST without every image whose SHA-256 equals a TRAIN or VAL
image's, keeping only the first image (by sorted id) of any group of identical TEST images. ... Reported
next to the official results, from the per-image outputs of the single TEST evaluation (no model is run
on TEST twice), descriptively: the dataset-level mIoU (all-class, GT-present, disease-only; class
eligibility recomputed on the subset) of every model of record and every arm; and, as a sensitivity
analysis, the eight family contrasts and the E3-vs-E6 check recomputed on the subset (same procedures,
unadjusted). They replace no official result and decide nothing: if one disagrees with an official
result, the official result stands and Chapter 4 reports both. ... The numbers of images removed for
each reason are reported next to the issue's counts."

  * The subset is B5-S2's kept ids (heldout_inputs.read_dedup), checked against every scored artifact
    read here (the 37 with their cells and the rows (i) reports; no other slot is read, LRB4): one image
    id set (am6_tables.ids_sha256 of each artifact's manifest_ids; a corrupted cell whose ids were
    suffixed, src/eval/adapters.py:13-14, is refused here, never mapped), the kept ids inside it, and
    |all| - |kept| <= the sum of removed_by_reason (TMAN's committed schema states no partition of the
    reasons, so the overlap is reported); dedup_invalid otherwise.
  * (i) Every row of the GT-present table (am6_tables with coverage, B-5 (d)) whose roles include
    family, per_seed, arm, arm_teacher, quant_descriptive or teacher_reference, and E6-KD's: its four
    numbers on the subset, eligibility recomputed by eligibility.class_totals on the kept rows, and the
    eligibility changes against full TEST (am6_tables.subset_values; scripts/dedup_val_scores.py's
    method).
  * (ii) The eight family contrasts from the 37 inputs with the frozen procedures, unadjusted (no Holm):
    the seven clean ones by align_runs then run_comparison, robustness_e1_e6 by assemble_miou_c and
    align_miou_c over the 15 inferential cells then run_comparison, as driver.assemble_observed does
    (driver.py:362-380). Each PairedVector comes from the full inputs and is then restricted to the kept
    ids: every per-image field together, by dataclasses.replace, with the Am5Pair fields
    (ingest.py:140-144) recomputed on the kept rows from the runs' excluded ids by am5_pair's
    method (ingest.py:439-458). The dataset-level difference: the pooled delta of PooledStages restricted
    by row (the kept ids map to manifest_ids positions, dedup_val_scores.py:109-116) for the clean ones;
    Part 1's mC (seed_stability.miou_c, equal to report.dataset_miou_c bitwise) on the kept rows' cell
    totals for robustness_e1_e6. Every bootstrap task the official procedure runs for these contrasts
    (driver.task_jobs' tasks for them, driver.py:425-449) is recomputed on the subset in
    lrep_descriptive's namespace (ids dedup__<comparison_id>, the official statistic names, B =
    PRODUCTION_B) and printed beside the official interval; no decision or label reads them.
  * (iii) The E3-vs-E6 check on the subset: PooledStages.from_dense on the kept rows, the one-sided BCa
    lower bound in lrep_descriptive's namespace with the callables run_job uses (driver.py:441-449), then
    noninferiority.build_non_inferiority (noninferiority.py:183-203).
  * (iv) B5-D3, like with like. Each contrast's unadjusted decision on the subset (p < 0.05, strict,
    STATS §6 :289-293) beside the official artifact's unadjusted decision (family.json's holm members'
    p_raw < 0.05; artifact.py:1043-1045, :1489); the AM-17 item 2 label from report.contrast_label on
    both sides with the same inputs (that decision, the contrast's dataset-level difference as
    report.py:296-298 chooses it, the committed MDE entry's caveat through report.committed_caveat);
    NI's `passed` beside the official one; the official Holm decision and label (report.family_rows)
    printed unchanged. "Disagrees" means a different decision, a different label, opposite signs of the
    dataset-level difference (sign product -1), or a different NI `passed`. A subset p-value of None
    reads "not evaluable", never a False decision; the contrast then disagrees on the sign alone, and is
    otherwise neither agreeing nor disagreeing (disagrees null). The official result stands (PA
    :1228-1230).
  * (v) removed_by_reason beside the issue's counts, AM-18 item 8(a)'s rows (PA :1196-1204), pinned
    here; the issue #11 CSV is never read. The issue's row labels are string values only, never keys.
  * B-5b (ruling LREP-B5-RULE-4 R4-1): (i)'s rows come from am6's coverage, so a descriptive_test slot (the
    arms' E7-X) is read there by the DT read under Policy.OFFICIAL, and its row carries `read` and prints
    DT_LABEL in its stage cell. (ii) and (iii) are official computations: before they take their inputs,
    every run is checked to be one of the 37 inputs of the verified statistics artifact
    (heldout_inputs.require_official_inputs; descriptive_input_in_official), so a DT run never reaches the
    eight contrasts or NI (item 5).

Descriptive; outside protocol 1.0.0. Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import numpy as np

from . import am6_tables as AM6
from . import bootstrap as BS
from . import driver as D
from . import heldout_inputs as H
from . import lrep_descriptive as LD

#: AM-18 item 8(a) (PA :1196-1204): issue tqwei05/PlantSeg #11's rows, in its order; the labels are the
#: issue's and are carried as string values only.
ISSUE_ROWS = (("train ↔ train", 124), ("test ↔ train", 70), ("train ↔ val", 32), ("test ↔ test", 13),
              ("test ↔ val", 11), ("3-way and more", 15))
ISSUE_TOTALS = {"duplicate_groups": 265, "redundant_files": 277, "files": 7774,
                "hash": "SHA-1 over the image files of the Zenodo 10.5281/zenodo.17719108 release",
                "source": "AM-18 item 8(a), PA :1196-1204"}
#: (i): the roles of the rows reported on the subset (every model of record and every arm, PA
#: :1226-1227; AM-23 item 5 for quant_descriptive), and E6-KD's row whatever its role.
SUBSET_ROLES = ("family", "per_seed", "arm", "arm_teacher", "quant_descriptive", "teacher_reference")
EIGHT = (*D.CLEAN_COMPARISON_IDS, D.ROBUSTNESS_ID)
NOT_EVALUABLE = "not evaluable"
TITLE = "Duplicate-free TEST (AM-18 item 8(c); descriptive; outside protocol 1.0.0)"
FOOTNOTE = ("Descriptive sensitivity analysis (AM-18 item 8(c)): unadjusted, no Holm; it replaces no "
            "official result and decides nothing. Where it disagrees, the official result stands and "
            "Chapter 4 reports both (PA :1228-1230). Intervals: lrep_descriptive_v1, outside protocol "
            "1.0.0.")


def check_ids(rec: H.DedupRecord, fingerprints: dict, all_ids) -> dict:
    """B5-S2's invariants: every scored artifact read has one image id set (its fingerprint), the kept
    ids are inside it, and |all| - |kept| <= the sum of removed_by_reason (the overlap is reported)."""
    ref = AM6.ids_sha256(all_ids)
    differ = sorted(label for label, fp in fingerprints.items() if fp != ref)
    if differ:
        raise H.refusal("dedup_invalid", f"the scored artifacts do not share one image id set: "
                                         f"{differ[:3]} differ from the 37's (B5-S2)")
    kept, every = set(rec.kept_ids), set(all_ids)
    outside = sorted(kept - every)
    if outside:
        raise H.refusal("dedup_invalid", f"{len(outside)} kept id(s) are not image ids of the scored "
                                         f"artifacts, e.g. {outside[:3]} (B5-S2)")
    removed, total = len(every) - len(kept), rec.removed_total
    if removed > total:
        raise H.refusal("dedup_invalid", f"{removed} images are not kept but removed_by_reason counts "
                                         f"{total} (B5-S2: |all| - |kept| <= the sum)")
    return {"n_all": len(every), "n_kept": len(kept), "n_removed": removed, "sum_removed_by_reason": total,
            "overlap": total - removed, "partition_stated": False, "image_ids_sha256": ref,
            "kept_ids_sha256": AM6.ids_sha256(kept)}


def am5_on_kept(baseline_run, candidate_run, kept):
    """ingest.am5_pair's method (ingest.py:439-458) on the kept rows: each run's excluded ids
    (Am5Exclusion.excluded_ids, sorted image ids) restricted to the kept ids, which must agree;
    n_included = the run's kept rows (every row, ingest.py:428) - the excluded ones; the sha256 of the
    excluded ids by the evaluator's am5_excluded_ids_sha256 (ingest.py:429)."""
    from ..eval.artifacts import am5_excluded_ids_sha256
    from .ingest import Am5Pair
    sides = []
    for run in (baseline_run, candidate_run):
        excluded = tuple(i for i in run.am5.excluded_ids if i in kept)
        rows = sum(1 for i in run.stats.manifest_ids if i in kept)
        sides.append((excluded, rows - len(excluded)))
    if sides[0] != sides[1]:
        raise H.refusal("cross_artifact_mismatch", "AM-5: the pair excludes different kept images (R-10)")
    excluded, included = sides[0]
    return Am5Pair(n_excluded_am5=len(excluded), n_included=included,
                   excluded_ids_sha256=am5_excluded_ids_sha256(excluded))


def restrict_pair(pv, kept, runs=None):
    """The PairedVector on the kept ids: every per-image field together (dataclasses.replace), the
    Am5Pair recomputed on the kept rows when the pair carries one (`runs`: its baseline and candidate)."""
    mask = np.fromiter((i in kept for i in pv.image_ids), dtype=bool, count=len(pv.image_ids))
    am5 = pv.am5 if pv.am5 is None else am5_on_kept(*runs, kept)
    return dataclasses.replace(pv, image_ids=tuple(i for i, k in zip(pv.image_ids, mask) if k),
                               baseline=pv.baseline[mask], candidate=pv.candidate[mask],
                               delta=pv.delta[mask], am5=am5)


def restrict_pooled(ps, manifest_ids, kept):
    """PooledStages restricted by row: the kept ids map to manifest_ids positions
    (dedup_val_scores.py:109-116), in manifest order; then PooledStages.from_dense."""
    from .noninferiority import PooledStages
    pos = {m: i for i, m in enumerate(manifest_ids)}
    rows = np.asarray(sorted(pos[i] for i in kept), dtype=np.int64)
    return PooledStages.from_dense((ps.base_tp[rows], ps.base_gt[rows], ps.base_pred[rows]),
                                   (ps.cand_tp[rows], ps.cand_gt[rows], ps.cand_pred[rows]))


def miou_c_on_kept(cell_runs, kept, names) -> float:
    """Part 1's mC (seed_stability.miou_c) on the kept rows' cell totals, each rebuilt by
    eligibility.class_totals on the kept triplets (dedup_val_scores.py:121-131's method)."""
    from . import seed_stability as SS
    from .eligibility import class_totals
    totals = {}
    for key, run in cell_runs.items():
        st = run.stats
        pos = {m: i for i, m in enumerate(st.manifest_ids)}
        index = np.asarray(st.image_index)
        keep = np.isin(index, np.asarray([pos[i] for i in sorted(kept)], dtype=index.dtype))
        t = class_totals({"class_id": np.asarray(st.class_id)[keep], "tp": np.asarray(st.tp)[keep],
                          "gt": np.asarray(st.gt)[keep], "pred": np.asarray(st.pred)[keep]},
                         num_classes=run.identity.num_classes)
        totals[key] = types.SimpleNamespace(dataset_tp=t.tp, dataset_gt=t.gt, dataset_pred=t.pred)
    return SS.miou_c(totals, names)


def unadjusted(p):
    """STATS §6 (:289-293): reject iff p < 0.05, strictly; None (not evaluable) when p is None."""
    from .artifact import ALPHA
    return None if p is None else bool(p < ALPHA)


def label_of(reject, difference, caveat) -> str:
    """AM-17 item 2 through report.contrast_label, with Python bool and float (report.py:91-94); a
    pending caveat is named as report.family_rows names it (report.py:298-300)."""
    from . import report as RP
    if reject is None:
        return NOT_EVALUABLE
    label = RP.contrast_label(bool(reject), float(difference), power_caveat=caveat is True)
    if label == RP.LABEL_NOT_REJECTED and caveat == "pending":
        label = f"{label}; {RP.CAVEAT_PENDING}"
    return label


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def compare(official: dict, subset: dict) -> tuple:
    """B5-D3: (disagrees, why) of one contrast -- a different decision, a different label, or opposite
    signs of the dataset-level difference (sign product -1). The sign test needs no p-value: a subset that
    is not evaluable disagrees on the sign alone, and otherwise neither agrees nor disagrees (None)."""
    opposite = _sign(official["dataset_level_difference"]) * _sign(subset["dataset_level_difference"]) == -1
    if subset["unadjusted_reject"] is None:
        why = ["not evaluable on the subset"]
        if opposite:
            why.append("sign of the dataset-level difference")
        return (True if opposite else None), why
    why = []
    if official["unadjusted_reject"] != subset["unadjusted_reject"]:
        why.append("decision")
    if official["label"] != subset["label"]:
        why.append("label")
    if opposite:
        why.append("sign of the dataset-level difference")
    return bool(why), why


def _official_intervals(fam: dict, cid: str) -> dict:
    return {t["statistic_name"]: {"lower_bound": t["lower_bound"], "upper_bound": t.get("upper_bound"),
                                  "observed": t["observed"], "interval_type": t["interval_type"]}
            for t in fam["bootstrap_tasks"] if t["comparison_id"] == cid}


def subset_interval(contrast_id, spec, *, pv=None, result=None, ps=None, B: int) -> dict:
    """One official task of a contrast (a FROZEN_TASK_MATRIX spec), on the subset, in lrep_descriptive's
    namespace, with run_job's callables (driver.py:441-449) and observed_value's sources (:399-405): the
    restricted PooledStages for dataset_miou_delta, else scalar_callables on the restricted delta with
    the subset comparison's typed observed value."""
    if spec.statistic_name == BS.DATASET_MIOU_DELTA:
        r = LD.descriptive_task(contrast_id, spec.statistic_name, spec.interval_type, ps.n_images, B,
                                ps.observed(), ps.replicate, ps.jackknife)
    else:
        delta = np.asarray(pv.delta, dtype=np.float64)
        rf, jf = BS.scalar_callables(spec.statistic_name, delta)
        r = LD.descriptive_task(contrast_id, spec.statistic_name, spec.interval_type, int(delta.shape[0]),
                                B, BS.comparison_observed(result, spec.statistic_name), rf, jf)
    return LD.task_json(r)


def model_rows(stats, inv, cells, kept, *, input_root: Path) -> list[dict]:
    """(i): the rows of the GT-present table (with coverage) that item 8(c) reports, with their full and
    subset values; each row's image-id fingerprint is checked by the caller (B5-S2). Only those rows are
    read: a coverage slot of another role is ignored here and never refused (LRB4)."""
    doc = AM6.build_am6_tables(stats, inv, cells, input_root=input_root, coverage=True, subset=kept,
                               roles=SUBSET_ROLES, stages=(H.E6KD_STAGE,))
    out = []
    for r in doc["rows"]:
        roles = [r["role"]] if AM6.is_planned(r) else list(r["roles"])
        if not (set(roles) & set(SUBSET_ROLES) or r["stage"] == H.E6KD_STAGE):
            continue
        full = {k: r[k] for k in r["subset"] if k not in ("eligibility_changes", "n_images_kept",
                                                         "n_images_removed", "image_ids_sha256")}
        out.append({"slot": r["slot"], "stage": r["stage"], "seed": r["seed"], "roles": roles,
                    "read": r.get("read", H.READ_OFFICIAL), "read_label": r.get("read_label"),
                    "run_id": r["run_id"], "full": full, "subset": r["subset"]})
    return out


def subset_contrast(cid: str, clean: dict, cell_runs: dict, kept, *, policy, grid):
    """(ii) for one of the eight: (restricted PairedVector, restricted PooledStages or None, dataset-level
    difference on the subset, run_comparison's result) by the frozen procedures (driver.py:362-380)."""
    from .align import AlignmentError, align_runs
    from .noninferiority import PooledStages
    from .robustness import align_miou_c, assemble_miou_c
    from .tests import run_comparison
    b, c, metric = D.COMPARISON_TABLE[cid]
    if cid == D.ROBUSTNESS_ID:
        mc = {st: assemble_miou_c(list(cell_runs[st].values()), grid, policy=policy, stage=st,
                                  clean=clean[st]) for st in (b, c)}
        pv = restrict_pair(align_miou_c(mc[b], mc[c], policy=policy), kept)
        ps = None
        diff = miou_c_on_kept(cell_runs[c], kept, grid.names) - miou_c_on_kept(cell_runs[b], kept, grid.names)
    else:
        try:
            full = align_runs(clean[b], clean[c], policy=policy, metric=metric)
        except AlignmentError as e:
            raise H.refusal("cross_artifact_mismatch", f"{cid}: the pair cannot be aligned: {e}") from e
        pv = restrict_pair(full, kept, (clean[b], clean[c]))
        ps = restrict_pooled(PooledStages.from_runs(clean[b], clean[c]), clean[b].stats.manifest_ids, kept)
        diff = ps.observed()
    return pv, ps, float(diff), run_comparison(cid, pv, baseline_stage=b, candidate_stage=c)


def subset_noninferiority(clean: dict, kept, *, B: int):
    """(iii): the E3-vs-E6 check on the kept rows -- PooledStages restricted by row, the one-sided BCa
    lower bound in lrep_descriptive's namespace with run_job's pooled callables, build_non_inferiority
    on that one bound (driver.artifact_inputs' recipe, driver.py:490-498)."""
    from .noninferiority import PooledStages, build_non_inferiority
    b, c, _ = D.COMPARISON_TABLE[BS.NONINFERIORITY_E3_E6]
    ps = restrict_pooled(PooledStages.from_runs(clean[b], clean[c]), clean[b].stats.manifest_ids, kept)
    r = LD.descriptive_task(LD.dedup_contrast_id(BS.NONINFERIORITY_E3_E6), BS.DATASET_MIOU_DELTA,
                            BS.ONE_SIDED_LOWER, ps.n_images, B, ps.observed(), ps.replicate, ps.jackknife)
    base_m, cand_m = ps.stage_mious()
    return build_non_inferiority(base_m, cand_m, r.bca.bounds[0]), r


def build_dedup(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *, input_root: Path,
                B: int = BS.PRODUCTION_B, power_caveat="pending", mde_w=None) -> dict:
    """The "Duplicate-free TEST" part. `power_caveat` and `mde_w` are report.committed_caveat's."""
    from . import report as RP
    from .artifact import serialize_a3a
    from .corruption_protocol import official_corruption_grid
    from .robustness import INFERENTIAL_SEVERITIES
    rec = H.read_dedup(inv)
    kept = frozenset(rec.kept_ids)
    policy, fam = stats.policy, stats.family
    clean = {stage: stats.clean(stage).run for stage in D.STAGES}
    grid = official_corruption_grid()
    cell_runs = {stage: {(n, s): stats.cell(stage, n, s).run for n in grid.names
                         for s in INFERENTIAL_SEVERITIES} for stage in D.CORRUPTION_STAGES}
    H.require_official_inputs(stats, {**{f"{st} clean": run for st, run in clean.items()},
                                      **{f"{st} {H.cell_id(n, s)}": run for st, by_cell in cell_runs.items()
                                         for (n, s), run in by_cell.items()}},
                              "dedup (ii)-(iii), the eight contrasts and NI recomputed on the subset")
    fingerprints = {f"{li.stage} {D.condition_tuple_label(li.condition)}":
                    AM6.ids_sha256(li.run.stats.manifest_ids) for li in stats.views.loaded.values()}
    models = model_rows(stats, inv, cells, kept, input_root=input_root)
    fingerprints.update({f"{m['slot']} ({m['stage']})": m["subset"]["image_ids_sha256"] for m in models})
    counts = check_ids(rec, fingerprints, clean[D.STAGES[0]].stats.manifest_ids)

    holm = {m["comparison_id"]: m for m in fam["a3a_family"]["holm"]["members"]}
    tasks = {t["task_id"]: t for t in fam["bootstrap_tasks"]}
    robust = RP.dataset_miou_c_contrast(stats.views)["difference_e6_minus_e1"]
    official_rows = {r["comparison_id"]: r
                     for r in RP.family_rows(fam, robust, power_caveat=power_caveat, mde_w=mde_w)}
    contrasts = []
    for cid in EIGHT:
        b, c, metric = D.COMPARISON_TABLE[cid]
        pv, ps, diff, result = subset_contrast(cid, clean, cell_runs, kept, policy=policy, grid=grid)
        off_diff = robust if cid == D.ROBUSTNESS_ID else tasks[f"{cid}__{BS.DATASET_MIOU_DELTA}"]["observed"]
        p, p_raw = result.wilcoxon.p_value, holm[cid]["p_raw"]
        off_reject, sub_reject = unadjusted(p_raw), unadjusted(p)
        official = {"p_raw": p_raw, "unadjusted_reject": off_reject, "dataset_level_difference": off_diff,
                    "label": label_of(off_reject, off_diff, power_caveat),
                    "holm_p_adjusted": holm[cid]["p_adjusted"],
                    "holm_reject": official_rows[cid]["holm_reject"],
                    "holm_label": official_rows[cid]["label"], "intervals": _official_intervals(fam, cid)}
        subset = {"p": p, "unadjusted_reject": sub_reject, "dataset_level_difference": diff,
                  "label": label_of(sub_reject, diff, power_caveat), "n_paired": pv.n,
                  "am5": None if pv.am5 is None else dataclasses.asdict(pv.am5),
                  "comparison": serialize_a3a(result),
                  "intervals": [subset_interval(LD.dedup_contrast_id(cid), spec, pv=pv, result=result, ps=ps,
                                                B=B)
                                for spec in BS.FROZEN_TASK_MATRIX if spec.comparison_id == cid]}
        disagrees, why = compare(official, subset)
        contrasts.append({"comparison_id": cid, "baseline_stage": b, "candidate_stage": c, "metric": metric,
                          "official": official, "subset": subset, "disagrees": disagrees, "why": why})

    ni, r = subset_noninferiority(clean, kept, B=B)
    off_ni = fam["noninferiority_e3_e6"]
    noninferiority = {"official": {k: off_ni[k] for k in ("passed", "lower_bound", "observed_delta",
                                                           "primary_margin")},
                      "subset": {"passed": ni.passed, "lower_bound": ni.lower_bound,
                                 "observed_delta": ni.observed_delta, "observed_drop": ni.observed_drop,
                                 "baseline_dataset_miou": ni.baseline_dataset_miou,
                                 "candidate_dataset_miou": ni.candidate_dataset_miou,
                                 "relative_retention": ni.relative_retention,
                                 "primary_margin": ni.primary_margin,
                                 "sensitivity": [{"margin": s.margin, "passed": s.passed}
                                                 for s in ni.sensitivity],
                                 "interval": LD.task_json(r)},
                      "disagrees": ni.passed is not off_ni["passed"]}
    return {"part": "dedup", "title": TITLE, "banner": LD.BANNER, "analysis_id": LD.ANALYSIS_ID, "B": B,
            "dedup": {"path": rec.path, "sha256": rec.sha256, "other_fields": list(rec.other_fields),
                      **counts},
            "removed_by_reason": [{"reason": k, "count": n} for k, n in rec.removed_by_reason],
            "issue": {"rows": [{"row": label, "count": n} for label, n in ISSUE_ROWS], **ISSUE_TOTALS},
            "power_caveat": power_caveat, "models": models, "contrasts": contrasts,
            "noninferiority": noninferiority, "footnote": FOOTNOTE}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _changes(m: dict) -> str:
    parts = [f"{k}: -{v['dropped']} +{v['added']}" for k, v in m["subset"]["eligibility_changes"].items()
             if v["dropped"] or v["added"]]
    return "; ".join(parts) or "none"


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    d, issue = doc["dedup"], doc["issue"]
    out = [f"## {doc['title']}", "", banner, f"> {doc['banner']}", "",
           f"Kept {d['n_kept']} of {d['n_all']} images ({d['n_removed']} removed; removed_by_reason sums "
           f"to {d['sum_removed_by_reason']}, overlap {d['overlap']}); `{d['path']}` sha256 `{d['sha256']}`.",
           "", "| removed for | count |", "|---|---|"]
    out += [f"| {x['reason']} | {x['count']} |" for x in doc["removed_by_reason"]]
    out += ["", f"The issue's counts ({issue['source']}; {issue['duplicate_groups']} groups, "
                f"{issue['redundant_files']} redundant files of {issue['files']}):", "", banner,
            "| issue row | count |", "|---|---|"]
    out += [f"| {x['row']} | {x['count']} |" for x in issue["rows"]]
    out += ["", "### Dataset-level mIoU on the subset", "", banner,
            "| stage | seed | roles | all-class full | all-class subset | GT-present subset | disease-only "
            "subset | eligibility changes |", "|---|---|---|---|---|---|---|---|"]
    for m in doc["models"]:
        stage = m["stage"] if not m.get("read_label") else f"{m['stage']} ({m['read_label']})"
        out.append(f"| {stage} | {_fmt(m['seed'])} | {', '.join(m['roles'])} | "
                   f"{_fmt(m['full']['union_present_all_class_float64'])} | "
                   f"{_fmt(m['subset']['union_present_all_class_float64'])} | "
                   f"{_fmt(m['subset']['gt_present_all_class_float64'])} | "
                   f"{_fmt(m['subset']['union_present_disease_only_float64'])} | {_changes(m)} |")
    out += ["", "### The eight contrasts on the subset (unadjusted; B5-D3)", "", banner,
            "| contrast | official p_raw | official | subset p | subset | official label | subset label | "
            "Δ official | Δ subset | Holm (official) | disagrees |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in doc["contrasts"]:
        o, s = c["official"], c["subset"]
        verdict = NOT_EVALUABLE if c["disagrees"] is None else ("; ".join(c["why"]) or "no")
        out.append(f"| {c['comparison_id']} | {_fmt(o['p_raw'])} | {_fmt(o['unadjusted_reject'])} | "
                   f"{_fmt(s['p'])} | {NOT_EVALUABLE if s['p'] is None else _fmt(s['unadjusted_reject'])} | "
                   f"{o['label']} | {s['label']} | {_fmt(o['dataset_level_difference'])} | "
                   f"{_fmt(s['dataset_level_difference'])} | {o['holm_reject']}: {o['holm_label']} | "
                   f"{verdict} |")
    out += ["", "### Intervals on the subset beside the official ones (lrep_descriptive_v1)", "", banner,
            "| contrast | statistic | official [lower, upper] | subset [lower, upper] |", "|---|---|---|---|"]
    for c in doc["contrasts"]:
        for i in c["subset"]["intervals"]:
            o = c["official"]["intervals"][i["statistic_name"]]
            out.append(f"| {c['comparison_id']} | {i['statistic_name']} | [{_fmt(o['lower_bound'])}, "
                       f"{_fmt(o['upper_bound'])}] | [{_fmt(i['lower_bound'])}, {_fmt(i['upper_bound'])}] |")
    n = doc["noninferiority"]
    out += ["", "### E3-vs-E6 non-inferiority on the subset", "", banner,
            "| | passed | lower bound | observed delta |", "|---|---|---|---|",
            f"| official | {n['official']['passed']} | {_fmt(n['official']['lower_bound'])} | "
            f"{_fmt(n['official']['observed_delta'])} |",
            f"| subset | {n['subset']['passed']} | {_fmt(n['subset']['lower_bound'])} | "
            f"{_fmt(n['subset']['observed_delta'])} |", "",
            f"Disagrees: {'yes' if n['disagrees'] else 'no'}.", "", f"_{doc['footnote']}_", ""]
    return out


__all__ = ["ISSUE_ROWS", "ISSUE_TOTALS", "SUBSET_ROLES", "EIGHT", "NOT_EVALUABLE", "check_ids", "am5_on_kept",
           "restrict_pair", "restrict_pooled", "miou_c_on_kept", "unadjusted", "label_of", "compare",
           "subset_interval", "model_rows", "subset_contrast", "subset_noninferiority", "build_dedup",
           "render_markdown"]
