"""The default candidates on TEST (lane LREP, B-2): AM-17 item 6 (PA:435-442), plan §2.2 and D-10
(LRB16).

"two default candidates are also scored, descriptively: E2 with λ = 1 ... and E3 with α = 50 ...;
where a default is the selected run the two coincide. The selected runs remain the runs of record
... No other candidate is evaluated on TEST." (PA:437-441.)

  * The selections of record are DL-109 (λ_logit 0.5) and DL-121 (α_CWD 25); both defaults differ
    from them (K5), so both are scored. SELECTION_RECORD holds the five checkpoints with their DL
    ids; the family E2 and E3 slots at seed 42 must carry the selected ones and each default slot
    (role default_candidate, kind default_candidate: complete, never cut, AM-19 item 2(d)) the
    default one. Any other default_candidate slot is refused.
  * Rows: D1 = E1 -> E2 at λ = 1, in place of the selected E2 for E1 vs E2; D2 = E2 of record -> E3 at
    α = 50, in place of the selected E3 for E2 vs E3. Each sits beside its run-of-record row
    (accuracy_e1_e2, accuracy_e2_e3), which is recomputed here from the 37 inputs and must equal
    the statistics artifact's stored values bitwise (R-11).
  * Values (D-10): m(base), m(candidate) and the dataset-level difference; the candidate's GT-present
    and disease-only dataset-level values (src.stats.eligibility.rule_variants, float64, as the AM-6
    table reads them); per-image disease-only n_paired, mean, median and Hodges-Lehmann shift on the
    pairs align_runs gives under the profile's policy, from the A3a primitives `engineering_shifts`
    and `hodges_lehmann` that bootstrap.descriptive_scalars uses (LRB19, L-14). No p-value, interval,
    Holm decision or AM-17 item 2 label.

Descriptive only. Import-time behaviour is side-effect free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import driver as D
from . import heldout_inputs as H
from .bootstrap import DATASET_MIOU_DELTA
from .noninferiority import pooled_miou_from_totals

#: DL-109 (λ selection of record) and DL-121 (α selection of record): best-checkpoint sha256s.
SELECTION_RECORD = {
    "E2": {"decision": "DL-109", "parameter": "lambda_logit", "selected_value": 0.5, "default_value": 1,
           "selected_run": "e2_s42_lambda0.5_80k_a1",
           "selected_checkpoint": "73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06",
           "default_checkpoint": "7e082eae3aaba14cf7c098bdfc92c5e0c09393dbbab77fedcc4a071df234c4bf",
           "other_checkpoints": {
               "0.25": "20a68e3a130aef0a0d83dfd252b537e654fe60cc54c2d631d937730f5997b164",
               "2": "438d6ee44c1e30cfe843ff14cf742484301af58cf7507b502d84177415397f7d",
               "4": "aadb2aeac5ac104d8925f360171f617b880346bccac317bdc16b5e992dd70ca2"}},
    "E3": {"decision": "DL-121", "parameter": "alpha_cwd", "selected_value": 25, "default_value": 50,
           "selected_run": "e3_s42_lambda0.5_alpha25_80k_a1",
           "selected_checkpoint": "4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d",
           "default_checkpoint": "b891406c5e734a48fece00fce8d3f978bb2bc762833b604214cf3f0df5afaea6",
           "other_checkpoints": {
               "100": "f7201e5a5042e27d420545f0a0075d29686d5102c927bb74bb27a13dc48a328f"}},
}
#: (row, baseline stage, candidate stage, the run-of-record comparison, the clause's wording).
ROWS = (("D1", "E1", "E2", "accuracy_e1_e2", "in place of the selected E2 for E1 vs E2"),
        ("D2", "E2", "E3", "accuracy_e2_e3", "in place of the selected E3 for E2 vs E3"))
RECORD_SENTENCE = ("The selected runs remain the runs of record whatever these comparisons show "
                   "(AM-17 item 6).")
COINCIDES = "coincides with the run of record (AM-17 item 6)"
FOOTNOTE = ("Descriptive: no p-value, interval, Holm decision or AM-17 item 2 label (D-10). Differences "
            "are candidate - baseline, as fractions.")


def check_selection_record(inv: H.Inventory, cells: H.PlannedCells, record=SELECTION_RECORD) -> None:
    """The family E2 and E3 slots at seed 42 carry DL-109's and DL-121's selected checkpoints."""
    bad = []
    for stage in ("E2", "E3"):
        row = H.model_row(inv, cells.slots[(stage, 42)], "fp32")
        if row["source"]["sha256"] != record[stage]["selected_checkpoint"]:
            bad.append(f"{stage}: {row['source']['sha256'][:12]}... is not {record[stage]['decision']}'s "
                       f"selected {record[stage]['selected_checkpoint'][:12]}...")
    if bad:
        raise H.refusal("selection_record_mismatch", "; ".join(bad))


def default_slots(inv: H.Inventory, record=SELECTION_RECORD) -> dict:
    """{stage: slot} of the default candidates: one default_candidate slot per stage that does not
    coincide with its selection, E2 or E3 at seed 42, complete, carrying the default checkpoint."""
    found: dict = {}
    for slot, items in sorted(inv.entries.items()):
        if not any("default_candidate" in e["role"] for e in items):
            continue
        e = inv.entry(slot)
        stage, seed, kind = e.get("stage"), e.get("seed"), e.get("kind")
        if stage not in record or seed != 42:
            raise H.refusal("default_candidate_invalid", f"default_candidate slot {slot!r} is {stage} at "
                                                         f"seed {seed}; only E2 and E3 at seed 42 have one")
        found.setdefault(stage, []).append((slot, kind))
    out = {}
    for stage, rec in record.items():
        got = found.get(stage, [])
        if rec["default_checkpoint"] == rec["selected_checkpoint"]:
            if got:                                   # coincides: no second artifact (AM-17 item 6)
                raise H.refusal("default_candidate_invalid", f"{stage}'s default is its selection, yet "
                                                             f"{[s for s, _ in got]} carry default_candidate")
            continue
        if not got:
            raise H.refusal("default_candidate_missing", f"no default_candidate slot for {stage} "
                                                         f"({rec['parameter']} = {rec['default_value']})")
        slot, kind = got[0]
        problems = []
        if len(got) > 1:
            problems.append(f"{len(got)} default_candidate slots; no other candidate is evaluated on "
                            "TEST (PA:441)")
        if kind != "default_candidate":
            problems.append(f"kind {kind!r}: a default is complete and never cut (AM-19 item 2(d))")
        # a default that is not complete is refused as such, before its model row is read
        row = H.model_row(inv, slot, "fp32") if not problems and inv.has_row(slot, "fp32") else None
        if row is not None and row["source"]["sha256"] != rec["default_checkpoint"]:
            problems.append(f"checkpoint {row['source']['sha256'][:12]}... is not {rec['decision']}'s "
                            f"default {rec['default_checkpoint'][:12]}...")
        if problems:
            raise H.refusal("default_candidate_invalid", f"{stage} ({slot!r}): " + "; ".join(problems))
        if row is None or not inv.has_record(slot, "fp32", "clean", None):
            raise H.refusal("default_candidate_missing", f"{stage} ({slot!r}) has no fp32 model row or "
                                                         "no bound clean artifact")
        out[stage] = slot
    return out


def m_of(stats) -> float:
    return pooled_miou_from_totals(stats.dataset_tp, stats.dataset_gt, stats.dataset_pred)


def pair_values(base_run, cand_run, policy) -> dict:
    """D-10's values for one pair of verified runs."""
    from .align import METRIC_DISEASE_ONLY, AlignmentError, align_runs
    from .eligibility import GT_PRESENT, UNION_PRESENT, rule_variants
    from .tests import engineering_shifts, hodges_lehmann
    try:
        pv = align_runs(base_run, cand_run, policy=policy, metric=METRIC_DISEASE_ONLY)
    except AlignmentError as e:
        raise H.refusal("cross_artifact_mismatch", f"the pair cannot be aligned: {e}") from e
    d = np.asarray(pv.delta, dtype=np.float64)
    shifts = engineering_shifts(d)
    rv = rule_variants(cand_run.stats)
    mb, mc = m_of(base_run.stats), m_of(cand_run.stats)
    return {"m_baseline": mb, "m_candidate": mc, "dataset_level_difference": mc - mb,
            "candidate_gt_present_all_class": rv[GT_PRESENT]["all_class"].value_float64,
            "candidate_gt_present_disease_only": rv[GT_PRESENT]["disease_only"].value_float64,
            "candidate_union_present_disease_only": rv[UNION_PRESENT]["disease_only"].value_float64,
            "per_image": {"metric": METRIC_DISEASE_ONLY, "n_paired": int(d.size),
                          "mean_delta": float(shifts.mean_delta),
                          "median_delta": float(shifts.median_delta),
                          "hodges_lehmann_shift": float(hodges_lehmann(d)),
                          "alignment_status": pv.alignment_status}}


def record_row(stats: H.StatsArtifact, cid: str) -> dict:
    """A run-of-record row recomputed from the 37 inputs, checked bitwise against family.json."""
    b, c, _ = D.COMPARISON_TABLE[cid]
    mine = pair_values(stats.clean(b).run, stats.clean(c).run, stats.policy)
    fam = stats.family
    comp = next(x for x in fam["a3a_family"]["comparisons"] if x["comparison_id"] == cid)
    task = next(t for t in fam["bootstrap_tasks"] if t["task_id"] == f"{cid}__{DATASET_MIOU_DELTA}")
    stored = {"dataset_level_difference": task["observed"], "n_paired": comp["n_paired"],
              "mean_delta": comp["shifts"]["mean_delta"], "median_delta": comp["shifts"]["median_delta"],
              "hodges_lehmann_shift": comp["hodges_lehmann_shift"]}
    got = {"dataset_level_difference": mine["dataset_level_difference"],
           **{k: mine["per_image"][k] for k in ("n_paired", "mean_delta", "median_delta",
                                                 "hodges_lehmann_shift")}}
    bad = [k for k in stored if got[k] != stored[k]]
    if bad:
        raise H.refusal("seed42_mismatch", f"{cid}: {bad} recomputed from the 37 inputs differ from the "
                                           "statistics artifact's")
    return {"comparison_id": cid, "baseline_stage": b, "candidate_stage": c, **mine}


def build_default_candidates(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *,
                             input_root: Path, record=SELECTION_RECORD) -> dict:
    """The "Default candidates on TEST (AM-17 item 6; descriptive)" part (plan §2.4)."""
    H.check_seed42_identity(inv, cells, stats)
    check_selection_record(inv, cells, record)
    slots = default_slots(inv, record)
    bound = {stage: H.load_bound(inv, slot=slot, stage=stage, precision="fp32", scope="clean",
                                 input_root=input_root, policy=stats.policy)
             for stage, slot in slots.items()}
    H.cross_check({**{f"{st} seed 42": stats.clean(st) for st in ("E1", "E2", "E3")},
                   **{f"{st} default": ba for st, ba in bound.items()}}, {},
                  evaluator_commit=inv.evaluator_commit, official=stats.profile == "official")
    rows = []
    for rid, b, c, cid, wording in ROWS:
        rec = record[c]
        out = {"row": rid, "baseline_stage": b, "candidate_stage": c,
               "candidate": f"{c} at {rec['parameter']} = {rec['default_value']} (default; "
                            f"{rec['decision']} selected {rec['selected_value']})",
               "note": wording, "run_of_record": record_row(stats, cid)}
        if c not in bound:
            out.update(coincides=True, slot=None, values=None, note=COINCIDES)
        else:
            ba = bound[c]
            out.update(coincides=False, slot=ba.slot, run_id=ba.summary["run"]["run_id"],
                       checkpoint_sha256=ba.summary["run"].get("checkpoint_sha256"),
                       values=pair_values(stats.clean(b).run, ba.run, stats.policy))
        rows.append(out)
    return {"part": "defaults", "title": "Default candidates on TEST (AM-17 item 6; descriptive)",
            "record_sentence": RECORD_SENTENCE,
            "selection_record": {st: {k: v for k, v in r.items() if k != "other_checkpoints"}
                                 for st, r in record.items()},
            "rows": rows, "footnote": FOOTNOTE}


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.6g}"


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    out = [f"## {doc['title']}", "", banner, doc["record_sentence"], "",
           "| row | pair | m base | m cand. | difference | diff. (pp) | GT-present all | disease-only | "
           "n | mean Δ | median Δ | HL |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        for label, v, pi in ((f"{r['row']} {r['candidate']}", r["values"], None),
                             (f"of record: {r['run_of_record']['comparison_id']}", r["run_of_record"],
                              None)):
            if v is None:
                out.append(f"| {label} | {r['baseline_stage']} -> {r['candidate_stage']} | "
                           f"{r['note']} |" + " |" * 10)
                continue
            p = v["per_image"]
            out.append(f"| {label} | {r['baseline_stage']} -> {r['candidate_stage']} | "
                       f"{_fmt(v['m_baseline'])} | {_fmt(v['m_candidate'])} | "
                       f"{_fmt(v['dataset_level_difference'])} | "
                       f"{100.0 * v['dataset_level_difference']:.2f} | "
                       f"{_fmt(v['candidate_gt_present_all_class'])} | "
                       f"{_fmt(v['candidate_union_present_disease_only'])} | {p['n_paired']} | "
                       f"{_fmt(p['mean_delta'])} | {_fmt(p['median_delta'])} | "
                       f"{_fmt(p['hodges_lehmann_shift'])} |")
    out += ["", f"_{doc['footnote']}_", ""]
    return out


__all__ = ["SELECTION_RECORD", "ROWS", "RECORD_SENTENCE", "COINCIDES", "check_selection_record",
           "default_slots", "pair_values", "record_row", "build_default_candidates", "render_markdown"]
