"""GT-present dataset-level mIoU beside the union-present headline, for every model of record on
TEST (lane LREP, B-3: L-AM6).

AM-6 (PA:138-140): "A sensitivity mIoU over the classes present in TEST ground truth is reported
descriptively"; AM-17 item 1(b) (PA:362-364): "for every model on VAL and TEST"; STATS §9.5
(STATS:709-721; ERR E-18): the four numbers, the per-class IoU and both eligibility flags, the float32
values reproducing summary.json bitwise, float64 beside them; ERR E-24: canvas artifacts only.

  * Rows: the planned runs of record (`family` slots at seed 42, `per_seed` slots at 43/44; IF§11.1),
    at the stage's own precision (never a qat_fakequant_off row). Seed 42 reads the 37 inputs of the
    verified statistics artifact (their identity checked against the family slots, R-11); seeds 43/44
    read the clean records the binding lists for the per_seed slots (LRB7: the list comes from the
    binding, not from a hard-coded list), each checked against its record (R-7). A planned run without
    a model is listed with its IF§11.2 label.
  * Per row (scripts/eligibility_variants.py's computation, reused through src.stats.eligibility):
    union-present and GT-present, all-class and disease-only, each as `value` (float32, the
    evaluator's arithmetic) and `value_float64`, with the class counts and the classes that are
    union-present but not GT-present; R1: the union-present values equal summary.json's bitwise, with
    the same eligible counts; R3: GT-present all-class = union-present x n_union / n_gt within 1e-12
    (float64), the extra classes having TP = 0. The JSON carries the per-class table
    (eligibility.per_class_table).
  * The seed-42 rows equal report.gt_present_rows on the same inputs exactly.
  * VAL (LRB10) and the AM-18 item 8(c) subset (LRB17) are not here: the function works on any split,
    and those invocations belong to a separate block and to B-5.

Descriptive only. Import-time behaviour is side-effect free.
"""
from __future__ import annotations

from pathlib import Path

from . import driver as D
from . import heldout_inputs as H

RELATION_TOL = 1e-12                       # scripts/eligibility_variants.py RELATION_TOL
FOOTNOTE = ("Descriptive (AM-6; AM-17 item 1(b)): the headline stays union-present. value = the "
            "evaluator's float32 arithmetic; value_float64 = an exactly rounded float64 mean.")


def gt_present_row(stats, summary: dict, *, with_classes: bool = True) -> dict:
    """The four numbers, the class counts, R1 and R3 for one artifact's sufficient statistics."""
    from .eligibility import GT_PRESENT, RULES, UNION_PRESENT, class_totals, per_class_table, rule_variants
    totals = class_totals(stats, int(summary["dataset"]["num_classes"]))
    rv = rule_variants(totals, background_index=int(summary["dataset"]["background_index"]))
    dl = summary["dataset_level"]
    r1 = {}
    for scope, key in (("all_class", "all_class_miou"), ("disease_only", "disease_only_miou")):
        mine = rv[UNION_PRESENT][scope]
        r1[key] = {"summary": dl.get(key), "reproduced": mine.value,
                   "bitwise_equal": isinstance(dl.get(key), float) and mine.value == dl.get(key),
                   "n_equal": mine.n_classes == dl.get(f"{key}_n_eligible")}
    up, gp = rv[UNION_PRESENT]["all_class"], rv[GT_PRESENT]["all_class"]
    extra = sorted(set(up.eligible) - set(gp.eligible))
    r3 = {"n_union_present": up.n_classes, "n_gt_present": gp.n_classes,
          "union_present_not_gt_present": extra, "their_tp": [int(totals.tp[c]) for c in extra],
          "tolerance": RELATION_TOL, "holds": False}
    if up.value_float64 is not None and gp.value_float64 is not None and set(gp.eligible) <= set(up.eligible):
        scaled = up.value_float64 * up.n_classes / gp.n_classes
        r3.update(abs_diff=abs(gp.value_float64 - scaled),
                  holds=abs(gp.value_float64 - scaled) <= RELATION_TOL and not any(r3["their_tp"]))
    row = {f"{rule}_{space}": rv[rule][space].value for rule in RULES for space in ("all_class",
                                                                                   "disease_only")}
    row.update({f"{rule}_{space}_float64": rv[rule][space].value_float64 for rule in RULES
                for space in ("all_class", "disease_only")})
    row.update({f"{rule}_{space}_n_classes": rv[rule][space].n_classes for rule in RULES
                for space in ("all_class", "disease_only")})
    row["union_present_not_gt_present"] = extra
    row["R1"] = r1
    row["R3"] = r3
    if with_classes:
        row["per_class"] = per_class_table(totals)
    return row


def faithful(row: dict) -> list[str]:
    bad = [f"R1 {k}" for k, v in row["R1"].items() if not (v["bitwise_equal"] and v["n_equal"])]
    if not row["R3"]["holds"]:
        bad.append("R3")
    return bad


def am6_rows(items) -> list[dict]:
    """`items`: (identity dict, run statistics, summary) per artifact. Refuses an unfaithful one."""
    rows = []
    for ident, stats, summary in items:
        row = {**ident, **gt_present_row(stats, summary)}
        bad = faithful(row)
        if bad:
            raise H.refusal("gt_present_unfaithful", f"{ident}: {bad} (STATS §9.5; ERR E-18)")
        rows.append(row)
    return rows


def build_am6_tables(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *,
                     input_root: Path) -> dict:
    """The "AM-6 / AM-17 item 1(b): GT-present on TEST for every model of record" part."""
    from .report import gt_present_rows
    H.check_seed42_identity(inv, cells, stats)
    rec_slots = {slot: key for key, slot in cells.slots.items()}
    items, labels, loaded = [], [], {}
    for stage in D.STAGES:
        li = stats.clean(stage)
        loaded[f"{stage} seed 42"] = li
        items.append(({"slot": cells.slots[(stage, 42)], "stage": stage, "seed": 42, "role": "family",
                       "run_id": li.summary["run"]["run_id"],
                       "checkpoint_sha256": li.summary["run"].get("checkpoint_sha256"),
                       "source": "statistics artifact (the 37)"}, li.run.stats, li.summary))
    listed = sorted({(r["slot"], r["precision"]) for r in inv.doc["binding"].get("artifacts", [])
                     if r.get("scope") == "clean" and r.get("cell") is None and r.get("slot") in rec_slots})
    for slot, precision in listed:
        stage, seed = rec_slots[slot]
        if seed == 42 or precision != D.STAGE_PRECISION[stage]:
            continue                       # the 37's block; never a qat_fakequant_off row
        if not cells.has_model(stage, seed):
            continue                       # refused by planned_cells (R-6, artifact_for_missing_run)
        ba = H.load_bound(inv, slot=slot, stage=stage, precision=precision, scope="clean",
                          input_root=input_root, policy=stats.policy)
        loaded[f"{stage} seed {seed}"] = ba
        items.append(({"slot": slot, "stage": stage, "seed": seed, "role": "per_seed",
                       "run_id": ba.summary["run"]["run_id"],
                       "checkpoint_sha256": ba.summary["run"].get("checkpoint_sha256"),
                       "source": "binding"}, ba.run.stats, ba.summary))
    for (stage, seed), st in sorted(cells.status.items()):
        if seed != 42 and st.has_model and f"{stage} seed {seed}" not in loaded:
            H.binding_record(inv, cells.slots[(stage, seed)], D.STAGE_PRECISION[stage], "clean")
        if not st.has_model:
            labels.append({"slot": st.slot, "stage": stage, "seed": seed, "label": st.label})
    H.cross_check(loaded, {}, evaluator_commit=inv.evaluator_commit, official=stats.profile == "official")
    rows = am6_rows(items)
    ref = {g["stage"]: g for g in gt_present_rows(stats.views)}
    keys = [k for k in ref[D.STAGES[0]] if k != "stage"]
    differ = []
    for r in rows:
        if r["seed"] != 42:
            continue
        mine = {k: r[k] if k.endswith("_n_classes") else r[f"{k}_float64"] for k in keys}
        if mine != {k: ref[r["stage"]][k] for k in keys}:
            differ.append(r["stage"])
    if differ:
        raise H.refusal("gt_present_unfaithful", f"seed-42 rows of {differ} differ from "
                                                 "report.gt_present_rows on the same inputs")
    order = {st: i for i, st in enumerate(D.STAGES)}
    rows.sort(key=lambda r: (r["seed"], order[r["stage"]]))
    return {"part": "am6", "title": "GT-present dataset-level mIoU on TEST (AM-6; AM-17 item 1(b); "
                                    "descriptive)",
            "rows": rows, "missing": labels, "footnote": FOOTNOTE}


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.6g}"


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    out = [f"## {doc['title']}", "", banner,
           "| stage | seed | union-present all-class | GT-present all-class | union-present "
           "disease-only | GT-present disease-only | n union / n GT | union-present only |",
           "|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        out.append(f"| {r['stage']} | {r['seed']} | {_fmt(r['union_present_all_class'])} | "
                   f"{_fmt(r['gt_present_all_class'])} | {_fmt(r['union_present_disease_only'])} | "
                   f"{_fmt(r['gt_present_disease_only'])} | {r['union_present_all_class_n_classes']} / "
                   f"{r['gt_present_all_class_n_classes']} | {r['union_present_not_gt_present']} |")
    for m in doc["missing"]:
        out.append(f"| {m['stage']} | {m['seed']} | {m['label']} | | | | | |")
    out += ["", f"_{doc['footnote']}_", ""]
    return out


__all__ = ["RELATION_TOL", "gt_present_row", "faithful", "am6_rows", "build_am6_tables",
           "render_markdown"]
