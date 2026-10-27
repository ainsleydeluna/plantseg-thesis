"""GT-present dataset-level mIoU beside the union-present headline, for every model of record on
TEST (lane LREP, B-3: L-AM6; B-5: the coverage beyond the 21 planned cells and the subset mode).

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
  * Coverage (B-5 (d); DL-24 R3-14; plan §3.2-3.3): with `coverage`, the rows extend to every other
    clean canvas binding record (scope clean, cell null) of any role -- the default candidates, the
    teacher (AM-13), R1, R2, A, F, G, E3-X, E7-X, the 160k controls, the calibration-sensitivity E4/E7,
    E6-KD and AM-23's slots, and any other the binding lists -- each read through load_bound at B5-S1's
    (evaluator stage, model_role, precision), with its model row at its entry's stage and seed (R-14),
    and carrying its slot's roles, kind, stage, seed and IF§11.2 label (None: a scored slot carries its
    value). Never an fq_off or upstream row. The 21 planned rows stay as they are; the seed-42
    faithfulness check applies to the family rows only; the sort covers every mapped stage and a null
    seed (the teachers'); one ground truth holds across every row (R-10). The extension report reads it
    with `coverage` (B5-S4).
  * Per row (scripts/eligibility_variants.py's computation, reused through src.stats.eligibility):
    union-present and GT-present, all-class and disease-only, each as `value` (float32, the
    evaluator's arithmetic) and `value_float64`, with the class counts and the classes that are
    union-present but not GT-present; R1: the union-present values equal summary.json's bitwise, with
    the same eligible counts; R3: GT-present all-class = union-present x n_union / n_gt within 1e-12
    (float64), the extra classes having TP = 0. The JSON carries the per-class table
    (eligibility.per_class_table).
  * The seed-42 rows equal report.gt_present_rows on the same inputs exactly; R-10 runs over the rows
    read, with DL-99's E1 checkpoint record over the planned E1 rows (a guard against a seed swap, as
    the seed part's).
  * Subset mode (plan §3.3; AM-18 item 8(c), TL-13, LRB17): with `subset` (B5-S2's kept ids), each row
    also carries its four numbers on the subset -- its rows dropped by id through manifest_ids and the
    totals rebuilt by eligibility.class_totals on the kept triplets (scripts/dedup_val_scores.py
    :109-116 and :121-131's method), so both eligibility rules are recomputed there -- the eligibility
    changes against the full split (:159's form) and the fingerprint of its image id set. VAL (LRB10) is
    a separate block: the functions take any artifact's statistics.

Descriptive only. Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import driver as D
from . import heldout_inputs as H

RELATION_TOL = 1e-12                       # scripts/eligibility_variants.py RELATION_TOL
FOOTNOTE = ("Descriptive (AM-6; AM-17 item 1(b)): the headline stays union-present. value = the "
            "evaluator's float32 arithmetic; value_float64 = an exactly rounded float64 mean.")
PLANNED_ROLES = ("family", "per_seed")
COVERAGE_SOURCE = "binding: every clean canvas record (B-5 (d); R3-14)"
COVERAGE_TITLE = "Every other clean canvas TEST record (B-5 (d); R3-14)"
_SPACES = ("all_class", "disease_only")


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


def ids_sha256(ids) -> str:
    """The fingerprint of an image id set: sha256 of its sorted ids joined by LF."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def subset_values(stats, summary: dict, kept) -> dict:
    """AM-18 item 8(c) on one artifact (plan §3.3's subset mode): the four numbers on the kept images,
    both eligibility rules recomputed there, and the eligibility changes against the full split
    (scripts/dedup_val_scores.py :109-116 drop by id through manifest_ids, :121-131 rebuild the totals by
    eligibility.class_totals on the kept triplets, :159 the changes); with the fingerprint of the
    artifact's image id set, which its caller compares across artifacts (B5-S2)."""
    from .eligibility import RULES, class_totals, rule_variants
    n_classes = int(summary["dataset"]["num_classes"])
    background = int(summary["dataset"]["background_index"])
    pos = {m: i for i, m in enumerate(stats.manifest_ids)}
    ids = sorted(set(kept))
    missing = [i for i in ids if i not in pos]
    if missing:
        raise H.refusal("dedup_invalid", f"{len(missing)} kept id(s) are not image ids of the artifact, "
                                         f"e.g. {missing[:3]} (B5-S2)")
    index = np.asarray(stats.image_index)
    keep = np.isin(index, np.asarray([pos[i] for i in ids], dtype=index.dtype))
    sub = class_totals({"class_id": np.asarray(stats.class_id)[keep], "tp": np.asarray(stats.tp)[keep],
                        "gt": np.asarray(stats.gt)[keep], "pred": np.asarray(stats.pred)[keep]},
                       num_classes=n_classes)
    full = rule_variants(class_totals(stats, n_classes), background_index=background)
    rv = rule_variants(sub, background_index=background)
    out = {}
    for rule in RULES:
        for space in _SPACES:
            d = rv[rule][space]
            out.update({f"{rule}_{space}": d.value, f"{rule}_{space}_float64": d.value_float64,
                        f"{rule}_{space}_n_classes": d.n_classes})
    out["eligibility_changes"] = {
        rule: {"dropped": sorted(set(full[rule]["all_class"].eligible) - set(rv[rule]["all_class"].eligible)),
               "added": sorted(set(rv[rule]["all_class"].eligible) - set(full[rule]["all_class"].eligible))}
        for rule in RULES}
    out["n_images_kept"] = len(ids)
    out["n_images_removed"] = len(pos) - len(ids)
    out["image_ids_sha256"] = ids_sha256(pos)
    return out


def faithful(row: dict) -> list[str]:
    bad = [f"R1 {k}" for k, v in row["R1"].items() if not (v["bitwise_equal"] and v["n_equal"])]
    if not row["R3"]["holds"]:
        bad.append("R3")
    return bad


def am6_rows(items, *, subset=None) -> list[dict]:
    """`items`: (identity dict, run statistics, summary) per artifact. Refuses an unfaithful one. With
    `subset` (kept image ids), each row also carries its subset values."""
    rows = []
    for ident, stats, summary in items:
        row = {**ident, **gt_present_row(stats, summary)}
        bad = faithful(row)
        if bad:
            raise H.refusal("gt_present_unfaithful", f"{ident}: {bad} (STATS §9.5; ERR E-18)")
        if subset is not None:
            row["subset"] = subset_values(stats, summary, subset)
        rows.append(row)
    return rows


def is_planned(row: dict) -> bool:
    """A row of the 21 planned cells (Part 1's rows: role family or per_seed)."""
    return row.get("role") in PLANNED_ROLES


def coverage_items(stats: H.StatsArtifact, inv: H.Inventory, rec_slots, *, input_root: Path) -> tuple:
    """B-5 (d): (items, loaded) of every clean canvas binding record of a slot that is not a planned
    cell's (never an fq_off row), read through load_bound at B5-S1's stage map, its model row at its
    entry's stage and seed (R-14)."""
    items, loaded = [], {}
    listed = sorted({(r["slot"], r["precision"]) for r in inv.doc["binding"].get("artifacts", [])
                     if r.get("scope") == "clean" and r.get("cell") is None and r.get("slot") not in rec_slots
                     and r.get("precision") != H.PRECISION_FQOFF})
    for slot, precision in listed:
        e = inv.entry(slot)
        stage, seed = e.get("stage"), e.get("seed")
        ba = H.load_bound(inv, slot=slot, stage=stage, precision=precision, scope="clean",
                          input_root=input_root, policy=stats.policy)
        H.check_row_at_entry(inv, ba)
        loaded[f"{slot} {precision}"] = ba
        items.append(({"slot": slot, "stage": stage, "seed": seed, "roles": list(e["role"]),
                       "kind": e.get("kind"), "label": None, "precision": precision,
                       "run_id": ba.summary["run"]["run_id"],
                       "checkpoint_sha256": ba.summary["run"].get("checkpoint_sha256"),
                       "source": COVERAGE_SOURCE}, ba.run.stats, ba.summary))
    return items, loaded


def _sort_key(r: dict):
    """Seed (a null seed, the teachers', last), then B5-S1's stage order, then the slot."""
    return (r["seed"] is None, r["seed"] if r["seed"] is not None else 0, H.STAGE_ORDER.index(r["stage"]),
            r["slot"])


def build_am6_tables(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *,
                     input_root: Path, coverage: bool = False, subset=None) -> dict:
    """The "AM-6 / AM-17 item 1(b): GT-present on TEST for every model of record" part; with `coverage`,
    every other clean canvas record too (B-5 (d)); with `subset`, each row's values on those image ids
    (AM-18 item 8(c))."""
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
    if coverage:
        more, more_loaded = coverage_items(stats, inv, rec_slots, input_root=input_root)
        items += more
        loaded.update(more_loaded)
    H.cross_check(loaded, {}, evaluator_commit=inv.evaluator_commit, official=stats.profile == "official")
    H.check_e1_record({ident["seed"]: ident["checkpoint_sha256"] for ident, _, _ in items
                       if ident["stage"] == "E1" and ident.get("role") in PLANNED_ROLES})
    rows = am6_rows(items, subset=subset)
    ref = {g["stage"]: g for g in gt_present_rows(stats.views)}
    keys = [k for k in ref[D.STAGES[0]] if k != "stage"]
    differ = []
    for r in rows:
        if r.get("role") != "family":
            continue
        mine = {k: r[k] if k.endswith("_n_classes") else r[f"{k}_float64"] for k in keys}
        if mine != {k: ref[r["stage"]][k] for k in keys}:
            differ.append(r["stage"])
    if differ:
        raise H.refusal("gt_present_unfaithful", f"seed-42 rows of {differ} differ from "
                                                 "report.gt_present_rows on the same inputs")
    rows.sort(key=_sort_key)
    return {"part": "am6", "title": "GT-present dataset-level mIoU on TEST (AM-6; AM-17 item 1(b); "
                                    "descriptive)",
            "rows": rows, "missing": labels, "footnote": FOOTNOTE}


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.6g}"


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    """The planned rows' table as Part 1 renders it; then (B-5 (d)) every other clean canvas record's."""
    out = [f"## {doc['title']}", "", banner,
           "| stage | seed | union-present all-class | GT-present all-class | union-present "
           "disease-only | GT-present disease-only | n union / n GT | union-present only |",
           "|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        if not is_planned(r):
            continue
        out.append(f"| {r['stage']} | {r['seed']} | {_fmt(r['union_present_all_class'])} | "
                   f"{_fmt(r['gt_present_all_class'])} | {_fmt(r['union_present_disease_only'])} | "
                   f"{_fmt(r['gt_present_disease_only'])} | {r['union_present_all_class_n_classes']} / "
                   f"{r['gt_present_all_class_n_classes']} | {r['union_present_not_gt_present']} |")
    for m in doc["missing"]:
        out.append(f"| {m['stage']} | {m['seed']} | {m['label']} | | | | | |")
    others = [r for r in doc["rows"] if not is_planned(r)]
    if others:
        out += ["", f"### {COVERAGE_TITLE}", "", banner,
                "| stage | seed | roles | precision | union-present all-class | GT-present all-class | "
                "union-present disease-only | GT-present disease-only | n union / n GT |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in others:
            seed = "—" if r["seed"] is None else r["seed"]
            out.append(f"| {r['stage']} | {seed} | {', '.join(r['roles'])} | "
                       f"{r['precision']} | {_fmt(r['union_present_all_class'])} | "
                       f"{_fmt(r['gt_present_all_class'])} | {_fmt(r['union_present_disease_only'])} | "
                       f"{_fmt(r['gt_present_disease_only'])} | {r['union_present_all_class_n_classes']} / "
                       f"{r['gt_present_all_class_n_classes']} |")
    out += ["", f"_{doc['footnote']}_", ""]
    return out


__all__ = ["RELATION_TOL", "PLANNED_ROLES", "COVERAGE_SOURCE", "COVERAGE_TITLE", "gt_present_row",
           "ids_sha256", "subset_values", "faithful", "am6_rows", "is_planned", "coverage_items",
           "build_am6_tables", "render_markdown"]
