"""Rare-class strata TEST tables (lane LREP, B-2): AM-17 item 8 (PA:456-462), plan §2.1, D-8 and D-9
(LRB16).

"After TEST, per-class IoU changes are tabulated by stratum for E1->E2, E2->E3, E1->E3, E1->E4,
E1->E5, E1->E6, E3->E6 and E3->E7, with the classes whose IoU falls to 0 after INT8 conversion;
classes with no TEST ground truth are reported as not evaluable. No test." (PA:459-462.)

  * Inputs: the seven seed-42 clean canvas artifacts of the verified statistics artifact (D-9), each
    checked against its family slot (R-11), and the strata file (perclass.load_strata and
    strata_index: exactly the disease classes 1..115, one tercile each). In the official profile the
    file must be the committed DL-63 file at its sha256; otherwise any file is read and its sha256
    and Git state recorded.
  * Per disease class and pair (b -> c): stratum, rare flag, TRAIN share and rank, TEST gt_support
    (one ground truth: identical across the seven models), IoU_b and IoU_c from summary.per_class.iou
    (the exact float64 TP/union), delta = IoU_c - IoU_b; a class with gt_support 0 is not evaluable,
    its delta null, and it enters no summary.
  * Per group (T1, T2, T3, rare, overall): evaluable and not-evaluable counts, mean delta (math.fsum /
    n), median delta (np.median, float64), and the counts candidate better, baseline better and exact
    ties -- perclass._summary's arithmetic under neutral names. Background (class 0) has its own line.
  * "Falls to 0 after INT8 conversion" (D-8): only the four direct conversions E1->E4, E1->E5, E3->E6
    and E3->E7 list the evaluable classes with IoU_b > 0 and IoU_c == 0 (intersection 0); E1->E6
    shows its deltas but no list.
  * Integrity per model: perclass.model_integrity (the per-class supports reproduce the summary's
    all-class and disease-only union-present mIoU bitwise, with equal eligible counts; the IoU column
    equals summary.per_class.iou and intersection / union).

Descriptive only. Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import driver as D
from . import heldout_inputs as H

PAIRS = (("E1", "E2"), ("E2", "E3"), ("E1", "E3"), ("E1", "E4"), ("E1", "E5"), ("E1", "E6"),
         ("E3", "E6"), ("E3", "E7"))
#: D-8: the candidate is the INT8 conversion of the baseline (src/eval/stage_artifacts.py:46-49).
ZERO_IOU_PAIRS = (("E1", "E4"), ("E1", "E5"), ("E3", "E6"), ("E3", "E7"))
GROUPS = ("T1", "T2", "T3", "rare", "overall")
TERCILES = ("T1", "T2", "T3")
STRATA_REL = "reports/strata/train_strata_v1.json"
#: DL-63: the strata file of record.
STRATA_SHA256_OF_RECORD = "ab0df4e3af32f0d25d416700effe51e036cb83ca1c9d451e7e80c9077bdc65b6"
OK, NOT_EVALUABLE = "ok", "not_evaluable"


def load_strata_checked(path, *, official: bool) -> tuple[dict, dict]:
    """(strata index, file record). Refuses an unreadable or malformed file (strata_invalid) and, in
    the official profile, any file but the committed DL-63 one (strata_not_of_record)."""
    from .perclass import PerclassError, load_strata, strata_index
    from .report import CODE_REPO, mde_entry_status
    try:
        doc, sha = load_strata(path)
        index = strata_index(doc, range(1, 116))
    except (PerclassError, OSError, ValueError, UnicodeDecodeError, AttributeError) as e:
        raise H.refusal("strata_invalid", f"strata file {path}: {type(e).__name__}: {e}") from e
    status = mde_entry_status(path)               # sha256 and Git state, as for the MDE entry
    record = {"path": str(Path(path).resolve()), "sha256": sha, "schema": doc.get("schema"),
              "inside_repository": status["inside_repository"], "tracked": status["tracked"],
              "clean_at_head": status["clean_at_head"], "last_commit": status["last_commit"],
              "of_record": (sha == STRATA_SHA256_OF_RECORD and status["tracked"] is True
                            and status["clean_at_head"] is True
                            and Path(path).resolve() == (CODE_REPO / STRATA_REL).resolve())}
    if official and not record["of_record"]:
        raise H.refusal("strata_not_of_record", f"the official profile reads {STRATA_REL} committed at "
                                                f"sha256 {STRATA_SHA256_OF_RECORD} (DL-63); got {record}")
    return index, record


def _per_class(summary: dict, stage: str) -> dict:
    pc = summary.get("per_class") or {}
    keys = ("class_ids", "gt_support", "pred_support", "intersection", "union", "iou", "iou_status")
    n = len(pc.get("class_ids") or [])
    if (any(k not in pc for k in keys) or pc["class_ids"] != list(range(n))
            or any(len(pc[k]) != n for k in keys) or n != 116):
        raise H.refusal("strata_integrity", f"{stage}: summary.per_class is not classes 0..115 with "
                                            "every array the same length")
    return pc


def model_checks(summaries: dict) -> dict:
    """Per model, perclass.model_integrity; and one TEST ground truth across the seven models."""
    from .perclass import PerclassError, model_integrity
    out, problems = {}, []
    for stage, s in summaries.items():
        pc = _per_class(s, stage)
        try:
            rec, probs = model_integrity(s, list(pc["iou"]))
        except PerclassError as e:
            rec, probs = None, [str(e)]
        out[stage] = rec
        problems += [f"{stage}: {p}" for p in probs]
    if problems:
        raise H.refusal("strata_integrity", "; ".join(problems))
    gts = {stage: [int(v) for v in s["per_class"]["gt_support"]] for stage, s in summaries.items()}
    first = gts[D.STAGES[0]]
    differ = [st for st, g in gts.items() if g != first]
    if differ:
        raise H.refusal("strata_gt_mismatch", f"TEST gt_support of {differ} differs from E1's: the "
                                              "seven models must score one ground truth")
    return out


def _summary(deltas: list, n_not_evaluable: int) -> dict:
    n = len(deltas)
    return {"n_classes": n, "n_not_evaluable": n_not_evaluable,
            "mean_delta": (math.fsum(deltas) / n) if n else None,
            "median_delta": float(np.median(np.asarray(deltas, dtype=np.float64))) if n else None,
            "candidate_better": sum(1 for d in deltas if d > 0),
            "baseline_better": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0)}


def pair_table(b: str, c: str, summaries: dict, strata: dict) -> dict:
    """One pair's per-class rows, background line, group summaries and (D-8) zero-IoU list."""
    pb, pc = summaries[b]["per_class"], summaries[c]["per_class"]
    rows = []
    for cls in range(1, 116):
        gt = int(pb["gt_support"][cls])
        ib, ic = pb["iou"][cls], pc["iou"][cls]
        evaluable = gt > 0
        if evaluable and (ib is None or ic is None):
            raise H.refusal("strata_integrity", f"{b}->{c} class {cls}: ground truth but an undefined "
                                                "IoU (malformed per_class)")
        s = strata[cls]
        rows.append({"class_id": cls, "stratum": s["tercile"], "rare": s["rare"],
                     "train_share": s["share"], "train_rank": s["rank"], "gt_support": gt,
                     "iou_baseline": ib, "iou_candidate": ic,
                     "intersection_candidate": int(pc["intersection"][cls]),
                     "delta": (ic - ib) if evaluable else None,
                     "status": OK if evaluable else NOT_EVALUABLE})
    bg = {"class_id": 0, "stratum": "background", "gt_support": int(pb["gt_support"][0]),
          "iou_baseline": pb["iou"][0], "iou_candidate": pc["iou"][0],
          "delta": (pc["iou"][0] - pb["iou"][0]) if int(pb["gt_support"][0]) > 0 else None}
    members = {"T1": lambda r: r["stratum"] == "T1", "T2": lambda r: r["stratum"] == "T2",
               "T3": lambda r: r["stratum"] == "T3", "rare": lambda r: r["rare"] is True,
               "overall": lambda r: True}
    groups = {}
    for g in GROUPS:
        sel = [r for r in rows if members[g](r)]
        groups[g] = _summary([r["delta"] for r in sel if r["status"] == OK],
                             sum(1 for r in sel if r["status"] == NOT_EVALUABLE))
    out = {"baseline_stage": b, "candidate_stage": c, "groups": groups, "background": bg,
           "rows": rows, "zero_iou_after_int8": None}
    if (b, c) in ZERO_IOU_PAIRS:
        zero = [r for r in rows if r["status"] == OK and r["iou_baseline"] > 0
                and r["intersection_candidate"] == 0]
        out["zero_iou_after_int8"] = {
            "classes": [{k: r[k] for k in ("class_id", "stratum", "rare", "iou_baseline")} for r in zero],
            "per_group": {g: sum(1 for r in zero if members[g](r)) for g in GROUPS}}
    return out


def build_strata_tables(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *,
                        strata_path, official: bool) -> dict:
    """The "Rare-class strata on TEST (AM-17 item 8; descriptive)" part (plan §2.4)."""
    H.check_seed42_identity(inv, cells, stats)
    strata, record = load_strata_checked(strata_path, official=official)
    summaries = {stage: stats.clean(stage).summary for stage in D.STAGES}
    integrity = model_checks(summaries)
    pairs = [pair_table(b, c, summaries, strata) for b, c in PAIRS]
    not_evaluable = [r["class_id"] for r in pairs[0]["rows"] if r["status"] == NOT_EVALUABLE]
    return {"part": "strata", "title": "Rare-class strata on TEST (AM-17 item 8; descriptive)",
            "seed": 42, "strata_file": record,
            "tercile_sizes": {t: sum(1 for c in strata.values() if c["tercile"] == t) for t in TERCILES},
            "rare_classes": sum(1 for c in strata.values() if c["rare"]),
            "not_evaluable_classes": not_evaluable, "integrity": integrity, "pairs": pairs,
            "footnote": "Descriptive: no test (AM-17 item 8). Deltas are candidate - baseline IoU, "
                        "as fractions; a class with no TEST ground truth is not evaluable."}


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.6g}"


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    out = [f"## {doc['title']}", "", banner,
           f"Strata file `{doc['strata_file']['path']}` sha256 `{doc['strata_file']['sha256']}` (of "
           f"record: {doc['strata_file']['of_record']}); terciles {doc['tercile_sizes']}, "
           f"{doc['rare_classes']} rare classes; not evaluable on TEST: {doc['not_evaluable_classes']}.",
           "", "| pair | group | n | not evaluable | mean Δ | mean Δ (pp) | median Δ | cand. better | "
               "base better | ties |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in doc["pairs"]:
        for g in GROUPS:
            s = p["groups"][g]
            pp = "—" if s["mean_delta"] is None else f"{100.0 * s['mean_delta']:.2f}"
            out.append(f"| {p['baseline_stage']} -> {p['candidate_stage']} | {g} | {s['n_classes']} | "
                       f"{s['n_not_evaluable']} | {_fmt(s['mean_delta'])} | {pp} | "
                       f"{_fmt(s['median_delta'])} | {s['candidate_better']} | {s['baseline_better']} | "
                       f"{s['ties']} |")
    out += ["", "Classes whose IoU falls to 0 after INT8 conversion (D-8):", ""]
    for p in doc["pairs"]:
        z = p["zero_iou_after_int8"]
        if z is not None:
            ids = ", ".join(f"{c['class_id']} ({c['stratum']}{', rare' if c['rare'] else ''})"
                            for c in z["classes"]) or "none"
            out.append(f"- {p['baseline_stage']} -> {p['candidate_stage']}: {ids}")
    out += ["", f"_{doc['footnote']}_", ""]
    return out


__all__ = ["PAIRS", "ZERO_IOU_PAIRS", "GROUPS", "STRATA_REL", "STRATA_SHA256_OF_RECORD",
           "load_strata_checked", "model_checks", "pair_table", "build_strata_tables",
           "render_markdown"]
