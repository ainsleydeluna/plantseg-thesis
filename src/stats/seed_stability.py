"""Seed stability and the per-seed tables of dataset-level effects (lane LREP, B-1: L-AM8).

AM-8 (PA:176-188), AM-16 item 1, AM-17 item 4 and AM-19 items 3(c), 4, 5 and 6 (PA:1623-1699), as the
plan of record (lrep_phase_a_plan_20261015.md §1) and rulings LRB4-LRB6 and LRB11-LRB15 read them.
Descriptive only: no test, no p-value and no interval (AM-19 item 6(f); STATS §9.4). Seeds 43 and 44
enter no test (AM-1); nothing here imports a test or p-value object (the X1 check of
scripts/smoke_seed_stability.py).

  * The table has the ten rows of driver.COMPARISON_TABLE, in its order, labelled family,
    descriptive or non-inferiority (LRB13), at seeds 42, 43 and 44.
  * m(S,k) = noninferiority.pooled_miou_from_totals on the clean artifact's float64 pooled totals
    (STATS §9.1; LRB14). A clean row's effect is m(c,k) - m(b,k) (candidate - baseline). The
    robustness row's is mC(E6,k) - mC(E1,k), mC being the mean over the five corruptions of the mean
    over severities 1-3 of each cell's pooled mIoU (DL-73/P28, as report.dataset_miou_c computes it,
    restated over an explicit cell map), at every seed (LRB12).
  * g_k = m(E3,k) - m(E1,k) and d_k = m(E3,k) - m(E6,k). Condition 1: g_k > 0. Condition 2: the mean
    gain math.fsum(g)/3 exceeds the larger of statistics.stdev (n = 3) of E1's and E3's m values.
    Condition 3: d_k < 0.020, so d_k >= 0.020 fails (AM-19 item 6(b): "2.0 pp or more"). Strict
    comparisons on exact float64 values, no tolerance (D-2; LRB14).
  * States, verdict and reasons follow AM-19 item 6(b)-(e) with D-6 and D-7 (LRB15): a missing run
    never gives "stable"; reasons are listed in a fixed order (E3 divergences, then the failing
    conditions in condition order, then "not evaluable: runs missing"); an E2 divergence at seed 43
    or 44 is stated in the verdict sentence and never changes the verdict.
  * Seed 42 comes from the verified statistics artifact's 37 inputs and is checked bitwise against
    its stored values (R-11): the seven clean `<cid>__dataset_miou_delta` observed values, the
    non-inferiority block (both stage values, the delta and the drop), the E6-KD drop, and
    report.dataset_miou_c for mIoU-C.
  * A cut run that wrote VAL rows is shown with its VAL curve and its comparators' VAL scores at the
    same points (AM-19 item 3(c); LRB6): a KD or E1 run's comparators are the runs it is paired with
    in a planned cell at that seed, matched by iteration; a QAT run's is its stage's seed-42
    model-of-record run, matched by epoch; the cut entry's comparator is always shown; a comparator
    without a model that wrote VAL rows (cut, diverged, aborted, non-finite) shows each attempt's
    rows; a derived artifact has no curve; nothing is interpolated.
  * The markdown names runs by stage and seed; TMAN's slot keys stay in the JSON (IF§2).

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import driver as D
from . import heldout_inputs as H
from .bootstrap import DATASET_MIOU_DELTA, DESCRIPTIVE_E1_E3, NONINFERIORITY_E3_E6
from .noninferiority import pooled_miou_from_totals
from .robustness import INFERENTIAL_SEVERITIES

SEEDS = H.SEEDS
DROP_THRESHOLD = 0.020                 # AM-8: 2.0 pp as the stored fraction (noninferiority.PRIMARY_MARGIN)
HOLDS, FAILS, NOT_EVALUABLE = "holds", "fails", "not evaluable"
STABLE, NOT_STABLE = "stable", "not stable"
#: LRB12: robustness_e1_e6 is reported at every seed.
ROBUSTNESS_SEEDS = (42, 43, 44)
ROW_LABELS = {cid: "family" for cid in D.CLEAN_COMPARISON_IDS + (D.ROBUSTNESS_ID,)}
ROW_LABELS.update({DESCRIPTIVE_E1_E3: "descriptive", NONINFERIORITY_E3_E6: "non-inferiority"})
ROW_IDS = tuple(D.COMPARISON_TABLE)
METRIC_TEXT = {D.METRIC_POOLED: "dataset-level all-class union-present mIoU (STATS §9.1)",
               "miou_c": "dataset-level mIoU-C (DL-73; P28)"}
C2_FAILS = "condition 2 fails (mean gain ḡ ≤ max SD s)"
FOOTNOTE = ("Descriptive: no test, no p-value and no interval (AM-19 item 6(f); STATS §9.4). Seeds 43 "
            "and 44 enter no test (AM-1). Effects are candidate - baseline, as fractions.")
LIMITATION = ("the recipe of record diverged at this seed and no clipped rerun was made; Chapters 4 "
              "and 5 state it as a limitation of the seed-42 results (AM-19 item 5(d))")
#: The kinds of a run without a model whose training wrote telemetry (IF§11.3).
TELEMETRY_KINDS = ("cut", "diverged", "aborted", "non_finite")
#: B-5 (e): an attempt whose telemetry holds no VAL row (R2-04).
WROTE_NO_VAL_ROWS = "wrote no VAL rows"


# --------------------------------------------------------------------------------------------------
# 1. the formulas (plan §1.2)
# --------------------------------------------------------------------------------------------------
def m_of(stats) -> float:
    """m(S,k): pooled all-class union-present mIoU from the dataset totals, float64 (STATS §9.1)."""
    return pooled_miou_from_totals(stats.dataset_tp, stats.dataset_gt, stats.dataset_pred)


def miou_c(cell_stats: Mapping[tuple[str, int], object], names: Sequence[str]) -> float:
    """mC: the mean over `names` (the frozen corruption order) of the mean over severities 1-3 of each
    cell's m, as np.float64 means (DL-73/P28; report.dataset_miou_c's arithmetic)."""
    per_corruption = []
    for name in names:
        cells = [m_of(cell_stats[(name, sev)]) for sev in INFERENTIAL_SEVERITIES]
        per_corruption.append(float(np.mean(np.array(cells, dtype=np.float64))))
    return float(np.mean(np.array(per_corruption, dtype=np.float64)))


def mean_and_sds(e1: Sequence[float], e3: Sequence[float]) -> tuple[float, float, float]:
    """(mean gain, SD_E1, SD_E3) over exactly three seeds: math.fsum of the gains / 3 and
    statistics.stdev (n - 1 = 2). R-15: never over fewer than three seeds."""
    if len(e1) != 3 or len(e3) != 3:
        raise AssertionError(f"condition 2 needs E1 and E3 at all three seeds, got {len(e1)} and "
                             f"{len(e3)} values (R-15)")
    gains = [c - b for b, c in zip(e1, e3)]
    return math.fsum(gains) / 3, statistics.stdev(e1), statistics.stdev(e3)


@dataclass(frozen=True)
class ConditionState:
    number: int
    status: str
    values: Mapping[int, float]            # the condition's per-seed values where they exist
    failing_seeds: tuple[int, ...]
    detail: Mapping                        # condition 2: mean gain, both SDs and their max

    def as_json(self) -> dict:
        return {"condition": self.number, "status": self.status,
                "values": {str(k): v for k, v in sorted(self.values.items())},
                "failing_seeds": list(self.failing_seeds), **dict(self.detail)}


def condition_1(g: Mapping[int, float]) -> ConditionState:
    """AM-19 item 6(b): fails if g_k <= 0 at a seed where E1 and E3 exist; holds if they exist at all
    three seeds and every g_k > 0; otherwise not evaluable. `g` holds the seeds where both exist."""
    fails = tuple(k for k in SEEDS if k in g and not g[k] > 0.0)
    if fails:
        status = FAILS
    elif all(k in g for k in SEEDS):
        status = HOLDS
    else:
        status = NOT_EVALUABLE
    return ConditionState(1, status, dict(g), fails, {})


def condition_2(m_e1: Mapping[int, float], m_e3: Mapping[int, float]) -> ConditionState:
    """AM-19 item 6(b): evaluated only if E1 and E3 exist at all three seeds (no mean or SD over two
    seeds); then it holds iff the mean gain exceeds max(SD_E1, SD_E3), strictly."""
    if not all(k in m_e1 and k in m_e3 for k in SEEDS):
        return ConditionState(2, NOT_EVALUABLE, {}, (), {"mean_gain": None, "sd_e1": None,
                                                         "sd_e3": None, "max_sd": None, "n": 3})
    mean, s1, s3 = mean_and_sds([m_e1[k] for k in SEEDS], [m_e3[k] for k in SEEDS])
    status = HOLDS if mean > max(s1, s3) else FAILS
    return ConditionState(2, status, {k: m_e3[k] - m_e1[k] for k in SEEDS}, (),
                          {"mean_gain": mean, "sd_e1": s1, "sd_e3": s3, "max_sd": max(s1, s3), "n": 3})


def condition_3(d: Mapping[int, float]) -> ConditionState:
    """AM-19 item 6(b): fails if d_k >= 0.020 at a seed where E3 and E6 exist; holds if they exist at
    all three seeds and every d_k < 0.020; otherwise not evaluable."""
    fails = tuple(k for k in SEEDS if k in d and not d[k] < DROP_THRESHOLD)
    if fails:
        status = FAILS
    elif all(k in d for k in SEEDS):
        status = HOLDS
    else:
        status = NOT_EVALUABLE
    return ConditionState(3, status, dict(d), fails, {"threshold": DROP_THRESHOLD})


@dataclass(frozen=True)
class Missing:
    """A planned run without a value: its stage, seed, cell label and the decision-log rows of the
    entry that decided it (D-6: the reason names the run and its entry)."""
    stage: str
    seed: int
    label: str
    entry: tuple = ()

    def text(self) -> str:
        rows = f" [{', '.join(self.entry)}]" if self.entry else ""
        return f"{self.stage} seed {self.seed}: {self.label}{rows}"


@dataclass(frozen=True)
class Divergence:
    stage: str
    seed: int
    unit: str                      # "iteration" or "epoch"
    at: int
    rule: str

    def text(self) -> str:
        return f"{self.stage} diverged at seed {self.seed} ({self.unit} {self.at}, {self.rule})"


def verdict(c1: ConditionState, c2: ConditionState, c3: ConditionState, *,
            e3_divergences: Sequence[Divergence], missing: Sequence[Missing],
            e2_divergences: Sequence[Divergence]) -> dict:
    """AM-19 item 6(c)-(e) with D-6 and D-7. `missing` lists the E1, E3 and E6 runs without a value;
    those of a not-evaluable condition are named in its reason."""
    stable = (c1.status == HOLDS and c2.status == HOLDS and c3.status == HOLDS
              and not e3_divergences)
    reasons = [f"E3 diverged at seed {d.seed}" for d in e3_divergences]
    reasons += [f"condition 1 fails at seed {k}" for k in c1.failing_seeds]
    if c2.status == FAILS:
        reasons.append(C2_FAILS)
    reasons += [f"condition 3 fails at seed {k}" for k in c3.failing_seeds]
    needs = set()
    if c1.status == NOT_EVALUABLE or c2.status == NOT_EVALUABLE:
        needs |= {"E1", "E3"}
    if c3.status == NOT_EVALUABLE:
        needs |= {"E3", "E6"}
    named = [m for m in missing if m.stage in needs]
    if named:
        reasons.append("not evaluable: runs missing — " + "; ".join(m.text() for m in named))
    word = STABLE if stable else NOT_STABLE
    sentence = f"Seed stability (AM-8; AM-19 item 6): {word}"
    if reasons:
        sentence += ": " + "; ".join(reasons)
    if e2_divergences:
        sentence += "; " + "; ".join(d.text() for d in e2_divergences)
    return {"verdict": word, "reasons": reasons, "verdict_sentence": sentence + ".",
            "e2_divergences": [d.text() for d in e2_divergences]}


# --------------------------------------------------------------------------------------------------
# 2. loading the cells (plan §1.1)
# --------------------------------------------------------------------------------------------------
@dataclass
class SeedInputs:
    """The clean artifacts of the planned runs with a model, and the E1/E6 corrupted cells."""
    cells: H.PlannedCells
    clean: dict                     # (stage, seed) -> driver.LoadedInput
    corrupted: dict                 # (stage, seed) -> {(name, severity): LoadedInput}
    labels: dict                    # (stage, seed) -> the label of a run without a model


def load_seed_inputs(stats: H.StatsArtifact, inv: H.Inventory, *, input_root: Path) -> SeedInputs:
    """Seed 42 from the 37 (with their identity against the family slots, R-11); seeds 43 and 44
    from the binding (R-6 to R-9); the E1/E6 cells at 43/44 when both runs have a model (LRB11);
    then R-10 over every artifact read, and DL-99's E1 record."""
    cells = H.planned_cells(inv)
    H.check_seed42_identity(inv, cells, stats)
    grid = H.inferential_cells()
    clean, corrupted, labels = {}, {}, {}
    for stage in D.STAGES:
        clean[(stage, 42)] = stats.clean(stage)
    for stage in D.CORRUPTION_STAGES:
        corrupted[(stage, 42)] = {(n, s): stats.cell(stage, n, s) for _, n, s in grid}
    for seed in SEEDS[1:]:
        for stage in D.STAGES:
            st = cells.status[(stage, seed)]
            if st.has_model:
                clean[(stage, seed)] = H.load_clean_of_record(inv, cells, stage, seed,
                                                              input_root=input_root,
                                                              policy=stats.policy).loaded
            else:
                labels[(stage, seed)] = st.label
        if all(cells.has_model(st, seed) for st in D.CORRUPTION_STAGES):
            absent = [(st, cid) for st in D.CORRUPTION_STAGES for cid, _, _ in grid
                      if not inv.has_record(cells.slots[(st, seed)], D.STAGE_PRECISION[st], "corrupted",
                                            cid)]
            if absent:
                raise H.refusal("robustness_cells_missing", f"seed {seed}: E1 and E6 both have a model "
                                                            f"but the binding lacks {len(absent)} of "
                                                            f"their 30 cells, e.g. {absent[:3]} (LRB11)")
            for st in D.CORRUPTION_STAGES:
                corrupted[(st, seed)] = {
                    (n, s): H.load_bound(inv, slot=cells.slots[(st, seed)], stage=st,
                                         precision=D.STAGE_PRECISION[st], scope="corrupted", cell=cid,
                                         input_root=input_root, policy=stats.policy).loaded
                    for cid, n, s in grid}
    H.cross_check({f"{st} seed {k}": li for (st, k), li in sorted(clean.items())},
                  {f"{st} seed {k} {n} s{s}": li for (st, k), m in sorted(corrupted.items())
                   for (n, s), li in m.items()},
                  evaluator_commit=inv.evaluator_commit, official=stats.profile == "official")
    H.check_e1_record({k: clean[("E1", k)].summary["run"].get("checkpoint_sha256")
                       for k in SEEDS if ("E1", k) in clean})
    return SeedInputs(cells=cells, clean=clean, corrupted=corrupted, labels=labels)


# --------------------------------------------------------------------------------------------------
# 3. the tables
# --------------------------------------------------------------------------------------------------
def _cell(values: Mapping, labels: Mapping, b: str, c: str, k: int) -> dict:
    """D-15: a cell with a missing run lists its labels, baseline first, each with its stage."""
    if (b, k) in values and (c, k) in values:
        return {"value": values[(c, k)] - values[(b, k)], "missing": []}
    return {"value": None, "missing": [f"{st}: {labels[(st, k)]}" for st in (b, c) if (st, k) in labels]}


def seed42_cross_checks(stats: H.StatsArtifact, m: Mapping, mc: Mapping) -> list[str]:
    """R-11: every seed-42 value with a stored counterpart equals it bitwise (plan §1.2)."""
    from .report import dataset_miou_c
    fam = stats.family
    tasks = {t["task_id"]: t["observed"] for t in fam["bootstrap_tasks"]}
    ni, e6 = fam["noninferiority_e3_e6"], fam["e6_kd_trigger"]
    checks = []
    for cid in D.CLEAN_COMPARISON_IDS:
        b, c, _ = D.COMPARISON_TABLE[cid]
        checks.append((f"{cid} effect", m[(c, 42)] - m[(b, 42)], tasks[f"{cid}__{DATASET_MIOU_DELTA}"]))
    checks += [("m(E3,42)", m[("E3", 42)], ni["baseline_dataset_miou"]),
               ("m(E6,42)", m[("E6", 42)], ni["candidate_dataset_miou"]),
               (f"{NONINFERIORITY_E3_E6} effect", m[("E6", 42)] - m[("E3", 42)], ni["observed_delta"]),
               ("d_42", m[("E3", 42)] - m[("E6", 42)], ni["observed_drop"]),
               ("d_42 (E6-KD block)", m[("E3", 42)] - m[("E6", 42)], e6["observed_drop"]),
               ("mC(E1,42)", mc[("E1", 42)], dataset_miou_c(stats.views, "E1")),
               ("mC(E6,42)", mc[("E6", 42)], dataset_miou_c(stats.views, "E6"))]
    bad = [f"{name}: {mine!r} != stored {theirs!r}" for name, mine, theirs in checks
           if type(theirs) is not float or mine != theirs]
    if bad:
        raise H.refusal("seed42_mismatch", "; ".join(bad))
    return [name for name, _, _ in checks]


def build_tables(stats: H.StatsArtifact, inputs: SeedInputs) -> dict:
    """The model table, the ten-row effects table, the conditions and the verdict."""
    names = H.corruption_names()
    m = {key: m_of(li.run.stats) for key, li in inputs.clean.items()}
    mc = {key: miou_c({ns: li.run.stats for ns, li in cells.items()}, names)
          for key, cells in inputs.corrupted.items()}
    checked = seed42_cross_checks(stats, m, mc)
    labels = inputs.labels
    rows = []
    for cid in ROW_IDS:
        b, c, _ = D.COMPARISON_TABLE[cid]
        robust = cid == D.ROBUSTNESS_ID       # a pair without both models carries its status (LRB11)
        values = mc if robust else m
        rows.append({"comparison_id": cid, "label": ROW_LABELS[cid], "baseline_stage": b,
                     "candidate_stage": c,
                     "metric": METRIC_TEXT["miou_c" if robust else D.METRIC_POOLED],
                     "cells": {str(k): _cell(values, labels, b, c, k) for k in SEEDS}})
    models = []
    for stage in D.STAGES:
        for k in SEEDS:
            li = inputs.clean.get((stage, k))
            models.append({"stage": stage, "seed": k, "slot": inputs.cells.slots[(stage, k)],
                           "m": m.get((stage, k)), "miou_c": mc.get((stage, k)),
                           "label": None if li is not None else labels[(stage, k)],
                           "run_id": li.summary["run"]["run_id"] if li is not None else None,
                           "checkpoint_sha256": (li.summary["run"].get("checkpoint_sha256")
                                                 if li is not None else None)})
    g = {k: m[("E3", k)] - m[("E1", k)] for k in SEEDS if ("E1", k) in m and ("E3", k) in m}
    d = {k: m[("E3", k)] - m[("E6", k)] for k in SEEDS if ("E3", k) in m and ("E6", k) in m}
    c1 = condition_1(g)
    c2 = condition_2({k: m[("E1", k)] for k in SEEDS if ("E1", k) in m},
                     {k: m[("E3", k)] for k in SEEDS if ("E3", k) in m})
    c3 = condition_3(d)
    divs = divergences(inputs)
    missing = [Missing(st, k, labels[(st, k)],
                       tuple(inputs.cells.status[(st, k)].record.get("dl_rows") or ()))
               for st in ("E1", "E3", "E6") for k in SEEDS if (st, k) not in m]
    v = verdict(c1, c2, c3, e3_divergences=[x for x in divs if x.stage == "E3"], missing=missing,
                e2_divergences=[x for x in divs if x.stage == "E2"])
    return {"rows": rows, "models": models,
            "conditions": {str(c.number): c.as_json() for c in (c1, c2, c3)}, **v,
            "divergences_43_44": [{"stage": x.stage, "seed": x.seed, "unit": x.unit, "at": x.at,
                                   "rule": x.rule, "text": x.text(),
                                   "limitation": LIMITATION if x.stage in ("E2", "E3") else None}
                                  for x in divs],
            "limitation_divergences": [x.text() for x in divs if x.stage in ("E2", "E3")],
            "seed42_cross_checks": {"bitwise_equal": True, "checked": checked}}


def divergences(inputs: SeedInputs) -> list[Divergence]:
    """Every planned run decided by a divergence at seed 43 or 44 (AM-19 items 5 and 9), stage order."""
    out = []
    for stage in D.STAGES:
        for k in SEEDS[1:]:
            st = inputs.cells.status[(stage, k)]
            if not st.has_model and st.kind == "diverged":
                unit = "iteration" if "iteration" in st.record else "epoch"
                out.append(Divergence(stage, k, unit, st.record[unit], st.record["rule"]))
    return out


# --------------------------------------------------------------------------------------------------
# 4. cut runs: VAL curves beside their comparators (AM-19 item 3(c); LRB6)
# --------------------------------------------------------------------------------------------------
def _pairs_of(stage: str) -> list[tuple[str, str]]:
    """(other stage, comparison_id) of every planned row that pairs `stage` with another stage."""
    out = []
    for cid in ROW_IDS:
        b, c, _ = D.COMPARISON_TABLE[cid]
        if stage in (b, c):
            other = c if stage == b else b
            if other not in [o for o, _ in out]:
                out.append((other, cid))
    return out


def _run_of_record(inv: H.Inventory, cells: H.PlannedCells, stage: str, seed: int) -> tuple:
    """(run slot, unit, model row) of a planned run that has VAL rows: the run itself for E1-E3, the
    QAT run behind the selection for E5/E6; E4/E7 are derived and have none (None)."""
    slot = cells.slots[(stage, seed)]
    if stage in H.PTQ_STAGES:
        return None, None, None
    row = (H.model_row(inv, slot, D.STAGE_PRECISION[stage]) if cells.has_model(stage, seed) else None)
    if stage in H.QAT_STAGES:
        return H.parent_slot(inv, slot), "epoch", row
    return slot, "iter", row


def _of_record_attempt(inv, slot, unit, row, input_root):
    attempts = H.read_telemetry(inv, slot, input_root, unit=unit)
    want = row["training_run_id"] if row is not None else None
    hits = [a for a in attempts if a.run_id == want]
    if len(hits) != 1:
        raise H.refusal("telemetry_invalid", f"slot {slot!r}: {len(hits)} telemetry attempts carry the "
                                             f"run of record's training_run_id {want!r}; exactly one "
                                             "is required")
    return hits[0]


def cut_curves(inv: H.Inventory, inputs: SeedInputs, *, input_root: Path) -> list[dict]:
    """One block per cut planned run at seed 43 or 44 (seed-42 runs are never cut, R-13)."""
    cells = inputs.cells
    out = []
    for stage in D.STAGES:
        for k in SEEDS[1:]:
            st = cells.status[(stage, k)]
            if st.has_model or st.kind != "cut":
                continue
            block = {"stage": stage, "seed": k, "slot": cells.slots[(stage, k)],
                     "cut_slot": st.source_slot, "label": st.label, "record": dict(st.record),
                     "attempts": [], "comparators": [], "note": None}
            out.append(block)
            if stage in H.PTQ_STAGES:
                block["note"] = "derived artifact: no VAL curve (LRB6)"
                continue
            unit = "epoch" if stage in H.QAT_STAGES else "iter"
            attempts = H.read_telemetry(inv, st.source_slot, input_root, unit=unit)
            if not attempts:
                block["note"] = "the cut run wrote no VAL rows"
                continue
            points = sorted({u for a in attempts for u, _, _ in a.points})
            block["unit"] = unit
            block["attempts"] = [{"run_id": a.run_id, "telemetry": a.path, "sha256": a.sha256,
                                  "points": [{unit: u, "all_class_miou": v, "nonfinite": t}
                                             for u, v, t in a.points],
                                  "note": None if a.points else WROTE_NO_VAL_ROWS} for a in attempts]
            comps = []                     # (stage, seed, why)
            if stage in H.QAT_STAGES:
                comps.append((stage, 42, f"{stage}'s seed-42 model-of-record run, matched by epoch "
                                         "(LRB6)"))
            else:
                comps += [(o, k, f"paired in {cid} at seed {k}, matched by iteration (LRB6; D-17)")
                          for o, cid in _pairs_of(stage)]
            for o, kk, why in comps:
                block["comparators"].append(_comparator(inv, cells, o, kk, why, unit, points,
                                                        input_root))
            entry_comp = st.record.get("comparator")
            if entry_comp is None:
                continue
            hit = [c for c in block["comparators"] if entry_comp in (c["slot"], c["run_slot"])]
            if not hit:                    # IF§11.3: it is always among LREP's; shown labelled
                key = next((key for key in cells.slots if entry_comp in (
                    cells.slots[key], _run_of_record(inv, cells, *key)[0])), None)
                if key is None:
                    raise H.refusal("status_invalid", f"cut {st.source_slot!r}: its comparator "
                                                      f"{entry_comp!r} is not a planned run (IF§11.3)")
                hit = [_comparator(inv, cells, key[0], key[1], "the cut entry's comparator "
                                   "(IF§11.2)", unit, points, input_root)]
                block["comparators"].append(hit[0])
            hit[0]["cut_entry_comparator"] = True
    return out


def _comparator(inv, cells, stage, seed, why, unit, points, input_root) -> dict:
    slot, cunit, row = _run_of_record(inv, cells, stage, seed)
    out = {"stage": stage, "seed": seed, "slot": cells.slots[(stage, seed)], "run_slot": slot,
           "why": why, "cut_entry_comparator": False, "unit": None, "run_id": None, "values": [],
           "shared_points": 0, "attempts": [], "note": None}
    st = cells.status[(stage, seed)]
    if not st.has_model:
        if slot is None or st.kind not in TELEMETRY_KINDS:
            out["note"] = f"no comparator values: {st.label}"
            return out
        # a paired run without a model that wrote VAL rows: every attempt's scores at the cut run's
        # points (AM-19 item 3(c)); no attempt is "of record", so each is shown with its run_id
        out.update(unit=cunit, note=f"no model ({st.label}): the VAL rows it wrote")
        if cunit != unit:
            out["note"] += (f"; counted in {H.TELEMETRY_UNITS[cunit]}s: no point at the same "
                            f"{H.TELEMETRY_UNITS[unit]}s (never interpolated)")
            return out
        for a in H.read_telemetry(inv, slot, input_root, unit=cunit):
            have, tags = a.values(), a.tags()
            out["attempts"].append({"run_id": a.run_id, "telemetry": a.path, "sha256": a.sha256,
                                    "values": [{unit: u, "all_class_miou": have.get(u),
                                                "nonfinite": tags.get(u)} for u in points],
                                    "shared_points": sum(1 for u in points if u in have),
                                    "note": None if a.points else WROTE_NO_VAL_ROWS})
        return out
    if slot is None:
        out["note"] = "derived artifact: no VAL curve (LRB6)"
        return out
    a = _of_record_attempt(inv, slot, cunit, row, input_root)
    out.update(unit=cunit, run_id=a.run_id, telemetry=a.path, sha256=a.sha256)
    if cunit != unit:
        out["note"] = (f"its VAL rows are counted in {H.TELEMETRY_UNITS[cunit]}s: no point at the same "
                       f"{H.TELEMETRY_UNITS[unit]}s (never interpolated)")
        return out
    have = a.values()
    out["values"] = [{unit: u, "all_class_miou": have.get(u)} for u in points]
    tags = a.tags()                        # B-5 (e), R2-09: a non-finite row keeps its tag
    for v in out["values"]:
        v["nonfinite"] = tags.get(v[unit])
    out["shared_points"] = sum(1 for u in points if u in have)
    if not out["shared_points"]:
        out["note"] = "no VAL point shared with the cut run (never interpolated)"
    return out


# --------------------------------------------------------------------------------------------------
# 5. the part
# --------------------------------------------------------------------------------------------------
def build_seed_stability(stats: H.StatsArtifact, inv: H.Inventory, *, input_root: Path) -> dict:
    """The "Seed stability (AM-8; AM-19 item 6)" part of the extension report (plan §1.6)."""
    inputs = load_seed_inputs(stats, inv, input_root=input_root)
    tables = build_tables(stats, inputs)
    return {"part": "seed", "title": "Seed stability (AM-8; AM-19 item 6)",
            "seeds": list(SEEDS), "robustness_seeds": list(ROBUSTNESS_SEEDS),
            **tables, "cut_runs": cut_curves(inv, inputs, input_root=input_root),
            "footnote": FOOTNOTE}


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.6g}"


def _pp(v) -> str:
    return "—" if v is None else f"{100.0 * v:.2f}"


def _val(v: dict) -> str:
    """A VAL value as printed: "non-finite (<tag>)" for a non-finite row (B-5 (e): R2-09), never the
    absent mark, which stays for a point the run did not write."""
    return f"non-finite ({v['nonfinite']})" if v.get("nonfinite") else _fmt(v["all_class_miou"])


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    """Report section 1.6, in order: the model table, the effects table, the conditions, the verdict
    sentence, the cut-run VAL curves and every divergence at seeds 43 and 44."""
    out = [f"## {doc['title']}", "", "### Models: m(S,k) (and mC for E1 and E6)", "", banner,
           "| stage | seed | m | m (pp) | mC | cell |", "|---|---|---|---|---|---|"]
    for r in doc["models"]:
        out.append(f"| {r['stage']} | {r['seed']} | {_fmt(r['m'])} | {_pp(r['m'])} | "
                   f"{_fmt(r['miou_c'])} | {r['label'] or 'value'} |")
    out += ["", "### Dataset-level effects per seed (candidate - baseline)", "", banner,
            "| row | label | b -> c | seed 42 | seed 43 | seed 44 |", "|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        cells = []
        for k in ("42", "43", "44"):
            c = r["cells"][k]
            cells.append(f"{_fmt(c['value'])} ({_pp(c['value'])} pp)" if c["value"] is not None
                         else "; ".join(c["missing"]))
        out.append(f"| {r['comparison_id']} | {r['label']} | {r['baseline_stage']} -> "
                   f"{r['candidate_stage']} | " + " | ".join(cells) + " |")
    out += ["", "### Conditions", "", banner, "| condition | status | values | detail |",
            "|---|---|---|---|"]
    for n in ("1", "2", "3"):
        c = doc["conditions"][n]
        vals = ", ".join(f"s{k}: {_fmt(v)}" for k, v in c["values"].items()) or "—"
        extra = (f"mean gain {_fmt(c['mean_gain'])}, SD_E1 {_fmt(c['sd_e1'])}, SD_E3 {_fmt(c['sd_e3'])}"
                 if n == "2" else (f"threshold {c['threshold']}" if n == "3" else "gain > 0"))
        out.append(f"| {n} | {c['status']} | {vals} | {extra} |")
    out += ["", f"**{doc['verdict_sentence']}**", ""]
    if doc["cut_runs"]:
        out += ["### Cut runs: VAL curves (AM-19 item 3(c))", ""]
        for b in doc["cut_runs"]:
            out.append(f"- {b['stage']} seed {b['seed']}: {b['label']}"
                       + (f" — {b['note']}" if b["note"] else ""))
            for a in b["attempts"]:
                pts = ", ".join(f"{p[b['unit']]}: {_val(p)}" for p in a["points"])
                out.append(f"  - attempt `{a['run_id']}`: {a['note'] or pts}")
            for c in b["comparators"]:
                tag = " [cut entry's comparator]" if c.get("cut_entry_comparator") else ""
                vals = ", ".join(f"{v[b['unit']]}: {_val(v)}" for v in c["values"])
                out.append(f"  - comparator {c['stage']} seed {c['seed']}{tag} ({c['why']}): "
                           + ("; ".join(x for x in (vals, c["note"]) if x) or "—"))
                for a in c.get("attempts", []):
                    pts = ", ".join(f"{v[b['unit']]}: {_val(v)}" for v in a["values"])
                    out.append(f"    - attempt `{a['run_id']}`: {a['note'] or pts}")
        out.append("")
    divs = doc["divergences_43_44"]
    out += ["### Divergences at seeds 43 and 44 (AM-19 item 9)", "",
            *([f"- {d['text']}" + (f"; {d['limitation']}" if d["limitation"] else "") for d in divs]
              or ["none"]), "", f"_{doc['footnote']}_", ""]
    return out


__all__ = ["SEEDS", "DROP_THRESHOLD", "HOLDS", "FAILS", "NOT_EVALUABLE", "STABLE", "NOT_STABLE",
           "ROBUSTNESS_SEEDS", "ROW_LABELS", "ROW_IDS", "C2_FAILS", "TELEMETRY_KINDS", "m_of", "miou_c",
           "mean_and_sds",
           "ConditionState", "condition_1", "condition_2", "condition_3", "Missing", "Divergence",
           "verdict", "SeedInputs", "load_seed_inputs", "seed42_cross_checks", "build_tables",
           "divergences", "cut_curves", "build_seed_stability", "render_markdown"]
