"""AM-17 item 9 dress rehearsal of the metrics -> Holm -> BCa pipeline on the E1 seed 42/43/44 VAL
outputs (lane L-STATS-OFFICIAL).

The rehearsal writes no section 12 artifact: its outputs are `nonofficial`, `"mode": "rehearsal"`
appears only in `rehearsal_manifest.json`, and the module imports neither `build_family` nor the
statistics writer. Its files are rehearsal_manifest.json, rehearsal_results.json,
rehearsal_bootstrap.npz, report.md and SHA256SUMS, written atomically (temporary sibling, SHA256SUMS,
rename) under <out-root>/<UTC>/, where the out-root is outside the repository (orchestrator Q10).

Inputs are read by `src.stats.val_artifacts.load_val_artifact` (verify, canvas guard, layout, the
REHEARSAL row rules: VAL only, 846 rows, a real-run artifact) and must each be an E1 student fp32
artifact; P20 refuses, by name, a per-image value that is not a float, not float32-representable or
outside [0, 1], and a row whose clean_image_id differs from its image_id.

Mapping (lane item 6), namespace plantseg_primary_analysis_v1 (Q4):
  accuracy_e1_e2   s42 -> s43            descriptive_e1_e3      s42 -> s44
  accuracy_e2_e3   s43 -> s44            noninferiority_e3_e6   s44 -> s42  (E3 := s44, E6 := s42)
  accuracy_e1_e6   s42 -> s44
Not rehearsed (`not_rehearsed`): accuracy_e4_e5, accuracy_e7_e6, accuracy_e4_e7 and accuracy_e5_e6
are self-pairs align_runs(s42, s42), and robustness_e1_e6 is the mIoU-C self-pair of s42; all five
have p = 1 by section 4.2's all-zero rule and are excluded from every criterion, the INVESTIGATE
trigger and every label (P22). Robustness is rehearsed structurally: s42 and s44 are each relabelled
as the 15 inferential corruption cells, the unmodified assemble_miou_c rebuilds mIoU-C, which must
equal the clean values exactly (P21), and the structural comparison s42 -> s44 must equal
accuracy_e1_e6 in every field except comparison_id and metric. 19 tasks are bootstrapped: the three
rehearsed comparisons' four tasks, the three structural robustness tasks, the three descriptive tasks
and the non-inferiority task; the 16 tasks of the four clean placeholders are not run.

Criteria (P23 and acceptance (e); any failure exits 1, outputs kept): closed containment of every
rehearsed interval (lower <= estimate <= upper; lower <= estimate one-sided), with z0 and p0 printed
on failure; every rehearsed p-value and Holm-adjusted p in [0, 1]; Holm-adjusted p equal to
statsmodels multipletests(method='holm') to 1e-12 and Holm decisions equal to a hand-rolled strict
Holm, for the 8-entry family and the 3-member family; t statistic and mean difference of equal sign
(sign(0) = 0; skipped for a null statistic) and t_test.mean_difference == shifts.mean_delta ==
cohens_dz.mean bitwise; per-image counts 846 - K_val and dataset-level counts 846; the 15-cell
invariant; the structural comparison. INVESTIGATE (AM-17 item 9; O5) fires on a rejection in the
8-entry Holm family or in the 3-member Holm of the three rehearsed tests; it is printed, never a
failure.

Report (lane item 7): the family table, the 3-member Holm, structural robustness, descriptive, NI with
its sensitivity, E6-KD, the intervals, k, AM-6 per seed (union-present and GT-present), the AM-17
labels with the item 3(f) caveat read from the committed MDE entry ("pending" when there is none), the
three extension points and the criteria. Analysis errors of the frozen modules are refusals (exit 2);
any other error is unexpected (exit 4).

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

from . import driver as D
from .align import METRIC_DISEASE_ONLY, AlignmentError, align_runs
from .artifact import (CODE_REPO, canonical_json, contract_sha256, serialize_a3a,
                       software_environment_block)
from .bootstrap import (ANALYSIS_ID_OFFICIAL, DATASET_MIOU_DELTA, DESCRIPTIVE_E1_E3,
                        FROZEN_TASK_MATRIX, NONINFERIORITY_E3_E6, ONE_SIDED_LOWER, BootstrapError,
                        descriptive_scalars)
from .corruption_protocol import official_corruption_grid
from .ingest import EXPECTED_ROWS_VAL, IngestError, Policy, am5_pair
from .noninferiority import PooledStages, build_e6_kd, build_non_inferiority
from .report import (EXTENSION_POINTS, LABEL_INCONCLUSIVE, committed_caveat, contrast_label,
                     gt_present_table, load_mde_entry, mde_fields)
from .robustness import INFERENTIAL_SEVERITIES, RobustnessError, align_miou_c, assemble_miou_c
from .tests import (ALPHA, CANONICAL_COMPARISON_IDS, StatsError, holm_audit, holm_family,
                    run_comparison)
from .val_artifacts import (ValArtifactError, code_provenance, load_val_artifact,
                            require_comparable, require_role)

POLICY = Policy.REHEARSAL
NAMESPACE = ANALYSIS_ID_OFFICIAL
SEEDS = ("s42", "s43", "s44")
MAPPING = {"accuracy_e1_e2": ("s42", "s43"), "accuracy_e2_e3": ("s43", "s44"),
           "accuracy_e1_e6": ("s42", "s44"), DESCRIPTIVE_E1_E3: ("s42", "s44"),
           NONINFERIORITY_E3_E6: ("s44", "s42"), "robustness_e1_e6 (structural)": ("s42", "s44")}
REHEARSED = ("accuracy_e1_e2", "accuracy_e2_e3", "accuracy_e1_e6")
NOT_REHEARSED = ("accuracy_e4_e5", "accuracy_e7_e6", "accuracy_e4_e7", "accuracy_e5_e6",
                 "robustness_e1_e6")
ROBUSTNESS = "robustness_e1_e6"
RUN_TASK_IDS = tuple(
    t.task_id for t in FROZEN_TASK_MATRIX
    if t.comparison_id in (*REHEARSED, ROBUSTNESS, DESCRIPTIVE_E1_E3, NONINFERIORITY_E3_E6))
FILES = ("rehearsal_manifest.json", "rehearsal_results.json", "rehearsal_bootstrap.npz",
         "report.md")
SUMS = "SHA256SUMS"
#: The code whose provenance the manifest records (P25).
CODE_FILES = ("src/stats/rehearsal.py", "src/stats/report.py", "src/stats/driver.py",
              "src/stats/artifact.py", "src/stats/align.py", "src/stats/bootstrap.py",
              "src/stats/ingest.py", "src/stats/noninferiority.py", "src/stats/robustness.py",
              "src/stats/tests.py", "src/stats/val_artifacts.py",
              "src/stats/corruption_protocol.py", "configs/corruption_protocol.json",
              "scripts/run_stats.py")


class RehearsalRefusal(RuntimeError):
    """A rehearsal request or input refused by name, before anything is written."""


#: The frozen modules' named analysis errors: a refusal of the inputs (exit 2). Anything else is an
#: unexpected error (exit 4; A3).
ANALYSIS_ERRORS = (AlignmentError, BootstrapError, IngestError, RobustnessError, StatsError,
                   ValArtifactError)


# --------------------------------------------------------------------------------------------------
# 1. inputs
# --------------------------------------------------------------------------------------------------
def require_out_root(out_root) -> Path:
    """Q10: the out-root must be outside the repository."""
    if out_root is None:
        raise RehearsalRefusal("rehearsal mode requires --out-root (a directory outside the "
                               "repository)")
    root = Path(out_root).resolve()
    if root.is_relative_to(CODE_REPO.resolve()):
        raise RehearsalRefusal(f"--out-root {root} is inside the repository {CODE_REPO}; rehearsal "
                               "outputs go outside it (Q10)")
    return root


def check_preconditions(path: Path, label: str) -> None:
    """P20, from the raw rows: every defined per-image value is a JSON float, float32-representable
    and in [0, 1]; clean_image_id == image_id. Refuses by name."""
    lines = (Path(path) / "per_image.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        o = json.loads(line)
        if o["clean_image_id"] != o["image_id"]:
            raise RehearsalRefusal(f"{label}: row {o['image_id']!r} has clean_image_id "
                                   f"{o['clean_image_id']!r} (P20: bare stems)")
        for key in ("all_class_miou", "disease_only_miou"):
            v = o[key]
            if v is None:
                continue
            if type(v) is not float or not 0.0 <= v <= 1.0 or float(np.float32(v)) != v:
                raise RehearsalRefusal(f"{label}: row {o['image_id']!r} {key} = {v!r} is not a "
                                       "float32-representable float in [0, 1] (P20)")


def load_seeds(paths: dict) -> dict:
    """The three E1 VAL artifacts, verified, canvas-guarded, distinct and comparable."""
    arts = {}
    for label in SEEDS:
        p = paths.get(label)
        if p is None:
            raise RehearsalRefusal(f"rehearsal mode requires --{label}")
        try:
            art = load_val_artifact(Path(p), label=label)
            require_role(art, stage="E1", model_role="student", precision="fp32", label=label)
        except ValArtifactError as e:
            raise RehearsalRefusal(str(e)) from e
        check_preconditions(Path(p), label)
        arts[label] = art
    for attr, what in ((lambda a: a.path.resolve(), "directory"), (lambda a: a.run_id, "run_id"),
                       (lambda a: a.summary["run"].get("checkpoint_sha256"), "checkpoint")):
        vals = [attr(arts[s]) for s in SEEDS]
        if len(set(vals)) != 3:
            raise RehearsalRefusal(f"the three seeds must be three runs: their {what}s are {vals}")
    try:
        for a, b in (("s42", "s43"), ("s42", "s44"), ("s43", "s44")):
            require_comparable(arts[a], arts[b])
    except ValArtifactError as e:
        raise RehearsalRefusal(str(e)) from e
    return arts


# --------------------------------------------------------------------------------------------------
# 2. the structural robustness grid (15 relabelled copies of a clean run)
# --------------------------------------------------------------------------------------------------
def relabel(run, *, stage: str, condition=("clean", None, None), suffix: str = ""):
    ctype, name, sev = condition
    ident = dataclasses.replace(run.identity, stage=stage, condition_type=ctype,
                                condition_name=name, condition_severity=sev,
                                run_id=run.identity.run_id + suffix)
    return dataclasses.replace(run, identity=ident)


def structural_cells(run, stage: str, grid) -> list:
    return [relabel(run, stage=stage, condition=("corruption", n, s), suffix=f"#{n}-s{s}")
            for n in grid.names for s in INFERENTIAL_SEVERITIES]


def structural_miou_c(run, stage: str, grid, cells=None):
    """mIoU-C of 15 identical cells, by the unmodified assemble_miou_c."""
    return assemble_miou_c(cells if cells is not None else structural_cells(run, stage, grid), grid,
                           policy=POLICY, stage=stage, clean=relabel(run, stage=stage))


def invariant_mismatches(mc, run) -> list:
    """P21: exact == of every mIoU-C element and its clean value (never bytes, never a tolerance)."""
    clean = {r.clean_image_id: float(r.disease_only_miou) for r in run.records}
    return [cid for cid, v in zip(mc.clean_image_ids, mc.values) if not v == clean[cid]]


# --------------------------------------------------------------------------------------------------
# 3. Holm checks
# --------------------------------------------------------------------------------------------------
def holm_members(pairs) -> list[dict]:
    """A Holm family over arbitrary (id, p) pairs: the frozen holm_audit decisions, the library's
    adjusted p-values and the same finalisation holm_family applies (P24: the 3-member family)."""
    audit = holm_audit(pairs, ALPHA)
    lib_rej, lib_adj, _, _ = multipletests([p for _, p in pairs], alpha=ALPHA, method="holm",
                                           is_sorted=False, returnsorted=False)
    out = []
    for m, adj, lrej in zip(audit, lib_adj, lib_rej):
        strict = m.reject and float(adj) != ALPHA
        out.append({"comparison_id": m.comparison_id, "p_raw": m.p_raw, "p_adjusted": float(adj),
                    "sorted_rank": m.sorted_rank, "step_denominator": m.step_denominator,
                    "step_threshold": m.step_threshold, "reject": bool(strict),
                    "library_reject": bool(lrej), "boundary_note": m.boundary_note})
    return out


def hand_holm(pvals, alpha: float = ALPHA) -> list[bool]:
    """An independent strict Holm step-down: sort ascending; reject while p < alpha / (m - rank)."""
    m = len(pvals)
    reject = [False] * m
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvals[j])):
        if not pvals[i] < alpha / (m - rank):
            break
        reject[i] = True
    return reject


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def p0_of(task) -> float:
    rep = task.replicates
    return (float(np.count_nonzero(rep < task.observed))
            + 0.5 * float(np.count_nonzero(rep == task.observed))) / rep.size


# --------------------------------------------------------------------------------------------------
# 4. the rehearsal
# --------------------------------------------------------------------------------------------------
@dataclass
class Rehearsal:
    arts: dict
    k_val: int
    results: dict            # comparison_id -> ComparisonResult (rehearsed and placeholders)
    holm8: object
    holm3: list
    structural: object
    descriptive: object
    pooled: dict
    tasks: tuple
    ni: object
    e6: object
    miou_c: dict
    criteria: list
    investigate: list


def compute(arts: dict, *, B: int, jobs: int = 1, expect_k=None) -> Rehearsal:
    runs = {s: arts[s].run for s in SEEDS}
    try:
        for a, b in (("s42", "s43"), ("s42", "s44")):
            am5_pair(runs[a], runs[b])
    except ANALYSIS_ERRORS as e:
        raise RehearsalRefusal(f"AM-5: {e}") from e
    k_val = runs["s42"].n_excluded_am5
    if expect_k is not None and k_val != expect_k:
        raise RehearsalRefusal(f"K_val is {k_val}, --expect-k-val says {expect_k}")
    grid = official_corruption_grid()
    try:
        results, paired, pooled = {}, {}, {}
        for cid in CANONICAL_COMPARISON_IDS[:7]:
            b_stage, c_stage, _ = D.COMPARISON_TABLE[cid]
            b, c = MAPPING.get(cid, ("s42", "s42"))
            pv = align_runs(runs[b], runs[c], policy=POLICY, metric=METRIC_DISEASE_ONLY)
            results[cid] = run_comparison(cid, pv, baseline_stage=b_stage, candidate_stage=c_stage)
            if cid in REHEARSED:
                paired[cid] = pv
                pooled[cid] = PooledStages.from_runs(runs[b], runs[c])
        rb, rc, _ = D.COMPARISON_TABLE[ROBUSTNESS]
        mc = {"s42": structural_miou_c(runs["s42"], rb, grid),
              "s44": structural_miou_c(runs["s44"], rc, grid)}
        results[ROBUSTNESS] = run_comparison(ROBUSTNESS, align_miou_c(mc["s42"], mc["s42"],
                                                                      policy=POLICY),
                                             baseline_stage=rb, candidate_stage=rc)
        pv_rob = align_miou_c(mc["s42"], mc["s44"], policy=POLICY)
        structural = run_comparison(ROBUSTNESS, pv_rob, baseline_stage=rb, candidate_stage=rc)
        paired[ROBUSTNESS] = pv_rob
        holm8 = holm_family([results[c] for c in CANONICAL_COMPARISON_IDS])
        holm3 = holm_members([(c, float(results[c].primary_p)) for c in REHEARSED])
        pv_d = align_runs(runs["s42"], runs["s44"], policy=POLICY, metric=METRIC_DISEASE_ONLY)
        paired[DESCRIPTIVE_E1_E3] = pv_d
        desc = descriptive_scalars(pv_d.delta, metric=pv_d.metric, policy=pv_d.policy.value,
                                   alignment_status=pv_d.alignment_status)
        pooled[NONINFERIORITY_E3_E6] = PooledStages.from_runs(runs["s44"], runs["s42"])
    except ANALYSIS_ERRORS as e:
        raise RehearsalRefusal(f"the seeds cannot be analysed: {type(e).__name__}: {e}") from e

    task_obs = D.Observed(policy=POLICY, results=tuple(
        structural if c == ROBUSTNESS else results[c] for c in CANONICAL_COMPARISON_IDS),
        holm=holm8, descriptive=desc, paired=paired, miou_c=mc, pooled=pooled,
        am5_excluded_count=k_val, excluded_clean_ids=runs["s42"].am5.excluded_clean_ids)
    jobs_list = [job for job in D.task_jobs(task_obs, NAMESPACE, B, only=RUN_TASK_IDS)]
    tasks = D.run_job_list(jobs_list, jobs=jobs)
    b_miou, c_miou = pooled[NONINFERIORITY_E3_E6].stage_mious()
    ni_task = next(t for t in tasks if t.task.comparison_id == NONINFERIORITY_E3_E6)
    ni = build_non_inferiority(b_miou, c_miou, ni_task.bca.bounds[0])
    e6 = build_e6_kd(b_miou, c_miou)
    reh = Rehearsal(arts, k_val, results, holm8, holm3, structural, desc, pooled, tasks, ni, e6,
                    mc, [], [])
    reh.criteria = criteria(reh, runs)
    reh.investigate = ([f"8-entry Holm rejects {m.comparison_id}" for m in holm8.members
                        if m.reject]
                       + [f"3-member Holm rejects {m['comparison_id']}" for m in holm3
                          if m["reject"]])
    return reh


def criteria(reh: Rehearsal, runs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for t in reh.tasks:
        lo, est = t.bca.bounds[0], t.observed
        ok = lo <= est if t.task.interval_type == ONE_SIDED_LOWER else lo <= est <= t.bca.bounds[1]
        out.append((f"containment {t.task_id}", ok,
                    "" if ok else f"bounds {t.bca.bounds}, estimate {est!r}, z0 {t.bca.z0!r}, "
                                  f"p0 {p0_of(t)!r}"))
    ps = ([(f"{c} p", reh.results[c].wilcoxon.p_value) for c in REHEARSED]
          + [(f"{m.comparison_id} p_adjusted (8)", m.p_adjusted) for m in reh.holm8.members
             if m.comparison_id in REHEARSED]
          + [(f"{m['comparison_id']} p_adjusted (3)", m["p_adjusted"]) for m in reh.holm3])
    bad_p = [(n, p) for n, p in ps if not (type(p) is float and 0.0 <= p <= 1.0)]
    out.append(("p in [0, 1]: the rehearsed tests' p-values and their Holm-adjusted p", not bad_p,
                str(bad_p)))
    p8 = [m.p_raw for m in reh.holm8.members]
    p3 = [m["p_raw"] for m in reh.holm3]
    for label, raw, adj in (("8 entries", p8, [m.p_adjusted for m in reh.holm8.members]),
                            ("3 rehearsed tests", p3, [m["p_adjusted"] for m in reh.holm3])):
        lib = multipletests(raw, alpha=ALPHA, method="holm")[1]
        dev = max(abs(float(a) - float(b)) for a, b in zip(adj, lib))
        out.append((f"Holm ({label}) adjusted p equals statsmodels multipletests(method='holm') to "
                    "1e-12", dev <= 1e-12, f"max |difference| {dev!r}"))
    out.append(("Holm (8 entries) equals a hand-rolled strict Holm",
                [m.reject for m in reh.holm8.members] == hand_holm(p8), str(p8)))
    out.append(("Holm (3 rehearsed tests) equals a hand-rolled strict Holm",
                [m["reject"] for m in reh.holm3] == hand_holm(p3), str(p3)))
    for cid in REHEARSED:
        r = reh.results[cid]
        t = r.t_test
        same_sign = t.statistic is None or _sign(t.statistic) == _sign(t.mean_difference)
        out.append((f"t-test sign {cid} (sign(0) = 0; null statistic skipped)", same_sign,
                    f"t {t.statistic!r}, mean difference {t.mean_difference!r}"))
        bitwise = (t.mean_difference == r.shifts.mean_delta == r.cohens_dz.mean
                   and np.float64(t.mean_difference).tobytes()
                   == np.float64(r.shifts.mean_delta).tobytes()
                   == np.float64(r.cohens_dz.mean).tobytes())
        out.append((f"t_test.mean_difference == shifts.mean_delta == cohens_dz.mean bitwise {cid}",
                    bitwise, f"{t.mean_difference!r} {r.shifts.mean_delta!r} {r.cohens_dz.mean!r}"))
    n_pi, n_ds = EXPECTED_ROWS_VAL - reh.k_val, EXPECTED_ROWS_VAL
    bad = [(t.task_id, t.jackknife.size) for t in reh.tasks
           if t.jackknife.size != (n_ds if t.task.statistic_name == DATASET_MIOU_DELTA else n_pi)]
    out.append((f"counts: per-image {n_pi} (846 - K_val), dataset-level {n_ds}", not bad, str(bad)))
    for seed in ("s42", "s44"):
        miss = invariant_mismatches(reh.miou_c[seed], runs[seed])
        out.append((f"15-cell invariant {seed}: mIoU-C == clean values exactly", not miss,
                    f"{len(miss)} image(s), e.g. {miss[:3]}"))
    a, s = serialize_a3a(reh.results["accuracy_e1_e6"]), serialize_a3a(reh.structural)
    diff = sorted(k for k in a if k not in ("comparison_id", "metric") and a[k] != s[k])
    out.append(("structural robustness equals accuracy_e1_e6 except comparison_id and metric",
                not diff, str(diff)))
    return out


# --------------------------------------------------------------------------------------------------
# 5. outputs
# --------------------------------------------------------------------------------------------------
def _cpu_features() -> list[str]:
    for mod in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            feats = __import__(mod, fromlist=["__cpu_features__"]).__cpu_features__
            return sorted(k for k, v in feats.items() if v)
        except (ImportError, AttributeError):
            continue
    return []


def _task_entry(t) -> dict:
    b = t.bca
    out = {"task_id": t.task_id, "interval_type": b.interval_type,
           "bootstrap_replicates": int(t.replicates.size), "jackknife_count": int(t.jackknife.size),
           "seed_sha256": t.seed.seed_sha256, "observed": float(t.observed), "z0": float(b.z0),
           "p0": p0_of(t), "acceleration": None if not b.acceleration_defined
           else float(b.acceleration), "interval_method": b.interval_method,
           "warnings": list(b.warnings), "lower_bound": float(b.bounds[0])}
    if len(b.bounds) == 2:
        out["upper_bound"] = float(b.bounds[1])
    return out


def _labels(reh: Rehearsal, mde: dict | None) -> dict:
    """AM-17 item 2 labels of the rehearsed contrasts; item 3(f) from the committed MDE entry only
    (lane item 7): a non-rejection reads "inconclusive ..." with MDE_W when its power caveat holds,
    and carries "power caveat pending" when there is no committed, valid entry."""
    caveat, mde_w = committed_caveat(mde)
    obs = {t.task_id: float(t.observed) for t in reh.tasks}
    reject = {m.comparison_id: m.reject for m in reh.holm8.members}
    out = {}
    for cid in REHEARSED:
        label = contrast_label(reject[cid], obs[f"{cid}__{DATASET_MIOU_DELTA}"],
                               power_caveat=caveat is True)
        if label == LABEL_INCONCLUSIVE:
            label += f" (MDE_W = {mde_w})"
        elif not reject[cid] and caveat == "pending":
            label += "; power caveat pending (no committed MDE entry)"
        out[cid] = label
    return out


def build_documents(reh: Rehearsal, *, B: int, jobs: int, created: str, mde: dict | None) -> dict:
    import src.stats as _pkg
    arts = reh.arts
    contract_hex, _ = contract_sha256()
    manifest = {
        "mode": "rehearsal", "artifact_status": "nonofficial", "registered_as_result": False,
        "statistics_namespace": NAMESPACE, "created_at_utc": created,
        "inputs": {s: dict(arts[s].describe(), path=str(arts[s].path.resolve())) for s in SEEDS},
        "mapping": {k: list(v) for k, v in MAPPING.items()}, "not_rehearsed": list(NOT_REHEARSED),
        "B": B, "jobs": jobs, "K_val": reh.k_val, "n_val": EXPECTED_ROWS_VAL,
        "code_provenance": code_provenance(CODE_FILES),
        "software_environment": software_environment_block(),
        "python_executable": sys.executable, "cpu_features": _cpu_features(),
        "statistical_contract_sha256": contract_hex,
        "src_stats_path": str(Path(_pkg.__file__).resolve().parent),
        "mde_entry": mde,
    }
    results = {
        "artifact_status": "nonofficial", "statistics_namespace": NAMESPACE,
        "K_val": reh.k_val, "n_per_image": EXPECTED_ROWS_VAL - reh.k_val,
        "n_dataset": EXPECTED_ROWS_VAL, "not_rehearsed": list(NOT_REHEARSED),
        "comparisons": [dict(serialize_a3a(reh.results[c]),
                             role="placeholder" if c in NOT_REHEARSED else "rehearsed")
                        for c in CANONICAL_COMPARISON_IDS],
        "holm_8": serialize_a3a(reh.holm8), "holm_3": reh.holm3,
        "investigate": {"fired": bool(reh.investigate), "reasons": reh.investigate},
        "labels": _labels(reh, mde),
        "structural_robustness": serialize_a3a(reh.structural),
        "descriptive_e1_e3": {k: getattr(reh.descriptive, k) for k in (
            "baseline_stage", "candidate_stage", "n_paired", "mean_delta", "mean_delta_pp",
            "median_delta", "hodges_lehmann_shift")},
        "noninferiority_e3_e6": {"baseline_dataset_miou": reh.ni.baseline_dataset_miou,
                                 "candidate_dataset_miou": reh.ni.candidate_dataset_miou,
                                 "observed_delta": reh.ni.observed_delta,
                                 "lower_bound": reh.ni.lower_bound, "passed": reh.ni.passed,
                                 "sensitivity": [{"margin": s.margin, "passed": s.passed}
                                                 for s in reh.ni.sensitivity]},
        "e6_kd_trigger": {"observed_drop": reh.e6.observed_drop, "triggered": reh.e6.triggered},
        "power_caveat": committed_caveat(mde)[0],
        "gt_present_am6": gt_present_table({s: arts[s].run.stats for s in SEEDS}, key="seed"),
        "extension_points": [{"name": n, "content": d, "populated": False}
                             for n, d in EXTENSION_POINTS],
        "tasks": [_task_entry(t) for t in reh.tasks],
        "tasks_not_run": [t.task_id for t in FROZEN_TASK_MATRIX if t.task_id not in RUN_TASK_IDS],
        "criteria": [{"name": n, "passed": ok, "detail": d} for n, ok, d in reh.criteria],
        "criteria_passed": all(ok for _, ok, _ in reh.criteria),
    }
    return {"rehearsal_manifest.json": manifest, "rehearsal_results.json": results}


def _g(v) -> str:
    return "null" if v is None else f"{v:.6g}"


def render_report(docs: dict) -> str:
    m, r = docs["rehearsal_manifest.json"], docs["rehearsal_results.json"]
    banner = ("> **REHEARSAL -- NONOFFICIAL.** AM-17 item 9 dress rehearsal on VAL (E1 seeds 42/43/44"
              " relabelled per the mapping). Not a thesis result; no statistics artifact.")
    lines = ["# Statistics dress rehearsal (AM-17 item 9)", "", banner, "",
             f"Created {m['created_at_utc']}; namespace {m['statistics_namespace']}; B = {m['B']}; "
             f"K_val = {m['K_val']} (per-image n = {r['n_per_image']}, dataset-level n = "
             f"{r['n_dataset']}).", "",
             "| seed | run_id | checkpoint | directory |", "|---|---|---|---|"]
    for s, d in m["inputs"].items():
        lines.append(f"| {s} | {d['run_id']} | {str(d['checkpoint_sha256'])[:16]} | {d['path']} |")
    lines += ["", "## Family (8 entries; placeholders are self-pairs with p = 1, not rehearsed)", "",
              banner, "",
              "| comparison | role | W | z | p (raw) | p (Holm) | reject | r_rb | HL shift | dz | "
              "mean delta [BCa 95 %] | t (p) | label |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    holm = {x["comparison_id"]: x for x in r["holm_8"]["members"]}
    tasks = {t["task_id"]: t for t in r["tasks"]}
    for c in r["comparisons"]:
        cid, h, t, w = c["comparison_id"], holm[c["comparison_id"]], c["t_test"], c["wilcoxon"]
        label = r["labels"].get(cid, "--")
        md = tasks.get(f"{cid}__mean_delta") if c["role"] == "rehearsed" else None
        ci = f" [{_g(md['lower_bound'])}, {_g(md.get('upper_bound'))}]" if md else " (not run)"
        lines.append(f"| {cid} | {c['role']} | {_g(w['statistic'])} | {_g(w['zstatistic'])} | "
                     f"{h['p_raw']:.6g} | {h['p_adjusted']:.6g} | {h['reject']} | "
                     f"{_g(c['rank_biserial']['value'])} | {c['hodges_lehmann_shift']:.6g} | "
                     f"{_g(c['cohens_dz']['value'])} | {c['shifts']['mean_delta']:.6g}{ci} | "
                     f"{_g(t['statistic'])} ({_g(t['p_value'])}) | {label} |")
    lines += ["", "## The three rehearsed tests: raw p-values and their 3-member Holm (P24)", "",
              banner, "", "| comparison | p (raw) | p (Holm, 3) | reject |", "|---|---|---|---|"]
    for x in r["holm_3"]:
        lines.append(f"| {x['comparison_id']} | {x['p_raw']:.6g} | {x['p_adjusted']:.6g} | "
                     f"{x['reject']} |")
    inv = r["investigate"]
    lines += ["", "INVESTIGATE (AM-17 item 9): " + ("**FIRED** -- " + "; ".join(inv["reasons"])
                                                    + ". A rejection at seed noise is a mandatory "
                                                    "investigation before the first KD run."
                                                    if inv["fired"] else "not fired."),
              "", "## Structural robustness (15 identical cells per seed)", "", banner, "",
              f"mean delta {r['structural_robustness']['shifts']['mean_delta']!r}; p "
              f"{r['structural_robustness']['wilcoxon']['p_value']!r}.", "",
              "## Descriptive E1 -> E3 (s42 -> s44) and non-inferiority E3 -> E6 (s44 -> s42)", "",
              banner, ""]
    d, ni = r["descriptive_e1_e3"], r["noninferiority_e3_e6"]
    lines.append(f"Descriptive mean delta {d['mean_delta']!r}, median {d['median_delta']!r}, "
                 f"HL {d['hodges_lehmann_shift']!r} (n = {d['n_paired']}).")
    lines.append(f"Non-inferiority: observed delta {ni['observed_delta']!r}, one-sided lower bound "
                 f"{ni['lower_bound']!r}, passed {ni['passed']}; sensitivity "
                 + ", ".join(f"{s['margin']}: {s['passed']}" for s in ni["sensitivity"]) + ".")
    lines.append(f"E6-KD observed drop {r['e6_kd_trigger']['observed_drop']!r}, triggered "
                 f"{r['e6_kd_trigger']['triggered']} (descriptive; launches nothing).")
    lines += ["", "## Bootstrap intervals (19 tasks; z0 and p0 per interval)", "", banner, "",
              "| task | estimate | lower | upper | z0 | p0 | method |", "|---|---|---|---|---|---|---|"]
    for t in r["tasks"]:
        lines.append(f"| {t['task_id']} | {t['observed']:.6g} | {t['lower_bound']:.6g} | "
                     f"{t.get('upper_bound', float('nan')):.6g} | {t['z0']:.4g} | {t['p0']:.4g} | "
                     f"{t['interval_method']} |")
    lines += ["", "## AM-6: dataset-level mIoU under the GT-present rule, per seed (descriptive)", "",
              banner, "", "| seed | union-present all-class | GT-present all-class | union-present "
              "disease-only | GT-present disease-only |", "|---|---|---|---|---|"]
    for g in r["gt_present_am6"]:
        lines.append(f"| {g['seed']} | {_g(g['union_present_all_class'])} | "
                     f"{_g(g['gt_present_all_class'])} | {_g(g['union_present_disease_only'])} | "
                     f"{_g(g['gt_present_disease_only'])} |")
    lines += ["", "## Extension points (not populated by the rehearsal)", "", banner, ""]
    lines += [f"- {e['name']}: {e['content']}" for e in r["extension_points"]]
    lines += ["", "## Criteria (P23)", "", banner, ""]
    for c in r["criteria"]:
        lines.append(f"- [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}"
                     + ("" if c["passed"] else f" -- {c['detail']}"))
    mde = m["mde_entry"]
    lines += ["", "## MDE entry (AM-17 item 3(f))", "",
              (f"{mde['path']}: sha256 {mde['sha256']}, committed "
               f"{bool(mde['tracked'] and mde['clean_at_head'])}, last commit {mde['last_commit']}; "
               f"mde_w {mde['mde_w']}, tau_p {mde['tau_p']}, power_caveat {mde['power_caveat']}; "
               f"problems {mde['problems']}" if mde else "none given")
              + f". Power caveat: {r['power_caveat']}.", ""]
    return "\n".join(lines)


def _npz_bytes(tasks) -> bytes:
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"{t.task_id}__{k}": v for t in tasks
                                for k, v in (("bootstrap", t.replicates),
                                             ("jackknife", t.jackknife))})
    return buf.getvalue()


def write_outputs(out_root: Path, stamp: str, payloads: dict) -> Path:
    """Atomic: a temporary sibling holding the files and SHA256SUMS, verified, then renamed to
    <out-root>/<stamp>/."""
    out_root.mkdir(parents=True, exist_ok=True)
    final = out_root / stamp
    if final.exists():
        raise RehearsalRefusal(f"refusing to overwrite {final}")
    tmp = out_root / f".{stamp}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir()
    try:
        for name, data in payloads.items():
            (tmp / name).write_bytes(data)
        sums = "".join(f"{hashlib.sha256(payloads[n]).hexdigest()}  {n}\n" for n in sorted(payloads))
        (tmp / SUMS).write_text(sums, encoding="utf-8", newline="\n")
        for line in (tmp / SUMS).read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if hashlib.sha256((tmp / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"{name}: SHA256SUMS mismatch after writing")
        tmp.rename(final)
        return final
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


@dataclass(frozen=True)
class RehearsalRun:
    out_dir: Path
    criteria_passed: bool
    investigate: list
    k_val: int
    seconds: float
    report: str


def run_rehearsal(*, s42, s43, s44, out_root, B: int, jobs: int = 1, expect_k=None,
                  mde_entry=None, log=print) -> RehearsalRun:
    t0 = time.time()
    root = require_out_root(out_root)
    mde = None
    if mde_entry is not None:
        doc, status, problems = load_mde_entry(mde_entry)
        mde = dict(status, **mde_fields(doc), problems=list(problems))
    arts = load_seeds({"s42": s42, "s43": s43, "s44": s44})
    created = datetime.now(timezone.utc)
    reh = compute(arts, B=B, jobs=jobs, expect_k=expect_k)
    docs = build_documents(reh, B=B, jobs=jobs, created=created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                           mde=mde)
    report = render_report(docs)
    payloads = {name: canonical_json(doc).encode("utf-8") for name, doc in docs.items()}
    payloads["rehearsal_bootstrap.npz"] = _npz_bytes(reh.tasks)
    payloads["report.md"] = report.encode("utf-8")
    out = write_outputs(root, created.strftime("%Y%m%dT%H%M%SZ"), payloads)
    passed = docs["rehearsal_results.json"]["criteria_passed"]
    return RehearsalRun(out, passed, reh.investigate, reh.k_val, time.time() - t0, report)


__all__ = ["POLICY", "NAMESPACE", "SEEDS", "MAPPING", "REHEARSED", "NOT_REHEARSED", "RUN_TASK_IDS",
           "FILES", "SUMS", "RehearsalRefusal", "ANALYSIS_ERRORS", "require_out_root",
           "check_preconditions",
           "load_seeds", "relabel", "structural_cells", "structural_miou_c",
           "invariant_mismatches", "holm_members", "hand_holm", "p0_of", "Rehearsal", "compute",
           "criteria", "build_documents", "render_report", "write_outputs", "RehearsalRun",
           "run_rehearsal"]
