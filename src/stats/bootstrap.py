"""Deterministic paired bootstrap + BCa (A3b).

Implements STATISTICAL_ANALYSIS_CONTRACT.md sections 8.1-8.7 exactly: the amended seed protocol
(root_seed rendered into the canonical bytes), row-wise draws, one independent Generator per
`(analysis_id, comparison_id, statistic_name)` task, `z0` with exact-tie half weighting, the
leave-one-image-out jackknife acceleration, explicit linear quantiles, adjusted-probability
saturation, and the deterministic first-trigger P1-P8 fallback ladder.

Nothing here touches the filesystem, Git, or a dataset, and importing the module has no side
effects. The pooled dataset-mIoU estimand lives in `noninferiority.py`; serialization lives in
`artifact.py`.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.special import ndtr, ndtri

from .tests import CANONICAL_COMPARISON_IDS, engineering_shifts, hodges_lehmann

# --------------------------------------------------------------------------------------------------
# frozen namespaces (contract section 8.7.1)
# --------------------------------------------------------------------------------------------------
SCHEMA_VERSION = "plantseg-stats/1.0.0"
STATISTICAL_PROTOCOL = "plantseg-stats-protocol/1.0.0"
FAMILY_ID = "plantseg_eight_test_holm_v1"

ROOT_SEED = 42
PRODUCTION_B = 10_000
OFFICIAL_JACKKNIFE_N = 1_561

ANALYSIS_ID_OFFICIAL = "plantseg_primary_analysis_v1"
ANALYSIS_ID_SMOKE = "plantseg_stats_smoke_v1"
ANALYSIS_IDS = (ANALYSIS_ID_OFFICIAL, ANALYSIS_ID_SMOKE)

DESCRIPTIVE_E1_E3 = "descriptive_e1_e3"
NONINFERIORITY_E3_E6 = "noninferiority_e3_e6"
NON_FAMILY_COMPARISON_IDS = (DESCRIPTIVE_E1_E3, NONINFERIORITY_E3_E6)
ALL_COMPARISON_IDS = tuple(CANONICAL_COMPARISON_IDS) + NON_FAMILY_COMPARISON_IDS

MEAN_DELTA = "mean_delta"
MEDIAN_DELTA = "median_delta"
HL_SHIFT = "hodges_lehmann_shift"
DATASET_MIOU_DELTA = "dataset_miou_delta"
STATISTIC_NAMES = (MEAN_DELTA, MEDIAN_DELTA, HL_SHIFT, DATASET_MIOU_DELTA)
SCALAR_STATISTICS = (MEAN_DELTA, MEDIAN_DELTA, HL_SHIFT)

#: Explicitly rejected aliases (contract section 8.7.1).
REJECTED_STATISTIC_ALIASES = ("mean", "median", "hl", "hodges_lehmann", "pooled_miou",
                              "miou_delta", "miou_c_delta", "dataset_robustness_delta")

TWO_SIDED = "two_sided"
ONE_SIDED_LOWER = "one_sided_lower"
INTERVAL_TYPES = (TWO_SIDED, ONE_SIDED_LOWER)

METHOD_BCA = "bca"
METHOD_FALLBACK = "percentile_fallback_degenerate_acceleration"
INTERVAL_METHODS = (METHOD_BCA, METHOD_FALLBACK)

STATUS_OK = "ok"
STATUS_OK_WARN = "ok_with_warnings"
TASK_STATUSES = (STATUS_OK, STATUS_OK_WARN)

OBSERVED_SOURCE_A3A = "a3a_comparison_result"
OBSERVED_SOURCE_PRIMITIVES = "a3a_primitives_descriptive"
OBSERVED_SOURCE_POOLED = "a3b_pooled_reaccumulation"
OBSERVED_SOURCES = (OBSERVED_SOURCE_A3A, OBSERVED_SOURCE_PRIMITIVES, OBSERVED_SOURCE_POOLED)

CONFIDENCE_LEVEL = 0.95
Q_TWO_SIDED = (0.025, 0.975)
Q_ONE_SIDED_LOWER = (0.05,)
P0_CLIP_LO = 1e-12
P0_CLIP_HI = 1.0 - 1e-12

#: Canonical warning vocabulary and serialization order (contract section 12.3.5).
W_CLIP_LOWER = "bias_correction_probability_clipped_lower"
W_CLIP_UPPER = "bias_correction_probability_clipped_upper"
W_SAT_LOWER = "adjusted_probability_saturated_lower"
W_SAT_UPPER = "adjusted_probability_saturated_upper"
FALLBACK_WARNINGS = (
    "fallback_p1_insufficient_jackknife",
    "fallback_p2_nonfinite_jackknife",
    "fallback_p3_zero_centered_jackknife_sum_squares",
    "fallback_p4_invalid_acceleration_denominator",
    "fallback_p5_nonfinite_acceleration",
    "fallback_p6_invalid_bca_transform_denominator",
    "fallback_p7_nonfinite_adjusted_probability",
    "fallback_p8_nonincreasing_adjusted_probabilities",
)
TASK_WARNINGS = (W_CLIP_LOWER, W_CLIP_UPPER, W_SAT_LOWER, W_SAT_UPPER) + FALLBACK_WARNINGS
_WARNING_RANK = {w: i for i, w in enumerate(TASK_WARNINGS)}


class BootstrapError(RuntimeError):
    """Integrity or mathematical failure. Never a statistical-result status."""


# --------------------------------------------------------------------------------------------------
# 1. the frozen 35-task matrix (contract section 8.7.2)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    comparison_id: str
    statistic_name: str
    interval_type: str
    observed_source: str

    @property
    def task_id(self) -> str:
        return f"{self.comparison_id}__{self.statistic_name}"


def _build_matrix() -> tuple[TaskSpec, ...]:
    clean = CANONICAL_COMPARISON_IDS[:7]
    robustness = CANONICAL_COMPARISON_IDS[7]
    out: list[TaskSpec] = []
    for cid in clean:
        for stat in SCALAR_STATISTICS:
            out.append(TaskSpec(cid, stat, TWO_SIDED, OBSERVED_SOURCE_A3A))
        out.append(TaskSpec(cid, DATASET_MIOU_DELTA, TWO_SIDED, OBSERVED_SOURCE_POOLED))
    for stat in SCALAR_STATISTICS:
        out.append(TaskSpec(robustness, stat, TWO_SIDED, OBSERVED_SOURCE_A3A))
    for stat in SCALAR_STATISTICS:
        out.append(TaskSpec(DESCRIPTIVE_E1_E3, stat, TWO_SIDED, OBSERVED_SOURCE_PRIMITIVES))
    out.append(TaskSpec(NONINFERIORITY_E3_E6, DATASET_MIOU_DELTA, ONE_SIDED_LOWER,
                        OBSERVED_SOURCE_POOLED))
    return tuple(out)


FROZEN_TASK_MATRIX: tuple[TaskSpec, ...] = _build_matrix()
FROZEN_TASK_IDS: tuple[str, ...] = tuple(t.task_id for t in FROZEN_TASK_MATRIX)
TASK_BY_ID = {t.task_id: t for t in FROZEN_TASK_MATRIX}

if len(FROZEN_TASK_MATRIX) != 35:                                    # pragma: no cover - guard
    raise BootstrapError("frozen task matrix must contain exactly 35 tasks")


def expected_comparison_ids() -> tuple[str, ...]:
    """The eight canonical IDs, imported (never retyped) from `src.stats.tests`."""
    return tuple(CANONICAL_COMPARISON_IDS)


# --------------------------------------------------------------------------------------------------
# 2. seed derivation (contract section 8.1, as amended by A3b-1)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SeedMaterial:
    root_seed: int
    canonical_bytes: bytes
    seed_input_display: str
    seed_input_hex: str
    seed_sha256: str
    entropy_bytes: bytes
    entropy_int: int
    bit_generator: str
    numpy_version: str


def _validate_identifier(value: str, allowed: Sequence[str], what: str) -> None:
    if not isinstance(value, str):
        raise BootstrapError(f"{what} must be a string, got {type(value).__name__}")
    if value != value.strip():
        raise BootstrapError(f"{what} has leading/trailing whitespace: {value!r}")
    if "\x00" in value:
        raise BootstrapError(f"{what} contains an embedded NUL: {value!r}")
    if value not in allowed:
        raise BootstrapError(f"unrecognised {what}: {value!r}")


def canonical_seed_bytes(analysis_id: str, comparison_id: str, statistic_name: str,
                         root_seed: int = ROOT_SEED) -> bytes:
    """Exact frozen byte layout. NOTE: no trailing NUL after `statistic_name`."""
    _validate_identifier(analysis_id, ANALYSIS_IDS, "analysis_id")
    _validate_identifier(comparison_id, ALL_COMPARISON_IDS, "comparison_id")
    _validate_identifier(statistic_name, STATISTIC_NAMES, "statistic_name")
    return (b"plantseg-stats/1.0.0\x00"
            + b"root_seed=" + str(root_seed).encode("ascii") + b"\x00"
            + analysis_id.encode("utf-8") + b"\x00"
            + comparison_id.encode("utf-8") + b"\x00"
            + statistic_name.encode("utf-8"))


def derive_seed(analysis_id: str, comparison_id: str, statistic_name: str,
                root_seed: int = ROOT_SEED) -> SeedMaterial:
    raw = canonical_seed_bytes(analysis_id, comparison_id, statistic_name, root_seed)
    digest = hashlib.sha256(raw).digest()
    entropy = digest[:16]
    display = (f"plantseg-stats/1.0.0\\0root_seed={root_seed}\\0{analysis_id}"
               f"\\0{comparison_id}\\0{statistic_name}")
    return SeedMaterial(
        root_seed=root_seed, canonical_bytes=raw, seed_input_display=display,
        seed_input_hex=raw.hex(), seed_sha256=digest.hex(), entropy_bytes=entropy,
        entropy_int=int.from_bytes(entropy, "big", signed=False),
        bit_generator="PCG64", numpy_version=np.__version__)


def make_generator(seed: SeedMaterial) -> np.random.Generator:
    """One local Generator per task. The 128-bit integer is the sole SeedSequence entropy."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed.entropy_int)))


def reconstruct_seed_material(analysis_id: str, comparison_id: str, statistic_name: str,
                              seed_input_hex: str, root_seed: int = ROOT_SEED) -> SeedMaterial:
    """Rebuild the canonical bytes and require exact equality with the recorded hex."""
    mat = derive_seed(analysis_id, comparison_id, statistic_name, root_seed)
    if bytes.fromhex(seed_input_hex) != mat.canonical_bytes:
        raise BootstrapError(
            f"seed byte reconstruction mismatch for {comparison_id}__{statistic_name}")
    return mat


# --------------------------------------------------------------------------------------------------
# 3. draws
# --------------------------------------------------------------------------------------------------
def draw_replicate_indices(generator: np.random.Generator, n_images: int) -> np.ndarray:
    """Exactly one row-wise call per replicate. No `[B, n]` block is ever materialised."""
    return generator.integers(low=0, high=n_images, size=n_images, dtype=np.int64,
                              endpoint=False)


# --------------------------------------------------------------------------------------------------
# 4. BCa (contract sections 8.2-8.6, 8.7.5-8.7.7)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BcaOutcome:
    z0: float
    acceleration: float | None
    acceleration_defined: bool
    interval_type: str
    interval_method: str
    confidence_level: float
    bounds: tuple[float, ...]
    warnings: tuple[str, ...]
    status: str


def _quantiles(sample: np.ndarray, qs: Sequence[float]) -> tuple[float, ...]:
    """Explicit linear interpolation -- never an evolving library default."""
    return tuple(float(np.quantile(sample, q, method="linear")) for q in qs)


def _sorted_warnings(ws: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for w in ws:
        if w not in _WARNING_RANK:
            raise BootstrapError(f"warning outside the frozen vocabulary: {w!r}")
        if w not in seen:
            seen.append(w)
    return tuple(sorted(seen, key=lambda w: _WARNING_RANK[w]))


def bca_interval(replicates: np.ndarray, jackknife: np.ndarray, observed: float,
                 interval_type: str) -> BcaOutcome:
    """z0 -> first-trigger P1-P8 ladder -> adjusted probabilities -> linear quantiles."""
    if interval_type not in INTERVAL_TYPES:
        raise BootstrapError(f"unknown interval_type {interval_type!r}")
    rep = np.asarray(replicates, dtype=np.float64)
    jack = np.asarray(jackknife, dtype=np.float64)
    if not np.isfinite(observed):
        raise BootstrapError("observed statistic is non-finite")
    if not np.all(np.isfinite(rep)):
        raise BootstrapError("non-finite bootstrap replicate")

    qs = Q_TWO_SIDED if interval_type == TWO_SIDED else Q_ONE_SIDED_LOWER
    warns: list[str] = []

    # ---- 8.2 bias correction, exact-tie half weighting -------------------------------------
    B = rep.size
    proportion = (float(np.count_nonzero(rep < observed))
                  + 0.5 * float(np.count_nonzero(rep == observed))) / B
    p0 = proportion
    if p0 < P0_CLIP_LO:
        p0 = P0_CLIP_LO
        warns.append(W_CLIP_LOWER)
    elif p0 > P0_CLIP_HI:
        p0 = P0_CLIP_HI
        warns.append(W_CLIP_UPPER)
    z0 = float(ndtri(p0))

    def fallback(pred_index: int, accel: float | None, defined: bool) -> BcaOutcome:
        ws = _sorted_warnings(warns + [FALLBACK_WARNINGS[pred_index - 1]])
        return BcaOutcome(z0=z0, acceleration=accel, acceleration_defined=defined,
                          interval_type=interval_type, interval_method=METHOD_FALLBACK,
                          confidence_level=CONFIDENCE_LEVEL, bounds=_quantiles(rep, qs),
                          warnings=ws, status=STATUS_OK_WARN if ws else STATUS_OK)

    # ---- 8.7.7 first-trigger ladder; each predicate short-circuits --------------------------
    if jack.size < 2:                                                            # P1
        return fallback(1, None, False)
    if not np.all(np.isfinite(jack)):                                            # P2
        return fallback(2, None, False)
    centered = float(np.mean(jack)) - jack
    ss = float(np.sum(centered ** 2))
    if ss == 0.0:                                                                # P3
        return fallback(3, None, False)
    den_a = 6.0 * (ss ** 1.5)
    if den_a == 0.0 or not np.isfinite(den_a):                                   # P4
        return fallback(4, None, False)
    accel = float(np.sum(centered ** 3) / den_a)
    if not np.isfinite(accel):                                                   # P5
        return fallback(5, None, False)

    adjusted: list[float] = []
    for q in qs:
        zq = float(ndtri(q))
        den = 1.0 - accel * (z0 + zq)
        if den == 0.0 or not np.isfinite(den):                                   # P6
            return fallback(6, accel, True)
        adj = float(ndtr(z0 + (z0 + zq) / den))
        if not np.isfinite(adj):                                                 # P7
            return fallback(7, accel, True)
        adjusted.append(adj)
    if interval_type == TWO_SIDED and adjusted[0] >= adjusted[1]:                # P8
        return fallback(8, accel, True)

    # ---- 8.7.6 saturation is a completed BCa result, never a fallback -----------------------
    for i, adj in enumerate(adjusted):
        if adj == 0.0 or adj == 1.0:
            warns.append(W_SAT_UPPER if (interval_type == TWO_SIDED and i == 1) else W_SAT_LOWER)

    ws = _sorted_warnings(warns)
    return BcaOutcome(z0=z0, acceleration=accel, acceleration_defined=True,
                      interval_type=interval_type, interval_method=METHOD_BCA,
                      confidence_level=CONFIDENCE_LEVEL, bounds=_quantiles(rep, adjusted),
                      warnings=ws, status=STATUS_OK_WARN if ws else STATUS_OK)


# --------------------------------------------------------------------------------------------------
# 5. one complete bootstrap task
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapTaskResult:
    task: TaskSpec
    analysis_id: str
    seed: SeedMaterial
    observed: float
    replicates: np.ndarray
    jackknife: np.ndarray
    bca: BcaOutcome

    @property
    def task_id(self) -> str:
        return self.task.task_id


def run_bootstrap_task(task: TaskSpec, analysis_id: str, n_images: int, B: int,
                       observed: float,
                       replicate_fn: Callable[[np.ndarray], float],
                       jackknife_fn: Callable[[int], float],
                       root_seed: int = ROOT_SEED) -> BootstrapTaskResult:
    """Row-wise draws from a task-local Generator; then the leave-one-image-out jackknife."""
    if n_images < 1:
        raise BootstrapError("n_images must be >= 1")
    if B < 1:
        raise BootstrapError("B must be >= 1")
    seed = derive_seed(analysis_id, task.comparison_id, task.statistic_name, root_seed)
    gen = make_generator(seed)

    reps = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = draw_replicate_indices(gen, n_images)
        reps[b] = replicate_fn(idx)
    jack = np.empty(n_images, dtype=np.float64)
    for i in range(n_images):
        jack[i] = jackknife_fn(i)

    bca = bca_interval(reps, jack, observed, task.interval_type)
    return BootstrapTaskResult(task=task, analysis_id=analysis_id, seed=seed,
                               observed=float(observed), replicates=reps, jackknife=jack,
                               bca=bca)


# --------------------------------------------------------------------------------------------------
# 6. scalar replicate/jackknife callables
# --------------------------------------------------------------------------------------------------
def scalar_statistic(name: str, d: np.ndarray) -> float:
    if name == MEAN_DELTA:
        return float(np.mean(d))
    if name == MEDIAN_DELTA:
        return float(np.median(d))
    if name == HL_SHIFT:
        return float(hodges_lehmann(d))                       # canonical A3a implementation
    raise BootstrapError(f"{name!r} is not a scalar statistic")


def scalar_callables(name: str, delta: np.ndarray):
    d = np.asarray(delta, dtype=np.float64)
    keep = np.arange(d.size)

    def replicate_fn(idx: np.ndarray) -> float:
        return scalar_statistic(name, d[idx])

    def jackknife_fn(i: int) -> float:
        return scalar_statistic(name, d[keep != i])

    return replicate_fn, jackknife_fn


# --------------------------------------------------------------------------------------------------
# 7. Source B -- descriptive E1->E3 carrier (Amendment C)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DescriptiveScalars:
    """Immutable carrier for the three descriptive E1->E3 observed values.

    Deliberately carries NO Wilcoxon, t-test, rank-biserial, Cohen's dz or Holm field -- not even a
    null placeholder -- so it can never be serialized as, or mistaken for, an inferential family
    member. It is an operational vehicle only and confers no inferential status.
    """
    comparison_id: str
    baseline_stage: str
    candidate_stage: str
    direction: str
    metric: str
    n_paired: int
    policy: str
    alignment_status: str
    mean_delta: float
    mean_delta_pp: float
    median_delta: float
    hodges_lehmann_shift: float
    observed_source: str = OBSERVED_SOURCE_PRIMITIVES

    def value(self, statistic_name: str) -> float:
        return {MEAN_DELTA: self.mean_delta, MEDIAN_DELTA: self.median_delta,
                HL_SHIFT: self.hodges_lehmann_shift}[statistic_name]


def descriptive_scalars(delta: np.ndarray, *, metric: str, policy: str,
                        alignment_status: str = "ok") -> DescriptiveScalars:
    """Source B: the public A3a primitives only -- never `run_comparison`."""
    d = np.asarray(delta, dtype=np.float64)
    shifts = engineering_shifts(d)
    return DescriptiveScalars(
        comparison_id=DESCRIPTIVE_E1_E3, baseline_stage="E1", candidate_stage="E3",
        direction="candidate_minus_baseline", metric=metric, n_paired=int(d.size),
        policy=policy, alignment_status=alignment_status,
        mean_delta=float(shifts.mean_delta), mean_delta_pp=float(shifts.mean_delta_pp),
        median_delta=float(shifts.median_delta),
        hodges_lehmann_shift=float(hodges_lehmann(d)))


def comparison_observed(result, statistic_name: str) -> float:
    """Source A: read the observed value out of a typed A3a `ComparisonResult`."""
    if statistic_name == MEAN_DELTA:
        return float(result.shifts.mean_delta)
    if statistic_name == MEDIAN_DELTA:
        return float(result.shifts.median_delta)
    if statistic_name == HL_SHIFT:
        return float(result.hodges_lehmann_shift)
    raise BootstrapError(f"{statistic_name!r} has no typed A3a source")


# --------------------------------------------------------------------------------------------------
# 8. runtime source-schema assertion (contract section 12.4.7.2)
# --------------------------------------------------------------------------------------------------
FROZEN_A3A_FIELDS: dict[str, tuple[str, ...]] = {
    "ComparisonResult": ("comparison_id", "baseline_stage", "candidate_stage", "metric",
                         "direction", "n_paired", "policy", "alignment_status", "wilcoxon",
                         "t_test", "rank_biserial", "cohens_dz", "hodges_lehmann_shift",
                         "shifts", "warnings", "status"),
    "WilcoxonResult": ("statistic", "zstatistic", "p_value", "status", "n_total", "n_zero",
                       "n_nonzero", "warnings", "zero_method", "correction", "alternative",
                       "method"),
    "TTestResult": ("statistic", "p_value", "df", "mean_difference", "sd_difference",
                    "standard_error", "status", "warnings", "alternative"),
    "RankBiserialResult": ("value", "r_plus", "r_minus", "denominator", "n_zero",
                           "n_tied_groups", "status", "sign_convention"),
    "CohensDzResult": ("value", "mean", "sd", "status"),
    "EngineeringShifts": ("mean_delta", "mean_delta_pp", "median_delta"),
    "HolmMember": ("comparison_id", "p_raw", "p_adjusted", "sorted_rank", "step_denominator",
                   "step_threshold", "reject", "library_reject", "boundary_note"),
    "HolmFamily": ("members", "alpha", "complete", "status"),
}


def assert_a3a_source_schema() -> None:
    """Fail closed on any added, missing, renamed or reordered A3a source field."""
    from . import tests as _t
    for name, frozen in FROZEN_A3A_FIELDS.items():
        cls = getattr(_t, name)
        actual = tuple(f.name for f in dataclasses.fields(cls))
        if actual != frozen:
            raise BootstrapError(
                f"{name} source field tuple {actual} != frozen contract tuple {frozen}; a source "
                "schema change requires a contract review and a protocol-version decision")


__all__ = [
    "BootstrapError", "TaskSpec", "FROZEN_TASK_MATRIX", "FROZEN_TASK_IDS", "TASK_BY_ID",
    "expected_comparison_ids", "SeedMaterial", "canonical_seed_bytes", "derive_seed",
    "make_generator", "reconstruct_seed_material", "draw_replicate_indices", "BcaOutcome",
    "bca_interval", "BootstrapTaskResult", "run_bootstrap_task", "scalar_statistic",
    "scalar_callables", "DescriptiveScalars", "descriptive_scalars", "comparison_observed",
    "FROZEN_A3A_FIELDS", "assert_a3a_source_schema",
]
