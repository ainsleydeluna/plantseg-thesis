"""Paired tests, Holm family control and point effect sizes (A3a).

Implements docs/STATISTICAL_ANALYSIS_CONTRACT.md sections 3-7. Every configuration here is frozen;
none may be re-decided in code.

A3a is the DETERMINISTIC layer only. Nothing here bootstraps, and no BCa field is emitted -- not
even as a null placeholder -- so a partial `family.json` cannot be mistaken for a complete one.
A3b consumes `ComparisonResult` without recomputing any paired test.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.stats as st
from statsmodels.stats.multitest import multipletests

from .align import PairedVector
from .ingest import Policy

ALPHA = 0.05

# Frozen Wilcoxon configuration (contract section 3) -- `method="approx"` is explicit, never "auto".
WILCOXON_KWARGS = dict(zero_method="pratt", correction=True,
                       alternative="greater", method="approx")

CANONICAL_COMPARISON_IDS = (
    "accuracy_e1_e2", "accuracy_e2_e3", "accuracy_e4_e5", "accuracy_e7_e6",
    "accuracy_e4_e7", "accuracy_e5_e6", "accuracy_e1_e6", "robustness_e1_e6",
)

STATUS_OK = "ok"
STATUS_ALL_ZERO = "degenerate_all_zero"
STATUS_INSUFFICIENT_NONZERO = "insufficient_nonzero_pairs"
STATUS_INSUFFICIENT_PAIRS = "insufficient_pairs"
STATUS_ZERO_VAR_NONZERO = "degenerate_zero_variance_nonzero"
STATUS_RB_UNDEFINED = "undefined_no_nonzero_rank_sum"
STATUS_DZ_ZERO_SD = "undefined_zero_standard_deviation"


class StatsError(RuntimeError):
    """Invalid input -- an integrity error, never a statistical-result status."""


def _validate(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1:
        raise StatsError(f"differences must be 1-D, got shape {d.shape}")
    if d.size == 0:
        raise StatsError("empty difference vector")
    if not np.isfinite(d).all():
        raise StatsError("non-finite value in the difference vector")
    return d


# --------------------------------------------------------------------------------------------------
# 3. Wilcoxon
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float | None
    zstatistic: float | None
    p_value: float | None
    status: str
    n_total: int
    n_zero: int
    n_nonzero: int
    warnings: tuple[str, ...] = ()
    zero_method: str = "pratt"
    correction: bool = True
    alternative: str = "greater"
    method: str = "approx"


def wilcoxon_test(d) -> WilcoxonResult:
    """Frozen primary test. TIES ARE VALID -- only all-zero and <2 nonzero are degenerate."""
    d = _validate(d)
    n_total = int(d.size)
    n_zero = int(np.count_nonzero(d == 0.0))
    n_nonzero = n_total - n_zero

    if n_nonzero == 0:
        return WilcoxonResult(None, None, 1.0, STATUS_ALL_ZERO, n_total, n_zero, 0)
    if n_nonzero < 2:
        return WilcoxonResult(None, None, None, STATUS_INSUFFICIENT_NONZERO,
                              n_total, n_zero, n_nonzero)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        res = st.wilcoxon(d, **WILCOXON_KWARGS)
    msgs = tuple(f"{w.category.__name__}: {w.message}" for w in caught)

    W, z, p = float(res.statistic), float(res.zstatistic), float(res.pvalue)
    if not (math.isfinite(W) and math.isfinite(z) and math.isfinite(p)):
        raise StatsError(
            f"SciPy returned a non-finite Wilcoxon result (W={W}, z={z}, p={p}) for a vector "
            f"with {n_nonzero} nonzero differences, which no frozen degenerate policy covers")
    return WilcoxonResult(W, z, p, STATUS_OK, n_total, n_zero, n_nonzero, msgs)


# --------------------------------------------------------------------------------------------------
# 5. paired t-test (sensitivity only -- never enters the Holm family)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TTestResult:
    statistic: float | None
    p_value: float | None
    df: int | None
    mean_difference: float
    sd_difference: float | None
    standard_error: float | None
    status: str
    warnings: tuple[str, ...] = ()
    alternative: str = "greater"


def paired_t_test(candidate, baseline) -> TTestResult:
    cand = _validate(candidate)
    base = _validate(baseline)
    if cand.size != base.size:
        raise StatsError(f"unequal vector lengths: {cand.size} vs {base.size}")
    d = cand - base
    n = int(d.size)
    mean_d = float(d.mean())

    if n < 2:
        return TTestResult(None, None, None, mean_d, None, None, STATUS_INSUFFICIENT_PAIRS)
    if np.all(d == 0.0):
        return TTestResult(None, 1.0, n - 1, 0.0, 0.0, 0.0, STATUS_ALL_ZERO)
    sd = float(d.std(ddof=1))
    if sd == 0.0:                                   # constant NONZERO differences -> t would be inf
        return TTestResult(None, None, n - 1, mean_d, 0.0, 0.0, STATUS_ZERO_VAR_NONZERO)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        res = st.ttest_rel(cand, base, alternative="greater")
    msgs = tuple(f"{w.category.__name__}: {w.message}" for w in caught)
    t, p = float(res.statistic), float(res.pvalue)
    if not (math.isfinite(t) and math.isfinite(p)):
        raise StatsError(f"SciPy returned a non-finite t-test result (t={t}, p={p})")
    return TTestResult(t, p, int(getattr(res, "df", n - 1)), mean_d, sd,
                       sd / math.sqrt(n), STATUS_OK, msgs)


# --------------------------------------------------------------------------------------------------
# 7. rank-biserial (Pratt-compatible, frozen denominator R+ + R-)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RankBiserialResult:
    value: float | None
    r_plus: float
    r_minus: float
    denominator: float
    n_zero: int
    n_tied_groups: int
    status: str
    sign_convention: str = "positive favours the candidate"


def rank_biserial(d) -> RankBiserialResult:
    """`(R+ - R-) / (R+ + R-)`; zeros are RANKED (Pratt) but their ranks enter neither sum."""
    d = _validate(d)
    absd = np.abs(d)
    ranks = st.rankdata(absd, method="average")          # zeros included in ranking
    pos = d > 0.0
    neg = d < 0.0
    r_plus = float(ranks[pos].sum())
    r_minus = float(ranks[neg].sum())
    denom = r_plus + r_minus                            # zero ranks excluded from BOTH sums
    n_zero = int(np.count_nonzero(d == 0.0))
    _, counts = np.unique(absd, return_counts=True)
    n_tied = int(np.count_nonzero(counts > 1))

    if denom == 0.0:
        return RankBiserialResult(None, r_plus, r_minus, 0.0, n_zero, n_tied,
                                  STATUS_RB_UNDEFINED)
    value = (r_plus - r_minus) / denom
    if not math.isfinite(value) or not (-1.0 - 1e-9 <= value <= 1.0 + 1e-9):
        raise StatsError(f"rank-biserial {value!r} outside [-1, 1]")
    return RankBiserialResult(float(value), r_plus, r_minus, denom, n_zero, n_tied, STATUS_OK)


# --------------------------------------------------------------------------------------------------
# 8. other point effect sizes
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CohensDzResult:
    value: float | None
    mean: float
    sd: float
    status: str


def cohens_dz(d) -> CohensDzResult:
    d = _validate(d)
    if d.size < 2:
        return CohensDzResult(None, float(d.mean()), 0.0, STATUS_INSUFFICIENT_PAIRS)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0.0:
        return CohensDzResult(None, m, 0.0, STATUS_DZ_ZERO_SD)
    return CohensDzResult(m / sd, m, sd, STATUS_OK)


def hodges_lehmann(d) -> float:
    """Exact median of Walsh averages `(d_i + d_j)/2` with i <= j. Deterministic, float64."""
    d = _validate(d)
    n = d.size
    walsh = (np.add.outer(d, d) * 0.5)[np.triu_indices(n, k=0)]
    m = walsh.size
    if m % 2:
        return float(np.partition(walsh, m // 2)[m // 2])
    lo = np.partition(walsh, [m // 2 - 1, m // 2])
    return float(0.5 * (lo[m // 2 - 1] + lo[m // 2]))


@dataclass(frozen=True)
class EngineeringShifts:
    mean_delta: float
    mean_delta_pp: float
    median_delta: float


def engineering_shifts(d) -> EngineeringShifts:
    d = _validate(d)
    m = float(d.mean())
    return EngineeringShifts(m, 100.0 * m, float(np.median(d)))


# --------------------------------------------------------------------------------------------------
# 9. comparison result (A3a-owned fields ONLY -- no BCa placeholders)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ComparisonResult:
    comparison_id: str
    baseline_stage: str
    candidate_stage: str
    metric: str
    direction: str
    n_paired: int
    policy: Policy
    alignment_status: str
    wilcoxon: WilcoxonResult
    t_test: TTestResult
    rank_biserial: RankBiserialResult
    cohens_dz: CohensDzResult
    hodges_lehmann_shift: float
    shifts: EngineeringShifts
    warnings: tuple[str, ...] = ()
    status: str = STATUS_OK

    @property
    def primary_p(self) -> float | None:
        return self.wilcoxon.p_value


def run_comparison(comparison_id: str, paired: PairedVector, *, baseline_stage: str,
                   candidate_stage: str) -> ComparisonResult:
    if comparison_id not in CANONICAL_COMPARISON_IDS:
        raise StatsError(f"unknown comparison_id {comparison_id!r}")
    d = paired.delta
    w = wilcoxon_test(d)
    t = paired_t_test(paired.candidate, paired.baseline)
    rb = rank_biserial(d)
    dz = cohens_dz(d)
    hl = hodges_lehmann(d)
    sh = engineering_shifts(d)
    status = STATUS_OK if w.status == STATUS_OK else w.status
    return ComparisonResult(
        comparison_id=comparison_id, baseline_stage=baseline_stage,
        candidate_stage=candidate_stage, metric=paired.metric,
        direction="candidate_minus_baseline", n_paired=paired.n, policy=paired.policy,
        alignment_status=paired.alignment_status, wilcoxon=w, t_test=t, rank_biserial=rb,
        cohens_dz=dz, hodges_lehmann_shift=hl, shifts=sh,
        warnings=tuple(w.warnings) + tuple(t.warnings), status=status)


# --------------------------------------------------------------------------------------------------
# 10. Holm family
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HolmMember:
    comparison_id: str
    p_raw: float
    p_adjusted: float
    sorted_rank: int
    step_denominator: int
    step_threshold: float
    reject: bool
    library_reject: bool
    boundary_note: str | None = None


@dataclass(frozen=True)
class HolmFamily:
    members: tuple[HolmMember, ...]
    alpha: float
    complete: bool
    status: str


def holm_audit(pairs: Sequence[tuple[str, float]], alpha: float = ALPHA) -> list[HolmMember]:
    """Pure strict-boundary Holm audit. Equality NEVER rejects (contract section 6).

    Independent of statsmodels so the protocol decision can be verified; the library's adjusted
    p-value is attached separately by `holm_family`.
    """
    k = len(pairs)
    order = sorted(range(k), key=lambda i: (pairs[i][1], pairs[i][0]))   # stable on ties
    out: dict[str, HolmMember] = {}
    still_rejecting = True
    for rank, idx in enumerate(order):
        cid, p = pairs[idx]
        denom = k - rank
        thr = alpha / denom
        rej = still_rejecting and (p < thr)          # STRICT: equality does not reject
        if not rej:
            still_rejecting = False
        note = "p_raw == step_threshold (strict rule: no rejection)" if p == thr else None
        out[cid] = HolmMember(cid, float(p), float("nan"), rank + 1, denom, float(thr),
                              rej, False, note)
    return [out[cid] for cid, _ in pairs]


def holm_family(results: Sequence[ComparisonResult], alpha: float = ALPHA) -> HolmFamily:
    """Finalise the family. Requires all eight canonical IDs exactly once."""
    ids = [r.comparison_id for r in results]
    missing = [c for c in CANONICAL_COMPARISON_IDS if c not in ids]
    dupes = sorted({c for c in ids if ids.count(c) > 1})
    unexpected = sorted({c for c in ids if c not in CANONICAL_COMPARISON_IDS})
    if missing or dupes or unexpected:
        raise StatsError(
            f"family is not finalisable: missing={missing} duplicate={dupes} "
            f"unexpected={unexpected}")
    bad = [r.comparison_id for r in results if r.primary_p is None]
    if bad:
        raise StatsError(f"undefined primary p-value for {bad}; family cannot be finalised")
    bad = [r.comparison_id for r in results
           if r.alignment_status not in ("ok", "ok_with_warnings")]
    if bad:
        raise StatsError(f"comparison(s) failed alignment integrity: {bad}")

    ordered = sorted(results, key=lambda r: CANONICAL_COMPARISON_IDS.index(r.comparison_id))
    pairs = [(r.comparison_id, float(r.primary_p)) for r in ordered]
    audit = holm_audit(pairs, alpha)
    lib_rej, lib_adj, _, _ = multipletests([p for _, p in pairs], alpha=alpha, method="holm",
                                           is_sorted=False, returnsorted=False)

    members = []
    for m, adj, lrej in zip(audit, lib_adj, lib_rej):
        note = m.boundary_note
        strict = m.reject
        if float(adj) == alpha:                      # adjusted p exactly alpha -> no rejection
            strict = False
            note = (note + "; " if note else "") + "p_adjusted == alpha (strict rule: no rejection)"
        if bool(lrej) != strict and note is None:
            note = "library reject differs from the strict protocol decision"
        members.append(HolmMember(m.comparison_id, m.p_raw, float(adj), m.sorted_rank,
                                  m.step_denominator, m.step_threshold, strict, bool(lrej), note))
    return HolmFamily(tuple(members), alpha, True, STATUS_OK)
