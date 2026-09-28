"""Teacher of record minus E1 seed 42 on VAL: the dataset-level mIoU gap with a paired image-level
bootstrap, and the per-image difference summary (lane L-AM17B-GAP).

AM-17b item 2 (DRAFT, adviser approval pending); docs/lane_specs/part1.md lane 7; DL-35 G3 and G5.
Descriptive: reported before any KD result, it gates, selects and replaces nothing.

Estimand. For one image weighting, each model's per-class TP, GT and PRED totals are re-accumulated
from its per-image sufficient statistics; union_c = GT_c + PRED_c - TP_c; the eligible classes are
recomputed under the rule (union-present {union_c > 0}, the R3 rule; GT-present {GT_c > 0}, AM-17 item
1(b)) within the requested class subset (all-class 0-115, or disease-only 1-115); mIoU is the exactly
rounded float64 mean (math.fsum) of TP_c / union_c over the eligible classes -- the arithmetic of
`src.stats.eligibility`'s `value_float64`, so the point estimate equals the difference of the two
artifacts' L-AM17-GTPRESENT `rules_float64` values. The statistic is mIoU(teacher) - mIoU(E1).

Entry point with a rule argument (DL-35 G5). `RuleGap` wraps `noninferiority.PooledStages` -- both
models' dense int64 per-image TP/GT/PRED aligned on one manifest order -- and evaluates the statistic at
any integer image weighting; PooledStages' own reducer is union-present only and is not called, and
neither the frozen task matrix (bootstrap.py) nor the TEST driver (ingest.py) is used. Totals are
computed in float64 from the int64 counts; every partial sum is an integer below 2**53 (checked), so
they are exact and equal the int64 totals.

Bootstrap (lane 7 (a)): `scipy.stats.bootstrap((idx,), stat, n_resamples=10000, method='BCa',
random_state=default_rng(42), vectorized=False)` over the 846 image indices, one shared index vector
for both models (the statistic resamples both). The leave-one-image-out jackknife acceleration is
computed first with SciPy's own formula; if it is non-finite, if BCa raises, or if BCa returns a
non-finite bound (SciPy then warns DegenerateDataWarning and returns NaN), the interval is the
percentile interval of the same B = 10,000 replicates (a fresh default_rng(42) draws them again) and
`method = 'percentile'` is recorded with the reason. Two-sided 95 %.

Per-image summary (lane 7 (b)): teacher - E1 per-image disease-only mIoU on the AM-5-included images
(REHEARSAL alignment): n, median, the quartiles and IQR (linear interpolation), mean, SD (ddof = 1, as
AM-17 item 3's SD_Delta), the shares teacher better / E1 better / exact ties, and the Hodges-Lehmann
shift (`src.stats.tests.hodges_lehmann`).

Import-time behaviour is side-effect free; SciPy is imported only when a bootstrap runs.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass

import numpy as np

from .eligibility import GT_PRESENT, RULES, UNION_PRESENT
from .noninferiority import PooledStages
from .tests import hodges_lehmann

B = 10_000
SEED = 42
CONFIDENCE_LEVEL = 0.95
SCOPES = ("all_class", "disease_only")
METHOD_BCA = "BCa"
METHOD_PERCENTILE = "percentile"
DIRECTION = "teacher_minus_e1"
_EXACT_LIMIT = 2 ** 53


class GapError(RuntimeError):
    """Malformed input or an undefined statistic. Always fatal."""


def scope_classes(scope: str, num_classes: int, background_index: int = 0) -> list[int]:
    if scope == "all_class":
        return list(range(num_classes))
    if scope == "disease_only":
        return [c for c in range(num_classes) if c != background_index]
    raise GapError(f"scope must be one of {SCOPES}, got {scope!r}")


def rule_miou(tp: np.ndarray, gt: np.ndarray, pred: np.ndarray, selected: np.ndarray,
              rule: str) -> tuple[float, int]:
    """(mIoU, eligible-class count) under `rule` within `selected`. No eligible class is fatal."""
    union = gt + pred - tp
    if np.any(union < 0):
        raise GapError("negative union: malformed sufficient statistics")
    present = (union > 0) if rule == UNION_PRESENT else (gt > 0)
    eligible = present & selected
    n = int(np.count_nonzero(eligible))
    if n == 0:
        raise GapError(f"no class is eligible under {rule}: the dataset-level mIoU is undefined")
    return math.fsum((tp[eligible] / union[eligible]).tolist()) / n, n


class RuleGap:
    """candidate - baseline dataset-level mIoU under one eligibility rule, at any image weighting.

    `stages` is a `noninferiority.PooledStages` (baseline = E1, candidate = teacher). Calling the object
    with an index vector (a bootstrap or jackknife sample) counts each image's multiplicity and returns
    the statistic; `at_counts` takes the multiplicities directly.
    """

    def __init__(self, stages: PooledStages, rule: str, classes):
        if rule not in RULES:
            raise GapError(f"rule must be one of {RULES}, got {rule!r}")
        n, c = stages.n_images, stages.num_classes
        subset = sorted({int(k) for k in classes})
        if not subset or subset[0] < 0 or subset[-1] >= c:
            raise GapError(f"classes must be a non-empty subset of 0..{c - 1}")
        blocks = (stages.base_tp, stages.base_gt, stages.base_pred,
                  stages.cand_tp, stages.cand_gt, stages.cand_pred)
        for blk in blocks:
            if blk.shape != (n, c) or blk.dtype != np.int64:
                raise GapError("PooledStages arrays must be int64 [n_images, num_classes]")
            if blk.size and int(blk.min()) < 0:
                raise GapError("negative sufficient statistic")
        dense = np.concatenate(blocks, axis=1)
        if n * (int(dense.max()) if dense.size else 0) >= _EXACT_LIMIT:
            raise GapError("sufficient statistics too large for exact float64 accumulation")
        self.rule, self.classes, self.n, self.num_classes = rule, tuple(subset), n, c
        self._selected = np.zeros(c, dtype=bool)
        self._selected[subset] = True
        self._dense = dense.astype(np.float64)          # exact: integers below 2**53

    def _models(self, counts) -> tuple[tuple, tuple]:
        w = np.asarray(counts, dtype=np.float64)
        if w.shape != (self.n,):
            raise GapError(f"weights must have shape ({self.n},), got {w.shape}")
        t = w @ self._dense
        c = self.num_classes
        base = rule_miou(t[0:c], t[c:2 * c], t[2 * c:3 * c], self._selected, self.rule)
        cand = rule_miou(t[3 * c:4 * c], t[4 * c:5 * c], t[5 * c:6 * c], self._selected, self.rule)
        return base, cand

    def at_counts(self, counts) -> float:
        base, cand = self._models(counts)
        return cand[0] - base[0]

    def __call__(self, idx) -> float:
        counts = np.bincount(np.asarray(idx, dtype=np.intp), minlength=self.n)
        if counts.shape != (self.n,):
            raise GapError("image index out of range")
        return self.at_counts(counts)

    def point(self) -> dict:
        """The full-sample statistic and each model's value and eligible-class count."""
        (vb, nb), (vc, nc) = self._models(np.ones(self.n, dtype=np.int64))
        return {"point": vc - vb, "teacher": vc, "e1": vb, "n_eligible_teacher": nc, "n_eligible_e1": nb}

    def jackknife(self) -> np.ndarray:
        """Leave-one-image-out values by re-accumulation (the samples SciPy's BCa jackknife uses)."""
        out = np.empty(self.n, dtype=np.float64)
        counts = np.ones(self.n, dtype=np.int64)
        for i in range(self.n):
            counts[i] = 0
            out[i] = self.at_counts(counts)
            counts[i] = 1
        return out


def jackknife_acceleration(jack: np.ndarray) -> float:
    """SciPy 1.11's BCa acceleration (scipy.stats._resampling._bca_interval), term for term.

    Non-finite (NaN) when every jackknife value is equal, e.g. two identical models.
    """
    jack = np.asarray(jack, dtype=np.float64)
    n = jack.shape[-1]
    u = (n - 1) * (jack.mean(axis=-1, keepdims=True) - jack)
    num = (u ** 3).sum(axis=-1) / n ** 3
    den = (u ** 2).sum(axis=-1) / n ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(1 / 6 * num / den ** (3 / 2))


def _z0(dist: np.ndarray, point: float) -> float:
    """SciPy's BCa bias correction: ndtri of the 'mean'-kind percentile of the point estimate."""
    from scipy.special import ndtri
    p = ((dist < point).sum() + (dist <= point).sum()) / (2 * dist.size)
    return float(ndtri(p))


def _finite_or_none(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def bootstrap_gap(gap: RuleGap, *, n_resamples: int = B, seed: int = SEED) -> dict:
    """The lane 7 (a) interval for one rule and class subset. Deterministic for a given seed."""
    from scipy import __version__ as scipy_version
    from scipy.stats import bootstrap

    idx = np.arange(gap.n)
    pt = gap.point()
    point = pt["point"]
    if gap(idx) != point:                                   # the statistic SciPy sees at unit weights
        raise GapError("the statistic is not reproducible at unit weights")
    accel = jackknife_acceleration(gap.jackknife())
    reason, caught, res_bca = None, [], None
    if not math.isfinite(accel):
        reason = "jackknife acceleration is non-finite (degenerate jackknife)"
    else:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res_bca = bootstrap((idx,), gap, n_resamples=n_resamples, method="BCa",
                                    random_state=np.random.default_rng(seed), vectorized=False)
            caught = sorted({f"{type(m.message).__name__}: {m.message}" for m in w})
            lo, hi = res_bca.confidence_interval
            if not (math.isfinite(lo) and math.isfinite(hi)):
                reason = "BCa returned a non-finite bound"
        except Exception as e:                              # noqa: BLE001 -- recorded, then percentile
            reason = f"BCa raised {type(e).__name__}: {e}"
    if reason is None:
        res, method = res_bca, METHOD_BCA
    else:
        res = bootstrap((idx,), gap, n_resamples=n_resamples, method="percentile",
                        random_state=np.random.default_rng(seed), vectorized=False)
        method = METHOD_PERCENTILE
        if res_bca is not None and not np.array_equal(res_bca.bootstrap_distribution,
                                                      res.bootstrap_distribution):
            raise GapError("the percentile fallback did not reproduce the BCa replicates")
    dist = np.asarray(res.bootstrap_distribution, dtype=np.float64)
    if dist.shape != (n_resamples,) or not np.all(np.isfinite(dist)):
        raise GapError("non-finite or missing bootstrap replicates")
    lo, hi = (float(v) for v in res.confidence_interval)
    return {
        "point": point, "ci_low": lo, "ci_high": hi, "method": method, "B": int(n_resamples),
        "seed": int(seed), "confidence_level": CONFIDENCE_LEVEL,
        "point_pp": 100 * point, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi,
        "fallback_reason": reason, "acceleration": _finite_or_none(accel),
        "z0": _finite_or_none(_z0(dist, point)), "contains_point": lo <= point <= hi,
        "teacher_miou": pt["teacher"], "e1_miou": pt["e1"],
        "n_eligible_teacher": pt["n_eligible_teacher"], "n_eligible_e1": pt["n_eligible_e1"],
        "bootstrap_standard_error": float(res.standard_error),
        "replicates_sha256": hashlib.sha256(dist.astype("<f8").tobytes()).hexdigest(),
        "warnings": caught, "scipy_version": scipy_version,
    }


def paired_totals_identical_gt(stages: PooledStages) -> None:
    """Both models must score the same per-image ground truth (the GT-present rule depends on it)."""
    if not np.array_equal(stages.base_gt, stages.cand_gt):
        raise GapError("the two artifacts' per-image ground-truth counts differ: not the same VAL masks")


def gap_rules(stages: PooledStages, *, background_index: int = 0, n_resamples: int = B,
              seed: int = SEED) -> dict:
    """{scope: {rule: interval}} for both rules and both class subsets (four bootstraps, one seed)."""
    paired_totals_identical_gt(stages)
    return {scope: {rule: bootstrap_gap(RuleGap(stages, rule, scope_classes(scope, stages.num_classes,
                                                                            background_index)),
                                        n_resamples=n_resamples, seed=seed)
                    for rule in (UNION_PRESENT, GT_PRESENT)}
            for scope in SCOPES}


# --------------------------------------------------------------------------------------------------
# per-image differences (lane 7 (b))
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PerImageSummary:
    n: int
    median: float
    q25: float
    q75: float
    iqr: float
    mean: float
    sd: float
    share_teacher_better: float
    share_e1_better: float
    share_ties: float
    hl_shift: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def per_image_summary(delta) -> PerImageSummary:
    """Descriptive summary of teacher - E1 per-image differences (positive: the teacher is better)."""
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 1 or d.size < 2 or not np.all(np.isfinite(d)):
        raise GapError("per-image differences must be a finite 1-D vector of at least two values")
    q25, q75 = (float(np.percentile(d, q, method="linear")) for q in (25, 75))
    n = int(d.size)
    return PerImageSummary(
        n=n, median=float(np.median(d)), q25=q25, q75=q75, iqr=q75 - q25,
        mean=math.fsum(d.tolist()) / n, sd=float(np.std(d, ddof=1)),
        share_teacher_better=int(np.count_nonzero(d > 0)) / n,
        share_e1_better=int(np.count_nonzero(d < 0)) / n,
        share_ties=int(np.count_nonzero(d == 0)) / n,
        hl_shift=float(hodges_lehmann(d)))


__all__ = ["B", "SEED", "CONFIDENCE_LEVEL", "SCOPES", "METHOD_BCA", "METHOD_PERCENTILE", "DIRECTION",
           "GapError", "scope_classes", "rule_miou", "RuleGap", "jackknife_acceleration",
           "bootstrap_gap", "paired_totals_identical_gt", "gap_rules", "PerImageSummary",
           "per_image_summary"]
