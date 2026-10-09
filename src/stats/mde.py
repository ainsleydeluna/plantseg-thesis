"""AM-17 item 3 minimum detectable effect: the pure computation (lane L-AM17-MDE; S4).

docs/PREREGISTRATION_AMENDMENTS.md AM-17 item 3 (b)-(d) and (f); docs/lane_specs/part1.md lane 6 (a). For each
seed pair the observed per-image VAL disease-only differences d (candidate - baseline, AM-5 rows already
dropped, in align_runs' sorted-id order) are shifted by a constant delta on the grid 0.000 ... 0.050 (step
0.001). B = 2,000 resamples of size n = n_planning are drawn with replacement and each is tested with the
pre-registered call scipy.stats.wilcoxon(x, zero_method='pratt', alternative='greater', correction=True,
method='approx'), one call per resample; the power at delta is the share of p < 0.00625. MDE_W(pair) is the
smallest delta with power >= 0.80 at delta and at the next two grid points (k = 0 ... 48 are the candidates:
the guard needs k + 1 and k + 2 on the grid), and MDE_W is the largest over the pairs. The analytic
cross-check is item 3(d) with n = n_planning; the power caveat is item 3(f), tau_P = report.TAU_P = 0.010.

Resampling: one index matrix per pair, numpy.random.default_rng(42).integers(0, m, size=(B, n),
dtype=numpy.int64), drawn once and reused at every delta (common random numbers). The draw depends neither
on delta nor on the values, so it is the matrix a generator re-seeded per (pair, delta) would draw:
default_rng(42).choice(d + delta, size=(B, n), replace=True) equals d[idx] + delta element for element
(scripts/smoke_mde.py U8b proves it on the pinned numpy). The three guarded grid points therefore share
their resampled rows: the guard protects against a single lucky grid point, not against independent noise.

d is not centred (item 3(c): "a constant shift delta is added to the observed differences"; lane 6 (a):
"(d + delta)"). Exact ties (d == 0, the Pratt zeros) all become +delta for every delta > 0, which deflates
MDE_W under the constant-shift null (lane 6 (f)); share_ties and n_zero are recorded with every pair.

Refused (MdeError): a d that is not a finite 1-D vector of at least two values, and a d of exact zeros only (two
runs that agree image for image), which the pre-registered call refuses at delta 0. A d whose differences are
nearly all exact zeros (share_ties near 1) can still draw an all-zero resample row at delta 0; each row has
probability share_ties ** n, and a pair about B times that, which is negligible unless all but a handful of the
differences are exact zeros. The call then raises its own ValueError (the CLI exits 4, unexpected). That case is
not guarded here: catching it would add a rule the pre-registered call does not have.

Pure: no I/O, no Git, no randomness beyond the seeded generator. Every rule is a module-level name read at
call time, so scripts/smoke_mde.py can replace one rule at a time (its mutation set).
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .ingest import EXPECTED_ROWS_TEST, EXPECTED_ROWS_VAL
from .report import TAU_P

N_TEST = EXPECTED_ROWS_TEST        # 1,561: item 3(b) scales to the TEST row count; no TEST file is read
N_VAL = EXPECTED_ROWS_VAL          # 846
B = 2000
RNG_SEED = 42
GRID = tuple(k / 1000 for k in range(51))       # 0.000 ... 0.050 as the correctly rounded doubles of k/1000
GUARD_POINTS = 3
ALPHA = 0.00625                    # the one-tailed alpha/8 of item 3(c)
POWER_MIN = 0.80
WILCOXON_KWARGS = {"zero_method": "pratt", "alternative": "greater", "correction": True, "method": "approx"}
WILCOXON_CALL = ("scipy.stats.wilcoxon(x, zero_method='pratt', alternative='greater', correction=True, "
                 "method='approx')")
INDEX_CALL = "numpy.random.default_rng(42).integers(0, m, size=(B, n), dtype=numpy.int64)"
Z_ALPHA, Z_POWER = 2.4977, 0.8416  # item 3(d): one-tailed alpha/8 = 0.00625 and power 0.80
DZ_FORMULA = "(2.4977 + 0.8416) / sqrt(n_planning)"
EFFICIENCY = (1.023, 1.076)        # item 3(d): MDE_t x [1.023, 1.076] (ARE 0.955 and 0.864)


class MdeError(ValueError):
    """An input the computation refuses: a programming or input error, never a verdict."""


def min_count(b: int) -> int:
    """The smallest rejection count whose power count / b is at least 0.80: ceil(4b / 5); 1600 at b = 2000."""
    return -(-4 * b // 5)


def n_planning(included: int) -> int:
    """Item 3(b), lane 6 (a): n = round(1561 x included / 846), as an exact rational, ties to even (the only
    tie is included = 423, which gives 780)."""
    if isinstance(included, bool) or not isinstance(included, int) or not 0 < included <= N_VAL:
        raise MdeError(f"n_included must be an int in 1..{N_VAL}, got {included!r}")
    return round(Fraction(N_TEST * included, N_VAL))


def n_planning_rule(included: int) -> str:
    return (f"round(Fraction({N_TEST} * {included}, {N_VAL})) = round({N_TEST * included}/{N_VAL}) = "
            f"{n_planning(included)} (exact rational, ties to even)")


def resample_indices(m: int, n: int, *, seed=None, b=None) -> np.ndarray:
    """The pair's index matrix: one draw of B x n indices into the m observed differences."""
    seed = RNG_SEED if seed is None else seed
    b = B if b is None else b
    return np.random.default_rng(seed).integers(0, m, size=(b, n), dtype=np.int64)


def wilcoxon_p(x) -> float:
    """The pre-registered call on one 1-D resample (item 3(c)); never vectorized."""
    from scipy import stats
    return float(stats.wilcoxon(x, **WILCOXON_KWARGS).pvalue)


def rejects(p: float) -> bool:
    """Lane 6 (a): power is the share of p < 0.00625 (strict)."""
    return p < ALPHA


def qualifies(count: int, need: int) -> bool:
    """Power >= 0.80, compared on the integer rejection count (count >= ceil(0.8 b))."""
    return count >= need


def rejection_counts(d: np.ndarray, idx: np.ndarray, grid) -> list[int]:
    """For each delta: the number of resampled rows of d[idx] + delta that the call rejects."""
    base = d[idx]
    counts = []
    for delta in grid:
        x = base + delta
        counts.append(sum(1 for r in range(x.shape[0]) if rejects(wilcoxon_p(x[r]))))
    return counts


def guard_index(counts, need: int):
    """The smallest grid index k with power >= 0.80 at k and at the next GUARD_POINTS - 1 points, or None."""
    for k in range(len(counts) - GUARD_POINTS + 1):
        if all(qualifies(c, need) for c in counts[k:k + GUARD_POINTS]):
            return k
    return None


def sample_sd(d) -> float:
    """Lane 6 (a): the SD with ddof = 1."""
    return float(np.std(np.asarray(d, dtype=np.float64), ddof=1))


def check_d(d) -> np.ndarray:
    """The pair's differences as float64, refused unless a finite 1-D vector of at least two values with at
    least one non-zero value."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or d.size < 2:
        raise MdeError(f"d must be a 1-D vector of at least two differences, got shape {d.shape}")
    if not np.isfinite(d).all():
        raise MdeError(f"d holds {int(np.count_nonzero(~np.isfinite(d)))} non-finite difference(s)")
    if np.count_nonzero(d) == 0:
        raise MdeError("every difference is an exact zero: the two runs agree image for image, and the "
                       "pre-registered call refuses an all-zero sample")
    return d


def pair_summary(d, n: int, *, seed=None, b=None, grid=None) -> dict:
    """One pair's shifted-null power curve and MDE_W, with the descriptive fields of lane 6 (c)."""
    d = check_d(d)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise MdeError(f"n must be a positive int, got {n!r}")
    seed = RNG_SEED if seed is None else seed
    b = B if b is None else b
    grid = GRID if grid is None else tuple(grid)
    m = int(d.size)
    idx = resample_indices(m, n, seed=seed, b=b)
    counts = [int(c) for c in rejection_counts(d, idx, grid)]
    need = min_count(b)
    k = guard_index(counts, need)
    n_zero = int(np.count_nonzero(d == 0.0))
    return {"m": m, "n": n, "n_zero": n_zero, "share_ties": n_zero / m, "mean": float(np.mean(d)),
            "sd": sample_sd(d), "rejections": counts,
            "power_curve": [[float(delta), c / b] for delta, c in zip(grid, counts)],
            "power_min_count": need, "mde_w_index": k, "mde_w": None if k is None else float(grid[k])}


def overall(pairs: dict) -> tuple[float, str]:
    """Item 3(c): MDE_W is the largest MDE_W over the pairs (the first pair in order on a tie)."""
    name = max(pairs, key=lambda key: pairs[key]["mde_w"])
    return pairs[name]["mde_w"], name


def analytic(sds, n: int) -> dict:
    """Item 3(d): SD_Delta = the largest of the pair SDs; dz_MDE = (2.4977 + 0.8416) / sqrt(n);
    MDE_t = dz_MDE x SD_Delta, with its Wilcoxon-efficiency range MDE_t x [1.023, 1.076]."""
    sd_delta = max(float(s) for s in sds)
    dz = (Z_ALPHA + Z_POWER) / math.sqrt(n)
    mde_t = dz * sd_delta
    return {"sd_delta": sd_delta, "dz_mde": dz, "dz_formula": DZ_FORMULA, "mde_t": mde_t,
            "mde_t_range": [mde_t * EFFICIENCY[0], mde_t * EFFICIENCY[1]]}


def power_caveat(mde_w: float) -> bool:
    """Item 3(f): the caveat holds when MDE_W > tau_P."""
    return mde_w > TAU_P


__all__ = ["N_TEST", "N_VAL", "B", "RNG_SEED", "GRID", "GUARD_POINTS", "ALPHA", "POWER_MIN",
           "WILCOXON_KWARGS", "WILCOXON_CALL", "INDEX_CALL", "Z_ALPHA", "Z_POWER", "DZ_FORMULA",
           "EFFICIENCY", "TAU_P", "MdeError", "min_count", "n_planning", "n_planning_rule",
           "resample_indices", "wilcoxon_p", "rejects", "qualifies", "rejection_counts", "guard_index",
           "sample_sd", "check_d", "pair_summary", "overall", "analytic", "power_caveat"]
