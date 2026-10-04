"""D1 statistics: NMF-draw sensitivity of the distillation targets (AM-18 item 1(a); lane L-TEACHER-DIAG).

Pure numpy: no model, no file, no repository state. One crop gives K = 8 teacher logit maps z_1..z_8
([8, C, 64, 64], the eight head calls on one backbone output, in call order) and V, the trainer's
all-valid min-pooled cells (losses.downsample_validity). Everything below is restricted to V.

Integers (Python ints, exact):
    N_i  = sum over the 28 pairs a < b of |{p in V : argmax z_a(p) != argmax z_b(p)}|   (first max on ties)
    D_i  = 28 * |V_i|
    N_L,i, D_L,i  the same over L_i = {p in V : the 8-draw mean probability at T = 1 peaks in 1..115}
    H_i  = |{p in V : argmax mean_{k<=4} p_k != argmax mean_{k>=5} p_k}|, p_k = softmax(z_k) at T = 1
The branch is decided in integers: below_0_03 = 100*N < 3*D; at_least_0_10 = 10*N >= D.

Float64 (logits cast to float64 before any softmax; KL in the log domain, never clamped):
    KL_logit  = sum over crops, draws and cells of KL(p_k || pbar) / (8 * sum |V_i|),
                p_k = softmax(z_k / 4) over classes, log pbar = logsumexp_k(log p_k) - ln 8
    KL_cwd    = mean over (crop with V_i non-empty, draw, channel) of KL(q_kc || qbar_c),
                q_kc = softmax over the cells of V_i of z_k[c] / 4 (the CWD logit-map construction,
                its mask, temperature and axis, in float64), log qbar_c = logsumexp_k(log q_kc) - ln 8
    d2 (first draw z_1): max-probability and entropy at T = 1 and T = 4, and the count of cells with
                max-probability > 0.99 at T = 1 (strict); pooled as sum over crops / sum |V_i|
Sums are exactly rounded (math.fsum).

Bootstrap of F: I = default_rng(1801).integers(0, 256, size=(10000, 256)) in one call, over all crops
(empty ones included); replicate = sum N[I] / sum D[I] from int64 sums; bounds =
np.percentile(reps, [2.5, 97.5], method="linear"). A replicate with sum D = 0 makes the CI null (the
count is recorded); nanpercentile is never used.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field

import numpy as np

K_DRAWS = 8
N_PAIRS = K_DRAWS * (K_DRAWS - 1) // 2          # 28
T_KD = 4.0                                      # T_logit = T_cwd = 4 (configs/distill.py)
HIGH_CONFIDENCE = 0.99
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 1801
BOOTSTRAP_LEVEL = (2.5, 97.5)
LN_K = math.log(K_DRAWS)


def log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    s = x - m
    return s - np.log(np.sum(np.exp(s), axis=axis, keepdims=True))


def logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def pairwise_flips(labels: np.ndarray) -> int:
    """sum over a < b of |{p : labels[a, p] != labels[b, p]}| as a Python int."""
    k = labels.shape[0]
    return sum(int(np.count_nonzero(labels[a] != labels[b])) for a in range(k) for b in range(a + 1, k))


@dataclass
class CropStats:
    n_valid: int = 0
    N: int = 0
    D: int = 0
    N_L: int = 0
    D_L: int = 0
    n_lesion: int = 0
    H: int = 0
    kl_logit_sum: float = 0.0
    kl_cwd_sum: float = 0.0
    n_cwd: int = 0
    d2: dict = field(default_factory=lambda: {"max_prob_t1_sum": 0.0, "entropy_t1_sum": 0.0,
                                              "max_prob_t4_sum": 0.0, "entropy_t4_sum": 0.0,
                                              "n_max_prob_gt_0_99_t1": 0})

    def as_dict(self) -> dict:
        return asdict(self)


def crop_statistics(logits8: np.ndarray, valid: np.ndarray, *, disease_min: int = 1,
                    disease_max: int = 115) -> CropStats:
    """One crop's integers and float64 sums. `logits8` is [8, C, h, w]; `valid` is bool [h, w]."""
    z = np.asarray(logits8)
    v = np.asarray(valid, dtype=bool)
    if z.ndim != 4 or z.shape[0] != K_DRAWS:
        raise ValueError(f"expected [{K_DRAWS}, C, h, w] logits, got {z.shape}")
    if v.shape != z.shape[-2:]:
        raise ValueError(f"valid mask {v.shape} does not match the logit grid {z.shape[-2:]}")
    if not np.isfinite(z).all():
        raise ValueError("non-finite logits")
    c = z.shape[1]
    idx = np.flatnonzero(v.reshape(-1))
    nv = int(idx.size)
    out = CropStats(n_valid=nv, D=N_PAIRS * nv)
    if nv == 0:
        return out
    zv = z.reshape(K_DRAWS, c, -1)[:, :, idx]                      # [K, C, nv], the given dtype
    labels = np.argmax(zv, axis=1)                                 # [K, nv], first max on ties
    out.N = pairwise_flips(labels)

    z64 = zv.astype(np.float64)
    lsm1 = log_softmax(z64, axis=1)                                # T = 1, over classes
    p1 = np.exp(lsm1)
    mean_all = p1.sum(axis=0) / K_DRAWS
    peak = np.argmax(mean_all, axis=0)
    lesion = (peak >= disease_min) & (peak <= disease_max)
    out.n_lesion = int(np.count_nonzero(lesion))
    out.D_L = N_PAIRS * out.n_lesion
    out.N_L = pairwise_flips(labels[:, lesion]) if out.n_lesion else 0
    half_a = np.argmax(p1[:4].sum(axis=0) / 4.0, axis=0)
    half_b = np.argmax(p1[4:].sum(axis=0) / 4.0, axis=0)
    out.H = int(np.count_nonzero(half_a != half_b))

    lsm4 = log_softmax(z64 / T_KD, axis=1)                         # p_k at T = 4
    log_pbar = logsumexp(lsm4, axis=0) - LN_K                      # [C, nv]
    kl_logit = np.sum(np.exp(lsm4) * (lsm4 - log_pbar[None]), axis=1)    # [K, nv]
    out.kl_logit_sum = math.fsum(kl_logit.ravel().tolist())

    lq = log_softmax(z64 / T_KD, axis=2)                           # q_kc over the cells of V
    log_qbar = logsumexp(lq, axis=0) - LN_K                        # [C, nv]
    kl_cwd = np.sum(np.exp(lq) * (lq - log_qbar[None]), axis=2)    # [K, C]
    out.kl_cwd_sum = math.fsum(kl_cwd.ravel().tolist())
    out.n_cwd = K_DRAWS * c

    first1, first4 = lsm1[0], lsm4[0]                              # the first draw, [C, nv]
    max1 = np.exp(np.max(first1, axis=0))
    max4 = np.exp(np.max(first4, axis=0))
    out.d2 = {"max_prob_t1_sum": math.fsum(max1.tolist()),
              "entropy_t1_sum": math.fsum((-np.sum(np.exp(first1) * first1, axis=0)).tolist()),
              "max_prob_t4_sum": math.fsum(max4.tolist()),
              "entropy_t4_sum": math.fsum((-np.sum(np.exp(first4) * first4, axis=0)).tolist()),
              "n_max_prob_gt_0_99_t1": int(np.count_nonzero(max1 > HIGH_CONFIDENCE))}
    return out


def branch(N: int, D: int) -> dict:
    """The integer rule. D == 0: no F, booleans and branch null."""
    if D == 0:
        return {"status": "F not produced", "below_0_03": None, "at_least_0_10": None, "branch": None}
    below = 100 * N < 3 * D
    at_least = 10 * N >= D
    return {"status": "F produced", "below_0_03": below, "at_least_0_10": at_least,
            "branch": "lt_0_03" if below else ("ge_0_10" if at_least else "mid")}


def bootstrap_f(n_i, d_i, *, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    """Image-level percentile bootstrap of F = sum N / sum D over the crops (all of them)."""
    n_arr = np.asarray(n_i, dtype=np.int64)
    d_arr = np.asarray(d_i, dtype=np.int64)
    if n_arr.shape != d_arr.shape or n_arr.ndim != 1 or n_arr.size == 0:
        raise ValueError("N_i and D_i must be equal-length non-empty vectors")
    n = int(n_arr.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    idx_sha = hashlib.sha256(np.ascontiguousarray(idx, dtype="<i8").tobytes()).hexdigest()
    s_n = n_arr[idx].sum(axis=1)
    s_d = d_arr[idx].sum(axis=1)
    zero_d = int(np.count_nonzero(s_d == 0))
    total_n, total_d = int(n_arr.sum()), int(d_arr.sum())
    point = (total_n / total_d) if total_d else None
    out = {"B": int(b), "seed": int(seed), "generator": "numpy.random.default_rng",
           "call": f"rng.integers(0, {n}, size=({b}, {n})), one call",
           "numpy": np.__version__, "index_sha256": idx_sha,
           "index_sha256_definition": "sha256 of I as int64, little-endian, C order",
           "level_percentiles": list(BOOTSTRAP_LEVEL), "method": 'np.percentile(method="linear")',
           "replicates_with_zero_D": zero_d, "point": point}
    if zero_d:
        out.update({"low": None, "high": None, "replicates_sha256": None, "contains_point": None,
                    "note": "a replicate has sum D = 0: the interval is not produced"})
        return out
    reps = s_n / s_d
    lo, hi = np.percentile(reps, list(BOOTSTRAP_LEVEL), method="linear")
    lo, hi = float(lo), float(hi)
    out.update({"low": lo, "high": hi,
                "replicates_sha256": hashlib.sha256(np.ascontiguousarray(reps, dtype="<f8").tobytes()).hexdigest(),
                "contains_point": (lo <= point <= hi) if point is not None else None})
    return out


def summarize(crops: list[CropStats], *, gated: bool = True) -> dict:
    """F (with the integer branch when gated), F_lesion, F_halves, KL_logit, KL_cwd and d2_crops."""
    n_total = sum(c.N for c in crops)
    d_total = sum(c.D for c in crops)
    nv = sum(c.n_valid for c in crops)
    f = {"N": n_total, "D": d_total, "n_pairs": N_PAIRS, "n_valid_cells": nv,
         "value": (n_total / d_total) if d_total else None,
         "rule": "below_0_03 = 100*N < 3*D; at_least_0_10 = 10*N >= D (integers; the float is reported only)"}
    br = branch(n_total, d_total)
    f["status"] = br["status"]
    if gated:
        f.update({k: br[k] for k in ("below_0_03", "at_least_0_10", "branch")})
    nl = sum(c.N_L for c in crops)
    dl = sum(c.D_L for c in crops)
    f_lesion = {"N": nl, "D": dl, "n_cells": sum(c.n_lesion for c in crops),
                "value": (nl / dl) if dl else "undefined",
                "cells": "the 8-draw mean probability at T = 1 peaks at a class in 1..115"}
    h = sum(c.H for c in crops)
    f_halves = {"H": h, "n_cells": nv, "value": (h / nv) if nv else None,
                "definition": "argmax of the mean probability (T = 1) of draws 1-4 vs draws 5-8"}
    kl_logit = {"value": (math.fsum(c.kl_logit_sum for c in crops) / (K_DRAWS * nv)) if nv else None,
                "T": T_KD, "units": "nats", "t_squared_factor": False,
                "pbar": "mean of the eight softmax(z_k / 4); log pbar = logsumexp_k - ln 8"}
    n_cwd = sum(c.n_cwd for c in crops)
    kl_cwd = {"value": (math.fsum(c.kl_cwd_sum for c in crops) / n_cwd) if n_cwd else None,
              "T": T_KD, "units": "nats", "n_distributions": n_cwd,
              "pooling": "unweighted mean over (crop with V non-empty, draw, channel)"}

    def pooled(key):
        return (math.fsum(c.d2[key] for c in crops) / nv) if nv else None

    n_high = sum(c.d2["n_max_prob_gt_0_99_t1"] for c in crops)
    d2 = {"draw": "first", "grid": "64x64 valid cells", "n_cells": nv,
          "mean_max_prob_t1": pooled("max_prob_t1_sum"), "mean_entropy_t1": pooled("entropy_t1_sum"),
          "share_max_prob_gt_0_99_t1": (n_high / nv) if nv else None, "n_max_prob_gt_0_99_t1": n_high,
          "mean_max_prob_t4": pooled("max_prob_t4_sum"), "mean_entropy_t4": pooled("entropy_t4_sum")}
    return {"F": f, "F_lesion": f_lesion, "F_halves": f_halves, "KL_logit": kl_logit, "KL_cwd": kl_cwd,
            "d2_crops": d2}


__all__ = ["K_DRAWS", "N_PAIRS", "T_KD", "BOOTSTRAP_B", "BOOTSTRAP_SEED", "CropStats", "log_softmax",
           "logsumexp", "pairwise_flips", "crop_statistics", "branch", "bootstrap_f", "summarize"]
