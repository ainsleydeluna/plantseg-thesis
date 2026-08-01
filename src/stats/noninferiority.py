"""Pooled dataset-mIoU estimand, E3->E6 non-inferiority and the E6-KD trigger (A3b).

Implements STATISTICAL_ANALYSIS_CONTRACT.md sections 9.1-9.3 and 12.4.9-12.4.10: int64
sufficient-statistic re-accumulation with duplicate multiplicity preserved, union-present
eligibility applied separately per stage, the one-sided BCa non-inferiority decision with its
strict boundary, the four sensitivity margins read from one distribution, and the strict E6-KD
observed-drop trigger.

Averaging per-image mIoU scalars is never used for this estimand: it is a mean-of-ratios, whereas
the frozen dataset-level metric is a ratio-of-sums.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import BootstrapError, CONFIDENCE_LEVEL, ONE_SIDED_LOWER

#: Contract section 9.2 / 9.3 -- both boundaries are STRICT.
PRIMARY_MARGIN = 0.020
SENSITIVITY_MARGINS = (0.010, 0.015, 0.020, 0.025)
E6KD_TRIGGER_THRESHOLD = 0.010
E6KD_DECISION_RULE = "strictly_greater"


# --------------------------------------------------------------------------------------------------
# 1. densification -- owned int64 arrays, never a mutated ingested array
# --------------------------------------------------------------------------------------------------
def densify(stats, n_images: int, num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse (image_index, class_id, tp/gt/pred) -> owned int64 `[n_images, num_classes]`.

    `src/stats/ingest.py` hands back read-only arrays; every array built here is a fresh owned
    copy, and the ingested arrays are never written to.
    """
    out = []
    for name in ("tp", "gt", "pred"):
        dense = np.zeros((n_images, num_classes), dtype=np.int64)
        np.add.at(dense, (np.asarray(stats.image_index), np.asarray(stats.class_id)),
                  np.asarray(getattr(stats, name), dtype=np.int64))
        if dense.dtype != np.int64:                                  # pragma: no cover - guard
            raise BootstrapError("dense accumulation must remain int64")
        out.append(dense)
    return out[0], out[1], out[2]


def pooled_miou_from_totals(tp: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> float:
    """All-class union-present pooled mIoU. Zero eligible classes is FATAL in every mode."""
    tp = np.asarray(tp, dtype=np.int64)
    union = np.asarray(gt, dtype=np.int64) + np.asarray(pred, dtype=np.int64) - tp
    if np.any(union < 0):
        raise BootstrapError("negative union: malformed sufficient statistics")
    eligible = union > 0
    n_eligible = int(np.count_nonzero(eligible))
    if n_eligible == 0:
        raise BootstrapError(
            "pooled dataset mIoU is undefined: zero union-present eligible classes. This is a "
            "mathematical integrity failure in EVERY policy mode -- no placeholder, NaN, null or "
            "empty mean may be returned.")
    return float(np.mean(tp[eligible].astype(np.float64) / union[eligible].astype(np.float64)))


# --------------------------------------------------------------------------------------------------
# 2. the paired pooled estimand
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PooledStages:
    """Two stages densified against one canonical image order."""
    n_images: int
    num_classes: int
    base_tp: np.ndarray
    base_gt: np.ndarray
    base_pred: np.ndarray
    cand_tp: np.ndarray
    cand_gt: np.ndarray
    cand_pred: np.ndarray

    # -- construction ---------------------------------------------------------------------------
    @staticmethod
    def from_dense(base: tuple[np.ndarray, np.ndarray, np.ndarray],
                   cand: tuple[np.ndarray, np.ndarray, np.ndarray]) -> "PooledStages":
        if base[0].shape != cand[0].shape:
            raise BootstrapError(f"stage shape mismatch: {base[0].shape} vs {cand[0].shape}")
        n, c = base[0].shape
        return PooledStages(n, c, base[0], base[1], base[2], cand[0], cand[1], cand[2])

    @staticmethod
    def from_runs(baseline_run, candidate_run) -> "PooledStages":
        nb = len(baseline_run.stats.manifest_ids)
        nc = len(candidate_run.stats.manifest_ids)
        if nb != nc:
            raise BootstrapError(f"stage row-count mismatch: {nb} vs {nc}")
        if tuple(baseline_run.stats.manifest_ids) != tuple(candidate_run.stats.manifest_ids):
            raise BootstrapError("stages are not aligned on the same canonical image order")
        cls = baseline_run.identity.num_classes
        return PooledStages.from_dense(densify(baseline_run.stats, nb, cls),
                                       densify(candidate_run.stats, nc, cls))

    # -- estimands ------------------------------------------------------------------------------
    def _delta_from_weights(self, counts: np.ndarray) -> float:
        w = np.asarray(counts, dtype=np.int64)
        b = pooled_miou_from_totals(w @ self.base_tp, w @ self.base_gt, w @ self.base_pred)
        c = pooled_miou_from_totals(w @ self.cand_tp, w @ self.cand_gt, w @ self.cand_pred)
        return c - b

    def stage_mious(self) -> tuple[float, float]:
        """(baseline, candidate) pooled mIoU at unit weights."""
        w = np.ones(self.n_images, dtype=np.int64)
        return (pooled_miou_from_totals(w @ self.base_tp, w @ self.base_gt, w @ self.base_pred),
                pooled_miou_from_totals(w @ self.cand_tp, w @ self.cand_gt, w @ self.cand_pred))

    def observed(self) -> float:
        b, c = self.stage_mious()
        return c - b

    def replicate(self, idx: np.ndarray) -> float:
        """One replicate: duplicate multiplicity preserved exactly via integer counts."""
        counts = np.bincount(np.asarray(idx), minlength=self.n_images).astype(np.int64)
        if counts.size != self.n_images:                             # pragma: no cover - guard
            raise BootstrapError("sampled index out of range")
        return self._delta_from_weights(counts)

    def jackknife(self, i: int) -> float:
        """Leave-one-IMAGE-out by re-accumulation -- never by deleting a per-image scalar."""
        counts = np.ones(self.n_images, dtype=np.int64)
        counts[i] = 0
        return self._delta_from_weights(counts)


# --------------------------------------------------------------------------------------------------
# 3. non-inferiority and E6-KD
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SensitivityDecision:
    margin: float
    passed: bool


@dataclass(frozen=True)
class NonInferiorityResult:
    comparison_id: str
    baseline_stage: str
    candidate_stage: str
    direction: str
    baseline_dataset_miou: float
    candidate_dataset_miou: float
    observed_delta: float
    observed_drop: float
    relative_retention: float
    interval_type: str
    confidence_level: float
    lower_bound: float
    primary_margin: float
    passed: bool
    sensitivity: tuple[SensitivityDecision, ...]


@dataclass(frozen=True)
class E6KDResult:
    baseline_stage: str
    candidate_stage: str
    observed_drop: float
    trigger_threshold: float
    decision_rule: str
    triggered: bool


def relative_retention(baseline_dataset_miou: float, candidate_dataset_miou: float) -> float:
    """candidate / baseline. A zero or negative denominator is FATAL in every policy mode."""
    b = float(baseline_dataset_miou)
    c = float(candidate_dataset_miou)
    if not np.isfinite(b) or b <= 0.0:
        raise BootstrapError(
            f"relative_retention denominator must be finite and strictly positive, got {b!r}")
    if not np.isfinite(c):
        raise BootstrapError(f"candidate_dataset_miou must be finite, got {c!r}")
    r = c / b
    if not np.isfinite(r):                                           # pragma: no cover - guard
        raise BootstrapError("relative_retention is non-finite")
    return r


def build_non_inferiority(baseline_dataset_miou: float, candidate_dataset_miou: float,
                          lower_bound: float) -> NonInferiorityResult:
    """E6 - E3, strict `lower_bound > -primary_margin`; all four margins reuse ONE bound."""
    b = float(baseline_dataset_miou)
    c = float(candidate_dataset_miou)
    if not np.isfinite(lower_bound):
        raise BootstrapError("non-inferiority lower bound must be finite")
    observed_delta = c - b
    observed_drop = b - c
    sens = tuple(SensitivityDecision(float(m), bool(lower_bound > -float(m)))
                 for m in SENSITIVITY_MARGINS)
    return NonInferiorityResult(
        comparison_id="noninferiority_e3_e6", baseline_stage="E3", candidate_stage="E6",
        direction="candidate_minus_baseline", baseline_dataset_miou=b,
        candidate_dataset_miou=c, observed_delta=observed_delta, observed_drop=observed_drop,
        relative_retention=relative_retention(b, c), interval_type=ONE_SIDED_LOWER,
        confidence_level=CONFIDENCE_LEVEL, lower_bound=float(lower_bound),
        primary_margin=PRIMARY_MARGIN, passed=bool(lower_bound > -PRIMARY_MARGIN),
        sensitivity=sens)


def build_e6_kd(baseline_dataset_miou: float, candidate_dataset_miou: float) -> E6KDResult:
    """Observed clean drop E3 - E6; triggers only when strictly greater than 0.010."""
    drop = float(baseline_dataset_miou) - float(candidate_dataset_miou)
    if not np.isfinite(drop):
        raise BootstrapError("E6-KD observed drop must be finite")
    return E6KDResult(baseline_stage="E3", candidate_stage="E6", observed_drop=drop,
                      trigger_threshold=E6KD_TRIGGER_THRESHOLD,
                      decision_rule=E6KD_DECISION_RULE,
                      triggered=bool(drop > E6KD_TRIGGER_THRESHOLD))


__all__ = [
    "PRIMARY_MARGIN", "SENSITIVITY_MARGINS", "E6KD_TRIGGER_THRESHOLD", "E6KD_DECISION_RULE",
    "densify", "pooled_miou_from_totals", "PooledStages", "SensitivityDecision",
    "NonInferiorityResult", "E6KDResult", "relative_retention", "build_non_inferiority",
    "build_e6_kd",
]
