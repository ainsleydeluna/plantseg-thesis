"""Canonical identity alignment of two evaluation runs (A3a).

Implements docs/STATISTICAL_ANALYSIS_CONTRACT.md section 10. Alignment is by canonical image ID
only -- never by row position, and never before identity equality has been proven.

Forbidden and enforced: silent row dropping, set intersection, imputation of any kind, reordering
without ID verification, and pairwise-complete observations.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import EvaluationRun, EvaluationRunIdentity, Policy

METRIC_DISEASE_ONLY = "per_image_disease_only_miou"
METRIC_ALL_CLASS = "per_image_all_class_miou"
METRIC_MIOU_C = "per_image_miou_c"


class AlignmentError(RuntimeError):
    """Two runs cannot be paired. Always fatal."""


@dataclass(frozen=True)
class PairedVector:
    """Ordered paired observations, `delta = candidate - baseline` (positive favours candidate)."""
    image_ids: tuple[str, ...]
    baseline: np.ndarray
    candidate: np.ndarray
    delta: np.ndarray
    metric: str
    policy: Policy
    baseline_identity: EvaluationRunIdentity | None
    candidate_identity: EvaluationRunIdentity | None
    alignment_status: str = "ok"

    @property
    def n(self) -> int:
        return len(self.image_ids)


# --------------------------------------------------------------------------------------------------
# compatibility
# --------------------------------------------------------------------------------------------------
#: Fields that must be identical for any pairing at all.
_ALWAYS = ("schema_version", "metric_protocol", "split", "split_manifest_sha256",
           "class_map_sha256", "preprocess_protocol", "num_classes", "background_index",
           "ignore_index", "expected_rows", "actual_rows")
#: Additionally required for OFFICIAL analysis -- two stages scored by different metric code are
#: not comparable, so this is an ERROR (not a warning) per contract section 10.
_OFFICIAL_ONLY = ("metric_impl_sha256",)


def assert_compatible(a: EvaluationRunIdentity, b: EvaluationRunIdentity, policy: Policy,
                      *, same_condition: bool = True) -> list[str]:
    """Raise on incompatibility; return non-fatal warnings."""
    for f in _ALWAYS:
        va, vb = getattr(a, f), getattr(b, f)
        if va != vb:
            raise AlignmentError(f"incompatible {f}: {va!r} (baseline) vs {vb!r} (candidate)")
    if same_condition:
        for f in ("condition_type", "condition_name", "condition_severity"):
            if getattr(a, f) != getattr(b, f):
                raise AlignmentError(
                    f"incompatible {f}: {getattr(a, f)!r} vs {getattr(b, f)!r}")
    warnings: list[str] = []
    for f in _OFFICIAL_ONLY:
        if getattr(a, f) != getattr(b, f):
            msg = (f"{f} differs between runs ({getattr(a, f)[:12]}... vs "
                   f"{getattr(b, f)[:12]}...) -- the two stages were scored by different metric "
                   "code")
            if policy is Policy.OFFICIAL:
                raise AlignmentError("OFFICIAL analysis forbids this: " + msg)
            warnings.append(msg)
    return warnings


def _assert_identity_sets(base_ids, cand_ids) -> None:
    for label, ids in (("baseline", base_ids), ("candidate", cand_ids)):
        if len(set(ids)) != len(ids):
            raise AlignmentError(f"{label} contains duplicate image_id")
        if len({i.casefold() for i in ids}) != len(ids):
            raise AlignmentError(f"{label} contains a case-folded image_id collision")
    sb, sc = set(base_ids), set(cand_ids)
    if sb != sc:
        only_b, only_c = sorted(sb - sc), sorted(sc - sb)
        raise AlignmentError(
            f"image_id sets are not equal: {len(only_b)} baseline-only (e.g. {only_b[:3]}), "
            f"{len(only_c)} candidate-only (e.g. {only_c[:3]}). Intersection is forbidden.")


def align_runs(baseline: EvaluationRun, candidate: EvaluationRun, *, policy: Policy,
               metric: str = METRIC_DISEASE_ONLY) -> PairedVector:
    """Pair two runs by canonical image ID and return `candidate - baseline`."""
    if baseline.policy is not policy or candidate.policy is not policy:
        raise AlignmentError(
            f"runs were ingested under a different policy than requested ({policy})")
    warns = assert_compatible(baseline.identity, candidate.identity, policy)

    b_ids = [r.image_id for r in baseline.records]
    c_ids = [r.image_id for r in candidate.records]
    _assert_identity_sets(b_ids, c_ids)

    bmap, cmap = baseline.by_id(), candidate.by_id()
    order = sorted(b_ids)                       # one canonical identity order for both sides

    # clean_image_id mapping must agree image-for-image
    for i in order:
        if bmap[i].clean_image_id != cmap[i].clean_image_id:
            raise AlignmentError(
                f"clean_image_id mapping differs for {i!r}: {bmap[i].clean_image_id!r} vs "
                f"{cmap[i].clean_image_id!r}")

    field = {METRIC_DISEASE_ONLY: "disease_only_miou",
             METRIC_ALL_CLASS: "all_class_miou"}.get(metric)
    if field is None:
        raise AlignmentError(f"align_runs does not serve metric {metric!r}")

    bv, cv, undefined = [], [], []
    for i in order:
        b, c = getattr(bmap[i], field), getattr(cmap[i], field)
        if b is None or c is None:
            undefined.append(i)
            bv.append(np.nan)
            cv.append(np.nan)
        else:
            bv.append(float(b))
            cv.append(float(c))

    if undefined:
        if policy is Policy.OFFICIAL:
            raise AlignmentError(
                f"OFFICIAL analysis forbids undefined primary values; {len(undefined)} pair(s) "
                f"undefined, e.g. {undefined[:3]}. Rows are never dropped or imputed.")
        # NONOFFICIAL: report the frozen status, still without deleting the pair
        return PairedVector(tuple(order), np.asarray(bv, float), np.asarray(cv, float),
                            np.asarray(cv, float) - np.asarray(bv, float), metric, policy,
                            baseline.identity, candidate.identity,
                            alignment_status="nonofficial_undefined_pairs_present")

    base = np.asarray(bv, dtype=np.float64)
    cand = np.asarray(cv, dtype=np.float64)
    if not (np.isfinite(base).all() and np.isfinite(cand).all()):
        raise AlignmentError("non-finite value present in an aligned vector")
    status = "ok" if not warns else "ok_with_warnings"
    return PairedVector(tuple(order), base, cand, cand - base, metric, policy,
                        baseline.identity, candidate.identity, alignment_status=status)


def align_vectors(baseline_ids, baseline_values, candidate_ids, candidate_values, *,
                  policy: Policy, metric: str) -> PairedVector:
    """Pair two already-assembled per-image vectors (used for mIoU-C, which is derived)."""
    b_ids, c_ids = list(baseline_ids), list(candidate_ids)
    _assert_identity_sets(b_ids, c_ids)
    bmap = dict(zip(b_ids, np.asarray(baseline_values, dtype=np.float64)))
    cmap = dict(zip(c_ids, np.asarray(candidate_values, dtype=np.float64)))
    order = sorted(b_ids)
    base = np.asarray([bmap[i] for i in order], dtype=np.float64)
    cand = np.asarray([cmap[i] for i in order], dtype=np.float64)
    if not (np.isfinite(base).all() and np.isfinite(cand).all()):
        raise AlignmentError("non-finite value present in an aligned vector")
    return PairedVector(tuple(order), base, cand, cand - base, metric, policy, None, None)
