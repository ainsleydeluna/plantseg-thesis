"""LREP's descriptive bootstrap namespace, outside protocol 1.0.0 (lane LREP, B-5: ruling B5-D4).

STATS §8.7.1 (:451-463) freezes exactly two analysis ids and ten comparison ids, and the frozen seed
derivation refuses any other (bootstrap.canonical_seed_bytes, bootstrap.py:174-184). B-5's descriptive
intervals -- AM-18 item 3(d)'s per-image mean ΔmIoU (B5-D2) and AM-18 item 8(c)'s duplicate-free
intervals -- therefore run in a namespace of their own:

  * analysis_id `lrep_descriptive_v1`, outside protocol 1.0.0;
  * seed bytes b"plantseg-lrep-descriptive/1.0.0\\x00" + b"root_seed=42\\x00" + analysis_id + b"\\x00" +
    contrast_id + b"\\x00" + statistic_name: bootstrap.py:180-184's layout with the first field replaced,
    and no trailing NUL; then SHA-256, the first 16 bytes as a big-endian integer, and a
    bootstrap.SeedMaterial (:150-160);
  * contrast ids of lowercase ASCII letters, digits and "_", with "__" as the separator:
    `arm__<x>__<stage>` (x r1, r2, flip; stage e3, e6, e7), `dedup__<official comparison_id>` and
    `quant__<stage>` (e6_e2, e7_e2; reserved: no interval uses it while AM-23's rows have none). They never
    pass through bootstrap's validators, and no `<contrast_id>__<statistic>` key is formed: the statistic is
    a separate field, because STATS :447-449 reserves "__";
  * bootstrap.make_generator, draw_replicate_indices and bca_interval run unchanged, in `run_task`, which
    mirrors run_bootstrap_task (:350-376) line for line; run_bootstrap_task itself cannot take the id,
    because its derive_seed call (:360) refuses it.

No stream collides with an official one: every official seed input starts with b"plantseg-stats/1.0.0\\x00"
(bootstrap.py:180) and every one made here with SEED_PREFIX, so equal digests need a SHA-256 collision;
the analysis id is not in bootstrap.ANALYSIS_IDS, so no official artifact can carry it; and every contrast
id holds "__", which none of the ten official ids does. The last two are assertions (like R-15), not
refusal codes.

Descriptive: no test, decision or label reads an interval made here. Import-time behaviour is side-effect
free.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from . import bootstrap as BS

ANALYSIS_ID = "lrep_descriptive_v1"
SEED_PREFIX = b"plantseg-lrep-descriptive/1.0.0\x00"
BANNER = "descriptive; outside protocol 1.0.0"
ARM_XS = ("r1", "r2", "flip")
ARM_STAGES = ("e3", "e6", "e7")
QUANT_STAGES = ("e6_e2", "e7_e2")
_PART = r"[a-z0-9]+(?:_[a-z0-9]+)*"
#: Lowercase ASCII letters, digits and single "_" inside a part; "__" only between parts, at least once.
CONTRAST_ID_RE = re.compile(rf"^{_PART}(?:__{_PART})+$")


def check_contrast_id(contrast_id) -> str:
    """The no-collision assertion (B5-D4, like R-15): the id's form, at least one "__", never one of the
    ten official comparison ids, and an analysis id outside bootstrap.ANALYSIS_IDS."""
    if (not isinstance(contrast_id, str) or not CONTRAST_ID_RE.fullmatch(contrast_id)
            or contrast_id in BS.ALL_COMPARISON_IDS or ANALYSIS_ID in BS.ANALYSIS_IDS):
        raise AssertionError(f"{contrast_id!r} is not an LREP descriptive contrast id (B5-D4)")
    return contrast_id


def arm_contrast_id(x: str, stage: str) -> str:
    """`arm__<x>__<stage>`: AM-18 item 3(d)'s contrast of arm x at stage E3, E6 or E7."""
    x, stage = x.lower(), stage.lower()
    if x not in ARM_XS or stage not in ARM_STAGES:
        raise AssertionError(f"no arm contrast for x {x!r} at stage {stage!r} (B5-D4)")
    return check_contrast_id(f"arm__{x}__{stage}")


def dedup_contrast_id(comparison_id: str) -> str:
    """`dedup__<official comparison_id>`: AM-18 item 8(c)'s duplicate-free recomputation of it."""
    if comparison_id not in BS.ALL_COMPARISON_IDS:
        raise AssertionError(f"{comparison_id!r} is not an official comparison id")
    return check_contrast_id(f"dedup__{comparison_id}")


def quant_contrast_id(stage: str) -> str:
    """`quant__<stage>` (e6_e2, e7_e2): reserved for AM-23's rows, unused while they carry no interval."""
    key = stage.lower().replace("-", "_")
    if key not in QUANT_STAGES:
        raise AssertionError(f"no quant contrast for stage {stage!r} (B5-D4)")
    return check_contrast_id(f"quant__{key}")


def seed_bytes(contrast_id: str, statistic_name: str) -> bytes:
    """bootstrap.py:180-184's layout with the first field replaced; no trailing NUL."""
    check_contrast_id(contrast_id)
    if statistic_name not in BS.STATISTIC_NAMES:
        raise AssertionError(f"statistic {statistic_name!r} is not one of {BS.STATISTIC_NAMES}")
    return (SEED_PREFIX
            + b"root_seed=" + str(BS.ROOT_SEED).encode("ascii") + b"\x00"
            + ANALYSIS_ID.encode("utf-8") + b"\x00"
            + contrast_id.encode("utf-8") + b"\x00"
            + statistic_name.encode("utf-8"))


def seed_material(contrast_id: str, statistic_name: str) -> BS.SeedMaterial:
    """bootstrap.derive_seed's procedure (bootstrap.py:187-198) on this namespace's bytes."""
    raw = seed_bytes(contrast_id, statistic_name)
    digest = hashlib.sha256(raw).digest()
    entropy = digest[:16]
    display = (f"plantseg-lrep-descriptive/1.0.0\\0root_seed={BS.ROOT_SEED}\\0{ANALYSIS_ID}"
               f"\\0{contrast_id}\\0{statistic_name}")
    return BS.SeedMaterial(
        root_seed=BS.ROOT_SEED, canonical_bytes=raw, seed_input_display=display,
        seed_input_hex=raw.hex(), seed_sha256=digest.hex(), entropy_bytes=entropy,
        entropy_int=int.from_bytes(entropy, "big", signed=False),
        bit_generator="PCG64", numpy_version=np.__version__)


@dataclass(frozen=True)
class DescriptiveTaskResult:
    contrast_id: str
    statistic_name: str
    interval_type: str
    seed: BS.SeedMaterial
    observed: float
    replicates: np.ndarray
    jackknife: np.ndarray
    bca: BS.BcaOutcome


def run_task(seed: BS.SeedMaterial, interval_type: str, n_images: int, B: int, observed: float,
             replicate_fn, jackknife_fn) -> tuple[np.ndarray, np.ndarray, BS.BcaOutcome]:
    """bootstrap.run_bootstrap_task (bootstrap.py:350-376) line for line, from a given seed: row-wise
    draws from a task-local Generator, then the leave-one-image-out jackknife, then bca_interval."""
    if n_images < 1:
        raise BS.BootstrapError("n_images must be >= 1")
    if B < 1:
        raise BS.BootstrapError("B must be >= 1")
    gen = BS.make_generator(seed)

    reps = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = BS.draw_replicate_indices(gen, n_images)
        reps[b] = replicate_fn(idx)
    jack = np.empty(n_images, dtype=np.float64)
    for i in range(n_images):
        jack[i] = jackknife_fn(i)

    bca = BS.bca_interval(reps, jack, observed, interval_type)
    return reps, jack, bca


def descriptive_task(contrast_id: str, statistic_name: str, interval_type: str, n_images: int, B: int,
                     observed: float, replicate_fn, jackknife_fn) -> DescriptiveTaskResult:
    """One interval in this namespace (B5-D4)."""
    seed = seed_material(contrast_id, statistic_name)
    reps, jack, bca = run_task(seed, interval_type, n_images, B, observed, replicate_fn, jackknife_fn)
    return DescriptiveTaskResult(contrast_id=contrast_id, statistic_name=statistic_name,
                                 interval_type=interval_type, seed=seed, observed=float(observed),
                                 replicates=reps, jackknife=jack, bca=bca)


def mean_delta_task(contrast_id: str, delta, *, B: int = BS.PRODUCTION_B) -> DescriptiveTaskResult:
    """B5-D2: the per-image mean Δ with a two-sided BCa 95% interval. The observed value is the A3a
    primitive's mean_delta, as bootstrap.descriptive_scalars computes it (bootstrap.py:433-444); the
    replicates are bootstrap.scalar_callables' (:390-400). B is PRODUCTION_B in the official profile; the
    smokes pass a small B."""
    from .tests import engineering_shifts                    # LRB19 L-14: the A3a primitive
    d = np.asarray(delta, dtype=np.float64)
    observed = float(engineering_shifts(d).mean_delta)
    rf, jf = BS.scalar_callables(BS.MEAN_DELTA, d)
    return descriptive_task(contrast_id, BS.MEAN_DELTA, BS.TWO_SIDED, int(d.shape[0]), B, observed, rf, jf)


def task_json(r: DescriptiveTaskResult) -> dict:
    """The interval as JSON: the contrast id and the statistic are separate fields (STATS :447-449)."""
    two = r.interval_type == BS.TWO_SIDED
    return {"contrast_id": r.contrast_id, "statistic_name": r.statistic_name, "analysis_id": ANALYSIS_ID,
            "interval_type": r.interval_type, "observed": r.observed, "B": int(r.replicates.size),
            "n_images": int(r.jackknife.size), "lower_bound": r.bca.bounds[0],
            "upper_bound": r.bca.bounds[1] if two else None, "interval_method": r.bca.interval_method,
            "confidence_level": r.bca.confidence_level, "z0": r.bca.z0,
            "acceleration": r.bca.acceleration, "warnings": list(r.bca.warnings), "status": r.bca.status,
            "seed_input_hex": r.seed.seed_input_hex, "seed_sha256": r.seed.seed_sha256, "banner": BANNER}


__all__ = ["ANALYSIS_ID", "SEED_PREFIX", "BANNER", "ARM_XS", "ARM_STAGES", "QUANT_STAGES", "CONTRAST_ID_RE",
           "check_contrast_id", "arm_contrast_id", "dedup_contrast_id", "quant_contrast_id", "seed_bytes",
           "seed_material", "DescriptiveTaskResult", "run_task", "descriptive_task", "mean_delta_task",
           "task_json"]
