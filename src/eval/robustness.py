"""Corruption-robustness execution helpers: vocabulary gates, ordering contract, aggregation.

Nothing here re-implements clean metrics, the corruption vocabulary, or the artifact contract:

  * the frozen vocabulary/severity roles come from `src.stats.corruption_protocol`
    (`load_corruption_protocol`), imported unmodified — this module adds no second vocabulary;
  * per-image and dataset IoU come from the existing evaluator/metric reducers;
  * `actual_rows`, manifest integrity and official finalisation stay with `src/eval/artifacts.py`.

Contract math (IMPLEMENTATION_CONTRACT lines 338-341, 370):
  * mIoU-C  = mean over corruption TYPES of (that type's mIoU averaged over severities 1-3)
  * RPD     = (mIoU_clean - mIoU_C) / mIoU_clean * 100      -- descriptive only
  * rCD     = Kamann-style, descriptive only, E1 is the internal reference (rCD(E1) = 1),
              and the TEACHER is excluded
  * severity 4 is a descriptive degradation trajectory only -- never in mIoU-C, RPD, rCD or
    inference; severity 5 is never produced
  * the only inferential robustness pairing is E1 vs E6 on per-image mIoU-C

THE CORRUPTION IMPLEMENTATION IS NOT VENDORED. `configs/corruption_protocol.json` states plainly
that "the corruption implementation bytes, parameters, seed policy and cache protocol are NOT
vendored here -- that is a future corruption-scaffold task", and `imagecorruptions` is not part of
the pinned student stack. This module therefore defines the ORDERING CONTRACT and takes the
transform as an injected callable; the production path fails loudly instead of inventing one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.stats.corruption_protocol import (CorruptionProtocolError,  # noqa: E402
                                           load_corruption_protocol)

IGNORE_INDEX = 255
E1_REFERENCE_STAGE = "E1"
INFERENTIAL_ROBUSTNESS_PAIR = ("E1", "E6")     # the ONLY inferential robustness comparison
RCD_EXCLUDED_STAGES = ("TEACHER",)             # teacher excluded from rCD (contract line 341)


class RobustnessError(RuntimeError):
    """A rejected corruption/severity request or an invalid aggregation input."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str):
    raise RobustnessError(code, message)


# ------------------------------------------------------------------ vocabulary / severity gates
def protocol():
    """The frozen corruption protocol, loaded and validated by the existing stats authority."""
    try:
        return load_corruption_protocol()
    except CorruptionProtocolError as e:      # surfaced, never silently defaulted
        _fail("protocol_invalid", str(e))


def registered_corruptions() -> tuple[str, ...]:
    return tuple(e.id for e in protocol().entries)


def validate_condition(name: str, severity: int, *, official: bool) -> dict:
    """Validate a corruption/severity request and return the artifact `condition` payload.

    Severity roles follow the frozen protocol: 1-3 inferential, 4 descriptive-only, 5 never
    produced. An `official` artifact may carry 1-3 (inferential) or 4 (descriptive-only); 5 is
    refused outright for any status.
    """
    p = protocol()
    ids = tuple(e.id for e in p.entries)
    if name not in ids:
        _fail("unknown_corruption",
              f"corruption {name!r} is not in the frozen vocabulary {ids}. Aliases such as "
              "'brightness_variation' or 'motion-blur' are rejected.")
    if severity in p.excluded_severities:
        _fail("severity_excluded",
              f"severity {severity} is never produced by the official study pipeline")
    if severity not in tuple(p.inferential_severities) + tuple(p.descriptive_only_severities):
        _fail("severity_unknown", f"severity {severity} is not a registered severity")
    return {"type": "corruption", "name": name, "severity": int(severity),
            "role": ("inferential" if p.is_inferential_severity(severity) else "descriptive_only"),
            "official_allowed": bool(official)}


def is_primary_severity(severity: int) -> bool:
    """True only for severities that enter mIoU-C / RPD / rCD / inference (1-3)."""
    return protocol().is_inferential_severity(severity)


# ------------------------------------------------------------------ ordering contract
def apply_corruption(image_uint8: np.ndarray, transform: Callable[[np.ndarray], np.ndarray],
                     mask: np.ndarray | None = None):
    """Apply a corruption to a uint8 RGB image BEFORE normalization, leaving the mask untouched.

    The registered acquisition-order pipeline corrupts the raw uint8 RGB representation; running an
    `imagecorruptions`-style function on a normalized float tensor is invalid. This helper enforces
    the input contract and returns `(corrupted_uint8, mask)` with the mask passed through by
    identity — corruption never edits ground truth, and ignore/padding label 255 is untouched.
    """
    if not isinstance(image_uint8, np.ndarray) or image_uint8.dtype != np.uint8:
        _fail("corruption_input_not_uint8",
              f"corruption operates on uint8 RGB before normalization, got "
              f"{getattr(image_uint8, 'dtype', type(image_uint8).__name__)}")
    if image_uint8.ndim != 3 or image_uint8.shape[-1] != 3:
        _fail("corruption_input_not_rgb", f"expected HxWx3 RGB, got {image_uint8.shape}")
    out = transform(image_uint8)
    if not isinstance(out, np.ndarray) or out.dtype != np.uint8 or out.shape != image_uint8.shape:
        _fail("corruption_output_invalid",
              "a corruption must return a uint8 array of the same shape as its input")
    return out, mask


def real_corruption_transform(name: str, severity: int):
    """The production corruption function — NOT vendored yet, so this fails loudly.

    `configs/corruption_protocol.json` records that the implementation bytes, parameters, seed
    policy and cache protocol are a separate future corruption-scaffold task, and the pinned student
    stack does not ship `imagecorruptions`. Inventing an implementation here would silently create
    an unregistered corruption definition.
    """
    validate_condition(name, severity, official=True)
    _fail("corruption_implementation_not_vendored",
          f"the {name!r} severity-{severity} implementation is not vendored: "
          "configs/corruption_protocol.json defines the VOCABULARY and severity roles only, and "
          "states the implementation bytes/parameters/seed policy remain a future corruption-"
          "scaffold task. Supply a validated transform explicitly rather than improvising one.")


# ------------------------------------------------------------------ aggregation
def corruption_type_miou(per_severity: Mapping[int, float]) -> float:
    """A single corruption type's mIoU: the mean over severities 1-3 ONLY."""
    used = {s: v for s, v in per_severity.items() if is_primary_severity(s)}
    if not used:
        _fail("no_primary_severities",
              "a corruption type contributes to mIoU-C only through severities 1-3")
    return float(np.mean([used[s] for s in sorted(used)]))


def miou_c(per_corruption: Mapping[str, Mapping[int, float]]) -> float:
    """PRIMARY mIoU-C: equal-weight mean across corruption TYPES, each averaged over severities 1-3.

    Equal weighting is per type, not per (type, severity) row, so a type with a missing severity
    cannot silently gain or lose influence — every registered type must be present.
    """
    registered = registered_corruptions()
    missing = [c for c in registered if c not in per_corruption]
    if missing:
        _fail("incomplete_corruption_set",
              f"mIoU-C requires every registered corruption type; missing {missing}")
    extra = [c for c in per_corruption if c not in registered]
    if extra:
        _fail("unknown_corruption", f"unregistered corruption types: {extra}")
    return float(np.mean([corruption_type_miou(per_corruption[c]) for c in registered]))


def rpd(miou_clean: float, miou_corrupted: float) -> float:
    """RPD = (mIoU_clean - mIoU_C) / mIoU_clean * 100. Descriptive only."""
    if not np.isfinite(miou_clean) or miou_clean <= 0:
        _fail("rpd_invalid_clean", f"RPD needs a positive finite clean mIoU, got {miou_clean!r}")
    return float((miou_clean - miou_corrupted) / miou_clean * 100.0)


def rcd(stage: str, stage_degradation: float, e1_degradation: float) -> float:
    """rCD relative to the E1 internal reference (rCD(E1) = 1). Descriptive only; teacher excluded."""
    key = str(stage).upper()
    if key in RCD_EXCLUDED_STAGES:
        _fail("rcd_stage_excluded",
              f"{key} is excluded from rCD by the contract (teacher is a descriptive reference)")
    if not np.isfinite(e1_degradation) or e1_degradation <= 0:
        _fail("rcd_invalid_reference",
              f"rCD needs a positive finite E1 reference degradation, got {e1_degradation!r}")
    return float(stage_degradation / e1_degradation)


def inferential_pair_error(stage_a: str, stage_b: str) -> str | None:
    """Refuse any robustness inferential pairing other than E1 vs E6."""
    pair = tuple(sorted((str(stage_a).upper(), str(stage_b).upper())))
    if pair != tuple(sorted(INFERENTIAL_ROBUSTNESS_PAIR)):
        return (f"the only inferential robustness comparison is "
                f"{INFERENTIAL_ROBUSTNESS_PAIR[0]} vs {INFERENTIAL_ROBUSTNESS_PAIR[1]}; "
                f"{stage_a} vs {stage_b} is descriptive only")
    return None
