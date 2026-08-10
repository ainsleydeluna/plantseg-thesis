"""Per-image mIoU-C assembly from a complete corruption grid (A3a).

Implements docs/STATISTICAL_ANALYSIS_CONTRACT.md section 2: the NESTED rule
  per-image disease-only mIoU per (corruption, severity)
    -> mean over severities 1-3 within each corruption
    -> equal-weight mean of the five corruption means

The nested form is retained deliberately even though it equals a flat 15-value mean on a complete
grid, so a partial grid can never be averaged away silently.

============================================================================================
VOCABULARY OWNERSHIP -- the former DOCUMENTATION GAP is CLOSED; this module stays agnostic
============================================================================================
A3a originally recorded that no document froze a machine-readable corruption identifier, so this
module refused to invent one. That gap was closed by A3b-0 on 2026-07-28: the identifiers and the
severity roles are now frozen machine-readably in `configs/corruption_protocol.json`
(`plantseg-corruptions/1.0.0`), and the decision is recorded in
docs/STATISTICAL_ANALYSIS_CONTRACT.md section 2.1, docs/EVALUATION_CONTRACT.md section 5.1.1,
docs/IMPLEMENTATION_CONTRACT.md (Robustness) and docs/open_questions.md D20. Official mIoU-C is
no longer blocked by vocabulary ambiguity.

This module is nevertheless left deliberately vocabulary-agnostic: it still takes a REQUIRED
injected `CorruptionGrid` with NO default and hard-codes no canonical slug. Loading, validating
and freezing the official vocabulary belongs to the corruption-protocol layer
(`src/stats/corruption_protocol.py`), which is the single place the identifiers are asserted;
official callers build their grid there, while NONOFFICIAL_SMOKE fixtures may inject an explicitly
synthetic grid that can never enter official statistics.
============================================================================================

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .align import METRIC_MIOU_C, PairedVector, align_vectors
from .ingest import EvaluationRun, Policy

#: Frozen by ch3: severities 1-3 are inferential; 4 is descriptive-only; 5 is excluded.
INFERENTIAL_SEVERITIES = (1, 2, 3)
N_CORRUPTIONS = 5
N_CELLS = N_CORRUPTIONS * len(INFERENTIAL_SEVERITIES)   # 15


class RobustnessError(RuntimeError):
    """Corruption grid is unusable. Always fatal."""


@dataclass(frozen=True)
class CorruptionGrid:
    """The five machine-readable corruption identifiers, supplied by the caller.

    There is deliberately NO default: see the module docstring. `display_names` is optional and
    used only for reporting.
    """
    names: tuple[str, ...]
    display_names: Mapping[str, str] | None = None
    provenance: str = "caller-supplied (vocabulary not frozen in the statistical contract)"

    def __post_init__(self) -> None:
        if len(self.names) != N_CORRUPTIONS:
            raise RobustnessError(
                f"a corruption grid needs exactly {N_CORRUPTIONS} identifiers, got "
                f"{len(self.names)}")
        if len(set(self.names)) != len(self.names):
            raise RobustnessError(f"duplicate corruption identifier in {self.names}")
        for n in self.names:
            if not isinstance(n, str) or not n:
                raise RobustnessError(f"corruption identifier must be a non-empty string: {n!r}")


@dataclass(frozen=True)
class MiouCVector:
    clean_image_ids: tuple[str, ...]
    values: np.ndarray                                   # final per-image mIoU-C
    corruption_means: Mapping[str, np.ndarray]           # per corruption, audit intermediate
    grid: CorruptionGrid
    stage: str
    source_run_ids: tuple[str, ...]
    policy: Policy


# --------------------------------------------------------------------------------------------------
def assemble_miou_c(runs: Sequence[EvaluationRun], grid: CorruptionGrid, *, policy: Policy,
                    stage: str | None = None) -> MiouCVector:
    """Build the per-image mIoU-C vector for ONE stage from exactly 15 verified condition runs."""
    if len(runs) != N_CELLS:
        raise RobustnessError(
            f"expected exactly {N_CELLS} corruption runs (5 corruptions x severities "
            f"{INFERENTIAL_SEVERITIES}), got {len(runs)}")

    expected_cells = {(c, s) for c in grid.names for s in INFERENTIAL_SEVERITIES}
    seen: dict[tuple[str, int], EvaluationRun] = {}
    for r in runs:
        idn = r.identity
        if idn.condition_type != "corruption":
            raise RobustnessError(
                f"run {idn.run_id!r} has condition type {idn.condition_type!r}; a clean artifact "
                "may not appear in the corruption grid")
        name, sev = idn.condition_name, idn.condition_severity
        if name not in grid.names:
            raise RobustnessError(
                f"unexpected corruption {name!r}; the declared grid is {list(grid.names)}")
        if sev not in INFERENTIAL_SEVERITIES:
            raise RobustnessError(
                f"unexpected severity {sev!r} for {name!r}; only {INFERENTIAL_SEVERITIES} are "
                "inferential (severity 4 is descriptive-only, severity 5 is excluded)")
        if (name, sev) in seen:
            raise RobustnessError(f"duplicate condition cell ({name!r}, {sev})")
        seen[(name, sev)] = r

    missing = expected_cells - set(seen)
    if missing:
        raise RobustnessError(
            f"{len(missing)} missing condition cell(s), e.g. {sorted(missing)[:3]}; "
            "available-case averaging is forbidden")

    # ---- cross-artifact compatibility + identical clean-image identity ----
    ref = next(iter(seen.values())).identity
    for (name, sev), r in seen.items():
        i = r.identity
        for f in ("schema_version", "metric_protocol", "split", "split_manifest_sha256",
                  "class_map_sha256", "preprocess_protocol", "num_classes",
                  "background_index", "ignore_index"):
            if getattr(i, f) != getattr(ref, f):
                raise RobustnessError(
                    f"cell ({name!r},{sev}) has incompatible {f}: {getattr(i, f)!r} != "
                    f"{getattr(ref, f)!r}")
        if stage is not None and i.stage != stage:
            raise RobustnessError(f"cell ({name!r},{sev}) is stage {i.stage!r}, expected {stage!r}")

    ref_clean = tuple(sorted(r.clean_image_id for r in next(iter(seen.values())).records))
    per_cell: dict[tuple[str, int], dict[str, float]] = {}
    for (name, sev), r in seen.items():
        clean_ids = [rec.clean_image_id for rec in r.records]
        if len(set(clean_ids)) != len(clean_ids):
            raise RobustnessError(f"cell ({name!r},{sev}) has duplicate clean_image_id")
        if tuple(sorted(clean_ids)) != ref_clean:
            raise RobustnessError(
                f"cell ({name!r},{sev}) has a different clean-image ID set than the grid")
        cell: dict[str, float] = {}
        for rec in r.records:
            if rec.disease_only_miou is None:
                raise RobustnessError(
                    f"cell ({name!r},{sev}) has an undefined disease-only score for "
                    f"{rec.clean_image_id!r}; official mIoU-C forbids available-case averaging")
            cell[rec.clean_image_id] = float(rec.disease_only_miou)
        per_cell[(name, sev)] = cell

    # ---- nested aggregation ----
    order = ref_clean
    corruption_means: dict[str, np.ndarray] = {}
    for name in grid.names:
        sev_stack = np.stack(
            [np.asarray([per_cell[(name, s)][cid] for cid in order], dtype=np.float64)
             for s in INFERENTIAL_SEVERITIES])
        corruption_means[name] = sev_stack.mean(axis=0)          # step 2: mean over severities
    values = np.stack([corruption_means[n] for n in grid.names]).mean(axis=0)  # step 3

    if not np.isfinite(values).all():
        raise RobustnessError("non-finite value produced during mIoU-C assembly")

    return MiouCVector(
        clean_image_ids=order, values=values,
        corruption_means={k: v.copy() for k, v in corruption_means.items()},
        grid=grid, stage=stage or ref.stage,
        source_run_ids=tuple(sorted(r.identity.run_id for r in seen.values())), policy=policy)


def align_miou_c(baseline: MiouCVector, candidate: MiouCVector, *, policy: Policy) -> PairedVector:
    """Pair two assembled mIoU-C vectors by clean image ID; returns `candidate - baseline`."""
    if baseline.grid.names != candidate.grid.names:
        raise RobustnessError("the two stages were assembled over different corruption grids")
    return align_vectors(baseline.clean_image_ids, baseline.values,
                         candidate.clean_image_ids, candidate.values,
                         policy=policy, metric=METRIC_MIOU_C)
