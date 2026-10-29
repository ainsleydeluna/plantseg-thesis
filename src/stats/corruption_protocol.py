"""Loader for the frozen machine-readable corruption vocabulary and severity roles (A3b-0).

`configs/corruption_protocol.json` is the ONE cross-cutting source of truth shared by corruption
generation, evaluation condition metadata, mIoU-C assembly, artifact naming, cache manifests,
and statistical analysis.

This module is the only stats-side bridge from that JSON to the A3a `CorruptionGrid`.
`src/stats/robustness.py` is deliberately left unchanged: it still requires an injected grid and
holds no vocabulary of its own, so there is exactly one place where the identifiers are asserted.

Schema `plantseg-corruptions/2.0.0` (L-AM16-SEV) gives each corruption its own severity roles,
following AM-16 item 6 (docs/PREREGISTRATION_AMENDMENTS.md :303-306): severities 1-3 are
inferential for all five corruptions, severity 4 is descriptive_only for all five, and severity 5
is descriptive_only for the four non-noise corruptions and excluded for gaussian_noise. The
document is parsed by the torch-free `src.corruption_roles`, which holds the anchors and which the
cache generator reads too; this module adds the agreement with the frozen mIoU-C rule
(`src/stats/robustness.py`) and the stats-side objects. The 15-cell inferential grid is unchanged.

Identifier basis (A3b-0 `[project-decision]`): the vendored `imagecorruptions` reference
function names that Chapter III selects. The machine identifier and the thesis display label are
distinct -- `brightness` is the official identifier; "brightness variation" is explanatory prose
and is NOT an identifier. Aliases (`brightness_variation`, `motion-blur`, `Motion_Blur`, spaced or
capitalised variants) are rejected outright; no official corruption artifact exists yet, so no
migration compatibility is owed.

NOT frozen here: the corruption implementation bytes, its parameters and the seed policy (vendored
in `src/vendor/imagecorruptions`), and the cache protocol (`src/corruption_cache.py`).

Import-time behaviour is side-effect free: no file is read until a loader function is called.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .. import corruption_roles as _roles
from ..corruption_roles import CANONICAL_IDS, Cell, CorruptionRolesError  # noqa: F401  re-exported
from .robustness import CorruptionGrid, INFERENTIAL_SEVERITIES, N_CORRUPTIONS

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "configs" / "corruption_protocol.json"

# --- anchoring constants: the single asserted copy lives in src/corruption_roles.py -------------
EXPECTED_SCHEMA = _roles.SCHEMA_VERSION
EXPECTED_IDENTIFIER_BASIS = _roles.IDENTIFIER_BASIS
EXPECTED_INFERENTIAL = _roles.INFERENTIAL_SEVERITIES


class CorruptionProtocolError(RuntimeError):
    """The protocol file does not match the frozen vocabulary. Always fatal."""


@dataclass(frozen=True)
class CorruptionEntry:
    id: str
    display_name: str
    reference_function: str


@dataclass(frozen=True)
class CorruptionProtocol:
    schema_version: str
    identifier_basis: str
    condition_type: str
    order: tuple[str, ...]
    entries: tuple[CorruptionEntry, ...]
    inferential_severities: tuple[int, ...]
    roles: _roles.Roles
    vocabulary_sha256: str
    source_path: str

    @property
    def display_names(self) -> Mapping[str, str]:
        return MappingProxyType({e.id: e.display_name for e in self.entries})

    def is_inferential_severity(self, severity: int) -> bool:
        return severity in self.inferential_severities

    def severity_role(self, corruption: str, severity) -> str | None:
        """'inferential', 'descriptive_only' or 'excluded'; None for a severity outside 1-5."""
        return self.roles.severity_role(corruption, severity)

    def scored_severities(self, corruption: str) -> tuple[int, ...]:
        return self.roles.scored_severities(corruption)

    @property
    def descriptive_only_by_corruption(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(self.roles.descriptive_only_by_corruption)

    @property
    def excluded_by_corruption(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(self.roles.excluded_by_corruption)

    @property
    def cells(self) -> tuple[Cell, ...]:
        """The 24 scored cells in protocol order."""
        return self.roles.cells

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self.roles.cell_ids


def vocabulary_sha256(entries) -> str:
    """Canonical hash of the ordered (id, display_name, reference_function) triples."""
    payload = [[e.id, e.display_name, e.reference_function] for e in entries]
    return hashlib.sha256(json.dumps(payload, sort_keys=False, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _fail(msg: str) -> None:
    raise CorruptionProtocolError(msg)


def load_corruption_protocol(path=PROTOCOL_PATH) -> CorruptionProtocol:
    """Parse and fully validate the frozen protocol. Reads the file only when called."""
    p = Path(path)
    try:
        roles = _roles.load(p)
    except CorruptionRolesError as e:
        raise CorruptionProtocolError(str(e)) from e

    if len(roles.order) != N_CORRUPTIONS:
        _fail(f"expected exactly {N_CORRUPTIONS} corruption entries, got {len(roles.order)}")
    if tuple(INFERENTIAL_SEVERITIES) != roles.inferential_severities:
        _fail(f"protocol inferential severities {list(roles.inferential_severities)} disagree "
              f"with the frozen mIoU-C rule {list(INFERENTIAL_SEVERITIES)}")

    entries = tuple(CorruptionEntry(*e) for e in roles.entries)
    return CorruptionProtocol(
        schema_version=roles.schema_version, identifier_basis=roles.identifier_basis,
        condition_type=roles.condition_type, order=roles.order, entries=entries,
        inferential_severities=roles.inferential_severities, roles=roles,
        vocabulary_sha256=vocabulary_sha256(entries),
        source_path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p))


def build_official_corruption_grid(protocol: CorruptionProtocol) -> CorruptionGrid:
    """Construct the A3a `CorruptionGrid` from the frozen protocol.

    Delegates to `CorruptionGrid`'s own validation rather than duplicating it.
    """
    return CorruptionGrid(
        names=protocol.order,
        display_names=dict(protocol.display_names),
        provenance=(f"official: {protocol.source_path} ({protocol.schema_version}, "
                    f"vocabulary sha256 {protocol.vocabulary_sha256[:16]}...)"))


def official_corruption_grid(path=PROTOCOL_PATH) -> CorruptionGrid:
    """Convenience: load the protocol and build the official grid in one call."""
    return build_official_corruption_grid(load_corruption_protocol(path))
