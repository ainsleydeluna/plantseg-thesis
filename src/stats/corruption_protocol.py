"""Loader for the frozen machine-readable corruption vocabulary (A3b-0).

`configs/corruption_protocol.json` is the ONE cross-cutting source of truth shared by corruption
generation, evaluation condition metadata, mIoU-C assembly, artifact naming, future cache
manifests, and statistical analysis.

This module is the only stats-side bridge from that JSON to the A3a `CorruptionGrid`.
`src/stats/robustness.py` is deliberately left unchanged: it still requires an injected grid and
holds no vocabulary of its own, so there is exactly one place where the identifiers are asserted.

Identifier basis (A3b-0 `[project-decision]`): the vendored `imagecorruptions` reference
function names that Chapter III selects. The machine identifier and the thesis display label are
distinct -- `brightness` is the official identifier; "brightness variation" is explanatory prose
and is NOT an identifier. Aliases (`brightness_variation`, `motion-blur`, `Motion_Blur`, spaced or
capitalised variants) are rejected outright; no official corruption artifact exists yet, so no
migration compatibility is owed.

NOT frozen here: the corruption implementation bytes, its parameters, the seed policy, and the
cache protocol. Those belong to the future corruption-scaffold task, which must also verify that
the vendored implementation really does expose these five function names.

Import-time behaviour is side-effect free: no file is read until a loader function is called.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .robustness import CorruptionGrid, INFERENTIAL_SEVERITIES, N_CORRUPTIONS

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "configs" / "corruption_protocol.json"

# --- minimal anchoring constants (permitted; the JSON remains the source of truth) -------------
EXPECTED_SCHEMA = "plantseg-corruptions/1.0.0"
EXPECTED_IDENTIFIER_BASIS = "imagecorruptions_reference_function_name"
#: The canonical ordered identifiers. This is the single asserted copy in the codebase --
#: `robustness.py` holds none. Order follows Chapter III's presentation order.
CANONICAL_IDS = ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog")
EXPECTED_INFERENTIAL = (1, 2, 3)
EXPECTED_DESCRIPTIVE_ONLY = (4,)
EXPECTED_EXCLUDED = (5,)

_SNAKE = "abcdefghijklmnopqrstuvwxyz0123456789_"


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
    descriptive_only_severities: tuple[int, ...]
    excluded_severities: tuple[int, ...]
    vocabulary_sha256: str
    source_path: str

    @property
    def display_names(self) -> Mapping[str, str]:
        return MappingProxyType({e.id: e.display_name for e in self.entries})

    def is_inferential_severity(self, severity: int) -> bool:
        return severity in self.inferential_severities


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
        doc = json.loads(p.read_text(encoding="utf-8"),
                         parse_constant=lambda x: _fail(f"non-finite JSON constant {x!r}"))
    except CorruptionProtocolError:
        raise
    except Exception as e:                                   # noqa: BLE001
        _fail(f"cannot parse {p.name} as strict JSON: {type(e).__name__}: {e}")

    if doc.get("schema_version") != EXPECTED_SCHEMA:
        _fail(f"schema_version must be {EXPECTED_SCHEMA!r}, got {doc.get('schema_version')!r}")
    if doc.get("identifier_basis") != EXPECTED_IDENTIFIER_BASIS:
        _fail(f"identifier_basis must be {EXPECTED_IDENTIFIER_BASIS!r}")
    if doc.get("condition_type") != "corruption":
        _fail(f"condition_type must be 'corruption', got {doc.get('condition_type')!r}")

    order = tuple(doc.get("corruption_order", []))
    if order != CANONICAL_IDS:
        _fail(f"corruption_order must be exactly {list(CANONICAL_IDS)} in that order; got "
              f"{list(order)}. Reordering, aliases and case variants are rejected.")

    raw = doc.get("corruptions", [])
    if len(raw) != N_CORRUPTIONS:
        _fail(f"expected exactly {N_CORRUPTIONS} corruption entries, got {len(raw)}")

    entries = []
    for i, item in enumerate(raw):
        missing = [k for k in ("id", "display_name", "reference_function") if k not in item]
        if missing:
            _fail(f"corruptions[{i}] is missing {missing}")
        cid = item["id"]
        if not isinstance(cid, str) or not cid or any(ch not in _SNAKE for ch in cid):
            _fail(f"corruption id {cid!r} must be non-empty lowercase snake_case "
                  "(no spaces, hyphens or capitals)")
        if not isinstance(item["display_name"], str) or not item["display_name"]:
            _fail(f"corruptions[{i}] display_name must be a non-empty string")
        if item["reference_function"] != cid:
            _fail(f"reference_function {item['reference_function']!r} must equal the machine id "
                  f"{cid!r}")
        entries.append(CorruptionEntry(cid, item["display_name"], item["reference_function"]))

    ids = tuple(e.id for e in entries)
    if len(set(ids)) != len(ids):
        _fail(f"duplicate corruption id in {list(ids)}")
    if ids != order:
        _fail(f"corruptions[] order {list(ids)} does not match corruption_order {list(order)}")
    if set(ids) != set(CANONICAL_IDS):
        missing = sorted(set(CANONICAL_IDS) - set(ids))
        extra = sorted(set(ids) - set(CANONICAL_IDS))
        _fail(f"corruption id set mismatch: missing={missing} unexpected={extra}")

    inf = tuple(doc.get("inferential_severities", []))
    des = tuple(doc.get("descriptive_only_severities", []))
    exc = tuple(doc.get("excluded_severities", []))
    if inf != EXPECTED_INFERENTIAL:
        _fail(f"inferential_severities must be {list(EXPECTED_INFERENTIAL)}, got {list(inf)}")
    if des != EXPECTED_DESCRIPTIVE_ONLY:
        _fail(f"descriptive_only_severities must be {list(EXPECTED_DESCRIPTIVE_ONLY)}")
    if exc != EXPECTED_EXCLUDED:
        _fail(f"excluded_severities must be {list(EXPECTED_EXCLUDED)}")
    if set(inf) & set(des) or set(inf) & set(exc) or set(des) & set(exc):
        _fail("severity groups must be disjoint")
    if tuple(INFERENTIAL_SEVERITIES) != inf:
        _fail(f"protocol inferential severities {list(inf)} disagree with the frozen mIoU-C rule "
              f"{list(INFERENTIAL_SEVERITIES)}")

    if "aliases" in doc or "alias_map" in doc:
        _fail("alias maps are rejected: official artifacts accept only the exact identifiers")

    return CorruptionProtocol(
        schema_version=doc["schema_version"], identifier_basis=doc["identifier_basis"],
        condition_type=doc["condition_type"], order=order, entries=tuple(entries),
        inferential_severities=inf, descriptive_only_severities=des, excluded_severities=exc,
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
