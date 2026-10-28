"""Torch-free corruption roles: the anchors, the parser and the cell ids (L-AM16-SEV).

`configs/corruption_protocol.json` (schema `plantseg-corruptions/2.0.0`) is the one source of the
corruption grid. This module holds the single anchor copy of its vocabulary and per-corruption
severity roles, validates a decoded document against them (`parse`), reads the file (`load`) and
names each scored cell `<corruption>_s<severity>` (`cell_id`, `CELL_IDS`). It imports only the
standard library -- never torch, numpy or `src.stats` -- so the torch-free cache generator, the
statistics loader (`src/stats/corruption_protocol.py`) and the TEST-day manifest's
`generator_cells_mismatch` all read one parser. Importing it reads no file.

Roles, per corruption (AM-16 item 6, docs/PREREGISTRATION_AMENDMENTS.md :303-306):
  * severities 1-3 are inferential for all five corruptions (mIoU-C, RPD and rCD);
  * severity 4 is descriptive_only for all five;
  * severity 5 is descriptive_only for motion_blur, jpeg_compression, brightness and fog, and
    excluded for gaussian_noise, which is never generated and never scored.

That gives 24 scored cells: 15 inferential and 9 descriptive_only. Protocol order is
`corruption_order`, then ascending severity.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO / "configs" / "corruption_protocol.json"

# --- anchors (the JSON stays the source of truth; parse() refuses any file that differs) ---------
SCHEMA_VERSION = "plantseg-corruptions/2.0.0"
IDENTIFIER_BASIS = "imagecorruptions_reference_function_name"
CONDITION_TYPE = "corruption"
#: The canonical ordered identifiers, in Chapter III's presentation order.
CANONICAL_IDS = ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog")
SEVERITIES = (1, 2, 3, 4, 5)
INFERENTIAL_SEVERITIES = (1, 2, 3)
DESCRIPTIVE_ONLY_BY_CORRUPTION = MappingProxyType({
    "motion_blur": (4, 5), "gaussian_noise": (4,), "jpeg_compression": (4, 5),
    "brightness": (4, 5), "fog": (4, 5)})
EXCLUDED_BY_CORRUPTION = MappingProxyType({
    "motion_blur": (), "gaussian_noise": (5,), "jpeg_compression": (), "brightness": (),
    "fog": ()})

ROLE_INFERENTIAL = "inferential"
ROLE_DESCRIPTIVE_ONLY = "descriptive_only"
ROLE_EXCLUDED = "excluded"

#: The 1.0.0 grid-wide role keys. A 2.0.0 document carrying either is refused: a 1.x reader of
#: them would read "severity 5 excluded" for all five corruptions.
GRID_WIDE_ROLE_KEYS = ("descriptive_only_severities", "excluded_severities")
ENTRY_KEYS = ("id", "display_name", "reference_function", "descriptive_only_severities",
              "excluded_severities")

_SNAKE = "abcdefghijklmnopqrstuvwxyz0123456789_"
_CELL_ID = re.compile(r"^([a-z0-9_]+)_s([0-9]+)$")


class CorruptionRolesError(RuntimeError):
    """The protocol document, a cell id or a role query does not match the frozen roles."""


def _fail(msg: str):
    raise CorruptionRolesError(msg)


def _is_int(x) -> bool:
    return isinstance(x, Integral) and not isinstance(x, bool)


def _anchor_role(corruption, severity) -> str | None:
    """The anchored role of one (corruption, severity) pair; None if it is not a severity."""
    if corruption not in CANONICAL_IDS:
        _fail(f"unknown corruption {corruption!r}; the frozen vocabulary is {CANONICAL_IDS}")
    if not _is_int(severity):
        return None
    s = int(severity)
    if s in INFERENTIAL_SEVERITIES:
        return ROLE_INFERENTIAL
    if s in DESCRIPTIVE_ONLY_BY_CORRUPTION[corruption]:
        return ROLE_DESCRIPTIVE_ONLY
    if s in EXCLUDED_BY_CORRUPTION[corruption]:
        return ROLE_EXCLUDED
    return None


def cell_id(corruption: str, severity: int) -> str:
    """`<corruption>_s<severity>` for one of the 24 scored cells, e.g. `fog_s5`.

    Refuses an unknown corruption, a severity outside the corruption's scored set and the
    excluded cell (gaussian_noise, 5).
    """
    role = _anchor_role(corruption, severity)
    if role == ROLE_EXCLUDED:
        _fail(f"{corruption} at severity {severity} is excluded (AM-16 item 6): it is never "
              "generated and has no cell id")
    if role is None:
        _fail(f"severity {severity!r} is not a scored severity of {corruption}")
    return f"{corruption}_s{int(severity)}"


def parse_cell_id(text: str) -> tuple[str, int]:
    """Inverse of `cell_id`: `fog_s5` -> ("fog", 5). Refuses anything `cell_id` would not make."""
    m = _CELL_ID.match(text) if isinstance(text, str) else None
    if m is None:
        _fail(f"{text!r} is not a cell id of the form <corruption>_s<severity>")
    corruption, severity = m.group(1), int(m.group(2))
    if corruption not in CANONICAL_IDS or cell_id(corruption, severity) != text:
        _fail(f"{text!r} is not one of the scored cells")
    return corruption, severity


@dataclass(frozen=True)
class Cell:
    """One scored cell of the grid."""
    corruption: str
    severity: int
    role: str

    @property
    def id(self) -> str:
        return cell_id(self.corruption, self.severity)


@dataclass(frozen=True)
class Roles:
    """A validated protocol document: vocabulary, order and per-corruption severity roles."""
    schema_version: str
    identifier_basis: str
    condition_type: str
    order: tuple[str, ...]
    entries: tuple[tuple[str, str, str], ...]      # (id, display_name, reference_function)
    inferential_severities: tuple[int, ...]
    descriptive_only: tuple[tuple[int, ...], ...]  # aligned with `order`
    excluded: tuple[tuple[int, ...], ...]          # aligned with `order`

    def _at(self, corruption) -> int:
        if corruption not in self.order:
            _fail(f"unknown corruption {corruption!r}; the frozen vocabulary is {self.order}")
        return self.order.index(corruption)

    def scored_severities(self, corruption: str) -> tuple[int, ...]:
        """The severities of `corruption` that are generated and scored, ascending."""
        i = self._at(corruption)
        return tuple(sorted(self.inferential_severities + self.descriptive_only[i]))

    def severity_role(self, corruption: str, severity) -> str | None:
        """ROLE_INFERENTIAL, ROLE_DESCRIPTIVE_ONLY or ROLE_EXCLUDED; None for anything that is
        not a severity of this protocol (0, 6, a bool, a non-integer). An unknown corruption
        raises."""
        i = self._at(corruption)
        if not _is_int(severity):
            return None
        s = int(severity)
        if s in self.inferential_severities:
            return ROLE_INFERENTIAL
        if s in self.descriptive_only[i]:
            return ROLE_DESCRIPTIVE_ONLY
        if s in self.excluded[i]:
            return ROLE_EXCLUDED
        return None

    @property
    def descriptive_only_by_corruption(self) -> dict[str, tuple[int, ...]]:
        return dict(zip(self.order, self.descriptive_only))

    @property
    def excluded_by_corruption(self) -> dict[str, tuple[int, ...]]:
        return dict(zip(self.order, self.excluded))

    @property
    def severities_by_corruption(self) -> dict[str, tuple[int, ...]]:
        return {c: self.scored_severities(c) for c in self.order}

    @property
    def cells(self) -> tuple[Cell, ...]:
        """The 24 scored cells in protocol order."""
        return tuple(Cell(c, s, self.severity_role(c, s))
                     for c in self.order for s in self.scored_severities(c))

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.cells)


def _severity_list(item: Mapping, key: str, i: int) -> tuple[int, ...]:
    v = item[key]
    if not isinstance(v, list) or not all(_is_int(x) for x in v):
        _fail(f"corruptions[{i}] {key} must be a JSON array of integers, got {v!r}")
    out = tuple(int(x) for x in v)
    bad = [s for s in out if s not in SEVERITIES]
    if bad:
        _fail(f"corruptions[{i}] {key}: severities {bad} lie outside {list(SEVERITIES)}")
    if list(out) != sorted(set(out)):
        _fail(f"corruptions[{i}] {key} must be strictly ascending without repeats, got {list(out)}")
    return out


def parse(doc: Mapping) -> Roles:
    """Validate a decoded protocol document against the frozen anchors; return its roles.

    The check order and messages follow the 1.0.0 loader's, so the same fault gives the same
    refusal through either loader.
    """
    if not isinstance(doc, Mapping):
        _fail(f"the protocol document must be a JSON object, got {type(doc).__name__}")
    if doc.get("schema_version") != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION!r}, got {doc.get('schema_version')!r}")
    if doc.get("identifier_basis") != IDENTIFIER_BASIS:
        _fail(f"identifier_basis must be {IDENTIFIER_BASIS!r}")
    if doc.get("condition_type") != CONDITION_TYPE:
        _fail(f"condition_type must be 'corruption', got {doc.get('condition_type')!r}")

    raw_order = doc.get("corruption_order", [])
    order = tuple(raw_order) if isinstance(raw_order, list) else (raw_order,)
    if order != CANONICAL_IDS:
        _fail(f"corruption_order must be exactly {list(CANONICAL_IDS)} in that order; got "
              f"{list(order)}. Reordering, aliases and case variants are rejected.")

    raw = doc.get("corruptions", [])
    if not isinstance(raw, list) or len(raw) != len(CANONICAL_IDS):
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        _fail(f"expected exactly {len(CANONICAL_IDS)} corruption entries, got {got}")

    entries, descriptive, excluded = [], [], []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            _fail(f"corruptions[{i}] must be a JSON object")
        missing = [k for k in ENTRY_KEYS if k not in item]
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
        entries.append((cid, item["display_name"], item["reference_function"]))
        descriptive.append(_severity_list(item, "descriptive_only_severities", i))
        excluded.append(_severity_list(item, "excluded_severities", i))

    ids = tuple(e[0] for e in entries)
    if len(set(ids)) != len(ids):
        _fail(f"duplicate corruption id in {list(ids)}")
    if ids != order:
        _fail(f"corruptions[] order {list(ids)} does not match corruption_order {list(order)}")

    inf = doc.get("inferential_severities", [])
    if (not isinstance(inf, list) or not all(_is_int(x) for x in inf)
            or tuple(inf) != INFERENTIAL_SEVERITIES):
        _fail(f"inferential_severities must be {list(INFERENTIAL_SEVERITIES)}, got {inf!r}")
    inf = tuple(int(x) for x in inf)

    present = [k for k in GRID_WIDE_ROLE_KEYS if k in doc]
    if present:
        _fail(f"grid-wide {present} are not part of {SCHEMA_VERSION}: severity roles are per "
              "corruption (corruptions[].descriptive_only_severities and "
              "corruptions[].excluded_severities)")

    for cid, des, exc in zip(ids, descriptive, excluded):
        if set(inf) & set(des) or set(inf) & set(exc) or set(des) & set(exc):
            _fail(f"{cid}: severity groups must be disjoint (inferential {list(inf)}, "
                  f"descriptive_only {list(des)}, excluded {list(exc)})")
        covered = sorted(inf + des + exc)
        if covered != list(SEVERITIES):
            _fail(f"{cid}: the severity roles must cover exactly {list(SEVERITIES)}, got {covered}")
        if des != DESCRIPTIVE_ONLY_BY_CORRUPTION[cid]:
            _fail(f"{cid}: descriptive_only_severities must be "
                  f"{list(DESCRIPTIVE_ONLY_BY_CORRUPTION[cid])}, got {list(des)} (AM-16 item 6)")
        if exc != EXCLUDED_BY_CORRUPTION[cid]:
            _fail(f"{cid}: excluded_severities must be {list(EXCLUDED_BY_CORRUPTION[cid])}, "
                  f"got {list(exc)} (AM-16 item 6)")

    if "aliases" in doc or "alias_map" in doc:
        _fail("alias maps are rejected: official artifacts accept only the exact identifiers")

    return Roles(schema_version=doc["schema_version"], identifier_basis=doc["identifier_basis"],
                 condition_type=doc["condition_type"], order=order, entries=tuple(entries),
                 inferential_severities=inf, descriptive_only=tuple(descriptive),
                 excluded=tuple(excluded))


def _refuse_constant(x):
    _fail(f"non-finite JSON constant {x!r}")


def _unique_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            _fail(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def load(path=PROTOCOL_PATH) -> Roles:
    """Read the protocol file as strict JSON and `parse` it. Reads the file only when called."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"), parse_constant=_refuse_constant,
                         object_pairs_hook=_unique_keys)
    except CorruptionRolesError:
        raise
    except Exception as e:                                   # noqa: BLE001
        _fail(f"cannot parse {p.name} as strict JSON: {type(e).__name__}: {e}")
    return parse(doc)


def registered_cell_ids(path=PROTOCOL_PATH) -> tuple[str, ...]:
    """The ordered cell ids the protocol file registers (equal to CELL_IDS for any file that
    loads)."""
    return load(path).cell_ids


def _anchor_cells() -> tuple[Cell, ...]:
    return tuple(Cell(c, s, _anchor_role(c, s)) for c in CANONICAL_IDS
                 for s in sorted(INFERENTIAL_SEVERITIES + DESCRIPTIVE_ONLY_BY_CORRUPTION[c]))


#: The 24 scored cells and their ids, in protocol order (IF §6), built from the anchors.
CELLS = _anchor_cells()
CELL_IDS = tuple(c.id for c in CELLS)
INFERENTIAL_CELL_IDS = tuple(c.id for c in CELLS if c.role == ROLE_INFERENTIAL)

__all__ = ["CANONICAL_IDS", "CELLS", "CELL_IDS", "CONDITION_TYPE", "Cell", "CorruptionRolesError",
           "DESCRIPTIVE_ONLY_BY_CORRUPTION", "EXCLUDED_BY_CORRUPTION", "GRID_WIDE_ROLE_KEYS",
           "IDENTIFIER_BASIS", "INFERENTIAL_CELL_IDS", "INFERENTIAL_SEVERITIES", "PROTOCOL_PATH",
           "ROLE_DESCRIPTIVE_ONLY", "ROLE_EXCLUDED", "ROLE_INFERENTIAL", "Roles", "SCHEMA_VERSION",
           "SEVERITIES", "cell_id", "load", "parse", "parse_cell_id", "registered_cell_ids"]
