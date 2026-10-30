"""tday_binding/1, the TEST-manifest binding (lane TMAN, B-3b; IF§11.4; DL-76; plan §0 decision (v)).

`scripts/tday_manifest.py bind` writes it once, after the single TEST session, outside the repository; Ice's block
copies it to BINDING_REL and commits it (DL-37). It ties every scored TEST artifact to the frozen manifest, and the
statistics code verifies it before it writes or accepts an official artifact (src/stats/artifact.py,
TEST_MANIFEST_BINDING): bound means the file exists at the repository root's BINDING_REL and file_errors accepts it.

The format (B3R3): exactly {format, references, artifacts, inputs} (schema.BINDING_FILE_SPEC), canonical JSON
(canonical_json_bytes), read strictly; every shape failure is binding_invalid.
  references  exactly {manifest {path, sha256, body_sha256, freeze_dl}, evaluator {commit}, execution_context (the
              manifest's TB1 tuple, copied), session {dedup {sha256}}, mde_entry {path, sha256}, written_utc}
  artifacts   one record per scored artifact, schema.BINDING_ARTIFACT_FIELDS exactly, unique by (slot, precision,
              scope, cell); files_sha256 keyed by the evaluator's four file names (file_names)
  inputs      the 37 of STATS §12.4.6 in src.stats.driver.official_inventory()'s order, each {stage, condition [type,
              name, severity], key [slot, precision, scope, cell]} naming one record of artifacts

The checks hold plain dicts, in layers (B3R5): file_errors needs no manifest (read_inventory, the statistics loader,
bind); export_errors adds each record against the export's rows and cells (read_inventory's bound branch, inventory
--binding). The 37 identities and the stage precisions are passed in as data: the statistics code passes its driver's,
and TMAN, which imports no src.stats module, passes stats_identities() and STATS_PRECISION, which
smoke_tman_binding holds equal to the driver's.

The module level imports the standard library only, since src/stats/artifact.py imports BINDING_REL at module level:
every other import sits inside a function (B3R2).
"""
from __future__ import annotations

from typing import Final

BINDING_REL: Final[str] = "reports/derived/tday_binding.json"
TDAY_REL: Final[str] = "reports/tday"               # the TEST artifact layout's root (artifact_rel; B3R9)

# STATS §12.4.6, restated for TMAN (src.stats.driver's STAGES, CORRUPTION_STAGES, STAGE_PRECISION and
# INPUT_LIST_SCHEMA; smoke_tman_binding holds them equal)
STATS_STAGES: Final = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
STATS_CELL_STAGES: Final = ("E1", "E6")
STATS_PRECISION: Final = {"E1": "fp32", "E2": "fp32", "E3": "fp32", "E4": "int8_ptq", "E5": "int8_qat",
                          "E6": "int8_qat", "E7": "int8_ptq"}
STATS_INPUT_LIST: Final = "plantseg-stats-inputs/1.0.0"
CLEAN: Final = ("clean", None, None)
CELL_SCOPES: Final = ("corrupted", "corrupted_sev45")             # the scopes whose records name a cell
UPSTREAM: Final = "upstream"                                      # the one scope scored under its own protocol


# --------------------------------------------------------------------------------------------- names and rules
def cell_of(condition) -> str | None:
    """The cell ↔ condition rule (B3R5; IF§6), its one definition in the lane: a condition (type, name, severity)
    names no cell when clean and `<name>_s<severity>` otherwise."""
    ctype, name, severity = condition
    return None if ctype == "clean" else f"{name}_s{severity}"


def condition_of(cell: str) -> tuple:
    """cell_of's inverse for a corruption cell: ("corruption", name, severity)."""
    name, _, severity = cell.rpartition("_s")
    return ("corruption", name, int(severity))


def stats_identities(cells) -> tuple:
    """The 37 (stage, condition) identities of STATS §12.4.6 in official_inventory()'s order: the seven clean stages,
    then the inferential cells `cells` (the corrupted scope's, in the frozen corruption order) of E1, then of E6."""
    corr = [condition_of(c) for c in cells]
    return tuple((stage, CLEAN) for stage in STATS_STAGES) + tuple(
        (stage, cond) for stage in STATS_CELL_STAGES for cond in corr)


def file_names() -> tuple:
    """files_sha256's keys: the evaluator's four file names, sorted (src/eval/artifacts.py :69-70), the tuple
    src.stats.driver builds as INPUT_FILES."""
    from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME
    return tuple(sorted((MANIFEST_NAME, *ARTIFACT_FILES)))


def artifact_rel(file_id: str, precision: str, scope: str, cell: str | None) -> str:
    """A scored TEST artifact's folder, relative to the repository root (B3R9; its one definition): reports/tday/
    <file_id>/<precision>/<scope>, and /<cell> when the record names a cell; a path holding "test" is refused. Rows
    are unique by (file_id, precision), so the path names one artifact. The TEST runner calls it."""
    from . import schema as S
    rel = f"{TDAY_REL}/{file_id}/{precision}/{scope}" + ("" if cell is None else f"/{cell}")
    S.refuse_test_path(rel)
    return rel


def key_of(record: dict) -> tuple:
    """A record's key: (slot, precision, scope, cell)."""
    return (record["slot"], record["precision"], record["scope"], record["cell"])


def read(data: bytes, what: str = BINDING_REL) -> dict:
    """The binding's bytes read strictly (schema.strict_json): binding_invalid otherwise."""
    from . import schema as S
    return S.strict_json(data, what, code="binding_invalid")


# ------------------------------------------------------------------------------------------- the file check
def file_errors(doc, identities, precision) -> list[str]:
    """Every way `doc` is not a binding (B3R3, B3R5), with no manifest: the format's exact keys and types; records
    unique by key, keyed by file_names(), in schema.relative_path_error's form, a cell exactly for the cell scopes and
    the upstream protocol exactly for the upstream scope; every record's repo_commit the evaluator commit; one
    split_manifest_sha256 across the canvas records and one across the upstream records; `inputs` exactly the
    `identities` in their order, each key a record of the stage's `precision`, scope clean and no cell for the clean
    condition, scope corrupted and cell_of(condition) for a corruption, no record named twice."""
    from . import schema as S
    errs = S.field_errors(doc, S.BINDING_FILE_SPEC, "binding")
    if errs:
        return errs
    names, commit = list(file_names()), doc["references"]["evaluator"]["commit"]
    recs, digests = {}, {"canvas": set(), UPSTREAM: set()}
    for i, r in enumerate(doc["artifacts"]):
        where = f"artifacts[{i}] {list(key_of(r))}"
        if key_of(r) in recs:
            errs.append(f"{where}: a second record of this (slot, precision, scope, cell)")
        recs[key_of(r)] = r
        if sorted(r["files_sha256"]) != names:
            errs.append(f"{where}: files_sha256 is not keyed by {names}")
        if (r["cell"] is not None) != (r["scope"] in CELL_SCOPES):
            errs.append(f"{where}: a record names a cell exactly when its scope is one of {list(CELL_SCOPES)}")
        if (r["protocol"] == UPSTREAM) != (r["scope"] == UPSTREAM):
            errs.append(f"{where}: the upstream protocol is the upstream scope's, and only its")
        err = S.relative_path_error(r["repo_relative_path"])
        if err:
            errs.append(f"{where}: repo_relative_path {r['repo_relative_path']!r}: {err}")
        if r["repo_commit"] != commit:
            errs.append(f"{where}: repo_commit is not the evaluator commit {commit}")
        digests[UPSTREAM if r["scope"] == UPSTREAM else "canvas"].add(r["split_manifest_sha256"])
    for kind, found in digests.items():
        if len(found) > 1:
            errs.append(f"the {kind} records score {len(found)} split manifests; one is required (DL-74)")
    got = [(it["stage"], tuple(it["condition"])) for it in doc["inputs"]]
    if got != list(identities):
        return errs + [f"inputs are not the {len(identities)} identities of STATS §12.4.6 in their order"]
    named = set()
    for (stage, cond), it in zip(got, doc["inputs"]):
        key = tuple(it["key"]) if len(it["key"]) == 4 and all(k is None or isinstance(k, str) for k in it["key"]) \
            else None
        want = (precision.get(stage), "clean" if cond == CLEAN else "corrupted", cell_of(cond))
        if key not in recs:
            errs.append(f"inputs {stage} {list(cond)}: its key {it['key']} names no record")
        elif key[1:] != want:
            errs.append(f"inputs {stage} {list(cond)}: its key {list(key)} is not of (precision, scope, cell) "
                        f"{list(want)}")
        elif key in named:
            errs.append(f"inputs {stage} {list(cond)}: its key {list(key)} is another input's")
        named.add(key)
    return errs


# ----------------------------------------------------------------------------------------- the export check
def export_errors(doc, export: dict, identities, precision) -> list[str]:
    """file_errors, then each record against the export's own rows and cells (B3R5; read_inventory's bound branch and
    inventory --binding): its (slot, precision) a row's, its cell one of that row's cells, its repo_commit the
    export's evaluator_commit; and the references of the export's manifest: its body sha256 and execution context."""
    errs = file_errors(doc, identities, precision)
    if errs:
        return errs
    rows = {(m["slot"], m["precision"]): m for m in export["models"]}
    for i, r in enumerate(doc["artifacts"]):
        where, row = f"artifacts[{i}] {list(key_of(r))}", rows.get((r["slot"], r["precision"]))
        if row is None:
            errs.append(f"{where}: no row of the export is its (slot, precision)")
        elif r["cell"] is not None and r["cell"] not in row["cells"]:
            errs.append(f"{where}: its cell is not one of its row's cells")
        if r["repo_commit"] != export["evaluator_commit"]:
            errs.append(f"{where}: repo_commit is not the export's evaluator_commit")
    refs = doc["references"]
    if refs["manifest"]["body_sha256"] != export["manifest_sha256"]:
        errs.append("references.manifest.body_sha256 is not the export's manifest_sha256")
    if refs["execution_context"] != export["execution_context"]:
        errs.append("references.execution_context is not the export's execution_context")
    return errs


__all__ = ["BINDING_REL", "CELL_SCOPES", "CLEAN", "STATS_CELL_STAGES", "STATS_INPUT_LIST", "STATS_PRECISION",
           "STATS_STAGES", "TDAY_REL", "artifact_rel", "cell_of", "condition_of", "export_errors", "file_errors",
           "file_names", "key_of", "read", "stats_identities"]
