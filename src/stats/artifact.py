"""The three-file statistics artifact: family.json + bootstrap.npz + MANIFEST.sha256 (A3b; writer gate
and integrity checks: lane L-STATS-OFFICIAL).

Implements STATISTICAL_ANALYSIS_CONTRACT.md sections 12.3 and 12.4 exactly -- the nineteen ordered
top-level keys, the A+ `a3a_family` envelope carrying the complete finalized `HolmFamily`, the
literal A3a dataclass field tuples with a fail-closed runtime source assertion, the ordered 35
bootstrap-task records with `observed_source`, the thirteen-check integrity block, and A2a writer
discipline (refuse-existing, sibling temp dir, strict JSON, NPZ re-read, hash verification, atomic
rename, cleanup on every failure).

Writer. `write_statistics_artifact(out_dir, inputs)` takes typed `ArtifactInputs` and builds the family
itself (`build_family`): the `software_environment` block comes from the running stack, the five
derivable officiality warnings are re-derived (every task's replicate count against 10,000 and its
jackknife count against 1561 for the eight dataset-level tasks and 1561 - k for the 27 per-image
tasks, section 12.3.7 as amended by AM-5), `synthetic_input_data` is the caller's declaration and must
equal what the inputs' dataset names say, and `artifact_status` is derived. Before anything is created
the gate (`check_writer_gate`) re-reads every input directory the records name and refuses, by name, a
policy other than `official` or `nonofficial_smoke`, a section 10.1 pin the contract does not state,
a driver snapshot (P5) that does not name every input's four files, and an official request: while
`TEST_MANIFEST_BINDING` is "unbound" every official request is refused, listing every unmet
condition. The thirteen checks are then established on the files in the temporary
directory by `verify_statistics_artifact`, with the input directories and the contract bytes; only then
is the directory renamed. Any failure removes the temporary directory.

Verifier. `verify_statistics_artifact` owns the thirteen checks, one function each (the rule-to-check
mapping is listed in each function's docstring). With the input directories it reloads the 37 inputs
through the statistics driver and recomputes the eight comparisons, the Holm family, the descriptive
block, the pooled point values and the 35 observed values, requiring exact equality (orchestrator
ruling O4; no bootstrap draws, and no check compares acceleration or bounds across machines). Without
the inputs and the contract bytes it re-establishes the artifact-only checks and reports the others as
not re-established. An `official` artifact is refused while the binding is unbound.

Import-time behaviour is side-effect free; the driver (O4) and the evaluator's dataset name (P6) are
imported lazily.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Final, Mapping

import numpy as np

from .bootstrap import (ANALYSIS_ID_OFFICIAL, ANALYSIS_IDS, CONFIDENCE_LEVEL, DATASET_MIOU_DELTA,
                        DESCRIPTIVE_E1_E3, FALLBACK_WARNINGS, FAMILY_ID, FROZEN_A3A_FIELDS,
                        FROZEN_TASK_IDS, HL_SHIFT, INTERVAL_METHODS, MEAN_DELTA,
                        MEDIAN_DELTA, METHOD_BCA, METHOD_FALLBACK, NONINFERIORITY_E3_E6,
                        OBSERVED_SOURCE_A3A, OBSERVED_SOURCE_PRIMITIVES, OBSERVED_SOURCES,
                        OFFICIAL_JACKKNIFE_N, PRODUCTION_B, ROOT_SEED, SCHEMA_VERSION,
                        STATISTICAL_PROTOCOL, STATUS_OK, STATUS_OK_WARN, TASK_BY_ID, TASK_STATUSES,
                        TASK_WARNINGS, TWO_SIDED, W_SAT_LOWER, W_SAT_UPPER, BootstrapError,
                        assert_a3a_source_schema, canonical_seed_bytes, expected_comparison_ids)
from .align import METRIC_DISEASE_ONLY
from .ingest import EXPECTED_ROWS_TEST, Policy
from .noninferiority import (E6KD_DECISION_RULE, E6KD_TRIGGER_THRESHOLD, PRIMARY_MARGIN,
                             SENSITIVITY_MARGINS)
from .tests import CANONICAL_COMPARISON_IDS, StatsError, holm_family

FAMILY_JSON = "family.json"
BOOTSTRAP_NPZ = "bootstrap.npz"
MANIFEST_NAME = "MANIFEST.sha256"
ARTIFACT_FILES = (BOOTSTRAP_NPZ, FAMILY_JSON)          # lexicographic; manifest excluded

ALPHA = 0.05
STATUS_OFFICIAL = "official"
STATUS_NONOFFICIAL = "nonofficial"
ARTIFACT_STATUSES = (STATUS_OFFICIAL, STATUS_NONOFFICIAL)

#: Contract section 12.4.3 -- exhaustive over all NON-FATAL officiality defects.
W_ANALYSIS_ID = "analysis_id_nonofficial"
W_SOFTWARE = "software_environment_unpinned"
W_INPUT = "input_artifact_nonofficial"
W_SYNTHETIC = "synthetic_input_data"
W_B = "bootstrap_replicates_nonproduction"
W_JACKKNIFE = "jackknife_count_nonproduction"
OFFICIALITY_WARNINGS = (W_ANALYSIS_ID, W_SOFTWARE, W_INPUT, W_SYNTHETIC, W_B, W_JACKKNIFE)
_W_RANK = {w: i for i, w in enumerate(OFFICIALITY_WARNINGS)}

PINNED_ENVIRONMENT = {"python": "3.11", "numpy": "1.26.4", "scipy": "1.11.4",
                      "statsmodels": "0.14.6"}

TOP_LEVEL_KEYS = ("schema_version", "statistical_protocol", "family_id", "analysis_id", "run_id",
                  "created_at_utc", "artifact_status", "warnings", "alpha",
                  "expected_comparison_ids", "statistical_contract_sha256",
                  "software_environment", "input_artifacts", "a3a_family", "descriptive_e1_e3",
                  "noninferiority_e3_e6", "e6_kd_trigger", "bootstrap_tasks", "integrity")

INPUT_ARTIFACT_FIELDS = ("stage", "condition", "repo_relative_path", "artifact_status",
                         "checkpoint_sha256", "split_manifest_sha256", "class_map_sha256",
                         "metric_impl_sha256", "config_sha256", "repo_commit")

INTEGRITY_CHECKS = ("input_artifact_provenance_verified", "statistical_contract_hash_verified",
                    "a3a_family_complete", "bootstrap_task_matrix_complete",
                    "bootstrap_task_order_canonical", "observed_source_mapping_verified",
                    "seed_material_verified", "json_npz_exact_match", "npz_schema_verified",
                    "finite_values_verified", "officiality_policy_verified",
                    "exact_file_set_verified", "manifest_verified")
(C_PROVENANCE, C_CONTRACT, C_A3A, C_MATRIX, C_ORDER, C_SOURCE, C_SEED, C_JSON_NPZ, C_NPZ, C_FINITE,
 C_POLICY, C_FILES, C_MANIFEST) = INTEGRITY_CHECKS

RUN_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

NPZ_SUFFIXES = ("bootstrap", "jackknife", "observed", "root_seed", "seed_digest",
                "seed_entropy_bytes", "z0", "acceleration", "acceleration_defined",
                "interval_type", "interval_method", "confidence_level", "bounds")

#: The repository this code lives in. The contract, the corruption protocol, the metric code and the
#: code-provenance check always come from here; `--repo-root` only resolves input paths (P16).
CODE_REPO = Path(__file__).resolve().parents[2]
CONTRACT_REL = "docs/STATISTICAL_ANALYSIS_CONTRACT.md"
METRICS_REL = "src/eval/metrics.py"
#: Orchestrator ruling O3: clean at HEAD of the code repository for an official request.
OFFICIAL_CODE_PATHS = ("src/stats", "src/stats/driver.py", "scripts/run_stats.py",
                       "configs/corruption_protocol.json", CONTRACT_REL)

#: The TEST-manifest binding (P2). "unbound" is the only value in this lane: no flag, environment
#: variable, argument or context field sets it, and no code path reads another value. While it is
#: unbound the writer gate refuses every official request and the verifier refuses every artifact
#: whose status is `official`. The TEST-manifest lane replaces this constant with the binding.
TEST_MANIFEST_BINDING: Final[str] = "unbound"

#: Policies a statistics artifact may be computed under (P3). REHEARSAL never writes one.
ARTIFACT_POLICIES = (Policy.OFFICIAL.value, Policy.NONOFFICIAL_SMOKE.value)
#: The evaluator's input statuses (section 12.4.3: a different vocabulary from the top level).
INPUT_STATUSES = ("official", "provisional", "smoke")
BIT_GENERATOR = "PCG64"
# Applied with fullmatch only: `$` alone also matches just before a trailing newline.
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ArtifactError(RuntimeError):
    """Refusal or integrity failure. Never downgraded into an officiality warning."""


class IntegrityError(ArtifactError):
    """One of the thirteen integrity checks of section 12.4.11 cannot be established."""

    def __init__(self, check: str, detail: str):
        if check not in INTEGRITY_CHECKS:
            raise ValueError(f"unknown integrity check {check!r}")
        super().__init__(f"integrity check {check} not established: {detail}")
        self.check = check
        self.detail = detail


class WriterGateRefused(ArtifactError):
    """The writer gate refused the request before anything was created; `unmet` names every
    reason as 'name: detail'."""

    def __init__(self, unmet):
        self.unmet = tuple(unmet)
        super().__init__("writer gate refused: " + " | ".join(self.unmet))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(u.split(":", 1)[0] for u in self.unmet)


def _fail(check: str, detail: str):
    raise IntegrityError(check, detail)


# --------------------------------------------------------------------------------------------------
# 1. ordered A3a serialization (contract section 12.4.7.2)
# --------------------------------------------------------------------------------------------------
def _finite(v: float, where: str) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ArtifactError(f"non-finite value at {where}: {v!r}")
    return f


def _serialize_value(v, where: str):
    if isinstance(v, Enum):                                    # Policy -> .value, never the object
        return v.value
    if type(v).__name__ in FROZEN_A3A_FIELDS:
        return serialize_a3a(v)
    if isinstance(v, (tuple, list)):
        return [_serialize_value(x, where) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _finite(v, where)
    if v is None or isinstance(v, str):
        return v
    raise ArtifactError(f"unserializable value at {where}: {type(v).__name__}")


def serialize_a3a(obj) -> dict:
    """Explicit ordered mapping from the contract whitelist. `asdict()` is never the authority."""
    name = type(obj).__name__
    frozen = FROZEN_A3A_FIELDS.get(name)
    if frozen is None:
        raise ArtifactError(f"{name} is not a frozen A3a schema")
    return {f: _serialize_value(getattr(obj, f), f"{name}.{f}") for f in frozen}


# --------------------------------------------------------------------------------------------------
# 2. officiality (sections 10, 10.1, 12.3.7, 12.4.3, 12.4.5)
# --------------------------------------------------------------------------------------------------
def observed_environment() -> dict:
    import scipy
    import statsmodels
    return {"python": "%d.%d" % sys.version_info[:2], "numpy": np.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def software_environment_block() -> dict:
    """Always the running stack; a caller never supplies it (section 12.4.3 note)."""
    obs = observed_environment()
    return {"pinned": dict(PINNED_ENVIRONMENT), "observed": obs,
            "matches_pinned": obs == PINNED_ENVIRONMENT}


def require_k(k) -> int:
    """P7: the AM-5 count is an int (never a bool) with 0 <= k < 1561."""
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < OFFICIAL_JACKKNIFE_N:
        raise ArtifactError(f"am5_excluded_count must be an int (not a bool) with 0 <= k < "
                            f"{OFFICIAL_JACKKNIFE_N}, got {k!r}")
    return k


def expected_jackknife_count(task_id: str, am5_excluded_count: int) -> int:
    """Section 12.3.7 as amended by AM-5: 1561 for the eight dataset-level tasks, 1561 - k for the 27
    tasks on per-image values."""
    if TASK_BY_ID[task_id].statistic_name == DATASET_MIOU_DELTA:
        return OFFICIAL_JACKKNIFE_N
    return OFFICIAL_JACKKNIFE_N - require_k(am5_excluded_count)


def officiality_warnings(*, analysis_id: str, software: dict, input_artifacts: list,
                         synthetic: bool, task_counts, am5_excluded_count: int) -> tuple[str, ...]:
    """Every NON-FATAL officiality defect maps here; anything else raises. No third path.

    `task_counts` holds (task_id, bootstrap_replicates, jackknife_count) for every task: every task
    is compared with its own expected values, never task 0 alone.
    """
    if not isinstance(synthetic, bool):
        raise ArtifactError(f"synthetic must be a bool, got {synthetic!r}")
    k = require_k(am5_excluded_count)
    ws: list[str] = []
    if analysis_id != ANALYSIS_ID_OFFICIAL:
        ws.append(W_ANALYSIS_ID)
    if not software["matches_pinned"]:
        ws.append(W_SOFTWARE)
    if any(a["artifact_status"] != STATUS_OFFICIAL for a in input_artifacts):
        ws.append(W_INPUT)
    if synthetic:
        ws.append(W_SYNTHETIC)
    if any(b != PRODUCTION_B for _, b, _ in task_counts):
        ws.append(W_B)
    if any(n != expected_jackknife_count(t, k) for t, _, n in task_counts):
        ws.append(W_JACKKNIFE)
    return tuple(sorted(set(ws), key=lambda w: _W_RANK[w]))


def derive_artifact_status(warnings: tuple[str, ...]) -> str:
    """Asserted biconditional -- status is derived, never set independently."""
    for w in warnings:
        if w not in _W_RANK:
            raise ArtifactError(f"warning outside the frozen officiality vocabulary: {w!r}")
    return STATUS_OFFICIAL if not warnings else STATUS_NONOFFICIAL


def contract_sha256(repo_root: Path = CODE_REPO) -> tuple[str, bytes]:
    """Read the governing contract bytes ONCE; hash that buffer. No reread, no re-encode."""
    raw = (Path(repo_root) / CONTRACT_REL).read_bytes()
    return hashlib.sha256(raw).hexdigest(), raw


_SECTION_10_1 = re.compile(r"^### 10\.1 Software version policy\b[^\n]*$", re.M)
_NEXT_HEADING = re.compile(r"^#{1,3} ", re.M)
_PIN_ANCHOR = "(`requirements.lock`:"
_PIN_PATTERNS = {"python": r"\bPython\s+(\d+\.\d+)(?![.\d])",
                 "numpy": r"\bnumpy\s+(\d+\.\d+\.\d+)(?![.\d])",
                 "scipy": r"\bscipy\s+(\d+\.\d+\.\d+)(?![.\d])",
                 "statsmodels": r"\bstatsmodels\s+(\d+\.\d+\.\d+)(?![.\d])"}


def contract_pins(contract_bytes: bytes) -> dict:
    """The section 10.1 pinned stack, read from the contract (orchestrator OK-2).

    Exactly one '### 10.1 Software version policy' heading, exactly one "(`requirements.lock`:"
    parenthetical inside that section, and exactly one version per package inside the parenthetical;
    anything else raises ArtifactError naming what was not found. There is no fallback.
    """
    try:
        text = bytes(contract_bytes).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ArtifactError(f"section 10.1: the contract is not UTF-8 ({e})") from e
    heads = list(_SECTION_10_1.finditer(text))
    if len(heads) != 1:
        raise ArtifactError(f"section 10.1: found {len(heads)} '### 10.1 Software version policy' "
                            "headings, expected exactly one")
    nxt = _NEXT_HEADING.search(text, heads[0].end())
    body = text[heads[0].end():nxt.start() if nxt else len(text)]
    if body.count(_PIN_ANCHOR) != 1:
        raise ArtifactError(f"section 10.1: the pinned-stack parenthetical {_PIN_ANCHOR!r} occurs "
                            f"{body.count(_PIN_ANCHOR)} times, expected exactly one")
    start = body.index(_PIN_ANCHOR) + len(_PIN_ANCHOR)
    end = body.find(")", start)
    if end < 0:
        raise ArtifactError("section 10.1: the pinned-stack parenthetical is not closed")
    inner = body[start:end]
    pins = {}
    for pkg, pattern in _PIN_PATTERNS.items():
        found = re.findall(pattern, inner)
        if len(found) != 1:
            raise ArtifactError(f"section 10.1: the {pkg} pin occurs {len(found)} times in the "
                                "pinned-stack parenthetical, expected exactly one")
        pins[pkg] = found[0]
    return pins


def require_contract_pins(contract_bytes: bytes) -> None:
    """PINNED_ENVIRONMENT is a double entry of the contract's section 10.1 literals (OK-2)."""
    pins = contract_pins(contract_bytes)
    if pins != PINNED_ENVIRONMENT:
        raise ArtifactError(f"PINNED_ENVIRONMENT {PINNED_ENVIRONMENT} differs from the section 10.1 "
                            f"pins {pins}")


# --------------------------------------------------------------------------------------------------
# 3. family.json assembly
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactInputs:
    """Everything the writer needs; it builds the family itself (P1). Every field is required."""
    analysis_id: str
    run_id: str
    created_at_utc: str
    contract_sha256: str              # recorded when the run started; re-read at finalization
    input_artifacts: list             # the 37 section 12.4.6 records
    comparison_results: tuple
    holm_family: object
    descriptive: object
    non_inferiority: object
    e6_kd: object
    task_results: tuple
    synthetic: bool                   # the caller's declaration; must equal the inputs (P6)
    am5_excluded_count: int           # k, read from the inputs (P7)
    input_root: Path                  # resolves repo_relative_path (P16)
    input_file_sha256: Mapping = field(repr=False)  # rel path -> {file: sha256}, the driver's (P5)


def _task_record(res, spec) -> dict:
    b = res.bca
    rec = {
        "task_id": spec.task_id,
        "comparison_id": spec.comparison_id,
        "statistic_name": spec.statistic_name,
        "interval_type": b.interval_type,
        "confidence_level": _finite(b.confidence_level, "confidence_level"),
        "bootstrap_replicates": int(res.replicates.size),
        "jackknife_count": int(res.jackknife.size),
        "root_seed": int(res.seed.root_seed),
        "seed_input_display": res.seed.seed_input_display,
        "seed_input_hex": res.seed.seed_input_hex,
        "seed_sha256": res.seed.seed_sha256,
        "seed_entropy_hex": res.seed.entropy_bytes.hex(),
        "seed_entropy_decimal": str(res.seed.entropy_int),
        "bit_generator": res.seed.bit_generator,
        "numpy_version": res.seed.numpy_version,
        "observed": _finite(res.observed, f"{spec.task_id}.observed"),
        "observed_source": spec.observed_source,
        "z0": _finite(b.z0, f"{spec.task_id}.z0"),
        "acceleration": (None if not b.acceleration_defined
                         else _finite(b.acceleration, f"{spec.task_id}.acceleration")),
        "acceleration_defined": bool(b.acceleration_defined),
        "interval_method": b.interval_method,
        "warnings": list(b.warnings),
        "status": b.status,
    }
    if rec["seed_entropy_hex"] != rec["seed_sha256"][:32]:
        raise ArtifactError(f"{spec.task_id}: seed_entropy_hex != seed_sha256[:32]")
    expect = STATUS_OK_WARN if rec["warnings"] else STATUS_OK
    if rec["status"] != expect:
        raise ArtifactError(f"{spec.task_id}: status/warnings biconditional violated")
    rec["lower_bound"] = _finite(b.bounds[0], f"{spec.task_id}.lower_bound")
    if b.interval_type == TWO_SIDED:
        if len(b.bounds) != 2:
            raise ArtifactError(f"{spec.task_id}: two_sided requires 2 bounds")
        rec["upper_bound"] = _finite(b.bounds[1], f"{spec.task_id}.upper_bound")
    else:
        if len(b.bounds) != 1:
            raise ArtifactError(f"{spec.task_id}: one_sided_lower requires exactly 1 bound")
        if W_SAT_UPPER in b.warnings or FALLBACK_WARNINGS[7] in b.warnings:
            raise ArtifactError(f"{spec.task_id}: one-sided task carries an upper-tail warning")
    return rec


def _per_image_n(inp: ArtifactInputs, spec) -> int | None:
    """The paired-image count a per-image task resamples; None for a dataset-level task."""
    if spec.statistic_name == DATASET_MIOU_DELTA:
        return None
    if spec.comparison_id == DESCRIPTIVE_E1_E3:
        return int(inp.descriptive.n_paired)
    return int(inp.comparison_results[CANONICAL_COMPARISON_IDS.index(spec.comparison_id)].n_paired)


def _descriptive_block(d) -> dict:
    return {"comparison_id": d.comparison_id, "baseline_stage": d.baseline_stage,
            "candidate_stage": d.candidate_stage, "direction": d.direction, "metric": d.metric,
            "n_paired": int(d.n_paired), "policy": d.policy,
            "alignment_status": d.alignment_status,
            "mean_delta": _finite(d.mean_delta, "descriptive.mean_delta"),
            "mean_delta_pp": _finite(d.mean_delta_pp, "descriptive.mean_delta_pp"),
            "median_delta": _finite(d.median_delta, "descriptive.median_delta"),
            "hodges_lehmann_shift": _finite(d.hodges_lehmann_shift, "descriptive.hl")}


def _valid_timestamp(s) -> bool:
    if not isinstance(s, str):
        return False
    try:
        return datetime.strptime(s, TIMESTAMP_FORMAT).strftime(TIMESTAMP_FORMAT) == s
    except ValueError:
        return False


def build_family(inp: ArtifactInputs) -> dict:
    """The family.json mapping, built from typed inputs. Pure: reads nothing, writes nothing."""
    assert_a3a_source_schema()
    if inp.analysis_id not in ANALYSIS_IDS:
        raise ArtifactError(f"analysis_id {inp.analysis_id!r} is not one of {ANALYSIS_IDS}")
    if not isinstance(inp.run_id, str) or not re.fullmatch(RUN_ID_PATTERN, inp.run_id):
        raise ArtifactError(f"run_id {inp.run_id!r} violates {RUN_ID_PATTERN}")
    if not _valid_timestamp(inp.created_at_utc):
        raise ArtifactError(f"created_at_utc {inp.created_at_utc!r} is not {TIMESTAMP_FORMAT}")
    k = require_k(inp.am5_excluded_count)
    tasks = tuple(inp.task_results)
    if tuple(r.task.task_id for r in tasks) != FROZEN_TASK_IDS:
        raise ArtifactError("bootstrap task set/order does not match the frozen 35-task matrix")
    mixed = sorted({r.analysis_id for r in tasks} - {inp.analysis_id})
    if mixed:
        raise IntegrityError(C_SEED, f"tasks were run under analysis_id {mixed}, the artifact "
                                     f"under {inp.analysis_id!r} (a mixed-namespace artifact)")
    ids = [c.comparison_id for c in inp.comparison_results]
    if ids != list(CANONICAL_COMPARISON_IDS):
        raise IntegrityError(C_A3A, f"comparison results {ids} are not the eight canonical IDs in "
                                    "order")
    for r in tasks:                                       # section 12.3.7: count == n_paired, fatal
        n = _per_image_n(inp, r.task)
        if n is not None and int(r.jackknife.size) != n:
            raise IntegrityError(C_MATRIX, f"{r.task_id}: jackknife count {r.jackknife.size} != its "
                                           f"comparison's n_paired {n}")
    sw = software_environment_block()
    counts = [(r.task_id, int(r.replicates.size), int(r.jackknife.size)) for r in tasks]
    warns = officiality_warnings(analysis_id=inp.analysis_id, software=sw,
                                 input_artifacts=inp.input_artifacts, synthetic=inp.synthetic,
                                 task_counts=counts, am5_excluded_count=k)
    status = derive_artifact_status(warns)

    comparisons = [serialize_a3a(c) for c in inp.comparison_results]
    holm = serialize_a3a(inp.holm_family)
    if holm["status"] != "ok" or holm["complete"] is not True:
        raise ArtifactError("only a finalized HolmFamily (status 'ok', complete true) may be "
                            "serialized")
    if any(not math.isfinite(m["p_adjusted"]) for m in holm["members"]):
        raise ArtifactError("non-finite p_adjusted: an intermediate HolmMember reached the writer")

    fam = {
        "schema_version": SCHEMA_VERSION,
        "statistical_protocol": STATISTICAL_PROTOCOL,
        "family_id": FAMILY_ID,
        "analysis_id": inp.analysis_id,
        "run_id": inp.run_id,
        "created_at_utc": inp.created_at_utc,
        "artifact_status": status,
        "warnings": list(warns),
        "alpha": ALPHA,
        "expected_comparison_ids": list(expected_comparison_ids()),
        "statistical_contract_sha256": inp.contract_sha256,
        "software_environment": sw,
        "input_artifacts": sorted(inp.input_artifacts, key=lambda a: a["repo_relative_path"]),
        "a3a_family": {"completion_status": "complete", "comparisons": comparisons,
                       "holm": holm},
        "descriptive_e1_e3": _descriptive_block(inp.descriptive),
        "noninferiority_e3_e6": _ni_block(inp.non_inferiority),
        "e6_kd_trigger": _e6_block(inp.e6_kd),
        "bootstrap_tasks": [_task_record(r, r.task) for r in tasks],
        "integrity": {"status": "passed", "checks": {c: True for c in INTEGRITY_CHECKS}},
    }
    if tuple(fam.keys()) != TOP_LEVEL_KEYS:
        raise ArtifactError("top-level key set/order does not match the frozen schema")
    _cross_block_identities(fam)
    return fam


def _ni_block(ni) -> dict:
    return {"comparison_id": ni.comparison_id, "baseline_stage": ni.baseline_stage,
            "candidate_stage": ni.candidate_stage, "direction": ni.direction,
            "baseline_dataset_miou": _finite(ni.baseline_dataset_miou, "ni.baseline"),
            "candidate_dataset_miou": _finite(ni.candidate_dataset_miou, "ni.candidate"),
            "observed_delta": _finite(ni.observed_delta, "ni.observed_delta"),
            "observed_drop": _finite(ni.observed_drop, "ni.observed_drop"),
            "relative_retention": _finite(ni.relative_retention, "ni.relative_retention"),
            "interval_type": ni.interval_type,
            "confidence_level": _finite(ni.confidence_level, "ni.confidence_level"),
            "lower_bound": _finite(ni.lower_bound, "ni.lower_bound"),
            "primary_margin": _finite(ni.primary_margin, "ni.primary_margin"),
            "passed": bool(ni.passed),
            "sensitivity": [{"margin": _finite(s.margin, "ni.sensitivity.margin"),
                             "passed": bool(s.passed)} for s in ni.sensitivity]}


def _e6_block(e) -> dict:
    return {"baseline_stage": e.baseline_stage, "candidate_stage": e.candidate_stage,
            "observed_drop": _finite(e.observed_drop, "e6kd.observed_drop"),
            "trigger_threshold": _finite(e.trigger_threshold, "e6kd.trigger_threshold"),
            "decision_rule": e.decision_rule, "triggered": bool(e.triggered)}


def _cross_block_identities(fam: dict) -> None:
    ni = fam["noninferiority_e3_e6"]
    b, c = ni["baseline_dataset_miou"], ni["candidate_dataset_miou"]
    if ni["observed_delta"] != c - b:
        raise ArtifactError("observed_delta != candidate - baseline")
    if ni["observed_drop"] != b - c:
        raise ArtifactError("observed_drop != baseline - candidate")
    if ni["observed_drop"] != -ni["observed_delta"]:
        raise ArtifactError("observed_drop != -observed_delta")
    if ni["relative_retention"] != c / b:
        raise ArtifactError("relative_retention != candidate / baseline")
    if fam["e6_kd_trigger"]["observed_drop"] != ni["observed_drop"]:
        raise ArtifactError("e6_kd_trigger.observed_drop != noninferiority observed_drop")
    task = next(t for t in fam["bootstrap_tasks"]
                if t["task_id"] == "noninferiority_e3_e6__dataset_miou_delta")
    if task["observed"] != ni["observed_delta"]:
        raise ArtifactError("NI task observed value != noninferiority observed_delta")
    if task["lower_bound"] != ni["lower_bound"]:
        raise ArtifactError("NI task lower_bound != noninferiority lower_bound")


# --------------------------------------------------------------------------------------------------
# 4. NPZ and serialization
# --------------------------------------------------------------------------------------------------
def _npz_arrays(task_results) -> dict:
    out = {}
    for r in task_results:
        p = r.task.task_id
        b = r.bca
        out[f"{p}__bootstrap"] = np.asarray(r.replicates, dtype=np.float64)
        out[f"{p}__jackknife"] = np.asarray(r.jackknife, dtype=np.float64)
        out[f"{p}__observed"] = np.array([r.observed], dtype=np.float64)
        out[f"{p}__root_seed"] = np.array([r.seed.root_seed], dtype=np.int64)
        out[f"{p}__seed_digest"] = np.frombuffer(bytes.fromhex(r.seed.seed_sha256),
                                                 dtype=np.uint8).copy()
        out[f"{p}__seed_entropy_bytes"] = np.frombuffer(r.seed.entropy_bytes,
                                                        dtype=np.uint8).copy()
        out[f"{p}__z0"] = np.array([b.z0], dtype=np.float64)
        out[f"{p}__acceleration"] = (np.array([b.acceleration], dtype=np.float64)
                                     if b.acceleration_defined
                                     else np.zeros(0, dtype=np.float64))
        out[f"{p}__acceleration_defined"] = np.array([b.acceleration_defined], dtype=np.bool_)
        out[f"{p}__interval_type"] = np.frombuffer(b.interval_type.encode("utf-8"),
                                                   dtype=np.uint8).copy()
        out[f"{p}__interval_method"] = np.frombuffer(b.interval_method.encode("utf-8"),
                                                     dtype=np.uint8).copy()
        out[f"{p}__confidence_level"] = np.array([b.confidence_level], dtype=np.float64)
        out[f"{p}__bounds"] = np.asarray(b.bounds, dtype=np.float64)
    return out


def canonical_json(fam: dict) -> str:
    """The writer's serialization of family.json (UTF-8, LF, indent 2, non-finite forbidden)."""
    return json.dumps(fam, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _manifest_text(directory: Path) -> str:
    lines = []
    for name in ARTIFACT_FILES:                     # already lexicographic
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")         # sha256sum-compatible: two ASCII spaces
    return "".join(lines)


def strict_json(text: str, check: str):
    """Order-preserving strict parse: duplicate keys, NaN/Infinity constants and number literals that
    overflow float64 (e.g. 1e999) are integrity failures of `check`."""
    def pairs(items):
        keys = [k for k, _ in items]
        if len(set(keys)) != len(keys):
            _fail(check, f"duplicate JSON key(s) {sorted({k for k in keys if keys.count(k) > 1})}")
        return dict(items)

    def constant(name):
        _fail(check, f"non-finite JSON constant {name!r}")

    def number(s):
        v = float(s)
        if not math.isfinite(v):
            _fail(check, f"number literal {s!r} is not a finite float64")
        return v
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except IntegrityError:
        raise
    except (ValueError, RecursionError) as e:
        _fail(check, f"not valid JSON: {e}")


# --------------------------------------------------------------------------------------------------
# 5. the inputs, re-read from the directories the records name (P1, P4, P5, P6, P7, O1)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class InputViews:
    """The 37 inputs as the statistics driver loads them, plus each input's raw per-image rows."""
    policy: Policy
    loaded: Mapping                    # (stage, condition tuple) -> driver.LoadedInput
    rows: Mapping                      # repo_relative_path -> tuple of per_image.jsonl objects
    am5_excluded_count: int
    excluded_clean_ids: tuple

    def by_path(self) -> dict:
        return {li.rel_path: li for li in self.loaded.values()}


def load_input_views(records, input_root, policy: Policy) -> InputViews:
    """Load every input a record names through the driver (four-file hash snapshots before and after
    load_run, MANIFEST agreement, canvas guard, identity), then its raw rows, then hash it again."""
    from . import driver as D
    try:
        entries = [(r["stage"], D.condition_tuple(r["condition"]), r["repo_relative_path"])
                   for r in records]
        loaded = D.load_inputs(entries, Path(input_root), policy)
        k, excluded = D.am5_identity(loaded)
        rows = {}
        for li in loaded.values():
            text = (li.directory / "per_image.jsonl").read_text(encoding="utf-8")
            rows[li.rel_path] = tuple(strict_json(line, C_PROVENANCE) for line in text.splitlines())
            if D.input_file_sha256(li.directory) != dict(li.file_sha256):
                _fail(C_PROVENANCE, f"{li.rel_path}: its files changed while they were being read")
    except D.DriverRefusal as e:
        _fail(C_PROVENANCE, str(e))
    except (OSError, UnicodeDecodeError) as e:
        _fail(C_PROVENANCE, f"an input could not be read: {type(e).__name__}: {e}")
    return InputViews(policy, loaded, rows, k, excluded)


def check_input_provenance(records, views: InputViews, *, expected_sha256=None,
                           declared_k=None) -> None:
    """Check 1 with the inputs: every record equals its input's summary.json; the three hash
    snapshots agree (P5); each input's split manifest recomputed from its rows equals the recorded
    digest, every row carries the artifact's condition, bare stems (O1) and float per-image values in
    [0, 1]; student role and stage precision (Q11); 37 distinct run_ids; one split manifest across
    the 37 (O1); every E1/E6 cell carries its stage's clean checkpoint; k equals the declared count."""
    from ..eval.artifacts import hash_split_manifest
    from ..eval.evaluate import ManifestEntry
    from . import driver as D
    by_path = views.by_path()
    for r in records:
        rel = r["repo_relative_path"]
        li = by_path.get(rel)
        if li is None:
            _fail(C_PROVENANCE, f"no loaded input for record {rel!r}")
        want = D.provenance_record(li)
        if want != r:
            diff = [f for f in INPUT_ARTIFACT_FIELDS if want.get(f) != r.get(f)]
            _fail(C_PROVENANCE, f"{rel}: record field(s) {diff} differ from the input's summary.json")
        if expected_sha256 is not None and dict(expected_sha256.get(rel) or {}) != dict(
                li.file_sha256):
            _fail(C_PROVENANCE, f"{rel}: the input's files differ from the snapshot the driver took "
                                "when it read them (P5)")
        ds, run, rows = li.summary["dataset"], li.summary["run"], views.rows[rel]
        if len(rows) != ds["actual_rows"]:
            _fail(C_PROVENANCE, f"{rel}: {len(rows)} rows, summary says {ds['actual_rows']}")
        try:
            entries = [ManifestEntry(o["manifest_index"], o["image_id"], o["clean_image_id"])
                       for o in rows]
        except (KeyError, TypeError, ValueError) as e:
            _fail(C_PROVENANCE, f"{rel}: a row has no valid manifest identity ({e})")
        if hash_split_manifest(entries) != ds["split_manifest_sha256"]:
            _fail(C_PROVENANCE, f"{rel}: the split manifest recomputed from its rows differs from "
                                "split_manifest_sha256")
        bad = [o.get("image_id") for o in rows if o.get("condition") != ds["condition"]]
        if bad:
            _fail(C_PROVENANCE, f"{rel}: {len(bad)} row(s) carry another condition, e.g. {bad[:3]}")
        stems = [o["image_id"] for o in rows if o["clean_image_id"] != o["image_id"]]
        if stems:
            _fail(C_PROVENANCE, f"{rel}: {len(stems)} row(s) with image_id != clean_image_id, e.g. "
                                f"{stems[:3]} (O1: bare stems)")
        vals = [(o.get("image_id"), key) for o in rows
                for key in ("all_class_miou", "disease_only_miou")
                if o.get(key) is not None and not (type(o[key]) is float and 0.0 <= o[key] <= 1.0)]
        if vals:
            _fail(C_PROVENANCE, f"{rel}: per-image value(s) not a JSON float in [0, 1]: {vals[:3]}")
        if run["model_role"] != "student" or run["precision"] != D.STAGE_PRECISION.get(run["stage"]):
            _fail(C_PROVENANCE, f"{rel}: role/precision ({run['model_role']}, {run['precision']}) is "
                                f"not (student, {D.STAGE_PRECISION.get(run['stage'])})")
    run_ids = [li.summary["run"]["run_id"] for li in views.loaded.values()]
    if len(set(run_ids)) != len(run_ids):
        _fail(C_PROVENANCE, f"run_ids repeat: {sorted({i for i in run_ids if run_ids.count(i) > 1})}")
    manifests = {li.summary["dataset"]["split_manifest_sha256"] for li in views.loaded.values()}
    if len(manifests) != 1:
        _fail(C_PROVENANCE, f"the inputs score {len(manifests)} split manifests; O1 requires one")
    for stage in D.CORRUPTION_STAGES:
        clean = views.loaded[(stage, D.CLEAN)].summary["run"].get("checkpoint_sha256")
        other = sorted(li.rel_path for (st, cond), li in views.loaded.items()
                       if st == stage and cond != D.CLEAN
                       and li.summary["run"].get("checkpoint_sha256") != clean)
        if other:
            _fail(C_PROVENANCE, f"{stage} corruption cell(s) {other[:3]} carry a checkpoint other "
                                f"than the {stage} clean artifact's")
    if declared_k is not None and declared_k != views.am5_excluded_count:
        _fail(C_PROVENANCE, f"am5_excluded_count {declared_k!r} differs from the inputs' AM-5 count "
                            f"{views.am5_excluded_count} (P7)")


def synthetic_from_inputs(views: InputViews) -> tuple[bool, tuple[str, ...]]:
    """P6: synthetic iff an input's dataset name is not the evaluator's DATASET_NAME (imported)."""
    from ..eval.adapters import DATASET_NAME
    names = tuple(sorted({li.summary["dataset"]["name"] for li in views.loaded.values()}))
    return any(n != DATASET_NAME for n in names), names


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_provenance_unmet(repo_root=None, paths=OFFICIAL_CODE_PATHS) -> list[tuple[str, str]]:
    """O3: each path must be tracked and unmodified at HEAD of the code repository (a `git status`
    over that explicit path only). Git unavailable: ('code_provenance_unprovable', why)."""
    root = Path(repo_root) if repo_root is not None else CODE_REPO
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    out = []
    for p in paths:
        try:
            status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--",
                                     p], cwd=str(root), capture_output=True, text=True, timeout=60,
                                    env=env)
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", p], cwd=str(root),
                                     capture_output=True, text=True, timeout=60, env=env)
        except (OSError, subprocess.SubprocessError) as e:
            return [("code_provenance_unprovable", f"git could not be run in {root}: "
                                                   f"{type(e).__name__}: {e}")]
        if status.returncode != 0:
            return [("code_provenance_unprovable", f"git status failed in {root}: "
                                                   f"{status.stderr.strip()[:200]}")]
        if tracked.returncode != 0:
            out.append(("code_not_committed", f"{p} is not tracked in {root}"))
        elif status.stdout.strip():
            out.append(("code_not_clean_at_head",
                        f"{p}: {' | '.join(status.stdout.strip().splitlines()[:3])}"))
    return out


_WARNING_TEXT = {
    W_ANALYSIS_ID: "analysis_id is not plantseg_primary_analysis_v1",
    W_SOFTWARE: "the running stack is not the section 10.1 pinned stack",
    W_INPUT: "an input artifact's status is not official",
    W_SYNTHETIC: "the inputs are declared synthetic",
    W_B: "a task's bootstrap_replicates is not 10,000",
    W_JACKKNIFE: "a task's jackknife_count is not 1561 (dataset-level) or 1561 - k (per-image)",
}
BINDING_CONDITION = "test_manifest_binding_unbound"


def official_conditions(fam: dict, views: InputViews, contract_bytes: bytes | None, *,
                        hooks: bool = False, code_status_repo=None) -> tuple[tuple[str, str], ...]:
    """Every condition an official artifact needs that this family and its inputs do not meet, as
    (name, detail) pairs (P2-P4, O3, OK-3). Never empty while the binding is unbound: the binding is
    always the last entry."""
    if TEST_MANIFEST_BINDING != "unbound":                  # fail closed: there is no bound branch
        raise ArtifactError(f"unknown TEST_MANIFEST_BINDING {TEST_MANIFEST_BINDING!r}")
    unmet: list[tuple[str, str]] = [(w, _WARNING_TEXT.get(w, "warning outside the vocabulary"))
                                    for w in fam["warnings"]]
    comps = fam["a3a_family"]["comparisons"]
    desc = fam["descriptive_e1_e3"]
    pols = sorted({c["policy"] for c in comps} | {desc["policy"]})
    if pols != [Policy.OFFICIAL.value]:
        unmet.append(("policy_not_official", f"policies {pols}"))
    k = views.am5_excluded_count
    ns = {c["comparison_id"]: c["n_paired"] for c in comps}
    ns[DESCRIPTIVE_E1_E3] = desc["n_paired"]
    bad = {cid: n for cid, n in ns.items() if n + k != OFFICIAL_JACKKNIFE_N}
    if bad:
        unmet.append(("n_paired_plus_k_not_1561", f"k = {k}; {bad}"))
    by_path = views.by_path()
    summaries = {rel: li.summary for rel, li in sorted(by_path.items())}
    from ..eval.protocols import CANVAS_PROTOCOL_ID
    noncanvas = [rel for rel, s in summaries.items()
                 if "protocol" in s or s["dataset"].get("preprocess_protocol") != CANVAS_PROTOCOL_ID]
    if noncanvas:
        unmet.append(("non_canvas_input", f"{noncanvas[:3]}"))
    for block, key, name in (("dataset", "class_map_sha256", "class_map_digest_differs"),
                             ("run", "metric_impl_sha256", "metric_impl_digest_differs"),
                             ("dataset", "split_manifest_sha256", "split_manifest_digest_differs")):
        vals = {s[block][key] for s in summaries.values()}
        if len(vals) != 1:
            unmet.append((name, f"{len(vals)} distinct {key} values across the 37 inputs"))
    versions = sorted({t["numpy_version"] for t in fam["bootstrap_tasks"]}
                      | {fam["software_environment"]["observed"]["numpy"]})
    if versions != [np.__version__]:
        unmet.append(("numpy_version_differs", f"tasks/software {versions}, running {np.__version__}"))
    if contract_bytes is None:
        unmet.append(("contract_unavailable", "the contract bytes were not supplied"))
    else:
        try:
            require_contract_pins(contract_bytes)
        except ArtifactError as e:
            unmet.append(("contract_pins_differ", str(e)))
    if fam["software_environment"]["pinned"] != PINNED_ENVIRONMENT:
        unmet.append(("pinned_block_differs", f"{fam['software_environment']['pinned']}"))
    from ..eval.artifacts import EVAL_RUNTIME_VERSION
    metrics_sha = _sha256_file(CODE_REPO / METRICS_REL)
    per_input: dict[str, list[str]] = {}
    for rel, s in summaries.items():
        run, ds = s["run"], s["dataset"]
        rt = run.get("eval_runtime")
        for ok, name in ((run.get("governed_paths_clean") is True, "governed_paths_not_clean"),
                         (isinstance(rt, dict)
                          and rt.get("eval_runtime_version") == EVAL_RUNTIME_VERSION,
                          "eval_runtime_missing"),
                         (bool(_HEX64.fullmatch(str(run.get("checkpoint_sha256") or ""))),
                          "checkpoint_not_64_hex"),
                         (bool(_HEX40.fullmatch(str(run.get("repo_commit") or ""))),
                          "repo_commit_not_40_hex"),
                         (run.get("metric_impl_sha256") == metrics_sha,
                          "metric_impl_not_this_checkout"),
                         (ds.get("split") == "test", "split_not_test"),
                         (ds.get("expected_rows") == ds.get("actual_rows") == EXPECTED_ROWS_TEST,
                          "rows_not_1561"),
                         (run.get("random_init") is False, "random_init")):
            if not ok:
                per_input.setdefault(name, []).append(rel)
    for name, rels in per_input.items():
        unmet.append((name, f"{len(rels)} input(s), e.g. {rels[:3]}"))
    unmet.extend(code_provenance_unmet(code_status_repo))
    if hooks:
        unmet.append(("test_hooks_on_official_request",
                      "_inject_failure/_tamper are refused on an official request (P11)"))
    unmet.append((BINDING_CONDITION, "TEST_MANIFEST_BINDING is 'unbound': no TEST-manifest binding "
                                     "is implemented, so every official request is refused (P2)"))
    return tuple(unmet)


# --------------------------------------------------------------------------------------------------
# 6. the writer gate and the writer
# --------------------------------------------------------------------------------------------------
def _policy_value(p) -> str:
    return p.value if isinstance(p, Enum) else p


def _request_policy(inp: ArtifactInputs) -> Policy:
    """P3, every status: one policy across the eight comparisons and the descriptive block, and it
    is official or nonofficial_smoke; REHEARSAL never writes a statistics artifact."""
    pols = [_policy_value(c.policy) for c in inp.comparison_results] + [
        _policy_value(inp.descriptive.policy)]
    bad = sorted({str(p) for p in pols if p not in ARTIFACT_POLICIES})
    if bad:
        raise WriterGateRefused((f"policy_not_allowed: {bad}; a statistics artifact is computed "
                                 f"under {list(ARTIFACT_POLICIES)} only (rehearsal is refused)",))
    if len(set(pols)) != 1:
        raise WriterGateRefused((f"mixed_policies: {sorted(set(pols))}",))
    return Policy(pols[0])


def _require_contract_hash(recorded, contract_bytes: bytes) -> None:
    now = hashlib.sha256(contract_bytes).hexdigest()
    if recorded != now:
        _fail(C_CONTRACT, f"the contract hash recorded at run start {recorded!r} differs from the "
                          f"contract bytes at finalization {now!r}")


def _require_driver_snapshot(snapshot, records) -> None:
    """P5: the writer compares the files it re-reads with the driver's snapshot, so the snapshot is
    required -- exactly the records' inputs, each mapping the four input files to 64-hex digests."""
    from .driver import INPUT_FILES
    paths = sorted(r["repo_relative_path"] for r in records)
    if not isinstance(snapshot, Mapping) or sorted(snapshot) != paths:
        _fail(C_PROVENANCE, "input_file_sha256 (the driver's P5 snapshot) must name exactly the "
                            "records' inputs")
    for rel in paths:
        files = snapshot[rel]
        if (not isinstance(files, Mapping) or sorted(files) != sorted(INPUT_FILES)
                or not all(isinstance(v, str) and _HEX64.fullmatch(v) for v in files.values())):
            _fail(C_PROVENANCE, f"{rel}: input_file_sha256 must map {list(INPUT_FILES)} to 64-hex "
                                "digests (P5)")


def check_writer_gate(inp: ArtifactInputs, fam: dict, contract_bytes: bytes, *,
                      hooks: bool = False) -> InputViews:
    """Writer step 3: before anything is created. Refuses by name (WriterGateRefused) or raises the
    integrity check that cannot be established (IntegrityError)."""
    policy = _request_policy(inp)
    _require_contract_hash(inp.contract_sha256, contract_bytes)
    try:
        require_contract_pins(contract_bytes)
    except ArtifactError as e:
        raise WriterGateRefused((f"contract_pins: {e}",)) from e
    _require_driver_snapshot(inp.input_file_sha256, fam["input_artifacts"])
    views = load_input_views(fam["input_artifacts"], inp.input_root, policy)
    check_input_provenance(fam["input_artifacts"], views, expected_sha256=inp.input_file_sha256,
                           declared_k=inp.am5_excluded_count)
    derived, names = synthetic_from_inputs(views)
    if derived != inp.synthetic:
        raise WriterGateRefused((f"synthetic_declaration: declared synthetic={inp.synthetic} but the "
                                 f"inputs' dataset names {list(names)} make it {derived} (P6)",))
    if fam["artifact_status"] == STATUS_OFFICIAL:
        raise WriterGateRefused(tuple(f"{n}: {d}" for n, d in official_conditions(
            fam, views, contract_bytes, hooks=hooks)))
    return views


def unmet_official_conditions(inp: ArtifactInputs, *, code_status_repo=None
                              ) -> tuple[tuple[str, str], ...]:
    """Read-only: every condition an official artifact would need that this request does not meet
    (reported by official mode; P30). Builds the family, reads the contract and the inputs, creates
    nothing."""
    fam = build_family(inp)
    _, raw = contract_sha256(CODE_REPO)
    views = load_input_views(fam["input_artifacts"], inp.input_root, _request_policy(inp))
    return official_conditions(fam, views, raw, code_status_repo=code_status_repo)


def stale_temp_dirs(out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    if not out_dir.parent.is_dir():
        return []
    return sorted(p for p in out_dir.parent.iterdir() if p.name.startswith(f".{out_dir.name}.tmp-"))


def preflight_out_dir(out_dir) -> Path:
    """P10: the run_id pattern, refuse-existing, and no stale temporary sibling from an interrupted
    write. The driver calls this before computing; the writer again before creating anything."""
    out_dir = Path(out_dir)
    if not re.fullmatch(RUN_ID_PATTERN, out_dir.name):
        raise ArtifactError(f"run_id {out_dir.name!r} violates {RUN_ID_PATTERN}")
    if out_dir.exists() or out_dir.is_symlink():
        raise ArtifactError(f"refusing to overwrite {out_dir}")
    stale = stale_temp_dirs(out_dir)
    if stale:
        raise ArtifactError(f"stale temporary directory {stale[0]} from an interrupted write; "
                            "inspect and remove it before writing this run_id")
    return out_dir


def write_statistics_artifact(out_dir, inp: ArtifactInputs, *, _inject_failure: bool = False,
                              _tamper: Callable[[Path, str], None] | None = None) -> Path:
    """Build -> gate -> temp sibling -> payloads -> manifest -> all thirteen checks on the temp dir
    with the inputs and the contract -> atomic rename. Every failure removes the temp dir.

    Test hooks (never on an official request; P11): `_inject_failure` raises after the payloads are
    written; `_tamper(tmp, stage)` is called with stage "payloads" (before the manifest) and
    "manifest" (after it). Both fire before the final verifier pass.
    """
    out_dir = Path(out_dir)
    fam = build_family(inp)                                                   # step 1
    _, contract_raw = contract_sha256(CODE_REPO)                              # step 2 (P16)
    hooks = bool(_inject_failure) or _tamper is not None
    check_writer_gate(inp, fam, contract_raw, hooks=hooks)                    # step 3
    if fam["run_id"] != out_dir.name:                                         # step 4
        raise ArtifactError("run_id must equal the final run-directory basename")
    preflight_out_dir(out_dir)

    tmp = out_dir.parent / f".{out_dir.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        (tmp / FAMILY_JSON).write_text(canonical_json(fam), encoding="utf-8", newline="\n")  # 5
        np.savez_compressed(tmp / BOOTSTRAP_NPZ, **_npz_arrays(inp.task_results))
        if hooks and fam["artifact_status"] == STATUS_OFFICIAL:               # step 6
            raise WriterGateRefused(("test_hooks_on_official_request: second guard",))
        if _inject_failure:
            raise ArtifactError("injected integrity failure (test hook)")
        if _tamper is not None:
            _tamper(tmp, "payloads")
        (tmp / MANIFEST_NAME).write_text(_manifest_text(tmp), encoding="utf-8", newline="\n")  # 7
        if _tamper is not None:
            _tamper(tmp, "manifest")
        rep = verify_statistics_artifact(tmp, input_root=inp.input_root,     # step 8
                                         contract_bytes=contract_raw, final_name=out_dir.name,
                                         expected_input_sha256=inp.input_file_sha256)
        if rep.established != INTEGRITY_CHECKS or rep.not_reestablished:
            raise ArtifactError(f"integrity checks not all established: {rep.not_reestablished}")
        if rep.family != fam:
            _fail(C_JSON_NPZ, "family.json read back differs from the family the writer built")
        with np.load(tmp / BOOTSTRAP_NPZ, allow_pickle=False) as z:
            for key, arr in _npz_arrays(inp.task_results).items():
                if z[key].dtype != arr.dtype or z[key].tobytes() != arr.tobytes():
                    _fail(C_JSON_NPZ, f"{key} read back differs from the array the writer built")
        if out_dir.exists() or out_dir.is_symlink():
            raise ArtifactError(f"refusing to overwrite {out_dir}")
        tmp.rename(out_dir)                                                   # step 9
        return out_dir
    except BaseException as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        if tmp.exists():
            raise ArtifactError(f"cleanup failed: temporary directory {tmp} still exists after "
                                f"{type(exc).__name__}: {exc}") from exc
        raise


# --------------------------------------------------------------------------------------------------
# 7. the verifier: one function per integrity check (section 12.4.11)
# --------------------------------------------------------------------------------------------------
def _obj(*fields):
    return ("obj", tuple(fields))


def _list(sub):
    return ("list", sub)


_TASK = ("task",)
_A3A_TYPES = {
    "WilcoxonResult": (("statistic", "float|null"), ("zstatistic", "float|null"),
                       ("p_value", "float|null"), ("status", "str"), ("n_total", "int"),
                       ("n_zero", "int"), ("n_nonzero", "int"), ("warnings", _list("str")),
                       ("zero_method", "str"), ("correction", "bool"), ("alternative", "str"),
                       ("method", "str")),
    "TTestResult": (("statistic", "float|null"), ("p_value", "float|null"), ("df", "int|null"),
                    ("mean_difference", "float"), ("sd_difference", "float|null"),
                    ("standard_error", "float|null"), ("status", "str"),
                    ("warnings", _list("str")), ("alternative", "str")),
    "RankBiserialResult": (("value", "float|null"), ("r_plus", "float"), ("r_minus", "float"),
                           ("denominator", "float"), ("n_zero", "int"), ("n_tied_groups", "int"),
                           ("status", "str"), ("sign_convention", "str")),
    "CohensDzResult": (("value", "float|null"), ("mean", "float"), ("sd", "float"),
                       ("status", "str")),
    "EngineeringShifts": (("mean_delta", "float"), ("mean_delta_pp", "float"),
                          ("median_delta", "float")),
    "HolmMember": (("comparison_id", "str"), ("p_raw", "float"), ("p_adjusted", "float"),
                   ("sorted_rank", "int"), ("step_denominator", "int"), ("step_threshold", "float"),
                   ("reject", "bool"), ("library_reject", "bool"), ("boundary_note", "str|null")),
}
_A3A_TYPES["ComparisonResult"] = (
    ("comparison_id", "str"), ("baseline_stage", "str"), ("candidate_stage", "str"),
    ("metric", "str"), ("direction", "str"), ("n_paired", "int"), ("policy", "str"),
    ("alignment_status", "str"), ("wilcoxon", _obj(*_A3A_TYPES["WilcoxonResult"])),
    ("t_test", _obj(*_A3A_TYPES["TTestResult"])),
    ("rank_biserial", _obj(*_A3A_TYPES["RankBiserialResult"])),
    ("cohens_dz", _obj(*_A3A_TYPES["CohensDzResult"])), ("hodges_lehmann_shift", "float"),
    ("shifts", _obj(*_A3A_TYPES["EngineeringShifts"])), ("warnings", _list("str")),
    ("status", "str"))
_A3A_TYPES["HolmFamily"] = (("members", _list(_obj(*_A3A_TYPES["HolmMember"]))),
                            ("alpha", "float"), ("complete", "bool"), ("status", "str"))
_PIN_KEYS = ("python", "numpy", "scipy", "statsmodels")
_TASK_FIELDS = (("task_id", "str"), ("comparison_id", "str"), ("statistic_name", "str"),
                ("interval_type", "str"), ("confidence_level", "float"),
                ("bootstrap_replicates", "int"), ("jackknife_count", "int"), ("root_seed", "int"),
                ("seed_input_display", "str"), ("seed_input_hex", "str"), ("seed_sha256", "str"),
                ("seed_entropy_hex", "str"), ("seed_entropy_decimal", "str"),
                ("bit_generator", "str"), ("numpy_version", "str"), ("observed", "float"),
                ("observed_source", "str"), ("z0", "float"), ("acceleration", "float|null"),
                ("acceleration_defined", "bool"), ("interval_method", "str"),
                ("warnings", _list("str")), ("status", "str"), ("lower_bound", "float"))
_TOP = _obj(
    ("schema_version", "str"), ("statistical_protocol", "str"), ("family_id", "str"),
    ("analysis_id", "str"), ("run_id", "str"), ("created_at_utc", "str"),
    ("artifact_status", "str"), ("warnings", _list("str")), ("alpha", "float"),
    ("expected_comparison_ids", _list("str")), ("statistical_contract_sha256", "str"),
    ("software_environment", _obj(("pinned", _obj(*[(k, "str") for k in _PIN_KEYS])),
                                  ("observed", _obj(*[(k, "str") for k in _PIN_KEYS])),
                                  ("matches_pinned", "bool"))),
    ("input_artifacts", _list(_obj(
        ("stage", "str"), ("condition", _obj(("type", "str"), ("name", "str|null"),
                                             ("severity", "int|null"))),
        ("repo_relative_path", "str"), ("artifact_status", "str"),
        ("checkpoint_sha256", "str|null"), ("split_manifest_sha256", "str"),
        ("class_map_sha256", "str"), ("metric_impl_sha256", "str"), ("config_sha256", "str"),
        ("repo_commit", "str")))),
    ("a3a_family", _obj(("completion_status", "str"),
                        ("comparisons", _list(_obj(*_A3A_TYPES["ComparisonResult"]))),
                        ("holm", _obj(*_A3A_TYPES["HolmFamily"])))),
    ("descriptive_e1_e3", _obj(
        ("comparison_id", "str"), ("baseline_stage", "str"), ("candidate_stage", "str"),
        ("direction", "str"), ("metric", "str"), ("n_paired", "int"), ("policy", "str"),
        ("alignment_status", "str"), ("mean_delta", "float"), ("mean_delta_pp", "float"),
        ("median_delta", "float"), ("hodges_lehmann_shift", "float"))),
    ("noninferiority_e3_e6", _obj(
        ("comparison_id", "str"), ("baseline_stage", "str"), ("candidate_stage", "str"),
        ("direction", "str"), ("baseline_dataset_miou", "float"),
        ("candidate_dataset_miou", "float"), ("observed_delta", "float"),
        ("observed_drop", "float"), ("relative_retention", "float"), ("interval_type", "str"),
        ("confidence_level", "float"), ("lower_bound", "float"), ("primary_margin", "float"),
        ("passed", "bool"), ("sensitivity", _list(_obj(("margin", "float"), ("passed", "bool")))))),
    ("e6_kd_trigger", _obj(("baseline_stage", "str"), ("candidate_stage", "str"),
                           ("observed_drop", "float"), ("trigger_threshold", "float"),
                           ("decision_rule", "str"), ("triggered", "bool"))),
    ("bootstrap_tasks", _list(_TASK)),
    ("integrity", _obj(("status", "str"), ("checks", _obj(*[(c, "bool") for c in INTEGRITY_CHECKS])))))


def _type_ok(v, spec: str) -> bool:
    for kind in spec.split("|"):
        if ((kind == "str" and isinstance(v, str)) or (kind == "float" and type(v) is float)
                or (kind == "int" and type(v) is int) or (kind == "bool" and type(v) is bool)
                or (kind == "null" and v is None)):
            return True
    return False


def _validate(v, schema, where: str) -> None:
    if isinstance(schema, str):
        if not _type_ok(v, schema):
            _fail(C_FINITE, f"{where}: expected {schema}, got {type(v).__name__} {str(v)[:60]!r}")
        return
    if schema[0] == "obj":
        keys = tuple(k for k, _ in schema[1])
        if not isinstance(v, dict):
            _fail(C_FINITE, f"{where}: expected an object, got {type(v).__name__}")
        if tuple(v) != keys:
            _fail(C_FINITE, f"{where}: keys {list(v)} != {list(keys)} (set and order are frozen)")
        for k, sub in schema[1]:
            _validate(v[k], sub, f"{where}.{k}")
    elif schema[0] == "list":
        if not isinstance(v, list):
            _fail(C_FINITE, f"{where}: expected an array, got {type(v).__name__}")
        for i, x in enumerate(v):
            _validate(x, schema[1], f"{where}[{i}]")
    else:                                                  # a bootstrap-task record
        fields = _TASK_FIELDS + ((("upper_bound", "float"),) if isinstance(v, dict)
                                 and "upper_bound" in v else ())
        _validate(v, _obj(*fields), where)


def _assert_schema_tables() -> None:
    for name, fields in _A3A_TYPES.items():
        if tuple(k for k, _ in fields) != FROZEN_A3A_FIELDS[name]:
            raise ArtifactError(f"verifier schema table for {name} drifted from FROZEN_A3A_FIELDS")
    if tuple(k for k, _ in _TOP[1]) != TOP_LEVEL_KEYS:
        raise ArtifactError("verifier top-level schema drifted from TOP_LEVEL_KEYS")
    if tuple(k for k, _ in _TOP[1][12][1][1][1]) != INPUT_ARTIFACT_FIELDS:
        raise ArtifactError("verifier input-record schema drifted from INPUT_ARTIFACT_FIELDS")


_NPZ_SPEC = {"bootstrap": (np.float64, "n"), "jackknife": (np.float64, "n"),
             "observed": (np.float64, 1), "root_seed": (np.int64, 1), "seed_digest": (np.uint8, 32),
             "seed_entropy_bytes": (np.uint8, 16), "z0": (np.float64, 1),
             "acceleration": (np.float64, "0|1"), "acceleration_defined": (np.bool_, 1),
             "interval_type": (np.uint8, "utf8"), "interval_method": (np.uint8, "utf8"),
             "confidence_level": (np.float64, 1), "bounds": (np.float64, "bounds")}


@dataclass
class _Ctx:
    path: Path
    final_name: str
    input_root: Path | None
    contract_bytes: bytes | None
    expected_input_sha256: Mapping | None
    fam: dict | None = None
    npz: dict | None = None
    views: InputViews | None = None
    observed: object = None
    n_dataset: int = 0
    n_per_image: int = 0


@dataclass(frozen=True)
class VerifyReport:
    established: tuple[str, ...]          # INTEGRITY_CHECKS order
    not_reestablished: tuple[str, ...]    # need the input directories or the contract bytes
    family: dict
    views: InputViews | None = None       # the inputs as loaded, when input_root was given


def _check_exact_file_set(ctx: _Ctx) -> None:
    """Check 12 (exact_file_set_verified): section 12 -- exactly family.json, bootstrap.npz and
    MANIFEST.sha256, each a regular file."""
    if not ctx.path.is_dir():
        _fail(C_FILES, f"{ctx.path} is not a directory")
    names = sorted(p.name for p in ctx.path.iterdir())
    want = sorted([BOOTSTRAP_NPZ, FAMILY_JSON, MANIFEST_NAME])
    if names != want:
        _fail(C_FILES, f"files {names} != {want}")
    for n in want:
        if (ctx.path / n).is_symlink() or not (ctx.path / n).is_file():
            _fail(C_FILES, f"{n} is not a regular file")


def _check_manifest(ctx: _Ctx) -> None:
    """Check 13 (manifest_verified): section 12.3.8 -- exactly two lines '<sha256>  bootstrap.npz'
    and '<sha256>  family.json', lowercase hex, two spaces, bare names, LF, lexicographic order, each
    digest equal to its file's; the manifest does not hash itself."""
    raw = (ctx.path / MANIFEST_NAME).read_bytes()
    if raw != _manifest_text(ctx.path).encode("utf-8"):
        lines = raw.decode("utf-8", "replace").splitlines()
        _fail(C_MANIFEST, f"MANIFEST.sha256 is not the two canonical lines for bootstrap.npz and "
                          f"family.json; it has {len(lines)} line(s): "
                          f"{[ln[-40:] for ln in lines[:4]]}")


def _check_npz_schema(ctx: _Ctx) -> None:
    """Check 9 (npz_schema_verified): sections 12.2, 12.3.2 -- allow_pickle=False; exactly the 35 x 13
    keys; every array one-dimensional with its frozen dtype and length (bounds [2] or [1] and the
    interval type from TASK_BY_ID; acceleration [0] or [1] as acceleration_defined says); the two
    vocabulary arrays decode strictly as UTF-8 into their vocabularies."""
    try:
        with np.load(ctx.path / BOOTSTRAP_NPZ, allow_pickle=False) as z:
            files = list(z.files)
            arrays = {k: z[k] for k in files}
    except Exception as e:                                              # noqa: BLE001
        _fail(C_NPZ, f"bootstrap.npz cannot be read with allow_pickle=False: {type(e).__name__}: {e}")
    want = {f"{t}__{s}" for t in FROZEN_TASK_IDS for s in NPZ_SUFFIXES}
    if len(files) != len(set(files)) or set(files) != want:
        _fail(C_NPZ, f"NPZ key set differs: missing {sorted(want - set(files))[:3]}, extra "
                     f"{sorted(set(files) - want)[:3]}")
    for tid in FROZEN_TASK_IDS:
        spec = TASK_BY_ID[tid]
        for s in NPZ_SUFFIXES:
            a, (dtype, shape) = arrays[f"{tid}__{s}"], _NPZ_SPEC[s]
            where = f"{tid}__{s}"
            if a.dtype != np.dtype(dtype) or a.ndim != 1:
                _fail(C_NPZ, f"{where}: dtype {a.dtype} shape {a.shape}, expected one-dimensional "
                             f"{np.dtype(dtype)}")
            n = a.shape[0]
            if shape == "n":
                ok = n >= 1
            elif shape == "0|1":
                ad = arrays[f"{tid}__acceleration_defined"]
                ok = (ad.dtype == np.bool_ and ad.shape == (1,)
                      and n == (1 if bool(ad[0]) else 0))
            elif shape == "bounds":
                ok = n == (2 if spec.interval_type == TWO_SIDED else 1)
            elif shape == "utf8":
                try:
                    text = a.tobytes().decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    text = None
                ok = (text == spec.interval_type if s == "interval_type"
                      else text in INTERVAL_METHODS)
            else:
                ok = n == shape
            if not ok:
                _fail(C_NPZ, f"{where}: shape {a.shape} or content violates section 12.3.2")
    ctx.npz = arrays


def _check_finite_values(ctx: _Ctx) -> None:
    """Check 10 (finite_values_verified): sections 12.1, 12.4.1, 12.4.12 -- strict UTF-8 JSON with no
    duplicate key, no NaN/Infinity constant and no overflowing number literal; the writer's canonical
    serialization; every block's exact key set, order and JSON types; the frozen literal values of
    every block; every float64 NPZ array finite."""
    _assert_schema_tables()
    raw = (ctx.path / FAMILY_JSON).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        _fail(C_FINITE, f"family.json is not UTF-8 ({e})")
    fam = strict_json(text, C_FINITE)
    _validate(fam, _TOP, "family")
    if canonical_json(fam) != text:
        _fail(C_FINITE, "family.json is not in the writer's canonical serialization")
    frozen = [("schema_version", fam["schema_version"], SCHEMA_VERSION),
              ("statistical_protocol", fam["statistical_protocol"], STATISTICAL_PROTOCOL),
              ("family_id", fam["family_id"], FAMILY_ID), ("alpha", fam["alpha"], ALPHA),
              ("expected_comparison_ids", fam["expected_comparison_ids"],
               list(CANONICAL_COMPARISON_IDS)),
              ("a3a_family.completion_status", fam["a3a_family"]["completion_status"], "complete"),
              ("a3a_family.holm.alpha", fam["a3a_family"]["holm"]["alpha"], ALPHA),
              ("a3a_family.holm.complete", fam["a3a_family"]["holm"]["complete"], True),
              ("a3a_family.holm.status", fam["a3a_family"]["holm"]["status"], "ok"),
              ("integrity", fam["integrity"],
               {"status": "passed", "checks": {c: True for c in INTEGRITY_CHECKS}})]
    d, ni, e6 = fam["descriptive_e1_e3"], fam["noninferiority_e3_e6"], fam["e6_kd_trigger"]
    frozen += [(f"descriptive_e1_e3.{k}", d[k], v) for k, v in (
        ("comparison_id", DESCRIPTIVE_E1_E3), ("baseline_stage", "E1"), ("candidate_stage", "E3"),
        ("direction", "candidate_minus_baseline"))]
    frozen += [(f"noninferiority_e3_e6.{k}", ni[k], v) for k, v in (
        ("comparison_id", NONINFERIORITY_E3_E6), ("baseline_stage", "E3"), ("candidate_stage", "E6"),
        ("direction", "candidate_minus_baseline"), ("interval_type", "one_sided_lower"),
        ("confidence_level", CONFIDENCE_LEVEL), ("primary_margin", PRIMARY_MARGIN))]
    frozen.append(("noninferiority_e3_e6.sensitivity margins", [s["margin"] for s in
                                                                ni["sensitivity"]],
                   list(SENSITIVITY_MARGINS)))
    frozen += [(f"e6_kd_trigger.{k}", e6[k], v) for k, v in (
        ("baseline_stage", "E3"), ("candidate_stage", "E6"),
        ("trigger_threshold", E6KD_TRIGGER_THRESHOLD), ("decision_rule", E6KD_DECISION_RULE))]
    for c in fam["a3a_family"]["comparisons"]:
        cid = c["comparison_id"]
        frozen += [(f"{cid}.direction", c["direction"], "candidate_minus_baseline"),
                   (f"{cid}.wilcoxon config", [c["wilcoxon"][k] for k in (
                       "zero_method", "correction", "alternative", "method")],
                    ["pratt", True, "greater", "approx"]),
                   (f"{cid}.t_test.alternative", c["t_test"]["alternative"], "greater"),
                   (f"{cid}.rank_biserial.sign_convention", c["rank_biserial"]["sign_convention"],
                    "positive favours the candidate")]
    for t in fam["bootstrap_tasks"]:
        frozen.append((f"{t['task_id']}.confidence_level", t["confidence_level"], CONFIDENCE_LEVEL))
    for where, got, want in frozen:
        if got != want or type(got) is not type(want):
            _fail(C_FINITE, f"{where} is {got!r}, frozen value {want!r}")
    for key, a in (ctx.npz or {}).items():
        if a.dtype == np.float64 and not np.all(np.isfinite(a)):
            _fail(C_FINITE, f"{key} contains a non-finite value")
    ctx.fam = fam


def _check_task_order(ctx: _Ctx) -> None:
    """Check 5 (bootstrap_task_order_canonical): section 8.7.2 -- the 35 records in canonical order."""
    ids = tuple(t["task_id"] for t in ctx.fam["bootstrap_tasks"])
    if ids != FROZEN_TASK_IDS:
        first = next((i for i, (a, b) in enumerate(zip(ids, FROZEN_TASK_IDS)) if a != b), len(ids))
        _fail(C_ORDER, f"bootstrap_tasks are not in the section 8.7.2 order (first difference at "
                       f"position {first})")


def _check_task_matrix(ctx: _Ctx) -> None:
    """Check 4 (bootstrap_task_matrix_complete): sections 8.7.2-8.7.3, 12.3.1, 12.3.3, 12.3.5, 12.3.7 --
    exactly the 35 tasks once each; task_id = comparison_id__statistic_name; interval type from the
    task, upper_bound present iff two-sided; interval_method, status and warning vocabularies, the
    status <=> warnings biconditional, canonical warning order, at most one fallback warning,
    fallback method iff one, saturation only on a completed BCa, no upper-tail warning on the
    one-sided task, acceleration null iff undefined; every per-image count equals its comparison's
    n_paired (fatal), one n across the 27 per-image tasks and one N across the 8 dataset-level tasks;
    with the inputs, each dataset-level count equals the pooled image count."""
    fam = ctx.fam
    tasks = fam["bootstrap_tasks"]
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(FROZEN_TASK_IDS) or set(ids) != set(FROZEN_TASK_IDS):
        missing = sorted(set(FROZEN_TASK_IDS) - set(ids))
        odd = [i for i in ids if ids.count(i) > 1 or i not in TASK_BY_ID]
        _fail(C_MATRIX, f"{len(ids)} task records; missing {missing[:3]}, unexpected or repeated "
                        f"{odd[:3]}")
    n_of = {c["comparison_id"]: c["n_paired"] for c in fam["a3a_family"]["comparisons"]}
    n_of[DESCRIPTIVE_E1_E3] = fam["descriptive_e1_e3"]["n_paired"]
    rank = {w: i for i, w in enumerate(TASK_WARNINGS)}
    per_image, dataset = set(), set()
    for t in tasks:
        tid, spec = t["task_id"], TASK_BY_ID[t["task_id"]]
        ws = t["warnings"]
        fb = [w for w in ws if w in FALLBACK_WARNINGS]
        problems = [
            ((t["comparison_id"], t["statistic_name"]) == (spec.comparison_id, spec.statistic_name),
             "comparison_id/statistic_name"),
            (t["interval_type"] == spec.interval_type, "interval_type"),
            (("upper_bound" in t) == (spec.interval_type == TWO_SIDED), "bounds"),
            (t["interval_method"] in INTERVAL_METHODS, "interval_method"),
            (t["status"] in TASK_STATUSES and (t["status"] == STATUS_OK) == (not ws),
             "status/warnings"),
            (all(w in rank for w in ws) and len(set(ws)) == len(ws)
             and ws == sorted(ws, key=lambda w: rank.get(w, -1)), "warnings vocabulary/order"),
            (len(fb) <= 1 and (t["interval_method"] == METHOD_FALLBACK) == (len(fb) == 1),
             "fallback warning"),
            (t["interval_method"] == METHOD_BCA or not ({W_SAT_LOWER, W_SAT_UPPER} & set(ws)),
             "saturation without BCa"),
            (spec.interval_type == TWO_SIDED
             or not ({W_SAT_UPPER, FALLBACK_WARNINGS[7]} & set(ws)), "one-sided upper tail"),
            ((t["acceleration"] is None) == (not t["acceleration_defined"]), "acceleration state"),
            (t["bootstrap_replicates"] >= 1 and t["jackknife_count"] >= 1, "counts")]
        bad = [what for ok, what in problems if not ok]
        if bad:
            _fail(C_MATRIX, f"{tid}: {bad}")
        if spec.statistic_name == DATASET_MIOU_DELTA:
            dataset.add(t["jackknife_count"])
        else:
            n = n_of.get(spec.comparison_id)
            if t["jackknife_count"] != n:
                _fail(C_MATRIX, f"{tid}: jackknife count {t['jackknife_count']} != its comparison's "
                                f"n_paired {n} (section 12.3.7)")
            per_image.add(t["jackknife_count"])
    if len(dataset) != 1 or len(per_image) != 1 or min(dataset) < min(per_image):
        _fail(C_MATRIX, f"counts are not one N across the dataset-level tasks {sorted(dataset)} and "
                        f"one n <= N across the per-image tasks {sorted(per_image)}")
    ctx.n_dataset, ctx.n_per_image = dataset.pop(), per_image.pop()


def _check_json_npz(ctx: _Ctx) -> None:
    """Check 8 (json_npz_exact_match): section 12.3.6 -- exact float64 equality after reload of
    observed, root_seed, seed digest and entropy bytes, z0, the acceleration state and value,
    interval type and method, confidence level and bounds; replicate and jackknife counts equal the
    NPZ array lengths."""
    z = ctx.npz
    for t in ctx.fam["bootstrap_tasks"]:
        p = t["task_id"]
        acc = z[f"{p}__acceleration"]
        bounds = [t["lower_bound"]] + ([t["upper_bound"]] if "upper_bound" in t else [])
        checks = [
            (float(z[f"{p}__observed"][0]) == t["observed"], "observed"),
            (int(z[f"{p}__root_seed"][0]) == t["root_seed"], "root_seed"),
            (z[f"{p}__seed_digest"].tobytes().hex() == t["seed_sha256"], "seed_digest"),
            (z[f"{p}__seed_entropy_bytes"].tobytes().hex() == t["seed_entropy_hex"],
             "seed_entropy_bytes"),
            (float(z[f"{p}__z0"][0]) == t["z0"], "z0"),
            (bool(z[f"{p}__acceleration_defined"][0]) is t["acceleration_defined"],
             "acceleration_defined"),
            ((acc.size == 1 and float(acc[0]) == t["acceleration"]) if t["acceleration_defined"]
             else (acc.size == 0 and t["acceleration"] is None), "acceleration"),
            (z[f"{p}__interval_type"].tobytes().decode("utf-8") == t["interval_type"],
             "interval_type"),
            (z[f"{p}__interval_method"].tobytes().decode("utf-8") == t["interval_method"],
             "interval_method"),
            (float(z[f"{p}__confidence_level"][0]) == t["confidence_level"], "confidence_level"),
            ([float(x) for x in z[f"{p}__bounds"]] == bounds, "bounds"),
            (int(z[f"{p}__bootstrap"].shape[0]) == t["bootstrap_replicates"],
             "bootstrap_replicates"),
            (int(z[f"{p}__jackknife"].shape[0]) == t["jackknife_count"], "jackknife_count")]
        bad = [what for ok, what in checks if not ok]
        if bad:
            _fail(C_JSON_NPZ, f"{p}: {bad} differ between family.json and bootstrap.npz")


def _check_seed_material(ctx: _Ctx) -> None:
    """Check 7 (seed_material_verified): sections 8.1, 8.7.1, 12.3.3-12.3.4 -- the artifact's
    analysis_id is in the frozen vocabulary; every task's root_seed is 42 and bit_generator PCG64;
    the canonical bytes rebuilt from the artifact-level analysis_id, the task's ids and root seed 42
    equal seed_input_hex exactly; then checks 1-5 of section 12.3.4 and the exact display string."""
    fam, z = ctx.fam, ctx.npz
    aid = fam["analysis_id"]
    if aid not in ANALYSIS_IDS:
        _fail(C_SEED, f"analysis_id {aid!r} is not one of {ANALYSIS_IDS}")
    for t in fam["bootstrap_tasks"]:
        p = t["task_id"]
        if t["root_seed"] != ROOT_SEED or t["bit_generator"] != BIT_GENERATOR:
            _fail(C_SEED, f"{p}: root_seed {t['root_seed']!r}, bit_generator "
                          f"{t['bit_generator']!r}; frozen {ROOT_SEED}, {BIT_GENERATOR!r}")
        try:
            raw = canonical_seed_bytes(aid, t["comparison_id"], t["statistic_name"], ROOT_SEED)
        except BootstrapError as e:
            _fail(C_SEED, f"{p}: {e}")
        digest = hashlib.sha256(raw).digest()
        display = (f"plantseg-stats/1.0.0\\0root_seed={ROOT_SEED}\\0{aid}"
                   f"\\0{t['comparison_id']}\\0{t['statistic_name']}")
        checks = [(t["seed_input_hex"] == raw.hex(), "seed_input_hex != rebuilt canonical bytes"),
                  (t["seed_sha256"] == digest.hex(), "1: seed_sha256"),
                  (z[f"{p}__seed_digest"].tobytes() == digest, "2: NPZ seed_digest"),
                  (t["seed_entropy_hex"] == t["seed_sha256"][:32], "3: seed_entropy_hex"),
                  (z[f"{p}__seed_entropy_bytes"].tobytes() == digest[:16],
                   "4: NPZ seed_entropy_bytes"),
                  (t["seed_entropy_decimal"] == str(int.from_bytes(digest[:16], "big")),
                   "5: seed_entropy_decimal"),
                  (t["seed_input_display"] == display, "seed_input_display"),
                  (int(z[f"{p}__root_seed"][0]) == ROOT_SEED, "NPZ root_seed")]
        bad = [what for ok, what in checks if not ok]
        if bad:
            _fail(C_SEED, f"{p}: {bad}")


def _holm_recomputed(comps) -> dict:
    """The frozen holm_family over duck-typed carriers of the eight stored p-values."""
    carriers = [types.SimpleNamespace(comparison_id=c["comparison_id"],
                                      primary_p=c["wilcoxon"]["p_value"],
                                      alignment_status=c["alignment_status"]) for c in comps]
    try:
        return serialize_a3a(holm_family(carriers))
    except (StatsError, ArtifactError, TypeError, ValueError) as e:
        _fail(C_A3A, f"the Holm family cannot be recomputed from the stored p-values: {e}")


def _check_a3a_family(ctx: _Ctx) -> None:
    """Check 3 (a3a_family_complete): sections 12.4.4, 12.4.7-12.4.7.3 -- the source dataclass field
    tuples; top-level alpha == holm.alpha == 0.05; completion_status <=> holm.complete; eight
    comparisons and eight members, positionally aligned with expected_comparison_ids; each p_raw ==
    its wilcoxon.p_value; stages and metric from the driver's comparison table; alignment status ok;
    the Holm family recomputed by the frozen holm_family equals the stored one field for field; with
    the inputs, the eight comparisons recomputed from the 37 inputs equal the stored ones (O4)."""
    from .driver import COMPARISON_TABLE
    try:
        assert_a3a_source_schema()
    except BootstrapError as e:
        _fail(C_A3A, str(e))
    fam = ctx.fam
    a3a, holm = fam["a3a_family"], fam["a3a_family"]["holm"]
    comps, members = a3a["comparisons"], holm["members"]
    expected = list(CANONICAL_COMPARISON_IDS)
    problems = [(fam["alpha"] == holm["alpha"] == ALPHA, "alpha"),
                ((a3a["completion_status"] == "complete") is (holm["complete"] is True),
                 "completion_status <=> holm.complete"),
                (len(comps) == 8 and len(members) == 8, "eight comparisons and eight members"),
                ([c["comparison_id"] for c in comps] == [m["comparison_id"] for m in members]
                 == fam["expected_comparison_ids"] == expected, "positional comparison IDs")]
    bad = [what for ok, what in problems if not ok]
    if bad:
        _fail(C_A3A, f"A+ invariants violated: {bad}")
    for c, m in zip(comps, members):
        cid = c["comparison_id"]
        if c["wilcoxon"]["p_value"] is None or m["p_raw"] != c["wilcoxon"]["p_value"]:
            _fail(C_A3A, f"{cid}: p_raw {m['p_raw']!r} != wilcoxon.p_value "
                         f"{c['wilcoxon']['p_value']!r}")
        if (c["baseline_stage"], c["candidate_stage"], c["metric"]) != COMPARISON_TABLE[cid]:
            _fail(C_A3A, f"{cid}: stages/metric are not the comparison table's")
        if c["alignment_status"] not in ("ok", "ok_with_warnings"):
            _fail(C_A3A, f"{cid}: alignment_status {c['alignment_status']!r}")
    again = _holm_recomputed(comps)
    if again != holm:
        diff = [(i, k) for i, (a, b) in enumerate(zip(again["members"], members))
                for k in a if a[k] != b[k]]
        _fail(C_A3A, f"the stored Holm family differs from holm_family recomputed from the eight "
                     f"p-values (member, field): {diff[:4]}")


def _check_observed_sources(ctx: _Ctx) -> None:
    """Check 6 (observed_source_mapping_verified): sections 12.3.3 (24/3/8), 12.4.8-12.4.10 -- every
    task's observed_source is its frozen one; Source A observed values equal their comparison record
    bit for bit, Source B the descriptive block's; the NI task's observed and lower bound equal the
    NI block's; the descriptive identity mean_delta_pp = 100 x mean_delta; the NI identities, the
    strictly positive baseline, the strict primary and sensitivity decisions read from the one stored
    bound; E6-KD drop and strict trigger. With the inputs: the 35 observed values, the descriptive
    block and the pooled stage values recomputed from the inputs equal the stored ones (O4)."""
    fam = ctx.fam
    comps = {c["comparison_id"]: c for c in fam["a3a_family"]["comparisons"]}
    d, ni, e6 = fam["descriptive_e1_e3"], fam["noninferiority_e3_e6"], fam["e6_kd_trigger"]
    field_of = {MEAN_DELTA: ("shifts", "mean_delta"), MEDIAN_DELTA: ("shifts", "median_delta"),
                HL_SHIFT: (None, "hodges_lehmann_shift")}
    for t in fam["bootstrap_tasks"]:
        spec = TASK_BY_ID[t["task_id"]]
        if t["observed_source"] != spec.observed_source or t["observed_source"] not in OBSERVED_SOURCES:
            _fail(C_SOURCE, f"{t['task_id']}: observed_source {t['observed_source']!r}, frozen "
                            f"{spec.observed_source!r}")
        if spec.observed_source == OBSERVED_SOURCE_A3A:
            block, key = field_of[spec.statistic_name]
            rec = comps[spec.comparison_id]
            want = rec[block][key] if block else rec[key]
        elif spec.observed_source == OBSERVED_SOURCE_PRIMITIVES:
            want = d[spec.statistic_name]
        elif spec.comparison_id == NONINFERIORITY_E3_E6:
            want = ni["observed_delta"]
            if t["lower_bound"] != ni["lower_bound"]:
                _fail(C_SOURCE, "NI task lower_bound != noninferiority_e3_e6.lower_bound")
        else:
            continue                                     # pooled clean deltas: recomputed with inputs
        if t["observed"] != want:
            _fail(C_SOURCE, f"{t['task_id']}: observed {t['observed']!r} != its source {want!r}")
    b, c = ni["baseline_dataset_miou"], ni["candidate_dataset_miou"]
    identities = [
        (d["mean_delta_pp"] == 100.0 * d["mean_delta"], "descriptive mean_delta_pp"),
        (d["metric"] == METRIC_DISEASE_ONLY, "descriptive metric"),
        (d["alignment_status"] in ("ok", "ok_with_warnings"), "descriptive alignment_status"),
        (math.isfinite(b) and b > 0.0, "baseline_dataset_miou > 0"),
        (ni["observed_delta"] == c - b, "observed_delta == candidate - baseline"),
        (ni["observed_drop"] == b - c, "observed_drop == baseline - candidate"),
        (ni["observed_drop"] == -ni["observed_delta"], "observed_drop == -observed_delta"),
        (b > 0.0 and ni["relative_retention"] == c / b, "relative_retention == candidate/baseline"),
        (ni["passed"] is (ni["lower_bound"] > -ni["primary_margin"]), "strict primary decision"),
        (all(s["passed"] is (ni["lower_bound"] > -s["margin"]) for s in ni["sensitivity"]),
         "sensitivity decisions from the one stored bound"),
        (e6["observed_drop"] == ni["observed_drop"], "e6_kd_trigger.observed_drop"),
        (e6["triggered"] is (e6["observed_drop"] > e6["trigger_threshold"]), "strict E6-KD trigger")]
    bad = [what for ok, what in identities if not ok]
    if bad:
        _fail(C_SOURCE, f"section 12.4.8-12.4.10 identities or decisions violated: {bad}")


def _check_officiality(ctx: _Ctx) -> None:
    """Check 11 (officiality_policy_verified): sections 10, 10.1, 12.4.2, 12.4.3, 12.4.5 -- run_id
    pattern and run_id == directory basename, created_at_utc format; artifact_status vocabulary and
    status <=> warnings; the TEST-manifest binding (an official artifact is refused while unbound,
    P2); warnings vocabulary, no duplicates, canonical order; the software block self-consistent and
    pinned == PINNED_ENVIRONMENT; one policy across the nine, official or nonofficial_smoke (P3); the
    re-derived warnings: analysis_id, software, input statuses, per-task replicate counts and
    jackknife counts. With the contract: PINNED_ENVIRONMENT equals the section 10.1 literals. With the
    inputs: synthetic_input_data equals what the dataset names say (P6) and the jackknife warning is
    re-derived with the inputs' k."""
    fam = ctx.fam
    ws = fam["warnings"]
    if not re.fullmatch(RUN_ID_PATTERN, fam["run_id"]) or fam["run_id"] != ctx.final_name:
        _fail(C_POLICY, f"run_id {fam['run_id']!r} violates the pattern or differs from the "
                        f"directory name {ctx.final_name!r}")
    if not _valid_timestamp(fam["created_at_utc"]):
        _fail(C_POLICY, f"created_at_utc {fam['created_at_utc']!r} is not {TIMESTAMP_FORMAT}")
    if fam["artifact_status"] not in ARTIFACT_STATUSES:
        _fail(C_POLICY, f"artifact_status {fam['artifact_status']!r}")
    if (any(w not in _W_RANK for w in ws) or len(set(ws)) != len(ws)
            or ws != sorted(ws, key=lambda w: _W_RANK.get(w, -1))):
        _fail(C_POLICY, f"warnings {ws} violate the frozen vocabulary, uniqueness or order")
    if derive_artifact_status(tuple(ws)) != fam["artifact_status"]:
        _fail(C_POLICY, "artifact_status is not derivable from warnings")
    if fam["artifact_status"] == STATUS_OFFICIAL:
        _fail(C_POLICY, "an official statistics artifact is refused while the TEST-manifest binding "
                        f"is {TEST_MANIFEST_BINDING!r} (P2)")
    sw = fam["software_environment"]
    matches = sw["observed"] == sw["pinned"]
    if sw["pinned"] != PINNED_ENVIRONMENT or sw["matches_pinned"] is not matches:
        _fail(C_POLICY, f"software_environment is not self-consistent: {sw}")
    pols = [c["policy"] for c in fam["a3a_family"]["comparisons"]] + [
        fam["descriptive_e1_e3"]["policy"]]
    if any(p not in ARTIFACT_POLICIES for p in pols) or len(set(pols)) != 1:
        _fail(C_POLICY, f"policies {sorted(set(pols))}: one of {list(ARTIFACT_POLICIES)} is required "
                        "across the eight comparisons and the descriptive block (P3)")
    k = (ctx.views.am5_excluded_count if ctx.views is not None
         else ctx.n_dataset - ctx.n_per_image)
    tasks = fam["bootstrap_tasks"]
    derived = {
        W_ANALYSIS_ID: fam["analysis_id"] != ANALYSIS_ID_OFFICIAL,
        W_SOFTWARE: not sw["matches_pinned"],
        W_INPUT: any(a["artifact_status"] != STATUS_OFFICIAL for a in fam["input_artifacts"]),
        W_B: any(t["bootstrap_replicates"] != PRODUCTION_B for t in tasks),
        W_JACKKNIFE: (k < 0 or k >= OFFICIAL_JACKKNIFE_N or any(
            t["jackknife_count"] != expected_jackknife_count(t["task_id"], k) for t in tasks))}
    if ctx.views is not None:
        derived[W_SYNTHETIC] = synthetic_from_inputs(ctx.views)[0]
    bad = [w for w, present in derived.items() if present is not (w in ws)]
    if bad:
        _fail(C_POLICY, f"warning(s) {bad} do not match their re-derivation from the artifact"
                        f"{' and its inputs' if ctx.views is not None else ''}")
    if ctx.contract_bytes is not None:
        try:
            require_contract_pins(ctx.contract_bytes)
        except ArtifactError as e:
            _fail(C_POLICY, str(e))


def _check_contract_hash(ctx: _Ctx) -> None:
    """Check 2 (statistical_contract_hash_verified): section 12.4.4 -- 64 lowercase hex; with the
    contract bytes, their SHA-256 (raw bytes, never re-encoded) equals the recorded digest."""
    rec = ctx.fam["statistical_contract_sha256"]
    if not _HEX64.fullmatch(rec):
        _fail(C_CONTRACT, f"statistical_contract_sha256 {rec!r} is not 64 lowercase hex")
    if ctx.contract_bytes is not None:
        _require_contract_hash(rec, ctx.contract_bytes)


def _relative_posix(p) -> bool:
    parts = p.split("/")
    return (bool(p) and not p.startswith("/") and "\\" not in p and ":" not in p
            and all(x not in ("", ".", "..") for x in parts))


def _check_input_records(ctx: _Ctx) -> None:
    """Check 1 (input_artifact_provenance_verified), artifact part: sections 12.1, 12.4.6 -- exactly the
    37 identities of section 12.4.6, unique repository-relative POSIX paths in ascending order, input
    status vocabulary, a null checkpoint only on a smoke record (Q13), 64-hex digests and a 40-hex
    commit. With the inputs: `check_input_provenance` (records equal the inputs; P4 row checks; P5
    snapshots; canvas; P7; O1)."""
    from . import driver as D
    recs = ctx.fam["input_artifacts"]
    paths = [r["repo_relative_path"] for r in recs]
    if len(recs) != 37 or paths != sorted(paths) or len(set(paths)) != len(paths):
        _fail(C_PROVENANCE, f"{len(recs)} records; 37 unique records sorted by repo_relative_path "
                            "are required")
    try:
        D.require_inventory([(r["stage"], D.condition_tuple(r["condition"])) for r in recs])
    except D.DriverRefusal as e:
        _fail(C_PROVENANCE, str(e))
    for r in recs:
        rel = r["repo_relative_path"]
        ckpt = r["checkpoint_sha256"]
        problems = [(_relative_posix(rel), "repo_relative_path is not relative POSIX"),
                    (r["artifact_status"] in INPUT_STATUSES, "artifact_status vocabulary"),
                    ((ckpt is None and r["artifact_status"] == "smoke")
                     or (ckpt is not None and bool(_HEX64.fullmatch(ckpt))), "checkpoint_sha256"),
                    (all(_HEX64.fullmatch(r[k]) for k in (
                        "split_manifest_sha256", "class_map_sha256", "metric_impl_sha256",
                        "config_sha256")), "digests"),
                    (bool(_HEX40.fullmatch(r["repo_commit"])), "repo_commit")]
        bad = [what for ok, what in problems if not ok]
        if bad:
            _fail(C_PROVENANCE, f"{rel}: {bad}")
    if ctx.input_root is not None:
        pol = ctx.fam["a3a_family"]["comparisons"][0]["policy"]
        if pol not in ARTIFACT_POLICIES:
            _fail(C_POLICY, f"policy {pol!r} is not one of {list(ARTIFACT_POLICIES)} (P3)")
        ctx.views = load_input_views(recs, ctx.input_root, Policy(pol))
        check_input_provenance(recs, ctx.views, expected_sha256=ctx.expected_input_sha256)


def _recompute_from_inputs(ctx: _Ctx) -> None:
    """O4: the input parts of checks 3, 4 and 6 -- everything recomputed from the 37 inputs, no
    bootstrap draws, exact equality."""
    from . import driver as D
    fam = ctx.fam
    try:
        obs = D.assemble_observed(ctx.views.loaded, ctx.views.policy)
    except Exception as e:                                             # noqa: BLE001
        _fail(C_A3A, f"the family cannot be recomputed from the inputs: {type(e).__name__}: {e}")
    stored = fam["a3a_family"]["comparisons"]
    again = [serialize_a3a(r) for r in obs.results]
    if again != stored:
        diff = [(a["comparison_id"], k) for a, b in zip(again, stored) for k in a if a[k] != b[k]]
        _fail(C_A3A, f"comparisons recomputed from the inputs differ (comparison, field): {diff[:4]}")
    if serialize_a3a(obs.holm) != fam["a3a_family"]["holm"]:
        _fail(C_A3A, "the Holm family recomputed from the inputs differs")
    for t in fam["bootstrap_tasks"]:
        if TASK_BY_ID[t["task_id"]].statistic_name == DATASET_MIOU_DELTA:
            n = obs.pooled[t["comparison_id"]].n_images
            if t["jackknife_count"] != n:
                _fail(C_MATRIX, f"{t['task_id']}: jackknife count {t['jackknife_count']} != the "
                                f"pooled image count {n} of the inputs")
    values = D.observed_values(obs)
    bad = [t["task_id"] for t in fam["bootstrap_tasks"] if values[t["task_id"]] != t["observed"]]
    if bad:
        _fail(C_SOURCE, f"observed values recomputed from the inputs differ: {bad[:4]}")
    if _descriptive_block(obs.descriptive) != fam["descriptive_e1_e3"]:
        _fail(C_SOURCE, "the descriptive block recomputed from the inputs differs")
    b, c = obs.pooled[NONINFERIORITY_E3_E6].stage_mious()
    ni = fam["noninferiority_e3_e6"]
    if (b, c) != (ni["baseline_dataset_miou"], ni["candidate_dataset_miou"]):
        _fail(C_SOURCE, "the E3/E6 pooled dataset mIoU recomputed from the inputs differs")
    ctx.observed = obs


def verify_statistics_artifact(path, *, input_root=None, contract_bytes: bytes | None = None,
                               final_name: str | None = None,
                               expected_input_sha256: Mapping | None = None) -> VerifyReport:
    """Re-establish the thirteen integrity checks from disk.

    `input_root` (the directory the records' repo_relative_path resolve against) adds the input
    parts of checks 1, 3, 4, 6 and 11; `contract_bytes` those of checks 2 and 11. Without them those
    checks are reported in `not_reestablished`. `final_name` is the directory name run_id must equal
    (the writer verifies its temporary directory under the final name). Raises IntegrityError naming
    the first check that cannot be established.
    """
    path = Path(path)
    ctx = _Ctx(path=path, final_name=final_name or path.name,
               input_root=Path(input_root) if input_root is not None else None,
               contract_bytes=None if contract_bytes is None else bytes(contract_bytes),
               expected_input_sha256=expected_input_sha256)
    for step in (_check_exact_file_set, _check_manifest, _check_npz_schema, _check_finite_values,
                 _check_task_order, _check_task_matrix, _check_json_npz, _check_seed_material,
                 _check_a3a_family, _check_observed_sources, _check_contract_hash,
                 _check_input_records):
        step(ctx)
    if ctx.views is not None:
        _recompute_from_inputs(ctx)
    _check_officiality(ctx)
    skipped = set()
    if ctx.views is None:
        skipped |= {C_PROVENANCE, C_A3A, C_MATRIX, C_SOURCE, C_POLICY}
    if ctx.contract_bytes is None:
        skipped |= {C_CONTRACT, C_POLICY}
    return VerifyReport(established=tuple(c for c in INTEGRITY_CHECKS if c not in skipped),
                        not_reestablished=tuple(c for c in INTEGRITY_CHECKS if c in skipped),
                        family=ctx.fam, views=ctx.views)


__all__ = [
    "ArtifactError", "IntegrityError", "WriterGateRefused", "FAMILY_JSON", "BOOTSTRAP_NPZ",
    "MANIFEST_NAME", "TOP_LEVEL_KEYS", "INPUT_ARTIFACT_FIELDS", "INTEGRITY_CHECKS", "NPZ_SUFFIXES",
    "OFFICIALITY_WARNINGS", "ARTIFACT_STATUSES", "PINNED_ENVIRONMENT", "RUN_ID_PATTERN",
    "TIMESTAMP_FORMAT", "ALPHA", "CODE_REPO", "OFFICIAL_CODE_PATHS", "TEST_MANIFEST_BINDING",
    "ARTIFACT_POLICIES", "BINDING_CONDITION", "ArtifactInputs", "InputViews", "VerifyReport",
    "serialize_a3a", "observed_environment", "software_environment_block", "require_k",
    "expected_jackknife_count", "officiality_warnings", "derive_artifact_status",
    "contract_sha256", "contract_pins", "require_contract_pins", "build_family", "canonical_json",
    "strict_json", "load_input_views", "check_input_provenance", "synthetic_from_inputs",
    "code_provenance_unmet", "official_conditions", "check_writer_gate",
    "unmet_official_conditions", "stale_temp_dirs", "preflight_out_dir",
    "write_statistics_artifact", "verify_statistics_artifact",
]
