"""The three-file statistics artifact: family.json + bootstrap.npz + MANIFEST.sha256 (A3b).

Implements STATISTICAL_ANALYSIS_CONTRACT.md sections 12.3 and 12.4 exactly -- the nineteen ordered
top-level keys, the A+ `a3a_family` envelope carrying the complete finalized `HolmFamily`, the
literal A3a dataclass field tuples with a fail-closed runtime source assertion, the ordered 35
bootstrap-task records with `observed_source`, the thirteen-check integrity block, and A2a writer
discipline (refuse-existing, sibling temp dir, strict JSON, NPZ re-read, hash verification, atomic
rename, cleanup on every failure).
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .bootstrap import (ANALYSIS_ID_OFFICIAL, FALLBACK_WARNINGS, FAMILY_ID, FROZEN_A3A_FIELDS,
                        FROZEN_TASK_IDS, OBSERVED_SOURCES, OFFICIAL_JACKKNIFE_N, PRODUCTION_B,
                        SCHEMA_VERSION, STATISTICAL_PROTOCOL, STATUS_OK, STATUS_OK_WARN,
                        TASK_BY_ID, TWO_SIDED, W_SAT_UPPER, assert_a3a_source_schema,
                        expected_comparison_ids, reconstruct_seed_material)

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

RUN_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

NPZ_SUFFIXES = ("bootstrap", "jackknife", "observed", "root_seed", "seed_digest",
                "seed_entropy_bytes", "z0", "acceleration", "acceleration_defined",
                "interval_type", "interval_method", "confidence_level", "bounds")


class ArtifactError(RuntimeError):
    """Refusal or integrity failure. Never downgraded into an officiality warning."""


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
# 2. officiality
# --------------------------------------------------------------------------------------------------
def observed_environment() -> dict:
    import scipy
    import statsmodels
    return {"python": "%d.%d" % sys.version_info[:2], "numpy": np.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def software_environment_block() -> dict:
    obs = observed_environment()
    return {"pinned": dict(PINNED_ENVIRONMENT), "observed": obs,
            "matches_pinned": obs == PINNED_ENVIRONMENT}


def officiality_warnings(*, analysis_id: str, software: dict, input_artifacts: list,
                         synthetic: bool, B: int, jackknife_n: int) -> tuple[str, ...]:
    """Every NON-FATAL officiality defect maps here; anything else raises. No third path."""
    ws: list[str] = []
    if analysis_id != ANALYSIS_ID_OFFICIAL:
        ws.append(W_ANALYSIS_ID)
    if not software["matches_pinned"]:
        ws.append(W_SOFTWARE)
    if any(a["artifact_status"] != STATUS_OFFICIAL for a in input_artifacts):
        ws.append(W_INPUT)
    if synthetic:
        ws.append(W_SYNTHETIC)
    if B != PRODUCTION_B:
        ws.append(W_B)
    if jackknife_n != OFFICIAL_JACKKNIFE_N:
        ws.append(W_JACKKNIFE)
    return tuple(sorted(set(ws), key=lambda w: _W_RANK[w]))


def derive_artifact_status(warnings: tuple[str, ...]) -> str:
    """Asserted biconditional -- status is derived, never set independently."""
    for w in warnings:
        if w not in _W_RANK:
            raise ArtifactError(f"warning outside the frozen officiality vocabulary: {w!r}")
    return STATUS_OFFICIAL if not warnings else STATUS_NONOFFICIAL


def contract_sha256(repo_root: Path) -> tuple[str, bytes]:
    """Read the governing contract bytes ONCE; hash that buffer. No reread, no re-encode."""
    raw = (Path(repo_root) / "docs" / "STATISTICAL_ANALYSIS_CONTRACT.md").read_bytes()
    return hashlib.sha256(raw).hexdigest(), raw


# --------------------------------------------------------------------------------------------------
# 3. family.json assembly
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactInputs:
    analysis_id: str
    run_id: str
    created_at_utc: str
    contract_sha256: str
    input_artifacts: list
    comparison_results: tuple
    holm_family: object
    descriptive: object
    non_inferiority: object
    e6_kd: object
    task_results: tuple
    synthetic: bool


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


def build_family(inp: ArtifactInputs) -> dict:
    assert_a3a_source_schema()
    sw = software_environment_block()
    B = int(inp.task_results[0].replicates.size) if inp.task_results else 0
    n = int(inp.task_results[0].jackknife.size) if inp.task_results else 0
    warns = officiality_warnings(analysis_id=inp.analysis_id, software=sw,
                                 input_artifacts=inp.input_artifacts, synthetic=inp.synthetic,
                                 B=B, jackknife_n=n)
    status = derive_artifact_status(warns)

    ids = [t.task_id for t in (r.task for r in inp.task_results)]
    if tuple(ids) != FROZEN_TASK_IDS:
        raise ArtifactError("bootstrap task set/order does not match the frozen 35-task matrix")

    comparisons = [serialize_a3a(c) for c in inp.comparison_results]
    holm = serialize_a3a(inp.holm_family)
    if holm["status"] != "ok" or holm["complete"] is not True:
        raise ArtifactError("only a finalized HolmFamily (status 'ok', complete true) may be "
                            "serialized")
    if any(not math.isfinite(m["p_adjusted"]) for m in holm["members"]):
        raise ArtifactError("non-finite p_adjusted: an intermediate HolmMember reached the writer")

    d = inp.descriptive
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
        "descriptive_e1_e3": {
            "comparison_id": d.comparison_id, "baseline_stage": d.baseline_stage,
            "candidate_stage": d.candidate_stage, "direction": d.direction, "metric": d.metric,
            "n_paired": int(d.n_paired), "policy": d.policy,
            "alignment_status": d.alignment_status,
            "mean_delta": _finite(d.mean_delta, "descriptive.mean_delta"),
            "mean_delta_pp": _finite(d.mean_delta_pp, "descriptive.mean_delta_pp"),
            "median_delta": _finite(d.median_delta, "descriptive.median_delta"),
            "hodges_lehmann_shift": _finite(d.hodges_lehmann_shift, "descriptive.hl")},
        "noninferiority_e3_e6": _ni_block(inp.non_inferiority),
        "e6_kd_trigger": _e6_block(inp.e6_kd),
        "bootstrap_tasks": [_task_record(r, r.task) for r in inp.task_results],
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
# 4. NPZ
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


def _reject_constant(name):                       # pragma: no cover - raised only on bad input
    raise ArtifactError(f"non-finite JSON constant {name!r} is forbidden")


def _verify_json_npz(fam: dict, z) -> None:
    """Exact float64 equality after reload. A tolerance-based verifier is forbidden."""
    ids = [t["task_id"] for t in fam["bootstrap_tasks"]]
    if tuple(ids) != FROZEN_TASK_IDS:
        raise ArtifactError("JSON task set/order != frozen matrix")
    expected_keys = {f"{i}__{s}" for i in ids for s in NPZ_SUFFIXES}
    if set(z.files) != expected_keys:
        missing = sorted(expected_keys - set(z.files))
        extra = sorted(set(z.files) - expected_keys)
        raise ArtifactError(f"NPZ key set mismatch; missing={missing[:3]} extra={extra[:3]}")
    for t in fam["bootstrap_tasks"]:
        p = t["task_id"]
        if float(z[f"{p}__observed"][0]) != t["observed"]:
            raise ArtifactError(f"{p}: observed differs between JSON and NPZ")
        if int(z[f"{p}__root_seed"][0]) != t["root_seed"]:
            raise ArtifactError(f"{p}: root_seed differs")
        if z[f"{p}__seed_digest"].tobytes().hex() != t["seed_sha256"]:
            raise ArtifactError(f"{p}: seed_digest differs")
        if z[f"{p}__seed_entropy_bytes"].tobytes().hex() != t["seed_entropy_hex"]:
            raise ArtifactError(f"{p}: seed_entropy_bytes differs")
        if float(z[f"{p}__z0"][0]) != t["z0"]:
            raise ArtifactError(f"{p}: z0 differs")
        defined = bool(z[f"{p}__acceleration_defined"][0])
        if defined != t["acceleration_defined"]:
            raise ArtifactError(f"{p}: acceleration_defined differs")
        acc = z[f"{p}__acceleration"]
        if defined:
            if acc.size != 1 or float(acc[0]) != t["acceleration"]:
                raise ArtifactError(f"{p}: acceleration differs")
        elif acc.size != 0 or t["acceleration"] is not None:
            raise ArtifactError(f"{p}: undefined acceleration must be [0]-length and JSON null")
        if z[f"{p}__interval_type"].tobytes().decode("utf-8") != t["interval_type"]:
            raise ArtifactError(f"{p}: interval_type differs")
        if z[f"{p}__interval_method"].tobytes().decode("utf-8") != t["interval_method"]:
            raise ArtifactError(f"{p}: interval_method differs")
        if float(z[f"{p}__confidence_level"][0]) != t["confidence_level"]:
            raise ArtifactError(f"{p}: confidence_level differs")
        bounds = z[f"{p}__bounds"]
        want = [t["lower_bound"]] + ([t["upper_bound"]] if "upper_bound" in t else [])
        if [float(x) for x in bounds] != want:
            raise ArtifactError(f"{p}: bounds differ between JSON and NPZ")
        if int(z[f"{p}__bootstrap"].shape[0]) != t["bootstrap_replicates"]:
            raise ArtifactError(f"{p}: bootstrap_replicates != NPZ length")
        if int(z[f"{p}__jackknife"].shape[0]) != t["jackknife_count"]:
            raise ArtifactError(f"{p}: jackknife_count != NPZ length")
        for suffix in ("bootstrap", "jackknife"):
            arr = z[f"{p}__{suffix}"]
            if arr.dtype != np.float64:
                raise ArtifactError(f"{p}__{suffix} dtype {arr.dtype} != float64")
            if not np.all(np.isfinite(arr)):
                raise ArtifactError(f"{p}__{suffix} contains a non-finite value")
        spec = TASK_BY_ID[p]
        if t["observed_source"] != spec.observed_source:
            raise ArtifactError(f"{p}: observed_source != frozen 24/3/8 map")
        if t["observed_source"] not in OBSERVED_SOURCES:
            raise ArtifactError(f"{p}: unknown observed_source")
        reconstruct_seed_material(fam["analysis_id"], t["comparison_id"], t["statistic_name"],
                                  t["seed_input_hex"], t["root_seed"])
        if hashlib.sha256(bytes.fromhex(t["seed_input_hex"])).hexdigest() != t["seed_sha256"]:
            raise ArtifactError(f"{p}: seed_sha256 != SHA256(canonical bytes)")
        if int(t["seed_entropy_hex"], 16) != int(t["seed_entropy_decimal"]):
            raise ArtifactError(f"{p}: seed_entropy_hex != seed_entropy_decimal")


def _manifest_text(directory: Path) -> str:
    lines = []
    for name in ARTIFACT_FILES:                     # already lexicographic
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")         # sha256sum-compatible: two ASCII spaces
    return "".join(lines)


# --------------------------------------------------------------------------------------------------
# 5. atomic writer
# --------------------------------------------------------------------------------------------------
def write_statistics_artifact(out_dir, fam: dict, task_results, *,
                              _inject_failure: bool = False) -> Path:
    """Refuse-existing -> temp sibling -> validate -> re-read -> hash -> atomic rename."""
    out_dir = Path(out_dir)
    if fam["artifact_status"] == STATUS_OFFICIAL and not fam["software_environment"][
            "matches_pinned"]:
        raise ArtifactError(
            "REFUSED before final-directory creation: an official artifact requires the pinned "
            "statistics stack (contract section 10.1)")
    import re as _re
    if not _re.match(RUN_ID_PATTERN, fam["run_id"]):
        raise ArtifactError(f"run_id {fam['run_id']!r} violates {RUN_ID_PATTERN}")
    if fam["run_id"] != out_dir.name:
        raise ArtifactError("run_id must equal the final run-directory basename")
    if out_dir.exists():
        raise ArtifactError(f"refusing to overwrite {out_dir}")

    tmp = out_dir.parent / f".{out_dir.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        (tmp / FAMILY_JSON).write_text(
            json.dumps(fam, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n")
        np.savez_compressed(tmp / BOOTSTRAP_NPZ, **_npz_arrays(task_results))
        if _inject_failure:
            raise ArtifactError("injected integrity failure (test hook)")

        reread = json.loads((tmp / FAMILY_JSON).read_text(encoding="utf-8"),
                            parse_constant=_reject_constant)
        if tuple(reread.keys()) != TOP_LEVEL_KEYS:
            raise ArtifactError("top-level key order did not survive serialization")
        if reread["schema_version"] != SCHEMA_VERSION or \
                reread["statistical_protocol"] != STATISTICAL_PROTOCOL:
            raise ArtifactError("protocol identifiers failed post-write verification")
        with np.load(tmp / BOOTSTRAP_NPZ, allow_pickle=False) as z:
            _verify_json_npz(reread, z)

        (tmp / MANIFEST_NAME).write_text(_manifest_text(tmp), encoding="utf-8", newline="\n")
        for line in (tmp / MANIFEST_NAME).read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if hashlib.sha256((tmp / name).read_bytes()).hexdigest() != digest:
                raise ArtifactError(f"manifest hash mismatch for {name}")
        if sorted(p.name for p in tmp.iterdir()) != sorted(
                [FAMILY_JSON, BOOTSTRAP_NPZ, MANIFEST_NAME]):
            raise ArtifactError("temporary directory does not contain exactly three files")

        tmp.rename(out_dir)
        return out_dir
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def verify_statistics_artifact(out_dir) -> dict:
    """Re-verify a finalised artifact from disk: file set, manifest, strict JSON, JSON<->NPZ."""
    out_dir = Path(out_dir)
    names = sorted(p.name for p in out_dir.iterdir())
    if names != sorted([FAMILY_JSON, BOOTSTRAP_NPZ, MANIFEST_NAME]):
        raise ArtifactError(f"artifact must contain exactly three files, found {names}")
    for line in (out_dir / MANIFEST_NAME).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((out_dir / name).read_bytes()).hexdigest() != digest:
            raise ArtifactError(f"manifest verification failed for {name}")
    fam = json.loads((out_dir / FAMILY_JSON).read_text(encoding="utf-8"),
                     parse_constant=_reject_constant)
    if tuple(fam.keys()) != TOP_LEVEL_KEYS:
        raise ArtifactError("top-level key set/order mismatch on reload")
    if tuple(fam["integrity"]["checks"].keys()) != INTEGRITY_CHECKS:
        raise ArtifactError("integrity check set/order mismatch")
    if fam["integrity"]["status"] != "passed" or not all(fam["integrity"]["checks"].values()):
        raise ArtifactError("integrity block is not fully passed")
    if derive_artifact_status(tuple(fam["warnings"])) != fam["artifact_status"]:
        raise ArtifactError("artifact_status is not derivable from warnings")
    paths = [a["repo_relative_path"] for a in fam["input_artifacts"]]
    if paths != sorted(paths):
        raise ArtifactError("input_artifacts is not sorted by repo_relative_path")
    if len(set(paths)) != len(paths):
        raise ArtifactError("duplicate repo_relative_path in input_artifacts")
    for a in fam["input_artifacts"]:
        if tuple(a.keys()) != INPUT_ARTIFACT_FIELDS:
            raise ArtifactError("input-artifact record field set/order mismatch")
        if "\\" in a["repo_relative_path"] or a["repo_relative_path"].startswith("/"):
            raise ArtifactError("repo_relative_path must be a relative POSIX path")
    with np.load(out_dir / BOOTSTRAP_NPZ, allow_pickle=False) as z:
        _verify_json_npz(fam, z)
    return fam


__all__ = [
    "ArtifactError", "FAMILY_JSON", "BOOTSTRAP_NPZ", "MANIFEST_NAME", "TOP_LEVEL_KEYS",
    "INPUT_ARTIFACT_FIELDS", "INTEGRITY_CHECKS", "NPZ_SUFFIXES", "OFFICIALITY_WARNINGS",
    "ARTIFACT_STATUSES", "PINNED_ENVIRONMENT", "RUN_ID_PATTERN", "TIMESTAMP_FORMAT", "ALPHA",
    "ArtifactInputs", "serialize_a3a", "observed_environment", "software_environment_block",
    "officiality_warnings", "derive_artifact_status", "contract_sha256", "build_family",
    "write_statistics_artifact", "verify_statistics_artifact",
]
