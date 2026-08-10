"""Read-only ingestion of a completed evaluation run directory (A3a).

Consumes the frozen `plantseg-eval/1.0.0` artifact produced by `src/eval/artifacts.py` and
exposes immutable typed records for the statistical layer. Implements
docs/STATISTICAL_ANALYSIS_CONTRACT.md section 10 (official integrity policy).

Nothing here computes a metric: every value is read from the artifact, and the sufficient
statistics are re-checked against the recorded dataset totals rather than recomputed from pixels.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np

# The evaluator's own public verifier is side-effect-free, so it is PREFERRED over a local
# re-implementation (A3a instruction section 1). We layer three extra format checks on top that
# it does not make: lowercase-64-hex digests, manifest filename uniqueness, and rejection of an
# unexpected payload substituted for an expected one.
from ..eval.artifacts import (ARTIFACT_FILES, MANIFEST_NAME, METRIC_PROTOCOL,  # noqa: F401
                              SCHEMA_VERSION, verify_artifact)

EXPECTED_ROWS_TEST = 1561
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

STATUS_OK = "ok"
UNDEFINED_STATUSES = ("undefined_no_eligible_class", "excluded_data_integrity", "evaluation_error")


class Policy(Enum):
    """Explicit named policy boundary -- never an ambiguous `strict=True` flag."""
    OFFICIAL = "official"
    NONOFFICIAL_SMOKE = "nonofficial_smoke"


class IngestError(RuntimeError):
    """Artifact is unusable for the requested policy. Always fatal."""


# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationRunIdentity:
    """Everything alignment needs to prove two runs are comparable."""
    schema_version: str
    metric_protocol: str
    artifact_status: str
    stage: str
    model_role: str
    precision: str
    random_init: bool
    split: str
    condition_type: str
    condition_name: str | None
    condition_severity: int | None
    expected_rows: int
    actual_rows: int
    num_classes: int
    background_index: int
    ignore_index: int
    split_manifest_sha256: str
    class_map_sha256: str
    preprocess_protocol: str
    metric_impl_sha256: str
    config_sha256: str
    checkpoint_sha256: str | None
    repo_commit: str
    run_id: str
    artifact_dir: str


@dataclass(frozen=True)
class PerImageRecord:
    image_id: str
    clean_image_id: str
    manifest_index: int
    all_class_miou: float | None
    all_class_miou_status: str
    disease_only_miou: float | None
    disease_only_miou_status: str
    n_eligible_all_class: int
    n_eligible_disease_only: int


@dataclass(frozen=True)
class PerImageSufficientStats:
    """Sparse per-(image,class) counts, plus the dataset totals they must reproduce."""
    image_index: np.ndarray      # row position into manifest_ids
    class_id: np.ndarray
    tp: np.ndarray
    gt: np.ndarray
    pred: np.ndarray
    dataset_tp: np.ndarray
    dataset_gt: np.ndarray
    dataset_pred: np.ndarray
    manifest_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationRun:
    identity: EvaluationRunIdentity
    records: tuple[PerImageRecord, ...]        # canonical order: ascending manifest_index
    stats: PerImageSufficientStats
    policy: Policy

    def by_id(self) -> Mapping[str, PerImageRecord]:
        return {r.image_id: r for r in self.records}

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(r.image_id for r in self.records)


# --------------------------------------------------------------------------------------------------
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_run_manifest(run_dir: Path) -> dict:
    """Verify the artifact via the evaluator's public verifier, then add A3a format checks."""
    run_dir = Path(run_dir)
    try:
        summary = verify_artifact(run_dir)          # filenames, hashes, strict-JSON summary
    except Exception as e:                          # noqa: BLE001
        raise IngestError(f"artifact verification failed for {run_dir.name}: "
                          f"{type(e).__name__}: {e}") from e

    lines = (run_dir / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    if len(lines) != len(ARTIFACT_FILES):
        raise IngestError(f"manifest must list exactly {len(ARTIFACT_FILES)} payloads, "
                          f"got {len(lines)}")
    names, digests = [], []
    for ln in lines:
        digest, _, name = ln.partition("  ")
        if not _HEX64.match(digest):
            raise IngestError(f"manifest digest is not lowercase 64-hex: {digest!r}")
        names.append(name)
        digests.append(digest)
    if len(set(names)) != len(names):
        raise IngestError(f"manifest filenames are not unique: {names}")
    if sorted(names) != sorted(ARTIFACT_FILES):
        raise IngestError(f"manifest lists {names}, expected {list(ARTIFACT_FILES)}")
    for name, digest in zip(names, digests):
        if _sha256_file(run_dir / name) != digest:
            raise IngestError(f"payload {name} does not match its manifest digest")
    return summary


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise IngestError(msg)


def load_run(run_dir, policy: Policy, *, expect_condition: tuple[str, int | None] | None = None
             ) -> EvaluationRun:
    """Load and validate one evaluation run under an explicit policy."""
    run_dir = Path(run_dir)
    summary = verify_run_manifest(run_dir)

    _require(summary.get("schema_version") == SCHEMA_VERSION,
             f"schema_version must be {SCHEMA_VERSION!r}, got {summary.get('schema_version')!r}")
    _require(summary.get("metric_protocol") == METRIC_PROTOCOL,
             f"metric_protocol must be {METRIC_PROTOCOL!r}")
    for block in ("run", "dataset", "dataset_level", "per_class", "integrity"):
        _require(block in summary, f"summary.json missing block {block!r}")

    run, ds, integ = summary["run"], summary["dataset"], summary["integrity"]
    _require(integ.get("invalid_pred_labels") == 0, "artifact reports invalid_pred_labels != 0")
    for flag in ("row_count_ok", "ids_unique", "ids_match_manifest", "order_canonical"):
        _require(bool(integ.get(flag)), f"artifact integrity flag {flag} is false")

    ident = EvaluationRunIdentity(
        schema_version=summary["schema_version"], metric_protocol=summary["metric_protocol"],
        artifact_status=run["artifact_status"], stage=run["stage"], model_role=run["model_role"],
        precision=run["precision"], random_init=bool(run["random_init"]),
        split=ds["split"], condition_type=ds["condition"]["type"],
        condition_name=ds["condition"]["name"], condition_severity=ds["condition"]["severity"],
        expected_rows=int(ds["expected_rows"]), actual_rows=int(ds["actual_rows"]),
        num_classes=int(ds["num_classes"]), background_index=int(ds["background_index"]),
        ignore_index=int(ds["ignore_index"]),
        split_manifest_sha256=ds["split_manifest_sha256"], class_map_sha256=ds["class_map_sha256"],
        preprocess_protocol=ds["preprocess_protocol"], metric_impl_sha256=run["metric_impl_sha256"],
        config_sha256=run["config_sha256"], checkpoint_sha256=run.get("checkpoint_sha256"),
        repo_commit=run["repo_commit"], run_id=run["run_id"], artifact_dir=run_dir.name)

    # ---- policy gate (contract section 10) ----
    if policy is Policy.OFFICIAL:
        _require(ident.artifact_status == "official",
                 f"OFFICIAL analysis requires artifact_status='official', got "
                 f"{ident.artifact_status!r}")
        _require(ident.random_init is False, "OFFICIAL analysis forbids random_init inputs")
        _require(ident.split == "test", f"OFFICIAL clean analysis requires split='test'")
        _require(ident.expected_rows == ident.actual_rows == EXPECTED_ROWS_TEST,
                 f"OFFICIAL analysis requires {EXPECTED_ROWS_TEST} rows, got "
                 f"expected={ident.expected_rows} actual={ident.actual_rows}")
    else:
        _require(ident.artifact_status in ("smoke", "provisional", "official"),
                 f"unknown artifact_status {ident.artifact_status!r}")
    _require(ident.expected_rows == ident.actual_rows,
             f"expected_rows {ident.expected_rows} != actual_rows {ident.actual_rows}")

    if expect_condition is not None:
        name, sev = expect_condition
        _require(ident.condition_name == name and ident.condition_severity == sev,
                 f"condition mismatch: artifact has ({ident.condition_name!r}, "
                 f"{ident.condition_severity!r}), expected ({name!r}, {sev!r})")

    # ---- per-image rows ----
    rows = []
    for line in (run_dir / "per_image.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
            IngestError(f"non-finite JSON constant {x!r} in per_image.jsonl")))
        rows.append(PerImageRecord(
            image_id=o["image_id"], clean_image_id=o["clean_image_id"],
            manifest_index=int(o["manifest_index"]),
            all_class_miou=o["all_class_miou"], all_class_miou_status=o["all_class_miou_status"],
            disease_only_miou=o["disease_only_miou"],
            disease_only_miou_status=o["disease_only_miou_status"],
            n_eligible_all_class=int(o["n_eligible_all_class"]),
            n_eligible_disease_only=int(o["n_eligible_disease_only"])))

    _require(len(rows) == ident.actual_rows,
             f"per_image.jsonl has {len(rows)} rows, summary says {ident.actual_rows}")
    idxs = [r.manifest_index for r in rows]
    _require(idxs == sorted(idxs), "per_image.jsonl is not sorted by manifest_index")
    ids = [r.image_id for r in rows]
    _require(len(set(ids)) == len(ids), "duplicate image_id in per_image.jsonl")
    _require(len({i.casefold() for i in ids}) == len(ids), "case-folded image_id collision")

    if policy is Policy.OFFICIAL:
        bad = [r.image_id for r in rows if r.disease_only_miou is None]
        _require(not bad, f"OFFICIAL analysis forbids undefined primary metric; {len(bad)} row(s) "
                          f"undefined, e.g. {bad[:3]}")

    # ---- sufficient statistics ----
    with np.load(run_dir / "sufficient_stats.npz", allow_pickle=False) as z:
        need = ("image_index", "class_id", "tp", "gt", "pred",
                "dataset_tp", "dataset_gt", "dataset_pred", "manifest_ids")
        for k in need:
            _require(k in z, f"sufficient_stats.npz missing {k!r}")
        for k in need[:-1]:
            _require(z[k].dtype == np.int64, f"{k} dtype {z[k].dtype} != int64")
        C = ident.num_classes
        for k in ("dataset_tp", "dataset_gt", "dataset_pred"):
            _require(z[k].shape == (C,), f"{k} shape {z[k].shape} != ({C},)")
        mids = tuple(str(x) for x in z["manifest_ids"])
        _require(list(mids) == ids, "manifest_ids do not match per_image.jsonl image_id order")
        for name in ("tp", "gt", "pred"):
            acc = np.zeros(C, dtype=np.int64)
            np.add.at(acc, z["class_id"], z[name])
            _require(np.array_equal(acc, z[f"dataset_{name}"]),
                     f"sparse {name} does not sum to dataset_{name}")
        if z["image_index"].size:
            _require(int(z["image_index"].max()) < len(ids),
                     "image_index out of range for manifest_ids")
        stats = PerImageSufficientStats(
            image_index=z["image_index"].copy(), class_id=z["class_id"].copy(),
            tp=z["tp"].copy(), gt=z["gt"].copy(), pred=z["pred"].copy(),
            dataset_tp=z["dataset_tp"].copy(), dataset_gt=z["dataset_gt"].copy(),
            dataset_pred=z["dataset_pred"].copy(), manifest_ids=mids)
    for arr in (stats.image_index, stats.class_id, stats.tp, stats.gt, stats.pred,
                stats.dataset_tp, stats.dataset_gt, stats.dataset_pred):
        arr.flags.writeable = False          # owned copies, frozen against caller mutation

    return EvaluationRun(identity=ident, records=tuple(rows), stats=stats, policy=policy)
