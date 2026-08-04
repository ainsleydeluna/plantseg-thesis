"""Artifact provenance, eligibility gating and atomic serialisation (A2a).

Implements docs/EVALUATION_CONTRACT.md sections 5 (artifact contract), 6 (identity) and 7/7.1
(official vs smoke). Split from `evaluate.py` so the pure compute core stays free of Git and the
filesystem, and so future teacher / PTQ / QAT / corruption adapters reuse both independently.

Three-phase flow -- the middle phase is the frozen pre-inference gate:

    request  = prepare_artifact_request(...)      # pure construction, no I/O
    prov     = validate_artifact_request(request) # READ-ONLY preflight; raises before inference
    result   = evaluate_model(...)                # (caller) model forward happens only after
    out_dir  = write_artifact(result, request, prov)

`validate_artifact_request` performs no writes and creates no directories: it reads `git status`
and hashes files. Running it before `evaluate_model` is what makes "official mode refuses before
inference" true rather than aspirational.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from . import metrics as _metrics_module
from .evaluate import Condition, DatasetMeta, EvalResult, ManifestEntry, RunMeta

SCHEMA_VERSION = "plantseg-eval/1.0.0"
METRIC_PROTOCOL = "plantseg-metrics/1.0.0"

ARTIFACT_STATUSES = ("official", "provisional", "smoke")
STAGES = ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
MODEL_ROLES = ("teacher", "student")
PRECISIONS = ("fp32", "int8_ptq", "int8_qat")
SPLITS = ("val", "test")

ARTIFACT_FILES = ("per_image.jsonl", "sufficient_stats.npz", "summary.json")  # sorted; manifest excluded
MANIFEST_NAME = "MANIFEST.sha256"

# Contract section 7.1 -- governed prefixes. Dirty/untracked here blocks `official`.
GOVERNED_PREFIXES = (
    "src/", "configs/", "scripts/", "requirements",
    "docs/EVALUATION_CONTRACT.md", "docs/IMPLEMENTATION_CONTRACT.md",
)

# The frozen PlantSeg class space (contract section 0). The writer enforces it: a non-116 result
# must never be serialised as a plantseg-eval/1.0.0 artifact.
FROZEN_NUM_CLASSES = 116
FROZEN_BACKGROUND_INDEX = 0
FROZEN_IGNORE_INDEX = 255

F32_TOL = 1e-6


class ArtifactRequestError(RuntimeError):
    """Raised by the pre-inference gate. Nothing has been written when this is raised."""


class ArtifactWriteError(RuntimeError):
    """Raised during finalisation. The temporary directory is removed before propagating."""


# --------------------------------------------------------------------------------------------------
# canonical hashing
# --------------------------------------------------------------------------------------------------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json_bytes(obj) -> bytes:
    """UTF-8, sorted keys, compact separators, non-finite forbidden."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def hash_split_manifest(entries: Sequence[ManifestEntry]) -> str:
    """`<manifest_index>\\t<image_id>\\t<clean_image_id>\\n` per row, ascending index, UTF-8/LF."""
    lines = []
    for e in sorted(entries, key=lambda x: x.manifest_index):
        lines.append(f"{e.manifest_index}\t{e.image_id}\t{e.clean_image_id}\n")
    return _sha256_bytes("".join(lines).encode("utf-8"))


def hash_class_map(class_map: Sequence[dict]) -> str:
    """Complete ordered per-class descriptor (id + name + role), canonical JSON."""
    return _sha256_bytes(canonical_json_bytes(list(class_map)))


def hash_metric_impl() -> tuple[str, str]:
    """SHA-256 of the raw bytes of the metrics module actually imported."""
    path = Path(_metrics_module.__file__).resolve()
    return _sha256_bytes(path.read_bytes()), str(path)


def git_porcelain_bytes(repo: Path) -> bytes:
    """Raw NUL-delimited porcelain bytes -- stable, not a human-rendered table."""
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", "status", "--porcelain=v1", "-z",
         "--untracked-files=all"],
        cwd=str(repo), capture_output=True, check=False)
    if proc.returncode != 0:
        raise ArtifactRequestError(
            f"git status failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[:200]}")
    return proc.stdout


def parse_porcelain_paths(raw: bytes) -> list[str]:
    """Extract paths from `-z` porcelain. Rename entries carry an extra NUL-separated origin."""
    parts = [p for p in raw.split(b"\x00") if p]
    paths, i = [], 0
    while i < len(parts):
        rec = parts[i].decode("utf-8", "replace")
        if len(rec) < 4:
            i += 1
            continue
        xy, path = rec[:2], rec[3:]
        paths.append(path)
        if "R" in xy or "C" in xy:   # rename/copy: the following record is the source path
            i += 2
        else:
            i += 1
    return paths


def git_commit(repo: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                          capture_output=True, check=False)
    if proc.returncode != 0:
        raise ArtifactRequestError("git rev-parse HEAD failed")
    return proc.stdout.decode().strip()


# --------------------------------------------------------------------------------------------------
# request / provenance
# --------------------------------------------------------------------------------------------------
# RunMeta / DatasetMeta are pure metadata dataclasses defined in `evaluate.py` and re-exported
# here for callers that build a request. Keeping them out of this module lets `src/eval/__init__`
# expose them without importing any filesystem/Git-touching code.
__all__ = ["RunMeta", "DatasetMeta", "ArtifactRequest", "Provenance", "ArtifactRequestError",
           "ArtifactWriteError", "prepare_artifact_request", "validate_artifact_request",
           "write_artifact", "verify_artifact", "canonical_json_bytes"]


@dataclass(frozen=True)
class ArtifactRequest:
    out_dir: Path
    artifact_status: str
    run_id: str
    run: RunMeta
    dataset: DatasetMeta
    expected_manifest: tuple[ManifestEntry, ...]
    class_map: tuple[dict, ...]
    num_classes: int = FROZEN_NUM_CLASSES
    background_index: int = FROZEN_BACKGROUND_INDEX
    ignore_index: int = FROZEN_IGNORE_INDEX
    repo_root: Path = field(default=Path(__file__).resolve().parents[2])


@dataclass(frozen=True)
class Provenance:
    repo_commit: str
    governed_paths_clean: bool
    governed_violations: tuple[str, ...]
    dirty_allowlisted: tuple[str, ...]
    worktree_state_sha256: str
    metric_impl_sha256: str
    metric_impl_path: str
    config_sha256: str
    split_manifest_sha256: str
    class_map_sha256: str
    config_payload: dict


def prepare_artifact_request(*, out_dir, artifact_status, run_id, run, dataset,
                             expected_manifest, class_map,
                             num_classes=FROZEN_NUM_CLASSES,
                             background_index=FROZEN_BACKGROUND_INDEX,
                             ignore_index=FROZEN_IGNORE_INDEX,
                             repo_root=None) -> ArtifactRequest:
    """Pure construction. Performs no I/O and creates nothing."""
    return ArtifactRequest(
        out_dir=Path(out_dir),
        artifact_status=artifact_status,
        run_id=run_id,
        run=run,
        dataset=dataset,
        expected_manifest=tuple(expected_manifest),
        class_map=tuple(class_map),
        num_classes=num_classes,
        background_index=background_index,
        ignore_index=ignore_index,
        repo_root=Path(repo_root) if repo_root else Path(__file__).resolve().parents[2],
    )


def build_config_payload(req: ArtifactRequest) -> dict:
    """Semantic run/evaluation configuration ONLY.

    Deliberately EXCLUDES run_id, timestamps, output paths, metric results, the worktree hash and
    the file-manifest hashes, so the same evaluation configuration hashes identically across runs.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_protocol": METRIC_PROTOCOL,
        "stage": req.run.stage,
        "model_role": req.run.model_role,
        "precision": req.run.precision,
        "quant_backend": req.run.quant_backend,
        "random_init": req.run.random_init,
        "device": req.run.device,
        "dataset_name": req.dataset.name,
        "dataset_doi": req.dataset.doi,
        "split": req.dataset.split,
        "condition": req.dataset.condition.as_dict(),
        "preprocess_protocol": req.dataset.preprocess_protocol,
        "expected_rows": req.dataset.expected_rows,
        "num_classes": req.num_classes,
        "background_index": req.background_index,
        "ignore_index": req.ignore_index,
    }


def validate_artifact_request(req: ArtifactRequest) -> Provenance:
    """PRE-INFERENCE GATE. Read-only: runs `git status`, hashes files, creates nothing.

    Raises ArtifactRequestError for anything knowable before the model is invoked.
    """
    if req.artifact_status not in ARTIFACT_STATUSES:
        raise ArtifactRequestError(
            f"artifact_status must be one of {ARTIFACT_STATUSES}, got {req.artifact_status!r}")
    if req.run.stage not in STAGES:
        raise ArtifactRequestError(f"stage must be one of {STAGES}, got {req.run.stage!r}")
    if req.run.model_role not in MODEL_ROLES:
        raise ArtifactRequestError(f"model_role must be one of {MODEL_ROLES}")
    if req.run.precision not in PRECISIONS:
        raise ArtifactRequestError(f"precision must be one of {PRECISIONS}")
    if req.dataset.split not in SPLITS:
        raise ArtifactRequestError(f"split must be one of {SPLITS}, got {req.dataset.split!r}")

    # frozen class space -- a non-116 result may never become a plantseg-eval/1.0.0 artifact
    if req.num_classes != FROZEN_NUM_CLASSES:
        raise ArtifactRequestError(
            f"{SCHEMA_VERSION} freezes num_classes={FROZEN_NUM_CLASSES}; got {req.num_classes}")
    if req.background_index != FROZEN_BACKGROUND_INDEX or req.ignore_index != FROZEN_IGNORE_INDEX:
        raise ArtifactRequestError(
            f"frozen contract requires background_index={FROZEN_BACKGROUND_INDEX} and "
            f"ignore_index={FROZEN_IGNORE_INDEX}")
    if len(req.class_map) != FROZEN_NUM_CLASSES:
        raise ArtifactRequestError(
            f"class_map must describe all {FROZEN_NUM_CLASSES} classes, got {len(req.class_map)}")

    # checkpoint / random-init consistency
    if req.run.random_init:
        if req.artifact_status != "smoke":
            raise ArtifactRequestError(
                "random_init=True requires artifact_status='smoke' (contract section 5.3)")
        if req.run.checkpoint_path is not None or req.run.checkpoint_sha256 is not None:
            raise ArtifactRequestError(
                "random_init=True requires checkpoint_path and checkpoint_sha256 to be null")
    else:
        if not req.run.checkpoint_path or not req.run.checkpoint_sha256:
            raise ArtifactRequestError(
                "checkpoint_path/checkpoint_sha256 may be null only when random_init=True")

    # manifest sanity + expected row count
    if len(req.expected_manifest) != req.dataset.expected_rows:
        raise ArtifactRequestError(
            f"expected_rows={req.dataset.expected_rows} != len(expected_manifest)="
            f"{len(req.expected_manifest)}")

    # overwrite refusal -- checked before anything is created
    if req.out_dir.exists():
        raise ArtifactRequestError(
            f"refusing to overwrite an existing artifact directory: {req.out_dir} "
            "(overwrite_policy=refuse_existing)")

    # provenance
    raw = git_porcelain_bytes(req.repo_root)
    worktree_sha = _sha256_bytes(raw)
    paths = parse_porcelain_paths(raw)
    violations = tuple(sorted(p for p in paths if any(p.startswith(g) for g in GOVERNED_PREFIXES)))
    allowlisted = tuple(sorted(p for p in paths if p not in violations))
    governed_clean = not violations

    metric_sha, metric_path = hash_metric_impl()
    config_payload = build_config_payload(req)
    prov = Provenance(
        repo_commit=git_commit(req.repo_root),
        governed_paths_clean=governed_clean,
        governed_violations=violations,
        dirty_allowlisted=allowlisted,
        worktree_state_sha256=worktree_sha,
        metric_impl_sha256=metric_sha,
        metric_impl_path=metric_path,
        config_sha256=_sha256_bytes(canonical_json_bytes(config_payload)),
        split_manifest_sha256=hash_split_manifest(req.expected_manifest),
        class_map_sha256=hash_class_map(req.class_map),
        config_payload=config_payload,
    )

    # contract section 7.1 scoped rule -- NOT a whole-repo-clean requirement
    if req.artifact_status == "official":
        if not governed_clean:
            raise ArtifactRequestError(
                "refusing to produce an 'official' artifact: governed paths are dirty or "
                f"untracked -> {list(violations)}. Non-governed paths are allowlisted and do not "
                f"block official status (observed: {list(allowlisted)}).")
        for label in ("checkpoint_sha256",):
            if getattr(req.run, label) is None:
                raise ArtifactRequestError(f"'official' requires {label} to be non-null")
    return prov


# --------------------------------------------------------------------------------------------------
# schema assembly + validation
# --------------------------------------------------------------------------------------------------
def _reject_constant(name):
    raise ArtifactWriteError(f"non-finite JSON constant {name!r} present in artifact")


def build_summary(result: EvalResult, req: ArtifactRequest, prov: Provenance,
                  timestamp_utc: str) -> dict:
    import torch
    try:
        import importlib.metadata as md
        tv = md.version("torchvision")
    except Exception:                                    # noqa: BLE001 -- optional at eval time
        tv = None
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_protocol": METRIC_PROTOCOL,
        "run": {
            "run_id": req.run_id,
            "artifact_status": req.artifact_status,
            "stage": req.run.stage,
            "model_role": req.run.model_role,
            "precision": req.run.precision,
            "quant_backend": req.run.quant_backend,
            "checkpoint_path": req.run.checkpoint_path,
            "checkpoint_sha256": req.run.checkpoint_sha256,
            "random_init": req.run.random_init,
            "repo_commit": prov.repo_commit,
            "governed_paths_clean": prov.governed_paths_clean,
            "dirty_allowlisted": list(prov.dirty_allowlisted),
            "worktree_state_sha256": prov.worktree_state_sha256,
            "metric_impl_sha256": prov.metric_impl_sha256,
            "timestamp_utc": timestamp_utc,
            "env": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torchvision": tv,
                "numpy": np.__version__,
                "device": req.run.device,
            },
            "config_sha256": prov.config_sha256,
        },
        "dataset": {
            "name": req.dataset.name,
            "doi": req.dataset.doi,
            "split": req.dataset.split,
            "split_manifest_sha256": prov.split_manifest_sha256,
            "expected_rows": req.dataset.expected_rows,
            "actual_rows": len(result.rows),
            "condition": req.dataset.condition.as_dict(),
            "preprocess_protocol": req.dataset.preprocess_protocol,
            "num_classes": req.num_classes,
            "background_index": req.background_index,
            "ignore_index": req.ignore_index,
            "class_map_sha256": prov.class_map_sha256,
        },
        "dataset_level": dict(result.dataset_level),
        "per_class": dict(result.per_class),
        "integrity": dict(result.integrity),
    }


def validate_summary(summary: dict, req: ArtifactRequest, result: EvalResult) -> None:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactWriteError(f"top-level schema_version must be {SCHEMA_VERSION!r}")
    if summary.get("metric_protocol") != METRIC_PROTOCOL:
        raise ArtifactWriteError(f"top-level metric_protocol must be {METRIC_PROTOCOL!r}")
    for block in ("run", "dataset", "dataset_level", "per_class", "integrity"):
        if block not in summary:
            raise ArtifactWriteError(f"summary.json missing required block {block!r}")

    d = summary["dataset"]
    if d["num_classes"] != FROZEN_NUM_CLASSES:
        raise ArtifactWriteError(f"artifact must use {FROZEN_NUM_CLASSES} classes")
    if d["actual_rows"] != d["expected_rows"]:
        raise ArtifactWriteError(
            f"actual_rows {d['actual_rows']} != expected_rows {d['expected_rows']}")
    pc = summary["per_class"]
    for k in ("class_ids", "gt_support", "pred_support", "intersection", "union",
              "iou", "dice", "acc", "iou_status"):
        if k not in pc:
            raise ArtifactWriteError(f"per_class missing {k!r}")
        if len(pc[k]) != FROZEN_NUM_CLASSES:
            raise ArtifactWriteError(
                f"per_class.{k} length {len(pc[k])} != {FROZEN_NUM_CLASSES}")
    integ = summary["integrity"]
    if integ["invalid_pred_labels"] != 0:
        raise ArtifactWriteError("invalid_pred_labels must be 0")
    if req.artifact_status == "official" and not summary["run"]["governed_paths_clean"]:
        raise ArtifactWriteError("official artifact requires governed_paths_clean=True")
    if summary["run"]["random_init"] and req.artifact_status != "smoke":
        raise ArtifactWriteError("random_init=True requires artifact_status='smoke'")


# --------------------------------------------------------------------------------------------------
# atomic write
# --------------------------------------------------------------------------------------------------
def _manifest_text(directory: Path) -> str:
    lines = []
    for name in ARTIFACT_FILES:                       # already lexicographically sorted
        digest = _sha256_bytes((directory / name).read_bytes())
        lines.append(f"{digest}  {name}\n")           # sha256sum-compatible: two ASCII spaces
    return "".join(lines)


def write_artifact(result: EvalResult, req: ArtifactRequest, prov: Provenance,
                   *, timestamp_utc: str | None = None,
                   _inject_failure: bool = False) -> Path:
    """All-or-nothing finalisation: temp sibling -> validate -> re-read -> hash -> rename."""
    if req.out_dir.exists():
        raise ArtifactWriteError(f"refusing to overwrite {req.out_dir}")
    ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = req.out_dir.parent / f".{req.out_dir.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        summary = build_summary(result, req, prov, ts)
        validate_summary(summary, req, result)
        if _inject_failure:
            raise ArtifactWriteError("injected integrity failure (test hook)")

        (tmp / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n")

        rows = "".join(
            json.dumps(r.as_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            for r in result.rows)
        (tmp / "per_image.jsonl").write_text(rows, encoding="utf-8", newline="\n")

        np.savez_compressed(
            tmp / "sufficient_stats.npz",
            image_index=result.sparse_image_index, class_id=result.sparse_class_id,
            tp=result.sparse_tp, gt=result.sparse_gt, pred=result.sparse_pred,
            dataset_tp=result.dataset_tp, dataset_gt=result.dataset_gt,
            dataset_pred=result.dataset_pred,
            manifest_ids=np.array(result.manifest_ids, dtype=np.str_))

        # strict re-read: JSON must parse with non-finite constants rejected
        reread = json.loads((tmp / "summary.json").read_text(encoding="utf-8"),
                            parse_constant=_reject_constant)
        if reread.get("schema_version") != SCHEMA_VERSION or \
                reread.get("metric_protocol") != METRIC_PROTOCOL:
            raise ArtifactWriteError("top-level version fields failed post-write verification")
        for line in (tmp / "per_image.jsonl").read_text(encoding="utf-8").splitlines():
            json.loads(line, parse_constant=_reject_constant)

        # NPZ re-read + sufficient-statistic invariants
        with np.load(tmp / "sufficient_stats.npz", allow_pickle=False) as z:
            for key in ("image_index", "class_id", "tp", "gt", "pred",
                        "dataset_tp", "dataset_gt", "dataset_pred", "manifest_ids"):
                if key not in z:
                    raise ArtifactWriteError(f"sufficient_stats.npz missing array {key!r}")
            for key in ("image_index", "class_id", "tp", "gt", "pred",
                        "dataset_tp", "dataset_gt", "dataset_pred"):
                if z[key].dtype != np.int64:
                    raise ArtifactWriteError(f"{key} dtype {z[key].dtype} != int64")
            for key in ("dataset_tp", "dataset_gt", "dataset_pred"):
                if z[key].shape != (FROZEN_NUM_CLASSES,):
                    raise ArtifactWriteError(f"{key} shape {z[key].shape} != ({FROZEN_NUM_CLASSES},)")
            n_rows = len(result.rows)
            if z["manifest_ids"].shape != (n_rows,):
                raise ArtifactWriteError("manifest_ids shape != n_rows")
            for name, totals in (("tp", z["dataset_tp"]), ("gt", z["dataset_gt"]),
                                 ("pred", z["dataset_pred"])):
                acc = np.zeros(FROZEN_NUM_CLASSES, dtype=np.int64)
                np.add.at(acc, z["class_id"], z[name])
                if not np.array_equal(acc, totals):
                    raise ArtifactWriteError(
                        f"sparse {name} does not sum to dataset_{name}")
            # image_index maps into manifest_ids by ROW POSITION
            ids = z["manifest_ids"]
            by_pos = {r.manifest_index: i for i, r in enumerate(result.rows)}
            for r in result.rows:
                if str(ids[by_pos[r.manifest_index]]) != r.image_id:
                    raise ArtifactWriteError(
                        f"manifest_ids[{by_pos[r.manifest_index]}] != {r.image_id!r}")
            if z["image_index"].size and int(z["image_index"].max()) >= n_rows:
                raise ArtifactWriteError("image_index out of range for manifest_ids")

        (tmp / MANIFEST_NAME).write_text(_manifest_text(tmp), encoding="utf-8", newline="\n")
        # verify every manifest line by re-reading and re-hashing
        for line in (tmp / MANIFEST_NAME).read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if _sha256_bytes((tmp / name).read_bytes()) != digest:
                raise ArtifactWriteError(f"manifest hash mismatch for {name}")

        tmp.rename(req.out_dir)
        return req.out_dir
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def verify_artifact(out_dir: Path) -> dict:
    """Re-verify a finalised artifact from disk: filenames, manifest hashes, strict JSON."""
    out_dir = Path(out_dir)
    present = sorted(p.name for p in out_dir.iterdir())
    expected = sorted([*ARTIFACT_FILES, MANIFEST_NAME])
    if present != expected:
        raise ArtifactWriteError(f"artifact files {present} != {expected}")
    checked = []
    for line in (out_dir / MANIFEST_NAME).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        actual = _sha256_bytes((out_dir / name).read_bytes())
        if actual != digest:
            raise ArtifactWriteError(f"manifest mismatch for {name}: {actual} != {digest}")
        checked.append(name)
    if checked != list(ARTIFACT_FILES):
        raise ArtifactWriteError(f"manifest covers {checked}, expected {list(ARTIFACT_FILES)}")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"),
                         parse_constant=_reject_constant)
    return summary
