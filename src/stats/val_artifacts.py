"""Real VAL evaluation artifacts for the descriptive AM-17 / AM-17b entry points (lanes L-AM17-PERCLASS
and L-AM17B-GAP; docs/lane_specs/part1.md lanes 4 and 7).

One loader serves both entry points, so they refuse the same inputs:

  * the artifact verifies (MANIFEST.sha256, strict JSON) and is read by the statistics ingest under
    `Policy.REHEARSAL`: VAL only (TEST stays behind the M11 isolation path), 846 rows, a real-run
    artifact (provisional or official, never random-init), AM-5 flags read or derived by the reader;
  * canvas only (S2 ruling 7; EVALUATION_CONTRACT section 11(a)): an artifact carrying a
    `summary.protocol` block, or whose `dataset.preprocess_protocol` is not `core_preprocess/1.0.0`,
    is refused. Upstream scores are descriptive and never enter a bootstrap or a paired table;
  * the artifact layout follows the ingest reader's rule (`src.stats.ingest._layout`): a declared
    `artifact_schema_version` must be `plantseg-eval-artifact/1.x.y` (additive), and an absent one is a
    pre-lane artifact (layout 1.0.0) whose AM-5 flags the reader derives. `min_layout` adds a floor:
    the gap lane's inputs are re-scored after the L-AM5 merge, so it requires 1.1.0 or later.

`code_provenance` records which analysis code wrote a derived output: HEAD, whether the named files are
unmodified at HEAD (a `git status` over those explicit paths only, never the whole tree) and each file's
sha256 as run.

The frozen OFFICIAL ingest branch, the TEST driver and the bootstrap task matrix are not touched; this
module only calls the public reader. Import-time behaviour is side-effect free: Git runs only inside
`code_provenance`.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME
from ..eval.protocols import CANVAS_PROTOCOL_ID
from .align import AlignmentError, assert_compatible
from .ingest import EvaluationRun, IngestError, Policy, _layout, load_run, verify_run_manifest

POLICY = Policy.REHEARSAL
AM5_LAYOUT = (1, 1, 0)                 # the first layout that carries the AM-5 fields (ingest)
REPO = Path(__file__).resolve().parents[2]
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ValArtifactError(RuntimeError):
    """An input that is not a usable real canvas VAL artifact, or a pair that is not comparable."""


@dataclass(frozen=True)
class ValArtifact:
    """A verified real VAL canvas artifact: the REHEARSAL run, its summary.json and its file hashes."""
    path: Path
    run: EvaluationRun
    summary: dict
    file_sha256s: dict
    layout: tuple | None                # None: a pre-lane artifact (layout 1.0.0)

    @property
    def run_id(self) -> str:
        return self.summary["run"]["run_id"]

    def describe(self) -> dict:
        """Identity and provenance fields recorded with every derived output."""
        run, ds = self.summary["run"], self.summary["dataset"]
        rt = run.get("eval_runtime") or {}
        return {
            "dir_name": self.path.name, "run_id": run["run_id"], "stage": run["stage"],
            "model_role": run["model_role"], "precision": run["precision"],
            "artifact_status": run["artifact_status"],
            "checkpoint_sha256": run.get("checkpoint_sha256"), "repo_commit": run["repo_commit"],
            "governed_paths_clean": run.get("governed_paths_clean"),
            "metric_impl_sha256": run["metric_impl_sha256"], "device": run["env"].get("device"),
            "model_device": rt.get("model_device"), "batch_size": rt.get("batch_size"),
            "image_digest": rt.get("image_digest"), "timestamp_utc": run.get("timestamp_utc"),
            "split": ds["split"], "split_manifest_sha256": ds["split_manifest_sha256"],
            "preprocess_protocol": ds["preprocess_protocol"],
            "artifact_schema_version": self.summary.get("artifact_schema_version"),
            "all_class_miou": self.summary["dataset_level"]["all_class_miou"],
            "artifact_sha256s": dict(self.file_sha256s),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_canvas(summary: dict, label: str) -> None:
    """Refuse an upstream (or any non-canvas) artifact: S2 ruling 7, canvas artifacts only."""
    pp = summary.get("dataset", {}).get("preprocess_protocol")
    if "protocol" in summary or pp != CANVAS_PROTOCOL_ID:
        raise ValArtifactError(
            f"{label}: only canvas artifacts ({CANVAS_PROTOCOL_ID!r}, no summary.protocol block) are "
            f"accepted; this one has preprocess_protocol={pp!r} and protocol="
            f"{(summary.get('protocol') or {}).get('name')!r} (EVALUATION_CONTRACT section 11(a): an "
            "upstream score never enters a bootstrap or a paired table)")


def load_val_artifact(run_dir, *, label: str, min_layout: tuple | None = None) -> ValArtifact:
    """Verify, check canvas and layout, then load under REHEARSAL. Raises ValArtifactError."""
    run_dir = Path(run_dir)
    try:
        summary = verify_run_manifest(run_dir)
        require_canvas(summary, label)
        layout = _layout(summary)                    # the reader's rule: absent, or 1.x.y
        if min_layout is not None and (layout is None or layout < min_layout):
            raise ValArtifactError(
                f"{label}: artifact_schema_version {summary.get('artifact_schema_version')!r} is below "
                f"the required plantseg-eval-artifact/{'.'.join(map(str, min_layout))} (any 1.x at or "
                "after it is accepted)")
        run = load_run(run_dir, POLICY)
    except IngestError as e:
        raise ValArtifactError(f"{label}: {e}") from e
    sha = {name: _sha256(run_dir / name) for name in sorted([*ARTIFACT_FILES, MANIFEST_NAME])}
    return ValArtifact(path=run_dir, run=run, summary=summary, file_sha256s=sha, layout=layout)


def require_role(art: ValArtifact, *, stage: str, model_role: str, precision: str = "fp32",
                 checkpoint_sha256: str | None = None, label: str) -> None:
    run = art.summary["run"]
    got = (run["stage"], run["model_role"], run["precision"])
    if got != (stage, model_role, precision):
        raise ValArtifactError(f"{label}: expected stage/role/precision {(stage, model_role, precision)}, "
                               f"got {got}")
    if checkpoint_sha256 is not None and run.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValArtifactError(f"{label}: checkpoint_sha256 {run.get('checkpoint_sha256')!r} is not the "
                               f"expected model {checkpoint_sha256!r}")


def _git(*args) -> str | None:
    """git output from the repository root, or None. GIT_OPTIONAL_LOCKS=0: never writes the index."""
    try:
        p = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True, timeout=60,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def code_provenance(rel_paths) -> dict:
    """{code_commit, code_clean_at_commit, code_files} for the analysis code that writes an output.

    `code_clean_at_commit` is None when Git is unavailable, else True iff none of `rel_paths` differs
    from HEAD or is untracked. Only those explicit paths are passed to `git status`.
    """
    rel = [str(p) for p in rel_paths]
    head = (_git("rev-parse", "HEAD") or "").strip()
    commit = head if _HEX40.match(head) else None
    status = _git("status", "--porcelain=v1", "--", *rel) if commit else None
    return {"code_commit": commit,
            "code_clean_at_commit": None if status is None else status == "",
            "code_files": {p: _sha256(REPO / p) for p in rel}}


def require_comparable(baseline: ValArtifact, candidate: ValArtifact) -> None:
    """Same split, manifest, class map, protocol and metric implementation (REHEARSAL strictness)."""
    try:
        assert_compatible(baseline.run.identity, candidate.run.identity, POLICY)
    except AlignmentError as e:
        raise ValArtifactError(f"the two artifacts are not comparable: {e}") from e


__all__ = ["POLICY", "AM5_LAYOUT", "ValArtifactError", "ValArtifact", "require_canvas",
           "load_val_artifact", "require_role", "require_comparable", "code_provenance"]
