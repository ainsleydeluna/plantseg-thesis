"""Statistics driver core (lane L-STATS-OFFICIAL): one comparison table, the 37-input inventory, the
canvas guard, the input loader and the assembly of every observed value from verified inputs.

docs/STATISTICAL_ANALYSIS_CONTRACT.md sections 1, 2, 8.7, 9, 10 and 12.4.6; EVALUATION_CONTRACT section
11(a). This module wires the frozen pieces together and changes none of them: the comparisons are
`src.stats.tests.run_comparison`, the family is `holm_family`, the bootstrap is
`src.stats.bootstrap.run_bootstrap_task` over `FROZEN_TASK_MATRIX`, the pooled estimand is
`src.stats.noninferiority.PooledStages`, and every input is read by the frozen ingest (`load_run`).

  * COMPARISON_TABLE is the ONE map comparison_id -> (baseline stage, candidate stage, metric). Every
    comparison call takes its stages from it and asserts that the loaded artifacts carry them. Its keys
    are bootstrap.ALL_COMPARISON_IDS in that order (scripts/smoke_stats_driver.py checks it).
  * The canvas guard (EVALUATION_CONTRACT section 11(a), S2 ruling 7) refuses, by name and in every
    mode, an input whose dataset.preprocess_protocol is not core_preprocess/1.0.0 or that carries a
    summary.protocol block.
  * Every input's four files are hashed before load_run and after it; both snapshots must equal the
    input's MANIFEST.sha256 lines, and the writer compares a third snapshot before its rename.
  * Each bootstrap task takes its n_images from its own data object, never from an argument.

The statistics artifact writer and verifier (`src.stats.artifact`) import this module lazily to
recompute the observed values from the inputs (orchestrator ruling O4). Import-time behaviour is
side-effect free: nothing is read until a function is called.
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .align import METRIC_DISEASE_ONLY, METRIC_MIOU_C, PairedVector, align_runs
from .bootstrap import (DATASET_MIOU_DELTA, DESCRIPTIVE_E1_E3,
                        FROZEN_TASK_MATRIX, NONINFERIORITY_E3_E6, BootstrapTaskResult,
                        DescriptiveScalars, TaskSpec, comparison_observed, descriptive_scalars,
                        run_bootstrap_task, scalar_callables)
from .corruption_protocol import official_corruption_grid
from .ingest import ARTIFACT_FILES, MANIFEST_NAME, EvaluationRun, Policy, load_run, verify_run_manifest
from .noninferiority import PooledStages
from .robustness import INFERENTIAL_SEVERITIES, MiouCVector, align_miou_c, assemble_miou_c
from .tests import CANONICAL_COMPARISON_IDS, ComparisonResult, HolmFamily, holm_family, run_comparison
from .val_artifacts import ValArtifactError, require_canvas

#: The pooled estimand of sections 9.1-9.2: clean, all-class, union-present dataset-level mIoU.
METRIC_POOLED = "dataset_all_class_miou_union_present"

#: comparison_id -> (baseline stage, candidate stage, metric). The one table (P17).
COMPARISON_TABLE: Mapping[str, tuple[str, str, str]] = {
    "accuracy_e1_e2": ("E1", "E2", METRIC_DISEASE_ONLY),
    "accuracy_e2_e3": ("E2", "E3", METRIC_DISEASE_ONLY),
    "accuracy_e4_e5": ("E4", "E5", METRIC_DISEASE_ONLY),
    "accuracy_e7_e6": ("E7", "E6", METRIC_DISEASE_ONLY),
    "accuracy_e4_e7": ("E4", "E7", METRIC_DISEASE_ONLY),
    "accuracy_e5_e6": ("E5", "E6", METRIC_DISEASE_ONLY),
    "accuracy_e1_e6": ("E1", "E6", METRIC_DISEASE_ONLY),
    "robustness_e1_e6": ("E1", "E6", METRIC_MIOU_C),
    DESCRIPTIVE_E1_E3: ("E1", "E3", METRIC_DISEASE_ONLY),
    NONINFERIORITY_E3_E6: ("E3", "E6", METRIC_POOLED),
}

CLEAN_COMPARISON_IDS = tuple(CANONICAL_COMPARISON_IDS[:7])
ROBUSTNESS_ID = CANONICAL_COMPARISON_IDS[7]
STAGES = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
CORRUPTION_STAGES = ("E1", "E6")
#: Evaluator precision per student stage (orchestrator answer Q11; the evaluator's PRECISIONS).
STAGE_PRECISION = {"E1": "fp32", "E2": "fp32", "E3": "fp32", "E4": "int8_ptq", "E5": "int8_qat",
                   "E6": "int8_qat", "E7": "int8_ptq"}
CLEAN = ("clean", None, None)
#: The four files of an evaluation artifact, hashed together (P5).
INPUT_FILES = (MANIFEST_NAME, *ARTIFACT_FILES)
#: The input-list file of official and smoke mode: the 37 inputs of section 12.4.6, by path.
INPUT_LIST_SCHEMA = "plantseg-stats-inputs/1.0.0"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class DriverRefusal(RuntimeError):
    """A request or an input the driver refuses, by name. Never an unexpected error."""


# --------------------------------------------------------------------------------------------------
# 1. the inventory
# --------------------------------------------------------------------------------------------------
def official_inventory() -> tuple[tuple[str, tuple], ...]:
    """The 37 (stage, condition) identities of section 12.4.6: seven clean stage artifacts, then the
    fifteen inferential cells of E1 and of E6 in the frozen corruption order (configs/
    corruption_protocol.json of this code's repository)."""
    grid = official_corruption_grid()
    cells = tuple((stage, ("corruption", name, sev)) for stage in CORRUPTION_STAGES
                  for name in grid.names for sev in INFERENTIAL_SEVERITIES)
    return tuple((stage, CLEAN) for stage in STAGES) + cells


def condition_tuple(condition: Mapping) -> tuple:
    """A summary.json / provenance-record condition object as a (type, name, severity) tuple:
    ("clean", None, None) or ("corruption", non-empty name, int severity). Refuses anything else."""
    if not isinstance(condition, Mapping) or set(condition) != {"type", "name", "severity"}:
        raise DriverRefusal(f"condition must be an object with type, name and severity: {condition!r}")
    ctype, name, sev = condition["type"], condition["name"], condition["severity"]
    clean = ctype == "clean" and name is None and sev is None
    cell = (ctype == "corruption" and isinstance(name, str) and bool(name)
            and isinstance(sev, int) and not isinstance(sev, bool))
    if not (clean or cell):
        raise DriverRefusal(f"condition {dict(condition)!r} is neither clean (name and severity "
                            "null) nor a corruption cell (non-empty name, integer severity)")
    return (ctype, name, sev)


def _strict_object(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise DriverRefusal(f"input list: duplicate key in {keys}")
    return dict(pairs)


def _reject_constant(name):
    raise DriverRefusal(f"input list: non-finite JSON constant {name!r}")


def _relative_path(value, where: str) -> str:
    """A repository-relative POSIX path: no absolute path, drive, backslash, '.' or '..' component."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise DriverRefusal(f"{where}: path must be a non-empty string without surrounding space")
    parts = value.split("/")
    if (value.startswith("/") or "\\" in value or ":" in value or "\x00" in value
            or any(p in ("", ".", "..") for p in parts)):
        raise DriverRefusal(f"{where}: {value!r} is not a plain repository-relative POSIX path")
    return value


def load_input_list(path: Path) -> tuple[tuple[str, tuple, str], ...]:
    """Parse an input-list file into (stage, condition, repo-relative path) entries.

    Strict: the schema string, exactly the keys {schema, inputs} and {stage, condition, path}, no
    duplicate key, no repeated path (case-folded), and exactly the 37 identities of section 12.4.6
    (each once). Nothing is read beyond this file.
    """
    p = Path(path)
    if not p.is_file():
        raise DriverRefusal(f"input list {p} does not exist or is not a file")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=_strict_object,
                         parse_constant=_reject_constant)
    except DriverRefusal:
        raise
    except (ValueError, UnicodeDecodeError) as e:
        raise DriverRefusal(f"input list {p}: not valid JSON ({e})") from e
    if not isinstance(doc, dict) or set(doc) != {"schema", "inputs"}:
        raise DriverRefusal(f"input list {p}: top level must be an object with exactly schema and "
                            "inputs")
    if doc["schema"] != INPUT_LIST_SCHEMA:
        raise DriverRefusal(f"input list {p}: schema {doc['schema']!r} is not {INPUT_LIST_SCHEMA!r}")
    if not isinstance(doc["inputs"], list):
        raise DriverRefusal(f"input list {p}: inputs must be a list")
    entries, folded = [], set()
    for i, item in enumerate(doc["inputs"]):
        where = f"input list {p}, entry {i}"
        if not isinstance(item, dict) or set(item) != {"stage", "condition", "path"}:
            raise DriverRefusal(f"{where}: must be an object with exactly stage, condition and path")
        if item["stage"] not in STAGES:
            raise DriverRefusal(f"{where}: unknown stage {item['stage']!r}")
        cond = condition_tuple(item["condition"])
        rel = _relative_path(item["path"], where)
        if rel.casefold() in folded:
            raise DriverRefusal(f"{where}: path {rel!r} is listed twice")
        folded.add(rel.casefold())
        entries.append((item["stage"], cond, rel))
    require_inventory([(st, c) for st, c, _ in entries])
    return tuple(entries)


def require_inventory(identities: Sequence[tuple[str, tuple]]) -> None:
    """The identities must be exactly the 37 of section 12.4.6, each once. Refuses by name."""
    want = official_inventory()
    got = [(stage, tuple(cond)) for stage, cond in identities]
    repeated = sorted({g for g in got if got.count(g) > 1}, key=repr)
    missing = [w for w in want if w not in got]
    unexpected = [g for g in got if g not in want]
    if repeated or missing or unexpected:
        raise DriverRefusal(
            f"inputs must be exactly the 37 identities of section 12.4.6: missing {len(missing)} "
            f"(e.g. {missing[:3]}), unexpected {len(unexpected)} (e.g. {unexpected[:3]}), repeated "
            f"{len(repeated)} (e.g. {repeated[:3]})")


# --------------------------------------------------------------------------------------------------
# 2. the canvas guard and the hash snapshots
# --------------------------------------------------------------------------------------------------
def refuse_non_canvas(summary: Mapping, label: str) -> None:
    """EVALUATION_CONTRACT section 11(a): canvas inputs only, in every mode. Refuses by name."""
    try:
        require_canvas(summary, label)
    except ValArtifactError as e:
        raise DriverRefusal(str(e)) from e


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_file_sha256(run_dir: Path) -> dict[str, str]:
    """sha256 of an evaluation artifact's four files (MANIFEST.sha256 and the three payloads)."""
    run_dir = Path(run_dir)
    out = {}
    for name in INPUT_FILES:
        p = run_dir / name
        if not p.is_file() or p.is_symlink():
            raise DriverRefusal(f"{run_dir}: {name} is missing or not a regular file")
        out[name] = _sha256_file(p)
    return out


def manifest_digests(run_dir: Path) -> dict[str, str]:
    """{payload name: digest} from an input's MANIFEST.sha256 (sha256sum format, bare names)."""
    lines = (Path(run_dir) / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    out: dict[str, str] = {}
    for ln in lines:
        digest, sep, name = ln.partition("  ")
        if not sep or not _HEX64.match(digest) or name in out or "/" in name or "\\" in name:
            raise DriverRefusal(f"{run_dir}: malformed MANIFEST.sha256 line {ln!r}")
        out[name] = digest
    if sorted(out) != sorted(ARTIFACT_FILES):
        raise DriverRefusal(f"{run_dir}: MANIFEST.sha256 lists {sorted(out)}, expected "
                            f"{sorted(ARTIFACT_FILES)}")
    return out


def snapshot_agrees(snapshot: Mapping[str, str], digests: Mapping[str, str]) -> bool:
    """True iff a four-file snapshot's payload hashes equal the MANIFEST lines (P5)."""
    return all(snapshot.get(name) == digests.get(name) for name in ARTIFACT_FILES)


# --------------------------------------------------------------------------------------------------
# 3. loading
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedInput:
    stage: str
    condition: tuple
    rel_path: str
    directory: Path
    run: EvaluationRun
    summary: dict
    file_sha256: Mapping[str, str]


def load_input(stage: str, condition: tuple, rel_path: str, input_root: Path,
               policy: Policy) -> LoadedInput:
    """Hash, verify, canvas-guard and ingest one input; refuse by name on any defect."""
    label = f"input {rel_path!r} ({stage}, {condition_tuple_label(condition)})"
    directory = Path(input_root) / rel_path
    if not directory.is_dir():
        raise DriverRefusal(f"{label}: no artifact directory at {directory}")
    if not directory.resolve().is_relative_to(Path(input_root).resolve()):
        raise DriverRefusal(f"{label}: {directory} resolves outside the input root {input_root}")
    before = input_file_sha256(directory)                       # snapshot 1 (P5)
    digests = manifest_digests(directory)
    if not snapshot_agrees(before, digests):
        raise DriverRefusal(f"{label}: payload hashes differ from its MANIFEST.sha256")
    try:
        summary = verify_run_manifest(directory)
    except Exception as e:                                       # noqa: BLE001 -- named refusal
        raise DriverRefusal(f"{label}: {e}") from e
    refuse_non_canvas(summary, label)
    ctype, cname, csev = condition
    try:
        run = load_run(directory, policy, expect_condition=(cname, csev))
    except Exception as e:                                       # noqa: BLE001 -- named refusal
        raise DriverRefusal(f"{label}: {e}") from e
    after = input_file_sha256(directory)                         # snapshot 2 (P5)
    if after != before:
        raise DriverRefusal(f"{label}: its files changed while it was being read")
    ident = run.identity
    if ident.stage != stage or (ident.condition_type, ident.condition_name,
                                ident.condition_severity) != tuple(condition):
        raise DriverRefusal(
            f"{label}: the artifact is ({ident.stage}, {ident.condition_type}, "
            f"{ident.condition_name}, {ident.condition_severity}), not the declared identity")
    return LoadedInput(stage=stage, condition=tuple(condition), rel_path=rel_path,
                       directory=directory, run=run, summary=summary, file_sha256=before)


def condition_tuple_label(condition: tuple) -> str:
    ctype, name, sev = condition
    return "clean" if ctype == "clean" else f"{name} s{sev}"


def load_inputs(entries: Sequence[tuple[str, tuple, str]], input_root: Path,
                policy: Policy) -> dict[tuple, LoadedInput]:
    """Load (stage, condition, repo-relative path) entries; the identities must be the inventory."""
    require_inventory([(stage, cond) for stage, cond, _ in entries])
    return {(stage, tuple(cond)): load_input(stage, tuple(cond), rel, input_root, policy)
            for stage, cond, rel in entries}


def provenance_record(loaded: LoadedInput) -> dict:
    """The ten fields of section 12.4.6, read from the verified summary, in the frozen order."""
    run, ds = loaded.summary["run"], loaded.summary["dataset"]
    return {"stage": run["stage"], "condition": dict(ds["condition"]),
            "repo_relative_path": loaded.rel_path, "artifact_status": run["artifact_status"],
            "checkpoint_sha256": run.get("checkpoint_sha256"),
            "split_manifest_sha256": ds["split_manifest_sha256"],
            "class_map_sha256": ds["class_map_sha256"],
            "metric_impl_sha256": run["metric_impl_sha256"], "config_sha256": run["config_sha256"],
            "repo_commit": run["repo_commit"]}


# --------------------------------------------------------------------------------------------------
# 4. the observed values
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Observed:
    """Everything the 35 tasks and the three blocks need, computed from verified inputs only."""
    policy: Policy
    results: tuple[ComparisonResult, ...]            # the eight, in CANONICAL_COMPARISON_IDS order
    holm: HolmFamily
    descriptive: DescriptiveScalars
    paired: Mapping[str, PairedVector]               # the eight comparisons and descriptive_e1_e3
    miou_c: Mapping[str, MiouCVector]                # E1, E6
    pooled: Mapping[str, PooledStages]               # the seven clean comparisons and the NI pair
    am5_excluded_count: int                          # k (P7)
    excluded_clean_ids: tuple[str, ...]

    def result(self, comparison_id: str) -> ComparisonResult:
        return self.results[CANONICAL_COMPARISON_IDS.index(comparison_id)]


def _assert_identities(pv: PairedVector, baseline: str, candidate: str, cid: str) -> None:
    b, c = pv.baseline_identity, pv.candidate_identity
    if b is None or c is None or b.stage != baseline or c.stage != candidate:
        raise DriverRefusal(f"{cid}: the paired artifacts are not {baseline} -> {candidate}")


def am5_identity(inputs: Mapping[tuple, LoadedInput]) -> tuple[int, tuple[str, ...]]:
    """k and the excluded clean ids, which must be one count and one set across every input (P7)."""
    counts = {li.run.am5.excluded_count for li in inputs.values()}
    sets = {li.run.am5.excluded_clean_ids for li in inputs.values()}
    if len(counts) != 1 or len(sets) != 1:
        raise DriverRefusal(f"AM-5: the inputs exclude different images (counts {sorted(counts)}); "
                            "ground truth decides the set, so this is a manifest mismatch")
    k = counts.pop()
    if not (isinstance(k, int) and not isinstance(k, bool) and 0 <= k):
        raise DriverRefusal(f"AM-5 count {k!r} is not a non-negative integer")
    return k, sets.pop()


def assemble_observed(inputs: Mapping[tuple, LoadedInput], policy: Policy) -> Observed:
    """The eight comparisons, the Holm family, the descriptive block and the pooled stages."""
    grid = official_corruption_grid()
    clean = {stage: inputs[(stage, CLEAN)].run for stage in STAGES}
    k, excluded = am5_identity(inputs)
    paired: dict[str, PairedVector] = {}
    results = []
    for cid in CLEAN_COMPARISON_IDS:
        b, c, metric = COMPARISON_TABLE[cid]
        pv = align_runs(clean[b], clean[c], policy=policy, metric=metric)
        _assert_identities(pv, b, c, cid)
        paired[cid] = pv
        results.append(run_comparison(cid, pv, baseline_stage=b, candidate_stage=c))
    b, c, metric = COMPARISON_TABLE[ROBUSTNESS_ID]
    mc = {}
    for stage in (b, c):
        cells = [inputs[(stage, ("corruption", name, sev))].run
                 for name in grid.names for sev in INFERENTIAL_SEVERITIES]
        mc[stage] = assemble_miou_c(cells, grid, policy=policy, stage=stage, clean=clean[stage])
        if mc[stage].stage != stage:
            raise DriverRefusal(f"{ROBUSTNESS_ID}: the {stage} grid assembled as {mc[stage].stage}")
    pv = align_miou_c(mc[b], mc[c], policy=policy)
    if pv.metric != metric:
        raise DriverRefusal(f"{ROBUSTNESS_ID}: metric {pv.metric!r} is not {metric!r}")
    paired[ROBUSTNESS_ID] = pv
    results.append(run_comparison(ROBUSTNESS_ID, pv, baseline_stage=b, candidate_stage=c))
    holm = holm_family(results)

    b, c, metric = COMPARISON_TABLE[DESCRIPTIVE_E1_E3]
    pv = align_runs(clean[b], clean[c], policy=policy, metric=metric)
    _assert_identities(pv, b, c, DESCRIPTIVE_E1_E3)
    paired[DESCRIPTIVE_E1_E3] = pv
    desc = descriptive_scalars(pv.delta, metric=pv.metric, policy=pv.policy.value,
                               alignment_status=pv.alignment_status)

    pooled = {}
    for cid in (*CLEAN_COMPARISON_IDS, NONINFERIORITY_E3_E6):
        b, c, _ = COMPARISON_TABLE[cid]
        pooled[cid] = PooledStages.from_runs(clean[b], clean[c])
    return Observed(policy=policy, results=tuple(results), holm=holm, descriptive=desc,
                    paired=paired, miou_c=mc, pooled=pooled, am5_excluded_count=k,
                    excluded_clean_ids=excluded)


def observed_value(obs: Observed, spec: TaskSpec) -> float:
    """The observed value of one task, from its frozen source (section 12.3.3, 24/3/8)."""
    if spec.statistic_name == DATASET_MIOU_DELTA:
        return float(obs.pooled[spec.comparison_id].observed())
    if spec.comparison_id == DESCRIPTIVE_E1_E3:
        return float(obs.descriptive.value(spec.statistic_name))
    return comparison_observed(obs.result(spec.comparison_id), spec.statistic_name)


def observed_values(obs: Observed) -> dict[str, float]:
    return {spec.task_id: observed_value(obs, spec) for spec in FROZEN_TASK_MATRIX}


# --------------------------------------------------------------------------------------------------
# 5. the 35 bootstrap tasks (whole-task parallelism only, section 8.7.4)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskJob:
    spec: TaskSpec
    analysis_id: str
    B: int
    observed: float
    delta: np.ndarray | None                 # scalar tasks: the aligned per-image differences
    dense: tuple | None                      # pooled tasks: (base tp, gt, pred, cand tp, gt, pred)


def task_jobs(obs: Observed, analysis_id: str, B: int) -> tuple[TaskJob, ...]:
    jobs = []
    for spec in FROZEN_TASK_MATRIX:
        if spec.statistic_name == DATASET_MIOU_DELTA:
            ps = obs.pooled[spec.comparison_id]
            dense = (ps.base_tp, ps.base_gt, ps.base_pred, ps.cand_tp, ps.cand_gt, ps.cand_pred)
            jobs.append(TaskJob(spec, analysis_id, B, observed_value(obs, spec), None, dense))
        else:
            delta = np.asarray(obs.paired[spec.comparison_id].delta, dtype=np.float64)
            jobs.append(TaskJob(spec, analysis_id, B, observed_value(obs, spec), delta, None))
    return tuple(jobs)


def run_job(job: TaskJob) -> BootstrapTaskResult:
    """One task. n_images comes from the task's own data object (P7)."""
    if job.dense is not None:
        ps = PooledStages.from_dense(job.dense[:3], job.dense[3:])
        n_images, rf, jf = ps.n_images, ps.replicate, ps.jackknife
    else:
        n_images = int(job.delta.shape[0])
        rf, jf = scalar_callables(job.spec.statistic_name, job.delta)
    return run_bootstrap_task(job.spec, job.analysis_id, n_images, job.B, job.observed, rf, jf)


class WorkerFailure(RuntimeError):
    """A bootstrap worker process failed. Never a refusal: the CLI exits 4 (P18)."""


def run_tasks(obs: Observed, analysis_id: str, B: int, *, jobs: int = 1
              ) -> tuple[BootstrapTaskResult, ...]:
    """The 35 tasks in canonical order. jobs > 1 runs whole tasks in spawned worker processes; each
    task owns its Generator, so the results are identical to a sequential run (section 8.7.4)."""
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise DriverRefusal(f"--jobs must be a positive integer, got {jobs!r}")
    work = task_jobs(obs, analysis_id, B)
    if jobs == 1:
        return tuple(run_job(j) for j in work)
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as ex:
            return tuple(ex.map(run_job, work))
    except Exception as e:                                       # noqa: BLE001 -- any worker failure
        raise WorkerFailure(f"a bootstrap worker failed: {type(e).__name__}: {e}") from e


def artifact_inputs(obs: Observed, inputs: Mapping[tuple, LoadedInput], tasks, *, analysis_id: str,
                    run_id: str, created_at_utc: str, contract_sha256: str, synthetic: bool,
                    input_root: Path):
    """The writer's typed inputs: the 37 provenance records, the computed blocks, k and the driver's
    hash snapshots. The NI decision reads the NI task's one stored lower bound (section 9.2)."""
    from .artifact import ArtifactInputs
    from .noninferiority import build_e6_kd, build_non_inferiority
    tasks = tuple(tasks)
    b, c = obs.pooled[NONINFERIORITY_E3_E6].stage_mious()
    ni_id = f"{NONINFERIORITY_E3_E6}__{DATASET_MIOU_DELTA}"
    ni_task = next(t for t in tasks if t.task_id == ni_id)
    return ArtifactInputs(
        analysis_id=analysis_id, run_id=run_id, created_at_utc=created_at_utc,
        contract_sha256=contract_sha256,
        input_artifacts=[provenance_record(li) for li in inputs.values()],
        comparison_results=obs.results, holm_family=obs.holm, descriptive=obs.descriptive,
        non_inferiority=build_non_inferiority(b, c, ni_task.bca.bounds[0]),
        e6_kd=build_e6_kd(b, c), task_results=tasks, synthetic=synthetic,
        am5_excluded_count=obs.am5_excluded_count, input_root=Path(input_root),
        input_file_sha256={li.rel_path: dict(li.file_sha256) for li in inputs.values()})


# --------------------------------------------------------------------------------------------------
# 6. modes (scripts/run_stats.py is the command-line wrapper)
# --------------------------------------------------------------------------------------------------
class PostWriteVerifyFailed(RuntimeError):
    """The artifact was renamed into place but its re-verification failed (P10: exit 4, naming it)."""


@dataclass(frozen=True)
class RunResult:
    mode: str
    artifact: Path
    family: dict
    established: tuple
    unmet_official: tuple             # official mode: (name, detail) pairs (P30); () otherwise
    am5_excluded_count: int
    input_root: Path
    seconds: float


def _now_utc() -> str:
    from datetime import datetime, timezone
    from .artifact import TIMESTAMP_FORMAT
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def official_doors(*, confirm: bool, synthetic: bool) -> None:
    """P13 doors 1 and 2, before any file is read: the confirmation flag, then the binding rule
    (while the TEST-manifest binding is unbound, official mode runs only on inputs declared
    synthetic; P2, P14)."""
    from .artifact import TEST_MANIFEST_BINDING
    if not confirm:
        raise DriverRefusal("official mode reads the 37 TEST inputs of section 12.4.6 and requires "
                            "--confirm-official-test-analysis")
    if TEST_MANIFEST_BINDING != "unbound":                # fail closed: there is no bound branch
        raise DriverRefusal(f"unknown TEST_MANIFEST_BINDING {TEST_MANIFEST_BINDING!r}")
    if not synthetic:
        raise DriverRefusal("the TEST-manifest binding is unbound: official mode runs only on inputs "
                            "declared synthetic (--synthetic-inputs), and no official statistics "
                            "artifact can be written (P2)")


def require_pinned_stack() -> None:
    """Official mode runs on the section 10.1 pinned stack only (Q6); refused before any read."""
    from .artifact import software_environment_block
    sw = software_environment_block()
    if not sw["matches_pinned"]:
        raise DriverRefusal(f"official mode requires the section 10.1 pinned statistics stack "
                            f"{sw['pinned']}; the running stack is {sw['observed']}")


def smoke_screen(entries, input_root: Path) -> None:
    """P15, from each input's summary.json alone (before any per-image file is read): smoke mode
    accepts only smoke-status inputs, and never an input with the evaluator's dataset name and
    split test."""
    from ..eval.adapters import DATASET_NAME
    for stage, cond, rel in entries:
        p = Path(input_root) / rel / "summary.json"
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            status, name, split = (s["run"]["artifact_status"], s["dataset"]["name"],
                                   s["dataset"]["split"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise DriverRefusal(f"input {rel!r}: summary.json cannot be read ({e})") from e
        if name == DATASET_NAME and split == "test":
            raise DriverRefusal(f"input {rel!r}: smoke mode never reads a {DATASET_NAME} TEST "
                                "artifact (P15)")
        if status != "smoke":
            raise DriverRefusal(f"input {rel!r}: smoke mode accepts only smoke-status inputs, this "
                                f"one is {status!r} (P15)")


def assemble_or_refuse(inputs: Mapping[tuple, LoadedInput], policy: Policy) -> Observed:
    """assemble_observed with the frozen modules' integrity errors refused by name."""
    from .align import AlignmentError
    from .bootstrap import BootstrapError
    from .ingest import IngestError
    from .robustness import RobustnessError
    from .tests import StatsError
    try:
        return assemble_observed(inputs, policy)
    except (AlignmentError, BootstrapError, IngestError, RobustnessError, StatsError) as e:
        raise DriverRefusal(f"the inputs cannot be analysed: {type(e).__name__}: {e}") from e


def _verify_written(path: Path, input_root: Path):
    from . import artifact as A
    try:
        rep = A.verify_statistics_artifact(path, input_root=input_root,
                                           contract_bytes=A.contract_sha256()[1])
    except Exception as e:                                       # noqa: BLE001 -- named, exit 4
        raise PostWriteVerifyFailed(f"{path}: written but its re-verification failed: "
                                    f"{type(e).__name__}: {e}") from e
    if rep.not_reestablished or rep.established != A.INTEGRITY_CHECKS:
        raise PostWriteVerifyFailed(f"{path}: written but checks {rep.not_reestablished} were not "
                                    "re-established")
    return rep


def run_official(*, inputs_list, out_dir, run_id, confirm: bool, synthetic: bool,
                 repo_root=None, B: int, jobs: int = 1, log=print) -> RunResult:
    """--mode official. Doors (P13): confirmation flag, binding rule, out-dir pre-check (P10), pinned
    stack (Q6), then reads. Inputs resolve against --repo-root (synthetic only; P16) or this code's
    repository; the contract always comes from this code's repository."""
    from . import artifact as A
    from .bootstrap import ANALYSIS_ID_OFFICIAL
    import time
    t0 = time.time()
    official_doors(confirm=confirm, synthetic=synthetic)
    target = A.preflight_out_dir(Path(out_dir) / run_id)
    require_pinned_stack()
    input_root = Path(repo_root) if repo_root is not None else A.CODE_REPO
    contract_hex, _ = A.contract_sha256()
    created = _now_utc()
    entries = load_input_list(inputs_list)                       # the first read
    inputs = load_inputs(entries, input_root, Policy.OFFICIAL)
    obs = assemble_or_refuse(inputs, Policy.OFFICIAL)
    log(f"inputs: 37 loaded from {input_root}; AM-5 k = {obs.am5_excluded_count}; per-image n = "
        f"{obs.results[0].n_paired}")
    tasks = run_tasks(obs, ANALYSIS_ID_OFFICIAL, B, jobs=jobs)
    inp = artifact_inputs(obs, inputs, tasks, analysis_id=ANALYSIS_ID_OFFICIAL, run_id=run_id,
                          created_at_utc=created, contract_sha256=contract_hex, synthetic=True,
                          input_root=input_root)
    unmet = A.unmet_official_conditions(inp)
    path = A.write_statistics_artifact(target, inp)
    rep = _verify_written(path, input_root)
    return RunResult("official", path, rep.family, rep.established, unmet, obs.am5_excluded_count,
                     input_root, time.time() - t0)


def run_smoke(*, inputs_list, out_dir, run_id, repo_root=None, B: int, jobs: int = 1,
              log=print) -> RunResult:
    """--mode smoke: smoke-status inputs only (P15), NONOFFICIAL_SMOKE, the smoke namespace; the
    synthetic declaration is what the inputs' dataset names say (P6)."""
    from . import artifact as A
    from ..eval.adapters import DATASET_NAME
    from .bootstrap import ANALYSIS_ID_SMOKE
    import time
    t0 = time.time()
    target = A.preflight_out_dir(Path(out_dir) / run_id)
    input_root = Path(repo_root) if repo_root is not None else A.CODE_REPO
    contract_hex, _ = A.contract_sha256()
    created = _now_utc()
    entries = load_input_list(inputs_list)
    smoke_screen(entries, input_root)
    inputs = load_inputs(entries, input_root, Policy.NONOFFICIAL_SMOKE)
    k, _ = am5_identity(inputs)
    if k:
        raise DriverRefusal(f"smoke mode needs inputs without AM-5 exclusions (k = {k}): "
                            "NONOFFICIAL_SMOKE keeps those rows, so the per-image comparisons would "
                            "carry undefined pairs")
    obs = assemble_or_refuse(inputs, Policy.NONOFFICIAL_SMOKE)
    log(f"inputs: 37 loaded from {input_root}; n = {obs.results[0].n_paired}")
    tasks = run_tasks(obs, ANALYSIS_ID_SMOKE, B, jobs=jobs)
    synthetic = any(li.summary["dataset"]["name"] != DATASET_NAME for li in inputs.values())
    inp = artifact_inputs(obs, inputs, tasks, analysis_id=ANALYSIS_ID_SMOKE, run_id=run_id,
                          created_at_utc=created, contract_sha256=contract_hex,
                          synthetic=synthetic, input_root=input_root)
    path = A.write_statistics_artifact(target, inp)
    rep = _verify_written(path, input_root)
    return RunResult("smoke", path, rep.family, rep.established, (), k, input_root,
                     time.time() - t0)


__all__ = [
    "METRIC_POOLED", "COMPARISON_TABLE", "CLEAN_COMPARISON_IDS", "ROBUSTNESS_ID", "STAGES",
    "CORRUPTION_STAGES", "STAGE_PRECISION", "CLEAN", "INPUT_FILES", "INPUT_LIST_SCHEMA",
    "DriverRefusal", "official_inventory", "condition_tuple", "load_input_list",
    "require_inventory", "refuse_non_canvas", "input_file_sha256",
    "manifest_digests", "snapshot_agrees", "LoadedInput", "load_input", "load_inputs",
    "provenance_record", "Observed", "am5_identity", "assemble_observed", "observed_value",
    "observed_values", "TaskJob", "task_jobs", "run_job", "WorkerFailure", "run_tasks",
    "artifact_inputs", "PostWriteVerifyFailed", "RunResult", "official_doors",
    "require_pinned_stack", "smoke_screen", "assemble_or_refuse", "run_official", "run_smoke",
]
