"""Descriptive robustness outputs (L-CORR B-4; plan section 4, with rulings LE1, LE2, LE4 and LE6-LE8).

Inputs: the inventory export (`tday-inventory/1`, TL interface section 8) and, under one artifacts
root, each scored model's clean artifact and its corrupted artifacts with their cell records, named
as src/eval/corrupted.py names them (`tday_clean_<file_id>` and `tday_corr_<file_id>_<cell id>` for
split test; the `val_` forms for VAL). Only export rows with a non-empty `cells` list are read. Each
artifact is read by the statistics ingest (`src.stats.ingest.load_run`) as a public reader, under
the policy asked for: OFFICIAL (split test), REHEARSAL (split val) or NONOFFICIAL_SMOKE (split val).

Every mIoU below is the dataset-level all-class union-present value `pooled_miou_from_totals` of the
artifact's NPZ totals, in float64 (DL-73, P28). The disease-only pooled value is a secondary column
of Q1 only. Means are numpy float64 means, nested as each rule names them. A null means a cell the
quantity needs is absent (listed under the model's `missing`) or not scored for that model; the
rCD nulls carry their reason.

  Q1 per cell        m(c, s) for every scored cell of every model
  Q2 per severity    two series (LE6): the four non-noise corruptions at s = 1-5, and all five at
                     s = 1-4; at each severity, the mean over the series' corruptions, from that
                     severity's own cells
  Q3 mIoU-C          the mean over the five corruptions of the mean over s = 1-3 (the arithmetic
                     of src/stats/report.dataset_miou_c, P28), for every model holding the 15
                     inferential cells, seeds 43/44 included (LE4)
  Q4 RPD             (m_clean - mIoU-C) / m_clean x 100 (src/eval/robustness.rpd); the same models
  Q5 rCD             per corruption, the sum over s = 1-3 of (D(c, s) - D_clean), D = 1 - m, divided
                     by the same sum for E1 at seed 42 (src/eval/robustness.rcd); E1 is 1 exactly.
                     Students at seed 42 only, never the teacher (LE4). When E1's sum is <= 0, the
                     rCD of that corruption is null for every stage, reason
                     `rcd_reference_nonpositive`, whatever else the stage lacks (LE7;
                     `rcd_invalid_reference` is caught)
  Q6 rCD mean        the mean of Q5 over the five corruptions, only when all five are defined (LE7)
  Q7 per corruption  each corruption's mean over s = 1-3, with Q5 for the students
  Q8 Kamann-Rother   AM-16 item 6's average, nested (LE8): each corruption's mean over its own
                     severities -- s = 1-5 for the four non-noise corruptions, s = 1-3 for
                     gaussian_noise -- then equal weight over the five; beside it, each LE6 series'
                     mean (within a series every corruption has the same severities)
  Q9 provenance      each input's four-file sha256 and record sha256, the one cache checksum, each
                     artifact's runtime tuple and commit, the inventory, and this code's commit
                     and files

Q2, Q7 and Q8 cover the teacher and the seed-42 students; seeds 43/44 carry Q1, Q3 and Q4 only.
E6-KD carries no corrupted cell (LE4).

Checks, each refused by code before anything is written:
  per artifact   `input_refused` (the four files do not verify, or the policy's rules refuse it);
                 `artifact_split_mismatch` (its split is not the policy's);
                 `condition_mismatch` (its condition is not its directory's cell);
                 `non_canvas_input` (EVALUATION_CONTRACT section 11(a));
                 `artifact_row_mismatch` (stage, precision or role not the export row's);
                 `checkpoint_identity_mismatch` (LE2: run.checkpoint_sha256 is not the export's
                 `converted_artifact_sha256` for INT8, else its `source.sha256`);
                 `record_mismatch` (the cell record is absent, does not read or verify -- the
                 inner code is named -- or names another stage, condition, run, role, split,
                 commit or row count)
  across the set `artifact_missing` (official: every artifact; otherwise the gap is reported and
                 every quantity that needs it is null); `split_digest_mismatch`;
                 `expected_cache_sha256_required` (official) and `expected_cache_sha256_format`;
                 `cache_checksum_mismatch` (one cache across the records, equal to the expected
                 one); `heterogeneous_runtime` (LE1, official: each artifact's runtime tuple and
                 commit equal the export's `execution_context` and `evaluator_commit`)
  the request    `policy_unknown`, `policy_split_mismatch`; for an official report, this code at
                 the export's evaluator commit (`report_commit_mismatch`) and unmodified there
                 (`report_code_modified`); the out-dir: `out_dir_name_invalid` (TL section 2: no
                 "test" in any case and none of :<>"\\|?* in any component, as given and resolved,
                 the strings checked before any filesystem call), `out_dir_inside_repo` (resolved),
                 `out_dir_exists`; the writer re-reads what it wrote (`output_verify_failed`)

Outputs: robustness_descriptive.json (strict JSON, sorted keys), robustness_descriptive.md and
SHA256SUMS, written atomically to a new directory outside the repository; a failed write removes its
own temporary directory. Nothing here is inferential: no hypothesis test, no bootstrap, no
significance value, and the outputs never enter family.json (D22). The JSON states its class space
and names no quantity after the rejected section 8.7.1 alias (DL-73, LE13). A banner names every
non-official input.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import corruption_roles
from .artifacts import ARTIFACT_FILES, MANIFEST_NAME
from .corrupted import (CorruptedEvalError, clean_dir_name, corrupted_dir_name,
                        created_path_problem, read_cell_record, record_path, verify_cell_record)
from .protocols import CANVAS_PROTOCOL_ID
from .robustness import RobustnessError, rcd, rpd

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "plantseg-robustness-descriptive/1.0.0"
JSON_NAME, MD_NAME, SUMS_NAME = ("robustness_descriptive.json", "robustness_descriptive.md",
                                 "SHA256SUMS")
POLICIES = ("official", "rehearsal", "smoke")
POLICY_SPLIT = {"official": "test", "rehearsal": "val", "smoke": "val"}
CORRUPTIONS = corruption_roles.CANONICAL_IDS
NON_NOISE = tuple(c for c in CORRUPTIONS if c != "gaussian_noise")
INFERENTIAL = corruption_roles.INFERENTIAL_SEVERITIES
#: Q3's severities (P28; IMPL (f)): mIoU-C, and so RPD, never reads severity 4 or 5.
MIOU_C_SEVERITIES = INFERENTIAL
#: Q8 (AM-16 item 6, after Kamann & Rother): non-noise over s = 1-5, noise over s = 1-3.
KR_SEVERITIES = {c: ((1, 2, 3) if c == "gaussian_noise" else (1, 2, 3, 4, 5)) for c in CORRUPTIONS}
#: LE6: the two series of Q2, (name, corruptions, severities).
SERIES = (("non_noise_s1_5", NON_NOISE, (1, 2, 3, 4, 5)), ("all_five_s1_4", CORRUPTIONS, (1, 2, 3, 4)))
#: LE1: the execution-context values every official artifact shares with the export (IF section 1).
RUNTIME_KEYS = ("model_device", "input_devices", "determinism_policy_applied", "batch_size",
                "num_workers", "torch_num_threads", "image_digest")
INT8 = ("int8_ptq", "int8_qat")
HEX64 = re.compile(r"[0-9a-f]{64}")
CODE_FILES = ("src/eval/robustness_report.py", "scripts/report_robustness.py",
              "scripts/run_corruption_grid.py", "src/eval/robustness.py", "src/eval/corrupted.py",
              "src/eval/artifacts.py", "src/eval/protocols.py", "src/corruption_cache.py",
              "src/corruption_roles.py", "configs/corruption_protocol.json", "src/stats/ingest.py",
              "src/stats/noninferiority.py", "src/stats/report.py", "src/stats/val_artifacts.py")


class ReportRefusal(RuntimeError):
    """An input or request the descriptive report refuses, by code. Nothing is written."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _refuse(code: str, message: str):
    raise ReportRefusal(code, message)


def utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def class_space() -> str:
    from ..stats.report import CLASS_SPACE
    return CLASS_SPACE


def code_state() -> dict:
    """{code_commit, code_clean_at_commit, code_files} of the code that writes these outputs."""
    from ..stats.val_artifacts import code_provenance
    return code_provenance(CODE_FILES)


def ingest_policy(policy: str):
    from ..stats.ingest import Policy
    return {"official": Policy.OFFICIAL, "rehearsal": Policy.REHEARSAL,
            "smoke": Policy.NONOFFICIAL_SMOKE}[policy]


# --------------------------------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------------------------------
def pooled(stats) -> float:
    """DL-73, P28: the all-class union-present pooled mIoU of an artifact's totals (float64)."""
    from ..stats.noninferiority import pooled_miou_from_totals
    return pooled_miou_from_totals(stats.dataset_tp, stats.dataset_gt, stats.dataset_pred)


def clean_value(run) -> float:
    """m_clean of Q4 and Q5: the clean artifact's pooled value, never a per-image mean."""
    return pooled(run.stats)


def disease_pooled(stats):
    """The same rule over the disease classes 1..115 (Q1's secondary column); None when no disease
    class is union-present."""
    tp = np.asarray(stats.dataset_tp, dtype=np.int64)[1:]
    union = (np.asarray(stats.dataset_gt, dtype=np.int64)[1:]
             + np.asarray(stats.dataset_pred, dtype=np.int64)[1:] - tp)
    eligible = union > 0
    if not eligible.any():
        return None
    return float(np.mean(tp[eligible].astype(np.float64) / union[eligible].astype(np.float64)))


# --------------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------------
@dataclass
class ModelInputs:
    row: dict
    clean: object = None                               # EvaluationRun
    cells: dict = field(default_factory=dict)          # cell id -> EvaluationRun
    dirs: dict = field(default_factory=dict)           # "clean" or cell id -> directory name
    files: dict = field(default_factory=dict)          # label -> {file: sha256}
    records: dict = field(default_factory=dict)        # cell id -> record
    record_sha256: dict = field(default_factory=dict)  # cell id -> sha256 of the record file
    summaries: dict = field(default_factory=dict)      # label -> verified summary.json
    missing: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.row["file_id"]


def expected_identity(row: dict) -> str:
    """LE2: the converted artifact for INT8 (evaluate_model.py:271-272), else the handed file."""
    return row["converted_artifact_sha256"] if row["precision"] in INT8 else row["source"]["sha256"]


def expected_condition(label: str) -> dict:
    if label == "clean":
        return {"type": "clean", "name": None, "severity": None}
    name, severity = corruption_roles.parse_cell_id(label)
    return {"type": "corruption", "name": name, "severity": severity}


def cell_role(label: str) -> str:
    severity = corruption_roles.parse_cell_id(label)[1]
    return (corruption_roles.ROLE_INFERENTIAL if severity in INFERENTIAL
            else corruption_roles.ROLE_DESCRIPTIVE_ONLY)


def check_artifact(model: ModelInputs, label: str, summary: dict, split: str) -> None:
    """The artifact is the policy's split, its directory's cell, canvas, and the export row's model
    (LE2)."""
    row, run, ds = model.row, summary["run"], summary["dataset"]
    name = model.dirs[label]
    if ds.get("split") != split:
        _refuse("artifact_split_mismatch", f"{name} is split {ds.get('split')!r}; this read is "
                                           f"split {split!r}")
    want = expected_condition(label)
    if ds.get("condition") != want:
        _refuse("condition_mismatch", f"{name} holds condition {ds.get('condition')}, its name "
                                      f"says {want}")
    if ds.get("preprocess_protocol") != CANVAS_PROTOCOL_ID or "protocol" in summary:
        _refuse("non_canvas_input",
                f"{name}: protocol {ds.get('preprocess_protocol')!r}; only canvas artifacts enter "
                "the robustness outputs (EVALUATION_CONTRACT section 11(a))")
    got = (run.get("stage"), run.get("precision"), run.get("model_role"))
    want_row = (row["stage"], row["precision"], row["model_role"])
    if got != want_row:
        _refuse("artifact_row_mismatch", f"{name} is {got}, the export row is {want_row}")
    if run.get("checkpoint_sha256") != expected_identity(row):
        field_name = "converted_artifact_sha256" if row["precision"] in INT8 else "source.sha256"
        _refuse("checkpoint_identity_mismatch",
                f"{name} scored {run.get('checkpoint_sha256')}, the export's {field_name} is "
                f"{expected_identity(row)}")


def check_record(model: ModelInputs, label: str, summary: dict, split: str) -> None:
    """The record, verified against the artifact's four files, names this run as well."""
    record, run, ds = model.records[label], summary["run"], summary["dataset"]
    rows = ds["actual_rows"]
    problems = [k for k, ok in (
        ("stage", record["stage"] == run["stage"]),
        ("condition", record["condition"] == ds["condition"]),
        ("run_id", record["run_id"] == run["run_id"]),
        ("role", record["role"] == cell_role(label)),
        ("cache.split", record["cache"]["split"] == split),
        ("evaluator_commit", record["evaluator_commit"] == run["repo_commit"]),
        ("counts", record["items_verified"] == record["source_digests_verified"] == rows),
    ) if not ok]
    if problems:
        _refuse("record_mismatch", f"the record of {model.dirs[label]} disagrees on {problems}")


def load_model(root: Path, split: str, row: dict, policy: str) -> ModelInputs:
    from ..stats.ingest import IngestError, load_run, verify_run_manifest
    model = ModelInputs(row=row)
    for label in ("clean", *row["cells"]):
        name = (clean_dir_name(split, row["file_id"]) if label == "clean"
                else corrupted_dir_name(split, row["file_id"], label))
        d = Path(root) / name
        model.dirs[label] = name
        if not d.is_dir():
            model.missing.append(label)
            continue
        try:
            summary = verify_run_manifest(d)
        except IngestError as e:
            _refuse("input_refused", f"{name}: {e}")
        check_artifact(model, label, summary, split)
        try:
            run = load_run(d, ingest_policy(policy))
        except IngestError as e:
            _refuse("input_refused", f"{name}: {e}")
        model.summaries[label] = summary
        model.files[label] = {n: _sha256_file(d / n) for n in sorted((*ARTIFACT_FILES,
                                                                       MANIFEST_NAME))}
        if label == "clean":
            model.clean = run
            continue
        rec_path = record_path(d)
        if not rec_path.is_file():
            _refuse("record_mismatch", f"{name} has no cell record {rec_path.name}")
        try:
            record = read_cell_record(rec_path)
            verify_cell_record(d, record)
        except CorruptedEvalError as e:
            _refuse("record_mismatch", f"{name}: {e}")
        model.records[label] = record
        model.record_sha256[label] = _sha256_file(rec_path)
        check_record(model, label, summary, split)
        model.cells[label] = run
    return model


def runtime_tuple(summary: dict) -> dict:
    rt = summary["run"].get("eval_runtime") or {}
    return {k: rt.get(k) for k in RUNTIME_KEYS}


def runtime_problems(models: list, export: dict) -> list:
    """LE1: every artifact of every model -- the clean one included -- whose runtime tuple or commit
    is not the export's execution context and evaluator commit."""
    ctx, commit = export["execution_context"], export["evaluator_commit"]
    want = {k: ctx.get(k) for k in RUNTIME_KEYS}
    out = []
    for m in models:
        for label, s in m.summaries.items():
            got = runtime_tuple(s)
            if got != want or s["run"].get("repo_commit") != commit:
                diff = {k: got[k] for k in RUNTIME_KEYS if got[k] != want[k]}
                out.append(f"{m.dirs[label]}: runtime {diff or 'equal'} at commit "
                           f"{s['run'].get('repo_commit')}")
    return out


def check_set(models: list, export: dict, policy: str, expected_cache) -> dict:
    """Across every loaded artifact: completeness (official), one split digest, one cache, and
    (LE1, official) one runtime tuple and commit, the export's. Returns Q9's cache block."""
    official = policy == "official"
    if official:
        missing = [f"{m.key}:{label}" for m in models for label in m.missing]
        if missing:
            _refuse("artifact_missing", f"an official set needs every artifact; missing "
                                        f"{len(missing)}: {missing[:5]}")
    digests = sorted({s["dataset"]["split_manifest_sha256"] for m in models
                      for s in m.summaries.values()})
    if len(digests) > 1:
        _refuse("split_digest_mismatch", f"the inputs carry {len(digests)} split digests: "
                                         f"{[d[:12] for d in digests]}")
    if official and expected_cache is None:
        _refuse("expected_cache_sha256_required",
                "an official set requires --expected-cache-sha256, the recorded canonical "
                "checksum of the cache (IF section 5)")
    if expected_cache is not None and not HEX64.fullmatch(str(expected_cache)):
        _refuse("expected_cache_sha256_format",
                f"--expected-cache-sha256 must be 64 lowercase hex characters, got "
                f"{expected_cache!r}")
    checksums = sorted({r["cache"]["canonical_manifest_sha256"] for m in models
                        for r in m.records.values()})
    if len(checksums) > 1 or (expected_cache is not None and set(checksums) - {expected_cache}):
        _refuse("cache_checksum_mismatch",
                f"the cell records name the caches {checksums}"
                + (f"; expected {expected_cache}" if expected_cache is not None else ""))
    if official:
        problems = runtime_problems(models, export)
        if problems:
            _refuse("heterogeneous_runtime",
                    f"{len(problems)} artifact(s) not at the export's execution context "
                    f"{dict((k, export['execution_context'].get(k)) for k in RUNTIME_KEYS)} and "
                    f"commit {export['evaluator_commit']}: {problems[:3]}")
    return {"canonical_manifest_sha256": checksums[0] if checksums else None,
            "expected": expected_cache,
            "verification_record_sha256": sorted({str(r["cache"]["verification_record_sha256"])
                                                  for m in models for r in m.records.values()})}


# --------------------------------------------------------------------------------------------------
# quantities
# --------------------------------------------------------------------------------------------------
def _mean(values) -> float:
    return float(np.mean(np.array(values, dtype=np.float64)))


def cell_values(model: ModelInputs) -> dict:
    """(corruption, severity) -> m, for the cells this model holds."""
    return {corruption_roles.parse_cell_id(cid): pooled(run.stats) for cid, run in model.cells.items()}


def miou_c(m: dict):
    """Q3, the arithmetic of src/stats/report.dataset_miou_c (P28): per corruption the mean over
    severities 1-3, then the mean over the five, in protocol order. None when a cell is missing."""
    if any((c, s) not in m for c in CORRUPTIONS for s in MIOU_C_SEVERITIES):
        return None
    per_corruption = []
    for name in CORRUPTIONS:
        per_corruption.append(_mean([m[(name, sev)] for sev in MIOU_C_SEVERITIES]))
    return _mean(per_corruption)


def kamann_rother(m: dict):
    """Q8 (AM-16 item 6; LE8): each corruption's mean over its own severities (KR_SEVERITIES),
    then equal weight over the five. None when a cell is missing."""
    if any((c, s) not in m for c in CORRUPTIONS for s in KR_SEVERITIES[c]):
        return None
    return _mean([_mean([m[(c, s)] for s in KR_SEVERITIES[c]]) for c in CORRUPTIONS])


def series_values(m: dict) -> tuple[dict, dict]:
    """Q2 (LE6): per series and severity, the mean over the series' corruptions of that severity's
    own cells; and each series' mean (nested, equal to flat within a series). None where a cell the
    value needs is missing."""
    q2, means = {}, {}
    for name, corruptions, severities in SERIES:
        q2[name] = {str(s): (_mean([m[(c, s)] for c in corruptions])
                             if all((c, s) in m for c in corruptions) else None)
                    for s in severities}
        have = all((c, s) in m for c in corruptions for s in severities)
        means[name] = (_mean([_mean([m[(c, s)] for s in severities]) for c in corruptions])
                       if have else None)
    return q2, means


def degradation(m: dict, m_clean, corruption: str):
    """The sum over s = 1-3 of (D(c, s) - D_clean), D = 1 - m; None when the clean value or a cell
    is missing."""
    if m_clean is None or any((corruption, s) not in m for s in INFERENTIAL):
        return None
    d_clean = 1.0 - m_clean
    total = 0.0
    for s in INFERENTIAL:
        total += (1.0 - m[(corruption, s)]) - d_clean
    return total


def in_rcd(row: dict) -> bool:
    """Q5 and Q6 subjects (LE4): the students at seed 42. Never the teacher."""
    return row["stage"] != "teacher" and row.get("seed") == 42


def in_series(row: dict) -> bool:
    """Q2, Q7 and Q8 subjects (LE4): the teacher and the seed-42 students."""
    return row["stage"] == "teacher" or row.get("seed") == 42


def reference_defined(ref: float) -> bool:
    """LE7: E1's summed degradation is a divisor only when positive and finite."""
    return math.isfinite(ref) and ref > 0


def rcd_table(stage: str, m: dict, m_clean, e1) -> tuple[dict, dict]:
    """Q5 and Q6 for one student against E1 at seed 42, `e1` = (cells, m_clean) or None. LE7's
    reason comes first: a corruption whose E1 sum is <= 0 is rcd_reference_nonpositive whatever
    the student lacks."""
    per = {}
    for c in CORRUPTIONS:
        if e1 is None:
            per[c] = {"value": None, "reason": "e1_reference_absent"}
            continue
        ref = degradation(e1[0], e1[1], c)
        if ref is not None and not reference_defined(ref):
            per[c] = {"value": None, "reason": "rcd_reference_nonpositive"}
            continue
        mine = degradation(m, m_clean, c)
        if ref is None or mine is None:
            per[c] = {"value": None, "reason": "clean_missing" if m_clean is None else "cell_missing"}
            continue
        try:
            per[c] = {"value": rcd(stage, mine, ref), "reason": None}
        except RobustnessError as e:
            if e.code != "rcd_invalid_reference":
                raise
            per[c] = {"value": None, "reason": "rcd_reference_nonpositive"}
    return per, rcd_mean(per)


def rcd_mean(per: dict) -> dict:
    """Q6 (LE7): the mean over the five corruptions, only when all five are defined."""
    undefined = [per[c]["reason"] for c in CORRUPTIONS if per[c]["value"] is None]
    if undefined:
        return {"value": None, "reason": undefined[0]}
    return {"value": _mean([per[c]["value"] for c in CORRUPTIONS]), "reason": None}


def compute(models: list) -> dict:
    """Q1-Q8 for the loaded models."""
    out = {"clean_miou": {}, "q1_per_cell": {}, "q2_per_severity": {}, "q3_miou_c": {},
           "q4_rpd": {}, "q5_rcd": {}, "q6_rcd_mean": {}, "q7_per_corruption": {},
           "q8_kamann_rother": {}}
    values = {}
    for model in models:
        m_clean = clean_value(model.clean) if model.clean is not None else None
        m = cell_values(model)
        values[model.key] = (m, m_clean)
        out["clean_miou"][model.key] = m_clean
        out["q1_per_cell"][model.key] = {
            cid: {"all_class_miou": m[corruption_roles.parse_cell_id(cid)],
                  "disease_only_miou": disease_pooled(model.cells[cid].stats)}
            for cid in corruption_roles.CELL_IDS if cid in model.cells}
        q3 = miou_c(m)
        out["q3_miou_c"][model.key] = q3
        if q3 is None:
            q4 = {"value": None, "reason": "cell_missing"}
        elif m_clean is None:
            q4 = {"value": None, "reason": "clean_missing"}
        else:
            try:
                q4 = {"value": rpd(m_clean, q3), "reason": None}
            except RobustnessError as e:
                q4 = {"value": None, "reason": e.code}
        out["q4_rpd"][model.key] = q4
    refs = [mo for mo in models if mo.row["stage"] == "E1" and mo.row.get("seed") == 42]
    e1 = values[refs[0].key] if len(refs) == 1 and values[refs[0].key][1] is not None else None
    for model in models:
        m, m_clean = values[model.key]
        if in_rcd(model.row):
            per, mean = rcd_table(model.row["stage"], m, m_clean, e1)
            out["q5_rcd"][model.key] = per
            out["q6_rcd_mean"][model.key] = mean
        if in_series(model.row):
            q2, means = series_values(m)
            out["q2_per_severity"][model.key] = q2
            out["q8_kamann_rother"][model.key] = {"average": kamann_rother(m),
                                                  "series_means": means}
            q7 = {c: {"mean_s1_3": (_mean([m[(c, s)] for s in INFERENTIAL])
                                    if all((c, s) in m for s in INFERENTIAL) else None)}
                  for c in CORRUPTIONS}
            if model.key in out["q5_rcd"]:
                for c in CORRUPTIONS:
                    q7[c]["rcd"] = out["q5_rcd"][model.key][c]["value"]
            out["q7_per_corruption"][model.key] = q7
    return out


# --------------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------------
def check_request(policy: str, split: str) -> None:
    if policy not in POLICIES:
        _refuse("policy_unknown", f"policy must be one of {POLICIES}, got {policy!r}")
    if split != POLICY_SPLIT[policy]:
        _refuse("policy_split_mismatch", f"policy {policy} reads split {POLICY_SPLIT[policy]}, "
                                         f"not {split!r}")


def check_code(code: dict, export: dict) -> None:
    """An official report is written by this code at the export's evaluator commit (TL-9: the
    session runs at S = E), unmodified there."""
    if code.get("code_commit") != export["evaluator_commit"]:
        _refuse("report_commit_mismatch",
                f"this checkout is at {code.get('code_commit')}, the export's evaluator_commit is "
                f"{export['evaluator_commit']}")
    if code.get("code_clean_at_commit") is not True:
        _refuse("report_code_modified",
                f"the report's code files are not unmodified at {code.get('code_commit')} "
                f"(code_clean_at_commit {code.get('code_clean_at_commit')!r})")


def build_report(export: dict, artifacts_root, *, split: str, policy: str,
                 expected_cache: str | None = None, inventory_sha256: str | None = None) -> dict:
    check_request(policy, split)
    code = code_state()
    if policy == "official":
        check_code(code, export)
    rows = [r for r in export["models"] if r["cells"]]
    models = [load_model(Path(artifacts_root), split, r, policy) for r in rows]
    cache = check_set(models, export, policy, expected_cache)
    quantities = compute(models)
    nonofficial = sorted(f"{m.dirs[label]} ({s['run']['artifact_status']})"
                         for m in models for label, s in m.summaries.items()
                         if s["run"]["artifact_status"] != "official")
    official = policy == "official" and not nonofficial
    refs = [m.key for m in models if m.row["stage"] == "E1" and m.row.get("seed") == 42]
    banner = None if official else (
        f"NONOFFICIAL - descriptive robustness outputs under policy {policy}; "
        + (f"{len(nonofficial)} non-official input(s), listed under nonofficial_inputs"
           if nonofficial else "no input is non-official, but the policy is not official"))
    return {
        "schema": SCHEMA,
        "status": "official" if official else "nonofficial",
        "banner": banner,
        "nonofficial_inputs": nonofficial,
        "policy": policy, "split": split,
        "class_space": class_space(),
        "metric": ("dataset-level all-class union-present pooled mIoU of each artifact's totals "
                   "(pooled_miou_from_totals, float64; DL-73, P28)"),
        "descriptive_only": ("descriptive only: no hypothesis test, no bootstrap and no "
                             "significance value; never in family.json (D22)"),
        "kamann_rother": {"severities": {c: list(s) for c, s in KR_SEVERITIES.items()},
                          "rule": "AM-16 item 6, nested (LE8)"},
        "series": {name: {"corruptions": list(cs), "severities": list(ss)}
                   for name, cs, ss in SERIES},
        "inventory": {"sha256": inventory_sha256, "schema": export.get("schema"),
                      "evaluator_commit": export["evaluator_commit"],
                      "manifest_sha256": export.get("manifest_sha256"),
                      "execution_context": export["execution_context"]},
        "cache": cache,
        "rcd_reference": ({"file_id": refs[0], "stage": "E1", "seed": 42} if len(refs) == 1
                          else None),
        "models": [{
            "file_id": m.key, "stage": m.row["stage"], "seed": m.row.get("seed"),
            "precision": m.row["precision"], "model_role": m.row["model_role"],
            "identity_sha256": expected_identity(m.row), "cells": list(m.row["cells"]),
            "missing": list(m.missing),
            "artifacts": {label: {"dir": m.dirs[label], "file_sha256": m.files[label],
                                  "record_sha256": m.record_sha256.get(label),
                                  "artifact_status": m.summaries[label]["run"]["artifact_status"],
                                  "runtime": runtime_tuple(m.summaries[label]),
                                  "repo_commit": m.summaries[label]["run"].get("repo_commit")}
                          for label in m.summaries},
        } for m in models],
        **quantities,
        "provenance": {"code": code, "generated_utc": utc_iso()},
    }


def json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def _fmt(v) -> str:
    return "null" if v is None else f"{v:.6f}"


def _cell(entry: dict) -> str:
    return _fmt(entry["value"]) if entry["value"] is not None else f"null ({entry['reason']})"


def render_markdown(doc: dict) -> str:
    lines = ["# Robustness, descriptive outputs", ""]
    if doc["banner"]:
        lines += [f"> **{doc['banner']}**", ""]
        lines += [f"> - {name}" for name in doc["nonofficial_inputs"]] + [""]
    lines += [f"- Status: {doc['status']}; policy {doc['policy']}; split {doc['split']}.",
              f"- Class space: {doc['class_space']}.", f"- Metric: {doc['metric']}.",
              f"- {doc['descriptive_only'][0].upper()}{doc['descriptive_only'][1:]}.",
              f"- Cache: {doc['cache']['canonical_manifest_sha256']}.",
              f"- rCD reference: {doc['rcd_reference']}.", ""]
    lines += ["## Q3 mIoU-C and Q4 RPD", "", "| model | stage | seed | clean | mIoU-C | RPD % |",
              "|---|---|---|---|---|---|"]
    for m in doc["models"]:
        k = m["file_id"]
        lines.append(f"| {k} | {m['stage']} | {m['seed']} | {_fmt(doc['clean_miou'][k])} | "
                     f"{_fmt(doc['q3_miou_c'][k])} | {_cell(doc['q4_rpd'][k])} |")
    lines += ["", "## Q5 rCD per corruption and Q6 their mean (students at seed 42; E1 = 1)", "",
              "| model | " + " | ".join(CORRUPTIONS) + " | mean |",
              "|---" * (len(CORRUPTIONS) + 2) + "|"]
    for k, per in doc["q5_rcd"].items():
        lines.append(f"| {k} | " + " | ".join(_cell(per[c]) for c in CORRUPTIONS)
                     + f" | {_cell(doc['q6_rcd_mean'][k])} |")
    lines += ["", "## Q7 per corruption: the mean over severities 1-3", "",
              "| model | " + " | ".join(CORRUPTIONS) + " |", "|---" * (len(CORRUPTIONS) + 1) + "|"]
    for k, per in doc["q7_per_corruption"].items():
        lines.append(f"| {k} | " + " | ".join(_fmt(per[c]["mean_s1_3"]) for c in CORRUPTIONS) + " |")
    lines += ["", "## Q8 the Kamann-Rother average (AM-16 item 6: non-noise s1-5, gaussian_noise "
                  "s1-3, nested) and the LE6 series means", "",
              "| model | average | " + " | ".join(name for name, _, _ in SERIES) + " |",
              "|---" * (len(SERIES) + 2) + "|"]
    for k, q8 in doc["q8_kamann_rother"].items():
        lines.append(f"| {k} | {_fmt(q8['average'])} | "
                     + " | ".join(_fmt(q8["series_means"][name]) for name, _, _ in SERIES) + " |")
    for name, _, severities in SERIES:
        lines += ["", f"## Q2 per severity: {name}", "",
                  "| model | " + " | ".join(f"s{s}" for s in severities) + " |",
                  "|---" * (len(severities) + 1) + "|"]
        for k, series in doc["q2_per_severity"].items():
            lines.append(f"| {k} | " + " | ".join(_fmt(series[name][str(s)]) for s in severities)
                         + " |")
    lines += ["", "## Q1 per cell: all-class (disease-only)", ""]
    for k, cells in doc["q1_per_cell"].items():
        lines.append(f"- {k}: " + "; ".join(
            f"{cid} {_fmt(v['all_class_miou'])} ({_fmt(v['disease_only_miou'])})"
            for cid, v in cells.items()))
    code = doc["provenance"]["code"]
    lines += ["", "## Q9 provenance", "",
              f"- Inventory sha256 {doc['inventory']['sha256']}; evaluator commit "
              f"{doc['inventory']['evaluator_commit']}.",
              f"- Code commit {code['code_commit']}; files clean at that commit: "
              f"{code['code_clean_at_commit']}.",
              "- Per-artifact file hashes, record hashes and runtime tuples: "
              "robustness_descriptive.json, `models`.", ""]
    return "\n".join(lines)


def inside_repo(path) -> bool:
    """The path resolved (symlinks followed) is the repository or below it."""
    p = Path(os.path.realpath(str(path)))
    return p == REPO or REPO in p.parents


def check_out_dir(out_dir) -> Path:
    """TL section 2 (as given and resolved), outside the repository (resolved), not existing."""
    problem = created_path_problem(out_dir)
    if problem:
        _refuse("out_dir_name_invalid", f"the out-dir {problem}; TL section 2 forbids 'test' in any "
                                        "case and :<>\"\\|?* in any component of a created path")
    out = Path(os.path.abspath(str(out_dir)))
    if inside_repo(out):
        _refuse("out_dir_inside_repo", "the robustness outputs are never written inside the "
                                       "repository")
    if out.exists():
        _refuse("out_dir_exists", f"refusing to overwrite {out}")
    return out


def _discard_dir(tmp: Path) -> None:
    """Remove this writer's own temporary directory after a failed write (never another path)."""
    shutil.rmtree(tmp, ignore_errors=True)


def verify_written(tmp: Path, payloads: dict) -> None:
    """Every file, and SHA256SUMS, reads back as written."""
    sums = {}
    for line in (tmp / SUMS_NAME).read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name] = digest
    for name, data in payloads.items():
        want = hashlib.sha256(data).hexdigest()
        if _sha256_file(tmp / name) != want or sums.get(name) != want:
            _refuse("output_verify_failed", f"{name} does not read back as written")


def write_dir_atomic(out_dir, payloads: dict) -> Path:
    """The files and their SHA256SUMS in a temporary sibling, re-read, then one rename; a failure
    removes the temporary sibling and leaves nothing at the out-dir."""
    out = check_out_dir(out_dir)
    tmp = out.parent / f".{out.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir(parents=True)
    try:
        for name, data in payloads.items():
            (tmp / name).write_bytes(data)
        sums = "".join(f"{hashlib.sha256(payloads[n]).hexdigest()}  {n}\n" for n in sorted(payloads))
        (tmp / SUMS_NAME).write_bytes(sums.encode("utf-8"))
        verify_written(tmp, payloads)
        tmp.rename(out)
    except BaseException:
        _discard_dir(tmp)
        raise
    return out


def write_outputs(out_dir, doc: dict) -> Path:
    return write_dir_atomic(out_dir, {JSON_NAME: json_bytes(doc),
                                      MD_NAME: render_markdown(doc).encode("utf-8")})


__all__ = ["CODE_FILES", "CORRUPTIONS", "JSON_NAME", "KR_SEVERITIES", "MD_NAME", "NON_NOISE",
           "POLICIES", "POLICY_SPLIT", "RUNTIME_KEYS", "ReportRefusal", "SCHEMA", "SERIES",
           "SUMS_NAME", "build_report", "cell_values", "check_code", "check_out_dir",
           "check_request", "clean_value", "code_state", "compute", "degradation",
           "disease_pooled", "expected_identity", "in_rcd", "in_series", "inside_repo",
           "json_bytes", "kamann_rother", "load_model", "miou_c", "pooled", "rcd_mean",
           "rcd_table", "reference_defined", "render_markdown", "runtime_problems",
           "series_values", "verify_written", "write_dir_atomic", "write_outputs"]
