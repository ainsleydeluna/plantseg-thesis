"""Read-only ingestion of a completed evaluation run directory (A3a).

Consumes the frozen `plantseg-eval/1.0.0` artifact produced by `src/eval/artifacts.py` and
exposes immutable typed records for the statistical layer. Implements
docs/STATISTICAL_ANALYSIS_CONTRACT.md section 10 (official integrity policy).

Nothing here computes a metric: every value is read from the artifact, and the sufficient
statistics are re-checked against the recorded dataset totals rather than recomputed from pixels.

AM-5 (lane L-AM5; EVALUATION_CONTRACT section 3.2, STATISTICAL_ANALYSIS_CONTRACT section 10): a row
whose evaluated mask has no disease ground truth (`n_eligible_disease_only == 0`) is excluded from
every per-image disease-only analysis. The flag is read from the artifact when present (and must equal
that derivation) or derived for a pre-lane artifact. OFFICIAL and REHEARSAL drop the flagged rows from
`EvaluationRun.records`; NONOFFICIAL_SMOKE keeps every row. `EvaluationRun.stats` always keeps every
row, because the excluded images stay in every dataset-level metric.

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
from ..eval.artifacts import (AM5_RULE, ARTIFACT_FILES, ARTIFACT_SCHEMA_VERSION,  # noqa: F401
                              MANIFEST_NAME, METRIC_PROTOCOL, SCHEMA_VERSION,
                              am5_excluded_ids_sha256, verify_artifact)

EXPECTED_ROWS_TEST = 1561
EXPECTED_ROWS_VAL = 846
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LAYOUT = re.compile(r"^plantseg-eval-artifact/(\d+)\.(\d+)\.(\d+)$")
AM5_MIN_LAYOUT = (1, 1, 0)           # the first artifact layout that carries the AM-5 fields

STATUS_OK = "ok"
UNDEFINED_STATUSES = ("undefined_no_eligible_class", "excluded_data_integrity", "evaluation_error")

AM5_SOURCE_ARTIFACT = "artifact"     # rows carry am5_excluded (artifact_schema_version 1.1.0)
AM5_SOURCE_DERIVED = "derived"       # pre-lane artifact: derived from n_eligible_disease_only == 0


class Policy(Enum):
    """Explicit named policy boundary -- never an ambiguous `strict=True` flag."""
    OFFICIAL = "official"
    NONOFFICIAL_SMOKE = "nonofficial_smoke"
    #: The OFFICIAL row rules on real VAL artifacts (the AM-17 item 9 dress rehearsal, the L-AM5 d5
    #: check): VAL only, so TEST is never read outside the M11 isolation path.
    REHEARSAL = "rehearsal"


#: Policies with the OFFICIAL row rules: AM-5 rows dropped, any other undefined primary value fatal,
#: and (src/stats/align.py) a metric-implementation mismatch or a differing AM-5 set fatal.
STRICT_POLICIES = (Policy.OFFICIAL, Policy.REHEARSAL)


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
    am5_excluded: bool | None = None     # AM-5 flag; None -> derived, any given value must agree

    def __post_init__(self) -> None:
        derived = self.n_eligible_disease_only == 0
        if self.am5_excluded is None:
            object.__setattr__(self, "am5_excluded", derived)
        elif self.am5_excluded is not derived:
            raise IngestError(f"row {self.image_id!r}: am5_excluded={self.am5_excluded!r} disagrees "
                              f"with n_eligible_disease_only={self.n_eligible_disease_only}")


@dataclass(frozen=True)
class Am5Exclusion:
    """The AM-5 zero-disease exclusion of one run (ground truth only, so identical across models)."""
    rule: str
    source: str                           # AM5_SOURCE_ARTIFACT or AM5_SOURCE_DERIVED
    applied: bool                         # flagged rows dropped from EvaluationRun.records
    excluded_ids: tuple[str, ...]         # sorted image_id
    excluded_clean_ids: tuple[str, ...]   # sorted clean_image_id (the robustness key)
    included_count: int
    excluded_ids_sha256: str              # sha256 of "\n".join(excluded_ids)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded_ids)


@dataclass(frozen=True)
class Am5Pair:
    """AM-5 counts of a paired analysis; the two excluded sets are identical by construction."""
    n_excluded_am5: int
    n_included: int
    excluded_ids_sha256: str


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
    records: tuple[PerImageRecord, ...]        # canonical order; OFFICIAL/REHEARSAL: AM-5 rows dropped
    stats: PerImageSufficientStats             # every row, always
    policy: Policy
    am5: Am5Exclusion | None = None            # set by load_run

    def by_id(self) -> Mapping[str, PerImageRecord]:
        return {r.image_id: r for r in self.records}

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(r.image_id for r in self.records)

    def _am5_or_raise(self) -> Am5Exclusion:
        if self.am5 is None:
            raise IngestError("AM-5 exclusion unknown: this run was not built by load_run")
        return self.am5

    @property
    def n_excluded_am5(self) -> int:
        return self._am5_or_raise().excluded_count

    @property
    def n_included(self) -> int:
        return self._am5_or_raise().included_count


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
    """Load and validate one evaluation run under an explicit policy.

    AM-5: OFFICIAL and REHEARSAL drop the zero-disease rows from `records` and abort on any other
    undefined primary value; `run.am5` records the exclusion under every policy.
    """
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
    # The OFFICIAL branch is frozen (docs/lane_specs/part1.md: "the TEST driver in ingest.py:202-203").
    if policy is Policy.OFFICIAL:
        _require(ident.artifact_status == "official",
                 f"OFFICIAL analysis requires artifact_status='official', got "
                 f"{ident.artifact_status!r}")
        _require(ident.random_init is False, "OFFICIAL analysis forbids random_init inputs")
        _require(ident.split == "test", f"OFFICIAL clean analysis requires split='test'")
        _require(ident.expected_rows == ident.actual_rows == EXPECTED_ROWS_TEST,
                 f"OFFICIAL analysis requires {EXPECTED_ROWS_TEST} rows, got "
                 f"expected={ident.expected_rows} actual={ident.actual_rows}")
    elif policy is Policy.REHEARSAL:
        _require(ident.artifact_status in ("provisional", "official"),
                 f"REHEARSAL analysis requires a real-run artifact (provisional or official), got "
                 f"{ident.artifact_status!r}")
        _require(ident.random_init is False, "REHEARSAL analysis forbids random_init inputs")
        _require(ident.split == "val", f"REHEARSAL analysis reads VAL artifacts only (TEST stays "
                                       f"behind the M11 isolation path), got split={ident.split!r}")
        _require(ident.expected_rows == ident.actual_rows == EXPECTED_ROWS_VAL,
                 f"REHEARSAL analysis requires {EXPECTED_ROWS_VAL} rows, got "
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
    rows, flag_in_file = [], []
    for line in (run_dir / "per_image.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
            IngestError(f"non-finite JSON constant {x!r} in per_image.jsonl")))
        flag_in_file.append("am5_excluded" in o)
        _require(not flag_in_file[-1] or isinstance(o["am5_excluded"], bool),
                 f"row {o.get('image_id')!r}: am5_excluded must be a JSON boolean")
        rows.append(PerImageRecord(
            image_id=o["image_id"], clean_image_id=o["clean_image_id"],
            manifest_index=int(o["manifest_index"]),
            all_class_miou=o["all_class_miou"], all_class_miou_status=o["all_class_miou_status"],
            disease_only_miou=o["disease_only_miou"],
            disease_only_miou_status=o["disease_only_miou_status"],
            n_eligible_all_class=int(o["n_eligible_all_class"]),
            n_eligible_disease_only=int(o["n_eligible_disease_only"]),
            am5_excluded=o.get("am5_excluded")))       # absent -> derived; present -> must agree

    _require(len(rows) == ident.actual_rows,
             f"per_image.jsonl has {len(rows)} rows, summary says {ident.actual_rows}")
    idxs = [r.manifest_index for r in rows]
    _require(idxs == sorted(idxs), "per_image.jsonl is not sorted by manifest_index")
    ids = [r.image_id for r in rows]
    _require(len(set(ids)) == len(ids), "duplicate image_id in per_image.jsonl")
    _require(len({i.casefold() for i in ids}) == len(ids), "case-folded image_id collision")

    am5 = _am5_exclusion(summary, rows, flag_in_file, applied=policy in STRICT_POLICIES)
    records = apply_am5_policy(rows, policy)

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

    return EvaluationRun(identity=ident, records=records, stats=stats, policy=policy, am5=am5)


# --------------------------------------------------------------------------------------------------
# AM-5 (lane L-AM5)
# --------------------------------------------------------------------------------------------------
def apply_am5_policy(rows, policy: Policy) -> tuple[PerImageRecord, ...]:
    """The rows a per-image analysis sees under `policy`.

    OFFICIAL and REHEARSAL drop the AM-5 rows and abort on any other undefined primary value, naming
    it; NONOFFICIAL_SMOKE keeps every row (undefined pairs stay visible downstream).
    """
    if policy not in STRICT_POLICIES:
        return tuple(rows)
    bad = [r.image_id for r in rows if r.disease_only_miou is None and not r.am5_excluded]
    _require(not bad, f"{policy.name} analysis forbids undefined primary metric on a row AM-5 does "
                      f"not exclude; {len(bad)} row(s) undefined, e.g. {bad[:3]}")
    return tuple(r for r in rows if not r.am5_excluded)


def _layout(summary: Mapping):
    """(major, minor, patch) of summary.json `artifact_schema_version`; None for a pre-lane artifact.

    Minor and patch bumps are additive (readers tolerate extra fields); another major is refused.
    """
    v = summary.get("artifact_schema_version")
    if v is None:
        return None
    m = _LAYOUT.match(v) if isinstance(v, str) else None
    _require(m is not None and int(m.group(1)) == AM5_MIN_LAYOUT[0],
             f"unsupported artifact_schema_version {v!r} (this reader knows "
             f"plantseg-eval-artifact/{AM5_MIN_LAYOUT[0]}.x.y)")
    return tuple(int(g) for g in m.groups())


def _am5_exclusion(summary: Mapping, rows, flag_in_file, *, applied: bool) -> Am5Exclusion:
    """Check the artifact's AM-5 fields against the per-row derivation and summarise the exclusion.

    A lane artifact (every row carries the flag) must declare an artifact layout of at least
    AM5_MIN_LAYOUT and an `am5` block equal to the rows' derivation; a pre-lane artifact (no row
    carries it) must declare neither.
    """
    layout = _layout(summary)
    if rows and all(flag_in_file):
        source = AM5_SOURCE_ARTIFACT
        _require(layout is not None and layout >= AM5_MIN_LAYOUT
                 and isinstance(summary.get("am5"), dict),
                 f"per-image rows carry am5_excluded but summary.json has artifact_schema_version="
                 f"{summary.get('artifact_schema_version')!r} and no valid am5 block (expected at "
                 f"least {ARTIFACT_SCHEMA_VERSION!r})")
    elif not any(flag_in_file):
        source = AM5_SOURCE_DERIVED
        _require((layout is None or layout < AM5_MIN_LAYOUT) and "am5" not in summary,
                 "summary.json declares the AM-5 layout but the per-image rows lack am5_excluded")
    else:
        raise IngestError("per_image.jsonl mixes rows with and without am5_excluded")

    for r in rows:
        _require(not r.am5_excluded or (r.disease_only_miou is None
                                        and r.disease_only_miou_status == UNDEFINED_STATUSES[0]),
                 f"row {r.image_id!r} has no disease ground truth but disease_only_miou="
                 f"{r.disease_only_miou!r} ({r.disease_only_miou_status!r})")
    excluded = tuple(sorted(r.image_id for r in rows if r.am5_excluded))
    am5 = Am5Exclusion(
        rule=AM5_RULE, source=source, applied=applied, excluded_ids=excluded,
        excluded_clean_ids=tuple(sorted(r.clean_image_id for r in rows if r.am5_excluded)),
        included_count=len(rows) - len(excluded),
        excluded_ids_sha256=am5_excluded_ids_sha256(excluded))
    if source == AM5_SOURCE_ARTIFACT:
        expect = {"rule": am5.rule, "excluded_count": am5.excluded_count,
                  "included_count": am5.included_count,
                  "excluded_ids_sha256": am5.excluded_ids_sha256}
        _require(summary["am5"] == expect,
                 f"summary.am5 {summary['am5']!r} disagrees with the per-image flags {expect!r}")
    return am5


def am5_pair(baseline: EvaluationRun, candidate: EvaluationRun) -> Am5Pair:
    """AM-5 counts for a pair, aborting unless both runs exclude exactly the same images.

    The exclusion depends on ground truth only, so on one split manifest a difference is a manifest
    mismatch, never a model effect.
    """
    a, b = baseline._am5_or_raise(), candidate._am5_or_raise()
    ia, ib = baseline.identity, candidate.identity
    _require(ia.split == ib.split and ia.split_manifest_sha256 == ib.split_manifest_sha256,
             f"AM-5 pair: the runs score different manifests ({ia.split} "
             f"{ia.split_manifest_sha256[:12]}... vs {ib.split} {ib.split_manifest_sha256[:12]}...)")
    if a.excluded_ids != b.excluded_ids:
        only_a = sorted(set(a.excluded_ids) - set(b.excluded_ids))
        only_b = sorted(set(b.excluded_ids) - set(a.excluded_ids))
        raise IngestError(
            f"AM-5 pair: the excluded id sets differ ({a.excluded_count} {ia.run_id!r} vs "
            f"{b.excluded_count} {ib.run_id!r}; only in {ia.run_id!r}: {only_a[:3]}, only in "
            f"{ib.run_id!r}: {only_b[:3]}). Ground truth decides the set, so this is a manifest "
            "mismatch.")
    return Am5Pair(n_excluded_am5=a.excluded_count, n_included=a.included_count,
                   excluded_ids_sha256=a.excluded_ids_sha256)
