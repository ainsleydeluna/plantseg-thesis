"""Teacher diagnostics seam (AM-18 item 1; lane L-TEACHER-DIAG): gates, the teacher load, provenance.

Shared by the DIAG scripts (scripts/teacher_d1_nmf_sensitivity.py, teacher_d2_calibration.py,
score_teacher_train.py, teacher_d3_perclass_strata.py, hash_split_files.py, dedup_val_scores.py). It
imports nothing from `src.stats` (whose package import pulls the whole statistics stack), and no teacher
file is edited: the KD adapter, the NMF stream and the evaluator are reached through their public load
paths and attributes.

The teacher load (`load_teacher`) has exactly two call lines, one per mode:
  kd         load_frozen_teacher(ckpt, config_path=cfg)       the KD trainer's form (train_distill.py)
  evaluator  load_teacher_model(resolved, config_path=cfg)    the evaluator's form (evaluate_model.py run)
A stub passes `builder=segnext_builder(model_factory=...)` to the same two calls. The NMF stream the
checks use is the adapter's live object (`frozen.teacher.nmf_stream`), read right after the begin call;
the dict that call returns is kept as the description and must equal `stream.describe()` at that moment.

Gates, in the order the scripts apply them:
  flags (no file read)        sha256 format, role and arm flags, correction flags, commit and repeat flags,
                              --generated-utc refused in real mode; TeacherProvenance field count (real)
  commit binding (real)       HEAD == --script-commit; CODE_FILES present and clean at HEAD;
                              `git status --porcelain=v1 -- src configs scripts` empty (count only)
  single output (real)        one non-smoke output per script, role, arm and purpose unless --repeat-of
  teacher inputs              stub: checkpoint <= 16 MiB by stat, sha256 not a registered teacher sha;
                              own sha256 == flag; role pin (record: 8c0e649a... and the repo config at blob
                              3c6a7b28...; arm: config sha256 == its flag, checkpoint not a record sha)
  load and after-load         validate_teacher_artifact; the load; provenance sha == verified sha;
                              stub: no mmseg imported, <= 1e6 parameters; real: every TeacherProvenance
                              field present and non-empty; one isolated NMF module holding the stream;
                              segmentor in eval mode; loaded_state_sha256; the nine frozen blob ids.

Refusals raise `Refused` (exit 2); a check the script decides raises `Stop` (exit 1).
Import-time behaviour is side-effect free apart from importing two standard-library-only scripts
(scripts/gap_bootstrap_val.py for PAIRING, scripts/smoke_frozen_blobs.py for FROZEN and git_blob_id).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scripts.gap_bootstrap_val import PAIRING
from scripts.smoke_frozen_blobs import FROZEN, git_blob_id

REPO = Path(__file__).resolve().parents[2]
LANE = "L-TEACHER-DIAG"
AUTHORITY = "AM-18 item 1 (final, not yet committed); lane L-TEACHER-DIAG"

# ---- the pins of record (imported, never retyped) ----
RECORD_SHA256 = PAIRING["teacher"]["checkpoint_sha256"]
REGISTERED_TEACHER_SHA256 = frozenset({RECORD_SHA256})
R3_VAL_MIOU = PAIRING["teacher"]["reference"]
R3_TOLERANCE = PAIRING["teacher"]["tolerance"]
TEACHER_CONFIG_REL = "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
RECORD_CONFIG_BLOB = FROZEN[TEACHER_CONFIG_REL][0]
EXPECTED_PROVENANCE_FIELDS = 12          # the K-part ruling on TeacherProvenance (DL-50, DL-53)
STUB_MAX_CKPT_BYTES = 16 * 2 ** 20
STUB_MAX_PARAMETERS = 10 ** 6
VAL_ROWS, TRAIN_ROWS = 846, 5367

ROLES = ("record", "arm")
ARM_IDS = ("R1", "R2")
PURPOSES = ("item1", "control")
CORRECTION_STATES = ("available", "declined", "no_approval")
REPEAT_CASES = ("i", "ii")
EXIT_OK, EXIT_STOP, EXIT_REFUSED, EXIT_ERROR = 0, 1, 2, 4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DL_ID_RE = re.compile(r"^DL-\d+$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: every repository file that shapes a number in a DIAG output, reused modules included. One list for
#: every script, so every output of one pin carries the same code digest (ruling R1).
CODE_FILES = (
    "configs/augment.py", "configs/data.py", "configs/distill.py", "configs/plantseg_class_map.json",
    TEACHER_CONFIG_REL,
    "scripts/build_train_strata.py", "scripts/dedup_val_scores.py", "scripts/evaluate_model.py",
    "scripts/gap_bootstrap_val.py", "scripts/hash_split_files.py", "scripts/score_teacher_train.py",
    "scripts/smoke_frozen_blobs.py", "scripts/teacher_d1_nmf_sensitivity.py",
    "scripts/teacher_d2_calibration.py", "scripts/teacher_d3_perclass_strata.py",
    "src/data/dataset.py", "src/data/isolation.py", "src/data/transforms.py",
    "src/distill/nmf_stream.py", "src/distill/segnext_teacher.py", "src/distill/teacher.py",
    "src/eval/adapters.py", "src/eval/artifacts.py", "src/eval/calibration.py", "src/eval/eval_runtime.py",
    "src/eval/evaluate.py", "src/eval/metrics.py", "src/eval/model_loading.py",
    "src/eval/nmf_sensitivity.py", "src/eval/protocols.py", "src/eval/stage_artifacts.py",
    "src/eval/teacher_diag.py",
    "src/quant/calibration.py",
    "src/stats/align.py", "src/stats/eligibility.py", "src/stats/gap.py", "src/stats/ingest.py",
    "src/stats/noninferiority.py", "src/stats/val_artifacts.py",
    "src/training/losses.py", "src/training/teacher_components.py",
)


class Refused(RuntimeError):
    """A refusal before anything is computed (exit 2)."""


class Stop(RuntimeError):
    """A STOP the script decided (exit 1)."""


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(UTC_FORMAT)


def utc_stamp(utc: str) -> str:
    return utc.replace("-", "").replace(":", "")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def name_digests(names) -> str:
    """The count and the sha256 of each name: how a refusal names paths without printing them."""
    names = [str(n) for n in names]
    return (f"{len(names)} offending entr{'y' if len(names) == 1 else 'ies'}, sha256 of each name: "
            + ", ".join(hashlib.sha256(n.encode("utf-8")).hexdigest() for n in names))


def refuse_test_path(path, what: str) -> None:
    """P21: the as-given string first (no filesystem call), then the resolved path."""
    s = str(path)
    if "test" in s.lower():
        raise Refused(f"{what}: a path containing 'test' is refused ({name_digests([s])})")
    real = os.path.realpath(s)
    if "test" in real.lower():
        raise Refused(f"{what}: a path resolving to a 'test' location is refused ({name_digests([real])})")


def refuse_test_names(names, what: str) -> None:
    bad = [n for n in names if "test" in str(n).lower()]
    if bad:
        raise Refused(f"{what}: names containing 'test' are refused ({name_digests(bad)})")


def require_outside_repo(path, what: str) -> Path:
    p = Path(path).resolve()
    if p == REPO or REPO in p.parents:
        raise Refused(f"{what} must resolve outside the repository ({p})")
    return p


def read_json_strict(path) -> dict:
    def _reject(c):
        raise Refused(f"{Path(path).name}: non-finite JSON constant {c!r}")
    doc = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject)
    if not isinstance(doc, dict):
        raise Refused(f"{Path(path).name} is not a JSON object")
    return doc


def json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_json_exclusive(path, doc: dict) -> Path:
    """Serialize first (allow_nan=False), then create exclusively; a failed write leaves no file."""
    data = json_bytes(doc)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "xb")
    try:
        with fh:
            fh.write(data)
    except BaseException:
        p.unlink(missing_ok=True)
        raise
    return p


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------------------------------
# git and code provenance (P1, P26)
# --------------------------------------------------------------------------------------------------
def _git(*args) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True, timeout=120,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def git_head() -> str | None:
    out = (_git("rev-parse", "HEAD") or "").strip()
    return out if COMMIT_RE.match(out) else None


def git_status(paths) -> str | None:
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("git_status needs explicit paths (never the whole tree)")
    return _git("status", "--porcelain=v1", "--", *paths)


def code_provenance() -> dict:
    """HEAD, each CODE_FILES sha256, whether they are clean at HEAD, and one digest over them."""
    files = {rel: (file_sha256(REPO / rel) if (REPO / rel).is_file() else None) for rel in CODE_FILES}
    digest = hashlib.sha256("".join(f"{files[r] or '-'}  {r}\n" for r in sorted(files)).encode("utf-8"))
    status = git_status(CODE_FILES)
    return {"head": git_head(), "code_files_clean_at_head": None if status is None else status == "",
            "code_files": files, "code_digest": digest.hexdigest(),
            "code_digest_definition": "sha256 over '<sha256>  <path>\\n' for each CODE_FILES path, sorted"}


def require_commit_binding(script_commit: str, dl_id: str) -> dict:
    """P26: HEAD equals the flag; CODE_FILES present and clean; src, configs and scripts clean."""
    head = git_head()
    if head is None:
        raise Refused("git HEAD could not be read; a real run needs the dedicated clone at its pin")
    if head != script_commit:
        raise Refused(f"HEAD {head} != --script-commit {script_commit}")
    missing = [r for r in CODE_FILES if not (REPO / r).is_file()]
    if missing:
        raise Refused(f"{len(missing)} CODE_FILES path(s) missing at HEAD")
    status = git_status(CODE_FILES)
    if status is None:
        raise Refused("git status over CODE_FILES failed")
    if status.strip():
        raise Refused(f"{len(status.splitlines())} CODE_FILES entr(ies) dirty or untracked at HEAD")
    tree = git_status(("src", "configs", "scripts"))
    if tree is None or tree.strip():
        n = "?" if tree is None else len(tree.splitlines())
        raise Refused(f"git status --porcelain=v1 -- src configs scripts lists {n} entr(ies)")
    prov = code_provenance()
    prov["script_commit"] = script_commit
    prov["script_commit_dl_id"] = dl_id
    return prov


# --------------------------------------------------------------------------------------------------
# flags (P3, P5, P26-P28; C2) -- nothing here reads a file
# --------------------------------------------------------------------------------------------------
def check_common_flags(args, *, real: bool) -> None:
    gen = getattr(args, "generated_utc", None)
    if gen is not None:
        if real:
            raise Refused("--generated-utc is refused in a real run: outputs are stamped from the script's "
                          "own clock")
        if not UTC_RE.match(gen):
            raise Refused(f"--generated-utc must look like 2026-10-01T00:00:00Z, got {gen!r}")
    commit, dl = getattr(args, "script_commit", None), getattr(args, "script_commit_dl_id", None)
    if real and (commit is None or dl is None):
        raise Refused("a real run requires --script-commit and --script-commit-dl-id")
    if commit is not None and not COMMIT_RE.match(commit):
        raise Refused("--script-commit must be 40 lowercase hex characters")
    if dl is not None and not DL_ID_RE.match(dl):
        raise Refused("--script-commit-dl-id must look like DL-<n>")
    rep = [getattr(args, k, None) for k in ("repeat_of", "repeat_case", "repeat_dl_id")]
    if any(v is not None for v in rep):
        if not all(v is not None for v in rep):
            raise Refused("--repeat-of, --repeat-case and --repeat-dl-id go together")
        if not SHA256_RE.match(rep[0]):
            raise Refused("--repeat-of must be a 64-hex sha256")
        if rep[1] not in REPEAT_CASES:
            raise Refused(f"--repeat-case must be one of {REPEAT_CASES}")
        if not DL_ID_RE.match(rep[2]):
            raise Refused("--repeat-dl-id must look like DL-<n>")


def check_teacher_flags(args, *, roles=ROLES) -> None:
    sha = getattr(args, "teacher_ckpt_sha256", None)
    if sha is None or not SHA256_RE.match(sha):
        raise Refused("--teacher-ckpt-sha256 must match ^[0-9a-f]{64}$ (missing, empty, uppercase or short "
                      "values are refused)")
    if not getattr(args, "teacher_ckpt", None) or not getattr(args, "teacher_config", None):
        raise Refused("--teacher-ckpt and --teacher-config are required")
    role = getattr(args, "teacher_role", None)
    if role not in roles:
        raise Refused(f"--teacher-role must be one of {roles} here, got {role!r}")
    arm = [getattr(args, k, None) for k in ("arm_id", "arm_dl_id", "teacher_config_sha256")]
    if role == "arm":
        if arm[0] not in ARM_IDS:
            raise Refused(f"the arm role requires --arm-id in {ARM_IDS}")
        if arm[1] is None or not DL_ID_RE.match(arm[1]):
            raise Refused("the arm role requires --arm-dl-id DL-<n>")
        if arm[2] is None or not SHA256_RE.match(arm[2]):
            raise Refused("the arm role requires --teacher-config-sha256 (64 lowercase hex)")
    elif any(v is not None for v in arm):
        raise Refused("--arm-id, --arm-dl-id and --teacher-config-sha256 belong to the arm role")


def check_correction_flags(args) -> None:
    """P28: both D1 parts, every role, before any file is read."""
    if getattr(args, "correction_state", None) not in CORRECTION_STATES:
        raise Refused(f"--correction-state must be one of {CORRECTION_STATES}")
    dl = getattr(args, "correction_dl_id", None)
    if dl is None or not re.match(r"^DL-\d+$", dl):
        raise Refused("--correction-dl-id must match ^DL-\\d+$")


def provenance_fields() -> list[str]:
    from src.distill.teacher import TeacherProvenance
    return [f.name for f in dataclasses.fields(TeacherProvenance)]


def require_provenance_field_count(real: bool) -> None:
    """P8, before any file is read: the merged TeacherProvenance must have the ruled field count."""
    names = provenance_fields()
    if real and len(names) != EXPECTED_PROVENANCE_FIELDS:
        raise Refused(f"TeacherProvenance has {len(names)} fields {names}, the K-part ruling expects "
                      f"{EXPECTED_PROVENANCE_FIELDS}: real runs wait for K-part (P8)")


# --------------------------------------------------------------------------------------------------
# one output per kind (P27)
# --------------------------------------------------------------------------------------------------
def require_single_output(out_dir: Path, kind: str, args, *, exempt: bool = False) -> dict | None:
    """Refuse a second non-smoke output of this kind in out_dir unless --repeat-of names one of them."""
    pattern = re.compile(rf"^{re.escape(kind)}_\d{{8}}T\d{{6}}Z\.json$")
    existing = []
    if Path(out_dir).is_dir():
        for e in os.scandir(out_dir):
            if e.is_file() and pattern.match(e.name):
                try:
                    status = read_json_strict(e.path).get("artifact_status")
                except Exception:                  # noqa: BLE001 -- unreadable counts as an output
                    status = None
                if status != "smoke":
                    existing.append(file_sha256(e.path))
    repeat_of = getattr(args, "repeat_of", None)
    if exempt:
        return None
    if existing and repeat_of is None:
        raise Refused(f"--out-dir already holds {len(existing)} non-smoke {kind} output(s); a repeat needs "
                      "--repeat-of <its sha256> --repeat-case {i,ii} --repeat-dl-id DL-<n>")
    if repeat_of is not None:
        if repeat_of not in existing:
            raise Refused(f"--repeat-of {repeat_of[:12]}... is not a {kind} output in --out-dir")
        return {"repeat_of": repeat_of, "repeat_case": args.repeat_case, "repeat_dl_id": args.repeat_dl_id}
    return None


# --------------------------------------------------------------------------------------------------
# teacher inputs, load and after-load checks (P2, P3, P5-P8; C1)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TeacherInputs:
    ckpt: Path
    sha256: str
    ckpt_bytes: int
    config: Path
    config_sha256: str
    config_blob: str
    role: str
    arm_id: str | None
    arm_dl_id: str | None


def verify_teacher_inputs(args, *, stub: bool) -> TeacherInputs:
    """P2 file guards, P5 own sha256, P3 role pins. Reads bytes; unpickles nothing."""
    ckpt = Path(args.teacher_ckpt)
    if not ckpt.is_file():
        raise Refused(f"teacher checkpoint not found: {ckpt}")
    size = os.stat(ckpt).st_size
    if stub and size > STUB_MAX_CKPT_BYTES:
        raise Refused(f"stub mode refuses a checkpoint over {STUB_MAX_CKPT_BYTES} bytes ({size})")
    sha = file_sha256(ckpt)
    if sha != args.teacher_ckpt_sha256:
        raise Refused(f"teacher checkpoint sha256 {sha} != --teacher-ckpt-sha256 {args.teacher_ckpt_sha256}")
    role = args.teacher_role
    if stub and sha in REGISTERED_TEACHER_SHA256:
        raise Refused("stub mode refuses a registered teacher checkpoint")
    if role == "record" and not stub and sha != RECORD_SHA256:
        raise Refused(f"the record role requires the teacher of record {RECORD_SHA256}")
    if role == "arm" and sha in REGISTERED_TEACHER_SHA256:
        raise Refused("the arm role refuses a record checkpoint sha256")
    config = Path(args.teacher_config)
    if not config.is_file():
        raise Refused(f"teacher config not found: {config}")
    data = config.read_bytes()
    config_sha, blob = hashlib.sha256(data).hexdigest(), git_blob_id(data)
    if role == "record":
        if config.resolve() != (REPO / TEACHER_CONFIG_REL).resolve():
            raise Refused(f"the record role requires the repository config {TEACHER_CONFIG_REL}")
        if blob != RECORD_CONFIG_BLOB:
            raise Refused(f"teacher config git blob {blob} != the frozen {RECORD_CONFIG_BLOB}")
    elif config_sha != args.teacher_config_sha256:
        raise Refused(f"teacher config sha256 {config_sha} != --teacher-config-sha256")
    return TeacherInputs(ckpt=ckpt, sha256=sha, ckpt_bytes=size, config=config, config_sha256=config_sha,
                         config_blob=blob, role=role, arm_id=getattr(args, "arm_id", None),
                         arm_dl_id=getattr(args, "arm_dl_id", None))


@dataclass
class LoadedTeacher:
    mode: str
    frozen: object
    adapter: object
    segmentor: object
    eval_model: object | None
    stream: object                       # the adapter's live NMFStream (C1)
    stream_description: dict             # what the begin call returned, == stream.describe() then
    resolved: dict


def load_teacher(mode: str, inputs: TeacherInputs, *, model_factory=None) -> LoadedTeacher:
    """P6: one function, two load lines. `model_factory` is the smoke stub; the CLI passes None."""
    from src.distill.nmf_stream import M4_NMF_SEED
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter, segnext_builder
    from src.distill.teacher import TeacherStackMissing, load_frozen_teacher
    from src.eval.model_loading import load_teacher_model
    from src.eval.stage_artifacts import StageArtifactError, validate_teacher_artifact

    if mode not in ("kd", "evaluator"):
        raise ValueError(f"unknown load mode {mode!r}")
    kw = {} if model_factory is None else {"builder": segnext_builder(model_factory=model_factory)}
    try:
        resolved = validate_teacher_artifact(inputs.ckpt, expected_sha256=inputs.sha256)
    except StageArtifactError as e:
        raise Refused(f"validate_teacher_artifact: {e}") from e
    try:
        if mode == "kd":
            frozen = load_frozen_teacher(str(inputs.ckpt), config_path=str(inputs.config), **kw)
            eval_model = None
            description = frozen.begin_nmf_stream("M4-KD", M4_NMF_SEED)
        else:
            eval_model, _ = load_teacher_model(resolved, config_path=str(inputs.config), **kw)
            frozen = eval_model.teacher
            description = eval_model.nmf_policy
    except TeacherStackMissing as e:
        raise Refused(f"the teacher stack is missing: {e}") from e
    adapter = frozen.teacher
    if not isinstance(adapter, SegNeXtTeacherAdapter):
        raise Refused(f"the loaded teacher is a {type(adapter).__name__}, not the SegNeXt KD adapter")
    stream = getattr(adapter, "nmf_stream", None)            # the live object, read right after the begin
    if description is None or stream is None:
        raise Refused("the teacher exposes no isolated NMF stream (begin_nmf_stream returned None)")
    if stream.describe() != description:
        raise Stop("the adapter's stream does not match the description its begin call returned")
    return LoadedTeacher(mode=mode, frozen=frozen, adapter=adapter, segmentor=adapter.model,
                         eval_model=eval_model, stream=stream, stream_description=dict(description),
                         resolved={k: str(v) if isinstance(v, Path) else v for k, v in resolved.items()
                                   if k != "spec"})


def loaded_state_sha256(module) -> str:
    """sha256 over the sorted state_dict keys with each tensor's dtype, shape and bytes."""
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8") + b"\0" + str(t.dtype).encode("ascii") + b"\0"
                 + repr(tuple(t.shape)).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def frozen_blob_record() -> dict:
    """P7: the nine frozen blob ids recomputed from scripts/smoke_frozen_blobs.py's table."""
    out, bad = {}, []
    for rel, (blob, _sha) in FROZEN.items():
        p = REPO / rel
        got = git_blob_id(p.read_bytes()) if p.is_file() else None
        out[rel] = got
        if got != blob:
            bad.append(rel)
    if bad:
        raise Refused(f"{len(bad)} frozen file(s) differ from their blob ids: {bad}")
    return out


def after_load_checks(loaded: LoadedTeacher, inputs: TeacherInputs, *, stub: bool) -> dict:
    from src.distill.nmf_stream import isolated_nmf_modules

    prov = loaded.frozen.provenance
    if prov is None or prov.ckpt_sha256 != inputs.sha256:
        raise Refused("the loaded teacher's provenance sha256 is not the verified sha256")
    n_params = sum(p.numel() for p in loaded.frozen.parameters())
    if stub:
        if "mmseg" in sys.modules:
            raise Refused("stub mode refuses a build that imported mmseg")
        if n_params > STUB_MAX_PARAMETERS:
            raise Refused(f"stub mode refuses a model with {n_params} parameters (> {STUB_MAX_PARAMETERS})")
    else:
        fields = dataclasses.fields(type(prov))
        if len(fields) != EXPECTED_PROVENANCE_FIELDS:
            raise Refused(f"TeacherProvenance has {len(fields)} fields, expected {EXPECTED_PROVENANCE_FIELDS}")
        empty = [f.name for f in fields if getattr(prov, f.name) in (None, "")]
        if empty:
            raise Refused(f"teacher provenance fields empty: {empty}")
    mods = isolated_nmf_modules(loaded.segmentor)
    if len(mods) != 1 or mods[0].nmf_stream is not loaded.stream:
        raise Stop("expected exactly one isolated NMF module holding the begun stream")
    if loaded.segmentor.training is not False:
        raise Stop("the teacher segmentor is in training mode")
    return {"loaded_state_sha256": loaded_state_sha256(loaded.segmentor),
            "frozen_blob_ids": frozen_blob_record(), "parameters": int(n_params)}


def teacher_record(loaded: LoadedTeacher, inputs: TeacherInputs, checks: dict) -> dict:
    """Every teacher output's identity block; the one provenance call site (P7, P8)."""
    provenance = loaded.frozen.provenance.as_dict()
    return {
        "role": inputs.role, "arm_id": inputs.arm_id, "arm_dl_id": inputs.arm_dl_id,
        "checkpoint": {"path": str(inputs.ckpt), "sha256_verified": inputs.sha256, "bytes": inputs.ckpt_bytes},
        "config": {"path": str(inputs.config), "sha256": inputs.config_sha256, "git_blob": inputs.config_blob},
        "loaded_state_sha256": checks["loaded_state_sha256"],
        "provenance": provenance,
        "provenance_fields": provenance_fields(),
        "frozen_blob_ids": checks["frozen_blob_ids"],
        "parameters": checks["parameters"],
        "load_mode": loaded.mode,
        "nmf_stream_begin": loaded.stream_description,
    }


def same_teacher(a: dict, b: dict) -> list[str]:
    """Fields on which two teacher blocks differ (P11): checkpoint sha, config sha, state, provenance."""
    pairs = (("checkpoint sha256", a.get("checkpoint", {}).get("sha256_verified"),
              b.get("checkpoint", {}).get("sha256_verified")),
             ("config sha256", a.get("config", {}).get("sha256"), b.get("config", {}).get("sha256")),
             ("loaded_state_sha256", a.get("loaded_state_sha256"), b.get("loaded_state_sha256")),
             ("provenance", a.get("provenance"), b.get("provenance")))
    return [name for name, x, y in pairs if x is None or x != y]


# --------------------------------------------------------------------------------------------------
# the split forward (P9)
# --------------------------------------------------------------------------------------------------
def feature_sha256(feats) -> str:
    h = hashlib.sha256()
    for t in feats:
        t = t.detach().cpu().contiguous()
        h.update(str(t.dtype).encode("ascii") + repr(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def rng_state_sha256() -> str:
    import torch
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


class SplitTeacher:
    """The adapter's forward split in two: the backbone once, the decode head as often as needed."""

    def __init__(self, loaded: LoadedTeacher):
        self.adapter = loaded.adapter
        self.segmentor = loaded.segmentor
        self.backbone_calls = 0
        self.head_calls = 0

    def features(self, x):
        import torch
        from src.distill.segnext_teacher import select_stride16_feature
        with torch.no_grad():
            if hasattr(self.segmentor, "extract_feat"):              # the adapter's own branch
                feats = self.segmentor.extract_feat(x)
            else:
                feats = self.segmentor.backbone(x)
            select_stride16_feature(feats, tuple(x.shape[-2:]), self.adapter.stage3_channels,
                                    self.adapter.stage3_stride)
        self.backbone_calls += 1
        return feats

    def attach(self, stream) -> None:
        from src.distill.nmf_stream import attach_nmf_stream
        attach_nmf_stream(self.segmentor, stream)

    def head(self, feats, stream, *, feat_hash: str, expect_shape: tuple | None = None):
        import torch
        from src.distill.nmf_stream import isolated_nmf_modules
        if stream is None:
            raise Stop("a head call needs its NMF stream object")
        mods = isolated_nmf_modules(self.segmentor)
        if len(mods) != 1:
            raise Stop(f"expected exactly one isolated NMF module, found {len(mods)}")
        if mods[0].nmf_stream is not stream:
            raise Stop("the attached NMF stream is not the expected stream object")
        before, rng = stream.draws, rng_state_sha256()
        with torch.no_grad():
            z = self.segmentor.decode_head.forward(feats)
        self.head_calls += 1
        if rng_state_sha256() != rng:
            raise Stop("a head call moved the caller's CPU RNG state")
        if stream.draws != before + 1:
            raise Stop(f"the head call advanced the stream by {stream.draws - before} draws, expected 1")
        if not torch.is_tensor(z) or z.dim() != 4 or z.shape[1] != self.adapter.num_classes:
            raise Stop(f"teacher logits have shape {tuple(getattr(z, 'shape', ()))}")
        if not bool(torch.isfinite(z).all()):
            raise Stop("non-finite teacher logits")
        if expect_shape is not None and tuple(z.shape) != tuple(expect_shape):
            raise Stop(f"teacher logits {tuple(z.shape)} != expected {tuple(expect_shape)}")
        if feature_sha256(feats) != feat_hash:
            raise Stop("the backbone features changed during a head call")
        return z


# --------------------------------------------------------------------------------------------------
# TRAIN on the evaluation canvas (P19)
# --------------------------------------------------------------------------------------------------
def train_canvas_dataset(source_indices=None):
    """The TRAIN pairs of PlantSegDataset("train"), each run through the VAL branch exactly."""
    from PIL import Image
    from torch.utils.data import Dataset

    from src.data.dataset import PlantSegDataset
    from src.data.transforms import core_preprocess, finalize

    class TrainCanvasDataset(Dataset):
        def __init__(self, indices):
            self.base = PlantSegDataset("train")             # unmodified: pairing, mask and count guards
            n = len(self.base)
            idx = list(range(n)) if indices is None else [int(i) for i in indices]
            if not idx or len(set(idx)) != len(idx) or not all(0 <= i < n for i in idx):
                raise Refused("TRAIN source indices must be unique and in range")
            self.source_indices = sorted(idx)

        def __len__(self):
            return len(self.source_indices)

        def stem_for(self, src):
            return self.base.pairs[src][0].stem

        def __getitem__(self, i):
            src = self.source_indices[i]
            img_path, mask_path = self.base.pairs[src]
            with Image.open(img_path) as im, Image.open(mask_path) as mk:
                img_np, mask_np = core_preprocess(im, mk)       # the VAL branch of __getitem__
            image, target = finalize(img_np, mask_np)
            stem = self.stem_for(src)
            return {"image": image, "target": target, "image_id": stem, "clean_image_id": stem,
                    "manifest_index": src}

    return TrainCanvasDataset(source_indices)


def check_val_reference(path, *, stub: bool, expected_rows: int) -> dict:
    """P18: a teacher-stage canvas VAL artifact of the record checkpoint (846 rows, mIoU within 1e-5 of
    R3). summary.json is read first, and a split other than "val" is refused before anything else is
    opened. Stub mode checks the same fields against the stub run (its own checkpoint, any mIoU)."""
    from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME, verify_artifact
    from src.eval.protocols import CANVAS_PROTOCOL_ID

    d = Path(path)
    summary = read_json_strict(d / "summary.json")
    ds, run = summary.get("dataset") or {}, summary.get("run") or {}
    if ds.get("split") != "val":
        raise Refused(f"--val-reference: split {ds.get('split')!r} is not 'val'")
    if (run.get("stage"), run.get("model_role"), run.get("precision")) != ("teacher", "teacher", "fp32"):
        raise Refused("--val-reference is not a teacher-stage fp32 artifact")
    if "protocol" in summary or ds.get("preprocess_protocol") != CANVAS_PROTOCOL_ID:
        raise Refused("--val-reference is not a canvas artifact")
    if ds.get("expected_rows") != expected_rows or ds.get("actual_rows") != expected_rows:
        raise Refused(f"--val-reference has {ds.get('actual_rows')} rows, expected {expected_rows}")
    miou = (summary.get("dataset_level") or {}).get("all_class_miou")
    if not isinstance(miou, float):
        raise Refused("--val-reference carries no all-class mIoU")
    if not stub:
        if run.get("checkpoint_sha256") != RECORD_SHA256:
            raise Refused("--val-reference is not an artifact of the teacher of record")
        if run.get("artifact_status") not in ("provisional", "official"):
            raise Refused("--val-reference must be a real-run artifact")
        if abs(miou - R3_VAL_MIOU) > R3_TOLERANCE:
            raise Refused(f"--val-reference mIoU {miou!r} is not within {R3_TOLERANCE} of R3 {R3_VAL_MIOU!r}")
    verify_artifact(d)
    files = {name: file_sha256(d / name) for name in sorted([*ARTIFACT_FILES, MANIFEST_NAME])}
    return {"dir": str(d), "run_id": run.get("run_id"), "artifact_status": run.get("artifact_status"),
            "checkpoint_sha256": run.get("checkpoint_sha256"), "all_class_miou": miou,
            "split_manifest_sha256": ds.get("split_manifest_sha256"), "rows": ds.get("actual_rows"),
            "files_sha256": files}


def check_m11(root) -> dict:
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    try:
        return assert_trainval_only_root(Path(root))
    except TrainValIsolationError as e:
        raise Refused(f"[{e.code}] {e}") from e


# --------------------------------------------------------------------------------------------------
# output blocks (P30)
# --------------------------------------------------------------------------------------------------
def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def environment_block(start_utc: str, end_utc: str | None = None) -> dict:
    import numpy
    import PIL
    import torch
    env = {"hostname": socket.gethostname(), "cpu_model": _cpu_model(),
           "torch_num_threads": int(torch.get_num_threads()), "python": sys.version.split()[0],
           "torch": torch.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
           "image_digest": (os.environ.get("PLANTSEG_IMAGE_DIGEST") or "").strip() or None,
           "start_utc": start_utc, "end_utc": end_utc}
    for mod in ("mmseg", "mmcv", "mmengine", "scipy"):
        m = sys.modules.get(mod)
        env[mod] = getattr(m, "__version__", None) if m is not None else None
    return env


def base_document(script: str, args, *, stub: bool, start_utc: str, code: dict, extra: dict | None = None) -> dict:
    doc = {"lane": LANE, "script": script, "authority": AUTHORITY,
           "artifact_status": "smoke" if stub else "provisional",
           "mode": "stub" if stub else "real",
           "parameters": jsonable({k: v for k, v in vars(args).items()}),
           "code": code, "start_utc": start_utc,
           "generated_utc": getattr(args, "generated_utc", None) or start_utc}
    if extra:
        doc.update(extra)
    return doc


def output_stamp(args, start_utc: str) -> str:
    return utc_stamp(getattr(args, "generated_utc", None) or start_utc)


__all__ = [
    "REPO", "LANE", "RECORD_SHA256", "REGISTERED_TEACHER_SHA256", "R3_VAL_MIOU", "R3_TOLERANCE",
    "TEACHER_CONFIG_REL", "RECORD_CONFIG_BLOB", "EXPECTED_PROVENANCE_FIELDS", "CODE_FILES", "ROLES",
    "ARM_IDS", "PURPOSES", "CORRECTION_STATES", "EXIT_OK", "EXIT_STOP", "EXIT_REFUSED", "EXIT_ERROR",
    "Refused", "Stop", "utc_now", "utc_stamp", "file_sha256", "name_digests", "refuse_test_path",
    "refuse_test_names", "require_outside_repo", "read_json_strict", "json_bytes", "write_json_exclusive",
    "git_head", "git_status", "code_provenance", "require_commit_binding", "check_common_flags",
    "check_teacher_flags", "check_correction_flags", "provenance_fields", "require_provenance_field_count",
    "require_single_output", "TeacherInputs", "verify_teacher_inputs", "LoadedTeacher", "load_teacher",
    "loaded_state_sha256", "frozen_blob_record", "after_load_checks", "teacher_record", "same_teacher",
    "feature_sha256", "rng_state_sha256", "SplitTeacher", "train_canvas_dataset", "check_val_reference",
    "check_m11",
    "environment_block", "base_document", "output_stamp",
]
