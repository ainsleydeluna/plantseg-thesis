#!/usr/bin/env python3
"""The corrupted-condition grid runner (L-CORR B-3): one scripts/evaluate_corruptions.py process per
cell, from the inventory export alone.

  run_corruption_grid.py --inventory EXPORT --split {val,test} [--confirm-test-split]
      --artifact-status {official,provisional,smoke}
      --corruption-cache ROOT --verification-record RECORD
      --expected-cache-sha256 C --expected-inferential-cells-sha256 I
      --expected-manifest-file-sha256 F --expected-verification-record-sha256 V
      --input-root DIR --out-root DIR [--teacher-config CFG] [--processes P] [--plan-only]
      [--max-samples N]          (a capped VAL run only; refused by the evaluator otherwise)

INPUTS (LE2; TL-6). The models, their sources, their cells and the thread count T come only from
the export (`tday-inventory/1`, TL interface section 8), never from free arguments. Each row with a
non-empty `cells` is scored, so a qat_fakequant_off row (no cells) never is. A row hands its
`source.path` (relative to --input-root) as --checkpoint, or as --provenance for INT8, with
`source.sha256` as --expected-checkpoint-sha256, `converted_artifact_sha256` as
--expected-converted-sha256 (INT8), `source.sha256` again as --teacher-ckpt-sha256 (the teacher),
and the export's T, batch size and device. For split test the cells must be the TL section 9 scopes:
all 24 for the teacher and for E1-E7 at seed 42, the 15 inferential for E1 and E6 at seeds 43/44
(both of a seed, or neither), none for any other row. A VAL rehearsal may score any cells.

START GATES, each a refusal by code before any process starts (exit 2), in this order: names (TL
section 2: no "test" in any case and none of :<>"\\|?* in any component of the out-root, checked as
strings before any filesystem call and again resolved; file ids fullmatch `[A-Za-z0-9._-]+` without
`__`); the inventory and, for split test, its scopes; --teacher-config an existing file when the
teacher is scored; every source inside the input root and an existing file; the export's evaluator
commit equals this checkout's HEAD (TL-9: S = E); an official grid's governed paths clean
(EVALUATION_CONTRACT section 7.1); the execution context (T, batch size, device; for an official
TEST grid the image named as sha256:<64 hex>; PLANTSEG_IMAGE_DIGEST equals the export's image; P x T
at most the physical cores of the CPUs this process may use, IF section 1); the four recorded values
of TL section 5 (LE3), each compared with the cache, its manifest file and the verification record;
the cache gates of the evaluator over every planned cell (split, coverage, completeness, status,
pin, checksum, protocol); the out-root (outside the repository); each cell's argv through the
evaluator's own argument checks; and the restart state of every cell. The cache, its record and the
teacher config are resolved once, so the gates and every cell name the same files. One runner holds
the out-root's `.grid_lock` while it runs (`out_root_locked`).

RESTART RULES (LE10, F-11). At start, a cell whose artifact and record verify, and whose record and
summary name this export row's identity, this commit and this artifact status, is skipped (R2); an
artifact without its record, a record without its artifact or a done cell that no longer verifies
is refused, by name (R3); every orphan temporary entry in the out-root is refused, by name (R5): the
operator moves it aside, never deletes it, and names it in a fault report in AM-8a's form. During
the run, a cell that wrote no artifact and left nothing behind is re-run once into the same out-dir,
and the re-run is logged (R6). Failures logged by earlier runs in the out-root count: a cell with
one gets one more attempt, a cell with two is refused at start. A second failure, an orphan left by
a failed cell, a finished cell that does not verify, a HEAD that moves during the session, or an
error of the runner itself is a STOP (exit 3): no further cell starts and the running ones are
awaited and logged.

ORDER (plan section 6(ix) step 6). Per model, the 15 inferential cells, then the 9 descriptive, in
protocol order. Models: E1 and E6 at seed 42, then E2, E3, E4, E5, E7, then E1 and E6 at seeds 43
and 44. With P >= 2 the teacher holds one process slot from the start; the other slots take
students, and any slot takes teacher cells once the students are done. No INT8 cell starts beside
another while scripts/evaluate_model.py lacks --threads (IF section 1's interim rule). Each cell is
one process (R4), and every start is logged in the order it happened (TL-1, one recorded order).

The run log `grid_run_<UTC>.jsonl` (one header line, then one line per event, each fsynced) and the
cell logs `grid_logs_<UTC>/<out-dir name>.<attempt>.log` are written in the out-root.
Exit 0: every cell done. Exit 2: refused, nothing started. Exit 3: STOP.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import corruption_roles  # noqa: E402
from src.corruption_cache import MANIFEST_NAME, _strict_json, load_manifest  # noqa: E402
from src.corruption_cache import CorruptionCacheError  # noqa: E402

INVENTORY_SCHEMA = "tday-inventory/1"
RUN_LOG_SCHEMA = "plantseg-corruption-grid-run/1.0.0"
EVALUATOR = REPO / "scripts" / "evaluate_corruptions.py"
STUDENT_ORDER = (("E1", 42), ("E6", 42), ("E2", 42), ("E3", 42), ("E4", 42), ("E5", 42),
                 ("E7", 42), ("E1", 43), ("E6", 43), ("E1", 44), ("E6", 44))
STAGE_PRECISION = {"teacher": "fp32", "E1": "fp32", "E2": "fp32", "E3": "fp32",
                   "E4": "int8_ptq", "E5": "int8_qat", "E6": "int8_qat", "E7": "int8_ptq"}
INT8 = ("int8_ptq", "int8_qat")
PRECISIONS = ("fp32", "int8_ptq", "int8_qat", "qat_fakequant_off")
CONTEXT_KEYS = ("model_device", "input_devices", "determinism_policy_applied", "batch_size",
                "num_workers", "torch_num_threads", "image_digest")
#: Applied with fullmatch: a trailing newline never passes.
FILE_ID = re.compile(r"[A-Za-z0-9._-]+")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: IF section 1: P x T is bounded by the physical cores of the CPUs this process may use.
SYSFS_CPU = Path("/sys/devices/system/cpu")
PROC_CPUINFO = Path("/proc/cpuinfo")
#: The checkout whose governed paths an official grid requires clean (EVALUATION_CONTRACT 7.1).
GOVERNED_REPO = REPO
#: IF section 1's interim rule: until both evaluators take --threads (scripts/evaluate_model.py
#: does not yet), INT8 processes run one at a time.
INT8_ONE_AT_A_TIME = True
LOCK_NAME = ".grid_lock"
EXIT_DONE, EXIT_REFUSED, EXIT_STOP = 0, 2, 3


class GridRefusal(RuntimeError):
    """A start gate refused the grid: nothing was started."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


class GridStop(RuntimeError):
    """A STOP during the run: no further cell starts."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _refuse(code: str, message: str):
    raise GridRefusal(code, message)


def _stop(code: str, message: str) -> GridStop:
    return GridStop(code, message)


def utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score the corrupted cells of an inventory export.")
    p.add_argument("--inventory", required=True, help="the tday-inventory/1 export")
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--confirm-test-split", action="store_true")
    p.add_argument("--artifact-status", required=True, choices=["official", "provisional", "smoke"])
    p.add_argument("--corruption-cache", required=True)
    p.add_argument("--verification-record", required=True)
    p.add_argument("--expected-cache-sha256", required=True)
    p.add_argument("--expected-inferential-cells-sha256", required=True)
    p.add_argument("--expected-manifest-file-sha256", required=True)
    p.add_argument("--expected-verification-record-sha256", required=True)
    p.add_argument("--input-root", required=True,
                   help="the directory the export's source paths are relative to")
    p.add_argument("--out-root", required=True,
                   help="where the out-dirs, records and logs are written; outside the repository")
    p.add_argument("--teacher-config", default=None, help="the thesis teacher config (teacher rows)")
    p.add_argument("--processes", type=int, default=1, help="P processes at once (P x T <= cores)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="a capped VAL run, passed to every cell (the evaluator refuses it on test "
                        "and for an official artifact)")
    p.add_argument("--plan-only", action="store_true",
                   help="run every start gate, print the ordered cells, start nothing")
    return p


# --------------------------------------------------------------------------------------------------
# names (TL section 2)
# --------------------------------------------------------------------------------------------------
def name_problem(text: str) -> bool:
    return "test" in text.lower()


def check_out_root_name(out_root) -> str:
    """TL section 2 over the whole out-root, as given and as resolved (every out-dir below it
    adds only a checked file id and a cell id)."""
    from src.eval.corrupted import created_path_problem
    problem = created_path_problem(out_root)
    if problem:
        _refuse("out_name_invalid", f"the out-root {problem}; TL section 2 forbids 'test' in any "
                                    "case and :<>\"\\|?* in any component of a created path")
    return os.path.abspath(str(out_root))


def out_dir_name(split: str, file_id: str, cell_id: str) -> str:
    """`tday_corr_<file_id>_<cell id>` (split test) or `val_corr_...` (src/eval/corrupted.py)."""
    from src.eval.corrupted import corrupted_dir_name
    return corrupted_dir_name(split, file_id, cell_id)


# --------------------------------------------------------------------------------------------------
# the inventory (TL section 8)
# --------------------------------------------------------------------------------------------------
def _row_problem(row) -> str | None:
    if not isinstance(row, dict):
        return "a row is not an object"
    fid = row.get("file_id")
    if not isinstance(fid, str) or not FILE_ID.fullmatch(fid) or "__" in fid or name_problem(fid):
        return f"file_id {fid!r} breaks the TL section 2 rule"
    stage, prec = row.get("stage"), row.get("precision")
    if prec not in PRECISIONS:
        return f"{fid}: precision {prec!r}"
    src = row.get("source")
    if not isinstance(src, dict) or src.get("kind") not in ("checkpoint", "provenance") \
            or not isinstance(src.get("path"), str) \
            or not HEX64.fullmatch(str(src.get("sha256"))):
        return f"{fid}: source {src!r}"
    cells = row.get("cells")
    if not isinstance(cells, list) or not all(isinstance(c, str) for c in cells):
        return f"{fid}: cells {cells!r}"
    if not cells:
        return None
    if prec == "qat_fakequant_off":
        return f"{fid}: a qat_fakequant_off row carries cells (TL-12: clean only)"
    if stage not in STAGE_PRECISION or STAGE_PRECISION[stage] != prec:
        return f"{fid}: stage {stage!r} at precision {prec!r}"
    if (row.get("model_role") == "teacher") != (stage == "teacher"):
        return f"{fid}: model_role {row.get('model_role')!r} for stage {stage!r}"
    if src["kind"] != ("provenance" if prec in INT8 else "checkpoint"):
        return f"{fid}: a {prec} row hands a {src['kind']}"
    conv = row.get("converted_artifact_sha256")
    if (prec in INT8) != (conv is not None) or \
            (conv is not None and not HEX64.fullmatch(str(conv))):
        return f"{fid}: converted_artifact_sha256 {conv!r} for {prec}"
    try:
        parsed = [corruption_roles.parse_cell_id(c) for c in cells]
    except corruption_roles.CorruptionRolesError as e:
        return f"{fid}: {e}"
    if len(set(parsed)) != len(parsed):
        return f"{fid}: a cell is listed twice"
    return None


def load_inventory(path) -> tuple[dict, str]:
    """The export, read as strict JSON, and its sha256. Refuses a malformed export by name."""
    p = Path(path)
    try:
        raw = p.read_bytes()
        doc = _strict_json(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        _refuse("inventory_invalid", f"{p}: {type(e).__name__}: {e}")
    if not isinstance(doc, dict) or doc.get("schema") != INVENTORY_SCHEMA:
        _refuse("inventory_invalid", f"{p} is not a {INVENTORY_SCHEMA} export")
    ctx = doc.get("execution_context")
    if not HEX40.fullmatch(str(doc.get("evaluator_commit"))) or not isinstance(ctx, dict) \
            or not all(k in ctx for k in CONTEXT_KEYS) or not isinstance(doc.get("models"), list) \
            or not HEX64.fullmatch(str(doc.get("manifest_sha256"))):
        _refuse("inventory_invalid", f"{p} lacks a valid evaluator_commit, execution_context, "
                                     "manifest_sha256 or models")
    seen, scored = set(), set()
    for row in doc["models"]:
        problem = _row_problem(row)
        if problem is not None:
            _refuse("inventory_invalid", problem)
        key = (row["file_id"], row["precision"])
        if key in seen:
            _refuse("inventory_invalid", f"two rows for {key}")
        seen.add(key)
        if row["cells"]:
            if row["file_id"] in scored:
                _refuse("inventory_invalid", f"two scored rows for file_id {row['file_id']!r}")
            scored.add(row["file_id"])
    return doc, hashlib.sha256(raw).hexdigest()


def check_scopes(doc: dict) -> None:
    """Split test: the TL section 9 scopes. The teacher and E1-E7 at seed 42: all 24 cells, once
    each; E1 and E6 at seeds 43 and 44: the 15 inferential, both of a seed or neither; none
    otherwise."""
    full, inferential = list(corruption_roles.CELL_IDS), list(corruption_roles.INFERENTIAL_CELL_IDS)
    holders: dict = {}
    for row in doc["models"]:
        if not row["cells"]:
            continue
        key = (row["stage"], None if row["stage"] == "teacher" else row.get("seed"))
        cells = sorted(row["cells"], key=full.index)
        if key[0] == "teacher" or key[1] == 42:
            ok = cells == full
        elif key[0] in ("E1", "E6") and key[1] in (43, 44):
            ok = cells == inferential
        else:
            ok = False
        if not ok or key in holders:
            _refuse("inventory_scope_invalid",
                    f"{row['file_id']}: stage {row['stage']} seed {row.get('seed')!r} carries "
                    f"{len(row['cells'])} cells, outside the TL section 9 scopes")
        holders[key] = row["file_id"]
    for seed in (43, 44):
        if (("E1", seed) in holders) != (("E6", seed) in holders):
            _refuse("inventory_scope_invalid",
                    f"seed {seed}: E1 and E6 carry corrupted cells as a pair, or neither does")


# --------------------------------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------------------------------
def _cell_rank(cell_id: str) -> tuple:
    corruption, severity = corruption_roles.parse_cell_id(cell_id)
    role = corruption_roles.ROLE_INFERENTIAL if severity in corruption_roles.INFERENTIAL_SEVERITIES \
        else corruption_roles.ROLE_DESCRIPTIVE_ONLY
    return (0 if role == corruption_roles.ROLE_INFERENTIAL else 1,
            corruption_roles.CELL_IDS.index(cell_id)), role, corruption, severity


def _model_rank(row) -> tuple:
    key = (row["stage"], row.get("seed"))
    if key in STUDENT_ORDER:
        return (0, STUDENT_ORDER.index(key), "")
    return (1, row.get("seed") if isinstance(row.get("seed"), int) else -1, row["stage"],
            row["file_id"])


def plan(doc: dict, split: str) -> tuple[list, list]:
    """(teacher jobs, student jobs), each in scoring order."""
    teacher, students = [], []
    for row in sorted((r for r in doc["models"] if r["cells"]), key=_model_rank):
        for cell_id in sorted(row["cells"], key=lambda c: _cell_rank(c)[0]):
            _, role, corruption, severity = _cell_rank(cell_id)
            job = {"file_id": row["file_id"], "stage": row["stage"], "seed": row.get("seed"),
                   "precision": row["precision"], "cell": cell_id, "corruption": corruption,
                   "severity": severity, "role": role,
                   "out_name": out_dir_name(split, row["file_id"], cell_id), "row": row}
            (teacher if row["stage"] == "teacher" else students).append(job)
    return teacher, students


# --------------------------------------------------------------------------------------------------
# start gates
# --------------------------------------------------------------------------------------------------
def git_head() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                           text=True, timeout=60, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    head = p.stdout.strip()
    return head if p.returncode == 0 and HEX40.fullmatch(head) else None


def check_commit(doc: dict) -> str:
    head = git_head()
    if head != doc["evaluator_commit"]:
        _refuse("evaluator_commit_mismatch",
                f"this checkout is at {head}, the export's evaluator_commit is "
                f"{doc['evaluator_commit']} (TL-9: the session runs at S = E)")
    return head


def host_cpus() -> set:
    """The CPUs this process may run on (its affinity), else every CPU."""
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return set(range(os.cpu_count() or 1))


def physical_cores(cpus, sysfs=None, cpuinfo=None) -> int | None:
    """The distinct (package, core) pairs among `cpus`: from the sysfs topology, else from
    /proc/cpuinfo; None when neither names every CPU's core."""
    sysfs = Path(sysfs or SYSFS_CPU)
    pairs = set()
    for cpu in sorted(cpus):
        topo = sysfs / f"cpu{cpu}" / "topology"
        try:
            pairs.add((topo.joinpath("physical_package_id").read_text().strip(),
                       topo.joinpath("core_id").read_text().strip()))
        except OSError:
            pairs = None
            break
    if pairs:
        return len(pairs)
    try:
        text = Path(cpuinfo or PROC_CPUINFO).read_text()
    except OSError:
        return None
    found, seen = set(), set()
    for block in text.split("\n\n"):
        fields = {k.strip(): v.strip() for k, sep, v in
                  (line.partition(":") for line in block.splitlines()) if sep}
        if "processor" not in fields or int(fields["processor"]) not in cpus:
            continue
        if "physical id" not in fields or "core id" not in fields:
            return None
        seen.add(int(fields["processor"]))
        found.add((fields["physical id"], fields["core id"]))
    return len(found) if found and seen == set(cpus) else None


def check_context(doc: dict, args) -> dict:
    ctx, processes = doc["execution_context"], args.processes
    t, batch, device = ctx["torch_num_threads"], ctx["batch_size"], ctx["model_device"]
    for label, v in (("torch_num_threads", t), ("batch_size", batch)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            _refuse("execution_context_invalid", f"execution_context.{label} is {v!r}")
    if not isinstance(device, str) or not device:
        _refuse("execution_context_invalid", f"execution_context.model_device is {device!r}")
    official_test = args.split == "test" and args.artifact_status == "official"
    if official_test and not IMAGE_DIGEST.fullmatch(str(ctx["image_digest"])):
        _refuse("image_digest_invalid",
                f"execution_context.image_digest is {ctx['image_digest']!r}; an official TEST grid "
                "names its image as sha256:<64 hex> (IF section 1)")
    got = os.environ.get("PLANTSEG_IMAGE_DIGEST")
    if (got or None) != ctx["image_digest"]:
        _refuse("image_digest_mismatch",
                f"PLANTSEG_IMAGE_DIGEST is {got!r}, the export's image is {ctx['image_digest']!r}")
    cpus = host_cpus()
    cores = physical_cores(cpus)
    if cores is None:
        _refuse("physical_cores_unknown",
                f"the physical cores of this host's {len(cpus)} usable CPUs cannot be read (sysfs "
                "topology or /proc/cpuinfo); IF section 1 bounds P x T by them")
    if processes < 1 or processes * t > (cores or 0):
        _refuse("processes_threads_exceed_cpus",
                f"--processes {processes} x T {t} exceeds the {cores} physical cores of this "
                f"host's {len(cpus)} usable CPUs (IF section 1)")
    return {"threads": t, "batch_size": batch, "device": device, "cpus": len(cpus),
            "physical_cores": cores}


def check_recorded_values(args) -> tuple[dict, dict]:
    """LE3, TL section 5: the four recorded values, each compared with the cache, its manifest file
    and the verification record, before anything is scored. Returns (values, manifest)."""
    values = {"canonical_manifest_sha256": args.expected_cache_sha256,
              "inferential_cells_sha256": args.expected_inferential_cells_sha256,
              "manifest_file_sha256": args.expected_manifest_file_sha256,
              "verification_record_sha256": args.expected_verification_record_sha256}
    bad = [k for k, v in values.items() if not HEX64.fullmatch(str(v))]
    if bad:
        _refuse("recorded_value_format", f"{bad} must be 64 lowercase hex characters")
    root = Path(args.corruption_cache)
    try:
        manifest = load_manifest(root)
        raw = Path(args.verification_record).read_bytes()
        record = _strict_json(raw.decode("utf-8"))
    except (CorruptionCacheError, OSError, ValueError, UnicodeDecodeError) as e:
        _refuse("recorded_values_mismatch", f"the cache or its record cannot be read: {e}")
    held = {"canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
            "inferential_cells_sha256": manifest["inferential_cells_sha256"],
            "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
            "verification_record_sha256": hashlib.sha256(raw).hexdigest()}
    bad = [k for k in values if held[k] != values[k]]
    bad += [f"record.{k}" for k in ("canonical_manifest_sha256", "inferential_cells_sha256",
                                    "manifest_file_sha256")
            if not isinstance(record, dict) or record.get(k) != values[k]]
    if bad:
        _refuse("recorded_values_mismatch", f"the recorded values disagree with the cache: {bad}")
    return values, manifest


def check_cache_scope(manifest: dict, jobs: list, args) -> None:
    """The evaluator's own cache gates (split, every planned cell, completeness, status, pin,
    checksum, protocol), once at start over every planned cell, so a cache that cannot serve the
    grid is a start refusal and not a STOP after two launches."""
    import scripts.evaluate_corruptions as EC
    cells = sorted({(j["corruption"], j["severity"]) for j in jobs})
    try:
        EC.check_cache_request(manifest, cells, split=args.split, status=args.artifact_status,
                               expected_sha256=args.expected_cache_sha256)
    except CorruptionCacheError as e:
        _refuse("cache_scope_mismatch", f"the cache cannot serve this grid: {EC.refusal_line(e)}")


def check_governed(args) -> None:
    """EVALUATION_CONTRACT section 7.1: an official artifact needs the governed paths clean. Checked
    at start, so a dirty checkout is refused before any process parses a checkpoint."""
    if args.artifact_status != "official":
        return
    from src.eval.artifacts import GOVERNED_PREFIXES, git_porcelain_bytes, parse_porcelain_paths
    dirty = sorted(p for p in parse_porcelain_paths(git_porcelain_bytes(GOVERNED_REPO))
                   if any(p.startswith(g) for g in GOVERNED_PREFIXES))
    if dirty:
        _refuse("governed_paths_dirty", f"an official grid needs clean governed paths; dirty or "
                                        f"untracked: {dirty[:5]}")


def resolve_inputs(args) -> None:
    """The cache, its verification record and the teacher config, resolved once against the
    runner's working directory: the start gates and every cell then name the same files."""
    args.corruption_cache = str(Path(args.corruption_cache).resolve())
    args.verification_record = str(Path(args.verification_record).resolve())
    if args.teacher_config:
        args.teacher_config = str(Path(args.teacher_config).resolve())


def check_out_root(raw: str) -> Path:
    out_root = Path(raw).resolve()
    if out_root == REPO or REPO in out_root.parents:
        _refuse("out_root_inside_repo", "the out-root is never inside the repository")
    return out_root


def check_cell_argvs(jobs: list, args, context: dict, input_root: Path, out_root: Path) -> None:
    """Each cell's argv through the evaluator's own parser and argument checks (pure: nothing is
    read), so a refusal every process would give is a start refusal."""
    import scripts.evaluate_corruptions as EC
    parser = EC.build_parser()
    for job in jobs:
        argv = cell_argv(job, args, context, input_root, out_root)[3:]
        try:
            EC.validate_args(parser.parse_args(argv))
        except Exception as e:                               # noqa: BLE001 -- refused by name
            _refuse("cell_argv_refused", f"{job['out_name']}: the evaluator refuses its argv: "
                                         f"{EC.refusal_line(e)}")


def is_orphan(name: str) -> bool:
    """A temporary sibling left by an interrupted writer: the artifact writer's
    `.<out-dir>.tmp-<uuid>` or the record writer's `.<out-dir>.corruption.json.tmp-<pid>`."""
    return name.startswith(".") and ".tmp-" in name


def start_orphans(out_root: Path, jobs) -> list:
    """At start every orphan in the out-root counts, not only the planned cells': a cell dropped
    from a corrected export leaves its orphans too, and each is named for the fault report."""
    return orphans(out_root)


def orphans(out_root: Path, names=None) -> list:
    """Every orphan in the out-root, or only those of the named out-dirs."""
    if not out_root.is_dir():
        return []
    mine = tuple(f".{n}." for n in names) if names is not None else (".",)
    return sorted(p.name for p in out_root.iterdir()
                  if is_orphan(p.name) and p.name.startswith(mine))


def expected_identity(row: dict) -> str:
    """LE2: the scored file's identity; for INT8 the converted artifact (evaluate_model.py:271)."""
    return row["converted_artifact_sha256"] if row["precision"] in INT8 else row["source"]["sha256"]


def done_problems(job: dict, summary: dict, record: dict, values: dict, expect: dict) -> list:
    """What a verifying cell must also name to count as done: its stage, condition, cache and
    verification record, the export row's identity, this checkout's commit and the requested
    artifact status."""
    want = {"stage": job["stage"],
            "condition": {"type": "corruption", "name": job["corruption"],
                          "severity": job["severity"]},
            "canonical_manifest_sha256": values["canonical_manifest_sha256"],
            "verification_record_sha256": values["verification_record_sha256"],
            "checkpoint_sha256": expected_identity(job["row"]),
            "evaluator_commit": expect["head"], "artifact_status": expect["status"]}
    got = {"stage": record["stage"], "condition": record["condition"],
           "canonical_manifest_sha256": record["cache"]["canonical_manifest_sha256"],
           "verification_record_sha256": record["cache"]["verification_record_sha256"],
           "checkpoint_sha256": summary["run"].get("checkpoint_sha256"),
           "evaluator_commit": record["evaluator_commit"],
           "artifact_status": summary["run"].get("artifact_status")}
    return [f"{k}: {got[k]!r}, not {want[k]!r}" for k in want if got[k] != want[k]]


def cell_state(out_root: Path, job: dict, values: dict, expect: dict) -> str:
    """'todo' or 'done'; R3 refusals for every other state."""
    from src.eval.artifacts import verify_artifact
    from src.eval.corrupted import read_cell_record, record_path, verify_cell_record
    out_dir = out_root / job["out_name"]
    rec = record_path(out_dir)
    if not out_dir.exists() and not rec.exists():
        return "todo"
    if out_dir.exists() and not rec.exists():
        _refuse("artifact_without_record", f"{out_dir} has no cell record {rec.name}")
    if rec.exists() and not out_dir.exists():
        _refuse("record_without_artifact", f"{rec} names an artifact directory that is absent")
    try:
        summary = verify_artifact(out_dir)
        record = read_cell_record(rec)
        verify_cell_record(out_dir, record)
    except Exception as e:                                   # noqa: BLE001 -- refused by name
        _refuse("cell_does_not_verify", f"{out_dir.name}: {type(e).__name__}: {e}")
    problems = done_problems(job, summary, record, values, expect)
    if problems:
        _refuse("cell_does_not_verify", f"{out_dir.name}: {problems}")
    return "done"


def prior_failures(out_root: Path) -> collections.Counter:
    """Failed attempts that earlier runs logged in this out-root (LE10 across restarts): each
    'rerun', each 'failed' after a STOP and each cell_failed_twice STOP counts one."""
    counts: collections.Counter = collections.Counter()
    if not out_root.is_dir():
        return counts
    for p in sorted(out_root.iterdir()):
        if not (p.name.startswith("grid_run_") and p.name.endswith(".jsonl")):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("kind")
            if kind in ("rerun", "failed") or (kind == "stop" and
                                                event.get("code") == "cell_failed_twice"):
                counts[event.get("out_dir")] += 1
    return counts


# --------------------------------------------------------------------------------------------------
# a cell
# --------------------------------------------------------------------------------------------------
def cell_argv(job: dict, args, context: dict, input_root: Path, out_root: Path) -> list[str]:
    row = job["row"]
    source = (input_root / row["source"]["path"]).resolve()
    argv = [sys.executable, "-B", str(EVALUATOR), "--stage", job["stage"],
            "--precision", job["precision"], "--corruption", job["corruption"],
            "--severity", str(job["severity"]), "--out-dir", str(out_root / job["out_name"]),
            "--run-id", job["out_name"], "--corruption-cache", str(args.corruption_cache),
            "--expected-cache-sha256", args.expected_cache_sha256,
            "--verification-record", str(args.verification_record),
            "--expected-checkpoint-sha256", row["source"]["sha256"],
            "--threads", str(context["threads"]), "--batch-size", str(context["batch_size"]),
            "--device", context["device"], "--split", args.split,
            "--artifact-status", args.artifact_status]
    if args.split == "test":
        argv.append("--confirm-test-split")
    if args.max_samples is not None:
        argv += ["--max-samples", str(args.max_samples)]
    if job["precision"] in INT8:
        argv += ["--provenance", str(source),
                 "--expected-converted-sha256", row["converted_artifact_sha256"]]
    else:
        argv += ["--checkpoint", str(source)]
    if job["stage"] == "teacher":
        argv += ["--teacher-config", str(args.teacher_config),
                 "--teacher-ckpt-sha256", row["source"]["sha256"]]
    return argv


def source_path_problem(row, input_root: Path) -> str | None:
    p = (input_root / row["source"]["path"]).resolve()
    if input_root != p and input_root not in p.parents:
        return f"{row['file_id']}: source path {row['source']['path']!r} leaves the input root"
    return None


def pick_job(teacher, students, running: dict, processes: int):
    """The next cell to start, or None. With P >= 2 the teacher holds one slot while it has cells;
    the students go in order and any slot takes teacher cells once they are done. While
    INT8_ONE_AT_A_TIME holds, no INT8 cell starts beside another (IF section 1's interim rule)."""
    busy = list(running.values())
    int8_busy = INT8_ONE_AT_A_TIME and any(j["precision"] in INT8 for j in busy)

    def take(queue):
        for i, job in enumerate(queue):
            if not (int8_busy and job["precision"] in INT8):
                del queue[i]
                return job
        return None
    if processes >= 2 and not any(j["stage"] == "teacher" for j in busy):
        job = take(teacher)
        if job is not None:
            return job
    return take(students) or take(teacher)


def acquire_lock(out_root: Path):
    """One runner per out-root: an exclusive lock on `<out-root>/.grid_lock`, held for the whole run
    and released when the runner exits. The lock file is never removed."""
    fd = os.open(out_root / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        _refuse("out_root_locked", f"another runner holds {out_root / LOCK_NAME}; one runner per "
                                   "out-root")
        return None
    return fd


class RunLog:
    """Append-only JSON lines, each flushed and fsynced."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, kind: str, **data) -> None:
        line = json.dumps({"kind": kind, "utc": utc_iso(), **data}, sort_keys=True,
                          ensure_ascii=False, allow_nan=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def default_launcher(argv: list[str], log_path: Path):
    log = open(log_path, "wb")
    try:
        return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
    finally:
        log.close()


# --------------------------------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------------------------------
def prepare(args) -> dict:
    """Every start gate, in order. Returns what the run needs; raises GridRefusal."""
    if args.split == "test" and not args.confirm_test_split:
        _refuse("test_split_unconfirmed", "--split test requires --confirm-test-split")
    out_raw = check_out_root_name(args.out_root)
    doc, inventory_sha = load_inventory(args.inventory)
    if args.split == "test":
        check_scopes(doc)
    teacher, students = plan(doc, args.split)
    jobs = teacher + students
    if not jobs:
        _refuse("inventory_invalid", "no row of the export carries a cell")
    resolve_inputs(args)
    if teacher and not (args.teacher_config and Path(args.teacher_config).is_file()):
        _refuse("teacher_config_required", f"the export scores the teacher: pass --teacher-config, "
                                           f"an existing file (got {args.teacher_config!r})")
    input_root = Path(args.input_root).resolve()
    for job in jobs:
        problem = source_path_problem(job["row"], input_root)
        if problem:
            _refuse("inventory_invalid", problem)
    for job in jobs:
        source = (input_root / job["row"]["source"]["path"]).resolve()
        if not source.is_file():
            _refuse("source_missing", f"{job['file_id']}: {source} is not a file")
    head = check_commit(doc)
    check_governed(args)
    context = check_context(doc, args)
    values, manifest = check_recorded_values(args)
    check_cache_scope(manifest, jobs, args)
    out_root = check_out_root(out_raw)
    check_cell_argvs(jobs, args, context, input_root, out_root)
    left = start_orphans(out_root, jobs)
    if left:
        _refuse("orphan_temporary_directory",
                f"orphan temporary entries in {out_root}: {left}; move each aside (never delete "
                "it) and name it in a fault report in AM-8a's form")
    expect = {"head": head, "status": args.artifact_status}
    states = {j["out_name"]: cell_state(out_root, j, values, expect) for j in jobs}
    failed = prior_failures(out_root)
    twice = sorted(j["out_name"] for j in jobs
                   if states[j["out_name"]] == "todo" and failed[j["out_name"]] >= 2)
    if twice:
        _refuse("cell_failed_twice", f"earlier runs in this out-root logged two failures for "
                                     f"{twice}; LE10 makes a second failure a STOP")
    return {"doc": doc, "inventory_sha256": inventory_sha, "teacher": teacher,
            "students": students, "head": head, "context": context, "values": values,
            "out_root": out_root, "input_root": input_root, "states": states, "expect": expect,
            "prior_failures": failed}


def finished_ok(job, out_root: Path, values: dict, expect: dict) -> bool:
    try:
        return cell_state(out_root, job, values, expect) == "done"
    except GridRefusal:
        return False


def run_grid(args, *, launcher=None, poll_seconds: float = 0.5) -> int:
    launcher = launcher or default_launcher
    try:
        ready = prepare(args)
    except GridRefusal as e:
        print(f"GRID REFUSED {e}", file=sys.stderr)
        return EXIT_REFUSED
    teacher = collections.deque(j for j in ready["teacher"]
                                if ready["states"][j["out_name"]] == "todo")
    students = collections.deque(j for j in ready["students"]
                                 if ready["states"][j["out_name"]] == "todo")
    order = [{"lane": "teacher" if j["stage"] == "teacher" else "students", "file_id": j["file_id"],
              "stage": j["stage"], "seed": j["seed"], "cell": j["cell"], "role": j["role"],
              "out_dir": j["out_name"], "state": ready["states"][j["out_name"]]}
             for j in ready["teacher"] + ready["students"]]
    if args.plan_only:
        for i, item in enumerate(order):
            print(json.dumps({"order": i, **item}, sort_keys=True))
        return EXIT_DONE
    out_root, values, context = ready["out_root"], ready["values"], ready["context"]
    expect = ready["expect"]
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        lock = acquire_lock(out_root)
    except GridRefusal as e:
        print(f"GRID REFUSED {e}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        return _run_cells(args, ready, order, teacher, students, launcher, poll_seconds,
                          out_root, values, context, expect)
    finally:
        if lock is not None:
            os.close(lock)


def _run_cells(args, ready, order, teacher, students, launcher, poll_seconds, out_root, values,
               context, expect) -> int:
    from src.eval.corrupted import record_path
    stamp = utc_stamp()
    log = RunLog(out_root / f"grid_run_{stamp}.jsonl")
    cell_logs = out_root / f"grid_logs_{stamp}"
    cell_logs.mkdir()
    log.write("header", schema=RUN_LOG_SCHEMA, argv=sys.argv, inventory=str(args.inventory),
              inventory_sha256=ready["inventory_sha256"], evaluator_commit=ready["head"],
              manifest_sha256=ready["doc"]["manifest_sha256"], split=args.split,
              artifact_status=args.artifact_status, recorded_values=values,
              processes=args.processes, threads=context["threads"], cpus=context["cpus"],
              physical_cores=context["physical_cores"], int8_one_at_a_time=INT8_ONE_AT_A_TIME,
              prior_failures=dict(ready["prior_failures"]), plan=order)
    for item in order:
        if item["state"] == "done":
            log.write("skip", out_dir=item["out_dir"], reason="artifact and record verify (R2)")
    running: dict = {}
    attempts = collections.Counter(ready["prior_failures"])
    stop = None
    try:
        while stop is None and (teacher or students or running):
            while stop is None and len(running) < args.processes:
                job = pick_job(teacher, students, running, args.processes)
                if job is None:
                    break
                head = git_head()
                if head != ready["head"]:
                    (teacher if job["stage"] == "teacher" else students).appendleft(job)
                    stop = _stop("head_changed", f"this checkout moved from {ready['head']} to "
                                                 f"{head} during the session; no cell starts")
                    break
                attempts[job["out_name"]] += 1
                n = attempts[job["out_name"]]
                argv = cell_argv(job, args, context, ready["input_root"], out_root)
                proc = launcher(argv, cell_logs / f"{job['out_name']}.{n}.log")
                running[proc] = job
                log.write("start", out_dir=job["out_name"], attempt=n,
                          pid=getattr(proc, "pid", None), stage=job["stage"], cell=job["cell"])
            if stop is not None:
                break
            finished = [p for p in running if p.poll() is not None]
            if not finished:
                time.sleep(poll_seconds)
                continue
            for proc in finished:
                job = running.pop(proc)
                rc, name = proc.returncode, job["out_name"]
                if rc == 0 and finished_ok(job, out_root, values, expect):
                    log.write("done", out_dir=name, attempt=attempts[name])
                    continue
                out_dir = out_root / name
                left = orphans(out_root, [name])
                if out_dir.exists() or record_path(out_dir).exists() or left:
                    stop = _stop("cell_left_partial_output",
                                 f"{name} (rc {rc}) left "
                                 f"{['artifact'] if out_dir.exists() else []}"
                                 f"{['record'] if record_path(out_dir).exists() else []}{left}: "
                                 "move each aside, never delete it, and name it in a fault report")
                elif attempts[name] >= 2:
                    stop = _stop("cell_failed_twice",
                                 f"{name} wrote no artifact twice (rc {rc}); see its cell logs")
                else:
                    log.write("rerun", out_dir=name, rc=rc,
                              reason="the cell wrote no artifact; re-run once into the same "
                                     "out-dir")
                    (teacher if job["stage"] == "teacher" else students).appendleft(job)
                if stop is not None:
                    stop.out_dir = name
                    break
    except Exception as e:                                   # noqa: BLE001 -- a STOP, recorded
        stop = _stop("runner_error", f"{type(e).__name__}: {e}; no further cell starts and the "
                                     "running cells are awaited")
    if stop is not None:
        _best_effort(log.write, "stop", code=stop.code, message=str(stop),
                     out_dir=getattr(stop, "out_dir", None))
        print(f"GRID STOP {stop}", file=sys.stderr)
        for proc, job in list(running.items()):
            proc.wait()
            ok = proc.returncode == 0 and finished_ok(job, out_root, values, expect)
            _best_effort(log.write, "done" if ok else "failed", out_dir=job["out_name"],
                         rc=proc.returncode, after_stop=True)
        return EXIT_STOP
    log.write("end", cells=len(order))
    print(f"grid done: {len(order)} cells ({sum(1 for i in order if i['state'] == 'done')} skipped "
          f"as done); log {log.path}")
    return EXIT_DONE


def _best_effort(write, *a, **kw) -> None:
    """A log write on the STOP path: an error there is reported, never raised over the STOP."""
    try:
        write(*a, **kw)
    except Exception as e:                                   # noqa: BLE001 -- reported
        print(f"run-log write failed on the STOP path: {type(e).__name__}: {e}", file=sys.stderr)


def main(argv=None, *, launcher=None) -> int:
    args = build_parser().parse_args(argv)
    return run_grid(args, launcher=launcher)


if __name__ == "__main__":
    raise SystemExit(main())
