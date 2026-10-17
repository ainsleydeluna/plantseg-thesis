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

START GATES, each a refusal by code before any process starts (exit 2): names (TL section 2: the
out-root and every out-dir name hold no "test" in any case and no forbidden character, checked as
strings before any filesystem call; file ids match `^[A-Za-z0-9._-]+$` without `__`); the
inventory; the export's evaluator commit equals this checkout's HEAD (TL-9: S = E); the execution
context (T, batch size, device; PLANTSEG_IMAGE_DIGEST equals the export's image digest; P x T is at
most the CPU count); the four recorded values of TL section 5 (LE3), each compared with the cache,
its manifest file and the verification record; the out-root (outside the repository); and the
restart state of every cell.

RESTART RULES (LE10, F-11). At start, a cell whose artifact and record verify is skipped (R2); an
artifact without its record, a record without its artifact or a done cell that no longer verifies is
refused, by name (R3); an orphan temporary directory in the out-root is refused, by name (R5): the
operator moves it aside, never deletes it, and names it in a fault report in AM-8a's form. During
the run, a cell that wrote no artifact and left nothing behind is re-run once into the same out-dir,
and the re-run is logged (R6); a second failure, an orphan left by a failed cell, or a finished
cell that does not verify is a STOP (exit 3): no further cell starts and the running ones finish.

ORDER (plan section 6(ix) step 6). Per model, the 15 inferential cells, then the 9 descriptive, in
protocol order. Models: E1 and E6 at seed 42, then E2, E3, E4, E5, E7, then E1 and E6 at seeds 43
and 44. With P >= 2 the teacher holds one process slot from the start; the other slots take
students, and any slot takes teacher cells once the students are done. Each cell is one process
(R4), and every start is logged in the order it happened (TL-1, one recorded order).

The run log `grid_run_<UTC>.jsonl` (one header line, then one line per event, each fsynced) and the
cell logs `grid_logs_<UTC>/<out-dir name>.<attempt>.log` are written in the out-root.
Exit 0: every cell done. Exit 2: refused, nothing started. Exit 3: STOP.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
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
FILE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
INVALID_CHARS = set(':<>"/\\|?*')
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
    raw = os.path.abspath(str(out_root))
    if name_problem(raw) or set(os.path.basename(raw)) & INVALID_CHARS:
        _refuse("out_name_invalid", f"the out-root {raw!r} holds 'test' or a forbidden character")
    return raw


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
    if not isinstance(fid, str) or not FILE_ID.match(fid) or "__" in fid or name_problem(fid):
        return f"file_id {fid!r} breaks the TL section 2 rule"
    stage, prec = row.get("stage"), row.get("precision")
    if prec not in PRECISIONS:
        return f"{fid}: precision {prec!r}"
    src = row.get("source")
    if not isinstance(src, dict) or src.get("kind") not in ("checkpoint", "provenance") \
            or not isinstance(src.get("path"), str) or not HEX64.match(str(src.get("sha256"))):
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
    if (prec in INT8) != (conv is not None) or (conv is not None and not HEX64.match(str(conv))):
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
    if not HEX40.match(str(doc.get("evaluator_commit"))) or not isinstance(ctx, dict) \
            or not all(k in ctx for k in CONTEXT_KEYS) or not isinstance(doc.get("models"), list) \
            or not HEX64.match(str(doc.get("manifest_sha256"))):
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
    return head if p.returncode == 0 and HEX40.match(head) else None


def check_commit(doc: dict) -> str:
    head = git_head()
    if head != doc["evaluator_commit"]:
        _refuse("evaluator_commit_mismatch",
                f"this checkout is at {head}, the export's evaluator_commit is "
                f"{doc['evaluator_commit']} (TL-9: the session runs at S = E)")
    return head


def check_context(doc: dict, processes: int) -> dict:
    ctx = doc["execution_context"]
    t, batch, device = ctx["torch_num_threads"], ctx["batch_size"], ctx["model_device"]
    for label, v in (("torch_num_threads", t), ("batch_size", batch)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            _refuse("execution_context_invalid", f"execution_context.{label} is {v!r}")
    if not isinstance(device, str) or not device:
        _refuse("execution_context_invalid", f"execution_context.model_device is {device!r}")
    got = os.environ.get("PLANTSEG_IMAGE_DIGEST")
    if (got or None) != ctx["image_digest"]:
        _refuse("image_digest_mismatch",
                f"PLANTSEG_IMAGE_DIGEST is {got!r}, the export's image is {ctx['image_digest']!r}")
    cpus = os.cpu_count() or 1
    if processes < 1 or processes * t > cpus:
        _refuse("processes_threads_exceed_cpus",
                f"--processes {processes} x T {t} exceeds the {cpus} CPUs of this host")
    return {"threads": t, "batch_size": batch, "device": device, "cpus": cpus}


def check_recorded_values(args) -> dict:
    """LE3, TL section 5: the four recorded values, each compared with the cache, its manifest file
    and the verification record, before anything is scored."""
    values = {"canonical_manifest_sha256": args.expected_cache_sha256,
              "inferential_cells_sha256": args.expected_inferential_cells_sha256,
              "manifest_file_sha256": args.expected_manifest_file_sha256,
              "verification_record_sha256": args.expected_verification_record_sha256}
    bad = [k for k, v in values.items() if not HEX64.match(str(v))]
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
    return values


def check_out_root(raw: str) -> Path:
    out_root = Path(raw).resolve()
    if out_root == REPO or REPO in out_root.parents:
        _refuse("out_root_inside_repo", "the out-root is never inside the repository")
    return out_root


def is_orphan(name: str) -> bool:
    """A temporary sibling left by an interrupted writer: the artifact writer's
    `.<out-dir>.tmp-<uuid>` or the record writer's `.<out-dir>.corruption.json.tmp-<pid>`."""
    return name.startswith(".") and ".tmp-" in name


def orphans(out_root: Path, names) -> list:
    if not out_root.is_dir():
        return []
    mine = tuple(f".{n}." for n in names)
    return sorted(p.name for p in out_root.iterdir() if is_orphan(p.name) and p.name.startswith(mine))


def cell_state(out_root: Path, job: dict, values: dict) -> str:
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
        verify_artifact(out_dir)
        record = read_cell_record(rec)
        verify_cell_record(out_dir, record)
    except Exception as e:                                   # noqa: BLE001 -- refused by name
        _refuse("cell_does_not_verify", f"{out_dir.name}: {type(e).__name__}: {e}")
    want = {"stage": job["stage"],
            "condition": {"type": "corruption", "name": job["corruption"],
                          "severity": job["severity"]},
            "canonical_manifest_sha256": values["canonical_manifest_sha256"],
            "verification_record_sha256": values["verification_record_sha256"]}
    got = {"stage": record["stage"], "condition": record["condition"],
           "canonical_manifest_sha256": record["cache"]["canonical_manifest_sha256"],
           "verification_record_sha256": record["cache"]["verification_record_sha256"]}
    if got != want:
        _refuse("cell_does_not_verify", f"{out_dir.name}: its record names {got}, not {want}")
    return "done"


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
    if not teacher and not students:
        _refuse("inventory_invalid", "no row of the export carries a cell")
    if teacher and not args.teacher_config:
        _refuse("teacher_config_required", "the export scores the teacher: pass --teacher-config")
    input_root = Path(args.input_root).resolve()
    for job in teacher + students:
        problem = source_path_problem(job["row"], input_root)
        if problem:
            _refuse("inventory_invalid", problem)
    head = check_commit(doc)
    context = check_context(doc, args.processes)
    values = check_recorded_values(args)
    out_root = check_out_root(out_raw)
    names = [j["out_name"] for j in teacher + students]
    left = orphans(out_root, names)
    if left:
        _refuse("orphan_temporary_directory",
                f"orphan temporary entries in {out_root}: {left}; move each aside (never delete "
                "it) and name it in a fault report in AM-8a's form")
    states = {j["out_name"]: cell_state(out_root, j, values) for j in teacher + students}
    return {"doc": doc, "inventory_sha256": inventory_sha, "teacher": teacher,
            "students": students, "head": head, "context": context, "values": values,
            "out_root": out_root, "input_root": input_root, "states": states}


def finished_ok(job, out_root: Path, values: dict) -> bool:
    try:
        return cell_state(out_root, job, values) == "done"
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
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    log = RunLog(out_root / f"grid_run_{stamp}.jsonl")
    cell_logs = out_root / f"grid_logs_{stamp}"
    cell_logs.mkdir()
    log.write("header", schema=RUN_LOG_SCHEMA, argv=sys.argv, inventory=str(args.inventory),
              inventory_sha256=ready["inventory_sha256"], evaluator_commit=ready["head"],
              manifest_sha256=ready["doc"]["manifest_sha256"], split=args.split,
              artifact_status=args.artifact_status, recorded_values=values,
              processes=args.processes, threads=context["threads"], cpus=context["cpus"],
              plan=order)
    for item in order:
        if item["state"] == "done":
            log.write("skip", out_dir=item["out_dir"], reason="artifact and record verify (R2)")
    running: dict = {}
    attempts: dict = collections.Counter()
    stop = None

    def pick():
        teacher_running = any(j["stage"] == "teacher" for j in running.values())
        if args.processes >= 2 and teacher and not teacher_running:
            return teacher.popleft()
        if students:
            return students.popleft()
        if teacher:
            return teacher.popleft()
        return None

    while stop is None and (teacher or students or running):
        while stop is None and len(running) < args.processes:
            job = pick()
            if job is None:
                break
            attempts[job["out_name"]] += 1
            n = attempts[job["out_name"]]
            argv = cell_argv(job, args, context, ready["input_root"], out_root)
            proc = launcher(argv, cell_logs / f"{job['out_name']}.{n}.log")
            running[proc] = job
            log.write("start", out_dir=job["out_name"], attempt=n, pid=getattr(proc, "pid", None),
                      stage=job["stage"], cell=job["cell"])
        finished = [p for p in running if p.poll() is not None]
        if not finished:
            time.sleep(poll_seconds)
            continue
        for proc in finished:
            job = running.pop(proc)
            rc, name = proc.returncode, job["out_name"]
            if rc == 0 and finished_ok(job, out_root, values):
                log.write("done", out_dir=name, attempt=attempts[name])
                continue
            out_dir = out_root / name
            from src.eval.corrupted import record_path
            left = orphans(out_root, [name])
            if out_dir.exists() or record_path(out_dir).exists() or left:
                stop = _stop("cell_left_partial_output",
                             f"{name} (rc {rc}) left {['artifact'] if out_dir.exists() else []}"
                             f"{['record'] if record_path(out_dir).exists() else []}{left}: move "
                             "each aside, never delete it, and name it in a fault report")
            elif attempts[name] >= 2:
                stop = _stop("cell_failed_twice",
                             f"{name} wrote no artifact twice (rc {rc}); see its cell logs")
            else:
                log.write("rerun", out_dir=name, rc=rc,
                          reason="the cell wrote no artifact; re-run once into the same out-dir")
                (teacher if job["stage"] == "teacher" else students).appendleft(job)
            if stop is not None:
                break
    if stop is not None:
        log.write("stop", code=stop.code, message=str(stop))
        print(f"GRID STOP {stop}", file=sys.stderr)
        for proc, job in list(running.items()):
            proc.wait()
            ok = proc.returncode == 0 and finished_ok(job, out_root, values)
            log.write("done" if ok else "failed", out_dir=job["out_name"], rc=proc.returncode,
                      after_stop=True)
        return EXIT_STOP
    log.write("end", cells=len(order))
    print(f"grid done: {len(order)} cells ({sum(1 for i in order if i['state'] == 'done')} skipped "
          f"as done); log {log.path}")
    return EXIT_DONE


def main(argv=None, *, launcher=None) -> int:
    args = build_parser().parse_args(argv)
    return run_grid(args, launcher=launcher)


if __name__ == "__main__":
    raise SystemExit(main())
