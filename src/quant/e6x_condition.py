"""Lane L-AM23Q, B-1: the E6-X condition of AM-23 item 3(b) (the Phase A plan's §1.1 and §3(v); rulings AQ1-AQ10).

AM-23 item 3(b): "E6-X runs only if E3-X is complete (AM-19 Definitions) and the best VAL all-class mIoU that its
best.json records is greater than E3 seed 42's, 0.36337557435035706." `record` decides that condition once, from
E3-X's run directory and the records committed at HEAD, and writes the record e6x_condition/1:

    verdict "yes" iff Decimal(repr(value)) > Decimal("0.36337557435035706"), strictly: the equal value gives "no"

value is the float that best.json records (best_val_miou_all_class). The threshold is the α selection's winner, E3
seed 42 (DL-121), pinned as THRESHOLD and checked against reports/derived/alpha_selection.json (G5). verdict() is the
one place the outcome is computed; read_condition and read_record_file recompute it from a record's own value and
refuse a record that disagrees as void (condition_void: item 3(b), "an entry whose verdict disagrees with its value
is void").

The record and its row (AQ2, AQ3). The record is created exclusively at --out, outside the repository; Ice commits a
byte-equal copy at reports/arms/e6x_condition_<X>.json, alone in its commit (scripts/arm_go_rule.py :34-37's
pattern). `record` and `print-dl` print its decision-log text: one line of ASCII without "|" or a line break (it sits
in a table cell), naming "E6-<X> condition (AM-23 item 3(b)): yes" (or ": no"), the value's repr, best.json's
sha256, the checkpoint's sha256 and the record's sha256. B-2's launch gate reads the committed copy with
read_condition and requires a RECORDED decision-log row carrying that text.

Determinism and the deadline (AQ7). The record is a function of its inputs and the code alone: no clock, no host and
no absolute path of the machine it runs on (file names and repository paths only), json.dumps(indent=2,
sort_keys=True, allow_nan=False) plus a newline, created exclusively. Its entry deadline is computed from E3-X's
run_end.wall_clock_end, never from a clock: the end of the Asia/Manila day (+08:00, sweep_select.MANILA) after the
Manila day of run_end. B-1 is merged and pinned before E3-R1 or E3-R2 completes; each condition is then recorded,
committed alone and entered in the decision log by that deadline, and before E6-X launches.

The gates of `record`, in order. Each refusal exits 2 (RESULT: REFUSED [<code>]) and nothing is written:
  G0   every flag is given                                                   missing_flags
  G1   no path flag names "test", as given or resolved (qat.names_test), before any file is opened   forbidden_path
  G2   --out: absolute; outside the repository (runner.check_output_dir, E1's guard); named e6x_condition_<X>.json
       for --arm X = R<n>; absent; its folder present
                          out_relative, out_inside_repo, out_name, out_exists, out_parent_missing
  G3   the tool's identity (AQ5): ptq.code_provenance(TOOL_CODE_PATHS), whole; --expect-head is its commit and every
       path is clean at it   expect_head_format, code_identity_unavailable, expect_head_mismatch, code_not_clean
  G4   the arm: the one registered kd_arm/1 entry (kd_arms.load_arms) whose arm teacher is X   arm_spec, arm_unknown
  G5   the λ and α selections of record, tracked and unchanged at HEAD with their sha256 (qat.read_tracked_selection
       and its codes), with finite winner values; the α winner's best_val is THRESHOLD
                          selection_not_of_record, selection_format, threshold_mismatch
  G6   reports/arms/arm_teacher_record_<X>.json committed at HEAD: arm_teacher_record/1, arm X, status complete, a
       selected checkpoint sha256   teacher_record_missing, teacher_record_uncommitted, teacher_record_invalid
  G7   the run directory: absolute, outside the repository, named <arm>_s42_<...>, holding e3_run_meta.jsonl,
       e3_telemetry.jsonl and best.json; one run_meta row, its fields those of E3-X of record, each refused alone
                          run_dir_relative, run_dir_inside_repo, run_dir_name, run_records_missing, run_meta_rows,
                          run_meta_mismatch
  G8   complete (AM-19 Definitions: the run_end record with its checks passed): every telemetry line JSON, the last
       one whole, no run_abort, exactly one run_end and it the last row, checks_passed true, iter 80000 and a finite
       wall_clock_end. No run_end: exit 3, RESULT: INCOMPLETE [run_end_missing]
                          telemetry_format, telemetry_torn, parent_aborted, run_end_not_last, parent_checks_failed,
                          run_end_format
  G9   best.json: exactly best_ckpt and best_val_miou_all_class, naming e3_student_best_iter<n>.pt; best.json's value,
       run_end's and the payload's each a float, finite and in [0, 1] (AQ4: before any verdict); the checkpoint
       present, hashed, and loaded from those bytes: stage E3, the arm's arm and target_construction, the same best;
       run_end's best and best_ckpt equal best.json's
                          best_json_format, value_invalid, checkpoint_missing, payload_mismatch, run_end_mismatch
  G10  reports/kd_launch_log.jsonl at HEAD (sweep_select.load_launch_log and its codes): the run's launch and launched
       lines and no stopped line; PL-14's fields (sweep_select.check_run_against_log :1809-1815) with the arm's run
       stage for stage, and sweep null                                       launch_log (the field named)
Then the verdict, and the record written exclusively and read back (a difference: exit 1, STOP [write_mismatch]).
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from datetime import time as day_start
from decimal import Decimal
from pathlib import Path, PurePosixPath

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training import kd_arms as K                                       # noqa: E402
from src.training import sweep_select as SS                                 # noqa: E402
from . import ptq as P                                                      # noqa: E402
from . import qat as Q                                                      # noqa: E402
from . import runner as R                                                   # noqa: E402
from .qat import QATRefused                                                 # noqa: E402

LANE = "L-AM23Q B-1 (AM-23 item 3(b); the Phase A plan §1.1 and §3(v); rulings AQ1-AQ10)"
FORMAT = "e6x_condition/1"
AMENDMENT = "AM-23 item 3(b)"
THRESHOLD = "0.36337557435035706"
THRESHOLD_SOURCE = ("reports/derived/alpha_selection.json winner.best_val: E3 seed 42, run "
                    "e3_s42_lambda0.5_alpha25_80k_a1 (DL-121)")
COMPARISON = "Decimal(repr(value)) > Decimal(threshold), strictly; the equal value gives no"
VALUE_SOURCE = "best.json"
DEADLINE_RULE = ("AM-23 item 3(b): entered in the decision log by the end of the day after E3-X's run_end "
                 "(Asia/Manila, +08:00), and before E6-X launches; computed from run_end.wall_clock_end")
SEED = 42
HORIZON = 80000
PARENT_STAGE = "E3"                    # the trainer's stage name of every arm run (kd_arms.STAGE "e3")
LAMBDA_SELECTION_REL = "reports/derived/lambda_selection.json"
ALPHA_SELECTION_REL = "reports/derived/alpha_selection.json"
TEACHER_RECORD_REL = "reports/arms/arm_teacher_record_{arm}.json"
TEACHER_RECORD_FORMAT = "arm_teacher_record/1"
RECORD_REL = "reports/arms/e6x_condition_{arm}.json"      # the committed copy, committed alone (AQ2)
OUT_NAME = "e6x_condition_{arm}.json"
META_NAME, TELEMETRY_NAME, BEST_JSON_NAME = "e3_run_meta.jsonl", "e3_telemetry.jsonl", "best.json"
ARM_RE = re.compile(r"R[0-9]+")
CKPT_RE = re.compile(r"e3_student_best_iter[0-9]+\.pt")
RUN_TAIL_RE = re.compile(r"[A-Za-z0-9._-]+")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# AQ5: the module, the script and every repository file whose code or data decides a gate or the verdict: qat.py
# (names_test, read_tracked_selection), runner.py and train_e1.py (check_output_dir and the _assert_outside_repo it
# calls), ptq.py (code_provenance), sweep_select.py (HeadSource, load_launch_log, _same) and kd_arms.py (load_arms),
# and the arm registry: the folder (git status names an untracked file in it) and each of its files (hashed).
TOOL_CODE_PATHS = (
    "src/quant/e6x_condition.py", "scripts/e6x_condition.py", "src/quant/qat.py", "src/quant/runner.py",
    "src/training/train_e1.py", "src/quant/ptq.py", "src/training/sweep_select.py", "src/training/kd_arms.py",
    "configs/kd_arms", "configs/kd_arms/e3_flip.json", "configs/kd_arms/e3_r1.json", "configs/kd_arms/e3_r2.json",
)
RECORD_KEYS = ("amendment", "arm", "best_json", "checkpoint", "dl_deadline", "format", "launch_log",
               "parent_run_stage", "run_end", "run_id", "run_meta", "seed", "selections", "stage", "teacher_record",
               "telemetry", "threshold", "tool", "value", "verdict")
RECORD_FLAGS = ("arm", "run_dir", "lambda_selection", "lambda_selection_sha256", "alpha_selection",
                "alpha_selection_sha256", "expect_head", "out")
RECORD_PATH_FLAGS = ("run_dir", "lambda_selection", "alpha_selection", "out")
PRINT_FLAGS = ("record", "record_sha256")
EXIT_WRITTEN, EXIT_STOP, EXIT_REFUSED, EXIT_INCOMPLETE, EXIT_ERROR = 0, 1, 2, 3, 4
# The repository the records are read from, at its HEAD (None: this checkout). No flag redirects it; the smokes point
# it at a records repository in-process, as sweep_select.GIT_ROOT and ptq_record.RECORDS_ROOT.
RECORDS_ROOT = None


class ConditionRefused(QATRefused):
    """A named refusal before anything is written (exit 2). A QATRefused, so B-2's launch gate keeps its code."""


class ConditionIncomplete(RuntimeError):
    """E3-X has no run_end record yet (exit 3); nothing is written."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ConditionStop(RuntimeError):
    """The written record does not read back as written (exit 1)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _refuse(code: str, message: str):
    raise ConditionRefused(code, message)


def records_root() -> Path:
    return Path(REPO if RECORDS_ROOT is None else RECORDS_ROOT)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


# ------------------------------------------------------------------ the verdict (AQ4)
def value_error(x) -> str | None:
    """AQ4: why x cannot be compared, or None. A float (never a bool, an int or a string), finite, in [0, 1]: a NaN
    would make the Decimal comparison raise, and json.loads accepts NaN and Infinity."""
    if not isinstance(x, float):
        return f"{x!r} is {type(x).__name__}, not a float"
    if not math.isfinite(x):
        return f"{x!r} is not finite"
    if not 0.0 <= x <= 1.0:
        return f"{x!r} is outside [0, 1]"
    return None


def verdict(value: float) -> str:
    """The one place the outcome is computed: "yes" iff Decimal(repr(value)) > Decimal(THRESHOLD), strictly."""
    err = value_error(value)
    if err is not None:
        raise ValueError(f"value_invalid: {err}")
    return "yes" if Decimal(repr(float(value))) > Decimal(THRESHOLD) else "no"


# ------------------------------------------------------------------ G0, G1
def check_flags(args, names) -> None:
    """G0: every flag of the command is given."""
    missing = [_flag(n) for n in names if getattr(args, n, None) in (None, "")]
    if missing:
        _refuse("missing_flags", f"{', '.join(missing)}: required (every flag of this command is)")


def check_names(args, names) -> None:
    """G1 (SL-1): no path flag names "test", on the given string or its resolved form (qat.names_test), before any
    file is opened."""
    for n in names:
        value = getattr(args, n)
        if Q.names_test(value):
            _refuse("forbidden_path", f"{_flag(n)} {value!r} contains 'test' as given or resolved (SL-1); nothing "
                                      "was opened")


# ------------------------------------------------------------------ G2, G3, G4
def check_out(value: str, arm: str) -> Path:
    """G2: the record's path, outside the repository and new."""
    if not Path(value).is_absolute():
        _refuse("out_relative", f"--out {value!r} must be an absolute path")
    try:
        out = R.check_output_dir(value)
    except R.QuantRunError as e:
        _refuse("out_inside_repo", f"--out {value}: {e}. The record is written outside the repository; Ice commits "
                                   "a byte-equal copy")
    want = OUT_NAME.format(arm=arm)
    if not ARM_RE.fullmatch(arm) or out.name != want:
        _refuse("out_name", f"--out is named {out.name!r}; with --arm R<n> it is {want!r}")
    if os.path.lexists(out):
        _refuse("out_exists", f"{out} exists: a condition record is created once, exclusively")
    if not out.parent.is_dir():
        _refuse("out_parent_missing", f"{out.parent} is not an existing folder")
    return out


def tool_identity(expect_head: str) -> dict:
    """G3 (AQ5): ptq.code_provenance over TOOL_CODE_PATHS, whole (commit, clean_at_commit, files_sha256)."""
    if not _HEX40.fullmatch(expect_head):
        _refuse("expect_head_format", f"--expect-head {expect_head!r} must be the clone's 40-hex commit")
    prov = P.code_provenance(TOOL_CODE_PATHS)
    if prov["commit"] is None:
        _refuse("code_identity_unavailable", "git names no HEAD commit for this clone, so the record could not name "
                                             "the code it ran (a mounted clone owned by another user needs "
                                             "safe.directory)")
    if prov["commit"] != expect_head:
        _refuse("expect_head_mismatch", f"the clone's HEAD is {prov['commit']}; --expect-head is {expect_head}")
    if prov["clean_at_commit"] is not True:
        _refuse("code_not_clean", "the tool's code paths (TOOL_CODE_PATHS: an edited or untracked file, the arm "
                                  "registry included) are not clean at HEAD")
    return {"code_paths": list(TOOL_CODE_PATHS), **prov}


def arm_spec(arm: str):
    """G4: the registered arm whose arm teacher is X (kd_arms: E3-X distils from X)."""
    try:
        arms = K.load_arms()
    except K.ArmSpecError as e:
        _refuse("arm_spec", f"the arm registry configs/kd_arms/ is not kd_arm/1: {e}")
    found = [s for s in arms.values() if s.teacher == "arm" and s.arm_teacher == arm]
    if len(found) != 1:
        _refuse("arm_unknown", f"--arm {arm!r} is the arm teacher of {len(found)} registered arms, not one "
                               f"(registered: {sorted(arms)})")
    return found[0]


# ------------------------------------------------------------------ G5, G6: the records at HEAD
def selections(args, root: Path) -> tuple[dict, dict]:
    """G5: the λ and α selections of record; the α winner's best_val is the pinned threshold (DL-121)."""
    if args.lambda_selection != LAMBDA_SELECTION_REL or args.alpha_selection != ALPHA_SELECTION_REL:
        _refuse("selection_not_of_record", f"the selections of record are {LAMBDA_SELECTION_REL} and "
                                           f"{ALPHA_SELECTION_REL}, not {args.lambda_selection!r} and "
                                           f"{args.alpha_selection!r}")
    lam = Q.read_tracked_selection(args.lambda_selection, args.lambda_selection_sha256, Q.LAMBDA_SELECTION_FORMAT,
                                   "lambda", root)
    alpha = Q.read_tracked_selection(args.alpha_selection, args.alpha_selection_sha256, Q.ALPHA_SELECTION_FORMAT,
                                     "alpha", root)
    if not (SS._finite_number(lam["winner"].get("lambda")) and SS._finite_number(alpha["winner"].get("alpha"))):
        _refuse("selection_format", "the λ winner's lambda and the α winner's alpha must be finite numbers")
    best = alpha["winner"].get("best_val")
    if not (isinstance(best, float) and repr(best) == THRESHOLD):
        _refuse("threshold_mismatch", f"the α winner's best_val is {best!r}; AM-23 item 3(b)'s value is {THRESHOLD}")
    return lam, alpha


def teacher_record(arm: str, root: Path) -> dict:
    """G6: the arm teacher's record, committed at HEAD (sweep_select.HeadSource: present and byte-equal)."""
    try:
        rel, data = SS.HeadSource(root).read(TEACHER_RECORD_REL.format(arm=arm), missing="teacher_record_missing",
                                             uncommitted="teacher_record_uncommitted")
    except SS.SelectionRefused as e:
        _refuse(e.code, str(e))
    try:
        doc = json.loads(data.decode("utf-8"))
        ck = doc["selection"]["checkpoint"]
        ok = (doc["format"] == TEACHER_RECORD_FORMAT and doc["arm"] == arm and doc["status"] == "complete"
              and isinstance(ck["sha256"], str) and _HEX64.fullmatch(ck["sha256"]) is not None)
    except (ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        _refuse("teacher_record_invalid", f"{rel} is not a complete {TEACHER_RECORD_FORMAT} record of arm {arm} "
                                          "with a selected checkpoint sha256")
    return {"path": rel, "sha256": Q.sha256_bytes(data), "checkpoint_name": ck.get("name"),
            "checkpoint_sha256": ck["sha256"]}


# ------------------------------------------------------------------ G7: the run directory and run_meta
def _jsonl(data: bytes) -> tuple[list, bool]:
    """The rows of JSONL bytes and whether the last line is torn (qat.read_jsonl's reading, on bytes already hashed);
    any other line that is not JSON raises ValueError."""
    lines = data.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    rows, torn = [], False
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            torn = True
    return rows, torn


def run_dir(value: str, spec) -> Path:
    """G7: E3-X's run directory, outside the repository, named for the arm, with its three record files."""
    if not Path(value).is_absolute():
        _refuse("run_dir_relative", f"--run-dir {value!r} must be an absolute path")
    try:
        d = R.check_output_dir(value)
    except R.QuantRunError as e:
        _refuse("run_dir_inside_repo", f"--run-dir {value}: {e}")
    prefix = f"{spec.name}_s{SEED}_"
    if not (d.name.startswith(prefix) and RUN_TAIL_RE.fullmatch(d.name[len(prefix):])):
        _refuse("run_dir_name", f"--run-dir is named {d.name!r}; {spec.run_stage}'s run is {prefix}<letters, digits, "
                                "'.', '_' or '-'> (scripts/preflight_arms.py run_id_of)")
    missing = [n for n in (META_NAME, TELEMETRY_NAME, BEST_JSON_NAME) if not (d / n).is_file()]
    if missing:
        _refuse("run_records_missing", f"{d} lacks {missing}")
    return d


def run_meta_row(d: Path) -> tuple[dict, bytes]:
    """G7: e3_run_meta.jsonl holds exactly one row, the run_meta row (a second row is an appended relaunch)."""
    data = (d / META_NAME).read_bytes()
    try:
        rows, torn = _jsonl(data)
    except ValueError:
        rows, torn = [], True
    if torn or len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("event") != "run_meta":
        _refuse("run_meta_rows", f"{META_NAME} holds {len(rows)} readable rows (torn: {torn}); exactly one run_meta "
                                 "row is required")
    return rows[0], data


def run_meta_checks(m: dict, spec, lam: dict, alpha: dict, teacher: dict) -> None:
    """G7: the run_meta fields of E3-X of record, each refused alone (sweep_select._same: numbers by value, never a
    bool; other values by type and value)."""
    prov = m.get("teacher_provenance")
    fields = (
        ("mode", m.get("mode"), "real"),
        ("stage", m.get("stage"), PARENT_STAGE),
        ("seed", m.get("seed"), SEED),
        ("arm", m.get("arm"), spec.run_stage),
        ("descriptive", m.get("descriptive"), True),
        ("parent_of_e4_e7", m.get("parent_of_e4_e7"), False),
        ("max_iters", m.get("max_iters"), HORIZON),
        ("poly_horizon", m.get("poly_horizon"), HORIZON),
        ("target_construction", m.get("target_construction"), spec.target_construction),
        ("lambda_logit", m.get("lambda_logit"), lam["winner"]["lambda"]),
        ("alpha_cwd", m.get("alpha_cwd"), alpha["winner"]["alpha"]),
        ("teacher_mock", m.get("teacher_mock"), False),
        ("teacher_provenance.ckpt_sha256", prov.get("ckpt_sha256") if isinstance(prov, dict) else None,
         teacher["checkpoint_sha256"]),
    )
    for key, got, want in fields:
        if not SS._same(got, want):
            _refuse("run_meta_mismatch", f"run_meta {key} is {got!r}; {spec.run_stage} of record has {want!r}")


# ------------------------------------------------------------------ G8: complete (AM-19 Definitions)
def completion(d: Path) -> tuple[dict, bytes]:
    """G8: the run_end record with its checks passed, the telemetry's last row; no run_abort."""
    data = (d / TELEMETRY_NAME).read_bytes()
    try:
        rows, torn = _jsonl(data)
    except ValueError as e:
        _refuse("telemetry_format", f"{TELEMETRY_NAME} has a line that is not JSON before its last line ({e})")
    if torn:
        _refuse("telemetry_torn", f"{TELEMETRY_NAME}'s last line is torn: the copy ended mid-write")
    if not all(isinstance(r, dict) for r in rows):
        _refuse("telemetry_format", f"{TELEMETRY_NAME} has a row that is not a JSON object")
    if any(r.get("event") == "run_abort" for r in rows):
        _refuse("parent_aborted", f"{TELEMETRY_NAME} holds a run_abort row: E3-X diverged or aborted, so no "
                                  "condition is recorded (AM-23 item 3(b): not run: parent diverged)")
    ends = [r for r in rows if r.get("event") == "run_end"]
    if not ends:
        raise ConditionIncomplete("run_end_missing", f"{TELEMETRY_NAME} has no run_end record: E3-X is not complete "
                                                     "(AM-19 Definitions)")
    if len(ends) != 1 or rows[-1] is not ends[0]:
        _refuse("run_end_not_last", f"{TELEMETRY_NAME} holds {len(ends)} run_end rows, or rows after it; exactly one, "
                                    "the last row, is the trainer's")
    end = ends[0]
    if end.get("checks_passed") is not True:
        _refuse("parent_checks_failed", f"run_end checks_passed is {end.get('checks_passed')!r}: complete means the "
                                        "run_end record with its checks passed (AM-19 Definitions)")
    it, t = end.get("iter"), end.get("wall_clock_end")
    if not (type(it) is int and it == HORIZON and SS._finite_number(t)):
        _refuse("run_end_format", f"run_end iter {it!r} (the horizon is {HORIZON}) and wall_clock_end {t!r} (a finite "
                                  "number) cannot date the run's end")
    return end, data


# ------------------------------------------------------------------ G9: best.json and the checkpoint
def best_checkpoint(d: Path, spec, end: dict) -> dict:
    """G9: the value best.json records, its checkpoint and the payload, and run_end's agreement."""
    best_data = (d / BEST_JSON_NAME).read_bytes()
    try:
        best = json.loads(best_data.decode("utf-8"))
    except ValueError as e:
        _refuse("best_json_format", f"{BEST_JSON_NAME} is not JSON ({e})")
    if not isinstance(best, dict) or sorted(best) != sorted(SS.BEST_JSON_KEYS) or not isinstance(best["best_ckpt"],
                                                                                                str):
        _refuse("best_json_format", f"{BEST_JSON_NAME} must hold exactly {SS.BEST_JSON_KEYS}, best_ckpt a string")
    name = PurePosixPath(best["best_ckpt"]).name
    if not CKPT_RE.fullmatch(name):
        _refuse("best_json_format", f"{BEST_JSON_NAME} names {name!r}, not e3_student_best_iter<n>.pt")
    value = best["best_val_miou_all_class"]
    for what, x in (("best.json's best_val_miou_all_class", value),
                    ("run_end's best_val_miou_all_class", end.get("best_val_miou_all_class"))):
        err = value_error(x)
        if err is not None:
            _refuse("value_invalid", f"{what}: {err} (AQ4: a float, finite, in [0, 1])")
    ck = d / name
    if not ck.is_file():
        _refuse("checkpoint_missing", f"{BEST_JSON_NAME} names {name}, absent from {d}")
    raw = ck.read_bytes()
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 -- any unreadable checkpoint is refused by name
        _refuse("payload_mismatch", f"{name} does not load ({type(e).__name__}: {e})")
    if not isinstance(payload, dict):
        _refuse("payload_mismatch", f"{name} holds a {type(payload).__name__}, not a checkpoint payload")
    err = value_error(payload.get("best_val_miou_all_class"))
    if err is not None:
        _refuse("value_invalid", f"the payload's best_val_miou_all_class: {err} (AQ4: a float, finite, in [0, 1])")
    for key, want in (("stage", PARENT_STAGE),
                      ("arm", spec.run_stage),
                      ("target_construction", spec.target_construction),
                      ("best_val_miou_all_class", value)):
        if not SS._same(payload.get(key), want):
            _refuse("payload_mismatch", f"{name}'s payload {key} is {payload.get(key)!r}, not {want!r}")
    for key, want in (("best_val_miou_all_class", value),
                      ("best_ckpt", name)):
        if not SS._same(end.get(key), want):
            _refuse("run_end_mismatch", f"run_end {key} is {end.get(key)!r}; best.json's is {want!r}")
    return {"value": float(value), "best_json_sha256": Q.sha256_bytes(best_data), "checkpoint": name,
            "checkpoint_sha256": Q.sha256_bytes(raw), "checkpoint_bytes": len(raw)}


# ------------------------------------------------------------------ G10: the launch log at HEAD
def launch_binding(root: Path, d: Path, m: dict, meta_data: bytes, spec) -> dict:
    """G10: the run's lines in the committed KD launch log (PL-13/PL-14), with the arm's run stage for stage."""
    try:
        log = SS.load_launch_log(SS.HeadSource(root))
    except SS.SelectionRefused as e:
        _refuse(e.code, str(e))
    rid = d.name
    line, ld = log.launch.get(rid), log.launched.get(rid)
    if line is None:
        _refuse("launch_log", f"launch: {SS.LAUNCH_LOG_REL} at HEAD has no launch line for {rid}")
    if ld is None:
        _refuse("launch_log", f"launched: {SS.LAUNCH_LOG_REL} at HEAD has no launched line for {rid}")
    if rid in log.stopped:
        _refuse("launch_log", f"stopped: {SS.LAUNCH_LOG_REL} names {rid} stopped, which no complete run is")
    fields = (
        ("run_meta_sha256", Q.sha256_bytes(meta_data), ld["run_meta_sha256"]),
        ("git_head", m.get("git_head"), ld["git_head"]),
        ("code_pin", ld["git_head"], line["code_pin"]),
        ("records_commit", m.get("records_commit"), ld["records_commit"]),
        ("stage", m.get("arm"), line["stage"]),
        ("seed", m.get("seed"), line["seed"]),
        ("horizon", m.get("poly_horizon"), line["horizon"]),
        ("sweep", line["sweep"], None),
        ("lambda_logit", m.get("lambda_logit"), line["lambda_logit"]),
        ("alpha_cwd", m.get("alpha_cwd"), line["alpha_cwd"]),
        ("launch_time_utc", m.get("wall_clock"), ld["launch_time_utc"]),
    )
    for key, got, want in fields:
        if not SS._same(got, want):
            _refuse("launch_log", f"{key}: run {rid} has {got!r}, its launch-log lines {want!r} (PL-14)")
    return {"path": SS.LAUNCH_LOG_REL, "sha256": log.sha256, "launch": line, "launched": ld}


# ------------------------------------------------------------------ the record
def deadline(wall_clock_end: float) -> dict:
    """AQ7: the entry deadline from run_end's time, never a clock: the end of the Manila day after run_end's. The time
    is taken in whole seconds, floored: datetime.fromtimestamp rounds to the microsecond, which would date the last
    half microsecond before Manila midnight to the next day (DL-24 NUM-3)."""
    end = datetime.fromtimestamp(math.floor(wall_clock_end), timezone.utc)
    day = end.astimezone(SS.MANILA).date()
    enter_by = day + timedelta(days=1)
    closes = datetime.combine(enter_by + timedelta(days=1), day_start(0, 0), tzinfo=SS.MANILA)
    return {"run_end_wall_clock_end": wall_clock_end, "run_end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_end_manila_date": day.isoformat(), "enter_by_manila_date": enter_by.isoformat(),
            "enter_by_end_utc": closes.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timezone": "Asia/Manila (+08:00)", "rule": DEADLINE_RULE}


def build_record(*, arm: str, spec, run_id: str, best: dict, meta_data: bytes, tel_data: bytes, end: dict,
                 lam: dict, alpha: dict, teacher: dict, launch: dict, tool: dict) -> dict:
    v = best["value"]
    return {
        "format": FORMAT,
        "amendment": AMENDMENT,
        "arm": arm,
        "stage": "E6-" + arm,
        "parent_run_stage": spec.run_stage,
        "run_id": run_id,
        "seed": SEED,
        "value": {"best_val_miou_all_class": v, "repr": repr(v), "source": VALUE_SOURCE},
        "threshold": {"repr": THRESHOLD, "source": THRESHOLD_SOURCE, "comparison": COMPARISON},
        "verdict": verdict(v),
        "best_json": {"file": BEST_JSON_NAME, "sha256": best["best_json_sha256"]},
        "checkpoint": {"file": best["checkpoint"], "sha256": best["checkpoint_sha256"],
                       "bytes": best["checkpoint_bytes"]},
        "run_meta": {"file": META_NAME, "sha256": Q.sha256_bytes(meta_data)},
        "telemetry": {"file": TELEMETRY_NAME, "sha256": Q.sha256_bytes(tel_data)},
        "run_end": {k: end.get(k) for k in ("iter", "checks_passed", "best_val_miou_all_class", "best_ckpt",
                                            "wall_clock_end")},
        "dl_deadline": deadline(end["wall_clock_end"]),
        "selections": {"lambda": {k: lam[k] for k in ("path", "sha256", "winner")},
                       "alpha": {k: alpha[k] for k in ("path", "sha256", "winner")}},
        "teacher_record": teacher,
        "launch_log": launch,
        "tool": tool,
    }


def record_bytes(rec: dict) -> bytes:
    return (json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_back(path: Path) -> bytes:
    return Path(path).read_bytes()


def write_record(out: Path, data: bytes) -> None:
    """Created exclusively ("xb") and read back; a record that does not read back as written is a STOP."""
    try:
        with open(out, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        _refuse("out_exists", f"{out} appeared after G2 and before the write; nothing was written")
    if _read_back(out) != data:
        raise ConditionStop("write_mismatch", f"{out} does not read back as written; the record is not usable")


def result_line(rec: dict) -> str:
    sign = ">" if rec["verdict"] == "yes" else "<="
    return f"RESULT: {rec['stage']} CONDITION {rec['verdict']} ({rec['value']['repr']} {sign} {THRESHOLD})"


def dl_text(rec: dict, record_sha256: str) -> str:
    """AQ3: the decision-log text, one line of ASCII, without "|" (it sits in a table cell)."""
    dd, stage = rec["dl_deadline"], rec["stage"]
    sign = ">" if rec["verdict"] == "yes" else "<="
    tail = (f"to be entered by the end of {dd['enter_by_manila_date']} (Asia/Manila) and before {stage} launches"
            if rec["verdict"] == "yes" else
            f"{stage} is ruled out; to be entered by the end of {dd['enter_by_manila_date']} (Asia/Manila)")
    text = (f"{stage} condition ({AMENDMENT}): {rec['verdict']}. {rec['parent_run_stage']} run {rec['run_id']} "
            f"complete (run_end {dd['run_end_utc']}, Manila {dd['run_end_manila_date']}); best VAL all-class mIoU "
            f"{rec['value']['repr']} (best.json) {sign} {THRESHOLD} (E3 seed 42, DL-121); best.json sha256 "
            f"{rec['best_json']['sha256']}; checkpoint {rec['checkpoint']['file']} sha256 "
            f"{rec['checkpoint']['sha256']}; record {RECORD_REL.format(arm=rec['arm'])} sha256 {record_sha256} "
            f"({FORMAT}, tool {rec['tool']['commit']}); {tail}.")
    if "|" in text or "\n" in text or "\r" in text or not text.isascii():
        raise ValueError("the decision-log text must be one line of ASCII without '|'")
    return text


def record_condition(args) -> tuple[dict, str, Path, str]:
    """`record`: G0-G10, the verdict, the record written once. Returns (record, sha256, out, decision-log text)."""
    check_flags(args, RECORD_FLAGS)                                                 # G0
    check_names(args, RECORD_PATH_FLAGS)                                            # G1
    out = check_out(args.out, args.arm)                                             # G2
    tool = tool_identity(args.expect_head)                                          # G3
    spec = arm_spec(args.arm)                                                       # G4
    root = records_root()
    lam, alpha = selections(args, root)                                             # G5
    teacher = teacher_record(args.arm, root)                                        # G6
    d = run_dir(args.run_dir, spec)                                                 # G7
    meta, meta_data = run_meta_row(d)
    run_meta_checks(meta, spec, lam, alpha, teacher)
    end, tel_data = completion(d)                                                   # G8
    best = best_checkpoint(d, spec, end)                                            # G9
    launch = launch_binding(root, d, meta, meta_data, spec)                         # G10
    rec = build_record(arm=args.arm, spec=spec, run_id=d.name, best=best, meta_data=meta_data, tel_data=tel_data,
                       end=end, lam=lam, alpha=alpha, teacher=teacher, launch=launch, tool=tool)
    data = record_bytes(rec)
    sha = Q.sha256_bytes(data)
    text = dl_text(rec, sha)                    # the row is checked before anything is written
    write_record(out, data)
    return rec, sha, out, text


# ------------------------------------------------------------------ the readers (print-dl; B-2's launch gate)
def _strict_pairs(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"repeated keys {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")


def parse_record(data: bytes) -> dict:
    """An intact e6x_condition/1 from its bytes: strict JSON (no NaN, Infinity or repeated key), exactly RECORD_KEYS,
    the pinned threshold, a comparable value whose repr it records, the arm's names, and its verdict the one that
    verdict() gives its value; a record whose verdict disagrees is void (AM-23 item 3(b))."""
    try:
        rec = json.loads(data.decode("utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_no_constant)
    except ValueError as e:
        _refuse("condition_format", f"not strict JSON ({e})")
    if not isinstance(rec, dict) or sorted(rec) != sorted(RECORD_KEYS) or rec["format"] != FORMAT:
        _refuse("condition_format", f"not a {FORMAT} record with exactly its keys")
    try:
        x, arm, v = rec["value"]["best_val_miou_all_class"], rec["arm"], rec["value"]["repr"]
        ok = (value_error(x) is None and v == repr(x) and rec["threshold"]["repr"] == THRESHOLD
              and rec["verdict"] in ("yes", "no") and ARM_RE.fullmatch(arm) is not None
              and rec["stage"] == "E6-" + arm and rec["parent_run_stage"] == "E3-" + arm
              and RUN_TAIL_RE.fullmatch(rec["run_id"]) is not None
              and CKPT_RE.fullmatch(rec["checkpoint"]["file"]) is not None
              and _HEX40.fullmatch(rec["tool"]["commit"]) is not None
              and all(isinstance(rec["dl_deadline"][k], str)
                      for k in ("run_end_utc", "run_end_manila_date", "enter_by_manila_date")))
    except (KeyError, TypeError):
        ok = False
    if not ok:
        _refuse("condition_format", f"a {FORMAT} field is missing or malformed (value, threshold, verdict, arm, "
                                    "run_id, checkpoint, tool commit or deadline)")
    if verdict(x) != rec["verdict"]:
        _refuse("condition_void", f"the record's verdict {rec['verdict']!r} disagrees with its value {v} against "
                                  f"{THRESHOLD}: void (AM-23 item 3(b)); a corrected entry is a new record")
    return rec


def read_record_file(path, expect_sha256) -> tuple[dict, str]:
    """print-dl's reader: a record file by path, with its expected sha256."""
    if not isinstance(expect_sha256, str) or not _HEX64.fullmatch(expect_sha256):
        _refuse("condition_sha256_format", "--record-sha256 must be 64 lowercase hex characters")
    p = Path(path)
    if Q.names_test(p):
        _refuse("condition_forbidden_path", f"{p} contains 'test' as given or resolved (SL-1)")
    if not p.is_file():
        _refuse("condition_missing_file", f"{p} is not a file")
    data = p.read_bytes()
    sha = Q.sha256_bytes(data)
    if sha != expect_sha256:
        _refuse("condition_sha256_mismatch", f"{p} has sha256 {sha}, expected {expect_sha256}")
    return parse_record(data), sha


def read_condition(rel_path, expect_sha256, repo_root=None) -> tuple[dict, str]:
    """B-2's reader: the committed copy, repo-relative, tracked and unchanged at HEAD, with its expected sha256."""
    root = records_root() if repo_root is None else Path(repo_root)
    if not isinstance(expect_sha256, str) or not _HEX64.fullmatch(expect_sha256):
        _refuse("condition_sha256_format", "the condition record's sha256 must be 64 lowercase hex characters")
    rel = Path(str(rel_path))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        _refuse("condition_path_not_repo_relative", f"{rel_path!r} must be repo-relative")
    full = root / rel
    if Q.names_test(rel) or Q.names_test(full):
        _refuse("condition_forbidden_path", f"{rel_path!r} contains 'test' as given or resolved (SL-1)")
    if not full.is_file():
        _refuse("condition_missing_file", f"{full} is not a file")
    data = full.read_bytes()
    head = Q._git_head_bytes(root, rel.as_posix())
    if head is None:
        _refuse("condition_not_tracked", f"{rel.as_posix()} is not tracked at HEAD")
    if head != data:
        _refuse("condition_changed_since_head", f"{rel.as_posix()} differs from its bytes at HEAD")
    sha = Q.sha256_bytes(data)
    if sha != expect_sha256:
        _refuse("condition_sha256_mismatch", f"{rel.as_posix()} has sha256 {sha}, expected {expect_sha256}")
    return parse_record(data), sha
