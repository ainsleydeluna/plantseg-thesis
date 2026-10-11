#!/usr/bin/env python3
"""The R1 / R2 teacher-arm fine-tune launcher (lane L-TEACHER-ARMS, commit C3; AM-18 items 2 and 7(a); ARMS plan
§1.1 and decision (ix) as the DL-24 audit's part 3 corrects them; Phase B-2 rulings R1 to R7).

It extends scripts/launch_teacher_finetune.py ("the frozen launcher", "ltf"), imported unmodified: its
preflight() (config file, data root, M11 TRAIN/VAL-only isolation, the ADE20K init checkpoint by sha256, the work
dir outside the repository, CUDA), check_inherited_cublas(), check_governed_clean(), check_locked_config() and
launch(). The frozen load_launch_config() is never run on an arm configuration: it pins the record's three hooks
and its train pipeline (ruling R1, audit F4).

  python -B scripts/launch_teacher_arm.py --arm R1|R2 --real-run --confirm-real-run
      --config configs/teacher/arms/<r1_bema|r2_bema_rare>.py --config-sha256 HEX --config-dl DL-n
      --data-root DIR --init-ckpt PATH --work-dir DIR [--create-work-dir]
      --code-pin P_T --records-commit H --approval-dl DL-n --recheck-dl DL-n
      --r2-config-dl DL-n --r2-table PATH --r2-table-sha256 HEX --r2-table-dl DL-n
      [--r1-record PATH --r1-record-sha256 HEX --r1-record-dl DL-n]                 R2: required; R1: refused
      [--previous-record PATH --previous-record-sha256 HEX --am8a-report REL]     attempt n > 1: required
      [--reference-log PATH --reference-log-sha256 HEX] [--skip-cuda-probe] [--launch]
  python -B scripts/launch_teacher_arm.py --show-diff --arm R1|R2
      REG[arm]'s canonical JSON and its sha256 (the configuration row's value); reads nothing else.

Gates, in this order; the first failure refuses ("PREFLIGHT FAILED [code]", exit 2). Without --launch the run is
the rehearsal: every gate but the launch runs, so PREFLIGHT PASS covers every non-GPU gate (ruling R7, audit F22).
  1  authorization  --real-run and --confirm-real-run; --launch refuses --skip-cuda-probe (as the frozen launcher)
  2  flags          the arm's flag set (explicit paths only: the frozen environment fallbacks never apply)
  3  paths          every path argument through teacher_diag.refuse_test_path, the as-given string first (SL-1)
  4  work dir       absolute, named teacher-<r1|r2>_s42_40k_a<n>, absent or empty; n is the attempt (R4(b))
  5  frozen gates   check_inherited_cublas() and check_governed_clean(), then preflight()
  6  checkout       --config is the arm's file with sha256 --config-sha256; HEAD = --code-pin (P_T); the records
                    commit H exists and P_T is an ancestor of H (ruling R3, the E-58 pattern)
  7  rows at H      every decision-log row is read by `git show H:docs/DECISION_LOG.md`, never at HEAD, and must
                    be RECORDED: the approval (AM-20 item 3: "AM-18 written approval"); the re-check token row
                    ("AM-20 item 3: open"; ruling R5); the arm's configuration row (the config's sha256, the
                    registered difference's sha256 and exactly one "P_T = <HEAD>"); R2 pinned for both arms (AM-18
                    item 7(a): R2's configuration row likewise, and the table row with the table's sha256); R2:
                    the R1 record row with the record's sha256
  8  attempt        n > 1 (ruling R4(b)): attempt n-1's arm_teacher_record by sha256 (diverged: no rerun;
                    complete: no repeat) and the AM-8a report, read at H
  9  environment    PLANTSEG_DATA_ROOT, SEGNEXT_ADE20K_CKPT, TEACHER_WORK_DIR (and PLANTSEG_R2_TABLE for R2) set
                    from the gated values before any Config.fromfile; an inherited value that resolves elsewhere is
                    refused (ruling R2, audit F5)
  10 configuration  check_locked_config(record config) is clean; the arm config's randomness, resume, work_dir,
                    load_from and data roots (and R2's table path) resolve to the gated values; diff_cfg(record,
                    arm) == REG[arm] (ruling R1)
  11 table          the file's sha256 is --r2-table-sha256; a table, never a STOP file; its rules hold and its
                    counts equal the committed strata file's at H; R2: the dataset the config builds repeats each
                    TRAIN index m times and has length sum(m) (AM-18 item 2(b)(i))
  12 R1 record      R2 only: by sha256, arm R1, status complete (r1_diverged / r1_not_complete; AM-18 item 7(a))
  13 arithmetic     AM-19 item 3(b): 40,000 iterations x the reference log's median s/iter against the time left
                    to the original freeze date of the schedule in force; no reference: "no measurement: not cut"
                    (ruling R7, audit F21); R2's reference is R1's own log
  14 provenance     teacher_arm_launch_provenance.json (the records commit, the resolved paths, every row, the
                    arithmetic); written exclusively with --launch, printed by the rehearsal
  15 launch         --launch only: the frozen launch() with the arm configuration
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.launch_teacher_finetune as ltf  # noqa: E402  (first: it records the inherited cuBLAS value)
import scripts.preflight_arms as pa  # noqa: E402
import src.training.sweep_select as ss  # noqa: E402

PreflightError = ltf.PreflightError
LAUNCHER_PATH = Path(__file__).resolve()

# ------------------------------------------------------------------------------------------ values of record
GIT_ROOT = REPO                           # the checkout whose HEAD is P_T (a smoke points it at a scratch repo)
RECORD_CONFIG = ltf.DEFAULT_CONFIG        # the configuration of record
CLASS_MAP = REPO / "configs" / "plantseg_class_map.json"
ARM_CONFIGS = {"R1": "configs/teacher/arms/r1_bema.py", "R2": "configs/teacher/arms/r2_bema_rare.py"}
CUT_PLACE = {"R1": 4, "R2": 1}            # AM-18 item 7(b)
WORK_DIR_RE = re.compile(r"teacher-(r1|r2)_s42_40k_a([1-9][0-9]*)")
MAX_ITERS = 40000
PROVENANCE_FILE = "teacher_arm_launch_provenance.json"
PROVENANCE_FORMAT = "teacher_arm_launch_provenance/1"
TEACHER_RECORD_FORMAT = pa.TEACHER_RECORD_FORMAT       # arm_teacher_record/1 (scripts/arm_teacher_record.py)
STRATA_REL = "reports/strata/train_strata_v1.json"
PIN_T_RE = re.compile(r"P_T\s*[=:]\s*([0-9a-f]{40})(?![0-9a-f])")
SHA256_RE, COMMIT_RE = pa.SHA256_RE, pa.COMMIT_RE
# The four variables the configurations read (ruling R2), with the key that holds each and its gated value.
ENV_KEYS = {"data_root": ltf.DATA_ROOT_ENV, "load_from": ltf.INIT_CKPT_ENV, "work_dir": ltf.WORK_DIR_ENV,
            "table_path": "PLANTSEG_R2_TABLE"}


# --------------------------------------------------------------------------------- the registered difference
ABSENT = type("Absent", (), {"__repr__": lambda self: "ABSENT"})()      # a key one side lacks


def record_classes() -> tuple:
    """The 116 class names in id order, from configs/plantseg_class_map.json (the record config's source)."""
    entries = json.loads(CLASS_MAP.read_text(encoding="utf-8"))["entries"]
    if [e["class_id"] for e in entries] != list(range(116)):
        raise PreflightError("class_map", f"{CLASS_MAP.name}: class ids are not 0..115 in order")
    return tuple(e["name"] for e in entries)


def reg(arm: str) -> list:
    """REG[arm]: the registered difference as (dotted path, record value, arm value), sorted by path; the
    host-dependent values appear by their variable's name (ruling R6, audit F15)."""
    hooks = [{"type": "TeacherInitCompatibilityHook"},
             {"type": "TeacherNMFEvalStreamHook", "seed": 42},
             {"type": "TeacherSelectionRecordHook", "key": "mIoU_full", "disease_key": "mIoU_disease_full"}]
    ema = {"type": "EMAHook", "ema_type": "ExponentialMovingAverage", "momentum": 0.0002, "update_buffers": True,
           "interval": 1, "begin_iter": 0, "strict_load": False}
    r1 = [("custom_hooks", hooks, hooks + [ema]), ("model.backbone.drop_path_rate", 0.1, 0.2)]
    if arm == "R1":
        return r1
    record_ds = {"type": "BaseSegDataset", "data_root": "$" + ENV_KEYS["data_root"],
                 "data_prefix": {"img_path": "images/train", "seg_map_path": "annotations/train"},
                 "pipeline": [{"type": "ThesisTeacherTrainTransform"}, {"type": "PackSegInputs"}],
                 "_scope_": "mmseg", "metainfo": {"classes": record_classes()}, "img_suffix": ".jpg",
                 "seg_map_suffix": ".png", "reduce_zero_label": False}
    r2_ds = {"type": "ClassBalancedDataset", "oversample_thr": 20 / 5367,
             "dataset": dict(record_ds, type="RareClassTrainDataset", table_path="$" + ENV_KEYS["table_path"],
                             pipeline=[{"type": "RareClassCopyPasteTrainTransform"}, {"type": "PackSegInputs"}])}
    return sorted(r1 + [("custom_imports.imports", ["src.training.teacher_components"],
                         ["src.training.teacher_components", "src.training.teacher_arm_components"]),
                        ("train_dataloader.dataset", record_ds, r2_ds)], key=lambda e: e[0])


def _jsonable(v):
    """An exact JSON form: tuples and absent keys are tagged, so two values have equal forms only if they are
    equal with their types (1, 1.0 and True differ; a tuple is not a list)."""
    if v is ABSENT:
        return {"__absent__": True}
    if isinstance(v, tuple):
        return {"__tuple__": [_jsonable(x) for x in v]}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        if not all(isinstance(k, str) for k in v):
            raise TypeError(f"a configuration key that is not a string: {sorted(map(repr, v))}")
        return {k: _jsonable(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, str)) or (isinstance(v, float) and math.isfinite(v)):
        return v
    raise TypeError(f"a configuration value with no canonical form: {v!r}")


def _form(v) -> str:
    return json.dumps(_jsonable(v), sort_keys=True, ensure_ascii=False, allow_nan=False)


def canonical_bytes(entries: list) -> bytes:
    """The canonical JSON of a difference (sorted keys, indent 2, LF), as the configuration row records it."""
    doc = [{"path": p, "record": _jsonable(a), "arm": _jsonable(b)} for p, a, b in entries]
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def reg_sha256(arm: str) -> str:
    return hashlib.sha256(canonical_bytes(reg(arm))).hexdigest()


def _by_variable(v, key=None):
    """A value read from one of the four variables appears as "$<NAME>" (only under its key, only when it equals
    the variable's value in this process)."""
    if isinstance(v, dict):
        return {k: _by_variable(x, k) for k, x in v.items()}
    if isinstance(v, list):
        return [_by_variable(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_by_variable(x) for x in v)
    name = ENV_KEYS.get(key)
    if name and isinstance(v, str) and v and v == os.environ.get(name):
        return "$" + name
    return v


def _diff(a, b, path: str) -> list:
    """A leaf is any pair that is not two dicts of the same component type: such dicts are compared key by key
    (a key on one side only is a leaf with ABSENT on the other), and a component whose type changes is one leaf."""
    if isinstance(a, dict) and isinstance(b, dict) and a.get("type") == b.get("type"):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k, ABSENT), b.get(k, ABSENT), f"{path}.{k}" if path else k)
        return out
    return [] if _form(a) == _form(b) else [(path, a, b)]


def diff_cfg(record_cfg, arm_cfg) -> list:
    """The canonical difference of two loaded configurations (plan §3.0 as ruling R6 corrects it), sorted by path."""
    return sorted(_diff(_by_variable(record_cfg.to_dict()), _by_variable(arm_cfg.to_dict()), ""),
                  key=lambda e: e[0])


# ------------------------------------------------------------------------------------------------ helpers
def _sha256(path) -> str:
    return ltf._sha256(Path(path))


def _row(dl_text: str, dl_id, *, what: str, code: str, need=()) -> str:
    """A RECORDED row of the decision log at H carrying every string in `need` (pa's reader)."""
    try:
        return pa.recorded_row(dl_text, dl_id, what=what, code=code, need=need)
    except pa.GateError as e:
        raise PreflightError(e.code, str(e)) from e


def _p_t(row: str, head: str, dl_id: str, code: str) -> str:
    pins = sorted(set(PIN_T_RE.findall(row)))
    if len(pins) != 1 or pins[0] != head:
        raise PreflightError(code, f"{dl_id} names the pins {pins} as 'P_T = <40 hex>'; it names exactly one, HEAD "
                                   f"{head} (the teacher-arms code pin)")
    return pins[0]


def _read_record(path, sha256: str, *, code: str, what: str) -> tuple[dict, str]:
    """An arm_teacher_record/1 file whose sha256 is the given one."""
    got = _sha256(path)
    if got != sha256:
        raise PreflightError(f"{code}_sha256_mismatch", f"{what} {path}: sha256 {got}, expected {sha256}")
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PreflightError(f"{code}_invalid", f"{what} {path}: not JSON ({type(e).__name__}: {e})") from e
    if not isinstance(doc, dict) or doc.get("format") != TEACHER_RECORD_FORMAT:
        raise PreflightError(f"{code}_invalid", f"{what} {path} is not an {TEACHER_RECORD_FORMAT} record")
    return doc, got


def _resolves_to(value, expected) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).resolve() == Path(expected).resolve()


def _unwrap(ds: dict) -> dict:
    """The dataset a ClassBalancedDataset wraps (R2), else the dataset itself."""
    while isinstance(ds, dict) and ds.get("type") == "ClassBalancedDataset":
        ds = ds.get("dataset")
    return ds if isinstance(ds, dict) else {}


# -------------------------------------------------------------------------------------------------- gates
FLAGS_BOTH = ("config", "config_sha256", "config_dl", "data_root", "init_ckpt", "work_dir", "code_pin",
              "records_commit", "approval_dl", "recheck_dl", "r2_config_dl", "r2_table", "r2_table_sha256",
              "r2_table_dl")
FLAGS_R2 = ("r1_record", "r1_record_sha256", "r1_record_dl")
FLAGS_REPEAT = ("previous_record", "previous_record_sha256", "am8a_report")
PATH_FLAGS = ("config", "data_root", "init_ckpt", "work_dir", "r2_table", "r1_record", "previous_record",
              "am8a_report", "reference_log")
HEX_FLAGS = {"config_sha256": SHA256_RE, "r2_table_sha256": SHA256_RE, "r1_record_sha256": SHA256_RE,
             "previous_record_sha256": SHA256_RE, "reference_log_sha256": SHA256_RE, "code_pin": COMMIT_RE,
             "records_commit": COMMIT_RE}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def gate_flags(args) -> None:
    missing = [_flag(n) for n in FLAGS_BOTH + (FLAGS_R2 if args.arm == "R2" else ()) if getattr(args, n) is None]
    if missing:
        raise PreflightError("flags", f"{args.arm} requires {missing} (explicit values only)")
    if args.arm == "R1":
        extra = [_flag(n) for n in FLAGS_R2 if getattr(args, n) is not None]
        if extra:
            raise PreflightError("flags", f"{extra} belong to R2 (R1's record); R1 refuses them")
    if (args.reference_log is None) != (args.reference_log_sha256 is None):
        raise PreflightError("flags", "--reference-log and --reference-log-sha256 go together")
    for name, rx in HEX_FLAGS.items():
        v = getattr(args, name)
        if v is not None and not rx.fullmatch(v):
            raise PreflightError("flags", f"{_flag(name)} {v!r} is not lowercase hex of the right length")


def gate_paths(args) -> None:
    from src.eval.teacher_diag import Refused, refuse_test_path
    try:
        for name in PATH_FLAGS:
            v = getattr(args, name)
            if v is not None:
                refuse_test_path(v, _flag(name))
    except Refused as e:
        raise PreflightError("test_path", str(e)) from e


def gate_work_dir(args) -> int:
    """Ruling R4(b): the attempt is the work dir's _a<n>; the work dir is absent or empty."""
    p = Path(args.work_dir)
    if not p.is_absolute():
        raise PreflightError("work_dir_relative", f"--work-dir must be absolute: {args.work_dir!r}")
    m = WORK_DIR_RE.fullmatch(p.name)
    if m is None or m.group(1) != args.arm.lower():
        raise PreflightError("work_dir_name", f"--work-dir {p.name!r}: {args.arm}'s work dir is named "
                                              f"teacher-{args.arm.lower()}_s42_40k_a<n>")
    if p.exists() and (not p.is_dir() or any(p.iterdir())):
        raise PreflightError("work_dir_not_empty", f"--work-dir {p} exists and is not an empty directory: every "
                                                   "attempt starts in a fresh work dir")
    n = int(m.group(2))
    given = [_flag(f) for f in FLAGS_REPEAT if getattr(args, f) is not None]
    if n == 1 and given:
        raise PreflightError("flags", f"attempt 1 takes no previous attempt: {given}")
    if n > 1 and len(given) != len(FLAGS_REPEAT):
        code = "repeat_report_missing" if args.am8a_report is None else "previous_record_missing"
        raise PreflightError(code, f"attempt {n} repeats attempt {n - 1}: it requires --previous-record, "
                                   "--previous-record-sha256 and --am8a-report (ruling R4(b))")
    return n


def gate_checkout(args, src, prov: dict) -> dict:
    """Ruling R3: HEAD = --code-pin = P_T; H exists; P_T is an ancestor of H. The arm config is the arm's file."""
    rel = ARM_CONFIGS[args.arm]
    if Path(prov["config_path"]) != (REPO / rel).resolve():
        raise PreflightError("config_not_registered", f"--config {args.config}: {args.arm}'s configuration is {rel}")
    if prov["config_sha256"] != args.config_sha256:
        raise PreflightError("config_sha_mismatch", f"{rel}: sha256 {prov['config_sha256']}, --config-sha256 "
                                                    f"{args.config_sha256}")
    head = src.head()
    if head != args.code_pin:
        raise PreflightError("code_pin_not_head", f"HEAD {head or None} is not --code-pin {args.code_pin} (P_T)")
    h = args.records_commit
    if not src.commit_exists(h):
        raise PreflightError("records_commit_unavailable", f"--records-commit {h} is not a commit of this clone "
                                                           "(fetch it)")
    if not src.is_ancestor(head, h):
        raise PreflightError("pin_not_ancestor_of_records_commit", f"P_T {head} is not an ancestor of the records "
                                                                   f"commit {h} (ruling R3)")
    blobs = {k: src.blob_at("HEAD", r) for k, r in (("config", rel), ("record_config", _rel(RECORD_CONFIG)),
                                                     ("r2_config", ARM_CONFIGS["R2"]))}
    return {"head": head, "records_commit": h, "config_rel": rel, "blobs": blobs}


def _rel(p) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


def gate_rows(args, src, head: str) -> dict:
    """Every row at H (ruling R3), RECORDED, with the values it must carry."""
    data = src.show(args.records_commit, ss.DECISION_LOG_REL)
    if data is None:
        raise PreflightError("decision_log_missing", f"{ss.DECISION_LOG_REL} is not in the records commit")
    dl = data.decode("utf-8", "replace")
    rows = {}

    def keep(name, dl_id, row):
        rows[name] = {"dl": dl_id, "row_sha256": hashlib.sha256(row.encode("utf-8")).hexdigest()}

    keep("approval", args.approval_dl, _row(dl, args.approval_dl, what="--approval-dl (AM-20 item 3)",
                                            code="approval_missing", need=(pa.APPROVAL_MARKER,)))
    keep("recheck", args.recheck_dl, _row(dl, args.recheck_dl, what="--recheck-dl (ruling R5: the Tier F re-check "
                                          "token)", code="approval_recheck_missing", need=(pa.RECHECK_TOKEN,)))
    row = _row(dl, args.config_dl, what=f"--config-dl ({args.arm}'s configuration row, AM-18 item 2)",
               code="config_row_invalid", need=(args.config_sha256, reg_sha256(args.arm)))
    _p_t(row, head, args.config_dl, "config_row_invalid")
    keep("config", args.config_dl, row)
    r2_sha = _sha256(REPO / ARM_CONFIGS["R2"])
    row = _row(dl, args.r2_config_dl, what="--r2-config-dl (R2 pinned before R1, AM-18 item 7(a))",
               code="r2_not_pinned", need=(r2_sha, reg_sha256("R2")))
    _p_t(row, head, args.r2_config_dl, "r2_not_pinned")
    keep("r2_config", args.r2_config_dl, row)
    keep("r2_table", args.r2_table_dl, _row(dl, args.r2_table_dl, what="--r2-table-dl (the R2 table's row)",
                                            code="r2_not_pinned", need=(args.r2_table_sha256,)))
    if args.arm == "R2":
        keep("r1_record", args.r1_record_dl, _row(dl, args.r1_record_dl, what="--r1-record-dl (R1's record)",
                                                  code="r1_record_row_invalid", need=(args.r1_record_sha256,)))
    return {"dl_text": dl, "rows": rows}


def gate_attempt(args, src, n: int) -> dict | None:
    """Ruling R4(b): attempt n > 1 follows an incomplete attempt n - 1 (never a diverged or a complete one) and
    names its committed AM-8a report."""
    if n == 1:
        return None
    doc, sha = _read_record(args.previous_record, args.previous_record_sha256, code="previous_record",
                            what="--previous-record")
    wd = doc.get("work_dir") if isinstance(doc.get("work_dir"), dict) else {}
    want = f"teacher-{args.arm.lower()}_s42_40k_a{n - 1}"
    if doc.get("arm") != args.arm or wd.get("name") != want or wd.get("attempt") != n - 1:
        raise PreflightError("previous_record_invalid", f"--previous-record: arm {doc.get('arm')!r}, work dir "
                                                        f"{wd.get('name')!r}; attempt {n} follows {want}")
    status = doc.get("status")
    if status == "diverged":
        raise PreflightError("arm_rerun_after_divergence", f"{want} is recorded as diverged: AM-18 item 2's arm is "
                                                           "reported and not rerun (ruling R4(b))")
    if status == "complete":
        raise PreflightError("arm_already_complete", f"{want} completed: {args.arm} is never repeated")
    if status != "incomplete":
        raise PreflightError("previous_record_invalid", f"--previous-record status {status!r}")
    report = src.show(args.records_commit, args.am8a_report)
    if report is None:
        raise PreflightError("repeat_report_missing", f"--am8a-report {args.am8a_report} is not in the records "
                                                       "commit (AM-8a; AM-19 item 3(a))")
    return {"previous_record": {"path": str(args.previous_record), "sha256": sha, "status": status, "work_dir": want},
            "am8a_report": {"path": args.am8a_report, "sha256": hashlib.sha256(report).hexdigest()}}


def gate_environment(args, prov: dict) -> dict:
    """Ruling R2: the variables the configurations read take the gated values before any Config.fromfile."""
    values = {ltf.DATA_ROOT_ENV: prov["plantseg_root"], ltf.INIT_CKPT_ENV: prov["ade20k_init_checkpoint"]["path"],
              ltf.WORK_DIR_ENV: prov["work_dir"]}
    if args.arm == "R2":
        values[ENV_KEYS["table_path"]] = str(Path(args.r2_table).resolve())
    out = {}
    for name, value in values.items():
        inherited = os.environ.get(name)
        if inherited and Path(inherited).resolve() != Path(value).resolve():
            raise PreflightError("env_conflict", f"{name} is inherited as {inherited!r}, but the gated value is "
                                                 f"{value!r}: unset it, or pass the flag that matches")
        os.environ[name] = value
        out[name] = {"value": value, "inherited": inherited}
    return out


def arm_config_problems(arm_cfg, prov: dict, args) -> list:
    """load_launch_config's checks other than the locks, on the arm configuration (ruling R1); the train dataset
    is the one a ClassBalancedDataset wraps, and R2's table path is gated too."""
    problems = []
    if arm_cfg.get("randomness") != dict(seed=42, deterministic=True):
        problems.append(f"randomness={arm_cfg.get('randomness')!r} (contract B6: seed 42, deterministic)")
    if arm_cfg.get("resume") is not False:
        problems.append(f"resume={arm_cfg.get('resume')!r} (official runs never resume)")
    wd = arm_cfg.get("work_dir")
    if not (_resolves_to(wd, prov["work_dir"]) and Path(wd).is_absolute()):
        problems.append(f"work_dir={wd!r}")
    if not _resolves_to(arm_cfg.get("load_from"), prov["ade20k_init_checkpoint"]["path"]):
        problems.append(f"load_from={arm_cfg.get('load_from')!r}")
    for loader in ("train_dataloader", "val_dataloader"):
        ds = _unwrap(dict(arm_cfg.get(loader, {})).get("dataset"))
        if not _resolves_to(ds.get("data_root"), prov["plantseg_root"]):
            problems.append(f"{loader} dataset data_root={ds.get('data_root')!r}")
    tp = _unwrap(dict(arm_cfg.get("train_dataloader", {})).get("dataset")).get("table_path")
    if args.arm == "R2" and not _resolves_to(tp, args.r2_table):
        problems.append(f"train dataset table_path={tp!r} (--r2-table {args.r2_table})")
    if args.arm == "R1" and tp is not None:
        problems.append(f"R1's train dataset has a table_path {tp!r}")
    return problems


def gate_configuration(args, prov: dict):
    """Ruling R1: the record's locks on the record config, the launch checks on the arm config, and
    diff_cfg(record, arm) == REG[arm]; anything else is launch_config_mismatch."""
    from mmengine.config import Config
    record_cfg = Config.fromfile(str(RECORD_CONFIG))
    arm_cfg = Config.fromfile(prov["config_path"])
    problems = [f"record config: {p}" for p in ltf.check_locked_config(record_cfg)]
    problems += arm_config_problems(arm_cfg, prov, args)
    got = diff_cfg(record_cfg, arm_cfg)
    if canonical_bytes(got) != canonical_bytes(reg(args.arm)):
        want = {p: (_form(a), _form(b)) for p, a, b in reg(args.arm)}
        seen = {p: (_form(a), _form(b)) for p, a, b in got}
        bad = sorted(p for p in set(want) | set(seen) if want.get(p) != seen.get(p))
        problems.append(f"diff_cfg(record, {args.arm}) != REG[{args.arm}] at {bad}")
    if problems:
        raise PreflightError("launch_config_mismatch", "the configuration is not the registered difference: "
                                                       + "; ".join(problems))
    return arm_cfg, {"entries": [p for p, _, _ in got], "sha256": hashlib.sha256(canonical_bytes(got)).hexdigest()}


def gate_table(args, src, arm_cfg) -> dict:
    """The pinned R2 table (both arms; AM-18 item 7(a)): its bytes, its rules, the committed strata counts; for R2
    the repeat counts of the dataset its config builds (AM-18 item 2(b)(i))."""
    from src.training import teacher_arm_components as tac
    data = Path(args.r2_table).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != args.r2_table_sha256:
        raise PreflightError("r2_table_sha256_mismatch", f"--r2-table {args.r2_table}: sha256 {sha}, the row's "
                                                         f"{args.r2_table_sha256}")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PreflightError("r2_table_invalid", f"--r2-table: not JSON ({e})") from e
    if isinstance(doc, dict) and doc.get("format") == tac.STOP_FORMAT:
        raise PreflightError("r2_table_stop", "--r2-table is the builder's STOP file: R2 does not launch while it "
                                              "stands, and R1 waits for R2's table (AM-18 items 2(b), 7(a))")
    try:
        table = tac.validate_table(doc)
    except tac.R2TableError as e:
        raise PreflightError("r2_table_invalid", f"--r2-table: {e}") from e
    strata_b = src.show(args.records_commit, STRATA_REL)
    if strata_b is None:
        raise PreflightError("r2_table_strata_mismatch", f"{STRATA_REL} is not in the records commit")
    try:
        strata = json.loads(strata_b.decode("utf-8"))
        n_s = {int(r["class_id"]): int(r["images"]) for r in strata["per_class"]}
        split_s, images_s = strata["split_list_sha256"], strata["n_images"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
        raise PreflightError("r2_table_strata_mismatch", f"{STRATA_REL} at the records commit is unreadable "
                                                         f"({type(e).__name__}: {e})") from e
    bad = sorted(c for c in tac.DISEASE_IDS if n_s.get(c) != table["n_c"][c])
    if (doc["strata"]["path"] != STRATA_REL or doc["strata"]["sha256"] != hashlib.sha256(strata_b).hexdigest()
            or doc["split_list_sha256"] != split_s or table["images"] != images_s or bad):
        raise PreflightError("r2_table_strata_mismatch", f"the table disagrees with the committed {STRATA_REL}: "
                                                         f"classes {bad[:10]}, strata sha256, split list or count")
    out = {"path": str(Path(args.r2_table).resolve()), "sha256": sha, "images": table["images"],
           "repeated": table["repeated"], "rare_classes": {str(c): m for c, m in sorted(table["rare"].items())}}
    if args.arm == "R2":
        out["repeat_check"] = repeat_check(arm_cfg, table, sha)
    return out


def repeat_check(arm_cfg, table: dict, sha: str) -> dict:
    """The dataset the R2 config builds: ClassBalancedDataset over RareClassTrainDataset, each TRAIN index repeated
    exactly its table m, length sum(m), reading the table the launcher checked."""
    from mmengine.dataset import ClassBalancedDataset
    from mmengine.registry import DATASETS, DefaultScope
    from src.eval.teacher_diag import Refused
    from src.training import teacher_arm_components as tac
    try:
        with DefaultScope.overwrite_default_scope(arm_cfg.get("default_scope", "mmseg")):
            ds = DATASETS.build(copy.deepcopy(arm_cfg.train_dataloader.dataset))
    except (tac.R2TableError, Refused, OSError) as e:
        raise PreflightError("r2_repeat_counts", f"the R2 train dataset cannot be built: {type(e).__name__}: {e}") \
            from e
    if not (isinstance(ds, ClassBalancedDataset) and isinstance(ds.dataset, tac.RareClassTrainDataset)):
        raise PreflightError("r2_repeat_counts", f"the train dataset is {type(ds).__name__}, not "
                                                 "ClassBalancedDataset(RareClassTrainDataset)")
    if ds.dataset.table_sha256 != sha:
        raise PreflightError("r2_repeat_counts", f"the dataset read a table with sha256 {ds.dataset.table_sha256}")
    counts = [0] * len(ds.dataset)
    for i in ds.repeat_indices:
        counts[i] += 1
    wrong = []
    for i, k in enumerate(counts):
        stem = Path(ds.dataset.get_data_info(i)["img_path"]).name[:-len(tac.IMG_SUFFIX)]
        if k != table["m_by_stem"][stem]:
            wrong.append((stem, k, table["m_by_stem"][stem]))
    if wrong or len(ds) != table["repeated"]:
        raise PreflightError("r2_repeat_counts", f"ClassBalancedDataset repeats {len(wrong)} TRAIN index(es) other "
                                                 f"than the table's m (e.g. {wrong[:3]}) or has length {len(ds)} != "
                                                 f"sum(m) {table['repeated']}")
    return {"indices": len(counts), "length": len(ds), "per_index_equal": True}


def gate_r1_record(args) -> dict | None:
    """AM-18 item 7(a): R2 launches once R1 completed its 40,000 iterations without divergence."""
    if args.arm != "R2":
        return None
    doc, sha = _read_record(args.r1_record, args.r1_record_sha256, code="r1_record", what="--r1-record")
    if doc.get("arm") != "R1":
        raise PreflightError("r1_record_invalid", f"--r1-record is the record of {doc.get('arm')!r}, not R1")
    if doc.get("status") == "diverged":
        raise PreflightError("r1_diverged", "R1 is recorded as diverged: R2 is not launched (AM-18 item 2(b); a DL "
                                            "row records R2 as cut under item 7(a))")
    if doc.get("status") != "complete":
        raise PreflightError("r1_not_complete", f"R1's record has status {doc.get('status')!r}: R2 launches once R1 "
                                                "completed its 40,000 iterations (AM-18 item 7(a))")
    logs = doc.get("logs") if isinstance(doc.get("logs"), dict) else {}
    return {"path": str(args.r1_record), "sha256": sha, "status": "complete",
            "scalars_sha256": logs.get("scalars_sha256")}


def reference_figure(path) -> dict:
    """An mmengine vis_data/scalars.json: the median of the "time" of every training row (a row with an integer
    "iter"; LoggerHook writes one every 50 iterations, each the mean seconds per iteration of its window)."""
    vals = []
    try:
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            t = row.get("time") if isinstance(row, dict) else None
            if isinstance(row, dict) and isinstance(row.get("iter"), int) and not isinstance(row.get("iter"), bool) \
                    and isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0:
                vals.append(float(t))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PreflightError("reference_unusable", f"--reference-log {path}: unreadable ({type(e).__name__}: {e})") \
            from e
    if not vals:
        raise PreflightError("reference_unusable", f"--reference-log {path}: no training row with a finite time")
    return {"source": "scalars", "path": str(path), "rows": len(vals), "median": float(statistics.median(vals)),
            "field": "time of every row with an integer iter (vis_data/scalars.json)"}


def gate_arithmetic(args, src, dl_text: str, r1: dict | None) -> dict:
    """AM-19 item 3(b) for the arm (ruling R7): pa's arithmetic with the reference log's figure, or none."""
    raw = src.show(args.records_commit, ss.SCHEDULE_REL)
    try:
        sched = json.loads(raw.decode("utf-8"))
        in_force, entry = sched["in_force"], sched["decision_log_entry"]
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise PreflightError("schedule_format", f"{ss.SCHEDULE_REL} at the records commit is unreadable "
                                                f"({type(e).__name__})") from e
    if in_force not in pa.FREEZE_DATES or pa.dl_row(dl_text, entry) is None:
        raise PreflightError("schedule_format", f"{ss.SCHEDULE_REL}: schedule {in_force!r}, entry {entry!r}")
    figures = []
    if args.reference_log is not None:
        sha = _sha256(args.reference_log)
        if sha != args.reference_log_sha256:
            raise PreflightError("reference_log_sha256_mismatch", f"--reference-log: sha256 {sha}, expected "
                                                                  f"{args.reference_log_sha256}")
        if r1 is not None and sha != r1.get("scalars_sha256"):
            raise PreflightError("reference_log_not_r1", "R2's reference is R1's own scalars log (audit F21): its "
                                                         f"sha256 {sha} is not R1's record's {r1.get('scalars_sha256')}")
        figures.append(dict(reference_figure(args.reference_log), sha256=sha))
    ar = pa.cut_arithmetic(ss, MAX_ITERS, figures, in_force, time.time())
    print(f"AM-19 item 3(b) arithmetic for {args.arm}: {ar['text']}", flush=True)
    if ar["cut"]:
        raise PreflightError("arm_cut_by_arithmetic", ar["text"])
    return ar


def provenance(args, *, n, prov, checkout, rows, attempt, env, config, table, r1, arithmetic, governed) -> dict:
    from src.training import teacher_arm_components as tac
    return {"format": PROVENANCE_FORMAT, "arm": args.arm, "attempt": n, "work_dir": prov["work_dir"],
            "records_commit": checkout["records_commit"], "code_pin": checkout["head"],
            "checkout_git_head": checkout["head"], "rows": rows,
            "config": {"path_rel": checkout["config_rel"], "sha256": args.config_sha256,
                       "git_blob": checkout["blobs"]["config"]},
            "record_config": {"path_rel": _rel(RECORD_CONFIG), "sha256": _sha256(RECORD_CONFIG),
                              "git_blob": checkout["blobs"]["record_config"]},
            "registered_difference": dict(config, reg_sha256=reg_sha256(args.arm)),
            "resolved_paths": env, "r2_table": table, "r1_record": r1, "previous_attempt": attempt,
            "arithmetic": arithmetic, "cut_place": CUT_PLACE[args.arm], "governed_paths": governed,
            "modules": {"launcher": {"path": str(LAUNCHER_PATH), "sha256": _sha256(LAUNCHER_PATH)},
                        "frozen_launcher": {"path": str(ltf.LAUNCHER_PATH), "sha256": ltf.LAUNCHER_SHA256},
                        "teacher_arm_components": tac.ARM_COMPONENTS_PROVENANCE},
            "launch": bool(args.launch), "preflight": prov,
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def run(args) -> int:
    """Gates 2 to 15 (see the module docstring); raises PreflightError on the first failure."""
    gate_flags(args)
    gate_paths(args)
    n = gate_work_dir(args)
    ltf.check_inherited_cublas()
    governed = ltf.check_governed_clean()
    prov = ltf.preflight(args, record_gpu_names=not args.launch)
    src = ss.HeadSource(GIT_ROOT)
    checkout = gate_checkout(args, src, prov)
    rows = gate_rows(args, src, checkout["head"])
    attempt = gate_attempt(args, src, n)
    env = gate_environment(args, prov)
    arm_cfg, config = gate_configuration(args, prov)
    table = gate_table(args, src, arm_cfg)
    r1 = gate_r1_record(args)
    arithmetic = gate_arithmetic(args, src, rows["dl_text"], r1)
    doc = provenance(args, n=n, prov=prov, checkout=checkout, rows=rows["rows"], attempt=attempt, env=env,
                     config=config, table=table, r1=r1, arithmetic=arithmetic, governed=governed)
    out = Path(prov["work_dir"]) / PROVENANCE_FILE
    print(f"cut-order place {CUT_PLACE[args.arm]} (AM-18 item 7(b)); attempt {n}; records commit "
          f"{checkout['records_commit']}; P_T {checkout['head']}", flush=True)
    if args.launch:
        from src.eval.teacher_diag import write_json_exclusive
        write_json_exclusive(out, doc)
        print(f"[arm launch record] {out}", flush=True)
        return ltf.launch(arm_cfg, prov, governed)
    if out.exists():
        raise PreflightError("work_dir_not_empty", f"{out} exists: the launch writes it once")
    print(json.dumps(doc, indent=2))
    if prov["cuda"] is None:
        print("\nPREFLIGHT PARTIAL — every non-GPU gate passed, but the CUDA check was SKIPPED (--skip-cuda-probe). "
              "This does NOT clear the arm for launch.")
    else:
        print(f"\nPREFLIGHT PASS — every gate of {args.arm}'s launch passed (rehearsal).")
    print("Training is NOT started without --launch.")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R1/R2 teacher-arm launcher (L-TEACHER-ARMS C3); trains only with "
                                            "--launch.", allow_abbrev=False)
    p.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    p.add_argument("--show-diff", action="store_true", help="print REG[arm]'s canonical JSON and sha256; exit")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    for name in FLAGS_BOTH + FLAGS_R2 + FLAGS_REPEAT + ("reference_log", "reference_log_sha256"):
        p.add_argument(_flag(name), dest=name, default=None)
    p.add_argument("--create-work-dir", action="store_true")
    p.add_argument("--skip-cuda-probe", action="store_true")
    p.add_argument("--launch", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.show_diff:
        data = canonical_bytes(reg(args.arm))
        sys.stdout.write(data.decode("utf-8"))
        print(f"REG[{args.arm}] sha256 {hashlib.sha256(data).hexdigest()} ({len(data)} bytes)")
        return 0
    if not args.real_run or not args.confirm_real_run:
        print("REFUSING: a teacher arm requires --real-run --confirm-real-run. Nothing was read, hashed, created or "
              "imported.", file=sys.stderr)
        return 2
    if args.launch and args.skip_cuda_probe:
        print("REFUSING: --launch cannot skip the CUDA gate (--skip-cuda-probe).", file=sys.stderr)
        return 2
    try:
        return run(args)
    except PreflightError as e:
        print(f"PREFLIGHT FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
