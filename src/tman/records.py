"""Score-blind readers of the committed records, the run directories and the PTQ records (lane TMAN, B-1, C1; plan
decision (vi)).

No check reads a score. Every reader projects at parse: it keeps an allowlist of identity, time, hash, status and
verdict keys and never touches another key's value (`project` reads a key only after it is allowed). No reader of
src/training/sweep_select.py that reads a score is reused (`_run_end_ending` reads best_val_miou_all_class,
sweep_select.py:1242-1245); LaunchLog reads launch lines only, whose `value` is a grid value, not a score.

Committed records go through sweep_select's two-source accessor (sweep_select.py:460-623): HeadSource (the working
tree at HEAD of the repository at a root, byte-equal to its blob) or RecordsSource (reports/ and the decision log
from a records folder verified by blob id against a records commit; code and configs at HEAD of the checkout, as
sweep_select splits them); the decision log is read at the records ref, as sweep_select reads it (:810). A
path containing "test" is refused as given, before any filesystem call, then as resolved. Run directories are read
only under an explicit --runs folder. A ptq_record/1 is read only through lane LQP's reader (A2).
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

from src.training import sweep_select as ss

from . import schema as S

# --------------------------------------------------------------------------------------------- projection
# An allowlist is {key: True | allowlist | [allowlist]}: True keeps the value, a dict recurses into an object, a
# one-element list recurses into each element of a list. A key absent from the allowlist is never read.
JSON_OBJECT = dict          # the smokes' sentinel replaces it with a mapping that raises on a score key


def project(doc, allow):
    if allow is True:
        return doc
    if isinstance(allow, list):
        return [project(x, allow[0]) for x in doc] if isinstance(doc, list) else None
    if not hasattr(doc, "keys"):
        return None
    out = {}
    for k in list(doc.keys()):
        if k in allow:
            out[k] = project(doc[k], allow[k])
    return out


def strict_json(data: bytes, what: str, *, code: str):
    """schema.strict_json with the module's object type (the sentinel in the smokes)."""
    return S.strict_json(data, what, code=code, object_type=JSON_OBJECT)


REF = {"path": True, "sha256": True}

# ------------------------------------------------------------------------- allowlists (field names at O, 2b3b556)
# KD trainer (src/training/train_distill.py:1305-1311 run_end; :1100-1110 run_abort; the train rows' iter_seconds;
# run_meta :1009-1049 with launch_fields :703-731)
KD_ROWS = {"run_end": {"event": True, "iter": True, "best_ckpt": True, "checks_passed": True,
                       "wall_clock_start": True, "wall_clock_end": True},
           "run_abort": {"event": True, "iter": True, "rule": True, "cause": True, "wall_clock": True,
                         "input_finite": True, "teacher_finite": True},
           "train": {"event": True, "iter": True, "wall_clock": True, "iter_seconds": True},
           "val": {"event": True, "iter": True, "wall_clock": True}}
KD_RUN_META = {"event": True, "stage": True, "mode": True, "seed": True, "max_iters": True, "wall_clock": True,
               "git_head": True, "image_digest": True, "arm": True, "descriptive": True, "selection_sha256": True,
               "records_commit": True}
# E1 trainer (src/training/train_e1.py:353-366 run_meta; :493-500 val)
E1_ROWS = {"run_meta": {"event": True, "wall_clock": True, "mode": True, "seed": True, "git_head": True,
                        "image_digest": True, "max_iters": True},
           "val": {"event": True, "iter": True, "wall_clock": True},
           "train": {"event": True, "iter": True, "wall_clock": True}}
# QAT trainer (src/quant/qat.py:885-930 run_meta; :943 state_nonfinite; :1070-1079 epoch_end; :1088-1094 stop/abort)
QAT_ROWS = {"run_meta": {"event": True, "wall_clock": True, "mode": True, "stage": True, "source_stage": True,
                         "seed": True, "run_id": True, "clip_norm": True, "clip_source": True,
                         "source_checkpoint_sha256": True, "image_digest": True, "git_head": True},
            "epoch_end": {"event": True, "wall_clock": True, "epoch": True, "step": True, "checkpoint": True,
                          "checkpoint_sha256": True, "checkpoint_bytes": True, "state_sha256": True,
                          "state_finite": True, "nonfinite_since_step": True, "state_nonfinite": True,
                          "run_complete": True},
            "state_nonfinite": {"event": True, "wall_clock": True, "epoch": True, "step": True,
                                "nonfinite_since_step": True},
            "run_stop": {"event": True, "wall_clock": True, "epoch": True, "step": True, "code": True},
            "run_abort": {"event": True, "wall_clock": True, "epoch": True, "step": True, "cause": True,
                          "exception_type": True}}
# the flags an epoch checkpoint stores (qat.py:712-742; qat_artifacts.py:52 STATE_FLAG_KEYS)
QAT_CHECKPOINT = {"stage": True, "epoch": True, "step": True, "seed": True, "run_id": True, "state_sha256": True,
                  "source_checkpoint_sha256": True, "state_finite": True, "state_nonfinite": True,
                  "nonfinite_since_step": True}
QAT_STATE_FLAG_KEYS = ("state_finite", "state_nonfinite", "nonfinite_since_step")
# qat_selection/1 (src/quant/qat_select.py:479-494) and qat_clip_selection/1 (qat_select.py clip_selection)
QAT_SELECTION_FORMAT, CLIP_SELECTION_FORMAT = "qat_selection/1", "qat_clip_selection/1"   # qat_select.py:38; qat.py:102
QAT_SELECTION = {"format": True, "rules_sha256": True, "stage": True, "run_id": True, "seed": True,
                 "clip_norm": True, "telemetry_sha256": True, "git_head": True, "host_label": True,
                 "cpu_model": True, "excluded_epochs": True, "smoke_inputs": True,
                 "winner": {"epoch": True, "checkpoint": True, "checkpoint_sha256": True, "run_id": True,
                            "artifact_of_record": REF, "provenance": REF, "summary": REF}}
_CLIP_SIDE = {"clip_norm": True, "run_id": True, "telemetry_sha256": True, "qat_selection": REF, "rejected": True,
              "epoch": True, "retained": True}
CLIP_SELECTION = {"format": True, "rules_sha256": True, "stage": True, "seed": True, "smoke_inputs": True,
                  "winner": _CLIP_SIDE, "loser": _CLIP_SIDE}
# arm records (scripts/arm_teacher_record.py:219-223 selection, :236-258 the record; scripts/arm_go_rule.py:245-262;
# scripts/run_ptq_arm.py:206-226; scripts/launch_teacher_arm.py:623-639)
ARM_TEACHER_RECORD = {"format": True, "arm": True, "status": True, "first_nonfinite": {"iteration": True},  # DL-24 #112
                      "generated_utc": True,
                      "work_dir": {"name": True, "attempt": True},
                      "selection": {"status": True, "selected_iteration": True, "recorded_checkpoint_sha256": True,
                                    "teacher_selection_json_sha256": True,
                                    "checkpoint": {"name": True, "sha256": True, "bytes": True}},
                      "launch": {"teacher_arm_launch_provenance_sha256": True},
                      "logs": {"scalars_sha256": True}}
GO_RULE_RECORD = {"format": True, "arm": True, "verdict": True, "generated_utc": True,
                  "checkpoint": {"name": True, "sha256": True, "bytes": True}, "teacher_record": REF}
E7_ARM_RECORD = {"format": True, "arm": True, "run_stage": True, "e7_stage": True, "generated_utc": True,
                 "registry": {"file": True, "sha256": True},
                 "e3_run": {"run_id": True, "run_meta_sha256": True, "best_json_sha256": True},
                 "calibration": {"list_file_sha256": True, "checksum_sha256": True, "seed": True, "role": True},
                 "ptq_run_role": True, "writer": {"commit": True},
                 "artifacts": {"torchscript": REF, "state_dict": REF, "x86_copy": REF, "run_meta": REF}}
LAUNCH_PROVENANCE = {"format": True, "arm": True, "attempt": True, "work_dir": True, "launch": True,
                     "recorded_utc": True, "code_pin": True, "records_commit": True}
# PTQ run provenance (src/quant/ptq.py:929-979, code and environment :724-745; src/quant/runner.py:176-200)
PTQ_PROVENANCE = {"schema": True, "stage": True, "source_stage": True, "source_checkpoint_sha256": True,
                  "quantization_method": True, "qconfig": {"fingerprint": True}, "converted_artifact": True,
                  "converted_artifact_sha256": True, "state_dict_artifact": REF, "x86_latency_copy": REF,
                  "calibration": {"list_file_sha256": True, "checksum_sha256": True, "seed": True, "role": True,
                                  "split": True, "data_root": True},
                  "ptq_run_role": True, "code": {"commit": True, "clean_at_commit": True},
                  "environment": {"image_digest": True}, "created_utc": True}
# a QAT epoch's conversion provenance, eNN_run_meta.json (src/quant/qat_artifacts.py:507-540 with
# src/quant/runner.py:176-203): its run and epoch, its artifacts and its time
QAT_PROVENANCE = {"converted_artifact_sha256": True, "state_dict_artifact": REF,
                  "training": {"run_id": True, "epoch": True}, "created_utc": True}
# the evaluator's runtime record (src/eval/eval_runtime.py:273-294): the execution-context keys of TB1 only
EVAL_RUNTIME = {"model_device": True, "input_devices": True, "batch_size": True, "num_workers": True,
                "determinism_policy_applied": True, "torch_num_threads": True, "image_digest": True}
# a sweep decision record (sweep_select.decision_record_doc, :2471-2484): its decision date and each grid value's
# status; `value` is the grid value (a launch value, as the launch log's), never a score -- the one allowlist that
# keeps a value(s) key; reason, directories and on_course are dropped
DECISION_RECORD = {"decision_date": True, "values": [{"value": True, "status": True}]}
# the sweep selection files (lambda_selection/1, alpha_selection/1): identity only
SELECTION = {"format": True,
             "winner": {"lambda": True, "alpha": True, "run_id": True, "ckpt_sha256": True},
             "candidates": [{"lambda": True, "alpha": True, "run_id": True, "ckpt_sha256": True,
                             "checkpoint_present": True}]}
# ptq_record/1 (lane LQP, src/quant/ptq_record.py at e37e339 and 62b599a): A2's projection, nothing else -- the
# record's parent block also holds the parent's VAL score, which is never read
PTQ_ARTIFACT_KEYS = ("state_dict", "torchscript", "x86_copy")
PTQ_RECORD = {"stage": True, "seed": True, "role": True, "descriptive": True,
              "parent": {"checkpoint_sha256": True}, "calibration": {"checksum_sha256": True},
              "artifacts": {k: {"sha256": True} for k in PTQ_ARTIFACT_KEYS}}
PTQ_RECORD_SPEC = {"stage": "str", "seed": "int", "role": ("enum", S.PTQ_RECORD_ROLES), "descriptive": "bool",
                   "parent": ("obj", {"checkpoint_sha256": "sha256"}),
                   "calibration": ("obj", {"checksum_sha256": "sha256"}),
                   "artifacts": ("obj", {k: ("obj", {"sha256": "sha256"}) for k in PTQ_ARTIFACT_KEYS})}
PTQ_RECORD_MODULE = "src.quant.ptq_record"      # lane LQP; absent at O, imported only when a record is read


# ------------------------------------------------------------------------------------------- the decision log
DL_ROW_RE = re.compile(r"^\|\s*(DL-\d+[a-z]?)\s*\|\s*([^|]*)\|", re.M)


class DecisionLog:
    """docs/DECISION_LOG.md's table rows `| DL-<n> | <Status> | ... |` (one line each)."""

    def __init__(self, text: str):
        self.text = text
        self.rows: dict[str, tuple[str, str]] = {}
        for line in text.splitlines():
            m = DL_ROW_RE.match(line)
            if m:
                self.rows.setdefault(m.group(1), (m.group(2).strip(), line))

    def has(self, dl: str) -> bool:
        return dl in self.rows

    def status(self, dl: str) -> str | None:
        return self.rows[dl][0] if dl in self.rows else None

    def row_text(self, dl: str) -> str | None:
        return self.rows[dl][1] if dl in self.rows else None

    def recorded_rows_with(self, sha256: str) -> list[str]:
        """The rows whose Status starts with RECORDED or IMPLEMENTED and whose text holds `sha256` (decision (iii):
        preflight_distill's substring rule, scripts/preflight_distill.py:277, narrowed to rows)."""
        return [dl for dl, (st, line) in self.rows.items()
                if st.startswith(("RECORDED", "IMPLEMENTED")) and sha256 in line]


# ------------------------------------------------------------------------------------------- committed records
class Records:
    """The committed records, through sweep_select's accessor. Each refusal is a TmanRefused."""

    def __init__(self, src: ss.HeadSource):
        self.src = src
        self._dl: DecisionLog | None = None

    @classmethod
    def at_head(cls, root=None) -> "Records":
        return cls(ss.HeadSource(root))

    @classmethod
    def at_records_commit(cls, folder, commit: str, root=None) -> "Records":
        return cls(ss.RecordsSource(folder, commit, root))

    @property
    def ref(self) -> str:
        return self.src.records_ref

    def base(self, rel: str) -> Path:
        """Where `rel` is read: the records folder for a record at the gate (RecordsSource), else the root."""
        folder = getattr(self.src, "folder", None)
        return folder if folder is not None and self.src.is_record(rel) else self.src.root

    def read(self, rel: str, *, missing: str = "record_uncommitted") -> bytes:
        S.refuse_test_path(rel, self.base(rel))
        try:
            _, data = self.src.read(rel, missing=missing, uncommitted="record_uncommitted")
        except ss.SelectionRefused as e:
            code = "test_path" if e.code == "test_path" else ("record_uncommitted" if e.code in (
                "record_uncommitted", "records_mismatch") else missing)
            raise S.TmanRefused(code, str(e)) from None
        return data

    def json(self, rel: str, allow, *, code: str, missing: str = "record_uncommitted"):
        data = self.read(rel, missing=missing)
        return project(strict_json(data, rel, code=code), allow), S.sha256_bytes(data)

    def decision_log(self) -> DecisionLog:
        if self._dl is None:
            try:
                data = self.src.read_at_head(S.DECISION_LOG_REL, missing="dl_row_missing")
            except ss.SelectionRefused as e:
                raise S.TmanRefused("dl_row_missing" if e.code == "dl_row_missing" else "record_uncommitted",
                                    str(e)) from None
            self._dl = DecisionLog(data.decode("utf-8", "replace"))
        return self._dl

    def has(self, rel: str) -> bool:
        S.refuse_test_path(rel, self.base(rel))
        return self.src.has(rel)

    def adding_commits(self, rel: str) -> list:
        return ss._adding_commits(self.src, rel)

    def written_once_error(self, rel: str) -> str | None:
        return ss.written_once_error(self.src, rel)

    def deleted_record_error(self, rel: str) -> str | None:
        return ss.deleted_record_error(self.src, rel)

    def history(self, rel: str) -> list:
        return ss._path_history(self.src, rel)

    def show(self, commit: str, rel: str) -> bytes | None:
        S.refuse_test_path(rel, self.base(rel))
        return self.src.show(commit, rel)

    def commit_time(self, commit: str) -> int | None:
        """A commit's committer time (Unix), or None when it is no commit."""
        out = self.src._out("show", "-s", "--format=%ct", f"{commit}^{{commit}}") if isinstance(commit, str) else None
        return int(out) if out and out.isdigit() else None

    def held_by(self, rel: str, until: int) -> set:
        """The sha256 of each content `rel` held in a commit of the records' history whose committer time meets
        `until` (sweep_select.meets): what was committed by then (decision (xvi): results known at that time)."""
        S.refuse_test_path(rel, self.base(rel))
        out = self.src._out("log", "--full-history", "--no-renames", "--format=%H %ct", self.ref, "--", rel)
        held = set()
        for c, t in (ln.split() for ln in (out or "").splitlines() if len(ln.split()) == 2):
            data = self.src.show(c, rel) if ss.meets(int(t), until) else None
            if data is not None:
                held.add(S.sha256_bytes(data))
        return held

    def launch_log(self, *, code: str) -> ss.LaunchLog:
        """sweep_select's LaunchLog (launch lines only). Absent or uncommitted: record_uncommitted; any other
        refusal of its reader: `code`, the caller's gate."""
        try:
            return ss.load_launch_log(self.src)
        except ss.SelectionRefused as e:
            raise S.TmanRefused("record_uncommitted" if e.code in ("launch_log_missing", "launch_log_uncommitted",
                                                                   "records_mismatch") else code,
                                f"[{e.code}] {e}") from None

    def at_commit(self, commit: str) -> "Records":
        """The records as committed by `commit` (CommitSource; AM-22 item 1(a)'s "by the end of X", audit M3)."""
        return Records(CommitSource(self.src, commit))


class CommitSource(ss.HeadSource):
    """The records committed by a commit (audit M3): a records file (sweep_select.RecordsSource.is_record: reports/
    and the decision log) is its blob at `commit`, and "exists" means there; code and configs are read at HEAD of the
    checkout, as HeadSource reads them (PL-28's split). The records' history (held_by, entered_day, the adding
    commits) ends at `commit`."""
    kind = "commit"
    is_record = staticmethod(ss.RecordsSource.is_record)

    def __init__(self, src: ss.HeadSource, commit: str):
        super().__init__(src.root)
        self.commit = commit

    @property
    def records_ref(self) -> str:
        return self.commit

    def has(self, rel: str) -> bool:
        return self.blob_at(self.commit, rel) is not None if self.is_record(rel) else super().has(rel)

    def _record(self, rel: str, *, missing: str) -> bytes:
        ss.refuse_test_path(self.root / rel)
        data = self.show(self.commit, rel)
        if data is None:
            raise ss.SelectionRefused(missing, f"{rel} does not exist at {self.commit[:12]}")
        return data

    def read(self, path, *, missing: str, uncommitted: str, record: bool = False) -> tuple[str, bytes]:
        rel = self.rel(path)
        if rel is not None and (record or self.is_record(rel)):
            return rel, self._record(rel, missing=missing)
        return super().read(path, missing=missing, uncommitted=uncommitted)

    def read_at_head(self, rel: str, *, missing: str) -> bytes:
        if self.is_record(rel):
            return self._record(rel, missing=missing)
        return super().read_at_head(rel, missing=missing)


# ------------------------------------------------------------------------------------------- run directories
def read_jsonl(path: Path, rows_allow: dict, *, code: str) -> tuple[list[dict], bool, str]:
    """(projected rows, torn last line, file sha256) of a strict-JSON telemetry file. Each row keeps only its
    event's allowlist; an event without one keeps {"event"} only. A torn last line is reported, not parsed. An
    unreadable file is refused as `code`."""
    S.refuse_test_path(path)
    data = _read_bytes(path, code)
    lines = data.split(b"\n")
    torn = lines[-1] != b""
    if not torn:
        lines = lines[:-1]
    rows = []
    for i, raw in enumerate(lines):
        if torn and i == len(lines) - 1:
            break
        doc = strict_json(raw, f"{path} line {i + 1}", code=code)
        if not hasattr(doc, "keys"):
            raise S.TmanRefused(code, f"{path} line {i + 1}: a row is a JSON object (DL-24 #83)")
        ev = doc["event"] if "event" in doc else None
        rows.append(project(doc, rows_allow.get(ev, {"event": True})))
    return rows, torn, S.sha256_bytes(data)


def read_json_file(path: Path, allow, *, code: str):
    S.refuse_test_path(path)
    data = _read_bytes(path, code)
    return project(strict_json(data, str(path), code=code), allow), S.sha256_bytes(data)


def file_sha256(path: Path, *, code: str) -> str:
    """A run file's sha256, its bytes not parsed (an unreadable file is refused as `code`)."""
    S.refuse_test_path(path)
    return S.sha256_bytes(_read_bytes(path, code))


def _read_bytes(path, code: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as e:
        raise S.TmanRefused(code, f"{path}: unreadable ({type(e).__name__})") from None


# ------------------------------------------------------------------------- the AM-3 trigger record (O-8; B2R13)
# L-AM3's reader of the committed AM-3 trigger record (merged at ebc1201, DL-140), read(path, bytes) ->
# {"triggered": bool}. Its module is imported when a record is read, as manifest.registered_cells imports
# src.corruption_roles (TB6); the smokes keep a fixture reader through the validator's Ctx(trigger_reader=...).
TRIGGER_MODULE = "src.quant.e6kd_trigger"
TRIGGER_RECORD_REL = "reports/derived/e6kd_trigger.json"       # src/quant/e6kd_trigger.py:75, the committed copy
TRIGGER_NOT_TRIGGERED = "trigger_not_triggered"                # its reader's last refusal: a record of triggered false


def l_am3_trigger(path: str, data: bytes, *, root=None) -> dict:
    """B2R13: the record at `path` read by L-AM3's read_trigger_record (src/quant/e6kd_trigger.py:594-637) -- tracked
    and unchanged at HEAD of `root` (the records' checkout; L-AM3's own without one), at the sha256 of `data`, the
    bytes the records hold, its decision following from its own values -- and its verdict kept, nothing else:
    {"triggered": True}, or {"triggered": False} when the reader's one refusal is that the record reads triggered
    false. Any other refusal is a ValueError (the validator's trigger_format_unknown)."""
    mod = importlib.import_module(TRIGGER_MODULE)
    kw = {} if root is None else {"repo_root": Path(root)}
    try:
        mod.read_trigger_record(path, S.sha256_bytes(data), **kw)
    except mod.QATRefused as e:
        if getattr(e, "code", None) == TRIGGER_NOT_TRIGGERED:
            return {"triggered": False}
        raise ValueError(f"L-AM3's reader refuses {path}: [{getattr(e, 'code', None)}] {e}") from None
    return {"triggered": True}


TRIGGER_READER = l_am3_trigger


def trigger_reader(records: Records):
    """The trigger reader of a validation or a check that names none: TRIGGER_READER -- L-AM3's, bound to the
    records' checkout -- or the stand-in a smoke put in its place (None: no reader)."""
    reader = TRIGGER_READER
    if reader is l_am3_trigger:
        root = records.src.root
        return lambda path, data: l_am3_trigger(path, data, root=root)
    return reader


# ------------------------------------------------------------------------------------- PTQ records (A2; lane LQP)
def ptq_record_reader():
    """Lane LQP's read_record(path, expect_sha256) -> (record, sha256), which re-hashes the record and every file it
    lists. Imported when a record is first read: the module is absent until lane/lqp merges (A2)."""
    return importlib.import_module(PTQ_RECORD_MODULE).read_record


def read_ptq_record(path, expect_sha256: str, *, reader=None) -> dict:
    """A2's projection of an intact ptq_record/1: stage, seed, role, descriptive, parent.checkpoint_sha256,
    calibration.checksum_sha256 and the artifacts' sha256 values. Read only through LQP's reader, or `reader`, its
    stand-in until lane/lqp merges. Refused as ptq_record_invalid: the reader cannot be imported or refuses the
    file, the sha256 differs, or the projection is incomplete."""
    S.refuse_test_path(path)
    try:
        read = ptq_record_reader() if reader is None else reader
    except ImportError as e:
        raise S.TmanRefused("ptq_record_invalid", f"{path}: {PTQ_RECORD_MODULE} cannot be imported ({e}); a "
                                                  "ptq_record is read only through lane LQP's reader (A2)") from None
    try:
        rec, sha = read(path, expect_sha256)
    except (OSError, ValueError, KeyError, TypeError) as e:          # LQP's RecordInvalid is a ValueError
        raise S.TmanRefused("ptq_record_invalid", f"{path}: {type(e).__name__}: {e}") from None
    got = project(rec, PTQ_RECORD)
    errs = S.field_errors(got, PTQ_RECORD_SPEC, str(path))
    if sha != expect_sha256 or errs:
        raise S.TmanRefused("ptq_record_invalid", f"{path}: sha256 {sha} (expected {expect_sha256}); "
                                                  + ("; ".join(errs) or "projection complete"))
    return got
