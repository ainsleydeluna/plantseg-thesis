"""Complete at an instant (AM-19 Definitions), read from run directories (lane TMAN, B-1, C2).

"A run is complete at a date when its trainer's end-of-run record meets that date":
  kd_run_end      a distillation run: the run_end record, the telemetry's single last row, at the run's max_iters,
                  with its checks passed (src/training/train_distill.py:1305-1311; the structure
                  sweep_select._run_end_ending checks, :1223-1246), at wall_clock_end;
  e1_val_result   an E1 run: the VAL row of its final iteration (iter == run_meta max_iters,
                  src/training/train_e1.py:486), at its wall_clock, with its log's last RESULT line reading PASS
                  (train_e1.py:553; the log is e1_stdout.log, reports/e1_launch_runbook_v2.md:172);
  qat_epoch15     a QAT run: the epoch-15 epoch_end row with run_complete true, the telemetry's last row, one
                  run_meta row first, epoch_end rows 1..15 in order, no run_stop or run_abort row, no torn line
                  (AM-21 item 1(e); src/quant/qat.py:1070-1079; the rule of src/quant/qat_artifacts.py:94-146);
  teacher_m12     a teacher fine-tune: the timestamp of the mmengine line "M12 selection (final): ..."
                  (src/training/teacher_components.py:388-390), read as UTC (ruling TB19 Q4; AM-22 item 2(b): a
                  line without a year is in 2026). Only the timestamp is read: the pattern captures nothing after
                  "(final): " (TB14);
  arm_output      the ARM latency measurement: its output and its agreement record, presence only (TB14), at the
                  instant supplied with them.

A timestamp meets a date when it is strictly earlier than the date's end, 16:00:00 UTC (sweep_select.cutoff_instant
and meets, src/training/sweep_select.py:420-428). Timestamps are kept to the second (floor): every cutoff is a whole
second, so floor(t) < cutoff exactly when t < cutoff.

A divergence is the trainer's own run_abort record (AM-7a): a distillation run diverged when its single last row is a
run_abort under rule AM-7(a) or AM-7(b) with cause student_divergence and finite inputs and teacher outputs
(sweep_select._abort_ending's split, :1212-1220); any other run_abort is an abort. QAT runs have no divergence rule
(AM-21 item 1; src/quant/qat.py:30): a non-finite state does not end the run, and a run_stop or run_abort row ends
it as stopped or aborted.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.training import sweep_select as ss

from . import records as R
from . import schema as S

# The statuses a run directory gives (`status_at`); B-2 adds AM-22's settled ways (plan §0, boundary 3).
COMPLETE, RUNNING, DIVERGED, ABORTED, STOPPED = "complete", "running", "diverged", "aborted", "stopped"
STATUSES = (COMPLETE, RUNNING, DIVERGED, ABORTED, STOPPED)
# AM-22's Definitions (docs/PREREGISTRATION_AMENDMENTS.md:2282-2310): the ways a registered run is settled at a date,
# the enum src/tman/am22.py reads (B2R3). A never-cut run settles only by completion or supersession; a cuttable run
# also by the rest. The run-directory statuses above stay as they are.
SETTLED_COMPLETE, SETTLED_SUPERSEDED, SETTLED_DIVERGED = "complete", "superseded", "diverged"
SETTLED_ABORTED, SETTLED_CUT, SETTLED_NOT_RUN = "aborted", "cut", "not_run"
NEVER_CUT_SETTLED_WAYS = (SETTLED_COMPLETE, SETTLED_SUPERSEDED)
SETTLED_WAYS = NEVER_CUT_SETTLED_WAYS + (SETTLED_DIVERGED, SETTLED_ABORTED, SETTLED_CUT, SETTLED_NOT_RUN)
DIVERGENCE_RULES = ss.DIVERGENCE_RULES                 # ("AM-7(a)", "AM-7(b)"), sweep_select.py:206
DIVERGENCE_CAUSE = ss.DIVERGENCE_CAUSE                 # "student_divergence", sweep_select.py:207

E1_TELEMETRY = "e1_telemetry.jsonl"
E1_LOG = "e1_stdout.log"
E1_RESULT_PASS = re.compile(r"^RESULT: PASS\b")
QAT_TELEMETRY = "qat_telemetry.jsonl"                  # src/quant/qat.py:75
QAT_EPOCH_DIR = "epoch_ckpts"                          # src/quant/qat.py:76
QAT_EPOCHS = 15                                        # src/quant/qat.py:81
# The M12 final line: an mmengine file-log line ('%Y/%m/%d %H:%M:%S', mmengine/logging/logger.py:267) or console
# line ('%m/%d %H:%M:%S', :217), logger name, level (colored on a console), then the message. Only the timestamp is
# captured (TB14); nothing after "(final): " is read.
M12_FINAL_RE = re.compile(r"^(?:(?P<year>\d{4})/)?(?P<month>\d{2})/(?P<day>\d{2}) (?P<hh>\d{2}):(?P<mm>\d{2}):"
                          r"(?P<ss>\d{2}) - \S+ - \S*INFO\S* - M12 selection \(final\): ")
M12_DEFAULT_YEAR = 2026                                # AM-22 item 2(b): a line with no year is in 2026
M12_LOG_DIR_RE = re.compile(r"^\d{8}_\d{6}$")          # mmengine's <work_dir>/<timestamp>/<timestamp>.log
# Any time-stamped mmengine line: its timestamp only (AM-22 item 2(b): "MM/DD HH:MM:SS" from the pod's UTC clock)
STAMP_RE = re.compile(r"^(?:(?P<year>\d{4})/)?(?P<month>\d{2})/(?P<day>\d{2}) (?P<hh>\d{2}):(?P<mm>\d{2}):"
                      r"(?P<ss>\d{2}) ")


def settled_ways(never_cut: bool) -> tuple:
    """The ways a registered run settles (AM-22 Definitions): a never-cut run's two, a cuttable run's six."""
    return NEVER_CUT_SETTLED_WAYS if never_cut else SETTLED_WAYS


def kd_telemetry_name(stage: str) -> str:
    """<stage key>_telemetry.jsonl (train_distill.py:1006-1008): the stage key is the base stage in lower case, e3
    for the AM-18 arms (configs/kd_arms: stage "e3") and e2/e3 for the L arm's students."""
    return f"{stage.split('-')[0].lower()}_telemetry.jsonl"


@dataclass(frozen=True)
class Completion:
    record_kind: str
    record: str                 # the record file, relative to the run directory
    sha256: str                 # that file's sha256
    at: int                     # the end-of-run record's Unix time, floored to the second
    line: int | None            # its 1-based line

    @property
    def at_utc(self) -> str:
        return utc_text(self.at)

    def meets(self, cutoff: int) -> bool:
        return ss.meets(self.at, cutoff)

    def as_entry(self) -> dict:
        """IF§11.2's completion block of a scored entry."""
        return {"record": self.record, "at_utc": self.at_utc, "sha256": self.sha256, "line": self.line}


@dataclass(frozen=True)
class LastRow:
    """The last time-stamped row a launched attempt wrote (AM-22 item 2(b)): its file, relative to the run directory,
    that file's sha256, and its Unix time floored to the second (None: no row or line carries a time)."""
    record: str
    sha256: str
    at: int | None


@dataclass(frozen=True)
class RunState:
    """What a run directory shows at the end of its records: its status, the completion record if complete, and
    the instant, rule, cause and iteration (or epoch) of a divergence, stop or abort (identity and time only)."""
    status: str
    completion: Completion | None = None
    ended_at: int | None = None
    rule: str | None = None
    cause: str | None = None
    iteration: int | None = None
    epoch: int | None = None
    telemetry_sha256: str | None = None

    def complete_at(self, cutoff: int) -> bool:
        return self.status == COMPLETE and self.completion is not None and self.completion.meets(cutoff)


def status_at(state: RunState, cutoff: int) -> str:
    """The run's status at an instant: its ending (complete, diverged, aborted, stopped) when the ending record meets
    the instant, otherwise running."""
    if state.status == RUNNING or state.ended_at is None or not ss.meets(state.ended_at, cutoff):
        return RUNNING
    return state.status


def utc_text(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def _second(x) -> int | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return int(math.floor(x))


def _invalid(detail: str):
    raise S.TmanRefused("completion_record_invalid", detail)


def _ending_time(rel: str, row: dict, key: str = "wall_clock") -> int:
    at = _second(row.get(key))
    if at is None:
        _invalid(f"{rel}: its {row.get('event')} record has no finite {key}")
    return at


# ------------------------------------------------------------------------------------------------- KD runs
def kd_state(run_dir: Path, stage: str) -> RunState:
    """A distillation run of catalogue stage `stage`: complete at its run_end's wall_clock_end; diverged or aborted
    at a run_abort record; otherwise running. A run_end or run_abort that is not the single last row, a run_end
    beside a run_abort, a run_end short of max_iters or with its checks failed is completion_record_invalid."""
    rel = kd_telemetry_name(stage)
    meta_rel = rel.replace("_telemetry.jsonl", "_run_meta.jsonl")
    rows, torn, sha = R.read_jsonl(Path(run_dir) / rel, R.KD_ROWS, code="completion_record_invalid")
    ends = [i for i, r in enumerate(rows) if r.get("event") in ("run_end", "run_abort")]
    if not ends:
        return RunState(RUNNING, telemetry_sha256=sha)
    i = ends[-1]
    r = rows[i]
    if len(ends) != 1 or i != len(rows) - 1 or torn:
        _invalid(f"{rel}: an end-of-run record (run_end or run_abort) must be the single last row")
    if r["event"] == "run_abort":
        at = _ending_time(rel, r)
        div = (r.get("rule") in DIVERGENCE_RULES and r.get("cause") == DIVERGENCE_CAUSE
               and r.get("input_finite") is True and r.get("teacher_finite") is True)
        return RunState(DIVERGED if div else ABORTED, ended_at=at, rule=r.get("rule"), cause=r.get("cause"),
                        iteration=r.get("iter"), telemetry_sha256=sha)
    metas, _, _ = R.read_jsonl(Path(run_dir) / meta_rel, {"run_meta": R.KD_RUN_META}, code="completion_record_invalid")
    metas = [m for m in metas if m.get("event") == "run_meta"]
    max_iters = metas[-1].get("max_iters") if metas else None
    if not S.TYPES["int"](max_iters) or r.get("iter") != max_iters:
        _invalid(f"{rel}: run_end at iter {r.get('iter')!r}; {meta_rel} gives max_iters {max_iters!r}")
    if r.get("checks_passed") is not True:
        _invalid(f"{rel}: run_end with checks_passed {r.get('checks_passed')!r}: not complete; a run whose checks "
                 "failed is a decision for a person")
    at = _ending_time(rel, r, "wall_clock_end")
    return RunState(COMPLETE, Completion("kd_run_end", rel, sha, at, i + 1), ended_at=at, telemetry_sha256=sha)


# ------------------------------------------------------------------------------------------------ E1 runs
def e1_state(run_dir: Path) -> RunState:
    """An E1 run: complete at the wall_clock of the VAL row of its final iteration (run_meta max_iters), with
    e1_stdout.log's last RESULT line reading PASS (train_e1 writes no run_end)."""
    rows, torn, sha = R.read_jsonl(Path(run_dir) / E1_TELEMETRY, R.E1_ROWS, code="completion_record_invalid")
    metas = [r for r in rows if r.get("event") == "run_meta"]
    if not metas or not S.TYPES["int"](metas[-1].get("max_iters")):
        _invalid(f"{E1_TELEMETRY}: no run_meta row with max_iters")
    last_it = metas[-1]["max_iters"]
    finals = [i for i, r in enumerate(rows) if r.get("event") == "val" and r.get("iter") == last_it]
    if not finals:
        return RunState(RUNNING, telemetry_sha256=sha)
    i = finals[-1]
    at = _ending_time(E1_TELEMETRY, rows[i])
    log = Path(run_dir) / E1_LOG
    S.refuse_test_path(log)
    if not log.is_file():
        _invalid(f"{E1_LOG} is missing: the final VAL row needs its log's RESULT line reading PASS")
    results = [ln for ln in log.read_bytes().decode("utf-8", "replace").splitlines() if ln.startswith("RESULT:")]
    if not results or not E1_RESULT_PASS.match(results[-1]):
        _invalid(f"{E1_LOG}: its last RESULT line is not PASS")
    return RunState(COMPLETE, Completion("e1_val_result", E1_TELEMETRY, sha, at, i + 1), ended_at=at,
                    telemetry_sha256=sha)


# ------------------------------------------------------------------------------------------------ QAT runs
def qat_rows(run_dir: Path) -> tuple[list[dict], bool, str]:
    return R.read_jsonl(Path(run_dir) / QAT_TELEMETRY, R.QAT_ROWS, code="completion_record_invalid")


def qat_state(run_dir: Path) -> RunState:
    """A QAT run: complete at its epoch-15 epoch_end row; stopped or aborted at a run_stop or run_abort row;
    otherwise running. More than one run_meta row, a run_meta row not first, or epoch_end rows out of order or
    repeated is completion_record_invalid."""
    rows, torn, sha = qat_rows(run_dir)
    if not rows or rows[0].get("event") != "run_meta" or sum(r.get("event") == "run_meta" for r in rows) != 1:
        _invalid(f"{QAT_TELEMETRY}: exactly one run_meta row, first, is required")
    ends = [r.get("epoch") for r in rows if r.get("event") == "epoch_end"]
    if not all(S.TYPES["int"](e) for e in ends) or ends != list(range(1, len(ends) + 1)):
        _invalid(f"{QAT_TELEMETRY}: epoch_end rows for epochs {ends}, not 1..n in order")
    stops = [r for r in rows if r.get("event") in ("run_stop", "run_abort")]
    if stops:
        r = stops[-1]
        return RunState(STOPPED if r["event"] == "run_stop" else ABORTED, ended_at=_ending_time(QAT_TELEMETRY, r),
                        rule=r.get("code"), cause=r.get("cause") or r.get("code"), epoch=r.get("epoch"),
                        telemetry_sha256=sha)
    last = rows[-1]
    if torn or len(ends) < QAT_EPOCHS:
        return RunState(RUNNING, telemetry_sha256=sha)
    if not (last.get("event") == "epoch_end" and last.get("epoch") == QAT_EPOCHS and last.get("run_complete") is True):
        _invalid(f"{QAT_TELEMETRY}: the last row is not the epoch-{QAT_EPOCHS} completion record")
    at = _ending_time(QAT_TELEMETRY, last)
    return RunState(COMPLETE, Completion("qat_epoch15", QAT_TELEMETRY, sha, at, len(rows)), ended_at=at,
                    telemetry_sha256=sha)


def qat_checkpoint_flags(path: Path) -> dict:
    """The fields an epoch checkpoint stores (AM-21 item 1(c); qat.py:712-742), loaded weights_only, projected."""
    import torch
    S.refuse_test_path(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    return R.project(payload, R.QAT_CHECKPOINT)


def qat_flag_errors(run_dir: Path, rows: list[dict]) -> list[str]:
    """Each epoch checkpoint present in the run directory against its epoch_end row: the file's sha256, then the
    three stored state flags, present and equal in type and value (qat_artifacts.state_flag_row_error's rule,
    :180-190). The validator refuses checkpoint_state_flag (TB13)."""
    errs = []
    for r in rows:
        if r.get("event") != "epoch_end":
            continue
        name = r.get("checkpoint")
        if not (isinstance(name, str) and name == f"{QAT_EPOCH_DIR}/e{r.get('epoch'):02d}.pt"):
            errs.append(f"epoch {r.get('epoch')!r}: its epoch_end row names checkpoint {name!r}")
            continue
        p = Path(run_dir) / name
        S.refuse_test_path(p)
        if not p.is_file():
            continue
        if S.sha256_bytes(p.read_bytes()) != r.get("checkpoint_sha256"):
            errs.append(f"{name}: its sha256 differs from its epoch_end row")
            continue
        flags = qat_checkpoint_flags(p)
        missing = [k for k in R.QAT_STATE_FLAG_KEYS if k not in flags or k not in r]
        if missing:
            errs.append(f"{name}: state flag(s) {missing} missing from the checkpoint or its epoch_end row")
            continue
        for k in R.QAT_STATE_FLAG_KEYS:
            if type(flags[k]) is not type(r[k]) or flags[k] != r[k]:
                errs.append(f"{name}: it stores {k} {flags[k]!r}, its epoch_end row {r[k]!r}")
    return errs


# --------------------------------------------------------------------------------------- teacher fine-tunes
def m12_final_instant(text: str) -> tuple[int, int] | None:
    """(Unix time, 1-based line) of the first M12 "(final)" line of an mmengine log, read as UTC; None if absent.
    Nothing after the line's prefix is read."""
    for n, line in enumerate(text.splitlines(), 1):
        m = M12_FINAL_RE.match(line)
        if m:
            year = int(m.group("year")) if m.group("year") else M12_DEFAULT_YEAR
            try:
                t = datetime(year, int(m.group("month")), int(m.group("day")), int(m.group("hh")),
                             int(m.group("mm")), int(m.group("ss")), tzinfo=timezone.utc)
            except ValueError:
                _invalid(f"M12 final line {n}: its timestamp is not a date")
            return int(t.timestamp()), n
    return None


def last_stamp(text: str) -> int | None:
    """The Unix time of a log's last time-stamped line, read as UTC (a line with no year is in 2026, AM-22 item
    2(b)); None when no line carries a time. Only the timestamp is read."""
    for line in reversed(text.splitlines()):
        m = STAMP_RE.match(line)
        if m:
            try:
                return int(datetime(int(m.group("year") or M12_DEFAULT_YEAR), int(m.group("month")),
                                    int(m.group("day")), int(m.group("hh")), int(m.group("mm")), int(m.group("ss")),
                                    tzinfo=timezone.utc).timestamp())
            except ValueError:
                _invalid(f"a log line's timestamp is not a date: {line[:19]!r}")
    return None


def teacher_state(work_dir: Path, log_rel: str | None = None) -> RunState:
    """A teacher fine-tune: complete at its M12 "(final)" line. `log_rel` names the log; without it the work dir's
    <timestamp>/<timestamp>.log files are read (one listing of the work dir; a name is matched before any call on
    it)."""
    work = Path(work_dir)
    S.refuse_test_path(work)
    if not work.is_dir():
        _invalid(f"{work}: the teacher's work dir is not a folder")
    logs = [log_rel] if log_rel else sorted(
        f"{d.name}/{d.name}.log" for d in work.iterdir() if M12_LOG_DIR_RE.match(d.name) and d.is_dir())
    for rel in logs:
        p = work / rel
        S.refuse_test_path(p)
        if not p.is_file():
            continue
        data = p.read_bytes()
        hit = m12_final_instant(data.decode("utf-8", "replace"))
        if hit:
            return RunState(COMPLETE, Completion("teacher_m12", rel, S.sha256_bytes(data), hit[0], hit[1]),
                            ended_at=hit[0])
    return RunState(RUNNING)


def teacher_last_row(work_dir: Path) -> LastRow:
    """AM-22 item 2(b) for a teacher fine-tune, whose scalar rows carry no wall-clock time: the last time-stamped line
    of its training logs, read in order as teacher_state lists them (one listing of the work dir; a name is matched
    before any call on it; a line with no year is in 2026). Only this reader parses the other lines' stamps, so
    teacher_state stays B-1's (DL-24 R1-N2)."""
    work = Path(work_dir)
    S.refuse_test_path(work)
    if not work.is_dir():
        raise S.TmanRefused("completion_record_invalid", f"{work}: the teacher's work dir is not a folder")
    last = LastRow("", "", None)
    for rel in sorted(f"{d.name}/{d.name}.log" for d in work.iterdir() if M12_LOG_DIR_RE.match(d.name)
                      and d.is_dir()):
        p = work / rel
        S.refuse_test_path(p)
        data = p.read_bytes() if p.is_file() else None
        at = last_stamp(data.decode("utf-8", "replace")) if data is not None else None
        if at is not None:
            last = LastRow(rel, S.sha256_bytes(data), at)
    return last


# --------------------------------------------------------------------------------------- the ARM measurement
def arm_state(out_dir: Path, output_rel: str, agreement_rel: str, at_utc: str) -> RunState:
    """L-AM16-ARM's output and agreement record (a reader slot until that lane fixes the format): complete when both
    files are present, at the instant supplied with them. Presence only (TB14)."""
    o, a = Path(out_dir) / output_rel, Path(out_dir) / agreement_rel
    for p in (o, a):
        S.refuse_test_path(p)
    if not (o.is_file() and a.is_file()):
        return RunState(RUNNING)
    if not S.TYPES["utc"](at_utc):
        _invalid(f"the measurement's instant {at_utc!r} is not a UTC instant")
    at = parse_utc(at_utc)
    return RunState(COMPLETE, Completion("arm_output", agreement_rel, S.sha256_bytes(a.read_bytes()), at, None),
                    ended_at=at)


# ------------------------------------------------------------------------------- launched attempts per slot
TEACHER_WORK_DIR_RE = re.compile(r"^teacher-(?P<arm>r1|r2|l)_s(?P<seed>\d+)_40k_a(?P<attempt>[1-9]\d*)$")
TEACHER_PROVENANCE = "teacher_arm_launch_provenance.json"      # scripts/launch_teacher_arm.py:92, in the work dir


def slot_of_launch(line: dict) -> str:
    """A KD launch line's catalogue slot: <stage>:s<seed>[:lambda<v>][:alpha<v>]:<horizon/1000>k (plan §1.3)."""
    parts = [line["stage"], f"s{line['seed']}"]
    if line.get("lambda_logit") is not None:
        parts.append(f"lambda{float(line['lambda_logit']):g}")
    if line.get("alpha_cwd") is not None:
        parts.append(f"alpha{float(line['alpha_cwd']):g}")
    parts.append(f"{int(line['horizon']) // 1000}k")
    return ":".join(parts)


class RunIndex:
    """Each catalogue run slot's launched attempts, from the records and the run directories, never from an entry
    (AM-19 Definitions: a run is launched when its run_meta row with mode "real" exists; AM-19a readings 6 and 20).
    KD slots: the launch log's launched lines, mapped by slot_of_launch, in attempt order (the log receives every KD
    launch). E1 and QAT slots: the run directories under --runs, found by one listing of that folder (each name
    checked for "test" before any call on it) and named by their own run_meta row with mode real (E1: seed and
    max_iters; QAT: stage, seed and, for the U4 pilot, clip_norm), in launch-time order. A teacher arm: its work dir,
    named teacher-<arm>_s<seed>_40k_a<n> by its launcher (scripts/launch_teacher_arm.py:90), with the launch
    provenance launch true. Without --runs only the KD slots are known. A launched run whose records name no
    catalogue slot is kept in `foreign`; the teacher of record and the ARM measurement have no attempts here."""

    def __init__(self, cat, log=None, runs_root=None, states: dict | None = None):
        self.cat, self.root = cat, None if runs_root is None else Path(runs_root)
        self.slot_of: dict = {}               # run_id -> its catalogue slot
        self.foreign: list = []               # (run_id, the slot its records name)
        self.stopped: set = set()             # KD stopped lines
        self._states = {} if states is None else states          # run_id -> RunState (shared with the caller)
        found: dict = {}
        if log is not None:
            for rid in log.order:
                if rid in log.launched:
                    self._add(found, rid, slot_of_launch(log.launch[rid]), (log.launch[rid]["attempt"], rid))
            self.stopped = set(log.stopped)
        kd = set(log.launch) if log is not None else set()
        if self.root is not None:
            S.refuse_test_path(self.root)
            if not self.root.is_dir():
                raise S.TmanRefused("launched_attempt_without_directory", f"--runs {self.root} is not a folder")
            for name in sorted(p.name for p in self.root.iterdir()):
                S.refuse_test_path(name, self.root)
                if name in kd:
                    continue
                got = self._identify(name)
                if got is not None:
                    self._add(found, name, got[0], got[1])
        self.by_slot = {k: [rid for _, rid in sorted(v)] for k, v in found.items()}
        self._kd_slots = {k for k, s in cat.slots.items() if s["record_kind"] == "kd_run_end"}

    def _add(self, found: dict, rid: str, slot: str, order) -> None:
        if slot in self.cat.slots and self.cat.slots[slot]["class"] == "run":
            self.slot_of[rid] = slot
            found.setdefault(slot, []).append(order)
        else:
            self.foreign.append((rid, slot))

    def _identify(self, name: str):
        """(slot, order key) of a directory under --runs that is a launched E1, QAT or teacher-arm run, else None."""
        d = self.root / name
        m = TEACHER_WORK_DIR_RE.fullmatch(name)
        if m:
            p = d / TEACHER_PROVENANCE
            if not p.is_file():
                return None
            prov, _ = R.read_json_file(p, R.LAUNCH_PROVENANCE, code="completion_record_invalid")
            if prov.get("launch") is not True:
                return None
            return f"teacher-{m.group('arm').upper()}:s{m.group('seed')}:40k", (int(m.group("attempt")), name)
        for rel, allow, key in ((QAT_TELEMETRY, R.QAT_ROWS, "qat"), (E1_TELEMETRY, R.E1_ROWS, "e1")):
            p = d / rel
            if not p.is_file():
                continue
            rows, _, _ = R.read_jsonl(p, allow, code="completion_record_invalid")
            meta = rows[0] if rows and rows[0].get("event") == "run_meta" else {}
            if meta.get("mode") != "real":
                return None
            at = _second(meta.get("wall_clock"))
            return (self._qat_slot(meta) if key == "qat" else self._e1_slot(meta)), (at if at is not None else 0, name)
        return None

    def _qat_slot(self, meta: dict) -> str:
        hits = [k for k, s in self.cat.slots.items() if s["record_kind"] == "qat_epoch15"
                and s["stage"] == meta.get("stage") and s["seed"] == meta.get("seed")
                and ("clip" not in s["params"] or (S.TYPES["number"](meta.get("clip_norm"))
                                                   and float(meta["clip_norm"]) == float(s["params"]["clip"])))]
        return hits[0] if len(hits) == 1 else f"{meta.get('stage')}:s{meta.get('seed')}:clip{meta.get('clip_norm')}"

    @staticmethod
    def _e1_slot(meta: dict) -> str:
        it = meta.get("max_iters")
        return f"E1:s{meta.get('seed')}:{it // 1000 if S.TYPES['int'](it) else it}k"

    def known(self, slot: str) -> bool:
        """Whether a slot's attempts are read: a KD slot always (the launch log); an E1, QAT or teacher-arm slot with
        --runs. The teacher of record's run folder is named by no launcher rule: its entry and DL-88 name it."""
        s = self.cat.slots[slot]
        if slot in self._kd_slots:
            return True
        return self.root is not None and s["class"] == "run" and (
            s["record_kind"] in ("e1_val_result", "qat_epoch15") or s["stage"].startswith("teacher-"))

    def attempts(self, slot: str):
        """The slot's launched attempts in order, or None when they are not read (a non-KD slot without --runs, a
        measurement)."""
        return list(self.by_slot.get(slot, [])) if self.known(slot) else None

    def state(self, run_id: str) -> RunState:
        """A launched attempt's state, read from its directory under its own slot's record kind."""
        if run_id not in self._states:
            S.refuse_test_path(run_id, self.root)
            if self.root is None:
                raise S.TmanRefused("launched_attempt_without_directory", "the run directories are read (--runs)")
            self._states[run_id] = run_state(self.cat.slots[self.slot_of[run_id]], self.root / run_id)
        return self._states[run_id]

    def status(self, slot: str, cutoff: int) -> str | None:
        """The slot's status at `cutoff` from its own attempts: complete when one of them is complete then, else its
        latest attempt's; None without --runs or without a launched attempt."""
        att = self.attempts(slot) if self.root is not None else None
        if not att:
            return None
        states = [self.state(r) for r in att]
        if any(st.complete_at(cutoff) for st in states):
            return COMPLETE
        return status_at(states[-1], cutoff)

    def stopped_ids(self) -> set:
        """The run_ids stopped awaiting a repeat (AM-19 item 3(a)): KD stopped lines and QAT run_stop rows."""
        out = set(self.stopped)
        if self.root is not None:
            out |= {r for r, k in self.slot_of.items() if self.cat.slots[k]["record_kind"] == "qat_epoch15"
                    and self.state(r).status == STOPPED}
        return out


# --------------------------------------------------------------------------------------- one reader per kind
def run_state(slot: dict, run_dir: Path, **kw) -> RunState:
    """The state of a catalogue slot's run directory, by its record kind (the run class; derived slots take their
    parents' and measurement slots arm_state)."""
    kind = slot["record_kind"]
    if kind == "kd_run_end":
        return kd_state(run_dir, slot["stage"])
    if kind == "e1_val_result":
        return e1_state(run_dir)
    if kind == "qat_epoch15":
        return qat_state(run_dir)
    if kind == "teacher_m12":
        return teacher_state(run_dir, kw.get("log_rel"))
    if kind == "arm_output":
        if any(k not in kw for k in ("output_rel", "agreement_rel", "at_utc")):
            raise S.TmanRefused("completion_record_invalid", f"slot {slot['slot']}: an ARM measurement is read from "
                                                             "its output and agreement record, not a run directory")
        return arm_state(run_dir, kw["output_rel"], kw["agreement_rel"], kw["at_utc"])
    raise S.TmanRefused("completion_record_invalid", f"slot {slot['slot']}: record kind {kind!r} has no run reader")


# --------------------------------------------------------------------------- the last row (AM-22 item 2(b))
class _EveryEvent(dict):
    """read_jsonl's allowlist by event, the same keys for every event: a row's time and nothing else."""

    def get(self, key, default=None):
        return {"event": True, "wall_clock": True, "wall_clock_end": True}


ROW_TIMES = _EveryEvent()


def last_row(slot: dict, run_dir: Path) -> LastRow:
    """The last time-stamped row a launched attempt of `slot` wrote (AM-22 item 2(b)): the last row of its telemetry
    that carries a wall_clock (a run_end's wall_clock_end), whatever its event; for a teacher fine-tune, whose scalar
    rows carry no wall-clock time, the last time-stamped line of its training log (teacher_last_row). Time only."""
    kind = slot["record_kind"]
    if kind == "teacher_m12":
        return teacher_last_row(run_dir)
    rel = {"kd_run_end": kd_telemetry_name(slot["stage"]), "e1_val_result": E1_TELEMETRY,
           "qat_epoch15": QAT_TELEMETRY}.get(kind)
    if rel is None:
        raise S.TmanRefused("completion_record_invalid", f"slot {slot['slot']}: record kind {kind!r} writes no rows")
    rows, _, sha = R.read_jsonl(Path(run_dir) / rel, ROW_TIMES, code="completion_record_invalid")
    for r in reversed(rows):
        at = _second(r.get("wall_clock_end") if r.get("event") == "run_end" else r.get("wall_clock"))
        if at is not None:
            return LastRow(rel, sha, at)
    return LastRow(rel, sha, None)
