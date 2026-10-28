"""Entry files (tday_entry/1): their fields, their checks and their printers (lane TMAN, B-1, C3).

An entry file records one decision-log entry in machine-readable form (plan decision (iii)): a cut (AM-19 item
3(b)), a "not run" (item 4(a)), a freeze delay and a new freeze (item 1(d)), a lane entered as not required (item
3(d)), the unlock, AM-23's condition verdict (item 3(b)) and "not made" (item 4), the freeze record, and the session
host's clock offset (AM-19a reading 21; TB11). Each file is written once, outside the repository, by its printer; a
local block commits it under reports/derived/tday_entries/ as <kind>__<file_id>__<YYYYMMDD>__<nn>.json, and the
decision-log row the printer prints names its path and sha256. The validator accepts the file only when its sha256
sits in a RECORDED or IMPLEMENTED row (DecisionLog.recorded_rows_with) and the file is written once.

A printer builds the entry, runs the entry's checks, writes the file exclusively and prints its row (the pattern of
scripts/preflight_arms.py:1202-1293). Field specs are additive per kind (T-19): B-2 adds rule "cutoff" and AM-22's
fields of kind "freeze"; an unknown kind is refused.

AM-19 item 3(b)'s arithmetic for any cuttable run (ruling TB19 Q6: the lane's printer is the record for non-arm
runs) is scripts/preflight_arms.py's cut_arithmetic (:537-558), reused (decision (xvii)): iterations x the larger of
the references' median seconds per iteration, against the time left to the original freeze date; no figure, no cut;
equality fits.
"""
from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath

from src.training import sweep_select as ss

from . import completion as C
from . import records as R
from . import schedule as D
from . import schema as S

REF = S.REF
ENTRY_FILE_COMMON = {
    "format": ("enum", (S.ENTRY_FORMAT,)), "kind": "str", "slots": ("list", "slot"), "date": "day",
    "amendment": "str", "printed_by": ("obj", {"script": "str", "commit": "?commit"}), "corrects": S.nullable(REF),
    "push_evidence": S.nullable(("obj", {"response": REF, "list": REF})),
}
REFERENCE = ("obj", {"source": ("enum", ("telemetry", "smoke")), "path": "str", "sha256": "sha256", "field": "str",
                     "median": "number", "detail": ("map", "any")})
ARITHMETIC = ("obj", {"iterations": "int", "freeze_date": "day", "freeze_utc": "utc", "now_utc": "str",
                      "seconds_left": "number", "references": ("list", REFERENCE), "measured": "bool",
                      "applied_s_per_iter": "?number", "seconds_needed": "?number", "cut": "bool", "text": "str"})
RUN_FILE = ("obj", {"run_id": "str", "path": "str", "sha256": "sha256"})
RESULT_REF = ("obj", {"dl": "?dl", "path": "str", "sha256": "sha256"})
ENTRY_FILE_FIELDS: dict[str, dict] = {
    "cut": {"rule": "str", "reason": "str", "launched": "bool", "run_ids": ("list", "str"),
            "since": S.nullable(("obj", {"date": "day", "record": REF})), "arithmetic": S.nullable(ARITHMETIC),
            "telemetry": ("list", RUN_FILE),
            "comparator": S.nullable(("obj", {"slot": "slot", "run_id": "str", "path": "str", "sha256": "sha256"})),
            "results_known": ("list", RESULT_REF), "cut_order_position": "?int",
            "rule_record": S.nullable(("obj", {"path": "str", "sha256": "sha256", "date": "day"}))},
    "not_run": {"reason": ("enum", S.NOT_RUN_REASONS), "parent": "slot", "parent_record": REF},
    "freeze_delay": {"original_date": "day", "runs": ("list", "slot")},
    "new_freeze": {"new_date": "day", "delay_entry": REF},
    "not_required": {"lane": "str", "cut_entries": ("list", REF)},
    "unlock": {"manifest": ("obj", {"path": "str", "sha256": "sha256", "body_sha256": "sha256",
                                    "appends_sha256": "sha256"}), "lanes": REF, "freeze_record": REF},
    "condition": {"arm": ("enum", ("R1", "R2")), "verdict": ("enum", S.CONDITION_VERDICTS),
                  "e3x": ("obj", {"run_id": "str", "checkpoint_sha256": "sha256", "best_json_sha256": "sha256"})},
    "not_made": {"reason": ("enum", (S.NOT_MADE_REASON,)), "lane": "str"},
    "freeze": {"body_sha256": "sha256", "freeze_date": "day", "source": "str"},
    "clock_offset": {"host": "str", "source": "str", "offset_seconds": "number", "measured_utc": "utc"},
}
# B2R5: kind freeze gains AM-22's fields, additively (schema.register_kind; T-19): its D and its five conditions,
# numbered 1 to 5, each with the records check-freeze recomputed for D; both null under item 1(f).
AM22_RECORD = ("obj", {"kind": "str", "ref": "str", "sha256": "?sha256", "date": "day"})
S.register_kind(ENTRY_FILE_FIELDS, "freeze", {"d_date": "?day", "conditions": S.nullable(("list", ("obj", {
    "number": "int", "records": ("list", AM22_RECORD)})))})
SESSION_KINDS = ("freeze_delay", "new_freeze", "unlock", "freeze", "clock_offset")     # entries about no slot
ROW_STATUS = "RECORDED"


def entry_spec(kind: str) -> dict:
    if not isinstance(kind, str) or kind not in ENTRY_FILE_FIELDS:
        raise S.TmanRefused("entry_kind_unknown", f"entry kind {kind!r} is not one of {sorted(ENTRY_FILE_FIELDS)}")
    return {**ENTRY_FILE_COMMON, **ENTRY_FILE_FIELDS[kind]}


def entry_field_errors(doc) -> list[str]:
    """The entry file's fields against its kind's spec (entry_kind_unknown for an unknown kind)."""
    kind = doc.get("kind") if isinstance(doc, dict) else None
    errs = S.field_errors(doc, entry_spec(kind), f"entry {kind}")
    if not errs and (kind in SESSION_KINDS) != (doc["slots"] == []):
        errs.append(f"entry {kind}: {'a session entry names no slot' if kind in SESSION_KINDS else 'names no slot'}")
    return errs


def require_fields(doc) -> None:
    errs = entry_field_errors(doc)
    if errs:
        raise S.TmanRefused("entry_field", "; ".join(errs[:4]))


# --------------------------------------------------------------------------------------------- arithmetic
def _pa():
    import scripts.preflight_arms as pa     # imported when used: its imports are heavy
    return pa


def cut_arithmetic(iterations: int, figures: list, in_force: str, now: float) -> dict:
    """AM-19 item 3(b) for one run against the original freeze date of the schedule in force: preflight_arms's
    cut_arithmetic (:537-558), reused (decision (xvii))."""
    return _pa().cut_arithmetic(ss, iterations, figures, in_force, now)


def arithmetic_error(a: dict, in_force: str) -> str | None:
    """The recorded arithmetic recomputed from its own references and time (cut_arithmetic_invalid)."""
    try:
        now = D.parse_utc_any(a["now_utc"])
        again = cut_arithmetic(a["iterations"], a["references"], in_force, now)
    except (KeyError, ValueError, TypeError) as e:
        return f"arithmetic: not recomputable ({type(e).__name__}: {e})"
    for k in ("freeze_date", "freeze_utc", "seconds_left", "measured", "applied_s_per_iter", "seconds_needed", "cut",
              "text"):
        x, y = a.get(k), again[k]
        if isinstance(x, float) or isinstance(y, float):
            if x is None or y is None or not math.isclose(float(x), float(y), rel_tol=0, abs_tol=1e-6):
                return f"arithmetic: {k} {x!r} differs from the recomputed {y!r}"
        elif x != y:
            return f"arithmetic: {k} {x!r} differs from the recomputed {y!r}"
    return None


HORIZON_RE = re.compile(r":(?P<k>\d+)k$")         # a run slot's key ends with its horizon, <n>k iterations


def slot_iterations(slot: dict, reference_steps: int | None = None) -> int | None:
    """AM-19 item 3(b)'s iterations of a run slot: its registered horizon, the key's <n>k (n x 1000; preflight_arms
    takes spec.horizon, :769); a QAT run's, its 15 epochs' steps as its reference run counted them (None without)."""
    m = HORIZON_RE.search(slot["slot"])
    if m:
        return int(m.group("k")) * 1000
    return reference_steps if slot["record_kind"] == "qat_epoch15" else None


def reference_telemetry(runs_root: Path, run_id: str, slot: dict) -> dict:
    """A reference figure from a finished run of the cut slot's stage (AM-19 item 3(b): the most recently finished
    run of its stage; the caller checks which one is), read as the slot's own runs are read (its record kind and
    stage; DL-24 #11, #111): its completion reader must find it complete; the figure is the median of every train
    row's iter_seconds (as preflight_arms.reference_telemetry, :511-534; KD, E1 and QAT train rows carry
    iter_seconds). The detail records the run's end and, for a QAT run, its steps (its last epoch_end row's)."""
    d = Path(runs_root) / run_id
    S.refuse_test_path(d)
    kind = slot["record_kind"]
    if kind not in ("kd_run_end", "e1_val_result", "qat_epoch15"):
        raise S.TmanRefused("cut_arithmetic_invalid", f"a {kind} run gives no iter_seconds figure")
    st = C.run_state(slot, d)
    if st.status != C.COMPLETE:
        raise S.TmanRefused("cut_arithmetic_invalid", f"{run_id}: not a finished run of {slot['stage']} ({st.status})")
    rel = st.completion.record
    rows, _, sha = R.read_jsonl(d / rel, {"train": {"event": True, "iter_seconds": True},
                                          "epoch_end": {"event": True, "step": True}}, code="cut_arithmetic_invalid")
    vals = [r["iter_seconds"] for r in rows if r.get("event") == "train" and S.TYPES["number"](r.get("iter_seconds"))
            and math.isfinite(r["iter_seconds"]) and r["iter_seconds"] > 0]
    if not vals:
        raise S.TmanRefused("cut_arithmetic_invalid", f"{run_id}/{rel}: no train row with a finite iter_seconds")
    detail = {"rows": len(vals), "end_utc": C.utc_text(st.completion.at)}
    if kind == "qat_epoch15":
        steps = [r.get("step") for r in rows if r.get("event") == "epoch_end"]
        if not (steps and S.TYPES["int"](steps[-1]) and steps[-1] > 0):
            raise S.TmanRefused("cut_arithmetic_invalid", f"{run_id}/{rel}: its last epoch_end row has no step count")
        detail["steps"] = steps[-1]
    return {"source": "telemetry", "path": f"{run_id}/{rel}", "sha256": sha, "field": "every train row's iter_seconds",
            "median": float(_pa().MEDIAN(vals)), "detail": detail}


def reference_smoke(runs_root: Path, rel: str, key: str) -> dict:
    """A reference figure from a pod smoke's JSON kept under the run directories, recorded by its path there so it
    can be verified again (DL-24 #11): throughput[<key>]["iter_seconds"]["median"] (preflight_arms's one reader,
    smoke_figure :485-491). A JSON value that is not an object is refused, never crashed on (#68)."""
    S.refuse_test_path(rel)
    p = Path(runs_root) / rel
    S.refuse_test_path(p)
    pa = _pa()
    try:
        median = pa.smoke_figure(p, key)
    except (pa.GateError, OSError, ValueError, AttributeError, TypeError) as e:
        raise S.TmanRefused("cut_arithmetic_invalid", f"--reference-smoke {rel}: {type(e).__name__}: {e}") from None
    data = p.read_bytes()
    fmt = R.strict_json(data, str(p), code="cut_arithmetic_invalid").get("format")
    return {"source": "smoke", "path": str(PurePosixPath(rel)), "sha256": S.sha256_bytes(data),
            "field": f"throughput[{key!r}]['iter_seconds']['median']", "median": median,
            "detail": {"format": fmt, "key": key}}


# ------------------------------------------------------------------------------- the cut checks, in gate order
ARM_GROUPS = ("R2", "W", "R1", "L")          # AM-18 item 7(a)/(b) and the AM-17 item 11 arm
CONDITION_STAGES = ("E6-R1", "E6-R2")        # AM-23 item 3(b)
NAMED_RULE_KINDS = ("condition", "go_rule")  # AM-23 item 3(b)'s condition entry; AM-18 item 5's go-rule record


class CutContext:
    """What the cut checks read: the catalogue, the freeze instant, the arm registry by run stage, each sweep's
    projected decision record (decision_date, values), and, from the records and the run directories, never from
    the entry (completion.RunIndex): each slot's launched attempts (None: not read), the slot a launched run_id
    belongs to, the run_ids stopped (a launch log's stopped line; a QAT run_stop row), each slot's status at the
    original instant from its own attempts (None: not read or never launched); the reading of a named rule's record
    (rule_record_reader) and E3-X's run of record; the day a committed cut entry was entered in the decision log; and
    whether a record was committed by the end of a day. For AM-22 item 2's cutoff (B-2): its dates (am22: a callable
    giving schedule.Am22Dates), each slot's cutoff-moving records (moving_reader), a launched attempt's last row
    (completion.LastRow), the day a record's sha256 entered the decision log (logged), and whether the run
    directories are read (reads_runs: without them a cutoff date is not recomputed, a draft's case)."""

    def __init__(self, catalogue, freeze, *, registry=None, decision_records=None, attempts=None, run_slot=None,
                 stopped=None, status=None, rule_record=None, e3x=None, entered=None, known_by=None,
                 reference_error=None, am22=None, moving=None, last_row=None, logged=None, reads_runs=False):
        self.cat, self.freeze = catalogue, freeze
        self.registry = registry or {}                      # run stage -> kd_arms.ArmSpec
        self.decision_records = decision_records or {}      # sweep key -> projected decision record
        self.attempts = attempts or (lambda slot: None)     # slot -> its launched run_ids in order, or None
        self.run_slot = run_slot or (lambda run_id: None)   # launched run_id -> its slot, or None
        self.stopped = set(stopped or ())
        self.status = status or (lambda slot: None)          # slot -> completion status at the original instant
        self.rule_record = rule_record or (lambda ref: None)  # a named rule's record ref -> its reading; None: not read
        self.e3x = e3x or (lambda arm: None)                 # arm -> E3-X's run of record; False: none; None: not read
        self.entered = entered or (lambda doc: None)         # cut entry -> its entered day (decision (xv)), or None
        # (path, sha256 or None: any content, day) -> committed by the end of that Manila day; None: not read
        self.known_by = known_by or (lambda rel, sha256, day: None)
        # (an arithmetic's reference, the cut slot, the arithmetic) -> why it does not read again, or None
        self.reference_error = reference_error or (lambda ref, slot, arithmetic: None)
        self.am22 = am22 or (lambda: None)                  # -> schedule.Am22Dates; None: not read
        self.moving = moving or (lambda slot: None)          # slot -> [{gate, record, unread?}]; None: not read
        self.last_row = last_row or (lambda run_id: None)    # run_id -> completion.LastRow; None: not read
        self.logged = logged or (lambda sha256: None)        # a record's sha256 -> the day it entered the decision log
        self.reads_runs = reads_runs


def rule_record_reader(records: R.Records):
    """CutContext.rule_record from the records, for the validator and the printer: the record a named-rule cut
    cites, committed at the cited sha256 (record_uncommitted), read as an AM-23 condition entry (tday_entry/1 of
    kind condition: its arm, verdict and e3x, dated its date) or an AM-18 go-rule record (arm_go_rule/1, projected:
    its arm and verdict, dated the Manila day of its generated_utc); any other record reads as kind "unknown"."""
    def read(ref):
        if not isinstance(ref, dict):
            return {"kind": None, "date": None}
        data = records.read(ref["path"])
        if S.sha256_bytes(data) != ref["sha256"]:
            raise S.TmanRefused("record_uncommitted", f"{ref['path']}: the committed file is not the cited sha256")
        try:
            doc = S.strict_json(data, ref["path"], code="cut_rule_unknown")
        except S.TmanRefused:
            doc = None
        if isinstance(doc, dict) and doc.get("format") == S.ENTRY_FORMAT and doc.get("kind") == "condition" \
                and not entry_field_errors(doc):
            return {"kind": "condition", "date": doc["date"], "arm": doc["arm"], "verdict": doc["verdict"],
                    "e3x": doc["e3x"]}
        if isinstance(doc, dict) and doc.get("format") == _pa().GO_RULE_FORMAT:
            go = R.project(doc, R.GO_RULE_RECORD)
            try:
                return {"kind": "go_rule", "date": D.manila_day(D.parse_utc_any(go["generated_utc"])),
                        "arm": str(go["arm"]), "verdict": go["verdict"]}
            except (KeyError, TypeError, ValueError, AttributeError):
                pass
        return {"kind": "unknown", "date": None}
    return read


def known_by_reader(records: R.Records):
    """CutContext.known_by from the records (decision (xvi)): `rel` committed by the end of `day` (Manila), at
    `sha256` when one is given."""
    def known_by(rel, sha256, day):
        held = records.held_by(rel, D.cutoff_instant(day))
        return bool(held) if sha256 is None else sha256 in held
    return known_by


# ------------------------------------------------------------------------------ the launch cutoff (AM-22 item 2)
R1_TEACHER = "teacher-R1:s42:40k"            # AM-18 item 7(a): R2's teacher waits for R1's end-of-run record
PILOT_SELECTION = "E5:s42:sel"               # its inputs name the pilot's winner record (TB9)
RUN_END_GATES = {"parent_end": "parent", "arm_teacher_selection": "parent", "r1_end": R1_TEACHER}


def moving_reader(cat, records: R.Records, index, pilot_winner: dict | None = None):
    """CutContext.moving (AM-22 item 2(a)): for each of a slot's gates that moves its cutoff (CUTOFF_MOVING_GATES,
    the catalogue's moves_cutoff), the record a registered rule writes, {"path", "sha256", "date"}, dated by its Manila
    day (schedule.manila_day): an end-of-run record -- the parent's, R1's for R2's teacher, the arm teacher's M12
    selection -- by its own time, read under the run directories (`index`, a completion.RunIndex; None: not read);
    the α selection file or the α decision record by the committer time of the commit that added its committed
    bytes; the pilot's winner record, kept uncommitted (DL-139), by the day the RECORDED or IMPLEMENTED decision-log
    row holding its sha256 states after it. A record not written yet is {"record": None} (the run's launch still
    waits for it); one that cannot be read is {"record": None, "unread": why}."""
    def run_end(slot):
        if index is None or index.root is None:
            return {"record": None, "unread": "the run directories are not read (--runs)"}
        att = index.attempts(slot)
        if att is None:
            return {"record": None, "unread": f"{slot}'s attempts are not read"}
        done = sorted((index.state(r).completion.at, r) for r in att if index.state(r).completion is not None)
        if not done:
            return {"record": None}
        c = index.state(done[0][1]).completion
        return {"record": {"path": f"{done[0][1]}/{c.record}", "sha256": c.sha256, "date": D.manila_day(c.at)}}

    def alpha():
        recs = []
        for rel in (ss.ALPHA_SELECTION_REL, ss.DECISION_RECORD_REL["alpha_cwd"]):
            if records.has(rel):
                commit = ss.adding_commit(records.src, rel)
                t = records.commit_time(commit) if commit else None
                if t is None:
                    return {"record": None, "unread": f"{rel}: no commit adds its committed bytes"}
                recs.append({"path": rel, "sha256": S.sha256_bytes(records.read(rel)), "date": D.manila_day(t)})
        return {"record": max(recs, key=lambda r: (r["date"], r["path"])) if recs else None}

    def pilot():
        if pilot_winner is None:
            return {"record": None, "unread": f"the pilot's winner record is not named ({PILOT_SELECTION}'s inputs)"}
        sha, dl = pilot_winner["sha256"], records.decision_log()
        for row in dl.recorded_rows_with(sha):
            text = dl.row_text(row)
            m = S.ISO_DAY_RE.search(text, text.index(sha) + len(sha))
            if m and S.TYPES["day"](m.group(0)):
                return {"record": {"path": pilot_winner["path"], "sha256": sha, "date": m.group(0)}}
        return {"record": None, "unread": f"no RECORDED or IMPLEMENTED decision-log row holding the pilot's winner "
                                          f"record's sha256 {sha[:12]} states its day after it (DL-139)"}

    def read(slot_key):
        s, out = cat.slots[slot_key], []
        for g in s["gates"]:
            if not g["moves_cutoff"]:
                continue
            gate = g["record"]
            if gate in RUN_END_GATES:
                src = s["parent"] if RUN_END_GATES[gate] == "parent" else RUN_END_GATES[gate]
                got = run_end(src) if src else {"record": None, "unread": f"{slot_key} has no parent"}
            elif gate == "alpha_selection":
                got = alpha()
            else:                                                     # pilot_winner
                got = pilot()
            out.append({"gate": gate, **got})
        return out
    return read


def last_row_reader(cat, index):
    """CutContext.last_row (AM-22 item 2(b)): a launched attempt's last time-stamped row (completion.last_row), read
    under the run directories as its own slot's runs are read; None when they are not read."""
    def last(run_id):
        if index is None or index.root is None or run_id not in index.slot_of:
            return None
        S.refuse_test_path(run_id, index.root)
        return C.last_row(cat.slots[index.slot_of[run_id]], index.root / run_id)
    return last


def logged_reader(records: R.Records):
    """CutContext.logged: the Manila day a record's sha256 entered the decision log (entered_day)."""
    return lambda sha256: entered_day(records, sha256)


def slot_cutoff(slot: str, ctx) -> tuple:
    """AM-22 item 2(a) for a run not launched: (its cutoff date, the rule_records that may name what set it, why it is
    not computable). The cutoff date is the later of item 2(a)'s cutoff date and the end of the second day after the
    day of the last record its launch waits for, never past the ceiling; a record it waits for that is not written
    leaves the ceiling (item 2(c): a parent still running exempts nothing). The record that set the date is the
    last-dated one when it moves the date past the cutoff date; otherwise none (null)."""
    dates, reads = ctx.am22(), ctx.moving(slot)
    if dates is None or reads is None:
        return None, None, f"{slot}: AM-22's dates or its cutoff-moving records are not read"
    unread = [f"{slot} {r['gate']}: {r['unread']}" for r in reads if r.get("unread")]
    if unread:
        return None, None, "; ".join(unread)
    absent = any(r["record"] is None for r in reads)
    written = [r["record"] for r in reads if r["record"] is not None]
    last = max((r["date"] for r in written), default=None)
    day = D.cutoff_day(dates, last, absent=absent)
    moved = not absent and last is not None and D.cutoff_day(dates, last) != dates.cutoff
    return day, ([r for r in written if r["date"] == last] if moved else [None]), None


def slot_repeat(doc, slot: str, ctx) -> tuple:
    """AM-22 item 2(b) for a repeat not launched: (the cutoff date, the rule_records allowed, why not computable). The
    date is the third day after the day of the last row the attempt before it wrote -- its slot's last launched
    attempt -- and that row is the record that set it; a teacher fine-tune with no time-stamped line takes the day
    its stop is entered in the decision log, the stop record the cut cites."""
    att = ctx.attempts(slot) or doc["run_ids"]
    if not att:
        return None, None, f"{slot}: no attempt before the repeat is named"
    if ctx.status(slot) == C.DIVERGED:
        return None, None, f"{slot}'s last attempt diverged: no repeat is awaited (AM-7a)"
    rid = att[-1]
    last = ctx.last_row(rid)
    if last is None:
        return None, None, f"{rid}'s rows are not read (--runs)"
    if last.at is not None:
        day = D.manila_day(last.at)
        return D.repeat_cutoff_day(day), [{"path": f"{rid}/{last.record}", "sha256": last.sha256, "date": day}], None
    rr = doc["rule_record"]
    day = ctx.logged(rr["sha256"]) if rr else None
    if day is None:
        return None, None, (f"{rid} wrote no time-stamped line, and the cut cites no stop record entered in the "
                            "decision log (AM-22 item 2(b))")
    return D.repeat_cutoff_day(day), [{"path": rr["path"], "sha256": rr["sha256"], "date": day}], None


def cutoff_rule(doc, ctx) -> tuple:
    """A cutoff cut's (date, rule_records allowed -- None: not compared --, why not computable): item 2(a) for runs
    not launched, item 2(b) for a repeat not launched (launched true, run_ids the earlier attempts). An arm cut from
    its first cut run onward takes that run's date (AM-18 item 7(b); DL-168). Without the run directories (a draft's
    context) the entry's date stands unrecomputed; with them, a date that cannot be computed is None."""
    keys = [k for k in ctx.cat.slots if k in doc["slots"]]
    groups = _arm_groups(doc, ctx)
    if len(groups) == 1 and all(ctx.cat.slots[k]["cut_group"] == groups[0] for k in keys):
        keys = keys[:1]
    got = [slot_repeat(doc, k, ctx) if doc["launched"] else slot_cutoff(k, ctx) for k in keys]
    why = "; ".join(w for _, _, w in got if w)
    if why:
        return (doc["date"], None, None) if not ctx.reads_runs else (None, None, why)
    days = sorted({d for d, _, _ in got})
    if len(days) != 1:
        return None, None, f"the cut's slots have different cutoff dates {days}"
    return days[0], [r for r in got[0][1] if all(r in a for _, a, _ in got)], None


def evidence_reader(records: R.Records):
    """B2R8: a record cited as item 2(f)'s evidence is committed at its sha256 and held by a RECORDED or IMPLEMENTED
    decision-log row."""
    def ok(ref) -> bool:
        if not (isinstance(ref.get("path"), str) and isinstance(ref.get("sha256"), str)):
            return False
        if not records.has(ref["path"]):
            return False
        return S.sha256_bytes(records.read(ref["path"])) == ref["sha256"] \
            and bool(records.decision_log().recorded_rows_with(ref["sha256"]))
    return ok


def no_card_report(doc: dict, evidenced) -> str | None:
    """B2R8 (AM-22 item 2(f)): a cutoff cut whose reason is "no card" prints a REPORT unless its since.record is a
    committed record of item 2(f)'s evidence (`evidenced(ref)`: committed at its sha256, held by a RECORDED or
    IMPLEMENTED decision-log row). Never a refusal: such a deviation "does not undo the cut"."""
    if doc.get("rule") != "cutoff" or doc.get("reason") != "no card":
        return None
    rec = (doc.get("since") or {}).get("record")
    if isinstance(rec, dict) and evidenced(rec):
        return None
    return f"deviation: 'no card' needs item 2(f)'s evidence ({doc['slots']}; AM-22 item 2(f): it does not undo the " \
           "cut)"


def _slots(doc, ctx) -> list[dict]:
    return [ctx.cat.slots[k] for k in doc["slots"]]


def _sweep_key(doc, ctx) -> str | None:
    keys = {s["sweep"]["key"] for s in _slots(doc, ctx) if s["sweep"]}
    return next(iter(keys)) if len(keys) == 1 else None


def rule_day(doc, ctx) -> str | None:
    """The cut's day under its rule (AM-19 items 3(b), 3(c)): the original freeze date; the sweep's decision date;
    the named rule's record date; the card-available day (the Manila day of the arithmetic's time)."""
    rule = doc["rule"]
    if rule == "freeze":
        return ctx.freeze.original_date
    if rule == "decision_date":
        rec = ctx.decision_records.get(_sweep_key(doc, ctx)) or {}
        return rec.get("decision_date")
    if rule == "named_rule":
        rr = doc["rule_record"]
        rec = ctx.rule_record(rr) if rr else None
        return rec["date"] if rec else (rr["date"] if rr else None)
    if rule == "arithmetic":
        a = doc["arithmetic"]
        try:
            return D.manila_day(D.parse_utc_any(a["now_utc"])) if a else None
        except ValueError:
            return None
    if rule == "cutoff":                                    # AM-22 item 2 (B-2): recomputed from the records
        return cutoff_rule(doc, ctx)[0]
    return None


def chk_rule(doc, ctx):
    """A rule of AM-19 item 3(b) or AM-22 item 2 (cutoff, never of the ARM latency measurement: item 1(g)); a named
    rule cites the record that names its cut, read through the records: an AM-23 condition entry or an AM-18 go-rule
    record (rule_record_reader; DL-24 #16)."""
    if doc["rule"] not in S.CUT_RULES:
        return f"rule {doc['rule']!r} is not one of {S.CUT_RULES} (AM-19 3(b))"
    measured = [s["slot"] for s in _slots(doc, ctx) if s["class"] == "measurement"] if doc["rule"] == "cutoff" else []
    if measured:
        return (f"{measured}: the ARM latency measurement falls under no cutoff; it is cut only at the freeze (AM-22 "
                "item 1(g))")
    rec = ctx.rule_record(doc["rule_record"]) if doc["rule"] == "named_rule" else None
    if rec is not None and rec["kind"] not in NAMED_RULE_KINDS:
        return (f"a named rule cites the committed record that names its cut, an AM-23 condition entry or an AM-18 "
                f"go-rule record, not {doc['rule_record']}")
    return None


def chk_never_cut(doc, ctx):
    bad = [s["slot"] for s in _slots(doc, ctx) if s["never_cut"]]
    return f"{bad} are never cut (AM-17 item 9 as AM-19 item 4 reads it)" if bad else None


def chk_complete(doc, ctx):
    if doc["rule"] == "decision_date":                      # TB10: AM-19 item 2(g)'s one exception
        return None
    done = [s["slot"] for s in _slots(doc, ctx) if ctx.status(s["slot"]) == C.COMPLETE]
    return f"{done} are complete at the original freeze instant (AM-19 item 3(b))" if done else None


def chk_reason(doc, ctx):
    """run_ids are the cut slots' launched attempts (the records' and run directories', never the entry's): each
    named, none of another slot, and `launched` true when any exists (AM-19 Definitions; AM-19a readings 6 and 20;
    R-099). A never-launched freeze cut states why, and since when (decision (xiv))."""
    att = [ctx.attempts(k) for k in doc["slots"]]
    unread = any(a is None for a in att)
    launched = sorted({r for a in att if a for r in a})
    omitted = [r for r in launched if r not in doc["run_ids"]]
    if omitted:
        return f"the slots' launched attempts {omitted} are not named in run_ids (AM-19a reading 20)"
    for r in doc["run_ids"]:
        owner = ctx.run_slot(r)
        if owner is not None and owner not in doc["slots"]:
            return f"run_id {r} is a launched attempt of {owner}, not of {doc['slots']}"
        if owner is None and not unread:
            return f"run_id {r} is no launched attempt of {doc['slots']}"
    if doc["launched"] != bool(launched or (unread and doc["run_ids"])):
        return f"the entry says launched {doc['launched']}; the slots' launched attempts are {launched}"
    if doc["launched"] and doc["rule"] != "cutoff":         # a repeat's cutoff cut takes (a)'s entry (AM-22 2(b))
        return None
    if doc["rule"] in ("freeze", "cutoff"):
        if doc["reason"] not in S.NEVER_LAUNCHED_REASONS:
            return (f"a never-launched run's cut gives why: one of {list(S.NEVER_LAUNCHED_REASONS)}, not "
                    f"{doc['reason']!r}")
        since = doc["since"]
        if since is None:
            return "a never-launched run's cut gives since when, with its record (AM-19 item 3(b); decision (xiv))"
        if since["date"] > doc["date"]:
            return f"since {since['date']} is after the cut date {doc['date']}"
    return None


def _cutoff_why(doc, ctx) -> str:
    why = cutoff_rule(doc, ctx)[2] if doc["rule"] == "cutoff" else None
    return f" ({why})" if why else ""


def chk_date(doc, ctx):
    """The cut is dated its rule's day, and is not entered in the decision log before that day (AM-19 item 3(c):
    "entered ... on the day it is made"; a later day is decision (xv)'s REPORT). A cutoff cut (AM-22 item 2) is
    dated its recomputed cutoff date, names as rule_record the record that set it (B2R2), and is entered no earlier
    than the day after that date."""
    want = rule_day(doc, ctx)
    if want != doc["date"]:
        return f"the cut is dated {doc['date']}; its rule's day is {want}" + _cutoff_why(doc, ctx)
    rr = doc["rule_record"] if doc["rule"] == "named_rule" else None
    if rr is not None and rr["date"] != want:
        return f"the cut cites its rule's record as dated {rr['date']}; the record's day is {want}"
    allowed = cutoff_rule(doc, ctx)[1] if doc["rule"] == "cutoff" else None
    if allowed is not None and doc["rule_record"] not in allowed:
        return (f"the cutoff cut names {doc['rule_record']} as the record that set its date; the records give "
                f"{allowed} (AM-22 item 2; B2R2)")
    entered = ctx.entered(doc)
    if entered is not None and entered < want:
        return f"the cut is entered in the decision log on {entered}, before its rule's day {want} (AM-19 item 3(c))"
    if doc["rule"] == "cutoff" and entered is not None and entered < D.expected_entered_day("cutoff", want):
        return (f"the cutoff cut is entered on {entered}, before the day after its cutoff date {want} (AM-22 item "
                "2(a))")
    return None


def chk_arithmetic(doc, ctx):
    """The recorded arithmetic recomputed from its own references and time; its iterations the cut slots'
    (slot_iterations); each reference read again, where the context reads the run directories (ctx.reference_error:
    the same file and figure, from the most recently finished run of the slot's stage; DL-24 #11)."""
    a = doc["arithmetic"]
    if doc["rule"] == "arithmetic" and not (a and a["measured"] and a["cut"] and not doc["launched"]):
        return "an arithmetic cut is a measured arithmetic that cuts a run before its launch (AM-19 item 3(b))"
    if a is None:
        return None
    err = arithmetic_error(a, ctx.freeze.schedule["in_force"])
    if err:
        return err
    steps = next((r["detail"].get("steps") for r in a["references"] if r["source"] == "telemetry"), None)
    want = sorted({slot_iterations(s, steps) for s in _slots(doc, ctx)}, key=str)
    if want != [a["iterations"]]:
        return f"arithmetic: {a['iterations']} iterations; the cut slots' horizons are {want} (AM-19 item 3(b))"
    errs = [e for r in a["references"] for s in _slots(doc, ctx)[:1] for e in [ctx.reference_error(r, s, a)] if e]
    return "; ".join(errs) if errs else None


def chk_val_records(doc, ctx):
    if not doc["launched"]:
        return None
    tel = {t["run_id"] for t in doc["telemetry"]}
    missing = [r for r in doc["run_ids"] if r not in tel]
    if missing or doc["comparator"] is None:
        return (f"a launched cut run needs its telemetry sha256 ({missing} missing) and its comparator "
                f"({doc['comparator']}) (AM-19 item 3(c))")
    return None


def chk_results(doc, ctx):
    """Decision (xvi): the results of record known at that time are records committed by the end of the cut's day,
    at the sha256 cited (each committed at it now: record_uncommitted; its decision-log row: dl_row_missing)."""
    late = [r["path"] for r in doc["results_known"] if ctx.known_by(r["path"], r["sha256"], doc["date"]) is False]
    return (f"{late} were not committed at the cited sha256 by the end of the cut's day {doc['date']}: not results "
            "known at that time (AM-19 item 3(c); decision (xvi))") if late else None


def _arm_groups(doc, ctx) -> list:
    """The arm groups among the entry's slots, in catalogue order (AM-18 item 7; the AM-17 item 11 arm): each is
    checked on its own (DL-24 #15)."""
    return [g for g in ctx.cat.groups if g in ARM_GROUPS and any(s["cut_group"] == g for s in _slots(doc, ctx))]


def arm_records(group: str, ctx) -> list:
    """An arm teacher's records, by path (scripts/preflight_arms.py:119-120): the teacher record and the go-rule
    record of each arm teacher the group's registered runs distil from (AM-18 items 2, 5)."""
    arms = sorted({ctx.registry[s["stage"]].arm_teacher for s in (ctx.cat.slots[k] for k in ctx.cat.group_slots(group))
                   if s["stage"] in ctx.registry and ctx.registry[s["stage"]].arm_teacher})
    return [rel.format(arm=a) for a in arms for rel in (_pa().TEACHER_RECORD_REL, _pa().GO_RULE_REL)]


def chk_arm_results(doc, ctx):
    """An arm's cut lists the arm results of record known at that time: at least one, and its arm teacher's teacher
    and go-rule records when they were committed by the end of the cut's day (AM-18 item 7(a); decision (xvi))."""
    listed = {r["path"] for r in doc["results_known"]}
    for g in _arm_groups(doc, ctx):
        if not listed:
            return f"arm {g}'s cut lists the arm results of record known at that time (AM-18 item 7(a); decision (xvi))"
        missing = [rel for rel in arm_records(g, ctx) if rel not in listed and ctx.known_by(rel, None, doc["date"])]
        if missing:
            return f"arm {g}'s cut does not list {missing}, committed by the end of its day (decision (xvi))"
    return None


def chk_arm_running(doc, ctx):
    """A launched arm run is cut only by the freeze (AM-18 item 7(a)), or, its attempt stopped and its repeat not
    launched, by AM-22 item 2(b)'s cutoff on its date: its slots' launched attempts, each arm on its own."""
    if doc["rule"] == "freeze":
        return None
    if doc["rule"] == "cutoff" and doc["launched"] and rule_day(doc, ctx) == doc["date"]:
        return None
    running = [k for g in _arm_groups(doc, ctx) for k in ctx.cat.group_slots(g) if k in doc["slots"]
               and (ctx.attempts(k) or (ctx.attempts(k) is None and doc["launched"]))]
    if running:
        return (f"{running}: a launched arm run ends only by completion, divergence, a fault or the freeze (AM-18 "
                "item 7(a))")
    return None


def chk_arm_partial(doc, ctx):
    """An arm is cut from its first incomplete run onward (AM-18 item 7(b)); under AM-22 item 2(c) a run cut at its
    cutoff leaves its parent unaffected, so the arm's earlier runs may still be running then."""
    for g in _arm_groups(doc, ctx):
        runs = [k for k in ctx.cat.group_slots(g) if ctx.cat.slots[k]["class"] == "run"]
        cut = [k for k in runs if k in doc["slots"]]
        first = runs.index(cut[0]) if cut else None
        if not cut or cut != runs[first:] or any(ctx.status(k) not in (None, C.COMPLETE) for k in runs[:first]
                                                 if doc["rule"] != "cutoff"):
            return (f"an arm is cut from its first incomplete run onward: arm {g}'s runs {runs}, cut {cut} (AM-18 item "
                    "7(b))")
    return None


def chk_order_position(doc, ctx):
    pos = doc["cut_order_position"]
    groups = {s["cut_group"] for s in _slots(doc, ctx)}
    want = ctx.cat.position(next(iter(groups))) if len(groups) == 1 and None not in groups else None
    if doc["rule"] == "freeze" and want is None:
        return f"a freeze cut names one cut group, not {sorted(map(str, groups))}"
    if doc["rule"] == "freeze" or pos is not None:          # TB10: required only for rule freeze
        return None if pos == want else f"cut_order_position {pos} != the catalogue's {want}"
    return None


def chk_decision_date(doc, ctx):
    """A decision-date cut is the sweep's decision record's (AM-19 items 2(g), 2(h)); a sweep candidate's on-course
    repeat has no AM-22 item 2(b) cutoff (AM-19 item 2(b); AM-19a reading 7; audit m2)."""
    cands = [s["slot"] for s in _slots(doc, ctx) if s["sweep"] and not s["sweep"]["default_candidate"]] \
        if doc["rule"] == "cutoff" and doc["launched"] else []
    if cands:
        return (f"{cands}: a sweep candidate's repeat is governed by its decision date; it has no item 2(b) cutoff "
                "(AM-22 item 2(b); AM-19 item 2(b); AM-19a reading 7)")
    if doc["rule"] != "decision_date":
        return None
    key = _sweep_key(doc, ctx)
    if key is None or any(not s["sweep"] or s["sweep"]["default_candidate"] or s["sweep"]["of_record"]
                          for s in _slots(doc, ctx)):
        return "a decision-date cut names non-default candidates of one sweep (AM-19 item 2(b))"
    rec = ctx.decision_records.get(key)
    cut_values = {float(v["value"]) for v in (rec or {}).get("values", []) if v.get("status") == "cut"}
    vals = {float(s["sweep"]["value"]) for s in _slots(doc, ctx)}
    if rec is None or not vals <= cut_values:
        return f"the sweep's decision record does not cut {sorted(vals)} (AM-19 items 2(g), 2(h))"
    return None


def chk_stopped(doc, ctx):
    """A stopped run awaiting a repeat (AM-19 item 3(a)) is cut only by the freeze, on the original freeze date, or,
    a sweep candidate, at its decision date when the sweep's decision record cuts it (AM-19 item 2(b); the record
    is decision_date_cut_mismatch's, checked before this)."""
    stopped = sorted(r for r in doc["run_ids"] if r in ctx.stopped)
    if not stopped or (doc["rule"] == "freeze" and doc["date"] == ctx.freeze.original_date) \
            or doc["rule"] == "decision_date":
        return None
    if doc["rule"] == "cutoff" and doc["launched"] and rule_day(doc, ctx) == doc["date"]:     # AM-22 item 2(b)
        return None
    return (f"{stopped} await a repeat (AM-19 item 3(a)); only the freeze, on the original freeze date, the sweep's "
            "decision record, or AM-22 item 2(b)'s cutoff on its date cuts them")


def chk_registered_arm(doc, ctx):
    groups = [g for g in _arm_groups(doc, ctx) if g in ("R2", "W", "R1")]
    stages = sorted({s["stage"] for s in _slots(doc, ctx) if s["cut_group"] in groups and s["class"] == "run"
                     and s["stage"].startswith("E3-")})
    missing = [st for st in stages if st not in ctx.registry]
    return f"{missing} have no registry file (configs/kd_arms; AM-17b item 4)" if missing else None


def chk_condition(doc, ctx):
    """A named-rule cut is the cut its record names (AM-19 item 3(b)): E6-X alone, by an AM-23 condition entry of arm
    X reading no whose E3-X run is E3-X's run of record (AM-23 item 3(b)); an arm's runs, by its arm teacher's AM-18
    go-rule record reading no_go (AM-18 item 5). Each arm needs its own record (DL-24 #16)."""
    if doc["rule"] != "named_rule":
        return None
    rec = ctx.rule_record(doc["rule_record"])
    if rec is None:                                         # not read: a context without the records
        return None
    if rec["kind"] == "condition":
        arm = rec["arm"]
        if [s["stage"] for s in _slots(doc, ctx)] != [f"E6-{arm}"] * len(doc["slots"]) or rec["verdict"] != "no":
            return (f"an AM-23 condition entry of arm {arm} reading {rec['verdict']!r} cuts E6-{arm} alone, when it "
                    f"reads no; the cut names {doc['slots']}")
        got = ctx.e3x(arm)
        if got is False or (got and (rec["e3x"]["run_id"], rec["e3x"]["checkpoint_sha256"])
                            != (got["run_id"], got["checkpoint_sha256"])):
            return f"the condition entry's E3-{arm} run {rec['e3x']['run_id']} is not E3-{arm}'s run of record ({got})"
        return None
    groups = sorted({str(s["cut_group"]) for s in _slots(doc, ctx)})       # E6-X's group is never an arm's
    if rec["verdict"] != "no_go" or groups != [rec["arm"]]:
        return (f"an AM-18 go-rule record of arm {rec['arm']} reading {rec['verdict']!r} cuts that arm's runs, when "
                f"it reads no_go; the cut names {doc['slots']}")
    return None


CUT_CHECKS = [("cut_rule_unknown", chk_rule), ("cut_of_never_cut", chk_never_cut),
              ("cut_of_complete_run", chk_complete), ("cut_reason_invalid", chk_reason),
              ("cut_date_mismatch", chk_date), ("cut_arithmetic_invalid", chk_arithmetic),
              ("cut_val_record_missing", chk_val_records), ("cut_results_unknown", chk_results),
              ("arm_cut_results_missing", chk_arm_results),
              ("arm_run_cut_while_running", chk_arm_running), ("arm_cut_partial", chk_arm_partial),
              ("cut_order_position_mismatch", chk_order_position), ("decision_date_cut_mismatch", chk_decision_date),
              ("stopped_run_cut_early", chk_stopped), ("unregistered_arm", chk_registered_arm),
              ("condition_unmet", chk_condition)]


def check_slots(doc: dict, cat, *, kinds_of_class: tuple) -> None:
    """The entry names known slots (slot_unknown) of the classes its kind applies to (derived_counted_as_run)."""
    unknown = [k for k in doc["slots"] if k not in cat.slots]
    if unknown:
        raise S.TmanRefused("slot_unknown", f"{unknown} are not catalogue slots")
    wrong = [k for k in doc["slots"] if cat.slots[k]["class"] not in kinds_of_class]
    if wrong:
        raise S.TmanRefused("derived_counted_as_run", f"{wrong}: a {doc['kind']} entry names slots of the classes "
                                                      f"{list(kinds_of_class)} only")


def check_cut(doc: dict, ctx: CutContext, passed: list | None = None) -> None:
    """Every cut check in gate order; `passed` collects the codes that passed (the smokes' witness that a refusal
    is reached with every earlier gate passing)."""
    require_fields(doc)
    check_slots(doc, ctx.cat, kinds_of_class=("run", "measurement"))
    for code, fn in CUT_CHECKS:
        err = fn(doc, ctx)
        if err:
            raise S.TmanRefused(code, f"{doc['slots']}: {err}")
        if passed is not None:
            passed.append(code)


# --------------------------------------------------------------------------------- the parent rule (AM-19 4(a))
NO_MODEL_KINDS = ("cut", "diverged", "not_run", "aborted", "non_finite", "superseded", "not_made")


def root_cause(slot: str, by_slot: dict, cat) -> str | None:
    """"parent cut" or "parent diverged" when `slot`'s parent chain ends in a cut or a divergence (a not-run parent
    passes its own cause on); None when the parent has a model or no entry."""
    parent, seen = cat.slots[slot]["parent"], set()
    while parent and parent not in seen:
        seen.add(parent)
        e = by_slot.get(parent)
        if e is None or e["kind"] not in ("cut", "diverged", "not_run"):
            return None
        if e["kind"] != "not_run":
            return f"parent {e['kind']}"
        parent = cat.slots[parent]["parent"]
    return None


def parent_record_sha256(e: dict) -> str | None:
    """The record a child's not-run entry cites for its parent's entry: a cut's or a not-run's entry file, a
    divergence's telemetry."""
    if e["kind"] in ("cut", "not_run"):
        return e["entry"]["sha256"]
    if e["kind"] == "diverged":
        return e["telemetry_sha256"]
    return None


def parent_rule_errors(by_slot: dict, cat) -> list[str]:
    """parent_rule_violation: a scored, derived or listed entry under a parent without a model."""
    errs = []
    for slot, e in sorted(by_slot.items()):
        p = cat.slots[slot]["parent"]
        pe = by_slot.get(p) if p else None
        if pe is not None and pe["kind"] in NO_MODEL_KINDS and e["kind"] not in S.TERMINAL_KINDS:
            errs.append(f"{slot} is {e['kind']} under {p}, which is {pe['kind']} (AM-19 item 4(a))")
    return errs


def not_run_errors(by_slot: dict, cat) -> list[str]:
    """not_run_without_parent_record: a not-run entry whose parent chain is neither cut nor diverged, or whose
    reason, parent or parent record differs from that chain's."""
    errs = []
    for slot, e in sorted(by_slot.items()):
        if e["kind"] != "not_run":
            continue
        p = cat.slots[slot]["parent"]
        pe = by_slot.get(p) or {"kind": "absent"}
        cause = root_cause(slot, by_slot, cat)
        if cause is None or e["reason"] != cause or e["parent"] != p \
                or e["parent_record"]["sha256"] != parent_record_sha256(pe):
            errs.append(f"{slot}: not run ({e['reason']}, parent {e['parent']}) although its parent {p} is "
                        f"{pe['kind']} (AM-19 item 4(a))")
    return errs


def descendant_errors(by_slot: dict, cat) -> list[str]:
    """descendant_entry_missing: a cut, diverged or not-run parent whose descendants at that seed are not entered as
    not run. Two exceptions: a QAT selection has no entry when its run yields no model (IF§11.2; readers take the
    run's kind), and an arm's runs cut together by one entry, from its first incomplete run onward (AM-18 item
    7(b))."""
    errs = []
    for slot, e in sorted(by_slot.items()):
        if e["kind"] not in ("cut", "diverged", "not_run"):
            continue
        for child in cat.children.get(slot, []):
            if cat.slots[child]["record_kind"] == "qat_selection":
                continue
            ce = by_slot.get(child) or {}
            same_cut = e["kind"] == ce.get("kind") == "cut" and ce["entry"]["sha256"] == e["entry"]["sha256"]
            if ce.get("kind") != "not_run" and not same_cut:
                errs.append(f"{child} is not entered as not run although its parent {slot} is {e['kind']} "
                            "(AM-19 item 4(a))")
    return errs


# ------------------------------------------------------------------------------------------------ the printers
def file_id_of_entry(doc: dict) -> str:
    """The entry file's file_id: its first slot's, or "session" for an entry about no slot."""
    return S.file_id_of(doc["slots"][0]) if doc["slots"] else S.SESSION_FILE_ID


def make_entry(kind: str, slots: list, date: str, fields: dict, *, amendment: str, script: str,
               commit: str | None, corrects: dict | None = None, push_evidence: dict | None = None) -> dict:
    """A tday_entry/1 document, fields checked (entry_kind_unknown, entry_field). `fields` are the kind's own: the
    common keys (kind, slots, date, printed_by, ...) are the printer's (DL-24 #71)."""
    entry_spec(kind)
    common = sorted(set(fields) & set(ENTRY_FILE_COMMON)) if isinstance(fields, dict) else []
    if common or not isinstance(fields, dict):
        raise S.TmanRefused("entry_field", f"the fields name {common or fields!r}: they hold the kind's own fields, "
                                           "never the printer's common keys")
    doc = {"format": S.ENTRY_FORMAT, "kind": kind, "slots": list(slots), "date": date, "amendment": amendment,
           "printed_by": {"script": script, "commit": commit}, "corrects": corrects, "push_evidence": push_evidence,
           **fields}
    require_fields(doc)
    return doc


def entry_name(doc: dict, nn: int = 1) -> str:
    return S.entry_file_name(doc["kind"], file_id_of_entry(doc), doc["date"], nn)


def dl_row(doc: dict, rel: str, sha256: str, summary: str) -> str:
    """The decision-log row the orchestrator enters (the ID is assigned then): its Status starts with RECORDED and
    its text names the entry file's path and sha256 (decision (iii))."""
    who = ", ".join(doc["slots"]) or "the session"
    return (f"| DL-<next> | {ROW_STATUS} ({doc['date']}) | TMAN {doc['kind']} entry for {who}: {summary} Entry file "
            f"{rel}, sha256 {sha256} ({doc['amendment']}). | {rel} | no |")


def write_entry(doc: dict, out_dir: Path, *, nn: int = 1, records: R.Records | None = None) -> tuple[Path, str, str]:
    """Write the entry file exclusively into `out_dir`, outside the repository (the name it creates obeys IF§2: the
    folder is the operator's, a Windows drive included; DL-24 #72); refuse a name the records already hold. Returns
    (path, repository path, sha256)."""
    name = entry_name(doc, nn)
    rel = S.entry_file_rel(name)
    err = S.created_name_error(name)
    if err:
        raise S.TmanRefused("test_path", f"refusing to create {name}: {err}")
    if records is not None and records.has(rel):
        raise S.TmanRefused("entry_file_rewritten", f"{rel} exists in the records: an entry file is written once")
    data = S.json_file_bytes(doc)
    out = Path(out_dir) / name
    try:
        with open(out, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        raise S.TmanRefused("output_exists", f"{out} exists: an output is created once") from None
    return out, rel, S.sha256_bytes(data)


def not_run_entry_errors(doc: dict, cat, parent_entry: dict | None) -> list[str]:
    """A not-run entry file against its parent's manifest entry (not_run_without_parent_record): every slot's
    catalogue parent is the entry's parent; the reason is the parent's cut or divergence, or the cause a not-run
    parent passes on; the parent record is the parent's entry file or divergence telemetry."""
    errs = [f"{k}: its catalogue parent is {cat.slots[k]['parent']}, not {doc['parent']}" for k in doc["slots"]
            if cat.slots[k]["parent"] != doc["parent"]]
    pe = parent_entry or {"kind": "absent"}
    want = {"cut": "parent cut", "diverged": "parent diverged"}.get(pe["kind"], pe.get("reason"))
    if pe["kind"] not in ("cut", "diverged", "not_run") or doc["reason"] != want:
        errs.append(f"the parent {doc['parent']} is {pe['kind']}; a not-run entry follows a cut or diverged parent "
                    f"(reason {doc['reason']!r}, AM-19 item 4(a))")
    elif doc["parent_record"]["sha256"] != parent_record_sha256(pe):
        errs.append(f"parent_record {doc['parent_record']['sha256']} is not the parent's record")
    return errs


def freeze_delay_errors(doc: dict, freeze, late: list) -> list[str]:
    """AM-19 item 1(d): the delay entry is dated the original freeze date and lists exactly the never-cut runs not
    complete at the original instant, `late` (validate.late_never_cut, the one reading of item 1(d)), at least one
    (freeze_source_conflict)."""
    errs = []
    if doc["date"] != freeze.original_date or doc["original_date"] != freeze.original_date:
        errs.append(f"the delay is entered on {doc['date']} for {doc['original_date']}, not on the original freeze "
                    f"date {freeze.original_date}")
    if not late or sorted(doc["runs"]) != late:
        errs.append(f"the delay lists {sorted(doc['runs'])}; the never-cut runs not complete at the original freeze "
                    f"instant are {late}")
    return errs


def new_freeze_errors(doc: dict, delay_ref: dict, instants: list, original_date: str) -> list[str]:
    """AM-19 item 1(d): the new freeze is the first later date at whose end every never-cut run is complete, entered
    on that day, citing the delay entry (freeze_source_conflict). `instants` are those runs' completion times."""
    want = D.first_complete_day(instants, original_date)
    errs = []
    if want is None or doc["new_date"] != want or doc["date"] != doc["new_date"]:
        errs.append(f"new freeze {doc['new_date']} entered on {doc['date']}; the first later date at whose end every "
                    f"never-cut run is complete is {want}")
    if doc["delay_entry"] != {"path": delay_ref["path"], "sha256": delay_ref["sha256"]}:
        errs.append("the new freeze does not cite the delay entry")
    return errs


# --------------------------------------------------------------------------------- committed entry files (iii)
def entry_name_error(rel: str, doc: dict) -> str | None:
    """TB7: an entry file lies in reports/derived/tday_entries/ under <kind>__<file_id>__<YYYYMMDD>__<nn>.json, its
    kind, file_id and date its own."""
    p = PurePosixPath(str(rel))
    m = S.ENTRY_FILE_RE.fullmatch(p.name)
    if str(p.parent) != S.ENTRIES_DIR_REL or not m or m.group("kind") != doc["kind"] \
            or m.group("date") != doc["date"].replace("-", "") or m.group("file_id") != file_id_of_entry(doc):
        return f"{rel}: its folder and name do not state its kind, file_id and date ({S.ENTRIES_DIR_REL}; TB7)"
    return None


def entry_file_errors(doc, kind: str, rel: str) -> list[str]:
    """A cited entry file is a tday_entry/1 of the kind it is cited as: its fields and its name."""
    try:
        require_fields(doc)
    except S.TmanRefused as e:
        return [f"{rel}: {e.detail}"]
    if doc["kind"] != kind:
        return [f"{rel}: a {doc['kind']} entry, cited as a {kind} entry"]
    err = entry_name_error(rel, doc)
    return [err] if err else []


KD_ARMS_REL = "configs/kd_arms"


def arm_registry(records: R.Records) -> dict:
    """The KD arm registry at S, the checkout the records name (DL-24 #95): kd_arms.load_arms over its
    configs/kd_arms/, each file committed and byte-equal to its blob there; {run_stage: ArmSpec}. A registry kd_arms
    refuses is unregistered_arm."""
    from src.training import kd_arms
    try:
        arms = kd_arms.load_arms(Path(records.src.root) / KD_ARMS_REL)
    except kd_arms.ArmSpecError as e:
        raise S.TmanRefused("unregistered_arm", f"the arm registry at S: {e}") from None
    for spec in arms.values():
        records.read(f"{KD_ARMS_REL}/{Path(spec.path).name}")
    return {s.run_stage: s for s in arms.values()}


def read_entry_file(records: R.Records, rel: str, kind: str | None = None) -> tuple[dict, str]:
    """A committed entry file: committed and byte-equal (record_uncommitted), written once (entry_file_rewritten),
    its sha256 in a RECORDED or IMPLEMENTED decision-log row (dl_row_missing), its fields and name (entry_field),
    and, when given, of `kind` (entry_file_invalid)."""
    S.refuse_test_path(rel)
    data = records.read(rel)
    err = records.written_once_error(rel)
    if err:
        raise S.TmanRefused("entry_file_rewritten", err)
    sha = S.sha256_bytes(data)
    if not records.decision_log().recorded_rows_with(sha):
        raise S.TmanRefused("dl_row_missing", f"{rel}: sha256 {sha} is in no RECORDED or IMPLEMENTED decision-log row")
    doc = S.strict_json(data, rel, code="entry_field")
    require_fields(doc)
    err = entry_name_error(rel, doc)
    if err:
        raise S.TmanRefused("entry_field", err)
    if kind is not None and doc["kind"] != kind:
        raise S.TmanRefused("entry_file_invalid", f"{rel}: a {doc['kind']} entry, cited as a {kind} entry")
    return doc, sha


def entered_day(records: R.Records, sha256: str) -> str | None:
    """The Manila day of the earliest commit, in the history of the records ref, whose decision log holds `sha256`
    (decision (xv); git log -S)."""
    out = records.src._out("log", "--format=%H %ct", f"-S{sha256}", records.ref, "--", S.DECISION_LOG_REL)
    times = [int(ln.split()[1]) for ln in (out or "").splitlines() if len(ln.split()) == 2]
    return D.manila_day(min(times)) if times else None


def late_entry_report(doc: dict, entered: str | None) -> str | None:
    """REPORT "deviation: entry later than its day" (decision (xv); T-15): entered after the day its rule makes it
    due. Never a refusal."""
    if entered is None:
        return None
    due = D.expected_entered_day(doc.get("rule", ""), doc["date"])
    return None if entered <= due else f"deviation: entry later than its day ({doc['kind']} {doc['slots']}: due " \
                                       f"{due}, entered {entered})"
