"""AM-22 item 1(a)'s check beside the validator: the registered runs, their settled ways and the five conditions
(lane TMAN, B-2; AM-22 Code, docs/PREREGISTRATION_AMENDMENTS.md:2453-2458).

D is the first date, no later than 2026-11-02, at whose end five conditions hold (item 1(a), :2314-2335). For a date X
the check reads the records committed by the end of X (audit M3): C_X, the newest commit on the records ref's
first-parent chain whose committer time meets the end of X (sweep_select.meets), and the decision log, the entry
files, the lane register, the draft manifest, the trigger record and the launch log as their blobs at C_X
(records.Records.at_commit); the run directories by their own timestamps (completion). It prints each condition MET
with its records -- a path, a DL id or a merge commit, a file's sha256, its date, and for a committed record the
commit that holds it and that commit's committer time (audit M2) -- or MISSING with what is missing, and REPORT lines
that never fail it. It decides nothing and writes nothing. A commit the orchestrator confirmed reached the remote on
a later day than its committer day (--pushed COMMIT=DAY, from the remote's activity; audit M2, DL-24 R2-4) counts
from that day: C_X stops before it and every first-parent commit after it, and a record it holds is dated that day;
the freeze entry carries each such confirmation as a record of kind "confirmed push", and the unlock re-check reads
them from there.

Each condition reads records only: statuses, dates, hashes, merge records, the trigger's verdict (L-AM3's reader,
records.trigger_reader; B2R13) and the validator's verdict. An arm's go-rule record and an AM-23 condition entry are
read for their verdicts, never their values; no score is read (B2R11).

Registered runs (Definitions, :2266-2280; B2R3): the catalogue's run slots -- E1 at seeds 43/44 and the lambda
candidates, registered by AM-1 and AM-2 in no cut order, among them (audit n3) -- less those a committed record rules
out: E6-KD unless AM-3's trigger record reads triggered; an E3-X whose go-rule record reads no_go or not_evaluable
(AM-18 item 5), with its E6-X (AM-23 item 3(c); audit m7); an E6-X whose condition entry reads no (AM-23 item 3(c)).
The SegNeXt-L KD runs would leave on an AM-17 item 11(b) no-go, of which no record exists (DL-129); K8 and B-avg have
no slot; AM-7's full consequence would register E1 reruns that the catalogue does not hold, so condition 1 names that
STOP. Derived and measurement slots are not runs (item 1(g)).

Settled at X (Definitions, :2282-2310; completion.SETTLED_WAYS): complete at the end of X, or entered as superseded;
a cuttable run also entered as diverged, as aborted (rule, cause), as cut by a rule other than the freeze, or as not
run. A never-cut run settles only by completion or supersession: one entered as aborted is a STOP that only a new
amendment settles, and a record of such an amendment is a STOP to ask. "Entered by the end of X": a body entry
(superseded, diverged, aborted) of the draft at C_X by its dl_rows in the decision log at C_X; an entry file (cut,
not run) by a RECORDED or IMPLEMENTED row at C_X holding its sha256 (decision (iii); entries.read_entry_file), which
is AM-22's "entered in the decision log" read through the entry file (audit n7).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import timedelta

from src.training import sweep_select as ss

from . import completion as C
from . import entries as E
from . import manifest as M
from . import records as R
from . import schedule as D
from . import schema as S

AM22_IN_FORCE = "2026-10-17"          # its commit reached the remote then (DL-137, DL-168; B2R1): the first date read
LAST_D = "2026-11-02"                 # item 1(a): D is no later than this; after it item 1(f) applies
FALLBACK_FREEZE = "2026-11-03"        # item 1(f): the freeze of AM-19 as committed
ARM_LANE = "L-AM16-ARM"               # item 1(g): condition 4 does not read the lane that serves only ARM latency
TRIGGER_DUE_DAYS = 2                  # item 1(a)2: entered by the end of the second day after E6 seed 42 is complete
E6_S42, E6_S42_SEL, E6_KD = "E6:s42:15ep", "E6:s42:sel", "E6-KD:s42:15ep"
GO_RULE_ARMS = ("R1", "R2")           # AM-18 item 5's arms; the L arm's go rule has no record (DL-129)
EXCLUDING_VERDICTS = ("no_go", "not_evaluable")                  # AM-18 item 5: E3-X is not run
BODY_SETTLED = {"superseded": C.SETTLED_SUPERSEDED, "diverged": C.SETTLED_DIVERGED, "aborted": C.SETTLED_ABORTED}
ENTRY_PATH_RE = re.compile(re.escape(S.ENTRIES_DIR_REL) + r"/[A-Za-z0-9._-]+\.json")
STOP_TEXT = ("entered as aborted (rule, cause): a STOP that only a new amendment settles (AM-22 Definitions; AM-19 "
             "item 3(a)); a record of such an amendment is a STOP to ask")
CONFIRMED_PUSH = "confirmed push"     # a freeze entry's record of a commit's confirmed later push day (audit M2)


def previous_day(day: str) -> str:
    return D.iso(D.day(day) - timedelta(days=1))


def commit_at_end(src: ss.HeadSource, day: str, pushed: dict | None = None) -> str | None:
    """C_X (audit M3): the newest commit on the first-parent chain of the records ref whose committer time meets the
    end of `day`, older than every first-parent commit whose confirmed push day (`pushed`: {commit: day}; audit M2) is
    later than `day`; None when there is none."""
    cutoff, pushed = D.cutoff_instant(day), pushed or {}
    chain = [(c, int(t)) for c, _, t in (ln.partition(" ") for ln in (src._out(
        "log", "--first-parent", "--format=%H %ct", src.records_ref) or "").splitlines()) if t.isdigit()]
    late = max((i for i, (c, _) in enumerate(chain) if pushed.get(c, day) > day), default=-1)
    return next((c for c, t in chain[late + 1:] if D.meets(t, cutoff)), None)


# ------------------------------------------------------------------------------------------- the records relied on
class Rec:
    """A record a condition relies on: its kind, its reference (a path, a DL id, a run's record or a merge commit),
    its sha256 (a file's, else None) and its date, the Manila day of its own time (a run's end-of-run record) or of
    the commits that hold it, the latest of them: a committed file's adding commit, the commit that entered its
    decision-log row (audit M2). Its commits are read when it is printed."""

    def __init__(self, view, kind: str, ref: str, sha256: str | None = None, *, how: str | None = None,
                 date: str | None = None):
        self.view, self.kind, self.ref, self.sha256, self.how, self._date = view, kind, ref, sha256, how, date
        self._commits = None

    @property
    def commits(self) -> tuple:
        """((what, commit, committer time), ...): the commits that hold the record at C_X."""
        if self._commits is None:
            self._commits = self.view.commits_of(self)
        return self._commits

    @property
    def date(self) -> str:
        """Its own day, else the latest Manila day of its commits, each its confirmed push day when later (audit M2)."""
        if self._date is not None:
            return self._date
        days = [max(D.manila_day(t), self.view.pushed.get(c, "")) for _, c, t in self.commits if t is not None]
        return max(days) if days else self.view.day

    def entry(self) -> dict:
        """The freeze entry's form of a record (B2R5): {kind, ref, sha256, date}."""
        return {"kind": self.kind, "ref": self.ref, "sha256": self.sha256, "date": self.date}


@dataclass
class Condition:
    number: int
    records: list = field(default_factory=list)      # [Rec]
    missing: list = field(default_factory=list)      # what is missing, each a line
    reports: list = field(default_factory=list)      # REPORT lines: they never fail the condition
    notes: list = field(default_factory=list)

    @property
    def met(self) -> bool:
        return not self.missing


TEXTS = {1: "every registered run is settled",
         2: "AM-3's trigger is evaluated and entered, E6-KD settled if it fired, E6 seed 42's derived artifacts made",
         3: "E6 seed 42's conversion and selection are hashed in the draft",
         4: "every lane serving TEST is IMPLEMENTED or entered as not required",
         5: "the draft passes the validator in freeze mode, with the evaluator commit among its entries"}


# ------------------------------------------------------------------------------------------ the records at X
class View:
    """The records committed by the end of `day` (C_X and its blobs; audit M3; `pushed`, the confirmed push days of
    audit M2) and the run directories under `runs_root`, their attempts named by the launch log as the records hold
    it now (`records`; a run's own records judge it at X: DL-24 R1-F5); `reader` is the trigger reader
    (records.trigger_reader). Every reading is cached per date; `states` shares the run states over dates."""

    def __init__(self, records: R.Records, cat, runs_root, day: str, *, reader=None, states: dict | None = None,
                 pushed: dict | None = None):
        self.day, self.cutoff, self.cat, self.runs_root = day, D.cutoff_instant(day), cat, runs_root
        self.head, self.pushed = records, dict(pushed or {})
        self.commit = commit_at_end(records.src, day, self.pushed)
        self.records = records.at_commit(self.commit) if self.commit else None
        self.committed = records.commit_time(self.commit) if self.commit else None
        self.reader, self.states = reader, ({} if states is None else states)
        self.notes: list = []
        self._cache: dict = {}

    def _once(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def dl(self) -> R.DecisionLog:
        return self.records.decision_log()

    # ----------------------------------------------------------------------------------------- commits (M2)
    def line_commit(self, text: str) -> tuple:
        """The newest commit at or before C_X whose change made the decision log hold `text` (git log -S), with its
        committer time; (None, None) when there is none."""
        def find():
            out = self.records.src._out("log", "--format=%H %ct", f"-S{text}", self.records.ref, "--",
                                        S.DECISION_LOG_REL)
            for ln in (out or "").splitlines():
                c, _, t = ln.partition(" ")
                if t.isdigit():
                    return c, int(t)
            return None, None
        return self._once(("line", text), find)

    def sha_commit(self, sha256: str) -> tuple:
        """The earliest commit at or before C_X whose decision log holds `sha256`, the commit that entered it
        (entries.entered_day's rule, decision (xv)), with its committer time."""
        def find():
            out = self.records.src._out("log", "--format=%H %ct", f"-S{sha256}", self.records.ref, "--",
                                        S.DECISION_LOG_REL)
            hits = [(int(t), c) for c, _, t in (ln.partition(" ") for ln in (out or "").splitlines()) if t.isdigit()]
            return (min(hits)[1], min(hits)[0]) if hits else (None, None)
        return self._once(("sha", sha256), find)

    def file_commit(self, rel: str) -> tuple:
        """The newest commit at or before C_X that set `rel` to its blob there, with its committer time: a records file
        committed again, such as the draft built for each candidate day (audit m5), is dated by its last change
        (DL-24 R2-6)."""
        def find():
            out = self.records.src._out("log", "-1", "--no-renames", "--format=%H %ct", self.records.ref, "--", rel)
            c, _, t = (out or "").strip().partition(" ")
            return (c, int(t)) if t.isdigit() else (None, None)
        return self._once(("file", rel), find)

    def commits_of(self, rec: Rec) -> tuple:
        out = []
        if rec.how == "row":
            c, t = self.line_commit(self.dl().row_text(rec.ref) or rec.ref)
            out.append((f"row {rec.ref}", c, t))
        if rec.how in ("file", "entry"):
            c, t = self.file_commit(rec.ref)
            out.append(("file", c, t))
        if rec.how in ("entry", "file") and rec.sha256 and self.dl().recorded_rows_with(rec.sha256):
            c, t = self.sha_commit(rec.sha256)
            out.append((f"row {','.join(self.dl().recorded_rows_with(rec.sha256))}", c, t))
        if rec.how == "commit":
            out.append(("commit", rec.ref, self.records.commit_time(rec.ref)))
        return tuple((w, c, t) for w, c, t in out if c)

    # ----------------------------------------------------------------------------------------- the records
    def draft(self) -> tuple:
        """The draft committed by the end of X, read as its blob at C_X (audit m5): ({doc, data, by_slot} or None,
        why not, its record)."""
        def read():
            rel = S.MANIFEST_REL
            if not self.records.has(rel):
                return None, f"the draft {rel} is not committed by the end of {self.day}", None
            data = self.records.read(rel)
            try:
                doc = R.strict_json(data, rel, code="manifest_unreadable")
            except S.TmanRefused as e:
                return None, f"the draft {rel} does not parse ({e.detail})", None
            body = doc.get("body") if hasattr(doc, "keys") else None
            ents = body.get("entries") if hasattr(body, "keys") else None
            if not (isinstance(ents, list) and all(hasattr(e, "keys") and isinstance(e.get("slot"), str)
                                                   for e in ents)):
                return None, f"the draft {rel} holds no body with its entries", None
            return ({"doc": doc, "data": data, "by_slot": {e["slot"]: e for e in ents}}, None,
                    Rec(self, "draft manifest", rel, S.sha256_bytes(data), how="file"))
        return self._once("draft", read)

    def by_slot(self) -> dict:
        d = self.draft()[0]
        return d["by_slot"] if d else {}

    def entry_files(self) -> list:
        """Every entry file a RECORDED or IMPLEMENTED decision-log row at C_X names, read as entries.read_entry_file
        reads it (committed, written once, its sha256 in such a row, its fields and its name), less those a later
        recorded entry corrects: [{path, sha256, doc}]."""
        def read():
            paths = sorted({m.group(0) for st, line in self.dl().rows.values()
                            if st.startswith(("RECORDED", "IMPLEMENTED")) for m in ENTRY_PATH_RE.finditer(line)})
            out = []
            for rel in paths:
                try:
                    doc, sha = E.read_entry_file(self.records, rel)
                except S.TmanRefused as e:
                    self.notes.append(f"{rel}: not an entry of record at {self.commit[:12]} ([{e.code}] {e.detail})")
                    continue
                out.append({"path": rel, "sha256": sha, "doc": doc})
            corrected = {d["doc"]["corrects"]["sha256"] for d in out if d["doc"]["corrects"]}
            return [d for d in out if d["sha256"] not in corrected]
        return self._once("entries", read)

    def index(self) -> C.RunIndex:
        """Each run slot's launched attempts: the launch log as the records hold it now and the run directories
        (completion.RunIndex); each attempt is judged at X by its own records (DL-24 R1-F5)."""
        def build():
            log = self.head.launch_log(code="completion_record_invalid") if self.head.has(ss.LAUNCH_LOG_REL) else None
            return C.RunIndex(self.cat, log, self.runs_root, states=self.states)
        return self._once("index", build)

    def run_state(self, slot: str, run_id: str) -> C.RunState:
        idx = self.index()
        if run_id in idx.slot_of:
            return idx.state(run_id)
        if run_id not in self.states:                       # the teacher of record: its draft entry names its run
            S.refuse_test_path(run_id, self.runs_root)
            self.states[run_id] = C.run_state(self.cat.slots[slot], self.runs_root / run_id)
        return self.states[run_id]

    def attempts(self, slot: str) -> list | None:
        """The slot's launched attempts; for a slot no launcher names (the teacher of record), its draft entry's
        run_id; None when the run directories are not read."""
        att = self.index().attempts(slot)
        if att is None and self.runs_root is not None and self.cat.slots[slot]["class"] == "run":
            rid = (self.by_slot().get(slot) or {}).get("run_id")
            att = [rid] if isinstance(rid, str) and rid else []
        return att

    def completion(self, slot: str) -> tuple | None:
        """(run_id, completion.Completion) of the slot's earliest attempt complete at the end of X, or None."""
        done = sorted((st.completion.at, rid, st.completion) for rid in (self.attempts(slot) or [])
                      for st in [self.run_state(slot, rid)] if st.complete_at(self.cutoff))
        return (done[0][1], done[0][2]) if done else None

    # ----------------------------------------------------------------------------------- verdicts of record
    def trigger(self) -> tuple:
        """AM-3's trigger at C_X: (triggered -- None when it is not evaluated and entered --, its records, what is
        missing). The record is records.TRIGGER_RECORD_REL, committed, its sha256 in a RECORDED or IMPLEMENTED row,
        read by the trigger reader for its verdict only."""
        def read():
            rel = R.TRIGGER_RECORD_REL
            if not self.records.has(rel):
                return None, [], f"AM-3's trigger record {rel} is not committed by the end of {self.day}"
            data = self.records.read(rel)
            sha = S.sha256_bytes(data)
            if not self.dl().recorded_rows_with(sha):
                return None, [], (f"{rel} (sha256 {sha[:12]}) is in no RECORDED or IMPLEMENTED decision-log row by "
                                  f"the end of {self.day}: the trigger is not entered")
            if self.reader is None:
                return None, [], f"{rel}: no reader of the AM-3 trigger record (plan O-8; B2R13)"
            try:
                got = self.reader(rel, data)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                return None, [], f"{rel}: the trigger reader refuses it ({type(e).__name__}: {e})"
            if not (isinstance(got, dict) and isinstance(got.get("triggered"), bool)):
                return None, [], f"{rel}: the trigger reader gives no triggered verdict"
            return got["triggered"], [Rec(self, "AM-3 trigger record", rel, sha, how="entry")], None
        return self._once("trigger", read)

    def go_rule(self, arm: str) -> tuple:
        """An AM-18 arm's go-rule record at C_X (scripts/preflight_arms.py GO_RULE_REL), entered (its sha256 in a
        RECORDED or IMPLEMENTED row): (its verdict, its record), or (None, None) without one of record."""
        pa = E._pa()
        rel = pa.GO_RULE_REL.format(arm=arm)
        if not self.records.has(rel):
            return None, None
        try:
            doc, sha = self.records.json(rel, R.GO_RULE_RECORD, code="am22_condition_unmet")
        except S.TmanRefused as e:
            self.notes.append(f"{rel}: not read ({e.detail})")
            return None, None
        if doc.get("format") != pa.GO_RULE_FORMAT or doc.get("arm") != arm or not self.dl().recorded_rows_with(sha):
            self.notes.append(f"{rel}: not a go-rule record of {arm} entered in the decision log at "
                              f"{self.commit[:12]}")
            return None, None
        return doc.get("verdict"), Rec(self, f"AM-18 go-rule record of {arm}", rel, sha, how="entry")

    def condition_entry(self, arm: str) -> tuple:
        """AM-23 item 3(b)'s condition entry of arm X at C_X (a recorded entry file of kind condition, not
        corrected): (its verdict -- "conflict" when two disagree --, its record), (None, None) without one."""
        docs = [d for d in self.entry_files() if d["doc"]["kind"] == "condition" and d["doc"]["arm"] == arm]
        if not docs:
            return None, None
        if len({d["doc"]["verdict"] for d in docs}) > 1:
            return "conflict", None
        return docs[0]["doc"]["verdict"], Rec(self, f"AM-23 condition entry of {arm}", docs[0]["path"],
                                              docs[0]["sha256"], how="entry")

    def stage_slot(self, stage: str) -> str | None:
        return next((k for k, s in self.cat.slots.items() if s["stage"] == stage and s["class"] == "run"), None)

    # ---------------------------------------------------------------------------------- registered and settled
    def registered(self) -> tuple:
        """(the registered run slots at X, in catalogue order; {excluded slot: why}; the records that exclude them;
        what keeps the set from being known)."""
        def read():
            excluded, recs, missing = {}, [], []
            triggered, trecs, _ = self.trigger()
            if triggered is not True:
                excluded[E6_KD] = ("AM-3's trigger record reads not triggered" if triggered is False else
                                   "AM-3 has not triggered it (no trigger of record yet)")
                recs += trecs
            for arm in GO_RULE_ARMS:
                e3x, e6x = self.stage_slot(f"E3-{arm}"), self.stage_slot(f"E6-{arm}")
                verdict, rec = self.go_rule(arm)
                if verdict in EXCLUDING_VERDICTS:
                    excluded[e3x] = f"its go-rule record reads {verdict} (AM-18 item 5)"
                    excluded[e6x] = f"E3-{arm} is excluded (AM-23 item 3(c))"
                    recs.append(rec)
                cond, crec = self.condition_entry(arm)
                if cond == "conflict":
                    missing.append(f"{e6x}: its AM-23 condition entries disagree (AM-23 item 3(b))")
                elif cond == "no" and e6x not in excluded:
                    excluded[e6x] = "its AM-23 condition entry reads no (AM-23 item 3(c))"
                    recs.append(crec)
            regs = [k for k, s in self.cat.slots.items() if s["class"] == "run" and k not in excluded]
            return regs, excluded, recs, missing
        return self._once("registered", read)

    def am7_fired(self) -> list:
        """AM-19 item 7: AM-7's full consequence fires when a default candidate at seed 42 diverges, as its run's
        records or its draft entry (diverged, superseded) show."""
        out = []
        for k, s in self.cat.slots.items():
            if not (s["sweep"] and s["sweep"]["default_candidate"] and s["seed"] == 42):
                continue
            kind = (self.by_slot().get(k) or {}).get("kind")
            att = self.attempts(k) or []
            if kind in ("diverged", "superseded"):
                out.append(f"{k} (entered {kind})")
            elif att and C.status_at(self.run_state(k, att[-1]), self.cutoff) == C.DIVERGED:
                out.append(f"{k} (its run {att[-1]} diverged)")
        return out

    def settled(self, slot: str) -> tuple:
        """(the way `slot` is settled at the end of X, one of completion.settled_ways, or None; its records; why it
        is not settled)."""
        s = self.cat.slots[slot]
        ways = C.settled_ways(s["never_cut"])
        done = self.completion(slot)
        if done is not None:
            rid, comp = done
            return C.SETTLED_COMPLETE, [Rec(self, f"completion of {slot}", f"{rid}/{comp.record}", comp.sha256,
                                            date=D.manila_day(comp.at))], None
        e = self.by_slot().get(slot) or {}
        way, body_why = BODY_SETTLED.get(e.get("kind")), None
        if way is not None:
            rows = e.get("dl_rows") if isinstance(e.get("dl_rows"), list) else []
            held = bool(rows) and all(isinstance(r, str) and self.dl().has(r) for r in rows)
            if way in ways and held:
                return way, [Rec(self, f"{e['kind']} {slot}", r, how="row") for r in rows], None
            if way == C.SETTLED_ABORTED and s["never_cut"]:
                return None, [], STOP_TEXT
            body_why = ("diverged: a never-cut run is settled once entered as superseded (AM-19 item 7(d))"
                        if way == C.SETTLED_DIVERGED and s["never_cut"] else
                        f"entered {e['kind']} in the draft, its rows {rows} not in the decision log by the end of "
                        f"{self.day}")
        at_freeze = None
        for d in self.entry_files():
            doc = d["doc"]
            if slot not in doc["slots"] or doc["kind"] not in ("cut", "not_run"):
                continue
            if s["never_cut"]:
                return None, [], (f"{d['path']} enters it as {doc['kind']}: a never-cut run settles only by "
                                  "completion or supersession (AM-22 Definitions)")
            if doc["kind"] == "not_run":
                return C.SETTLED_NOT_RUN, [Rec(self, f"not-run entry of {slot}", d["path"], d["sha256"],
                                               how="entry")], None
            if doc["rule"] != "freeze":
                return C.SETTLED_CUT, [Rec(self, f"cut entry of {slot} ({doc['rule']})", d["path"], d["sha256"],
                                           how="entry")], None
            at_freeze = d["path"]
        if at_freeze:
            return None, [], f"{at_freeze} cuts it at the freeze, which settles no run (AM-22 Definitions)"
        return None, [], body_why or self.unsettled_why(slot)

    def unsettled_why(self, slot: str) -> str:
        att = self.attempts(slot)
        if att is None:
            return "its attempts are not read (the run directories, --runs)"
        if not att and self.index().attempts(slot) is None:
            return (f"its run is named by no launcher, and the draft committed by the end of {self.day} names none "
                    "(its entry's run_id)")
        if not att:
            return f"not launched by the end of {self.day}" + (" (a never-cut run)" if self.cat.slots[slot]["never_cut"]
                                                               else "")
        st = C.status_at(self.run_state(slot, att[-1]), self.cutoff)
        why = {C.RUNNING: "launched and not complete", C.STOPPED: "stopped, waiting for a repeat (AM-19 item 3(a))",
               C.DIVERGED: "diverged, not entered", C.ABORTED: "aborted, not entered"}.get(st, st)
        return f"{att[-1]}: {why} at the end of {self.day}"


# ------------------------------------------------------------------------------------------------- the conditions
def condition_1(v: View) -> Condition:
    """Every registered run is settled (Definitions; B2R3)."""
    c = Condition(1)
    regs, excluded, recs, missing = v.registered()
    c.records += recs
    c.missing += missing
    c.notes += [f"not registered: {k} ({why})" for k, why in excluded.items()]
    fired = v.am7_fired()
    if fired:
        c.missing.append(f"AM-7's full consequence fired ({'; '.join(fired)}): the E1 reruns of AM-19 item 7(b) are "
                         "registered runs the catalogue does not hold; a catalogue change is a STOP to ask (audit n3)")
    for k in regs:
        way, rrecs, why = v.settled(k)
        if way is None:
            c.missing.append(f"{k}: {why}")
        else:
            c.records += rrecs
    return c


def condition_2(v: View) -> Condition:
    """AM-3's trigger evaluated and entered, read for `triggered` only; E6-KD settled if it fired; E6 seed 42's
    derived artifacts made (item 1(a)2)."""
    c = Condition(2)
    if (v.by_slot().get(E6_S42) or {}).get("kind") == "non_finite":
        c.missing.append(f"{E6_S42} is entered as 'non-finite: no model' (AM-21 item 2(b)): condition 2 holds only "
                         "once the new amendment AM-21 item 2(b) requires is committed and what it settles is done; "
                         "it is not met until a new amendment's code path is added (audit n4)")
        return c
    triggered, recs, why = v.trigger()
    if why:
        c.missing.append(why)
    c.records += recs
    if triggered is True:
        way, krecs, kwhy = v.settled(E6_KD)
        if way is None:
            c.missing.append(f"AM-3 triggered E6-KD: {E6_KD}: {kwhy}")
        c.records += krecs
    sel = v.by_slot().get(E6_S42_SEL) or {}
    h = sel.get("hashed") if sel.get("kind") == "derived" else None
    made = h.get("made_utc") if hasattr(h, "keys") else None
    if not (S.TYPES["utc"](made) and D.meets(D.parse_utc_any(made), v.cutoff)):
        c.missing.append(f"{E6_S42_SEL}: E6 seed 42's conversion and selection are not made by the end of {v.day} "
                         f"(the draft's hashed block: {made!r})")
    else:                       # the selection record, its own path and sha256 (DL-24 R2-14)
        rec = h.get("record") if hasattr(h.get("record"), "keys") else {}
        c.records.append(Rec(v, f"derived artifacts of {E6_S42}", str(rec.get("path")), rec.get("sha256"),
                             date=D.manila_day(D.parse_utc_any(made))))
    done = v.completion(E6_S42)
    if triggered is not None and done is not None:
        cm, t = v.sha_commit(recs[0].sha256)
        entered = max(D.manila_day(t), v.pushed.get(cm, "")) if t is not None else None      # audit M2
        due = D.iso(D.day(D.manila_day(done[1].at)) + timedelta(days=TRIGGER_DUE_DAYS))
        if entered is not None and entered > due:
            c.reports.append(f"REPORT deviation: trigger entered late (AM-22 item 1(a)2): entered {entered}, due by "
                             f"the end of {due} ({E6_S42} complete {D.manila_day(done[1].at)})")
    return c


def condition_3(v: View) -> Condition:
    """E6 seed 42's conversion and selection are hashed in the draft (B-1's e6s42_not_hashed_at_freeze)."""
    c = Condition(3)
    draft, why, rec = v.draft()
    if draft is None:
        c.missing.append(why)
        return c
    errs = M.e6s42_errors(draft["by_slot"])
    c.missing += errs
    if not errs:
        c.records.append(rec)
        h = (draft["by_slot"].get(E6_S42_SEL) or {}).get("hashed")
        if hasattr(h, "keys"):
            sel = h.get("record") if hasattr(h.get("record"), "keys") else {}
            c.records.append(Rec(v, f"{E6_S42_SEL} hashed in the draft", str(sel.get("path")), sel.get("sha256"),
                                 date=rec.date))
    return c


def condition_4(v: View) -> Condition:
    """Every lane in the lane register is IMPLEMENTED by X -- its row IMPLEMENTED at C_X naming it and its merge
    commit, an ancestor of C_X -- or entered as not required by X; L-AM16-ARM is not read (item 1(g))."""
    from . import validate as V
    c = Condition(4)
    rel = S.LANES_REL
    if not v.records.has(rel):
        c.missing.append(f"the lane register {rel} is not committed by the end of {v.day}")
        return c
    try:
        doc, sha = v.records.json(rel, True, code="lane_unmet")
    except S.TmanRefused as e:
        c.missing.append(f"{rel}: {e.detail}")
        return c
    errs = S.field_errors(doc, V.LANES_SPEC, rel)
    if errs or not doc["lanes"]:
        c.missing.append(f"{rel}: " + ("; ".join(errs[:3]) if errs else "lists no lane (plan §7)"))
        return c
    if not v.dl().recorded_rows_with(sha):
        c.missing.append(f"{rel}: sha256 {sha[:12]} is in no RECORDED or IMPLEMENTED row by the end of {v.day}")
    c.records.append(Rec(v, "lane register", rel, sha, how="entry"))
    src, dl = v.records.src, v.dl()
    for lane in doc["lanes"]:
        name, st = lane["lane"], lane["status"]
        if name == ARM_LANE:
            c.notes.append(f"{ARM_LANE} is not read (AM-22 item 1(g))")
            continue
        if st["kind"] == "implemented":
            mc = st["merge_commit"]
            ok = bool(st["dl"] and (dl.status(st["dl"]) or "").startswith("IMPLEMENTED") and mc
                      and src.commit_exists(mc) and src.is_ancestor(mc, v.commit))
            if not ok:
                c.missing.append(f"lane {name}: not IMPLEMENTED by the end of {v.day} (its row {st['dl']} and merge "
                                 f"commit {mc} at {v.commit[:12]})")
            elif not all(n in (dl.row_text(st["dl"]) or "") for n in (name, mc[:7])):
                c.missing.append(f"lane {name}: its IMPLEMENTED row {st['dl']} does not name the lane and its merge "
                                 "commit")
            else:
                c.records += [Rec(v, f"lane {name} IMPLEMENTED", st["dl"], how="row"),
                              Rec(v, f"lane {name} merge commit", mc, how="commit")]
            continue
        why = not_required_error(v, lane)
        if why:
            c.missing.append(f"lane {name}: {why}")
        else:
            c.records.append(Rec(v, f"lane {name} not required", st["entry"]["path"], st["entry"]["sha256"],
                                 how="entry"))
    return c


def not_required_error(v: View, lane: dict) -> str | None:
    """A lane entered as not required by X (AM-19 item 3(d)): its not-required entry and the cut entries it cites
    are recorded entry files at C_X, the cuts covering every slot the lane serves."""
    ref = lane["status"]["entry"]
    if not ref:
        return "not required without its entry"
    docs = {d["path"]: d for d in v.entry_files()}
    nr = docs.get(ref["path"])
    if nr is None or nr["sha256"] != ref["sha256"] or nr["doc"]["kind"] != "not_required" \
            or nr["doc"]["lane"] != lane["lane"]:
        return f"its not-required entry {ref['path']} is not entered by the end of {v.day}"
    covered = set()
    for c in nr["doc"]["cut_entries"]:
        d = docs.get(c["path"])
        if d is not None and d["sha256"] == c["sha256"] and d["doc"]["kind"] == "cut":
            covered |= set(d["doc"]["slots"])
    left = sorted(set(lane["serves"]) - covered)
    return f"serves {left} with no cut entry entered by the end of {v.day}" if left else None


def unentered(x, cat) -> list:
    """The run and measurement slots the draft does not enter (validate.g_cut_unentered_at_unlock's rule: E6-KD only
    when AM-3 triggered it)."""
    kd_off = (x.body["trigger"] or {}).get("triggered") is not True
    return [k for k, s in cat.slots.items() if s["class"] in ("run", "measurement") and k not in x.by_slot
            and not (kd_off and s["stage"] == "E6-KD")]


def condition_5(v: View, *, registered_cells=None) -> Condition:
    """The draft committed by X, read as its blob at C_X (audit m5), passes the validator in freeze mode with AM-22's
    candidate freeze for D = X (B2R6): its schedule block that candidate's, the evaluator commit among its entries.
    The ARM latency measurement's freeze cut cannot exist before the freeze (item 1(g); AM-19 item 3(b), first point),
    so a draft whose one unmet freeze-time item is that measurement's entry passes (TMAN-B2-RULE-3)."""
    from . import validate as V
    c = Condition(5)
    draft, why, rec = v.draft()
    if draft is None:
        c.missing.append(why)
        return c
    x = V.Ctx(mode="freeze", manifest_bytes=draft["data"], manifest_path=S.MANIFEST_REL, records=v.records,
              runs_root=v.runs_root, registered_cells=registered_cells, trigger_reader=v.reader, am22_d=v.day)
    try:
        c.reports += V.validate(x).reports
    except S.TmanRefused as e:
        left = unentered(x, v.cat) if e.code == "cut_unentered_at_unlock" else []
        if not (left and all(v.cat.slots[k]["class"] == "measurement" for k in left)):
            c.missing.append(f"the validator refuses the draft in freeze mode with AM-22's freeze for D = {v.day}: "
                             f"[{e.code}] {e.detail[:400]}")
            return c
        c.notes.append(f"{left}: the measurement is cut at the freeze, its cut entered then (AM-22 item 1(g); "
                       "TMAN-B2-RULE-3)")
        c.reports += V.report_lines(x)          # every gate before the last passed (DL-24 R1-F9)
    c.records.append(rec)
    ev = ((draft["doc"].get("body") or {}).get("evaluator") or {}).get("commit")
    if isinstance(ev, str):
        c.records.append(Rec(v, "evaluator commit", ev, how="commit"))
    return c


CONDITIONS = (condition_1, condition_2, condition_3, condition_4, condition_5)


# -------------------------------------------------------------------------------------------------- the check
@dataclass
class Check:
    day: str
    view: View
    conditions: list

    @property
    def met(self) -> bool:
        return all(c.met for c in self.conditions)


def evaluate(records: R.Records, cat, runs_root, day: str, *, reader=None, states: dict | None = None,
             quick: bool = False, pushed: dict | None = None, **kw) -> Check:
    """The five conditions at the end of `day`, from the records committed by then (`pushed`: audit M2's confirmed
    push days); `quick`, condition 5 (the validator's run) only once conditions 1 to 4 hold."""
    v = View(records, cat, runs_root, day, reader=reader, states=states, pushed=pushed)
    if v.commit is None:
        return Check(day, v, [Condition(n, missing=[f"no commit of the records ref by the end of {day}"])
                              for n in range(1, 6)])
    conds = [fn(v) for fn in CONDITIONS[:4]]
    if quick and not all(c.met for c in conds):
        conds.append(Condition(5, missing=["not read: a condition of 1 to 4 is MISSING"]))
    else:
        conds.append(condition_5(v, **kw))
    return Check(day, v, conds)


def days(first: str, last: str) -> list:
    out, d = [], D.day(first)
    while D.iso(d) <= last:
        out.append(D.iso(d))
        d += timedelta(days=1)
    return out


def first_d(records: R.Records, cat, runs_root, upto: str, *, reader=None, states: dict | None = None,
            **kw) -> tuple:
    """Audit m4: every date from AM-22's in-force day 2026-10-17 to `upto`, each from the records as of that day's end
    (M3); (the first at whose end all five conditions hold, or None; the dates' checks)."""
    states, checks = ({} if states is None else states), []
    for day in days(AM22_IN_FORCE, upto):
        checks.append(evaluate(records, cat, runs_root, day, reader=reader, states=states, quick=True, **kw))
        if checks[-1].met:
            return day, checks
    return None, checks


def candidate_draft(v: View) -> bool:
    """Condition 5's first need at X: the draft at C_X carries X's candidate schedule block, the one the validator
    compares with --am22-d X (validate.g_schedule_invalid; audit m5: the draft of each candidate day is built so)."""
    draft = v.draft()[0] if v.commit else None
    sch = (draft["doc"].get("body") or {}).get("schedule") if draft and hasattr(draft["doc"], "keys") else None
    if sch is None:                                     # no draft committed by the end of X, or none with a schedule
        return False
    try:
        return sch == D.freeze_instant(D.schedule_block(v.records.src), providers=[D.am22_candidate(v.day)]).as_block()
    except S.TmanRefused:
        return False


def fallback_error(records: R.Records, cat, runs_root, *, reader=None, states: dict | None = None,
                   **kw) -> str | None:
    """Item 1(f) applies only when item 1(a) has given no D by the end of 2026-11-02 (PA :2353; DL-24 R1-F2, R2-2): a
    date from 2026-10-17 to 2026-11-02 at whose end the five conditions hold refuses a freeze of item 1(f). Only a date
    whose draft carries its candidate freeze can have them (candidate_draft: condition 5), so only such a date is
    checked."""
    states = {} if states is None else states
    for day in days(AM22_IN_FORCE, LAST_D):
        if not candidate_draft(View(records, cat, runs_root, day, reader=reader)):
            continue
        if evaluate(records, cat, runs_root, day, reader=reader, states=states, quick=True, **kw).met:
            return (f"the five conditions hold at the end of {day}: D is {day}, and item 1(f) applies only when item "
                    f"1(a) gives no D by the end of {LAST_D} (PA :2353)")
    return None


# ----------------------------------------------------------------------------- the AM-22 provider (B2R6, m10)
def recorded_freezes(records: R.Records) -> list:
    """The freeze records a RECORDED or IMPLEMENTED decision-log row enters, [(path, document)]: each entry file of
    kind freeze it names that reads as entries.read_entry_file reads it (one that does not is no freeze of record),
    less those a later recorded entry corrects (DL-24 R1-N9)."""
    names = sorted({m.group(0) for st, line in records.decision_log().rows.values()
                    if st.startswith(("RECORDED", "IMPLEMENTED")) for m in ENTRY_PATH_RE.finditer(line)
                    if m.group(0).rsplit("/", 1)[1].startswith("freeze__")})
    out = []
    for rel in names:
        try:
            doc, sha = E.read_entry_file(records, rel, "freeze")
        except S.TmanRefused:
            continue
        out.append((rel, doc, sha))
    corrected = {d["corrects"]["sha256"] for _, d, _ in out if isinstance(d.get("corrects"), dict)}
    return [(rel, d) for rel, d, sha in out if sha not in corrected]


def entered_1b(records: R.Records) -> list:
    """The paths of the AM-22 item 1(b) freeze records the decision log enters."""
    return [rel for rel, d in recorded_freezes(records) if d.get("source") == D.AM22_SOURCE_1B]


def providers(records: R.Records, *, d_date: str | None = None, freeze_doc: dict | None = None,
              lookup: bool = False, conflict: bool = True) -> list:
    """B2R6: the provider each freeze_instant call puts in front of B-1's: the candidate for D = `d_date` (--am22-d,
    a date from 2026-10-17 to 2026-11-02; condition 5); else the freeze record `freeze_doc` (unlock mode:
    --freeze-record only); else, with `lookup`, the AM-22 freeze recorded in the decision log (two are
    freeze_source_conflict; `conflict` false takes the first, the conflict being reported). A record of item 1(f)
    gives none: B-1's providers answer (item 1(f); AM-19 item 1(d)). With an item 1(b) freeze, a RECORDED or
    IMPLEMENTED row holding a TMAN freeze_delay or new_freeze entry is freeze_source_conflict: under item 1 every
    never-cut run is complete at D (PA :2309-2311; audit m10)."""
    if d_date is not None:
        if not (S.TYPES["day"](d_date) and AM22_IN_FORCE <= d_date <= LAST_D):
            raise S.TmanRefused("schedule_invalid", f"--am22-d {d_date!r} is not a date from {AM22_IN_FORCE} to "
                                                    f"{LAST_D} (AM-22 item 1(a); DL-24 R2-15)")
        prov = D.am22_candidate(d_date)
    else:
        if freeze_doc is None and lookup:
            found = [d for _, d in recorded_freezes(records) if d.get("source") == D.AM22_SOURCE_1B]
            if len(found) > 1 and conflict:
                raise S.TmanRefused("freeze_source_conflict", f"{len(found)} AM-22 item 1(b) freeze records are "
                                                              "entered: the freeze is made once (item 1(b))")
            freeze_doc = found[0] if found else None
        if not (isinstance(freeze_doc, dict) and freeze_doc.get("source") == D.AM22_SOURCE_1B
                and S.TYPES["day"](freeze_doc.get("freeze_date"))):
            return []
        prov = D.am22_provider(freeze_doc["freeze_date"])
    if conflict:
        rows = [dl for dl, (st, line) in records.decision_log().rows.items()
                if st.startswith(("RECORDED", "IMPLEMENTED"))
                and ("TMAN freeze_delay entry" in line or "TMAN new_freeze entry" in line)]
        if rows:
            raise S.TmanRefused("freeze_source_conflict", f"{rows} enter an AM-19 item 1(d) freeze delay or new freeze "
                                                          "while the freeze is AM-22 item 1(b)'s: under item 1 every "
                                                          "never-cut run is complete at D (PA :2309-2311; audit m10)")
    return [prov]


# ------------------------------------------------------------------------- the freeze entry (B2R5, B2R7, m8, m9)
def freeze_conditions(chk: Check) -> list:
    """The freeze entry's conditions (B2R5): the five, numbered, each with the records the check read for D and, for
    each confirmed push of a commit one of them is held by (audit M2), a record {kind "confirmed push", ref the commit,
    sha256 null, date its push day}."""
    out = []
    for c in chk.conditions:
        held = {cm for r in c.records for _, cm, _ in r.commits}
        out.append({"number": c.number, "records": [r.entry() for r in c.records] + [
            {"kind": CONFIRMED_PUSH, "ref": cm, "sha256": None, "date": day}
            for cm, day in sorted(chk.view.pushed.items()) if cm in held]})
    return out


def confirmed(doc: dict) -> dict:
    """The confirmed push days a freeze entry carries, {commit: day} (audit M2): its records of kind "confirmed
    push"."""
    return {r["ref"]: r["date"] for c in (doc.get("conditions") or []) for r in c["records"]
            if r["kind"] == CONFIRMED_PUSH}


def pushed_error(records: R.Records, pushed: dict, chk: Check | None = None) -> str | None:
    """--pushed COMMIT=DAY (audit M2): each a commit of the records' history, confirmed pushed on a day later than its
    committer day; with `chk`, each held by a record the check for D relies on (a confirmation no record needs is not
    entered)."""
    for cm, day in sorted(pushed.items()):
        t = records.commit_time(cm) if S.COMMIT_RE.fullmatch(cm) else None
        if t is None or not S.TYPES["day"](day) or not records.src.is_ancestor(cm, records.ref):
            return f"--pushed {cm}={day}: not a commit of the records' history and a date (YYYY-MM-DD)"
        if day <= D.manila_day(t):
            return f"--pushed {cm}={day}: it was committed on {D.manila_day(t)}; a confirmation names a later day"
        if chk is not None and not any(cm == c for x in chk.conditions for r in x.records for _, c, _ in r.commits):
            return f"--pushed {cm}={day}: no record the check for {chk.day} relies on is held by that commit"
    return None


def conditions_error(recorded, chk: Check) -> str | None:
    """am22_condition_unmet: the recorded conditions are the recomputed ones -- the same records (kind, ref, sha256)
    for each, each dated no earlier than the day its commits give (audit M2: a push confirmed on a later day dates a
    record later, never earlier) and no later than D."""
    want = freeze_conditions(chk)
    for got, w in zip(recorded, want):
        g = {(r["kind"], r["ref"], r["sha256"]): r["date"] for r in got["records"]}
        h = {(r["kind"], r["ref"], r["sha256"]): r["date"] for r in w["records"]}
        if set(g) != set(h):
            return f"condition {w['number']}'s records are not the ones recomputed for D: {sorted(set(g) ^ set(h))[:3]}"
        early = sorted(k for k in g if g[k] < h[k] or g[k] > chk.day)
        if early:
            return f"condition {w['number']}'s records {early[:3]} are dated before their commits' day or after D"
    return None


def committed_by(v: View, rec: dict) -> bool:
    """A record the freeze entry cites is committed by the end of D (the records at C_D): a decision-log row present
    there, a commit in C_D's history, a records file there at its sha256; a run directory's record is not a
    committed record (its date is checked)."""
    ref, sha = rec["ref"], rec["sha256"]
    if S.DL_ID_RE.fullmatch(ref):
        return v.dl().has(ref)
    if S.COMMIT_RE.fullmatch(ref):
        return v.records.src.commit_exists(ref) and v.records.src.is_ancestor(ref, v.commit)
    if ss.RecordsSource.is_record(ref):
        return v.records.has(ref) and (sha is None or S.sha256_bytes(v.records.read(ref)) == sha)
    return True


def frozen_body_error(v: View, records: R.Records, body: dict, freeze_date: str) -> str | None:
    """TMAN-B2-RULE-3 (amending audit m9): the body to freeze is the draft committed by the end of D (its blob at C_D)
    plus, only when the ARM latency measurement is not complete at the freeze, its freeze cut -- rule freeze, dated the
    freeze date, its row entered that day (`records`: the decision log's history). Nothing else differs, and a
    measurement is in the draft or joins it (DL-24 R3-01). A measurement entered at the freeze otherwise than cut,
    absent from the draft: a STOP to ask."""
    draft, why, _ = v.draft()
    if draft is None:
        return f"no draft to freeze: {why}"
    dbody = draft["doc"].get("body") if hasattr(draft["doc"], "keys") else None
    ents = body.get("entries") if isinstance(body, dict) else None
    if not (isinstance(dbody, dict) and isinstance(ents, list) and all(isinstance(e, dict) for e in ents)):
        return "the body to freeze holds no entries"
    have = {e.get("slot") for e in dbody["entries"]}
    rest = dict(body, entries=[e for e in ents if e.get("slot") in have])
    if S.canonical_sha256(rest) != S.canonical_sha256(dbody):
        return (f"the body to freeze differs from the draft committed by the end of {v.day} beyond the measurement's "
                "freeze cut (TMAN-B2-RULE-3)")
    gone = [k for k, s in v.cat.slots.items() if s["class"] == "measurement"
            and k not in have | {e.get("slot") for e in ents}]
    if gone:
        return (f"{gone}: in neither the draft nor the body to freeze; a measurement not complete at the freeze is cut "
                "at it, its cut in the body (AM-22 item 1(g); TMAN-B2-RULE-3)")
    for e in (e for e in ents if e.get("slot") not in have):
        s = v.cat.slots.get(e.get("slot"))
        if s is None or s["class"] != "measurement":
            return (f"{e.get('slot')}: only the ARM latency measurement's freeze cut joins the draft at the freeze "
                    "(AM-22 item 1(g); TMAN-B2-RULE-3)")
        if e.get("kind") != "cut":
            return (f"{e['slot']}: entered {e.get('kind')} at the freeze and absent from the draft at C_D; a "
                    "measurement complete by the freeze is a STOP to ask (TMAN-B2-RULE-3)")
        ref = e.get("entry") if isinstance(e.get("entry"), dict) else {}
        entered = E.entered_day(records, ref["sha256"]) if isinstance(ref.get("sha256"), str) else None
        if (e.get("rule"), e.get("cut_date"), entered) != ("freeze", freeze_date, freeze_date):
            return (f"{e['slot']}: its freeze cut is rule {e.get('rule')!r} dated {e.get('cut_date')}, entered "
                    f"{entered}; it is the freeze's cut, dated and entered on the freeze date {freeze_date} "
                    "(TMAN-B2-RULE-3)")
    return None


def freeze_body(records: R.Records, cat, d_date: str, cuts: list, *, pushed: dict | None = None) -> dict:
    """`freeze-body --d-date D --cut <entry file>` (TMAN-B2-RULE-3; DL-24 R2-8): the manifest to freeze, the draft's
    blob at C_D with a body entry for each slot of the measurement's freeze cut given -- a committed entry file of kind
    cut that a RECORDED or IMPLEMENTED row enters -- in catalogue order; every other entry and block, lane_code
    included, as the draft holds it, so `entry freeze --d-date D` finds nothing else differing (frozen_body_error,
    checked here too)."""
    if not (S.TYPES["day"](d_date) and AM22_IN_FORCE <= d_date <= LAST_D):
        raise S.TmanRefused("schedule_invalid", f"--d-date {d_date!r} is not a date from {AM22_IN_FORCE} to {LAST_D}")
    v = View(records, cat, None, d_date, pushed=pushed)
    draft, why, _ = v.draft() if v.commit else (None, f"no commit of the records ref by the end of {d_date}", None)
    if draft is None:
        raise S.TmanRefused("freeze_entry_invalid", f"no draft to freeze: {why}")
    body, given = copy.deepcopy(draft["doc"]["body"]), {}
    for rel in cuts:
        doc, sha = E.read_entry_file(records, rel, "cut")
        cmp = doc["comparator"]["slot"] if doc["comparator"] else None
        for k in doc["slots"]:
            given[k] = {"kind": "cut", "dl_rows": records.decision_log().recorded_rows_with(sha),
                        "entry": {"path": rel, "sha256": sha}, "cut_date": doc["date"], "rule": doc["rule"],
                        "comparator": cmp}
    held = sorted(set(given) & {e["slot"] for e in body["entries"]})
    if held:
        raise S.TmanRefused("freeze_entry_invalid", f"{held} are entered in the draft at C_D already")
    order = {k: i for i, k in enumerate(cat.slots)}
    body["entries"] = sorted(body["entries"] + M.build_entries(cat, given), key=lambda e: order.get(e["slot"], -1))
    why = frozen_body_error(v, records, body, D.iso(D.day(d_date) + timedelta(days=1)))
    if why:
        raise S.TmanRefused("freeze_entry_invalid", why)
    return dict(draft["doc"], body=body)


def freeze_entry_error(doc: dict, resolver_date: str | None) -> str | None:
    """freeze_entry_invalid's field relations (B2R5; m8): the source is item 1(b)'s or item 1(f)'s; the entry is dated
    its freeze date; under 1(b), D and the freeze date D + 1, the five conditions numbered 1 to 5, no record dated
    after D; under 1(f), neither D nor conditions, the freeze date the resolver's (`resolver_date`: its moved date,
    else 2026-11-03)."""
    src = doc.get("source")
    if src not in (D.AM22_SOURCE_1B, D.AM22_SOURCE_1F):
        return f"source {src!r} is neither {D.AM22_SOURCE_1B!r} nor {D.AM22_SOURCE_1F!r} (B2R5)"
    if doc["date"] != doc["freeze_date"]:
        return f"the freeze entry is dated {doc['date']}, not its freeze date {doc['freeze_date']} (audit m8)"
    d, conds = doc["d_date"], doc["conditions"]
    if src == D.AM22_SOURCE_1F:
        if d is not None or conds is not None:
            return "a freeze of item 1(f) names no D and no conditions (B2R5)"
        if doc["freeze_date"] < FALLBACK_FREEZE:
            return (f"a freeze of item 1(f) is AM-19's as committed, {FALLBACK_FREEZE} or later as AM-19 item 1(d) "
                    f"moves it, not {doc['freeze_date']} (DL-24 R2-3)")
        if resolver_date is not None and doc["freeze_date"] != resolver_date:
            return f"the freeze date {doc['freeze_date']} is not the resolver's {resolver_date} (item 1(f); audit m8)"
        return None
    if d is None or conds is None or D.iso(D.day(d) + timedelta(days=1)) != doc["freeze_date"]:
        return f"a freeze of item 1(b) names D, and its freeze date is D + 1 (D {d}, freeze date {doc['freeze_date']})"
    if [c["number"] for c in conds] != [1, 2, 3, 4, 5]:
        return f"the conditions are the five, numbered 1 to 5, not {[c['number'] for c in conds]} (B2R5)"
    late = sorted({r["ref"] for c in conds for r in c["records"] if r["date"] > d})
    return f"records dated after D {d}: {late[:3]} (B2R7)" if late else None


def freeze_record_error(records: R.Records, cat, runs_root, doc: dict, body: dict,
                        resolver_date: str | None) -> str | None:
    """freeze_entry_invalid (B2R7): the field relations; under item 1(b), every record committed by the end of D
    (the records at C_D) and the frozen body the draft at C_D but for the measurement's freeze cut
    (TMAN-B2-RULE-3)."""
    err = freeze_entry_error(doc, resolver_date)
    if err or doc["source"] != D.AM22_SOURCE_1B:
        return err
    pushed = confirmed(doc)
    v = View(records, cat, runs_root, doc["d_date"], pushed=pushed)
    if v.commit is None:
        return f"no commit of the records ref by the end of D {doc['d_date']}"
    err = pushed_error(records, pushed)
    if err:
        return f"a confirmed push of the freeze entry: {err} (audit M2)"
    late = sorted({r["ref"] for c in doc["conditions"] for r in c["records"] if not committed_by(v, r)})
    if late:
        return f"records not committed by the end of D {doc['d_date']}: {late[:3]} (B2R7)"
    return frozen_body_error(v, records, body, doc["freeze_date"])


def condition_unmet_error(records: R.Records, cat, runs_root, doc: dict, *, reader=None, **kw) -> str | None:
    """am22_condition_unmet (B2R6, B2R7): the check run again for the record's D from the records at C_D (M3; the
    record's confirmed pushes, M2); a condition that does not hold, or recorded conditions that are not the recomputed
    ones."""
    chk = evaluate(records, cat, runs_root, doc["d_date"], reader=reader, pushed=confirmed(doc), **kw)
    unmet = [c for c in chk.conditions if not c.met]
    if unmet:
        return "; ".join(f"condition {c.number} does not hold at the end of D {doc['d_date']}: {c.missing[0]}"
                         for c in unmet[:2])
    return conditions_error(doc["conditions"], chk)


def d_not_first_error(records: R.Records, cat, runs_root, d_date: str, *, reader=None, **kw) -> str | None:
    """am22_d_not_first (B2R7; audit m4): D is a date from 2026-10-17 to 2026-11-02, and no earlier date from
    2026-10-17 has all five conditions at its end."""
    if d_date > LAST_D:
        return f"D {d_date} is later than {LAST_D}: AM-22 item 1(f) applies"
    if d_date < AM22_IN_FORCE:
        return f"D {d_date} is before AM-22 is in force ({AM22_IN_FORCE}; B2R1; DL-24 R2-15)"
    earlier, _ = first_d(records, cat, runs_root, previous_day(d_date), reader=reader, **kw)
    return (f"the five conditions already hold at the end of {earlier}: D is the first such date (item 1(a); audit "
            "m4)") if earlier else None


def freeze_fields(records: R.Records, cat, runs_root, d_date: str, body: dict, *, now: float, reader=None,
                  pushed: dict | None = None, manifest: tuple | None = None, **kw) -> tuple:
    """`entry freeze --d-date D` (audit m9 as TMAN-B2-RULE-3 amends it): D's end has passed; no item 1(b) freeze is
    entered yet (DL-24 R3-14); the check recomputed for D and every earlier date from the records at each date's end
    (`pushed`: the confirmed push days, audit M2), never pasted output (am22_d_not_first, am22_condition_unmet); the
    body to freeze the draft at C_D but for the measurement's freeze cut, and (`manifest`: its (bytes, path)) passing
    the validator in freeze mode with AM-22's freeze for D (freeze_entry_invalid). Returns (the fields {body_sha256 of
    that body, freeze_date D + 1, source item 1(b), d_date D, the recomputed conditions, condition 5 citing the draft
    at C_D, each confirmed push}, the check for D)."""
    from . import validate as V
    err = date_error(d_date, now)
    if err:
        raise err
    if entered_1b(records):
        raise S.TmanRefused("freeze_entry_invalid", f"an item 1(b) freeze is entered already: {entered_1b(records)}; "
                                                    "the freeze is made once (item 1(b); DL-24 R3-14)")
    pushed = dict(pushed or {})
    err = pushed_error(records, pushed)
    if err:
        raise S.TmanRefused("entry_field", f"{err} (audit M2)")
    states = {}
    why = d_not_first_error(records, cat, runs_root, d_date, reader=reader, states=states, pushed=pushed, **kw)
    if why:
        raise S.TmanRefused("am22_d_not_first", why)
    chk = evaluate(records, cat, runs_root, d_date, reader=reader, states=states, pushed=pushed, **kw)
    unmet = [c for c in chk.conditions if not c.met]
    if unmet:
        raise S.TmanRefused("am22_condition_unmet", "; ".join(f"condition {c.number} does not hold at the end of "
                                                              f"{d_date}: {c.missing[0]}" for c in unmet[:2]))
    err = pushed_error(records, pushed, chk)
    if err:
        raise S.TmanRefused("entry_field", f"{err} (audit M2)")
    freeze_date = D.iso(D.day(d_date) + timedelta(days=1))
    why = frozen_body_error(chk.view, records, body, freeze_date)
    if why:
        raise S.TmanRefused("freeze_entry_invalid", why)
    if manifest is not None:                                       # DL-24 R2: the body as the unlock will read it
        try:
            V.validate(V.Ctx(mode="freeze", manifest_bytes=manifest[0], manifest_path=manifest[1], records=records,
                             runs_root=runs_root, registered_cells=kw.get("registered_cells"), trigger_reader=reader,
                             am22_d=d_date))
        except S.TmanRefused as e:
            raise S.TmanRefused("freeze_entry_invalid", f"the body to freeze fails the validator in freeze mode with "
                                                        f"AM-22's freeze for D {d_date}: [{e.code}] {e.detail[:300]}")
    return ({"body_sha256": S.canonical_sha256(body), "freeze_date": freeze_date, "source": D.AM22_SOURCE_1B,
             "d_date": d_date, "conditions": freeze_conditions(chk)}, chk)


def date_error(day, now: float) -> S.TmanRefused | None:
    """The check runs for a date AM-22 can make D (item 1(a): from its in-force day to 2026-11-02; item 1(f) after)
    whose end has passed (audit m6)."""
    if not S.TYPES["day"](day):
        return S.TmanRefused("am22_date_not_ended", f"--date {day!r} is not a date (YYYY-MM-DD)")
    if not D.cutoff_instant(day) < now:
        return S.TmanRefused("am22_date_not_ended", f"the end of {day} (24:00 Asia/Manila, "
                                                    f"{ss.iso_utc(D.cutoff_instant(day))}) has not passed: the check "
                                                    "reads the records at a date's end (audit m6)")
    if day > LAST_D:
        return S.TmanRefused("am22_d_not_first", f"{day} is later than {LAST_D}: AM-22 item 1(f) applies -- AM-19 "
                                                 f"as committed, the freeze {FALLBACK_FREEZE}, item 2's cuts standing")
    if day < AM22_IN_FORCE:
        return S.TmanRefused("am22_d_not_first", f"{day} is before AM-22 is in force ({AM22_IN_FORCE}; B2R1)")
    return None


def check_freeze(records: R.Records, cat, runs_root, day: str, *, now: float, reader=None, pushed: dict | None = None,
                 **kw) -> tuple:
    """`check-freeze --date X` (B2R4): X refused unless its end has passed and it is no later than 2026-11-02; then
    (the five conditions at the end of X; D, the first date from 2026-10-17 to X at whose end all five hold, or None
    (audit m4); the earlier dates' checks), each date read with the confirmed push days `pushed` (audit M2)."""
    err = date_error(day, now)
    if err:
        raise err
    pushed = dict(pushed or {})
    err = pushed_error(records, pushed)
    if err:
        raise S.TmanRefused("entry_field", f"{err} (audit M2)")
    states = {}
    chk = evaluate(records, cat, runs_root, day, reader=reader, states=states, pushed=pushed, **kw)
    d, scans = first_d(records, cat, runs_root, previous_day(day), reader=reader, states=states, pushed=pushed, **kw)
    return chk, (d if d is not None else (day if chk.met else None)), scans


def boundary_reports(chk: Check) -> list:
    """Audit M2: a REPORT for each commit made on the check's date or the day before, whatever the hour, that holds a
    record a condition relies on; its push is confirmed from the remote before `entry freeze` runs (a commit whose
    push is confirmed already, --pushed, is not asked again)."""
    v, days, seen = chk.view, (chk.day, previous_day(chk.day)), {}
    for c in chk.conditions:
        for r in c.records:
            for _, cm, t in r.commits:
                if t is not None and cm not in v.pushed and D.manila_day(t) in days:
                    seen.setdefault(cm, D.manila_day(t))
    return [f"REPORT boundary-day entry: confirm from the remote that {cm} reached origin by the end of {day} "
            "(AM-19 :1374)" for cm, day in sorted(seen.items())]


def check_lines(chk: Check, title: str = "CHECK-FREEZE") -> list:
    """One check's printed lines: each condition MET with its records or MISSING with what is missing, every record's
    commits (M2) and confirmed push day, the conditions' REPORT lines, the boundary-day REPORTs (audit M2), NOTE lines
    and the RESULT."""
    v = chk.view
    out = [f"{title} {chk.day} (AM-22 item 1(a)): the records committed by its end, C_X {v.commit or 'none'}"
           + (f" (committed {C.utc_text(v.committed)})" if v.committed else "")]
    reports = []
    for c in chk.conditions:
        out.append(f"CONDITION {c.number} {'MET' if c.met else 'MISSING'}: {TEXTS[c.number]}")
        out += [f"  MISSING {m}" for m in c.missing]
        for r in c.records:
            held = "; ".join(f"{w} {cm[:12]} committed {C.utc_text(t)} ({D.manila_day(t)})"
                             + (f", pushed {v.pushed[cm]} (confirmed)" if cm in v.pushed else "")
                             for w, cm, t in r.commits if t is not None)
            out.append(f"  RECORD {r.kind} | {r.ref} | sha256 {r.sha256 or '-'} | date {r.date}"
                       + (f" | {held}" if held else ""))
        out += [f"  NOTE {n}" for n in c.notes]
        reports += c.reports
    out += reports + boundary_reports(chk) + [f"NOTE {n}" for n in v.notes]
    unmet = [str(c.number) for c in chk.conditions if not c.met]
    out.append(f"RESULT: MET (all five conditions at the end of {chk.day})" if not unmet else
               f"RESULT: MISSING (condition{'s' if len(unmet) > 1 else ''} {', '.join(unmet)} at the end of "
               f"{chk.day})")
    return out


def lines(chk: Check, d: str | None = None, scans=()) -> list:
    """The printed lines: each earlier date's verdict (audit m4); the check at X (check_lines); when D is earlier than
    X, D's own check, its records and its boundary-day REPORTs (DL-24 R3-02); then D, or that there is none by X."""
    out = [f"DATE {s.day} C_X {(s.view.commit or 'none')[:12]}: "
           + ("MET" if s.met else "MISSING " + ", ".join(str(c.number) for c in s.conditions if not c.met))
           for s in scans]
    out += check_lines(chk)
    if d and d != chk.day and scans and scans[-1].day == d:
        out += check_lines(scans[-1], "CHECK-FREEZE OF D")
    out.append(f"D: {d}" if d else f"NO D: no date from {AM22_IN_FORCE} to {chk.day} has all five conditions at its "
                                    "end (AM-22 item 1(a))")
    return out


def row_commit(records: R.Records, sha256: str) -> str | None:
    """The commit that first put `sha256` in the decision log, in the records' history (git log -S, the oldest)."""
    out = records.src._out("log", "--format=%H", f"-S{sha256}", records.ref, "--", S.DECISION_LOG_REL)
    found = (out or "").split()
    return found[-1] if found else None
