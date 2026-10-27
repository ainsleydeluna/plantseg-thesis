"""The TEST-day manifest's validator and the inventory reader (lane TMAN, B-1, C5; plan §2).

Modes (plan §2.1): `draft` checks the structure, identity and hashes of what is present and prints each absent
freeze-time item as `MISSING: <code>: ...` (exit 3), never refusing it; `freeze` adds every freeze-time requirement at
the original freeze instant (the moved one for never-cut runs); `unlock` adds the unlock's (the body unchanged since
the freeze record, the appends, every cut entered, the lane register, S = E with the evaluator file set unchanged,
the hasher's and generator's scripts at their pins, no session yet); `session` adds the session's own record (TB11).

One gate per refusal code, in a fixed order (GATES); a gate runs in the modes it names. `Result.passed` lists the
gates passed: the smokes' witness that a refusal is reached with every earlier gate passing. A refusal prints
`REFUSED [<code>] -- <detail>` (exit 2). REPORT lines are printed and never refused.

No gate reads a score: records are read through records.py's allowlists, the inventory reader's checks are key and
hash checks, and run directories are read only under the explicit runs root.

`read_inventory(path, *, expect_sha256=None)` reads, checks and returns the inventory export tday-inventory/1 (IF§8,
IF§11 v2); once its binding is bound it verifies the binding file and returns its `artifacts` records under
`binding.artifacts` (IF§11.6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.training import sweep_select as ss

from . import completion as C
from . import entries as E
from . import manifest as M
from . import records as R
from . import schedule as D
from . import schema as S

MODES = ("draft", "freeze", "unlock", "session")
ALL, FU, U, SES = MODES, ("freeze", "unlock", "session"), ("unlock", "session"), ("session",)
MODEL_KINDS = ("scored_run", "default_candidate", "arm_teacher", "derived")
RUN_KINDS = ("scored_run", "default_candidate", "arm_teacher", "listed_run", "non_finite", "diverged", "aborted")
TUPLE_KEYS = ("model_device", "input_devices", "determinism_policy_applied", "batch_size", "num_workers",
              "torch_num_threads", "image_digest", "evaluator_commit", "host", "processes")
TUPLE_FIXED = {"model_device": "cpu", "input_devices": ["cpu"], "determinism_policy_applied": False, "batch_size": 1,
               "num_workers": 0, "image_digest": S.KD_IMAGE}                        # TB1; IF§1
LANES_SPEC = {"format": ("enum", (S.LANES_FORMAT,)),
              "lanes": ("list", ("obj", {"lane": "str", "serves": ("list", "slot"),
                                         "status": ("obj", {"kind": ("enum", ("implemented", "not_required")),
                                                            "dl": "?dl", "merge_commit": "?commit",
                                                            "entry": S.nullable(S.REF)})}))}
SESSION_SPEC = {"unlock_entry": S.REF, "clock_offset": S.REF, "started_utc": "utc",
                "host": ("obj", {"name": "str", "cpu_model": "str"}),            # the execution context's (DL-24 #37)
                "values": ("map", "any"), "push_evidence": ("obj", {"response": S.REF, "list": S.REF})}
SESSION_IDENTITY = ("unlock_entry", "clock_offset", "started_utc", "host", "push_evidence")
# Each body block the gates do not compare whole, held to its exact keys (DL-24 #29, #78): a value only TEST files
# can give has no place beside them. The values are the later gates' to check.
BODY_SPECS = {"schedule": ("obj", {k: "any" for k in ("in_force", "file_sha256", "decision_log_entry",
                                                     "original_freeze_date", "original_freeze_utc", "moved_freeze_date",
                                                     "freeze_source", "test_deadline")} | {"file": "?str"}),
              "evaluator": ("obj", {"commit": "any", "files": "any"}),
              "expected_rows": ("obj", {"value": "any", "cites": ("list", "str")}),
              "generator_pins": ("obj", {"corrupted_cache": ("obj", {"pin": "any", "value": "any"}),
                                         "image_hasher": ("obj", {"pin": "any", "value": "any"})}),
              "lane_code": ("obj", {"commit": "?commit", "files_sha256": ("map", "sha256")}),
              "trigger": S.nullable(("obj", {"path": "any", "sha256": "any", "triggered": "any"}))}
PUSH_LIST_KEYS = ("format", "response", "unlock_entry", "adding_commit", "t_lo_utc", "remote", "pushes")   # TB11
BINDING_REL = "reports/derived/tday_binding.json"


@dataclass
class Result:
    mode: str
    passed: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    reports: list = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return S.EXIT_MISSING if self.missing else S.EXIT_PASS


class Ctx:
    """Everything the gates read. `records` is the committed records; `runs_root` the local run directories
    (freeze, unlock, session); `freeze_record` and `unlock_entry` committed entry-file paths; `registered_cells`,
    `ptq_reader` and `trigger_reader` stand in for src.corruption_roles, lane LQP's reader and L-AM3's trigger reader
    (records.TRIGGER_READER) until they merge; `artifact_runtimes` are projected eval_runtime records of scored
    artifacts (B-3b) compared with the execution context."""

    def __init__(self, *, mode: str, manifest_bytes: bytes, manifest_path: str, records: R.Records, runs_root=None,
                 freeze_record: str | None = None, unlock_entry: str | None = None, registered_cells=None,
                 ptq_reader=None, trigger_reader=None, artifact_runtimes=(), now: float | None = None,
                 repo_root=None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}")
        self.mode, self.data, self.path, self.records = mode, manifest_bytes, manifest_path, records
        self.runs_root = None if runs_root is None else Path(runs_root)
        self.freeze_rel, self.unlock_rel = freeze_record, unlock_entry
        self.registered_cells, self.ptq_reader = registered_cells, ptq_reader
        self.trigger_reader = R.TRIGGER_READER if trigger_reader is None else trigger_reader
        self.trigger_verdict: bool | None = None
        self.artifact_runtimes, self.now = list(artifact_runtimes), now
        self.repo_root = Path(records.src.root if repo_root is None else repo_root)
        self.doc = self.body = self.cat = self.freeze = self.dl = None
        self.by_slot: dict = {}
        self.states: dict = {}            # run_id -> completion.RunState
        self.entry_data: dict = {}        # rel -> committed bytes
        self.entry_files: dict = {}       # rel -> (doc, sha256)
        self.freeze_conflict: S.TmanRefused | None = None
        self.ptq_ok: set = set()
        self._index: C.RunIndex | None = None

    # ----------------------------------------------------------------------------------------- helpers
    @property
    def entries(self) -> list:
        return self.body["entries"]

    def slot(self, k: str) -> dict:
        return self.cat.slots[k]

    def instant(self, slot: str) -> int:
        """The instant a run is judged complete at: the original freeze instant, the moved one for a never-cut run
        (AM-19 item 1(c))."""
        return self.freeze.moved_utc if self.slot(slot)["never_cut"] else self.freeze.original_utc

    def run_dir(self, run_id: str) -> Path:
        S.refuse_test_path(run_id)
        if self.runs_root is None:                    # a gate before launched_attempt_without_directory needs --runs
            raise S.TmanRefused("launched_attempt_without_directory", "freeze, unlock and session modes read the run "
                                                                      "directories (--runs)")
        S.refuse_test_path(run_id, self.runs_root)
        return self.runs_root / run_id

    def state(self, e: dict) -> C.RunState:
        rid = e["run_id"]
        if rid not in self.states:
            self.states[rid] = C.run_state(self.slot(e["slot"]), self.run_dir(rid))
        return self.states[rid]

    def index(self) -> C.RunIndex:
        """Every run slot's launched attempts, from the launch log and the run directories (completion.RunIndex);
        it shares the run states read here."""
        if self._index is None:
            self._index = C.RunIndex(self.cat, self.launch_log(), self.runs_root, states=self.states)
        return self._index

    def slot_status(self, slot: str) -> str | None:
        """A run slot's status at the original freeze instant, from its own launched attempts."""
        return self.index().status(slot, self.freeze.original_utc)

    def entry_sha(self, rel: str) -> str:
        """An entry file's committed sha256 (record_uncommitted when it is not committed)."""
        if rel not in self.entry_data:
            self.entry_data[rel] = self.records.read(rel)
        return S.sha256_bytes(self.entry_data[rel])

    def entry_file(self, rel: str) -> tuple[dict, str]:
        if rel not in self.entry_files:
            sha = self.entry_sha(rel)
            self.entry_files[rel] = (S.strict_json(self.entry_data[rel], rel, code="entry_file_invalid"), sha)
        return self.entry_files[rel]

    def entry_doc(self, rel: str):
        """An entry file's document for finding the files it cites, or None when it does not parse (entry_file_invalid
        refuses it at its gate)."""
        try:
            return self.entry_file(rel)[0]
        except S.TmanRefused as e:
            if e.code != "entry_file_invalid":
                raise
            return None

    def cited_entries(self) -> list[dict]:
        """Every entry-file citation the validation reads (decision (iii)): {path, sha256 (the cited one; None: cited by
        path), kind (the kind it is cited as), entry (the manifest entry it records, or None)}: each cut, not-run and
        not-made entry's file; the condition entry an E6-X named-rule cut cites (AM-23 item 3(b)); the freeze delay or
        new freeze the schedule cites, and the delay the new freeze cites; the freeze record and the unlock entry; in
        session mode the clock-offset record; at the unlock the lane register's not-required entries and the cut
        entries they cite. A file cited twice is listed twice, each citation matched; a file that does not parse
        cites nothing here (entry_file_invalid refuses it)."""
        out = [{"path": e["entry"]["path"], "sha256": e["entry"]["sha256"], "kind": e["kind"], "entry": e}
               for e in self.entries if e["kind"] in ("cut", "not_run", "not_made")]

        def ref(r, kind):
            if isinstance(r, dict) and isinstance(r.get("path"), str):
                out.append({"path": r["path"], "sha256": r.get("sha256"), "kind": kind, "entry": None})
                return r["path"]
            return None
        for c in [c for c in out if c["kind"] == "cut"]:
            doc = self.entry_doc(c["path"])
            slots = doc.get("slots") if isinstance(doc, dict) and doc.get("rule") == "named_rule" else None
            if isinstance(slots, list) and any(isinstance(k, str) and k in self.cat.slots
                                               and self.slot(k)["stage"] in E.CONDITION_STAGES for k in slots):
                ref(doc.get("rule_record"), "condition")
        src = self.body["schedule"].get("freeze_source")
        if isinstance(src, dict) and src.get("provider") in ("freeze_delay", "new_freeze"):
            rel = ref(src, src["provider"])
            d = self.entry_doc(rel) if rel and src["provider"] == "new_freeze" else None
            ref(d.get("delay_entry") if isinstance(d, dict) else None, "freeze_delay")
        ref({"path": self.freeze_rel} if self.freeze_rel else None, "freeze")
        ref({"path": self.unlock_rel} if self.unlock_rel else None, "unlock")
        s = self.doc["session"] if self.mode == "session" else None
        ref(s.get("clock_offset") if isinstance(s, dict) else None, "clock_offset")
        for lane in self.lanes_listed() if self.mode in U else []:
            st = lane.get("status") if isinstance(lane, dict) else None
            rel = ref(st.get("entry") if isinstance(st, dict) and st.get("kind") == "not_required" else None,
                      "not_required")
            d = self.entry_doc(rel) if rel else None
            cuts = d.get("cut_entries") if isinstance(d, dict) else None
            for c in cuts if isinstance(cuts, list) else []:
                ref(c, "cut")
        return out

    def cited_paths(self) -> list[str]:
        return list(dict.fromkeys(c["path"] for c in self.cited_entries()))

    def lanes_listed(self) -> list:
        """The committed lane register's lanes as listed, unchecked ([] when it is not committed or does not parse;
        lane_unmet checks it at the unlock)."""
        if not self.records.has(S.LANES_REL):
            return []
        try:
            doc = S.strict_json(self.records.read(S.LANES_REL), S.LANES_REL, code="lane_unmet")
        except S.TmanRefused as e:
            if e.code != "lane_unmet":
                raise
            return []
        lanes = doc.get("lanes") if isinstance(doc, dict) else None
        return lanes if isinstance(lanes, list) else []

    def _cut_refs(self, field: str) -> list[dict]:
        """`field`'s references in the cut entry files, well-formed ones only (the entry-file gate checks the
        fields)."""
        out = []
        for c in self.cited_entries():
            doc = self.entry_doc(c["path"]) if c["kind"] == "cut" else None
            refs = doc.get(field) if isinstance(doc, dict) else None
            refs = [refs] if isinstance(refs, dict) else refs
            out += [r for r in refs if isinstance(r, dict) and isinstance(r.get("path"), str)
                    and isinstance(r.get("sha256"), str)] if isinstance(refs, list) else []
        return out

    def results_known(self) -> list[dict]:
        """The results-of-record references the cut entry files list (decision (xvi))."""
        return self._cut_refs("results_known")

    def rule_records(self) -> list[dict]:
        """The records the named-rule cuts cite (AM-19 item 3(b))."""
        return self._cut_refs("rule_record")

    def qat_hashed(self) -> list[dict]:
        """The QAT selections' hashed forms, in the body and appended (their fields are derived_inputs_incomplete's
        and append_invalid's)."""
        frozen = {e["slot"]: e for e in self.entries}
        out = [e.get("hashed") for e in self.entries if e["kind"] == "derived" and e.get("method") == "qat_selection"]
        out += [a.get("hashed") for a in self.doc["appends"]
                if (frozen.get(_append_slot(a)) or {}).get("method") == "qat_selection"]
        return [h for h in out if isinstance(h, dict)]

    def fault_reports(self) -> list[dict]:
        """The fault reports the QAT selections' later evaluation directories cite (AM-21 item 2(c); DL-24 #77)."""
        evs = [ev for h in self.qat_hashed() if isinstance(h.get("evaluations"), list) for ev in h["evaluations"]]
        return [ev["fault_report"] for ev in evs if isinstance(ev, dict) and isinstance(ev.get("fault_report"), dict)
                and isinstance(ev["fault_report"].get("path"), str)
                and isinstance(ev["fault_report"].get("sha256"), str)]

    def launch_log(self):
        if not hasattr(self, "_log"):
            self._log = self.records.launch_log(code="completion_record_invalid") \
                if self.records.has(ss.LAUNCH_LOG_REL) else None
        return self._log


def _err(errs, n: int = 4) -> str | None:
    return "; ".join(errs[:n]) + (f" (+{len(errs) - n} more)" if len(errs) > n else "") if errs else None


# ================================================================================================ input gates
def g_test_path(x: Ctx):
    """Each input path, before any call on it: the manifest, the runs root, the root S and the records folder, the
    freeze record and the unlock entry (DL-24 #7)."""
    for p in [x.path, x.runs_root, x.repo_root, getattr(x.records.src, "folder", None), x.freeze_rel, x.unlock_rel]:
        if p is not None and S.test_path_error(p):
            return S.test_path_error(p)


def g_manifest_unreadable(x: Ctx):
    x.doc = S.strict_json(x.data, x.path, code="manifest_unreadable")


def g_manifest_format(x: Ctx):
    d = x.doc
    if not isinstance(d, dict) or sorted(d) != list(S.MANIFEST_PARTS) or d["format"] != S.MANIFEST_FORMAT:
        return f"the manifest is not {S.MANIFEST_FORMAT} with exactly the parts {list(S.MANIFEST_PARTS)}"
    b = d["body"]
    if not isinstance(b, dict) or sorted(b) != list(S.BODY_BLOCKS) or not isinstance(b["entries"], list) \
            or not isinstance(d["appends"], list) or not (d["session"] is None or isinstance(d["session"], dict)) \
            or not isinstance(b["schedule"], dict) or not isinstance(b["evaluator"], dict):
        return f"the body holds {sorted(b) if isinstance(b, dict) else b!r}, not the blocks {list(S.BODY_BLOCKS)}"
    errs = [err for k, spec in BODY_SPECS.items() for err in S.value_errors(b[k], spec, f"body.{k}")]
    if errs:
        return _err(errs)
    x.body = b


def g_catalogue_invalid(x: Ctx):
    f = x.repo_root / S.CATALOGUE_REL
    x.cat = S.load_catalogue(f.read_bytes() if f.is_file() else b"", S.CATALOGUE_REL)


def g_expected_rows_mismatch(x: Ctx):
    v = (x.body["expected_rows"] or {}).get("value") if isinstance(x.body["expected_rows"], dict) else None
    if v != S.EXPECTED_ROWS:
        return f"expected_rows {v!r} != {S.EXPECTED_ROWS} (AM-17 item 10(a); TB14)"


def g_entry_kind_unknown(x: Ctx):
    bad = [e.get("kind") if isinstance(e, dict) else e for e in x.entries
           if not isinstance(e, dict) or e.get("kind") not in S.ENTRY_KINDS]
    return f"entry kinds {bad} outside {list(S.ENTRY_KINDS)}" if bad else None


def g_entry_field(x: Ctx):
    return _err([err for e in x.entries for err in S.entry_errors(e)])


def g_slot_unknown(x: Ctx):
    bad = [e["slot"] for e in x.entries if e["slot"] not in x.cat.slots]
    return f"{bad} are not catalogue slots" if bad else None


def g_slot_duplicate(x: Ctx):
    seen, dup = set(), []
    for e in x.entries:
        if e["slot"] in seen:
            dup.append(e["slot"])
        seen.add(e["slot"])
    x.by_slot = {e["slot"]: e for e in x.entries}
    return f"slots entered twice: {dup}" if dup else None


def g_derived_counted_as_run(x: Ctx):
    """An entry's kind is one its slot takes (schema.allowed_kinds: its class and the kind its role gives a complete
    run, DL-24 #2); its class, stage, seed, catalogue block, role, evaluator stage (#91) and, derived, its method
    (the slot's record kind, #21) are the catalogue's."""
    errs = []
    for e in x.entries:
        s = x.slot(e["slot"])
        if e["kind"] not in S.KINDS_BY_CLASS[s["class"]] or e["class"] != s["class"]:
            errs.append(f"{e['slot']}: kind {e['kind']} of class {e['class']}; the slot is a {s['class']}")
        elif e["kind"] not in S.allowed_kinds(s):
            errs.append(f"{e['slot']}: a {s['scope_class']} slot of role {s['roles']} does not take kind {e['kind']} "
                        f"(it takes {list(S.allowed_kinds(s))})")
        elif (e["stage"], e["seed"], e["catalogue"], e["role"]) != (s["stage"], s["seed"], x.cat.entry_block(e["slot"]),
                                                                    s["roles"]):
            errs.append(f"{e['slot']}: stage, seed, catalogue block or role differ from the catalogue's")
        elif "evaluator_stage" in e and s["evaluator_stage"] is not None \
                and e["evaluator_stage"] != s["evaluator_stage"]:            # a listed slot's is candidate_scored's
            errs.append(f"{e['slot']}: evaluator stage {e['evaluator_stage']!r} is not the catalogue's "
                        f"{s['evaluator_stage']!r}")
        elif e["kind"] == "derived" and e["method"] != s["record_kind"]:
            errs.append(f"{e['slot']}: method {e['method']!r} is not the slot's record kind {s['record_kind']!r}")
    return _err(errs)


# ============================================================================================== record gates
def g_record_uncommitted(x: Ctx):
    rels = [S.CATALOGUE_REL, x.body["schedule"].get("file") or "configs/kd_schedule.json"]
    if x.path == S.MANIFEST_REL or x.mode in U:
        rels.append(S.MANIFEST_REL)
    rels += x.cited_paths()
    ci = x.body["calibration_index"]
    if isinstance(ci, dict):
        rels += [v["path"] for v in [ci.get("of_record")] + list((ci.get("sensitivity") or {}).values())
                 if isinstance(v, dict)]
    refs = x.body["references"] if isinstance(x.body["references"], dict) else {}
    rels += [refs[k]["path"] for k in ("mde_entry", "strata") if isinstance(refs.get(k), dict)]
    if isinstance(x.body["trigger"], dict):
        rels.append(x.body["trigger"]["path"])
    if x.mode in U:
        rels.append(S.LANES_REL)
    if any(e["kind"] == "derived" and e.get("method") == "qat_selection" for e in x.entries):
        rels.append(M.QAT_RULES_REL)                       # TB13
    for rel in rels:
        x.records.read(rel)
    if x.mode in U and x.records.read(S.MANIFEST_REL) != x.data:
        return f"the manifest validated is not the committed {S.MANIFEST_REL}"
    cited = [e["selection"] for e in x.entries if e["kind"] == "default_candidate"]
    if isinstance(x.body["trigger"], dict):
        cited.append(x.body["trigger"])                  # the AM-3 trigger record by its cited sha256 (#1)
    cited += [r for e in x.entries if e["kind"] == "arm_teacher" for r in (e["arm_record"], e["launch_provenance"],
                                                                           e["go_rule"]) if r]
    cited += x.results_known()                             # each committed at its cited sha256 (decision (xvi); #69)
    cited += x.rule_records()                              # a named rule's record, at its cited sha256 (#16)
    cited += x.fault_reports()                             # AM-21 item 2(c)'s fault reports (#77)
    for ref in cited:
        if S.sha256_bytes(x.records.read(ref["path"])) != ref["sha256"]:
            return f"{ref['path']}: the committed file is not the cited sha256 {ref['sha256'][:12]}"


def g_dl_row_missing(x: Ctx):
    x.dl = x.records.decision_log()
    errs = [f"{e['slot']}: {r} is not a decision-log row" for e in x.entries for r in e["dl_rows"] if not x.dl.has(r)]
    for rel in x.cited_paths():
        sha = x.entry_sha(rel)
        if not x.dl.recorded_rows_with(sha):
            errs.append(f"{rel}: sha256 {sha[:12]} is in no RECORDED or IMPLEMENTED row (decision (iii))")
    for r in x.results_known():
        if isinstance(r.get("dl"), str) and r["sha256"] not in (x.dl.row_text(r["dl"]) or ""):
            errs.append(f"{r['path']}: decision-log row {r['dl']} does not hold its sha256 (decision (xvi))")
    return _err(errs)


def g_entry_file_rewritten(x: Ctx):
    errs = []
    for c in x.cited_entries():
        sha = x.entry_sha(c["path"])
        if c["sha256"] not in (None, sha):
            errs.append(f"{c['path']}: committed sha256 {sha[:12]} is not the cited {str(c['sha256'])[:12]}")
    for rel in x.cited_paths():
        err = x.records.written_once_error(rel)
        if err:
            errs.append(err)
    return _err(errs)


def _entry_match(c: dict, doc: dict) -> str | None:
    """A manifest entry's file records that entry: its slot, and a cut's date, rule and comparator; a not-run's
    reason, parent and parent record (a not-made's one reason is the enum's)."""
    e = c["entry"]
    if e is None:
        return None
    if e["slot"] not in doc["slots"]:
        return f"{e['slot']}: its {e['kind']} entry file {c['path']} names {doc['slots']}"
    if e["kind"] == "cut":
        cmp = doc["comparator"]["slot"] if doc["comparator"] else None
        if (e["cut_date"], e["rule"], e["comparator"]) != (doc["date"], doc["rule"], cmp):
            return f"{e['slot']}: its cut label is not its entry file's"
    elif e["kind"] == "not_run" and (e["reason"], e["parent"], e["parent_record"]) != (doc["reason"], doc["parent"],
                                                                                         doc["parent_record"]):
        return f"{e['slot']}: its reason, parent and parent record are not its not-run entry file's (AM-19 item 4(a))"
    return None


def g_entry_file_invalid(x: Ctx):
    """Each cited entry file is a tday_entry/1 of the kind it is cited as, its fields and its TB7 folder and name
    (entries.entry_file_errors), naming catalogue slots of the classes its kind applies to; a manifest entry's file
    records that entry (decision (iii); DL-24 #17, #38, #87-#89)."""
    errs = []
    for c in x.cited_entries():
        doc, _ = x.entry_file(c["path"])
        bad = E.entry_file_errors(doc, c["kind"], c["path"])
        if bad:
            errs += bad
            continue
        unknown = [k for k in doc["slots"] if k not in x.cat.slots]
        wrong = [k for k in doc["slots"] if k in x.cat.slots and c["kind"] in ("cut", "not_run", "not_made")
                 and c["kind"] not in S.KINDS_BY_CLASS[x.slot(k)["class"]]]
        err = (f"{c['path']}: {unknown} are not catalogue slots" if unknown else
               f"{c['path']}: a {c['kind']} entry names {wrong}, slots of another class" if wrong else
               _entry_match(c, doc))
        if err:
            errs.append(err)
    return _err(errs)


def g_evaluator_commit_invalid(x: Ctx):
    ev = x.body["evaluator"]
    e = ev.get("commit")
    if e is None:
        return None if x.mode == "draft" else "no evaluator commit E"
    src = x.records.src
    if not (isinstance(e, str) and S.COMMIT_RE.fullmatch(e) and src.commit_exists(e)
            and src.is_ancestor(e, src.records_ref)):
        return f"evaluator commit {e!r} is not a commit of the records' history ({src.records_ref[:12]})"
    if x.mode in U and src.head() != e:
        return f"the session checkout S = {src.head()[:12]} is not E = {e[:12]} (TB16: S = E)"
    if ev.get("files") != M.evaluator_files(src, e):
        return f"the recorded evaluator file set is not the one at {e[:12]} (decision (x); IF§10)"


def g_execution_context_split(x: Ctx):
    ec = x.body["execution_context"]
    if ec is None:
        return None
    if not isinstance(ec, dict) or sorted(ec) != sorted(TUPLE_KEYS):
        return f"the execution context is not one tuple of {list(TUPLE_KEYS)} (TB1: no per-precision form)"
    if x.mode == "draft":
        return None
    bad = [k for k, v in TUPLE_FIXED.items() if ec[k] != v]
    if not (S.TYPES["int"](ec["torch_num_threads"]) and ec["torch_num_threads"] >= 1) \
            or not (S.TYPES["int"](ec["processes"]) and ec["processes"] >= 1):
        bad.append("torch_num_threads/processes")
    if ec["evaluator_commit"] != x.body["evaluator"].get("commit"):
        bad.append("evaluator_commit")
    if not (isinstance(ec["host"], dict) and ec["host"].get("name") and ec["host"].get("cpu_model")):
        bad.append("host")
    for rt in x.artifact_runtimes:
        got = R.project(rt, R.EVAL_RUNTIME)
        if any(got.get(k) != ec[k] for k in R.EVAL_RUNTIME):
            bad.append(f"an artifact's runtime {got}")
    return f"the execution context differs from TB1's tuple at {bad}" if bad else None


def g_calibration_index_mismatch(x: Ctx):
    ci = x.body["calibration_index"]
    if ci is None:
        return None
    if ci != M.calibration_index(x.records):
        return "the calibration index differs from the committed lists (§1.6; DL-10)"
    return _err(M.calibration_dl10_errors(x.records, ci, x.dl.row_text("DL-10")))


def g_reference_mismatch(x: Ctx):
    refs = x.body["references"]
    if refs != M.references(x.records):
        return "the references differ from the committed files and the values of record (§1.8)"
    errs = [f"{k}: sha256 {v['sha256'][:12]} is not in {v['dl']}" for k, v in refs.items()
            if v["sha256"] not in (x.dl.row_text(v["dl"]) or "")]
    for key, rel, field_ in (("lambda_logit", ss.LAMBDA_SELECTION_REL, "lambda"),
                             ("alpha_cwd", ss.ALPHA_SELECTION_REL, "alpha")):
        if not x.records.has(rel):
            continue
        doc, _ = x.records.json(rel, R.SELECTION, code="reference_mismatch")
        win = doc.get("winner") or {}
        rec = [s for s in x.cat.sweep_slots(key) if s["sweep"]["of_record"]]
        if not rec or float(win.get(field_, -1)) != float(rec[0]["sweep"]["value"]):
            errs.append(f"{rel}: winner {field_} {win.get(field_)!r} is not the catalogue's slot of record")
            continue
        e = x.by_slot.get(rec[0]["slot"])
        if e and e.get("checkpoint") and e["checkpoint"]["sha256"] != win.get("ckpt_sha256"):
            errs.append(f"{rec[0]['slot']}: checkpoint sha256 is not {rel}'s winner's")
    return _err(errs)


def g_teacher_values_mismatch(x: Ctx):
    t = M.TEACHER_OF_RECORD
    e = x.by_slot.get("teacher:s42:40k")
    if e is None or e["kind"] != "scored_run":
        return None
    ck = e["checkpoint"]
    if (ck["name"], ck["sha256"], ck["bytes"], e["teacher_ckpt_sha256"]) != (t["path"], t["sha256"], t["bytes"],
                                                                              t["sha256"]):
        return f"the teacher of record's entry is not DL-88's {t['path']} {t['sha256'][:12]} {t['bytes']} B"


def g_image_digest_not_of_record(x: Ctx):
    errs = []
    ec = x.body["execution_context"]
    if isinstance(ec, dict) and ec.get("image_digest") not in (None, S.KD_IMAGE):
        errs.append(f"execution context image {ec['image_digest']} is not the KD image of record (TB1)")
    for e in x.entries:
        if e["kind"] == "derived" and e["inputs"].get("image_digest") not in S.IMAGE_DIGESTS_OF_RECORD:
            errs.append(f"{e['slot']}: image digest {e['inputs'].get('image_digest')} is not of record")
    return _err(errs)


def _freeze_docs(x: Ctx) -> tuple[list, list]:
    """The freeze-source entry files the schedule block cites; a malformed citation cites none (schedule_invalid
    compares the whole block; DL-24 #90)."""
    src = x.body["schedule"].get("freeze_source")
    delays, news = [], []
    if not (isinstance(src, dict) and isinstance(src.get("path"), str)):
        return delays, news
    if src.get("provider") == "new_freeze":
        doc, sha = x.entry_file(src["path"])
        news.append({"doc": doc, "path": src["path"], "sha256": sha})
        d = doc.get("delay_entry") or {}
        if d.get("path"):
            ddoc, dsha = x.entry_file(d["path"])
            delays.append({"doc": ddoc, "path": d["path"], "sha256": dsha})
    elif src.get("provider") == "freeze_delay":
        doc, sha = x.entry_file(src["path"])
        delays.append({"doc": doc, "path": src["path"], "sha256": sha})
    return delays, news


def g_schedule_invalid(x: Ctx):
    sch = D.schedule_block(x.records.src)
    delays, news = _freeze_docs(x)
    try:
        x.freeze = D.freeze_instant(sch, delays=delays, news=news)
    except S.TmanRefused as e:
        if e.code != "freeze_source_conflict":
            raise
        x.freeze_conflict, x.freeze = e, D.freeze_instant(sch)
        return None
    if x.body["schedule"] != x.freeze.as_block():
        return "the schedule block differs from the schedule in force and its freeze entries (plan §2.3)"


SETTLED_KINDS = ("not_run", "diverged", "aborted", "superseded")      # a never-cut slot's ending that is no completion


def never_cut_slots(cat, by_slot: dict, triggered: bool) -> list:
    """The never-cut run slots (E6-KD only when AM-3 triggered it; an absent E6-KD is trigger_inconsistent's)."""
    return [k for k, s in cat.slots.items() if s["never_cut"] and s["class"] == "run"
            and not (s["stage"] == "E6-KD" and (not triggered or k not in by_slot))]


def never_cut_instants(cat, by_slot: dict, triggered: bool, entry_state, index: C.RunIndex) -> dict:
    """{never-cut run slot: its completion instant, or None without one}: an entered run's from its entry's run
    directory (entry_state); an absent slot's, the earliest of its own launched attempts' (completion.RunIndex). A
    slot entered not run, diverged, aborted or superseded has ended without a completion and is left out. The one
    reading of AM-19 item 1(d), the validator's and the freeze-delay and new-freeze printers' (DL-24 #8, #74)."""
    out = {}
    for k in never_cut_slots(cat, by_slot, triggered):
        e = by_slot.get(k)
        if e is not None and e["kind"] in SETTLED_KINDS:
            continue
        if e is not None and e["kind"] in M.COMPLETE_KINDS + ("non_finite",):
            st = entry_state(e)
            out[k] = st.completion.at if st.completion else None
        else:
            done = [index.state(r).completion.at for r in index.attempts(k) or []
                    if index.state(r).completion is not None]
            out[k] = min(done) if done else None
    return out


def late_never_cut(instants: dict, original_utc) -> list:
    """The never-cut runs not complete at the original freeze instant (AM-19 item 1(d))."""
    return sorted(k for k, t in instants.items() if t is None or not D.meets(t, original_utc))


def _never_cut_slots(x: Ctx) -> list:
    return never_cut_slots(x.cat, x.by_slot, (x.body["trigger"] or {}).get("triggered") is True)


def _never_cut_instants(x: Ctx) -> dict:
    return never_cut_instants(x.cat, x.by_slot, (x.body["trigger"] or {}).get("triggered") is True, x.state,
                              x.index())


def g_freeze_source_conflict(x: Ctx):
    """AM-19 item 1(d): a never-cut run not complete at the original freeze instant moves the freeze. With no delay
    entry every never-cut run is complete then; a delay entry lists exactly the late ones, and the manifest stays a
    draft until the new freeze, the first later date at whose end every never-cut run is complete, when each late run
    is entered. (A never-cut slot complete at the original instant and absent is complete_run_omitted's.)"""
    if x.freeze_conflict is not None:
        return x.freeze_conflict.detail
    if x.runs_root is None:                      # draft: the absent ones only; F/U/S: launched_attempt_... refuses
        absent = [k for k in _never_cut_slots(x) if k not in x.by_slot] if x.mode == "draft" else []
        return f"never-cut runs without an entry: {absent} (AM-19 items 1(d), 3(d))" if absent else None
    delays, _ = _freeze_docs(x)
    inst = _never_cut_instants(x)
    late = late_never_cut(inst, x.freeze.original_utc)
    if not delays:
        return None if not late else f"the never-cut runs {late} are not complete at the original freeze instant: " \
                                     "the freeze moves, and the delay is entered on the original freeze date (AM-19 " \
                                     "item 1(d))"
    errs = E.freeze_delay_errors(delays[0]["doc"], x.freeze, late)
    if errs:
        return "; ".join(errs) + " (AM-19 item 1(d))"
    if x.freeze.moved_date is None:
        return f"the delay moves the freeze for {late}: the manifest stays a draft until the new freeze is entered " \
               "(AM-19 item 1(d))"
    want = D.first_complete_day(list(inst.values()), x.freeze.original_date)
    if want != x.freeze.moved_date:
        return f"the new freeze {x.freeze.moved_date} is not the first later date at whose end every never-cut " \
               f"run is complete ({want}; AM-19 item 1(d))"
    absent = [k for k in late if k not in x.by_slot]
    if absent:
        return f"the late never-cut runs {absent} have no entry at the new freeze (AM-19 item 1(d))"


def g_content_class_missing(x: Ctx):
    b, miss = x.body, []
    if not b["evaluator"].get("commit"):
        miss.append("evaluator commit")
    if not isinstance(b["execution_context"], dict):
        miss.append("execution context")
    if not b["calibration_index"]:
        miss.append("calibration index")
    gp = b["generator_pins"] if isinstance(b["generator_pins"], dict) else {}
    for k in ("corrupted_cache", "image_hasher"):
        if not (isinstance(gp.get(k), dict) and gp[k].get("pin")):
            miss.append(f"{k} pin")
    if not isinstance(b["trigger"], dict):
        miss.append("trigger record")
    miss += [f"{e['slot']} checkpoint" for e in x.entries if e["kind"] in ("scored_run", "default_candidate",
                                                                            "arm_teacher") and not e["checkpoint"]]
    return f"absent: {miss} (AM-17 item 9)" if miss else None


def _pins(x: Ctx) -> dict:
    return x.body["generator_pins"] if isinstance(x.body["generator_pins"], dict) else {}


def g_generator_pin_incomplete(x: Ctx):
    gp = _pins(x)
    return _err(M.pin_incomplete_errors((gp.get("corrupted_cache") or {}).get("pin"))
                + M.hasher_pin_errors((gp.get("image_hasher") or {}).get("pin")))


def g_generator_pin_sha256_mismatch(x: Ctx):
    pin = (_pins(x).get("corrupted_cache") or {}).get("pin")
    return M.pin_sha256_error(pin) if isinstance(pin, dict) else None


def g_generator_cells_mismatch(x: Ctx):
    return _err(M.pin_cells_errors(_pins(x)["corrupted_cache"]["pin"], x.registered_cells))


def g_test_value_before_unlock(x: Ctx):
    errs = M.test_value_errors(_pins(x)) + M.body_value_errors(x.body)
    if x.doc["session"] is not None and x.mode != "session":
        errs.append("a session part exists before the session (M11)")
    return _err(errs)


def g_trigger_format_unknown(x: Ctx):
    """Plan O-8: the AM-3 trigger record is read only through L-AM3's reader (records.TRIGGER_READER, or the Ctx's
    stand-in); until that lane fixes the format there is none, so freeze, unlock and session refuse and a draft lists
    it. The reader's verdict is the record's: trigger_inconsistent compares the trigger block with it."""
    t = x.body["trigger"]
    if not isinstance(t, dict):
        return None                                  # content_class_missing names an absent trigger record
    if x.trigger_reader is None:
        return f"{t.get('path')}: no reader of the AM-3 trigger record: L-AM3 has not fixed its format (plan O-8)"
    try:
        got = x.trigger_reader(t["path"], x.records.read(t["path"]))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return f"{t['path']}: the trigger reader refuses it ({type(e).__name__}: {e})"
    if not (isinstance(got, dict) and isinstance(got.get("triggered"), bool)):
        return f"{t['path']}: the trigger reader gives no triggered verdict"
    x.trigger_verdict = got["triggered"]


def g_trigger_inconsistent(x: Ctx):
    t = x.body["trigger"]
    kd = [e for e in x.entries if e["stage"] == "E6-KD"]
    if not isinstance(t.get("triggered"), bool):
        return "the trigger record states no triggered verdict (AM-3)"
    if x.trigger_verdict is not None and t["triggered"] != x.trigger_verdict:
        return f"the trigger block says triggered {t['triggered']}; the AM-3 trigger record says {x.trigger_verdict}"
    if not t["triggered"] and kd:
        return f"AM-3 is not triggered, yet {[e['slot'] for e in kd]} are entered"
    if t["triggered"] and "E6-KD:s42:15ep" not in x.by_slot:
        return "AM-3 is triggered, yet E6-KD:s42:15ep is not entered (never cut if triggered)"


# ========================================================================================== completion gates
def _run_entries(x: Ctx) -> list:
    return [e for e in x.entries if e["kind"] in RUN_KINDS and e.get("run_id") is not None]


def _cut_run_ids(x: Ctx) -> set:
    """The run_ids the cut entries' files name (their fields are the cut gates'; a malformed list names none)."""
    ids = set()
    for e in x.entries:
        if e["kind"] == "cut":
            doc = x.entry_file(e["entry"]["path"])[0]
            if isinstance(doc, dict) and isinstance(doc.get("run_ids"), list):
                ids |= {r for r in doc["run_ids"] if isinstance(r, str)}
    return ids


def g_launched_attempt_without_directory(x: Ctx):
    """AM-19a reading 20: a launched attempt is read, never taken for never launched: every entry's attempts, a
    diverged or aborted entry's runs, every run a cut entry names, and every KD launched line, under --runs."""
    if x.runs_root is None:
        return "freeze, unlock and session modes read the run directories (--runs)"
    ids = {a["run_id"] for e in _run_entries(x) for a in e.get("attempts", [])} | _cut_run_ids(x)
    for e in x.entries:
        if e["kind"] == "diverged":
            ids.add(e["run_id"])
        if e["kind"] == "aborted":
            ids |= set(e["run_ids"])
    log = x.launch_log()
    if log is not None:
        ids |= set(log.launched)
    missing = sorted(r for r in ids if not x.run_dir(r).is_dir())
    return f"launched attempts without a directory under --runs: {missing}" if missing else None


def _measurement(x: Ctx, e: dict) -> C.RunState:
    """An ARM measurement's state: its output and agreement under --runs, at the instant entered with them
    (completion.arm_state; DL-24 #36)."""
    return C.arm_state(x.runs_root, e["output"]["path"], e["agreement"]["path"], e["completion"]["at_utc"])


def g_completion_record_invalid(x: Ctx):
    """Each entry's run and every launched attempt of a run slot (completion.RunIndex) is read by its completion
    reader; a measurement's output and agreement are present under --runs."""
    for e in _run_entries(x):
        if e["kind"] in M.COMPLETE_KINDS or e["kind"] in ("non_finite", "diverged"):
            x.state(e)
    for rid in sorted(x.index().slot_of):
        x.index().state(rid)
    absent = [e["slot"] for e in x.entries if e["kind"] == "measurement" and _measurement(x, e).status != C.COMPLETE]
    if absent:
        return f"{absent}: the measurement's output and agreement are not under --runs (DL-24 #36)"


def _entry_run_ids(e: dict) -> list:
    if e["kind"] == "aborted":
        return list(e["run_ids"])
    if e["kind"] == "diverged":
        return [e["run_id"]]
    return [a["run_id"] for a in e.get("attempts", [])] + ([e["run_id"]] if e.get("run_id") else [])


def g_completion_source_mismatch(x: Ctx):
    """Each entry against its runs' records: a run entry's runs are launched attempts of its slot and its completion
    block the end record; a KD run directory's run_meta is its launched line's (DL-108); a QAT run is trained from
    its parent's checkpoint; an aborted entry names its slot's last two attempts, neither complete nor diverged, as
    they aborted (AM-19 item 3(a)); a measurement's files are the cited ones (DL-24 #36, #43, #45, #46, #52)."""
    errs = []
    idx = x.index()
    for e in x.entries:                     # a run entry's runs are launched attempts of its own slot (#52)
        if e["kind"] in RUN_KINDS and idx.known(e["slot"]):
            wrong = sorted({r for r in _entry_run_ids(e) if idx.slot_of.get(r) != e["slot"]})
            if wrong:
                errs.append(f"{e['slot']}: {wrong} are not launched attempts of this slot "
                            f"({[idx.slot_of.get(r) for r in wrong]}; the launch log and the run directories)")
    for e in _run_entries(x):
        st = x.state(e)
        if e["kind"] in M.COMPLETE_KINDS or e["kind"] == "non_finite":
            if st.completion is None or e["completion"] != st.completion.as_entry():
                errs.append(f"{e['slot']}: its completion block is not its run's end record")
        if e["kind"] in ("non_finite", "diverged") and e["telemetry_sha256"] != st.telemetry_sha256:
            errs.append(f"{e['slot']}: its telemetry sha256 is not its run's")
        if e["kind"] == "diverged" and (st.status != C.DIVERGED
                                        or (e["rule"], e["iteration"]) != (st.rule, st.iteration)):
            errs.append(f"{e['slot']}: its run's records show {st.status} ({st.rule}, {st.iteration})")
        s = x.slot(e["slot"])
        if s["record_kind"] == "qat_epoch15" and s["parent"]:
            rows, _, _ = C.qat_rows(x.run_dir(e["run_id"]))
            meta = rows[0] if rows and rows[0].get("event") == "run_meta" else {}
            pck = ((x.by_slot.get(s["parent"]) or {}).get("checkpoint") or {}).get("sha256")
            if pck is not None and meta.get("source_checkpoint_sha256") != pck:
                errs.append(f"{e['slot']}: its run's source checkpoint is not its parent {s['parent']}'s")
    log = x.launch_log()
    for rid in sorted(log.launched if log is not None else ()):
        slot = idx.slot_of.get(rid)
        if slot is None or x.slot(slot)["record_kind"] != "kd_run_end":
            continue
        rel = C.kd_telemetry_name(x.slot(slot)["stage"]).replace("_telemetry.jsonl", "_run_meta.jsonl")
        sha = R.file_sha256(x.run_dir(rid) / rel, code="completion_source_mismatch")
        if sha != log.launched[rid]["run_meta_sha256"]:
            errs.append(f"{rid}: its {rel} (sha256 {sha[:12]}) is not the launched line's run_meta (DL-108)")
    for e in x.entries:
        if e["kind"] == "aborted" and idx.known(e["slot"]):
            last = idx.by_slot.get(e["slot"], [])[-2:]
            if len(e["run_ids"]) != 2 or set(e["run_ids"]) != set(last):
                errs.append(f"{e['slot']}: an aborted entry names the run and its repeat, its slot's last two "
                            f"attempts {last}, not {e['run_ids']} (AM-19 item 3(a))")
                continue
            for rid in e["run_ids"]:
                st = idx.state(rid)
                if st.status in (C.COMPLETE, C.DIVERGED):
                    errs.append(f"{e['slot']}: {rid} is {st.status}, not aborted (AM-19 item 3(a))")
                elif st.status == C.ABORTED and (st.rule, st.cause) != (e["rule"], e["cause"]):
                    errs.append(f"{e['slot']}: {rid} aborted ({st.rule}, {st.cause}), not ({e['rule']}, "
                                f"{e['cause']})")
        if e["kind"] == "measurement":
            st = _measurement(x, e)
            shas = {k: R.file_sha256(x.runs_root / e[k]["path"], code="completion_source_mismatch")
                    for k in ("output", "agreement")}
            if any(shas[k] != e[k]["sha256"] for k in shas) or e["completion"] != st.completion.as_entry():
                errs.append(f"{e['slot']}: its output, agreement or completion block is not its files' under --runs")
    return _err(errs)


def g_checkpoint_state_flag(x: Ctx):
    errs = []
    for e in _run_entries(x):
        if x.slot(e["slot"])["record_kind"] == "qat_epoch15":
            rows, _, _ = C.qat_rows(x.run_dir(e["run_id"]))
            errs += C.qat_flag_errors(x.run_dir(e["run_id"]), rows)
    return _err(errs)


def g_entry_not_complete(x: Ctx):
    bad = [e["slot"] for e in _run_entries(x) if e["kind"] in M.COMPLETE_KINDS + ("non_finite",)
           and not x.state(e).complete_at(x.instant(e["slot"]))]
    bad += [e["slot"] for e in x.entries if e["kind"] == "measurement"
            and not _measurement(x, e).complete_at(x.instant(e["slot"]))]
    return f"{bad} are not complete at their instant (AM-19 Definitions; item 1(c))" if bad else None


slot_of_launch = C.slot_of_launch


def g_complete_run_omitted(x: Ctx):
    """AM-19 item 3(d): a run slot whose own launched attempt is complete at the original freeze instant has an
    entry (of any kind, TB10): every run slot, from the launch log and the run directories (completion.RunIndex),
    never only the KD ones. A launched run whose records name no catalogue run slot is refused too: the catalogue
    changes only with an amendment (Q12)."""
    idx = x.index()
    bad = [f"{rid} (its records name {slot}, no catalogue run slot)" for rid, slot in idx.foreign]
    if x.runs_root is not None:
        for slot, att in sorted(idx.by_slot.items()):
            if slot not in x.by_slot and any(idx.state(r).complete_at(x.freeze.original_utc) for r in att):
                bad.append(f"{slot} ({att[-1]})")
    return f"launched and not in the manifest: {bad} (AM-19 item 3(d))" if bad else None


ARM_REFS = (("arm_record", R.ARM_TEACHER_RECORD), ("go_rule", R.GO_RULE_RECORD),
            ("launch_provenance", R.LAUNCH_PROVENANCE))


def g_checkpoint_hash_mismatch(x: Ctx):
    """Each checkpoint's sources hold its sha256, and one at least is a committed record holding it (decision
    (xviii); DL-24 #35, #51): a RECORDED or IMPLEMENTED decision-log row a source names, its sweep's selection file,
    or an arm teacher's arm record, the teacher's other records agreeing with the entry (DL-24 #39)."""
    errs = []
    for e in x.entries:
        ck = e.get("checkpoint")
        if not ck:
            continue
        for src in ck["sources"]:
            if src["sha256"] != ck["sha256"] or (src["dl"] and ck["sha256"] not in (x.dl.row_text(src["dl"]) or "")):
                errs.append(f"{e['slot']}: source {src['kind']} {src['path']} does not hold {ck['sha256'][:12]}")
        rows = x.dl.recorded_rows_with(ck["sha256"])
        held = any(src["dl"] in rows for src in ck["sources"])
        s = x.slot(e["slot"])
        if s["sweep"]:
            rel = ss.LAMBDA_SELECTION_REL if s["sweep"]["key"] == "lambda_logit" else ss.ALPHA_SELECTION_REL
            if x.records.has(rel):
                doc, _ = x.records.json(rel, R.SELECTION, code="checkpoint_hash_mismatch")
                cand = [c for c in doc.get("candidates") or [] if c.get("run_id") == e.get("run_id")]
                if cand and cand[0].get("ckpt_sha256") != ck["sha256"]:
                    errs.append(f"{e['slot']}: {rel} records {str(cand[0].get('ckpt_sha256'))[:12]}")
                held = held or bool(cand)
        if e["kind"] == "arm_teacher":
            docs = [x.records.json(e[k]["path"], allow, code="checkpoint_hash_mismatch")[0] if e[k] else None
                    for k, allow in ARM_REFS]
            arm = M.arm_teacher_errors(e, M.arm_of(s), *docs)
            errs += arm
            held = held or not arm                        # the arm record's final selection holds it
        if not held:
            errs.append(f"{e['slot']}: no committed source holds its checkpoint sha256 {ck['sha256'][:12]} "
                        "(decision (xviii): a RECORDED or IMPLEMENTED row a source names, the selection file, or "
                        "the arm record)")
    return _err(errs)


def g_checkpoint_missing(x: Ctx):
    errs = []
    for e in x.entries:
        ck = e.get("checkpoint")
        if ck and e.get("run_id"):
            S.refuse_test_path(ck["name"], x.run_dir(e["run_id"]))
            f = x.run_dir(e["run_id"]) / ck["name"]
            if not f.is_file() or S.sha256_bytes(f.read_bytes()) != ck["sha256"]:
                errs.append(f"{e['slot']}: {e['run_id']}/{ck['name']} is missing or its sha256 fails (DL-110: STOP)")
    return _err(errs)


def g_restored_checkpoint_unrecorded(x: Ctx):
    bad = [e["slot"] for e in x.entries if e.get("checkpoint")
           for src in e["checkpoint"]["sources"] if src["kind"] == "restored" and not src["dl"]]
    return f"{bad}: a restored checkpoint without its AM-8a report's decision-log row" if bad else None


# ================================================================================================ cut gates
def cut_context(x: Ctx, docs: dict | None = None) -> E.CutContext:
    """The cut checks' context from the records: the arm registry, each sweep's decision record (projected:
    decision_date, values' value and status), each slot's launched attempts, the slot of each launched run_id, the
    stopped run_ids and each slot's status at the original freeze instant from its own attempts (completion.RunIndex:
    never from a cut entry), a named rule's record (entries.rule_record_reader) and E3-X's run of record (its
    manifest entry: a scored run, or none), the day each cut entry file of `docs` ({sha256: doc}) was entered in the
    decision log (decision (xv)), and the records committed by a day (decision (xvi))."""
    idx = x.index()
    sha_of = {id(d): sha for sha, d in (docs or {}).items()}

    def entered(doc):
        return E.entered_day(x.records, sha_of[id(doc)]) if id(doc) in sha_of else None

    def e3x(arm):
        e = x.by_slot.get(x.slot(f"E6-{arm}:s42:15ep")["parent"]) if f"E6-{arm}:s42:15ep" in x.cat.slots else None
        return {"run_id": e["run_id"], "checkpoint_sha256": e["checkpoint"]["sha256"]} \
            if e is not None and e["kind"] == "scored_run" else False

    def reference_error(ref, slot, a):
        """An arithmetic's reference read again under --runs: the same file, figure and detail; a telemetry
        reference, the most recently finished run of the slot's stage at the arithmetic's time (DL-24 #11)."""
        if x.runs_root is None:
            return None
        try:
            if ref["source"] == "smoke":
                got = E.reference_smoke(x.runs_root, ref["path"], ref["detail"].get("key"))
            else:
                rid = ref["path"].split("/")[0]
                latest = latest_finished(x.cat, idx, slot["stage"], D.parse_utc_any(a["now_utc"]))
                if rid not in latest:
                    return f"{rid} is not the most recently finished run of {slot['stage']} ({sorted(latest)})"
                got = E.reference_telemetry(x.runs_root, rid, slot)
        except S.TmanRefused as e:
            return e.detail
        if any(got[k] != ref[k] for k in ("path", "sha256", "median", "detail")):
            return f"{ref['path']}: the reference does not read again as recorded (its sha256, figure or detail)"
        return None
    recs = {}
    for key, rel in ss.DECISION_RECORD_REL.items():
        if x.records.has(rel):
            recs[key] = x.records.json(rel, R.DECISION_RECORD, code="decision_date_cut_mismatch")[0]
    return E.CutContext(x.cat, x.freeze, registry=E.arm_registry(x.records),
                        decision_records=recs, attempts=idx.attempts, run_slot=idx.slot_of.get,
                        stopped=idx.stopped_ids(), status=x.slot_status, rule_record=E.rule_record_reader(x.records),
                        e3x=e3x, entered=entered, known_by=E.known_by_reader(x.records),
                        reference_error=reference_error)


def latest_finished(cat, index: C.RunIndex, stage: str, now) -> set:
    """The launched runs of `stage` most recently finished by `now` (AM-19 item 3(b): the arithmetic's reference),
    from their own completion records (completion.RunIndex); the latest instant's runs, ties kept."""
    done = {}
    for slot, attempts in index.by_slot.items():
        if cat.slots[slot]["stage"] == stage:
            for r in attempts:
                c = index.state(r).completion
                if c is not None and c.at <= now:
                    done[r] = c.at
    last = max(done.values(), default=None)
    return {r for r, t in done.items() if t == last}


def _cut_gate(code: str):
    def gate(x: Ctx):
        if not hasattr(x, "_cut_docs"):
            docs = {}
            for e in x.entries:                 # each file checked and matched to its entry by entry_file_invalid
                if e["kind"] == "cut":
                    doc, sha = x.entry_file(e["entry"]["path"])
                    docs[sha] = doc
            x._cut_docs, x._cutctx = list(docs.values()), cut_context(x, docs)
        fn = dict(E.CUT_CHECKS)[code]
        return _err([f"{d['slots']}: {err}" for d in x._cut_docs for err in [fn(d, x._cutctx)] if err])
    return gate


# ============================================================================================ parent gates
def g_parent_rule_violation(x: Ctx):
    return _err(E.parent_rule_errors(x.by_slot, x.cat))


def g_descendant_entry_missing(x: Ctx):
    return _err(E.descendant_errors(x.by_slot, x.cat))


def g_not_run_without_parent_record(x: Ctx):
    return _err(E.not_run_errors(x.by_slot, x.cat))


# =========================================================================================== derived gates
def _derived(x: Ctx) -> list:
    return [e for e in x.entries if e["kind"] == "derived"]


def _append_slot(a) -> str | None:
    """An append's slot when it is a slot key (any other is append_invalid's to refuse; DL-24 #96)."""
    s = a.get("slot") if isinstance(a, dict) else None
    return s if isinstance(s, str) else None


def g_derived_parent_incomplete(x: Ctx):
    return _err([err for e in _derived(x) for err in M.derived_parent_errors(e, x.by_slot, x.cat)])


def g_derived_inputs_incomplete(x: Ctx):
    """A derived entry's inputs are all of its method's; a QAT selection's rules file is the committed one (TB13;
    DL-24 #18-#20)."""
    errs = [err for e in _derived(x) for err in M.derived_inputs_errors(e)]
    qat = [e for e in _derived(x) if e["method"] == "qat_selection"]
    if not errs and qat:
        rules = M.qat_rules_sha256(x.records)
        errs = [f"{e['slot']}: inputs.rules_sha256 is not the committed {M.QAT_RULES_REL}'s {rules[:12]} (TB13)"
                for e in qat if e["inputs"]["rules_sha256"] != rules]
    return _err(errs)


def g_derived_inputs_mismatch(x: Ctx):
    return _err([err for e in _derived(x) for err in M.derived_inputs_mismatch_errors(
        e, x.by_slot, x.cat, x.body["calibration_index"], x.records.src)])


def g_ptq_record_invalid(x: Ctx):
    """A2 for each PTQ entry, and for each PTQ append as the entry it makes; an append whose ptq_record is intact
    stands in for its isolation record (TB19 Q9; read by append_invalid)."""
    checksum = ((x.body["calibration_index"] or {}).get("of_record") or {}).get("checksum_sha256")
    errs = []
    for e in _derived(x):
        if e["method"] == "ptq":
            errs += M.ptq_record_errors(e, x.cat, x.by_slot.get(e["parent"]), checksum, runs_root=x.runs_root,
                                        reader=x.ptq_reader)
    for a in x.doc["appends"]:
        frozen = x.by_slot.get(_append_slot(a))
        if not M.appendable(a, frozen):
            continue                                                  # append_invalid refuses it
        got, ok = M.append_ptq_record(a, frozen, x.cat, x.by_slot, checksum, runs_root=x.runs_root,
                                      reader=x.ptq_reader)
        errs += got
        if ok:
            x.ptq_ok.add(a["slot"])
    return _err(errs)


def _merged_lanes(x: Ctx) -> dict:
    """{lane: merge commit} of the lanes the committed lane register lists as implemented with a merge commit made
    by the original freeze instant (its committer time; DL-24 #23, #76). The register is read when committed (it is
    required at the unlock: lane_unmet)."""
    out = {}
    for lane in x.lanes_listed():
        st = lane.get("status") if isinstance(lane, dict) else None
        mc = st.get("merge_commit") if isinstance(st, dict) and st.get("kind") == "implemented" else None
        t = x.records.commit_time(mc) if isinstance(mc, str) and S.COMMIT_RE.fullmatch(mc) else None
        if t is not None and D.meets(t, x.freeze.original_utc) and isinstance(lane.get("lane"), str):
            out[lane["lane"]] = mc
    return out


def g_derived_missing(x: Ctx):
    """A required derived slot is entered derived, or not made when its code is not merged: its catalogue's
    code_merged gate, and its lane not implemented by the freeze in the lane register when committed (AM-23 item 4;
    TB8)."""
    errs = M.derived_missing_errors(x.cat, x.by_slot)
    merged = _merged_lanes(x)
    for e in x.entries:
        lane = x.entry_file(e["entry"]["path"])[0]["lane"] if e["kind"] == "not_made" else None
        if lane in merged:
            errs.append(f"{e['slot']}: not made, but its lane {lane} is implemented with merge {merged[lane][:12]} "
                        "by the original freeze instant (AM-23 item 4: not made is for code not merged)")
    return _err(errs)


def g_e6s42_not_hashed_at_freeze(x: Ctx):
    return _err(M.e6s42_errors(x.by_slot))


def g_derived_selection_conflict(x: Ctx):
    """AM-21 item 2(c) on every QAT selection, in the body and appended (DL-24 #24)."""
    errs = [err for e in _derived(x) for err in M.selection_conflict_errors(e)]
    for app in x.doc["appends"]:
        listed = x.by_slot.get(_append_slot(app))
        if M.appendable(app, listed):
            errs += M.selection_conflict_errors(M.appended_entry(app, listed))
    return _err(errs)


def g_non_finite_invalid(x: Ctx):
    errs = []
    for e in x.entries:
        if e["kind"] != "non_finite":
            continue
        rows = sha = None
        if x.runs_root is not None:
            rows, _, sha = C.qat_rows(x.run_dir(e["run_id"]))
        errs += M.non_finite_errors(e, rows, sha, x.dl)
    return _err(errs)


def g_append_invalid(x: Ctx):
    """The appends only grow over the manifest's committed history (DL-24 #28, #54); each is checked structurally,
    then with the records it cites (DL-24 #26)."""
    errs, slots = M.appends_history_errors(x.records, x.doc["appends"]), set()
    for a in x.doc["appends"]:
        s = _append_slot(a)
        if not isinstance(a, dict) or (s is not None and s in slots):
            errs.append(f"append {a!r:.60}: malformed or a second append of its slot")
            continue
        slots.add(s)
        frozen = x.by_slot.get(s)
        first = M.append_errors(a, frozen, x.freeze.original_utc, structural=True)
        if first:
            errs += first
            continue
        read_errs, iso, prov = M.append_reads(a, frozen, x.records, x.runs_root)
        errs += read_errs or M.append_errors(a, frozen, x.freeze.original_utc, isolation=iso, provenance=prov,
                                             ptq_record_ok=a["slot"] in x.ptq_ok)
    return _err(errs)


def g_derived_hash_mismatch(x: Ctx):
    """Each hashed derived model, in the body and appended, against its records under --runs (DL-24 #27, #34, #49,
    #50); the appends are append_invalid's first."""
    if x.runs_root is None:
        return None
    made = [e for e in _derived(x) if e["hashed"] is not None]
    for app in x.doc["appends"]:
        listed = x.by_slot.get(_append_slot(app))
        if M.appendable(app, listed):
            made.append(M.appended_entry(app, listed))
    ci = x.body["calibration_index"]
    return _err([err for e in made for err in M.derived_hash_errors(e, x.by_slot.get(e["parent"]) or {}, x.cat, ci,
                                                                      x.runs_root)])


def _unlock_doc(x: Ctx) -> dict | None:
    return x.entry_file(x.unlock_rel)[0] if x.unlock_rel else None


def g_append_after_unlock(x: Ctx):
    u = _unlock_doc(x)
    if u is not None and S.canonical_sha256(x.doc["appends"]) != u["manifest"]["appends_sha256"]:
        return "the appends changed after the unlock record (AM-19 item 4(c))"


def g_derived_not_hashed_at_unlock(x: Ctx):
    hashed = {a["slot"] for a in x.doc["appends"]}
    bad = [e["slot"] for e in _derived(x) if e["hashed"] is None and e["slot"] not in hashed]
    return f"{bad} are still listed by inputs only at the unlock (AM-19 item 4(c))" if bad else None


# ============================================================================================= scope gates
def _models(x: Ctx, seed: int) -> bool:
    """Both runs of a seed's pair (E1 and E6) have a model (IF§9; TB2)."""
    e1, e6 = x.by_slot.get(f"E1:s{seed}:80k") or {}, x.by_slot.get(f"E6:s{seed}:sel") or {}
    return e1.get("kind") == "scored_run" and e6.get("kind") == "derived"


def scope_rule_of(x: Ctx, slot: str):
    s = x.slot(slot)
    return S.scope_rule(s, _models(x, s["seed"]) if s["scope_class"] == "pair_s4344" else None)


def g_scope_forbidden(x: Ctx):
    bad = [f"{e['slot']}: {sorted(set(e['scope']) & scope_rule_of(x, e['slot'])[1])}" for e in x.entries
           if e["kind"] in MODEL_KINDS and set(e["scope"]) & scope_rule_of(x, e["slot"])[1]]
    return f"forbidden scopes (IF§9): {bad}" if bad else None


def g_scope_required_missing(x: Ctx):
    bad = [f"{e['slot']}: {sorted(scope_rule_of(x, e['slot'])[0] - set(e['scope']))}" for e in x.entries
           if e["kind"] in MODEL_KINDS and scope_rule_of(x, e["slot"])[0] - set(e["scope"])]
    return f"required scopes missing (IF§9): {bad}" if bad else None


def g_candidate_scored(x: Ctx):
    bad = [e["slot"] for e in x.entries if x.slot(e["slot"])["sweep"] and e["kind"] in MODEL_KINDS
           and not x.slot(e["slot"])["sweep"]["of_record"]
           and (not x.slot(e["slot"])["sweep"]["default_candidate"] or e["kind"] != "default_candidate")]
    return f"{bad}: a sweep candidate other than the selected run and the default candidate is scored" if bad else None


def g_default_candidate_missing(x: Ctx):
    bad = [s["slot"] for s in x.cat.slots.values() if s["sweep"] and s["sweep"]["default_candidate"]
           and not s["sweep"]["of_record"] and (x.by_slot.get(s["slot"]) or {}).get("kind") != "default_candidate"]
    return f"{bad}: the default candidates are entered as default_candidate (AM-17 item 6)" if bad else None


def g_listed_or_terminal_scored(x: Ctx):
    bad = [e["slot"] for e in x.entries if e["kind"] not in MODEL_KINDS and e["scope"]]
    return f"{bad}: a listed or terminal entry carries a scope" if bad else None


# ============================================================================================ unlock gates
def g_run_added_after_freeze(x: Ctx):
    fr = x.entry_file(x.freeze_rel)[0] if x.freeze_rel else None
    if fr is None or fr.get("kind") != "freeze":
        return "the unlock reads the freeze record (--freeze-record)"
    u = _unlock_doc(x)
    if u is not None and (u.get("kind") != "unlock" or u["freeze_record"]["path"] != x.freeze_rel
                          or u["freeze_record"]["sha256"] != x.entry_file(x.freeze_rel)[1]):
        return "the unlock entry does not cite this freeze record"
    if fr["body_sha256"] != S.canonical_sha256(x.body):
        return "the body changed after the freeze record (no run is added to TEST after the freeze)"
    m = u["manifest"] if u is not None else None
    if m is not None and (m["path"] != S.MANIFEST_REL or m["body_sha256"] != S.canonical_sha256(x.body)
                          or (x.mode == "unlock" and m["sha256"] != S.sha256_bytes(x.data))):
        return "the manifest is not the one the unlock entry records (its body; in unlock mode, its file)"


def g_cut_unentered_at_unlock(x: Ctx):
    """Every run and measurement slot has an entry from the freeze on: the cut list is fixed at the original freeze
    date (AM-19 items 1(d) and 3(c); plan §1.9), so a slot that is not entered as a run is entered as cut or not run
    by then (E6-KD only when AM-3 triggered it). A draft lists them as MISSING."""
    kd_off = (x.body["trigger"] or {}).get("triggered") is not True
    bad = [k for k, s in x.cat.slots.items() if s["class"] in ("run", "measurement") and k not in x.by_slot
           and not (kd_off and s["stage"] == "E6-KD")]
    return f"{bad} have no entry: from the freeze on, a slot that is not a run is entered as cut or not run (AM-19 " \
           "items 1(d), 3(c))" if bad else None


def g_lane_unmet(x: Ctx):
    """AM-19 item 3(d), decision (iii) (DL-24 #55): the committed lane register lists the TEST-serving lanes, one at
    least; its sha256 sits in a RECORDED or IMPLEMENTED row (the current-plan line, plan §7); each lane is
    IMPLEMENTED -- its row naming the lane and its merge commit, an ancestor of S -- or entered as not required."""
    doc, sha = x.records.json(S.LANES_REL, True, code="lane_unmet")
    errs = S.field_errors(doc, LANES_SPEC, S.LANES_REL)
    if errs:
        return _err(errs)
    u = _unlock_doc(x)
    if u is not None and (u["lanes"]["path"], u["lanes"]["sha256"]) != (S.LANES_REL, sha):
        return f"{S.LANES_REL} is not the lane register the unlock entry cites"
    if not doc["lanes"]:
        return f"{S.LANES_REL} lists no lane: the lanes serving TEST are listed (plan §7)"
    src = x.records.src
    for lane in doc["lanes"]:
        st = lane["status"]
        if st["kind"] == "implemented":
            if not (st["dl"] and (x.dl.status(st["dl"]) or "").startswith("IMPLEMENTED") and st["merge_commit"]
                    and src.commit_exists(st["merge_commit"]) and src.is_ancestor(st["merge_commit"], "HEAD")):
                errs.append(f"lane {lane['lane']}: not IMPLEMENTED with a merge commit in S's history")
            elif not all(n in x.dl.row_text(st["dl"]) for n in (lane["lane"], st["merge_commit"][:7])):
                errs.append(f"lane {lane['lane']}: its IMPLEMENTED row {st['dl']} does not name the lane and its merge "
                            "commit")
        elif not st["entry"]:
            errs.append(f"lane {lane['lane']}: not required without its entry")
    if not x.dl.recorded_rows_with(sha):
        errs.append(f"{S.LANES_REL}: sha256 {sha[:12]} is in no RECORDED or IMPLEMENTED row (plan §7)")
    return _err(errs)


def g_not_required_invalid(x: Ctx):
    doc, _ = x.records.json(S.LANES_REL, True, code="lane_unmet")
    errs = []
    for lane in doc["lanes"]:
        if lane["status"]["kind"] != "not_required":
            continue
        nr, sha = x.entry_file(lane["status"]["entry"]["path"])
        if sha != lane["status"]["entry"]["sha256"] or nr.get("kind") != "not_required" \
                or nr.get("lane") != lane["lane"]:
            errs.append(f"lane {lane['lane']}: its not-required entry is not the cited one")
            continue
        covered = set()
        for ref in nr["cut_entries"]:
            cd, csha = x.entry_file(ref["path"])
            if csha == ref["sha256"] and cd.get("kind") == "cut":
                covered |= set(cd["slots"])
        if set(lane["serves"]) - covered:
            errs.append(f"lane {lane['lane']}: serves {sorted(set(lane['serves']) - covered)} with no cut entry")
            continue
        cuts = {r["path"] for r in nr["cut_entries"]}
        scored = sorted(k for k in set(lane["serves"]) if (x.by_slot.get(k) or {}).get("kind") != "cut"
                        or x.by_slot[k]["entry"]["path"] not in cuts)
        if scored:
            errs.append(f"lane {lane['lane']}: serves {scored}, not entered cut by its cut entries (DL-24 #55)")
    return _err(errs)


def g_evaluator_files_changed(x: Ctx):
    """Each evaluator file in the session checkout is E's: by its blob id, or, for a hash-only path, by its whole-file
    sha256 against the sha256 of its blob at E (manifest.hash_only)."""
    ev = x.body["evaluator"]
    src = x.records.src
    fmt = src.object_format()
    bad = []
    for rel, blob in sorted(ev["files"].items()):
        f = x.repo_root / rel
        if not f.is_file():
            bad.append(rel)
        elif M.hash_only(rel):
            if S.sha256_bytes(f.read_bytes()) != S.sha256_bytes(src.blob_bytes(blob)):
                bad.append(rel)
        elif ss.git_blob_id(f.read_bytes(), fmt) != blob:
            bad.append(rel)
    return f"evaluator files differ from E in the session checkout: {bad}" if bad else None


def g_hasher_pin_mismatch(x: Ctx):
    src, gp = x.records.src, _pins(x)
    pin, hasher = gp["corrupted_cache"]["pin"], gp["image_hasher"]["pin"]
    errs = []
    if src.blob_at("HEAD", pin["script"]["path"]) != pin["script"]["git_blob"]:
        errs.append(f"{pin['script']['path']} at S is not the pinned blob")
    hp = x.repo_root / hasher["script"]["path"]
    if not hp.is_file() or S.sha256_bytes(hp.read_bytes()) != hasher["script"]["sha256"]:
        errs.append(f"{hasher['script']['path']} at S is not the pinned script")
    return _err(errs)


def _committed_sessions(x: Ctx) -> list:
    """[(commit, session part)] of the manifest's committed versions that hold one, newest first."""
    out = []
    for commit, _ in x.records.history(S.MANIFEST_REL):
        data = x.records.show(commit, S.MANIFEST_REL)
        doc = R.strict_json(data, f"{S.MANIFEST_REL} at {commit[:12]}", code="second_session") if data else None
        part = doc.get("session") if hasattr(doc, "keys") else None
        if part is not None:
            out.append((commit, part))
    return out


def g_second_session(x: Ctx):
    """One TEST session (AM-17 item 9): no binding yet, one unlock entry in the decision log, and over the
    manifest's committed history no session part at the unlock, and in the session none but this one -- its
    identity fields unchanged, its recorded values' names only growing (the values are TEST values, not read:
    decision (vi); DL-24 #56)."""
    if x.records.has(BINDING_REL):
        return f"{BINDING_REL} exists: a binding is written once, after the single session"
    rows = [dl for dl, (_, line) in x.dl.rows.items() if "TMAN unlock entry" in line]
    if len(rows) > 1:
        return f"two unlock entries in the decision log: {rows} (AM-17 item 9: a single TEST evaluation)"
    cur = x.doc["session"] if isinstance(x.doc["session"], dict) else {}
    for commit, part in _committed_sessions(x):
        if x.mode != "session":
            return f"commit {commit[:12]} already holds a session part: a second unlock (AM-17 item 9)"
        if part == x.doc["session"]:
            continue                                   # the part validated (its fields are clock_offset_invalid's)
        old = part if isinstance(part, dict) else {}
        same = all(old.get(k) == cur.get(k) for k in SESSION_IDENTITY)
        names = (set(old["values"]) if isinstance(old.get("values"), dict) else None,
                 set(cur["values"]) if isinstance(cur.get("values"), dict) else None)
        if not (same and names[0] is not None and names[1] is not None and names[0] <= names[1]):
            return (f"commit {commit[:12]} holds another session part: one TEST session, its values only added "
                    "(AM-17 item 9)")


def g_session_after_deadline(x: Ctx):
    cutoff = D.cutoff_instant(x.freeze.test_deadline)
    if x.mode == "session":
        t = D.parse_utc_any(x.doc["session"]["started_utc"])
    else:
        u = _unlock_doc(x)
        t = D.cutoff_instant(u["date"]) - 1 if u else x.now
    if t is not None and not D.meets(t, cutoff):
        return f"the TEST session is after the TEST deadline {x.freeze.test_deadline} (AM-22 item 1(e); TB14)"


# ============================================================================================ session gates
def g_clock_offset_invalid(x: Ctx):
    """The session part (its fields, before any gate reads them: DL-24 #101), on the frozen execution context's
    host (TB1; DL-24 #37), with its clock-offset record of that host, within 60 s, before any scoring record."""
    s = x.doc["session"]
    errs = S.field_errors(s, SESSION_SPEC, "session")
    if errs:
        return _err(errs)
    if s["host"] != x.body["execution_context"]["host"]:
        return f"the session's host {s['host']} is not the frozen execution context's host (TB1)"
    doc, sha = x.entry_file(s["clock_offset"]["path"])
    if sha != s["clock_offset"]["sha256"] or doc.get("kind") != "clock_offset" \
            or abs(doc["offset_seconds"]) > S.CLOCK_OFFSET_LIMIT_S or doc["host"] != s["host"]["name"] \
            or D.parse_utc_any(doc["measured_utc"]) >= D.parse_utc_any(s["started_utc"]):
        return "the clock-offset record is not the session host's, within 60 s, before any scoring record (AM-19a r21)"


def push_evidence_error(records: R.Records, pe: dict, unlock_ref: dict, unlock_day: str,
                        started_utc: str) -> str | None:
    """TB11, the DL-109 pattern: AM-19a reading 10 as sweep_select.push_evidence reads it (DL-24 #57). The remote's
    activity response, saved verbatim, and the tday_push_list/1 list made from it were added in one commit, once,
    and are unchanged since. The list names that response, the unlock entry, the commit that added it, and T_lo,
    the first instant of the unlock entry's day (a record: no clock is the record). The response reaches back
    before T_lo. The list holds each push after T_lo (a branch deletion pushes nothing) as the response holds it,
    and nothing else. The earliest push carrying the commit lies in [T_lo, the session's start). A push carries
    the commit when its own commit is or descends from it: by the clone wherever the clone holds the pushed commit
    (a served push's listed answer is checked there), by the listed answer for a served push the clone lacks. An
    unserved push the clone lacks carries nothing here: the reading counts it toward the lambda push day, which it
    can only move earlier, and here it would vouch for an unlock that no record shows on the remote."""
    rr, lr = pe["response"]["path"], pe["list"]["path"]
    data = {k: records.read(pe[k]["path"]) for k in ("response", "list")}
    for k in ("response", "list"):
        if S.sha256_bytes(data[k]) != pe[k]["sha256"]:
            return f"push evidence {k} is not the cited file"
    err = records.written_once_error(rr) or records.written_once_error(lr)
    if err is None and records.adding_commits(rr) != records.adding_commits(lr):
        err = f"{rr} and {lr} were not added in one commit"
    if err:
        return f"{err}: the push evidence is settled once, its list committed with the response (AM-19a reading 10)"
    src = records.src
    commit = ss.adding_commit(src, unlock_ref["path"])
    t_lo, started = D.day_start(unlock_day), D.parse_utc_any(started_utc)
    try:
        timed = ss._activity_times(ss.activity_entries(data["response"], rr), what=rr)
    except ss.SelectionRefused as e:
        return f"{e} (TB11)"
    lst = R.strict_json(data["list"], lr, code="push_evidence_invalid")
    if not (isinstance(lst, dict) and sorted(lst) == sorted(PUSH_LIST_KEYS) and lst["format"] == S.PUSH_LIST_FORMAT
            and lst["response"] == pe["response"] and lst["unlock_entry"] == unlock_ref
            and commit is not None and lst["adding_commit"] == commit and lst["t_lo_utc"] == ss.iso_utc(t_lo)
            and isinstance(lst["remote"], str) and isinstance(lst["pushes"], list)
            and all(ss._push_entry_ok(p) for p in lst["pushes"])):
        return f"{lr}: not the {S.PUSH_LIST_FORMAT} list of this response, of the unlock entry and the commit that " \
               f"added it, and of T_lo {ss.iso_utc(t_lo)}, the first instant of its day (AM-19a reading 10)"
    if not any(t < t_lo for _, t in timed):
        return f"{rr}: no entry is older than T_lo {ss.iso_utc(t_lo)}: the response does not reach back far enough " \
               "(AM-19a reading 10)"
    pushes = ss._activity_pushes(timed)
    listed = {p["id"]: p for p in lst["pushes"]}
    after_lo = [e for e, t in pushes if t >= t_lo]
    for e in after_lo:
        p = listed.get(e.get("id")) if ss._is_int(e.get("id")) else None
        if p is None or any(p[k] != e.get(k) for k in ("ref", "before", "after", "timestamp", "activity_type")):
            return f"{lr}: the push {e.get('id')!r} at {e.get('timestamp')} (after T_lo) is not listed as the " \
                   "response holds it (AM-19a reading 10)"
    if len(listed) != len(lst["pushes"]) or sorted(listed) != sorted(e["id"] for e in after_lo):
        return f"{lr}: the list names pushes that are not the response's pushes after T_lo, each once"
    hits = []
    for e, t in pushes:
        after = e.get("after")
        held = isinstance(after, str) and ss._COMMIT_ID.fullmatch(after) is not None and src.commit_exists(after)
        carries = held and src.is_ancestor(commit, after)
        if t >= t_lo and listed[e["id"]]["served"]:
            if held and carries != listed[e["id"]]["carries_file"]:
                return f"{lr}: push {e['id']!r} ({after[:12]}): the list says carries_file=" \
                       f"{listed[e['id']]['carries_file']}, and the clone's commit says otherwise"
            carries = listed[e["id"]]["carries_file"]
        if carries:
            hits.append(t)
    if not hits:
        return f"{rr}: no push carries the unlock entry's commit {commit[:12]} (TB11)"
    if not t_lo <= min(hits) < started:
        return f"{rr}: the earliest push carrying the unlock entry, at {ss.iso_utc(min(hits))}, is not in [T_lo " \
               f"{ss.iso_utc(t_lo)}, the session's start {started_utc}) (TB11)"
    return None


def g_push_evidence_invalid(x: Ctx):
    """TB11 (the DL-109 pattern): the session's push evidence shows the unlock entry's commit on the remote before
    the session started (push_evidence_error)."""
    s = x.doc["session"]
    if x.unlock_rel is None or s["unlock_entry"]["path"] != x.unlock_rel:
        return "the session reads the unlock entry it names (--unlock-entry)"
    return push_evidence_error(x.records, s["push_evidence"], s["unlock_entry"], _unlock_doc(x)["date"],
                               s["started_utc"])


# ======================================================================================== the gate order
GATES = [
    ("test_path", ALL, g_test_path), ("manifest_unreadable", ALL, g_manifest_unreadable),
    ("manifest_format", ALL, g_manifest_format), ("catalogue_invalid", ALL, g_catalogue_invalid),
    ("expected_rows_mismatch", ALL, g_expected_rows_mismatch), ("entry_kind_unknown", ALL, g_entry_kind_unknown),
    ("entry_field", ALL, g_entry_field), ("slot_unknown", ALL, g_slot_unknown),
    ("slot_duplicate", ALL, g_slot_duplicate), ("derived_counted_as_run", ALL, g_derived_counted_as_run),
    ("record_uncommitted", ALL, g_record_uncommitted), ("dl_row_missing", ALL, g_dl_row_missing),
    ("entry_file_rewritten", ALL, g_entry_file_rewritten), ("entry_file_invalid", ALL, g_entry_file_invalid),
    ("evaluator_commit_invalid", ALL, g_evaluator_commit_invalid),
    ("execution_context_split", ALL, g_execution_context_split),
    ("calibration_index_mismatch", ALL, g_calibration_index_mismatch),
    ("reference_mismatch", ALL, g_reference_mismatch), ("teacher_values_mismatch", ALL, g_teacher_values_mismatch),
    ("image_digest_not_of_record", ALL, g_image_digest_not_of_record), ("schedule_invalid", ALL, g_schedule_invalid),
    ("freeze_source_conflict", FU, g_freeze_source_conflict), ("content_class_missing", FU, g_content_class_missing),
    ("generator_pin_incomplete", FU, g_generator_pin_incomplete),
    ("generator_pin_sha256_mismatch", ALL, g_generator_pin_sha256_mismatch),
    ("generator_cells_mismatch", FU, g_generator_cells_mismatch),
    ("test_value_before_unlock", ALL, g_test_value_before_unlock),
    ("trigger_format_unknown", FU, g_trigger_format_unknown), ("trigger_inconsistent", FU, g_trigger_inconsistent),
    ("launched_attempt_without_directory", FU, g_launched_attempt_without_directory),
    ("completion_record_invalid", FU, g_completion_record_invalid),
    ("completion_source_mismatch", FU, g_completion_source_mismatch),
    ("checkpoint_state_flag", FU, g_checkpoint_state_flag), ("entry_not_complete", FU, g_entry_not_complete),
    ("complete_run_omitted", FU, g_complete_run_omitted), ("checkpoint_hash_mismatch", ALL, g_checkpoint_hash_mismatch),
    ("checkpoint_missing", FU, g_checkpoint_missing),
    ("restored_checkpoint_unrecorded", ALL, g_restored_checkpoint_unrecorded),
    *[(code, ALL, _cut_gate(code)) for code, _ in E.CUT_CHECKS],
    ("parent_rule_violation", ALL, g_parent_rule_violation),
    ("descendant_entry_missing", FU, g_descendant_entry_missing),
    ("not_run_without_parent_record", ALL, g_not_run_without_parent_record),
    ("derived_parent_incomplete", ALL, g_derived_parent_incomplete),
    ("derived_inputs_incomplete", ALL, g_derived_inputs_incomplete),
    ("derived_inputs_mismatch", ALL, g_derived_inputs_mismatch), ("ptq_record_invalid", ALL, g_ptq_record_invalid),
    ("derived_missing", FU, g_derived_missing), ("e6s42_not_hashed_at_freeze", FU, g_e6s42_not_hashed_at_freeze),
    ("derived_selection_conflict", ALL, g_derived_selection_conflict),
    ("non_finite_invalid", ALL, g_non_finite_invalid),
    ("append_invalid", ALL, g_append_invalid), ("derived_hash_mismatch", ALL, g_derived_hash_mismatch),
    ("append_after_unlock", U, g_append_after_unlock),
    ("derived_not_hashed_at_unlock", U, g_derived_not_hashed_at_unlock),
    ("scope_forbidden", ALL, g_scope_forbidden), ("scope_required_missing", FU, g_scope_required_missing),
    ("candidate_scored", ALL, g_candidate_scored), ("default_candidate_missing", FU, g_default_candidate_missing),
    ("listed_or_terminal_scored", ALL, g_listed_or_terminal_scored),
    ("run_added_after_freeze", U, g_run_added_after_freeze),
    ("cut_unentered_at_unlock", FU, g_cut_unentered_at_unlock),
    ("lane_unmet", U, g_lane_unmet), ("not_required_invalid", U, g_not_required_invalid),
    ("evaluator_files_changed", U, g_evaluator_files_changed), ("hasher_pin_mismatch", U, g_hasher_pin_mismatch),
    ("second_session", U, g_second_session), ("clock_offset_invalid", SES, g_clock_offset_invalid),
    ("session_after_deadline", U, g_session_after_deadline), ("push_evidence_invalid", SES, g_push_evidence_invalid),
]
REPORTED_KINDS = ("cut", "not_run", "not_made", "freeze_delay", "new_freeze")   # "entry later than its day" (xv)
# F/U-only gates whose finding a draft prints as MISSING (plan §2.1) instead of skipping
MISSING_IN_DRAFT =("freeze_source_conflict", "content_class_missing", "generator_pin_incomplete",
                    "trigger_format_unknown", "complete_run_omitted", "derived_missing", "e6s42_not_hashed_at_freeze",
                    "descendant_entry_missing", "scope_required_missing", "default_candidate_missing",
                    "cut_unentered_at_unlock")


def validate(x: Ctx) -> Result:
    """Run the gates in order; raise the first refusal (TmanRefused); return the passed gates, the MISSING lines of a
    draft and the REPORT lines."""
    res = Result(x.mode)
    for code, modes, fn in GATES:
        if x.mode not in modes:
            if x.mode == "draft" and code in MISSING_IN_DRAFT:
                try:
                    err = fn(x)
                except S.TmanRefused as e:
                    err = e.detail
                if err:
                    res.missing.append(f"MISSING: {code}: {err}")
            continue
        try:
            err = fn(x)
        except S.TmanRefused as e:
            if e.code != code:
                e.passed = list(res.passed)
                raise
            err = e.detail
        if err:
            refusal = S.TmanRefused(code, err)
            refusal.passed = list(res.passed)
            raise refusal
        res.passed.append(code)
    for rel in dict.fromkeys(c["path"] for c in x.cited_entries() if c["kind"] in REPORTED_KINDS):
        doc, sha = x.entry_file(rel)
        rep = E.late_entry_report(doc, E.entered_day(x.records, sha))
        if rep:
            res.reports.append(f"REPORT {rep}")
    return res


# ===================================================================================== the inventory reader
ROW_SPEC = {"file_id": "str", "slot": "slot", "stage": "str", "model_role": ("enum", S.MODEL_ROLES),
            "precision": ("enum", S.INVENTORY_PRECISIONS), "seed": "?int",
            "source": ("obj", {"kind": ("enum", ("checkpoint", "provenance")), "path": "str", "sha256": "sha256"}),
            "converted_artifact_sha256": "?sha256", "cells": ("list", "str"),
            "entry_kind": ("enum", ("scored_run", "default_candidate", "arm_teacher", "derived")),
            "role": ("list", ("enum", S.ROLES)), "training_run_id": "str"}
NULL_ENTRY = {"slot": "slot", "class": ("enum", S.SLOT_CLASSES), "stage": "str", "seed": "?int",
              "catalogue": ("obj", {"never_cut": "bool", "cut_place": "?int", "parent": "?slot"}),
              "role": ("list", ("enum", S.ROLES))}                 # and "kind": null (IF§11.2)
TELEMETRY_ITEM = ("obj", {"run_id": "str", "path": "str", "sha256": "sha256"})
INVENTORY_TOP = ("binding", "dedup", "entries", "evaluator_commit", "execution_context", "freeze",
                 "manifest_file_sha256", "manifest_sha256", "models", "schema")
BINDING_STATE = ("obj", {"state": ("enum", ("unbound", "bound")), "path": "?str", "sha256": "?sha256"})
DEDUP_SPEC = ("obj", {"path": "str", "sha256": "sha256", "removed_by_reason": ("map", "int")})
FREEZE_SPEC = ("obj", {"dl_row": "dl", "manifest_sha256": "sha256"})
BINDING_FILE_SPEC = {"format": ("enum", (S.BINDING_FORMAT,)), "references": ("map", "any"),
                     "artifacts": ("list", ("obj", S.BINDING_ARTIFACT_FIELDS))}


def inventory_errors(doc, cat=None) -> list[str]:
    """Every way `doc` breaks tday-inventory/1 (IF§8; IF§11 v2). With the catalogue `cat`: one entries item per
    catalogue slot, in its order; each model row of a slot entered with a scored kind, under that slot's file_id,
    unique by (file_id, precision); the execution context at TB1's fixed values (DL-24 #105, #107). A malformed value
    is reported, never hashed (#104)."""
    if not isinstance(doc, dict) or sorted(doc) != list(INVENTORY_TOP):
        return [f"the export's keys are not {list(INVENTORY_TOP)}"]
    errs = [] if doc["schema"] == S.INVENTORY_FORMAT else [f"schema {doc['schema']!r}"]
    errs += S.value_errors(doc["evaluator_commit"], "commit", "evaluator_commit")
    errs += S.value_errors(doc["manifest_sha256"], "sha256", "manifest_sha256")
    errs += S.value_errors(doc["manifest_file_sha256"], "sha256", "manifest_file_sha256")
    ec = doc["execution_context"]
    if not (isinstance(ec, dict) and sorted(ec) == sorted(TUPLE_KEYS)):
        errs.append("execution_context is not TB1's tuple")
    elif cat is not None and any(ec[k] != v for k, v in TUPLE_FIXED.items()):
        errs.append(f"execution_context is not at TB1's fixed values {TUPLE_FIXED}")
    if not isinstance(doc["models"], list) or not isinstance(doc["entries"], list):
        return errs + ["models and entries are lists"]
    kinds = {it.get("slot"): it.get("kind") for it in doc["entries"] if isinstance(it, dict)
             and isinstance(it.get("slot"), str)}
    seen = set()
    for i, row in enumerate(doc["models"]):
        bad = S.field_errors(row, ROW_SPEC, f"models[{i}]")
        errs += bad
        if isinstance(row, dict) and not bad:
            for key, what in (((row["slot"], row["precision"]), "slot and precision"),
                              (("file_id", row["file_id"], row["precision"]), "file_id and precision")):
                if key in seen:
                    errs.append(f"models[{i}]: a second row of {key} (rows are unique by {what})")
                seen.add(key)
            int8 = row.get("precision") in ("int8_ptq", "int8_qat")
            if int8 != (row.get("converted_artifact_sha256") is not None):
                errs.append(f"models[{i}]: converted_artifact_sha256 is for int8_ptq and int8_qat rows only")
            if row.get("precision") == "qat_fakequant_off" and row.get("cells"):
                errs.append(f"models[{i}]: a qat_fakequant_off row has no cells")
            if cat is not None and (row["slot"] not in cat.slots or row["file_id"] != cat.slots[row["slot"]]["file_id"]
                                    or kinds.get(row["slot"]) not in M.SCORED_KINDS):
                errs.append(f"models[{i}]: {row['slot']} is not a catalogue slot entered with a scored kind under "
                            f"its file_id")
    slots = []
    for i, it in enumerate(doc["entries"]):
        if not isinstance(it, dict):
            errs.append(f"entries[{i}] is not an object")
            continue
        slots.append(it.get("slot") if isinstance(it.get("slot"), str) else repr(it.get("slot")))
        if "kind" in it and it["kind"] is None:
            errs += S.field_errors({k: v for k, v in it.items() if k != "kind"}, NULL_ENTRY, f"entries[{i}]")
        else:
            body = {k: v for k, v in it.items() if k != "telemetry"}
            errs += S.entry_errors(body)
            if "telemetry" in it:
                errs += S.value_errors(it["telemetry"], ("list", TELEMETRY_ITEM), f"entries[{i}].telemetry")
    if len(set(slots)) != len(slots):
        errs.append("entries lists a slot twice")
    elif cat is not None and slots != list(cat.slots):
        errs.append("entries are not one item per catalogue slot, in its order (IF§11.2)")
    for k, spec in (("dedup", DEDUP_SPEC), ("freeze", FREEZE_SPEC)):
        if doc[k] is not None:
            errs += S.value_errors(doc[k], spec, k)
    errs += S.value_errors(doc["binding"], BINDING_STATE, "binding")
    b = doc["binding"] if isinstance(doc["binding"], dict) else {}
    if (b.get("state") == "unbound") != (b.get("path") is None and b.get("sha256") is None):
        errs.append("binding: unbound has no path or sha256, bound has both")
    if isinstance(doc["freeze"], dict) and doc["freeze"].get("manifest_sha256") != doc["manifest_sha256"]:
        errs.append("freeze.manifest_sha256 is not the export's manifest_sha256")
    return errs


def read_inventory(path, *, expect_sha256: str | None = None) -> dict:
    """IF§11.6: read, check and return the inventory export; refuse inventory_sha256_mismatch, inventory_invalid or
    binding_invalid. A bound binding is verified against binding.sha256 (its path relative to the export's folder) and
    its artifacts records are returned under binding.artifacts."""
    S.refuse_test_path(path)
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise S.TmanRefused("inventory_invalid", f"{p}: unreadable ({type(e).__name__})") from None
    if expect_sha256 is not None and S.sha256_bytes(data) != expect_sha256:
        raise S.TmanRefused("inventory_sha256_mismatch", f"{p}: sha256 {S.sha256_bytes(data)} != {expect_sha256}")
    doc = S.strict_json(data, str(p), code="inventory_invalid")
    errs = inventory_errors(doc, S.load_catalogue((M.CODE_ROOT / S.CATALOGUE_REL).read_bytes(), S.CATALOGUE_REL))
    if errs:
        raise S.TmanRefused("inventory_invalid", f"{p}: " + "; ".join(errs[:6]))
    if doc["binding"]["state"] == "bound":
        bp = p.parent / doc["binding"]["path"]
        S.refuse_test_path(bp)
        try:
            bdata = bp.read_bytes()
        except OSError as e:
            raise S.TmanRefused("binding_invalid", f"{bp}: {e}") from None
        if S.sha256_bytes(bdata) != doc["binding"]["sha256"]:
            raise S.TmanRefused("binding_invalid", f"{bp}: its sha256 is not binding.sha256")
        bdoc = S.strict_json(bdata, str(bp), code="binding_invalid")
        berrs = S.field_errors(bdoc, BINDING_FILE_SPEC, str(bp))
        if berrs:
            raise S.TmanRefused("binding_invalid", "; ".join(berrs[:4]))
        doc["binding"]["artifacts"] = bdoc["artifacts"]
    return doc


__all__ = ["Ctx", "GATES", "MODES", "Result", "inventory_errors", "read_inventory", "slot_of_launch", "validate"]
