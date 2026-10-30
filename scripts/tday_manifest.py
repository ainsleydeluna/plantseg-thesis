#!/usr/bin/env python3
"""The TEST-day manifest's command line (lane TMAN, B-1, C5; plan §0 and decision (xvii)).

Commands:
  build             a draft manifest from the committed records and a sources file (plan §1.2), validated in draft
                    mode, written as tday_manifest.json into --out; the image_hasher pin is dedup.hasher_pin's (P2R2)
  validate          the validator (src/tman/validate.py) in mode draft, freeze, unlock or session; in session mode,
                    --dedup gives the in-session hasher's tday_dedup.json (P2R6(a))
  entry KIND        an entry file (tday_entry/1) checked, written into --out, and its decision-log row printed
  arithmetic        AM-19 item 3(b)'s arithmetic for a run (scripts/preflight_arms.py's cut_arithmetic, reused); with
                    --out, the cut entry it implies when the run does not fit
  append            a derived artifact's hashed form appended to the manifest, written into --out
  isolation-record  src/data/isolation.assert_trainval_only_root on --data-root, written as tday_isolation/1
  inventory         the inventory export tday-inventory/1 (IF§8; IF§11 v2), written into --out and read back; with
                    --dedup, its dedup block names the hasher's tday_dedup.json relative to --out (P2R4, P2R5); with
                    --binding, its binding is bound to the file bind wrote, inside --out (B3R4, B3R7)
  bind              after the single TEST session (B-3b; B3R8): the binding tday_binding/1 and the statistics input
                    list tday_stats_inputs.json, from the manifest validated in session mode, the unbound export and
                    each scored TEST artifact at its layout path under --root (S = E), read score-blind
  check-freeze      AM-22 item 1(a)'s check at the end of --date (src/tman/am22.py; B-2): each condition MET with its
                    records or MISSING; it writes nothing
  freeze-body       the body to freeze for AM-22 item 1(b)'s D: the draft committed by the end of D and the
                    measurement's freeze cut (TMAN-B2-RULE-3), written as tday_manifest.json into --out

Every output is a new file created exclusively in --out, a folder outside the repository; no command writes in the
repository, and none reads TEST data. Records are read at HEAD of the repository at --root (default: this
checkout); a shallow clone is refused first (repository_shallow; P2R6(e)). Exit codes follow the select scripts
(src/training/sweep_select.py:203): 0 pass or written, 2 refused, 3 incomplete (draft), 4 error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.tman import am22 as A  # noqa: E402
from src.tman import binding as B  # noqa: E402
from src.tman import completion as C  # noqa: E402
from src.tman import entries as E  # noqa: E402
from src.tman import manifest as M  # noqa: E402
from src.tman import records as R  # noqa: E402
from src.tman import schedule as D  # noqa: E402
from src.tman import schema as S  # noqa: E402
from src.tman import validate as V  # noqa: E402

SCRIPT_REL = "scripts/tday_manifest.py"


# --------------------------------------------------------------------------------------------- helpers
def out_folder(path, root=None) -> Path:
    """--out: an existing folder outside the repository at --root and this checkout, free of "test"."""
    S.refuse_test_path(path)
    p = Path(path).resolve()
    for repo in {REPO.resolve(), Path(root or REPO).resolve()}:
        if p == repo or repo in p.parents:
            raise S.TmanRefused("output_location", f"--out {p} is inside the repository: outputs are written "
                                                   "outside it")
    if not p.is_dir():
        raise S.TmanRefused("output_location", f"--out {p} is not an existing folder")
    return p


def write_new(folder: Path, name: str, data: bytes) -> Path:
    err = S.created_name_error(name)
    if err:
        raise S.TmanRefused("test_path", f"refusing to create {name}: {err}")
    out = folder / name
    try:
        with open(out, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        raise S.TmanRefused("output_exists", f"{out} exists: outputs are created once") from None
    return out


def read_bytes(path, code: str) -> bytes:
    """A file named on the command line, refused for "test" before any call on it; unreadable: refused as `code`
    (DL-24 #90)."""
    S.refuse_test_path(path)
    try:
        return Path(path).read_bytes()
    except OSError as e:
        raise S.TmanRefused(code, f"{path}: unreadable ({type(e).__name__})") from None


def read_json(path, code: str):
    return S.strict_json(read_bytes(path, code), str(path), code=code)


def records_at(args) -> R.Records:
    """The records at HEAD of --root, a shallow clone refused before any command reads them (P2R6(e)): B-2's AM-22
    lookups catch a refusal raised inside them, so the check cannot wait for a history reader (DL-24 R2-F1)."""
    records = R.Records.at_head(args.root)
    records.full_history()
    return records


def catalogue(records: R.Records):
    return S.load_catalogue(records.read(S.CATALOGUE_REL))


def registered(args):
    return None if args.registered_cells is None else list(read_json(args.registered_cells, "generator_cells_mismatch"))


def clock(args, code: str = "am22_date_not_ended") -> float:
    """The clock, or --now standing in for it in the smokes."""
    if args.now is not None and not S.TYPES["utc"](args.now):
        raise S.TmanRefused(code, f"--now {args.now!r} is not a UTC instant (YYYY-MM-DDTHH:MM:SSZ)")
    return time.time() if args.now is None else D.parse_utc_any(args.now)


def pushed_days(args) -> dict:
    """--pushed COMMIT=DAY: a commit the orchestrator confirmed, from the remote's activity, reached it on a day later
    than its committer day (audit M2), each commit once; am22.pushed_error checks each."""
    out = {}
    for p in getattr(args, "pushed", None) or []:
        c, sep, day = p.partition("=")
        if not sep or c in out:
            raise S.TmanRefused("entry_field", f"--pushed {p!r}: COMMIT=DAY, each commit once (audit M2)")
        out[c] = day
    return out


def print_result(res: V.Result) -> int:
    for line in res.missing + res.reports:
        print(line)
    state = "MISSING" if res.missing else "PASS"
    print(f"{state}: mode {res.mode}; {len(res.passed)} gates passed")
    return res.exit_code


# ------------------------------------------------------------------------------------------------- build
def cmd_build(args) -> int:
    """The draft; with --am22-d D its schedule block is AM-22's candidate freeze for D (B2R6; audit m5: the draft of
    each candidate day is built so, for condition 5), otherwise the AM-22 freeze recorded in the decision log, if
    any."""
    records = records_at(args)
    cat = catalogue(records)
    sources = read_json(args.sources, "manifest_format")
    doc = M.build_manifest(records, cat, sources, runs_root=args.runs,
                           providers=A.providers(records, d_date=args.am22_d, lookup=True))
    data = S.json_file_bytes(doc)
    folder = out_folder(args.out, args.root)
    res = V.validate(V.Ctx(mode="draft", manifest_bytes=data, manifest_path=str(folder / "tday_manifest.json"),
                           records=records, registered_cells=registered(args),
                           am22_d=args.am22_d))                                       # a refusal writes nothing (#61)
    out = write_new(folder, "tday_manifest.json", data)
    print(f"WROTE {out} sha256 {S.sha256_bytes(data)} body_sha256 {S.canonical_sha256(doc['body'])}")
    return print_result(res)


# ---------------------------------------------------------------------------------------------- validate
def cmd_validate(args) -> int:
    """Draft and freeze read the records at HEAD of --root; unlock and session run at the session checkout S = E
    (--root) and read the records from --records, a folder written from --records-commit (PL-28; TB16). --dedup, the
    in-session hasher's tday_dedup.json, goes with the session part's values.image_hasher, in session mode only
    (P2R6(a))."""
    if args.dedup is not None and args.mode != "session":
        raise S.TmanRefused("hasher_values_mismatch", f"--dedup is read in session mode only, not in {args.mode} mode "
                                                      "(P2R6(a))")
    data = read_bytes(args.manifest, "manifest_unreadable")
    if args.am22_d is not None and args.mode in ("unlock", "session"):
        raise S.TmanRefused("schedule_invalid", f"--am22-d is read in draft and freeze modes: in {args.mode} mode the "
                                                "AM-22 freeze is the freeze record's (--freeze-record; B2R6; DL-24 "
                                                "R2-3)")
    if (args.records is None) != (args.records_commit is None):
        raise S.TmanRefused("record_uncommitted", "--records and --records-commit go together (PL-28)")
    if args.mode in ("unlock", "session") and args.records is None:
        raise S.TmanRefused("record_uncommitted", f"{args.mode} mode reads the records from --records at "
                                                  "--records-commit; --root is the session checkout S = E (TB16)")
    if args.records is not None:
        S.refuse_test_path(args.records)
        if not S.COMMIT_RE.fullmatch(args.records_commit):
            raise S.TmanRefused("record_uncommitted", f"--records-commit {args.records_commit!r} is not 40 hex")
        records = R.Records.at_records_commit(args.records, args.records_commit, root=args.root)
        records.full_history()                                                  # as records_at (DL-24 R2-F1)
    else:
        records = records_at(args)
    rel = records.src.rel(args.manifest)
    res = V.validate(V.Ctx(mode=args.mode, manifest_bytes=data, manifest_path=rel or str(args.manifest),
                           records=records, runs_root=args.runs, dedup=args.dedup, freeze_record=args.freeze_record,
                           unlock_entry=args.unlock_entry, registered_cells=registered(args), now=time.time(),
                           am22_d=args.am22_d))
    return print_result(res)


# ------------------------------------------------------------------------------------------------- entry
def manifest_body(args, code: str) -> dict:
    """--manifest's body as a printer reads it: its entries (each naming its slot), trigger and schedule blocks;
    `code` refuses a missing --manifest."""
    if not args.manifest:
        raise S.TmanRefused(code, f"entry {args.kind} reads the manifest's entries (--manifest)")
    doc = read_json(args.manifest, "manifest_unreadable")
    body = doc.get("body") if isinstance(doc, dict) else None
    if not (isinstance(body, dict) and isinstance(body.get("entries"), list)
            and all(isinstance(e, dict) and isinstance(e.get("slot"), str) for e in body["entries"])):
        raise S.TmanRefused("manifest_format", f"{args.manifest}: no body with its entries")
    return body


def check_cut_entry(args, records, cat, freeze, doc):
    """The cut checks on the would-be record set; a cutoff cut (AM-22 item 2) reads the pilot's winner record from
    --manifest's E5:s42:sel inputs when a gate of its slots waits for it, and prints B2R8's "no card" REPORT."""
    pilot = None
    if args.manifest:
        e = next((x for x in manifest_body(args, "cut_date_mismatch")["entries"] if x["slot"] == E.PILOT_SELECTION),
                 None)
        pilot = ((e or {}).get("inputs") or {}).get("clip_selection")
    E.check_cut(doc, printer_cut_context(records, cat, freeze, doc, args.runs,
                                         pilot=pilot if isinstance(pilot, dict) else None))
    rep = E.no_card_report(doc, E.evidence_reader(records))
    if rep:
        print(f"REPORT {rep}")


def check_not_run(args, records, cat, freeze, doc):
    """Its slots (slot_unknown; DL-24 #68), then its parent's manifest entry (not_run_without_parent_record)."""
    E.check_slots(doc, cat, kinds_of_class=("run", "derived"))
    body = manifest_body(args, "not_run_without_parent_record")
    parent = next((e for e in body["entries"] if e["slot"] == doc["parent"]), None)
    errs = E.not_run_entry_errors(doc, cat, parent)
    if errs:
        raise S.TmanRefused("not_run_without_parent_record", "; ".join(errs))


def check_freeze_source(args, records, cat, freeze, doc):
    """AM-19 item 1(d) on the never-cut runs' records, read as the validator reads them (validate.never_cut_instants:
    the manifest's entries, the launch log and the run directories under --runs): the delay lists exactly the runs
    late at the original instant; the new freeze is the first later date at whose end every one is complete, citing
    the committed delay entry (freeze_source_conflict)."""
    from src.training import sweep_select as ss
    if args.runs is None:
        raise S.TmanRefused("launched_attempt_without_directory", "AM-19 item 1(d)'s entries read the never-cut runs' "
                                                                  "directories (--runs)")
    body = manifest_body(args, "freeze_source_conflict")
    by_slot = {e["slot"]: e for e in body["entries"]}
    triggered = isinstance(body.get("trigger"), dict) and body["trigger"].get("triggered") is True
    log = records.launch_log(code="completion_record_invalid") if records.has(ss.LAUNCH_LOG_REL) else None
    inst = V.never_cut_instants(cat, by_slot, triggered, lambda e: C.run_state(cat.slots[e["slot"]],
                                                                             Path(args.runs) / e["run_id"]),
                                C.RunIndex(cat, log, args.runs))
    if doc["kind"] == "freeze_delay":
        errs = E.freeze_delay_errors(doc, freeze, V.late_never_cut(inst, freeze.original_utc))
    else:
        _, dsha = E.read_entry_file(records, doc["delay_entry"]["path"], "freeze_delay")
        errs = E.new_freeze_errors(doc, {"path": doc["delay_entry"]["path"], "sha256": dsha}, list(inst.values()),
                                   freeze.original_date)
    if errs:
        raise S.TmanRefused("freeze_source_conflict", "; ".join(errs) + " (AM-19 item 1(d))")


def check_not_required(args, records, cat, freeze, doc):
    """The cut entries it cites are committed, recorded entry files of kind cut at the cited sha256, and cut every
    slot it names (not_required_invalid; AM-19 item 3(d))."""
    covered = set()
    for ref in doc["cut_entries"]:
        cut, sha = E.read_entry_file(records, ref["path"], "cut")
        if sha != ref["sha256"]:
            raise S.TmanRefused("not_required_invalid", f"{ref['path']}: the committed cut entry is not the cited one")
        covered |= set(cut["slots"])
    if set(doc["slots"]) - covered:
        raise S.TmanRefused("not_required_invalid", f"{sorted(set(doc['slots']) - covered)} have no cut entry")


def check_unlock(args, records, cat, freeze, doc):
    """The would-be unlock record set (AM-17 item 9; AM-19 item 4(c)): the committed manifest, its file, body and
    appends as cited, and the freeze record of that body (run_added_after_freeze); the committed lane register
    (lane_unmet); no unlock entry in the decision log yet (second_session)."""
    m = doc["manifest"]
    data = records.read(S.MANIFEST_REL)
    mdoc = S.strict_json(data, S.MANIFEST_REL, code="manifest_unreadable")
    if m["path"] != S.MANIFEST_REL or not isinstance(mdoc, dict) or S.sha256_bytes(data) != m["sha256"] \
            or S.canonical_sha256(mdoc.get("body")) != m["body_sha256"] \
            or S.canonical_sha256(mdoc.get("appends")) != m["appends_sha256"]:
        raise S.TmanRefused("run_added_after_freeze", f"the unlock entry records the committed {S.MANIFEST_REL}: its "
                                                      "file, body and appends")
    fr, fsha = E.read_entry_file(records, doc["freeze_record"]["path"], "freeze")
    if fsha != doc["freeze_record"]["sha256"] or fr["body_sha256"] != m["body_sha256"]:
        raise S.TmanRefused("run_added_after_freeze", "the unlock entry cites the freeze record of this body")
    if fr.get("source") == D.AM22_SOURCE_1B:                    # audit M4: the freeze record's push, before its end
        from src.training import sweep_select as ss
        added = ss.adding_commit(records.src, doc["freeze_record"]["path"])
        if added is None or A.row_commit(records, fsha) != added:
            raise S.TmanRefused("freeze_entry_late", "the AM-22 freeze record's decision-log row is entered by the "
                                                     "commit that adds the record: item 1(b) times the commit of the "
                                                     "freeze's decision-log entry (PA :2345-2346; DL-24 R1-F8)")
        if not doc["push_evidence"]:
            raise S.TmanRefused("freeze_entry_late", "the unlock entry of an AM-22 item 1(b) freeze cites the freeze "
                                                     "record's push evidence (--push-evidence; audit M4)")
        try:
            err = V.push_evidence_error(records, doc["push_evidence"], doc["freeze_record"], fr["date"],
                                        D.cutoff_instant(fr["freeze_date"]), key="entry")
        except S.TmanRefused as e:
            err = f"[{e.code}] {e.detail}"
        if err:
            raise S.TmanRefused("freeze_entry_late", f"the AM-22 freeze record's push evidence: {err}")
    if doc["lanes"] != {"path": S.LANES_REL, "sha256": S.sha256_bytes(records.read(S.LANES_REL))}:
        raise S.TmanRefused("lane_unmet", f"the unlock entry cites the committed {S.LANES_REL}")
    rows = [dl for dl, (_, line) in records.decision_log().rows.items() if "TMAN unlock entry" in line]
    if rows:
        raise S.TmanRefused("second_session", f"an unlock entry is in the decision log already: {rows} (AM-17 item 9)")


def check_condition(args, records, cat, freeze, doc):
    if doc["slots"] != [f"E6-{doc['arm']}:s42:15ep"]:
        raise S.TmanRefused("entry_field", f"a condition entry of arm {doc['arm']} names E6-{doc['arm']}:s42:15ep "
                                           "alone (AM-23 item 3(b))")


def check_not_made(args, records, cat, freeze, doc):
    E.check_slots(doc, cat, kinds_of_class=("derived",))
    bad = [k for k in doc["slots"] if not M.code_gated(k, cat)]
    if bad:
        raise S.TmanRefused("derived_missing", f"{bad}: not made is for a slot whose code is gated (AM-23 item 4)")


def check_freeze_record(args, records, cat, freeze, doc):
    """The freeze record records --manifest's body; its AM-22 fields (am22.freeze_entry_error; audit m8): a freeze of
    item 1(b) only through --d-date, its conditions recomputed (audit m9); one of item 1(f) at the resolver's freeze
    for the body -- AM-19's as committed, as the freeze delay or new freeze its schedule block cites moves it, that
    block the resolver's (audit m8; DL-24 R1) -- once the end of 2026-11-02 has passed with no D, read from the
    records at each date's end and the run directories (--runs; PA :2353; DL-24 R1-F2, R2-2)."""
    body = manifest_body(args, "run_added_after_freeze")
    if doc["body_sha256"] != S.canonical_sha256(body):
        raise S.TmanRefused("run_added_after_freeze", "the freeze record records --manifest's body")
    if doc["source"] == D.AM22_SOURCE_1B and args.d_date is None:
        raise S.TmanRefused("freeze_entry_invalid", "a freeze of AM-22 item 1(b) is printed with --d-date D: its "
                                                    "conditions are recomputed, never given (audit m9)")
    resolver = freeze
    if doc["source"] == D.AM22_SOURCE_1F:
        sch = body.get("schedule") if isinstance(body.get("schedule"), dict) else {}
        delays, news = V.freeze_docs(sch, lambda rel: (lambda r: (r["doc"], r["sha256"]))(M.entry_ref(records, rel)))
        resolver = D.freeze_instant(D.schedule_block(records.src), delays=delays, news=news,
                                    providers=A.providers(records, freeze_doc=doc))          # none for item 1(f)
        if sch != resolver.as_block():
            raise S.TmanRefused("freeze_entry_invalid", "a freeze of item 1(f) records a body whose schedule block is "
                                                        "the resolver's: AM-19's freeze as committed, as its freeze "
                                                        "delay or new freeze moves it (audit m8)")
    err = A.freeze_entry_error(doc, resolver.moved_date or resolver.original_date)
    if err:
        raise S.TmanRefused("freeze_entry_invalid", err)
    if doc["source"] == D.AM22_SOURCE_1F:
        if args.runs is None:
            raise S.TmanRefused("launched_attempt_without_directory", "a freeze of item 1(f) reads item 1(a)'s "
                                                                      f"check to {A.LAST_D} from the run directories "
                                                                      "(--runs)")
        if not D.cutoff_instant(A.LAST_D) < clock(args):
            raise S.TmanRefused("am22_date_not_ended", f"a freeze of item 1(f) is printed once the end of {A.LAST_D} "
                                                       "has passed with no D (PA :2353)")
        why = A.fallback_error(records, cat, Path(args.runs), reader=R.trigger_reader(records),
                               registered_cells=registered(args))
        if why:
            raise S.TmanRefused("am22_d_not_first", why)


def check_clock_offset(args, records, cat, freeze, doc):
    if abs(doc["offset_seconds"]) > S.CLOCK_OFFSET_LIMIT_S:
        raise S.TmanRefused("clock_offset_invalid", f"offset {doc['offset_seconds']} s: within 60 s (AM-19a "
                                                    "reading 21)")


ENTRY_CHECKS = {"cut": check_cut_entry, "not_run": check_not_run, "freeze_delay": check_freeze_source,
                "new_freeze": check_freeze_source, "not_required": check_not_required, "unlock": check_unlock,
                "condition": check_condition, "not_made": check_not_made, "freeze": check_freeze_record,
                "clock_offset": check_clock_offset}


def cmd_entry(args) -> int:
    """A printer builds its entry, runs that kind's checks on the would-be record set, then writes the file and
    prints its decision-log row (plan §0; DL-24 #8, #9, #74): the checks are chosen by the entry's own kind."""
    records = records_at(args)
    cat = catalogue(records)
    fields = read_json(args.fields, "entry_field")
    if args.rule is not None and (args.kind != "cut" or not isinstance(fields, dict)
                                  or fields.setdefault("rule", args.rule) != args.rule):
        raise S.TmanRefused("entry_field", f"--rule {args.rule} is a cut's rule: entry cut, its fields' rule the same")
    if args.pushed and (args.kind != "freeze" or args.d_date is None):
        raise S.TmanRefused("entry_field", "--pushed is the freeze entry's of AM-22 item 1(b): entry freeze --d-date D "
                                           "(audit M2)")
    reports = []
    if args.d_date is not None and args.kind == "cut":
        measured = [k for k in args.slot or [] if (cat.slots.get(k) or {}).get("class") == "measurement"]
        if not (isinstance(fields, dict) and fields.get("rule") == "freeze" and measured and measured == args.slot):
            raise S.TmanRefused("entry_field", "--d-date D prints a cut only for the measurement's freeze cut at "
                                               "AM-22's freeze for D: rule freeze, measurement slots (item 1(g); "
                                               "TMAN-B2-RULE-3; DL-24 R1-F1)")
    elif args.d_date is not None:
        fields, chk = am22_freeze_fields(args, records, cat, fields)
        reports = A.boundary_reports(chk)
    if args.push_evidence is not None and args.kind != "unlock":
        raise S.TmanRefused("entry_field", "--push-evidence is the unlock entry's (audit M4)")
    pe = read_json(args.push_evidence, "freeze_entry_late") if args.push_evidence else None
    doc = E.make_entry(args.kind, args.slot or [], args.date, fields, amendment=args.amendment, script=SCRIPT_REL,
                       commit=records.src.head() or None, push_evidence=pe)
    freeze = D.freeze_instant(D.schedule_block(records.src),
                              providers=A.providers(records, d_date=args.d_date, lookup=True))
    ENTRY_CHECKS[doc["kind"]](args, records, cat, freeze, doc)
    path, rel, sha = E.write_entry(doc, out_folder(args.out, args.root), nn=args.nn, records=records)
    for line in reports:
        print(line)
    print(f"WROTE {path} sha256 {sha}")
    print("DECISION-LOG ROW (the orchestrator assigns the ID; commit the file at its path with it):")
    print(E.dl_row(doc, rel, sha, args.summary))
    return S.EXIT_PASS


def am22_freeze_fields(args, records, cat, fields) -> tuple:
    """`entry freeze --d-date D` (B2R5; audit m9 as TMAN-B2-RULE-3 amends it): the freeze entry of AM-22 item 1(b),
    its fields recomputed, never given (--fields is {}), from the records at the end of D and every earlier date, each
    commit's confirmed push day (--pushed; audit M2) and the run directories (--runs); --manifest is the body to
    freeze, validated in freeze mode; --now stands in for the clock in the smokes. Returns (the fields, D's check)."""
    if args.kind != "freeze" or fields != {}:
        raise S.TmanRefused("entry_field", "--d-date D prints the freeze entry of AM-22 item 1(b): entry freeze with "
                                           "the fields {}, which it recomputes (audit m9)")
    if args.runs is None:
        raise S.TmanRefused("launched_attempt_without_directory", "the freeze entry's check reads the run "
                                                                  "directories (--runs)")
    now = clock(args)
    body = manifest_body(args, "run_added_after_freeze")
    data = read_bytes(args.manifest, "manifest_unreadable")
    return A.freeze_fields(records, cat, Path(args.runs), args.d_date, body, now=now, reader=R.trigger_reader(records),
                           registered_cells=registered(args), pushed=pushed_days(args),
                           manifest=(data, records.src.rel(args.manifest) or str(args.manifest)))


def printer_cut_context(records: R.Records, cat, freeze, doc: dict, runs, pilot: dict | None = None) -> E.CutContext:
    """The cut printer's context, the validator's (validate.cut_context): the registry, the decision records, every
    run slot's launched attempts and status from the launch log and the run directories under --runs
    (completion.RunIndex; never from the entry), a named rule's record and the records committed by a day (E3-X's
    run of record is read from the manifest by the validator); AM-22 item 2's dates of record (B2R1), the slots'
    cutoff-moving records (`pilot`: the pilot's winner record {path, sha256}), the attempts' last rows and the day a
    record entered the decision log. A cut is printed from the run directories: --runs is required."""
    from src.training import sweep_select as ss
    if runs is None:
        raise S.TmanRefused("launched_attempt_without_directory", "a cut is printed from the run directories: each "
                                                                  "cut slot's launched attempts and status (--runs)")
    log = records.launch_log(code="completion_record_invalid") if records.has(ss.LAUNCH_LOG_REL) else None
    recs = {k: records.json(rel, R.DECISION_RECORD, code="decision_date_cut_mismatch")[0]
            for k, rel in ss.DECISION_RECORD_REL.items() if records.has(rel)}
    idx = C.RunIndex(cat, log, runs)
    named = set(log.launched if log else ()) | set(doc["run_ids"])
    for r in sorted(named):
        S.refuse_test_path(r, runs)
    missing = sorted(r for r in named if not (Path(runs) / r).is_dir())
    if missing:
        raise S.TmanRefused("launched_attempt_without_directory", f"launched attempts without a directory under "
                                                                  f"--runs: {missing} (AM-19a reading 20)")

    return E.CutContext(cat, freeze, registry=E.arm_registry(records),
                        decision_records=recs, attempts=idx.attempts, run_slot=idx.slot_of.get,
                        stopped=idx.stopped_ids(), status=lambda slot: idx.status(slot, freeze.original_utc),
                        rule_record=E.rule_record_reader(records), known_by=E.known_by_reader(records),
                        am22=lambda: D.am22_dates_of_record(records.src),
                        moving=E.moving_reader(cat, records, idx, pilot), last_row=E.last_row_reader(cat, idx),
                        logged=E.logged_reader(records), reads_runs=True)


# -------------------------------------------------------------------------------------------- arithmetic
def cmd_arithmetic(args) -> int:
    """AM-19 item 3(b)'s arithmetic for --slot, its inputs taken from the slot (DL-24 #11, #111): its iterations, its
    horizon (a QAT run's, its reference run's steps); its telemetry references, the most recently finished run of its
    stage under --runs, read as the slot's runs are read; its smoke references, pod smokes kept under --runs."""
    records = records_at(args)
    cat = catalogue(records)
    if args.slot not in cat.slots or cat.slots[args.slot]["class"] != "run":
        raise S.TmanRefused("slot_unknown", f"{args.slot} is not a catalogue run slot")
    slot = cat.slots[args.slot]
    now = time.time() if args.now is None else D.parse_utc_any(args.now)
    if (args.reference_telemetry or args.reference_smoke) and args.runs is None:
        raise S.TmanRefused("cut_arithmetic_invalid", "the references are read under the run directories (--runs)")
    figures = []
    if args.reference_telemetry:
        from src.training import sweep_select as ss
        log = records.launch_log(code="completion_record_invalid") if records.has(ss.LAUNCH_LOG_REL) else None
        latest = V.latest_finished(cat, C.RunIndex(cat, log, args.runs), slot["stage"], now)
        for rid in args.reference_telemetry:
            if rid not in latest:
                raise S.TmanRefused("cut_arithmetic_invalid", f"{rid} is not the most recently finished run of "
                                                              f"{slot['stage']} ({sorted(latest)}; AM-19 item 3(b))")
            figures.append(E.reference_telemetry(Path(args.runs), rid, slot))
    if args.reference_smoke:
        figures.append(E.reference_smoke(Path(args.runs), args.reference_smoke, args.smoke_key))
    steps = next((f["detail"].get("steps") for f in figures if f["source"] == "telemetry"), None)
    iterations = E.slot_iterations(slot, steps)
    if iterations is None or args.iterations not in (None, iterations):
        raise S.TmanRefused("cut_arithmetic_invalid", f"{args.slot}: its iterations are its horizon, {iterations} "
                                                      f"(a QAT run's from its telemetry reference), not "
                                                      f"{args.iterations}")
    sch = D.schedule_block(records.src)
    a = E.cut_arithmetic(iterations, figures, sch["in_force"], now)
    print(a["text"])
    if a["cut"] and args.out:
        doc = E.make_entry("cut", [args.slot], D.manila_day(now), {
            "rule": "arithmetic", "reason": "it does not fit before the original freeze date", "launched": False,
            "run_ids": [], "since": None, "arithmetic": a, "telemetry": [], "comparator": None, "results_known": [],
            "cut_order_position": None, "rule_record": None}, amendment="AM-19 item 3(b)", script=SCRIPT_REL,
            commit=records.src.head() or None)
        E.check_cut(doc, printer_cut_context(records, cat, D.freeze_instant(sch, providers=A.providers(
            records, lookup=True)), doc, args.runs))
        path, rel, sha = E.write_entry(doc, out_folder(args.out, args.root), records=records)
        print(f"WROTE {path} sha256 {sha}")
        print(E.dl_row(doc, rel, sha, "cut before its launch by AM-19 item 3(b)'s arithmetic."))
    return S.EXIT_PASS


# ------------------------------------------------------------------------------------------------ append
def cmd_append(args) -> int:
    """The validator's append checks on the one new append: append_invalid (the appends' history, then the append
    structurally), ptq_record_invalid, append_invalid with the records it cites, and under --runs
    derived_hash_mismatch."""
    data = read_bytes(args.manifest, "manifest_unreadable")
    doc = S.strict_json(data, args.manifest, code="manifest_unreadable")
    if not (isinstance(doc, dict) and isinstance(doc.get("body"), dict) and isinstance(doc["body"].get("entries"), list)
            and all(isinstance(e, dict) and isinstance(e.get("slot"), str) for e in doc["body"]["entries"])
            and isinstance(doc.get("appends"), list)):
        raise S.TmanRefused("manifest_format", f"{args.manifest}: not a manifest with a body's entries and appends")
    if any(isinstance(a, dict) and a.get("slot") == args.slot for a in doc["appends"]):
        raise S.TmanRefused("append_invalid", f"append {args.slot}: a second append of its slot (DL-24 #63)")
    records = records_at(args)
    errs = M.appends_history_errors(records, doc["appends"])            # DL-24 #54: the committed appends grow
    if errs:
        raise S.TmanRefused("append_invalid", "; ".join(errs))
    cat = catalogue(records)
    freeze = D.freeze_instant(D.schedule_block(records.src), providers=A.providers(records, lookup=True))
    rec = None
    if args.ptq_record:
        if args.runs is None:
            raise S.TmanRefused("ptq_record_invalid", "--ptq-record is a path under --runs")
        rec = {"path": args.ptq_record, "sha256": R.file_sha256(Path(args.runs) / args.ptq_record,
                                                                code="ptq_record_invalid")}   # it checks the path
    append = {"slot": args.slot, "hashed": read_json(args.hashed, "append_invalid"), "ptq_record": rec}
    by_slot = {e["slot"]: e for e in doc["body"]["entries"]}
    frozen = by_slot.get(args.slot)
    errs = M.append_errors(append, frozen, freeze.original_utc, structural=True)
    if errs:
        raise S.TmanRefused("append_invalid", "; ".join(errs))
    checksum = ((doc["body"].get("calibration_index") or {}).get("of_record") or {}).get("checksum_sha256")
    perrs, ok = M.append_ptq_record(append, frozen, cat, by_slot, checksum, runs_root=args.runs)
    if perrs:
        raise S.TmanRefused("ptq_record_invalid", "; ".join(perrs))
    errs, iso, prov = M.append_reads(append, frozen, records, args.runs)
    errs = errs or M.append_errors(append, frozen, freeze.original_utc, isolation=iso, provenance=prov,
                                   ptq_record_ok=ok)
    if errs:
        raise S.TmanRefused("append_invalid", "; ".join(errs))
    if args.runs is not None:
        errs = M.derived_hash_errors(M.appended_entry(append, frozen), by_slot.get(frozen["parent"]) or {}, cat,
                                     doc["body"].get("calibration_index"), args.runs)
        if errs:
            raise S.TmanRefused("derived_hash_mismatch", "; ".join(errs))
    new = dict(doc, appends=doc["appends"] + [append])
    out = write_new(out_folder(args.out, args.root), "tday_manifest.json", S.json_file_bytes(new))
    print(f"WROTE {out} sha256 {S.sha256_bytes(out.read_bytes())} ({len(new['appends'])} appends)")
    return S.EXIT_PASS


# -------------------------------------------------------------------------------------- isolation record
def cmd_isolation(args) -> int:
    now = time.time()
    rec = M.isolation_record(args.data_root, utc=C.utc_text(int(now)),
                             image_digest=(os.environ.get("PLANTSEG_IMAGE_DIGEST", "").strip() or None))
    data = S.json_file_bytes(rec)
    name = f"tday_isolation_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(now))}.json"
    out = write_new(out_folder(args.out, args.root), name, data)
    print(f"WROTE {out} sha256 {S.sha256_bytes(data)}")
    return S.EXIT_PASS


# --------------------------------------------------------------------------------------------- inventory
def cmd_inventory(args) -> int:
    """The export, written into --out and read back; --dedup names the hasher's tday_dedup.json, which the export's
    dedup block names relative to --out (P2R4, P2R5); --binding names the binding bind wrote, which the export's
    binding names relative to --out, bound, after binding.py's export check (B3R7)."""
    data = read_bytes(args.manifest, "manifest_unreadable")
    doc = S.strict_json(data, args.manifest, code="manifest_unreadable")
    records = records_at(args)
    cat = catalogue(records)
    folder = out_folder(args.out, args.root)
    inv = M.build_inventory(doc, data, records, cat, runs_root=args.runs, registered=registered(args),
                            freeze_record=args.freeze_record,
                            binding=None if args.binding is None else (args.binding, folder),
                            dedup=None if args.dedup is None else (args.dedup, folder))
    errs = V.inventory_errors(S.strict_json(S.json_file_bytes(inv), "the inventory", code="inventory_invalid"), cat)
    if errs:                                                      # a refusal writes nothing (DL-24 #61)
        raise S.TmanRefused("inventory_invalid", "; ".join(errs[:6]))
    out = write_new(folder, "tday_inventory.json", S.json_file_bytes(inv))
    V.read_inventory(out)
    print(f"WROTE {out} sha256 {S.sha256_bytes(out.read_bytes())} ({len(inv['models'])} model rows, "
          f"{len(inv['entries'])} entries)")
    return S.EXIT_PASS


# -------------------------------------------------------------------------------------------------- bind
BIND_NAMES = ("tday_binding.json", "tday_stats_inputs.json")
# What bind reads (B3R8; decision (vi)): of summary.json run.checkpoint_sha256, run.repo_commit,
# dataset.split_manifest_sha256 and the TB1 tuple's fields an artifact records (run.eval_runtime's,
# records.EVAL_RUNTIME); of per_image.jsonl each row's two ids. No other key enters an object.
SUMMARY_KEEP = frozenset(("run", "dataset", "checkpoint_sha256", "repo_commit", "split_manifest_sha256",
                          "eval_runtime", *R.EVAL_RUNTIME))
ROW_KEEP = frozenset(("image_id", "clean_image_id"))


def blind(keep):
    """The JSON object hook of bind's reads: every object is built (records.JSON_OBJECT, the smokes' sentinel) from
    the pairs of the allowlisted keys `keep` alone, so no score key enters an object (B3R8): none of them is a score
    key under the schema smoke's score-key rule (smoke_tman_binding)."""
    return lambda pairs: R.JSON_OBJECT([(k, v) for k, v in pairs if k in keep])


def read_artifact(root, key) -> dict | None:
    """bind's reads of one required artifact, at its layout path under the session checkout `root` (B3R8, B3R9):
    None when its folder is absent (the full check names it); else its four files' sha256 and, through `blind`,
    summary.json's three values and recorded TB1 fields and per_image.jsonl's two ids per row. A file absent or
    unreadable, not strict JSON, or without a value bind reads is binding_artifact_invalid."""
    slot, precision, scope, cell = key
    rel = B.artifact_rel(S.file_id_of(slot), precision, scope, cell)
    S.refuse_test_path(rel, root)
    folder = Path(root) / rel
    if not folder.is_dir():
        return None
    data = {}
    for name in B.file_names():
        try:
            data[name] = (folder / name).read_bytes()
        except OSError as e:
            raise S.TmanRefused("binding_artifact_invalid", f"{rel}/{name}: unreadable ({type(e).__name__})") from None
    s = S.strict_json(data["summary.json"], f"{rel}/summary.json", code="binding_artifact_invalid",
                      object_type=blind(SUMMARY_KEEP))
    run = s.get("run") if hasattr(s, "keys") else None
    ds = s.get("dataset") if hasattr(s, "keys") else None
    rt = run.get("eval_runtime") if hasattr(run, "keys") else None
    if not (hasattr(rt, "keys") and hasattr(ds, "keys") and S.TYPES["sha256"](run.get("checkpoint_sha256"))
            and S.TYPES["commit"](run.get("repo_commit")) and S.TYPES["sha256"](ds.get("split_manifest_sha256"))):
        raise S.TmanRefused("binding_artifact_invalid", f"{rel}/summary.json: not the run.checkpoint_sha256, "
                                                        "run.repo_commit, run.eval_runtime and "
                                                        "dataset.split_manifest_sha256 bind reads")
    ids = []
    for n, line in enumerate(data["per_image.jsonl"].splitlines(), 1):
        row = S.strict_json(line, f"{rel}/per_image.jsonl:{n}", code="binding_artifact_invalid",
                            object_type=blind(ROW_KEEP))
        if not (hasattr(row, "keys") and isinstance(row.get("image_id"), str)
                and isinstance(row.get("clean_image_id"), str)):
            raise S.TmanRefused("binding_artifact_invalid", f"{rel}/per_image.jsonl:{n}: no image_id and "
                                                            "clean_image_id")
        ids.append((row["image_id"], row["clean_image_id"]))
    return {"files_sha256": {n: S.sha256_bytes(d) for n, d in data.items()},
            "checkpoint_sha256": run["checkpoint_sha256"], "repo_commit": run["repo_commit"],
            "split_manifest_sha256": ds["split_manifest_sha256"], "runtime": R.project(rt, R.EVAL_RUNTIME),
            "ids": ids}


def cmd_bind(args) -> int:
    """B3R8: after the single TEST session, the binding tday_binding/1 and the statistics input list. The manifest is
    validated in session mode (the records from --records at --records-commit, S = E at --root, --dedup, --runs; TB9's
    pilot-winner check among its gates); the unbound export is read (its sha256 printed, for the runbook to compare);
    each artifact its rows require is read at its layout path under --root (read_artifact); the full check runs on
    the binding built (binding.full_check); tday_binding.json (canonical JSON) and tday_stats_inputs.json (the 37 in
    load_input_list's format) are then created, both or neither, in --out, a folder outside the repository. Nothing
    is written inside the repository, and the manifest is never edited."""
    folder = out_folder(args.out, args.root)
    for name in BIND_NAMES:                                      # both outputs or neither, before any read
        if os.path.lexists(folder / name):
            raise S.TmanRefused("output_exists", f"{folder / name} exists: outputs are created once")
    S.refuse_test_path(args.records)
    if not S.COMMIT_RE.fullmatch(args.records_commit):
        raise S.TmanRefused("record_uncommitted", f"--records-commit {args.records_commit!r} is not 40 hex")
    data = read_bytes(args.manifest, "manifest_unreadable")
    records = R.Records.at_records_commit(args.records, args.records_commit, root=args.root)
    records.full_history()                                       # as records_at (DL-24 R2-F1)
    x = V.Ctx(mode="session", manifest_bytes=data, manifest_path=records.src.rel(args.manifest) or str(args.manifest),
              records=records, dedup=args.dedup, runs_root=args.runs, freeze_record=args.freeze_record,
              unlock_entry=args.unlock_entry, registered_cells=registered(args), now=clock(args))
    V.validate(x)
    export = V.read_inventory(args.inventory)
    print(f"READ {args.inventory} sha256 {S.sha256_bytes(read_bytes(args.inventory, 'inventory_invalid'))}")
    body, reg = x.body, registered(args)
    cells = {scope: M.cells_for({scope}, reg) for scope in B.CELL_SCOPES}
    identities = B.stats_identities(cells["corrupted"])
    scopes = {m["slot"]: sorted(V.scope_rule_of(x, m["slot"])[0]) for m in export["models"]}
    family = {s["evaluator_stage"]: k for k, s in x.cat.slots.items() if "family" in s["roles"] and s["seed"] == 42}
    reads = {}
    for key in sorted(B.required_keys(export, scopes, cells), key=repr):
        got = read_artifact(args.root, key)
        if got is not None:
            reads[key] = got
    _, fsha = E.read_entry_file(records, args.freeze_record, "freeze")
    manifest = {"path": S.MANIFEST_REL, "sha256": S.sha256_bytes(data), "body_sha256": S.canonical_sha256(body),
                "freeze_dl": records.decision_log().recorded_rows_with(fsha)[0],
                "evaluator_commit": body["evaluator"]["commit"], "execution_context": body["execution_context"],
                "mde_entry": {k: body["references"]["mde_entry"][k] for k in ("path", "sha256")}}
    doc = {"format": S.BINDING_FORMAT,
           "references": {"manifest": {k: manifest[k] for k in B.MANIFEST_REFERENCE},
                          "evaluator": {"commit": manifest["evaluator_commit"]},
                          "execution_context": manifest["execution_context"],
                          "session": {"dedup": {"sha256": S.sha256_bytes(read_bytes(args.dedup, "dedup_invalid"))}},
                          "mde_entry": manifest["mde_entry"], "written_utc": C.utc_text(int(clock(args)))},
           "artifacts": [B.record_of(k, reads[k]) for k in sorted(reads, key=repr)],
           "inputs": [{"stage": st, "condition": list(cond), "key": [family.get(st), B.STATS_PRECISION[st]] + (
               ["clean", None] if cond == B.CLEAN else ["corrupted", B.cell_of(cond)])} for st, cond in identities]}
    values = x.doc["session"]["values"].get(V.HASHER_VALUES_KEY) or {}
    B.full_check(doc, {"export": export, "manifest": manifest, "scopes": scopes, "cells": cells, "reads": reads,
                       "runtime_keys": tuple(R.EVAL_RUNTIME), "slots_of_record": family,
                       "dedup_list_sha256": values.get("dedup_list_sha256"), "identities": identities,
                       "precision": B.STATS_PRECISION})
    for name, blob in ((BIND_NAMES[0], S.canonical_json_bytes(doc)),
                       (BIND_NAMES[1], S.json_file_bytes(B.stats_inputs(doc)))):
        out = write_new(folder, name, blob)
        print(f"WROTE {out} sha256 {S.sha256_bytes(blob)}")
    return S.EXIT_PASS


# ------------------------------------------------------------------------------------------ check-freeze
def cmd_check_freeze(args) -> int:
    """AM-22 item 1(a)'s check (B2R4; src/tman/am22.py): at the end of --date X, from the records committed by then
    (C_X, audit M3) and the run directories under --runs, each condition MET with its records or MISSING with what is
    missing, and REPORT lines; then D, the first date from 2026-10-17 to X at whose end all five hold (audit m4). It
    writes nothing and decides nothing (the freeze entry's printer recomputes it). Exit 0 with `D: <date>`, 3 when no
    date qualifies, 2 refused (X whose end has not passed, audit m6; X later than 2026-11-02, item 1(f)), 4 error.
    --pushed COMMIT=DAY is a commit's confirmed later push day (audit M2); --now stands in for the clock in the smokes
    only."""
    now = clock(args)
    records = records_at(args)
    cat = catalogue(records)
    chk, d, scans = A.check_freeze(records, cat, Path(args.runs), args.date, now=now, pushed=pushed_days(args),
                                   reader=R.trigger_reader(records), registered_cells=registered(args))
    for line in A.lines(chk, d, scans):
        print(line)
    return S.EXIT_PASS if d else S.EXIT_MISSING


# ------------------------------------------------------------------------------------------- freeze-body
def cmd_freeze_body(args) -> int:
    """The body to freeze (TMAN-B2-RULE-3; DL-24 R2-8): the draft committed by the end of --d-date D (its blob at C_D,
    with --pushed's confirmed push days, audit M2) and the body entries of the measurement's freeze cut (--cut, its
    committed and entered entry file), every other block as the draft holds it; written as tday_manifest.json into
    --out for `entry freeze --d-date D --manifest`."""
    records = records_at(args)
    doc = A.freeze_body(records, catalogue(records), args.d_date, args.cut, pushed=pushed_days(args))
    data = S.json_file_bytes(doc)
    out = write_new(out_folder(args.out, args.root), "tday_manifest.json", data)
    print(f"WROTE {out} sha256 {S.sha256_bytes(data)} body_sha256 {S.canonical_sha256(doc['body'])}")
    return S.EXIT_PASS


# ------------------------------------------------------------------------------------------------- main
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tday_manifest.py", description=__doc__.split("\n")[0])
    p.add_argument("--root", default=str(REPO), help="the repository whose HEAD holds the records")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--runs")
    b.add_argument("--registered-cells")
    b.add_argument("--am22-d", help="AM-22 item 1(b)'s candidate D: the draft's freeze is the end of D + 1 (B2R6)")
    v = sub.add_parser("validate")
    v.add_argument("--manifest", required=True)
    v.add_argument("--mode", required=True, choices=V.MODES)
    v.add_argument("--runs")
    v.add_argument("--freeze-record")
    v.add_argument("--unlock-entry")
    v.add_argument("--registered-cells")
    v.add_argument("--records", help="unlock, session: the records folder written from --records-commit")
    v.add_argument("--records-commit")
    v.add_argument("--am22-d", help="draft, freeze: AM-22 item 1(b)'s candidate D (B2R6)")
    v.add_argument("--dedup", help="session: the in-session hasher's tday_dedup.json (P2R6(a))")
    e = sub.add_parser("entry")
    e.add_argument("kind", choices=sorted(E.ENTRY_FILE_FIELDS))
    e.add_argument("--slot", action="append")
    e.add_argument("--date", required=True)
    e.add_argument("--fields", required=True)
    e.add_argument("--amendment", required=True)
    e.add_argument("--summary", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--nn", type=int, default=1)
    e.add_argument("--runs")
    e.add_argument("--manifest")
    e.add_argument("--rule", help="entry cut: the cut's rule (cutoff: AM-22 item 2), filling the fields' rule or "
                                  "equal to it")
    e.add_argument("--d-date", help="entry freeze: AM-22 item 1(b)'s D, its fields recomputed (audit m9)")
    e.add_argument("--registered-cells")
    e.add_argument("--push-evidence", help="entry unlock: the AM-22 freeze record's push evidence {response, list} "
                                           "(audit M4)")
    e.add_argument("--pushed", action="append", help="entry freeze --d-date: COMMIT=DAY, the confirmed later push day "
                                                     "of a commit (audit M2)")
    e.add_argument("--now", help="the smokes only: the instant that stands in for the clock (YYYY-MM-DDTHH:MM:SSZ)")
    a = sub.add_parser("arithmetic")
    a.add_argument("--slot", required=True)
    a.add_argument("--iterations", type=int, help="optional: the slot's horizon (checked); a QAT run's, its "
                                                     "reference run's steps")
    a.add_argument("--reference-telemetry", action="append", help="a run_id under --runs: the most recently "
                                                                   "finished run of the slot's stage")
    a.add_argument("--reference-smoke", help="a pod smoke's JSON, its path under --runs")
    a.add_argument("--smoke-key", default="e3")
    a.add_argument("--runs")
    a.add_argument("--now")
    a.add_argument("--out")
    ap = sub.add_parser("append")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--hashed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--ptq-record", help="the made model's ptq_record/1, a path under --runs (A2)")
    i = sub.add_parser("isolation-record")
    i.add_argument("--data-root", required=True)
    i.add_argument("--out", required=True)
    n = sub.add_parser("inventory")
    n.add_argument("--manifest", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--runs")
    n.add_argument("--registered-cells")
    n.add_argument("--freeze-record", help="the freeze record: its committed entry file (DL-24 #33)")
    n.add_argument("--dedup", help="the hasher's tday_dedup.json, inside --out: the export's dedup block (P2R4)")
    n.add_argument("--binding", help="the binding bind wrote, inside --out: the export's binding, bound (B3R7)")
    bd = sub.add_parser("bind")
    bd.add_argument("--manifest", required=True, help="the frozen manifest with its session part")
    bd.add_argument("--records", required=True, help="the records folder written from --records-commit")
    bd.add_argument("--records-commit", required=True)
    bd.add_argument("--dedup", required=True, help="the in-session hasher's tday_dedup.json")
    bd.add_argument("--inventory", required=True, help="the unbound export (inventory)")
    bd.add_argument("--runs", required=True, help="the input root: the run directories")
    bd.add_argument("--freeze-record", required=True)
    bd.add_argument("--unlock-entry", required=True)
    bd.add_argument("--registered-cells")
    bd.add_argument("--out", required=True, help="a new folder outside the repository")
    bd.add_argument("--now", help="the smokes only: the instant that stands in for the clock (YYYY-MM-DDTHH:MM:SSZ)")
    k = sub.add_parser("check-freeze")
    k.add_argument("--date", required=True, help="X: the check reads the records committed by the end of X")
    k.add_argument("--runs", required=True)
    k.add_argument("--registered-cells")
    k.add_argument("--pushed", action="append", help="COMMIT=DAY: the confirmed later push day of a commit (audit M2)")
    k.add_argument("--now", help="the smokes only: the instant that stands in for the clock (YYYY-MM-DDTHH:MM:SSZ)")
    f = sub.add_parser("freeze-body")
    f.add_argument("--d-date", required=True, help="AM-22 item 1(b)'s D: the draft committed by its end")
    f.add_argument("--cut", action="append", required=True, help="the measurement's freeze cut: its committed entry "
                                                                 "file (TMAN-B2-RULE-3)")
    f.add_argument("--out", required=True)
    f.add_argument("--pushed", action="append", help="COMMIT=DAY: the confirmed later push day of a commit (audit M2)")
    return p


COMMANDS = {"build": cmd_build, "validate": cmd_validate, "entry": cmd_entry, "arithmetic": cmd_arithmetic,
            "append": cmd_append, "isolation-record": cmd_isolation, "inventory": cmd_inventory,
            "check-freeze": cmd_check_freeze, "freeze-body": cmd_freeze_body, "bind": cmd_bind}


# Every path-valued argument (a run id under --runs, a file, a folder): refused for "test" before any call on any of
# them (DL-24 #7, #10); a part read under --runs or --root is checked there again where it is joined.
PATH_ARGS = ("root", "sources", "out", "runs", "registered_cells", "manifest", "freeze_record", "unlock_entry",
             "records", "fields", "reference_telemetry", "reference_smoke", "hashed", "ptq_record", "data_root",
             "push_evidence", "cut", "dedup", "binding", "inventory")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        for name in PATH_ARGS:
            v = getattr(args, name, None)
            for p in v if isinstance(v, list) else [] if v is None else [v]:
                if "test" in str(p).lower():
                    raise S.TmanRefused("test_path", f"--{name.replace('_', '-')} {p}: a path containing 'test' is "
                                                     "refused (project rule 4)")
        S.refuse_test_path(args.root)               # as resolved too: the records' accessor resolves --root
        return COMMANDS[args.cmd](args)
    except S.TmanRefused as e:
        print(e.line())
        return S.EXIT_REFUSED
    except Exception as e:  # noqa: BLE001 -- an unexpected error is exit 4, never a pass
        print(f"ERROR -- {type(e).__name__}: {e}")
        return S.EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
