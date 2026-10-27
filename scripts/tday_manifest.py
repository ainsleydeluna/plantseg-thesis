#!/usr/bin/env python3
"""The TEST-day manifest's command line (lane TMAN, B-1, C5; plan §0 and decision (xvii)).

Commands:
  build             a draft manifest from the committed records and a sources file (plan §1.2), validated in draft
                    mode, written as tday_manifest.json into --out
  validate          the validator (src/tman/validate.py) in mode draft, freeze, unlock or session
  entry KIND        an entry file (tday_entry/1) checked, written into --out, and its decision-log row printed
  arithmetic        AM-19 item 3(b)'s arithmetic for a run (scripts/preflight_arms.py's cut_arithmetic, reused); with
                    --out, the cut entry it implies when the run does not fit
  append            a derived artifact's hashed form appended to the manifest, written into --out
  isolation-record  src/data/isolation.assert_trainval_only_root on --data-root, written as tday_isolation/1
  inventory         the inventory export tday-inventory/1 (IF§8; IF§11 v2), written into --out and read back

Every output is a new file created exclusively in --out, a folder outside the repository; no command writes in the
repository, and none reads TEST data. Records are read at HEAD of the repository at --root (default: this
checkout). Exit codes follow the select scripts (src/training/sweep_select.py:203): 0 pass or written, 2 refused,
3 incomplete (draft), 4 error.
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


def read_json(path, code: str):
    S.refuse_test_path(path)
    p = Path(path)
    return S.strict_json(p.read_bytes(), str(p), code=code)


def records_at(args) -> R.Records:
    return R.Records.at_head(args.root)


def catalogue(records: R.Records):
    return S.load_catalogue(records.read(S.CATALOGUE_REL))


def registered(args):
    return None if args.registered_cells is None else list(read_json(args.registered_cells, "generator_cells_mismatch"))


def print_result(res: V.Result) -> int:
    for line in res.missing + res.reports:
        print(line)
    state = "MISSING" if res.missing else "PASS"
    print(f"{state}: mode {res.mode}; {len(res.passed)} gates passed")
    return res.exit_code


# ------------------------------------------------------------------------------------------------- build
def cmd_build(args) -> int:
    records = records_at(args)
    cat = catalogue(records)
    sources = read_json(args.sources, "manifest_format")
    doc = M.build_manifest(records, cat, sources, runs_root=args.runs)
    data = S.json_file_bytes(doc)
    folder = out_folder(args.out, args.root)
    res = V.validate(V.Ctx(mode="draft", manifest_bytes=data, manifest_path=str(folder / "tday_manifest.json"),
                           records=records, registered_cells=registered(args)))       # a refusal writes nothing (#61)
    out = write_new(folder, "tday_manifest.json", data)
    print(f"WROTE {out} sha256 {S.sha256_bytes(data)} body_sha256 {S.canonical_sha256(doc['body'])}")
    return print_result(res)


# ---------------------------------------------------------------------------------------------- validate
def cmd_validate(args) -> int:
    """Draft and freeze read the records at HEAD of --root; unlock and session run at the session checkout S = E
    (--root) and read the records from --records, a folder written from --records-commit (PL-28; TB16)."""
    S.refuse_test_path(args.manifest)
    data = Path(args.manifest).read_bytes()
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
    else:
        records = records_at(args)
    rel = records.src.rel(args.manifest)
    res = V.validate(V.Ctx(mode=args.mode, manifest_bytes=data, manifest_path=rel or str(args.manifest),
                           records=records, runs_root=args.runs, freeze_record=args.freeze_record,
                           unlock_entry=args.unlock_entry, registered_cells=registered(args), now=time.time()))
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
    E.check_cut(doc, printer_cut_context(records, cat, freeze, doc, args.runs))


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
    if doc["body_sha256"] != S.canonical_sha256(manifest_body(args, "run_added_after_freeze")):
        raise S.TmanRefused("run_added_after_freeze", "the freeze record records --manifest's body")


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
    doc = E.make_entry(args.kind, args.slot or [], args.date, fields, amendment=args.amendment, script=SCRIPT_REL,
                       commit=records.src.head() or None)
    freeze = D.freeze_instant(D.schedule_block(records.src))
    ENTRY_CHECKS[doc["kind"]](args, records, cat, freeze, doc)
    path, rel, sha = E.write_entry(doc, out_folder(args.out, args.root), nn=args.nn, records=records)
    print(f"WROTE {path} sha256 {sha}")
    print("DECISION-LOG ROW (the orchestrator assigns the ID; commit the file at its path with it):")
    print(E.dl_row(doc, rel, sha, args.summary))
    return S.EXIT_PASS


def printer_cut_context(records: R.Records, cat, freeze, doc: dict, runs) -> E.CutContext:
    """The cut printer's context, the validator's (validate.cut_context): the registry, the decision records, every
    run slot's launched attempts and status from the launch log and the run directories under --runs
    (completion.RunIndex; never from the entry), a named rule's record and the records committed by a day (E3-X's
    run of record is read from the manifest by the validator). A cut is printed from the run directories: --runs is
    required."""
    from src.training import kd_arms
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

    return E.CutContext(cat, freeze, registry={s.run_stage: s for s in kd_arms.load_arms().values()},
                        decision_records=recs, attempts=idx.attempts, run_slot=idx.slot_of.get,
                        stopped=idx.stopped_ids(), status=lambda slot: idx.status(slot, freeze.original_utc),
                        rule_record=E.rule_record_reader(records), known_by=E.known_by_reader(records))


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
        E.check_cut(doc, printer_cut_context(records, cat, D.freeze_instant(sch), doc, args.runs))
        path, rel, sha = E.write_entry(doc, out_folder(args.out, args.root), records=records)
        print(f"WROTE {path} sha256 {sha}")
        print(E.dl_row(doc, rel, sha, "cut before its launch by AM-19 item 3(b)'s arithmetic."))
    return S.EXIT_PASS


# ------------------------------------------------------------------------------------------------ append
def cmd_append(args) -> int:
    """The validator's append checks on the one new append: append_invalid (the appends' history, then the append
    structurally), ptq_record_invalid, append_invalid with the records it cites, and under --runs
    derived_hash_mismatch."""
    S.refuse_test_path(args.manifest)
    data = Path(args.manifest).read_bytes()
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
    freeze = D.freeze_instant(D.schedule_block(records.src))
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
    S.refuse_test_path(args.manifest)
    data = Path(args.manifest).read_bytes()
    doc = S.strict_json(data, args.manifest, code="manifest_unreadable")
    records = records_at(args)
    cat = catalogue(records)
    inv = M.build_inventory(doc, data, records, cat, runs_root=args.runs, registered=registered(args),
                            freeze_record=args.freeze_record)
    errs = V.inventory_errors(S.strict_json(S.json_file_bytes(inv), "the inventory", code="inventory_invalid"), cat)
    if errs:                                                      # a refusal writes nothing (DL-24 #61)
        raise S.TmanRefused("inventory_invalid", "; ".join(errs[:6]))
    out = write_new(out_folder(args.out, args.root), "tday_inventory.json", S.json_file_bytes(inv))
    V.read_inventory(out)
    print(f"WROTE {out} sha256 {S.sha256_bytes(out.read_bytes())} ({len(inv['models'])} model rows, "
          f"{len(inv['entries'])} entries)")
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
    v = sub.add_parser("validate")
    v.add_argument("--manifest", required=True)
    v.add_argument("--mode", required=True, choices=V.MODES)
    v.add_argument("--runs")
    v.add_argument("--freeze-record")
    v.add_argument("--unlock-entry")
    v.add_argument("--registered-cells")
    v.add_argument("--records", help="unlock, session: the records folder written from --records-commit")
    v.add_argument("--records-commit")
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
    return p


COMMANDS = {"build": cmd_build, "validate": cmd_validate, "entry": cmd_entry, "arithmetic": cmd_arithmetic,
            "append": cmd_append, "isolation-record": cmd_isolation, "inventory": cmd_inventory}


# Every path-valued argument (a run id under --runs, a file, a folder): refused for "test" before any call on any of
# them (DL-24 #7, #10); a part read under --runs or --root is checked there again where it is joined.
PATH_ARGS = ("root", "sources", "out", "runs", "registered_cells", "manifest", "freeze_record", "unlock_entry",
             "records", "fields", "reference_telemetry", "reference_smoke", "hashed", "ptq_record", "data_root")


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
