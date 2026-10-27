#!/usr/bin/env python3
"""Smoke for the unlock and session modes (plan §2.1-§2.2; AM-19 items 3(c), 3(d), 4(c); AM-17 item 9; TB11, TB14,
TB16) (lane TMAN, B-1, C5): the validator at the session checkout S = E (TB16) with the records read from a records
folder verified against the records commit (sweep_select's RecordsSource, PL-28); one case per unlock and session
gate, each reached with every earlier gate passing.

The unlock world is smoke_tman_validate's world plus the lane register (one lane implemented, one not required with
its covering cut entry), the freeze record and the unlock entry; S is a linked checkout of E in the scratch
repository. The session world adds the session part: the clock-offset record and the push evidence (TB11).

Run:  python -B scripts/smoke_tman_unlock.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_pins as PINS  # noqa: E402
import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)
import scripts.smoke_tman_validate as VW  # noqa: E402  (the fixture world)

check, expect_refused = FX.check, FX.expect_refused
UNLOCK_DAY, SESSION_UTC = "2026-11-04", "2026-11-05T02:00:00Z"
WANTED: set = set()


class Unlock:
    """The fixture world at the unlock (and, after `start_session`, in the session)."""

    def __init__(self):
        from src.tman import schema as S
        self.w = w = VW.World()
        self.S = S
        self.wt = FX.scratch_dir("tmanwt")
        self.wt.rmdir()
        FX.git(w.repo, "worktree", "add", "-q", "--detach", str(self.wt), w.E)
        ref_arm = w.cut_refs["ARM:latency"]
        nr = w.entry_file("not_required", ["ARM:latency"], {"lane": "L-AM16-ARM", "cut_entries": [
            {"path": ref_arm["path"], "sha256": ref_arm["sha256"]}]})
        impl = w.row("L-CORR IMPLEMENTED (merge of record)", status="IMPLEMENTED (fixture merge)")
        self.lanes = {"format": "tday_lanes/1", "lanes": [
            {"lane": "L-CORR", "serves": ["E1:s42:80k", "E6:s42:sel"],
             "status": {"kind": "implemented", "dl": impl, "merge_commit": w.E, "entry": None}},
            {"lane": "L-AM16-ARM", "serves": ["ARM:latency"],
             "status": {"kind": "not_required", "dl": None, "merge_commit": None,
                        "entry": {"path": nr["path"], "sha256": nr["sha256"]}}}]}
        lanes = S.json_file_bytes(self.lanes)
        self.freeze = w.entry_file("freeze", [], {"body_sha256": S.canonical_sha256(w.doc["body"]),
                                                  "freeze_date": VW.FREEZE_DAY, "source": "schedule"})
        w.commit({nr["path"]: w.files[nr["path"]], S.LANES_REL: lanes,
                  self.freeze["path"]: w.files[self.freeze["path"]]}, "lanes, not-required entry, freeze record")
        self.unlock = w.entry_file("unlock", [], {
            "manifest": {"path": S.MANIFEST_REL, "sha256": S.sha256_bytes(w.data),
                         "body_sha256": S.canonical_sha256(w.doc["body"]), "appends_sha256": S.canonical_sha256([])},
            "lanes": {"path": S.LANES_REL, "sha256": S.sha256_bytes(lanes)},
            "freeze_record": {"path": self.freeze["path"], "sha256": self.freeze["sha256"]}}, date=UNLOCK_DAY)
        self.H = w.commit({self.unlock["path"]: w.files[self.unlock["path"]]}, "unlock entry")
        self.doc, self.data = w.doc, w.data

    def export(self, head: str | None = None) -> tuple[Path, str]:
        """(a records folder written from `head`, default the latest commit; that commit): reports/ and the decision
        log."""
        w = self.w
        head = head or FX.git(w.repo, "rev-parse", "HEAD")
        folder = FX.scratch_dir("tmanrec")
        names = FX.git(w.repo, "ls-tree", "-r", "--name-only", head, "--", "reports", "docs/DECISION_LOG.md")
        for rel in names.splitlines():
            f = folder / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(_blob(w.repo, head, rel))
        return folder, head

    def records(self):
        """The records at the latest commit, from a records folder written from it, read with S = E."""
        from src.tman import records as R
        folder, head = self.export()
        return R.Records.at_records_commit(folder, head, root=self.wt)

    def commit_manifest(self, doc: dict) -> bytes:
        data = self.S.json_file_bytes(doc)
        self.w.commit({self.S.MANIFEST_REL: data}, "manifest")
        return data

    def ctx(self, mode="unlock", data=None, **kw):
        from src.tman import validate as V
        args = dict(mode=mode, manifest_bytes=self.data if data is None else data, manifest_path=self.S.MANIFEST_REL,
                    records=self.records(), runs_root=self.w.runs, freeze_record=self.freeze["path"],
                    unlock_entry=self.unlock["path"], registered_cells=PINS.IF6_CELLS, ptq_reader=self.w.reader(),
                    trigger_reader=self.w.trigger_reader(), repo_root=self.wt)
        args.update(kw)
        return V.Ctx(**args)

    def run(self, mode="unlock", data=None, **kw):
        from src.tman import validate as V
        try:
            return None, V.validate(self.ctx(mode, data, **kw)).passed, None
        except self.S.TmanRefused as e:
            return e.code, getattr(e, "passed", None), e
        except AssertionError:
            raise                                       # the score sentinel's signal (smoke_tman_noscore)
        except Exception as e:  # noqa: BLE001 -- a crash where a refusal is due fails its check
            return type(e).__name__, None, self.S.TmanRefused(type(e).__name__, str(e))

    def start_session(self, offset: float = 0.4, push_utc: str = "2026-11-04T09:00:00Z",
                      clock_status: str = "RECORDED", started: str = SESSION_UTC, host: dict | None = None,
                      clock_host: str | None = None, edit=None) -> bytes:
        """The session part, committed once (`edit` changes it first): on `host` (the frozen execution context's
        by default), the clock-offset record of that host (or `clock_host`; its decision-log row of
        `clock_status`) and the push evidence of the unlock entry's commit."""
        S, w = self.S, self.w
        host = host or dict(self.doc["body"]["execution_context"]["host"])
        clock = w.entry_file("clock_offset", [], {"host": clock_host or host["name"], "source": "chronyc tracking",
                                                  "offset_seconds": offset, "measured_utc": "2026-11-05T01:50:00Z"},
                             date="2026-11-05", nn=_nn(w, "clock_offset"))
        w.dl[clock["dl"]] = (clock_status, w.dl[clock["dl"]][1])
        adding = FX.git(w.repo, "log", "--format=%H", "-1", "--", self.unlock["path"])
        resp = S.json_file_bytes({"fixture": "the remote's activity response, saved verbatim"})
        lst = S.json_file_bytes({"format": "tday_push_list/1", "pushes": [
            {"ref": "refs/heads/claude/keen-curie-u4a8ig", "after": adding, "timestamp": push_utc}]})
        tag = push_utc.replace(":", "").replace("-", "")
        rr, lr = f"reports/derived/tday_unlock_push_{tag}.json", f"reports/derived/tday_unlock_push_list_{tag}.json"
        w.commit({clock["path"]: w.files[clock["path"]], rr: resp, lr: lst}, "clock offset and push evidence")
        doc = copy.deepcopy(self.doc)
        doc["session"] = {"unlock_entry": {"path": self.unlock["path"], "sha256": self.unlock["sha256"]},
                          "clock_offset": {"path": clock["path"], "sha256": clock["sha256"]},
                          "started_utc": started, "host": host, "values": {},
                          "push_evidence": {"response": {"path": rr, "sha256": S.sha256_bytes(resp)},
                                            "list": {"path": lr, "sha256": S.sha256_bytes(lst)}}}
        if edit is not None:
            edit(doc)
        return self.commit_manifest(doc)


def _blob(repo: Path, commit: str, rel: str) -> bytes:
    import subprocess
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{rel}"], capture_output=True, check=True)
    return r.stdout


def _nn(w, kind: str) -> int:
    _nn.count = getattr(_nn, "count", {})
    _nn.count[kind] = _nn.count.get(kind, 0) + 1
    return _nn.count[kind]


def gate_codes(mode: str) -> list:
    return VW.gate_codes(mode)


def expect_gate(u: Unlock, name: str, want: str, mode: str = "unlock", data=None, detail: str = "", **kw) -> None:
    """`want` is refused, by the check whose message holds `detail`, with every earlier gate of `mode` passing."""
    WANTED.add(want)
    code, passed, e = u.run(mode, data, **kw)
    order = gate_codes(mode)
    before = order[:order.index(want)] if want in order else None
    check(name, code == want and passed == before and detail in (e.detail if e else ""),
          f"code {code} ({(e.detail[:160] if e else '')}); passed {len(passed or [])}, want {want} after "
          f"{len(before or [])}")


def committed_raw(u: Unlock, kind: str, doc: dict, nn: int, day: str) -> str:
    """A session entry file of `kind` written as given (no field check), committed with its RECORDED row."""
    S, w = u.S, u.w
    rel = S.entry_file_rel(S.entry_file_name(kind, S.SESSION_FILE_ID, day, nn))
    data = S.json_file_bytes(doc)
    w.row(f"TMAN {kind} entry for the session: {rel} sha256 {S.sha256_bytes(data)}")
    w.commit({rel: data}, f"a {kind} entry as given")
    return rel


def refreeze(u: Unlock, doc: dict, nn: int) -> dict:
    """A freeze record of a changed body, committed (the unlock checks it before the unlock entry exists)."""
    S, w = u.S, u.w
    fr = w.entry_file("freeze", [], {"body_sha256": S.canonical_sha256(doc["body"]), "freeze_date": VW.FREEZE_DAY,
                                     "source": "schedule"}, nn=nn)
    w.commit({fr["path"]: w.files[fr["path"]]}, "a freeze record of a changed body")
    return fr


# ======================================================================================================= cases
def case_unlock_world():
    u = UW
    code, passed, e = u.run("unlock")
    check("unlock_world_passes_every_unlock_gate_S_equals_E", code is None and passed == gate_codes("unlock"),
          f"{code}: {e.detail[:300] if e else ''}")
    code, passed, e = u.run("unlock", unlock_entry=None, now=VW.FX.utc("2026-11-04T09:00:00Z"))
    check("unlock_before_its_entry_passes", code is None, f"{code}: {e.detail[:200] if e else ''}")
    code, _, e = u.run("freeze")
    check("unlock_world_passes_freeze", code is None, f"{code}: {e.detail[:200] if e else ''}")


def case_cli():
    """The command line's unlock mode at the session checkout S = E (--root), the records from --records written
    from --records-commit (PL-28; TB16); refused without them. Lane LQP's reader is the world's stand-in for the
    call (records.ptq_record_reader patched and restored)."""
    import contextlib
    import io
    import scripts.tday_manifest as CLI
    from src.tman import records as R
    u, S = UW, UW.S
    folder, head = u.export()
    cells = FX.scratch_dir("tmancells") / "cells.json"
    cells.write_text(json.dumps(PINS.IF6_CELLS), encoding="utf-8")
    argv = ["--root", str(u.wt), "validate", "--manifest", str(folder / S.MANIFEST_REL), "--mode", "unlock",
            "--runs", str(u.w.runs), "--freeze-record", u.freeze["path"], "--unlock-entry", u.unlock["path"],
            "--registered-cells", str(cells)]

    def cli(args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CLI.main(args)
        return rc, buf.getvalue()
    saved = R.ptq_record_reader
    try:
        R.ptq_record_reader = lambda: u.w.reader()
        rc, out = cli(argv + ["--records", str(folder), "--records-commit", head])
        check("cli_unlock_without_a_trigger_reader_refused_O8", rc == 2 and "trigger_format_unknown" in out,
              out.strip()[-200:])
        R.TRIGGER_READER = u.w.trigger_reader()
        rc, out = cli(argv + ["--records", str(folder), "--records-commit", head])
        check("cli_unlock_at_S_with_the_records_folder_passes", rc == 0 and
              f"PASS: mode unlock; {len(gate_codes('unlock'))} gates passed" in out, out.strip()[-200:])
        rc, out = cli(argv)
        check("cli_unlock_without_the_records_folder_refused", rc == 2 and "record_uncommitted" in out
              and "--records" in out, out.strip()[-200:])
        rc, out = cli(argv + ["--records", str(folder)])
        check("cli_records_without_its_commit_refused", rc == 2 and "go together" in out, out.strip()[-200:])
    finally:
        R.ptq_record_reader = saved
        R.TRIGGER_READER = None


def case_unlock_printer():
    """The unlock printer checks the would-be record set before it writes (plan §0; DL-24 #9): the committed
    manifest's file, body and appends; the freeze record of that body; the committed lane register; and no unlock
    entry yet -- this world already records one, so the faithful fields reach second_session."""
    u, w = UW, UW.w
    ud = u.unlock["doc"]
    fields = {k: ud[k] for k in ("manifest", "lanes", "freeze_record")}
    other = refreeze(u, dict(u.doc, body=dict(u.doc["body"], expected_rows={"value": 1})), 8)
    for name, f, want in (
            ("printer_unlock_of_another_manifest_file_B9", dict(fields, manifest=dict(ud["manifest"], sha256="0" * 64)),
             "run_added_after_freeze"),
            ("printer_unlock_citing_the_freeze_record_of_another_body_B9",
             dict(fields, freeze_record={"path": other["path"], "sha256": other["sha256"]}), "run_added_after_freeze"),
            ("printer_unlock_citing_another_lane_register_B9", dict(fields, lanes=dict(ud["lanes"], sha256="0" * 64)),
             "lane_unmet"),
            ("printer_unlock_after_an_unlock_entry_B9", fields, "second_session")):
        VW.printed(w, name, "unlock", [], f, want, date=UNLOCK_DAY)


def case_unlock_gates():
    S, w = UW.S, UW.w
    u = UW
    ua = Unlock()                       # a committed append stays in the history (plan §1.1): a world of its own
    iso = S.json_file_bytes({"format": S.ISOLATION_FORMAT, "root": "/workspace/plantseg_trainval",
                             "image_digest": S.KD_IMAGE, "utc": "2026-11-04T09:00:00Z",
                             "result": {"method": "fixture"}})
    rel = "reports/derived/tday_isolation_unlock.json"
    ua.w.commit({rel: iso}, "an isolation record")
    h = ua.w.entry(ua.doc, "E4:s42:list45")["hashed"]
    prov = VW.runs_variant(ua.w, h["provenance"], f"variant_made_{VW.stamp(VW.LATE)}.json", {"created_utc": VW.LATE})
    d = copy.deepcopy(ua.doc)
    ua.w.entry(d, "E4:s42:list45")["hashed"] = None
    d["appends"] = [{"slot": "E4:s42:list45", "hashed": dict(
        h, appended="append", made_utc=VW.LATE, isolation={"path": rel, "sha256": S.sha256_bytes(iso)},
        provenance=prov), "ptq_record": None}]
    data = ua.commit_manifest(d)
    code, _, e = ua.run("freeze", data)
    check("the_append_itself_passes_freeze_mode", code is None, f"{code}: {e.detail[:200] if e else ''}")
    expect_gate(ua, "appends_changed_after_the_unlock_record", "append_after_unlock", data=data,
                detail="appends changed after the unlock record")
    d2 = copy.deepcopy(u.doc)
    w.entry(d2, "E4:s42:list45")["hashed"] = None
    data = u.commit_manifest(d2)
    expect_gate(u, "derived_by_inputs_at_the_unlock", "derived_not_hashed_at_unlock", data=data,
                detail="E4:s42:list45")
    d3 = copy.deepcopy(u.doc)
    d3["body"]["execution_context"]["processes"] = 3
    data = u.commit_manifest(d3)
    expect_gate(u, "body_changed_after_the_freeze_record", "run_added_after_freeze", data=data, unlock_entry=None,
                now=FX.utc("2026-11-04T09:00:00Z"), detail="the body changed after the freeze record")
    d4 = copy.deepcopy(u.doc)
    d4["body"]["entries"] = [e for e in d4["body"]["entries"] if e["slot"] != "G:s42:80k"]
    data = u.commit_manifest(d4)
    fr = refreeze(u, d4, 4)
    expect_gate(u, "slot_without_entry_at_the_unlock", "cut_unentered_at_unlock", data=data,
                freeze_record=fr["path"], unlock_entry=None, now=FX.utc("2026-11-04T09:00:00Z"), detail="G:s42:80k")
    data = u.commit_manifest(json.loads(u.data))
    check("the_manifest_of_record_recommitted_is_byte_equal", data == u.data)
    u.w.commit({S.MANIFEST_REL: json.dumps(u.doc, indent=1).encode()}, "the manifest re-serialised")
    expect_gate(u, "manifest_file_not_the_unlock_entry_s", "run_added_after_freeze",
                data=json.dumps(u.doc, indent=1).encode(), detail="not the one the unlock entry records")
    u.commit_manifest(u.doc)
    lanes = copy.deepcopy(u.lanes)
    lanes["lanes"][0]["serves"].append("E1:s43:80k")
    w.commit({S.LANES_REL: S.json_file_bytes(lanes)}, "a lane register changed after the unlock entry")
    expect_gate(u, "lanes_not_the_unlock_entry_s", "lane_unmet", detail="not the lane register the unlock entry")
    lanes = copy.deepcopy(u.lanes)
    lanes["lanes"][0]["status"]["dl"] = "DL-63"
    w.commit({S.LANES_REL: S.json_file_bytes(lanes)}, "a lane not implemented")
    expect_gate(u, "lane_neither_implemented_nor_not_required", "lane_unmet", unlock_entry=None,
                now=FX.utc("2026-11-04T09:00:00Z"), detail="lane L-CORR: not IMPLEMENTED")
    lanes = copy.deepcopy(u.lanes)
    lanes["lanes"][1]["serves"].append("G:s42:80k")
    w.commit({S.LANES_REL: S.json_file_bytes(lanes)}, "a not-required lane serving an uncut slot")
    expect_gate(u, "not_required_lane_serving_an_uncut_slot", "not_required_invalid", unlock_entry=None,
                now=FX.utc("2026-11-04T09:00:00Z"), detail="serves ['G:s42:80k'] with no cut entry")
    ref_arm = w.cut_refs["ARM:latency"]
    arm_cut = w.entry_file("cut", ["ARM:latency"], dict(w.cut_fields("ARM", ["ARM:latency"], "code not merged", False),
                                                        reason="gate not met"), nn=2)
    for name, cut_ref, status in (("not_required_entry_not_DL_recorded_B55c", ref_arm, "DECIDED"),
                                  ("not_required_cut_entry_not_DL_recorded_B55c", arm_cut, "RECORDED")):
        nr = w.entry_file("not_required", ["ARM:latency"], {"lane": "L-AM16-ARM", "cut_entries": [
            {"path": cut_ref["path"], "sha256": cut_ref["sha256"]}]}, nn=_nn(w, "not_required"), date=UNLOCK_DAY)
        w.dl[nr["dl"]] = (status, w.dl[nr["dl"]][1])
        if cut_ref is arm_cut:
            w.dl[arm_cut["dl"]] = ("DECIDED", w.dl[arm_cut["dl"]][1])
        lanes = copy.deepcopy(u.lanes)
        lanes["lanes"][1]["status"]["entry"] = {"path": nr["path"], "sha256": nr["sha256"]}
        w.commit({nr["path"]: w.files[nr["path"]], arm_cut["path"]: w.files[arm_cut["path"]],
                  S.LANES_REL: S.json_file_bytes(lanes)}, f"lanes: {name}")
        expect_gate(u, name, "dl_row_missing", unlock_entry=None, now=FX.utc("2026-11-04T09:00:00Z"),
                    detail=f"{(nr if status == 'DECIDED' else arm_cut)['path']}: sha256")
    w.commit({S.LANES_REL: S.json_file_bytes(u.lanes)}, "the lane register of record")
    f = u.wt / "scripts/evaluate_model.py"
    saved = f.read_bytes()
    try:
        f.write_bytes(saved + b"# edited in the session checkout\n")
        expect_gate(u, "evaluator_file_edited_at_S", "evaluator_files_changed", detail="scripts/evaluate_model.py")
    finally:
        f.write_bytes(saved)
    f = u.wt / "scripts/tday_hash_images.py"
    saved = f.read_bytes()
    try:
        f.write_bytes(saved + b"# edited\n")
        expect_gate(u, "hasher_script_not_its_pin", "hasher_pin_mismatch",
                    detail="scripts/tday_hash_images.py at S is not the pinned script")
    finally:
        f.write_bytes(saved)
    hash_only_branch(u)
    records_outside_e_history(u)
    session_checkout_not_e(u)
    expect_gate(u, "unlock_after_the_test_deadline_TB14", "session_after_deadline", unlock_entry=None,
                now=FX.utc("2026-11-08T16:00:00Z"), detail="after the TEST deadline 2026-11-08")
    code, _, _ = u.run("unlock", unlock_entry=None, now=FX.utc("2026-11-08T15:59:59Z"))
    check("unlock_on_the_deadline_day_passes", code is None)
    expect_gate(u, "unlock_entry_citing_another_freeze_record", "run_added_after_freeze",
                freeze_record=refreeze(u, u.doc, 5)["path"], detail="does not cite this freeze record")
    ud = u.unlock["doc"]
    other = w.entry_file("unlock", [], {"manifest": dict(ud["manifest"], body_sha256="0" * 64), "lanes": ud["lanes"],
                                        "freeze_record": ud["freeze_record"]}, nn=2, date=UNLOCK_DAY)
    w.commit({other["path"]: w.files[other["path"]]}, "an unlock entry of another body")
    expect_gate(u, "unlock_entry_of_another_body", "run_added_after_freeze", unlock_entry=other["path"],
                detail="not the one the unlock entry records")
    expect_gate(u, "second_unlock_entry_in_the_decision_log", "second_session", detail="two unlock entries")
    rel = committed_raw(u, "unlock", dict(ud, manifest=S.MANIFEST_REL), 3, UNLOCK_DAY)
    expect_gate(u, "unlock_entry_whose_manifest_is_a_string_B89", "entry_file_invalid", unlock_entry=rel,
                detail="manifest")
    fr = {k: v for k, v in u.freeze["doc"].items() if k != "body_sha256"}
    rel = committed_raw(u, "freeze", fr, 7, VW.FREEZE_DAY)
    expect_gate(u, "freeze_record_without_body_sha256_B89", "entry_file_invalid", freeze_record=rel,
                unlock_entry=None, now=FX.utc("2026-11-04T09:00:00Z"), detail="body_sha256")


def hash_only_branch(u: Unlock) -> None:
    """A hash-only evaluator file (manifest.hash_only; a path holding "test", which no fixture creates) is compared
    by its whole-file sha256 against its blob at E: shown on a fixture module marked hash-only for the call."""
    from src.tman import manifest as M
    rel = "src/eval/fixture_scoring.py"
    saved_fn, f = M.hash_only, u.wt / rel
    saved = f.read_bytes()
    try:
        M.hash_only = lambda r: r == rel or saved_fn(r)
        code, _, e = u.run("unlock")
        check("hash_only_evaluator_file_unchanged_passes", code is None, f"{code}: {e.detail[:200] if e else ''}")
        f.write_bytes(saved + b"# edited\n")
        expect_gate(u, "hash_only_evaluator_file_edited_at_S", "evaluator_files_changed", detail=rel)
    finally:
        f.write_bytes(saved)
        M.hash_only = saved_fn


def session_checkout_not_e(u: Unlock) -> None:
    """TB16: the session checkout S is E; a checkout of the records commit is refused."""
    from src.tman import records as R
    w = u.w
    other = FX.scratch_dir("tmanwt")
    other.rmdir()
    FX.git(w.repo, "worktree", "add", "-q", "--detach", str(other), FX.git(w.repo, "rev-parse", "HEAD"))
    folder, head = u.export()
    expect_gate(u, "session_checkout_not_e_TB16", "evaluator_commit_invalid",
                records=R.Records.at_records_commit(folder, head, root=other), repo_root=other, detail="TB16: S = E")


def records_outside_e_history(u: Unlock) -> None:
    """The records commit must hold E in its history (as HEAD must at the freeze): the same records committed as the
    root of another history are refused."""
    from src.tman import records as R
    w = u.w
    side = FX.scratch_dir("tmanside")
    side.rmdir()
    FX.git(w.repo, "worktree", "add", "-q", "--detach", str(side), FX.git(w.repo, "rev-parse", "HEAD"))
    FX.git(side, "checkout", "-q", "--orphan", "tman-side")
    FX.git(side, "commit", "-q", "-m", "the records as the root of another history")
    folder, h = u.export(FX.git(side, "rev-parse", "HEAD"))
    expect_gate(u, "records_commit_without_e_in_its_history", "evaluator_commit_invalid",
                records=R.Records.at_records_commit(folder, h, root=u.wt), detail="the records' history")


def case_session():
    """The session (AM-17 item 9; TB11, TB14; AM-19a reading 21): one session part over the manifest's history, its
    values only added (DL-24 #56); a variant of the part each in a world of its own (a second part is a second
    session)."""
    u = Unlock()
    data = u.start_session()
    code, passed, e = u.run("session", data)
    check("session_world_passes_every_session_gate", code is None and passed == gate_codes("session"),
          f"{code}: {e.detail[:300] if e else ''}")
    code, _, e = u.run("unlock", data)
    check("unlock_mode_refuses_a_session_part", code == "test_value_before_unlock", f"{code}")
    doc = json.loads(data)
    doc["session"]["values"] = {"recorded": "a value recorded in the session (fixture)"}
    code, _, e = u.run("session", u.commit_manifest(doc))
    check("session_values_added_in_the_same_session_pass_B56", code is None, f"{code}: {e.detail[:200] if e else ''}")
    expect_gate(u, "session_values_removed_B56", "second_session", mode="session", data=u.commit_manifest(
        json.loads(data)), detail="its values only added")
    code, _, e = u.run("unlock", u.commit_manifest(u.doc))
    check("unlock_after_a_committed_session_part_refused_B56", code == "second_session"
          and "already holds a session part" in (e.detail if e else ""), f"{code}: {e.detail[:200] if e else ''}")
    for name, kw, want, detail in (
            ("session_started_after_the_deadline_TB14", {"started": "2026-11-08T16:00:01Z"}, "session_after_deadline",
             "after the TEST deadline"),
            ("clock_offset_over_60_s_AM19a_r21", {"offset": 75.0}, "clock_offset_invalid", "within 60 s"),
            ("clock_offset_entry_not_DL_recorded_B102", {"offset": 0.6, "clock_status": "DECIDED"}, "dl_row_missing",
             "clock_offset__session__20261105"),
            ("unlock_pushed_after_the_session_started_TB11", {"push_utc": "2026-11-05T03:00:00Z"},
             "push_evidence_invalid", "no listed push carries the unlock entry"),
            ("session_host_not_the_frozen_host_B37", {"host": {"name": "another-host", "cpu_model": "x86-64 fixture"}},
             "clock_offset_invalid", "is not the frozen execution context's host"),
            ("clock_offset_record_of_another_host_B37", {"clock_host": "another-host"}, "clock_offset_invalid",
             "is not the session host's")):
        v = Unlock()
        expect_gate(v, name, want, mode="session", data=v.start_session(**kw), detail=detail)
    v = Unlock()
    data = v.start_session()
    again = json.loads(data)
    again["session"]["started_utc"] = "2026-11-05T03:00:00Z"
    expect_gate(v, "second_session_part_over_the_first_B56", "second_session", mode="session",
                data=v.commit_manifest(again), detail="holds another session part")
    v.commit_manifest(json.loads(data))
    v.w.commit({"reports/derived/tday_binding.json": b"{}\n"}, "a binding")
    expect_gate(v, "binding_already_written", "second_session", mode="session", data=data,
                detail="a binding is written once")
    for name, edit in (("session_part_null_refused_B101", lambda d: d.update(session=None)),
                       ("session_part_without_its_start_B101", lambda d: d.update(session={})),
                       ("session_start_not_utc_B101",
                        lambda d: d["session"].update(started_utc="2026-11-05 02:00:00"))):
        w = Unlock()
        expect_gate(w, name, "clock_offset_invalid", mode="session", data=w.start_session(edit=edit), detail="session")


def case_coverage():
    want = [c for c in gate_codes("session") if c not in gate_codes("freeze")]
    missing = [c for c in want if c not in WANTED]
    check("every_unlock_and_session_gate_has_a_case", not missing, str(missing))


UW: Unlock | None = None


def main() -> int:
    global UW
    try:
        UW = Unlock()
    except Exception as e:  # noqa: BLE001
        check("unlock_world_built", False, f"{type(e).__name__}: {e}")
        return FX.run_cases("TMAN UNLOCK SMOKE (lane TMAN, B-1, C5)", ())
    try:
        return FX.run_cases("TMAN UNLOCK SMOKE (lane TMAN, B-1, C5)",
                            (case_unlock_world, case_cli, case_unlock_printer, case_unlock_gates, case_session,
                             case_coverage))
    finally:
        if VW.REAL_TEACHER is not None:
            UW.w.M.TEACHER_OF_RECORD.clear()
            UW.w.M.TEACHER_OF_RECORD.update(VW.REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(main())
