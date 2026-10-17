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
                    repo_root=self.wt)
        args.update(kw)
        return V.Ctx(**args)

    def run(self, mode="unlock", data=None, **kw):
        from src.tman import validate as V
        try:
            return None, V.validate(self.ctx(mode, data, **kw)).passed, None
        except self.S.TmanRefused as e:
            return e.code, getattr(e, "passed", None), e

    def start_session(self, offset: float = 0.4, push_utc: str = "2026-11-04T09:00:00Z") -> bytes:
        """The session part: the clock-offset record and the push evidence of the unlock entry's commit."""
        S, w = self.S, self.w
        clock = w.entry_file("clock_offset", [], {"host": "tday-host", "source": "chronyc tracking",
                                                  "offset_seconds": offset, "measured_utc": "2026-11-05T01:50:00Z"},
                             date="2026-11-05", nn=_nn(w, "clock_offset"))
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
                          "started_utc": SESSION_UTC, "host": "tday-host", "values": {},
                          "push_evidence": {"response": {"path": rr, "sha256": S.sha256_bytes(resp)},
                                            "list": {"path": lr, "sha256": S.sha256_bytes(lst)}}}
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
        check("cli_unlock_at_S_with_the_records_folder_passes", rc == 0 and
              f"PASS: mode unlock; {len(gate_codes('unlock'))} gates passed" in out, out.strip()[-200:])
        rc, out = cli(argv)
        check("cli_unlock_without_the_records_folder_refused", rc == 2 and "record_uncommitted" in out
              and "--records" in out, out.strip()[-200:])
        rc, out = cli(argv + ["--records", str(folder)])
        check("cli_records_without_its_commit_refused", rc == 2 and "go together" in out, out.strip()[-200:])
    finally:
        R.ptq_record_reader = saved


def case_unlock_gates():
    S, w = UW.S, UW.w
    u = UW
    d = copy.deepcopy(u.doc)
    w.entry(d, "E4:s42:list45")["hashed"] = None
    d["appends"] = [{"slot": "E4:s42:list45", "hashed": dict(w.entry(u.doc, "E4:s42:list45")["hashed"],
                                                              appended="append"), "ptq_record": None}]
    data = u.commit_manifest(d)
    code, _, e = u.run("freeze", data)
    check("the_append_itself_passes_freeze_mode", code is None, f"{code}: {e.detail[:200] if e else ''}")
    expect_gate(u, "appends_changed_after_the_unlock_record", "append_after_unlock", data=data,
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
    u = Unlock()
    data = u.start_session()
    code, passed, e = u.run("session", data)
    check("session_world_passes_every_session_gate", code is None and passed == gate_codes("session"),
          f"{code}: {e.detail[:300] if e else ''}")
    code, _, e = u.run("unlock", data)
    check("unlock_mode_refuses_a_session_part", code == "test_value_before_unlock", f"{code}")
    late = copy.deepcopy(json.loads(data))
    late["session"]["started_utc"] = "2026-11-08T16:00:01Z"
    expect_gate(u, "session_started_after_the_deadline_TB14", "session_after_deadline", mode="session",
                data=u.commit_manifest(late), detail="after the TEST deadline")
    data = u.start_session(offset=75.0)
    expect_gate(u, "clock_offset_over_60_s_AM19a_r21", "clock_offset_invalid", mode="session", data=data,
                detail="within 60 s")
    data = u.start_session(push_utc="2026-11-05T03:00:00Z")
    expect_gate(u, "unlock_pushed_after_the_session_started_TB11", "push_evidence_invalid", mode="session",
                data=data, detail="no listed push carries the unlock entry")
    data = u.start_session()
    u.w.commit({"reports/derived/tday_binding.json": b"{}\n"}, "a binding")
    expect_gate(u, "binding_already_written", "second_session", mode="session", data=data,
                detail="a binding is written once")


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
                            (case_unlock_world, case_cli, case_unlock_gates, case_session, case_coverage))
    finally:
        if VW.REAL_TEACHER is not None:
            UW.w.M.TEACHER_OF_RECORD.clear()
            UW.w.M.TEACHER_OF_RECORD.update(VW.REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(main())
