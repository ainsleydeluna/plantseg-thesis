#!/usr/bin/env python3
"""Smoke for AM-22 item 1(a)'s check and its freeze (src/tman/am22.py; lane TMAN, B-2): the registered runs and every
exclusion of B2R3, every settled way with a never-cut run's limits, the records committed by the end of a date (C_X,
audit M3), the five conditions MET and MISSING with their REPORT lines (the boundary day, audit M2; a late trigger),
condition 5 with AM-22's candidate freeze and the ARM measurement's freeze cut (TMAN-B2-RULE-3), D the first such date
(audit m4), AM-22's provider in front of B-1's (B2R6; audit m10), the freeze entry of item 1(b) and its printer `entry
freeze --d-date` (B2R5, B2R7; audit m8, and m9 as TMAN-B2-RULE-3 amends it), the unlock gates of B2R7 with the freeze
record's push evidence (audit M4), `check-freeze`, `build` and `validate` on the command line (a date whose end has not
passed, audit m6; a date after 2026-11-02, item 1(f)), and the static cases of B2R6, B2R11 and B2R13.

The fixture world (`AmWorld`) is smoke_tman_validate's World on AM-22's timeline: its runs of record complete by
2026-10-20, every other cuttable registered run cut by item 2(a)'s cutoff -- on 2026-10-20, entered the next day, R2
at its ceiling 2026-10-29, entered on 2026-10-30 -- with its children not run, AM-3's trigger entered on 2026-10-22
(not triggered), and the lane register and the draft, built with AM-22's candidate freeze for 2026-10-30, committed on
2026-10-30: D is 2026-10-30. `AmUnlock` carries it to the unlock, the freeze at the end of 2026-10-31; `Five` is the
world whose D is 2026-10-31 (its draft of 2026-10-30 built without the candidate). A world is built when a case first
reads it (smoke_tman_mutations runs a case alone). Smaller fixtures (`Mini`) put one question each: a settled way, an
exclusion, a condition's record, a provider.

Synthetic inputs only, under the system temp dir (removed at the end); TEST is never read.

Run:  python -B scripts/smoke_tman_am22.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import contextlib  # noqa: E402
import copy  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_pins as PINS  # noqa: E402
import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)
import scripts.smoke_tman_unlock as UL  # noqa: E402  (its records folder's blob reader)
import scripts.smoke_tman_validate as VW  # noqa: E402  (the fixture world)

check, expect_refused = FX.check, FX.expect_refused

# AM-22's timeline in the fixture (UTC; each instant before 16:00, so its Manila day is its UTC day, but T_NR)
T_CODE = "2026-10-12T02:00:00+00:00"          # the code of record E with AM-22's text (it stands in for ce29549)
KD_END = "2026-10-12T10:00:00Z"               # the sweeps' runs end
T_SEL = "2026-10-14T02:00:00+00:00"           # the selections committed: the alpha cutoff's moving record, 10-14
PILOT = ("2026-10-15T00:00:00Z", "2026-10-16T12:00:00Z")      # the U4 pilot's runs: launched, ended
PILOT_MADE = "2026-10-17T02:00:00Z"
T_PILOT = "2026-10-18T02:00:00+00:00"         # the pilot's winner record entered (DL-139's form)
E6RUN = ("2026-10-19T00:00:00Z", "2026-10-20T12:00:00Z")      # E6 seed 42: launched, ended (Manila 2026-10-20)
E6_MADE = "2026-10-21T02:00:00Z"              # its conversion and selection
T_CUTS = "2026-10-21T02:00:00+00:00"          # item 2(a)'s cuts of 2026-10-20 entered, their children not run
T_TRIGGER = "2026-10-22T02:00:00+00:00"       # AM-3's trigger entered (due by the end of 2026-10-22)
PTQ_MADE = "2026-10-25T02:00:00Z"
T_NR = "2026-10-28T16:30:00+00:00"            # the lanes' not-required entries: Manila 00:30 on 2026-10-29 (X - 1)
T_R2 = "2026-10-30T01:00:00+00:00"            # R2's cut at its ceiling entered, the lane register
T_DRAFT = "2026-10-30T03:00:00+00:00"         # the draft
WORLD_PTQ_MADE = "2026-11-02T10:00:00Z"       # smoke_tman_validate.World.ptq's time, moved to PTQ_MADE here
NOW = "2026-11-06T00:00:00Z"                  # --now
D_WORLD, F_WORLD = "2026-10-30", "2026-10-31"  # the world's D and its freeze date, D + 1 (item 1(b))
T_ARM = "2026-10-31T01:00:00+00:00"           # ARM latency cut at the freeze (item 1(g)); its lane entered not required
T_BODY = "2026-10-31T02:00:00+00:00"          # the body to freeze committed: the draft and that cut (TMAN-B2-RULE-3)
NOW_FREEZE = "2026-10-31T02:30:00Z"           # --now of `entry freeze --d-date 2026-10-30`, after the end of D
T_FREEZE = "2026-10-31T03:00:00+00:00"        # the freeze record entered
PUSHED = "2026-10-31T04:00:00Z"               # its commit reaches the remote, before the end of its date (audit M4)
T_LO_FREEZE = "2026-10-30T16:00:00Z"          # the first instant of the freeze record's date (Manila)
T_EVIDENCE = "2026-11-01T02:00:00+00:00"      # the push evidence committed
UNLOCK_DAY, T_UNLOCK = "2026-11-01", "2026-11-01T03:00:00+00:00"
MDE_REL = "reports/derived/mde_entry_20261009T034825Z.json"
CUTOFF_GROUPS = {   # group -> (slots, reason, arm) of item 2(a)'s cuts on 2026-10-20
    "E2-s4344": ["E2:s43:lambda0.5:80k", "E2:s44:lambda0.5:80k"],
    "E3-s4344": ["E3:s43:lambda0.5:alpha25:80k", "E3:s44:lambda0.5:alpha25:80k"],
    "E5E6-s4344": ["E5:s43:15ep", "E5:s44:15ep"], "E1-160k": ["E1:s42:160k"], "E2-160k": ["E2:s42:lambda0.5:160k"],
    "E3-160k": ["E3:s42:lambda0.5:alpha25:160k"], "A": ["A:s42:alpha25:80k"], "F": ["F:s42:alpha25:80k"],
    "G": ["G:s42:80k"], "R1": ["teacher-R1:s42:40k", "E3-R1:s42:lambda0.5:alpha25:80k"],
    "W": ["E3-flip:s42:lambda0.5:alpha25:80k"],
    "L": ["teacher-L:s42:40k", "E2-L:s42:lambda0.5:80k", "E3-L:s42:lambda0.5:alpha25:80k"],
    "E6-E2": ["E6-E2:s42:15ep"]}
REASONS = {"L": "code not merged", "E6-E2": "code not merged", "G": "no card"}
NO_CARD_SINCE = "configs/plantseg_class_map.json"      # G's "no card" since-record: no item 2(f) evidence (B2R8)
ARMS = ("R1", "R2", "W", "L")
R2 = ["teacher-R2:s42:40k", "E3-R2:s42:lambda0.5:alpha25:80k"]
NOT_REQUIRED = (("L-AM23Q", "E6-E2:s42:15ep"), ("L-LAT-THREADS", "E1:s42:160k"))     # lane -> the slot it serves
CLOCK = {"host": "tday-host", "source": "chronyc tracking", "offset_seconds": 0.4,
         "measured_utc": "2026-11-05T01:50:00Z"}


def mods():
    from src.tman import am22 as A
    from src.tman import completion as C
    from src.tman import entries as E
    from src.tman import manifest as M
    from src.tman import records as R
    from src.tman import schedule as D
    from src.tman import schema as S
    return A, C, D, E, M, R, S


def catalogue():
    _, _, _, _, _, _, S = mods()
    return S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())


# ============================================================================================ worlds and stand-ins
BUILT: list = []            # every world built in this process; lane LQP's reader stand-in reads their PTQ records
_CELLS: list = []


def built(cls):
    """The world of class `cls`, built when a case first reads it."""
    w = next((w for w in BUILT if type(w) is cls), None)
    return w if w is not None else cls()


def world() -> "AmWorld":
    return built(AmWorld)


def five() -> "Five":
    return built(Five)


def unlocked() -> "AmUnlock":
    return built(AmUnlock)


def read_ptq(path, expect_sha256=None):
    """Lane LQP's reader stand-in (smoke_tman_validate.World.reader) over the worlds built here."""
    w = next((w for w in BUILT if getattr(w, "runs", None) and str(path).startswith(str(w.runs) + os.sep)), None)
    if w is None:
        raise OSError(f"{path}: under no world's run directories")
    return w.reader()(path, expect_sha256)


@contextlib.contextmanager
def stand_ins(w):
    """For the cases that run the validator through am22 or the command line: records.ptq_record_reader (lane LQP's
    reader: read_ptq), records.TRIGGER_READER (the fixture's am3_trigger/1 reader, smoke_tman_validate.World's) and
    schedule.AM22_COMMIT (the world's E, which holds AM-22's text: item 2(a)'s dates of record, B2R1) patched, and
    restored after."""
    A, C, D, E, M, R, S = mods()
    saved = R.ptq_record_reader, R.TRIGGER_READER, D.AM22_COMMIT
    R.ptq_record_reader, R.TRIGGER_READER, D.AM22_COMMIT = (lambda: read_ptq), VW.World.trigger_reader(), w.E
    try:
        yield
    finally:
        R.ptq_record_reader, R.TRIGGER_READER, D.AM22_COMMIT = saved


def scratch_file(prefix: str, name: str, data: bytes) -> Path:
    f = FX.scratch_dir(prefix) / name
    f.write_bytes(data)
    return f


def cells_file() -> Path:
    """smoke_tman_pins's IF6 cells as a --registered-cells file: the order the worlds' pins hold (audit M5)."""
    if not _CELLS:
        _CELLS.append(scratch_file("tmancells", "cells.json", json.dumps(PINS.IF6_CELLS).encode()))
    return _CELLS[0]


def cli(CLI, argv: list) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CLI.main([str(a) for a in argv])
    return rc, buf.getvalue()


class PrinterRefused(RuntimeError):
    """A world's printer refused or failed: case_unlock_world names which (DL-24 R3-04)."""

    def __init__(self, printer: str, rc, out: str):
        super().__init__(f"{printer}: rc {rc}: {out.strip()[-400:]}")
        self.printer = printer


# ================================================================================================ the AM-22 world
class AmWorld(VW.World):
    """smoke_tman_validate's World on AM-22's timeline (module docstring)."""
    DRAFTS = ((D_WORLD, T_DRAFT),)      # each draft: (the date whose candidate freeze it is built with, or None; when)

    def __init__(self):
        A, C, D, E, M, R, S = mods()
        BUILT.append(self)
        self.S, self.M, self.R = S, M, R
        self.folders, self.wt = {}, None             # the records folders by commit; S (checkout)
        self.repo, self.runs = FX.new_repo(), FX.scratch_dir("tmanruns")
        self.cat = S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())
        self.dl, self.next_dl, self.files = {}, 200, {}
        self.entries, self.ptq_docs, self.qat, self.qat_records, self.launch, self.cut_refs = {}, {}, {}, [], [], {}
        files = {rel: (REPO / rel).read_bytes() for rel in VW.REAL_FILES}
        files.update({k: v.encode() for k, v in VW.CODE_FIXTURES.items()})
        files[S.AMENDMENTS_REL] = D.am22_text((REPO / S.AMENDMENTS_REL).read_text(encoding="utf-8")).encode()
        self.E = FX.commit_files(self.repo, files, "the code of record (E) and AM-22's text", date=T_CODE)
        self.teacher_fixture()
        self.static_rows()
        self.complete_runs()
        self.derived_models()
        self.selection_files()
        self.files["reports/kd_launch_log.jsonl"] = b"".join(json.dumps(x).encode() + b"\n" for x in self.launch)
        self.H_SEL = self.commit_dl(self.files, "the selections, the launch log and the runs' rows", T_SEL)
        self.pilot_dl = self.row(f"The U4 pilot's winner record is {VW.CLIP_REL}, sha256 {self.clip['sha256']}, "
                                 "written by that block B on 2026-10-18; never committed")
        self.H_PILOT = self.commit_dl({}, "the pilot's winner record entered", T_PILOT)
        self.cutoff_entries({g: (k, "2026-10-20") for g, k in CUTOFF_GROUPS.items()}, "2026-10-21")
        self.H_CUTS = self.commit_dl(self.files, "item 2(a)'s cuts of 2026-10-20 and their children", T_CUTS)
        trig = S.json_file_bytes({"format": "am3_trigger/1", "triggered": False})
        self.trigger_dl = self.row(f"AM-3's trigger: {R.TRIGGER_RECORD_REL} sha256 {S.sha256_bytes(trig)}, not "
                                   "triggered")
        self.H_TRIGGER = self.commit_dl({R.TRIGGER_RECORD_REL: trig}, "AM-3's trigger entered", T_TRIGGER)
        self.not_required_entries()
        self.H_NR = self.commit_dl(self.files, "the lanes' not-required entries", T_NR)
        self.cutoff_entries({"R2": (R2, "2026-10-29")}, "2026-10-30")
        self.lane_register()
        self.H_R2 = self.commit_dl(self.files, "R2's cut at its ceiling; the lane register", T_R2)
        self.sources = self.make_sources()
        self.draft_docs: list = []
        self.drafts = [self.commit_draft(d, when) for d, when in self.DRAFTS]
        self.H_DRAFT = self.drafts[0]

    def commit_dl(self, files: dict, msg: str, date: str) -> str:
        """Commit `files` with the decision log as it stands; the next files start empty."""
        files = dict(files)
        files[self.S.DECISION_LOG_REL] = FX.dl_table(dict(sorted(self.dl.items(), key=lambda kv: int(
            kv[0].split("-")[1])))).encode()
        sha = FX.commit_files(self.repo, files, msg, date=date)
        self.files = {}
        self.records = self.R.Records.at_head(self.repo)
        return sha

    def commit_draft(self, d: str | None, when: str) -> str:
        """The draft built from the sources with AM-22's candidate freeze for `d` (B2R6; audit m5: the draft of each
        candidate day is built so), committed on `when`; it becomes the world's manifest."""
        D, S = mods()[2], self.S
        self.doc = self.M.build_manifest(self.records, self.cat, self.sources, runs_root=self.runs,
                                         providers=[D.am22_candidate(d)] if d else [])
        self.data = S.json_file_bytes(self.doc)
        self.draft_docs.append(self.doc)
        sha = FX.commit_files(self.repo, {S.MANIFEST_REL: self.data}, f"the draft (AM-22's candidate for {d})",
                              date=when)
        self.records = self.R.Records.at_head(self.repo)
        return sha

    def complete_runs(self):
        for slot, (kind, rid) in VW.KD_RUNS.items():
            d = FX.kd_run(self.runs, rid, stage_key=self.cat.slots[slot]["stage"].lower(), end=KD_END)
            ck, dl = self.ckpt(d, slot, "iter_80000.pth")
            self.run_entry(slot, kind, rid, ck, dl)
            self.launch_lines(slot, rid, d)
        for seed in (42, 43, 44):
            slot, rid = f"E1:s{seed}:80k", f"e1_s{seed}_80k"
            d = FX.e1_run(self.runs, rid, seed=seed, max_iters=80000)
            ck, dl = self.ckpt(d, slot, "best.pth")
            self.run_entry(slot, "scored_run", rid, ck, dl)
        t = self.run_entry("teacher:s42:40k", "scored_run", "teacher_s42_40k")
        tdata = (self.runs / "teacher_s42_40k" / "iter_24000.pth").read_bytes()
        sha = self.S.sha256_bytes(tdata)
        t.update(checkpoint={"name": "iter_24000.pth", "sha256": sha, "bytes": len(tdata),
                             "sources": [{"kind": "dl_row", "path": None, "dl": "DL-88", "sha256": sha}]},
                 teacher_ckpt_sha256=sha, dl_rows=["DL-88"])
        for slot, rid, stage, clip, (launched, end) in (("E5:s42:clip1:15ep", "e5_s42_clip1_15ep", "E5", 1.0, PILOT),
                                                        ("E5:s42:clip5:15ep", "e5_s42_clip5_15ep", "E5", 5.0, PILOT),
                                                        ("E6:s42:15ep", "e6_s42_15ep", "E6", 5.0, E6RUN)):
            FX.qat_run(self.runs, rid, stage=stage, clip_norm=clip, source_sha256=self.parent_sha256(slot),
                       launched=launched, end=end)
            self.run_entry(slot, "listed_run", rid, None, self.row(f"{slot} completes"))

    def ptq(self, slot: str, parent: str, list_seed: int, of_record: bool) -> dict:
        """World.ptq's model, made on PTQ_MADE."""
        e = super().ptq(slot, parent, list_seed, of_record)
        S, key = self.S, self.S.file_id_of(slot)
        d = self.runs / key
        prov = (d / "ptq_run_meta.json").read_bytes().replace(WORLD_PTQ_MADE.encode(), PTQ_MADE.encode())
        (d / "ptq_run_meta.json").write_bytes(prov)
        rec = S.json_file_bytes({"format": "ptq_run_record/fixture", "slot": slot, "provenance": S.sha256_bytes(prov)})
        (d / "ptq_record.json").write_bytes(rec)
        e["hashed"].update(provenance={"path": f"{key}/ptq_run_meta.json", "sha256": S.sha256_bytes(prov)},
                           record={"path": f"{key}/ptq_record.json", "sha256": S.sha256_bytes(rec)},
                           made_utc=PTQ_MADE)
        return e

    def derived_models(self):
        for seed in (42, 43, 44):
            self.ptq(f"E4:s{seed}:list42", f"E1:s{seed}:80k", 42, True)
        for lst in (43, 44, 45):
            self.ptq(f"E4:s42:list{lst}", "E1:s42:80k", lst, False)
            self.ptq(f"E7:s42:list{lst}", VW.E3S42, lst, False)
        self.ptq("E7:s42:list42", VW.E3S42, 42, True)
        for rid, stage, clip, made in (("e5_s42_clip1_15ep", "E5", 1.0, PILOT_MADE),
                                       ("e5_s42_clip5_15ep", "E5", 5.0, PILOT_MADE),
                                       ("e6_s42_15ep", "E6", 5.0, E6_MADE)):
            self.qat_files(rid, stage, clip, made=made)
        self.clip_selection()
        self.qat_selection("E5:s42:sel", "E5:s42:clip5:15ep")
        self.qat_selection("E6:s42:sel", "E6:s42:15ep")

    def cutoff_entries(self, groups: dict, children_day: str):
        """Item 2(a)'s cut of each group {group: (slots, its cutoff date)} (one entry each: an arm's is its "not
        launched" record with its known results), its children's not-run entries and, for E6-E2, E7-E2 not made,
        entered on `children_day`."""
        mde = {"path": MDE_REL, "sha256": self.S.sha256_bytes((REPO / MDE_REL).read_bytes())}
        card = {"path": NO_CARD_SINCE, "sha256": self.S.sha256_bytes(VW.CODE_FIXTURES[NO_CARD_SINCE].encode())}
        for group, (slots, day) in groups.items():
            since = card if REASONS.get(group) == "no card" else mde
            fields = {"rule": "cutoff", "reason": REASONS.get(group, "gate not met"), "launched": False, "run_ids": [],
                      "since": {"date": "2026-10-20", "record": dict(since)}, "arithmetic": None, "telemetry": [],
                      "comparator": None, "results_known": [dict(mde, dl="DL-99")] if group in ARMS else [],
                      "cut_order_position": None, "rule_record": None}
            ref = self.entry_file("cut", slots, fields, date=day)
            for k in slots:
                self.cut_refs[k] = ref
                self.entries[k] = {"kind": "cut", "dl_rows": [ref["dl"]], "entry": {"path": ref["path"],
                                                                                    "sha256": ref["sha256"]},
                                   "cut_date": day, "rule": "cutoff", "comparator": None}
        for parent, slots in VW.NOT_RUN.items():
            if not any(parent in s for s, _ in groups.values()):
                continue
            pref = {"path": self.cut_refs[parent]["path"], "sha256": self.cut_refs[parent]["sha256"]}
            ref = self.entry_file("not_run", slots, {"reason": "parent cut", "parent": parent, "parent_record": pref},
                                  date=children_day)
            for k in slots:
                self.entries[k] = {"kind": "not_run", "dl_rows": [ref["dl"]], "reason": "parent cut",
                                   "parent": parent, "parent_record": pref,
                                   "entry": {"path": ref["path"], "sha256": ref["sha256"]}}
        if "E6-E2" in groups:
            ref = self.entry_file("not_made", ["E7-E2:s42:list42"], {"reason": "code not merged", "lane": "L-AM23Q"},
                                  date=children_day)
            self.entries["E7-E2:s42:list42"] = {"kind": "not_made", "dl_rows": [ref["dl"]],
                                                "reason": "code not merged",
                                                "entry": {"path": ref["path"], "sha256": ref["sha256"]}}

    def not_required_entries(self):
        """Two lanes entered as not required, each entry citing the cut of the slot it serves: E6-E2's, its code not
        merged, so E7-E2 is not made; E1-160k's (committed at T_NR, Manila 2026-10-29)."""
        self.nr = {}
        for lane, slot in NOT_REQUIRED:
            cut = self.cut_refs[slot]
            self.nr[lane] = self.entry_file("not_required", [slot], {"lane": lane, "cut_entries": [
                {"path": cut["path"], "sha256": cut["sha256"]}]}, date="2026-10-29")

    def lane_register(self):
        """The lane register: L-AM3 IMPLEMENTED (its row naming it and its merge commit E, an ancestor of S = E), the
        two lanes entered as not required and L-AM16-ARM with no status a reader could accept (item 1(g): condition 4
        does not read it)."""
        S, nr = self.S, self.nr
        self.am3_dl = self.row(f"L-AM3 merged at {self.E[:7]}: the AM-3 trigger lane", "IMPLEMENTED (2026-10-25)")
        self.lanes = {"format": S.LANES_FORMAT, "lanes": [
            {"lane": "L-AM3", "serves": ["E6-KD:s42:15ep", "E6-KD:s42:sel"],
             "status": {"kind": "implemented", "dl": self.am3_dl, "merge_commit": self.E, "entry": None}},
            *[{"lane": lane, "serves": [slot], "status": {"kind": "not_required", "dl": None, "merge_commit": None,
                                                          "entry": {"path": nr[lane]["path"],
                                                                    "sha256": nr[lane]["sha256"]}}}
              for lane, slot in NOT_REQUIRED],
            {"lane": "L-AM16-ARM", "serves": ["ARM:latency"],
             "status": {"kind": "implemented", "dl": None, "merge_commit": None, "entry": None}}]}
        self.commit_lanes()

    def commit_lanes(self):
        """The lane register as it stands, with its RECORDED row (the current-plan line), into the next commit."""
        S = self.S
        self.lanes_data = S.json_file_bytes(self.lanes)
        self.files[S.LANES_REL] = self.lanes_data
        self.lanes_dl = self.row(f"the lane register {S.LANES_REL} sha256 {S.sha256_bytes(self.lanes_data)}")

    def make_sources(self) -> dict:
        """World.make_sources with the trigger record at L-AM3's path (records.TRIGGER_RECORD_REL)."""
        S, R = self.S, self.R
        blob = FX.git(self.repo, "rev-parse", f"{self.E}:scripts/build_corruption_cache.py")
        pin = PINS.pin(script={"path": "scripts/build_corruption_cache.py", "git_blob": blob}, commit=self.E)
        hasher = {"script": {"path": "scripts/tday_hash_images.py", "commit": self.E,
                             "sha256": S.sha256_bytes(VW.CODE_FIXTURES["scripts/tday_hash_images.py"].encode())},
                  "python": "3.11", "parameters": {"folder": "images/<split>", "suffixes": [".jpg", ".jpeg"]}}
        trig = (self.repo / R.TRIGGER_RECORD_REL).read_bytes()
        return {"evaluator_commit": self.E,
                "execution_context": {"model_device": "cpu", "input_devices": ["cpu"],
                                      "determinism_policy_applied": False, "batch_size": 1, "num_workers": 0,
                                      "torch_num_threads": 4, "image_digest": S.KD_IMAGE, "evaluator_commit": self.E,
                                      "host": {"name": "tday-host", "cpu_model": "x86-64 fixture"}, "processes": 2},
                "generator_pins": {"corrupted_cache": pin, "image_hasher": hasher},
                "trigger": {"path": R.TRIGGER_RECORD_REL, "sha256": S.sha256_bytes(trig), "triggered": False},
                "freeze_entries": {"delay": None, "new_freeze": None}, "entries": json.loads(json.dumps(self.entries))}

    def evaluate(self, day: str, **kw):
        """The five conditions at the end of `day` (am22.evaluate), the registered cells in smoke_tman_pins's IF6
        order (the world's pins; audit M5)."""
        A = mods()[0]
        kw.setdefault("registered_cells", PINS.IF6_CELLS)
        return A.evaluate(self.records, self.cat, self.runs, day, reader=self.trigger_reader(), **kw)

    def check_freeze(self, day: str, now: str = NOW, pushed: dict | None = None) -> tuple:
        A = mods()[0]
        return A.check_freeze(self.records, self.cat, self.runs, day, now=FX.utc(now), reader=self.trigger_reader(),
                              registered_cells=PINS.IF6_CELLS, pushed=pushed)

    def print_entry(self, kind: str, date: str, fields: dict, *, nn: int, extra=(), root=None) -> tuple:
        """The CLI's `entry` printer at --root (this world's repository by default): (its exit code, its output, the
        file it wrote or None). It commits nothing."""
        import scripts.tday_manifest as CLI
        out = FX.scratch_dir("tmanprint")
        (out / "fields.json").write_bytes(self.S.json_file_bytes(fields))
        dest = out / "out"
        dest.mkdir()
        rc, txt = cli(CLI, ["--root", root or self.repo, "entry", kind, "--date", date, "--fields", out / "fields.json",
                            "--amendment", "AM-22", "--summary", "printed by the AM-22 smoke.", "--out", dest,
                            "--nn", nn, *extra])
        made = sorted(dest.iterdir())
        return rc, txt, (made[0] if len(made) == 1 else None)

    def commit_entry(self, path: Path, when: str) -> dict:
        """An entry file committed at its records path with its RECORDED row on `when`: {path, sha256, dl, doc,
        commit}."""
        S = self.S
        data = path.read_bytes()
        rel, sha, doc = S.entry_file_rel(path.name), S.sha256_bytes(data), json.loads(data)
        dl = self.row(f"TMAN {doc['kind']} entry for the session: {rel} sha256 {sha}")
        return {"path": rel, "sha256": sha, "dl": dl, "doc": doc,
                "commit": self.commit_dl({rel: data}, f"the {doc['kind']} entry", when)}

    def freeze_args(self, d: str, man: Path, now: str = NOW_FREEZE, *, runs: bool = True) -> list:
        """`entry freeze --d-date d`'s arguments: the body to freeze `man`, the run directories, the registered cells
        and the clock's stand-in."""
        return ["--d-date", d, "--manifest", man, *(["--runs", self.runs] if runs else []), "--registered-cells",
                cells_file(), "--now", now]

    def worktree(self, commit: str) -> Path:
        wt = FX.scratch_dir("tmanwt")
        wt.rmdir()
        FX.git(self.repo, "worktree", "add", "-q", "--detach", str(wt), commit)
        return wt

    def checkout(self) -> Path:
        """S, a linked checkout of E (TB16), made when first read."""
        if self.wt is None:
            self.wt = self.worktree(self.E)
        return self.wt

    def records_at(self, head: str):
        """The records at `head`, from a records folder written from it (reports/ and the decision log), read with S
        = E (smoke_tman_unlock.Unlock.export; PL-28)."""
        if head not in self.folders:
            folder = FX.scratch_dir("tmanrec")
            for rel in FX.git(self.repo, "ls-tree", "-r", "--name-only", head, "--", "reports",
                              self.S.DECISION_LOG_REL).splitlines():
                (folder / rel).parent.mkdir(parents=True, exist_ok=True)
                (folder / rel).write_bytes(UL._blob(self.repo, head, rel))
            self.folders[head] = folder
        return self.R.Records.at_records_commit(self.folders[head], head, root=self.checkout())


class Five(AmWorld):
    """The AM-22 world whose D is 2026-10-31 (audit m4): its draft of 2026-10-30 is built without AM-22's candidate --
    condition 5 MISSING at that date's end (schedule_invalid) -- and the drafts of 2026-10-31 and 2026-11-02 with
    their dates' candidates; all five conditions hold at the end of 2026-10-31 and again at the end of 2026-11-02."""
    DRAFTS = ((None, T_DRAFT), ("2026-10-31", "2026-10-31T03:00:00+00:00"),
              ("2026-11-02", "2026-11-02T03:00:00+00:00"))


class AmUnlock(AmWorld):
    """The AM-22 world to the unlock: D 2026-10-30, the freeze at the end of 2026-10-31 (item 1(b)). On the freeze date
    the ARM latency measurement, not complete, is cut at the freeze (item 1(g); AM-19 item 3(b), first point) and its
    lane entered as not required; the body to freeze -- the draft at C_D and that cut (TMAN-B2-RULE-3) -- is committed
    as the manifest, and the freeze record `entry freeze --d-date` prints is entered; its commit reaches the remote that
    day. The push evidence is committed on 2026-11-01 and the unlock entry `entry unlock --push-evidence` prints cites
    it (audit M4). S is a linked checkout of E (TB16) and `tree` one of the unlock entry's commit, where the printers'
    cases run; the validator reads the records from a records folder written from a commit (PL-28)."""

    def __init__(self):
        super().__init__()
        A, C, D, E, M, R, S = mods()
        self.ticks, self.wt_body = 0, None
        self.draft_doc, self.draft_data = self.doc, self.data
        with stand_ins(self):
            arm = self.printed_file("cut", F_WORLD, self.cut_fields("ARM", ["ARM:latency"], "code not merged", False),
                                    ["--slot", "ARM:latency", "--d-date", D_WORLD, "--runs", self.runs])
            nr = self.entry_file("not_required", ["ARM:latency"], {"lane": A.ARM_LANE, "cut_entries": [
                {"path": arm["path"], "sha256": arm["sha256"]}]}, date=F_WORLD)
            self.lanes["lanes"][-1]["status"] = {"kind": "not_required", "dl": None, "merge_commit": None,
                                                 "entry": {"path": nr["path"], "sha256": nr["sha256"]}}
            self.commit_lanes()
            self.arm = arm
            self.H_ARM = self.commit_dl(self.files, "the ARM latency measurement cut at the freeze; its lane", T_ARM)
            self.man = self.freeze_body([arm["path"]])
            self.data = self.man.read_bytes()
            self.doc = json.loads(self.data)
            self.H_BODY = FX.commit_files(self.repo, {S.MANIFEST_REL: self.data}, "the body to freeze", date=T_BODY)
            self.records = R.Records.at_head(self.repo)
            self.freeze = self.commit_entry(self.printed("freeze", F_WORLD, {}, self.freeze_args(D_WORLD, self.man)),
                                            T_FREEZE)
            self.evidence = self.push_evidence(PUSHED)
            self.pe = scratch_file("tmanpe", "pe.json", S.json_file_bytes(self.evidence))
            self.unlock_fields = {
                "manifest": {"path": S.MANIFEST_REL, "sha256": S.sha256_bytes(self.data),
                             "body_sha256": S.canonical_sha256(self.doc["body"]),
                             "appends_sha256": S.canonical_sha256([])},
                "lanes": {"path": S.LANES_REL, "sha256": S.sha256_bytes(self.lanes_data)},
                "freeze_record": {"path": self.freeze["path"], "sha256": self.freeze["sha256"]}}
            self.unlock = self.commit_entry(self.printed("unlock", UNLOCK_DAY, self.unlock_fields,
                                                         ["--push-evidence", self.pe]), T_UNLOCK)
        self.H_UNLOCK = self.unlock["commit"]
        self.tree = self.worktree(self.H_UNLOCK)

    def printed(self, kind: str, date: str, fields: dict, extra: list) -> Path:
        """The entry file the CLI's printer writes for this world's records (a refusal: PrinterRefused)."""
        rc, out, made = self.print_entry(kind, date, fields, nn=1, extra=extra)
        if made is None:
            raise PrinterRefused(f"entry {kind}", rc, out)
        return made

    def printed_file(self, kind: str, date: str, fields: dict, extra: list) -> dict:
        """An entry file the CLI's printer writes, put at its records path with its RECORDED row into the next commit
        (World.entry_file's dict)."""
        S, path = self.S, self.printed(kind, date, fields, extra)
        data = path.read_bytes()
        rel, sha = S.entry_file_rel(path.name), S.sha256_bytes(data)
        self.files[rel] = data
        return {"path": rel, "sha256": sha, "dl": self.row(f"TMAN {kind} entry: {rel} sha256 {sha}"),
                "doc": json.loads(data)}

    def freeze_body(self, cuts: list) -> Path:
        """`freeze-body --d-date D --cut ...` at this world's records: the body to freeze it writes (DL-24 R2-8)."""
        import scripts.tday_manifest as CLI
        out = FX.scratch_dir("tmanfbody")
        rc, txt = cli(CLI, ["--root", self.repo, "freeze-body", "--d-date", D_WORLD,
                            *[a for c in cuts for a in ("--cut", c)], "--out", out])
        made = out / "tday_manifest.json"
        if rc != 0 or not made.is_file():
            raise PrinterRefused("freeze-body", rc, txt)
        return made

    def body_tree(self) -> Path:
        """A linked checkout of the body's commit, before the freeze record is entered: the freeze printer's cases run
        there (a second item 1(b) freeze is refused, DL-24 R3-14)."""
        if self.wt_body is None:
            self.wt_body = self.worktree(self.H_BODY)
        return self.wt_body

    def push_evidence(self, pushed: str, *, tag: str = "20261031", when: str = T_EVIDENCE,
                      older: str | None = "2026-10-30T10:00:00Z", key: str = "entry") -> dict:
        """The freeze record's push evidence (audit M4; TB11's form, smoke_tman_unlock.activity and push_list):
        GitHub's activity response, newest first, the record's commit pushed to master and to the working branch at
        `pushed` and, unless `older` is None, E pushed at `older`, before T_lo, the first instant of the record's date;
        the tday_push_list/1 list made from it names the record under `key` ("entry"). Both committed in one commit
        on `when`; returns {response, list}."""
        S, c = self.S, self.freeze["commit"]
        prior = FX.git(self.repo, "rev-parse", f"{c}^")

        def act(n, ref, after, before, ts):
            return {"id": n, "node_id": f"PA_fixture{n}", "before": before, "after": after, "ref": ref,
                    "timestamp": ts, "activity_type": "push", "actor": {"login": "fixture"}}
        acts = [act(5004, "refs/heads/master", c, prior, pushed),
                act(5003, "refs/heads/claude/keen-curie-u4a8ig", c, prior, pushed)]
        acts += [act(5001, "refs/heads/master", self.E, "0" * 40, older)] if older else []
        rr, lr = f"reports/derived/tday_freeze_push_{tag}.json", f"reports/derived/tday_freeze_push_list_{tag}.json"
        resp = (json.dumps(acts, indent=2) + "\n").encode()
        lst = S.json_file_bytes({
            "format": S.PUSH_LIST_FORMAT, "response": {"path": rr, "sha256": S.sha256_bytes(resp)},
            key: {"path": self.freeze["path"], "sha256": self.freeze["sha256"]}, "adding_commit": c,
            "t_lo_utc": T_LO_FREEZE, "remote": "https://github.com/fixture/plantseg-thesis",
            "pushes": [{**{k: a[k] for k in ("id", "ref", "before", "after", "timestamp", "activity_type")},
                        "served": True, "carries_file": True, "remote_answer": "fetched (fixture)"}
                       for a in acts if a["timestamp"] >= T_LO_FREEZE]})
        FX.commit_files(self.repo, {rr: resp, lr: lst}, "the freeze record's push evidence", date=when)
        return {"response": {"path": rr, "sha256": S.sha256_bytes(resp)},
                "list": {"path": lr, "sha256": S.sha256_bytes(lst)}}

    def tick(self) -> str:
        """The next commit time after the unlock entry's, on 2026-11-01 (a case's variants)."""
        self.ticks += 1
        return f"2026-11-01T{4 + self.ticks:02d}:00:00+00:00"

    def variant(self, kind: str, edit, nn: int, *, split: bool = False) -> dict:
        """The world's freeze record or unlock entry with `edit` applied, committed with its row after the unlock
        entry's commit (commit_entry's dict); `split`, the file committed first and its row in the next commit."""
        S = self.S
        doc = copy.deepcopy((self.freeze if kind == "freeze" else self.unlock)["doc"])
        edit(doc)
        f = FX.scratch_dir("tmanvar") / S.entry_file_name(kind, S.SESSION_FILE_ID, doc["date"], nn)
        f.write_bytes(S.json_file_bytes(doc))
        if not split:
            return self.commit_entry(f, self.tick())
        data = f.read_bytes()
        rel, sha = S.entry_file_rel(f.name), S.sha256_bytes(data)
        added = FX.commit_files(self.repo, {rel: data}, f"the {kind} entry, its row not yet", date=self.tick())
        dl = self.row(f"TMAN {kind} entry for the session: {rel} sha256 {sha}")
        return {"path": rel, "sha256": sha, "dl": dl, "doc": doc, "added": added,
                "commit": self.commit_dl({}, f"the {kind} entry's row", self.tick())}

    def ctx(self, mode: str = "unlock", data: bytes | None = None, *, head: str | None = None, **kw):
        from src.tman import validate as V
        args = dict(mode=mode, manifest_bytes=self.data if data is None else data, manifest_path=self.S.MANIFEST_REL,
                    records=self.records_at(head or self.H_UNLOCK), runs_root=self.runs,
                    freeze_record=self.freeze["path"], unlock_entry=self.unlock["path"],
                    registered_cells=PINS.IF6_CELLS, ptq_reader=self.reader(), trigger_reader=self.trigger_reader(),
                    repo_root=self.checkout())
        args.update(kw)
        return V.Ctx(**args)


# ======================================================================================== a fixture per question
class Mini:
    """A records repository and run directories for one question: decision-log rows, a draft whose body holds the
    entries given, entry files with their rows and the KD launch log, each committed on the day given."""

    def __init__(self):
        A, C, D, E, M, R, S = mods()
        self.repo, self.runs, self.cat = FX.new_repo(), FX.scratch_dir("tmanruns"), catalogue()
        self.dl, self.n, self.launch, self.draft = {}, 300, [], None        # draft: None until one is set
        self.commit({}, "2026-10-12T02:00:00+00:00")

    def row(self, text: str, status: str = "RECORDED") -> str:
        dl = f"DL-{self.n}"
        self.n += 1
        self.dl[dl] = (status, text)
        return dl

    def entry(self, kind: str, slots: list, fields: dict, day: str, nn: int = 1) -> dict:
        """An entry file and its RECORDED row (not committed yet): {path, sha256, data, dl}."""
        _, _, _, E, _, _, S = mods()
        doc = E.make_entry(kind, slots, day, fields, amendment="AM-22", script="scripts/tday_manifest.py", commit=None)
        rel = S.entry_file_rel(E.entry_name(doc, nn))
        data = S.json_file_bytes(doc)
        sha = S.sha256_bytes(data)
        return {"path": rel, "sha256": sha, "data": data, "dl": self.row(f"TMAN {kind} entry: {rel} sha256 {sha}")}

    def cut(self, slots: list, rule: str, day: str, **kw) -> dict:
        mde = {"path": MDE_REL, "sha256": "4" * 64}
        fields = {"rule": rule, "reason": "gate not met", "launched": False, "run_ids": [],
                  "since": {"date": day, "record": mde}, "arithmetic": None, "telemetry": [], "comparator": None,
                  "results_known": [], "cut_order_position": None, "rule_record": None}
        fields.update(kw)
        return self.entry("cut", slots, fields, day)

    def kd(self, slot: str, rid: str, **kw):
        """A KD run of `slot` with its launch lines."""
        s = self.cat.slots[slot]
        d = FX.kd_run(self.runs, rid, stage_key=s["stage"].split("-")[0].lower(), **kw)
        lam, alpha = s["params"].get("lambda"), s["params"].get("alpha")
        self.launch.append({"format": "kd_launch_log/1", "event": "launch", "run_id": rid, "stage": s["stage"],
                            "seed": s["seed"], "horizon": int(slot.rsplit(":", 1)[1][:-1]) * 1000,
                            "sweep": s["sweep"]["key"] if s["sweep"] else None,
                            "value": float(s["sweep"]["value"]) if s["sweep"] else None,
                            "lambda_logit": None if lam is None else float(lam),
                            "alpha_cwd": None if alpha is None else float(alpha), "attempt": 1, "schedule": "T",
                            "code_pin": "0" * 40, "ckpt_dir": f"/workspace/kd/{rid}", "am8a_report": None})
        meta = d / f"{s['stage'].split('-')[0].lower()}_run_meta.jsonl"
        self.launch.append({"event": "launched", "run_id": rid, "run_meta_sha256": mods()[6].sha256_bytes(
            meta.read_bytes()), "launch_time_utc": FX.utc("2026-10-10T00:00:00Z"), "git_head": "0" * 40,
            "records_commit": "0" * 40, "decision_date": None, "launch_order_position": None})
        return d

    def commit(self, files: dict, when: str, *, draft: bool = True) -> str:
        S = mods()[6]
        files = dict(files)
        files[S.DECISION_LOG_REL] = FX.dl_table(self.dl).encode()
        if self.launch:
            files["reports/kd_launch_log.jsonl"] = b"".join(json.dumps(x).encode() + b"\n" for x in self.launch)
        if draft and self.draft is not None:
            files[S.MANIFEST_REL] = S.json_file_bytes({"format": S.MANIFEST_FORMAT, "body": {"entries": self.draft},
                                                       "appends": [], "session": None})
        return FX.commit_files(self.repo, files, "records", date=when)

    def view(self, day: str, reader=None):
        A, _, _, _, _, R, _ = mods()
        return A.View(R.Records.at_head(self.repo), self.cat, self.runs, day, reader=reader)


def body(slot: str, kind: str, rows: list, **kw) -> dict:
    return {"slot": slot, "kind": kind, "dl_rows": rows, **kw}


# ================================================================================================= the cases
def case_static():
    """B2R11: am22.py in LANE_CODE_FILES and score-blind (the freeze entry's field `conditions` aside, as
    smoke_tman_noscore reads it); B2R13: L-AM3's reader is the trigger reader."""
    A, C, D, E, M, R, S = mods()
    import scripts.smoke_tman_noscore as NS
    check("am22_is_lane_code_B2R11", "src/tman/am22.py" in M.LANE_CODE_FILES)
    hits = [s for s in NS.code_literals(Path(A.__file__)) if NS.SCORE.search(s) and s != "conditions"]
    check("am22_holds_no_score_key_literal_B2R11", not hits, str(hits))
    import importlib
    mod = importlib.import_module(R.TRIGGER_MODULE)
    check("trigger_reader_is_l_am3_s_B2R13", R.TRIGGER_MODULE == "src.quant.e6kd_trigger"
          and R.TRIGGER_READER is R.l_am3_trigger and callable(getattr(mod, "read_trigger_record", None))
          and mod.TRIGGER_RECORD_REL == R.TRIGGER_RECORD_REL)
    check("settled_ways_are_completion_s_enum_B2R3", C.SETTLED_WAYS == ("complete", "superseded", "diverged", "aborted",
                                                                       "cut", "not_run")
          and C.settled_ways(True) == ("complete", "superseded") and C.settled_ways(False) == C.SETTLED_WAYS)


def case_trigger_reader():
    """L-AM3's reader through the wrapper on synthetic e6kd_trigger/1 records: its verdict and nothing else."""
    A, C, D, E, M, R, S = mods()
    import importlib
    mod = importlib.import_module(R.TRIGGER_MODULE)
    repo = FX.new_repo()

    def rec(e3, e6, **kw):
        dec = mod.drop_decision(e3, e6)
        return S.json_file_bytes({"format": mod.TRIGGER_FORMAT, "smoke_inputs": False, "seed": 42, "e3_val": e3,
                                  "e6_val": e6, "drop_pp": dec["drop_pp"], "triggered": dec["triggered"], **kw})
    files = {"reports/derived/trig_fired.json": rec(0.40, 0.38), "reports/derived/trig_quiet.json": rec(0.40, 0.395),
             "reports/derived/trig_bad.json": rec(0.40, 0.38, triggered=False)}
    FX.commit_files(repo, files)
    read = R.trigger_reader(R.Records.at_head(repo))

    def verdict(rel):
        try:
            return read(rel, files[rel])
        except ValueError as e:
            return f"ValueError: {e}"[:200]
    got = [verdict(rel) for rel in ("reports/derived/trig_fired.json", "reports/derived/trig_quiet.json")]
    check("l_am3_reader_gives_the_verdict_only_B2R13", got == [{"triggered": True}, {"triggered": False}], str(got))
    try:
        read("reports/derived/trig_bad.json", files["reports/derived/trig_bad.json"])
        why = None
    except ValueError as e:
        why = str(e)
    check("l_am3_reader_refusal_is_a_value_error_B2R13", why is not None and "trigger_inconsistent" in why, str(why))


def case_registered():
    """B2R3's exclusions: E6-KD unless triggered; an E3-X on a no_go or not_evaluable go rule, with its E6-X (m7);
    an E6-X on a condition entry reading no; derived and measurement slots are no runs."""
    A, C, D, E, M, R, S = mods()
    cat = catalogue()
    runs = [k for k, s in cat.slots.items() if s["class"] == "run"]
    m = Mini()
    m.commit({}, "2026-10-14T02:00:00+00:00")
    regs, excl, _, _ = m.view("2026-10-14").registered()
    check("registered_are_the_run_slots_but_e6kd_untriggered", len(runs) == 41 and regs == [k for k in runs if k !=
                                                                                         A.E6_KD]
          and set(excl) == {A.E6_KD} and "ARM:latency" not in regs and "E6:s42:sel" not in regs, str(excl))
    trig = S.json_file_bytes({"format": "am3_trigger/1", "triggered": True})
    m.row(f"the trigger {R.TRIGGER_RECORD_REL} sha256 {S.sha256_bytes(trig)}")
    m.commit({R.TRIGGER_RECORD_REL: trig}, "2026-10-15T02:00:00+00:00")
    regs, excl, _, _ = m.view("2026-10-15", VW.World.trigger_reader()).registered()
    check("e6kd_registered_once_triggered_B2R3", A.E6_KD in regs and not excl, str(excl))
    pa = E._pa()
    for verdict, day, out in (("no_go", "2026-10-16", True), ("not_evaluable", "2026-10-17", True),
                              ("go", "2026-10-18", False)):
        m2 = Mini()
        go = S.json_file_bytes({"format": pa.GO_RULE_FORMAT, "arm": "R1", "verdict": verdict,
                                "generated_utc": "2026-10-16T01:00:00Z"})
        m2.row(f"R1's go rule {S.sha256_bytes(go)} {verdict}")
        m2.commit({pa.GO_RULE_REL.format(arm="R1"): go}, f"{day}T02:00:00+00:00")
        _, excl, recs, _ = m2.view(day).registered()
        gone = {"E3-R1:s42:lambda0.5:alpha25:80k", "E6-R1:s42:15ep"} <= set(excl)
        check(f"go_rule_{verdict}_{'excludes' if out else 'keeps'}_e3_r1_and_e6_r1_m7", gone == out
              and (not out or any(r.kind == "AM-18 go-rule record of R1" for r in recs)), str(excl))
    m3 = Mini()
    go = S.json_file_bytes({"format": pa.GO_RULE_FORMAT, "arm": "R1", "verdict": "no_go",
                            "generated_utc": "2026-10-16T01:00:00Z"})
    m3.commit({pa.GO_RULE_REL.format(arm="R1"): go}, "2026-10-16T02:00:00+00:00")
    check("go_rule_not_entered_excludes_nothing", "E3-R1:s42:lambda0.5:alpha25:80k" not in
          m3.view("2026-10-16").registered()[1])
    for verdict, out in (("no", True), ("yes", False)):
        m4 = Mini()
        c = m4.entry("condition", ["E6-R2:s42:15ep"], {"arm": "R2", "verdict": verdict, "e3x": {
            "run_id": "e3-r2_s42_lambda0.5_alpha25_80k_a1", "checkpoint_sha256": "5" * 64,
            "best_json_sha256": "6" * 64}}, "2026-10-23")
        m4.commit({c["path"]: c["data"]}, "2026-10-23T02:00:00+00:00")
        excl = m4.view("2026-10-23").registered()[1]
        check(f"condition_entry_{verdict}_{'excludes' if out else 'keeps'}_e6_r2_AM23_3c",
              ("E6-R2:s42:15ep" in excl) == out and "E3-R2:s42:lambda0.5:alpha25:80k" not in excl, str(excl))
    m5 = Mini()
    two = [m5.entry("condition", ["E6-R2:s42:15ep"], {"arm": "R2", "verdict": verdict, "e3x": {
        "run_id": "e3-r2_s42_lambda0.5_alpha25_80k_a1", "checkpoint_sha256": "5" * 64, "best_json_sha256": "6" * 64}},
        "2026-10-23", nn=nn) for nn, verdict in ((1, "yes"), (2, "no"))]
    m5.commit({c["path"]: c["data"] for c in two}, "2026-10-23T02:00:00+00:00")
    _, excl, _, missing = m5.view("2026-10-23").registered()
    check("condition_entries_that_disagree_are_missing_AM23_3b", "E6-R2:s42:15ep" not in excl and any(
        x.startswith("E6-R2:s42:15ep: its AM-23 condition entries disagree") for x in missing), str(missing))


def case_settled_ways():
    """Every settled way at the end of a date, from the records committed by then, and a never-cut run's limits."""
    A, C, D, E, M, R, S = mods()
    m = Mini()
    g, lam1, e1 = "G:s42:80k", "E2:s42:lambda1:80k", "E1:s42:160k"
    m.kd(g, "g_s42_80k_a1", end="2026-10-20T10:00:00Z")
    m.commit({}, "2026-10-18T02:00:00+00:00")
    check("complete_at_the_end_of_x_B2R3", m.view("2026-10-20").settled(g)[0] == "complete")
    way, _, why = m.view("2026-10-19").settled(g)
    check("launched_not_complete_the_day_before", way is None and "launched and not complete" in why, str(why))
    # a never-cut default candidate diverged: settled once entered as superseded; AM-7's full consequence fires
    m.kd(lam1, "e2_s42_lambda1_80k_a1", abort={"rule": "AM-7(a)", "cause": "student_divergence", "input_finite": True,
                                               "teacher_finite": True, "at": "2026-10-15T00:00:00Z"})
    dv = m.row("E2 lambda 1 diverged")
    m.draft = [body(lam1, "diverged", [dv], run_id="e2_s42_lambda1_80k_a1")]
    m.commit({}, "2026-10-21T02:00:00+00:00")
    way, _, why = m.view("2026-10-21").settled(lam1)
    check("never_cut_diverged_is_not_settled_until_superseded", way is None and "superseded" in why, str(why))
    sp = m.row("E2 lambda 1 superseded (AM-19 item 7(d))")
    m.draft = [body(lam1, "superseded", [sp], run_id="e2_s42_lambda1_80k_a1", superseding_dl=sp)]
    m.commit({}, "2026-10-22T02:00:00+00:00")
    v = m.view("2026-10-22")
    check("never_cut_superseded_is_settled_B2R3", v.settled(lam1)[0] == "superseded")
    c1 = A.condition_1(v)
    check("am7_full_consequence_is_a_stop_to_ask_n3", any("AM-7's full consequence fired" in x for x in c1.missing),
          str(c1.missing[:3]))
    # a never-cut run entered aborted is a STOP; a cut does not settle it
    ab = m.row("E1 aborted")
    m.draft = [body("E1:s42:80k", "aborted", [ab], run_ids=["a", "b"], rule="x", cause="y")]
    c = m.cut(["E1:s42:80k"], "cutoff", "2026-10-20")
    m.commit({c["path"]: c["data"]}, "2026-10-23T02:00:00+00:00")
    way, _, why = m.view("2026-10-23").settled("E1:s42:80k")
    check("never_cut_aborted_is_a_stop_to_ask", way is None and why == A.STOP_TEXT, str(why))
    m.draft = []
    m.commit({}, "2026-10-24T02:00:00+00:00")
    way, _, why = m.view("2026-10-24").settled("E1:s42:80k")
    check("never_cut_cut_is_not_settled", way is None and "never-cut run settles only" in why, str(why))
    # cuttable: diverged, aborted, cut by a rule other than the freeze, not run; a freeze cut settles nothing
    dv2, ab2 = m.row("A diverged"), m.row("F aborted")
    m.draft = [body("A:s42:alpha25:80k", "diverged", [dv2], run_id="a1"),
               body("F:s42:alpha25:80k", "aborted", [ab2], run_ids=["f1", "f2"], rule="AM-8a", cause="fault")]
    cutoff, frz = m.cut([e1], "cutoff", "2026-10-20"), m.cut(["E2:s42:lambda0.5:160k"], "freeze", "2026-11-03")
    nr = m.entry("not_run", ["E6:s43:15ep", "E7:s43:list42"], {"reason": "parent cut", "parent":
                 "E3:s43:lambda0.5:alpha25:80k", "parent_record": {"path": cutoff["path"], "sha256": cutoff["sha256"]}},
                 "2026-10-25")
    m.commit({x["path"]: x["data"] for x in (cutoff, frz, nr)}, "2026-10-25T02:00:00+00:00")
    v = m.view("2026-10-25")
    got = {k: v.settled(k)[0] for k in ("A:s42:alpha25:80k", "F:s42:alpha25:80k", e1, "E6:s43:15ep")}
    check("cuttable_diverged_aborted_cut_and_not_run_settle_B2R3", got == {
        "A:s42:alpha25:80k": "diverged", "F:s42:alpha25:80k": "aborted", e1: "cut", "E6:s43:15ep": "not_run"},
        str(got))
    way, _, why = v.settled("E2:s42:lambda0.5:160k")
    check("a_freeze_cut_settles_no_run", way is None and "at the freeze" in why, str(why))
    # entered by the end of X: rows and files committed later are not read at X (C_X, audit M3)
    late = m.cut(["A:s42:alpha25:80k"], "cutoff", "2026-10-20")
    m.draft = []
    m.commit({}, "2026-10-26T02:00:00+00:00")
    m.commit({late["path"]: late["data"]}, "2026-10-27T02:00:00+00:00")
    check("entry_committed_after_x_is_not_read_at_x_M3", m.view("2026-10-26").settled("A:s42:alpha25:80k")[0] is None
          and m.view("2026-10-27").settled("A:s42:alpha25:80k")[0] == "cut")
    m.draft = [body("F:s42:alpha25:80k", "diverged", ["DL-999"], run_id="f1")]
    m.commit({}, "2026-10-28T02:00:00+00:00")
    way, _, why = m.view("2026-10-28").settled("F:s42:alpha25:80k")
    check("body_entry_whose_rows_are_absent_is_not_settled", way is None and "not in the decision log" in str(why),
          str(why))
    # a run complete at X by its own records, its launch lines committed after X: the records now name its attempt
    m2 = Mini()
    m2.kd(g, "g_s42_80k_a1", end="2026-10-20T10:00:00Z")
    lines, m2.launch = m2.launch, []
    m2.commit({}, "2026-10-20T02:00:00+00:00")
    m2.launch = lines
    m2.commit({}, "2026-10-22T02:00:00+00:00")
    way, _, why = m2.view("2026-10-20").settled(g)
    check("run_complete_at_x_its_launch_lines_committed_later_settles_R1_F5", way == "complete", str(why))


def case_commit_at_end():
    """C_X is the newest commit on the records ref's first-parent chain whose committer time meets the end of X, the
    Manila day's end; a commit whose push is confirmed on a later day (--pushed, audit M2) counts from that day: C_X
    stops before it."""
    A, C, D, E, M, R, S = mods()
    m = Mini()
    a = m.commit({}, "2026-10-20T02:00:00+00:00")
    b = m.commit({}, "2026-10-19T02:00:00+00:00")           # a commit dated before its parent
    c = m.commit({}, "2026-10-21T15:59:59+00:00")
    src = R.Records.at_head(m.repo).src
    got = [A.commit_at_end(src, day) for day in ("2026-10-11", "2026-10-19", "2026-10-20", "2026-10-21")]
    check("c_x_walks_back_from_the_tip_M3", got == [None, b, b, c], str([x and x[:7] for x in got]) + f" a {a[:7]}")
    d = m.commit({}, "2026-10-21T16:30:00+00:00")           # Manila 00:30 on 2026-10-22
    src = R.Records.at_head(m.repo).src
    got = [A.commit_at_end(src, day) for day in ("2026-10-21", "2026-10-22")]
    check("c_x_is_read_at_the_end_of_the_manila_day_R3_09", got == [c, d], str([x and x[:7] for x in got]))
    got = [A.commit_at_end(src, day, {c: "2026-10-23"}) for day in ("2026-10-21", "2026-10-22", "2026-10-23")]
    check("c_x_stops_before_a_commit_pushed_later_M2", got == [b, b, d], str([x and x[:7] for x in got]))
    m2 = Mini()                                              # a lane merge: its side commit is not C_X
    m2.commit({}, "2026-10-20T02:00:00+00:00")
    FX.git(m2.repo, "checkout", "-q", "-b", "side")
    side = m2.commit({}, "2026-10-21T10:00:00+00:00")
    FX.git(m2.repo, "checkout", "-q", "master")
    main = m2.commit({}, "2026-10-21T02:00:00+00:00")
    FX.git(m2.repo, "merge", "-q", "--no-ff", "-m", "a lane merge", "side", date="2026-10-23T02:00:00+00:00")
    got = A.commit_at_end(R.Records.at_head(m2.repo).src, "2026-10-21")
    check("c_x_reads_the_first_parent_chain_R3", got == main, f"{got and got[:7]} (side {side[:7]})")


def case_providers():
    """B2R6: AM-22's provider in front of B-1's at every freeze_instant call -- the candidate for D (--am22-d), the
    freeze record of item 1(b) (the unlock's, or the one the decision log enters), none for item 1(f) or under Schedule
    R; beside an item 1(b) freeze, a freeze delay or new freeze is freeze_source_conflict (audit m10)."""
    import ast
    A, C, D, E, M, R, S = mods()
    root, calls = Path(M.__file__).resolve().parents[2], []

    def name(node):
        return getattr(node, "attr", None) or getattr(node, "id", None)

    def given(kw):
        """What a `providers=` keyword passes: a call of am22's providers (or the validator's), AM-22's candidate,
        build_manifest's own parameter -- never nothing, an empty literal or anything else (DL-24 R2-17, R3)."""
        if isinstance(kw, ast.Call) and name(kw.func) in ("providers", "am22_providers"):
            return "providers()"
        if isinstance(kw, ast.List) and kw.elts and all(isinstance(e, ast.Call) and name(e.func) == "am22_candidate"
                                                        for e in kw.elts):
            return "[am22_candidate()]"
        return "parameter" if isinstance(kw, ast.Name) and kw.id == "providers" else ast.dump(kw) if kw else None
    for rel in M.LANE_CODE_FILES:
        if rel.endswith(".py"):
            for node in ast.walk(ast.parse((root / rel).read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and name(node.func) in ("freeze_instant", "build_manifest"):
                    kw = next((k.value for k in node.keywords if k.arg == "providers"), None)
                    calls.append((rel, name(node.func), node.lineno, given(kw)))
    ok = {("freeze_instant", "providers()"), ("freeze_instant", "[am22_candidate()]"),
          ("build_manifest", "providers()")}
    check("every_freeze_instant_call_puts_am22_s_provider_in_front_B2R6", len(calls) == 9 and all(
        (fn, g) in ok or (rel, fn, g) == ("src/tman/manifest.py", "freeze_instant", "parameter")
        for rel, fn, _, g in calls), str([c for c in calls if (c[1], c[3]) not in ok]))
    sch = {"in_force": "T", "file": "configs/kd_schedule.json", "file_sha256": "1" * 64, "decision_log_entry": "DL-70"}
    got = D.freeze_instant(sch, providers=[D.am22_candidate(D_WORLD)]).as_block()
    check("candidate_freeze_is_the_end_of_d_plus_1_B2R6", (got["original_freeze_date"], got["moved_freeze_date"],
                                                          got["freeze_source"], got["test_deadline"])
          == (F_WORLD, None, {"provider": "am22", "path": None, "sha256": None}, "2026-11-08"), str(got))
    got = D.freeze_instant(dict(sch, in_force="R"), providers=[D.am22_candidate(D_WORLD)])
    check("schedule_r_takes_no_am22_freeze_B2R6", got.source["provider"] == "schedule", str(got.source))
    m = Mini()
    rec = R.Records.at_head(m.repo)
    check("item_1f_record_gives_no_provider_B2R6", A.providers(rec, freeze_doc={
        "source": D.AM22_SOURCE_1F, "freeze_date": "2026-11-03"}) == [] and A.providers(rec, lookup=True) == [])
    one = A.providers(rec, freeze_doc={"source": D.AM22_SOURCE_1B, "freeze_date": F_WORLD})
    check("item_1b_record_gives_its_freeze_date_B2R6", len(one) == 1
          and one[0](D.schedule_provider(sch)).original_date == F_WORLD)
    def code_of(d_date):
        try:
            A.providers(rec, d_date=d_date)
        except S.TmanRefused as e:
            return e.code
        except ValueError as e:
            return type(e).__name__
        return None
    for name, d_date in (("am22_d_not_a_date_refused_B2R6", "2026-10-3"), ("am22_d_before_1017_refused_R2_15",
                                                                           "2026-10-16"),
                         ("am22_d_after_1102_refused_R2", "2026-11-03")):
        check(name, code_of(d_date) == "schedule_invalid", str(code_of(d_date)))
    m.row("TMAN freeze_delay entry for the session: reports/derived/tday_entries/"
          f"freeze_delay__session__20261103__01.json sha256 {'2' * 64}")
    m.commit({}, "2026-10-31T02:00:00+00:00")
    rec = R.Records.at_head(m.repo)
    expect_refused("freeze_delay_beside_an_item_1b_freeze_conflicts_m10", "freeze_source_conflict", A.providers, rec,
                   d_date=D_WORLD)
    check("freeze_delay_without_an_item_1b_freeze_is_b1_s_m10", A.providers(rec, lookup=True) == []
          and len(A.providers(rec, d_date=D_WORLD, conflict=False)) == 1)
    # item 1(b) freeze records the decision log enters: two conflict; one a later entry corrects is none (DL-24 N9)
    conds = [{"number": n, "records": []} for n in range(1, 6)]

    def frz(mini, d, nn, corrects=None):
        f = D.iso(D.day(d) + timedelta(days=1))
        doc = E.make_entry("freeze", [], f, {"body_sha256": str(nn) * 64, "freeze_date": f, "source": D.AM22_SOURCE_1B,
                                             "d_date": d, "conditions": conds}, amendment="AM-22",
                           script="scripts/tday_manifest.py", commit=None, corrects=corrects)
        rel, data = S.entry_file_rel(E.entry_name(doc, nn)), S.json_file_bytes(doc)
        mini.row(f"TMAN freeze entry for the session: {rel} sha256 {S.sha256_bytes(data)}")
        return {"path": rel, "sha256": S.sha256_bytes(data), "data": data}
    m2 = Mini()
    two = (frz(m2, D_WORLD, 1), frz(m2, "2026-10-31", 2))
    m2.commit({f["path"]: f["data"] for f in two}, "2026-11-01T02:00:00+00:00")
    rec = R.Records.at_head(m2.repo)
    expect_refused("two_item_1b_freeze_records_conflict_B2R6", "freeze_source_conflict", A.providers, rec, lookup=True)
    try:
        got = [p(D.schedule_provider(sch)).original_date for p in A.providers(rec, lookup=True, conflict=False)]
    except S.TmanRefused as e:
        got = [e.code]
    check("two_item_1b_records_without_the_conflict_check_give_the_first_R2_13", got == [F_WORLD], str(got))
    m3 = Mini()
    old = frz(m3, D_WORLD, 1)
    new = frz(m3, "2026-10-31", 2, corrects={"path": old["path"], "sha256": old["sha256"]})
    m3.commit({f["path"]: f["data"] for f in (old, new)}, "2026-11-01T02:00:00+00:00")
    rec = R.Records.at_head(m3.repo)
    try:
        got = [p(D.schedule_provider(sch)).original_date for p in A.providers(rec, lookup=True)] + [A.entered_1b(rec)]
    except S.TmanRefused as e:
        got = [e.code]
    check("a_corrected_item_1b_record_is_no_freeze_of_record_N9", got == ["2026-11-01", [new["path"]]], str(got))


def case_conditions_in_the_world():
    """The AM-22 world: all five conditions MET at the end of 2026-10-30 with their records; MISSING on earlier dates
    for what was not entered yet."""
    A, C, D, E, M, R, S = mods()
    w = world()
    with stand_ins(w):
        chk = w.evaluate(D_WORLD)
        c = {x.number: x for x in chk.conditions}
        check("world_conditions_1_to_5_met_at_1030", all(c[n].met for n in (1, 2, 3, 4, 5)),
              str({n: c[n].missing[:2] for n in (1, 2, 3, 4, 5) if not c[n].met}))
        regs = chk.view.registered()[0]
        check("world_registers_40_runs_and_settles_each", len(regs) == 40 and not c[1].missing, str(len(regs)))
        ways = sorted({r.kind.split(" ")[0] for r in c[1].records})
        check("world_settles_by_completion_cut_and_not_run", ways == ["AM-3", "completion", "cut", "not-run"],
              str(ways))
        early = {x.number: x for x in w.evaluate("2026-10-29").conditions}
        d21 = {x.number: x for x in w.evaluate("2026-10-21").conditions}
        d22 = {x.number: x for x in w.evaluate("2026-10-22").conditions}
    miss = sorted(x.split(": ")[0] for x in early[1].missing)
    check("r2_not_cut_by_1029_condition_1_missing", miss == sorted(R2 + ["E6-R2:s42:15ep", "teacher:s42:40k"])
          and any(x.startswith("teacher:s42:40k: its run is named by no launcher") for x in early[1].missing),
          str(miss))
    check("lane_register_not_committed_by_1029_condition_4_missing", not early[4].met
          and "not committed by the end of 2026-10-29" in early[4].missing[0], str(early[4].missing))
    check("trigger_not_entered_by_1021_condition_2_missing", not d21[2].met
          and "is not committed by the end of 2026-10-21" in d21[2].missing[0], str(d21[2].missing))
    check("trigger_entered_1022_read_for_triggered_draft_not_yet", not any("trigger" in m for m in d22[2].missing)
          and any("are not made by the end" in m for m in d22[2].missing), str(d22[2].missing))
    check("draft_not_committed_by_1029_condition_3_missing", not early[3].met
          and "is not committed by the end of 2026-10-29" in early[3].missing[0], str(early[3].missing))
    check("condition_3_reads_the_draft_s_e6s42_hashed_block", c[3].met and any(
        r.kind == "E6:s42:sel hashed in the draft" for r in c[3].records))
    check("trigger_entered_on_its_due_day_prints_no_report_R3_11", c[2].met and not any(
        "trigger entered late" in x for x in c[2].reports), str(c[2].reports))
    lanes = [r.kind for r in c[4].records]
    check("condition_4_reads_implemented_and_not_required_lanes_not_arm_1g", "lane L-AM3 IMPLEMENTED" in lanes
          and "lane L-LAT-THREADS not required" in lanes and "lane L-AM23Q not required" in lanes
          and not any("L-AM16-ARM" in k for k in lanes) and any("L-AM16-ARM is not read" in n for n in c[4].notes),
          str(lanes))


def case_condition_5():
    """Condition 5 (B2R6; TMAN-B2-RULE-3): the draft at C_X passes freeze mode with AM-22's candidate for D = X, the
    ARM measurement's entry its one unmet freeze-time item (its freeze cut cannot exist before the freeze); MISSING
    with a run slot unentered, or for a draft built without the candidate (schedule_invalid)."""
    A, C, D, E, M, R, S = mods()
    w = world()
    with stand_ins(w):
        v = A.View(w.records, w.cat, w.runs, D_WORLD, reader=w.trigger_reader())
        c5 = A.condition_5(v, registered_cells=PINS.IF6_CELLS)
        check("condition_5_met_its_one_unmet_item_the_arm_measurement_RULE3", c5.met and any(
            n.startswith("['ARM:latency']: the measurement is cut at the freeze") for n in c5.notes),
            str(c5.missing or c5.notes))
        check("condition_5_records_the_draft_at_c_x_and_the_evaluator_commit", [(r.kind, r.ref, r.sha256) for r in
                                                                               c5.records]
              == [("draft manifest", S.MANIFEST_REL, S.sha256_bytes(w.data)), ("evaluator commit", w.E, None)],
              str([(r.kind, r.ref) for r in c5.records]))
        check("condition_5_keeps_the_validator_s_reports_on_the_rule3_path_R1_F9", any(x.startswith(
            "REPORT deviation: 'no card' needs item 2(f)'s evidence (['G:s42:80k']") for x in c5.reports),
            str(c5.reports))
        draft, _, rec = v.draft()
        doc = copy.deepcopy(draft["doc"])
        doc["body"]["entries"] = [e for e in doc["body"]["entries"] if e["slot"] != "G:s42:80k"]
        v2 = A.View(w.records, w.cat, w.runs, D_WORLD, reader=w.trigger_reader())
        v2._cache["draft"] = ({"doc": doc, "data": S.json_file_bytes(doc),
                               "by_slot": {e["slot"]: e for e in doc["body"]["entries"]}}, None, rec)
        c5 = A.condition_5(v2, registered_cells=PINS.IF6_CELLS)
        check("condition_5_missing_with_a_run_slot_unentered_RULE3", not c5.met and c5.missing
              and "[cut_unentered_at_unlock]" in c5.missing[0] and "G:s42:80k" in c5.missing[0], str(c5.missing))
    f = five()
    with stand_ins(f):
        c5 = A.condition_5(A.View(f.records, f.cat, f.runs, D_WORLD, reader=f.trigger_reader()),
                           registered_cells=PINS.IF6_CELLS)
    check("condition_5_missing_for_a_draft_without_the_candidate_B2R6", not c5.met and c5.missing
          and "[schedule_invalid]" in c5.missing[0], str(c5.missing))


def case_condition_3():
    """Condition 3: a draft committed by X whose E6 seed 42 conversion and selection are not hashed is MISSING."""
    A = mods()[0]
    m = Mini()
    m.draft = [body("E6:s42:15ep", "listed_run", [], run_id="e6_s42_a1"),
               body("E6:s42:sel", "derived", [], hashed=None)]
    m.commit({}, "2026-10-23T02:00:00+00:00")
    c3 = A.condition_3(m.view("2026-10-23"))
    check("condition_3_missing_while_e6s42_is_not_hashed_R3_12", not c3.met and not c3.records and any(
        "E6:s42:sel is not hashed at the freeze" in x for x in c3.missing), str(c3.missing))


def case_condition_4_rows_at_c_x():
    """M3: a lane whose row turns IMPLEMENTED after X is not IMPLEMENTED by X; its merge commit an ancestor of C_X, its
    row naming it and its merge commit; a lane entered as not required by its entry, entered, whose cuts cover what
    the lane serves; the lane register itself committed, listing lanes in its form, its sha256 in a recorded row."""
    A, C, D, E, M, R, S = mods()
    m = Mini()
    base = m.commit({}, "2026-10-20T02:00:00+00:00")
    lane_dl = m.row(f"L-X merged at {base[:7]}")
    lanes = {"format": S.LANES_FORMAT, "lanes": [{"lane": "L-X", "serves": ["G:s42:80k"], "status": {
        "kind": "implemented", "dl": lane_dl, "merge_commit": base, "entry": None}}]}

    def register(day: str, doc: dict, row: bool = True):
        data = S.json_file_bytes(doc)
        if row:
            m.row(f"the lane register sha256 {S.sha256_bytes(data)}")
        m.commit({S.LANES_REL: data}, f"{day}T02:00:00+00:00")
        return A.condition_4(m.view(day))
    register("2026-10-21", lanes)
    m.dl[lane_dl] = ("IMPLEMENTED (2026-10-22)", m.dl[lane_dl][1])
    m.commit({}, "2026-10-22T02:00:00+00:00")
    at21, at22 = A.condition_4(m.view("2026-10-21")), A.condition_4(m.view("2026-10-22"))
    check("lane_row_implemented_after_x_is_missing_at_x_M3", not at21.met and at22.met,
          f"{at21.missing} / {at22.missing}")
    bad = copy.deepcopy(lanes)
    bad["lanes"][0]["status"]["merge_commit"] = "f" * 40
    named = m.row(f"merged at {base[:7]}", "IMPLEMENTED (2026-10-22)")       # names no lane
    cut = m.cut(["E1:s42:160k"], "cutoff", "2026-10-20")
    nr = m.entry("not_required", ["G:s42:80k"], {"lane": "L-N3", "cut_entries": [
        {"path": cut["path"], "sha256": cut["sha256"]}]}, "2026-10-23")
    m.commit({x["path"]: x["data"] for x in (cut, nr)}, "2026-10-23T01:00:00+00:00")

    def nreq(lane, entry):
        return {"lane": lane, "serves": ["G:s42:80k"], "status": {"kind": "not_required", "dl": None,
                                                                  "merge_commit": None, "entry": entry}}
    bad["lanes"] += [{"lane": "L-Y", "serves": ["G:s42:80k"], "status": {"kind": "implemented", "dl": named,
                                                                         "merge_commit": base, "entry": None}},
                     nreq("L-N1", None), nreq("L-N2", {"path": nr["path"], "sha256": "3" * 64}),
                     nreq("L-N3", {"path": nr["path"], "sha256": nr["sha256"]})]
    try:
        got = {x.split(":")[0]: x for x in register("2026-10-23", bad).missing}
    except (TypeError, KeyError, AttributeError) as e:          # a crash fails each lane's check by name
        got = {"crashed": f"{type(e).__name__}: {e}"}
    for lane, name, text in (
            ("lane L-X", "lane_merge_commit_not_in_the_records_is_missing", f"merge commit {'f' * 40} at "),
            ("lane L-Y", "lane_row_not_naming_the_lane_and_its_merge_is_missing_R3_15", "does not name the lane"),
            ("lane L-N1", "lane_not_required_without_its_entry_is_missing_R3_15", "not required without its entry"),
            ("lane L-N2", "lane_not_required_entry_not_entered_is_missing_R3_15", "is not entered by the end of"),
            ("lane L-N3", "lane_not_required_serving_an_uncut_slot_is_missing_R3_15",
             "serves ['G:s42:80k'] with no cut entry entered by the end of 2026-10-23")):
        check(name, text in got.get(lane, ""), str(got.get(lane)))
    for day, doc, row, name, text in (
            ("2026-10-24", {"format": S.LANES_FORMAT, "lanes": []}, True, "lane_register_listing_no_lane_R3_15",
             "lists no lane"),
            ("2026-10-25", {"format": S.LANES_FORMAT, "lanes": [{"lane": "L-Z"}]}, True,
             "lane_register_out_of_its_form_R3_15", "missing ['serves', 'status']"),
            ("2026-10-26", dict(lanes, lanes=[dict(lanes["lanes"][0], serves=["G:s42:80k", "E1:s42:160k"])]), False,
             "lane_register_in_no_recorded_row_R3_15", "is in no RECORDED or IMPLEMENTED row")):
        c4 = register(day, doc, row)
        check(name, not c4.met and any(text in x for x in c4.missing), str(c4.missing))


def case_condition_2_cases():
    """Condition 2: triggered and E6-KD not settled; E6 seed 42 non-finite (audit n4); its artifacts not made; a
    trigger entered after the end of the second day after E6 seed 42 completed (REPORT, never failing)."""
    A, C, D, E, M, R, S = mods()
    m = Mini()
    FX.qat_run(m.runs, "e6_s42_a1", stage="E6", launched="2026-10-19T00:00:00Z", end="2026-10-20T12:00:00Z")
    hashed = {"record": {"path": "e6_s42_a1/eval/qat_selection.json", "sha256": "7" * 64},
              "artifact_sha256": "8" * 64, "made_utc": "2026-10-21T02:00:00Z", "appended": "freeze"}
    m.draft = [body("E6:s42:15ep", "listed_run", [], run_id="e6_s42_a1"),
               body("E6:s42:sel", "derived", [], hashed=hashed)]
    trig = S.json_file_bytes({"format": "am3_trigger/1", "triggered": True})
    m.row(f"the trigger {R.TRIGGER_RECORD_REL} sha256 {S.sha256_bytes(trig)}")
    m.commit({R.TRIGGER_RECORD_REL: trig}, "2026-10-23T02:00:00+00:00")
    c2 = A.condition_2(m.view("2026-10-23", VW.World.trigger_reader()))
    check("triggered_e6kd_not_settled_condition_2_missing", any("AM-3 triggered E6-KD" in x for x in c2.missing),
          str(c2.missing))
    check("trigger_entered_late_is_a_report_never_a_failure", any(
        "deviation: trigger entered late (AM-22 item 1(a)2)" in x for x in c2.reports)
        and not any("late" in x for x in c2.missing), str(c2.reports))
    made = [(r.ref, r.sha256) for r in c2.records if r.kind == "derived artifacts of E6:s42:15ep"]
    check("e6s42_made_artifacts_read_from_the_draft", made == [(hashed["record"]["path"], "7" * 64)], str(made))
    m.draft = [body("E6:s42:15ep", "non_finite", [], run_id="e6_s42_a1")]
    m.commit({}, "2026-10-24T02:00:00+00:00")
    c2 = A.condition_2(m.view("2026-10-24", VW.World.trigger_reader()))
    check("e6s42_non_finite_needs_the_new_amendment_n4", len(c2.missing) == 1
          and "until a new amendment's code path is added" in c2.missing[0], str(c2.missing))
    m.draft = [body("E6:s42:15ep", "listed_run", [], run_id="e6_s42_a1")]
    m.commit({}, "2026-10-25T02:00:00+00:00")
    c2 = A.condition_2(m.view("2026-10-25", lambda path, data: {"triggered": False}))
    check("e6s42_artifacts_not_made_condition_2_missing", any("are not made by the end" in x for x in c2.missing),
          str(c2.missing))
    c2 = A.condition_2(m.view("2026-10-25", None))
    check("no_trigger_reader_condition_2_missing_O8", any("no reader of the AM-3 trigger record" in x
                                                          for x in c2.missing), str(c2.missing))
    # the trigger record committed but in no recorded row; its reader refusing it; a reader giving no verdict
    m2 = Mini()
    trig = S.json_file_bytes({"format": "am3_trigger/1", "triggered": False})
    m2.commit({R.TRIGGER_RECORD_REL: trig}, "2026-10-23T02:00:00+00:00")
    c2 = A.condition_2(m2.view("2026-10-23", VW.World.trigger_reader()))
    check("trigger_committed_in_no_recorded_row_is_missing_R3_15", any("the trigger is not entered" in x
                                                                       for x in c2.missing), str(c2.missing))
    m2.row(f"the trigger {R.TRIGGER_RECORD_REL} sha256 {S.sha256_bytes(trig)}")
    m2.commit({}, "2026-10-24T02:00:00+00:00")

    def refuses(path, data):
        raise ValueError("trigger_inconsistent: a fixture's refusal")
    for name, reader, text in (("trigger_reader_refusal_is_missing_R3_15", refuses,
                                "the trigger reader refuses it (ValueError: trigger_inconsistent"),
                               ("trigger_reader_without_a_boolean_verdict_is_missing_R3_15",
                                lambda path, data: {"verdict": "fired"}, "gives no triggered verdict")):
        c2 = A.condition_2(m2.view("2026-10-24", reader))
        check(name, any(text in x for x in c2.missing), str(c2.missing))
    # E6 seed 42 complete at 17:00Z, Manila the next day: the trigger is due by the end of the second day after that
    m3 = Mini()
    FX.qat_run(m3.runs, "e6_s42_a1", stage="E6", launched="2026-10-19T00:00:00Z", end="2026-10-20T17:00:00Z")
    m3.draft = [body("E6:s42:15ep", "listed_run", [], run_id="e6_s42_a1"),
                body("E6:s42:sel", "derived", [], hashed=hashed)]
    m3.row(f"the trigger {R.TRIGGER_RECORD_REL} sha256 {S.sha256_bytes(trig)}")
    m3.commit({R.TRIGGER_RECORD_REL: trig}, "2026-10-23T02:00:00+00:00")
    c2 = A.condition_2(m3.view("2026-10-23", VW.World.trigger_reader()))
    check("trigger_due_day_counts_from_the_manila_day_R3_09", c2.met and not c2.reports, str(c2.missing or c2.reports))


def case_report_lines():
    """M2: every committed record's commit and committer time, and a boundary-day REPORT for each made on X or X-1;
    m4: each earlier date's verdict, then D."""
    A = mods()[0]
    w = world()
    with stand_ins(w):
        chk, d, scans = w.check_freeze(D_WORLD)
    out = A.lines(chk, d, scans)
    rep = [x for x in out if x.startswith("REPORT boundary-day entry")]
    shas = {w.H_R2, w.H_DRAFT, w.H_NR}
    check("boundary_day_reports_name_the_commits_of_x_M2", rep and all(any(s in x for s in shas) for x in rep)
          and all(any(s in x and "by the end of 2026-10-30" in x for x in rep) for s in (w.H_R2, w.H_DRAFT))
          and not any(w.H_CUTS in x for x in rep), str(rep[:3]))
    check("boundary_day_report_of_x_minus_1_by_its_manila_day_R3_10", any(
        w.H_NR in x and "by the end of 2026-10-29" in x for x in rep), str(rep))
    check("records_print_their_commits_M2", any(x.startswith("  RECORD cut entry of G:s42:80k (cutoff)")
                                                and w.H_CUTS[:12] in x and "2026-10-21" in x for x in out))
    dated = [x for x in out if x.startswith("DATE ")]
    check("each_earlier_date_missing_then_d_m4", len(dated) == 13 and all(": MISSING " in x for x in dated)
          and out[-2:] == ["RESULT: MET (all five conditions at the end of 2026-10-30)", "D: 2026-10-30"],
          f"{len(dated)} {out[-2:]}")


def case_cli():
    """`check-freeze`: exit 0 with D once all five conditions hold at a date's end, 3 while none has; a date whose end
    has not passed (m6), one after 2026-11-02 (item 1(f)) or before AM-22 is in force refused."""
    import scripts.tday_manifest as CLI
    w = world()
    with stand_ins(w):
        base = ["--root", w.repo, "check-freeze", "--runs", w.runs, "--registered-cells", cells_file()]
        rc, out = cli(CLI, base + ["--date", D_WORLD, "--now", NOW])
        check("cli_check_freeze_prints_the_conditions_and_d_exit_0", rc == 0 and "CONDITION 1 MET" in out
              and "CONDITION 5 MET" in out and out.rstrip().endswith("D: 2026-10-30"), out[-300:])
        rc, out = cli(CLI, base + ["--date", "2026-10-29", "--now", NOW])
        check("cli_check_freeze_no_d_by_1029_exit_3", rc == 3 and "CONDITION 1 MISSING" in out
              and out.rstrip().splitlines()[-1].startswith("NO D: "), out[-300:])
        rc, out = cli(CLI, base + ["--date", D_WORLD, "--now", "2026-10-30T15:59:59Z"])
        check("cli_date_not_ended_refused_m6", rc == 2 and "am22_date_not_ended" in out, out[-200:])
        rc, out = cli(CLI, base + ["--date", "2026-11-03", "--now", NOW])
        check("cli_date_after_1102_refused_1f", rc == 2 and "am22_d_not_first" in out and "item 1(f)" in out,
              out[-200:])
        rc, out = cli(CLI, base + ["--date", "2026-10-16", "--now", NOW])
        check("cli_date_before_am22_refused", rc == 2 and "am22_d_not_first" in out, out[-200:])
        for name, extra, want, detail in (
                ("cli_date_not_a_date_refused", ["--date", "2026-10-3", "--now", NOW], "am22_date_not_ended",
                 "is not a date"),
                ("cli_now_not_a_utc_instant_refused", ["--date", D_WORLD, "--now", "2026-11-06"], "am22_date_not_ended",
                 "is not a UTC instant"),
                ("cli_pushed_not_commit_equals_day_refused_M2", ["--date", D_WORLD, "--now", NOW, "--pushed",
                                                                 "fixture"], "entry_field", "COMMIT=DAY"),
                ("cli_pushed_not_a_commit_refused_M2", ["--date", D_WORLD, "--now", NOW, "--pushed",
                                                        "fixture=2026-10-31"], "entry_field",
                 "not a commit of the records' history"),
                ("cli_pushed_on_its_committer_day_refused_M2", ["--date", D_WORLD, "--now", NOW, "--pushed",
                                                                f"{w.H_DRAFT}=2026-10-30"], "entry_field",
                 "a confirmation names a later day")):
            rc, out = cli(CLI, base + extra)
            check(name, rc == 2 and f"REFUSED [{want}]" in out and detail in out, out[-200:])
        rc, out = cli(CLI, base + ["--date", D_WORLD, "--now", NOW, "--pushed", f"{w.H_DRAFT}=2026-10-31"])
        check("cli_draft_pushed_after_d_leaves_no_d_at_d_M2", rc == 3 and f"C_X {w.H_R2}" in out
              and out.rstrip().splitlines()[-1].startswith("NO D: "), out[-300:])
        rc, out = cli(CLI, ["--root", w.repo, "validate", "--manifest", w.repo / mods()[6].MANIFEST_REL, "--mode",
                            "unlock", "--am22-d", D_WORLD])
        check("cli_validate_am22_d_in_unlock_mode_refused_R2_3", rc == 2 and "REFUSED [schedule_invalid]" in out
              and "--am22-d is read in draft and freeze modes" in out, out[-300:])


def case_confirmed_push():
    """Audit M2 (DL-24 R2-4, R4-1): a commit whose push is confirmed on a later day counts from that day -- the records
    it holds dated that day, C_X stopping before it, no boundary-day REPORT asked again -- and the freeze entry carries
    each confirmation; AM-3's trigger confirmed pushed the day after it was due is entered late (REPORT)."""
    A = mods()[0]
    w = world()
    with stand_ins(w):
        chk, d, _ = w.check_freeze(D_WORLD, pushed={w.H_TRIGGER: "2026-10-23", w.H_NR: "2026-10-30"})
    c = {x.number: x for x in chk.conditions}
    trig = [r.date for r in c[2].records if r.kind == "AM-3 trigger record"]
    check("confirmed_push_keeps_d_and_dates_the_trigger_by_its_push_M2", d == D_WORLD and chk.met
          and trig == ["2026-10-23"], f"{d} {trig}")
    check("trigger_confirmed_pushed_after_its_due_day_is_late_M2", any(
        "trigger entered late (AM-22 item 1(a)2): entered 2026-10-23, due by the end of 2026-10-22" in x
        for x in c[2].reports), str(c[2].reports))
    conds = A.freeze_conditions(chk)
    held = {(n["number"], r["ref"], r["date"]) for n in conds for r in n["records"] if r["kind"] == A.CONFIRMED_PUSH}
    check("freeze_conditions_carry_each_confirmed_push_M2", held == {(1, w.H_TRIGGER, "2026-10-23"),
                                                                     (2, w.H_TRIGGER, "2026-10-23"),
                                                                     (4, w.H_NR, "2026-10-30")}, str(held))
    rep = A.boundary_reports(chk)
    check("confirmed_commit_asked_no_boundary_report_again_M2", rep and not any(w.H_NR in x for x in rep), str(rep))
    lines = A.check_lines(chk)
    check("record_lines_print_the_confirmed_push_day_M2", any(
        x.startswith("  RECORD AM-3 trigger record") and "pushed 2026-10-23 (confirmed)" in x for x in lines))


def case_cli_am22_d():
    """B2R6 on the command line: `build --am22-d D` writes the draft with AM-22's candidate freeze for D (audit m5);
    `validate --am22-d D` checks the draft against it, freeze mode leaving only the ARM measurement's entry
    (TMAN-B2-RULE-3); without --am22-d the draft's schedule block is not the freeze B-1's providers give."""
    A, C, D, E, M, R, S = mods()
    import scripts.tday_manifest as CLI
    w = world()
    with stand_ins(w):
        src = scratch_file("tmansrc", "sources.json", S.json_file_bytes(w.sources))
        out = FX.scratch_dir("tmanbuild")
        rc, txt = cli(CLI, ["--root", w.repo, "build", "--sources", src, "--out", out, "--runs", w.runs,
                            "--registered-cells", cells_file(), "--am22-d", D_WORLD])
        made = out / "tday_manifest.json"
        doc = json.loads(made.read_bytes()) if made.is_file() else {}
        check("cli_build_am22_d_writes_the_candidate_s_draft_B2R6_m5", doc.get("body") == w.doc["body"]
              and doc["body"]["schedule"]["original_freeze_date"] == F_WORLD, f"rc {rc}: {txt.strip()[-300:]}")
        rc, txt = cli(CLI, ["--root", w.repo, "build", "--sources", src, "--out", FX.scratch_dir("tmanbuild"), "--runs",
                            w.runs, "--registered-cells", cells_file(), "--am22-d", "2026-11-03"])
        check("cli_build_am22_d_after_1102_refused_R2", rc == 2 and "REFUSED [schedule_invalid]" in txt
              and "is not a date from 2026-10-17 to 2026-11-02" in txt, txt[-300:])
        base = ["--root", w.repo, "validate", "--manifest", w.repo / S.MANIFEST_REL, "--mode", "freeze", "--runs",
                w.runs, "--registered-cells", cells_file()]
        rc, txt = cli(CLI, base + ["--am22-d", D_WORLD])
        check("cli_validate_am22_d_leaves_the_arm_measurement_RULE3", rc == 2
              and "REFUSED [cut_unentered_at_unlock]" in txt and "['ARM:latency'] have no entry" in txt, txt[-300:])
        rc, txt = cli(CLI, base)
        check("cli_validate_without_am22_d_schedule_invalid_B2R6", rc == 2 and "REFUSED [schedule_invalid]" in txt,
              txt[-300:])


def case_d_first():
    """Audit m4: D is the first date from 2026-10-17 at whose end all five conditions hold. In the world whose first
    such date is 2026-10-31, `check-freeze --date 2026-11-02` prints D: 2026-10-31; a freeze of item 1(b) for D
    2026-11-02 is refused by its printer and at the unlock (am22_d_not_first, every earlier unlock gate passing)."""
    A, C, D, E, M, R, S = mods()
    import scripts.tday_manifest as CLI
    f = five()
    with stand_ins(f):
        rc, out = cli(CLI, ["--root", f.repo, "check-freeze", "--runs", f.runs, "--registered-cells", cells_file(),
                            "--date", "2026-11-02", "--now", NOW])
        got = out.rstrip().splitlines()
        dated = {x.split(" ")[1]: x for x in got if x.startswith("DATE ")}
        check("d_is_the_first_date_whose_end_has_all_five_m4", rc == 0 and got[-1:] == ["D: 2026-10-31"]
              and "RESULT: MET (all five conditions at the end of 2026-11-02)" in got
              and sorted(dated) == A.days("2026-10-17", "2026-10-31") and dated["2026-10-31"].endswith(": MET")
              and dated["2026-10-30"].endswith(": MISSING 5"), out[-400:])
        rc, out = cli(CLI, ["--root", f.repo, "check-freeze", "--runs", f.runs, "--registered-cells", cells_file(),
                            "--date", "2026-11-01", "--now", NOW])
        got = out.rstrip().splitlines()
        i = next((n for n, x in enumerate(got) if x.startswith("CHECK-FREEZE OF D 2026-10-31")), len(got))
        check("x_missing_with_an_earlier_d_exits_0_R3_03", rc == 0 and got[-1:] == ["D: 2026-10-31"]
              and "RESULT: MISSING (condition 5 at the end of 2026-11-01)" in got[:i], out[-400:])
        check("check_freeze_prints_d_s_own_check_and_boundary_reports_R3_02", i < len(got) and any(
            x.startswith("REPORT boundary-day entry") and f.drafts[1] in x and "by the end of 2026-10-31" in x
            for x in got[i:]) and "RESULT: MET (all five conditions at the end of 2026-10-31)" in got[i:], out[-600:])
        check("draft_record_names_the_commit_of_its_last_change_R2_6", any(
            x.startswith("  RECORD draft manifest") and f"file {f.drafts[1][:12]}" in x for x in got[i:]),
            str([x for x in got[i:] if "draft manifest" in x]))
        why = A.d_not_first_error(f.records, f.cat, f.runs, "2026-11-02", reader=f.trigger_reader(),
                                  registered_cells=PINS.IF6_CELLS)
        check("d_not_first_names_the_first_date_m4", "already hold at the end of 2026-10-31" in str(why), str(why))
        for name, day, text in (("d_not_first_d_after_1102_is_item_1f_R3", "2026-11-03", "later than 2026-11-02"),
                                ("d_not_first_d_before_1017_refused_R2_15", "2026-10-16", "before AM-22 is in force")):
            why = A.d_not_first_error(f.records, f.cat, f.runs, day, reader=f.trigger_reader())
            check(name, text in str(why), str(why))
        man = scratch_file("tmanbody", "m.json", f.data)
        rc, out, made = f.print_entry("freeze", "2026-11-03", {}, nn=1, extra=f.freeze_args("2026-11-02", man, NOW))
        check("printer_freeze_of_a_later_d_refused_m4", rc == 2 and "REFUSED [am22_d_not_first]" in out
              and made is None, out.strip()[-300:])
        conds = A.freeze_conditions(f.evaluate("2026-11-02"))
        arm = f.entry_file("cut", ["ARM:latency"], f.cut_fields("ARM", ["ARM:latency"], "code not merged", False),
                           date="2026-11-03")                   # the ARM measurement cut at this freeze (RULE-3)
        f.commit({arm["path"]: f.files[arm["path"]]}, "the ARM latency cut at the freeze of D 2026-11-02",
                 date="2026-11-03T01:00:00+00:00")
        frozen = A.freeze_body(R.Records.at_head(f.repo), f.cat, "2026-11-02", [arm["path"]])
        data = S.json_file_bytes(frozen)
        fields = {"body_sha256": S.canonical_sha256(frozen["body"]), "freeze_date": "2026-11-03",
                  "source": D.AM22_SOURCE_1B, "d_date": "2026-11-02", "conditions": conds}
        ref = f.entry_file("freeze", [], fields, date="2026-11-03")
        head = f.commit({ref["path"]: f.files[ref["path"]], S.MANIFEST_REL: data}, "a freeze record of D 2026-11-02",
                        date="2026-11-03T02:00:00+00:00")
        VW.expect_gate(f, "unlock_freeze_of_a_later_d_refused_B2R7_m4", "am22_d_not_first", mode="unlock",
                       detail="already hold at the end of 2026-10-31", data=data, freeze_record=ref["path"],
                       manifest_path=S.MANIFEST_REL, records=f.records_at(head), repo_root=f.checkout())


def case_fallback_1f():
    """Item 1(f) applies only when item 1(a) gives no D by the end of 2026-11-02 (PA :2353; DL-24 R1-F2, R2-2): in the
    world whose D is 2026-10-31, a freeze of item 1(f) -- its body built without AM-22's candidate, AM-19's freeze as
    committed -- is refused by its printer and at the unlock (am22_d_not_first, every earlier unlock gate passing).
    The printer reads the run directories, runs once the end of 2026-11-02 has passed, and records a body whose
    schedule block is the resolver's (audit m8)."""
    A, C, D, E, M, R, S = mods()
    f = five()
    b1 = f.draft_docs[0]
    data = S.json_file_bytes(b1)
    man = scratch_file("tmanbody", "m.json", data)
    fields = {"body_sha256": S.canonical_sha256(b1["body"]), "freeze_date": A.FALLBACK_FREEZE, **VW.FREEZE_1F}
    cells = ["--registered-cells", cells_file()]
    with stand_ins(f):
        cand = scratch_file("tmanbody", "m.json", f.data)
        for name, flds, extra, want, detail in (
                ("printer_item_1f_with_a_d_refused_R1_F2", fields, ["--manifest", man, "--runs", f.runs, "--now", NOW],
                 "am22_d_not_first", "D is 2026-10-31"),
                ("printer_item_1f_before_the_end_of_1102_refused_R1_F2", fields,
                 ["--manifest", man, "--runs", f.runs, "--now", "2026-11-02T15:59:59Z"], "am22_date_not_ended",
                 "once the end of 2026-11-02 has passed"),
                ("printer_item_1f_without_the_run_directories_refused_R1_F2", fields, ["--manifest", man, "--now", NOW],
                 "launched_attempt_without_directory", "from the run directories (--runs)"),
                ("printer_item_1f_of_a_body_at_am22_s_freeze_refused_m8",
                 dict(fields, body_sha256=S.canonical_sha256(f.doc["body"])),
                 ["--manifest", cand, "--runs", f.runs, "--now", NOW], "freeze_entry_invalid", "the resolver's")):
            rc, out, made = f.print_entry("freeze", A.FALLBACK_FREEZE, flds, nn=7, extra=extra + cells)
            check(name, rc == 2 and f"REFUSED [{want}]" in out and detail in out and made is None,
                  f"rc {rc}: {out.strip()[-300:]}")
        ref = f.entry_file("freeze", [], fields, date=A.FALLBACK_FREEZE, nn=8)
        head = f.commit({ref["path"]: f.files[ref["path"]], S.MANIFEST_REL: data}, "the body of a freeze of item "
                        "1(f) and its record", date="2026-11-03T03:00:00+00:00")
        VW.expect_gate(f, "unlock_item_1f_freeze_with_a_d_refused_R1_F2", "am22_d_not_first", mode="unlock",
                       detail="D is 2026-10-31", data=data, freeze_record=ref["path"], manifest_path=S.MANIFEST_REL,
                       records=f.records_at(head), repo_root=f.checkout())


def case_unlock_world():
    """The AM-22 world at the unlock passes every unlock gate at S = E (B2R7; audit M4). Its freeze record, printed by
    `entry freeze --d-date`, holds D, D + 1 and the five conditions (B2R5), condition 5 citing the draft at C_D, and
    the frozen body's sha256 -- the draft at C_D and the ARM measurement's freeze cut (TMAN-B2-RULE-3)."""
    A, C, D, E, M, R, S = mods()
    try:
        u, why, at = unlocked(), None, None
    except PrinterRefused as e:                     # the printer that refused fails its check by name (DL-24 R3-04)
        u, why, at = None, str(e), e.printer
    for name, printer in (("am22_arm_freeze_cut_printed_by_entry_cut_d_date_R1_F1", "entry cut"),
                          ("am22_body_to_freeze_printed_by_freeze_body_R2_8", "freeze-body"),
                          ("am22_freeze_printer_passes_in_the_world_B2R5", "entry freeze"),
                          ("am22_unlock_printer_passes_in_the_world_M4", "entry unlock")):
        check(name, at != printer, str(why)[:400])
    if u is None:
        return
    with stand_ins(u):
        code, passed, e = u.run("unlock")
        check("am22_unlock_world_passes_every_unlock_gate_B2R7_M4", code is None
              and passed == VW.gate_codes("unlock"), f"{code}: {e.detail[:300] if e else ''}")
        code, _, e = u.run("unlock", unlock_entry=None)
        check("am22_unlock_before_its_entry_passes_M4", code is None, f"{code}: {e.detail[:200] if e else ''}")
        code, _, e = u.run("unlock", am22_d="2026-10-17")
        check("unlock_reads_the_freeze_record_not_am22_d_R2_3", code is None, f"{code}: {e.detail[:200] if e else ''}")
        rec = R.Records.at_head(u.tree)
        one = A.providers(rec, lookup=True)
        check("lookup_finds_the_entered_item_1b_freeze_B2R6", len(one) == 1
              and one[0](D.schedule_provider(D.schedule_block(rec.src))).original_date == F_WORLD)
    fr = u.freeze["doc"]
    check("freeze_record_of_item_1b_names_d_and_d_plus_1_B2R5", (fr["source"], fr["d_date"], fr["freeze_date"],
                                                                fr["date"]) == (D.AM22_SOURCE_1B, D_WORLD, F_WORLD,
                                                                                F_WORLD)
          and [c["number"] for c in fr["conditions"]] == [1, 2, 3, 4, 5]
          and all(r["date"] <= D_WORLD for c in fr["conditions"] for r in c["records"]), str(fr["conditions"])[:300])
    c5 = {(r["kind"], r["ref"], r["sha256"]) for r in fr["conditions"][4]["records"]}
    check("condition_5_cites_the_draft_at_c_d_RULE3_m9", c5 == {("draft manifest", S.MANIFEST_REL,
                                                                 S.sha256_bytes(u.draft_data)),
                                                                ("evaluator commit", u.E, None)}, str(c5))
    rest = [x for x in u.doc["body"]["entries"] if x["slot"] != "ARM:latency"]
    check("frozen_body_is_the_draft_and_the_arm_freeze_cut_RULE3", fr["body_sha256"] == S.canonical_sha256(
        u.doc["body"]) and dict(u.doc["body"], entries=rest) == u.draft_doc["body"]
        and (u.entry(u.doc, "ARM:latency")["rule"], u.entry(u.doc, "ARM:latency")["cut_date"]) == ("freeze", F_WORLD))
    sch = u.doc["body"]["schedule"]
    check("schedule_block_is_am22_s_freeze_B2R6", (sch["freeze_source"], sch["original_freeze_date"],
                                                   sch["test_deadline"])
          == ({"provider": "am22", "path": None, "sha256": None}, F_WORLD, "2026-11-08"), str(sch))


def case_freeze_printer():
    """`entry freeze --d-date D` at the body's commit, before the freeze record is entered (B2R5, B2R7; audit m8, and
    m9 as TMAN-B2-RULE-3 amends it): the same record recomputed, D's boundary-day REPORTs printed (DL-24 R3-02), a
    confirmed push recorded (--pushed, audit M2); refused for a D whose conditions do not hold, a body differing from
    the draft at C_D beyond the ARM measurement's freeze cut, that measurement entered complete (a STOP to ask) or in
    neither the draft nor the body, its cut not the freeze's (rule, date, day entered), another slot joining the
    draft, a body the validator refuses in freeze mode, fields given, a date other than D + 1, a freeze of item 1(b)
    given without --d-date, a --now before the end of D or not an instant, no run directories, a D after 2026-11-02
    (item 1(f)), a confirmation that is not one, and at the unlock entry's commit a second freeze of item 1(b)."""
    A, C, D, E, M, R, S = mods()
    u = unlocked()
    keys = ("body_sha256", "freeze_date", "source", "d_date", "conditions")
    g = u.cut_refs["G:s42:80k"]
    root = u.body_tree()

    def changed(edit) -> Path:
        doc = copy.deepcopy(u.doc)
        edit(doc)
        return scratch_file("tmanbody", "m.json", S.json_file_bytes(doc))

    def arm(**upd) -> Path:
        return changed(lambda d: u.entry(d, "ARM:latency").update(upd))

    def measured(d):
        i = next(i for i, x in enumerate(d["body"]["entries"]) if x["slot"] == "ARM:latency")
        d["body"]["entries"][i] = {"slot": "ARM:latency", "kind": "measurement", "dl_rows": [], "output": None,
                                   "agreement": None, "completion": None}

    def joined(d):
        d["body"]["entries"].append(dict(u.entry(d, "ARM:latency"), slot="E6-KD:s42:15ep"))
    with stand_ins(u):
        rc, out, made = u.print_entry("freeze", F_WORLD, {}, nn=2, extra=u.freeze_args(D_WORLD, u.man), root=root)
        got = json.loads(made.read_bytes()) if made else {}
        check("printer_recomputes_the_freeze_record_m9", rc == 0 and {k: got.get(k) for k in keys}
              == {k: u.freeze["doc"][k] for k in keys}, out.strip()[-300:])
        rep = [x for x in out.splitlines() if x.startswith("REPORT boundary-day entry")]
        check("printer_prints_d_s_boundary_day_reports_R3_02", {u.H_R2, u.H_DRAFT, u.H_NR} == {
            c for c in (u.H_R2, u.H_DRAFT, u.H_NR) if any(c in x for x in rep)} and len(rep) == 3, str(rep))
        rc, out, made = u.print_entry("freeze", F_WORLD, {}, nn=3, root=root, extra=u.freeze_args(D_WORLD, u.man) + [
            "--pushed", f"{u.H_TRIGGER}=2026-10-23"])
        got = json.loads(made.read_bytes()) if made else {}
        held = [r for c in got.get("conditions") or [] for r in c["records"] if r["kind"] == A.CONFIRMED_PUSH]
        check("printer_records_each_confirmed_push_M2", rc == 0 and held == [
            {"kind": A.CONFIRMED_PUSH, "ref": u.H_TRIGGER, "sha256": None, "date": "2026-10-23"}] * 2,
            f"rc {rc}: {out.strip()[-200:]} {held}")
        rc, out, made = u.print_entry("freeze", F_WORLD, {}, nn=4, extra=u.freeze_args(D_WORLD, u.man), root=u.tree)
        check("printer_second_item_1b_freeze_refused_R3_14", rc == 2 and "REFUSED [freeze_entry_invalid]" in out
              and "is entered already" in out and made is None, out.strip()[-300:])
        for name, date, fields, extra, want, detail in (
                ("printer_d_whose_conditions_do_not_hold_B2R7", D_WORLD, {}, u.freeze_args("2026-10-29", u.man),
                 "am22_condition_unmet", "condition 1 does not hold at the end of 2026-10-29"),
                ("printer_body_differing_beyond_the_arm_freeze_cut_RULE3", F_WORLD, {}, u.freeze_args(
                    D_WORLD, changed(lambda d: d["body"]["expected_rows"].update(value=1))), "freeze_entry_invalid",
                 "beyond the measurement's freeze cut"),
                ("printer_arm_measurement_complete_after_c_d_is_a_stop_RULE3", F_WORLD, {},
                 u.freeze_args(D_WORLD, changed(measured)), "freeze_entry_invalid", "a STOP to ask"),
                ("printer_arm_cut_dated_after_the_freeze_date_RULE3", F_WORLD, {},
                 u.freeze_args(D_WORLD, arm(cut_date="2026-11-01")), "freeze_entry_invalid", "the freeze's cut"),
                ("printer_arm_cut_by_another_rule_RULE3", F_WORLD, {}, u.freeze_args(D_WORLD, arm(rule="cutoff")),
                 "freeze_entry_invalid", "the freeze's cut"),
                ("printer_arm_cut_entered_on_another_day_RULE3", F_WORLD, {},
                 u.freeze_args(D_WORLD, arm(entry={"path": g["path"], "sha256": g["sha256"]})),
                 "freeze_entry_invalid", "entered 2026-10-21"),
                ("printer_another_slot_joining_the_draft_RULE3", F_WORLD, {}, u.freeze_args(D_WORLD, changed(joined)),
                 "freeze_entry_invalid", "only the ARM latency measurement's freeze cut joins"),
                ("printer_freeze_fields_given_with_d_date_m9", F_WORLD, {"body_sha256": "0" * 64},
                 u.freeze_args(D_WORLD, u.man), "entry_field", "which it recomputes"),
                ("printer_freeze_dated_other_than_d_plus_1_m8", "2026-11-01", {}, u.freeze_args(D_WORLD, u.man),
                 "freeze_entry_invalid", "not its freeze date 2026-10-31"),
                ("printer_item_1b_freeze_without_d_date_m9", F_WORLD, {k: u.freeze["doc"][k] for k in keys},
                 ["--manifest", u.man], "freeze_entry_invalid", "printed with --d-date D"),
                ("printer_d_whose_end_has_not_passed_m6", F_WORLD, {},
                 u.freeze_args(D_WORLD, u.man, "2026-10-30T15:59:59Z"), "am22_date_not_ended", "has not passed"),
                ("printer_now_not_a_utc_instant_refused", F_WORLD, {}, u.freeze_args(D_WORLD, u.man, "2026-10-31"),
                 "am22_date_not_ended", "is not a UTC instant"),
                ("printer_without_the_run_directories_refused", F_WORLD, {},
                 u.freeze_args(D_WORLD, u.man, runs=False), "launched_attempt_without_directory", "(--runs)"),
                ("printer_d_after_1102_item_1f", "2026-11-04", {}, u.freeze_args("2026-11-03", u.man, NOW),
                 "am22_d_not_first", "item 1(f)"),
                ("printer_measurement_in_neither_draft_nor_body_RULE3_R3_01", F_WORLD, {},
                 u.freeze_args(D_WORLD, scratch_file("tmanbody", "m.json", u.draft_data)), "freeze_entry_invalid",
                 "in neither the draft nor the body to freeze"),
                ("printer_body_refused_in_freeze_mode_R2", F_WORLD, {},
                 u.freeze_args(D_WORLD, arm(dl_rows=["DL-999"])), "freeze_entry_invalid",
                 "fails the validator in freeze mode with AM-22's freeze for D 2026-10-30: [dl_row_missing]"),
                ("printer_pushed_on_its_committer_day_refused_M2", F_WORLD, {},
                 u.freeze_args(D_WORLD, u.man) + ["--pushed", f"{u.H_DRAFT}=2026-10-30"], "entry_field",
                 "a confirmation names a later day"),
                ("printer_pushed_commit_no_record_needs_refused_M2", F_WORLD, {},
                 u.freeze_args(D_WORLD, u.man) + ["--pushed", f"{u.H_PILOT}=2026-10-19"], "entry_field",
                 "no record the check for 2026-10-30 relies on")):
            rc, out, made = u.print_entry("freeze", date, fields, nn=5, extra=extra, root=root)
            check(name, rc == 2 and f"REFUSED [{want}]" in out and detail in out and made is None,
                  f"rc {rc}: {out.strip()[-300:]}")


def case_freeze_body():
    """`freeze-body --d-date D --cut <entry>` (TMAN-B2-RULE-3; DL-24 R2-8): refused for a slot the draft at C_D enters
    already, for a measurement cut entered on a day other than the freeze date (frozen_body_error, checked as it
    writes), and for a D after 2026-11-02."""
    A, C, D, E, M, R, S = mods()
    import scripts.tday_manifest as CLI
    u = unlocked()
    g = u.cut_refs["G:s42:80k"]["path"]
    for name, d_date, cut, want, detail in (
            ("freeze_body_of_a_slot_the_draft_enters_refused_R2_8", D_WORLD, g, "freeze_entry_invalid",
             "entered in the draft at C_D already"),
            ("freeze_body_d_date_after_1102_refused_R2_8", "2026-11-03", u.arm["path"], "schedule_invalid",
             "is not a date from 2026-10-17 to 2026-11-02")):
        out = FX.scratch_dir("tmanfbody")
        with stand_ins(u):
            rc, txt = cli(CLI, ["--root", u.repo, "freeze-body", "--d-date", d_date, "--cut", cut, "--out", out])
        check(name, rc == 2 and f"REFUSED [{want}]" in txt and detail in txt and not any(out.iterdir()),
              f"rc {rc}: {txt.strip()[-300:]}")
    m = Mini()                                       # a draft at C_D and the ARM cut entered the day after its date
    m.draft = [body("E6:s42:15ep", "listed_run", [], run_id="e6_s42_a1")]
    m.commit({}, "2026-10-30T02:00:00+00:00")
    arm = m.cut(["ARM:latency"], "freeze", F_WORLD)
    m.commit({arm["path"]: arm["data"]}, "2026-11-01T02:00:00+00:00")
    try:
        A.freeze_body(R.Records.at_head(m.repo), catalogue(), D_WORLD, [arm["path"]])
        got = "written"
    except S.TmanRefused as e:
        got = f"[{e.code}] {e.detail}"
    check("freeze_body_of_a_cut_entered_after_the_freeze_date_refused_R2_8", got.startswith(
        "[freeze_entry_invalid]") and "entered 2026-11-01" in got, got[:300])


def case_measurement_cut_printer():
    """`entry cut --d-date D` (TMAN-B2-RULE-3; DL-24 R1-F1): before the freeze record exists, the ARM latency
    measurement's freeze cut is printed at AM-22's freeze for D (its date D + 1); without --d-date it would take B-1's
    freeze; --d-date prints no other cut, and --pushed is the freeze entry's."""
    w = world()
    fields = w.cut_fields("ARM", ["ARM:latency"], "code not merged", False)
    base = ["--d-date", D_WORLD, "--runs", w.runs]
    with stand_ins(w):
        rc, out, made = w.print_entry("cut", F_WORLD, fields, nn=9, extra=["--slot", "ARM:latency", *base])
        check("printer_measurement_freeze_cut_at_d_plus_1_R1_F1", rc == 0 and made is not None, out.strip()[-300:])
        for name, slot, flds, extra, want, detail in (
                ("printer_measurement_cut_without_d_date_takes_b1_s_freeze_R1_F1", "ARM:latency", fields,
                 ["--runs", w.runs], "cut_date_mismatch", "its rule's day is 2026-11-03"),
                ("printer_d_date_cut_of_a_run_slot_refused_R1_F1", "G:s42:80k",
                 w.cut_fields("G", ["G:s42:80k"], "gate not met", False), base, "entry_field",
                 "only for the measurement's freeze cut"),
                ("printer_d_date_cut_by_another_rule_refused_R1_F1", "ARM:latency", dict(fields, rule="cutoff"), base,
                 "entry_field", "only for the measurement's freeze cut"),
                ("printer_pushed_is_the_freeze_entry_s_M2", "ARM:latency", fields,
                 base + ["--pushed", f"{w.H_DRAFT}=2026-10-31"], "entry_field", "--pushed is the freeze entry's")):
            rc, out, made = w.print_entry("cut", F_WORLD, flds, nn=9, extra=["--slot", slot, *extra])
            check(name, rc == 2 and f"REFUSED [{want}]" in out and detail in out and made is None,
                  f"rc {rc}: {out.strip()[-300:]}")


def case_unlock_printer():
    """`entry unlock --push-evidence` (audit M4): with the freeze record's push evidence the would-be record set
    reaches second_session (this world already records its unlock entry); without it, or with evidence of a push after
    the end of the freeze date, or for a freeze record whose row a later commit entered (DL-24 R1-F8),
    freeze_entry_late; the evidence is the unlock entry's alone (entry_field)."""
    S = mods()[6]
    u = unlocked()
    with stand_ins(u):
        late = scratch_file("tmanpe", "pe.json", S.json_file_bytes(u.push_evidence("2026-10-31T17:00:00Z",
                                                                                  tag="plate", when=u.tick())))
        cases = [("printer_unlock_with_the_freeze_push_evidence_passes_M4", "unlock", UNLOCK_DAY, u.unlock_fields,
                  ["--push-evidence", u.pe], "second_session", "", u.tree),
                 ("printer_unlock_without_the_freeze_push_evidence_M4", "unlock", UNLOCK_DAY, u.unlock_fields, [],
                  "freeze_entry_late", "", u.tree),
                 ("printer_push_evidence_is_the_unlock_entry_s_M4", "clock_offset", "2026-11-05", CLOCK,
                  ["--push-evidence", u.pe], "entry_field", "", u.tree),
                 ("printer_freeze_pushed_after_the_end_of_its_date_M4", "unlock", UNLOCK_DAY, u.unlock_fields,
                  ["--push-evidence", late], "freeze_entry_late", "the end of its window 2026-10-31T16:00:00Z",
                  u.repo)]
        for name, kind, date, fields, extra, want, detail, root in cases:
            rc, out, made = u.print_entry(kind, date, fields, nn=5, extra=extra, root=root)
            check(name, rc == 2 and f"REFUSED [{want}]" in out and detail in out and made is None,
                  f"rc {rc}: {out.strip()[-300:]}")
        # the freeze record re-entered, correcting the first (no second freeze: DL-24 N9), its row a commit later
        split = u.variant("freeze", lambda d: d.update(corrects={"path": u.freeze["path"],
                                                                 "sha256": u.freeze["sha256"]}), 14, split=True)
        moved = dict(u.unlock_fields, freeze_record={"path": split["path"], "sha256": split["sha256"]})
        rc, out, made = u.print_entry("unlock", UNLOCK_DAY, moved, nn=5, extra=["--push-evidence", u.pe], root=u.repo)
        check("printer_freeze_row_entered_by_a_later_commit_R1_F8", rc == 2 and "REFUSED [freeze_entry_late]" in out
              and "is entered by the commit that adds the record" in out and made is None,
              f"rc {rc}: {out.strip()[-300:]}")


def case_unlock_gates():
    """B2R7's unlock gates, each reached with every earlier gate passing: a freeze record whose freeze date is not
    the day after its D (freeze_entry_invalid), one whose conditions are not the recomputed ones
    (am22_condition_unmet), one whose row a later commit entered, an unlock entry without the freeze record's push
    evidence, one whose evidence shows the push after the end of the freeze date and one whose list names the record
    under "unlock_entry" (freeze_entry_late, audit M4; DL-24 R1-F8, R2-12); a record carrying a confirmed push passes,
    its conditions recomputed with it (audit M2). Two freeze records of item 1(b) entered: the validator refuses
    freeze_source_conflict at its own gate in freeze mode and lists it MISSING in a draft (B2R6; DL-24 R2-13). Each
    variant is committed after the unlock entry's commit."""
    A, C, D, E, M, R, S = mods()
    from src.tman import validate as V
    u = unlocked()

    def drop(d):
        d["conditions"][3]["records"] = d["conditions"][3]["records"][1:]
    with stand_ins(u):
        bad = u.variant("freeze", lambda d: d.update(d_date="2026-10-29"), 11)
        VW.expect_gate(u, "unlock_freeze_date_not_d_plus_1_refused_B2R7", "freeze_entry_invalid", mode="unlock",
                       detail="its freeze date is D + 1", head=bad["commit"], freeze_record=bad["path"],
                       unlock_entry=None)
        code, passed, e = u.run("freeze", head=bad["commit"], freeze_record=None, unlock_entry=None)
        check("two_item_1b_records_refused_at_their_gate_in_freeze_mode_R2_13", code == "freeze_source_conflict"
              and passed == VW.gate_codes("freeze")[:VW.gate_codes("freeze").index("freeze_source_conflict")],
              f"{code} {len(passed or [])}: {e.detail[:200] if e else ''}")
        try:
            res = V.validate(u.ctx("draft", head=bad["commit"], freeze_record=None, unlock_entry=None))
            got = [x for x in res.missing if x.startswith("MISSING: freeze_source_conflict")]
        except S.TmanRefused as e:
            got = [f"[{e.code}] {e.detail[:200]}"]
        check("two_item_1b_records_missing_in_a_draft_R2_13", len(got) == 1 and got[0].startswith("MISSING"), str(got))
        unmet = u.variant("freeze", drop, 12)
        VW.expect_gate(u, "unlock_freeze_conditions_not_recomputed_refused_B2R7", "am22_condition_unmet",
                       mode="unlock", detail="are not the ones recomputed for D", head=unmet["commit"],
                       freeze_record=unmet["path"], unlock_entry=None)
        conds = A.freeze_conditions(u.evaluate(D_WORLD, pushed={u.H_TRIGGER: "2026-10-23"}))
        pv = u.variant("freeze", lambda d: d.update(conditions=conds), 16)   # before a second unlock entry is entered
        code, _, e = u.run("unlock", head=pv["commit"], freeze_record=pv["path"], unlock_entry=None)
        check("freeze_record_with_a_confirmed_push_recomputed_with_it_M2", code is None and any(
            r["kind"] == A.CONFIRMED_PUSH for r in conds[1]["records"]), f"{code}: {e.detail[:300] if e else ''}")
        none = u.variant("unlock", lambda d: d.update(push_evidence=None), 2)
        VW.expect_gate(u, "unlock_entry_without_the_freeze_push_evidence_refused_M4", "freeze_entry_late",
                       mode="unlock", detail="cites no push evidence", head=none["commit"], unlock_entry=none["path"])
        late = u.push_evidence("2026-10-31T17:00:00Z", tag="late", when=u.tick())
        lt = u.variant("unlock", lambda d: d.update(push_evidence=late), 3)
        VW.expect_gate(u, "freeze_record_pushed_after_the_end_of_its_date_refused_M4", "freeze_entry_late",
                       mode="unlock", detail="the end of its window 2026-10-31T16:00:00Z", head=lt["commit"],
                       unlock_entry=lt["path"])
        keyed = u.push_evidence(PUSHED, tag="keyed", when=u.tick(), key="unlock_entry")
        kv = u.variant("unlock", lambda d: d.update(push_evidence=keyed), 4)
        VW.expect_gate(u, "freeze_list_naming_the_record_under_unlock_entry_refused_R2_12", "freeze_entry_late",
                       mode="unlock", detail="not 'unlock_entry'", head=kv["commit"], unlock_entry=kv["path"])
        split = u.variant("freeze", lambda d: d["printed_by"].update(commit=u.H_DRAFT), 15, split=True)
        VW.expect_gate(u, "freeze_row_entered_by_a_later_commit_refused_R1_F8_R2_7", "freeze_entry_late",
                       mode="unlock", detail="decision-log row is entered by", head=split["commit"],
                       freeze_record=split["path"], unlock_entry=None)
        doc = copy.deepcopy(u.freeze["doc"])
        doc["conditions"][0]["records"].append({"kind": "a later record", "ref": u.freeze["path"],
                                                "sha256": u.freeze["sha256"], "date": D_WORLD})
        why = A.freeze_record_error(u.records, u.cat, u.runs, doc, u.doc["body"], None)
        check("freeze_record_citing_a_record_committed_after_d_B2R7", "records not committed by the end of D "
              "2026-10-30" in str(why), str(why))


def case_freeze_entry_units():
    """freeze_entry_invalid's field relations (B2R5, B2R7; audit m8), am22_condition_unmet's recorded conditions
    (audit M2) and its re-check at a D whose conditions do not hold (DL-24 R3-05, R3-06), each on its own."""
    A, C, D, E, M, R, S = mods()
    w = world()
    with stand_ins(w):
        chk = w.evaluate(D_WORLD)
        conds = A.freeze_conditions(chk)
        at29 = A.freeze_conditions(w.evaluate("2026-10-29"))
        why29 = A.condition_unmet_error(w.records, w.cat, w.runs, {"d_date": "2026-10-29", "conditions": at29},
                                        reader=w.trigger_reader(), registered_cells=PINS.IF6_CELLS)
    good = {"source": D.AM22_SOURCE_1B, "date": F_WORLD, "freeze_date": F_WORLD, "d_date": D_WORLD,
            "conditions": conds}
    f1 = {"source": D.AM22_SOURCE_1F, "date": A.FALLBACK_FREEZE, "freeze_date": A.FALLBACK_FREEZE, "d_date": None,
          "conditions": None}
    later = [dict(c, records=[dict(r, date=F_WORLD) for r in c["records"]]) for c in conds]
    for name, doc, resolver, text in (
            ("freeze_entry_of_another_source_refused_B2R5", dict(good, source="AM-19 item 1(d)"), None, "is neither"),
            ("item_1f_freeze_naming_d_refused_B2R5", dict(f1, d_date=D_WORLD), None, "names no D and no conditions"),
            ("item_1f_freeze_before_1103_refused_R2_3", dict(f1, date="2026-11-02", freeze_date="2026-11-02"), None,
             "AM-19's as committed"),
            ("item_1f_freeze_off_the_resolver_s_date_refused_m8", dict(f1, date="2026-11-04",
                                                                       freeze_date="2026-11-04"), A.FALLBACK_FREEZE,
             "not the resolver's 2026-11-03"),
            ("item_1b_conditions_misnumbered_refused_B2R5", dict(good, conditions=conds[:4] + [dict(conds[4],
                                                                                                    number=6)]),
             None, "numbered 1 to 5"),
            ("item_1b_record_dated_after_d_refused_B2R7", dict(good, conditions=later), None, "records dated after D")):
        why = A.freeze_entry_error(doc, resolver)
        check(name, text in str(why), str(why))
    check("freeze_entries_of_items_1b_and_1f_pass_their_relations", A.freeze_entry_error(good, None) is None
          and A.freeze_entry_error(f1, A.FALLBACK_FREEZE) is None)
    early = copy.deepcopy(conds)
    next(r for c in early for r in c["records"] if r["kind"] == "lane register")["date"] = "2026-10-12"
    check("recorded_condition_dated_before_its_commits_day_refused_M2", "dated before their commits' day or after D"
          in str(A.conditions_error(early, chk)) and A.conditions_error(conds, chk) is None)
    check("unlock_recheck_at_a_d_whose_conditions_do_not_hold_R3_05", "condition 1 does not hold at the end of D "
          "2026-10-29" in str(why29), str(why29))


def case_coverage():
    """B2R7's gates follow run_added_after_freeze in the unlock order, and each has a case here (smoke_tman_unlock's
    coverage leaves them to this smoke)."""
    from src.tman import validate as V
    codes = [c for c, _, _ in V.GATES]
    i = codes.index("run_added_after_freeze")
    check("b2r7_gates_follow_run_added_after_freeze", tuple(codes[i + 1:i + 5]) == UL.AM22_GATES, str(codes[i:i + 6]))
    check("every_b2r7_gate_has_a_case", set(UL.AM22_GATES) <= VW.WANTED, str(sorted(set(UL.AM22_GATES) - VW.WANTED)))


CASES = (case_static, case_trigger_reader, case_registered, case_settled_ways, case_commit_at_end, case_providers,
         case_conditions_in_the_world, case_condition_5, case_condition_3, case_condition_4_rows_at_c_x,
         case_condition_2_cases, case_report_lines, case_cli, case_confirmed_push, case_cli_am22_d, case_d_first,
         case_fallback_1f, case_unlock_world, case_freeze_printer, case_freeze_body, case_measurement_cut_printer,
         case_unlock_printer, case_unlock_gates, case_freeze_entry_units, case_coverage)


def main() -> int:
    try:
        return FX.run_cases("TMAN AM-22 SMOKE (lane TMAN, B-2)", CASES)
    finally:
        from src.tman import manifest as M
        if VW.REAL_TEACHER is not None:
            M.TEACHER_OF_RECORD.clear()
            M.TEACHER_OF_RECORD.update(VW.REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(main())
