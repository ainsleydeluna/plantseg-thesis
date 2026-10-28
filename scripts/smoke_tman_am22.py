#!/usr/bin/env python3
"""Smoke for AM-22 item 1(a)'s check (src/tman/am22.py; lane TMAN, B-2): the registered runs and every exclusion of
B2R3, every settled way with a never-cut run's limits, the records committed by the end of a date (C_X, audit M3),
the conditions MET and MISSING with their REPORT lines (the boundary day, audit M2; a late trigger), `check-freeze`
on the command line (exit codes; a date whose end has not passed, audit m6; a date after 2026-11-02, item 1(f)), and
the static cases of B2R11 and B2R13.

The fixture world (`AmWorld`) is smoke_tman_validate's World on AM-22's timeline: its runs of record complete by
2026-10-20, every other cuttable registered run cut by item 2(a)'s cutoff -- on 2026-10-20, entered the next day, R2
at its ceiling 2026-10-29, entered on 2026-10-30 -- with its children not run, AM-3's trigger entered on 2026-10-22
(not triggered), the lane register and the draft committed on 2026-10-30. Smaller fixtures (`Mini`) put one question
each: a settled way, an exclusion, a condition's record.

Synthetic inputs only, under the system temp dir (removed at the end); TEST is never read.

Run:  python -B scripts/smoke_tman_am22.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import contextlib  # noqa: E402
import io  # noqa: E402
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

# AM-22's timeline in the fixture (UTC; each instant before 16:00, so its Manila day is its UTC day)
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
T_R2 = "2026-10-30T01:00:00+00:00"            # R2's cut at its ceiling entered, the lane register
T_DRAFT = "2026-10-30T03:00:00+00:00"         # the draft
WORLD_PTQ_MADE = "2026-11-02T10:00:00Z"       # smoke_tman_validate.World.ptq's time, moved to PTQ_MADE here
NOW = "2026-11-06T00:00:00Z"                  # --now
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
REASONS = {"L": "code not merged", "E6-E2": "code not merged"}
ARMS = ("R1", "R2", "W", "L")
R2 = ["teacher-R2:s42:40k", "E3-R2:s42:lambda0.5:alpha25:80k"]


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


# ================================================================================================ the AM-22 world
class AmWorld(VW.World):
    """smoke_tman_validate's World on AM-22's timeline (module docstring)."""

    def __init__(self):
        A, C, D, E, M, R, S = mods()
        self.S, self.M, self.R = S, M, R
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
        self.cutoff_entries({"R2": (R2, "2026-10-29")}, "2026-10-30")
        self.lane_register()
        self.H_R2 = self.commit_dl(self.files, "R2's cut at its ceiling; the lane register", T_R2)
        self.records = R.Records.at_head(self.repo)
        self.sources = self.make_sources()
        self.doc = M.build_manifest(self.records, self.cat, self.sources, runs_root=self.runs)
        self.data = S.json_file_bytes(self.doc)
        self.H_DRAFT = FX.commit_files(self.repo, {S.MANIFEST_REL: self.data}, "the draft", date=T_DRAFT)
        self.records = R.Records.at_head(self.repo)

    def commit_dl(self, files: dict, msg: str, date: str) -> str:
        """Commit `files` with the decision log as it stands; the next files start empty."""
        files = dict(files)
        files[self.S.DECISION_LOG_REL] = FX.dl_table(dict(sorted(self.dl.items(), key=lambda kv: int(
            kv[0].split("-")[1])))).encode()
        sha = FX.commit_files(self.repo, files, msg, date=date)
        self.files = {}
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
        for group, (slots, day) in groups.items():
            fields = {"rule": "cutoff", "reason": REASONS.get(group, "gate not met"), "launched": False, "run_ids": [],
                      "since": {"date": "2026-10-20", "record": dict(mde)}, "arithmetic": None, "telemetry": [],
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

    def lane_register(self):
        """The lane register: two lanes IMPLEMENTED (their rows naming them and their merge commits), one not
        required (its entry citing E1-160k's cut), and L-AM16-ARM with no status a reader could accept."""
        S = self.S
        cut = self.cut_refs["E1:s42:160k"]
        nr = self.entry_file("not_required", ["E1:s42:160k"], {"lane": "L-LAT-THREADS", "cut_entries": [
            {"path": cut["path"], "sha256": cut["sha256"]}]}, date="2026-10-30")
        self.am3_dl = self.row(f"L-AM3 merged at {self.H_SEL[:7]}: the AM-3 trigger lane", "IMPLEMENTED (2026-10-25)")
        self.q_dl = self.row(f"L-AM23Q merged at {self.E[:7]}: the E6-E2 and E6-X lane", "IMPLEMENTED (2026-10-25)")
        self.lanes = {"format": S.LANES_FORMAT, "lanes": [
            {"lane": "L-AM3", "serves": ["E6-KD:s42:15ep", "E6-KD:s42:sel"],
             "status": {"kind": "implemented", "dl": self.am3_dl, "merge_commit": self.H_SEL, "entry": None}},
            {"lane": "L-AM23Q", "serves": ["E6-E2:s42:15ep", "E7-E2:s42:list42"],
             "status": {"kind": "implemented", "dl": self.q_dl, "merge_commit": self.E, "entry": None}},
            {"lane": "L-LAT-THREADS", "serves": ["E1:s42:160k"],
             "status": {"kind": "not_required", "dl": None, "merge_commit": None,
                        "entry": {"path": nr["path"], "sha256": nr["sha256"]}}},
            {"lane": "L-AM16-ARM", "serves": ["ARM:latency"],
             "status": {"kind": "implemented", "dl": None, "merge_commit": None, "entry": None}}]}
        data = S.json_file_bytes(self.lanes)
        self.files[S.LANES_REL] = data
        self.lanes_dl = self.row(f"the lane register {S.LANES_REL} sha256 {S.sha256_bytes(data)}")

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
        A = mods()[0]
        return A.evaluate(self.records, self.cat, self.runs, day, reader=self.trigger_reader(), **kw)


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
    """B2R11: am22.py in LANE_CODE_FILES and score-blind; B2R13: L-AM3's reader is the trigger reader."""
    A, C, D, E, M, R, S = mods()
    import scripts.smoke_tman_noscore as NS
    check("am22_is_lane_code_B2R11", "src/tman/am22.py" in M.LANE_CODE_FILES)
    hits = [s for s in NS.code_literals(REPO / "src/tman/am22.py") if NS.SCORE.search(s)]
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
    got = [read(rel, files[rel]) for rel in ("reports/derived/trig_fired.json", "reports/derived/trig_quiet.json")]
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


def case_commit_at_end():
    """C_X is the newest commit on the records ref's first-parent chain whose committer time meets the end of X."""
    A, C, D, E, M, R, S = mods()
    m = Mini()
    a = m.commit({}, "2026-10-20T02:00:00+00:00")
    b = m.commit({}, "2026-10-19T02:00:00+00:00")           # a commit dated before its parent
    c = m.commit({}, "2026-10-21T15:59:59+00:00")
    src = R.Records.at_head(m.repo).src
    got = [A.commit_at_end(src, day) for day in ("2026-10-11", "2026-10-19", "2026-10-20", "2026-10-21")]
    check("c_x_walks_back_from_the_tip_M3", got == [None, b, b, c], str([x and x[:7] for x in got]) + f" a {a[:7]}")


def case_conditions_in_the_world():
    """The AM-22 world: conditions 1-4 MET at the end of 2026-10-30 with their records; MISSING on earlier dates for
    what was not entered yet."""
    A, C, D, E, M, R, S = mods()
    w = WORLD
    chk = w.evaluate("2026-10-30")
    c = {x.number: x for x in chk.conditions}
    check("world_conditions_1_to_4_met_at_1030", all(c[n].met for n in (1, 2, 3, 4)),
          str({n: c[n].missing[:2] for n in (1, 2, 3, 4) if not c[n].met}))
    regs = chk.view.registered()[0]
    check("world_registers_40_runs_and_settles_each", len(regs) == 40 and not c[1].missing, str(len(regs)))
    ways = sorted({r.kind.split(" ")[0] for r in c[1].records})
    check("world_settles_by_completion_cut_and_not_run", ways == ["AM-3", "completion", "cut", "not-run"], str(ways))
    early = {x.number: x for x in w.evaluate("2026-10-29").conditions}
    miss = sorted(x.split(": ")[0] for x in early[1].missing)
    check("r2_not_cut_by_1029_condition_1_missing", miss == sorted(R2 + ["E6-R2:s42:15ep", "teacher:s42:40k"])
          and any(x.startswith("teacher:s42:40k: its run is named by no launcher") for x in early[1].missing),
          str(miss))
    check("lane_register_not_committed_by_1029_condition_4_missing", not early[4].met
          and "not committed by the end of 2026-10-29" in early[4].missing[0], str(early[4].missing))
    d21 = {x.number: x for x in w.evaluate("2026-10-21").conditions}
    check("trigger_not_entered_by_1021_condition_2_missing", not d21[2].met
          and "is not committed by the end of 2026-10-21" in d21[2].missing[0], str(d21[2].missing))
    d22 = {x.number: x for x in w.evaluate("2026-10-22").conditions}
    check("trigger_entered_1022_read_for_triggered_draft_not_yet", not any("trigger" in m for m in d22[2].missing)
          and any("are not made by the end" in m for m in d22[2].missing), str(d22[2].missing))
    check("draft_not_committed_by_1029_condition_3_missing", not early[3].met
          and "is not committed by the end of 2026-10-29" in early[3].missing[0], str(early[3].missing))
    check("condition_3_reads_the_draft_s_e6s42_hashed_block", c[3].met and any(
        r.kind == "E6:s42:sel hashed in the draft" for r in c[3].records))
    lanes = [r.kind for r in c[4].records]
    check("condition_4_reads_implemented_and_not_required_lanes_not_arm_1g", "lane L-AM3 IMPLEMENTED" in lanes
          and "lane L-LAT-THREADS not required" in lanes and not any("L-AM16-ARM" in k for k in lanes)
          and any("L-AM16-ARM is not read" in n for n in c[4].notes), str(lanes))


def case_condition_4_rows_at_c_x():
    """M3: a lane whose row turns IMPLEMENTED after X is not IMPLEMENTED by X; its merge commit an ancestor of C_X."""
    A, C, D, E, M, R, S = mods()
    m = Mini()
    base = m.commit({}, "2026-10-20T02:00:00+00:00")
    lane_dl = m.row(f"L-X merged at {base[:7]}")
    lanes = {"format": S.LANES_FORMAT, "lanes": [{"lane": "L-X", "serves": ["G:s42:80k"], "status": {
        "kind": "implemented", "dl": lane_dl, "merge_commit": base, "entry": None}}]}
    data = S.json_file_bytes(lanes)
    m.row(f"the lane register sha256 {S.sha256_bytes(data)}")
    m.commit({S.LANES_REL: data}, "2026-10-21T02:00:00+00:00")
    m.dl[lane_dl] = ("IMPLEMENTED (2026-10-22)", m.dl[lane_dl][1])
    m.commit({}, "2026-10-22T02:00:00+00:00")
    at21, at22 = A.condition_4(m.view("2026-10-21")), A.condition_4(m.view("2026-10-22"))
    check("lane_row_implemented_after_x_is_missing_at_x_M3", not at21.met and at22.met,
          f"{at21.missing} / {at22.missing}")
    lanes["lanes"][0]["status"]["merge_commit"] = "f" * 40
    data = S.json_file_bytes(lanes)
    m.row(f"the lane register sha256 {S.sha256_bytes(data)}")
    m.commit({S.LANES_REL: data}, "2026-10-23T02:00:00+00:00")
    check("lane_merge_commit_not_in_the_records_is_missing", not A.condition_4(m.view("2026-10-23")).met)


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
    check("e6s42_made_artifacts_read_from_the_draft", any(r.kind == "derived artifacts of E6:s42:15ep"
                                                          for r in c2.records))
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


def case_report_lines():
    """M2: every committed record's commit and committer time, and a boundary-day REPORT for each made on X or X-1."""
    A = mods()[0]
    out = A.lines(WORLD.evaluate("2026-10-30"))
    rep = [x for x in out if x.startswith("REPORT boundary-day entry")]
    shas = {WORLD.H_R2, WORLD.H_DRAFT}
    check("boundary_day_reports_name_the_commits_of_x_M2", rep and all(any(s in x for s in shas) for x in rep)
          and not any(WORLD.H_CUTS in x for x in rep), str(rep[:3]))
    check("records_print_their_commits_M2", any(x.startswith("  RECORD cut entry of G:s42:80k (cutoff)")
                                                and WORLD.H_CUTS[:12] in x and "2026-10-21" in x for x in out))
    check("result_line_names_the_conditions_missing", out[-1] == "RESULT: MISSING (condition 5 at the end of "
                                                                 "2026-10-30)", out[-1])


def case_cli():
    """`check-freeze`: exit 3 while a condition is MISSING; a date whose end has not passed (m6), one after
    2026-11-02 (item 1(f)) or before AM-22 is in force refused."""
    A, C, D, E, M, R, S = mods()
    import scripts.tday_manifest as CLI
    saved = R.TRIGGER_READER
    try:
        R.TRIGGER_READER = VW.World.trigger_reader()
        base = ["--root", str(WORLD.repo), "check-freeze", "--runs", str(WORLD.runs)]
        rc, out = cli(CLI, base + ["--date", "2026-10-30", "--now", NOW])
        check("cli_check_freeze_prints_conditions_exit_3", rc == 3 and "CONDITION 1 MET" in out
              and "CONDITION 5 MISSING" in out, out[-300:])
        rc, out = cli(CLI, base + ["--date", "2026-10-30", "--now", "2026-10-30T15:59:59Z"])
        check("cli_date_not_ended_refused_m6", rc == 2 and "am22_date_not_ended" in out, out[-200:])
        rc, out = cli(CLI, base + ["--date", "2026-11-03", "--now", NOW])
        check("cli_date_after_1102_refused_1f", rc == 2 and "am22_d_not_first" in out and "item 1(f)" in out,
              out[-200:])
        rc, out = cli(CLI, base + ["--date", "2026-10-16", "--now", NOW])
        check("cli_date_before_am22_refused", rc == 2 and "am22_d_not_first" in out, out[-200:])
    finally:
        R.TRIGGER_READER = saved


def cli(CLI, argv: list) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CLI.main(argv)
    return rc, buf.getvalue()


WORLD: AmWorld | None = None


def main() -> int:
    global WORLD
    try:
        WORLD = AmWorld()
    except Exception as e:  # noqa: BLE001
        check("am22_world_built", False, f"{type(e).__name__}: {e}")
        return FX.run_cases("TMAN AM-22 SMOKE (lane TMAN, B-2)", ())
    try:
        return FX.run_cases("TMAN AM-22 SMOKE (lane TMAN, B-2)",
                            (case_static, case_trigger_reader, case_registered, case_settled_ways, case_commit_at_end,
                             case_conditions_in_the_world, case_condition_4_rows_at_c_x, case_condition_2_cases,
                             case_report_lines, case_cli))
    finally:
        from src.tman import manifest as M
        if VW.REAL_TEACHER is not None:
            M.TEACHER_OF_RECORD.clear()
            M.TEACHER_OF_RECORD.update(VW.REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(main())
