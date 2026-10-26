#!/usr/bin/env python3
"""Smoke for the validator (plan §2.2; lane TMAN, B-1, C5): one case per gate that runs in freeze mode, each reached
with every earlier gate passing (the refusal's `passed` list is the gates before it, in order); draft mode's
MISSING lines (exit 3) never refused; the builder's draft passing; the CLI's validate command.

The fixture world (`World`) is a scratch repository and run directories in which every catalogue slot that AM-19
asks about is entered: the runs of record complete with their checkpoints (the teacher, E1-E3 at seed 42, the sweep
candidates, E1 at seeds 43/44, the U4 pilot, E6 at seed 42), their derived models hashed at the freeze (E4, E7, the
QAT selections; a model of record's ptq_record read through a stand-in for lane LQP's reader, A2), E7-E2 not made,
fifteen cut entries at the freeze (one per cut group), six not-run entries and the KD launch log. The other TMAN
smokes import it. The teacher of record's values are a fixture's for the run (TEACHER_OF_RECORD is patched in this
process and restored); a static case compares the real values with scripts/preflight_distill.py's.

Synthetic inputs only, under the system temp dir (removed at the end); TEST is never read.

Run:  python -B scripts/smoke_tman_validate.py
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

check, expect_refused = FX.check, FX.expect_refused
E3S42 = "E3:s42:lambda0.5:alpha25:80k"
CODE_DATE, RECORDS_DATE, MANIFEST_DATE = ("2026-11-01T08:00:00+00:00", "2026-11-03T10:00:00+00:00",
                                         "2026-11-03T12:00:00+00:00")
FREEZE_DAY = "2026-11-03"
QAT_RULES_SHA256 = "76addc57d0367dda9a1163f749ec32c9c240b20736104dc12aa77488344f9568"
REAL_FILES = ("configs/tday_catalogue.json", "configs/kd_schedule.json",
              *[f"configs/calibration/ptq_calibration_seed{s}.json" for s in (42, 43, 44, 45)],
              "reports/derived/mde_entry_20261009T034825Z.json", "reports/strata/train_strata_v1.json")
CODE_FIXTURES = {"scripts/evaluate_model.py": "import src.eval.fixture_scoring  # the evaluator's fixture root\n",
                 "src/eval/__init__.py": "", "src/eval/fixture_scoring.py": "VALUE = 1\n",
                 "configs/plantseg_class_map.json": "{}\n",
                 "scripts/build_corruption_cache.py": "# the cache generator's fixture\n",
                 "scripts/tday_hash_images.py": "# the in-session hasher's fixture\n"}
CUT_GROUPS = {   # group -> (slots cut at the freeze, reason, arm)
    "E2-s4344": (["E2:s43:lambda0.5:80k", "E2:s44:lambda0.5:80k"], "no card", False),
    "E3-s4344": (["E3:s43:lambda0.5:alpha25:80k", "E3:s44:lambda0.5:alpha25:80k"], "no card", False),
    "E5E6-s4344": (["E5:s43:15ep", "E5:s44:15ep"], "gate not met", False),
    "E1-160k": (["E1:s42:160k"], "no card", False), "E2-160k": (["E2:s42:lambda0.5:160k"], "no card", False),
    "E3-160k": (["E3:s42:lambda0.5:alpha25:160k"], "no card", False),
    "A": (["A:s42:alpha25:80k"], "no card", False), "F": (["F:s42:alpha25:80k"], "no card", False),
    "G": (["G:s42:80k"], "no card", False),
    "R2": (["teacher-R2:s42:40k", "E3-R2:s42:lambda0.5:alpha25:80k"], "gate not met", True),
    "R1": (["teacher-R1:s42:40k", "E3-R1:s42:lambda0.5:alpha25:80k"], "gate not met", True),
    "W": (["E3-flip:s42:lambda0.5:alpha25:80k"], "gate not met", True),
    "L": (["teacher-L:s42:40k", "E2-L:s42:lambda0.5:80k", "E3-L:s42:lambda0.5:alpha25:80k"], "code not merged", True),
    "E6-E2": (["E6-E2:s42:15ep"], "code not merged", False), "ARM": (["ARM:latency"], "code not merged", False)}
NOT_RUN = {"E3:s43:lambda0.5:alpha25:80k": ["E6:s43:15ep", "E7:s43:list42"],
           "E3:s44:lambda0.5:alpha25:80k": ["E6:s44:15ep", "E7:s44:list42"],
           "E3-R2:s42:lambda0.5:alpha25:80k": ["E6-R2:s42:15ep", "E7-R2:s42:list42"],
           "E3-R1:s42:lambda0.5:alpha25:80k": ["E6-R1:s42:15ep", "E7-R1:s42:list42"],
           "E3-flip:s42:lambda0.5:alpha25:80k": ["E7-flip:s42:list42"],
           "E3-L:s42:lambda0.5:alpha25:80k": ["E7-L:s42:list42"]}
KD_RUNS = {  # slot -> (kind, run_id)
    "E2:s42:lambda0.25:80k": ("listed_run", "e2_s42_lambda0.25_80k_a1"),
    "E2:s42:lambda0.5:80k": ("scored_run", "e2_s42_lambda0.5_80k_a1"),
    "E2:s42:lambda1:80k": ("default_candidate", "e2_s42_lambda1_80k_a1"),
    "E2:s42:lambda2:80k": ("listed_run", "e2_s42_lambda2_80k_a1"),
    "E2:s42:lambda4:80k": ("listed_run", "e2_s42_lambda4_80k_a1"),
    E3S42: ("scored_run", "e3_s42_lambda0.5_alpha25_80k_a1"),
    "E3:s42:lambda0.5:alpha50:80k": ("default_candidate", "e3_s42_lambda0.5_alpha50_80k_a1"),
    "E3:s42:lambda0.5:alpha100:80k": ("listed_run", "e3_s42_lambda0.5_alpha100_80k_a1")}
REAL_TEACHER = None


class World:
    """The fixture world; `doc` is the committed manifest (freeze mode passes every gate on it)."""

    def __init__(self):
        from src.tman import manifest as M
        from src.tman import records as R
        from src.tman import schema as S
        self.S, self.M, self.R = S, M, R
        self.repo, self.runs = FX.new_repo(), FX.scratch_dir("tmanruns")
        self.cat = S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())
        self.dl, self.next_dl, self.files = {}, 200, {}
        self.entries: dict = {}
        self.ptq_docs: dict = {}
        self.launch: list = []
        self.cut_refs: dict = {}
        for rel in REAL_FILES:
            self.files[rel] = (REPO / rel).read_bytes()
        self.files.update({k: v.encode() for k, v in CODE_FIXTURES.items()})
        self.E = FX.commit_files(self.repo, dict(self.files), "code of record (E)", date=CODE_DATE)
        self.files = {}
        self.teacher_fixture()
        self.static_rows()
        self.complete_runs()
        self.derived_models()
        self.terminal_entries()
        self.selection_files()
        self.files["reports/derived/am3_trigger.json"] = S.json_file_bytes({"format": "am3_trigger/1",
                                                                            "triggered": False})
        self.files["reports/kd_launch_log.jsonl"] = b"".join(json.dumps(x).encode() + b"\n" for x in self.launch)
        self.files[S.DECISION_LOG_REL] = FX.dl_table(dict(sorted(self.dl.items(), key=lambda kv: int(
            kv[0].split("-")[1])))).encode()
        self.H1 = FX.commit_files(self.repo, self.files, "records", date=RECORDS_DATE)
        self.records = R.Records.at_head(self.repo)
        self.sources = self.make_sources()
        self.doc = M.build_manifest(self.records, self.cat, self.sources, runs_root=self.runs)
        self.data = S.json_file_bytes(self.doc)
        self.H2 = FX.commit_files(self.repo, {S.MANIFEST_REL: self.data}, "manifest", date=MANIFEST_DATE)

    # ------------------------------------------------------------------------------------------- records
    def row(self, text: str, status: str = "RECORDED") -> str:
        dl = f"DL-{self.next_dl}"
        self.next_dl += 1
        self.dl[dl] = (status, text)
        return dl

    def teacher_fixture(self):
        global REAL_TEACHER
        d = FX.teacher_run(self.runs, "teacher_s42_40k")
        data = b"teacher of record (fixture)"
        (d / "iter_24000.pth").write_bytes(data)
        if REAL_TEACHER is None:
            REAL_TEACHER = dict(self.M.TEACHER_OF_RECORD)
        self.M.TEACHER_OF_RECORD.update(sha256=self.S.sha256_bytes(data), bytes=len(data))

    def static_rows(self):
        S, M = self.S, self.M
        t, h = M.TEACHER_OF_RECORD, M.TRAINVAL_HASH_LIST
        mde, strata = (S.sha256_bytes((REPO / rel).read_bytes()) for rel, _ in M.REFERENCE_FILES.values())
        self.dl.update({"DL-10": ("RECORDED", "calibration lists of record"),
                        "DL-63": ("RECORDED", f"strata file of record (sha256 {strata})"),
                        "DL-70": ("RECORDED", "AM-19 schedule T"),
                        "DL-88": ("RECORDED", f"teacher of record {t['path']} sha256 {t['sha256']} {t['bytes']} B"),
                        "DL-91": ("RECORDED", f"{h['path']} sha256 {h['sha256']}, {h['rows']:,} rows"),
                        "DL-99": ("RECORDED", f"MDE entry sha256 {mde}")})

    def ckpt(self, run_dir: Path, slot: str, name: str) -> dict:
        data = f"checkpoint of {slot}".encode()
        (run_dir / name).write_bytes(data)
        sha = self.S.sha256_bytes(data)
        dl = self.row(f"{slot} ends; checkpoint {name} sha256 {sha}")
        return {"name": name, "sha256": sha, "bytes": len(data),
                "sources": [{"kind": "run_dir", "path": f"{run_dir.name}/{name}", "dl": dl, "sha256": sha}]}, dl

    def run_entry(self, slot: str, kind: str, run_id: str, ck=None, dl=None) -> dict:
        s = self.cat.slots[slot]
        e = {"kind": kind, "dl_rows": [dl] if dl else [], "run_id": run_id,
             "attempts": [{"run_id": run_id, "attempt": 1}]}
        if kind == "listed_run":
            e.update(checkpoint=ck, reason="a listed run: not scored (AM-17 item 6; TB9)")
        else:
            e.update(checkpoint=ck, evaluator_stage=s["evaluator_stage"], teacher_ckpt_sha256=None)
        if kind == "default_candidate":
            e["selection"] = None                       # set in selection_files
        self.entries[slot] = e
        return e

    def complete_runs(self):
        for slot, (kind, rid) in KD_RUNS.items():
            s = self.cat.slots[slot]
            d = FX.kd_run(self.runs, rid, stage_key=s["stage"].lower())
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
        for slot, rid, stage, clip in (("E5:s42:clip1:15ep", "e5_s42_clip1_15ep", "E5", 1.0),
                                       ("E5:s42:clip5:15ep", "e5_s42_clip5_15ep", "E5", 5.0),
                                       ("E6:s42:15ep", "e6_s42_15ep", "E6", 5.0)):
            FX.qat_run(self.runs, rid, stage=stage, clip_norm=clip)
            self.run_entry(slot, "listed_run", rid, None, self.row(f"{slot} completes"))

    def launch_lines(self, slot: str, rid: str, d: Path):
        s = self.cat.slots[slot]
        lam, alpha = s["params"].get("lambda"), s["params"].get("alpha")
        sweep = s["sweep"]
        self.launch.append({"format": "kd_launch_log/1", "event": "launch", "run_id": rid, "stage": s["stage"],
                            "seed": s["seed"], "horizon": 80000, "sweep": sweep["key"] if sweep else None,
                            "value": float(sweep["value"]) if sweep else None,
                            "lambda_logit": None if lam is None else float(lam),
                            "alpha_cwd": None if alpha is None else float(alpha), "attempt": 1, "schedule": "T",
                            "code_pin": self.E, "ckpt_dir": f"/workspace/kd/{rid}", "am8a_report": None})
        meta = d / f"{s['stage'].split('-')[0].lower()}_run_meta.jsonl"
        self.launch.append({"event": "launched", "run_id": rid, "run_meta_sha256": self.S.sha256_bytes(
            meta.read_bytes()), "launch_time_utc": FX.utc("2026-10-10T00:00:00Z"), "git_head": self.E,
            "records_commit": self.E, "decision_date": None, "launch_order_position": None})

    def ptq(self, slot: str, parent: str, list_seed: int, of_record: bool) -> dict:
        S = self.S
        s = self.cat.slots[slot]
        key = S.file_id_of(slot)
        cal = f"configs/calibration/ptq_calibration_seed{list_seed}.json"
        prov = S.json_file_bytes({"schema": "ptq_run_meta/fixture", "stage": s["evaluator_stage"],
                                  "calibration": {"seed": list_seed, "data_root": "/workspace/plantseg_trainval"}})
        (self.runs / key).mkdir(exist_ok=True)
        (self.runs / key / "ptq_run_meta.json").write_bytes(prov)
        h = {"artifact_sha256": S.sha256_bytes(f"ts {slot}".encode()),
             "state_dict_sha256": S.sha256_bytes(f"sd {slot}".encode()),
             "x86_copy_sha256": S.sha256_bytes(f"x86 {slot}".encode()),
             "provenance": {"path": f"{key}/ptq_run_meta.json", "sha256": S.sha256_bytes(prov)},
             "record": {"path": f"{key}/ptq_record.json", "sha256": S.sha256_bytes(f"rec {slot}".encode())},
             "made_utc": "2026-11-02T10:00:00Z", "image_digest": S.KD_IMAGE, "isolation": None, "appended": "freeze"}
        e = {"kind": "derived", "dl_rows": [self.row(f"{slot} made")], "method": "ptq", "parent": parent,
             "inputs": {"parent_checkpoint_sha256": self.entries[parent]["checkpoint"]["sha256"],
                        "calibration_list": {"path": cal, "file_sha256": S.sha256_bytes((REPO / cal).read_bytes())},
                        "script": {"path": "scripts/run_ptq.py", "commit": self.E}, "image_digest": S.KD_IMAGE,
                        "qconfig_fingerprint": "0123456789abcdef"},
             "hashed": h, "evaluator_stage": s["evaluator_stage"], "ptq_record": None}
        if of_record:
            doc = FX.ptq_record_doc(stage=s["evaluator_stage"], seed=s["seed"])
            doc["parent"]["checkpoint_sha256"] = e["inputs"]["parent_checkpoint_sha256"]
            doc["calibration"]["checksum_sha256"] = json.loads((REPO / cal).read_text())["checksum_sha256"]
            for k, f in (("torchscript", "artifact_sha256"), ("state_dict", "state_dict_sha256"),
                         ("x86_copy", "x86_copy_sha256")):
                doc["artifacts"][k]["sha256"] = h[f]
            rel = f"{key}/{key}_ptq_record.json"
            self.ptq_docs[rel] = doc
            e["ptq_record"] = {"path": rel, "sha256": S.sha256_bytes(S.json_file_bytes(doc))}
        self.entries[slot] = e
        return e

    def qat_selection(self, slot: str, parent: str):
        S = self.S
        rid = self.entries[parent]["run_id"]
        tel = S.sha256_bytes((self.runs / rid / "qat_telemetry.jsonl").read_bytes())
        self.entries[slot] = {
            "kind": "derived", "dl_rows": [self.row(f"{slot} converted and selected")], "method": "qat_selection",
            "parent": parent, "evaluator_stage": self.cat.slots[slot]["evaluator_stage"], "ptq_record": None,
            "inputs": {"run_slot": parent, "telemetry_sha256": tel, "scripts_commit": self.E,
                       "image_digest": S.KD_IMAGE, "rules_sha256": QAT_RULES_SHA256,
                       "clip_selection_sha256": S.sha256_bytes(b"clip selection")},
            "hashed": {"artifact_sha256": S.sha256_bytes(f"ts {slot}".encode()),
                       "state_dict_sha256": S.sha256_bytes(f"sd {slot}".encode()),
                       "x86_copy_sha256": S.sha256_bytes(f"x86 {slot}".encode()),
                       "provenance": {"path": f"{rid}/eval/e12_provenance.json", "sha256": "1" * 64},
                       "record": {"path": f"{rid}/eval/qat_selection.json", "sha256": "2" * 64},
                       "made_utc": "2026-11-02T10:00:00Z", "image_digest": S.KD_IMAGE, "isolation": None,
                       "appended": "freeze", "evaluations": [{"dir": f"{rid}/eval", "fault_report": None}],
                       "selected_checkpoint": {"path": f"{rid}/epoch_ckpts/e12.pt", "sha256": "3" * 64}}}

    def derived_models(self):
        for seed in (42, 43, 44):
            self.ptq(f"E4:s{seed}:list42", f"E1:s{seed}:80k", 42, True)
        for lst in (43, 44, 45):
            self.ptq(f"E4:s42:list{lst}", "E1:s42:80k", lst, False)
            self.ptq(f"E7:s42:list{lst}", E3S42, lst, False)
        self.ptq("E7:s42:list42", E3S42, 42, True)
        self.qat_selection("E5:s42:sel", "E5:s42:clip5:15ep")
        self.qat_selection("E6:s42:sel", "E6:s42:15ep")

    def entry_file(self, kind: str, slots: list, fields: dict, *, nn: int = 1, date: str = FREEZE_DAY) -> dict:
        from src.tman import entries as E
        doc = E.make_entry(kind, slots, date, fields, amendment="AM-19", script="scripts/tday_manifest.py",
                           commit=self.E)
        rel = self.S.entry_file_rel(E.entry_name(doc, nn))
        data = self.S.json_file_bytes(doc)
        sha = self.S.sha256_bytes(data)
        self.files[rel] = data
        return {"path": rel, "sha256": sha, "dl": self.row(f"TMAN {kind} entry for {slots}: {rel} sha256 {sha}"),
                "doc": doc}

    def cut_fields(self, group: str, slots: list, reason: str, arm: bool) -> dict:
        mde = self.M.REFERENCE_FILES["mde_entry"][0]
        return {"rule": "freeze", "reason": reason, "launched": False, "run_ids": [],
                "since": {"date": "2026-10-20", "record": {"path": mde, "sha256": self.S.sha256_bytes(
                    (REPO / mde).read_bytes())}}, "arithmetic": None, "telemetry": [], "comparator": None,
                "results_known": [{"dl": "DL-63", "path": mde, "sha256": "5" * 64}] if arm else [],
                "cut_order_position": self.cat.position(group), "rule_record": None}

    def terminal_entries(self):
        for group, (slots, reason, arm) in CUT_GROUPS.items():
            ref = self.entry_file("cut", slots, self.cut_fields(group, slots, reason, arm))
            for k in slots:
                self.cut_refs[k] = ref
                self.entries[k] = {"kind": "cut", "dl_rows": [ref["dl"]], "entry": {"path": ref["path"],
                                                                                    "sha256": ref["sha256"]},
                                   "cut_date": FREEZE_DAY, "rule": "freeze", "comparator": None}
        for parent, slots in NOT_RUN.items():
            pref = {"path": self.cut_refs[parent]["path"], "sha256": self.cut_refs[parent]["sha256"]}
            ref = self.entry_file("not_run", slots, {"reason": "parent cut", "parent": parent,
                                                     "parent_record": pref})
            for k in slots:
                self.entries[k] = {"kind": "not_run", "dl_rows": [ref["dl"]], "reason": "parent cut",
                                   "parent": parent, "parent_record": pref,
                                   "entry": {"path": ref["path"], "sha256": ref["sha256"]}}
        ref = self.entry_file("not_made", ["E7-E2:s42:list42"], {"reason": "code not merged", "lane": "L-AM23"})
        self.entries["E7-E2:s42:list42"] = {"kind": "not_made", "dl_rows": [ref["dl"]], "reason": "code not merged",
                                            "entry": {"path": ref["path"], "sha256": ref["sha256"]}}

    def selection_files(self):
        S = self.S
        for key, rel, field, win in (("lambda_logit", "reports/derived/lambda_selection.json", "lambda", 0.5),
                                     ("alpha_cwd", "reports/derived/alpha_selection.json", "alpha", 25.0)):
            cands = []
            for s in self.cat.sweep_slots(key):
                e = self.entries[s["slot"]]
                cands.append({field: float(s["sweep"]["value"]), "run_id": e["run_id"], "best_val": 0.3,
                              "ckpt_sha256": e["checkpoint"]["sha256"], "checkpoint_present": True})
            w = next(c for c in cands if c[field] == win)
            data = S.json_file_bytes({"format": f"{field}_selection/1", "winner": dict(w), "candidates": cands,
                                      "rule_trace": ["best_val"], "band": 0.005})
            self.files[rel] = data
            for s in self.cat.sweep_slots(key):
                if self.entries[s["slot"]]["kind"] == "default_candidate":
                    self.entries[s["slot"]]["selection"] = {"path": rel, "sha256": S.sha256_bytes(data)}

    def make_sources(self) -> dict:
        S = self.S
        blob = FX.git(self.repo, "rev-parse", f"{self.E}:scripts/build_corruption_cache.py")
        pin = PINS.pin(script={"path": "scripts/build_corruption_cache.py", "git_blob": blob}, commit=self.E)
        hasher = {"script": {"path": "scripts/tday_hash_images.py", "commit": self.E,
                             "sha256": S.sha256_bytes(CODE_FIXTURES["scripts/tday_hash_images.py"].encode())},
                  "python": "3.11", "parameters": {"folder": "images/<split>", "suffixes": [".jpg", ".jpeg"]}}
        trig = (self.repo / "reports/derived/am3_trigger.json").read_bytes()
        return {"evaluator_commit": self.E,
                "execution_context": {"model_device": "cpu", "input_devices": ["cpu"],
                                      "determinism_policy_applied": False, "batch_size": 1, "num_workers": 0,
                                      "torch_num_threads": 4, "image_digest": S.KD_IMAGE, "evaluator_commit": self.E,
                                      "host": {"name": "tday-host", "cpu_model": "x86-64 fixture"}, "processes": 2},
                "generator_pins": {"corrupted_cache": pin, "image_hasher": hasher},
                "trigger": {"path": "reports/derived/am3_trigger.json", "sha256": S.sha256_bytes(trig),
                            "triggered": False},
                "freeze_entries": {"delay": None, "new_freeze": None}, "entries": copy.deepcopy(self.entries)}

    # ---------------------------------------------------------------------------------------- running
    def reader(self):
        """A stand-in for lane LQP's read_record over this world's ptq_record documents (A2)."""
        def read(path, expect_sha256=None):
            rel = next((r for r in self.ptq_docs if str(path).endswith(r)), None)
            if rel is None:
                raise OSError(f"{path}: no ptq_record there")
            read_one, sha = FX.ptq_reader_stub(self.ptq_docs[rel])
            return read_one(path, expect_sha256)
        return read

    def ctx(self, mode: str = "freeze", data: bytes | None = None, **kw):
        from src.tman import validate as V
        args = dict(mode=mode, manifest_bytes=self.data if data is None else data, manifest_path="tman_out/m.json",
                    records=self.records, runs_root=self.runs, registered_cells=PINS.IF6_CELLS,
                    ptq_reader=self.reader())
        args.update(kw)
        return V.Ctx(**args)

    def run(self, mode: str = "freeze", doc: dict | None = None, **kw):
        """(code or None, passed gates, Result or None) of the validator on the world, `doc` replacing the
        manifest."""
        from src.tman import validate as V
        data = kw.pop("data", None) if doc is None else self.S.json_file_bytes(doc)
        try:
            return None, V.validate(self.ctx(mode, data, **kw)).passed, None
        except self.S.TmanRefused as e:
            return e.code, getattr(e, "passed", None), e

    def commit(self, files: dict, msg: str = "more records", rows: dict | None = None) -> str:
        """Add records (and decision-log rows) in a new commit; the records accessor reads HEAD."""
        files = dict(files)
        if rows:
            self.dl.update(rows)
        if rows or any(r.startswith("reports/derived/tday_entries/") for r in files):
            files[self.S.DECISION_LOG_REL] = FX.dl_table(dict(sorted(self.dl.items(), key=lambda kv: int(
                kv[0].split("-")[1])))).encode()
        sha = FX.commit_files(self.repo, files, msg, date=MANIFEST_DATE)
        self.records = self.R.Records.at_head(self.repo)
        return sha

    def entry(self, doc: dict, slot: str) -> dict:
        return next(e for e in doc["body"]["entries"] if e["slot"] == slot)

    def rebuild(self, entries: dict | None = None, drop=()) -> dict:
        """The manifest the builder makes from this world's sources, `entries` replacing or adding slots' source
        entries and `drop` removed, from the records committed now; it becomes the world's manifest (not
        committed: freeze mode reads it from --manifest)."""
        src = copy.deepcopy(self.sources)
        src["entries"].update(copy.deepcopy(entries or {}))
        for k in drop:
            src["entries"].pop(k, None)
        self.doc = self.M.build_manifest(self.records, self.cat, src, runs_root=self.runs)
        self.data = self.S.json_file_bytes(self.doc)
        return self.doc

    def committed_cut(self, slots: list, fields: dict, *, nn: int, day: str = FREEZE_DAY) -> tuple[dict, dict]:
        """A cut entry file committed with its decision-log row; (its ref, each slot's source entry citing it)."""
        ref = self.entry_file("cut", slots, fields, nn=nn, date=day)
        self.commit({ref["path"]: self.files[ref["path"]]}, f"cut entry {slots}")
        cmp = fields.get("comparator")
        src = {"kind": "cut", "dl_rows": [ref["dl"]], "entry": {"path": ref["path"], "sha256": ref["sha256"]},
               "cut_date": day, "rule": fields["rule"], "comparator": cmp["slot"] if cmp else None}
        return ref, {k: copy.deepcopy(src) for k in slots}

    def committed_not_run(self, parent: str, parent_ref: dict, slots: list, *, nn: int) -> dict:
        """A not-run entry file for `slots` under the cut parent whose entry file is `parent_ref`, committed; each
        slot's source entry citing it."""
        pref = {"path": parent_ref["path"], "sha256": parent_ref["sha256"]}
        ref = self.entry_file("not_run", slots, {"reason": "parent cut", "parent": parent, "parent_record": pref},
                              nn=nn)
        self.commit({ref["path"]: self.files[ref["path"]]}, f"not-run entry {slots}")
        return {k: {"kind": "not_run", "dl_rows": [ref["dl"]], "reason": "parent cut", "parent": parent,
                    "parent_record": pref, "entry": {"path": ref["path"], "sha256": ref["sha256"]}} for k in slots}

    def launched_fields(self, runs: list, comparator: str = E3S42) -> dict:
        """A launched cut's own fields for `runs` [(slot, run_id)]: the run_ids, their telemetry hashed from the run
        directories, and a comparator with its telemetry (AM-19 item 3(c))."""
        def tel(slot, rid):
            name = self.M.telemetry_name(self.cat.slots[slot])
            return {"run_id": rid, "path": f"{rid}/{name}",
                    "sha256": self.S.sha256_bytes((self.runs / rid / name).read_bytes())}
        c = tel(comparator, self.entries[comparator]["run_id"])
        return {"launched": True, "run_ids": [rid for _, rid in runs], "reason": "not complete at the freeze",
                "since": None, "telemetry": [tel(k, rid) for k, rid in runs], "comparator": {"slot": comparator, **c}}


def gate_codes(mode: str) -> list:
    from src.tman import validate as V
    return [c for c, modes, _ in V.GATES if mode in modes]


WANTED: set = set()


def expect_gate(w: World, name: str, want: str, mode: str = "freeze", doc: dict | None = None, detail: str = "",
                **kw) -> None:
    """`want` is refused, by the check whose message holds `detail`, with every earlier gate of `mode` passing."""
    WANTED.add(want)
    code, passed, e = w.run(mode, doc, **kw)
    order = gate_codes(mode)
    before = order[:order.index(want)] if want in order else None
    check(name, code == want and passed == before and detail in (e.detail if e else ""),
          f"code {code} ({(e.detail[:160] if e else '')}); passed {len(passed or [])}, want {want} after "
          f"{len(before or [])}")


def mutated(w: World, fn) -> dict:
    d = copy.deepcopy(w.doc)
    fn(d)
    return d


# ======================================================================================================= cases
def case_world_passes():
    w = WORLD
    code, passed, e = w.run("freeze")
    check("world_passes_every_freeze_gate", code is None and passed == gate_codes("freeze"),
          f"{code}: {e.detail[:300] if e else ''}")
    code, passed, e = w.run("draft")
    check("world_passes_draft", code is None and passed == gate_codes("draft"),
          f"{code}: {e.detail[:200] if e else ''}")
    uncovered = sorted(k for k, s in w.cat.slots.items() if s["class"] in ("run", "measurement")
                       and k not in w.entries)
    check("world_covers_every_run_slot", uncovered == ["E6-KD:s42:15ep"], str(uncovered))
    check("teacher_of_record_values_are_preflight_distill_s_DL88", _real_teacher_ok())


def _real_teacher_ok() -> bool:
    import scripts.preflight_distill as pd
    return REAL_TEACHER == {"path": pd.TEACHER_FILE, "dl": "DL-88", "bytes": pd.TEACHER_BYTES,
                            "sha256": pd.TEACHER_SHA256}


def case_input_gates():
    w = WORLD
    expect_gate(w, "test_path_manifest_named_test", "test_path", manifest_path="out/te" + "st/m.json")
    expect_gate(w, "manifest_not_json", "manifest_unreadable", data=b"{")
    expect_gate(w, "manifest_duplicate_key", "manifest_unreadable", data=b'{"format": 1, "format": 2}',
                detail="duplicate key")
    expect_gate(w, "manifest_without_session_part", "manifest_format", doc=mutated(w, lambda d: d.pop("session")),
                detail="with exactly the parts")
    expect_gate(w, "body_without_a_block", "manifest_format", doc=mutated(w, lambda d: d["body"].pop("trigger")),
                detail="not the blocks")
    cat = w.repo / "configs/tday_catalogue.json"
    saved = cat.read_bytes()
    try:
        cat.write_bytes(saved.replace(b'"tday_catalogue/1"', b'"tday_catalogue/2"'))
        expect_gate(w, "catalogue_invalid", "catalogue_invalid")
    finally:
        cat.write_bytes(saved)
    expect_gate(w, "expected_rows_1560_TB14", "expected_rows_mismatch",
                doc=mutated(w, lambda d: d["body"]["expected_rows"].update(value=1560)))
    expect_gate(w, "entry_kind_unknown", "entry_kind_unknown",
                doc=mutated(w, lambda d: d["body"]["entries"][0].update(kind="launched")))
    expect_gate(w, "entry_field_extra_key", "entry_field",
                doc=mutated(w, lambda d: d["body"]["entries"][0].update(value=0.3)))
    expect_gate(w, "slot_unknown", "slot_unknown",
                doc=mutated(w, lambda d: d["body"]["entries"][0].update(slot="E2:s42:lambda8:80k")))
    expect_gate(w, "slot_duplicate", "slot_duplicate",
                doc=mutated(w, lambda d: d["body"]["entries"].append(copy.deepcopy(d["body"]["entries"][0]))))
    expect_gate(w, "derived_slot_in_a_run_kind", "derived_counted_as_run",
                doc=mutated(w, lambda d: w.entry(d, "E7-E2:s42:list42").update(**{"class": "run"})),
                detail="of class run; the slot is a derived")
    expect_gate(w, "entry_role_not_the_catalogue_s", "derived_counted_as_run",
                doc=mutated(w, lambda d: w.entry(d, "G:s42:80k").update(role=["arm", "control"])),
                detail="stage, seed, catalogue block or role differ")


def case_record_gates():
    w = WORLD
    trig = w.repo / "reports/derived/am3_trigger.json"
    saved = trig.read_bytes()
    try:
        trig.write_bytes(b"{}\n")
        expect_gate(w, "record_modified_in_the_tree", "record_uncommitted", detail="am3_trigger.json")
    finally:
        trig.write_bytes(saved)
    expect_gate(w, "default_candidate_selection_not_the_cited", "record_uncommitted", doc=mutated(
        w, lambda d: w.entry(d, "E2:s42:lambda1:80k")["selection"].update(sha256="0" * 64)),
                detail="is not the cited sha256")
    expect_gate(w, "dl_row_absent", "dl_row_missing",
                doc=mutated(w, lambda d: w.entry(d, "G:s42:80k")["dl_rows"].append("DL-9999")))
    expect_gate(w, "cut_entry_cited_with_another_sha", "entry_file_rewritten", doc=mutated(
        w, lambda d: w.entry(d, "G:s42:80k")["entry"].update(sha256="0" * 64)))
    expect_gate(w, "evaluator_commit_not_a_commit", "evaluator_commit_invalid",
                doc=mutated(w, lambda d: d["body"]["evaluator"].update(commit="0" * 40)),
                detail="is not a commit of the records' history")
    expect_gate(w, "evaluator_files_not_those_at_E", "evaluator_commit_invalid",
                doc=mutated(w, lambda d: d["body"]["evaluator"]["files"].update({"scripts/x.py": "1" * 40})),
                detail="the recorded evaluator file set")
    expect_gate(w, "execution_context_per_precision_TB1", "execution_context_split", doc=mutated(
        w, lambda d: d["body"]["execution_context"].update(int8={"model_device": "cpu"})), detail="not one tuple")
    expect_gate(w, "execution_context_on_a_gpu_TB1", "execution_context_split",
                doc=mutated(w, lambda d: d["body"]["execution_context"].update(model_device="cuda")),
                detail="model_device")
    expect_gate(w, "scored_artifact_runtime_differs_TB1", "execution_context_split", artifact_runtimes=[
        {"model_device": "cpu", "input_devices": ["cpu"], "batch_size": 1, "num_workers": 0,
         "determinism_policy_applied": False, "torch_num_threads": 8, "image_digest": w.S.KD_IMAGE,
         "best_val_miou": 0.4}], detail="an artifact's runtime")
    expect_gate(w, "calibration_index_other_checksum", "calibration_index_mismatch", doc=mutated(
        w, lambda d: d["body"]["calibration_index"]["of_record"].update(checksum_sha256="0" * 64)))
    expect_gate(w, "reference_other_strata_sha", "reference_mismatch",
                doc=mutated(w, lambda d: d["body"]["references"]["strata"].update(sha256="0" * 64)),
                detail="the references differ")
    expect_gate(w, "selection_winner_checkpoint_not_the_entry_s", "reference_mismatch", doc=mutated(
        w, lambda d: [w.entry(d, "E2:s42:lambda0.5:80k")["checkpoint"].update(sha256="0" * 64)]),
                detail="checkpoint sha256 is not")
    expect_gate(w, "teacher_entry_not_DL88_s", "teacher_values_mismatch",
                doc=mutated(w, lambda d: w.entry(d, "teacher:s42:40k")["checkpoint"].update(bytes=1)))
    expect_gate(w, "derived_image_not_of_record", "image_digest_not_of_record", doc=mutated(
        w, lambda d: w.entry(d, "E4:s42:list43")["inputs"].update(image_digest="sha256:" + "0" * 64)))
    expect_gate(w, "schedule_block_other_freeze_date", "schedule_invalid",
                doc=mutated(w, lambda d: d["body"]["schedule"].update(original_freeze_date="2026-11-04")))
    expect_gate(w, "content_class_missing_pin", "content_class_missing",
                doc=mutated(w, lambda d: d["body"]["generator_pins"]["image_hasher"].update(pin=None)))
    expect_gate(w, "generator_pin_without_storage", "generator_pin_incomplete",
                doc=mutated(w, lambda d: d["body"]["generator_pins"]["corrupted_cache"]["pin"]["parameters"]
                            .pop("storage")))
    expect_gate(w, "generator_pin_sha256_mismatch_TB6", "generator_pin_sha256_mismatch",
                doc=mutated(w, lambda d: d["body"]["generator_pins"]["corrupted_cache"]["pin"]["parameters"]
                            .update(master_seed=43)))
    expect_gate(w, "generator_cells_reordered_TB6", "generator_cells_mismatch",
                registered_cells=list(reversed(PINS.IF6_CELLS)))
    expect_gate(w, "pin_value_before_unlock_M11", "test_value_before_unlock",
                doc=mutated(w, lambda d: d["body"]["generator_pins"]["corrupted_cache"].update(value={"x": 1})),
                detail="value is set before the unlock")
    expect_gate(w, "session_part_before_the_session", "test_value_before_unlock",
                doc=mutated(w, lambda d: d.update(session={"started_utc": "2026-11-04T00:00:00Z"})),
                detail="a session part exists")
    expect_gate(w, "triggered_without_e6kd", "trigger_inconsistent",
                doc=mutated(w, lambda d: d["body"]["trigger"].update(triggered=True)),
                detail="E6-KD:s42:15ep is not entered")


def case_completion_gates():
    w = WORLD
    expect_gate(w, "freeze_without_runs", "launched_attempt_without_directory", runs_root=None,
                detail="read the run directories")
    expect_gate(w, "attempt_without_directory", "launched_attempt_without_directory", doc=mutated(
        w, lambda d: w.entry(d, "E1:s43:80k")["attempts"].append({"run_id": "e1_s43_80k_a2", "attempt": 2})),
                detail="e1_s43_80k_a2")
    tel = w.runs / "e1_s44_80k" / "e1_telemetry.jsonl"
    saved = tel.read_bytes()
    try:
        tel.write_bytes(saved.replace(b'"max_iters": 80000', b'"max_iters": "80000"'))
        expect_gate(w, "completion_record_unreadable", "completion_record_invalid")
    finally:
        tel.write_bytes(saved)
    expect_gate(w, "completion_block_other_line", "completion_source_mismatch",
                doc=mutated(w, lambda d: w.entry(d, "E1:s42:80k")["completion"].update(line=1)))
    lam = KD_RUNS["E2:s42:lambda0.5:80k"][1]
    expect_gate(w, "run_entry_citing_another_slot_s_run_B52", "completion_source_mismatch", doc=mutated(
        w, lambda d: w.entry(d, "E2:s42:lambda1:80k").update(run_id=lam, attempts=[{"run_id": lam, "attempt": 1}])),
        detail="are not launched attempts of this slot")
    q = w.runs / "e6_s42_15ep"
    tel = q / "qat_telemetry.jsonl"
    saved = tel.read_bytes()
    (q / "epoch_ckpts").mkdir(exist_ok=True)
    import torch
    torch.save({"stage": "E6", "epoch": 1, "step": 10, "seed": 42, "run_id": "e6_s42_15ep", "state_sha256": "0" * 64,
                "source_checkpoint_sha256": "c" * 64, "state_finite": False, "state_nonfinite": False,
                "nonfinite_since_step": None}, q / "epoch_ckpts" / "e01.pt")
    try:
        sha = w.S.sha256_bytes((q / "epoch_ckpts" / "e01.pt").read_bytes())
        rows = [json.loads(x) for x in saved.split(b"\n") if x.strip()]
        for r in rows:
            if r.get("event") == "epoch_end" and r["epoch"] == 1:
                r["checkpoint_sha256"] = sha
        FX.write_jsonl(tel, rows)
        d = mutated(w, lambda d: None)
        d["body"]["entries"] = [dict(e, completion=dict(e["completion"], sha256=w.S.sha256_bytes(tel.read_bytes())))
                                if e["slot"] == "E6:s42:15ep" else e for e in d["body"]["entries"]]
        expect_gate(w, "qat_checkpoint_flag_disagrees_TB13", "checkpoint_state_flag", doc=d)
    finally:
        tel.write_bytes(saved)
        (q / "epoch_ckpts" / "e01.pt").unlink()
        (q / "epoch_ckpts").rmdir()
    t = w.runs / "e2_s42_lambda4_80k_a1" / "e2_telemetry.jsonl"
    saved = t.read_bytes()
    try:
        rows = [json.loads(x) for x in saved.split(b"\n") if x.strip()]
        rows[-1]["wall_clock_end"] = FX.utc("2026-11-03T16:00:00Z")
        FX.write_jsonl(t, rows)
        d = mutated(w, lambda d: w.entry(d, "E2:s42:lambda4:80k")["completion"].update(
            at_utc="2026-11-03T16:00:00Z", sha256=w.S.sha256_bytes(t.read_bytes())))
        expect_gate(w, "complete_at_the_freeze_instant_is_late", "entry_not_complete", doc=d)
    finally:
        t.write_bytes(saved)
    expect_gate(w, "complete_kd_run_omitted_AM19_3d", "complete_run_omitted", doc=mutated(
        w, lambda d: d["body"].update(entries=[e for e in d["body"]["entries"] if e["slot"] != "E2:s42:lambda2:80k"])))
    expect_gate(w, "checkpoint_source_holds_another_sha", "checkpoint_hash_mismatch", doc=mutated(
        w, lambda d: w.entry(d, "E1:s43:80k")["checkpoint"]["sources"][0].update(sha256="0" * 64)),
                detail="does not hold")
    expect_gate(w, "sweep_candidate_other_than_its_selection", "checkpoint_hash_mismatch", doc=mutated(
        w, lambda d: [w.entry(d, "E2:s42:lambda2:80k")["checkpoint"].update(sha256="0" * 64),
                      w.entry(d, "E2:s42:lambda2:80k")["checkpoint"]["sources"][0].update(sha256="0" * 64, dl=None)]),
                detail="lambda_selection.json records")
    ck = w.runs / "e1_s44_80k" / "best.pth"
    saved = ck.read_bytes()
    try:
        ck.write_bytes(b"other bytes")
        expect_gate(w, "checkpoint_file_differs_DL110", "checkpoint_missing")
    finally:
        ck.write_bytes(saved)
    expect_gate(w, "restored_checkpoint_without_its_report", "restored_checkpoint_unrecorded", doc=mutated(
        w, lambda d: w.entry(d, "E1:s44:80k")["checkpoint"]["sources"].append(
            {"kind": "restored", "path": "backup/best.pth", "dl": None,
             "sha256": w.entry(d, "E1:s44:80k")["checkpoint"]["sha256"]})))


R1 = "E3-R1:s42:lambda0.5:alpha25:80k"
G = "G:s42:80k"
A = "A:s42:alpha25:80k"
A100 = "E3:s42:lambda0.5:alpha100:80k"


def cut_world() -> World:
    """A world of its own for the cut gates: the launch log gains a launched, running G run (with a stopped line: it
    awaits a repeat) and a launched, running E3-R1 run; the manifest enters G and the R1 arm as launched cuts at the
    freeze (their runs, telemetry and comparator; E6-R1 and E7-R1 not run under the new R1 cut), so the world still
    passes every freeze gate. The alpha sweep gains a decision record; an AM-23 condition entry reads yes."""
    w = World()
    w.g_rid, w.r1_rid = "g_s42_80k_a1", "e3-r1_s42_lambda0.5_alpha25_80k_a1"
    for rid, stage, slot in ((w.g_rid, "G", G), (w.r1_rid, "E3-R1", R1)):
        d = FX.kd_run(w.runs, rid, stage_key=stage.split("-")[0].lower(), end=None)
        w.launch_lines(slot, rid, d)
    mde = w.M.REFERENCE_FILES["mde_entry"][0]
    w.report = {"path": mde, "sha256": w.S.sha256_bytes((REPO / mde).read_bytes())}
    w.launch.append({"event": "stopped", "run_id": w.g_rid, "report": w.report})
    decision = w.S.json_file_bytes({"format": "sweep_decision_record/1", "sweep": "alpha_cwd",
                                    "decision_date": "2026-10-22", "values": [
                                        {"value": v, "status": "finished", "reason": None}
                                        for v in (25.0, 50.0, 100.0)]})
    w.cond = w.entry_file("condition", ["E6-R1:s42:15ep"], {"arm": "R1", "verdict": "yes", "e3x": {
        "run_id": w.r1_rid, "checkpoint_sha256": "6" * 64, "best_json_sha256": "7" * 64}}, date="2026-10-29")
    w.commit({"reports/kd_launch_log.jsonl": b"".join(json.dumps(x).encode() + b"\n" for x in w.launch),
              "reports/derived/alpha_decision_record.json": decision, w.cond["path"]: w.files[w.cond["path"]]},
             "launch log, decision record, condition")
    w.g_cut = {**w.cut_fields("G", [G], "no card", False), **w.launched_fields([(G, w.g_rid)])}
    w.r1_slots = CUT_GROUPS["R1"][0]
    w.r1_cut = {**w.cut_fields("R1", w.r1_slots, "gate not met", True), **w.launched_fields([(R1, w.r1_rid)])}
    _, g_src = w.committed_cut([G], w.g_cut, nn=2)
    r1_ref, r1_src = w.committed_cut(w.r1_slots, w.r1_cut, nn=2)
    w.rebuild({**g_src, **r1_src, **w.committed_not_run(R1, r1_ref, NOT_RUN[R1], nn=2)})
    return w


def case_cut_gates():
    """Each cut code through a committed cut entry file, cited by the manifest in place of the slots' entries, in
    cut_world: a never-launched A cut, the launched G and R1 cuts, the alpha sweep's decision date, a named rule."""
    from src.tman import entries as E
    from src.training import kd_arms
    w = cut_world()
    code, _, e = w.run("freeze")
    check("cut_world_passes_with_its_launched_cuts", code is None, f"{code}: {e.detail[:300] if e else ''}")
    a = w.cut_fields("A", [A], "no card", False)
    named = {"rule": "named_rule", "rule_record": {"path": w.report["path"], "sha256": w.report["sha256"],
                                                  "date": "2026-10-28"}, "cut_order_position": None, "since": None}
    a100_rid = KD_RUNS[A100][1]
    variants = [
        ("cut_rule_operator", "cut_rule_unknown", [A], dict(a, rule="operator"), FREEZE_DAY),
        ("cut_of_never_cut_slot", "cut_of_never_cut", ["E2:s42:lambda0.5:80k", A], a, FREEZE_DAY),
        ("cut_reason_outside_the_list", "cut_reason_invalid", [A], dict(a, reason="busy"), FREEZE_DAY),
        ("cut_dated_off_the_freeze", "cut_date_mismatch", [A], a, "2026-11-02"),
        ("cut_arithmetic_recomputed_differently", "cut_arithmetic_invalid", [A],
         dict(a, arithmetic=dict(E.cut_arithmetic(10, [], "T", FX.utc("2026-10-20T03:00:00Z")), cut=True)),
         FREEZE_DAY),
        ("cut_launched_without_telemetry", "cut_val_record_missing", [G], dict(w.g_cut, telemetry=[]), FREEZE_DAY),
        ("arm_cut_without_results", "arm_cut_results_missing", w.r1_slots, dict(w.r1_cut, results_known=[]),
         FREEZE_DAY),
        ("launched_arm_cut_by_a_named_rule", "arm_run_cut_while_running", w.r1_slots, {**w.r1_cut, **named},
         "2026-10-28"),
        ("arm_cut_missing_its_later_run", "arm_cut_partial", w.r1_slots[:1],
         w.cut_fields("R1", w.r1_slots[:1], "gate not met", True), FREEZE_DAY),
        ("cut_order_position_other", "cut_order_position_mismatch", [A], dict(a, cut_order_position=3), FREEZE_DAY),
        ("decision_date_cut_not_in_the_record", "decision_date_cut_mismatch", [A100],
         {**a, **w.launched_fields([(A100, a100_rid)]), "rule": "decision_date", "cut_order_position": None},
         "2026-10-22"),
        ("stopped_run_cut_by_a_named_rule", "stopped_run_cut_early", [G], {**w.g_cut, **named}, "2026-10-28"),
        ("e6x_cut_citing_a_yes_condition_AM23_3b", "condition_unmet", ["E6-R1:s42:15ep"],
         {**w.cut_fields("E6-R1", ["E6-R1:s42:15ep"], "condition not met", False), **named,
          "rule_record": {"path": w.cond["path"], "sha256": w.cond["sha256"], "date": "2026-10-29"}}, "2026-10-29"),
    ]
    for nn, (name, want, slots, fields, day) in enumerate(variants, 3):
        ref = w.entry_file("cut", slots, fields, nn=nn, date=day)
        w.commit({ref["path"]: w.files[ref["path"]]}, f"cut variant {name}")
        expect_gate(w, name, want, doc=mutated(w, lambda d, ref=ref, slots=slots: cite_cut(w, d, ref, slots)))
    saved = kd_arms.load_arms
    try:
        kd_arms.load_arms = lambda *a, **kw: {k: v for k, v in saved(*a, **kw).items() if v.run_stage != "E3-R1"}
        expect_gate(w, "arm_without_registry_file_AM17b_4", "unregistered_arm")
    finally:
        kd_arms.load_arms = saved
    check("world_still_passes_after_the_variants", w.run("freeze")[0] is None)


def cite_cut(w: World, d: dict, ref: dict, slots: list) -> None:
    """Cite the cut entry file `ref` as the manifest entry of each of `slots`."""
    for k in slots:
        s = w.cat.slots[k]
        new = {"slot": k, "kind": "cut", "class": s["class"], "stage": s["stage"], "seed": s["seed"],
               "catalogue": w.cat.entry_block(k), "dl_rows": [ref["dl"]], "scope": [], "role": list(s["roles"]),
               "entry": {"path": ref["path"], "sha256": ref["sha256"]}, "cut_date": ref["doc"]["date"],
               "rule": ref["doc"]["rule"],
               "comparator": ref["doc"]["comparator"]["slot"] if ref["doc"]["comparator"] else None}
        old = [e for e in d["body"]["entries"] if e["slot"] == k]
        if old:
            old[0].clear()
            old[0].update(new)
        else:
            d["body"]["entries"].append(new)


def case_cut_attribution():
    """DL-24 blockers #0 and #5 (with #12-#14 and #44): a cut slot's launched attempts and status come from the
    launch log and the run directories, never from the entry. Its own world: E2 seed 43 launched and complete
    (entered as its scored run), E2 seed 44 launched and running (cut at the freeze as launched), E5 seed 43
    launched and running (a QAT run, launched by its run_meta row with mode real; the E5E6-s4344 freeze cut names
    it)."""
    w = World()
    s43, s44, q43 = "E2:s43:lambda0.5:80k", "E2:s44:lambda0.5:80k", "E5:s43:15ep"
    r43, r44, q = "e2_s43_lambda0.5_80k_a1", "e2_s44_lambda0.5_80k_a1", "e5_s43_15ep_a1"
    d43 = FX.kd_run(w.runs, r43, stage_key="e2", seed=43)
    ck43, dl43 = w.ckpt(d43, s43, "iter_80000.pth")
    w.launch_lines(s43, r43, d43)
    w.launch_lines(s44, r44, FX.kd_run(w.runs, r44, stage_key="e2", seed=44, end=None))
    FX.qat_run(w.runs, q, stage="E5", seed=43, epochs=7)
    w.commit({"reports/kd_launch_log.jsonl": b"".join(json.dumps(x).encode() + b"\n" for x in w.launch)},
             "E2 seeds 43 and 44 launched", rows={dl43: w.dl[dl43]})
    scored = w.run_entry(s43, "scored_run", r43, ck43, dl43)
    _, c44 = w.committed_cut([s44], {**w.cut_fields("E2-s4344", [s44], "no card", False),
                                     **w.launched_fields([(s44, r44)])}, nn=2)
    e5 = ["E5:s43:15ep", "E5:s44:15ep"]
    _, c5 = w.committed_cut(e5, {**w.cut_fields("E5E6-s4344", e5, "gate not met", False),
                                 **w.launched_fields([(q43, q)])}, nn=2)
    w.rebuild({s43: scored, **c44, **c5})
    code, _, e = w.run("freeze")
    check("attribution_world_passes_launched_qat_cut_AM19_definitions", code is None,
          f"{code}: {e.detail[:300] if e else ''}")
    group = w.cut_fields("E2-s4344", [s43, s44], "no card", False)
    variants = [
        ("group_cut_naming_a_complete_run_first_B0", "cut_of_complete_run", [s43, s44],
         {**group, **w.launched_fields([(s43, r43), (s44, r44)])}, f"[{s43!r}] are complete"),
        ("group_cut_naming_a_complete_run_last_B5", "cut_of_complete_run", [s43, s44],
         {**group, **w.launched_fields([(s44, r44), (s43, r43)])}, f"[{s43!r}] are complete"),
        ("never_launched_cut_of_a_complete_run_B0", "cut_of_complete_run", [s43],
         w.cut_fields("E2-s4344", [s43], "no card", False), f"[{s43!r}] are complete"),
        ("cut_naming_another_slot_s_run_B5", "cut_reason_invalid", [s44],
         {**group, **w.launched_fields([(s44, r44), (s43, r43)])}, f"is a launched attempt of {s43}"),
        ("launched_kd_run_cut_as_never_launched_B13", "cut_reason_invalid", [s44],
         w.cut_fields("E2-s4344", [s44], "no card", False), "are not named in run_ids"),
        ("launched_qat_run_cut_as_never_launched_B12", "cut_reason_invalid", e5,
         w.cut_fields("E5E6-s4344", e5, "gate not met", False), "are not named in run_ids"),
    ]
    for nn, (name, want, slots, fields, detail) in enumerate(variants, 3):
        ref = w.entry_file("cut", slots, fields, nn=nn)
        w.commit({ref["path"]: w.files[ref["path"]]}, f"cut variant {name}")
        expect_gate(w, name, want, doc=mutated(w, lambda d, ref=ref, slots=slots: cite_cut(w, d, ref, slots)),
                    detail=detail)
    aside = w.runs / "e5_s43_15ep_aside"
    (w.runs / q).rename(aside)
    try:
        expect_gate(w, "cut_run_withheld_from_runs_B44", "launched_attempt_without_directory", detail=q)
    finally:
        aside.rename(w.runs / q)
    expect_gate(w, "complete_e1_run_omitted_B4", "complete_run_omitted", doc=mutated(
        w, lambda d: d["body"].update(entries=[x for x in d["body"]["entries"]
                                               if x["slot"] not in ("E1:s43:80k", "E4:s43:list42")])),
        detail="E1:s43:80k")
    expect_gate(w, "complete_never_cut_pilot_run_omitted_B3", "complete_run_omitted", doc=mutated(
        w, lambda d: d["body"].update(entries=[x for x in d["body"]["entries"] if x["slot"] != "E5:s42:clip1:15ep"])),
        detail="E5:s42:clip1:15ep")
    tel = w.runs / "e5_s42_clip1_15ep" / "qat_telemetry.jsonl"
    saved = tel.read_bytes()
    try:
        FX.qat_run(w.runs, "e5_s42_clip1_15ep", stage="E5", clip_norm=1.0, epochs=7)
        expect_gate(w, "late_never_cut_run_absent_without_delay_B3", "freeze_source_conflict", doc=mutated(
            w, lambda d: d["body"].update(entries=[x for x in d["body"]["entries"]
                                                   if x["slot"] != "E5:s42:clip1:15ep"])), detail="enter")
    finally:
        tel.write_bytes(saved)
    from src.tman import validate as V
    d = mutated(w, lambda d: d["body"].update(entries=[x for x in d["body"]["entries"]
                                                       if x["slot"] not in ("E5:s42:clip1:15ep", A)]))
    res = V.validate(w.ctx("draft", w.S.json_file_bytes(d), runs_root=None))
    check("draft_lists_absent_never_cut_and_cuttable_slots_B3", res.exit_code == 3 and any(
        m.startswith("MISSING: freeze_source_conflict") and "E5:s42:clip1:15ep" in m for m in res.missing)
          and any(m.startswith("MISSING: cut_unentered_at_unlock") and A in m for m in res.missing), str(res.missing))
    expect_gate(w, "cuttable_slot_without_entry_at_the_freeze_B4", "cut_unentered_at_unlock", doc=mutated(
        w, lambda d: d["body"].update(entries=[x for x in d["body"]["entries"] if x["slot"] != A])), detail=A)
    work = FX.teacher_run(w.runs, "teacher-r1_s42_40k_a1")
    (work / "teacher_arm_launch_provenance.json").write_bytes(w.S.json_file_bytes({
        "format": "teacher_arm_launch_provenance/1", "arm": "R1", "attempt": 1, "work_dir": str(work), "launch": True,
        "recorded_utc": "2026-10-12T00:00:00Z", "code_pin": w.E, "records_commit": w.E}))
    try:
        expect_gate(w, "cut_of_a_complete_arm_teacher_AM18_7b", "cut_of_complete_run", detail="teacher-R1:s42:40k")
        expect_gate(w, "complete_arm_teacher_omitted_B4", "complete_run_omitted", doc=mutated(
            w, lambda d: d["body"].update(entries=[x for x in d["body"]["entries"] if x["slot"] not in (
                "teacher-R1:s42:40k", R1, "E6-R1:s42:15ep", "E7-R1:s42:list42")])), detail="teacher-R1:s42:40k")
    finally:
        import shutil
        shutil.rmtree(work)
    launch = list(w.launch)
    rid = "a_s43_alpha25_80k_a1"
    w.launch_lines(A, rid, FX.kd_run(w.runs, rid, stage_key="a", seed=43))
    w.launch[-2]["seed"] = 43
    w.commit({"reports/kd_launch_log.jsonl": b"".join(json.dumps(x).encode() + b"\n" for x in w.launch)},
             "a launched run outside the catalogue")
    try:
        expect_gate(w, "launched_run_outside_the_catalogue_B94", "complete_run_omitted", detail="A:s43:alpha25:80k")
    finally:
        w.launch = launch
        w.commit({"reports/kd_launch_log.jsonl": b"".join(json.dumps(x).encode() + b"\n" for x in w.launch)},
                 "the launch log restored")
    check("attribution_world_still_passes", w.run("freeze")[0] is None)


def case_parent_and_derived_gates():
    w = WORLD
    def derived_under_cut(d):
        e, src = w.entry(d, "E7:s43:list42"), w.entry(d, "E4:s43:list42")
        for k in ("reason", "parent_record", "entry"):
            e.pop(k)
        e.update(kind="derived", method="ptq", inputs=copy.deepcopy(src["inputs"]), hashed=copy.deepcopy(src["hashed"]),
                 ptq_record=None, evaluator_stage="E7", parent="E3:s43:lambda0.5:alpha25:80k")
    expect_gate(w, "derived_under_a_cut_parent", "parent_rule_violation", doc=mutated(w, derived_under_cut))
    expect_gate(w, "cut_parent_child_not_entered", "descendant_entry_missing", doc=mutated(
        w, lambda d: d["body"].update(entries=[e for e in d["body"]["entries"] if e["slot"] != "E7:s44:list42"])))
    expect_gate(w, "not_run_with_another_parent_record", "not_run_without_parent_record", doc=mutated(
        w, lambda d: w.entry(d, "E7:s44:list42")["parent_record"].update(sha256="0" * 64)))
    expect_gate(w, "derived_parent_other_than_catalogue", "derived_parent_incomplete",
                doc=mutated(w, lambda d: w.entry(d, "E4:s42:list43").update(parent="E1:s43:80k")))
    expect_gate(w, "derived_inputs_without_qconfig", "derived_inputs_incomplete",
                doc=mutated(w, lambda d: w.entry(d, "E4:s42:list44")["inputs"].pop("qconfig_fingerprint")))
    expect_gate(w, "ptq_record_of_another_seed_A2", "ptq_record_invalid",
                doc=mutated(w, lambda d: w.entry(d, "E4:s43:list42").update(
                    ptq_record=w.entry(d, "E4:s44:list42")["ptq_record"])))
    expect_gate(w, "required_derived_absent", "derived_missing", doc=mutated(
        w, lambda d: d["body"].update(entries=[e for e in d["body"]["entries"] if e["slot"] != "E4:s42:list45"])))
    expect_gate(w, "e6s42_by_inputs_at_the_freeze", "e6s42_not_hashed_at_freeze", doc=mutated(
        w, lambda d: w.entry(d, "E6:s42:sel").update(hashed=None)))
    expect_gate(w, "second_evaluation_without_report", "derived_selection_conflict", doc=mutated(
        w, lambda d: w.entry(d, "E5:s42:sel")["hashed"]["evaluations"].append({"dir": "eval2", "fault_report": None})))
    expect_gate(w, "append_to_a_hashed_entry", "append_invalid", doc=mutated(
        w, lambda d: d["appends"].append({"slot": "E4:s42:list43", "hashed": copy.deepcopy(
            w.entry(d, "E4:s42:list43")["hashed"]), "ptq_record": None})),
        detail="no derived entry listed by inputs")


def case_appends():
    """AM-19 item 4(c), decision (xi), TB19 Q9 and A2: derived models made after the freeze, appended. A PTQ model
    of record cites its ptq_record, which stands in for the isolation record; a sensitivity model cites an isolation
    record of its provenance's root and digest, made before it."""
    w = WORLD

    def post(d, slot, rec="keep", isolation=None, **h):
        e = w.entry(d, slot)
        hashed = dict(e["hashed"], appended="append", made_utc="2026-11-04T10:00:00Z", isolation=isolation, **h)
        d["appends"].append({"slot": slot, "hashed": hashed, "ptq_record": e["ptq_record"] if rec == "keep" else rec})
        e.update(hashed=None, ptq_record=None)
    code, _, e = w.run("freeze", mutated(w, lambda d: post(d, "E4:s43:list42")))
    check("post_freeze_append_with_its_ptq_record_passes_TB19_Q9", code is None,
          f"{code}: {e.detail[:200] if e else ''}")
    expect_gate(w, "post_freeze_append_of_record_without_its_ptq_record_A2", "ptq_record_invalid",
                doc=mutated(w, lambda d: post(d, "E4:s43:list42", rec=None)), detail="without its ptq_record")
    expect_gate(w, "post_freeze_append_ptq_record_of_another_artifact", "ptq_record_invalid",
                doc=mutated(w, lambda d: post(d, "E4:s43:list42", artifact_sha256="0" * 64)),
                detail="torchscript sha256 is not")
    expect_gate(w, "post_freeze_append_without_an_isolation_record_xi", "append_invalid",
                doc=mutated(w, lambda d: post(d, "E4:s42:list43", rec=None)), detail="without an isolation record")
    S = w.S
    recs = {}
    for name, root in (("of_record", "/workspace/plantseg_trainval"), ("other_root", "/workspace/elsewhere")):
        data = S.json_file_bytes({"format": S.ISOLATION_FORMAT, "root": root, "image_digest": S.KD_IMAGE,
                                  "utc": "2026-11-04T09:00:00Z", "result": {"method": "fixture"}})
        rel = f"reports/derived/tday_isolation_{name}.json"
        recs[name] = {"path": rel, "sha256": S.sha256_bytes(data)}
        w.commit({rel: data}, f"isolation record {name}")
    code, _, e = w.run("freeze", mutated(w, lambda d: post(d, "E4:s42:list43", rec=None, isolation=recs["of_record"])))
    check("post_freeze_append_with_its_isolation_record_passes_xi", code is None,
          f"{code}: {e.detail[:200] if e else ''}")
    expect_gate(w, "isolation_record_of_another_root_xi", "append_invalid", doc=mutated(
        w, lambda d: post(d, "E4:s42:list43", rec=None, isolation=recs["other_root"])), detail="not of the same root")
    expect_gate(w, "isolation_record_not_the_cited_one", "append_invalid", doc=mutated(
        w, lambda d: post(d, "E4:s42:list43", rec=None, isolation=dict(recs["of_record"], sha256="0" * 64))),
        detail="not the cited one")
    expect_gate(w, "append_provenance_not_the_cited_one", "append_invalid", doc=mutated(
        w, lambda d: post(d, "E4:s42:list43", rec=None, isolation=recs["of_record"],
                          provenance={"path": "E4_s42_list43/ptq_run_meta.json", "sha256": "0" * 64})),
        detail="provenance is not the cited one")


def case_non_finite_gate():
    """A QAT run entered as non-finite with its row (AM-21 item 2(b)), then without the run's flags."""
    w = World()
    FX.qat_run(w.runs, "e6_s42_nf", stage="E6", nonfinite_from=1)
    dl = w.row("E6 seed 42: non-finite: no model (AM-21 item 2(b))")
    w.commit({}, rows={dl: w.dl[dl]})
    sha = w.S.sha256_bytes((w.runs / "e6_s42_nf" / "qat_telemetry.jsonl").read_bytes())

    def nf(d, run_id="e6_s42_nf", tsha=sha):
        e = w.entry(d, "E6:s42:15ep")
        e.update(kind="non_finite", run_id=run_id, attempts=[{"run_id": run_id, "attempt": 1}],
                 telemetry_sha256=tsha, refusal_text="non-finite: no model", dl_rows=[dl])
        e.pop("checkpoint")
        e.pop("reason")
        st_line = sum(1 for x in (w.runs / run_id / "qat_telemetry.jsonl").read_bytes().split(b"\n") if x.strip())
        e["completion"] = dict(e["completion"], sha256=tsha, line=st_line, at_utc="2026-10-25T12:00:00Z")
        d["body"]["entries"] = [x for x in d["body"]["entries"] if x["slot"] != "E6:s42:sel"]
    code, passed, e = w.run("freeze", mutated(w, nf))
    check("non_finite_run_without_selection_passes_A1", code is None, f"{code}: {e.detail[:200] if e else ''}")
    FX.qat_run(w.runs, "e6_s42_fin", stage="E6", nonfinite_from=None)
    s2 = w.S.sha256_bytes((w.runs / "e6_s42_fin" / "qat_telemetry.jsonl").read_bytes())
    expect_gate(w, "non_finite_with_a_convertible_epoch", "non_finite_invalid",
                doc=mutated(w, lambda d: nf(d, "e6_s42_fin", s2)))


def case_scope_gates():
    w = WORLD
    expect_gate(w, "scope_corrupted_for_a_per_seed_run", "scope_forbidden",
                doc=mutated(w, lambda d: w.entry(d, "E1:s43:80k")["scope"].append("corrupted")))
    expect_gate(w, "scope_upstream_missing_TB3", "scope_required_missing",
                doc=mutated(w, lambda d: w.entry(d, "E4:s43:list42")["scope"].remove("upstream")))
    expect_gate(w, "non_default_candidate_scored", "candidate_scored", doc=mutated(
        w, lambda d: w.entry(d, "E2:s42:lambda2:80k").update(kind="scored_run", evaluator_stage="E2",
                                                             teacher_ckpt_sha256=None)
        or w.entry(d, "E2:s42:lambda2:80k").pop("reason")))
    expect_gate(w, "default_candidate_entered_as_listed", "default_candidate_missing", doc=mutated(
        w, lambda d: w.entry(d, "E3:s42:lambda0.5:alpha50:80k").update(kind="listed_run", reason="x")
        or [w.entry(d, "E3:s42:lambda0.5:alpha50:80k").pop(k) for k in ("selection", "evaluator_stage",
                                                                          "teacher_ckpt_sha256")]))
    expect_gate(w, "listed_run_with_a_scope", "listed_or_terminal_scored",
                doc=mutated(w, lambda d: w.entry(d, "E2:s42:lambda4:80k")["scope"].append("clean")))


def case_moved_freeze():
    """AM-19 item 1(d): E6 seed 42 (never cut) completes on 2026-11-04; the delay entry lists it on the original
    freeze date; a new freeze on the first later date at whose end every never-cut run is complete passes, a later
    one is refused."""
    from src.tman import schedule as D
    w = WORLD
    FX.qat_run(w.runs, "e6_s42_late", stage="E6", end="2026-11-04T10:00:00Z")
    tel = w.runs / "e6_s42_late" / "qat_telemetry.jsonl"
    delay = w.entry_file("freeze_delay", [], {"original_date": FREEZE_DAY, "runs": ["E6:s42:15ep"]})
    dref = {"path": delay["path"], "sha256": delay["sha256"]}
    news = {day: w.entry_file("new_freeze", [], {"new_date": day, "delay_entry": dref}, date=day)
            for day in ("2026-11-04", "2026-11-05")}
    w.commit({r["path"]: w.files[r["path"]] for r in [delay, *news.values()]}, "freeze delay and new freezes")
    sch = D.schedule_block(w.records.src)

    def moved(d, day):
        n = news[day]
        d["body"]["schedule"] = D.freeze_instant(sch, delays=[dict(dref, doc=delay["doc"])], news=[
            {"doc": n["doc"], "path": n["path"], "sha256": n["sha256"]}]).as_block()
        e = w.entry(d, "E6:s42:15ep")
        e.update(run_id="e6_s42_late", attempts=[{"run_id": "e6_s42_late", "attempt": 1}],
                 completion={"record": "qat_telemetry.jsonl", "at_utc": "2026-11-04T10:00:00Z",
                             "sha256": w.S.sha256_bytes(tel.read_bytes()),
                             "line": sum(1 for x in tel.read_bytes().split(b"\n") if x.strip())})
    code, _, e = w.run("freeze", mutated(w, lambda d: moved(d, "2026-11-04")))
    check("delay_and_new_freeze_on_the_first_complete_day_AM19_1d", code is None,
          f"{code}: {e.detail[:200] if e else ''}")
    expect_gate(w, "new_freeze_not_the_first_complete_day", "freeze_source_conflict",
                doc=mutated(w, lambda d: moved(d, "2026-11-05")))

    def delayed(d):
        moved(d, "2026-11-04")
        d["body"]["schedule"] = D.freeze_instant(sch, delays=[dict(dref, doc=delay["doc"])]).as_block()
    expect_gate(w, "delay_without_new_freeze_stays_a_draft_B3", "freeze_source_conflict", doc=mutated(w, delayed),
                detail="stays a draft")
    expect_gate(w, "late_never_cut_run_without_delay_B3", "freeze_source_conflict", doc=mutated(
        w, lambda d: moved(d, "2026-11-04") or d["body"].update(schedule=w.doc["body"]["schedule"])),
        detail="the delay is entered")
    aside = FX.scratch_dir("tmanaside") / "e6_s42_15ep"
    (w.runs / "e6_s42_15ep").rename(aside)              # E6 seed 42's only launched attempt is the late one
    try:
        expect_gate(w, "late_never_cut_run_absent_at_the_new_freeze_B3", "freeze_source_conflict", doc=mutated(
            w, lambda d: moved(d, "2026-11-04") or d["body"].update(
                entries=[x for x in d["body"]["entries"] if x["slot"] != "E6:s42:15ep"])),
            detail="no entry at the new freeze")
    finally:
        aside.rename(w.runs / "e6_s42_15ep")
    expect_gate(w, "delay_entry_without_runs_refused_at_its_gate", "launched_attempt_without_directory",
                doc=mutated(w, lambda d: moved(d, "2026-11-04")), runs_root=None)


def case_coverage():
    missing = [c for c in gate_codes("freeze") if c not in WANTED]
    check("every_freeze_gate_has_a_case", not missing, str(missing))


def inv_refused(name: str, want: str, path, detail: str = "", **kw) -> None:
    """read_inventory refuses `path` with `want`, by the check whose message holds `detail`."""
    from src.tman import validate as V
    try:
        V.read_inventory(path, **kw)
        got, msg = None, ""
    except WORLD.S.TmanRefused as e:
        got, msg = e.code, e.detail
    check(name, got == want and detail in msg, f"got {got} ({msg[:200]})")


def case_inventory():
    """IF§11 v2 (A1): the export tday-inventory/1 built from the world's manifest and read back by read_inventory
    (IF§11.6): IF§11.1's four row keys, rows unique by (slot, precision), a selected QAT model's two rows; IF§11.2's
    entries, one per catalogue slot, kind null without an entry; IF§11.3's telemetry by bytes, a copy-back hash that
    differs or names an absent file refused; IF§11.5's dedup, freeze and binding. A schema case per new key and a
    malformed value of each; a bound binding returns its artifacts records."""
    from src.tman import manifest as M
    from src.tman import validate as V
    w = WORLD
    S = w.S
    inv = M.build_inventory(w.doc, w.data, w.records, w.cat, runs_root=w.runs, registered=PINS.IF6_CELLS,
                            freeze_dl="DL-300")
    out = FX.scratch_dir("tmaninv")
    data = S.json_file_bytes(inv)
    (out / "inv.json").write_bytes(data)
    check("inventory_built_and_read_IF11_6", V.read_inventory(out / "inv.json", expect_sha256=S.sha256_bytes(data))
          == inv)
    rows = inv["models"]
    keys = ("slot", "entry_kind", "role", "training_run_id")
    check("rows_carry_the_four_keys_IF11_1", bool(rows) and all(set(keys) <= set(r) for r in rows))
    sel = sorted((r["precision"], r["training_run_id"], r["source"]["kind"]) for r in rows if r["slot"] == "E6:s42:sel")
    check("qat_selection_rows_int8_qat_and_fq_off_TB5", sel == [("int8_qat", "e6_s42_15ep", "provenance"),
                                                                ("qat_fakequant_off", "e6_s42_15ep", "checkpoint")],
          str(sel))
    e4 = [r for r in rows if r["slot"] == "E4:s43:list42"]
    check("ptq_row_training_run_id_is_its_parent_s", [r["training_run_id"] for r in e4] == ["e1_s43_80k"])
    check("int8_rows_carry_the_converted_artifact_TB17", all(
        (r["converted_artifact_sha256"] is not None) == (r["precision"] in ("int8_ptq", "int8_qat")) for r in rows))
    check("teacher_row_has_no_seed", [r["seed"] for r in rows if r["slot"] == "teacher:s42:40k"] == [None])
    check("entries_one_per_catalogue_slot_IF11_2", [it["slot"] for it in inv["entries"]] == list(w.cat.slots))
    nulls = [it for it in inv["entries"] if it["kind"] is None]
    want_nulls = [k for k in w.cat.slots if k not in w.entries]
    check("null_kind_items_are_the_slots_without_entry", [it["slot"] for it in nulls] == want_nulls
          and "E6-KD:s42:15ep" in want_nulls
          and all(sorted(it) == ["catalogue", "class", "kind", "role", "seed", "slot", "stage"] for it in nulls),
          str([it["slot"] for it in nulls]))
    tel = next(it for it in inv["entries"] if it["slot"] == "E1:s42:80k")["telemetry"]
    f = w.runs / "e1_s42_80k" / "e1_telemetry.jsonl"
    check("telemetry_per_attempt_by_bytes_IF11_3", tel == [{"run_id": "e1_s42_80k",
                                                            "path": "e1_s42_80k/e1_telemetry.jsonl",
                                                            "sha256": S.sha256_bytes(f.read_bytes())}])
    cut = next(it for it in inv["entries"] if it["slot"] == "G:s42:80k")
    check("cut_item_carries_its_label_IF11_2", (cut["kind"], cut["cut_date"], cut["rule"], cut["comparator"])
          == ("cut", FREEZE_DAY, "freeze", None) and "telemetry" not in cut)
    check("manifest_sha256_is_the_body_s_and_the_file_s_kept", inv["manifest_sha256"] == S.canonical_sha256(
        w.doc["body"]) and inv["manifest_file_sha256"] == S.sha256_bytes(w.data))
    check("dedup_null_binding_unbound_freeze_row_IF11_5", inv["dedup"] is None and inv["binding"] == {
        "state": "unbound", "path": None, "sha256": None} and inv["freeze"] == {
        "dl_row": "DL-300", "manifest_sha256": inv["manifest_sha256"]})
    check("freeze_null_before_the_freeze", M.build_inventory(w.doc, w.data, w.records, w.cat, runs_root=w.runs,
                                                             registered=PINS.IF6_CELLS)["freeze"] is None)
    saved = f.read_bytes()
    try:
        f.write_bytes(saved + b"\n")
        expect_refused("telemetry_copy_back_differs_IF11_3", "completion_source_mismatch", M.build_inventory,
                       w.doc, w.data, w.records, w.cat, runs_root=w.runs, registered=PINS.IF6_CELLS)
        f.unlink()
        expect_refused("telemetry_with_a_copy_back_hash_absent", "completion_source_mismatch", M.build_inventory,
                       w.doc, w.data, w.records, w.cat, runs_root=w.runs, registered=PINS.IF6_CELLS)
    finally:
        f.write_bytes(saved)

    n = [0]

    def variant(fn) -> Path:
        d = copy.deepcopy(inv)
        fn(d)
        n[0] += 1
        p = out / f"inv_{n[0]}.json"
        p.write_bytes(S.json_file_bytes(d))
        return p
    inv_refused("inventory_sha256_mismatch", "inventory_sha256_mismatch", out / "inv.json", expect_sha256="0" * 64)
    for k in keys:
        inv_refused(f"row_without_{k}_IF11_1", "inventory_invalid", variant(lambda d, k=k: d["models"][0].pop(k)),
                    detail=k)
    for k in ("entries", "dedup", "freeze", "binding", "manifest_file_sha256"):
        inv_refused(f"export_without_{k}", "inventory_invalid", variant(lambda d, k=k: d.pop(k)), detail="keys")
    for name, fn, detail in (
            ("row_role_outside_the_vocabulary", lambda d: d["models"][0].update(role=["winner"]), "role"),
            ("row_entry_kind_not_a_scored_kind", lambda d: d["models"][0].update(entry_kind="listed_run"),
             "entry_kind"),
            ("row_training_run_id_empty", lambda d: d["models"][0].update(training_run_id=""), "training_run_id"),
            ("row_slot_malformed", lambda d: d["models"][0].update(slot="E1 s42"), "slot"),
            ("rows_unique_by_slot_and_precision_IF11_1", lambda d: d["models"].append(copy.deepcopy(d["models"][0])),
             "a second row"),
            ("telemetry_item_without_sha256_IF11_3", lambda d: next(
                it for it in d["entries"] if it["slot"] == "E1:s42:80k")["telemetry"][0].pop("sha256"), "telemetry"),
            ("null_kind_item_with_an_entry_field", lambda d: next(
                it for it in d["entries"] if it["kind"] is None).update(dl_rows=[]), "dl_rows"),
            ("null_kind_item_without_its_catalogue", lambda d: next(
                it for it in d["entries"] if it["kind"] is None).pop("catalogue"), "catalogue"),
            ("entries_list_a_slot_twice", lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])), "twice"),
            ("dedup_malformed_IF11_5", lambda d: d.update(dedup={"path": "tday_dedup.json"}), "dedup"),
            ("freeze_manifest_sha256_differs_IF11_5", lambda d: d["freeze"].update(manifest_sha256="0" * 64),
             "freeze.manifest_sha256"),
            ("binding_bound_without_its_path", lambda d: d["binding"].update(state="bound"), "binding"),
            ("binding_unbound_with_a_path", lambda d: d["binding"].update(path="b.json"), "binding")):
        inv_refused(name, "inventory_invalid", variant(fn), detail=detail)
    art = {"slot": "E1:s42:80k", "precision": "fp32", "scope": "clean", "cell": None, "protocol": "canvas",
           "repo_relative_path": "reports/heldout/E1_s42_80k/clean", "files_sha256": {"summary.json": "a" * 64},
           "checkpoint_sha256": w.entries["E1:s42:80k"]["checkpoint"]["sha256"], "repo_commit": w.E,
           "split_manifest_sha256": "b" * 64}
    bdata = S.json_file_bytes({"format": S.BINDING_FORMAT, "references": {"mde_entry": {"sha256": "c" * 64}},
                               "artifacts": [art]})
    (out / "tday_binding.json").write_bytes(bdata)
    bound = {"state": "bound", "path": "tday_binding.json", "sha256": S.sha256_bytes(bdata)}
    got = V.read_inventory(variant(lambda d: d.update(binding=dict(bound))))
    check("bound_binding_returns_its_artifacts_IF11_4", got["binding"].get("artifacts") == [art])
    inv_refused("bound_binding_other_sha256", "binding_invalid",
                variant(lambda d: d.update(binding=dict(bound, sha256="0" * 64))), detail="not binding.sha256")
    inv_refused("bound_binding_file_absent", "binding_invalid",
                variant(lambda d: d.update(binding=dict(bound, path="tday_binding_absent.json"))))
    bad = S.json_file_bytes({"format": S.BINDING_FORMAT, "references": {}, "artifacts": [dict(art, scope="dedup")]})
    (out / "tday_binding_dedup.json").write_bytes(bad)
    inv_refused("binding_artifact_scope_dedup_IF11_4", "binding_invalid", variant(lambda d: d.update(binding={
        "state": "bound", "path": "tday_binding_dedup.json", "sha256": S.sha256_bytes(bad)})), detail="scope")


def case_draft_and_cli():
    from src.tman import manifest as M
    from src.tman import validate as V
    w = WORLD
    d = mutated(w, lambda d: [d["body"]["generator_pins"][k].update(pin=None) for k in ("corrupted_cache",
                                                                                      "image_hasher")])
    d["body"]["entries"] = [e for e in d["body"]["entries"] if e["slot"] != "E4:s42:list45"]
    res = V.validate(w.ctx("draft", w.S.json_file_bytes(d), runs_root=None))
    check("draft_prints_missing_and_exits_3", res.exit_code == 3 and any("content_class_missing" in m
                                                                          for m in res.missing)
          and any("derived_missing" in m for m in res.missing), str(res.missing))
    code, _, _ = w.run("freeze", d)
    check("freeze_refuses_what_draft_lists", code == "content_class_missing")
    built = M.build_manifest(w.records, w.cat, w.sources, runs_root=w.runs)
    check("builder_is_deterministic", w.S.json_file_bytes(built) == w.data)
    import scripts.tday_manifest as CLI
    out = FX.scratch_dir("tmanout")
    (out / "m.json").write_bytes(w.data)
    cells = out / "cells.json"
    cells.write_text(json.dumps(PINS.IF6_CELLS), encoding="utf-8")
    rc = CLI.main(["--root", str(w.repo), "validate", "--manifest", str(out / "m.json"), "--mode", "draft",
                   "--registered-cells", str(cells)])
    check("cli_validate_draft_passes", rc == 0)
    rc = CLI.main(["--root", str(w.repo), "validate", "--manifest", str(out / "m.json"), "--mode", "freeze",
                   "--registered-cells", str(cells)])
    check("cli_validate_freeze_without_runs_refused", rc == 2)
    check("gate_codes_unique_and_ordered", len(set(c for c, _, _ in V.GATES)) == len(V.GATES) == 79)


WORLD: World | None = None


def main() -> int:
    global WORLD
    try:
        WORLD = World()
    except Exception as e:  # noqa: BLE001
        check("world_built", False, f"{type(e).__name__}: {e}")
        return FX.run_cases("TMAN VALIDATE SMOKE (lane TMAN, B-1, C5)", ())
    try:
        return FX.run_cases("TMAN VALIDATE SMOKE (lane TMAN, B-1, C5)",
                            (case_world_passes, case_input_gates, case_record_gates, case_completion_gates,
                             case_cut_gates, case_cut_attribution, case_parent_and_derived_gates, case_appends,
                             case_non_finite_gate, case_scope_gates, case_moved_freeze, case_draft_and_cli,
                             case_inventory, case_coverage))
    finally:
        if REAL_TEACHER is not None:
            WORLD.M.TEACHER_OF_RECORD.clear()
            WORLD.M.TEACHER_OF_RECORD.update(REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(main())
