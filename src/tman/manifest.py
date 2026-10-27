"""The manifest's derived entries, appends, isolation record and generator pins (lane TMAN, B-1, C4); the builder
and the inventory export follow at C5.

Derived artifacts (AM-19 Definitions; item 4(c)). The PTQ of a complete parent and the conversion and selection of
a complete QAT run are not runs. At the freeze the manifest lists each by its inputs; its hashes are appended
before the unlock, and appends are the only entries added after the freeze. E6 seed 42's conversion and selection
are hashed at the freeze (AM-19 Definitions). One made after the freeze cites an isolation record: AM-19 item 4(c)'s
"TRAIN/VAL-only staged root", checked in the same container just before the command by
src/data/isolation.assert_trainval_only_root (:64-76; existence checks only), with the same root and image digest
and an earlier time (decision (xi)); an intact ptq_record/1 of record stands in for it for a PTQ artifact (ruling
TB19 Q9; A2: its reader re-hashes what it lists, and LQP's run checks the TEST surfaces before it writes it).

Inputs: PTQ, AM-19 item 4(c)'s five (the parent checkpoint's sha256, src/quant/ptq.py:840; the calibration list's
file sha256; the PTQ script's path and commit, :724-733; the image digest, :736-745; the qconfig fingerprint,
:929-933 with src/quant/runner.py:208-213). QAT: the run's telemetry sha256 as recorded on the pod (it binds every
epoch checkpoint's sha256), the conversion and selection scripts' commit, the image digest,
configs/qat_selection_rules.json's sha256 (read at build and by the validator, TB13) and the pilot's clip-selection
record (its path under the run directories and its sha256, so the validator reads it: TB9).

Generator pins (AM-19 item 1(c)): the corrupted-TEST cache pin is L-CORR's JSON, embedded verbatim (IF§3): the
validator recomputes pin_sha256 and compares its cells with the registered set (src.corruption_roles, imported
lazily; until master has the module the set comes through a parameter, TB6). The image hasher's pin is B-3a's.
Neither carries a value before the unlock record (M11; test_value_before_unlock).
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

from src.training import sweep_select as ss

from . import records as R
from . import schedule as D
from . import schema as S

# ----------------------------------------------------------------------------------------- derived (§1.5)
PTQ_INPUTS = {"parent_checkpoint_sha256": "sha256",
              "calibration_list": ("obj", {"path": "str", "file_sha256": "sha256"}),
              "script": ("obj", {"path": "str", "commit": "commit"}),
              "image_digest": "str", "qconfig_fingerprint": "str"}
QAT_INPUTS = {"run_slot": "slot", "telemetry_sha256": "sha256", "scripts_commit": "commit", "image_digest": "str",
              "rules_sha256": "sha256", "clip_selection": S.REF}      # the pilot's record, under --runs (TB9)
HASHED = {"artifact_sha256": "sha256", "state_dict_sha256": "sha256", "x86_copy_sha256": "sha256",
          "provenance": S.REF, "record": S.REF, "made_utc": "utc", "image_digest": "str",
          "isolation": S.nullable(S.REF), "appended": ("enum", ("freeze", "append"))}
# AM-21 item 2(c): a second evaluation directory of a QAT run needs its fault report; the selected epoch's
# checkpoint is the fq_off row's source (TB5; IF§8)
HASHED_QAT = {**HASHED, "evaluations": ("list", ("obj", {"dir": "str", "fault_report": S.nullable(S.REF)})),
              "selected_checkpoint": S.REF}
APPEND = {"slot": "slot", "hashed": ("map", "any"), "ptq_record": S.nullable(S.REF)}    # A2: the made entry's
QAT_RULES_REL = "configs/qat_selection_rules.json"
COMPLETE_KINDS = ("scored_run", "default_candidate", "arm_teacher", "listed_run")     # a complete run: a model
# DL-10's lists (§1.6): AM-10's list of record (seed 42) and AM-16 item 5's sensitivity lists (seeds 43-45)
CALIBRATION_ROLES = {42: "am10_list_of_record", 43: "am16_calibration_sensitivity",
                     44: "am16_calibration_sensitivity", 45: "am16_calibration_sensitivity"}


def qat_rules_sha256(records: R.Records) -> str:
    """TB13: the sha256 of the committed QAT selection rules file, read at build and by the validator."""
    return S.sha256_bytes(records.read(QAT_RULES_REL))
MODEL_OF_RECORD_ROLES = ("family", "per_seed")                                       # A2: these carry ptq_record


def inputs_spec(method: str) -> dict:
    return PTQ_INPUTS if method == "ptq" else QAT_INPUTS


def hashed_spec(method: str) -> dict:
    return HASHED if method == "ptq" else HASHED_QAT


def derived_inputs_errors(entry: dict) -> list[str]:
    """derived_inputs_incomplete: a derived entry's inputs are all of its method's (AM-19 item 4(c)), and its hashed
    block, when present, all of its fields."""
    errs = S.value_errors(entry["inputs"], ("obj", inputs_spec(entry["method"])), f"{entry['slot']}.inputs")
    if entry["hashed"] is not None:
        errs += S.value_errors(entry["hashed"], ("obj", hashed_spec(entry["method"])), f"{entry['slot']}.hashed")
    return errs


def derived_parent_errors(entry: dict, by_slot: dict, cat) -> list[str]:
    """derived_parent_incomplete: the derived entry's parent is its catalogue parent (for the pilot's selection, one
    of its runs) and holds a complete run (AM-19 item 4(b))."""
    s = cat.slots[entry["slot"]]
    allowed = s["pilot"]["runs"] if s["pilot"] and "runs" in s["pilot"] else [s["parent"]]
    pe = by_slot.get(entry["parent"])
    if entry["parent"] not in allowed or pe is None or pe["kind"] not in COMPLETE_KINDS:
        return [f"{entry['slot']}: its parent {entry['parent']} ({(pe or {}).get('kind', 'no entry')}) is not a "
                f"complete run of {allowed} (AM-19 item 4(b))"]
    if s["record_kind"] == "qat_selection" and entry["inputs"].get("run_slot") != entry["parent"]:
        return [f"{entry['slot']}: inputs.run_slot {entry['inputs'].get('run_slot')} is not its parent"]
    return []


PTQ_SCRIPT, PTQ_ARM_SCRIPT = "scripts/run_ptq.py", "scripts/run_ptq_arm.py"


def ptq_script(slot: str, cat) -> str | None:
    """AM-19 item 4(c)'s PTQ script of a derived slot: scripts/run_ptq_arm.py for an arm's E7 (role arm),
    scripts/run_ptq.py for every other PTQ; none for a slot whose code is not merged at O (its lane's: not
    compared)."""
    if code_gated(slot, cat):
        return None
    return PTQ_ARM_SCRIPT if "arm" in cat.slots[slot]["roles"] else PTQ_SCRIPT


def calibration_item(ci: dict | None, slot: dict) -> dict | None:
    """The calibration index's list for a PTQ slot's params.list (42: of record; 43-45: sensitivity)."""
    lst, ci = slot["params"].get("list"), ci or {}
    return ci.get("of_record") if lst == 42 else (ci.get("sensitivity") or {}).get(str(lst))


def derived_inputs_mismatch_errors(entry: dict, by_slot: dict, cat, ci: dict | None, src) -> list[str]:
    """derived_inputs_mismatch: a derived entry's listed inputs are their sources' (AM-19 item 4(c); DL-24 #22,
    #53). A PTQ's parent checkpoint is its parent entry's, its calibration list the calibration index's list for the
    slot's params.list (when the index is given; content_class_missing otherwise) and its script the slot's PTQ
    script; a QAT selection's telemetry is its run's completion record; the scripts' commit is in the records'
    history."""
    slot, inp, pe = entry["slot"], entry["inputs"], by_slot.get(entry["parent"]) or {}
    s, errs = cat.slots[slot], []
    if entry["method"] == "ptq":
        if inp["parent_checkpoint_sha256"] != (pe.get("checkpoint") or {}).get("sha256"):
            errs.append(f"{slot}: inputs.parent_checkpoint_sha256 is not its parent {entry['parent']}'s checkpoint")
        item = calibration_item(ci, s) or {}
        if ci is not None and inp["calibration_list"] != {"path": item.get("path"),
                                                          "file_sha256": item.get("file_sha256")}:
            errs.append(f"{slot}: inputs.calibration_list is not the calibration index's list "
                        f"{s['params'].get('list')}")
        script = ptq_script(slot, cat)
        if script is not None and inp["script"]["path"] != script:
            errs.append(f"{slot}: inputs.script.path {inp['script']['path']} is not {script}")
        commit = inp["script"]["commit"]
    else:
        if inp["telemetry_sha256"] != (pe.get("completion") or {}).get("sha256"):
            errs.append(f"{slot}: inputs.telemetry_sha256 is not its run {entry['parent']}'s completion record")
        commit = inp["scripts_commit"]
    if not (src.commit_exists(commit) and src.is_ancestor(commit, src.records_ref)):
        errs.append(f"{slot}: its scripts' commit {commit[:12]} is not in the records' history")
    return errs


def _dotted(doc, key: str):
    for k in key.split("."):
        doc = doc.get(k) if isinstance(doc, dict) else None
    return doc


def derived_hash_errors(entry: dict, parent: dict, cat, ci: dict | None, runs_root) -> list[str]:
    """derived_hash_mismatch: a hashed derived model is what its records say, read under the run directories
    (decision (xviii); DL-24 #27, #34, #49, #50). Its cited files -- the provenance, the record and a QAT selection's
    selected checkpoint -- have the cited sha256. A PTQ model's provenance names the listed inputs (the parent
    checkpoint, the calibration list with the index's checksum and role, its seed, the qconfig, the script's commit)
    and the hashed artifacts and image digest. A QAT selection's record (qat_selection/1) names the rules file, the
    run with its telemetry and clip norm, and as its winner the selected checkpoint (its epoch_end row) with the
    hashed artifact and provenance. The clip selection the inputs cite (qat_clip_selection/1, the same rules) names
    the pilot's selection as its winner, with its run, telemetry and record (TB9), and its winner's clip norm is
    every run's (src/quant/qat.py clip_binding)."""
    from . import completion as C
    h, inp, slot, root = entry["hashed"], entry["inputs"], entry["slot"], Path(runs_root)
    code, qat = "derived_hash_mismatch", entry["method"] == "qat_selection"
    reads = {"provenance": None if qat else R.PTQ_PROVENANCE, "record": R.QAT_SELECTION if qat else None}
    if qat:
        reads["selected_checkpoint"] = None
    docs, bad = {}, []
    for k, allow in reads.items():
        p = root / h[k]["path"]
        docs[k], sha = R.read_json_file(p, allow, code=code) if allow else (None, R.file_sha256(p, code=code))
        if sha != h[k]["sha256"]:
            bad.append(k)
    if bad:
        return [f"{slot}: its {bad} under the run directories are not the cited sha256"]
    if not qat:
        s = cat.slots[slot]
        want = {"source_checkpoint_sha256": inp["parent_checkpoint_sha256"],
                "calibration.list_file_sha256": inp["calibration_list"]["file_sha256"],
                "calibration.seed": s["params"].get("list"), "qconfig.fingerprint": inp["qconfig_fingerprint"],
                "code.commit": inp["script"]["commit"], "converted_artifact_sha256": h["artifact_sha256"],
                "state_dict_artifact.sha256": h["state_dict_sha256"], "x86_latency_copy.sha256": h["x86_copy_sha256"],
                "environment.image_digest": h["image_digest"]}
        if ci is not None:
            item = calibration_item(ci, s) or {}
            want.update({"calibration.checksum_sha256": item.get("checksum_sha256"),
                         "calibration.role": item.get("role")})
        bad = sorted(k for k, v in want.items() if _dotted(docs["provenance"], k) != v)
        return [f"{slot}: its provenance names other {bad} than the entry"] if bad else []
    rec, rid = docs["record"], parent.get("run_id")
    rows, _, _ = C.qat_rows(root / rid)
    meta = rows[0] if rows and rows[0].get("event") == "run_meta" else {}
    epoch = _dotted(rec, "winner.epoch")
    row = next((r for r in rows if r.get("event") == "epoch_end" and r.get("epoch") == epoch), None) \
        if isinstance(epoch, int) else None
    want = {"format": R.QAT_SELECTION_FORMAT, "rules_sha256": inp["rules_sha256"], "run_id": rid,
            "telemetry_sha256": inp["telemetry_sha256"], "clip_norm": meta.get("clip_norm"), "winner.run_id": rid,
            "winner.checkpoint_sha256": h["selected_checkpoint"]["sha256"],
            "winner.artifact_of_record.sha256": h["artifact_sha256"],
            "winner.provenance.sha256": h["provenance"]["sha256"]}
    bad = sorted(k for k, v in want.items() if _dotted(rec, k) != v)
    if row is None or h["selected_checkpoint"] != {"path": f"{rid}/{row.get('checkpoint')}",
                                                   "sha256": row.get("checkpoint_sha256")} \
            or _dotted(rec, "winner.checkpoint") != row.get("checkpoint"):
        bad.append("selected_checkpoint (the winner's epoch_end row: its checkpoint and sha256)")
    clip, csha = R.read_json_file(root / inp["clip_selection"]["path"], R.CLIP_SELECTION, code=code)
    won = (_dotted(clip, "winner.run_id"), _dotted(clip, "winner.telemetry_sha256"),
           _dotted(clip, "winner.qat_selection.sha256"))
    if csha != inp["clip_selection"]["sha256"] or _dotted(clip, "format") != R.CLIP_SELECTION_FORMAT \
            or _dotted(clip, "rules_sha256") != inp["rules_sha256"]:
        bad.append("clip_selection (its sha256, format and rules)")
    elif cat.slots[slot]["pilot"] and won != (rid, inp["telemetry_sha256"], h["record"]["sha256"]):
        bad.append(f"the pilot's winner (the clip selection names {won[0]}, its telemetry and record; TB9)")
    elif _dotted(clip, "winner.clip_norm") != _dotted(rec, "clip_norm"):
        bad.append(f"clip_norm (the clip selection's winner's is {_dotted(clip, 'winner.clip_norm')!r})")
    return [f"{slot}: its records disagree with the entry and its run at {bad}"] if bad else []


def code_gated(slot: str, cat) -> bool:
    """A slot whose code is not merged yet (its own or its parent's code_merged gate): it may be entered as not
    made (AM-23 item 4; TB8)."""
    s = cat.slots[slot]
    gates = s["gates"] + (cat.slots[s["parent"]]["gates"] if s["parent"] else [])
    return any(g["record"] == "code_merged" for g in gates)


def required_derived(cat, by_slot: dict) -> list[str]:
    """The derived slots required at the freeze (§1.5; AM-19 item 4(b)): each derived slot whose parent run (or, for
    the pilot's selection, the winning pilot run its entry names) is complete with a model. A QAT run entered as
    non-finite has no selection (IF§11.2)."""
    out = []
    for k, s in cat.slots.items():
        if s["class"] != "derived":
            continue
        if s["pilot"] and "runs" in s["pilot"]:
            if any((by_slot.get(r) or {}).get("kind") in COMPLETE_KINDS for r in s["pilot"]["runs"]):
                out.append(k)
        elif (by_slot.get(s["parent"]) or {}).get("kind") in COMPLETE_KINDS:
            out.append(k)
    return out


def derived_missing_errors(cat, by_slot: dict) -> list[str]:
    """derived_missing: a required derived slot without a derived entry, or a not-made entry for a slot without a
    code_merged gate (E7-E2's rule, AM-23 items 2 and 4). Whether the gated code has merged is the validator's
    (the lane register's merge commit; validate.g_derived_missing)."""
    errs = []
    for k in required_derived(cat, by_slot):
        kind = (by_slot.get(k) or {}).get("kind")
        if kind == "derived" or (kind == "not_made" and code_gated(k, cat)):
            continue
        errs.append(f"{k}: required at the freeze (its parent is complete), entered as {kind or 'nothing'}")
    return errs


def e6s42_errors(by_slot: dict) -> list[str]:
    """e6s42_not_hashed_at_freeze: E6 seed 42's conversion and selection are hashed at the freeze (AM-19
    Definitions), unless the run has no model (AM-21 item 2(b))."""
    run, sel = by_slot.get("E6:s42:15ep") or {}, by_slot.get("E6:s42:sel") or {}
    if run.get("kind") == "non_finite":
        return []
    h = sel.get("hashed") if sel.get("kind") == "derived" else None
    if not (isinstance(h, dict) and h.get("appended") == "freeze"):
        return ["E6:s42:sel is not hashed at the freeze (AM-19 Definitions)"]
    return []


def selection_conflict_errors(entry: dict) -> list[str]:
    """derived_selection_conflict: a QAT run's second evaluation directory needs its fault report (AM-21 item
    2(c))."""
    if entry["method"] != "qat_selection" or entry["hashed"] is None:
        return []
    ev = entry["hashed"].get("evaluations")
    if not (isinstance(ev, list) and all(isinstance(e, dict) for e in ev)):
        return []                                  # an appended form's fields are append_invalid's
    if not ev or any(e.get("fault_report") is None for e in ev[1:]):
        return [f"{entry['slot']}: evaluation directories {[e['dir'] for e in ev]}: a second one needs a fault "
                "report (AM-21 item 2(c))"]
    return []


def non_finite_errors(entry: dict, rows: list | None, telemetry_sha256: str | None, dl) -> list[str]:
    """non_finite_invalid: a QAT run entered as "non-finite: no model" (AM-21 item 2(b)) has a decision-log row
    saying so, and, read under --runs, a complete telemetry with the recorded sha256 whose every epoch_end row has
    state_finite false (no epoch can be converted; flags only, no score)."""
    errs = []
    if not any(dl.has(r) and S.NO_MODEL_ENTRY in (dl.row_text(r) or "") for r in entry["dl_rows"]):
        errs.append(f"{entry['slot']}: no decision-log row of {entry['dl_rows']} says {S.NO_MODEL_ENTRY!r}")
    if rows is not None:
        ends = [r for r in rows if r.get("event") == "epoch_end"]
        if telemetry_sha256 != entry["telemetry_sha256"]:
            errs.append(f"{entry['slot']}: its telemetry sha256 {telemetry_sha256} is not the entry's")
        elif len(ends) != 15 or any(r.get("state_finite") is not False for r in ends):
            errs.append(f"{entry['slot']}: its telemetry shows a convertible epoch, or is not complete")
    return errs


def ptq_record_errors(entry: dict, cat, parent_entry: dict | None, checksum_sha256: str, *, runs_root,
                      reader=None) -> list[str]:
    """ptq_record_invalid (A2): a derived E4/E7 entry of role family or per_seed, once made, carries its
    ptq_record/1 {path, sha256}, read only through lane LQP's reader (or its stand-in) and projected; the record is
    of record (roles calibration_sensitivity and d3_replicate never are), of the entry's stage and seed, of its
    parent's checkpoint and of the calibration list of record, and lists the entry's three artifacts. A
    sensitivity entry's record, when it has one, is a calibration-sensitivity record."""
    s = cat.slots[entry["slot"]]
    of_record = bool(set(s["roles"]) & set(MODEL_OF_RECORD_ROLES))
    ref = entry["ptq_record"]
    if s["record_kind"] != "ptq" or (ref is None and (not of_record or entry["hashed"] is None)):
        return []
    if ref is None:
        return [f"{entry['slot']}: made, of role {s['roles']}, without its ptq_record (A2)"]
    if runs_root is None:
        return []
    S.refuse_test_path(ref["path"])
    rec = R.read_ptq_record(Path(runs_root) / ref["path"], ref["sha256"], reader=reader)
    want_role = S.PTQ_ROLE_OF_RECORD if of_record else "calibration_sensitivity"
    want = {"stage": s["evaluator_stage"], "seed": s["seed"], "role": want_role, "descriptive": not of_record}
    errs = [f"{entry['slot']}: ptq_record {k} {rec[k]!r} != {v!r}" for k, v in want.items() if rec[k] != v]
    pck = ((parent_entry or {}).get("checkpoint") or {}).get("sha256")
    if rec["parent"]["checkpoint_sha256"] != pck:
        errs.append(f"{entry['slot']}: ptq_record parent checkpoint {rec['parent']['checkpoint_sha256']} != the "
                    f"parent entry's {pck}")
    if of_record and rec["calibration"]["checksum_sha256"] != checksum_sha256:
        errs.append(f"{entry['slot']}: ptq_record calibration checksum is not the list of record's")
    h = entry["hashed"] or {}
    for k, field in (("torchscript", "artifact_sha256"), ("state_dict", "state_dict_sha256"),
                     ("x86_copy", "x86_copy_sha256")):
        if rec["artifacts"][k]["sha256"] != h.get(field):
            errs.append(f"{entry['slot']}: ptq_record {k} sha256 is not the entry's {field}")
    return errs


def not_hashed_errors(by_slot: dict) -> list[str]:
    """derived_not_hashed_at_unlock: every derived entry is hashed by the unlock (AM-19 item 4(c))."""
    return [f"{k}: still listed by inputs only" for k, e in sorted(by_slot.items())
            if e["kind"] == "derived" and e["hashed"] is None]


# ------------------------------------------------------------------------------------------------- appends
def append_errors(append: dict, frozen: dict | None, freeze_utc: int, *, isolation: dict | None = None,
                  provenance: dict | None = None, ptq_record_ok: bool = False) -> list[str]:
    """append_invalid: an append names a derived entry listed by inputs at the freeze, gives every hashed field of its
    method with an image digest of record equal to the declared one, and, when made after the freeze, cites an
    isolation record of the same root and digest made before it (decision (xi)) -- an intact ptq_record of record
    stands in for a PTQ artifact (TB19 Q9; `append_ptq_record`). `isolation` is the projected isolation record;
    `provenance` the artifact's projected PTQ provenance (None when --runs is not given: the root is not compared;
    freeze, unlock and session modes always read it)."""
    errs = S.field_errors(append, APPEND, "append")
    if errs:
        return errs
    slot = append["slot"]
    if frozen is None or frozen["kind"] != "derived" or frozen["hashed"] is not None:
        return [f"append {slot}: no derived entry listed by inputs at the freeze"]
    h = append["hashed"]
    errs = S.value_errors(h, ("obj", hashed_spec(frozen["method"])), f"append {slot}.hashed")
    if errs:
        return errs
    if h["appended"] != "append" or h["image_digest"] not in S.IMAGE_DIGESTS_OF_RECORD \
            or h["image_digest"] != frozen["inputs"]["image_digest"]:
        errs.append(f"append {slot}: image digest {h['image_digest']} is not the declared digest of record "
                    f"{frozen['inputs']['image_digest']}, or appended is {h['appended']!r}")
    made = D.parse_utc_any(h["made_utc"])
    if not D.meets(made, freeze_utc) and not (ptq_record_ok and frozen["method"] == "ptq"):
        iso = isolation or {}
        root = ((provenance or {}).get("calibration") or {}).get("data_root")
        if h["isolation"] is None or not iso:
            errs.append(f"append {slot}: made after the freeze without an isolation record (decision (xi))")
        elif iso.get("image_digest") != h["image_digest"] or not iso.get("utc") \
                or D.parse_utc_any(iso["utc"]) >= made \
                or (frozen["method"] == "ptq" and provenance is not None and iso.get("root") != root):
            errs.append(f"append {slot}: its isolation record is not of the same root ({iso.get('root')} vs {root}) "
                        f"and digest, made before it")
    return errs


def appended_entry(append: dict, frozen: dict) -> dict:
    """The by-inputs entry as its append makes it: the append's hashed form and ptq_record (A2)."""
    return dict(frozen, hashed=append["hashed"], ptq_record=append["ptq_record"])


def appendable(append, frozen: dict | None) -> bool:
    """A well-formed append of a derived entry listed by inputs at the freeze (append_invalid refuses the rest)."""
    return isinstance(append, dict) and not S.field_errors(append, APPEND, "append") \
        and isinstance(append["hashed"], dict) and frozen is not None and frozen["kind"] == "derived" \
        and frozen["hashed"] is None


def append_ptq_record(append: dict, frozen: dict, cat, by_slot: dict, checksum_sha256: str, *, runs_root,
                      reader=None) -> tuple[list[str], bool]:
    """A PTQ append's ptq_record, checked as the made entry's (ptq_record_invalid, A2): (errors, whether it stands
    in for the isolation record, TB19 Q9 -- cited and intact when read under --runs; without --runs the record is
    not read, and freeze, unlock and session modes always read it)."""
    if frozen["method"] != "ptq":
        return [], False
    errs = ptq_record_errors(appended_entry(append, frozen), cat, by_slot.get(frozen["parent"]), checksum_sha256,
                             runs_root=runs_root, reader=reader)
    return errs, not errs and append["ptq_record"] is not None


def append_reads(append: dict, frozen: dict, records: R.Records, runs_root=None) -> tuple[list, dict | None,
                                                                                            dict | None]:
    """The records an append cites, read and checked against its hashes: its isolation record (committed) and,
    under --runs, a PTQ artifact's provenance (its calibration data_root); (errors, projected isolation, projected
    provenance)."""
    h, slot, iso, prov = append["hashed"], append["slot"], None, None
    if isinstance(h.get("isolation"), dict):
        data = records.read(h["isolation"]["path"])
        if S.sha256_bytes(data) != h["isolation"]["sha256"]:
            return [f"append {slot}: its isolation record is not the cited one"], None, None
        iso = read_isolation(data, h["isolation"]["path"])
    if runs_root is not None and frozen["method"] == "ptq":                # unreadable: append_invalid
        prov, psha = R.read_json_file(Path(runs_root) / h["provenance"]["path"], R.PTQ_PROVENANCE,
                                      code="append_invalid")
        if psha != h["provenance"]["sha256"]:
            return [f"append {slot}: its provenance is not the cited one"], None, None
    return [], iso, prov


def appends_prefix_error(previous: list, current: list) -> str | None:
    """appends only grow: each committed version is a prefix of the next (plan §1.1)."""
    if current[:len(previous)] != previous:
        return f"the appends changed: {len(previous)} committed appends are not a prefix of the {len(current)} now"
    return None


# -------------------------------------------------------------------------------------- isolation (decision (xi))
ISOLATION_SPEC = {"format": ("enum", (S.ISOLATION_FORMAT,)), "root": "str", "image_digest": "?str", "utc": "utc",
                  "result": ("map", "any")}


def isolation_record(data_root, *, utc: str, image_digest: str | None, expected=None) -> dict:
    """Run src/data/isolation.assert_trainval_only_root on `data_root` (existence checks on the three TEST surfaces,
    TRAIN/VAL counted by name) and return the tday_isolation/1 record; a failure is append_invalid."""
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    S.refuse_test_path(data_root)
    try:
        result = assert_trainval_only_root(Path(data_root), expected)
    except TrainValIsolationError as e:
        raise S.TmanRefused("append_invalid", f"isolation: {e.code}: {e}") from None
    return {"format": S.ISOLATION_FORMAT, "root": str(data_root), "image_digest": image_digest, "utc": utc,
            "result": result}


def read_isolation(data: bytes, what: str) -> dict:
    doc = S.strict_json(data, what, code="append_invalid")
    errs = S.field_errors(doc, ISOLATION_SPEC, what)
    if errs:
        raise S.TmanRefused("append_invalid", "; ".join(errs))
    return doc


# ----------------------------------------------------------------------------------------- generator pins
PIN_SCHEMA = "plantseg-corruption-generator-pin/1.0.0"                 # IF§3
PIN_FIELDS = ("schema", "script", "commit", "library_versions", "parameters", "pin_sha256")
# IF§3's keys of the four fields; their presence is checked, their values are L-CORR's (TMAN re-derives no field)
PIN_KEYS = {"library_versions": ("python", "numpy", "pillow", "scikit_image", "image_digest"),
            "parameters": ("split", "expected_images", "protocol", "cells", "source_resize", "vendor_runtime_sha256",
                           "seed_policy_id", "master_seed", "cache_schema", "storage")}
PIN_CELLS_KEY = "cells"                                                # IF§3: "the 24 cell ids", in protocol order
PIN_VALUES = ("canonical_manifest_sha256", "inferential_cells_sha256", "manifest_file_sha256",
              "verification_record_sha256")                           # IF§5's recorded values: the session's
HASHER_VALUES = ("hash_list_sha256", "dedup_list_sha256", "counts_by_reason")    # plan §1.7: B-3a's, the session's
CORRUPTION_ROLES_MODULE = "src.corruption_roles"                       # IF§6: the ordered set (L-CORR)
CORRUPTION_ROLES_ATTR = "CELL_IDS"
CELL_ID_RE = re.compile(r"^(?P<corruption>[a-z_]+)_s(?P<severity>[1-5])$")
INFERENTIAL_SEVERITIES = (1, 2, 3)                                     # the corrupted scope (15 cells)


def cell_severity(cell: str) -> int:
    m = CELL_ID_RE.fullmatch(cell) if isinstance(cell, str) else None
    if not m:
        raise S.TmanRefused("generator_cells_mismatch", f"cell id {cell!r} is not <corruption>_s<severity>")
    return int(m.group("severity"))


def registered_cells(registered=None) -> list[str]:
    """The registered ordered cell set: `registered` when given (until master has src.corruption_roles; the smokes
    inject IF§6's 24 cells), else the module's, imported here and only here (TB6)."""
    if registered is not None:
        return list(registered)
    try:
        return list(getattr(importlib.import_module(CORRUPTION_ROLES_MODULE), CORRUPTION_ROLES_ATTR))
    except (ImportError, AttributeError) as e:
        raise S.TmanRefused("generator_cells_mismatch", f"{CORRUPTION_ROLES_MODULE}.{CORRUPTION_ROLES_ATTR} is not "
                                                        f"available ({e}): the registered cell set is L-CORR's "
                                                        "(TB6)") from None


def pin_sha256(pin: dict) -> str:
    """IF§3: the sha256 of canonical_json_bytes of the pin without pin_sha256."""
    return S.canonical_sha256({k: pin[k] for k in pin if k != "pin_sha256"})


def pin_incomplete_errors(pin) -> list[str]:
    """generator_pin_incomplete: the corrupted-cache pin's schema and AM-19 item 1(c)'s four fields (script, commit,
    library versions, parameters), each with IF§3's keys present; their values are L-CORR's."""
    if not isinstance(pin, dict):
        return ["the corrupted-cache pin is not an object"]
    errs = [f"pin lacks {k}" for k in PIN_FIELDS if k not in pin]
    if errs:
        return errs
    if pin["schema"] != PIN_SCHEMA:
        errs.append(f"pin schema {pin['schema']!r} != {PIN_SCHEMA!r}")
    if not (isinstance(pin["script"], dict) and pin["script"].get("path") and pin["script"].get("git_blob")):
        errs.append("pin script lacks path or git_blob")
    if not (isinstance(pin["commit"], str) and S.COMMIT_RE.fullmatch(pin["commit"])):
        errs.append("pin commit is not 40 hex")
    for block, keys in PIN_KEYS.items():
        got = pin[block] if isinstance(pin[block], dict) else {}
        errs += [f"pin {block} lacks {k}" for k in keys if got.get(k) in (None, "", [], {})]
    return errs


def pin_sha256_error(pin: dict) -> str | None:
    """generator_pin_sha256_mismatch (TB6): the recorded pin_sha256 is the recomputed one."""
    got = pin_sha256(pin)
    return None if pin.get("pin_sha256") == got else f"pin_sha256 {pin.get('pin_sha256')} != recomputed {got}"


def pin_cells_errors(pin: dict, registered=None) -> list[str]:
    """generator_cells_mismatch: the pin's cells equal the registered ordered set (R-055: 24 cells, TB6)."""
    cells = (pin.get("parameters") or {}).get(PIN_CELLS_KEY)
    reg = registered_cells(registered)
    return [] if cells == reg else [f"pin cells {cells!r} differ from the registered set {reg!r}"]


def hasher_pin_errors(pin) -> list[str]:
    """generator_pin_incomplete for the image hasher's pin (B-3a's; plan §1.7): script {path, commit, sha256}, the
    python version and its parameters."""
    if not isinstance(pin, dict):
        return ["the image-hasher pin is not an object"]
    errs = S.value_errors(pin.get("script"), ("obj", {"path": "str", "commit": "commit", "sha256": "sha256"}),
                          "image_hasher.script")
    if not (isinstance(pin.get("python"), str) and pin["python"]):
        errs.append("image_hasher lacks its python version")
    if not (isinstance(pin.get("parameters"), dict) and pin["parameters"]):
        errs.append("image_hasher lacks its parameters")
    return errs


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def test_value_errors(generator_pins: dict) -> list[str]:
    """test_value_before_unlock: before the unlock record no pin carries a value only TEST files can give (M11;
    AM-19 item 1(c)): each pin's `value` is null and no value key appears inside a pin."""
    errs = []
    for name, block in sorted((generator_pins or {}).items()):
        if not isinstance(block, dict):
            continue
        if block.get("value") is not None:
            errs.append(f"generator_pins.{name}.value is set before the unlock")
        found = sorted(set(_keys(block.get("pin"))) & set(PIN_VALUES + HASHER_VALUES))
        if found:
            errs.append(f"generator_pins.{name}.pin holds {found} before the unlock")
    return errs


def cells_for(scopes, registered=None) -> list[str]:
    """The cell ids a model's corrupted scopes ask for, in registered order: the inferential severities for
    corrupted, the others for corrupted_sev45 (IF§6; IF§8 `cells`)."""
    out = []
    for c in registered_cells(registered):
        sev = cell_severity(c)
        if ("corrupted" in scopes and sev in INFERENTIAL_SEVERITIES) or \
                ("corrupted_sev45" in scopes and sev not in INFERENTIAL_SEVERITIES):
            out.append(c)
    return out


# ------------------------------------------------------------------------------- the body's records (§1.2-§1.8)
# decision (x) and IF§10: the evaluator's roots (L-CORR's three are absent until it merges) and its data files
EVALUATOR_ROOTS = ("scripts/evaluate_model.py", "scripts/evaluate_corruptions.py", "scripts/profile_efficiency.py",
                   "scripts/build_corruption_cache.py", "scripts/verify_corruption_cache.py",
                   "scripts/run_corruption_grid.py", "scripts/report_robustness.py")
EVALUATOR_DATA = ("configs/plantseg_class_map.json", "configs/corruption_protocol.json",
                  "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py")
REFERENCE_FILES = {"mde_entry": ("reports/derived/mde_entry_20261009T034825Z.json", "DL-99"),
                   "strata": ("reports/strata/train_strata_v1.json", "DL-63")}
TRAINVAL_HASH_LIST = {"path": "train_val_image_hashes.tsv", "dl": "DL-91", "rows": 6213,       # kept outside the repo
                      "sha256": "62e5ba9d38e1de1111f59ee46d5cc8ff23d8253c546cde9fa13c11a03233837c"}
TEACHER_OF_RECORD = {"path": "iter_24000.pth", "dl": "DL-88", "bytes": 335_949_080,          # preflight_distill :93-95
                     "sha256": "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e"}
EXPECTED_ROWS_CITES = ("reports/dataset_report.md:30", "src/eval/stage_artifacts.py:30")
LANE_CODE_FILES = ("src/tman/__init__.py", "src/tman/schema.py", "src/tman/records.py", "src/tman/completion.py",
                   "src/tman/schedule.py", "src/tman/entries.py", "src/tman/manifest.py", "src/tman/validate.py",
                   "scripts/tday_manifest.py", "configs/tday_catalogue.json")


def hash_only(rel: str) -> bool:
    """A repository path holding "test" (any case): the lane never reads or parses it; it is recorded by the blob id
    git's tree gives and compared at the session by its whole-file sha256, the one operation the project's rules
    allow on such a path (src/stats/tests.py is in the evaluator's closure through src/stats/__init__.py)."""
    return "test" in rel.lower()


def evaluator_files(src: ss.HeadSource, commit: str) -> dict:
    """The evaluator file set at `commit` (decision (x); IF§10): its roots present there, their import closure over
    src/, scripts/ and configs/ (absolute, relative and function-level imports, with the packages they pass
    through), and its data files; {path: blob id}. A hash-only module is in the set but not parsed: its own imports
    are not followed."""
    out, todo = {}, [r for r in EVALUATOR_ROOTS + EVALUATOR_DATA if src.blob_at(commit, r)]
    while todo:
        rel = todo.pop()
        if rel in out:
            continue
        out[rel] = src.blob_at(commit, rel)
        if not rel.endswith(".py") or hash_only(rel):
            continue
        try:
            tree = ast.parse(src.show(commit, rel).decode("utf-8"))
        except (SyntaxError, UnicodeDecodeError, AttributeError):
            continue
        pkg = rel.rsplit("/", 1)[0].replace("/", ".")
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                mods = [base] + [f"{base}.{a.name}" for a in node.names]
            for m in mods:
                parts = m.split(".")
                if parts[0] not in ("src", "scripts", "configs"):
                    continue
                for i in range(1, len(parts) + 1):
                    stem = "/".join(parts[:i])
                    for cand in (f"{stem}.py", f"{stem}/__init__.py"):
                        if src.blob_at(commit, cand) and cand not in out:
                            todo.append(cand)
    return dict(sorted(out.items()))


def calibration_dl10_errors(records: R.Records, ci: dict, dl10_text: str | None) -> list[str]:
    """calibration_index_mismatch beyond the recomputation (§2.2; DL-24 #30, #40): each list's file sha256, and the
    list of record's checksum, in DL-10's row; each list's seed its key; its role AM-10's for seed 42 and AM-16 item
    5's for seeds 43-45."""
    errs = []
    for seed, item in [(42, ci["of_record"])] + [(int(k), v) for k, v in sorted(ci["sensitivity"].items())]:
        doc, _ = records.json(item["path"], {"seed": True, "role": True}, code="calibration_index_mismatch")
        if item["file_sha256"] not in (dl10_text or ""):
            errs.append(f"{item['path']}: sha256 {item['file_sha256'][:12]} is not DL-10's")
        if doc.get("seed") != seed or doc.get("role") != CALIBRATION_ROLES.get(seed):
            errs.append(f"{item['path']}: seed {doc.get('seed')!r} and role {doc.get('role')!r}, not {seed} and "
                        f"{CALIBRATION_ROLES.get(seed)}")
    if str(ci["of_record"]["checksum_sha256"]) not in (dl10_text or ""):
        errs.append("the list of record's checksum is not DL-10's")
    return errs


def calibration_index(records: R.Records) -> dict:
    """The calibration index of record (§1.6; DL-10) from the committed lists."""
    def one(seed):
        rel = f"configs/calibration/ptq_calibration_seed{seed}.json"
        doc, sha = records.json(rel, {"checksum_sha256": True, "role": True, "seed": True, "split_list_sha256": True},
                                code="calibration_index_mismatch")
        return {"path": rel, "file_sha256": sha, "checksum_sha256": doc.get("checksum_sha256"), "role": doc.get("role")}
    of_record = one(42)
    split = records.json(of_record["path"], {"split_list_sha256": True}, code="calibration_index_mismatch")[0]
    return {"of_record": of_record, "sensitivity": {str(s): one(s) for s in (43, 44, 45)},
            "split_list_sha256": split.get("split_list_sha256"), "dl": "DL-10"}


def references(records: R.Records) -> dict:
    """The references of record (§1.8): the MDE entry and the strata file by committed sha256, the TRAIN/VAL hash
    list and the teacher of record by their recorded values."""
    out = {}
    for k, (rel, dl) in REFERENCE_FILES.items():
        out[k] = {"path": rel, "sha256": S.sha256_bytes(records.read(rel)), "dl": dl}
    out["trainval_hash_list"] = dict(TRAINVAL_HASH_LIST)
    out["teacher_of_record"] = dict(TEACHER_OF_RECORD)
    return out


# ------------------------------------------------------------------------------------------------ the builder
SCORED_KINDS = ("scored_run", "default_candidate", "arm_teacher", "derived")      # the kinds that carry a scope
SOURCES_SPEC = {"evaluator_commit": "?commit", "execution_context": S.nullable(("map", "any")),
                "generator_pins": ("obj", {"corrupted_cache": S.nullable(("map", "any")),
                                           "image_hasher": S.nullable(("map", "any"))}),
                "trigger": S.nullable(("obj", {"path": "str", "sha256": "sha256", "triggered": "bool"})),
                "freeze_entries": ("obj", {"delay": "?str", "new_freeze": "?str"}),
                "entries": ("map", ("map", "any"))}
CODE_ROOT = Path(__file__).resolve().parents[2]


def lane_code() -> dict:
    """The lane's code as the builder runs it (the code_provenance pattern, src/quant/ptq.py:724-733)."""
    return {"commit": ss.HeadSource(CODE_ROOT).head() or None,
            "files_sha256": {rel: S.sha256_bytes((CODE_ROOT / rel).read_bytes()) for rel in LANE_CODE_FILES
                             if (CODE_ROOT / rel).is_file()}}


def entry_ref(records: R.Records, rel: str) -> dict:
    data = records.read(rel)
    return {"doc": S.strict_json(data, rel, code="entry_field"), "path": rel, "sha256": S.sha256_bytes(data)}


def build_entries(cat, given: dict, runs_root=None) -> list:
    """The body's entries in catalogue order: the catalogue's fields of each given slot, its scope (IF§9's required
    set for a kind that is scored; a pair's corrupted scope when both runs of the seed have a model, TB2), the given
    kind fields, and, with the run directories, the completion block and telemetry sha256 read from them."""
    from . import completion as C
    kinds = {k: v.get("kind") for k, v in given.items()}

    def pair(seed):
        return kinds.get(f"E1:s{seed}:80k") == "scored_run" and kinds.get(f"E6:s{seed}:sel") == "derived"
    out = []
    for slot, s in cat.slots.items():
        if slot not in given:
            continue
        g = given[slot]
        e = {"slot": slot, "kind": g.get("kind"), "class": s["class"], "stage": s["stage"], "seed": s["seed"],
             "catalogue": cat.entry_block(slot), "dl_rows": list(g.get("dl_rows", [])), "scope": [],
             "role": list(s["roles"])}
        if g.get("kind") in SCORED_KINDS:
            req, _ = S.scope_rule(s, pair(s["seed"]) if s["scope_class"] == "pair_s4344" else None)
            e["scope"] = [sc for sc in S.SCOPES if sc in req]
        e.update({k: v for k, v in g.items() if k not in S.ENTRY_COMMON})
        if runs_root is not None and g.get("kind") in COMPLETE_KINDS + ("non_finite",) and e.get("run_id"):
            S.refuse_test_path(e["run_id"])
            st = C.run_state(s, Path(runs_root) / e["run_id"])
            if st.completion is not None:
                e["completion"] = st.completion.as_entry()
            if g.get("kind") == "non_finite":
                e["telemetry_sha256"] = st.telemetry_sha256
        out.append(e)
    return out


def build_manifest(records: R.Records, cat, sources: dict, *, runs_root=None) -> dict:
    """A draft tday_manifest/1 from the committed records and the sources file (plan §1.2): the builder fills what
    the records give and copies the rest; the validator checks both."""
    errs = S.field_errors(sources, SOURCES_SPEC, "sources")
    if errs:
        raise S.TmanRefused("manifest_format", "; ".join(errs[:4]))
    src = records.src
    fe = sources["freeze_entries"]
    delays = [entry_ref(records, fe["delay"])] if fe["delay"] else []
    news = [entry_ref(records, fe["new_freeze"])] if fe["new_freeze"] else []
    freeze = D.freeze_instant(D.schedule_block(src), delays=delays, news=news)
    e = sources["evaluator_commit"]
    pins = sources["generator_pins"]
    entries = build_entries(cat, sources["entries"], runs_root)
    qat = [x for x in entries if x.get("kind") == "derived" and x.get("method") == "qat_selection"
           and isinstance(x.get("inputs"), dict)]
    rules = qat_rules_sha256(records) if qat else None
    for x in qat:                                   # TB13: the rules file's sha256 is read at build
        x["inputs"] = dict(x["inputs"])
        if x["inputs"].setdefault("rules_sha256", rules) != rules:
            raise S.TmanRefused("derived_inputs_incomplete", f"{x['slot']}: inputs.rules_sha256 is not the committed "
                                                             f"{QAT_RULES_REL}'s {rules} (TB13)")
    body = {"calibration_index": calibration_index(records),
            "entries": entries,
            "evaluator": {"commit": e, "files": evaluator_files(src, e) if e else {}},
            "execution_context": sources["execution_context"],
            "expected_rows": {"value": S.EXPECTED_ROWS, "cites": list(EXPECTED_ROWS_CITES)},
            "generator_pins": {k: {"pin": pins[k], "value": None} for k in ("corrupted_cache", "image_hasher")},
            "lane_code": lane_code(), "references": references(records), "schedule": freeze.as_block(),
            "trigger": sources["trigger"]}
    return {"format": S.MANIFEST_FORMAT, "body": body, "appends": [], "session": None}


# ------------------------------------------------------------------------------- the inventory export (IF§8)
def telemetry_name(slot: dict) -> str | None:
    """A run slot's training telemetry file (none for a teacher: its mmengine logs are not one)."""
    from . import completion as C
    k = slot["record_kind"]
    if k == "kd_run_end":
        return C.kd_telemetry_name(slot["stage"])
    return {"e1_val_result": C.E1_TELEMETRY, "qat_epoch15": C.QAT_TELEMETRY}.get(k)


def inventory_rows(body: dict, appends: list, cat, registered=None) -> list:
    """IF§8's model rows with IF§11.1's four fields: one per scored model and precision (a selected QAT model also
    has its qat_fakequant_off row, TB5), unique by (slot, precision); source.sha256, and for INT8 the converted
    artifact's (TB17); the cells its corrupted scopes ask for; a derived model listed by inputs only has no row."""
    by_slot = {e["slot"]: e for e in body["entries"]}
    hashed_by = {a["slot"]: a["hashed"] for a in appends}
    rows = []
    for e in body["entries"]:
        if e["kind"] not in SCORED_KINDS or not e["scope"]:
            continue
        s = cat.slots[e["slot"]]
        base = {"file_id": s["file_id"], "slot": e["slot"], "stage": s["evaluator_stage"],
                "model_role": s["model_role"], "seed": None if s["model_role"] == "teacher" else s["seed"],
                "entry_kind": e["kind"], "role": list(e["role"])}
        cells = cells_for(set(e["scope"]), registered)
        if e["kind"] != "derived":
            ck = e["checkpoint"]
            rows.append({**base, "precision": s["precision"],
                         "source": {"kind": "checkpoint", "path": f"{e['run_id']}/{ck['name']}",
                                    "sha256": ck["sha256"]},
                         "converted_artifact_sha256": None, "cells": cells, "training_run_id": e["run_id"]})
            continue
        h = e["hashed"] or hashed_by.get(e["slot"])
        if h is None:
            continue
        run_id = (by_slot.get(e["parent"]) or {}).get("run_id")
        rows.append({**base, "precision": s["precision"], "source": {"kind": "provenance", **h["provenance"]},
                     "converted_artifact_sha256": h["artifact_sha256"], "cells": cells, "training_run_id": run_id})
        if "fq_off" in e["scope"]:
            rows.append({**base, "precision": "qat_fakequant_off",
                         "source": {"kind": "checkpoint", **h["selected_checkpoint"]},
                         "converted_artifact_sha256": None, "cells": [], "training_run_id": run_id})
    return rows


def inventory_entries(body: dict, cat, records: R.Records, runs_root=None, appends=()) -> list:
    """IF§11.2's entries, one per catalogue slot (kind null without an entry; an appended derived entry as its append
    makes it), and IF§11.3's export-only telemetry: per attempt {run_id, path relative to the input root, sha256 of
    its bytes, never parsed}; a copy-back hash that differs, or names a file that is absent, is refused
    (completion_source_mismatch)."""
    by_slot = {e["slot"]: e for e in body["entries"]}
    appended = {a["slot"]: a for a in appends}
    out = []
    for slot, s in cat.slots.items():
        e = by_slot.get(slot)
        if e is None:
            out.append({"slot": slot, "kind": None, "class": s["class"], "stage": s["stage"], "seed": s["seed"],
                        "catalogue": cat.entry_block(slot), "role": list(s["roles"])})
            continue
        item = appended_entry(appended[slot], e) if slot in appended else dict(e)
        name = telemetry_name(s)
        run_ids, recorded = [], None
        if e["kind"] in ("scored_run", "default_candidate", "listed_run", "non_finite"):
            run_ids = [a["run_id"] for a in e["attempts"]]
            recorded = e["telemetry_sha256"] if e["kind"] == "non_finite" else (
                e["completion"]["sha256"] if e["completion"]["record"] == name else None)
        elif e["kind"] == "diverged":
            run_ids, recorded = [e["run_id"]], e["telemetry_sha256"]
        elif e["kind"] == "aborted":
            run_ids = list(e["run_ids"])
        elif e["kind"] == "cut":
            run_ids = list(S.strict_json(records.read(e["entry"]["path"]), e["entry"]["path"],
                                         code="entry_field").get("run_ids") or [])
        if name and run_ids and runs_root is not None:
            tel = []
            for rid in run_ids:
                S.refuse_test_path(rid)
                f = Path(runs_root) / rid / name
                if f.is_file():
                    tel.append({"run_id": rid, "path": f"{rid}/{name}", "sha256": S.sha256_bytes(f.read_bytes())})
            last = tel[-1] if tel and tel[-1]["run_id"] == run_ids[-1] else None
            if recorded is not None and (last is None or last["sha256"] != recorded):
                raise S.TmanRefused("completion_source_mismatch", f"{slot}: {run_ids[-1]}/{name} is absent or its "
                                                                  "sha256 is not its record's")
            item["telemetry"] = tel
        out.append(item)
    return out


def build_inventory(doc: dict, manifest_bytes: bytes, records: R.Records, cat, *, runs_root=None, registered=None,
                    freeze_dl: str | None = None) -> dict:
    """tday-inventory/1 (IF§8; IF§11 v2): manifest_sha256 is the body's canonical sha256 (the freeze record's
    body_sha256), manifest_file_sha256 the manifest file's; dedup null and binding unbound until B-3."""
    body = doc["body"]
    msha = S.canonical_sha256(body)
    return {"schema": S.INVENTORY_FORMAT, "evaluator_commit": body["evaluator"]["commit"],
            "execution_context": body["execution_context"], "manifest_sha256": msha,
            "manifest_file_sha256": S.sha256_bytes(manifest_bytes),
            "models": inventory_rows(body, doc["appends"], cat, registered),
            "entries": inventory_entries(body, cat, records, runs_root, doc["appends"]), "dedup": None,
            "freeze": {"dl_row": freeze_dl, "manifest_sha256": msha} if freeze_dl else None,
            "binding": {"state": "unbound", "path": None, "sha256": None}}
