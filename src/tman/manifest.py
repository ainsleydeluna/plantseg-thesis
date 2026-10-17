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
configs/qat_selection_rules.json's sha256 (read at build, TB13) and the pilot's clip-selection record's sha256.

Generator pins (AM-19 item 1(c)): the corrupted-TEST cache pin is L-CORR's JSON, embedded verbatim (IF§3): the
validator recomputes pin_sha256 and compares its cells with the registered set (src.corruption_roles, imported
lazily; until master has the module the set comes through a parameter, TB6). The image hasher's pin is B-3a's.
Neither carries a value before the unlock record (M11; test_value_before_unlock).
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

from . import records as R
from . import schedule as D
from . import schema as S

# ----------------------------------------------------------------------------------------- derived (§1.5)
PTQ_INPUTS = {"parent_checkpoint_sha256": "sha256",
              "calibration_list": ("obj", {"path": "str", "file_sha256": "sha256"}),
              "script": ("obj", {"path": "str", "commit": "commit"}),
              "image_digest": "str", "qconfig_fingerprint": "str"}
QAT_INPUTS = {"run_slot": "slot", "telemetry_sha256": "sha256", "scripts_commit": "commit", "image_digest": "str",
              "rules_sha256": "sha256", "clip_selection_sha256": "sha256"}
HASHED = {"artifact_sha256": "sha256", "state_dict_sha256": "sha256", "x86_copy_sha256": "sha256",
          "provenance": S.REF, "record": S.REF, "made_utc": "utc", "image_digest": "str",
          "isolation": S.nullable(S.REF), "appended": ("enum", ("freeze", "append"))}
# AM-21 item 2(c): a second evaluation directory of a QAT run needs its fault report
HASHED_QAT = {**HASHED, "evaluations": ("list", ("obj", {"dir": "str", "fault_report": S.nullable(S.REF)}))}
APPEND = {"slot": "slot", "hashed": ("map", "any")}
QAT_RULES_REL = "configs/qat_selection_rules.json"
COMPLETE_KINDS = ("scored_run", "default_candidate", "arm_teacher", "listed_run")     # a complete run: a model
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
    """derived_missing: a required derived slot without a derived entry, or a not-made entry for a slot whose code
    is merged (E7-E2's rule, AM-23 items 2 and 4)."""
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
    ev = entry["hashed"]["evaluations"]
    if not ev or any(e["fault_report"] is None for e in ev[1:]):
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
    stands in for a PTQ artifact (TB19 Q9). `isolation` is the projected isolation record; `provenance` the
    artifact's projected PTQ provenance."""
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
                or D.parse_utc_any(iso["utc"]) >= made or (frozen["method"] == "ptq" and iso.get("root") != root):
            errs.append(f"append {slot}: its isolation record is not of the same root ({iso.get('root')} vs {root}) "
                        f"and digest, made before it")
    return errs


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
