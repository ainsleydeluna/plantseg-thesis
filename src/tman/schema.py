"""The TEST-day manifest's formats, vocabularies, names, canonical JSON and catalogue (lane TMAN, B-1, C1).

Code tag `tman`, record tag `tday`: no path this lane creates contains "test" in any case (project rule 4; DL-58),
":" or any of <>"/\\|?* inside a component (NTFS: the working clone is on Windows, T-02; IF§2).

Formats:
  tday_manifest/1   reports/derived/tday_manifest.json: the parts body, appends and session (plan §1.1). The freeze
                    record states body_sha256, the sha256 of the body's canonical JSON; from then on the validator
                    recomputes it.
  tday_entry/1      reports/derived/tday_entries/<kind>__<file_id>__<YYYYMMDD>__<nn>.json, each written once; the
                    decision-log row that records it names its path and sha256 (decision (iii); IF§2; T-20).
  tday_lanes/1      reports/derived/tday_lanes.json, the lane register (AM-19 item 3(d)).
  tday_catalogue/1  configs/tday_catalogue.json, the registered runs and derived artifacts (decision (xii); Q12).
  tday-inventory/1  the inventory export (IF§8; IF§11 v2); TMAN owns the format and may add keys.
  tday_binding/1    B-3b's binding; B-1 reserves references.mde_entry {path, sha256} (TB11) and the `artifacts`
                    records of IF§11.4.
  tday_isolation/1  the isolation record of a derived artifact made after the freeze (decision (xi)).

Slot keys (decision (ii)) keep ":" and live inside JSON only; file names use the slot's file_id, the slot key with
":" read as "_" (TB7). A manifest entry carries IF§11.2's fields: slot, kind, class, stage, seed, catalogue
{never_cut, cut_place, parent}, dl_rows, scope, role, then its kind's own. `cut_place` is the 1-based place of the
slot's cut group in the catalogue's relative cut order (TB8 item 2); kd_arm/1's cut_place, AM-18 item 7(b)'s
numbering, is the catalogue's cut_order[].arm_cut_place.

Canonical JSON is src/eval/artifacts.py's canonical_json_bytes (:93), imported: one definition (TB14). Field specs
are additive per kind (T-19): a later sub-phase registers a kind or adds a field, and an unknown kind is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import PurePosixPath

from src.eval.artifacts import canonical_json_bytes

# ------------------------------------------------------------------------------------------- formats and paths
MANIFEST_FORMAT = "tday_manifest/1"
ENTRY_FORMAT = "tday_entry/1"
LANES_FORMAT = "tday_lanes/1"
CATALOGUE_FORMAT = "tday_catalogue/1"
INVENTORY_FORMAT = "tday-inventory/1"
BINDING_FORMAT = "tday_binding/1"
ISOLATION_FORMAT = "tday_isolation/1"
PUSH_LIST_FORMAT = "tday_push_list/1"           # the session's push list (TB11; sweep_select.push_list_doc's form)

MANIFEST_REL = "reports/derived/tday_manifest.json"
ENTRIES_DIR_REL = "reports/derived/tday_entries"
LANES_REL = "reports/derived/tday_lanes.json"
CATALOGUE_REL = "configs/tday_catalogue.json"
DECISION_LOG_REL = "docs/DECISION_LOG.md"
AMENDMENTS_REL = "docs/PREREGISTRATION_AMENDMENTS.md"

MANIFEST_PARTS = ("appends", "body", "format", "session")
BODY_BLOCKS = ("calibration_index", "entries", "evaluator", "execution_context", "expected_rows", "generator_pins",
               "lane_code", "references", "schedule", "trigger")

EXPECTED_ROWS = 1561            # AM-17 item 10(a); reports/dataset_report.md:30; src/eval/stage_artifacts.py:30

# --------------------------------------------------------------------------------------------------- refusals
EXIT_PASS, EXIT_REFUSED, EXIT_MISSING, EXIT_ERROR = 0, 2, 3, 4      # the select scripts' codes (sweep_select :203)


class TmanRefused(RuntimeError):
    """A named refusal, printed `REFUSED [<code>] -- <detail>` (exit 2)."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail

    def line(self) -> str:
        return f"REFUSED [{self.code}] -- {self.detail}"


# ----------------------------------------------------------------------------------------------- vocabularies
SCOPES = ("clean", "upstream", "dedup", "corrupted", "corrupted_sev45", "efficiency", "fq_off")   # TB5; TB10
ROLES = ("family", "per_seed", "teacher_reference", "default_candidate", "arm", "arm_teacher", "control",
         "calibration_sensitivity", "listed", "quant_descriptive", "e6kd")    # IF§11.1 (TL-14) and e6kd (E6-KD)
SLOT_CLASSES = ("run", "derived", "measurement")                                                 # TB8 item 1
RECORD_KINDS = {"run": ("kd_run_end", "e1_val_result", "qat_epoch15", "teacher_m12"),
                "derived": ("ptq", "qat_selection"), "measurement": ("arm_output",)}
PRECISIONS = ("fp32", "int8_ptq", "int8_qat")
INVENTORY_PRECISIONS = PRECISIONS + ("qat_fakequant_off",)                                       # TB5 (TL-12)
MODEL_ROLES = ("teacher", "student")
SWEEP_KEYS = ("lambda_logit", "alpha_cwd")
GATE_RECORDS = ("alpha_selection", "lambda_selection", "parent_end", "r1_end", "pilot_winner",
                "arm_teacher_selection", "go_rule", "condition_entry", "trigger", "code_merged")
# AM-22 item 2(a): the records a registered rule writes that move a run's cutoff; the others move none (TB8 item 7)
CUTOFF_MOVING_GATES = ("alpha_selection", "parent_end", "r1_end", "pilot_winner", "arm_teacher_selection")

# The manifest body's entry kinds (plan §1.4; decision (xiii); IF§11.2; AM-23 item 4's "not made").
ENTRY_KINDS = ("scored_run", "default_candidate", "arm_teacher", "listed_run", "derived", "not_run", "non_finite",
               "cut", "superseded", "diverged", "aborted", "measurement", "not_made")
TERMINAL_KINDS = ("cut", "not_run", "non_finite", "superseded", "diverged", "aborted", "not_made")
# A QAT run whose selection has no model is entered on the run's slot as non_finite; its selection slot has no entry
# (IF§11.2: readers take the QAT run's kind).
KINDS_BY_CLASS = {"run": ("scored_run", "default_candidate", "arm_teacher", "listed_run", "non_finite", "cut",
                          "not_run", "superseded", "diverged", "aborted"),
                  "derived": ("derived", "not_run", "not_made"),
                  "measurement": ("measurement", "cut")}
RUN_ENDINGS = ("cut", "not_run", "superseded", "diverged", "aborted")      # a run slot's entries without a model


def allowed_kinds(slot: dict) -> tuple:
    """The entry kinds a catalogue slot takes (plan §1.4; DL-24 #2): a derived slot derived, not run or not made; a
    measurement measured or cut; a run slot its endings (non_finite for a QAT run, AM-21 item 2(b)) and the one kind
    its role gives a complete run: listed_run for a listed slot (scope class none), arm_teacher for an arm teacher,
    nothing for an arm whose code is not built, scored_run otherwise. A sweep candidate other than the selected run
    may be entered listed, scored or as a default candidate: candidate_scored and default_candidate_missing decide
    (AM-17 item 6)."""
    if slot["class"] == "derived":
        return ("derived", "not_run", "not_made")
    if slot["class"] == "measurement":
        return ("measurement", "cut")
    ends = RUN_ENDINGS + (("non_finite",) if slot["record_kind"] == "qat_epoch15" else ())
    sweep = slot["sweep"]
    if sweep and not sweep["of_record"]:
        return ("listed_run", "scored_run", "default_candidate") + ends
    model = {"none": ("listed_run",), "not_built": (), "arm_teacher": ("arm_teacher",)}
    return model.get(slot["scope_class"], ("scored_run",)) + ends


# The entry files' kinds (decision (iii)); B-2 adds rule "cutoff" and AM-22's fields of kind "freeze" (T-19).
ENTRY_FILE_KINDS = ["cut", "not_run", "freeze_delay", "new_freeze", "not_required", "unlock", "condition",
                    "not_made", "freeze", "clock_offset"]
CUT_RULES = ["freeze", "decision_date", "named_rule", "arithmetic"]          # AM-19 item 3(b); B-2 appends "cutoff"
NEVER_LAUNCHED_REASONS = ("gate not met", "code not merged", "no card")       # AM-19 item 3(b), first point
NOT_RUN_REASONS = ("parent cut", "parent diverged")                           # AM-19 item 4(a)
NOT_MADE_REASON = "code not merged"                                            # AM-23 item 4
NO_MODEL_ENTRY = "non-finite: no model"                                        # AM-21 item 2(b); qat_select.py:44
CONDITION_VERDICTS = ("yes", "no")                                             # AM-23 item 3(b)
CLOCK_OFFSET_LIMIT_S = 60                                                      # AM-19a reading 21
STUDENT_IMAGE = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"   # DL-21
KD_IMAGE = "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b"        # DL-87; TB1
IMAGE_DIGESTS_OF_RECORD = (STUDENT_IMAGE, KD_IMAGE)
SESSION_FILE_ID = "session"     # the file_id of entries about no slot (freeze, unlock, clock offset)
# ptq_record/1's roles (lane LQP, src/quant/ptq_record.py at e37e339): only of_record is a model of record (A2)
PTQ_RECORD_ROLES = ("of_record", "calibration_sensitivity", "d3_replicate")
PTQ_ROLE_OF_RECORD = "of_record"

# ------------------------------------------------------------------------- the scope table of record (IF§9)
# scope class -> (required, forbidden). fq_off is decided per slot (scope_rule); "pair_s4344" is E1 and E6 at seeds
# 43/44, whose corrupted scope depends on both runs of the seed having a model (TB2).
ALL_SCOPES = frozenset(SCOPES)
_BASE = frozenset({"clean", "upstream", "dedup"})
_ROBUST = frozenset({"corrupted", "corrupted_sev45", "efficiency"})
SCOPE_CLASSES = {
    "family_s42": (_BASE | _ROBUST, frozenset()),
    "teacher_of_record": (_BASE | _ROBUST, frozenset({"fq_off"})),
    "pair_s4344": (_BASE, frozenset({"corrupted_sev45", "efficiency"})),
    "other_s4344": (_BASE, _ROBUST),
    "default_or_control": (_BASE, _ROBUST | {"fq_off"}),
    "arm_student": (_BASE, _ROBUST | {"fq_off"}),
    "e6kd": (_BASE | {"fq_off"}, _ROBUST),
    "quant_descriptive": (_BASE, _ROBUST),
    "calibration_sensitivity": (frozenset({"clean", "dedup"}), _ROBUST | {"upstream", "fq_off"}),
    "arm_teacher": (frozenset({"clean", "dedup"}), _ROBUST | {"upstream", "fq_off"}),
    "none": (frozenset(), ALL_SCOPES),
    "not_built": (frozenset(), ALL_SCOPES),
}
# the classes whose QAT model takes fq_off when its precision is int8_qat (IF§9: E5 and E6 at each seed, E6-E2,
# E6-R1, E6-R2); e6kd requires it outright
FQ_OFF_BY_PRECISION = ("family_s42", "pair_s4344", "other_s4344", "quant_descriptive")


def scope_rule(slot: dict, pair_has_models: bool | None = None) -> tuple[frozenset, frozenset]:
    """(required, forbidden) scopes of a catalogue slot (IF§9). `pair_has_models` matters for pair_s4344 only: True
    when both runs of the seed (E1 and E6) have a model; otherwise neither takes the corrupted scope (TB2)."""
    req, forb = (set(x) for x in SCOPE_CLASSES[slot["scope_class"]])
    if slot["scope_class"] == "pair_s4344":
        (req if pair_has_models else forb).add("corrupted")
    if slot["scope_class"] in FQ_OFF_BY_PRECISION:
        (req if slot.get("precision") == "int8_qat" else forb).add("fq_off")
    return frozenset(req), frozenset(forb)


# ----------------------------------------------------------------------------------------- names (IF§2; T-02)
SLOT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*(?::[A-Za-z0-9.]+)+$")
FILE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENTRY_FILE_RE = re.compile(r"^(?P<kind>[a-z_]+)__(?P<file_id>[A-Za-z0-9._-]+)__(?P<date>\d{8})__(?P<nn>\d{2})\.json$")
NTFS_RESERVED = '<>"/\\|?*'
DL_ID_RE = re.compile(r"DL-\d+[a-z]?")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ISO_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ISO_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def file_id_of(slot: str) -> str:
    return slot.replace(":", "_")


def file_id_error(file_id) -> str | None:
    """IF§2: [A-Za-z0-9._-]+, no "__", no "test" in any case."""
    if not isinstance(file_id, str) or not FILE_ID_RE.fullmatch(file_id):
        return f"file_id {file_id!r} is not [A-Za-z0-9._-]+"
    if "__" in file_id:
        return f"file_id {file_id!r} contains '__' (the entry-file separator)"
    if "test" in file_id.lower():
        return f"file_id {file_id!r} contains 'test' (project rule 4)"
    return None


def created_name_error(path) -> str | None:
    """A path this lane creates: no "test" in any case, no ":" and none of <>"/\\|?* inside a component (IF§2; T-02;
    TB7). Checked as a string; nothing is touched."""
    s = str(path)
    if "test" in s.lower():
        return f"{s!r} contains 'test'"
    for i, part in enumerate(PurePosixPath(s.replace("\\", "/")).parts):
        if i == 0 and part == "/":
            continue
        if ":" in part or any(c in part for c in NTFS_RESERVED):
            return f"{s!r}: its component {part!r} holds ':' or one of {NTFS_RESERVED}"
    return None


def entry_file_name(kind: str, file_id: str, day: str, nn: int) -> str:
    """<kind>__<file_id>__<YYYYMMDD>__<nn>.json (IF§2; T-20: nn is a two-digit sequence starting at 01)."""
    err = file_id_error(file_id)
    if err or kind not in ENTRY_FILE_KINDS or not (isinstance(day, str) and ISO_DAY_RE.fullmatch(day)) \
            or not (isinstance(nn, int) and not isinstance(nn, bool) and 1 <= nn <= 99):
        raise TmanRefused("entry_field", f"entry file name from kind {kind!r}, file_id {file_id!r}, day {day!r}, "
                                         f"nn {nn!r}" + (f": {err}" if err else ""))
    return f"{kind}__{file_id}__{day.replace('-', '')}__{nn:02d}.json"


def entry_file_rel(name: str) -> str:
    return f"{ENTRIES_DIR_REL}/{name}"


def test_path_error(path) -> str | None:
    """An input path containing "test" (any case): refused as given, before any filesystem call, then as resolved (a
    link into a "test" location) -- the rule of src/training/kd_arms.py:130-136."""
    if "test" in str(path).lower():
        return f"{path}: a path containing 'test' is refused (project rule 4)"
    if "test" in os.path.realpath(path).lower():
        return f"{path}: a path resolving to a 'test' location is refused (project rule 4)"
    return None


def refuse_test_path(path) -> None:
    err = test_path_error(path)
    if err:
        raise TmanRefused("test_path", err)


# ------------------------------------------------------------------------------------ canonical JSON (TB14)
def canonical_sha256(obj) -> str:
    """sha256 of src.eval.artifacts.canonical_json_bytes(obj)."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_file_bytes(doc) -> bytes:
    """A record file's bytes: sorted keys, two-space indent, UTF-8, LF, non-finite values refused."""
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def strict_json(data: bytes, what: str, *, code: str, object_type=dict):
    """JSON from UTF-8 bytes; a duplicate key, NaN or +-Infinity is refused as `code`. `object_type` builds each
    object (the smokes' sentinel)."""
    def pairs(items):
        seen = set()
        for k, _ in items:
            if k in seen:
                raise ValueError(f"duplicate key {k!r}")
            seen.add(k)
        return object_type(items)

    def constant(c):
        raise ValueError(f"non-finite constant {c}")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise TmanRefused(code, f"{what}: not strict UTF-8 JSON ({type(e).__name__}: {e})") from None


# --------------------------------------------------------------------------------- field specs (additive, T-19)
# A spec is a type name; "?<type>" (nullable); ("enum", values); ("list", spec); ("obj", {field: spec}) with exactly
# those keys; ("map", spec) with any string keys; ("nullable", spec).
def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


TYPES = {
    "str": lambda x: isinstance(x, str) and x != "",
    "int": _is_int,
    "bool": lambda x: isinstance(x, bool),
    "day": lambda x: isinstance(x, str) and ISO_DAY_RE.fullmatch(x) is not None,
    "utc": lambda x: isinstance(x, str) and ISO_UTC_RE.fullmatch(x) is not None,
    "sha256": lambda x: isinstance(x, str) and SHA256_RE.fullmatch(x) is not None,
    "commit": lambda x: isinstance(x, str) and COMMIT_RE.fullmatch(x) is not None,
    "dl": lambda x: isinstance(x, str) and DL_ID_RE.fullmatch(x) is not None,
    "slot": lambda x: isinstance(x, str) and SLOT_KEY_RE.fullmatch(x) is not None,
    "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    "any": lambda x: True,
}


def nullable(spec):
    return ("nullable", spec)


REF = ("obj", {"path": "str", "sha256": "sha256"})


def value_errors(x, spec, where: str) -> list[str]:
    if isinstance(spec, str):
        if spec.startswith("?"):
            return [] if x is None else value_errors(x, spec[1:], where)
        return [] if TYPES[spec](x) else [f"{where}: {x!r} is not {spec}"]
    kind, arg = spec
    if kind == "nullable":
        return [] if x is None else value_errors(x, arg, where)
    if kind == "enum":
        return [] if isinstance(x, str) and x in arg else [f"{where}: {x!r} is not one of {list(arg)}"]
    if kind == "list":
        if not isinstance(x, list):
            return [f"{where}: not a list"]
        return [e for i, v in enumerate(x) for e in value_errors(v, arg, f"{where}[{i}]")]
    if kind == "map":
        if not isinstance(x, dict):
            return [f"{where}: not an object"]
        return [e for k in sorted(x) for e in value_errors(x[k], arg, f"{where}.{k}")]
    if kind == "obj":
        return field_errors(x, arg, where)
    raise ValueError(f"unknown field spec {spec!r}")


def field_errors(doc, spec: dict, where: str) -> list[str]:
    """Every missing field, unexpected field and wrong type of `doc` against `spec` (exact key sets)."""
    if not isinstance(doc, dict):
        return [f"{where}: not a JSON object"]
    errs = []
    missing, extra = sorted(set(spec) - set(doc)), sorted(set(doc) - set(spec))
    if missing:
        errs.append(f"{where}: missing {missing}")
    if extra:
        errs.append(f"{where}: unexpected {extra}")
    for k in sorted(set(spec) & set(doc)):
        errs += value_errors(doc[k], spec[k], f"{where}.{k}")
    return errs


def register_kind(table: dict, kind: str, fields: dict) -> None:
    """Additive per-kind specs (T-19): register a new kind, or add fields to a registered one."""
    table.setdefault(kind, {}).update(fields)


# ------------------------------------------------------------------------- the manifest's entries (IF§11.2)
ENTRY_COMMON = {"slot": "slot", "kind": ("enum", ENTRY_KINDS), "class": ("enum", SLOT_CLASSES), "stage": "str",
                "seed": "?int", "catalogue": ("obj", {"never_cut": "bool", "cut_place": "?int", "parent": "?slot"}),
                "dl_rows": ("list", "dl"), "scope": ("list", ("enum", SCOPES)), "role": ("list", ("enum", ROLES))}
COMPLETION = ("obj", {"record": "str", "at_utc": "utc", "sha256": "sha256", "line": "?int"})
CHECKPOINT = ("obj", {"name": "str", "sha256": "sha256", "bytes": "?int",
                      "sources": ("list", ("obj", {"kind": "str", "path": "?str", "dl": "?dl", "sha256": "sha256"}))})
ATTEMPTS = ("list", ("obj", {"run_id": "str", "attempt": "int"}))
_RUN = {"run_id": "str", "attempts": ATTEMPTS, "completion": COMPLETION}
_SCORED = {**_RUN, "checkpoint": CHECKPOINT, "evaluator_stage": "str", "teacher_ckpt_sha256": "?sha256"}
ENTRY_FIELDS: dict[str, dict] = {
    "scored_run": dict(_SCORED),
    "default_candidate": {**_SCORED, "selection": REF},
    "arm_teacher": {**_SCORED, "arm_record": REF, "launch_provenance": REF,
                    "go_rule": nullable(("obj", {"path": "str", "sha256": "sha256",
                                                 "verdict": ("enum", ("go", "no_go", "not_evaluable"))}))},
    "listed_run": {**_RUN, "checkpoint": nullable(CHECKPOINT), "reason": "str"},
    "derived": {"method": ("enum", ("ptq", "qat_selection")), "parent": "slot", "inputs": ("map", "any"),
                "hashed": nullable(("map", "any")), "evaluator_stage": "str", "ptq_record": nullable(REF)},
    "not_run": {"reason": ("enum", NOT_RUN_REASONS), "parent": "slot", "parent_record": REF, "entry": REF},
    "non_finite": {**_RUN, "telemetry_sha256": "sha256", "refusal_text": ("enum", (NO_MODEL_ENTRY,))},
    "cut": {"entry": REF, "cut_date": "day", "rule": "str", "comparator": "?slot"},
    "superseded": {"run_id": "str", "superseding_dl": "dl"},
    "diverged": {"run_id": "str", "rule": "str", "cause": "str", "iteration": "?int", "epoch": "?int",
                 "telemetry_sha256": "sha256"},
    "aborted": {"run_ids": ("list", "str"), "rule": "str", "cause": "str"},
    "measurement": {"output": REF, "agreement": REF, "completion": COMPLETION},
    "not_made": {"reason": ("enum", (NOT_MADE_REASON,)), "entry": REF},
}


def entry_errors(entry) -> list[str]:
    """IF§11.2's common fields and the kind's own (exact key sets); a diverged entry gives its iteration or, for a
    QAT run, its epoch."""
    kind = entry.get("kind") if isinstance(entry, dict) else None
    if kind not in ENTRY_FIELDS:
        return [f"entry kind {kind!r} is not one of {list(ENTRY_FIELDS)}"]
    errs = field_errors(entry, {**ENTRY_COMMON, **ENTRY_FIELDS[kind]}, f"entry {entry.get('slot')!r}")
    if not errs and kind == "diverged" and (entry["iteration"] is None) == (entry["epoch"] is None):
        errs.append(f"entry {entry['slot']!r}: a diverged entry gives exactly one of iteration and epoch")
    return errs


# ------------------------------------------------------------------- the binding's reserved fields (TB11; IF§11.4)
BINDING_SCOPES = ("clean", "upstream", "corrupted", "corrupted_sev45", "fq_off")       # no dedup: no model run
BINDING_ARTIFACT_FIELDS = {"slot": "slot", "precision": ("enum", INVENTORY_PRECISIONS),
                           "scope": ("enum", BINDING_SCOPES), "cell": "?str",
                           "protocol": ("enum", ("canvas", "upstream")), "repo_relative_path": "str",
                           "files_sha256": ("map", "sha256"), "checkpoint_sha256": "sha256", "repo_commit": "commit",
                           "split_manifest_sha256": "sha256"}
BINDING_RESERVED = {"references": ("mde_entry",), "artifacts": tuple(sorted(BINDING_ARTIFACT_FIELDS))}

# ------------------------------------------------------------------------------------ the catalogue (Q12; TB8)
SLOT_SPEC = {"slot": "slot", "file_id": "str", "class": ("enum", SLOT_CLASSES), "stage": "str", "seed": "?int",
             "params": ("map", "number"), "record_kind": "str", "evaluator_stage": "?str",
             "precision": nullable(("enum", PRECISIONS)), "model_role": nullable(("enum", MODEL_ROLES)),
             "never_cut": "bool", "cut_group": "?str", "parent": "?slot", "roles": ("list", ("enum", ROLES)),
             "scope_class": ("enum", tuple(SCOPE_CLASSES)),
             "gates": ("list", ("obj", {"record": ("enum", GATE_RECORDS), "moves_cutoff": "bool"})),
             "registered_by": "str", "sweep": nullable(("map", "any")), "pilot": nullable(("map", "any"))}
SWEEP_SPEC = {"key": ("enum", SWEEP_KEYS), "value": "number", "default_candidate": "bool", "of_record": "bool"}
PARAM_KEYS = ("alpha", "clip", "lambda", "list")
CUT_ORDER_SPEC = {"group": "str", "source": "str", "arm_cut_place": "?int"}


class Catalogue:
    """configs/tday_catalogue.json, checked (catalogue_invalid). Slots keep the file's order."""

    def __init__(self, doc: dict, sha256: str):
        self.doc, self.sha256 = doc, sha256
        self.slots = {s["slot"]: s for s in doc["slots"]}
        self.by_file_id = {s["file_id"]: s for s in doc["slots"]}
        self.groups = [g["group"] for g in doc["cut_order"]]
        self.arm_places = {g["arm_cut_place"]: g["group"] for g in doc["cut_order"] if g["arm_cut_place"] is not None}
        self.children: dict[str, list[str]] = {}
        for s in doc["slots"]:
            if s["parent"]:
                self.children.setdefault(s["parent"], []).append(s["slot"])

    def position(self, group: str | None) -> int | None:
        """The 1-based place of a cut group in the relative cut order, first to last (TB8 item 2)."""
        return None if group is None else self.groups.index(group) + 1

    def cut_place(self, slot: str) -> int | None:
        return self.position(self.slots[slot]["cut_group"])

    def entry_block(self, slot: str) -> dict:
        """IF§11.2's `catalogue` block of a slot's entry."""
        s = self.slots[slot]
        return {"never_cut": s["never_cut"], "cut_place": self.cut_place(slot), "parent": s["parent"]}

    def group_slots(self, group: str) -> list[str]:
        return [k for k, s in self.slots.items() if s["cut_group"] == group]

    def sweep_slots(self, key: str) -> list[dict]:
        return [s for s in self.slots.values() if s["sweep"] and s["sweep"]["key"] == key]


def catalogue_errors(doc) -> list[str]:
    """Every way `doc` breaks tday_catalogue/1."""
    if not isinstance(doc, dict) or sorted(doc) != ["cut_order", "format", "slots"]:
        return ["the catalogue is not an object with exactly format, cut_order and slots"]
    errs = [] if doc["format"] == CATALOGUE_FORMAT else [f"format {doc['format']!r} != {CATALOGUE_FORMAT!r}"]
    order = doc["cut_order"] if isinstance(doc["cut_order"], list) else []
    errs += [e for i, g in enumerate(order) for e in field_errors(g, CUT_ORDER_SPEC, f"cut_order[{i}]")]
    groups = [g.get("group") for g in order if isinstance(g, dict)]
    if len(set(groups)) != len(groups):
        errs.append("cut_order repeats a group")
    places = [g["arm_cut_place"] for g in order if isinstance(g, dict) and _is_int(g.get("arm_cut_place"))]
    if places != sorted(set(places)):
        errs.append(f"arm cut places {places} are not in increasing relative order (AM-18 item 7(b))")
    slots = doc["slots"] if isinstance(doc["slots"], list) else []
    keys = [s.get("slot") for s in slots if isinstance(s, dict)]
    if len(set(keys)) != len(keys):
        errs.append("two slots share a key")
    files = [s.get("file_id") for s in slots if isinstance(s, dict)]
    if len(set(files)) != len(files):
        errs.append("two slots share a file_id")
    by_key = {s.get("slot"): s for s in slots if isinstance(s, dict)}
    for s in slots:
        where = f"slot {s.get('slot') if isinstance(s, dict) else s!r}"
        e = field_errors(s, SLOT_SPEC, where)
        if e:
            errs += e
            continue
        if s["file_id"] != file_id_of(s["slot"]) or file_id_error(s["file_id"]):
            errs.append(f"{where}: file_id {s['file_id']!r} is not the slot key with ':' read as '_', or breaks "
                        f"IF§2 ({file_id_error(s['file_id'])})")
        if s["record_kind"] not in RECORD_KINDS[s["class"]]:
            errs.append(f"{where}: record kind {s['record_kind']!r} is not one of class {s['class']}'s")
        if set(s["params"]) - set(PARAM_KEYS):
            errs.append(f"{where}: params keys {sorted(s['params'])} outside {PARAM_KEYS}")
        if s["cut_group"] is not None and s["cut_group"] not in groups:
            errs.append(f"{where}: cut group {s['cut_group']!r} is not in cut_order")
        if s["never_cut"] and s["cut_group"] is not None:
            errs.append(f"{where}: a never-cut slot has no cut group")
        if s["parent"] is not None and s["parent"] not in by_key:
            errs.append(f"{where}: parent {s['parent']!r} is not a slot")
        for g in s["gates"]:
            if g["moves_cutoff"] != (g["record"] in CUTOFF_MOVING_GATES):
                errs.append(f"{where}: gate {g['record']} moves_cutoff {g['moves_cutoff']} (AM-22 item 2(a))")
        if s["sweep"] is not None:
            errs += field_errors(s["sweep"], SWEEP_SPEC, f"{where}.sweep")
        if s["pilot"] is not None:
            runs = s["pilot"].get("runs")
            if sorted(s["pilot"]) == ["runs"] and isinstance(runs, list):
                if not runs or any((by_key.get(r) or {}).get("pilot") != {"of": s["slot"]} for r in runs):
                    errs.append(f"{where}: its pilot runs {runs} do not each name it as {{'of': {s['slot']!r}}}")
            elif sorted(s["pilot"]) != ["of"] or s["pilot"]["of"] not in by_key:
                errs.append(f"{where}: pilot {s['pilot']!r} is neither {{'runs': [...]}} nor {{'of': <slot>}}")
        required, _ = SCOPE_CLASSES[s["scope_class"]]
        if required and None in (s["evaluator_stage"], s["precision"], s["model_role"]):
            errs.append(f"{where}: a scored scope class needs evaluator_stage, precision and model_role")
    for key in SWEEP_KEYS:
        sw = [s["sweep"] for s in slots if isinstance(s, dict) and isinstance(s.get("sweep"), dict)
              and s["sweep"].get("key") == key]
        if sw and (sum(x.get("of_record") is True for x in sw) != 1
                   or sum(x.get("default_candidate") is True for x in sw) != 1):
            errs.append(f"sweep {key}: exactly one slot of record and one default candidate")
    return errs


def load_catalogue(data: bytes, what: str = CATALOGUE_REL) -> Catalogue:
    doc = strict_json(data, what, code="catalogue_invalid")
    errs = catalogue_errors(doc)
    if errs:
        raise TmanRefused("catalogue_invalid", f"{what}: " + "; ".join(errs[:6]) + (" ..." if len(errs) > 6 else ""))
    return Catalogue(doc, sha256_bytes(data))
