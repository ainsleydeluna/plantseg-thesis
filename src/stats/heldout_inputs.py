"""LREP's read-only input layer for the TEST tables beyond the 37 seed-42 inputs (lane LREP, B-1; B-5).

Interface of record: IF§11 v2 (tl_interface_lrep_20261016.md) over IF§8's inventory export
`tday-inventory/1` and TMAN's binding `tday_binding/1`. TMAN builds every field; LREP reads them, never
writes them, never parses the TEST-day manifest and never builds its path (IF§11.6).

  * The reader. `read_inventory(path, *, expect_sha256=None)` is a STUB with the signature of
    `src.tman.validate.read_inventory` (IF§11.6). It serves LREP's synthetic fixtures until TMAN
    merges; `open_inventory` refuses it under Policy.OFFICIAL, and B-4 swaps in TMAN's reader. The
    stub resolves `binding.path` against the export file's directory (an assumption B-4 settles).
  * Slots are opaque keys (IF§11.1). The model of record of a planned cell is the slot whose role
    includes `family` (seed 42) or `per_seed` (seeds 43/44). A slot or row a table does not read is
    ignored by that table and never refused (LRB4), so the export is validated lazily: only what is
    read is checked.
  * Status: TMAN's kinds verbatim, mapped to the AM-19 item 6(e) cell labels by IF§11.2's table. A QAT
    selection that is not `derived` reads its QAT run's kind; a PTQ model with kind null takes its
    parent's cut or divergence; `superseded` is refused (R-5); parentage runs through the QAT run
    (R-12; LRB5).
  * Every scored artifact outside the 37 is read through `driver.load_input` (four-file hash
    snapshots, MANIFEST agreement, canvas guard, declared identity, the policy's row rules) after its
    files are compared with its binding `artifacts` record, and every field of that record is checked
    (R-7; LRB3). The stage map (B5-S1, STAGE_MAP) gives each TMAN stage its evaluator stage, model_role
    and precision; (model_role, precision) must be (student, qat_fakequant_off) for fq_off rows (LRB7)
    and the map's for the others.
  * Telemetry files are hashed before they are parsed; only their VAL rows are read (LRB2).
  * B-5 (rulings LREP-B5-RULE-1 to LREP-B5-RULE-3): the readers of the slots of role arm, arm_teacher
    and quant_descriptive with the family slots they compare with (IF§11.1 :25), and the
    duplicate-free TEST list (B5-S2), in section 8.
  * B-5b (ruling LREP-B5-RULE-4 R4-1): the descriptive TEST read ("DT read", section 9). B5-S1's map gains
    the column `read`, set from a slot's role and its stage's precision, never from its artifact's status
    (slot_read): descriptive_test for the slots the evaluator scores descriptively only (role arm at
    int8_ptq: E7-R1, E7-R2, E7-flip; role calibration_sensitivity: E4 and E7 at lists 43-45), official for
    every other. Under Policy.OFFICIAL a descriptive_test slot's provisional TEST artifact is read by
    read_descriptive_test, each step a frozen function and any failure descriptive_input_invalid; its run
    keeps Policy.NONOFFICIAL_SMOKE, and a pair with it is aligned under that policy (dt_pair). Every other
    slot reads under Policy.OFFICIAL as before, with no downgrade path (load_slot), and an official
    computation takes the 37 inputs only (require_official_inputs: descriptive_input_in_official).

Every refusal is an `LrepRefusal` with a code from CODES, raised at a site that reads
`raise refusal("<code>", ...)` (scripts/smoke_lrep_mutations.py mutates those sites), except the three
raises of `strict_json_text`, which carry their caller's code (inventory_unreadable, binding_invalid,
dedup_invalid or arm_record_invalid).
Nothing here writes a file. Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..eval.protocols import CANVAS_PROTOCOL_ID
from . import driver as D
from .align import AlignmentError, assert_compatible
from .ingest import (EXPECTED_ROWS_TEST, IngestError, Policy, am5_pair, apply_am5_policy,
                     verify_run_manifest)

INVENTORY_SCHEMA = "tday-inventory/1"
BINDING_SCHEMA = "tday_binding/1"
#: TMAN's kinds and roles, verbatim (IF§11.1-11.2; TL-14); role e6kd is E6-KD's (src/tman/schema.py:81-82 at
#: lane/tman ee5593e; ruling LREP-B5-RULE-4 R4-5(c)).
KINDS = ("scored_run", "default_candidate", "arm_teacher", "listed_run", "derived", "not_run",
         "non_finite", "cut", "superseded", "diverged", "aborted", "measurement", "not_made")
ROLES = ("family", "per_seed", "teacher_reference", "default_candidate", "arm", "arm_teacher",
         "control", "calibration_sensitivity", "listed", "quant_descriptive", "e6kd")
#: IF§11.1: a scored row's entry_kind.
SCORED_ENTRY_KINDS = ("scored_run", "default_candidate", "arm_teacher", "derived")
SCOPES = ("clean", "upstream", "corrupted", "corrupted_sev45", "fq_off")
PROTOCOLS = ("canvas", "upstream")
PRECISIONS = ("fp32", "int8_ptq", "int8_qat", "qat_fakequant_off")
INT8_PRECISIONS = ("int8_ptq", "int8_qat")
PRECISION_FQOFF = "qat_fakequant_off"
BINDING_STATES = ("bound", "unbound")
ARTIFACT_FILE_NAMES = tuple(sorted(D.INPUT_FILES))

#: The 21 planned cells: E1-E7 at seeds 42, 43 and 44 (AM-1; AM-16 item 1; AM-17 item 4), runs for
#: E1-E3 and derived slots for E4-E7 (LRB19 L-18), keyed by the role of their model-of-record slot
#: (IF§11.1).
STAGES = D.STAGES
SEEDS = (42, 43, 44)
RECORD_ROLES = {"family": (42,), "per_seed": (43, 44)}
RUN_STAGES = ("E1", "E2", "E3")               # the model of record is a run (`scored_run`)
PTQ_STAGES = ("E4", "E7")                     # a derived PTQ model of its parent run
QAT_STAGES = ("E5", "E6")                     # a derived selection of a QAT run
#: AM-19 item 4(a); AM-1: E4 and E5 are built from E1, E6 and E7 from E3, at the same seed.
PARENT_STAGE = {"E4": "E1", "E5": "E1", "E6": "E3", "E7": "E3"}

#: B5-S1 (ruling LREP-B5-RULE-2 item 1): every TMAN stage LREP reads -> (evaluator stage, model_role,
#: precision); load_bound reads it for every row. Each precision comes from the frozen tables
#: (driver.STAGE_PRECISION, src/eval/stage_artifacts.STAGE_ARTIFACTS), never typed. B-4's second
#: assumption (E6-KD scored as E6) is retired: the evaluator names E6-KD (src/eval/artifacts.py:58-59;
#: stage_artifacts.py:56-58). The stages are TMAN's committed catalogue's (configs/tday_catalogue.json at
#: lane/tman ee5593e, blob 3cc45846; R4-5(a)): E3-X and E7-X at evaluator stages E3 and E7, and the arm
#: teachers teacher-R1 and teacher-R2 at "teacher" (their evaluator_stage there); B5-S1's teacher stages
#: "R1" and "R2" are retired. A stage the catalogue leaves unscored (evaluator_stage null: the not-built L
#: slots, ARM's measurement) is outside the map and never mapped.
ARM_XS = ("R1", "R2", "flip")
ARM_STUDENT_STAGES = tuple(f"E3-{x}" for x in ARM_XS) + tuple(f"E7-{x}" for x in ARM_XS)
QUANT_STAGES = ("E6-E2", "E7-E2", "E6-R1", "E6-R2")          # AM-23 items 1-3 (quant_descriptive)
EXPLORATORY_STAGES = ("A", "F", "G")
E6KD_STAGE = "E6-KD"
TEACHER_STAGES = ("teacher", "teacher-R1", "teacher-R2")     # the teacher of record; the arm teachers
EVALUATOR_TEACHER_STAGE = "teacher"                          # src/eval/artifacts.py:58


def _stage_map() -> dict:
    from ..eval.stage_artifacts import STAGE_ARTIFACTS, STUDENT_ROLE, TEACHER_ROLE, TEACHER_STAGE
    out = {st: (st, STUDENT_ROLE, D.STAGE_PRECISION[st]) for st in D.STAGES}
    for parent in ("E3", "E7"):
        for x in ARM_XS:
            out[f"{parent}-{x}"] = (parent, STUDENT_ROLE, D.STAGE_PRECISION[parent])
    for st in (*QUANT_STAGES, E6KD_STAGE, *EXPLORATORY_STAGES):
        out[st] = (st, STUDENT_ROLE, STAGE_ARTIFACTS[st]["precision"])
    for st in TEACHER_STAGES:
        out[st] = (EVALUATOR_TEACHER_STAGE, TEACHER_ROLE, STAGE_ARTIFACTS[TEACHER_STAGE]["precision"])
    return out


STAGE_MAP = _stage_map()
#: Every mapped stage in one fixed order (the AM-6 table's sort key covers every stage).
STAGE_ORDER = tuple(STAGE_MAP)
#: R4-1 item 2: B5-S1's column `read` of a slot (slot_read), set from its role and its stage's precision,
#: never from its artifact's status. descriptive_test: the slots whose INT8 TEST artifact the evaluator
#: refuses to score official (scripts/evaluate_model.py:263-270; src/eval/stage_artifacts.py:199, :209-240):
#: role arm at int8_ptq (E7-R1, E7-R2 and E7-flip at list 42: ptq_record_missing) and role
#: calibration_sensitivity (E4 and E7 at lists 43-45: ptq_record_role); official: every other slot.
READ_OFFICIAL = "official"
READ_DESCRIPTIVE_TEST = "descriptive_test"
READS = (READ_OFFICIAL, READ_DESCRIPTIVE_TEST)
DT_ROLE = "calibration_sensitivity"
DT_ARM_PRECISION = "int8_ptq"
#: R4-1 item 6: what every row built from a DT run prints, beside the existing banner.
DT_LABEL = "descriptive TEST; provisional artifact (the evaluator scores this stage descriptively only)"
#: R4-1 item 3: the rule text of each step of the DT read, which descriptive_input_invalid names.
DT_STEPS = {
    "a": "read by load_bound, the loader of every slot (four-file hashes, MANIFEST verify, the canvas guard "
         "driver.refuse_non_canvas at driver.py:272), under Policy.NONOFFICIAL_SMOKE",
    "b": "artifact_status provisional, split test, random_init false and expected_rows == actual_rows == "
         f"ingest.EXPECTED_ROWS_TEST ({EXPECTED_ROWS_TEST})",
    "c": "rows ingest.apply_am5_policy(rows, Policy.OFFICIAL): the official AM-5 rule, any other undefined "
         "primary value aborts",
    "d": "its reference, the 37's input of its evaluator stage (the family slot at seed 42) read under "
         "Policy.OFFICIAL, passes align.assert_compatible under Policy.OFFICIAL and ingest.am5_pair with it",
    "e": "its pair through align.align_runs under Policy.NONOFFICIAL_SMOKE, the reference re-loaded under "
         "that policy with (c)'s rows and its files those of its official read",
    "f": "its files, checkpoint, commit and split manifest equal its binding record (R-7)"}
#: The run each of B-5's derived stages comes from (AM-18 item 3(a); AM-23 items 1-3; AM-3): a PTQ model
#: of its parent run, or a QAT run's selection whose run descends from it (IF§11.2).
B5_PARENT_STAGE = {"E7-R1": "E3-R1", "E7-R2": "E3-R2", "E7-flip": "E3-flip", "E7-E2": "E2",
                   "E6-R1": "E3-R1", "E6-R2": "E3-R2", "E6-E2": "E2", E6KD_STAGE: "E3"}
B5_PTQ_STAGES = tuple(s for s in B5_PARENT_STAGE if STAGE_MAP[s][2] == "int8_ptq")
B5_QAT_STAGES = tuple(s for s in B5_PARENT_STAGE if STAGE_MAP[s][2] == "int8_qat")
#: The kinds of a B-5 run slot that carry a model (IF§11.1-11.2): a scored run, or an arm's teacher.
B5_RUN_MODEL_KINDS = ("scored_run", "arm_teacher")
#: The family slots B-5's tables compare with, at seed 42 (IF§11.1 :25).
B5_FAMILY_STAGES = ("E2", "E3", "E6", "E7")
#: B5-S2: the fields of the export's `dedup` (IF§11.5) and of the file it names (T4's wording, plan :62;
#: lane/tman holds no committed schema for it at STEP 1).
DEDUP_EXPORT_FIELDS = ("path", "sha256", "removed_by_reason")
DEDUP_FILE_FIELDS = ("kept_ids", "removed_by_reason")

#: DL-99: the E1 checkpoints of record (best VAL checkpoint sha256) at seeds 42, 43 and 44 -- an
#: independent guard against a seed swap (plan §1.1(3)).
E1_CHECKPOINT_RECORD = {
    42: "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03",
    43: "95f21204788f2986aba9311272c7f7e84f8d2049724c4bfdf5b0da66b6a78153",
    44: "ef95aefd8f383e3b23674e0c6abf040dfb1ef0da997a53140c59ced294108c8f",
}

#: IF§11.2's kind-to-cell map; `scored_run` and `derived` carry the value.
LABEL_NON_FINITE = "not evaluable: non-finite: no model (AM-21 item 2(b))"
LABEL_PARENT_CUT = "not run: parent cut"
LABEL_PARENT_DIVERGED = "not run: parent diverged"
MODEL_KINDS = ("scored_run", "derived")
#: The no-model kinds a planned cell may carry, by the kind of slot (IF§11.1-11.2).
RUN_NO_MODEL_KINDS = ("cut", "diverged", "aborted", "not_made")
PTQ_NO_MODEL_KINDS = ("cut", "aborted", "not_made", "not_run")
QAT_RUN_NO_MODEL_KINDS = ("cut", "diverged", "aborted", "not_made", "non_finite", "not_run")
TELEMETRY_UNITS = {"iter": "iteration", "epoch": "epoch"}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_FILE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Every LREP refusal code of B-1 to B-3, B-5 and B-5b, with the plan rule or ruling it implements. The CLI
#: exits 2 on any of them and writes nothing.
CODES = {
    # the statistics artifact (R-1)
    "stats_artifact_unverified": "R-1: the statistics artifact does not verify with its 37 inputs",
    # the export and the binding (R-2; IF§11.5-11.6)
    "stub_reader_official": "IF§11.6: the stub reader is refused under Policy.OFFICIAL",
    "inventory_unreadable": "R-2: the export cannot be read as strict JSON",
    "inventory_sha256_mismatch": "R-2: the export's sha256 is not the expected one",
    "inventory_invalid": "R-2: the export lacks, repeats or mistypes a field LREP reads",
    "binding_invalid": "R-2: the binding file is missing, differs from binding.sha256 or is malformed, "
                       "or a bound export carries no binding.artifacts",
    "binding_unbound": "IF§11.5: under Policy.OFFICIAL the binding must be bound",
    "freeze_mismatch": "IF§11.5: freeze.manifest_sha256 is not manifest_sha256 (or no freeze, OFFICIAL)",
    # planned cells (R-3; D-3 as LRB4 amends it)
    "planned_slot_missing": "R-3: a planned (stage, seed) has no family/per_seed slot",
    "planned_slot_duplicate": "R-3: two family/per_seed slots claim one planned (stage, seed)",
    "unplanned_slot": "R-3/LRB4: a family/per_seed slot outside the 21 planned cells",
    # status (R-4, R-5, R-12 to R-14; IF§11.2; LRB5)
    "status_invalid": "R-4: two entries for one planned slot, a kind outside the stage's kinds, or a "
                      "missing or malformed record field",
    "superseded": "R-5: kind superseded is refused, never mapped",
    "parent_rule": "R-12: a parentage or not-run inconsistency (AM-19 item 4(a); LRB5)",
    "seed42_not_complete": "R-13: a seed-42 run of record without a model",
    "seed_inconsistent": "R-14: a slot's model row or parent at another stage or seed, or its fq_off row "
                         "not the model of its int8_qat row (LRB5)",
    # artifacts (R-6 to R-9; LRB3, LRB7)
    "artifact_missing": "R-6: a model of record without its model row or bound artifact",
    "artifact_for_missing_run": "R-6: a bound clean artifact for a planned run without a model",
    "artifact_record_mismatch": "R-7/LRB3: an artifact differs from its binding record or model row",
    "not_canvas": "R-8: a binding record whose protocol is not canvas (an artifact whose protocol "
                  "differs from its record is R-7; LRB3)",
    "artifact_unreadable": "R-9: an artifact the driver refuses (files, manifest, the row rules)",
    "precision_mismatch": "LRB7; B5-S1: (model_role, precision) of an artifact or a model row is not the "
                          "one its stage maps to ((student, qat_fakequant_off) for an fq_off row)",
    "path_unsafe": "IF§2: a path that is not plain relative POSIX, or contains 'test'",
    # cross-artifact and seed-42 checks (R-10, R-11)
    "cross_artifact_mismatch": "R-10: manifest, class map, metric code, commit, AM-5, GT, run_id or "
                               "checkpoint disagreement",
    "e1_checkpoint_record": "R-10: an E1 checkpoint differs from DL-99's record for its seed",
    "seed42_mismatch": "R-11: a seed-42 value or identity differs from the statistics artifact's",
    # telemetry (LRB2) and robustness at seeds 43/44 (LRB11)
    "telemetry_sha256_mismatch": "LRB2: a telemetry file's sha256 differs from the export's",
    "telemetry_invalid": "LRB2: a telemetry item or file whose VAL rows cannot be read",
    "robustness_cells_missing": "LRB11: an E1/E6 pair with both models lacks one of its 15 cells",
    # strata tables (AM-17 item 8; plan §2.1)
    "strata_invalid": "§2.1: the strata file is unreadable or not one stratum per disease class",
    "strata_not_of_record": "§2.1: official profile, strata file not committed at its DL-63 sha256",
    "strata_gt_mismatch": "§2.1: the seven models' TEST gt_support differ",
    "strata_integrity": "§2.1: a model's per-class block does not reproduce its summary bitwise",
    # default candidates (AM-17 item 6; plan §2.2)
    "default_candidate_missing": "§2.2: a default candidate has no slot, model or artifact",
    "default_candidate_invalid": "§2.2: another candidate, or a default that is not complete",
    "selection_record_mismatch": "§2.2: a selected run's checkpoint is not DL-109's/DL-121's",
    # AM-6 tables (plan §3)
    "gt_present_unfaithful": "§3.3: R1 or R3 fails, or a seed-42 row differs from report.py's",
    # the extension report (plan §1.6, §2.4, D-13, D-14)
    "out_dir_invalid": "D-14: the out-dir exists, or lies inside the repository or an input",
    "parts_invalid": "§3.3: --parts names an unknown or repeated part",
    "mde_entry_invalid": "LRB18: the MDE entry cannot be read",
    "usage_invalid": "a bad command line (--repo-root without --synthetic-inputs, a missing input)",
    # B-5's input layer (B5-S3: rulings B5-S1 and B5-S2)
    "stage_unmapped": "B5-S1: a TMAN stage outside LREP's stage map (TMAN stage -> evaluator stage, "
                      "model_role, precision)",
    "dedup_invalid": "B5-S2: the export's dedup or the file it names is null, unreadable, not at its "
                     "sha256 or malformed, or its ids break the invariants against the scored artifacts",
    # the arm tables (AM-18 items 3(d) and 11; B5-S3)
    "arm_slot_invalid": "B5-S3: an arm or arm_teacher slot or row that the arm tables cannot place (two "
                        "slots for one (X, stage), a candidate at a seed other than 42, a row the export "
                        "has no slot for); every other arm or arm_teacher slot is skipped (LRB4; R4-5(b))",
    "arm_record_invalid": "B5-S3: a teacher or seed-noise record that is missing, unreadable or not at its "
                          "DL sha256, or whose √2·s differs from its band",
    # AM-23's rows (items 5-6; B5-S3)
    "quant_slot_invalid": "B5-S3: a quant_descriptive slot outside E6-E2, E7-E2, E6-R1 and E6-R2, or two "
                          "slots for one stage",
    # the descriptive TEST read (B-5b: ruling LREP-B5-RULE-4 R4-1)
    "descriptive_input_invalid": "R4-1 item 3: a descriptive_test slot's artifact fails a step of the DT "
                                 "read (a: its loader under NONOFFICIAL_SMOKE; b: provisional, split test, "
                                 "not random_init, 1561 rows; c: the official AM-5 rule; d: its family "
                                 "reference's identity and AM-5 set; e: its pair under NONOFFICIAL_SMOKE; f: "
                                 "its binding record)",
    "descriptive_input_in_official": "R4-1 item 5: an input that is not one of the 37 (a DT run among them) "
                                     "where an official computation takes its inputs",
}


class LrepRefusal(RuntimeError):
    """A named LREP refusal: the input is refused and nothing is written (CLI exit 2)."""

    def __init__(self, code: str, detail: str):
        if code not in CODES:
            raise ValueError(f"unknown LREP refusal code {code!r}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def refusal(code: str, detail: str) -> LrepRefusal:
    """The refusal to raise. Every site reads `raise refusal("<code>", ...)`."""
    return LrepRefusal(code, detail)


# --------------------------------------------------------------------------------------------------
# 1. small helpers
# --------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_text(v) -> bool:
    return isinstance(v, str) and bool(v) and v == v.strip()


def strict_json_text(text: str, where: str, code: str):
    """Order-preserving strict parse: a duplicate key or a non-finite constant is refused by `code`."""
    def pairs(items):
        keys = [k for k, _ in items]
        if len(set(keys)) != len(keys):
            raise LrepRefusal(code, f"{where}: duplicate JSON key in {keys}")
        return dict(items)

    def constant(name):
        raise LrepRefusal(code, f"{where}: non-finite JSON constant {name!r}")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except LrepRefusal:
        raise
    except (ValueError, RecursionError) as e:
        raise LrepRefusal(code, f"{where}: not valid JSON ({e})") from e


def path_problem(value) -> str | None:
    """Why `value` is not a plain relative POSIX path free of "test" (IF§2), or None."""
    if not _is_text(value):
        return "not a non-empty string without surrounding space"
    parts = value.split("/")
    if (value.startswith("/") or "\\" in value or ":" in value or "\x00" in value
            or any(p in ("", ".", "..") for p in parts)):
        return "not a plain relative POSIX path"
    if "test" in value.lower():
        return "contains 'test' (IF§2: no created path contains it, in any case)"
    return None


def safe_relative_path(value, where: str) -> str:
    problem = path_problem(value)
    if problem is not None:
        raise refusal("path_unsafe", f"{where}: path {value!r} is {problem}")
    return value


def cell_id(name: str, severity: int) -> str:
    """IF§6: a corrupted cell's id is `<corruption>_s<severity>`."""
    return f"{name}_s{severity}"


def corruption_names() -> tuple[str, ...]:
    """The five corruption identifiers in the frozen protocol order (configs/corruption_protocol.json
    of this code's repository)."""
    from .corruption_protocol import official_corruption_grid
    return tuple(official_corruption_grid().names)


def inferential_cells() -> tuple[tuple[str, str, int], ...]:
    """(cell id, corruption, severity) of the 15 inferential cells, in the frozen protocol order."""
    from .robustness import INFERENTIAL_SEVERITIES
    return tuple((cell_id(n, s), n, s) for n in corruption_names() for s in INFERENTIAL_SEVERITIES)


def stage_spec(stage) -> tuple[str, str, str]:
    """B5-S1: (evaluator stage, model_role, precision) of a TMAN stage; a stage outside the map is
    stage_unmapped."""
    spec = STAGE_MAP.get(stage) if isinstance(stage, str) else None
    if spec is None:
        raise refusal("stage_unmapped", f"stage {stage!r} is not in LREP's stage map (B5-S1), which holds "
                                        f"{list(STAGE_MAP)}")
    return spec


# --------------------------------------------------------------------------------------------------
# 2. the reader stub and the export (IF§8, IF§11.5-11.6)
# --------------------------------------------------------------------------------------------------
def _top_level_problem(doc) -> str | None:
    """The first defect of the export's top level that LREP depends on, or None."""
    if not isinstance(doc, dict):
        return "the export is not a JSON object"
    if doc.get("schema") != INVENTORY_SCHEMA:
        return f"schema {doc.get('schema')!r} is not {INVENTORY_SCHEMA!r}"
    if not isinstance(doc.get("evaluator_commit"), str) or not _HEX40.fullmatch(doc["evaluator_commit"]):
        return "evaluator_commit is not 40 lowercase hex"
    if not isinstance(doc.get("manifest_sha256"), str) or not _HEX64.fullmatch(doc["manifest_sha256"]):
        return "manifest_sha256 is not 64 lowercase hex"
    if not isinstance(doc.get("execution_context"), dict):
        return "execution_context is not an object"
    for key in ("models", "entries"):
        if not isinstance(doc.get(key), list) or not all(isinstance(x, dict) for x in doc[key]):
            return f"{key} is not a list of objects"
    b = doc.get("binding")
    if not isinstance(b, dict) or b.get("state") not in BINDING_STATES:
        return "binding is not an object with state 'bound' or 'unbound'"
    if b["state"] == "unbound" and (b.get("path") is not None or b.get("sha256") is not None):
        return "an unbound binding carries a path or a sha256"
    if b["state"] == "bound" and (not _is_text(b.get("path")) or not isinstance(b.get("sha256"), str)
                                  or not _HEX64.fullmatch(b["sha256"])):
        return "a bound binding needs a path and a 64-hex sha256"
    f = doc.get("freeze", None)
    if f is not None and (not isinstance(f, dict) or not isinstance(f.get("manifest_sha256"), str)
                          or not _HEX64.fullmatch(f["manifest_sha256"])):
        return "freeze is neither null nor an object with a 64-hex manifest_sha256"
    if "dedup" in doc and doc["dedup"] is not None and not isinstance(doc["dedup"], dict):
        return "dedup is neither null nor an object"
    for i, row in enumerate(doc["models"]):
        if not _is_text(row.get("slot")) or row.get("precision") not in PRECISIONS:
            return f"models[{i}] has no slot or a precision outside {PRECISIONS}"
    for i, e in enumerate(doc["entries"]):
        if not _is_text(e.get("slot")):
            return f"entries[{i}] has no slot"
        roles = e.get("role")
        if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
            return f"entries[{i}] ({e['slot']}): role is not a list of strings"
    return None


def read_inventory(path, *, expect_sha256=None) -> dict:
    """STUB of `src.tman.validate.read_inventory` (IF§11.6), for LREP's synthetic fixtures only.

    Reads and checks the export; once `binding.state` is "bound", verifies the binding file (resolved
    against the export's directory) against `binding.sha256` and returns its records under
    `binding.artifacts`. Refuses by code; never used under Policy.OFFICIAL (`open_inventory`)."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise refusal("inventory_unreadable", f"{p}: {type(e).__name__}: {e}") from e
    digest = hashlib.sha256(raw).hexdigest()
    if expect_sha256 is not None and digest != expect_sha256:
        raise refusal("inventory_sha256_mismatch", f"{p}: sha256 {digest}, expected {expect_sha256}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise refusal("inventory_unreadable", f"{p}: not UTF-8 ({e})") from e
    doc = strict_json_text(text, str(p), "inventory_unreadable")
    problem = _top_level_problem(doc)
    if problem is not None:
        raise refusal("inventory_invalid", f"{p}: {problem}")
    b = doc["binding"]
    if b["state"] == "bound":
        bp = p.parent / safe_relative_path(b["path"], f"{p}: binding.path")
        problem, bdoc = None, None
        try:
            braw = bp.read_bytes()
            if hashlib.sha256(braw).hexdigest() != b["sha256"]:
                problem = "its sha256 differs from binding.sha256"
            else:
                bdoc = strict_json_text(braw.decode("utf-8"), str(bp), "binding_invalid")
                if (not isinstance(bdoc, dict) or bdoc.get("schema") != BINDING_SCHEMA
                        or not isinstance(bdoc.get("artifacts"), list)):
                    problem = f"not a {BINDING_SCHEMA} object with an artifacts list"
        except (OSError, UnicodeDecodeError) as e:
            problem = f"{type(e).__name__}: {e}"
        if problem is not None:
            raise refusal("binding_invalid", f"binding {bp}: {problem}")
        doc["binding"] = dict(b, artifacts=bdoc["artifacts"])
    return doc


read_inventory.lrep_stub = True


@dataclass(frozen=True)
class Inventory:
    """The export as LREP reads it: lazily validated indexes over slots, model rows and records."""
    path: Path
    doc: dict
    policy: Policy
    reader: str
    entries: Mapping[str, tuple]                  # slot -> entries (more than one: refused on read)
    rows: Mapping[tuple, tuple]                   # (slot, precision) -> model rows
    records: Mapping[tuple, tuple]                # (slot, precision, scope, cell) -> binding records

    @property
    def evaluator_commit(self) -> str:
        return self.doc["evaluator_commit"]

    @property
    def bound(self) -> bool:
        return self.doc["binding"]["state"] == "bound"

    def has_entry(self, slot) -> bool:
        return isinstance(slot, str) and slot in self.entries

    def entry(self, slot: str) -> dict:
        items = self.entries.get(slot, ())
        if len(items) != 1:
            raise refusal("inventory_invalid", f"slot {slot!r} has {len(items)} entries; exactly one "
                                               "is required")
        return items[0]

    def has_row(self, slot: str, precision: str) -> bool:
        return bool(self.rows.get((slot, precision)))

    def has_record(self, slot: str, precision: str, scope: str, cell=None) -> bool:
        return bool(self.records.get((slot, precision, scope, cell)))


def _index(items, key) -> dict:
    out: dict = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return {k: tuple(v) for k, v in out.items()}


def open_inventory(path, *, policy: Policy, expect_sha256=None, reader=None) -> Inventory:
    """Read the export with `reader` (default: the stub) and apply IF§11.5's binding and freeze rules.

    Under Policy.OFFICIAL the stub is refused before anything is read, the binding must be bound and
    a freeze must be recorded."""
    reader = read_inventory if reader is None else reader
    stub = bool(getattr(reader, "lrep_stub", False))
    if stub and policy is Policy.OFFICIAL:
        raise refusal("stub_reader_official", "the LREP stub reader serves synthetic fixtures only; "
                                              "Policy.OFFICIAL needs TMAN's reader (IF§11.6, B-4)")
    try:
        doc = reader(path, expect_sha256=expect_sha256)
    except LrepRefusal:
        raise
    except Exception as e:                                          # noqa: BLE001 -- TMAN's refusal
        raise refusal("inventory_invalid", f"the reader refused {path}: {type(e).__name__}: "
                                           f"{e}") from e
    problem = _top_level_problem(doc)
    if problem is not None:
        raise refusal("inventory_invalid", f"{path}: {problem}")
    if policy is Policy.OFFICIAL and doc["binding"]["state"] != "bound":
        raise refusal("binding_unbound", f"{path}: binding.state is {doc['binding']['state']!r}; "
                                         "Policy.OFFICIAL reads a bound export only (IF§11.5)")
    freeze = doc.get("freeze")
    if ((freeze is None and policy is Policy.OFFICIAL)
            or (freeze is not None and freeze["manifest_sha256"] != doc["manifest_sha256"])):
        raise refusal("freeze_mismatch", f"{path}: freeze {freeze!r} does not record manifest_sha256 "
                                         f"{doc['manifest_sha256']} (IF§11.5)")
    if doc["binding"]["state"] == "bound" and "artifacts" not in doc["binding"]:
        raise refusal("binding_invalid", f"{path}: a bound export carries no binding.artifacts; the reader "
                                         "returns the binding's records under it (IF§11.6)")
    records = doc["binding"].get("artifacts", []) if doc["binding"]["state"] == "bound" else []
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise refusal("binding_invalid", f"{path}: binding.artifacts is not a list of objects")
    for i, r in enumerate(records):
        if (not _is_text(r.get("slot")) or r.get("precision") not in PRECISIONS
                or r.get("scope") not in SCOPES or not (r.get("cell") is None or _is_text(r["cell"]))):
            raise refusal("binding_invalid", f"{path}: binding.artifacts[{i}] lacks a slot, precision, "
                                             "scope or cell of the vocabulary")
    name = ("lrep_stub" if stub else
            f"{getattr(reader, '__module__', '?')}.{getattr(reader, '__qualname__', repr(reader))}")
    return Inventory(
        path=Path(path), doc=doc, policy=policy, reader=name,
        entries=_index(doc["entries"], lambda e: e["slot"]),
        rows=_index(doc["models"], lambda r: (r["slot"], r["precision"])),
        records=_index(records, lambda r: (r["slot"], r["precision"], r["scope"], r.get("cell"))))


# --------------------------------------------------------------------------------------------------
# 3. planned cells: slots, status, parentage (R-3 to R-6, R-12 to R-14)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotStatus:
    """A planned cell's run: whether it has a model, else its cell label and the deciding record."""
    slot: str
    stage: str
    seed: int
    kind: str                     # the kind that decides the cell, after IF§11.2's resolution
    has_model: bool
    label: str | None             # None when the run has a model
    source_slot: str              # the slot whose entry decided: itself, its QAT run or its parent
    record: Mapping               # that entry's kind fields

    def cell_text(self) -> str:
        """The cell as printed: the stage, then its label (D-15)."""
        return f"{self.stage}: {self.label}"


def record_slots(inv: Inventory) -> dict[tuple[str, int], str]:
    """{(stage, seed): slot} of the 21 planned cells: the family (seed 42) and per_seed (43/44) slots
    (IF§11.1). Every other slot is ignored and never refused (LRB4)."""
    found: dict[tuple, list] = {}
    for slot, items in inv.entries.items():
        roles = sorted({r for e in items for r in e["role"] if r in RECORD_ROLES})
        if not roles:
            continue
        if len(items) != 1:
            raise refusal("status_invalid", f"slot {slot!r} has {len(items)} entries: a planned run has one "
                                            "status (R-4)")
        e = items[0]
        stage, seed = e.get("stage"), e.get("seed")
        planned = (len(roles) == 1 and stage in STAGES and _is_int(seed)
                   and seed in RECORD_ROLES[roles[0]])
        if not planned:
            raise refusal("unplanned_slot", f"slot {slot!r} (stage {stage!r}, seed {seed!r}, role "
                                            f"{roles}) is not one of the 21 planned cells (D-3; LRB4)")
        found.setdefault((stage, seed), []).append(slot)
    repeated = {k: v for k, v in found.items() if len(v) > 1}
    if repeated:
        raise refusal("planned_slot_duplicate", f"planned cells with more than one slot: {repeated}")
    missing = [(st, s) for st in STAGES for s in SEEDS if (st, s) not in found]
    if missing:
        raise refusal("planned_slot_missing", f"planned cells without a family/per_seed slot: "
                                              f"{missing}")
    return {k: v[0] for k, v in found.items()}


def _kind_fields_problem(e: dict) -> str | None:
    """IF§11.2: the fields each no-model kind's label needs."""
    kind = e.get("kind")
    if kind == "cut":
        d = e.get("cut_date")
        try:
            ok_date = (isinstance(d, str) and bool(_DATE.fullmatch(d))
                       and _dt.date.fromisoformat(d).isoformat() == d)
        except ValueError:
            ok_date = False
        if not ok_date:
            return f"cut_date {d!r} is not YYYY-MM-DD"
        if not _is_text(e.get("rule")):
            return "a cut needs its rule"
        if not (e.get("comparator", None) is None or _is_text(e["comparator"])):
            return "comparator is neither null nor a slot"
    elif kind == "diverged":
        # TMAN's shape (src/tman/schema.py:398 and :413-414 at ee5593e; R4-5(f)): both keys present, exactly
        # one of them non-null, with the kind's other fields
        if "iteration" not in e or "epoch" not in e:
            return "a divergence carries both iteration and epoch, the unused one null"
        units = [u for u in ("iteration", "epoch") if e[u] is not None]
        if len(units) != 1 or not _is_int(e[units[0]]) or e[units[0]] < 0:
            return "a divergence needs exactly one of iteration or epoch non-null, a non-negative integer"
        if not all(_is_text(e.get(k)) for k in ("run_id", "rule", "cause")):
            return "a divergence needs its run_id, rule and cause"
        if not (isinstance(e.get("telemetry_sha256"), str) and _HEX64.fullmatch(e["telemetry_sha256"])):
            return "a divergence needs its telemetry_sha256 (64 lowercase hex)"
    elif kind == "aborted":
        if not (_is_text(e.get("rule")) and _is_text(e.get("cause"))):
            return "an abort needs its rule and cause"
    elif kind in ("not_run", "not_made"):
        if not _is_text(e.get("reason")):
            return f"{kind} needs its reason"
    return None


def _label(e: dict) -> str:
    kind = e["kind"]
    if kind == "cut":
        return f"cut ({e['cut_date']}, {e['rule']})"
    if kind == "diverged":
        unit = "iteration" if e["iteration"] is not None else "epoch"
        return f"diverged ({unit} {e[unit]}, {e['rule']})"
    if kind == "aborted":
        return f"aborted ({e['rule']}, {e['cause']})"
    if kind == "not_made":
        return f"not made: {e['reason']}"
    if kind == "non_finite":
        return LABEL_NON_FINITE
    raise ValueError(f"no label for kind {kind!r}")              # pragma: no cover - guard


def _record_of(e: dict) -> dict:
    keep = ("kind", "cut_date", "rule", "comparator", "iteration", "epoch", "cause", "reason",
            "dl_rows")
    return {k: e[k] for k in keep if k in e}


def entry_label(e: dict) -> str | None:
    """The IF§11.2 cell text of a slot a table skips, from its own entry and never a refusal (LRB4; R4-5(b)):
    None for a kind that carries a model or names none, "not run: <reason>" for not_run, the kind's label
    when its record fields are well formed, else the kind as written. Nothing else is read."""
    kind = e.get("kind")
    if kind in (*SCORED_ENTRY_KINDS, "listed_run", "measurement"):
        return None
    if kind == "not_run":
        return f"not run: {e.get('reason')}"
    if kind in ("cut", "diverged", "aborted", "not_made", "non_finite") and _kind_fields_problem(e) is None:
        return _label(e)
    return f"kind {kind!r}"


def _checked_kind(inv: Inventory, slot: str) -> tuple[dict, str | None]:
    """(entry, kind) with R-5 and the kind vocabulary and record fields checked (R-4)."""
    e = inv.entry(slot)
    kind = e.get("kind")
    if kind == "superseded":
        raise refusal("superseded", f"slot {slot!r} is superseded; AM-19 item 7 can no longer fire "
                                    "(R-5), so the kind is refused, never mapped")
    if kind is not None and kind not in KINDS:
        raise refusal("status_invalid", f"slot {slot!r}: kind {kind!r} is outside the vocabulary")
    problem = _kind_fields_problem(e)
    if problem is not None:
        raise refusal("status_invalid", f"slot {slot!r} ({kind}): {problem}")
    return e, kind


def _parent_of(inv: Inventory, e: dict, slot: str) -> str:
    cat = e.get("catalogue")
    parent = cat.get("parent") if isinstance(cat, dict) else None
    if not inv.has_entry(parent):
        raise refusal("parent_rule", f"slot {slot!r}: catalogue.parent {parent!r} is not a slot of the "
                                     "export")
    return parent


def parent_slot(inv: Inventory, slot: str) -> str:
    """A slot's catalogue.parent, which must be a slot of the export (R-12)."""
    return _parent_of(inv, inv.entry(slot), slot)


def _not_run_status(inv: Inventory, slot: str, e: dict, seed: int, depth: int = 0
                    ) -> tuple[str, str, dict]:
    """(label, deciding slot, its record) of a not_run entry: its parent chain ends in a cut or a
    divergence at the same seed (AM-19 item 4(a); R-12)."""
    parent = _parent_of(inv, e, slot)
    pe, pk = _checked_kind(inv, parent)
    if pe.get("seed") != seed:
        raise refusal("seed_inconsistent", f"slot {slot!r} (seed {seed}) has parent {parent!r} at seed "
                                           f"{pe.get('seed')!r}")
    if pk in ("cut", "diverged"):
        return (LABEL_PARENT_CUT if pk == "cut" else LABEL_PARENT_DIVERGED), parent, _record_of(pe)
    if pk == "not_run" and depth < 2:
        return _not_run_status(inv, parent, pe, seed, depth + 1)
    raise refusal("parent_rule", f"slot {slot!r} is not run, but its parent {parent!r} is {pk!r}, "
                                 "neither cut nor diverged (AM-19 item 4(a))")


def _check_unit(slot: str, e: dict, unit: str) -> None:
    """IF§11.2: a KD or E1 run diverges at an iteration, a QAT run at an epoch (R-4); in TMAN's shape the
    other unit is null (R4-5(f))."""
    if e.get("kind") == "diverged" and e.get(unit) is None:
        raise refusal("status_invalid", f"slot {slot!r}: this run's divergence is recorded by {unit} "
                                        "(IF§11.2: iteration for a KD or E1 run, epoch for a QAT run)")


def resolve_status(inv: Inventory, slot: str, stage: str, seed: int) -> SlotStatus:
    """IF§11.2's map for one planned cell's slot (LRB5)."""
    e, kind = _checked_kind(inv, slot)

    def no_model(label, source, record, decided_kind):
        return SlotStatus(slot, stage, seed, decided_kind, False, label, source, record)

    if stage in RUN_STAGES:
        if kind == "scored_run":
            return SlotStatus(slot, stage, seed, kind, True, None, slot, _record_of(e))
        if kind in RUN_NO_MODEL_KINDS:
            _check_unit(slot, e, "iteration")
            return no_model(_label(e), slot, _record_of(e), kind)
        raise refusal("status_invalid", f"slot {slot!r}: kind {kind!r} is not a status of an {stage} "
                                        f"run (scored_run or {RUN_NO_MODEL_KINDS})")
    if kind == "derived":
        return SlotStatus(slot, stage, seed, kind, True, None, slot, _record_of(e))
    if stage in QAT_STAGES:                    # the selection reads its QAT run's kind (IF§11.2)
        run_slot = _parent_of(inv, e, slot)
        run = inv.entry(run_slot)
        if run.get("stage") != stage or run.get("seed") != seed:
            raise refusal("parent_rule", f"selection {slot!r}: its parent {run_slot!r} is not the {stage} "
                                         f"QAT run at seed {seed} (LRB5)")
        re_, rk = _checked_kind(inv, run_slot)
        if rk == "not_run":
            label, src, rec = _not_run_status(inv, run_slot, re_, seed)
            return no_model(label, src, rec, rk)
        if rk in QAT_RUN_NO_MODEL_KINDS:
            _check_unit(run_slot, re_, "epoch")
            return no_model(_label(re_), run_slot, _record_of(re_), rk)
        raise refusal("status_invalid", f"selection {slot!r} has no model (kind {kind!r}) but its QAT "
                                        f"run {run_slot!r} is {rk!r}")
    if kind is None:                           # a PTQ model with kind null: its parent's state
        parent = _parent_of(inv, e, slot)
        pe = inv.entry(parent)
        if pe.get("stage") != PARENT_STAGE[stage]:
            raise refusal("parent_rule", f"PTQ slot {slot!r}: its parent {parent!r} is not an "
                                         f"{PARENT_STAGE[stage]} run (LRB5)")
        if pe.get("seed") != seed:
            raise refusal("seed_inconsistent", f"PTQ slot {slot!r} (seed {seed}) has parent {parent!r} at "
                                               f"seed {pe.get('seed')!r}")
        pe, pk = _checked_kind(inv, parent)
        if pk in ("cut", "diverged"):
            return no_model(LABEL_PARENT_CUT if pk == "cut" else LABEL_PARENT_DIVERGED, parent,
                            _record_of(pe), "not_run")
        raise refusal("status_invalid", f"PTQ slot {slot!r} has kind null and its parent {parent!r} is "
                                        f"{pk!r}: any other null case is refused (IF§11.2; R-4)")
    if kind == "not_run":
        label, src, rec = _not_run_status(inv, slot, e, seed)
        return no_model(label, src, rec, kind)
    if kind in PTQ_NO_MODEL_KINDS:
        return no_model(_label(e), slot, _record_of(e), kind)
    raise refusal("status_invalid", f"slot {slot!r}: kind {kind!r} is not a status of an {stage} "
                                    f"model (derived, null or {PTQ_NO_MODEL_KINDS})")


@dataclass(frozen=True)
class PlannedCells:
    slots: Mapping[tuple[str, int], str]               # (stage, seed) -> slot
    status: Mapping[tuple[str, int], SlotStatus]

    def has_model(self, stage: str, seed: int) -> bool:
        return self.status[(stage, seed)].has_model


def check_parentage(inv: Inventory, slots, status) -> None:
    """R-12 (LRB5): a PTQ model's parent is its E1/E3 run of record at that seed; a QAT selection's
    parent is its QAT run, whose parent is that run; a model never descends from a run without one; and
    the descendants of a cut or diverged run are not run (AM-19 item 4(a))."""
    for (stage, seed), slot in sorted(slots.items()):
        if stage not in PARENT_STAGE:
            continue
        want = slots[(PARENT_STAGE[stage], seed)]
        e = inv.entry(slot)
        first = _parent_of(inv, e, slot)
        if stage in QAT_STAGES:
            run = inv.entry(first)
            if run.get("stage") != stage or run.get("seed") != seed:
                raise refusal("parent_rule", f"selection {slot!r}: its parent {first!r} is not the "
                                             f"{stage} QAT run at seed {seed}")
            got = _parent_of(inv, run, first)
        else:
            got = first
        if got != want:
            raise refusal("parent_rule", f"{stage} seed {seed} ({slot!r}) descends from {got!r}, not "
                                         f"the {PARENT_STAGE[stage]} run of record {want!r}")
        parent, own = status[(PARENT_STAGE[stage], seed)], status[(stage, seed)]
        if own.has_model and not parent.has_model:
            raise refusal("parent_rule", f"{stage} seed {seed} has a model, but its parent "
                                         f"{parent.stage} has none ({parent.label})")
        if not parent.has_model and parent.kind in ("cut", "diverged") and own.kind != "not_run":
            raise refusal("parent_rule", f"{stage} seed {seed} ({slot!r}) reads {own.label!r}, but its "
                                         f"parent {parent.stage} is {parent.kind}: it is not run (AM-19 "
                                         "item 4(a))")


def model_row(inv: Inventory, slot: str, precision: str) -> dict:
    """The one model row of (slot, precision), with the IF§8/IF§11.1 fields LREP reads checked: its
    entry_kind is a scored kind and its slot's entry kind, and its role is its slot's role set."""
    items = inv.rows.get((slot, precision), ())
    if not items:
        raise refusal("artifact_missing", f"slot {slot!r} has no model row of precision {precision}")
    if len(items) > 1:
        raise refusal("inventory_invalid", f"slot {slot!r}: {len(items)} model rows of precision "
                                           f"{precision}; rows are unique by (slot, precision)")
    row = items[0]
    e = inv.entry(slot)
    src = row.get("source")
    conv = row.get("converted_artifact_sha256")
    problems = [what for ok, what in (
        (isinstance(row.get("file_id"), str) and bool(_FILE_ID.fullmatch(row["file_id"]))
         and "__" not in row["file_id"] and "test" not in row["file_id"].lower(), "file_id (IF§2)"),
        (_is_text(row.get("stage")), "stage"),
        (row.get("model_role") in ("teacher", "student"), "model_role"),
        (row.get("seed") is None or _is_int(row.get("seed")), "seed"),
        (isinstance(src, dict) and src.get("kind") in ("checkpoint", "provenance")
         and isinstance(src.get("sha256"), str) and bool(_HEX64.fullmatch(src["sha256"])), "source"),
        ((isinstance(conv, str) and bool(_HEX64.fullmatch(conv))) if precision in INT8_PRECISIONS
         else conv is None, "converted_artifact_sha256 (INT8 rows only)"),
        (isinstance(row.get("cells"), list), "cells"),
        (row.get("entry_kind") in SCORED_ENTRY_KINDS and row.get("entry_kind") == e.get("kind"),
         "entry_kind (IF§11.1: a scored kind, the slot's entry kind)"),
        (isinstance(row.get("role"), list) and set(row["role"]) <= set(ROLES)
         and set(row["role"]) == set(e.get("role") or ()), "role (IF§11.1: the slot's role set)"),
        (_is_text(row.get("training_run_id")), "training_run_id")) if not ok]
    if problems:
        raise refusal("inventory_invalid", f"model row ({slot!r}, {precision}): malformed {problems}")
    return row


def row_identity(row: dict) -> str:
    """IF§8 (TL-6): an INT8 artifact's run.checkpoint_sha256 is its converted artifact; every other
    row's is source.sha256."""
    if row["precision"] in INT8_PRECISIONS:
        return row["converted_artifact_sha256"]
    return row["source"]["sha256"]


def check_rows(inv: Inventory, slots, status) -> None:
    """R-14: each model of record's row of its stage precision carries the slot's stage and seed; and
    it is a student's row (LRB7)."""
    for (stage, seed), slot in sorted(slots.items()):
        if not status[(stage, seed)].has_model:
            continue
        row = model_row(inv, slot, D.STAGE_PRECISION[stage])
        if row["stage"] != stage or row["seed"] != seed:
            raise refusal("seed_inconsistent", f"slot {slot!r} ({stage}, seed {seed}): its model row "
                                               f"is ({row['stage']}, seed {row['seed']})")
        if row["model_role"] != "student":
            raise refusal("precision_mismatch", f"slot {slot!r} ({stage}, seed {seed}): its model row's "
                                                f"model_role is {row['model_role']!r}, not student (LRB7)")


def check_missing_runs(inv: Inventory, slots, status) -> None:
    """R-6, second half: a planned run without a model has no bound clean artifact (AM-19 3(d))."""
    for (stage, seed), slot in sorted(slots.items()):
        if not status[(stage, seed)].has_model and inv.has_record(
                slot, D.STAGE_PRECISION[stage], "clean", None):
            raise refusal("artifact_for_missing_run", f"{stage} seed {seed} ({slot!r}) reads "
                                                      f"{status[(stage, seed)].label!r} but the "
                                                      "binding lists a clean artifact for it")


def check_seed42(status) -> None:
    """R-13: the Holm family's runs are never cut, so every seed-42 run of record has a model."""
    bad = [(st, s.label) for (st, seed), s in sorted(status.items()) if seed == 42 and not s.has_model]
    if bad:
        raise refusal("seed42_not_complete", f"seed-42 runs of record without a model: {bad}")


def planned_cells(inv: Inventory) -> PlannedCells:
    """The 21 planned cells with their status, every R-3 to R-6 and R-12 to R-14 check applied."""
    slots = record_slots(inv)
    status = {key: resolve_status(inv, slot, *key) for key, slot in sorted(slots.items())}
    check_seed42(status)
    check_parentage(inv, slots, status)
    check_rows(inv, slots, status)
    check_missing_runs(inv, slots, status)
    return PlannedCells(slots=slots, status=status)


# --------------------------------------------------------------------------------------------------
# 4. bound artifacts (R-6 to R-9; LRB3, LRB7)
# --------------------------------------------------------------------------------------------------
def binding_record(inv: Inventory, slot: str, precision: str, scope: str, cell=None) -> dict:
    """The one binding record of (slot, precision, scope, cell), its IF§11.4 fields checked."""
    items = inv.records.get((slot, precision, scope, cell), ())
    if not items:
        raise refusal("artifact_missing", f"the binding has no {scope} artifact for ({slot!r}, "
                                          f"{precision}{', ' + cell if cell else ''})")
    if len(items) > 1:
        raise refusal("binding_invalid", f"({slot!r}, {precision}, {scope}, {cell}): {len(items)} "
                                         "records; one per (row, scope, cell) is required")
    rec = items[0]
    files = rec.get("files_sha256")
    problems = [what for ok, what in (
        (rec.get("protocol") in PROTOCOLS, "protocol"),
        (isinstance(rec.get("repo_relative_path"), str), "repo_relative_path"),
        (isinstance(files, dict) and sorted(files) == list(ARTIFACT_FILE_NAMES)
         and all(isinstance(v, str) and _HEX64.fullmatch(v) for v in files.values()), "files_sha256"),
        (isinstance(rec.get("checkpoint_sha256"), str)
         and bool(_HEX64.fullmatch(rec["checkpoint_sha256"])), "checkpoint_sha256"),
        (isinstance(rec.get("repo_commit"), str) and bool(_HEX40.fullmatch(rec["repo_commit"])),
         "repo_commit"),
        (isinstance(rec.get("split_manifest_sha256"), str)
         and bool(_HEX64.fullmatch(rec["split_manifest_sha256"])), "split_manifest_sha256")) if not ok]
    if problems:
        raise refusal("binding_invalid", f"record ({slot!r}, {precision}, {scope}, {cell}): malformed "
                                         f"{problems}")
    return rec


def cell_condition(cell: str | None) -> tuple:
    """The driver's condition tuple of a binding cell: clean for null, else `<corruption>_s<sev>`."""
    if cell is None:
        return D.CLEAN
    name, sep, sev = cell.rpartition("_s")
    if not sep or not name or not sev.isdigit():
        raise refusal("binding_invalid", f"cell {cell!r} is not <corruption>_s<severity> (IF§6)")
    return ("corruption", name, int(sev))


@dataclass(frozen=True)
class BoundArtifact:
    """One scored artifact outside the 37, read and checked against its record and model row; `read` is its
    slot's column (R4-1: load_slot sets it, read_descriptive_test reads descriptive_test)."""
    slot: str
    stage: str
    precision: str
    scope: str
    cell: str | None
    record: Mapping
    row: Mapping
    loaded: D.LoadedInput
    read: str = READ_OFFICIAL

    @property
    def run(self):
        return self.loaded.run

    @property
    def read_label(self) -> str | None:
        """R4-1 item 6: DT_LABEL for a descriptive_test read, else None."""
        return read_label(self.read)

    @property
    def summary(self) -> dict:
        return self.loaded.summary

    @property
    def label(self) -> str:
        return f"{self.slot} {self.precision} {self.scope}{' ' + self.cell if self.cell else ''}"


def load_bound(inv: Inventory, *, slot: str, stage: str, precision: str, scope: str, cell=None,
               input_root: Path, policy: Policy) -> BoundArtifact:
    """Read one artifact the binding lists, refusing by code on any defect (R-6 to R-9, LRB3, LRB7).

    Order: the model row and the record; the record's path and protocol; the four files against the
    record; the manifest and canvas; the stage's evaluator stage, model_role and precision (B5-S1);
    `driver.load_input` under `policy` at that evaluator stage; (model_role, precision); then the
    record's checkpoint, commit and split manifest, and the row's identity."""
    row = model_row(inv, slot, precision)
    rec = binding_record(inv, slot, precision, scope, cell)
    label = f"{stage} ({slot!r}, {precision}, {scope}{', ' + cell if cell else ''})"
    rel = safe_relative_path(rec["repo_relative_path"], label)
    if rec["protocol"] != "canvas":
        raise refusal("not_canvas", f"{label}: its record's protocol is {rec['protocol']!r}; LREP's "
                                    "tables read canvas artifacts only (EVAL section 11(a); ERR E-24)")
    condition = cell_condition(cell)
    directory = Path(input_root) / rel
    try:
        files = D.input_file_sha256(directory)
        summary = verify_run_manifest(directory)
    except (D.DriverRefusal, IngestError, OSError) as e:
        raise refusal("artifact_unreadable", f"{label}: {type(e).__name__}: {e}") from e
    if files != dict(rec["files_sha256"]):
        bad = sorted(n for n in files if files[n] != rec["files_sha256"].get(n))
        raise refusal("artifact_record_mismatch", f"{label}: files_sha256 differ from the binding "
                                                  f"record for {bad} (R-7)")
    pp = (summary.get("dataset") or {}).get("preprocess_protocol")
    art_protocol = "upstream" if ("protocol" in summary or pp != CANVAS_PROTOCOL_ID) else "canvas"
    if art_protocol != rec["protocol"]:
        raise refusal("artifact_record_mismatch", f"{label}: the artifact's protocol is "
                                                  f"{art_protocol!r} (preprocess_protocol {pp!r}), its "
                                                  f"record says {rec['protocol']!r} (R-7; LRB3)")
    ev_stage, want_role, want_precision = stage_spec(stage)
    if precision == PRECISION_FQOFF:
        want_role, want_precision = "student", PRECISION_FQOFF
    if row["model_role"] != want_role:
        raise refusal("precision_mismatch", f"{label}: its model row's model_role is {row['model_role']!r}, "
                                            f"not {want_role!r} (B5-S1)")
    try:
        loaded = D.load_input(ev_stage, condition, rel, Path(input_root), policy)
    except D.DriverRefusal as e:
        raise refusal("artifact_unreadable", f"{label}: {e}") from e
    run, ds = loaded.summary["run"], loaded.summary["dataset"]
    if (run["model_role"], run["precision"]) != (want_role, want_precision) or precision != want_precision:
        raise refusal("precision_mismatch", f"{label}: (model_role, precision) is "
                                            f"({run['model_role']}, {run['precision']}), not ({want_role}, "
                                            f"{want_precision}) (LRB7; B5-S1)")
    got = {"checkpoint_sha256": run.get("checkpoint_sha256"), "repo_commit": run.get("repo_commit"),
           "split_manifest_sha256": ds.get("split_manifest_sha256")}
    diff = [k for k, v in got.items() if v != rec[k]]
    if dict(loaded.file_sha256) != files:
        diff.append("files changed while read")
    if got["checkpoint_sha256"] != row_identity(row):
        diff.append("checkpoint vs the model row's identity (IF§8)")
    if diff:
        raise refusal("artifact_record_mismatch", f"{label}: {diff} differ between the artifact, its "
                                                  "binding record and its model row (R-7; LRB3)")
    return BoundArtifact(slot=slot, stage=stage, precision=precision, scope=scope, cell=cell,
                         record=rec, row=row, loaded=loaded)


def load_clean_of_record(inv: Inventory, cells: PlannedCells, stage: str, seed: int, *,
                         input_root: Path, policy: Policy) -> BoundArtifact:
    """The clean canvas artifact of a planned run with a model, at its stage precision (R-6)."""
    slot = cells.slots[(stage, seed)]
    return load_bound(inv, slot=slot, stage=stage, precision=D.STAGE_PRECISION[stage], scope="clean",
                      input_root=input_root, policy=policy)


def check_row_at_entry(inv: Inventory, ba: BoundArtifact) -> None:
    """R-14 for a slot outside the 21 planned cells (am6's coverage rows, the arms' and AM-23's candidates,
    the fq_off table's E6-KD and AM-23 rows): the model row read is at its entry's stage and seed
    (check_rows applies the same rule to the planned cells)."""
    e = inv.entry(ba.slot)
    if ba.row["stage"] != e.get("stage") or ba.row["seed"] != e.get("seed"):
        raise refusal("seed_inconsistent", f"{ba.slot!r}: its {ba.precision} model row is "
                                           f"({ba.row['stage']}, seed {ba.row['seed']}), its entry "
                                           f"({e.get('stage')}, seed {e.get('seed')}) (R-14)")


# --------------------------------------------------------------------------------------------------
# 5. cross-artifact checks (R-10)
# --------------------------------------------------------------------------------------------------
def metrics_sha256() -> str:
    """sha256 of this checkout's src/eval/metrics.py (as artifact.official_conditions reads it)."""
    from .artifact import CODE_REPO, METRICS_REL
    return sha256_file(CODE_REPO / METRICS_REL)


def cross_check(clean: Mapping[str, object], others: Mapping[str, object], *, evaluator_commit: str,
                official: bool, checkpoint_groups: Mapping[str, str] | None = None) -> None:
    """R-10 over one table's artifacts. `clean` and `others` map a label to a driver.LoadedInput or a
    BoundArtifact (clean: the clean artifacts of the stage precision; others: corrupted cells and
    other precisions). One split manifest, class map and metric implementation (this checkout's in
    the official profile); every repo_commit is the export's evaluator_commit; one AM-5 set; one
    dataset_gt across the clean artifacts; distinct summary run_ids; distinct checkpoints across the
    clean artifacts of different `checkpoint_groups` (label -> slot)."""
    def loaded_of(x):
        return x.loaded if isinstance(x, BoundArtifact) else x
    every = {**{k: loaded_of(v) for k, v in clean.items()}, **{k: loaded_of(v) for k, v in others.items()}}
    problems = []
    for block, key in (("dataset", "split_manifest_sha256"), ("dataset", "class_map_sha256"),
                       ("run", "metric_impl_sha256")):
        vals = {li.summary[block][key] for li in every.values()}
        if len(vals) != 1:
            problems.append(f"{len(vals)} distinct {key}")
    if official and {li.summary["run"]["metric_impl_sha256"] for li in every.values()} != {
            metrics_sha256()}:
        problems.append("metric_impl_sha256 is not this checkout's src/eval/metrics.py")
    commits = sorted({li.summary["run"]["repo_commit"] for li in every.values()} - {evaluator_commit})
    if commits:
        problems.append(f"repo_commit {commits[:2]} is not the export's evaluator_commit")
    try:
        D.am5_identity({k: li for k, li in every.items()})
    except D.DriverRefusal as e:
        problems.append(str(e))
    gts = [np.asarray(loaded_of(v).run.stats.dataset_gt) for v in clean.values()]
    if any(not np.array_equal(g, gts[0]) for g in gts[1:]):
        problems.append("the clean artifacts' dataset_gt differ: one ground truth scores every model")
    run_ids = [li.summary["run"]["run_id"] for li in every.values()]
    if len(set(run_ids)) != len(run_ids):
        problems.append(f"summary run_ids repeat: "
                        f"{sorted({r for r in run_ids if run_ids.count(r) > 1})[:3]} (LRB5)")
    groups = checkpoint_groups or {k: k for k in clean}
    by_ckpt: dict = {}
    for k, v in clean.items():
        by_ckpt.setdefault(loaded_of(v).summary["run"].get("checkpoint_sha256"), set()).add(groups[k])
    shared = [c[:12] for c, g in by_ckpt.items() if len(g) > 1]
    if shared:
        problems.append(f"checkpoint(s) {shared} shared by different runs of record")
    if problems:
        raise refusal("cross_artifact_mismatch", "; ".join(problems))


def check_e1_record(e1_checkpoints: Mapping[int, str]) -> None:
    """R-10's guard against a seed swap: each E1 clean checkpoint is DL-99's for its seed."""
    bad = {s: c for s, c in sorted(e1_checkpoints.items()) if c != E1_CHECKPOINT_RECORD.get(s)}
    if bad:
        detail = ", ".join(f"seed {s}: {str(c)[:12]}... (DL-99: "
                           f"{E1_CHECKPOINT_RECORD.get(s, 'no record')[:12]}...)" for s, c in bad.items())
        raise refusal("e1_checkpoint_record", f"E1 checkpoint(s) differ from DL-99's record: {detail}")


# --------------------------------------------------------------------------------------------------
# 6. the statistics artifact (R-1) and the seed-42 identity of the 37 (R-11)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StatsArtifact:
    path: Path
    family: dict
    views: object                    # artifact.InputViews: the 37 inputs as the verifier loaded them
    policy: Policy
    profile: str                     # the verified artifact_status

    def clean(self, stage: str):
        return self.views.loaded[(stage, D.CLEAN)]

    def cell(self, stage: str, name: str, severity: int):
        return self.views.loaded[(stage, ("corruption", name, severity))]


def load_stats_artifact(path, input_root) -> StatsArtifact:
    """R-1: all thirteen checks re-established with the 37 inputs and the contract bytes, as
    report.build_report requires (report.py:327-333)."""
    from . import artifact as A
    problem, rep = None, None
    try:
        rep = A.verify_statistics_artifact(path, input_root=input_root,
                                           contract_bytes=A.contract_sha256()[1])
        if rep.not_reestablished or rep.views is None:
            problem = f"checks {rep.not_reestablished} were not re-established"
    except (A.ArtifactError, OSError) as e:
        problem = f"{type(e).__name__}: {e}"
    if problem is not None:
        raise refusal("stats_artifact_unverified", f"{path}: {problem}")
    return StatsArtifact(path=Path(path), family=rep.family, views=rep.views,
                         policy=rep.views.policy, profile=rep.family["artifact_status"])


def check_seed42_identity(inv: Inventory, cells: PlannedCells, stats: StatsArtifact,
                          stages=STAGES) -> None:
    """R-11 (K6, an identity and pairing check): each of the 37's clean artifacts of `stages` carries its
    family slot's model identity, and any binding record of a family slot's clean artifact equals its
    files."""
    bad = []
    for stage in stages:
        slot = cells.slots[(stage, 42)]
        li = stats.clean(stage)
        row = model_row(inv, slot, D.STAGE_PRECISION[stage])
        if li.summary["run"].get("checkpoint_sha256") != row_identity(row):
            bad.append(f"{stage}: the 37's checkpoint is not {slot!r}'s model identity")
        if inv.has_record(slot, D.STAGE_PRECISION[stage], "clean", None):
            rec = binding_record(inv, slot, D.STAGE_PRECISION[stage], "clean", None)
            if dict(rec["files_sha256"]) != dict(li.file_sha256):
                bad.append(f"{stage}: the binding record of {slot!r} names other files than the 37's")
    if bad:
        raise refusal("seed42_mismatch", "; ".join(bad))


# --------------------------------------------------------------------------------------------------
# 7. telemetry (LRB2; IF§11.3)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TelemetryAttempt:
    """One attempt's VAL rows: (unit value, all-class mIoU or None, non-finite tag or None)."""
    slot: str
    run_id: str
    path: str
    sha256: str
    unit: str                                  # "iter" (KD and E1 trainers) or "epoch" (QAT)
    points: tuple

    def values(self) -> dict:
        return {u: v for u, v, _ in self.points}

    def tags(self) -> dict:
        """{unit value: non-finite tag} of the VAL rows whose all_class_miou is not finite (B-5 (e):
        R2-09; the tag is _val_point's "nan", "inf" or "-inf", or the row's own nonfinite tag)."""
        return {u: t for u, _, t in self.points if t is not None}


def _val_point(o: dict, unit: str, where: str):
    u = o.get(unit)
    if not _is_int(u) or u < 0 or "all_class_miou" not in o:
        raise refusal("telemetry_invalid", f"{where}: a VAL row without a non-negative integer {unit!r} "
                                           "or an all_class_miou field")
    v = o["all_class_miou"]
    nf = o.get("nonfinite")
    tag = nf.get("all_class_miou") if isinstance(nf, dict) else None
    if isinstance(v, bool) or not (v is None or isinstance(v, (int, float))):
        raise refusal("telemetry_invalid", f"{where}: all_class_miou {v!r} at {unit} {u} is not a number")
    if isinstance(v, (int, float)) and not math.isfinite(float(v)):
        tag, v = ("nan" if math.isnan(float(v)) else ("inf" if v > 0 else "-inf")), None
    return u, (None if v is None else float(v)), tag


def read_telemetry(inv: Inventory, slot: str, input_root: Path, *, unit: str) -> tuple:
    """The VAL rows of every telemetry attempt the export lists for `slot`, each file's sha256 checked
    before it is parsed (LRB2). `unit` is "iter" for the KD and E1 trainers and "epoch" for QAT
    (src/training/train_e1.py:492-500, src/training/train_distill.py:1262-1270,
    src/quant/qat.py:1060-1069). No entry, or an empty list, gives no attempt."""
    if unit not in TELEMETRY_UNITS:
        raise ValueError(f"unit must be one of {sorted(TELEMETRY_UNITS)}")
    items = inv.entry(slot).get("telemetry")
    if items is None:
        return ()
    if not isinstance(items, list):
        raise refusal("telemetry_invalid", f"slot {slot!r}: telemetry is not a list")
    out = []
    for i, it in enumerate(items):
        where = f"slot {slot!r} telemetry[{i}]"
        if (not isinstance(it, dict) or not _is_text(it.get("run_id")) or not _is_text(it.get("path"))
                or not isinstance(it.get("sha256"), str) or not _HEX64.fullmatch(it["sha256"])):
            raise refusal("telemetry_invalid", f"{where}: needs run_id, path and a 64-hex sha256")
        rel = safe_relative_path(it.get("path"), where)
        p = Path(input_root) / rel
        try:
            raw = p.read_bytes()             # read once: the rows parsed are the bytes hashed (LRB2)
        except OSError as e:
            raise refusal("telemetry_invalid", f"{where}: {p} cannot be read ({type(e).__name__})") from e
        digest = hashlib.sha256(raw).hexdigest()
        if digest != it["sha256"]:
            raise refusal("telemetry_sha256_mismatch", f"{where}: {rel} has sha256 {digest}, the export "
                                                       f"says {it['sha256']} (LRB2)")
        points, seen = [], set()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise refusal("telemetry_invalid", f"{where}: {rel} is not UTF-8 text") from e
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)              # E1's writer may emit NaN (train_e1.py:205-209)
            except ValueError as e:
                raise refusal("telemetry_invalid", f"{where}: {rel} line {n} is not JSON") from e
            if not isinstance(o, dict) or o.get("event") != "val":
                continue
            u, v, tag = _val_point(o, unit, f"{where} {rel} line {n}")
            if u in seen:
                raise refusal("telemetry_invalid", f"{where}: {rel} has two VAL rows at {unit} {u}")
            seen.add(u)
            points.append((u, v, tag))
        out.append(TelemetryAttempt(slot=slot, run_id=it["run_id"], path=rel, sha256=digest, unit=unit,
                                    points=tuple(points)))
    return tuple(out)


# --------------------------------------------------------------------------------------------------
# 8. B-5's readers: the slots of role arm, arm_teacher and quant_descriptive with the family slots they
#    compare with (IF§11.1 :25), and the duplicate-free TEST list (B5-S2)
# --------------------------------------------------------------------------------------------------
def role_slots(inv: Inventory, role: str) -> list[str]:
    """The slots whose entry's role includes `role`, in the export's order (IF§11.1). B-5's tables read
    arm, arm_teacher and quant_descriptive; every other slot is ignored by them (LRB4)."""
    if role not in ROLES:
        raise ValueError(f"role {role!r} is not one of {ROLES}")
    return [slot for slot, items in inv.entries.items() if any(role in e["role"] for e in items)]


def family_inputs(stats: StatsArtifact, inv: Inventory, cells: PlannedCells,
                  stages=B5_FAMILY_STAGES) -> dict:
    """The family slots B-5's tables compare with (E2, E3, E6 and E7 at seed 42; IF§11.1 :25): the 37's
    clean inputs, each checked to be its family slot's model (R-11)."""
    check_seed42_identity(inv, cells, stats, stages=stages)
    return {stage: stats.clean(stage) for stage in stages}


def _b5_parent(inv: Inventory, slot: str, e: dict, stage: str, seed) -> str:
    """A B-5 slot's catalogue.parent: a slot at B5_PARENT_STAGE[stage] and the same seed, and, when that
    stage is a planned one (E2 for AM-23's E2 parents, E3 for E6-KD), its family slot, the model of record
    (IF§11.1; R-12)."""
    parent = _parent_of(inv, e, slot)
    pe = inv.entry(parent)
    if pe.get("stage") != B5_PARENT_STAGE[stage]:
        raise refusal("parent_rule", f"{slot!r} ({stage}): its parent {parent!r} is {pe.get('stage')!r}, "
                                     f"not {B5_PARENT_STAGE[stage]} (AM-18 item 3(a); AM-23 items 1-3; AM-3)")
    if pe.get("seed") != seed:
        raise refusal("seed_inconsistent", f"{slot!r} (seed {seed}) has parent {parent!r} at seed "
                                           f"{pe.get('seed')!r}")
    if B5_PARENT_STAGE[stage] in STAGES and "family" not in (pe.get("role") or ()):
        raise refusal("parent_rule", f"{slot!r} ({stage}): its parent {parent!r} is not the family slot of "
                                     f"{B5_PARENT_STAGE[stage]} seed {seed}, the model of record (IF§11.1)")
    return parent


def _b5_with_model(inv: Inventory, st: SlotStatus, parent: str) -> SlotStatus:
    """R-12 for a B-5 model (check_parentage's first rule): a model never descends from a run without one.
    A slot without a model keeps its IF§11.2 label, never a refusal (B-5 (a)), so check_parentage's not-run
    rule is not applied to B-5's slots."""
    ps = b5_status(inv, parent)
    if not ps.has_model:
        raise refusal("parent_rule", f"{st.slot!r} ({st.stage}) has a model, but its parent {parent!r} has "
                                     f"none ({ps.label})")
    return st


def b5_status(inv: Inventory, slot: str) -> SlotStatus:
    """IF§11.2's map for a slot B-5's tables read (an arm's student or teacher, the teacher of record,
    A, F, G, AM-23's quant_descriptive slots, E6-KD), by its stage's kind (B5-S1): a QAT run's selection
    reads its QAT run's kind unless it is derived; a PTQ model with kind null takes its parent's cut or
    divergence; a run reads its own. Parentage runs through B5_PARENT_STAGE at the same seed (to the family
    slot when that stage is planned), and a model's parent run has a model (R-12)."""
    e, kind = _checked_kind(inv, slot)
    stage, seed = e.get("stage"), e.get("seed")
    stage_spec(stage)

    def status(has_model: bool, label, source: str, record, decided):
        return SlotStatus(slot, stage, seed, decided, has_model, label, source, record)

    if stage in B5_QAT_STAGES:
        run_slot = _parent_of(inv, e, slot)
        run = inv.entry(run_slot)
        if run.get("stage") != stage or run.get("seed") != seed:
            raise refusal("parent_rule", f"selection {slot!r}: its parent {run_slot!r} is not the {stage} "
                                         f"QAT run at seed {seed} (LRB5)")
        parent = _b5_parent(inv, run_slot, run, stage, seed)
        if kind == "derived":
            return _b5_with_model(inv, status(True, None, slot, _record_of(e), kind), parent)
        re_, rk = _checked_kind(inv, run_slot)
        if rk == "not_run":
            label, src, rec = _not_run_status(inv, run_slot, re_, seed)
            return status(False, label, src, rec, rk)
        if rk in QAT_RUN_NO_MODEL_KINDS:
            _check_unit(run_slot, re_, "epoch")
            return status(False, _label(re_), run_slot, _record_of(re_), rk)
        raise refusal("status_invalid", f"selection {slot!r} has no model (kind {kind!r}) but its QAT run "
                                        f"{run_slot!r} is {rk!r}")
    if stage in B5_PTQ_STAGES:
        parent = _b5_parent(inv, slot, e, stage, seed)
        if kind == "derived":
            return _b5_with_model(inv, status(True, None, slot, _record_of(e), kind), parent)
        if kind is None:
            pe, pk = _checked_kind(inv, parent)
            if pk in ("cut", "diverged"):
                return status(False, LABEL_PARENT_CUT if pk == "cut" else LABEL_PARENT_DIVERGED, parent,
                              _record_of(pe), "not_run")
            raise refusal("status_invalid", f"PTQ slot {slot!r} has kind null and its parent {parent!r} is "
                                            f"{pk!r}: any other null case is refused (IF§11.2; R-4)")
        if kind == "not_run":
            label, src, rec = _not_run_status(inv, slot, e, seed)
            return status(False, label, src, rec, kind)
        if kind in PTQ_NO_MODEL_KINDS:
            return status(False, _label(e), slot, _record_of(e), kind)
        raise refusal("status_invalid", f"slot {slot!r}: kind {kind!r} is not a status of a {stage} model "
                                        f"(derived, null or {PTQ_NO_MODEL_KINDS})")
    if kind in B5_RUN_MODEL_KINDS:
        return status(True, None, slot, _record_of(e), kind)
    if kind == "not_run":
        label, src, rec = _not_run_status(inv, slot, e, seed)
        return status(False, label, src, rec, kind)
    if kind in RUN_NO_MODEL_KINDS:
        _check_unit(slot, e, "iteration")
        return status(False, _label(e), slot, _record_of(e), kind)
    raise refusal("status_invalid", f"slot {slot!r}: kind {kind!r} is not a status of a {stage} run "
                                    f"({B5_RUN_MODEL_KINDS}, not_run or {RUN_NO_MODEL_KINDS})")


@dataclass(frozen=True)
class DedupRecord:
    """B5-S2: TMAN's duplicate-free TEST list (IF§11.5; T4), its sha256 checked before it was parsed."""
    path: str                                       # the export's dedup.path
    file: Path                                      # resolved against the export's directory (H:10's rule)
    sha256: str
    kept_ids: tuple[str, ...]                       # sorted and unique
    removed_by_reason: tuple[tuple[str, int], ...]  # (reason, count), sorted by reason
    other_fields: tuple[str, ...]                   # the file's keys LREP does not read

    @property
    def removed_total(self) -> int:
        return sum(n for _, n in self.removed_by_reason)


def _reason_counts(value):
    if not isinstance(value, dict) or not all(_is_text(k) and _is_int(n) and n >= 0
                                              for k, n in value.items()):
        return None
    return tuple(sorted(value.items()))


def read_dedup(inv: Inventory) -> DedupRecord:
    """B5-S2 (ruling LREP-B5-RULE-2 item 2): the export's `dedup` {path, sha256, removed_by_reason}
    (IF§11.5 :85) and the file at `path`, resolved like binding.path (against the export's directory; B-4
    checks), its sha256 compared before it is parsed as strict JSON. Without TMAN's committed schema the
    file is T4's (plan :62): `kept_ids`, the kept (duplicate-free) image ids, sorted, unique and not
    empty, and `removed_by_reason`, the per-reason removed counts, equal to the export's. Any failure, or
    `dedup` null, is dedup_invalid. The id-set invariants against the scored artifacts are checked where
    those are read (src/stats/dedup_subset.py)."""
    d = inv.doc.get("dedup")
    where = f"{inv.path}: dedup"
    if d is None:
        raise refusal("dedup_invalid", f"{where} is null: tday_dedup.json does not exist yet, and the "
                                       "duplicate-free part needs it (IF§11.5)")
    if not isinstance(d, dict) or any(k not in d for k in DEDUP_EXPORT_FIELDS):
        raise refusal("dedup_invalid", f"{where} is not an object with {list(DEDUP_EXPORT_FIELDS)}")
    counts, problem = _reason_counts(d["removed_by_reason"]), path_problem(d["path"])
    if problem is not None or counts is None or not (isinstance(d["sha256"], str)
                                                     and _HEX64.fullmatch(d["sha256"])):
        raise refusal("dedup_invalid", f"{where}: path {d['path']!r} ({problem or 'plain'}), sha256 "
                                       f"{d['sha256']!r} or removed_by_reason {d['removed_by_reason']!r} "
                                       "is malformed")
    f = inv.path.parent / d["path"]
    try:
        raw = f.read_bytes()
    except OSError as e:
        raise refusal("dedup_invalid", f"{where}: {f} cannot be read ({type(e).__name__})") from e
    digest = hashlib.sha256(raw).hexdigest()
    if digest != d["sha256"]:
        raise refusal("dedup_invalid", f"{where}: {f} has sha256 {digest}, the export says {d['sha256']} "
                                       "(B5-S2)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise refusal("dedup_invalid", f"{where}: {f} is not UTF-8") from e
    doc = strict_json_text(text, str(f), "dedup_invalid")
    if not isinstance(doc, dict) or any(k not in doc for k in DEDUP_FILE_FIELDS):
        raise refusal("dedup_invalid", f"{where}: {f} is not an object with {list(DEDUP_FILE_FIELDS)} (T4)")
    ids = doc["kept_ids"]
    if not (isinstance(ids, list) and ids and all(_is_text(i) for i in ids) and ids == sorted(set(ids))):
        raise refusal("dedup_invalid", f"{where}: kept_ids are not a non-empty sorted list of unique image "
                                       "ids (T4)")
    if _reason_counts(doc["removed_by_reason"]) != counts:
        raise refusal("dedup_invalid", f"{where}: the file's removed_by_reason {doc['removed_by_reason']!r} "
                                       f"is not the export's {d['removed_by_reason']!r}")
    return DedupRecord(path=d["path"], file=f, sha256=digest, kept_ids=tuple(ids), removed_by_reason=counts,
                       other_fields=tuple(sorted(set(doc) - set(DEDUP_FILE_FIELDS))))


# --------------------------------------------------------------------------------------------------
# 9. the descriptive TEST read (B-5b: ruling LREP-B5-RULE-4 R4-1). Nothing frozen changes: every step
#    calls a frozen function (driver.load_input through load_bound, ingest.apply_am5_policy,
#    align.assert_compatible, ingest.am5_pair, align.align_runs) and Policy gains no member.
# --------------------------------------------------------------------------------------------------
def slot_read(e: dict) -> str:
    """R4-1 item 2: B5-S1's column `read` of a slot, from its entry's role and its stage's precision in
    B5-S1's map, never from its artifact's status: descriptive_test for role calibration_sensitivity, or
    role arm at int8_ptq; official for every other slot (a stage outside the map reads official here, and
    load_bound refuses it: stage_unmapped)."""
    roles = e.get("role") or ()
    spec = STAGE_MAP.get(e.get("stage")) if isinstance(e.get("stage"), str) else None
    if DT_ROLE in roles or ("arm" in roles and spec is not None and spec[2] == DT_ARM_PRECISION):
        return READ_DESCRIPTIVE_TEST
    return READ_OFFICIAL


def read_label(read: str) -> str | None:
    """R4-1 item 6: the label a row built from a DT run prints (DT_LABEL), None for an official read."""
    return DT_LABEL if read == READ_DESCRIPTIVE_TEST else None


def dt_read(policy: Policy, read: str) -> bool:
    """R4-1 item 3: a slot goes through the DT read iff it reads descriptive_test and the profile reads under
    Policy.OFFICIAL. In the smoke profile every slot reads through load_bound under the profile's policy."""
    return policy is Policy.OFFICIAL and read == READ_DESCRIPTIVE_TEST


def read_descriptive_test(inv: Inventory, *, slot: str, stage: str, precision: str, input_root: Path,
                          reference) -> BoundArtifact:
    """R4-1 item 3: a descriptive_test slot's clean artifact, read for the official profile. Any failure is
    descriptive_input_invalid, naming its step and the step's rule text (DT_STEPS):
      (a) load_bound under Policy.NONOFFICIAL_SMOKE, the loader of every slot (four-file hashes, MANIFEST
          verify, driver.load_input's canvas guard, driver.py:272), and within it (f), the artifact's files,
          checkpoint, commit and split manifest against its binding record (R-7; load_bound's
          artifact_record_mismatch);
      (b) artifact_status provisional (a smoke or official DT artifact is refused), split test, random_init
          false, expected_rows == actual_rows == ingest.EXPECTED_ROWS_TEST;
      (c) its rows ingest.apply_am5_policy(rows, Policy.OFFICIAL), set by dataclasses.replace;
      (d) `reference` -- the 37's input of its evaluator stage (the family slot at seed 42: E7 for E7-X and
          the E7 lists, E4 for the E4 lists), a driver.LoadedInput read under Policy.OFFICIAL -- passes
          align.assert_compatible(reference, run, Policy.OFFICIAL) and ingest.am5_pair(reference, run).
    (e) is dt_pair's. The run keeps its Policy.NONOFFICIAL_SMOKE, which marks it non-official wherever it
    goes; the artifact reads descriptive_test."""
    where = f"the DT read of {stage} ({slot!r}, {precision})"
    try:
        ba = load_bound(inv, slot=slot, stage=stage, precision=precision, scope="clean",
                        input_root=input_root, policy=Policy.NONOFFICIAL_SMOKE)
    except LrepRefusal as e:
        if e.code == "artifact_record_mismatch":
            raise refusal("descriptive_input_invalid", f"{where}: (f) {DT_STEPS['f']}: {e}") from e
        raise refusal("descriptive_input_invalid", f"{where}: (a) {DT_STEPS['a']}: {e}") from e
    ident = ba.run.identity
    bad = [what for ok, what in (
        (ident.artifact_status == "provisional", f"artifact_status {ident.artifact_status!r}"),
        (ident.split == "test", f"split {ident.split!r}"),
        (ident.random_init is False, f"random_init {ident.random_init!r}"),
        (ident.expected_rows == ident.actual_rows == EXPECTED_ROWS_TEST,
         f"expected_rows {ident.expected_rows}, actual_rows {ident.actual_rows}")) if not ok]
    if bad:
        raise refusal("descriptive_input_invalid", f"{where}: (b) {DT_STEPS['b']}: {bad}")
    try:
        records = apply_am5_policy(ba.run.records, Policy.OFFICIAL)
    except IngestError as e:
        raise refusal("descriptive_input_invalid", f"{where}: (c) {DT_STEPS['c']}: {e}") from e
    run = dataclasses.replace(ba.run, records=records)
    ev_stage = stage_spec(stage)[0]
    ref = getattr(reference, "run", None)
    if (ref is None or ref.policy is not Policy.OFFICIAL or ref.identity.stage != ev_stage
            or not isinstance(reference, D.LoadedInput)):
        problem = (f"the reference is not the 37's {ev_stage} input read under Policy.OFFICIAL (a "
                   f"{type(reference).__name__}, policy {getattr(ref, 'policy', None)})")
    else:
        try:
            assert_compatible(ref.identity, run.identity, Policy.OFFICIAL)
            am5_pair(ref, run)
            problem = None
        except (AlignmentError, IngestError) as e:
            problem = f"{type(e).__name__}: {e}"
    if problem is not None:
        raise refusal("descriptive_input_invalid", f"{where}: (d) {DT_STEPS['d']}: {problem}")
    return dataclasses.replace(ba, loaded=dataclasses.replace(ba.loaded, run=run), read=READ_DESCRIPTIVE_TEST)


def dt_pair(reference, ba: BoundArtifact, *, input_root: Path, metric: str):
    """R4-1 item 3(e): the pair of a DT run with its reference (the arms' E7-X − E7, B5-D2) through
    align.align_runs under Policy.NONOFFICIAL_SMOKE. The reference is re-loaded under that policy by
    driver.load_input, its files those of its official read, with (c)'s rows (ingest.apply_am5_policy under
    Policy.OFFICIAL), so that the PairedVector's own policy marks the pair non-official. Any failure is
    descriptive_input_invalid."""
    from .align import align_runs
    where = f"the DT pair {ba.stage} ({ba.slot!r}) - {getattr(reference, 'stage', None)}"
    try:
        again = D.load_input(reference.stage, reference.condition, reference.rel_path, Path(input_root),
                             Policy.NONOFFICIAL_SMOKE)
        if dict(again.file_sha256) != dict(reference.file_sha256):
            problem = "the reference's files differ from those of its official read"
        else:
            ref = dataclasses.replace(again.run, records=apply_am5_policy(again.run.records, Policy.OFFICIAL))
            return align_runs(ref, ba.run, policy=Policy.NONOFFICIAL_SMOKE, metric=metric)
    except (D.DriverRefusal, IngestError, AlignmentError) as e:
        problem = f"{type(e).__name__}: {e}"
    raise refusal("descriptive_input_invalid", f"{where}: (e) {DT_STEPS['e']}: {problem}")


def load_slot(stats, inv: Inventory, *, slot: str, stage: str, precision: str, input_root: Path
              ) -> BoundArtifact:
    """R4-1 items 2-4: the clean artifact of a B-5 slot the arm tables or am6's coverage read, by its `read`
    (slot_read). Under Policy.OFFICIAL a descriptive_test slot goes through read_descriptive_test, its
    reference the 37's input of its evaluator stage (stats.clean); every other slot through load_bound under
    the statistics artifact's policy, as before, so that under Policy.OFFICIAL a provisional artifact of an
    official-capable slot is refused there (artifact_unreadable) with no downgrade path. In the smoke profile
    a descriptive_test slot is read as before and still reads descriptive_test: its read comes from its
    role and precision, never from its status."""
    read = slot_read(inv.entry(slot))
    if dt_read(stats.policy, read):
        return read_descriptive_test(inv, slot=slot, stage=stage, precision=precision, input_root=input_root,
                                     reference=stats.clean(stage_spec(stage)[0]))
    ba = load_bound(inv, slot=slot, stage=stage, precision=precision, scope="clean", input_root=input_root,
                    policy=stats.policy)
    return dataclasses.replace(ba, read=read)


def require_official_inputs(stats: StatsArtifact, inputs: Mapping[str, object], where: str) -> None:
    """R4-1 item 5: an official computation (here the eight contrasts and NI that the duplicate-free part
    recomputes, dedup_subset (ii)-(iii)) takes the 37 inputs of the verified statistics artifact only. Each
    input -- a driver.LoadedInput or its EvaluationRun -- must be one of stats.views' own objects (identity,
    not equality); anything else, a DT run among them, is descriptive_input_in_official."""
    ours = {id(x) for li in stats.views.loaded.values() for x in (li, li.run)}
    bad = sorted(label for label, x in inputs.items() if id(x) not in ours)
    if bad:
        raise refusal("descriptive_input_in_official", f"{where}: {bad[:4]} (of {len(bad)}) not among the "
                                                       "37 inputs of the verified statistics artifact; a DT "
                                                       "run never reaches an official computation (R4-1 "
                                                       "item 5)")


__all__ = [
    "INVENTORY_SCHEMA", "BINDING_SCHEMA", "KINDS", "ROLES", "SCOPES", "PROTOCOLS", "PRECISIONS",
    "INT8_PRECISIONS", "PRECISION_FQOFF", "SCORED_ENTRY_KINDS", "STAGES", "SEEDS", "RECORD_ROLES",
    "RUN_STAGES", "PTQ_STAGES", "QAT_STAGES", "PARENT_STAGE", "ARM_XS", "ARM_STUDENT_STAGES",
    "QUANT_STAGES", "EXPLORATORY_STAGES", "E6KD_STAGE", "TEACHER_STAGES", "EVALUATOR_TEACHER_STAGE",
    "STAGE_MAP", "STAGE_ORDER", "B5_PARENT_STAGE", "B5_PTQ_STAGES", "B5_QAT_STAGES", "B5_RUN_MODEL_KINDS",
    "B5_FAMILY_STAGES", "DEDUP_EXPORT_FIELDS", "DEDUP_FILE_FIELDS", "E1_CHECKPOINT_RECORD",
    "LABEL_NON_FINITE", "LABEL_PARENT_CUT", "LABEL_PARENT_DIVERGED", "TELEMETRY_UNITS", "CODES",
    "LrepRefusal", "refusal", "sha256_file", "strict_json_text", "path_problem", "safe_relative_path",
    "cell_id", "corruption_names", "inferential_cells", "stage_spec", "read_inventory",
    "Inventory", "open_inventory", "SlotStatus", "record_slots", "parent_slot", "resolve_status",
    "entry_label", "PlannedCells", "check_parentage", "model_row",
    "row_identity", "check_rows", "check_missing_runs", "check_seed42", "planned_cells",
    "binding_record", "cell_condition", "BoundArtifact", "load_bound", "load_clean_of_record",
    "check_row_at_entry", "metrics_sha256", "cross_check", "check_e1_record", "StatsArtifact",
    "load_stats_artifact",
    "check_seed42_identity", "TelemetryAttempt", "read_telemetry", "role_slots", "family_inputs",
    "b5_status", "DedupRecord", "read_dedup",
    "READ_OFFICIAL", "READ_DESCRIPTIVE_TEST", "READS", "DT_ROLE", "DT_ARM_PRECISION", "DT_LABEL", "DT_STEPS",
    "slot_read", "read_label", "dt_read", "read_descriptive_test", "dt_pair", "load_slot",
    "require_official_inputs",
]
