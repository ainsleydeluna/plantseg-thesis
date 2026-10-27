"""Synthetic TEST-day fixtures for lane LREP's smokes (B-1 to B-3; B-5).

SYNTHETIC ONLY (DL-41). Every evaluation artifact is generated from integer counts and written through
the unmodified evaluator writer by scripts/stats_fixtures.py, into a temporary clean fixture git
repository outside the working tree (the input root); no PlantSeg image, mask, split file or checkpoint
is read. The statistics artifact is built from the 37 seed-42 inputs by the frozen driver and writer
(driver.load_inputs -> assemble_or_refuse -> run_tasks -> artifact_inputs ->
artifact.write_statistics_artifact), as driver.run_smoke does but without its smoke-status screen: the
inputs are "provisional", so that they carry checkpoint identities (R-7, R-11).

The world also writes what TMAN will supply (IF§11 v2):
  * an inventory export `tday-inventory/1`: model rows and entries for the 21 planned slots, the two
    default candidates, E6-KD, B-5's slots (B5_SLOTS: the arms' students and teachers, the teacher of
    record, A, F, G, AM-23's quant_descriptive selections and their QAT runs, the 160k controls and the
    calibration-sensitivity E4 and E7), each scored model with a real clean artifact at the evaluator
    stage B5-S1's map gives it, and slots every LREP table must ignore (a listed sweep run, the losing
    pilot and the QAT run slots, the not-built L slots and ARM's measurement), whose records name
    directories that do not exist. Every slot key, stage, seed, role and parent is TMAN's committed
    catalogue's (CATALOGUE, typed from configs/tday_catalogue.json at lane/tman ee5593e; ruling
    LREP-B5-RULE-4 R4-5); the catalogue leaves E5:s42:sel's parent to its pilot (null), and the world
    names E5:s42:clip1:15ep, the pilot it takes as the winner;
  * B5-S2's duplicate-free TEST list tman/tday_dedup.json in T4's wording (kept_ids, removed_by_reason;
    four of the thirty images removed, the per-reason counts overlapping by one), named by the export's
    `dedup` {path, sha256, removed_by_reason};
  * a bound binding `tday_binding/1` with one record per (row, scope, cell), including records no
    B-1 to B-3 table reads (upstream, seed 42's corrupted cells of E2-E5 and E7, the corrupted_sev45
    cells; IF§9), which name directories that do not exist;
  * telemetry JSONL files for the run entries, in the trainers' VAL-row formats (iterations for the KD
    and E1 trainers, epochs for QAT).
Checkpoint identities: E1 carries DL-99's three checkpoints, E2 and E3 at seed 42 the selections of
DL-109 and DL-121, and the default candidates their defaults; every other identity is a synthetic
sha256. LRB7's exception: fq_off artifacts are written through the writer as int8_qat, then their
summary.json `run.precision` is rewritten to `qat_fakequant_off` and MANIFEST.sha256 rehashed
(stats_fixtures.edit_summary), because the writer refuses that value (src/eval/artifacts.py:57, :295).
A teacher's artifact (stage "teacher") gets `run.model_role` teacher the same way, because
stats_fixtures.write_fixture writes every artifact as a student's (scripts/stats_fixtures.py:218).

Names (LRB20; IF§2): runs/heldout/<stage>_s<seed>_<tail> (B-5's: runs/heldout/<file_id>_<tail>),
telemetry/<file_id>.jsonl, tman/, stats/, strata/. stats_fixtures.rel_name is never used (it names
folders by split), the JSON split value is "val", and no created path contains "test" or ":" (a slot
key's ":" stays a JSON string value; R4-6(i)). `make_root` follows scripts/smoke_mde.py: TMPDIR must be
set and free of "test" and ":", and the root's name ends in hex digits, which cannot spell it.

Not a smoke itself: imported by lane LREP's scripts/smoke_*.py.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import stats_fixtures as F                                               # noqa: E402
from src.eval.evaluate import Condition                                  # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats.bootstrap import ANALYSIS_ID_SMOKE                        # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

WORLD_SEED = 20261017
N_IMAGES = 30
STATS_B = 60
STATS_RUN_ID = "lrep-stats-0001"
CREATED_AT = "2026-10-17T00:00:00Z"
#: Mean prediction quality per stage and the per-seed shift; chosen so that the complete world is
#: "stable" (g_k > 0, mean gain above both SDs, d_k < 0.020); scripts/smoke_seed_stability.py
#: checks the states it gives.
QUALITY = {"E1": 0.700, "E2": 0.720, "E3": 0.760, "E4": 0.692, "E5": 0.694, "E6": 0.757, "E7": 0.752}
SEED_SHIFT = {42: 0.000, 43: -0.004, 44: 0.003}
FQOFF_GAIN = 0.004
DEFAULT_QUALITY = {"E2": 0.712, "E3": 0.731}
#: DL-109 and DL-121: the selected and default checkpoints at seed 42.
E2_SELECTED = "73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06"
E3_SELECTED = "4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d"
E2_DEFAULT = "7e082eae3aaba14cf7c098bdfc92c5e0c09393dbbab77fedcc4a071df234c4bf"
E3_DEFAULT = "b891406c5e734a48fece00fce8d3f978bb2bc762833b604214cf3f0df5afaea6"
KD_VAL_ITERS = tuple(range(4000, 80001, 4000))
QAT_VAL_EPOCHS = tuple(range(1, 16))
#: The default candidates' slots (the catalogue's keys; AM-17 item 6).
DEFAULT_SLOTS = {"E2": "E2:s42:lambda1:80k", "E3": "E3:s42:lambda0.5:alpha50:80k"}
EXPORT_NAME = "tday_inventory.json"
BINDING_NAME = "tday_binding.json"
DEDUP_NAME = "tday_dedup.json"
#: TMAN's committed catalogue, typed out (never imported; R4-5(e), (g)): slot -> (stage, seed, class, roles,
#: evaluator_stage, model_role, precision, parent), from configs/tday_catalogue.json at lane/tman ee5593e.
CATALOGUE_SOURCE = {"commit": "ee5593e9bd1f36d23b2ea2136120ae7890ed0728",
                    "path": "configs/tday_catalogue.json", "blob": "3cc45846d215964dbcda75669dd9185cf76e8b4b",
                    "sha256": "6c05a78ec34fd440999c939ea76d56779f891013c89668f4a0a5cfecca3438bd"}
CATALOGUE = {
    "teacher:s42:40k": ("teacher", 42, "run", ("teacher_reference",), "teacher", "teacher", "fp32", None),
    "E1:s42:80k": ("E1", 42, "run", ("family",), "E1", "student", "fp32", None),
    "E1:s43:80k": ("E1", 43, "run", ("per_seed",), "E1", "student", "fp32", None),
    "E1:s44:80k": ("E1", 44, "run", ("per_seed",), "E1", "student", "fp32", None),
    "E2:s42:lambda0.25:80k": ("E2", 42, "run", ("listed",), None, "student", "fp32", None),
    "E2:s42:lambda0.5:80k": ("E2", 42, "run", ("family",), "E2", "student", "fp32", None),
    "E2:s42:lambda1:80k": ("E2", 42, "run", ("default_candidate",), "E2", "student", "fp32", None),
    "E2:s42:lambda2:80k": ("E2", 42, "run", ("listed",), None, "student", "fp32", None),
    "E2:s42:lambda4:80k": ("E2", 42, "run", ("listed",), None, "student", "fp32", None),
    "E3:s42:lambda0.5:alpha25:80k": ("E3", 42, "run", ("family",), "E3", "student", "fp32", None),
    "E3:s42:lambda0.5:alpha50:80k": ("E3", 42, "run", ("default_candidate",), "E3", "student", "fp32", None),
    "E3:s42:lambda0.5:alpha100:80k": ("E3", 42, "run", ("listed",), None, "student", "fp32", None),
    "E2:s43:lambda0.5:80k": ("E2", 43, "run", ("per_seed",), "E2", "student", "fp32", None),
    "E2:s44:lambda0.5:80k": ("E2", 44, "run", ("per_seed",), "E2", "student", "fp32", None),
    "E3:s43:lambda0.5:alpha25:80k": ("E3", 43, "run", ("per_seed",), "E3", "student", "fp32", None),
    "E3:s44:lambda0.5:alpha25:80k": ("E3", 44, "run", ("per_seed",), "E3", "student", "fp32", None),
    "E1:s42:160k": ("E1", 42, "run", ("control",), "E1", "student", "fp32", None),
    "E2:s42:lambda0.5:160k": ("E2", 42, "run", ("control",), "E2", "student", "fp32", None),
    "E3:s42:lambda0.5:alpha25:160k": ("E3", 42, "run", ("control",), "E3", "student", "fp32", None),
    "A:s42:alpha25:80k": ("A", 42, "run", ("arm",), "A", "student", "fp32", None),
    "F:s42:alpha25:80k": ("F", 42, "run", ("arm",), "F", "student", "fp32", None),
    "G:s42:80k": ("G", 42, "run", ("arm",), "G", "student", "fp32", None),
    "teacher-R2:s42:40k": ("teacher-R2", 42, "run", ("arm_teacher",), "teacher", "teacher", "fp32", None),
    "E3-R2:s42:lambda0.5:alpha25:80k": ("E3-R2", 42, "run", ("arm",),
                                        "E3", "student", "fp32", "teacher-R2:s42:40k"),
    "E7-R2:s42:list42": ("E7-R2", 42, "derived", ("arm",),
                         "E7", "student", "int8_ptq", "E3-R2:s42:lambda0.5:alpha25:80k"),
    "teacher-R1:s42:40k": ("teacher-R1", 42, "run", ("arm_teacher",), "teacher", "teacher", "fp32", None),
    "E3-R1:s42:lambda0.5:alpha25:80k": ("E3-R1", 42, "run", ("arm",),
                                        "E3", "student", "fp32", "teacher-R1:s42:40k"),
    "E7-R1:s42:list42": ("E7-R1", 42, "derived", ("arm",),
                         "E7", "student", "int8_ptq", "E3-R1:s42:lambda0.5:alpha25:80k"),
    "E3-flip:s42:lambda0.5:alpha25:80k": ("E3-flip", 42, "run", ("arm",), "E3", "student", "fp32", None),
    "E7-flip:s42:list42": ("E7-flip", 42, "derived", ("arm",),
                           "E7", "student", "int8_ptq", "E3-flip:s42:lambda0.5:alpha25:80k"),
    "teacher-L:s42:40k": ("teacher-L", 42, "run", ("arm_teacher",), None, "teacher", "fp32", None),
    "E2-L:s42:lambda0.5:80k": ("E2-L", 42, "run", ("arm",), None, "student", "fp32", "teacher-L:s42:40k"),
    "E3-L:s42:lambda0.5:alpha25:80k": ("E3-L", 42, "run", ("arm",),
                                       None, "student", "fp32", "teacher-L:s42:40k"),
    "E7-L:s42:list42": ("E7-L", 42, "derived", ("arm",),
                        None, "student", "int8_ptq", "E3-L:s42:lambda0.5:alpha25:80k"),
    "E5:s42:clip1:15ep": ("E5", 42, "run", ("listed",), None, None, None, "E1:s42:80k"),
    "E5:s42:clip5:15ep": ("E5", 42, "run", ("listed",), None, None, None, "E1:s42:80k"),
    "E5:s42:sel": ("E5", 42, "derived", ("family",), "E5", "student", "int8_qat", None),
    "E6:s42:15ep": ("E6", 42, "run", ("listed",), None, None, None, "E3:s42:lambda0.5:alpha25:80k"),
    "E6:s42:sel": ("E6", 42, "derived", ("family",), "E6", "student", "int8_qat", "E6:s42:15ep"),
    "E5:s43:15ep": ("E5", 43, "run", ("listed",), None, None, None, "E1:s43:80k"),
    "E5:s43:sel": ("E5", 43, "derived", ("per_seed",), "E5", "student", "int8_qat", "E5:s43:15ep"),
    "E5:s44:15ep": ("E5", 44, "run", ("listed",), None, None, None, "E1:s44:80k"),
    "E5:s44:sel": ("E5", 44, "derived", ("per_seed",), "E5", "student", "int8_qat", "E5:s44:15ep"),
    "E6:s43:15ep": ("E6", 43, "run", ("listed",), None, None, None, "E3:s43:lambda0.5:alpha25:80k"),
    "E6:s43:sel": ("E6", 43, "derived", ("per_seed",), "E6", "student", "int8_qat", "E6:s43:15ep"),
    "E6:s44:15ep": ("E6", 44, "run", ("listed",), None, None, None, "E3:s44:lambda0.5:alpha25:80k"),
    "E6:s44:sel": ("E6", 44, "derived", ("per_seed",), "E6", "student", "int8_qat", "E6:s44:15ep"),
    "E6-KD:s42:15ep": ("E6-KD", 42, "run", ("listed",), None, None, None, "E3:s42:lambda0.5:alpha25:80k"),
    "E6-KD:s42:sel": ("E6-KD", 42, "derived", ("e6kd",), "E6-KD", "student", "int8_qat", "E6-KD:s42:15ep"),
    "E6-E2:s42:15ep": ("E6-E2", 42, "run", ("listed",), None, None, None, "E2:s42:lambda0.5:80k"),
    "E6-E2:s42:sel": ("E6-E2", 42, "derived", ("quant_descriptive",),
                      "E6-E2", "student", "int8_qat", "E6-E2:s42:15ep"),
    "E7-E2:s42:list42": ("E7-E2", 42, "derived", ("quant_descriptive",),
                         "E7-E2", "student", "int8_ptq", "E2:s42:lambda0.5:80k"),
    "E6-R1:s42:15ep": ("E6-R1", 42, "run", ("listed",), None, None, None, "E3-R1:s42:lambda0.5:alpha25:80k"),
    "E6-R1:s42:sel": ("E6-R1", 42, "derived", ("quant_descriptive",),
                      "E6-R1", "student", "int8_qat", "E6-R1:s42:15ep"),
    "E6-R2:s42:15ep": ("E6-R2", 42, "run", ("listed",), None, None, None, "E3-R2:s42:lambda0.5:alpha25:80k"),
    "E6-R2:s42:sel": ("E6-R2", 42, "derived", ("quant_descriptive",),
                      "E6-R2", "student", "int8_qat", "E6-R2:s42:15ep"),
    "E4:s42:list42": ("E4", 42, "derived", ("family",), "E4", "student", "int8_ptq", "E1:s42:80k"),
    "E4:s43:list42": ("E4", 43, "derived", ("per_seed",), "E4", "student", "int8_ptq", "E1:s43:80k"),
    "E4:s44:list42": ("E4", 44, "derived", ("per_seed",), "E4", "student", "int8_ptq", "E1:s44:80k"),
    "E7:s42:list42": ("E7", 42, "derived", ("family",),
                      "E7", "student", "int8_ptq", "E3:s42:lambda0.5:alpha25:80k"),
    "E7:s43:list42": ("E7", 43, "derived", ("per_seed",),
                      "E7", "student", "int8_ptq", "E3:s43:lambda0.5:alpha25:80k"),
    "E7:s44:list42": ("E7", 44, "derived", ("per_seed",),
                      "E7", "student", "int8_ptq", "E3:s44:lambda0.5:alpha25:80k"),
    "E4:s42:list43": ("E4", 42, "derived", ("calibration_sensitivity",),
                      "E4", "student", "int8_ptq", "E1:s42:80k"),
    "E4:s42:list44": ("E4", 42, "derived", ("calibration_sensitivity",),
                      "E4", "student", "int8_ptq", "E1:s42:80k"),
    "E4:s42:list45": ("E4", 42, "derived", ("calibration_sensitivity",),
                      "E4", "student", "int8_ptq", "E1:s42:80k"),
    "E7:s42:list43": ("E7", 42, "derived", ("calibration_sensitivity",),
                      "E7", "student", "int8_ptq", "E3:s42:lambda0.5:alpha25:80k"),
    "E7:s42:list44": ("E7", 42, "derived", ("calibration_sensitivity",),
                      "E7", "student", "int8_ptq", "E3:s42:lambda0.5:alpha25:80k"),
    "E7:s42:list45": ("E7", 42, "derived", ("calibration_sensitivity",),
                      "E7", "student", "int8_ptq", "E3:s42:lambda0.5:alpha25:80k"),
    "ARM:latency": ("ARM", None, "measurement", (), None, None, None, None),
}
#: src/tman/schema.py:182 at ee5593e, typed out (R4-5(e)): every fixture slot key matches it.
SLOT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*(?::[A-Za-z0-9.]+)+$")
#: The catalogue leaves E5:s42:sel's QAT run to the pilot decision (parent null, pilot runs clip1 and clip5);
#: the world takes clip1 as the winner, its run of record, and enters clip5 as the losing pilot.
E5_PILOT_OF_RECORD, E5_PILOT_LOST = "E5:s42:clip1:15ep", "E5:s42:clip5:15ep"


def _b5(slot: str, precisions: tuple, kind: str, quality):
    """A B-5 slot of the world: its stage, seed, class, roles and parent are the catalogue's."""
    stage, seed, cls, roles, _, _, _, parent = CATALOGUE[slot]
    return (slot, stage, seed, precisions, kind, cls, roles, parent, quality)


#: B-5's slots (IF§9; AM-18 item 3; AM-23 items 1-3; AM-3; AM-17 items 6-7; AM-16 item 5), each scored
#: model written as a real clean canvas artifact: (slot, stage, seed, precisions, kind, class, roles,
#: parent, quality), every field but precisions, kind and quality the catalogue's (R4-5).
B5_SLOTS = (
    _b5("E3-R1:s42:lambda0.5:alpha25:80k", ("fp32",), "scored_run", 0.768),
    _b5("E3-R2:s42:lambda0.5:alpha25:80k", ("fp32",), "scored_run", 0.790),
    _b5("E3-flip:s42:lambda0.5:alpha25:80k", ("fp32",), "scored_run", 0.757),
    _b5("E7-R1:s42:list42", ("int8_ptq",), "derived", 0.760),
    _b5("E7-R2:s42:list42", ("int8_ptq",), "derived", 0.781),
    _b5("E7-flip:s42:list42", ("int8_ptq",), "derived", 0.749),
    _b5("E6-R1:s42:15ep", (), "listed_run", None),
    _b5("E6-R1:s42:sel", ("int8_qat", "qat_fakequant_off"), "derived", 0.765),
    _b5("E6-R2:s42:15ep", (), "listed_run", None),
    _b5("E6-R2:s42:sel", ("int8_qat", "qat_fakequant_off"), "derived", 0.786),
    _b5("E6-E2:s42:15ep", (), "listed_run", None),
    _b5("E6-E2:s42:sel", ("int8_qat", "qat_fakequant_off"), "derived", 0.716),
    _b5("E7-E2:s42:list42", ("int8_ptq",), "derived", 0.713),
    _b5("teacher-R1:s42:40k", ("fp32",), "arm_teacher", 0.812),
    _b5("teacher-R2:s42:40k", ("fp32",), "arm_teacher", 0.821),
    _b5("teacher:s42:40k", ("fp32",), "scored_run", 0.805),
    _b5("A:s42:alpha25:80k", ("fp32",), "scored_run", 0.741),
    _b5("F:s42:alpha25:80k", ("fp32",), "scored_run", 0.744),
    _b5("G:s42:80k", ("fp32",), "scored_run", 0.738),
    _b5("E1:s42:160k", ("fp32",), "scored_run", 0.706),
    _b5("E2:s42:lambda0.5:160k", ("fp32",), "scored_run", 0.727),
    _b5("E4:s42:list43", ("int8_ptq",), "derived", 0.691),
    _b5("E7:s42:list43", ("int8_ptq",), "derived", 0.751),
)
#: The catalogue's not-built L arm (scope class not_built: never scored, evaluator_stage null) and ARM's
#: measurement (no role), entered without a model row or record: (slot, kind, the kind's fields). No LREP
#: table maps or loads them (R4-5(b)).
UNSCORED_SLOTS = (
    ("teacher-L:s42:40k", "cut", {"cut_date": "2026-11-02", "rule": "AM-19 item 3(b): code not merged",
                                  "comparator": None}),
    ("E2-L:s42:lambda0.5:80k", "not_run", {"reason": "parent cut"}),
    ("E3-L:s42:lambda0.5:alpha25:80k", "not_run", {"reason": "parent cut"}),
    ("E7-L:s42:list42", "not_run", {"reason": "parent cut"}),
    ("ARM:latency", "measurement", {}),
)
#: B5-S2's fixture: the images TMAN's hash list would remove (AM-18 item 8(c)), and per-reason counts that
#: overlap by one (an image equal to both a TRAIN and a VAL image), which the reader reports.
DEDUP_REMOVED = ("img_00003", "img_00011", "img_00017", "img_00024")
DEDUP_REASONS = {"equals_train_image": 2, "equals_val_image": 2, "repeats_an_earlier_image": 1}


def make_root(prefix: str) -> tuple[Path | None, str | None]:
    """A new directory under TMPDIR, created only after TMPDIR's string and realpath are known to hold
    no 'test'; its name is `prefix` plus hex digits (scripts/smoke_mde.py's rule). Left in place."""
    base = os.environ.get("TMPDIR")
    if not base:
        return None, "TMPDIR is not set; run with TMPDIR=<a scratch directory without 'test'>"
    if "test" in base.lower() or ":" in base:
        return None, "TMPDIR contains 'test' or ':'"
    real = os.path.realpath(base)
    if "test" in real.lower() or ":" in real or not os.path.isdir(real):
        return None, f"TMPDIR resolves to {real!r}, which contains 'test' or ':' or is not a directory"
    if "test" in prefix.lower() or ":" in prefix:
        return None, f"prefix {prefix!r} contains 'test' or ':'"
    root = Path(real) / f"{prefix}{secrets.token_hex(4)}"
    root.mkdir(exist_ok=False)
    return root, None


def new_dir(parent: Path, name: str) -> Path:
    """A fresh sub-directory whose name is checked for 'test' before it is created."""
    if "test" in name.lower() or "/" in name or ":" in name:
        raise ValueError(f"fixture directory name {name!r} is not allowed")
    d = Path(parent) / name
    d.mkdir(parents=False, exist_ok=False)
    return d


def syn_sha(*parts) -> str:
    return hashlib.sha256(("lrep synthetic " + " ".join(map(str, parts))).encode("utf-8")).hexdigest()


#: The catalogue's key tail of a planned stage's run (its QAT run for E5 and E6).
RUN_TAILS = {"E1": "80k", "E2": "lambda0.5:80k", "E3": "lambda0.5:alpha25:80k", "E5": "15ep", "E6": "15ep"}


def run_slot(stage: str, seed: int) -> str:
    """The catalogue's key of a planned stage's run: E1-E3's run, E5's and E6's QAT run (E5 at seed 42: the
    pilot of record)."""
    if (stage, seed) == ("E5", 42):
        return E5_PILOT_OF_RECORD
    return f"{stage}:s{seed}:{RUN_TAILS[stage]}"


def record_slot(stage: str, seed: int) -> str:
    """The model-of-record slot (the catalogue's key): a run for E1-E3, the PTQ model on list 42 for E4 and
    E7, a QAT run's selection for E5 and E6 (TB9)."""
    if stage in H.QAT_STAGES:
        return f"{stage}:s{seed}:sel"
    if stage in H.PTQ_STAGES:
        return f"{stage}:s{seed}:list42"
    return run_slot(stage, seed)


def file_id(slot: str) -> str:
    """A slot's file id: its ':' never becomes part of a path or file name (R4-6(i))."""
    return slot.lower().replace(":", "_").replace("-", "")


def training_run_id(stage: str, seed: int) -> str:
    return {"E1": f"e1_s{seed}_80k_a1", "E2": f"e2_s{seed}_lambda0.5_80k_a1",
            "E3": f"e3_s{seed}_lambda0.5_alpha25_80k_a1"}.get(stage, f"{stage.lower()}_s{seed}_qat_a1")


def fp32_checkpoint(stage: str, seed: int) -> str:
    if stage == "E1":
        return H.E1_CHECKPOINT_RECORD[seed]
    if (stage, seed) == ("E2", 42):
        return E2_SELECTED
    if (stage, seed) == ("E3", 42):
        return E3_SELECTED
    return syn_sha("checkpoint", stage, seed)


@dataclass
class Written:
    """One evaluation artifact the world wrote."""
    slot: str
    stage: str                      # the export stage (E6-KD for E6-KD)
    seed: int | None
    precision: str
    scope: str
    cell: str | None
    rel: str
    checkpoint: str
    files_sha256: dict
    run_id: str
    split_manifest_sha256: str


@dataclass
class World:
    root: Path
    input_root: Path
    commit: str
    truth: object
    n: int
    written: dict = field(default_factory=dict)        # (slot, precision, scope, cell) -> Written
    entries37: list = field(default_factory=list)      # (stage, condition, rel) of the 37
    telemetry: dict = field(default_factory=dict)      # slot -> [telemetry items]
    export: dict | None = None                         # binding.path/sha256 filled by write_inventory
    binding: dict | None = None
    dedup: dict | None = None                          # B5-S2's file; dedup.sha256 filled likewise
    stats_path: Path | None = None
    strata_path: Path | None = None
    inventory_path: Path | None = None
    seq: int = 0

    def art(self, stage: str, seed: int, precision: str | None = None, scope: str = "clean",
            cell: str | None = None) -> Written:
        slot = record_slot(stage, seed)
        return self.written[(slot, precision or D.STAGE_PRECISION[stage], scope, cell)]

    def fresh(self, name: str) -> Path:
        """A new case directory under root/cases."""
        self.seq += 1
        cases = self.root / "cases"
        if not cases.exists():
            new_dir(self.root, "cases")
        return new_dir(cases, f"{self.seq:03d}_{name}")


# --------------------------------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------------------------------
def _write(world: World, rel: str, *, slot: str, stage: str, seed, precision: str, scope: str,
           cell, cond: tuple, quality: float, checkpoint: str, run_id: str) -> Written:
    """One artifact, scored at the evaluator stage B5-S1's map gives `stage`. Every model at one seed and
    condition shares that (seed, condition)'s noise draw and differs only by its quality, as a
    conversion differs from its parent, so the world's m values are ordered by quality at each seed and
    its states are known before it is scored. A teacher's summary records model_role teacher and an
    fq_off one precision qat_fakequant_off (edit_summary; the writer takes neither)."""
    pred = F.predict(world.truth, quality=quality, seed=F.int_seed(WORLD_SEED, "noise", seed, *cond))
    condition = Condition(*cond)
    scored = F.score(world.truth, pred, condition)
    write_precision = "int8_qat" if precision == H.PRECISION_FQOFF else precision
    evaluator_stage, model_role, _ = H.stage_spec(stage)
    d = world.input_root / rel
    F.write_fixture(d, scored, world.n, stage=evaluator_stage, condition=condition,
                    split="val", status="provisional", repo_root=world.input_root, run_id=run_id,
                    precision=write_precision, checkpoint_sha256=checkpoint)
    edits = {}
    if precision == H.PRECISION_FQOFF:
        edits["precision"] = H.PRECISION_FQOFF
    if model_role != "student":
        edits["model_role"] = model_role
    if edits:
        F.edit_summary(d, lambda s: {**s, "run": {**s["run"], **edits}})
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    w = Written(slot=slot, stage=stage, seed=seed, precision=precision, scope=scope, cell=cell,
                rel=rel, checkpoint=checkpoint, files_sha256=D.input_file_sha256(d), run_id=run_id,
                split_manifest_sha256=summary["dataset"]["split_manifest_sha256"])
    world.written[(slot, precision, scope, cell)] = w
    return w


def _quality(stage: str, seed: int) -> float:
    return QUALITY[stage] + SEED_SHIFT[seed]


def identities(stage: str, seed: int) -> dict:
    """{source_kind, source_sha256, converted, artifact} for the model of record of (stage, seed)."""
    if stage in H.RUN_STAGES:
        c = fp32_checkpoint(stage, seed)
        return {"source_kind": "checkpoint", "source": c, "converted": None, "artifact": c}
    if stage in H.PTQ_STAGES:
        conv = syn_sha("converted", stage, seed)
        return {"source_kind": "provenance", "source": syn_sha("provenance", stage, seed),
                "converted": conv, "artifact": conv}
    conv = syn_sha("converted", stage, seed)
    return {"source_kind": "checkpoint", "source": syn_sha("qat checkpoint", stage, seed),
            "converted": conv, "artifact": conv}


def write_artifacts(world: World) -> None:
    grid = H.inferential_cells()
    for seed in H.SEEDS:
        for stage in D.STAGES:
            slot, prec, ids = record_slot(stage, seed), D.STAGE_PRECISION[stage], identities(stage, seed)
            q = _quality(stage, seed)
            w = _write(world, f"runs/heldout/{stage.lower()}_s{seed}_clean", slot=slot, stage=stage,
                       seed=seed, precision=prec, scope="clean", cell=None, cond=D.CLEAN, quality=q,
                       checkpoint=ids["artifact"], run_id=f"lrep-{stage.lower()}-s{seed}-clean")
            if seed == 42:
                world.entries37.append((stage, D.CLEAN, w.rel))
            if stage in D.CORRUPTION_STAGES:
                for cid, name, sev in grid:
                    cw = _write(world, f"runs/heldout/{stage.lower()}_s{seed}_{cid}", slot=slot,
                                stage=stage, seed=seed, precision=prec, scope="corrupted", cell=cid,
                                cond=("corruption", name, sev), quality=q - F.SEVERITY_DROP * sev,
                                checkpoint=ids["artifact"],
                                run_id=f"lrep-{stage.lower()}-s{seed}-{cid.replace('_', '-')}")
                    if seed == 42:
                        world.entries37.append((stage, ("corruption", name, sev), cw.rel))
            if stage in H.QAT_STAGES:
                _write(world, f"runs/heldout/{stage.lower()}_s{seed}_fqoff", slot=slot, stage=stage,
                       seed=seed, precision=H.PRECISION_FQOFF, scope="fq_off", cell=None, cond=D.CLEAN,
                       quality=q + FQOFF_GAIN, checkpoint=ids["source"],
                       run_id=f"lrep-{stage.lower()}-s{seed}-fqoff")
    for stage, ckpt in (("E2", E2_DEFAULT), ("E3", E3_DEFAULT)):
        tail = "lambda1" if stage == "E2" else "alpha50"
        _write(world, f"runs/heldout/{stage.lower()}_s42_{tail}_clean", slot=DEFAULT_SLOTS[stage],
               stage=stage, seed=42, precision="fp32", scope="clean", cell=None, cond=D.CLEAN,
               quality=DEFAULT_QUALITY[stage], checkpoint=ckpt, run_id=f"lrep-{stage.lower()}-s42-{tail}")
    kd = "E6-KD:s42:sel"
    _write(world, "runs/heldout/e6kd_s42_clean", slot=kd, stage="E6-KD", seed=42, precision="int8_qat",
           scope="clean", cell=None, cond=D.CLEAN, quality=0.758,
           checkpoint=syn_sha("converted", "E6-KD", 42), run_id="lrep-e6kd-s42-clean")
    _write(world, "runs/heldout/e6kd_s42_fqoff", slot=kd, stage="E6-KD", seed=42,
           precision=H.PRECISION_FQOFF, scope="fq_off", cell=None, cond=D.CLEAN, quality=0.762,
           checkpoint=syn_sha("qat checkpoint", "E6-KD", 42), run_id="lrep-e6kd-s42-fqoff")


def b5_identity(slot: str, precision: str) -> dict:
    """{source_kind, source, converted, artifact} of a B-5 slot's row of `precision` (IF§8, TL-6)."""
    if precision == "fp32":
        c = syn_sha("b5 checkpoint", slot)
        return {"source_kind": "checkpoint", "source": c, "converted": None, "artifact": c}
    if precision == "int8_ptq":
        conv = syn_sha("b5 converted", slot)
        return {"source_kind": "provenance", "source": syn_sha("b5 provenance", slot), "converted": conv,
                "artifact": conv}
    src = syn_sha("b5 qat checkpoint", slot)
    if precision == H.PRECISION_FQOFF:
        return {"source_kind": "checkpoint", "source": src, "converted": None, "artifact": src}
    conv = syn_sha("b5 converted", slot)
    return {"source_kind": "checkpoint", "source": src, "converted": conv, "artifact": conv}


def b5_run_id(slot: str, kind: str, parent) -> str:
    """The training run of a B-5 slot's rows (IF§11.1): a run's own; a QAT selection's QAT run's; a PTQ
    model's parent run's (a family run's id of record when the parent is one)."""
    src = parent if kind == "derived" else slot
    family = {run_slot(st, 42): training_run_id(st, 42) for st in H.RUN_STAGES}
    return family.get(src, f"{file_id(src)}_a1")


def write_b5_artifacts(world: World) -> None:
    """B-5's scored models (B5_SLOTS): one clean canvas artifact per row, and a QAT selection's fq_off
    model as Part 1's are."""
    for slot, stage, seed, precisions, kind, cls, roles, parent, quality in B5_SLOTS:
        for precision in precisions:
            ids = b5_identity(slot, precision)
            fq = precision == H.PRECISION_FQOFF
            tail = "fqoff" if fq else "clean"
            _write(world, f"runs/heldout/{file_id(slot)}_{tail}", slot=slot, stage=stage, seed=seed,
                   precision=precision, scope="fq_off" if fq else "clean", cell=None, cond=D.CLEAN,
                   quality=quality + (FQOFF_GAIN if fq else 0.0), checkpoint=ids["artifact"],
                   run_id=f"lrep-{file_id(slot).replace('_', '-')}-{tail}")


def edited_copy(world: World, src: Written, name: str, fn) -> Written:
    """A copy of an artifact under runs/heldout/<name>, its summary.json edited by `fn` and its
    MANIFEST rehashed; the copy is not in the binding (cases point records at it)."""
    if "test" in name.lower() or ":" in name or "/" in name:
        raise ValueError(name)
    d = world.input_root / "runs" / "heldout" / name
    shutil.copytree(world.input_root / src.rel, d)
    F.edit_summary(d, fn)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return Written(slot=src.slot, stage=src.stage, seed=src.seed, precision=src.precision,
                   scope=src.scope, cell=src.cell, rel=f"runs/heldout/{name}",
                   checkpoint=summary["run"].get("checkpoint_sha256"),
                   files_sha256=D.input_file_sha256(d), run_id=summary["run"]["run_id"],
                   split_manifest_sha256=summary["dataset"]["split_manifest_sha256"])


# --------------------------------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------------------------------
def write_telemetry(world: World, name: str, *, run_id: str, unit: str, points) -> dict:
    """A telemetry JSONL in the trainers' formats (src/training/train_e1.py:492-500,
    src/training/train_distill.py:1262-1270, src/quant/qat.py:1060-1069), with non-VAL rows between
    the VAL rows. Returns the export's telemetry item {run_id, path, sha256}."""
    rel = f"telemetry/{name}.jsonl"
    if "test" in rel.lower() or ":" in rel:
        raise ValueError(rel)
    tdir = world.input_root / "telemetry"
    if not tdir.exists():
        new_dir(world.input_root, "telemetry")
    rows = [{"event": "run_meta", "mode": "real", "run_id": run_id}]
    for u, v in points:
        if unit == "iter":
            rows.append({"event": "train", "iter": u, "loss": 0.5})
            rows.append({"event": "val", "iter": u, "all_class_miou": v,
                         "disease_only_miou_PROVISIONAL": v, "wall_clock": 1.79e9 + u})
        else:
            rows.append({"event": "val", "wall_clock": 1.79e9 + u, "epoch": u, "step": 1000 * u,
                         "kind": "fake_quant", "all_class_miou": v, "selects": False})
            rows.append({"event": "epoch_end", "epoch": u, "step": 1000 * u})
    p = world.input_root / rel
    if p.exists():
        raise FileExistsError(p)
    p.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8",
                 newline="\n")
    return {"run_id": run_id, "path": rel, "sha256": H.sha256_file(p)}


def kd_curve(final: float, iters=KD_VAL_ITERS, offset: float = 0.0):
    return [(it, round(final * (1.0 - math.exp(-it / 18000.0)) + offset, 8)) for it in iters]


def qat_curve(final: float, epochs=QAT_VAL_EPOCHS):
    return [(e, round(final - 0.02 * math.exp(-e / 3.0), 8)) for e in epochs]


def write_all_telemetry(world: World) -> None:
    for seed in H.SEEDS:
        for stage in H.RUN_STAGES:
            slot = run_slot(stage, seed)
            world.telemetry[slot] = [write_telemetry(
                world, file_id(slot), run_id=training_run_id(stage, seed), unit="iter",
                points=kd_curve(0.36 + 0.01 * H.RUN_STAGES.index(stage) + 0.001 * (seed - 42)))]
        for stage in H.QAT_STAGES:
            slot = run_slot(stage, seed)
            world.telemetry[slot] = [write_telemetry(
                world, file_id(slot), run_id=training_run_id(stage, seed), unit="epoch",
                points=qat_curve(0.35 + 0.01 * H.QAT_STAGES.index(stage) + 0.001 * (seed - 42)))]
    for slot, rid in ((DEFAULT_SLOTS["E2"], "e2_s42_lambda1_80k_a1"),
                      (DEFAULT_SLOTS["E3"], "e3_s42_lambda0.5_alpha50_80k_a1")):
        world.telemetry[slot] = [write_telemetry(world, file_id(slot), run_id=rid, unit="iter",
                                                 points=kd_curve(0.355))]


# --------------------------------------------------------------------------------------------------
# the statistics artifact (the 37 seed-42 inputs)
# --------------------------------------------------------------------------------------------------
def write_stats_artifact(world: World, B: int = STATS_B) -> Path:
    """driver.run_smoke's steps without its smoke-status screen, under NONOFFICIAL_SMOKE."""
    inputs = D.load_inputs(world.entries37, world.input_root, Policy.NONOFFICIAL_SMOKE)
    obs = D.assemble_or_refuse(inputs, Policy.NONOFFICIAL_SMOKE)
    tasks = D.run_tasks(obs, ANALYSIS_ID_SMOKE, B, jobs=1)
    inp = D.artifact_inputs(obs, inputs, tasks, analysis_id=ANALYSIS_ID_SMOKE, run_id=STATS_RUN_ID,
                            created_at_utc=CREATED_AT, contract_sha256=A.contract_sha256()[0],
                            synthetic=True, input_root=world.input_root)
    stats_dir = world.root / "stats"
    if not stats_dir.exists():
        new_dir(world.root, "stats")
    path = A.write_statistics_artifact(stats_dir / STATS_RUN_ID, inp)
    world.stats_path = path
    return path


# --------------------------------------------------------------------------------------------------
# the export and the binding (IF§8, IF§11)
# --------------------------------------------------------------------------------------------------
def _entry(slot, kind, stage, seed, roles, *, parent=None, cls="run", never_cut=True, **fields):
    return {"slot": slot, "kind": kind, "class": cls, "stage": stage, "seed": seed,
            "catalogue": {"never_cut": never_cut, "cut_place": None, "parent": parent},
            "dl_rows": [], "scope": ["clean", "upstream", "dedup"], "role": list(roles), **fields}


def diverged_fields(*, rule: str, run_id: str, iteration=None, epoch=None,
                    cause: str = "non-finite loss") -> dict:
    """A diverged entry's fields in TMAN's shape (src/tman/schema.py:398 and :413-414 at ee5593e; R4-5(f)):
    both iteration and epoch present, exactly one non-null, with run_id, rule, cause and telemetry_sha256."""
    return {"run_id": run_id, "rule": rule, "cause": cause, "iteration": iteration, "epoch": epoch,
            "telemetry_sha256": syn_sha("telemetry", run_id)}


def _row(slot, stage, seed, precision, *, source_kind, source_sha, converted=None, cells=(),
         entry_kind, roles, run_id, model_role="student"):
    row = {"file_id": file_id(slot), "stage": stage, "model_role": model_role,
           "precision": precision, "seed": seed,
           "source": {"kind": source_kind, "path": f"models/{file_id(slot)}.bin",
                      "sha256": source_sha},
           "cells": list(cells), "slot": slot, "entry_kind": entry_kind, "role": list(roles),
           "training_run_id": run_id}
    if precision in H.INT8_PRECISIONS:
        row["converted_artifact_sha256"] = converted
    return row


def _record(world: World, w: Written) -> dict:
    return {"slot": w.slot, "precision": w.precision, "scope": w.scope, "cell": w.cell,
            "protocol": "canvas", "repo_relative_path": w.rel, "files_sha256": dict(w.files_sha256),
            "checkpoint_sha256": w.checkpoint, "repo_commit": world.commit,
            "split_manifest_sha256": w.split_manifest_sha256}


def _ghost_record(world: World, slot: str, precision: str, scope: str | None = None,
                  cell: str | None = None) -> dict:
    """A record LREP's B-1 to B-3 tables never read, naming a directory that does not exist: an ignored
    slot's row (scope fq_off for a qat_fakequant_off row, else clean), or a scope of a planned row that
    no table of B-1 to B-3 reads (upstream, seed 42's corrupted cells of E2-E5 and E7, every
    corrupted_sev45 cell; IF§9)."""
    scope = scope or ("fq_off" if precision == H.PRECISION_FQOFF else "clean")
    tail = "_".join(x for x in (file_id(slot), scope, cell) if x)
    return {"slot": slot, "precision": precision, "scope": scope, "cell": cell,
            "protocol": "upstream" if scope == "upstream" else "canvas",
            "repo_relative_path": f"runs/heldout/never_written_{tail}",
            "files_sha256": {n: syn_sha("ghost", slot, scope, cell, n) for n in H.ARTIFACT_FILE_NAMES},
            "checkpoint_sha256": syn_sha("ghost checkpoint", slot), "repo_commit": world.commit,
            "split_manifest_sha256": syn_sha("ghost manifest")}


def b5_export_parts(world: World) -> tuple[list, list, list]:
    """(entries, model rows, upstream ghost records) of B-5's slots and the unscored ones (UNSCORED_SLOTS);
    their scored records come from world.written. A teacher's row carries model_role teacher (B5-S1); an
    fq_off row a student's."""
    entries, models, ghosts = [], [], []
    for slot, stage, seed, precisions, kind, cls, roles, parent, _ in B5_SLOTS:
        entries.append(_entry(slot, kind, stage, seed, list(roles), parent=parent, cls=cls,
                              never_cut=stage in H.TEACHER_STAGES))
        for precision in precisions:
            ids = b5_identity(slot, precision)
            role = "student" if precision == H.PRECISION_FQOFF else H.stage_spec(stage)[1]
            models.append(_row(slot, stage, seed, precision, source_kind=ids["source_kind"],
                               source_sha=ids["source"], converted=ids["converted"], entry_kind=kind,
                               roles=list(roles), run_id=b5_run_id(slot, kind, parent), model_role=role))
            if (precision != H.PRECISION_FQOFF and "arm_teacher" not in roles
                    and "calibration_sensitivity" not in roles):
                ghosts.append(_ghost_record(world, slot, precision, "upstream"))
    for slot, kind, fields in UNSCORED_SLOTS:
        stage, seed, cls, roles, _, _, _, parent = CATALOGUE[slot]
        entries.append(_entry(slot, kind, stage, seed, list(roles), parent=parent, cls=cls, never_cut=False,
                              **fields))
    return entries, models, ghosts


def dedup_doc(world: World) -> dict:
    """B5-S2's file in T4's wording (plan :62): the kept (duplicate-free) image ids, sorted and unique,
    with the per-reason removed counts."""
    return {"kept_ids": sorted(i for i in world.truth.ids if i not in DEDUP_REMOVED),
            "removed_by_reason": dict(DEDUP_REASONS)}


def dedup_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sev45_cells() -> list[str]:
    """IF§6's nine corrupted_sev45 cells: gaussian_noise at 4, the other four corruptions at 4 and 5."""
    return [H.cell_id(n, s) for n in H.corruption_names() for s in (4, 5)
            if not (n == "gaussian_noise" and s == 5)]


def build_export(world: World) -> tuple[dict, dict]:
    """(export, binding) for the complete world: every planned run has a model. Seed 42's rows carry
    the 15 inferential and 9 severity-4/5 cells, E1 and E6 at seeds 43/44 the 15 (IF§9)."""
    grid_ids = [cid for cid, _, _ in H.inferential_cells()]
    sev45 = sev45_cells()
    entries, models, records = [], [], []
    for seed in H.SEEDS:
        role = "family" if seed == 42 else "per_seed"
        for stage in D.STAGES:
            ids = identities(stage, seed)
            rslot = record_slot(stage, seed)
            prec = D.STAGE_PRECISION[stage]
            cells = (grid_ids + sev45 if seed == 42 else
                     grid_ids if stage in D.CORRUPTION_STAGES else [])
            records.append(_ghost_record(world, rslot, prec, "upstream"))
            if seed == 42:
                records += [_ghost_record(world, rslot, prec, "corrupted", cid) for cid in grid_ids
                            if stage not in D.CORRUPTION_STAGES]
                records += [_ghost_record(world, rslot, prec, "corrupted_sev45", cid) for cid in sev45]
            if stage in H.RUN_STAGES:
                entries.append(_entry(rslot, "scored_run", stage, seed, [role],
                                      telemetry=world.telemetry[rslot]))
                kind, run_id = "scored_run", training_run_id(stage, seed)
            elif stage in H.PTQ_STAGES:
                parent = run_slot(H.PARENT_STAGE[stage], seed)
                entries.append(_entry(rslot, "derived", stage, seed, [role], parent=parent,
                                      cls="derived"))
                kind, run_id = "derived", training_run_id(H.PARENT_STAGE[stage], seed)
            else:
                qrun = run_slot(stage, seed)
                entries.append(_entry(qrun, "listed_run", stage, seed, ["listed"], never_cut=False,
                                      parent=run_slot(H.PARENT_STAGE[stage], seed),
                                      telemetry=world.telemetry[qrun]))
                entries.append(_entry(rslot, "derived", stage, seed, [role], parent=qrun,
                                      cls="derived"))
                kind, run_id = "derived", training_run_id(stage, seed)
            models.append(_row(rslot, stage, seed, prec,
                               source_kind=ids["source_kind"], source_sha=ids["source"],
                               converted=ids["converted"], cells=cells, entry_kind=kind,
                               roles=[role], run_id=run_id))
            if stage in H.QAT_STAGES:
                models.append(_row(rslot, stage, seed, H.PRECISION_FQOFF, source_kind="checkpoint",
                                   source_sha=ids["source"], entry_kind=kind, roles=[role],
                                   run_id=run_id))
    for stage, ckpt in (("E2", E2_DEFAULT), ("E3", E3_DEFAULT)):
        slot = DEFAULT_SLOTS[stage]
        entries.append(_entry(slot, "default_candidate", stage, 42, ["default_candidate"],
                              never_cut=True, telemetry=world.telemetry[slot]))
        models.append(_row(slot, stage, 42, "fp32", source_kind="checkpoint", source_sha=ckpt,
                           entry_kind="default_candidate", roles=["default_candidate"],
                           run_id=world.telemetry[slot][0]["run_id"]))
    entries.append(_entry("E6-KD:s42:15ep", "listed_run", "E6-KD", 42, ["listed"], never_cut=False,
                          parent=run_slot("E3", 42)))
    entries.append(_entry("E6-KD:s42:sel", "derived", "E6-KD", 42, ["e6kd"], parent="E6-KD:s42:15ep",
                          cls="derived"))
    for prec, sha, conv in (("int8_qat", syn_sha("qat checkpoint", "E6-KD", 42),
                             syn_sha("converted", "E6-KD", 42)),
                            (H.PRECISION_FQOFF, syn_sha("qat checkpoint", "E6-KD", 42), None)):
        models.append(_row("E6-KD:s42:sel", "E6-KD", 42, prec, source_kind="checkpoint",
                           source_sha=sha, converted=conv, entry_kind="derived", roles=["e6kd"],
                           run_id="e6kd_s42_qat_a1"))
    # B-5's slots (B5_SLOTS), with real clean artifacts, and the unscored ones (UNSCORED_SLOTS)
    b5_entries, b5_models, b5_ghosts = b5_export_parts(world)
    entries += b5_entries
    models += b5_models
    records += b5_ghosts
    # slots every LREP table ignores (IF§11.1; LRB4): a listed sweep run and the losing pilot, no model row
    for slot in ("E2:s42:lambda0.25:80k", E5_PILOT_LOST):
        stage, seed, cls, roles, _, _, _, parent = CATALOGUE[slot]
        entries.append(_entry(slot, "listed_run", stage, seed, list(roles), parent=parent, cls=cls,
                              never_cut=False))
    records.extend(_record(world, w) for w in world.written.values())
    bad = sorted({x["slot"] for x in (*entries, *models, *records)} - set(CATALOGUE))
    if bad or not all(SLOT_KEY_RE.fullmatch(e["slot"]) for e in entries):
        raise ValueError(f"fixture slot keys outside TMAN's catalogue (R4-5(e)): {bad}")
    manifest = syn_sha("TEST-day manifest")
    export = {"schema": H.INVENTORY_SCHEMA, "evaluator_commit": world.commit,
              "execution_context": {"model_device": "cpu", "input_devices": ["cpu"],
                                    "determinism_policy_applied": False, "batch_size": 1, "num_workers": 0,
                                    "torch_num_threads": 1, "image_digest": "sha256:" + syn_sha("image")},
              "manifest_sha256": manifest, "models": models, "entries": entries,
              "dedup": {"path": DEDUP_NAME, "sha256": None, "removed_by_reason": dict(DEDUP_REASONS)},
              "binding": {"state": "bound", "path": BINDING_NAME, "sha256": None},
              "freeze": {"dl_row": "DL-synthetic", "manifest_sha256": manifest}}
    binding = {"schema": H.BINDING_SCHEMA, "artifacts": records}
    return export, binding


def write_inventory(out_dir: Path, export: dict, binding: dict | None, dedup: dict | None = None) -> Path:
    """Write `binding` and `dedup` (B5-S2's file; each if given) and `export` into `out_dir`; a bound
    export's binding.sha256, and its dedup.sha256, are set from the written bytes unless the export
    already pins another value."""
    out_dir = Path(out_dir)
    doc = copy.deepcopy(export)
    if binding is not None:
        bpath = out_dir / BINDING_NAME
        bpath.write_text(json.dumps(binding, indent=2, allow_nan=False) + "\n", encoding="utf-8",
                         newline="\n")
        if doc["binding"].get("state") == "bound" and doc["binding"].get("sha256") is None:
            doc["binding"]["sha256"] = H.sha256_file(bpath)
    if dedup is not None:
        dpath = out_dir / DEDUP_NAME
        dpath.write_bytes(dedup_bytes(dedup))
        if isinstance(doc.get("dedup"), dict) and doc["dedup"].get("sha256") is None:
            doc["dedup"]["sha256"] = H.sha256_file(dpath)
    p = out_dir / EXPORT_NAME
    p.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return p


# --------------------------------------------------------------------------------------------------
# the strata file (AM-17 item 8's schema, synthetic values)
# --------------------------------------------------------------------------------------------------
def write_strata(world: World) -> Path:
    """A synthetic plantseg-train-strata/1.0.0 file: 115 disease classes ranked by a synthetic TRAIN
    share, terciles 38/38/39 by rank, 28 classes flagged rare."""
    rng = np.random.default_rng(F.int_seed(WORLD_SEED, "strata"))
    pixels = rng.integers(10_000, 5_000_000, size=115).astype(np.int64)
    images = rng.integers(1, 200, size=115).astype(np.int64)
    order = sorted(range(115), key=lambda i: (-int(pixels[i]), i))
    rank = {i: r + 1 for r, i in enumerate(order)}
    rare = set(sorted(range(115), key=lambda i: (int(images[i]), i))[:28])
    total = int(pixels.sum())
    per_class = [{"class_id": i + 1, "pixels": int(pixels[i]), "share": int(pixels[i]) / total,
                  "images": int(images[i]), "rank": rank[i],
                  "tercile": "T1" if rank[i] <= 38 else ("T2" if rank[i] <= 76 else "T3"),
                  "rare": i in rare, "zero_train_pixels": False} for i in range(115)]
    doc = {"schema": "plantseg-train-strata/1.0.0", "lane": "LREP synthetic fixture",
           "split": "train", "num_classes": 116, "background_index": 0, "tercile_sizes": [38, 38, 39],
           "per_class": per_class}
    sdir = new_dir(world.root, "strata")
    p = sdir / "train_strata_synthetic.json"
    p.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    world.strata_path = p
    return p


# --------------------------------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------------------------------
def build_world(root: Path, *, n: int = N_IMAGES, B: int = STATS_B, stats: bool = True,
                strata: bool = True) -> World:
    """Every fixture LREP's smokes read, under `root` (a fresh directory from make_root)."""
    root = Path(root)
    input_root = F.clean_git_fixture(root / "fx")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(input_root), check=True,
                            capture_output=True, text=True).stdout.strip()
    truth = F.make_truth(n, 0, F.int_seed(WORLD_SEED, "truth"))
    world = World(root=root, input_root=input_root, commit=commit, truth=truth, n=n)
    write_artifacts(world)
    write_b5_artifacts(world)
    write_all_telemetry(world)
    if stats:
        write_stats_artifact(world, B=B)
    if strata:
        write_strata(world)
    world.export, world.binding = build_export(world)
    world.dedup = dedup_doc(world)
    world.inventory_path = write_inventory(new_dir(root, "tman"), world.export, world.binding, world.dedup)
    return world


__all__ = ["WORLD_SEED", "N_IMAGES", "STATS_B", "QUALITY", "SEED_SHIFT", "E2_SELECTED", "E3_SELECTED",
           "E2_DEFAULT", "E3_DEFAULT", "make_root", "new_dir", "syn_sha", "run_slot", "record_slot",
           "file_id", "training_run_id", "fp32_checkpoint", "identities", "Written", "World",
           "edited_copy", "write_telemetry", "kd_curve", "qat_curve", "write_stats_artifact",
           "sev45_cells", "build_export", "write_inventory", "write_strata", "build_world", "DEDUP_NAME",
           "B5_SLOTS", "DEDUP_REMOVED", "DEDUP_REASONS", "b5_identity", "b5_run_id", "write_b5_artifacts",
           "b5_export_parts", "dedup_doc", "dedup_bytes", "CATALOGUE_SOURCE", "CATALOGUE", "SLOT_KEY_RE",
           "E5_PILOT_OF_RECORD", "E5_PILOT_LOST", "UNSCORED_SLOTS", "DEFAULT_SLOTS", "RUN_TAILS",
           "diverged_fields"]
