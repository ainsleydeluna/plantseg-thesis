"""Synthetic TEST-day fixtures for lane LREP's smokes (B-1 to B-3).

SYNTHETIC ONLY (DL-41). Every evaluation artifact is generated from integer counts and written through
the unmodified evaluator writer by scripts/stats_fixtures.py, into a temporary clean fixture git
repository outside the working tree (the input root); no PlantSeg image, mask, split file or checkpoint
is read. The statistics artifact is built from the 37 seed-42 inputs by the frozen driver and writer
(driver.load_inputs -> assemble_or_refuse -> run_tasks -> artifact_inputs ->
artifact.write_statistics_artifact), as driver.run_smoke does but without its smoke-status screen: the
inputs are "provisional", so that they carry checkpoint identities (R-7, R-11).

The world also writes what TMAN will supply (IF§11 v2):
  * an inventory export `tday-inventory/1`: model rows and entries for the 21 planned slots, the two
    default candidates, E6-KD, and slots every LREP table must ignore (listed sweep runs, the losing
    pilot, a 160k control, a calibration-sensitivity E4, an arm, an arm teacher, the teacher, a
    quant_descriptive selection and the QAT run slots), whose records name directories that do not
    exist;
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

Names (LRB20; IF§2): runs/heldout/<stage>_s<seed>_<tail>, telemetry/<file_id>.jsonl, tman/, stats/,
strata/. stats_fixtures.rel_name is never used (it names folders by split), the JSON split value is
"val", and no created path contains "test". `make_root` follows scripts/smoke_mde.py: TMPDIR must be
set and free of "test", and the root's name ends in hex digits, which cannot spell it.

Not a smoke itself: imported by lane LREP's scripts/smoke_*.py.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
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
EXPORT_NAME = "tday_inventory.json"
BINDING_NAME = "tday_binding.json"


def make_root(prefix: str) -> tuple[Path | None, str | None]:
    """A new directory under TMPDIR, created only after TMPDIR's string and realpath are known to hold
    no 'test'; its name is `prefix` plus hex digits (scripts/smoke_mde.py's rule). Left in place."""
    base = os.environ.get("TMPDIR")
    if not base:
        return None, "TMPDIR is not set; run with TMPDIR=<a scratch directory without 'test'>"
    if "test" in base.lower():
        return None, "TMPDIR contains 'test'"
    real = os.path.realpath(base)
    if "test" in real.lower() or not os.path.isdir(real):
        return None, f"TMPDIR resolves to {real!r}, which contains 'test' or is not a directory"
    if "test" in prefix.lower():
        return None, f"prefix {prefix!r} contains 'test'"
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


def run_slot(stage: str, seed: int) -> str:
    return f"{stage}:s{seed}"


def record_slot(stage: str, seed: int) -> str:
    """The model-of-record slot: a run for E1-E4 and E7 (E4/E7 are derived slots of the same form),
    a QAT run's selection for E5 and E6 (TB9)."""
    return f"{stage}:s{seed}:sel" if stage in H.QAT_STAGES else run_slot(stage, seed)


def file_id(slot: str) -> str:
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
    """One artifact. Every model at one seed and condition shares that (seed, condition)'s noise draw
    and differs only by its quality, as a conversion differs from its parent, so the world's m values
    are ordered by quality at each seed and its states are known before it is scored."""
    pred = F.predict(world.truth, quality=quality, seed=F.int_seed(WORLD_SEED, "noise", seed, *cond))
    condition = Condition(*cond)
    scored = F.score(world.truth, pred, condition)
    write_precision = "int8_qat" if precision == H.PRECISION_FQOFF else precision
    d = world.input_root / rel
    F.write_fixture(d, scored, world.n, stage=H.evaluator_stage(stage), condition=condition,
                    split="val", status="provisional", repo_root=world.input_root, run_id=run_id,
                    precision=write_precision, checkpoint_sha256=checkpoint)
    if precision == H.PRECISION_FQOFF:
        F.edit_summary(d, lambda s: {**s, "run": {**s["run"], "precision": H.PRECISION_FQOFF}})
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
        _write(world, f"runs/heldout/{stage.lower()}_s42_{tail}_clean", slot=f"{stage}:s42:{tail}",
               stage=stage, seed=42, precision="fp32", scope="clean", cell=None, cond=D.CLEAN,
               quality=DEFAULT_QUALITY[stage], checkpoint=ckpt, run_id=f"lrep-{stage.lower()}-s42-{tail}")
    kd = "E6-KD:s42:sel"
    _write(world, "runs/heldout/e6kd_s42_clean", slot=kd, stage="E6-KD", seed=42, precision="int8_qat",
           scope="clean", cell=None, cond=D.CLEAN, quality=0.758,
           checkpoint=syn_sha("converted", "E6-KD", 42), run_id="lrep-e6kd-s42-clean")
    _write(world, "runs/heldout/e6kd_s42_fqoff", slot=kd, stage="E6-KD", seed=42,
           precision=H.PRECISION_FQOFF, scope="fq_off", cell=None, cond=D.CLEAN, quality=0.762,
           checkpoint=syn_sha("qat checkpoint", "E6-KD", 42), run_id="lrep-e6kd-s42-fqoff")


def edited_copy(world: World, src: Written, name: str, fn) -> Written:
    """A copy of an artifact under runs/heldout/<name>, its summary.json edited by `fn` and its
    MANIFEST rehashed; the copy is not in the binding (cases point records at it)."""
    if "test" in name.lower():
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
    if "test" in rel.lower():
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
    for slot, rid in (("E2:s42:lambda1", "e2_s42_lambda1_80k_a1"),
                      ("E3:s42:alpha50", "e3_s42_lambda0.5_alpha50_80k_a1")):
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
                entries.append(_entry(qrun, "listed_run", stage, seed, [], never_cut=False,
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
    for stage, tail, ckpt in (("E2", "lambda1", E2_DEFAULT), ("E3", "alpha50", E3_DEFAULT)):
        slot = f"{stage}:s42:{tail}"
        entries.append(_entry(slot, "default_candidate", stage, 42, ["default_candidate"],
                              never_cut=True, telemetry=world.telemetry[slot]))
        models.append(_row(slot, stage, 42, "fp32", source_kind="checkpoint", source_sha=ckpt,
                           entry_kind="default_candidate", roles=["default_candidate"],
                           run_id=world.telemetry[slot][0]["run_id"]))
    entries.append(_entry("E6-KD:s42", "listed_run", "E6-KD", 42, [], never_cut=False,
                          parent="E3:s42"))
    entries.append(_entry("E6-KD:s42:sel", "derived", "E6-KD", 42, ["listed"], parent="E6-KD:s42",
                          cls="derived"))
    for prec, sha, conv in (("int8_qat", syn_sha("qat checkpoint", "E6-KD", 42),
                             syn_sha("converted", "E6-KD", 42)),
                            (H.PRECISION_FQOFF, syn_sha("qat checkpoint", "E6-KD", 42), None)):
        models.append(_row("E6-KD:s42:sel", "E6-KD", 42, prec, source_kind="checkpoint",
                           source_sha=sha, converted=conv, entry_kind="derived", roles=["listed"],
                           run_id="e6kd_s42_qat_a1"))
    # slots every LREP table of B-1 to B-3 ignores (IF§11.1; LRB4)
    ghosts = (("E2:s42:lambda0.25", "listed_run", "E2", 42, ["listed"], None),
              ("U4:pilot", "listed_run", "E5", 42, ["listed"], None),
              ("E2:s42:160k", "scored_run", "E2", 42, ["control"], "fp32"),
              ("E4:s42:cal43", "derived", "E4", 42, ["calibration_sensitivity"], "int8_ptq"),
              ("E3-R1:s42", "scored_run", "E3-R1", 42, ["arm"], "fp32"),
              ("R1", "arm_teacher", "R1", None, ["arm_teacher"], "fp32"),
              ("teacher", "scored_run", "teacher", None, ["teacher_reference"], "fp32"),
              ("E6-E2:s42:sel", "derived", "E6-E2", 42, ["quant_descriptive"], "int8_qat"))
    for slot, kind, stage, seed, roles, prec in ghosts:
        entries.append(_entry(slot, kind, stage, seed, roles, never_cut=False))
        if prec is None:
            continue
        precs = [prec] + ([H.PRECISION_FQOFF] if prec == "int8_qat" else [])
        for pr in precs:
            models.append(_row(slot, stage, seed, pr, source_kind="checkpoint",
                               source_sha=syn_sha("ghost source", slot),
                               converted=syn_sha("ghost converted", slot), entry_kind=kind,
                               roles=roles, run_id=f"{file_id(slot)}_a1",
                               model_role="teacher" if stage in ("R1", "teacher") else "student"))
            records.append(_ghost_record(world, slot, pr))
    records.extend(_record(world, w) for w in world.written.values())
    manifest = syn_sha("TEST-day manifest")
    export = {"schema": H.INVENTORY_SCHEMA, "evaluator_commit": world.commit,
              "execution_context": {"model_device": "cpu", "input_devices": ["cpu"],
                                    "determinism_policy_applied": False, "batch_size": 1, "num_workers": 0,
                                    "torch_num_threads": 1, "image_digest": "sha256:" + syn_sha("image")},
              "manifest_sha256": manifest, "models": models, "entries": entries, "dedup": None,
              "binding": {"state": "bound", "path": BINDING_NAME, "sha256": None},
              "freeze": {"dl_row": "DL-synthetic", "manifest_sha256": manifest}}
    binding = {"schema": H.BINDING_SCHEMA, "artifacts": records}
    return export, binding


def write_inventory(out_dir: Path, export: dict, binding: dict | None) -> Path:
    """Write `binding` (if any) and `export` into `out_dir`; a bound export's binding.sha256 is set
    from the written binding bytes unless the export already pins another value."""
    out_dir = Path(out_dir)
    doc = copy.deepcopy(export)
    if binding is not None:
        bpath = out_dir / BINDING_NAME
        bpath.write_text(json.dumps(binding, indent=2, allow_nan=False) + "\n", encoding="utf-8",
                         newline="\n")
        if doc["binding"].get("state") == "bound" and doc["binding"].get("sha256") is None:
            doc["binding"]["sha256"] = H.sha256_file(bpath)
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
    write_all_telemetry(world)
    if stats:
        write_stats_artifact(world, B=B)
    if strata:
        write_strata(world)
    world.export, world.binding = build_export(world)
    world.inventory_path = write_inventory(new_dir(root, "tman"), world.export, world.binding)
    return world


__all__ = ["WORLD_SEED", "N_IMAGES", "STATS_B", "QUALITY", "SEED_SHIFT", "E2_SELECTED", "E3_SELECTED",
           "E2_DEFAULT", "E3_DEFAULT", "make_root", "new_dir", "syn_sha", "run_slot", "record_slot",
           "file_id", "training_run_id", "fp32_checkpoint", "identities", "Written", "World",
           "edited_copy", "write_telemetry", "kd_curve", "qat_curve", "write_stats_artifact",
           "sev45_cells", "build_export", "write_inventory", "write_strata", "build_world"]
