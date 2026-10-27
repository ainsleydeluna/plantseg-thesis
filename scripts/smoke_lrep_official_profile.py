#!/usr/bin/env python3
"""The official profile of lane LREP (B-5b: ruling LREP-B5-RULE-4 R4-1): the descriptive TEST read ("DT
read") and its guards on an official, TEST-shaped synthetic world.

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's official world -- n = 1561 images, k = 7 AM-5 images,
the fixture's synthetic dataset name (never PlantSeg), every artifact official but the nine
descriptive_test slots' (provisional) -- written through the unmodified evaluator writer into a clean
fixture repository under a new directory in TMPDIR (left in place), plus edited or re-scored copies of
one DT artifact, of E3-R1's and of E7's for the refusals. This is the only smoke whose artifacts hold
dataset.split "test", as a JSON value (R4-6(ii)), and the only process that writes them: the mutation
harness runs this smoke's cases mutated in this smoke's own process (--mutations; DL-24 R3-01). No
created path or file name contains "test" or ":", and no id, key or name contains "test". The export is
read under Policy.OFFICIAL through a stand-in for TMAN's reader (the stub's checks without its stub mark;
B-4 swaps in TMAN's), and a DT slot's reference is the family slot of its evaluator stage at seed 42, read
as the 37's input is (driver.load_input under Policy.OFFICIAL).

  G  R4-1 item 2: slot_read over the catalogue's 49 scored slots gives exactly the nine; the DT read is
     reached only through load_slot, which only the arm tables and am6's coverage call; dt_pair only by
     the arm tables; the official-input guard sits where dedup's (ii) and (iii) and am6's seed-42 rows
     take their inputs
  H  the happy paths: each of the world's nine DT slots reads descriptive_test through the DT read (its
     run non-official, its rows the official AM-5 rule's); a DT pair with E7 is non-official, its delta
     the rows' own; every other B-5 slot reads official under Policy.OFFICIAL; the arm rows and am6's
     coverage under Policy.OFFICIAL (a stand-in for the statistics artifact's policy and clean inputs)
  A-F one case per step of the DT read (item 3), each reached with every earlier gate passing:
     A (a) an artifact its loader refuses; B (b) a smoke, official, VAL, random_init or 1560-row DT
     artifact; C (c) an undefined primary value AM-5 does not exclude; D (d) a reference that is not the
     37's E7 read officially, another metric implementation (assert_compatible), another AM-5 set
     (am5_pair), a DT-classed slot whose evaluator stage has no family slot among the 37; E (e) a
     reference whose files changed after its official read, a clean_image_id that differs (align_runs),
     a reference whose re-load refuses (its summary changed without a rehash: driver.load_input's
     DriverRefusal; DL-24 R3-04); F (f) a record whose files differ from the artifact's
  K  item 4: an official-capable slot's provisional artifact is refused under Policy.OFFICIAL
     (artifact_unreadable), never read as descriptive
  N  DL-24 R2-01, named for B-4 (a difference is reported, not coded): E5:s42:sel as TMAN's catalogue
     gives it (catalogue.parent null, the pilot deciding) is refused parent_rule by R-12, which reads
     catalogue.parent only; the world names the pilot of record there
  X  the world: its artifacts' dataset name, split value and statuses; no created path contains 'test'
     or ':'
  P  R4-1 item 8, the official profile end to end (run by main only, after the cases above; the mutation
     harness runs the cases, never this section):
     P1 the official driver (scripts/run_stats.py --mode official --synthetic-inputs, as
        smoke_stats_official_fullsize runs it, at B = SMOKE_B) writes the statistics artifact of the
        world's 37, read back under Policy.OFFICIAL, its status nonofficial (the TEST-manifest binding
        is unbound: driver.py:540-545)
     P2 LREP's CLI (scripts/report_extensions.py) on it stops at the first refusal, stub_reader_official
        (named with file:line; B-4's reader swap), and writes nothing
     P3 the CLI's body (report_extensions.run) with a stand-in for TMAN's reader writes the report with
        every part, Part 1's and B-5's, under Policy.OFFICIAL (B = SMOKE_B); its profile nonofficial
     P4 the DT rows of that report: read, label, the arms' non-official pairs, the markdown
     P5 the official computations of that report (dedup (ii)-(iii)) under Policy.OFFICIAL with the 37:
        the kept rows' Am5Pair on the seven clean contrasts, NI's official block beside the subset's
        (DL-24 R3-03)
     P6-P8 one mixed run per refusal family, nothing written: a DT artifact that is official
        (descriptive_input_invalid), an official-capable slot's provisional artifact (artifact_unreadable),
        a DT run given to an official computation (descriptive_input_in_official)
     P9 the report on N's E5:s42:sel: parent_rule at heldout_inputs._parent_of, named with its file:line
        for B-4, nothing written
     P10 the path walk again, after P1-P9 created theirs (DL-24 R2-02)

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_lrep_official_profile.py
      (scripts/smoke_lrep_mutations.py alone runs it with --mutations <out.json>: the mutations its cases
      kill, on its own official world, their results written to <out.json>)
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import lrep_fixtures as LF                                               # noqa: E402
import stats_fixtures as F                                               # noqa: E402
from src.eval.adapters import DATASET_NAME                               # noqa: E402
from src.eval.evaluate import Condition                                  # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import dedup_subset as DS                                 # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import report_extensions as RX                            # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY                          # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

#: R4-6(ii): the official world's dataset.split, a JSON value only (never a path, file name, id or key).
TEST_SPLIT = "test"
CTX = types.SimpleNamespace(world=None, paths={}, copies={}, refs={}, inv=None, e2e={})
DT_SLOT = "E7-R1:s42:list42"                     # the DT slot the refusal cases edit (role arm, int8_ptq)
DT_PRECISION = "int8_ptq"
N_INCLUDED = LF.OFFICIAL_N - LF.OFFICIAL_K      # the rows a DT run keeps under the official AM-5 rule
#: The end-to-end section's B (the driver's tasks and B-5's intervals): at n = 1561 the production B of
#: dedup's 32 recomputed tasks would take hours, so the report is nonofficial on this count too.
SMOKE_B = 200
JOBS = max(1, min(4, os.cpu_count() or 1))
STATS_RUN_ID = "lrep-official-0001"
#: R4-1 item 6, typed from the ruling: what every row built from a DT run prints.
DT_LABEL = "descriptive TEST; provisional artifact (the evaluator scores this stage descriptively only)"


def tman_reader(path, *, expect_sha256=None):
    """A stand-in for TMAN's reader (src.tman.validate.read_inventory, which B-4 swaps in): the stub's
    checks without its stub mark, so that open_inventory reads the export under Policy.OFFICIAL."""
    return H.read_inventory(path, expect_sha256=expect_sha256)


def inventory(path=None) -> H.Inventory:
    if path is None and CTX.inv is not None:
        return CTX.inv
    inv = H.open_inventory(path or CTX.world.inventory_path, policy=Policy.OFFICIAL, reader=tman_reader)
    if path is None:
        CTX.inv = inv
    return inv


def reference(stage: str, policy: Policy = Policy.OFFICIAL):
    """The family slot of `stage` at seed 42 read as the 37's input is: driver.load_input under `policy`."""
    key = (stage, policy)
    if key not in CTX.refs:
        w = CTX.world.art(stage, 42)
        CTX.refs[key] = D.load_input(stage, D.CLEAN, w.rel, CTX.world.input_root, policy)
    return CTX.refs[key]


def stats_stand_in():
    """What load_slot reads of the verified statistics artifact: its policy and its clean inputs (the 37's
    E4 and E7 are the references of the DT slots)."""
    return types.SimpleNamespace(policy=Policy.OFFICIAL, clean=reference)


def variant(name, edit):
    if name not in CTX.paths:
        export, binding = copy.deepcopy(CTX.world.export), copy.deepcopy(CTX.world.binding)
        edit(export, binding)
        CTX.paths[name] = LF.write_inventory(CTX.world.fresh(name), export, binding, CTX.world.dedup)
    return CTX.paths[name]


def point_record(binding, slot, precision, copy_w, **fields):
    r = next(r for r in binding["artifacts"]
             if (r["slot"], r["precision"], r["scope"], r.get("cell")) == (slot, precision, "clean", None))
    r["repo_relative_path"], r["files_sha256"] = copy_w.rel, dict(copy_w.files_sha256)
    r.update(fields)


def pointed(name, slot=DT_SLOT, precision=DT_PRECISION, **fields):
    """An export whose clean record of (slot, precision) names the copy `name`."""
    return inventory(variant(name, lambda export, binding: point_record(
        binding, slot, precision, CTX.copies[name], **fields)))


def stage_of(slot: str) -> str:
    return next(e["stage"] for e in CTX.world.export["entries"] if e["slot"] == slot)


def dt_read(slot=DT_SLOT, inv=None, ref=None):
    stage = stage_of(slot)
    return H.read_descriptive_test(inv or inventory(), slot=slot, stage=stage,
                                   precision=H.stage_spec(stage)[2], input_root=CTX.world.input_root,
                                   reference=ref if ref is not None else reference(H.stage_spec(stage)[0]))


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:300]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def refused_at(step: str, fn, needle: str = ""):
    """descriptive_input_invalid at step `step` of the DT read (its detail names the step), with `needle` in
    the detail."""
    try:
        fn()
    except H.LrepRefusal as e:
        ok = e.code == "descriptive_input_invalid" and f": ({step}) {H.DT_STEPS[step]}" in e.detail
        return ok and needle in e.detail, f"{e.code}: {e.detail[:400]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected descriptive_input_invalid at ({step}))"


# --------------------------------------------------------------------------------------------------
# copies
# --------------------------------------------------------------------------------------------------
def rows_copy(src: LF.Written, name: str, fn) -> LF.Written:
    """A copy of an artifact under runs/heldout/<name>, its per_image.jsonl rows edited by `fn` (a list of
    dicts, in place) and its MANIFEST rehashed; not in the binding (cases point records at it)."""
    if "test" in name.lower() or ":" in name or "/" in name:
        raise ValueError(name)
    w = CTX.world
    d = w.input_root / "runs" / "heldout" / name
    shutil.copytree(w.input_root / src.rel, d)
    p = d / "per_image.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    fn(rows)
    p.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    F.rehash(d)
    return dataclasses.replace(src, rel=f"runs/heldout/{name}", files_sha256=D.input_file_sha256(d))


def rescored_copy(src: LF.Written, name: str, truth) -> LF.Written:
    """`src`'s model re-scored on `truth` through the writer, as a DT artifact (E7, int8_ptq, provisional,
    the official world's split), with `src`'s checkpoint."""
    w = CTX.world
    rel = f"runs/heldout/{name}"
    if "test" in rel.lower() or ":" in rel:
        raise ValueError(rel)
    scored = F.score(truth, F.predict(truth, quality=0.76, seed=F.int_seed(LF.WORLD_SEED, rel)),
                     Condition(*D.CLEAN))
    F.write_fixture(w.input_root / rel, scored, len(truth.ids), stage="E7", condition=Condition(*D.CLEAN),
                    split=TEST_SPLIT, status="provisional", repo_root=w.input_root, run_id=f"lrep-{name}",
                    precision=DT_PRECISION, checkpoint_sha256=src.checkpoint)
    summary = json.loads((w.input_root / rel / "summary.json").read_text(encoding="utf-8"))
    return dataclasses.replace(src, rel=rel, files_sha256=D.input_file_sha256(w.input_root / rel),
                               run_id=f"lrep-{name}",
                               split_manifest_sha256=summary["dataset"]["split_manifest_sha256"])


def _first_defined(rows) -> dict:
    return next(r for r in rows if r["n_eligible_disease_only"] > 0)


def setup(root: Path, world=None) -> None:
    w = world or LF.build_world(root, stats=False, strata=False, official_split=TEST_SPLIT)
    CTX.world, CTX.paths, CTX.copies, CTX.refs, CTX.inv = w, {}, {}, {}, None
    dt = w.written[(DT_SLOT, DT_PRECISION, "clean", None)]
    edits = {"dt_official": lambda s: {**s, "run": {**s["run"], "artifact_status": "official"}},
             "dt_smoke": lambda s: {**s, "run": {**s["run"], "artifact_status": "smoke"}},
             "dt_val": lambda s: {**s, "dataset": {**s["dataset"], "split": "val"}},
             "dt_random_init": lambda s: {**s, "run": {**s["run"], "random_init": True}},
             "dt_protocol": lambda s: {**s, "metric_protocol": "plantseg-metric/0.0.0"},
             "dt_metric": lambda s: {**s, "run": {**s["run"], "metric_impl_sha256": LF.syn_sha("metric")}}}
    for name, fn in edits.items():
        CTX.copies[name] = LF.edited_copy(w, dt, f"e7r1_s42_{name[3:]}", fn)

    def undefined(rows):
        r = _first_defined(rows)
        r.update(disease_only_miou=None, disease_only_miou_status="evaluation_error")

    def clean_id(rows):
        _first_defined(rows)["clean_image_id"] = "img_99999"
    CTX.copies["dt_undefined"] = rows_copy(dt, "e7r1_s42_undefined", undefined)
    CTX.copies["dt_clean_id"] = rows_copy(dt, "e7r1_s42_cleanid", clean_id)
    CTX.copies["dt_1560"] = rescored_copy(dt, "e7r1_s42_rows1560", F.make_truth(
        LF.OFFICIAL_N - 1, LF.OFFICIAL_K, F.int_seed(LF.WORLD_SEED, "truth")))
    CTX.copies["dt_other_am5"] = rescored_copy(dt, "e7r1_s42_otheram5", F.make_truth(
        LF.OFFICIAL_N, LF.OFFICIAL_K, F.int_seed(LF.WORLD_SEED, "another truth")))
    e3r1 = w.written[("E3-R1:s42:lambda0.5:alpha25:80k", "fp32", "clean", None)]
    CTX.copies["e3r1_provisional"] = LF.edited_copy(
        w, e3r1, "e3r1_s42_provisional",
        lambda s: {**s, "run": {**s["run"], "artifact_status": "provisional"}})
    # (e): E7's family artifact copied and read officially, then changed on disk
    src = w.art("E7", 42)
    d = w.input_root / "runs" / "heldout" / "e7_s42_refcopy"
    shutil.copytree(w.input_root / src.rel, d)
    CTX.copies["ref_changed"] = D.load_input("E7", D.CLEAN, "runs/heldout/e7_s42_refcopy", w.input_root,
                                             Policy.OFFICIAL)
    F.edit_summary(d, lambda s: {**s, "run": {**s["run"], "run_id": "lrep-e7-s42-refcopy-changed"}})
    # (e), DL-24 R3-04: another copy read officially, then its summary changed WITHOUT a rehash, so that the
    # re-load itself refuses (driver.load_input: its payload hashes differ from its MANIFEST.sha256)
    d = w.input_root / "runs" / "heldout" / "e7_s42_refunhashed"
    shutil.copytree(w.input_root / src.rel, d)
    CTX.copies["ref_unhashed"] = D.load_input("E7", D.CLEAN, "runs/heldout/e7_s42_refunhashed", w.input_root,
                                              Policy.OFFICIAL)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    s["run"]["run_id"] = "lrep-e7-s42-refunhashed-changed"
    (d / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------------------
# G: the DT set and its plumbing
# --------------------------------------------------------------------------------------------------
def g_dt_set():
    """R4-1 item 2: slot_read over the catalogue's 49 scored slots (role, stage) gives exactly the nine DT
    slots, from their role and their stage's precision; E7-E2 (int8_ptq, quant_descriptive), E3-X (arm,
    fp32) and the planned E4 and E7 (family, per_seed) read official."""
    scored = {s: v for s, v in LF.CATALOGUE.items() if v[4] is not None}
    got = sorted(s for s, v in scored.items() if H.slot_read({"stage": v[0], "role": list(v[3])})
                 == H.READ_DESCRIPTIVE_TEST)
    return (len(scored) == 49 and got == sorted(LF.DT_SLOTS) and len(got) == 9
            and H.slot_read({"stage": "E7-E2", "role": ["quant_descriptive"]}) == H.READ_OFFICIAL
            and H.slot_read({"stage": "E7", "role": ["per_seed"]}) == H.READ_OFFICIAL
            and H.slot_read({"stage": "E3-R1", "role": ["arm"]}) == H.READ_OFFICIAL), str(got)


def _callers(name: str) -> set:
    """{(module file, enclosing function)} of every call of `name` in LREP's src/stats files."""
    from src.stats.report_extensions import LREP_CODE_PATHS
    out = set()
    for rel in LREP_CODE_PATHS:
        if not rel.startswith("src/stats/"):
            continue
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                f = getattr(node, "func", None) if isinstance(node, ast.Call) else None
                if ((isinstance(f, ast.Attribute) and f.attr == name)
                        or (isinstance(f, ast.Name) and f.id == name)):
                    out.add((Path(rel).name, fn.name))
    return out


def g_plumbing():
    """R4-1 items 3-5: the DT read is reached through load_slot only, which only the arm tables' rows and
    am6's coverage call; dt_pair only by the arm rows; the official-input guard where dedup's (ii) and (iii)
    and am6's seed-42 rows take their inputs (load_bound itself is not the DT read: its other callers read
    official slots)."""
    got = {n: _callers(n)
           for n in ("read_descriptive_test", "load_slot", "dt_pair", "require_official_inputs")}
    want = {"read_descriptive_test": {("heldout_inputs.py", "load_slot")},
            "load_slot": {("arm_tables.py", "arm_row"), ("am6_tables.py", "coverage_items")},
            "dt_pair": {("arm_tables.py", "arm_row")},
            "require_official_inputs": {("dedup_subset.py", "build_dedup"),
                                        ("am6_tables.py", "build_am6_tables")}}
    return got == want, str(got)


# --------------------------------------------------------------------------------------------------
# H: the happy paths
# --------------------------------------------------------------------------------------------------
def h_dt_slots():
    """Each of the world's nine DT slots, read by load_slot under Policy.OFFICIAL, reads descriptive_test
    through the DT read: its artifact provisional on the official world's split, its run non-official
    (Policy.NONOFFICIAL_SMOKE), its rows the official AM-5 rule's (1561 - 7, none excluded, the dropped ids
    the reference's), and its label DT_LABEL."""
    inv, bad = inventory(), []
    for slot in LF.DT_SLOTS:
        stage = stage_of(slot)
        ba = H.load_slot(stats_stand_in(), inv, slot=slot, stage=stage, precision=H.stage_spec(stage)[2],
                         input_root=CTX.world.input_root)
        ref = reference(H.stage_spec(stage)[0]).run
        ident = ba.run.identity
        if not (ba.read == H.READ_DESCRIPTIVE_TEST and ba.read_label == H.DT_LABEL
                and ba.run.policy is Policy.NONOFFICIAL_SMOKE and ident.artifact_status == "provisional"
                and ident.split == TEST_SPLIT and len(ba.run.records) == N_INCLUDED
                and not any(r.am5_excluded for r in ba.run.records)
                and set(ba.run.stats.manifest_ids) - set(ba.run.image_ids) == set(ref.am5.excluded_ids)):
            bad.append(slot)
    return not bad, str(bad)


def h_pair():
    """(e): E7-R1's pair with E7 through dt_pair is aligned under Policy.NONOFFICIAL_SMOKE (the pair marks
    itself non-official), on 1561 - 7 images, with no Am5Pair (a non-strict pair carries none), and its
    delta is E7-R1's disease-only value minus E7's, image by image, from the rows."""
    ba = dt_read()
    ref = reference("E7")
    pv = H.dt_pair(ref, ba, input_root=CTX.world.input_root, metric=METRIC_DISEASE_ONLY)
    b, c = ref.run.by_id(), ba.run.by_id()
    want = np.asarray([c[i].disease_only_miou - b[i].disease_only_miou for i in sorted(b)], dtype=np.float64)
    return (pv.policy is Policy.NONOFFICIAL_SMOKE and pv.n == N_INCLUDED and pv.am5 is None
            and pv.alignment_status == "ok" and pv.image_ids == tuple(sorted(b))
            and np.array_equal(pv.delta, want)), f"{pv.policy} n={pv.n} {pv.alignment_status}"


def h_official_slots():
    """R4-1 item 4: every other clean record of the world outside the planned cells (E3-X, E6-X, AM-23's, the
    teachers, A, F, G, the three controls, E6-KD, the default candidates: 19) reads official through
    load_bound under Policy.OFFICIAL: its artifact official, its run official, no label."""
    planned = {e["slot"] for e in CTX.world.export["entries"] if set(e["role"]) & {"family", "per_seed"}}
    inv, bad, n = inventory(), [], 0
    for (slot, precision, scope, cell), w in sorted(CTX.world.written.items()):
        if scope != "clean" or slot in LF.DT_SLOTS or slot in planned:
            continue
        n += 1
        ba = H.load_slot(stats_stand_in(), inv, slot=slot, stage=w.stage, precision=precision,
                         input_root=CTX.world.input_root)
        if not (ba.read == H.READ_OFFICIAL and ba.read_label is None and ba.run.policy is Policy.OFFICIAL
                and ba.run.identity.artifact_status == "official"):
            bad.append(slot)
    return not bad and n == 19, f"{bad} of {n}"


def h_arm_row():
    """(e) in the arm rows under Policy.OFFICIAL (arm_tables.arm_row, the statistics artifact stood in by
    its policy and its clean inputs): E7-R1's row reads descriptive_test through the DT read, prints DT_LABEL
    and pairs with E7 under Policy.NONOFFICIAL_SMOKE (dt_pair) on 1561 - 7 images; E3-R1's reads official
    and pairs with E3 under Policy.OFFICIAL on as many."""
    from src.stats import arm_tables as AT
    inv, st = inventory(), stats_stand_in()
    teacher = {"x": "R1", "val_x": None, "ece_x": None, "go": None, "record": None, "note": None}
    got = {}
    for stage, slot in (("E7", DT_SLOT), ("E3", "E3-R1:s42:lambda0.5:alpha25:80k")):
        row, _ = AT.arm_row(st, inv, "R1", stage, slot, reference(stage), input_root=CTX.world.input_root,
                            B=SMOKE_B, scale=AT.SEED_NOISE_RECORD["band"], teacher=teacher)
        p = row["per_image"]
        got[stage] = (row["read"], row["read_label"], p["pair_policy"], p["n_paired"])
    return (got == {"E7": ("descriptive_test", DT_LABEL, "nonofficial_smoke", N_INCLUDED),
                    "E3": ("official", None, "official", N_INCLUDED)}), str(got)


def h_coverage():
    """am6's coverage under Policy.OFFICIAL (am6_tables.coverage_items, the stand-in): the 28 clean records
    outside the planned cells are read, the nine DT slots' through the DT read (descriptive_test, DT_LABEL,
    their runs non-official) and the 19 others' official."""
    from src.stats import am6_tables as AM6
    inv = inventory()
    planned = set(H.planned_cells(inv).slots.values())
    items, loaded = AM6.coverage_items(stats_stand_in(), inv, planned, input_root=CTX.world.input_root)
    dt = sorted(ident["slot"] for ident, _, _ in items if ident["read"] == H.READ_DESCRIPTIVE_TEST)
    policies = {k: ba.run.policy for k, ba in loaded.items()}
    return (len(items) == 28 and dt == sorted(LF.DT_SLOTS)
            and all(ident["read_label"] == (DT_LABEL if ident["slot"] in LF.DT_SLOTS else None)
                    for ident, _, _ in items)
            and sum(p is Policy.NONOFFICIAL_SMOKE for p in policies.values()) == 9
            and sum(p is Policy.OFFICIAL for p in policies.values()) == 19), str(dt)


# --------------------------------------------------------------------------------------------------
# A-F: one case per step of the DT read
# --------------------------------------------------------------------------------------------------
def a_loader():
    """(a): a DT artifact its loader refuses (its summary's metric_protocol is not the frozen one; files,
    record and manifest consistent) is refused at (a), naming load_bound's code."""
    return refused_at("a", lambda: dt_read(inv=pointed("dt_protocol")), "artifact_unreadable")


def b_status():
    """(b): a DT artifact that is official, smoke, on the VAL split, random_init, or of 1560 rows (each the
    only defect, every earlier gate passing) is refused at (b)."""
    out = [refused_at("b", lambda n=n: dt_read(inv=pointed(n)), what)
           for n, what in (("dt_official", "artifact_status 'official'"),
                           ("dt_smoke", "artifact_status 'smoke'"), ("dt_val", "split 'val'"),
                           ("dt_random_init", "random_init True"))]
    out.append(refused_at("b", lambda: dt_read(inv=pointed(
        "dt_1560", split_manifest_sha256=CTX.copies["dt_1560"].split_manifest_sha256)), "1560"))
    return all(ok for ok, _ in out), str([d for ok, d in out if not ok])[:600]


def c_undefined():
    """(c): a DT artifact with an undefined disease-only value on a row AM-5 does not exclude (status
    evaluation_error) is refused at (c) by ingest.apply_am5_policy under Policy.OFFICIAL."""
    return refused_at("c", lambda: dt_read(inv=pointed("dt_undefined")), "OFFICIAL analysis forbids")


def d_reference():
    """(d): the reference must be the 37's E7 input read under Policy.OFFICIAL: E7 read under
    Policy.NONOFFICIAL_SMOKE, the 37's E3 (official, another stage), or E7's family slot as a BoundArtifact
    are each refused at (d)."""
    inv = inventory()
    e7 = CTX.world.art("E7", 42)
    bound = H.load_bound(inv, slot=e7.slot, stage="E7", precision=DT_PRECISION, scope="clean",
                         input_root=CTX.world.input_root, policy=Policy.OFFICIAL)
    out = [refused_at("d", lambda r=r: dt_read(ref=r), "the reference is not")
           for r in (reference("E7", Policy.NONOFFICIAL_SMOKE), reference("E3"), bound)]
    return all(ok for ok, _ in out), str([d for ok, d in out if not ok])[:600]


def d_no_reference():
    """(d): a DT-classed slot whose evaluator stage has no family slot among the 37 -- A with role
    calibration_sensitivity (entry and model row), contradicting TMAN's catalogue -- is refused at (d) by
    load_slot, coded, before any reference is looked up."""
    def edit(export, binding):
        for x in (*export["entries"], *export["models"]):
            if x["slot"] == "A:s42:alpha25:80k":
                x["role"] = ["calibration_sensitivity"]
    inv = inventory(variant("a_calibration", edit))
    return refused_at("d", lambda: H.load_slot(stats_stand_in(), inv, slot="A:s42:alpha25:80k", stage="A",
                                               precision=H.stage_spec("A")[2],
                                               input_root=CTX.world.input_root),
                      "its evaluator stage 'A' has no family slot among the 37")


def d_metric():
    """(d): a DT artifact scored by another metric implementation is refused at (d) by
    align.assert_compatible under Policy.OFFICIAL."""
    return refused_at("d", lambda: dt_read(inv=pointed("dt_metric")), "metric_impl_sha256")


def d_am5():
    """(d): a DT artifact on the same images whose AM-5 set differs from E7's is refused at (d) by
    ingest.am5_pair."""
    return refused_at("d", lambda: dt_read(inv=pointed("dt_other_am5")), "AM-5 pair")


def e_reference_changed():
    """(e): a reference whose files changed after its official read is refused at (e), its re-load under
    Policy.NONOFFICIAL_SMOKE compared with the official read's files."""
    ba = dt_read()
    return refused_at("e", lambda: H.dt_pair(CTX.copies["ref_changed"], ba, input_root=CTX.world.input_root,
                                             metric=METRIC_DISEASE_ONLY), "files differ")


def e_reference_unloadable():
    """(e), DL-24 R3-04: a reference whose summary changed after its official read without a rehash: its
    re-load under Policy.NONOFFICIAL_SMOKE refuses (driver.load_input's DriverRefusal), named at (e)."""
    ba = dt_read()
    return refused_at("e", lambda: H.dt_pair(CTX.copies["ref_unhashed"], ba, input_root=CTX.world.input_root,
                                             metric=METRIC_DISEASE_ONLY),
                      "DriverRefusal: input 'runs/heldout/e7_s42_refunhashed' (E7, clean): payload hashes "
                      "differ from its MANIFEST.sha256")


def e_clean_id():
    """(e): a DT artifact whose clean_image_id differs from E7's on one image passes (a)-(d) and is refused
    at (e) by align.align_runs."""
    ba = dt_read(inv=pointed("dt_clean_id"))
    return refused_at("e", lambda: H.dt_pair(reference("E7"), ba, input_root=CTX.world.input_root,
                                             metric=METRIC_DISEASE_ONLY), "clean_image_id")


def f_record():
    """(f): a DT record whose summary.json sha256 differs from the artifact's is refused at (f), naming
    load_bound's artifact_record_mismatch."""
    def edit(export, binding):
        r = next(r for r in binding["artifacts"] if (r["slot"], r["precision"], r["scope"]) ==
                 (DT_SLOT, DT_PRECISION, "clean"))
        r["files_sha256"]["summary.json"] = LF.syn_sha("not the summary")
    return refused_at("f", lambda: dt_read(inv=inventory(variant("dt_record_files", edit))),
                      "artifact_record_mismatch")


# --------------------------------------------------------------------------------------------------
# K: no downgrade path (item 4)
# --------------------------------------------------------------------------------------------------
def k_no_downgrade():
    """R4-1 item 4: E3-R1 (official-capable) with a provisional TEST artifact is refused under
    Policy.OFFICIAL by load_slot (artifact_unreadable, the OFFICIAL row rule), never read as descriptive."""
    inv = pointed("e3r1_provisional", slot="E3-R1:s42:lambda0.5:alpha25:80k", precision="fp32")
    return refused("artifact_unreadable", lambda: H.load_slot(
        stats_stand_in(), inv, slot="E3-R1:s42:lambda0.5:alpha25:80k", stage="E3-R1", precision="fp32",
        input_root=CTX.world.input_root))


# --------------------------------------------------------------------------------------------------
# N: a difference from TMAN's catalogue, named for B-4 (reported, not coded)
# --------------------------------------------------------------------------------------------------
def catalogue_parent(export, binding):
    """E5:s42:sel as TMAN's catalogue and entry block give it (configs/tday_catalogue.json and
    src/tman/schema.py:461-464 at ee5593e): catalogue.parent null, the pilot deciding; the derived entry's
    own parent (schema.py:392) names the pilot run of record. The world names that run in catalogue.parent
    (scripts/lrep_fixtures.py: E5_PILOT_OF_RECORD)."""
    e = next(e for e in export["entries"] if e["slot"] == "E5:s42:sel")
    e["catalogue"]["parent"] = None
    e["parent"] = LF.E5_PILOT_OF_RECORD


def n_pilot_parent():
    """DL-24 R2-01, named for B-4: on E5:s42:sel as the catalogue gives it, LREP's R-12 reads
    catalogue.parent only (heldout_inputs._parent_of), so planned_cells, which every part's report runs,
    refuses parent_rule naming the slot."""
    inv = inventory(variant("e5_catalogue_parent", catalogue_parent))
    ok, detail = refused("parent_rule", lambda: H.planned_cells(inv))
    return ok and "'E5:s42:sel': catalogue.parent None is not a slot of the export" in detail, detail


# --------------------------------------------------------------------------------------------------
# X: the world
# --------------------------------------------------------------------------------------------------
def x_world():
    """The official world: every artifact's dataset name is the fixture's, never the evaluator's PlantSeg
    name; its split the official value; 1561 rows with 7 AM-5 images; its status official but the nine DT
    slots' (provisional); the five extra catalogue slots are written."""
    bad = []
    for (slot, precision, scope, cell), w in CTX.world.written.items():
        s = json.loads((CTX.world.input_root / w.rel / "summary.json").read_text(encoding="utf-8"))
        want = "provisional" if slot in LF.DT_SLOTS else "official"
        if not (s["dataset"]["name"] == F.FIXTURE_DATASET_NAME != DATASET_NAME
                and s["dataset"]["split"] == TEST_SPLIT and s["dataset"]["expected_rows"] == LF.OFFICIAL_N
                and s["run"]["artifact_status"] == want and s["am5"]["excluded_count"] == LF.OFFICIAL_K):
            bad.append((slot, precision, scope, cell))
    extra = {x[0] for x in LF.OFFICIAL_EXTRA_SLOTS} <= {k[0] for k in CTX.world.written}
    return not bad and extra and len(CTX.world.written) > 100, str(bad[:3])


def x_no_test_path():
    """LRB20 with ':' (R4-6): no created path or file name contains 'test' or ':'."""
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower() or ":" in n]
    return not bad, str(bad[:3])


# --------------------------------------------------------------------------------------------------
# P: the official profile end to end (R4-1 item 8; main only)
# --------------------------------------------------------------------------------------------------
def cli(script: str, *args, timeout=None):
    """scripts/<script> in a fresh interpreter: (exit code, stdout, stderr)."""
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / script), *map(str, args)],
                       cwd=str(REPO), capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return p.returncode, p.stdout, p.stderr


def raise_site(fn, code: str) -> str:
    """repo-relative file:line of the `raise refusal("<code>"` site in `fn`."""
    lines, first = inspect.getsourcelines(fn)
    i = next(k for k, line in enumerate(lines) if f'raise refusal("{code}"' in line)
    return f"{Path(inspect.getsourcefile(fn)).resolve().relative_to(REPO).as_posix()}:{first + i}"


def e2e_stats():
    """The official world's statistics artifact (written once by P1)."""
    return CTX.e2e["stats_path"]


def rx_run(name, inventory=None):
    """The CLI's body under Policy.OFFICIAL with the stand-in reader, into a new out-dir <root>/<name>."""
    w = CTX.world
    from report_extensions import default_mde_entry                     # the CLI's own default
    return RX.run(stats_artifact=e2e_stats(), inventory=inventory or w.inventory_path, out_dir=w.root / name,
                  repo_root=w.input_root, synthetic=True, strata=w.strata_path, mde_entry=default_mde_entry(),
                  parts=None, reader=tman_reader, B=SMOKE_B)


def p_driver():
    """P1: scripts/run_stats.py --mode official --confirm-official-test-analysis --synthetic-inputs on the
    world's 37 (B = SMOKE_B, --jobs as fullsize's): exit 0; the artifact verifies with its inputs and reads
    under Policy.OFFICIAL; its status nonofficial, warnings synthetic_input_data and
    bootstrap_replicates_nonproduction (the TEST-manifest binding is unbound, driver.py:540-545)."""
    w = CTX.world
    LF.write_strata(w)
    lst = F.write_input_list(w.root / "inputs.json", None, entries=w.entries37)
    t0 = time.time()
    code, so, se = cli("run_stats.py", "--mode", "official", "--confirm-official-test-analysis",
                       "--synthetic-inputs", "--inputs", lst, "--out-dir", w.root / "stats", "--run-id",
                       STATS_RUN_ID, "--repo-root", w.input_root, "--B", SMOKE_B, "--jobs", JOBS)
    secs = time.time() - t0
    if code != 0:
        return False, f"exit {code}: {se[-500:]}"
    lines = dict(ln.split(": ", 1) for ln in so.splitlines()
                 if ln.startswith(("ARTIFACT: ", "UNMET_OFFICIAL: ")))
    CTX.e2e["stats_path"] = Path(lines["ARTIFACT"])
    st = H.load_stats_artifact(e2e_stats(), w.input_root)
    return (st.policy is Policy.OFFICIAL and st.profile == "nonofficial"
            and set(st.family["warnings"]) == {A.W_SYNTHETIC, A.W_B}
            and st.views.am5_excluded_count == LF.OFFICIAL_K), \
        f"{secs:.0f} s; warnings {st.family['warnings']}; unmet {lines.get('UNMET_OFFICIAL')}"


def p_cli_first_refusal():
    """P2: LREP's CLI on the official artifact (--synthetic-inputs --repo-root) reads the export with LREP's
    stub, which Policy.OFFICIAL refuses: exit 2, stub_reader_official, nothing written. The first refusal
    of the official profile, named for B-4 (the reader swap) with its file:line."""
    w = CTX.world
    out = w.root / "rx_cli"
    code, so, se = cli("report_extensions.py", "--stats-artifact", e2e_stats(), "--inventory",
                       w.inventory_path, "--out-dir", out, "--repo-root", w.input_root, "--synthetic-inputs",
                       "--strata", w.strata_path)
    site = raise_site(H.open_inventory, "stub_reader_official")
    CTX.e2e["first_refusal"] = site
    return (code == 2 and "RESULT: REFUSED (stub_reader_official); nothing written" in so
            and "REFUSED stub_reader_official" in se and not out.exists()), \
        f"exit {code}; first refusal at {site}"


def p_report():
    """P3: the CLI's body (report_extensions.run) with the stand-in for TMAN's reader writes the report with
    every part (seed, strata, defaults, am6, fqoff, mde, arms, dedup, quant), the statistics artifact read
    under Policy.OFFICIAL and the export by the stand-in; its profile nonofficial, unmet naming the
    statistics artifact and the non-production B (and whatever else the checkout's state gives)."""
    t0 = time.time()
    out, doc = rx_run("rx_official")
    CTX.e2e["doc"], CTX.e2e["out"] = doc, out
    unmet = {u[0] for u in doc["unmet_official_conditions"]}
    files = sorted(p.name for p in out.iterdir())
    return ([p["part"] for p in doc["parts"]] == list(RX.PARTS) and doc["profile"] == "nonofficial"
            and doc["statistics_artifact"]["policy"] == "official"
            and doc["statistics_artifact"]["split"] == TEST_SPLIT
            and doc["inventory"]["reader"].endswith("tman_reader")
            and {"statistics_artifact_nonofficial", "bootstrap_replicates_nonproduction"} <= unmet
            and files == sorted(RX.OUT_FILES)), f"{time.time() - t0:.0f} s; unmet {sorted(unmet)}"


def _part(doc, name):
    return next(p for p in doc["parts"] if p["part"] == name)


def p_dt_rows():
    """P4: in that report the arms' E7-X rows read descriptive_test, print DT_LABEL and pair under
    Policy.NONOFFICIAL_SMOKE on 1561 - 7 images; E3-X and E6-X read official and pair under Policy.OFFICIAL;
    am6's coverage reads the nine DT slots descriptive_test and the others official; dedup's (i) rows of
    E7-X read descriptive_test; the markdown prints DT_LABEL once per DT row (3 + 9 + 3)."""
    doc = CTX.e2e["doc"]
    arms = _part(doc, "arms")["rows"]
    am6 = [r for r in _part(doc, "am6")["rows"] if "read" in r]
    dedup = _part(doc, "dedup")["models"]
    arm_ok = all((r["read"], r["read_label"], r["per_image"]["pair_policy"], r["per_image"]["n_paired"]) ==
                 (("descriptive_test", DT_LABEL, "nonofficial_smoke", N_INCLUDED) if r["stage"] == "E7" else
                  ("official", None, "official", N_INCLUDED)) for r in arms)
    am6_dt = sorted(r["slot"] for r in am6 if r["read"] == "descriptive_test")
    dedup_dt = sorted(m["slot"] for m in dedup if m["read"] == "descriptive_test")
    md = (CTX.e2e["out"] / "report_extensions.md").read_text(encoding="utf-8")
    return (arm_ok and len(arms) == 8 and am6_dt == sorted(LF.DT_SLOTS) and len(am6) == 28
            and all(r["read_label"] == (DT_LABEL if r["slot"] in LF.DT_SLOTS else None) for r in am6)
            and dedup_dt == ["E7-R1:s42:list42", "E7-R2:s42:list42", "E7-flip:s42:list42"]
            and md.count(DT_LABEL) == 15), f"am6 DT {am6_dt}; dedup DT {dedup_dt}; md {md.count(DT_LABEL)}"


def p_official_computations():
    """P5: dedup's (ii) and (iii), the eight contrasts and NI, ran on the 37 under Policy.OFFICIAL: each of
    the seven clean contrasts carries the Am5Pair of the kept rows (a strict pair: n_excluded_am5 the AM-5
    images among the kept ids, n_included the kept rows less those, equal to its n_paired; DL-24 R3-03),
    every contrast pairs some images, and NI reports the official block (passed, lower_bound,
    observed_delta, primary_margin) beside the subset's. That every value is finite or None is P3's: the
    report is written with artifact.canonical_json, which refuses NaN and infinity."""
    p = _part(CTX.e2e["doc"], "dedup")
    kept = set(CTX.world.dedup["kept_ids"])
    e1 = reference("E1").run
    excl = sum(1 for i in e1.am5.excluded_ids if i in kept)
    incl = sum(1 for i in e1.stats.manifest_ids if i in kept) - excl
    clean = [c for c in p["contrasts"] if c["comparison_id"] != D.ROBUSTNESS_ID]
    am5 = [c["subset"]["am5"] for c in clean]
    ni = p["noninferiority"]
    return (len(p["contrasts"]) == 8 and len(clean) == 7
            and all(a is not None and (a["n_excluded_am5"], a["n_included"]) == (excl, incl) for a in am5)
            and all(c["subset"]["n_paired"] == incl for c in clean)
            and all(c["subset"]["n_paired"] > 0 for c in p["contrasts"])
            and set(ni["official"]) == {"passed", "lower_bound", "observed_delta", "primary_margin"}
            and ni["official"]["passed"] in (True, False) and ni["subset"]["passed"] in (True, False)), \
        (f"kept AM-5 {excl}, included {incl}; "
         + str([(c["comparison_id"], c["subset"]["n_paired"]) for c in p["contrasts"]])[:300])


def p_mixed_dt():
    """P6: the report on an export whose E7-R1 record names an official DT artifact (everything else as P3):
    descriptive_input_invalid at (b), nothing written."""
    out = CTX.world.root / "rx_mixed_dt"
    ok, detail = refused("descriptive_input_invalid", lambda: rx_run(
        "rx_mixed_dt", variant("dt_official", lambda e, b: point_record(b, DT_SLOT, DT_PRECISION,
                                                                        CTX.copies["dt_official"]))))
    return ok and ": (b) " in detail and not out.exists(), detail


def p_mixed_provisional():
    """P7: the report on an export whose E3-R1 record names a provisional artifact: artifact_unreadable
    (item 4: no downgrade path), nothing written."""
    out = CTX.world.root / "rx_mixed_provisional"
    slot = "E3-R1:s42:lambda0.5:alpha25:80k"
    ok, detail = refused("artifact_unreadable", lambda: rx_run(
        "rx_mixed_provisional", variant("e3r1_provisional", lambda e, b: point_record(
            b, slot, "fp32", CTX.copies["e3r1_provisional"]))))
    return ok and not out.exists(), detail


class _Injected(H.StatsArtifact):
    """The official statistics artifact (its family, views and the 37 unchanged), except that its clean
    input of one stage is another run: the injection R4-1 item 5 guards against."""
    def clean(self, stage):
        return CTX.e2e.get("injected", {}).get(stage) or H.StatsArtifact.clean(self, stage)


def p_mixed_injected():
    """P8: the duplicate-free part on the official statistics artifact with E7's clean input replaced by
    E7-R1's DT run (read by load_slot under Policy.OFFICIAL, through the DT read): refused where the eight
    contrasts and NI take their inputs (descriptive_input_in_official)."""
    w = CTX.world
    st = H.load_stats_artifact(e2e_stats(), w.input_root)
    inv = inventory()
    ba = H.load_slot(st, inv, slot=DT_SLOT, stage="E7-R1", precision=DT_PRECISION, input_root=w.input_root)
    CTX.e2e["injected"] = {"E7": ba.loaded}
    fake = _Injected(**{f.name: getattr(st, f.name) for f in dataclasses.fields(st)})
    ok, detail = refused("descriptive_input_in_official", lambda: DS.build_dedup(
        fake, inv, H.planned_cells(inv), input_root=w.input_root, B=SMOKE_B))
    return (ok and ba.read == H.READ_DESCRIPTIVE_TEST and ba.run.policy is Policy.NONOFFICIAL_SMOKE
            and "['E7 clean']" in detail), detail


def p_mixed_pilot_parent():
    """P9 (DL-24 R2-01): the report on an export that differs from the world's only in E5:s42:sel, as the
    catalogue gives it (catalogue.parent null): parent_rule at heldout_inputs._parent_of, named for B-4
    with its file:line (on TMAN's export, the refusal after the reader swap); nothing written."""
    out = CTX.world.root / "rx_pilot_parent"
    site = raise_site(H._parent_of, "parent_rule")
    CTX.e2e["pilot_parent"] = site
    ok, detail = refused("parent_rule", lambda: rx_run("rx_pilot_parent",
                                                       variant("e5_catalogue_parent", catalogue_parent)))
    return ok and "'E5:s42:sel': catalogue.parent None" in detail and not out.exists(), f"at {site}; {detail}"


def p_paths():
    """P10 (DL-24 R2-02): X2's walk again after P1-P9 created theirs (the strata, inputs.json, the
    statistics artifact and its temporary sibling, the reports' out-dirs): no created path or file name
    contains 'test' or ':'."""
    return x_no_test_path()


def profile_cases():
    return [("P1", "the official driver writes the 37's statistics artifact (Policy.OFFICIAL, nonofficial)",
             p_driver),
            ("P2", "LREP's CLI stops at its first refusal: stub_reader_official (B-4), nothing written",
             p_cli_first_refusal),
            ("P3", "the CLI's body with a stand-in reader writes every part under Policy.OFFICIAL", p_report),
            ("P4", "the report's DT rows: read, label, non-official pairs, the markdown", p_dt_rows),
            ("P5", "dedup's eight contrasts and NI ran on the 37 under Policy.OFFICIAL",
             p_official_computations),
            ("P6", "mixed run: an official DT artifact -> descriptive_input_invalid, nothing written",
             p_mixed_dt),
            ("P7", "mixed run: an official-capable slot's provisional artifact -> artifact_unreadable",
             p_mixed_provisional),
            ("P8", "mixed run: a DT run given to an official computation -> descriptive_input_in_official",
             p_mixed_injected),
            ("P9", "mixed run: E5:s42:sel as the catalogue gives it -> parent_rule (R2-01, for B-4)",
             p_mixed_pilot_parent),
            ("P10", "no path created by P1-P9 contains 'test' or ':' (LRB20; R4-6)", p_paths)]


def cases():
    return [("G1", "R4-1 item 2: slot_read over the catalogue's 49 scored slots gives exactly the nine",
             g_dt_set),
            ("G2", "R4-1 items 3-5: who calls the DT read, load_slot, dt_pair and the official-input guard",
             g_plumbing),
            ("H1", "the nine DT slots read descriptive_test through the DT read, their runs non-official",
             h_dt_slots),
            ("H2", "(e): a DT pair with E7 is aligned under NONOFFICIAL_SMOKE, its delta the rows'", h_pair),
            ("H3", "item 4: every other B-5 slot reads official under Policy.OFFICIAL", h_official_slots),
            ("H4", "(e): the arm rows under Policy.OFFICIAL: E7-R1's DT pair non-official, E3-R1's official",
             h_arm_row),
            ("H5", "am6's coverage under Policy.OFFICIAL: the nine DT slots through the DT read, 19 official",
             h_coverage),
            ("A1", "descriptive_input_invalid (a): an artifact its loader refuses", a_loader),
            ("B1", "descriptive_input_invalid (b): official, smoke, VAL, random_init, 1560 rows", b_status),
            ("C1", "descriptive_input_invalid (c): an undefined value AM-5 does not exclude", c_undefined),
            ("D1", "descriptive_input_invalid (d): a reference that is not the 37's E7 read officially",
             d_reference),
            ("D2", "descriptive_input_invalid (d): another metric implementation (assert_compatible)",
             d_metric),
            ("D3", "descriptive_input_invalid (d): another AM-5 set (am5_pair)", d_am5),
            ("D4", "descriptive_input_invalid (d): a DT-classed slot with no family reference among the 37",
             d_no_reference),
            ("E1", "descriptive_input_invalid (e): the reference's files changed after its official read",
             e_reference_changed),
            ("E2", "descriptive_input_invalid (e): a clean_image_id that differs (align_runs)", e_clean_id),
            ("E3", "descriptive_input_invalid (e): the reference's re-load refuses (driver's DriverRefusal)",
             e_reference_unloadable),
            ("F1", "descriptive_input_invalid (f): a record whose files differ from the artifact's",
             f_record),
            ("K1", "item 4: an official-capable slot's provisional artifact is refused (artifact_unreadable)",
             k_no_downgrade),
            ("N1", "R2-01 (B-4): E5:s42:sel as the catalogue gives it (parent null) -> parent_rule (R-12)",
             n_pilot_parent),
            ("X1", "the official world: dataset name, split, rows, AM-5 count and statuses", x_world),
            ("X2", "no created path contains 'test' or ':' (LRB20; R4-6)", x_no_test_path)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_official_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_lrep_official_profile: temporary root {root} (left in place)", flush=True)
    setup(root)
    print(f"  official world written ({len(CTX.world.written)} artifacts, n = {CTX.world.n}): "
          f"{time.time() - t0:.0f} s", flush=True)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:600]}")
    print("  -- the official profile end to end (R4-1 item 8)", flush=True)
    for cid, desc, fn in profile_cases():
        t1 = time.time()
        ok, detail = run_case(fn)
        results.append((cid, desc, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc} ({time.time() - t1:.0f} s)")
        print(f"         {detail[:600]}", flush=True)
    good = sum(1 for r in results if r[2])
    verdict = "LREP OFFICIAL PROFILE OK" if good == len(results) else "LREP OFFICIAL PROFILE FAILED"
    print(f"\nRESULT: {verdict} ({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; no table here is a thesis result.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--mutations"]:
        # DL-24 R3-01: the mutation harness's official:* mutations, in this smoke's own process (R4-6(ii));
        # the harness imports this file as smoke_lrep_official_profile, which is this process's __main__
        sys.modules["smoke_lrep_official_profile"] = sys.modules["__main__"]
        import smoke_lrep_mutations
        raise SystemExit(smoke_lrep_mutations.official_mutations(Path(sys.argv[2])))
    raise SystemExit(main())
