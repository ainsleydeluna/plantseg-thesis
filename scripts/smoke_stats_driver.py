#!/usr/bin/env python3
"""Statistics driver core smoke (lane L-STATS-OFFICIAL; NONOFFICIAL, synthetic only).

Checks src/stats/driver.py against synthetic evaluation artifacts that scripts/stats_fixtures.py
writes through the unmodified evaluator writer into temporary clean fixture repositories outside the
working tree. No dataset, checkpoint, model or GPU; nothing is written into this repository.

  T  the one comparison table: its keys are bootstrap.ALL_COMPARISON_IDS in that order (the eight
     CANONICAL_COMPARISON_IDS, then NON_FAMILY_COMPARISON_IDS), and its stages and metrics
  I  the 37-input inventory of STATISTICAL_ANALYSIS_CONTRACT section 12.4.6
  L  the input-list file and its refusals
  D  the loader: P5 hash snapshots, both halves of the canvas guard (EVALUATION_CONTRACT 11(a)),
     identity and condition refusals
  O  every observed value assembled from the inputs, against the fixture arrays
  S  a strict policy with AM-5 exclusions (k > 0): per-image n = N - k, dataset-level n = N
  J  --jobs: all 35 tasks bitwise identical at jobs 1 and 4 (P38), on this machine only (P12)

Run:  python -B scripts/smoke_stats_driver.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import stats_fixtures as F                                               # noqa: E402
from src.eval.artifacts import PRECISIONS                                # noqa: E402
from src.stats import bootstrap as BS                                    # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, METRIC_MIOU_C           # noqa: E402
from src.stats.artifact import INPUT_ARTIFACT_FIELDS                     # noqa: E402
from src.stats.corruption_protocol import official_corruption_grid       # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402
from src.stats.noninferiority import pooled_miou_from_totals             # noqa: E402
from src.stats.robustness import INFERENTIAL_SEVERITIES                  # noqa: E402
from src.stats.tests import CANONICAL_COMPARISON_IDS, holm_family        # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SMALL_N = 30
STRICT_N, STRICT_K = 846, 5
JOBS_B = 40


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def refused(fn, *a, contains: str = "", **kw) -> tuple[bool, str]:
    """(True, message) iff fn raises DriverRefusal whose message contains `contains`."""
    try:
        fn(*a, **kw)
    except D.DriverRefusal as e:
        return (contains in str(e)), str(e)
    except Exception as e:                                               # noqa: BLE001
        return False, f"wrong exception {type(e).__name__}: {e}"
    return False, "no refusal"


def write_list(path: Path, doc) -> Path:
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return path


def list_doc(entries) -> dict:
    return {"schema": D.INPUT_LIST_SCHEMA,
            "inputs": [{"stage": st, "condition": {"type": c[0], "name": c[1], "severity": c[2]},
                        "path": rel} for st, c, rel in entries]}


def clone(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# --------------------------------------------------------------------------------------------------
def table_checks() -> None:
    keys = tuple(D.COMPARISON_TABLE)
    check("T1 COMPARISON_TABLE keys == bootstrap.ALL_COMPARISON_IDS, in order",
          keys == BS.ALL_COMPARISON_IDS, keys)
    check("T2 ... i.e. the eight CANONICAL_COMPARISON_IDS, then NON_FAMILY_COMPARISON_IDS",
          keys == tuple(CANONICAL_COMPARISON_IDS) + BS.NON_FAMILY_COMPARISON_IDS)
    stages_ok = all((b, c) == (f"E{cid[-4]}", f"E{cid[-1]}") for cid, (b, c, _)
                    in D.COMPARISON_TABLE.items())
    check("T3 every entry's stages are the ones its comparison_id names", stages_ok)
    metrics = {cid: m for cid, (_, _, m) in D.COMPARISON_TABLE.items()}
    check("T4 metrics: disease-only for the seven clean tests and descriptive, mIoU-C for "
          "robustness, the pooled estimand for NI",
          all(metrics[c] == METRIC_DISEASE_ONLY for c in D.CLEAN_COMPARISON_IDS)
          and metrics[BS.DESCRIPTIVE_E1_E3] == METRIC_DISEASE_ONLY
          and metrics[D.ROBUSTNESS_ID] == METRIC_MIOU_C
          and metrics[BS.NONINFERIORITY_E3_E6] == D.METRIC_POOLED, metrics)
    check("T5 STAGE_PRECISION covers E1..E7 with evaluator precisions",
          tuple(D.STAGE_PRECISION) == D.STAGES
          and all(v in PRECISIONS for v in D.STAGE_PRECISION.values()), D.STAGE_PRECISION)


def inventory_checks() -> None:
    inv = D.official_inventory()
    grid = official_corruption_grid()
    want = [(s, D.CLEAN) for s in D.STAGES] + [
        (s, ("corruption", n, v)) for s in ("E1", "E6") for n in grid.names
        for v in INFERENTIAL_SEVERITIES]
    check("I1 inventory = 7 clean stages, then E1 and E6 x 5 corruptions x severities 1-3 (37)",
          len(inv) == 37 and list(inv) == want, len(inv))
    check("I2 corruption names come from configs/corruption_protocol.json",
          {c[1] for _, c in inv[7:]} == set(grid.names), grid.names)


def input_list_checks(work: Path, fs) -> None:
    lst = F.write_input_list(work / "inputs.json", fs)
    entries = D.load_input_list(lst)
    check("L1 input list round-trips: the 37 entries in file order",
          list(entries) == [(s, tuple(c), r) for s, c, r in fs.entries])
    base = list_doc(fs.entries)
    cases = []
    cases.append(("L2 wrong schema string", dict(base, schema="plantseg-stats-inputs/0.9.0"),
                  "schema"))
    cases.append(("L3 extra top-level key", dict(base, note="x"), "exactly schema and inputs"))
    raw = json.dumps(base)
    cases.append(("L4 duplicate key", raw[:-1] + ', "schema": "' + D.INPUT_LIST_SCHEMA + '"}',
                  "duplicate key"))
    cases.append(("L5 non-finite constant", raw.replace('"severity": 1', '"severity": NaN', 1),
                  "non-finite"))
    bad = json.loads(raw)
    bad["inputs"][0]["extra"] = 1
    cases.append(("L6 entry with an extra key", bad, "exactly stage, condition and path"))
    bad = json.loads(raw)
    bad["inputs"][0]["stage"] = "E8"
    cases.append(("L7 unknown stage", bad, "unknown stage"))
    for tag, val in (("L8 absolute path", "/etc/passwd"), ("L9 '..' component", "runs/../x"),
                     ("L10 backslash", "runs\\val\\e1_clean"), ("L11 drive letter", "C:/runs/x")):
        bad = json.loads(raw)
        bad["inputs"][0]["path"] = val
        cases.append((tag, bad, "repository-relative"))
    bad = json.loads(raw)
    bad["inputs"][1]["path"] = bad["inputs"][0]["path"].upper()
    cases.append(("L12 the same path twice (case-folded)", bad, "listed twice"))
    bad = json.loads(raw)
    bad["inputs"].pop()
    cases.append(("L13 36 inputs (one identity missing)", bad, "missing 1"))
    bad = json.loads(raw)
    bad["inputs"][1]["stage"] = "E1"
    cases.append(("L14 an identity repeated", bad, "repeated 1"))
    bad = json.loads(raw)
    bad["inputs"][-1]["condition"]["severity"] = 4
    cases.append(("L15 a severity-4 cell (not in the inventory)", bad, "unexpected 1"))
    bad = json.loads(raw)
    bad["inputs"][-1]["condition"]["severity"] = [1]
    cases.append(("L16 a non-integer severity", bad, "neither clean"))
    bad = json.loads(raw)
    bad["inputs"][0]["condition"]["name"] = "gaussian_noise"
    cases.append(("L17 a clean condition with a name", bad, "neither clean"))
    for tag, doc, needle in cases:
        ok, msg = refused(D.load_input_list, write_list(work / "bad.json", doc), contains=needle)
        check(f"{tag} -> refused by name", ok, msg)
    ok, msg = refused(D.load_input_list, work / "does_not_exist.json", contains="does not exist")
    check("L18 a missing input-list file -> refused", ok, msg)


def loader_checks(work: Path, fs) -> dict:
    pol = Policy.NONOFFICIAL_SMOKE
    entries = D.load_input_list(F.write_input_list(work / "inputs.json", fs))
    inputs = D.load_inputs(entries, fs.input_root, pol)
    check("D1 all 37 inputs load under NONOFFICIAL_SMOKE", len(inputs) == 37)
    agree = all(D.snapshot_agrees(li.file_sha256, D.manifest_digests(li.directory))
                and li.file_sha256 == D.input_file_sha256(li.directory) for li in inputs.values())
    check("D2 each input's four-file snapshot equals its MANIFEST lines (P5)", agree)
    rec = D.provenance_record(inputs[("E1", D.CLEAN)])
    check("D3 provenance record carries the ten section 12.4.6 fields in the frozen order",
          tuple(rec) == INPUT_ARTIFACT_FIELDS, tuple(rec))

    e1 = fs.path("E1")
    rel_e1 = fs.rel("E1")
    # canvas guard, half 1: a non-canvas preprocess protocol (manifest kept consistent)
    alt = work / "canvas_pp"
    F.edit_summary(clone(e1, alt / rel_e1), lambda s: (s["dataset"].__setitem__(
        "preprocess_protocol", "letterbox_512/1.0.0"), s)[1])
    ok, msg = refused(D.load_input, "E1", D.CLEAN, rel_e1, alt, pol, contains="only canvas")
    check("D4 canvas guard: preprocess_protocol != core_preprocess/1.0.0 -> refused by name",
          ok and rel_e1 in msg, msg)
    # canvas guard, half 2: a summary.protocol block on a canvas-labelled artifact
    alt = work / "canvas_block"
    F.edit_summary(clone(e1, alt / rel_e1), lambda s: dict(s, protocol={"name": "upstream"}))
    ok, msg = refused(D.load_input, "E1", D.CLEAN, rel_e1, alt, pol, contains="only canvas")
    check("D5 canvas guard: a summary.protocol block -> refused by name", ok and rel_e1 in msg, msg)
    # a payload edited without re-hashing
    alt = work / "tamper"
    d = clone(e1, alt / rel_e1)
    with open(d / "per_image.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    ok, msg = refused(D.load_input, "E1", D.CLEAN, rel_e1, alt, pol, contains="MANIFEST")
    check("D6 a payload that differs from its MANIFEST line -> refused", ok, msg)
    # a third manifest line
    alt = work / "manifest3"
    d = clone(e1, alt / rel_e1)
    with open(d / "MANIFEST.sha256", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{'0' * 64}  extra.txt\n")
    ok, msg = refused(D.load_input, "E1", D.CLEAN, rel_e1, alt, pol, contains="MANIFEST")
    check("D7 a MANIFEST.sha256 with a third line -> refused", ok, msg)
    # declared identity != artifact identity
    ok, msg = refused(D.load_input, "E2", D.CLEAN, fs.rel("E3"), fs.input_root, pol,
                      contains="not the declared identity")
    check("D8 an entry whose path holds another stage -> refused", ok, msg)
    grid = official_corruption_grid()
    c1, c2 = ("corruption", grid.names[0], 1), ("corruption", grid.names[0], 2)
    ok, msg = refused(D.load_input, "E1", c1, fs.rel("E1", c2), fs.input_root, pol,
                      contains="condition mismatch")
    check("D9 an entry whose path holds another corruption cell -> refused", ok, msg)
    ok, msg = refused(D.load_input, "E1", D.CLEAN, "runs/nowhere", fs.input_root, pol,
                      contains="no artifact directory")
    check("D10 a path with no artifact directory -> refused", ok, msg)
    # P5: a file that changes while load_run reads it
    alt = work / "race"
    d = clone(e1, alt / rel_e1)
    real = D.load_run

    def racing_load_run(*a, **kw):
        run = real(*a, **kw)
        with open(d / "summary.json", "a", encoding="utf-8") as fh:
            fh.write(" ")
        return run
    D.load_run = racing_load_run
    try:
        ok, msg = refused(D.load_input, "E1", D.CLEAN, rel_e1, alt, pol, contains="changed while")
    finally:
        D.load_run = real
    check("D11 a file changed during load_run -> refused (snapshot 2 != snapshot 1)", ok, msg)
    ok, msg = refused(D.load_inputs, list(entries)[:-1], fs.input_root, pol, contains="missing 1")
    check("D12 load_inputs refuses an incomplete inventory before reading", ok, msg)
    # a directory inside the input root that resolves outside it
    root = work / "escape_root"
    (root / "runs").mkdir(parents=True)
    try:
        os.symlink(e1, root / "runs" / "e1_clean", target_is_directory=True)
    except (OSError, NotImplementedError) as e:
        check("D13 a symlinked directory resolving outside the input root -> refused (not run: "
              f"no symlink support here: {type(e).__name__})", True)
    else:
        ok, msg = refused(D.load_input, "E1", D.CLEAN, "runs/e1_clean", root, pol,
                          contains="outside the input root")
        check("D13 a symlinked directory resolving outside the input root -> refused", ok, msg)
    return inputs


def observed_checks(fs, inputs) -> D.Observed:
    pol = Policy.NONOFFICIAL_SMOKE
    obs = D.assemble_observed(inputs, pol)
    order = [r.comparison_id for r in obs.results]
    check("O1 eight results in canonical order, stages and metric from the table",
          order == list(CANONICAL_COMPARISON_IDS)
          and all((r.baseline_stage, r.candidate_stage, r.metric) == D.COMPARISON_TABLE[
              r.comparison_id] for r in obs.results), order)
    check("O2 every comparison pairs all images (k = 0)",
          all(r.n_paired == SMALL_N for r in obs.results) and obs.am5_excluded_count == 0)

    def dv(stage, cond=D.CLEAN):
        return fs.scored[(stage, cond)].disease_values

    clean_ok = []
    for cid in D.CLEAN_COMPARISON_IDS:
        b, c, _ = D.COMPARISON_TABLE[cid]
        clean_ok.append(obs.result(cid).shifts.mean_delta == float(np.mean(dv(c) - dv(b))))
    check("O3 each clean comparison's mean_delta equals the fixture arrays exactly", all(clean_ok),
          clean_ok)
    grid = official_corruption_grid()

    def miou_c(stage):
        return np.stack([np.stack([dv(stage, ("corruption", n, s)) for s in INFERENTIAL_SEVERITIES]
                                  ).mean(axis=0) for n in grid.names]).mean(axis=0)
    rob = obs.result(D.ROBUSTNESS_ID).shifts.mean_delta
    check("O4 robustness mean_delta equals the nested mean of the 30 fixture cells exactly",
          rob == float(np.mean(miou_c("E6") - miou_c("E1"))), rob)
    check("O5 descriptive E1->E3 mean_delta equals the fixture arrays exactly",
          obs.descriptive.mean_delta == float(np.mean(dv("E3") - dv("E1")))
          and obs.descriptive.n_paired == SMALL_N)

    def pooled(stage):
        tp, gt, pr = fs.scored[(stage, D.CLEAN)].totals
        return pooled_miou_from_totals(tp, gt, pr)
    pooled_ok = [float(obs.pooled[cid].observed()) == pooled(D.COMPARISON_TABLE[cid][1])
                 - pooled(D.COMPARISON_TABLE[cid][0])
                 for cid in (*D.CLEAN_COMPARISON_IDS, BS.NONINFERIORITY_E3_E6)]
    check("O6 the eight pooled deltas (seven clean + NI E6 - E3) equal the fixture totals exactly",
          all(pooled_ok), pooled_ok)
    vals = D.observed_values(obs)
    src_ok = (tuple(vals) == BS.FROZEN_TASK_IDS
              and all(vals[f"{cid}__{s}"] == BS.comparison_observed(obs.result(cid), s)
                      for cid in CANONICAL_COMPARISON_IDS for s in BS.SCALAR_STATISTICS)
              and all(vals[f"{BS.DESCRIPTIVE_E1_E3}__{s}"] == obs.descriptive.value(s)
                      for s in BS.SCALAR_STATISTICS))
    check("O7 35 observed values in FROZEN_TASK_IDS order, from their frozen sources", src_ok)
    again = holm_family(obs.results)
    check("O8 the Holm family equals holm_family over the eight results", again == obs.holm)
    return obs


def strict_checks(work: Path) -> None:
    pol = Policy.REHEARSAL
    fs = F.build_inventory(work / "strict", n=STRICT_N, k=STRICT_K, split="val",
                           status="provisional", run_prefix="strict")
    entries = D.load_input_list(F.write_input_list(work / "strict_inputs.json", fs))
    inputs = D.load_inputs(entries, fs.input_root, pol)
    obs = D.assemble_observed(inputs, pol)
    n_inc = STRICT_N - STRICT_K
    check("S1 k read from the inputs, one count and one set across all 37",
          obs.am5_excluded_count == STRICT_K
          and obs.excluded_clean_ids == tuple(sorted(np.array(fs.truth.ids)[fs.truth.am5])))
    check("S2 per-image n = N - k on the eight comparisons and the descriptive block",
          all(r.n_paired == n_inc for r in obs.results) and obs.descriptive.n_paired == n_inc,
          sorted({r.n_paired for r in obs.results}))
    check("S3 mIoU-C covers the clean included images; pooled stages keep all N rows",
          all(len(v.values) == n_inc for v in obs.miou_c.values())
          and all(p.n_images == STRICT_N for p in obs.pooled.values()))
    jobs = {j.spec.task_id: j for j in D.task_jobs(obs, BS.ANALYSIS_ID_OFFICIAL, 3)}
    per_image = D.run_job(jobs["accuracy_e1_e2__mean_delta"])
    dataset = D.run_job(jobs["accuracy_e1_e2__dataset_miou_delta"])
    check("S4 a task's jackknife count comes from its data object: N - k per-image, N pooled (P7)",
          per_image.jackknife.size == n_inc and dataset.jackknife.size == STRICT_N,
          (per_image.jackknife.size, dataset.jackknife.size))

    class _Am5:
        def __init__(self, ids):
            self.excluded_count, self.excluded_clean_ids = len(ids), tuple(ids)

    class _Li:
        def __init__(self, ids):
            self.run = type("R", (), {"am5": _Am5(ids)})()
    ok, msg = refused(D.am5_identity, {"a": _Li(["x"]), "b": _Li(["y"])}, contains="AM-5")
    check("S5 two inputs excluding different images -> refused (one set across the 37)", ok, msg)


def jobs_checks(obs: D.Observed) -> None:
    one = D.run_tasks(obs, BS.ANALYSIS_ID_SMOKE, JOBS_B, jobs=1)
    four = D.run_tasks(obs, BS.ANALYSIS_ID_SMOKE, JOBS_B, jobs=4)
    same = [a.task_id == b.task_id and a.observed == b.observed
            and np.array_equal(a.replicates, b.replicates) and np.array_equal(a.jackknife, b.jackknife)
            and a.bca == b.bca and a.seed == b.seed for a, b in zip(one, four)]
    check("J1 all 35 tasks identical at --jobs 1 and 4 (replicates, jackknife, observed, seed, "
          "z0, acceleration, bounds, warnings; one machine)",
          len(one) == len(four) == 35 and all(same), f"{sum(same)}/35")
    check("J2 tasks come back in FROZEN_TASK_IDS order",
          tuple(r.task_id for r in four) == BS.FROZEN_TASK_IDS)
    bad = [refused(D.run_tasks, obs, BS.ANALYSIS_ID_SMOKE, JOBS_B, jobs=j, contains="--jobs")[0]
           for j in (0, True, 1.5)]
    check("J3 --jobs 0, True and 1.5 -> refused", all(bad), bad)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="smoke_stats_driver_"))
    try:
        table_checks()
        inventory_checks()
        fs = F.build_inventory(work / "smoke", n=SMALL_N, k=0, split="val", status="smoke")
        input_list_checks(work, fs)
        inputs = loader_checks(work, fs)
        obs = observed_checks(fs, inputs)
        strict_checks(work)
        jobs_checks(obs)
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check("Z1 temporary fixtures removed", not work.exists())

    print()
    for name, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if det and not ok:
            print(f"         {det[:300]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'STATS DRIVER CORE OK' if allok else 'STATS DRIVER CORE FAILED'} "
          f"({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic fixtures only; no thesis result.")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
