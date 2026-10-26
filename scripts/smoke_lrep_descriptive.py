#!/usr/bin/env python3
"""LREP's descriptive bootstrap namespace (lane LREP, B-5: B5-D4 and B5-D2): src/stats/lrep_descriptive.py.

SYNTHETIC ONLY (DL-41): the loop-fidelity cases run on scripts/lrep_fixtures.py's world (its seed-42
inputs), written under a new directory in TMPDIR (left in place); no created path contains "test".

  N  the namespace: the analysis id is outside bootstrap.ANALYSIS_IDS; no contrast id is one of the ten
     official ids; bootstrap.canonical_seed_bytes refuses the id; the seed prefix differs from the
     official one; the no-collision assertion fires for an official id and for malformed ones
  B  the seed bytes of one hand case, typed out; its SHA-256 and 128-bit entropy, pinned; the
     SeedMaterial's fields
  L  loop fidelity: fed derive_seed(ANALYSIS_ID_SMOKE, an official comparison_id, a statistic),
     lrep_descriptive.run_task reproduces run_bootstrap_task's replicates, jackknife and BcaOutcome
     bitwise, for a TWO_SIDED mean_delta task and a ONE_SIDED_LOWER pooled task
  M  B5-D2: the observed value is the A3a primitive's mean_delta, the interval two-sided, and B is
     bootstrap.PRODUCTION_B by default, imported and never typed
  D  determinism; distinct streams per contrast and statistic, none an official stream; the JSON keeps
     the statistic in a field of its own and the banner

Cases are importable (setup / cases) so that scripts/smoke_lrep_mutations.py can run them mutated.

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_lrep_descriptive.py
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
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
from src.stats import bootstrap as BS                                    # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import lrep_descriptive as LD                             # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, align_runs              # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402
from src.stats.noninferiority import PooledStages                        # noqa: E402

CTX = types.SimpleNamespace(world=None, runs={})
#: B5-D4's hand case, typed out from the ruling (never built by the module under test).
HAND_ID, HAND_STAT = "arm__r1__e3", "mean_delta"
HAND_BYTES = (b"plantseg-lrep-descriptive/1.0.0\x00root_seed=42\x00lrep_descriptive_v1\x00arm__r1__e3\x00"
              b"mean_delta")
OFFICIAL_PREFIX = b"plantseg-stats/1.0.0\x00"
#: Pinned from the bytes above with printf | sha256sum and bc, outside Python (B5-D4: SHA-256, then the
#: first 16 bytes as a big-endian integer).
HAND_SHA256 = "4efd13806147cdff10ed55029a503f19b2ac83f6ff260de4db84feed9c243268"
HAND_ENTROPY = 104993830313526873160219995199226527513
LOOP_B = 400


def run(stage):
    if stage not in CTX.runs:
        w = CTX.world
        CTX.runs[stage] = D.load_input(stage, D.CLEAN, w.art(stage, 42).rel, w.input_root,
                                       Policy.NONOFFICIAL_SMOKE).run
    return CTX.runs[stage]


def every_contrast_id():
    ids = [LD.arm_contrast_id(x, st) for x in LD.ARM_XS for st in ("e3", "e7")]
    ids += [LD.arm_contrast_id(x, "e6") for x in ("r1", "r2")]
    ids += [LD.dedup_contrast_id(c) for c in (*D.CLEAN_COMPARISON_IDS, D.ROBUSTNESS_ID,
                                                 BS.NONINFERIORITY_E3_E6)]
    ids += [LD.quant_contrast_id(s) for s in ("E6-E2", "E7-E2")]
    return ids


def raises(exc, fn):
    try:
        fn()
    except exc as e:
        return True, f"{type(e).__name__}: {str(e)[:120]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, "no exception"


def same_outcome(a, b):
    return a == b and type(a) is type(b)


# --------------------------------------------------------------------------------------------------
def n_analysis_id():
    return (LD.ANALYSIS_ID == "lrep_descriptive_v1" and LD.ANALYSIS_ID not in BS.ANALYSIS_IDS), LD.ANALYSIS_ID


def n_contrast_ids():
    ids = every_contrast_id()
    official = BS.ALL_COMPARISON_IDS
    return (len(ids) == len(set(ids)) == 19 and not set(ids) & set(official)
            and all("__" in i for i in ids) and not any("__" in c for c in official)
            and len(official) == 10), str(ids)


def n_frozen_refuses():
    ok1, d1 = raises(BS.BootstrapError, lambda: BS.canonical_seed_bytes(LD.ANALYSIS_ID, "accuracy_e1_e2",
                                                                       BS.MEAN_DELTA))
    ok2, d2 = raises(BS.BootstrapError, lambda: BS.canonical_seed_bytes(BS.ANALYSIS_ID_SMOKE, HAND_ID,
                                                                       BS.MEAN_DELTA))
    return ok1 and ok2, f"{d1} | {d2}"


def n_prefix():
    lrep = LD.seed_bytes(HAND_ID, HAND_STAT)
    off = BS.derive_seed(BS.ANALYSIS_ID_OFFICIAL, "accuracy_e1_e2", BS.MEAN_DELTA).canonical_bytes
    return (LD.SEED_PREFIX != OFFICIAL_PREFIX and lrep.startswith(LD.SEED_PREFIX)
            and off.startswith(OFFICIAL_PREFIX) and not lrep.startswith(OFFICIAL_PREFIX)
            and not off.startswith(LD.SEED_PREFIX)), repr(lrep[:40])


def n_collision_assertion():
    checks = [raises(AssertionError, lambda: LD.check_contrast_id("accuracy_e1_e2")),
              raises(AssertionError, lambda: LD.seed_bytes(BS.NONINFERIORITY_E3_E6, BS.DATASET_MIOU_DELTA)),
              raises(AssertionError, lambda: LD.seed_material("robustness_e1_e6", BS.MEAN_DELTA))]
    return all(ok for ok, _ in checks), str([d for _, d in checks])


def n_malformed():
    bad_ids = ["Arm__r1__e3", "arm_r1_e3", "arm___r1", "arm__r1__", "arm__r1__e3 ", "__arm__r1", "arm__r-1"]
    checks = [raises(AssertionError, lambda b=b: LD.seed_bytes(b, BS.MEAN_DELTA)) for b in bad_ids]
    checks.append(raises(AssertionError, lambda: LD.seed_bytes(HAND_ID, "mean")))
    checks.append(raises(AssertionError, lambda: LD.arm_contrast_id("k8", "e3")))
    checks.append(raises(AssertionError, lambda: LD.arm_contrast_id("r1", "e5")))
    return all(ok for ok, _ in checks), str([d for ok, d in checks if not ok])


def b_hand_bytes():
    raw = LD.seed_bytes(HAND_ID, HAND_STAT)
    return raw == HAND_BYTES and not raw.endswith(b"\x00"), repr(raw)


def b_digest():
    s = LD.seed_material(HAND_ID, HAND_STAT)
    digest = hashlib.sha256(HAND_BYTES).digest()
    return (s.seed_sha256 == digest.hex() == HAND_SHA256 and s.entropy_bytes == digest[:16]
            and s.entropy_int == int.from_bytes(digest[:16], "big") == HAND_ENTROPY), s.seed_sha256


def b_fields():
    s = LD.seed_material(HAND_ID, HAND_STAT)
    return (s.root_seed == 42 and s.canonical_bytes == HAND_BYTES and s.seed_input_hex == HAND_BYTES.hex()
            and s.bit_generator == "PCG64" and s.numpy_version == np.__version__
            and s.seed_input_display == ("plantseg-lrep-descriptive/1.0.0\\0root_seed=42\\0lrep_descriptive_v1"
                                         "\\0arm__r1__e3\\0mean_delta")), s.seed_input_display


def l_mean_delta():
    """TWO_SIDED mean_delta: accuracy_e1_e2's per-image disease-only vector on the world's seed-42 E1/E2."""
    pv = align_runs(run("E1"), run("E2"), policy=Policy.NONOFFICIAL_SMOKE, metric=METRIC_DISEASE_ONLY)
    d = np.asarray(pv.delta, dtype=np.float64)
    task = BS.TASK_BY_ID["accuracy_e1_e2__mean_delta"]
    observed = float(np.mean(d))
    rf, jf = BS.scalar_callables(BS.MEAN_DELTA, d)
    ref = BS.run_bootstrap_task(task, BS.ANALYSIS_ID_SMOKE, d.size, LOOP_B, observed, rf, jf)
    seed = BS.derive_seed(BS.ANALYSIS_ID_SMOKE, "accuracy_e1_e2", BS.MEAN_DELTA)
    rf2, jf2 = BS.scalar_callables(BS.MEAN_DELTA, d)
    reps, jack, bca = LD.run_task(seed, BS.TWO_SIDED, d.size, LOOP_B, observed, rf2, jf2)
    return (np.array_equal(reps, ref.replicates) and np.array_equal(jack, ref.jackknife)
            and same_outcome(bca, ref.bca) and bca.interval_type == BS.TWO_SIDED), str(bca)[:200]


def l_pooled():
    """ONE_SIDED_LOWER pooled: noninferiority_e3_e6 on the world's seed-42 E3/E6."""
    ps = PooledStages.from_runs(run("E3"), run("E6"))
    task = BS.TASK_BY_ID["noninferiority_e3_e6__dataset_miou_delta"]
    ref = BS.run_bootstrap_task(task, BS.ANALYSIS_ID_SMOKE, ps.n_images, LOOP_B, ps.observed(),
                                ps.replicate, ps.jackknife)
    seed = BS.derive_seed(BS.ANALYSIS_ID_SMOKE, BS.NONINFERIORITY_E3_E6, BS.DATASET_MIOU_DELTA)
    reps, jack, bca = LD.run_task(seed, BS.ONE_SIDED_LOWER, ps.n_images, LOOP_B, ps.observed(),
                                  ps.replicate, ps.jackknife)
    return (np.array_equal(reps, ref.replicates) and np.array_equal(jack, ref.jackknife)
            and same_outcome(bca, ref.bca) and len(bca.bounds) == 1), str(bca)[:200]


def m_observed():
    """B5-D2: the observed mean delta is the A3a primitive's (bootstrap.descriptive_scalars'), the
    interval TWO_SIDED, the statistic mean_delta, the replicates scalar_callables' under this
    namespace's seed."""
    pv = align_runs(run("E3"), run("E7"), policy=Policy.NONOFFICIAL_SMOKE, metric=METRIC_DISEASE_ONLY)
    d = np.asarray(pv.delta, dtype=np.float64)
    r = LD.mean_delta_task(HAND_ID, d, B=LOOP_B)
    ref_obs = BS.descriptive_scalars(d, metric=METRIC_DISEASE_ONLY, policy="nonofficial_smoke").mean_delta
    rf, jf = BS.scalar_callables(BS.MEAN_DELTA, d)
    reps, jack, bca = LD.run_task(LD.seed_material(HAND_ID, BS.MEAN_DELTA), BS.TWO_SIDED, d.size, LOOP_B,
                                  ref_obs, rf, jf)
    return (r.observed == ref_obs and r.interval_type == BS.TWO_SIDED and r.statistic_name == BS.MEAN_DELTA
            and np.array_equal(r.replicates, reps) and same_outcome(r.bca, bca)
            and len(r.bca.bounds) == 2), f"{r.observed} {ref_obs}"


def m_default_b():
    """B is bootstrap.PRODUCTION_B by default, imported (the default is that object) and never typed:
    no integer literal 10000 appears in the module."""
    default = inspect.signature(LD.mean_delta_task).parameters["B"].default
    src = (REPO / "src" / "stats" / "lrep_descriptive.py").read_text(encoding="utf-8")
    typed = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant)
             and isinstance(n.value, int) and not isinstance(n.value, bool) and n.value == BS.PRODUCTION_B]
    return default is BS.PRODUCTION_B and BS.PRODUCTION_B == 10_000 and not typed, str(typed)


def d_deterministic():
    d = np.linspace(-0.2, 0.3, 37)
    a, b = LD.mean_delta_task(HAND_ID, d, B=300), LD.mean_delta_task(HAND_ID, d, B=300)
    return (np.array_equal(a.replicates, b.replicates) and np.array_equal(a.jackknife, b.jackknife)
            and a.bca == b.bca and LD.task_json(a) == LD.task_json(b)), ""


def d_distinct_streams():
    """Every LREP stream is its own (contrast and statistic each change the entropy) and none equals an
    official one (both analysis ids, the ten ids, the four statistics)."""
    lrep = {(c, s): LD.seed_material(c, s).entropy_int for c in every_contrast_id() for s in BS.STATISTIC_NAMES}
    official = {BS.derive_seed(a, c, s).entropy_int for a in BS.ANALYSIS_IDS for c in BS.ALL_COMPARISON_IDS
                for s in BS.STATISTIC_NAMES}
    return len(set(lrep.values())) == len(lrep) and not set(lrep.values()) & official, str(len(lrep))


def d_json():
    """task_json: the contrast id and the statistic are separate fields, no key holds "__", the banner
    reads B5-D4's words, and the JSON round-trips."""
    from src.stats.artifact import canonical_json
    r = LD.mean_delta_task(HAND_ID, np.linspace(-0.1, 0.2, 21), B=200)
    j = LD.task_json(r)
    keys_ok = not any("__" in k for k in j)
    return (keys_ok and j["contrast_id"] == HAND_ID and j["statistic_name"] == BS.MEAN_DELTA
            and j["banner"] == "descriptive; outside protocol 1.0.0" and j["analysis_id"] == LD.ANALYSIS_ID
            and [j["lower_bound"], j["upper_bound"]] == list(r.bca.bounds)
            and json.loads(canonical_json(j)) == j), str(sorted(j))


def setup(root: Path, world=None) -> None:
    CTX.world = world or LF.build_world(root, stats=False, strata=False)
    CTX.runs = {}


def cases():
    return [("N1", "the analysis id is lrep_descriptive_v1, outside bootstrap.ANALYSIS_IDS", n_analysis_id),
            ("N2", "the 19 contrast ids hold '__' and none is one of the ten official ids", n_contrast_ids),
            ("N3", "bootstrap.canonical_seed_bytes refuses the analysis id and a contrast id", n_frozen_refuses),
            ("N4", "the seed prefix differs from b'plantseg-stats/1.0.0\\x00'", n_prefix),
            ("N5", "the no-collision assertion fires for an official comparison id", n_collision_assertion),
            ("N6", "malformed contrast ids and an unknown statistic are refused by assertion", n_malformed),
            ("B1", "the seed bytes of arm__r1__e3 / mean_delta, typed out; no trailing NUL", b_hand_bytes),
            ("B2", "SHA-256 of those bytes, first 16 bytes big-endian, pinned", b_digest),
            ("B3", "the SeedMaterial's fields (PCG64, root seed 42, hex, display)", b_fields),
            ("L1", "loop fidelity: TWO_SIDED mean_delta task, bitwise", l_mean_delta),
            ("L2", "loop fidelity: ONE_SIDED_LOWER pooled task, bitwise", l_pooled),
            ("M1", "B5-D2: the observed mean is the A3a primitive's; two-sided", m_observed),
            ("M2", "B5-D2: B defaults to bootstrap.PRODUCTION_B, never typed", m_default_b),
            ("D1", "two runs give identical replicates, jackknife, interval and JSON", d_deterministic),
            ("D2", "distinct streams per contrast and statistic, none an official stream", d_distinct_streams),
            ("D3", "the JSON keeps the statistic in its own field, with the banner", d_json)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_descriptive_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_lrep_descriptive: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'LREP DESCRIPTIVE OK' if good == len(results) else 'LREP DESCRIPTIVE FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; descriptive; outside protocol 1.0.0.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
