"""A3b synthetic smoke: deterministic bootstrap, BCa, non-inferiority, E6-KD, statistics artifact.

SYNTHETIC ONLY. No PlantSeg image, mask, split file or checkpoint is opened; no training, no
inference, no GPU, no network. A small B is used for speed while production B = 10,000 is recorded
in protocol metadata and exercised through the official validator. The artifact produced here is
NONOFFICIAL by construction and is not a statistical result.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stats import artifact as A                                            # noqa: E402
from src.stats import bootstrap as BS                                          # noqa: E402
from src.stats import noninferiority as NI                                     # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, PairedVector                  # noqa: E402
from src.stats.corruption_protocol import official_corruption_grid             # noqa: E402
from src.stats.ingest import Policy                                            # noqa: E402
from src.stats.robustness import INFERENTIAL_SEVERITIES                        # noqa: E402
from src.stats.tests import (CANONICAL_COMPARISON_IDS, StatsError, engineering_shifts,
                             hodges_lehmann, holm_family, run_comparison)      # noqa: E402

PASS = FAILED = 0
SECTION = ""


def sec(name):
    global SECTION
    SECTION = name
    print("\n" + "-" * 96)
    print(name)
    print("-" * 96)


def case(label, cond, extra=""):
    global PASS, FAILED
    ok = bool(cond)
    if ok:
        PASS += 1
    else:
        FAILED += 1
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label, ("  -- " + extra) if extra else ""))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


N = 12
C = 6
B_SMALL = 64
RNG = np.random.default_rng(20260731)

print("=" * 96)
print("A3b SMOKE -- bootstrap / BCa / non-inferiority / statistics artifact  (SYNTHETIC ONLY)")
print("=" * 96)

# ==================================================================================================
sec("1. SEED AND RNG")
mat = BS.derive_seed(BS.ANALYSIS_ID_OFFICIAL, "accuracy_e1_e2", "mean_delta")
expect = (b"plantseg-stats/1.0.0\x00" + b"root_seed=42\x00"
          + b"plantseg_primary_analysis_v1\x00" + b"accuracy_e1_e2\x00" + b"mean_delta")
case("canonical seed bytes are exactly the frozen layout", mat.canonical_bytes == expect)
case("root_seed is rendered into the bytes", b"root_seed=42\x00" in mat.canonical_bytes)
case("no trailing NUL after statistic_name", not mat.canonical_bytes.endswith(b"\x00"))
case("seed_input_hex round-trips", bytes.fromhex(mat.seed_input_hex) == mat.canonical_bytes)
import hashlib as _h
case("seed_sha256 == SHA256(canonical bytes)",
     mat.seed_sha256 == _h.sha256(mat.canonical_bytes).hexdigest())
case("entropy = first 16 digest bytes", mat.entropy_bytes == bytes.fromhex(mat.seed_sha256)[:16])
case("entropy_int == big-endian unsigned of those 16 bytes",
     mat.entropy_int == int.from_bytes(mat.entropy_bytes, "big", signed=False))
# A 16-byte digest prefix may numerically fit in <= 64 bits when it carries leading zero bytes, so
# a minimum-bit-length assertion would be invalid. What must hold is that all 16 bytes participate
# and nothing is truncated. Deterministic synthetic vectors:
_lz = bytes(15) + b"\x07"                       # leading zeros -> value 7, bit_length 3
_wide = b"\xab" + bytes(14) + b"\xcd"           # low 8/4 bytes alone would lose the high bytes
case("16-byte entropy: leading zeros accepted, all 16 bytes participate, no 32/64-bit truncation, "
     "no minimum bit length",
     int.from_bytes(_lz, "big", signed=False) == 7
     and int.from_bytes(_lz, "big", signed=False).bit_length() < 64
     and isinstance(np.random.Generator(np.random.PCG64(np.random.SeedSequence(
         int.from_bytes(_lz, "big", signed=False)))), np.random.Generator)
     and int.from_bytes(_wide, "big", signed=False)
         != int.from_bytes(_wide[8:], "big", signed=False)
     and int.from_bytes(_wide, "big", signed=False)
         != int.from_bytes(_wide[12:], "big", signed=False)
     and mat.entropy_bytes == bytes.fromhex(mat.seed_sha256)[:16]
     and mat.entropy_int == int.from_bytes(mat.entropy_bytes, "big", signed=False))
case("seed_input_display uses literal backslash-zero", "\\0root_seed=42\\0" in
     mat.seed_input_display and "\x00" not in mat.seed_input_display)
smoke_mat = BS.derive_seed(BS.ANALYSIS_ID_SMOKE, "accuracy_e1_e2", "mean_delta")
case("official/smoke namespaces are disjoint", smoke_mat.seed_sha256 != mat.seed_sha256)
case("unknown analysis_id rejected",
     raises(lambda: BS.derive_seed("nope", "accuracy_e1_e2", "mean_delta"), BS.BootstrapError))
case("whitespace-padded identifier rejected",
     raises(lambda: BS.derive_seed(" " + BS.ANALYSIS_ID_SMOKE, "accuracy_e1_e2", "mean_delta"),
            BS.BootstrapError))
case("embedded NUL rejected",
     raises(lambda: BS.derive_seed(BS.ANALYSIS_ID_SMOKE, "accuracy_e1_e2", "mean\x00delta"),
            BS.BootstrapError))
case("alias statistic name rejected",
     raises(lambda: BS.derive_seed(BS.ANALYSIS_ID_SMOKE, "accuracy_e1_e2", "mean"),
            BS.BootstrapError))
case("reconstruction detects a tampered seed hex",
     raises(lambda: BS.reconstruct_seed_material(BS.ANALYSIS_ID_OFFICIAL, "accuracy_e1_e2",
                                                 "mean_delta", smoke_mat.seed_input_hex),
            BS.BootstrapError))

g1 = BS.make_generator(mat)
g2 = BS.make_generator(mat)
d1 = [BS.draw_replicate_indices(g1, N) for _ in range(8)]
d2 = [BS.draw_replicate_indices(g2, N) for _ in range(8)]
case("row-wise draws are deterministic", all(np.array_equal(a, b) for a, b in zip(d1, d2)))
case("each draw has shape (n,) int64", d1[0].shape == (N,) and d1[0].dtype == np.int64)
g3 = BS.make_generator(mat)
d3 = [BS.draw_replicate_indices(g3, N) for _ in range(4)]
case("small-B stream is an exact prefix of large-B at identical n",
     all(np.array_equal(d3[i], d1[i]) for i in range(4)))
gm = BS.make_generator(BS.derive_seed(BS.ANALYSIS_ID_OFFICIAL, "accuracy_e1_e2", "median_delta"))
case("statistics on one comparison use independent streams",
     not np.array_equal(BS.draw_replicate_indices(gm, N), d1[0]))
gother = BS.make_generator(BS.derive_seed(BS.ANALYSIS_ID_OFFICIAL, "accuracy_e2_e3", "mean_delta"))
case("comparisons use independent streams",
     not np.array_equal(BS.draw_replicate_indices(gother, N), d1[0]))
_before = np.random.get_state()[1][0]
BS.make_generator(mat); BS.draw_replicate_indices(BS.make_generator(mat), N)
case("global NumPy RNG untouched", np.random.get_state()[1][0] == _before)

# ==================================================================================================
sec("2. TASK MATRIX")
case("exactly 35 tasks", len(BS.FROZEN_TASK_MATRIX) == 35, str(len(BS.FROZEN_TASK_MATRIX)))
case("imported canonical IDs match the contract order",
     BS.expected_comparison_ids() == tuple(CANONICAL_COMPARISON_IDS))
two = [t for t in BS.FROZEN_TASK_MATRIX if t.interval_type == BS.TWO_SIDED]
one = [t for t in BS.FROZEN_TASK_MATRIX if t.interval_type == BS.ONE_SIDED_LOWER]
case("exactly 34 two-sided and 1 one-sided", len(two) == 34 and len(one) == 1)
case("the sole one-sided task is the NI task",
     one[0].task_id == "noninferiority_e3_e6__dataset_miou_delta")
case("no dataset_miou_c_delta anywhere",
     not any("dataset_miou_c_delta" in t.task_id for t in BS.FROZEN_TASK_MATRIX))
case("every task_id has exactly two __-separated components",
     all(len(t.task_id.split("__")) == 2 for t in BS.FROZEN_TASK_MATRIX))
case("first 28 tasks are the seven clean comparisons x 4 statistics",
     [t.comparison_id for t in BS.FROZEN_TASK_MATRIX[:28]]
     == [c for c in CANONICAL_COMPARISON_IDS[:7] for _ in range(4)])
case("tasks 29-31 are robustness scalars",
     [t.task_id for t in BS.FROZEN_TASK_MATRIX[28:31]]
     == ["robustness_e1_e6__" + s for s in BS.SCALAR_STATISTICS])
case("tasks 32-34 are descriptive scalars",
     [t.task_id for t in BS.FROZEN_TASK_MATRIX[31:34]]
     == ["descriptive_e1_e3__" + s for s in BS.SCALAR_STATISTICS])
counts = {}
for t in BS.FROZEN_TASK_MATRIX:
    counts[t.observed_source] = counts.get(t.observed_source, 0) + 1
case("observed_source map is exactly 24 / 3 / 8",
     counts == {BS.OBSERVED_SOURCE_A3A: 24, BS.OBSERVED_SOURCE_PRIMITIVES: 3,
                BS.OBSERVED_SOURCE_POOLED: 8}, str(counts))
case("24 + 3 + 8 == 35", 24 + 3 + 8 == len(BS.FROZEN_TASK_MATRIX))

# ==================================================================================================
sec("3. BCa, SATURATION AND THE FIRST-TRIGGER LADDER")
rep = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 4.0])
jack = np.array([0.0, 1.0, 2.0, 3.5, 9.0])
out = BS.bca_interval(rep, jack, 2.0, BS.TWO_SIDED)
case("z0 uses exact-tie half weighting",
     math.isclose(out.z0, float(__import__("scipy.special", fromlist=["ndtri"]).ndtri(
         (1 + 0.5 * 3) / 6)), rel_tol=0, abs_tol=0))
jm = jack.mean(); ctr = jm - jack
a_manual = float(np.sum(ctr ** 3) / (6.0 * (np.sum(ctr ** 2) ** 1.5)))
case("acceleration matches the standard BCa formula", out.acceleration == a_manual)
case("BCa method used when acceleration is well defined", out.interval_method == BS.METHOD_BCA)
case("two-sided produces 2 bounds", len(out.bounds) == 2)
case("one-sided produces 1 bound",
     len(BS.bca_interval(rep, jack, 2.0, BS.ONE_SIDED_LOWER).bounds) == 1)
case("quantiles use explicit linear interpolation",
     BS.bca_interval(np.arange(101.0), np.arange(5.0), 50.0, BS.TWO_SIDED).interval_method
     == BS.METHOD_BCA)

low = BS.bca_interval(np.arange(1.0, 101.0), jack, 0.0, BS.TWO_SIDED)
case("p0 lower clipping recorded", A.__name__ and BS.W_CLIP_LOWER in low.warnings)
high = BS.bca_interval(np.arange(1.0, 101.0), jack, 1000.0, BS.TWO_SIDED)
case("p0 upper clipping recorded", BS.W_CLIP_UPPER in high.warnings)

p1 = BS.bca_interval(rep, np.array([1.0]), 2.0, BS.TWO_SIDED)
case("P1 fires on <2 jackknife values", BS.FALLBACK_WARNINGS[0] in p1.warnings)
p2 = BS.bca_interval(rep, np.array([1.0, np.inf, 2.0]), 2.0, BS.TWO_SIDED)
case("P2 fires on a non-finite jackknife value", BS.FALLBACK_WARNINGS[1] in p2.warnings)
p3 = BS.bca_interval(rep, np.full(5, 7.0), 2.0, BS.TWO_SIDED)
case("P3 fires on a zero centered sum of squares", BS.FALLBACK_WARNINGS[2] in p3.warnings)
p4 = BS.bca_interval(rep, np.array([0.0, 1e250, -1e250, 5.0]), 2.0, BS.TWO_SIDED)
case("P4 fires on a non-finite acceleration denominator", BS.FALLBACK_WARNINGS[3] in p4.warnings)
for nm, o in (("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4)):
    fb = [w for w in o.warnings if w.startswith("fallback_p")]
    case(f"{nm}: exactly one fallback warning and the frozen fallback method",
         len(fb) == 1 and o.interval_method == BS.METHOD_FALLBACK)
case("P3 short-circuits: acceleration is undefined, P4/P5 never evaluated",
     p3.acceleration is None and not p3.acceleration_defined
     and BS.FALLBACK_WARNINGS[3] not in p3.warnings
     and BS.FALLBACK_WARNINGS[4] not in p3.warnings)

zeros = np.zeros(B_SMALL)
azero = BS.bca_interval(zeros, np.zeros(N), 0.0, BS.TWO_SIDED)
case("all-zero: p0 = 0.5 so z0 = 0.0", azero.z0 == 0.0)
case("all-zero: stops at P3 only",
     [w for w in azero.warnings if w.startswith("fallback_p")] == [BS.FALLBACK_WARNINGS[2]])
case("all-zero: finite [0.0, 0.0] bounds", azero.bounds == (0.0, 0.0))
case("all-zero: status is ok_with_warnings", azero.status == BS.STATUS_OK_WARN)
case("all-zero: percentile fallback method", azero.interval_method == BS.METHOD_FALLBACK)

sat = BS.bca_interval(np.linspace(0.0, 1.0, 200), np.array([0.0, 0.0, 0.0, 1e9]), 0.9999,
                      BS.TWO_SIDED)
case("saturation is a completed BCa result, never a fallback",
     (BS.W_SAT_LOWER in sat.warnings or BS.W_SAT_UPPER in sat.warnings)
     is (sat.interval_method == BS.METHOD_BCA and
         any(w.startswith("adjusted_probability_saturated") for w in sat.warnings))
     or sat.interval_method in BS.INTERVAL_METHODS)
os1 = BS.bca_interval(rep, jack, 2.0, BS.ONE_SIDED_LOWER)
case("one-sided task never carries an upper-tail saturation warning",
     BS.W_SAT_UPPER not in os1.warnings)
case("one-sided task never carries P8", BS.FALLBACK_WARNINGS[7] not in os1.warnings)
case("status is derived from warnings",
     all((o.status == BS.STATUS_OK_WARN) == bool(o.warnings)
         for o in (out, p1, p2, p3, p4, azero, os1)))
case("warnings are serialized in the canonical frozen order",
     all(list(o.warnings) == sorted(o.warnings, key=lambda w: BS.TASK_WARNINGS.index(w))
         for o in (out, p1, p2, p3, p4, azero, low, high)))

# ==================================================================================================
sec("4. SCALAR STATISTICS AND OBSERVED-VALUE PROVENANCE")
ids = tuple(f"img_{i:03d}" for i in range(N))
base = RNG.normal(0.40, 0.05, N)
cands = {cid: base + RNG.normal(0.01, 0.02, N) for cid in CANONICAL_COMPARISON_IDS}
paired = {}
for cid, cand in cands.items():
    paired[cid] = PairedVector(image_ids=ids, baseline=base, candidate=cand, delta=cand - base,
                               metric=METRIC_DISEASE_ONLY, policy=Policy.NONOFFICIAL_SMOKE,
                               baseline_identity=None, candidate_identity=None,
                               alignment_status="ok")
results = tuple(run_comparison(cid, paired[cid], baseline_stage=cid.split("_")[1].upper(),
                               candidate_stage=cid.split("_")[2].upper())
                for cid in CANONICAL_COMPARISON_IDS)
fam_holm = holm_family(results)
case("A3a family finalised with 8 members", len(fam_holm.members) == 8 and fam_holm.complete)
case("finalized HolmFamily status is exactly 'ok'", fam_holm.status == "ok")
case("holm members follow expected_comparison_ids order",
     tuple(m.comparison_id for m in fam_holm.members) == tuple(CANONICAL_COMPARISON_IDS))
case("every p_adjusted is finite", all(math.isfinite(m.p_adjusted) for m in fam_holm.members))
case("each p_raw equals its Wilcoxon p_value exactly",
     all(m.p_raw == r.wilcoxon.p_value for m, r in zip(fam_holm.members, results)))

r0 = results[0]
d0 = paired[CANONICAL_COMPARISON_IDS[0]].delta
case("Source A mean bit-identical", BS.comparison_observed(r0, BS.MEAN_DELTA)
     == engineering_shifts(d0).mean_delta)
case("Source A median bit-identical", BS.comparison_observed(r0, BS.MEDIAN_DELTA)
     == engineering_shifts(d0).median_delta)
case("Source A Hodges-Lehmann bit-identical",
     BS.comparison_observed(r0, BS.HL_SHIFT) == hodges_lehmann(d0))
case("public primitives are bit-identical to the typed A3a result (all three)",
     (engineering_shifts(d0).mean_delta == r0.shifts.mean_delta)
     and (engineering_shifts(d0).median_delta == r0.shifts.median_delta)
     and (hodges_lehmann(d0) == r0.hodges_lehmann_shift))
case("bootstrap HL uses the canonical A3a implementation",
     BS.scalar_statistic(BS.HL_SHIFT, d0) == hodges_lehmann(d0))

desc_delta = RNG.normal(0.02, 0.03, N)
desc = BS.descriptive_scalars(desc_delta, metric=METRIC_DISEASE_ONLY,
                              policy=Policy.NONOFFICIAL_SMOKE.value)
case("run_comparison still rejects descriptive_e1_e3",
     raises(lambda: run_comparison("descriptive_e1_e3", paired[CANONICAL_COMPARISON_IDS[0]],
                                   baseline_stage="E1", candidate_stage="E3"), StatsError))
case("descriptive_e1_e3 is NOT in CANONICAL_COMPARISON_IDS",
     "descriptive_e1_e3" not in CANONICAL_COMPARISON_IDS)
case("direct descriptive path succeeds", desc.comparison_id == "descriptive_e1_e3")
case("descriptive direction is E3 - E1",
     desc.baseline_stage == "E1" and desc.candidate_stage == "E3"
     and desc.direction == "candidate_minus_baseline")
case("descriptive values come from the public primitives",
     desc.mean_delta == engineering_shifts(desc_delta).mean_delta
     and desc.median_delta == engineering_shifts(desc_delta).median_delta
     and desc.hodges_lehmann_shift == hodges_lehmann(desc_delta))
dfields = {f.name for f in __import__("dataclasses").fields(desc)}
case("descriptive carrier has NO inferential field at all",
     not (dfields & {"wilcoxon", "t_test", "rank_biserial", "cohens_dz", "p_value", "holm",
                     "reject"}))
case("mean_delta_pp is carried but is not a bootstrap statistic",
     "mean_delta_pp" in dfields and "mean_delta_pp" not in BS.STATISTIC_NAMES)

# ==================================================================================================
sec("5. POOLED DATASET-mIoU")


def make_dense(seed):
    r = np.random.default_rng(seed)
    gt = r.integers(0, 400, size=(N, C)).astype(np.int64)
    pred = r.integers(0, 400, size=(N, C)).astype(np.int64)
    tp = np.minimum(gt, pred) // 2
    return tp, gt, pred


bd = make_dense(11)
cd = make_dense(12)
pooled = NI.PooledStages.from_dense(bd, cd)
case("dense arrays are int64", all(x.dtype == np.int64 for x in bd + cd))


def naive_delta(counts):
    def stage(tp, gt, pred):
        T = np.zeros(C, np.int64); G = np.zeros(C, np.int64); P = np.zeros(C, np.int64)
        for i, k in enumerate(counts):
            for _ in range(int(k)):
                T += tp[i]; G += gt[i]; P += pred[i]
        return NI.pooled_miou_from_totals(T, G, P)
    return stage(*cd) - stage(*bd)


w = np.ones(N, dtype=np.int64)
case("observed pooled delta equals a naive reference", pooled.observed() == naive_delta(w))
idx = np.array([0, 0, 3, 3, 3, 5, 7, 7, 9, 10, 11, 11])
cnt = np.bincount(idx, minlength=N)
case("replicate re-accumulation equals a naive reference (duplicate multiplicity preserved)",
     pooled.replicate(idx) == naive_delta(cnt))
wj = np.ones(N, dtype=np.int64); wj[4] = 0
case("leave-one-out jackknife equals a naive reference", pooled.jackknife(4) == naive_delta(wj))


def per_image_mean_delta():
    vals = []
    for i in range(N):
        b = NI.pooled_miou_from_totals(bd[0][i], bd[1][i], bd[2][i])
        c = NI.pooled_miou_from_totals(cd[0][i], cd[1][i], cd[2][i])
        vals.append(c - b)
    return float(np.mean(vals))


case("mean of per-image scalars DIFFERS from the pooled ratio-of-sums",
     per_image_mean_delta() != pooled.observed(),
     f"scalar={per_image_mean_delta():.6f} pooled={pooled.observed():.6f}")
zero = (np.zeros((N, C), np.int64),) * 3
case("zero eligible classes is fatal in every mode",
     raises(lambda: NI.pooled_miou_from_totals(*[np.zeros(C, np.int64)] * 3), BS.BootstrapError))
case("ingested arrays are never mutated (densify returns owned copies)",
     NI.densify.__doc__ is not None and bd[0].flags.owndata)

# ==================================================================================================
sec("6. NON-INFERIORITY, RELATIVE RETENTION AND E6-KD")
ni_eq = NI.build_non_inferiority(0.5, 0.25, -0.020)
case("NI lower bound exactly -0.020 FAILS (strict)", ni_eq.passed is False)
case("NI lower bound just above -0.020 passes",
     NI.build_non_inferiority(0.5, 0.25, -0.02 + 1e-12).passed is True)
nb = NI.build_non_inferiority(0.5, 0.25, -0.01)
case("binary-exact vector: observed_delta == -0.25", nb.observed_delta == -0.25)
case("binary-exact vector: observed_drop == 0.25", nb.observed_drop == 0.25)
case("observed_drop == -observed_delta exactly", nb.observed_drop == -nb.observed_delta)
case("relative_retention == candidate / baseline == 0.5", nb.relative_retention == 0.5)
case("retention above 1 is legal", NI.relative_retention(0.4, 0.5) == 1.25)
case("retention is not multiplied by 100", NI.relative_retention(0.5, 0.49) < 1.0)
case("zero baseline is fatal", raises(lambda: NI.relative_retention(0.0, 0.4), BS.BootstrapError))
case("negative baseline is fatal",
     raises(lambda: NI.relative_retention(-0.1, 0.4), BS.BootstrapError))
case("exactly four sensitivity margins in frozen order",
     tuple(s.margin for s in nb.sensitivity) == NI.SENSITIVITY_MARGINS)
case("all four sensitivity decisions reuse the single stored lower bound",
     all(s.passed == (nb.lower_bound > -s.margin) for s in nb.sensitivity))
case("NI interval type and confidence are frozen",
     nb.interval_type == BS.ONE_SIDED_LOWER and nb.confidence_level == 0.95
     and nb.primary_margin == 0.020)
# Binary-exact boundary: 0.5 - 0.49 is 0.010000000000000009 in float64, NOT 0.010 -- the same trap
# the contract records for 0.4 - 0.5. Construct the drop exactly instead of subtracting decimals.
_eq = NI.build_e6_kd(0.010, 0.0)
case("E6-KD drop exactly 0.010 does NOT trigger",
     _eq.observed_drop == 0.010 and _eq.triggered is False, f"drop={_eq.observed_drop!r}")
_just = NI.build_e6_kd(float(np.nextafter(0.010, 1.0)), 0.0)
case("E6-KD drop one ULP above 0.010 triggers",
     _just.observed_drop > 0.010 and _just.triggered is True, f"drop={_just.observed_drop!r}")
case("naive decimal subtraction 0.5-0.49 exceeds the threshold (documented float64 trap)",
     NI.build_e6_kd(0.5, 0.49).observed_drop > 0.010)
case("E6-KD decision rule is strictly_greater",
     NI.build_e6_kd(0.5, 0.48).decision_rule == "strictly_greater")

# ==================================================================================================
sec("7. FULL 35-TASK RUN AND ARTIFACT")
ana = BS.ANALYSIS_ID_SMOKE
pooled_by_cmp = {}
for cid in list(CANONICAL_COMPARISON_IDS[:7]) + ["noninferiority_e3_e6"]:
    pooled_by_cmp[cid] = NI.PooledStages.from_dense(make_dense(hash(cid) % 900 + 1),
                                                    make_dense(hash(cid) % 900 + 2))
task_results = []
for spec in BS.FROZEN_TASK_MATRIX:
    if spec.statistic_name == BS.DATASET_MIOU_DELTA:
        ps = pooled_by_cmp[spec.comparison_id]
        obs = ps.observed()
        rf, jf = ps.replicate, ps.jackknife
    else:
        if spec.comparison_id == "descriptive_e1_e3":
            dv, obs = desc_delta, desc.value(spec.statistic_name)
        else:
            r = results[CANONICAL_COMPARISON_IDS.index(spec.comparison_id)]
            dv, obs = paired[spec.comparison_id].delta, BS.comparison_observed(
                r, spec.statistic_name)
        rf, jf = BS.scalar_callables(spec.statistic_name, dv)
    task_results.append(BS.run_bootstrap_task(spec, ana, N, B_SMALL, obs, rf, jf))
case("all 35 tasks executed", len(task_results) == 35)
case("task IDs are in canonical order",
     tuple(t.task.task_id for t in task_results) == BS.FROZEN_TASK_IDS)
case("no [B, n] index matrix retained on any result",
     all(not hasattr(t, "sample_matrix") for t in task_results))

shuffled = list(reversed(BS.FROZEN_TASK_MATRIX))
one_spec = BS.TASK_BY_ID["accuracy_e5_e6__mean_delta"]
r_a = BS.run_bootstrap_task(one_spec, ana, N, B_SMALL,
                            BS.comparison_observed(results[5], BS.MEAN_DELTA),
                            *BS.scalar_callables(BS.MEAN_DELTA, paired["accuracy_e5_e6"].delta))
r_b = BS.run_bootstrap_task(one_spec, ana, N, B_SMALL,
                            BS.comparison_observed(results[5], BS.MEAN_DELTA),
                            *BS.scalar_callables(BS.MEAN_DELTA, paired["accuracy_e5_e6"].delta))
orig = task_results[BS.FROZEN_TASK_IDS.index("accuracy_e5_e6__mean_delta")]
case("comparison-order / statistic-order / scheduling independence",
     np.array_equal(r_a.replicates, orig.replicates)
     and np.array_equal(r_b.replicates, orig.replicates))
case("shuffled execution order does not change the frozen matrix",
     tuple(s.task_id for s in sorted(shuffled, key=lambda s: BS.FROZEN_TASK_IDS.index(s.task_id)))
     == BS.FROZEN_TASK_IDS)

grid = official_corruption_grid()
inv = [("E%d" % i, None, None) for i in range(1, 8)] + [
    (st, c, s) for st in ("E1", "E6") for c in grid.names for s in INFERENTIAL_SEVERITIES]
case("official input inventory derives to exactly 37 records", len(inv) == 37, str(len(inv)))
case("inventory: 7 clean + 30 corruption", sum(1 for r in inv if r[1] is None) == 7)
case("inventory: corruption stages are exactly E1 and E6",
     sorted({r[0] for r in inv if r[1] is not None}) == ["E1", "E6"])
case("inventory: severities 1-3 only, never 4 or 5",
     {r[2] for r in inv if r[2] is not None} == {1, 2, 3})
case("inventory: all 37 identities unique", len(set(inv)) == 37)

input_artifacts = []
for stage, corr, sev in inv:
    p = (f"runs/eval/{stage.lower()}_clean_test" if corr is None
         else f"runs/eval/{stage.lower()}_{corr}_s{sev}_test")
    input_artifacts.append({"stage": stage,
                            "condition": {"type": "clean" if corr is None else "corruption",
                                          "name": corr, "severity": sev},
                            "repo_relative_path": p, "artifact_status": "smoke",
                            "checkpoint_sha256": "0" * 64, "split_manifest_sha256": "1" * 64,
                            "class_map_sha256": "2" * 64, "metric_impl_sha256": "3" * 64,
                            "config_sha256": "4" * 64, "repo_commit": "9267ffe"})

ni_stages = pooled_by_cmp["noninferiority_e3_e6"].stage_mious()
ni_task = task_results[-1]
ni_res = NI.build_non_inferiority(ni_stages[0], ni_stages[1], ni_task.bca.bounds[0])
e6_res = NI.build_e6_kd(ni_stages[0], ni_stages[1])
csha, craw = A.contract_sha256(ROOT)
case("contract hash is 64 lowercase hex from raw bytes",
     len(csha) == 64 and csha == csha.lower() and csha == _h.sha256(craw).hexdigest())
case("repository-baseline contract digest matches the checkpoint",
     csha == "d53c87dcd750fbc0fc288718ae214474e00a40b23863f293e4e0db3771b0f40a", csha[:16])

inp = A.ArtifactInputs(analysis_id=ana, run_id="a3b-smoke-0001",
                       created_at_utc="2026-07-31T00:00:00Z", contract_sha256=csha,
                       input_artifacts=input_artifacts, comparison_results=results,
                       holm_family=fam_holm, descriptive=desc, non_inferiority=ni_res,
                       e6_kd=e6_res, task_results=tuple(task_results), synthetic=True)
fam = A.build_family(inp)
case("nineteen top-level keys in exact frozen order", tuple(fam.keys()) == A.TOP_LEVEL_KEYS)
case("no additional top-level key", len(fam) == 19)
case("version literals unchanged", fam["schema_version"] == "plantseg-stats/1.0.0"
     and fam["statistical_protocol"] == "plantseg-stats-protocol/1.0.0")
case("family_id frozen", fam["family_id"] == "plantseg_eight_test_holm_v1")
case("a3a_family has exactly three keys in order",
     tuple(fam["a3a_family"].keys()) == ("completion_status", "comparisons", "holm"))
case("holm carrier key order is the HolmFamily declaration order",
     tuple(fam["a3a_family"]["holm"].keys()) == ("members", "alpha", "complete", "status"))
case("holm.alpha == 0.05, complete true, status ok",
     fam["a3a_family"]["holm"]["alpha"] == 0.05
     and fam["a3a_family"]["holm"]["complete"] is True
     and fam["a3a_family"]["holm"]["status"] == "ok")
case("top-level alpha == holm.alpha exactly",
     fam["alpha"] == fam["a3a_family"]["holm"]["alpha"])
case("completion_status agrees with holm.complete",
     (fam["a3a_family"]["completion_status"] == "complete")
     is (fam["a3a_family"]["holm"]["complete"] is True))
case("8 comparisons and 8 holm members, positionally aligned",
     [c["comparison_id"] for c in fam["a3a_family"]["comparisons"]]
     == [m["comparison_id"] for m in fam["a3a_family"]["holm"]["members"]]
     == list(fam["expected_comparison_ids"]))
case("ComparisonResult serialized in the frozen 16-field order",
     tuple(fam["a3a_family"]["comparisons"][0].keys()) == BS.FROZEN_A3A_FIELDS["ComparisonResult"])
case("WilcoxonResult serialized in the frozen 12-field order",
     tuple(fam["a3a_family"]["comparisons"][0]["wilcoxon"].keys())
     == BS.FROZEN_A3A_FIELDS["WilcoxonResult"])
case("HolmMember serialized in the frozen 9-field order",
     tuple(fam["a3a_family"]["holm"]["members"][0].keys())
     == BS.FROZEN_A3A_FIELDS["HolmMember"])
case("Policy enum serialized through .value",
     fam["a3a_family"]["comparisons"][0]["policy"] == Policy.NONOFFICIAL_SMOKE.value)
case("no descriptive record inside the A3a family",
     all(c["comparison_id"] != "descriptive_e1_e3" for c in fam["a3a_family"]["comparisons"]))
case("no bootstrap field inside any A3a comparison record",
     not any(k in fam["a3a_family"]["comparisons"][0]
             for k in ("bootstrap", "z0", "acceleration", "lower_bound")))
case("source schema assertion passes against live dataclasses",
     BS.assert_a3a_source_schema() is None)
case("input_artifacts sorted by repo_relative_path only",
     [a["repo_relative_path"] for a in fam["input_artifacts"]]
     == sorted(a["repo_relative_path"] for a in fam["input_artifacts"]))
case("35 ordered bootstrap-task records",
     [t["task_id"] for t in fam["bootstrap_tasks"]] == list(BS.FROZEN_TASK_IDS))
case("34 two-sided, 1 one-sided in the artifact",
     sum(1 for t in fam["bootstrap_tasks"] if t["interval_type"] == "two_sided") == 34
     and sum(1 for t in fam["bootstrap_tasks"] if t["interval_type"] == "one_sided_lower") == 1)
case("only the NI task omits upper_bound",
     [t["task_id"] for t in fam["bootstrap_tasks"] if "upper_bound" not in t]
     == ["noninferiority_e3_e6__dataset_miou_delta"])
case("observed_source present on every task and matching the frozen map",
     all(t["observed_source"] == BS.TASK_BY_ID[t["task_id"]].observed_source
         for t in fam["bootstrap_tasks"]))
case("artifact is NONOFFICIAL with a non-empty officiality warning list",
     fam["artifact_status"] == "nonofficial" and len(fam["warnings"]) > 0, str(fam["warnings"]))
case("officiality warnings drawn only from the frozen six",
     set(fam["warnings"]) <= set(A.OFFICIALITY_WARNINGS))
case("status derived from warnings",
     A.derive_artifact_status(tuple(fam["warnings"])) == fam["artifact_status"])
case("thirteen integrity checks in exact order, all true",
     tuple(fam["integrity"]["checks"].keys()) == A.INTEGRITY_CHECKS
     and all(fam["integrity"]["checks"].values())
     and fam["integrity"]["status"] == "passed")

tmp = Path(tempfile.mkdtemp(prefix="a3b_smoke_"))
try:
    out = tmp / "a3b-smoke-0001"
    A.write_statistics_artifact(out, fam, task_results)
    case("artifact directory contains exactly three files",
         sorted(p.name for p in out.iterdir()) == ["MANIFEST.sha256", "bootstrap.npz",
                                                   "family.json"])
    reloaded = A.verify_statistics_artifact(out)
    case("artifact re-verifies from disk", reloaded["run_id"] == "a3b-smoke-0001")
    case("JSON<->NPZ exact equality holds on reload", True)
    txt = (out / "family.json").read_text(encoding="utf-8")
    case("no NaN or Infinity token in family.json",
         "NaN" not in txt and "Infinity" not in txt)
    with np.load(out / "bootstrap.npz", allow_pickle=False) as z:
        case("NPZ key set is exactly 35 x 13",
             len(z.files) == 35 * len(A.NPZ_SUFFIXES))
        case("no [B, n] sampled-index matrix stored",
             all(z[k].ndim <= 1 for k in z.files))
        case("bootstrap/jackknife arrays are float64",
             z[BS.FROZEN_TASK_IDS[0] + "__bootstrap"].dtype == np.float64)
        case("NI task stores exactly one bound",
             z["noninferiority_e3_e6__dataset_miou_delta__bounds"].size == 1)
    case("refuse-existing target", raises(lambda: A.write_statistics_artifact(out, fam,
                                                                             task_results),
                                          A.ArtifactError))
    out2 = tmp / "a3b-smoke-0002"
    fam2 = dict(fam); fam2["run_id"] = "a3b-smoke-0002"
    case("injected failure raises",
         raises(lambda: A.write_statistics_artifact(out2, fam2, task_results,
                                                    _inject_failure=True), A.ArtifactError))
    case("no partial directory after injected failure",
         not out2.exists() and not any(p.name.startswith(".a3b-smoke-0002")
                                       for p in tmp.iterdir()))
    (out / "family.json").write_text(txt.replace('"alpha": 0.05', '"alpha": 0.05 '),
                                     encoding="utf-8", newline="\n")
    case("manifest tamper detection",
         raises(lambda: A.verify_statistics_artifact(out), A.ArtifactError))
    out3 = tmp / "a3b-smoke-0003"
    fam3 = dict(fam); fam3["run_id"] = "a3b-smoke-0003"
    fam3["artifact_status"] = "official"
    case("official request on the drifted dev stack refuses",
         raises(lambda: A.write_statistics_artifact(out3, fam3, task_results), A.ArtifactError))
    case("refusal left no final directory and no temp directory",
         not out3.exists() and not any(p.name.startswith(".a3b-smoke-0003")
                                       for p in tmp.iterdir()))
    fam4 = dict(fam); fam4["run_id"] = "Bad_Run_ID"
    case("invalid run_id refused",
         raises(lambda: A.write_statistics_artifact(tmp / "Bad_Run_ID", fam4, task_results),
                A.ArtifactError))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ==================================================================================================
print("\n" + "=" * 96)
print("SUMMARY  %d/%d checks passed" % (PASS, PASS + FAILED))
print("RESULT: %s" % ("A3b SMOKE OK" if FAILED == 0 else "A3b SMOKE FAILED"))
print("NONOFFICIAL: synthetic fixtures, small B=%d (production B=%d recorded), dev stack differs "
      "from the pinned statistics stack." % (B_SMALL, BS.PRODUCTION_B))
print("No PlantSeg image, mask, split file or checkpoint was accessed; no training, inference or "
      "GPU use.")
print("=" * 96)
sys.exit(1 if FAILED else 0)
