#!/usr/bin/env python3
"""Smoke: D1, NMF-draw sensitivity of the distillation targets (lane L-TEACHER-DIAG; AM-18 item 1(a)).

Acceptance (b), the statistics of src/eval/nmf_sensitivity.py on crafted logits with hand-computed
answers (P12-P15, P32(b), P33): N, D and both integer branch booleans, including both boundaries
(100*N == 3*D and 10*N == D), D == 0, the two big-integer pairs where a float rule would decide the
other way, and the exact tie (5, 5, 0); F_lesion with its undefined case; F_halves; KL_logit and KL_cwd
against an independent numpy reference (1e-12; identity cases 1e-15) and the CWD logit-map construction;
the bootstrap vector of P32(b), the index sha256 of P15, identical replicates across runs, a replicate
with sum D == 0, and SciPy's percentile bootstrap; d2_crops; empty crops add nothing.

The seam (src/eval/teacher_diag.py) on the stub teacher of scripts/teacher_diag_fixtures.py: both load
lines (P6, P7), the stream object of C1 (a check against the begin call's dict misses a head call's draw,
the object sees it), the split-forward checks of P9 (no stream, a wrong stream object, shape, features
changed in place, NaN, a global-RNG draw), K=1 bit for bit against frozen(x).logits, eight interleaved
streams each equal to itself run alone (P10), the stub guards of P2, the sha256 formats of P5 with the
load counter at 0, the role pins of P3, the provenance count of P8, the flags of C2 and P26-P28, the
commit binding (fake git answers), one output per kind, and the P21 path guards.

The script (scripts/teacher_d1_nmf_sensitivity.py) on a synthetic 5,367/846 root. Crops (acceptance a):
the sample is the literal AM-10 draw (ids 'a', 'a-1', 'a (1)'); crop i equals the trainer's TRAIN
branch with RandomState(1801 + i); the all-valid 64x64 domain; backbone once and head 8 times per crop;
draws == 8 x crops with the end state of a reference M4-KD stream; F, KL and the CI; the arm role
without booleans; --d2-val-output identity (P11); D == 0; refusals before any load. VAL part
(acceptance c): eight streams seeded 42..49; stream 42 equal to the evaluator's M4-V pass (end state,
mIoU, ECE); the core's confusion equals stream 42's; probabilities averaged, not logits; the
reproduction gate both ways; the spread over streams.
Synthetic inputs only; no PlantSeg data, no checkpoint of record, no GPU.

    python -B scripts/smoke_teacher_d1.py
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import os
import shutil
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.eval import nmf_sensitivity as ns  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
P32_CI = (0.07283787157730934, 0.08666608984644292)
P15_INDEX_SHA = "73f7b7cb536b31834947c28405e632c81484adcd9ae07f7f0ec641b12d56a5d3"


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def crafted(labels, valid, c=4, high=10.0):
    """[8, C, h, w] logits whose argmax per draw and cell is `labels` ([8, h, w])."""
    labels = np.asarray(labels)
    z = np.zeros((8, c) + labels.shape[1:], dtype=np.float32)
    for k in range(8):
        for idx in np.ndindex(labels.shape[1:]):
            z[(k, int(labels[(k,) + idx])) + idx] = high
    return z, np.asarray(valid, dtype=bool)


def grid_with_deviants(n_valid: int, n_deviant: int, shape=(5, 5)):
    """All draws say class 1 except one draw saying 2 in the first n_deviant valid cells."""
    labels = np.ones((8,) + shape, dtype=int)
    valid = np.zeros(shape, dtype=bool)
    cells = list(np.ndindex(shape))[:n_valid]
    for cell in cells:
        valid[cell] = True
    for cell in cells[:n_deviant]:
        labels[(0,) + cell] = 2
    return crafted(labels, valid)


def ref_kl_logit(z, valid, t=4.0):
    zz = z.astype(np.float64).reshape(8, z.shape[1], -1)[:, :, valid.reshape(-1)]
    total = 0.0
    for p in range(zz.shape[2]):
        probs = []
        for k in range(8):
            e = np.exp(zz[k, :, p] / t - np.max(zz[k, :, p] / t))
            probs.append(e / e.sum())
        pbar = np.mean(probs, axis=0)
        for k in range(8):
            total += float(np.sum(probs[k] * (np.log(probs[k]) - np.log(pbar))))
    return total


def ref_kl_cwd(z, valid, t=4.0):
    zz = z.astype(np.float64).reshape(8, z.shape[1], -1)[:, :, valid.reshape(-1)]
    total = 0.0
    for c in range(zz.shape[1]):
        qs = []
        for k in range(8):
            e = np.exp(zz[k, c] / t - np.max(zz[k, c] / t))
            qs.append(e / e.sum())
        qbar = np.mean(qs, axis=0)
        for k in range(8):
            total += float(np.sum(qs[k] * (np.log(qs[k]) - np.log(qbar))))
    return total


def case_tie_per_draw() -> bool:
    """One draw tied (5, 5, 0), seven draws (5, 4, 0): first-index argmax gives class 0 everywhere, N == 0
    (a last-index rule would give 7 flips)."""
    z = np.zeros((8, 3, 1, 1), dtype=np.float32)
    z[:, 0], z[:, 1] = 5.0, 4.0
    z[0, 1] = 5.0
    return ns.crop_statistics(z, np.ones((1, 1), bool)).N == 0


def case_lesion_mean_probability() -> bool:
    """Draw 1 logits (8, 0), draws 2-8 (0, 1): the mean probability peaks at class 1 (a lesion cell), the mean
    logit (1, 0.875) at the background."""
    z = np.zeros((8, 2, 1, 1), dtype=np.float32)
    z[0, 0] = 8.0
    z[1:, 1] = 1.0
    return ns.crop_statistics(z, np.ones((1, 1), bool)).n_lesion == 1


def case_halves_mean_probability() -> bool:
    """Draws 1-4: (8, 0) once and (0, 1) three times, draws 5-8: (0, 1). The half means of probabilities both
    say class 1 (no flip); the half means of logits would say 0 and 1 (a flip)."""
    z = np.zeros((8, 2, 1, 1), dtype=np.float32)
    z[0, 0] = 8.0
    z[1:, 1] = 1.0
    return ns.crop_statistics(z, np.ones((1, 1), bool)).H == 0


def case_strict_099() -> bool:
    """The share > 0.99 is strict: one valid cell whose max-probability is p*; with the threshold patched to
    p* the cell is not counted, with the next float below p* it is."""
    z = np.zeros((8, 3, 1, 1), dtype=np.float32)
    z[:, 0] = 6.0
    v = np.ones((1, 1), bool)
    p_star = ns.crop_statistics(z, v).d2["max_prob_t1_sum"]         # one cell: the sum is its max-probability
    old = ns.HIGH_CONFIDENCE
    try:
        ns.HIGH_CONFIDENCE = p_star
        at = ns.crop_statistics(z, v).d2["n_max_prob_gt_0_99_t1"]
        ns.HIGH_CONFIDENCE = float(np.nextafter(p_star, 0.0))
        below = ns.crop_statistics(z, v).d2["n_max_prob_gt_0_99_t1"]
    finally:
        ns.HIGH_CONFIDENCE = old
    return at == 0 and below == 1


def stats_cases() -> None:
    # b1: pair counting
    check("b1 flips: labels 0000 1111 give 16 differing pairs; eight distinct labels give 28",
          ns.pairwise_flips(np.array([[0], [0], [0], [0], [1], [1], [1], [1]])) == 16
          and ns.pairwise_flips(np.arange(8).reshape(8, 1)) == 28)
    # b1: interior and both boundaries, |V| = 25 so D = 700
    for n_dev, want_n, want in ((1, 7, (True, False, "lt_0_03")), (3, 21, (False, False, "mid")),
                                (10, 70, (False, True, "ge_0_10")), (11, 77, (False, True, "ge_0_10"))):
        z, v = grid_with_deviants(25, n_dev)
        cs = ns.crop_statistics(z, v)
        br = ns.branch(cs.N, cs.D)
        check(f"b1 N={want_n}, D=700: booleans and branch {want}",
              (cs.N, cs.D) == (want_n, 700) and (br["below_0_03"], br["at_least_0_10"], br["branch"]) == want,
              f"N={cs.N} D={cs.D} {br}")
    check("b1 boundary 100*N == 3*D (N=21, D=700) is not below 0.03", ns.branch(21, 700)["below_0_03"] is False)
    check("b1 boundary 10*N == D (N=70, D=700) is at least 0.10", ns.branch(70, 700)["at_least_0_10"] is True)
    z, v = grid_with_deviants(0, 0)
    cs = ns.crop_statistics(z, v)
    s = ns.summarize([cs])
    check("b1 D == 0: 'F not produced', booleans and branch null",
          cs.D == 0 and s["F"]["status"] == "F not produced" and s["F"]["below_0_03"] is None
          and s["F"]["at_least_0_10"] is None and s["F"]["branch"] is None and s["F"]["value"] is None)
    b1 = ns.branch(210000000000000005, 7000000000000000168)
    b2 = ns.branch(1400000000000000011, 14000000000000000112)
    check("P33 (N, D) = (210000000000000005, 7000000000000000168): integers say below 0.03, a float rule not",
          b1["below_0_03"] is True and not (210000000000000005 / 7000000000000000168 < 0.03))
    check("P33 (N, D) = (1400000000000000011, 14000000000000000112): integers say not >= 0.10, a float rule does",
          b2["at_least_0_10"] is False and (1400000000000000011 / 14000000000000000112 >= 0.10))
    # tie (5, 5, 0) takes the first index, for the draws and for the mean-probability peak
    z = np.zeros((8, 3, 1, 1), dtype=np.float32)
    z[:, 0], z[:, 1] = 5.0, 5.0
    cs = ns.crop_statistics(z, np.ones((1, 1), bool))
    check("b1 exact tie (5, 5, 0) takes the first index: no flips, the cell peaks at class 0 (no lesion)",
          cs.N == 0 and cs.n_lesion == 0)
    z[:, 0], z[:, 1] = 5.0, 5.0
    z[:, 2] = 0.0
    z2 = z.copy()
    z2[:, 0], z2[:, 1] = 0.0, 5.0
    z2[:, 2] = 5.0
    cs2 = ns.crop_statistics(z2, np.ones((1, 1), bool))
    check("b1 tie (0, 5, 5) peaks at class 1 (the first maximum), a lesion cell", cs2.n_lesion == 1)
    check("b1 one tied draw (5, 5, 0) among seven (5, 4, 0): first index, N == 0", case_tie_per_draw())
    bad = np.zeros((8, 3, 2, 2), dtype=np.float32)
    bad[3, 1, 0, 0] = np.nan
    check("b1 a NaN logit is refused by the statistics", raises(lambda: ns.crop_statistics(bad, np.ones((2, 2), bool)),
                                                                ValueError))

    # b2: F_lesion -- cells peaking at background never count; the undefined case
    labels = np.zeros((8, 2, 2), dtype=int)
    labels[:, 0, 0] = 1                          # a lesion cell, no flip
    labels[:, 0, 1] = 2
    labels[0, 0, 1] = 1                          # a lesion cell with one deviant draw: 7 flips
    labels[0, 1, 0] = 3                          # a background cell with one deviant: 7 flips, not lesion
    z, v = crafted(labels, np.ones((2, 2), bool))
    cs = ns.crop_statistics(z, v)
    check("b2 F_lesion counts only cells peaking at a disease class: N_L = 7, D_L = 56, N = 14",
          (cs.N_L, cs.D_L, cs.n_lesion, cs.N) == (7, 56, 2, 14), f"{cs.N_L} {cs.D_L} {cs.n_lesion} {cs.N}")
    z, v = crafted(np.zeros((8, 2, 2), dtype=int), np.ones((2, 2), bool))
    s = ns.summarize([ns.crop_statistics(z, v)])
    check("b2 F_lesion is 'undefined' when no cell peaks at a disease class",
          s["F_lesion"]["value"] == "undefined" and s["F_lesion"]["D"] == 0)

    # b3: F_halves
    labels = np.ones((8, 2, 2), dtype=int)
    labels[4:, 0, 0] = 2                          # draws 5-8 disagree with 1-4 at one cell
    labels[7, 1, 1] = 3                           # one draw of the second half: the half-mean still says 1
    z, v = crafted(labels, np.ones((2, 2), bool))
    cs = ns.crop_statistics(z, v)
    s = ns.summarize([cs])
    check("b3 F_halves = 1 flip over 4 cells", cs.H == 1 and s["F_halves"]["value"] == 0.25, f"H={cs.H}")
    check("b2 F_lesion's cells come from the mean probability, not the mean logit", case_lesion_mean_probability())
    check("b3 F_halves compares half means of probabilities, not of logits", case_halves_mean_probability())

    # b4 / b5: KL against independent references
    rng = np.random.default_rng(3)
    z = rng.normal(0, 2.0, size=(8, 6, 4, 5)).astype(np.float32)
    v = rng.random((4, 5)) > 0.3
    cs = ns.crop_statistics(z, v)
    rl, rc = ref_kl_logit(z, v), ref_kl_cwd(z, v)
    check("b4 KL_logit sum equals the per-cell numpy reference to 1e-12",
          abs(cs.kl_logit_sum - rl) <= 1e-12 * max(1.0, abs(rl)), f"{cs.kl_logit_sum!r} vs {rl!r}")
    check("b5 KL_cwd sum equals the per-channel numpy reference to 1e-12",
          abs(cs.kl_cwd_sum - rc) <= 1e-12 * max(1.0, abs(rc)), f"{cs.kl_cwd_sum!r} vs {rc!r}")
    z2 = rng.normal(0, 3.0, size=(8, 6, 4, 5)).astype(np.float32)
    v2 = np.zeros((4, 5), dtype=bool)
    v2[0, :3] = True                                           # |V| = 3, against the first crop's larger set
    cs2 = ns.crop_statistics(z2, v2)
    pooled = ns.summarize([cs, cs2])
    want_cwd = (ref_kl_cwd(z, v) + ref_kl_cwd(z2, v2)) / (8 * 6 * 2)
    want_logit = (rl + ref_kl_logit(z2, v2)) / (8 * (int(v.sum()) + 3))
    check("b5 KL_cwd pools unweighted over (crop, draw, channel); KL_logit over (crop, draw, cell) -- two crops of "
          "different |V|, against the numpy references to 1e-12",
          abs(pooled["KL_cwd"]["value"] - want_cwd) <= 1e-12 * max(1.0, want_cwd)
          and abs(pooled["KL_logit"]["value"] - want_logit) <= 1e-12 * max(1.0, want_logit),
          f"{pooled['KL_cwd']['value']!r} vs {want_cwd!r}")
    same = np.repeat(z[:1], 8, axis=0)
    cs_same = ns.crop_statistics(same, v)
    check("b4/b5 identity: eight equal draws give KL_logit = KL_cwd = 0 within 1e-15",
          abs(cs_same.kl_logit_sum) <= 1e-15 and abs(cs_same.kl_cwd_sum) <= 1e-15,
          f"{cs_same.kl_logit_sum!r} {cs_same.kl_cwd_sum!r}")
    zz = z.astype(np.float64).reshape(8, 6, -1)[:, :, v.reshape(-1)]
    soft_mean = np.exp(zz.mean(0) / 4 - np.max(zz.mean(0) / 4, 0))
    soft_mean /= soft_mean.sum(0)
    p = np.exp(zz / 4 - zz.max(1, keepdims=True) / 4)
    p /= p.sum(1, keepdims=True)
    alt = float(np.sum(p * (np.log(p) - np.log(soft_mean)[None])))
    check("b4 pbar is the mean of probabilities: the softmax-of-mean-logits variant differs",
          abs(alt - rl) > 1e-6, f"alt={alt!r} ref={rl!r}")
    try:
        import torch
        import torch.nn.functional as F
        zt = torch.tensor(z.astype(np.float64).reshape(8, 6, -1))
        m = torch.tensor(v.reshape(1, 1, -1))
        s_ = (zt / 4).masked_fill(~m, torch.finfo(torch.float64).min)
        lq_t = F.log_softmax(s_, dim=-1)[:, :, torch.tensor(v.reshape(-1))].numpy()
        lq = ns.log_softmax(zz / 4, axis=2)
        check("b5 q_kc equals the CWD logit-map construction (masked fill, softmax over cells, T=4) to 1e-13",
              float(np.max(np.abs(lq - lq_t))) <= 1e-13, f"max |diff| {float(np.max(np.abs(lq - lq_t)))!r}")
    except ImportError:
        check("b5 torch present for the CWD construction check", False, "torch missing")

    # b6: bootstrap (P32(b) vector, P15 index hash)
    n = 256
    vv = [0 if i % 29 == 0 else 1 + (37 * i) % 101 for i in range(n)]
    d_i = [28 * x for x in vv]
    n_i = [(d_i[i] * ((11 * i) % 17)) // 100 for i in range(n)]
    br = ns.branch(sum(n_i), sum(d_i))
    check("b6 P32 vector: N = 27980, D = 351288, both booleans False",
          (sum(n_i), sum(d_i), br["below_0_03"], br["at_least_0_10"]) == (27980, 351288, False, False))
    bs1 = ns.bootstrap_f(n_i, d_i)
    bs2 = ns.bootstrap_f(n_i, d_i)
    check("b6 P32 vector: CI == [0.07283787157730934, 0.08666608984644292] exactly",
          (bs1["low"], bs1["high"]) == P32_CI, f"{bs1['low']!r} {bs1['high']!r}")
    check("b6 P15: sha256 of I == 73f7b7cb... (numpy 1.26.4)",
          bs1["index_sha256"] == P15_INDEX_SHA or np.__version__ != "1.26.4", f"{bs1['index_sha256']} numpy {np.__version__}")
    check("b6 identical across two runs (replicates sha256) and the CI contains the point",
          bs1["replicates_sha256"] == bs2["replicates_sha256"] and bs1["contains_point"] is True)
    zero = ns.bootstrap_f([0] * 255 + [3], [0] * 255 + [28])
    check("b6 a replicate with sum D == 0 makes the CI null, with the count",
          zero["low"] is None and zero["high"] is None and zero["replicates_with_zero_D"] > 0)
    try:
        from scipy.stats import bootstrap
        na, da = np.asarray(n_i, np.int64), np.asarray(d_i, np.int64)
        res = bootstrap((np.arange(n),), lambda idx: na[idx].sum() / da[idx].sum(), n_resamples=10000,
                        method="percentile", random_state=np.random.default_rng(1801), vectorized=False)
        reps = na[np.random.default_rng(1801).integers(0, n, size=(10000, n))].sum(1) / \
            da[np.random.default_rng(1801).integers(0, n, size=(10000, n))].sum(1)
        lo, hi = res.confidence_interval
        import hashlib
        scipy_sha = hashlib.sha256(np.ascontiguousarray(np.asarray(res.bootstrap_distribution), dtype="<f8")
                                   .tobytes()).hexdigest()
        check("b6 SciPy percentile bootstrap: bootstrap_f's replicates equal SciPy's bit for bit (sha256), bounds "
              "within 1e-15",
              np.array_equal(np.asarray(res.bootstrap_distribution), reps) and bs1["replicates_sha256"] == scipy_sha
              and abs(lo - bs1["low"]) <= 1e-15 and abs(hi - bs1["high"]) <= 1e-15,
              f"scipy [{lo!r}, {hi!r}]")
    except ImportError:
        check("b6 SciPy available for the cross-check", False, "scipy missing")

    # b7: d2_crops against numpy; empty crops add nothing
    zf = z[0].astype(np.float64).reshape(6, -1)[:, v.reshape(-1)]
    ref = {}
    for tag, t in (("t1", 1.0), ("t4", 4.0)):
        e = np.exp(zf / t - np.max(zf / t, 0))
        pr = e / e.sum(0)
        ref[f"mean_max_prob_{tag}"] = pr.max(0).mean()
        ref[f"mean_entropy_{tag}"] = (-(pr * np.log(pr)).sum(0)).mean()
    s = ns.summarize([cs])["d2_crops"]
    check("b7 d2_crops (first draw) means match numpy to 1e-12",
          all(abs(s[k] - ref[k]) <= 1e-12 for k in ref), str({k: (s[k], ref[k]) for k in ref}))
    e1 = np.exp(zf - np.max(zf, 0))
    check("b7 share > 0.99 matches numpy", s["n_max_prob_gt_0_99_t1"] == int(((e1 / e1.sum(0)).max(0) > 0.99).sum()))
    check("b7 share > 0.99 is strict: a cell exactly at the threshold is not counted, one just above it is",
          case_strict_099())
    empty = ns.crop_statistics(z, np.zeros((4, 5), bool))
    a, b = ns.summarize([cs]), ns.summarize([cs, empty])
    check("b7 a crop with no valid cell adds nothing",
          all(a[k] == b[k] for k in ("F", "F_lesion", "F_halves", "KL_logit", "KL_cwd", "d2_crops")))


# ---------------------------------------------------------------------------------------------------
# the seam (src/eval/teacher_diag.py): load, stream object, split forward, gates -- stub teacher only
# ---------------------------------------------------------------------------------------------------
def _seam_env():
    from scripts import teacher_diag_fixtures as fx
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    from src.eval import teacher_diag as td
    tmp = safe_tmpdir("diag_seam_")
    ckpt, sha = fx.write_stub_ckpt(tmp / "stub_teacher.pth")

    def args(**kw):
        base = dict(teacher_ckpt=str(ckpt), teacher_ckpt_sha256=sha,
                    teacher_config=str(td.REPO / td.TEACHER_CONFIG_REL), teacher_role="record",
                    arm_id=None, arm_dl_id=None, teacher_config_sha256=None)
        base.update(kw)
        return types.SimpleNamespace(**base)
    return fx, td, tmp, ckpt, sha, args


_PROV_NAMES = ("builder", "ckpt_path", "ckpt_sha256", "ckpt_bytes", "config_sha256", "teacher_components_sha256",
               "reused_module_hashes", "architecture_signature", "model_cfg_sha256", "field_10", "field_11", "field_12")


def prov_class(n: int, *, drop_from_record: str | None = None):
    """A stand-in TeacherProvenance with n fields (P8 cases independent of the merged dataclass)."""
    def as_dict(self):
        d = dataclasses.asdict(self)
        d.pop(drop_from_record, None)
        return d
    return dataclasses.make_dataclass(f"Prov{n}", [(f, object) for f in _PROV_NAMES[:n]], namespace={"as_dict": as_dict})


def case_p8_count(td) -> bool:
    import src.distill.teacher as dt
    from scripts import teacher_diag_fixtures as fx
    with fx.patched(dt, TeacherProvenance=prov_class(11)):
        eleven = raises(lambda: td.require_provenance_field_count(True), td.Refused)
        stub_ok = not raises(lambda: td.require_provenance_field_count(False), Exception)
    with fx.patched(dt, TeacherProvenance=prov_class(12)):
        twelve = not raises(lambda: td.require_provenance_field_count(True), Exception)
    return eleven and stub_ok and twelve


def case_p8_nonempty(td, loaded, inputs) -> bool:
    cls = prov_class(12)
    vals = {f: "v" for f in _PROV_NAMES}
    vals.update(ckpt_sha256=inputs.sha256, field_11=None, field_12="")
    old = loaded.frozen.provenance
    loaded.frozen.provenance = cls(**vals)
    try:
        td.after_load_checks(loaded, inputs, stub=False)
        return False
    except td.Refused as e:
        return "field_11" in str(e) and "field_12" in str(e)
    except Exception:  # noqa: BLE001
        return False
    finally:
        loaded.frozen.provenance = old


def case_p8_record(td, loaded, inputs, checks) -> bool:
    import src.distill.teacher as dt
    from scripts import teacher_diag_fixtures as fx
    vals = {f: "v" for f in _PROV_NAMES}
    vals["ckpt_sha256"] = inputs.sha256
    old = loaded.frozen.provenance
    try:
        with fx.patched(dt, TeacherProvenance=prov_class(12)):
            loaded.frozen.provenance = prov_class(12, drop_from_record="field_12")(**vals)
            try:
                td.teacher_record(loaded, inputs, checks, stub=False)
                lacking = False
            except td.Refused as e:
                lacking = "field_12" in str(e)
            loaded.frozen.provenance = prov_class(12)(**vals)
            complete = not raises(lambda: td.teacher_record(loaded, inputs, checks, stub=False), Exception)
    finally:
        loaded.frozen.provenance = old
    return lacking and complete


def case_frozen_blob(td) -> bool:
    from scripts import teacher_diag_fixtures as fx
    first = next(iter(td.FROZEN))
    table = dict(td.FROZEN)
    table[first] = ("0" * 40, table[first][1])
    with fx.patched(td, FROZEN=table):
        return raises(td.frozen_blob_record, td.Refused)


def case_stub_ckpt_exact(td, fx, tmp, loaded) -> bool:
    """h1: write_stub_ckpt holds exactly the factory's state_dict (keys, dtypes, shapes, values), for the default
    and the big factory, and the strictly loaded stub's state equals the factory's own seeded state."""
    import torch
    ok = True
    for factory, tag in ((None, "h1_default"), (fx.big_factory, "h1_big")):
        p, _ = fx.write_stub_ckpt(Path(tmp) / f"{tag}.pth", tag=tag, factory=factory)
        saved = torch.load(p, map_location="cpu")["state_dict"]
        ref = (factory or fx.stub_factory)().state_dict()
        ok &= sorted(saved) == sorted(ref) and all(
            saved[k].dtype == ref[k].dtype and saved[k].shape == ref[k].shape and torch.equal(saved[k], ref[k])
            for k in ref)
        p.unlink()
    ok &= td.loaded_state_sha256(loaded.segmentor) == td.loaded_state_sha256(fx.stub_factory())
    return bool(ok)


def case_reference_sha(fx, tmp, ckpt, sha) -> tuple[bool, str]:
    """h3: fx.reference_artifact passes --teacher-ckpt-sha256 (the sha write_stub_ckpt returned): with it the
    evaluator builds the reference and records that sha; a well-formed wrong sha is refused by its hash check."""
    import json
    import tempfile

    from src.eval.stage_artifacts import StageArtifactError
    tmp = Path(tempfile.mkdtemp(prefix="ref_h3_", dir=tmp))      # a fresh folder per call (the harness repeats it)
    try:
        d = fx.reference_artifact(Path(tmp) / "ref_h3", ckpt, sha, n=1, run_id="ref_h3")
        got = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))["run"]["checkpoint_sha256"]
        built, detail = got == sha, f"recorded {got}"
    except Exception as exc:  # noqa: BLE001
        built, detail = False, f"{type(exc).__name__}: {exc}"[:300]
    try:
        fx.reference_artifact(Path(tmp) / "ref_h3_wrong", ckpt, "0" * 64, n=1, run_id="ref_h3_wrong")
        wrong = False
    except StageArtifactError as exc:
        wrong = exc.code == "teacher_hash_mismatch"
    except Exception:  # noqa: BLE001
        wrong = False
    return built and wrong, detail


def case_big_model(td, fx, tmp, args) -> tuple[bool, str]:
    """h2: the oversized stub loads strictly from a checkpoint written from big_factory (about 8 MB, under the
    16 MiB stat guard), so the parameter guard is what refuses; its message must name the parameter count."""
    ckpt, sha = fx.write_stub_ckpt(Path(tmp) / "big_model_stub.pth", tag="big_model", factory=fx.big_factory)
    n = sum(p.numel() for p in fx.big_factory().parameters())
    try:
        gated_load(td, args(teacher_ckpt=str(ckpt), teacher_ckpt_sha256=sha), factory=fx.big_factory)
        msg = "no refusal"
    except td.Refused as e:
        msg = str(e)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
    ok = (n > td.STUB_MAX_PARAMETERS and ckpt.stat().st_size <= td.STUB_MAX_CKPT_BYTES
          and f"a model with {n} parameters" in msg)
    return ok, f"n={n} bytes={ckpt.stat().st_size} msg={msg[:200]}"


def _fresh(td, fx, args, factory=None):
    return gated_load(td, args(), factory=factory or fx.stub_factory)


def case_training_mode(td, fx, args) -> bool:
    inputs, loaded, _ = _fresh(td, fx, args)
    loaded.segmentor.train(True)
    try:
        td.after_load_checks(loaded, inputs, stub=True)
        return False
    except td.Stop as e:
        return "training mode" in str(e)


def case_extra_isolated(td, fx, args) -> bool:
    inputs, loaded, _ = _fresh(td, fx, args)
    loaded.segmentor.add_module("extra_ham", fx.StubIsolatedNMF())
    try:
        td.after_load_checks(loaded, inputs, stub=True)
        return False
    except td.Stop as e:
        return "exactly one isolated NMF module" in str(e)


def case_describe_mismatch(td, fx, args) -> bool:
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    original = SegNeXtTeacherAdapter.begin_nmf_stream

    def skewed(self, policy, seed=42):
        d = original(self, policy, seed)
        return None if d is None else dict(d, draws=d["draws"] + 1)
    with fx.patched(SegNeXtTeacherAdapter, begin_nmf_stream=skewed):
        try:
            _fresh(td, fx, args)
            return False
        except td.Stop as e:
            return "does not match" in str(e)
        except Exception:  # noqa: BLE001
            return False


def case_c1_object(td, fx, args) -> bool:
    try:
        _, loaded, _ = _fresh(td, fx, args)
    except Exception:  # noqa: BLE001
        return False
    return (loaded.stream is loaded.adapter.nmf_stream and loaded.stream_description == loaded.stream.describe()
            and loaded.stream.draws == 0)


def case_exit_codes(td) -> bool:
    import argparse

    from scripts.build_train_strata import StrataError
    from src.distill.nmf_stream import NMFStreamError
    from src.eval.artifacts import ArtifactRequestError, ArtifactWriteError
    from src.eval.calibration import CalibrationStop
    from src.eval.eval_runtime import EvalRuntimeError
    from src.eval.evaluate import EvaluationIntegrityError
    from src.eval.stage_artifacts import StageArtifactError
    from src.quant.calibration import CalibrationIndexError
    stops = (td.Stop, EvaluationIntegrityError, NMFStreamError, CalibrationStop, EvalRuntimeError, ArtifactWriteError)
    refusals = (td.Refused, ArtifactRequestError, StageArtifactError, StrataError, CalibrationIndexError)
    ok = all(td.exit_code_for(c.__new__(c)) == td.EXIT_STOP for c in stops)
    ok &= all(td.exit_code_for(c.__new__(c)) == td.EXIT_REFUSED for c in refusals)
    ok &= td.exit_code_for(KeyError("x")) is None

    def boom(_args):
        raise KeyError("unexpected")
    parser = argparse.ArgumentParser()
    parser.add_argument("--x")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ok &= td.cli_main(parser, boom, []) == td.EXIT_ERROR
        ok &= td.cli_main(parser, lambda a: td.EXIT_OK, ["--bad"]) == td.EXIT_REFUSED
        ok &= td.run_with_exit_codes(lambda: (_ for _ in ()).throw(EvaluationIntegrityError("x"))) == td.EXIT_STOP
    return bool(ok)


def gated_load(td, a, *, stub=True, mode="kd", factory=None):
    """The scripts' teacher gate order: flags, inputs, the load, the after-load checks."""
    td.check_teacher_flags(a)
    inputs = td.verify_teacher_inputs(a, stub=stub)
    loaded = td.load_teacher(mode, inputs, model_factory=factory)
    return inputs, loaded, td.after_load_checks(loaded, inputs, stub=stub)


def seam_cases() -> None:
    env = _seam_env()
    try:
        _seam_body(*env)
    finally:
        shutil.rmtree(env[2], ignore_errors=True)


def _seam_body(fx, td, tmp, ckpt, sha, args) -> None:
    import torch
    from src.distill.nmf_stream import NMFStream
    g = torch.Generator().manual_seed(9)
    x = torch.rand(1, 3, 512, 512, generator=g)
    xs = [torch.rand(1, 3, 64, 64, generator=g) for _ in range(3)]

    # load, both modes (P6, P7; C1)
    inputs, kd, checks = gated_load(td, args(), factory=fx.stub_factory)
    check("seam kd load: the stream is the adapter's live object, M4-KD seed 42, description == describe()",
          kd.stream is kd.adapter.nmf_stream and kd.stream.policy == "M4-KD" and kd.stream.seed == 42
          and kd.stream_description == kd.stream.describe() and kd.stream_description["draws"] == 0)
    check("h1 the stub checkpoint holds exactly the factory's state_dict (default and big factory); the strictly "
          "loaded stub's state equals the factory's own seeded state", case_stub_ckpt_exact(td, fx, tmp, kd))
    _, ev, checks_ev = gated_load(td, args(), mode="evaluator", factory=fx.stub_factory)
    check("seam evaluator load: M4-V stream, nmf_policy is the description, same loaded state as kd",
          ev.stream is ev.adapter.nmf_stream and ev.stream.policy == "M4-V"
          and ev.eval_model.nmf_policy == ev.stream_description
          and checks_ev["loaded_state_sha256"] == checks["loaded_state_sha256"])
    check("seam after-load: eval mode, nine frozen blob ids equal the table, stub parameters <= 1e6",
          kd.segmentor.training is False and checks["frozen_blob_ids"] == {k: v[0] for k, v in td.FROZEN.items()}
          and len(checks["frozen_blob_ids"]) == 9 and 0 < checks["parameters"] <= td.STUB_MAX_PARAMETERS,
          f"{len(checks['frozen_blob_ids'])} blobs, {checks['parameters']} parameters")
    _, kd2, checks2 = gated_load(td, args(), factory=fx.stub_factory)
    w = kd2.segmentor.decode_head.conv_seg.bias
    with torch.no_grad():
        w[0] += 1.0
    check("seam loaded_state_sha256: two loads agree; one changed weight changes it",
          checks2["loaded_state_sha256"] == checks["loaded_state_sha256"]
          and td.loaded_state_sha256(kd2.segmentor) != checks["loaded_state_sha256"])
    rec = td.teacher_record(kd, inputs, checks, stub=True)
    check("seam teacher_record: provenance is the loaded as_dict, sha verified, same_teacher(a, a) == []",
          rec["provenance"] == kd.frozen.provenance.as_dict() and rec["checkpoint"]["sha256_verified"] == sha
          and rec["nmf_stream_begin"]["policy"] == "M4-KD" and td.same_teacher(rec, rec) == [])
    other = dict(rec, loaded_state_sha256="0" * 64)
    check("seam same_teacher names the differing field", td.same_teacher(rec, other) == ["loaded_state_sha256"])

    # C1: a check against the dict misses a head call's draw; the object sees it
    split = td.SplitTeacher(kd)
    feats = split.features(x)
    h = td.feature_sha256(feats)
    desc = kd.stream_description
    d_before, o_before = desc["draws"], kd.stream.draws
    z1 = split.head(feats, kd.stream, feat_hash=h, expect_shape=(1, 116, 64, 64))
    check("C1 after one head call the dict still says draws == 0 (misses it), the object says 1",
          desc["draws"] == d_before == 0 and kd.stream.draws == o_before + 1 == 1)
    check("C1 handing the head check the dict instead of the object stops (identity is the object's)",
          raises(lambda: split.head(feats, desc, feat_hash=h), td.Stop) and kd.stream.draws == 1)

    # P9 split-forward checks
    check("P9 D1 shape [1, 116, 64, 64] from a 512x512 canvas, finite", tuple(z1.shape) == (1, 116, 64, 64))
    check("P9 a head call with no stream stops", raises(lambda: split.head(feats, None, feat_hash=h), td.Stop))
    try:
        split.head(feats, NMFStream(42, "M4-KD"), feat_hash=h)
        msg = "no stop"
    except td.Stop as ex:
        msg = str(ex)
    check("P9 a head call with a stream that is not the attached object stops (the identity check)",
          "not the expected stream object" in msg, msg)
    check("P9 an unexpected logit shape stops",
          raises(lambda: split.head(feats, kd.stream, feat_hash=h, expect_shape=(1, 116, 63, 64)), td.Stop))
    check("P9 a changed feature hash stops", raises(lambda: split.head(feats, kd.stream, feat_hash="0" * 64),
                                                     td.Stop))
    for name, factory, why in (("in-place-mutating head", fx.mutating_factory, "features changed"),
                               ("NaN head", fx.nan_factory, "non-finite"),
                               ("head drawing twice per call", fx.double_draw_factory, "by 2 draws"),
                               ("head drawing from the global CPU RNG", fx.global_draw_factory, "RNG")):
        _, bad, _ = gated_load(td, args(), factory=factory)
        sb = td.SplitTeacher(bad)
        fb = sb.features(x)
        try:
            sb.head(fb, bad.stream, feat_hash=td.feature_sha256(fb))
            ok, msg = False, "no stop"
        except td.Stop as e:
            ok, msg = why in str(e), str(e)
        check(f"P9 the {name} stops", ok, msg)
    rng0 = td.rng_state_sha256()
    split.head(feats, kd.stream, feat_hash=h)
    check("P9 a head call leaves the caller's CPU RNG state unchanged", td.rng_state_sha256() == rng0)

    # K = 1: the split forward is bit for bit the adapter's own forward
    _, a1, _ = gated_load(td, args(), factory=fx.stub_factory)
    _, a2, _ = gated_load(td, args(), factory=fx.stub_factory)
    s1 = td.SplitTeacher(a1)
    f1 = s1.features(x)
    z_split = s1.head(f1, a1.stream, feat_hash=td.feature_sha256(f1))
    z_full = a2.frozen(x).logits
    check("K=1 split forward == frozen(x).logits bit for bit; both streams end in the same state",
          torch.equal(z_split, z_full) and a1.stream.state_sha256() == a2.stream.state_sha256()
          and a1.stream.draws == a2.stream.draws == 1)

    # eight interleaved streams each equal themselves run alone (P10's draw discipline)
    _, il, _ = gated_load(td, args(), factory=fx.stub_factory)
    si = td.SplitTeacher(il)
    fs = [si.features(xi) for xi in xs]
    hs = [td.feature_sha256(f) for f in fs]
    streams = [NMFStream(42 + k, "M4-V") for k in range(8)]
    inter = {}
    for i, f in enumerate(fs):
        for k, s in enumerate(streams):
            si.attach(s)
            inter[(k, i)] = si.head(f, s, feat_hash=hs[i])
    ok, ends = True, []
    for k in range(8):
        alone = NMFStream(42 + k, "M4-V")
        for i, f in enumerate(fs):
            si.attach(alone)
            ok &= torch.equal(si.head(f, alone, feat_hash=hs[i]), inter[(k, i)])
        ends.append(alone.state_sha256() == streams[k].state_sha256() and alone.draws == streams[k].draws == 3)
    si.attach(il.stream)
    check("P10 each of eight interleaved streams equals itself run alone (logits and end state)", ok and all(ends))
    check("P10 different streams give different logits (the check is not vacuous)",
          not torch.equal(inter[(0, 0)], inter[(1, 0)]))

    # refusals: a stream-less model; P2 stub guards; P5 formats with the load counter at 0
    check("seam a model with no isolated NMF module is refused (begin returns None)",
          raises(lambda: gated_load(td, args(), factory=fx.streamless_factory), td.Refused))
    big, big_sha = fx.write_stub_ckpt(tmp / "big_stub.pth", pad_bytes=td.STUB_MAX_CKPT_BYTES)
    with fx.count_loads() as calls:
        r = raises(lambda: gated_load(td, args(teacher_ckpt=str(big), teacher_ckpt_sha256=big_sha),
                                      factory=fx.stub_factory), td.Refused)
    check("P2 stub refuses a checkpoint over 16 MiB by stat, before any load", r and sum(calls.values()) == 0,
          str(calls))
    with fx.patched(td, REGISTERED_TEACHER_SHA256=frozenset({sha})), fx.count_loads() as calls:
        r = raises(lambda: gated_load(td, args(), factory=fx.stub_factory), td.Refused)
    check("P2 stub refuses a registered teacher sha256, before any load", r and sum(calls.values()) == 0)
    pre = "mmseg" in sys.modules
    try:
        r = raises(lambda: gated_load(td, args(), factory=fx.mmseg_factory), td.Refused)
    finally:
        if not pre:
            sys.modules.pop("mmseg", None)
    check("P2 stub refuses a build that leaves mmseg imported", r and not pre, f"mmseg imported before: {pre}")
    big_ok, big_msg = case_big_model(td, fx, tmp, args)
    check("P2 stub refuses a model over 1e6 parameters: its own checkpoint (big_factory, <= 16 MiB) loads strictly, "
          "and the refusal names the parameter count (h2)", big_ok, big_msg)
    check("P5 the format check alone (no file read) refuses an empty, uppercase or 63-character sha256",
          all(raises(lambda b=b: td.check_teacher_flags(args(teacher_ckpt_sha256=b)), td.Refused)
              for b in ("", sha.upper(), sha[:63])))
    for label, bad in (("empty", ""), ("uppercase", sha.upper()), ("63 characters", sha[:63]),
                       ("a well-formed wrong sha256", "0" * 64)):
        with fx.count_loads() as calls:
            r = raises(lambda: gated_load(td, args(teacher_ckpt_sha256=bad), factory=fx.stub_factory), td.Refused)
        check(f"P5 --teacher-ckpt-sha256 {label} is refused with the load counter at 0",
              r and sum(calls.values()) == 0, str(calls))
    cfg_copy = Path(shutil.copy(td.REPO / td.TEACHER_CONFIG_REL, tmp / "teacher_cfg_copy.py"))
    with fx.count_loads() as calls:
        r1 = raises(lambda: gated_load(td, args(teacher_config=str(cfg_copy)), factory=fx.stub_factory), td.Refused)
        with fx.patched(td, RECORD_CONFIG_BLOB="0" * 40):
            r2 = raises(lambda: gated_load(td, args(), factory=fx.stub_factory), td.Refused)
    check("P3 record role: a config copy outside the repo and a wrong frozen blob are refused before any load",
          r1 and r2 and sum(calls.values()) == 0)
    arm = dict(teacher_role="arm", arm_id="R1", arm_dl_id="DL-60", teacher_config_sha256=td.file_sha256(cfg_copy),
               teacher_config=str(cfg_copy))
    _, arm_loaded, _ = gated_load(td, args(**arm), factory=fx.stub_factory)
    check("P3 arm role with its own config sha256 loads", arm_loaded.stream.policy == "M4-KD")
    with fx.patched(td, REGISTERED_TEACHER_SHA256=frozenset({sha})):
        r = raises(lambda: td.verify_teacher_inputs(args(**arm), stub=False), td.Refused)
    check("P3 the arm role refuses a record checkpoint sha256", r)
    check("P3 arm config sha256 mismatch is refused",
          raises(lambda: gated_load(td, args(**dict(arm, teacher_config_sha256="1" * 64)), factory=fx.stub_factory),
                 td.Refused))
    check("P3 the arm role requires --arm-id, --arm-dl-id and --teacher-config-sha256",
          raises(lambda: td.check_teacher_flags(args(**dict(arm, arm_id=None))), td.Refused)
          and raises(lambda: td.check_teacher_flags(args(**dict(arm, arm_dl_id="60"))), td.Refused)
          and raises(lambda: td.check_teacher_flags(args(**dict(arm, teacher_config_sha256=None))), td.Refused))
    check("P3 arm flags on the record role are refused",
          raises(lambda: td.check_teacher_flags(args(arm_id="R1")), td.Refused))
    check("P3 a role outside the script's roles is refused (TRAIN, D3, D1 VAL are record only)",
          raises(lambda: td.check_teacher_flags(args(**arm), roles=("record",)), td.Refused))
    with fx.count_loads() as calls:
        r = raises(lambda: gated_load(td, args(), stub=False, factory=fx.stub_factory), td.Refused)
    check("P5 real mode refuses a checkpoint that is not the teacher of record, before any load",
          r and sum(calls.values()) == 0)
    check("P5 after the load, a provenance sha256 other than the verified one is refused",
          raises(lambda: td.after_load_checks(kd, dataclasses.replace(inputs, sha256="0" * 64), stub=True),
                 td.Refused))
    check("P8 the field count: an 11-field TeacherProvenance refuses a real run, a 12-field one does not; stub "
          "runs never count (stand-in dataclasses, so the case holds before and after K-part)", case_p8_count(td))
    check("P8 a real run refuses a provenance whose fields are None or empty, naming them", case_p8_nonempty(td, kd, inputs))
    check("P8 the written record (as_dict) must hold every field: a record lacking one is refused",
          case_p8_record(td, kd, inputs, checks))
    check("P7 a frozen blob id other than the table's is refused", case_frozen_blob(td))
    check("P7 a segmentor in training mode stops", case_training_mode(td, fx, args))
    check("P9 a second isolated NMF module stops the after-load check", case_extra_isolated(td, fx, args))
    check("C1 a begin description that differs from stream.describe() stops the load", case_describe_mismatch(td, fx, args))
    check("C1 the stream is the adapter's live object after a fresh load", case_c1_object(td, fx, args))
    check("P29 exit codes: STOP classes 1, refusal classes 2, anything else 4 from main", case_exit_codes(td))

    # flags (C2, P26, P27, P28) and the commit binding
    ns_ = types.SimpleNamespace
    head = "c" * 40
    real_ok = ns_(generated_utc=None, script_commit=head, script_commit_dl_id="DL-61", repeat_of=None,
                  repeat_case=None, repeat_dl_id=None)
    check("C2 real mode refuses --generated-utc; stub mode takes a well-formed one and refuses a malformed one",
          raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), generated_utc="2026-10-02T00:00:00Z")),
                                               real=True), td.Refused)
          and not raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), generated_utc="2026-10-02T00:00:00Z")),
                                                       real=False), Exception)
          and raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), generated_utc="2026-10-02")),
                                                   real=False), td.Refused))
    check("P26 real mode requires --script-commit and --script-commit-dl-id, each well formed",
          not raises(lambda: td.check_common_flags(real_ok, real=True), Exception)
          and raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), script_commit=None)), real=True),
                     td.Refused)
          and raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), script_commit="C" * 40)), real=True),
                     td.Refused)
          and raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), script_commit_dl_id="61")), real=True),
                     td.Refused))
    check("P27 the repeat flags go together and are well formed",
          raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), repeat_of="a" * 64)), real=True), td.Refused)
          and raises(lambda: td.check_common_flags(ns_(**dict(vars(real_ok), repeat_of="a" * 64, repeat_case="iii",
                                                              repeat_dl_id="DL-1")), real=True), td.Refused))
    check("P28 correction flags: state in the three values, DL id ^DL-\\d+$",
          not raises(lambda: td.check_correction_flags(ns_(correction_state="declined", correction_dl_id="DL-62")),
                     Exception)
          and raises(lambda: td.check_correction_flags(ns_(correction_state="none", correction_dl_id="DL-62")),
                     td.Refused)
          and raises(lambda: td.check_correction_flags(ns_(correction_state="available", correction_dl_id="DL-")),
                     td.Refused))
    present = tuple(r for r in td.CODE_FILES if (td.REPO / r).is_file())
    with fx.patched(td, CODE_FILES=present):
        with fx.fake_git(head=head):
            prov = td.require_commit_binding(head, "DL-61")
        ok_bind = prov["script_commit"] == head and prov["code_files_clean_at_head"] is True
        with fx.fake_git(head="d" * 40):
            r_head = raises(lambda: td.require_commit_binding(head, "DL-61"), td.Refused)
        with fx.fake_git(head=head, status=" M src/eval/teacher_diag.py\n"):
            r_dirty = raises(lambda: td.require_commit_binding(head, "DL-61"), td.Refused)
        msg = ""
        with fx.fake_git(head=head, tree_status="?? scripts/a_new_name.py\n?? src/b_new_name.py\n"):
            try:
                td.require_commit_binding(head, "DL-61")
            except td.Refused as e:
                msg = str(e)
    with fx.patched(td, CODE_FILES=present + ("src/eval/no_such_module.py",)), fx.fake_git(head=head):
        r_missing = raises(lambda: td.require_commit_binding(head, "DL-61"), td.Refused)
    check("P26 commit binding: HEAD == flag and clean gives the provenance block", ok_bind)
    check("P26 HEAD other than --script-commit, a dirty CODE_FILES path and a missing one are refused",
          r_head and r_dirty and r_missing)
    check("P26 a dirty src/configs/scripts tree is refused with a count only, no names",
          "2 entr" in msg and "a_new_name" not in msg and "b_new_name" not in msg, msg)
    out = tmp / "outs"
    out.mkdir()
    kind = "teacher_d2"
    td.write_json_exclusive(out / f"{kind}_20261002T000000Z.json", {"artifact_status": "smoke"})
    none_yet = td.require_single_output(out, kind, real_ok)
    first = td.write_json_exclusive(out / f"{kind}_20261002T000100Z.json", {"artifact_status": "provisional"})
    r_second = raises(lambda: td.require_single_output(out, kind, real_ok), td.Refused)
    rep = ns_(**dict(vars(real_ok), repeat_of=td.file_sha256(first), repeat_case="i", repeat_dl_id="DL-63"))
    rep_bad = ns_(**dict(vars(rep), repeat_of="e" * 64))
    check("P27 smoke outputs do not count; a second non-smoke output is refused",
          none_yet is None and r_second)
    check("P27 --repeat-of naming the existing output passes and is recorded; another sha256 is refused",
          td.require_single_output(out, kind, rep) == {"repeat_of": rep.repeat_of, "repeat_case": "i",
                                                         "repeat_dl_id": "DL-63"}
          and raises(lambda: td.require_single_output(out, kind, rep_bad), td.Refused))
    check("P27 the control purpose is exempt", td.require_single_output(out, kind, real_ok, exempt=True) is None)
    check("P27 an out directory inside the repository (or the repository itself) is refused",
          raises(lambda: td.require_outside_repo(td.REPO / "runs_out", "--out-dir"), td.Refused)
          and raises(lambda: td.require_outside_repo(td.REPO, "--out-dir"), td.Refused)
          and td.require_outside_repo(out, "--out-dir") == out.resolve())

    # P21 path guards; JSON discipline; the document block (P30)
    link = tmp / "plain_link"
    os.symlink(tmp / ("x" + "TeSt" + "x"), link)           # the target is never created or read
    msgs = []
    for pth in (tmp / ("a" + "TEST"), link):
        try:
            td.refuse_test_path(pth, "--data-root")
            msgs.append(None)
        except td.Refused as e:
            msgs.append(str(e))
    check("P21 a 'test' path is refused by its string, and a link resolving to one by its real path",
          msgs[0] is not None and "containing" in msgs[0] and msgs[1] is not None and "resolving" in msgs[1]
          and "TEST" not in msgs[0] and "TeSt" not in msgs[1], str(msgs))
    try:
        td.refuse_test_names(["ok_a", "x" + "Test" + "_b"], "names")
        m = ""
    except td.Refused as e:
        m = str(e)
    check("P21 refused names print a count and sha256s, never the name", "1 offending entry" in m and "Test" not in m, m)
    p_nan = tmp / "nan.json"
    r_nan = raises(lambda: td.write_json_exclusive(p_nan, {"v": float("nan")}), ValueError) and not p_nan.exists()
    p_nan.write_text('{"v": NaN}')
    check("JSON: a NaN is never written (no file left) and a NaN constant is refused on read",
          r_nan and raises(lambda: td.read_json_strict(p_nan), td.Refused))
    check("JSON: writes are exclusive", raises(lambda: td.write_json_exclusive(first, {}), FileExistsError))
    a_stub = ns_(generated_utc="2026-10-02T01:02:03Z", x=1)
    d_stub = td.base_document("s.py", a_stub, stub=True, start_utc="2026-10-02T09:00:00Z", code={})
    d_real = td.base_document("s.py", ns_(x=1), stub=False, start_utc="2026-10-02T09:00:00Z", code={})
    check("P30/C2 stub documents are 'smoke' and may carry --generated-utc; real ones are 'provisional', "
          "stamped from the start UTC",
          d_stub["artifact_status"] == "smoke" and td.output_stamp(a_stub, "2026-10-02T09:00:00Z") == "20261002T010203Z"
          and d_real["artifact_status"] == "provisional" and d_real["generated_utc"] == "2026-10-02T09:00:00Z"
          and td.output_stamp(ns_(), "2026-10-02T09:00:00Z") == "20261002T090000Z")
    env = td.environment_block("2026-10-02T09:00:00Z")
    check("P30 environment block keys", {"hostname", "cpu_model", "torch_num_threads", "python", "torch", "numpy",
                                         "pillow", "image_digest", "start_utc", "end_utc"} <= set(env))


# ---------------------------------------------------------------------------------------------------
# the D1 script (scripts/teacher_d1_nmf_sensitivity.py): crops (acceptance a) and the VAL part (c)
# ---------------------------------------------------------------------------------------------------
UTC = "2026-10-02T00:00:00Z"
CORR = dict(correction_state="declined", correction_dl_id="DL-62")


def _d1_env():
    import json

    from scripts import teacher_d1_nmf_sensitivity as d1
    from scripts import teacher_d2_calibration as d2
    from scripts import teacher_diag_fixtures as fx
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    from src.eval import teacher_diag as td
    tmp = safe_tmpdir("diag_d1_")
    root, stems = fx.make_data_root("diag_d1_data_")
    fx.set_data_root(root)
    ckpt, sha = fx.write_stub_ckpt(tmp / "stub_teacher.pth")
    strata = fx.write_strata(tmp / "train_strata_v1.json", stems["train"])
    ref = fx.reference_artifact(tmp / "ref_artifact", ckpt, sha, n=3)
    cfg = str(td.REPO / td.TEACHER_CONFIG_REL)
    teacher = dict(teacher_ckpt=str(ckpt), teacher_ckpt_sha256=sha, teacher_config=cfg)
    fx.call_run(d2, out_dir=tmp / "d2", artifact_dir=tmp / "d2_art", teacher_role="record", purpose="item1",
                val_reference=str(ref), max_samples=3, generated_utc=UTC, **teacher)
    cfg_copy = Path(shutil.copy(cfg, tmp / "arm_cfg.py"))
    arm = dict(teacher_role="arm", arm_id="R1", arm_dl_id="DL-70", teacher_config=str(cfg_copy),
               teacher_config_sha256=td.file_sha256(cfg_copy))
    fx.call_run(d2, out_dir=tmp / "d2_arm", artifact_dir=tmp / "d2_arm_art", val_reference=str(ref), max_samples=3,
                generated_utc=UTC, **dict(teacher, **arm))
    arm2 = dict(arm, arm_id="R2")
    fx.call_run(d2, out_dir=tmp / "d2_arm2", artifact_dir=tmp / "d2_arm2_art", val_reference=str(ref), max_samples=3,
                generated_utc=UTC, **dict(teacher, **arm2))
    d2_json = fx.output_files(tmp / "d2")[0]
    return types.SimpleNamespace(
        d1=d1, d2=d2, fx=fx, td=td, tmp=tmp, root=root, stems=stems, ckpt=ckpt, sha=sha, strata=strata, ref=ref,
        teacher=teacher, arm=arm, d2_json=d2_json, d2_doc=json.loads(d2_json.read_text()),
        d2_arm_json=fx.output_files(tmp / "d2_arm")[0], d2_arm2_json=fx.output_files(tmp / "d2_arm2")[0])


def _stub_loaded(e, mode: str):
    a = types.SimpleNamespace(**e.teacher, teacher_role="record", arm_id=None, arm_dl_id=None, teacher_config_sha256=None)
    return e.td.load_teacher(mode, e.td.verify_teacher_inputs(a, stub=True), model_factory=e.fx.stub_factory)


def crop_feature_hashes(e, n: int) -> list:
    """Run-level wiring of crop i (TESTS-06): crops 0..n-1 rebuilt independently of the script -- the literal
    random.Random(1801).sample, train_preprocess with RandomState(1801 + i), finalize -- on a fresh stub load."""
    import random

    from PIL import Image

    from configs.augment import AUGMENT
    from scripts.build_train_strata import train_split_list
    from src.data.dataset import PlantSegDataset
    from src.data.transforms import finalize, train_preprocess
    td = e.td
    stems = [s for s, _ in train_split_list(e.root)]
    pairs = {p[0].stem: p for p in PlantSegDataset("train").pairs}
    split = td.SplitTeacher(_stub_loaded(e, "kd"))
    out = []
    for i, sid in enumerate(random.Random(1801).sample(sorted(stems), 256)[:n]):
        img_path, mask_path = pairs[sid]
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            image, _ = finalize(*train_preprocess(im, mk, np.random.RandomState(1801 + i), AUGMENT))
        out.append(td.feature_sha256(split.features(image[None])))
    return out


def mean_probability_miou(e, n: int = 3) -> float:
    """The D1 VAL part's mean-probability mIoU recomputed independently: eight fresh M4-V streams seeded 42..49,
    the resize, a float64 softmax summed over the draws, the evaluator's confusion and union-present reducer."""
    import torch
    import torch.nn.functional as F
    from src.distill.nmf_stream import NMFStream
    from src.eval.adapters import PlantSegEvalDataset, deterministic_subset
    from src.eval.metrics import confusion_matrix, miou_from_confusion
    td = e.td
    loaded = _stub_loaded(e, "evaluator")
    split = td.SplitTeacher(loaded)
    streams = [loaded.stream] + [NMFStream(s, "M4-V") for s in range(43, 50)]
    ds = PlantSegEvalDataset("val", deterministic_subset(td.VAL_ROWS, n))
    cm = torch.zeros(116, 116, dtype=torch.long)
    for k in range(len(ds)):
        item = ds[k]
        x = item["image"][None]
        feats = split.features(x)
        h = td.feature_sha256(feats)
        total = None
        for s in streams:
            split.attach(s)
            z = F.interpolate(split.head(feats, s, feat_hash=h), size=tuple(x.shape[-2:]), mode="bilinear",
                              align_corners=False)
            p = torch.softmax(z.to(torch.float64), dim=1)
            total = p if total is None else total + p
        cm += confusion_matrix(total.argmax(1)[0], item["target"], 116, 255)
    return miou_from_confusion(cm)


def _crops_cases(e) -> None:
    import json
    import random

    import torch
    from src.distill.nmf_stream import NMFStream
    from src.quant.calibration import build_calibration_index
    d1, fx, td = e.d1, e.fx, e.td

    # a1: the sample
    ids = ["a", "a-1", "a (1)"] + [f"id_{k:03d}" for k in range(300)]
    check("a1 sample == random.Random(1801).sample(sorted(ids), 256) == build_calibration_index; ids include "
          "'a', 'a-1', 'a (1)' (sorted: 'a' < 'a (1)' < 'a-1')",
          d1.sample_ids(ids) == random.Random(1801).sample(sorted(ids), 256)
          == build_calibration_index(ids, count=256, seed=1801)["selected_ids"]
          and sorted(["a", "a-1", "a (1)"]) == ["a", "a (1)", "a-1"])
    tr = e.stems["train"]
    check("a1 on the synthetic TRAIN stems the sample is the literal draw",
          d1.sample_ids(tr) == random.Random(1801).sample(sorted(tr), 256))

    # a2: crop i is the trainer's TRAIN branch with RandomState(1801 + i)
    from src.data.dataset import PlantSegDataset
    ds = PlantSegDataset("train")
    ok = True
    for i, src in ((0, 11), (5, 4000)):
        with fx.patched(np.random, randint=lambda *a, _s=1801 + i, **k: _s):
            img_t, mask_t = ds[src]
        mine_img, mine_mask = d1.crop(ds.pairs[src], i)
        ok &= torch.equal(img_t, mine_img) and torch.equal(mask_t, mine_mask)
    other = d1.crop(ds.pairs[11], 1)[0]
    check("a2 crop i equals PlantSegDataset('train')[src] with its global draw patched to 1801 + i "
          "(image and mask); seed 1802 gives another crop", ok and not torch.equal(other, d1.crop(ds.pairs[11], 0)[0]))
    m = torch.zeros(512, 512, dtype=torch.long)
    m[0, 0] = 255
    v = d1.valid_cells(m)
    check("a2 the domain is the all-valid min-pool on 64x64: one ignore pixel drops its cell only",
          v.shape == (64, 64) and not v[0, 0] and int(v.sum()) == 64 * 64 - 1)

    # the crops run, record role
    out = e.tmp / "d1_out"
    code, err = fx.call_run(d1, part="crops", out_dir=out, teacher_role="record", strata=str(e.strata),
                            d2_val_output=str(e.d2_json), n_crops=3, generated_utc=UTC, **CORR, **e.teacher)
    files = fx.output_files(out)
    doc = json.loads(files[0].read_text()) if files else {}
    check("a crops: exit 0, teacher_d1_<UTC>.json, status written, smoke",
          code == 0 and [p.name for p in files] == ["teacher_d1_20261002T000000Z.json"]
          and doc.get("status") == "written" and doc.get("artifact_status") == "smoke", err)
    c, nmf = doc.get("checks", {}), doc.get("nmf", {})
    check("a4 the backbone runs once per crop, the head 8 times", c.get("backbone_calls") == 3 and c.get("head_calls") == 24)
    ref_stream = NMFStream(42, "M4-KD")
    for _ in range(24):
        ref_stream.draw(lambda: torch.rand(1, 512, 1, 1))
    check("a5 stream draws == 8 x crops, and the end state equals a reference M4-KD stream after 24 draws",
          nmf.get("draws") == 24 and nmf.get("end", {}).get("state_sha256") == ref_stream.state_sha256()
          and nmf.get("begin", {}).get("draws") == 0)
    pc = doc.get("per_crop", [])
    check("a crop seeds 1801 + i in sample order; ids are the first n of the 256-draw",
          [p["seed"] for p in pc] == [1801, 1802, 1803] and [p["id"] for p in pc] == d1.sample_ids(tr)[:3])
    f = doc.get("F", {})
    check("a F pools the crops: N = sum N_i, D = 28 * sum |V_i|, branch booleans from the integers",
          f.get("N") == sum(p["N"] for p in pc) and f.get("D") == 28 * sum(p["n_valid"] for p in pc)
          and f.get("below_0_03") == (100 * f["N"] < 3 * f["D"]) and f.get("at_least_0_10") == (10 * f["N"] >= f["D"])
          and isinstance(f.get("ci95"), dict))
    kl = doc.get("KL_logit", {}).get("value")
    check("a KL_logit = fsum of per-crop sums / (8 * sum |V_i|)",
          kl is not None and kl == math.fsum(p["kl_logit_sum"] for p in pc) / (8 * sum(p["n_valid"] for p in pc)))
    check("P11 the D2 output's sha256 is recorded and names the same teacher",
          doc.get("inputs", {}).get("d2_val_output", {}).get("sha256") == td.file_sha256(e.d2_json)
          and td.same_teacher(doc.get("teacher", {}), e.d2_doc["teacher"]) == []
          and doc.get("correction") == {"state": "declined", "dl_id": "DL-62"})
    check("a2 run-level wiring: the run's per-crop feature hashes equal crops 0..2 recomputed with RandomState(1801 + i)",
          [p["feature_sha256"] for p in pc] == crop_feature_hashes(e, 3))

    # arm
    code, err = fx.call_run(d1, part="crops", out_dir=e.tmp / "d1_arm", strata=str(e.strata),
                            d2_val_output=str(e.d2_arm_json), n_crops=2, generated_utc=UTC, **CORR,
                            **dict(e.teacher, **e.arm))
    files = fx.output_files(e.tmp / "d1_arm")
    adoc = json.loads(files[0].read_text()) if files else {}
    check("P3 arm crops: teacher_d1_arm-R1_<UTC>.json, gates nothing, N, D and F without branch booleans",
          code == 0 and [p.name for p in files] == ["teacher_d1_arm-R1_20261002T000000Z.json"]
          and adoc.get("gates") == "nothing" and "below_0_03" not in adoc.get("F", {})
          and "N" in adoc.get("F", {}), err)
    code, err = fx.call_run(d1, part="crops", out_dir=e.tmp / "d1_arm_x", strata=str(e.strata),
                            d2_val_output=str(e.d2_arm2_json), n_crops=2, generated_utc=UTC, **CORR,
                            **dict(e.teacher, **e.arm))
    check("P11 an arm R1 run refuses arm R2's D2 output", code == 2 and "arm R1" in err, err)

    # refusals and stops
    base = dict(part="crops", teacher_role="record", strata=str(e.strata), d2_val_output=str(e.d2_json), n_crops=2,
                generated_utc=UTC, **CORR, **e.teacher)
    check("P11 the record role refuses an arm's D2 output", fx.call_run(
        d1, out_dir=e.tmp / "r1", **dict(base, d2_val_output=str(e.d2_arm_json)))[0] == 2)
    alt = json.loads(e.d2_json.read_text())
    alt["teacher"]["loaded_state_sha256"] = "0" * 64
    alt_p = e.tmp / "d2_other_teacher.json"
    alt_p.write_text(json.dumps(alt))
    code, _ = fx.call_run(d1, out_dir=e.tmp / "r2", **dict(base, d2_val_output=str(alt_p)))
    check("P11 a D2 output of another teacher (loaded_state_sha256) is refused after the load, nothing written",
          code == 2 and not fx.output_files(e.tmp / "r2"))
    prov_doc = json.loads(e.d2_json.read_text())
    prov_doc["artifact_status"] = "provisional"                 # only the status differs
    alt_p2 = e.tmp / "d2_provisional.json"
    alt_p2.write_text(json.dumps(prov_doc))
    code, err = fx.call_run(d1, out_dir=e.tmp / "r3", **dict(base, d2_val_output=str(alt_p2)))
    check("P2 stub mode refuses a non-smoke --d2-val-output (the smoke rule, not another check)",
          code == 2 and "stub mode takes only a smoke output" in err, err)
    try:
        td.read_diag_output(e.d2_json, script="scripts/teacher_d2_calibration.py", stub=False, what="--d2-val-output")
        m = ""
    except td.Refused as ex:
        m = str(ex)
    check("P2 real mode refuses a smoke --d2-val-output", "a real run refuses it" in m, m)
    bad_strata = fx.write_strata(e.tmp / "strata_other.json", e.stems["train"][:-1] + ["plant_leaf_zzzz"])
    with fx.count_loads() as calls:
        r_strata = fx.call_run(d1, out_dir=e.tmp / "r4", **dict(base, strata=str(bad_strata)))[0]
        fx.set_data_root(e.tmp / ("x" + "TeSt" + "x"))
        try:
            r_test = fx.call_run(d1, out_dir=e.tmp / "r5", **base)[0]
        finally:
            fx.set_data_root(e.root)
    check("a7 a strata split list other than the TRAIN list, and a 'test' data root, are refused before any load",
          (r_strata, r_test) == (2, 2) and sum(calls.values()) == 0)
    with fx.patched(d1, valid_cells=lambda mask: np.zeros((64, 64), dtype=bool)):
        code, _ = fx.call_run(d1, out_dir=e.tmp / "r6", **base)
    files = fx.output_files(e.tmp / "r6")
    zdoc = json.loads(files[0].read_text()) if files else {}
    check("P12 D == 0: status 'F not produced', booleans and branch null, exit 1",
          code == 1 and zdoc.get("status") == "F not produced" and zdoc.get("F", {}).get("below_0_03") is None
          and zdoc.get("F", {}).get("branch") is None)
    check("P31 a stream-less model is refused; a NaN head stops with nothing written",
          fx.call_run(d1, out_dir=e.tmp / "r7", factory=fx.streamless_factory, **base)[0] == 2
          and fx.call_run(d1, out_dir=e.tmp / "r8", factory=fx.nan_factory, **base)[0] == 1
          and not fx.output_files(e.tmp / "r8"))
    real = dict(base, generated_utc=None, n_crops=None, script_commit="c" * 40, script_commit_dl_id="DL-61")
    with fx.count_loads() as calls:
        cases = {"--correction-state": dict(real, correction_state=None, correction_dl_id=None),
                 "--correction-state must be": dict(real, correction_state="maybe"),
                 "--n-crops is a stub-mode flag": dict(real, n_crops=3),
                 "--teacher-ckpt-sha256 must match": dict(real, teacher_ckpt_sha256=None)}
        got = {why: fx.call_run(d1, factory=None, out_dir=e.tmp / f"r9_{k}", **kw) for k, (why, kw) in enumerate(cases.items())}
        r_p8 = fx.call_run(d1, factory=None, out_dir=e.tmp / "r11", **real)
    check("a6/P28 real weights are refused, each by its own gate (the message names it), before any load: no "
          "correction flags, a malformed state, --n-crops, no --teacher-ckpt-sha256",
          all(c == 2 and why in err for why, (c, err) in got.items()) and sum(calls.values()) == 0,
          str({why: err[-120:] for why, (c, err) in got.items() if why not in err}))
    check("P8 until K-part a well-formed real run is refused by the provenance field count",
          r_p8[0] == 2 and "TeacherProvenance" in r_p8[1], r_p8[1])
    h3_ok, h3_detail = case_reference_sha(fx, e.tmp, e.ckpt, e.sha)
    check("h3 the reference builder passes --teacher-ckpt-sha256: the evaluator builds it and records that sha; a "
          "wrong sha is refused (teacher_hash_mismatch)", h3_ok, h3_detail)


def _val_cases(e) -> None:
    import json

    import torch
    d1, fx, td = e.d1, e.fx, e.td
    out = e.tmp / "d1val_out"
    code, err = fx.call_run(d1, part="val", out_dir=out, teacher_role="record", val_reference=str(e.ref), max_samples=3,
                            generated_utc=UTC, **CORR, **e.teacher)
    files = fx.output_files(out)
    doc = json.loads(files[0].read_text()) if files else {}
    check("c VAL part: exit 0, teacher_d1val_<UTC>.json, status written, no evaluator artifact",
          code == 0 and [p.name for p in files] == ["teacher_d1val_20261002T000000Z.json"]
          and doc.get("status") == "written" and doc.get("pass", {}).get("evaluator_artifact") is None, err)
    nmf, ps = doc.get("nmf", {}), doc.get("per_stream", [])
    check("c1 eight streams seeded 42..49, each advanced one draw per image, end states distinct",
          nmf.get("seeds") == list(range(42, 50)) and nmf.get("draws") == [3] * 8
          and len(set(nmf.get("end_state_sha256", []))) == 8 and [r["seed"] for r in ps] == list(range(42, 50)))
    d2c = e.d2_doc["checks"]["nmf_stream"]
    check("c2 stream 42's draw sequence equals M4-V's: its end state equals the D2 pass's evaluator stream",
          nmf.get("end_state_sha256", [None])[0] == d2c["end"]["state_sha256"]
          and nmf.get("stream_42_begin") == d2c["begin"])
    cal = e.d2_doc["calibration"]
    check("c2 stream 42 scores like the evaluator's TeacherEvalModel: mIoU, ECE, n and correct equal D2's",
          ps and ps[0]["all_class_miou"] == e.d2_doc["pass"]["all_class_miou"] and ps[0]["ece"] == cal["ece"]
          and ps[0]["n"] == cal["n"] and ps[0]["correct"] == cal["correct"])
    check("c2 the evaluator core's confusion equals stream 42's; stream 42 re-attached; caller RNG unchanged",
          doc.get("checks", {}).get("core_confusion_equals_stream_42") is True
          and doc.get("checks", {}).get("stream_42_reattached") is True
          and doc.get("checks", {}).get("rng_state_unchanged") is True
          and doc.get("checks", {}).get("backbone_calls") == 3 and doc.get("checks", {}).get("head_calls") == 24)
    import statistics
    mi = [r["all_class_miou"] for r in ps]
    sp = doc.get("spread", {}).get("all_class_miou") or {}
    check("c4 spread over the eight streams: fsum mean, sample SD (ddof 1), range",
          sp.get("mean") == math.fsum(mi) / 8 and abs(sp.get("sd", -1) - statistics.stdev(mi)) <= 1e-15
          and sp.get("range") == max(mi) - min(mi))
    check("c4 the gate passed against the reference (delta 0.0) and the mean-probability mIoU is written",
          doc.get("reproduction", {}).get("passed") is True and doc.get("reproduction", {}).get("delta") == 0.0
          and isinstance(doc.get("mean_probability", {}).get("all_class_miou"), float))

    mp_ref = mean_probability_miou(e)
    check("c3 run-level wiring: the mean-probability mIoU equals an independent eight-stream recomputation "
          "(and differs from stream 42's alone)",
          doc.get("mean_probability", {}).get("all_class_miou") == mp_ref and ps and mp_ref != ps[0]["all_class_miou"],
          f"{doc.get('mean_probability', {}).get('all_class_miou')!r} vs {mp_ref!r}")
    import src.eval.metrics as em
    real_cm = em.confusion_matrix

    def skewed_cm(pred, target, num_classes, ignore_index=255):
        cm = real_cm(pred, target, num_classes, ignore_index)
        cm[0, 0] += 1
        return cm
    with fx.patched(em, confusion_matrix=skewed_cm):
        code, err = fx.call_run(d1, part="val", out_dir=e.tmp / "v_cm", teacher_role="record", val_reference=str(e.ref),
                                max_samples=3, generated_utc=UTC, **CORR, **e.teacher)
    check("c2 a stream-42 confusion that differs from the evaluator core's stops (exit 1), nothing written",
          code == 1 and "confusion differs" in err and not fx.output_files(e.tmp / "v_cm"), err)

    # c3: probabilities are averaged, not logits
    dk = [5.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    mp = d1.MeanProbability()
    for d in dk:
        mp.add(torch.tensor([[[[0.0]], [[d]]]]))
    logit_mean = torch.tensor([0.0, sum(dk) / 8]).argmax().item()
    check("c3 the mean-probability prediction averages probabilities: class 0, where the mean logit says 1",
          int(mp.prediction().reshape(-1)[0]) == 0 and logit_mean == 1 and mp.count == 8)

    # c4: the gate fails -> "not reproduced", statistics withheld
    with fx.patched(td, R3_TOLERANCE=-1.0):
        code, _ = fx.call_run(d1, part="val", out_dir=e.tmp / "v_gate", teacher_role="record", val_reference=str(e.ref),
                              max_samples=3, generated_utc=UTC, **CORR, **e.teacher)
    files = fx.output_files(e.tmp / "v_gate")
    gdoc = json.loads(files[0].read_text()) if files else {}
    check("c4 stream 42 not reproducing the reference: status 'not reproduced', statistics withheld, exit 1",
          code == 1 and gdoc.get("status") == "not reproduced" and "per_stream" not in gdoc
          and "spread" not in gdoc and gdoc.get("reproduction", {}).get("passed") is False)

    base = dict(part="val", teacher_role="record", val_reference=str(e.ref), max_samples=3, generated_utc=UTC,
                **CORR, **e.teacher)
    with fx.count_loads() as calls:
        r_arm = fx.call_run(d1, out_dir=e.tmp / "v1", **dict(base, **e.arm))[0]
        r_strata = fx.call_run(d1, out_dir=e.tmp / "v2", **dict(base, strata=str(e.strata)))[0]
        r_rows = fx.call_run(d1, out_dir=e.tmp / "v3", **dict(base, max_samples=2))[0]
        real = dict(base, max_samples=None, generated_utc=None, script_commit="c" * 40, script_commit_dl_id="DL-61")
        r_nocorr = fx.call_run(d1, factory=None, out_dir=e.tmp / "v4",
                               **dict(real, correction_state=None, correction_dl_id=None))
        r_ms = fx.call_run(d1, factory=None, out_dir=e.tmp / "v5", **dict(real, max_samples=3))
    check("P3/P28 the VAL part refuses the arm role, --strata, another row count, and real runs without the "
          "correction flags or with --max-samples (each by its own gate) -- before any load",
          (r_arm, r_strata, r_rows, r_nocorr[0], r_ms[0]) == (2, 2, 2, 2, 2) and "--correction-state" in r_nocorr[1]
          and "--max-samples is a stub-mode flag" in r_ms[1] and sum(calls.values()) == 0, r_nocorr[1] + r_ms[1])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m_real = d1.main(fx.argv(out_dir=e.tmp / "v6", **dict(base, max_samples=None, generated_utc=None,
                                                                script_commit="c" * 40, script_commit_dl_id="DL-61")))
    check("P29 the CLI is real mode and refuses until K-part (exit 2)", m_real == 2)


def d1_cases() -> None:
    e = _d1_env()
    try:
        for fn in (_crops_cases, _val_cases):
            try:
                fn(e)
            except Exception as exc:  # noqa: BLE001 -- a crash is a failed case
                check(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(e.tmp, ignore_errors=True)
        shutil.rmtree(e.root.parent, ignore_errors=True)


SECTIONS = [stats_cases, seam_cases, d1_cases]


def main() -> int:
    for fn in SECTIONS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- a crash is a failed case, never a pass
            check(f"{fn.__name__} raised", False, f"{type(e).__name__}: {e}")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    print(f"RESULT: {'PASS' if good == len(RESULTS) else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
