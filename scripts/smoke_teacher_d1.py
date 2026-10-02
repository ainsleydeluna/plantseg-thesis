#!/usr/bin/env python3
"""Smoke: D1, NMF-draw sensitivity of the distillation targets (lane L-TEACHER-DIAG; AM-18 item 1(a)).

Acceptance (b), the statistics of src/eval/nmf_sensitivity.py on crafted logits with hand-computed
answers (P12-P15, P32(b), P33): N, D and both integer branch booleans, including both boundaries
(100*N == 3*D and 10*N == D), D == 0, the two big-integer pairs where a float rule would decide the
other way, and the exact tie (5, 5, 0); F_lesion with its undefined case; F_halves; KL_logit and KL_cwd
against an independent numpy reference (1e-12; identity cases 1e-15) and the CWD logit-map construction;
the bootstrap vector of P32(b), the index sha256 of P15, identical replicates across runs, a replicate
with sum D == 0, and SciPy's percentile bootstrap; d2_crops; empty crops add nothing.
Synthetic inputs only; no PlantSeg data, no checkpoint of record, no GPU.

    python -B scripts/smoke_teacher_d1.py
"""
from __future__ import annotations

import math
import sys
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
        check("b6 SciPy percentile bootstrap: replicates bit for bit, bounds within 1e-15",
              np.array_equal(np.asarray(res.bootstrap_distribution), reps)
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
    check("b7 share > 0.99 is strict and matches numpy",
          s["n_max_prob_gt_0_99_t1"] == int(((e1 / e1.sum(0)).max(0) > 0.99).sum()))
    empty = ns.crop_statistics(z, np.zeros((4, 5), bool))
    a, b = ns.summarize([cs]), ns.summarize([cs, empty])
    check("b7 a crop with no valid cell adds nothing",
          all(a[k] == b[k] for k in ("F", "F_lesion", "F_halves", "KL_logit", "KL_cwd", "d2_crops")))


SECTIONS = [stats_cases]


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
