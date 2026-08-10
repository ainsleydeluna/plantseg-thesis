#!/usr/bin/env python3
"""A3a — deterministic paired-inference smoke (NONOFFICIAL).

Verifies docs/STATISTICAL_ANALYSIS_CONTRACT.md sections 1-7 and 10 against synthetic evaluation
artifacts built with the proven A2a writer. No dataset, no checkpoint, no model, no GPU, no
bootstrap.

NONOFFICIAL: the dev stack differs from the pinned statistics stack (contract section 10.1), so
nothing here may be presented as a thesis result.

Run:  set PYTHONIOENCODING=utf-8 && python -B scripts/smoke_stats_paired.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                       # noqa: E402
import scipy                             # noqa: E402
import scipy.stats as st                 # noqa: E402
import statsmodels                       # noqa: E402

from src.eval.artifacts import (DatasetMeta, RunMeta, prepare_artifact_request,  # noqa: E402
                                validate_artifact_request, write_artifact)
from src.eval.evaluate import Condition, EvalResult, ManifestEntry, PerImageRow  # noqa: E402
from src.stats import (ALPHA, CANONICAL_COMPARISON_IDS, AlignmentError,  # noqa: E402
                       CorruptionGrid, IngestError, Policy, RobustnessError, StatsError,
                       align_miou_c, align_runs, assemble_miou_c, cohens_dz, engineering_shifts,
                       hodges_lehmann, holm_audit, holm_family, load_run, paired_t_test,
                       rank_biserial, run_comparison, wilcoxon_test)

C = 116
CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def expect(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}") from e
    raise AssertionError("expected an exception, none raised")


# --------------------------------------------------------------------------------------------------
# synthetic artifact factory (uses the proven A2a writer)
# --------------------------------------------------------------------------------------------------
SYN_CLASS_MAP = [{"class_id": 0, "name": "", "role": "background_or_non_disease"}] + \
                [{"class_id": c, "name": f"syn_{c:03d}", "role": "disease"} for c in range(1, C)]


def make_artifact(out_dir: Path, ids, disease_vals, *, stage="E1", condition=("clean", None, None),
                  status="smoke", clean_ids=None, all_vals=None, undefined=()):
    """Build a synthetic evaluation artifact with EXACT per-image values.

    The EvalResult is constructed directly (rather than via inference) so per-image values are
    controllable; the artifact is still written and validated by the unmodified A2a writer.
    Sufficient statistics are internally consistent but intentionally simple -- this is a
    synthetic fixture, not a scored model.
    """
    clean_ids = list(clean_ids or ids)
    all_vals = list(all_vals if all_vals is not None else disease_vals)
    ctype, cname, csev = condition
    cond = Condition(ctype, cname, csev)
    manifest = [ManifestEntry(i, ids[i], clean_ids[i]) for i in range(len(ids))]

    rows, s_img, s_cls, s_tp, s_gt, s_pr = [], [], [], [], [], []
    for i, iid in enumerate(ids):
        und = iid in undefined
        rows.append(PerImageRow(
            image_id=iid, clean_image_id=clean_ids[i], manifest_index=i, condition=cond,
            all_class_miou=None if und else float(all_vals[i]),
            all_class_miou_status="undefined_no_eligible_class" if und else "ok",
            disease_only_miou=None if und else float(disease_vals[i]),
            disease_only_miou_status="undefined_no_eligible_class" if und else "ok",
            n_eligible_all_class=0 if und else 2, n_eligible_disease_only=0 if und else 1,
            gt_disease_classes=[] if und else [1]))
        if not und:
            for cid, (tp, gt, pr) in ((0, (8, 10, 9)), (1, (2, 3, 4))):
                s_img.append(i); s_cls.append(cid); s_tp.append(tp); s_gt.append(gt); s_pr.append(pr)

    d_tp = np.zeros(C, np.int64); d_gt = np.zeros(C, np.int64); d_pr = np.zeros(C, np.int64)
    np.add.at(d_tp, np.array(s_cls, np.int64), np.array(s_tp, np.int64))
    np.add.at(d_gt, np.array(s_cls, np.int64), np.array(s_gt, np.int64))
    np.add.at(d_pr, np.array(s_cls, np.int64), np.array(s_pr, np.int64))
    un = d_gt + d_pr - d_tp
    el_u, el_g = un > 0, d_gt > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(el_u, d_tp / np.maximum(un, 1), np.nan)
        dice = np.where((d_gt + d_pr) > 0, 2 * d_tp / np.maximum(d_gt + d_pr, 1), np.nan)
        acc = np.where(el_g, d_tp / np.maximum(d_gt, 1), np.nan)

    res = EvalResult(
        rows=rows, manifest_ids=list(ids),
        sparse_image_index=np.array(s_img, np.int64), sparse_class_id=np.array(s_cls, np.int64),
        sparse_tp=np.array(s_tp, np.int64), sparse_gt=np.array(s_gt, np.int64),
        sparse_pred=np.array(s_pr, np.int64),
        dataset_tp=d_tp, dataset_gt=d_gt, dataset_pred=d_pr,
        dataset_level={"all_class_miou": float(np.nanmean(iou[el_u])),
                       "all_class_miou_n_eligible": int(el_u.sum()),
                       "all_class_macro_dice": float(np.nanmean(dice[el_u])),
                       "all_class_dice_n_eligible": int(el_u.sum()),
                       "all_class_macc": float(np.nanmean(acc[el_g])),
                       "all_class_macc_n_eligible": int(el_g.sum()),
                       "disease_only_miou": float(iou[1]), "disease_only_miou_n_eligible": 1,
                       "aacc_diagnostic": float(d_tp.sum() / d_gt.sum())},
        per_class={"class_ids": list(range(C)), "gt_support": d_gt.tolist(),
                   "pred_support": d_pr.tolist(), "intersection": d_tp.tolist(),
                   "union": un.tolist(),
                   "iou": [None if not el_u[c] else float(iou[c]) for c in range(C)],
                   "dice": [None if not el_u[c] else float(dice[c]) for c in range(C)],
                   "acc": [None if not el_g[c] else float(acc[c]) for c in range(C)],
                   "iou_status": ["ok" if el_u[c] else "undefined_absent_from_gt_and_pred"
                                  for c in range(C)]},
        integrity={"row_count_ok": True, "ids_unique": True, "ids_match_manifest": True,
                   "order_canonical": True, "duplicate_ids": [], "missing_ids": [],
                   "invalid_pred_labels": 0, "undefined_per_image_scores": len(undefined),
                   "overwrite_policy": "refuse_existing"},
        num_classes=C, background_index=0, ignore_index=255, condition=cond)

    req = prepare_artifact_request(
        out_dir=out_dir, artifact_status=status, run_id=f"syn_{stage}_{cname or 'clean'}_{csev}",
        run=RunMeta(stage=stage, model_role="student", precision="fp32", random_init=True),
        dataset=DatasetMeta(name="SYNTHETIC stats fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split="val", condition=cond,
                            preprocess_protocol="synthetic-stats/1.0.0", expected_rows=len(ids)),
        expected_manifest=manifest, class_map=SYN_CLASS_MAP, repo_root=REPO)
    prov = validate_artifact_request(req)
    return write_artifact(res, req, prov, timestamp_utc="2026-07-28T00:00:00Z")


# --------------------------------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    print("=" * 100)
    print("A3a -- deterministic paired inference smoke  [NONOFFICIAL: dev stack != pinned stack]")
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | scipy {scipy.__version__} "
          f"| statsmodels {statsmodels.__version__}")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="a3a_"))
    try:
        # ---------------- Wilcoxon reference (pos, neg, zero, tied) ----------------
        d = np.array([0.10, -0.05, 0.0, 0.10, -0.05, 0.30, 0.0, 0.20], dtype=np.float64)
        w = wilcoxon_test(d)
        ref = st.wilcoxon(d, zero_method="pratt", correction=True,
                          alternative="greater", method="approx")
        check("W1 Wilcoxon matches a direct SciPy call with the frozen arguments",
              w.status == "ok" and abs(w.statistic - float(ref.statistic)) < 1e-12
              and abs(w.zstatistic - float(ref.zstatistic)) < 1e-12
              and abs(w.p_value - float(ref.pvalue)) < 1e-15,
              f"W={w.statistic} z={w.zstatistic:.6f} p={w.p_value:.6g} (zeros={w.n_zero}, "
              f"nonzero={w.n_nonzero})")
        check("W2 vector with zeros AND ties is classified VALID (not degenerate)",
              w.status == "ok", w.status)

        # ---------------- tied-nonzero validity ----------------
        tied = np.array([0.2, -0.2, 0.2, 0.2, -0.2], dtype=np.float64)
        wt = wilcoxon_test(tied)
        check("W3 all nonzero magnitudes tied -> still VALID",
              wt.status == "ok" and wt.p_value is not None,
              f"status={wt.status} p={wt.p_value:.6g}")

        # ---------------- all-zero ----------------
        z = np.zeros(6)
        wz, tz, rz, dzz = wilcoxon_test(z), paired_t_test(z, z), rank_biserial(z), cohens_dz(z)
        check("W4 all-zero: p=1.0, W/z null, status degenerate_all_zero",
              wz.p_value == 1.0 and wz.statistic is None and wz.zstatistic is None
              and wz.status == "degenerate_all_zero")
        check("W5 all-zero t-test: t null, p=1.0", tz.statistic is None and tz.p_value == 1.0
              and tz.status == "degenerate_all_zero")
        check("W6 all-zero effect sizes: r_rb null, dz null, HL/mean/median = 0",
              rz.value is None and rz.status == "undefined_no_nonzero_rank_sum"
              and dzz.value is None and hodges_lehmann(z) == 0.0
              and engineering_shifts(z).mean_delta == 0.0
              and engineering_shifts(z).median_delta == 0.0)

        # ---------------- insufficient nonzero ----------------
        one = np.array([0.0, 0.0, 0.5, 0.0])
        w1 = wilcoxon_test(one)
        check("W7 exactly one nonzero -> insufficient_nonzero_pairs, no conclusion",
              w1.status == "insufficient_nonzero_pairs" and w1.p_value is None
              and w1.zstatistic is None)

        # ---------------- t-test ----------------
        base = np.array([0.1, 0.2, 0.3, 0.4]); cand = base + np.array([0.05, 0.02, 0.09, 0.01])
        tt = paired_t_test(cand, base)
        tref = st.ttest_rel(cand, base, alternative="greater")
        check("T1 t-test matches a direct SciPy call",
              abs(tt.statistic - float(tref.statistic)) < 1e-12
              and abs(tt.p_value - float(tref.pvalue)) < 1e-15 and tt.df == 3,
              f"t={tt.statistic:.6f} p={tt.p_value:.6g} df={tt.df}")
        cc = np.array([0.3, 0.3, 0.3, 0.3])
        tc = paired_t_test(cc, np.zeros(4))
        check("T2 constant NONZERO differences -> no Infinity, status recorded",
              tc.statistic is None and tc.p_value is None
              and tc.status == "degenerate_zero_variance_nonzero" and tc.mean_difference == 0.3)

        # ---------------- rank-biserial hand calculation ----------------
        # d = [+3, +1, 0, -1, +2]; |d| = [3,1,0,1,2]; sorted -> 0,1,1,2,3
        # ranks: 0->1 ; the two 1s -> (2+3)/2 = 2.5 ; 2 -> 4 ; 3 -> 5
        # R+ = 5 (from +3) + 2.5 (from +1) + 4 (from +2) = 11.5 ; R- = 2.5 (from -1)
        # r_rb = (11.5 - 2.5) / 14.0 = 9/14
        hand = np.array([3.0, 1.0, 0.0, -1.0, 2.0])
        rb = rank_biserial(hand)
        exp_rb = 9.0 / 14.0
        wrong = (11.5 - 2.5) / (5 * 6 / 2)                     # n(n+1)/2 = 15 -> 0.6
        check("R1 rank-biserial hand check: R+=11.5, R-=2.5, denom=14, r_rb=9/14",
              abs(rb.value - exp_rb) < 1e-12 and rb.r_plus == 11.5 and rb.r_minus == 2.5
              and rb.denominator == 14.0,
              f"r_rb={rb.value:.9f} (expected {exp_rb:.9f})")
        check("R2 the rejected n(n+1)/2 denominator gives a DIFFERENT, contractually wrong value",
              abs(wrong - 0.6) < 1e-12 and abs(wrong - exp_rb) > 1e-3,
              f"n(n+1)/2 -> {wrong:.9f} vs frozen {exp_rb:.9f}")
        check("R3 r_rb stays within [-1, 1]", -1.0 <= rb.value <= 1.0)

        # ---------------- Hodges-Lehmann hand calculation ----------------
        # [1,2]   -> Walsh (i<=j): 1.0, 1.5, 2.0            -> 3 values (ODD)  -> median 1.5
        # [1,2,4] -> Walsh: 1,1.5,2.5,2,3,4 -> sorted 1,1.5,2,2.5,3,4 -> 6 (EVEN) -> (2+2.5)/2=2.25
        check("H1 Hodges-Lehmann, ODD Walsh count (n=2 -> 3 averages) = 1.5",
              hodges_lehmann(np.array([1.0, 2.0])) == 1.5)
        check("H2 Hodges-Lehmann, EVEN Walsh count (n=3 -> 6 averages) = 2.25",
              hodges_lehmann(np.array([1.0, 2.0, 4.0])) == 2.25)
        check("H3 HL differs from the plain median of differences",
              hodges_lehmann(np.array([1.0, 2.0, 4.0])) != float(np.median([1.0, 2.0, 4.0])))

        # ---------------- Holm ----------------
        ps = [0.0001, 0.004, 0.02, 0.03, 0.04, 0.2, 0.5, 0.9]
        pairs = list(zip(CANONICAL_COMPARISON_IDS, ps))
        audit = holm_audit(pairs, ALPHA)
        from statsmodels.stats.multitest import multipletests
        lrej, ladj, _, _ = multipletests(ps, alpha=ALPHA, method="holm")
        check("F1 Holm audit preserves canonical order",
              [m.comparison_id for m in audit] == list(CANONICAL_COMPARISON_IDS))
        check("F2 Holm audit reject pattern matches statsmodels on non-boundary inputs",
              [m.reject for m in audit] == list(lrej), f"{[m.reject for m in audit]}")
        # strict boundary: p exactly alpha/8 at rank 1
        b = holm_audit([(CANONICAL_COMPARISON_IDS[0], ALPHA / 8)] +
                       [(c, 0.9) for c in CANONICAL_COMPARISON_IDS[1:]], ALPHA)
        check("F3 STRICT boundary: p_raw exactly == step threshold does NOT reject",
              b[0].reject is False and b[0].step_threshold == ALPHA / 8,
              f"p={b[0].p_raw} thr={b[0].step_threshold} reject={b[0].reject}")
        # tied p-values -> stable order by (p, comparison_id)
        tie_pairs = [(c, 0.01) for c in CANONICAL_COMPARISON_IDS]
        ta = holm_audit(tie_pairs, ALPHA)
        ranks = {m.comparison_id: m.sorted_rank for m in ta}
        check("F4 tied p-values ranked deterministically by (p, comparison_id)",
              [c for c in sorted(ranks, key=lambda k: ranks[k])] ==
              sorted(CANONICAL_COMPARISON_IDS))

        # ---------------- synthetic artifacts, ingestion, alignment ----------------
        ids = [f"syn_{i:03d}" for i in range(8)]
        b_vals = [0.10, 0.20, 0.30, 0.40, 0.15, 0.25, 0.35, 0.45]
        c_vals = [0.20, 0.22, 0.29, 0.55, 0.15, 0.40, 0.30, 0.60]
        a_base = make_artifact(work / "base", ids, b_vals, stage="E1")
        a_cand = make_artifact(work / "cand", ids, c_vals, stage="E2")
        rb_run = load_run(a_base, Policy.NONOFFICIAL_SMOKE)
        rc_run = load_run(a_cand, Policy.NONOFFICIAL_SMOKE)
        check("I1 artifact ingested and manifest verified", rb_run.identity.stage == "E1"
              and len(rb_run.records) == 8)
        check("I2 OFFICIAL policy rejects a smoke/random-init artifact",
              "artifact_status='official'" in str(expect(IngestError, load_run, a_base,
                                                         Policy.OFFICIAL)))
        pv = align_runs(rb_run, rc_run, policy=Policy.NONOFFICIAL_SMOKE)
        check("A1 alignment produced candidate - baseline in canonical ID order",
              list(pv.image_ids) == sorted(ids)
              and np.allclose(pv.delta, np.array(c_vals) - np.array(b_vals)))

        # tampering + malformed
        tp = a_base / "per_image.jsonl"
        orig = tp.read_bytes()
        tp.write_bytes(orig + b'{"x":1}\n')
        check("I3 tampered payload rejected by manifest verification",
              isinstance(expect(IngestError, load_run, a_base, Policy.NONOFFICIAL_SMOKE),
                         IngestError))
        tp.write_bytes(orig)
        mp = a_base / "MANIFEST.sha256"
        morig = mp.read_bytes()
        mp.write_bytes(b"ZZZZ  per_image.jsonl\n")
        check("I4 malformed manifest rejected",
              isinstance(expect(IngestError, load_run, a_base, Policy.NONOFFICIAL_SMOKE),
                         IngestError))
        mp.write_bytes(morig)

        # ID mismatch / duplicate / case-fold / wrong clean map
        # NOTE: a different image_id set necessarily yields a different split_manifest_sha256, so
        # the compatibility guard fires BEFORE the ID-set guard. That ordering is correct; the
        # ID-set guard itself is exercised directly through align_vectors below.
        a_mis = make_artifact(work / "mis", [f"other_{i:03d}" for i in range(8)], b_vals)
        e_mis = expect(AlignmentError, align_runs, rb_run,
                       load_run(a_mis, Policy.NONOFFICIAL_SMOKE),
                       policy=Policy.NONOFFICIAL_SMOKE)
        check("A2 differing image_id sets rejected (compatibility guard fires first)",
              isinstance(e_mis, AlignmentError), str(e_mis)[:110])
        from src.stats import align_vectors
        e_set = expect(AlignmentError, align_vectors, ["a", "b"], [0.1, 0.2], ["a", "c"],
                       [0.1, 0.2], policy=Policy.NONOFFICIAL_SMOKE, metric="m")
        check("A2b ID-set-equality guard reached directly: no intersection fallback",
              "not equal" in str(e_set) and "Intersection is forbidden" in str(e_set),
              str(e_set)[:110])
        check("A2c duplicate ID rejected by the same guard",
              "duplicate image_id" in str(expect(AlignmentError, align_vectors, ["a", "a"],
                                                 [0.1, 0.2], ["a", "a"], [0.1, 0.2],
                                                 policy=Policy.NONOFFICIAL_SMOKE, metric="m")))
        check("A2d case-folded collision rejected by the same guard",
              "case-folded" in str(expect(AlignmentError, align_vectors, ["a", "A"], [0.1, 0.2],
                                          ["a", "A"], [0.1, 0.2],
                                          policy=Policy.NONOFFICIAL_SMOKE, metric="m")))
        cf_ids = list(ids); cf_ids[1] = ids[0].upper()
        a_cf = make_artifact(work / "cf", cf_ids, b_vals)
        check("A3 case-folded ID collision rejected at ingestion",
              isinstance(expect((AlignmentError, IngestError), load_run, a_cf,
                                Policy.NONOFFICIAL_SMOKE), (AlignmentError, IngestError)))
        a_wc = make_artifact(work / "wc", ids, b_vals, clean_ids=[f"wrong_{i}" for i in range(8)])
        e_wc = expect(AlignmentError, align_runs, rb_run,
                      load_run(a_wc, Policy.NONOFFICIAL_SMOKE),
                      policy=Policy.NONOFFICIAL_SMOKE)
        check("A4 wrong clean_image_id mapping rejected", isinstance(e_wc, AlignmentError),
              str(e_wc)[:110])

        # ---------------- mIoU-C assembly ----------------
        # A3a deliberately exercises the NONOFFICIAL injected-grid path with a synthetic
        # vocabulary. The official identifiers ARE frozen (A3b-0, configs/corruption_protocol.json)
        # and are validated by src/stats/corruption_protocol.py; this smoke is not the official
        # corruption-scaffold execution and never produces an official artifact.
        GRID = CorruptionGrid(names=("synA", "synB", "synC", "synD", "synE"),
                              provenance="SYNTHETIC nonofficial fixture vocabulary -- the official "
                                         "identifiers are frozen separately in the corruption "
                                         "protocol and can never come from this fixture")
        cells = {"synA": [0.10, 0.20, 0.30], "synB": [0.40, 0.40, 0.40],
                 "synC": [0.00, 0.30, 0.60], "synD": [0.50, 0.60, 0.70],
                 "synE": [0.10, 0.10, 0.40]}
        cids = [f"clean_{i:02d}" for i in range(4)]

        _tag = [0]

        def grid_runs(stage, bump=0.0, skip=None, dup=None, sev_override=None, undef=False):
            """Each invocation writes to its own directory namespace (artifacts refuse overwrite)."""
            _tag[0] += 1
            tag = _tag[0]
            runs = []
            for cname, vals in cells.items():
                for k, sev in enumerate((1, 2, 3)):
                    if skip == (cname, sev):
                        continue
                    s = sev_override if (sev_override and cname == "synA" and sev == 3) else sev
                    v = [min(1.0, vals[k] + bump + 0.01 * j) for j in range(len(cids))]
                    d = work / f"g{tag}_{stage}_{cname}_{s}_{k}"
                    runs.append(load_run(
                        make_artifact(d, cids, v, stage=stage,
                                      condition=("corruption", cname, s),
                                      undefined=(cids[0],) if (undef and cname == "synA"
                                                               and sev == 1) else ()),
                        Policy.NONOFFICIAL_SMOKE))
            if dup:
                cname, sev = dup
                runs.append(load_run(make_artifact(work / f"g{tag}_{stage}_dup", cids,
                                                   [0.5] * len(cids), stage=stage,
                                                   condition=("corruption", cname, sev)),
                                     Policy.NONOFFICIAL_SMOKE))
            return runs

        e1 = assemble_miou_c(grid_runs("E1"), GRID, policy=Policy.NONOFFICIAL_SMOKE, stage="E1")
        # image clean_00 (j=0): synA (0.10,0.20,0.30)->0.20 ; synB 0.40 ; synC (0,0.3,0.6)->0.30 ;
        # synD (0.5,0.6,0.7)->0.60 ; synE (0.1,0.1,0.4)->0.20  => (0.2+0.4+0.3+0.6+0.2)/5 = 0.34
        check("M1 mIoU-C nested hand check for clean_00 == 0.34",
              abs(float(e1.values[0]) - 0.34) < 1e-12,
              f"corruption means = "
              f"{ {k: round(float(v[0]), 4) for k, v in e1.corruption_means.items()} } -> "
              f"{float(e1.values[0]):.6f}")
        flat = np.mean([v for vals in cells.values() for v in vals])
        check("M2 nested mean equals the flat 15-value mean on a COMPLETE grid",
              abs(flat - 0.34) < 1e-12, f"flat={flat:.6f}")
        check("M3 grid requires exactly 15 cells",
              "expected exactly 15" in str(expect(RobustnessError, assemble_miou_c,
                                                  grid_runs("E1")[:14], GRID,
                                                  policy=Policy.NONOFFICIAL_SMOKE)))
        # A grid of exactly 15 runs with one cell missing must be compensated by a duplicate, so
        # the DUPLICATE guard fires first. With 15 unique, in-vocabulary cells the missing set is
        # empty by pigeonhole (5 x 3 = 15), which makes the missing-cell guard unreachable on this
        # path -- it is defence-in-depth behind the count and duplicate guards. Both orderings
        # reject; neither can produce an available-case average.
        _short = grid_runs("E1", skip=("synC", 2), dup=("synB", 2))
        _e4 = expect(RobustnessError, assemble_miou_c, _short, GRID,
                     policy=Policy.NONOFFICIAL_SMOKE)
        check("M4 a 15-run grid with a missing cell is rejected (duplicate guard fires first)",
              len(_short) == 15 and ("duplicate condition cell" in str(_e4)
                                     or "missing condition cell" in str(_e4)),
              str(_e4)[:110])
        check("M5 duplicate cell rejected",
              "duplicate condition cell" in str(expect(RobustnessError, assemble_miou_c,
                                                       grid_runs("E1", skip=("synA", 1),
                                                                 dup=("synB", 2)), GRID,
                                                       policy=Policy.NONOFFICIAL_SMOKE)))
        check("M6 unexpected severity rejected",
              "unexpected severity" in str(expect(RobustnessError, assemble_miou_c,
                                                  grid_runs("E1", sev_override=4), GRID,
                                                  policy=Policy.NONOFFICIAL_SMOKE)))
        check("M7 undefined condition score rejected",
              "undefined disease-only score" in str(expect(RobustnessError, assemble_miou_c,
                                                           grid_runs("E1", undef=True), GRID,
                                                           policy=Policy.NONOFFICIAL_SMOKE)))
        e6 = assemble_miou_c(grid_runs("E6", bump=0.05), GRID,
                             policy=Policy.NONOFFICIAL_SMOKE, stage="E6")
        pmc = align_miou_c(e1, e6, policy=Policy.NONOFFICIAL_SMOKE)
        check("M8 E1/E6 mIoU-C paired by clean image identity",
              pmc.n == len(cids) and np.allclose(pmc.delta, 0.05))

        # ---------------- full family ----------------
        results = []
        rng = np.random.default_rng(7)
        for k, cid in enumerate(CANONICAL_COMPARISON_IDS):
            dd = rng.normal(0.02 + 0.004 * k, 0.05, 40)
            pvv = type(pv)(tuple(f"x{i}" for i in range(40)), np.zeros(40), dd, dd,
                           "per_image_disease_only_miou", Policy.NONOFFICIAL_SMOKE, None, None)
            results.append(run_comparison(cid, pvv, baseline_stage="B", candidate_stage="Cd"))
        fam = holm_family(results)
        check("F5 full 8-member family finalised in canonical order",
              [m.comparison_id for m in fam.members] == list(CANONICAL_COMPARISON_IDS)
              and fam.complete)
        check("F6 missing ID refused",
              "missing=" in str(expect(StatsError, holm_family, results[:-1])))
        check("F7 duplicate ID refused",
              "duplicate=" in str(expect(StatsError, holm_family, results + [results[0]])))
        check("F8 undefined primary p refused",
              "undefined primary p-value" in str(expect(
                  StatsError, holm_family,
                  [run_comparison(CANONICAL_COMPARISON_IDS[0],
                                  type(pv)(tuple("abcd"), np.zeros(4), np.zeros(4), np.array(
                                      [0.0, 0.0, 0.5, 0.0]), "m", Policy.NONOFFICIAL_SMOKE,
                                      None, None),
                                  baseline_stage="B", candidate_stage="C")] + results[1:])))

        # ---------------- no NaN / Infinity anywhere ----------------
        blob = json.dumps([{k: v for k, v in r.__dict__.items() if isinstance(v, (int, float, str))}
                           for r in results], default=str)
        check("N1 no NaN/Infinity token in any serialised A3a result",
              "NaN" not in blob and "Infinity" not in blob)
        check("N2 A3a emits no BCa field (no partial family.json)",
              not any("bca" in k.lower() or "bootstrap" in k.lower()
                      for k in results[0].__dict__))

    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    check("Z1 temporary synthetic artifacts removed", not work.exists())

    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if det:
            print(f"         {det[:165]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'A3a OK -- PAIRED INFERENCE CORE PROVEN' if allok else 'A3a BLOCKED'}")
    print("NONOFFICIAL: dev stack differs from the pinned statistics stack (contract 10.1).")
    print("No bootstrap, no BCa, no family.json/bootstrap.npz emitted -- those belong to A3b.")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
