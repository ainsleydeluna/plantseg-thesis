#!/usr/bin/env python3
"""L-AM17B-GAP smoke (docs/lane_specs/part1.md lane 7 (d) d1-d4; NONOFFICIAL, synthetic only).

  G1  d1 unit: 20 synthetic images x 5 classes, two models with per-image counts -> the union-present
      and GT-present gaps (all-class and disease-only) equal the exact rational values to 1e-12; a
      class with GT = 0 predicted by one model (class 4, E1 only) changes the union-present gap only
      (the GT-present gap is bitwise unchanged); the entry point takes a rule argument over
      noninferiority.PooledStages, whose own union-present reducer agrees at unit weights.
  G2  d2: two identical models -> point 0 exactly, the degenerate path is exercised (the jackknife
      acceleration is non-finite), method 'percentile' is recorded with its reason and the interval
      is [0, 0].
  G3  d3: the same seed gives an identical interval and replicate hash; the interval contains the
      point; B = 10,000 and seed 42 are recorded; the replicates are exactly the SciPy resamples of
      default_rng(42) over the image indices (one shared vector for both models); the acceleration
      equals SciPy's own BCa acceleration bitwise; another seed gives other replicates.
  G4  d4: the per-image summary of a known vector (median, quartiles, IQR, mean, SD, shares, exact
      ties, Hodges-Lehmann shift) equals the exact rational values; bad vectors are refused.
  G5  re-accumulation: the statistic at bootstrap multiplicities and every jackknife value equal a
      pure-Python recomputation; no eligible class and a ground-truth mismatch are fatal.
  G6  the CLI end to end on 846-row VAL CPU re-scores written by the REAL evaluation core and writer
      (a throwaway git fixture is the pin) and their scripts/eligibility_variants.py outputs: written
      once, the lane 7 (c) structure, validity deltas recorded, the d5 cross-check exact, the AM-5
      exclusion applied to the per-image summary, two runs byte-identical.
  G7  refusals (exit 2): upstream protocol, a pre-lane layout (1.1.0 is the floor; 1.3.0 is read), a
      GPU re-score, batch 16, no image digest, two pins, dirty governed paths, another checkpoint,
      other ground truth, an eligibility file of another artifact, an existing output; STOPs (exit 1,
      nothing written): a validity delta above tolerance (E1 and teacher), a d5 mismatch.

Run:  python -B scripts/smoke_gap_bootstrap_val.py
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                       # noqa: E402
import torch                             # noqa: E402

from src.eval.artifacts import (ARTIFACT_FILES, EVAL_RUNTIME_VERSION, MANIFEST_NAME,  # noqa: E402
                                DatasetMeta, RunMeta, prepare_artifact_request,
                                validate_artifact_request, write_artifact)
from src.eval.eval_runtime import RECORD_KEYS  # noqa: E402
from src.eval.evaluate import Condition, EvalBatch, ManifestEntry, evaluate_model  # noqa: E402
from src.stats.eligibility import GT_PRESENT, UNION_PRESENT  # noqa: E402
from src.stats.gap import (B, GapError, RuleGap, bootstrap_gap, jackknife_acceleration,  # noqa: E402
                           paired_totals_identical_gt, per_image_summary, rule_miou)
from src.stats.noninferiority import PooledStages  # noqa: E402

C = 116
CANVAS = "core_preprocess/1.0.0"
UTC = "2026-09-28T00:00:00Z"
SHA_T, SHA_E = "a" * 64, "b" * 64
DIG_T, DIG_E = "sha256:" + "1" * 64, "sha256:" + "2" * 64
SYN_CLASS_MAP = [{"class_id": 0, "name": "", "role": "background_or_non_disease"}] + \
                [{"class_id": c, "name": f"syn_{c:03d}", "role": "disease"} for c in range(1, C)]
CHECKS: list[tuple[str, bool, str]] = []
F = Fraction


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


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def in_process(mod, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = mod.main(argv)
    return rc, buf.getvalue()


# --------------------------------------------------------------------------------------------------
# G1 fixture: 20 images x 5 classes (0 background; 1-3 diseases; 4 never in ground truth)
# --------------------------------------------------------------------------------------------------
def fixture(e1_predicts_4: bool):
    """{model: [{class: (tp, gt, pred)} per image]} for E1 (baseline) and the teacher (candidate)."""
    e1, te = [], []
    for i in range(20):
        d = 1 + i % 3
        e1.append({0: (10 + i % 2, 12, 11 + i % 3), d: (2 + i % 2, 4, 3 + (i % 4 == 0))})
        te.append({0: (11, 12, 11 + (i % 5 == 0)), d: (3, 4, 3 + i % 2)})
    if e1_predicts_4:
        e1[7][4] = (0, 0, 2)
    return e1, te


def dense(model, n_classes=5):
    out = [np.zeros((len(model), n_classes), dtype=np.int64) for _ in range(3)]
    for i, img in enumerate(model):
        for c, trip in img.items():
            for k in range(3):
                out[k][i, c] = trip[k]
    return tuple(out)


def exact_miou(model, counts, rule, classes):
    """Pure-Python exact rational mIoU of a model at image multiplicities `counts`."""
    tot = {}
    for w, img in zip(counts, model):
        for c, (tp, gt, pr) in img.items():
            a = tot.setdefault(c, [0, 0, 0])
            a[0] += w * tp
            a[1] += w * gt
            a[2] += w * pr
    ious = []
    for c in classes:
        tp, gt, pr = tot.get(c, (0, 0, 0))
        un = gt + pr - tp
        if (un > 0) if rule == UNION_PRESENT else (gt > 0):
            ious.append(F(tp, un))
    return sum(ious) / len(ious)


def exact_gap(e1, te, counts, rule, classes):
    return exact_miou(te, counts, rule, classes) - exact_miou(e1, counts, rule, classes)


# --------------------------------------------------------------------------------------------------
# 846-row VAL CPU re-scores from the REAL evaluation core and writer
# --------------------------------------------------------------------------------------------------
def clean_git_fixture(root: Path, tag: str) -> Path:
    """A throwaway repository with one commit and nothing dirty; never the project repository."""
    root.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-c", "user.name=gap-fixture", "-c", "user.email=gap@invalid",
                        "-c", "commit.gpgsign=false", *args], cwd=str(root), check=True, capture_output=True)
    git("init", "-q")
    (root / "seed.txt").write_text(f"gap fixture {tag}\n", encoding="utf-8")
    git("add", "--", "seed.txt")
    git("commit", "-q", "-m", "fixture baseline")
    return root


DISEASES = [c for c in range(1, C) if c != 69]


def label_maps(n=846, gt_shift=None):
    gts = torch.zeros(n, 4, 4, dtype=torch.long)
    for i in range(n):
        if i in (5, 400):
            continue                                            # AM-5: no disease ground truth
        k = i + 1 if i == gt_shift else i
        gts[i, 1:3, 1:3] = DISEASES[k % len(DISEASES)]
    t, e = gts.clone(), gts.clone()
    for i in range(n):
        d = DISEASES[i % len(DISEASES)]
        if i % 5 == 0:
            t[i, 1, 1] = 0
        if i % 7 == 0:
            t[i, 0, 0] = d
        if i % 3 == 0:
            e[i, 2, 2] = DISEASES[(i + 1) % len(DISEASES)]
        if i % 2 == 0:
            e[i, 1, 2] = 0
    e[10, 3, 3] = 69                                            # predicted by E1, never in GT
    return gts, t, e


def core_result(ids, gts, preds):
    manifest = [ManifestEntry(i, iid, iid) for i, iid in enumerate(ids)]
    logits = torch.nn.functional.one_hot(preds, C).permute(0, 3, 1, 2).float()
    batches = [EvalBatch(images=logits[i:i + 64], targets=gts[i:i + 64], image_ids=ids[i:i + 64],
                         clean_image_ids=ids[i:i + 64],
                         manifest_indices=list(range(i, min(i + 64, len(ids)))))
               for i in range(0, len(ids), 64)]
    res = evaluate_model(None, batches, expected_manifest=manifest, condition=Condition(),
                         num_classes=C, background_index=0, ignore_index=255, forward=lambda _m, x: x)
    return res, manifest


def runtime(device="cpu", batch_size=1, digest=None):
    values = {"eval_runtime_version": EVAL_RUNTIME_VERSION, "model_device": device,
              "model_tensor_devices": [device], "input_devices": [device], "forward_batches": 846,
              "batch_size": batch_size, "num_workers": 0, "determinism_policy_applied": False,
              "cuda_initialized_before_policy": None, "inherited_cublas_workspace_config": None,
              "determinism": {}, "fill_uninitialized_memory": None, "cudnn_enabled": True, "tf32": {},
              "gpu_name": None, "gpu_capability": None, "cudnn_version": None, "torch_cuda": None,
              "torch_num_threads": 4, "pillow": None, "checkpoint_iteration": None,
              "checkpoint_best_val_miou_all_class": None, "image_digest": digest, "eval_warnings": [],
              "eval_warnings_truncated": False, "nondeterministic_alert_count": 0}
    return {k: values[k] for k in RECORD_KEYS}


def write(out_dir, res, manifest, *, stage, role, sha, digest, repo_root, device="cpu", batch_size=1):
    req = prepare_artifact_request(
        out_dir=out_dir, artifact_status="provisional", run_id=f"syn_gap_{out_dir.name}",
        run=RunMeta(stage=stage, model_role=role, precision="fp32", checkpoint_path=f"synthetic_{stage}.pt",
                    checkpoint_sha256=sha, device=device),
        dataset=DatasetMeta(name="SYNTHETIC gap fixture (NOT PlantSeg data)", doi="10.5281/zenodo.17719108",
                            split="val", condition=Condition(), preprocess_protocol=CANVAS,
                            expected_rows=len(manifest)),
        expected_manifest=manifest, class_map=SYN_CLASS_MAP, repo_root=repo_root)
    return write_artifact(res, req, validate_artifact_request(req), timestamp_utc=UTC,
                          eval_runtime=runtime(device, batch_size, digest))


def rehash(d: Path) -> None:
    (d / MANIFEST_NAME).write_text(
        "".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in ARTIFACT_FILES),
        encoding="utf-8", newline="\n")


def edited(src: Path, dst: Path, fn=None, rows_fn=None) -> Path:
    shutil.copytree(src, dst)
    if fn is not None:
        s = fn(json.loads((dst / "summary.json").read_text(encoding="utf-8")))
        (dst / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                          encoding="utf-8", newline="\n")
    if rows_fn is not None:
        rows = [json.loads(ln) for ln in (dst / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
        (dst / "per_image.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                     for r in rows_fn(rows)), encoding="utf-8", newline="\n")
    rehash(dst)
    return dst


def eligibility(art: Path, out: Path) -> Path:
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "eligibility_variants.py"), str(art),
                        "--out", str(out)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"eligibility_variants.py failed: {(p.stdout + p.stderr)[-300:]}")
    run_id = json.loads((art / "summary.json").read_text(encoding="utf-8"))["run"]["run_id"]
    return out / f"{run_id}_eligibility_variants.json"


# --------------------------------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    print("=" * 100)
    print("L-AM17B-GAP -- VAL gap bootstrap smoke  [NONOFFICIAL: synthetic data only]")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="gap_smoke_"))
    try:
        # ================================ G1 d1 by hand ================================
        ones = [1] * 20
        scopes = {"all": range(5), "dis": range(1, 5)}
        res = {}
        for variant in (True, False):
            e1, te = fixture(variant)
            st = PooledStages.from_dense(dense(e1), dense(te))
            for rule in (UNION_PRESENT, GT_PRESENT):
                for sk, cls in scopes.items():
                    res[(variant, rule, sk)] = (RuleGap(st, rule, cls).point()["point"],
                                                exact_gap(e1, te, ones, rule, cls))
        check("G1-1 all eight point estimates equal the exact rational gaps (1e-12)",
              all(abs(got - float(want)) < 1e-12 for got, want in res.values()),
              {k: v[0] for k, v in res.items()})
        check("G1-2 E1 predicting the GT-absent class 4 changes the union-present gap (all-class, "
              "disease-only)",
              res[(True, UNION_PRESENT, "all")][0] != res[(False, UNION_PRESENT, "all")][0]
              and res[(True, UNION_PRESENT, "dis")][0] != res[(False, UNION_PRESENT, "dis")][0])
        check("G1-3 ... and leaves the GT-present gap bitwise unchanged",
              res[(True, GT_PRESENT, "all")][0] == res[(False, GT_PRESENT, "all")][0]
              and res[(True, GT_PRESENT, "dis")][0] == res[(False, GT_PRESENT, "dis")][0])
        e1, te = fixture(True)
        st = PooledStages.from_dense(dense(e1), dense(te))
        pt = RuleGap(st, UNION_PRESENT, range(5)).point()
        check("G1-4 eligible counts: union-present E1 5 (class 4 at IoU 0), teacher 4; GT-present 4 and 4",
              (pt["n_eligible_e1"], pt["n_eligible_teacher"]) == (5, 4)
              and RuleGap(st, GT_PRESENT, range(5)).point()["n_eligible_e1"] == 4)
        check("G1-5 the rule entry point agrees with PooledStages' own union-present reducer at unit weights",
              abs(RuleGap(st, UNION_PRESENT, range(5)).point()["point"] - st.observed()) < 1e-15)
        check("G1-6 an unknown rule or an empty/out-of-range class subset is refused",
              "rule must be" in str(expect(GapError, RuleGap, st, "mean", range(5)))
              and "non-empty subset" in str(expect(GapError, RuleGap, st, GT_PRESENT, []))
              and "non-empty subset" in str(expect(GapError, RuleGap, st, GT_PRESENT, [5])))

        # ================================ G2 d2 identical models ================================
        same = PooledStages.from_dense(dense(te), dense(te))
        r2 = bootstrap_gap(RuleGap(same, UNION_PRESENT, range(5)))
        check("G2-1 identical models: point 0 exactly, interval [0, 0]",
              r2["point"] == 0.0 and r2["ci_low"] == 0.0 and r2["ci_high"] == 0.0, r2)
        check("G2-2 the degenerate path: non-finite acceleration -> method 'percentile' with its reason",
              r2["method"] == "percentile" and r2["acceleration"] is None
              and "non-finite" in (r2["fallback_reason"] or ""), r2["fallback_reason"])
        check("G2-3 B = 10,000 and seed 42 recorded on the fallback path",
              r2["B"] == B == 10_000 and r2["seed"] == 42)

        # ================================ G3 d3 determinism ================================
        from scipy.stats import bootstrap as sp_bootstrap
        from scipy.stats._resampling import _bca_interval, _vectorize_statistic
        gap = RuleGap(st, UNION_PRESENT, range(5))
        a, b = bootstrap_gap(gap), bootstrap_gap(gap)
        check("G3-1 the same seed gives an identical interval and replicate hash (two calls)",
              a == b and a["method"] == "BCa", {k: a[k] for k in ("ci_low", "ci_high", "method")})
        check("G3-2 the interval contains the point estimate; B = 10,000, seed 42, confidence 0.95",
              a["contains_point"] and a["ci_low"] <= a["point"] <= a["ci_high"] and a["B"] == 10_000
              and a["seed"] == 42 and a["confidence_level"] == 0.95)
        rng = np.random.default_rng(42)
        draws = rng.integers(0, 20, size=(10_000, 20))
        manual = np.array([gap(row) for row in draws])
        check("G3-3 the replicates are the default_rng(42) resamples of the image indices (bitwise)",
              hashlib.sha256(manual.astype("<f8").tobytes()).hexdigest() == a["replicates_sha256"])
        ref = sp_bootstrap((np.arange(20),), gap, n_resamples=10_000, method="BCa",
                           random_state=np.random.default_rng(42), vectorized=False)
        _, _, a_hat = _bca_interval([np.arange(20)], _vectorize_statistic(gap), axis=-1, alpha=0.025,
                                    theta_hat_b=ref.bootstrap_distribution, batch=None)
        check("G3-4 the interval is SciPy's BCa interval and the acceleration equals SciPy's bitwise",
              (a["ci_low"], a["ci_high"]) == (float(ref.confidence_interval.low),
                                               float(ref.confidence_interval.high))
              and a["acceleration"] == float(a_hat), (a["acceleration"], float(a_hat)))
        other = bootstrap_gap(gap, seed=43)
        check("G3-5 another seed draws other replicates",
              other["replicates_sha256"] != a["replicates_sha256"])
        check("G3-6 the jackknife acceleration of identical values is non-finite (NaN), not an error",
              np.isnan(jackknife_acceleration(np.zeros(20))))

        # ================================ G4 d4 per-image summary ================================
        dv = [F(-2, 10), F(-1, 10), F(0), F(0), F(1, 10), F(1, 10), F(3, 10), F(5, 10)]
        s = per_image_summary([float(x) for x in dv])
        n = len(dv)
        mean = sum(dv) / n
        sd = float((sum((x - mean) ** 2 for x in dv) / (n - 1))) ** 0.5
        walsh = sorted((dv[i] + dv[j]) / 2 for i in range(n) for j in range(i, n))
        hl = (walsh[len(walsh) // 2 - 1] + walsh[len(walsh) // 2]) / 2
        exact = {"n": 8, "median": F(1, 20), "q25": F(-1, 40), "q75": F(3, 20), "iqr": F(7, 40),
                 "mean": F(7, 80), "share_teacher_better": F(1, 2), "share_e1_better": F(1, 4),
                 "share_ties": F(1, 4), "hl_shift": hl}
        got = s.as_dict()
        check("G4-1 median, quartiles (linear), IQR, mean, shares and exact ties equal the rational values",
              all(abs(got[k] - float(v)) < 1e-15 for k, v in exact.items()), got)
        check("G4-2 SD uses ddof = 1; the Hodges-Lehmann shift is the Walsh-average median (1/20 here)",
              abs(got["sd"] - sd) < 1e-15 and abs(got["hl_shift"] - float(hl)) < 1e-15 and hl == F(1, 20),
              (got["sd"], got["hl_shift"]))
        check("G4-3 a vector with NaN, or shorter than two, is refused",
              "finite 1-D" in str(expect(GapError, per_image_summary, [0.1, float("nan")]))
              and "finite 1-D" in str(expect(GapError, per_image_summary, [0.1])))

        # ================================ G5 re-accumulation ================================
        idx = np.random.default_rng(7).integers(0, 20, size=20)
        counts = np.bincount(idx, minlength=20)
        ok = all(abs(RuleGap(st, rule, cls)(idx)
                     - float(exact_gap(e1, te, counts.tolist(), rule, cls))) < 1e-12
                 for rule in (UNION_PRESENT, GT_PRESENT) for cls in scopes.values())
        check("G5-1 the statistic at bootstrap multiplicities equals the exact recomputation", ok)
        jk = RuleGap(st, GT_PRESENT, range(1, 5)).jackknife()
        want = [float(exact_gap(e1, te, [0 if j == i else 1 for j in range(20)], GT_PRESENT, range(1, 5)))
                for i in range(20)]
        check("G5-2 every jackknife value equals the leave-one-image-out recomputation",
              all(abs(x - y) < 1e-12 for x, y in zip(jk, want)))
        check("G5-3 no eligible class is fatal (never NaN)",
              "undefined" in str(expect(GapError, rule_miou, np.zeros(3), np.zeros(3), np.zeros(3),
                                        np.ones(3, bool), GT_PRESENT)))
        other_gt = PooledStages.from_dense(dense(e1), (dense(te)[0], dense(te)[1] + 1, dense(te)[2] + 1))
        check("G5-4 a ground-truth mismatch between the two models is refused",
              "ground-truth" in str(expect(GapError, paired_totals_identical_gt, other_gt)))

        # ================================ G6 the CLI end to end ================================
        cli = load("gap_bootstrap_val", "scripts/gap_bootstrap_val.py")
        fx = clean_git_fixture(work / "pin_repo", "a")
        ids = [f"v{i:04d}" for i in range(846)]
        gts, pt_, pe_ = label_maps()
        rt_, man = core_result(ids, gts, pt_)
        re_, _ = core_result(ids, gts, pe_)
        art_t = write(work / "teacher_cpu", rt_, man, stage="teacher", role="teacher", sha=SHA_T,
                      digest=DIG_T, repo_root=fx)
        art_e = write(work / "e1_cpu", re_, man, stage="E1", role="student", sha=SHA_E, digest=DIG_E,
                      repo_root=fx)
        el_t, el_e = eligibility(art_t, work / "elig"), eligibility(art_e, work / "elig")
        st_t = json.loads((art_t / "summary.json").read_text(encoding="utf-8"))
        st_e = json.loads((art_e / "summary.json").read_text(encoding="utf-8"))
        v_t, v_e = st_t["dataset_level"]["all_class_miou"], st_e["dataset_level"]["all_class_miou"]
        cli.PAIRING["teacher"].update(checkpoint_sha256=SHA_T, image_digest=DIG_T, reference=v_t - 5e-6)
        cli.PAIRING["e1"].update(checkpoint_sha256=SHA_E, image_digest=DIG_E, reference=v_e + 5e-8)
        base = ["--teacher", str(art_t), "--e1", str(art_e), "--eligibility-variants", str(el_t), str(el_e),
                "--generated-utc", UTC]
        out = work / "derived"
        rc, text = in_process(cli, [*base, "--out-dir", str(out)])
        jp = out / "gap_val_20260928T000000Z.json"
        check("G6-1 CLI exits 0 and writes gap_val_<UTC>.json", rc == 0 and jp.is_file(), text[-500:])
        doc = json.loads(jp.read_text(encoding="utf-8")) if jp.is_file() else {}
        need = {"point", "ci_low", "ci_high", "method", "B", "seed"}
        check("G6-2 lane 7 (c) structure: inputs, rules{union_present,gt_present}, disease_only, per_image, "
              "validity_deltas",
              {"inputs", "rules", "disease_only", "per_image", "validity_deltas"} <= set(doc)
              and set(doc["rules"]) == set(doc["disease_only"]) == {"union_present", "gt_present"}
              and all(need <= set(doc[b][r]) for b in ("rules", "disease_only") for r in doc[b])
              and {"n", "median", "iqr", "mean", "sd", "share_teacher_better", "share_ties", "hl_shift"}
              <= set(doc["per_image"]))
        up, gp = doc["rules"]["union_present"], doc["rules"]["gt_present"]
        check("G6-3 B = 10,000, seed 42, BCa, each interval contains its point",
              all(r["B"] == 10_000 and r["seed"] == 42 and r["method"] == "BCa" and r["contains_point"]
                  for b in ("rules", "disease_only") for r in doc[b].values()), (up["method"], gp["method"]))
        el_td = json.loads(el_t.read_text(encoding="utf-8"))["rules_float64"]
        el_ed = json.loads(el_e.read_text(encoding="utf-8"))["rules_float64"]
        d5 = doc["d5_cross_check"]
        check("G6-4 d5: every point equals the L-AM17-GTPRESENT rules_float64 difference (here bitwise)",
              all(d5[s][r]["holds"] and d5[s][r]["abs_diff"] == 0.0 and d5[s][r]["teacher_equal"]
                  and d5[s][r]["e1_equal"] for s in ("all_class", "disease_only")
                  for r in ("union_present", "gt_present"))
              and up["point"] == el_td["union_present"]["all_class"] - el_ed["union_present"]["all_class"])
        dis_up = doc["disease_only"]["union_present"]
        check("G6-5 union-present counts class 69 for E1 only (all-class 115 vs 116, disease-only 114 vs "
              "115); GT-present 115/115",
              (up["n_eligible_teacher"], up["n_eligible_e1"]) == (115, 116)
              and (dis_up["n_eligible_teacher"], dis_up["n_eligible_e1"]) == (114, 115)
              and (gp["n_eligible_teacher"], gp["n_eligible_e1"]) == (115, 115),
              (up["n_eligible_teacher"], up["n_eligible_e1"], gp["n_eligible_teacher"]))
        vd = doc["validity_deltas"]
        check("G6-6 validity deltas recorded (E1 -5e-8 within 1e-7; teacher +5e-6 within 1e-5)",
              vd["e1"]["within"] and vd["teacher"]["within"] and abs(vd["e1"]["delta"] + 5e-8) < 1e-15
              and abs(vd["teacher"]["delta"] - 5e-6) < 1e-15, vd)
        rows_t = [json.loads(x) for x in (art_t / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
        rows_e = [json.loads(x) for x in (art_e / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
        dvec = np.array([a_["disease_only_miou"] - b_["disease_only_miou"] for a_, b_ in zip(rows_t, rows_e)
                         if not a_["am5_excluded"]])
        pi = doc["per_image"]
        check("G6-7 per-image summary on the 844 AM-5-included images equals a direct recomputation",
              pi["n"] == 844 and pi["n_excluded_am5"] == 2 and abs(pi["mean"] - float(np.mean(dvec))) < 1e-15
              and pi["median"] == float(np.median(dvec))
              and pi["share_teacher_better"] == float(np.mean(dvec > 0))
              and pi["share_ties"] == float(np.mean(dvec == 0)),
              {k: pi[k] for k in ("n", "mean", "median", "share_ties")})
        check("G6-8 inputs: both artifacts' four file hashes, one pin, CPU at batch 1, eligibility file "
              "hashes",
              doc["inputs"]["teacher"]["artifact_sha256s"]["summary.json"]
              == hashlib.sha256((art_t / "summary.json").read_bytes()).hexdigest()
              and doc["inputs"]["pairing"]["device"] == "cpu" and doc["inputs"]["pairing"]["batch_size"] == 1
              and doc["inputs"]["eligibility_variants"]["e1"]["sha256"]
              == hashlib.sha256(el_e.read_bytes()).hexdigest())
        rc2, _ = in_process(cli, [*base, "--out-dir", str(work / "derived2")])
        check("G6-9 two runs write byte-identical results", rc2 == 0
              and (work / "derived2" / jp.name).read_bytes() == jp.read_bytes())

        # ================================ G7 refusals and STOPs ================================
        def run(label, t=art_t, e=art_e, et=el_t, ee=el_e, *, marker, code=2, extra=()):
            o = work / f"out_{len(CHECKS)}"
            rc, text = in_process(cli, ["--teacher", str(t), "--e1", str(e), "--eligibility-variants",
                                        str(et), str(ee), "--out-dir", str(o), "--generated-utc", UTC,
                                        *extra])
            check(label, rc == code and marker in text and not o.exists(), text[-260:])

        up_art = edited(art_t, work / "t_up", fn=lambda s: {
            **s, "protocol": {"name": "upstream"},
            "dataset": {**s["dataset"], "preprocess_protocol": "upstream/1.0.0"}})
        run("G7-1 an upstream-protocol artifact is refused (canvas only)", t=up_art, marker="only canvas")
        pre = edited(art_e, work / "e_pre",
                     fn=lambda s: {k: v for k, v in s.items() if k not in ("am5", "artifact_schema_version")},
                     rows_fn=lambda rows: [{k: v for k, v in r.items() if k != "am5_excluded"} for r in rows])
        run("G7-2 a pre-lane layout is refused (1.1.0 is the floor)", e=pre, marker="below the required")
        from src.stats.val_artifacts import AM5_LAYOUT, load_val_artifact
        v13 = edited(art_e, work / "e_v13", fn=lambda s: {**s, "artifact_schema_version":
                                                          "plantseg-eval-artifact/1.3.0"})
        check("G7-3 a 1.3.0 layout is read (any 1.x at or after 1.1.0)",
              load_val_artifact(v13, label="e1", min_layout=AM5_LAYOUT).layout == (1, 3, 0))
        gpu = write(work / "e1_gpu", re_, man, stage="E1", role="student", sha=SHA_E, digest=DIG_E,
                    repo_root=fx, device="cuda:0")
        run("G7-4 a GPU re-score is refused", e=gpu, marker="not a CPU re-score")
        b16 = write(work / "e1_b16", re_, man, stage="E1", role="student", sha=SHA_E, digest=DIG_E,
                    repo_root=fx, batch_size=16)
        run("G7-5 a batch-16 re-score is refused", e=b16, marker="batch size 16")
        nod = write(work / "e1_nodigest", re_, man, stage="E1", role="student", sha=SHA_E, digest=None,
                    repo_root=fx)
        run("G7-6 a re-score without the pinned image digest is refused", e=nod,
            marker="is not the pinned image")
        fx2 = clean_git_fixture(work / "pin_repo_b", "b")
        t2 = write(work / "teacher_pin_b", rt_, man, stage="teacher", role="teacher", sha=SHA_T, digest=DIG_T,
                   repo_root=fx2)
        run("G7-7 two different pins are refused", t=t2, marker="different commits")
        dirty = edited(art_t, work / "t_dirty", fn=lambda s: {**s, "run": {**s["run"],
                                                                         "governed_paths_clean": False}})
        run("G7-8 dirty governed paths at scoring time are refused", t=dirty,
            marker="governed paths were dirty")
        oth = edited(art_t, work / "t_other",
                     fn=lambda s: {**s, "run": {**s["run"], "checkpoint_sha256": "c" * 64}})
        run("G7-9 another teacher checkpoint is refused", t=oth, marker="is not the expected model")
        g2, _, pe2 = label_maps(gt_shift=100)
        rg, _ = core_result(ids, g2, pe2)
        othergt = write(work / "e1_othergt", rg, man, stage="E1", role="student", sha=SHA_E, digest=DIG_E,
                        repo_root=fx)
        run("G7-10 an E1 re-score on other ground truth is refused", e=othergt,
            marker="ground-truth counts differ")
        run("G7-11 an eligibility file of another artifact is refused", et=el_e,
            marker="other artifact files")
        before = jp.read_bytes()
        rc, text = in_process(cli, [*base, "--out-dir", str(out)])
        check("G7-12 an existing output is never overwritten (exit 2; unchanged)",
              rc == 2 and "refusing to overwrite" in text and jp.read_bytes() == before, text[-160:])
        ref_e = cli.PAIRING["e1"]["reference"]
        cli.PAIRING["e1"]["reference"] = v_e + 2e-7
        run("G7-13 an E1 validity delta above 1e-7 is a STOP (exit 1, nothing written)",
            marker="validity delta exceeds", code=1)
        cli.PAIRING["e1"]["reference"] = ref_e
        ref_t = cli.PAIRING["teacher"]["reference"]
        cli.PAIRING["teacher"]["reference"] = v_t + 2e-5
        run("G7-14 a teacher validity delta above 1e-5 is a STOP (exit 1)", marker="validity delta exceeds",
            code=1)
        cli.PAIRING["teacher"]["reference"] = ref_t
        bad = json.loads(el_t.read_text(encoding="utf-8"))
        bad["rules_float64"]["gt_present"]["all_class"] += 1e-9
        badp = work / "elig_bad.json"
        badp.write_text(json.dumps(bad), encoding="utf-8")
        run("G7-15 a d5 mismatch above 1e-12 is a STOP (exit 1, nothing written)", et=badp,
            marker="do not reproduce the L-AM17-GTPRESENT", code=1)
        p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "gap_bootstrap_val.py"), *base,
                            "--out-dir", str(work / "sub")], capture_output=True, text=True)
        check("G7-16 the subprocess CLI (registered pairing) refuses the synthetic pair -> exit 2",
              p.returncode == 2 and "is not the expected model" in p.stdout and not (work / "sub").exists(),
              (p.stdout + p.stderr)[-200:])
        h = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "gap_bootstrap_val.py"), "--help"],
                           capture_output=True, text=True)
        check("G7-17 --help renders (exit 0)", h.returncode == 0 and "usage:" in h.stdout, h.stderr[-200:])
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    check("Z1 temporary synthetic artifacts removed", not work.exists())

    print()
    for n_, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n_}")
        if det and not ok:
            print(f"         {det[:220]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'L-AM17B-GAP OK' if allok else 'L-AM17B-GAP BLOCKED'} ({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic fixtures only; the two CPU re-scores and d5 run locally after the merge "
          "(scripts/evaluate_model.py, scripts/eligibility_variants.py, scripts/gap_bootstrap_val.py).")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
