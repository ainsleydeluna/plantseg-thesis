#!/usr/bin/env python3
"""L-AM17-GTPRESENT smoke (docs/lane_specs/part1.md lane 3 (d) d1; NONOFFICIAL, synthetic only).

  E1  d1 unit: 3 images x 4 classes; class 3 has GT = 0 everywhere, model A predicts it, model B does
      not -> A: union-present n = 4, GT-present n = 3; B: 3 and 3; all eight means (all-class and
      disease-only, both rules, both models) equal the hand-computed fractions to 1e-12 (float64).
  E2  the float32 `value` is the production arithmetic: bitwise equal to src/eval/metrics.py
      miou_from_confusion on the same confusion matrix (all-class and disease-only); the per-image
      sparse triplets sum to the confusion-matrix totals.
  E3  malformed input and invalid requests fail loudly; an empty eligible set is None, never NaN.
  E4  scripts/eligibility_variants.py end to end on 116-class artifacts written by the REAL evaluation
      core and writer: R1 bitwise, R2 per-class and R3 relation hold, the output is written once and
      never overwritten, --expect-union-all-class is exact, an artifact whose summary was not produced
      by the production reducer is refused as unfaithful (nothing written), a tampered artifact errors.

Run:  python -B scripts/smoke_eligibility_variants.py
"""
from __future__ import annotations

import contextlib
import hashlib
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

from src.eval.artifacts import (DatasetMeta, RunMeta, prepare_artifact_request,  # noqa: E402
                                validate_artifact_request, write_artifact)
from src.eval.evaluate import (Condition, EvalBatch, EvalResult, ManifestEntry,  # noqa: E402
                               PerImageRow, evaluate_model)
from src.eval.metrics import confusion_matrix, miou_from_confusion  # noqa: E402
from src.stats.eligibility import (GT_PRESENT, UNION_PRESENT, EligibilityError,  # noqa: E402
                                   class_totals, dataset_miou, per_class_table, rule_variants)

C116 = 116
SYN_CLASS_MAP = [{"class_id": 0, "name": "", "role": "background_or_non_disease"}] + \
                [{"class_id": c, "name": f"syn_{c:03d}", "role": "disease"} for c in range(1, C116)]
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
# d1 fixture: 3 images x 4 classes (0 background, 1-3 diseases); class 3 never in ground truth
# --------------------------------------------------------------------------------------------------
GT = [[[0, 0, 1], [0, 1, 1]],
      [[0, 2, 2], [0, 0, 2]],
      [[0, 0, 0], [1, 2, 2]]]
PRED_A = [[[0, 1, 1], [0, 3, 1]],          # model A predicts class 3 once per image
          [[0, 2, 3], [0, 0, 2]],
          [[0, 0, 3], [1, 2, 0]]]
PRED_B = [[[0, 1, 1], [0, 0, 1]],          # model B never predicts class 3
          [[0, 2, 2], [1, 0, 2]],
          [[0, 0, 0], [1, 1, 2]]]
# Hand totals (TP / GT / PRED per class, summed over the three images):
#   A: c0 7/9/8 -> union 10, IoU 7/10;  c1 3/4/4 -> 5, 3/5;  c2 3/5/3 -> 5, 3/5;  c3 0/0/3 -> 3, 0
#   B: c0 7/9/8 -> union 10, IoU 7/10;  c1 3/4/6 -> 7, 3/7;  c2 4/5/4 -> 5, 4/5;  c3 absent
F = Fraction
HAND = {
    "A": {"totals": {0: (7, 9, 8), 1: (3, 4, 4), 2: (3, 5, 3), 3: (0, 0, 3)},
          UNION_PRESENT: {"all": (F(19, 40), 4), "dis": (F(2, 5), 3)},     # (7/10+3/5+3/5+0)/4
          GT_PRESENT: {"all": (F(19, 30), 3), "dis": (F(3, 5), 2)}},       # (7/10+3/5+3/5)/3
    "B": {"totals": {0: (7, 9, 8), 1: (3, 4, 6), 2: (4, 5, 4), 3: (0, 0, 0)},
          UNION_PRESENT: {"all": (F(9, 14), 3), "dis": (F(43, 70), 2)},    # (7/10+3/7+4/5)/3
          GT_PRESENT: {"all": (F(9, 14), 3), "dis": (F(43, 70), 2)}},
}


def sparse_stats(gt_maps, pred_maps, num_classes):
    """Per-image sparse triplets exactly as the evaluator stores them, plus the confusion total."""
    rows = {k: [] for k in ("image_index", "class_id", "tp", "gt", "pred")}
    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i, (g, p) in enumerate(zip(gt_maps, pred_maps)):
        cm = confusion_matrix(torch.tensor(p), torch.tensor(g), num_classes)
        cm_total += cm
        tp, gt, pr = torch.diag(cm), cm.sum(1), cm.sum(0)
        for c in range(num_classes):
            if tp[c] or gt[c] or pr[c]:
                for k, v in zip(rows, (i, c, int(tp[c]), int(gt[c]), int(pr[c]))):
                    rows[k].append(v)
    stats = {k: np.array(v, dtype=np.int64) for k, v in rows.items()}
    stats["dataset_tp"] = torch.diag(cm_total).numpy().astype(np.int64)
    stats["dataset_gt"] = cm_total.sum(1).numpy().astype(np.int64)
    stats["dataset_pred"] = cm_total.sum(0).numpy().astype(np.int64)
    return stats, cm_total


# --------------------------------------------------------------------------------------------------
# 116-class artifacts from the REAL evaluation core
# --------------------------------------------------------------------------------------------------
def core_artifact(out_dir: Path, gts: torch.Tensor, preds: torch.Tensor, tag: str) -> Path:
    n = gts.shape[0]
    ids = [f"{tag}_{i:02d}" for i in range(n)]
    manifest = [ManifestEntry(i, iid, iid) for i, iid in enumerate(ids)]
    logits = torch.nn.functional.one_hot(preds, C116).permute(0, 3, 1, 2).float()
    batches = [EvalBatch(images=logits[i:i + 2], targets=gts[i:i + 2], image_ids=ids[i:i + 2],
                         clean_image_ids=ids[i:i + 2], manifest_indices=list(range(i, min(i + 2, n))))
               for i in range(0, n, 2)]
    res = evaluate_model(None, batches, expected_manifest=manifest, condition=Condition(),
                         num_classes=C116, background_index=0, ignore_index=255,
                         forward=lambda _m, x: x)
    return _write(out_dir, res, manifest, tag)


def _write(out_dir, res, manifest, tag):
    req = prepare_artifact_request(
        out_dir=out_dir, artifact_status="smoke", run_id=f"syn_gtpresent_{tag}",
        run=RunMeta(stage="E1", model_role="student", precision="fp32", random_init=True),
        dataset=DatasetMeta(name="SYNTHETIC eligibility fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split="val", condition=Condition(),
                            preprocess_protocol="synthetic-eligibility/1.0.0",
                            expected_rows=len(manifest)),
        expected_manifest=manifest, class_map=SYN_CLASS_MAP, repo_root=REPO)
    return write_artifact(res, req, validate_artifact_request(req),
                          timestamp_utc="2026-09-28T00:00:00Z")


def float64_summary_artifact(out_dir: Path) -> Path:
    """A fixture whose summary.json holds a numpy float64 mean, NOT the production float32 reducer
    (as smoke_stats_paired's factory does): a faithful reconstruction must refuse it (R1)."""
    ids = [f"f64_{i}" for i in range(3)]
    manifest = [ManifestEntry(i, iid, iid) for i, iid in enumerate(ids)]
    rows, trip = [], []
    for i, iid in enumerate(ids):
        rows.append(PerImageRow(image_id=iid, clean_image_id=iid, manifest_index=i,
                                condition=Condition(), all_class_miou=0.5,
                                all_class_miou_status="ok", disease_only_miou=0.4,
                                disease_only_miou_status="ok", n_eligible_all_class=2,
                                n_eligible_disease_only=1, gt_disease_classes=[1]))
        trip += [(i, 0, 8, 10, 9), (i, 1, 2, 3, 4)]
    a = np.array(trip, dtype=np.int64)
    tot = {k: np.zeros(C116, np.int64) for k in ("tp", "gt", "pr")}
    for j, k in enumerate(("tp", "gt", "pr")):
        np.add.at(tot[k], a[:, 1], a[:, 2 + j])
    un = tot["gt"] + tot["pr"] - tot["tp"]
    el = un > 0
    iou = np.where(el, tot["tp"] / np.maximum(un, 1), np.nan)
    res = EvalResult(
        rows=rows, manifest_ids=ids, sparse_image_index=a[:, 0].copy(),
        sparse_class_id=a[:, 1].copy(), sparse_tp=a[:, 2].copy(), sparse_gt=a[:, 3].copy(),
        sparse_pred=a[:, 4].copy(), dataset_tp=tot["tp"], dataset_gt=tot["gt"],
        dataset_pred=tot["pr"],
        dataset_level={"all_class_miou": float(np.mean(iou[el])), "all_class_miou_n_eligible": 2,
                       "all_class_macro_dice": 0.5, "all_class_dice_n_eligible": 2,
                       "all_class_macc": 0.5, "all_class_macc_n_eligible": 2,
                       "disease_only_miou": float(iou[1]), "disease_only_miou_n_eligible": 1,
                       "aacc_diagnostic": 0.5},
        per_class={"class_ids": list(range(C116)), "gt_support": tot["gt"].tolist(),
                   "pred_support": tot["pr"].tolist(), "intersection": tot["tp"].tolist(),
                   "union": un.tolist(),
                   "iou": [float(iou[c]) if el[c] else None for c in range(C116)],
                   "dice": [0.5 if el[c] else None for c in range(C116)],
                   "acc": [0.5 if tot["gt"][c] else None for c in range(C116)],
                   "iou_status": ["ok" if el[c] else "undefined_absent_from_gt_and_pred"
                                  for c in range(C116)]},
        integrity={"row_count_ok": True, "ids_unique": True, "ids_match_manifest": True,
                   "order_canonical": True, "duplicate_ids": [], "missing_ids": [],
                   "invalid_pred_labels": 0, "undefined_per_image_scores": 0,
                   "overwrite_policy": "refuse_existing"},
        num_classes=C116, background_index=0, ignore_index=255, condition=Condition())
    return _write(out_dir, res, manifest, "f64")


def run_cli(argv):
    import importlib.util
    spec = importlib.util.spec_from_file_location("eligibility_variants",
                                                  REPO / "scripts" / "eligibility_variants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(argv)
    return rc, buf.getvalue()


# --------------------------------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    print("=" * 100)
    print("L-AM17-GTPRESENT -- eligibility variants smoke  [NONOFFICIAL: synthetic data only]")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="gtpresent_"))
    try:
        # ================================ E1 d1 by hand ================================
        for model, preds in (("A", PRED_A), ("B", PRED_B)):
            stats, cm = sparse_stats(GT, preds, 4)
            tot = class_totals(stats)
            hand = HAND[model]
            check(f"E1 {model}: class totals (TP/GT/PRED) equal the hand counts",
                  all((int(tot.tp[c]), int(tot.gt[c]), int(tot.pred[c])) == hand["totals"][c]
                      for c in range(4)), [(int(tot.tp[c]), int(tot.gt[c]), int(tot.pred[c]))
                                           for c in range(4)])
            v = rule_variants(stats, background_index=0)
            for rule in (UNION_PRESENT, GT_PRESENT):
                for scope, key in (("all_class", "all"), ("disease_only", "dis")):
                    want, n = hand[rule][key]
                    got = v[rule][scope]
                    check(f"E1 {model}: {rule} {scope} n={n}, mean == {want} (1e-12)",
                          got.n_classes == n and abs(got.value_float64 - float(want)) < 1e-12
                          and abs(got.value - float(want)) < 1e-6,
                          f"n={got.n_classes} f64={got.value_float64!r} f32={got.value!r}")
            # ================ E2 production float32 arithmetic, bitwise ================
            ref_all = miou_from_confusion(cm)
            ref_dis = miou_from_confusion(cm, torch.tensor([1, 2, 3]))
            check(f"E2 {model}: union-present values == metrics.miou_from_confusion BITWISE",
                  v[UNION_PRESENT]["all_class"].value == ref_all
                  and v[UNION_PRESENT]["disease_only"].value == ref_dis,
                  f"{v[UNION_PRESENT]['all_class'].value!r} vs {ref_all!r}")
            check(f"E2 {model}: sparse per-image triplets sum to the confusion-matrix totals",
                  np.array_equal(tot.tp, torch.diag(cm).numpy())
                  and np.array_equal(tot.gt, cm.sum(1).numpy())
                  and np.array_equal(tot.pred, cm.sum(0).numpy()))
        stats_a, _ = sparse_stats(GT, PRED_A, 4)
        va = rule_variants(stats_a)
        up, gp = va[UNION_PRESENT]["all_class"], va[GT_PRESENT]["all_class"]
        check("E1 A: GT-present == union-present x 4/3 (float64, 1e-12); class 3 is the extra one",
              abs(gp.value_float64 - up.value_float64 * 4 / 3) < 1e-12
              and set(up.eligible) - set(gp.eligible) == {3})
        table = per_class_table(stats_a)
        check("E1 A: per-class table: class 3 union-present, not GT-present, IoU 0.0; IoU exact",
              table[3] == {"class_id": 3, "tp": 0, "gt": 0, "pred": 3, "union": 3, "iou": 0.0,
                           "union_present": True, "gt_present": False, "iou_status": "ok"}
              and table[0]["iou"] == 7 / 10 and table[1]["iou"] == 3 / 5)
        stats_b, _ = sparse_stats(GT, PRED_B, 4)
        check("E1 B: class 3 absent from GT and prediction -> neither set, status undefined",
              per_class_table(stats_b)[3]["iou"] is None
              and per_class_table(stats_b)[3]["iou_status"] == "undefined_absent_from_gt_and_pred")

        # ================================ E3 failure modes ================================
        check("E3-1 unknown rule refused",
              "rule must be" in str(expect(EligibilityError, dataset_miou, stats_a, range(4), "mean")))
        bad = dict(stats_a)
        bad["tp"] = bad["tp"].copy()
        bad["tp"][0] = bad["gt"][0] + 1
        check("E3-2 a triplet with tp > gt refused",
              "tp > gt" in str(expect(EligibilityError, class_totals, bad)))
        bad2 = dict(stats_a)
        bad2["dataset_tp"] = bad2["dataset_tp"] + 1
        check("E3-3 sparse triplets that do not sum to dataset_* refused",
              "do not sum" in str(expect(EligibilityError, class_totals, bad2)))
        check("E3-4 empty or out-of-range class subsets refused",
              "non-empty subset" in str(expect(EligibilityError, dataset_miou, stats_a, [], GT_PRESENT))
              and "non-empty subset" in str(expect(EligibilityError, dataset_miou, stats_a, [4],
                                                   GT_PRESENT)))
        none = dataset_miou(stats_a, [3], GT_PRESENT)
        check("E3-5 nothing eligible -> value None (never NaN), n = 0",
              none.value is None and none.value_float64 is None and none.n_classes == 0)
        check("E3-6 num_classes is required without dataset totals",
              "num_classes is required" in str(expect(
                  EligibilityError, class_totals,
                  {k: stats_a[k] for k in ("class_id", "tp", "gt", "pred")})))

        # ================================ E4 the CLI end to end ================================
        g = torch.Generator().manual_seed(7)
        gts = torch.zeros(6, 8, 8, dtype=torch.long)
        present = [5, 12, 33, 47, 80, 115]                       # GT-present diseases
        for i in range(6):
            gts[i, 2:6, 2:6] = present[i]
        preds = gts.clone()
        noise = torch.rand(6, 8, 8, generator=g) < 0.2
        swap = torch.tensor(present)[torch.randint(0, 6, (6, 8, 8), generator=g)]
        preds[noise] = swap[noise]                               # confusions among GT classes
        preds[3, 0, 0] = 69                                      # predicted, never in GT
        art = core_artifact(work / "art", gts, preds, "cli")
        summary = json.loads((art / "summary.json").read_text(encoding="utf-8"))
        out = work / "derived"
        proc = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "eligibility_variants.py"),
                               str(art), "--out", str(out)], capture_output=True, text=True)
        target = out / "syn_gtpresent_cli_eligibility_variants.json"
        check("E4-1 CLI (subprocess) exits 0 and writes <run_id>_eligibility_variants.json",
              proc.returncode == 0 and target.is_file(), (proc.stdout + proc.stderr)[-200:])
        rep = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
        fa = rep.get("faithfulness", {})
        check("E4-2 R1: union-present all-class/disease-only == summary.json BITWISE",
              rep["rules"]["union_present"]["all_class"] == summary["dataset_level"]["all_class_miou"]
              and rep["rules"]["union_present"]["disease_only"]
              == summary["dataset_level"]["disease_only_miou"]
              and all(x["bitwise_equal"] for x in fa["R1_summary_bitwise"].values()))
        r3 = fa.get("R3_gt_present_relation", {})
        check("E4-3 R2 per-class equal; R3 holds with class 69 as the only union-present-only class",
              all(fa["R2_per_class_equal"].values()) and r3.get("holds") is True
              and r3.get("union_present_not_gt_present") == [69]
              and r3["n_union_present"] == r3["n_gt_present"] + 1 and r3["abs_diff"] <= 1e-12,
              {k: r3.get(k) for k in ("n_union_present", "n_gt_present", "abs_diff")})
        check("E4-4 output schema: artifact_sha256s, split, rules{union_present,gt_present}, per_class",
              set(rep["artifact_sha256s"]) == {"MANIFEST.sha256", "per_image.jsonl",
                                               "sufficient_stats.npz", "summary.json"}
              and rep["split"] == "val" and set(rep["rules"]) == {"union_present", "gt_present"}
              and all(set(rep["rules"][r]) >= {"all_class", "disease_only", "n_classes"}
                      for r in rep["rules"])
              and len(rep["per_class"]) == 116
              and set(rep["per_class"][0]) >= {"class_id", "tp", "gt", "pred", "union", "iou",
                                               "union_present", "gt_present"}
              and rep["artifact_sha256s"]["summary.json"]
              == hashlib.sha256((art / "summary.json").read_bytes()).hexdigest())
        before = target.read_bytes()
        rc, text = run_cli([str(art), "--out", str(out)])
        check("E4-5 an existing output is never overwritten (exit 2)",
              rc == 2 and "refusing to overwrite" in text and target.read_bytes() == before)
        value = rep["rules"]["union_present"]["all_class"]
        rc, text = run_cli([str(art), "--out", str(work / "exact"), "--expect-union-all-class",
                            repr(value)])
        check("E4-6 --expect-union-all-class with the exact value -> exit 0", rc == 0, text[-120:])
        rc, text = run_cli([str(art), "--out", str(work / "wrong"), "--expect-union-all-class",
                            repr(value + 1e-9)])
        check("E4-7 a value off by 1e-9 -> exit 1 and nothing written",
              rc == 1 and not (work / "wrong").exists(), text[-120:])
        f64 = float64_summary_artifact(work / "f64")
        rc, text = run_cli([str(f64), "--out", str(work / "f64_out")])
        check("E4-8 a summary not produced by the production reducer is refused as unfaithful (R1)",
              rc == 1 and "R1 union-present all_class" in text and not (work / "f64_out").exists(),
              text[-160:])
        tam = shutil.copytree(art, work / "tampered")
        (tam / "per_image.jsonl").write_bytes((tam / "per_image.jsonl").read_bytes() + b"\n")
        rc, text = run_cli([str(tam), "--out", str(work / "tam_out")])
        check("E4-9 a tampered artifact (manifest mismatch) -> exit 2, nothing written",
              rc == 2 and not (work / "tam_out").exists(), text[-120:])
        from src.stats.ingest import Policy, load_run
        run = load_run(art, Policy.NONOFFICIAL_SMOKE)
        up_run = dataset_miou(run.stats, range(116), UNION_PRESENT)
        check("E4-10 dataset_miou accepts the ingest's PerImageSufficientStats directly",
              up_run.value == summary["dataset_level"]["all_class_miou"])

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
    print(f"RESULT: {'L-AM17-GTPRESENT OK' if allok else 'L-AM17-GTPRESENT BLOCKED'} "
          f"({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic fixtures only; the real check d2 runs locally "
          "(scripts/eligibility_variants.py on R3 and DL-17 run1).")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
