#!/usr/bin/env python3
"""L-AM17-PERCLASS smoke (docs/lane_specs/part1.md lane 4 (d) d1; NONOFFICIAL, synthetic only).

  P1  d1 unit: synthetic per_class for two models (background + 6 diseases) and a synthetic strata file
      (terciles 2/2/2, classes 5 and 6 rare) -> the known table (deltas, statuses), the per-stratum,
      rare and overall summaries by hand (exact fractions to 1e-15), teacher-better / E1-better / tie
      counts, and the GT-absent class 6 (predicted by E1 only) is not_evaluable and in no summary.
  P2  integrity (d2 logic): per-class supports from the frozen metric code's confusion matrix reproduce
      miou_from_confusion BITWISE (all-class and disease-only); a summary whose mIoU is a float64 mean,
      or whose IoU column is altered, is reported; mismatched ground truth and a strata file that does
      not cover the classes are refused.
  P3  the CLI end to end on 846-row VAL artifacts written by the REAL evaluation core and writer (a
      throwaway git fixture is the repo root): the table is written once (115 rows + background, class
      69 not evaluable, d2 integrity bitwise, summaries equal an independent recomputation, input and
      strata sha256s recorded); two runs are byte-identical; a pre-lane and a 1.3.0 layout are read.
  P4  refusals: an upstream-protocol artifact (block or preprocess_protocol), a TEST-split, a smoke /
      random-init and a layout-2 artifact, swapped roles, another checkpoint, a metric-implementation
      mismatch, a smoke strata file, an existing output (exit 2); a summary mIoU the table cannot
      reproduce is a STOP (exit 1, nothing written).

Run:  python -B scripts/smoke_perclass_gap_table.py
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

from src.eval.artifacts import (ARTIFACT_FILES, MANIFEST_NAME, DatasetMeta, RunMeta,  # noqa: E402
                                prepare_artifact_request, validate_artifact_request, write_artifact)
from src.eval.evaluate import Condition, EvalBatch, ManifestEntry, evaluate_model  # noqa: E402
from src.eval.metrics import confusion_matrix, miou_from_confusion  # noqa: E402
from src.stats.perclass import (NOT_EVALUABLE, PerclassError, model_integrity,  # noqa: E402
                                perclass_rows, strata_index, summarize)

C = 116
CANVAS = "core_preprocess/1.0.0"
UTC = "2026-09-28T00:00:00Z"
SHA_T, SHA_E = "a" * 64, "b" * 64                      # synthetic checkpoint identities
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
# P1 fixture: per_class blocks for classes 0..6 (tp, gt, pred) and strata 2/2/2
# --------------------------------------------------------------------------------------------------
TOT_T = {0: (80, 100, 90), 1: (6, 10, 8), 2: (3, 6, 4), 3: (2, 8, 3), 4: (5, 5, 5), 5: (1, 4, 2),
         6: (0, 0, 0)}
TOT_E = {0: (78, 100, 92), 1: (5, 10, 7), 2: (3, 6, 4), 3: (4, 8, 5), 4: (4, 5, 4), 5: (0, 4, 0),
         6: (0, 0, 3)}
STRATA_SMALL = {"schema": "plantseg-train-strata/1.0.0",
                "per_class": [{"class_id": c, "tercile": t, "rare": c >= 5, "share": s, "rank": c}
                              for c, t, s in ((1, "T1", 0.3), (2, "T1", 0.25), (3, "T2", 0.2),
                                              (4, "T2", 0.15), (5, "T3", 0.1), (6, "T3", 0.0))]}


def per_class_block(tot: dict) -> dict:
    n = len(tot)
    pc = {k: [] for k in ("class_ids", "gt_support", "pred_support", "intersection", "union", "iou",
                          "dice", "acc", "iou_status")}
    for c in range(n):
        tp, gt, pr = tot[c]
        un = gt + pr - tp
        pc["class_ids"].append(c)
        pc["gt_support"].append(gt)
        pc["pred_support"].append(pr)
        pc["intersection"].append(tp)
        pc["union"].append(un)
        pc["iou"].append(tp / un if un else None)
        pc["dice"].append(None)
        pc["acc"].append(None)
        pc["iou_status"].append("ok" if un else "undefined_absent_from_gt_and_pred")
    return pc


def frac_iou(tot, c):
    tp, gt, pr = tot[c]
    return F(tp, gt + pr - tp)


# --------------------------------------------------------------------------------------------------
# 116-class VAL artifacts from the REAL evaluation core and writer
# --------------------------------------------------------------------------------------------------
def clean_git_fixture(root: Path) -> Path:
    """A throwaway repository with one commit and nothing dirty; never the project repository."""
    root.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-c", "user.name=perclass-fixture", "-c", "user.email=perclass@invalid",
                        "-c", "commit.gpgsign=false", *args], cwd=str(root), check=True,
                       capture_output=True)
    git("init", "-q")
    (root / "seed.txt").write_text("perclass fixture\n", encoding="utf-8")
    git("add", "--", "seed.txt")
    git("commit", "-q", "-m", "fixture baseline")
    return root


DISEASES = [c for c in range(1, C) if c != 69]           # class 69: never in ground truth


def label_maps(n=846, e1_extra=True, gt_shift=None):
    """GT: one 2x2 lesion per image (a cycle over 114 diseases); images 5 and 400 hold no disease.
    Teacher and E1 disagree with the GT in different places; E1 predicts class 69 once."""
    gts = torch.zeros(n, 4, 4, dtype=torch.long)
    for i in range(n):
        if i in (5, 400):
            continue
        gts[i, 1:3, 1:3] = DISEASES[i % len(DISEASES)] if gt_shift is None or i != gt_shift \
            else DISEASES[(i + 1) % len(DISEASES)]
    t, e = gts.clone(), gts.clone()
    for i in range(n):
        d = DISEASES[i % len(DISEASES)]
        if i % 5 == 0:
            t[i, 1, 1] = 0                                     # teacher misses a lesion pixel
        if i % 7 == 0:
            t[i, 0, 0] = d                                     # teacher false positive
        if i % 3 == 0:
            e[i, 2, 2] = DISEASES[(i + 1) % len(DISEASES)]     # E1 confuses a disease
        if i % 11 == 0:
            e[i, 1, 2] = 0
    if e1_extra:
        e[10, 3, 3] = 69                                       # predicted, never in GT
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


def write(out_dir, res, manifest, *, stage, role, sha, repo_root, status="provisional", split="val",
          preprocess_protocol=CANVAS):
    smoke = status == "smoke"
    req = prepare_artifact_request(
        out_dir=out_dir, artifact_status=status, run_id=f"syn_perclass_{out_dir.name}",
        run=RunMeta(stage=stage, model_role=role, precision="fp32", random_init=smoke,
                    checkpoint_path=None if smoke else f"synthetic_{stage}.pt",
                    checkpoint_sha256=None if smoke else sha, device="cpu"),
        dataset=DatasetMeta(name="SYNTHETIC perclass fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split=split, condition=Condition(),
                            preprocess_protocol=preprocess_protocol, expected_rows=len(manifest)),
        expected_manifest=manifest, class_map=SYN_CLASS_MAP, repo_root=repo_root)
    return write_artifact(res, req, validate_artifact_request(req), timestamp_utc=UTC)


def rehash(d: Path) -> None:
    (d / MANIFEST_NAME).write_text(
        "".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in ARTIFACT_FILES),
        encoding="utf-8", newline="\n")


def edited(src: Path, dst: Path, fn=None, rows_fn=None) -> Path:
    shutil.copytree(src, dst)
    if fn is not None:
        s = fn(json.loads((dst / "summary.json").read_text(encoding="utf-8")))
        (dst / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, allow_nan=False)
                                          + "\n", encoding="utf-8", newline="\n")
    if rows_fn is not None:
        rows = [json.loads(ln) for ln in (dst / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
        (dst / "per_image.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                     for r in rows_fn(rows)), encoding="utf-8",
                                             newline="\n")
    rehash(dst)
    return dst


def strata_file(path: Path, status="provisional") -> Path:
    mod = load("build_train_strata", "scripts/build_train_strata.py")
    ref = json.loads((REPO / "reports" / "e1_class_weights.json").read_text(encoding="utf-8"))
    rows = mod.build_rows(ref["pixel_counts"], [5367] + [10 + c for c in range(1, C)])
    doc = {"schema": mod.SCHEMA, "artifact_status": status, "split_list_sha256": "0" * 64,
           "per_class": rows, "tercile_sizes": [38, 38, 39], "script_commit": "0" * 40}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    print("=" * 100)
    print("L-AM17-PERCLASS -- per-class gap table smoke  [NONOFFICIAL: synthetic data only]")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="perclass_smoke_"))
    try:
        # ================================ P1 d1 by hand ================================
        pc_t, pc_e = per_class_block(TOT_T), per_class_block(TOT_E)
        strata = strata_index(STRATA_SMALL, range(1, 7))
        rows, bg = perclass_rows(pc_t, pc_e, strata)
        by = {r["class_id"]: r for r in rows}
        hand_delta = {c: frac_iou(TOT_T, c) - frac_iou(TOT_E, c) for c in range(1, 6)}
        check("P1-1 one row per disease class 1-6 with stratum, rare, gt_support, both IoUs and statuses",
              [r["class_id"] for r in rows] == [1, 2, 3, 4, 5, 6]
              and [r["stratum"] for r in rows] == ["T1", "T1", "T2", "T2", "T3", "T3"]
              and [r["rare"] for r in rows] == [False] * 4 + [True] * 2
              and [r["gt_support"] for r in rows] == [10, 6, 8, 5, 4, 0])
        check("P1-2 delta = iou_teacher - iou_e1 matches the exact fractions (1e-15)",
              all(abs(by[c]["delta"] - float(hand_delta[c])) < 1e-15 for c in range(1, 6)),
              {c: by[c]["delta"] for c in range(1, 6)})
        check("P1-3 class 6 (GT = 0, predicted by E1 only) is not_evaluable with a null delta",
              by[6]["status"] == NOT_EVALUABLE and by[6]["delta"] is None and by[6]["iou_teacher"] is None
              and by[6]["iou_e1"] == 0.0
              and by[6]["iou_status_teacher"] == "undefined_absent_from_gt_and_pred"
              and by[6]["iou_status_e1"] == "ok")
        check("P1-4 the background is one separate line (class 0), not a stratum row",
              bg["class_id"] == 0 and bg["stratum"] == "background"
              and abs(bg["delta"] - float(F(80, 110) - F(78, 114))) < 1e-15)
        s = summarize(rows)
        hand = {"T1": [hand_delta[1], hand_delta[2]], "T2": [hand_delta[3], hand_delta[4]],
                "T3": [hand_delta[5]], "rare": [hand_delta[5]],
                "overall": [hand_delta[c] for c in range(1, 6)]}
        ok = True
        for g, ds in hand.items():
            srt = sorted(ds)
            m = len(srt)
            med = srt[m // 2] if m % 2 else (srt[m // 2 - 1] + srt[m // 2]) / 2
            ok &= s[g]["n_classes"] == m and abs(s[g]["mean_delta"] - float(sum(ds) / m)) < 1e-15 \
                and abs(s[g]["median_delta"] - float(med)) < 1e-15
        check("P1-5 per-stratum, rare and overall n, mean and median by hand (1e-15)", ok, s)
        counts = {g: (s[g]["teacher_better"], s[g]["e1_better"], s[g]["ties"], s[g]["n_not_evaluable"])
                  for g in s}
        check("P1-6 teacher better / E1 better / ties (class 2 is an exact tie); not-evaluable counted",
              counts == {"T1": (1, 0, 1, 0), "T2": (1, 1, 0, 0), "T3": (1, 0, 0, 1), "rare": (1, 0, 0, 1),
                         "overall": (3, 1, 1, 1)}, counts)
        check("P1-7 a strata file missing a class, or with a bad tercile, is refused",
              "missing [6]" in str(expect(PerclassError, strata_index,
                                          {**STRATA_SMALL, "per_class": STRATA_SMALL["per_class"][:5]},
                                          range(1, 7)))
              and "tercile" in str(expect(PerclassError, strata_index,
                                          {**STRATA_SMALL, "per_class": [{**STRATA_SMALL["per_class"][0],
                                                                          "tercile": "T4"}]}, range(1, 2))))
        bad_gt = per_class_block({**TOT_E, 3: (4, 9, 5)})
        check("P1-8 two per_class blocks with different gt_support are refused (one VAL ground truth)",
              "gt_support differ" in str(expect(PerclassError, perclass_rows, pc_t, bad_gt, strata)))

        # ================================ P2 integrity ================================
        gts, pt, pe = label_maps(n=24)
        cm_t = sum(confusion_matrix(pt[i], gts[i], C) for i in range(24))
        tp, gt, pr = torch.diag(cm_t), cm_t.sum(1), cm_t.sum(0)
        blk = per_class_block({c: (int(tp[c]), int(gt[c]), int(pr[c])) for c in range(C)})
        un = [blk["union"][c] > 0 for c in range(C)]
        summ = {"per_class": blk, "dataset_level": {
            "all_class_miou": miou_from_confusion(cm_t), "all_class_miou_n_eligible": sum(un),
            "disease_only_miou": miou_from_confusion(cm_t, torch.arange(1, C)),
            "disease_only_miou_n_eligible": sum(un[1:])}}
        rec, probs = model_integrity(summ, blk["iou"])
        check("P2-1 table supports reproduce miou_from_confusion BITWISE (all-class and disease-only)",
              not probs and rec["all_class"]["bitwise_equal"] and rec["disease_only"]["bitwise_equal"]
              and rec["table_iou_equals_summary_and_quotient"], probs)
        check("P2-2 the float64 mean of the table IoU is recorded and differs only by float32 rounding",
              0 < rec["all_class"]["abs_diff_float64_vs_summary"] < 1e-6
              or rec["all_class"]["abs_diff_float64_vs_summary"] == 0.0, rec["all_class"])
        f64 = json.loads(json.dumps(summ))
        f64["dataset_level"]["all_class_miou"] = rec["all_class"]["float64_mean_of_table_iou"]
        _, probs = model_integrity(f64, blk["iou"])
        check("P2-3 a summary holding a float64 mean (not the evaluator's reducer) is reported",
              len(probs) == 1 and probs[0].startswith("all_class"), probs)
        alt = list(blk["iou"])
        alt[3] = alt[3] + 1e-12 if alt[3] is not None else 0.5
        _, probs = model_integrity(summ, alt)
        check("P2-4 an altered IoU column is reported", any("IoU differs" in p for p in probs), probs)

        # ================================ P3 the CLI end to end ================================
        cli = load("perclass_gap_table", "scripts/perclass_gap_table.py")
        cli.EXPECTED["teacher"]["checkpoint_sha256"] = SHA_T
        cli.EXPECTED["e1"]["checkpoint_sha256"] = SHA_E
        fx = clean_git_fixture(work / "fixture_repo")
        ids = [f"v{i:04d}" for i in range(846)]
        gts, pt, pe = label_maps()
        res_t, man = core_result(ids, gts, pt)
        res_e, _ = core_result(ids, gts, pe)
        art_t = write(work / "teacher_val", res_t, man, stage="teacher", role="teacher", sha=SHA_T,
                      repo_root=fx)
        art_e = write(work / "e1_val", res_e, man, stage="E1", role="student", sha=SHA_E, repo_root=fx)
        sfile = strata_file(work / "train_strata_v1.json")
        out = work / "derived"
        rc, text = in_process(cli, ["--teacher", str(art_t), "--e1", str(art_e), "--strata", str(sfile),
                                    "--out-dir", str(out), "--generated-utc", UTC])
        jp, cp = out / "perclass_gap_val_20260928T000000Z.json", out / "perclass_gap_val_20260928T000000Z.csv"
        check("P3-1 CLI exits 0 and writes perclass_gap_val_<UTC>.json and .csv", rc == 0 and jp.is_file()
              and cp.is_file(), text[-400:])
        doc = json.loads(jp.read_text(encoding="utf-8")) if jp.is_file() else {}
        rows = doc.get("rows", [])
        check("P3-2 115 disease rows + the background line; class 69 not evaluable (E1 IoU 0, teacher "
              "undefined)",
              len(rows) == 115 and doc["background"]["class_id"] == 0
              and doc["integrity"]["not_evaluable_classes"] == [69]
              and rows[68]["class_id"] == 69 and rows[68]["iou_e1"] == 0.0
              and rows[68]["iou_teacher"] is None)
        integ = doc.get("integrity", {})
        st = json.loads((art_t / "summary.json").read_text(encoding="utf-8"))
        se = json.loads((art_e / "summary.json").read_text(encoding="utf-8"))
        check("P3-3 d2 integrity: each model's table mean reproduces its summary mIoU BITWISE",
              integ["teacher"]["all_class"]["reproduced"] == st["dataset_level"]["all_class_miou"]
              and integ["e1"]["all_class"]["reproduced"] == se["dataset_level"]["all_class_miou"]
              and integ["teacher"]["all_class"]["bitwise_equal"] and integ["e1"]["all_class"]["bitwise_equal"]
              and integ["e1"]["all_class"]["n_union_present"]
              == integ["teacher"]["all_class"]["n_union_present"] + 1
              and integ["classes_in_exactly_one_stratum"])
        it, ie = np.array([x if x is not None else np.nan for x in st["per_class"]["iou"]]), \
            np.array([x if x is not None else np.nan for x in se["per_class"]["iou"]])
        gtv = np.array(st["per_class"]["gt_support"])
        sdoc = json.loads(sfile.read_text(encoding="utf-8"))
        terc = {r["class_id"]: r["tercile"] for r in sdoc["per_class"]}
        ok = True
        for g in ("T1", "T2", "T3", "overall"):
            cls = [c for c in range(1, C) if gtv[c] > 0 and (g == "overall" or terc[c] == g)]
            d = it[cls] - ie[cls]
            sg = doc["summaries"][g]
            ok &= sg["n_classes"] == len(cls) and abs(sg["mean_delta"] - float(np.mean(d))) < 1e-15 \
                and sg["median_delta"] == float(np.median(d)) and sg["teacher_better"] == int((d > 0).sum()) \
                and sg["e1_better"] == int((d < 0).sum()) and sg["ties"] == int((d == 0).sum())
        check("P3-4 summaries equal an independent numpy recomputation from the two summaries", ok,
              doc["summaries"]["overall"])
        check("P3-5 inputs recorded: the four files' sha256 of each artifact and the strata sha256",
              doc["inputs"]["teacher"]["artifact_sha256s"]["summary.json"]
              == hashlib.sha256((art_t / "summary.json").read_bytes()).hexdigest()
              and set(doc["inputs"]["e1"]["artifact_sha256s"]) == {*ARTIFACT_FILES, MANIFEST_NAME}
              and doc["inputs"]["strata"]["sha256"] == hashlib.sha256(sfile.read_bytes()).hexdigest())
        lines = cp.read_text(encoding="utf-8").splitlines()
        check("P3-6 CSV: header, 115 rows, background last; JSON binds the CSV hash",
              len(lines) == 117 and lines[-1].startswith("0,background,") and lines[1].startswith("1,")
              and doc["csv_sha256"] == hashlib.sha256(cp.read_bytes()).hexdigest())
        rc2, _ = in_process(cli, ["--teacher", str(art_t), "--e1", str(art_e), "--strata", str(sfile),
                                  "--out-dir", str(work / "derived2"), "--generated-utc", UTC])
        check("P3-7 two runs are byte-identical (JSON and CSV)", rc2 == 0
              and (work / "derived2" / jp.name).read_bytes() == jp.read_bytes()
              and (work / "derived2" / cp.name).read_bytes() == cp.read_bytes())
        pre = edited(art_t, work / "teacher_prelane",
                     fn=lambda s: {k: v for k, v in s.items() if k not in ("am5", "artifact_schema_version")},
                     rows_fn=lambda rows: [{k: v for k, v in r.items() if k != "am5_excluded"} for r in rows])
        v13 = edited(art_e, work / "e1_v13", fn=lambda s: {**s, "artifact_schema_version":
                                                           "plantseg-eval-artifact/1.3.0"})
        rc, text = in_process(cli, ["--teacher", str(pre), "--e1", str(v13), "--strata", str(sfile),
                                    "--out-dir", str(work / "derived3"), "--generated-utc", UTC])
        check("P3-8 a pre-lane teacher artifact (R3-like) and a 1.3.0 E1 layout are read (ingest rule)",
              rc == 0, text[-300:])

        # ================================ P4 refusals ================================
        def refused(label, t, e, *, strata=sfile, marker, code=2):
            o = work / f"out_{len(CHECKS)}"
            rc, text = in_process(cli, ["--teacher", str(t), "--e1", str(e), "--strata", str(strata),
                                        "--out-dir", str(o), "--generated-utc", UTC])
            check(label, rc == code and marker in text and not o.exists(), text[-260:])

        up = edited(art_t, work / "t_upstream", fn=lambda s: {
            **s, "protocol": {"name": "upstream", "version": "1.0.0"},
            "dataset": {**s["dataset"], "preprocess_protocol": "upstream/1.0.0"}})
        refused("P4-1 an upstream-protocol artifact is refused (canvas only)", up, art_e,
                marker="only canvas artifacts")
        up2 = edited(art_e, work / "e_pp", fn=lambda s: {**s, "dataset": {
            **s["dataset"], "preprocess_protocol": "upstream/1.0.0"}})
        refused("P4-2 a non-canvas preprocess_protocol without the block is refused", art_t, up2,
                marker="only canvas artifacts")
        tst = write(work / "e1_test", res_e, man, stage="E1", role="student", sha=SHA_E, repo_root=fx,
                    split="test")
        refused("P4-3 a TEST-split artifact is refused (REHEARSAL reads VAL only)", art_t, tst,
                marker="VAL artifacts only")
        smk = write(work / "e1_smoke", res_e, man, stage="E1", role="student", sha=None, repo_root=fx,
                    status="smoke")
        refused("P4-4 a smoke / random-init artifact is refused", art_t, smk, marker="real-run artifact")
        v2 = edited(art_e, work / "e_v2", fn=lambda s: {**s, "artifact_schema_version":
                                                        "plantseg-eval-artifact/2.0.0"})
        refused("P4-5 a layout-2 artifact is refused (the reader knows 1.x)", art_t, v2,
                marker="unsupported artifact_schema_version")
        refused("P4-6 swapped roles are refused", art_e, art_t, marker="expected stage/role/precision")
        other = edited(art_t, work / "t_other", fn=lambda s: {**s, "run": {**s["run"],
                                                                         "checkpoint_sha256": "c" * 64}})
        refused("P4-7 a teacher artifact of another checkpoint is refused", other, art_e,
                marker="is not the expected model")
        met = edited(art_e, work / "e_metric", fn=lambda s: {**s, "run": {**s["run"],
                                                                         "metric_impl_sha256": "d" * 64}})
        refused("P4-8 a metric-implementation mismatch is refused (REHEARSAL strictness)", art_t, met,
                marker="metric_impl_sha256 differs")
        refused("P4-9 a smoke strata file is refused", art_t, art_e,
                strata=strata_file(work / "strata_smoke.json", status="smoke"),
                marker="not the registered build")
        g2, _, pe2 = label_maps(gt_shift=100)
        res_g, _ = core_result(ids, g2, pe2)
        art_g = write(work / "e1_othergt", res_g, man, stage="E1", role="student", sha=SHA_E, repo_root=fx)
        refused("P4-10 an E1 artifact scored on different ground truth is refused", art_t, art_g,
                marker="gt_support differ")
        before = jp.read_bytes(), cp.read_bytes()
        rc, text = in_process(cli, ["--teacher", str(art_t), "--e1", str(art_e), "--strata", str(sfile),
                                    "--out-dir", str(out), "--generated-utc", UTC])
        check("P4-11 an existing output is never overwritten (exit 2; the table is unchanged)",
              rc == 2 and "refusing to overwrite" in text and (jp.read_bytes(), cp.read_bytes()) == before,
              text[-200:])
        tam = edited(art_t, work / "t_tampered", fn=lambda s: {**s, "dataset_level": {
            **s["dataset_level"], "all_class_miou": s["dataset_level"]["all_class_miou"] + 1e-9}})
        refused("P4-12 a summary mIoU the table cannot reproduce is a STOP (exit 1, nothing written)",
                tam, art_e, marker="d2 integrity failed", code=1)
        p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "perclass_gap_table.py"),
                            "--teacher", str(art_t), "--e1", str(art_e), "--strata", str(sfile),
                            "--out-dir", str(work / "sub")], capture_output=True, text=True)
        check("P4-13 the subprocess CLI (registered checkpoints) refuses the synthetic pair -> exit 2",
              p.returncode == 2 and "is not the expected model" in p.stdout and not (work / "sub").exists(),
              (p.stdout + p.stderr)[-200:])
        h = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "perclass_gap_table.py"), "--help"],
                           capture_output=True, text=True)
        check("P4-14 --help renders (exit 0)", h.returncode == 0 and "usage:" in h.stdout, h.stderr[-200:])
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    check("Z1 temporary synthetic artifacts removed", not work.exists())

    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if det and not ok:
            print(f"         {det[:220]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'L-AM17-PERCLASS OK' if allok else 'L-AM17-PERCLASS BLOCKED'} ({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic fixtures only; the real table (d2) runs locally after the strata file "
          "and DL-17 run1 exist (scripts/perclass_gap_table.py).")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
