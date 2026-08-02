#!/usr/bin/env python3
"""A2b — real-PlantSeg integration smoke: checkpoint loader, CLI guards, capped VALIDATION run.

Three sections:
  A. synthetic checkpoint-loader tests (temp files from a random-init student; NOT a real E1
     checkpoint and never used for the artifact)
  B. CLI guard / construction-order tests (pure argument + request validation; the test split is
     NEVER constructed, listed, or opened)
  C. capped REAL validation smoke -- 4 deterministic validation samples, batch 2, num_workers=0,
     random-init FP32 student, CPU, clean condition, artifact_status=smoke, temp output removed

Random-init metric values are smoke diagnostics ONLY and are never interpreted as performance.
No real checkpoint is loaded. The test split is never touched.

Run:  set PYTHONIOENCODING=utf-8 && C:\\Users\\admin\\anaconda3\\python.exe scripts/smoke_eval_plantseg.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np      # noqa: E402
import torch            # noqa: E402
from PIL import Image   # noqa: E402

import evaluate_model as CLI                                            # noqa: E402
from src.eval import evaluate_model as core_evaluate                    # noqa: E402
from src.eval import Condition, EvaluationIntegrityError, ManifestEntry  # noqa: E402
from src.eval.adapters import (CLASS_MAP_SEMANTIC_SHA256, PlantSegEvalDataset,  # noqa: E402
                               build_eval_loader, build_expected_manifest,
                               build_expected_manifest_for, deterministic_subset,
                               load_class_map)
from src.eval.artifacts import (ArtifactRequestError, validate_artifact_request,  # noqa: E402
                                verify_artifact)
from src.eval.metrics import (confusion_matrix, dice_from_confusion,   # noqa: E402
                              macc_from_confusion, miou_from_confusion, per_image_miou)
from src.eval.model_loading import (TORCH_LOAD_WEIGHTS_ONLY, CheckpointError,  # noqa: E402
                                    build_fp32_student, load_student_checkpoint, sha256_file)

C = 116
N_SAMPLES = 4
BATCH = 2
TOL = 1e-6

CHECKS: list[tuple[str, bool, str]] = []
OPENED: list[str] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def expect_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}") from e
    raise AssertionError("expected an exception, none raised")


def ns(**kw):
    """Minimal argparse-like namespace with CLI defaults."""
    base = dict(stage="E1", model_role="student", precision="fp32", split="val",
                condition="clean", corruption_severity=None, out_dir="X", checkpoint=None,
                random_init=True, artifact_status="smoke", batch_size=2, max_samples=None,
                confirm_test_split=False, run_id="t", device="cpu")
    base.update(kw)
    return type("A", (), base)()


# --------------------------------------------------------------------------------------------------
def section_a_checkpoint_loader(work: Path):
    """Synthetic checkpoints only -- proves the loader, never a real E1 checkpoint."""
    model = build_fp32_student()
    sd = model.state_dict()

    good = work / "good.pt"
    torch.save({"iter": 4000, "model_state_dict": sd, "num_classes": 116,
                "best_val_miou_all_class": 0.123}, good)
    m, info = load_student_checkpoint(good, map_location="cpu")
    check("A1 valid nested synthetic checkpoint loads strict=True",
          m is not None and info.num_classes == 116 and info.iteration == 4000)
    check("A7 recorded sha256 equals an independent raw-file re-hash",
          info.sha256 == sha256_file(good), info.sha256[:16] + "...")

    bare = work / "bare.pt"
    torch.save(sd, bare)
    e = expect_raises(CheckpointError, load_student_checkpoint, bare)
    check("A2 bare state_dict rejected", "BARE state_dict" in str(e), str(e)[:90])

    nomodel = work / "nomodel.pt"
    torch.save({"iter": 1, "num_classes": 116}, nomodel)
    e = expect_raises(CheckpointError, load_student_checkpoint, nomodel)
    check("A3 missing model_state_dict rejected", "model_state_dict" in str(e), str(e)[:90])

    nocls = work / "nocls.pt"
    torch.save({"model_state_dict": sd}, nocls)
    e = expect_raises(CheckpointError, load_student_checkpoint, nocls)
    check("A4 missing num_classes rejected", "num_classes" in str(e), str(e)[:90])

    wrongn = work / "wrongn.pt"
    torch.save({"model_state_dict": sd, "num_classes": 6}, wrongn)
    e = expect_raises(CheckpointError, load_student_checkpoint, wrongn)
    check("A5 wrong num_classes rejected", "!= required 116" in str(e), str(e)[:90])

    bad = dict(sd)
    k = next(iter(bad))
    bad[k] = torch.zeros(1)
    incompat = work / "incompat.pt"
    torch.save({"model_state_dict": bad, "num_classes": 116}, incompat)
    e = expect_raises(CheckpointError, load_student_checkpoint, incompat)
    check("A6 incompatible state_dict rejected (strict)", "strict state_dict load failed" in str(e),
          str(e)[:90])

    check("A8 torch.load weights_only set explicitly (not a library default)",
          TORCH_LOAD_WEIGHTS_ONLY is True, f"weights_only={TORCH_LOAD_WEIGHTS_ONLY}")


# --------------------------------------------------------------------------------------------------
def section_b_cli_guards(work: Path):
    """Pure argument/request validation. The test split is never constructed or listed."""
    cases = [
        ("B1 unsupported teacher role rejected", ns(model_role="teacher"), "not implemented"),
        ("B2 unsupported INT8 precision rejected", ns(precision="int8_ptq"), "not implemented"),
        ("B3 unsupported corruption condition rejected", ns(condition="fog"), "not implemented"),
        ("B4 random-init + checkpoint rejected", ns(random_init=True, checkpoint="x.pt"),
         "mutually exclusive"),
        ("B5 neither random-init nor checkpoint rejected", ns(random_init=False, checkpoint=None),
         "exactly one of"),
        ("B6 random-init with non-smoke status rejected",
         ns(random_init=True, artifact_status="official"), "requires --artifact-status smoke"),
        ("B7 unconfirmed test rejected", ns(split="test", random_init=False, checkpoint="x.pt",
                                            artifact_status="provisional"), "confirm-test-split"),
        ("B8 random-init test rejected", ns(split="test", confirm_test_split=True,
                                            random_init=True), "forbids --random-init"),
        ("B9 capped test rejected", ns(split="test", confirm_test_split=True, random_init=False,
                                       checkpoint="x.pt", max_samples=4,
                                       artifact_status="provisional"), "forbids any sample cap"),
        ("B10 smoke-status test rejected", ns(split="test", confirm_test_split=True,
                                              random_init=False, checkpoint="x.pt",
                                              artifact_status="smoke"), "refuses artifact_status=smoke"),
    ]
    for name, args, needle in cases:
        e = expect_raises(CLI.CliError, CLI.validate_cli_args, args)
        check(name, needle in str(e), str(e)[:100])

    # construction counters: a rejected run must construct NO dataset and NO model
    for label, args in (("teacher role", ns(model_role="teacher", out_dir=str(work / "n1"))),
                        ("capped test", ns(split="test", confirm_test_split=True,
                                           random_init=False, checkpoint="x.pt", max_samples=4,
                                           artifact_status="provisional",
                                           out_dir=str(work / "n2")))):
        ctr = CLI.Counters(dataset=[], model=[])
        expect_raises(CLI.CliError, CLI.run, args, counters=ctr)
        check(f"B11 no dataset/model construction after rejection ({label})",
              not ctr.dataset and not ctr.model,
              f"dataset={len(ctr.dataset)} model={len(ctr.model)}")
        check(f"B11b no output directory created ({label})", not Path(args.out_dir).exists())

    # official refusal (governed paths dirty) must also precede construction
    ctr = CLI.Counters(dataset=[], model=[])
    e = expect_raises(ArtifactRequestError, CLI.run,
                      ns(random_init=False, checkpoint=str(work / "good.pt"),
                         artifact_status="official", max_samples=N_SAMPLES,
                         out_dir=str(work / "n3")), counters=ctr)
    check("B12 official refused before dataset/model construction",
          "governed paths are dirty" in str(e) and not ctr.dataset and not ctr.model,
          f"dataset={len(ctr.dataset)} model={len(ctr.model)}")
    check("B12b no output directory after official refusal", not (work / "n3").exists())


# --------------------------------------------------------------------------------------------------
def section_c_capped_validation(work: Path):
    """Capped REAL validation smoke. 4 deterministic samples; the test split is never touched."""
    cm = load_class_map()
    check("C1 authoritative class map loaded and hash-verified",
          cm.semantic_sha256 == CLASS_MAP_SEMANTIC_SHA256 and len(cm.entries) == 116,
          f"{cm.semantic_sha256[:20]}... commit={cm.source_commit_sha[:12]}")

    indices = deterministic_subset(846, N_SAMPLES)
    manifest = build_expected_manifest_for("val", indices)
    check("C2 capped subset resolved BEFORE any loader", indices == sorted(indices)
          and len(indices) == N_SAMPLES, f"source indices = {indices}")
    check("C3 expected manifest describes exactly the capped subset",
          [m.manifest_index for m in manifest] == indices and len(manifest) == N_SAMPLES,
          "; ".join(f"{m.manifest_index}:{m.image_id}" for m in manifest))
    check("C4 image_id == clean_image_id for the clean condition",
          all(m.image_id == m.clean_image_id for m in manifest))

    out = work / "capped_val_smoke"
    args = ns(out_dir=str(out), max_samples=N_SAMPLES, batch_size=BATCH,
              run_id="a2b_capped_val_smoke_randominit")
    ctr = CLI.Counters(dataset=[], model=[])
    # C18/C19 below compare this run's metrics against an independent reference computation. Both
    # sides build a random-init student, so both constructions must start from the same RNG state
    # or the two models get different weights and the comparison is meaningless. Verified: CLI.run
    # consumes no torch RNG between this seed and its own build_fp32_student() call.
    torch.manual_seed(0)
    art = CLI.run(args, counters=ctr)
    check("C5 CLI produced an artifact", art.exists() and art == out, str(art.name))
    check("C6 dataset+model constructed exactly once, after preflight",
          len(ctr.dataset) == 1 and len(ctr.model) == 1)

    summary = verify_artifact(art)
    rows = [json.loads(l) for l in (art / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
    txt = (art / "summary.json").read_text(encoding="utf-8") + \
        (art / "per_image.jsonl").read_text(encoding="utf-8")

    check("C7 four artifact files validate + manifest hashes verify",
          sorted(p.name for p in art.iterdir()) ==
          ["MANIFEST.sha256", "per_image.jsonl", "sufficient_stats.npz", "summary.json"])
    check("C8 no NaN/Infinity token in JSON", not re.findall(r"\b(NaN|-?Infinity)\b", txt))
    check("C9 expected_rows == actual_rows == 4",
          summary["dataset"]["expected_rows"] == N_SAMPLES
          and summary["dataset"]["actual_rows"] == N_SAMPLES)
    check("C10 artifact self-identifies as random-init smoke, no checkpoint",
          summary["run"]["artifact_status"] == "smoke" and summary["run"]["random_init"] is True
          and summary["run"]["checkpoint_path"] is None
          and summary["run"]["checkpoint_sha256"] is None)
    check("C11 class_map_sha256 equals the pinned authoritative value",
          summary["dataset"]["class_map_sha256"] == CLASS_MAP_SEMANTIC_SHA256,
          summary["dataset"]["class_map_sha256"])
    check("C12 real dataset identity + validation preprocessing protocol",
          summary["dataset"]["name"] == "PlantSeg" and summary["dataset"]["split"] == "val"
          and summary["dataset"]["preprocess_protocol"] == "core_preprocess/1.0.0"
          and summary["dataset"]["num_classes"] == 116,
          summary["dataset"]["preprocess_protocol"])
    check("C13 all classwise arrays length 116",
          all(len(summary["per_class"][k]) == 116 for k in
              ("class_ids", "gt_support", "pred_support", "intersection", "union",
               "iou", "dice", "acc", "iou_status")))

    got_idx = [r["manifest_index"] for r in rows]
    got_ids = [r["image_id"] for r in rows]
    check("C14 rows canonically ordered by stable external manifest_index",
          got_idx == sorted(got_idx) == indices, f"{got_idx}")
    check("C15 real validation stems attached as image_id",
          got_ids == [m.image_id for m in manifest], "; ".join(got_ids))
    check("C16 image_id == clean_image_id in every row",
          all(r["image_id"] == r["clean_image_id"] for r in rows))

    # --- metric parity against src.eval.metrics on the SAME four rows ---
    adapter = PlantSegEvalDataset("val", indices)
    check("C17 adapter-derived manifest matches the listing-derived manifest",
          build_expected_manifest(adapter) == manifest)
    loader = build_eval_loader(adapter, BATCH, num_workers=0)
    preds, tgts = [], []
    torch.manual_seed(0)                 # MUST precede construction: seeding after it is a no-op
    model = build_fp32_student()
    with torch.no_grad():
        for b in loader:
            preds.append(model(b.images).argmax(1))
            tgts.append(b.targets)
    P, T = torch.cat(preds), torch.cat(tgts)
    ref_cm = confusion_matrix(P, T, C, 255)
    dis = torch.tensor([c for c in range(C) if c != 0])
    ref = {"all_class_miou": miou_from_confusion(ref_cm),
           "disease_only_miou": miou_from_confusion(ref_cm, dis),
           "all_class_macc": macc_from_confusion(ref_cm),
           "all_class_macro_dice": dice_from_confusion(ref_cm)}
    dl = summary["dataset_level"]
    check("C18 dataset aggregate metrics match src.eval.metrics",
          all(abs(dl[k] - v) <= TOL for k, v in ref.items()),
          ", ".join(f"{k}={dl[k]:.6f}" for k in ref))
    ra = per_image_miou(P, T, C, None, 255)
    rd = per_image_miou(P, T, C, dis, 255)
    ok = all(abs(rows[i]["all_class_miou"] - ra[i]) <= TOL
             and abs(rows[i]["disease_only_miou"] - rd[i]) <= TOL for i in range(len(rows)))
    check("C19 per-image metrics match src.eval.metrics", ok)

    with np.load(art / "sufficient_stats.npz", allow_pickle=False) as z:
        recon = True
        for nme in ("tp", "gt", "pred"):
            acc = np.zeros(C, dtype=np.int64)
            np.add.at(acc, z["class_id"], z[nme])
            if not np.array_equal(acc, z[f"dataset_{nme}"]):
                recon = False
        check("C20 sparse sufficient statistics sum to dataset totals", recon)
        mids = [str(x) for x in z["manifest_ids"]]
        check("C21 NPZ image_index -> manifest_ids row position (sparse external indices)",
              mids == got_ids and int(z["image_index"].max()) < N_SAMPLES,
              f"manifest_ids={mids[:2]}... external={indices}")

    e = expect_raises(ArtifactRequestError, validate_artifact_request, _same_request(out))
    check("C22 overwrite of an existing artifact refused", "refusing to overwrite" in str(e),
          str(e)[:80])
    return art


def _same_request(out: Path):
    from src.eval import Condition, DatasetMeta, RunMeta
    from src.eval.adapters import (DATASET_DOI, DATASET_NAME, PREPROCESS_PROTOCOL,
                                   build_expected_manifest_for, deterministic_subset,
                                   load_class_map)
    from src.eval.artifacts import prepare_artifact_request
    idx = deterministic_subset(846, N_SAMPLES)
    return prepare_artifact_request(
        out_dir=out, artifact_status="smoke", run_id="dup",
        run=RunMeta(stage="E1", model_role="student", precision="fp32", random_init=True),
        dataset=DatasetMeta(name=DATASET_NAME, doi=DATASET_DOI, split="val",
                            condition=Condition("clean"),
                            preprocess_protocol=PREPROCESS_PROTOCOL, expected_rows=N_SAMPLES),
        expected_manifest=build_expected_manifest_for("val", idx),
        class_map=list(load_class_map().entries), repo_root=REPO)


def section_d_manifest_divergence():
    """The adapter and the expected manifest must not be able to diverge silently."""
    idx = deterministic_subset(846, N_SAMPLES)
    adapter = PlantSegEvalDataset("val", idx)
    loader = build_eval_loader(adapter, BATCH, num_workers=0)
    model = build_fp32_student()

    wrong_subset = build_expected_manifest_for("val", [i + 1 for i in idx])
    e = expect_raises(EvaluationIntegrityError, core_evaluate, model, loader,
                      expected_manifest=wrong_subset, condition=Condition("clean"),
                      num_classes=C, background_index=0, ignore_index=255)
    check("D1 adapter/manifest subset divergence rejected", "manifest" in str(e).lower(),
          str(e)[:110])

    # Swap the IDENTITIES of the first two entries while keeping both indices and both ids
    # present exactly once -- so this tests a wrong id/index PAIRING, not a duplicate id.
    good = build_expected_manifest_for("val", idx)
    swapped = [ManifestEntry(good[0].manifest_index, good[1].image_id, good[1].clean_image_id),
               ManifestEntry(good[1].manifest_index, good[0].image_id, good[0].clean_image_id),
               *good[2:]]
    e = expect_raises(EvaluationIntegrityError, core_evaluate, model,
                      build_eval_loader(PlantSegEvalDataset("val", idx), BATCH, num_workers=0),
                      expected_manifest=swapped, condition=Condition("clean"),
                      num_classes=C, background_index=0, ignore_index=255)
    check("D2 wrong id/index pairing rejected", "paired with image_id" in str(e), str(e)[:110])


# --------------------------------------------------------------------------------------------------
def main() -> int:
    print("=" * 100)
    print("A2b -- real-PlantSeg integration smoke (capped VALIDATION only; test split untouched)")
    print(f"torch {torch.__version__} | numpy {np.__version__} | CPU")
    print("=" * 100)

    work = Path(tempfile.mkdtemp(prefix="a2b_"))
    real_open = Image.open

    def tracking_open(fp, *a, **kw):
        try:
            OPENED.append(str(fp))
        except Exception:                                    # noqa: BLE001
            pass
        return real_open(fp, *a, **kw)

    art = None
    try:
        section_a_checkpoint_loader(work)
        section_b_cli_guards(work)
        Image.open = tracking_open                          # narrowest real dataset boundary
        try:
            art = section_c_capped_validation(work)
            section_d_manifest_divergence()
        finally:
            Image.open = real_open                          # restored even on failure

        imgs = [p for p in OPENED if "/images/val/" in p.replace("\\", "/")]
        masks = [p for p in OPENED if "/annotations/val/" in p.replace("\\", "/")]
        tests = [p for p in OPENED if "/images/test/" in p.replace("\\", "/")
                 or "/annotations/test/" in p.replace("\\", "/")]
        uniq_i = sorted({Path(p).stem for p in imgs})
        uniq_m = sorted({Path(p).stem for p in masks})
        expected_stems = [m.image_id for m in build_expected_manifest_for(
            "val", deterministic_subset(846, N_SAMPLES))]
        check("E1 exactly 4 distinct validation images opened", len(uniq_i) == N_SAMPLES,
              f"{len(imgs)} opens, {len(uniq_i)} distinct")
        check("E2 exactly 4 distinct validation masks opened", len(uniq_m) == N_SAMPLES,
              f"{len(masks)} opens, {len(uniq_m)} distinct")
        check("E3 ZERO test paths opened", not tests, f"{len(tests)}")
        check("E4 every opened sample is in the selected subset",
              uniq_i == sorted(expected_stems) and uniq_m == sorted(expected_stems))
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        check("FATAL: smoke raised", False, traceback.format_exc(limit=2))
    finally:
        Image.open = real_open
        shutil.rmtree(work, ignore_errors=True)

    check("F1 temporary artifact directory removed after inspection",
          art is None or not art.exists())

    print()
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail[:170]}")
    n = sum(1 for _, ok, _ in CHECKS if ok)
    ok_all = n == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {n}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'A2b OK -- REAL VALIDATION PATH PROVEN' if ok_all else 'A2b BLOCKED'}")
    print("Random-init metrics are smoke diagnostics only -- NOT performance results.")
    print("No real checkpoint was loaded; the test split was never opened.")
    print("=" * 100)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
