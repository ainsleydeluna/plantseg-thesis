#!/usr/bin/env python3
"""A2a — synthetic contract smoke for the stage-neutral evaluator core + artifact writer.

SYNTHETIC FIXTURE ONLY. No PlantSeg image, mask, split file, checkpoint, or dataloader is
opened. Every tensor here is constructed inline. The fixture uses the FROZEN 116-class space
(background 0, diseases 1..115, ignore 255) with tiny 4x4 spatial tensors, so the real
plantseg-eval/1.0.0 artifact shape is exercised at negligible cost.

Model: a deterministic, untrained, no-checkpoint dummy that emits fixed one-hot logits and
counts its forward calls (used to prove official-mode refusal happens BEFORE inference).

Run:  set PYTHONIOENCODING=utf-8 && python -B scripts/smoke_eval_contract.py
Exit: 0 only if every check passes.
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

import numpy as np   # noqa: E402
import torch         # noqa: E402

from src.eval import (Condition, DatasetMeta, EvalBatch,  # noqa: E402
                      EvaluationIntegrityError, ManifestEntry, RunMeta, evaluate_model)
from src.eval.artifacts import (ArtifactRequestError, ArtifactWriteError,  # noqa: E402
                                canonical_json_bytes, prepare_artifact_request,
                                validate_artifact_request, verify_artifact, write_artifact)
from src.eval.metrics import (confusion_matrix, dice_from_confusion,  # noqa: E402
                              macc_from_confusion, miou_from_confusion, per_image_miou)

C = 116          # FROZEN class count -- the artifact contract requires exactly this
BG = 0
IGNORE = 255
TOL = 1e-6       # float32 reduction tolerance (contract case C4)

CHECKS: list[tuple[str, bool, str]] = []

# Instrument builtins.open so check 32 can PROVE no PlantSeg file was touched, rather than
# asserting it from the absence of an import.
OPENED_PATHS: list[str] = []
_REAL_OPEN = __builtins__["open"] if isinstance(__builtins__, dict) else __builtins__.open


def _tracking_open(file, *a, **kw):
    try:
        OPENED_PATHS.append(str(file))
    except Exception:                                    # noqa: BLE001
        pass
    return _REAL_OPEN(file, *a, **kw)


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


# --------------------------------------------------------------------------------------------------
# synthetic fixture (hand-computable; see reports/a2a_evaluator_core_smoke.md)
# --------------------------------------------------------------------------------------------------
def _t(rows):
    return torch.tensor(rows, dtype=torch.long)


FIXTURE = [
    # (manifest_index, image_id, target 4x4, pred 4x4)
    (0, "syn_0000_normal",
     _t([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
     _t([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])),
    (1, "syn_0001_multidisease",
     _t([[0, 0, 2, 2], [0, 0, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]]),
     _t([[0, 0, 2, 3], [0, 0, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]])),
    (2, "syn_0002_all_ignored",
     _t([[255] * 4] * 4),
     _t([[0] * 4] * 4)),
    (3, "syn_0003_fp_only_class",
     _t([[0, 0, 0, 0], [0, 4, 4, 0], [0, 4, 4, 0], [0, 0, 0, 0]]),
     _t([[5, 5, 0, 0], [0, 4, 4, 0], [0, 4, 4, 0], [0, 0, 0, 0]])),
    (4, "syn_0004_missed_disease",
     _t([[0, 0, 0, 0], [0, 6, 6, 0], [0, 6, 6, 0], [0, 0, 0, 0]]),
     _t([[0] * 4] * 4)),
    (5, "syn_0005_single_disease",
     _t([[7, 7, 0, 0], [7, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
     _t([[7, 7, 0, 0], [7, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
]

MANIFEST = [ManifestEntry(i, sid, sid) for i, sid, _, _ in FIXTURE]   # clean: id == clean_id
PRED_BY_INDEX = {i: p for i, _, _, p in FIXTURE}
TARGET_BY_INDEX = {i: t for i, _, t, _ in FIXTURE}

SYNTHETIC_CLASS_MAP = [
    {"id": 0, "name": "synthetic_background", "role": "background"},
    *[{"id": c, "name": f"synthetic_disease_{c:03d}", "role": "disease"} for c in range(1, C)],
]


class DummySegModel(torch.nn.Module):
    """Deterministic, untrained, NO-CHECKPOINT model. Counts forward calls."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        b, _, h, w = x.shape
        out = torch.zeros(b, self.num_classes, h, w)
        for i in range(b):
            idx = int(x[i, 0, 0, 0].item())
            out[i].scatter_(0, PRED_BY_INDEX[idx].unsqueeze(0), 1.0)
        return out


def _image_for(idx: int) -> torch.Tensor:
    """Image encodes its manifest index so the dummy model is order-independent."""
    img = torch.zeros(1, 4, 4)
    img[0, 0, 0] = float(idx)
    return img


def make_batches(order: list[list[int]]) -> list[EvalBatch]:
    """`order` is a list of batches, each a list of manifest indices -- deliberately shuffled."""
    out = []
    for group in order:
        imgs = torch.stack([_image_for(i) for i in group])
        tgts = torch.stack([TARGET_BY_INDEX[i] for i in group])
        ids = [MANIFEST[i].image_id for i in group]
        out.append(EvalBatch(imgs, tgts, ids, list(ids), list(group)))
    return out


# deliberately NON-canonical input order: [4,1] then [5,0,3] then [2]
NONCANONICAL = [[4, 1], [5, 0, 3], [2]]


def make_request(out_dir: Path, *, status="smoke", random_init=True,
                 checkpoint_path=None, checkpoint_sha256=None, expected_rows=None):
    return prepare_artifact_request(
        out_dir=out_dir,
        artifact_status=status,
        run_id="a2a_synthetic_contract_smoke",
        run=RunMeta(stage="E1", model_role="student", precision="fp32", quant_backend=None,
                    checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256,
                    random_init=random_init, device="cpu"),
        dataset=DatasetMeta(
            name="SYNTHETIC contract fixture (NOT PlantSeg data)",
            doi="10.5281/zenodo.17719108",
            split="val",
            condition=Condition("clean"),
            preprocess_protocol="synthetic-contract/1.0.0",
            expected_rows=expected_rows if expected_rows is not None else len(MANIFEST)),
        expected_manifest=MANIFEST,
        class_map=SYNTHETIC_CLASS_MAP,
        num_classes=C, background_index=BG, ignore_index=IGNORE,
        repo_root=REPO)


def run_eval(model, batches):
    return evaluate_model(model, batches, expected_manifest=MANIFEST,
                          condition=Condition("clean"), num_classes=C,
                          background_index=BG, ignore_index=IGNORE)


def expect_raises(exc_types, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_types as e:
        return e
    except Exception as e:                                # noqa: BLE001
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}") from e
    raise AssertionError("expected an exception, none raised")


# --------------------------------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    print("=" * 100)
    print("A2a -- synthetic evaluator-core + artifact-contract smoke")
    print(f"torch {torch.__version__} | numpy {np.__version__} | python {sys.version.split()[0]} | CPU")
    print(f"classes={C} background={BG} ignore={IGNORE} | SYNTHETIC FIXTURE, no PlantSeg data")
    print("=" * 100)

    import builtins
    builtins.open = _tracking_open                       # track every file opened from here on
    workdir = Path(tempfile.mkdtemp(prefix="a2a_eval_"))
    try:
        # ---------- 17. official refusal BEFORE inference ----------
        model_gate = DummySegModel(C)
        off_dir = workdir / "official_attempt"
        err = expect_raises(ArtifactRequestError, validate_artifact_request,
                            make_request(off_dir, status="official", random_init=False,
                                         checkpoint_path="x.pt", checkpoint_sha256="0" * 64))
        check("17 official refused by pre-inference gate", "governed paths are dirty" in str(err),
              str(err)[:150])
        check("17 forward-call count is zero at refusal", model_gate.forward_calls == 0,
              f"forward_calls={model_gate.forward_calls}")
        check("17 no target dir created", not off_dir.exists())
        check("17 no temp sibling left",
              not list(workdir.glob(".official_attempt.tmp-*")))

        # ---------- evaluate (inference happens only after the gate) ----------
        model = DummySegModel(C)
        req = make_request(workdir / "run1")
        prov = validate_artifact_request(req)
        check("smoke request accepted with dirty governed paths",
              prov.governed_paths_clean is False and len(prov.governed_violations) > 0,
              f"violations={list(prov.governed_violations)[:4]}")
        result = run_eval(model, make_batches(NONCANONICAL))
        check("model was invoked after the gate", model.forward_calls == len(NONCANONICAL),
              f"forward_calls={model.forward_calls}")

        # ---------- 6. deterministic ordering despite shuffled input ----------
        idxs = [r.manifest_index for r in result.rows]
        check("6 rows sorted by manifest_index despite non-canonical input",
              idxs == sorted(idxs) == list(range(len(MANIFEST))), f"{idxs}")

        # ---------- 9/10. metric parity vs src.eval.metrics ----------
        all_pred = torch.stack([PRED_BY_INDEX[i] for i in range(len(FIXTURE))])
        all_tgt = torch.stack([TARGET_BY_INDEX[i] for i in range(len(FIXTURE))])
        cm_ref = confusion_matrix(all_pred, all_tgt, C, IGNORE)
        dis_idx = torch.tensor([c for c in range(C) if c != BG])
        ref = {
            "all_class_miou": miou_from_confusion(cm_ref),
            "disease_only_miou": miou_from_confusion(cm_ref, dis_idx),
            "all_class_macc": macc_from_confusion(cm_ref),
            "all_class_macro_dice": dice_from_confusion(cm_ref),
        }
        parity = {k: abs(result.dataset_level[k] - v) <= TOL for k, v in ref.items()}
        check("9 dataset metric parity with src.eval.metrics", all(parity.values()),
              ", ".join(f"{k}={result.dataset_level[k]:.7f}" for k in ref))

        ref_all = per_image_miou(all_pred, all_tgt, C, None, IGNORE)
        ref_dis = per_image_miou(all_pred, all_tgt, C, dis_idx, IGNORE)
        pi_ok = True
        for r in result.rows:
            i = r.manifest_index
            a, d = ref_all[i], ref_dis[i]
            av = None if a != a else a
            dv = None if d != d else d
            if (r.all_class_miou is None) != (av is None) or (r.disease_only_miou is None) != (dv is None):
                pi_ok = False
            if av is not None and abs(r.all_class_miou - av) > TOL:
                pi_ok = False
            if dv is not None and abs(r.disease_only_miou - dv) > TOL:
                pi_ok = False
        check("10 per-image metric parity with src.eval.metrics", pi_ok)

        # hand-computed anchors (see report section 6)
        check("hand-anchor dataset all-class mIoU == 0.68125",
              abs(result.dataset_level["all_class_miou"] - 0.68125) <= TOL,
              f"{result.dataset_level['all_class_miou']!r}")
        check("hand-anchor dataset disease-only mIoU == 0.65",
              abs(result.dataset_level["disease_only_miou"] - 0.65) <= TOL,
              f"{result.dataset_level['disease_only_miou']!r}")
        check("hand-anchor aAcc == 73/80 == 0.9125",
              abs(result.dataset_level["aacc_diagnostic"] - 0.9125) <= TOL,
              f"{result.dataset_level['aacc_diagnostic']!r}")
        check("FP-only class 5 counted in mIoU denominator (8 eligible)",
              result.dataset_level["all_class_miou_n_eligible"] == 8,
              f"n_eligible={result.dataset_level['all_class_miou_n_eligible']}")
        check("FP-only class 5 EXCLUDED from mAcc denominator (7 eligible)",
              result.dataset_level["all_class_macc_n_eligible"] == 7,
              f"n_eligible={result.dataset_level['all_class_macc_n_eligible']}")

        # ---------- 14. undefined -> null + status ----------
        row2 = next(r for r in result.rows if r.manifest_index == 2)
        check("14 all-ignored image -> null + undefined_no_eligible_class",
              row2.all_class_miou is None and row2.disease_only_miou is None
              and row2.all_class_miou_status == "undefined_no_eligible_class"
              and row2.disease_only_miou_status == "undefined_no_eligible_class")
        check("14 undefined_per_image_scores counts ROWS not fields",
              result.integrity["undefined_per_image_scores"] == 1,
              f"={result.integrity['undefined_per_image_scores']}")

        # ---------- write the artifact ----------
        out1 = write_artifact(result, req, prov, timestamp_utc="2026-07-26T00:00:00Z")

        # ---------- 1/2. four files, exact names ----------
        names = sorted(p.name for p in out1.iterdir())
        check("1,2 exactly the four contract filenames",
              names == ["MANIFEST.sha256", "per_image.jsonl", "sufficient_stats.npz",
                        "summary.json"], f"{names}")

        # ---------- 3/4. strict JSON + no NaN/Infinity tokens ----------
        summary_txt = (out1 / "summary.json").read_text(encoding="utf-8")
        rows_txt = (out1 / "per_image.jsonl").read_text(encoding="utf-8")

        def _boom(x):
            raise AssertionError(f"non-finite constant {x}")

        summary = json.loads(summary_txt, parse_constant=_boom)
        parsed_rows = [json.loads(ln, parse_constant=_boom) for ln in rows_txt.splitlines()]
        check("3 strict JSON parse (summary + all JSONL rows)", len(parsed_rows) == len(MANIFEST))
        bad_tok = re.findall(r"\b(NaN|-?Infinity)\b", summary_txt + rows_txt)
        check("4 zero NaN/Infinity tokens in raw text", not bad_tok, f"{bad_tok[:5]}")

        # ---------- 5. schema fields + types ----------
        check("5 top-level schema_version", summary["schema_version"] == "plantseg-eval/1.0.0",
              summary["schema_version"])
        check("5 top-level metric_protocol", summary["metric_protocol"] == "plantseg-metrics/1.0.0",
              summary["metric_protocol"])
        run_keys = {"run_id", "artifact_status", "stage", "model_role", "precision",
                    "quant_backend", "checkpoint_path", "checkpoint_sha256", "random_init",
                    "repo_commit", "governed_paths_clean", "dirty_allowlisted",
                    "worktree_state_sha256", "metric_impl_sha256", "timestamp_utc", "env",
                    "config_sha256"}
        ds_keys = {"name", "doi", "split", "split_manifest_sha256", "expected_rows",
                   "actual_rows", "condition", "preprocess_protocol", "num_classes",
                   "background_index", "ignore_index", "class_map_sha256"}
        check("5 run block complete", run_keys <= set(summary["run"]),
              f"missing={run_keys - set(summary['run'])}")
        check("5 dataset block complete", ds_keys <= set(summary["dataset"]),
              f"missing={ds_keys - set(summary['dataset'])}")
        row_keys = {"image_id", "clean_image_id", "manifest_index", "condition",
                    "all_class_miou", "all_class_miou_status", "disease_only_miou",
                    "disease_only_miou_status", "n_eligible_all_class",
                    "n_eligible_disease_only", "gt_disease_classes"}
        check("5 per-image rows have exactly the frozen 11 keys",
              all(set(r) == row_keys for r in parsed_rows),
              f"{sorted(set(parsed_rows[0]) ^ row_keys)}")
        check("116-class artifact: num_classes field",
              summary["dataset"]["num_classes"] == 116)
        check("116-class artifact: every per_class array length 116",
              all(len(summary["per_class"][k]) == 116 for k in
                  ("class_ids", "gt_support", "pred_support", "intersection", "union",
                   "iou", "dice", "acc", "iou_status")))
        check("smoke disclosure: status + random_init + no checkpoint",
              summary["run"]["artifact_status"] == "smoke"
              and summary["run"]["random_init"] is True
              and summary["run"]["checkpoint_path"] is None
              and summary["run"]["checkpoint_sha256"] is None)
        check("fixture named as synthetic, not PlantSeg validation",
              "SYNTHETIC" in summary["dataset"]["name"]
              and summary["dataset"]["preprocess_protocol"] == "synthetic-contract/1.0.0",
              summary["dataset"]["name"])

        # ---------- 6/7/8. ordering + identity ----------
        jl_idx = [r["manifest_index"] for r in parsed_rows]
        check("6 JSONL ascending manifest_index", jl_idx == sorted(jl_idx))
        ids = [r["image_id"] for r in parsed_rows]
        check("7 IDs unique and match the expected manifest",
              len(set(ids)) == len(ids) and ids == [m.image_id for m in MANIFEST])
        check("7 clean_image_id carried directly (== image_id for clean)",
              all(r["image_id"] == r["clean_image_id"] for r in parsed_rows))
        check("8 case-folded ID uniqueness",
              len({i.casefold() for i in ids}) == len(ids))

        # ---------- 11/12/13. NPZ ----------
        with np.load(out1 / "sufficient_stats.npz", allow_pickle=False) as z:
            keys = sorted(z.files)
            check("13 NPZ arrays present",
                  keys == sorted(["image_index", "class_id", "tp", "gt", "pred",
                                  "dataset_tp", "dataset_gt", "dataset_pred", "manifest_ids"]),
                  f"{keys}")
            check("13 NPZ int64 dtypes",
                  all(z[k].dtype == np.int64 for k in
                      ("image_index", "class_id", "tp", "gt", "pred",
                       "dataset_tp", "dataset_gt", "dataset_pred")))
            check("13 dataset arrays shaped (116,)",
                  all(z[k].shape == (116,) for k in ("dataset_tp", "dataset_gt", "dataset_pred")))
            recon_ok = True
            for name in ("tp", "gt", "pred"):
                acc = np.zeros(C, dtype=np.int64)
                np.add.at(acc, z["class_id"], z[name])
                if not np.array_equal(acc, z[f"dataset_{name}"]):
                    recon_ok = False
            check("11,12 sparse triplets sum exactly to dataset totals", recon_ok)
            cm_tp = torch.diag(cm_ref).numpy()
            check("12 dataset_tp matches an independent confusion matrix",
                  np.array_equal(z["dataset_tp"], cm_tp))
            mids = [str(x) for x in z["manifest_ids"]]
            check("image_index -> manifest_ids row-position mapping holds",
                  all(mids[int(z['image_index'][k])] ==
                      ids[int(z['image_index'][k])] for k in range(z["image_index"].size))
                  and mids == ids)
            check("sparse rule: no all-zero (image,class) row stored",
                  bool(np.all((z["tp"] != 0) | (z["gt"] != 0) | (z["pred"] != 0))))
            npz1 = {k: z[k].copy() for k in z.files}

        # ---------- 15. manifest hash verification ----------
        verify_artifact(out1)
        man = (out1 / "MANIFEST.sha256").read_text(encoding="utf-8")
        check("15 manifest lines sha256sum-compatible, sorted, self excluded",
              [ln.split("  ", 1)[1] for ln in man.splitlines()] ==
              ["per_image.jsonl", "sufficient_stats.npz", "summary.json"]
              and all(re.fullmatch(r"[0-9a-f]{64}", ln.split("  ", 1)[0]) for ln in man.splitlines()))
        tampered = expect_raises(ArtifactWriteError, _tamper_and_verify, out1)
        check("15 manifest detects tampering", "mismatch" in str(tampered), str(tampered)[:120])

        # ---------- 16. overwrite refusal ----------
        e_ow = expect_raises(ArtifactRequestError, validate_artifact_request,
                             make_request(out1))
        check("16 overwrite refused for an existing directory",
              "refusing to overwrite" in str(e_ow), str(e_ow)[:120])

        # ---------- 18. injected failure leaves nothing behind ----------
        fail_dir = workdir / "run_fail"
        req_f = make_request(fail_dir)
        prov_f = validate_artifact_request(req_f)
        e_inj = expect_raises(ArtifactWriteError, write_artifact, result, req_f, prov_f,
                              timestamp_utc="2026-07-26T00:00:00Z", _inject_failure=True)
        check("18 injected failure raised", "injected" in str(e_inj))
        check("18 no target dir after injected failure", not fail_dir.exists())
        check("18 no temp sibling remains", not list(workdir.glob(".run_fail.tmp-*")))

        # ---------- 19. semantic determinism across two runs ----------
        model2 = DummySegModel(C)
        req2 = make_request(workdir / "run2")
        prov2 = validate_artifact_request(req2)
        result2 = run_eval(model2, make_batches([[2], [0, 5], [3, 1, 4]]))   # different order again
        out2 = write_artifact(result2, req2, prov2, timestamp_utc="2026-07-26T11:11:11Z")
        s2 = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))

        def strip_ephemeral(s):
            s = json.loads(json.dumps(s))
            for k in ("run_id", "timestamp_utc"):
                s["run"].pop(k, None)
            return s

        check("19 summary semantically identical (ephemeral fields removed)",
              canonical_json_bytes(strip_ephemeral(summary)) ==
              canonical_json_bytes(strip_ephemeral(s2)))
        check("19 per_image.jsonl byte-identical across runs",
              rows_txt == (out2 / "per_image.jsonl").read_text(encoding="utf-8"))
        with np.load(out2 / "sufficient_stats.npz", allow_pickle=False) as z2:
            same = sorted(z2.files) == sorted(npz1.keys()) and all(
                z2[k].dtype == npz1[k].dtype and z2[k].shape == npz1[k].shape
                and np.array_equal(z2[k], npz1[k]) for k in npz1)
        check("19 NPZ semantically identical (keys/dtypes/shapes/values)", same)
        check("19 config_sha256 stable across runs",
              summary["run"]["config_sha256"] == s2["run"]["config_sha256"])

        # ---------- 17. provenance hashes are REAL, not placeholders ----------
        hashes = {
            "worktree_state_sha256": summary["run"]["worktree_state_sha256"],
            "metric_impl_sha256": summary["run"]["metric_impl_sha256"],
            "config_sha256": summary["run"]["config_sha256"],
            "split_manifest_sha256": summary["dataset"]["split_manifest_sha256"],
            "class_map_sha256": summary["dataset"]["class_map_sha256"],
        }
        check("17 provenance hashes are real 64-hex, none zero/placeholder",
              all(re.fullmatch(r"[0-9a-f]{64}", h) and set(h) != {"0"} for h in hashes.values())
              and len(set(hashes.values())) == len(hashes),
              ", ".join(f"{k}={v[:8]}" for k, v in hashes.items()))
        from src.eval.artifacts import hash_metric_impl
        live_metric_sha, live_path = hash_metric_impl()
        check("17 metric_impl_sha256 equals a live hash of the imported metrics.py",
              hashes["metric_impl_sha256"] == live_metric_sha, Path(live_path).name)

        # ---------- 32. no PlantSeg filesystem access ----------
        plantseg_hits = [p for p in OPENED_PATHS
                         if "plantseg_data" in p.lower() or "annotations" in p.lower()
                         or p.lower().endswith((".jpg", ".jpeg"))]
        check("32 no PlantSeg image/mask/split file opened",
              not plantseg_hits and "src.data" not in sys.modules
              and "configs.data" not in sys.modules,
              f"{len(OPENED_PATHS)} paths opened, PlantSeg hits={plantseg_hits[:3]}")

        # ---------- 2b. the WRITER rejects a non-116 class count ----------
        e116 = expect_raises(ArtifactRequestError, validate_artifact_request,
                             prepare_artifact_request(
                                 out_dir=workdir / "bad116", artifact_status="smoke",
                                 run_id="x",
                                 run=RunMeta(stage="E1", model_role="student", precision="fp32",
                                             random_init=True),
                                 dataset=DatasetMeta(
                                     name="SYNTHETIC", doi="d", split="val",
                                     condition=Condition("clean"),
                                     preprocess_protocol="synthetic-contract/1.0.0",
                                     expected_rows=len(MANIFEST)),
                                 expected_manifest=MANIFEST,
                                 class_map=SYNTHETIC_CLASS_MAP[:6],
                                 num_classes=6, background_index=BG, ignore_index=IGNORE,
                                 repo_root=REPO))
        check("2b writer rejects a non-116 class count for plantseg-eval/1.0.0",
              "num_classes=116" in str(e116), str(e116)[:120])

        # ---------- integrity abort cases ----------
        cases = [
            ("invalid model output shape", _bad_shape),
            ("wrong output class count", _bad_class_count),
            ("duplicate canonical ID", _dup_id),
            ("case-only ID collision", _case_collision),
            ("duplicate manifest index", _dup_index),
            ("missing expected-manifest entry", _missing_row),
            ("unexpected manifest row", _unexpected_row),
            ("wrong ID/index mapping", _bad_id_index_pairing),
            ("wrong clean_image_id mapping", _bad_clean_id),
            ("out-of-range prediction label", _bad_label),
        ]
        for name, fn in cases:
            try:
                fn()
                check(f"abort: {name}", False, "NO exception raised")
            except (EvaluationIntegrityError, ValueError) as e:
                check(f"abort: {name}", True, f"{type(e).__name__}: {str(e)[:90]}")
            except Exception as e:                       # noqa: BLE001
                check(f"abort: {name}", False, f"wrong type {type(e).__name__}: {e}")

    finally:
        builtins.open = _REAL_OPEN
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail[:160]}")
    n_ok = sum(1 for _, ok, _ in CHECKS if ok)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {n_ok}/{len(CHECKS)} checks passed")
    ok_all = n_ok == len(CHECKS)
    print(f"RESULT: {'A2a OK -- CORE CONTRACT PROVEN' if ok_all else 'A2a BLOCKED'}")
    print("Fixture is SYNTHETIC; no PlantSeg image, mask, split file or checkpoint was accessed.")
    print("=" * 100)
    return 0 if ok_all else 1


# --------------------------------------------------------------------------------------------------
# integrity-abort scenarios
# --------------------------------------------------------------------------------------------------
def _tamper_and_verify(out_dir: Path):
    p = out_dir / "per_image.jsonl"
    original = p.read_bytes()
    try:
        p.write_bytes(original + b'{"tampered":true}\n')
        verify_artifact(out_dir)
    finally:
        p.write_bytes(original)


class _ShapeModel(DummySegModel):
    def forward(self, x):
        self.forward_calls += 1
        return torch.zeros(x.shape[0], C, 3)          # 3-D, not [B,C,H,W]


class _ClassCountModel(DummySegModel):
    def forward(self, x):
        self.forward_calls += 1
        return torch.zeros(x.shape[0], 7, 4, 4)       # wrong class count


def _bad_shape():
    run_eval(_ShapeModel(C), make_batches([[0]]))


def _bad_class_count():
    run_eval(_ClassCountModel(C), make_batches([[0]]))


def _dup_id():
    b = make_batches([[0]])[0]
    dup = EvalBatch(torch.stack([_image_for(0), _image_for(0)]),
                    torch.stack([TARGET_BY_INDEX[0], TARGET_BY_INDEX[0]]),
                    ["syn_0000_normal", "syn_0000_normal"],
                    ["syn_0000_normal", "syn_0000_normal"], [0, 0])
    run_eval(DummySegModel(C), [dup])


def _case_collision():
    entries = [ManifestEntry(0, "syn_a", "syn_a"), ManifestEntry(1, "SYN_A", "SYN_A")]
    evaluate_model(DummySegModel(C), [], expected_manifest=entries,
                   condition=Condition("clean"), num_classes=C,
                   background_index=BG, ignore_index=IGNORE)


def _dup_index():
    entries = [ManifestEntry(0, "a", "a"), ManifestEntry(0, "b", "b")]
    evaluate_model(DummySegModel(C), [], expected_manifest=entries,
                   condition=Condition("clean"), num_classes=C,
                   background_index=BG, ignore_index=IGNORE)


def _missing_row():
    run_eval(DummySegModel(C), make_batches([[0, 1]]))     # manifest expects 6 rows


def _unexpected_row():
    b = EvalBatch(torch.stack([_image_for(0)]), torch.stack([TARGET_BY_INDEX[0]]),
                  ["ghost"], ["ghost"], [99])
    run_eval(DummySegModel(C), [b])


def _bad_id_index_pairing():
    """A real expected ID submitted under a DIFFERENT (also real) manifest index."""
    b = EvalBatch(torch.stack([_image_for(0)]), torch.stack([TARGET_BY_INDEX[0]]),
                  ["syn_0000_normal"], ["syn_0000_normal"], [1])   # id 0 paired with index 1
    run_eval(DummySegModel(C), [b])


def _bad_clean_id():
    b = EvalBatch(torch.stack([_image_for(0)]), torch.stack([TARGET_BY_INDEX[0]]),
                  ["syn_0000_normal"], ["WRONG_CLEAN"], [0])
    run_eval(DummySegModel(C), [b])


def _bad_label():
    bad_target = torch.full((1, 4, 4), 900, dtype=torch.long)
    b = EvalBatch(torch.stack([_image_for(0)]), bad_target,
                  ["syn_0000_normal"], ["syn_0000_normal"], [0])
    run_eval(DummySegModel(C), [b])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(2)
