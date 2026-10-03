"""Synthetic evaluation-artifact fixtures for the statistics smokes (lane L-STATS-OFFICIAL).

SYNTHETIC ONLY. Every artifact is generated from integer counts and written through the unmodified
evaluator writer (src/eval/artifacts.py: prepare_artifact_request -> validate_artifact_request ->
write_artifact) into a temporary clean fixture git repository OUTSIDE the working tree, as
scripts/smoke_am5_exclusion.py does. No PlantSeg image, mask, split file or checkpoint is read, and
nothing is written into this repository.

The fixtures follow the conventions the statistics driver relies on:
  * one ground truth per image, shared by every stage and condition (a mask never depends on a model):
    background plus one disease class, or background only for the k AM-5 images;
  * bare stems: image_id == clean_image_id in every artifact, so all 37 share one split manifest (O1);
  * every per-image value is the evaluator's float32 IoU arithmetic, so it is float32-representable and
    in [0, 1] (P20, P21);
  * the canvas protocol (core_preprocess/1.0.0) unless a test asks for another;
  * a dataset name that is not the evaluator's, so the statistics layer derives synthetic_input_data (P6).

Not a smoke itself: imported by scripts/smoke_stats_*.py.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.artifacts import (ARTIFACT_FILES, EVAL_RUNTIME_VERSION, FROZEN_NUM_CLASSES,  # noqa: E402
                                MANIFEST_NAME, prepare_artifact_request,
                                validate_artifact_request, write_artifact)
from src.eval.evaluate import (Condition, DatasetMeta, EvalResult, ManifestEntry,  # noqa: E402
                               PerImageRow, RunMeta)
from src.stats.driver import CLEAN, STAGE_PRECISION, official_inventory  # noqa: E402
from src.stats.eligibility import UNION_PRESENT, ClassTotals, dataset_miou  # noqa: E402

C = FROZEN_NUM_CLASSES
FIXTURE_DATASET_NAME = "SYNTHETIC stats fixture (NOT PlantSeg data)"
CANVAS = "core_preprocess/1.0.0"
TIMESTAMP = "2026-10-02T00:00:00Z"
CLASS_MAP = [{"class_id": 0, "name": "", "role": "background_or_non_disease"}] + \
            [{"class_id": c, "name": f"syn_{c:03d}", "role": "disease"} for c in range(1, C)]
#: Mean prediction quality per stage; corruption cells subtract SEVERITY_DROP per severity level.
STAGE_QUALITY = {"E1": 0.70, "E2": 0.72, "E3": 0.74, "E4": 0.68, "E5": 0.69, "E6": 0.71, "E7": 0.70}
SEVERITY_DROP = 0.04


def int_seed(*parts) -> int:
    """A deterministic integer seed from labels (never Python's salted hash)."""
    return int.from_bytes(hashlib.sha256("\x00".join(map(str, parts)).encode()).digest()[:8], "big")


def clean_git_fixture(root: Path) -> Path:
    """A temporary one-commit git repository with a clean tree, used as the evaluator's repo_root so
    an 'official' synthetic fixture can be written. Never this repository."""
    root.mkdir(parents=True, exist_ok=False)

    def git(*args):
        subprocess.run(["git", "-c", "user.name=stats-fixture", "-c", "user.email=stats@invalid",
                        "-c", "commit.gpgsign=false", *args], cwd=str(root), check=True,
                       capture_output=True)
    git("init", "-q")
    (root / "README.fixture").write_text("statistics smoke fixture repository\n", encoding="utf-8")
    git("add", "README.fixture")
    git("commit", "-q", "-m", "fixture baseline")
    return root


def stage_checkpoint(stage: str) -> str:
    """A synthetic 64-hex checkpoint identity per stage (cells share their stage's clean value)."""
    return hashlib.sha256(f"synthetic checkpoint {stage}".encode()).hexdigest()


# --------------------------------------------------------------------------------------------------
# ground truth and predictions
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Truth:
    ids: tuple[str, ...]
    disease: np.ndarray            # the image's disease class 1..115; 0 for an AM-5 image
    bg_gt: np.ndarray
    dis_gt: np.ndarray             # 0 for an AM-5 image

    @property
    def am5(self) -> np.ndarray:
        return self.dis_gt == 0


def make_truth(n: int, k: int, seed: int) -> Truth:
    if not 0 <= k < n:
        raise ValueError(f"k must be in [0, n), got k={k}, n={n}")
    rng = np.random.default_rng(seed)
    disease = rng.integers(1, C, size=n).astype(np.int64)
    bg_gt = rng.integers(3000, 9000, size=n).astype(np.int64)
    dis_gt = rng.integers(200, 3000, size=n).astype(np.int64)
    if k:
        am5 = rng.choice(n, size=k, replace=False)
        disease[am5] = 0
        dis_gt[am5] = 0
    return Truth(tuple(f"img_{i:05d}" for i in range(n)), disease, bg_gt, dis_gt)


@dataclass(frozen=True)
class Prediction:
    fn: np.ndarray                 # disease pixels predicted as background
    fp: np.ndarray                 # background pixels predicted as a disease class
    stray: np.ndarray              # the class an AM-5 image's false positives are predicted as


def predict(truth: Truth, *, quality: float, seed: int) -> Prediction:
    rng = np.random.default_rng(seed)
    n = len(truth.ids)
    miss = np.clip(rng.normal(1.0 - quality, 0.12, n), 0.0, 0.95)
    false = np.clip(rng.normal((1.0 - quality) * 0.15, 0.03, n), 0.0, 0.5)
    fn = np.minimum(np.floor(truth.dis_gt * miss).astype(np.int64), truth.dis_gt)
    fp = np.minimum(np.floor(truth.bg_gt * false).astype(np.int64), truth.bg_gt - 1)
    return Prediction(fn, fp, rng.integers(1, C, size=n).astype(np.int64))


def _iou32(tp: int, gt: int, pred: int) -> np.float32:
    """The evaluator's float32 IoU (src/eval/metrics.py `_miou_from_cm`)."""
    return np.float32(tp) / np.float32(gt + pred - tp)


@dataclass
class Scored:
    result: EvalResult
    disease_values: np.ndarray      # per-image disease-only value, NaN for an AM-5 image
    totals: tuple                   # dataset (tp, gt, pred), int64 [C]


def score(truth: Truth, pred: Prediction, condition: Condition) -> Scored:
    n = len(truth.ids)
    rows, s_img, s_cls, s_tp, s_gt, s_pr = [], [], [], [], [], []
    dvals = np.full(n, np.nan)
    for i, iid in enumerate(truth.ids):
        bg, dg = int(truth.bg_gt[i]), int(truth.dis_gt[i])
        fn, fp = int(pred.fn[i]), int(pred.fp[i])
        tp_b = bg - fp
        entries = [(0, tp_b, bg, tp_b + fn)]
        if dg > 0:
            d = int(truth.disease[i])
            tp_d = dg - fn
            entries.append((d, tp_d, dg, tp_d + fp))
            iou_b, iou_d = _iou32(tp_b, bg, tp_b + fn), _iou32(tp_d, dg, tp_d + fp)
            dis_v = float(iou_d)
            all_v = float(np.mean(np.array([iou_b, iou_d], dtype=np.float32)))
            gt_dis, n_all, n_dis = [d], 2, 1
            dvals[i] = dis_v
        else:
            if fp > 0:
                entries.append((int(pred.stray[i]), 0, 0, fp))
            dis_v, all_v = None, float(_iou32(tp_b, bg, tp_b + fn))
            gt_dis, n_all, n_dis = [], 1, 0
        rows.append(PerImageRow(
            image_id=iid, clean_image_id=iid, manifest_index=i, condition=condition,
            all_class_miou=all_v, all_class_miou_status="ok",
            disease_only_miou=dis_v,
            disease_only_miou_status="ok" if dis_v is not None else "undefined_no_eligible_class",
            n_eligible_all_class=n_all, n_eligible_disease_only=n_dis, gt_disease_classes=gt_dis))
        for cid, tp, gt, pr in entries:
            for acc, v in ((s_img, i), (s_cls, cid), (s_tp, tp), (s_gt, gt), (s_pr, pr)):
                acc.append(v)

    cls = np.array(s_cls, np.int64)
    d_tp, d_gt, d_pr = (np.zeros(C, np.int64) for _ in range(3))
    np.add.at(d_tp, cls, np.array(s_tp, np.int64))
    np.add.at(d_gt, cls, np.array(s_gt, np.int64))
    np.add.at(d_pr, cls, np.array(s_pr, np.int64))
    un = d_gt + d_pr - d_tp
    el_u, el_g = un > 0, d_gt > 0
    totals = ClassTotals(tp=d_tp, gt=d_gt, pred=d_pr)
    allc = dataset_miou(totals, range(C), UNION_PRESENT)
    dis = dataset_miou(totals, range(1, C), UNION_PRESENT)
    iou = [float(d_tp[c]) / float(un[c]) if el_u[c] else None for c in range(C)]
    dice = [2.0 * float(d_tp[c]) / float(d_gt[c] + d_pr[c]) if el_u[c] else None for c in range(C)]
    acc = [float(d_tp[c]) / float(d_gt[c]) if el_g[c] else None for c in range(C)]
    result = EvalResult(
        rows=rows, manifest_ids=list(truth.ids),
        sparse_image_index=np.array(s_img, np.int64), sparse_class_id=cls,
        sparse_tp=np.array(s_tp, np.int64), sparse_gt=np.array(s_gt, np.int64),
        sparse_pred=np.array(s_pr, np.int64), dataset_tp=d_tp, dataset_gt=d_gt, dataset_pred=d_pr,
        dataset_level={"all_class_miou": allc.value, "all_class_miou_n_eligible": allc.n_classes,
                       "all_class_macro_dice": float(np.mean([v for v in dice if v is not None])),
                       "all_class_dice_n_eligible": int(el_u.sum()),
                       "all_class_macc": float(np.mean([v for v in acc if v is not None])),
                       "all_class_macc_n_eligible": int(el_g.sum()),
                       "disease_only_miou": dis.value, "disease_only_miou_n_eligible": dis.n_classes,
                       "aacc_diagnostic": float(d_tp.sum()) / float(d_gt.sum())},
        per_class={"class_ids": list(range(C)), "gt_support": d_gt.tolist(),
                   "pred_support": d_pr.tolist(), "intersection": d_tp.tolist(), "union": un.tolist(),
                   "iou": iou, "dice": dice, "acc": acc,
                   "iou_status": ["ok" if el_u[c] else "undefined_absent_from_gt_and_pred"
                                  for c in range(C)]},
        integrity={"row_count_ok": True, "ids_unique": True, "ids_match_manifest": True,
                   "order_canonical": True, "duplicate_ids": [], "missing_ids": [],
                   "invalid_pred_labels": 0,
                   "undefined_per_image_scores": int(np.count_nonzero(truth.dis_gt == 0)),
                   "overwrite_policy": "refuse_existing"},
        num_classes=C, background_index=0, ignore_index=255, condition=condition)
    return Scored(result, dvals, (d_tp, d_gt, d_pr))


def write_fixture(out_dir: Path, scored: Scored, n: int, *, stage: str, condition: Condition,
                  split: str, status: str, repo_root: Path, run_id: str,
                  dataset_name: str = FIXTURE_DATASET_NAME, preprocess: str = CANVAS,
                  precision: str | None = None, checkpoint_sha256: str | None = None) -> Path:
    """Write one artifact through the unmodified evaluator writer."""
    smoke = status == "smoke"
    req = prepare_artifact_request(
        out_dir=out_dir, artifact_status=status, run_id=run_id,
        run=RunMeta(stage=stage, model_role="student",
                    precision=precision or STAGE_PRECISION[stage],
                    checkpoint_path=None if smoke else f"synthetic/{stage.lower()}.pt",
                    checkpoint_sha256=None if smoke else (checkpoint_sha256 or stage_checkpoint(stage)),
                    random_init=smoke),
        dataset=DatasetMeta(name=dataset_name, doi="10.5281/zenodo.17719108", split=split,
                            condition=condition, preprocess_protocol=preprocess, expected_rows=n),
        expected_manifest=[ManifestEntry(i, iid, iid) for i, iid in
                           enumerate(scored.result.manifest_ids)],
        class_map=CLASS_MAP, repo_root=repo_root)
    prov = validate_artifact_request(req)
    rt = {"eval_runtime_version": EVAL_RUNTIME_VERSION} if status == "official" else None
    return write_artifact(scored.result, req, prov, timestamp_utc=TIMESTAMP, eval_runtime=rt)


# --------------------------------------------------------------------------------------------------
# the 37-artifact inventory
# --------------------------------------------------------------------------------------------------
@dataclass
class FixtureSet:
    input_root: Path
    entries: list = field(default_factory=list)          # (stage, condition tuple, rel path)
    truth: Truth | None = None
    scored: dict = field(default_factory=dict)           # identity -> Scored

    def path(self, stage: str, condition: tuple = CLEAN) -> Path:
        for st, cond, rel in self.entries:
            if st == stage and tuple(cond) == tuple(condition):
                return self.input_root / rel
        raise KeyError((stage, condition))

    def rel(self, stage: str, condition: tuple = CLEAN) -> str:
        return str(self.path(stage, condition).relative_to(self.input_root).as_posix())


def rel_name(stage: str, condition: tuple, split: str) -> str:
    tail = "clean" if condition[0] == "clean" else f"{condition[1]}_s{condition[2]}"
    return f"runs/{split}/{stage.lower()}_{tail}"


def build_inventory(root: Path, *, n: int, k: int, split: str, status: str, seed: int = 20261002,
                    dataset_name: str = FIXTURE_DATASET_NAME, run_prefix: str = "syn",
                    quality: dict | None = None) -> FixtureSet:
    """The 37 artifacts of section 12.4.6 under a fresh clean fixture repository at `root`."""
    repo = clean_git_fixture(Path(root))
    truth = make_truth(n, k, int_seed(seed, "truth"))
    q = dict(STAGE_QUALITY, **(quality or {}))
    fs = FixtureSet(input_root=repo, truth=truth)
    for stage, cond in official_inventory():
        qq = q[stage] - (SEVERITY_DROP * cond[2] if cond[0] == "corruption" else 0.0)
        pred = predict(truth, quality=qq, seed=int_seed(seed, stage, *cond))
        condition = Condition(*cond)
        scored = score(truth, pred, condition)
        rel = rel_name(stage, cond, split)
        tag = "clean" if cond[0] == "clean" else f"{cond[1].replace('_', '-')}-s{cond[2]}"
        write_fixture(repo / rel, scored, n, stage=stage, condition=condition, split=split,
                      status=status, repo_root=repo, run_id=f"{run_prefix}-{stage.lower()}-{tag}",
                      dataset_name=dataset_name)
        fs.entries.append((stage, cond, rel))
        fs.scored[(stage, cond)] = scored
    return fs


def write_input_list(path: Path, fs: FixtureSet, entries=None) -> Path:
    """The official / smoke input-list file (schema plantseg-stats-inputs/1.0.0)."""
    from src.stats.driver import INPUT_LIST_SCHEMA
    doc = {"schema": INPUT_LIST_SCHEMA,
           "inputs": [{"stage": st, "condition": {"type": c[0], "name": c[1], "severity": c[2]},
                       "path": rel} for st, c, rel in (entries if entries is not None else fs.entries)]}
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return Path(path)


# --------------------------------------------------------------------------------------------------
# deliberate edits (tests)
# --------------------------------------------------------------------------------------------------
def rehash(d: Path) -> None:
    """Rewrite MANIFEST.sha256 after a deliberate edit (sha256sum format, sorted bare names)."""
    d = Path(d)
    (d / MANIFEST_NAME).write_text(
        "".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in ARTIFACT_FILES),
        encoding="utf-8", newline="\n")


def edit_summary(d: Path, fn) -> Path:
    """Apply fn to summary.json in place and rehash (the manifest stays self-consistent)."""
    p = Path(d) / "summary.json"
    s = fn(json.loads(p.read_text(encoding="utf-8")))
    p.write_text(json.dumps(s, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                 encoding="utf-8", newline="\n")
    rehash(d)
    return Path(d)
