"""Stage- and condition-neutral evaluation core (A2a).

Pure compute: no filesystem, no Git, no hashing, no schema, no model construction. Given an
already-constructed model and an iterable of batches that carry identity directly, this module
produces an `EvalResult` holding everything `src/eval/artifacts.py` needs to serialise the frozen
artifact set (docs/EVALUATION_CONTRACT.md sections 5.3-5.5).

Design constraints (docs/EVALUATION_CONTRACT.md section 6 + A2a instructions):
  * Identity travels WITH the sample. `EvalBatch` carries `image_ids`, `clean_image_ids` and
    `manifest_indices`; nothing is recovered from `dataset.pairs`, inferred from batch position,
    or parsed back out of a filename after batching.
  * Every observed sample is checked against an EXPECTED MANIFEST (index + id + clean id), so a
    missing row, an unexpected row, a wrong index-to-id pairing, or a wrong clean-image mapping
    is a hard failure rather than a silent artifact.
  * Metric ARITHMETIC comes only from src.eval.metrics. Stage / model_role / precision /
    condition are metadata and never influence a number.
  * Classwise integer quantities (tp / gt / pred / union) and the counts derived from them are
    serialisation of sufficient statistics -- NOT a second macro reducer. Every aggregate
    written to summary.json comes from the canonical production functions, and the classwise
    reconstruction is cross-checked against them.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .metrics import (confusion_matrix, dice_from_confusion, macc_from_confusion,
                      miou_from_confusion, per_image_miou)

# Production reduces in float32 (contract case C4), so classwise reconstruction is compared to
# the production macros at float32 precision, never tighter.
F32_TOL = 1e-6

STATUS_OK = "ok"
STATUS_UNDEFINED = "undefined_no_eligible_class"
STATUS_EXCLUDED = "excluded_data_integrity"
STATUS_ERROR = "evaluation_error"
PER_IMAGE_STATUSES = (STATUS_OK, STATUS_UNDEFINED, STATUS_EXCLUDED, STATUS_ERROR)

CLASS_STATUS_OK = "ok"
CLASS_STATUS_UNDEFINED = "undefined_absent_from_gt_and_pred"

_FORBIDDEN_ID_CHARS = ("\t", "\n", "\r", "\x00")


class EvaluationIntegrityError(RuntimeError):
    """Raised when evaluation cannot produce a trustworthy artifact.

    Always fatal: an integrity failure must abort finalisation, never degrade into a normal row
    carrying an error-looking status.
    """


# --------------------------------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """Condition identity. `clean` today; corruption name/severity are representable so future
    adapters reuse this core unchanged. A2a does NOT generate corrupted images."""
    type: str = "clean"
    name: str | None = None
    severity: int | None = None

    def __post_init__(self) -> None:
        if self.type not in ("clean", "corruption"):
            raise ValueError(f"condition.type must be 'clean' or 'corruption', got {self.type!r}")
        if self.type == "clean" and (self.name is not None or self.severity is not None):
            raise ValueError("clean condition must have name=None and severity=None")
        if self.type == "corruption" and (not self.name or self.severity is None):
            raise ValueError("corruption condition requires both name and severity")

    def as_dict(self) -> dict:
        return {"type": self.type, "name": self.name, "severity": self.severity}


@dataclass(frozen=True)
class ManifestEntry:
    """One expected row: the canonical (index, id, clean id) triple."""
    manifest_index: int
    image_id: str
    clean_image_id: str

    def __post_init__(self) -> None:
        for label, value in (("image_id", self.image_id), ("clean_image_id", self.clean_image_id)):
            if not value:
                raise ValueError(f"{label} must be non-empty")
            for ch in _FORBIDDEN_ID_CHARS:
                if ch in value:
                    raise ValueError(f"{label}={value!r} contains a forbidden character {ch!r}")
        if self.manifest_index < 0:
            raise ValueError(f"manifest_index must be >= 0, got {self.manifest_index}")


@dataclass(frozen=True)
class RunMeta:
    """Run identity metadata. Pure data -- no filesystem or Git dependency, which is why it lives
    beside the core rather than in the artifact writer."""
    stage: str
    model_role: str
    precision: str
    quant_backend: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    random_init: bool = False
    device: str = "cpu"


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset identity metadata. Pure data."""
    name: str
    doi: str
    split: str
    condition: "Condition"
    preprocess_protocol: str
    expected_rows: int


@dataclass(frozen=True)
class EvalBatch:
    """A batch that carries identity directly. `clean_image_ids` is mandatory, not derived."""
    images: torch.Tensor
    targets: torch.Tensor
    image_ids: Sequence[str]
    clean_image_ids: Sequence[str]
    manifest_indices: Sequence[int]

    def __post_init__(self) -> None:
        n = len(self.image_ids)
        if not (len(self.clean_image_ids) == len(self.manifest_indices) == n):
            raise ValueError("image_ids, clean_image_ids and manifest_indices must be the same length")
        if self.images.shape[0] != n or self.targets.shape[0] != n:
            raise ValueError(
                f"batch size mismatch: images {self.images.shape[0]}, targets "
                f"{self.targets.shape[0]}, ids {n}")


@dataclass(frozen=True)
class PerImageRow:
    """Exactly the fields frozen by contract section 5.4 -- no per-image classwise arrays."""
    image_id: str
    clean_image_id: str
    manifest_index: int
    condition: Condition
    all_class_miou: float | None
    all_class_miou_status: str
    disease_only_miou: float | None
    disease_only_miou_status: str
    n_eligible_all_class: int
    n_eligible_disease_only: int
    gt_disease_classes: list[int]

    def as_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "clean_image_id": self.clean_image_id,
            "manifest_index": self.manifest_index,
            "condition": self.condition.as_dict(),
            "all_class_miou": self.all_class_miou,
            "all_class_miou_status": self.all_class_miou_status,
            "disease_only_miou": self.disease_only_miou,
            "disease_only_miou_status": self.disease_only_miou_status,
            "n_eligible_all_class": self.n_eligible_all_class,
            "n_eligible_disease_only": self.n_eligible_disease_only,
            "gt_disease_classes": self.gt_disease_classes,
        }


@dataclass
class EvalResult:
    """Everything the artifact writer needs; no filesystem or provenance concepts."""
    rows: list[PerImageRow]
    manifest_ids: list[str]                 # canonical order, index == NPZ image_index
    sparse_image_index: np.ndarray
    sparse_class_id: np.ndarray
    sparse_tp: np.ndarray
    sparse_gt: np.ndarray
    sparse_pred: np.ndarray
    dataset_tp: np.ndarray
    dataset_gt: np.ndarray
    dataset_pred: np.ndarray
    dataset_level: dict
    per_class: dict
    integrity: dict
    num_classes: int
    background_index: int
    ignore_index: int
    condition: Condition
    forward_batches: int = 0
    _cm: torch.Tensor | None = field(default=None, repr=False)


# --------------------------------------------------------------------------------------------------
# helpers (classwise serialisation only -- never a macro reducer)
# --------------------------------------------------------------------------------------------------
def _counts_from_cm(cm: torch.Tensor):
    """Exact int64 per-class counts. Mirrors the contract's notation, no eligibility policy."""
    tp = torch.diag(cm).to(torch.int64)
    gt = cm.sum(1).to(torch.int64)
    pr = cm.sum(0).to(torch.int64)
    return tp.numpy(), gt.numpy(), pr.numpy()


def _ratio_or_none(num: int, den: int) -> float | None:
    return float(num) / float(den) if den > 0 else None


def _finite_or_none(x: float) -> float | None:
    """NaN from a production reducer means 'no eligible class' -> the contract's null."""
    return None if (x is None or math.isnan(x)) else float(x)


# --------------------------------------------------------------------------------------------------
# the core
# --------------------------------------------------------------------------------------------------
def evaluate_model(
    model,
    batches: Iterable[EvalBatch],
    *,
    expected_manifest: Sequence[ManifestEntry],
    condition: Condition,
    num_classes: int,
    background_index: int,
    ignore_index: int,
    forward: Callable | None = None,
) -> EvalResult:
    """Evaluate an already-constructed model over identity-carrying batches.

    `model` may be anything whose forward returns `[B, num_classes, H, W]` logits (FP32 student,
    teacher, PTQ, QAT -- the core does not care). `forward` optionally overrides how the model is
    invoked. Raises EvaluationIntegrityError on any identity or output-shape violation.
    """
    if num_classes <= background_index or background_index < 0:
        raise EvaluationIntegrityError(
            f"background_index {background_index} outside 0..{num_classes - 1}")

    # ---- expected manifest: validate the contract of the manifest itself ----
    if not expected_manifest:
        raise EvaluationIntegrityError("expected_manifest is empty")
    by_index: dict[int, ManifestEntry] = {}
    seen_ids: dict[str, int] = {}
    seen_folded: dict[str, str] = {}
    for entry in expected_manifest:
        if entry.manifest_index in by_index:
            raise EvaluationIntegrityError(
                f"duplicate manifest_index {entry.manifest_index} in expected_manifest")
        if entry.image_id in seen_ids:
            raise EvaluationIntegrityError(
                f"duplicate image_id {entry.image_id!r} in expected_manifest")
        folded = entry.image_id.casefold()
        if folded in seen_folded:
            raise EvaluationIntegrityError(
                f"case-folded id collision in expected_manifest: {entry.image_id!r} vs "
                f"{seen_folded[folded]!r}")
        by_index[entry.manifest_index] = entry
        seen_ids[entry.image_id] = entry.manifest_index
        seen_folded[folded] = entry.image_id

    ordered = sorted(expected_manifest, key=lambda e: e.manifest_index)
    manifest_ids = [e.image_id for e in ordered]
    # NPZ image_index is the ZERO-BASED ROW POSITION into manifest_ids, not the external
    # manifest_index (which may be sparse). See the smoke report for the mapping proof.
    row_position = {e.manifest_index: i for i, e in enumerate(ordered)}

    disease_idx = torch.tensor([c for c in range(num_classes) if c != background_index],
                               dtype=torch.long)

    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    rows: list[PerImageRow] = []
    s_img, s_cls, s_tp, s_gt, s_pr = [], [], [], [], []
    observed: dict[str, int] = {}
    observed_indices: set[int] = set()
    forward_batches = 0

    invoke = forward if forward is not None else (lambda m, x: m(x))

    with torch.no_grad():
        for batch in batches:
            logits = invoke(model, batch.images)
            forward_batches += 1
            if logits.dim() != 4:
                raise EvaluationIntegrityError(
                    f"model output must be [B, C, H, W]; got shape {tuple(logits.shape)}")
            if logits.shape[1] != num_classes:
                raise EvaluationIntegrityError(
                    f"model output class count {logits.shape[1]} != num_classes {num_classes}")
            if logits.shape[0] != batch.targets.shape[0]:
                raise EvaluationIntegrityError(
                    f"model output batch {logits.shape[0]} != target batch {batch.targets.shape[0]}")
            preds = logits.argmax(1).cpu()
            targets = batch.targets.cpu()
            if preds.shape[-2:] != targets.shape[-2:]:
                raise EvaluationIntegrityError(
                    f"prediction spatial shape {tuple(preds.shape[-2:])} != target "
                    f"{tuple(targets.shape[-2:])}")

            all_scores = per_image_miou(preds, targets, num_classes,
                                        class_indices=None, ignore_index=ignore_index)
            dis_scores = per_image_miou(preds, targets, num_classes,
                                        class_indices=disease_idx, ignore_index=ignore_index)

            for i in range(len(batch.image_ids)):
                image_id = batch.image_ids[i]
                clean_id = batch.clean_image_ids[i]
                m_index = int(batch.manifest_indices[i])

                # ---- identity validation against the expected manifest ----
                if m_index not in by_index:
                    raise EvaluationIntegrityError(
                        f"unexpected manifest_index {m_index} (image_id={image_id!r}) "
                        "is absent from expected_manifest")
                entry = by_index[m_index]
                if entry.image_id != image_id:
                    raise EvaluationIntegrityError(
                        f"manifest_index {m_index} is paired with image_id {image_id!r} but the "
                        f"expected manifest has {entry.image_id!r}")
                if entry.clean_image_id != clean_id:
                    raise EvaluationIntegrityError(
                        f"image_id {image_id!r} has clean_image_id {clean_id!r} but the expected "
                        f"manifest has {entry.clean_image_id!r}")
                if image_id in observed:
                    raise EvaluationIntegrityError(
                        f"duplicate canonical image_id {image_id!r} observed during evaluation")
                folded = image_id.casefold()
                for prior in observed:
                    if prior.casefold() == folded:
                        raise EvaluationIntegrityError(
                            f"case-folded image_id collision: {image_id!r} vs {prior!r}")
                if m_index in observed_indices:
                    raise EvaluationIntegrityError(
                        f"duplicate manifest_index {m_index} observed during evaluation")
                observed[image_id] = m_index
                observed_indices.add(m_index)

                # ---- per-image counts + metrics ----
                cm_i = confusion_matrix(preds[i], targets[i], num_classes, ignore_index)
                cm_total += cm_i
                tp_i, gt_i, pr_i = _counts_from_cm(cm_i)
                pos = row_position[m_index]
                nz = np.nonzero((tp_i != 0) | (gt_i != 0) | (pr_i != 0))[0]
                for c in nz:
                    s_img.append(pos)
                    s_cls.append(int(c))
                    s_tp.append(int(tp_i[c]))
                    s_gt.append(int(gt_i[c]))
                    s_pr.append(int(pr_i[c]))

                gt_present = np.nonzero(gt_i > 0)[0]
                gt_disease = [int(c) for c in gt_present if c != background_index]
                all_v = _finite_or_none(all_scores[i])
                dis_v = _finite_or_none(dis_scores[i])
                rows.append(PerImageRow(
                    image_id=image_id,
                    clean_image_id=clean_id,
                    manifest_index=m_index,
                    condition=condition,
                    all_class_miou=all_v,
                    all_class_miou_status=STATUS_OK if all_v is not None else STATUS_UNDEFINED,
                    disease_only_miou=dis_v,
                    disease_only_miou_status=STATUS_OK if dis_v is not None else STATUS_UNDEFINED,
                    n_eligible_all_class=int(gt_present.size),
                    n_eligible_disease_only=len(gt_disease),
                    gt_disease_classes=gt_disease,
                ))

    # ---- completeness ----
    missing = [e.image_id for e in ordered if e.image_id not in observed]
    if missing:
        raise EvaluationIntegrityError(
            f"{len(missing)} expected image(s) never observed, e.g. {missing[:5]}")

    rows.sort(key=lambda r: r.manifest_index)   # canonical output order, whatever the input order

    # ---- dataset-level aggregates: ALL from the production reducers ----
    d_tp, d_gt, d_pr = _counts_from_cm(cm_total)
    d_union = d_gt + d_pr - d_tp
    disease_mask = np.ones(num_classes, dtype=bool)
    disease_mask[background_index] = False

    all_miou = _finite_or_none(miou_from_confusion(cm_total))
    dis_miou = _finite_or_none(miou_from_confusion(cm_total, disease_idx))
    all_macc = _finite_or_none(macc_from_confusion(cm_total))
    all_dice = _finite_or_none(dice_from_confusion(cm_total))

    elig_union = d_union > 0
    elig_gt = d_gt > 0
    elig_dice = (d_gt + d_pr) > 0
    dataset_level = {
        "all_class_miou": all_miou,
        "all_class_miou_n_eligible": int(elig_union.sum()),
        "all_class_macro_dice": all_dice,
        "all_class_dice_n_eligible": int(elig_dice.sum()),
        "all_class_macc": all_macc,
        "all_class_macc_n_eligible": int(elig_gt.sum()),
        "disease_only_miou": dis_miou,
        "disease_only_miou_n_eligible": int((elig_union & disease_mask).sum()),
        # aAcc: micro pixel accuracy from integer totals (contract section 3.1, diagnostic only).
        "aacc_diagnostic": _ratio_or_none(int(d_tp.sum()), int(d_gt.sum())),
    }

    # ---- classwise serialisation + cross-check against the production macros ----
    with np.errstate(invalid="ignore", divide="ignore"):
        iou_c = np.where(elig_union, d_tp / np.maximum(d_union, 1), np.nan)
        dice_c = np.where(elig_dice, 2.0 * d_tp / np.maximum(d_gt + d_pr, 1), np.nan)
        acc_c = np.where(elig_gt, d_tp / np.maximum(d_gt, 1), np.nan)

    def _recon(values, mask):
        return float(np.mean(values[mask])) if bool(mask.any()) else None

    for label, recon, produced in (
        ("all_class_miou", _recon(iou_c, elig_union), all_miou),
        ("all_class_macro_dice", _recon(dice_c, elig_dice), all_dice),
        ("all_class_macc", _recon(acc_c, elig_gt), all_macc),
        ("disease_only_miou", _recon(iou_c, elig_union & disease_mask), dis_miou),
    ):
        if (recon is None) != (produced is None):
            raise EvaluationIntegrityError(
                f"{label}: classwise reconstruction definedness {recon} disagrees with the "
                f"production reducer {produced}")
        if recon is not None and abs(recon - produced) > F32_TOL:
            raise EvaluationIntegrityError(
                f"{label}: classwise reconstruction {recon!r} != production {produced!r} "
                f"(tolerance {F32_TOL})")

    per_class = {
        "class_ids": list(range(num_classes)),
        "gt_support": [int(v) for v in d_gt],
        "pred_support": [int(v) for v in d_pr],
        "intersection": [int(v) for v in d_tp],
        "union": [int(v) for v in d_union],
        "iou": [None if not elig_union[c] else float(iou_c[c]) for c in range(num_classes)],
        "dice": [None if not elig_dice[c] else float(dice_c[c]) for c in range(num_classes)],
        "acc": [None if not elig_gt[c] else float(acc_c[c]) for c in range(num_classes)],
        "iou_status": [CLASS_STATUS_OK if elig_union[c] else CLASS_STATUS_UNDEFINED
                       for c in range(num_classes)],
    }

    undefined_rows = sum(
        1 for r in rows if r.all_class_miou is None or r.disease_only_miou is None)
    integrity = {
        "row_count_ok": len(rows) == len(ordered),
        "ids_unique": True,
        "ids_match_manifest": True,
        "order_canonical": all(rows[i].manifest_index < rows[i + 1].manifest_index
                               for i in range(len(rows) - 1)),
        "duplicate_ids": [],
        "missing_ids": [],
        # confusion_matrix raises on any out-of-range label, so reaching here means zero.
        "invalid_pred_labels": 0,
        "undefined_per_image_scores": undefined_rows,
        "overwrite_policy": "refuse_existing",
    }

    # Sparse rows are collected in BATCH-ITERATION order, which depends on how the caller
    # grouped and ordered its batches. Sort canonically by (image_index, class_id) so the NPZ
    # payload is a deterministic function of the data alone, exactly like per_image.jsonl.
    empty = np.zeros(0, dtype=np.int64)
    if s_img:
        a_img = np.array(s_img, dtype=np.int64)
        a_cls = np.array(s_cls, dtype=np.int64)
        order_idx = np.lexsort((a_cls, a_img))          # primary image_index, secondary class_id
        a_img, a_cls = a_img[order_idx], a_cls[order_idx]
        a_tp = np.array(s_tp, dtype=np.int64)[order_idx]
        a_gt = np.array(s_gt, dtype=np.int64)[order_idx]
        a_pr = np.array(s_pr, dtype=np.int64)[order_idx]
    else:
        a_img = a_cls = a_tp = a_gt = a_pr = empty

    return EvalResult(
        rows=rows,
        manifest_ids=manifest_ids,
        sparse_image_index=a_img,
        sparse_class_id=a_cls,
        sparse_tp=a_tp,
        sparse_gt=a_gt,
        sparse_pred=a_pr,
        dataset_tp=d_tp.astype(np.int64),
        dataset_gt=d_gt.astype(np.int64),
        dataset_pred=d_pr.astype(np.int64),
        dataset_level=dataset_level,
        per_class=per_class,
        integrity=integrity,
        num_classes=num_classes,
        background_index=background_index,
        ignore_index=ignore_index,
        condition=condition,
        forward_batches=forward_batches,
        _cm=cm_total,
    )
