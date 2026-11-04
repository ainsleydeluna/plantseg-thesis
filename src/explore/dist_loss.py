"""A2: DIST (Huang et al., NeurIPS 2022) in place of lambda * KL only (audit EXPLORE-AUDIT-1 §2 A2; lane L-AM24).

The official segmentation code's settings (audit source S1c, hunto/DIST_KD segmentation/losses/dist_kd.py, as the
audit read it on 3 Nov 2026; this lane had no web access and transcribed the formulas below from that description
and the paper's Eqs. 6-10, so the auditor should diff them against the official file): the logits are flattened to
rows (B * H * W, C); each row is softmaxed at tau = 1; the inter-class term is 1 - the mean over rows of the Pearson
correlation between the student's and the teacher's row; the intra-class term is the same with rows and columns
swapped (a Pearson correlation per class over every row of the batch); both are scaled by tau^2 and weighted
beta = gamma = 1; the loss is weighted 1 against CE. Pearson correlation is the cosine similarity of the
mean-centred vectors with eps = 1e-8 in the denominator.

Here the rows are the OS8 head grid of E2's logit-KD term (no upsampling) and only the valid rows are kept: the
validity mask downsampled to that grid by train_distill's own rule (downsample_validity, all-valid cells), so the
grid, the mask, the ramp and everything else stay E2's. Both terms are invariant to the row order, so this module's
(B, H, W) row order and the official (B, W, H) one give the same value (the smoke shows it).
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

BETA = 1.0
GAMMA = 1.0
TAU = 1.0
EPS = 1e-8
WEIGHT = 1.0                                  # against CE: train_distill's lambda_logit is passed as 1


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def rows(z: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B * H * W, C]."""
    c = z.shape[1]
    return z.permute(0, 2, 3, 1).reshape(-1, c)


def dist_loss(z_s: torch.Tensor, z_t: torch.Tensor, valid: torch.Tensor | None = None, *, beta: float = BETA,
              gamma: float = GAMMA, tau: float = TAU) -> tuple[torch.Tensor, dict]:
    """DIST on [B, C, h, w] logit maps, over the rows `valid` [B, h, w] keeps (all rows when None)."""
    if z_s.shape != z_t.shape:
        raise ValueError(f"DIST shape mismatch: student {tuple(z_s.shape)} != teacher {tuple(z_t.shape)}")
    s, t = rows(z_s), rows(z_t)
    if valid is not None:
        if tuple(valid.shape) != (z_s.shape[0], *z_s.shape[-2:]):
            raise ValueError(f"validity mask {tuple(valid.shape)} does not match the grid {tuple(z_s.shape)}")
        m = valid.reshape(-1)
        s, t = s[m], t[m]
    if s.shape[0] < 2:                         # no correlation over fewer than two rows: no term, exactly 0
        zero = z_s.new_zeros(())
        return zero * z_s.sum(), {"inter": 0.0, "intra": 0.0, "rows": int(s.shape[0])}
    y_s = (s / tau).softmax(dim=1)
    y_t = (t / tau).softmax(dim=1)
    inter = tau ** 2 * inter_class_relation(y_s, y_t)
    intra = tau ** 2 * intra_class_relation(y_s, y_t)
    return beta * inter + gamma * intra, {"inter": float(inter.detach()), "intra": float(intra.detach()),
                                          "rows": int(s.shape[0])}


def dist_distillation_losses(*, stage: dict, logits, head_logits, c5, mask, teacher_out, projection,
                             lambda_logit: float | None, ramp=1.0, alpha=None, beta=None, targets=None):
    """train_distill.distillation_losses' signature and contract for stage e2, with DIST as the logit-KD term:
    total = ramp * lambda_logit * DIST(head_logits, teacher logits, the OS8 validity mask); parts["logit_kd"] holds the
    unramped DIST value (the trainer's telemetry column of that name). Any other stage, a projection or arm targets is
    refused: A2 changes E2 only."""
    from src.training import train_distill as TD
    from src.training.losses import downsample_validity
    if stage.get("key") != "e2" or projection is not None or targets is not None:
        raise RuntimeError(f"A2's DIST term replaces E2's KL only (stage {stage.get('key')!r}, projection "
                           f"{projection is not None}, targets {targets is not None})")
    if lambda_logit is None:
        raise ValueError("stage e2 instantiates the logit term but lambda_logit is None")
    t_logits = teacher_out.logits
    if t_logits.shape[1] != logits.shape[1]:
        raise ValueError(f"teacher logits have {t_logits.shape[1]} classes, student has {logits.shape[1]}")
    TD.require_same_grid("the teacher logits", t_logits, "the student head logits", head_logits)
    valid = downsample_validity(mask, head_logits.shape[-2:], ignore_index=TD.IGNORE_INDEX)
    loss, _ = dist_loss(head_logits, t_logits, valid)
    factor = ramp["logit_kd"] if isinstance(ramp, Mapping) else ramp
    total = logits.new_zeros(()) + factor * lambda_logit * loss
    return total, {"logit_kd": float(loss.detach())}
