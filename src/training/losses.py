"""Loss scaffolds for E1/E2/E3 (Blocker B11). SMOKE-ONLY use here — no training, no backward.

Per docs/IMPLEMENTATION_CONTRACT.md B5 (supervised) + B3 (distillation):
  L_sup = L_CE + L_Dice (equal weight). CE: class-weighted (sqrt inverse-frequency, median-normalized,
  over non-255 pixels), ignore_index=255. Dice: soft, on softmax probs, per-class macro over classes
  PRESENT in the batch (absent excluded), smoothing 1e-5, validity-masked. Logit KD (E2/E3): KL on
  temperature-softened outputs (T_Logit=4), averaged over valid pixels. CWD (E3): PLACEHOLDER only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def compute_class_weights(masks: torch.Tensor, num_classes: int, ignore_index: int = IGNORE_INDEX):
    """sqrt inverse-frequency class weights over non-ignore pixels, normalized by the median.

    SCAFFOLD / SMOKE-ONLY: real thesis weights are computed over the FULL train set and are
    NEED_TO_CONFIRM. Classes absent from `masks` (common in a tiny batch) get weight 1.0 (= median)
    so the vector is finite. Returns (weights[num_classes], stats over present classes).
    """
    flat = masks.reshape(-1)
    valid = flat[flat != ignore_index]
    counts = torch.bincount(valid, minlength=num_classes).float()
    total = counts.sum().clamp_min(1.0)
    present = counts > 0

    weights = torch.ones(num_classes, dtype=torch.float32)
    if present.any():
        w_present = torch.sqrt(total / (num_classes * counts[present]))
        med = w_present.median()
        weights[present] = w_present / med  # median-normalized
    stats = {
        "n_present": int(present.sum().item()),
        "min": float(weights[present].min().item()) if present.any() else float("nan"),
        "median": float(weights[present].median().item()) if present.any() else float("nan"),
        "max": float(weights[present].max().item()) if present.any() else float("nan"),
    }
    return weights, stats


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index)


class SoftDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, smooth: float = 1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)                       # [B,C,H,W]
        valid = (target != self.ignore_index).unsqueeze(1).float()  # [B,1,H,W]
        tgt = target.clone()
        tgt[target == self.ignore_index] = 0                   # placeholder; masked out below
        onehot = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        probs = probs * valid
        onehot = onehot * valid
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)                     # [C]
        denom = probs.sum(dims) + onehot.sum(dims)             # [C]
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        present = onehot.sum(dims) > 0                          # classes present in this batch
        dice_sel = dice[present] if present.any() else dice
        return 1.0 - dice_sel.mean()


class CombinedCEDiceLoss(nn.Module):
    """L_sup = L_CE + L_Dice (equal weight) — E1 supervised objective."""

    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = IGNORE_INDEX,
                 smooth: float = 1e-5):
        super().__init__()
        self.ce = WeightedCrossEntropyLoss(weight, ignore_index)
        self.dice = SoftDiceLoss(ignore_index, smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.dice(logits, target)


def logit_kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                target: torch.Tensor, T: float = 4.0, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Hinton-style KL on temperature-softened outputs, averaged over valid (non-ignore) pixels (E2/E3).

    lambda_logit (the weight on this term relative to CE) is NEED_TO_CONFIRM (validation sweep).
    """
    valid = (target != ignore_index).float()                  # [B,H,W]
    s_logp = F.log_softmax(student_logits / T, dim=1)
    t_prob = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=1)  # [B,H,W]
    kl = kl * (T * T)                                          # T^2 absorbed into the term
    denom = valid.sum().clamp_min(1.0)
    return (kl * valid).sum() / denom


def cwd_channelwise_kl(*args, **kwargs):
    """PLACEHOLDER / SCAFFOLD only — Channel-Wise KD (Shu 2021). NOT implemented for E3 yet (B11).

    Documented spec (docs/IMPLEMENTATION_CONTRACT.md B3): treat each feature channel as a spatial
    probability distribution; minimize KL between teacher/student channel distributions with T_CWD=4
    and T^2/C normalization (C=320, MSCAN-B stride-16); feature-map weight alpha=50, logit-map weight
    beta=3; training-only 1x1 projection student 160ch -> teacher 320ch; validity mask downsampled to
    stride-16. Full feature distillation + projection head + E3 wiring are deferred to the E3 blocker.
    """
    raise NotImplementedError(
        "CWD is a placeholder scaffold (B11); full feature distillation is deferred to E3.")
