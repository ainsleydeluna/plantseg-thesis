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


def downsample_validity(target: torch.Tensor, size: tuple[int, int],
                        ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Validity mask [B,H,W] -> bool mask [B,h,w] at a coarser (e.g. stride-16) grid.

    Contract B3 ignore handling: "validity mask downsampled to stride-16", where CWD must operate
    only over valid, non-padded locations. The rule here is therefore CONSERVATIVE / ALL-VALID:

        a coarse location is valid  IFF  EVERY contributing source location is valid.

    A cell that mixes padding with real pixels is excluded, so every location admitted into the
    channel-wise softmax/KL has complete valid high-resolution support. Implemented as an exact
    adaptive MIN-pool (`-max(-x)`) over the 0/1 mask, which is correct for arbitrary H x W -> h x w
    because adaptive pooling windows cover every input position; an any-valid max-pool would admit
    mixed cells and nearest-neighbour sampling would keep or drop a cell based on one arbitrary
    representative pixel. The ground-truth mask itself is never modified.
    """
    valid = (target != ignore_index).float().unsqueeze(1)          # [B,1,H,W]
    pooled = -F.adaptive_max_pool2d(-valid, size)                  # [B,1,h,w] = min over each window
    return pooled.squeeze(1) >= 1.0                                # [B,h,w] bool — all-valid only


def cwd_channelwise_kl(student_map: torch.Tensor, teacher_map: torch.Tensor,
                       valid_mask: torch.Tensor | None = None, T: float = 4.0,
                       channels_norm: int | None = None) -> torch.Tensor:
    """Channel-Wise Knowledge Distillation (Shu 2021) — E3 feature/logit term.

    Each CHANNEL is treated as a probability distribution over spatial locations: soften by T, take
    a spatial softmax per channel, then minimise KL(teacher || student), normalised by T^2 / C
    (contract B3; C = 320 for the MSCAN-B stride-16 feature map, and the map's own channel count for
    the logit map unless `channels_norm` overrides it).

    `valid_mask` [B,h,w] restricts BOTH the softmax and the KL to valid locations, per the contract's
    ignore handling: invalid positions are pushed to -inf before the softmax so they receive exactly
    zero probability and contribute nothing. A sample with no valid location contributes 0 instead of
    NaN. Teacher input is used as-is; callers pass a detached teacher tensor so no gradient flows back
    into the frozen teacher.

    Returns a scalar: the per-sample CWD summed over channels/locations, averaged over the batch.
    """
    if student_map.shape != teacher_map.shape:
        raise ValueError(f"CWD shape mismatch: student {tuple(student_map.shape)} != teacher "
                         f"{tuple(teacher_map.shape)} (project/resample before calling)")
    b, c, h, w = student_map.shape
    s = student_map.reshape(b, c, h * w) / T
    t = teacher_map.reshape(b, c, h * w) / T

    if valid_mask is not None:
        if tuple(valid_mask.shape) != (b, h, w):
            raise ValueError(f"CWD validity mask {tuple(valid_mask.shape)} does not match map grid "
                             f"{(b, h, w)}")
        m = valid_mask.reshape(b, 1, h * w)                        # [B,1,HW] bool
        neg = torch.finfo(s.dtype).min
        s = s.masked_fill(~m, neg)
        t = t.masked_fill(~m, neg)
        has_valid = m.any(dim=-1).squeeze(1)                       # [B]
    else:
        m = None
        has_valid = torch.ones(b, dtype=torch.bool, device=student_map.device)

    s_logp = F.log_softmax(s, dim=-1)
    t_logp = F.log_softmax(t, dim=-1)
    kl = t_logp.exp() * (t_logp - s_logp)                          # [B,C,HW]
    if m is not None:
        kl = kl * m                                                # drop the all -inf columns
    kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)     # samples with no valid location

    cnorm = c if channels_norm is None else channels_norm
    per_sample = kl.sum(dim=(1, 2)) * (T * T) / cnorm              # [B]
    per_sample = per_sample * has_valid.to(per_sample.dtype)
    return per_sample.mean()
