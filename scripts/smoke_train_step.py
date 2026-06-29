#!/usr/bin/env python3
"""B17 — E1 train-step smoke (Week 2, Task 1). WIRING / GRADIENT-FLOW ONLY — NOT real E1 training.

Proves the E1 components compose into one working CPU train step:
  build_student() (weights=None, no download) + build_dataloader("train", 2) +
  CombinedCEDiceLoss(weight=None, ignore_index=255) + SGD (configs/e1_student.py) + backward + step.

NOT real training: no ImageNet init, no real class weights, no multi-scale aug, no checkpoint,
no validation, no epoch. weight=None is acceptable for B17 only (see reports/train_step_smoke.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.seeds import set_seed                                   # noqa: E402
from src.data import NUM_CLASSES, build_dataloader              # noqa: E402
from src.models.student import build_student                    # noqa: E402
from src.training.losses import (                               # noqa: E402
    CombinedCEDiceLoss, SoftDiceLoss, WeightedCrossEntropyLoss)
from configs.e1_student import E1_STUDENT                       # noqa: E402

N_STEPS = 3


def main() -> int:
    set_seed(42)
    device = torch.device("cpu")
    print(f"[env] torch {torch.__version__} | device=cpu | num_classes={NUM_CLASSES}")
    print(f"[opt] SGD lr={E1_STUDENT['learning_rate']} momentum={E1_STUDENT['momentum']} "
          f"weight_decay={E1_STUDENT['weight_decay']} (from configs/e1_student.py)")

    student = build_student().to(device)        # weights=None inside _build_backbone (no download)
    student.train()
    n_params = sum(p.numel() for p in student.parameters())
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[student] params={n_params:,} trainable={n_train:,} used_pretrained={student.used_pretrained}")

    # one real train batch (seeded shuffle -> deterministic first batch at seed 42)
    img, mask = next(iter(build_dataloader("train", 2)))
    img, mask = img.to(device), mask.to(device)
    uniq = sorted(int(v) for v in torch.unique(mask).tolist())
    print(f"[batch] img {tuple(img.shape)} {img.dtype} | mask {tuple(mask.shape)} {mask.dtype}")
    print(f"[batch] mask unique ({len(uniq)}): {uniq if len(uniq) <= 20 else uniq[:18] + ['...', uniq[-1]]}")

    checks: dict[str, bool] = {}
    checks["img_shape"] = (tuple(img.shape) == (2, 3, 512, 512) and img.dtype == torch.float32)
    checks["mask_shape"] = (tuple(mask.shape) == (2, 512, 512) and mask.dtype == torch.int64)

    ce_fn = WeightedCrossEntropyLoss(weight=None, ignore_index=255)
    dice_fn = SoftDiceLoss(ignore_index=255)
    crit = CombinedCEDiceLoss(weight=None, ignore_index=255)
    opt = torch.optim.SGD(student.parameters(), lr=E1_STUDENT["learning_rate"],
                          momentum=E1_STUDENT["momentum"], weight_decay=E1_STUDENT["weight_decay"])

    traj: list[float] = []
    ce0 = dice0 = comb0 = None
    grad_summary = {}
    opt_moved = None

    for step in range(N_STEPS):
        opt.zero_grad(set_to_none=True)
        logits = student(img)

        if step == 0:
            print(f"[forward] logits {tuple(logits.shape)} {logits.dtype} "
                  f"finite={bool(torch.isfinite(logits).all())}")
            checks["logits_shape"] = tuple(logits.shape) == (2, 116, 512, 512)
            ce0 = float(ce_fn(logits, mask).item())
            dice0 = float(dice_fn(logits, mask).item())

        loss = crit(logits, mask)
        if step == 0:
            comb0 = float(loss.item())
            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
            print(f"[loss] CE={ce0:.4f} Dice={dice0:.4f} CE+Dice={comb0:.4f} "
                  f"(scalar={loss.dim() == 0}, finite={bool(torch.isfinite(loss))})")

        loss.backward()

        if step == 0:
            total = with_grad = none_grad = nonfinite = 0
            none_examples = []
            for name, p in student.named_parameters():
                if not p.requires_grad:
                    continue
                total += 1
                if p.grad is None:
                    none_grad += 1
                    if len(none_examples) < 10:
                        none_examples.append(name)
                else:
                    with_grad += 1
                    if not torch.isfinite(p.grad).all():
                        nonfinite += 1
            grad_summary = {"trainable": total, "with_grad": with_grad,
                            "none_grad": none_grad, "nonfinite": nonfinite}
            checks["grad_flow"] = (none_grad == 0 and nonfinite == 0 and with_grad == total and total > 0)
            print(f"[grad] trainable={total} with_grad={with_grad} none_grad={none_grad} "
                  f"nonfinite={nonfinite}")
            if none_examples:
                print(f"[grad] none-grad examples: {none_examples}")
            # snapshot a trainable param with a grad, to verify the optimizer moves it
            pname, pp = next((n, p) for n, p in student.named_parameters()
                             if p.requires_grad and p.grad is not None)
            before = pp.detach().clone()

        opt.step()

        if step == 0:
            opt_moved = bool((pp.detach() - before).abs().sum().item() > 0.0)
            checks["opt_updates"] = opt_moved
            print(f"[optstep] param '{pname}' changed after step: {opt_moved}")

        traj.append(round(float(loss.item()), 4))

    with torch.no_grad():
        post = round(float(crit(student(img), mask).item()), 4)
    print(f"[trajectory] step losses {traj} -> post-step {post}")
    non_increasing = (all(traj[i + 1] <= traj[i] + 1e-6 for i in range(len(traj) - 1))
                      and post <= traj[-1] + 1e-6)
    decreased = post < traj[0]
    print(f"[trajectory] non_increasing(sanity, non-gating)={non_increasing} decreased_overall={decreased}")

    hard = ["img_shape", "mask_shape", "logits_shape", "loss_finite", "grad_flow", "opt_updates"]
    passed = all(checks.get(k, False) for k in hard)
    print("\n[CHECKS] (hard = gating)")
    for k in hard:
        print(f"  {k:14}: {'PASS' if checks.get(k) else 'FAIL'}")
    print(f"  trajectory     : {'non-increasing' if non_increasing else 'NOT strictly non-increasing'} "
          f"(sanity only, non-gating)")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
