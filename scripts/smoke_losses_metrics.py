#!/usr/bin/env python3
"""Smoke test for loss + metric scaffolds (Blocker B11). No training, no backward.

Builds one real val batch + the student, runs a forward pass, then exercises CE / Dice /
CE+Dice / logit-KD-KL (vs a dummy teacher) and all-class + provisional disease-only mIoU.

PASS depends ONLY on: finite losses, valid all-class mIoU, correct 255 handling, valid
shapes/dtypes, no NaN/Inf. Provisional disease-only mIoU does NOT gate PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.seeds import set_seed  # noqa: E402
from src.data import NUM_CLASSES, build_dataloader  # noqa: E402
from src.models import build_student  # noqa: E402
from src.training import (  # noqa: E402
    CombinedCEDiceLoss, SoftDiceLoss, WeightedCrossEntropyLoss, compute_class_weights, logit_kd_kl,
)
from src.eval import all_class_miou, confusion_matrix, disease_only_miou, per_image_miou  # noqa: E402

BACKGROUND_INDEX = 0  # empirical; disease-only convention is PROVISIONAL / NEED_TO_CONFIRM


def finite(x) -> bool:
    return bool(torch.isfinite(torch.as_tensor(x)).all())


def main() -> int:
    set_seed()
    checks: list[tuple[str, bool]] = []
    print(f"num_classes (configs/data.py) = {NUM_CLASSES} | ignore_index = 255")

    # --- one real batch + forward ---
    img, mask = next(iter(build_dataloader("val", batch_size=2)))
    model = build_student(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(img)

    shape_ok = tuple(logits.shape) == (2, NUM_CLASSES, 512, 512) and logits.dtype.is_floating_point
    logits_finite = finite(logits)
    print(f"[forward] logits {tuple(logits.shape)} dtype={logits.dtype} finite={logits_finite}")
    checks += [("output shape/dtype valid", shape_ok), ("logits finite (no NaN/Inf)", logits_finite)]

    # --- class weights (SMOKE-ONLY, batch-derived; real train-set weights NEED_TO_CONFIRM) ---
    weights, wstats = compute_class_weights(mask, NUM_CLASSES)
    print(f"[class weights] SMOKE-ONLY batch-derived: present={wstats['n_present']} "
          f"min={wstats['min']:.3f} median={wstats['median']:.3f} max={wstats['max']:.3f}")

    # --- losses (no backward) ---
    with torch.no_grad():
        ce = WeightedCrossEntropyLoss(weight=weights)(logits, mask)
        dice = SoftDiceLoss()(logits, mask)
        combined = CombinedCEDiceLoss(weight=weights)(logits, mask)
        teacher = torch.randn_like(logits)  # dummy teacher (smoke only)
        kd = logit_kd_kl(logits, teacher, mask, T=4.0)
    print(f"[loss] CE={ce.item():.4f}  Dice={dice.item():.4f}  CE+Dice={combined.item():.4f}  "
          f"logitKD-KL(T=4, dummy teacher)={kd.item():.4f}")
    for name, v in [("CE", ce), ("Dice", dice), ("CE+Dice", combined), ("logitKD-KL", kd)]:
        checks.append((f"{name} finite", finite(v)))

    # --- metrics on dummy predictions (argmax of untrained logits) ---
    pred = logits.argmax(dim=1)  # [2,512,512]
    allc = all_class_miou(pred, mask, NUM_CLASSES)
    allc_ok = finite(allc) and 0.0 <= allc <= 1.0
    print(f"[metric] all-class mIoU = {allc:.6f}  (valid range [0,1]: {allc_ok})")
    checks.append(("all-class mIoU in [0,1]", allc_ok))

    # ignore_index 255 handling: CM total must equal the non-255 pixel count
    valid_count = int((mask != 255).sum().item())
    cm_total = int(confusion_matrix(pred, mask, NUM_CLASSES).sum().item())
    ignore_ok = cm_total == valid_count
    print(f"[ignore] non-255 pixels={valid_count}  CM total={cm_total}  -> 255 excluded: {ignore_ok}")
    checks.append(("ignore_index 255 handled correctly", ignore_ok))

    # provisional disease-only mIoU (does NOT gate PASS)
    dis = disease_only_miou(pred, mask, NUM_CLASSES, background_index=BACKGROUND_INDEX)
    per_img = per_image_miou(pred, mask, NUM_CLASSES,
                             class_indices=torch.arange(1, NUM_CLASSES))  # exclude index 0
    print(f"[metric] disease-only mIoU = {dis:.6f}  "
          f"** PROVISIONAL / NEED_TO_CONFIRM (excludes index {BACKGROUND_INDEX}) **")
    print(f"[metric] per-image disease-only mIoU (PROVISIONAL) = {[round(v, 6) for v in per_img]}")

    print("\n--- checks ---")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(ok for _, ok in checks)
    print(f"\n[{'PASS' if all_ok else 'FAIL'}] losses and metrics smoke test")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
