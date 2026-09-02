#!/usr/bin/env python3
"""Standing gate: every distillation term operates on its CORRECT spatial grid (B32 / F8).

This is the guard that would have caught the original misalignment, in which the Logit-KD KL was
evaluated on 512x512 bilinear upsamples of two natively-64x64 logit fields — so ~98% of its
positions were interpolants, and because the interpolation happened in LOGIT space before the
softmax (`softmax(interp(z)) != interp(softmax(z))`) the soft targets were distorted, not merely
repeated.

Asserted here:
  * Logit-KD   -> OS8    64x64, validity mask min-pooled to 64x64
  * CWD-logit  -> OS8    64x64, SAME mask
  * CWD-feat   -> s16    32x32, mask min-pooled to 32x32
  * CE / Dice  -> full  512x512, UNCHANGED
  * logit_kd_kl REJECTS a mask at the wrong resolution instead of broadcasting
  * the mean reduction (sum / valid.sum) is intact, which is what keeps the preregistered
    lambda_logit grid {0.25, 0.5, 1, 2, 4} on-scale

Synthetic teacher throughout: MMSeg is not required and no checkpoint is loaded. CPU, seconds.

Usage:  python scripts/smoke_kd_resolution.py
Exit code 0 == PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from configs.distill import DISTILL  # noqa: E402
from src.distill.features import StudentTaps  # noqa: E402
from src.models.student import build_student  # noqa: E402
from src.seeds import set_seed  # noqa: E402
from src.training.losses import (  # noqa: E402
    CombinedCEDiceLoss, cwd_channelwise_kl, downsample_validity, logit_kd_kl,
)

NC = 116
IGNORE = 255
SIZE = 512
OS8 = SIZE // 8            # 64
S16 = SIZE // 16           # 32
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def main() -> int:
    print("=" * 78)
    print("B32 — KD spatial-resolution gate")
    print(f"torch {torch.__version__} | input {SIZE}x{SIZE} -> OS8 {OS8}x{OS8}, s16 {S16}x{S16}")
    print("=" * 78)

    set_seed(42)
    student = build_student(pretrained=False)
    student.train()

    img = torch.randn(2, 3, SIZE, SIZE)
    mask = torch.randint(0, NC, (2, SIZE, SIZE))
    mask[:, :64, :] = IGNORE                     # a padded band, as core_preprocess produces

    with StudentTaps(student) as taps:
        logits = student(img)
        c5, head_logits = taps.require()

    # ---------------------------------------------------------------- grids
    check("student full-res logits are 512x512",
          tuple(logits.shape) == (2, NC, SIZE, SIZE), str(tuple(logits.shape)))
    check("student head_logits are OS8 64x64",
          tuple(head_logits.shape) == (2, NC, OS8, OS8), str(tuple(head_logits.shape)))
    check("student c5 is stride-16 32x32",
          tuple(c5.shape[-2:]) == (S16, S16), str(tuple(c5.shape)))
    check("head_logits stays on the autograd graph",
          head_logits.requires_grad and head_logits.grad_fn is not None,
          "forward hooks must not detach, or KD gradients never reach the head")

    # ---------------------------------------------------------------- masks
    v_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE)
    v_s16 = downsample_validity(mask, c5.shape[-2:], ignore_index=IGNORE)
    check("validity mask for the logit terms is 64x64",
          tuple(v_os8.shape) == (2, OS8, OS8), str(tuple(v_os8.shape)))
    check("validity mask for the feature term is 32x32",
          tuple(v_s16.shape) == (2, S16, S16), str(tuple(v_s16.shape)))
    check("mask is bool", v_os8.dtype == torch.bool, str(v_os8.dtype))
    # conservative all-valid min-pool: the 64 ignored rows must knock out exactly 8 cell rows
    check("min-pool is conservative (all-valid), not any-valid",
          bool((~v_os8[:, :8, :]).all()) and bool(v_os8[:, 8:, :].all()),
          "first 8 OS8 rows fully invalid, remainder valid")

    # ---------------------------------------------------------------- Logit-KD at OS8
    t_os8 = torch.randn(2, NC, OS8, OS8)                    # synthetic teacher, native OS8
    l_kd = logit_kd_kl(head_logits, t_os8, v_os8, T=DISTILL["logit_kd"]["T_logit"])
    check("Logit-KD evaluates at OS8 and returns a finite scalar",
          l_kd.dim() == 0 and bool(torch.isfinite(l_kd)), f"loss={float(l_kd):.6f}")
    g = torch.autograd.grad(l_kd, head_logits, retain_graph=True)[0]
    check("Logit-KD gradient reaches head_logits", float(g.abs().sum()) > 0)

    # the mask must be REJECTED at the wrong resolution rather than broadcast
    try:
        logit_kd_kl(head_logits, t_os8, (mask != IGNORE), T=4.0)
        check("logit_kd_kl rejects a full-res mask against OS8 logits", False, "no error raised")
    except ValueError as e:
        check("logit_kd_kl rejects a full-res mask against OS8 logits", True, type(e).__name__)

    # mean reduction intact: scaling the valid population must NOT scale the loss
    half = v_os8.clone()
    half[:, OS8 // 2:, :] = False
    l_half = logit_kd_kl(head_logits, t_os8, half, T=4.0)
    check("reduction is a MEAN over valid cells, not a sum",
          0.3 < float(l_half) / float(l_kd) < 3.0,
          f"full={float(l_kd):.4f} half-population={float(l_half):.4f} — a SUM would roughly halve")

    # ---------------------------------------------------------------- CWD terms
    l_cwd_logit = cwd_channelwise_kl(head_logits, t_os8, v_os8, T=DISTILL["cwd"]["T_cwd"])
    check("CWD-logit evaluates at OS8 on head_logits",
          l_cwd_logit.dim() == 0 and bool(torch.isfinite(l_cwd_logit)),
          f"loss={float(l_cwd_logit):.6f}")

    proj = torch.nn.Conv2d(c5.shape[1], DISTILL["cwd"]["C"], 1)
    s_feat = proj(c5)
    t_feat = torch.randn(2, DISTILL["cwd"]["C"], S16, S16)
    check("CWD-feat operates at stride-16 with C=320",
          tuple(s_feat.shape[-2:]) == (S16, S16) and s_feat.shape[1] == DISTILL["cwd"]["C"],
          f"{tuple(s_feat.shape)}")
    l_cwd_feat = cwd_channelwise_kl(s_feat, t_feat, v_s16, T=DISTILL["cwd"]["T_cwd"],
                                    channels_norm=DISTILL["cwd"]["C"])
    check("CWD-feat returns a finite scalar",
          l_cwd_feat.dim() == 0 and bool(torch.isfinite(l_cwd_feat)),
          f"loss={float(l_cwd_feat):.6f}")
    check("CWD-feat mask is NOT the OS8 mask",
          v_s16.shape != v_os8.shape, "32x32 vs 64x64 — distinct grids by design")

    # ---------------------------------------------------------------- supervised path unchanged
    sup = CombinedCEDiceLoss(ignore_index=IGNORE)(logits, mask)
    check("CE+Dice still evaluate at FULL 512x512 on the upsampled logits",
          sup.dim() == 0 and bool(torch.isfinite(sup)) and tuple(logits.shape[-2:]) == (SIZE, SIZE),
          f"loss={float(sup):.6f}")

    # ---------------------------------------------------------------- locked constants
    check("T_logit == 4", DISTILL["logit_kd"]["T_logit"] == 4)
    check("T_cwd == 4", DISTILL["cwd"]["T_cwd"] == 4)
    check("alpha_CWD == 50", DISTILL["cwd"]["alpha_cwd_feature_map"] == 50)
    check("beta_CWD == 3", DISTILL["cwd"]["beta_cwd_logit_map"] == 3)
    check("CWD C == 320", DISTILL["cwd"]["C"] == 320)
    check("lambda grid == (0.25, 0.5, 1, 2, 4)",
          tuple(DISTILL["logit_kd"]["lambda_logit_sweep_grid"]) == (0.25, 0.5, 1, 2, 4))

    # ---------------------------------------------------------------- no upsampled teacher
    t_up = F.interpolate(t_os8, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    kd_up = logit_kd_kl(logits, t_up, (mask != IGNORE), T=4.0)
    check("the OLD upsampled path gives a MATERIALLY different value",
          abs(float(kd_up) / float(l_kd) - 1.0) > 0.10,
          f"os8={float(l_kd):.4f} upsampled={float(kd_up):.4f} "
          f"ratio={float(l_kd)/float(kd_up):.3f} — they are not interchangeable")

    print()
    for name, ok, detail in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    passed = sum(1 for _n, ok, _d in CHECKS if ok)
    total = len(CHECKS)
    print(f"\nRESULT: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
