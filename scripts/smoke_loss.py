#!/usr/bin/env python3
"""Loss smoke test (Blocker B11b). Reuses src/training/losses.py.

Tiny synthetic logits + mask (with 255 ignore). Forward CE + Dice, then ONE tiny backward as a
gradient smoke check (no optimizer, no loop, no scheduler, no checkpoint — NOT training). Verifies
finite loss + finite gradients, and that perturbing logits at 255 positions leaves CE and Dice
unchanged (ignore-255 correctness). Writes reports/loss_smoke.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.seeds import set_seed  # noqa: E402
from src.training import SoftDiceLoss, WeightedCrossEntropyLoss  # noqa: E402
from configs.loss import LOSS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NC = LOSS["num_classes"]          # 116
IGNORE = LOSS["ignore_index"]     # 255


def finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all())


def write_report(path: Path, title: str, body: list[str], checks: list[tuple[str, bool]],
                 all_ok: bool, ntc: list[str]) -> None:
    md = [f"# {title}", "", f"**RESULT: {'PASS' if all_ok else 'FAIL'}**  ",
          "_Generated: 2026-06-27 by `scripts/smoke_loss.py` (synthetic; tiny backward smoke; no training)._", "",
          "## Output", "```", *body, "```", "", "## Checks (gate PASS)"]
    md += [f"- [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    md += ["", "## NEED_TO_CONFIRM", *[f"- {x}" for x in ntc], ""]
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    set_seed()
    body: list[str] = []
    checks: list[tuple[str, bool]] = []

    def out(s: str) -> None:
        print(s)
        body.append(s)

    out(f"num_classes={NC}  ignore_index={IGNORE}  (CE weight=None: real class weights NEED_TO_CONFIRM)")

    # synthetic logits (require grad for the tiny backward) + mask with some 255 ignore pixels
    logits = torch.randn(2, NC, 4, 4, requires_grad=True)
    target = torch.full((2, 4, 4), IGNORE, dtype=torch.long)
    target[0, 0, 0] = 0
    target[0, 1, 2] = 5
    target[1, 2, 1] = 0
    target[1, 3, 3] = 7
    n_valid = int((target != IGNORE).sum())
    n_ignore = int((target == IGNORE).sum())
    out(f"valid pixels={n_valid}  ignore(255) pixels={n_ignore}")

    ce = WeightedCrossEntropyLoss(weight=None)(logits, target)
    dice = SoftDiceLoss()(logits, target)
    combined = ce + dice
    out(f"CE={ce.item():.4f}  Dice={dice.item():.4f}  CE+Dice={combined.item():.4f}")
    checks.append(("CE finite", finite(ce)))
    checks.append(("Dice finite", finite(dice)))
    checks.append(("CE+Dice finite", finite(combined)))

    # one tiny backward (gradient smoke only — no optimizer / loop / checkpoint)
    combined.backward()
    grad_finite = logits.grad is not None and finite(logits.grad)
    gmin, gmax = float(logits.grad.min()), float(logits.grad.max())
    out(f"grad finite={grad_finite}  grad range=[{gmin:.4e}, {gmax:.4e}]")
    checks.append(("gradients finite", grad_finite))

    # ignore-255 correctness: perturb logits at 255 positions -> CE and Dice unchanged
    with torch.no_grad():
        ig = (target == IGNORE).unsqueeze(1).expand(2, NC, 4, 4)
        pert = torch.where(ig, logits.detach() + 1000.0 * torch.randn_like(logits), logits.detach())
    ce2 = WeightedCrossEntropyLoss(weight=None)(pert, target)
    dice2 = SoftDiceLoss()(pert, target)
    ce_inv = torch.allclose(ce.detach(), ce2)
    dice_inv = torch.allclose(dice.detach(), dice2)
    out(f"ignore-perturb: CE {ce.item():.4f}->{ce2.item():.4f} ({'==' if ce_inv else '!='})  "
        f"Dice {dice.item():.4f}->{dice2.item():.4f} ({'==' if dice_inv else '!='})")
    checks.append(("ignore-255 perturbation leaves CE unchanged", ce_inv))
    checks.append(("ignore-255 perturbation leaves Dice unchanged", dice_inv))

    print("\n--- checks ---")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(ok for _, ok in checks)
    out(f"[{'PASS' if all_ok else 'FAIL'}] loss smoke test")

    write_report(REPO / "reports" / "loss_smoke.md", "Loss smoke test — B11b", body, checks, all_ok,
                 ["real CE class weights — batch/None here; full train-set weights NEED_TO_CONFIRM",
                  "logit KD weight (lambda_logit) — NEED_TO_CONFIRM (validation sweep)",
                  "reduce_zero_label — NEED_TO_CONFIRM"])
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
