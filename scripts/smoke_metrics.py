#!/usr/bin/env python3
"""Metrics smoke test (Blocker B11b). No training. Reuses src/eval/metrics.py.

Tiny synthetic batch: verifies ignore_index=255 exclusion, absent-class handling, all-class mIoU
sanity, and computes a PROVISIONAL disease-only mIoU (excludes index 0; NOT a settled convention,
does NOT gate PASS). Writes reports/metrics_smoke.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.seeds import set_seed  # noqa: E402
from src.eval import all_class_miou, confusion_matrix, disease_only_miou, per_image_miou  # noqa: E402
from configs.loss import LOSS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NC = LOSS["num_classes"]          # 116
IGNORE = LOSS["ignore_index"]     # 255
BG = 0                            # provisional background index


def write_report(path: Path, title: str, body: list[str], checks: list[tuple[str, bool]],
                 all_ok: bool, ntc: list[str]) -> None:
    md = [f"# {title}", "", f"**RESULT: {'PASS' if all_ok else 'FAIL'}**  ",
          "_Generated: 2026-06-27 by `scripts/smoke_metrics.py` (synthetic; no training)._", "",
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

    out(f"num_classes={NC}  ignore_index={IGNORE}")

    # Synthetic GT: classes {0,5} + 255 ignore; most of the 116 classes are absent.
    target = torch.tensor([[[0, 0, 5], [5, 255, 255]],
                           [[0, 5, 5], [255, 5, 0]]], dtype=torch.long)  # [2,2,3]
    pred_perfect = target.clone()
    pred_perfect[target == IGNORE] = 9  # arbitrary valid label at ignore positions

    # (1) ignore 255 excluded: CM total == non-255 pixel count
    valid_count = int((target != IGNORE).sum())
    cm_total = int(confusion_matrix(pred_perfect, target, NC, IGNORE).sum())
    ig_count_ok = cm_total == valid_count
    out(f"non-255 pixels={valid_count}  CM total={cm_total}")
    checks.append(("ignore 255 excluded (CM total == non-255 count)", ig_count_ok))

    # (1b) ignore 255 excluded: mIoU invariant to predictions at 255 positions
    pred_alt = pred_perfect.clone()
    pred_alt[target == IGNORE] = 50
    inv_ok = abs(all_class_miou(pred_perfect, target, NC) - all_class_miou(pred_alt, target, NC)) < 1e-9
    checks.append(("ignore 255 excluded (mIoU invariant to 255-position preds)", inv_ok))

    # (2) absent classes excluded -> perfect prediction gives mIoU == 1.0 over present classes only
    present = sorted(int(v) for v in torch.unique(target[target != IGNORE]).tolist())
    allc_perfect = all_class_miou(pred_perfect, target, NC)
    absent_ok = abs(allc_perfect - 1.0) < 1e-9
    out(f"present GT classes={present}  all-class mIoU(perfect)={allc_perfect:.6f}")
    checks.append((f"absent classes excluded (perfect mIoU==1.0 over present {present})", absent_ok))

    # (3) all-class mIoU sanity on a partial prediction (one class-0 pixel mislabeled as 5)
    pred_partial = pred_perfect.clone()
    pred_partial[1, 0, 0] = 5  # was class 0
    allc_partial = all_class_miou(pred_partial, target, NC)
    sanity_ok = 0.0 < allc_partial < 1.0
    out(f"all-class mIoU(partial)={allc_partial:.6f}")
    checks.append((f"all-class mIoU sanity (partial in (0,1): {allc_partial:.4f})", sanity_ok))

    # provisional disease-only (does NOT gate PASS)
    dis = disease_only_miou(pred_perfect, target, NC, background_index=BG)
    per_img = per_image_miou(pred_perfect, target, NC, class_indices=torch.arange(1, NC))
    out(f"disease-only mIoU(perfect) = {dis:.6f}  ** PROVISIONAL / NEED_TO_CONFIRM (excludes index {BG}) **")
    out(f"per-image disease-only (PROVISIONAL) = {[round(v, 4) for v in per_img]}")

    print("\n--- checks ---")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(ok for _, ok in checks)
    out(f"[{'PASS' if all_ok else 'FAIL'}] metrics smoke test")

    write_report(REPO / "reports" / "metrics_smoke.md", "Metrics smoke test — B11b", body, checks,
                 all_ok, ["disease-only / background convention (reduce_zero_label) — disease-only mIoU "
                          "is PROVISIONAL (excludes empirical background index 0; not yet settled)."])
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
