#!/usr/bin/env python3
"""Smoke test for the training-only CWD projection stub (Blocker B12). No training.

Builds the 1x1 projection (student C5 160ch -> teacher 320ch) and verifies a dummy
[2,160,32,32] tensor maps to [2,320,32,32] with no NaN/Inf.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.seeds import set_seed  # noqa: E402
from src.distill import build_cwd_projection  # noqa: E402

IN_CH, OUT_CH = 160, 320


def main() -> int:
    set_seed()
    proj = build_cwd_projection(in_ch=IN_CH, out_ch=OUT_CH)
    proj.eval()

    x = torch.randn(2, IN_CH, 32, 32)
    with torch.no_grad():
        y = proj(x)

    n_params = sum(p.numel() for p in proj.parameters())
    print(f"projection: bias-free 1x1 conv {IN_CH} -> {OUT_CH}")
    print(f"[shapes] input {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"[params] total = {n_params:,} (expect {IN_CH * OUT_CH:,})")

    shape_ok = tuple(y.shape) == (2, OUT_CH, 32, 32)
    finite_ok = bool(torch.isfinite(y).all())
    params_ok = n_params == IN_CH * OUT_CH and proj.bias is None
    print(f"[checks] shape={shape_ok} finite={finite_ok} bias_free+params={params_ok}")

    ok = shape_ok and finite_ok and params_ok
    print(f"\n[{'PASS' if ok else 'FAIL'}] CWD projection stub smoke test")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
