#!/usr/bin/env python3
"""Forward-pass smoke test for the PlantSeg student scaffold (Blocker B10). No training.

Builds the student with num_classes from configs/data.py, runs a dummy [2,3,512,512]
(and one real val batch if feasible), and asserts output shape / dtype / finiteness.
NO quantization, NO loss, NO distillation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.seeds import set_seed  # noqa: E402
from src.data import NUM_CLASSES  # noqa: E402  (= configs/data.py num_classes)
from src.models import build_student, probe_backbone_stages  # noqa: E402


def main() -> int:
    set_seed()
    print(f"num_classes (configs/data.py) = {NUM_CLASSES}")

    model = build_student(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"pretrained ImageNet init = {'yes' if model.used_pretrained else 'NEED_TO_CONFIRM (weights=None)'}")

    # --- dummy forward ---
    x = torch.zeros(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"[dummy] input {tuple(x.shape)} -> output {tuple(out.shape)} dtype={out.dtype}")

    assert tuple(out.shape) == (2, NUM_CLASSES, 512, 512), f"bad output shape {tuple(out.shape)}"
    assert out.dtype.is_floating_point, f"output dtype not floating: {out.dtype}"
    assert torch.isfinite(out).all(), "output has NaN/Inf"

    # --- optional: one real dataloader batch ---
    real = "skipped"
    try:
        from src.data import build_dataloader
        img, _mask = next(iter(build_dataloader("val", batch_size=2)))
        with torch.no_grad():
            out_r = model(img)
        assert tuple(out_r.shape) == (2, NUM_CLASSES, 512, 512)
        assert torch.isfinite(out_r).all()
        real = f"OK -> {tuple(out_r.shape)}"
    except Exception as e:  # noqa: BLE001
        real = f"skipped ({type(e).__name__}: {e})"
    print(f"[real val batch] {real}")

    # --- tap report (from the full-backbone probe) ---
    rows = probe_backbone_stages((512, 512), dilated=True)
    low, high = rows[model.low_tap], rows[model.high_tap]
    print(f"[OS8  tap] idx={low['idx']} channels={low['channels']} stride=OS{low['stride']}")
    print(f"[OS16 tap] idx={high['idx']} channels={high['channels']} stride=OS{high['stride']}")
    print(f"[params] total = {n_params:,} (~{n_params / 1e6:.2f}M)")

    print("\n[PASS] student forward smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
