#!/usr/bin/env python3
"""Sanity-check the ADE20K-pretrained SegNeXt-B / MSCAN-B teacher INIT checkpoint.

READ-ONLY: this does NOT train, fine-tune, or modify the checkpoint. It only loads
the downloaded config + checkpoint, runs one dummy forward, and verifies the
checkpoint matches the stock ADE20K (num_classes=150) architecture exactly.

Run inside the PINNED MMSegmentation env (torch 2.1.0, mmcv 2.1.0, mmseg 1.2.2).
Obtain the config + checkpoint first:

  mim download mmsegmentation \\
    --config segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512 --dest weights/

Usage:
  python scripts/test_teacher_init.py                 # auto-discover in weights/
  python scripts/test_teacher_init.py <config.py> <checkpoint.pth>

PASS iff: init_model loads, one dummy forward runs, the checkpoint state_dict
loads into the stock num_classes=150 model with ZERO missing AND ZERO unexpected
keys, and decode_head.conv_seg has 150 out-channels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights"
CONFIG_STEM = "segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512"
EXPECTED_NUM_CLASSES = 150
EXPECTED_PARAMS_M = 27.6  # sanity reference only


def _find_one(patterns: list[str]):
    for pat in patterns:
        hits = sorted(WEIGHTS.glob(pat))
        if hits:
            return hits[0]
    return None


def resolve_paths(argv: list[str]):
    if len(argv) >= 3:
        return Path(argv[1]), Path(argv[2])
    cfg = _find_one([f"{CONFIG_STEM}.py", "segnext_mscan-b*ade20k*.py", "*.py"])
    ckpt = _find_one(["segnext_mscan-b*ade20k*.pth", "segnext_mscan-b*.pth", "*.pth"])
    return cfg, ckpt


def main() -> int:
    try:
        from mmseg.apis import inference_model, init_model
        from mmengine.runner import CheckpointLoader
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: MMSegmentation env not importable ({type(e).__name__}: {e}).")
        print("      Run inside the pinned env: torch 2.1.0, mmcv 2.1.0, mmseg 1.2.2.")
        return 2

    cfg_path, ckpt_path = resolve_paths(sys.argv)
    if not (cfg_path and ckpt_path and cfg_path.exists() and ckpt_path.exists()):
        print(f"FAIL: config/checkpoint not found in {WEIGHTS} (cfg={cfg_path}, ckpt={ckpt_path}).")
        print("      Run the `mim download ... --dest weights/` step first.")
        return 2
    print(f"[paths] config     = {cfg_path}")
    print(f"[paths] checkpoint = {ckpt_path}")

    # 1) init_model loads the checkpoint onto CPU
    model = init_model(str(cfg_path), str(ckpt_path), device="cpu")
    print("[init_model] OK (checkpoint loaded onto CPU)")

    # 2) one dummy forward on a synthetic 512x512x3 uint8 array
    dummy = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)
    result = inference_model(model, dummy)
    seg = result.pred_sem_seg.data
    print(f"[forward] OK — pred_sem_seg shape {tuple(seg.shape)}, dtype {seg.dtype}")

    # 3) strict=False load of the checkpoint into the STOCK ADE20K (num_classes=150) model
    stock = init_model(str(cfg_path), None, device="cpu")  # build from config only (no weights)
    ckpt = CheckpointLoader.load_checkpoint(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = stock.load_state_dict(state, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    print(f"[state_dict] missing_keys ({len(missing)}): {missing}")
    print(f"[state_dict] unexpected_keys ({len(unexpected)}): {unexpected}")

    # 4) parameter count + classifier out-channels
    n_params = sum(p.numel() for p in model.parameters())
    out_ch = int(model.decode_head.conv_seg.out_channels)
    print(f"[params] total = {n_params:,} (~{n_params / 1e6:.1f}M; expect ~{EXPECTED_PARAMS_M}M)")
    print(f"[conv_seg] decode_head.conv_seg.out_channels = {out_ch} (expect {EXPECTED_NUM_CLASSES})")

    clean = (len(missing) == 0 and len(unexpected) == 0 and out_ch == EXPECTED_NUM_CLASSES)

    # Explicit assertions — PASS only if keys load cleanly with zero mismatches.
    assert len(missing) == 0, f"missing_keys not empty: {missing}"
    assert len(unexpected) == 0, f"unexpected_keys not empty: {unexpected}"
    assert out_ch == EXPECTED_NUM_CLASSES, f"conv_seg out-channels != {EXPECTED_NUM_CLASSES}: {out_ch}"

    print("\nRESULT: PASS" if clean else "\nRESULT: FAIL")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
