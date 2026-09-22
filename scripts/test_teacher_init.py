#!/usr/bin/env python3
"""Sanity-check the ADE20K-pretrained SegNeXt-B / MSCAN-B teacher INIT checkpoint.

READ-ONLY: this does NOT train, fine-tune, or modify the checkpoint. It only loads
the stock config + checkpoint, runs one dummy forward, and verifies the
checkpoint matches the stock ADE20K (num_classes=150) architecture exactly.

Run inside the PINNED MMSegmentation env (torch 2.1.0, mmcv 2.1.0, mmseg 1.2.2).
The checkpoint lives OUTSIDE the repository (B61 §1; runbook §5). Nothing is downloaded and there is
no repository `weights/` fallback (G11, B62):

  python scripts/test_teacher_init.py                               # $SEGNEXT_ADE20K_CKPT + stock config
  python scripts/test_teacher_init.py <config.py> <checkpoint.pth>  # explicit paths
  python scripts/test_teacher_init.py --expect-sha256 ANY ...       # skip the identity check (diagnostics)

The config defaults to the stock `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py` shipped in
the pinned mmseg package. The checkpoint's SHA-256 must equal the readiness-verified value unless
`--expect-sha256` says otherwise.

PASS iff: init_model loads, one dummy forward runs, the checkpoint state_dict
loads into the stock num_classes=150 model with ZERO missing AND ZERO unexpected
keys, and decode_head.conv_seg has 150 out-channels.

Exit codes: 0 PASS · 1 FAIL · 2 environment / input refused (stack missing, file missing,
checkpoint inside the repository, SHA-256 mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_STEM = "segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512"
CKPT_ENV = "SEGNEXT_ADE20K_CKPT"
EXPECTED_SHA256 = "647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1"   # B61 §1
EXPECTED_NUM_CLASSES = 150
EXPECTED_PARAMS_M = 27.6  # sanity reference only


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_paths(args, mmseg_dir: Path):
    """Explicit arguments, else $SEGNEXT_ADE20K_CKPT and the pinned stock config. Never weights/."""
    ckpt = args.checkpoint or os.environ.get(CKPT_ENV)
    cfg = args.config or str(mmseg_dir / ".mim" / "configs" / "segnext" / f"{CONFIG_STEM}.py")
    return (Path(cfg) if cfg else None), (Path(ckpt) if ckpt else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stock ADE20K SegNeXt-B init sanity check (read-only).")
    p.add_argument("config", nargs="?", default=None, help="stock config (default: pinned mmseg copy)")
    p.add_argument("checkpoint", nargs="?", default=None, help=f"checkpoint (default: ${CKPT_ENV})")
    p.add_argument("--expect-sha256", default=EXPECTED_SHA256,
                   help="required checkpoint SHA-256 (default: the readiness value); ANY skips the check")
    args = p.parse_args(argv)

    # G11: EVERY third-party import sits inside the guard, so a missing stack exits 2 with guidance
    # instead of a bare traceback.
    try:
        import numpy as np
        import mmseg
        from mmengine.runner import CheckpointLoader
        from mmseg.apis import inference_model, init_model
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: MMSegmentation env not importable ({type(e).__name__}: {e}).")
        print("      Run inside the pinned env: torch 2.1.0, mmcv 2.1.0, mmseg 1.2.2.")
        return 2

    cfg_path, ckpt_path = resolve_paths(args, Path(mmseg.__file__).resolve().parent)
    if ckpt_path is None:
        print(f"FAIL: no checkpoint given. Pass <config> <checkpoint> or set ${CKPT_ENV} to the "
              "out-of-repo ADE20K checkpoint. There is no repository weights/ fallback.")
        return 2
    if not (cfg_path and cfg_path.is_file() and ckpt_path.is_file()):
        print(f"FAIL: config/checkpoint not found (cfg={cfg_path}, ckpt={ckpt_path}).")
        return 2
    resolved = ckpt_path.resolve()
    if resolved == REPO or REPO in resolved.parents:
        print(f"FAIL: refusing a checkpoint inside the repository: {resolved}. The ADE20K checkpoint "
              "lives outside the Git repository (runbook §5).")
        return 2
    if args.expect_sha256 != "ANY":
        sha = _sha256(resolved)
        if sha != args.expect_sha256:
            print(f"FAIL: checkpoint sha256 {sha} != expected {args.expect_sha256} (B61 §1 readiness).")
            return 2
        print(f"[identity] sha256 = {sha} (matches the readiness-verified checkpoint)")
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
