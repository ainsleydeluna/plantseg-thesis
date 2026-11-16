#!/usr/bin/env python3
"""Parity of the exported FP32 TorchScript file with the eager FP32 model it came from.

    python3 -I demo/model/parity_fp32.py --model OUT.pt --checkpoint CKPT --source-sha256 HEX \
        --images PATH [PATH ...] [--n N]

  refuse  every path containing "test" (any case), as given or resolved: TEST data is never read
  check   the checkpoint's sha256 must equal --source-sha256 and the exported file's embedded
          source_checkpoint_sha256, before the checkpoint is read
  load    the exported file through demo/server/inference.py load_fp32_model (identity and 116 classes);
          the eager model through src/quant/checkpoint.py load_student_from_e3, as the export built it
  images  the first N image files (.jpg, .jpeg, .png; name order) of the given files and directories,
          each through the demo's preprocessing (core_preprocess + finalize, demo/server/preprocess.py)
  compare per image: the largest absolute logit difference, and argmax agreement over the canvas and
          over the photo's region
Prints one line per image and then PASS or STOP with the numbers; exit 0 on PASS, 1 on STOP.
PASS needs every image within MAX_ABS_DIFF_LIMIT and with ARGMAX_AGREEMENT_MIN agreement on the canvas.
Runs on CPU with PHONE-FIX-1 (mkldnn off). On the laptop it runs on VAL images; here, on synthetic ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402

MAX_ABS_DIFF_LIMIT = 1e-4        # largest |logit difference| allowed on any image
ARGMAX_AGREEMENT_MIN = 1.0       # share of canvas pixels whose argmax must agree, on every image
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_N = 16


def compare_logits(a, b) -> dict:
    """max |a - b| and argmax agreement of two [N, C, H, W] logit tensors."""
    import torch

    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "max_abs_diff": None, "argmax_agreement": 0.0}
    diff = float((a.float() - b.float()).abs().max())
    agree = float((a.argmax(1) == b.argmax(1)).float().mean())
    return {"shape": list(a.shape), "max_abs_diff": diff, "argmax_agreement": agree,
            "bitwise_equal": bool(torch.equal(a, b))}


def within_limits(row: dict) -> bool:
    return (row["max_abs_diff"] is not None and row["max_abs_diff"] <= MAX_ABS_DIFF_LIMIT
            and row["argmax_agreement"] >= ARGMAX_AGREEMENT_MIN)


def list_images(paths: list[str], n: int) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        common.refuse_test_named(raw, "image path")
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES))
        elif p.is_file():
            found.append(p)
        else:
            raise SystemExit(f"image path not found: {p}")
    for p in found:
        common.refuse_test_named(p, "image")
    if not found:
        raise SystemExit("no image files (.jpg, .jpeg, .png) in the given paths")
    return found[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FP32 TorchScript export vs the eager model: PASS or STOP")
    ap.add_argument("--model", required=True, help="the exported FP32 TorchScript file")
    ap.add_argument("--checkpoint", required=True, help="the checkpoint it was exported from")
    ap.add_argument("--source-sha256", required=True, help="the checkpoint's expected sha256")
    ap.add_argument("--images", nargs="+", required=True, help="image files and/or directories")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help=f"how many images (default {DEFAULT_N})")
    a = ap.parse_args(argv)
    if a.n < 1:
        raise SystemExit("--n must be at least 1")
    model_path, ckpt = config.resolve_model_path(a.model), config.resolve_model_path(a.checkpoint)
    for p, what in ((model_path, "model file"), (ckpt, "checkpoint")):
        common.refuse_test_named(p, what)
    images = list_images(a.images, a.n)

    phone_fix = common.phone_fix()
    import torch

    from demo.server.inference import load_fp32_model
    from demo.server.preprocess import evaluator_preprocess
    from PIL import Image

    want = a.source_sha256.strip().lower()
    got = common.sha256_file(ckpt)
    if got != want:
        raise SystemExit(f"STOP checkpoint sha256 {got} != --source-sha256 {want}; the checkpoint was not read")
    exported = load_fp32_model(model_path)
    if exported.meta["source_checkpoint_sha256"] != got:
        raise SystemExit(f"STOP {model_path.name} was exported from {exported.meta['source_checkpoint_sha256']}, "
                         f"not from this checkpoint ({got})")
    from src.quant.checkpoint import load_student_from_e3

    eager, _ = load_student_from_e3(ckpt)
    eager = eager.eval()
    rows = []
    with torch.inference_mode():
        for p in images:
            with Image.open(p) as im:
                pre = evaluator_preprocess(im)
            ye, yt = eager(pre.tensor), exported.module(pre.tensor)
            row = {"image": p.name, **compare_logits(ye, yt)}
            top, left, h, w = pre.box
            region = (ye.argmax(1)[0, top:top + h, left:left + w] == yt.argmax(1)[0, top:top + h, left:left + w])
            row["argmax_agreement_photo_region"] = float(region.float().mean())
            rows.append(row)
            print(f"[parity] {p.name}: max |diff| {row['max_abs_diff']:.3e}, argmax agreement canvas "
                  f"{row['argmax_agreement']:.6f}, photo region {row['argmax_agreement_photo_region']:.6f}, "
                  f"bitwise {row['bitwise_equal']}", flush=True)
    common.assert_no_stats()
    ok = all(within_limits(r) for r in rows)
    worst = max(r["max_abs_diff"] for r in rows)
    lowest = min(r["argmax_agreement"] for r in rows)
    summary = {"result": "PASS" if ok else "STOP", "images": len(rows), "max_abs_diff": worst,
               "min_argmax_agreement": lowest, "bitwise_equal_images": sum(r["bitwise_equal"] for r in rows),
               "limits": {"max_abs_diff": MAX_ABS_DIFF_LIMIT, "argmax_agreement_min": ARGMAX_AGREEMENT_MIN},
               "model": {"file": model_path.name, "sha256": exported.sha256, "stage": exported.stage,
                         "seed": exported.meta["seed"], "stand_in": exported.stand_in},
               "checkpoint_sha256": got, "phone_fix": phone_fix, "torch": torch.__version__}
    print(json.dumps(summary, indent=2))
    print(f"{summary['result']} parity_fp32: {len(rows)} images, max |logit diff| {worst:.3e} "
          f"(limit {MAX_ABS_DIFF_LIMIT:g}), min argmax agreement {lowest:.6f} (min {ARGMAX_AGREEMENT_MIN:g}), "
          f"bitwise-equal {summary['bitwise_equal_images']}/{len(rows)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
