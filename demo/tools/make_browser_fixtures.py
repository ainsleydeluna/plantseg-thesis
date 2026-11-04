#!/usr/bin/env python3
"""Fixtures for demo/tools/browser_check.cjs: self-made images and what the evaluator makes of them.

    python3 -I demo/tools/make_browser_fixtures.py OUT_DIR [--model INT8.pt]

For each synthetic image (PNG and JPEG, several aspect ratios) writes the file and records the sha256 of
the evaluator's input tensor (core_preprocess + finalize, float32 little-endian CHW bytes), the photo box,
and the sha256 of the cropped argmax mask from the INT8 TorchScript model and from the FP32 PyTorch model.
For the first image it also writes both models' raw float32 logits (the browser compares its own).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402

LOGIT_STRIDE = 4
CASES = (("fx_640x480.png", (640, 480), "PNG"), ("fx_480x640.png", (480, 640), "PNG"),
         ("fx_1023x767.png", (1023, 767), "PNG"), ("fx_512x512.png", (512, 512), "PNG"),
         ("fx_1024x765.png", (1024, 765), "PNG"),      # 765*512/1024 = 382.5: Python rounds half to even
         ("fx_1024x768.jpg", (1024, 768), "JPEG"), ("fx_720x1280.jpg", (720, 1280), "JPEG"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    import torch
    from PIL import Image

    from demo.server.inference import load_model
    from demo.server.preprocess import evaluator_preprocess
    from demo.synthetic import make_leaf_image
    from src.models.student import build_student

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = config.resolve_model_path(a.model)
    int8 = load_model(model_path).module
    fp32 = build_student(pretrained=False).eval()
    fp32.load_state_dict(torch.load(model_path.with_name(model_path.stem + "_fp32_state.pt"),
                                    map_location="cpu"), strict=True)
    rows = []
    for i, (name, size, fmt) in enumerate(CASES):
        img = make_leaf_image(300 + i, size)
        p = out / name
        img.save(p, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
        with Image.open(p) as im:
            pre = evaluator_preprocess(im)
        t, b = pre.tensor, pre.box
        row = {"file": name, "format": fmt, "size": list(size),
               "tensor_sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
               "box": {"top": b[0], "left": b[1], "height": b[2], "width": b[3]}}
        with torch.no_grad():
            for tag, m in (("int8", int8), ("fp32", fp32)):
                y = m(t)
                crop = y.argmax(1)[0].to(torch.uint8).numpy()[b[0]:b[0] + b[2], b[1]:b[1] + b[3]]
                row[f"mask_crop_sha256_{tag}"] = hashlib.sha256(crop.tobytes()).hexdigest()
                row[f"classes_{tag}"] = sorted(int(c) for c in set(crop.ravel().tolist()))
                if i == 0:      # every 4th row and column of all 116 channels (7.6 MB; full is 121 MB)
                    sub = y[0, :, ::LOGIT_STRIDE, ::LOGIT_STRIDE].contiguous().numpy().astype("<f4")
                    (out / f"ref_logits_{tag}.f32").write_bytes(sub.tobytes())
        rows.append(row)
    doc = {"model_file": model_path.name, "model_sha256": common.sha256_file(model_path), "cases": rows,
           "ref_logits_for": CASES[0][0], "ref_logits_stride": LOGIT_STRIDE}
    (out / "expected.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps({"cases": [r["file"] for r in rows], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
