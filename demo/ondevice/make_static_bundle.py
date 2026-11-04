#!/usr/bin/env python3
"""Write a self-contained static bundle of the in-browser page: no PyTorch and no demo server needed to
serve it, so a Windows laptop (whose official torch 2.1 wheels lack QNNPACK) can still serve the
on-device demo, and the phone does the computing.

    python3 -I demo/ondevice/make_static_bundle.py OUT_DIR      (after export_onnx.py and fetch_ort_web.py)
    then, on any machine with Python:  python -m http.server 8000 --directory OUT_DIR

OUT_DIR/ holds index.html, vendor/ (ONNX Runtime Web), models/*.onnx, manifest.json and info.json (the
repository's class list and the notices, in the /api/info fields the page reads). Refuses a non-empty
OUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=None, help="the INT8 TorchScript the ONNX files came from")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    vendor = HERE / "vendor"
    if not (vendor / "VERSION.json").is_file():
        raise SystemExit("run python3 -I demo/ondevice/fetch_ort_web.py first")
    from demo.server.app import NOTICE_PROTOTYPE, licence_notice
    from demo.server.inference import read_embedded_meta

    src = config.resolve_model_path(a.model)
    meta = read_embedded_meta(src)
    (out / "models").mkdir(parents=True, exist_ok=True)
    models = {}
    for p in (config.ONNX_INT8_PATH, config.ONNX_FP32_PATH):
        if p.is_file():
            shutil.copyfile(p, out / "models" / p.name)
            models[p.name] = {"size_bytes": p.stat().st_size, "sha256": common.sha256_file(p)}
    if not models:
        raise SystemExit("run python3 -I demo/ondevice/export_onnx.py first")
    shutil.copytree(vendor, out / "vendor")
    shutil.copyfile(HERE / "index.html", out / "index.html")
    ort = json.loads((vendor / "VERSION.json").read_text(encoding="utf-8"))
    src_info = {"file": src.name, "sha256": common.sha256_file(src)}
    (out / "manifest.json").write_text(json.dumps(
        {"models": models, "ort_web_vendored": True, "ort_web_version": ort["version"],
         "source_model": src_info}, indent=2), encoding="utf-8")
    classes, prov = common.load_classes()
    data = common.data_facts()
    info = {"model": {**src_info, "stand_in": bool(meta.get("stand_in", False)), "embedded_meta": meta},
            "classes": [{"class_id": c.class_id, "name": c.name, "label": c.label, "role": c.role,
                         "color": "#%02x%02x%02x" % c.color} for c in classes],
            "class_map": prov, "data": data,
            "notices": {"prototype": NOTICE_PROTOTYPE, "licence": licence_notice(data)}}
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    common.assert_no_stats()
    print(f"static bundle in {out}: {sorted(models)}; serve with: python -m http.server 8000 --directory {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
