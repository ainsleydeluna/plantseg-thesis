#!/usr/bin/env python3
"""Assemble the Docker build context for demo/Dockerfile from an allowlist (never the repository root).

    python3 -I demo/docker/build_context.py OUT_DIR [--model PATH] [--with-ondevice]

Copies only the files the server imports (TREES below: parts of src/ and configs/, and demo/ without
generated files, tools/ or docker/), skipping every path whose name contains "test" (any case), plus
the model file (default: demo/config.py MODEL_PATH) as demo/models/model.pt; the server is started from
the context in a smoke of this lane to prove the list suffices. --with-ondevice also copies the exported ONNX files and the
vendored ONNX Runtime Web files so the container serves /ondevice/ too. Writes the Dockerfile at the
context root and CONTEXT_MANIFEST.json (every file with its sha256). Refuses a non-empty OUT_DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import config  # noqa: E402

# Exactly what the server imports (demo/server -> src.eval.adapters/model_loading -> src.data, src.models,
# configs): no training, distillation, statistics or TEST-day files enter an image that might be hosted.
TREES = ("src/__init__.py", "src/seeds.py", "src/data", "src/models", "src/eval/__init__.py",
         "src/eval/evaluate.py", "src/eval/metrics.py", "src/eval/adapters.py", "src/eval/protocols.py",
         "src/eval/model_loading.py", "configs/data.py", "configs/model.py", "configs/augment.py",
         "configs/plantseg_class_map.json", "demo")
SKIP_DIRS = {"__pycache__", "models", "vendor", "screenshots", "tools", "docker"}   # demo/ extras
SKIP_TREES = {Path("src") / "stats"}                                # never imported by the demo


def skipped(rel: Path) -> bool:
    if any("test" in part.lower() for part in rel.parts):
        return True
    if any(rel == t or t in rel.parents for t in SKIP_TREES):
        return True
    return rel.parts[0] == "demo" and any(part in SKIP_DIRS for part in rel.parts[1:-1] + (rel.parts[-1],))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--with-ondevice", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    if REPO_ROOT == out or REPO_ROOT in out.parents:
        raise SystemExit("OUT_DIR must be outside the repository")
    model = config.resolve_model_path(a.model)
    if not model.is_file():
        raise SystemExit(f"model file not found: {model}")
    # Tracked files only, listed by git with the test-named paths and src/stats excluded up front, so
    # their names are never listed and generated or untracked files never enter the context.
    listing = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", *TREES,
                              ":(exclude,icase)*test*", ":(exclude)src/stats"],
                             check=True, capture_output=True).stdout.decode("utf-8")
    files = []
    for name in sorted(n for n in listing.split("\0") if n):
        rel = Path(name)
        if skipped(rel) or rel.suffix == ".pyc":
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, dst)
        files.append(rel.as_posix())
    extra = {Path("demo/models/model.pt"): model}
    if a.with_ondevice:
        for p in (config.ONNX_FP32_PATH, config.ONNX_INT8_PATH):
            if p.is_file():
                extra[Path("demo/models") / p.name] = p
        vendor = config.DEMO_DIR / "ondevice" / "vendor"
        for p in sorted(vendor.glob("*")) if vendor.is_dir() else []:
            extra[Path("demo/ondevice/vendor") / p.name] = p
    for rel, src in extra.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / rel)
        files.append(rel.as_posix())
    shutil.copyfile(config.DEMO_DIR / "Dockerfile", out / "Dockerfile")
    manifest = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in sorted(files)}
    (out / "CONTEXT_MANIFEST.json").write_text(json.dumps({"model_source": str(model), "files": manifest},
                                                          indent=2), encoding="utf-8")
    print(f"context: {out} ({len(files)} files). Next: docker build -t plantseg-demo {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
