#!/usr/bin/env python3
"""Fetch the pinned ONNX Runtime Web files for the in-browser page into demo/ondevice/vendor/ (once,
with internet), so the phone page later runs from the laptop with no internet at the venue.

    python3 -I demo/ondevice/fetch_ort_web.py

Downloads onnxruntime-web@VERSION's npm tarball, verifies it against the registry's sha512 integrity,
and extracts only FILES (about 33 MB) from package/dist/. Nothing is executed. vendor/ is gitignored.
Without vendor/, the page falls back to the same files on cdn.jsdelivr.net (needs internet).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.20.1"          # = the onnxruntime Python package in demo/requirements-demo.txt
FILES = ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm",
         "ort.webgpu.min.js", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm")
REGISTRY = "https://registry.npmjs.org/onnxruntime-web/" + VERSION
VENDOR = Path(__file__).resolve().parent / "vendor"


def main() -> int:
    with urllib.request.urlopen(REGISTRY, timeout=60) as r:
        meta = json.loads(r.read())
    tarball, integrity = meta["dist"]["tarball"], meta["dist"]["integrity"]
    algo, b64 = integrity.split("-", 1)
    if algo != "sha512":
        raise SystemExit(f"unexpected integrity algorithm {algo}")
    with urllib.request.urlopen(tarball, timeout=300) as r:
        blob = r.read()
    if base64.b64encode(hashlib.sha512(blob).digest()).decode() != b64:
        raise SystemExit("tarball does not match the registry's sha512 integrity; nothing written")
    VENDOR.mkdir(parents=True, exist_ok=True)
    written = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for name in FILES:
            member = tf.getmember("package/dist/" + name)
            if not member.isfile():
                raise SystemExit(f"{name} is not a regular file in the tarball")
            data = tf.extractfile(member).read()
            (VENDOR / name).write_bytes(data)          # flat names only: no path from the archive
            written[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"package": "onnxruntime-web", "version": VERSION, "tarball": tarball,
                "integrity": integrity, "files": written}
    (VENDOR / "VERSION.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
