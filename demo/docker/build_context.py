#!/usr/bin/env python3
"""Assemble the Docker build context for demo/Dockerfile from an allowlist (never the repository root).

    python3 -I demo/docker/build_context.py OUT_DIR [--model PATH] [--e3 PATH] [--shown PATH] [--with-ondevice]

Copies only the files the server imports (TREES below: parts of src/ and configs/, and demo/ without
generated files, tools/ or docker/), skipping every path whose name contains "test" (any case), plus
the registry's model files (demo/config.py; --model, --e3 and --shown override a slot's file): e6 as
demo/models/model.pt, e3 as demo/models/e3.pt and shown as demo/models/shown.pt. A slot is copied only if
its expected sha256 is set and the file matches it; e6 is required, and a mismatch stops the build. The
server is started from the context in a check of this lane to prove the list suffices. --with-ondevice
also copies the exported ONNX files and the vendored ONNX Runtime Web files so the container serves
/ondevice/ too. Writes at the context root: the Dockerfile, CONTEXT_MANIFEST.json (every file with its
sha256, and the slots), demo.env (the registry for this context: each copied slot's path, relative to the
context root, and its expected sha256; `docker run --env-file demo.env`, or `set -a; . ./demo.env` before
run_demo.py) and, last, SHA256SUMS (every other file, in `sha256sum -c` form). Refuses a non-empty OUT_DIR.
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

SLOT_FILES = {"e6": "demo/models/model.pt", "e3": "demo/models/e3.pt", "shown": "demo/models/shown.pt"}


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_files(paths: dict) -> tuple[dict, list[str]]:
    """{context path: (source file, slot, sha256)} for every configured slot, each file checked."""
    try:
        slots = config.resolve_slots(paths={k: v for k, v in paths.items() if v})
    except config.SlotConfigError as e:
        raise SystemExit(f"refused: {e}") from e
    out, notes = {}, []
    for rs in slots:
        key = rs.spec.key
        if not rs.configured:
            if rs.spec.required:
                raise SystemExit(f"refused: slot {key} is required and has no expected sha256 "
                                 f"({rs.spec.sha256_env} or its marked line in demo/config.py)")
            notes.append(f"{key}: not copied (no expected sha256 set)")
            continue
        if rs.path is None or not rs.path.is_file():
            raise SystemExit(f"refused: slot {key}: model file not found: {rs.path}")
        actual = sha256_path(rs.path)
        if actual != rs.expected_sha256:
            raise SystemExit(f"refused: slot {key}: {rs.path} has sha256 {actual}, expected "
                             f"{rs.expected_sha256} ({rs.sha256_source})")
        out[Path(SLOT_FILES[key])] = (rs.path, rs, actual)
        notes.append(f"{key}: {rs.path.name} -> {SLOT_FILES[key]} (sha256 {actual[:12]}...)")
    return out, notes

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
    ap.add_argument("--model", default=None, help="e6's file (default: the registry's)")
    ap.add_argument("--e3", default=None, help="e3's file (default: the registry's)")
    ap.add_argument("--shown", default=None, help="shown's file (default: the registry's)")
    ap.add_argument("--with-ondevice", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    if REPO_ROOT == out or REPO_ROOT in out.parents:
        raise SystemExit("OUT_DIR must be outside the repository")
    slot_files, notes = registry_files({"e6": a.model, "e3": a.e3, "shown": a.shown})
    model = next(src for src, rs, _ in slot_files.values() if rs.spec.key == "e6")
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
    extra = {rel: src for rel, (src, _, _) in slot_files.items()}
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
    for rel, (src, rs, sha) in slot_files.items():
        if sha256_path(out / rel) != sha:
            raise SystemExit(f"refused: the copy of slot {rs.spec.key} at {rel} differs from {src}")
    shutil.copyfile(config.DEMO_DIR / "Dockerfile", out / "Dockerfile")
    manifest = {f: sha256_path(out / f) for f in sorted(files)}
    slots_doc = {rs.spec.key: {"file": str(rel), "source": str(src), "sha256": sha, "label": rs.spec.label,
                               "sha256_source": rs.sha256_source}
                 for rel, (src, rs, sha) in slot_files.items()}
    (out / "CONTEXT_MANIFEST.json").write_text(json.dumps({"model_source": str(model), "slots": slots_doc,
                                                           "files": manifest}, indent=2), encoding="utf-8")
    env_lines = ["# The demo's registry for this context (demo/config.py), written by demo/docker/build_context.py.",
                 "# Paths are relative to the context root. docker run --env-file demo.env ...; or, in the",
                 "# unpacked context: set -a; . ./demo.env; set +a; python3 -I demo/run_demo.py --no-build"]
    for rel, (_, rs, sha) in slot_files.items():
        env_lines += [f"{rs.spec.path_env}={rel.as_posix()}", f"{rs.spec.sha256_env}={sha}"]
    (out / "demo.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    listed = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    sums = "".join(f"{sha256_path(out / f)}  {f}\n" for f in listed)
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    for note in notes:
        print(f"[context] {note}")
    print(f"context: {out} ({len(files)} files copied; {len(listed)} files listed in SHA256SUMS, "
          f"{len(listed) + 1} in all). SHA256SUMS sha256 {sha256_path(out / 'SHA256SUMS')}")
    print(f"Next: docker build -t plantseg-demo {out}; docker run --rm -p 8000:8000 --env-file {out / 'demo.env'} "
          "plantseg-demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
