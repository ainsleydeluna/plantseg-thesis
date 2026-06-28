#!/usr/bin/env python3
"""B14 — write a Week-1 reproducibility / environment manifest to reports/run_manifest.json.

Records the ACTUAL local environment + git state. NOT a training run, and NOT the pinned
training/RunPod environment. Safe: records only file PATHS from git status (never contents); no
dataset contents, weights, checkpoints, secrets/tokens, or home-directory scans.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "run_manifest.json"


def _run(cmd: list[str]):
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
        # rstrip only: never strip leading whitespace (git --porcelain uses a leading status column)
        return r.stdout.rstrip("\n") if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _git_info() -> dict:
    commit = _run(["git", "rev-parse", "HEAD"])
    short = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    modified: list[str] = []
    untracked: list[str] = []
    if porcelain is not None:
        for line in porcelain.splitlines():
            if not line:
                continue
            status, path = line[:2], line[3:]
            (untracked if status == "??" else modified).append(path)
    return {
        "commit": commit or "NEED_TO_CONFIRM",
        "short_commit": short or "NEED_TO_CONFIRM",
        "branch": branch or "NEED_TO_CONFIRM",
        "dirty": bool(porcelain) if porcelain is not None else "NEED_TO_CONFIRM",
        "modified_paths": modified,          # paths only, never contents
        "untracked_paths": untracked,        # paths only, never contents
    }


def _torch_info() -> dict:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"version": f"NEED_TO_CONFIRM (import failed: {type(e).__name__})"}
    cuda = bool(torch.cuda.is_available())
    return {
        "version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,                    # None on cpu builds
        "gpu_name": (torch.cuda.get_device_name(0) if cuda else None),
    }


def _pip_freeze() -> list[str]:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return out.splitlines() if out is not None else ["NEED_TO_CONFIRM (pip freeze failed)"]


def _seed() -> int:
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from src.seeds import SEED
        return int(SEED)
    except Exception:  # noqa: BLE001
        return 42


def build_manifest() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "torch": _torch_info(),
        "seed": _seed(),
        "pip_freeze": _pip_freeze(),
        "cwd": os.getcwd(),
        "script_path": str(Path(__file__).resolve()),
        "notes": [
            "Week-1 LOCAL reproducibility/environment manifest — NOT a training run.",
            "This is NOT the pinned training/RunPod environment.",
            "Local dev env (Python 3.13 / torch 2.9.1+cpu / CUDA False) differs from the pinned "
            "training stack (Python 3.11 / torch 2.1.0+cu121); the pinned environment and the "
            "deployment GPU are out of scope for this local manifest.",
            "gpu_name is null because CUDA is unavailable on this machine.",
            "git status here records file PATHS only (no contents); no dataset/weights/secrets collected.",
        ],
    }


def main() -> int:
    manifest = build_manifest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    g, t = manifest["git"], manifest["torch"]
    print(f"[manifest] written -> {OUT}")
    print(f"  timestamp : {manifest['timestamp_utc']}")
    print(f"  git       : {g['short_commit']} on {g['branch']} | dirty={g['dirty']} "
          f"(modified={len(g['modified_paths'])}, untracked={len(g['untracked_paths'])})")
    print(f"  python    : {manifest['python_version']} | platform: {manifest['platform']['system']} "
          f"{manifest['platform']['machine']}")
    print(f"  torch     : {t.get('version')} | cuda={t.get('cuda_available')} "
          f"cuda_version={t.get('cuda_version')} gpu={t.get('gpu_name')}")
    print(f"  seed      : {manifest['seed']} | pip_freeze entries: {len(manifest['pip_freeze'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
