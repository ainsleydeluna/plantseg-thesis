#!/usr/bin/env python3
"""Run the mobile web demo with one command (research prototype, CPU only).

    Linux/macOS:  python3 -I demo/run_demo.py
    Windows:      py -3.11 -I demo\\run_demo.py

Then open http://localhost:8000 on this computer, or the printed http://<this-computer's-IP>:8000 on
a phone on the same network. If e6's model file (demo/config.py MODEL_PATH) is the default stand-in and
does not exist yet, it is built first (about 15 s; random weights, synthetic calibration only).

The models come from the registry in demo/config.py (slots e6, e3, shown). A slot is served only if its
expected sha256 is set and the file matches it; e6 is required. At start-up each loaded file's name,
sha256, engine and stage are printed; a refusal is printed and the exit code is 2.

Options: --host 127.0.0.1 keeps the server local-only; --port; --threads (CPU threads for inference);
--model PATH overrides e6's file for this run; --no-build never builds the stand-in.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import config  # noqa: E402


def lan_addresses() -> list[str]:
    """This machine's local IPv4 addresses, as far as they can be found (no packet is sent)."""
    out = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 9))           # TEST-NET-1: routing lookup only
        out.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            out.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in out if not a.startswith("127."))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PlantSeg mobile web demo (research prototype)")
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--model", default=None, help="e6's model file (default: demo/config.py MODEL_PATH)")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args(argv)

    try:
        slots = config.resolve_slots(paths={"e6": a.model} if a.model else None)
    except config.SlotConfigError as e:
        print(f"[demo] refused to start: {e}", file=sys.stderr)
        return 2
    model = slots[0].path
    if model is not None and not model.is_file() and model == config.STAND_IN_PATH and not a.no_build:
        print(f"[demo] {model.name} not found: building the stand-in (random weights) ...", flush=True)
        subprocess.run([sys.executable, "-I", str(config.DEMO_DIR / "model" / "build_stand_in.py")],
                       check=True)

    import uvicorn

    from demo.server.app import create_app
    try:
        app = create_app(threads=a.threads, slots=slots)
    except (RuntimeError, FileNotFoundError) as e:     # SlotRefused, the evaluator's CheckpointError, a wrong kind
        print(f"[demo] refused to start: {e}", file=sys.stderr, flush=True)
        if str(e).startswith("slot e6:") and model is not None and model.is_file() and model == config.STAND_IN_PATH:
            from demo import common

            print(f"[demo] to serve the stand-in (random weights) instead, set "
                  f"PLANTSEG_DEMO_E6_SHA256={common.sha256_file(model)}", file=sys.stderr, flush=True)
        return 2
    seg = app.state.segmenter
    print(f"[demo] PHONE-FIX-1: mkldnn enabled={seg.runtime['mkldnn_enabled']}, quantized engine "
          f"{seg.runtime['quantized_engine']}", flush=True)
    for info in seg.models_info():
        print(f"[demo] loaded {info['slot']}: {info['file']} sha256 {info['sha256']} engine {info['engine']} "
              f"stage {info['stage']}{' (STAND-IN, random weights)' if info['stand_in'] else ''}", flush=True)
    for note in seg.notes:
        print(f"[demo] {note}", flush=True)
    print(f"[demo] open http://localhost:{a.port}/ on this computer")
    if a.host in ("0.0.0.0", "::"):
        for ip in lan_addresses():
            print(f"[demo] on a phone on the same network: http://{ip}:{a.port}/")
    print("[demo] research prototype; not for agronomic decisions. Ctrl+C stops the server.", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
