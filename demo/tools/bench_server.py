#!/usr/bin/env python3
"""What the demo server needs from a host: memory and per-request CPU time (Linux; synthetic images).

    python3 -I demo/tools/bench_server.py [--model PATH] [--threads 1 2 4] [--runs 10]

Each thread count runs in a fresh process (torch's thread pool is per process): resident memory after the
imports, after the model loads, and at its peak; then `--runs` requests through Segmenter.segment (the
code /api/segment runs, without HTTP) for a 1024x768 JPEG and a 4032x3024 JPEG (a 12 MP phone photo).
Prints one JSON document. Numbers describe this machine's CPU only.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def rss_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def peak_mb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def child(model: str, threads: int, runs: int) -> dict:
    out = {"threads": threads}
    import torch  # noqa: F401

    from demo import config
    from demo.server.inference import Segmenter
    from demo.synthetic import make_leaf_image

    out["rss_after_imports_mb"] = round(rss_mb(), 1)
    seg = Segmenter(config.resolve_model_path(model), threads)
    out["rss_after_model_load_mb"] = round(rss_mb(), 1)
    for name, size in (("1024x768", (1024, 768)), ("4032x3024", (4032, 3024))):
        buf = io.BytesIO()
        make_leaf_image(9, size).save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        seg.segment(data)                                   # warm-up
        inf, tot = [], []
        for _ in range(runs):
            r = seg.segment(data)
            inf.append(r["latency_ms"]["inference"])
            tot.append(r["latency_ms"]["total"])
        out[name] = {"upload_bytes": len(data), "inference_ms_median": round(statistics.median(inf), 1),
                     "inference_ms_max": round(max(inf), 1), "total_ms_median": round(statistics.median(tot), 1),
                     "total_ms_max": round(max(tot), 1)}
    out["rss_end_mb"] = round(rss_mb(), 1)
    out["peak_rss_mb"] = round(peak_mb(), 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child is not None:
        print(json.dumps(child(a.model, a.child, a.runs)))
        return 0
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo", encoding="ascii")
                if ln.startswith("model name")), platform.processor())
    rows = []
    for t in a.threads:
        cmd = [sys.executable, "-I", __file__, "--child", str(t), "--runs", str(a.runs)]
        if a.model:
            cmd += ["--model", a.model]
        rows.append(json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
                               .splitlines()[-1]))
    import os

    print(json.dumps({"cpu": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                      "runs_per_image": a.runs, "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
