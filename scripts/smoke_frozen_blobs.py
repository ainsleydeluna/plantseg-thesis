#!/usr/bin/env python3
"""Frozen-file check: src/eval/metrics.py and the nine frozen teacher-runtime files are byte-for-byte
the committed freeze (docs/lane_specs/part1.md cross-lane rules; errata E-5; STOP on any diff).

The git blob id is computed in Python, as errata E-5 prescribes (the protected-file hook refuses
`git hash-object`): sha1 over the ASCII bytes "blob ", the file size in decimal, one NUL byte, then
the file's bytes. `.gitattributes` is `* text=auto eol=lf`, so a checkout's bytes equal the committed
blob bytes on every platform. The expected ids are the Git blob column of
docs/teacher_prep_runbook.md §4a step 8 (B62 commit-based freeze at 3c43f89); metrics.py's
cbd5fa86... is also the id the lane specs pin (metric_impl_sha256 9898d6dc...). Reads only these
nine files.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FROZEN = {
    "src/training/teacher_runner.py": "7ceb85eb5a621ce0cc6ed0f1a357fd60f23e99a7",
    "scripts/launch_teacher_finetune.py": "98509d2351bd475064edca9fecf08aabd80cd363",
    "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py":
        "3c6a7b28ebad5fce6187c307f87fd7fc91f81933",
    "src/training/teacher_components.py": "33a2b6980e93c615483236522d1294916ef3441b",
    "src/data/transforms.py": "b68c1d25cbcedab22f89d45faac936471bdba98e",
    "configs/augment.py": "29061da522bd099c6e513bd6e7bede1487ad6238",
    "src/eval/metrics.py": "cbd5fa86db928529299a7dfff7c845137bda7e82",
    "src/distill/nmf_stream.py": "f0a1d4fcda085e9a0e8bc769158b4baf89914a8e",
    "src/data/isolation.py": "2f816dd7543779615402d0c03b950e55830805df",
}
METRICS_SHA256 = "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9"


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    results = []
    for rel, want in FROZEN.items():
        p = REPO / rel
        got = git_blob_id(p.read_bytes()) if p.is_file() else "<missing>"
        results.append((rel, got == want, f"{got} (expected {want})"))
    metrics = (REPO / "src/eval/metrics.py").read_bytes()
    got = hashlib.sha256(metrics).hexdigest()
    results.append(("src/eval/metrics.py sha256 (metric_impl_sha256)", got == METRICS_SHA256, got))
    print("FROZEN BLOBS — metrics.py and the nine frozen teacher-runtime files (runbook §4a)")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {'' if ok else detail}")
    passed = sum(ok for _, ok, _ in results)
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
