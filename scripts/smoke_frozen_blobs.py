#!/usr/bin/env python3
"""Frozen-blob smoke (docs/lane_specs/part1.md cross-lane rule "Frozen"; errata E-5; STOP on any diff).

`src/eval/metrics.py` is frozen at git blob cbd5fa86db928529299a7dfff7c845137bda7e82, recorded in every
evaluation artifact as `metric_impl_sha256` 9898d6dc... (both full values: docs/teacher_prep_runbook.md
:321). No lane computes a metric by editing it; every lane's test run includes this smoke. The other
eight files are the frozen teacher-runtime files; with metrics.py they are the nine files of the Git
blob column of docs/teacher_prep_runbook.md §4a step 8 (B62 commit-based freeze at 3c43f89).

Lanes S1 (lane/s1-am5-gtpresent) and K1 (lane/k1-switches-alpha) each added this script; the merge
combines both check sets: the git blob id of each of the nine files, plus the sha256 of metrics.py.

The protected-file hook refuses `git hash-object`, so the blob id is computed here in Python exactly as
git defines it: sha1 over the ASCII bytes "blob ", the file size in decimal, one NUL byte, then the
file's bytes (.gitattributes pins eol=lf, so the working-tree bytes are the blob bytes on every
platform). No git command runs, nothing is written, and no file outside FROZEN is read.

Run:  python -B scripts/smoke_frozen_blobs.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: repo-relative path -> (git blob id, sha256 of the raw bytes or None when only the blob id is pinned)
FROZEN = {
    "src/eval/metrics.py": ("cbd5fa86db928529299a7dfff7c845137bda7e82",
                            "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9"),
    "src/training/teacher_runner.py": ("7ceb85eb5a621ce0cc6ed0f1a357fd60f23e99a7", None),
    "scripts/launch_teacher_finetune.py": ("98509d2351bd475064edca9fecf08aabd80cd363", None),
    "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py":
        ("3c6a7b28ebad5fce6187c307f87fd7fc91f81933", None),
    "src/training/teacher_components.py": ("33a2b6980e93c615483236522d1294916ef3441b", None),
    "src/data/transforms.py": ("b68c1d25cbcedab22f89d45faac936471bdba98e", None),
    "configs/augment.py": ("29061da522bd099c6e513bd6e7bede1487ad6238", None),
    "src/distill/nmf_stream.py": ("f0a1d4fcda085e9a0e8bc769158b4baf89914a8e", None),
    "src/data/isolation.py": ("2f816dd7543779615402d0c03b950e55830805df", None),
}

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def git_blob_id(data: bytes) -> str:
    """The git object id of a blob with these bytes (what `git hash-object` prints for the file)."""
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\x00" + data).hexdigest()


def main() -> int:
    for rel, (blob, sha) in FROZEN.items():
        path = REPO / rel
        if not path.is_file():
            check(f"{rel} exists", False, f"missing: {path}")
            continue
        data = path.read_bytes()
        got_blob = git_blob_id(data)
        hint = " (the file holds CR bytes: not an eol=lf checkout)" if b"\r" in data else ""
        check(f"{rel} git blob id == {blob[:8]}...", got_blob == blob, f"computed {got_blob}{hint}")
        if sha is not None:
            got_sha = hashlib.sha256(data).hexdigest()
            check(f"{rel} sha256 == {sha[:8]}... (metric_impl_sha256)", got_sha == sha,
                  f"computed {got_sha}")

    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'FROZEN BLOBS OK' if allok else 'FROZEN BLOB CHANGED -- STOP'} ({good}/{len(CHECKS)})")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
