#!/usr/bin/env python3
"""Frozen-blob smoke (docs/lane_specs/part1.md cross-lane rule "Frozen"; errata E-5).

`src/eval/metrics.py` is frozen at git blob cbd5fa86db928529299a7dfff7c845137bda7e82, recorded in every
evaluation artifact as `metric_impl_sha256` 9898d6dc... (both full values: docs/teacher_prep_runbook.md
:321). No lane computes a metric by editing it; every lane's test run includes this smoke.

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

#: repo-relative path -> (git blob id, sha256 of the raw bytes)
FROZEN = {
    "src/eval/metrics.py": ("cbd5fa86db928529299a7dfff7c845137bda7e82",
                            "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9"),
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
        got_blob, got_sha = git_blob_id(data), hashlib.sha256(data).hexdigest()
        hint = " (the file holds CR bytes: not an eol=lf checkout)" if b"\r" in data else ""
        check(f"{rel} git blob id == {blob[:8]}...", got_blob == blob, f"computed {got_blob}{hint}")
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
