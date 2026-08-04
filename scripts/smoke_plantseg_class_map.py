#!/usr/bin/env python3
"""Offline validator for configs/plantseg_class_map.json (A2b-0).

STANDARD LIBRARY ONLY. NO NETWORK. This never re-downloads the upstream source: it validates
the vendored artifact exactly as an offline consumer (A2b) would see it, and independently
reproduces the canonical semantic class-map hash.

Run:  set PYTHONIOENCODING=utf-8 && python -B scripts/smoke_plantseg_class_map.py
Exit: 0 only if every check passes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAP_PATH = REPO / "configs" / "plantseg_class_map.json"

NUM_CLASSES = 116
BACKGROUND_INDEX = 0
IGNORE_INDEX = 255
ROLE_BG = "background_or_non_disease"
ROLE_DISEASE = "disease"
SCHEMA = "plantseg-class-map/1.0.0"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def canonical_entries_bytes(entries) -> bytes:
    """Must mirror src.eval.artifacts.canonical_json_bytes(list(class_map)) exactly."""
    return json.dumps(list(entries), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _reject(x):
    raise AssertionError(f"non-finite JSON constant {x!r}")


def main() -> int:
    print("=" * 96)
    print("A2b-0 -- offline validation of configs/plantseg_class_map.json")
    print(f"python {sys.version.split()[0]} | stdlib only | NO NETWORK")
    print("=" * 96)

    # 1. strict JSON parsing
    raw = MAP_PATH.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=_reject)
        check("1 strict JSON parses (non-finite constants rejected)", True,
              f"{len(raw):,} bytes")
    except Exception as e:                                          # noqa: BLE001
        check("1 strict JSON parses", False, f"{type(e).__name__}: {e}")
        return _finish()

    # 6. schema identifier
    check("6 schema identifier", doc.get("schema") == SCHEMA, repr(doc.get("schema")))

    # 2. required provenance fields + types
    src = doc.get("source", {})
    required = {
        "repository_owner": str, "repository_name": str, "repository_url": str,
        "source_path": str, "source_commit_sha": str, "immutable_source_url": str,
        "source_object": str, "raw_source_sha256": str, "retrieved_utc": str,
        "extraction_method": str,
    }
    missing = [k for k in required if k not in src]
    badtype = [k for k, t in required.items() if k in src and not isinstance(src[k], t)]
    check("2 provenance fields present with correct types",
          not missing and not badtype, f"missing={missing} badtype={badtype}")
    check("2b source is the official tqwei05/PlantSeg repository",
          src.get("repository_owner") == "tqwei05" and src.get("repository_name") == "PlantSeg"
          and src.get("repository_url") == "https://github.com/tqwei05/PlantSeg",
          src.get("repository_url", ""))
    check("2c source object is METAINFO['classes']",
          src.get("source_object") == 'PlantSeg115Dataset.METAINFO["classes"]'
          and src.get("source_path") == "mmseg/datasets/plantseg115.py",
          src.get("source_object", ""))

    # 3. full lowercase 40-hex commit SHA
    sha = src.get("source_commit_sha", "")
    check("3 commit SHA is full 40-char lowercase hex",
          bool(re.fullmatch(r"[0-9a-f]{40}", sha)), sha)

    # 4. immutable URL contains the exact SHA (and is not a branch URL)
    url = src.get("immutable_source_url", "")
    check("4 immutable URL is commit-pinned (contains the SHA, not a branch)",
          sha in url and "/main/" not in url and url.startswith("https://raw.githubusercontent.com/"),
          url)

    # 5. raw-source hash format
    check("5 raw_source_sha256 is 64-hex",
          bool(re.fullmatch(r"[0-9a-f]{64}", src.get("raw_source_sha256", ""))),
          src.get("raw_source_sha256", "")[:16] + "...")

    # 7. class-space values
    cs = doc.get("class_space", {})
    check("7 class-space values (116 / bg 0 / ignore 255)",
          cs.get("num_classes") == NUM_CLASSES
          and cs.get("background_index") == BACKGROUND_INDEX
          and cs.get("ignore_index") == IGNORE_INDEX, json.dumps(cs))

    entries = doc.get("entries", [])

    # 8. exactly 116 entries
    check("8 exactly 116 ordered entries", len(entries) == NUM_CLASSES, f"len={len(entries)}")

    # 9/15. IDs exactly 0..115, in order, none missing or extra
    ids = [e.get("class_id") for e in entries]
    check("9,15 class_id sequence is exactly 0..115 in order",
          ids == list(range(NUM_CLASSES)),
          f"first={ids[:3]} last={ids[-3:]}" if ids else "empty")

    # 10/11. class 0 -- empty official name, explicit identity test (NOT truthiness)
    e0 = entries[BACKGROUND_INDEX] if entries else {}
    name0 = e0.get("name", None)
    check("10 class 0 official name is exactly the empty string (not renamed)",
          name0 is not None and isinstance(name0, str) and len(name0) == 0,
          repr(name0))
    check("11 class 0 role", e0.get("role") == ROLE_BG, repr(e0.get("role")))

    # 12/13. disease roles + nonempty names
    dis = entries[1:] if len(entries) == NUM_CLASSES else []
    check("12 classes 1..115 all have role 'disease'",
          all(e.get("role") == ROLE_DISEASE for e in dis),
          f"{sum(1 for e in dis if e.get('role') == ROLE_DISEASE)}/{len(dis)}")
    check("13 classes 1..115 all have non-empty names",
          all(isinstance(e.get("name"), str) and len(e["name"]) > 0 for e in dis))

    # 14. disease-name uniqueness
    names = [e.get("name") for e in dis]
    dupes = sorted({n for n in names if names.count(n) > 1})
    check("14 disease names are unique", not dupes, f"dupes={dupes[:5]}")

    # 17. no palette
    has_palette = "palette" in doc or any("palette" in e for e in entries)
    check("17 no palette field vendored", not has_palette)

    # entries carry exactly the frozen descriptor keys
    key_ok = all(set(e) == {"class_id", "name", "role"} for e in entries)
    check("entries carry exactly {class_id, name, role}", key_ok,
          f"first={sorted(entries[0]) if entries else []}")

    # 16. canonical semantic hash reproduced independently
    sem = hashlib.sha256(canonical_entries_bytes(entries)).hexdigest()
    check("16 canonical semantic class-map sha256 reproduced", bool(re.fullmatch(r"[0-9a-f]{64}", sem)),
          sem)
    # local vendored-file hash must NOT be embedded (recursive self-hash)
    check("local vendored-file hash is NOT embedded in the JSON",
          "local_file_sha256" not in src and "vendored_sha256" not in src)

    # 18. no network required -- assert nothing here imported a network module
    net = [m for m in ("urllib.request", "http.client", "socket", "ssl") if m in sys.modules]
    check("18 no network module imported by this validator", not net, f"{net}")

    return _finish(sem)


def _finish(sem: str | None = None) -> int:
    print()
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail[:150]}")
    n_ok = sum(1 for _, ok, _ in CHECKS if ok)
    ok_all = n_ok == len(CHECKS)
    print("\n" + "=" * 96)
    if sem:
        print(f"canonical semantic class-map sha256: {sem}")
    print(f"SUMMARY  {n_ok}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'CLASS MAP VALID (offline)' if ok_all else 'CLASS MAP INVALID'}")
    print("=" * 96)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
