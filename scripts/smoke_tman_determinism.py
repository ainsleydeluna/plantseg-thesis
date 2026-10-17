#!/usr/bin/env python3
"""Smoke for the builder's determinism (plan §5's smoke_tman_determinism; lane TMAN, B-1, C5): the manifest, the
inventory export and the entry files are byte-identical across two fixture worlds, two builds in one world and the
command line's build; across a shuffled directory listing (the teacher's log folders are listed once, and the M12
log is chosen in sorted order), across the process time zone (every instant is UTC, the Manila day a fixed +08:00
offset) and across the key order of the sources and of the objects hashed canonically.

Synthetic inputs only, under the system temp dir (removed at the end); TEST is never read.

Run:  python -B scripts/smoke_tman_determinism.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_pins as PINS  # noqa: E402
import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)
import scripts.smoke_tman_validate as VW  # noqa: E402  (the fixture world)

check = FX.check
ZONES = ("Asia/Manila", "America/Los_Angeles", "Pacific/Kiritimati")
WORLDS: list = []


def build(w) -> tuple[bytes, bytes]:
    """(manifest bytes, inventory bytes) built from the world's records, sources and run directories."""
    from src.tman import manifest as M
    S = w.S
    doc = M.build_manifest(w.records, w.cat, w.sources, runs_root=w.runs)
    data = S.json_file_bytes(doc)
    inv = M.build_inventory(doc, data, w.records, w.cat, runs_root=w.runs, registered=PINS.IF6_CELLS,
                            freeze_dl="DL-300")
    return data, S.json_file_bytes(inv)


def shuffled(o, rng: random.Random):
    """`o` with every object's keys in a random order (lists keep theirs)."""
    if isinstance(o, dict):
        items = list(o.items())
        rng.shuffle(items)
        return {k: shuffled(v, rng) for k, v in items}
    if isinstance(o, list):
        return [shuffled(v, rng) for v in o]
    return o


# ======================================================================================================= cases
def case_two_worlds():
    w1, w2 = WORLDS
    check("two_worlds_build_byte_identical_manifests", w1.data == w2.data)
    b1, b2 = build(w1), build(w2)
    check("two_worlds_build_byte_identical_inventories", b1[1] == b2[1])
    check("a_rebuild_in_one_world_is_byte_identical", build(w1) == b1 and b1[0] == w1.data)
    e1 = {k: v["sha256"] for k, v in w1.cut_refs.items()}
    e2 = {k: v["sha256"] for k, v in w2.cut_refs.items()}
    check("entry_files_byte_identical_across_worlds", e1 == e2 and len(e1) == 22, str(len(e1)))


def case_cli_build():
    import scripts.tday_manifest as CLI
    w = WORLDS[0]
    S = w.S
    out = FX.scratch_dir("tmandet")
    (out / "sources.json").write_bytes(S.json_file_bytes(w.sources))
    (out / "cells.json").write_text(json.dumps(PINS.IF6_CELLS), encoding="utf-8")
    got = []
    for n in (1, 2):
        d = out / f"build{n}"
        d.mkdir()
        rc = CLI.main(["--root", str(w.repo), "build", "--sources", str(out / "sources.json"), "--out", str(d),
                       "--runs", str(w.runs), "--registered-cells", str(out / "cells.json")])
        got.append((rc, (d / "tday_manifest.json").read_bytes()))
    check("cli_build_twice_byte_identical_and_the_builder_s", got[0] == got[1] == (0, w.data), str([g[0] for g in got]))


def case_listing_order():
    """The teacher's work dir gains two more log folders, one earlier without an M12 line and one later with
    another; the earliest log with the final line is chosen, whatever order the listing returns."""
    w = WORLDS[0]
    work = w.runs / "teacher_s42_40k"
    for folder, line in (("20261011_000000", None), ("20261014_000000", "10/14 09:00:00")):
        FX.teacher_run(w.runs, "teacher_s42_40k", final=line, folder=folder)
    base = build(w)
    check("extra_log_folders_change_nothing", base[0] == w.data)
    real = Path.iterdir
    rng = random.Random(7)
    seen = []
    try:
        for _ in range(4):
            def iterdir(self, _real=real):
                items = list(_real(self))
                rng.shuffle(items)
                seen.append([p.name for p in items] if self == work else None)
                return iter(items)
            Path.iterdir = iterdir
            ok = build(w) == base
            check(f"shuffled_listing_{len(seen)}_byte_identical", ok)
    finally:
        Path.iterdir = real
    orders = {tuple(s) for s in seen if s}
    check("the_listing_was_shuffled", len(orders) > 1, str(orders))


def case_time_zone():
    from src.tman import entries as E
    w = WORLDS[1]
    base, verdict = build(w), w.run("freeze")[:2]
    entry = w.S.json_file_bytes(E.make_entry("not_made", ["E7-E2:s42:list42"], "2026-11-03",
                                             {"reason": "code not merged", "lane": "L-AM23"}, amendment="AM-23",
                                             script="scripts/tday_manifest.py", commit=w.E))
    saved = os.environ.get("TZ")
    try:
        for zone in ZONES:
            os.environ["TZ"] = zone
            time.tzset()
            got = w.S.json_file_bytes(E.make_entry("not_made", ["E7-E2:s42:list42"], "2026-11-03",
                                                   {"reason": "code not merged", "lane": "L-AM23"},
                                                   amendment="AM-23", script="scripts/tday_manifest.py", commit=w.E))
            check(f"tz_{zone.replace('/', '_')}_builds_verdict_and_entry_identical", build(w) == base
                  and w.run("freeze")[:2] == verdict and got == entry and time.localtime(0).tm_gmtoff != 0)
    finally:
        if saved is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved
        time.tzset()


def case_key_order():
    from src.eval import artifacts
    w = WORLDS[1]
    base = build(w)
    saved = w.sources
    rng = random.Random(11)
    try:
        w.sources = shuffled(saved, rng)
        same = build(w) == base
    finally:
        w.sources = saved
    check("sources_key_order_shuffled_byte_identical", same)
    doc = json.loads(w.data)
    check("canonical_sha256_key_order_independent", w.S.canonical_sha256(shuffled(doc["body"], rng))
          == w.S.canonical_sha256(doc["body"]))
    check("canonical_json_is_the_one_definition", w.S.canonical_json_bytes is artifacts.canonical_json_bytes)


def main() -> int:
    try:
        WORLDS.extend([VW.World(), VW.World()])
    except Exception as e:  # noqa: BLE001
        check("worlds_built", False, f"{type(e).__name__}: {e}")
        return FX.run_cases("TMAN DETERMINISM SMOKE (lane TMAN, B-1, C5)", ())
    try:
        return FX.run_cases("TMAN DETERMINISM SMOKE (lane TMAN, B-1, C5)",
                            (case_two_worlds, case_cli_build, case_listing_order, case_time_zone, case_key_order))
    finally:
        if VW.REAL_TEACHER is not None:
            WORLDS[0].M.TEACHER_OF_RECORD.clear()
            WORLDS[0].M.TEACHER_OF_RECORD.update(VW.REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(main())
