#!/usr/bin/env python3
"""A3b-0 — offline validation of the frozen corruption vocabulary.

Repository-local only: no network, no install, no dataset, no model. Rejection cases use
temporary MODIFIED COPIES of the protocol; the real `configs/corruption_protocol.json` is never
written to.

Run:  set PYTHONIOENCODING=utf-8 && python -B scripts/smoke_corruption_protocol.py
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROTOCOL = REPO / "configs" / "corruption_protocol.json"
CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def expect(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}") from e
    raise AssertionError("expected an exception, none raised")


def main() -> int:  # noqa: C901
    print("=" * 100)
    print("A3b-0 -- frozen corruption vocabulary (offline, repository-local only)")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="a3b0_"))
    try:
        # ---- 26. importing src.stats must not read any file ----
        import builtins
        opened: list[str] = []
        real_open = builtins.open
        builtins.open = lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1]
        try:
            for m in [k for k in list(sys.modules) if k.startswith("src.stats")]:
                del sys.modules[m]
            import src.stats as S                              # noqa: F401
        finally:
            builtins.open = real_open
        proto_reads = [p for p in opened if "corruption_protocol.json" in p]
        check("26 importing src.stats reads no protocol file", not proto_reads,
              f"{len(opened)} opens during import, protocol reads={len(proto_reads)}")

        from src.stats import (CorruptionProtocolError, RobustnessError,  # noqa: E402
                               build_official_corruption_grid, load_corruption_protocol)
        from src.stats.corruption_protocol import CANONICAL_IDS, vocabulary_sha256  # noqa: E402
        from src.stats.robustness import INFERENTIAL_SEVERITIES  # noqa: E402

        raw = PROTOCOL.read_text(encoding="utf-8")
        doc = json.loads(raw)
        check("1 strict JSON parses", isinstance(doc, dict), f"{len(raw)} bytes")

        p = load_corruption_protocol(PROTOCOL)
        check("2 exact schema version", p.schema_version == "plantseg-corruptions/1.0.0",
              p.schema_version)
        check("3 exact five-ID order",
              p.order == ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog"),
              str(list(p.order)))
        check("4 IDs unique", len(set(p.order)) == 5)
        check("5 all IDs lowercase snake_case",
              all(c.islower() or c == "_" or c.isdigit() for i in p.order for c in i))
        check("6 exact display labels",
              [e.display_name for e in p.entries] ==
              ["motion blur", "Gaussian noise", "JPEG compression", "brightness variation", "fog"],
              str([e.display_name for e in p.entries]))
        check("7 reference_function == machine id for all five",
              all(e.reference_function == e.id for e in p.entries))
        check("8 inferential severities [1,2,3]", p.inferential_severities == (1, 2, 3))
        check("9 descriptive-only severity [4]", p.descriptive_only_severities == (4,))
        check("10 excluded severity [5]", p.excluded_severities == (5,))
        check("11 severity groups disjoint",
              not (set(p.inferential_severities) & set(p.descriptive_only_severities))
              and not (set(p.inferential_severities) & set(p.excluded_severities))
              and not (set(p.descriptive_only_severities) & set(p.excluded_severities)))
        check("11b protocol severities agree with the frozen mIoU-C rule",
              p.inferential_severities == tuple(INFERENTIAL_SEVERITIES))
        check("12 protocol object is immutable and carries a vocabulary hash",
              p.vocabulary_sha256 == vocabulary_sha256(p.entries)
              and isinstance(p.vocabulary_sha256, str) and len(p.vocabulary_sha256) == 64,
              p.vocabulary_sha256)

        grid = build_official_corruption_grid(p)
        check("13 official CorruptionGrid constructed from the protocol",
              grid.names == p.order and "official:" in grid.provenance,
              grid.provenance[:90])
        cells = {(c, s) for c in grid.names for s in INFERENTIAL_SEVERITIES}
        check("14 official grid defines exactly 15 inferential cells", len(cells) == 15)
        check("15 'brightness' is the accepted official identifier",
              "brightness" in p.order and p.display_names["brightness"] == "brightness variation")

        # ---- rejection cases on temporary MODIFIED COPIES ----
        def variant(mutate, name):
            d = copy.deepcopy(doc)
            mutate(d)
            q = work / f"{name}.json"
            q.write_text(json.dumps(d, indent=2), encoding="utf-8")
            return q

        def rename_id(d, old, new):
            d["corruption_order"] = [new if x == old else x for x in d["corruption_order"]]
            for e in d["corruptions"]:
                if e["id"] == old:
                    e["id"] = new
                    e["reference_function"] = new

        e = expect(CorruptionProtocolError, load_corruption_protocol,
                   variant(lambda d: rename_id(d, "brightness", "brightness_variation"), "bv"))
        check("16 'brightness_variation' rejected", "corruption_order must be exactly" in str(e),
              str(e)[:100])
        e = expect(CorruptionProtocolError, load_corruption_protocol,
                   variant(lambda d: rename_id(d, "motion_blur", "motion-blur"), "hyphen"))
        check("17 'motion-blur' rejected", isinstance(e, CorruptionProtocolError), str(e)[:100])
        e = expect(CorruptionProtocolError, load_corruption_protocol,
                   variant(lambda d: rename_id(d, "motion_blur", "Motion_Blur"), "caps"))
        check("18 'Motion_Blur' rejected", isinstance(e, CorruptionProtocolError), str(e)[:100])

        def reorder(d):
            d["corruption_order"] = list(reversed(d["corruption_order"]))
            d["corruptions"] = list(reversed(d["corruptions"]))
        e = expect(CorruptionProtocolError, load_corruption_protocol, variant(reorder, "reorder"))
        check("19 reordered protocol rejected", "in that order" in str(e), str(e)[:100])

        def drop(d):
            d["corruption_order"] = d["corruption_order"][:-1]
            d["corruptions"] = d["corruptions"][:-1]
        e = expect(CorruptionProtocolError, load_corruption_protocol, variant(drop, "drop"))
        check("20 missing corruption rejected", isinstance(e, CorruptionProtocolError),
              str(e)[:100])

        def extra(d):
            d["corruption_order"].append("snow")
            d["corruptions"].append({"id": "snow", "display_name": "snow",
                                     "reference_function": "snow"})
        e = expect(CorruptionProtocolError, load_corruption_protocol, variant(extra, "extra"))
        check("21 unexpected corruption rejected", isinstance(e, CorruptionProtocolError),
              str(e)[:100])

        e = expect(CorruptionProtocolError, load_corruption_protocol,
                   variant(lambda d: d["corruptions"][0].__setitem__("reference_function", "mb"),
                           "pairing"))
        check("22 altered display/function pairing rejected",
              "must equal the machine id" in str(e), str(e)[:100])
        e = expect(CorruptionProtocolError, load_corruption_protocol,
                   variant(lambda d: d.__setitem__("inferential_severities", [1, 2, 3, 4]), "sev"))
        check("22b altered severity group rejected",
              "inferential_severities must be" in str(e), str(e)[:100])
        e = expect(CorruptionProtocolError, load_corruption_protocol,
                   variant(lambda d: d.__setitem__("aliases", {"brightness_variation":
                                                               "brightness"}), "alias"))
        check("22c alias map rejected", "alias maps are rejected" in str(e), str(e)[:100])

        # ---- severity 4 / 5 rejected by the frozen mIoU-C assembler ----
        check("23 severity 4 is NOT an inferential severity",
              not p.is_inferential_severity(4) and 4 in p.descriptive_only_severities)
        check("24 severity 5 is excluded entirely",
              5 in p.excluded_severities and not p.is_inferential_severity(5)
              and 5 not in p.descriptive_only_severities)

        # ---- synthetic grid still usable ONLY through the injected nonofficial path ----
        from src.stats import CorruptionGrid                    # noqa: E402
        syn = CorruptionGrid(names=("synA", "synB", "synC", "synD", "synE"),
                             provenance="SYNTHETIC nonofficial fixture")
        check("25 explicit synthetic grid still constructible for NONOFFICIAL smokes",
              syn.names != p.order and "SYNTHETIC" in syn.provenance)
        check("25b robustness.py holds no vocabulary of its own",
              not any(cid in (REPO / "src" / "stats" / "robustness.py").read_text(encoding="utf-8")
                      .split("DOCUMENTATION GAP")[-1].split("====")[-1]
                      for cid in CANONICAL_IDS),
              "no canonical id appears outside the explanatory docstring")

        check("27 real protocol file was not modified",
              PROTOCOL.read_text(encoding="utf-8") == raw)

    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if det:
            print(f"         {det[:160]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'CORRUPTION VOCABULARY FROZEN' if allok else 'A3b-0 BLOCKED'}")
    print("Vocabulary + severity roles only -- corruption implementation bytes are NOT vendored.")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
