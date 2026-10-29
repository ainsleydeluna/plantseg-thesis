#!/usr/bin/env python3
"""L-AM16-SEV: severity 5 is descriptive for the four non-noise corruptions; gaussian_noise at
severity 5 stays refused at every layer (AM-16 item 6, docs/PREREGISTRATION_AMENDMENTS.md :303-306).

Synthetic and repository-local: no dataset, model, GPU or network. Rejection cases parse temporary
MODIFIED COPIES of the protocol; the real configs/corruption_protocol.json is never written. No file
or folder this smoke creates has "test" in its name. This process loads torch (src.stats and
src.eval import it); the torch-free loader is proven in a child process (A2).

  A  loaders: both loaders, the anchors and the vendor agree on the 24 cells   A1-A9
  B  rejections of modified copies, by both loaders                            B1-B8
  C  validate_condition, path A and path B                                     C1-C6
  D  frozen pieces untouched                                                   D1-D3
  M  mutations, each killed by its named case                                  M1-M5

Run:  python -B scripts/smoke_am16_severity.py
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import MappingProxyType

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.corruption_cache as cache_mod  # noqa: E402
import src.vendor.imagecorruptions as V  # noqa: E402
from src import corruption_roles as R  # noqa: E402
from src.eval import robustness as ER  # noqa: E402
from src.eval.robustness import (RobustnessError, load_cached_corruption,  # noqa: E402
                                 real_corruption_transform, validate_condition)
from src.stats import driver as D  # noqa: E402
from src.stats.corruption_protocol import (CorruptionProtocolError,  # noqa: E402
                                           load_corruption_protocol, official_corruption_grid)
from src.stats.robustness import INFERENTIAL_SEVERITIES, N_CORRUPTIONS  # noqa: E402

PROTOCOL = REPO / "configs" / "corruption_protocol.json"
RAW = PROTOCOL.read_bytes()
DOC = json.loads(RAW)
TMP = Path(tempfile.mkdtemp(prefix="am16sev_"))

CANON = ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog")
NON_NOISE = ("motion_blur", "jpeg_compression", "brightness", "fog")
#: TL interface section 6: the 24 cells in protocol order, written out by hand.
EXPECTED_CELL_IDS = (
    "motion_blur_s1", "motion_blur_s2", "motion_blur_s3", "motion_blur_s4", "motion_blur_s5",
    "gaussian_noise_s1", "gaussian_noise_s2", "gaussian_noise_s3", "gaussian_noise_s4",
    "jpeg_compression_s1", "jpeg_compression_s2", "jpeg_compression_s3", "jpeg_compression_s4",
    "jpeg_compression_s5",
    "brightness_s1", "brightness_s2", "brightness_s3", "brightness_s4", "brightness_s5",
    "fog_s1", "fog_s2", "fog_s3", "fog_s4", "fog_s5")
VOCABULARY_SHA256 = "7aa33cc22845723e96e75a2a59027b210d812c30a5497ff61d2a3ac98828aa7b"
ROBUSTNESS_BLOB = "3fb16de20bd01a1e84a85ab564711a7b2bfb2467"     # frozen src/stats/robustness.py
#: The statistics driver's 37 inputs before this lane: seven clean stages, then E1 and E6 on the
#: fifteen inferential cells (STATISTICAL_ANALYSIS_CONTRACT section 12.4.6).
PRE_LANE_INVENTORY = (
    tuple((s, ("clean", None, None)) for s in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"))
    + tuple((s, ("corruption", c, v)) for s in ("E1", "E6") for c in CANON for v in (1, 2, 3)))

CASES: dict = {}
results: list[tuple[str, bool, str]] = []


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def run(name) -> tuple[bool, str]:
    try:
        ok, detail = CASES[name]()
        return bool(ok), str(detail)
    except Exception as e:                                   # noqa: BLE001
        return False, f"raised {type(e).__name__}: {e}"


def record(label, name):
    ok, detail = run(name)
    results.append((label, ok, detail))


@contextlib.contextmanager
def patched(obj, attr, value):
    saved = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, saved)


def code_of(fn, *a, **kw):
    """The refusal code (or exception type) `fn` raises; None when it returns."""
    try:
        fn(*a, **kw)
    except RobustnessError as e:
        return e.code
    except Exception as e:                                   # noqa: BLE001
        return type(e).__name__
    return None


def triples(cells):
    return [(c.corruption, c.severity, c.role) for c in cells]


# ---------------------------------------------------------------- A. loaders
@case("A1")
def a1():
    p = load_corruption_protocol()
    return (p.schema_version == "plantseg-corruptions/2.0.0" and p.order == CANON
            and len(p.cells) == 24), f"{p.schema_version}, {len(p.cells)} cells"


@case("A2")
def a2():
    child = ("import json, sys\n"
             f"sys.path.insert(0, {str(REPO)!r})\n"
             "from src import corruption_roles as R\n"
             "r = R.load()\n"
             "print(json.dumps({'schema': r.schema_version, 'cells': list(r.cell_ids),"
             " 'torch': 'torch' in sys.modules, 'numpy': 'numpy' in sys.modules}))\n")
    p = subprocess.run([sys.executable, "-B", "-c", child], cwd=str(REPO), capture_output=True,
                       text=True, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if p.returncode != 0:
        return False, p.stderr.strip()[-160:]
    out = json.loads(p.stdout)
    return (out["schema"] == "plantseg-corruptions/2.0.0"
            and tuple(out["cells"]) == EXPECTED_CELL_IDS
            and not out["torch"] and not out["numpy"]), f"torch={out['torch']} numpy={out['numpy']}"


@case("A3")
def a3():
    stats_cells = triples(load_corruption_protocol().cells)
    free_cells = triples(R.load().cells)
    anchor_cells = triples(R.CELLS)
    vendor_cells = [(c, s, "inferential" if s in (1, 2, 3) else "descriptive_only")
                    for c in V.CORRUPTION_IDS for s in V.SEVERITIES_BY_CORRUPTION[c]]
    return (stats_cells == free_cells == anchor_cells == vendor_cells
            and len(stats_cells) == 24), f"{len(stats_cells)}/{len(vendor_cells)} cells"


@case("A4")
def a4():
    cells = load_corruption_protocol().cells
    inferential = [c for c in cells if c.role == "inferential"]
    s4 = [c for c in cells if c.severity == 4]
    s5 = [c for c in cells if c.severity == 5]
    return (len(inferential) == 15 and {c.severity for c in inferential} == {1, 2, 3}
            and len(s4) == 5 and all(c.role == "descriptive_only" for c in s4)
            and len(s5) == 4 and all(c.role == "descriptive_only" for c in s5)
            and tuple(c.corruption for c in s5) == NON_NOISE), \
        f"inferential {len(inferential)}, severity 4 {len(s4)}, severity 5 {len(s5)}"


@case("A5")
def a5():
    p = load_corruption_protocol()
    try:
        R.cell_id("gaussian_noise", 5)
        id_refused = False
    except R.CorruptionRolesError:
        id_refused = True
    return (("gaussian_noise", 5) not in {(c.corruption, c.severity) for c in p.cells}
            and ("gaussian_noise", 5) not in {(c.corruption, c.severity) for c in R.load().cells}
            and 5 not in V.SEVERITIES_BY_CORRUPTION["gaussian_noise"]
            and tuple(V.EXCLUDED_CELLS) == (("gaussian_noise", 5),)
            and "gaussian_noise_s5" not in R.CELL_IDS and id_refused
            and p.severity_role("gaussian_noise", 5) == "excluded"), "absent from every layer"


@case("A6")
def a6():
    g = official_corruption_grid()
    return (tuple(g.names) == CANON and "plantseg-corruptions/2.0.0" in g.provenance), \
        f"{list(g.names)}"


@case("A7")
def a7():
    inv = tuple(D.official_inventory())
    return inv == PRE_LANE_INVENTORY and len(inv) == 37, f"{len(inv)} entries"


@case("A8")
def a8():
    v = load_corruption_protocol().vocabulary_sha256
    return v == VOCABULARY_SHA256, v[:16]


@case("A9")
def a9():
    refused = []
    for fn, args in ((R.parse_cell_id, ("gaussian_noise_s5",)), (R.cell_id, ("fog", 6)),
                     (R.parse_cell_id, ("fog_s05",)), (R.cell_id, ("brightness_variation", 1))):
        try:
            fn(*args)
            refused.append(False)
        except R.CorruptionRolesError:
            refused.append(True)
    return (R.CELL_IDS == EXPECTED_CELL_IDS and R.registered_cell_ids() == EXPECTED_CELL_IDS
            and load_corruption_protocol().cell_ids == EXPECTED_CELL_IDS
            and all(R.cell_id(*R.parse_cell_id(x)) == x for x in EXPECTED_CELL_IDS)
            and R.INFERENTIAL_CELL_IDS == tuple(x for x in EXPECTED_CELL_IDS if int(x[-1]) <= 3)
            and all(refused)), f"{len(R.CELL_IDS)} ids; refusals {refused}"


# ---------------------------------------------------------------- B. rejections of copies
def both_refuse(name, mutate, needle):
    d = copy.deepcopy(DOC)
    mutate(d)
    q = TMP / f"{name}.json"
    q.write_text(json.dumps(d, indent=2), encoding="utf-8")
    msgs = []
    for loader, exc in ((R.load, R.CorruptionRolesError),
                        (load_corruption_protocol, CorruptionProtocolError)):
        try:
            loader(q)
            return False, f"{loader.__module__}.{loader.__name__} accepted the copy"
        except exc as e:
            msgs.append(str(e))
    return all(needle in m for m in msgs), msgs[0][:120]


def set_roles(cid, descriptive, excluded):
    def _m(d):
        for e in d["corruptions"]:
            if e["id"] == cid:
                e["descriptive_only_severities"] = descriptive
                e["excluded_severities"] = excluded
    return _m


def as_v1(d):
    d.clear()
    d.update({"schema_version": "plantseg-corruptions/1.0.0",
              "identifier_basis": DOC["identifier_basis"], "condition_type": "corruption",
              "corruption_order": list(CANON), "inferential_severities": [1, 2, 3],
              "descriptive_only_severities": [4], "excluded_severities": [5],
              "corruptions": [{k: e[k] for k in ("id", "display_name", "reference_function")}
                              for e in DOC["corruptions"]]})


@case("B1")
def b1():
    return both_refuse("b1_gn_desc", set_roles("gaussian_noise", [4, 5], []),
                       "gaussian_noise: descriptive_only_severities must be [4]")


@case("B2")
def b2():
    return both_refuse("b2_fog_no5", set_roles("fog", [4], []),
                       "fog: the severity roles must cover exactly")


@case("B3")
def b3():
    return both_refuse("b3_fog_excl5", set_roles("fog", [4], [5]),
                       "fog: descriptive_only_severities must be [4, 5]")


@case("B4")
def b4():
    return both_refuse("b4_two_roles", set_roles("fog", [4, 5], [5]),
                       "fog: severity groups must be disjoint")


@case("B5")
def b5():
    return both_refuse("b5_sev6", set_roles("fog", [4, 5, 6], []), "lie outside")


@case("B6")
def b6():
    return both_refuse("b6_infer", lambda d: d.__setitem__("inferential_severities", [1, 2, 3, 4]),
                       "inferential_severities must be")


@case("B7")
def b7():
    return both_refuse("b7_v1", as_v1, "schema_version must be 'plantseg-corruptions/2.0.0'")


@case("B8")
def b8():
    return both_refuse("b8_gridwide", lambda d: d.__setitem__("excluded_severities", [5]),
                       "grid-wide")


# ---------------------------------------------------------------- C. condition gates and paths
@case("C1")
def c1():
    want = {(c, s): ("inferential" if s <= 3 else "descriptive_only")
            for c in CANON for s in range(1, 6)}
    want[("gaussian_noise", 5)] = "severity_excluded"
    got = {}
    for c, s in want:
        try:
            got[(c, s)] = validate_condition(c, s, official=True)["role"]
        except RobustnessError as e:
            got[(c, s)] = e.code
    bad = {k: v for k, v in got.items() if want[k] != v}
    return not bad, f"25 pairs; mismatches {bad}" if bad else "25 pairs"


@case("C2")
def c2():
    codes = [code_of(validate_condition, "gaussian_noise", 5, official=o) for o in (True, False)]
    return codes == ["severity_excluded", "severity_excluded"], str(codes)


@case("C3")
def c3():
    codes = {(c, s): code_of(validate_condition, c, s, official=True)
             for c in CANON for s in (0, 6)}
    return all(v == "severity_unknown" for v in codes.values()), f"{len(codes)} requests"


@case("C4")
def c4():
    codes = [code_of(validate_condition, n, s, official=True)
             for n, s in (("brightness_variation", 1), ("motion-blur", 5), ("Fog", 5),
                          ("gaussian-noise", 5))]
    return all(c == "unknown_corruption" for c in codes), str(codes)


@case("C5")
def c5():
    calls = []
    with patched(V, "apply_corruption", lambda *a, **k: calls.append(a)):
        code = code_of(real_corruption_transform, "gaussian_noise", 5, image_id="x")
        fog5 = real_corruption_transform("fog", 5, image_id="x")
    return (code == "severity_excluded" and callable(fog5) and not calls), \
        f"code={code}, vendor calls={len(calls)}"


@case("C6")
def c6():
    root = TMP / "never_created_cache"
    calls = []
    with patched(cache_mod, "load_cached", lambda *a, **k: calls.append(a)):
        code = code_of(load_cached_corruption, root, {}, "img_0", "gaussian_noise", 5)
    return (code == "severity_excluded" and not calls and not root.exists()), \
        f"code={code}, cache reads={len(calls)}"


# ---------------------------------------------------------------- D. frozen pieces
@case("D1")
def d1():
    data = (REPO / "src" / "stats" / "robustness.py").read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return blob == ROBUSTNESS_BLOB, blob[:12]


@case("D2")
def d2():
    return tuple(INFERENTIAL_SEVERITIES) == (1, 2, 3) and N_CORRUPTIONS == 5, \
        f"{tuple(INFERENTIAL_SEVERITIES)}, {N_CORRUPTIONS}"


@case("D3")
def d3():
    return PROTOCOL.read_bytes() == RAW, hashlib.sha256(RAW).hexdigest()[:16]


# ---------------------------------------------------------------- M. mutations
class GridWide:
    """A protocol object that applies severity roles grid-wide (severity 5 descriptive for all)."""

    def __init__(self, real):
        self.entries = real.entries

    def severity_role(self, corruption, severity):
        if severity in (1, 2, 3):
            return "inferential"
        return "descriptive_only" if severity in (4, 5) else None

    def is_inferential_severity(self, severity):
        return severity in (1, 2, 3)


def mutant(label, targets, *patches):
    with contextlib.ExitStack() as stack:
        for obj, attr, value in patches:
            stack.enter_context(patched(obj, attr, value))
        outcomes = {t: run(t)[0] for t in targets}
    killed = not any(outcomes.values())
    print(f"MUTANT {label} {'killed' if killed else 'SURVIVED'} by {', '.join(targets)}")
    results.append((f"{label} killed by {', '.join(targets)}", killed, str(outcomes)))


def mutations():
    mutant("M1 relaxed gaussian_noise anchor", ["B1"],
           (R, "DESCRIPTIVE_ONLY_BY_CORRUPTION",
            MappingProxyType({**R.DESCRIPTIVE_ONLY_BY_CORRUPTION, "gaussian_noise": (4, 5)})),
           (R, "EXCLUDED_BY_CORRUPTION",
            MappingProxyType({**R.EXCLUDED_BY_CORRUPTION, "gaussian_noise": ()})))
    real = load_corruption_protocol()
    mutant("M2 grid-wide roles", ["C1", "C2"], (ER, "protocol", lambda: GridWide(real)))
    mutant("M3 vendor set with (gaussian_noise, 5)", ["A3"],
           (V, "SEVERITIES_BY_CORRUPTION",
            MappingProxyType({**V.SEVERITIES_BY_CORRUPTION, "gaussian_noise": (1, 2, 3, 4, 5)})))
    mutant("M4 inferential (1, 2, 3, 4)", ["B6"], (R, "INFERENTIAL_SEVERITIES", (1, 2, 3, 4)))
    mutant("M5 driver inventory from the scored severities", ["A7"],
           (D, "official_inventory",
            lambda: tuple((s, D.CLEAN) for s in D.STAGES)
            + tuple((st, ("corruption", c.corruption, c.severity))
                    for st in D.CORRUPTION_STAGES for c in R.CELLS)))


LABELS = {
    "A1": "A1 the stats loader reads 2.0.0",
    "A2": "A2 the torch-free loader reads it in a child with neither torch nor numpy",
    "A3": "A3 stats loader, torch-free loader, anchors and vendor: the same 24 cells in order",
    "A4": "A4 15 inferential cells, 5 at severity 4, 4 at severity 5",
    "A5": "A5 no (gaussian_noise, 5) in any layer",
    "A6": "A6 the official grid names and order are unchanged",
    "A7": "A7 the driver inventory equals the pre-lane 37 entries",
    "A8": "A8 vocabulary_sha256 is 7aa33cc2",
    "A9": "A9 cell ids (LC1): CELL_IDS is IF section 6's list; cell_id and parse_cell_id agree",
    "B1": "B1 a copy with gaussian_noise descriptive at [4, 5] is refused",
    "B2": "B2 a copy with fog without severity 5 is refused",
    "B3": "B3 a copy with fog 5 excluded is refused",
    "B4": "B4 a copy with one severity in two roles is refused",
    "B5": "B5 a copy with severity 6 is refused",
    "B6": "B6 a copy with inferential [1, 2, 3, 4] is refused",
    "B7": "B7 a 1.0.0 file is refused",
    "B8": "B8 a copy with a grid-wide role key is refused",
    "C1": "C1 validate_condition: the protocol role for all 25 pairs",
    "C2": "C2 (gaussian_noise, 5) is severity_excluded for every status",
    "C3": "C3 severities 0 and 6 are severity_unknown",
    "C4": "C4 aliases are unknown_corruption",
    "C5": "C5 path A refuses (gaussian_noise, 5) before any vendor call",
    "C6": "C6 path B refuses (gaussian_noise, 5) before any cache read",
    "D1": "D1 src/stats/robustness.py blob is 3fb16de2",
    "D2": "D2 INFERENTIAL_SEVERITIES is (1, 2, 3) and N_CORRUPTIONS is 5",
    "D3": "D3 the real protocol file was not modified",
}


def main() -> int:
    print("=" * 100)
    print("L-AM16-SEV -- per-corruption severity roles (synthetic, repository-local)")
    print("=" * 100)
    try:
        for name in [n for n in CASES if n != "D3"]:
            record(LABELS[name], name)
        mutations()
        record(LABELS["D3"], "D3")
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        results.append(("FATAL", False, traceback.format_exc(limit=2)))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print()
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail:
            print(f"         {detail[:160]}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
