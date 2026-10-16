#!/usr/bin/env python3
"""S3 — corruption-cache verification against the real core_preprocess (L-CORR B-2). Torch side.

The capped synthetic cache is built by scripts/build_corruption_cache.py in a SUBPROCESS
(torch-free, so brightness never runs in a torch process) on S2's synthetic TRAIN/VAL-only root;
this process then verifies it with scripts/verify_corruption_cache.py, which runs the real, frozen
src/data/transforms.core_preprocess. No file or folder this smoke creates has "test" in its name;
TEST refusals run on never-created paths before any read, under a zero-call tracer.

  V1-V8   the plan's cases           V9-V15  every other refusal code of the verifier
  V16-V18 LC2: an official or provisional cache is verified only in the student image on the
          pinned stack (a copy re-sealed as provisional)
  V19-V24 a malformed EXIF orientation; strict-JSON manifests; the registered
          cell set; the data-root gate; a 1.0.0 manifest
  V25-V27 item paths by index (LCC1, LCC3): an entry not at its index path; a verified cache's
          item paths and created names (S2's zqmarker stem); the five-digit integer index guard
  M       mutations, each killed by its named case (a case that raised is no kill)

Run:  python -B scripts/smoke_corruption_verify.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import scripts.verify_corruption_cache as VC  # noqa: E402
import src.corruption_cache as cc  # noqa: E402
import src.corruption_source as cs  # noqa: E402
from scripts.smoke_corruption_generator import (DIGEST, MARKER, SPECIALS, SUBSET,  # noqa: E402
                                                _fake_surface_isolation, env, make_val_root,
                                                malformed_orientation_jpeg, names_holding,
                                                path_problems)

CASES: dict = {}
results: list[tuple[str, bool, str]] = []
S: dict = {}
_SEQ = itertools.count()
LABELS_ORDER = ("canonical_manifest_sha256", "inferential_cells_sha256", "manifest_file_sha256",
                "verification_record_sha256")


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


@contextlib.contextmanager
def patched(obj, attr, value):
    saved = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, saved)


@contextlib.contextmanager
def tracer():
    """Counts manifest loads, split listings and source reads (the zero-call tracer)."""
    calls = []
    real_load, real_list, real_window = cc.load_manifest, cs.list_split, VC.core_window

    def load(root):
        calls.append(("manifest", str(root)))
        return real_load(root)

    def listing(root, split):
        calls.append(("list", str(root), split))
        return real_list(root, split)

    def window(raw):
        calls.append(("read",))
        return real_window(raw)
    with patched(cc, "load_manifest", load), patched(cs, "list_split", listing), \
            patched(VC, "core_window", window):
        yield calls


def never(name: str) -> Path:
    return S["work"] / "never_created" / name


def verify_argv(cache_dir, *extra, split="val", data_root=None):
    return ["--cache-dir", str(cache_dir), "--data-root", str(data_root or S["data"]),
            "--split", split, *extra]


def verify_code(argv):
    args = VC.build_parser().parse_args(argv)
    try:
        VC.verify(args)
    except (cc.CorruptionCacheError, cs.CorruptionSourceError) as e:
        return e.code
    except Exception as e:                                   # noqa: BLE001
        return type(e).__name__
    return None


def verify_main(argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = VC.main(argv)
    return rc, out.getvalue(), err.getvalue()


def copy_cache(tag: str) -> Path:
    dst = S["work"] / f"cache_copy_{tag}_{next(_SEQ)}"
    shutil.copytree(S["cache"], dst)
    return dst


def forge(root: Path, fn) -> None:
    """Edit a cache copy's manifest and re-seal its canonical checksum."""
    m = json.loads((root / cc.MANIFEST_NAME).read_text(encoding="utf-8"))
    fn(m)
    m["canonical_manifest_sha256"] = cc.canonical_manifest_sha256(m)
    (root / cc.MANIFEST_NAME).write_bytes(cc.manifest_bytes(m))


def records(root: Path) -> list[Path]:
    d = root / VC.RECORD_DIR
    return sorted(d.iterdir()) if d.is_dir() else []


def setup(work: Path) -> None:
    S["work"] = work
    S["data"] = make_val_root(work / "data")
    S["cache"] = work / "cache_main"
    proc = subprocess.run(
        [sys.executable, "-B", str(REPO / "scripts" / "build_corruption_cache.py"),
         "--real-run", "--confirm-real-run", "--split", "val", "--cache-status", "smoke",
         "--data-root", str(S["data"]), "--cache-dir", str(S["cache"]), "--max-images", "6"],
        cwd=str(REPO), capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    S["build"] = (proc.returncode, proc.stderr[-300:])
    S["m"] = cc.load_manifest(S["cache"]) if proc.returncode == 0 else None
    S["listing"] = cs.list_split(S["data"], "val")


# ---------------------------------------------------------------- V. the plan's cases
@case("V1")
def v1():
    if S["m"] is None:
        return False, f"generator subprocess failed: {S['build']}"
    rc, out, err = verify_main(verify_argv(S["cache"]))
    lines = out.strip().splitlines()
    vals = dict(ln.split(": ", 1) for ln in lines) if rc == 0 else {}
    recs = records(S["cache"])
    S["v1"] = vals, recs
    return (rc == 0 and [ln.split(":")[0] for ln in lines] == list(LABELS_ORDER)
            and vals["canonical_manifest_sha256"] == S["m"]["canonical_manifest_sha256"]
            and vals["inferential_cells_sha256"] == S["m"]["inferential_cells_sha256"]
            and vals["manifest_file_sha256"]
            == hashlib.sha256((S["cache"] / cc.MANIFEST_NAME).read_bytes()).hexdigest()
            and len(recs) == 1
            and vals["verification_record_sha256"]
            == hashlib.sha256(recs[0].read_bytes()).hexdigest()), f"rc={rc} {err.strip()[-120:]}"


@case("V2")
def v2():
    """Every special image: the torch-free reader == the real core_preprocess window, and the
    verifier refuses a record whose pixel digest is changed."""
    rec = {s["manifest_index"]: s for s in S["m"]["sources"]}
    ok = True
    for i in SUBSET:
        src = S["listing"][i]
        raw = Path(src.path).read_bytes()
        window = VC.core_window(raw)[0]
        ok &= (np.array_equal(cs.read_source(src).pixels, window)
               and hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest()
               == rec[i]["resized_pixel_sha256"])
        VC.check_source(rec[i], src)
        flipped = {**rec[i], "resized_pixel_sha256": "0" * 64}
        try:
            VC.check_source(flipped, src)
            ok = False
        except cc.CorruptionCacheError as e:
            ok &= e.code == "source_pixels_mismatch"
    return ok, "six specials: reader == core_preprocess window; a changed digest refused"


@case("V3")
def v3():
    ok = True
    for i in SUBSET:
        h, w = SPECIALS[i][3]
        _, canvas, mcanvas, (top, left, hh, ww) = VC.core_window(
            Path(S["listing"][i].path).read_bytes())
        ok &= ((hh, ww) == (h, w) and (top, left) == ((512 - h) // 2, (512 - w) // 2)
               and int((mcanvas != 255).sum()) == h * w
               and np.all(canvas[mcanvas == 255] == np.array([124, 116, 104], np.uint8)))
    return ok, "window = the non-padding region; padding is the mean colour"


@case("V4")
def v4():
    import configs.data as cd
    from src.eval import adapters
    saved = cd.DATA["root"]
    try:
        cd.DATA["root"] = str(S["data"])
        stems = adapters.list_split_stems("val")
    finally:
        cd.DATA["root"] = saved
    return (len(stems) == 846 and stems == [s.image_id for s in S["listing"]]
            and all(stems[s["manifest_index"]] == s["image_id"] for s in S["m"]["sources"])), \
        "stems and manifest_index == adapters.list_split_stems"


@case("V5")
def v5():
    p = Path(S["listing"][SUBSET[1]].path)
    original = p.read_bytes()
    try:
        Image.fromarray(np.full((70, 1024, 3), 77, np.uint8)).convert("L").save(p, format="JPEG")
        code = verify_code(verify_argv(S["cache"]))
    finally:
        p.write_bytes(original)
    return code == "source_file_mismatch", str(code)


@case("V6")
def v6():
    root = copy_cache("v6")
    for r in records(root):
        r.unlink()
    e = S["m"]["entries"][5]
    path = root / e["cache_relative_path"]
    np.save(path, (np.load(path) + 1).astype(np.uint8), allow_pickle=False)
    code = verify_code(verify_argv(root))
    left = records(root)
    return code == "cache_file_hash_mismatch" and not left, f"{code}, records={left}"


@case("V7")
def v7():
    vals, recs = S.get("v1", ({}, []))
    if not recs:
        return False, "V1 wrote no record"
    rec = json.loads(recs[0].read_text(encoding="utf-8"))
    again = cc.load_manifest(S["cache"])
    return (rec["counts"] == {"sources_verified": 6, "items_verified": 144, "cells": 24}
            and rec["canonical_manifest_sha256"] == again["canonical_manifest_sha256"]
            and recs[0].parent.name == VC.RECORD_DIR
            and not any(e["cache_relative_path"].startswith(VC.RECORD_DIR)
                        for e in again["entries"])), str(rec["counts"])


@case("V8")
def v8():
    with tracer() as calls:
        code = verify_code(verify_argv(never("v8") / cc.TDAY_CACHE_ROOT_NAME, split="test",
                                       data_root=never("tday_root_v8")))
    return code == "test_split_unconfirmed" and not calls, f"code={code}, calls={len(calls)}"


# ---------------------------------------------------------------- V9-V15. other refusal codes
@case("V9")
def v9():
    with tracer() as calls:
        code = verify_code(verify_argv(S["cache"], "--confirm-test-split", split="test"))
    reads = [c for c in calls if c[0] != "manifest"]
    return code == "split_mismatch" and not reads, f"code={code}, reads={len(reads)}"


@case("V10")
def v10():
    root = copy_cache("v10")
    forge(root, lambda m: m["sources"][1].__setitem__("file_name", "syn_val_99999.jpg"))
    code = verify_code(verify_argv(root))
    return code == "sources_mismatch", str(code)


@case("V11")
def v11():
    root = copy_cache("v11")
    forge(root, lambda m: m["sources"][2].__setitem__("resized_pixel_sha256", "0" * 64))
    code = verify_code(verify_argv(root))
    return code == "source_pixels_mismatch", str(code)


@case("V12")
def v12():
    root = copy_cache("v12")
    forge(root, lambda m: m["cells"][7].__setitem__("cell_sha256", "0" * 64))
    code = verify_code(verify_argv(root))
    return code == "cell_digest_mismatch", str(code)


@case("V13")
def v13():
    root = copy_cache("v13")
    forge(root, lambda m: m["pin"].__setitem__("commit", "0" * 40))
    code = verify_code(verify_argv(root))
    return code == "pin_invalid", str(code)


@case("V14")
def v14():
    with tracer() as calls:
        code = verify_code(verify_argv(never("Verify_TeSt_root")))
    return code == "cache_root_test_named" and not calls, f"code={code}, calls={len(calls)}"


@case("V15")
def v15():
    root = copy_cache("v15")
    m = json.loads((root / cc.MANIFEST_NAME).read_text(encoding="utf-8"))
    e = m["entries"][3]
    path = root / e["cache_relative_path"]
    small = np.zeros((10, 10, 3), np.uint8)
    np.save(path, small, allow_pickle=False)

    def fix(mm):
        x = mm["entries"][3]
        x["shape"] = [10, 10, 3]
        x["cached_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        x["corrupted_pixel_sha256"] = hashlib.sha256(small.tobytes()).hexdigest()
    forge(root, fix)
    code = verify_code(verify_argv(root))
    return code == "cache_entry_corrupt", str(code)


# ---------------------------------------------------------------- V16-V18. LC2's environment
def fresh_copy(tag: str) -> Path:
    """A copy of the capped cache without the copied verification record."""
    root = copy_cache(tag)
    for r in records(root):
        r.unlink()
    return root


def provisional_copy(tag: str) -> Path:
    """A copy of the capped cache re-sealed as provisional, with no verification record."""
    root = fresh_copy(tag)
    forge(root, lambda m: m.__setitem__("status", "provisional"))
    return root


def _env_refusal(tag, code, *ctx):
    root = provisional_copy(tag)
    with contextlib.ExitStack() as stack:
        for c in ctx:
            stack.enter_context(c)
        calls = stack.enter_context(tracer())
        got = verify_code(verify_argv(root))
    reads = [c for c in calls if c[0] != "manifest"]
    return (got == code and not reads and not records(root)), f"code={got}, reads={len(reads)}"


@case("V16")
def v16():
    return _env_refusal("v16", "image_digest_mismatch", env(PLANTSEG_IMAGE_DIGEST=None))


@case("V17")
def v17():
    fake = {**cc.runtime_versions(), "numpy": "2.0.0"}
    return _env_refusal("v17", "official_dependency_version_mismatch", env(**DIGEST),
                        patched(cc, "runtime_versions", lambda: fake))


@case("V18")
def v18():
    root = provisional_copy("v18")
    with env(**DIGEST):
        rc, _, err = verify_main(verify_argv(root))
    recs = records(root)
    rec = json.loads(recs[0].read_text(encoding="utf-8")) if rc == 0 and len(recs) == 1 else {}
    return (rc == 0 and rec.get("status") == "provisional"
            and rec.get("image_digest") == cc.STUDENT_IMAGE_DIGEST
            and rec.get("versions", {}).get("pillow") == cc.OFFICIAL_PILLOW), \
        f"rc={rc} {err.strip()[-90:]}"


@case("V19")
def v19():
    """The torch-free reader equals the real core_preprocess on a malformed (UNDEFINED) EXIF
    orientation, which neither applies; it is recorded by its repr."""
    arr = (np.arange(48 * 40 * 3) % 253).astype(np.uint8).reshape(48, 40, 3)
    p = malformed_orientation_jpeg(S["work"] / "orient_undefined.jpg", arr, typ=7)
    si = cs.read_source(cs.Source(0, "orient_undefined", p))
    window = VC.core_window(p.read_bytes())[0]
    return (np.array_equal(si.pixels, window)
            and si.record["exif_orientation"] == repr(b"\x06")), str(si.pixels.shape)


@case("V20")
def v20():
    """load_manifest reads strict JSON: a duplicate key, NaN or 1e999 -> manifest_schema."""
    codes = []
    for insert in ('"split":"val",', '"zz_nan":NaN,', '"zz_big":1e999,'):
        root = copy_cache("v20")
        p = root / cc.MANIFEST_NAME
        text = p.read_text(encoding="utf-8")
        p.write_text("{" + insert + text.lstrip()[1:], encoding="utf-8")
        codes.append(verify_code(verify_argv(root)))
    return codes == ["manifest_schema"] * 3, str(codes)


@case("V22")
def v22():
    """A re-sealed manifest without the fog_s5 cell and its entries -> cell_digest_mismatch: the
    cells must be the registered 24 (LC13), as the pin lists them."""
    root = fresh_copy("v22")

    def drop(m):
        m["cells"] = [c for c in m["cells"] if c["cell_id"] != "fog_s5"]
        m["entries"] = [e for e in m["entries"]
                        if not (e["corruption_id"] == "fog" and e["severity"] == 5)]
    forge(root, drop)
    code = verify_code(verify_argv(root))
    return code == "cell_digest_mismatch" and not records(root), str(code)


@case("V23")
def v23():
    """The data-root gate runs before any listing or read: a VAL root reporting a TEST surface
    (S2's isolation stub; nothing is created) -> test_split_present."""
    root = fresh_copy("v23")
    with tracer() as calls, patched(cs, "load_isolation", _fake_surface_isolation):
        code = verify_code(verify_argv(root))
    reads = [c for c in calls if c[0] != "manifest"]
    return (code == "test_split_present" and not reads
            and not records(root)), f"code={code}, reads={len(reads)}"


@case("V24")
def v24():
    """A 1.0.0 manifest -> manifest_schema (TL section 4)."""
    root = fresh_copy("v24")
    forge(root, lambda m: m.__setitem__("schema", "plantseg-corruption-cache/1.0.0"))
    code = verify_code(verify_argv(root))
    return code == "manifest_schema", str(code)


# ---------------------------------------------------------------- V25-V26. item paths by index
@case("V25")
def v25():
    """A re-sealed manifest whose first entry names its stem path, not its index path ->
    item_path_mismatch, before any source or item is read, and no record (LCC1). The stem path
    is a string only: no such file is created."""
    root = fresh_copy("v25")

    def stem(m):
        e = m["entries"][0]
        e["cache_relative_path"] = f"{e['corruption_id']}/s{e['severity']}/{e['image_id']}.npy"
    forge(root, stem)
    with tracer() as calls:
        code = verify_code(verify_argv(root))
    reads = [c for c in calls if c[0] != "manifest"]
    return (code == "item_path_mismatch" and not reads
            and not records(root)), f"code={code}, reads={len(reads)}"


@case("V26")
def v26():
    """The verifier reads every item at its index path: a fresh copy verifies, its entries sit at
    <corruption>/s<severity>/<manifest_index, 5 digits>.npy with the item files exactly those, and
    no created name holds S2's zqmarker stem, 'test' or ':' (LCC1, LCC3)."""
    root = fresh_copy("v26")
    rc, _, err = verify_main(verify_argv(root))
    m = json.loads((root / cc.MANIFEST_NAME).read_text(encoding="utf-8"))
    bad = path_problems(root, m)
    held = names_holding(root, MARKER, "test", ":")
    marked = any(MARKER in s["image_id"] for s in m["sources"])
    return (rc == 0 and marked and not bad and not held and len(records(root)) == 1), \
        f"rc={rc} marked={marked} paths={bad[:2]} names={held[:2]} {err.strip()[-60:]}"


@case("V27")
def v27():
    """The five-digit integer index guard (DL-24 C4-N2): a re-sealed manifest whose source has
    manifest_index 100000 (its entries at the six-digit paths), an entry whose image_id no source
    holds, or a source whose manifest_index is the string "141" -> item_path_mismatch each, before
    any source or item is read, and no record (LCC1)."""
    def six_digits(m):
        s = m["sources"][0]
        s["manifest_index"] = 100000
        for e in m["entries"]:
            if e["image_id"] == s["image_id"]:
                e["cache_relative_path"] = f"{e['corruption_id']}/s{e['severity']}/100000.npy"

    def no_source(m):
        m["entries"][0]["image_id"] = "syn_val_unlisted"

    def text_index(m):
        s = m["sources"][1]
        s["manifest_index"] = str(s["manifest_index"])
    out = []
    for fn in (six_digits, no_source, text_index):
        root = fresh_copy("v27")
        forge(root, fn)
        with tracer() as calls:
            code = verify_code(verify_argv(root))
        out.append((code, len([c for c in calls if c[0] != "manifest"]), len(records(root))))
    return all(o == ("item_path_mismatch", 0, 0) for o in out), str(out)


# ---------------------------------------------------------------- M. mutations
def noop(*a, **k):
    return None


def mutant(label, targets, *patches):
    """Killed only when every named case fails by its own checks: a case that raised is not a
    kill, because its checks never ran."""
    with contextlib.ExitStack() as stack:
        for obj, attr, value in patches:
            stack.enter_context(patched(obj, attr, value))
        outcomes = {t: run(t) for t in targets}
    killed = all(not ok and not d.startswith("raised ") for ok, d in outcomes.values())
    why = "; ".join(f"{t}: {d[:90]}" for t, (_, d) in outcomes.items())
    print(f"MUTANT {label} {'killed' if killed else 'SURVIVED'} by {', '.join(targets)} [{why}]")
    results.append((f"{label} killed by {', '.join(targets)}", killed, why))


def _paths_without_index_guard(manifest):
    """A mutant check: the path comparison alone, without the five-digit integer index guard."""
    index_of = {s.get("image_id"): s.get("manifest_index") for s in manifest.get("sources") or []}
    return [e.get("cache_relative_path") for e in manifest.get("entries") or []
            if e.get("cache_relative_path") != cc.item_relative_path(
                e.get("corruption_id"), e.get("severity"), index_of.get(e.get("image_id")))]


def mutations():
    real_verify, real_gate_sources = VC.verify, VC.gate_sources

    def shapes_only(record, raw):
        h, w = VC.core_window(raw)[3][2:]
        if [h, w] != record["resized_shape"]:
            raise cc.CorruptionCacheError("source_pixels_mismatch", "shape")

    def record_first(args):
        d = Path(args.cache_dir) / VC.RECORD_DIR
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cc.utc_stamp()}.json").write_text("{}\n", encoding="utf-8")
        return real_verify(args)

    def unchecked_sources(args, manifest):
        listing = cs.list_split(args.data_root, args.split)
        return {r["image_id"]: listing[(r["manifest_index"] + 1) % len(listing)]
                for r in manifest["sources"]}

    mutant("M1 sources skipped", ["V5"], (VC, "check_source", noop))
    mutant("M2 shapes only", ["V2"], (VC, "check_window", shapes_only))
    mutant("M3 record written before the checks", ["V6"], (VC, "verify", record_first))
    mutant("M-split_mismatch", ["V9"], (VC, "gate_split", noop))
    mutant("M-sources_mismatch", ["V10"], (VC, "gate_sources", unchecked_sources))
    mutant("M-source_file_mismatch", ["V5"], (VC, "check_file", noop))
    mutant("M-source_pixels_mismatch", ["V11"], (VC, "check_window", noop))
    mutant("M-cell_digest_mismatch", ["V12"], (VC, "check_cells", noop))
    mutant("M-pin_invalid", ["V13"], (VC, "check_pin", noop))
    mutant("M-test_split_unconfirmed", ["V8"], (VC, "gate_test_confirm", noop))
    mutant("M-cache_root_test_named", ["V14"], (cc, "refuse_test_named", noop))
    mutant("M-cache_entry_corrupt (item shape vs source)", ["V15"],
           (VC, "check_item_shape", noop))
    mutant("M-verified outside the student image (LC2)", ["V16", "V17"],
           (VC, "gate_environment", noop))
    mutant("M-manifest parsed leniently", ["V20"], (cc, "_strict_json", json.loads))
    mutant("M-cell set not checked (LC13)", ["V22"], (VC, "cell_set_problems", lambda m: []))
    mutant("M-data-root gate skipped (LC4)", ["V23"], (cs, "check_split_surface", noop))
    mutant("M-item_path_mismatch (entry paths trusted, LCC1)", ["V25"],
           (cc, "require_item_paths", noop))
    mutant("M-index guard dropped (path comparison only, C4-N2)", ["V27"],
           (cc, "item_path_problems", _paths_without_index_guard))
    mutant("M-rounding int(x + 0.5) in the reader (S2 M2)", ["V2"],
           (cs, "resized_size", lambda w, h: (512, max(1, int(h * 512 / w + 0.5))) if w >= h
            else (max(1, int(w * 512 / h + 0.5)), 512)))
    assert VC.gate_sources is real_gate_sources


LABELS = {
    "V1": "V1 a capped synthetic cache verifies; the four values print, each labelled",
    "V2": "V2 each special source digest is the real core_preprocess window",
    "V3": "V3 the window is exactly the non-padding region",
    "V4": "V4 stems and manifest_index equal adapters.list_split_stems",
    "V5": "V5 a source JPEG changed after generation -> source_file_mismatch",
    "V6": "V6 a changed item is refused and no record is written",
    "V7": "V7 the record sits outside the checksum, with its counts",
    "V8": "V8 --split test without --confirm-test-split -> refused before any read",
    "V9": "V9 a val cache verified as test -> split_mismatch, before any TEST read",
    "V10": "V10 a recorded file name the listing lacks -> sources_mismatch",
    "V11": "V11 a recorded pixel digest that is not the window -> source_pixels_mismatch",
    "V12": "V12 a cell digest that does not recompute -> cell_digest_mismatch",
    "V13": "V13 an embedded pin whose sha does not recompute -> pin_invalid",
    "V14": "V14 a cache root containing 'test' -> cache_root_test_named, nothing read (LC6)",
    "V15": "V15 an item whose shape is not its source's -> cache_entry_corrupt",
    "V16": "V16 a provisional cache outside the student image -> image_digest_mismatch (LC2)",
    "V17": "V17 a provisional cache on an unpinned stack -> official_dependency_version_mismatch",
    "V18": "V18 the same provisional cache verifies in the student image on the pinned stack",
    "V19": "V19 reader == core_preprocess on a malformed (UNDEFINED) EXIF orientation",
    "V20": "V20 a manifest with a duplicate key, NaN or 1e999 -> manifest_schema",
    "V22": "V22 a re-sealed manifest without the fog_s5 cell -> cell_digest_mismatch (LC13)",
    "V23": "V23 a VAL root reporting a TEST surface -> test_split_present, nothing read (LC4)",
    "V24": "V24 a 1.0.0 manifest -> manifest_schema",
    "V25": "V25 a re-sealed entry at its stem path, not its index path -> item_path_mismatch",
    "V26": "V26 a verified cache: every item at its index path; no created name holds the marker",
    "V27": "V27 index 100000, no source, or a text index -> item_path_mismatch, nothing read",
}


def main() -> int:
    print("=" * 100)
    print("S3 CORRUPTION-CACHE VERIFICATION SMOKE — real core_preprocess, synthetic data only")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="corrverify_"))
    try:
        setup(work)
        for name in CASES:
            ok, detail = run(name)
            results.append((LABELS[name], ok, detail))
        mutations()
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        results.append(("FATAL", False, traceback.format_exc(limit=3)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print()
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail and not ok:
            print(f"         {detail[:200]}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
