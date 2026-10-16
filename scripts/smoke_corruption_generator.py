#!/usr/bin/env python3
"""S2 — the corruption-cache generator (L-CORR B-2). Torch-free and synthetic.

A synthetic TRAIN/VAL-only data root (5,367 TRAIN and 846 VAL names; the six VAL images that
`--max-images 6` selects are the special cases) and a throwaway git repository holding copies of the
generator script and the protocol stand in for the dataset and the checkout. No file or folder this
smoke creates has "test" in its name: every TEST refusal runs on never-created paths, before any
image is read, under a zero-call tracer. torch and src.data stay out of sys.modules (T1, T2).

  G  source reader       H  grid           D  determinism        R  resume
  P  pin                 I  index          T  torch-free
  L  LC4 split surfaces, LC5 manifest_exists, LC7 the 32-px floor, LC8 repair
  X  every other refusal code of the generator, each reached with every earlier gate passing
  M  mutations: each refusal gate disabled, and the plan's eight, each killed by its named case

Run:  python -B scripts/smoke_corruption_generator.py
"""
from __future__ import annotations

import ast
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
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import scripts.build_corruption_cache as B  # noqa: E402
import src.corruption_cache as cc  # noqa: E402
import src.corruption_source as cs  # noqa: E402
import src.vendor.imagecorruptions as V  # noqa: E402
from src import corruption_roles as R  # noqa: E402

FORBIDDEN = ("torch", "torchvision", "src.data")
TORCH_FREE_BEFORE = not any(m in sys.modules for m in FORBIDDEN)
SUBSET = (0, 141, 282, 423, 564, 705)          # deterministic_subset(846, 6)
NON_NOISE = ("motion_blur", "jpeg_compression", "brightness", "fog")
CASES: dict = {}
results: list[tuple[str, bool, str]] = []
S: dict = {}                                    # shared state built in setup()
_SEQ = itertools.count()


# ---------------------------------------------------------------- fixtures (S3 imports these)
def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=corr-fixture", "-c", "user.email=corr-fixture@invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), capture_output=True, text=True, check=False,
        env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(repo.parent)))
    if proc.returncode != 0:
        raise AssertionError(f"fixture git {args} failed ({proc.returncode}): {proc.stderr[:200]}")
    return proc.stdout


def make_fixture_repo(root: Path, *, protocol_bytes: bytes | None = None,
                      stray: bool = False) -> Path:
    """A throwaway clean repository (never the project) with the generator script and protocol."""
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    shutil.copyfile(REPO / "scripts" / "build_corruption_cache.py",
                    root / "scripts" / "build_corruption_cache.py")
    (root / "configs" / "corruption_protocol.json").write_bytes(
        protocol_bytes if protocol_bytes is not None
        else (REPO / "configs" / "corruption_protocol.json").read_bytes())
    _git(root, "init", "-q")
    _git(root, "add", "--", "scripts/build_corruption_cache.py", "configs/corruption_protocol.json")
    _git(root, "commit", "-q", "-m", "fixture")
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root.resolve():
        raise AssertionError("fixture is not its own repository")
    if stray:
        (root / "src").mkdir()
        (root / "src" / "stray.py").write_text("x = 1\n", encoding="utf-8")
    return root


def _jpeg(path: Path, arr: np.ndarray, *, mode: str = "RGB", orientation: int | None = None):
    im = Image.fromarray(arr)
    if mode != "RGB":
        im = im.convert(mode)
    kw = {"quality": 95}
    if orientation is not None:
        exif = Image.Exif()
        exif[cs.EXIF_ORIENTATION] = orientation
        kw["exif"] = exif
    im.save(path, format="JPEG", **kw)


#: The six images --max-images 6 selects: (stored w, h), JPEG mode, EXIF orientation and the
#: resized (h, w).
SPECIALS = {0: ((80, 1024), "RGB", 6, (40, 512)),       # EXIF 6: displayed 1024 x 80
            141: ((1024, 70), "L", None, (35, 512)),    # grayscale JPEG
            282: ((1024, 66), "CMYK", None, (33, 512)),  # CMYK JPEG
            423: ((1024, 61), "RGB", None, (30, 512)),  # 30.5 -> 30: below the 32-px floor (LC7)
            564: ((1024, 63), "RGB", None, (32, 512)),  # 31.5 -> 32: at the floor
            705: ((65, 1024), "RGB", 8, (32, 512))}     # EXIF 8: 1024 x 65, 32.5 -> 32 (half-even)
SUFFIXES = {7: ".JPG", 9: ".jpeg", 11: ".JPEG"}


def make_val_root(root: Path) -> Path:
    """A TRAIN/VAL-only root: 5,367 TRAIN and 846 VAL names (masks empty: they are only counted),
    VAL images real JPEGs, the six SPECIALS at the --max-images 6 indices, a stray non-image."""
    for split, n in (("train", 5367), ("val", 846)):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(n):
            (root / "annotations" / split / f"syn_{split}_{i:05d}.png").touch()
    for i in range(5367):
        (root / "images" / "train" / f"syn_train_{i:05d}.jpg").touch()
    rs = np.random.RandomState(20261016)
    for i in range(846):
        name = f"syn_val_{i:05d}{SUFFIXES.get(i, '.jpg')}"
        if i in SPECIALS:
            (w, h), mode, orient, _ = SPECIALS[i]
            _jpeg(root / "images" / "val" / name, (rs.rand(h, w, 3) * 255).astype(np.uint8),
                  mode=mode, orientation=orient)
        else:
            _jpeg(root / "images" / "val" / name, (rs.rand(8, 10, 3) * 255).astype(np.uint8))
    (root / "images" / "val" / "readme.txt").write_text("not an image\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------- harness
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
def env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def tracer():
    """Counts every source read and split listing (the zero-call tracer)."""
    calls = []
    real_read, real_list = cs.read_source, cs.list_split

    def read(src):
        calls.append(("read", str(src.path)))
        return real_read(src)

    def listing(root, split):
        calls.append(("list", str(root), split))
        return real_list(root, split)
    with patched(cs, "read_source", read), patched(cc, "read_source", read), \
            patched(cs, "list_split", listing):
        yield calls


def code_of(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except (cc.CorruptionCacheError, cs.CorruptionSourceError) as e:
        return e.code
    except Exception as e:                                   # noqa: BLE001
        return type(e).__name__
    return None


def gen_code(argv, repo_root=None):
    """The refusal code of one generator invocation, run in process (None when it completes)."""
    args = B.build_parser().parse_args(argv)
    return code_of(B.generate_cmd, args, repo_root or S["repo"])


def quiet_main(argv, repo_root=None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = B.main(argv, repo_root=repo_root or S["repo"])
    return rc, out.getvalue(), err.getvalue()


def never(name: str) -> Path:
    """A path under the work dir that no case creates."""
    return S["work"] / "never_created" / name


def smoke_argv(cache_dir, *extra, data_root=None, max_images=6):
    argv = ["--real-run", "--confirm-real-run", "--split", "val", "--cache-status", "smoke",
            "--data-root", str(data_root or S["data"]), "--cache-dir", str(cache_dir)]
    if max_images is not None:
        argv += ["--max-images", str(max_images)]
    return argv + list(extra)


def provisional_argv(cache_dir, *extra, pin=None, data_root=None):
    return ["--real-run", "--confirm-real-run", "--split", "val", "--cache-status", "provisional",
            "--data-root", str(data_root or never("val_root")), "--cache-dir", str(cache_dir),
            "--expected-pin-sha256", pin or S["pin_val"]["pin_sha256"], *extra]


def official_argv(cache_dir, *extra, pin=None, data_root=None):
    return ["--real-run", "--confirm-real-run", "--split", "test", "--confirm-test-split",
            "--cache-status", "official", "--data-root", str(data_root or never("tday_root")),
            "--cache-dir", str(cache_dir), "--expected-pin-sha256",
            pin or S["pin_test"]["pin_sha256"], *extra]


DIGEST = {"PLANTSEG_IMAGE_DIGEST": cc.STUDENT_IMAGE_DIGEST}


def capped_sources():
    listing = cs.list_split(S["data"], "val")
    return [listing[i] for i in cs.deterministic_subset(len(listing), 6)]


def library_build(root, **kw):
    return cc.build_cache(root, capped_sources(), split="val", status="smoke", pin=S["pin_val"],
                          expected_images=None, repo_root=S["repo"], **kw)


def crash_then(root, k):
    """A capped library build that crashes after k journalled items (R1's state)."""
    try:
        library_build(root, _fail_after_items=k)
    except RuntimeError as e:
        return str(e)
    return "no crash"


def setup(work: Path) -> None:
    S["work"] = work
    t = time.time()
    S["data"] = make_val_root(work / "data")
    S["repo"] = make_fixture_repo(work / "repo")
    S["caches"] = work / "caches"
    S["caches"].mkdir()
    S["pin_val"] = cc.build_pin("val", repo_root=S["repo"])
    S["pin_test"] = cc.build_pin("test", repo_root=S["repo"])
    rc, out, err = quiet_main(smoke_argv(S["caches"] / "d1a"))
    S["d1_rc"], S["d1_err"] = rc, err
    S["m1"] = cc.load_manifest(S["caches"] / "d1a") if rc == 0 else None
    print(f"setup {time.time() - t:.1f}s; first capped cache rc={rc}")


# ---------------------------------------------------------------- G. source reader
@case("G1")
def g1():
    srcs = cs.list_split(S["data"], "val")
    names = sorted((p for p in (S["data"] / "images" / "val").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg")), key=lambda p: p.name)
    return (len(srcs) == 846 and [s.image_id for s in srcs] == [p.stem for p in names]
            and [s.manifest_index for s in srcs] == list(range(846))
            and {srcs[i].path.suffix for i in SUFFIXES} == set(SUFFIXES.values())
            and "readme" not in {s.image_id for s in srcs}), f"{len(srcs)} sources by p.name"


@case("G2")
def g2():
    (S["work"] / "g2_root").mkdir(exist_ok=True)
    return code_of(cs.list_split, S["work"] / "g2_root", "val") == "split_dir_missing", \
        "images/val absent"


@case("G3")
def g3():
    root = S["work"] / "g3_root"
    if not root.exists():
        (root / "images" / "val").mkdir(parents=True)
        for i in range(845):
            (root / "images" / "val" / f"n_{i:04d}.jpg").touch()
    return code_of(cs.list_split, root, "val") == "split_count_mismatch", "845 names"


def _special(i):
    return cs.read_source(cs.list_split(S["data"], "val")[i])


@case("G4")
def g4():
    a, b = _special(0).record, _special(705).record
    return (a["exif_orientation"] == 6 and a["original_size"] == [80, 1024]
            and a["resized_shape"] == [40, 512] and b["exif_orientation"] == 8
            and b["resized_shape"] == [32, 512]), f"{a['resized_shape']} {b['resized_shape']}"


@case("G5")
def g5():
    gl, cm = _special(141), _special(282)
    return (gl.record["source_mode"] == "L" and cm.record["source_mode"] == "CMYK"
            and all(x.pixels.dtype == np.uint8 and x.pixels.ndim == 3 and x.pixels.shape[2] == 3
                    for x in (gl, cm))), f"{gl.record['source_mode']} {cm.record['source_mode']}"


@case("G6")
def g6():
    got = [cs.resized_size(1024, 61), cs.resized_size(1024, 63), cs.resized_size(1024, 65),
           cs.resized_size(65, 1024)]
    return got == [(512, 30), (512, 32), (512, 32), (32, 512)], str(got)


@case("G7")
def g7():
    p = cs.list_split(S["data"], "val")[423].path
    with Image.open(p) as im:
        rgb = im.convert("RGB")
        bil = np.asarray(rgb.resize((512, 30), Image.BILINEAR), dtype=np.uint8)
        near = np.asarray(rgb.resize((512, 30), Image.NEAREST), dtype=np.uint8)
    px = _special(423).pixels
    return np.array_equal(px, bil) and not np.array_equal(px, near), "PIL BILINEAR, not NEAREST"


@case("G8")
def g8():
    ok = True
    for i in SUBSET:
        si = _special(i)
        ok &= (si.record["source_file_sha256"]
               == hashlib.sha256(Path(cs.list_split(S["data"], "val")[i].path).read_bytes())
               .hexdigest()
               and si.record["resized_pixel_sha256"] == hashlib.sha256(si.pixels.tobytes())
               .hexdigest())
    return ok, "file and resized-pixel digests recompute for the six"


# ---------------------------------------------------------------- H. grid
@case("H1")
def h1():
    m = S["m1"]
    return ([c["cell_id"] for c in m["cells"]] == list(R.CELL_IDS)
            and all(c["items"] == 6 for c in m["cells"])
            and all(c["role"] == R.CELLS[i].role for i, c in enumerate(m["cells"]))), \
        f"{len(m['cells'])} cells"


@case("H2")
def h2():
    m = S["m1"]
    keys = {(e["image_id"], e["corruption_id"], e["severity"]) for e in m["entries"]}
    return len(m["entries"]) == 144 == len(keys), f"{len(m['entries'])} entries"


@case("H3")
def h3():
    root = S["caches"] / f"h3_{next(_SEQ)}"
    rc, _, err = quiet_main(smoke_argv(root))
    if rc != 0:
        return False, err.strip()[-120:]
    m = cc.load_manifest(root)
    return (not any(e["corruption_id"] == "gaussian_noise" and e["severity"] == 5
                    for e in m["entries"])
            and not (root / "gaussian_noise" / "s5").exists()), "no gaussian_noise s5 item"


@case("H4")
def h4():
    return all(e["item_seed"] == V.item_seed(e["image_id"], e["corruption_id"], e["severity"])
               for e in S["m1"]["entries"]), "item_seed(image_id, corruption, severity)"


@case("H5")
def h5():
    root, by_id = S["caches"] / "d1a", {s.image_id: s for s in cs.list_split(S["data"], "val")}
    bad = 0
    for e in S["m1"]["entries"]:
        si = cs.read_source(by_id[e["image_id"]])
        arr, _ = cc.corrupt_source(si.pixels, e["image_id"], e["corruption_id"], e["severity"],
                                   lift_floor=cc.floor_lifted(si.record))
        bad += not np.array_equal(arr, np.load(root / e["cache_relative_path"]))
    return bad == 0, f"144 items recomputed, {bad} differ"


@case("H6")
def h6():
    return all(cc.item_verifies(S["caches"] / "d1a", e) for e in S["m1"]["entries"]), "144 verify"


@case("H7")
def h7():
    m = S["m1"]
    by = {}
    for e in m["entries"]:
        by.setdefault((e["corruption_id"], e["severity"]), []).append(e)
    ok = all(c["cell_sha256"] == cc.cell_digest(by[(c["corruption"], c["severity"])])
             for c in m["cells"])
    inf = [c["cell_sha256"] for c in m["cells"] if c["role"] == "inferential"]
    return (ok and len(inf) == 15 and m["inferential_cells_sha256"]
            == hashlib.sha256(cc.canonical_json_bytes(inf)).hexdigest()), "24 cells, 15 inferential"


@case("H8")
def h8():
    m = S["m1"]
    body = {k: v for k, v in m.items()
            if k not in ("canonical_manifest_sha256", "versions", "generation")}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False).encode("utf-8")
    return (hashlib.sha256(canon).hexdigest() == m["canonical_manifest_sha256"]
            and {"versions", "generation"} <= set(m)), "versions and generation outside"


# ---------------------------------------------------------------- D. determinism
@case("D1")
def d1():
    rc, _, err = quiet_main(smoke_argv(S["caches"] / "d1b"))
    if rc != 0 or S["m1"] is None:
        return False, (S["d1_err"] + err).strip()[-120:]
    b = cc.load_manifest(S["caches"] / "d1b")
    return b["canonical_manifest_sha256"] == S["m1"]["canonical_manifest_sha256"], \
        S["m1"]["canonical_manifest_sha256"][:16]


@case("D2")
def d2():
    root = S["caches"] / f"d2_{next(_SEQ)}"
    rc, _, err = quiet_main(smoke_argv(root, "--workers", "3"))
    if rc != 0:
        return False, err.strip()[-120:]
    return cc.load_manifest(root)["canonical_manifest_sha256"] == \
        S["m1"]["canonical_manifest_sha256"], "workers 3 == workers 1"


@case("D3")
def d3():
    m = library_build(S["caches"] / f"d3_{next(_SEQ)}", _reverse=True)
    return m["canonical_manifest_sha256"] == S["m1"]["canonical_manifest_sha256"], \
        "reversed source order"


@case("D4")
def d4():
    m = json.loads(json.dumps(S["m1"]))
    m["versions"] = {**m["versions"], "platform": "injected-host", "python": "3.11.99"}
    m["generation"] = {"injected": True}
    return cc.canonical_manifest_sha256(m) == S["m1"]["canonical_manifest_sha256"], \
        "an injected runtime record leaves the checksum"


@case("D5")
def d5():
    np.random.seed(7)
    before = np.random.get_state()
    library_build(S["caches"] / f"d5_{next(_SEQ)}")
    after = np.random.get_state()
    return (before[0] == after[0] and np.array_equal(before[1], after[1])
            and before[2:] == after[2:]), "global RNG state restored"


# ---------------------------------------------------------------- R. resume
@case("R1")
def r1():
    root = S["caches"] / "r_crash"
    msg = crash_then(root, 50)
    j = cc.read_journal(root)
    return ("simulated crash" in msg and not (root / cc.MANIFEST_NAME).exists()
            and len(j["items"]) == 50), f"{len(j['items'])} journalled, no manifest"


@case("R2")
def r2():
    root = S["caches"] / f"r2_{next(_SEQ)}"
    crash_then(root, 50)
    m = library_build(root, resume=True)
    items = m["generation"]["items"]
    S["r2_checksum"] = m["canonical_manifest_sha256"]
    return items == {"generated": 94, "regenerated": 0, "kept": 50}, str(items)


@case("R3")
def r3():
    return S.get("r2_checksum") == S["m1"]["canonical_manifest_sha256"], "resumed == uninterrupted"


@case("R4")
def r4():
    root = S["caches"] / f"r4_{next(_SEQ)}"
    crash_then(root, 50)
    victim = next(iter(cc.read_journal(root)["items"].values()))
    path = root / victim["cache_relative_path"]
    np.save(path, (np.load(path) + 1).astype(np.uint8), allow_pickle=False)
    m = library_build(root, resume=True)
    items = m["generation"]["items"]
    arr = cc.load_cached(root, m, victim["image_id"], victim["corruption_id"], victim["severity"])
    return (items == {"generated": 94, "regenerated": 1, "kept": 49} and arr.dtype == np.uint8
            and m["canonical_manifest_sha256"] == S["m1"]["canonical_manifest_sha256"]), str(items)


@case("R5")
def r5():
    root = S["caches"] / f"r5_{next(_SEQ)}"
    crash_then(root, 10)
    other = {**S["pin_val"], "pin_sha256": "0" * 64}
    code = code_of(cc.build_cache, root, capped_sources(), split="val", status="smoke", pin=other,
                   expected_images=None, resume=True, repo_root=S["repo"])
    return code == "resume_pin_mismatch", str(code)


# ---------------------------------------------------------------- P. pin
@case("P1")
def p1():
    with tracer() as calls:
        rc, out, err = quiet_main(["--print-pin", "--split", "val"])
    pin = json.loads(out) if rc == 0 else {}
    return (rc == 0 and not calls
            and set(pin) == {"schema", "script", "commit", "library_versions", "parameters",
                             "pin_sha256"}
            and pin["parameters"]["cells"] == list(R.CELL_IDS) and pin == S["pin_val"]), \
        f"rc={rc}, data reads={len(calls)}"


@case("P2")
def p2():
    pin = S["pin_val"]
    body = {k: v for k, v in pin.items() if k != "pin_sha256"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False).encode("utf-8")
    return (hashlib.sha256(canon).hexdigest() == pin["pin_sha256"]
            and S["m1"]["pin"] == pin), pin["pin_sha256"][:16]


@case("P3")
def p3():
    blob = _git(S["repo"], "rev-parse", "HEAD:scripts/build_corruption_cache.py").strip()
    head = _git(S["repo"], "rev-parse", "HEAD").strip()
    pin = S["pin_val"]
    return (pin["script"]["git_blob"] == blob and pin["commit"] == head), blob[:12]


# ---------------------------------------------------------------- I. index
@case("I1")
def i1():
    m = S["m1"]
    idx = cc.cache_index(m)
    return all(idx.entries[(e["image_id"], e["corruption_id"], e["severity"])] is e
               for e in m["entries"]) and len(idx.sources) == 6, "index == linear scan"


@case("I2")
def i2():
    fake = {"entries": [{"image_id": f"i{i:05d}", "corruption_id": c, "severity": s}
                        for i in range(1561) for c, s in cc.full_grid(R.load())],
            "sources": []}
    t0 = time.perf_counter()
    idx = cc.cache_index(fake)
    t1 = time.perf_counter()
    keys = [(f"i{i % 1561:05d}", "fog", 1 + i % 5) for i in range(10000)]
    for k in keys:
        idx.entries[k]
    t2 = time.perf_counter()
    return (len(fake["entries"]) == 37464 and t1 - t0 < 1.0 and t2 - t1 < 0.1), \
        f"build {t1 - t0:.3f}s, 10,000 lookups {t2 - t1:.4f}s"


@case("I3")
def i3():
    return code_of(cc.load_cached, S["caches"] / "d1a", S["m1"], "syn_val_99999", "fog", 1) == \
        "cache_entry_missing", "unknown key refused"


# ---------------------------------------------------------------- T. torch-free
def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@case("T1")
def t1():
    mods = [REPO / "src" / "corruption_cache.py", REPO / "src" / "corruption_source.py",
            REPO / "src" / "corruption_roles.py", REPO / "scripts" / "build_corruption_cache.py"]
    bad = [f"{p.name}: {n}" for p in mods for n in _imports(p)
           if n.split(".")[0] in ("torch", "torchvision") or n.startswith("src.data")]
    return TORCH_FREE_BEFORE and not bad, f"before: {TORCH_FREE_BEFORE}; imports {bad or 'clean'}"


@case("T2")
def t2():
    return not any(m in sys.modules for m in FORBIDDEN), "after all work"


# ---------------------------------------------------------------- L. LC4, LC5, LC7, LC8
@case("L4a")
def l4a():
    """VAL root: the isolation module (loaded by file path) refuses a never-created root."""
    with tracer() as calls:
        code = gen_code(smoke_argv(never("cache_l4a"), data_root=never("val_root_l4a")))
    return (code == "data_root_missing" and not calls and "src.data" not in sys.modules
            and not never("cache_l4a").exists()), f"code={code}, reads={len(calls)}"


_REAL_LOAD_ISOLATION = cs.load_isolation


def _fake_surface_isolation():
    real = _REAL_LOAD_ISOLATION()

    def lexists(p):
        return str(p).replace("\\", "/").endswith("/images/test") or os.path.lexists(p)
    real.os = types.SimpleNamespace(path=types.SimpleNamespace(lexists=lexists, join=os.path.join,
                                                               splitext=os.path.splitext),
                                    scandir=os.scandir)
    return real


@case("L4b")
def l4b():
    """VAL root reporting a TEST surface (simulated: nothing is created) -> test_split_present."""
    with tracer() as calls, patched(cs, "load_isolation", _fake_surface_isolation):
        code = gen_code(smoke_argv(never("cache_l4b"), max_images=1))
    return code == "test_split_present" and not calls, f"code={code}, reads={len(calls)}"


@case("L4c")
def l4c():
    """TEST surface: an official run refuses a never-created root before any read."""
    with tracer() as calls, env(**DIGEST), patched(cc, "SEVERITY5_REFERENCE_VALIDATED", True):
        code = gen_code(official_argv(never("tday_a") / cc.TDAY_CACHE_ROOT_NAME))
    return code == "data_root_missing" and not calls, f"code={code}, reads={len(calls)}"


@case("L4d")
def l4d():
    """TEST surface: a root without images/test; only that path is probed, nothing is read."""
    root = S["work"] / "tday_data_root"
    root.mkdir(exist_ok=True)
    args = B.build_parser().parse_args(official_argv(never("tday_b") / cc.TDAY_CACHE_ROOT_NAME,
                                                     data_root=root))
    with tracer() as calls, env(**DIGEST), patched(cc, "SEVERITY5_REFERENCE_VALIDATED", True):
        try:
            B.generate_cmd(args, S["repo"])
            code, msg = None, ""
        except (cc.CorruptionCacheError, cs.CorruptionSourceError) as e:
            code, msg = e.code, str(e)
    return (code == "split_dir_missing" and msg.endswith(str(root / "images" / "test"))
            and not calls and not (root / "images").exists()), f"code={code}, reads={len(calls)}"


@case("L5")
def l5():
    root = S["caches"] / "d1a"
    a = gen_code(smoke_argv(root, data_root=never("val_root_l5")))
    b = gen_code(smoke_argv(root, "--resume", data_root=never("val_root_l5")))
    return a == b == "manifest_exists", f"{a} / {b}"


@case("L7")
def l7():
    m = S["m1"]
    rec = {s["manifest_index"]: s for s in m["sources"]}
    below, at = rec[423], rec[564]
    tiny = np.zeros((30, 512, 3), np.uint8)
    refused = code_of(V.apply_corruption, tiny, "fog", 1, image_id="x")
    return (m["floor_lifted"] == [below["image_id"]] and below["resized_shape"] == [30, 512]
            and at["resized_shape"] == [32, 512]
            and sum(e["image_id"] == below["image_id"] for e in m["entries"]) == 24
            and refused == "CorruptionInputError"), f"floor_lifted={m['floor_lifted']}"


def _repair_root():
    root = S["caches"] / f"repair_{next(_SEQ)}"
    rc, _, err = quiet_main(smoke_argv(root))
    assert rc == 0, err
    return root, cc.load_manifest(root)


def _repair_argv(root, entry, report):
    return ["--real-run", "--confirm-real-run", "--split", "val", "--data-root", str(S["data"]),
            "--cache-dir", str(root), "--repair", entry["image_id"], entry["corruption_id"],
            str(entry["severity"]), "--fault-report", str(report)]


@case("L8a")
def l8a():
    root, m = _repair_root()
    e = next(x for x in m["entries"] if x["corruption_id"] == "fog" and x["severity"] == 5)
    path = root / e["cache_relative_path"]
    np.save(path, (np.load(path) + 3).astype(np.uint8), allow_pickle=False)
    report = S["work"] / "fault_l8a.txt"
    report.write_text("cache_file_hash_mismatch at scoring\n", encoding="utf-8")
    before = (root / cc.MANIFEST_NAME).read_bytes()
    rc, out, err = quiet_main(_repair_argv(root, e, report))
    rec = json.loads(out) if rc == 0 else {}
    arr = cc.load_cached(root, m, e["image_id"], "fog", 5)
    return (rc == 0 and rec.get("outcome") == "repaired" and arr.dtype == np.uint8
            and (root / cc.MANIFEST_NAME).read_bytes() == before), f"rc={rc} {err.strip()[-80:]}"


@case("L8b")
def l8b():
    root, m = _repair_root()
    e = next(x for x in m["entries"] if x["corruption_id"] == "brightness" and x["severity"] == 2)
    report = S["work"] / "fault_l8b.txt"
    report.write_text("suspect item\n", encoding="utf-8")
    real = cc.corrupt_source

    def drifted(*a, **k):
        arr, seed = real(*a, **k)
        return (arr ^ np.uint8(1)), seed
    before = (root / e["cache_relative_path"]).read_bytes()
    with patched(cc, "corrupt_source", drifted):
        rc, _, err = quiet_main(_repair_argv(root, e, report))
    return (rc == 3 and "[repair_digest_mismatch]" in err
            and (root / e["cache_relative_path"]).read_bytes() == before), f"rc={rc}"


@case("L8c")
def l8c():
    root = S["caches"] / "d1a"
    e = S["m1"]["entries"][0]
    rc, _, err = quiet_main(_repair_argv(root, e, never("fault_report.txt")))
    return rc == 2 and "[repair_fault_report_missing]" in err, f"rc={rc}"


# ---------------------------------------------------------------- X. refusal codes
def _x(code, argv, *, ctx=None, repo_root=None):
    with tracer() as calls, (ctx or contextlib.nullcontext()):
        got = gen_code(argv, repo_root)
    reads = [c for c in calls if c[0] == "read"]
    return got == code and not reads, f"code={got}, reads={len(reads)}"


@case("X01")
def x01():
    root = S["work"] / "x01_parent_never" / "Corr_TeSt_root"
    ok, d = _x("cache_root_test_named", smoke_argv(root, data_root=never("val_root_x01")))
    return ok and not root.parent.exists(), d


@case("X02")
def x02():
    return _x("tday_cache_root_required", official_argv(never("corr_cache_other")),
              ctx=env(**DIGEST))


@case("X03")
def x03():
    argv = [a for a in official_argv(never("x03") / cc.TDAY_CACHE_ROOT_NAME)
            if a != "--confirm-test-split"]
    return _x("test_split_unconfirmed", argv)


@case("X04")
def x04():
    argv = official_argv(never("x04") / cc.TDAY_CACHE_ROOT_NAME)
    argv[argv.index("official")] = "provisional"
    return _x("test_split_requires_official", argv)


@case("X05")
def x05():
    argv = provisional_argv(never("x05"))
    argv[argv.index("provisional")] = "official"
    return _x("official_requires_test_split", argv)


@case("X06")
def x06():
    return _x("max_images_requires_smoke", provisional_argv(never("x06"), "--max-images", "6"))


@case("X07")
def x07():
    argv = provisional_argv(never("x07"))
    i = argv.index("--expected-pin-sha256")
    del argv[i:i + 2]
    return _x("expected_pin_required", argv)


@case("X08")
def x08():
    return _x("expected_pin_format", provisional_argv(never("x08"), pin="ABC123"))


@case("X09")
def x09():
    with env(PLANTSEG_IMAGE_DIGEST="sha256:" + "f" * 64):
        return _x("image_digest_mismatch", provisional_argv(never("x09")))


@case("X10")
def x10():
    return _x("severity5_reference_unvalidated",
              official_argv(never("x10") / cc.TDAY_CACHE_ROOT_NAME), ctx=env(**DIGEST))


@case("X11")
def x11():
    nogit = S["work"] / "nogit"
    if not nogit.exists():
        (nogit / "scripts").mkdir(parents=True)
        (nogit / "configs").mkdir()
        shutil.copyfile(REPO / "scripts" / "build_corruption_cache.py",
                        nogit / "scripts" / "build_corruption_cache.py")
        shutil.copyfile(REPO / "configs" / "corruption_protocol.json",
                        nogit / "configs" / "corruption_protocol.json")
    with env(GIT_CEILING_DIRECTORIES=str(S["work"]), **DIGEST):
        return _x("git_unavailable", provisional_argv(never("x11")), repo_root=nogit)


@case("X12")
def x12():
    return _x("governed_paths_dirty", provisional_argv(never("x12")), ctx=env(**DIGEST),
              repo_root=S["repo_dirty"])


@case("X13")
def x13():
    return _x("protocol_invalid", smoke_argv(never("x13"), data_root=never("val_root_x13")),
              repo_root=S["repo_badproto"])


@case("X14")
def x14():
    bent = {**dict(V.SEVERITIES_BY_CORRUPTION), "gaussian_noise": (1, 2, 3, 4, 5)}
    return _x("protocol_vendor_disagree",
              smoke_argv(never("x14"), data_root=never("val_root_x14")),
              ctx=patched(cc, "SEVERITIES_BY_CORRUPTION", bent))


@case("X15")
def x15():
    return _x("pin_mismatch", provisional_argv(never("x15"), pin="0" * 64), ctx=env(**DIGEST))


@case("X16")
def x16():
    root = S["caches"] / "x16"
    root.mkdir(exist_ok=True)
    (root / "stray.bin").write_bytes(b"x")
    return _x("cache_root_not_empty", smoke_argv(root, data_root=never("val_root_x16")))


@case("X17")
def x17():
    return _x("resume_without_journal",
              smoke_argv(never("x17"), "--resume", data_root=never("val_root_x17")))


@case("X18")
def x18():
    return _x("max_images_out_of_range", smoke_argv(never("x18"), max_images=0))


@case("X19")
def x19():
    return code_of(cs.check_split_surface, S["data"], "train") == "unknown_split", "train refused"


@case("X20")
def x20():
    bad = S["work"] / "bad.jpg"
    bad.write_bytes(b"\xff\xd8 not a jpeg")
    return code_of(cs.read_source, cs.Source(0, "bad", bad)) == "source_unreadable", "corrupt JPEG"


@case("X21")
def x21():
    a = code_of(cc.generate, never("x21a"), {"i": np.zeros((32, 32, 3), np.uint8)},
                corruptions=("fog",), severities=(1,), status="final")
    b = code_of(cc.build_cache, never("x21b"), [], split="val", status="final",
                pin=S["pin_val"], expected_images=None, repo_root=S["repo"])
    return a == b == "unknown_status" and not never("x21a").exists(), f"{a} / {b}"


@case("X22")
def x22():
    """Also with 3 spawned workers: a refusal raised in a worker keeps its code."""
    codes = []
    for workers in (1, 3):
        root = S["caches"] / f"x22_{next(_SEQ)}"
        crash_then(root, 30)
        jp = cc.journal_path(root)
        lines = jp.read_bytes().split(b"\n")
        for k, raw in enumerate(lines):
            if raw.startswith(b'{"kind":"source"'):
                rec = json.loads(raw)
                rec["record"]["resized_pixel_sha256"] = "0" * 64
                lines[k] = cc.canonical_json_bytes(rec)
                break
        jp.write_bytes(b"\n".join(lines))
        codes.append(code_of(library_build, root, resume=True, workers=workers))
    return codes == ["resume_source_mismatch"] * 2, str(codes)


@case("X23")
def x23():
    root = S["caches"] / f"x23_{next(_SEQ)}"
    crash_then(root, 30)
    jp = cc.journal_path(root)
    lines = jp.read_bytes().split(b"\n")
    for k, raw in enumerate(lines):
        if raw.startswith(b'{"entry"') or b'"kind":"item"' in raw:
            rec = json.loads(raw)
            rec["entry"]["cached_file_sha256"] = "0" * 64       # the item no longer verifies
            rec["entry"]["corrupted_pixel_sha256"] = "1" * 64   # and regenerates to other bytes
            lines[k] = cc.canonical_json_bytes(rec)
            break
    jp.write_bytes(b"\n".join(lines))
    code = code_of(library_build, root, resume=True)
    return code == "nondeterministic_item", str(code)


@case("X24")
def x24():
    root = S["caches"] / "d1a"
    e = S["m1"]["entries"][0]
    report = S["work"] / "fault_x24.txt"
    report.write_text("x\n", encoding="utf-8")
    argv = _repair_argv(root, e, report)
    argv[argv.index("val")] = "test"
    argv.append("--confirm-test-split")
    rc, _, err = quiet_main(argv)
    return rc == 2 and "[repair_split_mismatch]" in err, f"rc={rc}"


@case("X25")
def x25():
    def broken():
        raise V.provenance.VendorIntegrityError("simulated: the vendored numerics changed")
    return _x("vendor_checksum_mismatch", smoke_argv(never("x25"), data_root=never("vr_x25")),
              ctx=patched(B, "verify_vendor_integrity", broken))


# ---------------------------------------------------------------- M. mutations
def noop(*a, **k):
    return None


def mutant(label, targets, *patches):
    with contextlib.ExitStack() as stack:
        for obj, attr, value in patches:
            stack.enter_context(patched(obj, attr, value))
        outcomes = {t: run(t) for t in targets}
    killed = not any(ok for ok, _ in outcomes.values())
    why = "; ".join(f"{t}: {d[:90]}" for t, (_, d) in outcomes.items())
    print(f"MUTANT {label} {'killed' if killed else 'SURVIVED'} by {', '.join(targets)} [{why}]")
    results.append((f"{label} killed by {', '.join(targets)}", killed, why))


def _subset_without_guard(n_total, n_samples):
    step = n_total // n_samples
    return [i * step for i in range(n_samples)]


def _decode_unguarded(raw, name):
    from PIL import ImageOps
    with Image.open(io.BytesIO(raw)) as im:
        return (im.getexif().get(cs.EXIF_ORIENTATION), im.mode, im.size,
                ImageOps.exif_transpose(im).convert("RGB"))


def mutations():
    real_load_roles, real_canon = cc.load_roles, cc.canonical_manifest_sha256
    real_full_grid, real_corrupt = cc.full_grid, cc.corrupt_source
    counter = itertools.count()
    # every refusal gate disabled
    mutant("M-cache_root_test_named", ["X01"], (cc, "refuse_test_named", noop))
    mutant("M-tday_cache_root_required", ["X02"], (B, "gate_tday_root", noop))
    mutant("M-test_split_unconfirmed", ["X03"], (B, "gate_test_confirm", noop))
    mutant("M-test_split_requires_official", ["X04"], (B, "gate_test_official", noop))
    mutant("M-official_requires_test_split", ["X05"], (B, "gate_official_test", noop))
    mutant("M-max_images_requires_smoke", ["X06"], (B, "gate_max_images", noop))
    mutant("M-expected_pin_required", ["X07"], (B, "gate_expected_pin_present", noop))
    mutant("M-expected_pin_format", ["X08"], (B, "gate_expected_pin_format", noop))
    mutant("M-image_digest_mismatch", ["X09"], (B, "gate_image_digest", noop))
    mutant("M-severity5_reference_unvalidated", ["X10"], (cc, "require_severity5_validated", noop))
    mutant("M-git_unavailable", ["X11"], (cc, "governed_porcelain", lambda *a, **k: b""),
           (cc, "git_head", lambda *a, **k: "0" * 40))
    mutant("M-governed_paths_dirty", ["X12"],
           (cc, "require_governed_clean",
            lambda *a, **k: {"governed_paths_clean": True, "worktree_state_sha256": "0" * 64}))
    mutant("M-protocol_invalid", ["X13"],
           (cc, "load_roles", lambda repo_root=None: real_load_roles()))
    mutant("M-protocol_vendor_disagree", ["X14"],
           (cc, "check_protocol_vendor", lambda roles: roles.severities_by_corruption))
    mutant("M-pin_mismatch", ["X15"], (B, "pin_matches", lambda *a: True))
    mutant("M-manifest_exists", ["L5"], (cc, "refuse_manifest_exists", noop))
    mutant("M-cache_root_not_empty", ["X16"], (cc, "refuse_root_not_empty", noop))
    mutant("M-resume_without_journal", ["X17"], (cc, "refuse_resume_without_journal", noop))
    mutant("M-isolation skipped (data_root_missing, test_split_present)", ["L4a", "L4b"],
           (cs, "run_isolation", lambda root: {}))
    mutant("M-data_root_missing (test)", ["L4c"], (cs, "require_data_root", noop))
    mutant("M-split_dir_missing", ["G2", "L4d"], (cs, "require_split_dir", noop))
    mutant("M-split_count_mismatch", ["G3"], (cs, "require_split_count", noop))
    mutant("M-max_images_out_of_range", ["X18"],
           (cs, "deterministic_subset", _subset_without_guard))
    mutant("M-unknown_split", ["X19"], (cs, "require_split", noop))
    mutant("M-source_unreadable", ["X20"], (cs, "_decode", _decode_unguarded))
    mutant("M-unknown_status", ["X21"], (cc, "require_status", noop))
    mutant("M-resume_pin_mismatch", ["R5"], (cc, "journal_matches", lambda *a, **k: True))
    mutant("M-resume_source_mismatch", ["X22"], (cc, "same_source", lambda *a: True))
    mutant("M-nondeterministic_item", ["X23"], (cc, "same_digests", lambda *a: True))
    mutant("M-repair_split_mismatch", ["X24"], (B, "gate_repair_split", noop))
    mutant("M-vendor_checksum_mismatch", ["X25"], (B, "gate_vendor", noop))
    mutant("M-repair_fault_report_missing", ["L8c"], (cc, "require_fault_report", noop))
    mutant("M-repair_digest_mismatch", ["L8b"], (cc, "repair_matches", lambda *a: True))

    # the plan's eight
    def by_enumeration(pixels, image_id, corruption_id, severity, *, lift_floor):
        return real_corrupt(pixels, f"n{next(counter)}", corruption_id, severity,
                            lift_floor=lift_floor)
    mutant("M1 seed by enumeration", ["D3"], (cc, "corrupt_source", by_enumeration))
    mutant("M2 int(x + 0.5)", ["G6"],
           (cs, "resized_size", lambda w, h: (512, max(1, int(h * 512 / w + 0.5))) if w >= h
            else (max(1, int(w * 512 / h + 0.5)), 512)))
    mutant("M3 EXIF skipped", ["G4"],
           (cs, "ImageOps", types.SimpleNamespace(exif_transpose=lambda im: im)))
    mutant("M4 a volatile field in the checksum", ["D4"],
           (cc, "canonical_manifest_sha256",
            lambda m: hashlib.sha256(cc.canonical_json_bytes(
                {k: v for k, v in m.items() if k != "canonical_manifest_sha256"})).hexdigest()))
    mutant("M5 resume regenerates", ["R2"], (cc, "item_verifies", lambda *a: False))
    mutant("M6 journal not verified", ["R4"], (cc, "item_verifies", lambda *a: True))
    mutant("M7 unsorted entries", ["D2", "D3"], (cc, "sorted_entries", lambda e, s: list(e)))
    bent = {**dict(V.SEVERITIES_BY_CORRUPTION), "gaussian_noise": (1, 2, 3, 4, 5)}
    mutant("M8 gaussian_noise s5", ["H3"],
           (cc, "full_grid", lambda roles: real_full_grid(roles) + [("gaussian_noise", 5)]),
           (V, "SEVERITIES_BY_CORRUPTION", bent), (V, "EXCLUDED_CELLS", ()))
    assert cc.canonical_manifest_sha256 is real_canon


LABELS = {
    "G1": "G1 the stem rule: .jpg/.jpeg in any case, sorted by p.name, index = position",
    "G2": "G2 a split without its image directory -> split_dir_missing",
    "G3": "G3 845 VAL names -> split_count_mismatch",
    "G4": "G4 EXIF orientations 6 and 8 are applied before the resize",
    "G5": "G5 modes L and CMYK become uint8 RGB",
    "G6": "G6 the size rule rounds half to even (61 -> 30, 63 -> 32, 65 -> 32)",
    "G7": "G7 PIL BILINEAR, as core_preprocess",
    "G8": "G8 the source digests recompute",
    "H1": "H1 24 cells per image in protocol order",
    "H2": "H2 6 images x 24 = 144 entries, each once",
    "H3": "H3 no gaussian_noise s5 item",
    "H4": "H4 the item seed rule",
    "H5": "H5 every item equals the vendored corruption of its source",
    "H6": "H6 every entry verifies",
    "H7": "H7 cell digests and the inferential digest recompute",
    "H8": "H8 the checksum recomputes and excludes versions and generation",
    "D1": "D1 two runs, one checksum",
    "D2": "D2 workers 1 and 3, one checksum",
    "D3": "D3 reversed source order, one checksum",
    "D4": "D4 an injected runtime record leaves the checksum unchanged",
    "D5": "D5 NumPy's global RNG is unchanged",
    "R1": "R1 a crash after 50 items leaves no manifest and 50 journalled items",
    "R2": "R2 the resume keeps 50 and generates 94, regenerating none",
    "R3": "R3 the resumed cache has the uninterrupted checksum",
    "R4": "R4 an item tampered before the resume is regenerated, same checksum",
    "R5": "R5 a resume under another pin -> resume_pin_mismatch",
    "P1": "P1 --print-pin reads no data and has the four fields plus pin_sha256",
    "P2": "P2 pin_sha256 recomputes; the manifest embeds the pin",
    "P3": "P3 the script blob and commit equal git's",
    "I1": "I1 the index equals a linear scan",
    "I2": "I2 index build < 1 s for 37,464 entries; 10,000 lookups < 0.1 s",
    "I3": "I3 an unknown key -> cache_entry_missing",
    "T1": "T1 torch and src.data absent before; no generator module imports them",
    "T2": "T2 torch and src.data absent after",
    "L4a": "L4a VAL root never created -> data_root_missing from isolation, zero reads (LC4)",
    "L4b": "L4b VAL root reporting a TEST surface -> test_split_present, zero reads (LC4)",
    "L4c": "L4c TEST root never created -> data_root_missing, zero reads (LC4)",
    "L4d": "L4d TEST root without images/test -> split_dir_missing on that path only (LC4)",
    "L5": "L5 a root whose manifest exists -> manifest_exists, with or without --resume (LC5)",
    "L7": "L7 the 30-px image is generated with the floor lifted and listed; 32 px is not (LC7)",
    "L8a": "L8a --repair regenerates a tampered item whose digests match (LC8)",
    "L8b": "L8b --repair with drifted bytes -> repair_digest_mismatch, item kept (LC8)",
    "L8c": "L8c --repair without its fault report -> repair_fault_report_missing (LC8)",
    "X01": "X01 a cache root containing 'test' in any case -> cache_root_test_named (LC6)",
    "X02": "X02 a TEST root not named corr_cache_tday -> tday_cache_root_required (LC6)",
    "X03": "X03 --split test without --confirm-test-split -> test_split_unconfirmed",
    "X04": "X04 --split test, provisional -> test_split_requires_official",
    "X05": "X05 official on val -> official_requires_test_split",
    "X06": "X06 --max-images on a provisional cache -> max_images_requires_smoke",
    "X07": "X07 provisional without --expected-pin-sha256 -> expected_pin_required",
    "X08": "X08 a malformed expected pin -> expected_pin_format",
    "X09": "X09 PLANTSEG_IMAGE_DIGEST not the student image -> image_digest_mismatch",
    "X10": "X10 official while severity 5 is pending -> severity5_reference_unvalidated",
    "X11": "X11 no git repository -> git_unavailable",
    "X12": "X12 an untracked governed file -> governed_paths_dirty",
    "X13": "X13 an invalid protocol file -> protocol_invalid",
    "X14": "X14 protocol and vendor sets disagree -> protocol_vendor_disagree",
    "X15": "X15 a different expected pin -> pin_mismatch",
    "X16": "X16 a non-empty root without --resume -> cache_root_not_empty",
    "X17": "X17 --resume without a journal -> resume_without_journal",
    "X18": "X18 --max-images 0 -> max_images_out_of_range",
    "X19": "X19 split train -> unknown_split",
    "X20": "X20 an unreadable JPEG -> source_unreadable",
    "X21": "X21 an unknown cache status -> unknown_status",
    "X22": "X22 a source changed since its journal line -> resume_source_mismatch (1, 3 workers)",
    "X23": "X23 a regenerated item with other digests -> nondeterministic_item (STOP)",
    "X24": "X24 --repair with another split -> repair_split_mismatch",
    "X25": "X25 vendored numerics changed -> vendor_checksum_mismatch",
}


def main() -> int:
    print("=" * 100)
    print("S2 CORRUPTION-CACHE GENERATOR SMOKE — torch-free, synthetic; no PlantSeg data")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="corrgen_"))
    try:
        setup(work)
        S["repo_dirty"] = make_fixture_repo(work / "repo_dirty", stray=True)
        doc = json.loads((REPO / "configs" / "corruption_protocol.json").read_text("utf-8"))
        doc["corruptions"][1]["excluded_severities"] = []
        doc["corruptions"][1]["descriptive_only_severities"] = [4, 5]
        S["repo_badproto"] = make_fixture_repo(work / "repo_badproto",
                                               protocol_bytes=json.dumps(doc).encode())
        for name in CASES:
            if name == "T2":
                continue
            ok, detail = run(name)
            results.append((LABELS[name], ok, detail))
        mutations()
        ok, detail = run("T2")
        results.append((LABELS["T2"], ok, detail))
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
