#!/usr/bin/env python3
"""Synthetic smoke for the DL-49 aspect-ratio census and the DL-55 teacher suffix census (S4;
scripts/split_censuses.py).

SYNTHETIC ONLY. Small trees of tiny JPEGs are written with PIL under a temporary root whose realpath is
checked to contain no "test" before any CLI call; records fixtures follow the teacher hook's row schema
(src/training/teacher_components.py:356-366). No PlantSeg image, mask or TEST path is read; nothing is written
into the repository; the TEST-path refusals use paths that are never created. The synthetic trees are run
with --synthetic-inputs and --expect-train/--expect-val (smoke mode only; a real run fixes 5,367 and 846).

  A  aspect-ratio: counts; the widest image; strict "above" at 12.6 and 26.5 in integers; long/short for a
     portrait; no pixel decoded (PIL ImageFile.load patched to raise); exactly two folders listed, no recursion;
     byte-identical reruns; refusals (exit 2); an unreadable header (exit 1, nothing written); duplicate stems
  T  teacher-suffix: exact-suffix histograms and both rules; the records checks (rows, iterations, val_images
     in every row, one manifest hash equal to the census's); a subdirectory; the records basename; the retyped
     literals against their sources (ast, no import); byte-identical reruns; refusals; non-finite and
     non-object rows; the records pin
  M  mutations of one rule each, every one killed by the named check

Run (TMPDIR's realpath must contain no "test"):
    env TMPDIR=/tmp/s4scratch PYTHONPATH=<repo> python -B scripts/smoke_split_censuses.py
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PIL import Image, ImageFile  # noqa: E402

from scripts import split_censuses as SC  # noqa: E402
from src.eval import teacher_diag as td  # noqa: E402
from src.eval.artifacts import hash_split_manifest  # noqa: E402
from src.eval.evaluate import ManifestEntry  # noqa: E402

GEN_UTC = "2026-10-09T00:00:00Z"
AR = "synthetic_" + SC.AR_NAME
SUF = "synthetic_" + SC.SUFFIX_NAME
TEACHER_CONFIG = REPO / "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
TEACHER_COMPONENTS = REPO / "src/training/teacher_components.py"
RESULTS: list = []
NORMAL: dict = {}


class Ctx:
    pass


CTX = Ctx()


def record(cid, desc, ok, detail="") -> None:
    RESULTS.append((cid, desc, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid} {desc}" + ("" if ok else f"  -- {str(detail)[:900]}"), flush=True)


@contextlib.contextmanager
def patched(*triples):
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in triples]
    try:
        for obj, name, value in triples:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


@contextlib.contextmanager
def no_filesystem():
    calls = []

    def make(name):
        def blocked(*a, **k):
            calls.append((name, str(a[0]) if a else ""))
            raise AssertionError(f"filesystem call {name} before the test-path refusal")
        return blocked
    targets = [(os, "stat"), (os, "lstat"), (os, "scandir"), (os, "listdir"), (os, "mkdir"), (os.path, "realpath")]
    with patched(*[(obj, name, make(name)) for obj, name in targets]):
        yield calls


def run_sc(argv) -> tuple[int, str, str]:
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        code = SC.main([str(x) for x in argv])
    return code, so.getvalue(), se.getvalue()


def fresh_out(tag: str) -> Path:
    CTX.counter = getattr(CTX, "counter", 0) + 1
    return CTX.root / "outs" / f"{tag}_{CTX.counter:03d}"


def listing(d: Path) -> list:
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def jpeg(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w, h), color=128).save(path, format="JPEG")


# --------------------------------------------------------------------------------------------------
# trees and records
# --------------------------------------------------------------------------------------------------
TRAIN_A = [("sq_0.jpg", 8, 8), ("sq_1.jpg", 8, 8), ("sq_2.jpg", 8, 8), ("w126x10.jpg", 126, 10),
           ("w127x10.jpg", 127, 10), ("w265x10.jpg", 265, 10), ("w266x10.jpg", 266, 10),
           ("p10x300.jpg", 10, 300), ("up.JPEG", 8, 8)]
VAL_A = [(f"v_{i}.jpg", 8, 8) for i in range(4)]


def tree(name: str, train, val, *, extra=None) -> Path:
    root = CTX.root / "trees" / name
    for split, items in (("train", train), ("val", val)):
        if items is None:
            continue
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for fname, w, h in items:
            jpeg(root / "images" / split / fname, w, h)
    if extra:
        extra(root)
    return root


def build_trees() -> None:
    def a_extra(root):
        jpeg(root / "images" / "train" / "sub" / "wide.jpg", 1000, 10)          # a subfolder: never entered
        jpeg(root / "annotations" / "train" / "wide.jpg", 1000, 10)              # another folder: never listed
    CTX.ta = tree("ar_main", TRAIN_A, VAL_A, extra=a_extra)
    CTX.ta_noval = tree("ar_noval", TRAIN_A, None)
    CTX.ta_short = tree("ar_short", TRAIN_A[:8], VAL_A)

    def bad(root):
        (root / "images" / "train" / "bad.jpg").write_bytes(b"this is not a jpeg\n")
    CTX.ta_bad = tree("ar_bad", TRAIN_A[:8], VAL_A, extra=bad)
    CTX.ta_dup = tree("ar_dup", TRAIN_A[:7] + [("a.jpg", 8, 8), ("a.JPEG", 8, 8)], VAL_A)
    train_t = [(f"t_{i}.jpg", 8, 8) for i in range(5)]
    val_t = [(f"u_{i}.jpg", 8, 8) for i in range(3)]
    CTX.tt = tree("ts_main", train_t, val_t)
    CTX.tt_jpg = tree("ts_jpg", train_t, val_t[:2] + [("a.JPG", 8, 8)])

    def sub(root):
        (root / "images" / "train" / "extra").mkdir()
    CTX.tt_sub = tree("ts_sub", train_t, val_t, extra=sub)
    CTX.val_manifest = hash_split_manifest([ManifestEntry(i, s, s) for i, s in enumerate(["u_0", "u_1", "u_2"])])


def records(name: str, *, rows=None, mutate=None, raw=None, basename=SC.RECORDS_FILE) -> Path:
    p = CTX.root / "records" / name / basename
    p.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        p.write_text(raw, encoding="utf-8")
        return p
    rows = rows if rows is not None else [
        {"iteration": it, "mIoU_full": 0.30 + it / 1e6, "mIoU_disease_full": 0.28, "display_mIoU_percent": 30.0,
         "val_images": 3, "nmf_seed": 42, "val_manifest_sha256": CTX.val_manifest,
         "checkpoint_path": f"/work/iter_{it}.pth", "checkpoint_sha256": hashlib.sha256(str(it).encode()).hexdigest(),
         "checkpoint_bytes": 1000} for it in SC.EXPECTED_ITERATIONS]
    if mutate:
        mutate(rows)
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return p


def build_records() -> None:
    CTX.r_ok = records("ok")

    def last(rows):
        rows[-1]["val_images"] = 2
    CTX.r_last = records("last", mutate=last)
    CTX.r_9 = records("nine", mutate=lambda rows: rows.pop())

    def other(rows):
        for r in rows:
            r["val_manifest_sha256"] = "0" * 64
    CTX.r_manifest = records("manifest", mutate=other)

    def iter4001(rows):
        rows[0]["iteration"] = 4001
    CTX.r_iter = records("iteration", mutate=iter4001)

    def nonuniform(rows):
        rows[4]["val_manifest_sha256"] = "1" * 64
    CTX.r_nonuniform = records("nonuniform", mutate=nonuniform)
    ok_text = CTX.r_ok.read_text(encoding="utf-8").splitlines()
    CTX.r_nan = records("nan", raw="\n".join(ok_text[:3] + [ok_text[3].replace('"mIoU_full": ', '"mIoU_full": NaN, "x": ')]
                                             + ok_text[4:]) + "\n")
    CTX.r_nonobj = records("nonobject", raw="\n".join(ok_text[:5] + ["[1, 2]"] + ok_text[6:]) + "\n")
    CTX.r_name = records("badname", basename="teacher_records.jsonl")


def ar_argv(root: Path, out: Path, *, train=9, val=4, synthetic=True, extra=()) -> list:
    a = ["--census", "aspect-ratio", "--data-root", root, "--out-dir", out]
    if synthetic:
        a += ["--synthetic-inputs", "--generated-utc", GEN_UTC, "--expect-train", train, "--expect-val", val]
    return a + list(extra)


def ts_argv(root: Path, rec: Path, out: Path, *, train=5, val=3, synthetic=True, extra=()) -> list:
    a = ["--census", "teacher-suffix", "--data-root", root, "--teacher-records", rec, "--out-dir", out]
    if synthetic:
        a += ["--synthetic-inputs", "--generated-utc", GEN_UTC, "--expect-train", train, "--expect-val", val]
    return a + list(extra)


def load(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text(encoding="utf-8"))


def expect(argv, code_want: int, needle: str, out: Path, keep=()) -> tuple[bool, dict]:
    code, so, se = run_sc(argv)
    left = listing(out)
    ok = code == code_want and needle in se and left == sorted(keep)
    return ok, {"exit": code, "stderr": se.strip()[-500:], "left": left}


# --------------------------------------------------------------------------------------------------
# A: aspect-ratio
# --------------------------------------------------------------------------------------------------
def ar_run(root=None, **kw) -> tuple:
    out = fresh_out("ar")
    code, so, se = run_sc(ar_argv(root or CTX.ta, out, **kw))
    doc = load(out, AR) if (out / AR).exists() else None
    return code, doc, se, out


def A1():
    code, doc, se, _ = ar_run()
    ok = code == 0 and doc is not None and doc["splits"]["train"]["n"] == 9 and doc["splits"]["val"]["n"] == 4
    return ok, {"exit": code, "stderr": se[-300:]}


def A2():
    code, doc, se, _ = ar_run()
    c = doc["combined"] if doc else {}
    return code == 0 and c.get("max_ar") == 30.0 and c["max_ar_image"]["stem"] == "p10x300", c.get("max_ar_image")


def A3a():
    code, doc, se, _ = ar_run()
    return code == 0 and doc["splits"]["train"]["count_above_12_6"] == 4, doc["splits"]["train"] if doc else se


def A3b():
    code, doc, se, _ = ar_run()
    return code == 0 and doc["splits"]["train"]["count_above_26_5"] == 2, doc["splits"]["train"] if doc else se


def A4():
    code, doc, se, _ = ar_run()
    img = doc["splits"]["train"]["max_ar_image"] if doc else {}
    return (code == 0 and img == {"split": "train", "stem": "p10x300", "file": "p10x300.jpg", "w": 10, "h": 300}
            and doc["splits"]["train"]["max_ar"] == 30.0), img


def A5():
    def refuse(self, *a, **k):
        raise AssertionError("a pixel decode: ImageFile.load was called")
    with patched((ImageFile.ImageFile, "load", refuse)):
        code, doc, se, _ = ar_run()
    return code == 0 and doc is not None and doc["combined"]["n"] == 13, {"exit": code, "stderr": se[-300:]}


def A6a():
    code, doc, se, _ = ar_run()
    t = doc["splits"]["train"] if doc else {}
    return code == 0 and t.get("n") == 9 and t.get("max_ar") == 30.0 and doc["inputs"]["entries"]["train"]["dirs"] == 1, t


def A6b():
    real, seen = os.scandir, []

    def spy(path="."):
        seen.append(os.fspath(path))
        return real(path)
    with patched((os, "scandir", spy), (os, "walk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("walk"))),
                 (os, "listdir", lambda *a, **k: (_ for _ in ()).throw(AssertionError("listdir")))):
        code, doc, se, _ = ar_run()
    want = sorted([os.path.join(str(CTX.ta), "images", "train"), os.path.join(str(CTX.ta), "images", "val")])
    return code == 0 and sorted(seen) == want, {"seen": seen, "exit": code}


def A7():
    _, _, _, o1 = ar_run()
    _, _, _, o2 = ar_run()
    a, b = (o1 / AR).read_bytes(), (o2 / AR).read_bytes()
    return a == b, (hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())


def A8a():
    bad = CTX.root / "never" / "never_TeSt_root"
    out = fresh_out("a8a")
    with no_filesystem() as calls:
        code, so, se = run_sc(ar_argv(bad, out))
    return code == 2 and "test" in se.lower() and not calls and not os.path.lexists(bad), {"exit": code, "calls": calls}


def A8b():
    bad = CTX.root / "never" / "out_TEST"
    with no_filesystem() as calls:
        code, so, se = run_sc(ar_argv(CTX.ta, bad))
    return code == 2 and "test" in se.lower() and not calls and not os.path.lexists(bad), {"exit": code, "calls": calls}


def A8c():
    out = fresh_out("a8c")
    out.mkdir(parents=True)
    (out / AR).write_bytes(b"pre-existing\n")
    ok, det = expect(ar_argv(CTX.ta, out), 2, "already holds", out, keep=[AR])
    return ok and (out / AR).read_bytes() == b"pre-existing\n", det


def A8d():
    out = fresh_out("a8d")
    return expect(ar_argv(CTX.ta_noval, out), 2, "missing folder", out)


def A8e():
    out = fresh_out("a8e")
    return expect(ar_argv(CTX.ta_short, out), 2, "expected 9", out)


def A8f():
    inside = REPO / "s4_smoke_census_inside_never"
    ok, det = expect(ar_argv(CTX.ta, inside), 2, "outside the repository", inside)
    return ok and not inside.exists(), det


def A8g():
    out = fresh_out("a8g")
    return expect(ar_argv(CTX.ta, out, synthetic=False), 2, "--script-commit", out)


def A8h():
    out = fresh_out("a8h")
    extra = ["--script-commit", "0" * 40, "--script-commit-dl-id", "DL-99", "--expect-train", "9"]
    return expect(ar_argv(CTX.ta, out, synthetic=False, extra=extra), 2, "refused in a real run", out)


def A9():
    out = fresh_out("a9")
    return expect(ar_argv(CTX.ta_bad, out), 1, "could not be read", out)


def A10():
    out = fresh_out("a10")
    return expect(ar_argv(CTX.ta_dup, out), 2, "share a stem", out)


# --------------------------------------------------------------------------------------------------
# T: teacher-suffix
# --------------------------------------------------------------------------------------------------
def ts_run(root=None, rec=None, **kw) -> tuple:
    out = fresh_out("ts")
    code, so, se = run_sc(ts_argv(root or CTX.tt, rec or CTX.r_ok, out, **kw))
    doc = load(out, SUF) if (out / SUF).exists() else None
    return code, doc, se, out


def T1a():
    pin = td.file_sha256(CTX.r_ok)
    code, doc, se, _ = ts_run(extra=["--expect-records-sha256", pin])
    sp = doc["splits"] if doc else {}
    ok = (code == 0 and sp["train"]["suffix_counts"] == {".jpg": 5} and sp["val"]["suffix_counts"] == {".jpg": 3}
          and sp["train"]["teacher_rule_count"] == 5 and sp["val"]["student_rule_count"] == 3
          and sp["train"]["stem_sets_equal"] and sp["val"]["stem_sets_equal"])
    return ok, {"exit": code, "stderr": se[-300:]}


def T1b():
    pin = td.file_sha256(CTX.r_ok)
    code, doc, se, _ = ts_run(extra=["--expect-records-sha256", pin])
    r = doc["records"] if doc else {}
    ok = (code == 0 and r["rows"] == 10 and r["iterations"] == SC.EXPECTED_ITERATIONS and set(r["val_images"]) == {3}
          and r["records_sha256"] == pin and r["expected_sha256_matches"] is True and r["expected_source"] == SC.DL68
          and len(r["mIoU_full"]) == 10)
    return ok, r


def T1c():
    code, doc, se, _ = ts_run()
    return (code == 0 and doc["val_manifest"]["equal"] is True and doc["val_manifest"]["census"] == CTX.val_manifest
            and doc["expectation_met"] is True), doc["expectation"] if doc else se


def T2a():
    code, doc, se, _ = ts_run(CTX.tt_jpg)
    e, v = doc["expectation"], doc["splits"]["val"]
    ok = (e["val_teacher_rule_count_is_3"] is False and e["val_stem_sets_equal"] is False
          and v["teacher_rule_count"] == 2 and v["student_rule_count"] == 3 and v["suffix_counts"] == {".JPG": 1, ".jpg": 2})
    return ok, v


def T2b():
    code, doc, se, out = ts_run(CTX.tt_jpg)
    return code == 1 and (out / SUF).exists() and doc["expectation_met"] is False and "STOP" in se, {"exit": code}


def T3a():
    code, doc, se, _ = ts_run(rec=CTX.r_last)
    return code == 1 and doc["expectation"]["records_val_images_3_in_every_row"] is False, doc["expectation"]


def T3b():
    code, doc, se, _ = ts_run(rec=CTX.r_9)
    return code == 1 and doc["expectation"]["records_rows_10"] is False, doc["expectation"]


def T3c():
    code, doc, se, _ = ts_run(rec=CTX.r_manifest)
    e = doc["expectation"]
    return (code == 1 and e["records_val_manifest_equals_census"] is False and e["records_val_manifest_uniform"] is True), e


def T3d():
    code, doc, se, _ = ts_run(rec=CTX.r_iter)
    return code == 1 and doc["expectation"]["records_iterations_4000_to_40000"] is False, doc["expectation"]


def T4():
    code, doc, se, _ = ts_run(CTX.tt_sub)
    return code == 1 and doc["expectation"]["train_no_subdirectory"] is False and doc["splits"]["train"]["n_dirs"] == 1, \
        doc["expectation"]


def T5():
    out = fresh_out("t5")
    return expect(ts_argv(CTX.tt, CTX.r_name, out), 2, "must name", out)


def _assign(tree_, name):
    for node in tree_.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def T6():
    comp = ast.parse(TEACHER_COMPONENTS.read_text(encoding="utf-8"))
    cfg = ast.parse(TEACHER_CONFIG.read_text(encoding="utf-8"))
    suffixes = {ast.literal_eval(k.value) for n in ast.walk(cfg) if isinstance(n, ast.Call)
                for k in n.keywords if k.arg == "img_suffix"}
    got = {"RECORDS_FILE": _assign(comp, "RECORDS_FILE"), "img_suffix": suffixes,
           "MAX_ITERS": _assign(cfg, "MAX_ITERS"), "VAL_INTERVAL": _assign(cfg, "VAL_INTERVAL"),
           "TRAIN_ROWS": td.TRAIN_ROWS, "VAL_ROWS": td.VAL_ROWS}
    want = {"RECORDS_FILE": SC.RECORDS_FILE, "img_suffix": {SC.TEACHER_IMG_SUFFIX}, "MAX_ITERS": SC.TEACHER_MAX_ITERS,
            "VAL_INTERVAL": SC.TEACHER_VAL_INTERVAL, "TRAIN_ROWS": 5367, "VAL_ROWS": 846}
    return got == want, got


def T7():
    _, _, _, o1 = ts_run()
    _, _, _, o2 = ts_run()
    a, b = (o1 / SUF).read_bytes(), (o2 / SUF).read_bytes()
    return a == b, (hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())


def T8a():
    bad = CTX.root / "never" / "Test_records" / SC.RECORDS_FILE
    out = fresh_out("t8a")
    with no_filesystem() as calls:
        code, so, se = run_sc(ts_argv(CTX.tt, bad, out))
    return code == 2 and "test" in se.lower() and not calls and not os.path.lexists(bad), {"exit": code, "calls": calls}


def T8b():
    out = fresh_out("t8b")
    out.mkdir(parents=True)
    (out / SUF).write_bytes(b"pre-existing\n")
    ok, det = expect(ts_argv(CTX.tt, CTX.r_ok, out), 2, "already holds", out, keep=[SUF])
    return ok and (out / SUF).read_bytes() == b"pre-existing\n", det


def T8c():
    out = fresh_out("t8c")
    absent = CTX.root / "records" / "absent" / SC.RECORDS_FILE
    return expect(ts_argv(CTX.tt, absent, out), 2, "does not exist", out)


def T8d():
    out = fresh_out("t8d")
    extra = ["--script-commit", "0" * 40, "--script-commit-dl-id", "DL-99", "--expect-val", "3"]
    return expect(ts_argv(CTX.tt, CTX.r_ok, out, synthetic=False, extra=extra), 2, "refused in a real run", out)


def T9():
    out = fresh_out("t9")
    return expect(ts_argv(CTX.tt, CTX.r_nan, out), 2, "non-finite", out)


def T10():
    out = fresh_out("t10")
    return expect(ts_argv(CTX.tt, CTX.r_nonobj, out), 2, "not a JSON object", out)


def T11():
    code, doc, se, _ = ts_run(rec=CTX.r_nonuniform)
    e = doc["expectation"]
    return code == 1 and e["records_val_manifest_uniform"] is False and e["records_val_manifest_equals_census"] is False, e


def T12():
    out = fresh_out("t12")
    return expect(ts_argv(CTX.tt, CTX.r_ok, out, extra=["--expect-records-sha256", "0" * 64]), 2,
                  "--expect-records-sha256", out)


# --------------------------------------------------------------------------------------------------
# M: mutations
# --------------------------------------------------------------------------------------------------
def header_with_load(path):
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        return w, h, im.format, []


def scan_recursive(folder):
    if not os.path.isdir(folder):
        raise td.Refused("missing folder under --data-root")
    files = []
    for dirpath, _dirs, names in os.walk(folder):
        files += [os.path.relpath(os.path.join(dirpath, n), folder) for n in names]
    return {"files": sorted(files), "dirs": [], "n_other": 0}


MUTATIONS = [
    ("M19", "a pixel decode (load())", lambda: [(SC, "read_header", header_with_load)], "A5"),
    ("M20", "AR as W/H", lambda: [(SC, "long_short", lambda w, h: (w, h))], "A4"),
    ("M21", ">= at the threshold", lambda: [(SC, "above", lambda L, S, num, den: den * L >= num * S)], "A3a"),
    ("M22", "recursive listing", lambda: [(SC, "scan_folder", scan_recursive)], "A6a"),
    ("M23", "exact '.jpg' rule in the aspect-ratio census", lambda: [(SC, "student_rule", lambda n: n.endswith(".jpg"))],
     "A1"),
    ("M24", "case-insensitive teacher rule", lambda: [(SC, "teacher_rule", lambda n: n.lower().endswith(".jpg"))], "T2a"),
    ("M25", "val_images checked on the first row only",
     lambda: [(SC, "val_images_ok", lambda rows, want: rows[0]["val_images"] == want)], "T3a"),
    ("M26", "row count unchecked", lambda: [(SC, "rows_count_ok", lambda rows: True)], "T3b"),
    ("M27", "manifest check skipped", lambda: [(SC, "manifest_equal", lambda census, rows: True)], "T3c"),
]


def checks() -> list:
    return [
        ("A1", "aspect-ratio: exit 0, TRAIN 9 and VAL 4 under the evaluator's rule", A1),
        ("A2", "the widest image is the 10x300 portrait, AR 30", A2),
        ("A3a", "126x10 (AR 12.6) is not above 12.6; four TRAIN images are", A3a),
        ("A3b", "265x10 (AR 26.5) is not above 26.5; two TRAIN images are", A3b),
        ("A4", "the portrait counts by long/short", A4),
        ("A5", "no pixel decoded: ImageFile.load patched to raise, still exit 0", A5),
        ("A6a", "the images/train subfolder is counted as a directory and never entered", A6a),
        ("A6b", "os.scandir sees exactly images/train and images/val; no walk, no listdir", A6b),
        ("A7", "rerun with the same --generated-utc: byte-identical", A7),
        ("A8a", "a never-created test-string --data-root -> exit 2 before any filesystem call", A8a),
        ("A8b", "a never-created test-string --out-dir -> exit 2 before any filesystem call", A8b),
        ("A8c", "an existing output -> exit 2, bytes unchanged", A8c),
        ("A8d", "images/val missing -> exit 2", A8d),
        ("A8e", "8 TRAIN images where 9 are expected -> exit 2", A8e),
        ("A8f", "--out-dir inside the repository -> exit 2", A8f),
        ("A8g", "real mode without the binding flags -> exit 2", A8g),
        ("A8h", "--expect-train in real mode -> exit 2", A8h),
        ("A9", "an unreadable header -> exit 1, nothing written", A9),
        ("A10", "duplicate stems (a.jpg + a.JPEG) -> exit 2", A10),
        ("T1a", "teacher-suffix: exit 0, exact histograms, both rules 5/3, stem sets equal", T1a),
        ("T1b", "records: 10 rows, iterations 4000..40000, val_images 3, the sha256 pin and DL-68 source", T1b),
        ("T1c", "the census VAL manifest equals the records'; expectation met", T1c),
        ("T2a", "a.JPG: teacher 2 != student 3, sets differ, histogram {'.JPG': 1, '.jpg': 2}", T2a),
        ("T2b", "a.JPG: exit 1 with the census written", T2b),
        ("T3a", "the LAST row with val_images 2 -> not met", T3a),
        ("T3b", "9 rows -> records_rows_10 false", T3b),
        ("T3c", "another (uniform) manifest hash -> not equal", T3c),
        ("T3d", "a row at iteration 4001 -> iterations false", T3d),
        ("T4", "a subdirectory in images/train -> not met", T4),
        ("T5", "a records file not named teacher_selection_records.jsonl -> exit 2", T5),
        ("T6", "RECORDS_FILE, img_suffix, MAX_ITERS, VAL_INTERVAL equal their sources (ast); 5,367 and 846", T6),
        ("T7", "rerun: byte-identical", T7),
        ("T8a", "a never-created test-string --teacher-records -> exit 2 before any filesystem call", T8a),
        ("T8b", "an existing output -> exit 2, bytes unchanged", T8b),
        ("T8c", "missing records -> exit 2", T8c),
        ("T8d", "--expect-val in real mode -> exit 2", T8d),
        ("T9", "a records row with NaN -> exit 2", T9),
        ("T10", "a non-object records line -> exit 2", T10),
        ("T11", "non-uniform val_manifest_sha256 across rows -> not met", T11),
        ("T12", "a wrong --expect-records-sha256 -> exit 2", T12),
    ]


def run_check(cid, desc, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 -- a crash is a failed check
        ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    record(cid, desc, ok, detail)
    NORMAL[cid] = (bool(ok), fn)
    return ok


def main() -> int:
    t0 = time.time()
    root = Path(tempfile.mkdtemp(prefix="s4_smoke_censuses_"))
    real = os.path.realpath(root)
    if "test" in real.lower():
        print(f"ABORT: the temporary root {real} contains 'test'; set TMPDIR=/tmp/s4scratch", file=sys.stderr)
        return 2
    CTX.root = root
    print(f"smoke_split_censuses: temporary root {real} (left in place)", flush=True)
    build_trees()
    build_records()
    for cid, desc, fn in checks():
        run_check(cid, desc, fn)
    for mid, desc, make, killer in MUTATIONS:
        passed, fn = NORMAL[killer]
        try:
            with patched(*make()):
                ok, detail = fn()
        except Exception as e:  # noqa: BLE001 -- a crash under the mutation fails the killer: killed
            ok, detail = False, f"{type(e).__name__}: {e}"
        killed = passed and not ok
        record(mid, f"mutation '{desc}' killed by {killer}", killed, detail if not killed else "")
    n_ok = sum(1 for r in RESULTS if r[2])
    n_mut = sum(1 for r in RESULTS if r[0].startswith("M") and r[2])
    print(f"\nPASS {n_ok}/{len(RESULTS)} (mutations killed {n_mut}/{len(MUTATIONS)}); {time.time() - t0:.0f} s",
          flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
