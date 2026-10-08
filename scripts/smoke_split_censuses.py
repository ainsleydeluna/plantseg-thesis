#!/usr/bin/env python3
"""Synthetic smoke for the DL-49 aspect-ratio census and the DL-55 teacher suffix census (S4;
scripts/split_censuses.py).

SYNTHETIC ONLY. Small trees of tiny JPEGs are written with PIL under a temporary root. Its parent is TMPDIR,
whose string and realpath are both checked to contain no "test" before anything is created; its own name is a
fixed prefix plus hex digits, which cannot spell "test". Records fixtures follow the teacher hook's row schema
(src/training/teacher_components.py:356-366). No PlantSeg image, mask or TEST path is read; nothing is written
into the repository; the TEST-path refusals use names and resolved paths that are never created. The synthetic
trees are run with --synthetic-inputs and --expect-train/--expect-val (smoke mode only; a real run fixes 5,367
and 846).

  A  aspect-ratio: counts; the widest image; strict "above" at 12.6 and 26.5 in integers; long/short for a
     portrait; no pixel decoded (PIL ImageFile.load patched to raise); exactly two folders listed, no recursion;
     byte-identical reruns; the per-split and combined formats and warnings; refusals (exit 2), including the
     flag gate, the run-of-record binding, a data root or a listed entry resolving to a test location and a
     listed test name; an unreadable header (exit 1, nothing written); duplicate stems
  T  teacher-suffix: exact-suffix histograms and both rules (dot-named files skipped by the teacher's listing);
     the records checks (rows, iterations, val_images in every row, one manifest hash equal to the census's, in
     name order, not stem order); a subdirectory; the records basename; the retyped literals against their
     sources (ast, no import); byte-identical reruns; refusals; strict records (non-finite, duplicate keys,
     undecodable bytes, oversized integers, non-JSON, non-object, missing fields, empty); the records pin
  M  mutations of one rule each, every one killed by the named check

Run (TMPDIR must be set; its realpath must contain no "test"):
    env TMPDIR=/tmp/s4scratch PYTHONPATH=<repo> python -B scripts/smoke_split_censuses.py
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import io
import json
import os
import secrets
import sys
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
ITERATIONS = [4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000]   # literals, not SC's constants
DL68_SOURCE = "DL-68, docs/DECISION_LOG.md:127"
FLAG_GATE = "a real run requires --script-commit and --script-commit-dl-id"
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
    targets = [(os, "stat"), (os, "lstat"), (os, "scandir"), (os, "listdir"), (os, "mkdir"), (os.path, "realpath"),
               (builtins, "open"), (io, "open"), (os, "open"), (os, "access")]
    with patched(*[(obj, name, make(name)) for obj, name in targets]):
        yield calls


def tripwire(*a, **k):
    """Replaces td.write_files_exclusive where a regressed gate could otherwise write."""
    raise AssertionError("write attempted")


@contextlib.contextmanager
def header_spy():
    """Records every SC.read_header call (and still reads)."""
    calls, real = [], SC.read_header

    def spy(path):
        calls.append(str(path))
        return real(path)
    with patched((SC, "read_header", spy)):
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
    # name order u-1.jpg < u.jpg < v.jpg ('-' < '.'), stem order u < u-1 < v: Annex A row 24's name order
    val_t = [("u-1.jpg", 8, 8), ("u.jpg", 8, 8), ("v.jpg", 8, 8)]
    CTX.tt = tree("ts_main", train_t, val_t)
    CTX.tt_jpg = tree("ts_jpg", train_t, val_t[:2] + [("a.JPG", 8, 8)])

    def sub(root):
        (root / "images" / "train" / "extra").mkdir()
    CTX.tt_sub = tree("ts_sub", train_t, val_t, extra=sub)
    CTX.tt_dot = tree("ts_dot", train_t + [(".h.jpg", 8, 8)], val_t)

    def link(root):                                                   # an in-folder symlink to a sibling image
        os.symlink("t_0.jpg", root / "images" / "train" / "link_t0.jpg")
    CTX.tt_link = tree("ts_link", train_t, val_t, extra=link)
    CTX.ta_warn = tree("ar_warn", TRAIN_A, VAL_A[:3] + [("v_big.jpg", 50, 50)])     # 2,500 px: warns at 2,000
    CTX.val_manifest = hash_split_manifest([ManifestEntry(i, s, s) for i, s in enumerate(["u-1", "u", "v"])])
    CTX.val_manifest_stem_order = hash_split_manifest([ManifestEntry(i, s, s)
                                                       for i, s in enumerate(["u", "u-1", "v"])])


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
         "checkpoint_bytes": 1000} for it in ITERATIONS]
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
    CTX.r_inf = records("overflow", raw="\n".join(ok_text[:2] + [ok_text[2].replace('"mIoU_full": ', '"mIoU_full": 1e999, "x": ')]
                                                  + ok_text[3:]) + "\n")
    CTX.r_dup = records("duplicate", raw="\n".join(ok_text[:7] + [ok_text[7][:-1] + ', "val_images": 2}']
                                                   + ok_text[8:]) + "\n")
    CTX.r_bytes = CTX.root / "records" / "bytes" / SC.RECORDS_FILE
    CTX.r_bytes.parent.mkdir(parents=True)
    CTX.r_bytes.write_bytes(CTX.r_ok.read_bytes() + b'{"iteration": 1, "x": "\xff"}\n')
    CTX.r_bigint = records("bigint", raw="\n".join(ok_text + ['{"iteration": ' + "7" * 5000 + "}"]) + "\n")
    CTX.r_notjson = records("notjson", raw="\n".join(ok_text[:4] + ["not json"] + ok_text[4:]) + "\n")

    def no_val(rows):
        del rows[6]["val_images"]
    CTX.r_noval = records("noval", mutate=no_val)
    CTX.r_empty = records("empty", raw="")


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
    with patched((SC.td, "write_files_exclusive", tripwire)):
        ok, det = expect(ar_argv(CTX.ta, inside), 2, "outside the repository", inside)
    return ok and not inside.exists(), det


def A8g():
    out = fresh_out("a8g")
    return expect(ar_argv(CTX.ta, out, synthetic=False), 2, FLAG_GATE, out)


def A8i():
    out = fresh_out("a8i")
    return expect(ar_argv(CTX.ta, out, synthetic=False, extra=["--script-commit", "0" * 40]), 2, FLAG_GATE, out)


def fake_git(head, status=""):
    """td._git for the binding cases: no git process runs."""
    def _git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return None if head is None else head + "\n"
        return status if args[:1] == ("status",) else None
    return _git


def binding_case(tag: str, head, status: str, needle: str, extra_patches=()) -> tuple[bool, dict]:
    out = fresh_out(tag)
    real = ["--script-commit", "0" * 40, "--script-commit-dl-id", "DL-99"]
    with patched((SC.td, "_git", fake_git(head, status)), *extra_patches):
        return expect(ar_argv(CTX.ta, out, synthetic=False, extra=real), 2, needle, out)


def A11a():
    return binding_case("a11a", None, "", "git HEAD could not be read")


def A11b():
    return binding_case("a11b", "1" * 40, "", f"HEAD {'1' * 40} != --script-commit {'0' * 40}")


def A11c():
    return binding_case("a11c", "0" * 40, "", "code file(s) missing at HEAD",
                        ((SC, "CODE_FILES", SC.CODE_FILES + ("scripts/s4_absent_never.py",)),))


def A11d():
    return binding_case("a11d", "0" * 40, " M src/x.py\n", "needs them clean at HEAD")


def resolving_to_test(alias: str, target: str):
    """os.path.realpath with one never-created path resolving to a never-created test-named location."""
    real = os.path.realpath

    def fake(p, *a, **k):
        return target if os.fspath(p) == alias else real(p, *a, **k)
    return fake


def A12():
    out = fresh_out("a12")
    alias, target = CTX.root / "alias_root", CTX.root / "never" / "never_TeSt_root"      # neither is created
    with patched((os.path, "realpath", resolving_to_test(str(alias), str(target))),
                 (SC.td, "write_files_exclusive", tripwire)):
        ok, det = expect(ar_argv(alias, out), 2, "resolving to a 'test' location", out)
    return ok and not os.path.lexists(alias) and not os.path.lexists(target), det


def inject_listing(name: str, split: str = "val"):
    """SC.scan_folder with a never-created name added to one folder's listing."""
    real = SC.scan_folder

    def fake(folder):
        got = real(folder)
        if os.path.normpath(folder).endswith(os.path.join("images", split)):
            got = dict(got, files=sorted(got["files"] + [name]))
        return got
    return fake


def A13():
    out = fresh_out("a13")
    with patched((SC, "scan_folder", inject_listing("x_TeSt.jpg")), (SC.td, "write_files_exclusive", tripwire)), \
            header_spy() as reads:
        ok, det = expect(ar_argv(CTX.ta, out), 2, "images/val names", out)
    return ok and not reads, dict(det, header_reads=len(reads))


def A14():
    out = fresh_out("a14")
    entry = os.path.join(str(CTX.ta), "images", "train", "sq_0.jpg")
    target = str(CTX.root / "never" / "never_TeSt_link.jpg")                              # never created
    with patched((os.path, "realpath", resolving_to_test(entry, target)), (SC.td, "write_files_exclusive", tripwire)), \
            header_spy() as reads:
        ok, det = expect(ar_argv(CTX.ta, out), 2, "images/train entry", out)
    return ok and not reads and not os.path.lexists(target), dict(det, header_reads=len(reads))


def A15():
    code, doc, se, _ = ar_run()
    if doc is None:
        return False, {"exit": code, "stderr": se[-300:]}
    sp, c, e = doc["splits"], doc["combined"], doc["inputs"]["entries"]
    ok = (code == 0 and sp["train"]["formats"] == {"JPEG": 9} and sp["val"]["formats"] == {"JPEG": 4}
          and c["formats"] == {"JPEG": 13} and sp["train"]["header_warnings"] == {} == sp["val"]["header_warnings"]
          and c["header_warnings"] == {} and "header_warnings" not in doc
          and all(e[s]["symlinks"] == 0 and e[s]["dot_files"] == 0 for s in ("train", "val")))
    return ok, {"splits": {s: (sp[s]["formats"], sp[s]["header_warnings"]) for s in sp},
                "combined": (c["formats"], c["header_warnings"]), "entries": e}


def A16():
    out = fresh_out("a16")
    return expect(ar_argv(CTX.ta, out, extra=["--expect-records-sha256", "0" * 64]), 2,
                  "belong to --census teacher-suffix", out)


def A15b():
    """PIL warns (DecompressionBombWarning) above MAX_IMAGE_PIXELS and refuses above twice it: at 2,000 the 265x10,
    266x10 and 10x300 TRAIN images and the 50x50 VAL image warn, and nothing is refused."""
    with patched((Image, "MAX_IMAGE_PIXELS", 2000)):
        code, doc, se, _ = ar_run(CTX.ta_warn, val=4)
    if doc is None:
        return False, {"exit": code, "stderr": se[-300:]}
    n = {s: sum(doc["splits"][s]["header_warnings"].values()) for s in ("train", "val")}
    c = sum(doc["combined"]["header_warnings"].values())
    return code == 0 and n == {"train": 3, "val": 1} and c == 4, {"per_split": n, "combined": c}


def A17():
    code, doc, se, _ = ar_run(CTX.tt_link, train=6, val=3)
    e = doc["inputs"]["entries"] if doc else {}
    return code == 0 and e.get("train", {}).get("symlinks") == 1 and e.get("val", {}).get("symlinks") == 0, e


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
    ok = (code == 0 and r["rows"] == 10 and r["iterations"] == ITERATIONS == r["expected_iterations"]
          and set(r["val_images"]) == {3} and r["records_sha256"] == pin and r["expected_sha256_matches"] is True
          and r["expected_source"] == DL68_SOURCE and len(r["mIoU_full"]) == 10)
    return ok, r


def T1c():
    code, doc, se, _ = ts_run()
    return (code == 0 and doc["val_manifest"]["equal"] is True and doc["val_manifest"]["census"] == CTX.val_manifest
            and CTX.val_manifest != CTX.val_manifest_stem_order and doc["expectation_met"] is True), \
        doc["expectation"] if doc else se


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


def T13():
    out = fresh_out("t13")
    with patched((SC, "scan_folder", inject_listing("x_TeSt.jpg")), (SC.td, "write_files_exclusive", tripwire)):
        return expect(ts_argv(CTX.tt, CTX.r_ok, out), 2, "images/val names", out)


def T14():
    code, doc, se, _ = ts_run(CTX.tt_dot)
    t = doc["splits"]["train"] if doc else {}
    ok = (code == 1 and t.get("teacher_rule_count") == 5 and t.get("student_rule_count") == 6
          and t.get("n_dot_files") == 1 and t.get("stem_sets_equal") is False and t.get("only_student") == [".h"])
    return ok, t


def T23():
    code, doc, se, _ = ts_run(CTX.tt_link, train=6)
    t = doc["splits"]["train"] if doc else {}
    ok = (code == 0 and t.get("n_symlinks") == 1 and doc["splits"]["val"]["n_symlinks"] == 0
          and t.get("teacher_rule_count") == 6 == t.get("student_rule_count"))
    return ok, t


def records_refusal(tag: str, rec: Path, needle: str) -> tuple[bool, dict]:
    out = fresh_out(tag)
    return expect(ts_argv(CTX.tt, rec, out), 2, needle, out)


def T15():
    return records_refusal("t15", CTX.r_inf, "non-finite JSON number")


def T16():
    return records_refusal("t16", CTX.r_dup, "duplicate key")


def T17():
    return records_refusal("t17", CTX.r_bytes, "cannot be read as UTF-8 text")


def T18():
    return records_refusal("t18", CTX.r_bigint, "line 11 cannot be read")


def T19():
    out = fresh_out("t19")
    argv = ["--census", "teacher-suffix", "--data-root", CTX.tt, "--out-dir", out, "--synthetic-inputs",
            "--generated-utc", GEN_UTC, "--expect-train", 5, "--expect-val", 3]
    return expect(argv, 2, "requires --teacher-records", out)


def T20():
    return records_refusal("t20", CTX.r_notjson, "line 5 is not JSON")


def T21():
    return records_refusal("t21", CTX.r_noval, "lacks an int 'val_images'")


def T22():
    return records_refusal("t22", CTX.r_empty, "holds no row")


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


def binding_without_tree(args) -> dict:
    """Mutation: commit_binding without its clean-tree check."""
    head = SC.td.git_head()
    if head is None:
        raise SC.td.Refused("git HEAD could not be read")
    if head != args.script_commit:
        raise SC.td.Refused(f"HEAD {head} != --script-commit {args.script_commit}")
    return {"head": head, "script_commit": args.script_commit, "script_commit_dl_id": args.script_commit_dl_id}


def guard_names_only(args) -> None:
    """Mutation: guard_paths without its resolved-path stage."""
    named = [args.data_root, *SC.folder_paths(args.data_root).values(), args.out_dir]
    if args.teacher_records is not None:
        named.append(args.teacher_records)
    SC.td.refuse_test_names([str(v) for v in named], "path arguments")


_REAL_REFUSE_NAMES = td.refuse_test_names
_REAL_READ_HEADER = SC.read_header


def refuse_argument_names_only(names, what) -> None:
    """Mutation: the listed-name refusal disabled (the path-argument refusal kept)."""
    if not str(what).startswith("images/"):
        _REAL_REFUSE_NAMES(names, what)


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
    ("M43", "binding without its clean-tree check", lambda: [(SC, "commit_binding", binding_without_tree)], "A11d"),
    ("M44", "path guard without its resolved-path stage", lambda: [(SC, "guard_paths", guard_names_only)], "A12"),
    ("M45", "listed-name refusal disabled", lambda: [(SC.td, "refuse_test_names", refuse_argument_names_only)], "A13"),
    ("M46", "listed entries not resolved", lambda: [(SC, "refuse_test_entries", lambda folder, names, what: None)],
     "A14"),
    ("M47", "VAL stems in stem order", lambda: [(SC, "teacher_stems", lambda names: sorted(n[:-4] for n in names))],
     "T1c"),
    ("M48", "dot-named files kept by the teacher rule",
     lambda: [(SC, "teacher_rule", lambda n: n.endswith(SC.TEACHER_IMG_SUFFIX))], "T14"),
    ("M49", "an overflowing number accepted", lambda: [(SC, "parse_finite_float", float)], "T15"),
    ("M50", "a duplicate key resolved silently", lambda: [(SC, "no_duplicate_keys", dict)], "T16"),
    ("M53", "header warnings dropped", lambda: [(SC, "read_header", lambda p: _REAL_READ_HEADER(p)[:3] + ([],))],
     "A15b"),
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
        ("A8g", "real mode without the binding flags -> exit 2 (the flag gate's own words)", A8g),
        ("A8h", "--expect-train in real mode -> exit 2", A8h),
        ("A8i", "real mode with --script-commit but no --script-commit-dl-id -> exit 2", A8i),
        ("A9", "an unreadable header -> exit 1, nothing written", A9),
        ("A10", "duplicate stems (a.jpg + a.JPEG) -> exit 2", A10),
        ("A11a", "binding: HEAD unreadable -> exit 2", A11a),
        ("A11b", "binding: HEAD != --script-commit -> exit 2", A11b),
        ("A11c", "binding: a code file missing -> exit 2", A11c),
        ("A11d", "binding: src/configs/scripts not clean -> exit 2", A11d),
        ("A12", "a --data-root resolving to a test location -> exit 2, nothing created", A12),
        ("A13", "a listed name containing 'test' -> exit 2 before any header is read", A13),
        ("A14", "a listed entry resolving to a test location -> exit 2 before any header is read", A14),
        ("A15", "formats and header warnings per split and combined; no symlink, no dot-named file", A15),
        ("A16", "--expect-records-sha256 with --census aspect-ratio -> exit 2", A16),
        ("A15b", "header warnings counted per split and combined (TRAIN 3, VAL 1) at MAX_IMAGE_PIXELS 2,000", A15b),
        ("A17", "an in-folder symlink counted in inputs.entries (aspect-ratio)", A17),
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
        ("T13", "teacher-suffix: a listed name containing 'test' -> exit 2", T13),
        ("T14", "a dot-named .jpg: skipped by the teacher's listing, kept by the student's -> not met", T14),
        ("T15", "a records number that overflows to inf (1e999) -> exit 2", T15),
        ("T16", "a records row with a duplicate key -> exit 2", T16),
        ("T17", "records bytes that are not UTF-8 -> exit 2", T17),
        ("T18", "a records integer beyond Python's digit limit -> exit 2", T18),
        ("T19", "--census teacher-suffix without --teacher-records -> exit 2", T19),
        ("T20", "a records line that is not JSON -> exit 2", T20),
        ("T21", "a records row without val_images -> exit 2", T21),
        ("T22", "an empty records file -> exit 2", T22),
        ("T23", "an in-folder symlink counted in splits.train.n_symlinks (teacher-suffix)", T23),
    ]


def run_check(cid, desc, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 -- a crash is a failed check
        ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    record(cid, desc, ok, detail)
    NORMAL[cid] = (bool(ok), fn)
    return ok


def make_root(prefix: str):
    """The temporary root, created only after TMPDIR's realpath is known to hold no 'test'. Its name is the
    prefix plus hex digits, which contain neither 's' nor 't', so it cannot spell 'test' either."""
    base = os.environ.get("TMPDIR")
    if not base:
        return None, "TMPDIR is not set; run with TMPDIR=/tmp/s4scratch"
    if "test" in base.lower():                                        # the string first, before any filesystem call
        return None, "TMPDIR contains 'test'"
    real = os.path.realpath(base)
    if "test" in real.lower() or not os.path.isdir(real):
        return None, f"TMPDIR resolves to {real!r}, which contains 'test' or is not a directory"
    root = Path(real) / f"{prefix}{secrets.token_hex(4)}"
    root.mkdir(exist_ok=False)
    return root, None


def main() -> int:
    t0 = time.time()
    root, why = make_root("s4_smoke_censuses_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    real = os.path.realpath(root)
    if "test" in real.lower():
        print(f"ABORT: the temporary root {real} contains 'test'", file=sys.stderr)
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
