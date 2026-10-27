#!/usr/bin/env python3
"""Smoke for the in-session image hasher, scripts/tday_hash_images.py (lane TMAN, B-3a part 1, B3A-2; plan §4(iv),
§1.7 and §5; rulings H8-H10).

- Each step of the hash mode (H9), each refusal reached with every earlier gate passing: 1. the paths (B-1's guard;
  a string holding project rule 4's word gets no call: B-1's tracer form, smoke_tman_paths.py :55-81); 2. the
  records -- no unlock entry, an uncommitted, unrecorded, rewritten or other-kind entry, a manifest that is not the
  one the entry records, a pin that is not this code's (script, module, python, parameters), a shallow clone and a
  repository with no commit -- each refused before any data call; 3. the listing; 4. the TRAIN/VAL list; 5. hashing;
  6. the outputs.
- The call trace of a run: one os.scandir, one binary read per image and one of the list, two exclusive creates, no
  annotation path, no PIL or cv2 call (PIL.Image.open and the cv2 module raise). PIL and torch are in sys.modules
  through src.tman.schema -> src.eval.artifacts -> src.eval.protocols and src/eval/__init__.py: recorded, accepted.
- Exclusive outputs, both or neither (a refused second create, or a write that fails, leaves nothing the run created;
  an error is exit 4); an --out inside the repository refused; a shuffled os.scandir order gives identical bytes.
- --print-pin: its pin passes manifest.hasher_pin_errors and test_value_errors in the builder's wrapping
  (manifest.py :861), which fails when a HASHER_VALUES key is injected; its parameters are H8's.
- Static cases (H8, H10): the word in no identifier, attribute, key, f-string or path-like literal of the two new
  production files (REASON_LABELS' and ISSUE_ROWS' values and B-1's guard name excepted, each named here); no PIL or
  cv2 import; no src.stats import; the split read by import, never written; expected_images, the suffixes and the
  TRAIN/VAL counts tied to their sources; reads "rb", creates "xb"; one listing call; a no-score scan.

The production split is never exercised: split_of_record is patched to the fixture's "tday_fixture". The fixture
world is a scratch git repository with committed fixture records (the decision log, a freeze entry, the lane
register, the manifest with the hasher's pin, the unlock entry), built with B-1's fixture library, and a data root
of synthetic bytes. Nothing it creates holds the word or ":". Run with TMPDIR set to a new folder:
  TMPDIR=<new folder> python -B scripts/smoke_tman_hasher.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import builtins  # noqa: E402
import contextlib  # noqa: E402
import errno  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["LANGUAGE"] = "C"                         # as DL-90 and smoke_tman_paths
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_dedup as DS  # noqa: E402  (the dedup smoke's list writer)
import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check = FX.check
WORD = "te" + "st"                                   # project rule 4's word, never written whole here
GUARD_CODE = WORD + "_path"                          # B-1's guard's refusal code (schema.py :248-251)
SPLIT = "tday_fixture"
MARKER = "nonexistent_tman_hfx"                      # a top-level folder that is never created
COMMIT = "3" * 40
FREEZE_DAY, UNLOCK_DAY = "2026-11-03", "2026-11-08"
IMAGES = {"kx_t1.jpg": b"T1", "kx_tb.JPG": b"TB", "kx_v1.jpeg": b"V1", "kx_B1.jpg": b"G1", "kx_a1.jpg": b"G1",
          "kx_x10.jpg": b"G2", "kx_x9.jpg": b"G2", "kx_u1.jpg": b"U1"}
OTHERS = {"kx_notes.txt": b"notes", "kx_mask.png": b"png"}          # not images: never listed as one
TRAIN = {"tr_a": b"T1", "tr_c": b"TB", "tr_d": b"T9"}
VAL = {"va_a": b"V1", "va_b": b"TB"}
WANT = {"equal_train": ["kx_t1", "kx_tb"], "equal_val": ["kx_v1"], "duplicate_in_split": ["kx_a1", "kx_x9"]}
WANT_KEPT = ["kx_B1", "kx_u1", "kx_x10"]
DATA_CALLS = ("os.scandir", "os.listdir", "builtins.open", "io.open")
sha = DS.sha


def under(path: str, root) -> bool:
    root = str(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


# ================================================================================================ the world
class World:
    """A scratch repository whose HEAD holds the records the hasher reads -- the decision log, a freeze entry, the lane
    register, the manifest with its image_hasher pin, the unlock entry -- a data root (images/ and annotations/ of
    the fixture split, synthetic bytes), a TRAIN/VAL list in scripts/hash_split_files.py's format and an empty --out.
    The options make one record or input wrong at a time."""

    def __init__(self, *, images=None, raw_names=(), pin_edit=None, unlock_edit=None, row_status="RECORDED",
                 unlock_state="committed", manifest_bytes=None, tv=None):
        from src.tman import entries as E
        from src.tman import manifest as M
        from src.tman import schema as S
        import scripts.tday_hash_images as H
        self.repo, self.data = FX.new_repo(), FX.scratch_dir("tmanhdata")
        self.lists, self.out = FX.scratch_dir("tmanhlist"), FX.scratch_dir("tmanhout")
        self.images = dict(IMAGES if images is None else images)
        self.folder = self.data / "images" / SPLIT
        ann = self.data / "annotations" / SPLIT
        self.folder.mkdir(parents=True)
        ann.mkdir(parents=True)
        for name, content in self.images.items():
            (self.folder / name).write_bytes(content)
            (ann / (os.path.splitext(name)[0] + ".png")).write_bytes(b"mask " + content)
        for name, content in OTHERS.items():
            (self.folder / name).write_bytes(content)
        (self.folder / "kx_dir.jpg").mkdir()
        for raw in raw_names:
            with open(os.fsencode(self.folder) + b"/" + raw, "wb") as fh:
                fh.write(b"raw " + raw)
        self.tv = DS.tv_bytes(TRAIN, VAL) if tv is None else tv
        self.tv_path = self.lists / "train_val_image_hashes.tsv"
        self.tv_path.write_bytes(self.tv)
        pin = H.hasher_pin(COMMIT)
        pin["parameters"]["expected_images"] = len(self.images) + len(raw_names)
        pin["parameters"]["train_val_list"].update(rows=len(TRAIN) + len(VAL), sha256=sha(self.tv), train=len(TRAIN),
                                                   val=len(VAL))
        if pin_edit:
            pin_edit(pin)
        self.pin = pin
        body = {"expected_rows": {"value": S.EXPECTED_ROWS, "cites": list(M.EXPECTED_ROWS_CITES)},
                "generator_pins": {"corrupted_cache": {"pin": None, "value": None},
                                   "image_hasher": {"pin": pin, "value": None}}}
        self.mdata = S.json_file_bytes({"format": S.MANIFEST_FORMAT, "body": body, "appends": [], "session": None}) \
            if manifest_bytes is None else manifest_bytes
        lanes = S.json_file_bytes({"format": S.LANES_FORMAT, "lanes": []})
        freeze = E.make_entry("freeze", [], FREEZE_DAY, {"body_sha256": S.canonical_sha256(body),
                                                         "freeze_date": FREEZE_DAY, "source": "schedule"},
                              amendment="AM-19", script="scripts/tday_manifest.py", commit=COMMIT)
        self.freeze_rel = S.entry_file_rel(E.entry_name(freeze))
        fdata = S.json_file_bytes(freeze)
        m = {"path": S.MANIFEST_REL, "sha256": sha(self.mdata), "body_sha256": S.canonical_sha256(body),
             "appends_sha256": S.canonical_sha256([])}
        if unlock_edit:
            unlock_edit(m)
        unlock = E.make_entry("unlock", [], UNLOCK_DAY, {"manifest": m, "lanes": {"path": S.LANES_REL,
                                                                                  "sha256": sha(lanes)},
                                                         "freeze_record": {"path": self.freeze_rel,
                                                                           "sha256": sha(fdata)}},
                              amendment="AM-19", script="scripts/tday_manifest.py", commit=COMMIT)
        self.unlock_rel = S.entry_file_rel(E.entry_name(unlock))
        udata = S.json_file_bytes(unlock)
        rows = {"DL-301": ("RECORDED", f"TMAN freeze entry: {self.freeze_rel} sha256 {sha(fdata)}"),
                "DL-302": (row_status, f"TMAN unlock entry: {self.unlock_rel} sha256 {sha(udata)}")}
        files = {S.MANIFEST_REL: self.mdata, S.LANES_REL: lanes, self.freeze_rel: fdata,
                 S.DECISION_LOG_REL: FX.dl_table(rows)}
        if unlock_state != "uncommitted":
            files[self.unlock_rel] = udata
        FX.commit_files(self.repo, files, "the records of the unlock")
        if unlock_state == "uncommitted":
            (self.repo / self.unlock_rel).write_bytes(udata)
        elif unlock_state == "rewritten":
            FX.commit_files(self.repo, {self.unlock_rel: udata + b"\n"}, "the unlock entry rewritten")

    def argv(self, **over) -> list:
        a = {"repo": str(self.repo), "unlock_entry": self.unlock_rel, "root": str(self.data),
             "train_val_list": str(self.tv_path), "out": str(self.out)}
        a.update(over)
        return ["hash"] + [x for k, v in a.items() for x in (f"--{k.replace('_', '-')}", v)]

    def data_calls(self, t) -> list:
        """The calls that touch data: a listing or an open under the data root, of the TRAIN/VAL list or under
        --out."""
        return [c for c in t.calls if c[0] in DATA_CALLS
                and (under(c[1], self.data) or c[1] == str(self.tv_path) or under(c[1], self.out))]

    def image_reads(self, t) -> list:
        return [c for c in t.calls if c[0] in ("builtins.open", "io.open") and under(c[1], self.folder)]


# ============================================================================================ tracing and runs
class Stand(types.ModuleType):
    """The cv2 module's stand-in while a run is traced: any attribute is recorded and raises."""

    def __init__(self, log: list):
        super().__init__("cv2")
        self.__dict__["_log"] = log

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self.__dict__["_log"].append(("cv2", name))
        raise AssertionError(f"the hasher used cv2.{name}")


class Shuffled:
    """An os.scandir result whose entries come in a seeded random order (the order is recorded)."""

    def __init__(self, it, seed: int, seen: list):
        self.it, self.entries = it, list(it)
        random.Random(seed).shuffle(self.entries)
        seen.append([e.name for e in self.entries])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.it.close()
        return False

    def __iter__(self):
        return iter(self.entries)


class Trace:
    """Records (name, path, mode) of every os.stat, os.lstat, os.scandir, os.listdir, builtins.open and io.open call on
    a path and calls through (smoke_tman_paths.py's tracer, recording every path); PIL.Image.open and the cv2 module
    raise and are recorded; with `shuffle`, os.scandir results come in that seed's order."""

    NAMES = (("os", "stat"), ("os", "lstat"), ("os", "scandir"), ("os", "listdir"), ("builtins", "open"),
             ("io", "open"))

    def __init__(self, shuffle: int | None = None):
        self.calls, self.image_lib, self.seen, self.saved = [], [], [], {}
        self.shuffle, self.had_cv2 = shuffle, "cv2" in sys.modules

    def __enter__(self):
        import PIL.Image
        mods = {"os": os, "builtins": builtins, "io": io}
        for mod, name in self.NAMES:
            orig = getattr(mods[mod], name)
            self.saved[(mod, name)] = orig

            def wrapper(*a, _orig=orig, _name=f"{mod}.{name}", **kw):
                if a and isinstance(a[0], (str, bytes, os.PathLike)):
                    p = os.fspath(a[0])
                    mode = (a[1] if len(a) > 1 else kw.get("mode", "r")) if _name.endswith("open") else None
                    self.calls.append((_name, os.fsdecode(p) if isinstance(p, bytes) else p, mode))
                out = _orig(*a, **kw)
                if _name == "os.scandir" and self.shuffle is not None:
                    out = Shuffled(out, self.shuffle, self.seen)
                return out
            setattr(mods[mod], name, wrapper)
        self.saved[("PIL.Image", "open")] = PIL.Image.open

        def pil_open(*a, **kw):
            self.image_lib.append(("PIL.Image.open", str(a[0]) if a else ""))
            raise AssertionError("the hasher called PIL.Image.open")
        PIL.Image.open = pil_open
        self.saved_cv2 = sys.modules.get("cv2")
        sys.modules["cv2"] = Stand(self.image_lib)
        return self

    def __exit__(self, *exc):
        import PIL.Image
        mods = {"os": os, "builtins": builtins, "io": io}
        for (mod, name), orig in self.saved.items():
            if mod == "PIL.Image":
                PIL.Image.open = orig
            else:
                setattr(mods[mod], name, orig)
        if self.had_cv2:
            sys.modules["cv2"] = self.saved_cv2
        else:
            sys.modules.pop("cv2", None)
        return False


class Resolving:
    """os.path.realpath sends `target` alone to a location holding the word (smoke_tman_paths.py :84-100)."""

    HIT = f"/{MARKER}/resolved/la" + WORD + "/target"

    def __init__(self, target):
        self.target, self.saved = os.fspath(target), None

    def __enter__(self):
        self.saved = saved = os.path.realpath
        os.path.realpath = lambda q, *a, **kw: self.HIT if os.fspath(q) == self.target else saved(q, *a, **kw)
        return self

    def __exit__(self, *exc):
        os.path.realpath = self.saved
        return False


class FailOpen:
    """builtins.open raises PermissionError for one path (an unreadable image)."""

    def __init__(self, target):
        self.target, self.saved = str(target), None

    def __enter__(self):
        self.saved = saved = builtins.open

        def opener(f, *a, **kw):
            if isinstance(f, (str, os.PathLike)) and os.fspath(f) == self.target:
                raise PermissionError(13, "fixture: unreadable", self.target)
            return saved(f, *a, **kw)
        builtins.open = opener
        return self

    def __exit__(self, *exc):
        builtins.open = self.saved
        return False


class FullDisk:
    """builtins.open, for one path, gives a file whose write puts down half the bytes and raises OSError ENOSPC (the
    disk fills up during the write)."""

    class File:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.fh.close()
            return False

        def write(self, data):
            self.fh.write(data[:len(data) // 2])
            raise OSError(errno.ENOSPC, "fixture: no space left on the device")

    def __init__(self, target):
        self.target, self.saved = str(target), None

    def __enter__(self):
        self.saved = saved = builtins.open

        def opener(f, *a, **kw):
            fh = saved(f, *a, **kw)
            return self.File(fh) if isinstance(f, (str, os.PathLike)) and os.fspath(f) == self.target else fh
        builtins.open = opener
        return self

    def __exit__(self, *exc):
        builtins.open = self.saved
        return False


def run(argv: list, *, split: str = SPLIT, ctx=None) -> tuple[int, str]:
    """The hasher's main(argv) with split_of_record patched to `split` (the production split is never exercised),
    stdout captured, inside `ctx` (a Trace, Resolving, FailOpen, FullDisk or a list of them); argparse's exit is its
    code."""
    import scripts.tday_hash_images as H
    saved, buf = H.split_of_record, io.StringIO()
    H.split_of_record = lambda: split
    try:
        with contextlib.ExitStack() as stack:
            for c in (ctx if isinstance(ctx, list) else [ctx] if ctx is not None else []):
                stack.enter_context(c)
            with contextlib.redirect_stdout(buf):
                rc = H.main(argv)
    except SystemExit as e:
        rc = e.code
    except Exception as e:  # noqa: BLE001 -- main never raises; a crash fails the checks that read rc
        rc, buf = f"raised {type(e).__name__}: {e}", io.StringIO()
    finally:
        H.split_of_record = saved
    return rc, buf.getvalue()


def refused(rc, txt: str, code: str) -> bool:
    return rc == 2 and f"REFUSED [{code}]" in txt


# ============================================================================================== a full run
def expected_tsv(images: dict) -> bytes:
    rows = sorted((os.path.splitext(n)[0], n, c) for n, c in images.items())
    return ("stem\tfile\tbytes\tsha256\n" + "".join(f"{s}\t{n}\t{len(c)}\t{sha(c)}\n" for s, n, c in rows)).encode()


def case_pass_and_trace():
    from src.tman import dedup as D
    from src.tman import schema as S
    import scripts.tday_hash_images as H
    w = World()
    t = Trace()
    rc, txt = run(w.argv(), ctx=t)
    made = sorted(p.name for p in w.out.iterdir())
    check("hash_writes_both_outputs_exit_0", rc == 0 and made == [D.DEDUP_NAME, D.HASH_LIST_NAME],
          f"{rc} {made} {txt[-300:]}")
    tsv = (w.out / D.HASH_LIST_NAME).read_bytes() if D.HASH_LIST_NAME in made else b""
    data = (w.out / D.DEDUP_NAME).read_bytes() if D.DEDUP_NAME in made else b"{}"
    doc = json.loads(data)
    check("hash_list_is_the_images_sorted_by_stem_H6", tsv == expected_tsv(IMAGES))
    check("dedup_rule_on_the_images_H3", doc.get("removed_ids_by_reason") == WANT
          and doc.get("kept_ids") == WANT_KEPT and doc.get("also_counted") == {"equal_train_and_val": 1}, str(doc)[:300])
    check("dedup_names_its_hash_list_train_val_list_and_script", doc.get("hash_list_sha256") == sha(tsv)
          and doc.get("train_val_list_sha256") == sha(w.tv)
          and doc.get("script") == {"path": "scripts/tday_hash_images.py", "sha256": sha(Path(H.__file__).read_bytes())})
    got = DS.outcome(D.read_dedup, data, expect_sha256=sha(data))
    check("dedup_file_passes_the_strict_reader_H5", got[0] == "ok", str(got)[:200])
    want_lines = [f"hash_list_sha256 {sha(tsv)}", f"dedup_list_sha256 {sha(data)}",
                  "counts_by_reason " + S.canonical_json_bytes(doc.get("removed_by_reason")).decode("utf-8"),
                  D.issue_counts_line()]
    check("prints_exactly_hasher_values_and_the_issue_counts_H7", txt.splitlines() == want_lines, txt[-400:])
    check("prints_no_id_H7", "kx_" not in txt and not any(os.path.splitext(n)[0] in txt for n in IMAGES))
    check("suffix_rule_upper_case_kept_png_txt_and_folder_left", b"kx_tb\tkx_tb.JPG" in tsv
          and b"kx_v1\tkx_v1.jpeg" in tsv and b"kx_mask" not in tsv and b"kx_notes" not in tsv
          and b"kx_dir" not in tsv)
    scandirs = [c for c in t.calls if c[0] == "os.scandir"]
    check("trace_one_scandir_of_the_image_folder", scandirs == [("os.scandir", str(w.folder), None)], str(scandirs))
    check("trace_no_listdir", not [c for c in t.calls if c[0] == "os.listdir"])
    reads = w.image_reads(t)
    check("trace_one_binary_read_per_image", sorted(c[1] for c in reads) == sorted(str(w.folder / n) for n in IMAGES)
          and all(c[2] == "rb" for c in reads), str(reads)[:300])
    lists = [c for c in t.calls if c[0] in ("builtins.open", "io.open") and c[1] == str(w.tv_path)]
    check("trace_train_val_list_read_once_in_binary", [c[2] for c in lists] == ["rb"], str(lists))
    outs = sorted((c[1], c[2]) for c in t.calls if c[0] in ("builtins.open", "io.open") and under(c[1], w.out))
    check("trace_outputs_created_exclusively_xb", outs == sorted([(str(w.out / D.DEDUP_NAME), "xb"),
                                                                  (str(w.out / D.HASH_LIST_NAME), "xb")]), str(outs))
    check("trace_no_annotation_path", not any("annotation" in c[1] for c in t.calls))
    other = [c for c in w.data_calls(t) if c not in scandirs and c not in reads and c not in lists
             and not under(c[1], w.out)]
    check("trace_no_other_data_call", not other, str(other)[:300])
    check("trace_no_pil_or_cv2_call", not t.image_lib, str(t.image_lib))
    check("pil_and_torch_loaded_by_b1_schema_recorded_accepted", "PIL.Image" in sys.modules and "torch" in sys.modules
          and "src.eval.protocols" in sys.modules)
    check("no_src_stats_module_loaded", not [m for m in sys.modules if m == "src.stats" or m.startswith("src.stats.")])


# ===================================================================================== step 1: the paths
def case_step1_paths():
    w = World()
    bad = {"repo": f"/{MARKER}/{WORD}_repo", "unlock_entry": f"{MARKER}/{WORD}_unlock.json",
           "root": f"/{MARKER}/{WORD}_root", "train_val_list": f"/{MARKER}/{WORD}_list.tsv",
           "out": f"/{MARKER}/{WORD}_out"}
    for k, v in list(bad.items()) + [("root_upper_case", f"/{MARKER}/{WORD.upper()}_root")]:
        t = Trace()
        rc, txt = run(w.argv(**{k.replace("_upper_case", ""): v}), ctx=t)
        marked = [c for c in t.calls if MARKER in c[1]]
        check(f"step1_{k}_holding_the_word_refused_with_no_call", refused(rc, txt, GUARD_CODE) and not marked,
              f"{rc} {txt[-160:]} {marked[:2]}")
    plain = f"/{MARKER}/plain_root"
    t = Trace()
    rc, txt = run(w.argv(root=plain), ctx=[Resolving(plain), t])
    check("step1_root_resolving_to_the_word_refused", refused(rc, txt, GUARD_CODE)
          and not [c for c in t.calls if MARKER in c[1]], f"{rc} {txt[-160:]}")
    rel = f"{MARKER}_plain/unlock.json"
    t = Trace()
    rc, txt = run(w.argv(unlock_entry=rel), ctx=[Resolving(os.path.join(str(w.repo), rel)), t])
    check("step1_unlock_entry_resolved_under_repo", refused(rc, txt, GUARD_CODE)
          and not [c for c in t.calls if MARKER in c[1]], f"{rc} {txt[-160:]}")
    check("marker_never_created", not os.path.lexists(f"/{MARKER}"))


# ===================================================================================== step 2: the records
def case_step2_records():
    from src.tman import schema as S
    import scripts.tday_hash_images as H

    def m_edit(key, value):
        return lambda m: m.update({key: value})

    def module(p):
        p["parameters"]["modules"] = {H.DEDUP_REL: "8" * 64}

    def script_sha(p):
        p["script"]["sha256"] = "8" * 64

    def script_path(p):
        p["script"]["path"] = "scripts/tday_hash_images_v2.py"

    def python(p):
        p["python"] = "3.12"

    def suffixes(p):
        p["parameters"]["suffixes"] = [".jpg"]

    def no_modules(p):
        del p["parameters"]["modules"]

    no_body = S.json_file_bytes({"format": S.MANIFEST_FORMAT, "appends": [], "session": None})
    variants = (
        ("unlock_entry_absent", {}, {"unlock_entry": "reports/derived/tday_entries/unlock__session__20261108__09.json"},
         "record_uncommitted"),
        ("unlock_entry_uncommitted", {"unlock_state": "uncommitted"}, {}, "record_uncommitted"),
        ("unlock_row_not_recorded", {"row_status": "DECIDED"}, {}, "dl_row_missing"),
        ("unlock_entry_rewritten", {"unlock_state": "rewritten"}, {}, "entry_file_rewritten"),
        ("unlock_entry_of_another_kind", {}, {"unlock_entry": "FREEZE"}, "entry_file_invalid"),
        ("manifest_path_other", {"unlock_edit": m_edit("path", "reports/derived/tday_manifest_v2.json")}, {},
         "unlock_manifest_mismatch"),
        ("manifest_sha256_other", {"unlock_edit": m_edit("sha256", "7" * 64)}, {}, "unlock_manifest_mismatch"),
        ("manifest_body_sha256_other", {"unlock_edit": m_edit("body_sha256", "7" * 64)}, {},
         "unlock_manifest_mismatch"),
        ("manifest_appends_sha256_other", {"unlock_edit": m_edit("appends_sha256", "7" * 64)}, {},
         "unlock_manifest_mismatch"),
        ("manifest_not_json", {"manifest_bytes": b"not a manifest\n"}, {}, "manifest_unreadable"),
        ("manifest_without_body", {"manifest_bytes": no_body}, {}, "manifest_format"),
        ("pin_script_sha256_other", {"pin_edit": script_sha}, {}, "hasher_pin_mismatch"),
        ("pin_script_path_other", {"pin_edit": script_path}, {}, "hasher_pin_mismatch"),
        ("pin_module_sha256_other", {"pin_edit": module}, {}, "hasher_pin_mismatch"),
        ("pin_python_other", {"pin_edit": python}, {}, "hasher_pin_mismatch"),
        ("pin_parameters_other", {"pin_edit": suffixes}, {}, "hasher_pin_mismatch"),
        ("pin_without_modules", {"pin_edit": no_modules}, {}, "hasher_pin_mismatch"),
        ("pin_absent", {"pin_edit": lambda p: p.clear()}, {}, "hasher_pin_mismatch"))
    for name, opts, over, code in variants:
        w = World(**opts)
        if over.get("unlock_entry") == "FREEZE":
            over = {"unlock_entry": w.freeze_rel}
        t = Trace()
        rc, txt = run(w.argv(**over), ctx=t)
        check(f"step2_{name}_refused_before_any_data_call", refused(rc, txt, code) and not w.data_calls(t)
              and not any(w.out.iterdir()), f"{rc} {txt[-200:]} {w.data_calls(t)[:2]}")


def case_step2_repository():
    """--repo a depth-1 clone of the records (B-1's written-once check cannot see a rewrite in it) or a repository
    with no commit: refused before the unlock entry is read and before any data call."""
    w = World()
    clone = FX.scratch_dir("tmanhclone")
    FX.git(clone, "clone", "-q", "--depth", "1", w.repo.resolve().as_uri(), ".")
    shallow = FX.git(clone, "rev-parse", "--is-shallow-repository")
    t = Trace()
    rc, txt = run(w.argv(repo=str(clone)), ctx=t)
    check("step2_shallow_clone_refused_before_any_data_call", shallow == "true"
          and refused(rc, txt, "repository_shallow") and not w.data_calls(t) and not any(w.out.iterdir()),
          f"{shallow} {rc} {txt[-200:]}")
    t = Trace()
    rc, txt = run(w.argv(repo=str(FX.new_repo())), ctx=t)
    check("step2_repository_without_a_commit_refused_before_any_data_call", refused(rc, txt, "record_uncommitted")
          and "no git repository with a HEAD commit" in txt and not w.data_calls(t) and not any(w.out.iterdir()),
          f"{rc} {txt[-200:]}")


# ===================================================================================== step 3: the listing
def case_step3_listing():
    from src.tman import dedup as D
    variants = (("image_folder_absent", {}, "tday_absent", "images_unreadable"),
                ("two_images_one_stem", {"images": dict(IMAGES, **{"kx_u1.jpeg": b"U2"})}, SPLIT,
                 "image_ids_invalid"),
                ("stems_equal_once_case_folded", {"images": dict(IMAGES, **{"KX_U1.jpg": b"U3"})}, SPLIT,
                 "image_ids_invalid"),
                ("stem_with_surrounding_whitespace", {"images": dict(IMAGES, **{" kx_w.jpg": b"W1"})}, SPLIT,
                 "image_ids_invalid"),
                ("name_holding_a_tab", {"images": dict(IMAGES, **{"kx\tw.jpg": b"W2"})}, SPLIT, "image_ids_invalid"),
                ("name_not_utf8", {"raw_names": (b"kx_\xff.jpg",)}, SPLIT, "image_ids_invalid"),
                ("image_count_other", {"pin_edit": lambda p: p["parameters"].update(expected_images=9)}, SPLIT,
                 "image_count_mismatch"),
                ("image_count_one_short", {"pin_edit": lambda p: p["parameters"].update(expected_images=7)}, SPLIT,
                 "image_count_mismatch"))
    for name, opts, split, code in variants:
        w = World(**opts)
        t = Trace()
        rc, txt = run(w.argv(), split=split, ctx=t)
        listed = [c for c in t.calls if c[0] == "os.scandir"]
        check(f"step3_{name}_refused_before_any_image_or_list_read", refused(rc, txt, code) and not w.image_reads(t)
              and not [c for c in w.data_calls(t) if c[1] == str(w.tv_path)] and len(listed) == 1
              and not any(w.out.iterdir()), f"{rc} {txt[-200:]}")
    check("image_id_rules_are_dedup_s", D.image_id_errors([("a", "a.jpg"), ("A", "A.jpg")]) != [])


# ================================================================================ step 4: the TRAIN/VAL list
def case_step4_list():
    variants = (("list_sha256_not_the_pin_s", {"pin_edit": lambda p: p["parameters"]["train_val_list"].update(
        sha256="9" * 64)}, {}, "train_val_list_mismatch"),
                ("list_absent", {}, {"train_val_list": "ABSENT"}, "train_val_list_mismatch"),
                ("list_header_other", {"tv": DS.tv_bytes(TRAIN, VAL, header=DS.HSF_HEADER.replace("sha1", "sha_1"))},
                 {}, "train_val_list_invalid"),
                ("list_rows_not_the_pin_s", {"pin_edit": lambda p: p["parameters"]["train_val_list"].update(rows=6)},
                 {}, "train_val_list_invalid"),
                ("list_split_counts_not_the_pin_s", {"pin_edit": lambda p: p["parameters"]["train_val_list"].update(
                    train=2, val=3)}, {}, "train_val_list_invalid"))
    for name, opts, over, code in variants:
        w = World(**opts)
        if over.get("train_val_list") == "ABSENT":
            over = {"train_val_list": str(w.lists / "absent_list.tsv")}
        t = Trace()
        rc, txt = run(w.argv(**over), ctx=t)
        check(f"step4_{name}_refused_before_any_image_read", refused(rc, txt, code) and not w.image_reads(t)
              and not any(w.out.iterdir()), f"{rc} {txt[-200:]}")


# ====================================================================================== step 5: hashing
def case_step5_hashing():
    w = World()
    t = Trace()
    rc, txt = run(w.argv(), ctx=[FailOpen(w.folder / "kx_u1.jpg"), t])
    check("step5_unreadable_image_refused_nothing_written", refused(rc, txt, "images_unreadable")
          and not any(w.out.iterdir()) and "kx_u1" not in txt, f"{rc} {txt[-200:]}")
    check("step5_no_image_library_call", not t.image_lib, str(t.image_lib))


# ====================================================================================== step 6: outputs
def case_step6_outputs():
    from src.tman import dedup as D
    w = World()
    inside = w.repo / "reports"
    rc, txt = run(w.argv(out=str(inside)))
    check("step6_out_inside_the_repository_refused", refused(rc, txt, "output_location")
          and "inside the repository" in txt and not (inside / D.HASH_LIST_NAME).exists()
          and not (inside / D.DEDUP_NAME).exists(), txt[-200:])
    here = REPO / "nonexistent_tman_hout"
    rc, txt = run(w.argv(out=str(here)))
    check("step6_out_inside_this_checkout_refused", refused(rc, txt, "output_location")
          and "inside the repository" in txt and not os.path.lexists(here), txt[-200:])
    rc, txt = run(w.argv(out=str(w.lists / "absent_out")))
    check("step6_out_not_an_existing_folder_refused", refused(rc, txt, "output_location")
          and "not an existing folder" in txt, txt[-200:])
    rc, txt = run(w.argv())
    first = {p.name: p.read_bytes() for p in w.out.iterdir()}
    rc2, txt2 = run(w.argv())
    check("step6_second_run_refused_first_outputs_kept", rc == 0 and refused(rc2, txt2, "output_exists")
          and {p.name: p.read_bytes() for p in w.out.iterdir()} == first and len(first) == 2, txt2[-200:])
    w2 = World()
    (w2.out / D.DEDUP_NAME).write_bytes(b"older")
    t = Trace()
    rc, txt = run(w2.argv(), ctx=t)
    opened = [c for c in t.calls if c[0] in ("builtins.open", "io.open")
              and (under(c[1], w2.out) or under(c[1], w2.out.resolve()))]
    check("step6_one_output_present_refused_none_created", refused(rc, txt, "output_exists")
          and sorted(p.name for p in w2.out.iterdir()) == [D.DEDUP_NAME]
          and (w2.out / D.DEDUP_NAME).read_bytes() == b"older" and not opened, f"{txt[-200:]} {opened[:2]}")
    w3 = World()
    (w3.out / D.HASH_LIST_NAME).write_bytes(b"older")
    saved = os.path.lexists
    try:
        os.path.lexists = lambda p: False                   # the pre-check misses it: the exclusive open must refuse
        rc, txt = run(w3.argv())
    finally:
        os.path.lexists = saved
    check("step6_exclusive_create_refuses_an_existing_file_xb", refused(rc, txt, "output_exists")
          and (w3.out / D.HASH_LIST_NAME).read_bytes() == b"older", txt[-200:])
    w4 = World()
    (w4.out / D.DEDUP_NAME).write_bytes(b"older")
    try:
        os.path.lexists = lambda p: False                   # the pre-check misses it: the second create refuses
        rc, txt = run(w4.argv())
    finally:
        os.path.lexists = saved
    left = sorted(p.name for p in w4.out.iterdir())
    check("step6_second_create_refused_first_output_removed", refused(rc, txt, "output_exists")
          and left == [D.DEDUP_NAME] and (w4.out / D.DEDUP_NAME).read_bytes() == b"older", f"{txt[-200:]} {left}")
    w5 = World()
    rc, txt = run(w5.argv(), ctx=FullDisk(w5.out.resolve() / D.DEDUP_NAME))
    check("step6_write_failure_exit_4_leaves_neither_output", rc == 4 and "ERROR -- OSError" in txt
          and not any(w5.out.iterdir()), f"{rc} {txt[-200:]} {sorted(p.name for p in w5.out.iterdir())}")


def case_shuffled_scandir():
    from src.tman import dedup as D
    w = World()
    outs, seen = [], []
    for seed in (11, 12):
        out = FX.scratch_dir("tmanhshuf")
        t = Trace(shuffle=seed)
        rc, txt = run(w.argv(out=str(out)), ctx=t)
        outs.append((rc, (out / D.HASH_LIST_NAME).read_bytes() if rc == 0 else b"",
                     (out / D.DEDUP_NAME).read_bytes() if rc == 0 else b""))
        seen += t.seen
    check("shuffled_scandir_orders_differ", len(seen) == 2 and seen[0] != seen[1], str(seen)[:200])
    check("shuffled_scandir_gives_identical_bytes", outs[0][0] == outs[1][0] == 0 and outs[0] == outs[1])


# ======================================================================================= --print-pin (H8)
def case_print_pin():
    from src.tman import dedup as D
    from src.tman import manifest as M
    from src.tman import schema as S
    from src.eval import stage_artifacts as SA
    from src.eval import teacher_diag as TD
    import scripts.tday_hash_images as H
    t = Trace()
    rc, txt = run(["--print-pin", "--commit", COMMIT], ctx=t)
    try:
        pin = json.loads(txt)
    except ValueError:
        pin = None
    check("print_pin_exit_0_prints_the_pin_function_s", rc == 0 and pin == H.hasher_pin(COMMIT), f"{rc} {txt[-200:]}")
    pin = pin if isinstance(pin, dict) else H.hasher_pin(COMMIT)
    check("print_pin_passes_hasher_pin_errors_609", M.hasher_pin_errors(pin) == [], str(M.hasher_pin_errors(pin)))
    wrapped = {"image_hasher": {"pin": pin, "value": None}}
    check("print_pin_passes_the_builder_s_wrapping_861", M.test_value_errors(wrapped) == [],
          str(M.test_value_errors(wrapped)))
    for key in M.HASHER_VALUES:
        bad = json.loads(json.dumps(pin))
        bad["parameters"][key] = "9" * 64
        check(f"wrapping_fails_with_{key}_injected", M.test_value_errors(
            {"image_hasher": {"pin": bad, "value": None}}) != [])
    keys = set(DS.keys_of(pin))
    check("no_pin_key_is_a_pin_or_hasher_value_key", not keys & set(M.PIN_VALUES + M.HASHER_VALUES), str(keys))
    check("pin_script_is_its_path_commit_and_bytes", pin["script"] == {
        "path": "scripts/tday_hash_images.py", "commit": COMMIT, "sha256": sha(Path(H.__file__).read_bytes())})
    check("pin_python_3_11", pin["python"] == "3.11")
    p = pin["parameters"]
    check("pin_parameters_are_h8_s", sorted(p) == sorted(["folder", "split", "suffixes", "digest", "expected_images",
                                                           "train_val_list", "rule", "outputs", "modules"]))
    check("pin_folder_split_suffixes_digest", p["folder"] == "images/{split}"
          and p["split"] == {"from": "src.eval.stage_artifacts.OFFICIAL_SPLIT"} and p["suffixes"] == [".jpeg", ".jpg"]
          and p["digest"] == "sha256 of the file bytes")
    check("pin_expected_images_is_schema_s_and_official_rows", p["expected_images"] == S.EXPECTED_ROWS
          == SA.OFFICIAL_ROWS == 1561)
    check("pin_train_val_list_is_b1_s_with_the_split_counts", p["train_val_list"] == dict(
        M.TRAINVAL_HASH_LIST, train=TD.TRAIN_ROWS, val=TD.VAL_ROWS) and TD.TRAIN_ROWS + TD.VAL_ROWS
        == M.TRAINVAL_HASH_LIST["rows"] == 6213)
    check("pin_rule_and_outputs", p["rule"]["precedence"] == list(D.REASONS) and set(p["rule"]) == {
        "precedence", "ids", "keep"} and p["outputs"] == {"hash_list": "tday_image_hashes.tsv",
                                                          "dedup": "tday_dedup.json", "dedup_format": "tday_dedup/1"})
    check("pin_modules_hold_the_dedup_module_s_sha256", p["modules"] == {"src/tman/dedup.py":
                                                                         sha(Path(D.__file__).read_bytes())})
    opened = sorted({c[1] for c in t.calls if c[0] in ("builtins.open", "io.open")})
    check("print_pin_reads_no_data", not [c for c in t.calls if c[0] in ("os.scandir", "os.listdir")]
          and opened == sorted({str(Path(H.__file__)), str(Path(D.__file__))}), str(opened))
    for name, argv in (("commit_upper_case_refused", ["--print-pin", "--commit", "A" * 40]),
                       ("commit_of_39_hex_refused", ["--print-pin", "--commit", "3" * 39])):
        rc, txt = run(argv)
        check(f"print_pin_{name}", refused(rc, txt, "generator_pin_incomplete"), f"{rc} {txt[-160:]}")
    for name, argv in (("both_modes", ["--print-pin", "--commit", COMMIT, "hash", "--repo", "r", "--unlock-entry", "u",
                                       "--root", "d", "--train-val-list", "l", "--out", "o"]),
                       ("neither_mode", []), ("print_pin_without_commit", ["--print-pin"])):
        with contextlib.redirect_stderr(io.StringIO()):
            rc, _ = run(argv)
        check(f"usage_{name}_exit_2", rc == 2, str(rc))


# =============================================================================================== static (H8, H10)
def production_files() -> dict:
    from src.tman import dedup as D
    import scripts.tday_hash_images as H
    return {"dedup": Path(D.__file__), "hasher": Path(H.__file__)}


GUARD_ATTRS = {"refuse_" + WORD + "_path"}            # B-1's guard (schema.py :248), the one name allowed


def allowed_values(tree) -> dict:
    """{id: value} of the string constants in the values of REASON_LABELS and ISSUE_ROWS, the plan's and the issue's
    labels (the only literals allowed to hold the word)."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("REASON_LABELS", "ISSUE_ROWS")
                                             for t in n.targets) and isinstance(n.value, ast.Dict):
            for v in n.value.values:
                for c in ast.walk(v):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        out[id(c)] = c.value
    return out


def word_violations(tree) -> list:
    allowed = allowed_values(tree)
    bad = []
    for n in ast.walk(tree):
        names = []
        if isinstance(n, ast.Name):
            names = [n.id]
        elif isinstance(n, ast.Attribute):
            names = [] if n.attr in GUARD_ATTRS else [n.attr]
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, ast.arg):
            names = [n.arg]
        elif isinstance(n, ast.keyword) and n.arg:
            names = [n.arg]
        elif isinstance(n, ast.alias):
            names = [n.name, n.asname or ""]
        elif isinstance(n, ast.ImportFrom):
            names = [n.module or ""]
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names = list(n.names)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in allowed:
            names = [n.value]
        bad += [x for x in names if WORD in x.lower()]
    return bad


def case_static():
    from src.tman import dedup as D
    from src.tman import manifest as M
    import scripts.hash_split_files as HSF
    import scripts.tday_hash_images as H
    files = production_files()
    trees = {k: ast.parse(p.read_text(encoding="utf-8")) for k, p in files.items()}
    for k, tree in trees.items():
        bad = word_violations(tree)
        check(f"h10_{k}_no_word_in_identifiers_attributes_keys_fstrings_paths", not bad, str(bad))
    labels = sorted(allowed_values(trees["dedup"]).values())
    check("h10_allowed_literals_are_the_named_labels", sorted(x for x in labels if WORD in x.lower()) == sorted(
        ["duplicate " + WORD.upper() + " image", f"{WORD} ↔ train", f"{WORD} ↔ val",
         f"{WORD} ↔ {WORD}"]) and not allowed_values(trees["hasher"]), str(labels))
    for k, p in files.items():
        text = p.read_text(encoding="utf-8").lower()
        want = sum(v.lower().count(WORD) for v in allowed_values(trees[k]).values()) + text.count(
            "refuse_" + WORD + "_path") * 1
        check(f"h10_{k}_word_only_in_the_named_labels_and_the_guard_comments_included", text.count(WORD) == want,
              f"{text.count(WORD)} != {want}")
    imports = {k: sorted({a.name for n in ast.walk(t) if isinstance(n, ast.Import) for a in n.names}
                         | {n.module or "" for n in ast.walk(t) if isinstance(n, ast.ImportFrom)})
               for k, t in trees.items()}
    check("h9_no_pil_or_cv2_import_statement", not [m for ms in imports.values() for m in ms
                                                    if m.split(".")[0] in ("PIL", "cv2")], str(imports))
    check("h10_no_src_stats_import", not [m for ms in imports.values() for m in ms
                                          if m == "src.stats" or m.startswith("src.stats.")], str(imports))
    fn = next(n for n in ast.walk(trees["hasher"]) if isinstance(n, ast.FunctionDef) and n.name == "split_of_record")
    inner = [(n.module, [a.name for a in n.names]) for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)]
    check("h0_split_read_by_import_inside_the_resolver", inner == [("src.eval.stage_artifacts", ["OFFICIAL_SPLIT"])]
          and H.SPLIT_FROM == "src.eval.stage_artifacts.OFFICIAL_SPLIT"
          and not [n for n in ast.walk(trees["hasher"]) if isinstance(n, ast.ImportFrom)
                   and n.module == "src.eval.stage_artifacts" and n not in ast.walk(fn)], str(inner))
    check("h9_suffixes_are_hash_split_files_50", set(H.SUFFIXES) == set(HSF.IMAGE_SUFFIXES))
    modes = sorted(n.args[1].value for n in ast.walk(trees["hasher"]) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "open" and len(n.args) > 1
                   and isinstance(n.args[1], ast.Constant))
    check("h9_reads_rb_creates_xb_only", modes == ["rb", "rb", "xb"], str(modes))
    listing = sorted((k, n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id) for k, t in trees.items()
                     for n in ast.walk(t) if isinstance(n, ast.Call)
                     and (n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", ""))
                     in ("scandir", "listdir", "walk", "glob", "iglob", "rglob", "iterdir"))
    check("h9_one_listing_call_site_os_scandir", listing == [("hasher", "scandir")], str(listing))
    hits = [s for p in files.values() for s in DS.code_literals(p) if FX.SCORE_KEY_RE.search(s)]
    check("noscore_static_scan_of_both_files", not hits, str(hits))
    from src.tman import schema as S
    check("created_names_pass_b1_s_guard", all(S.created_name_error(n) is None for n in (
        D.HASH_LIST_NAME, D.DEDUP_NAME, SPLIT)) and M.HASHER_VALUES == D.VALUE_KEYS)


def case_created_names():
    """Last: nothing this smoke created holds the word or ':' (its scratch folders, walked before their removal)."""
    bad = []
    for root in FX.SCRATCH_DIRS:
        for d, dirs, names in os.walk(root):
            for x in dirs + names:
                if WORD in x.lower() or ":" in x:
                    bad.append(os.path.join(d, x))
    check("nothing_created_holds_the_word_or_a_colon", not bad and len(FX.SCRATCH_DIRS) > 10, str(bad[:3]))


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN HASHER SMOKE (lane TMAN, B-3a part 1, B3A-2)",
                                  (case_pass_and_trace, case_step1_paths, case_step2_records, case_step2_repository,
                                   case_step3_listing,
                                   case_step4_list, case_step5_hashing, case_step6_outputs, case_shuffled_scandir,
                                   case_print_pin, case_static, case_created_names)))
