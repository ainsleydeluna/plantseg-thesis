#!/usr/bin/env python3
"""DL-49 TRAIN+VAL aspect-ratio census and DL-55 teacher dataset suffix census (S4).

    python -B scripts/split_censuses.py --census aspect-ratio --data-root DIR --out-dir DIR \\
        (--script-commit SHA --script-commit-dl-id DL-n | --synthetic-inputs [--generated-utc T]
         [--expect-train N --expect-val N])
    python -B scripts/split_censuses.py --census teacher-suffix --data-root DIR --teacher-records FILE \\
        --out-dir DIR [--expect-records-sha256 HEX] (binding | synthetic flags as above)

It lists exactly two folders, <data-root>/images/train and <data-root>/images/val, by explicit path with one
os.scandir each (no recursion: a subdirectory is counted, never entered); the data root itself, annotations
and TEST are never listed or named. Every path string containing "test" (case-insensitive) is refused before
any filesystem call (teacher_diag.refuse_test_names), then again once resolved (teacher_diag.refuse_test_path);
so is any listed name containing "test".

aspect-ratio (DL-49; docs/IMPLEMENTATION_CONTRACT.md B3 "Zero-valid samples", :370): the images under the
evaluator's and training loader's rule (suffix .jpg or .jpeg, case-insensitive: scripts/hash_split_files.py
IMAGE_SUFFIXES; src/eval/adapters.py list_split_stems), which must number 5,367 and 846 with unique stems.
Each image's header is read with PIL Image.open(...).size; no pixel is decoded. AR = max(W, H) / min(W, H),
invariant under EXIF 90-degree rotations. Counted strictly above 12.6 and 26.5, in integers (10 L > 126 S;
10 L > 265 S). An unreadable header is a STOP (exit 1, nothing written).

teacher-suffix (DL-55, A1 N3): per folder the exact-suffix histogram (case-sensitive), the teacher's rule
name.endswith(".jpg") (the teacher config's img_suffix, configs/teacher/...:207) and the student's rule, their
stem sets, and the subdirectory count; the run of record's teacher_selection_records.jsonl (its basename must
be src/training/teacher_components.py RECORDS_FILE), read strictly: 10 rows at iterations 4,000 ... 40,000, val_images
846 in every row, one val_manifest_sha256 equal to the census's own hash of the name-sorted exact-suffix VAL
stems (hash_split_manifest over ManifestEntry(i, stem, stem): the pass order the teacher hook hashes,
teacher_components.py:214-224; "BaseSegDataset sorts by img_path", teacher config :249). The census is written
whatever it finds; exit 1 when the expectation is not met.

Outputs (serialized first, created exclusively; "synthetic_" prefix for synthetic inputs):
  <out-dir>/train_val_aspect_ratio_census.json     (aspect-ratio)
  <out-dir>/teacher_dataset_suffix_census.json     (teacher-suffix)
They record no out-dir, hostname or wall time: a rerun with the same inputs, code and stamp is byte-identical.

Exit codes: 0 written (teacher-suffix: expectation met); 1 a failed check (aspect-ratio: an unreadable header,
nothing written; teacher-suffix: expectation not met, the census written); 2 a refusal; 4 an unexpected error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

from scripts.hash_split_files import IMAGE_SUFFIXES  # noqa: E402
from src.eval import teacher_diag as td  # noqa: E402

SCRIPT = "scripts/split_censuses.py"
SPLITS = ("train", "val")
FOLDERS = {"train": ("images", "train"), "val": ("images", "val")}
AR_NAME, SUFFIX_NAME, SYNTHETIC_PREFIX = ("train_val_aspect_ratio_census.json",
                                          "teacher_dataset_suffix_census.json", "synthetic_")
TEACHER_IMG_SUFFIX = ".jpg"          # the teacher config's img_suffix (smoke T6 checks it by ast)
RECORDS_FILE = "teacher_selection_records.jsonl"   # src/training/teacher_components.py:55 (smoke T6)
TEACHER_MAX_ITERS, TEACHER_VAL_INTERVAL = 40000, 4000   # teacher config :331 and :334 (smoke T6)
EXPECTED_ITERATIONS = list(range(TEACHER_VAL_INTERVAL, TEACHER_MAX_ITERS + 1, TEACHER_VAL_INTERVAL))
THRESHOLDS = (("12.6", 126, 10), ("26.5", 265, 10))     # IMPLEMENTATION_CONTRACT.md B3 :370
DL68 = "DL-68, docs/DECISION_LOG.md:127"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CODE_FILES = ("scripts/split_censuses.py", "scripts/hash_split_files.py", "src/eval/teacher_diag.py",
              "src/eval/artifacts.py", "src/eval/evaluate.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DL-49 aspect-ratio and DL-55 teacher suffix censuses (S4).")
    p.add_argument("--census", required=True, choices=["aspect-ratio", "teacher-suffix"])
    p.add_argument("--data-root", required=True, help="holds images/train and images/val; only these are listed")
    p.add_argument("--teacher-records", help="teacher-suffix: the run of record's teacher_selection_records.jsonl")
    p.add_argument("--expect-records-sha256", help="teacher-suffix, optional: the records file's sha256 (DL-68)")
    p.add_argument("--out-dir", required=True, help="outside the repository")
    p.add_argument("--script-commit", help="real runs: 40-hex HEAD of the checkout")
    p.add_argument("--script-commit-dl-id", help="real runs: DL-<n> logging that commit")
    p.add_argument("--synthetic-inputs", action="store_true", help="synthetic trees: no binding, synthetic_ prefix")
    p.add_argument("--generated-utc", help="synthetic runs only: pin the output stamp")
    p.add_argument("--expect-train", type=int, help="synthetic runs only: TRAIN image count (real: 5,367)")
    p.add_argument("--expect-val", type=int, help="synthetic runs only: VAL image count (real: 846)")
    return p


# --------------------------------------------------------------------------------------------------
# rules (module-level so the smoke can replace one at a time)
# --------------------------------------------------------------------------------------------------
def student_rule(name: str) -> bool:
    """The evaluator's and training loader's image rule (case-insensitive .jpg/.jpeg)."""
    return os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES


def teacher_rule(name: str) -> bool:
    """The teacher dataset's rule: the exact, case-sensitive img_suffix '.jpg'."""
    return name.endswith(TEACHER_IMG_SUFFIX)


def scan_folder(folder: str) -> dict:
    """One os.scandir of one folder: sorted file names and the counts of directories and other entries."""
    if not os.path.isdir(folder):
        raise td.Refused(f"missing folder under --data-root ({td.name_digests([folder])})")
    files, dirs, other = [], [], 0
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file():
                files.append(e.name)
            elif e.is_dir():
                dirs.append(e.name)
            else:
                other += 1
    return {"files": sorted(files), "dirs": sorted(dirs), "n_other": other}


def read_header(path: str) -> tuple:
    """(W, H, format, warnings) from the header only: Image.open is lazy and nothing calls load()."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Image.open(path) as im:
            w, h = im.size
            fmt = im.format
    return w, h, fmt, [f"{c.category.__name__}: {str(c.message)[:160]}" for c in caught]


def long_short(w: int, h: int) -> tuple[int, int]:
    return max(w, h), min(w, h)


def above(long: int, short: int, num: int, den: int) -> bool:
    """AR = long / short strictly above num / den, in integers."""
    return den * long > num * short


def rows_count_ok(rows: list) -> bool:
    return len(rows) == len(EXPECTED_ITERATIONS)


def iterations_ok(rows: list) -> bool:
    return [r["iteration"] for r in rows] == EXPECTED_ITERATIONS


def val_images_ok(rows: list, want: int) -> bool:
    return all(r["val_images"] == want for r in rows)


def manifest_uniform(rows: list) -> bool:
    return len({r["val_manifest_sha256"] for r in rows}) == 1


def manifest_equal(census_sha: str, rows: list) -> bool:
    return manifest_uniform(rows) and rows[0]["val_manifest_sha256"] == census_sha


# --------------------------------------------------------------------------------------------------
# guards, flags, binding, documents
# --------------------------------------------------------------------------------------------------
def folder_paths(root: str) -> dict:
    return {s: os.path.join(root, *FOLDERS[s]) for s in SPLITS}


def guard_paths(args) -> None:
    named = {"--data-root": args.data_root,
             **{f"<data-root>/{'/'.join(FOLDERS[s])}": p for s, p in folder_paths(args.data_root).items()},
             "--out-dir": args.out_dir}
    if args.teacher_records is not None:
        named["--teacher-records"] = args.teacher_records
    td.refuse_test_names([str(v) for v in named.values()], "path arguments")
    for what, value in named.items():
        td.refuse_test_path(value, what)


def check_flags(args, synthetic: bool) -> None:
    td.check_common_flags(args, real=not synthetic)
    suffix = args.census == "teacher-suffix"
    if suffix and args.teacher_records is None:
        raise td.Refused("--census teacher-suffix requires --teacher-records")
    if not suffix and (args.teacher_records is not None or args.expect_records_sha256 is not None):
        raise td.Refused("--teacher-records and --expect-records-sha256 belong to --census teacher-suffix")
    if args.expect_records_sha256 is not None and not SHA256_RE.fullmatch(args.expect_records_sha256):
        raise td.Refused("--expect-records-sha256 must be 64 lowercase hex characters")
    for flag in ("expect_train", "expect_val"):
        v = getattr(args, flag)
        if v is not None and not synthetic:
            raise td.Refused(f"--{flag.replace('_', '-')} is refused in a real run: the counts are 5,367 and 846")
        if v is not None and v < 1:
            raise td.Refused(f"--{flag.replace('_', '-')} must be at least 1")


def commit_binding(args) -> dict:
    head = td.git_head()
    if head is None:
        raise td.Refused("git HEAD could not be read; a real run needs the checkout at its pin")
    if head != args.script_commit:
        raise td.Refused(f"HEAD {head} != --script-commit {args.script_commit}")
    missing = [p for p in CODE_FILES if not (REPO / p).is_file()]
    if missing:
        raise td.Refused(f"{len(missing)} code file(s) missing at HEAD: {missing}")
    tree = td.git_status(("src", "configs", "scripts"))
    if tree is None or tree.strip():
        n = "?" if tree is None else len(tree.splitlines())
        raise td.Refused(f"git status --porcelain=v1 -- src configs scripts lists {n} entr(ies); a real run "
                         "needs them clean at HEAD")
    return {"head": head, "script_commit": args.script_commit, "script_commit_dl_id": args.script_commit_dl_id}


def code_block(binding) -> dict:
    status = td.git_status(CODE_FILES)
    return {"commit": td.git_head(), "code_files_clean_at_commit": None if status is None else status == "",
            "code_files": {p: (td.file_sha256(REPO / p) if (REPO / p).is_file() else None) for p in CODE_FILES},
            "binding": binding}


def environment_block() -> dict:
    import numpy
    import PIL
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "pillow": PIL.__version__,
           "image_digest": (os.environ.get("PLANTSEG_IMAGE_DIGEST") or "").strip() or None}
    torch = sys.modules.get("torch")
    env["torch"] = getattr(torch, "__version__", None) if torch is not None else None
    return env


def header(args, synthetic: bool, generated: str, binding, schema: str, census: str, authority: str) -> dict:
    return {"schema": schema, "census": census, "lane": "S4", "script": SCRIPT, "authority": authority,
            "artifact_status": "smoke" if synthetic else "provisional", "synthetic_input_data": synthetic,
            "generated_utc": generated, "code": code_block(binding), "environment": environment_block()}


# --------------------------------------------------------------------------------------------------
# DL-49: the aspect-ratio census
# --------------------------------------------------------------------------------------------------
def summarize(rows: list) -> dict:
    if not rows:
        return {"n": 0}
    best = max(rows, key=lambda r: Fraction(r["long"], r["short"]))
    top = sorted(rows, key=lambda r: (-Fraction(r["long"], r["short"]), r["split"], r["file"]))[:10]
    out = {"n": len(rows), "max_ar": best["long"] / best["short"], "max_ar_exact": f"{best['long']}/{best['short']}",
           "max_ar_image": {k: best[k] for k in ("split", "stem", "file", "w", "h")}}
    for label, num, den in THRESHOLDS:
        out[f"count_above_{label.replace('.', '_')}"] = sum(1 for r in rows if above(r["long"], r["short"], num, den))
    out["top10"] = [{"split": r["split"], "stem": r["stem"], "w": r["w"], "h": r["h"], "ar": r["long"] / r["short"]}
                    for r in top]
    return out


def aspect_ratio(args, listings: dict, folders: dict, expected: dict, base: dict) -> tuple[dict, int]:
    rows, unreadable, warns, formats = [], [], Counter(), {}
    for s in SPLITS:
        imgs = [n for n in listings[s]["files"] if student_rule(n)]
        if len(imgs) != expected[s]:
            raise td.Refused(f"images/{s}: {len(imgs)} images under the evaluator's rule, expected {expected[s]}")
        stems = [os.path.splitext(n)[0] for n in imgs]
        if len(set(stems)) != len(stems):
            raise td.Refused(f"images/{s}: {len(stems) - len(set(stems))} image(s) share a stem with another")
        fmts = Counter()
        for name, stem in zip(imgs, stems):
            try:
                w, h, fmt, caught = read_header(os.path.join(folders[s], name))
            except Exception:  # noqa: BLE001 -- any unreadable header is counted, then a STOP
                unreadable.append(name)
                continue
            if isinstance(w, bool) or not isinstance(w, int) or not isinstance(h, int) or w < 1 or h < 1:
                unreadable.append(name)
                continue
            warns.update(caught)
            fmts[str(fmt)] += 1
            long, short = long_short(w, h)
            rows.append({"split": s, "stem": stem, "file": name, "w": w, "h": h, "long": long, "short": short})
        formats[s] = dict(sorted(fmts.items()))
    if unreadable:
        raise td.Stop(f"{len(unreadable)} image header(s) could not be read ({td.name_digests(unreadable)}): the "
                      "census is incomplete; nothing is written")
    dims = "".join(f"{r['split']}\t{r['stem']}\t{r['w']}\t{r['h']}\n" for r in rows).encode("utf-8")
    splits = {s: dict(summarize([r for r in rows if r["split"] == s]), formats=formats[s]) for s in SPLITS}
    doc = dict(base, **{
        "inputs": {"folders": ["images/train", "images/val"],
                   "listing": "one os.scandir per folder; no recursion; no other folder",
                   "image_rule": "os.path.splitext(name)[1].lower() in ('.jpg', '.jpeg') (scripts/hash_split_files.py "
                                 "IMAGE_SUFFIXES; src/eval/adapters.py list_split_stems)",
                   "expected": expected,
                   "entries": {s: {"files": len(listings[s]["files"]), "dirs": len(listings[s]["dirs"]),
                                   "other": listings[s]["n_other"]} for s in SPLITS}},
        "definition": {"aspect_ratio": "max(W, H) / min(W, H) of the stored header size (PIL Image.open(...).size; "
                                       "no pixel decoded); invariant under EXIF 90-degree rotations",
                       "above": "strictly above, in integers: 10 * long > 126 * short (12.6), 10 * long > 265 * short "
                                "(26.5)"},
        "thresholds": [{"value": 12.6, "source": "B3: every valid 32x32 cell is lost only above aspect ratio about "
                                                 "12.6 at the worst scale r = 0.75 (certain from about 24.8)"},
                       {"value": 26.5, "source": "B3: every valid 64x64 cell is lost only above about 26.5"}],
        "splits": splits, "combined": summarize(rows),
        "header_warnings": dict(sorted(warns.items())),
        "dims_list_sha256": hashlib.sha256(dims).hexdigest(),
        "dims_list_rule": "sha256 over 'split\\tstem\\tw\\th\\n' per image, TRAIN then VAL, name-sorted",
        "status": "written"})
    return doc, td.EXIT_OK


# --------------------------------------------------------------------------------------------------
# DL-55: the teacher suffix census
# --------------------------------------------------------------------------------------------------
def read_records(path: Path) -> list:
    def reject(c):
        raise td.Refused(f"{RECORDS_FILE}: non-finite JSON constant {c!r}")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            obj = json.loads(line, parse_constant=reject)
        except json.JSONDecodeError as e:
            raise td.Refused(f"{RECORDS_FILE}: line {i} is not JSON ({e})") from e
        if not isinstance(obj, dict):
            raise td.Refused(f"{RECORDS_FILE}: line {i} is not a JSON object")
        for key, typ in (("iteration", int), ("val_images", int), ("val_manifest_sha256", str)):
            if key not in obj or isinstance(obj[key], bool) or not isinstance(obj[key], typ):
                raise td.Refused(f"{RECORDS_FILE}: line {i} lacks an {typ.__name__} {key!r}")
        rows.append(obj)
    if not rows:
        raise td.Refused(f"{RECORDS_FILE} holds no row")
    return rows


def teacher_suffix(args, listings: dict, folders: dict, expected: dict, base: dict) -> tuple[dict, int]:
    from src.eval.artifacts import hash_split_manifest
    from src.eval.evaluate import ManifestEntry
    per, stems_t = {}, {}
    for s in SPLITS:
        files = listings[s]["files"]
        teacher = [n for n in files if teacher_rule(n)]
        student = [n for n in files if student_rule(n)]
        t_stems = [n[:-len(TEACHER_IMG_SUFFIX)] for n in teacher]
        s_stems = [os.path.splitext(n)[0] for n in student]
        only_t, only_s = sorted(set(t_stems) - set(s_stems)), sorted(set(s_stems) - set(t_stems))
        stems_t[s] = t_stems
        per[s] = {"n_files": len(files), "n_dirs": len(listings[s]["dirs"]), "n_other": listings[s]["n_other"],
                  "suffix_counts": dict(sorted(Counter(os.path.splitext(n)[1] for n in files).items())),
                  "teacher_rule_count": len(teacher), "student_rule_count": len(student),
                  "student_stems_unique": len(set(s_stems)) == len(s_stems),
                  "stem_sets_equal": set(t_stems) == set(s_stems),
                  "n_only_teacher": len(only_t), "only_teacher": only_t[:20],
                  "n_only_student": len(only_s), "only_student": only_s[:20]}
    rec_path = Path(args.teacher_records)
    if rec_path.name != RECORDS_FILE:
        raise td.Refused(f"--teacher-records must name {RECORDS_FILE} (src/training/teacher_components.py RECORDS_FILE)")
    if not rec_path.is_file():
        raise td.Refused(f"--teacher-records: the file does not exist ({td.name_digests([str(rec_path)])})")
    rec_sha = td.file_sha256(rec_path)
    if args.expect_records_sha256 is not None and rec_sha != args.expect_records_sha256:
        raise td.Refused(f"the records file's sha256 {rec_sha} != --expect-records-sha256 {args.expect_records_sha256}")
    rows = read_records(rec_path)
    census_sha = hash_split_manifest([ManifestEntry(i, st, st) for i, st in enumerate(stems_t["val"])])
    expectation = {}
    for s in SPLITS:
        p = per[s]
        expectation[f"{s}_teacher_rule_count_is_{expected[s]}"] = p["teacher_rule_count"] == expected[s]
        expectation[f"{s}_student_rule_count_is_{expected[s]}"] = p["student_rule_count"] == expected[s]
        expectation[f"{s}_stem_sets_equal"] = p["stem_sets_equal"] and p["student_stems_unique"]
        expectation[f"{s}_no_subdirectory"] = p["n_dirs"] == 0
    expectation["records_rows_10"] = rows_count_ok(rows)
    expectation["records_iterations_4000_to_40000"] = iterations_ok(rows)
    expectation[f"records_val_images_{expected['val']}_in_every_row"] = val_images_ok(rows, expected["val"])
    expectation["records_val_manifest_uniform"] = manifest_uniform(rows)
    expectation["records_val_manifest_equals_census"] = manifest_equal(census_sha, rows)
    met = all(expectation.values())
    manifests = sorted({r["val_manifest_sha256"] for r in rows})
    doc = dict(base, **{
        "inputs": {"folders": ["images/train", "images/val"],
                   "listing": "one os.scandir per folder; no recursion; no other folder",
                   "teacher_rule": "name.endswith('.jpg') (configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-"
                                   "512x512.py:207 img_suffix; exact and case-sensitive)",
                   "student_rule": "os.path.splitext(name)[1].lower() in ('.jpg', '.jpeg')",
                   "expected": expected},
        "splits": per,
        "records": {"file": RECORDS_FILE, "records_sha256": rec_sha, "expected_sha256": args.expect_records_sha256,
                    "expected_source": DL68 if args.expect_records_sha256 is not None else None,
                    "expected_sha256_matches": (rec_sha == args.expect_records_sha256
                                                if args.expect_records_sha256 is not None else None),
                    "rows": len(rows), "iterations": [r["iteration"] for r in rows],
                    "expected_iterations": EXPECTED_ITERATIONS, "val_images": [r["val_images"] for r in rows],
                    "val_manifest_sha256": manifests,
                    "mIoU_full": [r.get("mIoU_full") for r in rows],
                    "mIoU_full_note": "recorded, not gated: compare with DL-68's ten values",
                    "teacher_of_record_checkpoint_in_rows": any(r.get("checkpoint_sha256") == td.RECORD_SHA256
                                                                 for r in rows)},
        "val_manifest": {"census": census_sha, "records": manifests[0] if len(manifests) == 1 else None,
                         "equal": expectation["records_val_manifest_equals_census"],
                         "rule": "hash_split_manifest over ManifestEntry(i, stem, stem) for the name-sorted exact-suffix "
                                 "VAL files: the order the teacher hook hashes (src/training/teacher_components.py:214-224; "
                                 "'BaseSegDataset sorts by img_path', teacher config :249); mmseg's own listing is not "
                                 "in the repository"},
        "expectation": expectation, "expectation_met": met, "status": "written"})
    return doc, (td.EXIT_OK if met else td.EXIT_STOP)


# --------------------------------------------------------------------------------------------------
def run(args) -> int:
    return td.run_with_exit_codes(_run, args)


def _run(args) -> int:
    start_utc = td.utc_now()
    synthetic = bool(args.synthetic_inputs)
    guard_paths(args)                                                         # no filesystem call before it
    check_flags(args, synthetic)
    out = td.require_outside_repo(args.out_dir, "--out-dir")
    aspect = args.census == "aspect-ratio"
    name = (SYNTHETIC_PREFIX if synthetic else "") + (AR_NAME if aspect else SUFFIX_NAME)
    if os.path.lexists(out / name):
        raise td.Refused(f"--out-dir already holds {name}; a census is written once")
    binding = None if synthetic else commit_binding(args)
    expected = {"train": args.expect_train if args.expect_train is not None else td.TRAIN_ROWS,
                "val": args.expect_val if args.expect_val is not None else td.VAL_ROWS}
    folders = folder_paths(args.data_root)
    listings = {s: scan_folder(folders[s]) for s in SPLITS}
    for s in SPLITS:
        td.refuse_test_names(listings[s]["files"] + listings[s]["dirs"], f"images/{s} names")
    generated = args.generated_utc or start_utc
    if aspect:
        base = header(args, synthetic, generated, binding, "plantseg-aspect-ratio-census/1.0.0", "aspect-ratio",
                      "DL-49 (docs/DECISION_LOG.md:108); docs/IMPLEMENTATION_CONTRACT.md B3 'Zero-valid samples' (:370)")
        doc, code = aspect_ratio(args, listings, folders, expected, base)
    else:
        base = header(args, synthetic, generated, binding, "plantseg-teacher-suffix-census/1.0.0", "teacher-suffix",
                      "DL-55 (docs/DECISION_LOG.md:114; A1 N3)")
        doc, code = teacher_suffix(args, listings, folders, expected, base)
    data = td.json_bytes(doc)
    td.write_files_exclusive([(out / name, data)])
    print(f"written: {name} sha256 {hashlib.sha256(data).hexdigest()}")
    if aspect:
        c = doc["combined"]
        print(f"TRAIN {doc['splits']['train']['n']} + VAL {doc['splits']['val']['n']}: max AR {c['max_ar']:.6g} "
              f"({c['max_ar_image']['split']}/{c['max_ar_image']['stem']}, {c['max_ar_image']['w']}x"
              f"{c['max_ar_image']['h']}); above 12.6: {c['count_above_12_6']}; above 26.5: {c['count_above_26_5']}")
    else:
        for s in SPLITS:
            p = doc["splits"][s]
            print(f"images/{s}: {p['suffix_counts']}; teacher rule {p['teacher_rule_count']}, student rule "
                  f"{p['student_rule_count']}, stem sets equal {p['stem_sets_equal']}, subdirectories {p['n_dirs']}")
        r = doc["records"]
        print(f"records: {r['rows']} rows, val_images {sorted(set(r['val_images']))}, manifest equal "
              f"{doc['val_manifest']['equal']}; expectation met: {doc['expectation_met']}")
        if not doc["expectation_met"]:
            bad = [k for k, v in doc["expectation"].items() if not v]
            print(f"STOP: the DL-55 expectation is not met: {bad} (the census is written)", file=sys.stderr)
    return code


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
