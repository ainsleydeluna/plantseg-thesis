#!/usr/bin/env python3
"""L-AM17-STRATA: the TRAIN-only rare-class strata file (AM-17 item 8; docs/lane_specs/part1.md lane 5).

    python -B scripts/build_train_strata.py [--data-root ROOT] [--out-dir reports/strata]
    python -B scripts/build_train_strata.py --check reports/strata/train_strata_v1.json

BUILD. Reads the TRAIN annotation PNGs at their original resolution through the TRAIN split list only:
the name-sorted images/train listing paired with annotations/train/<stem>.png, exactly as
`PlantSegDataset` pairs them, with its missing-mask guard and the locked 5,367-image count (configs/
data.py SPLIT_SIZES). No image is decoded, VAL and TEST are never listed or opened, and any data path
containing "test" (case-insensitive, as given or after resolving symlinks) is refused before a single
mask is read (exit 2). For each disease class 1-115:

    pixels   TRAIN pixel count                    images   TRAIN images with at least one pixel
    share    pixels / (sum of pixels over 1-115)  rank     1 = largest share; ties -> smaller class id
    tercile  ranks 1-38 T1, 39-76 T2, 77-115 T3   rare     images < 20
    zero_train_pixels  a class with no TRAIN pixel: allowed, ranked last, flagged (lane 5 (f))

Background (class 0) is counted but takes no share or rank. A mask value outside 0..115 other than
the ignore index 255 is refused by name. Output: <out-dir>/train_strata_v1.json and .csv, sorted by
class id and deterministic: two runs over the same masks at the same commit write identical bytes when
--generated-utc is pinned (the only volatile field). An existing output is never overwritten.

CHECK (lane 5 d2, local, after the build). PASS iff the file is the registered build (5,367 images,
116 classes, rare threshold 20, status provisional) and internally consistent (shares, ranks, terciles,
rare flags and the CSV all re-derive from the counts), all 115 classes are present, the shares sum to
1 within 1e-12, the terciles hold 38/38/39 classes, Spearman's rho between share and 1/weight of
reports/e1_class_weights.json is at least 0.95 (lane 5 d2: an INFERRED sanity bound; below it, inspect
and explain before committing), and script_commit names a commit at which this script was clean.
Recorded, not gated: whether the per-class pixel counts equal the class-weight file's `pixel_counts`
(that file records its basis as the raw TRAIN PNGs at native resolution, i.e. these same counts). The
check prints the file's sha256 for the decision-log row.

--num-classes, --rare-threshold and --expected-images exist for synthetic fixtures only: any value
other than the registered one marks the output artifact_status "smoke", and CHECK refuses it.

Exit codes: 0 written / PASS; 1 CHECK FAIL (STOP: inspect before committing); 2 refused or unreadable
input (a path containing "test", a missing or extra TRAIN file, a bad mask, an existing output, or a
check file that cannot be read). Nothing here trains, downloads or uses a GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.data import DATA, SPLIT_SIZES  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2

SCHEMA = "plantseg-train-strata/1.0.0"
LANE = "L-AM17-STRATA"
OUT_STEM = "train_strata_v1"
DEFAULT_OUT_DIR = REPO / "reports" / "strata"
SCRIPT_REL = "scripts/build_train_strata.py"
CLASS_WEIGHTS = REPO / "reports" / "e1_class_weights.json"

NUM_CLASSES = int(DATA["num_classes"])                # 116: background 0 + diseases 1..115
BACKGROUND = int(DATA["background_index"])            # 0
IGNORE_INDEX = int(DATA["ignore_index"])              # 255 (absent from the raw masks)
TRAIN_IMAGES = int(SPLIT_SIZES["train"])              # 5,367 [counted]
RARE_THRESHOLD = 20                                   # AM-17 item 8: fewer than 20 TRAIN images
SPEARMAN_MIN = 0.95                                   # lane 5 d2 (INFERRED sanity bound)
SHARE_SUM_TOL = 1e-12                                 # lane 5 d2
IMAGE_SUFFIXES = (".jpg", ".jpeg")                    # as PlantSegDataset (case-insensitive)
MASK_SUFFIX = ".png"
FORBIDDEN = "test"
TERCILES = ("T1", "T2", "T3")
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
CSV_COLUMNS = ("class_id", "pixels", "share", "images", "rank", "tercile", "rare", "zero_train_pixels")
SPLIT_LIST_DEFINITION = ("sha256 of the name-sorted TRAIN image stems (images/train, .jpg/.jpeg), one "
                         "per line, UTF-8, LF, trailing LF; each stem's mask is "
                         "annotations/train/<stem>.png")


class StrataError(RuntimeError):
    """Refused or unreadable input. Nothing is written when it is raised."""


class TestPathRefused(StrataError):
    """A data path containing "test": TEST stays behind the M11 isolation path."""


# --------------------------------------------------------------------------------------------------
# the TRAIN split list
# --------------------------------------------------------------------------------------------------
def refuse_test_path(path, what: str) -> None:
    """Refuse a path that names "test" as given or once symlinks are resolved. Opens nothing."""
    for form in (str(path), os.path.realpath(path)):
        if FORBIDDEN in form.lower():
            raise TestPathRefused(f"refusing {what} {form!r}: it contains {FORBIDDEN!r}; only the TRAIN "
                                  "split list is read and TEST is never listed or opened")


def train_split_list(data_root, expected_images: int = TRAIN_IMAGES) -> list[tuple[str, Path]]:
    """[(stem, mask path)] in name order: images/train paired with annotations/train/<stem>.png.

    Lists exactly the two TRAIN directories (names only; no image is opened) and checks every path
    before any mask is read.
    """
    root = Path(data_root)
    refuse_test_path(root, "data root")
    img_dir, mask_dir = root / "images" / "train", root / "annotations" / "train"
    for d in (img_dir, mask_dir):
        refuse_test_path(d, "TRAIN directory")
        if not d.is_dir():
            raise StrataError(f"missing TRAIN directory: {d}")
    images = sorted((p for p in img_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name)
    pairs, missing = [], []
    for img in images:
        refuse_test_path(img, "TRAIN image path")
        mask = mask_dir / f"{img.stem}{MASK_SUFFIX}"
        refuse_test_path(mask, "TRAIN mask path")
        if not mask.is_file():
            missing.append(img.name)
        pairs.append((img.stem, mask))
    if missing:
        raise StrataError(f"{len(missing)} TRAIN image(s) have no mask under {mask_dir} (expected "
                          f"<stem>{MASK_SUFFIX}), e.g. {missing[:5]}")
    stems = [s for s, _ in pairs]
    if len(set(stems)) != len(stems):
        raise StrataError("two TRAIN images share a stem (for example .jpg and .jpeg)")
    masks = {p.stem for p in mask_dir.iterdir() if p.is_file() and p.suffix == MASK_SUFFIX}
    extra = sorted(masks - set(stems))
    if extra:
        raise StrataError(f"annotations/train holds {len(extra)} mask(s) without a TRAIN image, e.g. "
                          f"{extra[:5]}")
    if len(pairs) != expected_images:
        raise StrataError(f"TRAIN split list has {len(pairs)} images, expected {expected_images} "
                          "(configs/data.py SPLIT_SIZES['train'])")
    return pairs


def split_list_sha256(stems) -> str:
    return hashlib.sha256("".join(f"{s}\n" for s in stems).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------------
# counting and strata
# --------------------------------------------------------------------------------------------------
def count_masks(mask_paths, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
    """Per-class pixel and image counts (int64) over the masks, plus the ignore-pixel count."""
    import numpy as np
    from PIL import Image

    pixels = np.zeros(num_classes, dtype=np.int64)
    images = np.zeros(num_classes, dtype=np.int64)
    ignored = 0
    for path in mask_paths:
        name = Path(path).name
        try:
            with Image.open(path) as im:
                a = np.asarray(im)                   # raw class indices; never resized or remapped
        except (OSError, ValueError) as e:           # PIL.UnidentifiedImageError is an OSError
            raise StrataError(f"{name}: unreadable mask ({type(e).__name__}: {e})") from e
        if a.ndim != 2 or a.size == 0:
            raise StrataError(f"{name}: a mask must be a non-empty single-channel array, got {a.shape}")
        if a.dtype.kind not in "ui":
            raise StrataError(f"{name}: mask dtype {a.dtype} is not an integer type")
        if a.dtype != np.uint8 and (int(a.min()) < 0 or int(a.max()) > 255):
            raise StrataError(f"{name}: mask values outside 0..255")
        b = np.bincount(a.reshape(-1), minlength=256)
        bad = [int(v) for v in np.flatnonzero(b) if v >= num_classes and v != ignore_index]
        if bad:
            raise StrataError(f"{name}: label value(s) {bad[:5]} outside 0..{num_classes - 1} and not the "
                              f"ignore index {ignore_index}")
        pixels += b[:num_classes]
        images += b[:num_classes] > 0
        ignored += int(b[ignore_index]) if ignore_index < b.size else 0
    return pixels, images, ignored


def tercile_sizes(n: int) -> list[int]:
    """AM-17 item 8's 38/38/39 split of 115 classes; in general floor(n/3), floor(n/3), the rest."""
    return [n // 3, n // 3, n - 2 * (n // 3)]


def build_rows(pixels, images, *, background_index: int = BACKGROUND,
               rare_threshold: int = RARE_THRESHOLD) -> list[dict]:
    """The per-class strata rows for every class except the background, ascending class id."""
    n_classes = len(pixels)
    diseases = [c for c in range(n_classes) if c != background_index]
    if len(diseases) < 3:
        raise StrataError(f"at least three disease classes are needed for terciles, got {len(diseases)}")
    total = sum(int(pixels[c]) for c in diseases)
    if total <= 0:
        raise StrataError("the TRAIN split list holds no disease pixel")
    order = sorted(diseases, key=lambda c: (-int(pixels[c]), c))      # share descending, then id
    rank = {c: i + 1 for i, c in enumerate(order)}
    s1, s2, _ = tercile_sizes(len(diseases))
    rows = []
    for c in diseases:
        r = rank[c]
        px, im = int(pixels[c]), int(images[c])
        rows.append({"class_id": c, "pixels": px, "share": px / total, "images": im, "rank": r,
                     "tercile": TERCILES[0] if r <= s1 else TERCILES[1] if r <= s1 + s2 else TERCILES[2],
                     "rare": im < rare_threshold, "zero_train_pixels": px == 0})
    return rows


def csv_bytes(rows) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow([r["class_id"], r["pixels"], repr(r["share"]), r["images"], r["rank"], r["tercile"],
                    "true" if r["rare"] else "false", "true" if r["zero_train_pixels"] else "false"])
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(*args) -> str | None:
    """git output from the repository root, or None. GIT_OPTIONAL_LOCKS=0: never writes the index."""
    try:
        p = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True, timeout=60,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def script_provenance() -> dict:
    """HEAD and whether this script is unmodified at HEAD (an explicit-path status, never repo-wide)."""
    head = (_git("rev-parse", "HEAD") or "").strip()
    commit = head if _HEX40.match(head) else None
    status = _git("status", "--porcelain=v1", "--", SCRIPT_REL) if commit else None
    return {"script_commit": commit,
            "script_clean_at_commit": None if status is None else status == "",
            "script_sha256": _sha256_bytes(Path(__file__).read_bytes())}


# --------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------
def build_document(data_root, *, num_classes: int = NUM_CLASSES, rare_threshold: int = RARE_THRESHOLD,
                   expected_images: int = TRAIN_IMAGES,
                   generated_utc: str | None = None) -> tuple[dict, bytes]:
    """(the strata JSON document, the CSV bytes). Reads the TRAIN masks; writes nothing."""
    import numpy as np
    import PIL

    pairs = train_split_list(data_root, expected_images)
    pixels, images, ignored = count_masks([m for _, m in pairs], num_classes, IGNORE_INDEX)
    rows = build_rows(pixels, images, background_index=BACKGROUND, rare_threshold=rare_threshold)
    csv_b = csv_bytes(rows)
    registered = (num_classes, rare_threshold, expected_images) == (NUM_CLASSES, RARE_THRESHOLD,
                                                                    TRAIN_IMAGES)
    n_dis = len(rows)
    s1, s2, s3 = tercile_sizes(n_dis)
    root = Path(data_root)
    doc = {
        "schema": SCHEMA,
        "lane": LANE,
        "authority": "AM-17 item 8 (DRAFT, adviser approval pending); docs/lane_specs/part1.md lane 5",
        "artifact_status": "provisional" if registered else "smoke",
        "split": "train",
        "split_list_sha256": split_list_sha256([s for s, _ in pairs]),
        "split_list_definition": SPLIT_LIST_DEFINITION,
        "data_root_note": (f"TRAIN only, read at {root.as_posix()}: images/train (names, never decoded) "
                           "and annotations/train (masks at original resolution); VAL and TEST are "
                           "never listed or opened, and any path containing 'test' is refused"),
        "n_images": len(pairs),
        "num_classes": num_classes,
        "background_index": BACKGROUND,
        "ignore_index": IGNORE_INDEX,
        "rare_threshold": rare_threshold,
        "rank_rule": "share descending; ties broken by the smaller class id",
        "tercile_rule": (f"ranks 1-{s1} T1, {s1 + 1}-{s1 + s2} T2, {s1 + s2 + 1}-{n_dis} T3 "
                         "(floor(n/3), floor(n/3), the rest)"),
        "rare_rule": f"images < {rare_threshold}",
        "share_rule": "pixels_c / sum of pixels over the disease classes (background excluded)",
        "per_class": rows,
        "tercile_sizes": [s1, s2, s3],
        "total_disease_pixels": int(sum(r["pixels"] for r in rows)),
        "background": {"class_id": BACKGROUND, "pixels": int(pixels[BACKGROUND]),
                       "images": int(images[BACKGROUND])},
        "ignore_pixels": int(ignored),
        "zero_train_pixel_classes": [r["class_id"] for r in rows if r["zero_train_pixels"]],
        "csv_sha256": _sha256_bytes(csv_b),
        **script_provenance(),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pillow": PIL.__version__},
        "generated_utc": generated_utc or datetime.now(timezone.utc).strftime(UTC_FORMAT),
    }
    return doc, csv_b


def json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_outputs(out_dir, doc: dict, csv_b: bytes) -> tuple[Path, Path]:
    out = Path(out_dir)
    targets = (out / f"{OUT_STEM}.json", out / f"{OUT_STEM}.csv")
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise StrataError(f"refusing to overwrite {existing}")
    out.mkdir(parents=True, exist_ok=True)
    with open(targets[1], "xb") as fh:
        fh.write(csv_b)
    try:
        with open(targets[0], "xb") as fh:
            fh.write(json_bytes(doc))
    except BaseException:
        targets[1].unlink()                          # never leave a CSV without its JSON
        raise
    return targets


# --------------------------------------------------------------------------------------------------
# check (lane 5 d2)
# --------------------------------------------------------------------------------------------------
def check_document(path) -> tuple[list[tuple[str, bool, str]], dict]:
    """[(check, passed, detail)] for a written strata file, plus the recorded (ungated) cross-checks."""
    from scipy.stats import spearmanr

    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(
        StrataError(f"non-finite JSON constant {c!r}")))
    out: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), str(detail)))

    rows = doc.get("per_class") or []
    ids = [r.get("class_id") for r in rows]
    diseases = [c for c in range(NUM_CLASSES) if c != BACKGROUND]
    check("C1 schema, TRAIN split, status provisional",
          doc.get("schema") == SCHEMA and doc.get("split") == "train"
          and doc.get("artifact_status") == "provisional",
          f"schema={doc.get('schema')!r} split={doc.get('split')!r} status={doc.get('artifact_status')!r}")
    check(f"C2 registered build: {TRAIN_IMAGES} images, {NUM_CLASSES} classes, rare threshold "
          f"{RARE_THRESHOLD}",
          (doc.get("n_images"), doc.get("num_classes"), doc.get("rare_threshold"),
           doc.get("background_index")) == (TRAIN_IMAGES, NUM_CLASSES, RARE_THRESHOLD, BACKGROUND),
          f"n_images={doc.get('n_images')} num_classes={doc.get('num_classes')} "
          f"rare_threshold={doc.get('rare_threshold')}")
    check("C3 one row per disease class 1-115, ascending", ids == diseases, f"{len(ids)} rows")
    if ids != diseases:
        return out, {}
    zero = [r["class_id"] for r in rows if r["pixels"] <= 0]
    check("C4 all 115 disease classes present in TRAIN (d2)", not zero, f"zero-pixel classes {zero}")
    total = sum(r["pixels"] for r in rows)
    share_sum = math.fsum(r["share"] for r in rows)
    check(f"C5 sum of shares = 1 within {SHARE_SUM_TOL:g} (d2)", abs(share_sum - 1.0) <= SHARE_SUM_TOL,
          f"sum={share_sum!r}")
    sizes = [sum(1 for r in rows if r["tercile"] == t) for t in TERCILES]
    check("C6 tercile sizes 38/38/39 (d2)", sizes == [38, 38, 39] == doc.get("tercile_sizes"),
          f"counted {sizes}, recorded {doc.get('tercile_sizes')}")
    rederived = build_rows([0] + [r["pixels"] for r in rows], [0] + [r["images"] for r in rows],
                           background_index=BACKGROUND, rare_threshold=RARE_THRESHOLD)
    check("C7 shares, ranks, terciles and rare flags re-derive from the counts exactly",
          rederived == rows and total == doc.get("total_disease_pixels"),
          "first difference: " + next((f"class {a['class_id']}" for a, b in zip(rederived, rows)
                                       if a != b), "none"))
    weights_doc = json.loads(CLASS_WEIGHTS.read_text(encoding="utf-8"))
    weights = weights_doc["weights"]
    rho = float(spearmanr([r["share"] for r in rows], [1.0 / weights[c] for c in diseases])[0])
    check(f"C8 Spearman rho(share, 1/weight) >= {SPEARMAN_MIN} (d2; reports/e1_class_weights.json)",
          rho >= SPEARMAN_MIN, f"rho={rho!r}")
    csv_path = path.with_suffix(".csv")
    csv_ok = csv_path.is_file() and _sha256_bytes(csv_path.read_bytes()) == doc.get("csv_sha256") \
        and csv_path.read_bytes() == csv_bytes(rows)
    check("C9 the CSV companion matches csv_sha256 and the JSON rows", csv_ok, str(csv_path))
    commit = doc.get("script_commit")
    check("C10 script_commit names a commit at which this script was clean",
          isinstance(commit, str) and bool(_HEX40.match(commit))
          and doc.get("script_clean_at_commit") is True,
          f"script_commit={commit!r} clean={doc.get('script_clean_at_commit')!r}")
    ref = weights_doc.get("pixel_counts") or []
    differ = [c for c in diseases if len(ref) != NUM_CLASSES or ref[c] != rows[c - 1]["pixels"]]
    recorded = {"spearman_rho": rho,
                "pixel_counts_equal_class_weight_file": not differ,
                "pixel_count_differences": [{"class_id": c, "strata": rows[c - 1]["pixels"],
                                             "class_weights": ref[c] if len(ref) == NUM_CLASSES else None}
                                            for c in differ[:10]],
                "background_pixels_equal": bool(len(ref) == NUM_CLASSES
                                                and ref[BACKGROUND] == doc["background"]["pixels"]),
                "train_mask_count_equal": weights_doc.get("train_mask_count") == doc.get("n_images"),
                "file_sha256": _sha256_bytes(raw)}
    return out, recorded


def run_check(path) -> int:
    try:
        results, recorded = check_document(path)
    except Exception as e:                            # noqa: BLE001 -- unreadable, never a verdict
        print(f"RESULT: ERROR -- {path} cannot be checked: {type(e).__name__}: {e}")
        return EXIT_REFUSED
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")
    if recorded:
        print("  recorded (not gated):")
        print(f"    per-class pixel counts equal reports/e1_class_weights.json pixel_counts: "
              f"{recorded['pixel_counts_equal_class_weight_file']} "
              f"{recorded['pixel_count_differences'] or ''}".rstrip())
        print(f"    background pixels equal: {recorded['background_pixels_equal']}; TRAIN mask count "
              f"equal: {recorded['train_mask_count_equal']}")
        print(f"    sha256 {recorded['file_sha256']}  {Path(path).as_posix()}")
    good = sum(1 for _, ok, _ in results if ok)
    allok = good == len(results) and len(results) == 10
    print(f"SUMMARY  {good}/{len(results)} checks passed")
    print(f"RESULT: {'STRATA d2 PASS' if allok else 'STRATA d2 FAIL -- inspect before committing'} "
          f"({good}/{len(results)})")
    return EXIT_OK if allok else EXIT_FAIL


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="L-AM17-STRATA: TRAIN-only rare-class strata (AM-17 item 8).")
    p.add_argument("--data-root", default=DATA["root"],
                   help="dataset root holding images/train and annotations/train (default: "
                        "PLANTSEG_DATA_ROOT via configs/data.py)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--generated-utc", default=None,
                   help="pin generated_utc (for example 2026-10-01T00:00:00Z) so a rebuild is "
                        "byte-identical")
    p.add_argument("--check", type=Path, default=None, metavar="STRATA_JSON",
                   help="run the lane 5 d2 checks on a written strata file instead of building")
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="synthetic fixtures only")
    p.add_argument("--rare-threshold", type=int, default=RARE_THRESHOLD, help="synthetic fixtures only")
    p.add_argument("--expected-images", type=int, default=TRAIN_IMAGES, help="synthetic fixtures only")
    args = p.parse_args(argv)
    if args.check is not None:
        return run_check(args.check)
    if args.generated_utc is not None and not _UTC.match(args.generated_utc):
        print(f"error: --generated-utc must look like 2026-10-01T00:00:00Z, got {args.generated_utc!r}",
              file=sys.stderr)
        return EXIT_REFUSED
    try:
        doc, csv_b = build_document(args.data_root, num_classes=args.num_classes,
                                    rare_threshold=args.rare_threshold,
                                    expected_images=args.expected_images,
                                    generated_utc=args.generated_utc)
        json_path, csv_path = write_outputs(args.out_dir, doc, csv_b)
    except StrataError as e:
        print(f"RESULT: REFUSED -- {e}")
        return EXIT_REFUSED
    except Exception as e:                            # noqa: BLE001 -- an error is never a verdict
        print(f"RESULT: ERROR -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    print(f"{doc['n_images']} TRAIN masks, {len(doc['per_class'])} disease classes, tercile sizes "
          f"{doc['tercile_sizes']}, rare {sum(r['rare'] for r in doc['per_class'])}, zero-pixel "
          f"{doc['zero_train_pixel_classes']}, split list {doc['split_list_sha256'][:12]}...")
    print(f"  -> {json_path} (sha256 {_sha256_bytes(json_path.read_bytes())})")
    print(f"  -> {csv_path} (sha256 {doc['csv_sha256']})")
    print(f"RESULT: STRATA WRITTEN ({doc['artifact_status']})")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
