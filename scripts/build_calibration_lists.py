#!/usr/bin/env python3
"""L-AM10 / AM-16 item 5: the four 128-image TRAIN calibration lists (docs/lane_specs/part2.md lane 8).

    python -B scripts/build_calibration_lists.py --data-root ROOT --out-dir DIR [--generated-utc T]
    python -B scripts/build_calibration_lists.py --check DIR [--data-root ROOT] [--strata STRATA_JSON]

BUILD. The TRAIN split list is derived exactly as scripts/build_train_strata.py (L-AM17-STRATA) derives
it: the name-sorted images/train listing (.jpg/.jpeg, case-insensitive), each image paired with
annotations/train/<stem>.png as `PlantSegDataset` pairs them, with the missing-mask, shared-stem,
extra-mask and 5,367-image guards (configs/data.py SPLIT_SIZES), and `split_list_sha256` = sha256 of
the stems in that order, one per line, UTF-8, LF, trailing LF: the value the strata file records, so
both provably read the same TRAIN listing. Only names are listed; no image or mask is opened. Any data
path containing "test" (as scripts/build_train_strata.py) or with a component naming VAL (val, valid,
validation), as given or after resolving symlinks, is refused before anything is listed (exit 2).

Each list is `build_calibration_index(train_ids, seed=S)` from src/quant/calibration.py, unchanged:
random.Random(S).sample(sorted(train_ids), 128). Seed 42 is the AM-10 list of record, shared by E4 and
E7 (and it passes that module's E4/E7 validator); seeds 43, 44 and 45 are the AM-16 item 5
calibration-sensitivity subsets. Output: DIR/ptq_calibration_seed{42,43,44,45}.json, each with the ids
in draw order (the calibration order), checksum_sha256 (the id checksum the E4/E7 runners pin) and the
split-list hash. All four are serialized before any file is created, a failed write leaves none
behind, and an existing file is never overwritten. With --generated-utc pinned, two builds at the same
commit write identical bytes.

CHECK (lane 8 d1). PASS iff the four files parse and validate (128 unique ids, registered seed and role,
checksum, shared by E4 and E7), share one split-list hash, differ pairwise, the seed-42 list passes the
committed E4/E7 validator and the sensitivity lists are refused by it; with --data-root, the TRAIN
listing re-derives the same split-list hash and every list equals a fresh draw by the procedure (the
seed-42 file IS the AM-10 draw); with --strata, the split-list hash equals the strata file's. Prints
each file's sha256 and the seed-42 checksum_sha256 for the decision-log row. Disjointness from VAL is
by construction (TRAIN ids; zero identifier overlap across the partitions, configs/data.py); its
explicit check, which lists VAL file names, is `scripts/ptq_val_checks.py lists` (local).

--expected-images exists for synthetic fixtures only: another value marks the lists artifact_status
"smoke". Exit codes: 0 written / PASS; 1 CHECK FAIL (STOP); 2 refused or unreadable input. Nothing here
trains, downloads, uses a GPU or reads VAL or TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.data import SPLIT_SIZES  # noqa: E402
from src.quant.calibration import (CALIBRATION_COUNT, CalibrationIndexError,  # noqa: E402
                                   build_calibration_index)
from src.quant.ptq import (CALIBRATION_ORDER, CHECKSUM_DEFINITION, LIST_FILENAME,  # noqa: E402
                           LIST_PROCEDURE, LIST_SCHEMA, LIST_SEEDS, LIST_SHARED_BY,
                           LIST_STATUS_REGISTERED, PTQRefused, SEED_OF_RECORD, SPLIT_LIST_DEFINITION,
                           code_provenance, environment, list_role, refuse_eval_split_dir,
                           refuse_eval_split_file, verify_calibration_list, write_exclusive)
from src.quant.stages import require_shared_calibration_index  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2
TRAIN_IMAGES = int(SPLIT_SIZES["train"])              # 5,367 [counted]
IMAGE_SUFFIXES = (".jpg", ".jpeg")                    # as PlantSegDataset (case-insensitive)
MASK_SUFFIX = ".png"
LANE = "L-AM10 (docs/lane_specs/part2.md lane 8)"
AUTHORITY = ("AM-10 (seed 42: the list of record, shared by E4 and E7); AM-16 item 5 (seeds 43-45: "
             "descriptive calibration-sensitivity subsets); DL-10")
FILENAME = LIST_FILENAME                              # ptq_calibration_seed{seed}.json
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
PROVENANCE_PATHS = ("scripts/build_calibration_lists.py", "src/quant/calibration.py", "src/quant/ptq.py",
                    "configs/quant.py", "configs/data.py")


class ListError(RuntimeError):
    """Refused or unreadable input. Nothing is written when it is raised."""


# --------------------------------------------------------------------------------------------------
# the TRAIN split list (scripts/build_train_strata.py, plus the VAL refusal)
# --------------------------------------------------------------------------------------------------
def train_split_list(data_root, expected_images: int = TRAIN_IMAGES) -> list[str]:
    """TRAIN stems in name order: images/train paired with annotations/train/<stem>.png.

    Lists exactly the two TRAIN directories (names only; nothing is opened) and checks every path
    before the listing is returned.
    """
    root = Path(data_root)
    refuse_eval_split_dir(root, "data root")
    img_dir, mask_dir = root / "images" / "train", root / "annotations" / "train"
    for d in (img_dir, mask_dir):
        refuse_eval_split_dir(d, "TRAIN directory")
        if not d.is_dir():
            raise ListError(f"missing TRAIN directory: {d}")
    images = sorted((p for p in img_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name)
    stems, missing = [], []
    for img in images:
        refuse_eval_split_file(img, "TRAIN image path")
        mask = mask_dir / f"{img.stem}{MASK_SUFFIX}"
        refuse_eval_split_file(mask, "TRAIN mask path")
        if not mask.is_file():
            missing.append(img.name)
        stems.append(img.stem)
    if missing:
        raise ListError(f"{len(missing)} TRAIN image(s) have no mask under {mask_dir} (expected "
                        f"<stem>{MASK_SUFFIX}), e.g. {missing[:5]}")
    if len(set(stems)) != len(stems):
        raise ListError("two TRAIN images share a stem (for example .jpg and .jpeg)")
    masks = {p.stem for p in mask_dir.iterdir() if p.is_file() and p.suffix == MASK_SUFFIX}
    extra = sorted(masks - set(stems))
    if extra:
        raise ListError(f"annotations/train holds {len(extra)} mask(s) without a TRAIN image, e.g. "
                        f"{extra[:5]}")
    if len(stems) != expected_images:
        raise ListError(f"TRAIN split list has {len(stems)} images, expected {expected_images} "
                        "(configs/data.py SPLIT_SIZES['train'])")
    return stems


def split_list_sha256(stems) -> str:
    """= scripts/build_train_strata.py split_list_sha256."""
    return hashlib.sha256("".join(f"{s}\n" for s in stems).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------
def build_documents(data_root, *, expected_images: int = TRAIN_IMAGES,
                    generated_utc: str | None = None) -> list[tuple[str, dict]]:
    """[(file name, list document)] for the four seeds. Lists the TRAIN names; writes nothing."""
    stems = train_split_list(data_root, expected_images)
    split_sha = split_list_sha256(stems)
    registered = expected_images == TRAIN_IMAGES
    prov = code_provenance(PROVENANCE_PATHS)
    env = environment()
    stamp = generated_utc or datetime.now(timezone.utc).strftime(UTC_FORMAT)
    docs = []
    for seed in LIST_SEEDS:
        index = build_calibration_index(stems, seed=seed)       # the AM-10 procedure, unchanged
        doc = {
            **index,
            "list_schema": LIST_SCHEMA,
            "lane": LANE,
            "authority": AUTHORITY,
            "role": list_role(seed),
            "shared_by": list(LIST_SHARED_BY),
            "procedure": LIST_PROCEDURE,
            "checksum_definition": CHECKSUM_DEFINITION,
            "calibration_order": CALIBRATION_ORDER,
            "split_list_sha256": split_sha,
            "split_list_definition": SPLIT_LIST_DEFINITION,
            "n_train_images": len(stems),
            "artifact_status": LIST_STATUS_REGISTERED if registered else "smoke",
            "data_root_note": ("TRAIN names only: images/train and annotations/train were listed, no "
                               "image or mask was opened; VAL and TEST were never listed"),
            "script_commit": prov["commit"],
            "script_clean_at_commit": prov["clean_at_commit"],
            "code_sha256": prov["files_sha256"],
            "environment": {k: env[k] for k in ("python", "torch", "numpy", "image_digest")},
            "generated_utc": stamp,
        }
        verify_calibration_list(doc)
        docs.append((FILENAME.format(seed=seed), doc))
    return docs


def json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_lists(out_dir, docs) -> list[Path]:
    out = Path(out_dir)
    files = {out / name: json_bytes(doc) for name, doc in docs}      # serialized before any file exists
    out.mkdir(parents=True, exist_ok=True)
    return write_exclusive(files)


# --------------------------------------------------------------------------------------------------
# check (lane 8 d1)
# --------------------------------------------------------------------------------------------------
def _read(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant {c!r}")))
    return doc, raw


def check_lists(list_dir, *, data_root=None, expected_images: int = TRAIN_IMAGES,
                strata=None) -> tuple[list[tuple[str, bool, str]], dict]:
    """[(check, passed, detail)] for the four written lists, plus the values for the decision log."""
    d = Path(list_dir)
    out: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), str(detail)))

    docs, shas, errors = {}, {}, []
    for seed in LIST_SEEDS:
        p = d / FILENAME.format(seed=seed)
        try:
            doc, raw = _read(p)
            docs[seed] = verify_calibration_list(doc)
            shas[seed] = hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, CalibrationIndexError) as e:
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
    check("L1 four lists present and valid (128 unique ids, registered seed and role, checksum, "
          "shared by E4 and E7)", not errors and len(docs) == len(LIST_SEEDS), "; ".join(errors)[:400])
    if errors:
        return out, {}
    one = lambda key: {docs[s].get(key) for s in LIST_SEEDS}                      # noqa: E731
    check("L2 one TRAIN listing: split_list_sha256, candidate pool and status shared by all four",
          len(one("split_list_sha256")) == len(one("candidate_pool_size")) == len(one("artifact_status")) == 1,
          f"split_list_sha256={sorted(one('split_list_sha256'))} pool={sorted(one('candidate_pool_size'))}")
    id_sets = {s: tuple(docs[s]["selected_ids"]) for s in LIST_SEEDS}
    check("L3 the four lists differ pairwise", len(set(id_sets.values())) == len(LIST_SEEDS))
    try:
        require_shared_calibration_index(d / FILENAME.format(seed=SEED_OF_RECORD))
        official_ok = True
    except CalibrationIndexError as e:
        official_ok = False
        errors.append(str(e))
    refused = []
    for s in LIST_SEEDS:
        if s == SEED_OF_RECORD:
            continue
        try:
            require_shared_calibration_index(d / FILENAME.format(seed=s))
        except CalibrationIndexError:
            refused.append(s)
    check("L4 the seed-42 list passes the committed E4/E7 validator; seeds 43-45 are refused by it",
          official_ok and refused == [s for s in LIST_SEEDS if s != SEED_OF_RECORD],
          f"official={official_ok} refused={refused}")
    if data_root is not None:
        try:
            stems = train_split_list(data_root, expected_images)
        except (ListError, PTQRefused) as e:
            check("L5 TRAIN listing re-derived", False, f"{type(e).__name__}: {e}")
        else:
            split_sha = split_list_sha256(stems)
            train = set(stems)
            fresh = {s: random.Random(s).sample(sorted(stems), CALIBRATION_COUNT) for s in LIST_SEEDS}
            check("L5 the TRAIN listing re-derives the recorded split_list_sha256",
                  one("split_list_sha256") == {split_sha}, split_sha)
            check("L6 every list equals a fresh draw by the procedure (the seed-42 file IS the AM-10 draw)",
                  all(list(id_sets[s]) == fresh[s] for s in LIST_SEEDS),
                  f"mismatching seeds {[s for s in LIST_SEEDS if list(id_sets[s]) != fresh[s]]}")
            check("L7 every id is a TRAIN stem",
                  all(set(id_sets[s]) <= train for s in LIST_SEEDS))
    if strata is not None:
        try:
            sdoc, _ = _read(Path(strata))
            strata_sha = sdoc.get("split_list_sha256")
        except (OSError, ValueError) as e:
            strata_sha = None
            errors.append(f"strata: {e}")
        check("L8 split_list_sha256 equals the strata file's (same TRAIN listing as L-AM17-STRATA)",
              strata_sha is not None and one("split_list_sha256") == {strata_sha}, f"strata={strata_sha}")
    record = {"files_sha256": {FILENAME.format(seed=s): shas[s] for s in LIST_SEEDS},
              "seed42_checksum_sha256": docs[SEED_OF_RECORD]["checksum_sha256"],
              "checksums_sha256": {s: docs[s]["checksum_sha256"] for s in LIST_SEEDS},
              "split_list_sha256": " / ".join(sorted(map(str, one("split_list_sha256")))),
              "artifact_status": " / ".join(sorted(map(str, one("artifact_status"))))}
    return out, record


def run_check(args) -> int:
    try:
        results, record = check_lists(args.check, data_root=args.data_root,
                                      expected_images=args.expected_images, strata=args.strata)
    except Exception as e:                            # noqa: BLE001 -- unreadable, never a verdict
        print(f"RESULT: ERROR -- {args.check} cannot be checked: {type(e).__name__}: {e}")
        return EXIT_REFUSED
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")
    if record:
        print("  for the decision log:")
        for name, sha in record["files_sha256"].items():
            print(f"    sha256 {sha}  {name}")
        print(f"    seed-42 checksum_sha256 {record['seed42_checksum_sha256']}  (the E4/E7 --expect-list-checksum)")
        print(f"    split_list_sha256 {record['split_list_sha256']}  status {record['artifact_status']}")
    good = sum(1 for _, ok, _ in results if ok)
    allok = good == len(results) and bool(results)
    print(f"SUMMARY  {good}/{len(results)} checks passed")
    print(f"RESULT: {'CALIBRATION LISTS d1 PASS' if allok else 'CALIBRATION LISTS d1 FAIL -- STOP'} "
          f"({good}/{len(results)})")
    return EXIT_OK if allok else EXIT_FAIL


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="L-AM10: the four 128-image TRAIN calibration lists.")
    p.add_argument("--data-root", default=None,
                   help="dataset root holding images/train and annotations/train")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="destination (the committed home is configs/calibration)")
    p.add_argument("--generated-utc", default=None,
                   help="pin generated_utc (for example 2026-10-01T00:00:00Z) so a rebuild is byte-identical")
    p.add_argument("--check", type=Path, default=None, metavar="LIST_DIR",
                   help="run the lane 8 d1 checks on the four written lists instead of building")
    p.add_argument("--strata", type=Path, default=None, help="--check only: train_strata_v1.json")
    p.add_argument("--expected-images", type=int, default=TRAIN_IMAGES, help="synthetic fixtures only")
    args = p.parse_args(argv)
    if args.check is not None:
        return run_check(args)
    if args.data_root is None or args.out_dir is None:
        print("error: a build needs --data-root and --out-dir", file=sys.stderr)
        return EXIT_REFUSED
    if args.generated_utc is not None and not _UTC.match(args.generated_utc):
        print(f"error: --generated-utc must look like 2026-10-01T00:00:00Z, got {args.generated_utc!r}",
              file=sys.stderr)
        return EXIT_REFUSED
    try:
        docs = build_documents(args.data_root, expected_images=args.expected_images,
                               generated_utc=args.generated_utc)
        paths = write_lists(args.out_dir, docs)
    except (ListError, PTQRefused, CalibrationIndexError) as e:
        print(f"RESULT: REFUSED -- {e}")
        return EXIT_REFUSED
    except Exception as e:                            # noqa: BLE001 -- an error is never a verdict
        print(f"RESULT: ERROR -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    first = docs[0][1]
    print(f"{first['n_train_images']} TRAIN images, split list {first['split_list_sha256'][:12]}..., "
          f"status {first['artifact_status']}")
    for path, (_, doc) in zip(paths, docs):
        print(f"  -> {path} seed {doc['seed']} ({doc['role']}) checksum_sha256 {doc['checksum_sha256']} "
              f"file sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"RESULT: CALIBRATION LISTS WRITTEN ({len(paths)} lists, {first['artifact_status']})")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
