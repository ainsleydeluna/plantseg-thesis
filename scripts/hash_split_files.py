#!/usr/bin/env python3
"""D4 hasher: SHA-256 and SHA-1 of the file bytes of every TRAIN and VAL image, and the VAL images
byte-identical to a TRAIN image (AM-18 item 1(d); lane L-TEACHER-DIAG).

It opens exactly four folders, by explicit path: <data-root>/images/train, images/val, annotations/train
and annotations/val (P37 mounts them read-only). It never lists the data root or any other folder; it is
not the M11 guard (whose existence probes name TEST paths). A path containing "test" (case-insensitive)
is refused by its string before any filesystem call, then again once resolved; a refusal prints the
number of offending entries and the sha256 of each name, never the name (P21). It requires 5,367 TRAIN
and 846 VAL images, each with its <stem>.png mask.

One read per image feeds both digests. Images are grouped by SHA-256. A duplicate group holds at least one
TRAIN and one VAL image; its VAL members are the duplicate VAL images. For each group, whether the decoded
annotation masks of its members are identical. VAL images that match only other VAL images are reported
and not removed. Only byte-identical files are detected; re-encoded copies are not. Issue #11's counts
(32 TRAIN-VAL groups, 15 of three or more images) are reported beside this run's; a difference is
reported, never a STOP.

Outputs, in --out (outside the repository, never reports/): train_val_image_hashes.tsv (split, stem,
file, bytes, sha256, sha1) and val_train_duplicates.json (P22: artifact_status, the VAL manifest hash of
the name-sorted VAL stems, the TRAIN split-list sha256, which must equal the strata file's, group sizes,
the VAL-only groups, the duplicate VAL ids and their sha256 under the AM-5 convention). Both are
serialized first and created exclusively. Only their sha256 and the counts are printed.

--names-check (ruling R4; local command 0): lists the four folders and prints only the number of TRAIN
and VAL names containing "test"; a count above 0 is a STOP (exit 1). It writes nothing.

Exit codes: 0 written / names clean, 1 STOP, 2 refusal or usage, 4 unexpected exception. Every hashing
run is a run of record (--script-commit and --script-commit-dl-id; --generated-utc refused).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval import teacher_diag as td  # noqa: E402

SCRIPT = "scripts/hash_split_files.py"
SCHEMA = "plantseg-val-train-duplicates/1.0.0"
SPLITS = ("train", "val")
FOLDERS = ("images/train", "images/val", "annotations/train", "annotations/val")
IMAGE_SUFFIXES = (".jpg", ".jpeg")
MASK_SUFFIX = ".png"
EXPECTED = {"train": td.TRAIN_ROWS, "val": td.VAL_ROWS}
TSV_NAME, JSON_NAME = "train_val_image_hashes.tsv", "val_train_duplicates.json"
ISSUE_11 = {"train_val_groups": 32, "three_way_or_more": 15}
DETECTION = "byte-identical files only (SHA-256 of the file bytes); re-encoded copies are not detected"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D4: hash TRAIN and VAL images; VAL images byte-identical to TRAIN.")
    p.add_argument("--data-root", required=True, help="holds images/{train,val} and annotations/{train,val}")
    p.add_argument("--names-check", action="store_true", help="R4: count names containing 'test' only")
    p.add_argument("--out", help="output folder outside the repository (hashing mode)")
    p.add_argument("--strata", help="the strata file whose split_list_sha256 the TRAIN list must equal")
    p.add_argument("--script-commit")
    p.add_argument("--script-commit-dl-id")
    p.add_argument("--generated-utc", help="refused: outputs of record are stamped from the script's clock")
    p.add_argument("--repeat-of")
    p.add_argument("--repeat-case")
    p.add_argument("--repeat-dl-id")
    return p


def guard_strings(args) -> None:
    """P21, before any filesystem call: every path as given, by its string only."""
    named = [args.data_root, *(os.path.join(args.data_root, f) for f in FOLDERS)]
    named += [x for x in (args.out, args.strata) if x is not None]
    bad = [s for s in named if "test" in str(s).lower()]
    if bad:
        raise td.Refused(f"a path containing 'test' is refused ({td.name_digests(bad)})")


def guard_resolved(args) -> None:
    for s in [args.data_root, *(os.path.join(args.data_root, f) for f in FOLDERS),
              *(x for x in (args.out, args.strata) if x is not None)]:
        td.refuse_test_path(s, "path")


def list_folder(folder: str) -> list[str]:
    """File names in one of the four folders (one os.scandir call; nothing else is listed)."""
    if not os.path.isdir(folder):
        raise td.Refused(f"missing folder under --data-root ({td.name_digests([folder])})")
    with os.scandir(folder) as it:
        return sorted(e.name for e in it if e.is_file())


def listing(root: str) -> dict:
    """{split: {"images": [names], "masks": set(names)}} from exactly the four folders."""
    out = {}
    for split in SPLITS:
        imgs = [n for n in list_folder(os.path.join(root, "images", split))
                if os.path.splitext(n)[1].lower() in IMAGE_SUFFIXES]
        masks = {n for n in list_folder(os.path.join(root, "annotations", split))
                 if os.path.splitext(n)[1] == MASK_SUFFIX}
        out[split] = {"images": imgs, "masks": masks}
    return out


def names_check(root: str) -> int:
    counts = {}
    for split in SPLITS:
        names = list_folder(os.path.join(root, "images", split)) + list_folder(os.path.join(root, "annotations", split))
        counts[split] = sum(1 for n in names if "test" in n.lower())
    print(f"names-check: TRAIN names containing 'test': {counts['train']}; VAL names containing 'test': "
          f"{counts['val']}")
    if counts["train"] or counts["val"]:
        print("STOP: a name containing 'test' needs a ruling before any data-dependent command (R4)",
              file=sys.stderr)
        return td.EXIT_STOP
    return td.EXIT_OK


def digests(path: str) -> tuple[str, str, int]:
    """One read feeds SHA-256 and SHA-1."""
    h256, h1, n = hashlib.sha256(), hashlib.sha1(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h256.update(block)
            h1.update(block)
            n += len(block)
    return h256.hexdigest(), h1.hexdigest(), n


def masks_identical(root: str, members: list[tuple[str, str]]) -> bool:
    import numpy as np
    from PIL import Image
    arrays = []
    for split, stem in members:
        with Image.open(os.path.join(root, "annotations", split, stem + MASK_SUFFIX)) as mk:
            arrays.append(np.asarray(mk).copy())
    first = arrays[0]
    return all(a.shape == first.shape and a.dtype == first.dtype and np.array_equal(a, first) for a in arrays[1:])


def group(rows: list[dict]) -> dict:
    by_sha: dict[str, list[dict]] = {}
    for r in rows:
        by_sha.setdefault(r["sha256"], []).append(r)
    cross, val_only, train_only = [], [], 0
    for sha in sorted(by_sha):
        members = by_sha[sha]
        if len(members) < 2:
            continue
        splits = {m["split"] for m in members}
        if splits == {"train", "val"}:
            cross.append((sha, members))
        elif splits == {"val"}:
            val_only.append((sha, members))
        else:
            train_only += 1
    return {"cross": cross, "val_only": val_only, "train_only": train_only}


def run(args) -> int:
    return td.run_with_exit_codes(_run, args)


def _run(args) -> int:
    start, t0 = td.utc_now(), time.monotonic()
    guard_strings(args)                                   # no filesystem call before this
    if args.names_check:
        if args.out is not None or args.strata is not None:
            raise td.Refused("--names-check writes nothing: --out and --strata belong to the hashing mode")
        guard_resolved(args)
        return names_check(args.data_root)
    if args.out is None or args.strata is None:
        raise td.Refused("the hashing mode requires --out and --strata")
    td.check_common_flags(args, real=True)
    guard_resolved(args)
    out = td.require_outside_repo(args.out, "--out")
    code = td.require_commit_binding(args.script_commit, args.script_commit_dl_id)
    targets = (out / TSV_NAME, out / JSON_NAME)
    if any(os.path.lexists(p) for p in targets):
        raise td.Refused(f"--out already holds {TSV_NAME} or {JSON_NAME}; a repeat goes to a new --out")

    root = args.data_root
    lst = listing(root)
    for split in SPLITS:
        names = lst[split]["images"] + sorted(lst[split]["masks"])
        td.refuse_test_names(names, f"{split} names")
        n = len(lst[split]["images"])
        if n != EXPECTED[split]:
            raise td.Refused(f"{split}: {n} images, expected {EXPECTED[split]}")
        stems = [os.path.splitext(x)[0] for x in lst[split]["images"]]
        if len(set(stems)) != len(stems):
            raise td.Refused(f"{split}: two images share a stem")
        missing = [s for s in stems if s + MASK_SUFFIX not in lst[split]["masks"]]
        if missing:
            raise td.Refused(f"{split}: {len(missing)} image(s) without a {MASK_SUFFIX} mask")
        lst[split]["stems"] = stems

    from scripts.build_train_strata import split_list_sha256
    from src.eval.artifacts import am5_excluded_ids_sha256, hash_split_manifest
    from src.eval.evaluate import ManifestEntry
    train_list_sha = split_list_sha256(lst["train"]["stems"])
    strata = td.check_strata(args.strata, lst["train"]["stems"], stub=False)     # refuses another split list
    val_manifest_sha = hash_split_manifest([ManifestEntry(i, s, s) for i, s in enumerate(lst["val"]["stems"])])

    rows = []
    for split in SPLITS:
        for name, stem in zip(lst[split]["images"], lst[split]["stems"]):
            sha, sha1, nbytes = digests(os.path.join(root, "images", split, name))
            rows.append({"split": split, "stem": stem, "file": name, "bytes": nbytes, "sha256": sha, "sha1": sha1})
    g = group(rows)
    groups = []
    for sha, members in g["cross"]:
        groups.append({"sha256": sha, "size": len(members),
                       "train": sorted(m["stem"] for m in members if m["split"] == "train"),
                       "val": sorted(m["stem"] for m in members if m["split"] == "val"),
                       "masks_identical": masks_identical(root, [(m["split"], m["stem"]) for m in members])})
    val_only = [{"sha256": sha, "size": len(members), "val": sorted(m["stem"] for m in members),
                 "masks_identical": masks_identical(root, [("val", m["stem"]) for m in members])}
                for sha, members in g["val_only"]]
    dup_val = sorted({s for gr in groups for s in gr["val"]})
    sizes: dict[str, int] = {}
    for gr in groups:
        sizes[str(gr["size"])] = sizes.get(str(gr["size"]), 0) + 1
    here = {"train_val_groups": len(groups), "three_way_or_more": sum(1 for gr in groups if gr["size"] >= 3)}

    tsv = "split\tstem\tfile\tbytes\tsha256\tsha1\n" + "".join(
        f"{r['split']}\t{r['stem']}\t{r['file']}\t{r['bytes']}\t{r['sha256']}\t{r['sha1']}\n" for r in rows)
    tsv_b = tsv.encode("utf-8")
    doc = td.base_document(SCRIPT, args, stub=False, start_utc=start, code=code, extra={
        "schema": SCHEMA, "detection": DETECTION,
        "repeat": None if args.repeat_of is None else {"repeat_of": args.repeat_of, "repeat_case": args.repeat_case,
                                                       "repeat_dl_id": args.repeat_dl_id},
        "inputs": {"folders": list(FOLDERS), "strata": strata},
        "counts": {"train_images": len(lst["train"]["stems"]), "val_images": len(lst["val"]["stems"])},
        "hash_list": {"name": TSV_NAME, "sha256": hashlib.sha256(tsv_b).hexdigest(), "rows": len(rows),
                      "columns": ["split", "stem", "file", "bytes", "sha256", "sha1"]},
        "val_manifest": {"hash_split_manifest": val_manifest_sha,
                         "rule": "ManifestEntry(i, stem, stem) over the name-sorted VAL stems (the evaluator's)"},
        "train_split_list_sha256": train_list_sha, "train_split_list_equals_strata": True,
        "groups": groups, "group_sizes": sizes, "val_only_groups": val_only,
        "val_only_note": "VAL images matching only other VAL images: reported, not removed",
        "train_only_groups": g["train_only"],
        "duplicate_val_ids": dup_val, "n_duplicate_val_ids": len(dup_val),
        "duplicate_val_ids_sha256": am5_excluded_ids_sha256(dup_val),
        "duplicate_val_ids_sha256_rule": "sha256 of the sorted ids joined by '\\n' (AM-5 convention)",
        "issue_11": {"reported": ISSUE_11, "this_run": here, "differs": here != ISSUE_11,
                     "note": "a difference is reported, never a STOP; issue #11 may count other files"},
        "status": "written",
    })
    doc["environment"] = td.environment_block(start, td.utc_now())
    doc["wall_seconds"] = round(time.monotonic() - t0, 3)
    json_b = td.json_bytes(doc)
    td.write_files_exclusive([(out / TSV_NAME, tsv_b), (out / JSON_NAME, json_b)])
    print(f"written: {TSV_NAME} sha256 {hashlib.sha256(tsv_b).hexdigest()} ({len(rows)} rows); "
          f"{JSON_NAME} sha256 {hashlib.sha256(json_b).hexdigest()}")
    print(f"TRAIN {len(lst['train']['stems'])}, VAL {len(lst['val']['stems'])}; duplicate groups {len(groups)} "
          f"(three or more: {here['three_way_or_more']}); duplicate VAL images {len(dup_val)}; VAL-only groups "
          f"{len(val_only)}; issue #11 {ISSUE_11['train_val_groups']}/{ISSUE_11['three_way_or_more']}")
    return td.EXIT_OK


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
