#!/usr/bin/env python3
"""Verify a corruption cache against its split files and the real core_preprocess (L-CORR B-2).

Torch side: the source check runs the REAL src/data/transforms.core_preprocess, the frozen function
the evaluator uses, so every source digest the torch-free generator recorded is proven equal to the
window of the clean canvas the model sees. The window is located by core_preprocess itself (the
non-padding region of its mask canvas), not by a second copy of the resize rule. The corrupted
items are re-hashed here, never regenerated.

  verify_corruption_cache.py --cache-dir C --data-root R --split {val,test} [--confirm-test-split]

Checks, in order, each a refusal by code: names (LC6: the cache root and the record directory);
--confirm-test-split for test, before any read; the manifest (schema 2.0.0, canonical checksum,
seed policy, vendored runtime, item paths by index: item_path_mismatch); the manifest's split; for
an official or provisional cache, the student image (PLANTSEG_IMAGE_DIGEST) and the pinned stack
(LC2); the data root (the M11 isolation check for val, images/test alone for test); the sources
against the split listing (each recorded manifest_index and file name, and all of them for a full
cache); each source file's sha256; its core_preprocess window (shape, pixel sha256, mean-colour
padding outside it); every item (file sha256, dtype, its source's resized shape, pixel sha256); the
cell digests, the registered 24-cell set (also the pin's cells), the inferential digest and the
pin's own sha256.

On success it writes verification/<UTC>.json (outside the checksum) and prints the four recorded
values of the TL interface section 5, one per line, each labelled:
  canonical_manifest_sha256: <hex>
  inferential_cells_sha256: <hex>
  manifest_file_sha256: <hex>
  verification_record_sha256: <hex>
A refused verification writes no record. Exit 0: verified; exit 2: refused.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src import corruption_cache as cc  # noqa: E402
from src import corruption_roles  # noqa: E402
from src import corruption_source as cs  # noqa: E402
from src.corruption_cache import CorruptionCacheError  # noqa: E402
from src.corruption_source import CorruptionSourceError  # noqa: E402
from src.data.transforms import IMAGENET_MEAN_8BIT, MASK_IGNORE, core_preprocess  # noqa: E402

VERIFICATION_SCHEMA = "plantseg-corruption-verification/1.0.0"
RECORD_DIR = "verification"
PINNED_STATUSES = ("official", "provisional")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify a PlantSeg corruption cache.")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", required=True, choices=list(cs.SPLITS))
    p.add_argument("--confirm-test-split", action="store_true")
    return p


def _refuse(code: str, message: str):
    raise CorruptionCacheError(code, message)


# ---------------------------------------------------------------- gates, in order
def gate_names(args) -> None:
    cc.refuse_test_named(cache_root=args.cache_dir,
                         verification_dir=str(Path(args.cache_dir) / RECORD_DIR))


def gate_test_confirm(args) -> None:
    if args.split == "test" and not args.confirm_test_split:
        _refuse("test_split_unconfirmed",
                "--split test requires --confirm-test-split; nothing was read")


def gate_split(args, manifest: dict) -> None:
    split = manifest.get("split")
    if split != args.split or manifest.get("dataset", {}).get("split") != args.split:
        _refuse("split_mismatch",
                f"the cache holds split {split!r}, --split is {args.split!r}")


def gate_environment(manifest: dict) -> None:
    """LC2, TL-1: an official or provisional cache is verified in the student image on the pinned
    stack, as it was generated (the window check runs Pillow's BILINEAR). Before any data read."""
    if manifest.get("status") not in PINNED_STATUSES:
        return
    got = os.environ.get("PLANTSEG_IMAGE_DIGEST")
    if got != cc.STUDENT_IMAGE_DIGEST:
        _refuse("image_digest_mismatch",
                f"PLANTSEG_IMAGE_DIGEST is {got!r}; a {manifest['status']} cache is verified in "
                f"the student image {cc.STUDENT_IMAGE_DIGEST}")
    cc.require_official_dependency_environment()


def gate_sources(args, manifest: dict) -> dict:
    """Recorded sources against the split listing: index, stem and file name; all for a full
    cache. Returns image_id -> Source."""
    listing = cs.list_split(args.data_root, args.split)
    recorded = manifest["sources"]
    bad = [r["image_id"] for r in recorded
           if not (0 <= r["manifest_index"] < len(listing))
           or listing[r["manifest_index"]].image_id != r["image_id"]
           or listing[r["manifest_index"]].path.name != r["file_name"]]
    if manifest.get("expected_images") is not None and len(recorded) != len(listing):
        bad.append(f"{len(recorded)} recorded of {len(listing)} listed")
    if bad:
        _refuse("sources_mismatch",
                f"the recorded sources differ from the split listing: {bad[:5]}")
    return {r["image_id"]: listing[r["manifest_index"]] for r in recorded}


def core_window(raw: bytes):
    """(window, canvas, mask canvas, (top, left, h, w)) from the real core_preprocess."""
    with Image.open(io.BytesIO(raw)) as im:
        canvas, mcanvas = core_preprocess(im, Image.new("L", (1, 1), 1))
    ys, xs = np.nonzero(mcanvas != MASK_IGNORE)
    top, left = int(ys.min()), int(xs.min())
    h, w = int(ys.max()) - top + 1, int(xs.max()) - left + 1
    return canvas[top:top + h, left:left + w], canvas, mcanvas, (top, left, h, w)


def check_file(record: dict, raw: bytes) -> None:
    if cc._sha256_bytes(raw) != record["source_file_sha256"]:
        _refuse("source_file_mismatch",
                f"{record['image_id']}: the source file changed since generation")


def check_window(record: dict, raw: bytes) -> None:
    window, canvas, mcanvas, (top, left, h, w) = core_window(raw)
    padding = canvas[mcanvas == MASK_IGNORE]
    if ([h, w] != record["resized_shape"] or int((mcanvas != MASK_IGNORE).sum()) != h * w
            or cc._sha256_bytes(np.ascontiguousarray(window).tobytes())
            != record["resized_pixel_sha256"]
            or (padding.size and not np.all(padding == np.array(IMAGENET_MEAN_8BIT, np.uint8)))):
        _refuse("source_pixels_mismatch",
                f"{record['image_id']}: the recorded source digest is not the core_preprocess "
                f"window ({h}x{w} at {top},{left}; recorded {record['resized_shape']})")


def check_source(record: dict, source) -> None:
    raw = Path(source.path).read_bytes()
    check_file(record, raw)
    check_window(record, raw)


def check_item_shape(entry: dict, arr: np.ndarray, resized_shape) -> None:
    if list(arr.shape) != list(resized_shape) + [3]:
        _refuse("cache_entry_corrupt",
                f"{entry['cache_relative_path']} is {list(arr.shape)}, its source resized to "
                f"{list(resized_shape)}")


def check_items(cache_root, manifest: dict) -> int:
    shapes = {s["image_id"]: s["resized_shape"] for s in manifest["sources"]}
    for e in manifest["entries"]:
        arr = cc.load_cached(cache_root, manifest, e["image_id"], e["corruption_id"], e["severity"])
        check_item_shape(e, arr, shapes[e["image_id"]])
    return len(manifest["entries"])


def cell_set_problems(manifest: dict) -> list:
    """LC13, TL section 6: the cells are the registered 24 in protocol order, and the embedded
    pin's cells are the same list."""
    ids = [c.get("cell_id") for c in manifest["cells"]]
    out = [] if ids == list(corruption_roles.CELL_IDS) else ["the registered cell set"]
    pin = manifest.get("pin")
    if pin is not None and (pin.get("parameters") or {}).get("cells") != ids:
        out.append("the pin's cells")
    return out


def check_cells(manifest: dict) -> None:
    sources = manifest["sources"]
    index_of = {s["image_id"]: s["manifest_index"] for s in sources}
    by_cell: dict = {}
    for e in manifest["entries"]:
        by_cell.setdefault((e["corruption_id"], e["severity"]), []).append(e)
    problems = []
    for cell in manifest["cells"]:
        ce = sorted(by_cell.get((cell["corruption"], cell["severity"]), []),
                    key=lambda e: index_of[e["image_id"]])
        try:
            named = corruption_roles.cell_id(cell["corruption"], cell["severity"])
        except corruption_roles.CorruptionRolesError:
            named = None
        if (cell["items"] != len(ce) or len(ce) != len(sources) or cell["cell_id"] != named
                or cell["cell_sha256"] != cc.cell_digest(ce)):
            problems.append(cell["cell_id"])
    if len(manifest["entries"]) != len(sources) * len(manifest["cells"]):
        problems.append("entries != sources x cells")
    problems.extend(cell_set_problems(manifest))
    inferential = [c["cell_sha256"] for c in manifest["cells"]
                   if c["role"] == corruption_roles.ROLE_INFERENTIAL]
    inferential_sha = cc._sha256_bytes(cc.canonical_json_bytes(inferential))
    if inferential_sha != manifest["inferential_cells_sha256"]:
        problems.append("inferential_cells_sha256")
    if problems:
        _refuse("cell_digest_mismatch", f"cell digests do not recompute: {problems[:5]}")


def check_pin(manifest: dict) -> None:
    pin = manifest.get("pin")
    if pin is not None and cc.pin_digest(pin) != pin.get("pin_sha256"):
        _refuse("pin_invalid", "the embedded pin's pin_sha256 does not recompute")


def verify(args) -> dict:
    gate_names(args)
    gate_test_confirm(args)
    root = Path(args.cache_dir)
    manifest = cc.load_manifest(root)
    gate_split(args, manifest)
    gate_environment(manifest)
    cs.check_split_surface(args.data_root, args.split)
    by_id = gate_sources(args, manifest)
    for record in manifest["sources"]:
        check_source(record, by_id[record["image_id"]])
    items = check_items(root, manifest)
    check_cells(manifest)
    check_pin(manifest)

    import torch
    record = {"schema": VERIFICATION_SCHEMA, "utc": cc.utc_iso(), "cache_root": str(root.resolve()),
              "split": manifest["split"], "status": manifest["status"],
              "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
              "inferential_cells_sha256": manifest["inferential_cells_sha256"],
              "manifest_file_sha256": cc._sha256_file(root / cc.MANIFEST_NAME),
              "counts": {"sources_verified": len(manifest["sources"]), "items_verified": items,
                         "cells": len(manifest["cells"])},
              "pin_sha256": (manifest.get("pin") or {}).get("pin_sha256"),
              "python": sys.version.split()[0], "torch": torch.__version__,
              "versions": cc.runtime_versions(),
              "image_digest": os.environ.get("PLANTSEG_IMAGE_DIGEST")}
    data = cc.canonical_json_bytes(record) + b"\n"
    rdir = root / RECORD_DIR
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"{cc.utc_stamp()}.json"
    cc._write_atomic(path, data)
    return {**record, "verification_record": str(path),
            "verification_record_sha256": cc._sha256_bytes(data)}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = verify(args)
    except (CorruptionCacheError, CorruptionSourceError) as e:
        print(f"CACHE VERIFICATION FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2
    for key in ("canonical_manifest_sha256", "inferential_cells_sha256", "manifest_file_sha256",
                "verification_record_sha256"):
        print(f"{key}: {out[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
