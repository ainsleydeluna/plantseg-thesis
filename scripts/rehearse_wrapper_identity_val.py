#!/usr/bin/env python3
"""R-G4 on real data: the corrupted-condition wrapper's identity on every VAL image (L-CORR B-6;
plan section 6(xii); rehearsal only, never TEST).

  python -B scripts/rehearse_wrapper_identity_val.py --split val --data-root <TRAIN/VAL-only root>
      --corruption-cache <the provisional VAL cache> --expected-cache-sha256 <its checksum>
      --verification-record <the verifier's record> --out <new folder outside the repository>

Gates, in this order, each a refusal (exit 2) before anything is written:
  * `--split` must be val (`split_not_val`), checked before any path is read;
  * the out-dir: TL interface section 2 (no "test" in any case, none of :<>"\\|?* in any
    component, as given and resolved), outside the repository, not existing (B-4's
    check_out_dir: `out_dir_name_invalid`, `out_dir_inside_repo`, `out_dir_exists`);
  * `--expected-cache-sha256`, when given, is 64 lowercase hex (`expected_cache_sha256_format`);
  * the data root passes the M11 isolation check (src/data/isolation.py: no TEST surface, the
    locked TRAIN and VAL counts);
  * the cache passes scripts/evaluate_corruptions.py's own gates for status provisional: the
    manifest, its split (from the pin), the cell CELL, a provisional or official pinned cache whose
    canonical checksum is --expected-cache-sha256, the protocol file it was generated under, and the
    verifier's record of it; its sources are the VAL listing's 846 (manifest_index, stem) pairs
    (`cache_sources_mismatch`).

Then, for each of the 846 sources in ascending manifest_index, it builds src/eval/corrupted.py's
wrapper sample for the one cell CELL, whose item is a stub returning the image's clean resized
pixels: the window of the clean canvas (the frozen core_preprocess) at the cache's recorded resized
shape. The wrapper's own checks still run (the source digest against the clean canvas window, the
window and the canvas outside it, the item's shape). The sample's image and target must equal the
clean evaluator's dataset sample (src.eval.adapters.PlantSegEvalDataset) bitwise (dtype, shape and
bytes), and its image_id and clean_image_id must be the listing's stem and its manifest_index the
listing's index. A wrapper refusal for one image is that image's mismatch, by its code.

Outputs: rehearsal_wrapper_identity.json {schema, split, n_checked, n_equal, mismatches (the first
5), cache_canonical_manifest_sha256, commit, started_utc, written_utc} and SHA256SUMS, written
atomically to a new folder outside the repository by B-4's writer. A rehearsal check: nonofficial.

Exit codes: 0 written, every sample equal; 1 written, a mismatch; 2 refused, nothing written; 4 an
unexpected error, nothing written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from configs.data import DATA  # noqa: E402
from src import corruption_roles  # noqa: E402
from src.corruption_cache import CorruptionCacheError  # noqa: E402
from src.data.isolation import TrainValIsolationError, assert_trainval_only_root  # noqa: E402
from src.data.transforms import core_preprocess  # noqa: E402
from src.eval.adapters import PlantSegEvalDataset, build_expected_manifest_for  # noqa: E402
from src.eval.corrupted import (CacheCell, CorruptedEvalError,  # noqa: E402
                                PlantSegCorruptedEvalDataset, utc_iso, window)
from src.eval.robustness_report import ReportRefusal, check_out_dir, write_dir_atomic  # noqa: E402
from scripts.evaluate_corruptions import (check_cache_sources,  # noqa: E402
                                          check_verification_record, refusal_line,
                                          resolve_corruption_cache)

SCHEMA = "plantseg-wrapper-identity-rehearsal/1.0.0"
OUT_NAME = "rehearsal_wrapper_identity.json"
SPLIT, ROWS, STATUS = "val", 846, "provisional"
#: The one cell whose wrapper is built; its item is the stub, so the cell need only be in the cache.
CELL = corruption_roles.INFERENTIAL_CELL_IDS[0]
FIRST_MISMATCHES = 5
EXIT_EQUAL, EXIT_MISMATCH, EXIT_REFUSED, EXIT_UNEXPECTED = 0, 1, 2, 4
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class IdentityRefusal(RuntimeError):
    """A request this check refuses by code, before anything is written."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _refuse(code: str, message: str):
    raise IdentityRefusal(code, message)


#: Every coded refusal: this check's own, the evaluator's cache gates, M11, B-4's out-dir, writer.
REFUSALS = (IdentityRefusal, CorruptionCacheError, TrainValIsolationError, ReportRefusal)


def git_head() -> str | None:
    """HEAD of this checkout (40 hex), or None when Git is unavailable."""
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                           text=True, timeout=60, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    head = p.stdout.strip()
    return head if p.returncode == 0 and _HEX40.fullmatch(head) else None


def check_split(split: str) -> None:
    if split != SPLIT:
        _refuse("split_not_val", f"--split {split!r}: this rehearsal check reads VAL only; nothing "
                                 "was read")


def check_expected_format(expected) -> None:
    if expected is not None and not _HEX64.fullmatch(str(expected)):
        _refuse("expected_cache_sha256_format",
                f"--expected-cache-sha256 {expected!r} is not 64 lowercase hex characters")


def check_cache(args) -> dict:
    """The evaluator's own cache gates for status provisional, on the cell CELL, then the
    verifier's record of this cache. Returns the loaded manifest."""
    corruption, severity = corruption_roles.parse_cell_id(CELL)
    manifest = resolve_corruption_cache(args.corruption_cache, corruption, severity, split=SPLIT,
                                        official=False, artifact_status=STATUS,
                                        expected_sha256=args.expected_cache_sha256)
    check_verification_record(args.corruption_cache, manifest, args.verification_record,
                              artifact_status=STATUS)
    return manifest


class CleanWindowItem:
    """The stub item: the image's clean resized pixels, the window of the clean canvas (the frozen
    core_preprocess) at the cache's recorded resized shape."""

    def __init__(self, pairs: dict, source):
        self.pairs, self.source = pairs, source       # stem -> (image, mask); the cell's source()

    def __call__(self, stem: str) -> np.ndarray:
        img, msk = self.pairs[stem]
        with Image.open(img) as im, Image.open(msk) as mk:
            canvas, _ = core_preprocess(im, mk)
        top, left, h, w = window(self.source(stem)["resized_shape"])
        return np.ascontiguousarray(canvas[top:top + h, left:left + w])


def same_tensor(a, b) -> bool:
    """Bitwise: one dtype, one shape and the same bytes."""
    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes())


def identity_problems(got: dict, want: dict, entry) -> list:
    """image_id == clean_image_id == the listing's stem and manifest_index the listing's index, in
    the wrapper's sample and in the clean one."""
    ok = all(s["image_id"] == s["clean_image_id"] == entry.image_id
             and s["manifest_index"] == entry.manifest_index for s in (got, want))
    return [] if ok else ["identity"]


def sample_problems(wrapper, clean, i: int, entry) -> list:
    try:
        got = wrapper[i]
    except CorruptedEvalError as e:
        return [e.code]
    want = clean[i]
    problems = [k for k in ("image", "target") if not same_tensor(got[k], want[k])]
    return problems + identity_problems(got, want, entry)


def compare(args, manifest: dict) -> tuple[int, int, list]:
    """(n_checked, n_equal, the first mismatches) over every source, ascending manifest_index."""
    expected = build_expected_manifest_for(SPLIT, range(ROWS))
    check_cache_sources(manifest, expected, full=True)
    corruption, severity = corruption_roles.parse_cell_id(CELL)
    cell = CacheCell(Path(args.corruption_cache).resolve(), manifest, corruption, severity)
    wrapper = PlantSegCorruptedEvalDataset(SPLIT, range(ROWS), cell)
    clean = PlantSegEvalDataset(SPLIT, range(ROWS))
    cell.load = CleanWindowItem({p[0].stem: p for p in wrapper.base.pairs}, cell.source)
    n_equal, mismatches = 0, []
    for i, entry in enumerate(expected):
        problems = sample_problems(wrapper, clean, i, entry)
        if not problems:
            n_equal += 1
        elif len(mismatches) < FIRST_MISMATCHES:
            mismatches.append({"manifest_index": entry.manifest_index, "image_id": entry.image_id,
                               "problems": problems})
    return len(expected), n_equal, mismatches


def run(args, *, started_utc: str | None = None) -> dict:
    started = started_utc or utc_iso()
    check_split(args.split)
    check_out_dir(args.out)
    check_expected_format(args.expected_cache_sha256)
    assert_trainval_only_root(args.data_root)
    saved = DATA["root"]
    DATA["root"] = str(Path(args.data_root).resolve())
    try:
        manifest = check_cache(args)
        n_checked, n_equal, mismatches = compare(args, manifest)
    finally:
        DATA["root"] = saved
    return {"schema": SCHEMA, "split": SPLIT, "n_checked": n_checked, "n_equal": n_equal,
            "mismatches": mismatches,
            "cache_canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
            "commit": git_head(), "started_utc": started, "written_utc": utc_iso()}


def write_outputs(out_dir, doc: dict) -> Path:
    """The JSON and its SHA256SUMS through B-4's atomic writer."""
    payload = (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
               + "\n").encode("utf-8")
    return write_dir_atomic(out_dir, {OUT_NAME: payload})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", required=True, help="val (any other split is refused)")
    p.add_argument("--data-root", required=True, help="the TRAIN/VAL-only staged root (M11)")
    p.add_argument("--corruption-cache", required=True, help="the provisional VAL cache's root")
    p.add_argument("--expected-cache-sha256", default=None,
                   help="the cache's canonical_manifest_sha256 of record")
    p.add_argument("--verification-record", default=None,
                   help="the record scripts/verify_corruption_cache.py wrote for this cache")
    p.add_argument("--out", required=True, help="a new folder outside the repository")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        doc = run(args)
        out = write_outputs(args.out, doc)
    except REFUSALS as e:
        print(refusal_line(e), file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001 -- reported, exit 4
        print(f"unexpected error, nothing written: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    print(f"wrapper identity outputs: {out}")
    print(f"R-G4: {doc['n_equal']} of {doc['n_checked']} samples bitwise equal to the clean "
          f"evaluator's; first mismatches {doc['mismatches']}")
    return EXIT_EQUAL if doc["n_equal"] == doc["n_checked"] == ROWS else EXIT_MISMATCH


if __name__ == "__main__":
    raise SystemExit(main())
