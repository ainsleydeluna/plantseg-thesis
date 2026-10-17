"""Corrupted-condition evaluation: the cache cell, the dataset wrapper and the cell record (L-CORR B-3).

`scripts/evaluate_corruptions.py` scores one (stage, corruption, severity) cell through the unchanged
core (`src/eval/evaluate.py`, `artifacts.py`, `eval_runtime.py`). This module supplies the three
things that core leaves to an adapter, without touching its semantics:

  * `CacheCell` -- one scored cell of a loaded cache. Sources and items are found through B-2's
    `CacheIndex` (`src.corruption_cache.cache_index`); items are read by an injected PATH B loader
    (the runner passes `src.eval.robustness.load_cached_corruption`, which reads through
    `src.corruption_cache.load_cached`), hash-verified on every read and never regenerated.
  * `PlantSegCorruptedEvalDataset` -- the identity-carrying wrapper. `PlantSegDataset` is never
    modified: its pairs, missing-mask guard and count guard are reused. `__getitem__` runs the frozen
    `core_preprocess` on the image and its mask, so the target is exactly the clean path's
    (src/data/transforms.py:52-58). The cell's recorded source digest must equal the window of the
    clean canvas (`source_pixels_mismatch`), and the canvas outside that window must be the mean
    colour (`canvas_window_mismatch`). The window is `_pad_to_square`'s rule (transforms.py:41-49).
    The cached item, of the window's shape (`corrupted_shape_mismatch`), replaces the window. The
    padding therefore stays the clean canvas's, and `finalize` normalises as on the clean path.
  * The cell record `<out_dir>.corruption.json` (`plantseg-corruption-eval-record/1.0.0`; TL interface
    section 7). It is written atomically after the artifact, an existing record is refused, and it
    binds the artifact's four files to the cache it scored. The artifact itself stays layout 1.2.0
    (LE9), so `src/eval/artifacts.py` is unchanged.

Identity (DL-74; EVALUATION_CONTRACT section 6): every row carries the bare stem as `image_id` and
`clean_image_id`, resolved inside `__getitem__`, and the clean split's `manifest_index`. The condition
travels only in `condition`. The expected manifest is built from the filename listing, never from the
cache, so a corrupted artifact has the clean artifact's `split_manifest_sha256`.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .. import corruption_roles
from ..corruption_cache import _strict_json, cache_index
from ..data.dataset import PlantSegDataset
from ..data.transforms import IMAGENET_MEAN_8BIT, SIZE, core_preprocess, finalize
from .adapters import AdapterError
from .artifacts import ARTIFACT_FILES, MANIFEST_NAME

CELL_RECORD_SCHEMA = "plantseg-corruption-eval-record/1.0.0"
RECORD_SUFFIX = ".corruption.json"
EVALUATOR_SCRIPT = "scripts/evaluate_corruptions.py"
#: The artifact's four files, in the order the record lists their sha256.
ARTIFACT_FOUR_FILES = tuple(sorted((*ARTIFACT_FILES, MANIFEST_NAME)))
#: The record's keys (TL interface section 7), in this order.
CELL_RECORD_KEYS = ("schema", "artifact_dir", "artifact_sha256", "run_id", "stage", "condition",
                    "role", "cache", "items_verified", "source_digests_verified",
                    "evaluator_script", "evaluator_commit", "started_utc", "written_utc")
CACHE_KEYS = ("canonical_manifest_sha256", "cell_sha256", "inferential_cells_sha256", "split",
              "status", "source_image_count", "verification_record_sha256")
CORRUPTED_SPLITS = ("val", "test")
#: Out-dir names (TL interface section 2): TEST `tday_corr_<file_id>_<cell id>` and
#: `tday_clean_<file_id>`; a VAL rehearsal uses the same forms with the prefix `val`.
OUT_PREFIX = {"test": "tday", "val": "val"}


def corrupted_dir_name(split: str, file_id: str, cell_id: str) -> str:
    return f"{OUT_PREFIX[split]}_corr_{file_id}_{cell_id}"


def clean_dir_name(split: str, file_id: str) -> str:
    return f"{OUT_PREFIX[split]}_clean_{file_id}"


class CorruptedEvalError(RuntimeError):
    """A refused corrupted-condition request, cell, sample or record. `code` keeps it testable."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _fail(code: str, message: str):
    raise CorruptedEvalError(code, message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --------------------------------------------------------------------------------------------------
# the cell
# --------------------------------------------------------------------------------------------------
@dataclass
class CacheCell:
    """One scored cell of a loaded cache manifest.

    `load(image_id)` is the PATH B loader the caller injects; it must hash-verify every read and
    never regenerate. Sources come from B-2's index, built once per manifest.
    """
    cache_root: Path
    manifest: dict
    corruption: str
    severity: int
    load: Callable[[str], np.ndarray] = field(repr=False, default=None)

    @property
    def cell_id(self) -> str:
        return corruption_roles.cell_id(self.corruption, self.severity)

    def entry(self) -> dict:
        """The manifest's `cells[]` record of this cell; refused when the cache does not hold it."""
        for c in self.manifest.get("cells") or []:
            if c.get("cell_id") == self.cell_id:
                return c
        _fail("cache_cell_absent", f"the cache holds no cell {self.cell_id!r}; its cells are "
                                   f"{[c.get('cell_id') for c in self.manifest.get('cells') or []]}")

    def source(self, image_id: str) -> dict:
        rec = cache_index(self.manifest).sources.get(image_id)
        if rec is None:
            _fail("cache_source_absent", f"the cache records no source {image_id!r}")
        return rec


# --------------------------------------------------------------------------------------------------
# the canvas
# --------------------------------------------------------------------------------------------------
def window(resized_shape) -> tuple[int, int, int, int]:
    """(top, left, h, w) of a resized image on the 512x512 canvas: _pad_to_square's rule."""
    h, w = (int(v) for v in resized_shape)
    return (SIZE - h) // 2, (SIZE - w) // 2, h, w


def corrupted_canvas(stem: str, manifest_index: int, clean_canvas: np.ndarray, cell) -> np.ndarray:
    """The clean canvas with its image window replaced by the cell's cached item.

    Proves first that the item was made from exactly the pixels the clean evaluator sees: the
    recorded source digest equals the window of this clean canvas, and everything outside the
    window is the mean colour. The item is read only after both checks pass.
    """
    rec = cell.source(stem)
    if rec.get("manifest_index") != manifest_index:
        _fail("cache_source_index_mismatch",
              f"{stem}: the cache records manifest_index {rec.get('manifest_index')!r}, the split "
              f"listing gives {manifest_index}")
    top, left, h, w = window(rec["resized_shape"])
    if not (0 <= top and 0 <= left and h >= 1 and w >= 1 and top + h <= SIZE and left + w <= SIZE):
        _fail("canvas_window_mismatch", f"{stem}: recorded resized shape {rec['resized_shape']} "
                                        f"does not fit the {SIZE}x{SIZE} canvas")
    win = np.ascontiguousarray(clean_canvas[top:top + h, left:left + w])
    if sha256_bytes(win.tobytes()) != rec["resized_pixel_sha256"]:
        _fail("source_pixels_mismatch",
              f"{stem}: the clean canvas window ({h}x{w} at {top},{left}) does not match the "
              "source digest the cache recorded; the item was not read")
    outside = np.ones((SIZE, SIZE), dtype=bool)
    outside[top:top + h, left:left + w] = False
    if not np.all(clean_canvas[outside] == np.asarray(IMAGENET_MEAN_8BIT, dtype=np.uint8)):
        _fail("canvas_window_mismatch",
              f"{stem}: the clean canvas holds image pixels outside the recorded {h}x{w} window")
    item = cell.load(stem)
    if not isinstance(item, np.ndarray) or item.dtype != np.uint8 or item.shape != (h, w, 3):
        _fail("corrupted_shape_mismatch",
              f"{stem}: the cached item is {getattr(item, 'dtype', type(item).__name__)} "
              f"{tuple(getattr(item, 'shape', ()))}, the resized clean image is ({h}, {w}, 3)")
    canvas = clean_canvas.copy()
    canvas[top:top + h, left:left + w] = item
    return canvas


# --------------------------------------------------------------------------------------------------
# the dataset wrapper
# --------------------------------------------------------------------------------------------------
class PlantSegCorruptedEvalDataset(Dataset):
    """Evaluation-only wrapper for one corrupted cell. Attaches identity inside `__getitem__`.

    Counts every verified source digest and every verified item, so the caller can require both
    to equal the scored rows (`num_workers` is 0, so the counts are this process's).
    """

    def __init__(self, split: str, source_indices: Sequence[int], cell):
        if split not in CORRUPTED_SPLITS:           # the clean wrapper's refusals, reused
            raise AdapterError(f"a corrupted condition is scored on {CORRUPTED_SPLITS}, got "
                               f"{split!r}")
        self.base = PlantSegDataset(split)          # unmodified
        self.split = split
        n = len(self.base)
        idx = [int(i) for i in source_indices]
        if not idx:
            raise AdapterError("source_indices selects zero samples")
        if len(set(idx)) != len(idx):
            raise AdapterError("source_indices contains duplicates")
        for i in idx:
            if not 0 <= i < n:
                raise AdapterError(f"source index {i} out of range for split={split} (n={n})")
        self.source_indices = sorted(idx)           # ascending stable external index
        self.cell = cell
        self.items_verified = 0
        self.source_digests_verified = 0

    def __len__(self) -> int:
        return len(self.source_indices)

    def stem_for(self, source_index: int) -> str:
        return self.base.pairs[source_index][0].stem

    def __getitem__(self, i: int) -> dict:
        src = self.source_indices[i]
        img_path, mask_path = self.base.pairs[src]
        stem = self.stem_for(src)                   # resolved HERE, before the sample leaves
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            clean, target_canvas = core_preprocess(im, mk)
        canvas = corrupted_canvas(stem, src, clean, self.cell)
        self.source_digests_verified += 1
        self.items_verified += 1
        image, target = finalize(canvas, target_canvas)
        return {"image": image, "target": target, "image_id": stem, "clean_image_id": stem,
                "manifest_index": src}


# --------------------------------------------------------------------------------------------------
# the cell record (TL interface section 7)
# --------------------------------------------------------------------------------------------------
def record_path(out_dir) -> Path:
    """`<out_dir>.corruption.json`, the artifact directory's sibling."""
    out_dir = Path(out_dir)
    return out_dir.with_name(out_dir.name + RECORD_SUFFIX)


def artifact_file_hashes(out_dir) -> dict:
    return {name: sha256_file(Path(out_dir) / name) for name in ARTIFACT_FOUR_FILES}


def build_cell_record(*, out_dir, run_id: str, stage: str, condition: dict, role: str,
                      manifest: dict, cell_sha256: str, verification_record_sha256: str | None,
                      rows: int, items_verified: int, source_digests_verified: int,
                      evaluator_commit: str, started_utc: str, split: str) -> dict:
    if not int(items_verified) == int(source_digests_verified) == int(rows):
        _fail("verification_count_mismatch",
              f"{items_verified} items and {source_digests_verified} source digests verified "
              f"for {rows} rows")
    record = {
        "schema": CELL_RECORD_SCHEMA,
        "artifact_dir": Path(out_dir).name,
        "artifact_sha256": artifact_file_hashes(out_dir),
        "run_id": run_id,
        "stage": stage,
        "condition": {"type": condition["type"], "name": condition["name"],
                      "severity": condition["severity"]},
        "role": role,
        "cache": {"canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
                  "cell_sha256": cell_sha256,
                  "inferential_cells_sha256": manifest["inferential_cells_sha256"],
                  "split": split, "status": manifest["status"],
                  "source_image_count": manifest["source_image_count"],
                  "verification_record_sha256": verification_record_sha256},
        "items_verified": int(items_verified),
        "source_digests_verified": int(source_digests_verified),
        "evaluator_script": EVALUATOR_SCRIPT,
        "evaluator_commit": evaluator_commit,
        "started_utc": started_utc,
        "written_utc": utc_iso(),
    }
    return record


def record_bytes(record: dict) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def write_cell_record(path, record: dict) -> Path:
    """Atomic: temporary sibling, fsync, os.replace. An existing record is refused."""
    path = Path(path)
    if path.exists():
        _fail("record_exists", f"refusing to overwrite the cell record {path}")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(record_bytes(record))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_cell_record(path) -> dict:
    """Strict JSON (B-2's reader: no NaN, no duplicate key), the schema and exactly the TL section 7
    keys."""
    path = Path(path)
    try:
        record = _strict_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _fail("record_unreadable", f"{path}: {type(e).__name__}: {e}")
    if not isinstance(record, dict) or record.get("schema") != CELL_RECORD_SCHEMA:
        _fail("record_schema", f"{path} is not a {CELL_RECORD_SCHEMA} record")
    if tuple(sorted(record)) != tuple(sorted(CELL_RECORD_KEYS)) or \
            not isinstance(record.get("cache"), dict) or \
            tuple(sorted(record["cache"])) != tuple(sorted(CACHE_KEYS)):
        _fail("record_schema", f"{path} does not hold exactly the record keys")
    return record


def verify_cell_record(out_dir, record: dict) -> None:
    """The record names this artifact directory and the sha256 of its four files as they are now."""
    out_dir = Path(out_dir)
    if record.get("artifact_dir") != out_dir.name:
        _fail("record_artifact_mismatch",
              f"the record names {record.get('artifact_dir')!r}, not {out_dir.name!r}")
    if record.get("artifact_sha256") != artifact_file_hashes(out_dir):
        _fail("record_artifact_mismatch",
              f"the four files of {out_dir.name} no longer match their record")


__all__ = ["ARTIFACT_FOUR_FILES", "CACHE_KEYS", "CELL_RECORD_KEYS", "CELL_RECORD_SCHEMA", "CacheCell",
           "CorruptedEvalError", "EVALUATOR_SCRIPT", "OUT_PREFIX", "PlantSegCorruptedEvalDataset",
           "RECORD_SUFFIX", "artifact_file_hashes", "build_cell_record", "clean_dir_name",
           "corrupted_canvas", "corrupted_dir_name", "read_cell_record", "record_bytes",
           "record_path", "sha256_file", "utc_iso", "verify_cell_record", "window",
           "write_cell_record"]
