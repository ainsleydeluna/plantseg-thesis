"""Torch-free source reader for the corruption cache: split listing, EXIF and resize (L-CORR).

The cache holds the corruption of each image AS core_preprocess RESIZES IT (src/data/transforms.py
:52-58, frozen): EXIF transpose, RGB, aspect-preserving resize of the long side to 512 with PIL
BILINEAR (Python `round`, half to even, and `max(1, ...)`), uint8. `src.data` imports torch, which
the cache generator may not load, so this module repeats that image path with NumPy and Pillow
only. scripts/verify_corruption_cache.py proves the copy equal to the real core_preprocess on every
cached image (each source digest is the canvas window), and scripts/smoke_corruption_verify.py does
so on synthetic EXIF, mode and rounding cases.

Listing follows src/eval/adapters.list_split_stems (:224-246) and PlantSegDataset: the files of
images/<split> whose suffix is .jpg or .jpeg in any case, sorted by the file name with its
extension (`p.name`), at the count locked in configs/data.py SPLIT_SIZES. manifest_index is the
position in that list.

A stem is data only (LCC2). Items are stored at <corruption>/s<severity>/<manifest_index:05d>.npy
(src/corruption_cache.item_relative_path, LCC1), so no created path derives from a stem, and no
stem refuses a split: neither "test" in any case nor any character. The former gate
source_name_forbidden refused such stems only because item paths then embedded them; it went with
that rule. Every created path is still tested (LC6): the cache root, the journal and record paths,
and the TEST root corr_cache_tday; the split directories are checked below.

Split surfaces (M11):
  * val: src/data/isolation.py is loaded BY FILE PATH (never through the src.data package, which
    imports torch) and assert_trainval_only_root runs on the data root before any image is read;
  * test: only <root>/images/test is opened -- no annotation, no other split.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
ISOLATION_PATH = REPO / "src" / "data" / "isolation.py"

SIZE = 512
IMAGE_SUFFIXES = (".jpg", ".jpeg")
EXIF_ORIENTATION = 0x0112
SOURCE_RESIZE = "core_preprocess/1.0.0 image path"
SPLITS = ("val", "test")


class CorruptionSourceError(RuntimeError):
    """A refused data root, split listing or source image. `code` names the gate."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def __reduce__(self):                    # a refusal raised in a worker keeps its code
        return (self.__class__, (self.code, str(self)))


def _fail(code: str, message: str):
    raise CorruptionSourceError(code, message)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def split_sizes() -> dict:
    from configs.data import SPLIT_SIZES
    return dict(SPLIT_SIZES)


def require_split(split: str) -> None:
    if split not in SPLITS:
        _fail("unknown_split", f"split must be one of {SPLITS}, got {split!r}")


def require_data_root(root: Path) -> None:
    if not Path(root).is_dir():
        _fail("data_root_missing", f"data root does not resolve: {root}")


def require_split_dir(img_dir: Path) -> None:
    if not Path(img_dir).is_dir():
        _fail("split_dir_missing", f"missing split directory: {img_dir}")


def require_split_count(n: int, split: str) -> None:
    expected = split_sizes()[split]
    if n != expected:
        _fail("split_count_mismatch",
              f"split {split} has {n} images but the locked count is {expected}")


def load_isolation():
    """src/data/isolation.py loaded from its file, bypassing the src.data package (torch)."""
    spec = importlib.util.spec_from_file_location("plantseg_trainval_isolation", ISOLATION_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_isolation(root: Path) -> dict:
    """The M11 TRAIN/VAL-only check of src/data/isolation.py, its refusals kept by code."""
    iso = load_isolation()
    try:
        return iso.assert_trainval_only_root(root)
    except iso.TrainValIsolationError as e:
        _fail(e.code, str(e))


def check_split_surface(data_root, split: str) -> dict:
    """The data-root gate, run before any image is read.

    val: the M11 TRAIN/VAL-only check (TEST surfaces absent by existence test, TRAIN and VAL
    counted by name). test: the root and <root>/images/test must be directories; nothing else
    under the root is touched.
    """
    require_split(split)
    root = Path(data_root)
    if split == "val":
        return {"split": "val", "isolation": run_isolation(root)}
    require_data_root(root)
    require_split_dir(root / "images" / "test")
    return {"split": "test", "opened": "images/test only"}


@dataclass(frozen=True)
class Source:
    manifest_index: int
    image_id: str
    path: Path


def split_files(img_dir: Path) -> list[Path]:
    """The image files of one split directory, sorted by the file name (`p.name`)."""
    return sorted((p for p in Path(img_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name)


def list_split(data_root, split: str) -> list[Source]:
    """The split's sources in manifest order. Call check_split_surface first."""
    require_split(split)
    img_dir = Path(data_root) / "images" / split
    require_split_dir(img_dir)
    files = split_files(img_dir)
    require_split_count(len(files), split)
    return [Source(i, p.stem, p) for i, p in enumerate(files)]


def deterministic_subset(n_total: int, n_samples: int) -> list[int]:
    """Evenly spaced source indices, as src/eval/adapters.deterministic_subset (:300-306)."""
    if not 0 < n_samples <= n_total:
        _fail("max_images_out_of_range", f"--max-images must lie in 1..{n_total}, got {n_samples}")
    step = n_total // n_samples
    return [i * step for i in range(n_samples)]


def resized_size(w: int, h: int) -> tuple[int, int]:
    """(new_w, new_h) by transforms._resize_keep_ratio's rule (:32-38)."""
    if w >= h:
        return SIZE, max(1, round(h * SIZE / w))
    return max(1, round(w * SIZE / h)), SIZE


@dataclass(frozen=True)
class SourceImage:
    record: dict            # the manifest `sources` entry
    pixels: np.ndarray      # resized uint8 H*W*3, before padding


def _decode(raw: bytes, name: str):
    """(EXIF orientation, mode, stored (w, h), transposed RGB image) of one source file."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            orientation = im.getexif().get(EXIF_ORIENTATION)
            mode, size = im.mode, im.size
            rgb = ImageOps.exif_transpose(im).convert("RGB")
    except (OSError, ValueError, SyntaxError) as e:
        _fail("source_unreadable", f"{name}: {type(e).__name__}: {e}")
    return orientation, mode, size, rgb


def _read_bytes(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as e:
        _fail("source_unreadable", f"{Path(path).name}: {type(e).__name__}: {e}")


def orientation_value(v):
    """The EXIF orientation as recorded: an int as is, anything else (a malformed BYTE, ASCII or
    UNDEFINED entry, which ImageOps.exif_transpose does not apply) by its repr, never coerced."""
    if v is None or (isinstance(v, int) and not isinstance(v, bool)):
        return v
    return repr(v)


def read_source(src: Source) -> SourceImage:
    """Read one source file and resize it as core_preprocess's image path does."""
    raw = _read_bytes(src.path)
    orientation, mode, (w, h), rgb = _decode(raw, Path(src.path).name)
    nw, nh = resized_size(*rgb.size)
    pixels = np.ascontiguousarray(np.asarray(rgb.resize((nw, nh), Image.BILINEAR),
                                             dtype=np.uint8))
    record = {"manifest_index": int(src.manifest_index), "image_id": src.image_id,
              "file_name": Path(src.path).name, "source_file_sha256": _sha256(raw),
              "exif_orientation": orientation_value(orientation),
              "source_mode": mode, "original_size": [int(w), int(h)],
              "resized_shape": [int(pixels.shape[0]), int(pixels.shape[1])],
              "resized_pixel_sha256": _sha256(pixels.tobytes())}
    return SourceImage(record, pixels)


def array_source(manifest_index: int, image_id: str, pixels: np.ndarray) -> SourceImage:
    """A source record for an in-memory, already-resized array (synthetic smoke caches only)."""
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    record = {"manifest_index": int(manifest_index), "image_id": image_id, "file_name": None,
              "source_file_sha256": None, "exif_orientation": None, "source_mode": "array",
              "original_size": [int(arr.shape[1]), int(arr.shape[0])],
              "resized_shape": [int(arr.shape[0]), int(arr.shape[1])],
              "resized_pixel_sha256": _sha256(arr.tobytes())}
    return SourceImage(record, arr)


__all__ = ["CorruptionSourceError", "IMAGE_SUFFIXES", "SIZE",
           "SOURCE_RESIZE", "SPLITS", "Source", "SourceImage", "array_source",
           "check_split_surface", "deterministic_subset",
           "list_split", "load_isolation", "orientation_value", "read_source", "require_data_root",
           "require_split", "require_split_count", "require_split_dir",
           "resized_size", "run_isolation", "split_files", "split_sizes"]
