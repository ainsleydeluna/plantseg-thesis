"""Lossless, deterministic, shared corruption cache: generation and verified consumption.

CACHE POSITION IN THE PIPELINE (frozen):

    clean image -> aspect-ratio-preserving resize -> CORRUPT (uint8 RGB) -> [CACHE HERE]
                -> mean-value padding -> tensor -> ImageNet normalization -> inference

Only the corrupted RESIZED uint8 RGB array is cached. Normalized tensors, padded model inputs and
predictions are never cached, so every stage pads and normalizes identically from one frozen pixel
array.

LOSSLESS ON PURPOSE. Entries are stored as `.npy` uint8 arrays, never re-encoded as JPEG: a second
uncontrolled compression would contaminate gaussian_noise / motion_blur / brightness / fog and would
confound the registered `jpeg_compression` corruption itself.

GENERATION AND CONSUMPTION ARE SEPARATE. The evaluator only ever *reads* a verified cache; it never
regenerates a missing or mismatched entry, which is what guarantees the teacher and E1-E7 score the
identical stochastic realisation.

TORCH-FREE BY CONTRACT. This module is thesis-owned infrastructure (the vendored third-party
numerics live in `src.vendor.imagecorruptions`), and neither side may import `src.data`, `torch` or
`torchvision`. Cache generation is a pure NumPy/Pillow/scikit-image job, and `src.data.__init__`
imports the torch dataloader, which on the Anaconda development box makes scikit-image abort with
`OMP: Error #15` once torch's OpenMP runtime has initialised first. `scripts/smoke_corruption_vendor.py`
asserts torch is absent from `sys.modules` before and after the corruption/cache work.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from src.vendor.imagecorruptions import (CORRUPTION_IDS, GENERATED_SEVERITIES, MASTER_SEED,
                                         SEED_POLICY_ID, apply_corruption, item_seed)
from src.vendor.imagecorruptions.provenance import as_dict as vendor_provenance
from src.vendor.imagecorruptions.provenance import compute_runtime_sha256

CACHE_SCHEMA = "plantseg-corruption-cache/1.0.0"
OFFICIAL_SPLIT = "test"
OFFICIAL_EXPECTED_IMAGES = 1561          # checked only when the REAL cache is built
MANIFEST_NAME = "corruption_cache_manifest.json"


class CorruptionCacheError(RuntimeError):
    """A rejected cache generation or a failed cache verification."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str):
    raise CorruptionCacheError(code, message)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def runtime_versions() -> dict:
    """Actual generation-environment versions — always recorded, never assumed."""
    import PIL
    import skimage
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pillow": PIL.__version__, "scikit_image": skimage.__version__,
            "platform": platform.platform()}


def relative_path(image_id: str, corruption_id: str, severity: int) -> str:
    return f"{corruption_id}/s{severity}/{image_id}.npy"


def write_entry(cache_root: Path, image_id: str, corruption_id: str, severity: int,
                array: np.ndarray, seed: int) -> dict:
    """Persist one corrupted array losslessly and return its manifest entry."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        _fail("entry_not_uint8_rgb", f"cache entries are uint8 H*W*3; got {array.dtype}/{array.shape}")
    rel = relative_path(image_id, corruption_id, severity)
    path = cache_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {
        "image_id": image_id, "corruption_id": corruption_id, "severity": int(severity),
        "master_seed": MASTER_SEED, "item_seed": int(seed),
        "cache_relative_path": rel, "shape": list(array.shape), "dtype": "uint8",
        "cached_file_sha256": _sha256_bytes(path.read_bytes()),
        "corrupted_pixel_sha256": _sha256_bytes(array.tobytes()),
        "patched_vendor_runtime_sha256": compute_runtime_sha256(),
    }


def generate(cache_root: Path, images: dict[str, np.ndarray], *,
             corruptions=CORRUPTION_IDS, severities=GENERATED_SEVERITIES,
             split: str = OFFICIAL_SPLIT, expected_images: int | None = None) -> dict:
    """Build the cache grid for the supplied RESIZED uint8 images. Severity 5 is never generated."""
    if any(s not in GENERATED_SEVERITIES for s in severities):
        _fail("severity_not_generated", f"only severities {GENERATED_SEVERITIES} are generated")
    unknown = [c for c in corruptions if c not in CORRUPTION_IDS]
    if unknown:
        _fail("unknown_corruption", f"unregistered corruptions: {unknown}")

    cache_root = Path(cache_root)
    entries = []
    for image_id in sorted(images):                     # canonical order
        for corruption_id in corruptions:
            for severity in severities:
                arr, seed = apply_corruption(images[image_id], corruption_id, severity,
                                             image_id=image_id)
                entries.append(write_entry(cache_root, image_id, corruption_id, severity,
                                           arr, seed))

    manifest = {
        "schema": CACHE_SCHEMA, "dataset": "PlantSeg", "split": split,
        "official_expected_images": OFFICIAL_EXPECTED_IMAGES,
        "source_image_count": len(images),
        "corruption_order": list(corruptions), "severities": list(severities),
        "seed_policy_id": SEED_POLICY_ID, "master_seed": MASTER_SEED,
        "vendor": vendor_provenance(), "versions": runtime_versions(),
        "entries": entries,
    }
    if expected_images is not None and len(images) != expected_images:
        _fail("image_count_mismatch",
              f"expected {expected_images} source images, got {len(images)}")
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    (cache_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                            encoding="utf-8")
    return manifest


def canonical_manifest_sha256(manifest: dict) -> str:
    """Digest over the manifest WITHOUT its own checksum field — never self-referential."""
    body = {k: v for k, v in manifest.items() if k != "canonical_manifest_sha256"}
    return _sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def load_manifest(cache_root: Path) -> dict:
    path = Path(cache_root) / MANIFEST_NAME
    if not path.is_file():
        _fail("manifest_missing", f"cache manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CACHE_SCHEMA:
        _fail("manifest_schema", f"unknown cache schema {manifest.get('schema')!r}")
    if manifest.get("canonical_manifest_sha256") != canonical_manifest_sha256(manifest):
        _fail("manifest_checksum_mismatch", "cache manifest checksum does not match its contents")
    if manifest.get("seed_policy_id") != SEED_POLICY_ID:
        _fail("seed_policy_mismatch",
              f"cache was built under seed policy {manifest.get('seed_policy_id')!r}, "
              f"this build uses {SEED_POLICY_ID!r}")
    recorded = manifest.get("vendor", {}).get("patched_vendor_runtime_sha256")
    if recorded != compute_runtime_sha256():
        _fail("vendor_checksum_mismatch",
              "cache was generated by a different vendored corruption implementation")
    return manifest


def load_cached(cache_root: Path, manifest: dict, image_id: str, corruption_id: str,
                severity: int) -> np.ndarray:
    """Load one cached corrupted array with full verification. NEVER regenerates on failure."""
    key = (image_id, corruption_id, int(severity))
    entry = next((e for e in manifest["entries"]
                  if (e["image_id"], e["corruption_id"], e["severity"]) == key), None)
    if entry is None:
        _fail("cache_entry_missing",
              f"no cached entry for {key}; the evaluator never regenerates corruptions")
    path = Path(cache_root) / entry["cache_relative_path"]
    if not path.is_file():
        _fail("cache_file_missing", f"cached file missing: {path}")
    if _sha256_bytes(path.read_bytes()) != entry["cached_file_sha256"]:
        _fail("cache_file_hash_mismatch", f"cached file has been modified: {path}")
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.uint8 or list(arr.shape) != entry["shape"]:
        _fail("cache_entry_corrupt", f"cached array is not the recorded uint8 {entry['shape']}")
    if _sha256_bytes(arr.tobytes()) != entry["corrupted_pixel_sha256"]:
        _fail("cache_pixel_hash_mismatch", f"cached pixels do not match their recorded digest: {path}")
    return arr
