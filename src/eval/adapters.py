"""Real-PlantSeg adapters for the proven A2a evaluation core (A2b).

Supplies the three things the A2a core deliberately left out, WITHOUT touching its semantics:
an identity-carrying dataset wrapper, a collate producing `EvalBatch`, and an expected-manifest
builder. Metric arithmetic, artifact fields, identity rules and the official-run guard all remain
exactly as frozen in `evaluate.py` / `artifacts.py`.

Identity contract (docs/EVALUATION_CONTRACT.md section 6):
  * `PlantSegDataset` is NEVER modified.
  * The canonical stem is resolved INSIDE `__getitem__` and attached to the sample before it
    leaves the dataset. Nothing is recovered from `dataset.pairs` after batching, inferred from
    batch position, or parsed back out of another id.
  * `clean_image_id` is carried explicitly. For the clean condition it equals `image_id`; a future
    corruption adapter sets `image_id` to a suffixed form and keeps `clean_image_id` canonical.

Capping rule (A2b): a capped run selects explicit SOURCE INDICES *before* the DataLoader exists,
builds the adapter over exactly those, and builds the expected manifest from exactly those same
indices. There is no "make a full loader and stop early" path -- that would let the evaluator
receive an 846-row manifest while only a handful of rows were produced.

`manifest_index` is the STABLE EXTERNAL index: the row's position in the full canonical
name-sorted split, which is preserved even when the subset is non-contiguous.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.dataset import PlantSegDataset
from .evaluate import EvalBatch, ManifestEntry

REPO = Path(__file__).resolve().parents[2]
CLASS_MAP_PATH = REPO / "configs" / "plantseg_class_map.json"

# Pinned authoritative class map (A2b-0). Any drift must fail loudly.
CLASS_MAP_SEMANTIC_SHA256 = "d14182423b6f176f940cada979adb364701fe186be091655ce38a80ce791a729"

# Frozen real-dataset identity (contract section 5.3 / configs/data.py).
DATASET_NAME = "PlantSeg"
DATASET_DOI = "10.5281/zenodo.17719108"
PREPROCESS_PROTOCOL = "core_preprocess/1.0.0"   # src/data/transforms.py core_preprocess (val/test)


class AdapterError(RuntimeError):
    """Configuration/identity failure in the real-data adapter layer."""


# --------------------------------------------------------------------------------------------------
# class map
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassMap:
    entries: tuple[dict, ...]
    semantic_sha256: str
    source_commit_sha: str
    immutable_source_url: str


def load_class_map(path: Path = CLASS_MAP_PATH) -> ClassMap:
    """Load the AUTHORITATIVE vendored map and verify it against the pinned semantic hash.

    The ordered `entries` list is passed straight through to the A2a class-map hasher. It is never
    re-derived from Metadata.csv, re-ordered, renamed, or majority-voted, and class 0's official
    empty name is preserved verbatim.
    """
    import hashlib

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = doc.get("entries", [])
    if len(entries) != 116:
        raise AdapterError(f"class map must have 116 entries, got {len(entries)}")
    if [e["class_id"] for e in entries] != list(range(116)):
        raise AdapterError("class map ids are not exactly 0..115 in order")
    if entries[0]["name"] != "" or entries[0]["role"] != "background_or_non_disease":
        raise AdapterError(
            f"class 0 must keep the official empty name and background role; got "
            f"{entries[0]['name']!r} / {entries[0]['role']!r}")
    if not all(e["role"] == "disease" and len(e["name"]) > 0 for e in entries[1:]):
        raise AdapterError("classes 1..115 must all be non-empty 'disease' entries")

    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    sem = hashlib.sha256(canonical).hexdigest()
    if sem != CLASS_MAP_SEMANTIC_SHA256:
        raise AdapterError(
            f"class-map semantic hash mismatch: {sem} != pinned {CLASS_MAP_SEMANTIC_SHA256}. "
            "Refusing to evaluate against an unverified class map.")
    src = doc.get("source", {})
    return ClassMap(tuple(entries), sem, src.get("source_commit_sha", ""),
                    src.get("immutable_source_url", ""))


# --------------------------------------------------------------------------------------------------
# identity-carrying dataset wrapper
# --------------------------------------------------------------------------------------------------
class PlantSegEvalDataset(Dataset):
    """Evaluation-only wrapper. Attaches identity inside `__getitem__`; never mutates the base."""

    def __init__(self, split: str, source_indices: Sequence[int] | None = None):
        self.base = PlantSegDataset(split)          # unmodified
        self.split = split
        n = len(self.base)
        if source_indices is None:
            idx = list(range(n))
        else:
            idx = [int(i) for i in source_indices]
        if not idx:
            raise AdapterError("source_indices selects zero samples")
        if len(set(idx)) != len(idx):
            raise AdapterError("source_indices contains duplicates")
        for i in idx:
            if not (0 <= i < n):
                raise AdapterError(f"source index {i} out of range for split={split} (n={n})")
        # canonical order: ascending stable external index
        self.source_indices = sorted(idx)

    def __len__(self) -> int:
        return len(self.source_indices)

    def stem_for(self, source_index: int) -> str:
        return self.base.pairs[source_index][0].stem

    def __getitem__(self, i: int) -> dict:
        src = self.source_indices[i]
        image, target = self.base[src]
        stem = self.stem_for(src)                   # resolved HERE, before the sample leaves
        return {
            "image": image,
            "target": target,
            "image_id": stem,
            "clean_image_id": stem,                 # clean condition: identical by definition
            "manifest_index": src,                  # STABLE EXTERNAL index in the full split
        }


def eval_collate(samples: Sequence[dict]) -> EvalBatch:
    """Collate into the proven A2a `EvalBatch`, carrying both identities."""
    return EvalBatch(
        images=torch.stack([s["image"] for s in samples]),
        targets=torch.stack([s["target"] for s in samples]),
        image_ids=[s["image_id"] for s in samples],
        clean_image_ids=[s["clean_image_id"] for s in samples],
        manifest_indices=[int(s["manifest_index"]) for s in samples],
    )


def list_split_stems(split: str) -> list[str]:
    """Canonical name-sorted stems for a split.

    FILENAME LISTING ONLY -- no image or mask is opened and no `PlantSegDataset` is constructed,
    so the expected manifest can be built BEFORE the request preflight without touching the
    dataset adapter. Mirrors `PlantSegDataset`'s ordering and count guard; any drift between the
    two is caught by the A2a core, which compares every observed sample against this manifest.
    """
    from configs.data import DATA

    root = Path(DATA["root"])
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        raise AdapterError(f"split image directory not found: {img_dir}")
    stems = [p.stem for p in sorted(
        (p for p in img_dir.iterdir()
         if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda p: p.name)]
    expected = DATA.get("splits", {}).get("sizes", {}).get(split)
    if expected is not None and len(stems) != expected:
        raise AdapterError(
            f"split={split} has {len(stems)} images but the locked count is {expected}")
    return stems


def build_expected_manifest_for(split: str, source_indices: Sequence[int]) -> list[ManifestEntry]:
    """Expected manifest for EXACTLY these source indices, built from a filename listing only.

    Runs before the request preflight; constructs no dataset adapter and opens no image or mask.
    """
    stems = list_split_stems(split)
    out = []
    for src in sorted(int(i) for i in source_indices):
        if not (0 <= src < len(stems)):
            raise AdapterError(f"source index {src} out of range for split={split}")
        out.append(ManifestEntry(manifest_index=src, image_id=stems[src],
                                 clean_image_id=stems[src]))
    return out


def build_expected_manifest(adapter: PlantSegEvalDataset) -> list[ManifestEntry]:
    """Expected manifest from an already-built adapter -- used to cross-check `..._for()`."""
    return [ManifestEntry(manifest_index=src,
                          image_id=adapter.stem_for(src),
                          clean_image_id=adapter.stem_for(src))
            for src in adapter.source_indices]


def build_eval_loader(adapter: PlantSegEvalDataset, batch_size: int,
                      num_workers: int = 0) -> DataLoader:
    """DataLoader over the ALREADY-CAPPED adapter. shuffle=False; identity does not depend on it."""
    return DataLoader(adapter, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=eval_collate, drop_last=False)


def deterministic_subset(n_total: int, n_samples: int) -> list[int]:
    """Evenly spaced deterministic source indices -- reproducible and intentionally NON-contiguous,
    so the NPZ row-position mapping is exercised against sparse external manifest indices."""
    if n_samples > n_total:
        raise AdapterError(f"requested {n_samples} samples but the split has {n_total}")
    step = n_total // n_samples
    return [i * step for i in range(n_samples)]
