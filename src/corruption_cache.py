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

MANIFEST plantseg-corruption-cache/2.0.0 (L-CORR; the TL interface sections 3-6). One root holds
the manifest `corruption_cache_manifest.json`, the items
`<corruption>/s<severity>/<manifest_index:05d>.npy`, the generation journal `journal/entries.jsonl`
with one `journal/run_<UTC>.json` per invocation, verification records `verification/<UTC>.json`
and repair records `repair/<UTC>.json`. Only the manifest is a contract file. It is written once,
atomically, after every expected item has verified, so a crashed run leaves no manifest and no
consumer can read a partial cache.

ITEM PATHS COME FROM THE INDEX, NEVER FROM A STEM (LCC1). An item's path is
<corruption>/s<severity>/<manifest_index>.npy, its source's manifest_index written as five decimal
digits (f"{i:05d}"; VAL has 846 sources and TEST 1,561). item_relative_path is the one rule: the
in-memory generate() and the workers write through it (write_entry), resume and repair find their
items through it (resume_keeps, repair_target), and load_manifest, which the verifier, repair and
every consumer call, refuses an entry that is not at its index path (item_path_mismatch). Each
entry's `cache_relative_path` names its file. `image_id` stays in entries, sources and records as
data only, so no created file or folder name derives from a stem, and no stem can bring "test" or
a character the TL interface section 2 forbids into a created path.

  canonical_manifest_sha256 = sha256 of canonical_json_bytes(the manifest without
                              canonical_manifest_sha256, versions and generation)
  cell_sha256               = sha256 of canonical_json_bytes(the cell's entries, in ascending
                              manifest_index)
  inferential_cells_sha256  = sha256 of canonical_json_bytes(the list of the inferential cells'
                              cell_sha256 values, in protocol order: 15 for a full grid)

canonical_json_bytes is src/eval/artifacts.py's rule (:93-96), repeated here because src.eval
imports torch: sorted keys, compact separators, ensure_ascii=False, allow_nan=False.

`versions` (the runtime record, platform included) and `generation` (invocations, workers, times,
commit, worktree state) sit outside the checksum, so two byte-identical caches built on two hosts,
at any worker count or after a resume, carry one checksum. That is sound only because an official
or provisional cache refuses any stack whose versions differ from the registered pins
(require_official_dependency_environment): for those caches `versions` equals `dependency_pins`
in every field the bytes depend on (numpy, Pillow and scikit-image exactly, Python at
major.minor). A smoke cache may be built on any stack and claims nothing.
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from src import corruption_roles
from src.corruption_source import (SOURCE_RESIZE, CorruptionSourceError, Source, array_source,
                                   read_source)
from src.vendor.imagecorruptions import (CORRUPTION_IDS, MASTER_SEED, MIN_SIDE, SEED_POLICY_ID,
                                         SEVERITIES_BY_CORRUPTION, apply_corruption)
from src.vendor.imagecorruptions.provenance import VENDOR_RUNTIME_SHA256
from src.vendor.imagecorruptions.provenance import as_dict as vendor_provenance
from src.vendor.imagecorruptions.provenance import compute_runtime_sha256

REPO = Path(__file__).resolve().parents[1]

CACHE_SCHEMA = "plantseg-corruption-cache/2.0.0"
JOURNAL_SCHEMA = "plantseg-corruption-journal/1.0.0"
PIN_SCHEMA = "plantseg-corruption-generator-pin/1.0.0"
REPAIR_SCHEMA = "plantseg-corruption-repair/1.0.0"
OFFICIAL_SPLIT = "test"
OFFICIAL_EXPECTED_IMAGES = 1561          # checked only when the REAL cache is built
MANIFEST_NAME = "corruption_cache_manifest.json"
JOURNAL_DIR, JOURNAL_NAME = "journal", "entries.jsonl"
CACHE_STATUSES = ("smoke", "provisional", "official")
DATASET_NAME, DATASET_DOI = "PlantSeg", "10.5281/zenodo.17719108"
#: The TEST cache root's last component (TL interface section 2).
TDAY_CACHE_ROOT_NAME = "corr_cache_tday"
GENERATOR_SCRIPT = "scripts/build_corruption_cache.py"
PROTOCOL_REL = "configs/corruption_protocol.json"
#: The student image of record (DL-87), in which the cache is generated and verified (TL-1).
STUDENT_IMAGE_DIGEST = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
#: EVALUATION_CONTRACT section 7.1's governed paths, as a git pathspec.
GOVERNED_PATHSPEC = ("src", "configs", "scripts", "requirements*", "docs/EVALUATION_CONTRACT.md",
                     "docs/IMPLEMENTATION_CONTRACT.md")
STORAGE = {"format": "npy", "dtype": "uint8", "layout": "HxWx3",
           "relative_path": "<corruption>/s<severity>/<manifest_index:05d>.npy"}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

# ---------------------------------------------------------------- official dependency pins
# The corrupted BYTES depend on these three libraries, so reproducibility needs exact versions:
#   numpy         -> every corruption
#   Pillow        -> jpeg_compression (the JPEG encoder)
#   scikit-image  -> brightness (rgb2hsv / hsv2rgb)
# Values are TAKEN FROM `requirements.lock`, the repository's pinned-stack manifest; they are
# recorded here so cache generation can verify them at runtime, not re-decided here.
#
# Verified against primary PyPI metadata for the recorded official Python 3.11:
#   Pillow 12.3.0        requires_python ">=3.10"; cp311 wheels published; released 2026-07-01 as a
#                        security-maintenance release over 12.2.0 (CVE-2026-55798 and the
#                        CVE-2026-54059/54060/55379/55380 decompression-bomb set, plus out-of-bounds
#                        fixes in TGA/RankFilter/Image.paste/ImageCmsTransform). Its release notes
#                        document NO JPEG encode/decode change, and that was confirmed executably
#                        rather than assumed: the registered jpeg_compression grid is byte-identical
#                        under 12.2.0 and 12.3.0, so adopting it changes no corruption output.
#   scikit-image 0.23.2  requires_python ">=3.10"; numpy>=1.23, scipy>=1.9, pillow>=9.1;
#                        cp311 wheels published. (0.23.x dropped the hard PyWavelets dependency,
#                        which is why the lock contains no PyWavelets entry.)
OFFICIAL_PYTHON = "3.11"                 # recorded in docs/reference/context.md + the contracts
OFFICIAL_NUMPY = "1.26.4"
OFFICIAL_PILLOW = "12.3.0"
OFFICIAL_SCIKIT_IMAGE = "0.23.2"
OFFICIAL_DEPENDENCY_SOURCE = "requirements.lock"

# TWO PHASES, DELIBERATELY SEPARATE. Writing a version number down is not the same as having run
# the corruption verification under that environment, and one Boolean cannot honestly say both.
DEPENDENCY_VERSIONS_REGISTERED = True

# HISTORICAL RECORD, NOT A STATEMENT ABOUT THE CURRENT PROCESS. True means "this exact corruption
# dependency stack has been installed and the deterministic corruption verification passed under
# it". Whether the *running* interpreter matches is a separate question, answered independently by
# `runtime_pin_mismatches()`; both must hold before the cache may be generated.
OFFICIAL_DEPENDENCY_ENVIRONMENT_VALIDATED = True
VALIDATION_EVIDENCE = {
    "validated_on_python": "3.11.15",
    "numpy": "1.26.4", "pillow": "12.3.0", "scikit_image": "0.23.2", "scipy": "1.11.4",
    "check": "scripts/smoke_corruption_vendor.py",
    "reference_equivalence": "40/40 byte-identical, zero tolerance "
                             "(5 corruptions x severities 1-4 x {square, non-square})",
    "isolated_environment": True,
}

# SCOPE LIMIT — the two validation claims are NOT interchangeable. Only the corruption dependency
# closure (python/numpy/Pillow/scikit-image and scikit-image's own runtime deps) was reproduced and
# executed. torch 2.1.0+cu121, torchvision 0.16.0, mmsegmentation 1.2.2, mmcv 2.1.0, fvcore, CUDA
# and the RunPod container were NOT, so the full official experiment environment stays unvalidated.
VALIDATION_SCOPE = "corruption_dependency_closure_only"
FULL_EXPERIMENT_ENVIRONMENT_VALIDATED = False

# SEVERITY 5 (L-AM16-SEV). The 40/40 evidence above covers severities 1-4. Severity 5 of the four
# non-noise corruptions (AM-16 item 6) joins the reference grid in
# scripts/smoke_corruption_vendor.py (56 cases), and that grid is evidence of record only when run
# in the student image (local block L1). Until the L1 evidence commit sets this True and extends
# VALIDATION_EVIDENCE, an official cache refuses severity-5 items; smoke and provisional caches may
# hold them, and their manifests record the status "pending".
SEVERITY5_REFERENCE_VALIDATED = False

# The corruption RUNTIME closure imports exactly these third-party modules (AST-verified in
# scripts/smoke_corruption_vendor.py). scikit-image's own transitive dependencies are not corruption
# imports, and cv2/scipy are NOT in this closure.
CORRUPTION_RUNTIME_DEPENDENCIES = ("numpy", "pillow", "scikit_image")


class CorruptionCacheError(RuntimeError):
    """A rejected cache generation or a failed cache verification."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def __reduce__(self):                    # a refusal raised in a worker keeps its code
        return (self.__class__, (self.code, str(self)))


def _fail(code: str, message: str):
    raise CorruptionCacheError(code, message)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(obj) -> bytes:
    """src/eval/artifacts.py's rule (:93-96): UTF-8, sorted keys, compact, non-finite forbidden."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def runtime_versions() -> dict:
    """Actual generation-environment versions — always recorded, never assumed."""
    import PIL
    import skimage
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pillow": PIL.__version__, "scikit_image": skimage.__version__,
            "platform": platform.platform()}


def official_dependency_pins() -> dict:
    """The registered pins plus their two-phase status. Machine-readable, never inferred."""
    return {
        "source": OFFICIAL_DEPENDENCY_SOURCE,
        "python": OFFICIAL_PYTHON,
        "numpy": OFFICIAL_NUMPY,
        "pillow": OFFICIAL_PILLOW,
        "scikit_image": OFFICIAL_SCIKIT_IMAGE,
        "corruption_runtime_dependencies": list(CORRUPTION_RUNTIME_DEPENDENCIES),
        "dependency_versions_registered": DEPENDENCY_VERSIONS_REGISTERED,
        "official_dependency_environment_validated": OFFICIAL_DEPENDENCY_ENVIRONMENT_VALIDATED,
        # True only because the 40/40 zero-tolerance reference grid actually passed under these
        # exact versions; it is a claim about the corruption closure, nothing wider.
        "byte_reproducibility_proven_under_these_pins": OFFICIAL_DEPENDENCY_ENVIRONMENT_VALIDATED,
        "validation_scope": VALIDATION_SCOPE,
        "full_experiment_environment_validated": FULL_EXPERIMENT_ENVIRONMENT_VALIDATED,
        "validation_evidence": dict(VALIDATION_EVIDENCE),
        "severity5_reference_validated": SEVERITY5_REFERENCE_VALIDATED,
        "severity5_reference_status": "validated" if SEVERITY5_REFERENCE_VALIDATED else "pending",
    }


def runtime_pin_mismatches(actual: dict | None = None) -> list[str]:
    """Which registered pins the running interpreter does NOT satisfy.

    Python is compared at MAJOR.MINOR (the pin is a series, e.g. '3.11'); the three libraries are
    compared exactly, because the corrupted bytes depend on their precise implementations.
    """
    actual = actual or runtime_versions()
    mismatches = []
    running_python = ".".join(str(actual.get("python", "")).split(".")[:2])
    if running_python != OFFICIAL_PYTHON:
        mismatches.append(f"python {actual.get('python')!r} != {OFFICIAL_PYTHON}.x")
    for key, pin in (("numpy", OFFICIAL_NUMPY), ("pillow", OFFICIAL_PILLOW),
                     ("scikit_image", OFFICIAL_SCIKIT_IMAGE)):
        if actual.get(key) != pin:
            mismatches.append(f"{key} {actual.get(key)!r} != {pin}")
    return mismatches


def require_official_dependency_environment(actual: dict | None = None) -> dict:
    """Gate for REAL cache generation. Refuses while validation is pending or versions differ.

    Two distinct refusals, never collapsed: an unvalidated environment is refused even if the
    versions happen to match, because "the numbers were written down" is not evidence that the
    deterministic corruption verification has ever run under them.
    """
    mismatches = runtime_pin_mismatches(actual)
    if mismatches:
        _fail("official_dependency_version_mismatch",
              "the running environment does not match the registered corruption pins: "
              + "; ".join(mismatches)
              + f". The corrupted bytes depend on numpy/Pillow/scikit-image, so the cache may only "
                f"be generated on the pinned stack ({OFFICIAL_DEPENDENCY_SOURCE}).")
    if not OFFICIAL_DEPENDENCY_ENVIRONMENT_VALIDATED:
        _fail("official_dependency_environment_unvalidated",
              "OFFICIAL CORRUPTION CACHE GENERATION BLOCKED: the versions are REGISTERED "
              f"(python {OFFICIAL_PYTHON}, numpy {OFFICIAL_NUMPY}, pillow {OFFICIAL_PILLOW}, "
              f"scikit-image {OFFICIAL_SCIKIT_IMAGE}) but the environment has not been EXECUTABLY "
              "VALIDATED. Install that exact stack and re-run the deterministic corruption "
              "verification (scripts/smoke_corruption_vendor.py) there; only then may "
              "OFFICIAL_DEPENDENCY_ENVIRONMENT_VALIDATED become True. Byte reproducibility under "
              "these pins has not been demonstrated.")
    return official_dependency_pins()


# ---------------------------------------------------------------- names (TL interface section 2)
def refuse_test_named(**paths) -> None:
    """LC6: refuse a cache root, journal path or output path whose string contains "test" in any
    case. A string test only, run before any filesystem call; a relative path is tested as the
    absolute path it names (os.path.abspath: the working directory, never the path itself)."""
    for label, p in paths.items():
        if p is not None and "test" in os.path.abspath(str(p)).lower():
            _fail("cache_root_test_named",
                  f"the {label.replace('_', ' ')} {os.path.abspath(str(p))!r} contains 'test': no "
                  f"path this lane creates may (the TEST cache root's last component is "
                  f"{TDAY_CACHE_ROOT_NAME!r})")


# ---------------------------------------------------------------- git, pin and protocol
def _git(repo_root, *args) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=str(repo_root),
                              capture_output=True, timeout=120, check=False,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError) as e:
        _fail("git_unavailable", f"git could not run in {repo_root}: {type(e).__name__}: {e}")


def git_head(repo_root=REPO) -> str:
    p = _git(repo_root, "rev-parse", "HEAD")
    commit = p.stdout.decode("utf-8", "replace").strip()
    if p.returncode != 0 or not _HEX40.fullmatch(commit):
        _fail("git_unavailable", f"git rev-parse HEAD failed in {repo_root}: "
                                 f"{p.stderr.decode('utf-8', 'replace')[:200]}")
    return commit


def governed_porcelain(repo_root=REPO) -> bytes:
    p = _git(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--",
             *GOVERNED_PATHSPEC)
    if p.returncode != 0:
        _fail("git_unavailable", f"git status failed in {repo_root}: "
                                 f"{p.stderr.decode('utf-8', 'replace')[:200]}")
    return p.stdout


def require_governed_clean(repo_root=REPO) -> dict:
    """EVALUATION_CONTRACT section 7.1: no dirty or untracked governed path."""
    raw = governed_porcelain(repo_root)
    if raw:
        paths = [r[3:].decode("utf-8", "replace") for r in raw.split(b"\x00") if len(r) > 3]
        _fail("governed_paths_dirty",
              f"governed paths are dirty or untracked in {repo_root}: {paths[:5]}")
    return {"governed_paths_clean": True, "worktree_state_sha256": _sha256_bytes(raw)}


def git_blob_sha1(data: bytes) -> str:
    """The git blob id of `data` (what `git hash-object` prints), computed in Python."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load_roles(repo_root=REPO) -> corruption_roles.Roles:
    try:
        return corruption_roles.load(Path(repo_root) / PROTOCOL_REL)
    except corruption_roles.CorruptionRolesError as e:
        _fail("protocol_invalid", str(e))


def check_protocol_vendor(roles: corruption_roles.Roles) -> dict[str, tuple[int, ...]]:
    """The protocol's scored sets, refused unless they equal the vendored dispatch's."""
    sets = roles.severities_by_corruption
    vendor = {c: tuple(SEVERITIES_BY_CORRUPTION.get(c, ())) for c in CORRUPTION_IDS}
    if tuple(roles.order) != tuple(CORRUPTION_IDS) or sets != vendor:
        _fail("protocol_vendor_disagree",
              f"the protocol scores {sets} but the vendored dispatch generates {vendor}")
    return sets


def protocol_severity_sets() -> dict[str, tuple[int, ...]]:
    """Each corruption's scored severities from configs/corruption_protocol.json (torch-free
    loader), refused unless they equal the vendored dispatch's `SEVERITIES_BY_CORRUPTION`."""
    return check_protocol_vendor(load_roles())


def protocol_block(repo_root=REPO) -> dict:
    roles = load_roles(repo_root)
    raw = (Path(repo_root) / PROTOCOL_REL).read_bytes()
    return {"path": PROTOCOL_REL, "sha256": _sha256_bytes(raw),
            "schema_version": roles.schema_version}


def pin_digest(pin: dict) -> str:
    """pin_sha256: sha256 of canonical_json_bytes(the pin without pin_sha256) (TL section 3)."""
    return _sha256_bytes(canonical_json_bytes({k: v for k, v in pin.items()
                                               if k != "pin_sha256"}))


def build_pin(split: str, *, repo_root=REPO) -> dict:
    """The generator pin (TL interface section 3). Reads no data: the repository only."""
    from src.corruption_source import require_split, split_sizes
    try:
        require_split(split)
    except CorruptionSourceError as e:
        _fail(e.code, str(e))
    roles = load_roles(repo_root)
    check_protocol_vendor(roles)
    script = (Path(repo_root) / GENERATOR_SCRIPT).read_bytes()
    body = {
        "schema": PIN_SCHEMA,
        "script": {"path": GENERATOR_SCRIPT, "git_blob": git_blob_sha1(script)},
        "commit": git_head(repo_root),
        "library_versions": {"python": OFFICIAL_PYTHON, "numpy": OFFICIAL_NUMPY,
                             "pillow": OFFICIAL_PILLOW, "scikit_image": OFFICIAL_SCIKIT_IMAGE,
                             "image_digest": STUDENT_IMAGE_DIGEST},
        "parameters": {"split": split, "expected_images": split_sizes()[split],
                       "protocol": protocol_block(repo_root), "cells": list(roles.cell_ids),
                       "source_resize": SOURCE_RESIZE,
                       "vendor_runtime_sha256": VENDOR_RUNTIME_SHA256,
                       "seed_policy_id": SEED_POLICY_ID, "master_seed": MASTER_SEED,
                       "cache_schema": CACHE_SCHEMA, "storage": dict(STORAGE)},
    }
    return {**body, "pin_sha256": pin_digest(body)}


def full_grid(roles: corruption_roles.Roles) -> list[tuple[str, int]]:
    """The 24 scored cells in protocol order (corruption order, then ascending severity)."""
    return [(c.corruption, c.severity) for c in roles.cells]


# ---------------------------------------------------------------- items
def item_relative_path(corruption_id: str, severity: int, manifest_index: int) -> str:
    """LCC1, the one item-path rule: the source's manifest_index in five digits, never its stem."""
    return f"{corruption_id}/s{severity}/{manifest_index:05d}.npy"


def write_entry(cache_root: Path, image_id: str, manifest_index: int, corruption_id: str,
                severity: int, array: np.ndarray, seed: int) -> dict:
    """Persist one corrupted array losslessly and atomically at its index path; return its
    manifest entry (`image_id` is recorded as data, never used in the path).

    Both digests are computed from the final file, after the move into place.
    """
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        _fail("entry_not_uint8_rgb", f"cache entries are uint8 H*W*3; got {array.dtype}/{array.shape}")
    rel = item_relative_path(corruption_id, severity, manifest_index)
    path = Path(cache_root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    stored = np.load(path, allow_pickle=False)
    return {
        "image_id": image_id, "corruption_id": corruption_id, "severity": int(severity),
        "master_seed": MASTER_SEED, "item_seed": int(seed),
        "cache_relative_path": rel, "shape": list(stored.shape), "dtype": "uint8",
        "cached_file_sha256": _sha256_file(path),
        "corrupted_pixel_sha256": _sha256_bytes(stored.tobytes()),
        "patched_vendor_runtime_sha256": compute_runtime_sha256(),
    }


def item_verifies(cache_root: Path, entry: dict) -> bool:
    """True when the item's file, shape, dtype and pixels all match the recorded entry."""
    path = Path(cache_root) / entry["cache_relative_path"]
    if not path.is_file() or _sha256_file(path) != entry["cached_file_sha256"]:
        return False
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (arr.dtype == np.uint8 and list(arr.shape) == list(entry["shape"])
            and _sha256_bytes(arr.tobytes()) == entry["corrupted_pixel_sha256"])


def corrupt_source(source_pixels: np.ndarray, image_id: str, corruption_id: str, severity: int,
                   *, lift_floor: bool):
    """One item from one resized source: the vendored corruption, floor lifted only when told."""
    return apply_corruption(source_pixels, corruption_id, severity, image_id=image_id,
                            lift_floor=lift_floor)


def floor_lifted(record: dict) -> bool:
    """LC7: an image whose resized short side is below upstream's 32-px floor."""
    return min(record["resized_shape"]) < MIN_SIDE


# ---------------------------------------------------------------- manifest assembly
def canonical_manifest_sha256(manifest: dict) -> str:
    """Digest over the manifest without its checksum, `versions` and `generation`."""
    body = {k: v for k, v in manifest.items()
            if k not in ("canonical_manifest_sha256", "versions", "generation")}
    return _sha256_bytes(canonical_json_bytes(body))


def cell_digest(entries: list[dict]) -> str:
    return _sha256_bytes(canonical_json_bytes(entries))


def sorted_entries(entries: list[dict], sources: list[dict]) -> list[dict]:
    """Manifest entry order: corruption order, severity, then the source's manifest_index."""
    index_of = {s["image_id"]: s["manifest_index"] for s in sources}
    rank = {c: i for i, c in enumerate(CORRUPTION_IDS)}
    return sorted(entries, key=lambda e: (rank[e["corruption_id"]], e["severity"],
                                          index_of[e["image_id"]]))


def assemble_manifest(*, status: str, split: str, expected_images, grid, roles, protocol: dict,
                      sources: list[dict], entries: list[dict], pin, versions: dict,
                      generation: dict) -> dict:
    sources = sorted(sources, key=lambda s: s["manifest_index"])
    order = [c for c in CORRUPTION_IDS if any(c == g for g, _ in grid)]
    entries = sorted_entries(entries, sources)
    by_cell: dict[tuple[str, int], list[dict]] = {}
    for e in entries:
        by_cell.setdefault((e["corruption_id"], e["severity"]), []).append(e)
    cells = []
    for c, s in grid:
        ce = by_cell.get((c, s), [])
        cells.append({"cell_id": corruption_roles.cell_id(c, s), "corruption": c, "severity": s,
                      "role": roles.severity_role(c, s), "items": len(ce),
                      "cell_sha256": cell_digest(ce)})
    inferential = [x["cell_sha256"] for x in cells
                   if x["role"] == corruption_roles.ROLE_INFERENTIAL]
    manifest = {
        "schema": CACHE_SCHEMA, "status": status,
        "dataset": {"name": DATASET_NAME, "doi": DATASET_DOI, "split": split},
        "split": split,
        "official_expected_images": OFFICIAL_EXPECTED_IMAGES, "expected_images": expected_images,
        "source_image_count": len(sources),
        "protocol": protocol,
        "corruption_order": order,
        "severities_by_corruption": {c: [s for g, s in grid if g == c] for c in order},
        "cells": cells,
        "inferential_cells_sha256": _sha256_bytes(canonical_json_bytes(inferential)),
        "sources": sources, "entries": entries,
        "seed_policy_id": SEED_POLICY_ID, "master_seed": MASTER_SEED,
        "vendor": vendor_provenance(), "dependency_pins": official_dependency_pins(),
        "pin": pin, "storage": dict(STORAGE),
        "floor_lifted": [s["image_id"] for s in sources if floor_lifted(s)],
        "versions": versions, "generation": generation,
    }
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    return manifest


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def _grid_for(corruptions, severities, roles) -> list[tuple[str, int]]:
    scored = check_protocol_vendor(roles)
    grid = []
    for corruption_id in corruptions:
        wanted = scored[corruption_id] if severities is None else tuple(severities)
        refused = [s for s in wanted if s not in scored[corruption_id]]
        if refused:
            _fail("severity_not_generated",
                  f"{corruption_id} is generated only at severities {scored[corruption_id]}; "
                  f"refused {refused}")
        grid.extend((corruption_id, int(s)) for s in wanted)
    return grid


def require_status(status: str) -> None:
    if status not in CACHE_STATUSES:
        _fail("unknown_status", f"cache status must be one of {CACHE_STATUSES}, got {status!r}")


def require_severity5_validated(status: str, grid) -> None:
    if status == "official" and not SEVERITY5_REFERENCE_VALIDATED and any(s == 5 for _, s in grid):
        _fail("severity5_reference_unvalidated",
              "an official cache may not hold severity-5 items until the severity-5 reference "
              "grid is validated in the student image (local block L1); the status is pending")


# ---------------------------------------------------------------- in-memory (synthetic) caches
def generate(cache_root: Path, images: dict[str, np.ndarray], *,
             corruptions=CORRUPTION_IDS, severities=None,
             split: str = OFFICIAL_SPLIT, expected_images: int | None = None,
             status: str = "smoke") -> dict:
    """Build a cache for the supplied RESIZED uint8 arrays (synthetic smoke caches).

    `severities=None` builds every scored severity of each corruption, as the protocol sets them
    (1-5 for the four non-noise corruptions, 1-4 for gaussian_noise); an explicit tuple applies to
    every listed corruption and must lie in each one's scored set. gaussian_noise at severity 5 is
    never generated. An official cache refuses severity-5 items until
    SEVERITY5_REFERENCE_VALIDATED. Real caches come from build_cache
    (scripts/build_corruption_cache.py).
    """
    refuse_test_named(cache_root=cache_root)
    require_status(status)
    unknown = [c for c in corruptions if c not in CORRUPTION_IDS]
    if unknown:
        _fail("unknown_corruption", f"unregistered corruptions: {unknown}")
    roles = load_roles()
    grid = _grid_for(corruptions, severities, roles)
    require_severity5_validated(status, grid)

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    sources, entries = [], []
    for index, image_id in enumerate(sorted(images)):          # canonical order
        src = array_source(index, image_id, images[image_id])
        sources.append(src.record)
        lift = floor_lifted(src.record)
        for corruption_id, severity in grid:
            arr, seed = corrupt_source(src.pixels, image_id, corruption_id, severity,
                                       lift_floor=lift)
            entries.append(write_entry(cache_root, image_id, index, corruption_id, severity, arr,
                                       seed))
    if expected_images is not None and len(images) != expected_images:
        _fail("image_count_mismatch",
              f"expected {expected_images} source images, got {len(images)}")
    manifest = assemble_manifest(
        status=status, split=split, expected_images=expected_images, grid=grid, roles=roles,
        protocol=protocol_block(), sources=sources, entries=entries, pin=None,
        versions=runtime_versions(),
        generation={"mode": "in_memory", "end_utc": utc_iso(), "python": sys.version.split()[0]})
    _write_atomic(cache_root / MANIFEST_NAME, manifest_bytes(manifest))
    return manifest


# ---------------------------------------------------------------- real caches: journal and resume
def journal_path(cache_root) -> Path:
    return Path(cache_root) / JOURNAL_DIR / JOURNAL_NAME


def _journal_append(path: Path, record: dict) -> None:
    with open(path, "ab") as f:
        f.write(canonical_json_bytes(record) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def end_torn_line(path: Path) -> bool:
    """A crash mid-write leaves a last line without its newline: end it, so the next record
    starts a line of its own (the torn line stays, ignored and counted). True if it was torn."""
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return False
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return False
        f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())
    return True


def require_journal_header(journal: dict) -> None:
    if journal["header"] is None:
        _fail("resume_journal_invalid",
              "the journal holds no complete header (a crash before it was written): nothing "
              "usable was journalled, so there is nothing to resume; start again in an empty root")


def read_journal(cache_root) -> dict:
    """The journal's header, run lines, sources by image_id and items by (image_id, cell).

    A line that does not parse (a crash mid-write) is ignored and counted; every item it would
    have named is regenerated, and only journalled digests are compared for determinism.
    """
    out = {"header": None, "runs": [], "sources": {}, "items": {}, "ignored": 0}
    path = journal_path(cache_root)
    for raw in path.read_bytes().split(b"\n"):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
            kind = rec["kind"]
        except (ValueError, TypeError, KeyError):
            out["ignored"] += 1
            continue
        if kind == "header":
            out["header"] = rec
        elif kind == "run":
            out["runs"].append(rec)
        elif kind == "source":
            out["sources"][rec["record"]["image_id"]] = rec["record"]
        elif kind == "item":
            e = rec["entry"]
            out["items"][(e["image_id"], e["corruption_id"], e["severity"])] = e
        else:
            out["ignored"] += 1
    return out


def refuse_manifest_exists(root: Path) -> None:
    """LC5: a root whose manifest exists is complete; it is never regenerated or resumed."""
    if (Path(root) / MANIFEST_NAME).exists():
        _fail("manifest_exists",
              f"{root} already holds a manifest: a complete cache is never regenerated or resumed "
              "(repair one item with --repair)")


def refuse_resume_without_journal(root: Path) -> None:
    if not journal_path(root).is_file():
        _fail("resume_without_journal", f"--resume needs {journal_path(root)}")


def refuse_root_not_empty(root: Path) -> None:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        _fail("cache_root_not_empty",
              f"{root} is not empty; use --resume to continue a crashed run")


def check_cache_root_state(cache_root, *, resume: bool) -> None:
    """LC5 and the resume rules, before anything is written."""
    root = Path(cache_root)
    refuse_manifest_exists(root)
    if resume:
        refuse_resume_without_journal(root)
    else:
        refuse_root_not_empty(root)


def journal_matches(header, *, pin: dict, cells: list, split: str, status: str) -> bool:
    return (header is not None and header.get("pin_sha256") == pin["pin_sha256"]
            and header.get("cells") == cells and header.get("split") == split
            and header.get("status") == status)


def same_source(prior: dict, record: dict) -> bool:
    return prior == record


def same_digests(entry: dict, prior: dict) -> bool:
    return (entry["cached_file_sha256"], entry["corrupted_pixel_sha256"]) == \
        (prior["cached_file_sha256"], prior["corrupted_pixel_sha256"])


def resume_keeps(root: Path, prior: dict | None, rel: str) -> bool:
    """Resume finds a journalled item at its index path (LCC1): it is kept only when its journal
    line names that path and the file there verifies against the journalled digests."""
    return (prior is not None and prior.get("cache_relative_path") == rel
            and item_verifies(root, prior))


def _image_task(task: dict) -> dict:
    """Read one source, then keep, generate or regenerate its items. Runs in-process or in a
    spawned worker; the parent alone writes the journal."""
    if task.get("_die") and multiprocessing.parent_process() is not None:
        os._exit(70)                     # smoke hook (S2 X33): a worker killed outright
    src = Source(task["manifest_index"], task["image_id"], Path(task["path"]))
    try:
        si = read_source(src)
    except CorruptionSourceError as e:
        _fail(e.code, str(e))
    prior_source = task["prior_source"]
    if prior_source is not None and not same_source(prior_source, si.record):
        _fail("resume_source_mismatch",
              f"{src.image_id}: the source differs from the journalled one; refusing to mix "
              "two sources in one cache")
    lift = floor_lifted(si.record)
    root = Path(task["cache_root"])
    out = []
    for corruption_id, severity in task["grid"]:
        prior = task["prior_items"].get(f"{corruption_id}/{severity}")
        if resume_keeps(root, prior, item_relative_path(corruption_id, severity,
                                                        src.manifest_index)):
            out.append(("kept", prior))
            continue
        arr, seed = corrupt_source(si.pixels, src.image_id, corruption_id, severity,
                                   lift_floor=lift)
        entry = write_entry(root, src.image_id, src.manifest_index, corruption_id, severity, arr,
                            seed)
        if prior is not None:
            if not same_digests(entry, prior):
                _fail("nondeterministic_item",
                      f"STOP: {src.image_id} {corruption_id} s{severity} regenerated to digests "
                      "that differ from the journalled ones")
            out.append(("regenerated", entry))
        else:
            out.append(("generated", entry))
    return {"record": si.record, "prior_source": prior_source is not None, "items": out}


def worker_died(e: BaseException):
    _fail("worker_died",
          f"a generator worker died outright ({type(e).__name__}: {e}); every journalled item is "
          "kept and --resume continues the run")


def run_workers(tasks: list[dict], workers: int, consume) -> None:
    """Spawned workers, results consumed as they complete. A refusal raised in a worker keeps its
    code; a worker that dies outright (the OOM killer, SIGKILL) ends the run as worker_died, where
    multiprocessing.Pool would wait forever for the lost task."""
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as ex:
        futures = [ex.submit(_image_task, t) for t in tasks]
        try:
            for fut in concurrent.futures.as_completed(futures):
                consume(fut.result())
        except BrokenProcessPool as e:
            for f in futures:
                f.cancel()
            worker_died(e)
        except BaseException:
            for f in futures:
                f.cancel()
            raise


def build_cache(cache_root, sources: list[Source], *, split: str, status: str, pin: dict,
                expected_images, workers: int = 1, resume: bool = False, repo_root=REPO,
                provenance: dict | None = None, _fail_after_items: int | None = None,
                _reverse: bool = False, _kill_worker_at: int | None = None) -> dict:
    """Generate (or resume) a real cache from source files. The CLI runs every gate first.
    The underscored arguments are smoke hooks (a crash, the source order, a worker killed)."""
    root = Path(cache_root)
    refuse_test_named(cache_root=root, journal_path=journal_path(root))
    require_status(status)
    check_cache_root_state(root, resume=resume)
    roles = load_roles(repo_root)
    grid = full_grid(roles)
    check_protocol_vendor(roles)
    require_severity5_validated(status, grid)
    cells = [corruption_roles.cell_id(c, s) for c, s in grid]

    jpath = journal_path(root)
    if resume:
        journal = read_journal(root)
        require_journal_header(journal)
        if not journal_matches(journal["header"], pin=pin, cells=cells, split=split,
                               status=status):
            _fail("resume_pin_mismatch",
                  "the journal was written under another pin, split, status or cell set")
        end_torn_line(jpath)                         # after every refusal, before the first append
    else:
        jpath.parent.mkdir(parents=True, exist_ok=True)
        journal = {"header": None, "runs": [], "sources": {}, "items": {}, "ignored": 0}
        _journal_append(jpath, {"kind": "header", "journal_schema": JOURNAL_SCHEMA,
                                "cache_schema": CACHE_SCHEMA, "pin_sha256": pin["pin_sha256"],
                                "split": split, "status": status, "cells": cells,
                                "expected_sources": len(sources)})
    run = {"kind": "run", "start_utc": utc_iso(), "workers": int(workers), "resume": bool(resume),
           "python": sys.version.split()[0], "journal_lines_ignored": journal["ignored"]}
    _journal_append(jpath, run)

    ordered = list(reversed(sources)) if _reverse else list(sources)
    tasks = [{"manifest_index": s.manifest_index, "image_id": s.image_id, "path": str(s.path),
              "cache_root": str(root), "grid": grid,
              "prior_source": journal["sources"].get(s.image_id),
              "prior_items": {f"{c}/{v}": journal["items"][(s.image_id, c, v)]
                              for c, v in grid if (s.image_id, c, v) in journal["items"]}}
             for s in ordered]
    counts = {"generated": 0, "regenerated": 0, "kept": 0}
    regenerated = []
    records, entries = [], []
    appended = 0

    def consume(result):
        nonlocal appended
        rec = result["record"]
        records.append(rec)
        if not result["prior_source"]:
            _journal_append(jpath, {"kind": "source", "record": rec})
        for how, entry in result["items"]:
            counts[how] += 1
            entries.append(entry)
            if how == "kept":
                continue
            if how == "regenerated":
                regenerated.append(f"{entry['image_id']} {entry['corruption_id']} "
                                   f"s{entry['severity']}: journalled item failed verification")
            _journal_append(jpath, {"kind": "item", "entry": entry})
            appended += 1
            if _fail_after_items is not None and appended >= _fail_after_items:
                raise RuntimeError(f"simulated crash after {appended} journalled items")

    if _kill_worker_at is not None:
        tasks[_kill_worker_at]["_die"] = True
    if workers <= 1:
        for task in tasks:
            consume(_image_task(task))
    else:
        run_workers(tasks, workers, consume)

    expected_entries = len(sources) * len(grid)
    if len(entries) != expected_entries:            # internal invariant, not a refusal
        raise RuntimeError(f"internal: {len(entries)} items, expected {expected_entries}")
    end = utc_iso()
    runs = journal["runs"] + [run]
    generation = {"invocations": [{k: r[k] for k in ("start_utc", "workers", "resume", "python")}
                                  for r in runs],
                  "end_utc": end, "workers": int(workers),
                  "items": dict(counts), "regenerated": regenerated,
                  **(provenance or {})}
    manifest = assemble_manifest(
        status=status, split=split, expected_images=expected_images, grid=grid, roles=roles,
        protocol=protocol_block(repo_root), sources=records, entries=entries, pin=pin,
        versions=runtime_versions(), generation=generation)
    _write_atomic(root / MANIFEST_NAME, manifest_bytes(manifest))
    _write_atomic(root / JOURNAL_DIR / f"run_{utc_stamp()}.json",
                  canonical_json_bytes({**run, "end_utc": end, "items": counts,
                                        "regenerated": regenerated,
                                        "canonical_manifest_sha256":
                                            manifest["canonical_manifest_sha256"]}) + b"\n")
    return manifest


# ---------------------------------------------------------------- consumption
def _strict_json(text: str):
    """Strict JSON, as corruption_roles.load reads the protocol: no NaN or Infinity, no
    duplicate key (the file's sha256 would cover both copies, the parse only the last)."""
    def constant(x):
        raise ValueError(f"non-finite JSON constant {x!r}")

    def finite(s):
        x = float(s)
        if not math.isfinite(x):
            raise ValueError(f"non-finite JSON number {s!r}")
        return x

    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate JSON key {k!r}")
            out[k] = v
        return out
    return json.loads(text, parse_constant=constant, parse_float=finite, object_pairs_hook=unique)


def item_path_problems(manifest: dict) -> list:
    """LCC1: the entries whose cache_relative_path is not item_relative_path(corruption, severity,
    the manifest_index of the source with its image_id), or whose source has no five-digit index."""
    index_of = {s.get("image_id"): s.get("manifest_index") for s in manifest.get("sources") or []}
    bad = []
    for e in manifest.get("entries") or []:
        i = index_of.get(e.get("image_id"))
        if type(i) is not int or not 0 <= i < 10 ** 5 or e.get("cache_relative_path") != \
                item_relative_path(e.get("corruption_id"), e.get("severity"), i):
            bad.append(e.get("cache_relative_path"))
    return bad


def require_item_paths(manifest: dict) -> None:
    bad = item_path_problems(manifest)
    if bad:
        _fail("item_path_mismatch",
              f"{len(bad)} entr{'y is' if len(bad) == 1 else 'ies are'} not at "
              f"{STORAGE['relative_path']} (item paths come from manifest_index, never from a "
              f"stem): {bad[:3]}")


def load_manifest(cache_root: Path) -> dict:
    path = Path(cache_root) / MANIFEST_NAME
    if not path.is_file():
        _fail("manifest_missing", f"cache manifest not found: {path}")
    try:
        manifest = _strict_json(path.read_text(encoding="utf-8"))
    except ValueError as e:
        _fail("manifest_schema", f"cache manifest is not strict JSON: {e}")
    if not isinstance(manifest, dict) or manifest.get("schema") != CACHE_SCHEMA:
        _fail("manifest_schema", f"unknown cache schema {manifest.get('schema')!r}"
              if isinstance(manifest, dict) else "cache manifest is not an object")
    if manifest.get("canonical_manifest_sha256") != canonical_manifest_sha256(manifest):
        _fail("manifest_checksum_mismatch", "cache manifest checksum does not match its contents")
    if manifest.get("seed_policy_id") != SEED_POLICY_ID:
        _fail("seed_policy_mismatch",
              f"cache was built under seed policy {manifest.get('seed_policy_id')!r}, "
              f"this build uses {SEED_POLICY_ID!r}")
    vendor = manifest.get("vendor")
    recorded = vendor.get("patched_vendor_runtime_sha256") if isinstance(vendor, dict) else None
    if recorded != compute_runtime_sha256():
        _fail("vendor_checksum_mismatch",
              "cache was generated by a different vendored corruption implementation")
    require_item_paths(manifest)
    return manifest


class CacheIndex:
    """(image_id, corruption, severity) -> entry, and image_id -> source, built once."""

    def __init__(self, manifest: dict):
        self.entries = {(e["image_id"], e["corruption_id"], int(e["severity"])): e
                        for e in manifest["entries"]}
        self.sources = {s["image_id"]: s for s in manifest.get("sources", [])}


_INDEXES: "OrderedDict[int, tuple[list, int, CacheIndex]]" = OrderedDict()
_INDEX_SLOTS = 8


def cache_index(manifest: dict) -> CacheIndex:
    """The manifest's index, memoised on its entries list. The memo holds a reference to that
    list (so its id is never reused) and its length; a smoke that forges a copy gets a new index.
    """
    entries = manifest["entries"]
    hit = _INDEXES.get(id(entries))
    if hit is not None and hit[0] is entries and hit[1] == len(entries):
        _INDEXES.move_to_end(id(entries))
        return hit[2]
    index = CacheIndex(manifest)
    _INDEXES[id(entries)] = (entries, len(entries), index)
    while len(_INDEXES) > _INDEX_SLOTS:
        _INDEXES.popitem(last=False)
    return index


def load_cached(cache_root: Path, manifest: dict, image_id: str, corruption_id: str,
                severity: int) -> np.ndarray:
    """Load one cached corrupted array with full verification. NEVER regenerates on failure."""
    key = (image_id, corruption_id, int(severity))
    entry = cache_index(manifest).entries.get(key)
    if entry is None:
        _fail("cache_entry_missing",
              f"no cached entry for {key}; the evaluator never regenerates corruptions")
    path = Path(cache_root) / entry["cache_relative_path"]
    if not path.is_file():
        _fail("cache_file_missing", f"cached file missing: {path}")
    if _sha256_file(path) != entry["cached_file_sha256"]:
        _fail("cache_file_hash_mismatch", f"cached file has been modified: {path}")
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.uint8 or list(arr.shape) != entry["shape"]:
        _fail("cache_entry_corrupt", f"cached array is not the recorded uint8 {entry['shape']}")
    if _sha256_bytes(arr.tobytes()) != entry["corrupted_pixel_sha256"]:
        _fail("cache_pixel_hash_mismatch", f"cached pixels do not match their recorded digest: {path}")
    return arr


# ---------------------------------------------------------------- repair (LC8)
def require_fault_report(report: Path) -> None:
    if not Path(report).is_file():
        _fail("repair_fault_report_missing", f"--fault-report {report} is not a file")


def repair_matches(file_sha: str, pixel_sha: str, seed: int, entry: dict) -> bool:
    return (file_sha == entry["cached_file_sha256"]
            and pixel_sha == entry["corrupted_pixel_sha256"] and int(seed) == entry["item_seed"])


def require_repair_source(manifest: dict, record: dict) -> None:
    """A repair reads the source the cache records, or nothing is regenerated."""
    recorded = cache_index(manifest).sources.get(record["image_id"])
    if not same_source(recorded, record):
        _fail("repair_source_mismatch",
              f"{record['image_id']}: the source differs from the one the cache records; nothing "
              "was regenerated")


def repair_target(cache_root: Path, record: dict, corruption_id: str, severity: int) -> Path:
    """LC8, LCC1: the file a repair replaces, found by the recorded source's manifest_index."""
    return Path(cache_root) / item_relative_path(corruption_id, int(severity),
                                                 record["manifest_index"])


def repair_item(cache_root, source: Source, corruption_id: str, severity: int, *,
                fault_report) -> dict:
    """Regenerate ONE item of a cache whose manifest verifies, from the source it records; keep it
    only if both recorded digests match, at its index path (repair_target). The manifest is never
    rewritten; repair/<UTC>.json sits outside the checksum and is written before the item is
    replaced.
    """
    root = Path(cache_root)
    report = Path(fault_report)
    require_fault_report(report)
    manifest = load_manifest(root)
    entry = cache_index(manifest).entries.get((source.image_id, corruption_id, int(severity)))
    if entry is None:
        _fail("cache_entry_missing", f"no cached entry for ({source.image_id}, {corruption_id}, "
                                     f"{severity}) in {root}")
    try:
        si = read_source(source)
    except CorruptionSourceError as e:
        _fail(e.code, str(e))
    require_repair_source(manifest, si.record)
    arr, seed = corrupt_source(si.pixels, source.image_id, corruption_id, int(severity),
                               lift_floor=floor_lifted(si.record))
    stamp = utc_stamp()
    rdir = root / "repair"
    rdir.mkdir(parents=True, exist_ok=True)
    candidate = rdir / f"{stamp}.candidate.npy"
    with open(candidate, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    file_sha = _sha256_file(candidate)
    pixel_sha = _sha256_bytes(np.load(candidate, allow_pickle=False).tobytes())
    match = repair_matches(file_sha, pixel_sha, seed, entry)
    record = {"schema": REPAIR_SCHEMA, "utc": utc_iso(), "image_id": source.image_id,
              "corruption": corruption_id, "severity": int(severity),
              "fault_report": {"path": str(report), "sha256": _sha256_file(report)},
              "recorded": {"cached_file_sha256": entry["cached_file_sha256"],
                           "corrupted_pixel_sha256": entry["corrupted_pixel_sha256"]},
              "regenerated": {"cached_file_sha256": file_sha, "corrupted_pixel_sha256": pixel_sha},
              "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
              "outcome": "repaired" if match else "repair_digest_mismatch"}
    _write_atomic(rdir / f"{stamp}.json", canonical_json_bytes(record) + b"\n")
    if match:
        os.replace(candidate, repair_target(root, si.record, corruption_id, severity))
    else:
        _fail("repair_digest_mismatch",
              f"STOP: the regenerated {source.image_id} {corruption_id} s{severity} differs from "
              f"its recorded digests; the item was not replaced (record {rdir / (stamp + '.json')}"
              f"; the regenerated candidate is kept at {candidate})")
    return record
