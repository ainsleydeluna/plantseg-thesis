#!/usr/bin/env python3
"""Synthetic verification that corrupted-model evaluation consumes the SHARED cache (path B).

No PlantSeg, no GPU, no model artifact, no dataset. A tiny synthetic cache is generated in TEMP and
the runner's cache-resolution/consumption surface is exercised against it.

Only `fog` is generated here. It is pure NumPy, whereas `brightness` reaches into scikit-image,
whose OpenMP runtime aborts this already-torch-loaded process on the Anaconda development box
(importing the runner pulls `src.eval` -> torch). The full five-corruption grid is proven in a
torch-free process by `scripts/smoke_corruption_vendor.py`.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from scripts.evaluate_corruptions import (build_parser, cached_image_loader,  # noqa: E402
                                          resolve_corruption_cache)
from src import corruption_cache as cache  # noqa: E402
from src.corruption_cache import CorruptionCacheError  # noqa: E402
from src.eval.robustness import RobustnessError  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="smoke_eval_corr_run_"))
RUNNER_PATH = REPO / "scripts/evaluate_corruptions.py"
RUNNER_SRC = RUNNER_PATH.read_text(encoding="utf-8")
results: list[tuple[str, bool, str]] = []


def code_only(src: str) -> str:
    """Source with docstrings and comments stripped.

    These invariants are about what the runner DOES, so prose must neither satisfy nor break them —
    the module docstring legitimately explains that it never calls `real_corruption_transform` and
    never touches a `mask`, and a naive substring scan would fail on that explanation.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return ast.unparse(tree)          # unparse also drops comments


def imported_names(src: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.name for a in node.names)
    return names


RUNNER_CODE = code_only(RUNNER_SRC)


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect(name: str, exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no exception raised")
    except exc as e:
        check(name, True, str(getattr(e, "code", type(e).__name__)))
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


def build_cache(root: Path, severities=(1, 2), split="test", n=3):
    rng = np.random.RandomState(17)
    images = {f"leaf_{i:03d}": rng.randint(0, 256, (32, 48, 3), dtype=np.uint8) for i in range(n)}
    manifest = cache.generate(root, images, corruptions=("fog",), severities=severities,
                              split=split)
    return images, manifest


# ---------------------------------------------------------------- 1. architecture (source level)
def test_runner_architecture() -> None:
    body = RUNNER_CODE[RUNNER_CODE.index("def main("):]
    imports = imported_names(RUNNER_SRC)
    check("runner_never_generates_corruptions",
          "real_corruption_transform" not in RUNNER_CODE
          and "apply_corruption" not in RUNNER_CODE
          and not ({"real_corruption_transform", "apply_corruption"} & imports),
          f"path A neither imported nor called; imports={sorted(imports)[:3]}…")
    check("runner_consumes_cache",
          "load_cached_corruption" in RUNNER_CODE and "load_manifest" in RUNNER_CODE
          and {"load_cached_corruption", "load_manifest"} <= imports)
    check("runner_validates_cache_before_inference",
          body.index("resolve_evaluation_source") < body.index("resolve_corruption_cache"),
          "model source proven first, then cache provenance, then inference")
    check("no_silent_regeneration_path",
          "generate(" not in body and "build_corruption_cache" not in RUNNER_CODE,
          "no fallback that would rebuild a missing item")
    check("mask_never_enters_cache_path",
          "mask" not in RUNNER_CODE.lower(),
          "ground truth never reaches corruption/cache code")
    check("no_plantseg_dataset_access",
          all(t not in RUNNER_CODE for t in ("PlantSegEvalDataset", "build_eval_loader",
                                             "build_dataloader", "PLANTSEG_DATA_ROOT")))
    builder_src = (REPO / "scripts/build_corruption_cache.py").read_text(encoding="utf-8")
    builder = code_only(builder_src)
    check("direct_generation_still_owned_by_builder",
          "generate" in builder and "verify_vendor_integrity" in builder
          and "load_cached_corruption" not in builder,
          "path A remains available exclusively for cache creation")
    check("cache_cli_input_required", "--corruption-cache" in RUNNER_CODE)


# ---------------------------------------------------------------- 2. shared-byte consumption
def test_shared_consumption() -> None:
    root = TMP / "cache_ok"
    images, manifest = build_cache(root)
    man = resolve_corruption_cache(root, "fog", 2, split="test", official=False)
    check("cache_manifest_resolves", man["schema"] == cache.CACHE_SCHEMA,
          man["canonical_manifest_sha256"][:16] + "…")

    # three independent "stage" consumers must receive byte-identical pixels
    loaders = {stage: cached_image_loader(root, man, "fog", 2)
               for stage in ("teacher", "E1", "E6")}
    arrays = {stage: load("leaf_001") for stage, load in loaders.items()}
    check("teacher_e1_e6_receive_identical_bytes",
          np.array_equal(arrays["teacher"], arrays["E1"])
          and np.array_equal(arrays["E1"], arrays["E6"])
          and arrays["E1"].dtype == np.uint8,
          "one frozen realisation shared by every stage")

    entry = next(e for e in man["entries"]
                 if e["image_id"] == "leaf_001" and e["severity"] == 2)
    check("cached_payload_is_pre_padding_uint8_rgb",
          arrays["E1"].shape == tuple(entry["shape"]) == (32, 48, 3)
          and entry["dtype"] == "uint8",
          "already resized+corrupted; evaluation pads and normalizes from here")
    check("cached_bytes_match_recorded_digest",
          cache._sha256_bytes(arrays["E1"].tobytes()) == entry["corrupted_pixel_sha256"])


# ---------------------------------------------------------------- 3. refusals
def test_refusals() -> None:
    root = TMP / "cache_ok"
    man = resolve_corruption_cache(root, "fog", 2, split="test", official=False)
    load = cached_image_loader(root, man, "fog", 2)

    # tampered file bytes
    victim = root / next(e["cache_relative_path"] for e in man["entries"]
                         if e["image_id"] == "leaf_000" and e["severity"] == 2)
    good = np.load(victim)
    np.save(victim, (good + 7).astype(np.uint8))
    expect("tampered_cache_file_rejected", CorruptionCacheError, load, "leaf_000")

    # pixel-digest mismatch specifically: file digest realigned, pixels still wrong
    forged = json.loads(json.dumps(man))
    for e in forged["entries"]:
        if e["image_id"] == "leaf_000" and e["severity"] == 2:
            e["cached_file_sha256"] = cache._sha256_bytes(victim.read_bytes())
    expect("wrong_pixel_digest_rejected", CorruptionCacheError,
           cached_image_loader(root, forged, "fog", 2), "leaf_000")
    np.save(victim, good)

    # missing item, and an id the cache never held -> never silently regenerated
    victim.unlink()
    expect("missing_cache_item_rejected", CorruptionCacheError, load, "leaf_000")
    expect("unknown_image_never_regenerated", CorruptionCacheError, load, "leaf_999")

    # request outside the cached grid
    expect("wrong_corruption_rejected", CorruptionCacheError, resolve_corruption_cache,
           root, "brightness", 2, split="test", official=False)
    expect("wrong_severity_rejected", CorruptionCacheError, resolve_corruption_cache,
           root, "fog", 3, split="test", official=False)
    expect("severity_5_rejected", RobustnessError, load_severity_5, root, man)
    expect("split_mismatch_rejected", CorruptionCacheError, resolve_corruption_cache,
           root, "fog", 2, split="val", official=False)
    expect("missing_cache_root_rejected", CorruptionCacheError, resolve_corruption_cache,
           TMP / "no_such_cache", "fog", 2, split="test", official=False)

    # foreign vendor runtime hash
    root2 = TMP / "cache_foreign"
    build_cache(root2)
    bad = json.loads((root2 / cache.MANIFEST_NAME).read_text(encoding="utf-8"))
    bad["vendor"]["patched_vendor_runtime_sha256"] = "f" * 64
    bad["canonical_manifest_sha256"] = cache.canonical_manifest_sha256(bad)
    (root2 / cache.MANIFEST_NAME).write_text(json.dumps(bad, indent=2, sort_keys=True),
                                             encoding="utf-8")
    expect("foreign_vendor_runtime_hash_rejected", CorruptionCacheError,
           resolve_corruption_cache, root2, "fog", 2, split="test", official=False)

    # an official run demands the full cached test split
    expect("official_requires_complete_cache", CorruptionCacheError, resolve_corruption_cache,
           root, "fog", 2, split="test", official=True)


def load_severity_5(root, man):
    """Severity 5 is never produced; the vocabulary gate must refuse before any cache lookup."""
    return cached_image_loader(root, man, "fog", 5)("leaf_000")


# ---------------------------------------------------------------- 4. CLI surface
def test_cli() -> None:
    p = build_parser()
    try:
        p.parse_args(["--stage", "E1", "--corruption", "fog", "--severity", "2",
                      "--out-dir", str(TMP / "out")])
        check("cache_arg_is_mandatory", False, "parser accepted a run with no cache")
    except SystemExit:
        check("cache_arg_is_mandatory", True, "--corruption-cache required")
    ns = p.parse_args(["--stage", "E1", "--corruption", "fog", "--severity", "2",
                       "--out-dir", str(TMP / "out"), "--corruption-cache", str(TMP / "cache_ok")])
    check("cache_arg_parsed", ns.corruption_cache == str(TMP / "cache_ok"))


def main() -> int:
    print("=" * 78)
    print("CORRUPTED-EVALUATION RUNNER SMOKE — synthetic cache only; no PlantSeg, no model, no GPU")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_runner_architecture, test_shared_consumption, test_refusals, test_cli):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:44}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
