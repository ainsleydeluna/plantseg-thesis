#!/usr/bin/env python3
"""Verification of the vendored corruption closure, seed policy and lossless cache. No PlantSeg.

TORCH-FREE BY CONSTRUCTION, AND THAT IS AN ASSERTION, NOT A CONVENTION. Corruption generation is a
pure NumPy/Pillow/scikit-image job. `src.data.__init__` imports the torch dataloader, and on the
Anaconda development box scikit-image aborts the process with `OMP: Error #15` when torch's OpenMP
runtime initialises first (only `brightness` touches scikit-image, so the abort is corruption-
specific and easy to miss). The vendored third-party numerics therefore live in `src.vendor.
imagecorruptions` and the thesis-owned cache in `src.corruption_cache`, neither of which may import
`src.data`, `torch` or `torchvision`. This smoke proves torch is absent from `sys.modules` both
before and after the corruption/cache work, and greps the module sources for the forbidden imports.

Reference equivalence builds an ISOLATED namespace from the verified upstream source segments —
upstream `corruptions.py` imports cv2/scipy at module scope for unrelated corruptions, so it cannot
simply be imported here. Both sides are canonicalised identically and compared for byte equality
with zero tolerance. No KMP_DUPLICATE_LIB_OK: an OpenMP workaround that "may silently produce
incorrect results" cannot underwrite a byte-equivalence claim.

The evaluator-side wiring (paths A and B through `src.eval.robustness`) is deliberately NOT checked
here — importing `src.eval` pulls torch, which would defeat the isolation this file exists to prove.
`scripts/smoke_eval_robustness.py` owns those assertions.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

TORCH_FREE_BEFORE = "torch" not in sys.modules and "torchvision" not in sys.modules

from src.corruption_cache import CorruptionCacheError  # noqa: E402
from src import corruption_cache as cache  # noqa: E402
from src.vendor.imagecorruptions import (CORRUPTION_IDS, GENERATED_SEVERITIES,  # noqa: E402
                                         CorruptionInputError, apply_corruption,
                                         canonical_seed_payload, item_seed, numpy_seed)
from src.vendor.imagecorruptions import provenance as prov  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="smoke_corruption_"))
UPSTREAM_SHA = "adb5944eccfafe0118e777e3300c94420eab486556ca805ea101d3374d130cbb"
FORBIDDEN_IMPORTS = ("import torch", "from torch", "import torchvision", "from torchvision",
                     "from src.data", "import src.data")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect(name: str, exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no exception raised")
    except exc as e:
        check(name, True, getattr(e, "code", type(e).__name__))
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


def upstream_path():
    """The byte-verified pinned upstream source. Env override keeps this runnable off this box."""
    override = os.environ.get("PLANTSEG_UPSTREAM_CORRUPTIONS")
    if override:
        return override if os.path.isfile(override) else None
    dirs = sorted(glob.glob(os.path.join(os.environ["TEMP"], "iccvendor_*")))
    if not dirs:
        return None
    p = os.path.join(dirs[-1], "imagecorruptions-1.1.2", "imagecorruptions", "corruptions.py")
    return p if os.path.isfile(p) else None


def build_reference_namespace(path):
    """Exec ONLY the closure's pristine upstream segments with the minimal real dependencies."""
    from io import BytesIO

    import skimage as sk
    from PIL import Image

    raw = io.open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA:
        return None
    src = raw.decode("utf-8")
    segs = {n.name: ast.get_source_segment(src, n)
            for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    order = ["gauss_function", "getOptimalKernelWidth1D", "getMotionBlurKernel", "shift",
             "_motion_blur", "plasma_fractal", "next_power_of_2",
             "gaussian_noise", "jpeg_compression", "brightness", "fog", "motion_blur"]
    ns = {"np": np, "math": math, "Image": Image, "BytesIO": BytesIO, "sk": sk}
    for name in order:
        seg = segs[name]
        if name == "plasma_fractal":
            # execution-time alias only: NumPy 2.x removed np.float_, which named float64.
            seg = seg.replace("np.float_", "np.float64")
        exec(compile(seg, f"<upstream:{name}>", "exec"), ns)
    return ns


def canon(out, shape):
    """Mirror upstream corrupt()'s final np.uint8(...) wrap."""
    return np.ascontiguousarray(np.uint8(out), dtype=np.uint8).reshape(shape)


# ---------------------------------------------------------------- 0. torch-free isolation
def test_isolation() -> None:
    check("torch_absent_before", TORCH_FREE_BEFORE,
          "no torch/torchvision loaded when the vendor+cache imports ran")

    vendor_dir = REPO / "src" / "vendor"
    sources = {p: p.read_text(encoding="utf-8") for p in vendor_dir.rglob("*.py")}
    sources[REPO / "src" / "corruption_cache.py"] = \
        (REPO / "src" / "corruption_cache.py").read_text(encoding="utf-8")

    offenders = []
    for path, text in sources.items():
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or '"' in stripped and stripped.startswith(('"', "'")):
                continue
            if any(stripped.startswith(f) for f in FORBIDDEN_IMPORTS):
                offenders.append(f"{path.name}: {stripped}")
    check("vendor_and_cache_never_import_torch_or_src_data", not offenders,
          "; ".join(offenders) if offenders else f"{len(sources)} modules clean")

    check("cache_module_is_first_party_not_vendored",
          (REPO / "src" / "corruption_cache.py").is_file()
          and not (vendor_dir / "corruption_cache.py").exists(),
          "thesis infrastructure lives outside src/vendor")
    check("vendor_subtree_relocated",
          (vendor_dir / "imagecorruptions" / "corruptions.py").is_file()
          and not (REPO / "src" / "data" / "imagecorruptions").exists(),
          "src/vendor/imagecorruptions")
    check("no_compatibility_shim_left_behind",
          not (REPO / "src" / "data" / "corruption_cache.py").exists(),
          "old src/data paths fully removed")


# ---------------------------------------------------------------- 1. source / provenance
def test_source() -> None:
    check("exact_five_ids", CORRUPTION_IDS ==
          ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog"),
          str(CORRUPTION_IDS))
    d = prov.as_dict()
    check("upstream_identity_recorded",
          d["upstream_version"] == "1.1.2" and d["upstream_tag"] == "v1.1.2"
          and d["upstream_commit"] == "d03ee68843a9be8fda4c94a9ad8aad767a3f437e")
    check("pristine_upstream_sha_recorded",
          d["pristine_upstream_corruptions_sha256"] == UPSTREAM_SHA)
    check("patches_enumerated",
          len(d["compatibility_patches"]) == 1
          and d["compatibility_patches"][0]["change"] == "np.float_ -> np.float64"
          and d["compatibility_patches"][0]["function"] == "plasma_fractal",
          str(d["compatibility_patches"][0]["change"]))
    check("not_claimed_pristine", d["bytes_are_pristine_upstream"] is False)
    check("runtime_hash_recomputes", prov.verify_vendor_integrity() == prov.VENDOR_RUNTIME_SHA256,
          prov.VENDOR_RUNTIME_SHA256[:16] + "…")

    # non-self-referential: the digest covers corruptions.py, which never contains that digest
    runtime_text = prov.runtime_path().read_text(encoding="utf-8")
    check("checksum_not_self_referential",
          prov.VENDOR_RUNTIME_SHA256 not in runtime_text
          and prov.VENDOR_RUNTIME_FILE == "corruptions.py")

    # Two licence identities, deliberately not conflated. `.gitattributes` declares `eol=lf`, so the
    # committed/checked-out bytes are LF while the pinned commit was retrieved CRLF. A fresh clone
    # can only ever verify the VENDORED digest; the retrieval digest stays as provenance evidence.
    lic = prov.license_path()
    check("apache_license_vendored",
          lic.is_file() and "Apache License" in lic.read_text(encoding="utf-8")[:200]
          and prov.verify_vendored_license() == prov.VENDORED_LICENSE_SHA256,
          f"LF-normalised, {lic.stat().st_size} bytes")
    check("license_digest_verifiable_after_fresh_checkout",
          lic.read_bytes().count(b"\r\n") == 0
          and prov.VENDORED_LICENSE_SHA256 != prov.UPSTREAM_LICENSE_SHA256,
          "worktree bytes == git index bytes")
    check("upstream_license_retrieval_digest_preserved",
          d["upstream_license_sha256_as_retrieved"] == prov.UPSTREAM_LICENSE_SHA256
          and d["vendored_license_sha256"] == prov.VENDORED_LICENSE_SHA256
          and d["license_normalization"] == "CRLF upstream retrieval -> LF vendored text",
          "retrieval evidence kept, never relabelled as the committed file")

    # no unrelated corruption leaked into the runtime module
    unrelated = ("def snow", "def frost", "def spatter", "def glass_blur", "def elastic_transform",
                 "def zoom_blur", "def contrast", "def pixelate", "def saturate")
    check("no_unrelated_corruptions_vendored",
          all(u not in runtime_text for u in unrelated))

    # AST, not substring: the module's own docstring explains that it imports "neither cv2 nor
    # scipy", so a naive text scan fails on its own prose. Resolve real import statements instead.
    imported = set()
    for node in ast.walk(ast.parse(runtime_text)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    check("no_cv2_or_scipy_import", not ({"cv2", "scipy"} & imported),
          f"resolved imports: {sorted(imported)}")


# ---------------------------------------------------------------- 2. reference equivalence
def test_reference_equivalence() -> None:
    path = upstream_path()
    if path is None:
        check("upstream_source_available", False, "verified pinned source missing")
        return
    ref = build_reference_namespace(path)
    if ref is None:
        check("upstream_source_available", False, "upstream hash mismatch")
        return
    check("upstream_source_available", True, "isolated reference namespace built")

    rng = np.random.RandomState(0)
    images = {"square": rng.randint(0, 256, (64, 64, 3), dtype=np.uint8),
              "nonsquare": rng.randint(0, 256, (40, 72, 3), dtype=np.uint8)}
    mismatches = []
    cases = 0
    for label, img in images.items():
        for cid in CORRUPTION_IDS:
            for sev in GENERATED_SEVERITIES:
                seed = item_seed(f"img_{label}", cid, sev)
                with numpy_seed(seed):
                    r = canon(ref[cid](Image.fromarray(img.copy()), sev), img.shape)
                v, vseed = apply_corruption(img.copy(), cid, sev, image_id=f"img_{label}")
                cases += 1
                if vseed != seed or not np.array_equal(r, v):
                    diff = (r.astype(int) - v.astype(int))
                    mismatches.append((label, cid, sev, img.shape, int((diff != 0).sum()),
                                       int(np.abs(diff).max())))
            check(f"equiv_{label}_{cid}",
                  not [m for m in mismatches if m[0] == label and m[1] == cid],
                  "severities 1-4 byte-identical")
    check("reference_equivalence_all_cases", mismatches == [] and cases == 40,
          f"{cases} cases: 5 corruptions x severities 1-4 x {{square, non-square}}"
          if not mismatches else str(mismatches[:3]))


# ---------------------------------------------------------------- 3. seed policy
def test_seed_policy() -> None:
    base = item_seed("img_0001", "fog", 2)
    check("seed_deterministic", base == item_seed("img_0001", "fog", 2))
    check("seed_changes_with_image", base != item_seed("img_0002", "fog", 2))
    check("seed_changes_with_corruption", base != item_seed("img_0001", "brightness", 2))
    check("seed_changes_with_severity", base != item_seed("img_0001", "fog", 3))
    check("seed_is_uint32", 0 <= base <= 0xFFFFFFFF, str(base))
    payload = canonical_seed_payload("img_0001", "fog", 2)
    check("seed_payload_documented",
          payload == b'["plantseg-corruption-v1",42,"img_0001","fog",2]', payload.decode())
    check("image_id_normalised", item_seed("  img_0001  ", "fog", 2) == base)

    st = np.random.get_state()
    img = np.random.RandomState(1).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    apply_corruption(img, "gaussian_noise", 3, image_id="x")
    after = np.random.get_state()
    check("numpy_rng_state_restored",
          st[0] == after[0] and np.array_equal(st[1], after[1]) and st[2:] == after[2:],
          "no RNG leak to the caller")

    a, _ = apply_corruption(img, "fog", 2, image_id="a")
    _b, _ = apply_corruption(img, "fog", 2, image_id="b")
    a2, _ = apply_corruption(img, "fog", 2, image_id="a")
    check("generation_order_independent", np.array_equal(a, a2),
          "interleaving another item does not change bytes")


# ---------------------------------------------------------------- 4. input contract
def test_input_contract() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    expect("severity_5_rejected", CorruptionInputError, apply_corruption, img, "fog", 5,
           image_id="x")
    expect("severity_0_rejected", CorruptionInputError, apply_corruption, img, "fog", 0,
           image_id="x")
    expect("unknown_corruption_rejected", CorruptionInputError, apply_corruption, img, "snow", 1,
           image_id="x")
    expect("alias_rejected", CorruptionInputError, apply_corruption, img, "brightness_variation",
           1, image_id="x")
    expect("float_input_rejected", CorruptionInputError, apply_corruption,
           img.astype(np.float32), "fog", 1, image_id="x")
    expect("non_rgb_rejected", CorruptionInputError, apply_corruption,
           np.zeros((32, 32), dtype=np.uint8), "fog", 1, image_id="x")
    out, _ = apply_corruption(img, "jpeg_compression", 1, image_id="x")
    check("jpeg_canonicalised_to_uint8_array",
          isinstance(out, np.ndarray) and out.dtype == np.uint8 and out.shape == img.shape,
          "upstream returns a PIL Image")


# ---------------------------------------------------------------- 5. cache contract
def test_cache() -> None:
    rng = np.random.RandomState(3)
    images = {f"img_{i:03d}": rng.randint(0, 256, (32, 48, 3), dtype=np.uint8) for i in range(3)}
    root = TMP / "cache"
    m = cache.generate(root, images, split="test")
    check("cache_grid_size", len(m["entries"]) == 3 * 5 * 4, f"{len(m['entries'])} entries")
    check("cache_never_generates_severity_5",
          all(e["severity"] in GENERATED_SEVERITIES for e in m["entries"]))
    check("cache_records_versions",
          {"numpy", "pillow", "scikit_image", "python"} <= set(m["versions"]),
          str(m["versions"]["numpy"]))
    check("cache_records_vendor_hash",
          m["vendor"]["patched_vendor_runtime_sha256"] == prov.VENDOR_RUNTIME_SHA256)
    check("cache_records_official_expectation",
          m["official_expected_images"] == 1561 and m["source_image_count"] == 3,
          "official count is expectation metadata, not a fabricated actual")

    # lossless round-trip
    loaded = cache.load_cached(root, cache.load_manifest(root), "img_000", "fog", 2)
    direct, _ = apply_corruption(images["img_000"], "fog", 2, image_id="img_000")
    check("cache_lossless_roundtrip",
          loaded.dtype == np.uint8 and loaded.shape == direct.shape
          and np.array_equal(loaded, direct), "array-identical, never re-encoded")
    check("cache_uses_npy_not_jpeg",
          all(e["cache_relative_path"].endswith(".npy") for e in m["entries"]),
          "a second JPEG pass would confound jpeg_compression")

    # repeated generation is byte-identical
    root2 = TMP / "cache2"
    m2 = cache.generate(root2, images, split="test")
    check("regeneration_byte_identical",
          [e["corrupted_pixel_sha256"] for e in m["entries"]]
          == [e["corrupted_pixel_sha256"] for e in m2["entries"]])
    check("manifest_checksum_stable",
          m["canonical_manifest_sha256"] == m2["canonical_manifest_sha256"])

    man = cache.load_manifest(root)
    check("manifest_verifies", man["schema"] == cache.CACHE_SCHEMA)

    # the cached array is pre-padding and pre-normalization
    entry = next(e for e in man["entries"] if e["image_id"] == "img_001"
                 and e["corruption_id"] == "brightness" and e["severity"] == 1)
    check("cache_is_pre_padding_pre_normalization",
          entry["dtype"] == "uint8" and entry["shape"] == [32, 48, 3],
          "resized corrupted pixels only; no padded/normalized tensor is cached")

    # tampering / missing / foreign vendor / seed policy
    victim = root / man["entries"][0]["cache_relative_path"]
    good = np.load(victim)
    np.save(victim, (good + 1).astype(np.uint8))
    expect("tampered_item_rejected", CorruptionCacheError, cache.load_cached, root, man,
           man["entries"][0]["image_id"], man["entries"][0]["corruption_id"],
           man["entries"][0]["severity"])
    np.save(victim, good)
    victim.unlink()
    expect("missing_item_rejected", CorruptionCacheError, cache.load_cached, root, man,
           man["entries"][0]["image_id"], man["entries"][0]["corruption_id"],
           man["entries"][0]["severity"])
    expect("unknown_cell_rejected", CorruptionCacheError, cache.load_cached, root, man,
           "img_999", "fog", 2)

    foreign = json.loads((root2 / cache.MANIFEST_NAME).read_text(encoding="utf-8"))
    foreign["vendor"]["patched_vendor_runtime_sha256"] = "0" * 64
    foreign["canonical_manifest_sha256"] = cache.canonical_manifest_sha256(foreign)
    (root2 / cache.MANIFEST_NAME).write_text(json.dumps(foreign, indent=2, sort_keys=True),
                                             encoding="utf-8")
    expect("foreign_vendor_checksum_rejected", CorruptionCacheError, cache.load_manifest, root2)

    root3 = TMP / "cache3"
    cache.generate(root3, images, split="test")
    bad = json.loads((root3 / cache.MANIFEST_NAME).read_text(encoding="utf-8"))
    bad["seed_policy_id"] = "some-other-policy"
    bad["canonical_manifest_sha256"] = cache.canonical_manifest_sha256(bad)
    (root3 / cache.MANIFEST_NAME).write_text(json.dumps(bad, indent=2, sort_keys=True),
                                             encoding="utf-8")
    expect("seed_policy_mismatch_rejected", CorruptionCacheError, cache.load_manifest, root3)

    root4 = TMP / "cache4"
    cache.generate(root4, images, split="test")
    tampered = json.loads((root4 / cache.MANIFEST_NAME).read_text(encoding="utf-8"))
    tampered["source_image_count"] = 999
    (root4 / cache.MANIFEST_NAME).write_text(json.dumps(tampered, indent=2, sort_keys=True),
                                             encoding="utf-8")
    expect("manifest_tamper_rejected", CorruptionCacheError, cache.load_manifest, root4)

    check("cache_written_outside_repo", REPO not in root.resolve().parents and root.exists())


# ---------------------------------------------------------------- 6. pipeline position (text)
def test_pipeline_position() -> None:
    """Source-level checks only — importing src.eval would pull torch and break isolation."""
    rob = (REPO / "src/eval/robustness.py").read_text(encoding="utf-8")
    check("dead_end_replaced", "corruption_implementation_not_vendored" not in rob)
    check("robustness_gates_preserved",
          all(t in rob for t in ("corruption_input_not_uint8", "unknown_corruption",
                                 "severity_excluded", "rcd_stage_excluded",
                                 "INFERENTIAL_ROBUSTNESS_PAIR")),
          "vocabulary, uint8, severity, rCD and pairing gates intact")
    check("robustness_points_at_relocated_vendor",
          "src.vendor.imagecorruptions" in rob and "src.corruption_cache" in rob
          and "src.data.imagecorruptions" not in rob and "src.data.corruption_cache" not in rob)
    check("mask_never_enters_corruption",
          "mask" not in prov.runtime_path().read_text(encoding="utf-8").lower())


def main() -> int:
    print("=" * 78)
    print("CORRUPTION VENDOR SMOKE — synthetic only; no PlantSeg, no official cache")
    import PIL
    import skimage
    print(f"numpy {np.__version__} | Pillow {PIL.__version__} | scikit-image {skimage.__version__}")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_isolation, test_source, test_reference_equivalence, test_seed_policy,
               test_input_contract, test_cache, test_pipeline_position):
        print(f"\n--- {fn.__name__} ---")
        fn()

    # proven AFTER every corruption and cache operation, not just at import time
    check("torch_absent_after", "torch" not in sys.modules and "torchvision" not in sys.modules,
          "corruption + cache work never pulled the model stack")

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    print("\nNOTE: local Pillow/scikit-image versions are DEVELOPMENT evidence; OFFICIAL cache "
          "generation stays blocked until they are frozen in the official environment.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
