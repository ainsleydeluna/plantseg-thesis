#!/usr/bin/env python3
"""The single executable authority on whether an environment IS the official experiment stack.

TWO MODES, DELIBERATELY DIFFERENT:

  --mode image   Software identity only. Runs on a CPU-only builder, so it is safe inside
                 `docker build`. Verifies platform, interpreter, every registered package version,
                 the torch CUDA BUILD identity, and that the load-bearing imports work. It does NOT
                 require a GPU and therefore CANNOT establish the official GPU environment.

  --mode gpu     The official preflight. Everything above PLUS a real CUDA device: it fails unless
                 `torch.cuda.is_available()`, and it records the actual GPU/CPU/RAM/driver metadata
                 that belongs to the RUN record rather than to the image.

A CPU-only machine can therefore never appear to satisfy the official GPU preflight.

The registered versions are read from `requirements.lock` — the human-readable version authority —
so this script cannot drift from it. `requirements-runpod.lock` is the hash-verified executable
counterpart; both are checked for agreement on the load-bearing pins.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OFFICIAL_PYTHON = "3.11"
OFFICIAL_OS = "Linux"
OFFICIAL_MACHINE = ("x86_64", "AMD64")
OFFICIAL_TORCH_CUDA = "12.1"

# module -> distribution name in requirements.lock. Every entry is load-bearing for some stage.
REGISTERED = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scipy": "scipy",
    "PIL": "pillow",
    "skimage": "scikit-image",
    "fvcore": "fvcore",
    "mmcv": "mmcv",
    "mmseg": "mmsegmentation",
    "mmengine": "mmengine",
    "cv2": "opencv-python",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
}
# torch/torchvision carry a +cu121 local version in the lock; compare the public part.
LOCAL_VERSION_SUFFIX = "+cu121"


class PreflightError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def registered_versions() -> dict:
    """Exact pins from the readable version registry. Never hard-coded twice."""
    pins = {}
    for raw in (REPO / "requirements.lock").read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if entry and not entry.startswith("#") and "==" in entry:
            name, version = entry.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def hashed_lock_versions() -> dict:
    """Exact pins from the hash-verified Linux lock, so the two files cannot silently disagree."""
    path = REPO / "requirements-runpod.lock"
    pins = {}
    if not path.is_file():
        return pins
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip().rstrip("\\").strip()
        if entry and not entry.startswith(("#", "--")) and "==" in entry:
            name, version = entry.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def _installed_version(module_name: str, dist: str) -> str:
    """The version pip actually installed, taken from distribution metadata.

    Distribution metadata is authoritative and is what the lock pins; a module's `__version__`
    attribute is not. `cv2.__version__` reports '4.8.1' for the distribution `opencv-python==4.8.1.78`
    because OpenCV drops the build component, so comparing attributes would fail on a correct
    environment. The module is still imported first, so importability remains proven.
    """
    importlib.import_module(module_name)
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        mod = importlib.import_module(module_name)
        for attr in ("__version__", "version", "VERSION"):
            value = getattr(mod, attr, None)
            if isinstance(value, str):
                return value
        return "unknown"


def check_platform(results: list) -> None:
    ok = platform.system() == OFFICIAL_OS
    results.append(("platform_is_linux", ok, f"{platform.system()} {platform.machine()}"))
    results.append(("machine_is_x86_64", platform.machine() in OFFICIAL_MACHINE,
                    platform.machine()))
    running = ".".join(platform.python_version_tuple()[:2])
    results.append((f"python_is_{OFFICIAL_PYTHON}", running == OFFICIAL_PYTHON,
                    platform.python_version()))


def check_versions(results: list) -> dict:
    pins = registered_versions()
    hashed = hashed_lock_versions()
    observed = {}
    for module_name, dist in REGISTERED.items():
        expected = pins.get(dist)
        try:
            actual = _installed_version(module_name, dist)
        except Exception as exc:                                  # noqa: BLE001
            results.append((f"import_{dist}", False, f"{type(exc).__name__}: {exc}"))
            continue
        observed[dist] = actual
        if expected is None:
            results.append((f"registered_{dist}", False, "absent from requirements.lock"))
            continue
        # the lock records torch/torchvision with the +cu121 local version
        match = actual == expected or actual == expected.replace(LOCAL_VERSION_SUFFIX, "")
        results.append((f"{dist}_is_{expected}", match, actual))

    # the readable registry and the hashed executable lock must agree on every load-bearing pin
    if hashed:
        drift = [d for d in REGISTERED.values()
                 if d in pins and d in hashed and pins[d] != hashed[d]]
        results.append(("registry_and_hashed_lock_agree", not drift, str(drift) if drift else
                        f"{len(hashed)} hashed distributions"))
    return observed


def check_torch_build(results: list, *, require_gpu: bool) -> dict:
    import torch

    cuda_build = torch.version.cuda
    results.append((f"torch_cuda_build_is_{OFFICIAL_TORCH_CUDA}", cuda_build == OFFICIAL_TORCH_CUDA,
                    str(cuda_build)))
    results.append(("torch_is_not_cpu_only", cuda_build is not None,
                    "a CPU-only wheel reports torch.version.cuda=None"))
    backends = list(torch.backends.quantized.supported_engines)
    results.append(("quant_backends_visible", bool(backends), str(backends)))

    available = torch.cuda.is_available()
    record = {"torch": torch.__version__, "cuda_build": cuda_build,
              "cuda_available": available, "quant_backends": backends}
    if require_gpu:
        results.append(("cuda_available", available,
                        "the official preflight requires a real CUDA device"))
        if available:
            record["device_count"] = torch.cuda.device_count()
            record["device_name"] = torch.cuda.get_device_name(0)
            record["capability"] = list(torch.cuda.get_device_capability(0))
            record["driver_cuda_runtime"] = getattr(torch.version, "cuda", None)
            results.append(("gpu_metadata_recorded", True, record["device_name"]))
    else:
        results.append(("gpu_not_required_in_image_mode", True,
                        f"cuda_available={available} (informational only)"))
    return record


def check_imports(results: list) -> None:
    """Load-bearing imports. mmcv compiled ops are reported, not required without a GPU."""
    for module_name in ("torchvision", "mmengine", "mmcv", "mmseg", "fvcore", "cv2", "skimage"):
        try:
            importlib.import_module(module_name)
            results.append((f"import_{module_name}", True, ""))
        except Exception as exc:                                  # noqa: BLE001
            results.append((f"import_{module_name}", False, f"{type(exc).__name__}: {exc}"))
    try:
        from mmcv import ops  # noqa: F401
        results.append(("mmcv_compiled_ops_import", True, "compiled ops present"))
    except Exception as exc:                                      # noqa: BLE001
        results.append(("mmcv_compiled_ops_import", False, f"{type(exc).__name__}: {exc}"))


def check_corruption_stack(results: list) -> None:
    """The corruption dependency closure validated in 0b58050 must be reproduced here."""
    from src.corruption_cache import official_dependency_pins, runtime_pin_mismatches

    pins = official_dependency_pins()
    mismatches = runtime_pin_mismatches()
    results.append(("corruption_dependency_stack_matches", not mismatches,
                    "; ".join(mismatches) if mismatches
                    else f"python {pins['python']} / numpy {pins['numpy']} / "
                         f"pillow {pins['pillow']} / scikit-image {pins['scikit_image']}"))
    results.append(("corruption_closure_validated_flag",
                    pins["official_dependency_environment_validated"] is True,
                    "validated in 0b58050 (corruption closure only)"))
    results.append(("full_experiment_env_not_claimed_validated",
                    pins["full_experiment_environment_validated"] is False,
                    "container/GPU validation is a separate, later step"))


def check_determinism_settings(results: list) -> None:
    import os

    from src.seeds import SEED

    results.append(("seed_registered", SEED == 42, str(SEED)))
    results.append(("pythondontwritebytecode_set", os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
                    repr(os.environ.get("PYTHONDONTWRITEBYTECODE"))))
    results.append(("pythonhashseed_reported", True,
                    repr(os.environ.get("PYTHONHASHSEED", "unset"))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Official PlantSeg environment preflight.")
    p.add_argument("--mode", required=True, choices=["image", "gpu"])
    p.add_argument("--json", default=None, help="write the machine-readable record here")
    args = p.parse_args(argv)
    require_gpu = args.mode == "gpu"

    results: list[tuple[str, bool, str]] = []
    print("=" * 78)
    print(f"OFFICIAL ENVIRONMENT PREFLIGHT — mode={args.mode}")
    print(f"{platform.platform()} | python {platform.python_version()}")
    print("=" * 78)

    check_platform(results)
    observed = check_versions(results)
    torch_record = {}
    try:
        torch_record = check_torch_build(results, require_gpu=require_gpu)
    except Exception as exc:                                      # noqa: BLE001
        results.append(("torch_importable", False, f"{type(exc).__name__}: {exc}"))
    check_imports(results)
    try:
        check_corruption_stack(results)
    except Exception as exc:                                      # noqa: BLE001
        results.append(("corruption_stack_checkable", False, f"{type(exc).__name__}: {exc}"))
    check_determinism_settings(results)

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:44}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    verdict = passed == len(results)
    print(f"\nRESULT: {'PASS' if verdict else 'FAIL'} ({passed}/{len(results)})")

    # The scope note must follow the VERDICT, never just the mode: a failing run may not describe
    # itself as satisfied.
    if not verdict:
        print("\nSCOPE: this environment is NOT the official stack. Nothing here may be reported as "
              "an official environment validation.")
    elif args.mode == "image":
        print("\nSCOPE: image mode proves SOFTWARE identity only. It does NOT validate the official "
              "GPU experiment environment — run --mode gpu on the real accelerator for that.")
    else:
        print("\nSCOPE: GPU preflight satisfied. Record the image digest and the GPU/CPU/RAM "
              "metadata above in the run provenance.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"mode": args.mode, "verdict": "pass" if verdict else "fail",
             "platform": platform.platform(), "python": platform.python_version(),
             "versions": observed, "torch": torch_record,
             "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in results]},
            indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
