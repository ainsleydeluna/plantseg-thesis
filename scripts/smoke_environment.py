#!/usr/bin/env python3
"""Static verification of the official reproducibility envelope. No install, no build, no GPU.

Checks the SPECIFICATION — Dockerfile, hash-verified lock, readable registry, build context and
preflight — from the repository text, so it runs on any development machine. It deliberately does
NOT claim the container works: that requires an actual build plus `--mode gpu` on a real
accelerator, and `full_experiment_environment_validated` stays False until then.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []

SPEC_FILES = ("Dockerfile", ".dockerignore", "requirements-runpod.lock", "requirements-runpod.in",
              "requirements.lock", "scripts/preflight_environment.py", "docs/runpod_environment.md")
_absent = [f for f in SPEC_FILES if not (REPO / f).is_file()]
if _absent:
    # This suite verifies the repository SPECIFICATION, so it runs against a checkout — not inside the
    # built image, whose allow-listed context deliberately omits the Dockerfile and recipe. Say so
    # plainly instead of raising FileNotFoundError.
    print("SKIPPED: environment-specification smoke needs a full repository checkout; "
          f"missing here: {_absent}. Run it on the repository, not inside the image.")
    raise SystemExit(0)

DOCKERFILE = (REPO / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (REPO / ".dockerignore").read_text(encoding="utf-8")
RUNPOD_LOCK = (REPO / "requirements-runpod.lock").read_text(encoding="utf-8")
RUNPOD_IN = (REPO / "requirements-runpod.in").read_text(encoding="utf-8")
REGISTRY = (REPO / "requirements.lock").read_text(encoding="utf-8")
PREFLIGHT = (REPO / "scripts/preflight_environment.py").read_text(encoding="utf-8")

# Every version the thesis is pinned to. Single source for the assertions below.
REGISTERED = {
    "torch": "2.1.0+cu121", "torchvision": "0.16.0+cu121",
    "mmcv": "2.1.0", "mmsegmentation": "1.2.2", "mmengine": "0.10.7",
    "numpy": "1.26.4", "scipy": "1.11.4",
    "pillow": "12.3.0", "scikit-image": "0.23.2",
    "fvcore": "0.1.5.post20221221", "opencv-python": "4.8.1.78",
    "statsmodels": "0.14.6",
}


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def pins(text: str) -> dict:
    out = {}
    for raw in text.splitlines():
        entry = raw.strip().rstrip("\\").strip()
        if entry and not entry.startswith(("#", "--")) and "==" in entry:
            name, version = entry.split("==", 1)
            out[name.strip().lower().replace("_", "-")] = version.strip()
    return out


RUNPOD_PINS = pins(RUNPOD_LOCK)
REGISTRY_PINS = pins(REGISTRY)
IN_PINS = pins(RUNPOD_IN)


# ---------------------------------------------------------------- 1. container specification
def test_container_spec() -> None:
    check("dockerfile_exists", (REPO / "Dockerfile").is_file(), f"{len(DOCKERFILE)} bytes")
    check("dockerignore_exists", (REPO / ".dockerignore").is_file())
    m = re.search(r"^FROM\s+(\S+)@sha256:([0-9a-f]{64})\s*$", DOCKERFILE, re.M)
    check("base_pinned_by_immutable_digest", m is not None,
          f"{m.group(1)}@sha256:{m.group(2)[:16]}…" if m else "no digest-pinned FROM")
    check("base_is_python_311", m is not None and m.group(1) == "python"
          and "python:3.11-slim-bookworm" in DOCKERFILE,
          "official Python 3.11 image, not the historical pytorch image")
    check("historical_pytorch_image_rejected_with_reason",
          "PYTHON_VERSION=3.8" in DOCKERFILE and "pytorch/pytorch:2.1.0-cuda12.1" in DOCKERFILE,
          "documents why the tag-matching image is unusable")
    for var in ("PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1", "DEBIAN_FRONTEND=noninteractive"):
        check(f"env_{var.split('=')[0].lower()}", var in DOCKERFILE, var)
    check("workdir_defined", re.search(r"^WORKDIR\s+/workspace/", DOCKERFILE, re.M) is not None)
    check("lock_copied_before_source",
          DOCKERFILE.index("requirements-runpod.lock") < DOCKERFILE.index("COPY src"),
          "dependency layer caches independently of source churn")
    check("build_runs_image_preflight",
          "preflight_environment.py --mode image" in DOCKERFILE,
          "version drift fails the build")
    check("no_gpu_model_baked",
          not re.search(r"(RTX|A100|4090|H100|A6000)", DOCKERFILE),
          "GPU model is run provenance, not image content")
    check("dataset_root_mounted_not_baked",
          "PLANTSEG_DATA_ROOT=" in DOCKERFILE and "COPY data" not in DOCKERFILE
          and "COPY plantseg_data" not in DOCKERFILE)


# ---------------------------------------------------------------- 2. registered versions
def test_registered_versions() -> None:
    for dist, version in REGISTERED.items():
        check(f"lock_{dist}_is_{version}", RUNPOD_PINS.get(dist) == version,
              str(RUNPOD_PINS.get(dist)))
    check("python_311_declared",
          "python:3.11-slim-bookworm" in DOCKERFILE and 'OFFICIAL_PYTHON = "3.11"' in PREFLIGHT)


# ---------------------------------------------------------------- 3. no silent fallback
def test_no_silent_fallback() -> None:
    check("cpu_only_torch_impossible",
          RUNPOD_PINS.get("torch") == "2.1.0+cu121"
          and "download.pytorch.org/whl/cu121" in DOCKERFILE
          and 'torch_is_not_cpu_only' in PREFLIGHT,
          "the +cu121 local version and the hash both differ from any CPU wheel")
    check("hash_checking_enforced_at_install",
          "--require-hashes" in DOCKERFILE,
          "pip refuses an unpinned or altered artifact")
    check("mmcv_is_registered_binary_not_source_build",
          "download.openmmlab.com/mmcv/dist/cu121/torch2.1.0" in DOCKERFILE
          and "mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl" in
          (REPO / "docs/runpod_environment.md").read_text(encoding="utf-8"),
          "torch2.1.0/cu121/cp311 prebuilt wheel")
    check("no_source_compile_of_mmcv",
          "MMCV_WITH_OPS" not in DOCKERFILE and "pip install mmcv --no-binary" not in DOCKERFILE)
    check("torch_cuda_build_asserted",
          'OFFICIAL_TORCH_CUDA = "12.1"' in PREFLIGHT)


# ---------------------------------------------------------------- 4. genuine hashes
def test_hashes() -> None:
    hash_lines = re.findall(r"--hash=sha256:([0-9a-f]{64})", RUNPOD_LOCK)
    check("lock_contains_real_hash_constraints", len(hash_lines) >= 70, f"{len(hash_lines)} hashes")
    check("every_pin_has_a_hash", len(hash_lines) == len(RUNPOD_PINS),
          f"{len(RUNPOD_PINS)} pins / {len(hash_lines)} hashes")
    check("hashes_are_distinct", len(set(hash_lines)) == len(hash_lines),
          "no copied placeholder hash")
    check("torch_hash_matches_pytorch_index",
          "aa984599c2c4ffbc57c48d0d965cbe832e610c967e8179d4ac0a582c733fe112" in RUNPOD_LOCK,
          "independently cross-checked against download.pytorch.org")
    check("regeneration_recipe_recorded",
          "--report" in RUNPOD_LOCK and "docker run" in RUNPOD_LOCK,
          "the lock states how it was produced")
    check("lock_declares_platform_scope",
          "Linux x86-64" in RUNPOD_LOCK and "CPython 3.11" in RUNPOD_LOCK)
    check("vendored_imagecorruptions_absent_from_lock",
          "imagecorruptions" not in RUNPOD_PINS,
          "the vendored closure stays vendored")


# ---------------------------------------------------------------- 5. registry vs hashed lock
def test_no_drift() -> None:
    drift = [d for d in REGISTERED
             if d in REGISTRY_PINS and RUNPOD_PINS.get(d) != REGISTRY_PINS.get(d)]
    check("registry_and_hashed_lock_agree", not drift,
          str(drift) if drift else f"{len(REGISTERED)} load-bearing pins identical")
    in_drift = [d for d, v in IN_PINS.items()
                if d in REGISTRY_PINS and v != REGISTRY_PINS[d]
                and d not in ("torch", "torchvision")]
    check("resolver_input_matches_registry", not in_drift, str(in_drift[:3]))
    check("windows_only_excluded_from_linux_lock",
          "pywin32" in REGISTRY_PINS and "pywin32" not in RUNPOD_PINS,
          "pywin32 stays out of the Linux lock")
    check("registry_still_readable_plain_format",
          "--hash" not in REGISTRY,
          "requirements.lock remains the simple version registry its consumers expect")


# ---------------------------------------------------------------- 6. build context hygiene
def test_build_context() -> None:
    check("dockerignore_denies_by_default", DOCKERIGNORE.splitlines()[0].strip() == "**"
          or "\n**\n" in DOCKERIGNORE, "allow-list, not deny-list")
    check("protected_pdf_excluded", "docs/reference/**" in DOCKERIGNORE,
          "the reference PDF can never enter the build context")
    for pattern, label in (("**/*.pt", "checkpoints_pt"), ("**/*.pth", "checkpoints_pth"),
                           ("**/*.ckpt", "checkpoints_ckpt"), ("datasets/**", "datasets"),
                           ("plantseg_data/**", "plantseg_data"), ("**/id_rsa*", "ssh_keys"),
                           ("**/.env", "dotenv"), ("**/*.pem", "pem_keys"),
                           ("**/*.zip", "archives")):
        check(f"context_excludes_{label}", pattern in DOCKERIGNORE, pattern)
    check("only_runtime_docs_allowed",
          "!docs/IMPLEMENTATION_CONTRACT.md" in DOCKERIGNORE
          and "!docs/reference" not in DOCKERIGNORE,
          "three contract files, nothing else from docs/")


# ---------------------------------------------------------------- 7. preflight semantics
def test_preflight() -> None:
    check("preflight_has_two_modes",
          'choices=["image", "gpu"]' in PREFLIGHT,
          "image = software identity; gpu = official accelerator")
    check("gpu_mode_requires_real_device",
          'results.append(("cuda_available", available' in PREFLIGHT,
          "a CPU-only host cannot satisfy the official preflight")
    check("image_mode_cannot_claim_gpu_validation",
          "does NOT validate the official" in PREFLIGHT)
    for token in ("torch.version.cuda", "mmcv", "mmseg", "fvcore", "torchvision",
                  "supported_engines", "sys.path"):
        check(f"preflight_checks_{token.split('.')[-1].replace('(','')}", token in PREFLIGHT, token)
    check("preflight_reads_registry_not_hardcoded_twice",
          "requirements.lock" in PREFLIGHT and "registered_versions" in PREFLIGHT,
          "versions come from the registry, so they cannot drift")
    check("preflight_reproduces_corruption_closure",
          "runtime_pin_mismatches" in PREFLIGHT
          and "full_experiment_env_not_claimed_validated" in PREFLIGHT)


# ---------------------------------------------------------------- 8. validation-status honesty
def test_validation_status() -> None:
    from src.corruption_cache import official_dependency_pins

    pins_ = official_dependency_pins()
    check("corruption_closure_still_validated",
          pins_["official_dependency_environment_validated"] is True
          and pins_["pillow"] == "12.3.0" and pins_["scikit_image"] == "0.23.2",
          "0b58050 result intact")
    check("full_experiment_environment_not_validated",
          pins_["full_experiment_environment_validated"] is False,
          "no container build + GPU run has occurred")
    recipe = (REPO / "docs/runpod_environment.md").read_text(encoding="utf-8")
    check("recipe_separates_specified_from_validated",
          "CONTAINER SPECIFICATION COMPLETE" in recipe
          and "FULL OFFICIAL EXPERIMENT ENVIRONMENT EXECUTABLY VALIDATED" in recipe)
    required = {"build": "docker build", "verify_image": "--mode image",
                "gpu_preflight": "--mode gpu", "record_digest": "RepoDigests",
                "mount_dataset": "PLANTSEG_DATA_ROOT", "regenerate_lock": "--report"}
    absent = [k for k, token in required.items() if token not in recipe]
    check("recipe_covers_required_steps", not absent,
          str(absent) if absent else "build, verify, record, mount, GPU preflight, regenerate")
    check("recipe_records_hardware_separately",
          "run provenance" in recipe.lower() and "GPU model" in recipe)


def main() -> int:
    print("=" * 78)
    print("ENVIRONMENT SPECIFICATION SMOKE — static; no install, no build, no GPU")
    print("=" * 78)
    for fn in (test_container_spec, test_registered_versions, test_no_silent_fallback, test_hashes,
               test_no_drift, test_build_context, test_preflight, test_validation_status):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    print("\nNOTE: this proves the SPECIFICATION is coherent. The full official experiment "
          "environment stays UNVALIDATED until the image is built and `--mode gpu` passes on a real "
          "accelerator.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
