#!/usr/bin/env python3
"""Pre-flight gate and launcher for the SegNeXt-B / MSCAN-B teacher fine-tune.

This is the boundary between "everything is in place" and "start the run". It mirrors E1's
triple-gate discipline: the real fine-tune requires BOTH `--real-run` and `--confirm-real-run`, and
every non-framework precondition is proven before MMSegmentation is imported or CUDA is touched.

FAILURE ORDER is deliberate so that each guard is independently testable on a CPU box:

    authorization -> config -> data root -> split dirs -> split counts -> init checkpoint
                  -> work dir (absolute, OUTSIDE the repo) -> CUDA -> provenance

Every failure raises `PreflightError` with a distinct `code`, and `main()` maps that to exit 2. The
unauthorized path returns before ANY dataset access, checkpoint hashing, work-dir creation, mmseg
import or CUDA initialization.

Without `--launch`, nothing here decodes an image, reads a mask, downloads a checkpoint, allocates a
CUDA tensor, or launches training: the gate stops at ready-to-launch and reports.

`--launch` (G18) refuses `--skip-cuda-probe` and adds a gate at each end of that order:
`CUBLAS_WORKSPACE_CONFIG` must be inherited as ':4096:8' before the gates run, and the loaded config
must name what they validated after they pass. It then writes `teacher_launch_provenance.json`,
builds `TeacherRunner.from_cfg(cfg)`, registers the determinism attestation hook and calls `train()`.
On this path the CUDA gate does not query device names: `torch.cuda.get_device_name` initializes CUDA,
and the policy has to be established first. MMEngine's environment log records the names after
`TeacherRunner.set_randomness`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path

# G18 launcher-entry echo (B58 §11): the inherited value, read before this module imports torch or
# MMEngine (the `src.training.train_e1` import below loads torch). src/training/teacher_runner.py
# asserts the same name and value.
CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_REQUIRED = ":4096:8"
CUBLAS_AT_LAUNCHER_ENTRY = os.environ.get(CUBLAS_ENV)
LAUNCHER_PATH = Path(__file__).resolve()
LAUNCHER_SHA256 = hashlib.sha256(LAUNCHER_PATH.read_bytes()).hexdigest()

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reused unmodified from the audited E1 loop — one out-of-repo guard, not a second implementation —
# and its declared-image-digest reader.
from src.training.train_e1 import _assert_outside_repo, _image_digest  # noqa: E402

DEFAULT_CONFIG = (REPO / "configs" / "teacher"
                  / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py")
DATA_ROOT_ENV = "PLANTSEG_DATA_ROOT"
INIT_CKPT_ENV = "SEGNEXT_ADE20K_CKPT"
WORK_DIR_ENV = "TEACHER_WORK_DIR"

SPLIT_COUNTS = {"train": 5367, "val": 846, "test": 1561}
IMAGE_SUFFIXES = (".jpg", ".jpeg")
MASK_SUFFIX = ".png"
SENTINEL_MARKER = "NEED_TO_CONFIRM"
PUBLIC_PLANTSEG_SOURCE_COMMIT = "1a3dd4d9224bcc97a5850af7dd1c423abc24eae0"


class PreflightError(RuntimeError):
    """A named pre-flight failure. `code` makes each guard deterministically testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _reject_sentinel(value: str, code: str, what: str) -> str:
    if not value or SENTINEL_MARKER in str(value):
        raise PreflightError(code, f"{what} is unresolved ({value!r}). The config leaves it as a "
                                   f"{SENTINEL_MARKER} sentinel until it is supplied explicitly.")
    return value


def check_config(config_arg: str | None) -> tuple[Path, str]:
    path = Path(config_arg) if config_arg else DEFAULT_CONFIG
    if not path.is_file():
        raise PreflightError("config_missing", f"teacher config not found: {path}")
    return path.resolve(), _sha256(path)


def check_data_root(value: str | None) -> Path:
    raw = value if value is not None else os.environ.get(DATA_ROOT_ENV, "")
    if not raw:
        raise PreflightError("data_root_unset",
                             f"{DATA_ROOT_ENV} is not set. The PlantSeg root is supplied by "
                             "environment, never hard-coded.")
    _reject_sentinel(raw, "data_root_sentinel", DATA_ROOT_ENV)
    root = Path(raw)
    if not root.is_dir():
        raise PreflightError("data_root_missing", f"{DATA_ROOT_ENV} does not resolve: {root}")
    return root.resolve()


def check_splits(root: Path) -> dict:
    """Verify the six split dirs and their pair counts. Names only — no image or mask is opened."""
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        img_dir, mask_dir = root / "images" / split, root / "annotations" / split
        for d in (img_dir, mask_dir):
            if not d.is_dir():
                raise PreflightError("split_dir_missing", f"missing split directory: {d}")
        n_img = sum(1 for e in os.scandir(img_dir)
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_SUFFIXES)
        n_mask = sum(1 for e in os.scandir(mask_dir)
                     if e.is_file() and os.path.splitext(e.name)[1] == MASK_SUFFIX)
        expected = SPLIT_COUNTS[split]
        if n_img != expected or n_mask != expected:
            raise PreflightError(
                "split_count_mismatch",
                f"split {split}: {n_img} images / {n_mask} masks, expected {expected} of each")
        counts[split] = {"images": n_img, "masks": n_mask}
    return counts


def check_init_checkpoint(value: str | None) -> dict:
    raw = value if value is not None else os.environ.get(INIT_CKPT_ENV, "")
    if not raw:
        raise PreflightError("init_ckpt_unset",
                             f"{INIT_CKPT_ENV} is not set. The ADE20K initialization checkpoint is "
                             "REQUIRED — there is no random-init fallback and nothing is downloaded.")
    _reject_sentinel(raw, "init_ckpt_sentinel", INIT_CKPT_ENV)
    path = Path(raw)
    if not path.is_file():
        raise PreflightError("init_ckpt_missing",
                             f"ADE20K initialization checkpoint not found: {path}. Obtain it "
                             "separately (docs/teacher_prep_runbook.md §4); this gate never downloads.")
    resolved = path.resolve()
    try:
        _assert_outside_repo(resolved.parent)
    except RuntimeError as e:
        raise PreflightError("init_ckpt_inside_repo",
                             f"checkpoints must live outside the repository: {e}") from e
    return {"path": str(resolved), "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def check_work_dir(value: str | None, create: bool = False) -> Path:
    raw = value if value is not None else os.environ.get(WORK_DIR_ENV, "")
    if not raw:
        raise PreflightError("work_dir_unset", f"{WORK_DIR_ENV} is not set.")
    _reject_sentinel(raw, "work_dir_sentinel", WORK_DIR_ENV)
    path = Path(raw)
    if not path.is_absolute():
        raise PreflightError("work_dir_relative",
                             f"{WORK_DIR_ENV} must be an absolute path, got {raw!r}.")
    try:
        resolved = _assert_outside_repo(path)      # reused E1 guard
    except RuntimeError as e:
        raise PreflightError("work_dir_inside_repo",
                             f"teacher work dir must be outside the repository: {e}") from e
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def check_cuda(record_names: bool = True) -> dict:
    """Availability query only — no CUDA tensor is allocated.

    `is_available()` and `device_count()` leave PyTorch's CUDA state uninitialized; `get_device_name()`
    initializes it. `--launch` passes `record_names=False` so that `TeacherRunner.set_randomness` still
    finds CUDA uninitialized (G18)."""
    import torch
    if not torch.cuda.is_available():
        raise PreflightError("cuda_unavailable",
                             "the real teacher fine-tune requires a CUDA GPU "
                             "(torch.cuda.is_available() is False).")
    count = torch.cuda.device_count()
    return {"gpu_count": count,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(count)] if record_names else None}


def _optional_version(dist: str) -> str | None:
    """Version via package metadata — never imports the package."""
    try:
        return dist_version(dist)
    except PackageNotFoundError:
        return None


def build_provenance(*, config_path: Path, config_sha: str, data_root: Path, counts: dict,
                     init_ckpt: dict, work_dir: Path, cuda: dict | None) -> dict:
    import torch
    return {
        "stage": "teacher",
        "architecture": "SegNeXt-B / MSCAN-B",
        "protocol_classification": "thesis-derived",
        "public_plantseg_source_commit": PUBLIC_PLANTSEG_SOURCE_COMMIT,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "plantseg_root": str(data_root),
        "split_counts": counts,
        "ade20k_init_checkpoint": init_ckpt,
        "work_dir": str(work_dir),
        "seed": 42,
        "optimizer": "AdamW lr=6e-5 wd=0.01 betas=(0.9,0.999) head_lr_mult=10",
        "max_iters": 40000,
        "val_interval": 10000,
        "preprocessing": "512x512 aspect-preserving resize + ImageNet-mean-equivalent pad "
                         "(post-normalisation), ignore_index=255",
        "augmentation_source": "public-plantseg-segnext-family",
        "loss": "CrossEntropyLoss only",
        "checkpoint_selection": "validation mIoU (test split never used)",
        "versions": {                       # None where genuinely unavailable — never fabricated
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mmsegmentation": _optional_version("mmsegmentation"),
            "mmcv": _optional_version("mmcv"),
        },
        "cuda": cuda,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def preflight(args, record_gpu_names: bool = True) -> dict:
    """Run every gate in order. Raises `PreflightError` on the first failure."""
    config_path, config_sha = check_config(args.config)
    data_root = check_data_root(args.data_root)
    counts = check_splits(data_root)
    init_ckpt = check_init_checkpoint(args.init_ckpt)
    work_dir = check_work_dir(args.work_dir, create=args.create_work_dir)
    cuda = None if args.skip_cuda_probe else check_cuda(record_gpu_names)
    return build_provenance(config_path=config_path, config_sha=config_sha, data_root=data_root,
                            counts=counts, init_ckpt=init_ckpt, work_dir=work_dir, cuda=cuda)


# ---------------------------------------------------------------- G18: --launch only
def _exec_environ_value(name: str) -> tuple[bool, str | None]:
    """(readable, value) of `name` in /proc/self/environ, the environment this process was exec'd with.
    Later os.environ assignments do not change that block, so it separates an inherited value from one
    set in-process. It is not readable off Linux."""
    try:
        block = Path("/proc/self/environ").read_bytes()
    except OSError:
        return False, None
    prefix = name.encode() + b"="
    for entry in block.split(b"\0"):
        if entry.startswith(prefix):
            return True, entry[len(prefix):].decode("utf-8", "replace")
    return True, None


def check_inherited_cublas() -> None:
    """The cuBLAS value must come from the process environment (G8 image ENV), never from this process."""
    readable, at_exec = _exec_environ_value(CUBLAS_ENV)
    if CUBLAS_AT_LAUNCHER_ENTRY != CUBLAS_REQUIRED or (readable and at_exec != CUBLAS_REQUIRED):
        seen = at_exec if readable else "unreadable"
        raise PreflightError("cublas_not_inherited",
                             f"{CUBLAS_ENV} must be inherited as {CUBLAS_REQUIRED!r}: launcher entry "
                             f"{CUBLAS_AT_LAUNCHER_ENTRY!r}, exec environment {seen!r}.")


def _checkout_head() -> dict:
    """`git rev-parse HEAD` of the checkout this launcher runs from. `_git_provenance()` is not reused: it
    prefers the image's PLANTSEG_GIT_COMMIT, which names the image build, not a mounted checkout."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                           timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        return {"commit": None, "error": repr(e)}
    if r.returncode != 0:
        return {"commit": None, "error": r.stderr.strip()}
    return {"commit": r.stdout.strip(), "error": None}


def load_launch_config(provenance: dict):
    """Load the config as MMEngine will, and prove it names what the gates validated."""
    from mmengine.config import Config
    cfg = Config.fromfile(provenance["config_path"])

    def resolves_to(value, expected: str) -> bool:
        return isinstance(value, str) and bool(value) and Path(value).resolve() == Path(expected)

    problems = []
    if cfg.get("randomness") != dict(seed=42, deterministic=True):
        problems.append(f"randomness={cfg.get('randomness')!r} (contract B6: seed 42, deterministic)")
    if cfg.get("resume") is not False:
        problems.append(f"resume={cfg.get('resume')!r} (official runs never resume)")
    work_dir = cfg.get("work_dir")
    if not (resolves_to(work_dir, provenance["work_dir"]) and Path(work_dir).is_absolute()):
        problems.append(f"work_dir={work_dir!r}")
    load_from = cfg.get("load_from")
    if not resolves_to(load_from, provenance["ade20k_init_checkpoint"]["path"]):
        problems.append(f"load_from={load_from!r}")
    for loader in ("train_dataloader", "val_dataloader"):
        data_root = cfg.get(loader, {}).get("dataset", {}).get("data_root")
        if not resolves_to(data_root, provenance["plantseg_root"]):
            problems.append(f"{loader}.dataset.data_root={data_root!r}")
    if problems:
        raise PreflightError("launch_config_mismatch",
                             "the loaded config does not name what the gates validated: "
                             + "; ".join(problems))
    return cfg


def launch(cfg, provenance: dict) -> int:
    """Write the launch record, then TeacherRunner.from_cfg(cfg) -> attestation hook -> train()."""
    import torch
    from src.training.teacher_runner import (MODULE_PROVENANCE, TeacherDeterminismAttestationHook,
                                             TeacherRunner)
    readable, at_exec = _exec_environ_value(CUBLAS_ENV)
    record = {                  # in-process facts only; image and host facts belong to Step 0
        "stage": "teacher",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cublas_at_launcher_entry": CUBLAS_AT_LAUNCHER_ENTRY,
        "cublas_exec_environ": at_exec if readable else None,
        "exec_environ_readable": readable,
        "launcher": {"path": str(LAUNCHER_PATH), "sha256": LAUNCHER_SHA256},
        "teacher_runner_module": MODULE_PROVENANCE,
        "config": {"path": provenance["config_path"], "sha256": provenance["config_sha256"]},
        "checkout_git_head": _checkout_head(),
        "image_env_plantseg_git_commit": os.environ.get("PLANTSEG_GIT_COMMIT"),
        "image_digest_declared": _image_digest(),
        "requested_randomness": dict(cfg.randomness),
        "requested_env_cfg_cudnn_benchmark": cfg.get("env_cfg", {}).get("cudnn_benchmark"),
        "cuda_initialized_before_runner": torch.cuda.is_initialized(),
        "preflight": provenance,
    }
    work_dir = Path(provenance["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "teacher_launch_provenance.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[launch record] {out}")
    runner = TeacherRunner.from_cfg(cfg)
    runner.register_hook(TeacherDeterminismAttestationHook())
    # Runner.__init__ logged the hook order before this hook existed (runner.py:444-445); log it again.
    runner.logger.info(f"Hooks after G18 registration:\n{runner.get_hooks_info()}")
    runner.train()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pre-flight gate for the SegNeXt-B teacher fine-tune (validates; trains only with "
                    "--launch).")
    p.add_argument("--real-run", action="store_true", help="intent to launch the real fine-tune")
    p.add_argument("--confirm-real-run", action="store_true", help="explicit confirmation gate")
    p.add_argument("--config", default=None, help=f"teacher config (default: {DEFAULT_CONFIG.name})")
    p.add_argument("--data-root", default=None, help=f"overrides ${DATA_ROOT_ENV}")
    p.add_argument("--init-ckpt", default=None, help=f"overrides ${INIT_CKPT_ENV}")
    p.add_argument("--work-dir", default=None, help=f"overrides ${WORK_DIR_ENV}")
    p.add_argument("--create-work-dir", action="store_true",
                   help="create the work dir once every earlier gate has passed")
    p.add_argument("--skip-cuda-probe", action="store_true",
                   help="skip ONLY the CUDA availability query (for CPU-side gate verification); "
                        "it does not authorize a real run")
    p.add_argument("--launch", action="store_true",
                   help="after every gate passes, build TeacherRunner from the config and call train() "
                        "(G18); refuses --skip-cuda-probe")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- authorization first: return before touching anything at all ----
    if not args.real_run:
        print("REFUSING: the teacher fine-tune requires --real-run --confirm-real-run. Nothing was "
              "read, hashed, created or imported.", file=sys.stderr)
        return 2
    if not args.confirm_real_run:
        print("REFUSING: --real-run requires --confirm-real-run.", file=sys.stderr)
        return 2
    if args.launch and args.skip_cuda_probe:
        print("REFUSING: --launch cannot skip the CUDA gate (--skip-cuda-probe).", file=sys.stderr)
        return 2

    try:
        if args.launch:
            check_inherited_cublas()
        provenance = preflight(args, record_gpu_names=not args.launch)
        cfg = load_launch_config(provenance) if args.launch else None
    except PreflightError as e:
        print(f"PREFLIGHT FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2

    if args.launch:
        return launch(cfg, provenance)

    work_dir = Path(provenance["work_dir"])
    if args.create_work_dir:
        out = work_dir / "teacher_preflight_provenance.json"
        out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"[provenance] {out}")
    print(json.dumps(provenance, indent=2))
    if provenance["cuda"] is None:
        print("\nPREFLIGHT PARTIAL — every non-GPU precondition is satisfied, but the CUDA check "
              "was SKIPPED (--skip-cuda-probe). This does NOT clear the run for launch.")
    else:
        print("\nPREFLIGHT PASS — teacher fine-tune preconditions satisfied.")
    print("Training is NOT started without --launch: the MMSegmentation launch is a separate, "
          "explicitly authorized step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
