#!/usr/bin/env python3
"""Pre-flight gate for the SegNeXt-B / MSCAN-B teacher fine-tune. Validates; does NOT train.

This is the boundary between "everything is in place" and "start the run". It mirrors E1's
triple-gate discipline: the real fine-tune requires BOTH `--real-run` and `--confirm-real-run`, and
every non-framework precondition is proven before MMSegmentation is imported or CUDA is touched.

FAILURE ORDER is deliberate so that each guard is independently testable on a CPU box:

    authorization -> config -> data root -> split dirs -> split counts -> init checkpoint
                  -> work dir (must be OUTSIDE the repo) -> CUDA -> provenance

Every failure raises `PreflightError` with a distinct `code`, and `main()` maps that to exit 2. The
unauthorized path returns before ANY dataset access, checkpoint hashing, work-dir creation, mmseg
import or CUDA initialization.

Nothing here decodes an image, reads a mask, downloads a checkpoint, allocates a CUDA tensor, or
launches training. The actual `runner.train()` invocation is future work and is explicitly not
performed: the gate stops at ready-to-launch and reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reused unmodified from the audited E1 loop — one out-of-repo guard, not a second implementation.
from src.training.train_e1 import _assert_outside_repo  # noqa: E402

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
    try:
        resolved = _assert_outside_repo(path)      # reused E1 guard
    except RuntimeError as e:
        raise PreflightError("work_dir_inside_repo",
                             f"teacher work dir must be outside the repository: {e}") from e
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def check_cuda() -> dict:
    """Availability query only — no CUDA tensor is allocated."""
    import torch
    if not torch.cuda.is_available():
        raise PreflightError("cuda_unavailable",
                             "the real teacher fine-tune requires a CUDA GPU "
                             "(torch.cuda.is_available() is False).")
    return {"gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


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


def preflight(args) -> dict:
    """Run every gate in order. Raises `PreflightError` on the first failure."""
    config_path, config_sha = check_config(args.config)
    data_root = check_data_root(args.data_root)
    counts = check_splits(data_root)
    init_ckpt = check_init_checkpoint(args.init_ckpt)
    work_dir = check_work_dir(args.work_dir, create=args.create_work_dir)
    cuda = None if args.skip_cuda_probe else check_cuda()
    return build_provenance(config_path=config_path, config_sha=config_sha, data_root=data_root,
                            counts=counts, init_ckpt=init_ckpt, work_dir=work_dir, cuda=cuda)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pre-flight gate for the SegNeXt-B teacher fine-tune (validates; never trains).")
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

    try:
        provenance = preflight(args)
    except PreflightError as e:
        print(f"PREFLIGHT FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2

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
    print("Training is NOT started by this script: the MMSegmentation launch is a separate, "
          "explicitly authorized step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
