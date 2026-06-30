#!/usr/bin/env python3
"""B20a — Strict-G0 compute-platform verification (diagnostic only; NOT training).

Captures the ACTUAL local environment and decides whether it is suitable for the E1 DRY-RUN only or
for the REAL E1 training run. Side-effect-free w.r.t. the repo: it performs NO downloads, NO installs,
NO GPU compute (availability queries only), writes NO repo files and NO repo checkpoints. The only
disk writes are a temporary checkpoint dir (OUTSIDE the repo) used by the tiny train_e1 --dry-run
subprocess. Missing optional/pinned packages (mmcv, mmseg, cv2, albumentations) are reported as
"not installed" and never crash the script. Author reports/platform_verify.md from this output.
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MOBILENET_CKPT = "mobilenet_v3_large-5c1a4163.pth"      # torchvision IMAGENET1K_V2 backbone file
IMAGENET_ALIAS = "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"


def _run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        return r.stdout.rstrip("\n") if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def git_info() -> dict:
    porcelain = _run(["git", "status", "--porcelain"])
    modified, untracked = [], []
    if porcelain:
        for line in porcelain.splitlines():
            if not line:
                continue
            (untracked if line[:2] == "??" else modified).append(line[3:])  # paths only, never contents
    return {
        "head": _run(["git", "rev-parse", "HEAD"]) or "UNKNOWN",
        "short": _run(["git", "rev-parse", "--short", "HEAD"]) or "UNKNOWN",
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "UNKNOWN",
        "dirty": bool(porcelain),
        "modified": modified,
        "untracked": untracked,
    }


def import_version(module_name: str, dist_name: str | None = None):
    """Return version string, or None if the package is not installed (never raises)."""
    try:
        m = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 -- not installed / import error
        return None
    v = getattr(m, "__version__", None)
    if v is None and dist_name:
        try:
            v = dist_version(dist_name)
        except PackageNotFoundError:
            v = None
    return v or "installed (version unknown)"


def parse_lock() -> dict:
    pins = {}
    lock = REPO / "requirements.lock"
    if lock.exists():
        for line in lock.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, ver = line.split("==", 1)
                pins[name.lower()] = ver
    return pins


def main() -> int:
    pins = parse_lock()
    print("=" * 78)
    print("B20a PLATFORM VERIFICATION (diagnostic only — no training, no download, no GPU compute)")
    print("=" * 78)

    # (1) python executable + version, (2) OS/platform, (3) repo root
    print("\n[1-3] Python / OS / repo")
    print(f"  python_executable : {sys.executable}")
    print(f"  python_version    : {platform.python_version()}")
    print(f"  os_system         : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  machine           : {platform.machine()}")
    print(f"  platform          : {platform.platform()}")
    print(f"  repo_root         : {REPO}")

    # (4) git HEAD + dirty status (paths only)
    g = git_info()
    print("\n[4] git")
    print(f"  HEAD   : {g['short']} ({g['head']}) on {g['branch']} | dirty={g['dirty']}")
    print(f"  modified ({len(g['modified'])}): {g['modified']}")
    print(f"  untracked ({len(g['untracked'])}): {g['untracked']}")

    # (5-11) torch / torchvision / CUDA / GPU / cuDNN / CPU threads
    print("\n[5-11] torch / CUDA / GPU / threads")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"  torch import FAILED: {type(e).__name__}: {e}")
        print("\nVERDICT: FAIL (torch unavailable) -> NOT OK for real E1 training")
        return 1
    torch_ver = torch.__version__
    tv_ver = import_version("torchvision")
    cuda = bool(torch.cuda.is_available())
    cuda_ver = torch.version.cuda
    gpu_count = torch.cuda.device_count() if cuda else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda else []
    cudnn_avail = bool(torch.backends.cudnn.is_available())
    cudnn_ver = torch.backends.cudnn.version() if cudnn_avail else None
    print(f"  torch_version       : {torch_ver}")
    print(f"  torchvision_version : {tv_ver}")
    print(f"  cuda_available      : {cuda}")
    print(f"  cuda_version        : {cuda_ver}")
    print(f"  gpu_count           : {gpu_count}")
    print(f"  gpu_names           : {gpu_names}")
    print(f"  cudnn_available     : {cudnn_avail}")
    print(f"  cudnn_version       : {cudnn_ver}")
    print(f"  cpu_count(logical)  : {os.cpu_count()}")
    print(f"  torch_num_threads   : {torch.get_num_threads()}")

    # (12) key package versions (actual vs pinned); missing optional pkgs => "not installed"
    print("\n[12] packages (actual vs requirements.lock pin)")
    pkgs = [
        ("torch", "torch", "torch"),
        ("torchvision", "torchvision", "torchvision"),
        ("numpy", "numpy", "numpy"),
        ("scipy", "scipy", "scipy"),
        ("Pillow (PIL)", "PIL", "pillow"),
        ("opencv (cv2)", "cv2", "opencv-python"),
        ("albumentations", "albumentations", "albumentations"),
        ("mmcv", "mmcv", "mmcv"),
        ("mmseg", "mmseg", "mmsegmentation"),
        ("statsmodels", "statsmodels", "statsmodels"),
    ]
    pkg_rows = []
    for label, mod, distname in pkgs:
        actual = import_version(mod, distname)
        actual_str = actual if actual is not None else "not installed"
        pinned = pins.get(distname.lower(), "—")
        pkg_rows.append((label, actual_str, pinned))
        print(f"  {label:16}: actual={actual_str:28} pinned={pinned}")
    missing = [label for (label, a, _) in pkg_rows if a == "not installed"]

    # (13-14) torch hub cache + ImageNet checkpoint cached?
    print("\n[13-14] torch hub cache / ImageNet weights")
    hub_dir = torch.hub.get_dir()
    ckpt_path = os.path.join(hub_dir, "checkpoints", MOBILENET_CKPT)
    cached = os.path.exists(ckpt_path)
    print(f"  hub_dir          : {hub_dir}")
    print(f"  imagenet_ckpt    : {ckpt_path}")
    print(f"  imagenet_cached  : {cached}")

    # (15) random-init student build + tiny CPU forward (NO download)
    print("\n[15] build_student(pretrained=False) — CPU random init")
    try:
        from src.models.student import build_student
        m = build_student(pretrained=False)
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(1, 3, 64, 64))
        student_ok = (tuple(y.shape)[:2] == (1, 116) and bool(torch.isfinite(y).all())
                      and m.used_pretrained is False)
        student_detail = f"out={tuple(y.shape)} used_pretrained={m.used_pretrained}"
    except Exception as e:  # noqa: BLE001
        student_ok = False
        student_detail = f"{type(e).__name__}: {e}"
    print(f"  student_ok       : {student_ok} ({student_detail})")

    # (16) ImageNet pretrained build — ONLY if cached (else SKIP to avoid download)
    print("\n[16] build_student(IMAGENET1K_V2) — skipped unless cached")
    if cached:
        try:
            mi = build_student(pretrained=IMAGENET_ALIAS)
            imagenet_skipped = False
            imagenet_status = f"loaded from cache (used_pretrained={mi.used_pretrained}) — no download"
        except Exception as e:  # noqa: BLE001
            imagenet_skipped = True
            imagenet_status = f"cached but load failed: {type(e).__name__}: {e}"
    else:
        imagenet_skipped = True
        imagenet_status = "SKIPPED to avoid download (checkpoint not cached)"
    print(f"  imagenet_skipped : {imagenet_skipped} ({imagenet_status})")

    # (17) tiny train_e1.py --dry-run subprocess -> temp ckpt dir OUTSIDE repo
    print("\n[17] train_e1.py --dry-run subprocess (tiny; temp checkpoint dir outside repo)")
    tmp_ckpt = Path(tempfile.mkdtemp(prefix="verify_env_dryrun_ckpt_")).resolve()
    repo_guard_ok = not (tmp_ckpt == REPO or REPO in tmp_ckpt.parents)
    cmd = [sys.executable, str(REPO / "src" / "training" / "train_e1.py"), "--dry-run",
           "--batch-size", "1", "--max-iters", "2", "--val-interval", "2",
           "--max-val-batches", "1", "--ckpt-dir", str(tmp_ckpt)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600, env=env)
        dry_rc = r.returncode
        result_line = next((ln for ln in reversed(r.stdout.splitlines())
                            if ln.startswith("RESULT:")), "RESULT: (not found)")
    except Exception as e:  # noqa: BLE001
        dry_rc = -1
        result_line = f"RESULT: (subprocess error {type(e).__name__}: {e})"
    tmp_ckpts = sorted(str(p) for p in tmp_ckpt.glob("*.pt"))
    repo_pt = sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*.pt") if ".git" not in p.parts)
    dry_ok = (dry_rc == 0 and "RESULT: PASS" in result_line)
    print(f"  command          : {' '.join(cmd)}")
    print(f"  temp_ckpt_dir    : {tmp_ckpt}")
    print(f"  temp_dir_outside_repo : {repo_guard_ok}")
    print(f"  return_code      : {dry_rc}")
    print(f"  {result_line}")
    print(f"  temp_ckpt_files  : {tmp_ckpts}")
    print(f"  repo_.pt_files   : {repo_pt}  (MUST be empty)")
    print(f"  no_repo_checkpoint : {len(repo_pt) == 0}")

    # (18) verdict
    print("\n[18] VERDICT")
    real_ready = cuda and gpu_count >= 1 and student_ok and dry_ok
    if not (student_ok and dry_ok):
        verdict, label = "FAIL", "NOT OK for real E1 training (E1 scaffold not runnable on this machine)"
    elif real_ready:
        verdict, label = "PASS", "OK for real E1 training"
    else:
        verdict, label = "PARTIAL", "OK for DRY-RUN only; NOT OK for real E1 training"

    blockers = []
    if not cuda:
        blockers.append(f"No CUDA GPU (torch.cuda.is_available()=False; local torch is CPU-only "
                        f"build '{torch_ver}'). Real E1 (80k iters @ bs16/512^2) needs a GPU.")
    if pins.get("torch") and pins["torch"] not in torch_ver:
        blockers.append(f"torch mismatch vs pinned: local '{torch_ver}' != requirements.lock "
                        f"'{pins['torch']}' (cu121 GPU training stack).")
    if not cached:
        blockers.append("ImageNet backbone not cached; real run with --init imagenet needs network "
                        "to populate the torch-hub cache OR a pre-staged checkpoint.")
    if missing:
        blockers.append(f"Not installed locally: {', '.join(missing)} "
                        "(teacher/aug stack — needed for teacher fine-tune / E2-E3, not E1 supervised).")
    if not repo_guard_ok or repo_pt:
        blockers.append("Checkpoint isolation problem (temp dir inside repo OR a .pt found in repo).")

    print(f"  verdict          : {verdict}")
    print(f"  environment_label: {label}")
    print(f"  can_run_real_E1  : {real_ready}")
    print("  blockers_before_real_E1:")
    for b in blockers:
        print(f"    - {b}")
    if not blockers:
        print("    - (none)")

    print("\n" + "=" * 78)
    print(f"RESULT: {verdict} | {label}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
