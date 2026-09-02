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

# Highest CUDA compute capability the PINNED stack can emit kernels for. torch 2.1.0+cu121 ships
# cubins/PTX for sm_50..sm_90 only; a Blackwell-class device (sm_100/sm_120) has no compatible
# kernel and fails at the FIRST kernel launch, after the pod is already provisioned and paid for.
MAX_SM = (9, 0)


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

    # Compute-capability gate: FAIL (not warn) above MAX_SM — the pinned stack has no kernel.
    gpu_caps = [torch.cuda.get_device_capability(i) for i in range(gpu_count)] if cuda else []
    unsupported = [(i, gpu_names[i], gpu_caps[i]) for i in range(gpu_count)
                   if gpu_caps[i] > MAX_SM]
    cap_ok = not unsupported
    print(f"  gpu_capabilities    : {[f'sm_{a}{b}' for a, b in gpu_caps]} "
          f"(max supported by the pinned stack: sm_{MAX_SM[0]}{MAX_SM[1]})")
    if not cuda:
        print("  capability_gate     : SKIPPED (no CUDA device visible)")
    elif cap_ok:
        print("  capability_gate     : PASS")
    else:
        for i, name, (a, b) in unsupported:
            print(f"  capability_gate     : FAIL — cuda:{i} '{name}' is sm_{a}{b}, above the "
                  f"sm_{MAX_SM[0]}{MAX_SM[1]} ceiling of torch {torch_ver} / "
                  f"torchvision {tv_ver}. Blackwell-class pods (RTX 5090, RTX Pro 6000, B200, "
                  f"B300) are INCOMPATIBLE with the pinned stack and will abort at the first "
                  f"CUDA kernel launch. Choose an Ada/Hopper/Ampere pod (sm_80–sm_90, e.g. "
                  f"A100 / H100 / L40S / RTX 4090) or re-pin the stack.")

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

    # (17b) dataset: root resolves, all six split dirs exist, per-split PAIR counts match the
    # single source of truth in configs/data.py. Index/stat only — NO image is decoded.
    # Without this, verify_env could print "OK for real E1 training" and the run would then die
    # inside PlantSegDataset.__init__ after the pod was already provisioned.
    print("\n[17b] dataset (root / split dirs / per-split pair counts vs configs/data.py SPLIT_SIZES)")
    from configs.data import DATA as DATA_CFG, SPLIT_SIZES, SPLIT_TOTAL
    ds_root = Path(DATA_CFG["root"])
    ds_problems, ds_counts = [], {}
    print(f"  PLANTSEG_DATA_ROOT env : {os.environ.get('PLANTSEG_DATA_ROOT', '(unset -> default)')}")
    print(f"  resolved root          : {ds_root}")
    if not ds_root.is_dir():
        ds_problems.append(f"dataset root does not exist or is not a directory: {ds_root}")
    else:
        for sp in ("train", "val", "test"):
            img_dir, mask_dir = ds_root / "images" / sp, ds_root / "annotations" / sp
            if not img_dir.is_dir():
                ds_problems.append(f"missing image dir: {img_dir}")
                continue
            if not mask_dir.is_dir():
                ds_problems.append(f"missing annotation dir: {mask_dir}")
                continue
            imgs = [p for p in img_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
            pairs = sum(1 for p in imgs if (mask_dir / f"{p.stem}.png").exists())
            expected = SPLIT_SIZES[sp]
            ds_counts[sp] = pairs
            print(f"  {sp:<5}: images={len(imgs):<6} pairs={pairs:<6} expected={expected:<6} "
                  f"{'OK' if pairs == expected else 'MISMATCH'}")
            if len(imgs) != pairs:
                ds_problems.append(
                    f"split={sp}: {len(imgs) - pairs} image(s) have no matching <stem>.png mask")
            if pairs != expected:
                ds_problems.append(
                    f"split={sp}: pair count {pairs} != expected {expected} "
                    f"(delta {pairs - expected:+d})")
    total = sum(ds_counts.values())
    if ds_counts:
        print(f"  total: {total} (expected {SPLIT_TOTAL}) "
              f"{'OK' if total == SPLIT_TOTAL else 'MISMATCH'}")
        if total != SPLIT_TOTAL:
            ds_problems.append(f"total pair count {total} != SPLIT_TOTAL {SPLIT_TOTAL}")
    dataset_ok = not ds_problems
    print(f"  dataset_ok             : {dataset_ok}")
    for p in ds_problems:
        print(f"    - {p}")

    # (18) verdict
    print("\n[18] VERDICT")
    real_ready = cuda and gpu_count >= 1 and student_ok and dry_ok and cap_ok and dataset_ok
    if not (student_ok and dry_ok):
        verdict, label = "FAIL", "NOT OK for real E1 training (E1 scaffold not runnable on this machine)"
    elif not cap_ok:
        verdict, label = "FAIL", ("NOT OK for real E1 training (GPU compute capability exceeds the "
                                  "pinned stack's sm_90 ceiling)")
    elif not dataset_ok:
        verdict, label = "FAIL", "NOT OK for real E1 training (dataset verification failed)"
    elif real_ready:
        verdict, label = "PASS", "OK for real E1 training"
    else:
        verdict, label = "PARTIAL", "OK for DRY-RUN only; NOT OK for real E1 training"

    blockers = []
    if not dataset_ok:
        blockers.append(
            f"Dataset verification failed under root {ds_root}: " + "; ".join(ds_problems)
            + ". Fix the upload/extraction (or PLANTSEG_DATA_ROOT) before launching — the real run "
              "would otherwise abort inside PlantSegDataset.__init__.")
    if not cap_ok:
        blockers.append(
            "GPU compute capability above sm_90: "
            + "; ".join(f"cuda:{i} '{n}' = sm_{a}{b}" for i, n, (a, b) in unsupported)
            + f". torch {torch_ver} ships sm_50..sm_90 only. Re-provision on an Ampere/Ada/Hopper "
              "pod (A100 / H100 / L40S / RTX 4090) or re-pin torch+cu.")
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
