# B20a — Strict-G0 Platform Verification

**VERDICT: ⚠ PARTIAL — OK for DRY-RUN only; NOT OK for real E1 training on this machine.**
`can_run_real_E1 = False`. This is a diagnostic capture — **no training, no download, no GPU compute,
no repo writes** (besides this report) and **no repo checkpoints**.

_Generated: 2026-06-30 from `scripts/verify_env.py` stdout. Interpreter `C:\Users\admin\anaconda3\python.exe`._

---

## 1. Verdict — **PARTIAL**
The E1 scaffold is runnable here (random-init build + tiny `train_e1.py --dry-run` both PASS), but the
machine **cannot host the real E1 run**: no CUDA GPU, a CPU-only torch that diverges from the pinned
GPU stack, and the ImageNet backbone is not cached. Environment label:
**OK for DRY-RUN only; NOT OK for real E1 training.**

## 2. Command used
```
PYTHONIOENCODING=utf-8 "C:/Users/admin/anaconda3/python.exe" scripts/verify_env.py
```
(The script internally spawns one tiny `src/training/train_e1.py --dry-run` subprocess with a temp,
out-of-repo checkpoint dir; exit 0.)

## 3. git HEAD + status summary
- HEAD: **`93e57bb1c040a334ead5bf76287303f55bc7e180`** (`93e57bb`) on `master`, dirty=True.
- modified (1): `docs/reference/reference.pdf` (pre-existing deferred — untouched).
- untracked (1): `scripts/verify_env.py` (this step's new script; report added after capture).

## 4. Python / OS
| Field | Value |
|---|---|
| python_executable | `C:\Users\admin\anaconda3\python.exe` |
| python_version | 3.13.5 |
| OS | Windows 11 (10.0.26200), `Windows-11-10.0.26200-SP0` |
| machine | AMD64 |
| repo_root | `C:\Users\admin\plantseg-thesis` |

## 5. PyTorch / torchvision / CUDA / GPU / cuDNN / threads
| Field | Value |
|---|---|
| torch_version | **2.9.1+cpu** |
| torchvision_version | **0.24.1+cpu** |
| cuda_available | **False** |
| cuda_version | None |
| gpu_count | 0 |
| gpu_names | [] |
| cudnn_available | False |
| cudnn_version | None |
| cpu_count (logical) | 16 |
| torch_num_threads | 12 |

## 6. Package versions — actual vs pinned (`requirements.lock`)
| Package | Actual (local) | Pinned (training stack) |
|---|---|---|
| torch | **2.9.1+cpu** | **2.1.0+cu121** |
| torchvision | **0.24.1+cpu** | **0.16.0+cu121** |
| numpy | 2.1.3 | 1.26.4 |
| scipy | 1.15.3 | 1.11.4 |
| Pillow (PIL) | 11.1.0 | 12.2.0 |
| opencv (cv2) | **not installed** | 4.8.1.78 |
| albumentations | **not installed** | — (not in lock) |
| mmcv | **not installed** | 2.1.0 |
| mmseg | **not installed** | 1.2.2 |
| statsmodels | 0.14.4 | 0.14.6 |

**Pinned-vs-actual mismatch (key point):** the local environment is a **CPU-only** build
(`torch 2.9.1+cpu` / `torchvision 0.24.1+cpu`) and does **not** match the pinned **GPU** training stack
(`torch 2.1.0+cu121` / `torchvision 0.16.0+cu121`). The teacher/MMSeg stack (`mmcv`, `mmsegmentation`,
`opencv-python`) and `albumentations` are **not installed locally** — these matter for the teacher
fine-tune and E2/E3 distillation, **not** for E1 supervised training (E1 needs only torch/torchvision +
numpy + Pillow, all present). E1's train-time augmentation is hand-implemented in `src/data/transforms.py`
(no albumentations dependency), so its absence does not block E1.

## 7. ImageNet cache status
- hub_dir: `C:\Users\admin/.cache\torch\hub`
- checkpoint: `…\checkpoints\mobilenet_v3_large-5c1a4163.pth`
- **imagenet_cached: False.**

## 8. Random-init student smoke
**`build_student(pretrained=False)` works (CPU): PASS** — output `(1, 116, 64, 64)`, finite,
`used_pretrained=False` (no download path taken).

## 9. ImageNet pretrained construction
**Skipped to avoid download** — `build_student("torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2")`
was **not** attempted because the checkpoint is not cached (`imagenet_skipped=True`). Had it been cached,
the script would have loaded it from cache offline; it never reaches the network.

## 10. Dry-run subprocess result
- command: `… src/training/train_e1.py --dry-run --batch-size 1 --max-iters 2 --val-interval 2 --max-val-batches 1 --ckpt-dir <temp>`
- temp checkpoint dir (**outside repo, confirmed**): `C:\Users\admin\AppData\Local\Temp\verify_env_dryrun_ckpt__ppsz_q1`
- return_code: **0**, **RESULT: PASS**
- checkpoint written: `…\verify_env_dryrun_ckpt__ppsz_q1\e1_student_best_iter2.pt` (temp only)
- **repo `.pt` scan: empty → no checkpoint written inside the repo** (`no_repo_checkpoint=True`).

## 11. Is real E1 training allowed on this environment?
**No.** `can_run_real_E1 = False`. The 80k-iter run at batch size 16 / 512² is not feasible on a CPU-only
box; it requires a CUDA GPU and the pinned `cu121` stack.

## 12. Exact blockers before the real E1 run
1. **No CUDA GPU** — `cuda_available=False`; local torch is the CPU-only build `2.9.1+cpu`. Real E1 needs a GPU.
2. **torch/torchvision mismatch vs pinned** — local `2.9.1+cpu` / `0.24.1+cpu` ≠ pinned `2.1.0+cu121` / `0.16.0+cu121`. Provision the pinned `cu121` stack on the GPU target.
3. **ImageNet backbone not cached** — `--init imagenet` needs network to populate the torch-hub cache, or a pre-staged `mobilenet_v3_large-5c1a4163.pth`.
4. **Teacher/aug stack absent locally** — `opencv (cv2)`, `albumentations`, `mmcv`, `mmseg` not installed (needed for the teacher fine-tune / E2–E3, **not** E1 supervised; listed for completeness).

> Non-blocking for E1 specifically: items 3 is required only for ImageNet init; item 4 is for later
> phases. Item 1 (and the matching item 2) is the hard blocker for any real E1 run.

## 13. Exact recommended next step
Provision the **GPU compute target** (e.g. RunPod/cloud GPU) with the pinned stack
(`pip install -r requirements.lock` → `torch 2.1.0+cu121`), **re-run `scripts/verify_env.py` there** to
get a PASS, populate the ImageNet cache (network once), then launch the real run:
```
python src/training/train_e1.py --real-run --confirm-real-run --init imagenet --ckpt-dir <out-of-repo path>
```
Until then, the next safe **local** repo task is committing B20a, then the remaining strict-G0 hygiene
items (persisted dataloader smoke report; GitHub remote/push), which are environment-independent.

---

## Provenance / guardrails honored
Diagnostic only: no training, no download (ImageNet build skipped — not cached), no GPU compute
(availability queries only), no installs. The script wrote **no repo files and no repo checkpoints**; the
only disk write was a temp checkpoint dir **outside** the repo (`…\AppData\Local\Temp\verify_env_dryrun_ckpt__ppsz_q1`),
and a repo-wide `.pt` scan confirmed none landed in the repo. `docs/reference/reference.pdf`,
`docs/reference/context.md`, and teacher-init files untouched; no training/model/data/loss/config files
modified; nothing staged or committed; Claude memory and `~/.claude` untouched.
