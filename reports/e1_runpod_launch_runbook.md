# B28 — E1 RunPod/GPU Launch Runbook

Operational runbook for launching the **real E1 training run** on a RunPod/GPU box. **This document is
instructions only** — no training, download, install, or GPU use was performed to produce it, and no
code/config was changed. All repo-side audit/decision blockers are resolved; the real run is gated only by
GPU-environment setup below.

_Generated 2026-07-01 at repo HEAD `8288f82`. Read `configs/e1_student.py`, `configs/data.py`,
`src/training/train_e1.py`, `scripts/verify_env.py`, `requirements.lock`, `reports/platform_verify.md`
alongside this runbook._

---

## 1. Current repo baseline
- **Expected remote HEAD:** latest `master` — must include the **B29 `PLANTSEG_DATA_ROOT`** commit (adds the portable dataset-root env var; ≥ `8288f82`).
- **Branch:** `master` (local and `origin/master` in sync).
- **Deferred local dirty file:** `docs/reference/reference.pdf` (` M`) — **do NOT stage/commit/restore it**; it is a pre-existing deferred artifact and is unrelated to the run.
- **No remaining repo-side decision blockers:** D1 (metric policy = all-class), D2/D-A (E1 unclipped documented deviation), D3 (aug wording), F1/F2 (stale wording) are all resolved & pushed.
- The GPU box should be provisioned from this exact commit; verify `git rev-parse HEAD` equals `8288f82` after checkout.

## 2. What still blocks real E1
| Item | Required state | Current/local state | Action |
|---|---|---|---|
| CUDA GPU | CUDA GPU available (`torch.cuda.is_available()=True`) | **CPU-only** (`torch 2.9.1+cpu`, 0 GPUs) — `reports/platform_verify.md` | Provision a RunPod GPU (e.g. RTX 4090, 24 GB). |
| Pinned torch/torchvision (cu121) | `torch 2.1.0+cu121` / `torchvision 0.16.0+cu121`, Python 3.11 | local `torch 2.9.1+cpu` / `tv 0.24.1+cpu`, Python 3.13 | `pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu121`. |
| Dataset root resolution | `DATA["root"]` resolves to the pod dataset dir | Root = `PLANTSEG_DATA_ROOT` if set, else the Windows default `C:\Users\admin\plantseg_data\plantseg` (`configs/data.py`, B29) | **Set `PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg`** (preferred; no file edit — see §4). Editing `DATA["root"]` is a fallback only. |
| Dataset mounted/uploaded | 7,774 pairs present under the pod dataset root | Dataset is on the **local** machine only | Upload/mount to the pod (see §4). |
| ImageNet MobileNetV3 weights | `IMAGENET1K_V2` cached or download permitted | **Not cached** (`mobilenet_v3_large-5c1a4163.pth` absent) | Allow one-time in-pod download, or pre-stage the hub file (see §5). |
| Environment verification PASS | `scripts/verify_env.py` → **PASS** (CUDA true, deps match, dataloader dry-run OK) | Local verdict **PARTIAL** (dry-run only) | Re-run `verify_env.py` **on the pod** and require PASS. |
| GPU dry-run smoke | `train_e1.py --dry-run` PASS on the pod | Passes on local CPU | Run on the pod before the real run (see §6). |

## 3. RunPod setup checklist (run ON the GPU pod)
> Linux shell on the pod. Do **not** run any history-modifying git commands (no force-push, no rewrite).

```bash
# 3.1 Clone + checkout the exact baseline
cd /workspace
git clone https://github.com/ainsleydeluna/plantseg-thesis.git
cd plantseg-thesis
git checkout master
git rev-parse HEAD          # MUST be latest master incl. the B29 PLANTSEG_DATA_ROOT commit (>= 8288f82)
git status -sb              # expect clean (no reference.pdf change on a fresh clone)

# 3.2 Python 3.11 environment (conda or venv)
conda create -y -n plantseg python=3.11 && conda activate plantseg
#   (or: python3.11 -m venv .venv && source .venv/bin/activate)

# 3.3 Install the pinned cu121 stack
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.1.0+cu121 True

# 3.4 Place/upload the dataset (see §4), then set the portable env var (B29 — no file edit needed):
export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg
#     configs/data.py reads DATA['root'] = os.environ.get("PLANTSEG_DATA_ROOT", <windows default>).
#     Fallback only (not required): editing configs/data.py DATA['root'] still works if you prefer.

# 3.5 Verify dataset structure (depth-1, no full scan)
ls /workspace/plantseg_data/plantseg                       # images/ annotations/ 3 JSONs Metadata.csv
ls -d /workspace/plantseg_data/plantseg/images/*/          # train/ val/ test/
ls -d /workspace/plantseg_data/plantseg/annotations/*/     # train/ val/ test/

# 3.6 ImageNet cache: allow first-use download (default) OR pre-stage (see §5)

# 3.7 Environment verification — REQUIRE PASS
python scripts/verify_env.py                               # expect verdict PASS (CUDA true, deps, dry-run OK)

# 3.8 Safe dry-run smoke on the pod (tiny; temp checkpoint; no real training)
python src/training/train_e1.py --dry-run                  # expect RESULT: PASS

# 3.9 Only after 3.1–3.8 pass, prepare the real-run command (see §7)
```

## 4. Dataset transfer checklist
- **Local source path (from repo docs):** `C:\Users\admin\plantseg_data\plantseg` (external, git-ignored; `reports/dataset_download_log.md`).
- **Pod dataset root (placeholder):** `/workspace/plantseg_data/plantseg`.
- **Expected split counts (do not re-audit; from `reports/dataset_audit_summary.md`):**
  - train **5,367**, val **846**, test **1,561**, **total 7,774**; 0 silent drops.
- ✅ **Preferred (B29): set the `PLANTSEG_DATA_ROOT` env var** — `configs/data.py` resolves
  `DATA["root"] = os.environ.get("PLANTSEG_DATA_ROOT", <windows default>)`, so on RunPod/Linux just run
  `export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg` **before** any dataloader use (the loader,
  `verify_env.py`'s dry-run subprocess, and the real run all read `DATA["root"]`). **No `configs/data.py`
  edit is needed.**
- **Fallback only:** editing `configs/data.py` `DATA["root"]` to the Linux path still works, but treat it as
  an environment-specific change and do **not** commit/push it (it would break the Windows dev box). The env
  var is the portable, commit-safe method.
- **Never commit dataset files into git** (the dataset is external and `.gitignore`-protected; keep it that way).

## 5. ImageNet initialization checklist
- **Expected:** MobileNetV3-Large **`IMAGENET1K_V2`** (`E1_STUDENT["init_weights"] = "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"`; top-1 75.27%).
- **Cache/download issue:** the weight file `mobilenet_v3_large-5c1a4163.pth` is **not cached** anywhere yet; `build_student(pretrained="…IMAGENET1K_V2")` loads it from the torch-hub cache **or downloads** on first use, and **fails loud** if offline+uncached (no silent random fallback).
- **Recommended path:** allow a **one-time download inside the GPU env** (network on; the real run's `--init imagenet` triggers it). Alternatively **pre-stage** the file:
  ```bash
  mkdir -p ~/.cache/torch/hub/checkpoints
  # place mobilenet_v3_large-5c1a4163.pth there (from a trusted mirror/torchvision), then verify:
  ls -l ~/.cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth
  ```
- ⚠ **Do not fall back to random init** (`--init none`) for the real E1 run — it **deviates from Chapter 3 B2** (which mandates ImageNet init) and is **not recommended**.

## 6. Safe pre-real-run checks (all must pass first)
```bash
git status -sb                                             # clean (PLANTSEG_DATA_ROOT env var used; no configs/data.py edit)
python scripts/verify_env.py                               # verdict PASS (CUDA true, deps, dry-run OK)
ls -d /workspace/plantseg_data/plantseg/{images,annotations}/{train,val,test}/   # 6 dirs present
python scripts/smoke_dataloader.py                         # [PASS] (builds train+val batches)
python src/training/train_e1.py --dry-run                  # RESULT: PASS (few CPU/GPU iters, temp ckpt)
git status --porcelain docs/reference/reference.pdf        # if present it must stay UNSTAGED; never add it
```
Confirm: no unexpected staged/dirty files; `docs/reference/reference.pdf` (if it appears) remains unstaged and untouched.

## 7. Real E1 command template
Exact CLI is defined in `src/training/train_e1.py` (`parse_args` `:258–278`; real-run gate + defaults `:281–328`). **Both** `--real-run` **and** `--confirm-real-run` are required (either alone exits 2).

```bash
python src/training/train_e1.py \
  --real-run --confirm-real-run \
  --init imagenet \
  --ckpt-dir /workspace/e1_ckpts
# optional explicit device: add  --device cuda
```
- **`--init imagenet`** → ImageNet `IMAGENET1K_V2` backbone (real-run default is already `imagenet`; stated explicitly for clarity).
- **Real-mode defaults** (no need to pass): `device` = cuda-if-available (`:316`), batch **16**, **80,000** iters, val every **4000**, **full** validation, `num_workers` **4** — all from `configs/e1_student.py`.
- **`--ckpt-dir` MUST be OUTSIDE the repo** (`train_e1.py` hard-guards and raises if the dir is the repo or inside it; auto temp dir if omitted, but for a real run use a persistent out-of-repo path like `/workspace/e1_ckpts`).
- **Gradient clipping:** E1 is **intentionally unclipped by default** (B27 D-A: `grad_clip_max_norm=None`, a documented deviation from ch3's value-less "global-norm, throughout"). **Add `--grad-clip-norm <value>` ONLY if divergence/instability is observed** later — not by default, and do not invent a value pre-emptively.
- Checkpoint selection = **all-class validation mIoU** (D1); disease-only mIoU logged as PROVISIONAL secondary.

> If any exact flag is uncertain on your build, re-quote `train_e1.py:258–328` rather than inventing options.

## 8. Stop conditions (abort / do not launch the real run if any is true)
- **`scripts/verify_env.py` fails** or reports **CUDA not available** (`cuda_available=False`) / wrong torch build.
- **Dataset split counts mismatch** the expected **5,367 / 846 / 1,561 / 7,774** (or `DATA["root"]` does not resolve to the pod dataset — set `PLANTSEG_DATA_ROOT`).
- **ImageNet init fails unexpectedly** (e.g. `RuntimeError` from `build_student` when offline+uncached) — resolve caching/network first; do **not** switch to random init.
- **Dry-run loss is NaN/inf**, or the `--dry-run` smoke does not print `RESULT: PASS`.
- **Checkpoint/output path is inside a forbidden location** — the repo working tree, `docs/reference/`, or the dataset dir. `--ckpt-dir` must be outside the repo (e.g. `/workspace/e1_ckpts`).
- **Any unexpected uncommitted/dirty file** appears — with `PLANTSEG_DATA_ROOT` set there should be **no** `configs/data.py` edit; only the always-deferred `reference.pdf` may show on a machine that has it — investigate before launching.

## 9. Final launch approval checklist (review before "start real E1")
- [ ] Pod at latest `master` (includes the **B29 `PLANTSEG_DATA_ROOT`** commit), branch `master`; working tree **clean** (no `configs/data.py` edit needed).
- [ ] `pip` shows **`torch 2.1.0+cu121`**, `torchvision 0.16.0+cu121`; `torch.cuda.is_available()` = **True**.
- [ ] Dataset present at the pod root; **`PLANTSEG_DATA_ROOT` exported** so `DATA["root"]` resolves to it; split dirs verified.
- [ ] Split counts confirmed **5,367 / 846 / 1,561 / 7,774**.
- [ ] ImageNet `IMAGENET1K_V2` cached or download permitted (network on).
- [ ] `scripts/verify_env.py` → **PASS**.
- [ ] `train_e1.py --dry-run` → **RESULT: PASS** on the pod.
- [ ] `--ckpt-dir` set to an **out-of-repo** path (e.g. `/workspace/e1_ckpts`).
- [ ] Understood: E1 **unclipped by default** (B27); `--grad-clip-norm` only if divergence.
- [ ] `docs/reference/reference.pdf` **not staged**; no stray `configs/data.py` edit (env var used, so none needed).
- [ ] No forbidden-path checkpoint; no unexpected dirty files.
- → If **all** checked: launch the §7 real-run command.

---

## Provenance / guardrails honored
Documentation only. No training, no download, no install, no GPU use; no code/config/dataset changes on
this machine (the `DATA["root"]` Linux edit is **documented as a pod-side action**, not performed here).
`docs/reference/reference.pdf`, manuscript PDFs, and dataset files untouched; nothing staged, committed, or
pushed.
