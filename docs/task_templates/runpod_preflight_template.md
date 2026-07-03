# RunPod Pre-flight Checklist — E1 real run

Run **on the GPU pod**, in order. Do not launch the real 80k run until steps 1–12 pass. Mirrors the B31g
safe order. Full detail: [reports/e1_runpod_launch_runbook.md](../../reports/e1_runpod_launch_runbook.md).

Legend: **[RO]** read-only / temp-only (no repo writes) · **[W]** writes a `reports/*.md` (fine on the
throwaway pod — do NOT commit it).

1. **[RO]** Clone + checkout: `git clone <repo> && cd plantseg-thesis && git checkout master`
2. **[RO]** Verify baseline: `git rev-parse HEAD` → **≥ `885523a`** (must include the B29 `PLANTSEG_DATA_ROOT` commit, i.e. ≥ `1576d7c`); `git status -sb` clean.
3. **[RO]** Env: `conda create -y -n plantseg python=3.11 && conda activate plantseg`
4. **[RO]** Install E1 stack only: `pip install -r requirements-e1.txt`  (NOT `requirements.lock`; no mmcv/mmseg/teacher)
5. **[RO]** Torch/CUDA: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → `2.1.0+cu121 True`
6. **[RO]** Dataset: upload to `/workspace/plantseg_data/plantseg`; `export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg`; `ls -d /workspace/plantseg_data/plantseg/{images,annotations}/{train,val,test}/` → 6 dirs.
7. **[RO]** `python scripts/verify_env.py` → **PASS** (CUDA true, deps, dry-run OK).
8. **[RO]** `python scripts/smoke_dataloader.py` → `[PASS]` (builds train+val; enforces locked counts 5367/846/1561 + mask pairing).
9. **[RO]** `python scripts/smoke_student_forward.py` → `[PASS]` (output `[2,116,512,512]`; confirms OS8/OS16 taps on the **pinned torchvision 0.16.0**).
10. **[W]** `python scripts/smoke_loss.py` and `python scripts/smoke_metrics.py` → `PASS` (loss + metric wiring; these **write** `reports/loss_smoke.md` / `reports/metrics_smoke.md` — do not commit).
11. **[RO]** `python src/training/train_e1.py --dry-run` → `RESULT: PASS` (GPU dry-run, temp checkpoint outside repo).
12. **[RO]** `git status --porcelain docs/reference/reference.pdf` → stays **unstaged**; confirm no stray staged/dirty files.
13. **Only if 1–12 pass — real run:**
    ```bash
    python src/training/train_e1.py \
      --real-run --confirm-real-run \
      --init imagenet \
      --ckpt-dir /workspace/e1_ckpts
    ```
    Real-run notes: requires CUDA (hard-aborts on CPU); `--init imagenet` uses IMAGENET1K_V2; `--ckpt-dir` MUST be out-of-repo; checkpoint selection = all-class val mIoU; disease-only mIoU is logged as PROVISIONAL only.

## Optional full dataset re-audit (writes reports)
- **[W]** `python scripts/verify_plantseg_dataset.py` (now `PLANTSEG_DATA_ROOT`-aware) — full-dataset scan; writes `reports/dataset_report.{md,json}`. Not required on the pod (the committed report + `smoke_dataloader` cover E1); use only for a fresh pod-side audit, and do not commit its output.
