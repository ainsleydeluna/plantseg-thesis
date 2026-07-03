# CLAUDE.md — plantseg-thesis (THESIS2)

> Read this first. It is the short operating contract for Claude Code in this repo.
> Fuller detail: [docs/ai_guardrails.md](docs/ai_guardrails.md). Reusable task prompts: [docs/task_templates/](docs/task_templates/).

## Project identity
- **Project:** THESIS2 — `plantseg-thesis`. Resource-constrained plant-lesion segmentation (knowledge distillation + INT8 quantization); the current focus is the **E1 FP32 student baseline**.
- **Repo path:** `C:\Users\admin\plantseg-thesis`
- **Branch:** `master` (keep local in sync with `origin/master`).
- **E1 baseline commit expectation:** **`885523a` or newer**.

## Always, every task
1. **Inspect git status first** (`git status -sb`, `git log --oneline -5`). Expect a clean tree with **only `docs/reference/reference.pdf` dirty**, nothing staged, local `master` even with `origin/master`.
2. **PROTECT `docs/reference/reference.pdf`** — never open, read, stage, restore, or modify it. It stays dirty/unstaged.
3. **Explicit-path staging only.** NEVER `git add -A`, `git add .`, or wildcards. Stage the exact files you changed.
4. **Plan-gated edits:** inspect → propose a minimal plan → **wait for the user's "go"** before editing, unless the task prompt explicitly says the edits are approved.
5. **No training, downloads, installs, GPU use, or pushes** unless the user explicitly approves them in the task.
6. **No commits/pushes until after `git diff` + `git status` verification**; then commit with an explicit message and push only if approved.

## E1 invariants (do not violate)
- E1 = **FP32 MobileNetV3-Large + LR-ASPP** baseline.
- **No teacher, no KD, no CWD, no QAT, no PTQ** in E1.
- `num_classes = 116` · **background = 0** · disease labels **1..115**.
- `reduce_zero_label = False` (no label remap).
- `ignore_index = 255` — padding/ignore only, never a model output class; excluded from all loss/metrics.
- Dataset root is portable via **`PLANTSEG_DATA_ROOT`** (RunPod: `/workspace/plantseg_data/plantseg`).
- **Checkpoints live OUTSIDE the repo** (`--ckpt-dir /workspace/e1_ckpts`); `train_e1.py` hard-guards this.
- **Checkpoint/headline validation metric = all-class mIoU.**
- **Disease-only mIoU is provisional/reporting only** — never the checkpoint criterion.
- E1 install file = **`requirements-e1.txt`** (student-only; NO mmcv/mmseg/teacher stack).
- Real E1 requires **`--real-run` AND `--confirm-real-run`**, requires **CUDA** (hard-aborts on CPU). Dry-run stays CPU/random/no-download.

## Mode guidance
- **Max + Accept Edits** — documentation, read-only audits, runbooks, small consistency/cleanup fixes.
- **UltraCode + Accept Edits** — behavior-changing code, runtime debugging, RunPod training/debug, and any data/model/loss/metric/training-loop/quantization change. Use plan-gated execution.

## Where things live
- E1 launch: [reports/e1_runpod_launch_runbook.md](reports/e1_runpod_launch_runbook.md)
- RunPod pre-flight checklist: [docs/task_templates/runpod_preflight_template.md](docs/task_templates/runpod_preflight_template.md)
- Teacher prep (separate A6000 workflow, NOT for E1): [docs/teacher_prep_runbook.md](docs/teacher_prep_runbook.md)
- Locked configs/methodology: [docs/IMPLEMENTATION_CONTRACT.md](docs/IMPLEMENTATION_CONTRACT.md) · open items: [docs/open_questions.md](docs/open_questions.md)
- E1 install: `requirements-e1.txt` (student) · full pinned stack: `requirements.lock`
