# AI Guardrails & Workflow — plantseg-thesis

Fuller companion to [CLAUDE.md](../CLAUDE.md). These are the standing rules for Claude Code work in this
repo. When a task prompt and this file disagree, the task prompt wins **for that task** — but never for the
hard safety rules (protect `reference.pdf`, no wildcard staging, no unapproved training/push).

## 1. Git safety
- Inspect first: `git status -sb` and `git log --oneline -5` before any work.
- Expected clean state: **only `docs/reference/reference.pdf` dirty (` M`), nothing staged**, local `master` even with `origin/master`.
- **Explicit-path staging only.** Never `git add -A`, `git add .`, or globs — stage exactly the files the task approved.
- No commits or pushes until you have shown `git diff` + `git status` and confirmed only the approved paths changed.
- Never run history-modifying git (force-push, rebase, `reset --hard`) unless explicitly asked.
- End commit messages with the repo's co-author trailer.

## 2. Artifact safety
- **Checkpoints, logs, and datasets never go in the repo.** `.gitignore` covers `*.pt/*.pth/*.ckpt/*.onnx`, `*.log`, `outputs/**/checkpoints/`, `weights/`, `plantseg_data/`, `datasets/`, and archives.
- `train_e1.py` hard-guards `--ckpt-dir` OUTSIDE the repo (raises if inside). Keep it that way.
- The training path writes NOTHING into the repo. The only in-repo writers are **dev smoke scripts** that emit `reports/*.md` (see §3) — tools, not the training path.
- Never commit `reports/*.md`/`*.json` produced by re-running a smoke on a throwaway pod.

## 3. RunPod safety
- Pod paths are separate and outside the repo: repo `/workspace/plantseg-thesis`, dataset `/workspace/plantseg_data/plantseg`, checkpoints `/workspace/e1_ckpts`.
- `export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg` before any dataloader use.
- E1 install = `pip install -r requirements-e1.txt` (student stack; NO mmcv/mmseg/teacher — that is the separate A6000 teacher workflow).
- Run the full pre-flight ([task_templates/runpod_preflight_template.md](task_templates/runpod_preflight_template.md)) before the real run. **Report-writing** smokes (`smoke_loss.py`, `smoke_metrics.py`, `verify_plantseg_dataset.py`) are fine on the pod but must not be committed.
- Real run is triple-gated: `--real-run` + `--confirm-real-run` + CUDA present (hard-abort on CPU). Do not push from the pod.

## 4. Dataset safety
- Do not modify, move, or re-download the dataset. It is external and git-ignored.
- Locked split counts: **train 5,367 / val 846 / test 1,561 (total 7,774)** — the loader enforces these and fails loud on a mismatch or a missing mask.
- Masks are raw indices 0–115 (bg 0 + diseases 1–115); never remap, shift, clamp, or RGB-convert them. 255 is pad/ignore only.
- Val/test are never augmented; corruptions are never applied during E1 clean training.

## 5. Thesis methodology invariants (E1)
Canonical list in [CLAUDE.md](../CLAUDE.md): FP32 MobileNetV3-Large + LR-ASPP; no teacher/KD/CWD/QAT/PTQ; `num_classes=116`; background 0; diseases 1–115; `reduce_zero_label=False`; `ignore_index=255` pad-only; `PLANTSEG_DATA_ROOT` portable root; out-of-repo checkpoints; all-class mIoU checkpoint criterion; disease-only mIoU provisional/reporting.

## 6. When to use Max vs UltraCode
- **Max + Accept Edits:** documentation, read-only audits, runbooks, small consistency/cleanup fixes (stale placeholders, wording), template authoring.
- **UltraCode + Accept Edits:** behavior-changing code, runtime debugging, RunPod training/debug, and any change to dataset/model/loss/metric/training-loop/quantization logic. Use plan-gated execution (inspect → propose → wait for go / self-verify if pre-approved).

## 7. How to report final results
Every task ends with a concise report.
- **Read-only audits:** use the [B31 review template](task_templates/B31_review_template.md) (sections A–H).
- **Fixes/changes:** include **files changed** (exact paths) · **verification commands + results** (CPU-only unless approved) · **commit hash** + pushed branch (if any) · **final `git status`** · confirmation that `docs/reference/reference.pdf` remains dirty/unstaged/untouched and no unrelated files changed.

## 8. Expected final git status pattern
Unless the task explicitly changed files, the tree should end as:
```
## master...origin/master
 M docs/reference/reference.pdf
```
If a task changed files, the ONLY additional entries are the approved paths — nothing else, and `reference.pdf` is never staged.
