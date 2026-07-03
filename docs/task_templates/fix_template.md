# Fix Template (plan-gated change)

For file-changing / behavior-changing work. Default mode: **UltraCode + Accept Edits** for code/runtime;
**Max + Accept Edits** for docs / small consistency fixes. See [CLAUDE.md](../../CLAUDE.md) and
[ai_guardrails.md](../ai_guardrails.md).

## Plan-gated execution
1. **Inspect first** — confirm git state (branch `master`, HEAD ≥ `885523a`, synced, only `reference.pdf` dirty, nothing staged); read the files in scope.
2. **Propose a minimal edit plan** — the smallest change that meets the goal; list the exact files and the intent of each edit.
3. **Wait for approval** — do NOT edit until the user says "go", UNLESS the task prompt explicitly states the edits are approved (then self-verify the plan matches the approved goals and proceed).
4. **Edit only approved files** — never touch `src/`, `configs/`, requirements, reports/runbooks, or teacher docs unless the task approved them. Never touch `reference.pdf`.
5. **Preserve invariants** — do not change model/dataset/loss/metric/training/quant/teacher behavior unless that IS the approved goal.

## Verification (lightweight, CPU-only unless approved)
- `python -m py_compile <edited .py files>`
- Relevant smoke(s): `scripts/smoke_dataloader.py`, `scripts/smoke_student_forward.py`, or an **inline** equivalent for `smoke_loss.py`/`smoke_metrics.py` (those write `reports/*.md` — prefer inline to keep the tree clean).
- No real training, downloads, installs, or GPU.
- For a real-run/guard change, prove it via `main([...])` returning the expected exit code **without** starting training.

## Land the change
1. `git diff` + `git status` — confirm ONLY the approved paths changed and `reference.pdf` is still unstaged/untouched.
2. Stage explicit paths only: `git add <path1> <path2>` (never `-A`/`.`/wildcards).
3. Commit: `git commit -m "<concise imperative message>"` (with the repo's co-author trailer).
4. Push only if the task approved it: `git push origin master`.

## Final report
**Plan executed** · **Files changed** · **Verification commands + results** · **Commit hash** · **Final git status** · confirmation `reference.pdf` remains dirty/unstaged/untouched and no runtime code/config changed unless that was the approved goal.
