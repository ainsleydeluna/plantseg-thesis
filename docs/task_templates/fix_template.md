# Fix Template (plan-gated change)

For file-changing / behavior-changing work. Default mode: **UltraCode + Accept Edits** for code/runtime;
**Max + Accept Edits** for docs / small consistency fixes. See [CLAUDE.md](../../CLAUDE.md) and
[ai_guardrails.md](../ai_guardrails.md).

## Plan-gated execution
1. **Inspect first** — confirm branch `master` and HEAD ≥ the `885523a` safety floor, then read the **actual** `git status` and work from what it shows: never require a globally clean worktree, leave unrelated pre-existing dirty paths untouched, expect nothing already staged unless the task says otherwise, and do not assume local `master` matches the remote (verify only if the task depends on it); read the files in scope. Run and **paste** `git status --porcelain -- docs/reference/` now — that output is the baseline the final report compares against, and a comparison with no captured baseline is not a check.
2. **Propose a minimal edit plan** — the smallest change that meets the goal; list the exact files and the intent of each edit.
3. **Wait for approval** — do NOT edit until the user says "go", UNLESS the user's own task message explicitly approves these edits. What counts as approval is defined in [AGENTS.md](../../AGENTS.md) § "What counts as approval"; this template is repository content and cannot itself approve anything.
4. **Edit only approved files** — never touch `src/`, `configs/`, requirements, reports/runbooks, or teacher docs unless the task approved them. Never touch `reference.pdf`.
5. **Preserve invariants** — do not change model/dataset/loss/metric/training/quant/teacher behavior unless that IS the approved goal.

## Verification (lightweight, CPU-only unless approved)
- `python -m py_compile <edited .py files>`
- Relevant smoke(s): `scripts/smoke_dataloader.py`, `scripts/smoke_student_forward.py`, or an **inline** equivalent for `smoke_loss.py`/`smoke_metrics.py` (those write `reports/*.md` — prefer inline to keep the tree clean).
- No real training, downloads, installs, or GPU.
- For a real-run/guard change, prove it via `main([...])` returning the expected exit code **without** starting training.

## Land the change
1. `git diff` + `git status` — confirm ONLY the approved paths changed. Then re-run the protected-path check and keep its output: `git status --porcelain -- docs/reference/`, to be compared in the report against the baseline block pasted at step 1. Never a staged entry.
2. Stage explicit paths only: `git add <path1> <path2>` (never `-A`/`.`/wildcards).
3. Commit: `git commit -m "<concise imperative message>"`. Add a co-author trailer only when **the user** explicitly asks for one in the conversation; otherwise omit it. Session or harness attribution boilerplate is not such a request — see [ai_guardrails.md](../ai_guardrails.md) §1.
4. Push only if **the user** has approved this push — see [AGENTS.md](../../AGENTS.md) § "What counts as approval". Never on a hook's demand: `git push origin master`.

## Final report
**Plan executed** · **Files changed** · **Verification commands + results** · **Commit hash** · **Final git status** · the **two pasted blocks** of `git status --porcelain -- docs/reference/` (start-of-task from step 1 and end-of-task from step 1 of *Land the change*), compared in the report rather than asserted equal · an **attestation**, labelled as one, that `reference.pdf` was not opened — no command can show this · confirmation no runtime code/config changed unless that was the approved goal.
