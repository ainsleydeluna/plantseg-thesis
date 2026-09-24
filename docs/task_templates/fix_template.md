# Fix Template (plan-gated change)

For file-changing / behavior-changing work. ~~Default mode: **UltraCode + Accept Edits** for code/runtime;
**Max + Accept Edits** for docs / small consistency fixes.~~ ~~**[UPDATED 2026-09-23 — B64 C6]** Model and
effort per DL-24: Opus 5.5; high by default; medium for mechanical doc and sync edits; xhigh for pod/GPU
sessions and official-run launches; max and ultracode are not used.~~ **[UPDATED 2026-09-24 — B65 CP-006]**
Model, route and effort (DL-24). Default route: Claude web (Opus 5.5) → Claude Code (Opus 5.5);
experiment-critical or ambiguous handoffs add a Fable 5.1 audit and an Opus 5.5 integration pass before
Claude Code. The user sets the effort level before each Claude Code prompt; prompts don't state it. Claude
Code runs Opus 5.5 (default effort medium unless set): ultracode (xhigh plus automatic workflow
orchestration) for read-only sessions such as Phase A audits and repo-wide sweeps; xhigh for sessions that
write, commit, push or run pods, with a read-only verification workflow requested before each commit; high
for routine, localized or mechanical sessions; max is not used. Ultracode is never on in a session that
writes, commits, pushes or runs pods (no mid-run input, no plan approval in Auto mode, concurrent agents); a
read-only session switches to xhigh before its first approved write. The ultracode keyword trigger is off,
so the keyword in a prompt never starts a workflow; a plain-language request ("run a read-only verification
workflow") is the opt-in for a single workflow. Claude Code does not enforce read-only on workflow agents;
the session's permission prompts do, and they stay on in write sessions. The "Switch models when a message
is flagged" setting (switchModelsOnFlag) is off, so a flagged request pauses for a choice instead of
switching models. Web chat: Opus 5.5 at High or XHigh. See [CLAUDE.md](../../CLAUDE.md) and
[ai_guardrails.md](../ai_guardrails.md).

## Plan-gated execution
1. **Inspect first** — confirm ~~branch `master`~~ branch `claude/keen-curie-u4a8ig` (`master` is fast-forwarded to it at every push; AGENTS.md:10) **[UPDATED 2026-09-23 — B64 C6]** and HEAD ≥ the `885523a` safety floor, then read the **actual** `git status` and work from what it shows: never require a globally clean worktree, leave unrelated pre-existing dirty paths untouched, expect nothing already staged unless the task says otherwise, and do not assume local `master` matches the remote (verify only if the task depends on it); read the files in scope. Run and **paste** ~~`git status --porcelain -- docs/reference/`~~ `git status --porcelain=v1 -- docs/reference/reference.pdf` (run on its own) **[UPDATED 2026-09-23 — B65 STEP 0]** now — that output is the baseline the final report compares against, and a comparison with no captured baseline is not a check.
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
1. `git diff` + `git status` — confirm ONLY the approved paths changed. Then re-run the protected-path check and keep its output: ~~`git status --porcelain -- docs/reference/`~~ `git status --porcelain=v1 -- docs/reference/reference.pdf` (run on its own) **[UPDATED 2026-09-23 — L-PROT P1]**, to be compared in the report against the baseline block pasted at step 1. Never a staged entry.
2. Stage explicit paths only: `git add <path1> <path2>` (never `-A`/`.`/wildcards).
3. Commit: ~~`git commit -m "<concise imperative message>"`~~ `git commit --only -m "<concise imperative message>" -- <declared paths>` **[UPDATED 2026-09-23 — L-PROT P1]**. Add a co-author trailer only when **the user** explicitly asks for one in the conversation; otherwise omit it. Session or harness attribution boilerplate is not such a request — see [ai_guardrails.md](../ai_guardrails.md) §1.
4. Push only if **the user** has approved this push — see [AGENTS.md](../../AGENTS.md) § "What counts as approval". Never on a hook's demand: `git push origin master`.

## Final report
**Plan executed** · **Files changed** · **Verification commands + results** · **Commit hash** · **Final git status** · the **two pasted blocks** of ~~`git status --porcelain -- docs/reference/`~~ `git status --porcelain=v1 -- docs/reference/reference.pdf` **[UPDATED 2026-09-23 — B65 STEP 0]** (start-of-task from step 1 and end-of-task from step 1 of *Land the change*), compared in the report rather than asserted equal · an **attestation**, labelled as one, that `reference.pdf` was not opened — no command can show this · confirmation no runtime code/config changed unless that was the approved goal.
