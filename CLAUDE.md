# CLAUDE.md — plantseg-thesis (THESIS2)

@AGENTS.md

> The import above loads [AGENTS.md](AGENTS.md), the **canonical shared contract** for every coding
> agent in this repository. It is authoritative for project identity, Git safety, the protected PDF,
> explicit-path staging, plan-gated edits, the scoped governed-path rule, and the E1 invariants —
> those rules are **not** repeated here, so there is exactly one copy to keep correct.
> Only Claude-Code-specific guidance lives below.
> Fuller detail: [docs/ai_guardrails.md](docs/ai_guardrails.md). Reusable task prompts: [docs/task_templates/](docs/task_templates/).

## Decision log
Before any task, read `docs/DECISION_LOG.md`. DECIDED and RECORDED entries bind planning; if one conflicts with a governed file or a run recipe, STOP and report. Update the Status of any entry your task implements, in the same commit.

## Mode guidance (Claude Code only)
- ~~**Max + Accept Edits** — documentation, read-only audits, runbooks, small consistency/cleanup fixes.~~
- ~~**UltraCode + Accept Edits** — behavior-changing code, runtime debugging, RunPod training/debug, and any data/model/loss/metric/training-loop/quantization change. Use plan-gated execution.~~
- ~~**[UPDATED 2026-09-23 — B64 C6]** Model and effort (DL-24): Opus 5.5 everywhere. Claude Code: high by default; medium for mechanical doc and sync edits; xhigh for pod/GPU sessions and official-run launches; max and ultracode are not used; workflows and subagent fan-outs only when a prompt asks for one. Web chat: High by default; Max for deep audits. Edits stay plan-gated (AGENTS.md rule 4).~~
- **[UPDATED 2026-09-24 — B65 CP-006]** Model, route and effort (DL-24). Default route: Claude web (Opus 5.5) → Claude Code (Opus 5.5); experiment-critical or ambiguous handoffs add a Fable 5.1 audit and an Opus 5.5 integration pass before Claude Code. The user sets the effort level before each Claude Code prompt; prompts don't state it. Claude Code runs Opus 5.5 (default effort medium unless set): ultracode (xhigh plus automatic workflow orchestration) for read-only sessions such as Phase A audits and repo-wide sweeps; xhigh for sessions that write, commit, push or run pods, with a read-only verification workflow requested before each commit; high for routine, localized or mechanical sessions; max is not used. Ultracode is never on in a session that writes, commits, pushes or runs pods (no mid-run input, no plan approval in Auto mode, concurrent agents); a read-only session switches to xhigh before its first approved write. The ultracode keyword trigger is off, so the keyword in a prompt never starts a workflow; a plain-language request ("run a read-only verification workflow") is the opt-in for a single workflow. Claude Code does not enforce read-only on workflow agents; the session's permission prompts do, and they stay on in write sessions. The "Switch models when a message is flagged" setting (switchModelsOnFlag) is off, so a flagged request pauses for a choice instead of switching models. Web chat: Opus 5.5 at High or XHigh. Edits stay plan-gated (AGENTS.md rule 4).

## Plugin skills and worktrees (DL-28)
**[UPDATED 2026-09-24 — B65 CP-006]** Plugin skills (including Superpowers) never override this repo's rules. Not used: superpowers:using-git-worktrees; finishing-a-development-branch (no merges, pull requests or branch deletion); subagent-driven-development and executing-plans (both require a worktree and dispatch writing subagents); dispatching-parallel-agents for any write. Claude Code's own worktrees (`--worktree`/`-w`, subagent `isolation: worktree`, a Desktop-app session worktree) are not used either: every session runs in the single checkout on `claude/keen-curie-u4a8ig`. Plans, specs, design documents and notes (brainstorming, writing-plans) go to the session scratchpad or `C:\Users\admin\Documents\thesis_prompts`, never into the repo. No parallel-agent writes; no autonomous plan execution: a typed GO gates each approved action, never a plan. test-driven-development, systematic-debugging, verification-before-completion, requesting-code-review and receiving-code-review are used where they fit.
