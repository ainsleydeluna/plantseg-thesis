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
- **[UPDATED 2026-09-23 — B64 C6]** Model and effort (DL-24): Opus 5.5 everywhere. Claude Code: high by default; medium for mechanical doc and sync edits; xhigh for pod/GPU sessions and official-run launches; max and ultracode are not used; workflows and subagent fan-outs only when a prompt asks for one. Web chat: High by default; Max for deep audits. Edits stay plan-gated (AGENTS.md rule 4).
