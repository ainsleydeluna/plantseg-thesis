# CLAUDE.md — plantseg-thesis (THESIS2)

@AGENTS.md

> The import above loads [AGENTS.md](AGENTS.md), the **canonical shared contract** for every coding
> agent in this repository. It is authoritative for project identity, Git safety, the protected PDF,
> explicit-path staging, plan-gated edits, the scoped governed-path rule, and the E1 invariants —
> those rules are **not** repeated here, so there is exactly one copy to keep correct.
> Only Claude-Code-specific guidance lives below.
> Fuller detail: [docs/ai_guardrails.md](docs/ai_guardrails.md). Reusable task prompts: [docs/task_templates/](docs/task_templates/).

## Mode guidance (Claude Code only)
- **Max + Accept Edits** — documentation, read-only audits, runbooks, small consistency/cleanup fixes.
- **UltraCode + Accept Edits** — behavior-changing code, runtime debugging, RunPod training/debug, and any data/model/loss/metric/training-loop/quantization change. Use plan-gated execution.
