# Task Templates

Short, reusable prompts/checklists so future Claude Code tasks don't need giant copied meta-prompts.
Read [../../CLAUDE.md](../../CLAUDE.md) first, then [../ai_guardrails.md](../ai_guardrails.md).

| Template | Use for |
|---|---|
| [B31_review_template.md](B31_review_template.md) | Read-only audit / review passes (report sections A–H). |
| [fix_template.md](fix_template.md) | Plan-gated, file-changing fixes (inspect → propose → verify → commit). |
| [runpod_preflight_template.md](runpod_preflight_template.md) | The exact pod pre-flight command order before the real E1 run. |

**Usage:** reference a template by path in your task prompt (e.g. "follow `docs/task_templates/fix_template.md`")
instead of pasting a full meta-prompt.
