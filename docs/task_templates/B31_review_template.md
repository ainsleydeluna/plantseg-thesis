# Review Template (B31-style read-only audit)

Reusable prompt/checklist for any read-only audit. **Review-only unless the user explicitly approves edits
later.** See [CLAUDE.md](../../CLAUDE.md) and [ai_guardrails.md](../ai_guardrails.md).

## Standing rules for a review pass
- Read-only: **inspect → report only**. No edits, commits, pushes, training, downloads, installs, GPU, or dataset changes.
- Protect `docs/reference/reference.pdf` (never open/stage/modify).
- Start by confirming git state: branch `master`, HEAD ≥ `885523a`, synced with `origin/master`, only `reference.pdf` dirty, nothing staged.

## Fill-in scope (state these in the task prompt)
- **Task goal:** <what subsystem/behavior is being audited>
- **Focus:** <narrow the scope; note what NOT to re-audit>
- **Invariants to check against:** <the relevant E1 invariants>
- **Files to inspect:** <list>

## Report format — produce exactly these sections
**A. Pre-flight git confirmation** — table: branch / latest commit / staged / only-dirty-file, each with expected vs actual.
**B. Files inspected** — the read-only set (note `reference.pdf` not opened).
**C. PASS / WARN / BLOCKER table** — one row per checked area with a verdict.
**D. Exact findings** — each with `file_path:line` references and a concrete failure scenario / rationale.
**E. Must fix before next phase** — blocking items only (empty if none).
**F. Can defer until later** — non-blocking items.
**G. Better alternatives / cleanup ideas** — safer/cleaner options; do not implement.
**H. Recommended next action** — either "No edit needed; proceed to <next>" or "Edits recommended; ask for approval before changing files."

## Severity guide
- **BLOCKER** — would cause a crash, wrong result, data/label corruption, split leakage, or an unsafe/irreversible action if run as-is.
- **WARN** — a real risk or inconsistency that has a workaround, is gated by another check, or matters only in a later phase.
- **PASS** — correct and consistent; state briefly why.
