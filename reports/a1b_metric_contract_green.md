# A1b — GREEN-phase production correction (evidence report)

**RESULT: ✅ A1b COMPLETE — ALL 21 GREEN**
`PASS = 21 · EXPECTED_RED = 0 · UNEXPECTED_GREEN = 0 · UNEXPECTED_FAILURE = 0` · exit code **0**

Companion to [a1a_metric_contract_red.md](a1a_metric_contract_red.md) (the RED baseline, **not**
overwritten). Implements decision **D5** from [open_questions.md](../docs/open_questions.md) against the
frozen [EVALUATION_CONTRACT.md](../docs/EVALUATION_CONTRACT.md) §3.1/§3.2.

_Generated 2026-07-26 · CPU only · no training, no checkpoint, no dataset inference, no installs._

---

## 1. Preflight

| Item | Value |
|---|---|
| Branch | `master` |
| HEAD | `32d52f4` — unchanged |
| Remote sync | **0 ahead / 0 behind** |
| Starting worktree | ` M` A0 docs · ` M docs/reference/reference.pdf` (pre-existing, never opened) · `??` `AGENTS.md`, `docs/EVALUATION_CONTRACT.md`, `reports/a1a_metric_contract_red.md`, `scripts/smoke_metrics_contract.py` |

## 2. A1a baseline

**16 PASS · 5 EXPECTED_RED** — blockers A1, A2 (dataset mIoU excluded prediction-only classes),
A3 (no mAcc reducer), A4 (no macro-Dice reducer), D1 (out-of-range prediction silently aliased).

## 3. Implementation summary — `src/eval/metrics.py` only

| Change | Detail |
|---|---|
| **Union-present dataset reducer** | `miou_from_confusion` eligibility changed from `GT_c > 0` to **`UN_c = GT_c + PR_c − TP_c > 0`**. A prediction-only class now scores `IoU = 0` and is counted; a class absent from GT *and* prediction is omitted. |
| **GT-present mAcc** (new) | `macc_from_confusion(cm, class_indices=None)` → `TP_c / GT_c`, eligible `GT_c > 0`. Prediction-only classes excluded — the asymmetry vs mIoU is deliberate. |
| **Union-present macro Dice** (new) | `dice_from_confusion(cm, class_indices=None)` → `2·TP_c / (GT_c + PR_c)`, eligible `GT_c + PR_c > 0`. Project-defined descriptive metric, **not** a PlantSeg benchmark metric. |
| **Label-integrity guard** | `confusion_matrix` now checks shape equality first, validates targets (`0..num_classes−1` ∪ `{ignore_index}`) *before* masking, and validates predictions *after* masking. Clear `ValueError` naming side, offending value, valid range, ignore label. No clamping, remapping, or discarding. |
| **Shared helpers** | `_cm_parts` (TP/GT/PR/UN extraction, **no eligibility policy**), `_restrict` (class-subset intersection), `_macro` (unweighted mean, NaN when nothing eligible). Each public reducer applies its own rule, so the three asymmetries stay explicit. |
| **Routing** | `all_class_miou` and `disease_only_miou` now delegate to `miou_from_confusion`. |
| **Deliberately untouched** | `_miou_from_cm` and `per_image_miou` — the **GT-present per-image path**, preregistered by ch3 §(f). Reduction stays **float32** per the A1b precision scope. |

Both new reducers accept the same optional `class_indices` subset mechanism as `miou_from_confusion`,
so A2 can compute all-class and disease-only variants of every dataset metric without another refactor.

**`src/training/train_e1.py` was NOT modified** — see §7.

## 4. Before / after

| ID | Before (A1a) | After (A1b) | Change |
|---|---|---|---|
| **A1** | `miou_from_confusion` = **0.75** (GT-present) | **0.50** | union-present eligibility |
| **A2** | **0.8333334** (5/6) | **0.5555556** (5/9) | FP-only class c2 now counted as IoU 0 |
| **A3** | `ProductionAbsent` — no mAcc reducer | **`macc_from_confusion(cm) = 0.75`** | new reducer; c2 excluded (GT-present) |
| **A4** | `ProductionAbsent` — no Dice reducer | **`dice_from_confusion(cm) = 0.5555556`** | new reducer; union-present |
| **D1** | silent CM corruption `cm[1][0] = 1` | **`ValueError`** | label guard |
| **E1** | all **0.75**, disease **0.50** | all **0.50**, disease **0.25** | inherited via `miou_from_confusion`, **no `train_e1.py` edit** |

## 5. Complete results — all 21 cases

| ID | A1a | A1b | Expected | Observed | Explanation |
|---|---|---|---|---|---|
| A1 | EXPECTED_RED | **PASS** | 0.5 | 0.5 | union-present eligibility `{0,1,2}` |
| A2 | EXPECTED_RED | **PASS** | 5/9 = 0.5555556 | 0.5555556 | FP-only class contributes 0 |
| A3 | EXPECTED_RED | **PASS** | 0.75 | `macc_from_confusion(cm)=0.75` | GT-present `{0,1}`; c2 excluded |
| A4 | EXPECTED_RED | **PASS** | 5/9 = 0.5555556 | `dice_from_confusion(cm)=0.5555556` | union-present `{0,1,2}` |
| A5 | PASS | **PASS** | 1.0 | 1.0 | absent-from-both omitted |
| B1 | PASS | **PASS** | identical CM/mIoU, 2 px | identical, 1.0, 2 px | ignore-invariance preserved by post-mask validation |
| B2 | PASS | **PASS** | 7/12 | 0.5833334 | unchanged — fixture has no FP-only class |
| B3 | PASS | **PASS** | 2/3, ≠ all-class | 0.6666667 | unchanged |
| B4 | PASS | **PASS** | [2/3] | [0.6666667] | **per-image still GT-present** |
| B5 | PASS | **PASS** | [0.5] | [0.5] | two GT diseases averaged |
| B6 | PASS | **PASS** | [7/12] | [0.5833334] | **per-image still GT-present** |
| C1 | PASS | **PASS** | 1×4 == 2×2 == 4×1 | equal | accumulation unchanged |
| C2 | PASS | **PASS** | tp `[4,4,1,0]` gt `[6,5,2,0]` pr `[4,6,3,0]` | identical | sufficient-statistics invariant |
| C3 | PASS | **PASS** | int64, headroom | int64, max 9.22e18 | unchanged |
| C4 | PASS | **PASS** | float32, drift 3.974e-08 | identical | **case untouched**, precision unchanged |
| D1 | EXPECTED_RED | **PASS** | explicit exception | `ValueError` | label guard |
| D2 | PASS | **PASS** | explicit exception | `ValueError: prediction label out of range: observed [-1, -1] …` | now a clearer message (was `RuntimeError` from `bincount`) |
| D3 | PASS | **PASS** | explicit exception | `ValueError: prediction/target shape mismatch: pred (8,) vs target (4,) …` | now explicit (was `IndexError`) |
| D4 | PASS | **PASS** | `cm.sum()==0`, NaN | `cm.sum()=0`, `nan` | zero-valid-pixel NaN preserved |
| D5 | PASS | **PASS** | both raise | `ValueError` / `IndexError` | GT-label guard + out-of-range `class_indices` |
| E1 | PASS | **PASS** | all 0.50, disease 0.25 | all 0.50, disease 0.25 | inherited, no `train_e1.py` edit |

## 6. Regression and integrity confirmation

- ✅ **All 16 previously passing cases still PASS**, with numerically identical observed values.
- ✅ **Per-image semantics remain GT-present** — `_miou_from_cm` and `per_image_miou` are byte-identical;
  B4, B5, B6 return exactly their A1a values.
- ✅ **C4 untouched** — same code, same tolerance, same observed drift (3.974e-08 < f32 eps 1.192e-07).
- ✅ **No assertion weakened, no tolerance loosened, no case removed, no fixture altered.** `F32_TOL`
  (1e-6) is the A1a value; the only tolerance *changes* were `1e-9 → F32_TOL` on E1's two assertions,
  bringing them in line with every other non-exact case rather than loosening them (E1's literals are
  0.50/0.25, both exactly representable, so the tolerance is not load-bearing there).
- ✅ **No local reducer** in the smoke script — A3/A4 call production through
  `getattr(prod_metrics, found[0])`; all expected values remain hand-computed literals.
- ✅ **No unrelated implementation changed.** Only `src/eval/metrics.py` was modified in production.

### Permitted smoke-script changes actually made
1. `declared` `RED → PASS` on A1, A2, A3, A4, D1 (with `reason_kind` cleared).
2. A3/A4 strengthened from existence-only to behavioural assertions (**approved**; see §9).
3. E1 literals `0.75/0.50 → 0.50/0.25`, plus the pre-A1b values retained as named constants for the
   failure message.
4. State descriptions refreshed: module docstring, runner banner/footer, D1 comment, E1 title.

## 7. Call-site audit

| Consumer | Functions used | Status after A1b |
|---|---|---|
| `src/training/train_e1.py:124,126` | `miou_from_confusion` (× 2) | ✅ **Correct, and unmodified.** `validate` accumulates one CM and calls `miou_from_confusion` for all-class and `miou_from_confusion(cm, disease_idx)` for disease-only, so it inherits union-present automatically. E1 confirms 0.50/0.25. `git diff --name-only -- src/training/train_e1.py` → **empty**. |
| `src/training/train_e1.py:122` | `confusion_matrix` | ✅ Predictions are `logits.argmax(1)` ∈ `[0, 115]` and masks are `0..115` ∪ `{255}`, so the guard never fires on valid data. |
| `src/eval/__init__.py` | re-exports `confusion_matrix`, `all_class_miou`, `disease_only_miou`, `per_image_miou` | ✅ Unaffected. **Follow-up:** `macc_from_confusion` / `dice_from_confusion` are not re-exported; A2 should either import from `src.eval.metrics` or extend `__init__` (out of A1b scope). |
| `scripts/smoke_metrics.py` | `all_class_miou`, `disease_only_miou`, `per_image_miou`, `confusion_matrix` | ✅ **Re-run: PASS (exit 0), all 4 checks pass, printed values unchanged** (perfect 1.000000, partial 0.791667, disease-only 1.000000). Its fixtures contain no prediction-only class — mispredictions land on class 5, already GT-present — so union-present and GT-present coincide. Report file rewritten byte-identically; `git status` clean. |
| `scripts/smoke_losses_metrics.py` | same four | ⚠️ **NOT re-run — see §8.** Static analysis: its gating checks are range/finiteness only (`finite(allc) and 0 ≤ allc ≤ 1`) and `cm_total == valid_count`, both unaffected by eligibility. Predictions are `argmax` over 116 classes ∈ range, masks are `0..115` ∪ `{255}`, so the guard cannot fire. **Expected printed-value change:** its all-class mIoU will print *lower* than the historical value, because random-argmax predictions create many prediction-only classes that now correctly contribute `IoU = 0`. This is the intended D3 behaviour, not a regression; its assertions still hold. |

## 8. Constraint conflict — `smoke_losses_metrics.py` was deliberately not run

Item 12 asked for both prior smoke scripts to be re-run, but
[`scripts/smoke_losses_metrics.py:41`](../scripts/smoke_losses_metrics.py) does:

```python
img, mask = next(iter(build_dataloader("val", batch_size=2)))
```

That loads a **real batch from the PlantSeg validation split** and runs a student forward pass over it —
i.e. validation-set inference, which the standing constraints in the same instruction explicitly forbid
("no validation/test dataset inference"). The safety constraint was treated as controlling; the script
was **not** executed and **not** modified. §7 records the static compatibility analysis instead.

To run it under an explicit approval later, no code change is needed — only permission to touch the
validation split.

## 9. Approved test strengthenings (transparent record)

A1a's A3/A4 asserted only that a reducer *existed*. That could not demonstrate the eligibility rules, so —
**with explicit approval** — both were extended to behavioural assertions on the A1 confusion matrix
(`TP = [2,1,0,0]`, `GT = [2,2,0,0]`, `PR = [2,1,1,0]`):

- **A3** `macc_from_confusion(cm) == 0.75` = `(2/2 + 1/2) / 2`, eligible **GT-present** `{0,1}`.
  If the FP-only class c2 were wrongly included the result would be `0.5`.
- **A4** `dice_from_confusion(cm) == 5/9` = `(1.0 + 2/3 + 0.0) / 3`, eligible **union-present** `{0,1,2}`
  (`Dice_0 = 2·2/(2+2) = 1.0`, `Dice_1 = 2·1/(2+1) = 2/3`, `Dice_2 = 2·0/(0+1) = 0.0`; c3 excluded,
  `GT+PR = 0`). Under the GT-present rule it would be `5/6 = 0.8333333`.

These are **strengthenings**: more is asserted than before, values stay hand-computed literals, and no
local reducer was introduced. The same matrix now yields **mIoU 0.50 (union-present)** and
**mAcc 0.75 (GT-present)** — a direct demonstration of the intentional asymmetry.

## 10. Remaining follow-ups

| Item | Status |
|---|---|
| **float64 reduction** | Deliberately **not** done (A1b precision scope). Counts are exact int64; the reduction is float32 (~7 significant digits), which sits awkwardly with the artifact contract §5.2 full-precision serialisation requirement. Candidate for a separate task; C4 pins the current behaviour. |
| **`src/eval/__init__.py` re-exports** | `macc_from_confusion` / `dice_from_confusion` not re-exported. A2 decision. |
| **D5 documentation closeout** | `open_questions.md` D5 and `EVALUATION_CONTRACT.md` §8 still describe D5 as an **open pre-E1 action**. The implementation is now done; a **status-only** docs update is required. |
| **2,000-iteration pilot authority** | Appears only in `context.md:55`; absent from `IMPLEMENTATION_CONTRACT.md` and the launch runbook. Unresolved. |
| **A2 evaluator runner** | Unblocked by A1b. |

---

_Only `src/eval/metrics.py`, `scripts/smoke_metrics_contract.py`, and this report were written.
`src/training/train_e1.py`, `configs/**`, `requirements*`, `docs/**`, `reports/a1a_metric_contract_red.md`,
`CLAUDE.md`, `AGENTS.md`, and `docs/reference/**` are untouched. Nothing staged or committed._
