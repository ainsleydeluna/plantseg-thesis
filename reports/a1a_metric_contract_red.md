# A1a — RED-phase metric-contract tests (evidence report)

**RESULT: ✅ A1a COMPLETE — EXPECTED RED CONFIRMED**
`PASS = 16 · EXPECTED_RED = 5 · UNEXPECTED_GREEN = 0 · UNEXPECTED_FAILURE = 0` (21 cases, exit code **0**)

> **The five RED cases are the deliverable of A1a. They are NOT production passes.** They record exactly
> where `src/eval/metrics.py` and `src/training/train_e1.py::validate` do not yet satisfy
> [EVALUATION_CONTRACT.md](../docs/EVALUATION_CONTRACT.md). A1b must turn precisely these five GREEN by
> correcting production code, **rerunning these same cases unchanged**.

_Generated 2026-07-26 · read-only w.r.t. production code · CPU only · no training, no checkpoint, no
dataset access, no installs, no downloads._

---

## 1. Repository preflight

| Item | Value |
|---|---|
| Branch | `master` |
| HEAD | `32d52f4` (unchanged by A1a) |
| Remote sync | **0 ahead / 0 behind** `origin/master` |
| Pre-existing dirty | `docs/reference/reference.pdf` (standing policy — never opened) |
| Pre-existing untracked | `AGENTS.md` (untouched) |
| A0 / A0-FIX changes present | `docs/IMPLEMENTATION_CONTRACT.md`, `docs/open_questions.md`, `docs/EVALUATION_CONTRACT.md` |
| D5 already implemented? | **No** — verified before writing tests |

**Preflight confirmations (all made *before* any test was written):**
- `src/eval/metrics.py:37` still reads `present = row > 0` → dataset-level mIoU is **GT-present**.
- `src/eval/metrics.py` exports only `confusion_matrix`, `_miou_from_cm`, `miou_from_confusion`,
  `all_class_miou`, `disease_only_miou`, `per_image_miou`. **No mAcc reducer. No macro-Dice reducer.**
- The `Dice` occurrences in `train_e1.py` are `CombinedCEDiceLoss` — a **loss**, not a metric reducer.

## 2. Production interfaces inspected

| Interface | Module |
|---|---|
| `confusion_matrix`, `miou_from_confusion`, `_miou_from_cm`, `all_class_miou`, `disease_only_miou`, `per_image_miou` | `src.eval.metrics` |
| `validate` | `src.training.train_e1` |

Both imported by their real module paths. `prod_metrics.__file__` resolved to
`<repository root>/src/eval/metrics.py`, confirming no shadowing.

## 3. Test environment

Local Anaconda interpreter · Python **3.13.5** · torch **2.9.1+cpu** · `PYTHONIOENCODING=utf-8` ·
CPU only. Synthetic tensors and a 4-class space (`C = 4`, background 0, "diseases" 1–3) throughout — the
contract's rules are class-count agnostic, and a tiny space makes every expected value hand-checkable.

---

## 4. Results

| ID | Declared | Classification | Asserted (contract) | Observed (production) | Production reason |
|---|---|---|---|---|---|
| **A1** | RED | **EXPECTED_RED** | mIoU = **0.5** | **0.75** | `present = row > 0` drops the FP-only class c2 from the denominator |
| **A2** | RED | **EXPECTED_RED** | mIoU = **5/9** = 0.5555556 | **0.8333334** (= 5/6) | same GT-present exclusion; c2 (GT 0, PR 1) omitted |
| **A3** | RED | **EXPECTED_RED** | an mAcc reducer exists | **ABSENT** | `src.eval.metrics` has no `macc_from_confusion` / `macc` / `mean_accuracy` / `accuracy_from_confusion` / `class_accuracy` |
| **A4** | RED | **EXPECTED_RED** | a macro-Dice reducer exists | **ABSENT** | no `dice_from_confusion` / `macro_dice` / `mdice` / `dice_from_cm` / `macro_dice_from_confusion` |
| **A5** | PASS | PASS | 1.0 | 1.0 | absent-from-both already omitted — both conventions agree |
| **B1** | PASS | PASS | identical CM + mIoU; 2 valid px | CM equal, mIoU 1.0, 2 px | ignore-255 masks on GT before accumulation |
| **B2** | PASS | PASS | 7/12 | 0.5833334 | background included in all-class |
| **B3** | PASS | PASS | 2/3, and ≠ all-class | 0.6666667, ≠ 0.5833334 | excludes exactly class 0; class 1 retained |
| **B4** | PASS | PASS | [2/3] | [0.6666667] | per-image disease-only = IoU of the single GT disease |
| **B5** | PASS | PASS | [0.5] | [0.5] | **two** GT diseases averaged correctly — no one-disease assumption |
| **B6** | PASS | PASS | [7/12] | [0.5833334] | per-image all-class = mean over GT-present classes |
| **C1** | PASS | PASS | 1×4 == 2×2 == 4×1 | all equal | accumulation is order-independent |
| **C2** | PASS | PASS | tp `[4,4,1,0]` gt `[6,5,2,0]` pr `[4,6,3,0]` | identical | per-image counts sum to CM diag/rows/cols |
| **C3** | PASS | PASS | int64, headroom > 409,206,784 | int64, max 9.22e18 | ~22 billion× headroom at full test scale |
| **C4** | PASS | PASS | int64 counts, float32 reduction, 0 < drift < f32 eps | drift **3.974e-08**, eps 1.192e-07 | reduction casts to float32 (see §6) |
| **D1** | RED | **EXPECTED_RED** | explicit exception | **silently corrupted CM** `cm[1][0] = 1` | `k = target*C + pred` aliases pred 4 onto row 1; no range guard |
| **D2** | PASS | PASS | explicit exception | `RuntimeError: bincount only supports 1-d non-negative integral inputs.` | fails loudly (message opaque) |
| **D3** | PASS | PASS | explicit exception | `IndexError: shape of the mask [4] … indexed tensor [8]` | fails loudly |
| **D4** | PASS | PASS | `cm.sum()==0` and mIoU is NaN | `cm.sum()=0`, mIoU `nan` | explicit non-finite undefined signal |
| **D5** | PASS | PASS | both variants raise | `RuntimeError` / `IndexError` | GT label ≥ C and out-of-range `class_indices` both raise |
| **E1** | PASS | PASS | all = 0.75, disease = 0.50 (GT-present) | all **0.75**, disease **0.50** | `validate` matches the current reducer exactly; contract requires **0.50 / 0.25** (Δ **+0.25 / +0.25**) |

### Summary counts
`PASS = 16` · `EXPECTED_RED = 5` · `UNEXPECTED_GREEN = 0` · `UNEXPECTED_FAILURE = 0` · exit **0**.

---

## 5. Hand-computed arithmetic

Notation: `TP_c` intersection · `GT_c` row sum · `PR_c` column sum · `UN_c = GT_c + PR_c − TP_c`.

### §A1 — the headline divergence
`target = [0,0,1,1]`, `pred = [0,0,1,2]` → CM rows = GT, cols = pred:

```
        pred0 pred1 pred2 pred3
 GT0  [   2     0     0     0 ]
 GT1  [   0     1     1     0 ]
 GT2  [   0     0     0     0 ]
 GT3  [   0     0     0     0 ]
```

| c | TP | GT | PR | UN | IoU |
|---|---|---|---|---|---|
| 0 | 2 | 2 | 2 | 2 | 2/2 = **1.0** |
| 1 | 1 | 2 | 1 | 2 | 1/2 = **0.5** |
| 2 | 0 | **0** | 1 | 1 | 0/1 = **0.0** ← false-positive-only |
| 3 | 0 | 0 | 0 | 0 | 0/0 = undefined |

- **GT-present** (current production) — eligible `{0,1}` → `(1.0 + 0.5) / 2 = 1.5/2 = ` **0.75**
- **Union-present** (contract §3.1) — eligible `{0,1,2}` → `(1.0 + 0.5 + 0.0) / 3 = 1.5/3 = ` **0.5**

Production returned **0.75**. Divergence **+0.25**.

### §A2 — FP-only drawn from a background pixel
`target = [0,0,0,1,1]`, `pred = [0,0,2,1,1]`:

| c | TP | GT | PR | UN | IoU |
|---|---|---|---|---|---|
| 0 | 2 | 3 | 2 | 3 | 2/3 |
| 1 | 2 | 2 | 2 | 2 | 1.0 |
| 2 | 0 | **0** | 1 | 1 | 0.0 |

- GT-present `{0,1}` → `(2/3 + 1)/2 = (5/3)/2 = ` **5/6 = 0.8333333**
- Union-present `{0,1,2}` → `(2/3 + 1 + 0)/3 = (5/3)/3 = ` **5/9 = 0.5555556**

Production returned **0.8333334**.

### §B2 / §B6 — background inclusion and per-image all-class
`target = [0,0,1,1]`, `pred = [0,1,1,1]`:

| c | TP | GT | PR | UN | IoU |
|---|---|---|---|---|---|
| 0 | 1 | 2 | 1 | 2 | 1/2 = 0.5 |
| 1 | 2 | 2 | 3 | 3 | 2/3 |

- All-class GT-present → `(1/2 + 2/3)/2 = (3/6 + 4/6)/2 = (7/6)/2 = ` **7/12 = 0.5833333**
- Disease-only GT-present (§B3/§B4), eligible `{1}` → **2/3 = 0.6666667**

The two differ by 0.0833, so B3 simultaneously proves class 0 is excluded (else 7/12) and class 1 is
retained (else NaN).

### §E1 — trainer validation parity and the contract delta
Same CM as §A1, produced through the real `train_e1.validate` with a deterministic one-hot dummy model:

| Metric | Eligible set | Value |
|---|---|---|
| all-class, **GT-present** (production) | `{0,1}` | `(1.0 + 0.5)/2` = **0.75** |
| all-class, **union-present** (contract) | `{0,1,2}` | `(1.0 + 0.5 + 0.0)/3` = **0.50** |
| disease-only, **GT-present** (production) | `{1}` | **0.50** |
| disease-only, **union-present** (contract) | `{1,2}` | `(0.5 + 0.0)/2` = **0.25** |

`validate` returned **0.75 / 0.50** — exact parity with the current reducer, and its accumulated CM is
byte-identical to a direct `confusion_matrix` call. **Δ all-class = +0.25 · Δ disease-only = +0.25.**

---

## 6. Run 1 → run 2: a test defect, disclosed

**The first execution produced 4 `UNEXPECTED_FAILURE`s (B2, B3, B4, B6) and exited 1.** They are recorded
here rather than quietly overwritten.

Cause: production reduces in **float32** (`_miou_from_cm` does `torch.diag(cm).float()`), so `7/12` came
back as `0.5833333730697632` against the float64 literal `0.5833333333333334` — a drift of
**3.97e-08**. My tolerance was `1e-9`, **below float32 epsilon (1.19e-07)**, so those assertions were
arithmetically impossible to satisfy regardless of production correctness.

This was a **defect in the test, not in production**. The fix was to set a float32-appropriate tolerance
(`F32_TOL = 1e-6`), which still separates the competing conventions by ~5 orders of magnitude
(`|7/12 − 2/3| = 8.3e-02`). **No assertion's substance was weakened and no production code was touched.**

The finding was then promoted to a first-class case, **C4**, which asserts the precision regime explicitly
(int64 counts, float32 reduction, `0 < drift < float32 eps`).

> **Carry-forward for A1b/A2 (not fixed here):** counts are exact `int64` but the reduction is `float32`,
> giving ~7 significant digits. Artifact contract §5.2 mandates full-precision float serialisation — writing
> 17 significant digits of a float32 value stores ~10 digits of noise. A1b should consider reducing in
> **float64** from the int64 counts, which costs nothing and makes the artifact honest.

---

## 7. Compliance confirmation

- ✅ **No local metric reducer introduced.** The script contains no union-present reducer, no mAcc, no Dice,
  and no copy of a production formula under another name. Every expected value is a literal hand-computed
  constant, derived in §5. Absences (A3/A4) are probed through the real module via `hasattr`.
- ✅ **Production code not modified.** `src/**`, `configs/**`, `requirements*`, `docs/**` and all prior
  `reports/**` are untouched.
- ✅ **No training, no checkpoint, no dataset inference.** No validation or test split was opened; the only
  model is a deterministic one-hot dummy over a 2×2 synthetic image.
- ✅ **No install, download, GPU use, staging, or commit.**
- ✅ **Memory-safe.** C3 reasons about the 409,206,784-pixel scale using `torch.iinfo` and a single-element
  int64 tensor — nothing large was allocated.

---

## 8. Blockers A1b must turn GREEN

| ID | Required production change |
|---|---|
| **A1, A2** | Add a **union-present** dataset-level reduction (`UN_c > 0`) and use it for dataset-level all-class and disease-only mIoU. Keep the GT-present reduction for **per-image** metrics. |
| **A3** | Add a dataset-level **mAcc** reducer, `TP_c / GT_c` over **GT-present** classes — deliberately a different denominator from mIoU (contract §3.1). |
| **A4** | Add a dataset-level **macro Dice** reducer, `2·TP_c / (GT_c + PR_c)`, using the **union-present** rule. |
| **D1** | Guard prediction/target label ranges so an out-of-range label **raises** instead of aliasing into a neighbouring confusion-matrix row. |
| **E1** *(no reclassification)* | After the above, `train_e1.validate` must return **0.50 / 0.25** on the §E1 fixture. E1's assertion changes from the GT-present literals to the contract literals — the **only** case whose expected values change, and it is a deliberate, documented consequence of D5. |

Out of scope for A1b: the float32→float64 reduction (§6 carry-forward) is recommended but separable, and no
unrelated refactor is proposed.

## 9. Cases A1b must rerun unchanged

**All 21 cases** in `scripts/smoke_metrics_contract.py`, with **A1–A5, B1–B6, C1–C4, D1–D5 byte-identical**.
After A1b:

- **A1, A2, A3, A4, D1** flip `declared` from `RED` to `PASS` and must classify **PASS**.
- **The other 16 must remain PASS** — any regression there is an `UNEXPECTED_FAILURE` and blocks A1b.
- **E1**'s literals move to the contract values (0.50 / 0.25) as itemised above.

_End of A1a. Only `scripts/smoke_metrics_contract.py` and this report were created; nothing staged or
committed; `docs/reference/reference.pdf` never opened; `AGENTS.md` untouched._
