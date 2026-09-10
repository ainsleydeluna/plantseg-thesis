# Open Questions — every OPEN / NEED_TO_CONFIRM and how it gets resolved

Companion to [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) and [conflicts.md](conflicts.md).
No value here is guessed. Each item lists **how** it will be settled. Nothing below blocks Week-1
analysis/docs/smoke-test work; items flagged "before E1" are pre-training gates.

Source tags as in the contract.

---

## Empirically resolved (2026-06-27) — formerly OPEN conflicts

### 1. ✅ RESOLVED — `num_classes` = 116
- **Result:** `np.unique()` over **all 7,774** annotation PNGs → contiguous **0–115** (max non-ignore 115).
  Background = **0**, diseases = **1–115** (`mask = category_id + 1`, 99.85% consistent). Wei's "0–114" are
  the COCO `category_id`s. All-class classifier output = **116**.
- **Status:** **RESOLVED.** Evidence: [reports/dataset_report.md](../reports/dataset_report.md). Do **not**
  change masks; apply `ImageOps.exif_transpose` to images (never masks) before pairing/transform.

### 2. ✅ RESOLVED (D1 + A0-FIX 2026-07-26) — disease-only / background convention (`reduce_zero_label`)
> Class-index mapping **confirmed against the official PlantSeg source**; only a cosmetic naming caveat
> remains (index 0 is named `''`, not `"background"`). Details at the end of this item.
> **D1 (2026-07-01) — metric policy:** **all-class mIoU is the checkpoint-selection metric and the headline
> E1/E-stage validation metric** (`src/training/train_e1.py` already selects the best-val checkpoint on
> all-class mIoU and logs disease-only mIoU as a **secondary/provisional** metric). Confirming the exact
> official disease-only exclusion convention is still useful for **Ch4** lesion-focused interpretation/
> reporting, **but it does NOT block E1 training, checkpointing, or all-class reporting.** A future switch to
> disease-only headline/checkpointing requires a separate explicit decision. This is **separate from — and
> does not change — the manuscript's per-image disease-only mIoU _primary inferential_ unit** for the Ch4
> hypothesis tests (see `IMPLEMENTATION_CONTRACT.md` §(f)).

- **Empirical finding:** background = **index 0** (in 7773/7774 masks, 80.6% of pixels); diseases = 1–115.
  So disease-only mIoU should **exclude index 0**.
- **✅ SUBSTANTIVE QUESTION RESOLVED (2026-07-26, A0-FIX).** The resolution step below named
  `https://github.com/tqwei05/PlantSeg` as the source to check. It has now been checked, and the official
  benchmark source confirms the empirical mapping outright:
  - `mmseg/datasets/plantseg115.py` — `METAINFO['classes'] = ('', <115 disease names>)`: index **0** is
    the non-disease slot; indices **1–115** are the 115 diseases, in `category_id + 1` order.
  - `configs/segnext/segnext_mscan-l_…plantseg115-512x512.py` — `decode_head.num_classes = **116**`.
  - `configs/_base_/datasets/plantseg115.py` — **`reduce_zero_label=False`**, set in `LoadAnnotations`
    for train *and* test and in both dataloader dataset dicts.

  The official benchmark therefore uses **exactly** the thesis's class space: 116 outputs, non-disease at
  index 0, diseases 1–115, **no label remap**. Excluding index 0 for disease-only metrics is now confirmed
  by the official source, not merely empirically inferred.
- **Residual caveat (cosmetic only):** the official METAINFO names index 0 with the **empty string** `''`
  rather than the literal token `"background"`, and the COCO `categories` list is still empty. So no source
  *spells* the word "background" — but the slot's role and index are no longer in doubt.
- **Consequence:** disease-only metric definitions are **no longer marked PROVISIONAL** on account of the
  class mapping. Exact arithmetic is frozen in [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md) §3.
  `[empirical; official PlantSeg source; ch3; ctx]`
- **Minor data-quality note:** 12/7,774 masks (0.15%) have a disease value off by one from their
  `Metadata` `Index` (logged in the report) — does not affect the num_classes/background conclusion.

### D2 — E1 gradient clipping `max_norm` — ✅ RESOLVED (D-A: unclipped-E1 deviation, 2026-07-01)
**D2 / D-A RESOLVED:** E1 will proceed unclipped as an explicit documented deviation from Chapter 3's
value-less "global-norm, throughout" wording. The existing `src/training/train_e1.py` hook remains
available through `--grad-clip-norm` if divergence or instability is observed, but no numeric `max_norm` is
chosen by default. This resolves the pre-real-E1 decision gate without changing training code.
`configs/e1_student.py` records `grad_clip_max_norm=None`; `gradient_clipping="global_norm"` is the method
family, not a numeric value. `[ch3 §C; D2; D-A]`

---

## Resolved 2026-09-01 (B31 / B31c) — E1 launch-blocking fixes and their recorded deviations

### D25 — `num_workers` is a reproducibility parameter — ✅ RESOLVED `[empirical, measured]`
Seed 42 alone does **not** determine the realized augmentation sequence. Measured on a fixed
8-sample / 4-batch train subset at identical seed, `num_workers ∈ {0, 2, 4, 8}` produced **three**
distinct streams (`{0}`, `{2}`, `{4,8}`); the `{4,8}` tie is a probe artifact of
`num_workers >= n_batches` and does not hold at real scale (335 batches/epoch vs 12 workers).
Cross-process reproducibility **at a fixed `num_workers`** is preserved and was re-measured
byte-identical after the B31-5 loader change. Resolution: `num_workers` is pinned as a **reported**
parameter alongside seed 42 in Ch4, and the B31-5 default change (4 → `min(cpu_count-2, 12)`) is
recorded as changing the realized sequence but not the distribution or any locked value.
**Manuscript follow-up (outside this repo):** ch3 §D's reproducibility paragraph is under-specified
as written — it pins seeds, cuDNN flags and `CUBLAS_WORKSPACE_CONFIG` but not `num_workers`, so two
runs satisfying every stated condition can still differ. `[ch3 §D; project; B31-5/A1]`

### D26 — resume is not bitwise-identical to an uninterrupted run — ✅ RESOLVED as a documented deviation `[project]`
Same register as D-A. `--resume` restores model, optimizer, scheduler and all RNG state, and the LR
curve continues exactly (verified at full float64 against the analytic 80k `PolynomialLR` curve).
Data **order** is not recoverable under the infinite `cycle(train_loader)`, so a resumed run is a
valid E1 run but not a byte-reproduction. It must be reported as resumed if it produces a headline
result. No code change is proposed to make order recoverable: doing so would require persisting
sampler position and is not required by any source. `[project; B31-2]`

> **SUPERSEDED FOR OFFICIAL RUNS (B44, 2026-09-09) — a tightening under changed cost, not a
> correction.** D26 balanced a real trade: salvage the compute already spent against carrying an
> asterisk on a headline result. That balance held while a lost run meant many GPU-hours at
> Secure-tier prices. Under the B44 tier decision
> ([reports/b42_pod_gpu_validation.md](../reports/b42_pod_gpu_validation.md) §7) seed-replicated runs
> go on Community Cloud, where losing a full 13-hour seed costs roughly **$4.40** at ~$0.34/hr
> `[INFERRED — no interruption rate has been measured]`. At that price the trade no longer balances,
> so official runs discard and relaunch from iteration 0. The rule lives at `AGENTS.md`
> § E1 invariants.
>
> **What D26 still governs:** every non-official resumed run — debugging, rehearsals, and the
> *k*-segment resume verification behind D27. Its disclosure requirement is unchanged for those, and
> the `prev_lr` machinery D27 added remains live tooling, scoped out of official runs rather than
> retired. `--resume` is not removed from `train_e1.py`.

### D28 — the λ_logit semantics boundary — ✅ RESOLVED `[project]` (B32/F8 + B32c-2)
ch3 pins *which pixels* enter the Logit-KD KL but not *which grid*. B32 pinned it to the head's
native OS8 64×64. Because λ_logit weights that term, **its numeric value is only meaningful relative
to the grid the term is computed on**, and the move changed the term's magnitude by **1.95×**
(MEASURED against a synthetic teacher, therefore an **upper bound**; gradient contribution 1.76×).

The preregistered grid `{0.25, 0.5, 1, 2, 4}` is geometric with ratio 2, so ~1.95× is about **one
grid step**. Resolution: **the grid is NOT re-centred.** It still brackets a sensible optimum, and
ch3 already handles a boundary winner ("reported as such rather than the grid being extended"), so
F11's preregistration is intact. The optimum is expected roughly one step lower than it would have
been under the old semantics.

**A λ selected under one semantics is not consumable under the other.** Guarded, not just
documented: `configs/distill.py` defines `LOGIT_KD_SEMANTICS = "logitkd@os8-64x64-of-512"` (and
records the superseded `"logitkd@full-512x512-upsampled"`); every E2/E3 checkpoint and a
`<stage>_run_meta.jsonl` carry it; `--lambda-semantics` is optional but checked when declared, and a
mismatch refuses unless `--allow-semantics-mismatch` is passed — with that override stamped durably
into both artifacts. Declaring is optional by design: the sweep runs *produce* the tag rather than
consume it, so requiring it there would add friction at the point of lowest risk; the risk is a λ
read out of a Ch4 table months later and passed to a re-run.

**Ch4 obligation:** state that the λ sweep ran under OS8 semantics, and report λ with its tag.
`[ch3 §C.1; project; B32/B32c]`

### D27 — scaffold check coverage under resume — ✅ RESOLVED `[project]`
B31c V1 asked whether a resumed run can report success while exercising fewer checks than a fresh
run. It could, in two ways, both now closed or made visible:
1. **LR transitions across a resume boundary were unverified.** `last.pt` now carries `prev_lr`; a
   *k*-segment run leaves **zero** unverified transitions (measured 8/8 on a 3-segment run).
2. **`all([]) is True` printed as `PASS`** when no transition was compared. It now reports
   `SKIPPED`. Deliberately **not fatal** — a legitimate one-iteration resume is not a defect, and
   failing it would make the preflight brittle for no gain.
3. **The already-complete resume path** ran zero checks and printed `RESULT: PASS`. It now prints
   the distinct token `RESULT: NOOP (already complete at iter N)` with exit 0.
Every outcome is distinguishable from the **last line of stdout alone** via
`RESULT: <PASS|FAIL> (n/6 checks exercised, m skipped)`. `scripts/preflight_e1.py` asserts the
exercised count, not merely the exit code. **Residual:** a fresh one-iteration run still legitimately
exercises 5/6; this is visible, not silent. `[project; B31c V1]`

---

## Resolved 2026-07-26 (A0) — evaluation metric semantics + result-artifact contract

Full specification: **[EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md)** (frozen). Summarised here because
this file is the repo's decision log of record.

### D3 — dataset-level metric eligibility = **union-present** (mIoU); **GT-present** (mAcc) — ✅ RESOLVED
Dataset-level all-class mIoU includes every class with `GT_c + PR_c − TP_c > 0`, so a class absent from
ground truth but **predicted** contributes `IoU = 0` rather than being dropped. `mAcc` uses
`intersect/label`, hence GT-present. **The asymmetry is genuine benchmark behaviour and must not be
"harmonised".** `aAcc` is a diagnostic, never a thesis metric.

**Authority chain (kept separate on purpose — see EVALUATION_CONTRACT.md §2.1):**
1. **Official PlantSeg evaluator config** — `configs/_base_/datasets/plantseg115.py`:
   `val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])`, `reduce_zero_label=False`, and
   **`nan_to_num` is not configured**.
2. **Official PlantSeg dataset metadata** — `mmseg/datasets/plantseg115.py`: index 0 named `''`,
   indices 1–115 the diseases.
3. **Official PlantSeg model config** — `configs/segnext/segnext_mscan-l_…plantseg115-512x512.py`:
   `num_classes = 116`.
4. **Shared evaluator implementation** — MMSegmentation `mmseg/evaluation/metrics/iou_metric.py`.

**Version qualification:** the PlantSeg README says only *"Follow MMSegmentation to build the
environment"* and **pins no MMSegmentation version**. **PlantSeg does not pin v1.2.2**; v1.2.2 is the
*thesis's own* pin (`requirements.lock`, contract §(d) B1) and must never be described as the benchmark's
governing version. The denominator rule follows from authority 1 + authority 4 together.

**Decisive detail — `nan_to_num` unset:** had the benchmark configured it, undefined per-class values
would be replaced by a constant before averaging and the union-present conclusion would not hold. It is
not configured, so the reduction is genuinely NaN-aware.

**Macro Dice is NOT a benchmark metric.** The official evaluator requests `iou_metrics=['mIoU']` only, so
it computes mIoU/mAcc/aAcc and **never `mDice`**. Dice is retained as a **thesis-internal descriptive
extension** that reuses the mIoU eligibility rule for internal consistency only — never as evidence of
comparability with published PlantSeg results.

### D3b — per-image eligibility stays **GT-present** (preregistered) — ✅ RESOLVED, with recorded implications
`[ch3 §(f)]` "per-image absent-class exclusion" and `[ctx:118]` are preserved unchanged. Two consequences
are now documented rather than discovered later: (1) because **one-disease-per-image holds dataset-wide**
`[empirical: trainval_mask_value_audit §3]`, per-image disease-only mIoU is *identically* the IoU of the
image's single true disease class; (2) a wrong-class prediction is penalised only as a false negative on
the true class, with no extra `IoU = 0` term. Mitigation: per-image classwise **sufficient statistics are
persisted**, so union-present per-image variants remain recomputable **without re-running inference**.

### D3c — non-inferiority bootstrap unit = **image, with confusion-matrix re-accumulation** — ✅ RESOLVED
`[ch3 §(f)]` names a *paired BCa CI on dataset-level* `ΔmIoU`, which is a ratio-of-sums and therefore
cannot be reproduced by averaging per-image scalar mIoUs (a mean-of-ratios). Each of the `B = 10,000`
replicates resamples image IDs with replacement using **one shared index vector for both stages**,
re-accumulates each stage's per-class sufficient statistics over the resampled multiset, and recomputes
dataset-level mIoU under D3. This is why per-image scalars alone are an insufficient artifact.

### D4 — result-artifact contract frozen — ✅ RESOLVED
One stage- and condition-neutral contract covers teacher + E1–E7, FP32/PTQ/QAT, val/test, and every
corruption × severity: `summary.json` + `per_image.jsonl` + `sufficient_stats.npz` + `MANIFEST.sha256`
(`schema_version = plantseg-eval/1.0.0`, `metric_protocol = plantseg-metrics/1.0.0`). Undefined values are
**`null` + a status enum** — never `NaN`/`Infinity`/sentinels. Canonical image ID = **file stem**, emitted
**directly with each sample** by an evaluation-only dataset wrapper (never reconstructed after batching,
never inferred from batch position). Official-run eligibility uses a **scoped** worktree rule
(EVALUATION_CONTRACT.md §7.1) — a whole-repo-clean gate is impossible here because `reference.pdf` is
permanently dirty by policy. No new dependency is introduced (NumPy is already in `requirements-e1.txt`).

### D5 — E1 validation dataset-level arithmetic — ✅ RESOLVED (implemented in A1b, 2026-07-26)
**No implementation blocker remains.** Evidence: [reports/a1b_metric_contract_green.md](../reports/a1b_metric_contract_green.md)
— **all 21 frozen metric-contract cases PASS, exit code 0** (A1a recorded the same cases at 16 PASS /
5 EXPECTED_RED before the fix).

**What shipped** (`src/eval/metrics.py` only):
- `miou_from_confusion` — dataset-level, now **union-present** (`UN_c = GT_c + PR_c − TP_c > 0`), so a
  prediction-only class contributes `IoU = 0` and is counted.
- `macc_from_confusion` — **new**, dataset-level, **GT-present** (`GT_c > 0`).
- `dice_from_confusion` — **new**, dataset-level, **union-present**; project-defined descriptive metric.
- `confusion_matrix` — now **rejects invalid labels** (shape mismatch, out-of-range prediction or target)
  with a clear `ValueError` instead of allowing index aliasing through `target * num_classes + pred`.
- `_miou_from_cm` and `per_image_miou` — **unchanged, still GT-present** (the preregistered per-image rule).
- `src/training/train_e1.py` — **not edited**: `validate` already consumed `miou_from_confusion`, so
  checkpoint-selection arithmetic inherited the correction automatically.

**Original rationale, retained for the record.** The binding reason was exact benchmark/protocol fidelity of
every reported all-class mIoU. Checkpoint drift was bounded and modest: val GT holds 115 of 116 classes
(69 absent), and a false-positive-only class changes the denominator but not the numerator, so
`mIoU_union = mIoU_gt-present × 115/116` when class 69 is predicted and they are equal otherwise —
≈ **0.86 % relative** (≈0.13 pp at mIoU ≈ 15 %). D1 (all-class mIoU as the selection metric) is unchanged;
only the arithmetic was corrected. Full record: [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md) §8.

### D6 — E1 pilot / precondition verification — ✅ RESOLVED (2026-07-26, A1c)
Settles the long-standing tension between `[ctx:55]` ("2,000-iteration E1 pilot on RunPod") and the
absence of any such gate from the contract and the launch runbook. Chapter III was searched directly
(text extracted read-only, validated against known anchors — `80,000` ×5, `iteration` ×39, `mIoU` ×354 —
before any conclusion was drawn). **Three distinct things were being conflated:**

**A. `[ch3]` REQUIRED — precondition verification.** Chapter III does require preconditions to be verified
via pilot/smoke checks before the pipeline proceeds:
> *"Tool-specific preconditions (e.g., QNNPACK backend operator support for INT8 Sigmoid in the LR-ASPP
> head, MMSegmentation model-zoo checkpoint availability for SegNeXt-B, and RunPod GPU pod availability)
> are verified during pilot runs before E1 training begins."* `[ch3 ~p.112]`
>
> *"backend operator support is treated as a gating precondition rather than an assumption: a pilot
> quantization smoke test confirms that all required operators … are supported before the quantization
> stages are run."* `[ch3 ~p.113]` · *"QNNPACK backend operator support is verified during pilot
> quantization"* `[ch3 ~p.11–12]`

**Documented source tension, resolved stage-specifically.** The ~p.112 sentence lists all three
preconditions as verified "before E1 training begins", but ~p.113 and ~p.11–12 specifically tie QNNPACK to
*pilot quantization, before the quantization stages*. The **stage-specific reading is adopted**:

| Precondition | Governs | Status |
|---|---|---|
| RunPod GPU/pod availability | **before real E1 execution** | pending (environment) |
| SegNeXt-B / MMSeg teacher-checkpoint availability | **before teacher-dependent execution** (E2/E3 prep) | `docs/B8_checkpoint.md` — none public; in-house fine-tune planned |
| QNNPACK INT8 operator support | **before the quantization stages (E4–E7)** | proxy `PASS_CLEAN` (`docs/b7_result.md`); authoritative run env-gated |

**FP32 E1 is NOT blocked by the QNNPACK item.** It is a pre-quantization gate, tracked but not an E1
launch blocker.

**B. `[ctx]` — the "2,000 iterations" value.** **`2,000` does not appear anywhere in Chapter III** (the
only `2000` matches are the OpenCV citation *(Bradski, 2000)* and numeric-table residue). It originates
solely in `[ctx:55]`, inside that file's operational *"Environment & commands"* section. It is **not part
of the locked methodology** and **does not modify the official 80,000-iteration E1 budget**. It must never
be recorded as a `[ch3]` requirement. Note also that at `max_iters = 2000` with `val_interval = 4000`,
`src/training/train_e1.py` validates **only at termination** (via its `it == max_iters` clause) and never
exercises the ordinary periodic 4,000-iteration validation event.

**C. `[operational]` — optional short training rehearsal → `RECOMMENDED_NOT_BLOCKING`.**
Chapter III preregisters no training pilot, and the launch runbook already requires `verify_env.py` → PASS
and `train_e1.py --dry-run` → PASS on the pod. A short rehearsal may still de-risk an 80k run by exercising
data loading, loss finiteness, validation execution, checkpoint writing, logging, artifact persistence,
runtime/memory estimation, and resume mechanics. **Guardrails:** not an official E1 result · excluded from
Chapter IV comparisons · never touches the test split · selects no final checkpoint · artifacts labelled
provisional/pilot · the official 80,000-iteration run **starts fresh** · no silent continuation from the
rehearsal. **No duration is locked here.** Exercising the regular periodic-validation event would require
more than 4,000 iterations (≈5,000 is a reasonable future operational candidate); the exact value stays a
later execution choice. `[ch3 ~p.112, ~p.113, ~p.11–12; ctx:55; operational]`

### Test count — settled, do not restate as approximate
The authoritative test count is **1,561** `[empirical: dataset_report.md; dataset_audit_summary.md §1]`.
The value **≈1,554 is ratio arithmetic** (7,774 × 0.20) originating in `[ctx:107]`, already dismissed in
`dataset_report.json` as *"empirical split is authoritative"*. Expected defined per-image scores on test =
**1,561/1,561** for both vectors; **no exclusions are preregistered**, so any undefined score is a
data-integrity failure that must abort the run.

---

## Resolved 2026-07-28 (A3-0) — statistical analysis protocol

Full specification: **[STATISTICAL_ANALYSIS_CONTRACT.md](STATISTICAL_ANALYSIS_CONTRACT.md)** (frozen).
Chapter III was read directly (read-only text extraction, anchor-validated) before any decision.
Each item below is marked **`[ch3]`** (preregistered) or **`[project]`** (ch3 silent or internally
ambiguous).

### D7 — the eight-test Holm family — ✅ RESOLVED `[project]` (ch3-internal conflict)
Chapter III contradicts itself. A **§B summary sentence** lists E1→E3 and E3→E6 as though they
complete the eight-test family; **Table 3.6** marks E1→E3 *"reported descriptively … not included in
the eight-test Holm-Bonferroni family"*; and ch3 states **three separate times** that the E3→E6
non-inferiority check is *"reported separately from the eight-test Holm-Bonferroni superiority
family"*. Admitting both would give **ten** members against ch3's own explicit eight-test and
"seven clean + one robustness" structure. **Reconciliation: the comparison table, the formal
hypotheses, and the repeated exclusions govern the summary sentence.** Neither reading is hidden —
both are recorded in the contract's authority ledger. Frozen family (candidate − baseline):
`accuracy_e1_e2`, `accuracy_e2_e3`, `accuracy_e4_e5`, `accuracy_e7_e6`, `accuracy_e4_e7`,
`accuracy_e5_e6`, `accuracy_e1_e6` (clean per-image disease-only mIoU) and `robustness_e1_e6`
(per-image mIoU-C). E1→E3 stays descriptive with BCa CIs but no family p-value.

### D8 — mIoU-C nested aggregation — ✅ RESOLVED `[ch3]` + `[project]` retention of the nested form
Per clean image and stage: per-image **disease-only** mIoU from each of **5 corruptions × 3
severities = 15** cells → average severities 1–3 within each corruption → equal-weight mean of the
five corruption means. Corruptions: motion blur, Gaussian noise, **JPEG compression**, brightness,
fog `[ch3]` — "shot noise" appears nowhere in ch3. Severity 4 descriptive only; severity 5 excluded
`[ch3]`. The nested form is retained even though it equals a flat 15-value mean on a complete grid
`[project]`, so a partial grid cannot be averaged away silently. Built from per-image scores, never
from dataset-level mIoU-C.

### D9 — Wilcoxon configuration and tie policy — ✅ RESOLVED `[ch3]` + `[project]` for degeneracy
`scipy.stats.wilcoxon(d, zero_method='pratt', correction=True, alternative='greater',
method='approx')` `[ch3]` — ch3's software table names `method='approx'`, so no reliance on the
evolving `'auto'` default. **Ties are VALID, not degenerate** `[project]`: repeated or uniformly
tied nonzero differences remain proper inputs because SciPy's tie correction is part of the selected
procedure. Degenerate statuses are limited to `degenerate_all_zero` (p = 1.0, reject false, W/z null,
shifts 0.0) and `insufficient_nonzero_pairs` (< 2 nonzero). Unequal lengths, non-finite values, empty
vectors and identity problems are **integrity errors**, not statuses. SciPy returns NaN for the
all-zero case; no NaN may reach JSON.

### D10 — Holm strict boundary — ✅ RESOLVED `[project]`
`multipletests(..., alpha=0.05, method='holm', is_sorted=False, returnsorted=False)`, canonical order
preserved. ch3 writes the threshold with strict `<`, so **equality does not reject** — at the step
threshold or between the adjusted p-value and α. Statsmodels supplies adjusted p-values; the reject
decision is independently verified against the strict rule and any exact-equality divergence is
recorded. Official finalisation requires all eight IDs exactly once.

### D11 — rank-biserial denominator — ✅ RESOLVED `[project]`
`r_rb = (R₊ − R₋) / (R₊ + R₋)` with **Pratt-compatible ranking**: all absolute differences are
ranked including zeros (average ranks for ties), but **zero-difference ranks enter neither signed
sum**. This matches ch3's signed-rank formulation, lets zeros influence the ranks of nonzero
differences as Pratt requires, and preserves the declared `[−1, 1]` range. Using `n(n+1)/2` as the
denominator is **rejected** — it would include zero ranks and shrink the attainable range whenever
zeros exist. `R₊ + R₋ = 0` → null + `undefined_no_nonzero_rank_sum`.

### D12 — BCa algorithm details — ✅ RESOLVED `[ch3]` for B/method, `[project]` for the internals
### ⚠️ SEED DERIVATION SUPERSEDED IN PART by **D21 (A3b-1, 2026-07-29)** — see below
B = **10,000** `[ch3]`, set explicitly (SciPy's own default is 9,999). Paired, **unstratified**
resampling over canonical image IDs `[ch3]` — ch3's only "stratified" mentions concern dataset
*sampling design*, not bootstrap strata. Frozen `[project]`: root seed **42**; `z0` from
`(count(boot < obs) + 0.5·count(boot == obs))/B` with probability clipped to `[1e-12, 1−1e-12]`;
leave-one-image-out jackknife acceleration; explicit **linear** quantile interpolation.

**Everything above remains active.** Only the per-statistic seed *string* is superseded.

> **Superseded seed derivation (pre-A3b-1), quoted verbatim for audit:**
> *"per-statistic seed = SHA-256 of `"plantseg-stats/1.0.0\0" + analysis_id + "\0" + comparison_id +
> "\0" + statistic_name`, first 16 digest bytes as an unsigned big-endian integer → `SeedSequence` →
> `Generator(PCG64)`, making results independent of comparison order and parallelism"*
>
> **Why superseded.** The string named root seed 42 but never consumed it, so `root_seed` was inert
> provenance and could not act as a protocol lever for a future replication study. **D21** renders it
> into the canonical bytes as `b"root_seed=" + str(root_seed).encode("ascii") + b"\x00"`.
>
> **Why the amendment was free.** At amendment time `src/stats/bootstrap.py`,
> `src/stats/noninferiority.py` and `src/stats/artifact.py` did not exist, and no `family.json` or
> `bootstrap.npz` had ever been written anywhere in the repository. **No implementation and no
> official statistical artifact ever consumed the old bytes**, so no published number changes.

**Active derivation:** see D21 and `STATISTICAL_ANALYSIS_CONTRACT.md` §8.1.

### D13 — BCa fallback — ✅ RESOLVED `[project]` within ch3's permission
ch3 permits "a percentile or basic" interval when acceleration is degenerate. **Exactly one**
fallback is frozen for v1.0.0: a **percentile** interval matching the requested confidence type,
labelled `percentile_fallback_degenerate_acceleration`. Basic intervals are not used. Non-finite
bootstrap estimates, or an official replicate with no eligible class, **abort** official analysis —
no silent replicate removal, no reduced B, no estimand switch.

> **Extended by D21 (A3b-1).** The label and the single-fallback rule are unchanged, but the
> *predicate* was unstated here. It is now the deterministic **first-trigger P1–P8 ladder** of
> `STATISTICAL_ANALYSIS_CONTRACT.md` §8.7.7 — evaluate in order, short-circuit, record **at most one**
> `fallback_p*` warning. D13 alone is **not** the complete fallback rule.

### D14 — non-inferiority interval type and boundary — ✅ RESOLVED `[ch3]` + `[project]` boundary
E3→E6 uses a **one-sided 95 % BCa lower bound** on the dataset-level all-class ΔmIoU, not a two-sided
interval; strict JSON records a dedicated `lower_bound` field with no fabricated upper bound.
**PASS iff lower bound > −0.020**; equality **fails** `[project]`. Sensitivity margins 1.0/1.5/2.0/2.5
pp are read from the **same** distribution. The **E6-KD** trigger fires when the *observed* clean
E3→E6 drop is **> 0.010**; equality does not trigger `[project]`. It carries no p-value and is not a
Holm test.

### D15 — dataset-level bootstrap estimand — ✅ RESOLVED `[ch3]`, restating D3c
Each replicate resamples image IDs with replacement, uses one shared index vector for both stages,
preserves duplicate multiplicity, **re-accumulates per-class (TP, GT, PRED)**, applies the frozen
union-present rule, and recomputes dataset-level all-class mIoU. Averaging per-image scalars is a
different estimand and is forbidden. Classes absent from both GT and prediction inside a replicate are
omitted by the union-present rule; a replicate with zero eligible classes is an integrity failure.

> **Extended by D21 (A3b-1).** The estimand is unchanged. Added in §9.1: **int64** accumulation for
> all TP/GT/PRED arithmetic (making "subtract the omitted image" and "re-sum the remainder" exactly
> equivalent), and the leave-one-image-out **re-accumulation** jackknife for this pooled quantity —
> deleting a per-image mIoU scalar is the same mean-of-ratios error one level down.

### D16 — official-input requirement — ✅ RESOLVED `[project]`
Every input evaluation artifact for an official statistical result must have
`artifact_status = "official"`. `provisional` and `smoke` artifacts may be used only for explicitly
non-official analyses and may never contribute to an official family, non-inferiority result, or
thesis conclusion. For official analysis a **`metric_impl_sha256` mismatch is an error, not a
warning**. No silent deletion, pairwise-complete fallback, or imputation of any kind.

### D17 — raw bootstrap distributions are persisted — ✅ RESOLVED `[project]`
`bootstrap.npz` stores the **float64 replicate distributions `[B]` and leave-one-out jackknife values
`[n]`** alongside the observed statistic, seed representation, `z0`, acceleration, requested and
actual interval method, confidence level and bounds. They are tiny beside model artifacts and spare
an auditor a ≈40-minute regeneration, allowing direct verification of `z0`, percentile positions and
every sensitivity margin without rerunning inference. Sampled index matrices `[B, n]` are **not**
stored — reproducible from the frozen seed and needlessly large. The result artifact is exactly
**three** files: `family.json`, `bootstrap.npz`, `MANIFEST.sha256`.

> **Extended by D21 (A3b-1).** What is stored is unchanged; *how* it is keyed was unstated. The exact
> NPZ key suffixes, dtypes and shapes, the `family.json` bootstrap-task record, the frozen
> interval/method/status/warning vocabularies, exact JSON↔NPZ equality and intra-run manifest
> semantics are frozen in `STATISTICAL_ANALYSIS_CONTRACT.md` §12.3.

### D18 — Hodges-Lehmann exact feasibility — ✅ RESOLVED `[empirical, measured]`
At n = 1,561 there are **1,219,141** Walsh averages (i ≤ j); one exact estimate takes **27.3 ms** via
partition-select (67.0 ms with a full sort), ~9.8 MB. BCa (B = 10,000 + 1,561 jackknife) costs
**≈ 5.3 min per comparison**, ≈ 42 min for eight. **No feasibility conflict** — the preregistered
exact HL BCa interval is affordable and is **not** replaced by a percentile, median, normal, or
absent interval.

### D19 — statistics software environment — ✅ RESOLVED `[project]`
Official p-values and CIs may be produced **only on the pinned stack** (Python 3.11, numpy 1.26.4,
scipy 1.11.4, statsmodels 0.14.6). The dev box currently runs 3.13.5 / 2.1.3 / 1.15.3 / 0.14.4 and is
permitted **only for contract smokes**; any mismatch forces non-official status. SciPy's RNG keyword
differs across versions (`random_state` vs `rng`), so the project uses **its own deterministic
bootstrap engine** consuming the D12 `Generator`; `scipy.stats.bootstrap` is a reference cross-check
only.

---

## Resolved 2026-07-28 (A3b-0) — canonical corruption vocabulary

### D20 — machine-readable corruption identifiers — ✅ RESOLVED `[project]`
**Gap discovered during A3a.** Chapter III, `IMPLEMENTATION_CONTRACT.md` §(f) and
`STATISTICAL_ANALYSIS_CONTRACT.md` §2 all named the five corruptions in **prose only**, while
`EVALUATION_CONTRACT.md` typed `condition.name` as an unconstrained `"string | null"`. Nothing
disambiguated `motion_blur` vs `motion-blur`, or `brightness` vs `brightness_variation`, so
`src/stats/robustness.py` correctly refused to invent slugs and required an injected grid.

**Resolution:** adopt the **vendored `imagecorruptions` reference function names** that ch3 already
selects, frozen machine-readably in **`configs/corruption_protocol.json`**
(`plantseg-corruptions/1.0.0`):

| machine identifier | display label | reference function |
|---|---|---|
| `motion_blur` | motion blur | `motion_blur` |
| `gaussian_noise` | Gaussian noise | `gaussian_noise` |
| `jpeg_compression` | JPEG compression | `jpeg_compression` |
| **`brightness`** | brightness variation | `brightness` |
| `fog` | fog | `fog` |

- **`brightness`, not `brightness_variation`** — the reference function is `brightness`;
  "brightness variation" is explanatory thesis prose. Recording the prose as an identifier would
  make artifacts un-joinable with the generator.
- **Aliases rejected outright**: `brightness_variation`, `motion-blur`, `Motion_Blur`, spaced or
  capitalised variants, filename-derived aliases, and any alias map. No official corruption
  artifact exists yet, so no migration compatibility is owed.
- Severity roles frozen alongside: **1–3 inferential**, **4 descriptive-only**, **5 excluded**.
- **Official mIoU-C is no longer blocked by vocabulary ambiguity.** Official analysis must load the
  protocol file; `NONOFFICIAL_SMOKE` may inject a synthetic grid that can never enter official
  statistics.

**Still open (deliberately not closed here):** the corruption **implementation** provenance — the
vendored bytes, its parameters, the `np.float_`→`np.float64` patch, the seed policy and the cache
protocol — remains a future corruption-scaffold task. That task must also **verify** the vendored
implementation actually exposes these five function names: `imagecorruptions` is currently neither
installed nor pinned in `requirements.lock`, so A3b-0 could not check it against real bytes.

---

## Resolved 2026-07-29 (A3b-1) — bootstrap reproducibility protocol

### D21 — bootstrap reproducibility: namespaces, seed bytes, draw order, fallback ladder, serialization — ✅ RESOLVED `[project]`, with **two recorded amendments**

**Gap discovered during A3b preflight.** §8.1 consumed `analysis_id`, `comparison_id` and
`statistic_name` as seed inputs, but **none of those vocabularies was frozen anywhere**; §8.5 scoped
"supplementary dataset-level Δ intervals" to nothing; the adjusted-probability transform, the
degeneracy predicate, the draw-call pattern, the accumulation dtype and every NPZ/JSON key were
unstated. Each could change published replicates or prevent exact reconstruction, and none was
verifiable after the fact. All are now frozen in `STATISTICAL_ANALYSIS_CONTRACT.md` §8.7 and §12.3.

**Two amendments, made before any implementation or official use.** No A3b module existed and no
statistics artifact had ever been written, so **no published number changes**. Both are recorded as
amendments, not clarifications.

- **Amendment A — seed bytes.** The superseded string is quoted verbatim in D12. Root seed 42 is now
  **rendered into** the canonical bytes as a parameter, giving `root_seed` real effect and leaving a
  future protocol version a lever for independent streams. Canonical bytes, in order:
  `b"plantseg-stats/1.0.0\x00"` + `b"root_seed=" + str(root_seed).encode("ascii") + b"\x00"` +
  `analysis_id` + NUL + `comparison_id` + NUL + `statistic_name` (**no trailing NUL**). SHA-256 → first
  16 digest bytes → one unsigned big-endian **128-bit** integer → sole entropy to `SeedSequence` →
  `PCG64` → one `Generator`. Never a second entropy word, never `SeedSequence(42)` spawning, never
  Python `hash`, never truncated to 32/64 bits, never stored as `int64`.
- **Amendment B — interval-type field name.** `confidence_type` → **`interval_type`**, pairing with
  `interval_method` and matching §12.2's existing "requested interval type · actual interval method".
  Superseded §8.5 wording, verbatim: *"Recorded as `confidence_type = "one_sided_lower"`,
  `confidence_level = 0.95`, a finite `lower_bound` field"*. The **value** `one_sided_lower`,
  `confidence_level = 0.95` and the dedicated `lower_bound` key are **unchanged**. `confidence_type`
  occurred in exactly one place repo-wide and in **no** decision record, so **no** open-questions entry
  required an in-place edit — D14 was inspected and left unchanged because it never names the field.

**Namespaces.** Analysis IDs `plantseg_primary_analysis_v1` (official) and `plantseg_stats_smoke_v1`
(smoke) — a stream namespace, never a run UUID/timestamp/path/commit/checkpoint. Two non-family
comparison IDs `descriptive_e1_e3` and `noninferiority_e3_e6`, neither added to
`CANONICAL_COMPARISON_IDS`. Four statistic names `mean_delta`, `median_delta`,
`hodges_lehmann_shift`, `dataset_miou_delta`; aliases rejected. Renaming any of these changes the seed
namespace and requires a new protocol version.

**35-task matrix.** Seven clean superiority comparisons × 4 statistics = 28 · `robustness_e1_e6` × 3
scalar statistics on the per-image mIoU-C vector = 3 · `descriptive_e1_e3` × 3 = 3 ·
`noninferiority_e3_e6` × `dataset_miou_delta` = 1. **Exactly one** task is `one_sided_lower`; the other
**34** are `two_sided`. **Interval type is a property of the task pair, never of `statistic_name`** —
`dataset_miou_delta` is two-sided on the clean comparisons and one-sided on non-inferiority, so keying
the tail off the statistic name alone silently produces a wrong-tail NI bound. Canonical record order
is frozen for serialization only and changes no number.

**Draw order and streams.** One **row-wise** `generator.integers(low=0, high=n_images, size=n_images,
dtype=np.int64, endpoint=False)` per replicate; the bulk `(B, n_images)` call is forbidden — not
because the two must differ, but because the API call sequence is protocol and must not depend on
undocumented cross-version equivalence. **35 independent streams**, one per
`(analysis_id, comparison_id, statistic_name)`; statistics on the same comparison do **not** share
draws. Smaller `B` is a prefix of larger `B` only when namespace, `n_images`, NumPy version and
BitGenerator all match.

**Fallback is a deterministic first-trigger ladder.** P1…P8 are evaluated **in order and
short-circuit**: at the first true predicate, stop, use `percentile_fallback_degenerate_acceleration`,
record **exactly one** `fallback_p*` warning, and do not compute values whose prerequisites already
failed. This replaces any "multiple predicates may be recorded" reading, which was **not
deterministic** — `0/0` acceleration makes P3, P4 and P5 simultaneously true and the outcome would
otherwise depend on guard placement. Worked case: an all-zero paired vector gives `p0 = 0.5`,
`z0 = 0`, a centered-jackknife squared sum of exactly zero → **P3 only**, percentile fallback, bounds
`[0.0, 0.0]`, `ok_with_warnings`; P4 and P5 are never evaluated. Saturation (`Phi` returning exactly
0.0/1.0) is **not** a fallback — it is a completed BCa result recorded as a warning.
`acceleration_defined == false` implies fallback; **the converse is false**, since P6–P8 fire with
finite acceleration.

**Accumulation.** All TP/GT/PRED accumulation, resampled multiplicity and leave-one-out subtraction
are **int64**; only the final division and macro mean are float64, making "subtract" and "re-sum"
exactly equivalent. Matches `src/stats/ingest.py`, which already enforces int64 and returns
non-writeable arrays — so any dense working copy must be **owned**.

**Serialization.** `task_id = "<comparison_id>__<statistic_name>"` — exactly **two** components,
unambiguous because neither vocabulary permits `__`. `analysis_id` is stored once at artifact level and
verified per task by **exact seed-byte reconstruction**, never by substring containment. Verifier
checks 1–5 (digest, NPZ digest, entropy hex, NPZ entropy bytes, decimal) are recomputable from the
artifact; the sixth — that this integer reached `SeedSequence` — is a **writer-side invariant**, since
confirming it post hoc would require regenerating the draws that persistence exists to avoid.
`seed_input_hex` is the sole byte authority; `seed_input_display` has a frozen literal-`\0` spelling
and is non-authoritative. Frozen vocabularies for `interval_type`, `interval_method`, task `status`
(`ok` / `ok_with_warnings`, derived from `warnings` by asserted biconditional) and the twelve
`warnings` strings in canonical order. JSON↔NPZ equality is **exact** float64 after reload — a
tolerance-based verifier is forbidden because it would mask the divergence it exists to detect.
Official artifacts require `bootstrap_replicates == 10000` and `jackknife_count == 1561` for **all 35**
tasks, robustness and descriptive included. `MANIFEST.sha256` is **intra-run**: `np.savez` embeds
archive metadata, so cross-run byte equality must not be asserted.

**Task status is not A3a status.** An A3a comparison may be `degenerate_all_zero` while its A3b tasks
are validly `ok_with_warnings`. The A3a status is never copied into the task record, and task `ok`
does not imply the comparison was non-degenerate, significant or officially eligible.

### D22 — pooled dataset-level **robustness** estimand — ⏸️ DEFERRED `[project]`, not a limitation

A pooled dataset-level mIoU-C statistic (`dataset_miou_c_delta`) was proposed during A3b-1 planning
and is **deliberately not adopted** in protocol 1.0.0. Chapter III defines the inferential robustness
comparison as **paired per-image mIoU-C for E1 vs E6** and defines no pooled counterpart: not its
purpose, not its class space, not its sufficient-statistic aggregation, not its clean-image alignment.
`STATISTICAL_ANALYSIS_CONTRACT.md` §9.1 defines only the **clean all-class** pooled estimand, and §2's
sole mention of "dataset-level mIoU-C summaries" is a **prohibition** on building the per-image vector
from them — not an authorization.

This is a **methodology-scope decision**, not an implementation shortfall: protocol 1.0.0 implements
the analysis actually governed by Chapter III and the reconciled contracts. No statistic name, NPZ
key, `task_id` or `family.json` record containing `dataset_miou_c_delta` is valid under 1.0.0.

A future protocol version **could** adopt one, but only after explicitly freezing all of: its
methodological purpose · **all-class versus disease-only** class space (both dataset-level forms exist
in `EVALUATION_CONTRACT.md`, and the per-image mIoU-C it would sit beside is **disease-only**, so the
choice is not automatic) · its fifteen-cell aggregation · canonical `clean_image_id` alignment across
all thirty condition artifacts · its bootstrap and jackknife estimand · and its relationship to the
existing per-image mIoU-C inference. **Nothing here implies that future adoption is required.**

---

## Resolved 2026-07-30 (A3b-2) — observed-value provenance and the top-level result schema

### D23 — observed-value provenance (**Amendment C**) — ✅ RESOLVED `[project]`

**Contradiction found during A3b implementation planning.** The A3b-1 rule required the observed
scalar of every `mean_delta` / `median_delta` / `hodges_lehmann_shift` task to be *"taken from the
typed A3a result"*, and counted **27 inherited + 8 computed** across the 35 tasks. That count presumes
nine comparisons can yield a typed `ComparisonResult`. Only **eight** can:
`src/stats/tests.run_comparison` raises `StatsError` for any id outside `CANONICAL_COMPARISON_IDS`, and
`STATISTICAL_ANALYSIS_CONTRACT.md` §8.7.1 forbids adding `descriptive_e1_e3` to that tuple. The three
`descriptive_e1_e3` scalar tasks were therefore **unsatisfiable** — they demanded an object that
cannot lawfully exist.

> **Superseded wording (pre-A3b-2), quoted verbatim for audit:**
> *"**Observed-value provenance.** For `mean_delta`, `median_delta` and `hodges_lehmann_shift` the
> observed statistic is **taken from** the typed A3a result and must match it **bit-for-bit** in
> float64; A3b recomputes only replicate and jackknife values. For `dataset_miou_delta` A3b computes
> the observed pooled value, because A3a produces no dataset-level pooled statistic. Across the 35
> tasks that is 27 inherited and 8 computed."*

**Corrected mapping — 24 / 3 / 8 = 35** (`STATISTICAL_ANALYSIS_CONTRACT.md` §12.3.3):

| `observed_source` | tasks | count |
|---|---|---:|
| `a3a_comparison_result` | eight canonical comparison IDs × three scalar statistics | 24 |
| `a3a_primitives_descriptive` | the three `descriptive_e1_e3` scalar tasks | 3 |
| `a3b_pooled_reaccumulation` | seven clean + one non-inferiority `dataset_miou_delta` | 8 |

**Direct descriptive primitive path.** The three descriptive values are derived from the aligned
E1/E3 clean `PairedVector` by calling the **existing public A3a estimators** `engineering_shifts` and
`hodges_lehmann`, direction `E3 − E1`, and packaged in an immutable typed A3b-side carrier. That
carrier is an operational vehicle only and confers no inferential status. Prohibited: adding
`descriptive_e1_e3` to `CANONICAL_COMPARISON_IDS` · calling `run_comparison` for E1→E3 · constructing a
fake or surrogate `ComparisonResult` · computing Wilcoxon, the paired t-test, rank-biserial, Cohen's dz
or any Holm member for E1→E3 · serializing inferential placeholders, including nulls, in the
descriptive block.

**Canonical-family membership is unchanged.** The eight-member Holm family of §1 and §0.1's ch3
reconciliation are untouched; E1→E3 remains descriptive-only and outside the family, exactly as
Table 3.6 requires. Amendment C changes only *where an observed number is read from*, not which
comparisons exist or what is tested.

**`observed_source` vocabulary** — exactly `a3a_comparison_result`, `a3a_primitives_descriptive`,
`a3b_pooled_reaccumulation`; required on every bootstrap-task record; rejected if missing, null,
unknown, aliased, or attached to the wrong task; **never inferred from `statistic_name` alone**, since
`mean_delta` maps to two different sources depending on the comparison. It is `family.json`-only and
never an NPZ key.

**`mean_delta_pp`** is descriptive reporting supplied by `EngineeringShifts`. It is not a bootstrap
statistic, not a 36th task, not an NPZ namespace and not a bootstrap-task record.

**Required bit-identity smoke.** A3b must build a paired vector under a *canonical* comparison ID,
compute its `ComparisonResult` via `run_comparison`, independently call `engineering_shifts` and
`hodges_lehmann` on the same delta vector, and assert bit-for-bit float64 equality for **all three**
statistics — proving Source B uses the frozen A3a arithmetic and not a shadow estimator.

**Amendment discipline.** This is an **amendment**, not a clarification. It was made **before any A3b
implementation and before any official statistical artifact existed** — `src/stats/bootstrap.py`,
`noninferiority.py` and `artifact.py` did not exist and no `family.json` had ever been written — so no
published number changes. Cross-reference: D21 (A3b-1 amendments A and B), D24 (schema).

### D24 — top-level `family.json` schema — ✅ RESOLVED `[project]` + `[implementation]`, **newly frozen, not an amendment**

**Gap found during the same planning pass.** `STATISTICAL_ANALYSIS_CONTRACT.md` §12.1 named the
required top-level content in prose but froze field spellings **only** for the per-input-artifact
provenance record. The artifact-level status field and its vocabulary, the family-completion
semantics, the integrity-results structure, the artifact-level warning enum, the block and field
spellings for the descriptive / non-inferiority / E6-KD blocks, the timestamp format, the
analysis/family identity fields, the A3a serialization order and the relative-retention formula were
all **previously unfrozen implementation details**. An implementer would have had to invent them —
precisely the failure mode A3b-1 existed to prevent. They are now frozen in
`STATISTICAL_ANALYSIS_CONTRACT.md` **§12.4**.

Frozen: the **nineteen** top-level keys and their exact order · `family_id = "plantseg_eight_test_holm_v1"` ·
the `run_id` pattern `^[a-z0-9][a-z0-9._-]{0,63}$`, equal to the run-directory basename and with no RNG
effect · `created_at_utc` as `%Y-%m-%dT%H:%M:%SZ`, matching the proven A2a writer · the **level-scoped**
`artifact_status` vocabularies (top-level `official`/`nonofficial`; input records
`official`/`provisional`/`smoke`; ingest `Policy` `official`/`nonofficial_smoke`) which never merge
despite the shared spelling · the **exhaustive** six-string officiality-warning vocabulary with the
no-third-path rule and the fatal-error boundary · `alpha = 0.05` and `expected_comparison_ids` imported
from `CANONICAL_COMPARISON_IDS` rather than retyped · the `statistical_contract_sha256` field rules
(exact-byte hashing, 64 lowercase hex, runtime verification, fatal mismatch) **with no active expected
hash literal in the contract itself**, since such a literal would be self-invalidating on every edit ·
the `software_environment` shape with `matches_pinned` derived and asserted · the **37-record** official
input set (7 clean E1–E7 + 15 E1 corruption + 15 E6 corruption), sorted **only** by ascending
lexicographic `repo_relative_path` · the **A+** `a3a_family` envelope · the eight literal A3a/Holm
dataclass field tuples plus the serializer rule and runtime `dataclasses.fields()` assertion · the
descriptive, non-inferiority and E6-KD block schemas · the `relative_retention` formula · the integrity
block with thirteen mandatory checks · and the canonical ordering and strict-validation rules.

**Relative retention `[project-decision]`.** Chapter III requires reporting the actual E3→E6 drop, the
relative retention of E3 mIoU, and the sensitivity conclusions, but froze **no formula** — a
case-insensitive search of the governing documents found no definition. Frozen now over the same clean
dataset-level all-class union-present pooled mIoU values already required by the §9.1 estimand:
`relative_retention = candidate_dataset_miou / baseline_dataset_miou`, i.e.
`dataset_mIoU(E6) / dataset_mIoU(E3)`. A dimensionless float64 ratio, never ×100 in `family.json`;
values above 1 are valid; the denominator must be finite and strictly positive, and a zero or negative
denominator is **fatal in every policy mode**. It is **descriptive only** and affects no interval,
decision or bootstrap stream. The non-inferiority block grew from thirteen to **fifteen** fields to
carry `baseline_dataset_miou` and `candidate_dataset_miou` explicitly, so all four identities are
verifiable from the record itself. The frozen synthetic vector is **binary-exact** — baseline `0.5`,
candidate `0.25`, delta `-0.25`, drop `0.25`, retention `0.5` — because with baseline `0.5` and
candidate `0.4` float64 yields `-0.09999999999999998`, so asserting the literal `-0.10` would wrongly
fail.

**A+ Holm structure.** `a3a_family` has exactly three keys — `completion_status`, `comparisons`,
`holm` — and `holm` serializes the **complete finalized `HolmFamily`** dataclass in its own declaration
order `("members", "alpha", "complete", "status")`. `complete` is retained rather than folded into
`completion_status`, and `alpha` rather than deferred to the top-level key. The duplication is
deliberate: the values sit at different schema levels with independent audit roles, and their required
agreement is what lets a verifier detect a writer that combined incompatible objects or serialized a
stale family. Fifteen cross-field invariants are frozen, all by **exact** float64 equality with
`isclose` forbidden.

**Finalized `HolmFamily` status is exactly `"ok"`** `[implementation]`. `src/stats/tests.holm_family`
has a single return, constructing the family with `complete=True` and `status=STATUS_OK` (`"ok"`);
every other path raises `StatsError`. The dataclass constructor is public, so a caller *could*
hand-build another status or `complete=False` — such an object is transient and unsanctioned and is
**not legal** in a finalized `family.json`. The contract records that transient-versus-finalized
distinction explicitly.

**Runtime source-schema assertions.** For each of the eight frozen dataclasses the writer asserts
`tuple(f.name for f in dataclasses.fields(cls))` equals the contract tuple exactly; a missing, added,
renamed or **reordered** source field fails closed. `dataclasses.asdict()` is never the schema
authority. After this task **the contract, not the source, is the protocol authority**, so a future
source reorder cannot silently change protocol 1.0.0. Three specific serializer rules are frozen:
`Policy` is emitted through `.value`; `p_adjusted` must be finite and an intermediate NaN member is
fatal; `boundary_note = null` is legitimate and means no boundary condition arose.

**Integrity.** No new check key was added for A+. `a3a_family_complete` is defined to require the
entire A+ verification set, and any failure leaves it unestablished and prevents final-directory
creation — never weakened into a top-level officiality warning.

**Classification.** D24 **replaces no active rule** and is therefore **not an amendment**, with one
narrow exception: it adds a level-scoped top-level value vocabulary to the existing `artifact_status`
field name (§12.4.3). No A3b implementation and no statistics artifact existed when these details were
frozen. Cross-references: D21, D23. **D22 (deferred pooled robustness estimand) is unchanged and is
not reopened.**

---

## Open — pending a governed-path session

### D29 — the no-resume rule's contract-level home is not yet settled `[project; B44]`
The rule ("official runs of any stage are never launched or continued with `--resume`", `AGENTS.md`
§ E1 invariants) currently lives **operationally** in `AGENTS.md` and in the preemption playbook at
[reports/e1_launch_runbook_v2.md](../reports/e1_launch_runbook_v2.md) §5. Its durable methodological
home is [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md), a **governed path**
(`AGENTS.md` rule 8), which was out of scope for the B44 session.

- **How it gets resolved:** a session explicitly approved to edit governed paths adds it to the
  implementation contract, after which `AGENTS.md` can cite the contract rather than carrying the
  statement itself.
- **Also pending:** ch3 will need to reflect it. The manuscript reproducibility follow-up at
  `docs/open_questions.md:76-78` pins seeds and cuDNN flags but does not address interruption
  handling at all.

---

### D30 — the INFERRED 11–14 GB VRAM band is falsified for every student stage under determinism `[project; B48]`
[IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md)`:249-252` records an **INFERRED 11–14 GB**
peak on CUDA at batch 16 for E2/E3 and says of it: "This is **not a GO** — settle it on the pod with
`torch.cuda.max_memory_allocated()`." [B48](../reports/b48_e1_oom_investigation.md) §2 did exactly
that, for E1.

**Measured** (RTX 4090, batch 16, 512², `num_classes=116`): **20.667 GiB** with
`use_deterministic_algorithms(True)` as [src/seeds.py](../src/seeds.py)`:37` sets it, **17.014 GiB**
without. The band's *component* estimates are not the problem — its omitted categories are. It
accounts for allocator overhead, the teacher transient, cuDNN workspace and backward temporaries,
but not for the **12.658 GiB forward transient** determinism introduces by rerouting
`F.interpolate` to `torch._decomp.decompositions.upsample_bilinear2d_vec`. That one omitted term is
larger than the entire inferred band.

**This generalizes past E1.** The transient scales with `num_classes × H × W × batch` at the final
upsample, none of which is stage-specific: E1, E2, E3, E5 and E6 all carry the same 116-class 512×512
head at batch 16. **Every student stage needs a 48 GB card**, not E1 alone.
[b32c_closeout.md](../reports/b32c_closeout.md)`:17` — that a 24 GB → 48 GB switch "looks to have
been wrong even before B32" — is superseded on measurement.

- **How it gets resolved:** a session explicitly approved to edit governed paths corrects
  `IMPLEMENTATION_CONTRACT.md:249-252` (the INFERRED band → the measured figure with its determinism
  condition) and `:393`, whose Compute row still reads "RunPod RTX 4090 (~$0.34/hr)" — a card now
  measured as unable to reach iteration 2 of E1.
- **Not in scope of that fix:** whether E2/E3 add materially *on top of* E1's 20.667 GiB. B32c's
  teacher-transient reasoning is untouched by this and stays INFERRED until measured on the pod.

---

### D31 — `write_report`'s generation date is hardcoded `[project; B47]`
[scripts/smoke_loss.py](../scripts/smoke_loss.py)`:36` emits `_Generated: 2026-06-27 by
scripts/smoke_loss.py …_` as a string literal, so every `reports/loss_smoke.md` it writes claims that
date regardless of when it ran. B47 added `--no-report` to stop the gate dirtying the worktree and
deliberately left this alone, as it fell outside that session's narrow governed-path scope.

- **How it gets resolved:** a session explicitly approved to edit governed paths replaces the literal
  with the run's own date. `scripts/` is governed (`AGENTS.md` rule 8).
- **Severity:** cosmetic, but misleading in the wrong direction — the file is a provenance artifact,
  and a wrong date in a provenance artifact reads as authoritative later.

---

## NEED_TO_CONFIRM (not stated in any source; filled by selection/measurement, never guessed)

### 3. `λ_logit` (Logit-KD weight)
- **Resolution:** pre-registered validation grid sweep over **{0.25, 0.5, 1, 2, 4}** at seed 42; select
  the candidate maximizing dataset-level **validation** mIoU (ties → smaller weight; boundary value
  reported as such). Selected value reported in Ch4 and **reused unchanged in E3**. `[ch3 §C "E2"]`
- **When:** during E2 (training phase — out of Week-1 scope).

### 4. E6-KD reduced distillation weights
- **Resolution:** only relevant **if** the contingency triggers (E3→E6 clean mIoU drop > 1.0 pp); weights
  set **below** their E3 values so soft targets do not override INT8 adaptation. Exact reduced values
  `NEED_TO_CONFIRM`. `[ch3 §C "E6"]`
- **When:** conditional, post-E6.

### 5. Library versions without pinned numbers
- **Albumentations:** "version pinned in the reproducibility manifest" but no number given → confirm from
  the actual `requirements.lock` / environment manifest once present. `[ch3 §D; ctx]`
- **statsmodels:** no version stated → confirm from the manifest. `[ch3 §D; ctx]`
- **Resolution:** read the pinned `requirements.lock` / Dockerfile when committed to the repo.

### 6. Additional seed values (multi-seed runs)
- **Resolution:** three-seed validation is planned for **E1 and E3**; the two seeds beyond **42** are not
  specified ("all seeds used are reported in the appendix") → confirm at run time. `[ch3 §F]`
- **When:** if/when multi-seed runs execute (compute-permitting; out of Week-1 scope).

### 7. RunPod hardware specifics
- **Resolution:** final pod type, GPU model, VRAM, CPU model + thread count, CUDA/container image, storage,
  and per-stage GPU-hours/cost are "reported in Chapter 4". `[ctx]` names RunPod **RTX 4090 (~$0.34/hr)** as
  the working assumption; treat exact values as `NEED_TO_CONFIRM` until logged. `[ch3 §D; ctx]`

### 8. Citation details (DOIs / venues / years) in reference.pdf
- **Resolution:** `[ctx]` warns two divergent citation lists exist historically — **never fabricate** a
  DOI/venue/author-year. Verify each against the primary PDF before use. Confirmed-good anchors:
  PlantSeg DOI **10.5281/zenodo.17719108**, Wei et al. *Scientific Data* (2026) 13:205,
  https://doi.org/10.1038/s41597-025-06513-4 `[Wei]`.

### 9. All experimental result numbers (teacher recovered mIoU; E1–E7 accuracy/efficiency/robustness)
- **Resolution:** produced by training/evaluation runs — **out of Week-1 scope** (analysis/setup/docs/smoke
  tests only). Teacher success criterion is recovery of 42.05% within ±1.5–2.0 pp. `[ch3]`

---

## Pre-training verification gates (mandatory before ANY E1 training) `[ch3 Table 3.1; ctx]`

These are not "open questions" so much as **blocking checks**; logged here for completeness. Store each
result as a JSON artifact next to the run.

1. **Mask class count** — `np.unique()` over all annotation PNGs (settles #1/#2 above).
2. **Image↔mask pairing** — shared filename/ID, both readable, identical dimensions; zero unmatched/
   unreadable/dimension-mismatched pairs.
3. **Split integrity** — zero identifier overlap across train/val/test (overlap count = 0).
4. **Mask value range** — every mask pixel ∈ (verified class set) ∪ {255}; no out-of-range values.
5. **Ignore-label unit tests** — CE / Dice / Logit-KD / CWD losses unchanged when padded (255) pixel
   values are altered; CWD channel-wise spatial softmax sums to 1 over valid locations only; teacher
   forward hook returns the stride-16 MSCAN-B Stage-3 feature (**320 channels**) for a synthetic input.
6. **QNNPACK operator support** — confirm **INT8 Sigmoid** in the LR-ASPP global-pool branch is supported
   (gate before E4); any unsupported op → FP32 fallback, reported.
