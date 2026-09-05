# PRE-REGISTRATION — Analysis commitments for E1–E7

**Anchor commit:** `569cfbb` · **Branch:** `master` · **Date written:** 2026-09-05
**Status at this commit: NO E1–E7 RESULT EXISTS.** Evidence in §11.

---

## 0. What this document is

A **consolidation** of analysis commitments already made across `docs/reference/ch3.pdf`,
`docs/IMPLEMENTATION_CONTRACT.md`, `docs/EVALUATION_CONTRACT.md`,
`docs/STATISTICAL_ANALYSIS_CONTRACT.md`, and `configs/`. Every item below is quoted from a source and
cited to `file:line` or `ch3 p.N`.

**It creates no commitment.** Nothing here is invented, altered, loosened, or resolved. Where a value
is unresolved it is recorded as unresolved (§10). Where sources disagree, **both readings are recorded
and the disagreement is flagged — none is adjudicated** (§12). Fields the sources do not state are
left explicitly empty rather than filled.

Its evidentiary value rests entirely on the ordering: this file is committed **before** any E1–E7
number exists. See §13.

---

## 1. Hypotheses, verbatim, with test direction

`ch3` pp. 3–6. Quoted as written; emphasis not added.

**H₁ (umbrella).**
> "The proposed pipeline utilizing Logit Knowledge Distillation, Channel-Wise Knowledge Distillation,
> and INT8 Quantization-Aware Training improves segmentation accuracy and robustness of the
> MobileNetV3-based student on plant lesion segmentation relative to the baseline FP32 student (E1)
> and the alternative ablation configurations (E2–E5, E7) and yields the most favorable
> accuracy–efficiency–robustness trade-off among the configurations evaluated in this study (E1–E7),
> with the SegNeXt-B teacher as a descriptive upper-bound reference only." — ch3 p. 3

Wei et al. (2026) published results are explicitly **not inferential comparators**: "per-image scores
are not available for paired testing, and unmatched preprocessing, training procedure, and evaluation
conditions would in any case invalidate any statistical comparison." (ch3 p. 3)

**H₁a — ch3 p. 3.**
> "Student trained with Logit Knowledge Distillation (E2) provides higher per-image mIoU than the FP32
> student baseline (E1), measured by per-image mIoU (the unit of paired inferential testing, Section
> F.1.b). Dice Coefficient is reported descriptively at the dataset level only and is not separately
> tested. This hypothesis will be tested using a one-tailed Wilcoxon signed-rank test on per-image
> ΔmIoU = mIoU(E2) − mIoU(E1), where ΔmIoU > 0 indicates improvement."

**Direction:** one-tailed, `ΔmIoU = mIoU(E2) − mIoU(E1) > 0`.

**H₁b — ch3 pp. 3–4.**
> "Student trained with Logit Knowledge Distillation and Channel-Wise Knowledge Distillation (added in
> E3) yields higher per-image mIoU than the Logit KD-only student (E2). … This hypothesis will be
> tested using a one-tailed Wilcoxon signed-rank test on per-image ΔmIoU = mIoU(E3) − mIoU(E2), where
> ΔmIoU > 0 indicates improvement."

**Direction:** one-tailed, `ΔmIoU = mIoU(E3) − mIoU(E2) > 0`.

**H₁c — ch3 p. 4.**
> "INT8 QAT yields higher per-image mIoU than INT8 Post-Training Quantization (PTQ) when each
> quantization technique originates from the same source checkpoint, tested by two paired comparisons:
> E4 vs E5 (undistilled) and E7 vs E6 (distilled), both contributing to the family-wise
> Holm-Bonferroni correction over all 8 inferential tests. … Improvement is indicated when ΔmIoU > 0
> for the following comparisons: (i) ΔmIoU = mIoU(E5) − mIoU(E4); and (ii) ΔmIoU = mIoU(E6) −
> mIoU(E7)."

**Direction:** two one-tailed tests, `mIoU(E5) − mIoU(E4) > 0` and `mIoU(E6) − mIoU(E7) > 0`.
Efficiency differences across E4–E7 are "reported as supplementary observations and are not subject to
inferential testing."

**H₁d — ch3 pp. 5–6.**
> "The full proposed pipeline, E6 = Logit KD + CWD + INT8 QAT, yields the most favorable accuracy,
> efficiency, and robustness trade-off among all student configurations, E1 to E5 and E7.
> Operationally, this is supported when all four of the following hold: (i) accuracy retention, where
> the E3 → E6 drop in clean-test mIoU is small in absolute terms … and assessed formally through the
> E3 vs E6 dataset-level mIoU non-inferiority check; (ii) accuracy improvement over baseline, where
> per-image mIoU(E6) is higher than per-image mIoU(E1), tested through the E1 vs. E6 comparison; (iii)
> robustness improvement over baseline, where per-image mIoU-C(E6) is higher than per-image mIoU-C(E1),
> tested through the E1 vs. E6 comparison on mIoU-C; and (iv) efficiency, where the model size and CPU
> proxy latency of E6 are descriptively lower than all FP32 student variants, E1, E2, and E3."

**Direction:** conjunctive — all four conditions. (i) is the §5 non-inferiority check; (ii) and (iii)
are one-tailed family members; (iv) is **descriptive, not tested**. "The SegNeXt-B teacher serves only
as a descriptive upper-bound reference and is not subject to any inferential comparison."

---

## 2. The eight-test family, named pairwise

`IMPLEMENTATION_CONTRACT.md:495-499` — "**The 8 comparison IDs** (candidate − baseline, positive
favours the candidate)":

| # | comparison ID | candidate − baseline | metric |
|---|---|---|---|
| 1 | `accuracy_e1_e2` | E2 − E1 | clean per-image disease-only mIoU |
| 2 | `accuracy_e2_e3` | E3 − E2 | clean per-image disease-only mIoU |
| 3 | `accuracy_e4_e5` | E5 − E4 | clean per-image disease-only mIoU |
| 4 | `accuracy_e7_e6` | E6 − E7 | clean per-image disease-only mIoU |
| 5 | `accuracy_e4_e7` | E7 − E4 | clean per-image disease-only mIoU |
| 6 | `accuracy_e5_e6` | E6 − E5 | clean per-image disease-only mIoU |
| 7 | `accuracy_e1_e6` | E6 − E1 | clean per-image disease-only mIoU |
| 8 | `robustness_e1_e6` | E6 − E1 | per-image **mIoU-C** |

Seven clean + one robustness. Implemented as `CANONICAL_COMPARISON_IDS` at
[`src/stats/tests.py:32-35`](../src/stats/tests.py#L32) in this exact order; an unknown
`comparison_id` raises ([`:263-264`](../src/stats/tests.py#L263)).

**Excluded from the family, reported but not Holm-corrected**
(`IMPLEMENTATION_CONTRACT.md:508-512`): **E1→E3** (descriptive total-FP32-distillation gain: mean
ΔmIoU and Hodges-Lehmann shift with BCa 95 % CIs, "no family p-value, no superiority claim");
**E3→E6** (non-inferiority, §5); teacher comparisons, Dice, mAcc, aAcc, RPD, rCD and efficiency.

> See §12, disagreement 1 — ch3 is internally inconsistent about this membership, and the
> discrepancy changes every adjusted p-value.

---

## 3. Primary test, sensitivity test, correction, α

**Primary — one-tailed Wilcoxon signed-rank.**

`IMPLEMENTATION_CONTRACT.md:487-489`:
```
scipy.stats.wilcoxon(d, zero_method='pratt', correction=True, alternative='greater',
                     method='approx'),  d = candidate − baseline
```

ch3 p. 59 states the call as:
> "The Wilcoxon signed-rank test is computed as scipy.stats.wilcoxon(d, zero_method='pratt',
> alternative='greater')."

> See §12, disagreement 3.

**Sensitivity — paired t-test.** `IMPLEMENTATION_CONTRACT.md:490-491`:
`scipy.stats.ttest_rel(..., alternative='greater')`, "(CLT at n = **1,561**). Sensitivity p-values
**never** enter the Holm family." ch3 p. 55: "the paired t-test using scipy.stats.ttest_rel is
computed and reported in each case as a" sensitivity check; ch3 p. 61: "If conclusions differ between
tests, Wilcoxon will be" the deciding test.

**Correction.** `IMPLEMENTATION_CONTRACT.md:492-493`:
`statsmodels.stats.multitest.multipletests(method='holm')`, "strict boundary (equality does not
reject)". Call form at `STATISTICAL_ANALYSIS_CONTRACT.md:238`:
`multipletests(pvals, alpha=0.05, method="holm", ...)`.

**α = 0.05**, family-wise. ch3 p. 2 ("α = 0.05 via Holm-Bonferroni correction"), p. 60 ("performed
one-tailed, all at the family-wise significance level α = 0.05"), p. 63.
`STATISTICAL_ANALYSIS_CONTRACT.md:1048` — `alpha` is "exactly `0.05`"; frozen finalized value
`holm.alpha = 0.05` (`:1206`).

**Zero handling.** Pratt: ch3 p. 60 — "Under Pratt's method (Pratt, 1959), zero differences are
assigned ranks before being excluded from R⁺ and R⁻, preserving their contribution to the variance
rather than discarding them."

---

## 4. Effect sizes reported with each test

`STATISTICAL_ANALYSIS_CONTRACT.md` §7.1–§7.4 (`:263-295`). Only what the contract defines is listed;
no conventional additions.

| effect size | definition | source |
|---|---|---|
| **Matched-pairs rank-biserial** | rank all absolute differences **including zeros**; average ranks for ties; zero-difference ranks enter neither signed sum; `r_rb = (R₊ − R₋)/(R₊ + R₋)`; range `[−1, 1]`, positive favours candidate. `R₊ + R₋ == 0` → `null`, status `undefined_no_nonzero_rank_sum` | §7.1 (`:263-280`) `[project-decision, Pratt-compatible]` |
| **Cohen's dz** | `mean(d) / std(d, ddof=1)`; positive favours candidate. Benchmarks 0.20/0.50/0.80 "interpretive only". Zero sample SD → `null` + status, "**never Infinity**" | §7.2 (`:281-285`) `[ch3]` |
| **Hodges-Lehmann paired shift** | median of all Walsh averages `(dᵢ+dⱼ)/2`, `i ≤ j` (`n(n+1)/2` terms); exact, "**No approximation is required at n = 1,561**" | §7.3 (`:286-290`) `[ch3]` |
| **Engineering shifts** | mean Δ in metric units · mean Δ in pp = `100 × mean_delta` · median Δ (descriptive). "No Chapter-IV display rounding in the artifact" | §7.4 (`:291-295`) `[ch3]` |

Degenerate-case values are pre-registered (`:194-196`): mean Δ / median Δ / Hodges-Lehmann → `0.0`;
rank-biserial → `null` with status `undefined_no_nonzero_rank_sum`; Cohen's dz → `null` with zero-SD
status. "**no fabricated**" values (`:204`).

---

## 5. E3-vs-E6 non-inferiority — outside the eight-test family

`IMPLEMENTATION_CONTRACT.md:515-519`:

> "E6 non-inferior to E3 iff the **one-sided 95 % BCa lower bound** on `ΔmIoU = mIoU(E6) − mIoU(E3)`
> (dataset-level, all-class, union-present) is **strictly greater than −2.0 pp**; equality fails.
> **B = 10,000**. Sensitivity decisions at 1.0/1.5/2.0/2.5 pp are read from the **same** bootstrap
> distribution."

ch3 p. 5:
> "E6 is considered non-inferior to E3 if the lower bound of the paired BCa 95% confidence interval
> (B = 10,000) for mIoU(E6) − mIoU(E3) exceeds −2.0 percentage points. This check is reported
> separately from the eight-test Holm-Bonferroni superiority family."

| element | value |
|---|---|
| margin | **−2.0 pp**, strict (equality fails) |
| interval | one-sided 95 % **BCa** lower bound |
| B | **10,000** |
| estimand | dataset-level, **all-class**, union-present mIoU |
| family membership | **OUTSIDE** the eight-test Holm family — stated in ch3 §A/§F three times, and at `IMPLEMENTATION_CONTRACT.md:510` |
| sensitivity margins | 1.0 / 1.5 / 2.0 / 2.5 pp, from the **same** bootstrap distribution |

Nominal one-sided lower probability `0.05` (`STATISTICAL_ANALYSIS_CONTRACT.md:389`).

---

## 6. The E6-KD contingency — pre-registered, not reactive

`IMPLEMENTATION_CONTRACT.md:520-521`:
> "**E6-KD contingency trigger** (stricter, distinct): run if the **observed** clean E3→E6 mIoU drop
> is **> 1.0 pp**; equality does not trigger. No p-value, not a Holm test."

ch3 p. 57:
> "As a separate, stricter pre-registered trigger, the distillation-assisted contingency arm (E6-KD) is
> run if the observed E3 → E6 reduction in clean-test mIoU exceeds 1.0 percentage point … This
> 1.0-point trigger for running the contingency is distinct from, and stricter than, the 2.0-point
> margin used for the formal E3-versus-E6 non-inferiority decision."

| element | value |
|---|---|
| trigger | observed clean E3→E6 mIoU drop **> 1.0 pp**; equality does not trigger |
| relation to §5 | distinct and **stricter** than the −2.0 pp non-inferiority margin |
| what it changes | runs an **additive contingency arm** (E6-KD), reported alongside E6, **not a replacement** for E6 (`docs/updated_ch3_sync_audit.md:35-39`) |
| what it changes numerically | reduced distillation weights, set **below** their E3 values; exact values **UNRESOLVED** — see §10 |
| status | **no p-value, not a Holm member** |
| pre-registered | ch3 calls it a "**pre-registered trigger**"; recording it here before any E3 or E6 number exists is what makes that claim checkable |

---

## 7. λ_logit — grid, selection rule, and carry-forward

| element | value | source |
|---|---|---|
| grid | **{0.25, 0.5, 1, 2, 4}** | `configs/distill.py:30`; `IMPLEMENTATION_CONTRACT.md:182` |
| selecting split | **validation** — "validation-only" | `IMPLEMENTATION_CONTRACT.md:627`; `configs/distill.py:29` "selected via validation sweep" |
| selection criterion | candidate maximizing dataset-level **validation** mIoU; ties → smaller weight; boundary value reported as such | `docs/open_questions.md:716-721` |
| sweep seed | **42** | `configs/distill.py:31`; `IMPLEMENTATION_CONTRACT.md:182` |
| boundary policy | "boundary winner reported rather than extending the grid" | `IMPLEMENTATION_CONTRACT.md:627` |
| carried into E3 | **unchanged** — `"reused_unchanged_in_e3": True` | `configs/distill.py:33`; `IMPLEMENTATION_CONTRACT.md:182` "reused **unchanged** in E3" |
| test-set firewall | `test_used_for_selection: False` | `configs/distill.py:74`; asserted at `scripts/smoke_realrun_decisions.py:151-152` |
| decision order | select `DISTILLATION_GRAD_CLIP_NORM` → freeze → run λ sweep → freeze λ → official E2/E3 | `IMPLEMENTATION_CONTRACT.md:623-625`; `configs/distill.py:89-91` |
| value | **UNRESOLVED** — see §10 | `configs/distill.py:29` |

**Semantics tag.** λ_logit's numeric value is only meaningful relative to the spatial grid the KD term
is computed on: `LOGIT_KD_SEMANTICS = "logitkd@os8-64x64-of-512"`, with
`"logitkd@full-512x512-upsampled"` recorded as superseded (`configs/distill.py:18-22`). A λ selected
under the superseded semantics "must NEVER be consumed under the current tag without an explicit,
recorded override."

---

## 8. Metric definitions — three eligibility levels, coexisting by design

`src/eval/metrics.py:3-18` states the policy directly:

> "Semantics are frozen by docs/EVALUATION_CONTRACT.md sections 3.1/3.2 (decisions D3/D3b in
> docs/open_questions.md). **THREE ELIGIBILITY RULES COEXIST BY DESIGN -- do not "harmonise" them**"

| level | rule | implementation |
|---|---|---|
| **dataset-level mIoU** | **UNION-present**: eligible iff `UN_c = GT_c + PR_c − TP_c > 0`. A class with `GT_c = 0` but `PR_c > 0` scores IoU 0 and **is counted**; a class absent from GT and prediction alike is omitted | [`metrics.py:128-130`](../src/eval/metrics.py#L128) — `eligible = _restrict(un > 0, class_indices)` |
| **dataset-level mAcc** | **GT-present**: eligible iff `GT_c > 0`. Prediction-only classes omitted. "The asymmetry vs mIoU is genuine benchmark behaviour" | [`metrics.py:140-141`](../src/eval/metrics.py#L140) — `eligible = _restrict(gt > 0, class_indices)` |
| **dataset-level Dice** | **UNION-present**, project-defined descriptive extension; "the official benchmark never computes mDice" | [`metrics.py:153-154`](../src/eval/metrics.py#L153) |
| **per-image mIoU** | **GT-present**: "preregistered by ch3 section (f) 'per-image absent-class exclusion'"; "`_miou_from_cm` / `per_image_miou` are the per-image path and are **DELIBERATELY** left on the GT-present rule" | [`metrics.py:74-83`](../src/eval/metrics.py#L74) — `present = row > 0`, reached via [`:177-184`](../src/eval/metrics.py#L177) |

`_macro` returns `NaN` when nothing is eligible ([`metrics.py:109-113`](../src/eval/metrics.py#L109)).

**Primary inferential unit** (`IMPLEMENTATION_CONTRACT.md:425-428`): per-image **disease-only mIoU**,
115 disease classes = mask values 1–115, background index 0 excluded, per-image absent-class exclusion,
255 always excluded.

**Checkpoint/headline metric is different and deliberately so** (`IMPLEMENTATION_CONTRACT.md:430-436`,
decision D1): best-val checkpoint selection and headline validation reporting use **all-class**
validation mIoU; disease-only is secondary/provisional there. "This is **distinct from and does not
change** the **per-image disease-only mIoU _primary inferential_ unit**."

**Eligible-class count is recorded per run**, because under union-present it is model-dependent at the
dataset level (`EVALUATION_CONTRACT.md:188-190`). Per-image counts `n_eligible_all_class`,
`n_eligible_disease_only` and `gt_disease_classes` are frozen fields of `per_image.jsonl`
(`EVALUATION_CONTRACT.md` §5.4; [`src/eval/evaluate.py:145-158`](../src/eval/evaluate.py#L145)).

**mIoU-C** (`IMPLEMENTATION_CONTRACT.md:478`): "mean of per-corruption-type mIoU, each averaged over
**severities 1–3**." Severity 4 descriptive-only; severity 5 excluded. Five corruptions, canonical
identifiers frozen in `configs/corruption_protocol.json`: `motion_blur`, `gaussian_noise`,
`jpeg_compression`, `brightness`, `fog`.

---

## 9. Seeds

`IMPLEMENTATION_CONTRACT.md:300-306` (B6):

| element | value |
|---|---|
| **primary seed** | **42**, across `torch`, `numpy`, python `random` `[ch3]` |
| determinism flags | set **before CUDA init**: `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| **multi-seed plan** | three-seed validation planned for **E1 and E3** |
| the two extra seed values | **UNRESOLVED** — see §10 |
| E4/E7 | recomputed per seed |
| E5/E6 | fine-tuned per seed **where compute permits** |
| Teacher | trained **once** |
| E2 | repetition **optional** |
| bootstrap root seed | **42** (`STATISTICAL_ANALYSIS_CONTRACT.md:812`) |
| λ sweep seed | **42** (§7) |
| PTQ calibration seed | **42**, ~128 images from the training partition (`IMPLEMENTATION_CONTRACT.md:282`) |

**`num_workers` is reproducibility-relevant** and "must be reported alongside seed 42"
(`IMPLEMENTATION_CONTRACT.md:307-309`) — "Seed 42 alone does **not** determine the realized
augmentation sequence."

**Resume behaviour, measured before any result existed** (`reports/b36_resume_identity.md`, commit
`569cfbb`): a resumed run restores model, optimizer, scheduler and all RNG streams but not intra-epoch
sample position; it is reproducible bit-for-bit from its own checkpoint, not bit-identical to an
uninterrupted run, and the divergence is a reshuffle within the same permutation stream.

---

## 10. UNRESOLVED commitments — recorded, not resolved

**None of these is resolved by this document.** Each is being fixed **before** any E1–E7 result
exists; that ordering is the point.

| # | commitment | resolving mechanism | resolves at | source |
|---|---|---|---|---|
| U1 | **λ_logit** | pre-registered validation grid sweep {0.25, 0.5, 1, 2, 4}, seed 42, validation-only, ties → smaller weight | **E2**, before official E2/E3 runs | `configs/distill.py:29`; `open_questions.md:716-721` |
| U2 | **E6-KD reduced distillation weights** | only if the §6 contingency triggers (>1.0 pp); weights set **below** their E3 values | conditional, **post-E6** | `configs/quant.py:33`; `open_questions.md:722-726` |
| U3 | **`DISTILLATION_GRAD_CLIP_NORM`** | pilot on E2 over candidates (1.0, 5.0), 8,000 iters, λ fixed at 1.0; higher validation mIoU wins; tie → 5.0 | **before** official E2/E3; `status: PILOT_REQUIRED`, `selected_value: None` | `configs/distill.py:65-95` |
| U4 | **`QAT_GRAD_CLIP_NORM`** | separate pilot on E5, candidates (1.0, 5.0); independent of U3 | **before** official E5/E6; `status: PILOT_REQUIRED`, `selected_value: None` | `configs/quant.py:88-99` |
| U5 | **the two multi-seed values beyond 42** | confirmed at run time; "all seeds used are reported in the appendix" | if/when multi-seed E1/E3 runs execute | `IMPLEMENTATION_CONTRACT.md:304-305`, `:767`; `open_questions.md:734-738` |
| U6 | **Sigmoid FixedQParams `scale` / `zero_point`** | runtime-derived; gated on verifying QNNPACK INT8 Sigmoid support in the LR-ASPP global-pool branch | **before E4** | `configs/quant.py:136-137` |
| U7 | **pooled dataset-level robustness estimand** | ⏸️ **DEFERRED** `[project]`, explicitly "not a limitation" | post-E1 | `open_questions.md:550` (D22) |
| U8 | **RunPod hardware specifics** | logged at run time; RTX 4090 (~$0.34/hr) is the working assumption, not a commitment | at execution | `open_questions.md:739-743` |
| U9 | **NMF/Hamburger seed control (teacher)** | to be settled against the pinned stack; deliberately not faked as a config key MMSeg would ignore | teacher stage | `configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:356-358` |

**U1, U3 and U4 are selection parameters and are fixed on validation only.** The test split is never
used for selection (`configs/distill.py:74`, `configs/quant.py:98`), and both pilots are excluded from
"test evaluation, robustness, hypothesis testing, statistics family" (`configs/distill.py:94`).

---

## 11. Anchor — and the proof that no result exists yet

| field | value |
|---|---|
| repo HEAD at time of writing | **`569cfbb`** |
| branch | `master`, in sync with `origin/master` |
| date written | **2026-09-05** |
| preceding commits | `281d765` (B34b gate hardening), `591a85e` (B32c closeout) |

**No E1–E7 result existed at this commit.** Proof of absence — commands and their actual results:

```
$ git ls-files | grep -E "summary\.json|per_image\.jsonl|bootstrap\.npz|family\.json|MANIFEST\.sha256"
[exit 1]

$ git ls-files | grep -E "\.pt$|\.pth$|\.npz$"
[exit 1]

$ git ls-files | grep -iE "outputs/|results/"
[exit 1]
```

All three returned no output and exit status 1: **no result artifact of the frozen schema, no model
checkpoint, and no output directory is tracked at `569cfbb`.** This is consistent with the repository's
design — `train_e1.py` hard-guards against an in-repo `--ckpt-dir`, so checkpoints live outside the
tree by construction.

**What has been produced at this commit is preparation, not results:** dataset audits, the CE
class-weight artifact (B18a), smoke tests, the pre-flight gate, and the B36 resume drill — none of
which contains an E1–E7 accuracy, efficiency, or robustness number.

---

## 12. Source disagreements — recorded, NOT adjudicated

Five found. Both readings are given with sources. **This document picks no winner.** Where the
repository has already frozen a project-decision, that fact is reported as repository state, which is
not the same as this document resolving anything.

### Disagreement 1 — membership of the eight-test family `[ch3 internal]`

| location | what it says |
|---|---|
| **ch3 §B summary sentence** | "Additional comparisons, including E1 vs. E2 (H₁a), **E1 vs. E3** …, **E3 vs. E6** (accuracy retention, H₁d), and E1 vs. E6 … complete the inferential set of eight tests under Holm-Bonferroni correction." |
| **ch3 Table 3.6, E1 vs E3 row** | "Reported **descriptively** … **Not included in the eight-test Holm-Bonferroni family**." |
| **ch3 §A / §F, three separate times** | "This check is reported separately from the eight-test Holm-Bonferroni superiority family." |

Admitting both E1→E3 and E3→E6 yields **ten** members against ch3's own explicit eight-test,
"seven clean + one robustness" structure.

> **CONSEQUENCE — this is not bookkeeping.** Family size is the Holm divisor: the smallest p-value is
> compared against `α/m`, the next against `α/(m−1)`, and so on. Changing `m` from 8 to 10 changes
> **every adjusted p-value in Chapter 4**, and can flip a significance decision at the margin without
> any change to the underlying data.

**Which number the repo currently implements: EIGHT.** `CANONICAL_COMPARISON_IDS` is an eight-member
tuple at [`src/stats/tests.py:32-35`](../src/stats/tests.py#L32); `family.json` requires "exactly eight
serialized `ComparisonResult` records" (`STATISTICAL_ANALYSIS_CONTRACT.md:1193`), and
`expected_comparison_ids` is "the eight-member canonical tuple" whose order the implementation asserts
against the contract (`:1127`).

Full ledger, with both readings and the frozen project-decision:
`STATISTICAL_ANALYSIS_CONTRACT.md:24-45` (§0.1).

### Disagreement 2 — test-set n, and the split counts

| source | value |
|---|---|
| **ch3** p. 43 | "approximate 70/10/20 train/validation/test split, yielding roughly **5,442 / 778 / 1,554** images" |
| **ch3** pp. 50, 55, 58, 61, 63 | "**1,554** test images"; "Since there are **1,554** per-image differences at hand"; "n ≈ **1,554**" |
| **`EVALUATION_CONTRACT.md:42`** | "**The authoritative test count is 1,561.** Any document, prompt, or code path stating ≈1,554 is stale." |
| **`EVALUATION_CONTRACT.md:35-39`** | "The `1,554` value is NOT a count — never use it": 7,774 × 0.20 = 1,554.8, i.e. a nominal proportion, not a measurement |
| **`IMPLEMENTATION_CONTRACT.md:491`** | sensitivity t-test "(CLT at n = **1,561**)" |
| **`configs/data.py:22`** | `SPLIT_SIZES = {"train": 5367, "val": 846, "test": 1561}` `[empirical, counted]` |

Affects the reported n of every paired test and the Hodges-Lehmann feasibility argument (which is
stated at n = 1,561, `STATISTICAL_ANALYSIS_CONTRACT.md:288-289`). Recorded, not adjudicated.

### Disagreement 3 — the Wilcoxon call signature

| source | signature |
|---|---|
| **ch3** p. 59 | `scipy.stats.wilcoxon(d, zero_method='pratt', alternative='greater')` |
| **`IMPLEMENTATION_CONTRACT.md:487-488`** | `scipy.stats.wilcoxon(d, zero_method='pratt', correction=True, alternative='greater', method='approx')` |

The contract's call adds `correction=True` and `method='approx'`. These are **not absent from ch3's
prose**: p. 60 specifies "a continuity correction of ±0.5 is additionally applied to W before computing
z" and derives the large-sample normal approximation explicitly. So the contract's signature reads as
an operationalisation of ch3's prose rather than a contradiction of it — **but the two literal call
strings differ, and that is recorded here rather than smoothed.**

### Disagreement 4 — the per-image artifact schema

| source | specification |
|---|---|
| **ch3** p. 50 | "stored as a **CSV** with columns **{image_id, stage, mIoU, Dice, mAcc}**" |
| **ch3** p. 65 | "Reported only at the dataset level, **per-image Dice is not computed** because it would be redundant with per-image mIoU under the monotonic Dice = 2·IoU/(1+IoU) relationship." |
| **`EVALUATION_CONTRACT.md` §5.4** | `per_image.jsonl` (JSON Lines) with `image_id`, `clean_image_id`, `manifest_index`, `condition`, `all_class_miou`(+status), `disease_only_miou`(+status), `n_eligible_all_class`, `n_eligible_disease_only`, `gt_disease_classes` |

**ch3 contradicts itself**: p. 50 lists per-image `Dice` and `mAcc` columns, p. 65 says per-image Dice
is not computed. The format difference (CSV → JSONL) *is* reconciled — `EVALUATION_CONTRACT.md:280`
explicitly rejects "summary + CSV only" as a layout with stated reasons — and the 1,554 → 1,561 count
is covered by disagreement 2. The **column list itself is reconciled nowhere**. Analysis:
`reports/b35_eligibility_and_stale_labels.md` §2.3.

### Disagreement 5 — Albumentations

| source | statement |
|---|---|
| **ch3 §D** (via `open_questions.md:729-730`) | "version pinned in the reproducibility manifest" but no number given |
| **`requirements-e1.txt:10-11`** | "Also EXCLUDED: opencv-python and albumentations — the E1 graph uses NumPy/PIL transforms only (no OpenCV / Albumentations import anywhere in E1)" |

Not a version lookup awaiting an answer: the library is absent from the E1 pipeline entirely. The
companion item, **statsmodels**, *is* resolved — `requirements.lock:63` pins `statsmodels==0.14.6`,
though `open_questions.md:731` still records it as unconfirmed.

---

## 13. What this document does NOT do

- **It resolves nothing.** Every item in §10 remains **UNRESOLVED** after this document exists, exactly
  as it was before. U1–U9 are recorded with their resolving mechanism and stage; none is decided,
  narrowed, or given a provisional value here.
- **It adjudicates none of the five disagreements in §12.** Both readings are recorded with sources.
  Where §12 reports that the repository has frozen a project-decision (disagreement 1) or implements
  one number rather than another, that is a statement of repository state — not an adjudication by
  this document, and not a claim that the manuscript has been corrected.
- **It changes no code, config, contract, or manuscript.** It is a consolidation of commitments that
  already existed in the sources cited; it introduces none of its own.
- **It is not a results document and contains no result.** No E1–E7 accuracy, efficiency, or
  robustness number appears anywhere above.
- **It does not make the analysis correct.** It makes the analysis *checkable* — a reader can compare
  what was committed here against what Chapter 4 eventually reports.

**Its evidentiary value depends entirely on this file's commit preceding any E1–E7 result.** A
pre-registration written after results exist is worthless, and that ordering is verifiable from git
history rather than from any assertion made here.

**The artifact patterns whose later appearance post-dates this document** — the same globs as the §11
proof of absence, all of which returned no tracked file at `569cfbb`:

```
summary.json      per_image.jsonl      bootstrap.npz      family.json      MANIFEST.sha256
*.pt              *.pth                *.npz
outputs/          results/
```

If any file matching these enters the repository in a commit **after** the one that adds this
document, this pre-registration precedes it and the ordering claim holds. If any such file is found in
a commit **at or before** this document's commit, the claim is void and must be withdrawn — that check
is the reader's, and it is deliberately made easy to run.
