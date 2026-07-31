# Statistical Analysis Contract — frozen inferential protocol + result artifact (A3-0)

> **Status: FROZEN (2026-07-28).** This is the authoritative definition of *how every inferential
> number in this thesis is computed* and *what a statistics run must emit*. A3a (paired inference)
> and A3b (BCa) implement this document and must not invent additional statistical decisions.
>
> Companions: [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §(f) (which analyses the
> thesis reports) · [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md) §§3–5 (how the per-image and
> dataset-level inputs are computed) · [open_questions.md](open_questions.md) D7-series (decision
> records).

**Protocol identifiers:** `schema_version = "plantseg-stats/1.0.0"` ·
`statistical_protocol = "plantseg-stats-protocol/1.0.0"`

**Rule labels used throughout:**
`[ch3]` preregistered in Chapter III · `[empirical]` corrected repository fact ·
`[project-decision]` resolution where ch3 is silent or internally ambiguous ·
`[implementation]` serialization or software behaviour.

---

## 0. Authority and contradiction ledger

### 0.1 The eight-test family — an internal Chapter III inconsistency

Chapter III is **internally inconsistent** about the membership of the Holm family. Both readings
are recorded here; neither is hidden, and nothing external "corrects" ch3.

| Location | What it says |
|---|---|
| **§B summary sentence** | *"Additional comparisons, including E1 vs. E2 (H₁a), **E1 vs. E3** (supporting H₁a and H₁b), **E3 vs. E6** (accuracy retention, H₁d), and E1 vs. E6 … complete the inferential set of eight tests under Holm-Bonferroni correction."* |
| **Table 3.6, E1 vs E3 row** | *"Reported **descriptively** as the total FP32 distillation gain (E1→E3): mean ΔmIoU and Hodges-Lehmann shift with BCa 95 % confidence intervals. **Not included in the eight-test Holm-Bonferroni family**."* |
| **§A / §F, stated three separate times** | *"E6 is considered non-inferior to E3 if the lower bound of the paired BCa 95 % confidence interval (B = 10,000) … exceeds −2.0 percentage points. **This check is reported separately from the eight-test Holm-Bonferroni superiority family.**"* |
| **§F robustness** | *"The one inferential robustness test uses per-image mIoU-C values from E1 and E6 … subject to the same Holm-Bonferroni correction applied across all 8 inferential tests."* |
| **§F H₁c** | *"tested by two paired comparisons: E4 vs E5 (undistilled) and E7 vs E6 (distilled), **both contributing to the family-wise Holm-Bonferroni correction over all 8 inferential tests**."* |

**Arithmetic check:** the family is stated everywhere to contain **eight** members, and is
independently described as **seven clean per-image comparisons plus one per-image mIoU-C
comparison**. Admitting both E1→E3 and E3→E6 would yield **ten**, contradicting the explicit
eight-test structure in the same chapter.

**Frozen reconciliation `[project-decision]`.** The **specific comparison table (Table 3.6)**, the
**formal hypothesis statements**, the explicit **"seven clean + one robustness"** structure, and the
**repeated explicit exclusions** govern the ambiguous §B summary sentence. The §B sentence is
recorded as loose prose, not as a competing specification.

### 0.2 Sample size

Chapter III writes `n = 1,554` `[ch3]`. That value is nominal-ratio arithmetic (7,774 × 0.20) and is
superseded by the empirical audit: **n = 1,561** `[empirical]` — see `EVALUATION_CONTRACT.md` §0 and
`open_questions.md`. All power/CLT reasoning in ch3 is unaffected in substance.

### 0.3 Software-argument gap

ch3's software table specifies `zero_method='pratt'`, `alternative='greater'`, `method='approx'` but
does not mention `correction`. `IMPLEMENTATION_CONTRACT.md` §(f) additionally specifies
`correction=True`. These are compatible — SciPy applies the continuity correction only under
`method='approx'` — so **both are frozen together** `[ch3 + project-decision]`.

---

## 1. The primary Holm family — exactly eight comparisons

For every comparison, **`delta = candidate − baseline`**, and **positive favours the candidate**.

| # | comparison_id | baseline | candidate | metric | label |
|---|---|---|---|---|---|
| 1 | `accuracy_e1_e2` | E1 | E2 | clean per-image **disease-only** mIoU | `[ch3]` H₁a |
| 2 | `accuracy_e2_e3` | E2 | E3 | clean per-image disease-only mIoU | `[ch3]` H₁b |
| 3 | `accuracy_e4_e5` | E4 | E5 | clean per-image disease-only mIoU | `[ch3]` H₁c |
| 4 | `accuracy_e7_e6` | E7 | E6 | clean per-image disease-only mIoU | `[ch3]` H₁c ext. |
| 5 | `accuracy_e4_e7` | E4 | E7 | clean per-image disease-only mIoU | `[ch3]` H₁d support |
| 6 | `accuracy_e5_e6` | E5 | E6 | clean per-image disease-only mIoU | `[ch3]` 2×2 ablation |
| 7 | `accuracy_e1_e6` | E1 | E6 | clean per-image disease-only mIoU | `[ch3]` H₁d (ii) |
| 8 | `robustness_e1_e6` | E1 | E6 | **per-image mIoU-C** | `[ch3]` H₁d (iii) |

### Explicitly EXCLUDED from the primary family

- **E1→E3** — descriptive total-FP32-distillation gain. Reported with point estimates **and** the
  ch3-required BCa 95 % uncertainty intervals, but **no Holm-family p-value and no superiority
  claim** `[ch3 Table 3.6]`.
- **E3→E6** — dataset-level non-inferiority, §7 `[ch3]`.
- Paired **t-test** sensitivity p-values — they never enter the family `[ch3]`.
- **Teacher** comparisons — descriptive upper bound only `[ch3]`.
- **Dice, mAcc, aAcc, RPD, rCD, efficiency** metrics — descriptive only `[ch3]`.

---

## 2. The per-image mIoU-C inferential vector `[ch3]`

One scalar per **clean image** and stage, built from the **frozen per-image disease-only mIoU**
field of each condition artifact — **never** from dataset-level mIoU-C summaries.

```
1. read per-image disease-only mIoU for each required (corruption, severity) cell
2. average severities 1, 2, 3 within each corruption type      -> 5 corruption means
3. equal-weight mean of the five corruption means               -> mIoU-C for that image
```

### 2.1 Canonical machine vocabulary `[project-decision, A3b-0 2026-07-28]`

ch3 confirms the five types in prose (JPEG compression, not shot noise — "shot noise" appears
nowhere) but froze **no machine-readable identifier**, and `EVALUATION_CONTRACT.md` types
`condition.name` as an unconstrained string. A3a surfaced this as a blocker. It is now closed by
adopting the **vendored `imagecorruptions` reference function names** that ch3 already selects:

| machine identifier (`condition.name`) | display label | reference function |
|---|---|---|
| `motion_blur` | motion blur | `motion_blur` |
| `gaussian_noise` | Gaussian noise | `gaussian_noise` |
| `jpeg_compression` | JPEG compression | `jpeg_compression` |
| **`brightness`** | brightness variation | `brightness` |
| `fog` | fog | `fog` |

Machine-readable source of truth: **`configs/corruption_protocol.json`**
(`schema_version = plantseg-corruptions/1.0.0`). The order above is the frozen
`corruption_order` and follows ch3's presentation order.

- **`brightness` is the official identifier**; *"brightness variation"* is explanatory thesis prose
  and is **not** an identifier. `brightness_variation` is **rejected**.
- Also rejected: `motion-blur`, `Motion_Blur`, spaces, capitalisation variants, filename-derived
  aliases, and any alias map. No official corruption artifact exists yet, so no migration
  compatibility is owed.
- **Display wording never controls artifact identity.**
- Severity means are computed in this canonical order. Equal weighting makes the arithmetic
  order-invariant, but the order remains **binding** for deterministic validation, serialization
  and reporting.
- **Official** corruption analysis must load the protocol file and may **not** inject an
  alternative vocabulary. `NONOFFICIAL_SMOKE` tests may inject an explicitly synthetic grid, which
  can never enter official statistics.

This freezes **vocabulary and severity roles only**. The corruption implementation bytes, its
parameters, the seed policy and the cache protocol remain a future corruption-scaffold task, which
must also verify that the vendored implementation really exposes these five function names.

**Official inferential grid: exactly 5 × 3 = 15 cells** per clean image and stage. **Severity 4** is
descriptive-degradation only and is **not** part of the inferential scalar; **severity 5 is excluded**.

The nested mean equals a flat 15-value mean on a complete grid, but **the nested definition is
retained in code and reporting** `[project-decision]` because it mirrors the methodology and makes a
partial grid impossible to average away silently.

**Official integrity — fail on any of:** missing cell · duplicate cell · unexpected corruption ·
unexpected severity · unequal clean-image ID sets across condition artifacts · undefined
condition-level per-image score · mismatched split or `split_manifest_sha256` · mismatched
`class_map_sha256` · mismatched `metric_protocol` · mismatched `preprocess_protocol` · any
available-case averaging or silent row deletion.

---

## 3. Primary test — Wilcoxon signed-rank `[ch3]`

```python
scipy.stats.wilcoxon(d, zero_method="pratt", correction=True,
                     alternative="greater", method="approx")
```

where `d = candidate − baseline`, float64, aligned by canonical image ID.

Frozen: one-tailed improvement alternative · **explicit normal approximation** (`method="approx"`,
never `"auto"`) · continuity correction · Pratt zero treatment · α = 0.05 **before** family
correction · **no normality-test gatekeeper** (ch3 explicitly rejects Shapiro-Wilk at this n).

Recorded outputs: `W` statistic · normal-approximation `z` · raw one-sided p-value · the full
configuration · status and warnings. Verified behaviour: `z > 0` for a positive shift, agreeing with
the `candidate − baseline` direction.

**Official execution must occur on the pinned stack (§10).** Development smokes may run on the dev
stack but must record themselves as non-official.

---

## 4. Degenerate and tie policy

### 4.1 Ties are VALID — not degenerate `[project-decision]`

The following remain **valid** Wilcoxon inputs and must **not** be flagged degenerate:

- repeated absolute differences
- tied nonzero differences
- every nonzero difference sharing one absolute magnitude, provided ≥2 nonzero observations remain

SciPy's tie correction is an intended part of the selected approximate procedure. A vector is
**never** marked degenerate merely because nonzero values are tied.

### 4.2 All-zero paired differences

| Field | Frozen value |
|---|---|
| status | `degenerate_all_zero` |
| `W`, `z` | `null` |
| raw p-value | **1.0** |
| reject | `false` |
| mean Δ, median Δ, Hodges-Lehmann | **0.0** |
| rank-biserial | `null`, status `undefined_no_nonzero_rank_sum` |
| Cohen's dz | `null`, zero-SD status |

Library warnings are suppressed **and recorded**; SciPy returns `NaN` here, and **no NaN may reach
JSON** `[implementation]`.

### 4.3 Fewer than two nonzero paired differences

status `insufficient_nonzero_pairs` · no official inferential conclusion · Wilcoxon `z` and `p` are
`null` · effect sizes that remain mathematically defined may still be reported · **no fabricated
finite statistic**.

### 4.4 Invalid inputs — integrity errors, not statuses

Unequal lengths · non-finite values · empty vectors · duplicate identities · unmatched identities.
These **fail before testing** and are never represented as a statistical-result status.

---

## 5. Sensitivity test — paired t-test `[ch3]`

```python
scipy.stats.ttest_rel(candidate, baseline, alternative="greater")
```

Recorded: `t` · one-sided raw p · degrees of freedom · mean paired difference · sample SD of
differences (`ddof=1`) · standard error · status.

The Wilcoxon result is **primary**. The t-test is a sensitivity analysis: it adds **no** p-values to
the Holm family, does **not** override Wilcoxon, and **must still be reported when the two
conclusions disagree** (ch3: *"both results are still reported, with the Wilcoxon conclusion
preferred"*).

**Degenerate handling:** all-zero → `t = null`, `p = 1.0`, status `degenerate_all_zero` ·
constant nonzero differences → `t`, `p` = `null`, status `degenerate_zero_variance_nonzero`, with
mean and median shift still reportable and **no ±Infinity serialized** · fewer than two observations
→ status `insufficient_pairs`.

---

## 6. Family-wise control — Holm-Bonferroni `[ch3]`

```python
statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method="holm",
                                          is_sorted=False, returnsorted=False)
```

One family containing exactly the **eight primary Wilcoxon raw p-values**. Canonical comparison
order is preserved in the output (verified: statsmodels returns results in input order).

**An official family may NOT be finalised when** an expected ID is missing · an ID is duplicated ·
an unexpected ID is present · any primary raw p-value is undefined · any comparison failed integrity
validation.

**Audit fields:** raw p · Holm-adjusted p · sorted rank · step denominator · nominal step threshold ·
reject flag · family completion status. The audit sort key is the stable
`(raw_p_value, comparison_id)` `[implementation]`.

**Strict boundary `[project-decision]`.** Chapter III writes the threshold with strict `<`. Frozen:
equality at the step threshold does **not** reject; equality of the Holm-adjusted p-value with α does
**not** reject. Statsmodels remains the adjusted-p implementation, but the reject decision is
independently verified against the strict rule; where the library Boolean differs only by exact
equality, the boundary condition is recorded and the **frozen strict decision** is used.

---

## 7. Effect sizes

### 7.1 Matched-pairs rank-biserial correlation `[project-decision, Pratt-compatible]`

1. rank **all** absolute differences, **including zeros**
2. average ranks for ties
3. zeros participate in rank assignment
4. **zero-difference ranks enter neither signed sum**
5. `r_rb = (R₊ − R₋) / (R₊ + R₋)`

Positive favours the candidate; range `[−1, 1]`.

**Why this denominator:** it matches ch3's signed-rank formulation ("the rank sums of positive and
negative differences"); zeros still influence the ranks assigned to nonzero differences, as Pratt
requires; zero ranks carry no directional evidence; and the sum of nonzero signed ranks preserves the
declared `[−1, 1]` range. Using the total `n(n+1)/2` would include zero ranks in the denominator and
**shrink the attainable range whenever zeros exist** — rejected.

`R₊ + R₋ == 0` → value `null`, status `undefined_no_nonzero_rank_sum`.

### 7.2 Cohen's dz `[ch3]`

`mean(d) / std(d, ddof=1)`; positive favours the candidate. Benchmarks 0.20 / 0.50 / 0.80 are
interpretive only. Zero sample SD → `null` + explicit status, **never Infinity**.

### 7.3 Hodges-Lehmann paired shift `[ch3]`

Exact one-sample paired-difference estimator: median of all Walsh averages `(dᵢ + dⱼ)/2` with
**i ≤ j** (`n(n+1)/2` terms). Deterministic partition-selection is permitted. **No approximation is
required at n = 1,561** — see §11. Positive favours the candidate.

### 7.4 Engineering shifts `[ch3]`

Mean Δ in metric units · **mean Δ in percentage points = `100 × mean_delta`** · median Δ in metric
units (descriptive). **No Chapter-IV display rounding in the artifact** `[implementation]`.

---

## 8. BCa bootstrap

Production count **B = 10,000** `[ch3]`, set **explicitly** (SciPy's own default is 9,999).
Paired, **unstratified** resampling over canonical image IDs — ch3 preregisters resampling test
images and specifies no strata `[ch3]`. Every replicate uses **one shared index vector for both
compared stages**. float64 throughout.

### 8.1 Deterministic seed protocol `[project-decision]` — **AMENDED A3b-1, 2026-07-29**

> **AMENDMENT A (not a clarification).** The original derivation named root seed 42 but never
> consumed it, so `root_seed` was inert provenance and could not act as a protocol lever. The
> **superseded** pre-A3b-1 canonical string was, verbatim:
>
> ```
> "plantseg-stats/1.0.0\0" + analysis_id + "\0" + comparison_id + "\0" + statistic_name
> ```
>
> It is retained here only as quoted history and is **no longer active**. No A3b implementation and
> no official statistical artifact ever used it — at amendment time `src/stats/bootstrap.py`,
> `noninferiority.py` and `artifact.py` did not exist and no `family.json` / `bootstrap.npz` had ever
> been written — so no published number changes. Decision record: `open_questions.md` D21.

Root seed **42**, frozen for protocol 1.0.0 as a **rendered parameter**, not a hardcoded literal.
Per-statistic seed from canonical UTF-8 bytes, in exactly this field order:

```python
b"plantseg-stats/1.0.0\x00"
+ b"root_seed=" + str(root_seed).encode("ascii") + b"\x00"
+ analysis_id.encode("utf-8")    + b"\x00"
+ comparison_id.encode("utf-8")  + b"\x00"
+ statistic_name.encode("utf-8")          # NO trailing NUL
```

Then: SHA-256 the exact bytes · take the **first 16 digest bytes** · interpret them as **one unsigned
big-endian 128-bit integer** · pass that integer **alone** as entropy to `numpy.random.SeedSequence` ·
construct `numpy.random.PCG64` · construct one `numpy.random.Generator`.

**Forbidden:** passing 42 as a second `SeedSequence` entropy word · building `SeedSequence(42)` and
spawning children · prepending or appending any other textual seed · Python's built-in `hash` ·
truncating the entropy to 32 or 64 bits · storing the 128-bit integer in `int64` anywhere. The
derived integer is carried as a **decimal string** in JSON and as `uint8[16]` in the NPZ.

Namespace strings are rejected on leading whitespace · trailing whitespace · embedded NUL · any
identifier outside §8.7.1.

Recorded: root seed · canonical seed input · canonical bytes as hex · full SHA-256 · derived integer ·
BitGenerator · NumPy version. This makes results **independent of comparison ordering and parallel
execution**.

### 8.2 Bias correction `z0`

```
proportion = ( count(boot < observed) + 0.5 * count(boot == observed) ) / B
z0 = Phi^-1( clip(proportion, 1e-12, 1 - 1e-12) )
```

Ties at the observed statistic count as **one half** `[project-decision]`; clipping prevents infinite
transformed quantiles.

### 8.3 Acceleration

Leave-one-image-out jackknife estimates with the standard BCa acceleration formula. Recorded:
jackknife count · jackknife values · jackknife mean · numerator · denominator · acceleration ·
status.

### 8.4 Quantiles

NumPy **linear** interpolation, specified explicitly — never an evolving library default
`[implementation]`.

### 8.5 Interval types — **field name AMENDED A3b-1, 2026-07-29**

> **AMENDMENT B (not a clarification).** The recorded field name was `confidence_type`. The
> **superseded** pre-A3b-1 wording was, verbatim: *"Recorded as `confidence_type = "one_sided_lower"`,
> `confidence_level = 0.95`, a finite `lower_bound` field"*. The active field name is now
> **`interval_type`**, pairing with `interval_method` and matching §12.2's existing "requested
> interval type · **actual** interval method" wording. The **value** `one_sided_lower`,
> `confidence_level = 0.95` and the dedicated `lower_bound` key are **unchanged** — only the field
> name changes. `confidence_type` was named in exactly one place repo-wide and in no decision record,
> so no `open_questions.md` entry required an in-place edit. Decision record: `open_questions.md` D21.

**Two-sided 95 % BCa** for: mean paired Δ · median paired Δ · Hodges-Lehmann shift · the descriptive
E1→E3 intervals `[ch3]` · the supplementary dataset-level Δ intervals, whose scope was previously
unstated and is now frozen to the exact task matrix of §8.7.2 `[project-decision]`.

**One-sided 95 % BCa lower bound** for the E3→E6 non-inferiority check (§9). It is **not** an
ordinary two-sided interval. Recorded as `interval_type = "one_sided_lower"`,
`confidence_level = 0.95`, a finite `lower_bound` field, and **no fabricated finite upper bound** —
strict JSON uses a dedicated `lower_bound` key rather than serializing `Infinity` `[implementation]`.

Nominal probabilities are explicit: two-sided **0.025 / 0.975**; one-sided lower **0.05**. The
non-inferiority lower bound must **never** use 0.025.

### 8.6 Fallback — percentile only `[project-decision]`

Chapter III permits "a percentile or basic bootstrap interval" when BCa acceleration is degenerate.
To keep the procedure deterministic, **exactly one** fallback is frozen for v1.0.0: a **percentile**
interval matching the requested confidence type, labelled
`percentile_fallback_degenerate_acceleration`. **Basic intervals are not used in v1.0.0.** The method
actually used is always recorded, as ch3 requires.

**Hard failures — abort official analysis:** non-finite bootstrap estimates, or an official replicate
with no eligible dataset-level class. Do **not** silently drop the replicate, reduce `B`, or switch
estimand. Synthetic tests may exercise the failure status without weakening official behaviour.

### 8.7 Frozen reproducibility protocol `[project-decision, A3b-1 2026-07-29]`

§§8.1–8.6 left several choices that two conforming implementations could resolve differently, each of
which changes published replicates or prevents exact reconstruction. They are frozen here. **Every
identifier in §8.7.1 is part of the §8.1 seed namespace**: renaming any of them changes the random
stream and requires a new statistical-protocol version.

Identifier rules, applied to every name below: lowercase ASCII letters, digits and single underscores
only · must not contain `__` · must not begin or end with `_`. `__` is therefore reserved as an
unambiguous namespace separator (§12.3).

#### 8.7.1 Namespaces

**Analysis IDs — exactly two.** `analysis_id` is a protocol-level random-stream namespace, **not** a
run UUID, timestamp, output path, Git commit or checkpoint identifier. A separate artifact-instance
`run_id` remains permitted and does not affect sampling. Exactly one `analysis_id` governs one
statistics artifact; a mixed-namespace artifact is an integrity failure (§12.3).

| analysis_id | use |
|---|---|
| `plantseg_primary_analysis_v1` | official thesis analysis |
| `plantseg_stats_smoke_v1` | non-official synthetic contract smoke |

Arbitrary official analysis IDs are rejected under protocol 1.0.0.

**Comparison IDs — exactly ten.** The eight superiority IDs of §1 are unchanged. Two non-family IDs
are added, both **outside** the Holm family, and neither is added to `CANONICAL_COMPARISON_IDS` in
`src/stats/tests.py`:

| comparison_id | role |
|---|---|
| `descriptive_e1_e3` | descriptive total-FP32-distillation gain (§1 exclusions) |
| `noninferiority_e3_e6` | dataset-level non-inferiority (§9.2) |

Because `descriptive_e1_e3` is outside `CANONICAL_COMPARISON_IDS`, `src/stats/tests.run_comparison`
refuses it by design. Its observed scalar values are therefore obtained **only** through the public
A3a estimator primitives — see §12.3.3, Source B. It is never routed through `run_comparison`, and no
surrogate `ComparisonResult` is constructed for it.

**Statistic names — exactly four.**

| statistic_name | operates on |
|---|---|
| `mean_delta` | aligned per-image paired scalar vector |
| `median_delta` | aligned per-image paired scalar vector |
| `hodges_lehmann_shift` | aligned per-image paired scalar vector (exact Walsh median, §7.3) |
| `dataset_miou_delta` | pooled sufficient statistics — the §9.1 clean, all-class, union-present estimand |

Rejected aliases: `mean` · `median` · `hl` · `hodges_lehmann` · `pooled_miou` · `miou_delta` ·
`miou_c_delta` · `dataset_robustness_delta`.

**`dataset_miou_c_delta` is NOT frozen and does not exist under protocol 1.0.0.** See D22 — a pooled
dataset-level *robustness* estimand is deliberately deferred, not omitted by oversight.

#### 8.7.2 The 35 bootstrap tasks, in canonical order

A bootstrap task is one `(comparison_id, statistic_name)` pair. The matrix is exhaustive:

| # | comparison_id | statistic_name(s) | tasks | interval_type |
|---:|---|---|---:|---|
| 1–28 | the **seven** clean superiority IDs, in §1 order: `accuracy_e1_e2`, `accuracy_e2_e3`, `accuracy_e4_e5`, `accuracy_e7_e6`, `accuracy_e4_e7`, `accuracy_e5_e6`, `accuracy_e1_e6` | `mean_delta`, `median_delta`, `hodges_lehmann_shift`, `dataset_miou_delta` | 7 × 4 = 28 | `two_sided` |
| 29–31 | `robustness_e1_e6` (on the frozen per-image mIoU-C vector of §2) | `mean_delta`, `median_delta`, `hodges_lehmann_shift` | 3 | `two_sided` |
| 32–34 | `descriptive_e1_e3` | `mean_delta`, `median_delta`, `hodges_lehmann_shift` | 3 | `two_sided` |
| 35 | `noninferiority_e3_e6` | `dataset_miou_delta` | 1 | **`one_sided_lower`** |

**Total = 35.** Exactly **one** task is one-sided (bounds `[1]`); the other **34** are two-sided
(bounds `[2]`). `robustness_e1_e6` receives **no** pooled statistic; `descriptive_e1_e3` receives no
superiority p-value, no Holm result and no pooled dataset interval.

**Canonical record order** is the table order above: comparisons in the listed order, and within each
comparison the statistics in the listed order. This order is a **serialization rule only** — it
changes no number, because §8.1 derives each task's `Generator` independently. Reordering records must
never be understood to alter a bootstrap value.

#### 8.7.3 Interval type is a property of the task, never of the statistic

`dataset_miou_delta` appears in **two** roles with different interval types: `two_sided` on the seven
clean superiority comparisons, `one_sided_lower` on `noninferiority_e3_e6`. An implementation that
selects the interval type from `statistic_name` alone will silently emit a **two-sided
non-inferiority bound** at the wrong nominal tail. Interval type is determined by the complete
`(comparison_id, statistic_name)` pair.

#### 8.7.4 Draw order and stream ownership

Exactly one **row-wise** call per replicate:

```python
generator.integers(low=0, high=n_images, size=n_images, dtype=np.int64, endpoint=False)
```

A single bulk `(B, n_images)` call is **forbidden**. The reason is *not* a claim that bulk and
row-wise draws always differ — it is that the API call sequence is itself part of the protocol and
must not rest on undocumented cross-version equivalence. Equivalence must not be "verified" and then
used to justify the bulk call.

Each `(analysis_id, comparison_id, statistic_name)` owns **one** `Generator`, reconstructed locally
from its namespace — **35 independent streams**. `mean_delta`, `median_delta` and
`hodges_lehmann_shift` on the same comparison therefore do **not** share draws; combining streams as
a performance optimisation is forbidden. No comparison, statistic, thread or process shares a mutable
`Generator`; no global NumPy RNG is used. Parallelism is permitted only at whole-task granularity.

Consequences: working index memory is O(`n_images`) · no `[B, n_images]` matrix is ever materialised
or stored · a smaller-`B` run is an exact **prefix** of a larger-`B` run **only** when namespace,
`n_images`, NumPy version and BitGenerator are all identical. A synthetic smoke with a different
`n_images` is **not** a prefix of the official n = 1,561 stream and must not be described as one.

#### 8.7.5 BCa adjusted probabilities and nominal tails

With `z0` from §8.2 and acceleration `a` from §8.3, for nominal probability `q`, all float64:

```
zq          = Phi_inverse(q)
denominator = 1 - a * (z0 + zq)
adjusted_q  = Phi( z0 + (z0 + zq) / denominator )
```

Nominal probabilities: two-sided `q_lower = 0.025`, `q_upper = 0.975`; one-sided lower
`q_lower = 0.05`. Final quantiles use the explicit **linear** interpolation of §8.4.

**Tail evaluation order.** The lower-tail transform is evaluated first; the upper-tail transform is
evaluated second and **only** for `two_sided`. The one-sided non-inferiority task never evaluates an
upper tail.

#### 8.7.6 Adjusted-probability saturation

`Phi` returns exactly `0.0` or `1.0` for |z| beyond roughly 8.3. Such a value is **finite**, so it is
**not** a fallback condition; the corresponding bound is the minimum or maximum of the bootstrap
distribution. This is a completed BCa result and must be recorded with
`adjusted_probability_saturated_lower` / `_upper` for whichever tail saturated. Saturation warnings
arise **only** when the BCa path reaches and completes the adjusted-probability calculation.

#### 8.7.7 First-trigger fallback ladder

The §8.6 label `percentile_fallback_degenerate_acceleration` is the **sole** fallback method, but its
triggers are heterogeneous, so the ladder is evaluated **strictly in order and short-circuits**:

| # | predicate |
|---|---|
| **P1** | fewer than two jackknife estimates exist |
| **P2** | any jackknife estimate is non-finite |
| **P3** | the centered-jackknife squared sum is exactly zero |
| **P4** | the acceleration denominator is zero or non-finite |
| **P5** | the calculated acceleration is non-finite |
| **P6** | a required BCa transform denominator `1 - a*(z0+zq)` is zero or non-finite |
| **P7** | a required adjusted probability is non-finite |
| **P8** | two-sided adjusted lower probability **>=** adjusted upper probability |

At the **first** true predicate: stop evaluating later predicates · use the percentile fallback ·
record **exactly one** `fallback_p*` warning, for that predicate · **do not compute values whose
prerequisites have already failed**. There is **at most one** `fallback_p*` warning per task. This
supersedes any wording permitting multiple fallback predicates to be recorded, which was not
deterministic: `0/0` acceleration makes P3, P4 and P5 simultaneously true, and which of them
"holds" would otherwise depend on guard placement.

Independent non-fallback warnings established **before** the fallback — notably §8.2 bias-correction
clipping — remain recordable alongside it. P8 is evaluated **only** after both two-sided adjusted
probabilities are finite, so the one-sided task never evaluates P8.

**Not fallback conditions:** small acceleration · ties among draws · a skewed distribution · BCa and
percentile disagreeing · an inconvenient conclusion.

**Implication, not biconditional.** `acceleration_defined == false` **implies** the fallback method,
but the converse is **false**: P6, P7 and P8 fire with finite, well-defined acceleration. No validator
may assert the biconditional.

**Worked degenerate case — an all-zero paired vector** (A3a `degenerate_all_zero`, §4.2): every
replicate is `0.0`, so `p0 = 0.5` and `z0 = 0.0`; every leave-one-out value is `0.0`, so the centered
jackknife squared sum is exactly zero. **P3 is the first true predicate.** The acceleration
denominator and acceleration are therefore **never evaluated**, and P4 and P5 are **not** recorded.
The task records only `fallback_p3_zero_centered_jackknife_sum_squares`, uses the percentile fallback,
yields finite bounds `[0.0, 0.0]`, and carries task status `ok_with_warnings`.

---

## 9. Dataset-level bootstrap estimand and non-inferiority

### 9.1 Replicate rule `[ch3 + EVALUATION_CONTRACT §4.2]`

```
1. sample image IDs with replacement          4. re-accumulate per-class TP, GT, PRED
2. preserve duplicate multiplicity            5. derive union per class
3. same sampled positions for both stages     6. apply the frozen UNION-PRESENT rule
                                              7. dataset-level all-class mIoU per stage
                                              8. delta = candidate_mIoU - baseline_mIoU
```

**Averaging per-image mIoU is forbidden for this estimand** — it is a mean-of-ratios, whereas the
dataset-level metric is a ratio-of-sums. Classes absent from both GT and prediction within a
replicate are omitted by the union-present rule. A replicate with **zero** eligible classes is an
**integrity failure** for official analysis.

**Accumulation dtype `[project-decision, A3b-1]`.** All TP/GT/PRED accumulation, resampled
multiplicity summation and leave-one-out subtraction are **int64**; only the final division and macro
mean are float64. This makes "subtract the omitted image from the total" and "re-sum the remaining
images" **exactly** equivalent rather than approximately so. It matches the input contract already
enforced by `src/stats/ingest.py`, which rejects any non-int64 `tp`/`gt`/`pred` and returns the arrays
with `flags.writeable = False` — so any densified or reordered working representation must be an
**owned copy**, and in-place mutation of an ingested array is forbidden.

**Jackknife for this estimand `[project-decision, A3b-1]`.** §8.3's leave-one-**image**-out applies to
the pooled quantity by re-accumulation, not by scalar deletion. For each omitted image: remove its
TP/GT/PRED from **both** stages · re-accumulate the remaining class totals · recompute the union ·
recompute union-present eligibility **separately for each stage** · recompute both dataset-level
all-class mIoUs · return `candidate − baseline`. Removing one per-image mIoU scalar, or reusing
full-sample eligibility, is the same mean-of-ratios error this section forbids one level up.

**Jackknife for the scalar statistics.** Remove one paired difference and recompute the **complete**
statistic on the remainder. For `hodges_lehmann_shift` this means recomputing the exact Walsh-average
median of §7.3 — never an ordinary median, never the full-sample estimate reused.

### 9.2 Non-inferiority `[ch3]`

```
delta_NI = dataset_mIoU(E6) - dataset_mIoU(E3)            (all-class, union-present)
PASS  iff  one_sided_95pct_BCa_lower_bound  >  -0.020     (-2.0 percentage points)
```

**Boundary:** a lower bound **exactly equal** to `−0.020` **fails**; strict greater-than is required
`[project-decision]`.

Sensitivity decisions at **1.0 / 1.5 / 2.0 / 2.5 pp** are read from the **same** bootstrap
distribution — four unrelated distributions are **not** generated `[ch3 + project-decision]`.

This check is reported **separately** from the eight-test Holm family.

### 9.3 E6-KD contingency `[ch3]`

```
observed_drop = dataset_mIoU(E3) - dataset_mIoU(E6)       (clean test, all-class)
TRIGGER  iff  observed_drop  >  0.010                     (1.0 percentage point)
```

Equality at `0.010` does **not** trigger `[project-decision]`. **No p-value, no CI, not a Holm test,
not a non-inferiority conclusion, not a superiority conclusion.** ch3: *"distinct from, and stricter
than, the 2.0-point margin used for the formal E3-versus-E6 non-inferiority decision."*

Four concepts are kept strictly separate: observed-drop contingency · formal non-inferiority · clean
superiority · robustness superiority.

---

## 10. Official integrity policy

**Every input evaluation artifact for an official statistical result must have
`artifact_status = "official"`.** A `provisional` or `smoke` artifact may be used **only** for
explicitly non-official development or smoke analyses, and may never contribute to an official
eight-test family, an official non-inferiority result, or any official thesis conclusion.

Official clean-test comparisons additionally require: `split = test` · `actual_rows = expected_rows =`
**1,561** · exact canonical image-ID set equality · no duplicate IDs · no case-folded ID collision ·
no undefined primary per-image metric · **no random initialization** · verified artifact
`MANIFEST.sha256` · compatible `schema_version` and `metric_protocol` · identical
`split_manifest_sha256`, `class_map_sha256`, `preprocess_protocol`, class-space metadata, expected and
actual row counts, and condition identity · **`metric_impl_sha256` equality**.

For official analysis a **metric-implementation mismatch is an error, not a warning** — two stages
scored by different metric code are not comparable.

**Forbidden:** silent deletion · pairwise-complete fallback · mean/median/zero imputation ·
reordering without identity verification.

Non-official synthetic tests must use an **explicit policy mode** that cannot be mistaken for the
official default.

### 10.1 Software version policy `[implementation]`

Official statistics run under the **pinned** stack (`requirements.lock`: Python 3.11, numpy 1.26.4,
scipy 1.11.4, statsmodels 0.14.6). The current dev environment (Python 3.13.5, numpy 2.1.3, scipy
1.15.3, statsmodels 0.14.4) is permitted **only for contract smokes**. Every result records Python,
NumPy, SciPy and Statsmodels versions; **any mismatch from the pinned stack forces non-official
status**. No official p-value or confidence interval may be generated on the drifted dev stack.

Frozen compatibility requirements:
- Wilcoxon uses explicit `method="approx"`.
- Bootstrap count is always explicit.
- SciPy's RNG keyword differs across versions (`random_state` in 1.11.4, `rng` in 1.15.x). This must
  **not** be allowed to change the frozen random stream: the project uses **its own deterministic
  bootstrap engine** consuming the §8.1 `numpy.random.Generator` directly.
- `scipy.stats.bootstrap` may be used as a **smoke/reference cross-check only**, never as an
  uncontrolled source of version-dependent sampling.

---

## 11. Hodges-Lehmann feasibility `[empirical, measured 2026-07-28]`

Measured at n = 1,561 on the dev machine:

| Quantity | Value |
|---|---|
| Walsh averages (i ≤ j) | **1,219,141** |
| Memory: outer float64 / triu vector | 19.5 MB / **9.8 MB** |
| One exact HL — full sort | 67.0 ms |
| One exact HL — **partition-select** | **27.3 ms** (identical value) |
| B = 10,000 bootstrap | **≈ 4.6 min** per comparison |
| Leave-one-out jackknife (1,561) | **≈ 0.7 min** |
| **BCa per comparison** | **≈ 5.3 min** |
| Eight comparisons | **≈ 42 min** |

**There is no feasibility conflict.** The preregistered BCa Hodges-Lehmann interval is affordable
**exactly**, so it is **not** substituted with a percentile, median, normal, or absent interval.

---

## 12. Statistics result artifact `[implementation]`

`schema_version = "plantseg-stats/1.0.0"` · `statistical_protocol = "plantseg-stats-protocol/1.0.0"`

**Exactly one atomic run directory containing exactly THREE files:**

| File | Content |
|---|---|
| `family.json` | strict JSON — identity, per-comparison results, non-inferiority, provenance, integrity |
| `bootstrap.npz` | float64 replicate distributions + jackknife values + BCa parameters |
| `MANIFEST.sha256` | SHA-256 of the two files above; **does not hash itself** |

Writer discipline mirrors the proven A2a evaluator: refuse-existing overwrite · sibling temporary
directory · strict JSON with non-finite values forbidden · NPZ re-read validation · SHA-256
verification · atomic final rename · cleanup on failure.

### 12.1 `family.json`

Analysis and family identity · protocol and schema versions · official/non-official status ·
creation timestamp · α · expected comparison IDs · family completion status · pinned **and** observed
software versions · **statistical contract SHA-256** · full input provenance · per-comparison
deterministic results · Holm results · BCa summaries · descriptive **E1→E3** block · **E3→E6**
non-inferiority block · **E6-KD** trigger block · integrity results · warnings/status enums.

Per input artifact: stage · condition · repo-relative path · `artifact_status` · `checkpoint_sha256` ·
`split_manifest_sha256` · `class_map_sha256` · `metric_impl_sha256` · `config_sha256` · `repo_commit`.

Every undefined numeric field is **JSON `null` plus an explicit status**. Never `NaN`, `Infinity`,
numeric sentinels, or empty-string sentinels.

### 12.2 `bootstrap.npz` — raw replicate distributions ARE retained

For each bootstrapped statistic: **bootstrap replicate values float64 `[B]`** · **leave-one-out
jackknife values float64 `[n]`** · observed statistic · derived seed representation · `z0` ·
acceleration · requested interval type · **actual** interval method · confidence level · computed
bound(s).

Retained because they are tiny beside model artifacts, they avoid a ≈40-minute regeneration for
independent audit, and they permit direct verification of `z0`, percentile positions and every
sensitivity margin **without rerunning model inference**.

**Not stored:** sampled index matrices of shape `[B, n]` — reproducible from the frozen seed and
needlessly large.

NPZ keys are deterministic and documented, derived from `comparison_id` and `statistic_name`.

### 12.3 Task serialization and cross-representation integrity `[implementation, A3b-1 2026-07-29]`

#### 12.3.1 `task_id`

```
task_id = "<comparison_id>__<statistic_name>"
```

Exactly **two** components joined by the reserved `__` separator, which is unambiguous because
neither vocabulary permits `__` (§8.7). Each `task_id` corresponds to exactly one pair of the frozen
35-task matrix, is unique within one artifact, and never contains `dataset_miou_c_delta`.

`analysis_id` is deliberately **not** part of `task_id`. It is stored once at artifact level in
`family.json` and is verified per task through exact seed-byte reconstruction (§12.3.4).

#### 12.3.2 `bootstrap.npz` keys and dtypes

Every key is `<task_id>` + one suffix below. Examples:
`accuracy_e1_e2__mean_delta__bootstrap` · `descriptive_e1_e3__hodges_lehmann_shift__jackknife` ·
`noninferiority_e3_e6__dataset_miou_delta__bounds`.

| suffix | dtype | shape |
|---|---|---|
| `__bootstrap` | float64 | `[B]` |
| `__jackknife` | float64 | `[n]` |
| `__observed` | float64 | `[1]` |
| `__root_seed` | int64 | `[1]` |
| `__seed_digest` | uint8 | `[32]` |
| `__seed_entropy_bytes` | uint8 | `[16]` |
| `__z0` | float64 | `[1]` |
| `__acceleration` | float64 | `[1]` when defined, `[0]` when undefined |
| `__acceleration_defined` | bool | `[1]` |
| `__interval_type` | uint8 | `[len]`, UTF-8 bytes, decoded strictly |
| `__interval_method` | uint8 | `[len]`, UTF-8 bytes, decoded strictly |
| `__confidence_level` | float64 | `[1]` |
| `__bounds` | float64 | `[2]` two-sided, `[1]` one-sided lower |

The writer **asserts** `seed_entropy_bytes == seed_digest[:16]` rather than deriving the two
independently. **Never stored:** object arrays · pickle-dependent arrays · NaN or Infinity ·
`[B, n]` sampled-index matrices · any `int64` rendering of the 128-bit entropy integer. All NPZ reads
use `allow_pickle=False`. No key containing `dataset_miou_c_delta` is valid.

#### 12.3.3 `family.json` bootstrap-task record

One record per frozen task, in the §8.7.2 canonical order, with these fields:

`task_id` · `comparison_id` · `statistic_name` · `interval_type` · `confidence_level` ·
`bootstrap_replicates` · `jackknife_count` · `root_seed` · `seed_input_display` · `seed_input_hex` ·
`seed_sha256` · `seed_entropy_hex` · `seed_entropy_decimal` · `bit_generator` · `numpy_version` ·
`observed` · `observed_source` · `z0` · `acceleration` · `acceleration_defined` ·
`interval_method` · `warnings` · `status`, plus the interval bounds below.

- `seed_input_hex` is the **sole byte authority** — the exact canonical bytes of §8.1 in hex.
- `seed_input_display` is human-readable and **non-authoritative**, frozen as the exact
  concatenation
  `plantseg-stats/1.0.0\0root_seed=<decimal>\0<analysis_id>\0<comparison_id>\0<statistic_name>`
  where `\0` is the **literal two-character sequence** backslash + zero. No surrounding quotes, no
  spaces, no trailing `\0`, root seed in base-10 ASCII, identifiers exactly as frozen. The spellings
  `\x00`, `\u0000` and an actual NUL byte are **not** accepted for this field. Example:
  `plantseg-stats/1.0.0\0root_seed=42\0plantseg_primary_analysis_v1\0accuracy_e1_e2\0mean_delta`
- `seed_sha256` is 64 lowercase hex characters; `seed_entropy_hex` is the first 16 digest bytes as 32
  lowercase hex characters; `seed_entropy_decimal` is the full unsigned 128-bit integer as a
  **decimal string**.
- `observed`, `z0`, all bounds and any defined acceleration must be **finite**. Undefined acceleration
  is JSON `null` with `acceleration_defined = false` — never NaN.
- `bootstrap_replicates` equals the NPZ bootstrap-array length; `jackknife_count` equals the NPZ
  jackknife-array length and is the resampling *n* for that task.

**Interval bounds.** `two_sided` → both `lower_bound` and `upper_bound`. The sole one-sided task
`noninferiority_e3_e6__dataset_miou_delta` → `lower_bound` only, with `upper_bound` **omitted
entirely**: never `null`, never `Infinity`, and never a two-element bound merely because the statistic
is named `dataset_miou_delta`.

**Observed-value provenance — AMENDED A3b-2, 2026-07-30 (Amendment C).**

> **AMENDMENT C (not a clarification).** The **superseded** pre-A3b-2 wording was, verbatim:
>
> *"**Observed-value provenance.** For `mean_delta`, `median_delta` and `hodges_lehmann_shift` the
> observed statistic is **taken from** the typed A3a result and must match it **bit-for-bit** in
> float64; A3b recomputes only replicate and jackknife values. For `dataset_miou_delta` A3b computes
> the observed pooled value, because A3a produces no dataset-level pooled statistic. Across the 35
> tasks that is 27 inherited and 8 computed."*
>
> It is retained here only as quoted history and is **no longer active**. The defect: "27 inherited"
> presumes nine comparisons yield a typed `ComparisonResult`, but only the **eight canonical** ones
> can — `src/stats/tests.run_comparison` raises `StatsError` for any id outside
> `CANONICAL_COMPARISON_IDS`, and §8.7.1 forbids adding `descriptive_e1_e3` to that tuple. The three
> descriptive scalar tasks were therefore unsatisfiable. No A3b implementation and no official
> statistical artifact existed at amendment time, so **no published value changes**; a blocked path
> is made lawful. Decision record: `open_questions.md` D23.

Every bootstrap task carries a required `observed_source` naming exactly where its observed value
came from. The mapping is **24 / 3 / 8**:

| `observed_source` | tasks | count |
|---|---|---:|
| `a3a_comparison_result` | the eight canonical comparison IDs × `mean_delta`, `median_delta`, `hodges_lehmann_shift` | **24** |
| `a3a_primitives_descriptive` | `descriptive_e1_e3__mean_delta`, `descriptive_e1_e3__median_delta`, `descriptive_e1_e3__hodges_lehmann_shift` | **3** |
| `a3b_pooled_reaccumulation` | the seven clean `*__dataset_miou_delta` tasks + `noninferiority_e3_e6__dataset_miou_delta` | **8** |

**24 + 3 + 8 = 35.** The task matrix of §8.7.2 is unchanged.

**Source A.** Observed values are **taken from** the typed A3a `ComparisonResult` and must match it
**bit-for-bit** in float64 — `shifts.mean_delta`, `shifts.median_delta` and `hodges_lehmann_shift` of
the corresponding record in `a3a_family.comparisons` (§12.4). A3b recomputes only replicate and
jackknife values. A **complete** typed eight-comparison family is required before any Source-A task
may be serialized; a partial family fails closed (§6, §12.4).

**Source B.** Observed values are derived directly from the aligned E1/E3 clean `PairedVector` by
calling the existing **public A3a estimator primitives** `engineering_shifts` and `hodges_lehmann`,
with direction `E3 − E1` (`delta = candidate − baseline`, §1). The three values are packaged in an
immutable typed A3b-side carrier before serialization. That carrier is an **operational vehicle
only**: it confers no inferential status and does not make E1→E3 a member of the Holm family.

Source B prohibitions, all binding: do **not** add `descriptive_e1_e3` to
`CANONICAL_COMPARISON_IDS` · do **not** call `run_comparison` for E1→E3 · do **not** construct a
fake, surrogate or partially populated `ComparisonResult` · do **not** compute Wilcoxon, the paired
t-test, rank-biserial, Cohen's dz or any Holm member for E1→E3 · do **not** serialize inferential
placeholders, including nulls, in the descriptive block (§12.4).

**Source C.** A3b computes the observed pooled value through the frozen int64 sufficient-statistics
re-accumulation of §9.1, because A3a produces no dataset-level pooled statistic.

**`mean_delta_pp`** is supplied by `EngineeringShifts` as **descriptive reporting only**. It is
**not** a bootstrap statistic, **not** a 36th task, **not** an NPZ namespace and **not** a
bootstrap-task record. It may appear only inside `a3a_family.comparisons[].shifts` and the
`descriptive_e1_e3` block.

**Required bit-identity smoke `[implementation]`.** A3b's synthetic smoke must build one paired
vector under a **canonical** comparison ID, compute its typed `ComparisonResult` via
`run_comparison`, independently call `engineering_shifts` and `hodges_lehmann` on the same delta
vector, and assert **bit-for-bit float64 equality for all three** statistics — mean delta, median
delta and Hodges-Lehmann shift. This proves Source B uses the frozen A3a arithmetic rather than a
shadow estimator. Companion assertions: `run_comparison("descriptive_e1_e3", …)` still raises; the
direct descriptive path succeeds on the same data; it emits no Wilcoxon, t-test, Holm or
inferential-family record; and its three task records carry
`observed_source = "a3a_primitives_descriptive"`.

#### 12.3.4 Exact seed reconstruction

A containment or substring test against `analysis_id` is **not** sufficient. For every task the
canonical bytes are **rebuilt** from the artifact-level `analysis_id`, the task `comparison_id`, the
task `statistic_name`, the frozen `root_seed`, and the §8.1 rendering rule, and must equal the bytes
decoded from `seed_input_hex` **exactly**. Then, independently:

1. `SHA256(reconstructed_bytes).hexdigest() == seed_sha256`
2. decoded NPZ `__seed_digest` equals the complete 32-byte digest
3. `seed_entropy_hex == seed_sha256[:32]`
4. decoded NPZ `__seed_entropy_bytes == decoded_seed_digest[:16]`
5. `int(seed_entropy_hex, 16) == int(seed_entropy_decimal)`

Checks 1–5 are recomputable from the artifact and belong to the **verifier**. A sixth requirement —
that this same complete unsigned 128-bit integer is what was passed to `SeedSequence` — occurred at
construction time and is a **writer-side invariant**; confirming it post hoc would require
regenerating the draws, which is exactly what persisting them avoids. No verifier step may require
regeneration.

Forbidden in all of the above: substring checks · case-insensitive comparison · whitespace
normalisation · treating reconstructed display text as the byte authority.

A task whose seed bytes use a different analysis ID, root seed, comparison ID, statistic name, field
order, separator, encoding or trailing-NUL convention is an **integrity failure**. An artifact mixing
`plantseg_primary_analysis_v1` and `plantseg_stats_smoke_v1` across its tasks is an integrity failure,
and exact reconstruction is what proves its absence.

#### 12.3.5 Frozen vocabularies

**`interval_type`** — exactly `two_sided` · `one_sided_lower`.
**`interval_method`** — exactly `bca` · `percentile_fallback_degenerate_acceleration`, determined by
the actual outcome, never by comparison ID or statistic name. No aliases for either.

**`observed_source`** `[project-decision, A3b-2 2026-07-30]` — exactly `a3a_comparison_result` ·
`a3a_primitives_descriptive` · `a3b_pooled_reaccumulation`, mapped to tasks by the frozen 24/3/8
table of §12.3.3. Required on every task record. **Rejected:** a missing field · `null` · an unknown
value · a free-form variant or alias · a legal value attached to the wrong task · **any inference
from `statistic_name` alone** — `mean_delta` maps to two different sources depending on the
comparison, so the **complete task pair** controls the mapping. `observed_source` is
`family.json`-only and is **never** an NPZ key or suffix.

**Task `status`** — exactly `ok` · `ok_with_warnings`. `status == "ok"` **iff** `warnings` is empty;
`status == "ok_with_warnings"` **iff** `warnings` is non-empty. Because status is fully derivable, the
writer **asserts** this biconditional rather than setting the two fields independently. A fatal
integrity or numeric error prevents completion of the artifact and produces **no third status**;
fallback and saturation are *completed* results and use `ok_with_warnings`. No free-form explanatory
text may appear as a status.

**Task status is not A3a comparison status.** Bootstrap-task `status` describes **only** whether the
A3b interval task completed. It does not replace, reinterpret or overwrite the deterministic A3a
comparison status of §4 — an A3a comparison may be `degenerate_all_zero` while its A3b tasks are
validly `ok_with_warnings`, because a deterministic percentile fallback produced finite bounds. Both
remain visible in their own blocks; the A3a status is never copied into the task record. Task status
`ok` must **not** be read as evidence that the primary Wilcoxon comparison was ordinary,
non-degenerate, significant, or eligible for an official conclusion — official-family eligibility
remains governed by §§4, 6 and 10.

**`warnings`** — always a JSON array, never `null`; entries drawn only from the vocabulary below; no
duplicates; serialised in this canonical order:

| # | warning |
|---:|---|
| 1 | `bias_correction_probability_clipped_lower` |
| 2 | `bias_correction_probability_clipped_upper` |
| 3 | `adjusted_probability_saturated_lower` |
| 4 | `adjusted_probability_saturated_upper` |
| 5 | `fallback_p1_insufficient_jackknife` |
| 6 | `fallback_p2_nonfinite_jackknife` |
| 7 | `fallback_p3_zero_centered_jackknife_sum_squares` |
| 8 | `fallback_p4_invalid_acceleration_denominator` |
| 9 | `fallback_p5_nonfinite_acceleration` |
| 10 | `fallback_p6_invalid_bca_transform_denominator` |
| 11 | `fallback_p7_nonfinite_adjusted_probability` |
| 12 | `fallback_p8_nonincreasing_adjusted_probabilities` |

Per task there is **at most one** `fallback_p*` warning (§8.7.7 short-circuit), zero or more
independently applicable clipping warnings, and saturation warnings only when the BCa path completed
its adjusted-probability calculation. `interval_method == "percentile_fallback_degenerate_acceleration"`
requires **exactly one** `fallback_p*` warning. Longer explanations belong in the Markdown report,
never in `family.json`.

**One-sided structural invariants.** `noninferiority_e3_e6__dataset_miou_delta` has no upper tail and
never evaluates P8, so it may **never** carry `adjusted_probability_saturated_upper` or
`fallback_p8_nonincreasing_adjusted_probabilities`. Both are verifier assertions.

#### 12.3.6 JSON ↔ NPZ consistency

All JSON floats are written at **full round-trip float64 precision** — no display rounding, no fixed
decimal formatting (§7.4 already forbids display rounding in the artifact). Every comparison below is
**exact** float64 equality after reload. A tolerance-based or `isclose` verifier is **forbidden**,
because it would mask the divergence the check exists to detect.

For every task the writer and verifier prove: JSON `task_id` == NPZ key prefix · `observed` ==
`__observed` · `root_seed` == `__root_seed` · `seed_sha256` == `__seed_digest` · `seed_entropy_hex` ==
`__seed_entropy_bytes` · `z0` == `__z0` · acceleration state == the NPZ acceleration arrays, including
the `[0]`-length undefined encoding · `interval_type` and `interval_method` == the strictly decoded
NPZ byte arrays · `confidence_level` == `__confidence_level` · bounds == `__bounds` ·
`bootstrap_replicates` == `__bootstrap.shape[0]` · `jackknife_count` == `__jackknife.shape[0]` · and
the JSON task set == the exact frozen 35-pair matrix **in canonical order**.

`observed_source` is validated against the frozen 24/3/8 map of §12.3.3 rather than against the NPZ,
because it is categorical provenance with no NPZ counterpart. For the 24 Source-A tasks the verifier
additionally proves `observed` equals the corresponding field of the matching
`a3a_family.comparisons` record by **exact** float64 equality.

No partial, missing, duplicated or extra bootstrap-task record is valid. NPZ member ordering is not
semantic, but its complete key set must correspond exactly to the 35 ordered JSON records.

#### 12.3.7 Official counts

For an **official** protocol-1.0.0 artifact, **every one of the 35 tasks** must have
`bootstrap_replicates == 10000` (§8) and `jackknife_count == 1561` (§10) — without exception, and
explicitly including the per-image mIoU-C robustness comparison, the descriptive E1→E3 block and the
E3→E6 non-inferiority task, not only the clean superiority comparisons. The robustness task resamples
the same 1,561 canonical clean-image identities once the complete corruption grid has been aligned.

For a **non-official** synthetic artifact, `bootstrap_replicates` may be smaller and `jackknife_count`
may differ, but both must still equal their corresponding NPZ array lengths and the task's verified
paired-image count.

#### 12.3.8 Manifest semantics

`MANIFEST.sha256` lines are `<64 lowercase hex><two ASCII spaces><filename>\n`, UTF-8, LF only, in
lexicographic order — which yields `bootstrap.npz` then `family.json`, matching the proven A2a
convention in `src/eval/artifacts.py`. The manifest does not hash itself.

`np.savez` embeds archive metadata, so **byte-identical NPZ output across separate runs is not
required and must not be asserted**. `MANIFEST.sha256` is an **intra-run** integrity record: it proves
the two payloads in *this* directory are the ones that were validated. No test may require cross-run
equality of the NPZ or manifest bytes. Semantic reproducibility is guaranteed by the frozen seed
protocol, not by archive bytes.

### 12.4 Top-level `family.json` schema `[implementation, A3b-2 2026-07-30]`

§12.1 named the required content in prose but froze field spellings only for the per-input-artifact
provenance record. Everything else was **previously unfrozen implementation detail**, which an
implementer would otherwise have had to invent. It is frozen here. No A3b implementation and no
statistics artifact existed when these details were frozen, so nothing published changes. This
subsection **replaces no active rule** and is therefore *not* an amendment, with one narrow
exception noted in §12.4.3. Decision record: `open_questions.md` D24.

Serialization is strict UTF-8 JSON, LF newlines, `allow_nan=False`. Object key **order** is frozen
for deterministic serialization and review; it affects no statistic, bootstrap stream, interval or
decision, because §8.1 derives every task's `Generator` independently.

#### 12.4.1 The nineteen top-level keys, in canonical order

| # | key | type | notes |
|---:|---|---|---|
| 1 | `schema_version` | string | exactly `plantseg-stats/1.0.0` |
| 2 | `statistical_protocol` | string | exactly `plantseg-stats-protocol/1.0.0` |
| 3 | `family_id` | string | exactly `plantseg_eight_test_holm_v1` |
| 4 | `analysis_id` | string | §8.7.1 vocabulary; the RNG namespace |
| 5 | `run_id` | string | §12.4.2 |
| 6 | `created_at_utc` | string | §12.4.2 |
| 7 | `artifact_status` | string | §12.4.3, derived |
| 8 | `warnings` | array[string] | §12.4.3, drives key 7 |
| 9 | `alpha` | float | exactly `0.05` |
| 10 | `expected_comparison_ids` | array[string] | the eight canonical IDs, §1 order |
| 11 | `statistical_contract_sha256` | string | §12.4.4 |
| 12 | `software_environment` | object | §12.4.5 |
| 13 | `input_artifacts` | array[object] | §12.4.6 |
| 14 | `a3a_family` | object | §12.4.7 |
| 15 | `descriptive_e1_e3` | object | §12.4.8 |
| 16 | `noninferiority_e3_e6` | object | §12.4.9 |
| 17 | `e6_kd_trigger` | object | §12.4.10 |
| 18 | `bootstrap_tasks` | array[object] | the ordered 35 records of §12.3 |
| 19 | `integrity` | object | §12.4.11 |

The key set and order are exact. **No additional top-level key is permitted under protocol 1.0.0.**
Neither version literal is renamed or version-bumped by A3b-1, A3b-2 or any 1.0.0 revision.

`family_id` identifies the canonical eight-member Holm family. `analysis_id` identifies the broader
analysis and continues to govern RNG namespaces (§8.7.1); exactly one `analysis_id` governs one
artifact, and a mixed-namespace artifact is an integrity failure (§12.3.4).

#### 12.4.2 `run_id` and `created_at_utc`

`run_id` is required, equals the final statistics run-directory basename (the writer asserts
equality), has **no** effect on any RNG stream, and must match `^[a-z0-9][a-z0-9._-]{0,63}$`. Leading
or trailing whitespace, path separators and embedded NUL are forbidden. The lowercase-only pattern
also removes case-folding collisions on case-insensitive filesystems.

`created_at_utc` is `YYYY-MM-DDTHH:MM:SSZ`, produced with `%Y-%m-%dT%H:%M:%SZ`: timezone-aware UTC,
seconds precision, no fractional seconds, literal `Z`, no local offset, no naive datetime. This is
the same convention as the proven A2a writer in `src/eval/artifacts.py`.

#### 12.4.3 `artifact_status` and top-level `warnings`

Top-level `artifact_status` takes exactly `official` · `nonofficial`.

**The field name `artifact_status` is deliberately reused at two nesting levels, and the two value
vocabularies do NOT merge.** This is the one place §12.4 adds a vocabulary to an existing field name;
validation must be **nesting-aware and never generic on the key name**:

| Level | Field | Legal values | Governing source |
|---|---|---|---|
| top-level statistics artifact | `artifact_status` | `official`, `nonofficial` | §12.4 |
| each `input_artifacts[]` record | `artifact_status` | `official`, `provisional`, `smoke` | `EVALUATION_CONTRACT.md` |
| ingest execution policy | `Policy` | `official`, `nonofficial_smoke` | `src/stats/ingest.py` |

`nonofficial` ≠ `nonofficial_smoke` ≠ `smoke`. A validator that checks `artifact_status` generically
produces false rejections in both directions.

Top-level `warnings` is always a JSON array, **never `null`**, without duplicates, drawn only from
this vocabulary and serialized in this canonical order:

| # | warning |
|---:|---|
| 1 | `analysis_id_nonofficial` |
| 2 | `software_environment_unpinned` |
| 3 | `input_artifact_nonofficial` |
| 4 | `synthetic_input_data` |
| 5 | `bootstrap_replicates_nonproduction` |
| 6 | `jackknife_count_nonproduction` |

`artifact_status == "official"` **iff** `warnings` is empty **and** every official requirement of §10
passes; `artifact_status == "nonofficial"` **iff** `warnings` is non-empty. Status is **derived** from
warnings plus the officiality checks and asserted, never set independently.

**Exhaustiveness `[project-decision]`.** These six strings are **exhaustive over all non-fatal
conditions that prevent official status.** Every such condition either maps to one or more of them,
or is a **fatal** error that prevents artifact creation. **There is no third path**, and the writer
asserts this: each officiality check either emits a frozen warning or raises. Without this clause the
biconditional above is unsound, because a non-fatal defect outside the vocabulary would yield empty
warnings and a spurious `official`.

**Fatal boundary.** Identity, schema, mathematical and integrity failures — including a
`statistical_contract_sha256` mismatch, a zero or negative retention denominator (§12.4.9), and any
unestablished `integrity` check (§12.4.11) — remove the temporary directory and create **no**
artifact. They are never downgraded into a nonofficial warning. Task-level numerical warnings
(bias-correction clipping, saturation, percentile fallback; §12.3.5) are **never** copied to top level
and never make an otherwise valid artifact nonofficial.

#### 12.4.4 `alpha`, `expected_comparison_ids`, `statistical_contract_sha256`

`alpha` is exactly `0.05` (§§3, 6). `expected_comparison_ids` is the eight-member canonical tuple in
its §1 order; the implementation **imports `CANONICAL_COMPARISON_IDS` from `src/stats/tests.py`** and
asserts it equals the documented order. A second independently typed authoritative eight-ID tuple
must **not** exist in A3b source.

`statistical_contract_sha256` is the SHA-256 of the **exact bytes** of this file as used by the run:
64 lowercase hexadecimal characters, no whitespace, hashed from the raw bytes and **never** from a
normalized, re-encoded or line-ending-converted representation. It is recomputed and compared
independently before finalization, and a mismatch is an **integrity failure, not a warning**.

**This contract does not contain its own active expected hash literal**, because any such literal
would be self-invalidating the moment the contract is edited. Historical digests may appear only in
explicit forensic history, never as the active expected value. The run computes the hash from the
final checkpointed contract bytes.

#### 12.4.5 `software_environment`

Exactly three keys in order: `pinned`, `observed`, `matches_pinned`. `pinned` and `observed` each
hold exactly four keys in order — `python`, `numpy`, `scipy`, `statsmodels`. Pinned values are those
of §10.1 (Python 3.11, numpy 1.26.4, scipy 1.11.4, statsmodels 0.14.6); the three library pins are
also `requirements.lock` lines, whereas the Python version is asserted by §10.1 itself.

`matches_pinned` is a boolean that is **derived and asserted**, never set independently.
`matches_pinned == false` **must** imply `software_environment_unpinned ∈ warnings`, which in turn
implies `artifact_status == "nonofficial"`.

#### 12.4.6 `input_artifacts`

An array of unique provenance records, each with exactly these ten fields in this order:

| # | field | # | field |
|---:|---|---:|---|
| 1 | `stage` | 6 | `split_manifest_sha256` |
| 2 | `condition` | 7 | `class_map_sha256` |
| 3 | `repo_relative_path` | 8 | `metric_impl_sha256` |
| 4 | `artifact_status` | 9 | `config_sha256` |
| 5 | `checkpoint_sha256` | 10 | `repo_commit` |

Field semantics are those already frozen in §12.1, §10 and the evaluation contract. Paths are
repository-relative **POSIX-style**; absolute paths and backslashes are forbidden. Records are unique
by `repo_relative_path`, and duplicates are an integrity failure. Every digest is 64 lowercase
hexadecimal characters. Complete evaluation artifacts are **never** embedded in `family.json`.

**Ordering:** ascending lexicographic order by `repo_relative_path` — and by nothing else. Stage or
condition ordering must not be used; the sort key is reproducible from the record itself and needs no
auxiliary field.

**Official expected set: exactly 37 unique records** `[project-decision]` — seven clean stage
artifacts E1–E7, fifteen E1 corruption artifacts, and fifteen E6 corruption artifacts (5 corruptions
× the 3 inferential severities of §2, per stage). This follows from the one-artifact-per-
stage-condition model: `EVALUATION_CONTRACT.md` gives each artifact a single
`condition.type`/`condition.name`/`condition.severity` triple, and `src/stats/robustness.py` requires
exactly fifteen **separate** evaluation runs per stage to assemble one mIoU-C vector. No artifact
packages multiple conditions.

An **official** artifact requires every input record's `artifact_status` to be `official`. A valid
nonofficial artifact containing one or more non-official inputs records the single top-level warning
`input_artifact_nonofficial`.

#### 12.4.7 `a3a_family` — the A+ envelope

Exactly three keys in this order:

| # | key | content |
|---:|---|---|
| 1 | `completion_status` | exactly the string `complete` |
| 2 | `comparisons` | exactly eight serialized `ComparisonResult` records, in `expected_comparison_ids` order |
| 3 | `holm` | the **complete finalized `HolmFamily`** carrier |

The envelope is not itself a source dataclass; its frozen order is
`("completion_status", "comparisons", "holm")`. Incomplete or partial family state prevents
finalization: `incomplete`, `partial`, `failed` and `null` are **never** serialized, and a partial
A3a family fails closed and **cannot** be downgraded to a nonofficial completed artifact.

`holm` serializes the finalized `HolmFamily` dataclass itself, so its key order is that dataclass's
declaration order — `("members", "alpha", "complete", "status")`. It is **not** reduced, reordered, or
replaced by an adapter: `complete` is retained rather than folded into `completion_status`, and
`alpha` is retained rather than deferred to the top-level key.

**Frozen finalized values:** `holm.alpha = 0.05` · `holm.complete = true` · `holm.status = "ok"`.

**`"ok"` is the only legal finalized serialized `HolmFamily` status** `[implementation]`. Source:
`src/stats/tests.holm_family` has exactly one return, constructing the family with `complete=True`
and `status=STATUS_OK` (`STATUS_OK = "ok"`); every other path **raises `StatsError`** rather than
returning a differently-statused family. The `HolmFamily` constructor is public, so a caller could
hand-build an object with another status or `complete=False`; such an object is **transient and
unsanctioned**, and is not legal in a finalized `family.json`. The distinction is frozen:

| Level | Permitted |
|---|---|
| source dataclass, transient | any string in `status`, any boolean in `complete` — reachable only by direct construction, never by `holm_family` |
| finalized `family.json` | `status` exactly `"ok"`; `complete` exactly `true` |

**Why the apparent duplication is intentional.** Top-level `alpha` is the protocol-level family
alpha; `a3a_family.holm.alpha` is the alpha carried by the finalized typed result;
`a3a_family.completion_status` is the artifact-envelope summary; `a3a_family.holm.complete` and
`holm.status` are the typed carrier's own fields. None may be removed merely because it is related to
another. Their required agreement is precisely what lets a verifier detect a writer that has combined
incompatible objects or serialized a stale `HolmFamily`.

##### 12.4.7.1 Frozen A3a serialization schemas

The following field tuples are frozen **literally**, in the exact current declaration order of
`src/stats/tests.py`. After this documentation task **the contract, not the source, is the protocol
authority**.

`ComparisonResult` — 16 fields:

| # | field | JSON type | # | field | JSON type |
|---:|---|---|---:|---|---|
| 1 | `comparison_id` | string | 9 | `wilcoxon` | object |
| 2 | `baseline_stage` | string | 10 | `t_test` | object |
| 3 | `candidate_stage` | string | 11 | `rank_biserial` | object |
| 4 | `metric` | string | 12 | `cohens_dz` | object |
| 5 | `direction` | string | 13 | `hodges_lehmann_shift` | float |
| 6 | `n_paired` | integer | 14 | `shifts` | object |
| 7 | `policy` | string | 15 | `warnings` | array[string] |
| 8 | `alignment_status` | string | 16 | `status` | string |

`WilcoxonResult` — 12 fields: `statistic` (float\|null) · `zstatistic` (float\|null) · `p_value`
(float\|null) · `status` (string) · `n_total` (integer) · `n_zero` (integer) · `n_nonzero` (integer) ·
`warnings` (array[string]) · `zero_method` (string) · `correction` (boolean) · `alternative` (string) ·
`method` (string).

`TTestResult` — 9 fields: `statistic` (float\|null) · `p_value` (float\|null) · `df` (integer\|null) ·
`mean_difference` (float) · `sd_difference` (float\|null) · `standard_error` (float\|null) · `status`
(string) · `warnings` (array[string]) · `alternative` (string).

`RankBiserialResult` — 8 fields: `value` (float\|null) · `r_plus` (float) · `r_minus` (float) ·
`denominator` (float) · `n_zero` (integer) · `n_tied_groups` (integer) · `status` (string) ·
`sign_convention` (string).

`CohensDzResult` — 4 fields: `value` (float\|null) · `mean` (float) · `sd` (float) · `status` (string).

`EngineeringShifts` — 3 fields: `mean_delta` (float) · `mean_delta_pp` (float) · `median_delta`
(float).

`HolmMember` — 9 fields: `comparison_id` (string) · `p_raw` (float) · `p_adjusted` (float) ·
`sorted_rank` (integer) · `step_denominator` (integer) · `step_threshold` (float) · `reject`
(boolean) · `library_reject` (boolean) · `boundary_note` (string\|null).

`HolmFamily` — 4 fields: `members` (array[object]) · `alpha` (float) · `complete` (boolean) ·
`status` (string).

##### 12.4.7.2 Serializer rule and runtime source-schema assertion

Each frozen dataclass is serialized as an **ordered JSON object** built from the contract's explicit
field whitelist and order. Nested result dataclasses become nested ordered objects. Tuple and list
fields become JSON arrays **without reordering**. Booleans stay booleans; integers stay integers;
finite float64 values are written at round-trip precision; strings use their frozen vocabularies.
Undefined values follow the already governing A3a rules — JSON `null` beside the sibling `status`
that explains it (§§4, 12.1). No NaN or Infinity. **No additional dataclass field may silently appear
in `family.json`.**

Three specific rules, each of which a naive serializer gets wrong:

- **`ComparisonResult.policy` is a `Policy` enum, not a string.** It is serialized through `.value`.
- **`HolmMember.p_adjusted` must be finite.** `holm_audit` constructs intermediate members with a
  NaN `p_adjusted` which `holm_family` replaces; serializing an intermediate member is a **fatal**
  error. Only a finalized `HolmFamily` may be serialized.
- **`HolmMember.boundary_note = null` is legitimate** and means no boundary condition arose. It is a
  string field, so §12.1's "undefined numeric ⇒ null plus status" does not apply to it.

At runtime, for each of the eight frozen classes, the writer obtains `dataclasses.fields(cls)`, forms
the tuple of `.name` values in order, and asserts it **exactly equals** the contract tuple — for
example `("members", "alpha", "complete", "status")` for `HolmFamily`. A missing, additional, renamed
or **reordered** source field **fails closed** before anything is written.
**`dataclasses.asdict()` is never the schema authority**; it may be used internally only after the
field-set and order assertion passes, and the final ordered mapping is still constructed explicitly.
A future source-code schema change requires a contract review, a protocol-version decision and
updated smoke coverage; it must **not** silently alter protocol 1.0.0.

##### 12.4.7.3 A+ cross-field invariants

All comparisons are **exact** float64 equality after JSON reload. `isclose` and tolerance-based
verifiers are **forbidden**.

1. top-level `alpha == 0.05`
2. `a3a_family.holm.alpha == 0.05`
3. top-level `alpha` equals `a3a_family.holm.alpha` exactly
4. `a3a_family.completion_status == "complete"`
5. `a3a_family.holm.complete` is `true`
6. `completion_status == "complete"` **iff** `holm.complete` is `true`
7. `a3a_family.comparisons` contains exactly eight records
8. `a3a_family.holm.members` contains exactly eight records
9. both arrays are in `expected_comparison_ids` order
10. the comparison ID at each position is identical across `expected_comparison_ids`,
    `a3a_family.comparisons` and `a3a_family.holm.members`
11. each Holm member's `p_raw` equals the corresponding comparison's finalized
    `wilcoxon.p_value` exactly
12. only a finalized `HolmFamily` may be serialized
13. every `p_adjusted` is finite
14. an intermediate `HolmMember` carrying NaN is a **fatal** error
15. partial or incomplete family state can never be emitted, including as nonofficial output

Additionally: **no `descriptive_e1_e3` record appears in either A3a array**, and **no bootstrap field
appears inside any A3a comparison or Holm record**. Invariants 9–11 hold by construction —
`holm_family` orders results by `CANONICAL_COMPARISON_IDS.index` before building members, and each
`p_raw` derives from `ComparisonResult.wilcoxon.p_value` — but they are verified rather than assumed.

The 24 Source-A bootstrap observed values are verified bit-for-bit against `shifts.mean_delta`,
`shifts.median_delta` and `hodges_lehmann_shift` of these eight records.

#### 12.4.8 `descriptive_e1_e3`

Exactly twelve fields in this order:

| # | field | # | field |
|---:|---|---:|---|
| 1 | `comparison_id` | 7 | `policy` |
| 2 | `baseline_stage` | 8 | `alignment_status` |
| 3 | `candidate_stage` | 9 | `mean_delta` |
| 4 | `direction` | 10 | `mean_delta_pp` |
| 5 | `metric` | 11 | `median_delta` |
| 6 | `n_paired` | 12 | `hodges_lehmann_shift` |

Frozen values: `comparison_id = "descriptive_e1_e3"` · `baseline_stage = "E1"` ·
`candidate_stage = "E3"` · `direction = "candidate_minus_baseline"`. `metric`, `policy` and
`alignment_status` reuse the already frozen A3a vocabularies; no aliases are created.

All numeric values are finite. The block is produced through `engineering_shifts` and
`hodges_lehmann` from the immutable typed descriptive carrier (§12.3.3, Source B), never through
`run_comparison`. It contains **no** Wilcoxon, t-test, Holm, rank-biserial, Cohen's dz or p-value
field, and **no null placeholders** for them — their absence is structural, exactly as
`ComparisonResult` carries no bootstrap field. Its three bootstrap tasks carry
`observed_source = "a3a_primitives_descriptive"`.

#### 12.4.9 `noninferiority_e3_e6` and relative retention

Exactly fifteen fields in this order:

| # | field | # | field |
|---:|---|---:|---|
| 1 | `comparison_id` | 9 | `relative_retention` |
| 2 | `baseline_stage` | 10 | `interval_type` |
| 3 | `candidate_stage` | 11 | `confidence_level` |
| 4 | `direction` | 12 | `lower_bound` |
| 5 | `baseline_dataset_miou` | 13 | `primary_margin` |
| 6 | `candidate_dataset_miou` | 14 | `passed` |
| 7 | `observed_delta` | 15 | `sensitivity` |
| 8 | `observed_drop` | | |

Frozen values: `comparison_id = "noninferiority_e3_e6"` · `baseline_stage = "E3"` ·
`candidate_stage = "E6"` · `direction = "candidate_minus_baseline"` ·
`interval_type = "one_sided_lower"` · `confidence_level = 0.95` · `primary_margin = 0.020`.

**Relative retention `[project-decision, A3b-2 2026-07-30]`.** Chapter III requires reporting the
actual E3→E6 drop, the relative retention of E3 mIoU, and the sensitivity conclusions; it froze no
formula. Frozen now, over the same clean dataset-level all-class union-present pooled mIoU values
already required by the §9.1 estimand:

```
relative_retention = candidate_dataset_miou / baseline_dataset_miou
                   = dataset_mIoU(E6) / dataset_mIoU(E3)
```

A dimensionless float64 ratio. It is **not** multiplied by 100 in `family.json`: `0.98` means 98 %
retention, `1.00` parity, and a value **greater than 1 is valid** and means E6 exceeded E3.
Percentage formatting such as "98 %" belongs only to reports and presentation layers.

Validity: `baseline_dataset_miou` must be finite and **strictly greater than zero**;
`candidate_dataset_miou` and `relative_retention` must be finite. A zero or negative denominator is a
**fatal mathematical error in every policy mode** — never `null`, NaN, Infinity or a placeholder
ratio.

Relative retention is **descriptive only**. It does not affect the BCa lower bound, the primary
non-inferiority decision, the sensitivity-margin decisions, E6-KD triggering, Holm correction, or any
bootstrap stream.

Exact identities, all verified by **exact** float64 equality after JSON reload — never `isclose`:

```
observed_delta     = candidate_dataset_miou - baseline_dataset_miou
observed_drop      = baseline_dataset_miou - candidate_dataset_miou
observed_drop      = -observed_delta
relative_retention = candidate_dataset_miou / baseline_dataset_miou
```

These identities are asserted against **recomputed expressions**, not against decimal literals: with
`baseline = 0.5` and `candidate = 0.4`, float64 gives `0.4 - 0.5 = -0.09999999999999998`, so an
assertion against the literal `-0.10` would wrongly fail. The frozen synthetic vector uses
**binary-exact** values: `baseline_dataset_miou = 0.5`, `candidate_dataset_miou = 0.25`,
`observed_delta = -0.25`, `observed_drop = 0.25`, `relative_retention = 0.5`.

`passed` **iff** `lower_bound > -primary_margin`; equality **fails** (§9.2). All numeric values are
finite.

`sensitivity` is an ordered array of exactly four objects, each with keys `margin` then `passed`, at
margins `0.010`, `0.015`, `0.020`, `0.025` in that order. Every conclusion is read from the **same**
stored bootstrap distribution and the **same** stored one-sided lower bound; no new distribution or
interval is generated per margin.

**Cross-block consistency.** `observed_delta` must equal the observed value of the
`noninferiority_e3_e6__dataset_miou_delta` task in both `family.json` and `bootstrap.npz`;
`e6_kd_trigger.observed_drop` must equal `noninferiority_e3_e6.observed_drop` exactly; and the pooled
E3 and E6 values used here must be the same values from which both decisions are derived.

#### 12.4.10 `e6_kd_trigger`

Exactly six fields in this order: `baseline_stage` · `candidate_stage` · `observed_drop` ·
`trigger_threshold` · `decision_rule` · `triggered`.

Frozen values: `baseline_stage = "E3"` · `candidate_stage = "E6"` · `trigger_threshold = 0.010` ·
`decision_rule = "strictly_greater"`. `observed_drop = dataset_mIoU(E3) - dataset_mIoU(E6)`;
`triggered` **iff** `observed_drop > trigger_threshold`; equality does **not** trigger (§9.3). The
block carries no p-value, no confidence interval, no Holm field and no non-inferiority conclusion.

#### 12.4.11 `integrity`

Exactly two keys in order: `status`, then `checks`. The only valid finalized status is `passed`;
`failed`, `partial`, `null` and free-form strings are **never** serialized. `checks` holds exactly
these thirteen keys, in this order, every value JSON `true`:

| # | check | # | check |
|---:|---|---:|---|
| 1 | `input_artifact_provenance_verified` | 8 | `json_npz_exact_match` |
| 2 | `statistical_contract_hash_verified` | 9 | `npz_schema_verified` |
| 3 | `a3a_family_complete` | 10 | `finite_values_verified` |
| 4 | `bootstrap_task_matrix_complete` | 11 | `officiality_policy_verified` |
| 5 | `bootstrap_task_order_canonical` | 12 | `exact_file_set_verified` |
| 6 | `observed_source_mapping_verified` | 13 | `manifest_verified` |
| 7 | `seed_material_verified` | | |

Missing, `false`, additional, reordered and unknown checks are all invalid. **No new check key is
added for A+**: `a3a_family_complete` is defined to require successful verification of the **entire**
A+ family structure — the §12.4.7.2 source dataclass field-tuple assertions, the exact ordered
comparison and Holm schemas, the eight/eight record counts, canonical positional ID alignment, a
finalized `HolmFamily` with `status == "ok"` and `complete` `true`, alpha equality,
completion-status consistency, finite `p_adjusted`, exact `p_raw` ↔ `wilcoxon.p_value` equality, and
the absence of any descriptive or bootstrap record inside the A3a family. Failure of **any** of those
conditions leaves `a3a_family_complete` unestablished and prevents final-directory creation; it is
never weakened into a top-level officiality warning.

**Writer sequencing.** The writer independently completes and verifies all thirteen checks inside the
sibling temporary directory; the block is then serialized as the required final assertion; the
directory may be atomically renamed **only** afterwards. If any check cannot be established, the
temporary directory is removed and **no final artifact exists**. `MANIFEST.sha256` remains the
cryptographic integrity source, does not hash itself, and `family.json` contains **no** self-hash and
no manifest hash.

#### 12.4.12 Canonical ordering and strict validation

Frozen: the exact top-level key set and order; the exact key set and order of every nested block; and
the exact array order of top-level `warnings`, `expected_comparison_ids`, `input_artifacts`,
`a3a_family.comparisons`, `a3a_family.holm.members`, `noninferiority_e3_e6.sensitivity`,
`bootstrap_tasks` and `integrity.checks`.

The verifier uses an **order-preserving** JSON parser and rejects missing keys · extra keys ·
duplicate keys · wrong ordering · wrong nesting · `null` substitutes · aliases. Ordering is a
deterministic serialization and review rule only; it affects no statistic, bootstrap stream, interval
or decision.

---

## 13. What A3a and A3b must not re-decide

Frozen here: the eight-comparison family and its exclusions · delta direction · the nested mIoU-C
rule and its 15-cell grid · the Wilcoxon configuration including `method="approx"` · the tie policy
(ties are **valid**) · every degenerate status · the t-test's sensitivity role · Holm membership and
the strict boundary · the rank-biserial denominator `R₊ + R₋` · dz · exact Hodges-Lehmann · **the
amended §8.1 seed derivation, in which `root_seed` is rendered into the canonical bytes** · `z0` tie
handling · jackknife acceleration · linear quantiles · percentile-only fallback · one-sided vs
two-sided interval types **under the amended §8.5 field name `interval_type`** · the non-inferiority
margin and its strict boundary · the E6-KD trigger · the official-input requirement · and the
three-file artifact.

Frozen by A3b-1 (2026-07-29) and equally not re-decidable: the §8.7.1 analysis / comparison /
statistic namespaces · the §8.7.2 35-task matrix and its canonical order · interval type as a
property of the task rather than the statistic (§8.7.3) · the §8.7.4 row-wise draw call and per-task
stream ownership · the §8.7.5 adjusted-probability transform and nominal tails · §8.7.6 saturation
handling · the §8.7.7 **first-trigger** P1–P8 fallback ladder · the §9.1 int64 accumulation rule and
pooled jackknife mechanics · and the whole of §12.3 — `task_id`, NPZ and `family.json` schemas, exact
seed reconstruction, the interval / method / status / warning vocabularies, exact JSON↔NPZ equality,
official counts, and intra-run manifest semantics.

Frozen by A3b-2 (2026-07-30) and equally not re-decidable: the **24 / 3 / 8** observed-value
provenance mapping and the `observed_source` vocabulary (§12.3.3, §12.3.5) · the whole of **§12.4** —
the nineteen top-level keys and their order, `family_id`, the `run_id` pattern, the timestamp format,
the level-scoped `artifact_status` vocabularies, the exhaustive six-string officiality-warning
vocabulary, the `software_environment` shape, the 37-record official input set and its
`repo_relative_path` ordering, the A+ `a3a_family` envelope with the complete finalized `HolmFamily`
carrier, the eight literal A3a/Holm dataclass field tuples and the serializer plus runtime
`dataclasses.fields()` assertion, the descriptive / non-inferiority / E6-KD block schemas, the
`relative_retention` ratio and its fatal denominator policy, the integrity block with its thirteen
mandatory checks, and the canonical ordering and strict-validation rules.

Three rules were **amended** rather than merely extended: the §8.1 seed bytes (**Amendment A**,
A3b-1), the §8.5 interval-type field name (**Amendment B**, A3b-1), and the §12.3.3 observed-value
provenance rule (**Amendment C**, A3b-2). Each is recorded with its superseded wording quoted verbatim
— in §8.1, §8.5, §12.3.3 and `open_questions.md` D21 and D23. None may be silently reverted.

§12.4 is a **newly frozen** schema, not an amendment: it replaces no active rule, with the single
narrow exception of adding a level-scoped top-level vocabulary to the existing `artifact_status` field
name (§12.4.3). Decision record: D24.

A pooled dataset-level **robustness** estimand (`dataset_miou_c_delta`) is **deferred**, not omitted —
see D22. It must not be introduced under protocol 1.0.0.

**A3a** implements §§1–7 and §10 ingestion/alignment. **A3b** implements §§8–9, §8.7, §12.2, §12.3 and
§12.4.
