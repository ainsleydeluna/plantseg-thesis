# A3a — Deterministic paired inference: ingestion, alignment, mIoU-C, Wilcoxon, Holm, effect sizes

**RESULT: ✅ A3a COMPLETE — PAIRED INFERENCE CORE PROVEN** · `45/45 checks passed`, exit code **0**

> **NONOFFICIAL.** The dev stack (Python 3.13.5 / numpy 2.1.3 / scipy 1.15.3 / statsmodels 0.14.4)
> differs from the pinned statistics stack (3.11 / 1.26.4 / **1.11.4** / **0.14.6**), so per
> `STATISTICAL_ANALYSIS_CONTRACT.md` §10.1 nothing here may be presented as a thesis result. No
> dataset, checkpoint, model, GPU, or bootstrap was involved.

Implements `STATISTICAL_ANALYSIS_CONTRACT.md` §§1–7 and §10. _Generated 2026-07-28 · CPU only._

---

## ✅ SUPERSEDED · 🛑 Blocking gap for official mIoU-C: the corruption vocabulary is not frozen

> ### ✅ SUPERSEDED — resolved 2026-07-28 by A3b-0. This section is a historical A3a snapshot.
>
> **Everything below records the state at the time A3a ran and is no longer the state of the
> repository.** The corruption vocabulary was frozen immediately afterwards, and the blocker this
> section raised is closed:
>
> - `configs/corruption_protocol.json` (`plantseg-corruptions/1.0.0`) is now the machine-readable
>   source of truth, loaded and fully validated by `src/stats/corruption_protocol.py`.
> - `STATISTICAL_ANALYSIS_CONTRACT.md` §2.1, `EVALUATION_CONTRACT.md` §5.1.1,
>   `IMPLEMENTATION_CONTRACT.md` (Robustness) and `open_questions.md` **D20** all freeze the five
>   identifiers and the severity roles, and D20 records that **official mIoU-C is no longer blocked
>   by vocabulary ambiguity**.
> - The `brightness` vs `brightness_variation` question the section below asks was decided in favour
>   of `brightness`; the prose form is a display label only.
>
> **Current authority is the contract and the protocol file, never this A3a snapshot.** See
> [`a3b0_corruption_vocabulary_smoke.md`](a3b0_corruption_vocabulary_smoke.md). The A3a text is
> retained unedited as historical evidence; its measurements and results below remain valid.

`STATISTICAL_ANALYSIS_CONTRACT.md` §2, `IMPLEMENTATION_CONTRACT.md` §(f) and Chapter III all name
the five corruptions in **prose only** — *"motion blur"*, *"Gaussian noise"*, *"JPEG compression"*,
*"brightness"* (elsewhere *"brightness variation"*), *"fog"* — and `EVALUATION_CONTRACT.md` types
`condition.name` as an unconstrained `"string | null"`. **No document freezes a machine-readable
identifier.** `motion_blur` vs `motion-blur` vs `motion blur`, or `brightness` vs
`brightness_variation`, are all equally consistent with the prose.

Per the A3a instruction (*"stop before inventing slugs and report the documentation gap"*),
`src/stats/robustness.py` **hard-codes no canonical slug**. It requires an injected
`CorruptionGrid` with **no default**, and the smoke supplies an explicitly synthetic vocabulary
(`synA…synE`). The nested aggregation, the 15-cell grid validation and every rejection path are
fully implemented and proven — only the official vocabulary is missing.

**Required before official mIoU-C:** freeze the five identifiers in
`STATISTICAL_ANALYSIS_CONTRACT.md` §2, ideally matching whatever the vendored `imagecorruptions`
reference implementation emits, and resolve `brightness` vs `brightness_variation`.

---

## 1. Preflight

`master` · HEAD `32d52f4` · 0 ahead/behind. Regressions before editing: A1b **21/21** · A2a
**63/63** · A2b-0 **21/21** · A2b **53/53**. `src/stats/__init__.py` was 0 bytes; no A3a file
existed; `STATISTICAL_ANALYSIS_CONTRACT.md` present (358 lines).

## 2. Module and API design

| Module | Role |
|---|---|
| `src/stats/ingest.py` | read-only artifact ingestion, manifest verification, `Policy` enum, immutable records |
| `src/stats/align.py` | compatibility + canonical identity alignment → `PairedVector` |
| `src/stats/robustness.py` | 15-cell corruption grid → nested per-image mIoU-C |
| `src/stats/tests.py` | Wilcoxon, paired t-test, Holm, point effect sizes, `ComparisonResult` |
| `src/stats/__init__.py` | 39 stable exports, side-effect free |

**Verified: importing `src.stats` performs 0 subprocess/`makedirs` calls** — no Git, no artifact
read, no SciPy test, no file creation, no RNG.

## 3. Artifact verification

The evaluator's own **public, side-effect-free** `src.eval.artifacts.verify_artifact` is used as
the base (filenames, manifest hashes, strict-JSON summary), with three A3a-specific format checks
layered on top that it does not make: **lowercase-64-hex digests**, **manifest filename
uniqueness**, and **rejection of an unexpected payload substituted for an expected one**. No
evaluation metric arithmetic or writer behaviour is duplicated, and `src/eval/**` is untouched.

Ingestion additionally re-validates: frozen `schema_version`/`metric_protocol` · all five summary
blocks · `invalid_pred_labels == 0` and the four artifact integrity flags · rows sorted by
`manifest_index` · exact and case-folded ID uniqueness · NPZ dtypes/shapes · `manifest_ids`
consistency with `per_image.jsonl` · `image_index` range · **sparse statistics summing exactly to
the dataset totals**. Arrays are copied and marked non-writeable so callers cannot corrupt a run.

## 4. Policy modes

An explicit `Policy` enum — never a `strict=True` boolean.

**`OFFICIAL`** requires `artifact_status == "official"`, `random_init == false`, `split == "test"`,
`expected_rows == actual_rows == 1561`, and no undefined primary metric.
**`NONOFFICIAL_SMOKE`** accepts smoke/provisional artifacts and synthetic row counts but still
requires valid hashes, valid schema, unique IDs and internally consistent sufficient statistics.
Verified: OFFICIAL correctly rejects the smoke fixture (`artifact_status='official'` required).

## 5. Alignment

Compatibility is asserted on `schema_version`, `metric_protocol`, `split`,
`split_manifest_sha256`, `class_map_sha256`, `preprocess_protocol`, class-space metadata, row
counts and condition identity; `metric_impl_sha256` inequality is an **error under OFFICIAL** and a
recorded warning otherwise. Then identity: duplicate and case-fold rejection, **exact set
equality** (no intersection), `clean_image_id` mapping equality, and a single canonical sort order.

**Guard-ordering observation (correct, worth recording):** two runs over different image sets
necessarily have different `split_manifest_sha256`, so the *compatibility* guard fires before the
*ID-set* guard. Both reject. The ID-set guard is therefore exercised directly through
`align_vectors` (A2b/A2c/A2d), which is also the path mIoU-C uses.

## 6. mIoU-C hand calculation

For clean image `clean_00`, all fifteen inputs:

| corruption | sev 1 | sev 2 | sev 3 | **severity mean** |
|---|---|---|---|---|
| synA | 0.10 | 0.20 | 0.30 | **0.20** |
| synB | 0.40 | 0.40 | 0.40 | **0.40** |
| synC | 0.00 | 0.30 | 0.60 | **0.30** |
| synD | 0.50 | 0.60 | 0.70 | **0.60** |
| synE | 0.10 | 0.10 | 0.40 | **0.20** |

`mIoU-C = (0.20 + 0.40 + 0.30 + 0.60 + 0.20) / 5 = 1.70 / 5 = ` **0.34** — observed **0.340000**.
Flat 15-value mean = `5.1 / 15 = 0.34`, confirming equivalence on a complete grid while the nested
form is retained.

Rejections proven: fewer than 15 cells · duplicate cell · unexpected severity (4) · undefined
condition score · a 15-run grid containing a missing cell.

> **Guard reachability note:** with exactly 15 unique, in-vocabulary cells the missing set is empty
> by pigeonhole (5 × 3 = 15), so the missing-cell guard is unreachable on that path and sits behind
> the count and duplicate guards as defence-in-depth. Every ordering rejects; none can produce an
> available-case average.

## 7. Wilcoxon and t-test reference values

`d = [0.10, −0.05, 0.0, 0.10, −0.05, 0.30, 0.0, 0.20]` — positive, negative, **zero** and **tied**
magnitudes together:

| | value |
|---|---|
| W | **26.0** |
| z | **1.279204** |
| p (one-sided) | **0.100413** |
| zeros / nonzero | 2 / 6 |
| status | **ok — VALID, not degenerate** |

Matches a direct `scipy.stats.wilcoxon(d, zero_method="pratt", correction=True,
alternative="greater", method="approx")` call to < 1e-12.

**Tie validity (the corrected policy):** `d = [0.2, −0.2, 0.2, 0.2, −0.2]` — every nonzero
magnitude tied — is classified **ok**, p = 0.382797. Ties are never treated as degenerate.

**t-test:** t = **2.365068**, p = **0.0494723**, df = **3**, matching a direct `ttest_rel` call.
Constant nonzero differences (`d ≡ 0.3`) return `t = null`, `p = null`,
`degenerate_zero_variance_nonzero`, mean 0.3 — **no Infinity serialized**.

**Degenerate vectors:** all-zero → W/z `null`, p **1.0**, `degenerate_all_zero`; t `null`, p 1.0;
r_rb `null` (`undefined_no_nonzero_rank_sum`); dz `null`; HL/mean/median **0.0**. Exactly one
nonzero → `insufficient_nonzero_pairs`, no conclusion.

## 8. Rank-biserial hand calculation

`d = [+3, +1, 0, −1, +2]` · `|d| = [3, 1, 0, 1, 2]` · sorted `0, 1, 1, 2, 3`

| value | rank |
|---|---|
| 0 | 1 (ranked, Pratt) |
| the two 1s | (2+3)/2 = **2.5** each |
| 2 | 4 |
| 3 | 5 |

`R₊ = 5 (+3) + 2.5 (+1) + 4 (+2) = ` **11.5** · `R₋ = 2.5 (−1)` · denominator **14.0**

**`r_rb = (11.5 − 2.5) / 14 = 9/14 = 0.642857143`** — observed **0.642857143**.

**The rejected denominator:** `n(n+1)/2 = 15` gives `9/15 = 0.600000000` — a **different** value,
demonstrating concretely why the frozen `R₊ + R₋` denominator matters when zeros are present.

## 9. Hodges-Lehmann hand calculation

| input | Walsh averages (i ≤ j) | count | median |
|---|---|---|---|
| `[1, 2]` | 1.0, 1.5, 2.0 | 3 (**odd**) | **1.5** |
| `[1, 2, 4]` | 1.0, 1.5, 2.5, 2.0, 3.0, 4.0 → sorted 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 | 6 (**even**) | **2.25** |

Both observed exactly. HL (2.25) differs from the plain median of the differences (2.0), confirming
the estimator is not a shortcut median.

## 10. Holm family

Full 8-member family finalised in canonical order; reject pattern matches statsmodels on
non-boundary inputs (`[True, True, False, False, False, False, False, False]`).

**Strict boundary:** `p_raw` exactly equal to the step threshold `α/8 = 0.00625` → **reject =
False**, with the boundary condition recorded. Tied p-values rank deterministically by
`(p, comparison_id)`.

Refusals proven: missing ID · duplicate ID · undefined primary p-value. (Unexpected IDs are
rejected at `run_comparison`, which only accepts canonical IDs.)

## 11. No non-finite output, no partial artifact

No `NaN` or `Infinity` token appears in any serialised A3a result. `ComparisonResult` carries **no
BCa or bootstrap field at all** — not even a null placeholder — so a partial `family.json` cannot be
mistaken for a complete one. **No `family.json`, `bootstrap.npz` or `MANIFEST.sha256` was emitted**;
those belong to A3b. All temporary synthetic artifacts were removed.

## 12. Two fixture defects the smoke caught

1. **A2/A4** initially asserted the ID-set/clean-mapping error message, but the compatibility guard
   correctly fires first (differing IDs ⇒ differing `split_manifest_sha256`). **Production was
   right; the assertions were wrong.** Fixed, and the ID-set guard is now reached directly via
   `align_vectors`.
2. **`grid_runs`** reused directory names, so a second invocation hit the writer's refuse-overwrite
   guard — the guard working as designed. Fixed by namespacing each invocation.

No assertion was weakened in either case.

## 13. Regressions

| Suite | Required | Observed |
|---|---|---|
| `smoke_metrics_contract.py` (A1b) | 21/21 | **21/21**, exit 0 |
| `smoke_eval_contract.py` (A2a) | 63/63 | **63/63**, exit 0 |
| `smoke_plantseg_class_map.py` (A2b-0) | 21/21 | **21/21**, exit 0 |
| `smoke_eval_plantseg.py` (A2b) | 53/53 | **53/53**, exit 0 |
| `smoke_stats_paired.py` (A3a) | all PASS | **45/45**, exit 0 |

No prior smoke script was modified.

---

_Files created: `src/stats/{__init__,ingest,align,robustness,tests}.py`,
`scripts/smoke_stats_paired.py`, and this report. `src/eval/**` and all documentation are
untouched. Nothing staged or committed._
