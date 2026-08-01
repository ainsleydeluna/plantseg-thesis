# A3b — deterministic paired bootstrap, BCa, non-inferiority and the statistics artifact

> **Status: implemented, synthetic smoke 153/153, exit 0 (2026-07-31).** Implements
> [STATISTICAL_ANALYSIS_CONTRACT.md](../docs/STATISTICAL_ANALYSIS_CONTRACT.md) §§8–9, §8.7, §12.2,
> §12.3 and §12.4 against the governing contracts committed at `9267ffe`.
>
> **This smoke is NOT a statistical result.** Every fixture is synthetic, `B = 64` was used for
> speed while production `B = 10,000` is recorded in protocol metadata, and the development stack
> differs from the pinned one. The artifact it produces is `artifact_status = "nonofficial"` by
> construction.

## 1. Implementation summary

Three new modules complete the statistics layer. A3a's typed inference is consumed, never
recomputed; the frozen 35-task matrix, the amended seed protocol, the first-trigger fallback ladder
and the nineteen-key artifact schema are implemented exactly as committed.

| Module | Responsibility |
|---|---|
| `src/stats/bootstrap.py` | frozen namespaces and the 35-task matrix · amended §8.1 seed derivation · row-wise draws · BCa with the P1–P8 ladder · scalar replicate/jackknife callables · the descriptive Source-B carrier · the runtime A3a source-schema assertion |
| `src/stats/noninferiority.py` | int64 densification · union-present pooled dataset mIoU · paired replicate/jackknife re-accumulation · relative retention · E3→E6 non-inferiority · E6-KD trigger |
| `src/stats/artifact.py` | ordered A3a/Holm serialization · officiality derivation · the nineteen-key `family.json` · NPZ · JSON↔NPZ verification · A2a writer discipline and the artifact verifier |

`src/stats/__init__.py` was extended with **exports only** — no execution, no filesystem access, no
import-time validation.

## 2. Public API

`bootstrap`: `TaskSpec` · `FROZEN_TASK_MATRIX` / `FROZEN_TASK_IDS` / `TASK_BY_ID` ·
`expected_comparison_ids` · `SeedMaterial` · `canonical_seed_bytes` · `derive_seed` ·
`make_generator` · `reconstruct_seed_material` · `draw_replicate_indices` · `BcaOutcome` ·
`bca_interval` · `BootstrapTaskResult` · `run_bootstrap_task` · `scalar_statistic` ·
`scalar_callables` · `DescriptiveScalars` · `descriptive_scalars` · `comparison_observed` ·
`FROZEN_A3A_FIELDS` · `assert_a3a_source_schema`.

`noninferiority`: `densify` · `pooled_miou_from_totals` · `PooledStages` · `relative_retention` ·
`build_non_inferiority` · `build_e6_kd` · `NonInferiorityResult` · `E6KDResult` ·
`SensitivityDecision`.

`artifact`: `ArtifactInputs` · `serialize_a3a` · `software_environment_block` ·
`officiality_warnings` · `derive_artifact_status` · `contract_sha256` · `build_family` ·
`write_statistics_artifact` · `verify_statistics_artifact` · the frozen key/vocabulary constants.

## 3. Files changed

| File | Change |
|---|---|
| `src/stats/bootstrap.py` | created |
| `src/stats/noninferiority.py` | created |
| `src/stats/artifact.py` | created |
| `scripts/smoke_stats_bootstrap.py` | created |
| `reports/a3b_bootstrap_noninferiority_smoke.md` | created (this file) |
| `src/stats/__init__.py` | modified — exports only |

No other path was touched. No documentation was modified during A3b.

## 4. The frozen 35-task matrix

Built by importing `CANONICAL_COMPARISON_IDS` from `src/stats/tests.py`; no second authoritative
copy of the eight IDs exists in A3b source.

| Group | Tasks |
|---|---:|
| seven clean superiority comparisons × `mean_delta`, `median_delta`, `hodges_lehmann_shift`, `dataset_miou_delta` | 28 |
| `robustness_e1_e6` × three scalar statistics | 3 |
| `descriptive_e1_e3` × three scalar statistics | 3 |
| `noninferiority_e3_e6` × `dataset_miou_delta` | 1 |
| **total** | **35** |

**34 `two_sided` + 1 `one_sided_lower`**; the sole one-sided task is
`noninferiority_e3_e6__dataset_miou_delta`, and it is the only record that omits `upper_bound`.
No `dataset_miou_c_delta`, no pooled robustness statistic, no 36th task.

## 5. Observed-source result — 24 / 3 / 8

| `observed_source` | tasks | verified |
|---|---:|---|
| `a3a_comparison_result` | 24 | bit-for-bit against `shifts.mean_delta`, `shifts.median_delta`, `hodges_lehmann_shift` |
| `a3a_primitives_descriptive` | 3 | `engineering_shifts` + `hodges_lehmann` on the aligned E1/E3 vector, direction E3 − E1 |
| `a3b_pooled_reaccumulation` | 8 | int64 pooled re-accumulation |

`run_comparison` still rejects `descriptive_e1_e3`; the carrier holds no Wilcoxon, t-test,
rank-biserial, Cohen's dz, p-value or Holm field — their absence is structural, not a null
placeholder. `mean_delta_pp` is carried for reporting and is not a bootstrap statistic.

## 6. Synthetic test coverage — 153/153

`n = 12` synthetic images, `C = 6` classes, `B = 64`.

| Section | Checks | Result |
|---|---:|---|
| 1. Seed and RNG — exact canonical bytes, rendered `root_seed=42`, no trailing NUL, digest/entropy derivation, 16-byte entropy preservation including leading zero bytes, namespace separation, identifier rejection, deterministic row-wise draws, small-B prefix, independent per-statistic and per-comparison streams, global RNG untouched | 22 | all PASS |
| 2. Task matrix — 35 tasks, imported canonical order, 34/1 split, two-component task IDs, group ordering, 24/3/8 map | 10 | all PASS |
| 3. BCa and the ladder — half-weight ties, acceleration formula, linear quantiles, p0 clipping both ends, P1–P4 each fires with exactly one fallback warning, P3 short-circuit, all-zero path, saturation, one-sided prohibitions, status/warning biconditional, canonical warning order | 27 | all PASS |
| 4. Scalars and provenance — Holm family finalised, `p_raw == wilcoxon.p_value`, Source-A bit identity ×3, primitives ≡ typed result, canonical HL reuse, `run_comparison` refusal, descriptive path, carrier has no inferential field | 17 | all PASS |
| 5. Pooled estimand — int64, observed/replicate/jackknife each equal a naive reference, duplicate multiplicity, scalar averaging ≠ ratio-of-sums, zero-eligible fatal | 7 | all PASS |
| 6. NI / retention / E6-KD — strict −0.020 boundary, binary-exact vector, identities, retention rules, four margins from one bound, strict 0.010 boundary | 18 | all PASS |
| 7. Full run and artifact — 35 tasks, order independence, 37-record inventory, contract hash, nineteen keys, A+ envelope, dataclass field orders, enum serialization, sorting, JSON↔NPZ, officiality, integrity, three-file writer, refuse-overwrite, injected-failure cleanup, tamper detection, official refusal, run_id validation | 52 | all PASS |

**Cross-block identities — three separate requirements.** The writer enforces each independently,
and they must not be conflated:

1. `noninferiority_e3_e6.observed_delta` equals the **observed value of the NI bootstrap task**
   `noninferiority_e3_e6__dataset_miou_delta`;
2. `noninferiority_e3_e6.lower_bound` equals that **same task's lower bound**;
3. `e6_kd_trigger.observed_drop` equals `noninferiority_e3_e6.observed_drop`.

There is **no** requirement that a bootstrap task's observed value equal its own lower bound; those
are unrelated quantities and no such check exists.

**Seed entropy — what is actually guaranteed.** Production preserves the **exact first 16 SHA-256
digest bytes**, and the unsigned big-endian integer derived from those bytes is supplied **alone**
to `SeedSequence`. There is **no minimum bit length**: a digest prefix carrying leading zero bytes
is entirely valid and is accepted, so correctness is never argued from the entropy integer being
"large". The smoke verifies leading-zero preservation and the absence of 32-bit and 64-bit
truncation deterministically.

**Contract hashing — measured, not inferred.** Across one complete end-to-end synthetic
build-and-write operation, instrumentation counted the raw contract read **exactly once in total**:
`contract_sha256` reads `docs/STATISTICAL_ANALYSIS_CONTRACT.md` once as raw bytes while the artifact
document is being constructed, and that same buffer is the one hashed. `write_statistics_artifact`
consumes the already-constructed document and performs **zero** further contract reads, as does
`verify_statistics_artifact`.

**A fixture defect the smoke caught.** The first run failed one case: the E6-KD boundary test used
`0.5 − 0.49`, which is `0.010000000000000009` in float64 — strictly greater than the threshold, so
the production code correctly triggered. The *test vector* was wrong, not the implementation. It is
the same binary-exactness trap the contract records for `0.4 − 0.5`. The boundary is now constructed
exactly (`build_e6_kd(0.010, 0.0)` → drop `0.01`, no trigger; one ULP above → trigger), and the
naive decimal case is retained as an explicit documented-trap assertion.

## 7. Official/nonofficial limitation

The artifact produced here carries all six officiality warnings —
`analysis_id_nonofficial`, `software_environment_unpinned`, `input_artifact_nonofficial`,
`synthetic_input_data`, `bootstrap_replicates_nonproduction`, `jackknife_count_nonproduction` —
so `artifact_status = "nonofficial"`, derived from the warnings by asserted biconditional. An
official request on this stack **refuses before final-directory creation** and leaves no temporary
directory; that refusal is tested. Nothing here is an official statistical result.

**On official sizes.** The full official-size combination — `B = 10,000` together with
`jackknife_count = 1,561` — was **not exercised in the smoke because doing so was unnecessary and
substantially more expensive**, not because it is impossible with synthetic fixtures. The
official-size requirements are instead verified through the **validation and refusal paths**: the
officiality-warning derivation emits `bootstrap_replicates_nonproduction` and
`jackknife_count_nonproduction` whenever the counts depart from `10,000` / `1,561`, those warnings
force `artifact_status = "nonofficial"`, and an official request on a non-conforming run refuses
before any final directory is created.

## 8. Regressions

| Suite | Result |
|---|---|
| A1b `smoke_metrics_contract.py` | `PASS=21 EXPECTED_RED=0 UNEXPECTED_GREEN=0 UNEXPECTED_FAILURE=0 (total 21)` |
| A2a `smoke_eval_contract.py` | 63/63 |
| A2b-0 `smoke_plantseg_class_map.py` | 21/21 |
| A2b `smoke_eval_plantseg.py` | 53/53 |
| A3a `smoke_stats_paired.py` | 45/45 |
| A3b-0 `smoke_corruption_protocol.py` | 31/31 |
| **A3b `smoke_stats_bootstrap.py`** | **153/153** |

All exit 0.

## 9. Repository safety

Only the six allowed files changed. `git diff --check` clean, nothing staged, nothing committed,
nothing pushed. The checkpoint commit `9267ffe` remains HEAD and the branch is exactly 1 ahead of
`origin/master`. `docs/reference/reference.pdf` was never opened and remains modified and unstaged.
The forbidden modules — `ingest.py`, `align.py`, `robustness.py`, `tests.py`,
`corruption_protocol.py`, `configs/**`, `src/eval/**`, `docs/**` — are untouched.

## 10. No data, training, or hardware use

No PlantSeg image, mask, split file or checkpoint was opened. The test split was never accessed. No
training, no inference, no GPU, no installs, no downloads, no network. `imagecorruptions` remains
neither installed nor pinned; the corruption grid is read from `configs/corruption_protocol.json`
through `official_corruption_grid()` for the 37-record inventory test only.

## 11. Recorded implementation findings

1. **P5 is provably unreachable and is a defensive guard.** By the power-mean inequality
   `|Σc³| ≤ (Σc²)^{3/2}`, so `|a| = |Σc³| / (6·(Σc²)^{3/2}) ≤ 1/6` whenever `Σc²` and the
   denominator are finite and non-zero. Any input that would make the acceleration non-finite first
   makes `Σc²` or the denominator non-finite, which P3/P4 catch. P1–P4 are exercised constructively;
   P5 retains its guard but cannot be reached with float64 inputs.
2. **P8 is constructively reachable; P6 and P7 are neither claimed reachable nor unreachable.**
   The independent A3b review reached **P8** with ordinary finite synthetic inputs (a clipped `z0`
   together with a strongly skewed jackknife), so it must not be described as requiring exotic
   conditions. **P6** and **P7** were not constructively reached, and no claim is made in either
   direction about them. One-sided-lower execution remains **structurally barred** from P8, which is
   guarded by `interval_type == TWO_SIDED`.
3. **The frozen fallback label is heterogeneous by design.** `acceleration_defined == false` implies
   the fallback method, but the converse is false — P6–P8 fire with a finite, well-defined
   acceleration. The verifier does not assert the biconditional.
