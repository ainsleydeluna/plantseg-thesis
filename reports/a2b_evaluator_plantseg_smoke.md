# A2b — Real PlantSeg validation path: adapter, checkpoint loader, CLI, capped smoke

**RESULT: ✅ A2b COMPLETE — REAL VALIDATION PATH PROVEN** · `53/53 checks passed`, exit code **0**

> **Capped validation smoke with a RANDOM-INIT student. No real checkpoint exists or was loaded.
> The metric values produced here are smoke diagnostics only and are NOT performance results, NOT
> an official evaluation, and NOT comparable to anything.** The test split was never opened.

Connects the proven A2a core to real PlantSeg validation data without altering any A2a semantics.
_Generated 2026-07-27 · CPU only._

---

## 1. Preflight

`master` · HEAD `32d52f4` · **0 ahead / 0 behind**. All pre-edit confirmations passed:

| Item | Result |
|---|---|
| A1b / A2a / A2b-0 regressions | **21/21 · 63/63 · 21/21**, all exit 0 |
| A2b-0 files present + unchanged | ✅ 4/4 |
| Class map: 116 entries, ids `0..115`, class 0 name `""`, role `background_or_non_disease`, 1–115 `disease` | ✅ |
| Class map semantic hash == pinned `d14182…a729` | ✅ |
| Validation count | **846 images / 846 masks** |
| Real E1 checkpoint exists | **No** |
| `torch.load` `weights_only` default (torch 2.9.1) | `None` → set **explicitly** |

## 2. Real-data adapter and identity

`src/eval/adapters.py` · `PlantSegEvalDataset(split, source_indices)` wraps `PlantSegDataset`
**without modifying it**. `__getitem__` calls the base, resolves the canonical stem via
`base.pairs[src][0].stem` **at that moment**, and returns
`{image, target, image_id, clean_image_id, manifest_index}`. `eval_collate` produces the proven
A2a `EvalBatch`.

Nothing is recovered after batching, inferred from batch position, or parsed out of another id.
`clean_image_id` is carried explicitly (equal to `image_id` for the clean condition; a future
corruption adapter sets a suffixed `image_id` and keeps the canonical stem here).

`manifest_index` is the **stable external index** — the row's position in the full canonical
name-sorted split, preserved even when the subset is non-contiguous.

## 3. Expected manifest and capped subset

**Cap resolved to explicit source indices before any loader exists.** `deterministic_subset(846, 4)`
returns evenly spaced, deliberately **non-contiguous** indices so the NPZ row-position mapping is
exercised against sparse external indices:

| external `manifest_index` | stem |
|---|---|
| **0** | `apple_black_rot_28` |
| **211** | `citrus_canker_179` |
| **422** | `grape_downy_mildew_google_0242` |
| **633** | `tomato_late_blight_google_0007` |

`build_expected_manifest_for(split, indices)` builds the manifest from a **filename listing only** —
no `PlantSegDataset` construction, no image or mask opened — so it is available *before* the request
preflight. It enforces the locked split count. `build_expected_manifest(adapter)` derives the same
manifest from a constructed adapter, and the smoke asserts the two are **equal** (C17), so the
listing and the dataset cannot drift apart unnoticed. Any residual drift would still be caught by
the A2a core, which compares every observed sample against the manifest.

There is no `--max-batches`; only `--max-samples`, because sample count defines the manifest exactly.
`num_workers=0` throughout, so no unselected row can be prefetched.

## 4. Class map and dataset metadata

Loaded from `configs/plantseg_class_map.json` (A2b-0) and passed through **unchanged** as the A2a
class-map descriptor. `load_class_map()` independently re-verifies: 116 entries · ids `0..115` ·
class 0 name exactly `""` with role `background_or_non_disease` · 1–115 non-empty `disease` ·
semantic hash **`d14182423b6f176f940cada979adb364701fe186be091655ce38a80ce791a729`**, and refuses to
evaluate on mismatch. The artifact's `class_map_sha256` equals that same value (C11).

No re-derivation from `Metadata.csv`, no majority vote, no reordering, no renaming, and no
substitution of a display label for class 0's empty official name.

Real dataset identity: `PlantSeg` · DOI `10.5281/zenodo.17719108` · split `val` · `num_classes 116` ·
`background_index 0` · `ignore_index 255` · `preprocess_protocol core_preprocess/1.0.0` (the real
deterministic validation pipeline: EXIF-transpose image only → keep-ratio resize long-side 512 →
center pad to 512² with image `[124,116,104]` / mask `255` → ImageNet normalize).

## 5. FP32 model and checkpoint loader

`src/eval/model_loading.py`.

`build_fp32_student(num_classes=116, device)` uses the repository factory `build_student` with
`pretrained=False` (no download, no architecture duplication) and asserts `used_pretrained` is False.

`load_student_checkpoint(path, map_location, ...)` accepts **only** the one documented repository
schema written by `train_e1.save_checkpoint`. It requires `model_state_dict` and `num_classes`,
asserts `num_classes == 116`, builds with `pretrained=False`, loads `strict=True`, and returns real
`sha256` over the raw checkpoint bytes.

**`torch.load` safety choice.** torch 2.9.1 reports `weights_only`'s default as `None`
(version-dependent resolution), so it is passed **explicitly** as `weights_only=True` rather than
inheriting a shifting library default. This is compatible with the repository schema because every
stored value is a plain `int`/`float`/`str`, a `dict`/`list` thereof, or a `torch.Tensor` — all
allowed by the weights-only unpickler. No custom classes, numpy scalars, or lambdas are stored. A
future checkpoint needing richer types is a deliberate schema change to be handled explicitly, not
by relaxing the flag.

## 6. Stage-neutral CLI

`scripts/evaluate_model.py` — stage-neutral in name and metadata. Execution order:

```
validate_cli_args (pure)  ->  load + verify class map  ->  resolve capped source indices
  ->  build expected manifest (filename listing only)  ->  prepare_artifact_request
  ->  validate_artifact_request        <-- refusal happens HERE
  ->  construct dataset adapter        <-- step 5 begins only after preflight succeeds
  ->  construct model  ->  evaluate_model  ->  write_artifact
```

A2b implements **only** `model_role=student`, `precision=fp32`, `condition=clean`. Teacher, INT8
PTQ/QAT and corruption requests are rejected explicitly — metadata neutrality is never presented as
runtime support. Importing the module has no side effects.

## 7. Checkpoint-loader tests — 8/8

Temporary files generated from a random-init student, **never used for the artifact**.

| # | Case | Result |
|---|---|---|
| A1 | valid nested synthetic checkpoint loads `strict=True` | ✅ |
| A2 | bare `state_dict` rejected | ✅ "this is a BARE state_dict…" |
| A3 | missing `model_state_dict` rejected | ✅ |
| A4 | missing `num_classes` rejected | ✅ |
| A5 | wrong `num_classes` (6) rejected | ✅ |
| A6 | incompatible state dict rejected under `strict=True` | ✅ RuntimeError surfaced |
| A7 | recorded SHA-256 == independent raw-file re-hash | ✅ |
| A8 | `weights_only` set explicitly, not inherited | ✅ `True` |

## 8. CLI-guard and construction-order tests — 15/15

Unsupported teacher role · unsupported INT8 precision · unsupported corruption condition ·
random-init + checkpoint · neither random-init nor checkpoint · random-init with non-smoke status ·
unconfirmed test · random-init test · capped test · smoke-status test — all rejected.

**Construction counters prove ordering:** for a rejected teacher-role run, a rejected capped-test
run, and an `official` refusal (governed paths dirty), `dataset=0, model=0` and **no output
directory was created**. The test-split guards are exercised purely through argument/request
validation — `PlantSegDataset("test")` was never constructed, no test directory was listed, and no
subprocess containing `--split test` was executed.

## 9. Capped validation smoke — 22/22

4 deterministic validation samples · batch 2 · `num_workers=0` · random-init FP32 student · CPU ·
clean condition · `artifact_status=smoke` · temporary output directory.

Verified: four artifact files present and manifest hashes verify · strict JSON with **zero**
`NaN`/`Infinity` tokens · `expected_rows == actual_rows == 4` · artifact self-identifies as
random-init smoke with `checkpoint_path`/`checkpoint_sha256` both `null` · `class_map_sha256` equals
the pinned value · real dataset identity and `core_preprocess/1.0.0` · all classwise arrays length
**116** · rows canonically ordered by external `manifest_index` `[0, 211, 422, 633]` despite batching
· real stems attached as `image_id` · `image_id == clean_image_id` in every row · dataset aggregate
metrics match `src.eval.metrics` on the same four rows · per-image metrics match · sparse sufficient
statistics sum exactly to dataset totals · NPZ `image_index` maps by **row position** into
`manifest_ids` while `per_image.jsonl` retains the sparse external indices · overwrite refused ·
temporary artifact removed after inspection.

**Manifest/adapter divergence (D1, D2):** a manifest describing a different subset is rejected
(`unexpected manifest_index …`), and a manifest with the first two identities **swapped** — both ids
present exactly once, only the pairings wrong — is rejected with
`manifest_index N is paired with image_id X but the expected manifest has Y`.

> Only four images are evaluated, so the overwhelming majority of the 116 classwise entries are
> undefined. **That is correct, not a defect.** The random-init aggregate values are not reproduced
> or interpreted here.

## 10. Path-access and test-non-access evidence

`PIL.Image.open` is the narrowest real file-opening boundary used by `PlantSegDataset`
(`with Image.open(img_path) as im, Image.open(mask_path) as mk:`). It was instrumented for the
capped smoke and **restored in a `finally` block**, including on failure.

| Category | Count |
|---|---|
| Validation **images** opened (distinct) | **4** |
| Validation **masks** opened (distinct) | **4** |
| **Test image/mask paths opened** | **0** |
| Opened samples outside the selected subset | **0** |

Directory roles are reported repo-relatively (`images/val/`, `annotations/val/`); no
machine-specific absolute paths are recorded. Configuration and class-map reads are counted
separately and are not dataset image/mask access. (Repeat opens of the same four stems occur
because parity re-evaluates the same rows; the **distinct** counts are 4 and 4.)

## 11. Regression results

| Suite | Required | Observed |
|---|---|---|
| `smoke_metrics_contract.py` (A1b) | 21/21 | **21/21**, exit 0 |
| `smoke_eval_contract.py` (A2a) | 63/63 | **63/63**, exit 0 |
| `smoke_plantseg_class_map.py` (A2b-0) | all PASS | **21/21**, exit 0 |
| `smoke_eval_plantseg.py` (A2b) | all PASS | **53/53**, exit 0 |

None of the prior smoke scripts was modified.

### A defect the smoke caught in its own fixture
The first run failed **D2**: the "wrong id/index pairing" fixture copied one entry's id onto
another while leaving the original in place, producing a **duplicate `image_id`**. The core
correctly rejected it — but for the duplicate-id reason, not the pairing reason the check claimed to
test. **The defect was in the test fixture, not in production.** Fixed by swapping the two
identities so both ids remain present exactly once and only the pairings are wrong. No assertion was
weakened.

## 12. Confirmations

- ✅ **No real checkpoint was loaded** — none exists. All checkpoint tests used temporary synthetic
  files from a random-init student, never reused for the artifact.
- ✅ **The test split was never opened, listed, or constructed.**
- ✅ **No full validation evaluation** — exactly 4 of 846 rows.
- ✅ **Random-init metrics were not interpreted** as performance anywhere.
- ✅ No install, download, GPU use, training, backward pass, staging, or commit.
- ✅ A2a core files (`evaluate.py`, `artifacts.py`), `metrics.py`, `__init__.py`, and the class-map
  JSON are all unchanged; the class-map hash still equals the pinned value.
- ✅ No smoke artifact remains in the working tree.

---

_Files created: `src/eval/adapters.py`, `src/eval/model_loading.py`, `scripts/evaluate_model.py`,
`scripts/smoke_eval_plantseg.py`, and this report. Nothing staged or committed._
