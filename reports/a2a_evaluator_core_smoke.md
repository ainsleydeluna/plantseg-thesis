# A2a — Stage-neutral evaluator core + frozen artifact writer (evidence report)

**RESULT: ✅ A2a COMPLETE — CORE CONTRACT PROVEN** · `63/63 checks passed`, exit code **0**

> **SYNTHETIC FIXTURE ONLY.** No PlantSeg image, mask, split JSON, `Metadata.csv`, checkpoint, or
> dataloader was opened at any point. Every tensor is constructed inline in
> `scripts/smoke_eval_contract.py`. The model is a **deterministic, untrained, no-checkpoint** dummy.
> These numbers are **not** PlantSeg results and must never be reported as such.

Implements [EVALUATION_CONTRACT.md](../docs/EVALUATION_CONTRACT.md) §§5–7 against the metric core
delivered in A1b. _Generated 2026-07-26 · CPU only._

---

## 1. Preflight

| Item | Value |
|---|---|
| Branch / HEAD / sync | `master` · `32d52f4` · **0 ahead / 0 behind** |
| A1b regression | **21/21 PASS, exit 0** |
| D5 | closed in both governing documents |
| Evaluator pre-existing | none — `src/eval/` held only `metrics.py`, `__init__.py` |
| Governed paths dirty | **yes** → A2a can only produce `smoke`, and `official` must refuse |

## 2. Core API implemented

```
prepare_artifact_request(...)  -> ArtifactRequest   # pure construction, zero I/O
validate_artifact_request(req) -> Provenance        # READ-ONLY pre-inference gate
evaluate_model(model, batches, ...) -> EvalResult   # pure compute, no fs/git/hashing
write_artifact(result, req, prov) -> Path           # atomic four-file finalisation
verify_artifact(out_dir) -> dict                    # re-verify from disk
```

**`src/eval/evaluate.py`** — `Condition`, `ManifestEntry`, `EvalBatch`, `PerImageRow`, `EvalResult`,
`EvaluationIntegrityError`, `evaluate_model`. Knows nothing about MobileNetV3, SegNeXt, FP32/PTQ/QAT
construction, `PlantSegDataset`, corruption datasets, or checkpoints.

**`src/eval/artifacts.py`** — provenance/hashing, the official-vs-smoke gate, schema assembly and
validation, strict JSON/NPZ serialisation, atomic finalisation.

**`EvalBatch` carries identity directly** — `images`, `targets`, `image_ids`, **`clean_image_ids`**,
`manifest_indices`. Nothing is recovered from `dataset.pairs`, inferred from batch position, or parsed
back out of a filename. `clean_image_ids` is mandatory now, not deferred to a corruption adapter.

**Neutrality:** any model returning `[B, C, H, W]` logits is accepted; `stage`, `model_role`,
`precision`, `quant_backend` and `condition` are metadata that never reach metric arithmetic.
`Condition` can already express `corruption`/severity — **A2a implements and tests `clean` only and
generates no corrupted images.**

## 3. Artifact schema — field-by-field

| Contract | Implemented | Note |
|---|---|---|
| §5.3 top-level `schema_version`, `metric_protocol` | ✅ top-level, not nested | validated pre-write **and** after strict re-read |
| §5.3 `run.*` (17 fields) | ✅ all present | `checkpoint_path`/`checkpoint_sha256` = `null` under `random_init` |
| §5.3 `dataset.*` (12 fields) | ✅ all present | `actual_rows == expected_rows` enforced |
| §5.3 `dataset_level.*` (9 fields) | ✅ all present | aggregates from production reducers; `aacc_diagnostic` = `Σtp/Σgt` |
| §5.3 `per_class.*` (9 arrays) | ✅ all length **116** | `iou/dice/acc` = `null` where undefined |
| §5.3 `integrity.*` (9 fields) | ✅ all present | `invalid_pred_labels` must be 0 |
| §5.4 per-image row | ✅ **exactly the 11 frozen keys** | **no** per-image classwise arrays — §5.4 does not list them |
| §5.5 NPZ arrays (9) | ✅ all present, int64 | `manifest_ids` unicode `[n_rows]` |
| §5.2 validity | ✅ | `allow_nan=False`; `null`+status; full float repr; ascending rows |
| §5.1 layout | ✅ exactly 4 files | `MANIFEST.sha256` excludes itself |

**Aggregates come only from production** (`miou_from_confusion`, `macc_from_confusion`,
`dice_from_confusion`, `per_image_miou`). Classwise `tp/gt/pred/union/iou/dice/acc` and the
`n_eligible_*` counts are derived from int64 sufficient statistics — serialisation, not a second
reducer — and the classwise reconstruction is **asserted equal to the production macro** within
float32 tolerance (`1e-6`) or evaluation aborts.

## 4. Identity and integrity

Every observed sample is checked against an **expected manifest** of `(manifest_index, image_id,
clean_image_id)`. Hard failure on: unexpected index · wrong index↔id pairing · **wrong clean-image
mapping** · duplicate exact ID · case-folded collision · duplicate manifest index · missing row.
Rows are sorted by `manifest_index` **after** validation, so a deliberately non-canonical input order
(`[[4,1],[5,0,3],[2]]`) still yields canonical output.

**NPZ index semantics (resolved and documented).** `manifest_ids` is ordered canonically by
`manifest_index`; **`image_index` is the zero-based ROW POSITION into `manifest_ids`**, *not* the
external `manifest_index` (which may be sparse in general). `per_image.jsonl` separately retains the
stable external `manifest_index`. Asserted for every sparse row:
`manifest_ids[image_index] == image_id`.

## 5. Synthetic smoke results — 63/63

All 32 required verifications, plus data and integrity-abort cases:

| # | Check | Result |
|---|---|---|
| 2b | **writer rejects a non-116 class count** for `plantseg-eval/1.0.0` | ✅ `"…freezes num_classes=116; got 6"` |
| 17 | **provenance hashes are real** 64-hex, none zero/placeholder, all distinct; `metric_impl_sha256` equals a live re-hash of the imported `metrics.py` | ✅ |
| 28 | **wrong ID/index mapping** — a real ID submitted under a different real index | ✅ aborts |
| 32 | **no PlantSeg filesystem access** — `builtins.open` instrumented for the whole run | ✅ 4 paths opened, **0** PlantSeg hits; `src.data`/`configs.data` never imported |

Earlier verifications (all still passing):

| # | Check | Result |
|---|---|---|
| 1,2 | exactly the four contract filenames | ✅ |
| 3 | strict JSON parse (summary + all rows) | ✅ |
| 4 | zero `NaN`/`Infinity` tokens in raw text | ✅ |
| 5 | schema field presence + types, 116-length arrays | ✅ |
| 6 | ascending `manifest_index` despite shuffled input | ✅ `[0,1,2,3,4,5]` |
| 7 | ID/manifest integrity, `clean_image_id` carried | ✅ |
| 8 | case-folded uniqueness | ✅ |
| 9 | dataset metric parity vs `src.eval.metrics` | ✅ |
| 10 | per-image metric parity vs `src.eval.metrics` | ✅ |
| 11,12 | sparse reconstruction + dataset totals | ✅ |
| 13 | NPZ dtypes/shapes | ✅ |
| 14 | undefined → `null` + status | ✅ |
| 15 | manifest hash verification (+ tamper detection) | ✅ |
| 16 | overwrite refusal | ✅ |
| 17 | **official refusal before inference** | ✅ |
| 18 | no partial artifact after injected failure | ✅ |
| 19 | semantic two-run determinism | ✅ |

**Integrity aborts (10):** invalid output shape · wrong class count · duplicate ID · case-only
collision · duplicate manifest index · **missing expected-manifest entry** · unexpected row ·
**wrong ID/index pairing** · wrong `clean_image_id` · out-of-range label. All raise and abort
finalisation; none degrades into a normal row carrying an error-looking status.

The missing-entry case tests a genuine **mapping** failure (four expected entries never observed),
not merely a count mismatch.

### A real bug the smoke caught
The first run failed check 19: sparse NPZ rows were appended in **batch-iteration order**, making the
NPZ payload depend on how the caller grouped batches. **This was a genuine implementation defect, not
a test problem.** Fixed by sorting sparse rows canonically by `(image_index, class_id)` via
`np.lexsort` before serialisation. The check was **not** relaxed.

## 6. Generated artifact — representative content

Hand-computed dataset anchors, all matched exactly:

| Quantity | Hand-derived | Observed |
|---|---|---|
| all-class mIoU | `(0.9+1+0.75+0.8+1+0+0+1)/8 = 5.45/8` = **0.68125** | `0.6812499761581421` |
| disease-only mIoU | `4.55/7` = **0.65** | `0.6500000357627869` |
| aAcc | `73/80` = **0.9125** | `0.9125` |
| mIoU eligible (union-present) | **8** (incl. FP-only class 5) | 8 |
| mAcc eligible (GT-present) | **7** (class 5 excluded) | 7 |

Class 5 appears only as a prediction (`gt=0, pred=2`): it is **counted** in mIoU/Dice and **excluded**
from mAcc — the intentional asymmetry, proven end-to-end through the artifact.

`summary.json` (per-class arrays truncated):

```json
{
  "schema_version": "plantseg-eval/1.0.0",
  "metric_protocol": "plantseg-metrics/1.0.0",
  "run": {
    "artifact_status": "smoke", "stage": "E1", "model_role": "student", "precision": "fp32",
    "checkpoint_path": null, "checkpoint_sha256": null, "random_init": true,
    "repo_commit": "32d52f48d3daebd067888525a6b9435ec349e717",
    "governed_paths_clean": false,
    "dirty_allowlisted": ["AGENTS.md", "docs/open_questions.md",
      "docs/reference/reference.pdf", "reports/a1a_metric_contract_red.md",
      "reports/a1b_metric_contract_green.md", "reports/e1_runpod_launch_runbook.md"],
    "worktree_state_sha256": "80e45ab0...14cb", "metric_impl_sha256": "9898d6dc...86d9",
    "config_sha256": "6a8bd552...0174"
  },
  "dataset": {
    "name": "SYNTHETIC contract fixture (NOT PlantSeg data)",
    "split": "val", "expected_rows": 6, "actual_rows": 6,
    "condition": {"type": "clean", "name": null, "severity": null},
    "preprocess_protocol": "synthetic-contract/1.0.0",
    "num_classes": 116, "background_index": 0, "ignore_index": 255
  },
  "dataset_level": {
    "all_class_miou": 0.6812499761581421, "all_class_miou_n_eligible": 8,
    "all_class_macro_dice": 0.7116750478744507, "all_class_dice_n_eligible": 8,
    "all_class_macc": 0.8163265585899353, "all_class_macc_n_eligible": 7,
    "disease_only_miou": 0.6500000357627869, "disease_only_miou_n_eligible": 7,
    "aacc_diagnostic": 0.9125
  },
  "per_class": {
    "gt_support":   [56, 4, 4, 4, 4, 0, 4, 4, 0, "…len 116"],
    "pred_support": [58, 4, 3, 5, 4, 2, 0, 4, 0, "…len 116"],
    "intersection": [54, 4, 3, 4, 4, 0, 0, 4, 0, "…len 116"]
  }
}
```

`per_image.jsonl` — 2 of 6 rows verbatim (note the all-ignored row):

```json
{"image_id": "syn_0001_multidisease", "clean_image_id": "syn_0001_multidisease", "manifest_index": 1, "condition": {"type": "clean", "name": null, "severity": null}, "all_class_miou": 0.8499999642372131, "all_class_miou_status": "ok", "disease_only_miou": 0.7749999761581421, "disease_only_miou_status": "ok", "n_eligible_all_class": 3, "n_eligible_disease_only": 2, "gt_disease_classes": [2, 3]}
{"image_id": "syn_0002_all_ignored", "clean_image_id": "syn_0002_all_ignored", "manifest_index": 2, "condition": {"type": "clean", "name": null, "severity": null}, "all_class_miou": null, "all_class_miou_status": "undefined_no_eligible_class", "disease_only_miou": null, "disease_only_miou_status": "undefined_no_eligible_class", "n_eligible_all_class": 0, "n_eligible_disease_only": 0, "gt_disease_classes": []}
```

`sufficient_stats.npz`: `image_index/class_id/tp/gt/pred` int64 `(12,)` · `dataset_tp/gt/pred` int64
`(116,)` · `manifest_ids` `<U23` `(6,)`. 12 sparse rows for 6 images — the all-ignored image
contributes none, exactly as the sparse rule requires.

**`undefined_per_image_scores` interpretation (recorded):** the number of **rows** with at least one
undefined required metric. The all-ignored image counts **once**, not twice. Observed: `1`.

## 7. Hash and atomic-write verification

**Canonical bytes — all real, no placeholders:**

| Hash | Canonical bytes |
|---|---|
| `worktree_state_sha256` | raw stdout of `git -c core.quotepath=false status --porcelain=v1 -z --untracked-files=all` — NUL-delimited, hashed unmodified |
| `metric_impl_sha256` | raw bytes of the imported `src/eval/metrics.py` |
| `config_sha256` | canonical JSON (UTF-8, sorted keys, compact separators, `allow_nan=False`) of the config payload |
| `split_manifest_sha256` | `<manifest_index>\t<image_id>\t<clean_image_id>\n` per row, ascending index, UTF-8/LF; tab/newline/CR/NUL rejected in IDs |
| `class_map_sha256` | canonical JSON of the complete ordered 116-entry descriptor (`id`, `name`, `role`) |
| `checkpoint_sha256` | `null` — no checkpoint exists; **never fabricated** |

`config_sha256` **excludes** `run_id`, timestamps, output paths, metric results, the worktree hash and
file-manifest hashes. Included keys: `schema_version`, `metric_protocol`, `stage`, `model_role`,
`precision`, `quant_backend`, `random_init`, `device`, `dataset_name`, `dataset_doi`, `split`,
`condition`, `preprocess_protocol`, `expected_rows`, `num_classes`, `background_index`, `ignore_index`.
Verified stable across two runs.

**`MANIFEST.sha256`** — `<64 lowercase hex><two ASCII spaces><filename>\n`, UTF-8, LF, filenames sorted
`per_image.jsonl` → `sufficient_stats.npz` → `summary.json`, self excluded, every line re-verified by
re-reading and re-hashing before the final rename. Tampering is detected (proved by appending a byte
to `per_image.jsonl` and observing the mismatch).

**Atomicity** — write into a sibling `.<name>.tmp-<uuid>` → validate schema → strict JSON re-read →
NPZ re-read + invariant checks → manifest → verify → `rename`. Any exception removes the temp
directory. Two failure phases tested: **before** temp creation (official refusal) and **after** temp
creation but **before** rename (injected failure). In both cases the target directory is absent and no
temp sibling remains.

**Determinism claim, stated precisely:** `summary.json` is semantically identical after removing
`run_id` and `timestamp_utc`; `per_image.jsonl` is **byte-identical**; the NPZ is compared
**semantically** (keys, dtypes, shapes, values) because ZIP-container metadata may differ.
**Byte-for-byte NPZ determinism is not claimed.** Within a single run, `MANIFEST.sha256` matches the
exact raw bytes written.

## 8. Official-guard verification

`validate_artifact_request(status="official")` raised **before any model forward**:

```
refusing to produce an 'official' artifact: governed paths are dirty or untracked ->
['docs/EVALUATION_CONTRACT.md', 'docs/IMPLEMENTATION_CONTRACT.md',
 'scripts/smoke_eval_contract.py', 'scripts/smoke_metrics_contract.py', 'src/eval/...']
Non-governed paths are allowlisted and do not block official status.
```

Proved: **`model.forward_calls == 0`** at refusal · no target directory · no temp sibling. The same
request in `smoke` mode succeeds and records `governed_paths_clean: false` plus the six observed
non-governed `dirty_allowlisted` paths — the §7.1 **scoped** rule, neither a whole-repo-clean
requirement nor a permissive "dirty is fine".

## 9. Regression and scope

- ✅ A1b's **21 metric-contract cases still PASS, exit 0**.
- ✅ `src/eval/metrics.py` unchanged by A2a (diffstat still the A1b `+124/−20`).
- ✅ No PlantSeg dataset, split file, or checkpoint accessed; no training, no backward pass, no GPU.
- ✅ **No import side effects** — instrumented `import src.eval`: **0** `subprocess`/`makedirs`
  calls, and `src.eval.artifacts` is **not pulled in at all** (`RunMeta`/`DatasetMeta` are pure
  dataclasses living in `evaluate.py`, so the package exposes them without importing any
  Git/filesystem-touching module). `src.data` and `configs.data` are never imported.
- ✅ **`builtins.open` instrumented** for the entire smoke: 4 paths opened, **zero** under any
  PlantSeg location — direct evidence for check 32 rather than an inference from imports.
- ✅ Artifacts written only under a system temp directory, removed afterwards; the repository was
  never used as an artifact target.

## 10. Verdict

## **A2a COMPLETE — CORE CONTRACT PROVEN**

## 11. Follow-ups for A2b

Real FP32 student construction · checkpoint loading (`strict=True`, `map_location`, `num_classes==116`
assertion) · a `PlantSegDataset` wrapper emitting `(image, target, image_id, clean_image_id,
manifest_index)` · the split manifest built from the real ordered stems · capped **validation** smoke
· production CLI with the test-split guards. `macc_from_confusion` / `dice_from_confusion` are now
exported from `src.eval`; the artifact writer stays importable only from `src.eval.artifacts`.

---

_Files created: `src/eval/evaluate.py`, `src/eval/artifacts.py`, `scripts/smoke_eval_contract.py`,
this report; `src/eval/__init__.py` edited for exports. Nothing staged or committed._
