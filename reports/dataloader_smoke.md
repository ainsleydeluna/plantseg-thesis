# B20b — Dataloader Smoke (persisted)

**VERDICT: ✅ PASS** — train/val/test dataloaders build and produce valid `[B,3,512,512]` f32 image /
`[B,512,512]` i64 mask batches with labels confined to `{0..115, 255}`. **Not training** — forward/data
pulls only; no model step, no checkpoint, no GPU, no download.

_Generated: 2026-06-30. Interpreter `C:\Users\admin\anaconda3\python.exe` (torch 2.9.1+cpu), CPU, seed 42.
Persists the strict-G0 "dataloader smoke report" hygiene item._

---

## Commands used
```
# (A) committed smoke — UNMODIFIED
PYTHONIOENCODING=utf-8 "C:/Users/admin/anaconda3/python.exe" scripts/smoke_dataloader.py
# (B) scratchpad-only gap-fill probe (read-only; NOT added to the repo)
PYTHONIOENCODING=utf-8 "C:/Users/admin/anaconda3/python.exe" <scratchpad>/dataloader_gapfill_probe.py
```

## Repo HEAD + git status
- HEAD: **`7452b894c984facf8f92fd318d65a59473d23f9f`** (`7452b89` — *add strict-G0 platform verification (B20a)*) on `master`.
- `git status --short`: only ` M docs/reference/reference.pdf` (pre-existing deferred — untouched). No dataset files modified (the dataset lives outside the repo at `C:\Users\admin\plantseg_data\plantseg` and was read-only).

## Evidence sources (kept separate)
- **[CS] committed smoke** — `scripts/smoke_dataloader.py` (unmodified), train+val batches.
- **[PROBE] scratchpad probe** — `dataloader_gapfill_probe.py` (read-only, not committed): test batch, normalization, effective `reduce_zero_label`, val determinism, train RRC label-safety + supporting train-vs-core.
- **[CODE] code inspection** — `src/data/dataset.py`, `src/data/transforms.py`, `configs/data.py`.
- **[PRIOR] prior committed reports** — `reports/rrc_augmentation_smoke.md` (B18c), `reports/dataset_audit_summary.md` (B16).

## Coverage matrix
| # | Item | Result | Source |
|---|---|---|---|
| 1 | command used | documented above | report |
| 2 | repo HEAD | `7452b89` | report (git) |
| 3 | git status | only deferred `reference.pdf` | report (git) |
| 4 | train/val/test construction | train+val [CS] PASS · test [PROBE] PASS | CS + PROBE |
| 5 | image shape/dtype `[B,3,512,512]` f32 | PASS | CS + PROBE |
| 6 | mask shape/dtype `[B,512,512]` i64 | PASS | CS + PROBE |
| 7 | normalization sanity | finite; per-ch in (-3,3) | PROBE |
| 8 | valid mask label range | non-255 ∈ [0,115] | CS + PROBE |
| 9 | values ∈ {0..115,255} | PASS (asserted) | CS + PROBE |
| 10 | 255 = ignore/pad, not raw class | 255 in batches (pad); absent from raw masks | CS (batch) + PRIOR/CODE (raw) |
| 11 | num_classes=116 | PASS | CS + CODE |
| 12 | effective reduce_zero_label=False | bg label 0 present, no remap | PROBE + CODE |
| 13 | train path uses B18c RRC | routing + RRC PASS + label-safe | CODE + PRIOR + PROBE |
| 14 | val/test deterministic core | byte-identical re-pull | PROBE + CODE + PRIOR |
| 15 | no dataset files modified | confirmed | report (git) |
| 16 | no training performed | confirmed | report |

## Key findings
**[CS] committed smoke (`scripts/smoke_dataloader.py`) → `[PASS]`**
- TRAIN: image `(2,3,512,512)` float32 · mask `(2,512,512)` int64 · non-255 labels `[0, 87, 109]` · contains 255 = True.
- VAL:   image `(2,3,512,512)` float32 · mask `(2,512,512)` int64 · non-255 labels `[0, 1]` · contains 255 = True.

**[PROBE] scratchpad gap-fill → `PASS (7/7 gates)`**
- **test_batch_shapes_dtypes**: image `(2,3,512,512)`/float32 · mask `(2,512,512)`/int64.
- **test_labels_in_range**: non-255 labels `[0, 1]`, contains 255 = True.
- **normalization_finite**: per-channel min `[-2.118, -2.036, -1.804]`, mean `[-0.810, -0.527, -0.680]`, max `[2.249, 2.429, 2.588]` — all finite.
- **normalization_band_sane**: every channel within (-3, 3), consistent with ImageNet `mean/std` normalization of a `[0,1]` image.
- **background_label_0_present**: label 0 present across train/val/test masks ⇒ no zero-label remap.
- **val_deterministic**: two independent `build_dataloader("val")` pulls are **byte-identical**.
- **train_rrc_label_safe**: `train_preprocess` output labels `[0, 1, 255]` ⊆ `{0..115, 255}` (RRC/rotation/crop introduce no new labels — nearest-neighbour mask ops).
- **[INFO, SUPPORTING, non-gating]** train_vs_core differ = True (multi-scale resize + crop active; core deterministic). *Per the B20b correction this is supporting evidence only, not a PASS/FAIL gate.*

## Required conclusions
- **train/val/test dataloaders build and produce valid batches** — ✅ (train+val [CS]; test [PROBE]).
- **images are `[B,3,512,512]` float32** — ✅.
- **masks are `[B,512,512]` int64** — ✅.
- **mask values stay within `{0..115, 255}`** — ✅ (asserted by [CS] and [PROBE]; train RRC output also label-safe).
- **num_classes = 116** — ✅ (`configs/data.py`).
- **effective `reduce_zero_label = False`** — ✅ (background label 0 survives; no remap in `transforms.py`). **Nuance:** the `configs/data.py` literal is still `"NEED_TO_CONFIRM"` — that wording is a **stale/placeholder** for the *formal naming convention only*; the **proven pipeline behavior is no label remapping (effective `reduce_zero_label=False`)**, settled on the data side per `reports/dataset_audit_summary.md` / `docs/WEEK1_CLOSEOUT.md` and re-confirmed empirically here.
- **255 is ignore/pad from preprocessing, not a raw class** — ✅. It appears in batches only from the symmetric pad / rotation fill (`transforms.py` mask pad value 255); `reports/dataset_audit_summary.md` proves 255 is **absent from the raw released masks** (raw labels are 0–115).
- **train path uses B18c true train-time RRC** — ✅. **[CODE]** `src/data/dataset.py` routes `split=="train"` → `train_preprocess(...)` (multi-scale RRC) and val/test → `core_preprocess`; **[PRIOR]** `reports/rrc_augmentation_smoke.md` shows B18c RRC passed (multi-scale, `cat_max_ratio`, 0 label violations); **[PROBE]** the train RRC output is label-safe (and differs from core — supporting).
- **val/test use deterministic core preprocessing** — ✅. **[CODE]** non-train splits call `core_preprocess` (no RNG); **[PROBE]** val re-pull is byte-identical; **[PRIOR]** `rrc_augmentation_smoke.md` "val deterministic".
- **no dataset files modified** — ✅ (read-only; dataset is external; git clean except deferred `reference.pdf`).
- **no training performed** — ✅ (data pulls + forward-free checks only; no model step, optimizer, checkpoint, or GPU).

## PASS/PARTIAL/FAIL — **PASS**
Committed smoke PASS + probe 7/7 gates PASS; all required conclusions satisfied with sources separated.

## Exact recommended next step
Commit B20b (the single report) with explicit-path staging. After that, the last environment-independent
strict-G0 hygiene item is **GitHub remote/push** (configure a remote and push `master`); the real E1 run
remains gated on GPU provisioning per `reports/platform_verify.md`.

---

## Provenance / guardrails honored
`scripts/smoke_dataloader.py` **not modified** (run as committed); the gap-fill probe stayed in the
session scratchpad (not added to the repo). No training, no download, no GPU, no checkpoint; no
dataloader/model/loss/training/config/dataset files changed; `docs/reference/reference.pdf`,
`docs/reference/context.md`, and teacher-init files untouched; nothing staged or committed; Claude memory
and `~/.claude` untouched.
