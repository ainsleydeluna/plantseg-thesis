# Dataset Code Expectation Audit (B16-0)

**Scope:** read-only audit of what the **dataset-loading code expects** — *before* validating any dataset
file. No dataset files were opened (`annotation_*.json`, `Metadata.csv`, `images/<split>`,
`annotations/<split>` were **not** read). No code/config/data was modified.

**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `b46176c` · **Status at audit:** read-only.

**Sources inspected (code + config + prior reports only):**
`configs/data.py` · `configs/augment.py` · `src/data/dataset.py` · `src/data/transforms.py` ·
`src/data/__init__.py` · `scripts/verify_plantseg_dataset.py` · `scripts/smoke_dataloader.py` ·
`reports/dataset_report.{md,json}` · `reports/dataset_location_log.md` · `.gitignore`.

---

## 0. Overall verdict — **PASS (with 5 flags, 0 load-blocking)**

The student dataloader's expectations are **internally consistent and consistent with the empirically
verified dataset** on every load-critical dimension: root path, folder-based split layout, stem-pairing,
single-channel raw masks, `num_classes = 116`, `ignore_index = 255`, and "255 absent from raw masks."
The 5 flags are latent / forward-looking (augmentation config-vs-impl divergence, the `reduce_zero_label`
open question, a `.jpg`-only glob, silent drop of unpaired files) — none block loading this dataset.

> **Central structural fact:** the **dataloader (`src/data/dataset.py`) reads only folders**
> (`images/<split>/*.jpg` ↔ `annotations/<split>/<stem>.png`). It **does not parse**
> `annotation_train/val/test.json` or `Metadata.csv`. Those files are read **only** by the *verification*
> script (`scripts/verify_plantseg_dataset.py`), not by the training data path. This distinction drives
> the answers to Q2/Q3 below.

---

## 1. What the CODE expects (the 15 questions)

Legend — **Verdict:** `CONFIRMED` = code matches empirical report · `CONSISTENT` = code matches
contract/config (not independently re-verified here) · `N/A` = premise doesn't apply to the loader ·
`FLAG` = divergence / latent risk · `NTC` = NEED_TO_CONFIRM.

| # | Question | What the code expects | Evidence (file:line) | Verdict |
|---|---|---|---|---|
| 1 | Dataset root | `C:\Users\admin\plantseg_data\plantseg`; loader checks `root.exists()` and raises if absent | [data.py:10](configs/data.py:10), [dataset.py:39-41](src/data/dataset.py:39) | CONFIRMED |
| 2 | JSON/CSV files | **Loader: none** (folders only). Verifier optionally reads `*.json` (COCO) + `Metadata.csv` if present | [dataset.py:43-48](src/data/dataset.py:43), [data.py:24](configs/data.py:24), [verify:146-164](scripts/verify_plantseg_dataset.py:146) | N/A (loader) |
| 3 | JSON entry fields | **Loader consumes none.** Verifier reads (optional) COCO `annotations[].category_id`, `categories[]`; CSV `Name,Index,Plant,Disease,Split` | [verify:151-164](scripts/verify_plantseg_dataset.py:151), [verify:336-344](scripts/verify_plantseg_dataset.py:336) | N/A (loader) |
| 4 | Image↔mask path map | `images/<split>/<stem>.jpg` ↔ `annotations/<split>/<stem>.png`, paired by **identical stem**; pair kept only if the `.png` exists | [dataset.py:43-48](src/data/dataset.py:43) | CONFIRMED |
| 5 | `images/<split>` folders | **Yes** — `root/"images"/split`, split ∈ {train,val,test} | [dataset.py:43](src/data/dataset.py:43), [data.py:22-23](configs/data.py:22) | CONFIRMED |
| 6 | `annotations/<split>` folders | **Yes** — `root/"annotations"/split` (per-pixel PNG masks; **not** the COCO `annotation_*.json`) | [dataset.py:43](src/data/dataset.py:43), [data.py:22](configs/data.py:22) | CONFIRMED |
| 7 | Image preprocessing | EXIF-transpose → RGB → keep-ratio resize long-side 512 (bilinear) → center-pad to 512² (fill `[124,116,104]`) → /255 → ImageNet normalize → float32 CHW. Train adds joint flips/rotation + image-only hue/sat | [transforms.py:51-57](src/data/transforms.py:51), [transforms.py:94-101](src/data/transforms.py:94), [transforms.py:67-91](src/data/transforms.py:67) | CONSISTENT |
| 8 | EXIF-safe load | **Yes** — `ImageOps.exif_transpose(img)` on the **image only**, before resize | [transforms.py:53](src/data/transforms.py:53) | CONFIRMED |
| 9 | Masks shielded from image-only transforms | **Yes** — mask never EXIF-transposed; hue/sat applied to image only; geometric ops applied **jointly** (correct); mask read as raw `L` indices (no RGB/palette expansion) | [transforms.py:53-56](src/data/transforms.py:53), [transforms.py:60-64](src/data/transforms.py:60), [transforms.py:84-89](src/data/transforms.py:84) | CONFIRMED |
| 10 | Resize / pad / normalize / ignore | Resize keep-ratio→512 (bilinear img / **nearest** mask); center-pad to 512² (img `[124,116,104]`, **mask 255**); /255 then ImageNet mean/std; ignore = 255 | [transforms.py:31-48](src/data/transforms.py:31), [transforms.py:96-97](src/data/transforms.py:96), [data.py:35-47](configs/data.py:35) | CONSISTENT |
| 11 | `num_classes` | **116** (output layer; valid labels 0–115) | [data.py:13](configs/data.py:13), [dataset.py:31](src/data/dataset.py:31), [smoke:34,44](scripts/smoke_dataloader.py:34) | CONFIRMED |
| 12 | `ignore_index` | **255** | [data.py:15](configs/data.py:15), [transforms.py:25](src/data/transforms.py:25), [smoke:33-34](scripts/smoke_dataloader.py:33) | CONFIRMED |
| 13 | Raw masks contain 255? | **No** — code treats 255 as **introduced at preprocessing** (pad fill + rotation fill); never assumes raw masks carry 255 | [transforms.py:46](src/data/transforms.py:46), [transforms.py:81](src/data/transforms.py:81), [data.py:15](configs/data.py:15) | CONFIRMED |
| 14 | Expected split counts | train **5367** / val **846** / test **1561** (total 7774) | [data.py:25](configs/data.py:25); [dataset_report.json](reports/dataset_report.json) `split.counts` | CONFIRMED |
| 15 | Next check (annotation_train.json) | Synthesized in §6 (B16-1) — since the loader ignores the JSON, B16-1 validates the JSON **independently** and cross-checks it against the folder-authoritative split | — | see §6 |

---

## 2. Expected dataset folder structure (per the loader)

```
C:\Users\admin\plantseg_data\plantseg\         # DATA["root"]  (must exist or loader raises)
├── images/
│   ├── train/  <stem>.jpg          ← globbed: images/train/*.jpg   (5367 expected)
│   ├── val/    <stem>.jpg          ← images/val/*.jpg              (846 expected)
│   └── test/   <stem>.jpg          ← images/test/*.jpg             (1561 expected)
├── annotations/
│   ├── train/  <stem>.png          ← mask = annotations/train/<img.stem>.png
│   ├── val/    <stem>.png
│   └── test/   <stem>.png
├── annotation_train.json           ← present, COCO; NOT read by the loader
├── annotation_val.json             ← present, COCO; NOT read by the loader
├── annotation_test.json            ← present, COCO; NOT read by the loader
└── Metadata.csv                    ← present; NOT read by the loader
```

**Folder layout is the split source of truth.** `annotation_*.json` and `Metadata.csv` are **inert to the
loader**. ("`annotations/`" the folder ≠ "`annotation_*.json`" the COCO files — same word, different
artifacts.)

---

## 3. Expected JSON / CSV files & fields

| Artifact | Read by loader? | Read by verifier? | Fields the **verifier** uses |
|---|---|---|---|
| `annotation_{train,val,test}.json` | **No** | Optional | `annotations[].category_id`, `categories[]` (COCO label-space scan only) |
| `Metadata.csv` | **No** | Optional | `Name`, `Index`, `Plant`, `Disease`, `Split` (class-name + per-image disease-id scan) |

> **The dataloader expects no JSON/CSV schema at all.** Any field-level requirement on these files comes
> from the *verification/analysis* path, not the training path. This is why Q2/Q3 are "N/A (loader)."

---

## 4. Expected image/mask path rules

- **Image discovery:** `sorted((root/"images"/split).glob("*.jpg"))` — *lowercase `.jpg` only*.
- **Mask resolution:** for each image, mask = `(root/"annotations"/split)/f"{img.stem}.png"`.
- **Pairing rule:** a sample is kept **only if** the `.png` mask exists; otherwise the image is **silently
  skipped**. If zero pairs result, the loader raises `RuntimeError`.
- **No count assertion:** the loader does **not** check that the discovered pair count equals the report's
  7774 — a missing mask would silently shrink a split without error (see FLAG-5).
- **Image decode:** `convert("RGB")` (handles grayscale/RGBA sources → 3-channel).
- **Mask decode:** `np.asarray(mask).astype(np.int64)` — preserves raw `L`/`P` indices; **no** RGB or
  palette expansion.

---

## 5. Expected preprocessing & label/ignore conventions

**Core (identical for train/val/test) — `core_preprocess` + `finalize`:**
1. `ImageOps.exif_transpose(image)` → `convert("RGB")` (image only).
2. Keep-ratio resize, long side → **512**; **bilinear** image, **nearest** mask.
3. **Center**-pad to **512×512**: image fill `(124,116,104)`, mask fill **255**.
4. Scale `/255` → ImageNet normalize `mean (.485,.456,.406)`, `std (.229,.224,.225)`.
5. Output: image **float32 CHW**, mask **int64 HW** — matches `smoke_dataloader` asserts
   `(2,3,512,512) f32` / `(2,512,512) i64`, labels ⊆ `[0,115] ∪ {255}`.

**Train-only augmentation — `augment` (uint8, pre-normalize):**
- Joint geometric: horizontal flip p=0.5, vertical flip p=0.5, rotation ±10° p=0.5 (image fill mean,
  mask fill 255, nearest).
- Image-only photometric: hue ±0.015, saturation ×[0.8,1.2], p=0.5 (mask untouched).
- Per-sample deterministic RNG `seed (SEED+idx)`; val/test never augmented.

**Label / ignore conventions the code assumes:**
- Valid class labels **0–115** (`num_classes=116`); background = index **0** (kept as a class, **not**
  reduced).
- `ignore_index = 255`, **introduced only at preprocessing** (pad + rotation fill); raw masks assumed
  255-free.
- **No `reduce_zero_label` / label remap is applied** — raw indices pass straight through (see FLAG-3 / §7).

---

## 6. WHAT PRIOR REPORTS ALREADY CONFIRMED (empirical)

From `reports/dataset_report.{md,json}` (full read-only EXIF-aware scan, 2026-06-27, **STATUS: PASS**) and
`reports/dataset_location_log.md`:

| Code expectation | Empirically verified? | Evidence |
|---|---|---|
| Root exists at that path | ✅ Yes | location_log: `Test-Path → True` |
| `images/` + `annotations/` dirs, 3 split JSONs, `Metadata.csv` | ✅ Yes | report `top_level` |
| Folder split = train 5367 / val 846 / test 1561 | ✅ Yes (folder authoritative; json & csv agree) | report `split.counts*` |
| Zero identifier overlap across splits | ✅ Yes (0/0/0) | report `overlaps_counts` |
| 7774 images, 7774 masks, 0 unpaired, 0 dup stems | ✅ Yes | report `counts` |
| Images `.jpg` only (7774), masks `.png` only (7774) | ✅ Yes | report `extensions` |
| Masks single-channel indexed, mode `L`, ndim 2 | ✅ Yes | report `masks.channel_modes/ndim` |
| Mask labels contiguous **0–115**, max=115, min=0 | ✅ Yes | report `masks.*` → supports `num_classes=116` |
| Raw masks **do not** contain 255 | ✅ Yes (`contains_255=false`) | report `note_255` |
| EXIF matters for images only (8 transposes, masks unaffected) | ✅ Yes | report `exif`, `dimensions` |
| Background = index 0 (ubiquitous 7773/7774, 80.6% px); encoding `mask = category_id+1` (99.85%) | ✅ Yes (empirical) | report `background_determination` |

**Nothing in the loader contradicts the empirical report.** The `.jpg`-only glob (FLAG-4) is safe *for this
dataset* precisely because the report shows extensions are exclusively `.jpg`/`.png`.

---

## 7. WHAT REMAINS `NEED_TO_CONFIRM` / FLAGS

| ID | Item | Detail | Severity |
|---|---|---|---|
| **NTC-1** | `reduce_zero_label` / background convention | `data.py:14` literally `"NEED_TO_CONFIRM"`. Loader applies **no** zero-reduction (keeps 0 as background within 116). If the contract later sets `reduce_zero_label=True`, the loader **and** `num_classes` would need changes. Tracked as open_questions #2. | Medium |
| **FLAG-2** | Augmentation config ↔ implementation divergence | `configs/augment.py` declares `library="Albumentations"` and a `random_resized_crop` (scale 0.75–2.0, `cat_max_ratio` 0.95). `transforms.py` implements a **numpy/PIL subset** with **no** RRC / no Albumentations (docstring says RRC is "out of scope for this loader"). Confirm whether multi-scale RRC belongs in the training loader (affects B2). | Medium |
| **FLAG-3** | No label remap | Loader passes raw 0–115 through unchanged; correct **iff** NTC-1 resolves to "keep background as class 0." Linked to NTC-1. | Medium |
| **FLAG-4** | `.jpg`-only glob | `glob("*.jpg")` won't match `.jpeg`/other case (Win glob is case-insensitive for `.jpg`↔`.JPG`, but not `.jpeg`). Empirically moot (all `.jpg`), but a latent assumption if the dataset is ever re-exported. | Low |
| **FLAG-5** | Silent drop of unpaired images | No assertion that pair count == 7774; a missing mask silently shrinks a split. Recommend an optional count check against `dataset_report.json`. | Low |
| **NTC-6** | Disease-only / background "explicit" naming | Report keeps a residual NTC: background is derived (empirically certain) but never literally named (COCO `categories` list empty). Affects disease-only mIoU convention, not loading. | Low (metrics) |
| **DATA-7** | ~0.15% mask-encoding shift exceptions | 12/7774 masks deviate from `category_id+1` (report `shift_exceptions`). Not a loader issue; note for label-integrity follow-up. | Info |

---

## 8. WHAT B16-1 SHOULD CHECK NEXT (`annotation_train.json`)

Because the loader ignores the JSON, B16-1 validates `annotation_train.json` **on its own terms** and
**cross-checks** it against the folder-authoritative split (the thing the loader actually uses):

1. **Parse & schema** — valid JSON; record top-level COCO keys (`images`, `annotations`, `categories`,
   `info`/`licenses` if present).
2. **Image-set agreement** — `len(images) == 5367`; the set of `images[].file_name` stems **==** the
   stems in `images/train/` (filename listing only, no pixel decode). Report any JSON-only or folder-only
   stems. This is the load-relevant cross-check.
3. **Category space** — `annotations[].category_id` range **0–114**, distinct == 115; record
   `len(categories)` (expected **0** per prior scan → names live in `Metadata.csv`, not the JSON).
4. **Encoding sanity** — confirm at the JSON/metadata level that `mask_value = category_id + 1` is the
   intended map (don't re-scan masks; reference `dataset_report.json`).
5. **Shift exceptions** — determine which of the 12 known shift-exception stems fall in **train**.
6. **Background convention** — check whether the JSON names an explicit background category (expected
   **not**); keep NTC-1 open accordingly.
7. **No mutation** — read-only; write only `reports/annotation_train_audit.md`.

> Note: step 2 needs a **filename-only** listing of `images/train/` (and optionally `annotations/train/`).
> That is a directory listing, **not** opening/decoding image or mask pixels. B16-1 should request explicit
> approval for that listing and must **not** read `val`/`test` artifacts.

---

## 9. Summary tables (the four required separations)

**(1) What the code expects** → §1 + §2 + §4 + §5.
**(2) What prior reports confirmed** → §6 (all load-critical expectations ✅).
**(3) What remains NEED_TO_CONFIRM** → §7 (NTC-1 reduce_zero_label; FLAG-2 augmentation; FLAG-3/4/5; NTC-6; DATA-7).
**(4) What B16-1 checks next** → §8.

_End of B16-0 audit. No dataset files validated; no files modified except this report._
