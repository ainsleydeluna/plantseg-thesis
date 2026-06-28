# Updated Chapter 3 — Sync Audit (read-only)

_Generated: 2026-06-27. Read-only audit; no code/config/doc edits, no commit._

## Authoritative source status
- **Authoritative methodology = `docs/reference/ch3.pdf`.** Older Ch3 files, playbooks, `context.md`, and
  earlier notes are **secondary** unless they match this ch3.
- **⚠ The "updated" ch3.pdf is unchanged on disk.** It is byte-identical to git HEAD (`git diff --quiet HEAD`
  passes; last commit `70d4920 init thesis workspace`) and its extracted text is identical to the version
  previously audited (0 diff, 19,630 words). SHA256 `8a558e8222e536d298cd330ec757fc37f988d5d37d211f8d8dc4ec952bfe1950`.
  → No new methodology text landed; this audit is a **consistency re-confirmation**, not a diff to a changed spec.
- Method: re-extracted ch3.pdf via `pdftotext -layout`, diffed against the prior extraction, and grep/read-checked
  every repo file under `configs/ src/ scripts/ reports/ docs/`.

## PASS / FAIL summary
**PASS** — the repo is consistent with the current authoritative ch3. **0 conflicts.** Standing items are
properly documented: 1 OPEN question (disease-only/background convention) and a set of *documented-but-not-yet-
implemented* future blockers (teacher model, corruption suite, training loop, quant prepare/convert). None are
ch3 conflicts.

---

## 1. Authoritative file naming — CONSISTENT
- ch3.pdf is authoritative; repo source-tags use `[ch3]` → ch3.pdf (`IMPLEMENTATION_CONTRACT.md`, `conflicts.md`).
- **No `newchap3.pdf` / `new chap 3.pdf` / `newch2` / `references_1` references** in any `.md`/`.py` (grep clean).
- `context.md` is now **secondary** (reconciliation doc; mostly mirrors ch3). Informational: it self-titles
  "CLAUDE.md" and its internal pointers (`docs/ch3-methodology.md`, etc.) don't exist — pre-existing, and it is
  **do-not-touch** per rules. No action.

## 2. E1–E7 experimental structure — CONSISTENT
- ch3 Table 3.2 stages match the repo: Teacher, E1 (FP32), E2 (+LogitKD), E3 (+CWD), E4 (PTQ←E1), E5 (QAT←E1),
  E6 (QAT←E3 head-removed, full pipeline), E7 (PTQ←E3 head-removed).
- `configs/quant.py` (`e4_from=E1`, `e5_from=E1`, `e6_from=E3 head removed`, `e7_from=E3 head removed`) and
  `IMPLEMENTATION_CONTRACT.md §(b)` agree. **No invented stage** in any script/doc.

## 3. E6-KD — CONSISTENT (contingency only; not implemented)
- ch3 specifies **E6-KD** as a pre-registered **additive contingency arm** (run only if the E3→E6 clean-test
  mIoU drop exceeds **1.0 pp**), reported alongside E6, **not a replacement** for E6.
- Repo records exactly this: `IMPLEMENTATION_CONTRACT.md §(b)` E6-KD bullet + `configs/quant.py`
  `e6kd_reduced_weights=NEED_TO_CONFIRM`. **Not implemented** (correct).

## 4. Dataset & label space — CONSISTENT (+1 OPEN, documented)
- Empirical **`num_classes=116`**, mask values **0–115** (background 0 + diseases 1–115; `mask=category_id+1`)
  — `reports/dataset_report.md`, `configs/data.py`, contract. The ch3 "0–114 / provisional 116" wording is
  reconciled empirically toward **116** (`conflicts.md #1 = RESOLVED`).
- **Disease-only / background convention remains `NEED_TO_CONFIRM`** (`open_questions.md #2`): background=index 0
  empirically, but not explicitly named in dataset files; `reduce_zero_label` **not used**. Carefully documented
  across contract / metrics / reports. → OPEN, correctly flagged.

## 5. Preprocessing — CONSISTENT
- Resize long-side→512 (bilinear image / nearest mask), symmetric pad to 512² (image `[124,116,104]`, mask `255`),
  scale `[0,1]`, ImageNet norm `mean[0.485,0.456,0.406]/std[0.229,0.224,0.225]`, float32 CHW image / int64 HW mask,
  `ignore_index=255`. In `configs/data.py` + `src/data/transforms.py`. Matches ch3.
- **EXIF-aware loading** recorded as an empirical implementation requirement (`exif_transpose` on **images only**,
  never masks) — `transforms.core_preprocess`, contract dataset-facts, `dataset_report.md`. Consistent.

## 6. Augmentation — CONSISTENT (exclusions); 1 partial-impl note
- **Exclusions match ch3**: `configs/augment.py excluded=(brightness, contrast, blur, noise, jpeg)`; `transforms.augment`
  implements only joint hflip/vflip/rotation (mask fill 255) + image-only hue/sat. No corruption-test leakage.
- **Partial-impl note (not a conflict):** ch3's training aug also includes **multi-scale random-resized-crop**
  (scale [0.75,2.0], `cat_max_ratio` 0.95). It is present in `configs/augment.py` but **intentionally not wired
  into the current dataloader** (documented scope decision in `transforms.py`). To fully match ch3 before training,
  RRC should be added — tracked as forward work, not a ch3 conflict.

## 7. Corruption robustness — CONSISTENT (documented; suite not yet coded)
- **5 corruption types match ch3**: motion blur, Gaussian noise, JPEG compression, brightness variation, fog.
  **No "shot noise" anywhere** in the repo (grep clean) — nothing outdated to flag.
- **Severity**: 1–3 for summary/inferential, **4 descriptive-only**, **5 excluded** — contract §(f).
- **Stage handling per ch3**: mIoU-C/RPD for **E1–E7 + SegNeXt-B teacher** (teacher descriptive reference);
  **E2 included** to decompose the distillation contribution; **teacher excluded** from rCD and from all
  inferential testing; one inferential robustness test **E1 vs E6 on per-image mIoU-C**. All recorded in the contract.
- **Status:** the corruption generation/eval **code is not implemented yet** (no `imagecorruptions`/corruption module).
  Documented-but-unimplemented → future blocker, not a conflict.

## 8. Student model — CONSISTENT
- `MobileNetV3-Large` (quantizable TorchVision) + **LR-ASPP-style** head, native **Hard-Swish/ReLU retained**,
  **no ReLU6 substitution**, output **`[B,116,512,512]`**. Verified in `src/models/student.py` +
  `reports/student_forward_smoke.md` (B10 PASS: OS8 tap idx6/40ch, OS16 C5 tap idx15/160ch, ~2.93M params).

## 9. CWD — CONSISTENT
- Student **C5 = 160 ch** (B10-verified), teacher **stride-16 MSCAN-B Stage-3 = 320 ch** (LOCKED per ch3/B3),
  **160→320 training-only** bias-free 1×1 projection, **removed before E6/E7** via state_dict editing.
  `src/distill/cwd_projection.py` + `reports/cwd_projection_smoke.md` (B12 PASS, 51,200 params).

## 10. Logit KD & losses — CONSISTENT
- **T=4** — `src/training/losses.logit_kd_kl(T=4)`, `configs/distill.py T_logit=4`, contract.
- **`λ_logit` is NOT fixed**: chosen by validation sweep over {0.25,0.5,1,2,4} (reused unchanged in E3), reported
  in Ch4 → recorded as **`NEED_TO_CONFIRM`** everywhere (distill.py, losses docstring, contract). No inconsistency.
- Supervised `L_CE + L_Dice` (equal weight); CE √inverse-freq class weights, Dice soft smoothing 1e-5, ignore 255
  — `losses.py` matches ch3 B5.

## 11. Quantization — CONSISTENT (scaffold only; not run)
- **QNNPACK + eager-mode `torch.ao.quantization`** (`configs/quant.py`). PTQ static INT8 (~128 calib, seed 42);
  QAT SGD lr 3e-4 cosine ~15 ep; per-channel symmetric INT8 weights / per-tensor asymmetric UINT8 activations;
  Conv-BN-ReLU fusion; Hard-Swish/Hardsigmoid standalone.
- **No KD during standard E6 QAT** — `configs/quant.py distillation_during_qat=False`; KD only in the separate
  **E6-KD contingency** (item 3). Matches ch3.
- **Status:** quant prepare/convert **not run**; `src/quant/` is an empty placeholder package. Future blocker.

## 12. Repo file classification

### CONSISTENT — no change needed
- `configs/`: `data.py`, `augment.py`, `model.py`, `distill.py`, `quant.py`, `loss.py`, `e1_student.py`, `teacher_finetune.py`
- `src/`: `seeds.py`, `data/{__init__,dataset,transforms}.py`, `models/{__init__,student}.py`,
  `eval/{__init__,metrics}.py`, `training/{__init__,losses}.py`, `distill/{__init__,cwd_projection}.py`
- `scripts/`: `verify_plantseg_dataset.py`, `smoke_dataloader.py`, `smoke_student_forward.py`,
  `smoke_losses_metrics.py`, `smoke_cwd_projection.py`
- `reports/`: `dataset_report.{md,json}`, `dataset_location_log.md`, `student_forward_smoke.md`,
  `losses_metrics_smoke.md`, `cwd_projection_smoke.md`
- `docs/`: `IMPLEMENTATION_CONTRACT.md`, `conflicts.md`, `open_questions.md`, `B8_checkpoint.md`

### NEEDS PATCHING (vs current ch3)
- **None.** No file conflicts with ch3. Forward-work items (NOT ch3 conflicts, do only on your approval):
  - wire multi-scale RRC into the train augmentation to fully realize ch3 B2 (item 6);
  - resolve the disease-only/background convention (item 4 / `open_questions #2`) — needs the PlantSeg repo
    convention, not a code patch.

### DO NOT TOUCH
- `docs/reference/ch3.pdf` (authoritative), `docs/reference/reference.pdf`, `docs/reference/context.md` (secondary; rule).
- **Pending teacher-init (uncommitted, awaiting `mim` values):** `scripts/test_teacher_init.py`, `docs/teacher_init_source.md`.
- External dataset files (`C:\Users\admin\plantseg_data\plantseg`).
- Empty future-blocker placeholders: `src/__init__.py`, `src/quant/__init__.py`, `src/stats/__init__.py`.

---

## Conflicts found
**None.** ch3 is unchanged and the repo was built from it. The only standing items are 1 OPEN question
(disease-only/background) and documented-but-unimplemented future blockers — neither is a conflict.

## Recommended patch order
No ch3-sync patches are required. When you choose to advance the build (separate from this audit):
1. **Confirm your ch3 update actually landed** if you believe it changed — current `docs/reference/ch3.pdf` is
   the original (see Authoritative source status). If you re-copy a truly updated ch3, re-run this audit to diff.
2. (Optional, pre-training) add multi-scale RRC to the train augmentation (item 6).
3. Resolve disease-only/background convention from the PlantSeg repo (item 4).
4. Proceed to future blockers in existing order: teacher fine-tune (env-gated), corruption suite, training loop,
   quant prepare/convert, stats.
