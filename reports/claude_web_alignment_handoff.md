# CLAUDE WEB ALIGNMENT HANDOFF — THESIS2

> **Purpose.** This is a **transfer package for Claude Web to INDEPENDENTLY VERIFY** whether the current
> repository (implementation, reports, configs) is aligned with the **thesis manuscript**. **Do not assume
> alignment.** Everything below is **"current implementation status" / "known repo + report evidence" /
> "items Claude Web must verify"** — none of it is a claim of confirmed alignment. The **authoritative
> truth is the manuscript** (`ch3.pdf` = method source of truth; `Wei (2026)` = dataset facts).
> **Claude Web MUST read the attached manuscript PDFs and the attached repo reports/source files** and
> check each item against them — do not rely on this handoff alone.
>
> Scope: computer-vision / ML / implementation / statistics / software / hardware / reproducibility
> alignment only. **No biological or medical advice.**

---

## 1. Project identity
- **Thesis (working title):** *Resource-Constrained Plant-Lesion Semantic Segmentation* — compress a
  **SegNeXt-B / MSCAN-B teacher** into a **MobileNetV3-Large + LR-ASPP student** via knowledge distillation
  (**Logit-KD + Channel-Wise-KD**) and **INT8 quantization (QAT + PTQ)**, evaluated on the **PlantSeg**
  in-the-wild plant-disease benchmark on an **accuracy ⊕ efficiency ⊕ robustness** trade-off.
- **Repo:** `C:\Users\admin\plantseg-thesis` · **Branch:** `master` · **HEAD:** `495866d` · **Remote:** none.
- **Dataset (external, git-ignored):** `C:\Users\admin\plantseg_data\plantseg`.
- **Latest relevant commits (oldest→newest):** `e73d712` B16 · `30ee402` B17 · `4eb5644` B18a · `d6b0f1a`
  B18b · `666ef38` B18c · `895e608` B18d · `9539e74` Pre-B19 · `93e57bb` B19 · `7452b89` B20a · `73aa06d`
  B20b · `67f8737` B20c · `c5e5d16` B20d · `495866d` B20e (HEAD).

## 2. Critical guardrails (apply to any follow-up work in the repo)
- **`docs/reference/reference.pdf`** is a **pre-existing dirty/deferred** file (` M`). **Do not touch, stage,
  commit, restore, or modify it.** `docs/reference/context.md` and the teacher-init pair
  (`docs/teacher_init_source.md`, `scripts/test_teacher_init.py`) are also do-not-touch.
- **Git:** explicit-path staging only — **no `git add -A`, no `git add .`, no wildcards**; commit only when
  asked; no amend/push/`--no-verify`.
- **No training / download / install / GPU** unless explicitly approved.
- **Plan-gated workflow:** inspect (read-only) → show plan → wait for `go` → act.

## 3. Manuscript sources Claude Web must read (authoritative)
**Attach these to the Claude Web conversation — Claude Web cannot see the repo.**
- **Method SoT:** `docs/reference/ch3.pdf` (when anything conflicts, **ch3 wins**).
- **Reconciliation SoT (read-only):** `docs/reference/context.md`.
- **Dataset facts:** `docs/reference/Wei (2026) - PlantSeg dataset paper (2).pdf`.
- **Secondary:** `docs/reference/ch1.pdf`, `docs/reference/ch2.pdf`; `docs/reference/reference.pdf`
  (citation list — verify, **never fabricate DOIs**).
- **Repo method docs (derived from the manuscript):** `docs/IMPLEMENTATION_CONTRACT.md` (§-traced contract),
  `docs/conflicts.md` (4 resolved cross-source items), `docs/open_questions.md`,
  `docs/updated_ch3_sync_audit.md`.
  - ⚠ **`updated_ch3_sync_audit.md` is a 2026-06-27 snapshot that PREDATES B18c/B19/B20.** Treat its
    "multi-scale RRC not wired" (item 6) and its file inventory (item 12) as **superseded** — RRC was wired
    in B18c (`666ef38`) and the E1 loop in B19 (`93e57bb`). Claude Web should flag this staleness.
- **Mechanism papers (justify mechanisms, not configs):** Howard 2019 (MobileNetV3), Shu 2021 (CWD),
  Hinton (logit KD), Jacob 2018 + Krishnamoorthi 2018 (quantization), Guo 2022 (SegNeXt), Hendrycks 2019 +
  Kamann 2020 (robustness).
- **Repo reports/code to attach as evidence:** `reports/{strict_G0_closeout_update,week1_G0_status,
  dataset_audit_summary,dataset_download_log,dataloader_smoke,platform_verify,e1_class_weights,
  e1_preprocessing_init_readiness,rrc_augmentation_smoke,imagenet_init_wiring,e1_training_loop_readiness,
  e1_train_scaffold_dryrun}.md`; `configs/{data,augment,e1_student,loss}.py`;
  `src/data/{dataset,transforms}.py`; `src/models/student.py`; `src/training/{losses,train_e1}.py`;
  `src/eval/metrics.py`.

## 4. Current implementation status (what exists — verify vs manuscript)
| Step | What | Commit | Evidence (report) |
|---|---|---|---|
| B16 | Full dataset audit (16 reports → consolidated) | `e73d712` | `reports/dataset_audit_summary.md` |
| B17 | E1 train-step smoke (wiring/grad-flow, weight=None) | `30ee402` | `reports/train_step_smoke.md` |
| B18a | CE class weights (√inv-freq, median-norm, no cap) | `4eb5644` | `reports/e1_class_weights.{md,json}` |
| B18b | Preprocessing/aug/init readiness audit | `d6b0f1a` | `reports/e1_preprocessing_init_readiness.md` |
| B18c | **True train-time multi-scale RRC** wired into train path | `666ef38` | `reports/rrc_augmentation_smoke.md` |
| B18d | **ImageNet student-init** wiring | `895e608` | `reports/imagenet_init_wiring.md` |
| B19 | **E1 training-loop scaffold** + CPU dry-run PASS | `93e57bb` | `reports/e1_train_scaffold_dryrun.md` (+ `…training_loop_readiness.md`) |
| B20a | Platform verification (verdict PARTIAL) | `7452b89` | `reports/platform_verify.md` (+ `scripts/verify_env.py`) |
| B20b | Persisted dataloader smoke | `73aa06d` | `reports/dataloader_smoke.md` |
| B20c/B20e | Strict-G0 closeout update / mark download-log DONE | `67f8737`/`495866d` | `reports/strict_G0_closeout_update.md`, `reports/week1_G0_status.md` |
| B20d | Dataset download/location/provenance log | `c5e5d16` | `reports/dataset_download_log.md` |

> Not yet implemented (documented future blockers): teacher fine-tune (B1, env-gated), E2/E3 distillation,
> E4–E7 quantization prepare/convert, corruption suite, statistics. Claude Web should NOT treat these as
> done.

## 5. Dataset and preprocessing facts (current implementation — verify vs `ch3`/`Wei`)
- **Root:** `C:\Users\admin\plantseg_data\plantseg` (external). **Counts:** 7,774 pairs = train **5,367** /
  val **846** / test **1,561**; 0 silent drops. Folder+stem is the loader's split source of truth (JSON/CSV
  not parsed for membership).
- **Labels:** masks single-channel, values **0–115**; **background = 0**; **diseases = 1–115**;
  **`num_classes = 116`** (`mask = category_id + 1`). **255 = preprocessing pad/ignore only** — absent from
  raw masks.
- **Effective `reduce_zero_label = False`** (no label remap; bg 0 kept). ⚠ **`configs/data.py` literal is
  still `"NEED_TO_CONFIRM"`** (stale placeholder for the *formal naming convention*; proven behavior is
  no-remap — see §10).
- **Train path (B18c true RRC, `src/data/transforms.train_preprocess`):** EXIF(image)→RGB →
  aspect-preserving **multi-scale resize** (long side = round(512·r), r~U[0.75,2.0]) → **rotation ±10°
  before crop** (image fill ImageNet-mean, mask fill 255) → **random 512² crop+pad, `cat_max_ratio=0.95`**
  → joint H/V flips (p 0.5) → image-only hue(±0.015)/sat([0.8,1.2]) p 0.5. **Excluded:** brightness,
  contrast, blur, noise, JPEG.
- **Val/test (`core_preprocess`, deterministic):** EXIF(image only) → keep-ratio resize long-side 512
  (**bilinear image / nearest mask**) → **center-pad** 512² (image `[124,116,104]`, mask **255**) → /255 +
  ImageNet normalize (mean `.485/.456/.406`, std `.229/.224/.225`) → float32 CHW / int64 HW.
- **Absent-class caveats:** class **41** has no **test** support; class **68** has no **val** support
  (both present in train) → per-class IoU must NaN/skip for those.
- Evidence: `reports/dataset_audit_summary.md`, `reports/dataloader_smoke.md`,
  `reports/rrc_augmentation_smoke.md`, `configs/{data,augment}.py`, `src/data/{dataset,transforms}.py`.

## 6. Model / student facts (current implementation — verify vs `ch3 §(e)`)
- **Backbone:** `torchvision.models.quantization.mobilenet_v3_large` (**quantizable**, `dilated=True` →
  **OS16**), truncated `features[0:16]`. Native **Hard-Swish/ReLU** retained (no ReLU6 substitution).
- **Head (LR-ASPP, additive class-logit fusion — NOT concat):** high branch (C5/OS16) 1×1 **Conv-BN-ReLU→256**
  × (avgpool→1×1→sigmoid) multiply → upsample to OS8; **low skip (OS8) 1×1→`num_classes`**; upsampled high
  branch 1×1→`num_classes`; the two summed → bilinear upsample to full res. (Matches contract §(e) and
  `conflicts.md` #3 = ch3 "high-branch-only 256-ch" topology.)
- **Output:** `[B, 116, H, W]`; **~2,933,688 params**. Taps: OS8 idx6 (40ch), OS16 C5 idx15 (160ch).
- **ImageNet init (B18d):** `build_student(pretrained=False)` → random, no download (dry-run default);
  `True`/`"IMAGENET1K_V2"`/`"torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"` → IMAGENET1K_V2
  (cache-or-download). **Real E1 init** = the config string in `configs/e1_student.py["init_weights"]`.
- **No architecture change** is proposed here; verify topology/channel counts vs `ch3`.
- Evidence: `src/models/student.py`, `reports/imagenet_init_wiring.md`, `docs/IMPLEMENTATION_CONTRACT.md §(e)`.

## 7. Loss / class-weight facts (current — verify vs `ch3 §C.1 / B5`)
- **Supervised:** `L_sup = L_CE + L_Dice` (equal weight). **CE:** class-weighted **√ inverse-frequency**,
  weight = `sqrt(total_train_pixels / (num_classes · pixels_in_class))`, over non-255 pixels,
  **median-normalized, no cap**, `ignore_index=255`. **Dice:** soft on softmax probs, per-class macro over
  classes **present in batch**, smoothing **1e-5**, unweighted, validity-masked.
- **Class-weight artifact:** `reports/e1_class_weights.json` — length-**116**, index-aligned to class id
  0–115, **all finite** (min 0.0370 = bg class 0, median ≈1.0, max 9.5915 = class 42); **background weighted
  as a normal CE class** (not excluded). Computed over raw train masks.
- Evidence: `src/training/losses.py`, `configs/loss.py`, `reports/e1_class_weights.md`.

## 8. E1 training-loop scaffold facts (current — verify vs `ch3 B2 / §(f)`)
- **File:** `src/training/train_e1.py` (CPU dry-run PASS in B19).
- **Iteration-based** loop with a **cycling** train dataloader (not epoch-based); **80,000** iters horizon.
- **Optimizer:** SGD lr **1e-2**, momentum **0.9**, weight_decay **1e-4** (`configs/e1_student.py`).
- **Scheduler:** **per-iteration `PolynomialLR(total_iters=80000, power=0.9)`** (LambdaLR fallback), stepped
  once per iter after `optimizer.step()`.
- **Validation:** accumulates **ONE confusion matrix** over the val set, computes mIoU once;
  **checkpoint selection = best all-class val mIoU**; **disease-only val mIoU logged as PROVISIONAL
  secondary** (background-0 exclusion).
- **Safety gate:** real 80k run requires **`--real-run --confirm-real-run`**; default is dry-run.
- **Dry-run:** `pretrained=False`, few CPU iters, **checkpoint written only to a temp/out-of-repo dir**
  (hard-guarded; never in the repo).
- Evidence: `src/training/train_e1.py`, `src/eval/metrics.py`, `reports/e1_training_loop_readiness.md`,
  `reports/e1_train_scaffold_dryrun.md`.

## 9. Environment & strict-G0 status (current — verify reproducibility expectations)
- **Local machine: dry-run only, NOT real E1.** `python 3.13.5`, **`torch 2.9.1+cpu` / `torchvision
  0.24.1+cpu`**, **CUDA unavailable**, 0 GPUs.
- **Pinned-vs-actual mismatch:** `requirements.lock` / contract pin **`torch 2.1.0+cu121` /
  `torchvision 0.16.0+cu121`** (Python 3.11, mmcv 2.1.0, mmseg 1.2.2, opencv 4.8.1.78, numpy 1.26.4) — the
  GPU training stack. Locally `cv2`, `albumentations`, `mmcv`, `mmseg` are **not installed**.
- **ImageNet backbone cache: missing** (`mobilenet_v3_large-5c1a4163.pth` absent).
- **Strict-G0 overall: ⚠ PARTIAL.** **Closed:** platform verification (B20a), dataloader smoke (B20b),
  dataset download/location log (B20d), plus B19 scaffold (dry-run PASS). **Deferred:** GitHub remote/push,
  teacher-init weights/load test.
- Evidence: `reports/platform_verify.md`, `reports/strict_G0_closeout_update.md`, `reports/week1_G0_status.md`.

## 10. Known unresolved / deferred + alignment-risk items (Claude Web must scrutinize)
- **GitHub remote/push** — deferred (no remote URL provided).
- **Teacher-init weights + load test** — env-gated (pinned MMSeg env; teacher/E2–E3 phase); `weights/` empty,
  `docs/teacher_init_source.md` fields = `NEED_TO_CONFIRM`.
- **GPU/CUDA + ImageNet cache** — required for the **real E1 run**; not available locally.
- **Disease-only reporting convention** (`open_questions.md` #2) — **still OPEN**: background = index 0
  empirically, but no literal "background" category named in the dataset; `reduce_zero_label`/disease-only
  exclusion stay `NEED_TO_CONFIRM` pending the official PlantSeg repo convention. Metrics label disease-only
  mIoU **PROVISIONAL**.
- **E1 gradient clipping** — **contract B2 (`ch3`) specifies global-norm clipping "throughout"** for the
  shared E1/E2/E3 recipe, **but no numeric `max_norm` is given in any source**, and the **B19 scaffold
  currently OMITS clipping by default** (`grad_clip_norm=None`). Claude Web must verify: does the manuscript
  give a clip value, and must the real E1 run enable global-norm clipping?
- **`reduce_zero_label = NEED_TO_CONFIRM`** (stale placeholder in `configs/data.py`) vs **proven effective
  `reduce_zero_label = False`** (no remap; verified in `dataloader_smoke.md` / `dataset_audit_summary.md`).
- **Augmentation library naming/dependency** — `ch3 §D` and `configs/augment.py` name **Albumentations**
  (`"library": "Albumentations"`), but the **implementation is hand-written NumPy/PIL** with **no
  albumentations dependency** (`src/data/transforms.py`; albumentations is **not in `requirements.lock`**).
  Augmentation *semantics* match the contract; the *named library/dependency* does not. Claude Web must
  decide if this is a manuscript/repo wording fix.
- **Checkpoint-selection metric** — scaffold selects on **all-class** val mIoU; `ch3 B2` says "best
  validation-mIoU" without stating all-class vs disease-only. Verify intended metric. **[D1 RESOLVED
  2026-07-01: checkpoint selection + headline validation = all-class val mIoU; disease-only mIoU is
  secondary/provisional (still the per-image _inferential_ unit for Ch4); a switch needs a separate explicit
  decision. See docs/open_questions.md #2.]**
- **`updated_ch3_sync_audit.md` staleness** — 2026-06-27 snapshot; its "RRC not wired" (item 6) and file
  inventory predate B18c/B19/B20. Verify and flag.

## 11. Exact audit questions for Claude Web
1. Are the current code/reports aligned with **Chapter 3** methodology?
2. Are any implementation choices **contradicting** the manuscript?
3. Are any **manuscript statements now stale** because of the B18/B19/B20 changes?
4. Are the **dataset label conventions** consistent across manuscript, reports, configs, and code?
5. Is the **RRC** implementation thesis-aligned?
6. Is **ImageNet-init** wiring aligned with E1 methodology?
7. Is the **class-weight artifact** aligned with the manuscript formula?
8. Is the **training-loop scaffold** aligned with the intended E1 method?
9. Are **metrics and checkpoint selection** aligned with the manuscript?
10. Are there any **remaining blockers** before the real E1 run?
11. What **exact Ctrl+F** manuscript/code/report fixes are needed, if any?

## 12. Exact Claude Web prompt (paste this into Claude Web with the files attached)
> Read this handoff plus all attached manuscript/repo report files. Audit whether the current
> implementation and documentation are fully aligned with the thesis manuscript. Do not assume alignment.
> Check every major methodological decision against the manuscript: dataset, label mapping, preprocessing,
> RRC, model, ImageNet init, class weights, E1 training loop, metrics, checkpoint selection, strict-G0
> status, and real-run blockers. Produce a table with: item, manuscript claim, current
> implementation/report status, aligned yes/no/partial, evidence file/path, issue, exact fix needed. If a
> fix is needed, give Ctrl+F-able exact replacement/removal/addition instructions. Do not give biological
> or medical advice. Focus only on computer-vision, machine-learning, implementation, statistics,
> software, hardware, and reproducibility alignment.

---

_Handoff generated 2026-06-30 at HEAD `495866d`. This file documents **current repo/report status for
independent verification** — it is **not** an assertion of confirmed manuscript alignment. No code, config,
dataset, or manuscript file was modified; `docs/reference/reference.pdf` untouched; nothing staged or
committed._
