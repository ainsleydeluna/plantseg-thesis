# Week 1 — Study Reviewer (Plain-English)

A beginner-friendly guide to **what happened in Week 1 and why it mattered**, so you can move into
Week 2 with confidence. This is a learning document — it summarizes work already committed; it changes no
code or data.

_Generated: 2026-06-28 · current HEAD `30ee402` · Week-1 core closed at `7a52213` · B16 dataset audit at
`e73d712` · B17 train-step smoke at `30ee402`._

---

## 1. Week 1 in one paragraph
Week 1 was **setup and proof**, not training. The goal was to make the project **reproducible** and to
**prove the dataset and all the building blocks are trustworthy** before spending money/GPU time on real
training. We pinned the exact software versions, wrote down the locked plan (the "contract"), added the
code configs and a fixed random seed, and then **exhaustively verified the PlantSeg dataset** (every image,
mask, label, and split). We also ran small "**smoke tests**" — tiny sanity checks that each component
(data loader, student model, losses, metrics, the distillation projection, and the quantization path)
runs and produces the right shapes. Week 1 ended by formally **closing the core work** and honestly listing
the stricter checklist items that are still open but don't block the next step. **No model was trained.**

## 2. Timeline (setup → G0)
Each step is a git commit; together they tell the story:

| Order | Commit | Step | Why it mattered |
|---|---|---|---|
| 1 | `70d4920` | init workspace | empty starting point |
| 2 | `a7cefef` | **environment lock** | freeze exact library versions so results are reproducible |
| 3 | `f9c5a45` | **implementation contract** | the single locked plan — every setting traced to the thesis Ch3 |
| 4 | `2b94a77` | **configs** | machine-readable settings (data, model, loss, augment, distill, quant) |
| 5 | `f458c81` | **seed utility** | seed 42 everywhere → repeatable randomness |
| 6 | `c4048c4` | dataset location + `.gitignore` | record where the (huge) dataset lives; stop it being committed by accident |
| 7 | `1fb15a0` | **dataset verification** | first full scan; resolved `num_classes=116` empirically |
| 8 | `2181543` | B8 checkpoint check | confirmed no public teacher weights exist → must train teacher in-house |
| 9 | `1bbbeee` | dataloader smoke | data loads into correct tensors |
| 10 | `764093e` | student forward smoke | the student model runs and outputs the right shape |
| 11 | `84fa62a`,`56149a9` | loss/metric smokes | losses & metrics compute correctly (incl. ignore-255) |
| 12 | `0527823` | CWD projection smoke | the distillation projection layer is shaped right |
| 13 | `1d082b2`,`82b0c6c` | QNNPACK smokes + quant-ready head | the model can be made INT8-convertible (proxy) |
| 14 | `a94af20` | run manifest | snapshot of the local environment for the record |
| 15 | `b46176c` | week-1 docs | tracked the supporting docs |
| 16 | `e73d712` | **B16 dataset audit** | the deep, file-by-file dataset proof (see §4) |
| 17 | `7a52213` | **Week-1 closeout + G0 status** | formally declared Week-1 core done; reconciled the strict G0 gate |
| 18 | `30ee402` | B17 train-step smoke | **first Week-2 task** — proved one training step works |

## 3. What each major commit means (the "why")
- **Environment lock (`requirements.lock`)** — pins exact versions (e.g. torch 2.1.0, mmseg 1.2.2). Without
  this, a future install could behave differently and silently change results. This is the foundation of
  *reproducibility*.
- **Implementation contract (`docs/IMPLEMENTATION_CONTRACT.md`)** — the "rulebook." Every number (learning
  rate, batch size, loss formula, label conventions) is written down and **traced to a source** (mostly
  Chapter 3). If a value isn't in a source, it's marked `NEED_TO_CONFIRM` rather than guessed.
- **Configs (`configs/*.py`)** — turn the contract into code-readable settings. They contain **no training
  logic** — just locked values. This separation keeps the plan auditable.
- **Seed utility (`src/seeds.py`)** — sets seed 42 across Python/NumPy/PyTorch so runs are repeatable.
- **Dataset verification + B16 audit** — the heart of Week 1. Before trusting the data, we *proved* its
  structure, counts, labels, and pairing (see §4).
- **Smoke tests** — fast, cheap checks that each part works in isolation *before* wiring them together
  (see §5).
- **Closeout + G0 status** — a formal "Week-1 is done" statement, plus an honest reconciliation of the
  stricter original gate (see §7–§8).

## 4. What B16 proved about the dataset
B16 was a 16-step, file-by-file audit (committed at `e73d712`; summary in
`reports/dataset_audit_summary.md`). It proved:
- **7,774 image–mask pairs**, split **train 5,367 / val 846 / test 1,561** — every image has exactly one
  matching mask.
- **0 silent drops** — the data loader will build all 7,774 pairs; none are accidentally skipped.
- **`num_classes = 116`** — the model predicts 116 categories.
- **Background = mask label 0**; **disease classes = mask labels 1–115** (115 diseases + background = 116).
- **`mask = category_id + 1`** — the link between the COCO annotation file's `category_id` (0–114) and the
  mask pixel value (1–115) is **proven 100%** against the per-image annotations (we decoded the actual mask
  pixels to confirm).
- **255 = padding/ignore only** — when images are resized and padded to 512×512, the padded mask area is
  filled with 255, which is *ignored* by losses and metrics. 255 never appears in the real (unpadded) masks.
- **`reduce_zero_label = False`** — because the masks already use 0=background and 1–115=diseases, we do
  **not** shift labels. (This is a common pitfall: some pipelines drop label 0; here we must not.)
- **`Metadata.csv` has 12 wrong `Index` values** (out of 7,774) — but **nothing in the training pipeline
  uses that column** (labels come from the masks), so it has **no pipeline impact**. We documented it for
  honesty.

**Why it mattered:** training on bad data wastes time and produces meaningless results. B16 means we can
trust the data completely.

## 5. What the smoke tests proved
Think of a smoke test as turning on a new appliance just to check it powers on without sparks — not a full
performance review.
- **Dataloader smoke** — pulls a real batch: image tensor `[2,3,512,512]`, mask tensor `[2,512,512]`,
  labels in the valid range. *Data feeds the model correctly.*
- **Student forward smoke** — the MobileNetV3-Large + LR-ASPP student runs and outputs `[2,116,512,512]`
  (~2.93M parameters). *The model is wired correctly.*
- **Loss / metric smokes** — CE+Dice loss returns finite numbers; gradients flow; the 255 ignore label is
  correctly excluded; mIoU is computed correctly. *The training signal and scoring are correct.*
- **CWD projection smoke** — the distillation helper layer maps the student's 160 channels to the teacher's
  320 channels (51,200 parameters). *The Week-3 distillation plumbing is the right shape.*
- **QNNPACK proxy smoke** — checks the model can be converted to INT8 (8-bit) form. The exact INT8 backend
  ("QNNPACK") isn't available on this machine, so it ran a **proxy** ("onednn") that **passed cleanly** —
  strong evidence the model is convertible, but **not** the final QNNPACK confirmation (that needs the
  pinned environment).
- **B17 train-step smoke (Week-2 Task 1, committed `30ee402`)** — the first real wiring of training: one
  batch → forward → CE+Dice loss → backward → optimizer step, repeated 3× on a fixed batch. Loss dropped
  6.12 → 4.53. *The pieces compose into a working training step.* (Still not real training — see §6.)

## 6. What Week 1 did NOT prove
Be clear about the limits — Week 1 produced **no experimental results**:
- ❌ No real **E1** training result (the FP32 student baseline hasn't been trained — 80,000 iterations).
- ❌ No **teacher** fine-tuning (the SegNeXt-B teacher hasn't been trained yet).
- ❌ No **distillation** results (E2 logit-KD, E3 channel-wise-KD not run).
- ❌ No **quantization** results (E4–E7 INT8 PTQ/QAT not run).
- ❌ No **statistics** or **Chapter 4** numbers (those come from real runs).

Week 1 proved the *setup is correct*; it did not produce *accuracy/efficiency/robustness numbers*.

## 7. Core PASS vs strict G0 (and why the gap doesn't block B17)
There are two "levels" of done:
- **Week-1 CORE = PASS** — setup, dataset proof, and all component smokes are committed and passing. This
  is enough to *start* implementation wiring.
- **Strict original G0 gate = NOT fully closed** — the original setup guide had a stricter checklist (G0 =
  the gate you pass before any training). A few of those items are still missing/deferred (see §8).

**Why the open items don't block B17:** B17 is a *wiring smoke* that runs on the **local CPU** with **no
teacher, no downloads, no GPU**. The open G0 items (platform verification, teacher weights, remote push,
etc.) are needed for the **real training runs** later, not for a tiny local sanity check. So the project is
*practically* ready for Week-2 wiring while the strict gate is *formally* still being finished.

## 8. The remaining strict-G0 gaps (explained simply)
| Gap | What it means | Why it's not blocking B17 |
|---|---|---|
| **Platform verification** | We haven't recorded/verified the actual training machine (the cloud GPU box). Only the *local* dev machine is captured. | B17 runs locally on CPU; the GPU box matters only for the real run. |
| **Teacher-init weights + load test** | The teacher's starting weights (ADE20K SegNeXt-B) aren't downloaded yet, and the load-test hasn't run (needs the special MMSeg environment). | E1 (and B17) use the *student*, not the teacher. |
| **GitHub remote / push** | The repo has no remote configured, so nothing is pushed/backed-up online. | Doesn't affect local code correctness; just backup/sharing. |
| **Persisted dataloader smoke report** | The dataloader smoke ran and passed, but its output isn't saved as a `.md` report like the others. | The smoke still passed and is committed as a script. |
| **Optional dataset download log** | We have a dataset *location* log but not a separate *download* log. | The dataset is verified; this is provenance bookkeeping. |

## 9. Glossary
- **Image–mask pair** — one photo (`.jpg`) plus its label map (`.png`), same filename stem. The mask says
  what each pixel is.
- **Mask label** — the integer stored at each mask pixel (here 0–115, or 255 for padding).
- **Background class** — pixels that aren't a disease. Here it's mask value **0**.
- **Disease class** — one of the 115 plant-disease categories. Here mask values **1–115**.
- **Ignore index 255** — a special value telling losses/metrics "skip this pixel." Used only for the padded
  border added during resizing.
- **`num_classes`** — how many categories the model predicts. Here **116** (background + 115 diseases).
- **`reduce_zero_label`** — a setting that, if `True`, drops label 0 and shifts the rest. Here it's
  **`False`** because 0 is a real class (background) we keep.
- **Smoke test** — a tiny, fast check that a component runs and gives correct shapes/values; not a quality
  measurement.
- **G0 gate** — the "gate zero" checklist that must be satisfied before any real training begins.
- **E1–E7** — the seven experiments:
  - **Teacher** — big SegNeXt-B model, fine-tuned on PlantSeg, used only as a reference upper bound.
  - **E1** — the student trained alone (FP32, no tricks). The baseline.
  - **E2** — E1 + **Logit KD** (student learns from the teacher's softened outputs).
  - **E3** — E2 + **Channel-Wise KD** (student also matches the teacher's feature patterns). The proposed
    full-precision student.
  - **E4** — INT8 **PTQ** (post-training quantization) of E1.
  - **E5** — INT8 **QAT** (quantization-aware training) of E1.
  - **E6** — INT8 **QAT** of E3 (the full proposed pipeline).
  - **E7** — INT8 **PTQ** of E3.
  - (E4–E7 form a 2×2: PTQ vs QAT, with/without distillation.)

## 10. "What I should understand before moving on" checklist
- [ ] The dataset is **proven** (7,774 pairs; 0 drops; labels 0–115; 255=pad; `num_classes=116`).
- [ ] **`mask = category_id + 1`** and **`reduce_zero_label = False`** — why we keep label 0 as background.
- [ ] A **smoke test** proves wiring, **not** accuracy.
- [ ] Week 1 produced **no training results** — only setup + proof.
- [ ] **Core PASS** vs **strict G0 open** — and why the open items don't block local wiring (B17).
- [ ] What **E1** is (FP32 student baseline) and how E2–E7 build on it.
- [ ] The teacher and quantization confirmations are **environment-gated** (need the pinned/cloud setup).

## 11. Am I ready for Week 2?
**Yes — for Week-2 *wiring* work.** The evidence supports it: the dataset is fully verified (B16), every
component smoke passes, and B17 already proved a complete train step composes and runs on CPU. You can
safely continue building the E1 training loop.

**With two honest caveats:**
1. The **real E1 training run** (and everything after) is **compute/environment-gated** — it needs the
   pinned stack, ImageNet student weights, real class weights, and the full augmentation. These are
   documented carry-forwards, not surprises.
2. The **strict G0 gate** isn't formally closed (platform verify, teacher weights, remote push). None block
   local wiring, but they should be finished before the real training run.

So: **ready to keep wiring; not yet ready to launch a real training run.**

## 12. Recommended next study step
Before more Week-2 code, **read these three in order** to ground yourself:
1. `reports/dataset_audit_summary.md` — the dataset proof in one place (then skim a couple of the per-split
   B16 reports it links to).
2. `docs/WEEK1_CLOSEOUT.md` + `reports/week1_G0_status.md` — what's done vs what's still open.
3. `reports/train_step_smoke.md` — exactly what B17 did and what it deliberately left out.

Then the natural next *implementation* step is **B18 — the E1 training-loop scaffold** (plan-gated), after
landing the carry-forwards (real CE class weights, ImageNet init, multi-scale augmentation).

---

_Study document only. No code/config/dataset changed; `docs/reference/reference.pdf` untouched (deferred);
nothing committed._
