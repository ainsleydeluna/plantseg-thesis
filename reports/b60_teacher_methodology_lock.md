# B60 — Teacher methodology locks: M2, M3, M5, M11

**Type:** methodology decision record (governance only).
**Date:** 2026-09-22.
**Repository:** branch `claude/keen-curie-u4a8ig`, HEAD `5ae4ec754e9679a3c6b083ea95af817e9aa4a80d`.
**Status:** **M2, M3, M5 and M11 are LOCKED** as decisions. **None is implemented in runtime code.**
The current runtime teacher config, launcher, runner, transforms and smoke tests are unchanged.
**M4 and M12 remain OPEN. OFFICIAL TEACHER = NO-GO.**

**Approval:** explicit human instruction, "GO — RECORD THE FOUR METHODOLOGY LOCKS ONLY", which approved
the four locks proposed by the read-only decision dossier with three refinements (§5).
**Not done here:**
- no runtime or config behaviour change;
- no checkpoint download;
- no training;
- no TEST access;
- no PREREGISTRATION or historical-report edit;
- no Chapter PDF edit.

**Protected file:** `docs/reference/reference.pdf` was observed by path-level `git status` only.

> **Evidence labels.**
> **MS** = manuscript-specified ·
> **PRIMARY** = primary-source-supported (Wei 2026; Guo 2022) ·
> **REPO-UP** = upstream code/configs (pinned MMSegmentation 1.2.2; public PlantSeg repository at
> `1a3dd4d9224bcc97a5850af7dd1c423abc24eae0`) ·
> **REPO-THESIS** = this repository ·
> **MEASURED** ·
> **INFERRED** ·
> **THESIS-DERIVED** = a thesis choice adopted by this lock (formerly a PROPOSAL).

---

## 1. Evidence reviewed

| Source | What it establishes | Label |
|---|---|---|
| **Ch3 p.102**, teacher paragraph | Specifies AdamW 6e-5, wd 0.01, betas (0.9, 0.999), head lr_mult 10, poly, 40,000 iterations, batch 16, 512×512, cross-entropy. It attributes this recipe to "the Wei et al. (2026) PlantSeg repository configuration that produced the published 42.05%" and sets a "±1.5–2.0 pp" *protocol-matching* criterion | MS (the attribution and criterion are superseded here) |
| **Ch3 §C.2, E1–E3 text** | The teacher runs in eval mode online and "consumes the identical augmented input tensor used by the student" | MS |
| **Ch3 §E.2.c** | The training-augmentation recipe, its exclusion of brightness/contrast (diagnostic colour, and overlap with the brightness/fog corruptions), and its exclusion of blur/noise/JPEG (Hendrycks & Dietterich 2019: networks must not be trained on the corruption types used for evaluation) | MS |
| **Ch3 pp.128–129** | "the SegNeXt-B teacher is fine-tuned at the same crop, so matching this resolution keeps the teacher features, the student inputs … on a common spatial footing" | MS |
| **Ch3 §A, §E.2.d, §F** | The teacher is a descriptive upper-bound reference and a KD source. Wei values are contextual only. TEST is not consulted for selection. The teacher is scored for mIoU-C and RPD | MS |
| **Ch1 §D** | The teacher choice is "verified empirically during the teacher fine-tuning process before proceeding with knowledge distillation" | MS |
| **Wei et al. (2026)** | Benchmark training used SGD, lr 0.001, momentum 0.9, wd 0.0005, CE, batch 16. SegNeXt MSCAN-B = 28M params, 42.05 mIoU, 56.30 mAcc. The paper does not state which split Table 3 was evaluated on | PRIMARY |
| **Guo et al. (2022)** | Segmentation training uses "random horizontal flipping, random scaling (from 0.5 to 2) and random cropping", AdamW at initial lr 0.00006, and the poly decay policy. **No photometric augmentation is stated** | PRIMARY |
| **MMSeg 1.2.2** `configs/segnext/segnext_mscan-t_…_ade20k` | Official SegNeXt configs **define no pipeline**; they inherit `_base_/datasets/ade20k.py`. The schedule is LinearLR (start 1e-6, 0→1,500) plus PolyLR power 1.0, ending at `end=160000` with `max_iters` 160k | REPO-UP |
| **MMSeg 1.2.2** `_base_/datasets/ade20k.py` | `RandomResize(scale=(2048,512), ratio_range=(0.5,2.0))`, `RandomCrop(cat_max_ratio=0.75)`, `RandomFlip(0.5)`, **`PhotoMetricDistortion`** — a *generic MMSeg ADE20K dataset default* | REPO-UP |
| **MMSeg 1.2.2** `_base_/schedules/schedule_40k.py` | `val_interval=4000`; `CheckpointHook(interval=4000)` | REPO-UP |
| **mmcv** `rescale_size` | Using `scale=(512,512)` with `keep_ratio` gives scale factor = min(512/long, 512/short) = 512/long, so **long side = 512** | REPO-UP (source read) |
| **Public PlantSeg** `configs/_base_/datasets/plantseg115.py` | A copy of the ADE20K training pipeline, **including `PhotoMetricDistortion`**. **`val_dataloader` → `images/test`**; `test_dataloader = val_dataloader` | REPO-UP (fetched read-only from `raw.githubusercontent.com` at the pinned commit) |
| **Public PlantSeg** `configs/_base_/schedules/schedule_40k.py` | **`val_interval=10000`**, checkpoint interval 10000; SGD 0.01, PolyLR 0.9 (overridden by the SegNeXt configs) | REPO-UP (fetched) |
| **Public PlantSeg** `configs/segnext/segnext_mscan-t_…_plantseg115` | AdamW 6e-5, betas (0.9, 0.999), wd 0.01, head `lr_mult 10`; LinearLR 1,500 + PolyLR power 1.0, `begin=1500`, **`end=160000`** under a 40k schedule. ImageNet-pretrained MSCAN | REPO-UP (fetched) |
| **Thesis runtime config** (`510b212b…`) | The live teacher pipeline is the upstream family pipeline. `VAL_INTERVAL = 10000`; `test_dataloader`/`test_evaluator` active; horizon-corrected poly end 40,000 | REPO-THESIS |
| **E1 run of record** | Best VAL all-class mIoU **0.36314016580581665** at iteration 80,000, computed by `train_e1.validate` → `miou_from_confusion` (union-present) on the `core_preprocess` VAL canvas | MEASURED (B52) |
| **G20 / B59** | Teacher CUDA seam PASS. CPU: real Stage-3 tap and KD input equivalence PASS. NMF eval RNG dependence measured (random init) | MEASURED (B59 §5, §11) |

**Premise corrections recorded by this review:**
1. **PhotoMetricDistortion is not SegNeXt-specific.** It is MMSeg's generic ADE20K dataset default,
   copied into PlantSeg. Guo's paper does not state it.
2. **The runtime config's "VAL_INTERVAL = 10000, source-backed: public schedule_40k.py" refers to the
   PlantSeg repository's file.** MMSeg's own 40k schedule uses 4,000. In PlantSeg's code, validation
   runs on the TEST split.
3. **The Wei paper does not state its evaluation split.** The public code evaluates on TEST, so 42.05
   is most likely a TEST-split number *(INFERRED)*. The paper's stated SGD recipe also differs from
   the public SegNeXt configs' AdamW recipe, an inconsistency inside the upstream project.

---

## 2. LOCKED — M5: teacher classification, schedule, selection intent, readiness

### 2.1 Classification

The teacher is a **THESIS-DERIVED SegNeXt-B / MSCAN-B teacher**. Wei et al.'s **42.05 mIoU, 56.30 mAcc
and ~28M parameters are CONTEXTUAL PUBLISHED VALUES ONLY**. They are **not**:
- a reproduction target;
- an acceptance band;
- a protocol-match criterion;
- a trigger for retraining.

The ch3 "recover 42.05% within ±1.5–2.0 pp — protocol matching" rule **has no methodological authority**
from this date. Historical records that contain it are not edited. This record supersedes them.
**ADE20K initialisation is unchanged** (ch3's explicit choice; B1 in the implementation contract).

### 2.2 Schedule (LOCKED)

| Setting | Locked value | Label |
|---|---|---|
| Optimizer | AdamW | MS · PRIMARY (Guo) · REPO-UP |
| Learning rate | 6e-5 | MS · PRIMARY (Guo) · REPO-UP |
| Weight decay | 0.01 | MS · REPO-UP |
| Betas | (0.9, 0.999) | MS · REPO-UP |
| Decode-head lr multiplier | 10 | MS · REPO-UP |
| Warmup | LinearLR, 1,500 iterations, start_factor 1e-6 | REPO-UP (official SegNeXt and PlantSeg SegNeXt configs); ch3 says only "poly" — adopted as **THESIS-DERIVED** |
| Poly power | PolyLR power 1.0 | REPO-UP; adopted as THESIS-DERIVED |
| Poly end | **40,000** (full decay over the actual run) | **THESIS-DERIVED.** It matches official SegNeXt semantics (end = max_iters). PlantSeg's 40k config keeps `end=160000`, which would stop at ≈76% of base LR *(INFERRED arithmetic)* |
| Total iterations | 40,000 | MS |
| Batch size | 16 | MS |
| Crop | 512×512 | MS |
| Supervised loss | **unweighted** cross-entropy | MS (the ch3 teacher recipe names "cross-entropy loss"; the student's class weighting is student-specific) |
| Validation cadence | **every 4,000 iterations** | REPO-UP (MMSeg 1.2.2 official 40k schedule) plus MS by analogy: the E1–E3 cadence of 4,000 iterations at batch 16 is 64k samples, about 11.9 epochs. PlantSeg's 10,000 is rejected because it was TEST-selected upstream. Adopted as **THESIS-DERIVED** |
| Validation data | **VAL only; no TEST consultation** | MS (TEST policy) |

### 2.3 Checkpoint selection (intent LOCKED; operation NOT resolved)

The intended criterion is the **best VAL all-class mIoU** (REPO-THESIS; the ch3 rule for the FP32 stages,
applied by analogy). **Its final operational behaviour is subject to M12**, which cannot be settled
before M4. Three reasons:
- NMF evaluation draws RNG (B59 §5 A);
- validation every 4,000 iterations exposes selection to that variation 10 times;
- `IoUMetric` uses `torch.histc`, which warns as nondeterministic (B59 §11).

**M12 is not resolved by this record.**

### 2.4 Readiness rule (LOCKED)

- **R1 — integrity.**
  - The official preflight passes.
  - The run's own first-train and first-val determinism attestations pass.
  - The loss stays finite throughout.
  - All 40,000 iterations complete.
  - **No resume.**
- **R2 — selection isolation.** TRAIN/VAL only. **TEST is never consulted.**
- **R3 — role floor.**
  - **What is evaluated.** After M4 is locked, the selected teacher checkpoint is evaluated **once**,
    as a **controlled deterministic re-evaluation under the M4-locked evaluation rule**, using the
    repository's thesis VAL evaluator (`docs/EVALUATION_CONTRACT.md` §7.2).
  - **The floor.** The teacher's VAL all-class mIoU must be **strictly greater than** the E1
    run-of-record VAL all-class mIoU, **0.36314016580581665**. **No additional margin.**
  - **Also reported:** the absolute teacher–E1 VAL mIoU gap, and the teacher's disease-only VAL mIoU.
  - **What R3 is not.** The same VAL partition took part in checkpoint selection, so R3 is **not
    independent** and **not a fresh unbiased estimate**. It is an **operational competence floor, not
    an inferential comparison**. It enters no Holm family, no bootstrap, and no thesis claim of teacher
    superiority.
  - **Basis:** MS, the teacher's role as a high-capacity KD source and upper-bound reference.
- **R3 failure:** **STOP and escalate** as a methodology question. None of the following:
  - automatic retraining;
  - hyperparameter search;
  - relaxing the acceptance band;
  - TEST inspection.
- **R4 — KD readiness.** On the trained teacher, verify:
  - Stage-3 is 320 channels at stride 16 (the B59 §5 B instrument);
  - the teacher/student input-equivalence guard passes (the B59 §5 C instrument).

---

## 3. LOCKED — M2: teacher training augmentation

**Decision.** Teacher fine-tuning uses the **same semantic training-augmentation recipe as the E1–E3
student pathway**:

| Element | Locked value | Label |
|---|---|---|
| Rotation | ±10°, probability 0.5, **before** the crop | MS (ch3 §E.2.c, student pathway); applied to the teacher as THESIS-DERIVED |
| Rotation fill | image: ImageNet-mean equivalent; mask: 255 | MS |
| Crop | 512×512 | MS; the crop concept is PRIMARY (Guo) |
| Crop rejection | cat_max_ratio = **0.95** | MS (student); upstream 0.75 rejected |
| Horizontal flip | probability 0.5 | MS; the flip concept is PRIMARY (Guo) |
| Vertical flip | probability 0.5, independent of the horizontal flip | MS (student); applied to the teacher as THESIS-DERIVED |
| Photometric | image-only: hue ±0.015 and saturation [0.8, 1.2], **jointly** with probability 0.5 | MS (student); applied to the teacher as THESIS-DERIVED |
| Excluded | **no** brightness, **no** contrast, **no** blur, **no** noise, **no** JPEG. `PhotoMetricDistortion` is therefore **not** part of the locked teacher recipe | MS (ch3 §E.2.c exclusions and the Hendrycks & Dietterich contamination rule; the teacher is scored for mIoU-C/RPD) |
| Loss | unweighted CE (unchanged, M5) | MS |

**Source-labelling rule.** Guo et al. (2022) directly support **only** the SegNeXt-family concepts they
state: horizontal flip, random scaling and random cropping. The following are **manuscript-specified for
the student pathway and thesis-derived for the teacher**. Guo does not specify them:
- rotation and vertical flip;
- hue/saturation;
- cat_max_ratio 0.95;
- the exact probabilities and ranges.

**What the lock requires, and what it does not.** The scientific lock is **semantic parity** with the
student augmentation. It does **not** require:
- identical code;
- identical random draws;
- shared RNG state.

A **shared transform implementation is preferred later**, provided verification proves semantic parity
and it prevents drift. That is an implementation choice for step H (§9), not part of this lock.

---

## 4. LOCKED — M3: teacher training scale and evaluation scale

**Train scale.** Aspect-ratio-preserving **long-side** scaling: **long_side = 512 · r, with
r ~ Uniform[0.75, 2.0]**. It is applied to the **unpadded source image before** the remaining training
transforms, and training then crops or pads to 512×512 under the M2 semantics.
- MS: the ch3 §E.2.c student scale range and anchor, applied to the teacher as THESIS-DERIVED.
- Natively expressible in MMSeg: `rescale_size` with `scale=(512,512)` and a ratio range gives long-side
  semantics (REPO-UP, source read).

**Clean VAL evaluation (unchanged).**
- Aspect-ratio-preserving long side = 512.
- Thesis-defined padding to 512×512.
- Scored by the repository's thesis evaluator.

**Historical only.** The upstream short-side SegNeXt/ADE20K training pipeline (`RandomResize(scale=(2048,
512), ratio_range=(0.5, 2.0))`) is **not** the locked thesis teacher preprocessing. It remains upstream
context only.

---

## 5. LOCKED — M11: development data isolation

**Which runs.** The development and training data roots for the **official teacher**, **E2** and **E3**
are physically staged with **TRAIN and VAL only**.

**Where.** Within the **configured data root**, `images/test` and `annotations/test` **must be absent**.

**Preflight:**
- verifies TRAIN count = **5,367** and VAL count = **846**;
- verifies the TRAIN/VAL metadata the run needs;
- **asserts that the exact TEST split paths are absent**;
- **fails closed** if they are present.

**Never during development:** TEST is not enumerated, counted or inspected.

**Config surfaces.** The teacher config exposes **no active** `test_dataloader`, `test_evaluator` or
`test_cfg`.

**Scope of "TEST-absent".** It means the **configured, staged experiment data root**. It does **not**
mean proving that no unrelated file named "test" exists anywhere on the host.

**Official TEST integrity** (1,561 rows, manifest identity) is checked only after the final TEST unlock,
by the evaluation contract's guards (`docs/EVALUATION_CONTRACT.md` §7, §7.2).

**Scope over time.** Forward-looking. It does not reclassify E1 (KEEP — NO RERUN), whose training path
constructed TRAIN and VAL datasets only.

**Labels:** MS (TEST policy) · THESIS-DERIVED mechanism · MEASURED precedent (the G20 payload and
canary asserted TEST absence).

---

## 6. Options considered and rejected

| Decision | Rejected option | Rationale |
|---|---|---|
| M5 | Keep the ±1.5–2.0 pp band on Wei 42.05 | Not a protocol match: the optimizer, initialisation, eval scale (short side 512, no pad vs long side 512 + pad) and split all differ. Accepting against a TEST-split number invites TEST use, and missing the band invites retuning |
| M5 | Retrain on Wei's paper recipe (SGD 1e-3) to restore the band | Contradicts ch3's explicit AdamW + ADE20K choice. Upstream paper and code disagree, and upstream selected on TEST, so exact reproduction is unattainable without TEST |
| M5 | No readiness rule | A defective teacher could feed KD |
| M5 | A paired per-image test of teacher vs E1 on VAL | Would invent an inferential test outside the registered family |
| M5 | Validation every 10,000 iterations | Source practice selected on TEST; only 4 selection points; mismatched to the student cadence |
| M5 | Validation every 2,000 iterations or finer | No source; chases noise |
| M5 / M12 | Final-iteration checkpoint | Belongs to M12 after M4; not adopted here |
| M2 | Full `PhotoMetricDistortion` (current runtime) | Breaks ch3's contamination rationale for a model scored on the brightness/fog corruptions. It is a generic MMSeg default, not a SegNeXt requirement |
| M2 | SegNeXt geometry + hue/saturation | No vertical flip, no rotation, cat_max 0.75, so the teacher trains off the KD-input distribution |
| M2 | Geometry only | Guo-faithful for photometrics, but no hue/sat exposure, which is a KD-input gap with no robustness benefit |
| M3 | SegNeXt short-side scaling | Clean evaluation (long side 512) sits near r ≈ 0.75 of the teacher's train range for 4:3 images, and further off for elongated images |
| M3 | Long side with Guo's range [0.5, 2.0] | Wider than the student's distribution; no advantage |
| M3 | Evaluate the teacher at short side 512 | Breaks the thesis evaluation parity (one canvas for all models) |
| M11 | Count TEST filenames at preflight | Supplies no teacher-readiness evidence and enumerates TEST during development |
| M11 | Ban enumeration only | Weaker than physical absence with a fail-closed assertion |

---

## 7. The three refinements recorded with the approval

1. **R3 wording.** R3 is a *controlled deterministic re-evaluation under the M4-locked evaluation rule*
   on the same VAL partition that took part in selection. It is an **operational competence floor**,
   **not** an independent or fresh unbiased estimate, and **not** an inferential comparison (§2.4).
2. **M2 is a semantic-parity lock.** It is not a code-identity, identical-draw or shared-RNG lock. A
   shared implementation is a preferred later implementation route, conditional on proven parity (§3).
3. **M11 scope.** The absence rule applies to the **configured, staged experiment data root**, not to
   the whole host (§5).

---

## 8. Remaining M4 / M12 status

- **M4, NMF/Hamburger control — OPEN.**
  - It needs the real-checkpoint measurement, since the B59 figures come from random-init weights.
  - R3's evaluation rule and M12 both depend on it.
- **M12, checkpoint selection under M4 — OPEN.**
  - The best-VAL intent is recorded (§2.3).
  - The operational selection behaviour is not.

---

## 9. Dependency plan (approved sequence)

| Step | Action | Status |
|---|---|---|
| A | Record these four locks | **this record** |
| B | Review, commit and push the decision record | pending approval |
| C | Obtain approval for ADE20K checkpoint readiness | pending |
| D | Download, verify provenance, run the init test, do the 150→116 key audit | pending |
| E | Measure M4 NMF behaviour on the **real** initialised teacher | pending |
| F | Lock M4 | pending |
| G | Lock M12 | pending |
| H | Implement M2/M3/M5/M11/M4/M12 together in **one** governed runtime change | pending |
| I | Freeze the new runtime hashes | pending |
| J | Run one G20-style CUDA re-canary, if required | pending |
| K | G2: batch-16 deterministic-policy VRAM | pending |
| L | Official teacher preflight (TRAIN/VAL-only data root) | pending |
| M | Explicit GO for the official teacher, followed after the run by readiness R1–R4 on VAL | pending |

Batching the implementation at step H avoids repeated runtime freezes and canaries.

---

## 10. Current runtime vs the locks — known, intentional non-conformance until step H

These are recorded, not fixed, in this phase:

| Runtime item | Current state | Locked target |
|---|---|---|
| `configs/teacher/segnext_mscan-b_…_plantseg116-512x512.py` `train_pipeline` | `RandomResize(2048,512, 0.5–2.0)`, `RandomCrop(cat_max 0.75)`, horizontal flip only, `PhotoMetricDistortion` | M2 + M3 semantics |
| same config, `VAL_INTERVAL` and checkpoint-hook interval | 10,000 | 4,000 |
| same config, `test_dataloader` / `test_evaluator` / `test_cfg` | active | none (M11) |
| `scripts/launch_teacher_finetune.py` `SPLIT_COUNTS` / `check_splits` | counts TEST filenames | TRAIN/VAL counts plus a fail-closed TEST-absence assertion (M11) |
| `scripts/launch_teacher_finetune.py` `build_provenance` | `val_interval 10000`, `augmentation_source public-plantseg-segnext-family` | locked values |
| `scripts/smoke_teacher_config.py` | asserts the old pipeline and `val_interval_is_source_backed_10000` | rewritten to the locks |
| `scripts/smoke_teacher_launch.py` | tests the TEST filename count | TEST-absence tests |
| `configs/teacher_finetune.py` (documentary metadata) | `success_criterion` = `NEED_TO_CONFIRM` | readiness rule R1–R4 — updated at H, not now |

**Consequences:**
- The **current runtime (config `510b212b…`, launcher `58575276…`) must not be used for an official
  teacher run.** The teacher is already NO-GO.
- The G20 PASS remains valid evidence for the determinism seam. It ran with `load_from=None`, a
  TRAIN/VAL-only payload, the TEST surfaces disabled, and one iteration, and it drew no conclusion that
  depends on the augmentation, scale or validation cadence.

---

## 11. Manuscript consequences (for later amendment; no chapter edited)

1. **Ch3 p.102, teacher paragraph.** Replace the "Wei … configuration that produced the published 42.05%"
   attribution, the "±1.5–2.0 pp" reproduction rule and "protocol matching … is the objective" with:
   - the thesis-derived classification;
   - Wei's values as contextual only;
   - the locked schedule, including warmup, poly power and end, and validation every 4,000 iterations;
   - readiness R1–R4.
2. **Ch3 §E.2.c.** State that the training-augmentation recipe (M2) and scale (M3) also govern teacher
   fine-tuning, with the brightness/contrast/blur/noise/JPEG exclusions justified identically.
3. **Ch3 pp.128–129,** "common spatial footing … published reference points". Qualify it: Wei's
   evaluation scale and preprocessing differ.
4. **Ch3 Table 3.2, teacher row.** Add the readiness rule.
5. **Teacher selection sentence.** Add "best VAL all-class mIoU, validated every 4,000 iterations",
   pending M12's operational rule.
6. **TEST policy.** Add the M11 development-data-isolation sentence for the teacher, E2 and E3.
7. **Ch1 §D verification sentence.** Cite R3 as the operational verification. The Ch1 wording itself may
   stay.

## 12. Future runtime consequences (step H; nothing changed now)

- **`configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py`:**
  - `train_pipeline`;
  - `AUGMENTATION_SOURCE`;
  - `VAL_INTERVAL` and the checkpoint interval;
  - the TEST surfaces;
  - provenance comments;
  - its hash changes.
- **Optional new `src/**` bridge.** Registers the student transform as an MMSeg transform, if the shared
  route is chosen and parity is proven.
- **`scripts/launch_teacher_finetune.py`:** `SPLIT_COUNTS`, `check_splits` (TRAIN/VAL plus the absence
  assertion) and `build_provenance`. Its hash changes.
- **`scripts/smoke_teacher_config.py` and `scripts/smoke_teacher_launch.py`:** assertions rewritten.
- **`configs/teacher_finetune.py`:** metadata updated.
- **The M4 implementation** (teacher runner or evaluation path, per the M4 lock).
- **New frozen hashes,** a re-canary as required, and runbook §4a updates.

## 13. Governance documents changed in this phase

- **`docs/IMPLEMENTATION_CONTRACT.md` (B1):** criterion row, new locked-parameter rows, provenance note,
  §(g) register.
- **`docs/EVALUATION_CONTRACT.md`:** new §7.2 for the R3 evaluation semantics and M11 data isolation.
- **`docs/open_questions.md`:** M2, M3, M5 and M11 marked LOCKED; M4 and M12 still OPEN; #9 and the G20
  consequence pointer updated.
- **`docs/conflicts.md`:** entry #14.
- **`docs/teacher_prep_runbook.md`:** §1, §3, §9, §11 and the open-items list, plus the M11 staging rule
  for the official run.
- **This report.**

## 14. Validation of this record (2026-09-22)

No runtime, config, `src/**` or smoke file was modified. The runtime-critical hashes match HEAD
`5ae4ec7`: `teacher_runner.py` `72206af7…`, `launch_teacher_finetune.py` `58575276…`, teacher config
`510b212b…`.

| Smoke | Environment | Result |
|---|---|---|
| `smoke_teacher_config` | pinned teacher image `sha256:cb413304…`, read-only mount, `--network none` | **PASS 92/92** |
| `smoke_teacher_runner` | same teacher image | **PASS 48/48** |
| `smoke_teacher_launch` | host student-stack Python (torch present; mmseg, mmcv, mmengine absent) | **PASS 55/55** |
| `smoke_environment` | host student-stack Python | **PASS 71/71** |
| `smoke_teacher_launch` | teacher image | 54/55. **EXPECTED / NOT APPLICABLE in the teacher image** |

In the teacher image, the only failing check is `provenance_versions_not_fabricated`. That check
assumes mmsegmentation and mmcv are absent. The teacher image intentionally installs mmsegmentation
1.2.2 and mmcv 2.1.0, and the launcher correctly reports those real versions. This is an environment
mismatch, not a runtime defect. The smoke was not weakened or edited. `smoke_teacher_launch` is
recorded as passing only in its intended host environment.

**Status: OFFICIAL TEACHER = NO-GO.**
