# B59 — pre-RunPod reconciliation, G20 readiness, and G20 execution (PASS)

**Type:** manuscript ↔ source ↔ repository reconciliation record, plus scratch CPU verification, plus the
G20 CUDA canary execution record (§11).
**Date:** 2026-09-21. **G20: PASS** (RunPod RTX A5000, §11). **Official teacher: NO-GO** (§12).
**Repository:** branch `claude/keen-curie-u4a8ig`, HEAD `1895641fa91cdfd3a4ac457989abf704ecc6aa79`
(parent `7576ba8`). Not staged, not committed; all Phase-3 changes are left in the working tree for review.
**Scope, as approved:**
- safe documentation reconciliation;
- one scratch-only wording fix in the G20 canary;
- scratch CPU verification in the pinned teacher image with `--network none`.

**Out of scope:** no pod, no GPU, no training, no checkpoint download, no TEST access, no
methodology decision, and no experiment-behaviour change.
**Protected file:** `docs/reference/reference.pdf` was observed only through path-level `git status`
(` M`). A Phase-1 `ls -la docs/reference` also stat'ed it — filenames kept, the directory's aggregate
block count printed, no per-file metadata recorded. That slip was disclosed at the time and not repeated.

> **Classification labels used throughout:**
> **MS-ERR** = MANUSCRIPT ERROR ·
> **IMPL-DEV** = IMPLEMENTATION DEVIATION ·
> **DOC-GAP** = DOCUMENTATION GAP ·
> **SRC-DETAIL** = SOURCE-DERIVED IMPLEMENTATION DETAIL ·
> **METH-OPEN** = METHODOLOGY DECISION OPEN ·
> **MEASURED** = VERIFIED MEASURED BEHAVIOUR.

---

## 1. Baseline (Phase 1, verified live)

| Item | Observed |
|---|---|
| Branch / HEAD | `claude/keen-curie-u4a8ig` / `1895641f…`; upstream `origin/claude/keen-curie-u4a8ig`, 0 ahead / 0 behind (`git ls-remote` confirms remote tip `1895641f…`) |
| vs `origin/master` | 28 ahead / 0 behind (`9a02bac`) |
| Index | empty |
| Governed paths | clean at Phase-1 start |
| Ancestry | `885523a` (E1 safety floor) and `0bb6996` (G20 runtime source) are both ancestors of HEAD |
| Runtime-critical hashes | identical at `0bb6996`, at HEAD, and in the worktree: `src/training/teacher_runner.py` `72206af7…2d0e` · `scripts/launch_teacher_finetune.py` `58575276…c7e` · teacher config `510b212b…1ab5` |

---

## 2. E1 ruling — **E1 KEEP — NO RERUN**

Phase 2 looked for a training-affecting defect in E1 seed 42 (B52) and found none:
- **Labels and class count:** 116 classes, background 0, `reduce_zero_label=False`.
- **Ignore handling:** CE `ignore_index=255`; Dice validity-masked on both probabilities and one-hot.
- **Objective:** weighted CE + soft Dice over GT-present-in-batch classes, smoothing 1e-5, unweighted
  Dice — matches ch3.
- **Optimiser and schedule:** match ch3.
- **Checkpoint metric:** all-class union-present validation mIoU. `n_eligible_classes = 116` at all
  20 validations, so the zero-GT class 69 never shifted the denominator or the ranking.
- **Augmentation semantics:** match ch3. Only the padding order (§4, A2) and the library name (A3) differ.

No metric recomputation or checkpoint re-evaluation is needed to *repair* anything. The one open E1
item is a reproducibility attestation, not a repair: evaluate `e1_student_best_iter80000.pt` twice on
VAL (B52 N12). The E1 checkpoint (`cf0879f7…6a03`, B52 §8.1) was not touched.

---

## 3. VAL-only final-canvas zero-disease check — VERIFIED

**Instrument:** `src.data.dataset.PlantSegDataset('val')` → `src.data.transforms.core_preprocess`. This is
the exact E1 validation / evaluator canvas path. Only the VAL split was constructed; TEST was not listed,
opened or hashed. Instrument and output are in Appendices A and B.

| Measure | Result |
|---|---|
| VAL images | 846 (manifest sha256 `92de2a96…72b2f`) |
| Final 512 canvases with **zero disease pixels** | **0 / 846** |
| Disease lost by the nearest-neighbour resize | 0 |
| Raw masks with zero disease pixels | 0 |
| Disease class count per canvas | exactly one, unchanged from the raw mask (846/846) |
| Background present on the canvas | 846/846 |
| EXIF dimension mismatch after transpose | 0 |
| Smallest canvas lesion | **259 px** (`potato_early_blight_9`); 3 images < 500 px; median 24,270.5 |

**Repository non-interference:** `git status --porcelain --ignored=matching --untracked-files=all` had the
same inventory hash before and after the run, `d1260cec…56e9d6` (21 lines).

**TEST is NOT verified on the canvas.** Handling a canvas-empty TEST mask is METH-OPEN (M10).

---

## 4. Reconciliation table (Phase 2, updated with Phase-3 evidence)

The "G20 / E1 rerun / Teacher / Fix / Approval" columns mean, in order:
- blocks the G20 canary;
- E1 rerun required;
- blocks the official teacher;
- a safe fix was made now (Phase 3);
- methodology approval is needed.

### A. E1 and data

| ID | Issue | Manuscript / source | Live repository | Class | G20 | E1 rerun | Teacher | Fix | Approval |
|---|---|---|---|---|---|---|---|---|---|
| A1 | E1 rerun necessity | ch3 E1 recipe | Every recipe value matches (B52); no training-affecting defect | MEASURED | N | **N** | N | – | N |
| A2 | Augmentation order | ch3 §E.2.b "applied after padding"; §E.2.c rotation before crop, scale [0.75, 2.0], crop 512, cat_max 0.95 | Scale the unpadded image to long side 512·r → rotate → random crop with pad → flips → hue/sat | IMPL-DEV (non-material) + MS-ERR (wording) | N | N | N | contract B2 order row | manuscript wording |
| A3 | Albumentations | ch3 Table 3.4 | NumPy/PIL; `configs/augment.py` already relabelled | IMPL-DEV (library only) + DOC-GAP | N | N | N | contract `:173`, `:390`, `:766` | manuscript wording |
| A4 | A40 vs RTX 4090 | ch3 locks no GPU | E1 on an A40, Secure Cloud. The 4090 was measured unable to run E1 (20.667 GiB peak, OOM at iteration 2). **No repo text called A40 a deviation**; stale "RTX 4090" remained in contract `:393`, runbook §7, `open_questions` #7, PREREGISTRATION U8 | DOC-GAP | N | N | N | contract, runbook, `open_questions` (PREREG not edited) | N |
| A5 | E1 unclipped | ch3 p.104 attaches clipping to "the distillation stages" | D2/D-A called unclipped E1 a "documented deviation"; the contract B2 row put clipping in the shared recipe | DOC-GAP | N | N | N | contract B2 row + D-A note, `open_questions` D2 | N |
| A6 | Class-weight basis | ch3: valid train pixels; resolution not stated | Raw native-resolution masks, median-normalised, no cap | SRC-DETAIL | N | N | N | – | N |
| A7 | E1 nondeterminism | ch3 §D `warn_only=True` plus cross-GPU caveat | `nll_loss2d` nondeterministic, 2–3 ULP; checkpoint not regenerable; N12 attestation pending | MEASURED | N | N | N | – | later manuscript edit (G6) |
| A8 | Checkpoint metric | best validation mIoU | All-class, union-present; eligible count constant (116) at all validations | MEASURED | N | N | N | – | N |
| A9 | Split counts | ch3 5,442 / 778 / 1,554 | 5,367 / 846 / 1,561 everywhere | MS-ERR | N | N | N | – | N |
| A10 | Class count prose | ch3 "0–114", "115 output channels", "115 verified classes incl. background" | 116 (0 bg, 1–115, 255 ignore) | MS-ERR | N | N | N | – | N |
| A11 | Per-image eligibility | ch1 drops classes with "zero union"; ch3 drops classes with zero ground truth | EVALUATION_CONTRACT §3.2: GT-present (ch3) | MS-ERR (ch1 vs ch3) | N | N | N | `conflicts.md` #8 | N |
| A12 | Zero-disease images | ch3 does not define the case | Abort on an undefined score. **VAL canvas 0/846 (MEASURED)**; TEST canvas unverified | METH-OPEN + MEASURED | N | N | N | scope notes in EVAL contract + `open_questions` | **Y (M10)** |

### B. Teacher

| ID | Issue | Manuscript / source | Live repository | Class | G20 | E1 rerun | Teacher | Fix | Approval |
|---|---|---|---|---|---|---|---|---|---|
| B1 | ADE20K vs ImageNet init | ch3 explicitly chooses ADE20K SegNeXt-B; public PlantSeg T/L configs and Guo use ImageNet-pretrained MSCAN | ADE20K `load_from` required; no fallback | SRC-DETAIL (thesis choice; repo conforms) | N | N | lock | unchanged | only to change it |
| B2 | Wei SGD vs thesis AdamW | Wei: SGD 1e-3 / 0.9 / 5e-4 / CE / b16; ch3 misattributes AdamW to Wei | AdamW; config labelled thesis-derived | MS-ERR | N | N | framing | contract provenance note; `conflicts.md` #5 | N |
| B3 | "Protocol match ±1.5–2.0 pp" | ch3 p.102 | Stale in `teacher_finetune.py`, contract `:51`/`:135`, runbook §1/§9, `B8_checkpoint`, `open_questions` #9 | DOC-GAP + MS-ERR | N | N | Y | all relabelled `NEED_TO_CONFIRM` | **Y (M5)** |
| B4 | Schedule details | ch3 "poly, 40k" | warmup 1,500; poly power 1.0; horizon corrected to 40k; val every 10k | SRC-DETAIL | N | N | lock | recorded in contract B1 | **Y (M5)** |
| B5 | Teacher augmentation | ch3 silent; Guo: flip / scale / crop | Full `PhotoMetricDistortion` including brightness and contrast | METH-OPEN | N | N | **Y** | – | **Y (M2)** |
| B6 | Teacher train/eval scale | same-preprocessing principle | Train short side ≈ 512·r (r 0.5–2.0); eval long side 512 | SRC-DETAIL → METH-OPEN | N | N | Y | – | **Y (M3)** |
| B7 | Teacher excluded from rCD | ch3 | `RCD_EXCLUDED_STAGES=("TEACHER",)`; no stale claim found | MEASURED | N | N | N | – | N |
| B8 | Teacher eval parity | same preprocessing | Official teacher evaluation runs through the repo evaluator on the `core_preprocess` canvas (`TeacherEvalModel`) | SRC-DETAIL (conforms) | N | N | N | runbook §11 note | N |
| B9 | NMF/Hamburger RNG | ch3 silent | Eval consumes CPU RNG; outputs RNG- and batch-position-dependent (§5 A) | METH-OPEN + DOC-GAP | N | N | **Y (hard)** | runbook §3/§6 corrected | **Y (M4)** |
| B10 | Stage-3 tap | ch3 Table 3.1; Guo: stage 3 C=320 @ H/16 | **Verified on the real MSCAN-B** (§5 B) | MEASURED | N | N | closed at architecture level | – | N |
| B11 | Input double-preprocessing | ch3: identical tensor | **PASS on the live API** with negative controls (§5 C) | MEASURED | N | N | N (E2 prerequisite met) | – | N |
| B12 | ADE20K checkpoint readiness | ch3 Table 3.5 | URL / SHA / date / init test `NEED_TO_CONFIRM` | open (operational) | N | N | Y | – | download approval |
| B13 | Head swap 150→116 / key audit | runbook §5 | B55 Appendix B never run | DOC-GAP | N | N | Y | – | N |
| B14 | Teacher b16 VRAM | – | "~31 GB, A6000" was unmeasured; D30 transient applies | open (needs measurement) | N | N | Y | runbook §7 corrected | GPU approval |
| B15 | mmengine / ftfy | ch3 omits both | Contract B6 lacked both | DOC-GAP + MS-ERR | N | N | N | contract B6 rows | N |
| B16 | Teacher preflight counts TEST filenames | TEST final-only | `check_splits` counts names only; G20 skips it | METH-OPEN (minor) | N | N | minor | – | **Y (M11)** |

### C. Distillation

| ID | Issue | Manuscript / source | Live repository | Class | G20 | E1 rerun | Teacher | Fix | Approval |
|---|---|---|---|---|---|---|---|---|---|
| C1 | CWD C divisor | Shu Eq. 4: C = channels of the distilled map | Feature map C=320, logit map C=116 — correct | DOC-GAP (contract said only 320) | N | N | N | contract B3 row | N |
| C2 | E2/E3 clipping | ch3: distillation stages clipped **and** E1–E3 share an identical recipe | Preregistered pilot {1.0, 5.0}; the launcher refuses an unclipped real run | MS-ERR + METH-OPEN | N | N | N | labelled only | **Y (M1)** |
| C3 | λ sweep length / noise band | grid given; "noise band" undefined | not registered | METH-OPEN | N | N | N | labelled only | **Y (M6)** |
| C4 | E6-KD TEST trigger | ch3: clean-test E3→E6 drop > 1.0 pp | frozen in contract, stats contract §9.3, PREREG §6 | METH-OPEN (design flaw) | N | N | N | labelled only | **Y (M7)** |
| C5 | E6-KD weights | ch3: "reduced" | `NEED_TO_CONFIRM` | METH-OPEN | N | N | N | – | **Y (M7)** |

### D. Statistics

| ID | Issue | Manuscript / source | Live repository | Class | G20 | E1 rerun | Teacher | Fix | Approval |
|---|---|---|---|---|---|---|---|---|---|
| D1 | Holm family | ch3 §F.1.d = the 8; §B lists 10; **Table 3.6's E3-vs-E6 row says Holm across 8**; ch1 "ten" | Eight frozen; stats contract §0.1 had omitted the Table 3.6 row and ch1 | MS-ERR + DOC-GAP | N | N | N | stats contract §0.1 rows + clarification | N |
| D2 | Pratt z formula | ch3 p.143 | SciPy 1.11.4 Cureton adjustment; code calls SciPy | MS-ERR | N | N | N | stats contract §3 note | N |
| D3 | Wilcoxon call arguments | ch3's literal call omits `correction` / `method` | frozen: pratt / correction / greater / approx | MS-ERR (minor) | N | N | N | – | N |
| D4 | Multi-seed | ch3 §C.2 "optional" vs §D/§F "three seeds" | treated as planned | MS-ERR + METH-OPEN | N | N | N | labelled | **Y (M8)** |
| D5 | QAT checkpoint rule | ch3 E5 best-val vs §E.2.d "no validation selection" | QAT best-val + early stop | MS-ERR + METH-OPEN | N | N | N | labelled | **Y (M9)** |

### E. Robustness, efficiency, quantization

| ID | Issue | Manuscript / source | Live repository | Class | G20 | E1 rerun | Teacher | Fix | Approval |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Corruption cache | fixed seed, cached, checksummed | `MASTER_SEED` + item seeds + per-file SHA-256; 40/40 reference grid; official cache not built | MEASURED (closure) | N | N | N | – | N |
| E2 | Severity 4 / 5 | severity 4 descriptive only; 5 excluded | `corruption_protocol.json` roles; severity 5 never generated | MEASURED | N | N | N | – | N |
| E3 | CPU benchmark threads | ch3 §D: single-threaded, b1, 20 + 100 | 20 + 100 and b1 enforced; `--threads` default `None` | IMPL-DEV (default) | N | N | N | pass `--threads 1` at run time | N |
| E4 | GPU-hour / cost | ch3 §D: per stage, plus the E2/E3 per-iteration split | E1 manual only (23.02 h; $11.28 training window) | DOC-GAP | N | N | N | – | N |
| E5 | QNNPACK operators | verify before E4 | onednn proxy only; the image has qnnpack | DOC-GAP (pending) | N | N | N | – | N |

### F. G20

| ID | Issue | Evidence | Class | G20 |
|---|---|---|---|---|
| F1 | Canary wording | "no accuracy is reported" was inaccurate; **fixed** (§6) | DOC-GAP → fixed | cleared |
| F2 | Checkpoint tripwires | Patching `CheckpointLoader.load_checkpoint` on the class catches `Runner.load_checkpoint` → `_load_checkpoint` → `CheckpointLoader.load_checkpoint` (pinned mmengine `runner.py:46,2127`; `checkpoint.py` `_load_checkpoint`). `default_hooks.checkpoint=None` is popped by `register_default_hooks`. Layered guards: `load_from=None`, `init_cfg=None`, `TORCH_HOME` empty before/after, log-string check. **No coverage gap proven**, so no network-level tripwire was added | MEASURED | N |
| F3 | P1 probe | `6ebeefe9…`; arm read from the environment; P1N runs under `env -u CUBLAS_WORKSPACE_CONFIG` | MEASURED (static) | N |
| F4 | Runtime identity | §1 hashes; image `cb413304…`; image-baked source is stale, so the run needs a checkout at `0bb6996` | MEASURED | N |
| F5 | Payload | 2 TRAIN + 2 VAL files + manifest; tar has 0 TEST entries; the canary requires the TEST dirs to be absent | MEASURED | N |
| F6 | B6 "before CUDA init" vs ch3 "where applicable" | B58 §16's "contract governs" inverts the manuscript-first authority order. The TeacherRunner satisfies the stricter version anyway | DOC-GAP (minor) | N |
| F7 | **G20 executed on CUDA** | 2026-09-21, RTX A5000: all 17 pass criteria met (§11) | MEASURED | **PASS** |

---

## 5. Scratch CPU verification (Phase 3) — pinned teacher image, `--network none`

**Environment:**
- image `plantseg-teacher:local` = `cb413304…c32c2b` (the published teacher digest);
- Python 3.11.16 · torch 2.1.0+cu121 · mmseg 1.2.2 · mmcv 2.1.0 · mmengine 0.10.7 · CPU only;
- teacher config sha256 `510b212b…` (the frozen runtime value); `ham_head.py` sha256 `eb2f0963…`.

**Inputs:**
- Code: a `git archive HEAD -- src configs scripts` export, which never reads `docs/`.
- Weights: a **random initialisation** (`torch.manual_seed(0)` + `init_weights()`). No checkpoint of any
  kind was downloaded or loaded, apart from a scratch random-init `.pth` created and deleted inside the
  run.
- Real input: the G20 payload **VAL** image `apple_black_rot_28` through the student path.

**Instrument:** `cpu_verify.py` (sha256 `eceefa5c…`, Appendix C). Output
`cpu_verify_result.json` (sha256 `cb4f91fe…`, Appendix D).

**Scope:** measurement only; no policy, config or code was changed.

### A. NMF / Hamburger RNG — measured; no policy adopted

Architecture: `EncoderDecoder(MSCAN, LightHamHead)`, 116 classes, 27,618,868 parameters, `ham_kwargs =
{MD_S 1, MD_R 16, train_steps 6, eval_steps 7, inv_t 100, rand_init True}`, eval mode.

| Measurement | Result |
|---|---|
| Backbone forward repeated | bitwise equal (deterministic) |
| One eval decode-head forward consumes CPU RNG | **yes** — exactly one `torch.rand((1, 512, 16))` (state advance matched bit-for-bit) |
| 20 repeated eval forwards, RNG **not** reset | **20 distinct outputs**; max \|Δlogit\| vs run 0 = 0.137 (logit max 0.554, mean 0.113 → max relative 0.248); mean \|Δ\| 0.0083; max pairwise 0.157 |
| argmax change vs run 0 | 11.8% of 64×64 cells on average (max 16.5%); 10.7% of valid 512² pixels on average (max 13.2%) |
| RNG reset to the same seed before each forward | **bitwise identical** (1 distinct output in 5); restoring a saved RNG state also reproduces exactly |
| Same image at batch positions 0 and 1 | **different** outputs (max \|Δ\| 0.086). A single image equals batch position 0 under the same seed (3.0e-7) |
| Diagnostic attribution control, `rand_init=False` | identical parameters; outputs constant after the first forward (max \|Δ\| 0.0). **Diagnostic only, not a proposed policy** |

**Conclusion (measurement only):**
- With the inherited `rand_init=True`, teacher evaluation outputs are a function of CPU RNG state and of
  batch composition, not of the weights and input alone.
- This bears on validation mIoU, `save_best` checkpoint selection, and E2/E3 soft targets.
- The *magnitude* above comes from random-init weights with near-uniform logits. It is **not** a proxy
  for the trained teacher, and must be re-measured on the real checkpoint.
- The control policy remains **METHODOLOGY DECISION OPEN (M4)**, and it still blocks the official teacher.

### B. Real Stage-3 tap — PASS

| Check | Result |
|---|---|
| MSCAN-B stage outputs for 1×3×512×512 | [1,64,128,128] s4 · [1,128,64,64] s8 · **[1,320,32,32] s16** · [1,512,16,16] s32; blocks per stage 3/3/12/3 |
| Selected feature (`select_stride16_feature`) | backbone output **index 2 = Stage 3**, shape **1×320×32×32**, stride **16**, channels **320** |
| Semantic stage | selected tensor **bitwise equal** to `backbone.norm3`'s output (Stage-3 final norm), reshaped |
| Live adapter | `SegNeXtTeacherAdapter.feat_s16` bitwise equal to the selection; logits 1×116×64×64 |
| Negative control: Stage 3 removed | raises `TeacherArchitectureMismatch` ("found 0") |
| Negative control: wrong stride (32) | raises `TeacherArchitectureMismatch` |
| Negative control: compare against Stage 4 | not equal (shape [1,512,16,16]) — the equality check can fail |

### C. Teacher input-equivalence guard — PASS

Tolerances were fixed in the script before any result was read: input ≤ 1e-5 absolute; feature and logits
≤ 1e-4 relative; each negative control ≥ 0.1 at the input and ≥ 1e-2 relative at the feature.

**Paths compared:**
- **KD path.** The student's `finalize()` tensor goes through the live KD API: `load_frozen_teacher` →
  `build_segnext_teacher` → `mmseg.apis.init_model` → `SegNeXtTeacherAdapter` → `FrozenTeacher.forward`.
  This is the call `train_distill.py:395` makes with `model_input = img`.
- **MMSeg's own path.** The same pixels go through `SegDataPreProcessor`: BGR as `LoadImageFromFile`
  yields, then `bgr_to_rgb`, mean `[123.675, 116.28, 103.53]` / std `[58.395, 57.12, 57.375]`,
  `test_cfg size_divisor 32`.
- The NMF draw was held fixed by reseeding before each compared forward.

| Comparison | Input max \|Δ\| | Stage-3 rel. Δ | Logits rel. Δ | argmax agreement |
|---|---|---|---|---|
| **Positive:** KD path vs MMSeg path | **2.4e-7** | **2.9e-6** | **3.3e-6** | **1.000** |
| NC1 channel reversal (RGB→BGR) | 1.595 | 1.099 | 0.671 | 0.722 |
| NC2 double normalisation (student tensor re-run through `SegDataPreProcessor`) | 3.968 | 1.429 | 0.787 | 0.661 |
| NC3 RGB fed to the BGR-expecting preprocessor | 1.952 | 1.083 | 0.783 | 0.504 |

`SegDataPreProcessor` calls during the KD forward: **0**. The weights loaded through `init_model`
equal the scratch weights bitwise. **Verdict: PASS.** The KD path neither double-normalises nor reverses
channels, and each negative control is detected. No defect was found and no KD code was modified.

---

## 6. G20 canary — wording fix (scratch only)

| | |
|---|---|
| File | `…/a94ed0f0-98f0-42d1-a875-491f9376b547/scratchpad/g20_canary.py` (outside the repository) |
| Old sha256 | `ac4e14f2cb60893258d33ca885c63f3eccf71f61aab76e06a14695b4236efa54` (preserved alongside as `g20_canary.ac4e14f2.py`) |
| **New sha256** | **`68b8bd00e5e406339d6b6ed3317718315eade55017a0f134fa46cd8af76f7ad0`** |
| Diff | line 387 → 387–389, inside the `provenance["g20_canary"]["grade"]` string only |
| Old phrase | "…not checkpoint-selection evidence, no accuracy is reported from this run" |
| New phrase | "…not checkpoint-selection evidence. G20 produces no accuracy result that is used, selected, interpreted, compared, or reported as thesis evidence; any one-sample random-init validation metric is a diagnostic side effect only" |
| AST proof | Both files compile under Python 3.11.16. `ast.dump` with the single `grade` constant masked is identical (`ec11ee43…c6f` for both). `grade` occurs once; `EXPECTED_HASHES` unchanged; LF endings on both (0 CR bytes) |
| Behavioural proof | Both versions were run in the teacher image with `--network none`. Each passed the same 11 pre-CUDA gates, including the three-way `CUBLAS_WORKSPACE_CONFIG=:4096:8` view inherited from the image ENV, the image digest and the image commit. Each then failed closed at `git_head_is_runtime_commit`, as expected with no runtime checkout. Their evidence JSONL is **identical once timestamps are masked** |

**Why:** a real `IoUMetric` validation step logs a one-sample mIoU, so the old wording was false. The change
is text-only in a provenance string. Runtime behaviour is unchanged.

### 6.1 Frozen G20 references (updated)

| Item | Value |
|---|---|
| Image (by digest) | `ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b` |
| Image revision label (build commit) | `48bccc70caef4b8e252ffd11182c001f0f4368f5` |
| Runtime source | detached checkout `0bb69961dfedcb4d00ff42990a6543f5dda505ec` |
| `src/training/teacher_runner.py` | `72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e` |
| `scripts/launch_teacher_finetune.py` | `5857527622ea3dc4b12fc316a410e8144eb06ffa5199e002ad1b3c957c89ac7e` |
| teacher config | `510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5` |
| **`g20_canary.py`** | **`68b8bd00e5e406339d6b6ed3317718315eade55017a0f134fa46cd8af76f7ad0`** (supersedes `ac4e14f2…`) |
| `p1_bmm_probe.py` | `6ebeefe9d56665b55657e4693bd32f0bbd85353507252700e360b7b523d0861e` (unchanged) |
| `g20_payload.tar` | `49a28cce9c86bbadbcac44e12bf9513622b84caf9ddd0688557b7f4e1ddf41e7` (manifest + 4 files, 0 TEST) |
| `g20_subset_manifest.json` | `cc654bf752e998a5b9ea58e95b53be4df531cc8d91abd506afac5c51bb40121a` |
| TRAIN `apple_black_rot_1.jpg` / `.png` | `b5441c7d…f418` / `4fc9d2d5…d628` |
| VAL `apple_black_rot_28.jpg` / `.png` | `e701786b…5c52` / `5763bd1b…9d25` |

The Phase-3 documentation edits touch none of the three runtime-critical files, so the runtime
commit `0bb6996` and its three hashes stay valid. Only the canary hash changed.

**Packet addition:** B58 §13 asks for a pre/post repository-inventory comparison. The canary does not
perform one, so the RunPod evidence step must: hash `git status --porcelain --ignored=matching
--untracked-files=all` of the pod checkout before and after the canary.

---

## 7. Corrections recorded here instead of rewriting history

These historical records were **not edited**; the corrections below supersede them for current use.

- **PREREGISTRATION.md** (anchor `569cfbb`) — see `docs/conflicts.md` #13:
  - U3/U4 (clipping pilots) → M1;
  - §6 (E6-KD clean-TEST trigger) → M7;
  - U8 (RTX 4090 working assumption) → stale by measurement;
  - U9 (NMF) → M4, now measured on CPU.
- **reports/b58_teacher_runner_adjudication.md §16:** "under the authority order the implementation
  contract governs" inverts the manuscript-first order. There is no practical effect, because the
  TeacherRunner satisfies the stricter unqualified B6 wording.
- **reports/b57_strict_mode_evidence.md** and **reports/pre_e1_launch_audit.md** quote the old
  `configs/teacher_finetune.py` criterion "recover 42.05% mIoU within +/-1.5-2.0 pp". That criterion is
  now `NEED_TO_CONFIRM` (M5).
- **docs/reference/context.md** (reference material, not edited): `:92` "Success = recover 42.05% …
  protocol match" and its RTX 4090 compute assumption are superseded as above.
- **G10 is not done in this pass.** The MMSeg 0.x config alias `segnext_mscan-b_512x512_160k_ade20k` at
  `configs/teacher_finetune.py:9` and `docs/IMPLEMENTATION_CONTRACT.md:123` is a wording lag, not a
  conflict. It stays queued so both sites move together.

---

## 8. Open methodology decisions (not decided here)

Registered in `docs/open_questions.md` as **M1–M12**:

| # | Decision |
|---|---|
| M1 | E2/E3 clipping and `max_norm` |
| M2 | teacher augmentation |
| M3 | teacher train/eval scaling |
| M4 | NMF control |
| M5 | teacher acceptance band and schedule lock |
| M6 | λ sweep run length and noise band |
| M7 | E6-KD trigger and weights |
| M8 | multi-seed obligation |
| M9 | QAT checkpoint rule |
| M10 | TEST canvas zero-disease handling |
| M11 | TEST filename count in the teacher preflight |
| M12 | teacher checkpoint selection under M4 |

---

## 9. Gate status

> **[UPDATED 2026-09-21 — after G20 execution]** The paragraphs below are the pre-execution status,
> kept as written. **G20 is now PASS** (§11). The official teacher remains **NO-GO**; its recomputed
> blocker list, in dependency order, is §12.

**G20 (development CUDA canary): GO in principle** *(pre-execution status)*. Every locally checkable
gate passes, and no finding blocks it. This authorizes nothing by itself: executing it needs a separate
explicit GO for a paid pod. Items that only a pod can close: P1N fails as expected, P1G8 succeeds,
first-train and first-val attestation, a real CUDA train step and val step, host Step 0. *(All closed
by §11.)*

**Official teacher: NO-GO** *(pre-execution list — superseded by §12)*. Blockers:
1. teacher protocol lock — framing done; acceptance band and schedule M5 open;
2. teacher augmentation (M2) and train/eval scaling (M3);
3. NMF/Hamburger RNG control (M4) and checkpoint selection under it (M12);
4. ADE20K checkpoint readiness — download, SHA, date, init test;
5. init compatibility — 150→116 head, key audit;
6. batch-16 deterministic-policy VRAM measurement (G2);
7. final GPU selection;
8. final official preflight evidence, and the TEST-count policy (M11);
9. ~~G20 itself executed and closed.~~ **Closed — G20 PASS (§11).**

Now closed at architecture level on CPU: **Stage-3 320-channel stride-16 tap** (§5 B). The teacher
input-equivalence guard also passes (§5 C), which removes that E2 prerequisite.

---

## 10. Files changed in Phase 3 (working tree only; nothing staged or committed)

| Path | Nature |
|---|---|
| `docs/IMPLEMENTATION_CONTRACT.md` | documentation only (governed) |
| `docs/EVALUATION_CONTRACT.md` | documentation only (governed) |
| `docs/STATISTICAL_ANALYSIS_CONTRACT.md` | documentation only |
| `configs/teacher_finetune.py` | text-only analysis dict (governed path; no importers) |
| `docs/teacher_prep_runbook.md` | documentation only |
| `docs/open_questions.md` | documentation only |
| `docs/B8_checkpoint.md` | documentation only |
| `docs/conflicts.md` | documentation only |
| `docs/runpod_environment.md` | documentation only |
| `reports/b59_pre_runpod_reconciliation.md` | this report (new) |

**Governed-path note.** `configs/**` and the two governed contracts are dirty in the working tree until
this change is committed or discarded. Rule 8 therefore blocks an *official* artifact from this
worktree meanwhile. It does not affect G20, which runs from a clean detached checkout of `0bb6996`.

**Post-G20 closeout edits (2026-09-21, documentation only; no governed path added):**
- `docs/open_questions.md` — G20 PASS entry and the post-G20 blocker pointer.
- `docs/runpod_environment.md` — teacher-image status note: G20 PASS on CUDA.
- `docs/teacher_prep_runbook.md` — §4a: G20 closed; the operational findings from the pod.
- This report — header, F7, §9 status note, §11, §12, and appendices I–M.

---

## 11. G20 execution record — **PASS** (2026-09-21)

Executed under an explicit "GO — EXECUTE G20 ONLY" as a paid CUDA canary. The operator terminated the
pod afterwards; all evidence was secured before termination.

### 11.1 Identity

| Item | Value |
|---|---|
| Provider / pod | RunPod, pod `i6dg3nll0l8k00`, datacenter `CA-MTL-1`; tier (Secure/Community) not machine-captured |
| GPU | **NVIDIA RTX A5000**, 24,564 MiB (23.55 GiB visible to torch), sm_86, 64 multiprocessors |
| Driver / CUDA | driver 580.159.04 (supports CUDA 13.0); torch 2.1.0+cu121, CUDA 12.1, cuDNN 8902 |
| Software | Python 3.11.16 · mmseg 1.2.2 · mmcv 2.1.0 · mmengine 0.10.7 · torchvision 0.16.0+cu121 · numpy 1.26.4 · scipy 1.11.4; `pip freeze` 77 lines, sha256 `f162674a…` |
| Image | `sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b`. Declared by the pod template (`PLANTSEG_IMAGE_DIGEST`) and present in PID 1's environment; image revision `48bccc70…` |
| Host resources | 9 vCPU allocated (`RUNPOD_CPU_COUNT`), 96 seen; 30 GB container disk; no network volume; **no PlantSeg dataset on the pod** |
| Runtime source | fresh clone at `/root/g20/plantseg-thesis`, detached at `0bb69961dfedcb4d00ff42990a6543f5dda505ec`. Governed paths clean, no `__pycache__`. `teacher_runner.py` `72206af7…`, launcher `58575276…`, config `510b212b…` matched before the run, in-process (module self-hashes) and after the run |
| Frozen inputs | Canary **`68b8bd00…`**; the superseded `ac4e14f2…` was refused both by hash and by a wording check. Probe `6ebeefe9…`. Payload tar `49a28cce…`; manifest `cc654bf7…`; all four payload files match the manifest |
| Timeline (UTC) | Container start 15:41:15. Step 0 15:45:30. Canary 15:47:39–15:47:49 (9.6 s). Last pod check 15:49:44 (container age 8.5 min) |

### 11.2 The 17 pass criteria

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Step 0 host health | **PASS** | `/dev/nvidiactl`, `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` open; `cuInit(0)=0`; `is_available` True; 1 device; tiny alloc + compute + synchronize OK (App. I) |
| 2 | P1N fails for the deterministic-cuBLAS reason | **PASS** | Variable absent three ways; strict mode; `torch.bmm` raised the documented "uses CuBLAS … CUBLAS_WORKSPACE_CONFIG" `RuntimeError`. Detector qualified (App. K) |
| 3 | P1G8 succeeds | **PASS** | Same bytes and inputs (x `9a3fa537…`, bases `6b541605…`); image-inherited `:4096:8` seen three ways; `bmm` completed, shape (16, 4096, 16) (App. K) |
| 4 | Runtime commit matches | **PASS** | HEAD `0bb69961…` (App. J; canary `git_head_is_runtime_commit`) |
| 5 | Critical hashes match | **PASS** | App. J, canary `hash::` checks, launch provenance, post-run re-hash (App. M) |
| 6 | Canary hash matches | **PASS** | `68b8bd00…` on the pod before launch |
| 7 | Payload manifest matches | **PASS** | 4/4 MATCH; splits `train`, `val` only (App. J) |
| 8 | No TEST present or accessed | **PASS** | No dataset on the pod; no test directories in the payload; canary `test_split_absent` requires passed; dumped config has `test_dataloader`/`test_cfg`/`test_evaluator = None`. The only `…/test` strings in stdout are those two assertion rows |
| 9 | TeacherRunner first-call assertions | **PASS** | `cuda_initialized_before_runner: false`; pre-CUDA marker `true` at both attestation points (the marker is set only after the first-call assertions pass); runner class `src.training.teacher_runner.TeacherRunner` |
| 10 | First-train attestation | **PASS** | Policy {deterministic True, `warn_only` True, `cudnn.deterministic` True, `cudnn.benchmark` **False** — overriding the config's `env_cfg.cudnn_benchmark=True`, `CUBLAS_WORKSPACE_CONFIG` `:4096:8`}, seed 42 |
| 11 | Real CUDA train step | **PASS** | `Iter(train) [1/1]`: forward, backward, optimizer step; `loss_ce` 4.5586. The LR was 6.0e-11 because of the configured LinearLR warmup (start factor 1e-6 × 6e-5) — config behaviour, not a defect |
| 12 | First-val attestation | **PASS** | same five-state policy |
| 13 | Real CUDA validation | **PASS** | `Iter(val) [1/1]` completed |
| 14 | No checkpoint load or download | **PASS** | `load_from = None`; tripwires armed and silent; "Loads checkpoint by" absent; `TORCH_HOME` empty before and after |
| 15 | No checkpoint artifact | **PASS** | 0 `.pth`/`.pt`/`.ckpt`/`.safetensors` created under `/root` or `/tmp`; no `last_checkpoint` |
| 16 | Pre/post inventory explained | **PASS** | See §11.4 |
| 17 | Diagnostic metric classified | **PASS** | aAcc 0.0000 / mIoU 0.0000 / mAcc 0.0000 on one random-init sample. **NON-THESIS — a diagnostic side effect only**; never selected, interpreted, compared or reported |

### 11.3 New measured facts

- **`_histc_cuda` nondeterminism warning** at mmseg `IoUMetric` (`iou_metric.py:190`, `:193`, `:196`),
  under `warn_only=True`. This confirms B56 §4.2 site 3 on real CUDA. Under MMEngine's unmodified strict
  path it would have raised at the first validation — `[INFERRED from the warning text; strict mode was
  not run]`. The numerical impact on the metric was not measured.
- **No `nll_loss2d` alert appeared** for mmseg's cross-entropy (`reduction='none'`) in this
  single-iteration run. This is recorded as *not observed*, **not** as absent: coverage of that
  warning path was not established (N17).
- **Batch-1 peak memory** (`torch.cuda.max_memory_allocated`): **train 2.64 GB (2,641,997,312 B)**,
  **val 2.14 GB (2,141,736,448 B)**.
  **This does NOT determine batch-16 teacher VRAM.** Batch 16, the deterministic bilinear-upsample
  transient at 116×512×512 (D30), and optimiser state scale differently. **G2 remains open.**
- **Image quirk.** The image's `PYTHONPATH=/workspace/plantseg-thesis` points at the image-baked,
  non-git, stale source.

### 11.4 Side-effect audit

- **Inventory.** `git status --porcelain --ignored=matching --untracked-files=all` of the pod checkout
  was empty both **before and after** (sha256 `e3b0c442…b855`): identical.
- **New files outside the work dir** (newer than the pre-canary marker), each explained:
  - `.git/index` — stat refresh by the canary's own `git status`, 15:47:39;
  - `/root/.cache/matplotlib/fontlist-v3.11.0.json` — local font cache built at import, 15:47:43;
  - `/root/.bash_history` and `/root/g20/bin/post.sh` — the operator's SSH sessions, after the canary.
- **External work dir contents:** the mmengine log, `vis_data`, the config dump, the canary evidence and
  `teacher_launch_provenance.json`. No checkpoint.

### 11.5 Operational deviations (not methodology)

- **Runtime checkout location.** Following runbook §4a step 3, the non-git baked
  `/workspace/plantseg-thesis` was not reused. The checkout went to `/root/g20/plantseg-thesis`, and
  `PYTHONPATH`/`G20_REPO` pointed there for the canary. The canary's `launcher_module_from_checkout`
  and self-hash checks confirm that the checkout's code ran.
- **Environment.** `PLANTSEG_IMAGE_DIGEST` came from the pod template and `CUBLAS_WORKSPACE_CONFIG`
  from the image ENV; neither was set by hand.
- **Access path.** RunPod's proxied SSH is PTY-only. Files moved as base64 over the session and were
  hash-verified on both sides.

### 11.6 Evidence

Stored at `C:\Users\admin\plantseg_runs\g20_canary_20260921\`, outside the repository, with
`SHA256SUMS`:
- `g20_evidence_bundle.tgz`: sha256 `415dd7d705ba8c3929af3b7c17b357ca1b78c55e5288f8f61bae17f788fdeeaf`,
  62,592 B, 28 members. The pod-side hash equals the local hash, and gzip integrity is OK.
- `post_full_from_session.log`: sha256 `358f6a85…6abd`, equal to the pod-side log hash.
  - The bundled `logs/post.log` (`c02482de…`) is that log minus its final line. That line records the
    bundle's own hash, so it could not be inside the bundle.

Key member hashes:
- canary evidence `a47e171b…`;
- canary stdout `c56a95ca…`;
- `step0.log` `463d42e5…`;
- `stage.log` `be6c043e…`;
- `p1.log` `26664cf2…`;
- `p1n.log` `bb376fb5…`;
- `p1g8.log` `5bc8f7f0…`;
- launch provenance `673bb204…`.

Appendices I–M render the Step 0, staging, P1, canary-launch and full post-run logs as faithful text.
- Appendices J, K and L are byte-identical to their sources.
- Appendices I and M carry one disclosed whitespace normalisation: trailing spaces were stripped from
  11 lines, and no visible content changed. The appendix preamble below gives the details.
- **The external bundle and its recorded hashes are the authoritative byte-exact record.**

### 11.7 What this supersedes, and its limit

**Supersedes.** These statements now read **G20 PASS (2026-09-21)**:
- B52 §11.1 row G20 and B58 §15 ("STILL OPEN — CUDA evidence pending") — historical records, not edited;
- every current-document "G20 open" statement.

**Informs G15** (teacher strict mode):
- the G18 policy (`warn_only=True`) is in effect on CUDA;
- `histc` is a confirmed nondeterministic-warning site;
- the complete op set is still not enumerated.

**Limit.** G20 is development-canary evidence only. It does not authorize the official teacher run,
which must record its own first-train and first-validation attestations (B58 §13). It says nothing
about teacher accuracy or batch-16 VRAM.

---

## 12. Official teacher — **NO-GO**; remaining blockers after G20, in dependency order

**Closed:**
- G20 (§11);
- Stage-3 320-channel stride-16 tap, at architecture level (§5 B);
- KD input equivalence (§5 C);
- provenance *framing* (B2/B3 wording).

1. **Commit the reconciliation documentation.** This worktree's governed paths are dirty, and rule 8
   blocks official artifacts until they are clean. Pending staging approval.
2. **Methodology locks — human decisions:**
   - **M5:** the teacher acceptance band, plus the schedule details (LinearLR warmup 1,500; poly power
     1.0; horizon 40k; validation every 10k).
   - **M2 + M3:** teacher augmentation and teacher train/eval scaling, decided together.
   - **M11:** whether the official preflight may count TEST filenames.
3. **ADE20K checkpoint readiness** (needs explicit download approval):
   - the gated `mim` download, outside the repository;
   - URL / SHA-256 / date recorded in `docs/teacher_init_source.md`;
   - `scripts/test_teacher_init.py` PASS (G11 import-guard defect is cosmetic);
   - the 150→116 re-head key audit (B55 Appendix B: only `decode_head.conv_seg` dropped).
4. **M4, then M12.**
   - Re-measure repeated-eval NMF variation on the **real ADE20K-initialised** model (CPU suffices; §5 A
     harness).
   - Then decide the NMF control, implement it, and verify it with a negative control.
   - M12 (checkpoint selection under M4) follows.
5. **Implement and verify the locked decisions.**
   - Governed config / launcher / runner edits.
   - CPU re-verification: `smoke_teacher_config`, `smoke_teacher_runner`, `smoke_teacher_launch`, and
     the NMF harness.
   - Commit and push, producing a **new runtime commit and new frozen hashes**.
   - If `teacher_runner.py` or the `--launch` path changes, re-run a G20-style canary on the new commit.
     A config-only change does not require one, because the official run attests itself.
6. **G2 — batch-16, 512², deterministic-policy VRAM on CUDA,** under the final config. Record all five
   determinism states at launch, at the first training forward and before validation. The G20 batch-1
   figures (2.64 / 2.14 GB) do not substitute.
7. **GPU selection** from the G2 measurement, with margin. Secure Cloud is preferred (B53). Rebuild
   the campaign cost estimate from measured per-run cost (N16).
8. **Official preflight on the chosen pod.**
   - Step 0 and a guarded checkout at the new runtime commit.
   - Hash verification.
   - Every launcher gate, including the init checkpoint and the M11-resolved split check.
9. **An explicit human GO for the official teacher run.** The run records its own first-train and
   first-val attestations.

**Not blocking the teacher:**
- G10 (config-name alias) and G6 (manuscript caveat);
- G15 op-set completeness;
- the manuscript corrections (A9–A11, B2, D1–D5);
- E1 N12 double evaluation.

---

## Appendices — instruments and raw results

Each appendix renders the file named in its heading. The byte-level status of each:
- **Appendices A, C–H:** byte-identical copies of the named scratch file.
- **Appendix B:** CR bytes removed (the file was written by Windows Python stdout, CRLF). Nothing else
  was transformed.
- **Appendix D:** mmengine's initialisation log went to stdout. The JSON result is the separate file
  embedded here.
- **Appendices J, K, L:** G20 pod logs, byte-identical to their members of the durable bundle
  (`415dd7d7…`), LF.
- **Appendices I and M — disclosed whitespace normalisation `[2026-09-22, repository hygiene]`.**
  - **Change.** Trailing space characters were stripped from **exactly 11 lines**, solely so that
    `git diff --cached --check` passes. Appendix I lost 1 line's trailing spaces (the date line of the
    embedded `nvidia-smi` table). Appendix M lost 10 lines' trailing spaces: 8 rows of the mmengine
    hook-priority tables, the `Iter(val)` timing line, and the mmseg `build_loss` deprecation-warning
    line.
  - **Scope.** In total 116 space characters were removed, and only at line ends. No visible
    character, word, value or line order changed. Each changed line equals its source line with
    trailing whitespace stripped.
  - **Consequence.** These two appendices are therefore **faithful textual renderings, not
    byte-identical copies**. The sha256 values quoted in their headings are those of the **source
    files**, not of the rendered blocks.
  - **Authority.** The authoritative byte-exact record is the external G20 evidence at
    `C:\Users\admin\plantseg_runs\g20_canary_20260921\`, whose hashes are unchanged:
    - `g20_evidence_bundle.tgz` `415dd7d7…eeaf` — `logs/step0.log` inside it is `463d42e5…`;
    - `post_full_from_session.log` `358f6a85…6abd`, equal to the pod-side value.
- **Appendix M's source** is the complete post-run log recovered from the session transcript, with
  terminal CR bytes removed. Its sha256 equals the pod-side value, so the *source file* is
  byte-identical to the pod file.

### Appendix A — `val_zero_disease_check.py`

sha256 `5b21901fb5be1b6d907107ec5428232b1808e327f6fbc13b6cc0142ddbf50b1d`. Run with the conda Python 3 on the development host, `python -B`, `PYTHONDONTWRITEBYTECODE=1`, `PLANTSEG_DATA_ROOT` unset (default local root).

```python
"""VAL-ONLY check: does any image's FINAL 512x512 canvas mask (the exact E1/evaluator path,
src.data.transforms.core_preprocess) contain zero disease pixels (values 1..115)?

Scope guard: constructs PlantSegDataset for split='val' ONLY. No other split is constructed,
listed, opened or hashed. Read-only: writes nothing inside the repository (bytecode disabled).
"""
import sys

sys.dont_write_bytecode = True

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO = Path(r"C:\Users\admin\plantseg-thesis")
sys.path.insert(0, str(REPO))

from src.data.dataset import PlantSegDataset  # noqa: E402
from src.data.transforms import core_preprocess  # noqa: E402

SPLIT = "val"
assert SPLIT == "val"

ds = PlantSegDataset(SPLIT)
pairs = ds.pairs
for img_p, mask_p in pairs:  # every path must sit under the val folders
    assert img_p.parent.name == "val" and mask_p.parent.name == "val", (img_p, mask_p)

rows = []
for img_p, mask_p in pairs:
    with Image.open(img_p) as im, Image.open(mask_p) as mk:
        raw = np.asarray(mk)
        img_exif = ImageOps.exif_transpose(im)
        exif_dims_match = img_exif.size == mk.size
        img_np, canvas = core_preprocess(im, mk)
    raw_dis = int(((raw >= 1) & (raw <= 115)).sum())
    raw_classes = sorted(int(v) for v in np.unique(raw) if 1 <= v <= 115)
    can_vals = np.unique(canvas)
    can_dis_mask = (canvas >= 1) & (canvas <= 115)
    can_dis = int(can_dis_mask.sum())
    can_classes = sorted(int(v) for v in can_vals if 1 <= v <= 115)
    rows.append({
        "stem": img_p.stem,
        "raw_hw": list(raw.shape),
        "exif_dims_match_mask": bool(exif_dims_match),
        "raw_disease_px": raw_dis,
        "raw_disease_classes": raw_classes,
        "canvas_disease_px": can_dis,
        "canvas_disease_classes": can_classes,
        "canvas_background_px": int((canvas == 0).sum()),
        "canvas_ignore_px": int((canvas == 255).sum()),
        "canvas_values_in_domain": bool(all((0 <= v <= 115) or v == 255 for v in can_vals.tolist())),
    })

n = len(rows)
zero_canvas = [r for r in rows if r["canvas_disease_px"] == 0]
zero_raw = [r for r in rows if r["raw_disease_px"] == 0]
lost_by_resize = [r for r in zero_canvas if r["raw_disease_px"] > 0]
multi_class = [r for r in rows if len(r["canvas_disease_classes"]) > 1]
class_changed = [r for r in rows if r["canvas_disease_classes"] and r["canvas_disease_classes"] != r["raw_disease_classes"]]
bg_zero = [r for r in rows if r["canvas_background_px"] == 0]
exif_mismatch = [r for r in rows if not r["exif_dims_match_mask"]]
out_of_domain = [r for r in rows if not r["canvas_values_in_domain"]]
can_px = np.array([r["canvas_disease_px"] for r in rows])
thresholds = {t: int((can_px < t).sum()) for t in (1, 10, 50, 100, 500)}
smallest = sorted(rows, key=lambda r: r["canvas_disease_px"])[:10]

manifest = "\n".join(f"{r['stem']}" for r in rows).encode()
result = {
    "split": SPLIT,
    "instrument": "src.data.transforms.core_preprocess (E1 validation / evaluator path) via "
                  "src.data.dataset.PlantSegDataset('val')",
    "data_root": str(Path(ds.pairs[0][0]).parents[2]),
    "n_images": n,
    "val_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    "zero_disease_on_final_canvas": len(zero_canvas),
    "zero_disease_on_raw_mask": len(zero_raw),
    "disease_lost_by_resize": len(lost_by_resize),
    "images_with_multiple_disease_classes_on_canvas": len(multi_class),
    "images_whose_disease_class_changed": len(class_changed),
    "images_with_zero_background_on_canvas": len(bg_zero),
    "exif_dim_mismatch_after_transpose": len(exif_mismatch),
    "canvas_values_out_of_domain": len(out_of_domain),
    "canvas_disease_px_min": int(can_px.min()),
    "canvas_disease_px_median": float(np.median(can_px)),
    "count_canvas_disease_px_below": thresholds,
    "smallest_10": [{k: r[k] for k in ("stem", "raw_hw", "raw_disease_px", "canvas_disease_px",
                                       "canvas_disease_classes")} for r in smallest],
    "zero_canvas_list": [r["stem"] for r in zero_canvas],
    "canvas_disease_class_histogram_n_classes": len(Counter(c for r in rows for c in r["canvas_disease_classes"])),
}
print(json.dumps(result, indent=2))
```

### Appendix B — `val_zero_disease_result.json`

sha256 of the CRLF source file `354c53b0e4915739440f2a4cbcd7f7374fae76b8d7b7ab3f91aad322317f8d2c`; embedded with CR bytes removed (148 CR bytes).

```json
{
  "split": "val",
  "instrument": "src.data.transforms.core_preprocess (E1 validation / evaluator path) via src.data.dataset.PlantSegDataset('val')",
  "data_root": "C:\\Users\\admin\\plantseg_data\\plantseg",
  "n_images": 846,
  "val_manifest_sha256": "92de2a9678cbe93988abc49e27408a80769d38bee559bdacbcc2f9414f272b2f",
  "zero_disease_on_final_canvas": 0,
  "zero_disease_on_raw_mask": 0,
  "disease_lost_by_resize": 0,
  "images_with_multiple_disease_classes_on_canvas": 0,
  "images_whose_disease_class_changed": 0,
  "images_with_zero_background_on_canvas": 0,
  "exif_dim_mismatch_after_transpose": 0,
  "canvas_values_out_of_domain": 0,
  "canvas_disease_px_min": 259,
  "canvas_disease_px_median": 24270.5,
  "count_canvas_disease_px_below": {
    "1": 0,
    "10": 0,
    "50": 0,
    "100": 0,
    "500": 3
  },
  "smallest_10": [
    {
      "stem": "potato_early_blight_9",
      "raw_hw": [
        380,
        600
      ],
      "raw_disease_px": 349,
      "canvas_disease_px": 259,
      "canvas_disease_classes": [
        76
      ]
    },
    {
      "stem": "wheat_powdery_mildew_Baidu_0309",
      "raw_hw": [
        427,
        640
      ],
      "raw_disease_px": 579,
      "canvas_disease_px": 368,
      "canvas_disease_classes": [
        108
      ]
    },
    {
      "stem": "apple_rust_84",
      "raw_hw": [
        380,
        600
      ],
      "raw_disease_px": 671,
      "canvas_disease_px": 487,
      "canvas_disease_classes": [
        3
      ]
    },
    {
      "stem": "wheat_stripe_rust_Baidu_0430",
      "raw_hw": [
        880,
        660
      ],
      "raw_disease_px": 1626,
      "canvas_disease_px": 570,
      "canvas_disease_classes": [
        111
      ]
    },
    {
      "stem": "wheat_stripe_rust_Baidu_0168",
      "raw_hw": [
        853,
        640
      ],
      "raw_disease_px": 1759,
      "canvas_disease_px": 638,
      "canvas_disease_classes": [
        111
      ]
    },
    {
      "stem": "banana_anthracnose_Baidu_0115",
      "raw_hw": [
        325,
        440
      ],
      "raw_disease_px": 538,
      "canvas_disease_px": 728,
      "canvas_disease_classes": [
        5
      ]
    },
    {
      "stem": "wheat_leaf_rust_Baidu_0345",
      "raw_hw": [
        1200,
        800
      ],
      "raw_disease_px": 4249,
      "canvas_disease_px": 762,
      "canvas_disease_classes": [
        106
      ]
    },
    {
      "stem": "apple_scab_45",
      "raw_hw": [
        350,
        600
      ],
      "raw_disease_px": 1284,
      "canvas_disease_px": 948,
      "canvas_disease_classes": [
        4
      ]
    },
    {
      "stem": "bell_pepper_frogeye_leaf_spot_Bing_0215",
      "raw_hw": [
        675,
        900
      ],
      "raw_disease_px": 3162,
      "canvas_disease_px": 1024,
      "canvas_disease_classes": [
        17
      ]
    },
    {
      "stem": "cucumber_powdery_mildew_25",
      "raw_hw": [
        853,
        640
      ],
      "raw_disease_px": 3005,
      "canvas_disease_px": 1086,
      "canvas_disease_classes": [
        51
      ]
    }
  ],
  "zero_canvas_list": [],
  "canvas_disease_class_histogram_n_classes": 114
}
```

### Appendix C — `cpu_verify.py`

sha256 `eceefa5c1b496ec2af5c584dc2f930322ace9e17ffd8f1f887115a9fb31e1b42`. Run: `docker run --rm --network none` with read-only mounts of the `git archive HEAD -- src configs scripts` export, the G20 payload and this script, and one writable scratch output mount; `python -B -W ignore`.

```python
"""B59 scratch CPU verification — A (NMF/Hamburger RNG), B (real Stage-3 tap), C (teacher input
equivalence). Runs INSIDE the pinned teacher image, CPU only, --network none.

Scope: measurement only. Nothing here changes a config, a policy, or repository code. The teacher
weights are a RANDOM initialisation (seeded) — no official or ADE20K checkpoint exists or is loaded.
The one real image is the G20 payload VAL image (TRAIN/VAL policy). TEST is never referenced.

Tolerances for C are fixed HERE, before any result is read:
  INPUT_ABS_TOL      = 1e-5   max |N - S| on the normalised input tensor
  OUTPUT_REL_TOL     = 1e-4   max|a-b| / max|b| on the Stage-3 feature and the logits
  NEG_INPUT_MIN      = 1e-1   a negative control must differ by at least this at the input
  NEG_OUTPUT_REL_MIN = 1e-2   ... and at least this, relatively, at the feature/logits
"""
import sys

sys.dont_write_bytecode = True

import copy
import hashlib
import json
import os
import platform

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/work/repo"
sys.path.insert(0, REPO)
os.chdir(REPO)

import mmcv  # noqa: E402
import mmengine  # noqa: E402
import mmseg  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from PIL import Image  # noqa: E402

from mmseg.registry import MODELS  # noqa: E402

from src.data.transforms import core_preprocess, finalize  # noqa: E402
from src.distill.segnext_teacher import (TeacherArchitectureMismatch, SegNeXtTeacherAdapter,  # noqa: E402
                                         select_stride16_feature)
from src.distill.teacher import load_frozen_teacher  # noqa: E402

INPUT_ABS_TOL = 1e-5
OUTPUT_REL_TOL = 1e-4
NEG_INPUT_MIN = 1e-1
NEG_OUTPUT_REL_MIN = 1e-2

CFG_PATH = f"{REPO}/configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
IMG = "/work/payload/plantseg/images/val/apple_black_rot_28.jpg"
MSK = "/work/payload/plantseg/annotations/val/apple_black_rot_28.png"
OUT = "/work/out"
assert "/test/" not in IMG and "/test/" not in MSK

torch.set_num_threads(4)
R = {"env": {"python": platform.python_version(), "torch": torch.__version__,
             "mmseg": mmseg.__version__, "mmcv": mmcv.__version__, "mmengine": mmengine.__version__,
             "cuda_available": torch.cuda.is_available(),
             "config_sha256": hashlib.sha256(open(CFG_PATH, "rb").read()).hexdigest(),
             "ham_head_sha256": hashlib.sha256(open(
                 os.path.join(os.path.dirname(mmseg.__file__), "models/decode_heads/ham_head.py"),
                 "rb").read()).hexdigest()}}

cfg = Config.fromfile(CFG_PATH)
init_default_scope("mmseg")


def build(seed: int, rand_init=None):
    c = copy.deepcopy(cfg.model)
    if rand_init is not None:
        c["decode_head"]["ham_kwargs"]["rand_init"] = rand_init
    torch.manual_seed(seed)
    m = MODELS.build(c)
    m.init_weights()
    m.eval()
    return m


def mx(a, b):
    return float((a - b).abs().max())


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def digest(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- fixed real input (VAL payload)
with Image.open(IMG) as im, Image.open(MSK) as mk:
    img_np, mask_np = core_preprocess(im, mk)          # RGB uint8 HWC 512x512, int64 mask
S, M = finalize(img_np, mask_np)                        # student tensor: float32 CHW, ImageNet-normalised RGB
x = S.unsqueeze(0)
valid = (M != 255)
R["input"] = {"image": os.path.basename(IMG), "tensor_shape": list(x.shape),
              "tensor_sha256_16": digest(x), "valid_fraction": float(valid.float().mean())}

model = build(seed=0)
R["model"] = {
    "type": type(model).__name__, "backbone": type(model.backbone).__name__,
    "decode_head": type(model.decode_head).__name__,
    "embed_dims": list(cfg.model.backbone.embed_dims), "depths": list(cfg.model.backbone.depths),
    "ham_kwargs": dict(cfg.model.decode_head.ham_kwargs),
    "num_classes": int(model.decode_head.num_classes),
    "params": int(sum(p.numel() for p in model.parameters())),
    "weights": "random init (torch.manual_seed(0) + init_weights()); NO checkpoint",
}

# =============================================================== A. NMF / Hamburger RNG measurement
A = {}
with torch.no_grad():
    feats = model.extract_feat(x)                               # backbone: deterministic
    feats_again = model.extract_feat(x)
    A["backbone_repeat_bitwise_equal"] = all(torch.equal(a, b) for a, b in zip(feats, feats_again))

    # A1. RNG consumption by one eval decode-head forward
    s0 = torch.get_rng_state()
    _ = model.decode_head.forward(feats)
    s1 = torch.get_rng_state()
    torch.set_rng_state(s0)
    _ = torch.rand((1 * 1, 512, 16))                            # (B*S, D, R) for B=1, S=1, D=512, R=16
    s_expected = torch.get_rng_state()
    A["eval_forward_consumes_cpu_rng"] = not torch.equal(s0, s1)
    A["consumption_equals_one_torch_rand_BxSxDxR_1x512x16"] = torch.equal(s1, s_expected)

    # A2. repeated eval forwards on fixed weights + fixed input, RNG NOT reset
    N_REP = 20
    torch.manual_seed(2026)
    outs = [model.decode_head.forward(feats) for _ in range(N_REP)]
    base = outs[0]
    diffs = [(o - base).abs() for o in outs[1:]]
    amax0 = base.argmax(1)
    up = lambda t: F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)  # noqa: E731
    amax0_512 = up(base).argmax(1)[0]
    flip64 = [float((o.argmax(1) != amax0).float().mean()) for o in outs[1:]]
    flip512 = [float(((up(o).argmax(1)[0] != amax0_512) & valid).sum() / valid.sum()) for o in outs[1:]]
    pair_max = max(mx(outs[i], outs[j]) for i in range(N_REP) for j in range(i + 1, N_REP))
    A["repeated_forwards_no_reset"] = {
        "n_forwards": N_REP,
        "distinct_output_digests": len({digest(o) for o in outs}),
        "logit_abs_max_of_run0": float(base.abs().max()),
        "logit_abs_mean_of_run0": float(base.abs().mean()),
        "max_abs_diff_vs_run0": float(max(d.max() for d in diffs)),
        "mean_abs_diff_vs_run0_avg": float(np.mean([float(d.mean()) for d in diffs])),
        "max_rel_diff_vs_run0": float(max(float(d.max()) for d in diffs) / float(base.abs().max())),
        "max_abs_diff_any_pair": pair_max,
        "argmax_flip_fraction_64x64_mean": float(np.mean(flip64)),
        "argmax_flip_fraction_64x64_max": float(np.max(flip64)),
        "argmax_flip_fraction_512_valid_px_mean": float(np.mean(flip512)),
        "argmax_flip_fraction_512_valid_px_max": float(np.max(flip512)),
    }

    # A3. reset CPU RNG to the same seed before every forward
    reset = []
    for _ in range(5):
        torch.manual_seed(1234)
        reset.append(model.decode_head.forward(feats))
    A["reset_seed_before_each_forward"] = {
        "n_forwards": 5,
        "all_bitwise_equal": all(torch.equal(reset[0], r) for r in reset[1:]),
        "distinct_output_digests": len({digest(r) for r in reset}),
    }
    torch.manual_seed(1234)
    ref_state = torch.get_rng_state()
    a = model.decode_head.forward(feats)
    torch.set_rng_state(ref_state)
    b = model.decode_head.forward(feats)
    A["restore_saved_rng_state_bitwise_equal"] = torch.equal(a, b)

    # A4. batch-position dependence: the same image twice in one batch
    x2 = torch.cat([x, x], 0)
    feats2 = model.extract_feat(x2)
    torch.manual_seed(99)
    o2 = model.decode_head.forward(feats2)
    torch.manual_seed(99)
    o1 = model.decode_head.forward(feats)
    A["batch_position"] = {
        "same_image_pos0_vs_pos1_max_abs_diff": mx(o2[0], o2[1]),
        "same_image_pos0_vs_pos1_bitwise_equal": torch.equal(o2[0], o2[1]),
        "single_vs_batch_pos0_same_seed_max_abs_diff": mx(o1[0], o2[0]),
        "single_vs_batch_pos1_same_seed_max_abs_diff": mx(o1[0], o2[1]),
    }

# A5. DIAGNOSTIC CONTROL ONLY (not a policy): rand_init=False, identical construction seed
model_fixed = build(seed=0, rand_init=False)
same_params = all(torch.equal(p, q) for p, q in zip(model.state_dict().values(),
                                                     model_fixed.state_dict().values()))
with torch.no_grad():
    ff = model_fixed.extract_feat(x)
    torch.manual_seed(2026)
    fo = [model_fixed.decode_head.forward(ff) for _ in range(10)]
A["diagnostic_control_rand_init_false"] = {
    "note": "diagnostic attribution control only; NOT a proposed or implemented policy",
    "parameters_identical_to_rand_init_true_model": same_params,
    "n_forwards": 10,
    "distinct_output_digests_after_first_forward": len({digest(o) for o in fo[1:]}),
    "max_abs_diff_forward2_vs_forward10": mx(fo[1], fo[-1]),
}
R["A_nmf"] = A

# =============================================================== B. real Stage-3 tap
B = {}
hooks = {}


def hook(name):
    def _h(mod, inp, out):
        hooks[name] = out.detach()
    return _h


bb = model.backbone
handles = [getattr(bb, f"norm{i}").register_forward_hook(hook(f"norm{i}")) for i in (1, 2, 3, 4)]
with torch.no_grad():
    feats = model.extract_feat(x)
for h in handles:
    h.remove()
B["stage_outputs"] = [{"index": i, "stage": i + 1, "shape": list(f.shape),
                       "stride": 512 // f.shape[-1]} for i, f in enumerate(feats)]
B["blocks_per_stage"] = [len(getattr(bb, f"block{i}")) for i in (1, 2, 3, 4)]
sel = select_stride16_feature(feats, (512, 512))
B["selected_shape"] = list(sel.shape)
B["selected_is_backbone_output_index"] = [i for i, f in enumerate(feats) if f is sel]
n3 = hooks["norm3"]
n3_img = n3.reshape(1, 32, 32, -1).permute(0, 3, 1, 2)
B["selected_equals_norm3_stage3_output"] = torch.equal(sel, n3_img)
B["selected_stride"] = 512 // sel.shape[-1]
B["selected_channels"] = int(sel.shape[1])
adapter = SegNeXtTeacherAdapter(model)
torch.manual_seed(5)
with torch.no_grad():
    aout = adapter(x)
B["adapter_feat_s16_shape"] = list(aout["feat_s16"].shape)
B["adapter_feat_s16_equals_selected"] = torch.equal(aout["feat_s16"], sel)
B["adapter_logits_shape"] = list(aout["logits"].shape)

neg = {}
try:
    select_stride16_feature(list(feats[:2]) + list(feats[3:]), (512, 512))
    neg["stage3_removed"] = "NOT RAISED (control failed)"
except TeacherArchitectureMismatch as e:
    neg["stage3_removed"] = f"RAISED TeacherArchitectureMismatch: {str(e)[:120]}"
try:
    select_stride16_feature(feats, (512, 512), expected_ch=320, expected_stride=32)
    neg["wrong_stride_32"] = "NOT RAISED (control failed)"
except TeacherArchitectureMismatch as e:
    neg["wrong_stride_32"] = f"RAISED TeacherArchitectureMismatch: {str(e)[:120]}"
n4 = hooks["norm4"]
neg["semantic_check_against_stage4_norm4"] = {
    "stage4_shape": list(feats[3].shape),
    "selected_equals_stage4": bool(sel.shape == feats[3].shape and torch.equal(sel, feats[3])),
    "norm4_numel_matches_selected": n4.numel() == sel.numel(),
}
B["negative_controls"] = neg
R["B_stage3"] = B

# =============================================================== C. teacher input-equivalence guard
C = {"tolerances": {"INPUT_ABS_TOL": INPUT_ABS_TOL, "OUTPUT_REL_TOL": OUTPUT_REL_TOL,
                    "NEG_INPUT_MIN": NEG_INPUT_MIN, "NEG_OUTPUT_REL_MIN": NEG_OUTPUT_REL_MIN}}
# Live KD API: load_frozen_teacher -> build_mmseg_teacher -> build_segnext_teacher -> mmseg.apis.init_model
ckpt = f"{OUT}/scratch_random_init_teacher.pth"
torch.save({"meta": {"note": "B59 scratch random-init weights; NOT a teacher checkpoint"},
            "state_dict": model.state_dict()}, ckpt)
frozen = load_frozen_teacher(ckpt, config_path=CFG_PATH)
seg = frozen.teacher.model
dp = seg.data_preprocessor
C["live_path"] = ("src.distill.teacher.load_frozen_teacher -> build_segnext_teacher -> "
                  "mmseg.apis.init_model(config, ckpt, device='cpu') -> SegNeXtTeacherAdapter -> "
                  "FrozenTeacher.forward (the call train_distill.py:395 makes with model_input = img)")
C["data_preprocessor"] = {"type": type(dp).__name__, "bgr_to_rgb": bool(dp.channel_conversion),
                          "mean": dp.mean.flatten().tolist(), "std": dp.std.flatten().tolist()}
C["loaded_weights_equal_scratch"] = all(torch.equal(a, b) for a, b in zip(
    seg.state_dict().values(), model.state_dict().values()))

calls = {"n": 0}
orig_fwd = dp.forward


def counting_forward(*a, **k):
    calls["n"] += 1
    return orig_fwd(*a, **k)


dp.forward = counting_forward


def kd(inp, seed=7):
    torch.manual_seed(seed)                     # hold the NMF draw fixed across compared paths
    with torch.no_grad():
        return frozen(inp, logits_size=(64, 64), feat_size=(32, 32))


out_K = kd(x)
C["data_preprocessor_calls_during_kd_forward"] = calls["n"]
dp.forward = orig_fwd

# Reference: MMSeg's own path on the SAME pixels. LoadImageFromFile yields BGR uint8 HWC;
# PackSegInputs gives uint8 CHW; SegDataPreProcessor does bgr->rgb + (x-mean)/std.
from mmseg.structures import SegDataSample  # noqa: E402


def run_dp(chw):
    """MMSeg's eval data path. The merged config gives the preprocessor a test_cfg, so stack_batch
    runs and needs data_samples (as the real dataloader supplies)."""
    ds = SegDataSample()
    ds.set_metainfo(dict(img_shape=tuple(chw.shape[-2:]), ori_shape=tuple(chw.shape[-2:])))
    return dp({"inputs": [chw], "data_samples": [ds]}, False)["inputs"]


C["data_preprocessor"]["test_cfg"] = dict(dp.test_cfg) if dp.test_cfg else None
bgr_chw = torch.from_numpy(np.ascontiguousarray(img_np[..., ::-1])).permute(2, 0, 1).contiguous()
N = run_dp(bgr_chw)


def ref_path(inp, seed=7):
    torch.manual_seed(seed)
    with torch.no_grad():
        f = seg.extract_feat(inp)
        return select_stride16_feature(f, (512, 512)), seg.decode_head.forward(f)


ref_feat, ref_logits = ref_path(N)
C["positive"] = {
    "compared": "student finalize() tensor -> live KD path  VS  raw pixels -> MMSeg SegDataPreProcessor -> same model",
    "input_max_abs_diff": mx(x, N),
    "feat_s16_max_rel_diff": rel(out_K.feat_s16, ref_feat),
    "logits_max_rel_diff": rel(out_K.logits, ref_logits),
    "logits_argmax_agreement": float((out_K.logits.argmax(1) == ref_logits.argmax(1)).float().mean()),
}


def negative(name, inp_to_kd):
    o = kd(inp_to_kd)
    return {"control": name, "input_max_abs_diff": mx(inp_to_kd, N),
            "feat_s16_max_rel_diff": rel(o.feat_s16, ref_feat),
            "logits_max_rel_diff": rel(o.logits, ref_logits),
            "logits_argmax_agreement": float((o.logits.argmax(1) == ref_logits.argmax(1)).float().mean())}


rgb_chw = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).contiguous()
negs = [
    negative("NC1 channel reversal of the student tensor (RGB->BGR)", x[:, [2, 1, 0]]),
    negative("NC2 double normalisation: student tensor re-run through SegDataPreProcessor",
             run_dp(x[0])),
    negative("NC3 RGB fed to the BGR-expecting preprocessor (channel order reversed once)",
             run_dp(rgb_chw)),
]
C["negative_controls"] = negs
pos = C["positive"]
pos_ok = (pos["input_max_abs_diff"] <= INPUT_ABS_TOL and pos["feat_s16_max_rel_diff"] <= OUTPUT_REL_TOL
          and pos["logits_max_rel_diff"] <= OUTPUT_REL_TOL
          and C["data_preprocessor_calls_during_kd_forward"] == 0)
neg_ok = all(n["input_max_abs_diff"] >= NEG_INPUT_MIN and n["feat_s16_max_rel_diff"] >= NEG_OUTPUT_REL_MIN
             for n in negs)
C["verdict"] = {"positive_within_tolerance": pos_ok, "all_negative_controls_detected": neg_ok,
                "PASS": bool(pos_ok and neg_ok)}
os.remove(ckpt)
C["scratch_checkpoint_deleted"] = not os.path.exists(ckpt)
R["C_input_equivalence"] = C

with open(f"{OUT}/cpu_verify_result.json", "w", encoding="utf-8") as fh:
    fh.write(json.dumps(R, indent=2, default=str) + "\n")
print("RESULT_WRITTEN")
```

### Appendix D — `cpu_verify_result.json`

sha256 `cb4f91fee114e1ce88fb26f1bf3f3b97b6641826e552b9b7f7a3040103346fc1`.

```json
{
  "env": {
    "python": "3.11.16",
    "torch": "2.1.0+cu121",
    "mmseg": "1.2.2",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "cuda_available": false,
    "config_sha256": "510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5",
    "ham_head_sha256": "eb2f0963769926e303a832874d01c7d854596fe9839f400bcaf3c596110790b3"
  },
  "input": {
    "image": "apple_black_rot_28.jpg",
    "tensor_shape": [
      1,
      3,
      512,
      512
    ],
    "tensor_sha256_16": "8b91c3824d2fd304",
    "valid_fraction": 0.6328125
  },
  "model": {
    "type": "EncoderDecoder",
    "backbone": "MSCAN",
    "decode_head": "LightHamHead",
    "embed_dims": [
      64,
      128,
      320,
      512
    ],
    "depths": [
      3,
      3,
      12,
      3
    ],
    "ham_kwargs": {
      "MD_S": 1,
      "MD_R": 16,
      "train_steps": 6,
      "eval_steps": 7,
      "inv_t": 100,
      "rand_init": true
    },
    "num_classes": 116,
    "params": 27618868,
    "weights": "random init (torch.manual_seed(0) + init_weights()); NO checkpoint"
  },
  "A_nmf": {
    "backbone_repeat_bitwise_equal": true,
    "eval_forward_consumes_cpu_rng": true,
    "consumption_equals_one_torch_rand_BxSxDxR_1x512x16": true,
    "repeated_forwards_no_reset": {
      "n_forwards": 20,
      "distinct_output_digests": 20,
      "logit_abs_max_of_run0": 0.5537388920783997,
      "logit_abs_mean_of_run0": 0.11296766996383667,
      "max_abs_diff_vs_run0": 0.13712897896766663,
      "mean_abs_diff_vs_run0_avg": 0.008345697687840775,
      "max_rel_diff_vs_run0": 0.24764194989621857,
      "max_abs_diff_any_pair": 0.15691745281219482,
      "argmax_flip_fraction_64x64_mean": 0.11762438322368421,
      "argmax_flip_fraction_64x64_max": 0.165283203125,
      "argmax_flip_fraction_512_valid_px_mean": 0.1072829091235211,
      "argmax_flip_fraction_512_valid_px_max": 0.13164906203746796
    },
    "reset_seed_before_each_forward": {
      "n_forwards": 5,
      "all_bitwise_equal": true,
      "distinct_output_digests": 1
    },
    "restore_saved_rng_state_bitwise_equal": true,
    "batch_position": {
      "same_image_pos0_vs_pos1_max_abs_diff": 0.08615816384553909,
      "same_image_pos0_vs_pos1_bitwise_equal": false,
      "single_vs_batch_pos0_same_seed_max_abs_diff": 3.0174851417541504e-07,
      "single_vs_batch_pos1_same_seed_max_abs_diff": 0.08615822345018387
    },
    "diagnostic_control_rand_init_false": {
      "note": "diagnostic attribution control only; NOT a proposed or implemented policy",
      "parameters_identical_to_rand_init_true_model": true,
      "n_forwards": 10,
      "distinct_output_digests_after_first_forward": 1,
      "max_abs_diff_forward2_vs_forward10": 0.0
    }
  },
  "B_stage3": {
    "stage_outputs": [
      {
        "index": 0,
        "stage": 1,
        "shape": [
          1,
          64,
          128,
          128
        ],
        "stride": 4
      },
      {
        "index": 1,
        "stage": 2,
        "shape": [
          1,
          128,
          64,
          64
        ],
        "stride": 8
      },
      {
        "index": 2,
        "stage": 3,
        "shape": [
          1,
          320,
          32,
          32
        ],
        "stride": 16
      },
      {
        "index": 3,
        "stage": 4,
        "shape": [
          1,
          512,
          16,
          16
        ],
        "stride": 32
      }
    ],
    "blocks_per_stage": [
      3,
      3,
      12,
      3
    ],
    "selected_shape": [
      1,
      320,
      32,
      32
    ],
    "selected_is_backbone_output_index": [
      2
    ],
    "selected_equals_norm3_stage3_output": true,
    "selected_stride": 16,
    "selected_channels": 320,
    "adapter_feat_s16_shape": [
      1,
      320,
      32,
      32
    ],
    "adapter_feat_s16_equals_selected": true,
    "adapter_logits_shape": [
      1,
      116,
      64,
      64
    ],
    "negative_controls": {
      "stage3_removed": "RAISED TeacherArchitectureMismatch: expected exactly one stride-16 feature with 320 channels (MSCAN-B Stage-3, embed_dims=[64, 128, 320, 512]); found 0. Bac",
      "wrong_stride_32": "RAISED TeacherArchitectureMismatch: expected exactly one stride-32 feature with 320 channels (MSCAN-B Stage-3, embed_dims=[64, 128, 320, 512]); found 0. Bac",
      "semantic_check_against_stage4_norm4": {
        "stage4_shape": [
          1,
          512,
          16,
          16
        ],
        "selected_equals_stage4": false,
        "norm4_numel_matches_selected": false
      }
    }
  },
  "C_input_equivalence": {
    "tolerances": {
      "INPUT_ABS_TOL": 1e-05,
      "OUTPUT_REL_TOL": 0.0001,
      "NEG_INPUT_MIN": 0.1,
      "NEG_OUTPUT_REL_MIN": 0.01
    },
    "live_path": "src.distill.teacher.load_frozen_teacher -> build_segnext_teacher -> mmseg.apis.init_model(config, ckpt, device='cpu') -> SegNeXtTeacherAdapter -> FrozenTeacher.forward (the call train_distill.py:395 makes with model_input = img)",
    "data_preprocessor": {
      "type": "SegDataPreProcessor",
      "bgr_to_rgb": true,
      "mean": [
        123.67500305175781,
        116.27999877929688,
        103.52999877929688
      ],
      "std": [
        58.39500045776367,
        57.119998931884766,
        57.375
      ],
      "test_cfg": {
        "size_divisor": 32
      }
    },
    "loaded_weights_equal_scratch": true,
    "data_preprocessor_calls_during_kd_forward": 0,
    "positive": {
      "compared": "student finalize() tensor -> live KD path  VS  raw pixels -> MMSeg SegDataPreProcessor -> same model",
      "input_max_abs_diff": 2.384185791015625e-07,
      "feat_s16_max_rel_diff": 2.9227760478534037e-06,
      "logits_max_rel_diff": 3.3105645798059413e-06,
      "logits_argmax_agreement": 1.0
    },
    "negative_controls": [
      {
        "control": "NC1 channel reversal of the student tensor (RGB->BGR)",
        "input_max_abs_diff": 1.5950753688812256,
        "feat_s16_max_rel_diff": 1.098610758781433,
        "logits_max_rel_diff": 0.671310544013977,
        "logits_argmax_agreement": 0.722412109375
      },
      {
        "control": "NC2 double normalisation: student tensor re-run through SegDataPreProcessor",
        "input_max_abs_diff": 3.9678406715393066,
        "feat_s16_max_rel_diff": 1.4285527467727661,
        "logits_max_rel_diff": 0.7874304056167603,
        "logits_argmax_agreement": 0.660888671875
      },
      {
        "control": "NC3 RGB fed to the BGR-expecting preprocessor (channel order reversed once)",
        "input_max_abs_diff": 1.9520697593688965,
        "feat_s16_max_rel_diff": 1.0830800533294678,
        "logits_max_rel_diff": 0.7830727696418762,
        "logits_argmax_agreement": 0.50439453125
      }
    ],
    "verdict": {
      "positive_within_tolerance": true,
      "all_negative_controls_detected": true,
      "PASS": true
    },
    "scratch_checkpoint_deleted": true
  }
}
```

### Appendix E — `canary_equiv.py`

sha256 `ca7915eb29fca59f3dd1f1eeff26abfa4676c45a053d492e2dd54b19692a6561`.

```python
"""Prove the g20_canary.py edit is text/provenance-only: the two ASTs are identical once the single
provenance 'grade' string constant is masked, both files compile, and 'grade' is used nowhere else."""
import sys

sys.dont_write_bytecode = True

import ast
import hashlib
import json

OLD, NEW = "/work/c/g20_canary.ac4e14f2.py", "/work/c/g20_canary.py"


def load(p):
    b = open(p, "rb").read()
    return b, ast.parse(b.decode("utf-8"), filename=p)


def grade_nodes(tree):
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "grade":
                    hits.append(v)
    return hits


res = {}
for tag, p in (("old", OLD), ("new", NEW)):
    b, t = load(p)
    compile(b, p, "exec")                      # raises on any syntax error
    g = grade_nodes(t)
    res[tag] = {"sha256": hashlib.sha256(b).hexdigest(), "compiles": True,
                "grade_nodes": len(g), "grade_value": g[0].value if g else None,
                "grade_is_plain_string_constant": all(isinstance(n, ast.Constant) and isinstance(n.value, str) for n in g),
                "grade_string_occurrences_in_source": b.decode().count('"grade"')}
    for n in g:
        n.value = "<GRADE>"
    res[tag]["masked_dump_sha256"] = hashlib.sha256(ast.dump(t, include_attributes=False).encode()).hexdigest()

res["ast_identical_after_masking_grade"] = res["old"]["masked_dump_sha256"] == res["new"]["masked_dump_sha256"]
res["old_phrase_absent_in_new"] = "no accuracy is reported" not in open(NEW, encoding="utf-8").read()
res["expected_hashes_block_unchanged"] = (
    open(OLD, encoding="utf-8").read().split("EXPECTED_HASHES = {")[1].split("}")[0]
    == open(NEW, encoding="utf-8").read().split("EXPECTED_HASHES = {")[1].split("}")[0])
print(json.dumps(res, indent=2))
```

### Appendix F — `canary_equiv_result.json`

sha256 `103cafc9285ed2579874937f3b944b607e60184ae3ed707f981c9fa106f51ad3`.

```json
{
  "old": {
    "sha256": "ac4e14f2cb60893258d33ca885c63f3eccf71f61aab76e06a14695b4236efa54",
    "compiles": true,
    "grade_nodes": 1,
    "grade_value": "development-canary only - not an E1-E7 experiment, not a Chapter IV result, not checkpoint-selection evidence, no accuracy is reported from this run",
    "grade_is_plain_string_constant": true,
    "grade_string_occurrences_in_source": 1,
    "masked_dump_sha256": "ec11ee4362f9e28fab7fb8badedecc70214076f5e496741727114445cb0c9c6f"
  },
  "new": {
    "sha256": "68b8bd00e5e406339d6b6ed3317718315eade55017a0f134fa46cd8af76f7ad0",
    "compiles": true,
    "grade_nodes": 1,
    "grade_value": "development-canary only - not an E1-E7 experiment, not a Chapter IV result, not checkpoint-selection evidence. G20 produces no accuracy result that is used, selected, interpreted, compared, or reported as thesis evidence; any one-sample random-init validation metric is a diagnostic side effect only",
    "grade_is_plain_string_constant": true,
    "grade_string_occurrences_in_source": 1,
    "masked_dump_sha256": "ec11ee4362f9e28fab7fb8badedecc70214076f5e496741727114445cb0c9c6f"
  },
  "ast_identical_after_masking_grade": true,
  "old_phrase_absent_in_new": true,
  "expected_hashes_block_unchanged": true
}
```

### Appendix G — `diff g20_canary.ac4e14f2.py g20_canary.py`

sha256 `67651ec59d34d1b233c719ad4183c54821536e97f22f3106dbe7324ad6a0c8d4`; `diff` exit 1 (differences found).

```diff
387c387,389
<                      "not checkpoint-selection evidence, no accuracy is reported from this run",
---
>                      "not checkpoint-selection evidence. G20 produces no accuracy result that is "
>                      "used, selected, interpreted, compared, or reported as thesis evidence; any "
>                      "one-sample random-init validation metric is a diagnostic side effect only",
```

### Appendix H — canary pre-CUDA evidence, new version, timestamps masked

sha256 `1b5778dc7e446d775c95ca949c9806423248735275a78ce48bbdb65c45abb99f`. `cmp` against the old version's masked evidence: identical. Run: `docker run --rm --network none -e PLANTSEG_IMAGE_DIGEST=sha256:cb413304… -e TORCH_HOME=/tmp/th -e TEACHER_WORK_DIR=/work/out plantseg-teacher:local python -B /work/canary.py`; exit 1 for both versions.

```json
{"event": "cublas_views", "libc_getenv": ":4096:8", "os.environ": ":4096:8", "proc_self_environ": ":4096:8", "utc": "X"}
{"check": "cublas_views_agree", "detail": "{'os.environ': ':4096:8', 'proc_self_environ': ':4096:8', 'libc_getenv': ':4096:8'}", "event": "require", "ok": true, "utc": "X"}
{"check": "cublas_inherited_from_image", "detail": "entry=':4096:8' - this harness never creates or repairs it", "event": "require", "ok": true, "utc": "X"}
{"check": "init_checkpoint_env_unset", "detail": "the canary must run with no teacher checkpoint", "event": "require", "ok": true, "utc": "X"}
{"check": "torch_home_set", "detail": "TORCH_HOME must point at a new empty external directory", "event": "require", "ok": true, "utc": "X"}
{"check": "torch_home_empty_before", "detail": "0 entries", "event": "require", "ok": true, "utc": "X"}
{"check": "image_digest_supplied", "detail": "set PLANTSEG_IMAGE_DIGEST in the pod template - the image cannot know its own digest", "event": "require", "ok": true, "utc": "X"}
{"check": "image_digest_matches", "detail": "got 'sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b'", "event": "require", "ok": true, "utc": "X"}
{"check": "image_commit_matches", "detail": "got '48bccc70caef4b8e252ffd11182c001f0f4368f5' - this is the image's build commit, NOT the runtime source", "event": "require", "ok": true, "utc": "X"}
{"check": "image_and_runtime_identities_differ", "detail": "they are intentionally different; the image's baked source predates G18", "event": "require", "ok": true, "utc": "X"}
{"digest": "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b", "event": "image_identity", "image_commit": "48bccc70caef4b8e252ffd11182c001f0f4368f5", "runtime_commit": "0bb69961dfedcb4d00ff42990a6543f5dda505ec", "utc": "X"}
{"check": "git_head_is_runtime_commit", "detail": "got '' (want 0bb69961dfedcb4d00ff42990a6543f5dda505ec)", "event": "require", "ok": false, "utc": "X"}
{"event": "G20_CANARY_RESULT", "failed": ["git_head_is_runtime_commit"], "reason": "git_head_is_runtime_commit: got '' (want 0bb69961dfedcb4d00ff42990a6543f5dda505ec)", "utc": "X", "verdict": "FAIL"}
```

### Appendix I — G20 `logs/step0.log` (Step 0 host health)

Bundle member; **source** sha256 `463d42e5a58e372ea6570d0ad0a0a276af7cd92229862723becaab979f054008`. Environment read by explicit variable name only. *Rendering note:* trailing spaces stripped from 1 line (the `nvidia-smi` date line); not byte-identical to the source — see the appendix preamble.

```text
=== G20 STEP 0 — host health ===
utc            : 2026-09-21T15:45:30Z
hostname       : a938809e082c
container_up_s : 4755609.33
RUNPOD_POD_ID    = [i6dg3nll0l8k00]
RUNPOD_DC_ID     = [CA-MTL-1]
RUNPOD_GPU_NAME  = [NVIDIA+RTX+A5000]
RUNPOD_GPU_COUNT = [1]
RUNPOD_CPU_COUNT = [9]
RUNPOD_MEM_GB    = [50]
RUNPOD_VOLUME_ID = [<unset>]
--- shell environment (explicit names)
shell PLANTSEG_IMAGE_DIGEST    = [sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b]
shell PLANTSEG_GIT_COMMIT      = [48bccc70caef4b8e252ffd11182c001f0f4368f5]
shell CUBLAS_WORKSPACE_CONFIG  = [:4096:8]
shell SEGNEXT_ADE20K_CKPT      = [<unset>]
shell PLANTSEG_DATA_ROOT       = [/workspace/plantseg_data/plantseg]
shell TEACHER_WORK_DIR         = [<unset>]
shell TORCH_HOME               = [<unset>]
shell G20_REPO                 = [<unset>]
shell CUDA_VISIBLE_DEVICES     = [<unset>]
shell PYTHONPATH               = [/workspace/plantseg-thesis]
--- PID 1 environment at container start (explicit names)
pid1 CUBLAS_WORKSPACE_CONFIG  = [:4096:8]
pid1 PLANTSEG_IMAGE_DIGEST    = [sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b]
pid1 PLANTSEG_GIT_COMMIT      = [48bccc70caef4b8e252ffd11182c001f0f4368f5]
pid1 SEGNEXT_ADE20K_CKPT      = [<unset>]
pid1 cmdline : /sbin/docker-init -- sleep infinity
--- nvidia-smi
NVIDIA RTX A5000, 24564 MiB, 8.6, 580.159.04, 00000000:D6:00.0
Mon Sep 21 15:45:30 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A5000               On  |   00000000:D6:00.0 Off |                  Off |
| 30%   22C    P8             25W /  230W |       1MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
--- device open, cuInit, torch CUDA (verdict computed from results)
open /dev/nvidiactl OK
open /dev/nvidia-uvm OK
open /dev/nvidia-uvm-tools OK
cuInit rc 0
python 3.11.16 | torch 2.1.0+cu121 | torch.version.cuda 12.1 | cudnn 8902
is_available True
device_count 1
props name=NVIDIA RTX A5000 total=23.55GiB sm_86 multiprocessors=64
tiny alloc + compute + synchronize OK 2048.0
STEP0 VERDICT: PASS
--- framework versions
mmseg 1.2.2 | mmcv 2.1.0 | mmengine 0.10.7 | torchvision 0.16.0+cu121 | numpy 1.26.4 | scipy 1.11.4
pip freeze: 77 lines, sha256 f162674a89a6668d551456dbb85e54ce2819a2ceb1b5043a4657ba0d89965620
--- cpu / memory / disk / mounts
nproc 96 | cpu.max [n/a]
/root/g20/bin/step0.sh: line 70: free: command not found
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   16M   30G   1% /
overlay          30G   16M   30G   1% /
no separate /workspace mount
--- dataset presence (existence tests only; nothing is listed)
absent /workspace/plantseg_data
absent /workspace/plantseg_data/plantseg
EXISTS /root/g20
--- image WORKDIR /workspace/plantseg-thesis
not a git checkout (image-baked source or shadowed) — will NOT be used
git: git version 2.39.5
=== STEP 0 END ===
```

### Appendix J — G20 `logs/stage.log` (frozen inputs, payload, guarded checkout, pre-run inventory)

Bundle member; sha256 `be6c043ee8b66e743b2162dd5c8d0cfb66f7487294ac271e56b8f3e41c514d7e`.

```text
=== frozen inputs ===
68b8bd00e5e406339d6b6ed3317718315eade55017a0f134fa46cd8af76f7ad0  /root/g20/in/g20_canary.py
6ebeefe9d56665b55657e4693bd32f0bbd85353507252700e360b7b523d0861e  /root/g20/in/p1_bmm_probe.py
49a28cce9c86bbadbcac44e12bf9513622b84caf9ddd0688557b7f4e1ddf41e7  /root/g20/in/g20_payload.tar
canary = new Phase-3 version (68b8bd00…); probe and payload tar match frozen values
=== payload (TRAIN + VAL only) ===
./g20_subset_manifest.json
./plantseg/annotations/train/apple_black_rot_1.png
./plantseg/annotations/val/apple_black_rot_28.png
./plantseg/images/train/apple_black_rot_1.jpg
./plantseg/images/val/apple_black_rot_28.jpg
TEST directories absent from payload root: images/test, annotations/test
train images      apple_black_rot_1.jpg      MATCH b5441c7d7fdbf6fba772f049bc2d5cdb7b9f9b779ec62d847bc319643d5ff418
train annotations apple_black_rot_1.png      MATCH 4fc9d2d52cdcc57cb5d1599eabfef019780834d2bfebefb0e3f9dae9d675d628
val   images      apple_black_rot_28.jpg     MATCH e701786b69ad7e0f94cb15c4c7b877f3af4d36955ad521c5d94ae817c45a5c52
val   annotations apple_black_rot_28.png     MATCH 5763bd1bd76d383c82c3844960e0d8903c72deb995859a30ff7d3adc423e9d25
manifest splits: ['train', 'val'] | test_split: NOT ENUMERATED, NOT LISTED, NOT READ, NOT HASHED, NOT STAGED
cc654bf752e998a5b9ea58e95b53be4df531cc8d91abd506afac5c51bb40121a  /root/g20/payload/g20_subset_manifest.json
=== guarded runtime checkout ===
HEAD = 0bb69961dfedcb4d00ff42990a6543f5dda505ec
commit 0bb69961dfedcb4d00ff42990a6543f5dda505ec | G18: add TeacherRunner and wire the teacher launcher's --launch path
governed runtime paths: clean
72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e  src/training/teacher_runner.py
5857527622ea3dc4b12fc316a410e8144eb06ffa5199e002ad1b3c957c89ac7e  scripts/launch_teacher_finetune.py
510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5  configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
__pycache__ dirs: 0
=== pre-run inventory (B58 §13) ===
inv_pre: 0 lines | sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
STAGE VERDICT: PASS
```

### Appendix K — G20 `logs/p1.log` (P1N and P1G8)

Bundle member; sha256 `26664cf2f9d750eac824148ec0ea8da2dbea950efffd7d19b9fd0afea1a1e652`.

```text
probe sha256: 6ebeefe9d56665b55657e4693bd32f0bbd85353507252700e360b7b523d0861e
=== P1N: CUBLAS_WORKSPACE_CONFIG removed from the process environment BEFORE python starts ===
==============================================================================
P-1 cuBLAS strict-mode mechanism probe
  CUBLAS_WORKSPACE_CONFIG via os.environ           = None
  CUBLAS_WORKSPACE_CONFIG via /proc/self/environ   = None
  CUBLAS_WORKSPACE_CONFIG via libc getenv          = None
  ARM = P-1N  (must_raise=True)
  torch 2.1.0+cu121 | cuda available True
  deterministic_algorithms       = True
  deterministic_algorithms_warn_only = False
  x     (16, 512, 4096) sha256 9a3fa5376a71c9dce19d814e84b45fb6
  bases (16, 512, 16) sha256 6b5416056decee38a8b46ecb9ac29054
  device NVIDIA RTX A5000 | cuda initialized True
  bmm RAISED RuntimeError:
    Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    identifies cuBLAS            : True
    names CUBLAS_WORKSPACE_CONFIG : True
RESULT: PASS — P-1N qualified the detector.
P1N_EXIT=0
=== P1G8: CUBLAS_WORKSPACE_CONFIG inherited from the image ENV (nothing set by hand) ===
==============================================================================
P-1 cuBLAS strict-mode mechanism probe
  CUBLAS_WORKSPACE_CONFIG via os.environ           = ':4096:8'
  CUBLAS_WORKSPACE_CONFIG via /proc/self/environ   = ':4096:8'
  CUBLAS_WORKSPACE_CONFIG via libc getenv          = ':4096:8'
  ARM = P-1G8  (must_raise=False)
  torch 2.1.0+cu121 | cuda available True
  deterministic_algorithms       = True
  deterministic_algorithms_warn_only = False
  x     (16, 512, 4096) sha256 9a3fa5376a71c9dce19d814e84b45fb6
  bases (16, 512, 16) sha256 6b5416056decee38a8b46ecb9ac29054
  device NVIDIA RTX A5000 | cuda initialized True
  bmm COMPLETED, output shape (16, 4096, 16) (expected (16, 4096, 16))
RESULT: PASS — P-1G8 completed with the image-supplied CUBLAS_WORKSPACE_CONFIG.
P1G8_EXIT=0
P1N expected-failure observed: 1 | P1G8 expected-success observed: 1
P1 VERDICT: PASS
```

### Appendix L — G20 `logs/canary.log` (canary launch wrapper)

Bundle member; sha256 `73b3ed8c41e8eba85e0f1769066d3be7e95ded940cc41a1b929bc7fbb9507f54`. Only run-location variables were set; cuBLAS and the image digest were inherited.

```text
canary env PLANTSEG_DATA_ROOT       = [/root/g20/payload/plantseg]
canary env TEACHER_WORK_DIR         = [/root/g20/work_g20]
canary env TORCH_HOME               = [/root/g20/torch_home_g20]
canary env G20_REPO                 = [/root/g20/plantseg-thesis]
canary env PYTHONPATH               = [/root/g20/plantseg-thesis]
canary env SEGNEXT_ADE20K_CKPT      = [<unset>]
canary env CUBLAS_WORKSPACE_CONFIG  = [:4096:8]
canary env PLANTSEG_IMAGE_DIGEST    = [sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b]
canary env PLANTSEG_GIT_COMMIT      = [48bccc70caef4b8e252ffd11182c001f0f4368f5]
canary start utc: 2026-09-21T15:47:39Z
canary end utc: 2026-09-21T15:47:49Z
CANARY_EXIT=0
```

### Appendix M — G20 full post-run audit log

`post_full_from_session.log`; **source** sha256 `358f6a854d9344de2bded805b185a088552262085caa2adbf9046d540a996abd` (= pod-side value). *Rendering note:* trailing spaces stripped from 10 lines (8 mmengine hook-table rows, the `Iter(val)` timing line, the `build_loss` deprecation line); not byte-identical to the source — see the appendix preamble. It contains the post-run inventory comparison, the side-effect checks, the complete canary evidence JSONL, the G18 attestation lines, the warnings, the diagnostic metric (NON-THESIS) and the launch provenance.

```text
=== post-run inventory (same command as pre-run) ===
inv_pre : 0 lines sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
inv_post: 0 lines sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
INVENTORY: IDENTICAL
HEAD 0bb69961dfedcb4d00ff42990a6543f5dda505ec
72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e  src/training/teacher_runner.py
5857527622ea3dc4b12fc316a410e8144eb06ffa5199e002ad1b3c957c89ac7e  scripts/launch_teacher_finetune.py
510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5  configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
__pycache__ in checkout: 0
=== external work dir (all files) ===
    177007  /root/g20/work_g20/20260921_154742/20260921_154742.log
       349  /root/g20/work_g20/20260921_154742/vis_data/20260921_154742.json
     30369  /root/g20/work_g20/20260921_154742/vis_data/config.py
       349  /root/g20/work_g20/20260921_154742/vis_data/scalars.json
      9194  /root/g20/work_g20/g20_canary_evidence.jsonl
     30369  /root/g20/work_g20/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
      3873  /root/g20/work_g20/teacher_launch_provenance.json
*.pth / *.pt in work dir: 0 | last_checkpoint: absent
=== TORCH_HOME after run ===
entries: 0
=== new files since the pre-canary marker, outside work dir / evidence / logs (/root, /tmp) ===
     37407  /root/.bash_history
     26961  /root/.cache/matplotlib/fontlist-v3.11.0.json
      3694  /root/g20/bin/post.sh
     24178  /root/g20/plantseg-thesis/.git/index
(end of list)
any new *.pth/*.pt/*.ckpt under /root or /tmp: 0
=== TEST checks ===
absent  payload/plantseg/images/test
absent  payload/plantseg/annotations/test
absent  /workspace/plantseg_data (no dataset volume on this pod)
config dump: /root/g20/work_g20/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
323:load_from = None
583:resume = False
584:test_cfg = None
585:test_dataloader = None
586:test_evaluator = None
'images/test' occurrences in dumped config: 0
'test' path tokens in canary stdout: 2
=== canary evidence file ===
{"event": "cublas_views", "libc_getenv": ":4096:8", "os.environ": ":4096:8", "proc_self_environ": ":4096:8", "utc": "2026-09-21T15:47:39Z"}
{"check": "cublas_views_agree", "detail": "{'os.environ': ':4096:8', 'proc_self_environ': ':4096:8', 'libc_getenv': ':4096:8'}", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "cublas_inherited_from_image", "detail": "entry=':4096:8' - this harness never creates or repairs it", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "init_checkpoint_env_unset", "detail": "the canary must run with no teacher checkpoint", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "torch_home_set", "detail": "TORCH_HOME must point at a new empty external directory", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "torch_home_empty_before", "detail": "0 entries", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "image_digest_supplied", "detail": "set PLANTSEG_IMAGE_DIGEST in the pod template - the image cannot know its own digest", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "image_digest_matches", "detail": "got 'sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b'", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "image_commit_matches", "detail": "got '48bccc70caef4b8e252ffd11182c001f0f4368f5' - this is the image's build commit, NOT the runtime source", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "image_and_runtime_identities_differ", "detail": "they are intentionally different; the image's baked source predates G18", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"digest": "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b", "event": "image_identity", "image_commit": "48bccc70caef4b8e252ffd11182c001f0f4368f5", "runtime_commit": "0bb69961dfedcb4d00ff42990a6543f5dda505ec", "utc": "2026-09-21T15:47:39Z"}
{"check": "git_head_is_runtime_commit", "detail": "got '0bb69961dfedcb4d00ff42990a6543f5dda505ec' (want 0bb69961dfedcb4d00ff42990a6543f5dda505ec)", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "governed_runtime_paths_clean", "detail": "clean", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "hash::src/training/teacher_runner.py", "detail": "got 72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "hash::scripts/launch_teacher_finetune.py", "detail": "got 5857527622ea3dc4b12fc316a410e8144eb06ffa5199e002ad1b3c957c89ac7e", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "hash::configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py", "detail": "got 510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "no_stale_pycache_in_checkout", "detail": "0 found", "event": "require", "ok": true, "utc": "2026-09-21T15:47:39Z"}
{"check": "launcher_module_from_checkout", "detail": "/root/g20/plantseg-thesis/scripts/launch_teacher_finetune.py", "event": "require", "ok": true, "utc": "2026-09-21T15:47:41Z"}
{"check": "launcher_self_hash_matches", "detail": "5857527622ea3dc4b12fc316a410e8144eb06ffa5199e002ad1b3c957c89ac7e", "event": "require", "ok": true, "utc": "2026-09-21T15:47:41Z"}
{"check": "teacher_runner_self_hash_matches", "detail": "72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e", "event": "require", "ok": true, "utc": "2026-09-21T15:47:41Z"}
{"check": "cuda_not_initialized_at_start", "detail": "", "event": "require", "ok": true, "utc": "2026-09-21T15:47:41Z"}
{"event": "gate", "name": "L.check_inherited_cublas", "ok": true, "utc": "2026-09-21T15:47:41Z"}
{"check": "config_sha_matches", "detail": "510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5", "event": "require", "ok": true, "utc": "2026-09-21T15:47:41Z"}
{"check": "work_dir_absolute_and_new", "detail": "/root/g20/work_g20 must not exist before the run. NOTE: a previous attempt writes the evidence file here, so EVERY retry needs a fresh TEACHER_WORK_DIR.", "event": "require", "ok": true, "utc": "2026-09-21T15:47:41Z"}
{"event": "gate", "gpu_count": 1, "gpu_names": null, "name": "L.check_cuda", "utc": "2026-09-21T15:47:42Z"}
{"check_init_checkpoint": "canary has no checkpoint (load_from=None)", "check_splits": "it enumerates TEST filenames; the canary must not", "event": "gates_deliberately_skipped", "utc": "2026-09-21T15:47:42Z"}
{"check": "data_present::images/train", "detail": "/root/g20/payload/plantseg/images/train", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "data_present::annotations/train", "detail": "/root/g20/payload/plantseg/annotations/train", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "data_present::images/val", "detail": "/root/g20/payload/plantseg/images/val", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "data_present::annotations/val", "detail": "/root/g20/payload/plantseg/annotations/val", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "test_split_absent::images/test", "detail": "/root/g20/payload/plantseg/images/test must not exist on this pod - the canary never enumerates TEST", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "test_split_absent::annotations/test", "detail": "/root/g20/payload/plantseg/annotations/test must not exist on this pod - the canary never enumerates TEST", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"event": "tripwires_armed", "targets": ["mmengine CheckpointLoader.load_checkpoint", "mmengine _load_checkpoint", "mmengine load_checkpoint", "torch.hub.load_state_dict_from_url"], "utc": "2026-09-21T15:47:42Z"}
{"check": "tripwires_armed", "detail": "['mmengine CheckpointLoader.load_checkpoint', 'mmengine _load_checkpoint', 'mmengine load_checkpoint', 'torch.hub.load_state_dict_from_url']", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "cfg_resume_false", "detail": "False", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "cfg_randomness_unchanged", "detail": "{'seed': 42, 'deterministic': True}", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "cfg_env_cfg_cudnn_benchmark_unchanged", "detail": "kept True so the TeacherRunner override is observable", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "cfg_backbone_init_cfg_none", "detail": "None", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "cfg_load_from_none", "detail": "None", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "cfg_checkpoint_hook_disabled", "detail": "key retained and set to None", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"check": "cfg_test_surfaces_none", "detail": "", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
{"event": "hooks_before_train", "hooks": ["RuntimeInfoHook", "IterTimerHook", "DistSamplerSeedHook", "SegVisualizationHook", "G20CanaryProbeHook", "LoggerHook", "ParamSchedulerHook", "TeacherDeterminismAttestationHook"], "utc": "2026-09-21T15:47:47Z"}
{"attestation_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_benchmark": false, "cudnn_deterministic": true, "deterministic_algorithms": true, "deterministic_algorithms_warn_only": true}, "attestation_present": true, "batch_idx": 0, "cuda_initialized": true, "event": "cuda_train_step_completed", "max_mem_alloc_bytes": 2641997312, "outputs": {"decode.acc_seg": 4.8290252685546875, "decode.loss_ce": 4.558577537536621, "loss": 4.558577537536621}, "utc": "2026-09-21T15:47:48Z"}
{"attestation_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_benchmark": false, "cudnn_deterministic": true, "deterministic_algorithms": true, "deterministic_algorithms_warn_only": true}, "attestation_present": true, "batch_idx": 0, "cuda_initialized": true, "event": "cuda_val_step_completed", "max_mem_alloc_bytes": 2141736448, "outputs": null, "utc": "2026-09-21T15:47:48Z"}
{"check": "launch_returned_zero", "detail": "rc=0", "event": "require", "ok": true, "utc": "2026-09-21T15:47:48Z"}
{"check": "no_checkpoint_artifact", "detail": "[]", "event": "require", "ok": true, "utc": "2026-09-21T15:47:48Z"}
{"check": "no_last_checkpoint_marker", "detail": "", "event": "require", "ok": true, "utc": "2026-09-21T15:47:48Z"}
{"check": "torch_home_empty_after", "detail": "[]", "event": "require", "ok": true, "utc": "2026-09-21T15:47:48Z"}
{"check": "no_checkpoint_load_in_log", "detail": "mmengine logs that string whenever a checkpoint is loaded", "event": "require", "ok": true, "utc": "2026-09-21T15:47:48Z"}
{"check": "train_step_completed", "detail": "", "event": "require", "ok": true, "utc": "2026-09-21T15:47:48Z"}
{"check": "val_step_completed", "detail": "", "event": "require", "ok": true, "utc": "2026-09-21T15:47:48Z"}
{"event": "G20_CANARY_RESULT", "seconds": 9.640098, "utc": "2026-09-21T15:47:48Z", "verdict": "PASS"}
=== G18 attestation log lines ===
09/21 15:47:47 - mmengine - INFO - G18 determinism attestation: {"batch_idx": 0, "cuda_initialized": true, "point": "first_train_iter", "policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_benchmark": false, "cudnn_deterministic": true, "deterministic_algorithms": true, "deterministic_algorithms_warn_only": true}, "pre_cuda_policy_established": true, "runner_class": "src.training.teacher_runner.TeacherRunner", "seed": 42, "teacher_runner_module": {"path": "/root/g20/plantseg-thesis/src/training/teacher_runner.py", "sha256": "72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e"}}
09/21 15:47:48 - mmengine - INFO - G18 determinism attestation: {"batch_idx": 0, "cuda_initialized": true, "point": "first_val_iter", "policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_benchmark": false, "cudnn_deterministic": true, "deterministic_algorithms": true, "deterministic_algorithms_warn_only": true}, "pre_cuda_policy_established": true, "runner_class": "src.training.teacher_runner.TeacherRunner", "seed": 42, "teacher_runner_module": {"path": "/root/g20/plantseg-thesis/src/training/teacher_runner.py", "sha256": "72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e"}}
=== hooks / runner / loop lines ===
361:    dict(type='G20CanaryProbeHook'),
1062:(NORMAL      ) G20CanaryProbeHook
1082:(NORMAL      ) G20CanaryProbeHook
1102:(NORMAL      ) G20CanaryProbeHook
1142:09/21 15:47:47 - mmengine - INFO - Hooks after G18 registration:
1145:(NORMAL      ) G20CanaryProbeHook
1160:(LOWEST      ) TeacherDeterminismAttestationHook
1166:(NORMAL      ) G20CanaryProbeHook
1182:(LOWEST      ) TeacherDeterminismAttestationHook
1187:(NORMAL      ) G20CanaryProbeHook
1563:[evidence] {"event": "hooks_before_train", "hooks": ["RuntimeInfoHook", "IterTimerHook", "DistSamplerSeedHook", "SegVisualizationHook", "G20CanaryProbeHook", "LoggerHook", "ParamSchedulerHook", "TeacherDeterminismAttestationHook"], "utc": "2026-09-21T15:47:47Z"}
1568:09/21 15:47:48 - mmengine - INFO - Iter(train) [1/1]  base_lr: 6.0000e-11 lr: 6.0000e-11  eta: 0:00:00  time: 1.0127  data_time: 0.0532  memory: 2519  loss: 4.5586  decode.loss_ce: 4.5586  decode.acc_seg: 4.8290
1577:09/21 15:47:48 - mmengine - INFO - Iter(val) [1/1]    eta: 0:00:00  time: 0.1001  data_time: 0.0085  memory: 2042
1700:09/21 15:47:48 - mmengine - INFO - Iter(val) [1/1]    aAcc: 0.0000  mIoU: 0.0000  mAcc: 0.0000  data_time: 0.0085  time: 0.1001
=== warnings (nondeterminism / others) ===
39:[evidence] {"check": "cfg_randomness_unchanged", "detail": "{'seed': 42, 'deterministic': True}", "event": "require", "ok": true, "utc": "2026-09-21T15:47:42Z"}
83:    deterministic: True
671:randomness = dict(deterministic=True, seed=42)
1052:/usr/local/lib/python3.11/site-packages/mmseg/models/builder.py:36: UserWarning: ``build_loss`` would be deprecated soon, please use ``mmseg.registry.MODELS.build()``
1053:  warnings.warn('``build_loss`` would be deprecated soon, please use '
1054:/usr/local/lib/python3.11/site-packages/mmseg/models/losses/cross_entropy_loss.py:250: UserWarning: Default ``avg_non_ignore`` is False, if you would like to ignore the certain label and average loss over non-ignore labels, which is the same with PyTorch official cross_entropy, set ``avg_non_ignore=True``.
1055:  warnings.warn(
1057:/usr/local/lib/python3.11/site-packages/mmseg/engine/hooks/visualization_hook.py:60: UserWarning: The draw is False, it means that the hook for visualization will not take effect. The results will NOT be visualized or stored.
1058:  warnings.warn('The draw is False, it means that the '
1227:/usr/local/lib/python3.11/site-packages/mmseg/datasets/transforms/loading.py:83: UserWarning: `reduce_zero_label` will be deprecated, if you would like to ignore the zero label, please set `reduce_zero_label=True` when dataset initialized
1228:  warnings.warn('`reduce_zero_label` will be deprecated, '
1566:[evidence] {"attestation_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_benchmark": false, "cudnn_deterministic": true, "deterministic_algorithms": true, "deterministic_algorithms_warn_only": true}, "attestation_present": true, "batch_idx": 0, "cuda_initialized": true, "event": "cuda_train_step_completed", "max_mem_alloc_bytes": 2641997312, "outputs": {"decode.acc_seg": 4.8290252685546875, "decode.loss_ce": 4.558577537536621, "loss": 4.558577537536621}, "utc": "2026-09-21T15:47:48Z"}
1570:/usr/local/lib/python3.11/site-packages/mmseg/evaluation/metrics/iou_metric.py:190: UserWarning: _histc_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
1572:/usr/local/lib/python3.11/site-packages/mmseg/evaluation/metrics/iou_metric.py:193: UserWarning: _histc_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
1574:/usr/local/lib/python3.11/site-packages/mmseg/evaluation/metrics/iou_metric.py:196: UserWarning: _histc_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
1576:[evidence] {"attestation_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_benchmark": false, "cudnn_deterministic": true, "deterministic_algorithms": true, "deterministic_algorithms_warn_only": true}, "attestation_present": true, "batch_idx": 0, "cuda_initialized": true, "event": "cuda_val_step_completed", "max_mem_alloc_bytes": 2141736448, "outputs": null, "utc": "2026-09-21T15:47:48Z"}
=== diagnostic validation metric (NON-THESIS) ===
1038:        'mIoU',
1700:09/21 15:47:48 - mmengine - INFO - Iter(val) [1/1]    aAcc: 0.0000  mIoU: 0.0000  mAcc: 0.0000  data_time: 0.0085  time: 0.1001
=== launch provenance ===
{
  "stage": "teacher",
  "recorded_utc": "2026-09-21T15:47:42Z",
  "cublas_at_launcher_entry": ":4096:8",
  "cublas_exec_environ": ":4096:8",
  "exec_environ_readable": true,
  "launcher": {
    "path": "/root/g20/plantseg-thesis/scripts/launch_teacher_finetune.py",
    "sha256": "5857527622ea3dc4b12fc316a410e8144eb06ffa5199e002ad1b3c957c89ac7e"
  },
  "teacher_runner_module": {
    "path": "/root/g20/plantseg-thesis/src/training/teacher_runner.py",
    "sha256": "72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e"
  },
  "config": {
    "path": "/root/g20/plantseg-thesis/configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py",
    "sha256": "510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5"
  },
  "checkout_git_head": {
    "commit": "0bb69961dfedcb4d00ff42990a6543f5dda505ec",
    "error": null
  },
  "image_env_plantseg_git_commit": "48bccc70caef4b8e252ffd11182c001f0f4368f5",
  "image_digest_declared": "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b",
  "requested_randomness": {
    "seed": 42,
    "deterministic": true
  },
  "requested_env_cfg_cudnn_benchmark": true,
  "cuda_initialized_before_runner": false,
  "preflight": {
    "stage": "teacher",
    "architecture": "SegNeXt-B / MSCAN-B",
    "protocol_classification": "thesis-derived",
    "public_plantseg_source_commit": "1a3dd4d9224bcc97a5850af7dd1c423abc24eae0",
    "config_path": "/root/g20/plantseg-thesis/configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py",
    "config_sha256": "510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5",
    "plantseg_root": "/root/g20/payload/plantseg",
    "split_counts": {
      "canary": "check_splits skipped; TEST never enumerated"
    },
    "ade20k_init_checkpoint": {
      "canary": "NO CHECKPOINT - load_from=None",
      "path": null,
      "sha256": null
    },
    "work_dir": "/root/g20/work_g20",
    "seed": 42,
    "optimizer": "AdamW lr=6e-5 wd=0.01 betas=(0.9,0.999) head_lr_mult=10",
    "max_iters": 40000,
    "val_interval": 10000,
    "preprocessing": "512x512 aspect-preserving resize + ImageNet-mean-equivalent pad (post-normalisation), ignore_index=255",
    "augmentation_source": "public-plantseg-segnext-family",
    "loss": "CrossEntropyLoss only",
    "checkpoint_selection": "validation mIoU (test split never used)",
    "versions": {
      "python": "3.11.16",
      "torch": "2.1.0+cu121",
      "mmsegmentation": "1.2.2",
      "mmcv": "2.1.0"
    },
    "cuda": {
      "gpu_count": 1,
      "gpu_names": null
    },
    "recorded_utc": "2026-09-21T15:47:42Z",
    "g20_canary": {
      "runtime_commit": "0bb69961dfedcb4d00ff42990a6543f5dda505ec",
      "hashes": {
        "src/training/teacher_runner.py": "72206af7e00939bd02f69c99fc5f428adeca10911d657e72279dce784c1a2d0e",
        "scripts/launch_teacher_finetune.py": "5857527622ea3dc4b12fc316a410e8144eb06ffa5199e002ad1b3c957c89ac7e",
        "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py": "510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5"
      },
      "tripwires": [
        "mmengine CheckpointLoader.load_checkpoint",
        "mmengine _load_checkpoint",
        "mmengine load_checkpoint",
        "torch.hub.load_state_dict_from_url"
      ],
      "image": {
        "digest": "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b",
        "image_commit": "48bccc70caef4b8e252ffd11182c001f0f4368f5"
      },
      "torch_home": "/root/g20/torch_home_g20",
      "grade": "development-canary only - not an E1-E7 experiment, not a Chapter IV result, not checkpoint-selection evidence. G20 produces no accuracy result that is used, selected, interpreted, compared, or reported as thesis evidence; any one-sample random-init validation metric is a diagnostic side effect only"
    }
  }
}=== evidence bundle ===
bundle sha256 415dd7d705ba8c3929af3b7c17b357ca1b78c55e5288f8f61bae17f788fdeeaf bytes 62592
```
