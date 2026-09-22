# B61 — Teacher methodology locks: M4 (NMF/Hamburger RNG), M12 (checkpoint selection), M13 (CE ignore normalisation)

**Date:** 2026-09-22. **Type:** decision record (governance). **Repository authority:** branch
`claude/keen-curie-u4a8ig`, HEAD = origin = `161e735fe54180812aef2a47180b00493ea21ca4`.

**Status:** M4, M12 and the new M13 are **LOCKED** by this record. The decisions are **not implemented in
runtime code**. They join M2, M3, M5 and M11 (B60) in the single governed runtime change at §9 step 2.
**OFFICIAL TEACHER = NO-GO.**

**Scope of this phase:** documentation only.
- No runtime, config, `src/**`, `scripts/**` or smoke file changed.
- No training, no TEST access, no GPU, no RunPod.
- The downloaded checkpoint was not modified and is not embedded; it is referenced by path and SHA-256.

**Evidence labels:** MANUSCRIPT-SPECIFIED (MS) · PRIMARY-SOURCE-SUPPORTED · UPSTREAM-SOURCE · REPO-SOURCE ·
MEASURED · INFERRED · THESIS-DERIVED.

---

## 1. Checkpoint readiness — PASS (step D of B60 §9)

External evidence: `C:\Users\admin\plantseg_runs\teacher_checkpoint_readiness_20260922\`
(`SHA256SUMS` sha256 `3b80d584…d998`, 26 entries, verified; `PROVENANCE.md` sha256 `298fa9ca…06f4`).

| Field | Value | Label |
|---|---|---|
| MMSeg 1.x config | `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512` (pinned file `33dcb71b…`, byte-identical to upstream tag v1.2.2 and to the wheel RECORD hash) | UPSTREAM-SOURCE |
| Checkpoint | `segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth` | UPSTREAM-SOURCE |
| URL | `https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth` | UPSTREAM-SOURCE |
| Downloaded | 2026-09-22T02:46:54Z–02:50:45Z UTC; HTTP 200, 0 redirects, `application/octet-stream`; 110,977,141 bytes = Content-Length | MEASURED |
| **SHA-256** | **`647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1`** | MEASURED |
| Transport integrity | MD5 `53b45828…7c79` = server Content-MD5 / ETag | MEASURED |
| Reported ADE20K mIoU | 48.03 SS / 49.68 MS+flip | UPSTREAM-SOURCE |
| Stock 150-class init test | `scripts/test_teacher_init.py` at HEAD, pinned image, CPU, `--network none`: **RESULT: PASS**; missing 0 / unexpected 0; `conv_seg` 150; 27,636,310 params | MEASURED |
| 150 → 116 key audit | 854 tensors; 852 loaded bitwise; only `decode_head.conv_seg.weight` [150,512,1,1]→[116,512,1,1] and `.bias` [150]→[116] mismatch, left at fresh init | MEASURED |
| G11 (init-script import guard) | does not affect execution in the pinned image (numpy present); script not edited | MEASURED |

**Filename-suffix finding.** The suffix `b6f6c70c` is not the SHA-256 prefix. All four SegNeXt ADE20K
objects carry Last-Modified 2023-02-24, 10–15 days after their filename dates, and the MMSeg README says
the SegNeXt checkpoints were key-converted and re-uploaded (INFERRED). No official SHA-256 exists; the
value above is the first record (trust-on-first-use), anchored by HTTPS from the official host and the
server MD5 match. The suffix is not treated as cryptographic verification.

**Loading safety.** A static pickle scan (no execution) found only `collections.OrderedDict`,
`torch.FloatStorage`, `torch.LongStorage` and `torch._utils._rebuild_tensor_v2`; a `weights_only` load
succeeds (MEASURED).

---

## 2. Real-checkpoint NMF measurement (step E)

External evidence: `C:\Users\admin\plantseg_runs\teacher_m4_nmf_measurement_20260922\`
(`SHA256SUMS` sha256 `00664954…62df`, 25 entries, verified). Design pre-registered and hash-sealed before
any measurement (`PREREGISTRATION_M4_MEASUREMENT.md`, `df8661ba…098b`). Authoritative result
`out/m4_result.json` (`1da47b3c…4236`) from script v2 (`65ddc90e…d24f`). A v1 run is kept as superseded:
one field was read after a later reseed (a script-ordering bug). Every other v1 value is bit-identical to
v2, which doubles as a whole-run reproducibility check.

**Setup.** Pinned image `cb413304…c32c2b`; torch 2.1.0, mmseg 1.2.2, `ham_head.py` `eb2f0963…`; CPU,
4 threads, `--network none`. Model: the **stock 150-class** checkpoint (trained classifier). Input: VAL
`apple_black_rot_28` (the B59 image, byte-identical) through `core_preprocess` + `finalize`, the M3
clean-VAL canvas. Only `images/val` and `annotations/val` were mounted.

| Measurement | Result | Label |
|---|---|---|
| P0: 20 eval forwards, RNG not reset | **20/20 distinct.** max \|Δlogit\| 22.28 vs max \|logit\| 74.91 (relative 0.297 max, 0.180 mean). Predicted class changes on **26.0%** of valid 512² pixels on average (median 25.5%, max 35.6%); 64×64 grid 30.1%; all 190 pairs 23.9% | MEASURED |
| Pre-classifier (`align` output) / NMF output | relative Frobenius change 0.206 / 0.199 on average (max 0.263 / 0.248) | MEASURED |
| B59 random-init comparison | 0.137 vs 0.554 (relative 0.248); 10.7% of pixels (max 13.2%). The real checkpoint is **larger**, not smaller | MEASURED |
| Four rule-selected VAL images | mean pixel change 26.9%, 0.003%, 7.8%, 3.2%; pre-classifier change 0.22, 0.15, 0.24, 0.11; tracks the logit margin (median 1.42, 5.47, 4.12, 2.70) | MEASURED |
| RNG consumption | each eval forward = exactly one `torch.rand((B, 512, 16))` on the CPU default generator (B = 1, 2, bit-exact); numpy and python RNG unchanged | MEASURED |
| CUDA RNG | not measurable (no GPU). Bases are drawn on the CPU and moved with `.to(device)` (`ham_head.py:123`; the original SegNeXt code uses `torch.rand(...).cuda()`) | UPSTREAM-SOURCE / INFERRED |
| Batch position | same RNG state: alone ≡ position 0 (≤1.9e-5); position 1 differs (20.1% px). With a fixed basis all positions are identical (≤1.5e-5): the dependence is entirely the slice of the draw each position receives | MEASURED |
| P1 pass-level seed | repeated passes bit-identical; seed 42 vs 43 differs; reversed order differs; **batch size 2 ≡ batch size 1** (the CPU generator fills serially, so an image's basis is fixed by its offset in the stream since seeding); a restore variant returns the caller RNG exactly | MEASURED |
| P2 per-forward save/fixed-state/restore | 20/20 identical; caller RNG transparent (P0 perturbs it); position 1 differs (22.1% px); at batch 1 ≡ P3 with the same generator state | MEASURED |
| P2b per-sample keyed bases | repeat identical; alone ≡ position 0 ≡ position 1; caller RNG untouched | MEASURED (scratch) |
| P3 `rand_init=False` | 20/20 identical; batch-invariant; a typical single draw (20.9% px vs P0 draws); basis seed changes → 18.9% px. The lazily registered `bases` buffer is saved but dropped on reload (mmengine "unexpected key"; strict load raises), so a new basis is silently drawn | MEASURED |
| P4 K-draw averaging | variance ratio K4 0.26, K8 0.12; vs a 64-draw reference 14.1% / 8.5% / 6.1% px for K1/K4/K8; cost K× (≈1.3× / 1.8× with a shared backbone, CPU) | MEASURED |
| 116-class thesis model | DIAGNOSTIC ONLY: pre-classifier and NMF outputs bit-identical to the stock model under the same RNG | MEASURED |

**Limitation.** The stock classifier predicts ADE20K classes on out-of-domain PlantSeg images, with small
margins. Pixel-change percentages therefore do not forecast the fine-tuned teacher. The ~20% feature-level
variation is the architecture- and weights-level measure, and it is large enough that evaluation must not
depend on accidental global RNG state.

---

## 3. Source basis

- **PRIMARY-SOURCE-SUPPORTED.** Geng et al., *Is Attention Better Than Matrix Decomposition?* (ICLR 2021),
  Appendix F, Table 8: NMF dictionary initialisation on PASCAL VOC (5 runs, best(mean)) — fixed 77.4
  (77.3), learned 76.8 (76.5), **random 78.3 (77.8)**, online 77.8 (77.5). The authors attribute the benefit
  of random initialisation to the network adapting to different initialisations, "acting like an inner
  augmentation". The paper does not address evaluation-time control.
- **UPSTREAM-SOURCE.** The original SegNeXt and Enjoy-Hamburger light-ham heads default to `RAND_INIT=True`
  (their `ETA` is defined but unused: no online update). mmseg 1.2.2 `NMF2D._build_bases` draws
  `torch.rand((B*S, D, R))` per forward. The MMSeg SegNeXt README says test results vary because NMF is
  initialised randomly and advises setting the random seed at test time. The checkpoint's own meta config
  has `rand_init=True` (MEASURED).
- **MS.** ch3 is silent on NMF control (B59 B9).

---

## 4. LOCKED — M4: random NMF, preserved and isolated

**Principle.** The NMF/Hamburger algorithm remains **`rand_init = True`**. Randomness is **preserved** but
**isolated**. Not adopted: `rand_init=False`; a persisted fixed basis; multi-draw averaging as the primary
evaluator; image-keyed or per-sample deterministic bases as the primary method.

**Rationale.**
- The Hamburger primary paper favours random initialisation and describes it as an inner augmentation
  (PRIMARY-SOURCE-SUPPORTED).
- The thesis teacher preserves upstream SegNeXt semantics as far as possible (THESIS-DERIVED).
- The measured real-checkpoint NMF variability (§2) is too large to leave evaluation dependent on
  accidental global RNG state (MEASURED).

### M4-T — teacher fine-tuning

Upstream behaviour: `rand_init=True`, fresh NMF bases on every training forward, drawn from the run's
seeded global CPU stream. Bases are not frozen or keyed per image during training. No change is made
merely to remove training-time NMF stochasticity.

### M4-V — VAL / checkpoint selection / R3 / final teacher evaluation

A **dedicated CPU NMF RNG stream**. At the beginning of **every complete evaluation pass**:
1. save the caller/global CPU RNG state;
2. initialise the dedicated NMF sequence from **seed 42**;
3. evaluate the entire split in a **frozen deterministic sample order**;
4. use **batch_size = 1**;
5. let `rand_init=True` draw a fresh NMF basis for each image, as upstream normally does;
6. after the complete pass, **restore the caller/global CPU RNG state exactly**.

Consequences:
- every checkpoint sees the same ordered sequence of NMF draws;
- repeated evaluation passes are bitwise reproducible under the same runtime;
- evaluation does not alter the subsequent training RNG state;
- NMF still uses random initialisation, and no fixed basis is introduced.

The evaluation manifest/order must be **persisted or hash-attested**. Changing the manifest/order, the
batch size or the seed **invalidates comparability** unless explicitly disclosed and rerun consistently.

### M4-KD — frozen teacher during E2/E3

The frozen teacher retains `rand_init=True`, but NMF must **not consume or perturb the student's
caller/global CPU RNG**. A **private NMF RNG stream**, initialised **once** at the beginning of the E2/E3
run from **seed 42**. Before each frozen-teacher forward:
- (a) save the caller CPU RNG state;
- (b) install the current private NMF RNG state;
- (c) run the teacher forward;
- (d) capture the advanced NMF RNG state;
- (e) restore the caller CPU RNG state.

It is **not** reset to the same seed on every batch. Thus NMF receives fresh bases on successive teacher
forwards; the sequence is deterministic for a fixed call sequence; the student's RNG stream is not
perturbed; and E2 and E3 receive the same NMF sequence when their teacher-call order and batch protocol
are the same. The P2 behaviour (identical basis on every forward) is **not** the KD policy.

### Implementation requirements that follow from the lock (THESIS-DERIVED; verified at step 2 of §9)

- **The dedicated stream has exactly one consumer: NMF.** An image's basis is fixed by its offset in the
  stream (MEASURED, §2 P1), so any other CPU-RNG draw between the pass-start seeding and an NMF call shifts
  every later image. The robust realisation installs the private state only around teacher decode
  forwards (the M4-KD mechanism), with the V stream re-initialised to 42 at each pass start. INFERRED risk
  to test: a DataLoader without its own generator draws its base seed from the default CPU generator when
  its iterator is first created, which under a naive whole-pass install would offset the first pass only.
- **CPU-only seeding.** Initialise from a CPU generator state (for example a `torch.Generator` seeded 42,
  installed with `torch.set_rng_state`), never `torch.manual_seed`, which also seeds the CUDA generators and
  would alter training's CUDA RNG.
- **Unchanged:** the G18 seam (`TeacherRunner.set_randomness`, global seed 42, determinism flags).
- **R3 and the repository evaluator** apply M4-V in their own evaluation path.

---

## 5. Rejected alternatives

| Alternative | Measured behaviour | Why not adopted |
|---|---|---|
| **G / P2b — per-sample keyed bases** | highly reproducible; order-, batch- and position-invariant; caller-RNG-transparent | It replaces upstream **sequential random initialisation** with a new **image-identity-dependent basis-generation algorithm** — more intrusive than necessary for this teacher. M4-V obtains the needed reproducibility and common draws across checkpoints with the upstream draw rule, at the accepted price of a frozen order and batch size 1. Kept as a documented rejected alternative |
| **P3 / D — `rand_init=False`, fixed basis** | deterministic, batch-invariant; one frozen draw (18.9% px between basis seeds); reload drops the lazy buffer | Changes configured SegNeXt behaviour; fixed initialisation is worse than random in the Hamburger ablation; persistence hazard |
| **P2 — identical reset before every forward** | deterministic; at batch 1 identical to P3 | A fixed-basis policy in disguise; rejected for V and KD |
| **P4 / E — K-draw averaging** | variance reduction ≈1/K; K× compute | Turns the model into an internal ensemble estimator and adds substantial compute. Retained only as descriptive evidence; not for checkpoint selection, R3, the final teacher metric or KD targets |
| **A — upstream behaviour unchanged at evaluation** | outputs depend on accidental global RNG state; perturbs the caller stream | Checkpoints would be compared under different draws; kept for training only (M4-T) |

---

## 6. LOCKED — M12: teacher checkpoint selection

**Validations** occur every 4,000 iterations — **4k, 8k, …, 40k** (10 passes). Every checkpoint-selection
validation uses **M4-V**:
- NMF seed 42 at the beginning of every full VAL pass;
- the same frozen VAL manifest and exact order;
- batch size 1; no shuffle;
- `rand_init=True`;
- the same preprocessing (M3 clean-VAL canvas);
- the same all-class dataset-level mIoU computation;
- the caller RNG restored afterwards.

**Selection rule.**
- Select the checkpoint with the **numerically highest VAL all-class mIoU**.
- If two or more checkpoints have **exactly the same stored mIoU value**, select the **earliest
  iteration**.
- No tolerance or noise-band tie rule. No TEST inspection. No training extension because of the result.

**Persist:** every validation iteration; its all-class mIoU; disease-only mIoU where reporting requires
it; the NMF evaluation seed; the VAL manifest hash/order identity; and the selected iteration and
checkpoint SHA-256.

**R3.** After training, load the selected checkpoint and perform the controlled R3 re-evaluation under
M4-V (EVALUATION_CONTRACT §7.2–§7.3). R3 remains a same-VAL **operational competence floor**: not an
independent estimate, not inferential; teacher VAL all-class mIoU must exceed **0.36314016580581665**,
with no additional margin; failure = **STOP and methodology escalation**; no automatic retraining and no
TEST. R3's repository-evaluator mIoU is **not required to be byte-identical** to the MMSeg
checkpoint-selection metric unless both are proven to use the exact same evaluation implementation.

### Implementation requirements that follow from the lock (THESIS-DERIVED unless labelled)

- **Full precision.** MMSeg's `IoUMetric` rounds its summary mIoU to two decimals of a percent
  (`iou_metric.py:136`, `np.round(np.nanmean(...) * 100, 2)`; UPSTREAM-SOURCE). Comparing that value would
  make "numerically highest" and "exactly equal" operate at 0.01-pp resolution, an implicit tolerance band.
  The stored and compared selection value must be the **unrounded** dataset-level all-class mIoU.
- **Ties.** `CheckpointHook` with `rule='greater'` compares strictly (`x > y`, `checkpoint_hook.py:123`;
  UPSTREAM-SOURCE), so an equal later value does not replace the earlier best — consistent with
  earliest-on-tie once the compared value is full precision.
- **Every validated state recoverable.** The checkpoint interval must cover every validation iteration
  (4,000); the current runtime saves every 10,000.
- **Order identity.** MMSeg's `BaseSegDataset` sorts its data list by `img_path` (UPSTREAM-SOURCE); the
  manifest hash must be recorded with each validation.

---

## 7. NEW — M13: teacher cross-entropy ignore/padding normalisation — LOCKED

**Finding (MEASURED, pinned mmseg 1.2.2).** The thesis teacher config's `loss_decode` has no
`avg_non_ignore` key, so the default `False` applies. On a synthetic batch with 37.5% ignore pixels:
- ignore-255 pixels have zero individual loss and zero gradient;
- the mmseg loss equals the mean over **all** pixels (3.3758), i.e. 0.625× the mean over valid pixels,
  where 0.625 is the valid fraction;
- source: `cross_entropy()` sets `avg_factor = label.numel()` unless `avg_non_ignore` (UPSTREAM-SOURCE).

The loss scale therefore changes with the padded fraction. This conflicts with the manuscript: padding
"serves as the ignore label, excluding it from all loss calculations" (ch3 p. 128; MS). It also differs
from E1, whose `F.cross_entropy(..., ignore_index=255)` averages over valid pixels (`src/training/losses.py:53`;
REPO-SOURCE).

**M13 — LOCKED 2026-09-22.**
- Teacher decode-head CE uses **`avg_non_ignore = True`** with **`ignore_index = 255`**.
- Ignore/padded 255 pixels contribute to **neither** the numerator **nor** the mean denominator; teacher
  CE is averaged over valid, non-ignore pixels.
- Class weighting is unchanged (unweighted, M5) unless separately specified.
- This is a **thesis-derived correction**, not a claim about Wei or upstream SegNeXt (whose configs use the
  default).
- **M13 must be implemented before the official teacher run.**

Note: with `avg_non_ignore=True` an all-ignore batch gives `sum / (0 + eps)` = 0, not NaN
(`weight_reduce_loss`; UPSTREAM-SOURCE). M13 concerns the teacher loss only; E1–E3 student losses are
unchanged.

---

## 8. Runtime consequences for the unified implementation (§9 step 2)

1. **Teacher config.** Keep the inherited `ham_kwargs.rand_init=True`; replace the
   `NMF_SEED_CONTROL = 'NEED_TO_CONFIRM'` placeholder with the M4 record; add `avg_non_ignore=True` to
   `loss_decode`; checkpoint interval 4,000 covering every validation; selection on the full-precision
   all-class mIoU; VAL `batch_size=1`, no shuffle, frozen manifest; plus the B60 M2/M3/M5/M11 changes.
2. **M4-V mechanism** in the teacher runner path: pass-start seeding of the private stream from 42,
   install around NMF/decode forwards only, exact caller-RNG restore, CPU-only seeding, per-validation
   attestation of seed and manifest hash.
3. **Selection record:** per-validation JSON (iteration, full-precision all-class mIoU, disease-only where
   required, NMF seed, manifest hash, checkpoint SHA-256) and the selected iteration.
4. **R3 evaluator path** (repository evaluator with the teacher adapter) under M4-V.
5. **M4-KD** in the frozen-teacher wrapper used by E2/E3: private stream initialised once from 42;
   save/install/capture/restore around each forward.
6. **Launcher provenance:** record the M4/M12/M13 policy; M11 TRAIN/VAL-only checks; the readiness
   SHA-256 `647a0cda…40ef1` as the expected init checkpoint.
7. **Tests:**
   - existing smokes rewritten to the locks;
   - M4: two consecutive VAL passes bitwise identical, including the first; exact caller-CPU-RNG restore
     and CUDA RNG untouched; KD stream caller-transparent with fresh bases on successive forwards and a
     deterministic sequence; negative controls;
   - M13: the loss equals the valid-pixel mean, and an all-ignore batch is finite.
8. **Freeze** runtime commit and hashes; G20-style CUDA re-canary if the launch/runner path changes.

---

## 9. Remaining blocker chain (supersedes B60 §9 steps E–M)

1. Review, commit and push this B61 decision record.
2. Unified governed runtime implementation of M2 + M3 + M5 + M11 + M4 + M12 + M13.
3. Update the checkpoint source/provenance fields to the measured readiness values where appropriate.
4. Run all CPU/config/launch/teacher-runner smokes and the new M4/M13 tests.
5. Freeze the runtime commit and hashes.
6. G20-style CUDA re-canary if the runtime launch/runner path changes.
7. G2: batch-16 VRAM with the real teacher configuration.
8. GPU selection.
9. Official TRAIN/VAL-only preflight.
10. Explicit GO for teacher fine-tuning.

**OFFICIAL TEACHER = NO-GO.**

---

## 10. Governance documents changed in this phase

- `docs/IMPLEMENTATION_CONTRACT.md` (B1): selection row, NMF and CE-normalisation rows, runtime
  non-conformance note, provenance note, §(g) register.
- `docs/EVALUATION_CONTRACT.md`: §7.2 pointer and new §7.3 (M4-V evaluation rule, M12 selection).
- `docs/open_questions.md`: M4 and M12 LOCKED, new M13 row, register note, #9 and the G20 pointer.
- `docs/conflicts.md`: entry #15; #14's "still open" line updated.
- `docs/teacher_prep_runbook.md`: §1, §2, §3, §6, §8, §9, §11 and the open-items list.
- `docs/teacher_init_source.md`: readiness values recorded (URL, SHA-256, date, init-test PASS).
- This report.

Not modified: runtime code and configs, smokes, `PREREGISTRATION.md`, historical reports (B59, B60 and
earlier), chapter PDFs.
