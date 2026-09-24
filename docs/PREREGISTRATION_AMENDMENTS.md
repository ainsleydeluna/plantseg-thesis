# Pre-registration amendments (dated)

`docs/PREREGISTRATION.md` (added `fcc8acc`, anchor `569cfbb`, 2026-09-05) stays **frozen**. Its evidentiary
value depends on never being edited (`docs/conflicts.md` #13). A change to a pre-registered commitment is
recorded here instead. Each amendment is dated and numbered, names the register item it resolves, and
leaves the original readable beside it.

**Ordering evidence.** All amendments below are dated **2026-09-23** (B64). At that date:
- the only completed thesis training run is **E1 seed 42**, scored on VALIDATION only
  (`reports/b52_e1_seed42_completion.md`);
- no teacher, E2–E7 or **TEST** result exists for any stage.

[AM-16 onward carry their own dates.]

| # | Resolves | Subject |
|---|---|---|
| AM-1 | M8; PREREGISTRATION §9, U5 | Seeds |
| AM-2 | M6; PREREGISTRATION §7, U1 | λ_logit sweep budget and tie rule |
| AM-3 | M7; PREREGISTRATION §6, U2 | E6-KD trigger and weights |
| AM-4 | M9 | QAT schedule and checkpoint selection |
| AM-5 | M10 | TEST images with zero disease pixels |
| AM-6 | G7 | A class absent from TEST ground truth |
| AM-7 | M1; PREREGISTRATION U3 | Gradient clipping for E1, E2 and E3 |
| AM-8 | — | Seed-stability criterion and per-seed table |
| AM-9 | — | **Withdrawn before recording** |
| AM-10 | — | PTQ calibration |
| AM-11 | — | Optional controls not run |
| AM-12 | G5 | Augmentation library naming |
| AM-13 | (record) | Teacher acceptance and the published comparison |
| AM-14 | conflicts #9; PREREGISTRATION §12 disagreement 1 | The inferential family |
| AM-15 | DL-14 (completes AM-11) | Contingencies not invoked |
| AM-16 | — (amends AM-1, AM-8, AM-10, AM-11 and AM-13; notes AM-9) | Run additions |

Code that implements an amendment is named by lane (L-AM…). Until that lane lands, the committed runtime
keeps its pre-amendment behaviour and its launch gates.

## AM-1 — Seeds (resolves M8; amends PREREGISTRATION §9 and U5)

Seeds **42 (primary), 43 and 44** for **E1 and E3**.
- E4/E7 are recomputed per seed.
- E5/E6 run once per seed and take the seed of their FP32 parent.
- **E2 runs at seed 42.** Seeds 43 and 44 are optional, budget permitting, with λ fixed from the seed-42
  sweep (ch3 f.136–137 treats E2 repetition as optional).
- The teacher is trained once.

All training randomness derives from the run seed, including data order and augmentation: the student
DataLoader generator follows `--seed` (B64 C1).

The frozen teacher's NMF inference stream stays seeded at 42 in every KD run (M4-KD), so teacher behaviour
does not depend on the student seed. QAT seeding and determinism flags arrive in lane L-AM1q.

The eight-test Holm family and the E3-vs-E6 non-inferiority check use the **seed-42 models only**. Seeds 43
and 44 enter only AM-8 and the per-seed table.

[Extended by AM-16: E2 seeds 43 and 44 also run.]

## AM-2 — λ_logit sweep (resolves M6; extends PREREGISTRATION §7 and U1)

- **Grid and budget.** {0.25, 0.5, 1, 2, 4} at seed 42, with the **full 80,000 iterations per candidate**.
- **Selection.** Select the highest dataset-level VAL all-class mIoU. A candidate's value is its
  best-checkpoint VAL all-class mIoU (strict >, earliest tie, as E1).
- **Ties.** Candidates within **0.5 percentage points** of the best are tied, and the tie goes to the
  **smallest λ**.
- **Boundary.** A boundary winner is reported as such; the grid is not extended.
- **Reuse.** The winning run **is E2 seed 42**, and λ is reused unchanged in E3.

## AM-3 — E6-KD trigger and weights (resolves M7; supersedes PREREGISTRATION §6 and U2)

- **Trigger.** E6-KD runs if the E3 → E6 drop in dataset-level **VAL** all-class mIoU is **greater than
  1.0 percentage point at seed 42**, with E6 scored on the **converted INT8** model.
- **If triggered.** E6-KD runs at seed 42 only. The Logit-KD and CWD weights are **0.5×** their E3 values,
  and T is unchanged. E6-KD is descriptive and outside the Holm family.
- **Withdrawn.** The clean-TEST trigger. The `family.json` E6-KD block stays a descriptive TEST observation
  and never launches a model.

Code: lane L-AM3.

## AM-4 — QAT (resolves M9)

**Schedule.**
- **15 epochs, fixed**, with no early stopping.
- BN statistics are frozen after epoch 10, and observers after epoch 12.
- Physical batch size **16** (`configs/quant.py` `qat_real_run.batch_size_physical`).

**Selection.**
- A checkpoint is saved every epoch.
- After training, each checkpoint is converted (QNNPACK) and scored on VAL on CPU.
- The evaluated checkpoint is the epoch with the highest converted VAL all-class mIoU; a tie goes to the
  earlier epoch.

**Supplementary, descriptive.** Each selected QAT model is also scored with fake quantization disabled (its
FP32 weights after QAT). This separates extra fine-tuning from INT8 adaptation.

**Clipping pilot.** The QAT clipping pilot (U4) stays as registered. Once L-AM4 lands, the pilot compares its
candidates on converted-model VAL mIoU (AM-4's scoring).

Code: lane L-AM4.

## AM-5 — Zero-disease TEST images (resolves M10)

- TEST images whose evaluated mask has **no disease pixels** are excluded from per-image disease-only
  analyses: the paired tests and the per-image effect sizes.
- Their count is reported. They remain in every dataset-level metric.
- The exclusion depends on ground truth only, so the eligible set is identical across models.
- **mIoU-C.** Per-image mIoU-C is the mean of the image's per-image disease-class mIoU over its 15 corrupted
  variants (five corruptions × severities 1–3). The same exclusion applies.

Code: lane L-AM5.

## AM-6 — A class absent from TEST ground truth (resolves G7)

The headline dataset-level mIoU uses the **union-present** convention: a class absent from TEST ground truth
enters only if it is predicted. A sensitivity mIoU over the classes present in TEST ground truth is reported
**descriptively**.

Code: lane L-AM6.

## AM-7 — Gradient clipping (resolves M1; withdraws PREREGISTRATION U3)

**Rule.** One clipping rule for E1, E2 and E3.
- If E1 seed 42 logged gradient norms: max_norm = the smallest value in the 1-2-5 series that is ≥ 1.5 ×
  its maximum logged norm, applied to all three stages.
- Otherwise: **no clipping** in any of the three.

A NaN or divergence in any E2/E3 run stops the stage. One clipping rule is then adopted for all three, and
the FP32 stages are rerun.

**Divergence** means either of:
- (a) any non-finite loss or gradient norm;
- (b) after the distillation ramp, the 100-iteration mean total loss exceeding 5× its running minimum.

Both are read from per-iteration telemetry. (b) becomes an in-trainer abort in lane L-AM7.

**Branch applied: "Otherwise".** E1 seed 42 logged no per-iteration gradient norm. Its telemetry `train`
rows carry `loss, ce, dice, lr, iter, iter_seconds, samples_per_sec, wall_clock`, and `run_meta.grad_clip_norm`
is null (run directory verified 2026-09-23 against its `SHA256SUMS.txt`). E1, E2 and E3 are therefore
**unclipped**. From B64 C1 onward, the E1 and KD trainers log the per-iteration total gradient norm.

**Withdrawn.** The 8,000-iteration E2/E3 clip pilot (`DISTILLATION_GRAD_CLIP_NORM`).

Code: lane L-AM7. The E2/E3 real-run launcher still requires `--grad-clip-norm` until that lane lands.

## AM-8 — Seed stability (descriptive; reported at the single TEST evaluation)

Stable iff all three hold:
- the seed-paired E1 → E3 dataset-level mIoU gain is **positive at all three seeds**;
- its mean exceeds the larger of the across-seed standard deviations of E1 and E3;
- the E3 → E6 drop is **below 2.0 percentage points at every seed**.

A per-seed table of dataset-level effects is reported for every planned comparison (descriptive).

The eight-test Holm family and the E3-vs-E6 non-inferiority check use the seed-42 models only. Seeds 43 and
44 enter only this criterion and the per-seed table.

[Extended by AM-16: the table includes E2 vs E3 at all three seeds.]

## AM-9 — Withdrawn before recording

Latency and memory stay on the approved x86 path, with the fbgemm/x86 copy (ch3 f.154).

[AM-16 adds a supplementary ARM latency measurement; the x86 path stays official.]

## AM-10 — PTQ calibration

- The configuration is fixed: the registered qconfig.
- **128 TRAIN images, seed 42, one image per mini-batch.**
- One identifier list is shared by E4 and E7.
- There is no validation-based calibration choice.

Code: lane L-AM10 (enforce a calibration batch of 1).

[AM-16 adds three descriptive calibration subsets; the official models keep this list.]

## AM-11 — Optional controls

The extended-schedule E2 and the α_CWD sensitivity sweep are **not run**. Both are recorded as future work.

[Partly superseded by AM-16: the extended-schedule E2 and the α_CWD sweep now run; the DIST fallback and 'E2 as deliverable' remain not invoked (AM-15).]

## AM-12 — Augmentation library (resolves G5)

The augmentation library is described as the hand-written NumPy/PIL implementation (`src/data/transforms.py`).
Code is unchanged.

## AM-13 — Teacher acceptance (record)

- **Acceptance.** Teacher acceptance is R3 on VAL: strictly greater than **0.36314016580581665**.
- **Published comparison.** The comparison with the published 42.05% is descriptive and made at TEST
  evaluation only.
- **Upstream-protocol score.** At the single TEST evaluation the teacher is additionally scored under the
  upstream PlantSeg protocol, descriptively. That protocol is the repository's aspect-ratio-preserving
  resize, scored against original-resolution ground truth. That score, not the 512-canvas score, is the one
  compared with 42.05%.

Code: lane L-AM13.

[Extended by AM-16: every student is also scored under the upstream protocol at TEST; E1 also on VAL before the first KD run.]

## AM-14 — The inferential family (resolves the family ambiguity, conflicts #9)

The inferential family is the eight tests listed at ch3 f.139. The primary test is a one-tailed Wilcoxon with
Pratt zeros; the paired t-test is a sensitivity check; correction is Holm step-down. E3 vs E6 is assessed only
by the paired-BCa non-inferiority check. E1 vs E3 is descriptive.

**Verified read-only, 2026-09-23 (B64). `src/stats` builds exactly this family.**
- `CANONICAL_COMPARISON_IDS` (`src/stats/tests.py:32-35`) lists these eight in ch3's order.
- The Holm input `primary_p` is the Wilcoxon p-value (`tests.py:257-258`), with the call pinned at
  `tests.py:29-30` (`zero_method="pratt"`, `alternative="greater"`). The t-test never enters the family.
- Holm is step-down with a strict boundary (`tests.py:306-351`).
- E3 vs E6 exists only as the one-sided BCa non-inferiority task, and E1 vs E3 only as descriptive tasks
  (`src/stats/bootstrap.py:40-41`, `:117-131`).

No code lane is needed.

## AM-15 — Contingencies not invoked (completes DL-14)
The DIST fallback and the 'E2 as distilled deliverable' switch are not invoked under any outcome: E6 is
always built from E3, and E3 is reported whatever its result against E2. Dated 2026-09-23; no teacher,
E2–E7 or TEST result exists.

## AM-16 — Run additions (amends AM-1, AM-8, AM-10, AM-11 and AM-13; notes AM-9)

Dated 2026-09-24. State at amendment: E1 seed 42 exists (best VAL all-class mIoU 0.3631; TEST not
evaluated); no teacher, E2–E7 or TEST result exists. No item adds a test to the Holm family. Item 2 fixes
E3's α_CWD on VAL before any E3 result, and E6 and E7 inherit it through the E3 checkpoint. Every other
item is descriptive and changes no primary analysis.

1. E2 seeds 43 and 44 are planned runs, with λ fixed from the seed-42 sweep. The per-seed effect table
   (AM-8) includes E2 vs E3 at all three seeds.
2. α_CWD sweep: after λ is fixed, E3 runs at seed 42 with α_CWD in {25, 50, 100} (β and T unchanged),
   80,000 iterations each. The highest best-checkpoint VAL all-class mIoU wins. The tie band is the
   larger of 0.5 pp and √2·s, where s is the sample standard deviation (n = 3) of E1's best-checkpoint
   VAL all-class mIoU over seeds 42, 43 and 44; √2·s is the standard deviation of a difference between
   two single runs, and the 0.5 pp floor guards against the n = 3 estimate running small. It is wider
   than DL-06's fixed 0.5 pp λ band because α has a pre-registered default (50) that a single-seed pick
   should not override on noise. s, the band and the three E1 values are recorded in the decision log
   after B66 and before the sweep launches; the sweep does not launch before that entry exists.
   Candidates within the band of the best are tied; a tie goes to 50 when 50 is tied, because 50 is the
   Chapter 3 and Shu et al. (2021) default, otherwise to the smallest α, as in the Chapter 3 λ rule. A
   winner at 25 or 100 is reported as a boundary result, and the grid is not extended. This departs from
   Chapter 3 p. 98, which fixes α_CWD = 50 and treats the sweep as a stability check: selection on VAL is
   pre-registered here before any E3 run, otherwise mirrors the λ_logit protocol (pp. 104–105) and never
   consults TEST. The winning run is E3 seed 42, and its α is used for E3 seeds 43 and 44. All three
   runs are reported as the Chapter 3 neighborhood-stability check.
3. Longer-schedule controls: E1, E2 and E3 at seed 42 with 160,000 iterations each (poly schedule over
   the 160,000-iteration horizon; VAL every 4,000 iterations; best-checkpoint selection as in the
   80,000-iteration runs; E2/E3 use the selected λ and α). Descriptive, on clean TEST mIoU, with each
   run's measured GPU-hours reported alongside: each 160,000-iteration run against its 80,000-iteration
   run; E3 at 80,000 against E2 at 160,000 (the Chapter 3 sanity check); and E2 and E3 at 80,000
   against E1 at 160,000 (a longer-trained-baseline control; the runs are not compute-matched, since a
   KD iteration also runs the teacher forward pass and E1 at 160,000 stays cheaper than E2 or E3 at
   80,000; the measured GPU-hours make the gap visible).
4. Every student is also scored under the upstream PlantSeg protocol at the single TEST evaluation
   (descriptive), as the teacher is under AM-13. E1 is also scored this way on VAL, on the existing
   seed-42 best checkpoint, before the first KD run.
5. PTQ calibration sensitivity: E4 and E7 at seed 42 are also calibrated on three further 128-image
   TRAIN subsets, drawn by the AM-10 procedure with seeds 43, 44 and 45. Descriptive: reported per
   subset, with the range over all four calibration sets. The official E4/E7 models keep the AM-10
   list.
6. Robustness: the four non-noise corruptions (motion blur, JPEG compression, brightness, fog) are also
   scored at severities 4 and 5 (descriptive), following Kamann & Rother (2020), who average non-noise
   corruptions over severities 1–5 and noise over severities 1–3. Reported per severity and as that
   average. The mIoU-C, RPD and rCD definitions (inferential and descriptive) stay on severities 1–3.
7. Supplementary latency: the INT8 models E4–E7 (the QNNPACK artifacts of record) and their FP32
   parents E1 and E3 are also timed on an AWS c6g.xlarge (c6g.2xlarge if that thread count exceeds 4)
   (Graviton2, Arm Neoverse N1, the Cortex-A76-derived server core and the closest EC2 analogue to a
   mobile big core), on-demand, with the QNNPACK engine for INT8 and the existing latency protocol at
   the x86 path's fixed thread count. The aarch64 runtime is recorded: torch 2.1.0 and torchvision
   0.16.0 aarch64 CPU wheels with their hashes, OS image, kernel, CPU model and RAM; the x86 training
   image does not run there. Before timing, the ARM INT8 outputs on
   the first 16 VAL images (sorted by file name) are compared with the accuracy outputs of record (the
   QNNPACK-configured models as evaluated on x86 with the QNNPACK engine, never the fbgemm/x86 latency
   copies), with pixel agreement computed over valid (non-255) pixels; ARM timings are reported only if
   agreement is at least 99%, and the agreement is reported either way. Descriptive: the x86
   fbgemm-copy path stays official, and ARM and x86 latencies are not compared with each other (the
   FP32 parents give the within-host reference). This is a server-class ARM measurement, not an
   on-device one: it supersedes Chapter 3 p. 154's statement that no tests run on actual ARM devices
   only to that extent, and p. 141's statement that on-device mobile latency is not claimed stands.

Cut order if the schedule slips: the longer-schedule E3, then ARM latency, then the longer-schedule
E1, then the longer-schedule E2; the α sweep and E2 seeds 43/44 are cut last; items 4–6 are never
cut. If item 2 is cut, E3 runs at α_CWD = 50 (the Chapter 3 default) and no sweep is reported. Any cut
is recorded here before the affected run.

## Status of PREREGISTRATION §10 items after these amendments

| Item | Status |
|---|---|
| U1 | mechanism completed by AM-2; the value is still selected by the sweep |
| U2 | AM-3 |
| U3 | withdrawn by AM-7 |
| U4 | unchanged (AM-4 fixes only its future scoring) |
| U5 | AM-1 |
| U6 | unchanged |
| U7 | unchanged (D22 deferred) |
| U8 | stale; hardware is logged at run time |
| U9 | superseded by M4 (B61) |
