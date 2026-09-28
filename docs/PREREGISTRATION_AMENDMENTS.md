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
| AM-17 | — (extends AM-1, AM-8, AM-13, AM-14 and AM-16) | Teacher record, effect-size rules, CWD-only arm, strata, TEST schedule and a descriptive SegNeXt-L arm (DRAFT) |
| AM-17b | — (extends AM-17 items 7, 9 and 1(f)) | Channel-wise decomposition arms and teacher–student gap robustness (DRAFT) |
| AM-8a | — (extends AM-8) | Repeat rule for failed runs |
| AM-4a | — (extends AM-4 and AM-1) | QAT recipe pins, converted-model selection and the U4 clipping pilot (DRAFT) |

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

[Extended by AM-17 item 4.]

[Extended by AM-4a items 3 and 4: E5 seed 42 runs twice for the U4 clipping pilot, and the winner is the run of record; the QAT loader follows the run seed.]

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

[Extended by DL-39: E6-KD uses E3's one-epoch ramp per term and retains the supervised CE + Dice loss.]

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

[Extended by AM-4a.]

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

[Extended by AM-17 item 4 and AM-8a.]

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

[Extended by AM-17 item 1.]

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

[Extended by AM-17 items 2 and 10(d).]

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

[Schedule extended by AM-17 item 9 and AM-17b item 3: overall cut order, launch order, λ shortfall rule and freeze dates.]

## AM-17 — Teacher record, effect-size rules, CWD-only arm, strata, TEST schedule and a descriptive SegNeXt-L arm (extends AM-1, AM-8, AM-13, AM-14 and AM-16)

Dated 2026-09-28. Status: DRAFT (group-recorded; adviser approval pending). State at amendment: E1
seed 42 exists (best VAL all-class mIoU 0.36314016580581665); the SegNeXt-B teacher run exists and
passed R3 (item 1). No E1 seed-43/44, E2–E7, KD, SegNeXt-L or TEST result exists; TEST has not been
evaluated. No item adds a test to the Holm family or changes an existing test, threshold, family
member or the teacher of record. Provenance: the 1.0 pp smallest effect of interest (item 2) and the
CWD-only arm (item 7) were proposed in the 26 Sep 2026 design review (rows 7 and 14) before the
teacher's score existed; items 1(f) and 11 were proposed after it and are descriptive only.

1. Teacher record (AM-13 applied; no rule changed).
   (a) Teacher of record: SegNeXt-B (MSCAN-B), iteration 24000 of run teacher_official_20260925T122252Z
       (sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e), the best-VAL
       checkpoint of the 40,000-iteration recipe. R3: VAL all-class mIoU 0.38576993346214294 >
       0.36314016580581665. PASS. The threshold and comparator were recorded in DL-08/AM-13 at commit
       8084671 (2026-09-23T05:22:41Z), before the run started (2026-09-25T12:22:52Z). Checkpoint selection by
       best VAL among the interval evaluations was fixed by M12 (reports/b61_teacher_nmf_checkpoint_selection_lock.md §6, commit 5ecdd84, 2026-09-22T10:00:09Z; its best-VAL intent was locked as M5 in reports/b60_teacher_methodology_lock.md §2.3, commit 161e735, 2026-09-22T00:50:37Z)
       before the run. E1's own checkpoint was selected by the same best-VAL rule (20
       evaluations), so the R3 comparison is symmetric.
   (b) Protocol identity: the R3 score and E1's comparator share the VAL manifest (846 images,
       35f6788e…), preprocessing (transforms.py blob b68c1d25…), class map (d1418242…), ignore index
       255 and metric implementation (9898d6dc…); F1 of the CP-007 fact-check, SAME on every field.
       Disclosed differences: execution context (teacher CPU, batch 1, evaluator at 3c43f89, NMF
       stream seeded 42; E1 A40, batch 16, train_e1 at f77d05d7; E1 re-scored on CPU differs by
       6.49e-5, immaterial against the 2.26 pp margin; the E1 GPU pass did not hash its manifest, which
       DL-17's B66 re-score closes) and the union-present outcome for class 69 (absent from VAL ground
       truth; predicted by E1, IoU 0 and eligible; not predicted by the teacher, not eligible).
       Excluding class 69 from both would put E1 at about 0.3663 and the margin at about 1.95 pp. The
       union-present rule is the MMSegmentation nanmean convention of the upstream benchmark and is
       unchanged. Descriptive addition (recorded after the teacher score; changes no decision): for
       every model on VAL and TEST, dataset-level mIoU is also reported restricted to classes with
       ground truth in that split ("GT-present"). On TEST day every model is scored under one
       evaluator commit and one execution context (item 9). Because the teacher head's NMF bases are
       randomly initialised (rand_init = True; seeded stream M4-V), a GPU re-score of the teacher is a
       tolerance check only, never a gate; if one is logged during B66 it is recorded next to E1's
       6.49e-5. [Superseded in part by AM-17b item 2(c): the teacher is not re-scored on a B66 pod.]
   (c) The teacher of record is final for every inferential run and for E4–E7: it is not retuned,
       replaced or re-selected under any E2–E7, SegNeXt-L or TEST outcome. A conditional replacement
       by SegNeXt-L was considered on 2026-09-27 and rejected as post-hoc (decision log entry
       DL-30). The descriptive SegNeXt-L arm of item 11 never becomes the teacher of record.
       ConvNeXt-L and teacher-assistant chains remain future work (Chapter 5).
   (d) The teacher VAL curve (4k–40k) is reported descriptively: VAL mIoU peaks at 24k (71.5 epochs
       over 5,367 TRAIN images) and does not improve through 40k.
   (e) Descriptive diagnostic: alongside AM-16 item 4's VAL score for E1, the teacher of record is
       scored on VAL under the upstream protocol (AM-13). The pair reports the protocol effect
       (upstream minus 512-canvas) on VAL. The 42.05% comparison stays at TEST (AM-13).
   (f) Descriptive diagnostic from existing artifacts (no run, no decision): per-class VAL IoU of the
       teacher of record minus E1 seed 42, from the R3 summary and the DL-17 B66 re-score artifact,
       tabulated by the item 8 strata, so the size of the teacher's edge on rare classes is visible
       before any KD result. It is reported with the results and is not used to select anything.

2. Smallest effect of interest (defines Chapter 3 p. 144's "sizable"). For every superiority
   contrast in the Holm family (E1 vs E2, E2 vs E3, E4 vs E5, E7 vs E6, E4 vs E7, E5 vs E6, E1 vs E6
   clean, E1 vs E6 mIoU-C), an effect is called sizable only if its Holm-adjusted test rejects and
   the seed-42 difference in dataset-level TEST all-class mIoU (dataset-level mIoU-C for the
   robustness contrast) is at least 1.0 pp in the hypothesised direction. A rejection with a smaller
   difference is reported as "detected, below the smallest effect of interest". Per-image mean ΔmIoU
   and the Hodges–Lehmann shift are reported alongside (Chapter 3 p. 144) and carry no second
   threshold. 1.0 pp mirrors the E6-KD trigger (AM-3). The 2.0 pp non-inferiority margin for E3 vs
   E6 is unchanged.

3. Minimum detectable effect (descriptive; recorded before any KD result). After B66 and before the
   first KD run, in the same decision-log entry as AM-16 item 2's α tie band:
   (a) per-image VAL disease-only mIoU (class indices 1–115, GT-present rule, AM-5 eligibility) for
       E1 seeds 42, 43 and 44, all from scripts/evaluate_model.py at one pinned commit on the same
       device (seed 42 from the DL-17 B66 re-score artifact); differences for the pairs 43−42, 44−42
       and 44−43;
   (b) planning n = 1,561 × the VAL eligible fraction (TEST masks are not read);
   (c) primary MDE_W by shifted-null simulation: for each pair, a constant shift δ is added to the
       observed differences; B = 2,000 resamples of size n (RNG seed 42) are tested with the
       pre-registered call scipy.stats.wilcoxon(zero_method='pratt', alternative='greater',
       correction=True, method='approx') at α = 0.00625; MDE_W is the smallest δ on a 0.001 grid with
       power ≥ 0.80 at δ and at the next two grid points (Monte Carlo noise guard;
       docs/lane_specs/part1.md lane 6); the largest MDE_W over the three pairs is reported;
   (d) analytic cross-check: SD_Δ = the largest of the three standard deviations;
       dz_MDE = (2.4977 + 0.8416)/√n (one-tailed α/8 = 0.00625, power 0.80, paired normal
       approximation); MDE_t = dz_MDE × SD_Δ, with its Wilcoxon-efficiency range MDE_t × [1.023,
       1.076] (asymptotic relative efficiency 0.955 and 0.864);
   (e) label: planning proxy — seed-pair differences approximate noise, not the spread of
       between-recipe differences, so the true MDE may be larger. Reported with the results; it
       changes no test or decision.
   (f) Power caveat (interpretation only; no decision). For interpretation, the per-image planning
       threshold is τ_P = 1.0 pp per-image mean shift in disease-only mIoU; it is a convention on the
       test's own scale and is not a conversion of item 2's dataset-level SESOI (no such conversion
       exists). If MDE_W > τ_P, every non-rejection in the Holm family is reported as "inconclusive at
       the smallest effect of interest" rather than as absence of an effect, and the MDE is stated
       next to each such result. This caveat is fixed by the item 3 entry before any KD result.

4. Seeds (restates AM-8). Seed 42 is the only inferential seed. Seeds 43 and 44 are reported per seed
   as descriptive tables of the eight contrasts, pairing same-seed runs; E4 and E7 are recomputed per
   seed, E5 and E6 fine-tuned per seed (Chapter 3 p. 122). Images are never pooled across seeds into
   one test.

5. Readings of the AM-16 item 3 controls (fixed before those runs exist; descriptive; dataset-level
   all-class mIoU on TEST as AM-16 specifies, and on VAL where available; none changes which
   80,000-iteration checkpoint enters a test, a sweep selection or a threshold). With
   δ1 = E2@80k − E1@160k, δ2 = E3@80k − E1@160k, δ3 = E3@80k − E2@160k: δ ≥ 1.0 pp → the
   distillation gain survives the longer-training control; 0 < δ < 1.0 pp → it survives but is
   below the smallest effect of interest; δ ≤ 0 → the gain is reported as not separable from
   additional training and H1a (δ1), the full pipeline (δ2) or CWD's increment (δ3, Chapter 3 p. 98)
   is caveated accordingly. Each arm's 160k vs 80k difference is reported as training-budget
   sensitivity, with GPU-hours as AM-16 requires.

6. Selection asymmetry. E3 seed 42 is selected from three α candidates after λ selection, so E3
   receives three VAL selections that E2 does not. Chapter 4 discloses this. VAL scores of every λ
   and α candidate are tabulated (Chapter 3 p. 105). At the single TEST evaluation, two default
   candidates are also scored, descriptively: E2 with λ = 1 (Chapter 3's unit weight) in place of
   the selected E2 for E1 vs E2, and E3 with α = 50 (the pre-registered default) in place of the
   selected E3 for E2 vs E3; where a default is the selected run the two coincide. The selected runs
   remain the runs of record whatever these comparisons show. No other candidate is evaluated on
   TEST.

7. CWD-only arm (descriptive; identifies the pixel-wise Logit-KD term). One run, seed 42, 80,000
   iterations: E1's recipe plus both channel-wise terms (feature map at 32×32 through the bias-free
   1×1 160→320 projection against MSCAN-B Stage 3, T²/C with C = 320; logit map at 64×64, C = 116)
   at E3 seed 42's selected α, β = 3, T = 4, with the same distillation-weight ramp, and no
   pixel-wise Logit-KD term; same teacher of record, projection, data, schedule, hardware rule and
   best-VAL checkpoint selection as E3. α is inherited from E3 (selected with Logit-KD present), not
   re-selected; this is disclosed. It completes the seed-42 2×2 grid with E1, E2 and E3. Readings
   (dataset-level all-class mIoU, VAL and TEST): A − E1 = channel-wise terms alone; E3 − A =
   pixel-wise Logit-KD given the channel-wise terms; (E3 − A) − (E2 − E1) = interaction. Reported on
   VAL and TEST clean mIoU (all-class and disease-only). No test; not in the Holm family; not a
   parent of E4–E7; not in the robustness analyses. Launches after the α sweep.

8. Rare-class strata (descriptive; TRAIN only). Before the first KD run, the 115 disease classes are
   ranked by pixel share in the original-resolution TRAIN annotation masks (background 0 excluded)
   and assigned to terciles by rank (38/38/39 classes); classes present in fewer than 20 TRAIN images
   are flagged. The assignment is committed as a file with its sha256. After TEST, per-class IoU
   changes are tabulated by stratum for E1→E2, E2→E3, E1→E3, E1→E4, E1→E5, E1→E6, E3→E6 and E3→E7,
   with the classes whose IoU falls to 0 after INT8 conversion; classes with no TEST ground truth are
   reported as not evaluable. No test.

9. Schedule and TEST freeze (extends AM-16's cut rule; AM-16's own order, its α = 50 fallback and its
   never-cut items 4–6 stand unchanged). TEST manifest freeze 2026-12-18; single TEST evaluation no
   later than 2026-12-21 (Asia/Manila); no run is added to TEST after the freeze. A run not complete
   and hashed at the freeze is cut in this order, each cut recorded in the decision log before TEST:
   the SegNeXt-L arm (item 11); the CWD-only arm; then AM-16's order (longer-schedule E3, ARM latency,
   longer-schedule E1, longer-schedule E2, then the α sweep and E2 seeds 43/44); then E5 and E6 seeds
   43/44; then E3 seeds 43/44. Launch order: the α = 50 candidate is launched first in the α sweep and
   the λ = 1 candidate first in the λ sweep, so the AM-16 fallback (a cut sweep means α = 50) and the
   λ rule are always executable; if fewer than five λ candidates finish by the freeze, λ is selected
   among the finished candidates by the AM-2 rule and the shortfall is disclosed. Never cut: every
   seed-42 run the Holm family and the non-inferiority check need (E1–E7; E6-KD if triggered); E4 and
   E7 for all seeds (calibration passes only); AM-16 items 4–6; the item 3 MDE entry and the item 8
   strata file. E6 seed 42 must have its converted-model VAL score by 2026-12-14 so that E6-KD, if
   triggered (AM-3), finishes before the freeze. If a never-cut run cannot finish by the freeze, the
   TEST date moves and, if it moves past 2026-12-28, the MVA 2027 submission is dropped; TEST is
   evaluated once and is never split or repeated; the analysis does not change. Before the first KD
   run, the full metrics → Holm → BCa pipeline is dress-rehearsed on the E1 seed-42/43/44 VAL outputs.
   The TEST-day manifest (checkpoint hashes, evaluator commit, corrupted-TEST cache checksums,
   calibration index) is drafted by 2026-11-30 and frozen at the freeze date.

10. Record corrections (no rule change).
    (a) Counted splits 5,367 / 846 / 1,561 (TRAIN and VAL counted in the CP-007 fact-check; TEST from
        reports/dataset_report.md:30-31, not re-counted, per EVALUATION_CONTRACT §7.2); Chapter 3
        pp. 126, 133 and 146 approximations superseded.
    (b) 116 output classes (background 0 plus 115 diseases; Chapter 3 p. 103's "115 output channels"
        superseded); student parameters 2,933,688 (backbone and LR-ASPP head with the 116-class
        classifiers, training-only projection excluded; Chapter 3 Table 3.3 p. 111's "5M-param"
        superseded); image padding with the 8-bit ImageNet mean (124, 116, 104), mask padding 255.
    (c) CWD feature term at 32×32 (student C5, 160→320 projection, against MSCAN-B Stage 3, 320
        channels), CWD logit-map and Logit-KD terms at the native 64×64 (stride-8) logit grid, validity
        mask downsampled by nearest neighbour to each grid.
    (d) Per-image mIoU-C for the E1 vs E6 robustness test (STATISTICAL_ANALYSIS_CONTRACT :106-114;
        src/stats/robustness.py): per-image disease-only mIoU in each of the 15 cells (five
        corruptions × severities 1–3), averaged over severities within each corruption and then with
        equal weight over the five corruptions, which equals the flat 15-cell mean because eligibility
        is a per-image property; images with no disease ground truth are excluded under AM-5 (lane
        L-AM5); n equals the clean-test n.
    (e) Chapter 3 p. 91's summary sentence superseded by p. 139 and AM-14; 42.05% is the single-run
        SegNeXt-B result on the 7,774-image release (Wei et al., 2026); the 2024 preprint's 53.89%
        refers to the older release and is not a comparator.
    (f) Upstream recipe (tqwei05/PlantSeg at 1a3dd4d, 2025-03-25): the repository ships MSCAN-T and
        MSCAN-L PlantSeg configurations only (no MSCAN-B); both use AdamW 6e-5, betas (0.9, 0.999),
        weight decay 0.01, head lr_mult 10, batch 16, LinearLR warm-up 1,500 iterations then PolyLR
        power 1.0 with end = 160,000 inside a 40,000-iteration loop, checkpoints every 10,000 (no
        best-checkpoint selection, per B60 §1; base files not re-fetched), backbone-only initialisation from the ImageNet MSCAN pretrain
        (mscan_l_20230227-cef260d4.pth for L), and the MMSeg pipeline (RandomResize (2048, 512) ×
        [0.5, 2.0], RandomCrop 512 with cat_max_ratio 0.75, RandomFlip, PhotoMetricDistortion); the L
        configuration sets num_classes = 116, the T configuration 115. The MSCAN-B configuration of
        record (configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py) takes its optimizer and schedule settings from these configurations and its base from the MSCAN-B ADE20K 160k configuration: initialisation ADE20K full-model checkpoint
        segnext_mscan-b_…_20230209_172053-b6f6c70c.pth, scheduler end
        40,000, training pipeline repository. Chapter 3 p. 102's "the
        repository configuration that produced the published 42.05%" is read as "the repository's
        PlantSeg schedule with the MSCAN-B architecture"; the differences above are disclosed in
        Chapter 4 and qualify the descriptive 42.05% comparison (AM-13).
    (g) Chapter 3 Table 3.3 p. 111 is corrected: MSCAN-L's Stage-3 width equals MSCAN-B's (320
        channels; Guo et al., 2022, Table 2; upstream configs); SegNeXt-L differs in depth
        ([3, 5, 27, 3] vs [3, 3, 12, 3]) and decoder width (1,024 vs 512), so the channel-width reason
        for rejecting it applies only to ConvNeXt-L (768 channels, different family).

11. SegNeXt-L descriptive arm (recorded after the teacher of record's R3 score and before any
    SegNeXt-L, KD or TEST result; motivated by the measured 2.26 pp VAL gap; descriptive only).
    (a) Fine-tune: SegNeXt-L (MSCAN-L), configuration identical to the MSCAN-B configuration of record
        in every field (initialisation policy, scheduler, pipeline, 116 classes, seed 42, 40,000
        iterations, batch 16, evaluator, readiness rules R1–R4 and checkpoint selection as in item
        1(a)) except the architecture fields of the upstream L configuration: depths [3, 5, 27, 3],
        embed_dims [64, 128, 320, 512], drop_path_rate 0.3, LightHamHead channels = ham_channels =
        1,024, and the matching L initialisation checkpoint (segnext_mscan-l_1x16_512x512_adamw_160k_ade20k_20230209_172055-19b14b63.pth (ADE20K full model, matching the B configuration's full-model initialisation), sha256 [sha256 recorded before L launch]). Any other
        difference is recorded before launch. Hardware: A40 at batch 16 after a memory preflight (the
        ADE20K L fine-tune is recorded at 43.3 GB); if the preflight fails at batch 16, the arm is not
        run at a smaller batch — an 80 GB card may be used for this fine-tune only, recorded; DL-17
        (student runs on A40) is unaffected. Launch: after this amendment is committed; may run in
        parallel with B66; never on the critical path.
    (b) Go/no-go for the KD arm (VAL only, descriptive): E2-L, E3-L and E7-L run only if L's best VAL
        all-class mIoU exceeds 0.38576993346214294 (the teacher of record). Otherwise L is reported as
        a teacher-strength note and no L-teacher KD is run.
    (c) Arm: E2-L and E3-L at seed 42, 80,000 iterations, E1's recipe, λ and α inherited from the
        seed-42 sweeps of record (tuned for the B teacher; disclosed, may understate L); the same
        160→320 projection (MSCAN-L Stage 3 is 320 channels), 64×64 logit grid, ramp, hardware rule
        and best-VAL checkpoint selection as E2/E3. E7-L = PTQ of E3-L with the AM-10 calibration
        list (calibration pass only). No E4/E5/E6-L, no seeds 43/44, no robustness scoring. Before
        E2-L launches, a smoke step (L forward in eval mode plus a student KD step at batch 16) must
        fit on the A40; otherwise the KD arm is not run and this is recorded.
    (d) Reporting (descriptive; no test; not in the Holm family; not a parent of any inferential
        run): VAL and TEST clean dataset-level mIoU (all-class, GT-present, disease-only) and
        per-image mean ΔmIoU with BCa 95% CIs for E2-L − E2, E3-L − E3 and E7-L − E7 (teacher-strength
        effect at fixed λ, α, and whether it survives PTQ); L is also scored under the upstream
        protocol at TEST (AM-13 procedure) against the published 44.52%, with the initialisation and
        scheduler differences of item 10(f) disclosed.
    (e) No promotion: whatever the L arm shows on VAL or TEST, the teacher of record (item 1) and
        every E2–E7 of record are unchanged, and E6 and E7 of record are built only from E3. The arm
        is cut first (item 9). A conditional replacement of the teacher of record by SegNeXt-L
        (promotion if L ≥ B + 1.0 pp on VAL) was considered and rejected as post-hoc (DL-30).
    (f) Pre-registered reading of the L arm against a null (descriptive). If H1a or H1b is not rejected
        and, at the same TEST evaluation, E3-L − E3 (or E2-L − E2) is at least 1.0 pp in dataset-level
        all-class mIoU, the discussion attributes the null to teacher strength as one candidate
        explanation, alongside the item 3(f) power caveat; if the L arm shows no such gain, teacher
        strength is reported as not supported. Neither reading changes any test, threshold or run of
        record.

Code: L-AM17-MDE (item 3, simulation + analytic), L-AM17-STRATA (item 8), L-AM17-CWDONLY (item 7),
L-AM17-DEFAULTS (item 6 TEST inference of the two default candidates; part of the TEST manifest),
L-AM17-GTPRESENT (item 1(b) metric variant; metrics lane), L-AM17-PERCLASS (item 1(f), artifacts
only), L-AM17-L (item 11: teacher config, preflight, E2-L/E3-L/E7-L launchers); item 1(e) rides on
L-AM13.

## AM-17b — Channel-wise decomposition arms and teacher–student gap robustness (extends AM-17 items 7, 9 and 1(f))

Dated 2026-09-28. Status: DRAFT (group-recorded; adviser approval pending). State at amendment: as
AM-17 — E1 seed 42 and the SegNeXt-B teacher of record exist; no E1 seed-43/44, E2–E7, KD, SegNeXt-L
or TEST result exists. Both items are descriptive: no test is added to the Holm family and no test,
threshold, family member, teacher of record or run of record changes.

1. Channel-wise decomposition arms F and G (descriptive; seed 42; teacher of record).
   (a) F = feature-map channel-wise term only: E1's recipe plus the feature-map CWD term (student C5
       through the bias-free 1×1 160→320 projection against MSCAN-B Stage 3 at 32×32, T = 4, T²/C
       with C = 320) at E3 seed 42's selected α; no logit-map channel-wise term; no pixel-wise
       Logit-KD. α is inherited (selected with Logit-KD and the logit-map term present); disclosed.
   (b) G = logit-map channel-wise term only: E1's recipe plus the logit-map CWD term (116-channel
       logits at the native 64×64 grid, T = 4, T²/C with C = 116, β = 3); no feature-map term, hence
       no projection module; no pixel-wise Logit-KD. G depends on no sweep.
   (c) Common spec: 80,000 iterations, seed 42, the same data, schedule, distillation-weight ramp,
       validity-mask handling, hardware rule and best-VAL checkpoint selection as E3 and arm A (AM-17
       item 7). A term that is off is not instantiated, not zero-weighted. F and G correspond to the
       feature-only and logits-only rows of Shu et al. (2021) Table 5.
   (d) Grid: with E1 (neither term) and A (both), F and G complete the 2×2 {feature-map term,
       logit-map term} within the no-Logit-KD stratum. The remaining 2³ cells (Logit-KD plus one
       channel-wise term) are not run.
   (e) Readings (dataset-level all-class mIoU on VAL and TEST; disease-only and GT-present alongside;
       per-image mean ΔmIoU with BCa 95% CIs): F − E1 and G − E1 (each term alone), A − E1 (both),
       A − F − G + E1 (interaction), E3 − A (pixel-wise Logit-KD given both terms, as AM-17 item 7).
       Each reading is reported next to the seed-noise scale of AM-16 item 2: s for a single run,
       √2·s for a two-run contrast, 2s for the interaction; a reading inside that scale is reported
       as not distinguishable from seed noise. No test; not in the Holm family; not parents of
       E4–E7; not in the robustness or efficiency analyses.
   (f) Launch: G after the KD code lane lands (no dependency on λ or α); F and A after the α winner
       exists. Cost ≈ 29–33 h per arm (INFERRED from the measured E1 and teacher rates).

2. Teacher–student gap robustness on VAL (descriptive; selects nothing; reported before any KD
   result; existing checkpoints only).
   (a) Image-level bootstrap (B = 10,000, RNG seed 42, BCa; percentile fallback where the
       acceleration is degenerate, as Chapter 3 p. 122) 95% CI of the dataset-level VAL mIoU gap,
       teacher of record minus E1 seed 42, under the union-present rule (all-class, the R3 rule) and
       the GT-present rule (AM-17 item 1(b)); class eligibility is recomputed inside every replicate
       under each rule; the statistic is computed from per-image per-class confusion contributions
       resampled over the 846 VAL images. Point estimates of record: 2.26 pp (union-present) and
       about 1.95 pp (GT-present; INFERRED arithmetic until computed).
   (b) Per-image VAL disease-only mIoU differences (teacher − E1) on the AM-5-eligible images:
       median, IQR, mean, SD, share of images where the teacher is better, share of exact ties, and
       the Hodges–Lehmann shift. The SD is reported next to AM-17 item 3's SD_Δ as a between-model
       spread reference; it is not used in the MDE or in any decision.
   (c) Inputs: both models scored on one device with one metrics implementation. Pairing of record:
       the R3 teacher artifact (CPU, evaluator 3c43f89, manifest 35f6788e…) with the B66-prep Q12 CPU
       re-score of E1 seed 42 (same manifest, metrics blob cbd5fa86…); the Q12 artifact lacks per-image
       contributions (CP-007a finding G3), so both models are re-scored once on CPU in the pinned images (teacher and student)
       after B66 (lane L-AM17B-GAP) and the two artifacts are recorded; the teacher is never re-scored on a
       B66 pod. The pairing used is stated with the result.
   (d) Uses: interpretation of the KD results and of AM-17 item 1(f); the CI does not gate, select
       or replace anything.

3. Placement in AM-17 item 9. Cut order for runs not complete and hashed at the freeze, first to
   last: the SegNeXt-L arm (AM-17 item 11); G; F; A (AM-17 item 7); then AM-16's order; then E5 and
   E6 seeds 43/44; then E3 seeds 43/44. Item 2 of this amendment has no run to cut. Never-cut items
   are unchanged.

4. Arm freeze. AM-17 and AM-17b close the set of exploratory arms. Any later arm requires its own
   amendment before its launch and is cut before every arm listed here.

Code: L-AM17B-FG (per-term instantiation switches in the KD lane; F/G launchers), L-AM17B-GAP
(paired image-level BCa on VAL from per-image confusion contributions; per-image difference summary;
reuses the paired dataset-level statistic of the non-inferiority lane where it exists).

## AM-8a — Repeat rule for failed runs (extends AM-8)

Dated 2026-09-28. A run is repeated with the same seed only after a documented infrastructure fault
(crash, host or pod failure, corrupted artifact, wrong configuration). The fault report is written and
sha256-listed in the run's durable evidence folder before the repeat launches, and committed at the next
write session. The failed run's logs and partial checkpoints are kept. No selection between a failed run
and its repeat is ever made: the repeat replaces the failed run. A run that finishes without a documented
fault is never repeated. The same rule governs a repeat of the DL-17 evaluator pair or its comparator.
Non-finite losses in E2/E3 follow AM-7/DL-04, not this rule.

## AM-4a — QAT recipe pins, converted-model selection and the U4 clipping pilot (extends AM-4 and AM-1)

Dated 2026-09-28. Status: DRAFT (group-recorded; adviser approval pending). State at amendment: no E4–E7 run or result exists. Source: the 28 Sep 2026 lane-spec audit (docs/lane_specs/part2.md, lane 6). No test is added to the Holm family.

1. Recipe pins (the values of record; AM-4 already states the epochs, both freeze points and batch 16, and this item adds the rest): SGD with momentum 0.9, learning rate 3e-4, cosine schedule with T_max = 15 × steps per epoch, weight decay 1e-4, batch 16, no EMA; 15 fixed epochs with no early stopping; observers on from the first step (moving average); BN statistics frozen from the first step of epoch 11 and observers from the first step of epoch 13 (epoch boundaries replace step fractions); the quantization configuration of record in configs/quant.py, with its weight and activation observers recorded in run_meta.
2. Checkpoint selection: each epoch's checkpoint is converted (QNNPACK) and scored on VAL on CPU with scripts/evaluate_model.py (canvas protocol); the highest dataset-level VAL all-class mIoU wins, and ties go to the earlier epoch. Fake-quant VAL is recorded and never selects.
3. U4 clipping pilot (a VAL selection, disclosed): E5 seed 42 is run twice in full, with global-norm clipping at 1.0 and at 5.0. The winner under item 2's score is E5 seed 42 of record; a tie within 0.1 pp goes to 5.0. The losing run is retained and reported. The winning value is used for E5 seeds 43/44, E6 (all seeds) and E6-KD. Chapter 4 discloses that QAT receives 2 × 15 VAL evaluations (clip value, epoch) while PTQ receives one calibration configuration (Chapter 3 p. 127), the E4-vs-E5 analogue of AM-17 item 6.
4. Seeding (AM-1): the QAT data loader is built with the run seed; torch, numpy and Python are seeded from --seed; determinism settings as in train_e1.
5. Sources: E5 starts from E1's best.json checkpoint of the same seed; E6 from E3's, which carries no projection keys (asserted on load).

Code: L-AM4, L-AM1q.

## Status of PREREGISTRATION §10 items after these amendments

| Item | Status |
|---|---|
| U1 | mechanism completed by AM-2; the value is still selected by the sweep |
| U2 | AM-3 |
| U3 | withdrawn by AM-7 |
| U4 | unchanged (AM-4 fixes only its future scoring) [Superseded by AM-4a item 3: the pilot is two full E5 seed-42 runs, clipped at 1.0 and 5.0, scored on converted-model VAL mIoU.] |
| U5 | AM-1 |
| U6 | unchanged |
| U7 | unchanged (D22 deferred) |
| U8 | stale; hardware is logged at run time |
| U9 | superseded by M4 (B61) |
