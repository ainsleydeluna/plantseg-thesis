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
| AM-17 | — (extends AM-1, AM-8, AM-13, AM-14 and AM-16) | Teacher record, effect-size rules, CWD-only arm, strata, TEST schedule and a descriptive SegNeXt-L arm (~~DRAFT~~ approved by the adviser, DL-65) |
| AM-17b | — (extends AM-17 items 7, 9 and 1(f)) | Channel-wise decomposition arms and teacher–student gap robustness (~~DRAFT~~ reported as approved by the adviser in person, DL-83) |
| AM-8a | — (extends AM-8) | Repeat rule for failed runs |
| AM-4a | — (extends AM-4 and AM-1) | QAT recipe pins, converted-model selection and the U4 clipping pilot (~~DRAFT~~ reported as approved by the adviser in person, DL-83; amended by AM-21) |
| AM-7a | — (extends AM-7; applies to AM-2, AM-16 items 2–3, AM-17 item 9, AM-8a) | Divergence handling for sweep candidates; pre-committed clipping value (~~DRAFT~~ reported as approved by the adviser in person, DL-83) |
| AM-18 | — (extends AM-1, AM-7a item 3, AM-8a, AM-13, AM-17 items 1, 9 and 11, AM-17b items 2–4; for KD targets only, conditionally supersedes contract M4's non-adoption of multi-draw averaging) | Teacher diagnostics R0, descriptive arm teachers (R1, R2; B-avg not executable) and KD arms (E3-X, W, conditional K8), go rule, pre-set defect rules (D1, D4), dataset duplicates (DRAFT) |
| AM-19 | — (amends AM-7, AM-7a items 1–4 and 6, AM-8a and AM-17 item 9; extends AM-1, AM-2, AM-7a item 5, AM-8, AM-16 items 1–3 and its cut rule, AM-17 items 4 and 6, AM-17b item 3 and AM-18 items 1, 2(c), 3, 6 and 7) | Schedule T/R, sweep decision dates, stops, repeats and cut records, card and launch order, runs with a cut or diverged parent, seed-43/44 divergence, seed stability with a missing run, AM-7's consequence under a schedule (DRAFT) |
| AM-20 | — (amends AM-18's header request rule and item 6(a)'s availability rule; settles a reading of AM-18 item 6(b)) | The unsent approval request; AM-18 item 6(a) state "no approval"; the 2026-10-05 in-person approval, as reported (DRAFT) |
| AM-21 | — (amends AM-4a items 2–3; extends AM-19 item 3(a) for E5 and E6 runs) | Non-finite states in QAT runs (DRAFT) |
| AM-19a | — (settles readings of AM-19's Definitions and items 2 and 3) | Readings of AM-19 for the KD sweeps (DRAFT) |

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

[Extended by AM-18: the teacher of record is still trained once. Arm teachers R1 and R2 are separate descriptive fine-tunes; B-avg, an average of the record run's checkpoints, is not executable (only iteration 24,000 was retained).]

[AM-19 item 4: E4–E7 are made at a seed only when their parent is complete. Item 6(f): seeds 43 and 44 still enter no test.]

## AM-2 — λ_logit sweep (resolves M6; extends PREREGISTRATION §7 and U1)

- **Grid and budget.** {0.25, 0.5, 1, 2, 4} at seed 42, with the **full 80,000 iterations per candidate**.
- **Selection.** Select the highest dataset-level VAL all-class mIoU. A candidate's value is its
  best-checkpoint VAL all-class mIoU (strict >, earliest tie, as E1).
- **Ties.** Candidates within **0.5 percentage points** of the best are tied, and the tie goes to the
  **smallest λ**.
- **Boundary.** A boundary winner is reported as such; the grid is not extended.
- **Reuse.** The winning run **is E2 seed 42**, and λ is reused unchanged in E3.

[Extended by AM-19 item 2: a λ decision date; a non-default candidate that is neither finished nor diverged by its end is cut, and λ is selected among the finished candidates.]

## AM-3 — E6-KD trigger and weights (resolves M7; supersedes PREREGISTRATION §6 and U2)

- **Trigger.** E6-KD runs if the E3 → E6 drop in dataset-level **VAL** all-class mIoU is **greater than
  1.0 percentage point at seed 42**, with E6 scored on the **converted INT8** model.
- **If triggered.** E6-KD runs at seed 42 only. The Logit-KD and CWD weights are **0.5×** their E3 values,
  and T is unchanged. E6-KD is descriptive and outside the Holm family.
- **Withdrawn.** The clean-TEST trigger. The `family.json` E6-KD block stays a descriptive TEST observation
  and never launches a model.

Code: lane L-AM3.

[Extended by DL-39: E6-KD uses E3's one-epoch ramp per term and retains the supervised CE + Dice loss.]

[Extended by DL-51: E6-KD's projection starts from E3's trained projection.pt weights paired with the E3 checkpoint; FP32, never fake-quantized.]

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

[Extended by AM-4a.] [Amended through AM-4a by AM-21, 2026-10-07: non-finite states in QAT runs.]

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
- (a) any non-finite loss or gradient norm; [K8-1 (R8-1, DL-04): at iteration 1 a non-finite total loss or gradient
  norm is a step-1 check failure (STOP; AM-7a item 4), not (a); (a) applies from iteration 2.]
- (b) after the distillation ramp, the 100-iteration mean total loss exceeding 5× its running minimum.

Both are read from per-iteration telemetry. (b) becomes an in-trainer abort in lane L-AM7.

**Branch applied: "Otherwise".** E1 seed 42 logged no per-iteration gradient norm. Its telemetry `train`
rows carry `loss, ce, dice, lr, iter, iter_seconds, samples_per_sec, wall_clock`, and `run_meta.grad_clip_norm`
is null (run directory verified 2026-09-23 against its `SHA256SUMS.txt`). E1, E2 and E3 are therefore
**unclipped**. From B64 C1 onward, the E1 and KD trainers log the per-iteration total gradient norm.

**Withdrawn.** The 8,000-iteration E2/E3 clip pilot (`DISTILLATION_GRAD_CLIP_NORM`).

Code: lane L-AM7. The E2/E3 real-run launcher still requires `--grad-clip-norm` until that lane lands.

[Interpretation recorded 2026-09-30 (DL-04): the (b) window and its running minimum are both computed over post-ramp iterations only; (a)'s gradient norm is the pre-clip norm over all trainable parameters; scope for arms A/F/G and E6-KD as in DL-04. Code: lane L-KD-HARDEN, which folds L-AM7.]

[Extended by AM-7a: a non-default sweep candidate's divergence stops that run only; the clipping value, if ever adopted, is pre-committed (AM-7a item 5).]

[Amended by AM-19 item 5: a divergence of E2 or E3 at seed 43 or 44 follows the arm rule. Item 7: when the full consequence fires, the schedule's dates do not move and a rerun sweep keeps its decision date.]

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

[Extended by AM-19 item 6: each condition holds, fails or is not evaluable; conditions 1 and 3 fail at any existing seed that violates them; the verdict keeps two values and "not stable" carries its reason; an E3 divergence at seed 43 or 44 makes it not stable.]

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

[Extended by AM-18 items 1(d) and 6(b): the R3 comparison is also computed on duplicate-free VAL; if the teacher's margin over E1 seed 42 there is ≤ 0, no KD run launches and R3 is restated by its own amendment.]

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
   (AM-8) includes E2 vs E3 at all three seeds. [AM-19 item 6(e): missing cells of the per-seed table; an E2 divergence at seed 43 or 44 is reported with the verdict.]
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
   runs are reported as the Chapter 3 neighborhood-stability check. [Extended by AM-19 item 2: an α decision date, with selection among the finished candidates. The α = 50 fallback applies when no non-default candidate was launched by then.]
3. Longer-schedule controls: E1, E2 and E3 at seed 42 with 160,000 iterations each (poly schedule over
   the 160,000-iteration horizon; VAL every 4,000 iterations; best-checkpoint selection as in the
   80,000-iteration runs; E2/E3 use the selected λ and α). Descriptive, on clean TEST mIoU, with each
   run's measured GPU-hours reported alongside: each 160,000-iteration run against its 80,000-iteration
   run; E3 at 80,000 against E2 at 160,000 (the Chapter 3 sanity check); and E2 and E3 at 80,000
   against E1 at 160,000 (a longer-trained-baseline control; the runs are not compute-matched, since a
   KD iteration also runs the teacher forward pass and E1 at 160,000 stays cheaper than E2 or E3 at
   80,000; the measured GPU-hours make the gap visible). [DL-44: GPU-hours = the run's wall-clock from
   run_meta to run_end, validations included, from the run's own telemetry.] [AM-19 item 7(d): a control run under a superseded recipe is reported as superseded.]
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

[AM-19 item 3: cuts are made only by the listed rules and are recorded in the decision log on the day they are made. Item 2(g): the α sweep is settled at its decision date, the one exception to this order.]

## AM-17 — Teacher record, effect-size rules, CWD-only arm, strata, TEST schedule and a descriptive SegNeXt-L arm (extends AM-1, AM-8, AM-13, AM-14 and AM-16)

Dated 2026-09-28. Status: DRAFT (group-recorded; adviser approval pending). [Approved by the adviser: decision log DL-65, recorded 2026-10-05.] State at amendment: E1
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
       over 5,367 TRAIN images) and does not improve through 40k. [AM-18 item 9(d) records the ten interval VAL scores.]
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
   one test. [Extended by AM-19 item 6.]

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
   TEST. [AM-19 item 2(e): a candidate cut at a decision date is tabulated with its reason and partial scores. Under this item it is never scored on TEST.]

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
   calibration index) is drafted by 2026-11-30 and frozen at the freeze date. [Extended by AM-18 item 7(b): every AM-18 arm is cut before the arms listed here.] [Amended by AM-19: the dates of item 1 (Schedule T or R, whichever is in force), with the E6 and manifest-draft dates as planning dates; a late never-cut run moves the freeze day by day (item 1(d)); card and launch order (item 1(e)); the shortfall at the decision dates (item 2); stops and cut records (item 3); "complete and hashed" (Definitions); E4 and E7 with their parent (item 4); the manifest's TEST-derived entries (item 1(c)); rerun E1 at seeds 43/44 at the end of the cut order (item 7(b)).]

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
        mask downsampled by nearest neighbour to each grid. [Corrected 2026-09-30 (DL-48): the validity
        mask is downsampled by an all-valid min-pool (a cell is valid only if every pixel it covers is
        valid), the rule the code has implemented since B32 and the contract records; "nearest neighbour"
        was a misdescription. No rule or output changes.]
    (d) Per-image mIoU-C for the E1 vs E6 robustness test (STATISTICAL_ANALYSIS_CONTRACT :106-114;
        src/stats/robustness.py): per-image disease-only mIoU in each of the 15 cells (five
        corruptions × severities 1–3), averaged over severities within each corruption and then with
        equal weight over the five corruptions, which equals the flat 15-cell mean because eligibility
        is a per-image property; images with no disease ground truth are excluded under AM-5 (lane
        L-AM5); n equals the clean-test n.
    (e) Chapter 3 p. 91's summary sentence superseded by p. 139 and AM-14; 42.05% is the single-run
        SegNeXt-B result on the 7,774-image release (Wei et al., 2026); the 2024 preprint's 53.89%
        refers to the older release and is not a comparator. [Source recorded by AM-18 item 9(a): arXiv 2409.04038v1, "Evaluation on PlantSeg" (text: 53.89%; its Table 3: 44.52% for MSCAN-L).]
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
        (configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py) in every field —
        initialisation policy (ADE20K full-model checkpoint via load_from, classifier-only
        re-initialisation to 116 classes), M2/M3 transforms, optimizer, warm-up and horizon-corrected
        schedule, seed 42, 40,000 iterations, batch 16, M4 NMF policy, M11 isolation, M12 best-VAL
        checkpoint selection, readiness rules R1–R4 — except the architecture fields inherited from the
        upstream MSCAN-L ADE20K base configuration: depths [3, 5, 27, 3], drop_path_rate 0.3,
        LightHamHead channels = ham_channels = 1,024 (embed_dims [64, 128, 320, 512] are identical to
        B's), and the matching initialisation checkpoint
        segnext_mscan-l_1x16_512x512_adamw_160k_ade20k_20230209_172055-19b14b63.pth (ADE20K full model,
        50.99 mIoU; sha256 recorded on first download, as for B). The configuration diff against the B
        file is measured and recorded before launch; any other difference is recorded before launch.
        Hardware: the fine-tune's peak memory at batch 16 is projected at 49.6 GiB from a measurement
        that reproduces the B fine-tune's measured A40 peak within 0.26%, so it does not fit an A40; it
        runs at batch 16 on an NVIDIA A100 PCIe 80GB (Secure Cloud), and the first-iteration peak is
        recorded as the measurement. The batch size is never reduced. DL-17 (student runs on A40) is
        unaffected; teacher hardware is not an inferential invariant. Launch: only after adviser
        approval of AM-17 and the audited lane plan (DL-32), on its own pod; never on the critical
        path; cut first (item 9).
    (b) Go/no-go for the KD arm (VAL only, descriptive). L's readiness score is computed by the same
        path that produced the teacher of record's R3 value: scripts/teacher_readiness_r3.py with the L
        configuration, the thesis evaluator at the L-lane pin, CPU, batch 1, M4-V stream seeded 42, VAL
        manifest 35f6788e…. Before L's score is read, the teacher of record is re-scored by that path on
        the same host and must lie within 1e-4 of 0.38576993346214294 (the difference is recorded);
        otherwise the evaluator is diagnosed first. E2-L, E3-L and E7-L run only if L's score exceeds
        0.38576993346214294; if L's score lies within 1e-4 of that constant, the comparison against the
        same-host re-score of the teacher of record decides and the case is disclosed. L's in-training
        best VAL (the M12 selection record) is recorded, not used. This rule is recorded in the decision
        log before any L score exists. Otherwise L is reported as a teacher-strength note and no
        L-teacher KD is run.
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
        record. [AM-18 item 7 places the L arm in the run and cut orders; items 3 and 3(e) apply the target construction of record and the arm divergence rule to its KD runs; item 11 is otherwise unchanged.]

Code: L-AM17-MDE (item 3, simulation + analytic), L-AM17-STRATA (item 8), L-AM17-CWDONLY (item 7),
L-AM17-DEFAULTS (item 6 TEST inference of the two default candidates; part of the TEST manifest),
L-AM17-GTPRESENT (item 1(b) metric variant; metrics lane), L-AM17-PERCLASS (item 1(f), artifacts
only), L-AM17-L (item 11: teacher config, preflight, E2-L/E3-L/E7-L launchers); item 1(e) rides on
L-AM13.

## AM-17b — Channel-wise decomposition arms and teacher–student gap robustness (extends AM-17 items 7, 9 and 1(f))

Dated 2026-09-28. Status: DRAFT (group-recorded; adviser approval pending). [Reported as approved by the adviser in person: decision log DL-83, recorded 2026-10-07.] State at amendment: as
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
       or replace anything. [AM-18 item 6(b) uses the item 2(c) artifacts for a pre-set gate on duplicate-free VAL. The CI itself still gates nothing.]

3. Placement in AM-17 item 9. Cut order for runs not complete and hashed at the freeze, first to
   last: the SegNeXt-L arm (AM-17 item 11); G; F; A (AM-17 item 7); then AM-16's order; then E5 and
   E6 seeds 43/44; then E3 seeds 43/44. Item 2 of this amendment has no run to cut. Never-cut items
   are unchanged. [Extended by AM-18 item 7(b).] [AM-19 item 1(c): "the freeze" in this cut rule is the original freeze date of the schedule in force. Item 7(b) adds the rerun E1 runs at seeds 43/44 at the end.]

4. Arm freeze. AM-17 and AM-17b close the set of exploratory arms. Any later arm requires its own
   amendment before its launch and is cut before every arm listed here. [AM-18 registers arms under this rule (B-avg, R1, R2, W, conditional K8); they are cut first.]

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

[K8-1 (R8-1, DL-04): a non-finite total loss or gradient norm at iteration 1 is a step-1 check failure, not AM-7 (a): the run stops for investigation and this rule governs any repeat (AM-7a item 4).]

[Extended by AM-18's execution rules to the item 1 statistic scripts: a repeat also follows a script defect shown by a diff.]

[Amended by AM-19 item 3(a): a run stopped by the operator, the pod or the host is repeated under this rule, and a non-divergence abort is repeated once.]

## AM-4a — QAT recipe pins, converted-model selection and the U4 clipping pilot (extends AM-4 and AM-1)

Dated 2026-09-28. Status: DRAFT (group-recorded; adviser approval pending). [Reported as approved by the adviser in person: decision log DL-83, recorded 2026-10-07.] [In force for every QAT run, whatever its approval status: AM-21 item 4.] State at amendment: no E4–E7 run or result exists. Source: the 28 Sep 2026 lane-spec audit (docs/lane_specs/part2.md, lane 6). No test is added to the Holm family.

1. Recipe pins (the values of record; AM-4 already states the epochs, both freeze points and batch 16, and this item adds the rest): SGD with momentum 0.9, learning rate 3e-4, cosine schedule with T_max = 15 × steps per epoch, weight decay 1e-4, batch 16, no EMA; 15 fixed epochs with no early stopping; observers on from the first step (moving average); BN statistics frozen from the first step of epoch 11 and observers from the first step of epoch 13 (epoch boundaries replace step fractions); the quantization configuration of record in configs/quant.py, with its weight and activation observers recorded in run_meta.
2. Checkpoint selection: each epoch's checkpoint is converted (QNNPACK) and scored on VAL on CPU with scripts/evaluate_model.py (canvas protocol); the highest dataset-level VAL all-class mIoU wins, and ties go to the earlier epoch. Fake-quant VAL is recorded and never selects. [Amended 2026-10-07 — AM-21 item 2: an epoch whose checkpoint state is not finite is excluded.]
3. U4 clipping pilot (a VAL selection, disclosed): E5 seed 42 is run twice in full, with global-norm clipping at 1.0 and at 5.0. The winner under item 2's score is E5 seed 42 of record; a tie within 0.1 pp goes to 5.0. The losing run is retained and reported. The winning value is used for E5 seeds 43/44, E6 (all seeds) and E6-KD. Chapter 4 discloses that QAT receives 2 × 15 VAL evaluations (clip value, epoch) while PTQ receives one calibration configuration (Chapter 3 p. 127), the E4-vs-E5 analogue of AM-17 item 6. [Amended 2026-10-07 — AM-21 item 3: a pilot run that becomes non-finite (state, logged loss or pre-clip gradient norm) is rejected.]
4. Seeding (AM-1): the QAT data loader is built with the run seed; torch, numpy and Python are seeded from --seed; determinism settings as in train_e1.
5. Sources: E5 starts from E1's best.json checkpoint of the same seed; E6 from E3's, which carries no projection keys (asserted on load).

Code: L-AM4, L-AM1q.

## AM-7a — Divergence handling for sweep candidates and the pre-committed clipping value (extends AM-7; applies to AM-2, AM-16 items 2 and 3, AM-17 item 9 and AM-8a)

Dated 2026-09-30. Status: DRAFT (group-recorded; adviser approval requested). [Reported as approved by the adviser in person: decision log DL-83, recorded 2026-10-07.] In force for every
abort that occurs after this amendment is committed; the adviser's later approval or rejection does
not reclassify an abort that has already occurred (a rejection restores AM-7's full consequence for
later aborts only). State at amendment: E1 seeds 42, 43 and 44 (VAL only) and the SegNeXt-B teacher
of record exist. No E2, E3, A, F, G or other KD run has started, no KD result exists, and TEST has
not been evaluated. No test is added to the Holm family; no test, threshold, family member, grid,
tie rule, seed or teacher changes.

1. Definitions. A divergence is AM-7 (a) or (b) under the reading of record (DL-04), determined
   solely by the trainer's own abort, which writes a `run_abort` record naming the rule, the
   iteration and its diagnostics (input finiteness, teacher-output finiteness, parameter finiteness,
   the offending values). A run that finished (has a `run_end` record) is never reclassified as
   diverged, and a `run_abort` record is never reinterpreted after the fact. An abort whose
   diagnostics show non-finite input tensors or non-finite teacher outputs is a fault, not a
   divergence: the run stops, is investigated, and any repeat follows AM-8a. A run stopped by any
   other means (operator, pod, host) has no divergence status: it is an unfinished run under AM-17
   item 9, and the reason for the stop is recorded. Default candidates: λ = 1 (AM-17 items 6 and 9)
   and α = 50 (Chapter 3 p. 98; Shu et al., 2021). Non-default candidates: λ ∈ {0.25, 0.5, 2, 4}
   (AM-2) and α ∈ {25, 100} (AM-16 item 2). [AM-19 items 2(c) and 3(a): a run stopped by any other means is repeated under AM-8a; a non-default candidate with no finished run at the end of its decision date is cut there.]

2. Non-default candidate. A divergence in a non-default candidate stops that run only. The run is
   recorded as diverged; its `run_abort` record, telemetry and partial checkpoints are kept and
   hashed as durable evidence. It is excluded from its selection; the other candidates continue; no
   clipping rule is adopted and nothing is rerun. The diverged value is never relaunched — not with
   clipping, not with the same seed, not at another seed (AM-8a: a divergence is not an
   infrastructure fault) — and the grid is not extended. The selection runs as soon as every
   non-diverged candidate has finished, among the finished candidates, under the unchanged AM-2 or
   AM-16 item 2 rule (value, band, tie-break). The diverged run's directory is a required input of
   the selection: the selection refuses to run without it, verifies the abort record, and writes the
   exclusion into the selection file. Boundary: a winner at a registered grid end is flagged as
   before; a winner that is the smallest or largest finished value with a diverged neighbour is
   flagged "edge of the finished set (AM-7a)". The candidate table (AM-17 item 6) lists the
   diverged value as "diverged (AM-7a)" with the rule, the iteration, its partial VAL scores up to
   the abort and its GPU-hours; partial scores never enter a selection. [AM-19 item 2: with a cut candidate, the selection runs after the end of the decision date.]

3. Default candidates and other runs. AM-7's full consequence — the stage stops, one clipping rule
   is adopted for E1, E2 and E3, the FP32 stages are rerun — applies unchanged to a divergence in a
   default candidate (λ = 1; α = 50 at the selected λ) and in E2 or E3 at seeds 43 and 44 (the
   recipe of record at another seed). When it fires, the fate of the other candidates is
   immaterial: every E2/E3 run is rerun under the adopted rule. Arms A, F and G and the
   160,000-iteration controls (AM-16 item 3) follow DL-04's arm rule: the run stops, is recorded as
   diverged, is not rerun and triggers no clipping rule; the 160,000-iteration schedule is a
   descriptive control whose divergence is evidence about the longer schedule, not about the
   80,000-iteration recipe of record. [AM-18 item 3(e): the AM-18 KD arms and the SegNeXt-L arm's KD runs follow this arm rule.] [Amended by AM-19 item 5: E2 and E3 at seeds 43 and 44 follow the arm rule. Item 7 fixes what the full consequence reruns under a schedule; "not rerun" applies within one recipe.]

4. Edge cases. If only λ = 1 finished, λ = 1 is selected as the sole finished candidate
   (n_finished = 1) and the whole sweep is reported; if only α = 50 finished, α = 50 is selected the
   same way. Each is a completed sweep, not AM-16 item 2's cut (a cut means the runs did not happen
   and no selection file exists). A diverged default candidate follows item 3. A run that stops for
   any other reason — an iteration-1 check failure, a non-finite VAL score, a fault — is refused by
   the selection as a STOP for investigation: it is never excluded and never a shortfall, and AM-8a
   governs any repeat. [Amended by AM-19 items 2(b)–(c) and 3(a): after the end of a decision date, a non-default candidate that is neither finished nor diverged is cut, whatever stopped it; a non-divergence abort is repeated once. The default candidate keeps this STOP.]

5. Pre-committed clipping value. If AM-7's full consequence fires, no clipping value is chosen at
   that time. max_norm = the smallest value in the 1-2-5 series that is ≥ 1.5 × the maximum
   pre-clipping gradient norm logged over all iterations of E1 seeds 43 and 44 (DL-03 telemetry;
   seed 42 logged none, so AM-7's "otherwise" branch stands), computed by scripts/am7_clip_value.py
   from the archived telemetry and recorded here before the first non-default candidate launches:
   maximum logged norm 37.523643493652344 (seed 44, iteration 28110); 1.5× = 56.2854652404785160;
   max_norm = 100 (reports/derived/am7_clip_value.json, sha256
   38267e82f562c6ed640b76e5bf7c770403923fde35e32947477661af7bb4e7ed). Applied identically to the full
   trainable set of every rerun FP32 stage, the arms and the controls. [AM-19 item 7(e): the rerun selections use this value as the recipe's clipping value.]

6. Relation to AM-17 item 9. Item 9's shortfall selection (fewer than five λ candidates finished at
   the TEST freeze) is unchanged and distinct: it concerns runs not finished at the freeze and is
   executed at the freeze; this item concerns diverged runs and is executed when the non-diverged
   candidates finish. Both can apply to one sweep; each excluded candidate is disclosed under the
   item that excluded it. [Amended by AM-19 item 2(g): "is unchanged" reads "is replaced by AM-19 item 2, for λ and α", and the references to the freeze read as the sweep's decision date.]

7. Rationale. A divergence at a non-default weight under the shared unclipped recipe is evidence
   against that weight; the default candidates, launched first (AM-17 item 9), remain the alarm for
   the recipe itself. Without this item one diverged non-default candidate would force E1 (three
   seeds), E2 and E3 to be retrained with clipping, and the clipping value would be chosen after
   the divergence.

8. Disclosure. Chapter 4 lists every diverged run (value, rule, iteration, diagnostics), states
   that each affected selection ran among the finished candidates, and states the pre-committed
   clipping value and whether it was applied.

Code: lane L-KD-HARDEN — Phase A: `run_abort` diagnostics (`cause`, `input_finite`,
`teacher_finite`, `params_finite`); Phase B: src/training/sweep_select.py, scripts/select_lambda.py,
scripts/select_alpha.py, configs/sweep_rules.json, configs/distill.py, scripts/smoke_select_sweeps.py.
Local: scripts/am7_clip_value.py (reads the E1 seed-43/44 telemetry; no pod).

## AM-18 — Teacher diagnostics, descriptive teacher arms, pre-set defect rules and dataset duplicates (extends AM-1, AM-7a item 3, AM-8a, AM-13, AM-17 items 1, 9 and 11, and AM-17b items 2–4; for KD targets only, conditionally supersedes contract M4's non-adoption of multi-draw averaging)

Dated 2026-10-01. Status: DRAFT (group-decided 2026-10-01, ballot 1C 2A 3B 4A 5A 6A; adviser
approval requested).
[Erratum E-52, 2026-10-07: the approval request was not sent (AM-20, DL-82). The adviser's in-person approval of 2026-10-05 is recorded as information only (DL-83).]
- Fixed text. From its commit the text is changed only by a new amendment, which lists the item 1
  outputs known when it is written. A reading this text leaves open is settled the same way (for
  a statistic, by the first output of record). A dated bracket note only points to a later record
  or records an outcome, such as the adviser's reply.
- In force from the commit: the execution rules and items 1, 5, 6, 7, 8 and 9, which launch no
  training run. No arm of items 2 and 3 launches before the adviser approves this amendment. [AM-20 items 3 and 4, 2026-10-07: only a written reply as this header defines it opens an arm; the in-person approval of 2026-10-05 (DL-83) opens none.]
- Approval means an explicit written reply from the adviser that approves this amendment, either
  naming AM-18 or answering the request email, whose subject names AM-18; it is recorded verbatim
  with its date in the decision log before the action it gates. An acknowledgement ("noted") is not
  approval. The request is sent when this amendment is committed, names item 6(a)'s correction
  separately, and is entered in the decision log with its date; each reply is entered on the day
  it is received. A reply that approves AM-18 without excepting an item approves every item; one
  that excepts an item approves the rest. A requested change, or an exception to an item already
  in force other than declining item 6(a)'s correction, needs a new amendment. [Note 2026-10-07 — AM-20 item 1: the request was not sent (DL-82, erratum E-52).]
- A rejection cancels the arms not yet launched (R1, R2, E3-avg, W, K8), item 4's additions and,
  if it is in the decision log before item 6(a)'s state is entered, the correction (the
  registered single-draw targets then stand). After that entry the state is final. The SegNeXt-L
  arm stays as AM-17 item 11 registers it. The execution rules and items 1, 6(b), 7(a), 8 and 9
  stay in force; no statistic of item 1 is withdrawn and no existing run is reclassified.
- Naming. Arm teachers are B-avg, R1 and R2; SegNeXt-L is "L" and is always named separately.
  "R3" means the contract's readiness rule 3 (AM-13); the readiness rules are written "readiness
  rule 1" to "4". The R3 evaluation procedure is that of AM-17 item 1(b): the repository evaluator
  on CPU in the pinned teacher image, batch 1, canvas protocol, the VAL manifest of record, NMF
  stream M4-V seeded 42. A session is one sequence of processes on one host at one commit.

State at amendment (as of the commit):
- AM-7a, the 2026-09-30 correction of AM-17 item 10(c), the 2026-09-28 replacement of AM-17
  items 11(a) and 11(b) and the decision-log row with the adviser's approval of AM-17 are committed
  before or with this amendment.
- E1 seeds 42, 43 and 44 (VAL only) and the SegNeXt-B teacher of record (AM-17 item 1(a)) exist.
- Of the record run's interval checkpoints, only iteration 24,000 was retained (local copies and
  SHA256SUMS of run teacher_official_20260925T122252Z; the pod was terminated on 2026-09-26).
  Item 1(e) is therefore not executable and is reported as such; B-avg, E3-avg and E7-avg do not
  exist.
- On 1–2 October 2026, during a read-only file inventory, automated agents read the saved issue #11
  list (plantseg_exact_duplicates.csv, sha256 1354add9…) and printed its header, one row naming a
  TEST image, three TRAIN/VAL rows and line counts. They also listed the member names of the dataset
  archive in memory, filtering TEST names before printing, and scanned its raw bytes. No TEST image
  or mask was decoded, no model was evaluated, and nothing in this amendment or in any run was chosen
  from these reads (decision-log deviation row DL-58).
- No E2, E3, A, F, G, SegNeXt-L, extra-teacher or other KD run has started. No KD result exists.
- No statistic of item 1 has been computed. The per-class VAL table of AM-17 item 1(f) and the
  AM-17b item 2 re-scores are registered separately and may already exist. TEST has not been
  evaluated.
- Already known, so nothing below is blind to it:
  - the teacher of record's ten interval VAL scores (item 9(d)), its per-class VAL IoU (the R3
    summary) and its full-VAL margin over E1 seed 42 (2.26 pp; about 1.95 pp GT-present);
  - E1's three VAL scores and their seed SD (s = 1.10 pp);
  - B61's measurements on the ADE20K checkpoint: between NMF draws the predicted class changed on
    26.0% of valid pixels on one VAL image (30.1% on the 64×64 grid; 23.9% over all 190 draw
    pairs), and on 26.9%, 0.003%, 7.8% and 3.2% on four rule-selected VAL images; an 8-draw
    average still differed from a 64-draw reference on 6.1% of pixels (14.1% for one draw);
  - issue #11's counts (item 8(a));
  - the outputs of the local data session, each with its decision-log entry: the upstream-protocol
    VAL scores of the teacher of record (AM-17 item 1(e); 0.36319661140441895) and of E1 seed 42
    (AM-16 item 4; 0.35182613134384155) (DL-08, DL-27);
    the AM-17 item 8 strata files reports/strata/train_strata_v1.json and .csv (sha256 ab0df4e3… and
    c793c11f…; DL-63); the AM-17b item 2(c) CPU re-scores (teacher of record
    0.38576993346214294; E1 seed 42 0.36307525634765625), with the item 2(a) gap estimates
    (union-present 0.022694691891384178; GT-present 0.0195375159424252) and their BCa 95% CIs and
    the item 2(b) per-image summary, in reports/derived/gap_val_20261003T014324Z.json (sha256
    3597a82f…; DL-64); and the AM-17 item 1(f) per-class table,
    reports/derived/perclass_gap_val_20261003T014403Z.json and .csv (sha256 4a8fa2b4… and
    0ba3aa7d…; DL-64); and the AM-10 list of record and the AM-16 item 5 lists,
    configs/calibration/ptq_calibration_seed{42,43,44,45}.json (DL-10).
  Every arm and the item 1(e) window were chosen with the teacher's VAL curve known, so every arm
  is descriptive.
- The only rules that can change a run of record are item 6's two pre-set rules. Item 6(a) fixes
  its statistic, thresholds and consequence here. Item 6(b) fixes its statistic, threshold and a
  halt; what follows the halt is not pre-registered.
No test is added to the Holm family. No test, threshold, family member, grid, seed or teacher of
record changes.

Execution rules for every statistic and gate below:
- Development. The scripts are developed on synthetic inputs, or on real TRAIN and VAL images
  with the stub teacher only. A script's execution of record, for each teacher or input set, is
  its first execution that loads those real teacher weights or, for a script that loads no model,
  the real split files or score artifacts. The script's commit is entered in the decision log
  before its first execution of record and is recorded in every output; later executions of
  record use the same commit, or the repeat's commit after case (ii) below. Executions use the
  pinned teacher image on CPU.
- One output. The first complete output of an execution of record is the output of record. No
  statistic is recomputed with another sample, seed, K or definition. A control re-score (item 5;
  the seed-42 stream of item 1(a)'s VAL part) is a reproduction check, not a second output.
- Repeats (this extends AM-8a to these scripts). An execution is repeated only:
  (i) after a documented infrastructure fault (AM-8a), or after any other documented stop that
      left no complete output, with identical inputs and the same script commit; or
  (ii) after a script defect: a departure of the script from a definition written in this
      amendment, shown by a diff. A wording with two readings is not a defect. The defect report
      lists every execution made with the defective script, and all of them are repeated. The
      report, the diff and the new commit are entered in the decision log before the repeats.
      Each repeat is then the output of record, and both outputs are reported.
  Where an output and its case (ii) repeat select different outcomes, the outcome is fixed here
  and not chosen: item 6(a) takes the branch of the larger F, item 6(b) takes the halt, and item 5
  takes the no-go. A repeat changes an outcome only before it is acted on: once the first KD run
  has launched, the entered item 6 branches stand, and once E3-X has launched, its item 5 outcome
  stands; the repeat is still made and both outputs are reported. A STOP below is handled as case
  (i), or as case (ii) if a script defect caused it; if it recurs, the statistic is reported as
  not produced.
- Gate. Outputs are hashed and entered in the decision log. The item 1(a) output that holds F and
  the item 1(d) output, with the item 6 branches they select, are entered before the first KD run
  (the first real-mode training run of any distillation stage; smokes and the invariance harness
  are not runs). The first KD run does not launch before those entries exist. If F or
  margin_dedup cannot be produced, no KD run launches and the adviser is informed. [AM-19 item 2(h): the first KD run also waits for the sweep selection code of K2.]
- Comparison. F is compared in integers: with N the pooled flip count and D = 28 · Σ_i |V_i|
  (item 1(a)), F < 0.03 means 100·N < 3·D, and F ≥ 0.10 means 10·N ≥ D. Every other threshold is
  compared on the float64 values written to the outputs, without rounding. margin_dedup is also
  computed as an exact rational and in the evaluator's float32 arithmetic; if any of the three
  values is ≤ 0, item 6(b)'s halt applies.

1. Teacher diagnostics R0. Descriptive and training-free. F and margin_dedup enter item 6; the ECE
   method of item 1(b) and B-avg (item 1(e)) enter item 5; nothing else enters a rule.
   (a) D1 — NMF-draw sensitivity of the distillation targets. Item 6(a) fixes when it may run.
       - Sample: random.Random(1801).sample(sorted TRAIN ids, 256), with the TRAIN ids derived as
         for the AM-10 lists (split-list sha256 equal to that of the AM-17 item 8 strata file,
         which exists first).
       - Crop i (i = 0 to 255, in sample order): one pass through the KD training pipeline (M2/M3)
         with np.random.RandomState(1801 + i), then the trainer's normalisation.
       - Forwards: the teacher of record through the KD teacher adapter, batch 1. The backbone
         runs once per crop and the decode head K = 8 times on that backbone output. One private
         NMF stream seeded 42 (the M4-KD construction) advances once per head forward, crops in
         sample order (2,048 draws).
       - Statistics, on the 64×64 logit grid over the cells of the trainer's all-valid min-pooled
         mask (contract B3; losses.downsample_validity; the rule AM-17 item 10(c) records after
         its 2026-09-30 correction). V_i is that set for crop i; a crop with no valid cell adds
         nothing.
         - F (the only statistic item 6(a) uses): the pairwise argmax flip rate over the 28 draw
           pairs, pooled over cells and crops. F = N / D, with
           N = Σ_i Σ_{a<b} |{p ∈ V_i : argmax z_a(p) ≠ argmax z_b(p)}| and D = 28 · Σ_i |V_i|.
           An image-level percentile bootstrap 95% CI (B = 10,000; numpy default_rng(1801)) is
           reported with it; the point estimate alone selects the branch.
         - F_lesion (secondary): F restricted to the cells whose mean probability (T = 1, eight
           draws) peaks at a disease class; reported as undefined if there is no such cell.
         - F_halves (secondary): the argmax flip rate between the mean-probability predictions
           (T = 1) of draws 1–4 and of draws 5–8, which shows how far averaging removes the flips.
         - KL_logit: the mean over draws and valid cells, pooled over crops, of KL(p_k ‖ p̄) in
           nats, with no T² factor, where p_k = softmax(z_k/4) and p̄ is the mean of the eight p_k
           (the K8 target, not the softmax of the mean logits).
         - KL_cwd: the same for the channel-wise spatial distributions of the logit map (per
           channel, softmax over the valid cells at T = 4, as the CWD logit-map term computes
           them), averaged over draws and channels.
       - Output: reports/derived/teacher_d1_<UTC>.json, with the sample list's sha256, the
         checkpoint and configuration sha256, the script commit and every statistic.
       - VAL part (descriptive; a separate output that gates nothing): VAL in the R3 manifest
         order, batch 1, the backbone once per image and the head once on each of eight streams
         seeded 42 to 49 (each stream advances one draw per image, as M4-V). Reported per stream:
         the dataset-level all-class mIoU and the item 1(b) ECE, with their mean, sample SD and
         range; and the mIoU of the mean-probability prediction (softmax at T = 1 on the canvas,
         averaged over the eight draws). The seed-42 mIoU must reproduce the R3 value of record
         within 1e-5 (the L-AM17B-GAP tolerance); otherwise the VAL part stops and is reported as
         not reproduced, which blocks no other item. Whatever the spread shows, the seed-42
         single-draw scores remain the scores of record for every rule.
   (b) D2 — confidence and calibration of the teacher of record.
       - On the 256 D1 crops (first draw; 64×64 grid; valid cells): mean max-probability and mean
         entropy (nats) at T = 1 and T = 4, and the share of valid cells with max-probability
         > 0.99 at T = 1.
       - On VAL (846 images; the R3 evaluation procedure): the same three quantities over valid
         canvas pixels, and the ECE. This pass must reproduce the R3 value of record within 1e-5;
         otherwise it stops and is run on another host.
       - VAL ECE: pixel-level, top-1, T = 1, on the 512×512 canvas. A pixel's confidence is the
         largest softmax probability of the bilinearly resized logits whose argmax is the scored
         prediction. All n valid (non-255) VAL pixels are pooled into M = 15 equal-width bins
         ((m−1)/15, m/15] (Guo et al., 2017); ECE = Σ_m (n_m / n)·|acc_m − conf_m|. The ECE over
         pixels with disease ground truth is reported alongside; item 5 uses the all-pixel ECE.
       - Teacher TRAIN mIoU: all 5,367 TRAIN images under the R3 evaluation procedure, with the
         sorted TRAIN list in place of the VAL manifest, through a diagnostic entry point
         (all-class, union-present; per class for item 1(c)). The evaluator's split gate is
         unchanged.
   (c) D3 — per-class TRAIN and VAL IoU of the teacher of record by the AM-17 item 8 strata (the VAL
       part is AM-17 item 1(f)).
   (d) D4 — duplicate-free VAL.
       - SHA-256 (and SHA-1, the issue's hash) of the file bytes of all 5,367 TRAIN and 846 VAL
         images. The script refuses any path containing "test". Both hash lists are kept in the
         run's evidence folder and their sha256 is recorded.
       - A VAL image is a duplicate when its SHA-256 equals that of any TRAIN image. Recorded: the
         duplicate VAL images and groups, and for each group whether the decoded annotation masks
         are identical. Issue #11 reports 32 "train ↔ val" groups and 15 groups it labels "3-way
         and more"; a different count is reported, not a STOP. Only byte-identical files are
         detected; re-encoded or resized copies are not, which is disclosed.
       - From the per-image per-class confusion contributions of the AM-17b item 2(c) CPU
         re-scores (both models on CPU; on the full VAL this pairing gives about 2.27 pp):
         dataset-level VAL all-class mIoU (union-present, the R3 rule) and GT-present mIoU of the
         teacher of record and of E1 seed 42 on VAL without the duplicate images, with class
         eligibility recomputed on that subset.
       - margin_dedup = teacher − E1 seed 42 (union-present, all-class) on that subset.
       - Recorded with the duplicate list's sha256.
   (e) D5 — B-avg, a post-hoc averaged teacher from existing checkpoints.
       - Weights: every floating-point entry of state_dict is the uniform average, computed in
         float64, of that entry in the checkpoints at iterations 16,000, 20,000, 24,000, 28,000,
         32,000, 36,000 and 40,000 of run teacher_official_20260925T122252Z; other entries come
         from the 24,000 checkpoint. Each checkpoint is verified against the run's SHA256SUMS
         first.
       - BatchNorm: the backbone's running statistics are reset and re-estimated as
         torch.optim.swa_utils.update_bn does (cumulative average; training mode, so stochastic
         depth is active as in training), running only the backbone, over one pass of all 5,367
         TRAIN images in sorted order: batch 16 (336 batches), image i through the training
         pipeline (M2/M3) with np.random.RandomState(1801 + i) and the teacher's input
         normalisation, torch seed 1801, no labels, no gradients, CPU.
       - The result is saved as a checkpoint with its sha256 and scored on VAL by the R3
         evaluation procedure, with the item 1(b) VAL ECE, in a session that follows item 5.
       - If any of the seven checkpoints is missing or fails its hash, D5 is not run and is
         reported as not executable.
       - The window is fixed here and never changes. It was chosen with the ten interval scores
         known.

2. Fine-tuned arm teachers R1 and R2. Descriptive. One fine-tune each, on one A40 at batch 16
   (never reduced); launched only after the adviser approves this amendment and the arm's code is
   merged and pinned; never on the critical path.
   - Each arm has a new configuration file. Its sha256 and its differences from the configuration
     of record are entered in the decision log before launch.
   - Each arm launches through an arm launcher that applies the frozen launcher's gates to that
     registered difference (the frozen launcher refuses a fourth hook and any other train pipeline).
     None of the nine frozen teacher-runtime files changes.
   - Readiness rules 1 (integrity, no resume), 2 (TRAIN/VAL only) and 4 apply as for the teacher of
     record. Item 5 takes the place of readiness rule 3.
   - A fine-tune whose logged training loss or VAL score becomes non-finite is recorded as
     diverged, reported and not rerun. An infrastructure fault follows AM-8a.
   (a) R1 — "B-EMA". The configuration of record (contract B1; M2, M3, M4, M5 40,000 iterations,
       M11, M12, M13; ADE20K initialisation sha256 647a0cda…; seed 42; deterministic) with exactly
       two changes:
       - MSCAN-B drop_path_rate 0.2: the SegNeXt author's recommendation for B (MMSeg PR #2247);
         the record inherits 0.1.
       - mmengine 0.10.7 EMAHook: ExponentialMovingAverage, momentum 0.0002 (average ← 0.9998 ×
         average + 0.0002 × current weights; a 5,000-iteration window), update_buffers True,
         interval 1, begin_iter 0, strict_load False.
       The first update copies the model and there is no bias correction: at iteration t the
       average still holds 0.9998^(t−1) of the first-iteration weights (0.449 at 4,000; 0.041 at
       16,000; 0.008 at 24,000). Every interval VAL scores the EMA weights and BatchNorm statistics,
       and M12 selects among those ten scores. Each saved checkpoint's state_dict holds the EMA
       weights; KD and every evaluation load state_dict. The raw weights saved under ema_state_dict
       are never used.
   (b) R2 — "B-EMA + rare classes". R1 plus two changes to the teacher's TRAIN data, fixed here. A
       rare class has n_c < 20, where n_c is the strata file's TRAIN image count of disease class c
       (AM-17 item 8). Every PlantSeg image holds one disease class (EVALUATION_CONTRACT §0).
       (i) Repeat-factor sampling (Gupta et al., 2019) with t = 20/5,367. An image of class c
           appears m_c times per pass: m_c = ⌈√(20 / n_c)⌉ for a rare class and 1 otherwise, that
           is 5, 4, 3, 3 for n_c = 1, 2, 3, 4 and 2 for n_c = 5 to 19. Background is not a class
           here, and an image whose mask holds no disease pixel appears once. This is mmengine
           ClassBalancedDataset with oversample_thr = t; the launcher asserts that its repeat
           counts equal this table. Each pass is a random permutation of the repeated list. The
           40,000-iteration budget is unchanged.
       (ii) Rare-class copy-paste between images of the same class, applied to the EXIF-corrected
           raw image and mask before M2/M3. It keeps one disease class per image.
           - When the target T's disease class c is rare: with probability 0.5, one source S other
             than T is drawn uniformly from the TRAIN images of class c. If class c has no other
             TRAIN image, nothing is pasted.
           - Patch: every pixel of S labelled c, with its image content. It is resized by the ratio
             of T's to S's long side times u ~ U[0.75, 1.25] (bilinear image, nearest mask; sizes
             rounded to the nearest pixel, at least 1). If its bounding box then exceeds T in
             either dimension, it is scaled down, aspect ratio kept, by the largest factor at
             which it fits.
           - Placement: the bounding box is placed uniformly at random inside T and accepted when
             at most 5% of the patch's pixels fall on disease pixels of T. Up to 10 placements are
             tried; otherwise nothing is pasted.
           - Pasted pixels overwrite image and label; there is no blending. All draws come from the
             sample's own RandomState, before the M2/M3 draws, in this order: the paste decision,
             the source, u, the placements.
           The table of n_c and m_c and the per-class source lists are written by the arm's
           builder from the strata file and the TRAIN masks, and their sha256 is entered in the
           decision log before R2 launches. A count that disagrees with the strata file is a STOP;
           R2 does not launch while it stands.
       The student's pipeline, every KD-time input, VAL and TEST are unchanged. If R1's fine-tune
       is recorded as diverged, R2, which shares its recipe, is not launched.
   (c) Scoring. Each arm teacher (B-avg, R1, R2) is scored on VAL by the R3 evaluation procedure,
       R1 and R2 at their M12-selected checkpoint: VAL_X, with the item 1(b) VAL ECE (ECE_X), the
       item 1(b) TRAIN-crop profile and the item 1(a) TRAIN statistics (descriptive; item 6(a)
       uses the teacher of record's F only). An arm teacher that is complete and hashed at the
       freeze is scored once at the TEST evaluation: clean, canvas protocol, descriptive. [AM-19 item 1(c): the freeze is that of the schedule in force; "complete and hashed" reads as AM-19's "Complete" (Definitions).]

3. KD arms. Descriptive; seed 42; launched after E3 seed 42's λ and α are selected. λ and α are
   inherited and disclosed, as in AM-17 item 11(c). Every KD arm of this item and of AM-17 item 11
   builds its targets as the runs of record do (one NMF draw per step, or the K = 8 average under
   item 6(a)'s correction), so an arm differs from E3 only in what its definition names. Each KD
   run's run_meta records its target construction (K and the number of views).
   (a) For each arm teacher X ∈ {B-avg, R1, R2} that passes the item 5 go rule:
       - E3-X = E3's recipe of record (Logit-KD + both channel-wise terms, 80,000 iterations,
         seed 42; the same projection against Stage 3 at 320 channels; the same grids, ramp,
         validity handling, loader policy, hardware rule and best-VAL selection) with X in place of
         the teacher of record. X is the checkpoint item 2(c) scored.
       - E7-X = PTQ of E3-X with the AM-10 calibration list of record (calibration pass only).
       - No E2-X, no E4/E5/E6-X, no seeds 43/44, no robustness or efficiency scoring. The runs are
         written E3-avg, E3-R1 and E3-R2 (E7-avg, E7-R1, E7-R2).
   (b) W — flip-averaged targets on the teacher of record. E3-flip = E3's recipe of record in which,
       at every step:
       - the teacher runs on the crop and then on its horizontal flip, and the second set of logits
         is flipped back (for this arm only, a departure from contract B2's single teacher input);
       - the Logit-KD and CWD logit-map targets are the mean of the two views' distributions:
         probabilities are averaged, each term in its own softmax at T = 4, never logits;
       - the CWD feature-map term uses the unflipped view's Stage 3;
       - the M4-KD stream advances once per head forward;
       - the student sees only the original crop.
       E7-flip = PTQ of E3-flip, as in (a). The flipped view carries a second NMF draw, so
       E3-flip − E3 reads as flip plus a second draw; where K8 runs, it is also read against E3-K8.
       Under item 6(a)'s correction each view's distribution is its own K = 8 average, the
       unflipped view's eight draws first.
   (c) K8 — K-draw averaged targets on the teacher of record. It is an arm under item 6(a)'s middle
       branch, and under its upper branch when the correction is unavailable. E3-K8 = E3's recipe
       of record with the decode head run K = 8 times per step on the one backbone output (M4-KD
       stream advancing once per head forward). The Logit-KD and CWD logit-map targets are the mean
       of the eight distributions (probabilities, as in (b)); the feature-map term is unchanged.
       E7-K8 as in (a).
   (d) Reporting, as in AM-17 item 11(d) and AM-17b item 1(e), for X ∈ {avg, R1, R2, flip, K8}:
       - VAL and TEST clean dataset-level mIoU (all-class, GT-present, disease-only);
       - the dataset-level all-class differences E3-X − E3 and E7-X − E7, each read next to the
         AM-16 item 2 seed-noise scale (√2·s for a two-run contrast); a difference inside that
         scale is reported as not distinguishable from seed noise;
       - per-image mean ΔmIoU with BCa 95% CIs for the same contrasts;
       - each arm teacher's VAL_X and ECE_X alongside.
       No test; not in the Holm family; never a parent of a run of record. E6 and E7 of record are
       built only from E3 (AM-15; AM-17 item 11(e)).
   (e) Divergence: the KD runs of this item and of AM-17 item 11 follow AM-7a item 3's arm rule
       (DL-04: stop, record, no rerun, no clipping rule). [AM-19 item 2(g): where the α sweep is cut, its committed decision record stands in for the α selection. Item 7(d): "no rerun" in item 3(e) applies within one recipe; an arm run of a superseded recipe is reported as superseded, and its rerun is a cuttable run in its place in the cut order.]

4. SegNeXt-L (AM-17 item 11) keeps its configuration, go rule 11(b), arm 11(c), reporting and the
   DL-32 gate exactly as item 11 stands when this amendment is committed. Any later change to
   item 11 is a new amendment that lists the item 1 outputs then known. This amendment places L in
   the run and cut orders (item 7) and applies items 3 (target construction) and 3(e)
   (divergence) to its KD arm.

5. Go rule for the arm teachers (B-avg, R1, R2; L keeps 11(b)). E3-X and E7-X run if either:
   - VAL_X > 0.38576993346214294; or
   - VAL_X ≥ 0.37576993346214294 and ECE_X < ECE_B.
   Otherwise X is reported as a teacher-only result and no KD runs with it.
   - The session that scores X runs on the host that produced item 1(b)'s VAL output (another host
     only after a documented fault of that host). The teacher of record is re-scored first, at the
     same commit. It must reproduce 0.38576993346214294 within 1e-5; otherwise the session is void
     (a STOP under the execution rules), and if that recurs X is reported as not evaluable and no
     KD runs with it. ECE_B is the teacher of record's VAL ECE in that session; item 1(b)'s value
     is reported beside it.
   - Each outcome is entered in the decision log, with the values and the artifacts' sha256, before
     E3-X launches.
   - The second condition admits a teacher that is a better source without being a better model
     (Kim et al., 2025; Wang et al., 2022), which a strict accuracy gate would never admit. The
     1 pp allowance is a convention at the scale of E1's measured seed SD (s = 1.10 pp, n = 3); no
     teacher seed SD exists (AM-1).

6. Pre-set defect rules: one correction and one halt. Fixed before the statistics exist. Under
   item 6(a), DL-30 stands and the teacher of record's weights never change.
   (a) D1 rule, on F only (item 1(a)):
       - F < 0.03: no change, and no K8 arm.
       - 0.03 ≤ F < 0.10: no run of record changes; K8 runs as a descriptive arm (item 3(c)).
       - F ≥ 0.10: recorded as a defect of the single-draw targets. If the correction is available
         (next point), then in every KD run — the λ and α sweep candidates, E2 and E3 at every
         seed, the 160,000-iteration E2 and E3 controls, arms A, F and G, E6-KD if triggered, the
         SegNeXt-L arm and the item 3 arms — the Logit-KD and CWD logit-map targets are the K = 8
         draw average of item 3(c), computed on that run's own teacher. The feature-map term is
         unchanged. The head runs eight times per step and per view in every stage (F included),
         so the NMF stream advances identically across stages. E5 and E6 have no KD targets; E6
         and E7 inherit the correction only through E3. In the corrected state K8 is not a
         separate arm.
       - Availability is settled before F exists. The correction is available if and only if a
         reply that approves it, dated no later than the seventh day after the request was sent
         (Asia/Manila), is in the decision log when the state is entered; a reply that approves
         AM-18 without excepting item 6(a) approves it. Until a reply that approves or declines
         the correction is in the decision log, or that seventh day has ended, nothing is
         executed that draws more than one NMF sample per input on real teacher weights: no part
         of item 1(a) on any teacher, and no smoke of the K8 or flip targets. The state
         (available, declined or no approval) is entered in the decision log before any such
         execution and is final. A later reply can still approve the arms. [Amended 2026-10-07 — AM-20 item 2: no request was sent, so no window exists; the state is "no approval", entered by AM-20's commit (DL-84); the correction is unavailable for every run.]
       - If the correction is unavailable and F ≥ 0.10, the registered single-draw targets stand
         for every run and F is disclosed as a limitation. K8 is then an arm as in the middle
         branch; like every arm, it launches only if this amendment is approved.
       - If the correction is available and F ≥ 0.10, it applies from the first KD run and is not
         withdrawn, for cost or for any other reason. Before the first KD run it needs the
         target-averaging code merged, pinned and smoke-tested, with its measured slowdown
         recorded, and the decision-log entry with the contract notes (B1 M4, B2, B3); the first
         KD run waits for both. An overrun is handled by item 7(b) and AM-17 item 9.
       - This is the multi-draw averaging that contract M4 and B61 §5 did not adopt for KD targets.
         It is adopted here, for KD targets only, in the W and K8 arms and under this correction,
         because M4 was locked on measurements of the ADE20K checkpoint, before the fine-tuned
         teacher existed. M4-T, M4-V and every evaluation score are unchanged.
       - F is measured on the teacher of record only; B-avg, R1, R2 and L inherit the branch. [AM-19 item 2(a): the decision dates do not move for the window or the correction; item 2 handles the shortfall. Item 7(c): if AM-7's full consequence fires, the item 1 and item 6 entries stand, labelled as computed on the superseded recipe, and are not recomputed.]
   (b) D4 rule on margin_dedup (item 1(d)):
       - margin_dedup > 0: recorded, no change.
       - margin_dedup ≤ 0: the R3 floor is not met on duplicate-free VAL. No KD run launches and
         the adviser is informed, as contract B1's R3-failure rule requires (STOP and escalate; no
         retraining, search, band relaxation or TEST). R3 (AM-13; AM-17 item 1(a)) is then restated
         on duplicate-free VAL by its own amendment, approved by the adviser before any KD run. [Reading settled 2026-10-07 — AM-20 item 4: that approval is a written reply that names the amendment.]
         Only this halt is pre-registered: what that amendment decides is not fixed here. It would
         be written with margin_dedup known, it says so, and it uses no KD or TEST result.
   (c) Neither rule is evaluated on any KD result. Item 6(a) never replaces the teacher of record
       (DL-30). No other statistic of item 1 triggers or vetoes a branch.

7. Order, launch and cut order.
   (a) Run order:
       - item 1 first (F and margin_dedup before the first KD run);
       - R1 and L in parallel on separate cards, once their gates are met;
       - R2 once R1's fine-tune has completed its 40,000 iterations without divergence, whatever
         R1's scores;
       - the item 3 KD runs after E3 seed 42's selections and, when cards are scarce, in the
         reverse of the cut order of (b).
       R1 launches only when R2's code, configuration and table are pinned too. The code of all
       item 3 arms is pinned in one commit before E3 seed 42's α selection exists; if it is not,
       all of them are recorded as cut at that point, with the reason and the results known.
       An arm whose gates are met launches when a card is free that no run of record is waiting
       for; it is not stopped for one later. A launched arm run ends only by completion, divergence
       (item 2 or item 3(e)), a documented fault (AM-8a) or the freeze. An arm that is not launched
       is recorded as cut, with the reason and the arm results known at that time. [AM-19 item 1(e) extends this card rule to every cuttable run and sets the order in which waiting runs, R1, R2 and L included, take a free card; item 3(a) governs a stop that no rule names and a trainer abort that is not a divergence; item 3(b)'s arithmetic cut is an arm's "not launched" record.]
   (b) Cut order for runs not complete and hashed at the freeze (2026-12-18), first to last:
       1. R2 (teacher, E3-R2, E7-R2);
       2. W;
       3. K8;
       4. R1 (teacher, E3-R1, E7-R1);
       5. E3-avg and E7-avg;
       6. then AM-17b item 3's order (SegNeXt-L arm, G, F, A, AM-16's order, E5/E6 seeds 43/44,
          E3 seeds 43/44).
       An arm is cut from its first incomplete run onward; a finished arm teacher is still reported
       as a teacher-only result. Item 1 is never cut. Per AM-17b item 4, every arm of this
       amendment is cut before every arm listed there. Each cut is recorded in the decision log
       before TEST (AM-17 item 9). [AM-19 item 1(c): "the freeze (2026-12-18)" means the original freeze date of the schedule in force, and "complete and hashed" reads as AM-19's "Complete" (Definitions). Item 2(g): a non-default α candidate cut at its decision date is the one exception to this order. Item 3(c): each cut is entered in the decision log on the day it is made. Item 7(b): rerun E1 runs at seeds 43/44, if any, follow E3 seeds 43/44.]
   (c) Cost (INFERRED; each run's measured GPU-hours are reported):
       - item 1 runs locally on CPU, roughly 12 to 18 hours in all, from the lane specification's
         estimate of about one hour for a teacher VAL pass;
       - two 40,000-iteration teacher fine-tunes, each at the record run's cost;
       - up to four 80,000-iteration KD arms (E3-R1, E3-R2, E3-flip, E3-K8; E3-avg does not exist)
         at 29–33 h each (AM-17b item 1(f)), before the extra teacher forwards of W and K8, which
         are measured at their smoke.

8. Dataset duplicates: record and disclosure.
   (a) Record: issue tqwei05/PlantSeg #11 reports, by SHA-1 over the image files of the Zenodo
       10.5281/zenodo.17719108 release used here (5,367/846/1,561), "265 duplicate groups (277
       redundant files)", about 3.6% of 7,774, in these rows (the labels are the issue's):
       - train ↔ train 124;
       - test ↔ train 70;
       - train ↔ val 32;
       - test ↔ test 13;
       - test ↔ val 11;
       - 3-way and more 15.
       Its attached list (plantseg_exact_duplicates.csv) is saved with its retrieval date and
       sha256 and is not opened again before the TEST evaluation session (State at amendment);
       nothing on TRAIN or VAL is chosen from it. The repository's dataset report checks file-stem
       duplicates only (none found). R0 reads no TEST file: until the TEST evaluation the TEST side
       is known only from the issue's counts and the reads recorded under State at amendment.
   (b) VAL: items 1(d) and 6(b). Wherever a VAL score is reported from an evaluator artifact with
       per-image contributions, its duplicate-free VAL value is reported next to it. Apart from
       item 6(b), no selection, trigger or go rule is evaluated on duplicate-free VAL: every
       registered selection rule is written for the full 846 images, and E1's three seeds were
       already selected on them.
   (c) TEST:
       - The official TEST evaluation, every test of the Holm family and the non-inferiority check
         stay on the full 1,561-image split.
       - No step of this amendment reads TEST before the single TEST evaluation session. In that
         session, before any model is scored, the TEST image files are hashed (SHA-256 of the file
         bytes; no annotation is read) by the script frozen in the TEST manifest.
       - Duplicate-free TEST = TEST without every image whose SHA-256 equals a TRAIN or VAL
         image's, keeping only the first image (by sorted id) of any group of identical TEST
         images. The list follows from file bytes alone and from no model output.
       - Reported next to the official results, from the per-image outputs of the single TEST
         evaluation (no model is run on TEST twice), descriptively: the dataset-level mIoU
         (all-class, GT-present, disease-only; class eligibility recomputed on the subset) of every
         model of record and every arm; and, as a sensitivity analysis, the eight family contrasts
         and the E3-vs-E6 check recomputed on the subset (same procedures, unadjusted). They
         replace no official result and decide nothing: if one disagrees with an official result,
         the official result stands and Chapter 4 reports both.
       - The numbers of images removed for each reason are reported next to the issue's counts.
   (d) Chapter 4 states the duplicates, this procedure and its results, and Chapter 5 lists them as
       a dataset limitation. Chapters 1–3 are not rewritten (DL-20).

9. Record corrections. No rule changes; recorded in the docs session that commits this amendment.
   (a) AM-17 item 10(e) stands, and its source is recorded: arXiv 2409.04038v1, "Evaluation on
       PlantSeg", gives SegNeXt "the MIoU of 53.89% and the mAcc of 65.91%" on the 11,400-image
       release. The same preprint's Table 3 lists SegNeXt MSCAN-L at 44.52% mIoU, as do the pinned
       upstream README and Sci. Data Table 3. 53.89% is not a comparator; 44.52% stays the
       descriptive reference of AM-17 item 11(d).
   (b) The teacher runbook's §3 drop_path_rate 0.1 is an inherited default that was never examined.
       The SegNeXt author recommends 0.2 for B (MMSeg PR #2247). The teacher of record keeps 0.1;
       R1 tests 0.2.
   (c) SegNeXt-L memory: the 43.3 GB in AM-17 item 11(a) as first registered is the MMSeg figure for training SegNeXt-L on
       ADE20K (43.32 GB; 160,000 iterations on one A100). The measured peak of this pipeline is
       entered in the decision log when it is first measured (item 11(a)'s preflight, or the launch
       if item 11(a) has none), before any L score is read. No other figure is of record.
   (d) The teacher of record's ten interval VAL scores (4,000 to 40,000) are entered in the
       decision log at full precision from the run's teacher_selection_records.jsonl, with that
       file's sha256.

10. Rationale. The adviser asked for all possible teacher configurations to be checked. A 200-query
    audit (2026-10-01) assessed 49 candidates, and the group adopted the evidence-led subset:
    - weight averaging and stochastic depth (the best-supported teacher change and the architecture
      author's advice);
    - a rare-class data arm (the long tail is the dataset's main difficulty);
    - the registered size arm (L);
    - a target-level wrapper that needs no new teacher.
    The other candidates were dropped or deferred by the group; the lists are in this amendment's
    decision-log row. The vote's condition for a second wave (only if this wave shows that the
    student responds to teacher changes) is not registered as a rule: a later wave needs its own
    amendment before any of its runs (AM-17b item 4), and the choice to write one would be made
    with this amendment's results known, which is disclosed.
    AM-17 item 1(c) fixes the teacher of record. Choosing it by a student result would select on
    the quantity the Holm family tests, and choosing it by a teacher score is the switch DL-30
    rejected, so every teacher here stays an arm.
    Two diagnostics can expose a defect rather than a weak score: NMF-draw noise in the targets, and
    duplicates inflating the VAL margin. Their triggers are fixed now, with item 6(a)'s consequence
    and item 6(b)'s halt, so applying them is pre-registered and not a post-hoc switch. The 3% and
    10% thresholds and K = 8 were set by the group before any measurement on the teacher of record;
    no external source prescribes them. F is computed on the 64×64 grid over all valid cells
    because that is where the KD targets live; F_lesion shows what the background share hides.
    Limits of the arms, stated in advance:
    - R1 changes two things at once, so their effects are not separated.
    - R2 repeats TRAIN images that may have byte-identical VAL copies; its duplicate-free VAL score
      is shown beside its go-rule score (item 8(b)).
    - The paste stays inside a class because every PlantSeg image holds one disease class and the
      classes are host-specific. No paste rate is published for PlantSeg. On a three-class wheat
      set, Wei et al. (2025) report one pasted patch per image as best, a loss below baseline from
      three patches on, and little gain from plain copy-paste without their random-projection
      filter.

11. Disclosure.
    - Chapter 4 lists every arm with its teacher's VAL score (and ECE for the arm teachers), its
      go/no-go outcome, its KD results next to the seed-noise scale, and every cut; it also gives
      R0's statistics, which item 6 branch applied (a correction that applied is listed in the
      deviation table) and the item 8 results.
    - Chapter 5 states the teacher-recipe findings as recommendations for future work.

Code:
- L-TEACHER-DIAG (item 1: the D1–D5 scripts, the ECE reducer, a TRAIN scoring entry point,
  TRAIN/VAL hashing, the duplicate-free reducer and the B-avg builder; the arm role of item 2(c)
  and the control re-score mode of item 5; local CPU executions);
- L-TEACHER-ARMS (item 2: the R1 and R2 configurations, the arm launcher, R2's dataset wrapper,
  paste transform and table builder; the provenance record of teacher-stage evaluation artifacts
  for a teacher built from another configuration, which L-AM17-L carries instead if it is
  implemented first);
- L-KD-WRAP (items 3(b) and 3(c) and item 6(a)'s correction: per-switch target averaging in the KD
  trainer, default off, and under the correction also in the E6-KD step (L-AM3) and the SegNeXt-L
  arm's KD runs; arm stage registration and launch profiles);
- the TEST-manifest lane (item 8(c); arm scoring and reporting at TEST);
- L-AM17-L for SegNeXt-L.

## AM-19 — Schedule, sweep decision dates, stops and cut records, runs with a cut or diverged parent, divergence at seeds 43/44, seed stability with a missing run, and AM-7's full consequence under a schedule (amends AM-7, AM-7a items 1–4 and 6, AM-8a and AM-17 item 9; extends AM-1, AM-2, AM-7a item 5, AM-8, AM-16 items 1–3 and its cut rule, AM-17 items 4 and 6, AM-17b item 3 and AM-18 items 1, 2(c), 3, 6 and 7)

Dated 2026-10-02. Status: DRAFT (group-recorded; adviser approval requested). [Erratum E-52, 2026-10-07: no approval request was sent (AM-20, DL-82). The adviser's in-person approval of 2026-10-05 is recorded as information only (DL-83).]
- Fixed text. From its commit this text is changed only by a new amendment, which lists the results
  known when it is written. A reading this text leaves open is settled the same way. A dated
  bracket note only points to a later record or records an outcome, such as an adviser's reply.
- In force from the commit, for every event after it. Approval gates nothing here: no item waits
  for it.
- Approval means an explicit written reply from the adviser that approves this amendment, either
  naming AM-19 or answering a request whose subject names AM-19 and which was sent after this
  amendment's commit, recorded verbatim with its date in the decision log.
  An acknowledgement ("noted") is not approval. A reply that approves without excepting an item
  approves every item. An item that a reply excepts stays in force unless the reply says in words
  that it rejects that item. A requested change needs a new amendment.
- A rejection changes a rule only if it is recorded verbatim before the first KD run launches. The
  fallbacks are then:
  - item 1: Schedule R, with its two decision dates;
  - item 2: AM-17 item 9's shortfall rule and AM-7a as written; their conflict with the never-cut
    E3, E6 and E7 at seed 42 (item 2(a)) is settled by a new amendment before the first
    non-default candidate launches;
  - item 3: AM-16's place of record and AM-8a as written;
  - item 4: AM-17 item 9's never-cut wording; its conflict with the cut order is settled by a new
    amendment before the freeze;
  - item 5: AM-7a item 3 for seeds 43 and 44 (AM-7's full consequence);
  - item 6: AM-8 as written: "stable" only if all three conditions hold;
  - item 7: nothing is pre-set, and a new amendment is written before any rerun launches.
  A rejection recorded later is entered verbatim and reported. It changes a rule only through a new
  amendment, which lists the results known when it is written, and it never reopens what was
  already decided: a selection made, a run stopped or cut, a divergence recorded, a verdict
  reported.
- Naming. "Schedule T" and "Schedule R" are the two schedules of item 1. They are unrelated to the
  arm teachers R1 and R2 and to readiness rule R3 (AM-18's naming).

State at amendment:
- Committed before this amendment: AM-7a with its records (the DL-04 scope line and the DL-53
  entry among them), the 2026-09-30 correction of AM-17 item 10(c), the 2026-09-28 replacement of
  AM-17 items 11(a) and 11(b), and AM-18.
- This amendment is committed before the first KD run. As AM-18 defines it, the first KD run is the
  first real-mode training run of any distillation stage; smokes and the invariance harness are
  not runs.
- E1 seeds 42, 43 and 44 (VAL only) and the SegNeXt-B teacher of record exist.
- No E2, E3, A, F, G, SegNeXt-L, arm-teacher or other KD run has started. No E6, E7, KD or TEST
  result exists, and TEST has not been evaluated.
- Already known: everything AM-18's State block lists (the teacher's VAL curve and per-class IoU,
  its margin over E1 seed 42, E1's three VAL scores and their seed SD of 1.10 pp, B61's
  measurements, issue #11's counts). Also known when this amendment is committed: E4 or E5 VAL
  results: none; AM-18 item 1 outputs: none; the local data session's outputs outside AM-7a's
  record, each with its decision-log entry: the upstream-protocol VAL scores of the teacher of
  record (0.36319661140441895) and of E1 seed 42 (0.35182613134384155)
  (DL-08, DL-27); the strata file
  reports/strata/train_strata_v1.json (DL-63); the AM-17b item 2 re-scores and gap
  records, reports/derived/gap_val_20261003T014324Z.json (DL-64); and the AM-17 item
  1(f) table, reports/derived/perclass_gap_val_20261003T014403Z.json and .csv
  (DL-64). No
  item uses any of them.
- On 2026-10-02 the adviser was asked whether the manuscript due 2026-11-17 must contain final TEST
  results. Item 1(b) fixes how her answer selects the schedule. Her answer at the commit: none yet.
- The dates of item 1 were set from run-time estimates on record and from AM-18 item 6(a)'s window,
  not from any score: about 29–33 h for 80,000 KD iterations (AM-17b item 1(f)), and about 5,040
  steps plus 15 converted-model VAL passes for a QAT run (docs/lane_specs/part2.md, lane 6 and the
  order table).
No test is added to the Holm family. No test, threshold, family member, grid, tie rule, seed, recipe
or teacher of record changes. Items 1–4 and 6 fix how registered rules are carried out and reported.
Items 5 and 7 each change a consequence of AM-7. Item 8 names the five gates or conditions that this
amendment narrows.

Definitions used below:
- Dates. A date ends at 24:00 Asia/Manila of that day (16:00:00 UTC). A timestamp meets a date when
  it is earlier than that instant. A run's timestamps are those its trainer writes, in Unix time
  (wall_clock in the run_meta, train, VAL and run_abort rows; wall_clock_end in the run_end
  record). A record's timestamp is the time its commit reaches the remote. [Readings settled 2026-10-07 — AM-19a readings 4, 10 and 21.]
- Launched. A run is launched when its run_meta row with mode "real" exists; that row's wall_clock
  is its launch time. A launch log (a committed file) receives one entry per launch of a sweep
  candidate, with the stage, value, seed and run directory, pushed before the launch. [Reading settled 2026-10-07 — AM-19a reading 6.]
- Complete. A run is complete at a date when its trainer's end-of-run record meets that date: the
  run_end record with its checks passed for a distillation run; for an E1 run, whose trainer
  writes no run_end record, the VAL row of its final iteration, with its log's RESULT line reading
  PASS; the last epoch's record for a QAT run; the selection record (M12) for a teacher fine-tune;
  the output with its agreement check for the ARM latency measurement. Listing a run's sha256
  values in its durable evidence folder is a duty, done within 24 hours of its end. A late or
  missing listing is a deviation and never changes whether a run is complete, cut or in TEST.
  Where AM-17 item 9, AM-17b item 3 or AM-18 say "complete and hashed", this definition applies. [Extended 2026-10-07 — AM-21 item 1: an E5 or E6 run that reaches a non-finite state still writes its last epoch's record.]
- Derived artifacts. The PTQ of a complete parent, and the conversions and the AM-4a item 2
  selection of a complete QAT run, are not runs. They are made from inputs the manifest lists,
  before the TEST unlock (item 4(c)). Those of E6 seed 42 are made before the freeze, because the
  AM-3 trigger needs them.
- Ready. A run is ready when every gate that an amendment, a decision-log entry or a lane
  specification in force places before its launch is met, its code is merged and pinned, and its
  inputs are on the pod. A run that has been cut is not ready and is never launched.
- Card. One GPU that meets the run's registered hardware rule.
- Default and non-default candidates: as AM-7a items 1 and 3 (λ = 1; α = 50 at the selected λ).
- K2: the K-lane that holds the sweep selection code and the launch profiles.

1. Schedule (amends AM-17 item 9: its dates, the force of its E6 and manifest-draft dates, its rule
   for a late never-cut run and its manifest clause).
   (a) Two schedules are fixed here.
       Schedule T:
       - λ decision date (item 2): 2026-10-19;
       - α decision date (item 2): 2026-10-22, or later under item 2(a);
       - E6 seed 42 converted-model VAL score (AM-3): 2026-10-26;
       - TEST-day manifest drafted: 2026-10-30;
       - TEST manifest freeze: 2026-11-03;
       - single TEST evaluation, no later than: 2026-11-08.
       Schedule R:
       - TEST-day manifest drafted: 2026-11-30;
       - λ decision date: 2026-12-07;
       - α decision date: 2026-12-10, or later under item 2(a);
       - E6 seed 42 converted-model VAL score: 2026-12-14;
       - TEST manifest freeze: 2026-12-18;
       - single TEST evaluation, no later than: 2026-12-21.
       Schedule R keeps AM-17 item 9's dates and adds the two decision dates. Schedule T moves the
       dates so that the manuscript due 2026-11-17 can contain final TEST results. The E6 date and
       the manifest-draft date are planning dates: missing one changes nothing by itself.
   (b) Schedule T is in force from this amendment's commit. Schedule R replaces it only if, before
       the first KD run launches, one of two entries is committed to the decision log:
       - a written reply from the adviser, dated 2026-10-02 or later and recorded verbatim, that
         says in words that the manuscript due 2026-11-17 does not need final TEST results; a
         reply that accepts a complete draft with VAL results in their place is such a reply; or
       - a rejection of item 1 (header).
       Schedule R is then in force from the day of that entry. An acknowledgement, or a reply that
       is conditional or does not answer the question, changes nothing. A reply received before the
       first KD run launches is entered before that launch, and the launch checklist records that
       none is waiting. Schedule R is never replaced by Schedule T. Once the first KD run has
       launched, the schedule in force changes only by a new amendment, which lists the results
       known when it is written; no such amendment changes the decision date of a sweep whose first
       candidate has launched.
   (c) The dated deadlines in AM-17 item 9 (2026-11-30, 2026-12-14, 2026-12-18, 2026-12-21), in
       AM-17b item 3 and in AM-18 items 2(c) and 7 mean the dates of the schedule in force. In
       every cut rule "the freeze" is the original freeze date. In "no run is added to TEST after
       the freeze" and "frozen at the freeze date" it is the freeze as moved under (d). The
       following are unchanged under both schedules:
       - the cut order (item 2(g) states its one exception);
       - the never-cut list, as item 4 reads it;
       - the launch order (λ = 1 and α = 50 first);
       - the dress rehearsal before the first KD run;
       - no run is added to TEST after the freeze;
       - a single TEST evaluation, never split or repeated;
       - the MVA 2027 clause and its date (2026-12-28).
       Amended, to apply M11 to the manifest: TEST is not enumerated, counted or inspected before
       the TEST unlock, so a manifest entry that can only be computed from TEST files (the
       corrupted-TEST cache checksums, unless they are already on record) is drafted and frozen as
       its generator's pin (script, commit, library versions, parameters). Its values are computed
       in the TEST session, before any model is scored, and are recorded then.
   (d) A never-cut run that is not complete at the freeze date moves the freeze: the freeze is then
       the first later date at whose end every never-cut run is complete, and the TEST date moves by
       the same number of days. Nobody estimates the move. The delay and the runs that cause it are
       entered in the decision log on the original freeze date, the adviser is informed the same
       day, and the new freeze is entered on the day it occurs. The cut list is fixed at the
       original freeze date: a cuttable run that is not complete on that date is cut, whether or
       not the freeze moves. The analysis does not change. Under Schedule T, if the TEST date moves
       past 2026-11-17, the manuscript is submitted without final TEST results and says so.
   (e) Cards and launch order.
       - Waiting runs take a free card in this order: (1) a sweep's default candidate, or its
         repeat; (2) the other candidates of that sweep in the launch order below, a repeat taking
         its value's place; (3) the other never-cut runs, seed 42 first, in stage order; (4)
         cuttable runs in the reverse of the cut order, subject to their dependencies, and within
         one place of that order seed 43 before seed 44.
       - Launch order within a sweep: λ = 1, 0.5, 2, 0.25, 4, and α = 50, 25, 100 (the default
         first, as AM-17 item 9 requires; then the nearest values, the smaller one first). A
         non-default candidate has no gate beyond its sweep's own (for λ, the AM-7a item 5 value on
         record; for α, the AM-16 item 2 band entry): once its sweep's default has launched, it is
         ready.
       - The candidates of a sweep launch together: each non-default candidate launches on its own
         card within 6 hours of its sweep's default. One that does not is entered in the decision
         log that day, with the provider's dated evidence that no card was available and the VAL
         rows of the sweep written by then. A card held by a cuttable run is not a reason: that
         case is reported as a deviation.
       - A ready run is launched on the first card these rules give it. A launched run is never
         stopped to give its card to another run.
       This extends AM-18 item 7(a)'s card rule to every cuttable run.

2. Sweep decision dates (amends AM-17 item 9's shortfall rule and AM-7a items 1, 2, 4 and 6;
   extends AM-2 and AM-16 item 2).
   (a) Reason and dates. E3 seed 42 is trained at the selected λ, and E6 and E7 seed 42 are built
       from it. All three are never cut, and all must be complete at the freeze. AM-17 item 9's
       shortfall rule selects λ at the freeze, which leaves no time to train them and contradicts
       the same item's date for E6 seed 42's VAL score. The same holds for α: AM-16's α = 50
       fallback, applied at the freeze, leaves no time for E6 and E7, and AM-16 item 2 has no rule
       for a sweep that is only partly finished. Each sweep therefore has a decision date (item
       1(a)). The λ decision date is a calendar date. The α decision date is the later of its
       calendar date and the third day after the day the λ selection file is committed, so a late
       λ selection does not by itself cut the α sweep. Neither date moves for any other reason: not
       when a sweep starts late, not when AM-18 item 6(a)'s window or correction delays or slows
       the runs, and not when a default candidate is waited for. [Note 2026-10-07 — AM-20: no request was sent, so AM-18 item 6(a) has no window and its correction is unavailable; the dates stand.] [Reading settled 2026-10-07 — AM-19a reading 10.]
   (b) Status at the decision date. Each candidate's status is read from its own telemetry, from
       the rows whose timestamps meet the decision date. Later rows are kept, and for a non-default
       candidate they change nothing. A non-default candidate is:
       - finished, if its run_end record with its checks passed meets the date and the run passes
         every format, recipe and consistency check of the selection code. A finished status is
         final: no later fault report, repeat or missing file changes it, its score of record is
         the best VAL score in its run_end record, and it is never repeated to change a selection.
         If the selected run's checkpoint is lost, an AM-8a repeat restores the checkpoint and the
         selection is not rerun;
       - diverged, if a valid AM-7a run_abort record for a student divergence meets the date;
       - otherwise cut at the decision date, with its reason: still running; never launched;
         stopped with no finished repeat; another abort (a step-1 failure, a non-finite VAL score, a
         non-finite input or teacher output); a failed check, or a refusal by the selection's
         checks (the refusal code is the reason); or finished or aborted after the end of the
         date. A valid divergence record written after the end of the date does not change the
         status and is still listed under AM-7a item 8.
       One exception keeps a stop from removing a candidate. A non-default candidate is "on course"
       if its last row's timestamp, plus its remaining iterations times its median iter_seconds,
       plus its remaining validations times its largest val_seconds, meets the date. If a stop that
       is not the trainer's own record ends a candidate that was on course, it is not cut: its
       repeat (item 3(a)) is waited for as a default candidate is, once. If that repeat is stopped
       too, the candidate is cut. The repeat's VAL rows are reported beside the stopped run's.
       A candidate still running at the end of the date is stopped. Because the status comes from
       the timestamps, stopping it late, or letting it finish, changes nothing. [Readings settled 2026-10-07 — AM-19a readings 2, 3, 4, 5, 7, 13, 14, 17 and 20.]
   (c) Before the decision date, a candidate ends only by finishing, by its own abort (AM-7a item
       1), or by a stop that item 3(a) governs. For a grid value with no finished directory, the
       status of record is that of its latest repeat, and every earlier directory is kept and
       listed. For a non-default candidate, the cut of (b) replaces AM-7a item 4's STOP once the
       decision date has ended; the stop is still investigated and reported. In AM-7a item 1, a run
       "stopped by any other means" is handled by this item and item 3(a), not by AM-17 item 9. In
       AM-7a item 2, the selection runs "as soon as every non-diverged candidate has finished" or,
       with a cut candidate, after the end of the decision date. [Readings settled 2026-10-07 — AM-19a readings 16 and 18.]
   (d) The default candidate is never stopped, and no time limit applies to it. If it has not
       finished at the decision date, the selection waits for it, the statuses of (b) stay as they
       were at the decision date, and item 1(d) governs any delay. Its divergence follows AM-7a
       item 3 and item 7 below. Its other stops follow item 3(a) and keep AM-7a item 4's STOP. [Readings settled 2026-10-07 — AM-19a readings 11 and 16.]
   (e) The selection then runs among the finished candidates:
       - by the unchanged AM-2 rule for λ, or the AM-16 item 2 rule for α, with AM-7a's exclusions;
       - if only the default finished, it is selected as the sole finished candidate (AM-7a item 4);
       - a winner that is the smallest finished value while a cut or diverged value is smaller, or
         the largest finished value while a cut or diverged value is larger, is flagged "edge of
         the finished set", naming for each such value the rule (AM-7a or AM-19) that removed it;
       - the candidate table (AM-17 item 6) lists each cut candidate as "cut at the decision date
         (AM-19)", with its reason, its last iteration, its VAL score at every validation completed
         before the end of the date, its GPU-hours, and the selected candidate's VAL scores at the
         same validations;
       - partial scores never enter a selection (AM-7a item 2);
       - the shortfall is disclosed with the selection.
       A selection with a cut candidate never runs before the end of the decision date. [Readings settled 2026-10-07 — AM-19a readings 8, 11 and 21.]
   (f) A sweep whose candidates have all finished or diverged before its decision date is selected
       as before (AM-7a item 2), at any time; the decision date does not affect it.
   (g) Relation to the registered rules.
       - This item replaces AM-17 item 9's shortfall selection at the freeze for λ, and gives α the
         same rule. In AM-7a item 6, "is unchanged" reads "is replaced by AM-19 item 2, for λ and
         α", and the references to the freeze ("at the TEST freeze", "not finished at the freeze",
         "executed at the freeze") read as the sweep's decision date.
       - The cut order ranks runs at the freeze. A decision-date cut is outside it, and it is the
         one exception to that order: a non-default α candidate can be cut at its decision date
         while runs that the order cuts earlier are still running. In AM-16's order, "the α sweep"
         is therefore settled at its decision date and not at the freeze. This is disclosed.
       - AM-16's α = 50 fallback keeps its meaning (AM-7a item 4): if no non-default α candidate
         was launched before the end of the α decision date, the sweep is cut, E3 seed 42 is the
         α = 50 run, and no selection file is written. The decision record of (h) states the cut. [Readings settled 2026-10-07 — AM-19a readings 16 and 19.]
         Where AM-17 item 7, AM-17b item 1(f), AM-18 items 3 and 7(a) or a launcher refer to the α
         selection or the α winner, that committed record stands in for it, and run_meta records
         its sha256. If at least one non-default α candidate was launched, (e) applies and a
         selection file is written.
   (h) Decision record. A selection with a cut candidate, and an α sweep cut under (g), needs a
       decision record. It is committed and pushed within 24 hours of the later of the end of the
       decision date and the default's run_end, and its sha256 is entered in the decision log. [Readings settled 2026-10-07 — AM-19a readings 1, 7, 15, 18, 19, 20 and 21.] The
       record gives:
       - the sweep, the schedule in force with its decision-log entry, and the decision date;
       - for each grid value, its status (finished, diverged, cut) and, for a cut, the reason; the
         default's entry may read "running (item 2(d))";
       - for each grid value, every run directory in launch order, with its launch time, its last
         iteration before the end of the date, the sha256 of its run_meta and telemetry, for a
         finished run also of its best.json and best checkpoint, and the sha256 of the AM-8a report
         that precedes each repeat;
       - "never launched" for a value with no run directory.
       The selection does not trust the record. It derives every status again from the run
       directories under (b), the default's when it runs, and it refuses:
       - if a derived status differs from the record's, or a hash differs;
       - if the record is not committed, its sha256 is not in the decision log, or the decision
         date has not ended;
       - if a launch-log entry has no directory in the record, or a directory has no launch-log
         entry;
       - if a directory that is not the first of its value has no preceding report, or an earlier
         directory of a value ended in a run_end record or a valid divergence record.
       Without a record it refuses a shortfall as now (exit 3). It writes the cut candidates and
       the record's sha256 into the selection file. A cut non-default candidate whose last row is
       more than 20 minutes before the end of the date (or twice its largest val_seconds, if that
       is longer), with no report and no repeat, is marked "stopped early, unexplained", and
       Chapter 4 reports it as a deviation. These checks make the record reproducible from the run
       directories; they do not detect a fabricated telemetry file or pod clock, so the launch
       checklist records each pod's clock offset. This replaces DL-53's rule that the AM-17 item 9
       shortfall selection is implemented only in the TEST-manifest lane against a committed
       freeze record. The code is merged and pinned before the first KD run, which does not launch
       without it.

3. Stops, repeats, cut records and the TEST unlock (amends AM-8a; extends AM-16's cut rule, AM-17
   item 9 and AM-18 item 7).
   (a) Stops and repeats.
       - The operator stops a launched run only under a written rule: a sweep candidate at the end
         of its decision date (item 2(b)), a run cut at the freeze, AM-7's stage stop (AM-7a item
         3; item 7), or a rule of AM-18 item 7(a).
       - A run that stops in any other way (operator, pod, host) is repeated under AM-8a from the
         first iteration, in a new run directory, on the first card item 1(e) gives it. The report
         that AM-8a requires before the repeat states who or what stopped the run, and the VAL rows
         written by then by the run and by the other runs of its sweep, or by its comparator. A
         stop made by the operator outside those rules is reported as a deviation.
       - A run that ends in a trainer abort that is not a student divergence (a step-1 failure, a
         non-finite VAL score, a non-finite input or teacher output), or with a failed check, is
         repeated once in the same way, whatever the investigation finds. If the repeat ends the
         same way, the run is recorded as "aborted (rule, cause)" and is not repeated again: a
         non-default candidate is cut, a default candidate is a STOP that a new amendment settles,
         and a seed-43/44 run's table cell reads "aborted (rule, cause)".
       - A run whose trainer has no divergence rule (a QAT run, an E1 run, the SegNeXt-L fine-tune)
         and whose logged loss or score becomes non-finite is recorded and reported, and is
         repeated only after a documented infrastructure fault (AM-8a). AM-18 item 2 keeps its own
         rule for R1 and R2. [Extended 2026-10-07 — AM-21 item 1: an E5 or E6 run that reaches a non-finite state completes its 15 epochs.]
       This amends AM-8a in two ways: its faults include every stop of the second point, and one
       repeat follows the aborts of the third point without a documented infrastructure fault. [Readings settled 2026-10-07 — AM-19a readings 1, 9, 12, 13, 16 and 17.]
   (b) Cuts. A run is cut only:
       - at the original freeze date, if it is cuttable and not complete (AM-17 item 9; item 1(d)).
         A cuttable run that was never launched is cut there, and its record states why it was not
         launched (a gate not met, code not merged, no card) and since when;
       - at a decision date (item 2);
       - by a rule of AM-17 item 11, AM-17b or AM-18 that names its own cut, no-go or cancellation,
         whose wording stands; or
       - before its launch, by arithmetic. When a card first becomes available to a cuttable run,
         its iterations times the median seconds per iteration of the most recently finished run
         of its stage (the first KD pod's smoke if there is none) are compared with the time left
         to the original freeze date. If they exceed it, the run is cut then and is not launched;
         the operator has no choice, and the entry gives the arithmetic. A stage with no
         measurement is not cut before its launch. For an AM-18 arm this entry is its "not
         launched" record under AM-18 item 7(a).
       No other cut exists. AM-16's "if the schedule slips" is carried out by these rules only. [Reading settled 2026-10-07 — AM-19a reading 17.]
   (c) Records. Each cut is entered in the decision log on the day it is made, and always before
       TEST, with the runs, the rule, the date, the reason and the results of record known at that
       time. A cut run that had written VAL rows is reported, where its registered reading would
       appear, with its VAL curve and its comparator's VAL scores at the same iterations. In AM-16,
       "recorded here before the affected run" reads "recorded in the decision log on the day the
       cut is made, and before TEST".
   (d) TEST unlock and manifest. The unlock gate is unchanged for every lane that serves a run or a
       quantity in TEST. A lane that serves only cut runs and cut quantities is entered in the
       decision log as not required, with the cut entries it relies on, before the unlock. Every
       run that is complete at the original freeze date is in the TEST manifest, and the manifest
       validator refuses a manifest that omits one.

4. Runs whose parent is cut or diverged (extends AM-1 and AM-17 item 9).
   (a) At each seed, E4 and E5 are built from E1, and E6 and E7 from E3 (AM-1; AM-4a item 5; AM-15;
       IMPLEMENTATION_CONTRACT (b) and the two "Source ckpt" rows of B4). When a parent is cut or
       diverged, its descendants at that seed are not run. They are recorded as "not run: parent
       cut" or "not run: parent diverged", and that entry is their record under item 3(c). This is
       not a breach of the never-cut list.
   (b) AM-17 item 9's never-cut "E4 and E7 for all seeds (calibration passes only)" reads "E4 at
       every seed whose E1 run of record is complete, and E7 at every seed whose E3 run is
       complete".
   (c) E7 at seed 43 or 44 is made as soon as its E3 parent is complete. An E3 run at seed 43 or 44
       is never cut because its E7 is missing. When that E3 run is complete at the freeze, E7 at
       that seed is in TEST: the manifest lists it by its inputs (the E3 checkpoint's sha256, the
       AM-10 list's sha256, the PTQ script's commit, the image digest and the qconfig). If its
       artifact does not exist at the freeze, it is made after the freeze and before the TEST
       unlock, in the pinned image on a TRAIN/VAL-only staged root, and its sha256 is appended to
       the manifest before the unlock. The same holds for every derived artifact (Definitions).
       These are the only entries added to the manifest after the freeze. A derived artifact
       therefore never moves the freeze or the TEST date, and cannot be withheld.

5. Divergence of E2 or E3 at seed 43 or 44 (amends AM-7a item 3 and, through it, AM-7).
   (a) A divergence (AM-7a item 1) of E2 or E3 at seed 43 or 44 follows AM-7a item 3's arm rule
       (DL-04):
       - the run stops and is recorded as diverged, and its run_abort record, telemetry and partial
         checkpoints are kept and hashed;
       - it is never relaunched: not with clipping, not with the same seed, not at another seed
         (AM-8a);
       - it triggers no clipping rule and no rerun of any other run.
       Its descendants follow item 4, and it is reported under item 6.
   (b) A divergence of a default candidate at seed 42 (λ = 1; α = 50 at the selected λ) still has
       AM-7's full consequence (AM-7a item 3; item 7 below). Nothing else in AM-7a item 3 changes.
   (c) Reason. This item holds under either schedule.
       - Seeds 43 and 44 enter only AM-8 and the per-seed tables (AM-1); no test uses them.
       - They launch after the selections, when the seed-42 FP32 runs of record are complete under
         one unclipped rule, or are being completed under it.
       - AM-7's consequence would retrain E1 at three seeds, and E2 and E3, with clipping because of
         a descriptive seed.
       Item 6 reports the divergence instead: an E3 divergence makes the verdict "not stable", and
       an E2 divergence is stated with the verdict.
   (d) Chapters 4 and 5 state such a divergence as a limitation of the seed-42 results: the recipe
       of record diverged at that seed, and no clipped rerun was made.

6. Seed stability with a missing run (extends AM-8, AM-16 item 1 and AM-17 item 4).
   (a) AM-8's criterion, its metric and its thresholds are unchanged. Conditions 1 and 2 use E1 and
       E3; condition 3 uses E3 and E6. A run exists when it is complete at the original freeze
       date; every such run is in TEST (item 3(d)).
   (b) Each condition is reported as "holds", "fails" or "not evaluable":
       - condition 1 fails if the E1 → E3 gain is zero or negative at any seed where both runs
         exist; it holds if both runs exist at all three seeds and the gain is positive at each;
         otherwise it is not evaluable;
       - condition 2 is evaluated only if E1 and E3 exist at all three seeds; otherwise it is not
         evaluable. No mean or standard deviation is taken over two seeds;
       - condition 3 fails if the E3 → E6 drop is 2.0 pp or more at any seed where both runs exist;
         it holds if both runs exist at all three seeds and the drop is below 2.0 pp at each;
         otherwise it is not evaluable.
   (c) A divergence of E3 at seed 43 or 44 is a seed-stability failure, whatever (b) gives.
   (d) Verdict. It keeps AM-8's two values: "stable" only if all three conditions hold, and "not
       stable" otherwise. A verdict of "not stable" is reported with its reason: "condition n fails
       at seed k", "E3 diverged at seed k", or "not evaluable: runs missing", naming the runs and
       why they are missing. A missing run never gives "stable", and it never softens the verdict.
       Each condition is reported with its status and with its values at every seed where they
       exist.
   (e) The per-seed table lists every planned cell. A missing cell reads "cut (date, rule)",
       "diverged (iteration, rule)", "aborted (rule, cause)", "not run: parent cut" or "not run:
       parent diverged". A divergence of E2 at seed 43 or 44 is shown in the table and is stated in
       the same sentence as the verdict wherever the verdict is reported. It does not change the
       verdict, because AM-8's conditions do not use E2.
   (f) Seeds 43 and 44 still enter no test (AM-1).

7. AM-7's full consequence under a schedule (amends AM-7 and AM-7a item 3; extends AM-7a item 5 and
   the cut order). If AM-7's full consequence fires (a default candidate at seed 42, or a run to
   which a rejection restores it):
   (a) The dates of the schedule in force do not move. Item 2 applies to a rerun sweep with its
       decision date: a rerun non-default candidate launches only if its launch meets that date,
       and after it only the sweep's default candidate is rerun.
   (b) Never cut, as before: the seed-42 runs E1–E7 of the clipped recipe (item 1(d) governs a
       delay). The U4 pilot is rerun on the clipped E1 seed 42, because it is E5 seed 42. The rerun
       E1 runs at seeds 43 and 44 are cuttable; they are added at the end of the cut order, after
       E3 seeds 43/44, and item 4 applies to their E4 and E5.
   (c) Entries already recorded from the unclipped E1 runs stand, and are labelled as computed on
       the superseded recipe: the AM-16 item 2 band and its s, the AM-16 item 4 score, AM-17 items
       1(f) and 3 and the dress rehearsal of item 9, AM-17b item 2, R3, and AM-18 items 1 and 6
       with their branches. None is recomputed; the rerun E1 values are reported beside them.
   (d) Runs of the superseded unclipped recipe (finished, cut or diverged), their selection files,
       and any arm or control run under that recipe are kept, are reported as superseded and enter
       nothing. A rerun of such an arm or control is a cuttable run in its existing place in the
       cut order. "Never relaunched" (item 5), "not rerun" (AM-7a item 3) and "no rerun" (AM-18
       item 3(e)) apply within one recipe.
   (e) The rerun sweeps are selected by the same rules, with the AM-7a item 5 value as the recipe's
       clipping value, in new selection files. The code for it (the trainers' real-run gates, the
       selection code's recipe check, the new selection-file paths and the launchers that read
       them) is merged and pinned before the first rerun candidate launches.

8. Rationale. Each item settles, before any run it governs exists, a case that the registered rules
   leave open or cannot carry out:
   - a λ shortfall selected at the freeze cannot be followed, and neither can an α fallback applied
     there (item 2);
   - a stop or a cut that an operator may choose after seeing partial scores is a selection, so
     stops, repeats and cuts are tied to timestamps and written rules (items 2 and 3);
   - the never-cut "E7 for all seeds" conflicts with E3 seeds 43/44 being cut last (item 4);
   - AM-8 does not say what a missing seed means (item 6);
   - AM-7 and AM-7a do not say what a rerun does to the dates, the band or the earlier runs
     (item 7).
   The manuscript due 2026-11-17 is the occasion for Schedule T, and only scope flexes to meet it:
   no test, threshold or recipe is relaxed. Item 5's ground is 5(c), not the deadline. Five gates or
   conditions are narrowed, and each is named here:
   - AM-7a item 4's STOP becomes a cut for a non-default candidate once its decision date has
     ended (item 2(b));
   - the selection's refusal of a failed check becomes a cut in the same case (item 2(b));
   - AM-8a's "only after a documented infrastructure fault" yields to one repeat after a
     non-divergence abort (item 3(a));
   - "complete and hashed" at the freeze becomes "complete", with the hashing a duty (Definitions);
   - the TEST unlock no longer needs a lane that serves only cut runs (item 3(d)).
   The decision dates and the launch order within a sweep were set before any KD run, from run-time
   estimates and AM-18 item 6(a)'s window only.

9. Disclosure. Chapter 4 states:
   - the schedule in force, and why;
   - each decision date's effect: the candidates cut, with their reasons and partial scores, and
     the shortfall;
   - every cut, with its rule, date and reason, and every run that was never launched;
   - every stop that was not the trainer's own, every repeat and every deviation;
   - the runs not run because a parent was cut or diverged;
   - every divergence at seeds 43 and 44;
   - the AM-8 verdict, with its reason and each condition's status and values;
   - any move of the freeze, and whether item 7 applied.

Code:
- K2, in src/training/sweep_select.py, scripts/select_lambda.py, scripts/select_alpha.py and
  configs/sweep_rules.json: the decision dates of both schedules and the launch order; the launch
  log; the status at the decision date and the on-course test (item 2(b)); the decision record and
  its checks (item 2(h)); repeats listed per grid value; the cut entries of the selection file; the
  α-sweep cut of item 2(g), which the launchers that read α accept from the committed decision
  record. Merged and pinned before the first KD run.
- L-AM8: item 6's condition statuses, verdict, reason and table cells.
- The TEST-manifest lane: the cut records, the parent rule, item 4(c)'s derived-artifact entries,
  item 3(d)'s completeness check and the generator pins of item 1(c), in the manifest and its
  validator.
- Item 7(e), only if it fires.
- Items 1(b), 1(d), 3(a)–(c), 5, 6(f) and 7(a)–(d) need no code. Item 5 uses the trainer's existing
  abort. The pod runbook's launch checklist checks the decision log for a recorded divergence of
  the same stage and seed, the launch order and launch log of item 1(e), a waiting adviser reply
  (item 1(b)) and the pod's clock offset.

## AM-20 — The unsent approval request: AM-18 item 6(a) settled without a window, and the 2026-10-05 in-person approval recorded as reported (amends AM-18's header request rule and item 6(a)'s availability rule; settles a reading of AM-18 item 6(b))

Dated 2026-10-06. Status: DRAFT (group-recorded; no approval request has been sent for it).
- Fixed text. From its commit this text is changed only by a new amendment, which lists the results
  and the AM-18 item 1 outputs known when it is written. A dated bracket note only points to a later
  record or records an outcome.
- In force from the time its commit reaches the remote (AM-19 Definitions: a record's timestamp).
  Approval gates nothing here.
- What changes: the request that AM-18's header requires is not sent (item 1), and item 6(a)'s hold
  ends through item 2, not through a reply or a window. Item 4 settles how item 6(b)'s "approved by
  the adviser" is read. Nothing else in AM-18 or AM-19 changes. Their "Approval means" bullets
  stand. No test, threshold, family member, grid, tie rule, seed, recipe, schedule, date, arm rule
  or rejection rule changes, and no teacher of record. Every other gate stands.
- A rejection of this amendment is entered verbatim in the decision log on the day it arrives, and
  is reported. It never reopens item 2's entry. It changes a rule only through a new amendment,
  which lists the results and the AM-18 item 1 outputs known when it is written.

State at amendment (as of the commit):
- No execution of AM-18 item 1 has started and none of its outputs exists, so F and margin_dedup do
  not exist. The AM-17 item 1(f) table, which is the VAL part of item 1(c), is registered separately
  and exists (DL-64).
- Nothing that item 6(a) holds back has run: no part of item 1(a) on any teacher, and no smoke of
  the K8 or flip targets on real teacher weights.
- No KD or QAT run has launched, and no E2–E7 or TEST result exists. The results known are those
  that AM-19's State block lists. Nothing that bears on F has been measured since AM-18's commit.
- The request that AM-18's header requires was not sent when AM-18 was committed (7a06ef7,
  2026-10-05), and no request was sent for AM-19. On 2026-10-06 Ice De Luna reported that neither
  will be sent.
- Five committed texts therefore record what did not happen (erratum E-52):
  - DL-69: a request sent on 2026-10-05, and a window ending on 2026-10-12;
  - the Status cells of DL-66 ("adviser approval requested") and DL-70 ("adviser approval requested
    2026-10-05");
  - the Status lines of AM-18 and AM-19 ("adviser approval requested").
- On 2026-10-05 at 13:44 (Asia/Manila) Ice De Luna reported in writing: "we alr talked to adviser
  now in person about the changes and she approved all too". That report is the only record of the
  approval:
  - no second member has confirmed it, and what the adviser was shown is not recorded;
  - no written reply from the adviser exists for AM-4a, AM-7a, AM-17b, AM-18 or AM-19 (her replies
    of 2026-09-29, DL-57, answer an email about AM-17 item 11), and none has arrived before this
    commit;
  - AM-7a, AM-18 and AM-19 were committed after the report (4f7d1b0, 7a06ef7 and 376d77f; commit
    times 15:56 to 15:58 that day), so for them it can only be an approval of the changes as
    presented.

1. The request.
   - The request was not sent at AM-18's commit and is not sent (State). This is a recorded
     deviation from AM-18's header (DL-82, E-52).
   - Approval of AM-18 or of AM-19 still means exactly what its header says. A reply that meets that
     definition counts whenever it arrives, and it is entered verbatim in the decision log on the
     day it is received. With no request sent, that is a reply that names the amendment.
   - A written message that confirms the 2026-10-05 approval without naming an amendment is
     information only.
   - If the group later asks the adviser in writing to approve AM-18's arms, the request states that
     item 6(a)'s correction is unavailable (item 2).
2. AM-18 item 6(a).
   - With no request sent, item 6(a)'s seven-day window does not exist, and its hold could end only
     through a reply.
   - Item 6(a)'s state is "no approval": no written reply that approves or declines the correction
     is in the decision log, and the in-person approval of item 4 is not such a reply. This commit
     enters the state in the decision log (DL-84).
   - Nothing that item 6(a) holds back starts before this commit has reached the remote: no part of
     item 1(a) on any teacher, and no smoke of the K8 or flip targets on real teacher weights. From
     then on the hold has ended, and item 1(a) runs under AM-18's execution rules.
   - The item 1(a) output names DL-84 as its state entry. The decision-log entry that records F
     gives the time this commit reached the remote, from GitHub's activity record, and the time
     item 1(a) started.
   - The entry is final (AM-18 item 6(a)), and the correction is unavailable for every run. If
     F ≥ 0.10, the registered single-draw targets stand, F is disclosed as a limitation, and K8 is
     an arm as in the middle branch.
   - No later reply, rejection or erratum makes the correction available, and neither does an
     amendment written with F known: with F known, K = 8 targets enter only as the descriptive arm
     K8. The entry stands even if another item of this amendment is later corrected.
3. What still waits for a written approval of AM-18.
   - No arm of AM-18 items 2 and 3 launches before a written reply that approves AM-18, as its
     header defines it, is in the decision log (AM-18's header, unchanged). The same holds for
     everything else that AM-18's header does not put in force from the commit.
   - Such a reply opens the arms whenever it arrives, and never the correction (item 2). The arms it
     opens follow AM-18 item 7 and AM-19's dates and cut rules. An arm that is already cut stays
     cut.
4. The 2026-10-05 in-person approval. It is recorded verbatim, with its date and its evidence
   (DL-83).
   - "All" means AM-4a, AM-7a, AM-17b, AM-18 and AM-19 as the group presented them that day (DL-83).
     It covers no later change to those texts, and no text written after the meeting: not AM-20,
     AM-21 or AM-19a.
   - AM-4a, AM-7a and AM-17b define no form of approval, and no run, arm or selection waits for
     their approval. For them it is recorded as a reported approval: approved by the adviser in
     person, as reported by one member, with no written record from her. This follows DL-65 with
     less evidence: DL-65 has two reports and a written acknowledgement.
   - For AM-18 and AM-19 it is information only. It is not the written reply their headers define.
     It opens no arm, puts nothing in AM-18 in force, makes no correction available and selects no
     schedule (AM-19 item 1(b) needs a written reply). It is not a rejection and excepts no item.
   - From this commit on, a reported in-person approval meets no gate. Where a text makes an action
     wait for the adviser's approval and defines no form of it, approval means an explicit written
     reply from the adviser that names that text, recorded verbatim before the action. This settles
     AM-18 item 6(b)'s "approved by the adviser", before margin_dedup exists. DL-65 stands as
     recorded.
   - Whatever the adviser later writes about the report is entered verbatim on the day it is
     received. If it contradicts the report for an amendment, that amendment's markers are corrected
     by an erratum, and no run is reclassified.
5. Disclosure. Chapter 4 states:
   - that the approval requests that the headers of AM-18 and AM-19 required were not sent, and that
     five entries which said otherwise were corrected (E-52);
   - that the adviser approved AM-4a, AM-7a, AM-17b, AM-18 and AM-19 in person on 2026-10-05, as
     reported by one group member, with no written record from her, and before AM-7a, AM-18 and
     AM-19 were committed;
   - that item 6(a)'s correction was unavailable because AM-18 requires a written approval and none
     was requested, although she had approved AM-18 in person; with F's value, the branch it
     selects, and the times of the state entry and of item 1(a)'s start;
   - which AM-18 arms did not run for want of a written approval.

## AM-21 — Non-finite states in QAT runs (amends AM-4a items 2 and 3; extends AM-19 item 3(a) for E5 and E6 runs)

Dated 2026-10-06. Status: DRAFT (group-recorded; no approval request has been sent for it).
- Fixed text. From its commit this text is changed only by a new amendment, which lists the results
  known when it is written. A dated bracket note only points to a later record or records an outcome.
- In force from the time its commit reaches the remote, for every E5 and E6 run launched after it.
  No QAT run launches before then. Approval gates nothing here.
- E6-KD keeps its generic abort on a non-finite loss or gradient norm (DL-04, DL-51) and follows
  AM-19 item 3(a). Items 2, 4 and 5 below apply to it.
- A rejection is entered verbatim in the decision log on the day it arrives. It changes a rule only
  if it is recorded before the first QAT run launches; the rejected item is then settled by a new
  amendment before that launch. A rejection recorded later is reported. It changes a rule only
  through a new amendment, which lists the results known when it is written, and it never reopens a
  selection already made or a run already launched.

State at amendment (as of the commit):
- No E4, E5, E6 or E7 run has launched, no QAT or PTQ result exists, and the U4 pilot has not run.
- The results known are those that AM-19's State block lists, and no output of AM-18 item 1 exists.
- Also known, and no QAT result: the Q2 plan audit's measurements on synthetic tensors with the
  student of record. A healthy prepared student holds five never-observed fake-quants; a non-finite
  gradient makes every parameter non-finite at that step; while observers are on, the next forward
  then raises.
- The QAT code exists on a lane branch (lane/q2-qat) and has run on synthetic data only. It is
  changed to match this text before it is merged and pinned.
No test is added to the Holm family. No test, threshold, family member, grid, tie rule, seed, recipe
value, freeze point, schedule or teacher of record changes.

1. An E5 or E6 run that reaches a non-finite state completes.
   (a) The state of a QAT model is finite when all of these hold: every parameter and every
       BatchNorm buffer is finite; every fake-quant scale is finite; and every moving-average
       observer holds finite minimum and maximum values with minimum ≤ maximum in every element, or
       holds exactly (+∞, −∞) and is a never-observed module. The never-observed modules are the
       fake-quants that no forward of the prepared student calls. A forward-hook census finds them
       before training. They are listed in run_meta, are the same in every QAT run, and are compared
       with the list of record at launch (DL-85).
   (b) The trainer evaluates (a) after every step whose loss or pre-clip gradient norm is
       non-finite, whenever a step or a VAL pass raises, and at the end of every epoch, on the state
       that its checkpoint holds.
   (c) Once the state is not finite, the trainer records the step at which it found this. It then
       keeps drawing batches and stepping the schedule through all 15 epochs. An exception that a
       step or a VAL pass raises in that state is recorded and does not end the run. Each epoch
       still writes its checkpoint, its VAL row and its end-of-epoch record, each carrying the state
       flag.
   (d) The trainer never skips an update while the state is finite and never repairs a state.
   (e) The run is complete (AM-19 Definitions) when its fifteenth epoch's record exists. An
       exception raised while the state is finite still ends the run as an abort, and a process that
       dies is a stop (AM-19 item 3(a)).
   (f) A run whose state became non-finite, or whose logged loss or pre-clip gradient norm was
       non-finite, is repeated only under AM-8a, and only if its fault report is committed before
       any record conversion of its checkpoints exists (for a pilot run: of either pilot run's
       checkpoints).
   (g) A seed-42 run that AM-19 item 3(a) records as "aborted (rule, cause)" is a STOP that a new
       amendment settles, as a default candidate is.
2. Epoch selection (AM-4a item 2).
   (a) An epoch whose checkpoint state is not finite is excluded from the selection and reported,
       and the selection continues over the remaining epochs. The checkpoint decides: item 1(a) is
       applied to it again at conversion and at selection.
   (b) If no epoch is convertible, no epoch is selected for the run. The run stays complete, so it
       does not move the freeze (AM-19 item 1(d)). It is entered in the decision log that day as
       "non-finite: no model". Nothing that needs its model is computed or launched until a new
       amendment settles it (for E6 seed 42: the AM-3 trigger and E6-KD). Only this halt is
       pre-registered. That amendment is written with the run's records known, says so, and uses no
       TEST result. If none is committed before the TEST manifest freeze, the run is in the manifest
       as "non-finite: no model", and every quantity that needs its model is reported as not
       evaluable.
   (c) Every other failure of conversion or scoring refuses the selection. The selection is re-made
       only after a written fault report (AM-8a), in a new evaluation directory, and two versions
       are never chosen between.
3. U4 pilot (AM-4a item 3).
   (a) A pilot run whose state became non-finite at any step is rejected. So is a pilot run with a
       step whose logged loss or pre-clip gradient norm is non-finite, whatever its state. The
       rejection is read from the run's own records: its telemetry (the file whose sha256 was taken
       on the pod), or an epoch checkpoint that fails item 1(a). A checkpoint that fails with no
       state flag in the telemetry rejects the run too, and is reported as a deviation of the
       trainer.
   (b) If one run is rejected, the other is the winner, provided it is complete and has a selected
       epoch.
   (c) If both are rejected, there is no winner, and no later QAT run launches until a new amendment
       decides the clip value. Only this halt is pre-registered. That amendment is written with both
       runs' records known, says so, and uses no TEST result.
   (d) A rejected run is still converted and scored as item 2(a) reads, and its records are kept.
       Its scores are reported and select nothing. Item 2(b) does not apply to it. If its conversion
       or scoring fails, item 2(c) applies to it alone, and the other run's win does not wait.
   (e) A rejection is lifted in one case only: a repeat under item 1(f), which then stands in the
       rejected run's place. If the repeat is rejected under (a) too, that clip value is rejected
       for good.
   (f) Nothing else rejects a pilot run. The registered pilot rule (configs/quant.py at 569cfbb, to
       which PREREGISTRATION U4 points; IMPLEMENTATION_CONTRACT restates it under "Remaining
       real-run values") rejects "a candidate whose training becomes non-finite or numerically
       unstable". Its first ground is (a). Its second ground has no registered definition and is
       not applied.
   (g) Otherwise AM-4a item 3 applies as written.
4. AM-4a in force. AM-4a (committed at 826d851; item 1 reworded at 73fd4d7 the same day), as this
   amendment amends it, governs every QAT run. No run is reclassified by the adviser's approval of
   AM-4a (DL-83), by its absence or by a later rejection. A rejection of AM-4a follows this
   amendment's own rejection rule (header).
5. Disclosure. Chapter 4 reports every excluded epoch, every rejected pilot run, every run with no
   convertible epoch, and the step at which each non-finite state was first recorded.

## AM-19a — Readings of AM-19 for the KD sweeps (settles readings that AM-19's Definitions and items 2 and 3 leave open)

Dated 2026-10-06. Status: DRAFT (group-recorded; no approval request has been sent for it).
- Fixed text. From its commit this text is changed only by a new amendment, which lists the results
  known when it is written. A dated bracket note only points to a later record or records an outcome.
- In force from the time its commit reaches the remote. No KD run launches before then. Approval
  gates nothing here.
- What it does. AM-19's header requires a new amendment for a reading that its text leaves open, and
  this is that amendment. No reading adds a cut, a selection or a way for a candidate to win, and
  none moves a date. Where two clauses of AM-19 conflict, or AM-19 names no rule for a case, the
  stricter reading is taken. Five readings say in words that they go beyond the text: readings 2, 5
  and 13 narrow a clause of item 2(b), reading 6 widens a record, and reading 19 adds how a void
  decision record is replaced.
- A refusal that no later input can lift is a STOP that a new amendment settles, with the results
  known then.
- A rejection is entered verbatim in the decision log on the day it arrives. It changes a reading
  only if it is recorded before the first KD run launches. The rejected reading is then settled by a
  new amendment before that launch.

State at amendment (as of the commit): no KD run has launched; no λ or α candidate, KD result, E4–E7
result or TEST result exists. The results known are those that AM-19's State block lists, and no
output of AM-18 item 1 exists. The selection code exists on a lane branch (lane/k2-kd-launch,
11afa15) and has run on synthetic run directories only. It implements these readings except readings
9, 10, 17, 18, 19 and 21, where this text differs from it or adds a step. The code and the runbook
are changed to match before the code is merged and pinned.

Terms. C is the instant at which a decision date ends (AM-19 Definitions). An attempt is one launch
line of a grid value in the launch log, with its run directory once it is launched. A value's
attempts are taken in the order of their launch lines; reading 12 makes that the order of their
launches.

1. Ended (item 2(h)). "Ended in a run_end record" means a run_end record with its checks passed. A
   run_end with a failed check is item 3(a)'s failed check, and item 3(a) governs its repeat.
2. What can cut (item 2(b)). Item 2(b) cuts a non-default candidate for "a refusal by the
   selection's checks", and it also makes a finished status final against a later fault or a missing
   file. A file that can change after C cannot serve both clauses, so this reading narrows the
   first.
   - Only checks on a run's run_meta and telemetry can make a candidate cut at its decision date.
   - A fault in its best.json file or its checkpoint (a file that is present and is unreadable, or
     disagrees with the run_end record) refuses the selection at any time and is never a cut.
   - A missing best.json or checkpoint changes nothing (item 2(b)). A file that is removed after it
     refused a selection is reported as a deviation.
3. On course (item 2(b)). The on-course test uses only the stopped attempt's own rows.
   - The remaining validations are the validation points (every multiple of the VAL interval, and
     the last iteration) that have no VAL row.
   - An attempt is not on course when an input of the formula is missing or is not a finite number:
     it has no train row or no VAL row; a row has no finite timestamp; an iter_seconds or
     val_seconds value is not finite; or an iteration is not an integer.
4. Timestamps (Definitions; item 2(b)). In a non-default attempt, a timestamp that steps backward is
   a fault only when it crosses C: a row earlier than C that follows, in file order, a row at or
   after C. That is a fault of the attempt's telemetry (reading 2). Every other backstep is counted
   and recorded. For an α run the test is made once the α decision date is derived (reading 10).
5. Identical recipe (item 2(b)). The candidates of a sweep share four values: num_workers, and the
   sha256 of the teacher's checkpoint, configuration and model configuration (DL-53). A difference
   among the finished and diverged candidates refuses the selection at any time. It is never
   attributed to one candidate and is never a cut. Item 2(b) does not say whose fault a difference
   is, so this reading narrows its "refusal by the selection's checks" for this check.
6. Launch log (Definitions).
   - The launch log covers every KD launch, not only a sweep candidate's. This widens the record;
     the schedule check needs it.
   - It holds launch lines (pushed before the launch), launched lines, not_launched lines and
     stopped lines.
   - A start is "not launched" when it wrote no run_meta row with mode real, whatever its return
     code. It is not a run: the attempt after it is not a repeat and needs no AM-8a report.
7. On-course repeat (items 2(b) and 2(h)).
   - The on-course exception is used once per value, by its first on-course stop. The repeat is
     read as a default candidate is, with no time limit: the value is finished if the repeat
     finishes and diverged if it diverges. A later stop, abort or failed check of the repeat is a
     cut.
   - While the repeat runs, the decision record lists it as "running (on-course repeat, item 2(b))"
     and is still written within item 2(h)'s 24 hours.
   - The record is not written before the repeat has launched and the stop has its stopped line. If
     that makes the record late, the delay is reported as a deviation from item 2(h).
   - The entry stays valid whatever status the repeat later reaches. The selection derives that
     status from the repeat's own rows.
8. The candidate table (item 2(e)). A cut candidate's GPU-hours run from its launch to its last row
   and are reported both up to C and in total. For a value with more than one attempt the table
   lists each attempt, and "its last iteration" is the largest over them.
9. Ends the same way (item 3(a)). An attempt ends the same way when an earlier attempt of the same
   value ended with the same (rule, cause), or when it and an earlier attempt both ended with a
   failed check. Attempts between the two change nothing, a stopped attempt included. An ending that
   differs from every earlier ending of the value gets its own single repeat.
10. The λ selection's commit day (item 2(a); Definitions, a record's timestamp).
    - D is the Asia/Manila day on which the commit that added the λ selection file reached the
      remote, on any ref. The α decision date is the later of its calendar date and D + 3 days.
    - Two records bound that instant: the latest telemetry timestamp the λ selection read, less 60
      seconds, and the launch time in the α default's first launched line, plus 60 seconds. If the
      upper bound's day + 3 days is no later than the calendar date, the calendar date stands and D
      is not needed. If both bounds fall on one day, that day is D.
    - Otherwise (the bounds fall on different days, or the α default has no launched line, so there
      is no upper bound) D comes from GitHub's activity record for the repository, saved verbatim
      and committed. D is the day of the earliest push, on any ref, of a commit that is or descends
      from the commit that added the file. The record must reach back past the lower bound, and that
      push must lie inside the bounds; otherwise the selection refuses.
    - A branch deletion pushes no commit and is not counted.
    - Whether each push after the lower bound carries the file is settled once, when the record is
      saved, and that list is committed with it. The launch gate and the selection use the list, and
      check it wherever the clone holds the pushed commit. A pushed commit that the remote no longer
      serves when the record is saved counts as carrying the file. The remote's answer is saved with
      the list, and the entry is named in the selection file and in Chapter 4.
    - No local clock is the record.
11. The default in the finished set (items 2(d) and 2(e)). The selection runs only on a finished set
    that contains the sweep's default candidate, and refuses a set without it. A diverged default is
    item 7's case.
12. Order of attempts (item 3(a)). A repeat launches after the attempt before it has ended. An
    attempt whose launch time is not later than the largest timestamp of the previous launched
    attempt refuses the selection; it is never a cut.
13. A non-finite row with no abort record (items 2(b) and 3(a)). The trainer flags a train row that
    holds a non-finite value, and a VAL row whose all-class mIoU is non-finite, just before it
    aborts. An attempt with such a row and no run_abort record after it ends with (rule, cause) =
    (non-finite row, abort record missing), whatever follows the row.
    - It is not a divergence: no valid abort record exists (AM-7a).
    - It is not treated as a stop either, because its own rows show why it ended. The on-course
      exception does not apply. This narrows item 2(b)'s "a stop that is not the trainer's own
      record".
    - It is repeated once under item 3(a).
14. A failed check at or after C (item 2(b)). A run_end with a failed check written at or after C
    gives the cut reason "aborted after the end of the date".
15. Stopped early with no row (item 2(h)). For "stopped early, unexplained", an attempt with no row
    is measured from its launch time.
16. The default's abort (items 2(c), 2(d), 2(g) and 3(a)). While the latest attempt of a default
    candidate is one that ended in an abort or a failed check, the selection refuses, on item 2(g)'s
    α-cut path too. The default is repeated under item 3(a). From the repeat's launch, its status is
    the default's (item 2(c)).
17. Ending the same way before C (items 2(b), 3(a) and 3(b)).
    - When a non-default value's latest attempt ends the same way (reading 9), the value is not
      repeated again.
    - Item 3(a) says that it "is cut", and item 3(b) allows a cut only at a decision date. So before
      C the selection waits for C (exit 3).
    - At C the value is cut, with the reason "aborted (rule, cause)".
18. Refusal before waiting (items 2(c) and 2(h)).
    - Before C, the selection gives every refusal that the complete launch log would also give
      before it waits for a missing record.
    - It waits instead of refusing only while an earlier attempt of a value is incomplete and could
      still make that value's latest ending its second same ending (readings 9 and 17).
19. A decision record and later launch lines (items 2(g) and 2(h)).
    - Item 2(h) sets a deadline for the record, and it lets the default's entry read "running (item
      2(d))", so the record may be written while the default runs. Both stand. The 24 hours count
      from the later of the end of the decision date and the default's last ending: its run_end with
      its checks passed, or its valid divergence record.
    - The record states the launch log it was written from (its line count and sha256), and item
      2(h)'s directory checks compare the record with those lines. A launch line of the default
      that is added later does not void the record: the selection derives the default's status and
      directories when it runs (item 2(h)), and it checks them against the launch log.
    - A launch line of a non-default value of the same sweep that is added after the record refuses
      the selection.
    - The record's decision date and instant must equal the derived ones. The way the α date was
      derived (reading 10) may differ once the α default has launched, as long as the date is the
      same.
    - AM-19 has no rule for a record that the selection's checks refuse for another reason. Such a
      record is void. It is kept and never edited. A corrected record is written once, to a new
      file, after a written fault report in AM-8a's form, and both sha256 values are entered in the
      decision log. This adds a step to item 2(h).
20. A launched attempt without its run directory (items 2(b) and 2(h)). A launched attempt whose run
    directory is not supplied is never "never launched" and never a cut. Before C the selection
    waits; from C it refuses. The same holds for a launch line that has neither a launched nor a
    not_launched line.
21. The end of a decision date (Definitions; items 2(e) and 2(h)). The selection and the record
    writer take the present time from the machine that runs them. Before a selection with a cut, and
    before a record is written, that machine's offset from a network time source is recorded; an
    offset above 60 seconds is a STOP. A decision record, or a selection file with a cut, whose
    commit reached the remote before C is void.
22. Disclosure. Chapter 4 lists these readings with AM-19 and names the five that go beyond its
    text.

## Status of PREREGISTRATION §10 items after these amendments

| Item | Status |
|---|---|
| U1 | mechanism completed by AM-2; the value is still selected by the sweep |
| U2 | AM-3 |
| U3 | withdrawn by AM-7 (AM-7a pre-commits the fallback value) |
| U4 | unchanged (AM-4 fixes only its future scoring) [Superseded by AM-4a item 3: the pilot is two full E5 seed-42 runs, clipped at 1.0 and 5.0, scored on converted-model VAL mIoU.] [AM-21 item 3, 2026-10-07: a pilot run that becomes non-finite (state, logged loss or pre-clip gradient norm) is rejected; the registered rule's "numerically unstable" is not applied.] |
| U5 | AM-1 |
| U6 | unchanged |
| U7 | unchanged (D22 deferred) |
| U8 | stale; hardware is logged at run time |
| U9 | superseded by M4 (B61) |
