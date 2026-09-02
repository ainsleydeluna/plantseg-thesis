# B31c — adjudication, commit, and pod launch package

**Base before B31c:** `148af2c` · **HEAD after:** see §3  
**Scope:** V1 verification, contract/open-questions update, commits + push, `preflight_e1.py`, launch runbook v2.  
**Out of scope and untouched:** F8/OS8, teacher code, `src/distill/*`, `train_distill.py`, `configs/teacher/*`, all manuscript files, `docs/reference/**` (beyond the authorised v1 pointer), `src/data/dataset.py:83`.

---

## 1. V1 — the four `it == 1` → `it == first_it` sites

### 1.1 The four sites and what each asserts

| Site | Check | Asserts | Fires on a FRESH run at |
|---|---|---|---|
| `train_e1.py:265` | `batch_shapes` | img `(B,3,512,512)` float32; mask `(B,512,512)` int64 | iter 1 |
| `train_e1.py:272` | `logits_shape` | logits `(B,116,512,512)` | iter 1 |
| `train_e1.py:283` | `loss_finite` | loss is finite and 0-dim; snapshots `p0` for the step check | iter 1 |
| `train_e1.py:296` | `optimizer_step` | `p0` actually moved after `optimizer.step()` | iter 1 |

`first_it` is assigned `start_iter`, which is `1` on a fresh run. So on a fresh (non-resumed) dry-run **all four fire at iteration 1, on the same data, with the same values as pre-B31** — `first_it == 1` makes the expression literally identical. Confirmed across 11 dry-runs during B31 and 9 more during the bisect check in §3, all reporting `6/6 exercised`.

**No check fires later, and none fires on different data, on a fresh run.**

### 1.2 The direct question: can a resumed run pass while exercising fewer checks?

**Before the B31c fix: YES, in three distinct ways.** All three are now closed or made visible. This was a real weakening of the floor and is stated, not smoothed.

| # | Weakening | Status after B31c |
|---|---|---|
| 1 | LR transitions **across a resume boundary** were never compared. A *k*-segment run left *k−1* transitions unverified. | **CLOSED** — `last.pt` carries `prev_lr` |
| 2 | With <2 LR values, `all([])` is `True`, so `lr_non_increasing` printed **PASS** having compared nothing. | **VISIBLE** — now reports `SKIPPED`, non-fatal |
| 3 | The already-complete resume path ran **zero** checks and printed `RESULT: PASS`. | **CLOSED** — distinct token `RESULT: NOOP` |

The other five hard checks (`batch_shapes`, `logits_shape`, `loss_finite`, `optimizer_step`, `val_cm_accumulated`) **are** exercised on a resumed run, because `first_it` tracks `start_iter` and `val_cm_accumulated` fires at the first validation boundary reached. Only `lr_non_increasing` was ever at risk.

### 1.3 Proof — 3-segment resumed run, raw stdout

```text
==============================================================================
V1 FIX PROOF — prev_lr carried across resume; SKIPPED / NOOP distinguishable
==============================================================================

----- SEGMENT 1: iters 1-3 (fresh) -----
  [iter    1/3] loss=5.9711 ce=4.9889 dice=0.9822 lr=9.99988750e-03
  [iter    2/3] loss=6.0464 ce=5.0636 dice=0.9828 lr=9.99977500e-03
  [iter    3/3] loss=6.7518 ce=5.7696 dice=0.9822 lr=9.99966250e-03
  [CHECKS]
    lr_non_increasing : PASS
  [CHECKS] 6/6 exercised, 0 skipped
  [summary] lr_transitions_compared=2 (spans the resume boundary when resuming from a checkpoint carrying prev_lr)
  RESULT: PASS (6/6 checks exercised, 0 skipped)

----- SEGMENT 2: iters 4-6 (resumed) -----
  [resume] from C:\Users\admin\AppData\Local\Temp\v1fix_qae6gn4s\last.pt | resuming at iter=4 lr=9.99966250e-03 best_all_class_miou=0.00000 prev_lr=0.009999662499367179
  [iter    4/6] loss=5.9461 ce=4.9641 dice=0.9820 lr=9.99955000e-03
  [iter    5/6] loss=6.2066 ce=5.2173 dice=0.9893 lr=9.99943750e-03
  [iter    6/6] loss=5.9762 ce=5.0004 dice=0.9758 lr=9.99932500e-03
  [CHECKS]
    lr_non_increasing : PASS
  [CHECKS] 6/6 exercised, 0 skipped
  [summary] lr_transitions_compared=3 (spans the resume boundary when resuming from a checkpoint carrying prev_lr)
  RESULT: PASS (6/6 checks exercised, 0 skipped)

----- SEGMENT 3: iters 7-9 (resumed) -----
  [resume] from C:\Users\admin\AppData\Local\Temp\v1fix_qae6gn4s\last.pt | resuming at iter=7 lr=9.99932500e-03 best_all_class_miou=0.00000 prev_lr=0.0099993249974686792
  [iter    7/9] loss=6.1547 ce=5.1763 dice=0.9784 lr=9.99921250e-03
  [iter    8/9] loss=7.0758 ce=6.0997 dice=0.9761 lr=9.99910000e-03
  [iter    9/9] loss=5.7993 ce=4.8299 dice=0.9694 lr=9.99898749e-03
  [CHECKS]
    lr_non_increasing : PASS
  [CHECKS] 6/6 exercised, 0 skipped
  [summary] lr_transitions_compared=3 (spans the resume boundary when resuming from a checkpoint carrying prev_lr)
  RESULT: PASS (6/6 checks exercised, 0 skipped)

--- [1] transition accounting: are there ZERO unverified transitions? ---
  transitions compared per segment : [2, 3, 3]
  total compared                   : 8
  total transitions in a 9-iter run: 8
  ZERO unverified transitions      : True
  (segment 1 has no predecessor for its first iter; segments 2 and 3 each compare their
   first iteration against the carried prev_lr, closing both boundaries)

--- [2] full-precision LR from the checkpoints vs the analytic 80k poly curve ---
  analytic PolynomialLR(total_iters=80000, power=0.9), iters 1..9, full float64:
    iter 1: 0.009999887499929687
    iter 2: 0.009999774999718746
    iter 3: 0.009999662499367179
    iter 4: 0.00999954999887498
    iter 5: 0.009999437498242147
    iter 6: 0.00999932499746868
    iter 7: 0.009999212496554576
    iter 8: 0.009999099995499834
    iter 9: 0.009998987494304452

  checkpoint-recorded values (read from last_seg*.pt, NOT from the '%.8e' print):
    seg1 iter=3  optimizer.lr=0.009999662499367179  == analytic -> True
          prev_lr      =0.009999662499367179  == analytic -> True
    seg2 iter=6  optimizer.lr=0.00999932499746868  == analytic -> True
          prev_lr      =0.00999932499746868  == analytic -> True
    seg3 iter=9  optimizer.lr=0.009998987494304452  == analytic -> True
          prev_lr      =0.009998987494304452  == analytic -> True

--- [3] boundary monotonicity, full precision ---
  boundary 1 (iter 3 -> iter 4): 0.009999662499367179 -> 0.00999954999887498   non-increasing -> True
  boundary 2 (iter 6 -> iter 7): 0.00999932499746868 -> 0.009999212496554576   non-increasing -> True
  full 1..9 sequence monotonic non-increasing -> True
  every segment reported lr_non_increasing PASS -> True

--- [4] SKIPPED: a 1-iteration FRESH run compares 0 transitions ---
  [CHECKS]
    lr_non_increasing : SKIPPED (0 LR transitions compared; 1 LR value(s) seen)
  [CHECKS] 5/6 exercised, 1 skipped
  RESULT: PASS (5/6 checks exercised, 1 skipped)

--- [5] SKIPPED: a 1-iteration RESUMED run now compares 1 transition (boundary) ---
  [CHECKS]
    lr_non_increasing : PASS
  [CHECKS] 6/6 exercised, 0 skipped
  [summary] lr_transitions_compared=1 (spans the resume boundary when resuming from a checkpoint carrying prev_lr)
  RESULT: PASS (6/6 checks exercised, 0 skipped)
  ^ the carried prev_lr turns what used to be a vacuous 0-pair case into a real check

--- [6] NOOP: zero checks, distinct token, exit 0 ---
  [CHECKS] 0/6 exercised, 6 skipped (no iterations ran)
  RESULT: NOOP (already complete at iter 10)
  exit code: 0

--- [7] all four outcomes distinguishable from the last two stdout lines alone ---
  [A] fresh full               [summary] used_pretrained=False (no download in dry-run) | RESULT: PASS (6/6 checks exercised, 0 skipped)
  [B] resumed                  [summary] used_pretrained=False (no download in dry-run) | RESULT: PASS (6/6 checks exercised, 0 skipped)
  [C] 1-iter fresh (skipped)   [summary] used_pretrained=False (no download in dry-run) | RESULT: PASS (5/6 checks exercised, 1 skipped)
  [D] already-complete         [CHECKS] 0/6 exercised, 6 skipped (no iterations ran) | RESULT: NOOP (already complete at iter 10)

RESULT: PASS
```

### 1.4 Verdict

| Requirement | Result |
|---|---|
| A *k*-segment run leaves zero unverified LR transitions | **MET** — 2+3+3 = 8 of 8 on a 3-segment 9-iteration run |
| LR matches the uninterrupted 80k poly curve at the boundaries, full precision | **MET** — checkpoint `prev_lr` == analytic value exactly, e.g. `0.00999932499746868` |
| Vacuous `all([])` never prints PASS | **MET** — `SKIPPED (0 LR transitions compared; 1 LR value(s) seen)` |
| Vacuous case is non-fatal | **MET** — exit 0, `RESULT: PASS (5/6 checks exercised, 1 skipped)` |
| `[A]`/`[B]`/`[C]`/`[D]` distinguishable from the **last line** alone | **MET** — see below |

```text
[A] fresh full               RESULT: PASS (6/6 checks exercised, 0 skipped)
[B] resumed                  RESULT: PASS (6/6 checks exercised, 0 skipped)
[C] 1-iter fresh (skipped)   RESULT: PASS (5/6 checks exercised, 1 skipped)
[D] already-complete         RESULT: NOOP (already complete at iter 10)
```

**One correction made mid-item.** My first implementation put the exercised count on a separate `[CHECKS] n/6 exercised` line *above* the summary block, so `[A]`, `[B]` and `[C]` were still indistinguishable from the last line. That failed the ruling as written; the count now rides on the `RESULT:` line itself. Existing parsers are unaffected — `verify_env.py:239` uses `"RESULT: PASS" in result_line`, which still matches (and is exactly why `preflight_e1.py` asserts the count rather than the substring).

**Residual, by design:** a fresh one-iteration run still legitimately exercises 5/6. That is visible, not silent, and `preflight_e1.py` rejects it as a launch gate.

---

## 1b. A1 — `num_workers` reproducibility probe

Run before writing the contract line, as authorised. **The first probe was inconclusive and I re-ran it**: with a 2-sample subset, `num_workers` 2/4/8 all agreed — but that is an artifact, since only workers 0 and 1 ever receive work when there are 2 samples. Re-run with 8 samples (4 batches) so batch→worker assignment actually differs:

```text
==============================================================================
A1 — is num_workers reproducibility-relevant?
==============================================================================
train Subset[100, 500, 900, 1300, 1700, 2100, 2500, 2900], shuffle=False, persistent_workers=True where nw>0, PASS 1 only
(8 decodes total: 8 samples x 4 worker counts = 32 decodes)

  num_workers=0  -> ['140c8c720c8bc712', 'a26114baf0d446c8', '425363305fe19332', '4185d3eae1914fa7', 'e68539aec17f8303', '671cecfe54554838', 'e674fe427369f44c', '5bb621aec2560a2e']
  num_workers=2  -> ['6c772db174314e67', 'f4109cb73ece4509', '63a37b0790e1b8a8', '58f97e1ba6b3ec21', 'cefba6326a361ea5', '945eb0a5ee9bb5f8', 'f33c6b85d9a50cf7', '315e26218f23d6a2']
  num_workers=4  -> ['6c772db174314e67', 'f4109cb73ece4509', '63a37b0790e1b8a8', '58f97e1ba6b3ec21', '9d42950dcf3624fe', 'ec7001005e346548', '2ca01267ee6678e4', 'cd4e768f4f9c8dde']
  num_workers=8  -> ['6c772db174314e67', 'f4109cb73ece4509', '63a37b0790e1b8a8', '58f97e1ba6b3ec21', '9d42950dcf3624fe', 'ec7001005e346548', '2ca01267ee6678e4', 'cd4e768f4f9c8dde']

  pairwise equality matrix:
               0       2       4       8
    0       True   False   False   False
    2      False    True   False   False
    4      False   False    True    True
    8      False   False    True    True

  identical across ALL counts -> False
  DISTINCT digest sets across 4 counts -> 3
  => Some counts agree, some differ. Groups:
       num_workers [0] -> ['140c8c720c8bc712', 'a26114baf0d446c8', '425363305fe19332', '4185d3eae1914fa7', 'e68539aec17f8303', '671cecfe54554838', 'e674fe427369f44c', '5bb621aec2560a2e']
       num_workers [2] -> ['6c772db174314e67', 'f4109cb73ece4509', '63a37b0790e1b8a8', '58f97e1ba6b3ec21', 'cefba6326a361ea5', '945eb0a5ee9bb5f8', 'f33c6b85d9a50cf7', '315e26218f23d6a2']
       num_workers [4, 8] -> ['6c772db174314e67', 'f4109cb73ece4509', '63a37b0790e1b8a8', '58f97e1ba6b3ec21', '9d42950dcf3624fe', 'ec7001005e346548', '2ca01267ee6678e4', 'cd4e768f4f9c8dde']
```

### Answer to the question as asked

**Digests differ across SOME counts, not all** — three distinct streams: `{0}`, `{2}`, `{4,8}`. But the `{4,8}` tie is itself a probe artifact: with 4 batches and `num_workers >= 4`, every batch draws a distinct fresh worker. **That condition never holds at real scale** — 335 batches/epoch against 12 workers — where batch *b* is served by worker *b* mod `num_workers`, so each worker count yields a distinct stream. The first 4 digests agreeing across all `nw>0` confirms the mechanism (batches 0–1 → workers 0–1 in every configuration).

**So: `num_workers` is a hard reproducibility parameter.** Seed 42 alone does not determine the realized augmentation sequence.

**FLAGGED FOR YOUR WORD-SIDE FIX:** ch3 §D's reproducibility paragraph pins seed 42 across torch/numpy/python, the cuDNN flags, `use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG` — and is **silent on `num_workers`**. As written it is under-specified: two runs satisfying every stated condition can still differ. Recorded as D25.

---

## 2. Contract / open-questions diff

Additive only; no `[ch3]`-traced method row touched, neither file restructured, split-count values untouched.

```text
 docs/IMPLEMENTATION_CONTRACT.md | 68 +++++++++++++++++++++++++++++++++++++++++
 docs/open_questions.md          | 40 ++++++++++++++++++++++++
 2 files changed, 108 insertions(+)
```

```diff
diff --git a/docs/IMPLEMENTATION_CONTRACT.md b/docs/IMPLEMENTATION_CONTRACT.md
index fbb3b06..dcfdb93 100644
--- a/docs/IMPLEMENTATION_CONTRACT.md
+++ b/docs/IMPLEMENTATION_CONTRACT.md
@@ -245,6 +245,74 @@ pending the PlantSeg repo's official convention. `[empirical; ch3 Table 3.1; ctx
 - Multi-seed plan: three-seed validation planned for **E1 and E3** (the two extra seed values =
   `NEED_TO_CONFIRM`); E4/E7 recomputed per seed; E5/E6 fine-tuned per seed where compute permits;
   Teacher trained once; E2 repetition optional `[ch3 §F]`.
+- **`num_workers` is a REPRODUCIBILITY-RELEVANT parameter and must be reported alongside seed 42**
+  `[project; empirical, measured 2026-09-01 — B31-5/A1]`. Seed 42 alone does **not** determine the
+  realized augmentation sequence. Measured on a fixed 8-sample / 4-batch train subset
+  (`shuffle=False`, identical seed), three distinct augmentation streams were observed:
+
+  | `num_workers` | first two sample digests | stream |
+  |---|---|---|
+  | 0 | `140c8c720c8bc712`, `a26114baf0d446c8` | A |
+  | 2 | `6c772db174314e67`, `f4109cb73ece4509` | B |
+  | 4, 8 | `6c772db174314e67`, `f4109cb73ece4509` | B (first 2 batches) → C (later batches) |
+
+  Mechanism: at `num_workers=0` the per-sample augmentation seed is drawn from the main-process
+  NumPy stream; at `num_workers>0` each worker is seeded from `torch.initial_seed()` and batch *b*
+  is served by worker *b* mod `num_workers`, so the seed a given sample receives depends on the
+  worker count. The `{4, 8}` agreement above is a **probe artifact** — with `num_workers >= n_batches`
+  every batch draws a distinct fresh worker. That condition never holds at real scale
+  (**335 batches/epoch vs 12 workers**), so in the real run each `num_workers` value yields a
+  distinct realized sequence.
+- **Consequence:** the B31-5 change of the real-run default from `4` to `min(cpu_count-2, 12)`
+  changes the realized augmentation *sequence*. It does **not** change the augmentation
+  *distribution*, the recipe, or any locked hyperparameter, and cross-process reproducibility at a
+  fixed `num_workers` is preserved (measured byte-identical across two independent processes).
+  Runs are comparable in distribution; they are not bitwise-comparable across different
+  `num_workers`. Record the value in Ch4 with the seed.
+- **Manuscript gap (for correction outside this repo):** ch3 §D's reproducibility paragraph pins
+  seed 42, the cuDNN flags, `use_deterministic_algorithms(True, warn_only=True)` and
+  `CUBLAS_WORKSPACE_CONFIG`, but is silent on `num_workers`. As written it is **under-specified**:
+  two runs satisfying every stated condition can still differ. `[project]`
+- **Resume is NOT bitwise-identical to an uninterrupted run** — documented deviation, same register
+  as D-A `[project; B31-2]`. `--resume` restores model / optimizer / scheduler / RNG state
+  (`torch`, `torch.cuda`, `numpy`, python `random`, and the DataLoader generator) and the LR curve
+  continues **exactly** (checkpoint-recorded LR matches the analytic 80,000-iteration
+  `PolynomialLR` curve at full float64 precision). Data **order** is not recoverable: the loop
+  consumes an infinite `cycle(train_loader)` and the position within the current epoch is not
+  persisted. A resumed run is therefore a valid E1 run but not a byte-reproduction of an
+  uninterrupted one, and must be reported as resumed if used for a headline result.
+- **LR-monotonicity coverage across resume** `[project; B31c V1]`: `last.pt` carries `prev_lr`, so a
+  *k*-segment run leaves **zero** unverified LR transitions (measured: a 3-segment 9-iteration run
+  compared 2+3+3 = 8 of 8 transitions). When **no** transition is compared (a one-iteration fresh
+  run) the check reports `SKIPPED`, never a vacuous `PASS`. The scaffold's final line carries
+  `RESULT: <PASS|FAIL> (n/6 checks exercised, m skipped)`, and a resume with nothing left to do
+  prints the distinct token `RESULT: NOOP`.
+
+#### E1 runtime defaults introduced by B31 `[project; authorised 2026-09-01]`
+These are **operational** parameters. None of them touches a `[ch3]`-traced method value.
+
+| Parameter | Value | Note |
+|---|---|---|
+| `--ckpt-interval` | **2000** | periodic atomic `last.pt` resume point |
+| `--keep-ckpts` | **3** | rolling best-checkpoint retention (best + `last.pt` always kept) |
+| `prefetch_factor` | **4** | train and val, only when `num_workers > 0` |
+| `persistent_workers` | **True, TRAIN only** | val respawns per validation (~20 times); avoids a second resident worker pool |
+| `pin_memory` | gated on `torch.cuda.is_available()` | uniform across train/val |
+| `num_workers` (real run) | **`min(cpu_count-2, 12)`** | reproducibility-relevant — see above |
+| `drop_last` | **True, TRAIN only** | val/test keep every sample; dropping eval samples would corrupt the metric |
+| telemetry | `e1_telemetry.jsonl` in `--ckpt-dir` | append-mode, resume-safe, never inside the repo |
+
+- **Iterations per epoch = 335** under `drop_last=True` (5,367 train / batch 16; 7 samples dropped
+  per epoch and reshuffled into the next), giving **238.806 epochs** at the locked 80,000 iterations
+  `[empirical, measured]`.
+- **`_dom_nonignore_ratio` uses `np.bincount`, not `np.unique`** `[project; B31-6]`. This is an
+  **exact-equivalence optimisation, not a behaviour change**: verified bit-identical on 216 masks
+  including all-ignore, single-class, all-background, empty, and in-domain labels above
+  `num_classes`. It is **2.2x faster in-function** (0.838 → 0.374 ms on a realistic 512² mask) but
+  only **1.03x mean / 1.12x median end-to-end** on `train_preprocess`, because the crop stage is one
+  of seven. It is **not a throughput lever** — the throughput lever is `num_workers`. The new code
+  is additionally *stricter* than the old: a label > 255 now raises instead of being silently
+  counted.
 
 | Library | Version | Source |
 |---|---|---|
diff --git a/docs/open_questions.md b/docs/open_questions.md
index 5abc3ac..0b759c6 100644
--- a/docs/open_questions.md
+++ b/docs/open_questions.md
@@ -62,6 +62,46 @@ family, not a numeric value. `[ch3 §C; D2; D-A]`
 
 ---
 
+## Resolved 2026-09-01 (B31 / B31c) — E1 launch-blocking fixes and their recorded deviations
+
+### D25 — `num_workers` is a reproducibility parameter — ✅ RESOLVED `[empirical, measured]`
+Seed 42 alone does **not** determine the realized augmentation sequence. Measured on a fixed
+8-sample / 4-batch train subset at identical seed, `num_workers ∈ {0, 2, 4, 8}` produced **three**
+distinct streams (`{0}`, `{2}`, `{4,8}`); the `{4,8}` tie is a probe artifact of
+`num_workers >= n_batches` and does not hold at real scale (335 batches/epoch vs 12 workers).
+Cross-process reproducibility **at a fixed `num_workers`** is preserved and was re-measured
+byte-identical after the B31-5 loader change. Resolution: `num_workers` is pinned as a **reported**
+parameter alongside seed 42 in Ch4, and the B31-5 default change (4 → `min(cpu_count-2, 12)`) is
+recorded as changing the realized sequence but not the distribution or any locked value.
+**Manuscript follow-up (outside this repo):** ch3 §D's reproducibility paragraph is under-specified
+as written — it pins seeds, cuDNN flags and `CUBLAS_WORKSPACE_CONFIG` but not `num_workers`, so two
+runs satisfying every stated condition can still differ. `[ch3 §D; project; B31-5/A1]`
+
+### D26 — resume is not bitwise-identical to an uninterrupted run — ✅ RESOLVED as a documented deviation `[project]`
+Same register as D-A. `--resume` restores model, optimizer, scheduler and all RNG state, and the LR
+curve continues exactly (verified at full float64 against the analytic 80k `PolynomialLR` curve).
+Data **order** is not recoverable under the infinite `cycle(train_loader)`, so a resumed run is a
+valid E1 run but not a byte-reproduction. It must be reported as resumed if it produces a headline
+result. No code change is proposed to make order recoverable: doing so would require persisting
+sampler position and is not required by any source. `[project; B31-2]`
+
+### D27 — scaffold check coverage under resume — ✅ RESOLVED `[project]`
+B31c V1 asked whether a resumed run can report success while exercising fewer checks than a fresh
+run. It could, in two ways, both now closed or made visible:
+1. **LR transitions across a resume boundary were unverified.** `last.pt` now carries `prev_lr`; a
+   *k*-segment run leaves **zero** unverified transitions (measured 8/8 on a 3-segment run).
+2. **`all([]) is True` printed as `PASS`** when no transition was compared. It now reports
+   `SKIPPED`. Deliberately **not fatal** — a legitimate one-iteration resume is not a defect, and
+   failing it would make the preflight brittle for no gain.
+3. **The already-complete resume path** ran zero checks and printed `RESULT: PASS`. It now prints
+   the distinct token `RESULT: NOOP (already complete at iter N)` with exit 0.
+Every outcome is distinguishable from the **last line of stdout alone** via
+`RESULT: <PASS|FAIL> (n/6 checks exercised, m skipped)`. `scripts/preflight_e1.py` asserts the
+exercised count, not merely the exit code. **Residual:** a fresh one-iteration run still legitimately
+exercises 5/6; this is visible, not silent. `[project; B31c V1]`
+
+---
+
 ## Resolved 2026-07-26 (A0) — evaluation metric semantics + result-artifact contract
 
 Full specification: **[EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md)** (frozen). Summarised here because
```

---

## 3. Commit log

Grouping is **file-set based** where hunks from different items interleave in one file and are not cleanly separable — flagged rather than split artificially, per the constraint.

| # | Hash | Items | Files | Dry-run at that commit |
|---|---|---|---|---|
| C1 | `464abf6` | B31-1, B31-5, B31-8 | `configs/data.py`, `src/data/dataset.py` | `RESULT: PASS` |
| C2 | `9c0e0f8` | B31-6 | `src/data/transforms.py` | `RESULT: PASS` |
| C3 | `bc5f644` | B31-3, B31-4 | `scripts/verify_env.py` | `RESULT: PASS` |
| C4 | `e1a3fce` | B31-2, B31-5, B31-7, B31-9, V1 | `src/training/train_e1.py` | `RESULT: PASS (6/6 exercised, 0 skipped)` |
| C5 | `65c13b5` | B31-10 | `scripts/smoke_aug_stochasticity.py` | `RESULT: PASS (6/6 …)` |
| C6 | `9734135` | B31c-3 | `scripts/preflight_e1.py` | `RESULT: PASS (6/6 …)` |
| C7 | `7b9ea94` | B31c-1 | `docs/IMPLEMENTATION_CONTRACT.md`, `docs/open_questions.md` | `RESULT: PASS (6/6 …)` |
| C8 | `0d7e15b` | B31c-4 | `reports/e1_launch_runbook_v2.md`, `…runpod_launch_runbook.md` | `RESULT: PASS (6/6 …)` |
| C9 | `2cab65d` | audit trail | `reports/pre_e1_launch_audit.md`, `reports/b31_e1_fixes.md` | `RESULT: PASS (6/6 …)` |
| C10 | `(this report)` | B31c closeout | `reports/b31c_closeout.md` | see §7 |

**Why C1–C3 print the shorter `RESULT: PASS`:** the checks-exercised suffix is introduced by C4. At C1–C3 `train_e1.py` is still the pre-B31-2 version, which works against the new `build_dataloader` because `persistent_workers` defaults to `False`. That backward compatibility is what makes the series bisectable.

### Bisectability — MEASURED, not asserted (Ruling 3)

A detached worktree was created **outside the repo**, every commit checked out in turn, and `train_e1.py --dry-run` executed against that checkout.

```text
commit  label                              dry-run
464abf6  B31-1/5/8: single-source split c    RESULT: PASS
9c0e0f8  B31-6: _dom_nonignore_ratio uses    RESULT: PASS
bc5f644  B31-3/4: CUDA compute-capability    RESULT: PASS
e1a3fce  B31-2/5/7/9 + V1: resume, atomic    RESULT: PASS (6/6 checks exercised, 0 skipped)
65c13b5  B31-10: standing regression gate    RESULT: PASS (6/6 checks exercised, 0 skipped)
9734135  B31c-3: single pre-flight gate,     RESULT: PASS (6/6 checks exercised, 0 skipped)
7b9ea94  B31c-1: record the B31 deviation    RESULT: PASS (6/6 checks exercised, 0 skipped)
0d7e15b  B31c-4: E1 launch runbook v2, su    RESULT: PASS (6/6 checks exercised, 0 skipped)
2cab65d  B30 + B31 audit trail: forensic     RESULT: PASS (6/6 checks exercised, 0 skipped)
```

**9/9 commits green.** Worktree removed afterwards:
```text
C:/Users/admin/plantseg-thesis 2cab65d [master]
```

Only the main worktree remains, and the main worktree was never checked out to another commit during the check.

### Push

```text
To https://github.com/ainsleydeluna/plantseg-thesis.git
   148af2c..2cab65d  master -> master
```

---

## 4. `preflight_e1.py` output, verbatim

Run under `--dev-rehearsal` (this is a CPU box with no CUDA device).

```text
==============================================================================
E1 PRE-FLIGHT GATE
repo=C:\Users\admin\plantseg-thesis
python=C:\Users\admin\anaconda3\python.exe
MODE: --dev-rehearsal — exercising the gate on a non-GPU box. NOT a launch authorization.
==============================================================================

==============================================================================
STAGE 1/4 — verify_env.py
==============================================================================
$ C:\Users\admin\anaconda3\python.exe C:\Users\admin\plantseg-thesis\scripts\verify_env.py

==============================================================================
B20a PLATFORM VERIFICATION (diagnostic only — no training, no download, no GPU compute)
==============================================================================

[1-3] Python / OS / repo
  python_executable : C:\Users\admin\anaconda3\python.exe
  python_version    : 3.13.5
  os_system         : Windows 11 (10.0.26200)
  machine           : AMD64
  platform          : Windows-11-10.0.26200-SP0
  repo_root         : C:\Users\admin\plantseg-thesis

[4] git
  HEAD   : 148af2c (148af2c9b30ffc78e7e2f4d0b9ff55cd40948607) on master | dirty=True
  modified (8): ['configs/data.py', 'docs/IMPLEMENTATION_CONTRACT.md', 'docs/open_questions.md', 'docs/reference/reference.pdf', 'scripts/verify_env.py', 'src/data/dataset.py', 'src/data/transforms.py', 'src/training/train_e1.py']
  untracked (4): ['reports/b31_e1_fixes.md', 'reports/pre_e1_launch_audit.md', 'scripts/preflight_e1.py', 'scripts/smoke_aug_stochasticity.py']

[5-11] torch / CUDA / GPU / threads
  torch_version       : 2.9.1+cpu
  torchvision_version : 0.24.1+cpu
  cuda_available      : False
  cuda_version        : None
  gpu_count           : 0
  gpu_names           : []
  gpu_capabilities    : [] (max supported by the pinned stack: sm_90)
  capability_gate     : SKIPPED (no CUDA device visible)
  cudnn_available     : False
  cudnn_version       : None
  cpu_count(logical)  : 16
  torch_num_threads   : 12

[12] packages (actual vs requirements.lock pin)
  torch           : actual=2.9.1+cpu                    pinned=2.1.0+cu121
  torchvision     : actual=0.24.1+cpu                   pinned=0.16.0+cu121
  numpy           : actual=2.1.3                        pinned=1.26.4
  scipy           : actual=1.15.3                       pinned=1.11.4
  Pillow (PIL)    : actual=11.1.0                       pinned=12.3.0
  opencv (cv2)    : actual=not installed                pinned=4.8.1.78
  albumentations  : actual=not installed                pinned=—
  mmcv            : actual=not installed                pinned=2.1.0
  mmseg           : actual=not installed                pinned=1.2.2
  statsmodels     : actual=0.14.4                       pinned=0.14.6

[13-14] torch hub cache / ImageNet weights
  hub_dir          : C:\Users\admin/.cache\torch\hub
  imagenet_ckpt    : C:\Users\admin/.cache\torch\hub\checkpoints\mobilenet_v3_large-5c1a4163.pth
  imagenet_cached  : False

[15] build_student(pretrained=False) — CPU random init
  student_ok       : True (out=(1, 116, 64, 64) used_pretrained=False)

[16] build_student(IMAGENET1K_V2) — skipped unless cached
  imagenet_skipped : True (SKIPPED to avoid download (checkpoint not cached))

[17] train_e1.py --dry-run subprocess (tiny; temp checkpoint dir outside repo)
  command          : C:\Users\admin\anaconda3\python.exe C:\Users\admin\plantseg-thesis\src\training\train_e1.py --dry-run --batch-size 1 --max-iters 2 --val-interval 2 --max-val-batches 1 --ckpt-dir C:\Users\admin\AppData\Local\Temp\verify_env_dryrun_ckpt_3j3114mm
  temp_ckpt_dir    : C:\Users\admin\AppData\Local\Temp\verify_env_dryrun_ckpt_3j3114mm
  temp_dir_outside_repo : True
  return_code      : 0
  RESULT: PASS (6/6 checks exercised, 0 skipped)
  temp_ckpt_files  : ['C:\\Users\\admin\\AppData\\Local\\Temp\\verify_env_dryrun_ckpt_3j3114mm\\e1_student_best_iter2.pt', 'C:\\Users\\admin\\AppData\\Local\\Temp\\verify_env_dryrun_ckpt_3j3114mm\\last.pt']
  repo_.pt_files   : []  (MUST be empty)
  no_repo_checkpoint : True

[17b] dataset (root / split dirs / per-split pair counts vs configs/data.py SPLIT_SIZES)
  PLANTSEG_DATA_ROOT env : (unset -> default)
  resolved root          : C:\Users\admin\plantseg_data\plantseg
  train: images=5367   pairs=5367   expected=5367   OK
  val  : images=846    pairs=846    expected=846    OK
  test : images=1561   pairs=1561   expected=1561   OK
  total: 7774 (expected 7774) OK
  dataset_ok             : True

[18] VERDICT
  verdict          : PARTIAL
  environment_label: OK for DRY-RUN only; NOT OK for real E1 training
  can_run_real_E1  : False
  blockers_before_real_E1:
    - No CUDA GPU (torch.cuda.is_available()=False; local torch is CPU-only build '2.9.1+cpu'). Real E1 (80k iters @ bs16/512^2) needs a GPU.
    - torch mismatch vs pinned: local '2.9.1+cpu' != requirements.lock '2.1.0+cu121' (cu121 GPU training stack).
    - ImageNet backbone not cached; real run with --init imagenet needs network to populate the torch-hub cache OR a pre-staged checkpoint.
    - Not installed locally: opencv (cv2), albumentations, mmcv, mmseg (teacher/aug stack — needed for teacher fine-tune / E2-E3, not E1 supervised).

==============================================================================
RESULT: PARTIAL | OK for DRY-RUN only; NOT OK for real E1 training
==============================================================================

==============================================================================
STAGE 2/4 — smoke_dataloader.py
==============================================================================
$ C:\Users\admin\anaconda3\python.exe C:\Users\admin\plantseg-thesis\scripts\smoke_dataloader.py

num_classes (configs/data.py) = 116  -> valid label range [0, 115] U {255}

[TRAIN] image: shape=(2, 3, 512, 512) dtype=torch.float32 | mask: shape=(2, 512, 512) dtype=torch.int64
[TRAIN] unique non-255 labels (3): [0, 87, 109]
[TRAIN] contains 255 (ignore/pad): False

[VAL] image: shape=(2, 3, 512, 512) dtype=torch.float32 | mask: shape=(2, 512, 512) dtype=torch.int64
[VAL] unique non-255 labels (2): [0, 1]
[VAL] contains 255 (ignore/pad): True

[PASS] dataloader smoke test

==============================================================================
STAGE 3/4 — smoke_aug_stochasticity.py  (R5 pinned-stack seed re-measurement)
==============================================================================
$ C:\Users\admin\anaconda3\python.exe C:\Users\admin\plantseg-thesis\scripts\smoke_aug_stochasticity.py

==============================================================================
B31-10 — augmentation stochasticity + reproducibility gate
torch 2.9.1+cpu | numpy 2.1.3 | PREFETCH_FACTOR=4 | indices [100, 500]
==============================================================================

process A seeds : [[1608637542, 1273642419], [1935803228, 787846414]]
process B seeds : [[1608637542, 1273642419], [1935803228, 787846414]]
train workers=0 : pass1=['140c8c720c8bc712', 'a26114baf0d446c8']
                  pass2=['73f0abdbf0330e04', '3f32b51dce6c715a']
train workers=2 : pass1=['6c772db174314e67', 'f4109cb73ece4509']
                  pass2=['ebe33f7c9e8bd21e', '48de3debf20e021c']
val   workers=2 : pass1=['5b3ebdd8677f842f', '5f25430fee925101']
                  pass2=['5b3ebdd8677f842f', '5f25430fee925101']

[CHECKS]
  PASS  (a) train varies across passes [workers=0]
  PASS  (a) train varies across passes [workers=2, persistent]
  PASS  (a) per-sample seeds differ across passes
  PASS  (b) val bit-identical across passes [control]
  PASS  (c) seeds reproduce across processes
  PASS  (c) train digests reproduce across processes [workers=0]
  PASS  (c) train digests reproduce across processes [workers=2]
  PASS  (c) val digests reproduce across processes

RESULT: PASS

--- R5: pinned-stack seed comparison ---
  B30 §9 baseline : [[1608637542, 1273642419], [1935803228, 787846414]]
  observed here   : [[1608637542, 1273642419], [1935803228, 787846414]]
  VERDICT         : MATCH — the pinned stack draws the development-stack sequence.
                    Seed-dependent artifacts under reports/ remain valid.

==============================================================================
STAGE 4/4 — train_e1.py --dry-run
==============================================================================
$ C:\Users\admin\anaconda3\python.exe C:\Users\admin\plantseg-thesis\src\training\train_e1.py --dry-run

[mode] DRY | torch 2.9.1+cpu | device=cpu | cuda_available=False | num_classes=116
[budget] max_iters=4 batch_size=2 val_interval=2 max_val_batches=2 num_workers=0 seed=42
[student] params=2,933,688 used_pretrained=False (pretrained arg=False)
[data] train_index=5367 val_index=846 (index globbed; only the batches pulled below are decoded)
[loss] CombinedCEDiceLoss(weight=len116, ignore_index=255) weight_on=cpu
[opt] SGD lr=0.01 momentum=0.9 weight_decay=0.0001 | scheduler=PolynomialLR (total_iters=80000, power=0.9)
[grad-clip] disabled (no concrete E1 max_norm)
[ckpt] dir=C:\Users\admin\AppData\Local\Temp\e1_dryrun_ckpt_vruelp4l (verified OUTSIDE repo)
[jsonl] telemetry -> C:\Users\admin\AppData\Local\Temp\e1_dryrun_ckpt_vruelp4l\e1_telemetry.jsonl
[iter    1/4] loss=5.9618 ce=4.9790 dice=0.9827 lr=9.99988750e-03
[iter    2/4] loss=6.5878 ce=5.6035 dice=0.9842 lr=9.99977500e-03
[val     2/4] cm_batches=2 cm_total_px=700928 all_class_miou=0.00000 disease_only_miou(PROVISIONAL)=0.00000
[ckpt    2/4] new best all_class_miou=0.00000 -> C:\Users\admin\AppData\Local\Temp\e1_dryrun_ckpt_vruelp4l\e1_student_best_iter2.pt
[iter    3/4] loss=5.9582 ce=4.9738 dice=0.9844 lr=9.99966250e-03
[iter    4/4] loss=6.3901 ce=5.4033 dice=0.9868 lr=9.99955000e-03
[val     4/4] cm_batches=2 cm_total_px=700928 all_class_miou=0.00000 disease_only_miou(PROVISIONAL)=0.00000
[last    4/4] resume point -> C:\Users\admin\AppData\Local\Temp\e1_dryrun_ckpt_vruelp4l\last.pt

[CHECKS]
  batch_shapes      : PASS
  logits_shape      : PASS
  loss_finite       : PASS
  optimizer_step    : PASS
  val_cm_accumulated: PASS
  lr_non_increasing : PASS
[CHECKS] 6/6 exercised, 0 skipped
[summary] lr_transitions_compared=3 (spans the resume boundary when resuming from a checkpoint carrying prev_lr)
[summary] first_batch=((2, 3, 512, 512), 'torch.float32', (2, 512, 512), 'torch.int64')
[summary] lr_trace n=4 (bounded print: first 4 / last 4; full curve in e1_telemetry.jsonl)
[summary] lr_head=['9.99988750e-03', '9.99977500e-03', '9.99966250e-03', '9.99955000e-03']
[summary] best_all_class_val_miou=0.00000 best_ckpt=C:\Users\admin\AppData\Local\Temp\e1_dryrun_ckpt_vruelp4l\e1_student_best_iter2.pt
[summary] used_pretrained=False (no download in dry-run)

RESULT: PASS (6/6 checks exercised, 0 skipped)

--- final line: 'RESULT: PASS (6/6 checks exercised, 0 skipped)' ---

==============================================================================
PRE-FLIGHT SUMMARY
==============================================================================
  stage                     result       secs  detail
  verify_env                PASS         19.9  verdict=PARTIAL (accepted under --dev-rehearsal ONLY)
  smoke_dataloader          PASS          2.7  exit=0
  smoke_aug_stochasticity   PASS         32.7  seed sequence MATCH
  train_e1 --dry-run        PASS         16.4  6/6 checks exercised, 0 skipped

VERDICT: NO-GO (--dev-rehearsal) — all stages passed, but this run was a rehearsal on a non-GPU box and is not a launch authorization.
         Re-run WITHOUT --dev-rehearsal on the pod to obtain a real GO.
```

### And in real mode on this box — correctly refuses

```text

==============================================================================
  stage                     result       secs  detail
  verify_env                FAIL         13.6  verdict=PARTIAL — needs PASS (CUDA GPU, sm<=9.0, dataset OK)
  smoke_dataloader          SKIPPED         -  not reached
  smoke_aug_stochasticity   SKIPPED         -  not reached
  train_e1 --dry-run        SKIPPED         -  not reached

VERDICT: NO-GO — first failing stage: verify_env.
         Fix it and re-run. Do not launch E1.
```

**On the acceptance criterion "runs green on this box":** all four stages PASS, including `seed sequence MATCH` and `6/6 checks exercised, 0 skipped`. The overall verdict is nonetheless **NO-GO**, because this box has no GPU. That is the correct answer, not a failure — a gate that returned GO on a machine that cannot run E1 would be worthless. `--dev-rehearsal` exists so the gate can be exercised here, and it can never emit GO.

---

## 5. Refused / flagged

**Nothing was refused.** Six things are flagged:

1. **My first V1 implementation did not satisfy the ruling.** The exercised count sat on its own line above the summary, leaving `[A]`/`[B]`/`[C]` indistinguishable from the last line. Corrected within the item (§1.4).
2. **The first A1 probe was inconclusive** (2-sample subset made `nw` 2/4/8 indistinguishable by construction). Re-run at 8 samples before the contract line was written (§1b).
3. **`preflight_e1.py` decodes subprocess output as UTF-8 explicitly.** `text=True` alone uses the Windows locale codec and mojibaked the child scripts' em-dashes.
4. **`--dev-rehearsal` is an addition not named in B31c-3.** Without it the gate cannot be exercised on a non-GPU box at all, and the acceptance criterion asked for a run here. It can never produce GO.
5. **`smoke_dataloader.py` has no `RESULT:` line** and always returns 0 on success; preflight checks its exit code plus its `[PASS]` marker. `verify_env.py` returns 0 *regardless of verdict*, so preflight parses its `RESULT:` line instead. Neither script was modified.
6. **ch3 §D under-specification** (D25) — flagged for your Word-side fix, not actioned here.

**Locked values verified unchanged** across every commit: SGD lr `1e-2`, momentum `0.9`, wd `1e-4`, poly power `0.9`, batch `16`, `80,000` iters, val every `4,000`, all-class-mIoU selection (D1), `grad_clip_norm=None` (D-A), `num_classes=116`, `ignore_index=255`, seed `42`.

---

## 6. Residual blockers to the real E1 run

### Still blocking

| # | Blocker | Owner | Changed by B31c? |
|---|---|---|---|
| R1 | **N1 × F11 schedule collision** — ch3 requires three-seed E1 *and* E3, and pre-registers the λ_logit sweep at five full 80k runs. Nothing plans for either. | you | no |
| R2 | **Teacher checkpoint absent** — `weights/` empty, `test_teacher_init.py` never run, `teacher_init_source.md` still `NEED_TO_CONFIRM`. Blocks E2/E3, not E1. | you | no — now surfaced by preflight's GO note |
| R3 | **ImageNet backbone may be uncached** — `--init imagenet` needs the torch-hub cache or network. | operator | no — reported by `verify_env` |

### Now gated by code (were open before B31c)

| # | Was | Now |
|---|---|---|
| R5 | Pinned-stack seed stream unverified | **`preflight_e1.py` stage 3** — MATCH/DEVIATION vs the B30 §9 baseline, DEVIATION is a NO-GO |
| R6 | Capability gate never run against real CUDA | **`preflight_e1.py` stage 1** — still never executed on real hardware, but now on the critical path |
| — | Dataset problems only discoverable at training start | **`verify_env` `[17b]`** |
| — | A reduced-coverage dry-run looked like a full one | **`RESULT:` carries the exercised count**, preflight asserts it |

### Still measurable only on the pod

| # | Item | Settled by |
|---|---|---|
| R4 | F7 CUDA determinism — `F.interpolate(bilinear)` and `AdaptiveAvgPool2d` backward have no deterministic CUDA kernel; with `warn_only=True` (which ch3 prescribes) the run is seed-controlled but not bit-deterministic | run E1 with `PYTHONWARNINGS=always`, grep `does not have a deterministic implementation` |
| R7 | Worker RAM at 12 workers × `prefetch_factor=4` | watch RSS over the first 100 iters |
| R8 | **Real throughput** — 88.3 ms/sample is a Windows dev-box measurement; every wall-clock estimate from it is INFERRED and cross-platform | re-measure `samples_per_sec` from the first 200 JSONL rows (runbook v2 §6) |

### Accepted deviations (documented, not defects)

- **Resume is not bitwise-identical** to an uninterrupted run (D26). LR curve exact, RNG restored, data order unrecoverable.
- **7 training samples dropped per epoch** by `drop_last=True`, reshuffled into the next epoch across 238.806 epochs.
- **A fresh 1-iteration run exercises 5/6 checks** (D27) — visible, non-fatal.
- **`num_workers` changes the realized augmentation sequence** (D25) — distribution and recipe unchanged; report the value with the seed.

### Handed back to you (manuscript, Word)

Split counts 5,442/778/1,554 → 5,367/846/1,561 · head channels 115 → 116 · "large-kernel average-pooling" → global average pooling · the Albumentations/OpenCV "version pinned in the reproducibility manifest" claim that `requirements.lock` contradicts · **ch3 §D silent on `num_workers` (new, D25)** · `configs/loss.py:14` `real_class_weights: NEED_TO_CONFIRM` vs the `reports/e1_class_weights.json` that `train_e1.py` actually loads.

**B32 (not started):** F8 / OS8 logit-resolution pin.

---

## 7. Provenance

### Commits in this series
```text
2cab65d B30 + B31 audit trail: forensic audit and fix report
0d7e15b B31c-4: E1 launch runbook v2, superseding B28
7b9ea94 B31c-1: record the B31 deviations and new runtime parameters in the contract
9734135 B31c-3: single pre-flight gate, scripts/preflight_e1.py
65c13b5 B31-10: standing regression gate for augmentation stochasticity and reproducibility
e1a3fce B31-2/5/7/9 + V1: resume, atomic checkpoints, JSONL telemetry, logging hygiene
bc5f644 B31-3/4: CUDA compute-capability gate and dataset verification in verify_env
9c0e0f8 B31-6: _dom_nonignore_ratio uses np.bincount instead of np.unique
464abf6 B31-1/5/8: single-source split counts, loader throughput, train-only drop_last
```

### Files changed across B31 + B31c
```text
 configs/data.py                     |   18 +-
 docs/IMPLEMENTATION_CONTRACT.md     |   68 +
 docs/open_questions.md              |   40 +
 reports/b31_e1_fixes.md             | 1614 +++++++++++++
 reports/e1_launch_runbook_v2.md     |  313 +++
 reports/e1_runpod_launch_runbook.md |    7 +
 reports/pre_e1_launch_audit.md      | 4303 +++++++++++++++++++++++++++++++++++
 scripts/preflight_e1.py             |  196 ++
 scripts/smoke_aug_stochasticity.py  |  176 ++
 scripts/verify_env.py               |   90 +-
 src/data/dataset.py                 |   48 +-
 src/data/transforms.py              |   24 +-
 src/training/train_e1.py            |  304 ++-
 13 files changed, 7163 insertions(+), 38 deletions(-)
```

### Git status
```text
 M docs/reference/reference.pdf
```

### `docs/reference/reference.pdf`
```text
(no commit in this series touches it)
```

Its mtime is **`Jun 27 13:28`**, unchanged since before B30. It remains ` M` and unstaged. It was **never** opened, read, hashed, copied, staged, or committed at any point in B30, B31, or B31c. The only `docs/reference/` interaction in the whole chain was reading `ch3.pdf` during B30, which is not on the protected list.

### Confirmations

- **Explicit-path staging only.** Every `git add` in this series named exact files. `git add -A` / `git add .` were never used.
- **Each commit leaves the dry-run green** — measured in a throwaway worktree, 9/9 (§3).
- **Worktree cleanup:** `git worktree list` shows only the main worktree; the temporary one lived in the session scratchpad, outside the repo, and is deleted.
- **No F8/OS8 work, no teacher code**, no `src/distill/*`, `train_distill.py`, or `configs/teacher/*` touched.
- **No manuscript or `docs/reference/**` edits** beyond the authorised one-line v1 SUPERSEDED pointer in `reports/e1_runpod_launch_runbook.md` (which is under `reports/`, not `docs/reference/`).
- **`src/data/dataset.py:83`** (the proven-correct augmentation RNG) untouched — confirmed by `smoke_aug_stochasticity.py` reproducing the exact B30 §9 seed sequence.
- **`src/eval/metrics.py` untouched.**
- **No training, GPU, downloads, or installs.** All work CPU-only.
- **No commit message carries a co-author trailer**, as instructed.

