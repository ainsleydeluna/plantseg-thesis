# Lane report — S4 · MDE, band, censuses (`lane/s4-mde-band`)

Session S4 "MDE, band, censuses (cloud)", Phase B, Thu 8 Oct 2026 (UTC); Opus 5.5 at max effort (the GO's line 2).
Tags as in the earlier lane reports: MEASURED (run here), REPOSITORY-PROVEN (file:line at the lane's last code
commit), DOCUMENTED (the GO, the plan or a record), INFERRED.

## 1. GO summary (DOCUMENTED)

- **GO.** Phase B GO (orchestrator, Thu 8 Oct 2026). The Phase A plan `s4_phase_a_plan_20261008.md`, sha256
  `b1a23815c0844f1343ce2b2f38750267984a250f927a6a5480aa51755d415e63`, passed its Tier F audit "approve with
  corrections". Annex A is the audit verbatim. The plan, as patched by Annex A and the rulings R1–R11, is
  the specification.
- **Base.** master = claude/keen-curie-u4a8ig = `31b53b70d0617e4c17959db8e63235b7b75ae011`, the K2 merge (F1). The
  lane branches from it by PB-1a (R1).
- **Writes.** Only the plan's C1–C3 paths: `src/stats/mde.py`, `scripts/mde_entry.py`, `scripts/smoke_mde.py`,
  `scripts/split_censuses.py`, `scripts/smoke_split_censuses.py` and this report. No record is written (R10).
  No output of record is made (R11); synthetic outputs stay under `/tmp/s4scratch`.
- **Never changed:** docs/DECISION_LOG.md, docs/PREREGISTRATION_AMENDMENTS.md, the contracts, docs/lane_specs/,
  requirements*.lock, .claude/, AGENTS.md, CLAUDE.md, configs/, reports/derived/, the frozen files and the K2
  files (SCOPE). §13 item 3 verifies this.

## 2. Patch list as applied (STEP 0; DOCUMENTED, REPOSITORY-PROVEN)

There is one row per Annex A §3 bullet and one per ruling that changes code or tests. Function names refer
to the files at the last code commit.

| # | Annex A §3 bullet / ruling | Applied in | Checked by |
|---|---|---|---|
| A1 | §0/§5/§8: base `31b53b7`; "pass" = the same exit code and counts as at `31b53b7`; acceptance 3 reads `git diff --stat 31b53b7..HEAD -- . <X>` | PB-1a (§3.1); baselines at `31b53b7` (§3.3) | §13 |
| A2 | §1.2: no `td.base_document` / `td.environment_block` / `td.finish_output`; own header and environment blocks | `mde_entry.assemble` (own keys; `software_environment` from src/stats/artifact.py); `split_censuses.header`, `code_block`, `environment_block` (own functions). None records vars(args), a hostname, a CPU model, start/end times or wall seconds | C4a–b, A7, T7 (byte-identical reruns) |
| A3 | §1.2: smoke-side guard on the temp root | `smoke_mde.main`, `smoke_split_censuses.main`: abort (exit 2) unless `realpath(mkdtemp(...))` holds no "test"; every run sets TMPDIR=/tmp/s4scratch (R7) | every run |
| A4 | §1.3 step 6: `--expect-manifest-sha256-42/43/44` (optional; mismatch exits 2; results in provenance_checks); no seed-43/44 checkpoint pin (R2) | `mde_entry.check_flags` (format), `check_manifest_pin`, `provenance_checks`, `require_passed` | C2 (three pins recorded and passed), R12 |
| A5 | §1.3 P2: `read_json(p, code="best_json_format")`; `sorted(doc) == BEST_JSON_KEYS`; floor from `load_rules()["alpha_cwd"]["band"]`; `s, band, _trace = dl27_band(...)`; gate the value equality, record the basename comparison | `read_best_json`, `band_floor`, `band_inputs` (records `best_ckpt_basename_matches_checkpoint_path`), `band_self_check` | B1–B6, R4f, R10; M17, M18 |
| A6 | §1.4: `min_count(b) = -(-4*b // 5)` pinned by U0; `pair_summary` records m and n; the resampling record | `mde.min_count`, `mde.pair_summary`; the entry's `resampling` block | U0, U3a–d, S2, C2 |
| A7 | §1.5 P4: `--expect-records-sha256`; `records_sha256`, `expected_sha256`, `expected_source` "DL-68, docs/DECISION_LOG.md:127"; mIoU_full recorded, not gated | `split_censuses.check_flags`, `teacher_suffix` | T1b, T12 |
| A8 | §2.1 row :396-397 and §3 (ii): the reasons for the extensions | each provenance check's name states its reason: `check_image_digest`, `check_inputs_on_model_device`, `check_policy_applied` | C2, R4c |
| A9 | §2.5 (xvi): the config's "BaseSegDataset sorts by img_path" replaces "[U] pass order" | the split_censuses docstring and the census's `val_manifest.rule` | (text) |
| A10 | §3 (iv): common random numbers; the literal per-(pair, δ) `default_rng(42)` draws the identical matrix | the mde.py docstring; the entry's `resampling.equivalent_to` and `guard_points_share_rows`; the DL line | U8, U8b |
| A11 | §3 (xv), §4.1 D: the erratum; D1a with dz from literals; D2 expected MDE_W 0.009, ratio 1.065, powers at 0.008–0.010 recorded; D1e; D2b | `smoke_mde.make_d1`, `make_d2`, D1a–e, D2, D2b | D group |
| A12 | §4.1 new cases: U0, U8b, R1 over all seven path flags, R8, R9, R10, S1, S2, R11; C fixtures with the DL-17 pins incl. all_class_miou and gt_support | `smoke_mde` (`dress`, `setup_fixtures`, `make_r1`, ...) | as named |
| A13 | §4.2 new cases: A10, T3a on the LAST row, T3d, T9, T10, T11 | `smoke_split_censuses` | as named; M25 killed by T3a |
| A14 | §6 block: `--expect-manifest-sha256-NN` on P1; `--expect-records-sha256` on P4; S8 as a scheduling precondition | §12 | — |
| A15 | §7.3: the docs commit precedes the α-default launch; the band is never amended | §10, §14 | — |
| A16 | §7.4: the band text | §10.6 | — |
| A17 | §9: Q5 becomes the erratum; Q6 per rows 3 and 20; Q2 per row 23 | §10.3 (E-54); A4, A7 | — |
| R2 | no `require_role` checkpoint pin for seeds 43/44 | `load_artifacts` calls `require_role(stage="E1", model_role="student", precision="fp32")` only; identity rests on the R4 pins, the recorded run_id and checkpoint_sha256, and the distinctness gate | R8, C2 |
| R4 | the four optional real-mode flags; exit 2 on mismatch; results recorded | A4, A7. They are optional in every mode; the smokes use them (§9 item 4) | R12, T12 |
| R5 | E-54 replaces Q5; D1a as row 33, D2 as row 31, D1e, D2b | A11 | D group |
| R7 | TMPDIR and the temp-root assertion | A3 | every run |
| R8 | as Annex A; U0 and U8b on numpy 1.26.4 | U0; U8b asserts `np.__version__ == "1.26.4"` | U0, U8b |
| R9 Q3 | no qualifying δ: STOP, exit 1, nothing written | `mde_entry.compute_mde` raises `td.Stop` after printing the three rejection-count curves | S2 |
| R9 Q4 | share_ties and n_zero beside every MDE_W | `mde.pair_summary`; `decision_log_line`; the CLI's per-pair lines | C1 (the DL line) |
| R9 Q7 | the binding with `--script-commit-dl-id DL-99` | the binding records the id; the §12 lines pass DL-99 | — |
| R11 | never write under reports/derived/ | every output goes to `--out-dir`, which must resolve outside the repository | R6, A8f |

## 3. STEPS 1–3 (MEASURED unless marked)

### 3.1 STEP 1, PB-1a (13:15 UTC)

STEP 1's eleven commands ran exactly as the GO gives them, one per call, while Ice was present. Ice approved
the chain once.

The verbatim outputs were shown in the session when the commands ran. They are not reproduced here: the
session's context was compacted afterwards, and recovering them from the session transcript was refused by the
L-PROT S hook (§8, Phase B refusal 1), which per the rules is not retried in any form.

Phase B continued past STEP 1, and any output other than the expected one is a STOP, so every output matched
its expectation (INFERRED from the session's course). These records confirm it now:
- **The ref HEAD named before step 5:** `refs/heads/claude/new-session-sc2u98` at `888e8f6`. This is the
  harness's session branch. `git reflog`:
  ```
  888e8f6 HEAD@{2026-10-08 07:39:18 +0000}: checkout: moving from 888e8f6855e558b3d117fb8183723c0b4bc46294 to claude/new-session-sc2u98
  31b53b7 HEAD@{2026-10-08 13:15:04 +0000}:
  647f031 HEAD@{2026-10-08 13:45:18 +0000}: commit: S4 C1: AM-17 item 3 MDE entry and the DL-27 band writer (lane L-AM17-MDE)
  b0a4669 HEAD@{2026-10-08 13:45:26 +0000}: commit: S4 C2: DL-49 aspect-ratio and DL-55 teacher suffix censuses
  ```
  The entry with an empty message at 13:15:04 is the chain's `git symbolic-ref` plus checkout.
- **C1's parent is `31b53b70d0617e4c17959db8e63235b7b75ae011`** (§4).
- **The scoped status was empty before C1's files were added** (§4.1). The one PDF probe (step 11) was not run
  again.

### 3.2 STEP 2, the venv (Q1)

```
/usr/bin/python3.11 -m venv /tmp/s4venv
/tmp/s4venv/bin/python -m pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4 statsmodels==0.14.6 pillow==12.3.0
/tmp/s4venv/bin/python -m pip install --no-cache-dir torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
/tmp/s4venv/bin/python -m pip freeze
mkdir -p /tmp/s4scratch
```

- The Q1 packages are numpy, scipy, statsmodels, CPU torch and Pillow, each at its requirements.lock pin:
  numpy :35, pillow :45, scipy :61, statsmodels :63, torch :68 (2.1.0+cu121; the GO's CPU build).
- Q1 does not list torchvision, so it was not installed.
- Every package installed; there was no STOP.
- **Environment note:** Python 3.11.17 (`/tmp/s4venv/bin/python -V`), with the venv's ensurepip pip 24.0 and
  setuptools.

`pip freeze`, re-run during STEP 5 with the same command (the venv was not changed after STEP 2):
```
filelock==3.32.3
fsspec==2026.7.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==1.26.4
packaging==26.3
pandas==3.0.6
patsy==1.0.3
pillow==12.3.0
python-dateutil==2.9.0.post0
scipy==1.11.4
six==1.17.0
statsmodels==0.14.6
sympy==1.14.0
torch==2.1.0+cpu
typing_extensions==4.16.0
```

**Deviation.** pip resolved statsmodels' dependencies pandas 3.0.6 and patsy 1.0.3, against the lock's 3.0.3
(:43) and 1.0.2 (:44). Neither is a Q1 package. Neither is part of `PINNED_ENVIRONMENT` (src/stats/artifact.py
:91-92: python, numpy, scipy, statsmodels). No S4 code imports either.

### 3.3 STEP 3, baselines at `31b53b7` (13:17–13:27 UTC)

Each smoke ran as
`env TMPDIR=/tmp/s4scratch PYTHONPATH=/home/user/plantseg-thesis /tmp/s4venv/bin/python -B scripts/<smoke>.py`,
one after another, with nothing else running.

| smoke | exit | counts | seconds |
|---|---|---|---|
| smoke_stats_rehearsal | 0 | 41/41 | 341 |
| smoke_stats_report | 0 | 63/63 | 77 |
| smoke_frozen_blobs | 0 | 10/10 | 0 |
| smoke_select_sweeps | 0 | 499/499 | 89 |
| smoke_gap_bootstrap_val | 0 | 77/77 | 38 |
| smoke_teacher_diag_drift | 0 | 8/8 | 3 |
| smoke_teacher_d4 | 0 | 32/32 | 9 |
| smoke_eval_determinism | 1 | 64/65 | 23 |

smoke_eval_determinism fails exactly one check, "R section completed". Its error is `ModuleNotFoundError: No
module named 'torchvision'` in `make_checkpoint → ML.build_fp32_student()`. That is expected: Q1 does not list
torchvision. Per Annex A §3's first bullet, "pass" means the same exit code and counts as here (§5).

## 4. The commits (MEASURED)

Every commit was made on lane/s4-mde-band with `git add -- <paths>` and then `git commit --only -F /tmp/s4scratch/msg_cN.txt -- <paths>`. `git log --format=%B 31b53b70d0617e4c17959db8e63235b7b75ae011..HEAD | git interpret-trailers --parse` printed nothing after each. Each was inspected with `git show --stat --format= HEAD -- . <X>`. Author and committer are `Claude <noreply@anthropic.com>`, as on the K2 lane's commits.

| sha | subject | paths (+/−) |
|---|---|---|
| `647f031` | S4 C1: AM-17 item 3 MDE entry and the DL-27 band writer (lane L-AM17-MDE) | `scripts/mde_entry.py` +636/−0, `scripts/smoke_mde.py` +939/−0, `src/stats/mde.py` +169/−0 |
| `b0a4669` | S4 C2: DL-49 aspect-ratio and DL-55 teacher suffix censuses | `scripts/smoke_split_censuses.py` +640/−0, `scripts/split_censuses.py` +460/−0 |
| `b5a79a7` | S4 C1b: DL-24 review fixes to the MDE entry and its smoke | `scripts/mde_entry.py` +97/−32, `scripts/smoke_mde.py` +448/−43, `src/stats/mde.py` +22/−4 |
| `7f103ef` | S4 C2b: DL-24 review fixes to the censuses and their smoke | `scripts/smoke_split_censuses.py` +305/−24, `scripts/split_censuses.py` +91/−30 |
| `3ec873b` | S4 C1c: DL-24 round-2 fixes to the MDE entry and its smoke | `scripts/mde_entry.py` +23/−8, `scripts/smoke_mde.py` +122/−28, `src/stats/mde.py` +5/−4 |
| `3b5a418` | S4 C2c: DL-24 round-2 fixes to the censuses and their smoke | `scripts/smoke_split_censuses.py` +41/−2, `scripts/split_censuses.py` +13/−8 |
| (C3) | S4 C3: the lane report | `docs/lane_reports/s4-mde-band.md` (this file) |

`git diff --stat 31b53b70d0617e4c17959db8e63235b7b75ae011..HEAD -- . <X>` at C2c (the code; C3 adds this report):
```
 scripts/mde_entry.py            |  716 +++++++++++++++++++
 scripts/smoke_mde.py            | 1438 +++++++++++++++++++++++++++++++++++++++
 scripts/smoke_split_censuses.py |  960 ++++++++++++++++++++++++++
 scripts/split_censuses.py       |  526 ++++++++++++++
 src/stats/mde.py                |  188 +++++
 5 files changed, 3828 insertions(+)
```

## 5. The final runs (STEP 6; MEASURED)

All ran at `3b5a4189c08ea17f7fb4b7b4a55ba38ede9b7efe` (C2c), one after another with nothing else running, from 2026-10-08T23:38:19Z to 2026-10-08T23:55:53Z, each as `timeout -k 60 10800 env TMPDIR=/tmp/s4scratch PYTHONPATH=/home/user/plantseg-thesis /tmp/s4venv/bin/python -B scripts/<smoke>.py` (`/tmp/s4scratch/run_final.sh`). No run came near the 3-hour stop. The two new smokes' mutation runs are inside their own runs (§6).

| smoke | estimate | measured | exit | counts | at 31b53b7 (STEP 3) |
|---|---|---|---|---|---|
| smoke_mde | ≈ 13 min | 571 s | 0 | 129/129 (mutations killed 35/35) | new |
| smoke_split_censuses | < 10 s | 1 s | 0 | 81/81 (mutations killed 18/18) | new |
| smoke_stats_rehearsal | ≈ 6 min | 280 s | 0 | 41/41 | 0, 41/41 |
| smoke_stats_report | ≈ 80 s | 61 s | 0 | 63/63 | 0, 63/63 |
| smoke_frozen_blobs | < 5 s | 1 s | 0 | 10/10 | 0, 10/10 |
| smoke_select_sweeps | ≈ 90 s | 77 s | 0 | 499/499 | 0, 499/499 |
| smoke_gap_bootstrap_val | ≈ 40 s | 35 s | 0 | 77/77 | 0, 77/77 |
| smoke_teacher_diag_drift | < 5 s | 3 s | 0 | 8/8 | 0, 8/8 |
| smoke_teacher_d4 | ≈ 10 s | 6 s | 0 | 32/32 | 0, 32/32 |
| smoke_eval_determinism | ≈ 25 s | 19 s | 1 | 64/65 | 1, 64/65 |

- Every case passed and every mutation was killed by its named check. The eight baseline smokes give the same exit codes and counts as at 31b53b7.
- smoke_eval_determinism's one failure is the baseline's: "R section completed", `ModuleNotFoundError: No module named 'torchvision'` (§3.3).
- smoke_mde's slow stages took 159 s (C1, the subprocess CLI), 150 s (C4, in-process) and 100 s (D, two full curves at n = 1,561), and the C and D checks 253 s after setup.
- D2 realised MDE_W 0.009 against MDE_t 0.008451892915205214, ratio 1.0648502164300642, with power 0.724, 0.8395 and 0.9205 at 0.008, 0.009 and 0.010: the 1.065 that Annex A row 31 expects (§10.5).
- Development runs before the final ones: smoke_mde 86/86 at C1, 125/125 at C1b, and 112/112 for the fast part at C1c; smoke_split_censuses 49/49 at C2, 77/77 at C2b and 81/81 at C2c.

## 6. Mutations and their killing checks (MEASURED; 53/53 killed in the final runs)

Each mutation replaces one module-level rule. It counts as killed only if its named check passed in the normal run and fails under the mutation; the harness records a crash under a mutation as a kill. Kill reasons were checked for every mutation added after round 1 (`/tmp/s4scratch/review/{mde,census}_kill_reasons*.log`). Each fails for the intended reason (a wrong number, a refusal message that differs, or an output written where none may be), and none is killed by a crash. The numbers have gaps: M28–M42 and M51–M52 are smoke_mde's additions, M43–M50 and M53 are smoke_split_censuses'.

**scripts/smoke_mde.py** (35 mutations, 94 checks)

| id | mutation | killed by | the killing check |
|---|---|---|---|
| M1 | zero_method 'wilcox' | U7 | call spy: the pre-registered kwargs, 1-D float64 of length n, len(grid) x b calls |
| M2 | correction False | U7 | call spy: the pre-registered kwargs, 1-D float64 of length n, len(grid) x b calls |
| M3 | alternative 'two-sided' | U7 | call spy: the pre-registered kwargs, 1-D float64 of length n, len(grid) x b calls |
| M4 | ALPHA 0.05 | D1b | d1: power at delta 0 in [0.001, 0.015] |
| M5 | p <= ALPHA | U6 | p == 0.00625 is not a rejection |
| M6 | fresh indices per delta | U8 | index spy: one default_rng(42) and one integers(0, m, size=(b, n), int64) per pair |
| M7 | RNG seed 43 | U8 | index spy: one default_rng(42) and one integers(0, m, size=(b, n), int64) per pair |
| M8 | GUARD_POINTS 1 | U3b | guard: a 1599 in the window skips k |
| M9 | count > need | U3a | guard: 1600 at k..k+2 qualifies |
| M10 | min over pairs | U4 | MDE_W is the max over pairs, first pair on a tie |
| M11 | SD with ddof 0 | U9 | analytic: ddof 1, max SD, dz from literals, MDE_t and its range |
| M12 | n_planning = included | U1a | n_planning(846) == 1561 |
| M13 | caveat mde_w >= tau_p | U5a | caveat: 0.010 -> False, validator ok |
| M14 | pair orientation flipped | C5 | orientation: 43-42 mean > 0 and 44-43 mean < 0 when seed 43 is better |
| M15 | DL-17 check skipped | R4d | s42 at batch 8 (not DL-17) -> exit 2 |
| M16 | commit check skipped | R4a | s43 with another repo_commit -> exit 2 |
| M17 | band s with pstdev | B1 | band: the writer's file passes dl27_band, (s, band) bitwise equal |
| M18 | band without its floor | B5 | band: near-equal values give the 0.005 floor |
| M28 | pairs resampled at m = n_included | C6 | in-process CLI (fast counts): every headline number recomputed from the entry |
| M29 | analytic block at m = n_included | C6 | in-process CLI (fast counts): every headline number recomputed from the entry |
| M30 | binding without its clean-tree check | R13d | binding: src/configs/scripts not clean -> exit 2 |
| M31 | path guard without its resolved-path stage | R14 | a --s43 resolving to a test location -> exit 2, nothing created |
| M32 | require_role skipped | R4g | s44 at stage E2 -> exit 2 (require_role) |
| M33 | mode gate in one direction only | R7e | the evaluator's dataset name with --synthetic-inputs -> exit 2 |
| M34 | distinctness without the checkpoints | R8c | --s43 given s42 under another run_id (same checkpoint) -> exit 2 |
| M35 | AM-5 sets compared by count only | R5 | s44 with another AM-5 set of the same size -> exit 2 STOP (lane 6 (f)) |
| M36 | report validator skipped | S3 | an entry the report validator rejects -> exit 1, nothing written |
| M37 | class-count rule removed | S4 | a per-image class count off by one -> exit 1, nothing written |
| M38 | band read-back skipped | S5 | a band its reader refuses -> exit 1, nothing written |
| M39 | rule file's band not validated | R15a | the rule file's band floor 0.01 -> exit 2 |
| M40 | the import-time stack guard removed (in the child's source) | R16a | a child process without statsmodels -> exit 2 (the import-time guard), nothing written |
| M41 | failed checks named without their details | R12 | a wrong --expect-manifest-sha256-43 -> exit 2, the observed hash in the message |
| M42 | a d of exact zeros only accepted | R17 | two runs that agree image for image -> exit 2 |
| M51 | the DL line prints dz_MDE as MDE_t | C7 | a completed real-mode run (git faked, fast counts): record paths, DL-99 and every DL-line field |
| M52 | binding without its code-files check | R13c | binding: a code file missing -> exit 2 |

**scripts/smoke_split_censuses.py** (18 mutations, 63 checks)

| id | mutation | killed by | the killing check |
|---|---|---|---|
| M19 | a pixel decode (load()) | A5 | no pixel decoded: ImageFile.load patched to raise, still exit 0 |
| M20 | AR as W/H | A4 | the portrait counts by long/short |
| M21 | >= at the threshold | A3a | 126x10 (AR 12.6) is not above 12.6; four TRAIN images are |
| M22 | recursive listing | A6a | the images/train subfolder is counted as a directory and never entered |
| M23 | exact '.jpg' rule in the aspect-ratio census | A1 | aspect-ratio: exit 0, TRAIN 9 and VAL 4 under the evaluator's rule |
| M24 | case-insensitive teacher rule | T2a | a.JPG: teacher 2 != student 3, sets differ, histogram {'.JPG': 1, '.jpg': 2} |
| M25 | val_images checked on the first row only | T3a | the LAST row with val_images 2 -> not met |
| M26 | row count unchecked | T3b | 9 rows -> records_rows_10 false |
| M27 | manifest check skipped | T3c | another (uniform) manifest hash -> not equal |
| M43 | binding without its clean-tree check | A11d | binding: src/configs/scripts not clean -> exit 2 |
| M44 | path guard without its resolved-path stage | A12 | a --data-root resolving to a test location -> exit 2, nothing created |
| M45 | listed-name refusal disabled | A13 | a listed name containing 'test' -> exit 2 before any header is read |
| M46 | listed entries not resolved | A14 | a listed entry resolving to a test location -> exit 2 before any header is read |
| M47 | VAL stems in stem order | T1c | the census VAL manifest equals the records'; expectation met |
| M48 | dot-named files kept by the teacher rule | T14 | a dot-named .jpg: skipped by the teacher's listing, kept by the student's -> not met |
| M49 | an overflowing number accepted | T15 | a records number that overflows to inf (1e999) -> exit 2 |
| M50 | a duplicate key resolved silently | T16 | a records row with a duplicate key -> exit 2 |
| M53 | header warnings dropped | A15b | header warnings counted per split and combined (TRAIN 3, VAL 1) at MAX_IMAGE_PIXELS 2,000 |

## 7. DL-24 review (MEASURED; STEP 5)

Each round ran a read-only workflow: reviewers, each followed by an adversarial verifier that tried to refute
every finding and re-checked the reviewer's "checked OK" list. Every agent prompt carried the GO's hard limits
verbatim, plus read-only rules:
- no edits;
- no git;
- Grep only inside src/stats, src/eval, src/training, scripts or configs, with `!*test*`;
- docs/ by exact path only;
- Python only in /tmp/s4scratch.

HEAD, the scoped status and the five files' sha256 were snapshotted before and after each round, and were
identical both times (`/tmp/s4scratch/review/snapshot_*.txt`). Every finding was in scope (`fix_in_scope`
true). "Item" quotes the reviewer's claim, cut at 90 characters. Line numbers in the findings refer to the code
under review, not to the final commit.

### 7.1 Round 1: `wf_05f21a2d-987`, on 31b53b7..b0a4669

Four reviewers each had an adversarial verifier (8 agents, 92 min):
- mde-stats: the statistics;
- entry-cli: the entry CLI's gates and the band;
- censuses: the two censuses;
- smokes: the two smokes.

They returned 34 findings, all CONFIRMED and none refuted (verifier severities: 0 blocking, 0 major, 24 minor,
10 nit), plus 3 high-confidence extras from the verifiers. C1b `b5a79a7` and C2b `7f103ef` resolve every one.
Two "add a gate" proposals (entry-cli-1, entry-cli-2) were resolved as "record, do not gate", per their
verifiers, and stand as open items O3 and O4 for a ruling.

*mde-stats* (3 findings)

| Finding | Reviewer | Verifier | Item | Resolution |
|---|---|---|---|---|
| mde-stats-1 | minor | CONFIRMED (minor) | decision_log_line renders Annex A section 4's replacement MDE clause but leaves out the p… | The printed DL line is the row's whole decision cell: the three artifacts (run_id, MANIFEST.sha256 sha256, checkpoint sha256), the commit, GPU and image, and reports/derived/ names for outputs of record. C1 checks the run_ids, the MANIFEST and checkpoint sha256s and both file sha256s (C1b); round 2 added every other field and the real-mode line (C1c, r2-mde-2) |
| mde-stats-2 | minor | CONFIRMED (minor) | No smoke check covers how the CLI feeds the pre-registered quantities into the entry. Not… | C2 recomputes every headline number from the entry with literals (sizes, curves, guard, max, dz, MDE_t, range, n_planning). C6 does the same on an in-process run so M28 (resample at m) and M29 (analytic at m) are killed (C1b) |
| mde-stats-3 | nit | CONFIRMED (nit) | pair_summary accepts an all-zero d, and a very small m whose d contains zeros. The pinned… | check_d refuses a d of exact zeros only and names a non-finite value; compute_mde turns MdeError into a refusal; the tiny-m all-zero row is documented. U10, R17; M42 (C1b) |

*entry-cli* (9 findings)

| Finding | Reviewer | Verifier | Item | Resolution |
|---|---|---|---|---|
| entry-cli-1 | minor | CONFIRMED (minor) | The 'one evaluate_model.py commit' gate compares only run.repo_commit. Neither the gate n… | governed_paths_clean recorded per artifact, not gated. Gating it is open item O3, for a ruling (C1b) |
| entry-cli-2 | minor | CONFIRMED (minor) | The band-input tie (best.json value == the artifact's checkpoint_best_val_miou_all_class)… | checkpoint_iteration and checkpoint_path_basename recorded per artifact, not gated. The gate (iteration equals best_ckpt's) is open item O4, for a ruling (C1b) |
| entry-cli-3 | minor | CONFIRMED (minor) | On a provenance or band-input refusal, require_passed prints only the names of the failed… | require_passed names each failed check with its detail. R12 requires the observed hash in the message; M41 (C1b) |
| entry-cli-4 | minor | CONFIRMED (minor) | decision_log_line goes straight from the entry name to n_included. It omits the specified… | As mde-stats-1 |
| entry-cli-5 | minor | CONFIRMED (nit) | CODE_FILES leaves out src/eval/adapters.py. mde_entry imports it directly to get DATASET_… | CODE_FILES adds src/eval/adapters.py and src/eval/protocols.py (C1b) |
| entry-cli-6 | minor | CONFIRMED (minor) | No smoke case runs the real commit_binding. R7a and R7b stop in check_flags, and R7c and … | R13a-d run the real commit_binding with td._git faked: HEAD unreadable, HEAD mismatch, a missing code file, a dirty tree; M30 (C1b) |
| entry-cli-7 | minor | CONFIRMED (minor) | The mode gate is tested in one direction only: synthetic names in real mode (R7c). Real d… | R7e: the evaluator's dataset name with --synthetic-inputs; the gate is the module-level check_mode; M33 (C1b) |
| entry-cli-8 | minor | CONFIRMED (minor) | R8 passes the s42 directory as --s43, which trips the 'directories' entry of the distinct… | R8b (a copy of s42: run_ids) and R8c (s42 under another run_id: checkpoints); the gate's entries are the module-level DISTINCT; M34 (C1b) |
| entry-cli-9 | nit | CONFIRMED (nit) | artifacts['42'] has no dl17_delta, which plan §1.6 lists ('Seed 42 also has dl17_delta').… | artifacts['42'].dl17_delta, from the same helper as the DL17_BAND check; C2 checks it (C1b) |
| entry-cli-v1 (verifier) | — | nit | If scipy or statsmodels is missing from the running image, the run exits 4 ('unexpected')… | C1b's environment_check branch was unreachable (round 2: partial). C1c refuses a missing numpy, scipy or statsmodels at import (exit 2): R16a, R16b; M40 (C1c). The P1+P2 image must carry the pinned stats stack: open item O5 |

*censuses* (7 findings)

| Finding | Reviewer | Verifier | Item | Resolution |
|---|---|---|---|---|
| censuses-1 | major | CONFIRMED (minor) | teacher_rule (`name.endswith('.jpg')`) is not the listing mmseg applies. BaseSegDataset.l… | teacher_rule skips dot-named files, as mmengine's LocalBackend does ([U], transcribed); per-folder dot-file and symlink counts. T14; M48 (C2b) |
| censuses-2 | minor | CONFIRMED (minor) | No smoke case pins Annex A row 24's order rule (VAL stems in name-sorted file order, not … | VAL fixture u-1.jpg, u.jpg, v.jpg, whose name order differs from its stem order; T1c asserts the two hashes differ; teacher_stems is module-level; M47 (C2b) |
| censuses-3 | minor | CONFIRMED (minor) | No smoke case covers the listed-name 'test' refusal (split_censuses.py:422-423), a hard-l… | A13 (aspect-ratio) and T13 (teacher-suffix) add a never-created 'test' name to a listing: exit 2, no header read; M45 (C2b) |
| censuses-4 | minor | CONFIRMED (minor) | read_records is not as strict as plan §1.5 P4 ('non-finite values are refused') and §1.2 … | parse_finite_float, no_duplicate_keys, and read and decode errors as refusals (the records sha256 included). T15-T18; M49, M50 (C2b) |
| censuses-5 | minor | CONFIRMED (minor) | Symlinked entries in images/train and images/val are followed, and read_header opens thei… | refuse_test_entries resolves every listed entry before any header is read. A14; M46 (C2b) |
| censuses-6 | nit | CONFIRMED (nit) | The student rule uses os.path.splitext, but the loader and evaluator use Path.suffix and … | The docstring and the image_rule string state where splitext and Path.suffix differ (names of leading dots plus the suffix); the plan's splitext form is kept (C2b) |
| censuses-7 | nit | CONFIRMED (nit) | The aspect-ratio output does not follow plan §1.5 P3's layout ('Per split and combined: …… | Formats and header warnings per split and combined (the top-level header_warnings is gone). A15 (C2b) |

*smokes* (15 findings)

| Finding | Reviewer | Verifier | Item | Resolution |
|---|---|---|---|---|
| smokes-1 | major | CONFIRMED (minor) | C1 claims 'the DL line with both sha256', but it never reads the DL line. Its predicate i… | As mde-stats-1. C1 reads the DL line, and the stored line equals the printed one without the entry's sha256; decision_log_line is in C2's keys (C1b) |
| smokes-2 | minor | CONFIRMED (minor) | R7a (smoke_mde) and A8g (smoke_split_censuses) use the needle '--script-commit'. commit_b… | R7a and A8g use the gate's own words; R7d and A8i give a commit without a DL id (C1b, C2b) |
| smokes-3 | minor | CONFIRMED (minor) | Neither smoke tests the run-of-record binding: HEAD == --script-commit, CODE_FILES presen… | R13a-d and A11a-d; M30, M43 (C1b, C2b) |
| smokes-4 | minor | CONFIRMED (minor) | Two TEST-protection gates have no killing check. (1) The resolved-path stage of td.refuse… | R14 and A12: a path resolving to a test location. A13 and T13: a listed test name. M31, M44, M45 (C1b, C2b) |
| smokes-5 | minor | CONFIRMED (minor) | Several mde_entry provenance gates have no check that fails when they are deleted. (1) re… | R4g (stage E2), R4h (int8), R4i (inputs on the CPU), R4j (no determinism policy), R8b, R8c, R7e; M32, M33, M34 (C1b) |
| smokes-6 | minor | CONFIRMED (minor) | Three of the CLI's exit-1 gates are never exercised, so deleting them goes unnoticed: (1)… | S3 (the report validator), S4 (a class count off by one), S5 (a band its reader refuses); M36, M37, M38 (C1b) |
| smokes-7 | minor | CONFIRMED (nit) | Two band-input gates have no killing check. (1) band_floor's validation of the rule file:… | R10b (best_ckpt 7). R15a and R15b patch the rule file's band (floor 0.01, kind 'fixed'); M39 (C1b) |
| smokes-8 | nit | CONFIRMED (nit) | Several split_censuses input refusals are untested. Deleting any of them turns an exit-2 … | T19 (no --teacher-records), A16 (the records flags with aspect-ratio), T20 (a non-JSON line), T21 (no val_images), T22 (an empty file) (C2b) |
| smokes-9 | minor | CONFIRMED (minor) | C2 is the only check of the written entry's numbers. It checks key presence, the validato… | As mde-stats-2 |
| smokes-10 | minor | CONFIRMED (minor) | R5's alternative AM-5 set has K + 2 = 8 exclusions against 6, so R5 cannot tell an id-set… | R5 uses an AM-5 set of the same size with other ids (both asserted); R5b keeps K + 2; same_excluded is module-level; M35 (C1b) |
| smokes-11 | minor | CONFIRMED (minor) | No census check fixes the VAL pass order that Annex A row 24 settles: name-sorted files, … | As censuses-2 |
| smokes-12 | minor | CONFIRMED (minor) | R6 and A8f pass an inside-repo --out-dir with nothing patched. The refusal at require_out… | R6 and A8f run with td.write_files_exclusive tripwired (R6 also with fast counts) (C1b, C2b) |
| smokes-13 | minor | CONFIRMED (nit) | Both smokes create the temporary root with tempfile.mkdtemp before testing its realpath f… | make_root checks TMPDIR's realpath before anything is created; the root is a fixed prefix plus secrets.token_hex(4), which cannot spell 'test' (C1b, C2b) |
| smokes-14 | nit | CONFIRMED (nit) | S1 copies into the fixed CTX.root/variants/s44_value, so a second call raises FileExistsE… | S1's variant is built once in setup_variants (C1b) |
| smokes-15 | nit | CONFIRMED (nit) | T1b's 'iterations 4000..40000' and 'DL-68 source' sub-checks compare the census with the … | T1b compares with literal iterations and the literal DL-68 source; the records fixture is built from the literal list (C2b) |
| smokes-v1 (verifier) | — | nit | no_filesystem intercepts stat, lstat, scandir, listdir, mkdir and realpath, but not open,… | no_filesystem also intercepts builtins.open, io.open, os.open and os.access, in both smokes (C1b, C2b) |
| smokes-v2 (verifier) | — | nit | No check pins the entry's label to the verbatim text of AM-17 item 3(e). The plan require… | C2 checks the entry's label against item 3(e)'s literal text (C1b) |

### 7.2 Round 2: `wf_2ef46ca4-ed5`, on b0a4669..7f103ef (the round-1 fixes)

Two reviewers each had an adversarial verifier (4 agents, 50 min):
- r2-mde: C1b, the three MDE files;
- r2-census: C2b, the two census files.

They checked each round-1 resolution and looked for defects the fix diff introduced. The 43 resolutions checked
(27 and 16) were all resolved except entry-cli-v1, which both the reviewer and the verifier rated partial.
There were 10 new findings, all CONFIRMED and none refuted (verifier severities: 2 minor, 8 nit), and no
extras. C1c `3ec873b` and C2c `3b5a418` resolve every one.

| Finding | Reviewer | Verifier | Item | Resolution |
|---|---|---|---|---|
| entry-cli-v1 (round 1) | partial | partial | environment_check's ImportError branch cannot be reached: src.stats imports scipy and statsmodels at import, so a missing package exits 1 | As r2-mde-1 |
| r2-mde-1 | major | CONFIRMED (minor) | A stack without scipy or statsmodels fails at import (exit 1); R16 and M40 tested an injected raise no missing package produces | `require_stats_stack()` runs before the src.stats imports: exit 2, naming the package; the dead branch is removed. R16a and R16b run the CLI in a child process without statsmodels or scipy, whose excepthook prints only the type and message (no traceback line is read). M40 removes the guard from the child's source and is killed by R16a (C1c) |
| r2-mde-2 | minor | CONFIRMED (minor) | Nothing compared the DL line's numbers, commit clause, title or record-path prefix; the real-mode line was never produced | C1 checks every field (`dl_fields`) on the full run. C7 is a completed real-mode run (git faked, fast counts) that checks the reports/derived/ names, "(DL-99)." and every field. M51 (the line printing dz_MDE as MDE_t) is killed by C7. The C fixtures have no exact ties, so share_ties and n_zero are pinned at 0/0/0 only: open item O9 (C1c) |
| r2-mde-3 | nit | CONFIRMED (nit) | The docstring's tripwire sentence overstated which checks are tripwired | Reworded: R6 and R14 are tripwired, and every other check writes only under the temporary root (C1c) |
| r2-mde-4 | nit | CONFIRMED (nit) | M30's mutant removed two rules (the code-files check too) | `binding_without(rule)` removes one rule; M52 (code-files check) is killed by R13c (C1c) |
| r2-mde-5 | nit | CONFIRMED (nit) | R4g–R4j and R10b ran the full MDE under a regressed gate | Fast counts in R4g–R4j and R10b (C1c) |
| r2-mde-6 | nit | CONFIRMED (nit) | mde.py's "when m is tiny" and its probability wording | Reworded: share_ties near 1; share_ties ** n per row and about B times that per pair; the CLI exits 4 (C1c) |
| r2-census-1 | minor | CONFIRMED (nit) | A15's header-warning sub-checks passed vacuously | A15b runs at MAX_IMAGE_PIXELS 2,000 (TRAIN 3, VAL 1, combined 4); M53 (warnings dropped) is killed by A15b (C2c) |
| r2-census-2 | nit | CONFIRMED (nit) | Two rule texts were not updated for the dot-file skip | The docstring and val_manifest.rule name the teacher's rule. The teacher_rule texts say dot-named files are skipped and dot-named directories entered; the no-subdirectory check fails such directories (C2c) |
| r2-census-3 | nit | CONFIRMED (nit) | make_root called realpath on TMPDIR before any string check | The string is checked first, then the realpath, in both smokes (C1c, C2c) |
| r2-census-4 | nit | CONFIRMED (nit) | The symlink counts were checked only at zero | A17 and T23 count an in-folder symlink (C2c) |

**Hook refusals and slips inside the workflows** (from the agents' own reports; none retried):
- **Round 1, reviewer entry-cli: a slip.** One Glob from the repository root listed the name src/stats/tests.py
  (§9 item 2).
- **Round 1, verifier smokes: a slip.** One Grep on `src/`, and a scenario label holding "test", created by
  nothing (§9 item 3).
- **Round 2, reviewer r2-mde: a refusal.** L-PROT S refused one inline `python -c` probe whose code held the
  words git and commit: "[L-PROT S] inline interpreter code that runs git is not analysed; run git from the
  shell."
- **Round 2, reviewer r2-mde: a slip.** Its first missing-package probe let Python's default traceback print one
  source line of src/stats/tests.py (line 21, the `statsmodels.stats.multitest` import, read through linecache;
  §9 item 4). Its later probes patched linecache and the excepthook. C1c's R16a/b child shows no traceback.
- **Round 2, reviewer r2-mde: a slip.** One `ls -la /home/user/plantseg-thesis | grep -i s4_smoke`, a listing of
  the repository root filtered to one pattern, which matched nothing (§9 item 5).
- **Round 2, verifier r2-mde: a refusal.** L-PROT N refused a `for d in *` loop in a scratch folder: "[L-PROT N]
  the glob `*` matches docs/reference/reference.pdf. …"

No tool call by any agent opened, read, grepped or hashed src/stats/tests.py or anything under docs/reference/.
A search of the round-2 agents' transcripts for tool inputs naming tests.py found none.

## 8. SL-1 lines and hook refusals (MEASURED)

### 8.1 SL-1

The guard runs in log mode (DL-79), and its log `~/.claude/sl1_guard.log` holds only this session's lines.
At the end of the session it holds 60 lines, verbatim below (the guard cuts a command at 200
characters):

```
2026-10-08T07:42:15Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T07:42:52Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/training glob=!*test*
2026-10-08T07:43:39Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs glob=!*test*
2026-10-08T07:43:40Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T07:43:41Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/training glob=!*test*
2026-10-08T07:43:48Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs glob=!*test*
2026-10-08T07:44:01Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T07:45:16Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T07:45:18Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs glob=!*test*
2026-10-08T07:45:19Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T07:45:20Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T07:53:00Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs glob=!*test*
2026-10-08T07:53:09Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T08:05:52Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T08:21:48Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T13:18:43Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T13:20:30Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/data glob=!*test*
2026-10-08T13:29:25Z	Bash	SL1-B1	cat /tmp/s4scratch/baseline_summary.txt; echo; for s in smoke_stats_rehearsal smoke_stats_report smoke_frozen_blobs smoke_select_sweeps smoke_gap_bootstrap_val smoke_teacher_diag_drift smoke_teacher_d
2026-10-08T13:42:09Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T14:18:20Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T14:19:27Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T14:22:51Z	Glob	SL1-G1	/home/user/plantseg-thesis pattern=src/stats/*.py
2026-10-08T14:26:03Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T14:28:19Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T14:35:00Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T14:40:39Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T14:56:12Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T14:56:13Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T14:56:19Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T14:58:26Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T15:06:22Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T15:09:41Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T15:10:56Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T15:11:06Z	Grep	SL1-G1	/home/user/plantseg-thesis/src glob=!*test*
2026-10-08T15:14:00Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T15:16:42Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T15:16:49Z	Glob	SL1-G1	/home/user/plantseg-thesis/src/eval pattern=__init__.py
2026-10-08T15:30:47Z	Grep	SL1-G1	/home/user/plantseg-thesis/src glob=!*test*
2026-10-08T15:31:51Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T15:31:52Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T15:37:02Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T15:42:39Z	Bash	SL1-B1	cd /tmp/s4scratch && env TMPDIR=/tmp/s4scratch PYTHONPATH=/home/user/plantseg-thesis timeout 590 /tmp/s4venv/bin/python -B /tmp/s4scratch/review/smokes_verify/v5_checked.py > /tmp/s4scratch/review/smo
2026-10-08T15:43:48Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T15:54:00Z	Glob	SL1-G1	/home/user/plantseg-thesis/src/eval pattern=protocols.py
2026-10-08T22:48:51Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T22:48:52Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T22:48:53Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/training glob=!*test*
2026-10-08T22:48:54Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T22:49:28Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T22:49:29Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/eval glob=!*test*
2026-10-08T22:49:30Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/training glob=!*test*
2026-10-08T22:49:31Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T22:49:40Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T22:56:37Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T23:04:30Z	Bash	SL1-B1	find /tmp/s4scratch/review2/r2-mde -maxdepth 3 | awk '{print tolower($0)}' | grep -c 'test'; find /tmp/s4scratch/review2/r2-mde -maxdepth 2 | head -30; sha256sum /home/user/plantseg-thesis/src/stats/m
2026-10-08T23:05:25Z	Grep	SL1-G1	/home/user/plantseg-thesis/configs/teacher glob=!*test*
2026-10-08T23:10:24Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/stats glob=!*test*
2026-10-08T23:13:44Z	Bash	SL1-B1	cd /tmp/s4scratch/s4_smoke_mde_d885d938/outs && for d in *; do printf "%s " "$d"; stat -c '%y' "$d"; ls "$d" | tr '\n' ' '; echo; done | sort -k2
2026-10-08T23:28:43Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*test*
2026-10-08T23:55:59Z	Bash	SL1-B1	cat /tmp/s4scratch/final_summary.txt; echo ---; for s in smoke_mde smoke_split_censuses smoke_stats_rehearsal smoke_stats_report smoke_frozen_blobs smoke_select_sweeps smoke_gap_bootstrap_val smoke_te
```

- **SL1-G1** (55 lines). A Grep or Glob over a repository directory. The guard logs these because a directory
  search can reach test-named files.
  - **Phase A** (15 lines, 07:42–08:21): Greps with `!*test*` in src/eval, src/training, src/stats, scripts and
    configs.
  - **This session in Phase B:** Greps with `!*test*` in configs/teacher, src/data, src/stats, src/eval and
    scripts, and one Glob of src/eval for protocols.py (15:54:00).
  - **The review agents** (§7): Greps with `!*test*` in src/stats, src/eval, src/training, scripts and
    configs/teacher, plus three that went wider:
    - two Greps on `src` as a whole (15:11:06 and 15:30:47; §9 item 3);
    - one Glob from the repository root with `pattern=src/stats/*.py` (14:22:51; §9 item 2);
    - one Glob of src/eval for `__init__.py` (15:16:49).
- **SL1-B1** (5 lines). A command over a path the guard cannot resolve.
  - 13:29:25 and 23:55:59: this session's loops over its /tmp/s4scratch logs.
  - 15:42:39, 23:04:30 and 23:13:44: review agents' commands in their scratch folders.
    - 23:13:44 is the `for d in *` loop that L-PROT N refused (§7.2).
    - 23:04:30 counted, without printing them, the names containing "test" in the agent's own scratch folder,
      and hashed the lane files.

No session command, the agents' included, read, listed, grepped or hashed anything under docs/reference/.
None used docs/ as a search path. Only the agents' slips in §9 items 2 and 5 used the repository root.

### 8.2 Hook refusals (none retried in any form)

**Phase A** (from the plan's Appendix R):
1. L-PROT C refused a scratch probe that held `git commit --allow-empty` for a throwaway repository:
   "[L-PROT C] git commit must name its paths".
2. L-PROT G refused an exact-path existence loop: "[L-PROT G] git ls-files: the pathspec scope includes
   docs/reference/reference.pdf". Its pathspec was a shell variable.

**Phase B:**
1. **STEP 5, while the review ran.** L-PROT S refused a compound command that set `T=<the session transcript
   .jsonl>`, then ran `ls -la "$T"` and two greps over it. The purpose was to recover STEP 1's outputs after the
   context compaction. Message: "[L-PROT S] `ls $T` lists docs/reference/reference.pdf." The guard cannot
   resolve `$T`. Not retried in any form; §3.1 uses git's own reflog instead.
2. **STEP 5.** L-PROT N refused a `printf … >> /tmp/s4scratch/hook_refusals.txt` whose text quoted refusal 1's
   message, and so named the protected PDF's path. Message: "[L-PROT N] the command names
   docs/reference/reference.pdf; only the exact status probe may name it. …" Not retried; the refusals are
   recorded here instead.

Two more refusals hit review agents inside the workflows (§7.2), and neither was retried: L-PROT S on an inline `python -c` probe, and L-PROT N on a `for d in *` loop. No other command was refused.

### 8.3 Tooling requests that were not followed (reported once)

- **Attribution trailers.** A session reminder asked for `Co-Authored-By` and `Claude-Session` trailers on
  commits. The GO says "No trailers", and AGENTS.md treats attribution boilerplate as non-approval. No commit
  carries a trailer; `git interpret-trailers --parse` printed nothing after every commit (§4).
- **The session branch.** The harness's text names `claude/new-session-sc2u98` as the branch to develop on and
  push. The GO names `lane/s4-mde-band`, built from 31b53b7 by PB-1a, and authorizes one push of that branch.
  Nothing was committed to or pushed from `claude/new-session-sc2u98`.
- **Stop hooks.** The session's stop hook (`~/.claude/stop-hook-git-check.sh`) asked once, at the end of a turn while C1b and C2b were still being verified, to "commit and push these changes to the remote branch". This was not followed: commits follow the GO's steps after verification, and the one push is STEP 8.

## 9. Slips and deviations

**Slips.**
1. **Bytecode files (STEP 4, before C1).** A syntax check ran `python -B -m py_compile`. Despite `-B`, it wrote
   three gitignored files:
   - `scripts/__pycache__/mde_entry.cpython-311.pyc`
   - `scripts/__pycache__/smoke_mde.cpython-311.pyc`
   - `src/stats/__pycache__/mde.cpython-311.pyc`

   They are untracked and ignored, and appear only in `git status --ignored`. They are left in place, because
   the GO says never delete. Later syntax checks used `ast.parse`. Python checks a cached file's source
   timestamp, so a stale .pyc is never used.
2. **A review agent searched from the repository root (STEP 5).** Reviewer entry-cli reported it in its
   hook_refusals field: one Glob call used `/home/user/plantseg-thesis` as its search path, with the pattern
   `src/stats/*.py`. Glob takes no exclude, so the listing included the name `src/stats/tests.py`. The file was
   not opened, read or hashed.

   This breaches "never use the repo root … as a search path" and the limit on listing a test-named path, both
   of which were stated in every agent's prompt. The agent's later searches used Grep inside src/eval,
   src/training or scripts with `!*test*`.
3. **Review agents searched too wide (STEP 5).** Two Greps with the path `src/` (with `!*test*`) appear in the
   SL-1 log, at 15:11:06 and 15:30:47. Verifier smokes reported one of them. Both are wider than the src/stats,
   src/eval and src/training the agents were given. It returned only matching lines;
   no test-named path was opened.

   It also reported that a scenario label it chose contained "test". Its own guard asserted before mkdir, so
   nothing with that name was created, and the label was renamed.
4. **A review agent's traceback showed one line of a test-named file (STEP 5, round 2).** Reviewer r2-mde
   probed a stack without statsmodels. Python's default traceback printed one source line of
   `src/stats/tests.py`: line 21, `from statsmodels.stats.multitest import multipletests`, read by
   linecache.

   No tool opened the file. A search of the agents' transcripts finds no tool input naming it. The agent's
   later probes patched linecache and the excepthook. This is running code (Q2-4 ruling 3), but a source line
   was displayed.

   C1c's R16a/b run the same probe in a child whose excepthook prints only the type and message. C1c's import
   guard means a real run on such a stack no longer ends in that traceback.
5. **A review agent listed the repository root (STEP 5, round 2).** Reviewer r2-mde ran `ls -la
   /home/user/plantseg-thesis | grep -i s4_smoke` to confirm that R6 had left nothing in the checkout. Only
   lines matching s4_smoke could be shown, and none were. This breaches "never use the repo root … as a search
   path" in letter.

**Deviations.**
1. **pandas and patsy versions** (§3.2): 3.0.6 and 1.0.3 against the lock's 3.0.3 and 1.0.2. They are
   statsmodels' dependencies, not Q1 packages, outside `PINNED_ENVIRONMENT`, and not imported by S4 code.
2. **smoke_eval_determinism at 64/65 and exit 1** (§3.3): it needs torchvision, which Q1 does not list. The
   baseline and the final run agree (§5).
3. **The R4 pins in every mode.** R4 says synthetic and smoke modes "do not need" the four flags. The CLIs
   accept them in every mode, and the smokes use them (C1, C2, R12, T1a, T1b, T12). No mode refuses them.
4. **The teacher's listing rule** (censuses-1): `not name.startswith('.') and name.endswith('.jpg')` replaces
   the plan's (xvi) literal `name.endswith('.jpg')`. This follows mmengine's LocalBackend dot-file skip, which is
   transcribed, not checked: mmengine is not in the repository ([U]). It changes nothing for a folder with no
   dot-named file, and the census records each folder's dot-file count.
5. **Fields added to the entry beyond the plan's §1.6:**
   - each artifact's governed_paths_clean, checkpoint_iteration and checkpoint_path_basename (recorded, not
     gated: open items O3 and O4);
   - artifacts["42"].dl17_delta, as §1.6 asks (entry-cli-9).
6. **The aspect-ratio census layout.** formats and header_warnings are given per split and combined, per the
   plan's §1.5 (censuses-7), instead of one top-level header_warnings.
7. **Smoke counts.** smoke_mde has 129 results (94 checks and 35 mutations) and smoke_split_censuses 81 results (63 checks and 18 mutations). The plan's estimates
   were 71 and 43; Annex A's were ≈83 and ≈49. The review's resolutions (§7) add the rest.
8. **Temporary-root names.** `s4_smoke_mde_<8 hex>` and `s4_smoke_censuses_<8 hex>`, created under TMPDIR
   after its string and realpath are checked. The plan's §4 said `tempfile.mkdtemp(prefix="s4_smoke_")`.
9. **The DL-24 review ran after C1 and C2**, in the GO's step order (STEP 4 commits, STEP 5 review). Its fixes
   are new commits: C1b and C2b for round 1, C1c and C2c for round 2 (§4).
10. **Two statements in C1b's commit message are wrong, and the commit is not amended.**
    - "A stack without scipy or statsmodels is refused (exit 2)" was not true at C1b. The package failed at
      import with exit 1 (r2-mde-1). It is true from C1c on.
    - "every refusal that could write under a regression run with writes tripwired" overstates. Only R6 and R14
      are tripwired, and every other check writes only under the temporary root (r2-mde-3).

    C1c's message and docstrings state both correctly. History is not rewritten.

## 10. Records proposed (R10; this lane writes none of them)

The pre-launch docs commit enters these; DL-99, DL-100 and E-54 are created there (R3, R5; F4). Placeholders
in angle brackets are filled from the real entry; §10.1's table maps each to its JSON key. The CLI's printed
`DL line: …` is the row's whole decision cell, already filled, from "AM-17 item 3 MDE entry and AM-16 item 2 / DL-27
tie band (one entry, AM-17 item 3)." to "Code <C_S4> (DL-99)." This follows mde-stats-1 / entry-cli-4 (§7). The
entry's `decision_log_line` is the same text without the entry's own sha256.

### 10.1 The DL-99 row (Annex A §4's corrected text, filled from the code)

`| DL-99 | RECORDED <date> (<docs commit>) | AM-17 item 3 MDE entry and AM-16 item 2 / DL-27 tie band (one entry, AM-17 item 3). MDE: reports/derived/mde_entry_<UTC>.json sha256 <entry sha256>; E1 s42/s43/s44 VAL artifacts <run_id, MANIFEST.sha256 sha256 and checkpoint sha256 each> at evaluate_model.py commit <repo_commit>, <gpu_name>, image <image_digest>; n_included <m> (K <k>); n_planning <n>; shifted-null MDE_W <x> (43−42 <a>, 44−42 <b>, 44−43 <c>; share of exact-zero differences <s1>/<s2>/<s3>, n_zero <z1>/<z2>/<z3>; B 2,000 resamples of size n from default_rng(42) — one index matrix per pair, identical to re-seeding per (pair, δ), so the three guard points share rows; the pre-registered Wilcoxon call, α 0.00625, p < α, 3-point guard; scipy <v>, numpy <v>); SD_Δ <s>, dz_MDE <dz>, MDE_t <t> [<lo>, <hi>]; τ_P 0.010; power_caveat <bool>; label: planning proxy (item 3(e)); lane 6 (f): exact ties deflate MDE_W under the constant-shift null. Band: reports/derived/dl27_band.json sha256 <band sha256>; E1 best VAL 42/43/44 <x42>/<x43>/<x44> (best.json sha256s in the entry); s <s>; band = max(0.005, √2·s) = <band>. Code <C_S4> (DL-99). | reports/derived/mde_entry_<UTC>.json; reports/derived/dl27_band.json; src/stats/mde.py; scripts/mde_entry.py | no |`

Changes against the plan's §7.1:
- the MDE part is Annex A §4's corrected text;
- n_zero stands beside share_ties (R9 Q4);
- the seed-43/44 checkpoint sha256s are named, since DL-99 documents them from its commit on (R2);
- the code id is DL-99 (R3, R9 Q7).

| placeholder | entry key (mde_entry_<UTC>.json) |
|---|---|
| entry sha256 | the CLI's `written: mde_entry_<UTC>.json sha256 <hex>` line (the entry cannot hold its own hash) |
| run_id, MANIFEST.sha256 sha256, checkpoint sha256 | `artifacts.<seed>.run_id`, `artifacts.<seed>.sha256s["MANIFEST.sha256"]`, `artifacts.<seed>.checkpoint_sha256` |
| repo_commit, gpu_name, image_digest | `artifacts.<seed>.repo_commit`, `artifacts.<seed>.device_class[2]`, `artifacts.<seed>.image_digest` (one value each: provenance_checks) |
| m, k, n | `n_included`, `K_excluded`, `n_planning` |
| x, a, b, c | `mde_w`; `pairs["43-42"].mde_w`, `pairs["44-42"].mde_w`, `pairs["44-43"].mde_w` |
| s1–s3, z1–z3 | `pairs[<pair>].share_ties`, `pairs[<pair>].n_zero` |
| scipy, numpy | `scipy_version`, `numpy_version` |
| s (MDE), dz, t, lo, hi | `sd_delta`, `dz_mde`, `mde_t`, `mde_t_range[0]`, `mde_t_range[1]` |
| bool | `power_caveat` |
| band sha256, x42–x44, s (band), band | `dl27_band.sha256`, `dl27_band.e1_best_val`, `dl27_band.s`, `dl27_band.band` (equal to dl27_band.json's own keys) |
| C_S4 | `git_commit` (= `script_commit` in a real run) |

### 10.2 Status texts

- **DL-27** (append): "; DL-27 band RECORDED <docs commit> (DL-99): band <value> (reports/derived/dl27_band.json
  sha256 <…>); the band-entry precondition of the α sweep is met (committed before the α-default launch; PL-16
  band_changed_after_launch)".
- **DL-49:** "DECIDED 2026-09-30; census RECORDED <docs commit> (S4): TRAIN 5,367 + VAL 846, max AR <x>
  (<split>/<stem>, <W>×<H>), above 12.6: <a>, above 26.5: <b>
  (reports/derived/train_val_aspect_ratio_census.json sha256 <…>)". The fields are `combined.max_ar`,
  `combined.max_ar_image`, `combined.count_above_12_6` and `combined.count_above_26_5`; the per-split values are
  in `splits`.
- **DL-55:** "DECIDED 2026-09-30; census RECORDED <docs commit> (S4): images/train exact '.jpg' 5,367,
  images/val 846 (other suffixes <…>, subdirectories <…>); the run of record's teacher_selection_records.jsonl
  (sha256 3448daf3…, the DL-68 value, checked in-script) val_images 846 at all 10 validations, VAL manifest hash
  equal to the census's <bool> — <PASS|FAIL>". The fields are:
  - `splits.<split>.suffix_counts`, `teacher_rule_count` and `n_dirs`;
  - `records.records_sha256`, `records.expected_sha256_matches` and `records.val_images`;
  - `val_manifest.equal` and `expectation_met` (PASS iff true; the CLI exits 1 otherwise).

### 10.3 E-54 (docs/lane_specs/errata.md; lane 6 (d), docs/lane_specs/part1.md:62)

"d1/d2 wording: the synthetic d is an exactly symmetric (antithetic ±x) sample of the stated N(0, σ) with the
stated zero share, because the procedure does not centre d and the stated bounds presuppose location 0 (i.i.d.
draws meet them with probability ≈ 0.2); d2's sample SD is normalised to 0.10; d1's 'MDE_t within 1e-9 of
0.0845·SD' reads '|MDE_t − dz·SD_Δ| ≤ 1e-9 with dz recomputed from 2.4977, 0.8416 and √1561, and |dz − 0.0845| <
5e-5' (dz(1561) = 0.08451892915205214). Test tolerances only; the analysis is unchanged."

This is Annex A §4 verbatim. smoke_mde's `make_d1` and `make_d2` and its D1a are this construction (§5).

### 10.4 The Ch4 sentence (method text, not Ch1–3)

"The planning MDE of AM-17 item 3 is computed under a constant-shift null that treats exact ties as shifted:
once δ > 0, every image on which two seeds score identically counts as a positive difference, which deflates
MDE_W, so each seed pair's MDE_W is read together with its share of exact-zero differences (lane 6 (f))."

### 10.5 Lane report note (Annex A §4)

D2's reachable ratio is 1.065 at δ = 0.009, not the plan's indicative 1.010. share_ties and n_zero stand next to
every MDE_W: in the entry's `pairs`, in the CLI's per-pair lines and in the DL line.

### 10.6 Band, at the α selection (Annex A §3, §7.4; REPOSITORY-PROVEN at 31b53b7)

"`alpha_inputs` reads `sweep['band']['file']` through `require_committed` (committed at HEAD, byte-equal;
band_missing/band_uncommitted) and refuses `band_changed_after_launch` unless the blob at the α-default's launch
records_commit equals HEAD's (sweep_select.py:1938-1949); `--band` is optional and must resolve to that path
(select_alpha.py:119; :1941-1942)." The docs commit that adds `reports/derived/dl27_band.json` therefore precedes
the λ launch and the α = 50 launch's records commit (R6). The file is never amended after that commit.

## 11. The four outputs: schemas and one synthetic example each

All four are written by `td.json_bytes`: JSON with indent 2, `ensure_ascii=False` and `allow_nan=False`, LF
line ends and a final newline. Each is serialized before any file is created and created exclusively. None
records an out-dir, a hostname or a wall time, so a rerun with the same inputs, code and stamp is
byte-identical (C4a–b, A7, T7).

Real runs write the names below. Synthetic runs prefix them with `synthetic_` and set `artifact_status`
"smoke". A real run's status is "provisional": the docs commit makes the four files records (§10).

The examples come from STEP 6's final smoke runs at `3b5a418` (C2c, the final code commit). They are synthetic fixture values, not
results.

### 11.1 `mde_entry_<UTC>.json` (schema `plantseg-mde-entry/1.0.0`)

| key | content |
|---|---|
| schema, lane, script, authority, artifact_status, synthetic_input_data, generated_utc | identity |
| artifacts.{42,43,44} | run_id, dir_name, sha256s (the 4 artifact files), checkpoint_sha256, repo_commit, governed_paths_clean, metric_impl_sha256, device_class [env.device, model device type, gpu_name], image_digest, batch_size, forward_batches, cudnn_version, torch_cuda, all_class_miou, checkpoint_best_val_miou_all_class, checkpoint_iteration, checkpoint_path_basename, artifact_schema_version, role; "42" adds dl17_delta |
| provenance_checks | [[name, passed, detail], …]: one commit, one device class, inputs on the model device and the policy applied (each seed), one image digest, the DL-17 validity, the DL17_BAND delta, and the optional MANIFEST pins. All passed, or nothing is written |
| n_val, n_included, K_excluded, excluded_ids_sha256, am5_rule, am5_source, included_ids_sha256, included_ids_order | AM-5 and the rows |
| n_planning, n_planning_rule, grid, resampling, test | the method (item 3(b)–(c); lane 6 (a); resampling with drawn_once_per_pair, equivalent_to, guard_points_share_rows) |
| pairs.{43-42,44-42,44-43} | baseline, candidate, m, n, n_zero, share_ties, mean, sd, rejections [51 ints], power_curve [[δ, power] × 51], power_min_count, mde_w_index, mde_w |
| mde_w, mde_w_pair, sd_delta, dz_mde, dz_formula, mde_t, mde_t_range, tau_p, power_caveat | the results (item 3(c), (d), (f)) |
| label, ties_note | item 3(e) verbatim; lane 6 (f) |
| derivation_check | rule, rows_checked, max_abs_diff, tolerance, mismatches (decision (i)) |
| dl27_band | file, sha256, e1_best_val, s, band, floor, rule, best_json {seed: sha256, best_ckpt_basename, best_ckpt_basename_matches_checkpoint_path}, cross_checks, reader_trace |
| scipy_version, numpy_version, software_environment, seed, git_commit, code, script_commit, script_commit_dl_id | provenance |
| decision_log_line | the DL row's decision text without the entry's own sha256 (§10.1) |

<details><summary>Synthetic example (smoke_mde C1)</summary>

```json
{
  "schema": "plantseg-mde-entry/1.0.0",
  "lane": "L-AM17-MDE",
  "script": "scripts/mde_entry.py",
  "authority": "AM-17 item 3 (docs/PREREGISTRATION_AMENDMENTS.md:393-418); docs/lane_specs/part1.md lane 6; DL-27 / AM-16 item 2 (the alpha tie band)",
  "artifact_status": "smoke",
  "synthetic_input_data": true,
  "generated_utc": "2026-10-09T00:00:00Z",
  "artifacts": {
    "42": {
      "run_id": "fxA-e1-s42",
      "dir_name": "e1_s42",
      "sha256s": {
        "MANIFEST.sha256": "307852a5ab53c78badd436c5cd421a67e4c8c3acba8aa0549cf06805d373105f",
        "per_image.jsonl": "473290c2a15ee669365d236d87493ad7c4947cc4c0cfa49d92a952dec9320351",
        "sufficient_stats.npz": "36b6a54226c5a87e4ba341801a0c716ab749e6ab30d19b119244724a93b9b20c",
        "summary.json": "7d83a93390ee21f80a6c3dfe0fb2763c41701f088c8d97aad16141716ae25985"
      },
      "checkpoint_sha256": "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03",
      "repo_commit": "d0fd2e2eb438c463bed7d51238660db3a3a02485",
      "governed_paths_clean": true,
      "metric_impl_sha256": "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9",
      "device_class": [
        "cuda",
        "cuda",
        "NVIDIA A40"
      ],
      "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
      "batch_size": 16,
      "forward_batches": 53,
      "cudnn_version": 8902,
      "torch_cuda": "12.1",
      "all_class_miou": 0.36314016580581665,
      "checkpoint_best_val_miou_all_class": 0.36314016580581665,
      "checkpoint_iteration": 80000,
      "checkpoint_path_basename": "e1_student_best_iter80000.pt",
      "artifact_schema_version": "plantseg-eval-artifact/1.2.0",
      "role": "DL-17 B66 re-score (run1)",
      "dl17_delta": 0.0
    },
    "43": {
      "run_id": "fxA-e1-s43",
      "dir_name": "e1_s43",
      "sha256s": {
        "MANIFEST.sha256": "74a6860e12549c387da7ac08ac664a1f8bb260f6f408f7e14b572b206f35ba03",
        "per_image.jsonl": "eacff74ec76c1cdbb4e9dbaca16e549d78dadd4f578a6aa1257afc72f3cb8175",
        "sufficient_stats.npz": "f723cc9f75f8cd1b2be18881f334316272a6b163b188a49fed7041c1f06267fd",
        "summary.json": "e7fde5262bd1bf7c08337e439701410fa1c06fa5f3295925603ae90dddd9cc1c"
      },
      "checkpoint_sha256": "8e1e696d9ec27e102e36fa3af4040b792b5749b8c183eb50b25ae455fd698ae1",
      "repo_commit": "d0fd2e2eb438c463bed7d51238660db3a3a02485",
      "governed_paths_clean": true,
      "metric_impl_sha256": "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9",
      "device_class": [
        "cuda",
        "cuda",
        "NVIDIA A40"
      ],
      "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
      "batch_size": 16,
      "forward_batches": 53,
      "cudnn_version": 8902,
      "torch_cuda": "12.1",
      "all_class_miou": 0.5999314785003662,
      "checkpoint_best_val_miou_all_class": 0.34126096963882446,
      "checkpoint_iteration": 80000,
      "checkpoint_path_basename": "e1_student_best_iter80000.pt",
      "artifact_schema_version": "plantseg-eval-artifact/1.2.0",
      "role": "in-chain A40 re-score"
    },
    "44": {
      "run_id": "fxA-e1-s44",
      "dir_name": "e1_s44",
      "sha256s": {
        "MANIFEST.sha256": "5214b03b4c39c8060c8989f3dc349521603a1ef24b03416414cce5622df1d883",
        "per_image.jsonl": "574523b2b6b18ded93f87990abe3e5fc3401c06858387714992555670dc65bd1",
        "sufficient_stats.npz": "df8efe515736f11bfe50d1df41f5ddcc13df96d9dafee1bf37a92e7ed1a44e00",
        "summary.json": "84194ddce77de4a4f0ba06b7b8cdcc30e7f2f4542ded26ca4288fab9a98a4ff3"
      },
      "checkpoint_sha256": "05a7e3569cfc89da80dde6238c05ff1780f43d30f08ed6288e8ddac8aad111c4",
      "repo_commit": "d0fd2e2eb438c463bed7d51238660db3a3a02485",
      "governed_paths_clean": true,
      "metric_impl_sha256": "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9",
      "device_class": [
        "cuda",
        "cuda",
        "NVIDIA A40"
      ],
      "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
      "batch_size": 16,
      "forward_batches": 53,
      "cudnn_version": 8902,
      "torch_cuda": "12.1",
      "all_class_miou": 0.5975386500358582,
      "checkpoint_best_val_miou_all_class": 0.3548465073108673,
      "checkpoint_iteration": 80000,
      "checkpoint_path_basename": "e1_student_best_iter80000.pt",
      "artifact_schema_version": "plantseg-eval-artifact/1.2.0",
      "role": "in-chain A40 re-score"
    }
  },
  "provenance_checks": [
    [
      "one evaluate_model.py commit (run.repo_commit) across the three re-scores (AM-17 item 3(a): 'at one pinned commit')",
      true,
      {
        "42": "d0fd2e2eb438c463bed7d51238660db3a3a02485",
        "43": "d0fd2e2eb438c463bed7d51238660db3a3a02485",
        "44": "d0fd2e2eb438c463bed7d51238660db3a3a02485"
      }
    ],
    [
      "one device class (run.env.device, model device type, gpu_name) across the three (AM-17 item 3(a): 'on the same device')",
      true,
      {
        "42": [
          "cuda",
          "cuda",
          "NVIDIA A40"
        ],
        "43": [
          "cuda",
          "cuda",
          "NVIDIA A40"
        ],
        "44": [
          "cuda",
          "cuda",
          "NVIDIA A40"
        ]
      }
    ],
    [
      "s42: inputs on the model device (input_devices == [model_device]); a consistency check implied by 'one pinned commit': check_post_eval enforces it (src/eval/eval_runtime.py:195-205)",
      true,
      {
        "model_device": "cuda:0",
        "input_devices": [
          "cuda:0"
        ]
      }
    ],
    [
      "s42: the determinism policy applied; a consistency check implied by 'one pinned commit': the policy is unconditional for an FP32 student on CUDA (src/eval/eval_runtime.py:88-91)",
      true,
      true
    ],
    [
      "s43: inputs on the model device (input_devices == [model_device]); a consistency check implied by 'one pinned commit': check_post_eval enforces it (src/eval/eval_runtime.py:195-205)",
      true,
      {
        "model_device": "cuda:0",
        "input_devices": [
          "cuda:0"
        ]
      }
    ],
    [
      "s43: the determinism policy applied; a consistency check implied by 'one pinned commit': the policy is unconditional for an FP32 student on CUDA (src/eval/eval_runtime.py:88-91)",
      true,
      true
    ],
    [
      "s44: inputs on the model device (input_devices == [model_device]); a consistency check implied by 'one pinned commit': check_post_eval enforces it (src/eval/eval_runtime.py:195-205)",
      true,
      {
        "model_device": "cuda:0",
        "input_devices": [
          "cuda:0"
        ]
      }
    ],
    [
      "s44: the determinism policy applied; a consistency check implied by 'one pinned commit': the policy is unconditional for an FP32 student on CUDA (src/eval/eval_runtime.py:88-91)",
      true,
      true
    ],
    [
      "one non-null image digest across the three (lane 6 (a) 'in-chain A40 re-scores' and DL-17's pinned image)",
      true,
      {
        "42": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
        "43": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
        "44": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
      }
    ],
    [
      "s42 is a valid DL-17 B66 re-score (scripts/compare_eval_artifacts.py validity_problems)",
      true,
      []
    ],
    [
      "s42 all-class VAL mIoU within DL17_BAND 0.0001 of the DL-17 reference 0.36314016580581665 (DL-17 PASS)",
      true,
      {
        "all_class_miou": 0.36314016580581665,
        "abs_delta": 0.0
      }
    ],
    [
      "s42: the sha256 of MANIFEST.sha256 equals --expect-manifest-sha256-42",
      true,
      {
        "got": "307852a5ab53c78badd436c5cd421a67e4c8c3acba8aa0549cf06805d373105f",
        "expected": "307852a5ab53c78badd436c5cd421a67e4c8c3acba8aa0549cf06805d373105f"
      }
    ],
    [
      "s43: the sha256 of MANIFEST.sha256 equals --expect-manifest-sha256-43",
      true,
      {
        "got": "74a6860e12549c387da7ac08ac664a1f8bb260f6f408f7e14b572b206f35ba03",
        "expected": "74a6860e12549c387da7ac08ac664a1f8bb260f6f408f7e14b572b206f35ba03"
      }
    ],
    [
      "s44: the sha256 of MANIFEST.sha256 equals --expect-manifest-sha256-44",
      true,
      {
        "got": "5214b03b4c39c8060c8989f3dc349521603a1ef24b03416414cce5622df1d883",
        "expected": "5214b03b4c39c8060c8989f3dc349521603a1ef24b03416414cce5622df1d883"
      }
    ]
  ],
  "n_val": 846,
  "n_included": 840,
  "K_excluded": 6,
  "excluded_ids_sha256": "dcbde6dc29aaf7c6a726ab0502ec391b9b52dd4351d2296162324ba1f9ca1764",
  "am5_rule": "no_disease_gt",
  "am5_source": {
    "42": "artifact",
    "43": "artifact",
    "44": "artifact"
  },
  "included_ids_sha256": "2edcfe2cd3762e9a57caf4e8b452db7f913da91e7ee444f418dd4f46abb1acea",
  "included_ids_order": "sorted image_id (src/stats/align.py align_runs)",
  "n_planning": 1550,
  "n_planning_rule": "round(Fraction(1561 * 840, 846)) = round(1311240/846) = 1550 (exact rational, ties to even)",
  "grid": {
    "delta": "k / 1000 for k = 0..50 (0.000 ... 0.050)",
    "points": 51,
    "candidates": "k = 0..48 (the guard needs k + 1 and k + 2 on the grid)"
  },
  "resampling": {
    "B": 2000,
    "size": 1550,
    "rng": "numpy.random.default_rng(42).integers(0, m, size=(B, n), dtype=numpy.int64)",
    "drawn_once_per_pair": true,
    "equivalent_to": "default_rng(42) re-seeded per (pair, delta): the draw depends neither on delta nor on the values, so default_rng(42).choice(d + delta, size=(B, n), replace=True) == d[idx] + delta",
    "guard_points_share_rows": true,
    "numpy_version": "1.26.4"
  },
  "test": {
    "call": "scipy.stats.wilcoxon(x, zero_method='pratt', alternative='greater', correction=True, method='approx')",
    "alpha": 0.00625,
    "rejection": "p < alpha",
    "power": "rejections / B",
    "guard": "power >= 0.80 at delta and at the next two grid points (MC noise guard over shared rows)",
    "power_min_count": 1600,
    "centring": "none: d + delta (item 3(c); lane 6 (a))"
  },
  "pairs": {
    "43-42": {
      "baseline": "s42",
      "candidate": "s43",
      "m": 840,
      "n": 1550,
      "n_zero": 0,
      "share_ties": 0.0,
      "mean": 0.004963863144318263,
      "sd": 0.1755460863759459,
      "rejections": [
        289,
        410,
        528,
        691,
        873,
        1043,
        1218,
        1378,
        1526,
        1668,
        1760,
        1837,
        1893,
        1935,
        1962,
        1979,
        1987,
        1994,
        1999,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000
      ],
      "power_curve": [
        [
          0.0,
          0.1445
        ],
        [
          0.001,
          0.205
        ],
        [
          0.002,
          0.264
        ],
        [
          0.003,
          0.3455
        ],
        [
          0.004,
          0.4365
        ],
        [
          0.005,
          0.5215
        ],
        [
          0.006,
          0.609
        ],
        [
          0.007,
          0.689
        ],
        [
          0.008,
          0.763
        ],
        [
          0.009,
          0.834
        ],
        [
          0.01,
          0.88
        ],
        [
          0.011,
          0.9185
        ],
        [
          0.012,
          0.9465
        ],
        [
          0.013,
          0.9675
        ],
        [
          0.014,
          0.981
        ],
        [
          0.015,
          0.9895
        ],
        [
          0.016,
          0.9935
        ],
        [
          0.017,
          0.997
        ],
        [
          0.018,
          0.9995
        ],
        [
          0.019,
          1.0
        ],
        [
          0.02,
          1.0
        ],
        [
          0.021,
          1.0
        ],
        [
          0.022,
          1.0
        ],
        [
          0.023,
          1.0
        ],
        [
          0.024,
          1.0
        ],
        [
          0.025,
          1.0
        ],
        [
          0.026,
          1.0
        ],
        [
          0.027,
          1.0
        ],
        [
          0.028,
          1.0
        ],
        [
          0.029,
          1.0
        ],
        [
          0.03,
          1.0
        ],
        [
          0.031,
          1.0
        ],
        [
          0.032,
          1.0
        ],
        [
          0.033,
          1.0
        ],
        [
          0.034,
          1.0
        ],
        [
          0.035,
          1.0
        ],
        [
          0.036,
          1.0
        ],
        [
          0.037,
          1.0
        ],
        [
          0.038,
          1.0
        ],
        [
          0.039,
          1.0
        ],
        [
          0.04,
          1.0
        ],
        [
          0.041,
          1.0
        ],
        [
          0.042,
          1.0
        ],
        [
          0.043,
          1.0
        ],
        [
          0.044,
          1.0
        ],
        [
          0.045,
          1.0
        ],
        [
          0.046,
          1.0
        ],
        [
          0.047,
          1.0
        ],
        [
          0.048,
          1.0
        ],
        [
          0.049,
          1.0
        ],
        [
          0.05,
          1.0
        ]
      ],
      "power_min_count": 1600,
      "mde_w_index": 9,
      "mde_w": 0.009
    },
    "44-42": {
      "baseline": "s42",
      "candidate": "s44",
      "m": 840,
      "n": 1550,
      "n_zero": 0,
      "share_ties": 0.0,
      "mean": -0.004235425751124109,
      "sd": 0.1732499340624324,
      "rejections": [
        0,
        1,
        2,
        3,
        7,
        11,
        19,
        36,
        59,
        104,
        155,
        243,
        330,
        453,
        592,
        755,
        945,
        1129,
        1294,
        1445,
        1589,
        1712,
        1805,
        1869,
        1919,
        1955,
        1976,
        1982,
        1989,
        1992,
        1996,
        1998,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000
      ],
      "power_curve": [
        [
          0.0,
          0.0
        ],
        [
          0.001,
          0.0005
        ],
        [
          0.002,
          0.001
        ],
        [
          0.003,
          0.0015
        ],
        [
          0.004,
          0.0035
        ],
        [
          0.005,
          0.0055
        ],
        [
          0.006,
          0.0095
        ],
        [
          0.007,
          0.018
        ],
        [
          0.008,
          0.0295
        ],
        [
          0.009,
          0.052
        ],
        [
          0.01,
          0.0775
        ],
        [
          0.011,
          0.1215
        ],
        [
          0.012,
          0.165
        ],
        [
          0.013,
          0.2265
        ],
        [
          0.014,
          0.296
        ],
        [
          0.015,
          0.3775
        ],
        [
          0.016,
          0.4725
        ],
        [
          0.017,
          0.5645
        ],
        [
          0.018,
          0.647
        ],
        [
          0.019,
          0.7225
        ],
        [
          0.02,
          0.7945
        ],
        [
          0.021,
          0.856
        ],
        [
          0.022,
          0.9025
        ],
        [
          0.023,
          0.9345
        ],
        [
          0.024,
          0.9595
        ],
        [
          0.025,
          0.9775
        ],
        [
          0.026,
          0.988
        ],
        [
          0.027,
          0.991
        ],
        [
          0.028,
          0.9945
        ],
        [
          0.029,
          0.996
        ],
        [
          0.03,
          0.998
        ],
        [
          0.031,
          0.999
        ],
        [
          0.032,
          1.0
        ],
        [
          0.033,
          1.0
        ],
        [
          0.034,
          1.0
        ],
        [
          0.035,
          1.0
        ],
        [
          0.036,
          1.0
        ],
        [
          0.037,
          1.0
        ],
        [
          0.038,
          1.0
        ],
        [
          0.039,
          1.0
        ],
        [
          0.04,
          1.0
        ],
        [
          0.041,
          1.0
        ],
        [
          0.042,
          1.0
        ],
        [
          0.043,
          1.0
        ],
        [
          0.044,
          1.0
        ],
        [
          0.045,
          1.0
        ],
        [
          0.046,
          1.0
        ],
        [
          0.047,
          1.0
        ],
        [
          0.048,
          1.0
        ],
        [
          0.049,
          1.0
        ],
        [
          0.05,
          1.0
        ]
      ],
      "power_min_count": 1600,
      "mde_w_index": 21,
      "mde_w": 0.021
    },
    "44-43": {
      "baseline": "s43",
      "candidate": "s44",
      "m": 840,
      "n": 1550,
      "n_zero": 0,
      "share_ties": 0.0,
      "mean": -0.009199288895442373,
      "sd": 0.17119572828535815,
      "rejections": [
        0,
        0,
        1,
        1,
        2,
        3,
        7,
        15,
        20,
        37,
        69,
        106,
        170,
        252,
        359,
        488,
        654,
        802,
        1021,
        1211,
        1348,
        1499,
        1624,
        1733,
        1822,
        1884,
        1923,
        1952,
        1975,
        1990,
        1996,
        1999,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000,
        2000
      ],
      "power_curve": [
        [
          0.0,
          0.0
        ],
        [
          0.001,
          0.0
        ],
        [
          0.002,
          0.0005
        ],
        [
          0.003,
          0.0005
        ],
        [
          0.004,
          0.001
        ],
        [
          0.005,
          0.0015
        ],
        [
          0.006,
          0.0035
        ],
        [
          0.007,
          0.0075
        ],
        [
          0.008,
          0.01
        ],
        [
          0.009,
          0.0185
        ],
        [
          0.01,
          0.0345
        ],
        [
          0.011,
          0.053
        ],
        [
          0.012,
          0.085
        ],
        [
          0.013,
          0.126
        ],
        [
          0.014,
          0.1795
        ],
        [
          0.015,
          0.244
        ],
        [
          0.016,
          0.327
        ],
        [
          0.017,
          0.401
        ],
        [
          0.018,
          0.5105
        ],
        [
          0.019,
          0.6055
        ],
        [
          0.02,
          0.674
        ],
        [
          0.021,
          0.7495
        ],
        [
          0.022,
          0.812
        ],
        [
          0.023,
          0.8665
        ],
        [
          0.024,
          0.911
        ],
        [
          0.025,
          0.942
        ],
        [
          0.026,
          0.9615
        ],
        [
          0.027,
          0.976
        ],
        [
          0.028,
          0.9875
        ],
        [
          0.029,
          0.995
        ],
        [
          0.03,
          0.998
        ],
        [
          0.031,
          0.9995
        ],
        [
          0.032,
          1.0
        ],
        [
          0.033,
          1.0
        ],
        [
          0.034,
          1.0
        ],
        [
          0.035,
          1.0
        ],
        [
          0.036,
          1.0
        ],
        [
          0.037,
          1.0
        ],
        [
          0.038,
          1.0
        ],
        [
          0.039,
          1.0
        ],
        [
          0.04,
          1.0
        ],
        [
          0.041,
          1.0
        ],
        [
          0.042,
          1.0
        ],
        [
          0.043,
          1.0
        ],
        [
          0.044,
          1.0
        ],
        [
          0.045,
          1.0
        ],
        [
          0.046,
          1.0
        ],
        [
          0.047,
          1.0
        ],
        [
          0.048,
          1.0
        ],
        [
          0.049,
          1.0
        ],
        [
          0.05,
          1.0
        ]
      ],
      "power_min_count": 1600,
      "mde_w_index": 22,
      "mde_w": 0.022
    }
  },
  "mde_w": 0.022,
  "mde_w_pair": "44-43",
  "sd_delta": 0.1755460863759459,
  "dz_mde": 0.08481830481834723,
  "dz_formula": "(2.4977 + 0.8416) / sqrt(n_planning)",
  "mde_t": 0.014889521463902893,
  "mde_t_range": [
    0.015231980457572658,
    0.016021125095159513
  ],
  "tau_p": 0.01,
  "power_caveat": true,
  "label": "planning proxy — seed-pair differences approximate noise, not the spread of between-recipe differences, so the true MDE may be larger. Reported with the results; it changes no test or decision.",
  "ties_note": "lane 6 (f): the constant-shift null removes Pratt zeros (the pre-registered approximation): every exact-zero difference becomes +delta for delta > 0, so exact ties deflate MDE_W; read MDE_W with each pair's share_ties and n_zero",
  "derivation_check": {
    "rule": "per image: mean over classes 1..115 with gt > 0 of float32 tp / (gt + pred - tp), from sufficient_stats.npz, against per_image.jsonl disease_only_miou",
    "rows_checked": 2520,
    "max_abs_diff": 0.0,
    "tolerance": 1e-06,
    "mismatches": 0
  },
  "dl27_band": {
    "file": "synthetic_dl27_band.json",
    "sha256": "3f33ab525037e9744b21b9abc19b60190f58f7832e3006783d759ec23ddf6ee4",
    "e1_best_val": {
      "42": 0.36314016580581665,
      "43": 0.34126096963882446,
      "44": 0.3548465073108673
    },
    "s": 0.011045744504733232,
    "band": 0.015621041685101825,
    "floor": 0.005,
    "rule": "band = max(floor, sqrt(2) * s); s = statistics.stdev of the three values (sweep_select.dl27_band)",
    "best_json": {
      "42": {
        "sha256": "3b9315ed9363db27b018821a91715793e57273a3014a3fb0684b59bc5894f14b",
        "best_ckpt_basename": "e1_student_best_iter80000.pt",
        "best_ckpt_basename_matches_checkpoint_path": true
      },
      "43": {
        "sha256": "59d5ebd14f113a97bd2535413d2d42c8c16429f51884ce3f6d9ffd63c7d6f8b0",
        "best_ckpt_basename": "e1_student_best_iter80000.pt",
        "best_ckpt_basename_matches_checkpoint_path": true
      },
      "44": {
        "sha256": "b00c0331437440fa905b4f44f66d79cc6ef6c0d3e9cabd6527da0c7801b4d178",
        "best_ckpt_basename": "e1_student_best_iter80000.pt",
        "best_ckpt_basename_matches_checkpoint_path": true
      }
    },
    "cross_checks": [
      [
        "s42: best.json best_val_miou_all_class == the artifact's eval_runtime.checkpoint_best_val_miou_all_class",
        true,
        {
          "best_json": 0.36314016580581665,
          "artifact": 0.36314016580581665
        }
      ],
      [
        "s43: best.json best_val_miou_all_class == the artifact's eval_runtime.checkpoint_best_val_miou_all_class",
        true,
        {
          "best_json": 0.34126096963882446,
          "artifact": 0.34126096963882446
        }
      ],
      [
        "s44: best.json best_val_miou_all_class == the artifact's eval_runtime.checkpoint_best_val_miou_all_class",
        true,
        {
          "best_json": 0.3548465073108673,
          "artifact": 0.3548465073108673
        }
      ]
    ],
    "reader_trace": [
      "band: s = 0.011045744504733232 (sample SD, n=3, E1 seeds 42-44 = [0.36314016580581665, 0.34126096963882446, 0.3548465073108673]); sqrt(2)*s = 0.015621041685101825; band = max(0.005, sqrt(2)*s) = 0.015621041685101825"
    ]
  },
  "scipy_version": "1.11.4",
  "numpy_version": "1.26.4",
  "software_environment": {
    "pinned": {
      "python": "3.11",
      "numpy": "1.26.4",
      "scipy": "1.11.4",
      "statsmodels": "0.14.6"
    },
    "observed": {
      "python": "3.11",
      "numpy": "1.26.4",
      "scipy": "1.11.4",
      "statsmodels": "0.14.6"
    },
    "matches_pinned": true
  },
  "seed": 42,
  "git_commit": "3b5a4189c08ea17f7fb4b7b4a55ba38ede9b7efe",
  "code": {
    "code_commit": "3b5a4189c08ea17f7fb4b7b4a55ba38ede9b7efe",
    "code_clean_at_commit": true,
    "code_files": {
      "scripts/mde_entry.py": "8de3c79f8ec1eedd13fa6049a2cf5141dea5c21e0ed190297fce7cbe3f7bc108",
      "src/stats/mde.py": "08c6decdb2645f3e2d0cbf9087eaecd977a33ad241d1ff5b2bc09d080f939ba9",
      "src/stats/val_artifacts.py": "ea365d3d905cd78206c010bdb9d8476ee112596894d73624cf4718250d1a3aec",
      "src/stats/ingest.py": "bf44a0fafa4ccbca53caaaa8e1e1d429424387eaace6bc6c45b99e5ca8732fca",
      "src/stats/align.py": "61c48956358e1d8b258b412fa5d4dcde29c6d3c46156243f8d68d102dd1df309",
      "src/stats/report.py": "eeb40ceed5e2e32dd7ff8bb691883609a3d6f9a1c887c1a131f94c0d71b7b156",
      "src/stats/artifact.py": "a93cd331ca334cdeea766700b0aec56edb5819a006e1a03cd74674b2bae170d9",
      "src/eval/artifacts.py": "540eea7906d29674d9231bcb53758a20a41ee36cd8bfbcf3207db3905e8bf181",
      "src/eval/evaluate.py": "0b609a0e8e6843940805237b8fd2d76f3a888cbf3141d45b95fae086986789d2",
      "src/eval/metrics.py": "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9",
      "src/eval/adapters.py": "9e7de9f0f56d5fb457a4e67754db1cac23ea57fb7ef4e14ec980785c8baceb0c",
      "src/eval/protocols.py": "e67e75f028bf1a8ddec61d04009491f51a92ceeaedf099fc8d3a67e8b6257bfc",
      "src/eval/teacher_diag.py": "2e0933be4659f4ace3acc1f174a1012f05b76811ffb7261ccd64cd3130bff91b",
      "src/training/sweep_select.py": "8a32479cc1426a1cf0676d163bc503af1982683c6fcb77ebfec4bf5908f2e3cf",
      "scripts/compare_eval_artifacts.py": "94e7c3e57a6c4fda813c71afb6e1fb28329f21996939360192260a8e0eb13818",
      "configs/sweep_rules.json": "ebfff9cc606be613a4011cb0cc0dea9bb90789704a144ac514ea056970eef661",
      "configs/distill.py": "025dab78e4097b48f79cdf273a339a40848747ded540eaba66b32bea2ee14a1e"
    }
  },
  "script_commit": null,
  "script_commit_dl_id": null,
  "decision_log_line": "AM-17 item 3 MDE entry and AM-16 item 2 / DL-27 tie band (one entry, AM-17 item 3). MDE: synthetic_mde_entry_20261009T000000Z.json; E1 s42/s43/s44 VAL artifacts s42 fxA-e1-s42 (MANIFEST.sha256 sha256 307852a5ab53c78badd436c5cd421a67e4c8c3acba8aa0549cf06805d373105f, checkpoint sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03), s43 fxA-e1-s43 (MANIFEST.sha256 sha256 74a6860e12549c387da7ac08ac664a1f8bb260f6f408f7e14b572b206f35ba03, checkpoint sha256 8e1e696d9ec27e102e36fa3af4040b792b5749b8c183eb50b25ae455fd698ae1), s44 fxA-e1-s44 (MANIFEST.sha256 sha256 5214b03b4c39c8060c8989f3dc349521603a1ef24b03416414cce5622df1d883, checkpoint sha256 05a7e3569cfc89da80dde6238c05ff1780f43d30f08ed6288e8ddac8aad111c4) at evaluate_model.py commit d0fd2e2eb438c463bed7d51238660db3a3a02485, NVIDIA A40, image sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf; n_included 840 (K 6); n_planning 1550; shifted-null MDE_W 0.022 (43−42 0.009, 44−42 0.021, 44−43 0.022; share of exact-zero differences 0/0/0, n_zero 0/0/0; B 2,000 resamples of size n from default_rng(42) — one index matrix per pair, identical to re-seeding per (pair, δ), so the three guard points share rows; the pre-registered Wilcoxon call, α 0.00625, p < α, 3-point guard; scipy 1.11.4, numpy 1.26.4); SD_Δ 0.175546, dz_MDE 0.0848183, MDE_t 0.0148895 [0.015232, 0.0160211]; τ_P 0.010; power_caveat True; label: planning proxy (item 3(e)); lane 6 (f): exact ties deflate MDE_W under the constant-shift null. Band: synthetic_dl27_band.json sha256 3f33ab525037e9744b21b9abc19b60190f58f7832e3006783d759ec23ddf6ee4; E1 best VAL 42/43/44 0.36314016580581665/0.34126096963882446/0.3548465073108673 (best.json sha256s in the entry); s 0.011045744504733232; band = max(0.005, √2·s) = 0.015621041685101825. Code 3b5a4189c08ea17f7fb4b7b4a55ba38ede9b7efe (None)."
}
```
</details>

### 11.2 `dl27_band.json`

Exactly the reader's keys (sweep_select.dl27_band; select_alpha.py:29-32): `e1_best_val` {"42", "43", "44"}
(floats), `s` (statistics.stdev of the three) and `band` (max(0.005, √2·s)). The provenance sits in the entry's
`dl27_band` block.

```json
{
  "e1_best_val": {
    "42": 0.36314016580581665,
    "43": 0.34126096963882446,
    "44": 0.3548465073108673
  },
  "s": 0.011045744504733232,
  "band": 0.015621041685101825
}
```

### 11.3 `train_val_aspect_ratio_census.json` (schema `plantseg-aspect-ratio-census/1.0.0`)

| key | content |
|---|---|
| schema, census, lane, script, authority, artifact_status, synthetic_input_data, generated_utc | identity |
| code | commit, code_files_clean_at_commit, code_files {path: sha256}, binding {head, script_commit, script_commit_dl_id} or null |
| environment | python, numpy, pillow, image_digest (PLANTSEG_IMAGE_DIGEST), torch |
| inputs | folders, listing, image_rule, expected {train, val}, entries {split: files, dirs, other, symlinks, dot_files} |
| definition, thresholds | AR = long/short of the header size; strictly above 12.6 and 26.5, in integers; B3 sources |
| splits.{train,val}, combined | n, max_ar, max_ar_exact, max_ar_image {split, stem, file, w, h}, count_above_12_6, count_above_26_5, top10, formats, header_warnings |
| dims_list_sha256, dims_list_rule, status | the list of (split, stem, w, h), hashed but not stored |

<details><summary>Synthetic example (smoke_split_censuses A1 tree)</summary>

```json
{
  "schema": "plantseg-aspect-ratio-census/1.0.0",
  "census": "aspect-ratio",
  "lane": "S4",
  "script": "scripts/split_censuses.py",
  "authority": "DL-49 (docs/DECISION_LOG.md:108); docs/IMPLEMENTATION_CONTRACT.md B3 'Zero-valid samples' (:370)",
  "artifact_status": "smoke",
  "synthetic_input_data": true,
  "generated_utc": "2026-10-09T00:00:00Z",
  "code": {
    "commit": "3b5a4189c08ea17f7fb4b7b4a55ba38ede9b7efe",
    "code_files_clean_at_commit": true,
    "code_files": {
      "scripts/split_censuses.py": "5fca2e59cd6ee0c200a84c31892907f693d9e70a3e9296e3dbefc799d9c7c727",
      "scripts/hash_split_files.py": "fdf5a7e2762a85a3b878ff9b51282e5a0b6bea8dae4500f469aee68f5669ce9e",
      "src/eval/teacher_diag.py": "2e0933be4659f4ace3acc1f174a1012f05b76811ffb7261ccd64cd3130bff91b",
      "src/eval/artifacts.py": "540eea7906d29674d9231bcb53758a20a41ee36cd8bfbcf3207db3905e8bf181",
      "src/eval/evaluate.py": "0b609a0e8e6843940805237b8fd2d76f3a888cbf3141d45b95fae086986789d2"
    },
    "binding": null
  },
  "environment": {
    "python": "3.11.17",
    "numpy": "1.26.4",
    "pillow": "12.3.0",
    "image_digest": null,
    "torch": "2.1.0+cpu"
  },
  "inputs": {
    "folders": [
      "images/train",
      "images/val"
    ],
    "listing": "one os.scandir per folder; no recursion; no other folder",
    "image_rule": "os.path.splitext(name)[1].lower() in ('.jpg', '.jpeg') (scripts/hash_split_files.py IMAGE_SUFFIXES; src/eval/adapters.py list_split_stems, which differs only for names made of leading dots and the suffix)",
    "expected": {
      "train": 9,
      "val": 4
    },
    "entries": {
      "train": {
        "files": 9,
        "dirs": 1,
        "other": 0,
        "symlinks": 0,
        "dot_files": 0
      },
      "val": {
        "files": 4,
        "dirs": 0,
        "other": 0,
        "symlinks": 0,
        "dot_files": 0
      }
    }
  },
  "definition": {
    "aspect_ratio": "max(W, H) / min(W, H) of the stored header size (PIL Image.open(...).size; no pixel decoded); invariant under EXIF 90-degree rotations",
    "above": "strictly above, in integers: 10 * long > 126 * short (12.6), 10 * long > 265 * short (26.5)"
  },
  "thresholds": [
    {
      "value": 12.6,
      "source": "B3: every valid 32x32 cell is lost only above aspect ratio about 12.6 at the worst scale r = 0.75 (certain from about 24.8)"
    },
    {
      "value": 26.5,
      "source": "B3: every valid 64x64 cell is lost only above about 26.5"
    }
  ],
  "splits": {
    "train": {
      "n": 9,
      "max_ar": 30.0,
      "max_ar_exact": "300/10",
      "max_ar_image": {
        "split": "train",
        "stem": "p10x300",
        "file": "p10x300.jpg",
        "w": 10,
        "h": 300
      },
      "count_above_12_6": 4,
      "count_above_26_5": 2,
      "top10": [
        {
          "split": "train",
          "stem": "p10x300",
          "w": 10,
          "h": 300,
          "ar": 30.0
        },
        {
          "split": "train",
          "stem": "w266x10",
          "w": 266,
          "h": 10,
          "ar": 26.6
        },
        {
          "split": "train",
          "stem": "w265x10",
          "w": 265,
          "h": 10,
          "ar": 26.5
        },
        {
          "split": "train",
          "stem": "w127x10",
          "w": 127,
          "h": 10,
          "ar": 12.7
        },
        {
          "split": "train",
          "stem": "w126x10",
          "w": 126,
          "h": 10,
          "ar": 12.6
        },
        {
          "split": "train",
          "stem": "sq_0",
          "w": 8,
          "h": 8,
          "ar": 1.0
        },
        {
          "split": "train",
          "stem": "sq_1",
          "w": 8,
          "h": 8,
          "ar": 1.0
        },
        {
          "split": "train",
          "stem": "sq_2",
          "w": 8,
          "h": 8,
          "ar": 1.0
        },
        {
          "split": "train",
          "stem": "up",
          "w": 8,
          "h": 8,
          "ar": 1.0
        }
      ],
      "formats": {
        "JPEG": 9
      },
      "header_warnings": {}
    },
    "val": {
      "n": 4,
      "max_ar": 1.0,
      "max_ar_exact": "8/8",
      "max_ar_image": {
        "split": "val",
        "stem": "v_0",
        "file": "v_0.jpg",
        "w": 8,
        "h": 8
      },
      "count_above_12_6": 0,
      "count_above_26_5": 0,
      "top10": [
        {
          "split": "val",
          "stem": "v_0",
          "w": 8,
          "h": 8,
          "ar": 1.0
        },
        {
          "split": "val",
          "stem": "v_1",
          "w": 8,
          "h": 8,
          "ar": 1.0
        },
        {
          "split": "val",
          "stem": "v_2",
          "w": 8,
          "h": 8,
          "ar": 1.0
        },
        {
          "split": "val",
          "stem": "v_3",
          "w": 8,
          "h": 8,
          "ar": 1.0
        }
      ],
      "formats": {
        "JPEG": 4
      },
      "header_warnings": {}
    }
  },
  "combined": {
    "n": 13,
    "max_ar": 30.0,
    "max_ar_exact": "300/10",
    "max_ar_image": {
      "split": "train",
      "stem": "p10x300",
      "file": "p10x300.jpg",
      "w": 10,
      "h": 300
    },
    "count_above_12_6": 4,
    "count_above_26_5": 2,
    "top10": [
      {
        "split": "train",
        "stem": "p10x300",
        "w": 10,
        "h": 300,
        "ar": 30.0
      },
      {
        "split": "train",
        "stem": "w266x10",
        "w": 266,
        "h": 10,
        "ar": 26.6
      },
      {
        "split": "train",
        "stem": "w265x10",
        "w": 265,
        "h": 10,
        "ar": 26.5
      },
      {
        "split": "train",
        "stem": "w127x10",
        "w": 127,
        "h": 10,
        "ar": 12.7
      },
      {
        "split": "train",
        "stem": "w126x10",
        "w": 126,
        "h": 10,
        "ar": 12.6
      },
      {
        "split": "train",
        "stem": "sq_0",
        "w": 8,
        "h": 8,
        "ar": 1.0
      },
      {
        "split": "train",
        "stem": "sq_1",
        "w": 8,
        "h": 8,
        "ar": 1.0
      },
      {
        "split": "train",
        "stem": "sq_2",
        "w": 8,
        "h": 8,
        "ar": 1.0
      },
      {
        "split": "train",
        "stem": "up",
        "w": 8,
        "h": 8,
        "ar": 1.0
      },
      {
        "split": "val",
        "stem": "v_0",
        "w": 8,
        "h": 8,
        "ar": 1.0
      }
    ],
    "formats": {
      "JPEG": 13
    },
    "header_warnings": {}
  },
  "dims_list_sha256": "28311bf905e24ce676587aadbe7dcc9594db2fb6080942ff15356c4487c3c1dc",
  "dims_list_rule": "sha256 over 'split\\tstem\\tw\\th\\n' per image, TRAIN then VAL, name-sorted",
  "status": "written"
}
```
</details>

### 11.4 `teacher_dataset_suffix_census.json` (schema `plantseg-teacher-suffix-census/1.0.0`)

| key | content |
|---|---|
| schema … generated_utc, code, environment | as 11.3 |
| inputs | folders, listing, teacher_rule, student_rule, expected |
| splits.{train,val} | n_files, n_dirs, n_other, n_symlinks, n_dot_files, suffix_counts (exact, case-sensitive), teacher_rule_count, student_rule_count, student_stems_unique, stem_sets_equal, n_only_teacher, only_teacher (≤ 20), n_only_student, only_student (≤ 20) |
| records | file, records_sha256, expected_sha256, expected_source (DL-68), expected_sha256_matches, rows, iterations, expected_iterations, val_images, val_manifest_sha256 (the distinct values), mIoU_full (recorded, not gated), mIoU_full_note, teacher_of_record_checkpoint_in_rows |
| val_manifest | census, records, equal, rule |
| expectation, expectation_met, status | the DL-55 checks; exit 1 when not met (the census is still written) |

<details><summary>Synthetic example (smoke_split_censuses T1a tree, with the records pin)</summary>

```json
{
  "schema": "plantseg-teacher-suffix-census/1.0.0",
  "census": "teacher-suffix",
  "lane": "S4",
  "script": "scripts/split_censuses.py",
  "authority": "DL-55 (docs/DECISION_LOG.md:114; A1 N3)",
  "artifact_status": "smoke",
  "synthetic_input_data": true,
  "generated_utc": "2026-10-09T00:00:00Z",
  "code": {
    "commit": "3b5a4189c08ea17f7fb4b7b4a55ba38ede9b7efe",
    "code_files_clean_at_commit": true,
    "code_files": {
      "scripts/split_censuses.py": "5fca2e59cd6ee0c200a84c31892907f693d9e70a3e9296e3dbefc799d9c7c727",
      "scripts/hash_split_files.py": "fdf5a7e2762a85a3b878ff9b51282e5a0b6bea8dae4500f469aee68f5669ce9e",
      "src/eval/teacher_diag.py": "2e0933be4659f4ace3acc1f174a1012f05b76811ffb7261ccd64cd3130bff91b",
      "src/eval/artifacts.py": "540eea7906d29674d9231bcb53758a20a41ee36cd8bfbcf3207db3905e8bf181",
      "src/eval/evaluate.py": "0b609a0e8e6843940805237b8fd2d76f3a888cbf3141d45b95fae086986789d2"
    },
    "binding": null
  },
  "environment": {
    "python": "3.11.17",
    "numpy": "1.26.4",
    "pillow": "12.3.0",
    "image_digest": null,
    "torch": "2.1.0+cpu"
  },
  "inputs": {
    "folders": [
      "images/train",
      "images/val"
    ],
    "listing": "one os.scandir per folder; no recursion; no other folder",
    "teacher_rule": "not name.startswith('.') and name.endswith('.jpg') (configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:207 img_suffix, exact and case-sensitive; mmengine LocalBackend.list_dir_or_file skips dot-named files and still enters dot-named directories, which the no-subdirectory check fails [U: transcribed, mmengine is not in the repository])",
    "student_rule": "os.path.splitext(name)[1].lower() in ('.jpg', '.jpeg')",
    "expected": {
      "train": 5,
      "val": 3
    }
  },
  "splits": {
    "train": {
      "n_files": 5,
      "n_dirs": 0,
      "n_other": 0,
      "n_symlinks": 0,
      "n_dot_files": 0,
      "suffix_counts": {
        ".jpg": 5
      },
      "teacher_rule_count": 5,
      "student_rule_count": 5,
      "student_stems_unique": true,
      "stem_sets_equal": true,
      "n_only_teacher": 0,
      "only_teacher": [],
      "n_only_student": 0,
      "only_student": []
    },
    "val": {
      "n_files": 3,
      "n_dirs": 0,
      "n_other": 0,
      "n_symlinks": 0,
      "n_dot_files": 0,
      "suffix_counts": {
        ".jpg": 3
      },
      "teacher_rule_count": 3,
      "student_rule_count": 3,
      "student_stems_unique": true,
      "stem_sets_equal": true,
      "n_only_teacher": 0,
      "only_teacher": [],
      "n_only_student": 0,
      "only_student": []
    }
  },
  "records": {
    "file": "teacher_selection_records.jsonl",
    "records_sha256": "259ff9825d2c41372c51cba4daa0150778dfaa9ddf3ed3260a64f2e4cfc0f8a3",
    "expected_sha256": "259ff9825d2c41372c51cba4daa0150778dfaa9ddf3ed3260a64f2e4cfc0f8a3",
    "expected_source": "DL-68, docs/DECISION_LOG.md:127",
    "expected_sha256_matches": true,
    "rows": 10,
    "iterations": [
      4000,
      8000,
      12000,
      16000,
      20000,
      24000,
      28000,
      32000,
      36000,
      40000
    ],
    "expected_iterations": [
      4000,
      8000,
      12000,
      16000,
      20000,
      24000,
      28000,
      32000,
      36000,
      40000
    ],
    "val_images": [
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3,
      3
    ],
    "val_manifest_sha256": [
      "5fd568acc60eb840272eae8cd12587ff6b4fcfe8d71ac16c53dc6608d35f23f9"
    ],
    "mIoU_full": [
      0.304,
      0.308,
      0.312,
      0.316,
      0.32,
      0.324,
      0.328,
      0.33199999999999996,
      0.33599999999999997,
      0.33999999999999997
    ],
    "mIoU_full_note": "recorded, not gated: compare with DL-68's ten values",
    "teacher_of_record_checkpoint_in_rows": false
  },
  "val_manifest": {
    "census": "5fd568acc60eb840272eae8cd12587ff6b4fcfe8d71ac16c53dc6608d35f23f9",
    "records": "5fd568acc60eb840272eae8cd12587ff6b4fcfe8d71ac16c53dc6608d35f23f9",
    "equal": true,
    "rule": "hash_split_manifest over ManifestEntry(i, stem, stem) for the name-sorted VAL files under the teacher's rule (exact '.jpg', no leading dot): the order the teacher hook hashes (src/training/teacher_components.py:214-224; 'BaseSegDataset sorts by img_path', teacher config :249); mmseg's own listing is not in the repository"
  },
  "expectation": {
    "train_teacher_rule_count_is_5": true,
    "train_student_rule_count_is_5": true,
    "train_stem_sets_equal": true,
    "train_no_subdirectory": true,
    "val_teacher_rule_count_is_3": true,
    "val_student_rule_count_is_3": true,
    "val_stem_sets_equal": true,
    "val_no_subdirectory": true,
    "records_rows_10": true,
    "records_iterations_4000_to_40000": true,
    "records_val_images_3_in_every_row": true,
    "records_val_manifest_uniform": true,
    "records_val_manifest_equals_census": true
  },
  "expectation_met": true,
  "status": "written"
}
```
</details>

## 12. The local block's command lines (the plan's §6 as patched; R4 pins as placeholders)

This is Ice's block: PowerShell on the laptop, the student image, after the merge (R11). Each line below is
exactly what the CLI's argument parser takes. Angle brackets are placeholders filled by the block's GO:

| placeholder | value |
|---|---|
| `<C_S4>` | the checkout's HEAD containing this lane's code; the binding refuses any other HEAD |
| `<MANIFEST_SHA256_S42>`, `<MANIFEST_SHA256_S43>`, `<MANIFEST_SHA256_S44>` | the sha256 of each re-score's `MANIFEST.sha256` (the stats rehearsal measured s42 79b06c9d…, s43 9b4227a5…, s44 e9e03ed4…; F7). The block carries the full values. |
| `<RECORDS_SHA256>` | the run of record's `teacher_selection_records.jsonl` sha256, DL-68 (docs/DECISION_LOG.md:127): 3448daf348f3e9ddf0c1db5bca229599a8b9aae702486ea17d0500f079816cac (F5) |
| `<K of record>` | optional; K_val if a record states it |

The four R4 pins are always passed. The block also checks the same hashes with Get-FileHash before each call
(R4).

**Order:** P0 → P4 → P3 (the pod gates, DL-55 first) → DL-17 PASS → P1+P2.

**P0** (on the laptop):
```
git -C $REPO rev-parse HEAD
git -C $REPO status --porcelain=v1 --untracked-files=all -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
```
The first must print `<C_S4>`; the second must print nothing. The binding refuses dirty `src`, `configs` or
`scripts`.

**Container shape** (the plan's §6, unchanged; one container per step). The repo is mounted read-only at /repo
with `-w /repo`. Only `images\train` and `images\val` are mounted, read-only, at /data/images/train and
/data/images/val. Other inputs are mounted read-only under /in, and the out folder at /out. The container
runs with `--network none -e LANGUAGE=C -e PLANTSEG_IMAGE_DIGEST=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf`.
Git inside the container needs the safe-directory handling of the earlier local blocks; it is not re-derived
here.

**P4, the DL-55 teacher-suffix census** (the records file is mounted at /in/teacher_selection_records.jsonl):
```
python -B scripts/split_censuses.py --census teacher-suffix --data-root /data --teacher-records /in/teacher_selection_records.jsonl --expect-records-sha256 <RECORDS_SHA256> --out-dir /out --script-commit <C_S4> --script-commit-dl-id DL-99
```

**P3, the DL-49 aspect-ratio census** (the same mounts, without the records file):
```
python -B scripts/split_censuses.py --census aspect-ratio --data-root /data --out-dir /out --script-commit <C_S4> --script-commit-dl-id DL-99
```

**DL-17** (run1 and run2 mounted read-only; must exit 0):
```
python -B scripts/compare_eval_artifacts.py /in/s42_run1 /in/s42_run2
```

**P1+P2, the MDE entry and the DL-27 band.** Mount s42 (run1), s43, s44 and the three best.json files
read-only, plus /out:
```
python -B scripts/mde_entry.py --s42 /in/s42 --s43 /in/s43 --s44 /in/s44 --best-json-42 /in/best42.json --best-json-43 /in/best43.json --best-json-44 /in/best44.json --out-dir /out --expect-manifest-sha256-42 <MANIFEST_SHA256_S42> --expect-manifest-sha256-43 <MANIFEST_SHA256_S43> --expect-manifest-sha256-44 <MANIFEST_SHA256_S44> --script-commit <C_S4> --script-commit-dl-id DL-99
```
Append `--expect-k-val <K of record>` only if a record states K_val.

**Outputs (4).** Each is printed as `written: <name> sha256 <hex>`; Ice confirms each with
`(Get-FileHash -Algorithm SHA256 $OUT\<name>).Hash.ToLower()`:
- teacher_dataset_suffix_census.json;
- train_val_aspect_ratio_census.json;
- dl27_band.json;
- mde_entry_<UTC>.json.

P1 also prints three per-pair lines, a summary line and `DL line: …` (§10.1).

**Run time.**
| step | time |
|---|---|
| P4 | < 10 s |
| P3 | about 0.5–2 min (6,213 header reads over a bind mount) |
| DL-17 | < 10 s |
| P1+P2 | about 2.5–3 min on this container (one process, 4 vCPU): one pair's full curve at n = 1,561 took 50 s (two in 100 s, D group), and the C1 subprocess run, 3 pairs at n = 1,550 plus loading, took 159 s (final run); the plan's estimate was 2–6 min |

**STOP conditions.** Report verbatim; no rerun without a ruling.
| | condition |
|---|---|
| S1 | compare_eval_artifacts exits ≠ 0: DL-17 is not PASS (lane 6 (f)) |
| S2 | mde_entry exits 2 with "STOP (lane 6 (f))": the excluded sets differ across seeds |
| S3 | any other exit 2: a provenance mismatch (including an R4 pin), the wrong commit, the environment |
| S4 | mde_entry exits 1: MDE_W not determined on the grid (three rejection-count curves printed; R9 Q3), or a failed cross-check |
| S5 | split_censuses teacher-suffix exits 1: the DL-55 expectation is not met, so STOP before the first pod |
| S6 | any exit 4, or an unreadable header (aspect-ratio exit 1, nothing written) |
| S7 | a printed sha256 ≠ Get-FileHash |
| S8 | scheduling precondition, not a run-time STOP: if the α = 50 launch's records_commit predates the band's docs commit, select_alpha refuses band_changed_after_launch (Annex A §3, §6) |

## 13. Acceptance against the plan's §8, as patched (MEASURED unless marked)

| # | criterion (plan §8, patched by Annex A §3 and the rulings) | result |
|---|---|---|
| 1 | smoke_mde and smoke_split_censuses pass every case (the counts Annex A estimated, ≈83 and ≈49, grew with the review resolutions, §9 item 7) | smoke_mde 129/129 (35/35 mutations killed); smoke_split_censuses 81/81 (18/18) (§5) |
| 2 | The §5 smokes pass unchanged: the same exit code and counts as at `31b53b7` | All eight: the same exit codes and counts as at 31b53b7, including smoke_eval_determinism's baseline 64/65 (§5) |
| 3 | `git diff --stat 31b53b70d0617e4c17959db8e63235b7b75ae011..HEAD -- . <X>` lists only the five new files and the report | At C2c: the five new files only (§4); C3 adds this report, and no other path changes |
| 4 | No created path contains "test"; the TEST refusals are shown on never-created paths | Every refusal of a test path uses a name or resolved path that is never created, and asserts it absent afterwards: R1a–g, R14, A8a, A8b, A12, A13, A14, T8a, T13. The temporary roots are TMPDIR (string and realpath checked) plus a prefix and hex digits (§9 item 8). The workflow agents' slips are in §9 (items 2–5); none created a test-named path |
| 5 | Exit codes and refusals behave as §1 specifies; outputs land only outside the repo | R-, S-, A- and T-groups (0, 1, 2; 4 never expected). Every output goes to an out-dir outside the repository (R6, A8f, tripwired) and under /tmp/s4scratch; nothing was written under reports/derived/ (R11) |
| 6 | Reruns are byte-identical | C4a–b, A7, T7 |
| 7 | The band round-trips through dl27_band; the entry passes validate_mde_entry | B1–B6, S5, C1–C2, C7; C2, S3 |
| 8 | The measured runtime of one pair at n = 1,561 is reported | One pair's full curve at n = 1,561: 50 s on this container (D group: two in 100 s, final run) |
| 9 | Explicit-path staging with `--only`, commits without trailers, one approved push | Every commit used `git add -- <paths>` and `git commit --only -F … -- <paths>`, and `git interpret-trailers --parse` printed nothing after each (§4). The push is STEP 8's one push of lane/s4-mde-band (§4) |

## 14. Open items

- **O1 — the outputs of record.** Ice's local block makes the four outputs of record after the merge (R11), in
  the order P0 → P4 → P3 → DL-17 PASS → P1+P2, with the four R4 pins (§12). STOP conditions S1–S7 apply, and
  S8 is the scheduling precondition.
- **O2 — the records.** The pre-launch docs commit enters §10's texts: DL-99, the DL-27, DL-49 and DL-55 status
  texts, E-54 and the Ch4 sentence (R3, R5, R10). DL-100 is reserved there too. The commit that adds
  `reports/derived/dl27_band.json` precedes the λ launch and the α = 50 launch's records commit, and the band is
  never amended after it (R6; Annex A row 17).
- **O3 — a ruling on governed_paths_clean** (entry-cli-1). Should the entry gate `run.governed_paths_clean is
  True` for the three re-scores? It records the flag per artifact now. The B66 artifact may predate the flag's
  layout. The block can print the three flags before P1+P2.
- **O4 — a ruling on the checkpoint iteration** (entry-cli-2). Should the entry gate
  `eval_runtime.checkpoint_iteration` against the iteration in best.json's `best_ckpt` name
  (`e1_student_best_iter{it}.pt`, train_e1.py:136) for seeds 43 and 44? It records both now. A re-score of
  `last.pt` would pass the value tie, because `last.pt` stores the same best value.
- **O5 — the P1+P2 image.** It must carry the pinned stats stack: python 3.11, numpy 1.26.4, scipy 1.11.4,
  statsmodels 0.14.6 (`PINNED_ENVIRONMENT`). The real-mode gate refuses any other stack, and a missing package
  is now a refusal (exit 2). `requirements-e1.txt`, the student install, lists only torch, torchvision, numpy and
  Pillow. The censuses need torch and Pillow only. The block's GO names the image.
- **O6 — torchvision.** smoke_eval_determinism's R section needs torchvision, which Q1 does not list. It ran
  64/65 (exit 1) at the baseline and in the final run.
- **O7 — the teacher's listing rule is [U].** mmengine's dot-file skip is transcribed. mmseg and mmengine are
  not in the repository or the venv, and the census records each folder's dot-file count. On PlantSeg, a
  folder without dot-named files makes the old and new rules identical.
- **O8 — STEP 1's verbatim outputs** are not in this report (§3.1). Only the reflog and the session's course
  attest them.
- **O9 — share_ties and n_zero in the DL line, with ties present** (r2-mde-2). The CLI fixtures have no exact-zero differences, so the DL line's share_ties and n_zero are checked at 0/0/0 only (C1, C7). The D group checks pair_summary's n_zero 254 and share_ties 254/846 on d1, but not through the CLI. A fixture with ties would pin the line's two fields against a swap.
- **O10 — the 12-hour limit and the container restart.** The container was down from 16:31 to 22:42 UTC (a worker restart), and no work ran in that gap. Phase B ran 13:06–16:31 and from 22:42 UTC to the push (§4): about 3 h 25 min of work before the restart and about 1 h 25 min after it. By the wall clock, Phase B ends just under 11 h after it began, inside the GO's 12 h limit.
