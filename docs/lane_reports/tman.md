# TMAN lane report — B-1 (code tag `tman`)

Session TMAN-B1, a cloud lane under DL-37, on branch `lane/tman`. Written 2026-10-27 as commit C7 of the GO's B-1 STEPS
1–9, amended by rulings TB1–TB20, ADDENDUM 1, RULING TMAN-STEP4-1 and TMAN-B1-RULE-Q.

- **O** (the base, lane Q2's merge, equal to the pin): `2b3b55651e8092fd7fff6e786b73d971096deb17`.
- **Last code commit:** `005a8912c487510321c6c8b7af5b263c8ced03e7` (C6h2). This report is C7 and changes no code.

**Labels.**
- **MEASURED:** I ran it and saw the output.
- **DOCUMENTED:** a ruling, the GO, the plan or a project document says it.
- **REPOSITORY-PROVEN:** git shows it.
- **INFERRED:** my reading or estimate.

Every file:line below was re-read at `005a8912c487510321c6c8b7af5b263c8ced03e7`.

## 1. What B-1 delivers

B-1 adds new files only (REPOSITORY-PROVEN, §2): no existing file is edited.

**Code:**
- eight modules under `src/tman/`;
- the command line `scripts/tday_manifest.py`;
- the catalogue `configs/tday_catalogue.json`, with 69 slots.

**Smokes:** twelve, `scripts/smoke_tman_*.py`.

**What the code does** (DOCUMENTED, the plan §1–§2 as amended):
- builds the TEST-day manifest (`tday_manifest/1`) and validates it in four modes: draft, freeze, unlock and session. There are 84 gates and 89 refusal codes (§7).
- prints and checks the entry files (`tday_entry/1`);
- reads the lane register (`tday_lanes/1`);
- writes the inventory export (`tday-inventory/1`) and reads it back.

**What it never does** (MEASURED from the call audit, §19):
- read TEST data, the issue #11 CSV, a dataset archive or any path holding "test";
- train, use a GPU or download anything (STEP 3's venv install aside).

## 2. Base, commits, pushes and the diff

- **STEP 1** (MEASURED):
  - O equals the pin;
  - the diff `2b3b556..O -- src scripts configs` with the test exclude is empty;
  - `lane/tman` and `lane/test-manifest` were absent on the remote;
  - the attachments' sha256 matched: audit 6bf0375d, interface 3dbd47f6, AM-23 v6 c6fafec1, AM-22 v6 26c1231a.
- **STEP 2** (MEASURED): the chain `git branch lane/tman 2b3b556`, `git symbolic-ref HEAD refs/heads/lane/tman`, then `git checkout --no-overlay HEAD -- . ':(exclude)docs/reference/reference.pdf'` exited 0. HEAD was O and the scoped status was empty.

**Pushes** (MEASURED):

| Push | Commits carried | Authority |
|---|---|---|
| 1 | C1 | the GO |
| 2 | up to `ee5593e` | STATUS-MD, as STATUS-ASK-1 ruled |
| 3 | C6g5 and C6h, `01eb92b` | TMAN-B1-RULE-Q item 1 (a push after every commit) |
| 4 | C6h2, `005a891` | TMAN-B1-RULE-Q item 1 |
| 5 | C7 | STEP 9 (§20) |

`git ls-remote` printed the pushed tip each time.

Every commit (REPOSITORY-PROVEN; no commit carries a trailer):

| # | commit | subject | pushed by |
|---|---|---|---|
| 1 | `13ddad9` | TMAN B-1 C1: manifest schema, catalogue and score-blind readers | push 1 |
| 2 | `3e72c49` | TMAN B-1 C2: completion at an instant and the freeze resolver | push 2 |
| 3 | `7095914` | TMAN B-1 C3: entry files, cut checks, the parent rule and the printers | push 2 |
| 4 | `def8d92` | TMAN B-1 C4: derived artifacts, appends, the isolation record and the pins | push 2 |
| 5 | `d93ac40` | TMAN B-1 C5: the builder, the validator, the command line and the inventory export | push 2 |
| 6 | `0728f93` | TMAN B-1 C6: the mutation smoke | push 2 |
| 7 | `c806c08` | TMAN B-1 C6a: per-slot run attribution for cuts; every slot entered at the freeze (DL-24) | push 2 |
| 8 | `0e9b813` | TMAN B-1 C6b: the AM-3 trigger reader slot; entry kinds by role; named-check kills (DL-24) | push 2 |
| 9 | `be061a6` | TMAN B-1 C6c: cited entry files read and matched; results known; entered day; not-made lane; arm groups (DL-24) | push 2 |
| 10 | `783299f` | TMAN B-1 C6c2: a named rule's record read through the records; the condition per arm (DL-24) | push 2 |
| 11 | `43ab323` | TMAN B-1 C6c3: each printer runs its kind's checks; the fields are the kind's own; the output codes (DL-24) | push 2 |
| 12 | `13d9986` | TMAN B-1 C6c4: the arithmetic's inputs are the slot's, and its references are read again (DL-24) | push 2 |
| 13 | `a49503c` | TMAN B-1 C6d: the QAT rules file and DL-10's lists of record; appended selections and fault reports (DL-24) | push 2 |
| 14 | `0af9178` | TMAN B-1 C6d2: derived inputs against their sources; hashed forms against their records (DL-24) | push 2 |
| 15 | `0f3fe16` | TMAN B-1 C6d3: the made instant is the provenance's; QAT appends after the freeze; appends only grow (DL-24) | push 2 |
| 16 | `a59ab96` | TMAN B-1 C6e: a checkpoint's committed source; an arm teacher's records (DL-24) | push 2 |
| 17 | `23f1cb4` | TMAN B-1 C6e2: aborted entries, a run's identity and the ARM measurement against their records (DL-24) | push 2 |
| 18 | `0b1d122` | TMAN B-1 C6f: the session on the frozen host; one session over the manifest's history (DL-24) | push 2 |
| 19 | `7a64c30` | TMAN B-1 C6f2: the lane register recorded and non-vacuous; each lane's row names it; not-required slots entered cut (DL-24) | push 2 |
| 20 | `9cc14a6` | TMAN B-1 C6f3: the session's push evidence by the DL-109 pattern (DL-24) | push 2 |
| 21 | `6ed00d3` | TMAN B-1 C6g: a "test" path refused as given before any call, and resolved where it is read (DL-24) | push 2 |
| 22 | `a3079ac` | TMAN B-1 C6g2: body blocks held to their keys; no TEST-value key in the body; no sources slot dropped; build writes after validation (DL-24) | push 2 |
| 23 | `28b2acb` | TMAN B-1 C6g3: the inventory export from verified records, read back whole (DL-24) | push 2 |
| 24 | `ee5593e` | TMAN B-1 C6g4: malformed inputs refused by name, never crashed (DL-24) | push 2 |
| 25 | `34518d0` | TMAN B-1 C6g5: the evaluator closure follows config imports; the arm registry is S's; killed workers leave no scratch (DL-24) | push 3 |
| 26 | `01eb92b` | TMAN B-1 C6h: the cases the verifiers found missing -- R-016, the delay's runs, the TB2 pair, G2, the moved deadline, I1 and the IF§10 roots (DL-24) | push 3 |
| 27 | `005a891` | TMAN B-1 C6h2: the score-blindness test covers the unlock, the session and the command line's outputs; the mutation smoke's lines wrapped (DL-24) | push 4 |
| 28 | (this report) | TMAN B-1 C7: the lane report | push 5 (STEP 9) |

The scoped diff from O to the last code commit (REPOSITORY-PROVEN): `git diff --stat O 005a8912c487510321c6c8b7af5b263c8ced03e7 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'`

```
 configs/tday_catalogue.json       | 2044 +++++++++++++++++++++++++++++++
 scripts/smoke_tman_complete.py    |  371 ++++++
 scripts/smoke_tman_cuts.py        |  473 ++++++++
 scripts/smoke_tman_derived.py     |  372 ++++++
 scripts/smoke_tman_determinism.py |  182 +++
 scripts/smoke_tman_mutations.py   | 1461 ++++++++++++++++++++++
 scripts/smoke_tman_noscore.py     |  394 ++++++
 scripts/smoke_tman_parent.py      |  179 +++
 scripts/smoke_tman_paths.py       |  663 ++++++++++
 scripts/smoke_tman_pins.py        |  156 +++
 scripts/smoke_tman_schema.py      |  901 ++++++++++++++
 scripts/smoke_tman_unlock.py      |  684 +++++++++++
 scripts/smoke_tman_validate.py    | 2405 +++++++++++++++++++++++++++++++++++++
 scripts/tday_manifest.py          |  526 ++++++++
 src/tman/__init__.py              |   10 +
 src/tman/completion.py            |  481 ++++++++
 src/tman/entries.py               |  807 +++++++++++++
 src/tman/manifest.py              | 1000 +++++++++++++++
 src/tman/records.py               |  360 ++++++
 src/tman/schedule.py              |  213 ++++
 src/tman/schema.py                |  542 +++++++++
 src/tman/validate.py              | 1718 ++++++++++++++++++++++++++
 22 files changed, 15942 insertions(+)
```

STEP 7's check (MEASURED): `git diff --name-status O HEAD -- src scripts configs ':(exclude,icase)*test*'` printed 22 lines, all `A`.

## 3. The environment and the lock

STEP 3, MEASURED:
- `$SCR/tmanvenv` was made from `/usr/bin/python3.11 -m venv` (Python 3.11.17, pip 24.0, setuptools 79.0.1), constrained by `requirements.lock`, with `+cu121` changed to `+cpu` and pywin32 dropped.
- torch 2.1.0+cpu and torchvision 0.16.0+cpu were installed with `--no-deps` from the PyTorch CPU index.
- mmcv 2.1.0 was built from the PyPI sdist (`MMCV_WITH_OPS=1`); `import mmcv.ops` works.

Against the lock:
- 47 MATCH;
- DIFFER (2): torch 2.1.0+cpu (lock 2.1.0+cu121) and torchvision 0.16.0+cpu (lock 0.16.0+cu121);
- NOT IN LOCK (3): ftfy 6.3.0, pip 24.0, setuptools 79.0.1;
- `pip check` reports nothing broken.

**regex, under RULING TMAN-STEP4-1 item 2** (MEASURED):
- `pip install --no-deps regex==2026.5.9` (the lock's pin) succeeded, and `pip show regex` gives 2026.5.9.
- The freeze diff against `freeze_before.txt` is one added line, `regex==2026.5.9`. Today's freeze shows the same diff.
- The ruling allowed no other install, and none was made.

The environment of record (RULING TMAN-STEP4-1 item 3) is the venv with regex:
- PYTHONDONTWRITEBYTECODE=1 and `python -B`;
- PYTHONPATH set to the checkout;
- TMPDIR=$SCR for smoke_stats_report and smoke_stats_official_fullsize only (EXEMPT-1, §14);
- each smoke run alone.

## 4. Smokes: the baseline at O and the final runs

**Run A** (STEP 4, MEASURED; PYTHONPATH unset):
- select_sweeps PASS 499/499
- preflight_distill FAIL 183/184
- arms_kd FAIL 58/61
- teacher_arms: no RESULT line
- arms_invariance FAIL 15/16
- kd_flip PASS 34/34
- eval_stage_artifacts PASS 66/66
- frozen_blobs OK 10/10
- stats_bootstrap 155/155 and A3b SMOKE OK
- sl1_guard PASS 319/319 (MODE=log)
- qat_runner PASS 214/214
- stats_integrity OK 129/129
- stats_driver OK 55/55
- stats_report OK 63/63
- stats_official_fullsize OK 37/37 (1,245 s)

**Run B** (PYTHONPATH set to the checkout) equals run A except arms_kd PASS 61/61. arms_invariance gave 15/16 in run B too (308 s), and stats_official_fullsize OK 37/37 (1,255 s).

**Run C** (EXEMPT-1, PYTHONPATH plus TMPDIR=$SCR) ran smoke_stats_report (OK 63/63, 102 s) and smoke_stats_official_fullsize (OK 37/37, 1,370 s). Case Z1, which checks that temporary fixtures are removed, passed in both.

**Each run-A count that differs from run B's, with its cause** (MEASURED):
- **smoke_arms_kd 58/61 in run A, 61/61 in run B.**
  - Two record-teacher cases are refused `arguments:pythonpath_not_checkout`.
  - The cause is `scripts/preflight_e1_trainval.py:248–249`, which requires PYTHONPATH to be exactly the checkout. That line is unchanged from O to HEAD.
  - The two refused cases are `rf_record_teacher_arm_with_arm_teacher` and `rf_existing_output_gate_record`.
  - The third, `mut_record_teacher_arm_accepts_arm_teacher`, fails because its base case does not pass unmutated.
- **smoke_teacher_arms had no RESULT line in either run.** It stopped at `ModuleNotFoundError: No module named 'regex'` (mmseg's tokenizer, through `src/training/teacher_arm_components.py:56`). After the ruling's regex install it gives PASS 98/98 (DL-105's count).

**RULING TMAN-STEP4-1** (DOCUMENTED) took run B as the baseline. Its cause of record for arms_invariance, verbatim: "Q2's merge 2b3b556 modified 16 files this smoke pins at 8487551 (ruling TMAN-STEP4-1)." STEP 7 must show the same 15/16 and the same lists; only `added` may grow. `--old <commit>` was not approved and was not used.

**STEP 7, every smoke alone at `005a8912c487510321c6c8b7af5b263c8ced03e7`** (MEASURED; `$SCR/step7/summary_final.txt`):

| smoke | rc | seconds | final lines | expected (baseline run B / C6h2) |
|---|---|---|---|---|
| smoke_select_sweeps | 0 | 62 | RESULT: PASS (499/499) | 499/499 |
| smoke_preflight_distill | 1 | 40 | RESULT: FAIL (183/184) | 183/184 (the fingerprint case) |
| smoke_arms_kd | 0 | 48 | RESULT: PASS (61/61) | 61/61 |
| smoke_teacher_arms | 0 | 27 | RESULT: PASS (98/98) | 98/98 (with regex) |
| smoke_arms_invariance | 1 | 227 | RESULT: FAIL (15/16) | 15/16 with the ruling's lists |
| smoke_kd_flip | 0 | 48 | RESULT: PASS (34/34) | 34/34 |
| smoke_eval_stage_artifacts | 0 | 3 | RESULT: PASS (66/66) | 66/66 |
| smoke_frozen_blobs | 0 | 0 | SUMMARY  10/10 checks passed RESULT: FROZEN BLOBS OK (10/10) | 10/10 |
| smoke_stats_bootstrap | 0 | 11 | SUMMARY  155/155 checks passed RESULT: A3b SMOKE OK | 155/155 + A3b |
| smoke_sl1_guard | 0 | 27 | SUMMARY  319/319 named cases passed (287 decision cases x in-process, log, deny; 32 launcher/settings/invariant checks) RESULT PASS: smoke_sl1_guard 319/319 (MODE=log) | 319/319 |
| smoke_qat_runner | 0 | 93 | RESULT: PASS (214/214) | 214/214 |
| smoke_stats_integrity | 0 | 31 | SUMMARY  129/129 checks passed RESULT: STATS INTEGRITY OK (129/129) | 129/129 |
| smoke_stats_driver | 0 | 7 | SUMMARY  55/55 checks passed RESULT: STATS DRIVER CORE OK (55/55) | 55/55 |
| smoke_stats_report | 0 | 49 | SUMMARY  63/63 checks passed RESULT: STATS REPORT OK (63/63) | 63/63 (TMPDIR=$SCR) |
| smoke_stats_official_fullsize | 0 | 965 | SUMMARY  37/37 checks passed RESULT: STATS OFFICIAL FULL-SIZE OK (37/37; pinned stack, (a) run) | 37/37 (TMPDIR=$SCR) |
| smoke_tman_schema | 0 | 2 | RESULT: PASS (167/167) | 167/167 |
| smoke_tman_paths | 0 | 6 | RESULT: PASS (89/89) | 89/89 |
| smoke_tman_complete | 0 | 2 | RESULT: PASS (90/90) | 90/90 |
| smoke_tman_cuts | 0 | 1 | RESULT: PASS (129/129) | 129/129 |
| smoke_tman_parent | 0 | 2 | RESULT: PASS (33/33) | 33/33 |
| smoke_tman_derived | 0 | 1 | RESULT: PASS (101/101) | 101/101 |
| smoke_tman_pins | 0 | 1 | RESULT: PASS (49/49) | 49/49 |
| smoke_tman_validate | 0 | 205 | RESULT: PASS (369/369) | 369/369 |
| smoke_tman_unlock | 0 | 59 | RESULT: PASS (86/86) | 86/86 |
| smoke_tman_noscore | 0 | 35 | RESULT: PASS (35/35) | 35/35 |
| smoke_tman_determinism | 0 | 11 | RESULT: PASS (18/18) | 18/18 |
| smoke_tman_mutations | 0 | 1000 | RESULT: PASS (429/429) | 429/429 |

STEP 7's notes (MEASURED):
- **preflight_distill's one failure** is the same case as run B's: `f_this_venvs_fingerprint_is_not_the_kd_images_and_its_imports_have_values_of_record`, failing on Python 3.11.17 vs 3.11.16 and torch/torchvision `+cpu` vs `+cu121`. Its detail no longer lists the `import mmseg.apis` regex failure, since regex is now installed.
- **arms_invariance's summary** (helper `$SCR/check_invariance_summary.py`) is an EXACT-MATCH with the ruling:
  - 16 checks, failing only `inv_blob_ids_unchanged`;
  - 223 OLD files; vanished `[]`; C1 paths missing `[]`; added modules shadowing OLD's `[]`;
  - changed: 17, equal to the ruling's 17; changed minus the seam: 16;
  - added: 54. At STEP 4 it was 32; the 22 more are this lane's new files.

**The twelve B-1 smokes against the plan's targets** (DOCUMENTED targets, the plan §5; MEASURED counts):

| smoke | plan target | at C6 (`0728f93`) | final |
|---|---|---|---|
| smoke_tman_schema | 45 | 165 | 167 |
| smoke_tman_cuts | 60 | 91 | 129 |
| smoke_tman_parent | 28 | 33 | 33 |
| smoke_tman_derived | 45 | 94 | 101 |
| smoke_tman_complete | 55 | 90 | 90 |
| smoke_tman_pins | 30 | 49 | 49 |
| smoke_tman_validate | 80 | 144 | 369 |
| smoke_tman_unlock | 25 | 35 | 86 |
| smoke_tman_noscore | 20 | 27 | 35 |
| smoke_tman_determinism | 12 | 18 | 18 |
| smoke_tman_paths | 24 | 34 | 89 |
| smoke_tman_mutations | 40 | 138 | 429 |

**The counts each commit changed** (MEASURED; each commit's own full run, every smoke alone):

| commit | counts that changed (smoke: before -> after) |
|---|---|
| C1 | schema: — -> 165; paths: — -> 13 |
| C2 | complete: — -> 90 |
| C3 | cuts: — -> 91; parent: — -> 33 |
| C4 | derived: — -> 86; pins: — -> 49 |
| C5 | paths: 13 -> 32; derived: 86 -> 94; validate: — -> 144; unlock: — -> 34; noscore: — -> 27; determinism: — -> 18 |
| C6 | paths: 32 -> 34; unlock: 34 -> 35; mutations: — -> 138 |
| C6a | cuts: 91 -> 98; validate: 144 -> 165; mutations: 138 -> 156 |
| C6b | validate: 165 -> 177; unlock: 35 -> 36; mutations: 156 -> 164 |
| C6c | cuts: 98 -> 111; validate: 177 -> 200; unlock: 36 -> 41; mutations: 164 -> 192 |
| C6c2 | cuts: 111 -> 124; validate: 200 -> 206; mutations: 192 -> 203 |
| C6c3 | paths: 34 -> 35; cuts: 124 -> 126; validate: 206 -> 231; unlock: 41 -> 45; mutations: 203 -> 228 |
| C6c4 | cuts: 126 -> 129; validate: 231 -> 238; mutations: 228 -> 237 |
| C6d | paths: 35 -> 37; validate: 238 -> 250; mutations: 237 -> 248 |
| C6d2 | derived: 94 -> 98; validate: 250 -> 289; mutations: 248 -> 272 |
| C6d3 | paths: 37 -> 41; derived: 98 -> 101; validate: 289 -> 305; mutations: 272 -> 285 |
| C6e | validate: 305 -> 316; mutations: 285 -> 297 |
| C6e2 | validate: 316 -> 328; mutations: 297 -> 309 |
| C6f | unlock: 45 -> 54; mutations: 309 -> 316 |
| C6f2 | unlock: 54 -> 61; mutations: 316 -> 323 |
| C6f3 | unlock: 61 -> 85; mutations: 323 -> 347 |
| C6g | paths: 41 -> 79; mutations: 347 -> 365 |
| C6g2 | paths: 79 -> 80; validate: 328 -> 336; mutations: 365 -> 376 |
| C6g3 | paths: 80 -> 82; validate: 336 -> 351; mutations: 376 -> 391 |
| C6g4 | paths: 82 -> 85; validate: 351 -> 361; mutations: 391 -> 404 |
| C6g5 | schema: 165 -> 166; paths: 85 -> 88; validate: 361 -> 363; mutations: 404 -> 415 |
| C6h | schema: 166 -> 167; paths: 88 -> 89; validate: 363 -> 369; unlock: 85 -> 86; mutations: 415 -> 426 |
| C6h2 | noscore: 27 -> 35; mutations: 426 -> 429 |

## 5. Rulings TB1–TB20 at `005a8912c487510321c6c8b7af5b263c8ced03e7`

| TB | the ruling, in brief (DOCUMENTED) | where it is implemented (REPOSITORY-PROVEN) |
|---|---|---|
| TB1 | One execution-context tuple: cpu, batch size 1, 0 workers, T threads, KD image digest, evaluator commit, host. `execution_context_split` refuses a scored artifact whose tuple differs. | `src/tman/schema.py:138` (KD_IMAGE); `src/tman/validate.py:38` (TUPLE_KEYS), `:477` (g_execution_context_split), `:482` |
| TB2 | corrupted is required for E1/E6 at s43/44 when both runs of the seed have a model; sev45 is forbidden for per_seed; E6-KD takes no corrupted scope. | `src/tman/schema.py:154` (pair_s4344), `:158` (e6kd), `:170` (scope_rule); `src/tman/validate.py:1203` (_models), `:1209` (scope_rule_of); `src/tman/manifest.py:818` (the builder's pair test) |
| TB3 | upstream is required for every student slot and the teacher of record; R1/R2 are canvas only; the six calibration-sensitivity E4/E7 keep the upstream forbid. | `src/tman/schema.py:149` (_BASE holds upstream), `:160` (calibration_sensitivity forbids it), `:161` (arm_teacher forbids it) |
| TB4 | dedup is required wherever clean is; corrupted, sev45 and efficiency are required for E1–E7 s42 and the teacher. | `src/tman/schema.py:149` (_BASE: clean, upstream, dedup), `:150` (_ROBUST), `:152` (family_s42), `:153` (teacher_of_record) |
| TB5 | fq_off is required for every selected QAT model of record; its inventory precision is qat_fakequant_off; rows are unique by (slot, precision). | `src/tman/schema.py:80` (SCOPES), `:87` (INVENTORY_PRECISIONS), `:167` (FQ_OFF_BY_PRECISION), `:176–177` (scope_rule); `src/tman/manifest.py:892` (the rows) |
| TB6 | The pin is embedded verbatim; pin_sha256 is recomputed; generator_cells_mismatch uses src.corruption_roles' cells (24, R-055), imported lazily or injected; IF§10's four scripts join the evaluator file set. | `src/tman/manifest.py:561` (cells), `:598` (pin_sha256), `:604` (cells mismatch), `:671–673` (EVALUATOR_ROOTS with IF§10's four; pinned by smoke_tman_paths `evaluator_roots_are_decision_x_s_scripts_and_IF10_s_B79`) |
| TB7 | file_id per slot; entry file names `<kind>__<file_id>__<YYYYMMDD>__<nn>.json`; no ":", none of `<>"/\|?*`, no "test". | `src/tman/schema.py:193` (file_id_of), `:197` (file_id_error), `:208` (created_name_error), `:222` (entry_file_name); `src/tman/entries.py:731` (entry_name_error) |
| TB8 | Classes run, derived and measurement; the AM-23 slots; the relative cut order with E6-R2, E6-R1 and E6-E2 first; per-slot gates; evaluator_stage against src/eval/artifacts.py STAGES. | `src/tman/schema.py:427` (the catalogue), `:440` (Catalogue), `:455` (position), `:458` (cut_place); `configs/tday_catalogue.json`; `src/tman/validate.py:1113` |
| TB9 | Each QAT run has a derived `:sel` slot; roles and scopes sit on `:sel`, completeness on the run; the pilot runs map to E5:s42:sel; the loser has no TEST scope. | `src/tman/manifest.py:169`, `:228` (the clip selection and the pilot's winner); the catalogue's `:sel` slots (§8) |
| TB10 | cut_of_complete_run exempts decision_date; complete_run_omitted is met by any kind; cut_order_position is required only for rule freeze; every terminal kind is accepted at the unlock; arm_latency is dropped. | `src/tman/entries.py:304`, `:453`; `src/tman/validate.py:883` |
| TB11 | The binding reserves references.mde_entry; push evidence by the DL-109 pattern; a clock-offset record before any scoring record. | `src/tman/schema.py:425` (BINDING_RESERVED), `:48` (PUSH_LIST_FORMAT); `src/tman/validate.py:1409` (g_clock_offset_invalid), `:1425` (push_evidence_error), `:1495` (g_push_evidence_invalid) |
| TB12 | A dates provider (AM-22's cutoff and ceiling from the committed text, a fixture until then); rule-aware dating; specs additive per kind, an unknown kind refused. | `src/tman/schedule.py:18`, `:166` (Am22Dates and am22_text) |
| TB13 | QAT fixtures and the allowlist pinned at 2b3b556 (state_nonfinite, checkpoint-stored flags); the rules file's sha256 read at build; a disagreeing flag refused. | `src/tman/completion.py:240`; `src/tman/manifest.py:62`; `src/tman/validate.py:378` |
| TB14 | M12's "(final)" line and the ARM record: a timestamp or presence only; canonical JSON has one definition; FREEZE_DATES are constants with a cross-check; expected_rows_mismatch (≠ 1,561); session_after_deadline (2026-11-08, moved as the resolver moves it). | `src/tman/completion.py:57`, `:310`; `src/tman/schema.py:25`, `:255`; `src/tman/schedule.py:13`; `src/tman/validate.py:306` (g_expected_rows_mismatch), `:1397` (g_session_after_deadline) |
| TB15 | B-3b and every TEST-serving lane IMPLEMENTED before D; before the unlock only under item 1(f); the lane merges once, after L-CORR's second merge. | `src/tman/validate.py:1272` (g_lane_unmet), `:1302` (g_not_required_invalid). The merge is Ice's block. |
| TB16 | S = E; evaluator_files_changed stays as a safeguard. | `src/tman/validate.py:472`, `:1329` (g_evaluator_files_changed); `scripts/tday_manifest.py:123` |
| TB17 | The inventory carries source.sha256, plus converted_artifact_sha256 for INT8; the binding's post-check stays. | `src/tman/manifest.py:893` |
| TB18 | The lane builds the entries, the duplicate-free list, the binding and the inventory, but no report tables; its open IF§11 fields are resolved by ADDENDUM 1. | `src/tman/manifest.py:843` (build_manifest), `:978` (build_inventory); `src/tman/validate.py:1685` (read_inventory) |
| TB19 | Q1 the branch is lane/tman; Q4 M12's "(final)" is read as UTC; Q5 Ice's block; Q6 the arithmetic printer is the record for non-arm runs; Q9 the isolation record, including LQP's ptq_record/1 isolation; Q11 the plan, the GO and a DL row stand in for the spec; Q12 the catalogue is the governed configs/tday_catalogue.json, with a cross-check smoke. | Q4 `src/tman/completion.py:14`; Q6 `src/tman/entries.py:15`; Q9 `src/tman/manifest.py:398`, `src/tman/validate.py:1077`; Q12 `configs/tday_catalogue.json` and smoke_tman_schema's case_catalogue_cross_checks |
| TB20 | The other §4/§9 answers stand. | no code site (the plan's §4 and §9) |

## 6. Formats

All are in `src/tman/schema.py`.

| Format | Constant (file:line) |
|---|---|
| `tday_manifest/1` | MANIFEST_FORMAT `src/tman/schema.py:41` |
| `tday_entry/1` | ENTRY_FORMAT `:42` |
| `tday_lanes/1` | LANES_FORMAT `:43` |
| `tday_catalogue/1` | CATALOGUE_FORMAT `:44` |
| `tday-inventory/1` | INVENTORY_FORMAT `:45` |
| `tday_binding/1` | BINDING_FORMAT `:46` (reserved for B-3b: BINDING_RESERVED `:425`, references.mde_entry and `artifacts`) |
| `tday_isolation/1` | ISOLATION_FORMAT `:47` |
| `tday_push_list/1` | PUSH_LIST_FORMAT `:48` (new in C6f3) |

**Format changes made during the DL-24 fixes** (REPOSITORY-PROVEN):
- **Arithmetic command line:** `--reference-kind` and `--reference-stage` are removed, and `--iterations` is optional (C6c4). The slot gives the reference's kind and stage; the iterations must equal the slot's horizon.
- **New output codes** `output_location` and `output_exists` (C6c3).
- **QAT inputs:** `clip_selection_sha256` became `clip_selection {path, sha256}` (C6d2).
- **`session.host`** became `{name, cpu_model}` (C6f).
- **GATES order:** `clock_offset_invalid` now runs before `session_after_deadline` (C6f).
- **New format** `tday_push_list/1` (C6f3).
- **Inventory command line:** `--freeze-dl` became `--freeze-record` (C6g3). The export's freeze block keeps `{dl_row, manifest_sha256}`.
- **Body blocks** are held to exact key sets (C6g2, C6g4).
- **The evaluator file set** follows a config's `custom_imports` and `_base_` files, and holds the two arm teacher configs (C6g5). A manifest's `evaluator.files` therefore lists more files.
- **ARM_TEACHER_RECORD** keeps only `first_nonfinite`'s `iteration` (C6g5).
- **Not adopted:** a fixed vocabulary for `sources[].kind`.

## 7. Refusal codes

There are 89 codes:
- 84 gates in `src/tman/validate.py` GATES, in their order;
- 5 more raised by `read_inventory` and the command line.

Each has a killed mutation (§9: `every_refusal_code_has_a_killed_mutation`). The modes (`src/tman/validate.py:35`) are:
- all (draft, freeze, unlock and session);
- F, U (freeze, unlock and session);
- U (unlock and session);
- S (session only).

A gate outside a draft's modes that a draft lists in MISSING_IN_DRAFT is printed there as MISSING instead of refused.

| # | code | modes | gate (file:line) | raised directly at |
|---|---|---|---|---|
| 1 | `test_path` | all | `g_test_path` src/tman/validate.py:274 | src/tman/schema.py:251, src/tman/entries.py:673, scripts/tday_manifest.py:61, scripts/tday_manifest.py:513 |
| 2 | `manifest_unreadable` | all | `g_manifest_unreadable` src/tman/validate.py:282 |  |
| 3 | `manifest_format` | all | `g_manifest_format` src/tman/validate.py:286 | src/tman/manifest.py:848, scripts/tday_manifest.py:154, scripts/tday_manifest.py:365 |
| 4 | `catalogue_invalid` | all | `g_catalogue_invalid` src/tman/validate.py:301 | src/tman/schema.py:541 |
| 5 | `expected_rows_mismatch` | all | `g_expected_rows_mismatch` src/tman/validate.py:306 |  |
| 6 | `entry_kind_unknown` | all | `g_entry_kind_unknown` src/tman/validate.py:312 | src/tman/entries.py:71 |
| 7 | `entry_field` | all | `g_entry_field` src/tman/validate.py:318 | src/tman/schema.py:227, src/tman/entries.py:87, src/tman/entries.py:644, src/tman/entries.py:786 … |
| 8 | `slot_unknown` | all | `g_slot_unknown` src/tman/validate.py:322 | src/tman/entries.py:534, src/tman/manifest.py:851, scripts/tday_manifest.py:316 |
| 9 | `slot_duplicate` | all | `g_slot_duplicate` src/tman/validate.py:327 |  |
| 10 | `derived_counted_as_run` | all | `g_derived_counted_as_run` src/tman/validate.py:337 | src/tman/entries.py:537 |
| 11 | `record_uncommitted` | all | `g_record_uncommitted` src/tman/validate.py:362 | src/tman/records.py:279, src/tman/entries.py:226, src/tman/manifest.py:955, scripts/tday_manifest.py:126 … |
| 12 | `dl_row_missing` | all | `g_dl_row_missing` src/tman/validate.py:396 | src/tman/records.py:231, src/tman/entries.py:781 |
| 13 | `entry_file_rewritten` | all | `g_entry_file_rewritten` src/tman/validate.py:409 | src/tman/entries.py:675, src/tman/entries.py:778 |
| 14 | `entry_file_invalid` | all | `g_entry_file_invalid` src/tman/validate.py:440 | src/tman/entries.py:788 |
| 15 | `evaluator_commit_invalid` | all | `g_evaluator_commit_invalid` src/tman/validate.py:462 |  |
| 16 | `execution_context_split` | all | `g_execution_context_split` src/tman/validate.py:477 |  |
| 17 | `calibration_index_mismatch` | all | `g_calibration_index_mismatch` src/tman/validate.py:500 |  |
| 18 | `reference_mismatch` | all | `g_reference_mismatch` src/tman/validate.py:509 |  |
| 19 | `teacher_values_mismatch` | all | `g_teacher_values_mismatch` src/tman/validate.py:531 |  |
| 20 | `image_digest_not_of_record` | all | `g_image_digest_not_of_record` src/tman/validate.py:542 |  |
| 21 | `schedule_invalid` | all | `g_schedule_invalid` src/tman/validate.py:573 | src/tman/schedule.py:95, src/tman/schedule.py:104, src/tman/schedule.py:191 |
| 22 | `freeze_source_conflict` | F, U | `g_freeze_source_conflict` src/tman/validate.py:630 | src/tman/schedule.py:111, scripts/tday_manifest.py:195 |
| 23 | `content_class_missing` | F, U | `g_content_class_missing` src/tman/validate.py:662 |  |
| 24 | `generator_pin_incomplete` | F, U | `g_generator_pin_incomplete` src/tman/validate.py:685 |  |
| 25 | `generator_pin_sha256_mismatch` | all | `g_generator_pin_sha256_mismatch` src/tman/validate.py:691 |  |
| 26 | `generator_cells_mismatch` | F, U | `g_generator_cells_mismatch` src/tman/validate.py:696 | src/tman/manifest.py:555, src/tman/manifest.py:567 |
| 27 | `test_value_before_unlock` | all | `g_test_value_before_unlock` src/tman/validate.py:700 |  |
| 28 | `trigger_format_unknown` | F, U | `g_trigger_format_unknown` src/tman/validate.py:707 |  |
| 29 | `trigger_inconsistent` | F, U | `g_trigger_inconsistent` src/tman/validate.py:725 |  |
| 30 | `launched_attempt_without_directory` | F, U | `g_launched_attempt_without_directory` src/tman/validate.py:754 | src/tman/completion.py:366, src/tman/completion.py:439, src/tman/validate.py:124, scripts/tday_manifest.py:179 … |
| 31 | `completion_record_invalid` | F, U | `g_completion_record_invalid` src/tman/validate.py:778 | src/tman/completion.py:130, src/tman/completion.py:478, src/tman/completion.py:481 |
| 32 | `completion_source_mismatch` | F, U | `g_completion_source_mismatch` src/tman/validate.py:799 | src/tman/manifest.py:971 |
| 33 | `checkpoint_state_flag` | F, U | `g_checkpoint_state_flag` src/tman/validate.py:861 |  |
| 34 | `entry_not_complete` | F, U | `g_entry_not_complete` src/tman/validate.py:870 |  |
| 35 | `complete_run_omitted` | F, U | `g_complete_run_omitted` src/tman/validate.py:881 |  |
| 36 | `checkpoint_hash_mismatch` | all | `g_checkpoint_hash_mismatch` src/tman/validate.py:899 |  |
| 37 | `checkpoint_missing` | F, U | `g_checkpoint_missing` src/tman/validate.py:935 |  |
| 38 | `restored_checkpoint_unrecorded` | all | `g_restored_checkpoint_unrecorded` src/tman/validate.py:947 |  |
| 39 | `cut_rule_unknown` | all | `gate` src/tman/validate.py:None |  |
| 40 | `cut_of_never_cut` | all | `gate` src/tman/validate.py:None |  |
| 41 | `cut_of_complete_run` | all | `gate` src/tman/validate.py:None |  |
| 42 | `cut_reason_invalid` | all | `gate` src/tman/validate.py:None |  |
| 43 | `cut_date_mismatch` | all | `gate` src/tman/validate.py:None |  |
| 44 | `cut_arithmetic_invalid` | all | `gate` src/tman/validate.py:None | src/tman/entries.py:142, src/tman/entries.py:145, src/tman/entries.py:152, src/tman/entries.py:157 … |
| 45 | `cut_val_record_missing` | all | `gate` src/tman/validate.py:None |  |
| 46 | `cut_results_unknown` | all | `gate` src/tman/validate.py:None |  |
| 47 | `arm_cut_results_missing` | all | `gate` src/tman/validate.py:None |  |
| 48 | `arm_run_cut_while_running` | all | `gate` src/tman/validate.py:None |  |
| 49 | `arm_cut_partial` | all | `gate` src/tman/validate.py:None |  |
| 50 | `cut_order_position_mismatch` | all | `gate` src/tman/validate.py:None |  |
| 51 | `decision_date_cut_mismatch` | all | `gate` src/tman/validate.py:None |  |
| 52 | `stopped_run_cut_early` | all | `gate` src/tman/validate.py:None |  |
| 53 | `unregistered_arm` | all | `gate` src/tman/validate.py:None | src/tman/entries.py:764 |
| 54 | `condition_unmet` | all | `gate` src/tman/validate.py:None |  |
| 55 | `parent_rule_violation` | all | `g_parent_rule_violation` src/tman/validate.py:1031 |  |
| 56 | `descendant_entry_missing` | F, U | `g_descendant_entry_missing` src/tman/validate.py:1035 |  |
| 57 | `not_run_without_parent_record` | all | `g_not_run_without_parent_record` src/tman/validate.py:1039 | scripts/tday_manifest.py:169 |
| 58 | `derived_parent_incomplete` | all | `g_derived_parent_incomplete` src/tman/validate.py:1054 |  |
| 59 | `derived_inputs_incomplete` | all | `g_derived_inputs_incomplete` src/tman/validate.py:1058 | src/tman/manifest.py:867 |
| 60 | `derived_inputs_mismatch` | all | `g_derived_inputs_mismatch` src/tman/validate.py:1070 |  |
| 61 | `ptq_record_invalid` | all | `g_ptq_record_invalid` src/tman/validate.py:1075 | src/tman/records.py:349, src/tman/records.py:354, src/tman/records.py:358, scripts/tday_manifest.py:377 … |
| 62 | `derived_missing` | F, U | `g_derived_missing` src/tman/validate.py:1110 | scripts/tday_manifest.py:243 |
| 63 | `e6s42_not_hashed_at_freeze` | F, U | `g_e6s42_not_hashed_at_freeze` src/tman/validate.py:1124 |  |
| 64 | `derived_selection_conflict` | all | `g_derived_selection_conflict` src/tman/validate.py:1128 |  |
| 65 | `non_finite_invalid` | all | `g_non_finite_invalid` src/tman/validate.py:1138 |  |
| 66 | `append_invalid` | all | `g_append_invalid` src/tman/validate.py:1150 | src/tman/manifest.py:494, src/tman/manifest.py:522, src/tman/manifest.py:531, scripts/tday_manifest.py:367 … |
| 67 | `derived_hash_mismatch` | all | `g_derived_hash_mismatch` src/tman/validate.py:1171 | scripts/tday_manifest.py:399 |
| 68 | `append_after_unlock` | U | `g_append_after_unlock` src/tman/validate.py:1190 |  |
| 69 | `derived_not_hashed_at_unlock` | U | `g_derived_not_hashed_at_unlock` src/tman/validate.py:1196 |  |
| 70 | `scope_forbidden` | all | `g_scope_forbidden` src/tman/validate.py:1214 |  |
| 71 | `scope_required_missing` | F, U | `g_scope_required_missing` src/tman/validate.py:1220 |  |
| 72 | `candidate_scored` | all | `g_candidate_scored` src/tman/validate.py:1226 |  |
| 73 | `default_candidate_missing` | F, U | `g_default_candidate_missing` src/tman/validate.py:1233 |  |
| 74 | `listed_or_terminal_scored` | all | `g_listed_or_terminal_scored` src/tman/validate.py:1239 |  |
| 75 | `run_added_after_freeze` | U | `g_run_added_after_freeze` src/tman/validate.py:1245 | scripts/tday_manifest.py:221, scripts/tday_manifest.py:225, scripts/tday_manifest.py:248 |
| 76 | `cut_unentered_at_unlock` | F, U | `g_cut_unentered_at_unlock` src/tman/validate.py:1261 |  |
| 77 | `lane_unmet` | U | `g_lane_unmet` src/tman/validate.py:1272 | scripts/tday_manifest.py:227 |
| 78 | `not_required_invalid` | U | `g_not_required_invalid` src/tman/validate.py:1302 | scripts/tday_manifest.py:205, scripts/tday_manifest.py:208 |
| 79 | `evaluator_files_changed` | U | `g_evaluator_files_changed` src/tman/validate.py:1329 |  |
| 80 | `hasher_pin_mismatch` | U | `g_hasher_pin_mismatch` src/tman/validate.py:1348 |  |
| 81 | `second_session` | U | `g_second_session` src/tman/validate.py:1372 | scripts/tday_manifest.py:230 |
| 82 | `clock_offset_invalid` | S | `g_clock_offset_invalid` src/tman/validate.py:1409 | scripts/tday_manifest.py:253 |
| 83 | `session_after_deadline` | U | `g_session_after_deadline` src/tman/validate.py:1397 |  |
| 84 | `push_evidence_invalid` | S | `g_push_evidence_invalid` src/tman/validate.py:1495 |  |
| 85 | `inventory_invalid` | read_inventory / CLI | — | src/tman/manifest.py:991, src/tman/validate.py:1694, src/tman/validate.py:1700, scripts/tday_manifest.py:428 |
| 86 | `inventory_sha256_mismatch` | read_inventory / CLI | — | src/tman/validate.py:1696 |
| 87 | `binding_invalid` | read_inventory / CLI | — | src/tman/validate.py:1707, src/tman/validate.py:1709, src/tman/validate.py:1713 |
| 88 | `output_location` | read_inventory / CLI | — | scripts/tday_manifest.py:51, scripts/tday_manifest.py:54 |
| 89 | `output_exists` | read_inventory / CLI | — | src/tman/entries.py:682, scripts/tday_manifest.py:67 |

## 8. The catalogue

`configs/tday_catalogue.json` (`tday_catalogue/1`) has 69 slots, in catalogue order.

- **Scopes:** the required and forbidden scopes come from `schema.scope_rule` (IF§9; TB2–TB5).
- **pair_s4344:** E1 and E6 at seeds 43 and 44 also take `corrupted` when both runs of the seed have a model.
- **Cut place:** the 1-based place of the slot's cut group in the relative cut order (TL-14 `cut_place`).
- **Gates:** the records each slot waits for.

| slot | file_id | class | roles | required scopes | forbidden scopes | parent | cut place | gates |
|---|---|---|---|---|---|---|---|---|
| `teacher:s42:40k` | teacher_s42_40k | run | teacher_reference | clean, corrupted, corrupted_sev45, dedup, efficiency, upstream | fq_off | — | never cut | — |
| `E1:s42:80k` | E1_s42_80k | run | family | clean, corrupted, corrupted_sev45, dedup, efficiency, upstream | fq_off | — | never cut | — |
| `E1:s43:80k` | E1_s43_80k | run | per_seed | clean, dedup, upstream (+corrupted when both of the seed's pair have a model) | corrupted, corrupted_sev45, efficiency, fq_off | — | — | — |
| `E1:s44:80k` | E1_s44_80k | run | per_seed | clean, dedup, upstream (+corrupted when both of the seed's pair have a model) | corrupted, corrupted_sev45, efficiency, fq_off | — | — | — |
| `E2:s42:lambda0.25:80k` | E2_s42_lambda0.25_80k | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | — | — |
| `E2:s42:lambda0.5:80k` | E2_s42_lambda0.5_80k | run | family | clean, corrupted, corrupted_sev45, dedup, efficiency, upstream | fq_off | — | never cut | — |
| `E2:s42:lambda1:80k` | E2_s42_lambda1_80k | run | default_candidate | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | never cut | — |
| `E2:s42:lambda2:80k` | E2_s42_lambda2_80k | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | — | — |
| `E2:s42:lambda4:80k` | E2_s42_lambda4_80k | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | — | — |
| `E3:s42:lambda0.5:alpha25:80k` | E3_s42_lambda0.5_alpha25_80k | run | family | clean, corrupted, corrupted_sev45, dedup, efficiency, upstream | fq_off | — | never cut | lambda_selection |
| `E3:s42:lambda0.5:alpha50:80k` | E3_s42_lambda0.5_alpha50_80k | run | default_candidate | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | never cut | lambda_selection |
| `E3:s42:lambda0.5:alpha100:80k` | E3_s42_lambda0.5_alpha100_80k | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | 17 | lambda_selection |
| `E2:s43:lambda0.5:80k` | E2_s43_lambda0.5_80k | run | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 18 | lambda_selection |
| `E2:s44:lambda0.5:80k` | E2_s44_lambda0.5_80k | run | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 18 | lambda_selection |
| `E3:s43:lambda0.5:alpha25:80k` | E3_s43_lambda0.5_alpha25_80k | run | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 20 | alpha_selection |
| `E3:s44:lambda0.5:alpha25:80k` | E3_s44_lambda0.5_alpha25_80k | run | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 20 | alpha_selection |
| `E1:s42:160k` | E1_s42_160k | run | control | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 15 | — |
| `E2:s42:lambda0.5:160k` | E2_s42_lambda0.5_160k | run | control | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 16 | lambda_selection |
| `E3:s42:lambda0.5:alpha25:160k` | E3_s42_lambda0.5_alpha25_160k | run | control | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 13 | alpha_selection |
| `A:s42:alpha25:80k` | A_s42_alpha25_80k | run | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 12 | alpha_selection |
| `F:s42:alpha25:80k` | F_s42_alpha25_80k | run | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 11 | alpha_selection |
| `G:s42:80k` | G_s42_80k | run | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 10 | — |
| `teacher-R2:s42:40k` | teacher-R2_s42_40k | run | arm_teacher | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | — | 4 | r1_end |
| `E3-R2:s42:lambda0.5:alpha25:80k` | E3-R2_s42_lambda0.5_alpha25_80k | run | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | teacher-R2:s42:40k | 4 | arm_teacher_selection; alpha_selection; go_rule |
| `E7-R2:s42:list42` | E7-R2_s42_list42 | derived | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E3-R2:s42:lambda0.5:alpha25:80k | 4 | — |
| `teacher-R1:s42:40k` | teacher-R1_s42_40k | run | arm_teacher | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | — | 7 | — |
| `E3-R1:s42:lambda0.5:alpha25:80k` | E3-R1_s42_lambda0.5_alpha25_80k | run | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | teacher-R1:s42:40k | 7 | arm_teacher_selection; alpha_selection; go_rule |
| `E7-R1:s42:list42` | E7-R1_s42_list42 | derived | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E3-R1:s42:lambda0.5:alpha25:80k | 7 | — |
| `E3-flip:s42:lambda0.5:alpha25:80k` | E3-flip_s42_lambda0.5_alpha25_80k | run | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | — | 5 | alpha_selection |
| `E7-flip:s42:list42` | E7-flip_s42_list42 | derived | arm | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E3-flip:s42:lambda0.5:alpha25:80k | 5 | — |
| `teacher-L:s42:40k` | teacher-L_s42_40k | run | arm_teacher | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | 9 | code_merged |
| `E2-L:s42:lambda0.5:80k` | E2-L_s42_lambda0.5_80k | run | arm | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | teacher-L:s42:40k | 9 | arm_teacher_selection; go_rule |
| `E3-L:s42:lambda0.5:alpha25:80k` | E3-L_s42_lambda0.5_alpha25_80k | run | arm | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | teacher-L:s42:40k | 9 | arm_teacher_selection; alpha_selection; go_rule |
| `E7-L:s42:list42` | E7-L_s42_list42 | derived | arm | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E3-L:s42:lambda0.5:alpha25:80k | 9 | — |
| `E5:s42:clip1:15ep` | E5_s42_clip1_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E1:s42:80k | never cut | — |
| `E5:s42:clip5:15ep` | E5_s42_clip5_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E1:s42:80k | never cut | — |
| `E5:s42:sel` | E5_s42_sel | derived | family | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | — | never cut | — |
| `E6:s42:15ep` | E6_s42_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E3:s42:lambda0.5:alpha25:80k | never cut | pilot_winner |
| `E6:s42:sel` | E6_s42_sel | derived | family | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | E6:s42:15ep | never cut | — |
| `E5:s43:15ep` | E5_s43_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E1:s43:80k | 19 | pilot_winner |
| `E5:s43:sel` | E5_s43_sel | derived | per_seed | clean, dedup, fq_off, upstream | corrupted, corrupted_sev45, efficiency | E5:s43:15ep | 19 | — |
| `E5:s44:15ep` | E5_s44_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E1:s44:80k | 19 | pilot_winner |
| `E5:s44:sel` | E5_s44_sel | derived | per_seed | clean, dedup, fq_off, upstream | corrupted, corrupted_sev45, efficiency | E5:s44:15ep | 19 | — |
| `E6:s43:15ep` | E6_s43_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E3:s43:lambda0.5:alpha25:80k | 19 | parent_end; pilot_winner |
| `E6:s43:sel` | E6_s43_sel | derived | per_seed | clean, dedup, fq_off, upstream (+corrupted when both of the seed's pair have a model) | corrupted, corrupted_sev45, efficiency | E6:s43:15ep | 19 | — |
| `E6:s44:15ep` | E6_s44_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E3:s44:lambda0.5:alpha25:80k | 19 | parent_end; pilot_winner |
| `E6:s44:sel` | E6_s44_sel | derived | per_seed | clean, dedup, fq_off, upstream (+corrupted when both of the seed's pair have a model) | corrupted, corrupted_sev45, efficiency | E6:s44:15ep | 19 | — |
| `E6-KD:s42:15ep` | E6-KD_s42_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E3:s42:lambda0.5:alpha25:80k | never cut | trigger; code_merged |
| `E6-KD:s42:sel` | E6-KD_s42_sel | derived | e6kd | clean, dedup, fq_off, upstream | corrupted, corrupted_sev45, efficiency | E6-KD:s42:15ep | never cut | — |
| `E6-E2:s42:15ep` | E6-E2_s42_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E2:s42:lambda0.5:80k | 3 | pilot_winner; code_merged |
| `E6-E2:s42:sel` | E6-E2_s42_sel | derived | quant_descriptive | clean, dedup, fq_off, upstream | corrupted, corrupted_sev45, efficiency | E6-E2:s42:15ep | 3 | — |
| `E7-E2:s42:list42` | E7-E2_s42_list42 | derived | quant_descriptive | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E2:s42:lambda0.5:80k | — | code_merged |
| `E6-R1:s42:15ep` | E6-R1_s42_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E3-R1:s42:lambda0.5:alpha25:80k | 2 | parent_end; pilot_winner; condition_entry; code_merged |
| `E6-R1:s42:sel` | E6-R1_s42_sel | derived | quant_descriptive | clean, dedup, fq_off, upstream | corrupted, corrupted_sev45, efficiency | E6-R1:s42:15ep | 2 | — |
| `E6-R2:s42:15ep` | E6-R2_s42_15ep | run | listed | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | E3-R2:s42:lambda0.5:alpha25:80k | 1 | parent_end; pilot_winner; condition_entry; code_merged |
| `E6-R2:s42:sel` | E6-R2_s42_sel | derived | quant_descriptive | clean, dedup, fq_off, upstream | corrupted, corrupted_sev45, efficiency | E6-R2:s42:15ep | 1 | — |
| `E4:s42:list42` | E4_s42_list42 | derived | family | clean, corrupted, corrupted_sev45, dedup, efficiency, upstream | fq_off | E1:s42:80k | never cut | — |
| `E4:s43:list42` | E4_s43_list42 | derived | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E1:s43:80k | never cut | — |
| `E4:s44:list42` | E4_s44_list42 | derived | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E1:s44:80k | never cut | — |
| `E7:s42:list42` | E7_s42_list42 | derived | family | clean, corrupted, corrupted_sev45, dedup, efficiency, upstream | fq_off | E3:s42:lambda0.5:alpha25:80k | never cut | — |
| `E7:s43:list42` | E7_s43_list42 | derived | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E3:s43:lambda0.5:alpha25:80k | never cut | — |
| `E7:s44:list42` | E7_s44_list42 | derived | per_seed | clean, dedup, upstream | corrupted, corrupted_sev45, efficiency, fq_off | E3:s44:lambda0.5:alpha25:80k | never cut | — |
| `E4:s42:list43` | E4_s42_list43 | derived | calibration_sensitivity | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | E1:s42:80k | never cut | — |
| `E4:s42:list44` | E4_s42_list44 | derived | calibration_sensitivity | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | E1:s42:80k | never cut | — |
| `E4:s42:list45` | E4_s42_list45 | derived | calibration_sensitivity | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | E1:s42:80k | never cut | — |
| `E7:s42:list43` | E7_s42_list43 | derived | calibration_sensitivity | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | E3:s42:lambda0.5:alpha25:80k | never cut | — |
| `E7:s42:list44` | E7_s42_list44 | derived | calibration_sensitivity | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | E3:s42:lambda0.5:alpha25:80k | never cut | — |
| `E7:s42:list45` | E7_s42_list45 | derived | calibration_sensitivity | clean, dedup | corrupted, corrupted_sev45, efficiency, fq_off, upstream | E3:s42:lambda0.5:alpha25:80k | never cut | — |
| `ARM:latency` | ARM_latency | measurement | — | — | clean, corrupted, corrupted_sev45, dedup, efficiency, fq_off, upstream | — | 14 | — |

## 9. Mutations

`smoke_tman_mutations` ran at `005a8912c487510321c6c8b7af5b263c8ced03e7`: RESULT: PASS (429/429).
- Each mutation copies the lane's code into a scratch folder, edits one text that must occur exactly once, and runs its named case alone in a fresh interpreter.
- A mutation is killed only when its named check fails. A crash or another check failing is no kill (DL-24 #6).

**The plan's sixteen samples** (S01–S16) are all present (`plan_s_sixteen_sample_mutations_present`).

**Sample 9 is remapped** (DL-24 #58):
- The plan's sample 9 was "the original instant used for 'no run added' → R-016's fixture".
- S09 as built is the reverse fault: the original instant used for a never-cut run, killed by `entry_not_complete`.
- R-016's fixture itself now exists (C6h): `cuttable_run_complete_between_the_freezes_R016_B58`. Its mutation is `instant_moved_for_every_slot`: the moved instant for every slot.

**Mutations made by hand** (MEASURED): the harness mutates only the lane's code, so `case_worker_scratch`'s two mutations were made by hand in a dev run, and each failed its check.
- With the `rmtree` dropped, both B108 checks fail.
- With TMPDIR dropped, the first fails.

| # | mutation | file | killed by (smoke:case) | its named check | codes | plan sample |
|---|---|---|---|---|---|---|
| 1 | `test_path_gate` | validate.py | validate:case_input_gates | `test_path_manifest_named_test` | test_path |  |
| 2 | `manifest_strict_json` | validate.py | validate:case_input_gates | `manifest_duplicate_key` | manifest_unreadable |  |
| 3 | `manifest_parts` | validate.py | validate:case_input_gates | `manifest_without_session_part` | manifest_format |  |
| 4 | `catalogue_from_the_checkout` | validate.py | validate:case_input_gates | `catalogue_invalid` | catalogue_invalid |  |
| 5 | `expected_rows` | validate.py | validate:case_input_gates | `expected_rows_1560_TB14` | expected_rows_mismatch |  |
| 6 | `entry_kind` | validate.py | validate:case_input_gates | `entry_kind_unknown` | entry_kind_unknown |  |
| 7 | `entry_fields` | validate.py | validate:case_input_gates | `entry_field_extra_key` | entry_field |  |
| 8 | `slot_known` | validate.py | validate:case_input_gates | `slot_unknown` | slot_unknown |  |
| 9 | `slot_once` | validate.py | validate:case_input_gates | `slot_duplicate` | slot_duplicate |  |
| 10 | `class_of_the_slot` | validate.py | validate:case_input_gates | `derived_slot_in_a_run_kind` | derived_counted_as_run |  |
| 11 | `records_read` | validate.py | validate:case_record_gates | `record_modified_in_the_tree` | record_uncommitted |  |
| 12 | `dl_rows_exist` | validate.py | validate:case_record_gates | `dl_row_absent` | dl_row_missing |  |
| 13 | `entry_file_sha` | validate.py | validate:case_record_gates | `cut_entry_cited_with_another_sha` | entry_file_rewritten |  |
| 14 | `entry_file_dl_row` | validate.py | validate:case_entry_files | `entry_file_whose_row_is_only_DECIDED_B92` | dl_row_missing |  |
| 15 | `entry_file_written_once` | validate.py | validate:case_entry_files | `entry_file_rewritten_after_its_commit_B92` | entry_file_rewritten |  |
| 16 | `entry_file_kind` | entries.py | validate:case_entry_files | `not_run_citing_a_cut_entry_file_B38` | entry_file_invalid |  |
| 17 | `entry_file_name` | entries.py | validate:case_entry_files | `cut_entry_file_outside_its_folder_B89` | entry_file_invalid |  |
| 18 | `entry_file_fields` | entries.py | validate:case_moved_freeze | `freeze_delay_file_without_runs_B88` | entry_file_invalid |  |
| 19 | `entry_file_slots_known` | validate.py | validate:case_entry_files | `cut_file_naming_an_unknown_slot` | entry_file_invalid |  |
| 20 | `entry_file_slot_class` | validate.py | validate:case_entry_files | `cut_file_naming_a_derived_slot` | entry_file_invalid |  |
| 21 | `entry_file_of_its_slot` | validate.py | validate:case_entry_files | `not_run_citing_another_slot_s_not_run_file_B17` | entry_file_invalid |  |
| 22 | `entry_file_cut_label` | validate.py | validate:case_entry_files | `cut_label_against_its_entry_file` | entry_file_invalid |  |
| 23 | `entry_file_not_run_fields` | validate.py | validate:case_parent_and_derived_gates | `not_run_entry_against_its_file` | entry_file_invalid |  |
| 24 | `cited_condition` | validate.py | validate:case_cut_gates | `e6x_cut_citing_a_cut_entry_as_its_condition` | entry_file_invalid |  |
| 25 | `cited_freeze_source` | validate.py | validate:case_moved_freeze | `freeze_delay_file_without_runs_B88` | entry_file_invalid |  |
| 26 | `cited_freeze_record` | validate.py | unlock:case_unlock_gates | `freeze_record_without_body_sha256_B89` | entry_file_invalid |  |
| 27 | `cited_unlock_entry` | validate.py | unlock:case_unlock_gates | `unlock_entry_whose_manifest_is_a_string_B89` | entry_file_invalid |  |
| 28 | `cited_clock_offset` | validate.py | unlock:case_session | `clock_offset_entry_not_DL_recorded_B102` | dl_row_missing |  |
| 29 | `cited_not_required` | validate.py | unlock:case_unlock_gates | `not_required_entry_not_DL_recorded_B55c` | dl_row_missing |  |
| 30 | `cited_not_required_cuts` | validate.py | unlock:case_unlock_gates | `not_required_cut_entry_not_DL_recorded_B55c` | dl_row_missing |  |
| 31 | `results_known_committed` | validate.py | validate:case_entry_files | `results_known_at_another_sha256_B69` | record_uncommitted |  |
| 32 | `results_known_dl_row` | validate.py | validate:case_entry_files | `results_known_row_not_holding_its_sha256_B69` | dl_row_missing |  |
| 33 | `not_made_lane_merged` | validate.py | validate:case_parent_and_derived_gates | `not_made_whose_lane_merged_by_the_freeze_B23_B76` | derived_missing |  |
| 34 | `lane_merged_by_the_freeze` | validate.py | validate:case_parent_and_derived_gates | `not_made_whose_lane_merged_after_the_freeze_passes` | — |  |
| 35 | `evaluator_commit_exists` | validate.py | validate:case_record_gates | `evaluator_commit_not_a_commit` | evaluator_commit_invalid |  |
| 36 | `evaluator_in_the_records_history` | validate.py | unlock:case_unlock_gates | `records_commit_without_e_in_its_history` | evaluator_commit_invalid |  |
| 37 | `s_equals_e` | validate.py | unlock:case_unlock_gates | `session_checkout_not_e_TB16` | evaluator_commit_invalid |  |
| 38 | `execution_tuple` | validate.py | validate:case_record_gates | `execution_context_on_a_gpu_TB1` | execution_context_split |  |
| 39 | `calibration_index` | validate.py | validate:case_record_gates | `calibration_index_other_split_list` | calibration_index_mismatch |  |
| 40 | `references` | validate.py | validate:case_record_gates | `reference_other_strata_sha` | reference_mismatch |  |
| 41 | `teacher_values` | validate.py | validate:case_record_gates | `teacher_entry_not_DL88_s` | teacher_values_mismatch |  |
| 42 | `derived_image_digest` | validate.py | validate:case_record_gates | `derived_image_not_of_record` | image_digest_not_of_record |  |
| 43 | `schedule_block` | validate.py | validate:case_record_gates | `schedule_block_other_freeze_date` | schedule_invalid |  |
| 44 | `new_freeze_first_complete_day` | validate.py | validate:case_moved_freeze | `new_freeze_not_the_first_complete_day` | freeze_source_conflict |  |
| 45 | `content_class` | validate.py | validate:case_record_gates | `content_class_missing_pin` | content_class_missing |  |
| 46 | `S11_pin_without_library_versions` | manifest.py | pins:case_pin_fields | `pin_without_library_versions_incomplete` | generator_pin_incomplete | 11 |
| 47 | `pin_sha256` | manifest.py | pins:case_pin_sha256 | `pin_sha256_mismatch_after_an_edit` | generator_pin_sha256_mismatch |  |
| 48 | `pin_cells_order` | manifest.py | pins:case_cells | `cells_reordered_refused` | generator_cells_mismatch |  |
| 49 | `S10_test_value_before_the_unlock` | manifest.py | pins:case_values_and_hasher | `corrupted_cache_value_before_the_unlock_refused` | test_value_before_unlock | 10 |
| 50 | `session_part_before_the_session` | validate.py | validate:case_record_gates | `session_part_before_the_session` | test_value_before_unlock |  |
| 51 | `trigger_e6kd` | validate.py | validate:case_record_gates | `triggered_without_e6kd` | trigger_inconsistent |  |
| 52 | `trigger_sha256_compared` | validate.py | validate:case_record_gates | `trigger_record_other_sha256_B1` | record_uncommitted |  |
| 53 | `trigger_reader_required` | validate.py | validate:case_record_gates | `trigger_format_unknown_without_a_reader_O8` | trigger_format_unknown |  |
| 54 | `trigger_verdict_compared` | validate.py | validate:case_record_gates | `trigger_block_against_the_record_s_verdict_B1` | trigger_inconsistent |  |
| 55 | `kind_of_the_slot_s_role` | schema.py | validate:case_input_gates | `family_run_entered_as_listed_run_B2` | derived_counted_as_run |  |
| 56 | `sweep_of_record_strict` | schema.py | validate:case_input_gates | `lambda_of_record_entered_as_listed_run_B2` | derived_counted_as_run |  |
| 57 | `non_finite_qat_only` | schema.py | validate:case_input_gates | `non_finite_entry_on_a_kd_slot` | derived_counted_as_run |  |
| 58 | `evaluator_stage_of_the_catalogue` | validate.py | validate:case_input_gates | `evaluator_stage_other_than_the_catalogue_s_B91` | derived_counted_as_run |  |
| 59 | `derived_method_of_the_slot` | validate.py | validate:case_input_gates | `derived_method_other_than_the_slot_s_B21` | derived_counted_as_run |  |
| 60 | `attempt_directories` | validate.py | validate:case_completion_gates | `attempt_without_directory` | launched_attempt_without_directory |  |
| 61 | `cut_run_directories` | validate.py | validate:case_cut_attribution | `cut_run_withheld_from_runs_B44` | launched_attempt_without_directory |  |
| 62 | `runs_needed_in_freeze_mode` | validate.py | validate:case_moved_freeze | `delay_entry_without_runs_refused_at_its_gate` | launched_attempt_without_directory |  |
| 63 | `e1_max_iters` | completion.py | validate:case_completion_gates | `completion_record_unreadable` | completion_record_invalid |  |
| 64 | `completion_block` | validate.py | validate:case_completion_gates | `completion_block_other_line` | completion_source_mismatch |  |
| 65 | `telemetry_copy_back` | manifest.py | validate:case_inventory | `telemetry_copy_back_differs_IF11_3` | completion_source_mismatch |  |
| 66 | `qat_checkpoint_flags` | validate.py | validate:case_completion_gates | `qat_checkpoint_flag_disagrees_TB13` | checkpoint_state_flag |  |
| 67 | `S01_meets_flipped` | completion.py | validate:case_completion_gates | `complete_at_the_freeze_instant_is_late` | entry_not_complete | 1 |
| 68 | `S02_completeness_loop_removed` | validate.py | validate:case_completion_gates | `complete_kd_run_omitted_AM19_3d` | complete_run_omitted | 2 |
| 69 | `completeness_over_kd_slots_only` | validate.py | validate:case_cut_attribution | `complete_e1_run_omitted_B4` | complete_run_omitted |  |
| 70 | `launched_run_outside_the_catalogue` | validate.py | validate:case_cut_attribution | `launched_run_outside_the_catalogue_B94` | complete_run_omitted |  |
| 71 | `entry_runs_of_its_own_slot` | validate.py | validate:case_completion_gates | `run_entry_citing_another_slot_s_run_B52` | completion_source_mismatch |  |
| 72 | `late_never_cut_run_needs_the_delay` | validate.py | validate:case_cut_attribution | `late_never_cut_run_absent_without_delay_B3` | freeze_source_conflict |  |
| 73 | `delay_keeps_the_manifest_a_draft` | validate.py | validate:case_moved_freeze | `delay_without_new_freeze_stays_a_draft_B3` | freeze_source_conflict |  |
| 74 | `late_run_entered_at_the_new_freeze` | validate.py | validate:case_moved_freeze | `late_never_cut_run_absent_at_the_new_freeze_B3` | freeze_source_conflict |  |
| 75 | `draft_lists_absent_never_cut_runs` | validate.py | validate:case_cut_attribution | `draft_lists_absent_never_cut_and_cuttable_slots_B3` | freeze_source_conflict |  |
| 76 | `every_slot_entered_at_the_freeze` | validate.py | validate:case_cut_attribution | `cuttable_slot_without_entry_at_the_freeze_B4` | cut_unentered_at_unlock |  |
| 77 | `kd_attempts_from_the_launch_log` | completion.py | validate:case_cut_attribution | `launched_kd_run_cut_as_never_launched_B13` | cut_reason_invalid |  |
| 78 | `qat_attempts_from_run_meta` | completion.py | validate:case_cut_attribution | `launched_qat_run_cut_as_never_launched_B12` | cut_reason_invalid |  |
| 79 | `teacher_arm_attempts_from_the_work_dir` | completion.py | validate:case_cut_attribution | `cut_of_a_complete_arm_teacher_AM18_7b` | cut_of_complete_run |  |
| 80 | `slot_status_from_its_own_attempts` | completion.py | validate:case_cut_attribution | `group_cut_naming_a_complete_run_first_B0` | cut_of_complete_run |  |
| 81 | `checkpoint_sources` | validate.py | validate:case_completion_gates | `checkpoint_source_holds_another_sha` | checkpoint_hash_mismatch |  |
| 82 | `checkpoint_bytes` | validate.py | validate:case_completion_gates | `checkpoint_file_differs_DL110` | checkpoint_missing |  |
| 83 | `restored_checkpoint` | validate.py | validate:case_completion_gates | `restored_checkpoint_without_its_report` | restored_checkpoint_unrecorded |  |
| 84 | `S03_rule_operator_allowed` | entries.py | cuts:case_gate_order | `rule_operator_refused` | cut_rule_unknown | 3 |
| 85 | `S04_never_cut_test_skipped` | entries.py | cuts:case_gate_order | `never_cut_slot_refused` | cut_of_never_cut | 4 |
| 86 | `complete_run_cut` | entries.py | cuts:case_gate_order | `complete_run_cut_refused` | cut_of_complete_run |  |
| 87 | `never_launched_reason` | entries.py | cuts:case_gate_order | `never_launched_reason_outside_list` | cut_reason_invalid |  |
| 88 | `cut_day` | entries.py | cuts:case_gate_order | `freeze_cut_dated_off_its_day` | cut_date_mismatch |  |
| 89 | `cut_entered_early` | entries.py | validate:case_cut_gates | `cut_entered_before_its_rule_s_day_B75` | cut_date_mismatch |  |
| 90 | `cut_results_known_by_its_day` | entries.py | validate:case_cut_gates | `results_known_committed_after_the_cut_s_day_B69` | cut_results_unknown |  |
| 91 | `results_held_by_the_day` | records.py | validate:case_cut_gates | `results_known_committed_after_the_cut_s_day_B69` | cut_results_unknown |  |
| 92 | `arm_records_listed` | entries.py | validate:case_cut_gates | `arm_cut_without_its_teacher_record_known_by_its_day_B69` | arm_cut_results_missing |  |
| 93 | `arm_checks_per_group` | entries.py | cuts:case_gate_order | `two_group_named_rule_cut_of_a_running_arm_run_B15` | arm_run_cut_while_running |  |
| 94 | `S05_arithmetic_ge` | preflight_arms.py | cuts:case_arithmetic | `arithmetic_equality_fits` | cut_arithmetic_invalid | 5 |
| 95 | `arithmetic_recomputed` | entries.py | cuts:case_arithmetic | `arithmetic_recomputation_catches_cut` | cut_arithmetic_invalid |  |
| 96 | `arithmetic_iterations_of_the_slot` | entries.py | validate:case_moved_freeze | `arithmetic_iterations_not_the_slot_s_B11` | cut_arithmetic_invalid |  |
| 97 | `arithmetic_reference_read_again` | validate.py | validate:case_moved_freeze | `arithmetic_reference_read_again_differs_B11` | cut_arithmetic_invalid |  |
| 98 | `arithmetic_reference_the_most_recent` | validate.py | validate:case_moved_freeze | `arithmetic_reference_not_the_most_recent_run_B11` | cut_arithmetic_invalid |  |
| 99 | `latest_finished_the_latest` | validate.py | validate:case_moved_freeze | `arithmetic_reference_not_the_most_recent_run_B11` | cut_arithmetic_invalid |  |
| 100 | `printer_reference_the_most_recent` | tday_manifest.py | validate:case_moved_freeze | `printer_arithmetic_reference_not_the_most_recent_B11` | cut_arithmetic_invalid |  |
| 101 | `printer_iterations_of_the_slot` | tday_manifest.py | validate:case_moved_freeze | `printer_arithmetic_iterations_not_the_horizon_B11` | cut_arithmetic_invalid |  |
| 102 | `reference_read_as_its_slot` | entries.py | cuts:case_references | `reference_of_another_stage_unread_B111` | completion_record_invalid |  |
| 103 | `reference_smoke_not_an_object` | entries.py | cuts:case_references | `reference_smoke_holding_a_list_B68` | cut_arithmetic_invalid |  |
| 104 | `launched_cut_records` | entries.py | cuts:case_gate_order | `launched_cut_without_telemetry` | cut_val_record_missing |  |
| 105 | `arm_results_known` | entries.py | cuts:case_gate_order | `arm_cut_without_results` | arm_cut_results_missing |  |
| 106 | `arm_running` | entries.py | cuts:case_gate_order | `launched_arm_cut_by_named_rule_refused` | arm_run_cut_while_running |  |
| 107 | `arm_partial` | entries.py | cuts:case_gate_order | `arm_cut_missing_its_later_run` | arm_cut_partial |  |
| 108 | `cut_order_position` | entries.py | cuts:case_gate_order | `freeze_cut_wrong_position` | cut_order_position_mismatch |  |
| 109 | `decision_record_cuts` | entries.py | cuts:case_gate_order | `decision_date_cut_not_in_the_record` | decision_date_cut_mismatch |  |
| 110 | `stopped_run` | entries.py | cuts:case_gate_order | `stopped_run_cut_by_named_rule_refused` | stopped_run_cut_early |  |
| 111 | `stopped_candidate_at_its_decision_date` | entries.py | cuts:case_gate_order | `stopped_candidate_cut_at_its_decision_date_AM19_2b` | stopped_run_cut_early |  |
| 112 | `launched_attempts_named` | entries.py | validate:case_cut_attribution | `launched_kd_run_cut_as_never_launched_B13` | cut_reason_invalid |  |
| 113 | `run_ids_of_the_cut_slots` | entries.py | validate:case_cut_attribution | `cut_naming_another_slot_s_run_B5` | cut_reason_invalid |  |
| 114 | `run_ids_launched` | entries.py | cuts:case_gate_order | `run_id_no_record_launched` | cut_reason_invalid |  |
| 115 | `arm_registry` | entries.py | cuts:case_gate_order | `arm_without_registry_file_refused` | unregistered_arm |  |
| 116 | `condition_no` | entries.py | cuts:case_gate_order | `e6x_cut_citing_a_yes_condition_refused` | condition_unmet |  |
| 117 | `named_rule_record_kind` | entries.py | validate:case_cut_gates | `named_rule_citing_an_unknown_record_B16` | cut_rule_unknown |  |
| 118 | `named_rule_record_date` | entries.py | validate:case_cut_gates | `named_rule_record_dated_otherwise_B16` | cut_date_mismatch |  |
| 119 | `rule_day_from_the_record` | entries.py | cuts:case_gate_order | `named_rule_cut_dated_by_its_entry_not_its_record_B16` | cut_date_mismatch |  |
| 120 | `condition_of_its_arm` | entries.py | cuts:case_gate_order | `e6x_cut_citing_another_arm_s_no_condition_B16` | condition_unmet |  |
| 121 | `condition_e3x_run_of_record` | entries.py | validate:case_cut_gates | `e6x_cut_without_an_e3x_run_of_record_B16` | condition_unmet |  |
| 122 | `e3x_from_the_manifest` | validate.py | validate:case_cut_gates | `e6x_cut_without_an_e3x_run_of_record_B16` | condition_unmet |  |
| 123 | `go_rule_of_its_arm` | entries.py | validate:case_cut_gates | `go_rule_no_go_of_another_arm_B16` | condition_unmet |  |
| 124 | `go_rule_no_go` | entries.py | cuts:case_gate_order | `go_rule_reading_go_refused_B16` | condition_unmet |  |
| 125 | `go_rule_record_read` | entries.py | validate:case_cut_gates | `launched_arm_cut_by_a_named_rule` | arm_run_cut_while_running |  |
| 126 | `condition_entry_read` | entries.py | validate:case_cut_gates | `e6x_cut_citing_a_yes_condition_AM23_3b` | condition_unmet |  |
| 127 | `rule_records_committed` | validate.py | validate:case_cut_gates | `named_rule_record_at_another_sha256_B16` | record_uncommitted |  |
| 128 | `parent_without_a_model` | entries.py | parent:case_refusals | `derived_under_a_cut_parent_refused` | parent_rule_violation |  |
| 129 | `S06_parent_map_E7_to_E1` | tday_catalogue.json | validate:case_parent_and_derived_gates | `derived_under_a_cut_parent` | parent_rule_violation | 6 |
| 130 | `descendants_entered` | entries.py | parent:case_refusals | `cut_parent_without_its_descendants_refused` | descendant_entry_missing |  |
| 131 | `not_run_parent_record` | entries.py | parent:case_refusals | `not_run_citing_another_record_refused` | not_run_without_parent_record |  |
| 132 | `derived_parent` | manifest.py | validate:case_parent_and_derived_gates | `derived_parent_other_than_catalogue` | derived_parent_incomplete |  |
| 133 | `derived_inputs` | manifest.py | validate:case_parent_and_derived_gates | `derived_inputs_without_qconfig` | derived_inputs_incomplete |  |
| 134 | `ptq_record_seed` | manifest.py | derived:case_ptq_record | `ptq_record_other_seed_refused_A2` | ptq_record_invalid |  |
| 135 | `append_ptq_record_read` | validate.py | validate:case_appends | `post_freeze_append_of_record_without_its_ptq_record_A2` | ptq_record_invalid |  |
| 136 | `required_derived` | manifest.py | validate:case_parent_and_derived_gates | `required_derived_absent` | derived_missing |  |
| 137 | `e6s42_hashed` | manifest.py | validate:case_parent_and_derived_gates | `e6s42_by_inputs_at_the_freeze` | e6s42_not_hashed_at_freeze |  |
| 138 | `fault_report` | manifest.py | validate:case_parent_and_derived_gates | `second_evaluation_without_report` | derived_selection_conflict |  |
| 139 | `qat_rules_of_record` | validate.py | validate:case_record_gates | `qat_rules_sha256_not_the_committed_file_B18` | derived_inputs_incomplete |  |
| 140 | `builder_rules_of_record` | manifest.py | validate:case_record_gates | `builder_refuses_another_rules_sha256_B19` | derived_inputs_incomplete |  |
| 141 | `builder_reads_the_rules` | manifest.py | validate:case_record_gates | `builder_reads_the_rules_sha256_TB13` | — |  |
| 142 | `calibration_lists_in_dl10` | manifest.py | validate:case_record_gates | `calibration_list_not_dl10_s_B30` | calibration_index_mismatch |  |
| 143 | `calibration_checksum_in_dl10` | manifest.py | validate:case_record_gates | `calibration_checksum_not_dl10_s_B30` | calibration_index_mismatch |  |
| 144 | `calibration_seed_its_key` | manifest.py | validate:case_record_gates | `calibration_list_seed_not_its_key_B40` | calibration_index_mismatch |  |
| 145 | `calibration_role_of_its_list` | manifest.py | validate:case_record_gates | `calibration_list_role_not_am16_s_B40` | calibration_index_mismatch |  |
| 146 | `appended_selection_conflict` | validate.py | validate:case_parent_and_derived_gates | `appended_qat_selection_second_evaluation_without_report_B24` | derived_selection_conflict |  |
| 147 | `fault_reports_committed` | validate.py | validate:case_parent_and_derived_gates | `qat_fault_report_not_committed_B77` | record_uncommitted |  |
| 148 | `cli_second_append` | tday_manifest.py | paths:case_cli_outputs | `cli_append_refuses_a_second_append_of_its_slot_B63` | append_invalid |  |
| 149 | `cli_append_reads_a_manifest` | tday_manifest.py | paths:case_cli_outputs | `cli_append_refuses_a_file_that_is_no_manifest_B63` | manifest_format |  |
| 150 | `non_finite_flags` | manifest.py | validate:case_non_finite_gate | `non_finite_with_a_convertible_epoch` | non_finite_invalid |  |
| 151 | `S07_isolation_test_dropped` | manifest.py | validate:case_appends | `post_freeze_append_without_an_isolation_record_xi` | append_invalid | 7 |
| 152 | `isolation_cited` | manifest.py | validate:case_appends | `isolation_record_not_the_cited_one` | append_invalid |  |
| 153 | `cli_ptq_stand_in` | tday_manifest.py | paths:case_cli_outputs | `cli_post_freeze_ptq_append_with_its_record_TB19_Q9` | append_invalid |  |
| 154 | `S08_appends_after_the_unlock` | validate.py | unlock:case_unlock_gates | `appends_changed_after_the_unlock_record` | append_after_unlock | 8 |
| 155 | `hashed_at_the_unlock` | validate.py | unlock:case_unlock_gates | `derived_by_inputs_at_the_unlock` | derived_not_hashed_at_unlock |  |
| 156 | `S12_scope_rule_relaxed` | schema.py | validate:case_scope_gates | `scope_corrupted_for_a_per_seed_run` | scope_forbidden | 12 |
| 157 | `scope_required` | validate.py | validate:case_scope_gates | `scope_upstream_missing_TB3` | scope_required_missing |  |
| 158 | `S13_default_candidate_rule_dropped` | validate.py | validate:case_scope_gates | `non_default_candidate_scored` | candidate_scored | 13 |
| 159 | `default_candidate_kind` | validate.py | validate:case_scope_gates | `default_candidate_entered_as_listed` | default_candidate_missing |  |
| 160 | `listed_scope` | validate.py | validate:case_scope_gates | `listed_run_with_a_scope` | listed_or_terminal_scored |  |
| 161 | `S15_body_hash_not_recomputed` | validate.py | unlock:case_unlock_gates | `body_changed_after_the_freeze_record` | run_added_after_freeze | 15 |
| 162 | `S09_original_instant_for_never_cut` | validate.py | validate:case_moved_freeze | `delay_and_new_freeze_on_the_first_complete_day_AM19_1d` | entry_not_complete | 9 |
| 163 | `unlock_cites_the_freeze_record` | validate.py | unlock:case_unlock_gates | `unlock_entry_citing_another_freeze_record` | run_added_after_freeze |  |
| 164 | `unlock_records_the_body` | validate.py | unlock:case_unlock_gates | `unlock_entry_of_another_body` | run_added_after_freeze |  |
| 165 | `unlock_records_the_file` | validate.py | unlock:case_unlock_gates | `manifest_file_not_the_unlock_entry_s` | run_added_after_freeze |  |
| 166 | `every_slot_entered` | validate.py | unlock:case_unlock_gates | `slot_without_entry_at_the_unlock` | cut_unentered_at_unlock |  |
| 167 | `lane_implemented` | validate.py | unlock:case_unlock_gates | `lane_neither_implemented_nor_not_required` | lane_unmet |  |
| 168 | `lanes_the_unlock_cites` | validate.py | unlock:case_unlock_gates | `lanes_not_the_unlock_entry_s` | lane_unmet |  |
| 169 | `not_required_covered` | validate.py | unlock:case_unlock_gates | `not_required_lane_serving_an_uncut_slot` | not_required_invalid |  |
| 170 | `lane_register_lists_a_lane` | validate.py | unlock:case_unlock_gates | `lane_register_listing_no_lane_B55a` | lane_unmet |  |
| 171 | `lane_register_dl_recorded` | validate.py | unlock:case_unlock_gates | `lane_register_in_no_decision_log_row` | lane_unmet |  |
| 172 | `lane_register_row_recorded` | validate.py | unlock:case_unlock_gates | `lane_register_in_a_DECIDED_row` | lane_unmet |  |
| 173 | `implemented_row_names_the_lane` | validate.py | unlock:case_unlock_gates | `implemented_row_of_another_lane_B55b` | lane_unmet |  |
| 174 | `implemented_row_names_the_merge` | validate.py | unlock:case_unlock_gates | `implemented_row_naming_another_merge_commit` | lane_unmet |  |
| 175 | `not_required_slots_entered_cut` | validate.py | unlock:case_unlock_gates | `not_required_lane_serving_a_scored_slot` | not_required_invalid |  |
| 176 | `not_required_cut_cited` | validate.py | unlock:case_unlock_gates | `not_required_lane_citing_another_cut_of_its_slot` | not_required_invalid |  |
| 177 | `S16_file_set_comparison_dropped` | validate.py | unlock:case_unlock_gates | `evaluator_file_edited_at_S` | evaluator_files_changed | 16 |
| 178 | `hash_only_compared` | validate.py | unlock:case_unlock_gates | `hash_only_evaluator_file_edited_at_S` | evaluator_files_changed |  |
| 179 | `hasher_script` | validate.py | unlock:case_unlock_gates | `hasher_script_not_its_pin` | hasher_pin_mismatch |  |
| 180 | `one_unlock_row` | validate.py | unlock:case_unlock_gates | `second_unlock_entry_in_the_decision_log` | second_session |  |
| 181 | `binding_written_once` | validate.py | unlock:case_session | `binding_already_written` | second_session |  |
| 182 | `test_deadline` | validate.py | unlock:case_unlock_gates | `unlock_after_the_test_deadline_TB14` | session_after_deadline |  |
| 183 | `clock_offset_limit` | validate.py | unlock:case_session | `clock_offset_over_60_s_AM19a_r21` | clock_offset_invalid |  |
| 184 | `push_before_the_session` | validate.py | unlock:case_session | `unlock_pushed_after_the_session_started_TB11` | push_evidence_invalid |  |
| 185 | `push_evidence_cited_files` | validate.py | unlock:case_push_evidence | `push_evidence_list_not_the_cited_file` | push_evidence_invalid |  |
| 186 | `push_pair_written_once` | validate.py | unlock:case_push_evidence | `push_list_rewritten_after_it_was_added` | push_evidence_invalid |  |
| 187 | `push_pair_one_commit` | validate.py | unlock:case_push_evidence | `push_list_added_apart_from_its_response` | push_evidence_invalid |  |
| 188 | `push_response_parsed` | validate.py | unlock:case_push_evidence | `push_response_not_an_activity_response_B57` | push_evidence_invalid |  |
| 189 | `push_list_an_object` | validate.py | unlock:case_push_evidence | `push_list_not_an_object_B57` | push_evidence_invalid |  |
| 190 | `push_list_format` | validate.py | unlock:case_push_evidence | `push_list_of_another_format` | push_evidence_invalid |  |
| 191 | `push_list_response` | validate.py | unlock:case_push_evidence | `push_list_of_another_response` | push_evidence_invalid |  |
| 192 | `push_list_unlock_entry` | validate.py | unlock:case_push_evidence | `push_list_of_another_unlock_entry` | push_evidence_invalid |  |
| 193 | `push_list_adding_commit` | validate.py | unlock:case_push_evidence | `push_list_of_another_commit` | push_evidence_invalid |  |
| 194 | `push_list_t_lo` | validate.py | unlock:case_push_evidence | `push_list_t_lo_not_the_first_instant_of_the_unlock_day` | push_evidence_invalid |  |
| 195 | `push_list_remote` | validate.py | unlock:case_push_evidence | `push_list_remote_not_a_string` | push_evidence_invalid |  |
| 196 | `push_list_entries` | validate.py | unlock:case_push_evidence | `push_list_entry_malformed` | push_evidence_invalid |  |
| 197 | `push_response_reaches_back` | validate.py | unlock:case_push_evidence | `push_response_not_reaching_back_before_t_lo` | push_evidence_invalid |  |
| 198 | `push_branch_deletion_pushes_nothing` | validate.py | unlock:case_push_evidence | `push_evidence_of_record_passes` | push_evidence_invalid |  |
| 199 | `push_listed_as_the_response_holds_it` | validate.py | unlock:case_push_evidence | `push_listed_unlike_the_response` | push_evidence_invalid |  |
| 200 | `push_after_t_lo_listed` | validate.py | unlock:case_push_evidence | `push_after_t_lo_unlisted` | push_evidence_invalid |  |
| 201 | `push_list_nothing_else` | validate.py | unlock:case_push_evidence | `push_listed_not_in_the_response` | push_evidence_invalid |  |
| 202 | `push_list_each_once` | validate.py | unlock:case_push_evidence | `push_listed_twice` | push_evidence_invalid |  |
| 203 | `push_list_checked_by_the_clone` | validate.py | unlock:case_push_evidence | `push_carries_file_against_the_clone` | push_evidence_invalid |  |
| 204 | `push_unserved_needs_the_clone` | validate.py | unlock:case_push_evidence | `unserved_push_the_clone_lacks_carries_nothing` | push_evidence_invalid |  |
| 205 | `push_held_by_the_clone_counts` | validate.py | unlock:case_push_evidence | `unserved_push_the_clone_holds_counts` | push_evidence_invalid |  |
| 206 | `push_carrying_the_unlock` | validate.py | unlock:case_push_evidence | `no_push_carries_the_unlock_B57` | push_evidence_invalid |  |
| 207 | `push_not_before_the_unlock_day` | validate.py | unlock:case_push_evidence | `unlock_pushed_before_its_own_day` | push_evidence_invalid |  |
| 208 | `rows_unique` | validate.py | validate:case_inventory | `rows_unique_by_slot_and_precision_IF11_1` | inventory_invalid |  |
| 209 | `inventory_sha256` | validate.py | validate:case_inventory | `inventory_sha256_mismatch` | inventory_sha256_mismatch |  |
| 210 | `binding_sha256` | validate.py | validate:case_inventory | `bound_binding_other_sha256` | binding_invalid |  |
| 211 | `S14_reader_reads_best_val` | records.py | noscore:case_sentinel | `sentinel_quiet_through_every_mode_builder_export_and_printer` | — | 14 |
| 212 | `decision_record_keeps_directories` | records.py | noscore:case_sentinel | `decision_record_keeps_grid_values_and_statuses_only` | — |  |
| 213 | `score_literal_in_the_code` | validate.py | noscore:case_static_scan | `no_score_key_literal_in_the_lane_s_code` | — |  |
| 214 | `records_from_the_folder_only` | records.py | unlock:case_unlock_world | `unlock_world_passes_every_unlock_gate_S_equals_E` | record_uncommitted |  |
| 215 | `unreadable_run_file` | records.py | noscore:case_removal_changes_the_verdict | `telemetry_removed_refused` | completion_record_invalid |  |
| 216 | `test_named_module_parsed` | manifest.py | paths:case_evaluator_closure_never_reads_a_test_named_module | `closure_records_a_test_named_module_by_blob_id_unread` | — |  |
| 217 | `entry_written_over` | entries.py | paths:case_cli_outputs | `a_second_write_is_refused_and_the_first_kept` | output_exists |  |
| 218 | `test_path_resolved_under_its_base` | schema.py | paths:case_paths_resolved_where_read | `record_read_resolved_under_its_root_B113` | test_path |  |
| 219 | `records_base_is_the_folder` | records.py | paths:case_paths_resolved_where_read | `record_read_resolved_under_the_records_folder_B84` | test_path |  |
| 220 | `records_read_under_its_base` | records.py | paths:case_paths_resolved_where_read | `record_read_resolved_under_its_root_B113` | test_path |  |
| 221 | `run_dir_joined` | validate.py | paths:case_paths_resolved_where_read | `run_dir_resolved_under_the_runs_root_B84` | test_path |  |
| 222 | `checkpoint_name_joined` | validate.py | paths:case_paths_resolved_where_read | `checkpoint_name_resolved_under_its_run_directory_B84` | test_path |  |
| 223 | `builder_run_id_joined` | manifest.py | paths:case_paths_resolved_where_read | `builder_run_id_resolved_under_the_runs_root_B84` | test_path |  |
| 224 | `inventory_telemetry_joined` | manifest.py | paths:case_paths_resolved_where_read | `inventory_telemetry_resolved_under_the_runs_root_B84` | test_path |  |
| 225 | `run_index_name_joined` | completion.py | paths:case_paths_resolved_where_read | `run_index_name_resolved_under_the_runs_root_B84` | test_path |  |
| 226 | `run_index_state_joined` | completion.py | paths:case_paths_resolved_where_read | `run_index_state_resolved_under_the_runs_root_B84` | test_path |  |
| 227 | `printer_run_ids_joined` | tday_manifest.py | paths:case_paths_resolved_where_read | `printer_run_ids_resolved_under_the_runs_root_B10` | test_path |  |
| 228 | `cli_path_arguments_as_given` | tday_manifest.py | paths:case_cli_test_named_arguments | `cli_entry_runs_holding_test_B10` | test_path |  |
| 229 | `cli_runs_argument_as_given` | tday_manifest.py | paths:case_cli_test_named_arguments | `cli_append_runs_holding_test_B10` | test_path |  |
| 230 | `cli_root_refused` | tday_manifest.py | paths:case_cli_test_named_arguments | `cli_root_resolving_to_a_test_location_B7` | test_path |  |
| 231 | `cli_ptq_record_unreadable_refused` | tday_manifest.py | paths:case_cli_outputs | `cli_append_ptq_record_absent_refused_B10` | ptq_record_invalid |  |
| 232 | `validator_root_checked` | validate.py | paths:case_cli_outputs | `validator_root_holding_test_refused_before_any_call_B7` | test_path |  |
| 233 | `validator_records_folder_checked` | validate.py | paths:case_cli_outputs | `validator_records_folder_holding_test_refused_before_any_call_B84` | test_path |  |
| 234 | `body_blocks_held_to_their_keys` | validate.py | validate:case_input_gates | `evaluator_holding_a_dedup_list_B29` | manifest_format |  |
| 235 | `body_spec_evaluator` | validate.py | validate:case_input_gates | `evaluator_holding_a_dedup_list_B29` | manifest_format |  |
| 236 | `body_spec_expected_rows` | validate.py | validate:case_input_gates | `expected_rows_holding_counts_B29` | manifest_format |  |
| 237 | `body_spec_pin_block` | validate.py | validate:case_input_gates | `pin_block_holding_a_value_beside_its_pin_B78` | manifest_format |  |
| 238 | `body_spec_pin_blocks_named` | validate.py | validate:case_input_gates | `generator_pins_holding_another_block_B78` | manifest_format |  |
| 239 | `body_spec_lane_code` | validate.py | validate:case_input_gates | `lane_code_holding_a_cells_sha256_B29` | manifest_format |  |
| 240 | `body_spec_trigger` | validate.py | validate:case_input_gates | `trigger_holding_another_key_B29` | manifest_format |  |
| 241 | `body_value_keys_scanned` | manifest.py | validate:case_record_gates | `test_value_key_in_an_entry_B29` | test_value_before_unlock |  |
| 242 | `body_value_gate` | validate.py | validate:case_record_gates | `test_value_key_in_an_entry_B29` | test_value_before_unlock |  |
| 243 | `builder_sources_slots_known` | manifest.py | validate:case_record_gates | `builder_refuses_a_sources_slot_outside_the_catalogue_B31` | slot_unknown |  |
| 244 | `build_validates_before_it_writes` | tday_manifest.py | paths:case_cli_outputs | `cli_build_refused_writes_nothing_B61` | entry_kind_unknown |  |
| 245 | `inventory_recorded_telemetry_checked` | manifest.py | validate:case_inventory | `inventory_launched_cut_telemetry_unlike_its_entry_B32` | completion_source_mismatch |  |
| 246 | `inventory_cut_records_its_telemetry` | manifest.py | validate:case_inventory | `inventory_launched_cut_telemetry_unlike_its_entry_B32` | completion_source_mismatch |  |
| 247 | `inventory_cut_entry_at_its_sha256` | manifest.py | validate:case_inventory | `inventory_cut_entry_of_another_sha256_B32` | record_uncommitted |  |
| 248 | `inventory_freeze_of_this_body` | manifest.py | validate:case_inventory | `inventory_freeze_record_of_another_body_B33` | inventory_invalid |  |
| 249 | `inventory_freeze_record_of_kind_freeze` | manifest.py | validate:case_inventory | `inventory_freeze_record_of_another_kind_B82` | entry_file_invalid |  |
| 250 | `inventory_entries_cover_the_catalogue` | validate.py | validate:case_inventory | `entries_missing_a_catalogue_slot_B105` | inventory_invalid |  |
| 251 | `inventory_rows_of_scored_entries` | validate.py | validate:case_inventory | `model_row_of_a_slot_entered_cut_B105` | inventory_invalid |  |
| 252 | `inventory_rows_under_their_file_id` | validate.py | validate:case_inventory | `model_row_file_id_not_its_slot_s_B107` | inventory_invalid |  |
| 253 | `inventory_rows_unique_by_file_id` | validate.py | validate:case_inventory | `rows_unique_by_file_id_and_precision_B107` | inventory_invalid |  |
| 254 | `inventory_context_at_tb1_values` | validate.py | validate:case_inventory | `execution_context_off_tb1_s_values_B107` | inventory_invalid |  |
| 255 | `inventory_reader_reads_the_catalogue` | validate.py | validate:case_inventory | `entries_missing_a_catalogue_slot_B105` | inventory_invalid |  |
| 256 | `inventory_rows_hashed_when_well_formed` | validate.py | validate:case_inventory | `model_row_slot_unhashable_B104` | inventory_invalid |  |
| 257 | `inventory_entry_slots_hashable` | validate.py | validate:case_inventory | `entries_item_slot_an_object_B104` | inventory_invalid |  |
| 258 | `inventory_reader_unreadable_refused` | validate.py | validate:case_inventory | `read_inventory_absent_export_B106` | inventory_invalid |  |
| 259 | `strict_json_finite_numbers` | schema.py | validate:case_malformed_inputs_refused | `number_overflowing_to_infinity_refused_B85` | manifest_unreadable |  |
| 260 | `day_of_the_calendar` | schema.py | validate:case_malformed_inputs_refused | `days_and_instants_of_the_calendar_B86` | entry_field |  |
| 261 | `instant_of_the_calendar` | schema.py | validate:case_malformed_inputs_refused | `days_and_instants_of_the_calendar_B86` | entry_field |  |
| 262 | `telemetry_rows_are_objects` | records.py | validate:case_malformed_inputs_refused | `telemetry_row_not_an_object_B83` | completion_record_invalid |  |
| 263 | `arm_measured_from_its_record` | completion.py | validate:case_malformed_inputs_refused | `arm_slot_read_as_a_run_directory_refused_B67` | completion_record_invalid |  |
| 264 | `teacher_work_dir_a_folder` | completion.py | validate:case_malformed_inputs_refused | `teacher_work_dir_absent_B67_B100` | completion_record_invalid |  |
| 265 | `arm_instant_of_the_calendar` | completion.py | validate:case_malformed_inputs_refused | `arm_instant_off_the_calendar_refused_B67` | completion_record_invalid |  |
| 266 | `qat_epochs_are_ints` | completion.py | validate:case_malformed_inputs_refused | `qat_epoch_not_an_int_refused_B67` | completion_record_invalid |  |
| 267 | `schedule_file_a_path` | validate.py | validate:case_malformed_inputs_refused | `schedule_file_not_a_path_B90` | manifest_format |  |
| 268 | `append_slot_a_key` | validate.py | validate:case_malformed_inputs_refused | `append_slot_not_a_key_B96` | append_invalid |  |
| 269 | `freeze_source_cited_by_path` | validate.py | validate:case_malformed_inputs_refused | `freeze_source_not_an_object_B90` | schedule_invalid |  |
| 270 | `cli_unreadable_file_refused` | tday_manifest.py | paths:case_cli_outputs | `cli_validate_manifest_absent_B90` | manifest_unreadable |  |
| 271 | `inventory_checked_before_it_is_written` | tday_manifest.py | paths:case_cli_outputs | `cli_inventory_refused_writes_nothing_B61` | inventory_invalid |  |
| 272 | `out_inside_the_repository` | tday_manifest.py | paths:case_cli_outputs | `out_in_the_repository_at_root_refused` | output_location |  |
| 273 | `output_exists_code_of_the_cli` | tday_manifest.py | paths:case_cli_outputs | `a_second_build_is_refused_and_the_first_kept_B110` | output_exists |  |
| 274 | `output_exists_code_of_the_printer` | entries.py | cuts:case_printers_and_committed_files | `printer_writes_once_B110` | output_exists |  |
| 275 | `output_location_code` | tday_manifest.py | paths:case_cli_outputs | `out_in_the_repository_at_root_refused` | output_location |  |
| 276 | `write_entry_checks_the_created_name_only` | entries.py | cuts:case_printers_and_committed_files | `printer_writes_into_a_folder_whose_path_holds_a_colon_B72` | test_path |  |
| 277 | `make_entry_common_keys` | entries.py | validate:case_printers | `printer_fields_naming_a_common_key_B71` | entry_field |  |
| 278 | `printer_checks_run` | tday_manifest.py | validate:case_moved_freeze | `printer_freeze_delay_listing_a_complete_run_B8` | freeze_source_conflict |  |
| 279 | `printer_late_runs_from_the_records` | tday_manifest.py | validate:case_moved_freeze | `printer_freeze_delay_listing_a_complete_run_B8` | freeze_source_conflict |  |
| 280 | `freeze_delay_runs` | entries.py | validate:case_moved_freeze | `delay_listing_a_complete_run_too_B93` | freeze_source_conflict |  |
| 281 | `new_freeze_first_day_of_the_printer` | entries.py | validate:case_moved_freeze | `printer_new_freeze_a_day_late_B8` | freeze_source_conflict |  |
| 282 | `printer_not_run_slots` | tday_manifest.py | validate:case_printers | `printer_not_run_naming_an_unknown_slot_B68` | slot_unknown |  |
| 283 | `printer_not_required_coverage` | tday_manifest.py | validate:case_printers | `printer_not_required_slot_without_a_cut_entry_B9` | not_required_invalid |  |
| 284 | `printer_not_required_cut_kind` | tday_manifest.py | validate:case_printers | `printer_not_required_citing_a_not_run_entry_as_a_cut` | entry_file_invalid |  |
| 285 | `printer_unlock_manifest_file` | tday_manifest.py | unlock:case_unlock_printer | `printer_unlock_of_another_manifest_file_B9` | run_added_after_freeze |  |
| 286 | `printer_unlock_freeze_record_body` | tday_manifest.py | unlock:case_unlock_printer | `printer_unlock_citing_the_freeze_record_of_another_body_B9` | run_added_after_freeze |  |
| 287 | `printer_unlock_lanes` | tday_manifest.py | unlock:case_unlock_printer | `printer_unlock_citing_another_lane_register_B9` | lane_unmet |  |
| 288 | `printer_unlock_second` | tday_manifest.py | unlock:case_unlock_printer | `printer_unlock_after_an_unlock_entry_B9` | second_session |  |
| 289 | `printer_condition_slot` | tday_manifest.py | validate:case_printers | `printer_condition_naming_another_arm_s_slot_B16` | entry_field |  |
| 290 | `printer_not_made_gated` | tday_manifest.py | validate:case_printers | `printer_not_made_for_a_slot_without_a_code_gate_B9` | derived_missing |  |
| 291 | `printer_not_made_class` | tday_manifest.py | validate:case_printers | `printer_not_made_naming_a_run_slot` | derived_counted_as_run |  |
| 292 | `printer_freeze_record_body` | tday_manifest.py | validate:case_printers | `printer_freeze_record_of_another_body_B9` | run_added_after_freeze |  |
| 293 | `printer_clock_offset` | tday_manifest.py | validate:case_printers | `printer_clock_offset_over_60_s_B9` | clock_offset_invalid |  |
| 294 | `printer_named_rule_record` | tday_manifest.py | validate:case_printers | `printer_named_rule_cut_citing_an_unknown_record_B16` | cut_rule_unknown |  |
| 295 | `teacher_logs_unsorted` | completion.py | determinism:case_listing_order | `shuffled_listing_4_byte_identical` | — |  |
| 296 | `inputs_parent_checkpoint` | manifest.py | validate:case_derived_sources | `derived_parent_checkpoint_not_its_parent_s_B22` | derived_inputs_mismatch |  |
| 297 | `inputs_calibration_list` | manifest.py | validate:case_derived_sources | `derived_calibration_list_not_the_index_s_B22` | derived_inputs_mismatch |  |
| 298 | `inputs_script_path` | manifest.py | validate:case_derived_sources | `derived_script_not_its_ptq_script_B53` | derived_inputs_mismatch |  |
| 299 | `inputs_commit_in_history` | manifest.py | validate:case_derived_sources | `derived_script_commit_outside_the_records_history` | derived_inputs_mismatch |  |
| 300 | `inputs_qat_telemetry` | manifest.py | validate:case_derived_sources | `qat_telemetry_not_its_run_s_completion_B22` | derived_inputs_mismatch |  |
| 301 | `ptq_script_of_an_arm` | manifest.py | derived:case_inputs_and_parents | `ptq_script_of_an_arm_s_e7_is_run_ptq_arm_B53` | derived_inputs_mismatch |  |
| 302 | `ptq_script_of_a_lane_s_slot` | manifest.py | derived:case_inputs_and_parents | `ptq_script_of_a_slot_whose_code_is_not_merged_is_its_lane_s` | — |  |
| 303 | `hashed_files_cited` | manifest.py | validate:case_derived_sources | `ptq_provenance_not_the_cited_sha256_B50` | derived_hash_mismatch |  |
| 304 | `hashed_selected_checkpoint_read` | manifest.py | validate:case_derived_sources | `qat_selected_checkpoint_absent_B49` | derived_hash_mismatch |  |
| 305 | `ptq_provenance_parent` | manifest.py | validate:case_derived_sources | `ptq_provenance_naming_other_parent_B27` | derived_hash_mismatch |  |
| 306 | `ptq_provenance_index_fields` | manifest.py | validate:case_derived_sources | `ptq_provenance_naming_other_checksum_B27` | derived_hash_mismatch |  |
| 307 | `ptq_provenance_artifact` | manifest.py | validate:case_derived_sources | `ptq_provenance_naming_other_artifact_B27` | derived_hash_mismatch |  |
| 308 | `qat_record_winner_checkpoint` | manifest.py | validate:case_derived_sources | `qat_record_naming_winner_of_another_checkpoint_B34` | derived_hash_mismatch |  |
| 309 | `qat_record_telemetry` | manifest.py | validate:case_derived_sources | `qat_record_naming_other_telemetry_B34` | derived_hash_mismatch |  |
| 310 | `qat_record_clip_norm_of_its_run` | manifest.py | validate:case_derived_sources | `qat_record_naming_other_clip_norm_B34` | derived_hash_mismatch |  |
| 311 | `qat_selected_checkpoint_row` | manifest.py | validate:case_derived_sources | `qat_record_naming_winner_of_another_epoch_B34` | derived_hash_mismatch |  |
| 312 | `clip_selection_cited` | manifest.py | validate:case_derived_sources | `clip_selection_not_the_cited_sha256_B34` | derived_hash_mismatch |  |
| 313 | `clip_selection_rules` | manifest.py | validate:case_derived_sources | `clip_selection_of_other_rules_B34` | derived_hash_mismatch |  |
| 314 | `pilot_winner_TB9` | manifest.py | validate:case_derived_sources | `pilot_selection_of_the_retained_loser_TB9_B34` | derived_hash_mismatch |  |
| 315 | `pilot_winner_record_TB9` | manifest.py | validate:case_derived_sources | `pilot_winner_naming_another_record_TB9` | derived_hash_mismatch |  |
| 316 | `clip_norm_of_the_winner` | manifest.py | validate:case_derived_sources | `run_clip_not_the_clip_winner_s_B34` | derived_hash_mismatch |  |
| 317 | `derived_hash_of_appends` | validate.py | validate:case_appends | `appended_model_whose_provenance_names_another_parent_B27` | derived_hash_mismatch |  |
| 318 | `append_made_by_the_freeze` | manifest.py | validate:case_appends | `append_made_by_the_freeze_B25` | append_invalid |  |
| 319 | `append_first_pass_structural` | validate.py | validate:case_appends | `post_freeze_qat_append_with_its_isolation_record_passes_B26` | append_invalid |  |
| 320 | `cli_append_first_pass_structural` | tday_manifest.py | paths:case_cli_outputs | `cli_append_of_a_qat_selection_made_after_the_freeze_B26` | append_invalid |  |
| 321 | `ptq_made_instant` | manifest.py | validate:case_appends | `appended_made_utc_not_its_provenance_s_B25` | derived_hash_mismatch |  |
| 322 | `qat_made_instant` | manifest.py | validate:case_derived_sources | `qat_provenance_made_at_another_time_B25_B50` | derived_hash_mismatch |  |
| 323 | `qat_provenance_artifact` | manifest.py | validate:case_derived_sources | `qat_provenance_of_another_artifact_B25_B50` | derived_hash_mismatch |  |
| 324 | `qat_provenance_epoch` | manifest.py | validate:case_derived_sources | `qat_provenance_of_another_epoch_B25_B50` | derived_hash_mismatch |  |
| 325 | `appends_history_pairs` | manifest.py | validate:case_appends_history | `committed_append_rewritten_in_a_later_commit_B28` | append_invalid |  |
| 326 | `appends_history_newest` | manifest.py | validate:case_appends_history | `candidate_appends_behind_the_committed_B54` | append_invalid |  |
| 327 | `validator_reads_the_appends_history` | validate.py | validate:case_appends_history | `candidate_rewriting_a_committed_append_B54` | append_invalid |  |
| 328 | `cli_append_history` | tday_manifest.py | paths:case_cli_outputs | `cli_append_to_a_manifest_behind_the_committed_appends_B54` | append_invalid |  |
| 329 | `cli_append_derived_hash` | tday_manifest.py | paths:case_cli_outputs | `cli_append_made_utc_not_its_provenance_s_B25` | derived_hash_mismatch |  |
| 330 | `checkpoint_held_by_a_committed_source` | validate.py | validate:case_checkpoint_sources | `checkpoint_without_a_committed_source_B35` | checkpoint_hash_mismatch |  |
| 331 | `checkpoint_source_row_recorded` | validate.py | validate:case_checkpoint_sources | `checkpoint_source_row_not_recorded_B51` | checkpoint_hash_mismatch |  |
| 332 | `checkpoint_held_by_its_selection_file` | validate.py | validate:case_checkpoint_sources | `default_candidate_held_by_its_selection_file_passes` | — |  |
| 333 | `arm_teacher_records_read` | validate.py | validate:case_arm_teacher | `arm_teacher_go_rule_verdict_not_the_record_s_B39` | checkpoint_hash_mismatch |  |
| 334 | `arm_record_selection_checkpoint` | manifest.py | validate:case_arm_teacher | `arm_teacher_checkpoint_not_the_arm_record_s_selection_B39` | checkpoint_hash_mismatch |  |
| 335 | `arm_record_complete` | manifest.py | validate:case_arm_teacher | `arm_teacher_record_incomplete_B39` | checkpoint_hash_mismatch |  |
| 336 | `arm_record_final_selection` | manifest.py | validate:case_arm_teacher | `arm_teacher_without_a_final_selection_B39` | checkpoint_hash_mismatch |  |
| 337 | `arm_record_attempt` | manifest.py | validate:case_arm_teacher | `arm_teacher_record_of_another_attempt_B39` | checkpoint_hash_mismatch |  |
| 338 | `arm_launch_provenance_arm` | manifest.py | validate:case_arm_teacher | `arm_teacher_launch_provenance_of_another_arm_B39` | checkpoint_hash_mismatch |  |
| 339 | `arm_go_rule_verdict` | manifest.py | validate:case_arm_teacher | `arm_teacher_go_rule_verdict_not_the_record_s_B39` | checkpoint_hash_mismatch |  |
| 340 | `measurement_files_present` | validate.py | validate:case_measurement | `arm_measurement_files_absent_B36` | completion_record_invalid |  |
| 341 | `measurement_files_cited` | validate.py | validate:case_measurement | `arm_measurement_output_not_the_cited_B36` | completion_source_mismatch |  |
| 342 | `measurement_completion_block` | validate.py | validate:case_measurement | `arm_measurement_completion_block_not_its_record_B36` | completion_source_mismatch |  |
| 343 | `measurement_complete_at_its_instant` | validate.py | validate:case_measurement | `arm_measurement_after_its_instant_B36` | entry_not_complete |  |
| 344 | `aborted_names_the_last_two_attempts` | validate.py | validate:case_aborted | `aborted_entry_naming_one_attempt_B43` | completion_source_mismatch |  |
| 345 | `aborted_runs_not_complete` | validate.py | validate:case_aborted | `complete_run_entered_as_aborted_B43` | completion_source_mismatch |  |
| 346 | `aborted_rule_and_cause` | validate.py | validate:case_aborted | `aborted_entry_of_another_rule_B43` | completion_source_mismatch |  |
| 347 | `kd_run_meta_of_the_launched_line` | validate.py | validate:case_run_identity | `kd_run_meta_not_the_launched_line_s_B46` | completion_source_mismatch |  |
| 348 | `qat_run_of_its_parent_checkpoint` | validate.py | validate:case_run_identity | `qat_run_of_another_parent_checkpoint_B52` | completion_source_mismatch |  |
| 349 | `session_on_the_frozen_host` | validate.py | unlock:case_session | `session_host_not_the_frozen_host_B37` | clock_offset_invalid |  |
| 350 | `clock_offset_record_of_the_session_host` | validate.py | unlock:case_session | `clock_offset_record_of_another_host_B37` | clock_offset_invalid |  |
| 351 | `second_session_over_history` | validate.py | unlock:case_session | `second_session_part_over_the_first_B56` | second_session |  |
| 352 | `second_unlock_after_a_session` | validate.py | unlock:case_session | `unlock_after_a_committed_session_part_refused_B56` | second_session |  |
| 353 | `session_values_only_added` | validate.py | unlock:case_session | `session_values_removed_B56` | second_session |  |
| 354 | `session_identity_fields` | validate.py | unlock:case_session | `second_session_part_over_the_first_B56` | second_session |  |
| 355 | `session_part_read_first` | validate.py | unlock:case_session | `session_part_null_refused_B101` | clock_offset_invalid |  |
| 356 | `closure_custom_imports` | manifest.py | paths:case_evaluator_closure_never_reads_a_test_named_module | `closure_follows_a_config_s_custom_imports_B81` | — |  |
| 357 | `closure_bases` | manifest.py | paths:case_evaluator_closure_never_reads_a_test_named_module | `closure_follows_a_config_s_bases_in_the_three_roots_B81` | — |  |
| 358 | `closure_bases_in_the_three_roots` | manifest.py | paths:case_evaluator_closure_never_reads_a_test_named_module | `closure_follows_a_config_s_bases_in_the_three_roots_B81` | — |  |
| 359 | `closure_arm_teacher_configs` | manifest.py | paths:case_evaluator_closure_never_reads_a_test_named_module | `closure_holds_the_arm_teacher_configs_B81` | — |  |
| 360 | `arm_registry_at_s` | entries.py | validate:case_cut_gates | `arm_registry_absent_at_S_B95` | unregistered_arm |  |
| 361 | `arm_registry_files_committed` | entries.py | validate:case_cut_gates | `arm_registry_file_uncommitted_at_S_B95` | record_uncommitted |  |
| 362 | `arm_registry_refusal_named` | entries.py | validate:case_cut_gates | `arm_registry_absent_at_S_B95` | unregistered_arm |  |
| 363 | `arm_teacher_first_nonfinite_iteration` | records.py | schema:case_readers | `arm_teacher_record_keeps_the_first_non_finite_iteration_only_B112` | — |  |
| 364 | `instant_moved_for_every_slot` | validate.py | validate:case_moved_freeze | `cuttable_run_complete_between_the_freezes_R016_B58` | entry_not_complete |  |
| 365 | `freeze_delay_lists_every_late_run` | entries.py | validate:case_moved_freeze | `delay_missing_the_late_run_B93` | freeze_source_conflict |  |
| 366 | `freeze_delay_needs_a_late_run` | entries.py | validate:case_moved_freeze | `delay_when_no_never_cut_run_is_late_B93` | freeze_source_conflict |  |
| 367 | `pair_models_at_the_gate` | validate.py | validate:case_scope_gates | `validator_requires_the_corrupted_scope_of_a_pair_with_both_models_TB2_B97` | scope_required_missing |  |
| 368 | `pair_models_in_the_builder` | manifest.py | validate:case_scope_gates | `builder_gives_a_pair_with_both_models_the_corrupted_scope_TB2_B97` | — |  |
| 369 | `generator_script_at_s_against_its_pin` | validate.py | unlock:case_generator_pin_at_another_commit | `generator_script_not_its_pin_G2_B98` | hasher_pin_mismatch |  |
| 370 | `test_deadline_from_the_resolver` | validate.py | validate:case_moved_freeze | `moved_freeze_moves_the_test_deadline_at_the_gate_TB14_B99` | session_after_deadline |  |
| 371 | `evaluator_roots_of_if10` | manifest.py | paths:case_evaluator_closure_never_reads_a_test_named_module | `evaluator_roots_are_decision_x_s_scripts_and_IF10_s_B79` | — |  |
| 372 | `kd_slot_at_a_seed_the_launcher_lacks` | tday_catalogue.json | schema:case_catalogue_cross_checks | `kd_slots_at_the_launcher_s_seeds_and_profiles_I1_B60` | — |  |
| 373 | `unlock_gate_reading_a_score` | validate.py | noscore:case_mutation_invariance | `mutation_finite_unlock_and_session_verdicts_identical_B59` | — |  |
| 374 | `append_printer_reading_a_score` | tday_manifest.py | noscore:case_mutation_invariance | `mutation_finite_printed_freeze_record_and_append_identical_but_hashes_B59` | — |  |

## 10. The DL-24 verification workflow and its findings

**Run `wf_d9d76687-91f`** (MEASURED), a read-only verification workflow. It ran once, resumed after the usage limits.
- **Launch 1** (17 Oct): its 7 reviewers all failed on the weekly limit.
- **Launch 2** (26 Oct): 7 reviewers produced 115 findings, 114 after deduplication. 90 verifiers failed on the session limit.
- **Launch 3:** 122 of 122 agents done, 0 errors, 17.3M subagent tokens.
- **Before and after each launch:** the 22 lane files' sha256 matched, HEAD stayed `0728f93`, and the scoped status was unchanged.
- **Call audit of the agents** (DL-113 pattern, launches 2 and 3: 212 agents, 5,916 calls):
  - no Write or Edit outside the workflow's work folder;
  - no test-named path read, and nothing under `docs/reference` read;
  - slips: 5 Grep calls rooted at the repository with a brace glob of named lane files (one launch-2 reviewer); one `grep -o` on the decision log; one `grep -rn` on a named file; one for-loop of `grep -n` over 7 listed docs files; one `rm -rf` of a verifier's own `/tmp` scratch folders;
  - two L-PROT G refusals, on a variable path in a `git cat-file` loop, neither retried.

**Verdicts:**
- 108 confirmed: 6 blocker, 52 major, 45 minor, 5 nit.
- 6 refuted, named here:
  - **#64** (the inventory export trusts its input);
  - **#65** (a KD abort missing after a non-finite row);
  - **#66** (teacher_state never DIVERGED);
  - **#73** (write_entry with ':' in `--out`; its twin #72 is confirmed);
  - **#80** (roots absent at E dropped);
  - **#109** (the mutation baseline never runs the preflight_arms copy).

**The fixes** are commits C6a–C6h2 (§2), each with its cases and mutations:
- **#92** needed nothing beyond C6c: its two validate cases are killed by `entry_file_dl_row` and `entry_file_written_once`, and since C6g3 `read_entry_file` has a production caller.
- **#103** is closed by ruling TMAN-B1-RULE-Q item 2 (§18).

| # | severity | verdict | finding | fixed by |
|---|---|---|---|---|
| 0 | blocker | confirmed | The cut checks take a cut slot's launch and completion state from the entry's own run_ids, so a run complete at the f... | C6a |
| 1 | blocker | confirmed | The trigger block is copied unverified and never checked against the AM-3 record, so E6-KD can be dropped in every mode | C6b |
| 2 | blocker | confirmed | Entry kind is not tied to the slot's role: a family or teacher slot entered as listed_run is never scored, and every ... | C6b |
| 3 | blocker | confirmed | Freeze mode passes while a never-cut run is absent or not complete and the freeze has not moved | C6a |
| 4 | blocker | confirmed | (adjacent: completion gate) complete_run_omitted covers only KD launch-log runs, so the teacher of record or E1/QAT r... | C6a |
| 5 | blocker | confirmed | A cut entry naming several runs judges every slot by its last run_id, so a complete run is cut | C6a |
| 6 | major | confirmed | Mutation S03 is broken (list + tuple TypeError) and is the only mutation counted as killing cut_rule_unknown | C6b |
| 7 | major | confirmed | --root is never refused for "test"; files under it are stat-ed, read and parsed | C6g |
| 8 | major | confirmed | The item 1(d) printers (entry freeze_delay / new_freeze) do not run 1(d)'s checks; freeze_delay_errors and new_freeze... | C6c3 |
| 9 | major | confirmed | entry printers write freeze_delay, new_freeze, unlock, not_required, freeze, clock_offset, condition and not_made ent... | C6c3 |
| 10 | major | confirmed | --runs holding "test" is not refused by the entry, inventory and append commands; filesystem calls reach it first | C6g |
| 11 | major | confirmed | The arithmetic recomputation only re-checks the entry's own sums: iterations, the reference figures and their sources... | C6c4 |
| 12 | major | confirmed | 'Launched' comes only from the KD launch log, so a launched QAT, E1-control or teacher-arm run cannot be truthfully c... | C6a |
| 13 | major | confirmed | Cut checks trust the entry's own run_ids, so a launched run can be cut as 'never launched', and the validate smoke's ... | C6a |
| 14 | major | confirmed | cut_reason_invalid refuses a launched cut of any run outside the KD launch log | C6a |
| 15 | major | confirmed | Putting slots of two cut groups in one entry switches off every arm check | C6c |
| 16 | major | confirmed | A named-rule cut's rule record is never checked, and the AM-23 condition verdict is read without its arm | C6c2 |
| 17 | major | confirmed | Not-run, not-made and freeze-delay entry files are never field-checked or matched to the manifest; read_entry_file's ... | C6c |
| 18 | major | confirmed | TB13: configs/qat_selection_rules.json's sha256 is never read at build or checked | C6d |
| 19 | major | confirmed | TB13's rules-file sha256 is never read: QAT_RULES_REL is defined but unused | C6d |
| 20 | major | confirmed | TB13's rules-file sha256 is never read: a QAT selection's inputs.rules_sha256 can be anything | C6d |
| 21 | major | confirmed | A derived entry's `method` is never tied to its slot's record_kind; a PTQ model of record entered as qat_selection sk... | C6b |
| 22 | major | confirmed | By-inputs listings are never cross-checked against the parent, the calibration index or the run's telemetry | C6d2 |
| 23 | major | confirmed | A not_made entry is accepted on the catalogue's static gate alone; its cited entry file and the lane's merge state ar... | C6c |
| 24 | major | confirmed | derived_selection_conflict (AM-21 item 2(c)) is never applied to appended QAT selections | C6d |
| 25 | major | confirmed | Whether an append needs an isolation record depends on its own unverified made_utc | C6d3 |
| 26 | major | confirmed | A QAT selection appended after the freeze is always refused, even with a valid isolation record | C6d3 |
| 27 | major | confirmed | The hashed form is never checked against the provenance or record it cites (PTQ provenance read only for data_root; Q... | C6d2 |
| 28 | major | confirmed | Appends' prefix-only growth (plan §1.1) is never checked; appends_prefix_error has no caller | C6d3 |
| 29 | major | confirmed | test_value_before_unlock scans only a pin's value and the keys inside the pin | C6g2 |
| 30 | major | confirmed | The calibration index is never compared with DL-10, and its seed and role are never checked | C6d |
| 31 | major | confirmed | The builder silently drops sources entries whose key is not a catalogue slot | C6g2 |
| 32 | major | confirmed | Inventory telemetry for cut runs is not compared with the hashes the cut entry records, and missing attempt files are... | C6g3 |
| 33 | major | confirmed | The inventory's `freeze` block comes from an unverified --freeze-dl flag | C6g3 |
| 34 | major | confirmed | QAT selection records are never read: the E5 pilot winner and every QAT derived hash go unchecked | C6d2 |
| 35 | major | confirmed | A checkpoint needs no committed source: sources may be empty or self-declared | C6e |
| 36 | major | confirmed | A measurement (ARM) entry's completion is never checked; arm_state has no production caller | C6e2 |
| 37 | major | confirmed | The session's host is never compared with the frozen execution context's host | C6f |
| 38 | major | confirmed | Not-run and not-made entries may cite any decision-log entry file as their record | C6c |
| 39 | major | confirmed | Arm-teacher entries are never compared with their arm record or go-rule record | C6e |
| 40 | major | confirmed | calibration_index_mismatch never compares the lists with DL-10, nor their seed and role | C6d |
| 41 | major | confirmed | trigger_format_unknown is not implemented and the AM-3 trigger record is never read, so E6-KD (never cut if triggered... | C6b |
| 42 | major | confirmed | The AM-3 trigger record is never read: its sha256 and triggered verdict are taken from the manifest | C6b |
| 43 | major | confirmed | An aborted entry is never compared with its runs' records; only diverged entries are | C6e2 |
| 44 | major | confirmed | A cut entry's run_ids need no directory under --runs, so a complete run can be cut by withholding its directory | C6a |
| 45 | major | confirmed | completion_source_mismatch omits two of its three comparisons, and the run's identity is never compared | C6e2 |
| 46 | major | confirmed | The launched line's run_meta sha256 is never compared with the supplied run directory | C6e2 |
| 47 | major | confirmed | complete_run_omitted covers only KD runs in the launch log; a complete E1, teacher, teacher-arm or QAT run can be omi... | C6a |
| 48 | major | confirmed | complete_run_omitted enumerates only KD launch-log runs, so freeze passes a manifest that omits a complete QAT run or... | C6a |
| 49 | major | confirmed | Derived-artifact and arm-teacher hashes are never checked against the records the plan names as their sources; the fi... | C6d2 |
| 50 | major | confirmed | Derived entries' hashed artifacts are never checked against their provenance, selection record or files | C6d2 |
| 51 | major | confirmed | A checkpoint with no committed source passes checkpoint_hash_mismatch | C6e |
| 52 | major | confirmed | An entry's run directory is never tied to its slot's identity (stage, seed, lambda/alpha, horizon, mode) | C6e2 |
| 53 | major | confirmed | A derived entry's listed inputs are not tied to its parent, the calibration index or its run's telemetry | C6d2 |
| 54 | major | confirmed | Appends are not checked against the manifest's history: a committed append can be rewritten before the unlock entry | C6d3 |
| 55 | major | confirmed | The lane register gate can pass vacuously, and a not-required entry need not be in the decision log | C6c, C6f2 |
| 56 | major | confirmed | second_session does not see a second session record in the manifest's history | C6f |
| 57 | major | confirmed | Push evidence does not follow the DL-109 pattern: the activity response is never read | C6f3 |
| 58 | minor | confirmed | Plan sample 9 is remapped and R-016's fixture is missing: the opposite mutation (moved instant for every slot) survives | C6h |
| 59 | minor | confirmed | The noscore mutation test compares only the freeze and draft verdicts, not every mode and printer | C6h2 |
| 60 | minor | confirmed | The catalogue cross-check omits preflight_distill.SEEDS and PROFILES | C6h |
| 61 | minor | confirmed | build and inventory write into --out before validating; a refused run leaves the file | C6g2, C6g3 |
| 62 | minor | confirmed | Cut printer's status reader uses the entry's last run id for every slot | C6a |
| 63 | minor | confirmed | The append command accepts a second append of an already-appended slot | C6d |
| 64 | minor | refuted | The inventory export trusts its input: built from an unvalidated manifest, written before its read-back check, --runs... | refuted (named) |
| 65 | minor | refuted | A KD run whose run_abort record is missing after a non-finite train row reads as running, while sweep_select reads it... | refuted (named) |
| 66 | minor | refuted | teacher_state never yields DIVERGED, so an AM-18 item 2 teacher-arm divergence cannot be entered as diverged | refuted (named) |
| 67 | minor | confirmed | Completion readers raise non-TmanRefused exceptions on absent or malformed run records | C6g4 |
| 68 | minor | confirmed | Malformed inputs crash where a refusal is due | C6c3, C6c4, C6g4 |
| 69 | minor | confirmed | 'Results of record known at that time' are required only for arm cuts and are never verified | C6c |
| 70 | minor | confirmed | stopped_run_cut_early refuses the decision-date cut AM-19 item 2(b) prescribes for a stopped candidate | C6a |
| 71 | minor | confirmed | make_entry lets the fields file override kind, date, slots and printed_by after the kind was checked | C6c3 |
| 72 | minor | confirmed | write_entry applies the created-name check to the whole output path, so every Windows --out is refused | C6c3 |
| 73 | minor | refuted | write_entry applies the created-name rule to the --out folder, so any ':' in it, including every Windows drive path, ... | refuted (named) |
| 74 | minor | confirmed | The freeze-delay and new-freeze printers skip the checks entries.py defines for them, and freeze_delay_errors disagre... | C6c3 |
| 75 | minor | confirmed | A cut entered in the decision log before its rule's day is neither refused nor reported | C6c |
| 76 | minor | confirmed | TB8's 'not made: code not merged' is never verified: code_gated() is static | C6c |
| 77 | minor | confirmed | derived_selection_conflict accepts a fault report that is not a committed record | C6d |
| 78 | minor | confirmed | test_value_before_unlock inspects only each pin block's `value` and the keys inside `pin`; values beside them pass | C6g2 |
| 79 | minor | confirmed | TB6's IF§10 evaluator roots are not pinned by any case | C6h |
| 80 | minor | refuted | Roots absent at E are dropped silently, and freeze never ties the generator pin's script blob to E | refuted (named) |
| 81 | minor | confirmed | The evaluator file set misses modules the scoring imports through an mmengine config's custom_imports (src/training/t... | C6g5 |
| 82 | minor | confirmed | inventory --freeze-dl is copied into the export unchecked | C6g3 |
| 83 | minor | confirmed | A non-object telemetry row is projected to None and crashes the completion readers | C6g4 |
| 84 | minor | confirmed | test_path_error resolves relative parts against the process CWD, not the path actually read | C6g |
| 85 | minor | confirmed | strict_json admits overflowing number literals as +-inf | C6g4 |
| 86 | minor | confirmed | Field types day/utc are regex-only; impossible dates pass and later crash | C6g4 |
| 87 | minor | confirmed | Cited entry files are not checked as tday_entry/1: not_run/not_made can cite any file, TB7 names go unchecked, a non-... | C6c |
| 88 | minor | confirmed | Freeze-source entry files and the session deadline gate crash instead of refusing | C6c |
| 89 | minor | confirmed | Validator never field-checks or name-checks committed non-cut entry files | C6c |
| 90 | minor | confirmed | Other malformed inputs crash (exit 4) where a named refusal is due | C6g4 |
| 91 | minor | confirmed | An entry's evaluator_stage is never compared with the catalogue's | C6b |
| 92 | minor | confirmed | The validator's entry-file checks (RECORDED row; written once) have no effective case; the cuts smoke tests entries.r... | C6c |
| 93 | minor | confirmed | The validator's delay-runs comparison (AM-19 item 1(d)) has no effective smoke case | C6c3, C6h |
| 94 | minor | confirmed | A launched, complete run outside the catalogue is skipped silently; no smoke cross-checks the catalogue with SEEDS/PR... | C6a, C6h |
| 95 | minor | confirmed | The arm registry used by unregistered_arm is read from the CLI's own working tree, not HEAD of --root (S) | C6g5 |
| 96 | minor | confirmed | Malformed appends crash the validator and the inventory builder instead of being refused | C6g4 |
| 97 | minor | confirmed | TB2's 'both runs of the seed have a model' branch has no validator-level case (validate._models → False survives) | C6h |
| 98 | minor | confirmed | hasher_pin_mismatch's generator-script branch has no effective case | C6h |
| 99 | minor | confirmed | session_after_deadline under a moved freeze (TB14 'moved as the resolver moves it') has no gate-level case | C6h |
| 100 | minor | confirmed | Inputs that should be refused crash the validator instead (exception other than TmanRefused) | C6c, C6f, C6g2, C6g4 |
| 101 | minor | confirmed | Session mode crashes on a null or malformed session part instead of refusing | C6f |
| 102 | minor | confirmed | The clock-offset entry file need not be in the decision log or written once | C6c |
| 103 | minor | confirmed | Two of the plan's three REPORT lines are not implemented | dropped by ruling TMAN-B1-RULE-Q item 2 (no proxy; readings not adopted) |
| 104 | minor | confirmed | read_inventory raises TypeError or FileNotFoundError instead of refusing | C6g3 |
| 105 | minor | confirmed | read_inventory does not check one entries item per catalogue slot | C6g3 |
| 106 | minor | confirmed | read_inventory raises FileNotFoundError on an absent export instead of refusing | C6g3 |
| 107 | minor | confirmed | read_inventory crashes on missing or malformed exports and skips IF§8/IF§11 checks | C6g3 |
| 108 | nit | confirmed | Scratch folders leak when a smoke or mutation worker is killed; eight stale folders from 2026-10-17 remain in /tmp | C6g5 |
| 109 | nit | refuted | The mutation baseline never runs the scratch copy of preflight_arms that S05's worker loads | refuted (named) |
| 110 | nit | confirmed | Refusal codes that misstate the cause | C6c3 |
| 111 | nit | confirmed | arithmetic --reference-stage defaults to "", so a KD reference run is always refused as completion_record_invalid | C6c4 |
| 112 | nit | confirmed | ARM_TEACHER_RECORD would keep first_nonfinite whole, a block that carries a score key name and the offending value | C6g5 |
| 113 | nit | confirmed | test_path_error resolves repo-relative record paths against the process CWD | C6g |

## 11. The SL-1 log

`tail -n 80 ~/.claude/sl1_guard.log` (MEASURED at 2026-10-27T08:53:05Z; 61 lines, all of the log), verbatim:

```
2026-10-16T09:44:31Z	Bash	SL1-B2	git ls-files -- src/tman.py src/tman scripts/tday_manifest.py configs/tday_catalogue.json scripts/smoke_tman_schema.py | head -n 3; echo "(empty above = none of the planned names exists at O)"
2026-10-17T06:23:58Z	Bash	SL1-B2	git grep -n "launch_log\|LAUNCH_LOG" -- src/quant scripts/run_e5.py scripts/run_e6.py ':(exclude,icase)*test*' | head -8; git ls-files -- reports | grep -i "launch\|teacher_arm\|arm_record\|qat" | hea
2026-10-17T06:24:36Z	Bash	SL1-B2	git ls-files -- configs/calibration reports/derived/mde_entry_20261009T034825Z.json reports/strata/train_strata_v1.json && /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df
2026-10-17T06:59:53Z	Bash	SL1-B1	grep -n "at_records_commit\|RecordsSource\|record=True" src/tman/*.py scripts/smoke_tman_schema.py scripts/smoke_tman_paths.py scripts/smoke_tman_complete.py scripts/smoke_tman_cuts.py scripts/smoke_t
2026-10-17T07:04:11Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && for s in smoke_tman_validate smoke_tman_unlock; do start=$(date +%s); PYTHONDONTWRITEBYTECODE=1 PYTHONPA
2026-10-17T07:11:09Z	Bash	SL1-B1	sed -i 's/out_folder(args\.out)/out_folder(args.out, args.root)/g' scripts/tday_manifest.py && grep -c "out_folder(args.out, args.root)" scripts/tday_manifest.py && awk 'length > 120 {print FILENAME":
2026-10-17T07:11:32Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && cat > $SCR/wrap_lines.py <<'EOF' """Re-wrap four over-long lines of the validate smoke (scratch helper; 
2026-10-17T07:12:29Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && for s in smoke_tman_derived smoke_tman_validate smoke_tman_unlock; do start=$(date +%s); PYTHONDONTWRITE
2026-10-17T07:21:04Z	Bash	SL1-B1	grep -n "FileNotFoundError\|OSError" scripts/smoke_tman_complete.py scripts/smoke_tman_schema.py scripts/smoke_tman_cuts.py scripts/smoke_tman_derived.py | head; grep -n '"decision_date": True, "value
2026-10-17T07:26:09Z	Bash	SL1-B1	grep -n "iterdir\|scandir\|listdir\|glob\|localtime\|fromtimestamp\|strftime\|time\.time\|datetime\.now\|tzset" src/tman/*.py scripts/tday_manifest.py | grep -v "^\s*#"
2026-10-17T07:27:13Z	Bash	SL1-B1	grep -n "write_bytes\|write_text\|open(\|mkdir\|unlink\|rename\|replace(\|shutil" src/tman/*.py scripts/tday_manifest.py | grep -v "\.replace(\"\|\.replace('\|str.replace\|text.replace" | head -30; gr
2026-10-17T07:30:28Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && for s in schema paths complete cuts parent derived pins validate unlock noscore determinism; do start=$(
2026-10-17T07:38:32Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && awk 'length > 120 {print FILENAME":"FNR": "length}' src/tman/*.py scripts/tday_manifest.py scripts/smoke
2026-10-17T07:41:29Z	Bash	SL1-B1	grep -n "def entered_day" -A 16 src/tman/entries.py; grep -n "_out(\|\.git(\|subprocess" src/tman/*.py scripts/tday_manifest.py | head -20
2026-10-17T07:41:46Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && for s in schema paths complete cuts parent derived pins validate unlock noscore determinism; do start=$(
2026-10-17T07:46:17Z	Bash	SL1-B1	grep -ohE "TmanRefused\(\s*\"[a-z0-9_]+\"" src/tman/*.py scripts/tday_manifest.py | sed -E 's/.*"([a-z0-9_]+)"/\1/' | sort | uniq -c | sort -k2 && echo ---- && grep -ohE "code=\"[a-z0-9_]+\"" src/tman
2026-10-17T07:48:04Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && for s in cuts parent pins complete; do echo "== $s"; grep -E "^\s+[a-z0-9_]+\s+: (PASS|FAIL)" $SCR/smoke
2026-10-17T07:52:05Z	Bash	SL1-B1	grep -n "__file__\|^REPO\|^ROOT\|^GIT_ROOT" scripts/preflight_arms.py | head; grep -n "__file__" src/tman/*.py scripts/tday_manifest.py
2026-10-17T07:53:02Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad && awk 'length > 120 {print FILENAME":"FNR": "length}' scripts/smoke_tman_*.py; for s in paths unlock nosco
2026-10-17T08:15:49Z	Bash	SL1-B2	git grep -n "evaluator_stage\|STAGES" -- src/tman scripts/tday_manifest.py scripts/smoke_tman_schema.py scripts/smoke_tman_validate.py scripts/smoke_tman_derived.py | head -60
2026-10-17T08:17:44Z	Bash	SL1-B1	wc -l src/tman/*.py && grep -n 'def read_jsonl\|^KD_ROWS\|^KD_RUN_META\|^E1_ROWS\|^QAT_ROWS\|^QAT_CHECKPOINT\|^QAT_STATE_FLAG_KEYS\|^def project\|^KD_\|^E1_\|^QAT_' src/tman/records.py
2026-10-17T08:17:53Z	Bash	SL1-B2	wc -l src/tman/completion.py src/tman/schedule.py src/tman/entries.py src/tman/manifest.py src/tman/validate.py scripts/tday_manifest.py && git grep -n "R\.\(KD_ROWS\|KD_RUN_META\|E1_ROWS\|QAT_ROWS\|Q
2026-10-17T08:17:59Z	Bash	SL1-B2	git grep -n "arm_record\|go_rule\|launch_provenance\|qat_selection\|clip_selection\|e7_arm\|QAT_SELECTION\|CLIP_SELECTION\|ARM_TEACHER\|GO_RULE\|E7_ARM\|LAUNCH_PROV" -- src/tman scripts/tday_manifest.
2026-10-26T06:43:59Z	Bash	SL1-B1	for f in docs/lane_specs/errata.md docs/lane_specs/part1.md docs/lane_specs/part2.md docs/lane_reports/q2-qat.md docs/EVALUATION_CONTRACT.md docs/IMPLEMENTATION_CONTRACT.md docs/STATISTICAL_ANALYSIS_C
2026-10-26T07:17:01Z	Bash	SL1-B1	grep -n 'cut_am19' src/training/sweep_select.py | head -20; ls /home/user/plantseg-thesis/reports/derived/ | grep -i -e decision -e selection
2026-10-26T07:33:01Z	Bash	SL1-B1	sed -n '33,40p' configs/distill.py; grep -n "^    \"[A-Za-z0-9]*\": {" configs/distill.py | head -10; grep -n "def validate_stage_config" -A 25 src/distill/*.py 2>/dev/null | grep -n "key" | head -5
2026-10-26T07:43:15Z	Grep	SL1-G1	/home/user/plantseg-thesis glob={src/tman/schema.py,src/tman/records.py,src/tman/completion.py,src/tman/schedule.py,src/tman/entries.py,src/tman/manifest.py,src/tman/validate.py,scripts/tday_manifest.
2026-10-26T07:43:52Z	Grep	SL1-G1	/home/user/plantseg-thesis glob={src/tman/schema.py,src/tman/records.py,src/tman/completion.py,src/tman/schedule.py,src/tman/entries.py,src/tman/manifest.py,src/tman/validate.py,scripts/tday_manifest.
2026-10-26T07:58:09Z	Grep	SL1-G1	/home/user/plantseg-thesis glob={src/tman/schema.py,src/tman/records.py,src/tman/completion.py,src/tman/schedule.py,src/tman/entries.py,src/tman/manifest.py,src/tman/validate.py,scripts/tday_manifest.
2026-10-26T08:01:47Z	Grep	SL1-G1	/home/user/plantseg-thesis glob={src/tman/schema.py,src/tman/records.py,src/tman/completion.py,src/tman/schedule.py,src/tman/entries.py,src/tman/manifest.py,src/tman/validate.py,scripts/tday_manifest.
2026-10-26T08:01:53Z	Grep	SL1-G1	/home/user/plantseg-thesis glob={scripts/smoke_tman_complete.py,scripts/smoke_tman_cuts.py,scripts/smoke_tman_validate.py}
2026-10-26T14:19:01Z	Bash	SL1-B1	for f in configs/calibration/ptq_calibration_seed42.json configs/calibration/ptq_calibration_seed43.json; do echo "== $f"; grep -n -v '^    "' "$f" | head -30; done
2026-10-26T17:19:34Z	Bash	SL1-B1	W=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/dl24/work/adv_entryfile_branches && PY=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b
2026-10-26T17:34:39Z	Bash	SL1-B1	W=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/dl24/work/adv_entryfile_branches/logs; for f in ctl_dl_rows_exist__validate ctl_entry_file_sha__validate no_r
2026-10-26T19:23:54Z	Bash	SL1-B1	for s in smoke_tman_cuts smoke_tman_complete smoke_tman_parent; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58
2026-10-26T19:24:22Z	Bash	SL1-B1	for s in smoke_tman_unlock smoke_tman_paths; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9d
2026-10-26T19:26:08Z	Bash	SL1-B1	for s in smoke_tman_paths smoke_tman_noscore smoke_tman_determinism; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-15
2026-10-26T19:27:35Z	Bash	SL1-B1	for s in smoke_tman_derived smoke_tman_pins smoke_tman_schema; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b
2026-10-26T19:52:09Z	Bash	SL1-B1	for s in smoke_tman_validate smoke_tman_unlock; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3
2026-10-26T19:56:19Z	Bash	SL1-B1	for s in smoke_tman_validate smoke_tman_paths smoke_tman_determinism smoke_tman_noscore smoke_tman_schema; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/
2026-10-26T20:09:44Z	Bash	SL1-B1	for s in smoke_tman_validate smoke_tman_unlock smoke_tman_noscore; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581
2026-10-27T00:13:53Z	Bash	SL1-B1	sed -n 340,372p scripts/qat_epoch_eval.py && grep -n "CLIP_SELECTION_FORMAT =" src/quant/*.py
2026-10-27T00:14:38Z	Bash	SL1-B1	grep -n "qat_selection/1\|qat_clip_selection/1\|QAT_EPOCH_DIR" src/tman/*.py | head
2026-10-27T00:19:06Z	Bash	SL1-B1	grep -n "iterdir\|listdir\|glob(\|scandir\|os.walk" src/tman/*.py scripts/tday_manifest.py | head
2026-10-27T00:21:57Z	Bash	SL1-B2	git grep -n "clip_selection_sha256\|ptq_script(" -- src/tman scripts/tday_manifest.py scripts/smoke_tman_derived.py scripts/smoke_tman_validate.py scripts/smoke_tman_mutations.py scripts/smoke_tman_pa
2026-10-27T01:09:32Z	Bash	SL1-B1	grep -n "commit_manifest(" scripts/smoke_tman_unlock.py scripts/smoke_tman_*.py | head -30
2026-10-27T01:19:02Z	Bash	SL1-B1	for s in smoke_tman_schema smoke_tman_derived smoke_tman_paths; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis timeout 900 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58
2026-10-27T01:52:37Z	Bash	SL1-B1	grep -n "teacher_ckpt_sha256" src/tman/*.py | head -20
2026-10-27T03:06:55Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_*.py
2026-10-27T03:07:04Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_*.py
2026-10-27T04:37:58Z	Bash	SL1-B1	python3 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/c6g2/patch_src.py /home/user/plantseg-thesis && python3 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e
2026-10-27T04:38:39Z	Bash	SL1-B1	until grep -q "^DONE" /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/fix/summary_c6g2dev.txt 2>/dev/null; do sleep 5; done; cat /tmp/claude-0/-home-user-plant
2026-10-27T05:05:15Z	Bash	SL1-B1	python3 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/c6g3/patch_src.py /home/user/plantseg-thesis && python3 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e
2026-10-27T05:06:12Z	Bash	SL1-B1	until grep -q "^DONE" /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/fix/summary_c6g3dev.txt 2>/dev/null; do sleep 5; done; cat /tmp/claude-0/-home-user-plant
2026-10-27T05:09:24Z	Bash	SL1-B1	python3 - <<'EOF' from pathlib import Path def sub(rel, old, new):     p = Path(rel)     s = p.read_text(encoding="utf-8")     assert s.count(old) == 1, (rel, old[:60])     p.write_text(s.replace(old,
2026-10-27T05:51:47Z	Bash	SL1-B1	python3 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/c6g4/patch_src.py /home/user/plantseg-thesis && python3 /tmp/claude-0/-home-user-plantseg-thesis/a18f2e
2026-10-27T05:52:16Z	Bash	SL1-B1	until grep -q "^DONE" /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/fix/summary_c6g4dev.txt 2>/dev/null; do sleep 5; done; cat /tmp/claude-0/-home-user-plant
2026-10-27T06:23:50Z	Bash	SL1-B1	F=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/fix; ls $F/smoke_tman_mutations_c6g3.log >/dev/null && for t in c6g3 c6g4; do grep -E '^  [A-Za-z0-9_]+ +: (P
2026-10-27T06:23:59Z	Bash	SL1-B1	cd /tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad/fix && for t in c6g3 c6g4; do grep -E '^  [A-Za-z0-9_]+ +: (PASS|FAIL)' smoke_tman_mutations_$t.log | awk '{
2026-10-27T08:09:12Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad; for f in $S/smoke_logs/baseline_pp/smoke_preflight_distill.log $S/step7/smoke_preflight_distill_final.log; d
2026-10-27T08:09:19Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/a18f2e9f-1581-58b6-8654-66b2b2a3e9df/scratchpad; for f in $S/smoke_logs/baseline_pp/smoke_preflight_distill.log $S/step7/smoke_preflight_distill_final.log; d
```

**Classes** (INFERRED from the lines; the guard runs in log mode until its flip to deny, DL-79):
- **SL1-B1:** shell commands that name lane files or scratch logs by literal path, in loops or with globs.
- **SL1-B2:** git searches or listings with folder pathspecs.
- **SL1-G1:** Grep tool calls on a folder.

None reached TEST data, `docs/reference` or a path holding "test".

**Slips** (MEASURED; recorded in `$SCR/LANE_STATE.md` as they happened):

- **STEP 4 (16–17 Oct), four hook refusals.** None was retried; each was replaced by a plain shell or exact-path form. Nothing ran.
  - Two L-PROT S: "inline interpreter code that runs git is not analysed" (a python heredoc whose text mentioned git).
  - One L-PROT S on an `ls` of a variable path.
  - One L-PROT N on a state note that named the protected file's path.
- **(1)** 2026-10-27T01:09:32Z, SL1-B1: `grep -n "commit_manifest(" scripts/smoke_tman_unlock.py scripts/smoke_tman_*.py`, a shell glob over `scripts/`. It printed lane smoke files only.
- **(2)** 2026-10-27T00:21:57Z, SL1-B2: a `git grep … -- src/tman scripts/tday_manifest.py …`, a folder path without the exact exclude form.
- **(3)** L-PROT S refused an `ls -la` of the SL-1 log: the hook reads any `ls` as a listing that could name the protected file. The GO's `tail -n` form was used instead. A later note that named the protected file's path in prose was refused by L-PROT N and rewritten without it.
- **(4)** 2026-10-27: a heredoc `cat > $SCR/c6f2_lane.py` ran with SCR unset in that shell, so the helper landed at `/c6f2_lane.py`, outside the repository and the scratchpad. Its edit of `src/tman/validate.py` was the intended one. The file was left in place (no rm). Since then every scratch path is written in full.
- **(5)** 2026-10-27: `grep -rn … $SCR/dl24/*.md` (grep -r) was refused by L-PROT S before it ran. The finding was read from `dl24/final_findings.json` by exact path.
- **(6)** 2026-10-27T03:06:55Z and 03:07:04Z, SL1-G1: two Grep tool calls on the `scripts/` folder with the glob `smoke_tman_*.py`, where the GO asks for named files and the test-exclusion glob. They printed lane smoke lines only.
- **(7)** 2026-10-27T00:13:53Z, 00:14:38Z, 00:19:06Z and 01:52:37Z, SL1-B1: `grep -n` over `src/quant/*.py` and `src/tman/*.py`, shell globs over a folder. They printed lane and src/quant lines only, and no test-named path.
- **(8)** 2026-10-27 ~05:20Z: an `ls` of the scratchpad's c6g4 folder. Its output was discarded, and it touched neither the repository, `scripts/` nor `.claude/hooks/`.
- **(9)** 2026-10-27: an `ls` of a scratch log by a variable path was refused by L-PROT S; nothing ran. It was redone with literal paths and no `ls`.
- **(10)** 2026-10-27: a state note quoting that L-PROT message, which spells out the protected file's path, was refused by L-PROT N; nothing ran. It was rewritten without the path.
- **(11)** 2026-10-27: a `grep -rn` on one named scratch file (`$SCR/dl24/brief.md`) was refused by L-PROT S; nothing ran. It was redone with `grep -n`.
- **(12)** 2026-10-27: three commands were refused by L-PROT S as inline interpreter code that runs git; nothing ran any time.
  - an inline python heredoc whose regex mentioned git;
  - a `python3` call chained with a note naming git commands;
  - an inline edit of the audit script whose text named git.

  The parser was moved into a script file and run alone, and the audit edits were made with the Edit tool.
- **(13)** Found by the call audit (§19): at STEP 1 (2026-10-16 09:02–09:39Z), named docs files were read with `wc -l`, `awk`, `grep -c`, `grep -o` or a plain `grep`, instead of `grep -n` then `sed -n`. Each was a named file, not a search of `docs/`.
- **(14)** Found by the call audit: at 2026-10-17T06:31Z, the decision log was read with `grep -c` and a plain `grep` in a loop.
- **(15)** Found by the call audit: `git ls-files` or `git status` over a folder without the exclude form, at 2026-10-16T09:44Z (the planned names, absent at O), 10-17T05:53Z, 10-17T06:23Z (filtered through `grep -i`), 10-17T06:24Z and 10-26T20:49Z (a line count only). None printed a test-named path (MEASURED from the outputs).
- **(16)** Found by the call audit: shell globs over a folder, the class that holds (1) and (7). They matched:
  - `src/tman/*.py` (the lane's own modules) and `scripts/smoke_tman_*.py` (the lane's smokes);
  - `src/quant/*.py` once (2026-10-27T00:13:53Z);
  - scratch logs.

  None matched a test-named file. `src/quant` holds 12 files with and without the test exclude (MEASURED by counts only; the second count carried no exclude form).

**Open items for SL-1** (at most five):
1. Whether a git search over named folders with the test exclude (`-- src scripts configs ':(exclude,icase)*test*'`, the form of ruling TMAN-STEP4-1's own STEP 7 check) should log as SL1-B2.
2. Whether a for-loop over literal lane files should log as SL1-B1.

## 12. Records proposed (the lane never writes them)

The plan's §7, as amended (DOCUMENTED; the amendments are named).

**The current plan line (B-9).** Insert before "TEST: one unlock …":

> "Before the freeze: the TEST-manifest lane's B-1 and B-3a (code tag `tman`; the manifest `reports/derived/tday_manifest.json`, its entry files and the lane register `reports/derived/tday_lanes.json`, sha256 in this row), L-CORR with L-AM16-SEV (the cache pin), and every lane that changes the evaluator file set, since the evaluator commit is frozen with the manifest. Before the unlock: B-3b (the binding), L-AM3, L-AM6 (entry point), L-AM8, L-RESIDUAL-SHA, L-AM17-DEFAULTS (carried by the TEST-manifest lane), and every lane in the register, IMPLEMENTED or entered as not required."

TB15 amends it: "B-3b, and every TEST-serving lane, IMPLEMENTED before D (AM-22 item 1(a)4); before the unlock only under item 1(f)". The lane merges once, after L-CORR's second merge.

**Decision-log rows to record:**
1. the lane's formats and paths (§6), now including `tday_push_list/1`, the catalogue file and the entry-file rule (iii);
2. the TEST-day execution context (viii; TB1);
3. the duplicate-free rule (ix), before the unlock;
4. the teacher fine-tune completion reading (vii; TB19 Q4: M12's "(final)" read as UTC);
5. the "since" reading (xiv) and entry dating (xv);
6. the scope table (§1.4 with Q3's answers; TB2–TB5), before the freeze, because it decides what is scored;
7. L-AM17-DEFAULTS folded into this lane;
8. a pin row for each merge (DL-37).

**Errata:**
- **E-new 1:** AM-19's Code and AM-18's lane list name the "TEST-manifest lane", and its files carry `tman` or `tday` (project rule 4; the branch is `lane/tman` by TB19 Q1).
- **E-new 2:** R-055's cell count waits for L-CORR (`configs/corruption_protocol.json:13-15` vs AM-16 item 6; TB6 fixes R-055 at 24).
- **E-new 3:** K2's gate computes no AM-19 item 3(b) arithmetic for non-arm runs; the lane's printer is the record (TB19 Q6).
- **New, by TMAN-B1-RULE-Q item 2:** the REPORT lines "deviation: evidence listing late" (R-010) and "deviation: since unknown before <date>" are not B-1's. Those deviations are kept in the decision-log records (§18).

**Ch4 sentences** (the plan's, unchanged):
- "The TEST-day manifest (sha256 …) was drafted on …, frozen at the end of … under Schedule T with N entries, and closed with M derived-artifact hashes appended before the unlock."
- "Candidates other than the selected and the two default runs are listed and were not evaluated on TEST."
- "The TEST images were hashed in the session before any model was scored (script … at …); D images were removed (a = TRAIN, b = VAL, c duplicate TEST; issue #11: 70/11/13)."
- "Every model was scored at one evaluator commit in one image (…)."

## 13. Phase B plans, as amended, with hours

This is a report only. TMAN-B1-RULE-Q item 4 assigns B-3a part 1 to session TMAN-B3A on `lane/tman-b3`; this session prepared no B-3a or B-2 work.

**B-3a (5–7 h, before the freeze):**
- the hasher with its smoke;
- the duplicate-free module with its smoke;
- the hasher pin wired into the builder.

B-1 already checks that pin at the unlock (`hasher_pin_mismatch`, `src/tman/validate.py:1348`) and reserves it in the body (`generator_pins.image_hasher`).

**B-3b (7–9 h, before the unlock):**
- the binding writer and verifier, with its smoke;
- the stats change, with the three expected smoke changes (a ruling) and a rerun of every stats smoke;
- the report.

B-1 reserves `tday_binding/1`, `references.mde_entry` and `artifacts` (TB11; `schema.py:425`). TB17 keeps the binding's post-check, and TB18 has the lane build the binding.

**B-2 (8–12 h, after AM-22):**
- AM-22's check, with its smoke;
- the freeze-entry and cutoff-entry printers;
- the report.

TB12's dates provider (`src/tman/schedule.py:166`) reads AM-22's cutoff and ceiling from the committed amendment text, with a fixture until then. B-2 makes a record's push time compulsory for the AM-22 freeze entry (decision (xv)).

**What must merge first** (the plan's §8, unchanged):
- Nothing for B-1.
- For a real freeze-mode manifest:
  - Q2 (E5/E6 and the pilot);
  - L-CKPT-GUARD's Q part (before E7 seed 42);
  - L-CORR with L-AM16-SEV (the cache pin);
  - L-AM3 (the trigger format);
  - L-RESIDUAL-SHA and the L-AM6 entry point;
  - B-3a.
- For the unlock: B-3b, L-AM8, and every lane in the register.

## 14. EXEMPT-1

ADDENDUM 1 A3 (DOCUMENTED): smoke_stats_report and smoke_stats_official_fullsize run with TMPDIR=$SCR.
- Runs A and B ran them, before A3, with the default temp dir. Their work dirs were removed: case Z1 passed (MEASURED).
- Run C ran both with TMPDIR=$SCR (§4), and so did STEP 7. No other smoke sets TMPDIR.
- smoke_tman_mutations gives each of its own worker processes a temp folder of its parent's (C6g5, DL-24 #108). That is a smoke's own subprocess environment, under the system temp dir, not the environment of record.

## 15. ADDENDUM 1: the counts it changed and the cases behind them

**What ADDENDUM 1 asked for** (DOCUMENTED):
- **A1**, IF§11 v2:
  - inventory rows gain `slot`, `entry_kind`, `role` (a list) and `training_run_id`, and are unique by (slot, precision);
  - a top-level `entries` list holds one item per catalogue slot;
  - labels for cut, diverged, aborted, not_run and not_made;
  - `telemetry` per attempt, hashed from bytes;
  - `dedup` and `freeze` are null until each exists;
  - `binding` is unbound;
  - the reader is `read_inventory`.
- **A2:** a derived E4/E7 family or per_seed entry carries `ptq_record`, read only through LQP's `read_record`, with the projection named there.

**The checks it added** (MEASURED from the final logs): the checks whose names carry `_A1`, `_A2` or `IF11_n`, per smoke.

| smoke | checks named for A1 | for A2 | for IF§11 |
|---|---|---|---|
| smoke_tman_schema | 4 | 3 | 2 |
| smoke_tman_paths | 0 | 1 | 0 |
| smoke_tman_parent | 2 | 0 | 0 |
| smoke_tman_derived | 1 | 6 | 0 |
| smoke_tman_validate | 1 | 2 | 17 |

The cases behind them, by name:

- smoke_tman_schema: `binding_reserves_mde_entry_TB11_and_artifacts_A1`, `binding_scopes_exclude_dedup_A1`, `not_made_kind_A1`, `non_finite_on_the_run_class_A1`, `ptq_role_of_record_A2`, `entry_kinds_well_formed_IF11_2`, `cut_place_is_group_position_IF11_2`, `ptq_record_projection_A2`, `ptq_record_reader_is_lqp_read_record_static_A2`
- smoke_tman_paths: `cli_post_freeze_ptq_append_without_its_record_refused_A2`
- smoke_tman_parent: `qat_run_without_model_its_sel_without_entry_A1`, `qat_sel_entered_under_a_run_without_model_refused_A1`
- smoke_tman_derived: `non_finite_run_requires_no_selection_A1`, `ptq_record_of_record_accepted_A2`, `ptq_record_other_stage_refused_A2`, `ptq_record_other_seed_refused_A2`, `ptq_record_other_role_d3_replicate_refused_A2`, `ptq_record_other_role_calibration_sensitivity_refused_A2`, `append_of_record_without_ptq_record_refused_A2`
- smoke_tman_validate: `ptq_record_of_another_seed_A2`, `post_freeze_append_of_record_without_its_ptq_record_A2`, `non_finite_run_without_selection_passes_A1`, `inventory_built_and_read_IF11_6`, `rows_carry_the_four_keys_IF11_1`, `entries_one_per_catalogue_slot_IF11_2`, `telemetry_per_attempt_by_bytes_IF11_3`, `cut_item_carries_its_label_IF11_2`, `dedup_null_binding_unbound_freeze_row_IF11_5`, `telemetry_copy_back_differs_IF11_3`, `row_without_slot_IF11_1`, `row_without_entry_kind_IF11_1`, `row_without_role_IF11_1`, `row_without_training_run_id_IF11_1`, `rows_unique_by_slot_and_precision_IF11_1`, `telemetry_item_without_sha256_IF11_3`, `dedup_malformed_IF11_5`, `freeze_manifest_sha256_differs_IF11_5`, `bound_binding_returns_its_artifacts_IF11_4`, `binding_artifact_scope_dedup_IF11_4`

## 16. Lane-chosen names (TL-14)

- **`cut_place`:** a catalogue slot's 1-based place of its cut group in the relative cut order. It is null for a never-cut slot or a slot outside every cut group (`src/tman/schema.py:458`, entry block `:464`). The KD arm registry's `cut_place` is cross-checked against it (smoke_tman_schema).
- **`manifest_sha256 := body_sha256`, plus `manifest_file_sha256`:**
  - the export's `manifest_sha256` is the canonical sha256 of the manifest's body, which is the freeze record's `body_sha256`;
  - `manifest_file_sha256` is the manifest file's own sha256 (`src/tman/manifest.py:978–983`).
- **The `e6kd` role:** the catalogue role, and the scope class, of the E6-KD model (`src/tman/schema.py:82`, `:158`). It requires `fq_off`, and forbids the corrupted, sev45 and efficiency scopes (TB2: E6-KD takes no corrupted scope).
- **`read_inventory`:** LREP's reader (T6), `src.tman.validate.read_inventory(path, *, expect_sha256=None) -> dict` at `src/tman/validate.py:1685`. It refuses `inventory_sha256_mismatch`, `inventory_invalid` or `binding_invalid`.

## 17. Score blindness and the telemetry hashes

`smoke_tman_noscore`'s mutation test sets every score field of the fixtures to random finite values, to "nan" or deletes it. The score fields are:
- the KD, E1 and QAT telemetry rows;
- the E1 and teacher logs' scores;
- the selection files;
- the ptq_record's parent score.

**What it compares under each mode** (MEASURED, 35/35):
- the draft and freeze verdicts;
- an unlock world's unlock and session verdicts (C6h2, DL-24 #59);
- the manifest and inventory bytes;
- the arithmetic;
- the command line's printed freeze record and post-freeze append.

**Why the outputs differ at all.** The lane cites files by sha256 and never copies a value from them. It cites:
- each run's end record (`completion.sha256`) and telemetry (`telemetry_sha256`);
- each QAT selection record, which cites its run's telemetry sha256;
- each attempt's telemetry in the export;
- the manifest itself, by its file and body sha256.

When a score changes, the file's bytes change, so its sha256 changes. So does every hash that cites it: the QAT selection records, the manifest's file and body sha256, and the freeze row. The comparison therefore maps each new sha256 back to the old, and maps the commit a printer names, before comparing bytes. Verdicts are compared directly. Anything left over would be a score read.

**What it does not compare:** the sentinel case raises on any reader that projects a score key, and the static scan finds no score-key literal in the lane's code outside docstrings. The one value key kept is DECISION_RECORD's grid value.

**Mutations that read a score are killed** (C6h2):
- `unlock_gate_reading_a_score` (an unlock gate reading E1's raw telemetry);
- `append_printer_reading_a_score` (the append command reading a KD telemetry).

## 18. Open items, questions and rulings

1. **OPEN ITEM, the pin's cells** (carried by TMAN-B1-RULE-Q item 3). The pin check follows IF§3: presence of its keys only (TMAN re-derives no field), with pin_sha256 recomputed. Cells are checked against `src.corruption_roles.CELL_IDS`. The attribute name and the pin's `cells` key are TMAN's reading of IF§3/§6, to confirm at L-CORR's merge. L-CORR-B1's `a508359` is not in this clone, and no fetch was made.
2. **OPEN ITEM, TB11's push evidence** (carried by TMAN-B1-RULE-Q item 3). A push the clone lacks and the remote has not served carries nothing. AM-19a reading 10 counts it for the lambda day. The lane's reading is in `push_evidence_error` (`src/tman/validate.py:1425`); see C6f3.
3. **#103 is CLOSED by ruling TMAN-B1-RULE-Q item 2** (27 Oct). Both REPORT lines are dropped from B-1, and no proxy is implemented:
   - "deviation: evidence listing late" (R-010);
   - "deviation: since unknown before <date>".

   No record the lane reads defines a run's durable evidence folder (the decision log mentions one only in DL-34). The orchestrator keeps those deviations in the decision-log records. The lane's one REPORT line is "deviation: entry later than its day" (decision (xv); `src/tman/validate.py:1589–1593`).

   **Candidate readings, not adopted:**
   - (a) "evidence listing late": a run's listing time is the earliest commit, in the records' history, of a cited checkpoint source file that holds the run's sha256, against the run's end plus 24 hours (AM-19 Definitions, "Complete").
   - (b) "since unknown before <date>": printed for a never-launched cut whose `since` record is dated after the first day the stated reason held, the date being that record's.
4. **The push count** (DOCUMENTED). STATUS-ASK-1 amends the GO's "two pushes of lane/tman only", and TMAN-B1-RULE-Q item 1 makes it a push after every commit. STEP 9's push stands (§2).

## 19. The call audit (the DL-113 pattern)

`$SCR/c7/audit_session.py` reads this session's transcript (MEASURED at 2026-10-27T08:53:42Z). It applies the checks the workflow audit used to every tool call, adjusted for a writing session:
- Write and Edit only on the GO's files or the scratchpad;
- git writes only in the GO's forms.

**Calls:** 2,659 (MEASURED): Bash 1,853, Edit 437, Read 136, Write 110, TaskUpdate 51, Grep 23, TaskCreate 23, ToolSearch 11, Monitor 6, Glob 3, Workflow 3, SendUserFile 1, Skill 1, TaskStop 1.

**Root or docs/ searches: 0** (the acceptance item).
- No Glob or Grep call is rooted at the repository or at `docs/`.
- No shell search names the root or `docs/` as its path.
- `docs/` files were read only by exact path. §11's slips (13) and (14) name the reads that used another command than `grep -n` and `sed -n`.

**The audit's 74 other flags, each classified** (MEASURED from `$SCR/c7/audit_final.txt`):

**Phase A (2026-10-14, before this GO), 10 flags:**
- six `git show FETCH_HEAD:docs/lane_reports/q2-qat.md`, reading one named docs file;
- three `git grep` over named folders without the exclude form (`src/stats`, `src/teacher`, `src/quant` with `src/eval`);
- the branch name `lane/test-manifest` in the plan's question Q1.

**Sanctioned, 16 flags:** git searches and listings over named folders that carry the test exclude, such as `git grep -n … -- src scripts configs ':(exclude,icase)*test*'`. This is the form of ruling TMAN-STEP4-1's own STEP 7 check. None names `docs/` or the root.

**Shell globs over a folder, 22 flags:** slip (16), which includes (1) and (7).

**Slips already listed in §11, 21 flags:**
- (13): nine flags, STEP 1's docs reads;
- (14): two flags;
- (15): five flags;
- (2): one flag;
- (5) and (11): one flag each, the two `grep -rn`, both refused by L-PROT before they ran;
- (6): two flags, the two Grep calls on `scripts/`.

**Not slips, 5 flags:**
- `lane/test-manifest`, a ref name in STEP 1's `git ls-remote`, which the GO asked for;
- `refuse_test_path/`, inside a quoted `sed` address on a named file;
- `.tests` twice, inside a `git grep` pattern over code that carries the test exclude (`src/stats/tests.py` was never opened);
- one Edit of the workflow's own script file under the session's `workflows/` folder, made for its resume, which the Workflow tool supports. It is not a repository file.

The groups add up: 10 + 16 + 22 + 21 + 5 = 74.

**After the audit** (INFERRED; the same prescribed forms as every earlier commit and push): the remaining calls are the report's write, the verbatim pre-commit checks, the commit and STEP 9's push and `ls-remote`.

## 20. STEP 9

After this report's commit: `git push origin refs/heads/lane/tman:refs/heads/lane/tman`, then `git ls-remote --heads origin lane/tman`, then STOP.

## B-2

Session TMAN-B2, a cloud lane under DL-37, on branch `lane/tman`. Written 2026-10-28 as step B6 of ADDENDUM B-2 (ORCH-3). It follows the rulings B2R1–B2R13, the audit TMANB2-CHECK-1 as its AUDIT RULINGS direct (TMAN-B2-RULE-2), and the orchestrator's ruling TMAN-B2-RULE-3, which amends audit m9. Everything above this heading is MT's file, byte for byte (B-2.16).

- **MT** (the merge-in and B-2's base): `8f85c55a53aaf5a5e7fe5cc9f1563305dbe05d7f`, "Merge master (524116b) into lane/tman".
- **Last code commit:** `5fb27593840f62f6843635dafffd29ad7f2789b6` (B2-F2). This section is the report commit and changes no code.
- **Labels** as in §1: MEASURED, DOCUMENTED, REPOSITORY-PROVEN, INFERRED.
- Every file:line below was re-read at `5fb2759` unless a commit is named. "PA :n" is docs/PREREGISTRATION_AMENDMENTS.md at 524116b, which MT holds.

### B-2.1 What B-2 delivers

DOCUMENTED (ADDENDUM B-2) and REPOSITORY-PROVEN (B-2.2):
- **AM-22 item 1(a)'s check.** `scripts/tday_manifest.py check-freeze --date X --runs DIR` (:579) runs `src/tman/am22.py` (new) and writes nothing. It gives:
  - the registered runs and how each is settled;
  - the five conditions, each MET with its records or MISSING with what is missing;
  - D, the first date from 2026-10-17 at whose end all five hold;
  - REPORT lines for boundary-day commits (audit M2) and a late trigger.
- **The freeze entry of AM-22 items 1(b) and 1(f).** Kind `freeze` gains `d_date` and `conditions` (src/tman/entries.py:67–69).
  - `entry freeze --d-date D` recomputes its fields (audit m9 as TMAN-B2-RULE-3 amends it).
  - `freeze-body` writes the body to freeze.
  - `entry cut --d-date D` prints the ARM latency measurement's freeze cut.
- **AM-22's freeze provider** goes in front of B-1's at every `freeze_instant` call (B2R6). Four unlock gates follow `run_added_after_freeze` (B2R7): 84 gates become 88, and 89 codes become 94.
- **AM-22 item 2's launch cutoff** is cut rule `cutoff` (B2R2). Its date is recomputed from the records, never taken from the entry. `entry cut --rule cutoff` prints one.
- **L-AM3's trigger reader** is wired in (B2R13). TB6's live case is added (B2R10).
- **Smokes:**
  - smoke_tman_am22 (new, 191 checks);
  - ten B-1 smokes edited to keep their cases true under B-2 (B-2.4);
  - the mutation smoke grows from 374 mutations to 563, every one KILLED.

**What it never does** (MEASURED in this session; B2R12):
- run on real data: no dataset, checkpoint or run directory is read, and every case is synthetic;
- read TEST data, the issue #11 CSV, a dataset archive or a path holding "test";
- train, use a GPU or download anything (B0's venv install aside).

### B-2.2 Base, commits, pushes and the diff

**B0** (MEASURED):
- **State:** HEAD 524116b on `refs/heads/claude/compassionate-davinci-x73bei` (recorded); the scoped status was empty; the venv was absent.
- **Fetch:** the clone was shallow, so `git fetch --unshallow origin` ran first, then the ADDENDUM's fetch of lane/tman and master.
- **MT verified:** its sha, parents, tree `1219b3f0…`, OT2 in master, and the 23 A lines.
  - One deviation: L-PROT G refused the named `git cat-file commit 8f85c55…` (a bare object id).
  - The message was read instead with `git show -s --format=%B 8f85c55… -- <the exclude form>`: "Merge master (524116b) into lane/tman".
- **The move** (the new-session path): `git branch lane/tman 8f85c55…`, `git symbolic-ref HEAD refs/heads/lane/tman`, then `git checkout --no-overlay HEAD -- . ':(exclude)docs/reference/reference.pdf'`.
  - Post-checks: HEAD is MT, HEAD names refs/heads/lane/tman, and the scoped status is empty.
  - update-ref was not used (audit n5).

**Commits** (REPOSITORY-PROVEN; no commit carries a trailer):

| Step | Commit | Committed (UTC, 2026-10-28) | What |
|---|---|---|---|
| B2-0 | `c75082dbe6a10ed7eed13095f7286d41d961e63d` | 06:06 | the rebaseline at MT (B2R9; audit M1) |
| B2-1 | `0ce0067ba67ca36fd787b6aa0079b9e461fb2ed0` | 06:36 | cut rule "cutoff" (B2R1, B2R2, B2R8) |
| B2-2 | `68835201554489106e01192ec9b1f0befe371feb` | 07:14 | registered runs, settled ways, conditions 1–4, check-freeze (B2R3, B2R4, B2R11, B2R13) |
| B2-3 | `1a0b14644f24477e3aeb46ffcc3ad71daf3e477b` | 08:38 | condition 5, D, the provider, the freeze entry and its unlock gates (B2R4–B2R7; TMAN-B2-RULE-3) |
| B2-4 | `431524632b7eaf3e9618597354c1f112876e8d04` | 09:23 | 73 mutations (B2R7; M1–M4; m1–m10; TMAN-B2-RULE-3) |
| B2-5 | `5f45c8591b6396ea72781dcc1bb66e7935d587c1` | 09:24 | TB6's live case (B2R10; M5) |
| B2-F1 | `d676a975204ce24614c6f331efbb9313f89f91d7` | 13:44 | DL-24 fixes to AM-22 item 1 |
| B2-F2 | `5fb27593840f62f6843635dafffd29ad7f2789b6` | 14:20 | DL-24 fixes to item 2's cutoff |
| B6 | this report | | |

**Pushes** (MEASURED): one after each commit, with `git push origin refs/heads/lane/tman:refs/heads/lane/tman`.
- Each was a fast-forward, from `8f85c55..c75082d` to `d676a97..5fb2759`.
- `git ls-remote --heads origin lane/tman` printed each new tip.
- B7 is the ninth push (B-2.16).

**The scoped diff** `git diff --no-renames --stat 8f85c55… HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'` at 5fb2759 (MEASURED):

```
 scripts/smoke_tman_am22.py        | 1808 +++++
 scripts/smoke_tman_complete.py    |   24 +
 scripts/smoke_tman_cuts.py        |  389 ++++-
 scripts/smoke_tman_determinism.py |    4 +-
 scripts/smoke_tman_mutations.py   |  687 ++++-
 scripts/smoke_tman_noscore.py     |   21 +-
 scripts/smoke_tman_paths.py       |   20 +-
 scripts/smoke_tman_pins.py        |   37 +-
 scripts/smoke_tman_schema.py      |   12 +-
 scripts/smoke_tman_unlock.py      |   16 +-
 scripts/smoke_tman_validate.py    |   50 +-
 scripts/tday_manifest.py          |  230 ++-
 src/tman/am22.py                  | 1125 +++++
 src/tman/completion.py            |   90 +
 src/tman/entries.py               |  259 ++-
 src/tman/manifest.py              |    9 +-
 src/tman/records.py               |   84 +-
 src/tman/schedule.py              |   68 +-
 src/tman/schema.py                |    2 +-
 src/tman/validate.py              |  160 +-
 20 files changed, 4972 insertions(+), 123 deletions(-)
```

(The bars are shortened here; the counts are git's.)

**ACCEPTANCE's lists** (MEASURED at 5fb2759 and compared with cmp):
- the scoped `--name-status` and the `-- src scripts configs docs/lane_reports ':(exclude)docs/reference'` form print the same 20 lines;
- they are A src/tman/am22.py, A scripts/smoke_tman_am22.py and M for 18 other files, all in SCOPE;
- configs/tday_catalogue.json, smoke_tman_parent.py and smoke_tman_derived.py are unchanged;
- the report commit adds `M docs/lane_reports/tman.md`.

### B-2.3 The environment and the hooks

**The venv** `$SCR/tmanvenv`, built at B0.5. MEASURED again at B6 with `pip freeze --all` against requirements.lock, read by exact path:
- Python 3.11.17, from /usr/bin/python3.11;
- `python -m pip check`: "No broken requirements found.";
- the torch line: `2.1.0+cpu 0.16.0+cpu`;
- the hard pins, each exact: numpy 1.26.4, scipy 1.11.4, statsmodels 0.14.6, patsy 1.0.2, pandas 3.0.3, pillow 12.3.0, scikit-image 0.23.2, mmcv 2.1.0, mmengine 0.10.7, mmsegmentation 1.2.2, ftfy 6.3.0, regex 2026.5.9.

**Against the lock's 76 pins** (MATCH is a different count from the B-1 lock's, which the ADDENDUM reports rather than STOPs on):
- **MATCH:** 48.
- **DIFF:** 2, the recorded difference:
  - torch 2.1.0+cu121 → 2.1.0+cpu;
  - torchvision 0.16.0+cu121 → 0.16.0+cpu.
- **ABSENT:** 26, each one line here: aliyun-python-sdk-core; aliyun-python-sdk-kms; cffi; click; colorama; crcmod; cryptography; fvcore; iopath; jmespath; markdown; model-index; opendatalab; openmim; openxlab; ordered-set; oss2; portalocker; pycparser; pycryptodome; pytz; pywin32; tabulate; tqdm; tzdata; yacs.
- **EXTRA:** ftfy 6.3.0 (DL-105), pip 24.0, setuptools 79.0.1.

mmcv was built with `--no-build-isolation`, in about 25 min (MEASURED at B0).

**The hooks** (MEASURED):
- **L-PROT is live** (`.claude/hooks/protect_reference_pdf.sh`, which runs the `.py`). It refused three commands in this session, each before it ran (B-2.13).
- **The SL-1 guard is installed in log mode** (`.claude/hooks/sl1_guard.sh` → `sl1_guard.py`). It logged the lines in B-2.12 and denied nothing.

### B-2.4 Smokes: the rebaseline at MT (B1) and the final runs (B5)

Each smoke ran alone (MEASURED). B5 ran at 5fb2759 with nothing else running, each smoke in its own new TMPDIR under $SCR, in two batches: first the thirteen less the mutation smoke, then the mutation smoke alone (14:22–15:22 UTC).

| Smoke | B1 at MT | B5 at 5fb2759 | B5 seconds |
|---|---|---|---|
| smoke_tman_schema | RESULT: FAIL (166/167) | RESULT: PASS (167/167) | 2 |
| smoke_tman_paths | RESULT: PASS (89/89) | RESULT: PASS (89/89) | 13 |
| smoke_tman_complete | RESULT: FAIL (89/90) | RESULT: PASS (93/93) | 3 |
| smoke_tman_cuts | RESULT: PASS (129/129) | RESULT: PASS (204/204) | 4 |
| smoke_tman_parent | RESULT: PASS (33/33) | RESULT: PASS (33/33) | 2 |
| smoke_tman_derived | RESULT: PASS (101/101) | RESULT: PASS (101/101) | 2 |
| smoke_tman_pins | RESULT: PASS (49/49) | RESULT: PASS (53/53) | 2 |
| smoke_tman_validate | RESULT: PASS (369/369) | RESULT: PASS (369/369) | 363 |
| smoke_tman_unlock | RESULT: PASS (86/86) | RESULT: PASS (86/86) | 108 |
| smoke_tman_noscore | RESULT: PASS (35/35) | RESULT: PASS (36/36) | 59 |
| smoke_tman_determinism | RESULT: PASS (18/18) | RESULT: PASS (18/18) | 17 |
| smoke_tman_mutations | RESULT: FAIL (428/429) | RESULT: PASS (647/647) | 2,612 |
| smoke_frozen_blobs | RESULT: FROZEN BLOBS OK (10/10) | RESULT: FROZEN BLOBS OK (10/10) | 1 |
| smoke_tman_am22 (new) | — | RESULT: PASS (191/191) | 376 |

**B5's mutations:** 563, every one KILLED (MEASURED: 563 KILLED lines and no other outcome).

**B1's FAILs** (B2R9). Each cause is a master change since 2b3b556, fixed in B2-0 `c75082d`:
1. **smoke_tman_schema,** `evaluator_stages_from_artifacts_STAGES_or_pending_code_TB8`: FAIL "pending []".
   - Cause: src/eval/artifacts.py :58–59 at 524116b. STAGES gained E6-E2, E7-E2, E6-R1, E6-R2 and E6-KD.
   - Fix: the check expects no pending stage, with the five in STAGES.
2. **smoke_tman_complete,** `case_committed_cross_checks_raised`: FAIL "TmanRefused: AM-22's text gives no item 2(a) cutoff date and ceiling".
   - Cause: AM-22 is committed, and PA :2375–2376 wraps "in no" / "case later than 2026-10-29". B-1's AM22_CEILING_RE needed single spaces.
   - Fix: both item 2(a) regexes are whitespace-tolerant (src/tman/schedule.py:39–40; audit M1).
3. **smoke_tman_mutations,** `baseline_schema_case_catalogue_cross_checks_passes_alone`: FAIL. This is cause 1 again, through the baseline of the case that kills `kd_slot_at_a_seed_the_launcher_lacks`.

After B2-0: schema 167/167 and complete 90/90, and that mutation and its baseline passed alone (`--only`).

**Every changed count, with its cases** (counts MEASURED; cases REPOSITORY-PROVEN in their commits):
- **smoke_tman_schema** 167 → 167, with changed checks:
  - TB8 expects pending [] (B2-0), and checks `pend == []` with the five stages in STAGES, without the code-gate conjunct that all([]) made vacuous (B2-F2; DL-24 R3-17);
  - `cut_rules_b1` expects "cutoff" (B2-1).
- **smoke_tman_complete** 90 → 93 (B2-F2), adding:
  - `teacher_complete_at_its_earliest_log_s_final_line_R3_15`;
  - `teacher_complete_with_a_later_malformed_stamp_reads_complete_N2`;
  - `teacher_last_row_listing_matches_names_before_any_call_B2`.
- **smoke_tman_cuts** 129 → 185 → 204:
  - B2-1 adds 2(a) with and without a moving record, the ceiling, 2(b), 2(c), 2(d), the late-entry REPORT, m1, m2, m3, m7, B2R1, B2R2, B2R8, and DL-137, DL-138 and DL-168 on synthetic records.
  - B2-F2 adds R1-F3, R1-F7, R4, R4-2, R4-4, R3-07, R3-08, R3-13 and R3-15.
- **smoke_tman_pins** 49 → 53: case_live_cells (B2-5; B2R10; audit M5).
- **smoke_tman_noscore** 35 → 36 (B2-3): `conditions_is_the_freeze_entry_s_field_and_no_reader_keeps_it_B2R5`.
  - B2-2 makes its static scan read ten modules and renames the check `static_scan_reads_the_lane_s_ten_modules_B2R11`.
  - B2-F1 narrows the `conditions` exemption to entries.py and am22.py (FREEZE_FIELD_SITES, :47), and its item 1(f) printer case takes `--runs` and `--now`.
- **smoke_tman_paths** 89 → 89:
  - B2-2: LANE_FILES 23 → 25, created names 3,317 → 3,319, lane modules 9 → 10, and the trigger reader restored after its O8 cases;
  - B2-F2: the listing check expects teacher_last_row's listing of the teacher's work dir.
- **smoke_tman_validate** 369 → 369:
  - B2-2: the trigger reader restored; O8's cases patch it to None;
  - B2-3: the gate count 84 → 88, and B-1's freeze fixtures gain source "AM-22 item 1(f)" with d_date and conditions null;
  - B2-F1: the item 1(f) printer case takes `--runs` and `--now`.
- **smoke_tman_unlock** 86 → 86:
  - B2-2: the trigger reader restored;
  - B2-3: the freeze fixtures, and its coverage leaves the four B2R7 gates to smoke_tman_am22 (AM22_GATES), whose case_coverage checks their order and cases.
- **smoke_tman_determinism** 18 → 18: the trigger reader restored (B2-2).
- **smoke_tman_parent** (33) and **smoke_tman_derived** (101): their files are unchanged.
- **smoke_tman_mutations** 429 → 523 (B2-4, 447 mutations) → 609 (B2-F1, 527) → 647 (B2-F2, 563). Each count is the mutations, plus one baseline per killing case and the static checks.
- **smoke_tman_am22:** 52 (B2-2) → 102 (B2-3) → 103 (B2-4) → 188 (B2-F1) → 191 (B2-F2). B2-F2 adds:
  - `validator_cut_context_reads_runs_pilot_and_am22_dates_R3_15`;
  - `printer_cutoff_reads_the_pilot_s_winner_record_from_the_manifest_R3_15`;
  - `printer_cutoff_without_the_manifest_s_pilot_record_refused_R3_15`.

  The plan targeted about 60 checks (plan :652). The DL-24 branches tripled it (INFERRED).

### B-2.5 Rulings B2R1–B2R13, with file:line

- **B2R1, the dates.**
  - **Constants:** AM22_PUSHED_DAY "2026-10-17" and AM22_COMMIT ce29549 (src/tman/schedule.py:45–46).
  - **The dates:** am22_dates (:215–225) gives (2026-10-20, 2026-10-29, shift 0) on the committed text with M1's regexes (:39–40).
  - **The refusals:** am22_dates_of_record (:228–240) refuses schedule_invalid when ce29549 is not an ancestor of the records' commit (:232–235), or when the text there holds no AM-22 section (:238–239).
  - **Cases** in scripts/smoke_tman_cuts.py:
    - :781, the committed text (M1);
    - :785, the constant tied to DL-137's and DL-168's committed rows, read by exact path;
    - :792, this checkout;
    - :800 and :808, the two refusals.
- **B2R2, cutoff cuts in B-1's cut format.**
  - **The rule:** CUT_RULES gains "cutoff" (src/tman/schema.py:130). tday_entry/1 gains no key.
  - **The date:** rule_day's cutoff branch (src/tman/entries.py:481–482) recomputes the date through cutoff_rule (:398–418):
    - 2(a): slot_cutoff (:352–370), with schedule.cutoff_day (:243–252);
    - 2(b): slot_repeat (:373–395), with repeat_cutoff_day (:255–258).
  - **The readers:**
    - moving_reader (:274–333), last_row_reader (:336–344) and logged_reader (:347–349);
    - completion.last_row (src/tman/completion.py:555–570), teacher_last_row (:346–364) and last_stamp (:308–320).
  - **chk_date** (:552–573) checks the date, the rule_record that set it (:563–566), and an entry no earlier than the day after the cutoff date (:570–572; ENTERED_DAY_OFFSET, schedule.py:42).
  - **The other checks that admit the rule:** chk_reason (:533–544), chk_arm_running (:647–648), chk_stopped (:712–713), chk_arm_partial (:664–665) and chk_decision_date (:685–689).
  - **The printer:** `entry cut --rule cutoff` (scripts/tday_manifest.py:365–367), check_cut_entry (:192–204) and printer_cut_context (:417–446).
  - **Fixtures:** smoke_tman_cuts.py:535–577.
- **B2R3, registered and settled** (src/tman/am22.py).
  - **The records at C_X:** View (:136), commit_at_end (:72–81), records.Records.at_commit (src/tman/records.py:283) and CommitSource (:288).
  - **The registered runs** (:345–370):
    - the catalogue's run slots, less E6-KD unless triggered;
    - less an E3-X whose go-rule record reads no_go or not_evaluable, together with its E6-X (:59–60);
    - less an E6-X whose condition entry reads no.
  - **AM-7's full consequence** is a STOP to ask (am7_fired, :372).
  - **Settled:**
    - settled (:387–427) uses completion's settled ways (completion.py:49–52; settled_ways :74);
    - STOP_TEXT (:63) and unsettled_why (:429) give the reasons.
  - **"Entered by the end of X"** is a row's commit at C_X (line_commit :162, sha_commit :175, file_commit :185). Each RECORD line prints the commit and its committer time.
  - **Boundary-day REPORTs:** boundary_reports (:1066–1077), with M2's confirmed pushes (B-2.10).
- **B2R4, the check.**
  - **The conditions:**
    - condition_1 (:446);
    - condition_2 (:466–503): the late-trigger REPORT at :500, n4's wording at :473, TRIGGER_DUE_DAYS at :57;
    - condition_3 (:505);
    - condition_4 (:524–571): ARM_LANE at :56, not_required_error at :574;
    - condition_5 (:602–630).
  - **The scan:** evaluate (:648), and first_d (:672–681), which reads the dates from AM22_IN_FORCE (:53) to X (m4).
  - **Refusals:** date_error (:1031–1045) for m6 and item 1(f).
  - **Printing:** check_freeze (:1048), check_lines (:1080) and lines (:1107).
  - **The command** `check-freeze` (scripts/tday_manifest.py:579–596; parser :679–684). It exits 0 with D, 3 when MISSING, 2 when refused, 4 on an error.
- **B2R5, the freeze entry.**
  - **Fields:** d_date and conditions, registered additively (src/tman/entries.py:67–69). B-1's kind (:62) is unchanged.
  - **Building it:** freeze_fields (src/tman/am22.py:983–1028) and freeze_conditions (:774–784).
  - **Checking it:** freeze_entry_error (:910–935) and freeze_record_error (:938–956).
  - **The command** `entry freeze --d-date D` (scripts/tday_manifest.py:398–414; check_freeze_record :308–344).
  - **Push evidence:** the entry's own push_evidence stays null. The unlock entry cites the evidence (audit M4; src/tman/validate.py:1321, :1514).
- **B2R6, the AM-22 provider.**
  - **The provider:** am22_provider (src/tman/schedule.py:167–177; None under Schedule R, :173) and am22_candidate (:180–183).
  - **Choosing one:** am22.providers (src/tman/am22.py:737–771) gives the candidate, the record, or the record the decision log enters. It refuses dates out of bounds (:748), two item 1(b) records (:755) and m10's rows (:767).
  - **In the validator:** validate.am22_providers (:578–587; the unlock and the session read `--freeze-record` only) and g_schedule_invalid (:590–599).
  - **Call sites (nine):**
    - validate.py:594 and :598;
    - manifest.py:858, through build_manifest's parameter (:843);
    - tday_manifest.py:137 (build), :324, :386, :489 and :515.
  - **The static case:** smoke_tman_am22.py's case_providers (:888) finds every call (:916). Its `given()` (:899–907) refuses an absent or empty `providers` (DL-24 R2-17).
- **B2R7, the new codes.**
  - The four codes are in B-2.8; their GATES rows are validate.py:1649–1650.
  - smoke_tman_mutations' SMOKES gains am22 (:41–42), REFUSAL_EXTRA gains am22_date_not_ended (:1938–1940), and the code count is 94 (:2096).
- **B2R8, cards.**
  - entries.no_card_report (:440–450) uses evidence_reader (:421–437).
  - The validator prints it in report_lines (validate.py:1695–1705), and the printer at tday_manifest.py:202–204.
  - It is never a refusal.
- **B2R9, the rebaseline:** B-2.4.
- **B2R10, TB6's live case:** smoke_tman_pins.py's case_live_cells (:130), against manifest.registered_cells (src/tman/manifest.py:559–568) and PIN_CELLS_KEY (:542).
- **B2R11, score blindness.**
  - LANE_CODE_FILES gains src/tman/am22.py (manifest.py:684–686).
  - smoke_tman_noscore's static scan reads the ten modules (:251–259). The literal `conditions` is exempt only at the freeze kind's sites, and no records allowlist keeps it (:254–264).
  - So check-freeze, the cutoff cut and the freeze entry read no score (ACCEPTANCE).
- **B2R12, where the real check runs:** nowhere in this session. Every smoke builds synthetic fixtures under its TMPDIR. The real check-freeze, cutoff entries and freeze entry are a local block's.
- **B2R13, the trigger reader.**
  - TRIGGER_READER is l_am3_trigger (src/tman/records.py:394). It wraps L-AM3's read_trigger_record from src.quant.e6kd_trigger (TRIGGER_MODULE :372), imported lazily (:377–391).
  - `trigger_not_triggered` gives triggered false (:388). trigger_reader(records) binds it to the records' checkout (:397).
  - Cases: smoke_tman_am22's case_static (`trigger_reader_is_l_am3_s_B2R13`) and case_trigger_reader.

### B-2.6 AM-22's map

Each clause of AM-22, with its PA line and its code at 5fb2759. am22.py is src/tman/am22.py. A records matter is a clause the lane prints or the orchestrator records, not code.

**The header, and AM-19's timestamp:**

| Clause | PA | Code | Note |
|---|---|---|---|
| Dates move when the commit reached the remote after 2026-10-17 | :2160–2164 | schedule.am22_dates (:215–225), AM22_PUSH_BASE (:37), AM22_PUSHED_DAY (:45) | shift 0 (B2R1) |
| Under Schedule R nothing applies | :2160–2161 | schedule.py:173 (am22_provider gives None) | R1-N1's reading (B-2.11) |
| A record's timestamp is its commit's push (AM-19 Definitions) | :1374 | am22.commit_at_end (:72–81, with confirmed pushes), Rec.date (:103–108), boundary_reports (:1066–1077) | audit M2 |

**Definitions:**

| Clause | PA | Code | Note |
|---|---|---|---|
| Registered runs: the never-cut runs, and the cuttable runs in the cut orders | :2266–2270 | View.registered (am22.py:345–370) | the catalogue's run slots (n3) |
| Excluded: K8; B-avg with E3-avg | :2273–2274 | not in the catalogue | — |
| Excluded: E3-X on no-go or not evaluable; the SegNeXt-L KD run on AM-17 item 11(b)'s no-go | :2275–2276 | GO_RULE_ARMS and EXCLUDING_VERDICTS (am22.py:59–60), View.go_rule (:312) | the L arm's go rule has no record (DL-129), and the arm is cut (DL-168); R1-N5 |
| Excluded: E6-KD unless AM-3 triggers it | :2277 | View.trigger (:288) | |
| Derived artifacts and the ARM latency measurement are not runs | :2279–2281 | registered reads class "run" only | |
| Settled: complete or superseded by the end of the date; a cuttable run also diverged, aborted, cut by a rule other than the freeze, or not run | :2282–2288 | View.settled (:387–427); completion.py:49–52, :74 | n7's reading |
| A never-cut run settles in no other way; AM-7's full consequence | :2290–2292 | NEVER_CUT_SETTLED_WAYS (completion.py:51); am7_fired (am22.py:372) | a STOP to ask |
| A run whose record is a STOP that a new amendment settles | :2294–2303 | STOP_TEXT (am22.py:63), settled | the "refusal that no later input can lift" is not coded (R1-N5) |
| AM-21 item 2(b): a complete run stays settled; its halts hold conditions 1 and 2 | :2305–2307 | condition_2 (:466; :473) | n4 |
| Not settled: launched and not complete, not launched, waiting for a repeat; AM-19 item 1(d) only under 1(f) | :2309–2311 | settled; providers' m10 conflict (:767) | |

**Item 1:**

| Clause | PA | Code | Note |
|---|---|---|---|
| (a) D, the first date no later than 2026-11-02 at whose end all five hold | :2314–2315 | first_d (:672), d_not_first_error (:971), date_error (:1031), LAST_D (:54), AM22_IN_FORCE (:53) | m4 |
| (a)1, every registered run settled | :2316 | condition_1 (:446) | |
| (a)2, the trigger evaluated and entered; E6-KD settled if it fired; due by the second day; the non-finite case | :2317–2323 | condition_2 (:466–503), TRIGGER_DUE_DAYS (:57), records.l_am3_trigger (records.py:377) | |
| (a)3, E6 seed 42's derived artifacts | :2324–2325 | condition_3 (:505) | |
| (a)4, every lane serving TEST IMPLEMENTED or entered as not required | :2326–2327 | condition_4 (:524–571), not_required_error (:574) | |
| (a)5, the manifest drafted and its validator passing | :2328–2330 | condition_5 (:602–630) | TMAN-B2-RULE-3; m5 |
| records only; verdicts, never scores | :2332–2335 | View.go_rule (:312), View.trigger (:288); B2R11's static scan | |
| (b) the freeze at the end of the day after D | :2336–2344 | schedule.am22_provider (:167), am22_candidate (:180), am22.providers (:737) | |
| (b) the freeze entry lists each condition with its record and date; its commit reaches the remote before the end of the freeze date | :2345–2346 | entries.py:67–69; freeze_fields (:983), freeze_conditions (:774); validate.g_freeze_entry_late (:1321), push_evidence_error (:1514) | audit M4 |
| (c) before the freeze entry exists, 2026-11-03 | :2347–2348 | providers(lookup=True) finds no record, so B-1's providers answer | R1-N6 (B-2.14) |
| (d) no discretion | :2349–2350 | D is computed, never chosen (d_not_first_error, :971) | |
| (e) TEST by 2026-11-08 | :2351–2352 | AM22_TEST_DEADLINE (schedule.py:41), the provider's test_deadline | |
| (f) the fallback | :2353–2357 | fallback_error (:697), freeze_entry_error's 1(f) branch (:910), FALLBACK_FREEZE (:55), tday_manifest.check_freeze_record (:308–344) | DL-24 R1-F2 |
| (g) ARM latency: not under item 2; condition 4 does not read L-AM16-ARM; cut at the freeze | :2358–2362 | ARM_LANE (:56); entries.chk_rule (:492–495); frozen_body_error (:839); `entry cut --d-date` (tday_manifest.py:372–377) | TMAN-B2-RULE-3 |
| (h) unchanged | :2363–2369 | B-1 | |

**Item 2:**

| Clause | PA | Code | Note |
|---|---|---|---|
| (a) the cutoff: 2026-10-20, or two days after the last moving record, never past 2026-10-29 | :2372–2380 | slot_cutoff (entries.py:352–370), cutoff_day (schedule.py:243–252), moving_reader (entries.py:274–333), CUTOFF_MOVING_GATES (schema.py:93) | DL-24 R1-F3 |
| (a) a teacher fine-tune is launched at its launch record | :2380–2384 | B-1's RunIndex | |
| (a) the entry is made the day after, from the timestamps that meet the date; why, since when, and the arm's "not launched" record | :2384–2389 | chk_date (:570–572), late_entry_report (:1033), chk_reason (:533–544), chk_arm_results (:629) | |
| (b) a repeat: three days after the attempt's last row; a teacher's last stamped log line (UTC, year 2026), else the day its stop is entered | :2390–2399 | slot_repeat (entries.py:373–395), repeat_cutoff_day (schedule.py:255), last_row (completion.py:555), teacher_last_row (:346), last_stamp (:308), logged_reader (entries.py:347) | |
| (b) not a sweep candidate's on-course repeat | :2399–2400 | chk_decision_date (:685–689) | m2 |
| (c) a parent still running exempts nothing | :2401–2403 | cutoff_day's `absent` (schedule.py:247–248), slot_cutoff (:365–366), chk_arm_partial (:664–665) | |
| (d) never-cut runs have no cutoff; the cut order is not consulted | :2404–2406 | chk_never_cut (entries.py:503), chk_order_position (:671–679) | |
| (f) cards: "no card" only with its evidence, else a deviation | :2413–2420 | no_card_report (:440), evidence_reader (:421), validate.report_lines (:1705), tday_manifest.py:202–204 | B2R8 |

**Item 4 and Code:**

| Clause | PA | Code | Note |
|---|---|---|---|
| 4. the freeze date and its source, each condition's record and date, every launch-cutoff cut, the ARM measurement's status | :2440–2451 | check-freeze's RECORD lines (am22.check_lines, :1080), and the entries | a records matter (B-2.15) |
| Code: the check beside the validator, printing each condition's record or what is missing | :2453–2458 | check-freeze (tday_manifest.py:579), src/tman/am22.py | |
| Code: item 2's cuts need no code beyond their entries | :2459 | rule cutoff | |

**AM-23's extensions:**
- E6-X is excluded with its excluded E3-X and on a "no" condition (View.registered; condition_entry, :330).
- E6-E2 is a cuttable run.
- E7-E2's "not made: code not merged" is B-1's not_made kind (R1-N7, B-2.14).

### B-2.7 Formats

**Kind `freeze`** (B2R5) has B-1's common keys (ENTRY_FILE_COMMON) and:
- **B-1's keys** `body_sha256` (sha256), `freeze_date` (day) and `source` (str) (entries.py:62);
- **B-2's keys** `d_date` (day or null) and `conditions` (entries.py:67–69). `conditions` is null or a list of `{number int, records [{kind str, ref str, sha256 sha256-or-null, date day}]}`.

The source is "AM-22 item 1(b)" or "AM-22 item 1(f)". Under 1(b), `date` = `freeze_date` = D + 1, the five conditions are numbered 1 to 5, and every record is dated no later than D and committed by D's end. Under 1(f), `d_date` and `conditions` are null and `freeze_date` is the resolver's, 2026-11-03 or later (m8).

**A fixture example.** The record that smoke_tman_am22's AmUnlock world prints with `entry freeze --d-date 2026-10-30 --manifest <the body to freeze>`, then enters on 2026-10-31. Values are the fixture's; sha256s and most records are elided as "…", and the condition texts are am22.TEXTS (:128–132):

```json
{
  "format": "tday_entry/1",
  "kind": "freeze",
  "slots": [],
  "date": "2026-10-31",
  "amendment": "AM-22",
  "printed_by": {"script": "scripts/tday_manifest.py", "commit": "<HEAD of the records checkout when printed>"},
  "corrects": null,
  "push_evidence": null,
  "body_sha256": "<canonical sha256 of the body: the draft at C_D plus ARM:latency's freeze cut>",
  "freeze_date": "2026-10-31",
  "source": "AM-22 item 1(b)",
  "d_date": "2026-10-30",
  "conditions": [
    {"number": 1, "records": [
      {"kind": "cut entry of G:s42:80k (cutoff)", "ref": "reports/derived/tday_entries/cut__G_s42_80k__20261020__01.json",
       "sha256": "…", "date": "2026-10-21"},
      "… one record or more per registered run: completion, cut, not-run, the AM-3 trigger for E6-KD"]},
    {"number": 2, "records": [
      {"kind": "AM-3 trigger record", "ref": "reports/derived/e6kd_trigger.json", "sha256": "…", "date": "2026-10-22"},
      {"kind": "derived artifacts of E6:s42:15ep", "ref": "…", "sha256": "…", "date": "…"}]},
    {"number": 3, "records": [{"kind": "E6:s42:sel hashed in the draft", "ref": "…", "sha256": "…", "date": "…"}]},
    {"number": 4, "records": [
      {"kind": "lane register", "ref": "reports/derived/tday_lanes.json", "sha256": "…", "date": "2026-10-30"},
      {"kind": "lane L-AM3 IMPLEMENTED", "ref": "DL-…", "sha256": null, "date": "2026-10-30"},
      {"kind": "lane L-AM3 merge commit", "ref": "<E>", "sha256": null, "date": "2026-10-12"},
      {"kind": "lane L-AM23Q not required", "ref": "reports/derived/tday_entries/…", "sha256": "…", "date": "2026-10-29"},
      {"kind": "lane L-LAT-THREADS not required", "ref": "reports/derived/tday_entries/…", "sha256": "…",
       "date": "2026-10-29"}]},
    {"number": 5, "records": [
      {"kind": "draft manifest", "ref": "reports/derived/tday_manifest.json", "sha256": "…", "date": "2026-10-30"},
      {"kind": "evaluator commit", "ref": "<E>", "sha256": null, "date": "2026-10-12"}]}]
}
```

A commit whose push the orchestrator confirmed on a later day adds `{"kind": "confirmed push", "ref": <commit>, "sha256": null, "date": <the push day>}` to the condition that relies on it (audit M2; freeze_conditions :774–784).

**The cutoff cut, with the DL-137 fixture** (smoke_tman_cuts.py:515–557). The fixture's moving records are:
- **E3-R1's end-of-run record**, from run `e3-r1_s42_lambda0.5_alpha25_80k_a2`. Its wall_clock_end is DL-135's 2026-10-21T16:58:32.849198Z, which is 2026-10-22 00:58 in Manila.
- **The pilot's winner record**, `u4_clip/clip_selection.json` (sha256 c4…c4). It is dated 2026-10-18 by DL-139's RECORDED row: the first ISO date after its sha256.

The cutoff is max(2026-10-20, 2026-10-22 + 2) = 2026-10-24, inside the ceiling 2026-10-29. The record that set it is E3-R1's, the last-dated.

```json
{
  "format": "tday_entry/1", "kind": "cut", "slots": ["E6-R1:s42:15ep"], "date": "2026-10-24",
  "amendment": "AM-19 item 3(b)", "printed_by": {"script": "scripts/tday_manifest.py", "commit": "000…000"},
  "corrects": null, "push_evidence": null,
  "rule": "cutoff", "reason": "gate not met", "launched": false, "run_ids": [],
  "since": {"date": "2026-10-22", "record": {"path": "reports/derived/tday_entries/x.json", "sha256": "fff…fff"}},
  "arithmetic": null, "telemetry": [], "comparator": null, "results_known": [], "cut_order_position": null,
  "rule_record": {"path": "e3-r1_s42_lambda0.5_alpha25_80k_a2/e3_telemetry.jsonl",
                  "sha256": "<the fixture telemetry's sha256>", "date": "2026-10-22"}
}
```

**What its checks show** (MEASURED, B5):
- the cut passes (`dl137_e6r1_cut_at_the_end_of_1024_m3`);
- dated by the UTC day, 2026-10-23, it is refused cut_date_mismatch (`dl137_dated_by_the_utc_day_refused_m3`);
- naming the pilot's record as its setter, it is refused the same (`dl137_naming_the_pilot_record_as_its_setter_refused_B2R2`);
- entered on 2026-10-26 it gives REPORT "deviation: entry later than its day … due 2026-10-25, entered 2026-10-26" (`dl137_entered_a_day_late_is_a_report`).

The fixture's `amendment` is the cut helper's. The real printer takes `--amendment`, and the real entry's results_known follows n8 (B-2.10).

### B-2.8 Refusal codes

B-2 adds four gate codes and one command-line code. Codes go from 89 to 94 (smoke_tman_mutations.py:2096) and gates from 84 to 88 (validate.py:1649–1650).

| Code | Where (5fb2759) | Meaning |
|---|---|---|
| `freeze_entry_invalid` | gate validate.py:1649 (g_freeze_entry_invalid :1289 → am22.freeze_record_error :938, freeze_entry_error :910, frozen_body_error :839); printer am22.py:890, :901, :906, :997, :1018, :1025, tday_manifest.py:318, :327, :332 | kind freeze's AM-22 fields: the source; the D-to-freeze_date relation; the five conditions; a record dated after D or not committed by D's end; the frozen body beyond TMAN-B2-RULE-3; a second item 1(b) freeze |
| `am22_condition_unmet` | gate validate.py:1649 (g_am22_condition_unmet :1298 → am22.condition_unmet_error :959); printer am22.py:1010 | the freeze record names a D at which a condition does not hold, or records conditions other than the recomputed ones |
| `am22_d_not_first` | gate validate.py:1650 (g_am22_d_not_first :1307 → d_not_first_error :971, fallback_error :697); printer am22.py:1006, tday_manifest.py:344; check-freeze date_error :1041, :1044 | the conditions also hold at an earlier date, D is later than 2026-11-02 or before 2026-10-17, or an item 1(f) freeze has a D |
| `freeze_entry_late` | gate validate.py:1650 (g_freeze_entry_late :1321, push_evidence_error :1514); the unlock printer tday_manifest.py:256–280 | the freeze record's commit did not reach the remote in [the first instant of its date, the end of its freeze date), or its row was entered by a later commit |
| `am22_date_not_ended` (command line; REFUSAL_EXTRA) | am22.date_error :1031–1037; tday_manifest.clock :102, :339 | a date whose end has not passed, or not a date (m6) |

**Each gate code has:**
- **a case reached with every earlier gate passing:** smoke_tman_am22.py case_unlock_gates (:1677), case_freeze_entry_units (:1740), case_freeze_printer (:1491) and case_cli (:1249);
- **mutations killed by named cases** (B-2.9), for example:
  - freeze_entry_invalid_unchecked, condition_unmet_unchecked and d_accepted_when_an_earlier_date_passes;
  - freeze_push_evidence_ignored and date_whose_end_has_not_passed.

**Existing codes B-2 reuses:**
- cut_date_mismatch: a cutoff cut dated off its date, or naming another record;
- cut_rule_unknown: a cutoff cut of the measurement (m7);
- decision_date_cut_mismatch: m2;
- schedule_invalid: the dates provider, and the bounds of `--am22-d`;
- freeze_source_conflict: two item 1(b) records, or m10's rows;
- entry_field: the printers' flags.

### B-2.9 Mutations

**Counts:** 374 mutations at MT, 563 at 5fb2759. B-2 adds 189 (B2-4 73, B2-F1 80, B2-F2 36) and removes none.

**Renames.** B2-F1 renamed two of B2-4's (DL-24 R3-16): `unlock_freeze_by_lookup` became `unlock_freeze_record_unread`, and `item_1b_provider_cites_the_record_s_sha256` became `item_1b_provider_source_carries_a_sha256`. The tables below therefore list 71, 82 and 36 ids, by the commit that holds each id now.

**Existing entries B-2 changed** (REPOSITORY-PROVEN by comparing the lists at MT, 4315246, d676a97 and 5fb2759):
- **B-1's `printer_freeze_record_body`:** its text moved with the item 1(b) path (B2-F1).
- **In B2-F1:**
  - `c_x_is_the_tip`, `boundary_day_report_dropped` and `printer_skips_d_first`: texts moved;
  - `condition_2_reads_the_trigger_s_score`: its guard list emptied (R3-16 (c));
  - `body_sha256_of_the_draft` and `freeze_list_s_entry_key_unread`: retargeted from the world builder's check (R3-04);
  - `two_item_1b_freezes_no_conflict`: retargeted (R2-13).
- **In B2-F2:** `arm_cutoff_dated_by_each_run`, retargeted to the DL-168 case through the moving reader (R3-08).

**Audit m1.** B-1's mutation texts on the lines m1 names (chk_reason, chk_stopped, chk_arm_running) are unchanged. Each text of the 563 occurs exactly once (MEASURED with $SCR/mut_texts.py).

**Detail-only kills** (R3-16 (d), stated): `freeze_date_not_d_plus_1` and `arm_measured_after_c_d_not_a_stop` stay freeze_entry_invalid under mutation. Each is killed by its check's detail text.

**The new mutations, each KILLED in B5** (smoke, case, the named check; then the codes it guards):

#### B2-4 (4315246): 71

| # | mutation | file | killed by (smoke, case, check) | codes |
|---|---|---|---|---|
| 1 | `am22_ceiling_regex_single_space` | schedule | cuts, case_cutoff_readers, `am22_dates_of_the_committed_text_M1` | schedule_invalid |
| 2 | `cutoff_ceiling_dropped` | schedule | cuts, case_cutoff_dates, `cutoff_never_past_the_ceiling_1029` | cut_date_mismatch |
| 3 | `cutoff_one_day_after_its_record` | schedule | cuts, case_cutoff_dates, `cutoff_two_days_after_the_last_record` | cut_date_mismatch |
| 4 | `repeat_cutoff_two_days_after_the_row` | schedule | cuts, case_cutoff_dates, `repeat_cutoff_three_days_after_the_last_row` | cut_date_mismatch |
| 5 | `unwritten_record_leaves_no_ceiling` | schedule | cuts, case_cutoff_dates, `cutoff_of_a_record_not_written_is_the_ceiling` | cut_date_mismatch |
| 6 | `moving_gate_treated_as_non_moving` | entries | cuts, case_cutoff_readers, `moving_records_of_e6r1_parent_end_and_pilot_m3` | cut_date_mismatch |
| 7 | `run_end_dated_by_its_utc_day` | entries | cuts, case_cutoff_dates, `dl137_e6r1_cut_at_the_end_of_1024_m3` | cut_date_mismatch |
| 8 | `cut_entered_day_from_the_entry_s_date` | entries | cuts, case_cutoff_rules, `cutoff_entered_the_day_after_passes` | cut_date_mismatch |
| 9 | `cutoff_entered_on_its_cutoff_date` | entries | cuts, case_cutoff_rules, `cutoff_entered_on_its_cutoff_date_refused` | cut_date_mismatch |
| 10 | `cutoff_s_setting_record_not_compared` | entries | cuts, case_cutoff_rules, `cutoff_2a_moved_record_not_named_refused` | cut_date_mismatch |
| 11 | `cutoff_2b_without_2a_s_reason` | entries | cuts, case_cutoff_rules, `cutoff_2b_takes_2a_s_reason_m1` | cut_reason_invalid |
| 12 | `cutoff_2b_of_a_stopped_run_refused` | entries | cuts, case_cutoff_rules, `cutoff_2b_repeat_three_days_after_the_last_row_passes` | stopped_run_cut_early |
| 13 | `cutoff_2c_arm_s_earlier_runs_complete` | entries | cuts, case_cutoff_rules, `cutoff_2c_arm_run_cut_while_its_teacher_runs` | arm_cut_partial |
| 14 | `cutoff_of_the_measurement` | entries | cuts, case_cutoff_rules, `cutoff_of_the_arm_measurement_refused_1g_m7` | cut_rule_unknown |
| 15 | `cutoff_2b_of_a_sweep_candidate` | entries | cuts, case_cutoff_rules, `cutoff_2b_of_a_sweep_candidate_s_repeat_refused_m2` | decision_date_cut_mismatch |
| 16 | `cutoff_2b_teacher_stop_entry_unread` | entries | cuts, case_cutoff_rules, `cutoff_2b_teacher_without_a_stamped_line_takes_its_stop_entry_m2` | cut_date_mismatch |
| 17 | `no_card_refused_instead_of_reported` | entries | cuts, case_cutoff_rules, `no_card_cut_is_never_refused_B2R8` | - |
| 18 | `arm_cutoff_dated_by_each_run` | entries | cuts, case_cutoff_dates, `dl168_l_arm_cut_by_its_first_run_s_date_through_the_moving_reader_R3_08` | cut_date_mismatch |
| 19 | `c_x_is_the_tip` | am22 | am22, case_commit_at_end, `c_x_walks_back_from_the_tip_M3` | - |
| 20 | `records_read_at_head_not_c_x` | am22 | am22, case_settled_ways, `entry_committed_after_x_is_not_read_at_x_M3` | - |
| 21 | `e6kd_registered_untriggered` | am22 | am22, case_registered, `registered_are_the_run_slots_but_e6kd_untriggered` | - |
| 22 | `go_rule_no_go_keeps_e6x` | am22 | am22, case_registered, `go_rule_no_go_excludes_e3_r1_and_e6_r1_m7` | - |
| 23 | `condition_entry_no_ignored` | am22 | am22, case_registered, `condition_entry_no_excludes_e6_r2_AM23_3c` | - |
| 24 | `never_cut_settles_every_way` | completion | am22, case_settled_ways, `never_cut_diverged_is_not_settled_until_superseded` | - |
| 25 | `never_cut_aborted_not_a_stop` | am22 | am22, case_settled_ways, `never_cut_aborted_is_a_stop_to_ask` | - |
| 26 | `never_cut_run_settled_by_a_cut` | am22 | am22, case_settled_ways, `never_cut_cut_is_not_settled` | - |
| 27 | `freeze_cut_settles_a_run` | am22 | am22, case_settled_ways, `a_freeze_cut_settles_no_run` | - |
| 28 | `am7_consequence_not_a_stop` | am22 | am22, case_settled_ways, `am7_full_consequence_is_a_stop_to_ask_n3` | - |
| 29 | `condition_2_reads_the_trigger_s_score` | am22 | am22, case_condition_2_cases, `triggered_e6kd_not_settled_condition_2_missing` | - |
| 30 | `late_trigger_fails_condition_2` | am22 | am22, case_condition_2_cases, `trigger_entered_late_is_a_report_never_a_failure` | - |
| 31 | `trigger_due_by_the_third_day` | am22 | am22, case_condition_2_cases, `trigger_entered_late_is_a_report_never_a_failure` | - |
| 32 | `non_finite_e6s42_read_on` | am22 | am22, case_condition_2_cases, `e6s42_non_finite_needs_the_new_amendment_n4` | - |
| 33 | `condition_4_reads_l_am16_arm` | am22 | am22, case_conditions_in_the_world, `condition_4_reads_implemented_and_not_required_lanes_not_arm_1g` | - |
| 34 | `lane_row_status_not_read` | am22 | am22, case_condition_4_rows_at_c_x, `lane_row_implemented_after_x_is_missing_at_x_M3` | - |
| 35 | `boundary_day_report_dropped` | am22 | am22, case_report_lines, `boundary_day_reports_name_the_commits_of_x_M2` | - |
| 36 | `trigger_not_triggered_refused` | records | am22, case_trigger_reader, `l_am3_reader_gives_the_verdict_only_B2R13` | - |
| 37 | `trigger_reader_unwired` | records | am22, case_static, `trigger_reader_is_l_am3_s_B2R13` | - |
| 38 | `date_whose_end_has_not_passed` | am22 | am22, case_cli, `cli_date_not_ended_refused_m6` | am22_date_not_ended |
| 39 | `date_after_1102_checked` | am22 | am22, case_cli, `cli_date_after_1102_refused_1f` | am22_d_not_first |
| 40 | `d_from_x_alone` | am22 | am22, case_d_first, `d_is_the_first_date_whose_end_has_all_five_m4` | - |
| 41 | `d_scanned_from_the_day_before` | am22 | am22, case_d_first, `d_is_the_first_date_whose_end_has_all_five_m4` | - |
| 42 | `d_accepted_when_an_earlier_date_passes` | am22 | am22, case_d_first, `unlock_freeze_of_a_later_d_refused_B2R7_m4` | am22_d_not_first |
| 43 | `printer_skips_d_first` | am22 | am22, case_d_first, `printer_freeze_of_a_later_d_refused_m4` | am22_d_not_first |
| 44 | `condition_5_accepts_any_unentered_slot` | am22 | am22, case_condition_5, `condition_5_missing_with_a_run_slot_unentered_RULE3` | - |
| 45 | `condition_5_refuses_the_measurement_s_entry` | am22 | am22, case_condition_5, `condition_5_met_its_one_unmet_item_the_arm_measurement_RULE3` | - |
| 46 | `condition_5_without_the_draft` | am22 | am22, case_condition_5, `condition_5_records_the_draft_at_c_x_and_the_evaluator_commit` | - |
| 47 | `frozen_body_differs_from_the_draft` | am22 | am22, case_freeze_printer, `printer_body_differing_beyond_the_arm_freeze_cut_RULE3` | freeze_entry_invalid |
| 48 | `frozen_body_any_slot_joins` | am22 | am22, case_freeze_printer, `printer_another_slot_joining_the_draft_RULE3` | freeze_entry_invalid |
| 49 | `frozen_body_any_arm_cut` | am22 | am22, case_freeze_printer, `printer_arm_cut_dated_after_the_freeze_date_RULE3` | freeze_entry_invalid |
| 50 | `frozen_cut_entered_by_its_date` | am22 | am22, case_freeze_printer, `printer_arm_cut_entered_on_another_day_RULE3` | freeze_entry_invalid |
| 51 | `arm_measured_after_c_d_not_a_stop` | am22 | am22, case_freeze_printer, `printer_arm_measurement_complete_after_c_d_is_a_stop_RULE3` | freeze_entry_invalid |
| 52 | `body_sha256_of_the_draft` | am22 | am22, case_unlock_world, `am22_freeze_printer_passes_in_the_world_B2R5` | - |
| 53 | `freeze_dated_any_day` | am22 | am22, case_freeze_printer, `printer_freeze_dated_other_than_d_plus_1_m8` | freeze_entry_invalid |
| 54 | `freeze_date_not_d_plus_1` | am22 | am22, case_unlock_gates, `unlock_freeze_date_not_d_plus_1_refused_B2R7` | freeze_entry_invalid |
| 55 | `freeze_entry_invalid_unchecked` | validate | am22, case_unlock_gates, `unlock_freeze_date_not_d_plus_1_refused_B2R7` | freeze_entry_invalid |
| 56 | `condition_unmet_unchecked` | validate | am22, case_unlock_gates, `unlock_freeze_conditions_not_recomputed_refused_B2R7` | am22_condition_unmet |
| 57 | `freeze_push_evidence_ignored` | validate | am22, case_unlock_gates, `unlock_entry_without_the_freeze_push_evidence_refused_M4` | freeze_entry_late |
| 58 | `freeze_push_window_unbounded` | validate | am22, case_unlock_gates, `freeze_record_pushed_after_the_end_of_its_date_refused_M4` | freeze_entry_late |
| 59 | `freeze_list_s_entry_key_unread` | validate | am22, case_unlock_world, `am22_unlock_printer_passes_in_the_world_M4` | freeze_entry_late |
| 60 | `unlock_printer_skips_m4` | tday_manifest | am22, case_unlock_printer, `printer_unlock_without_the_freeze_push_evidence_M4` | freeze_entry_late |
| 61 | `printer_item_1b_without_d_date` | tday_manifest | am22, case_freeze_printer, `printer_item_1b_freeze_without_d_date_m9` | freeze_entry_invalid |
| 62 | `printer_takes_given_fields` | tday_manifest | am22, case_freeze_printer, `printer_freeze_fields_given_with_d_date_m9` | entry_field |
| 63 | `build_ignores_am22_d` | tday_manifest | am22, case_cli_am22_d, `cli_build_am22_d_writes_the_candidate_s_draft_B2R6_m5` | - |
| 64 | `validate_ignores_am22_d` | tday_manifest | am22, case_cli_am22_d, `cli_validate_am22_d_leaves_the_arm_measurement_RULE3` | - |
| 65 | `append_without_am22_s_provider` | tday_manifest | am22, case_providers, `every_freeze_instant_call_puts_am22_s_provider_in_front_B2R6` | - |
| 66 | `am22_provider_under_schedule_r` | schedule | am22, case_providers, `schedule_r_takes_no_am22_freeze_B2R6` | - |
| 67 | `candidate_freeze_at_the_end_of_d` | schedule | am22, case_providers, `candidate_freeze_is_the_end_of_d_plus_1_B2R6` | - |
| 68 | `item_1f_record_gives_a_provider` | am22 | am22, case_providers, `item_1f_record_gives_no_provider_B2R6` | - |
| 69 | `two_item_1b_freezes_no_conflict` | am22 | am22, case_providers, `two_item_1b_freeze_records_conflict_B2R6` | freeze_source_conflict |
| 70 | `freeze_delay_beside_item_1b_no_conflict` | am22 | am22, case_providers, `freeze_delay_beside_an_item_1b_freeze_conflicts_m10` | freeze_source_conflict |
| 71 | `freeze_kind_requires_d` | entries | validate, case_printers, `printer_freeze_record_of_this_body_passes` | entry_field |

#### B2-F1 (d676a97): 82

| # | mutation | file | killed by (smoke, case, check) | codes |
|---|---|---|---|---|
| 1 | `unlock_freeze_record_unread` | validate | am22, case_unlock_world, `am22_unlock_world_passes_every_unlock_gate_B2R7_M4` | schedule_invalid |
| 2 | `unlock_takes_the_candidate_for_am22_d` | validate | am22, case_unlock_world, `unlock_reads_the_freeze_record_not_am22_d_R2_3` | - |
| 3 | `item_1b_provider_source_carries_a_sha256` | am22 | am22, case_unlock_world, `am22_unlock_world_passes_every_unlock_gate_B2R7_M4` | schedule_invalid |
| 4 | `c_x_ignores_confirmed_push_days` | am22 | am22, case_commit_at_end, `c_x_stops_before_a_commit_pushed_later_M2` | - |
| 5 | `c_x_at_the_end_of_the_utc_day` | am22 | am22, case_commit_at_end, `c_x_is_read_at_the_end_of_the_manila_day_R3_09` | - |
| 6 | `c_x_off_the_first_parent_chain` | am22 | am22, case_commit_at_end, `c_x_reads_the_first_parent_chain_R3` | - |
| 7 | `record_dated_by_its_commit_day_only` | am22 | am22, case_confirmed_push, `confirmed_push_keeps_d_and_dates_the_trigger_by_its_push_M2` | - |
| 8 | `late_trigger_by_its_commit_day_only` | am22 | am22, case_confirmed_push, `trigger_confirmed_pushed_after_its_due_day_is_late_M2` | - |
| 9 | `freeze_conditions_drop_confirmed_pushes` | am22 | am22, case_confirmed_push, `freeze_conditions_carry_each_confirmed_push_M2` | - |
| 10 | `boundary_report_asks_a_confirmed_commit_again` | am22 | am22, case_confirmed_push, `confirmed_commit_asked_no_boundary_report_again_M2` | - |
| 11 | `record_line_hides_the_push_day` | am22 | am22, case_confirmed_push, `record_lines_print_the_confirmed_push_day_M2` | - |
| 12 | `unlock_recheck_ignores_confirmed_pushes` | am22 | am22, case_unlock_gates, `freeze_record_with_a_confirmed_push_recomputed_with_it_M2` | - |
| 13 | `pushed_not_commit_equals_day_read` | tday_manifest | am22, case_cli, `cli_pushed_not_commit_equals_day_refused_M2` | - |
| 14 | `pushed_not_a_commit_accepted` | am22 | am22, case_cli, `cli_pushed_not_a_commit_refused_M2` | - |
| 15 | `pushed_on_its_committer_day_accepted` | am22 | am22, case_cli, `cli_pushed_on_its_committer_day_refused_M2` | - |
| 16 | `pushed_commit_no_record_needs_accepted` | am22 | am22, case_freeze_printer, `printer_pushed_commit_no_record_needs_refused_M2` | - |
| 17 | `pushed_on_any_entry` | tday_manifest | am22, case_measurement_cut_printer, `printer_pushed_is_the_freeze_entry_s_M2` | - |
| 18 | `boundary_reports_of_x_only` | am22 | am22, case_report_lines, `boundary_day_report_of_x_minus_1_by_its_manila_day_R3_10` | - |
| 19 | `boundary_report_by_the_utc_day` | am22 | am22, case_report_lines, `boundary_day_report_of_x_minus_1_by_its_manila_day_R3_10` | - |
| 20 | `d_s_own_check_not_printed` | am22 | am22, case_d_first, `check_freeze_prints_d_s_own_check_and_boundary_reports_R3_02` | - |
| 21 | `printer_prints_no_boundary_report` | tday_manifest | am22, case_freeze_printer, `printer_prints_d_s_boundary_day_reports_R3_02` | - |
| 22 | `draft_dated_by_its_adding_commit` | am22 | am22, case_d_first, `draft_record_names_the_commit_of_its_last_change_R2_6` | - |
| 23 | `index_reads_the_launch_log_at_c_x` | am22 | am22, case_settled_ways, `run_complete_at_x_its_launch_lines_committed_later_settles_R1_F5` | - |
| 24 | `condition_5_drops_the_reports_on_the_rule3_path` | am22 | am22, case_condition_5, `condition_5_keeps_the_validator_s_reports_on_the_rule3_path_R1_F9` | - |
| 25 | `e6s42_record_cites_the_artifact_s_sha256` | am22 | am22, case_condition_2_cases, `e6s42_made_artifacts_read_from_the_draft` | - |
| 26 | `trigger_due_day_by_the_utc_day` | am22 | am22, case_condition_2_cases, `trigger_due_day_counts_from_the_manila_day_R3_09` | - |
| 27 | `trigger_due_by_the_end_of_the_first_day` | am22 | am22, case_conditions_in_the_world, `trigger_entered_on_its_due_day_prints_no_report_R3_11` | - |
| 28 | `trigger_late_on_its_due_day` | am22 | am22, case_conditions_in_the_world, `trigger_entered_on_its_due_day_prints_no_report_R3_11` | - |
| 29 | `trigger_not_entered_read` | am22 | am22, case_condition_2_cases, `trigger_committed_in_no_recorded_row_is_missing_R3_15` | - |
| 30 | `trigger_reader_refusal_read_as_no_trigger` | am22 | am22, case_condition_2_cases, `trigger_reader_refusal_is_missing_R3_15` | - |
| 31 | `trigger_reader_without_a_verdict_read_as_no_trigger` | am22 | am22, case_condition_2_cases, `trigger_reader_without_a_boolean_verdict_is_missing_R3_15` | - |
| 32 | `condition_entries_disagreeing_read_as_one` | am22 | am22, case_registered, `condition_entries_that_disagree_are_missing_AM23_3b` | - |
| 33 | `condition_3_reads_no_hash` | am22 | am22, case_condition_3, `condition_3_missing_while_e6s42_is_not_hashed_R3_12` | - |
| 34 | `lane_register_out_of_form_or_empty_read` | am22 | am22, case_condition_4_rows_at_c_x, `lane_register_listing_no_lane_R3_15` | - |
| 35 | `lane_register_unrecorded_read` | am22 | am22, case_condition_4_rows_at_c_x, `lane_register_in_no_recorded_row_R3_15` | - |
| 36 | `lane_merge_commit_unchecked` | am22 | am22, case_condition_4_rows_at_c_x, `lane_merge_commit_not_in_the_records_is_missing` | - |
| 37 | `lane_row_naming_unchecked` | am22 | am22, case_condition_4_rows_at_c_x, `lane_row_not_naming_the_lane_and_its_merge_is_missing_R3_15` | - |
| 38 | `not_required_without_its_entry_read` | am22 | am22, case_condition_4_rows_at_c_x, `lane_not_required_without_its_entry_is_missing_R3_15` | - |
| 39 | `not_required_entry_unentered_read` | am22 | am22, case_condition_4_rows_at_c_x, `lane_not_required_entry_not_entered_is_missing_R3_15` | - |
| 40 | `not_required_coverage_unchecked` | am22 | am22, case_condition_4_rows_at_c_x, `lane_not_required_serving_an_uncut_slot_is_missing_R3_15` | - |
| 41 | `fallback_unchecked_at_the_unlock` | validate | am22, case_fallback_1f, `unlock_item_1f_freeze_with_a_d_refused_R1_F2` | am22_d_not_first |
| 42 | `fallback_finds_no_candidate_draft` | am22 | am22, case_fallback_1f, `printer_item_1f_with_a_d_refused_R1_F2` | - |
| 43 | `printer_item_1f_without_the_fallback` | tday_manifest | am22, case_fallback_1f, `printer_item_1f_with_a_d_refused_R1_F2` | am22_d_not_first |
| 44 | `printer_item_1f_before_the_end_of_1102` | tday_manifest | am22, case_fallback_1f, `printer_item_1f_before_the_end_of_1102_refused_R1_F2` | - |
| 45 | `printer_item_1f_without_the_run_directories` | tday_manifest | am22, case_fallback_1f, `printer_item_1f_without_the_run_directories_refused_R1_F2` | - |
| 46 | `printer_item_1f_resolver_from_the_body` | tday_manifest | am22, case_fallback_1f, `printer_item_1f_of_a_body_at_am22_s_freeze_refused_m8` | - |
| 47 | `freeze_source_unchecked` | am22 | am22, case_freeze_entry_units, `freeze_entry_of_another_source_refused_B2R5` | - |
| 48 | `item_1f_freeze_naming_d_accepted` | am22 | am22, case_freeze_entry_units, `item_1f_freeze_naming_d_refused_B2R5` | - |
| 49 | `item_1f_freeze_before_1103_accepted` | am22 | am22, case_freeze_entry_units, `item_1f_freeze_before_1103_refused_R2_3` | - |
| 50 | `item_1f_resolver_unread` | am22 | am22, case_freeze_entry_units, `item_1f_freeze_off_the_resolver_s_date_refused_m8` | - |
| 51 | `conditions_misnumbered_accepted` | am22 | am22, case_freeze_entry_units, `item_1b_conditions_misnumbered_refused_B2R5` | - |
| 52 | `records_dated_after_d_accepted` | am22 | am22, case_freeze_entry_units, `item_1b_record_dated_after_d_refused_B2R7` | - |
| 53 | `recorded_condition_dated_early_accepted` | am22 | am22, case_freeze_entry_units, `recorded_condition_dated_before_its_commits_day_refused_M2` | - |
| 54 | `unlock_recheck_skips_unmet_conditions` | am22 | am22, case_freeze_entry_units, `unlock_recheck_at_a_d_whose_conditions_do_not_hold_R3_05` | am22_condition_unmet |
| 55 | `printer_skips_unmet_conditions` | am22 | am22, case_freeze_printer, `printer_d_whose_conditions_do_not_hold_B2R7` | am22_condition_unmet |
| 56 | `freeze_record_after_d_accepted` | am22 | am22, case_unlock_gates, `freeze_record_citing_a_record_committed_after_d_B2R7` | - |
| 57 | `d_not_first_unbounded_above` | am22 | am22, case_d_first, `d_not_first_d_after_1102_is_item_1f_R3` | - |
| 58 | `d_not_first_unbounded_below` | am22 | am22, case_d_first, `d_not_first_d_before_1017_refused_R2_15` | - |
| 59 | `date_not_a_date_unchecked` | am22 | am22, case_cli, `cli_date_not_a_date_refused` | - |
| 60 | `date_before_am22_unchecked` | am22 | am22, case_cli, `cli_date_before_am22_refused` | - |
| 61 | `check_freeze_now_unparsed` | tday_manifest | am22, case_cli, `cli_now_not_a_utc_instant_refused` | - |
| 62 | `am22_d_unbounded` | am22 | am22, case_providers, `am22_d_before_1017_refused_R2_15` | schedule_invalid |
| 63 | `am22_d_unbounded_above` | am22 | am22, case_providers, `am22_d_after_1102_refused_R2` | - |
| 64 | `am22_d_format_unchecked` | am22 | am22, case_providers, `am22_d_not_a_date_refused_B2R6` | - |
| 65 | `two_item_1b_conflict_flag_ignored` | am22 | am22, case_providers, `two_item_1b_records_without_the_conflict_check_give_the_first_R2_13` | - |
| 66 | `schedule_gate_raises_the_conflict` | validate | am22, case_unlock_gates, `two_item_1b_records_refused_at_their_gate_in_freeze_mode_R2_13` | freeze_source_conflict |
| 67 | `corrected_freeze_record_kept` | am22 | am22, case_providers, `a_corrected_item_1b_record_is_no_freeze_of_record_N9` | - |
| 68 | `validate_reads_am22_d_in_unlock_mode` | tday_manifest | am22, case_cli, `cli_validate_am22_d_in_unlock_mode_refused_R2_3` | schedule_invalid |
| 69 | `build_validates_without_am22_d` | tday_manifest | am22, case_cli_am22_d, `cli_build_am22_d_writes_the_candidate_s_draft_B2R6_m5` | schedule_invalid |
| 70 | `d_date_cut_of_any_slot` | tday_manifest | am22, case_measurement_cut_printer, `printer_d_date_cut_of_a_run_slot_refused_R1_F1` | - |
| 71 | `freeze_body_adds_no_cut` | am22 | am22, case_unlock_world, `am22_body_to_freeze_printed_by_freeze_body_R2_8` | - |
| 72 | `freeze_body_duplicates_a_draft_entry` | am22 | am22, case_freeze_body, `freeze_body_of_a_slot_the_draft_enters_refused_R2_8` | - |
| 73 | `freeze_body_without_its_check` | am22 | am22, case_freeze_body, `freeze_body_of_a_cut_entered_after_the_freeze_date_refused_R2_8` | - |
| 74 | `freeze_body_d_date_unbounded` | am22 | am22, case_freeze_body, `freeze_body_d_date_after_1102_refused_R2_8` | - |
| 75 | `frozen_body_measurement_gone_unchecked` | am22 | am22, case_freeze_printer, `printer_measurement_in_neither_draft_nor_body_RULE3_R3_01` | - |
| 76 | `printer_body_not_validated` | am22 | am22, case_freeze_printer, `printer_body_refused_in_freeze_mode_R2` | - |
| 77 | `second_item_1b_freeze_printed` | am22 | am22, case_freeze_printer, `printer_second_item_1b_freeze_refused_R3_14` | - |
| 78 | `printer_without_the_run_directories` | tday_manifest | am22, case_freeze_printer, `printer_without_the_run_directories_refused` | - |
| 79 | `unlock_printer_row_commit_unchecked` | tday_manifest | am22, case_unlock_printer, `printer_freeze_row_entered_by_a_later_commit_R1_F8` | - |
| 80 | `unlock_printer_window_unbounded` | tday_manifest | am22, case_unlock_printer, `printer_freeze_pushed_after_the_end_of_its_date_M4` | freeze_entry_late |
| 81 | `freeze_row_entered_later_accepted` | validate | am22, case_unlock_gates, `freeze_row_entered_by_a_later_commit_refused_R1_F8_R2_7` | freeze_entry_late |
| 82 | `freeze_list_keyed_unlock_entry_accepted` | validate | am22, case_unlock_gates, `freeze_list_naming_the_record_under_unlock_entry_refused_R2_12` | freeze_entry_late |

#### B2-F2 (5fb2759): 36

| # | mutation | file | killed by (smoke, case, check) | codes |
|---|---|---|---|---|
| 1 | `record_after_the_ceiling_counts` | entries | cuts, case_cutoff_rules, `cutoff_2c_record_written_after_the_ceiling_is_unwritten_R1_F3` | cut_date_mismatch |
| 2 | `one_unwritten_record_still_names_a_setter` | entries | cuts, case_cutoff_rules, `cutoff_2c_one_record_unwritten_rule_record_null_R4_4` | cut_date_mismatch |
| 3 | `repeat_row_dated_by_its_utc_day` | entries | cuts, case_cutoff_rules, `cutoff_2b_row_after_16z_counts_from_its_manila_day_R4_2` | cut_date_mismatch |
| 4 | `multi_slot_setters_unexplained` | entries | cuts, case_cutoff_rules, `cutoff_slots_set_by_different_records_one_entry_each_R4` | - |
| 5 | `diverged_attempt_awaits_a_repeat` | entries | cuts, case_cutoff_rules, `cutoff_2b_after_a_diverged_attempt_refused_R3_15` | cut_date_mismatch |
| 6 | `slots_with_different_dates_unchecked` | entries | cuts, case_cutoff_rules, `cutoff_slots_with_different_dates_refused_R3_15` | - |
| 7 | `unread_cutoff_refused_in_a_draft` | entries | cuts, case_cutoff_rules, `cutoff_unread_in_a_draft_s_context_stands_R3_15` | - |
| 8 | `cutoff_2b_arm_run_exemption_dropped` | entries | cuts, case_cutoff_rules, `cutoff_2b_of_a_stopped_arm_run_passes_m1` | arm_run_cut_while_running |
| 9 | `cutoff_reason_rule_freeze_only` | entries | cuts, case_cutoff_rules, `cutoff_reason_outside_the_list_refused` | cut_reason_invalid |
| 10 | `evidence_unreadable_refuses` | entries | cuts, case_cutoff_printer, `printer_no_card_since_an_uncommitted_file_is_a_report_R1_F7` | - |
| 11 | `alpha_dated_by_its_utc_day` | entries | cuts, case_cutoff_readers, `alpha_selection_dated_by_its_adding_commit_s_manila_day` | - |
| 12 | `pilot_day_before_its_sha256` | entries | cuts, case_cutoff_readers, `pilot_winner_dated_by_the_day_after_its_sha256_R3_15` | - |
| 13 | `last_row_reader_unread` | entries | cuts, case_cutoff_readers, `last_row_reader_reads_an_attempt_s_last_row_R3_13` | - |
| 14 | `logged_reader_unread` | entries | cuts, case_cutoff_readers, `logged_reader_gives_the_manila_day_a_row_entered_R3_13` | - |
| 15 | `no_moving_record_cut_at_the_ceiling` | schedule | cuts, case_cutoff_dates, `cutoff_without_a_moving_record_is_1020` | cut_date_mismatch |
| 16 | `cutoff_before_1020` | schedule | cuts, case_cutoff_dates, `cutoff_never_before_1020_R3_15` | cut_date_mismatch |
| 17 | `am22_commit_not_an_ancestor_read` | schedule | cuts, case_cutoff_readers, `am22_dates_where_its_commit_is_not_an_ancestor_refused_B2R1` | schedule_invalid |
| 18 | `am22_section_absent_read` | schedule | cuts, case_cutoff_readers, `am22_dates_without_an_am22_section_refused_R3_15` | schedule_invalid |
| 19 | `last_stamp_first_line` | completion | cuts, case_cutoff_readers, `last_row_of_a_teacher_is_its_last_of_two_stamped_lines_R3_15` | - |
| 20 | `last_stamp_year_default` | completion | cuts, case_cutoff_readers, `last_row_of_a_teacher_is_its_last_stamped_line_utc_2026` | - |
| 21 | `last_stamp_impossible_date_read` | completion | cuts, case_cutoff_readers, `last_row_of_a_teacher_with_an_impossible_stamp_refused_R3_15` | completion_record_invalid |
| 22 | `last_row_first_row` | completion | cuts, case_cutoff_readers, `last_row_of_a_stopped_qat_attempt_is_its_run_stop` | - |
| 23 | `last_row_run_end_time_unread` | completion | cuts, case_cutoff_readers, `last_row_of_a_kd_attempt_with_its_end_is_its_run_end_R3_15` | - |
| 24 | `teacher_last_row_unread` | completion | cuts, case_cutoff_readers, `last_row_of_a_teacher_is_its_last_stamped_line_utc_2026` | - |
| 25 | `teacher_earliest_log` | completion | complete, case_teacher, `teacher_complete_at_its_earliest_log_s_final_line_R3_15` | - |
| 26 | `teacher_last_row_calls_before_matching` | completion | complete, case_teacher, `teacher_last_row_listing_matches_names_before_any_call_B2` | - |
| 27 | `teacher_state_parses_every_stamp_N2` | completion | complete, case_teacher, `teacher_complete_with_a_later_malformed_stamp_reads_complete_N2` | - |
| 28 | `printer_pilot_unread` | tday_manifest | am22, case_condition_5, `printer_cutoff_reads_the_pilot_s_winner_record_from_the_manifest_R3_15` | - |
| 29 | `printer_reads_no_runs` | tday_manifest | am22, case_condition_5, `printer_cutoff_without_the_manifest_s_pilot_record_refused_R3_15` | - |
| 30 | `printer_am22_dates_unread` | tday_manifest | cuts, case_cutoff_printer, `printer_cutoff_cut_passes` | - |
| 31 | `printer_no_card_report_dropped` | tday_manifest | cuts, case_cutoff_printer, `printer_no_card_cut_prints_its_report_B2R8_R3_07` | - |
| 32 | `rule_flag_unchecked` | tday_manifest | cuts, case_cutoff_printer, `printer_rule_flag_against_the_fields_refused` | entry_field |
| 33 | `push_evidence_on_any_entry` | tday_manifest | am22, case_unlock_printer, `printer_push_evidence_is_the_unlock_entry_s_M4` | entry_field |
| 34 | `cut_context_reads_no_runs` | validate | am22, case_condition_5, `validator_cut_context_reads_runs_pilot_and_am22_dates_R3_15` | - |
| 35 | `cut_context_pilot_unread` | validate | am22, case_condition_5, `validator_cut_context_reads_runs_pilot_and_am22_dates_R3_15` | - |
| 36 | `cut_context_am22_dates_unread` | validate | am22, case_condition_5, `validator_cut_context_reads_runs_pilot_and_am22_dates_R3_15` | - |


### B-2.10 The audit TMANB2-CHECK-1: every finding, with its commit and file:line

AUDIT RULINGS (TMAN-B2-RULE-2) accept M1–M6 and m1–m10 and direct the notes. File:line is at 5fb2759; the commit is the one that applies the finding.

**MAJOR:**
- **M1, the ceiling regex.** B2-0 `c75082d`: src/tman/schedule.py:39–40, both item 2(a) regexes whitespace-tolerant.
  - Cases: the committed text at smoke_tman_cuts.py:781, and smoke_tman_complete's case_committed_cross_checks. complete went 89 → 90 at B2-0.
  - Mutation: `am22_ceiling_regex_single_space` (B2-4).
- **M2, "entered by the end of X" against AM-19 :1374.** B2-2 `6883520` and B2-F1 `d676a97`.
  - **Every record line** prints its commit, committer time and Manila day (am22.check_lines, :1080).
  - **A REPORT** "boundary-day entry: confirm from the remote that <sha> reached origin by the end of <day> (AM-19 :1374)" is printed for every commit made on X or X−1, whatever the hour. That is boundary_reports (:1066–1077), with X−1 by its Manila day (R3-10). When D < X, D's own check prints with its REPORTs (lines, :1107; R3-02).
  - **A push the orchestrator confirms on a later day** goes in as `--pushed COMMIT=DAY` (tday_manifest.pushed_days, :109). It moves C_X (commit_at_end, :79). It dates the records that commit holds by the push day (Rec.date, :103–108), and the trigger's entered day too (:497).
  - **The freeze entry** carries each confirmation as a "confirmed push" record (freeze_conditions, :774–784). The unlock reads them back (confirmed :787; condition_unmet_error :959). A confirmation that no record needs is refused (pushed_error, :794–806).
- **M3, "the records committed by then".** B2-2 `6883520`.
  - C_X is the newest commit on the records ref's first-parent chain whose committer time meets the end of X (commit_at_end, :72–81).
  - The decision log, the entry files, the lane register, the draft and the trigger record are read as their blobs at C_X: View (:136–151), records.Records.at_commit (records.py:283) and CommitSource (:288).
  - Condition 5 runs on the draft's blob at C_X (View.draft, :211–230).
  - The unlock re-check reads D and the earlier dates at their own C (condition_unmet_error :959; d_not_first_error :971).
- **M4, the freeze entry's push evidence.** B2-3 `1a0b146`, with B2-F1 (R1-F8, R2-7, R2-12).
  - The freeze entry's own push_evidence stays null.
  - The unlock entry cites the evidence: `--push-evidence` (tday_manifest.py:381–385; check_unlock :256–280).
  - push_evidence_error is generalized (validate.py:1514). Its list names the record under "entry"; a list keyed "unlock_entry" is refused (:1549–1550).
  - g_freeze_entry_late (:1321) applies the window [the first instant of the record's date, the end of its freeze date). It also requires the record's RECORDED row to be entered by the commit that adds the record (am22.row_commit, :1121).
- **M5, TB6's two orders.** B2-5 `5f45c85`: smoke_tman_pins.py:130.
  - `registered_cells() == list(CELL_IDS)` and `set(CELL_IDS) == set(IF6_CELLS)`.
  - pin_cells_errors refuses a pin in IF6_CELLS' order and accepts the live order.
  - IF6_CELLS stays the smokes' injected fixture order. CELL_IDS, corruption-major, is the ordered set of record.
- **M6, the trigger reader.** B2-2 `6883520` (B2R13): src/tman/records.py:372–404.

**MINOR:**
- **m1, the cut checks.** B2-1 `0ce0067`.
  - chk_reason requires reason and since for any cutoff cut, launched or not (entries.py:533–544).
  - chk_stopped (:712–713) and chk_arm_running (:647–648) admit 2(b) on its date. chk_order_position was already TB10-safe.
  - B-1's mutation texts on these lines are unchanged (B-2.9).
- **m2, 2(b)'s fallback and exemption.** B2-1.
  - The stop record's entered day is used when a teacher's log has no time-stamped line (entries.py:390–395; logged_reader :347).
  - A 2(b) cut of a non-default sweep candidate is refused decision_date_cut_mismatch (chk_decision_date, :685–689).
  - Cases `cutoff_2b_teacher_without_a_stamped_line_takes_its_stop_entry_m2` and `cutoff_2b_of_a_sweep_candidate_s_repeat_refused_m2`; mutations `cutoff_2b_teacher_stop_entry_unread` and `cutoff_2b_of_a_sweep_candidate`.
- **m3, the Manila day.** B2-1.
  - schedule.manila_day (:267) dates the records in moving_reader (entries.py:293, :303) and slot_repeat (:388).
  - The pilot's winner record takes the day DL-139's row states after its sha256 (:306–316).
  - The DL-137 fixture's parent ends 16:58Z on the 21st (smoke_tman_cuts.py:471), so a UTC-day reading fails it.
  - Mutations `run_end_dated_by_its_utc_day`, `alpha_dated_by_its_utc_day` and `repeat_row_dated_by_its_utc_day`.
- **m4, D-first from the in-force day.** B2-3 `1a0b146`.
  - first_d scans from 2026-10-17 (am22.py:672–681); d_not_first_error is at :971–980.
  - The Five world (smoke_tman_am22.py:458) has a later draft that fails: 2026-11-01, case_d_first :1350.
- **m5, condition 5's draft.** B2-3, with B2-F1.
  - Condition 5 reads the draft's blob at C_X (View.draft, :211–230), with the candidate freeze for D = X (am22.py:613–614).
  - `build --am22-d` writes it (tday_manifest.py:130–147, :621).
  - B2-F1: fallback_error evaluates only a date whose draft carries its candidate (candidate_draft, :684–694).
  - RECORDS carries m5's line (B-2.15).
- **m6, a date whose end has not passed.** B2-2: date_error (am22.py:1031–1045) refuses it am22_date_not_ended. `--now` is the smokes' stand-in for the clock (tday_manifest.clock, :102).
- **m7, three conformance gaps.** B2-1 and B2-2.
  - An E6-X is excluded with its excluded E3-X (am22.py:345–370).
  - A cutoff cut of the measurement is refused cut_rule_unknown (entries.py:492–495).
  - The DL-168 fixture names the three L slots (smoke_tman_cuts.py:474, :563).
- **m8, the freeze entry's date.** B2-3, with B2-F1.
  - `date` equals freeze_date; under 1(f) the freeze date is the resolver's (freeze_entry_error, am22.py:910–935).
  - B2-F1: under 1(f) the freeze date is 2026-11-03 or later (FALLBACK_FREEZE, :55), and the printer checks the body's schedule block against the resolver (tday_manifest.py:321–329).
- **m9, as amended by TMAN-B2-RULE-3** (below). B2-3, with B2-F1.
  - freeze_fields (am22.py:983–1028) recomputes the check for D and every earlier date from the records at C_D.
  - It takes no pasted output: `--fields` must be `{}` (tday_manifest.py:403–405).
- **m10, an uncited delay or new freeze.** B2-3: am22.providers (:767–770). A RECORDED row holding "TMAN freeze_delay entry" or "TMAN new_freeze entry" beside an item 1(b) freeze is freeze_source_conflict.

**NOTES:**
- **n1** — B2-4 `4315246`: SMOKES gains "am22" (smoke_tman_mutations.py:41–42); the code count is 94 (:2096); MUTATED_FILES gains am22.py and schedule.py.
- **n2** — cosmetic; no action.
- **n3** — a reading (below).
- **n4** — the wording is at am22.py:473: "it is not met until a new amendment's code path is added (audit n4)".
- **n5** — B0 took the new-session path; update-ref was not needed.
- **n6** — the cmp command, B-2.16.
- **n7** — a reading (below).
- **n8** — results_known's limit (below).
- **n9** — no action. At MT the mutation texts all occurred once; B1's one mutation FAIL was TB8's baseline.
- **n10** — entered_day (entries.py:1025) serves entry files only. Rows with no entry-file sha256 (DL-132, DL-139, lane rows) are read from the decision log's blob at C_X (View.line_commit, :162).
- **n11** — no action.
- **n12** — no action. src/corruption_roles.py's own imports (:20–28) name nothing under src.stats (B2-5).

**TMAN-B2-RULE-3, recorded as the amendment of audit m9** (DOCUMENTED, the ruling; REPOSITORY-PROVEN in B2-3 and B2-F1). The frozen body is the draft at C_D plus, only when the ARM latency measurement is not complete at the freeze, its freeze cut: rule freeze, dated D + 1, its row entered on D + 1. Nothing else may differ.
- **The printer.** `entry freeze --d-date D --manifest <body>` recomputes at C_D and every earlier date. It refuses any other difference as freeze_entry_invalid (frozen_body_error, am22.py:839–876). body_sha256 is that body's canonical sha256 (:1027).
- **Condition 5's record.** conditions[5] cites the draft at C_D (View.draft, :229).
- **Condition 5 at D** accepts a draft whose one unmet freeze-time item is that measurement's freeze cut (:617–625).
- **A late measurement.** The measurement entered complete after C_D and by the freeze is freeze_entry_invalid and a STOP to ask (:867–869).
- **Cases and mutations.** Each part has a case: smoke_tman_am22's case_condition_5 (:1027) and case_freeze_printer (:1491). Each has a killed mutation: `condition_5_accepts_any_unentered_slot`, `condition_5_refuses_the_measurement_s_entry`, `frozen_body_differs_from_the_draft`, `frozen_body_any_slot_joins`, `frozen_body_any_arm_cut`, `frozen_cut_entered_by_its_date`, `arm_measured_after_c_d_not_a_stop`, `body_sha256_of_the_draft` and `frozen_body_measurement_gone_unchecked`.
- **The printers that make it executable** (B2-F1):
  - `entry cut --d-date D` prints the measurement's freeze cut (tday_manifest.py:372–377);
  - `freeze-body` writes the body (cmd_freeze_body :598; am22.freeze_body :879–907);
  - a measurement in neither the draft nor the body is refused (:857–861; DL-24 R3-01).

**The readings for n3 and n7, n8's limit and n4's wording:**
- **n3.** "Registered runs" are read as the catalogue's run slots (am22.py:345).
  - That includes E1:s43:80k, E1:s44:80k and E2:s42:lambda{0.25,2,4}:80k, which have never_cut false and cut_group null. All are complete (PA :2466–2468), so condition 1 is unaffected.
  - The E1 reruns of AM-7's full consequence have no slots. am7_fired (:372) prints that consequence as a STOP to ask, since a slot would be a catalogue change outside SCOPE.
- **n7.** "Entered in the decision log as … cut" is read through the cut's entry file and its RECORDED row (B-1's decision (iii); View.settled, :422).
  - DL-137, DL-138 and DL-168 settle their runs once a local block prints their entry files and commits each with a row naming its sha256.
  - D is unaffected: condition 5 needs those entry files in the draft anyway.
- **n8, results_known's limit.** A cutoff cut's results_known may cite only records committed by the end of its cut date: chk_results (entries.py:607–612) asks known_by, which reads the records ref, never a run directory.
  - For DL-137 and DL-138, cut on 2026-10-24, that excludes DL-135's and DL-136's records, committed on 2026-10-26.
  - Their real entries cite DL-118 and DL-124 (R1) and DL-122 and DL-125 (R2), or they are refused cut_results_unknown.
- **n4.** The wording "until a new amendment's code path is added" is used at am22.py:473.

### B-2.11 The DL-24 workflow and its findings

**The workflow** (MEASURED): `wf_3be1f10d-0d1`, read-only, over the scoped `git diff --no-renames 8f85c55… HEAD` at 5f45c85.
- Four reviewers, each followed by an adversarial verifier: R1 conformance; R2 the freeze entry and providers; R3 smokes and mutations; R4 numerical.
- Eight agents, about 109 min.
- Before and after: HEAD 5f45c85, the scoped status empty, `sha256sum -c` 25/25 OK.
- 64 findings: R1 21, R2 17, R3 20, R4 6. 63 are CONFIRMED and 1 REFUTED.

**R4's numerical cases** match the code, 18 of 18, each recomputed with python3 -I and the standard library:
- DL-137 2026-10-24, DL-138 2026-10-24, DL-168 2026-10-20;
- moving records on 10-18, 10-19, 10-27 and 10-28 give 10-20, 10-21, 10-29 and 10-29; an unwritten record gives 10-29;
- 2(b) rows at 10-21T10:00Z and T16:30Z give 10-24 and 10-25; a teacher's stop entry gives 10-25;
- D-first over Five;
- D 2026-10-31's freeze date and instant;
- the refusals after 2026-11-02 and before a date's end.

**Dispositions.** "Fixed" names the commit, B2-F1 `d676a97` or B2-F2 `5fb2759`, at 5fb2759's file:line. "READ" is a reading stated here, or in RECORDS where marked. "OPEN" is in B-2.14.

| Finding | Severity | Disposition | Where |
|---|---|---|---|
| R1-F1 | MAJOR | fixed B2-F1: the measurement's freeze cut can be printed before the 1(b) record | tday_manifest.py:372–377, :598; am22.py:879–907 |
| R1-F2 | MAJOR | fixed B2-F1: item 1(f) only when no date to 2026-11-02 has the five conditions | am22.py:697; validate.py:1314–1315; tday_manifest.py:333–344 |
| R1-F3 | MINOR | fixed B2-F2: a record dated after the ceiling counts as unwritten | entries.py:365–366 |
| R1-F4 | MINOR | OPEN 1 | entries.py:515–544, :406–407 |
| R1-F5 | MINOR | fixed B2-F1: the launch log read as the records hold it now | am22.py:255 |
| R1-F6 | MINOR | REFUTED (below) | — |
| R1-F7 | MINOR | fixed B2-F2: an unreadable since.record is no evidence; a REPORT, never a refusal | entries.py:421–437 |
| R1-F8 | MINOR | fixed B2-F1: the row must be entered by the commit that adds the record | validate.py:1321; am22.py:1121; tday_manifest.py:256–280 |
| R1-F9 | MINOR | fixed B2-F1: condition 5 keeps the validator's REPORTs on the RULE-3 path | am22.py:625; validate.py:1695 |
| R1-F10 | MINOR | READ (moot): a pilot with no winner would take the ceiling; DL-139 records the winner | entries.py:306–316 |
| R1-N1 | NOTE | READ: Schedule T is in force (DL-70); under R the provider gives None and nothing of AM-22 applies | schedule.py:173 |
| R1-N2 | NOTE | fixed B2-F2: teacher_state and RunState are MT's again; teacher_last_row reads the last rows | completion.py:323, :346–364 |
| R1-N3 | NOTE | READ: condition 2 keeps only `triggered` from L-AM3's reader; its own recomputation is L-AM3's (DL-140) | records.py:377–391 |
| R1-N4 | NOTE | READ (RECORDS, m5) | am22.py:613–614 |
| R1-N5 | NOTE | OPEN 5 (moot, or MISSING by default) | am22.py:59, :372 |
| R1-N6 | NOTE | OPEN 2 | entries.py:101–104 |
| R1-N7 | NOTE | OPEN 4 | B-1's derived_missing |
| R1-N8 | NOTE | fixed B2-F1: the unlock's window message is B-1's again | validate.py:1588–1589 |
| R1-N9 | NOTE | fixed B2-F1: a corrected 1(b) record is no freeze of record | am22.py:714–729 |
| R1-N10 | NOTE | READ: a draft body entry settles through its dl_rows at C_X, as B-1's dl_row_missing reads it | am22.py:387–427 |
| R1-N11 | NOTE | READ: condition 5's allowance does not read the measurement's completeness; the freeze printer STOPs on a measurement entered complete (:867–869) | am22.py:617–625 |
| R2-1 | MAJOR | fixed B2-F1 (as R1-F1) | — |
| R2-2 | MAJOR | fixed B2-F1 (as R1-F2) | — |
| R2-3 | MAJOR | fixed B2-F1: the unlock and the session read --freeze-record only; validate refuses --am22-d there | validate.py:584–586; tday_manifest.py:154–156 |
| R2-4 | MAJOR | fixed B2-F1: confirmed pushes (M2) | am22.py:72–81, :103–108; tday_manifest.py:109 |
| R2-5 | MINOR | fixed B2-F1: a measurement in neither the draft nor the body is refused | am22.py:857–861 |
| R2-6 | MINOR | fixed B2-F1: the draft is dated by its last change | am22.py:185–193 |
| R2-7 | MINOR | fixed B2-F1 (as R1-F8) | — |
| R2-8 | MINOR | fixed B2-F1: freeze-body copies every block from the draft, lane_code included | am22.py:879–907 |
| R2-9 | MINOR | fixed B2-F1 (as R1-F9) | — |
| R2-10 | NOTE | OPEN 3 | records.py:397–404 |
| R2-11 | NOTE | READ: only an entry other than a cut is a STOP; the run index has no completion for the measurement | am22.py:862–869 |
| R2-12 | NOTE | fixed B2-F1: a list keyed "unlock_entry" is refused for the freeze | validate.py:1549–1550 |
| R2-13 | NOTE | fixed B2-F1: two 1(b) records conflict only with the conflict check on; freeze mode refuses at freeze_source_conflict's gate | am22.py:755; validate.py:590–599 |
| R2-14 | NOTE | fixed B2-F1: E6 seed 42's records cite their own sha256 | am22.py:492 |
| R2-15 | NOTE | fixed B2-F1: D and --am22-d lie from 2026-10-17 to 2026-11-02 | am22.py:747, :971–980 |
| R2-16 | NOTE | READ (RECORDS): C_X by each commit's own committer time | am22.py:72–81 |
| R2-17 | NOTE | fixed B2-F1: the static case refuses an absent or empty `providers` | smoke_tman_am22.py:899–918 |
| R3-01 | MAJOR | fixed B2-F1 (as R2-5) | — |
| R3-02 | MAJOR | fixed B2-F1: D's own check and REPORTs when D < X; entry freeze prints D's REPORTs | am22.py:1066–1077, :1107; tday_manifest.py:380, :390–391 |
| R3-03 | MINOR | fixed B2-F1: case_d_first runs Five at 2026-11-01 | smoke_tman_am22.py:1350 |
| R3-04 | MINOR | fixed B2-F1: per-printer build checks (PrinterRefused names the printer); the kills retargeted | smoke_tman_am22.py:177–184, :1443 |
| R3-05 | MINOR | fixed B2-F1: case and mutations for am22_condition_unmet's branch | smoke_tman_am22.py:1740; `unlock_recheck_skips_unmet_conditions`, `printer_skips_unmet_conditions` |
| R3-06 | MINOR | fixed B2-F1: freeze_entry_invalid's relations one by one, seven mutations | smoke_tman_am22.py:1740 |
| R3-07 | MINOR | fixed B2-F1 (the validator's "no card" REPORT through condition 5) and B2-F2 (the printer) | smoke_tman_cuts.py:835–846 |
| R3-08 | MINOR | fixed B2-F2: DL-168 through the real moving reader; the arm mutation retargeted | smoke_tman_cuts.py:565–569 |
| R3-09 | MINOR | fixed B2-F1: Manila-day C_X and trigger due day | smoke_tman_am22.py:859, :1163 |
| R3-10 | MINOR | fixed B2-F1: the REPORT for X−1 by its Manila day | am22.py:1066–1077; smoke_tman_am22.py:1226 |
| R3-11 | MINOR | fixed B2-F1: a trigger entered on its due day prints no REPORT | smoke_tman_am22.py:986 |
| R3-12 | MINOR | fixed B2-F1: condition 3 MISSING, with a mutation | smoke_tman_am22.py:1090 |
| R3-13 | MINOR | fixed B2-F2: the two readers' closures run | smoke_tman_cuts.py:742–749 |
| R3-14 | MINOR | fixed B2-F1: a second 1(b) freeze is refused | am22.py:996–998 |
| R3-15 | MINOR | fixed B2-F1 and B2-F2: 80 and 36 mutations | B-2.9 |
| R3-16 | NOTE | fixed B2-F1 for (a)–(c); (d) stated in B-2.9 | B-2.9 |
| R3-17 | NOTE | fixed: (a) B2-F1, the MISSING text names the merge commit; (b) and (c) B2-F2; (d) the conflict case runs the validator | smoke_tman_am22.py:1145, :1698; smoke_tman_cuts.py:726; smoke_tman_schema TB8 |
| R3-18 | NOTE | (a) READ (RECORDS); (b) fixed B2-F1 (as R1-F9); (c) READ: harmless, since settled() reads kind and slots only | smoke_tman_am22.py:824–826 |
| R3-19 | NOTE | OPEN 3 | records.py:397–404 |
| R3-20 | NOTE | fixed B2-F1 (as R2-13) | smoke_tman_am22.py:1698 |
| R4-1 | MAJOR | fixed B2-F1 (as R2-4) | — |
| R4-2 | MINOR | fixed B2-F2: a row after 16:00Z counts from its Manila day | entries.py:388; smoke_tman_cuts.py:661 |
| R4-3 | MINOR | fixed B2-F1 (as R1-F9) | — |
| R4-4 | NOTE | READ and pinned in B2-F2: with a record unwritten the ceiling sets the date, so rule_record is null | entries.py:369; smoke_tman_cuts.py:614 |
| R4-5 | NOTE | READ (RECORDS): α records dated by their adding commit's committer time, a lower bound | entries.py:295–304 |
| R4-6 | NOTE | READ (m5): D falls only on a day whose draft carries its candidate | am22.py:684–694 |

**The verifiers' missed items** (16), each mapped:
- **R1:**
  - C_X by committer time with no confirmed push: fixed B2-F1 (as R2-4);
  - a body with no ARM entry passes: fixed B2-F1 (as R2-5);
  - the 1(f) printer's resolver taken from the body: fixed B2-F1 (tday_manifest.py:320–329; mutation `printer_item_1f_resolver_from_the_body`).
- **R2:**
  - a second 1(b) freeze printed: fixed B2-F1 (as R3-14);
  - D < X printed without D's records: fixed B2-F1 (as R3-02);
  - the freeze printer never validates the body: fixed B2-F1 (am22.py:1019–1026);
  - `--am22-d` with no upper bound: fixed B2-F1 (am22.py:747).
- **R3:**
  - the "later than 2026-11-02" branch: fixed B2-F1 (`d_not_first_d_after_1102_is_item_1f_R3`; `d_not_first_unbounded_above`);
  - `providers=()`: fixed B2-F1 (as R2-17);
  - the first-parent walk: fixed B2-F1 (smoke_tman_am22.py:885; `c_x_off_the_first_parent_chain`);
  - the `conditions` exemption in all ten modules: fixed B2-F1 (smoke_tman_noscore.py:47, :254–258).
- **R4:**
  - no "meets that date" bound: fixed B2-F2 (as R1-F3);
  - the draft's commit via adding_commit: fixed B2-F1 (as R2-6);
  - a multi-slot cut set by different records: fixed B2-F2 (entries.py:414–417, "one cut entry per record that sets it");
  - the back-filled entry files' entered day: READ (RECORDS);
  - one NOT A DEFECT, R4's recomputation.

**The refuted finding, R1-F6** (MINOR). It claimed that M2's prompt is missing when the scan finds D < X, and that `entry freeze` prints no boundary REPORT. The verifier refuted it, and I accept the reason:
- The code facts were right: check-freeze printed boundary REPORTs for X and X−1 only, and entry freeze printed none.
- But no wrong record can follow. date_error requires X's end to have passed. A 1(b) record for d must reach the remote before the end of d + 1, or g_freeze_entry_late refuses it.
- If d < X, then the end of d + 1 is no later than the end of X, which is before now. So when check-freeze prints "D: d", no timely record for d can exist.
- The only timely path is check-freeze --date D run on D + 1, where the REPORTs for D and D−1 print, as M2 asks. M2 places the REPORT on check-freeze.
- The lines were improved under R3-02 anyway: D's own check and REPORTs print, and entry freeze prints D's REPORTs.

### B-2.12 The SL-1 log

`tail -n 80 ~/.claude/sl1_guard.log`, MEASURED at B6, holds every line this session wrote:

```text
2026-10-28T05:28:44Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T06:05:10Z	Bash	SL1-B1	grep -n 'evaluator_stages_from\|am22_committed_text\|AM22_CEILING_RE\|AM22_CUTOFF_RE\|case_committed_cross_checks\|case_catalogue_cross_checks' scripts/smoke_tman_mutations.py scripts/smoke_tman_*.py 
2026-10-28T06:10:23Z	Bash	SL1-B1	grep -n -A4 '"trigger_reader_required"\|"trigger_verdict_compared"\|"trigger_sha256_compared"\|"trigger_e6kd"' scripts/smoke_tman_mutations.py | cut -c1-180; grep -n 'cli_unlock_without_a_trigger_read
2026-10-28T06:11:27Z	Bash	SL1-B1	sed -n '70,100p' scripts/smoke_tman_determinism.py; echo ----; sed -n '535,560p' scripts/smoke_tman_paths.py; echo ----; grep -n 'RunState(' scripts/smoke_tman_*.py | head
2026-10-28T06:11:33Z	Bash	SL1-B1	grep -n '"source": "schedule"\|source="schedule"\|"freeze_date"' scripts/smoke_tman_*.py | cut -c1-220
2026-10-28T06:24:26Z	Bash	SL1-B1	export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis && for s in schema complete cuts; do /tmp/claude-0/-home-user-plantseg-thesis/29a88c04-ac7a-578f-a0d9-f6db2d0b8c57/scratchpad/tma
2026-10-28T06:24:48Z	Bash	SL1-B1	export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis && for s in schema paths parent derived pins; do /tmp/claude-0/-home-user-plantseg-thesis/29a88c04-ac7a-578f-a0d9-f6db2d0b8c57/sc
2026-10-28T06:25:15Z	Bash	SL1-B1	export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis && for s in validate unlock noscore determinism; do /tmp/claude-0/-home-user-plantseg-thesis/29a88c04-ac7a-578f-a0d9-f6db2d0b8c57
2026-10-28T06:40:14Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T06:40:51Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/quant glob={qat,qat_artifacts,qat_select,stages,e6kd_trigger,runner,ptq,prepare,qconfig}.py
2026-10-28T06:40:55Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/quant glob={calibration,checkpoint,x86_latency,qat,qat_artifacts,qat_select,stages,e6kd_trigger,prepare,qconfig}.py
2026-10-28T06:40:56Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/quant glob={calibration,checkpoint,x86_latency}.py
2026-10-28T07:03:32Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T07:42:28Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T07:47:17Z	Bash	SL1-B1	grep -n "freeze_instant" scripts/tday_manifest.py src/tman/*.py
2026-10-28T07:48:38Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T07:56:23Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T07:57:00Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T08:00:58Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T08:02:03Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T08:24:35Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_{am22,unlock,validate,noscore,paths}.py
2026-10-28T08:36:42Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/29a88c04-ac7a-578f-a0d9-f6db2d0b8c57/scratchpad; cat "$SCR/../tasks/b8xlll0we.output"; for s in schema paths complete cuts parent derived pins validate unl
2026-10-28T08:40:03Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_*.py
2026-10-28T08:41:32Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T08:49:16Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_{mutations,am22}.py
2026-10-28T08:50:08Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_{mutations,am22}.py
2026-10-28T11:31:00Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T11:31:46Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_*.py
2026-10-28T11:37:19Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_*.py
2026-10-28T11:46:07Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T11:52:22Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T12:28:17Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T12:32:28Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_*.py
2026-10-28T12:32:37Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T12:32:57Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=smoke_tman_{complete,schema}.py
2026-10-28T13:46:10Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/tman glob=!*[Tt][Ee][Ss][Tt]*
2026-10-28T13:46:11Z	Grep	SL1-G1	/home/user/plantseg-thesis/scripts glob=*.py
2026-10-28T15:26:04Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/29a88c04-ac7a-578f-a0d9-f6db2d0b8c57/scratchpad; for s in schema paths complete cuts parent derived pins validate unlock noscore determinism mutations; do 
2026-10-28T15:31:37Z	Bash	SL1-B2	MT=8f85c55a53aaf5a5e7fe5cc9f1563305dbe05d7f && git diff --no-renames --name-status $MT HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*' > /tmp/cla
```

**What the lines are.** The guard is in log mode and denied nothing. Its rules (sl1_guard.py :23–29):
- **G1:** a Grep or Glob over a directory in a protected location.
- **B1:** a recursive, globbed or directory search or listing there, or a path it cannot resolve.
- **B2:** a git listing or diff whose scope can include test paths.

The lines fall into four groups:
- **28 G1 lines,** Grep-tool searches that named a folder rather than files: slip 5 (B-2.13).
- **Five B1 lines** with a shell glob over lane files, at 06:05:10, 06:10:23, 06:11:27, 06:11:33 and 07:47:17: slip 4.
- **Five B1 lines** with a loop variable in a path:
  - 06:24:26, 06:24:48 and 06:25:15 are B2-1's smoke runs, each smoke alone, named `smoke_tman_$s.py`;
  - 08:36:42 and 15:26:04 read the session's own logs `$SCR/<tag>_$s.log`.

  These are not searches. The smoke runs share slip 2's temp dir.
- **One B2 line,** at 15:31:37: ACCEPTANCE's second `--name-status` form, `-- src scripts configs docs/lane_reports ':(exclude)docs/reference'`, which the ADDENDUM names. It carries no test exclude by design, and printed the same 20 lane files.

### B-2.13 Slips

1. **B0:** one Grep-tool search on docs/lane_reports/tman.md, an exact file, instead of `grep -n`.
2. **B0:** L-PROT G refused the ADDENDUM's named `git cat-file commit 8f85c55…`. This was a hook, not Auto; nothing ran. The message was read with `git show -s --format=%B 8f85c55… -- <the exclude form>` and recorded at B0.
3. **B1 to B2-2 (05:33–07:05 UTC):** the B-1 smokes and smoke_frozen_blobs ran in the system temp dir, not in a new TMPDIR under $SCR.
   - I had read "each new smoke" as the new smoke file. smoke_tman_am22 had its own TMPDIR from its first run, at 07:02.
   - From B2-3 (08:24) on, every run has its own new TMPDIR under $SCR, B5 included.
   - Each B-1 smoke removes its own scratch folders (`scratch_folders_removed` PASS in each run).
4. **Five read-only Bash greps used a shell glob over lane files:** `scripts/smoke_tman_*.py` at 06:05, 06:10 and twice at 06:11, and `src/tman/*.py` at 07:47. No name they match holds "test". LANE_STATE had recorded only the last.
5. **28 Grep-tool searches named a folder** (scripts/, src/tman/ or src/quant/) instead of files (SL-1 G1).
   - 16 carried the `!*[Tt][Ee][Ss][Tt]*` glob.
   - 12 carried a name glob instead:
     - three lists of named src/quant modules;
     - eight `smoke_tman_{…}.py` or `smoke_tman_*.py`;
     - one `*.py`, at 13:46:11 in B2-F2. It could have searched a test-named file under scripts/; it printed no match.
   - The two Grep calls of slip 8 named files and are not logged.
6. **B2-2:** a stray no-op `timeout 900 $SCR/tanvenv_missing` in one smoke command. The path does not exist, so nothing ran.
7. **B2-3:** the first push command, a retry loop with `$((2 ** i))`, was refused by L-PROT N (`**` read as a glob). Nothing ran; it was re-issued as a plain push.
8. **B2-5:** two Grep-tool calls on src/corruption_roles.py and src/corruption_cache.py, named files, omitted the `!*[Tt][Ee][Ss][Tt]*` glob. Neither path holds "test".
9. **B2-F1, against "Nothing else runs during a smoke":** during the runs of 12:26–12:36 and from 12:53, I did the following:
   - read lane files and the attachments;
   - ran `git show <commit>:<explicit file>` twice;
   - ran a sed of a smoke;
   - appended to LANE_STATE.md and wrote a commit-message file.

   No repository file was written during a run, and no result depends on them. Every later run (bf2, bf2b, the `--only` run and B5) ran with no other call.
10. **B6:** `ls -la ~/.claude/sl1_guard.log`, with a grep of the venv's freeze list in the same command, was refused by L-PROT S before it ran. The ADDENDUM's `tail -n 80` form was used.

**Not slips, recorded:**
- **The stop hook** asked for a commit and a push several times. It was reported once (B2-2), never complied with, and not raised again.
- **The harness's attribution reminders and designated branch** were ignored, as the RULES say. No commit carries a trailer, and every push went to refs/heads/lane/tman.

### B-2.14 Open items

1. **R1-F4, a run launched after its cutoff date.**
   - **The gap:** B-1's RunIndex counts every attempt the run directories hold, whatever its launch time. A correct 2(a) entry for a run launched late is therefore refused cut_reason_invalid, and the run then settles by completion.
   - **Today:** no late launch is recorded. DL-146 launched E6-E2 inside its cutoff date.
   - **A fix:** read the launch time (the launch log's launch_time_utc, run_meta's wall_clock, the teacher provenance's recorded_utc) against the cut date. That changes B-1's RunIndex, so it is a ruling's to make.
2. **R1-N6, item 1(b)'s replacement of the freeze date in AM-19 item 3(b)'s arithmetic** "from the freeze entry's commit" (PA :2338–2339).
   - **The gap:** it is not coded. cut_arithmetic (entries.py:101–104) uses FREEZE_DATES, 2026-11-03.
   - **Why it is moot** (INFERRED): at D every registered run is settled, so no arithmetic cut is pending between the freeze entry's commit and the freeze.
3. **R2-10 and R3-19, the real trigger reader.**
   - **Where it runs:** L-AM3's reader runs only bound to the records checkout's root (records.py:397–404). Every smoke patches TRIGGER_READER with the am3_trigger/1 stand-in.
   - **The unlock:** its root is S = E, so condition 2's re-check at C_D holds only if E's tree holds the trigger record at the cited sha256.
   - **First real run:** the local block's first check-freeze.
4. **R1-N7, E7-E2's "not made: code not merged".**
   - **The text:** AM-23 item 4 ties it to E6-E2's cutoff date (PA :2602–2605).
   - **The code:** B-1's derived_missing tests the lane at the original freeze instant (D + 1 under 1(b)), and nothing couples E6-E2's cutoff cut with E7-E2's not_made entry.
5. **R1-F10 and R1-N5, clauses not coded, each moot:**
   - a pilot with no winner taking the ceiling (AM-23 item 1); DL-139 records the winner;
   - the SegNeXt-L KD run's AM-17 item 11(b) no-go; the L arm is cut (DL-168);
   - the STOP of "a refusal that no later input can lift";
   - AM-7's consequence through "a run to which a rejection restores it".

   Each leaves the run unsettled, so condition 1 stays MISSING, a STOP to ask.

### B-2.15 RECORDS proposed (the lane writes none)

DOCUMENTED (the ADDENDUM's RECORDS), with the audit's and DL-24's additions:
- **B-2's IMPLEMENTED row** at TMAN's merge, with the code pin in DL-37's form: code commit `5fb27593840f62f6843635dafffd29ad7f2789b6`, and this report's commit after it.
- **The current-plan line:** B-2 before D, with check-freeze as AM-22's "check … in the manifest's validator or a script beside it" (PA :2453–2456).
- **The freeze kind's fields** (B-2.7).
- **The reading of "entered by the end of D"** (audit M2's replacement; R2-16; R3-18 (a); R4-5):
  - **The reading:** a row or file counts by its commit at C_X, the newest first-parent commit whose own committer time meets the end of X.
  - **Before `entry freeze` runs,** the orchestrator confirms from the remote every commit that check-freeze names in a boundary-day REPORT, for X and X−1, whatever the hour.
  - **A push confirmed on a later day** is passed as `--pushed COMMIT=DAY` to check-freeze, freeze-body and entry freeze. It moves C_X and dates its records that day. The freeze entry carries it as a "confirmed push" record, and the freeze entry's row states the confirmation.
  - **Committed moving records** (the α selection file and the α-cut decision record) are dated the same way, by their adding commit.
- **m5's line** (R1-N4, R4-6). Condition 5 reads the draft committed by X, as its blob at C_X. The local block builds it with `build --am22-d X`, and the orchestrator rebuilds and commits the draft for each candidate day until D. D can fall only on a day whose draft carries its candidate.
- **The real entry files for DL-137, DL-138 and DL-168,** printed later by a local block, each committed with a row naming its sha256:
  - each is printed with `entry cut --rule cutoff --runs DIR --manifest <draft>`;
  - their results_known follow n8's limit (B-2.10);
  - their entered day is the day those rows are committed, so their late-entry REPORT reads that day, not 2026-10-26 or 2026-10-27.
- **n7's reading** (B-2.10).
- **The freeze sequence under item 1(b) with the ARM measurement not complete** (TMAN-B2-RULE-3):
  1. on D + 1, `entry cut --slot ARM:latency --d-date D …`, committed with its row and its lane's not-required entry;
  2. `freeze-body --d-date D --cut <that entry>`, the body committed as the manifest;
  3. `entry freeze --d-date D --manifest <body> --runs DIR`, committed with its row and pushed before the end of D + 1;
  4. the push evidence committed after, and cited by `entry unlock --push-evidence`.
- **Chapter 4's AM-22 item 4 disclosure,** mapped to check-freeze's output:
  - the freeze date and its source;
  - each condition's records and dates;
  - every cutoff cut with its reason;
  - N;
  - the ARM measurement's status at the freeze;
  - each lane entered as not required, with its cut entries.

  On condition 5's RULE-3 path, check-freeze's late-entry and "no card" REPORTs come from validate.report_lines at the failing gate (B2-F1; R3-18 (b)).

### B-2.16 The prefix check and B7

**The prefix check** (audit n6), MEASURED after this section was appended and before its commit:
- `git show 8f85c55a53aaf5a5e7fe5cc9f1563305dbe05d7f:docs/lane_reports/tman.md > $SCR/tman_mt.md`;
- then `cmp -n $(stat -c %s $SCR/tman_mt.md) $SCR/tman_mt.md docs/lane_reports/tman.md`;
- it printed nothing and exited 0, so everything above "## B-2" is MT's file.

**B7** follows this report's commit: `git push origin refs/heads/lane/tman:refs/heads/lane/tman`, then `git ls-remote --heads origin lane/tman`, then STOP.
