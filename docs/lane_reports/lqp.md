# Lane LQP · PTQ of record (L-CKPT-GUARD Q-part, L-AM16-CALIB): Phase B report

- GO: the LQP-B Phase B GO (STEPS 1–9, rulings R1–R18), on the read-only Phase A plan; ruling Q1-R2 ("old runners
  refuse real runs"). Cloud lane session (DL-37).
- O = origin/master at STEP 1 = `2b3b55651e8092fd7fff6e786b73d971096deb17` (MEASURED after `git fetch --unshallow
  origin`). H = the first merge on the ancestry path 850f9e8..origin/master = `2b3b55651e8092fd7fff6e786b73d971096deb17`
  = O, "Merge lane/q2-qat (850f9e8)" (H^{tree} f469670d…, H^1 31e12c4d…, H^2 850f9e89…, as expected; `git diff --stat
  H O -- src scripts configs` empty). The E1 safety floor `885523a47e427e84b551846d2d4a6a9723b9e53b` is an ancestor
  of O (MEASURED: `git merge-base --is-ancestor`, exit 0).
- Branch `lane/lqp`, cut from O (STEP 2). Last code commit (FINAL, the commit every file:line below is read at):
  `d38495f8bd0e390eb82acc12ea15e4de4b363ce3`. This report is the lane's last commit; STEP 9 pushes it, and `git ls-remote` then prints its SHA.
- Labels: MEASURED (run in this session), DOCUMENTED (the GO, the plan, the rulings), REPOSITORY-PROVEN (read in the
  code at FINAL), INFERRED.
- `$SCR` is this session's scratchpad, `/tmp/claude-0/-home-user-plantseg-thesis/16100988-d746-58c9-971a-7a52e9412643/scratchpad`,
  outside the repository: the run scripts, logs, LANE_STATE.md and drafts named below. Nothing in it is committed.

## 1. Commits, the scoped diff and the venv

Commits on lane/lqp after O, oldest first (MEASURED, `git log --reverse --format='%h %s' 2b3b556..d38495f`; no commit carries
a trailer):

- `2bc8943 LQP C1: the old E4/E7 runners refuse every run (Q1-R2, L-CKPT-GUARD Q-part)`
- `44fecc4 LQP C2: ptq.run_ptq refuses a non-empty out-dir (L-CKPT-GUARD Q-part)`
- `e94f2d3 LQP C3: the PTQ of record bound to its parent, and its record (L-CKPT-GUARD Q-part)`
- `d65aa90 LQP C7: the official INT8 gate requires the PTQ record of record (Q1-R2, R2)`
- `87a7021 LQP C8: the calibration range over the four lists (L-AM16-CALIB, R13)`
- `9d12b77 LQP C4: smoke_ptq_record.py, the bound path, the record gate and the range`
- `9b40867 LQP C5: smoke_ptq_record_mutations.py, each new refusal killed by a named check`
- `9cc512d LQP C9: the bound gates refuse an input they cannot read (DL-24 A-1)`
- `103dfad LQP C10: the range reporter refuses a malformed input instead of crashing (DL-24 B-1)`
- `324dddc LQP C11: runner.py's SAFETY MODEL names each guard's real users (DL-24 C-1)`
- `e60940a LQP C12: each condition of the multi-condition refusals killed alone (DL-24 D-1 to D-4, D-6)`
- `d38495f LQP C13: a harness job always returns its shadow; LQ38 cites sweep_select.py:1815 (DL-24 D-5, D-10)`

Scoped diff (MEASURED at FINAL):
```
git diff --stat 2b3b556 d38495f8bd0e390eb82acc12ea15e4de4b363ce3 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
```
```
 scripts/evaluate_model.py             |   13 +-
 scripts/ptq_calibration_range.py      |  214 +++++
 scripts/run_e4.py                     |    8 +-
 scripts/run_e7.py                     |    8 +-
 scripts/run_ptq.py                    |  128 ++-
 scripts/smoke_ptq_record.py           | 1502 +++++++++++++++++++++++++++++++++
 scripts/smoke_ptq_record_mutations.py |  681 +++++++++++++++
 scripts/smoke_quant_runners.py        |   15 +-
 scripts/smoke_run_ptq.py              |   10 +-
 src/eval/stage_artifacts.py           |   52 ++
 src/quant/ptq.py                      |    3 +
 src/quant/ptq_record.py               |  562 ++++++++++++
 src/quant/runner.py                   |  110 +--
 13 files changed, 3171 insertions(+), 135 deletions(-)
```

The venv against requirements.lock (MEASURED, STEP 3):

- The venv (MEASURED, STEP 3): `/usr/bin/python3.11 -m venv <scratchpad>/lqpvenv`, Python 3.11.17, pip 24.0 and
  setuptools 79.0.1 from ensurepip; `pip check`: "No broken requirements found"; quantized engines qnnpack, none,
  onednn, x86, fbgemm.
- Against requirements.lock: torch 2.1.0+cpu and torchvision 0.16.0+cpu, where the lock pins +cu121 (the GO's
  CPU build, from download.pytorch.org/whl/cpu, `--no-deps`); every other installed distribution at its lock pin:
  numpy 1.26.4, scipy 1.11.4, pillow 12.3.0, statsmodels 0.14.6, patsy 1.0.2, pandas 3.0.3, python-dateutil
  2.9.0.post0, six 1.17.0, tzdata 2026.2, packaging 24.2, filelock 3.14.0, typing_extensions 4.15.0, sympy 1.14.0,
  mpmath 1.3.0, networkx 3.6.1, Jinja2 3.1.6, MarkupSafe 3.0.3, fsspec 2026.4.0, requests 2.28.2,
  charset-normalizer 3.4.7, idna 3.18, urllib3 1.26.20, certifi 2026.6.17. 25 of the lock's 76 pins are installed;
  the other 51 are not (the smokes run here pass without them), the teacher stack (mmcv 2.1.0, mmengine 0.10.7,
  mmsegmentation 1.2.2) and pytz 2023.4 among them.
- Every smoke ran with the venv's python, `-B` and `PYTHONDONTWRITEBYTECODE=1`, PYTHONPATH unset (smoke_arms_kd also
  with PYTHONPATH = the checkout, §2).

## 2. RESULT lines: the STEP 4 baseline at O and the STEP 7 final run at FINAL

MEASURED. Each smoke alone, in this order, with the venv's python `-B` (`$SCR/run_baseline.sh` at O,
`$SCR/run_final.sh` at FINAL: one smoke at a time, no other smoke or dev run alongside; from 05:02Z a 5 s
memory sampler, `free` and `ps` only, ran beside the last LQP harness run and Q2's harness, §7); logs in
`$SCR/baseline/` and `$SCR/final/`. STEP 4: 2026-10-15T23:53:05Z to 2026-10-16T00:10:54Z, HEAD `2b3b556`. STEP 7: from
2026-10-16T02:50:40Z, HEAD `d38495f8bd0e390eb82acc12ea15e4de4b363ce3` (the summary's BEGIN and ALL-DONE lines name it).

| Smoke | Expected (GO) | STEP 4 at O (rc, RESULT) | STEP 7 at FINAL (rc, RESULT) | Final run (UTC) |
|---|---|---|---|---|
| smoke_frozen_blobs | 10/10 | 0, `RESULT: FROZEN BLOBS OK (10/10)` | 0, `RESULT: FROZEN BLOBS OK (10/10)` | 02:50:40-02:50:40 |
| smoke_run_ptq | 69/69 | 0, `RESULT: PASS (69/69)` | 0, `RESULT: PASS (69/69)` | 02:50:40-02:57:23 |
| smoke_calibration_lists | 42/42 | 0, `RESULT: PASS (42/42)` | 0, `RESULT: PASS (42/42)` | 02:57:23-02:57:26 |
| smoke_qnnpack_ops | 8/8 | 0, `RESULT: PASS (8/8) synthetic artifact` | 0, `RESULT: PASS (8/8) synthetic artifact` | 02:57:26-03:00:49 |
| smoke_eval_int8 | 32/32 | 0, `RESULT: PASS (32/32)` | 0, `RESULT: PASS (32/32)` | 03:00:49-03:00:56 |
| smoke_efficiency | 82/82 | 0, `RESULT: PASS (82/82)` | 0, `RESULT: PASS (82/82)` | 03:00:56-03:01:00 |
| smoke_quant_e4_e5 | 58/58 | 0, `RESULT: PASS (58/58)` | 0, `RESULT: PASS (58/58)` | 03:01:00-03:01:04 |
| smoke_quant_e6_e7 | 37/37 | 0, `RESULT: PASS (37/37)` | 0, `RESULT: PASS (37/37)` | 03:01:04-03:01:08 |
| smoke_quant_runners | 77/77 (STEP 4); 78/78 (STEP 7) | 0, `RESULT: PASS (77/77)` | 0, `RESULT: PASS (78/78)` | 03:01:08-03:01:11 |
| smoke_realrun_decisions | 76/76 | 0, `RESULT: PASS (76/76)` | 0, `RESULT: PASS (76/76)` | 03:01:11-03:01:12 |
| smoke_qat_runner | 214/214 | 0, `RESULT: PASS (214/214)` | 0, `RESULT: PASS (214/214)` | 03:01:12-03:03:08 |
| smoke_qat_seeding | 9/9 | 0, `RESULT: PASS (9/9)` | 0, `RESULT: PASS (9/9)` | 03:03:08-03:03:38 |
| smoke_quant_x86_efficiency | 103/103 | 0, `RESULT: PASS (103/103)` | 0, `RESULT: PASS (103/103)` | 03:03:38-03:03:41 |
| smoke_eval_determinism | (none) | 0, `RESULT: EVAL-DET OK (74/74)` | 0, `RESULT: EVAL-DET OK (74/74)` | 03:03:41-03:04:04 |
| smoke_am13_upstream | (none) | 0, `RESULT: L-AM13 OK (79/79)` | 0, `RESULT: L-AM13 OK (79/79)` | 03:04:04-03:04:21 |
| smoke_teacher_d1 | (none) | 0, `RESULT: PASS (145/145)` | 0, `RESULT: PASS (145/145)` | 03:04:21-03:05:41 |
| smoke_arms_kd | (none) | 1, `RESULT: FAIL (58/61)` | 1, `RESULT: FAIL (58/61)` | 03:05:41-03:06:29 |
| smoke_arms_kd (PYTHONPATH=checkout) | (none) | 0, `RESULT: PASS (61/61)` | 0, `RESULT: PASS (61/61)` | 03:06:29-03:07:17 |
| smoke_qat_selection | 174/174 (STEP 7) | not run | 0, `RESULT: PASS (174/174)` | 03:07:17-03:21:27 |
| smoke_qat_artifacts | 94/94 (STEP 7) | not run | 0, `RESULT: PASS (94/94)` | 03:21:27-04:09:03 |
| smoke_ptq_record | (none) | not run | 0, `RESULT: PASS (162/162)` | 04:09:03-04:19:20 |
| smoke_ptq_record_mutations | (none) | not run | 0, `RESULT: PASS (436/436 checks; 108/108 edits killed)` | 05:09:45-05:40:28 |

smoke_ptq_record_mutations: the row is the third full run, alone, at FINAL. The first two full runs (04:19:20Z,
the last step of the 02:50:40Z sequence, and 04:43:35Z) were stopped by container restarts, each with its 32
baselines PASS and 91 edits killed: all but LQ07 and LQ93, still running, and LQ94-LQ108, not started (§7.1).
Every other row is from the sequence that began at 02:50:40Z.

**The counts that changed, and the cases behind them.**
- smoke_quant_runners 77 → 78 (C1): e4_full_launch_stops_at_backend_gate became two checks,
  e4_full_launch_refused_old_runner and e7_full_launch_refused_old_runner (exit 2, code old_runner_real_run, the
  out-dir never created).
- smoke_run_ptq 69 → 69 (C1): run_e4_refuses_batch_of_2 became run_e4_refuses_every_real_run (`RESULT: REFUSED
  [old_runner_real_run]`, no --batch-size); the count is unchanged.
- smoke_ptq_record (new, C4; no baseline): `RESULT: PASS (162/162)` at FINAL: 134 at C4, +3 at C9 (s5_unreadable), +3 at C10
  (range_refuses_dataset_level_null, range_refuses_rows_not_integer, range_refuses_record_without_list_seed), +22 at
  C12 (refuses_e4_parent_max_iters_160k; nine refuses_e7_s43_launch_field_<field>; six
  refuses_e7_parent_not_of_record_<condition>; three refuses_e7_parent_teacher_<condition>;
  refuses_e7_payload_target_construction_key; official_refuses_record_of_another_torchscript and
  _record_of_another_list). C12 also changed two checks without adding one: refuses_arm_flag (D-1) and
  unbound_run_never_imports_record_module (D-4, now a full unbound run). The plan's estimate was 106.
- smoke_ptq_record_mutations (new, C5; no baseline): `RESULT: PASS (436/436 checks; 108/108 edits killed)` at FINAL, 108 edits (the plan's 58; 77 at C5,
  +1 C9, +3 C10, +27 C12). Its checks: mutation_ids_unique, 108 `_edit_applies`, the self-check (C13), a baseline
  per edit and shadow (108 × 2), 108 `_killed`, harness_ran_unsignalled and every_shadow_clean_at_the_end.
- smoke_qat_selection 174/174 and smoke_qat_artifacts 94/94 were not in STEP 4; both are named by STEP 7.
- smoke_arms_kd: FAIL (58/61) with PYTHONPATH unset at O and at FINAL, the same three checks
  (rf_record_teacher_arm_with_arm_teacher and rf_existing_output_gate_record at `arguments:pythonpath_not_checkout`,
  scripts/preflight_e1_trainval.py:246-251; mut_record_teacher_arm_accepts_arm_teacher, whose case fails unmutated);
  PASS (61/61) with PYTHONPATH = the checkout, at O and at FINAL. An environment requirement of that smoke, not a
  change of this lane (open item 4).
- Every other smoke: the same count at O and at FINAL.

## 3. Rulings R1-R18 at the final commit (file:line at FINAL; REPOSITORY-PROVEN unless marked)

| Ruling | Met | Where (file:line at FINAL) |
|---|---|---|
| Q1-R2 old runners refuse real runs | yes | src/quant/runner.py:231-236 (unconditional `old_runner_real_run`, exit 2, after `parser.parse_args` :230); scripts/run_e4.py:4-6, run_e7.py:4-6 (docstrings); smoke_quant_runners e4/e7_full_launch_refused_old_runner, smoke_run_ptq run_e4_refuses_every_real_run, smoke_ptq_record old_runners_write_nothing; LQ01 |
| R1 one merge; C7, C8 in Part 1 | yes | C1-C8 and the DL-24 fixes C9-C13 on lane/lqp before the one merge; no Part 2, no M2; both lanes proposed IMPLEMENTED at that merge (§10) |
| R2 int8_official_error, record gate, codes, QAT None, the one call, lazy import | yes | src/eval/stage_artifacts.py:190-197 (wrapper: calibration gate first, `or` the record gate), :200-239 (`_ptq_record_error`), codes :214 legacy_ptq_artifact, :221 ptq_record_missing, :227 ptq_record_invalid, :229 ptq_record_role, :238 (comparisons :231-236) ptq_record_mismatch; QAT: the `spec["method"] != "ptq"` return in `_ptq_record_error`; lazy import :217; scripts/evaluate_model.py:267-270 (the one call, `raise CliError`); evaluate_corruptions.py and profile_efficiency.py unchanged (not in the diff); the run provenance, TorchScript, source checkpoint and list checksum comparisons each refused alone (official_refuses_another_provenance, _record_of_another_torchscript, _record_of_another_run, _record_of_another_list; DL-24 D-6); s8_gate, s8_e2e, s8_source; LQ55-LQ64, LQ106-LQ108 |
| R3 E1 rule, absent/null poly_horizon = 80,000 | yes | src/quant/ptq_record.py bind_e1 :239-256 (parent_horizon :245-247, parent_payload :248-255); horizon_of_record reads configs/e1_student.py (`iterations` 80000; train_e1.py:336 default); max_iters and poly_horizon each refused alone (refuses_e4_parent_160k, refuses_e4_parent_poly_horizon_160k, refuses_e4_parent_max_iters_160k), an absent poly_horizon accepted (accepts_e4_parent_without_poly_horizon); LQ26-LQ28, LQ83; E1 log sha256 and RESULT line in L1's evidence (§6); qat.py untouched |
| R4 G0 exactly one source; G3 keeps source_run_dir_required, binding_needs_registered_list | yes | scripts/run_ptq.py:162-165 (`source_exactly_one`), :182-186 (`binding_needs_registered_list`); src/quant/ptq_record.py:385 (`source_run_dir_required`) |
| R5 docstrings name scripts/run_ptq.py, no "--stage" | yes | src/quant/runner.py:1-26, scripts/run_e4.py:2-10, scripts/run_e7.py:2-10; smoke_quant_runners stage_is_not_a_cli_flag, smoke_quant_e6_e7 e6_e7_launchers_are_gated_and_stage_pinned PASS |
| R6 isolation, descriptive, ptq_script, G6 on every bound run | yes | ptq_record.py ISOLATION_METHOD :79, isolation() :197-203 (called in bound_gates for every bound run), record "descriptive"/"isolation" in build_record, manifest_entry ptq_script; LQ22, LQ51, LQ52, LQ54 |
| R7 one identity reader, expect-head vs that commit, None refused, same commit | yes | ptq_record.py identity() :176-193 (`code_identity_unavailable`, `expect_head_mismatch`, `code_not_clean`), build_record's STOP identity_changed (run provenance code.commit = the gate's commit; the code paths re-read after the run); no git call added (ptq_record calls ptq.code_provenance, Q2's read_tracked_selection and sweep_select.HeadSource only); LQ18-LQ20, LQ42, LQ53 |
| R8 launch-log binding, full field set mirrored | yes | ptq_record.py launch_log_binding :312-342 mirrors sweep_select.py:1809-1815 (fields) and :1800-1806 (launched line, run_meta sha256); check_run_against_log (:1793) takes an `Attempt`, whose `float(self.line["value"])` (sweep_select.py:976-985) fails on a seed-43/44 line, so it is mirrored; no launched line at HEAD, or a stopped line, refused; each field refused alone (refuses_e7_s43_launch_line_fields for launch_time_utc, refuses_e7_s43_launch_field_<field> for the nine others; DL-24 D-3); LQ33-LQ38, LQ84-LQ92 |
| R9 Q2 codes kept, message led by stage and rule | yes | ptq_record.py `_q2` :207-213 (prefix "<stage> parent rule (<rule id>): "); smoke checks refuses_selection_sha256_mismatch and refuses_parent_sha256_mismatch assert the lead; LQ23, LQ24 |
| R10 the registered rule stated; fixture follows it | yes | scripts/build_calibration_lists.py:65 (`TRAIN_IMAGES = int(SPLIT_SIZES["train"])`, 5,367), :118-120 (a stem count other than expected refused), :137 (`registered = expected_images == TRAIN_IMAGES`), :157 (`artifact_status` "registered" else "smoke"); smoke_ptq_record builds a 5,367-pair synthetic TRAIN tree (cached) and its lists with the default expected count (fixture_lists_registered) |
| R11 unconditional refusal; run_ptq and build_calibration_loader removed after the grep | yes | grep at O: only runner.py:169 and :226; removed in C1 (2bc8943) with `_qconfig_summary` and the imports only they used; the helpers ptq.py, qat.py, qat_artifacts.py and the smokes import remain (runner.py check_output_dir, check_source, check_backend, check_calibration, CalibrationDataset, build_run_provenance) |
| R12 e7_record_projection_free | dropped | not in smoke_ptq_record.py |
| R13 range reporter | yes | scripts/ptq_calibration_range.py: metric and alongside :56-57, artifact_status :129-133, every summary sha256 in per_subset (`summary_sha256`), run.stage :113-115, clean TEST canvas :134-137; nothing on VAL (range_protocol refuses a VAL summary); malformed inputs refused as range_input_invalid (read_pair :83-106, DL-24 B-1); LQ65-LQ77, LQ79-LQ81 |
| R14 PARENT_RULES E4, E7 only | yes | ptq_record.py PARENT_RULES :345-348 |
| R15 citation slips | fixed in this report | §10 |
| R16 d3 via --replicate-of; L1 same container, threads, invocation | yes | ptq_record.py replicate_binding :352-366, role d3_replicate (descriptive, refused official: official_refuses_replicate_record); L1 §6.3 step 2 |
| R17 teacher values imported lazily; preflight_distill in RECORD_CODE_PATHS | yes | ptq_record.py:264, `import scripts.preflight_distill as pd` inside bind_e3; RECORD_EXTRA_CODE_PATHS :75-77 |
| R18 counts named | yes | §2, §5, §7 |

## 4. The command of record (scripts/run_ptq.py, bound path; REPOSITORY-PROVEN at FINAL)

### 4.1 Flags

`--real-run --confirm-real-run --stage {E4,E7} --seed {42,43,44} --source-run-dir RUN_DIR --expect-source-sha256 SHA256
--calibration-list configs/calibration/ptq_calibration_seed<S>.json --expect-list-checksum SHA256 --expect-head COMMIT
--data-root ROOT --out-dir /out/<stage>_s<seed>_cal<S>_a<n>[_d3] [--threads N] [--replicate-of RECORD]`; E7 adds
`--lambda-selection reports/derived/lambda_selection.json --lambda-selection-sha256 SHA256
--alpha-selection reports/derived/alpha_selection.json --alpha-selection-sha256 SHA256`. `--compare A B` (d3) is unchanged.
Unbound path (a list whose artifact_status is not "registered"): `--source-ckpt` instead of `--source-run-dir`, no binding
flag. There is no `--arm` flag (argparse refuses it: refuses_arm_flag).

### 4.2 Gates, in order (each exit 2, nothing written; no image, mask or checkpoint opened before G8)

| Gate | Code | Where (at FINAL) | Rule | Smoke case(s) (smoke_ptq_record unless named) | Mutation(s) |
|---|---|---|---|---|---|
| G0 | (REFUSING: ..., stderr; L-AM10's form, unchanged) | scripts/run_ptq.py:147-161 | --real-run and --confirm-real-run; --stage, --expect-source-sha256, --calibration-list, --expect-list-checksum, --data-root, --out-dir present | smoke_run_ptq (L-AM10's) | (L-AM10's) |
| G0 | source_exactly_one | scripts/run_ptq.py:162-165 | exactly one of --source-ckpt, --source-run-dir (R4) | refuses_source_exactly_one | LQ03 |
| G1 | test_path | scripts/run_ptq.py:166-172 | no path flag names "test", as given or resolved (qat.names_test, src/quant/qat.py:153), before anything is opened | refuses_test_path_run_dir, refuses_test_path_out_dir | LQ04 |
| G2 | calibration_list_* (ptq.load_calibration_list) | scripts/run_ptq.py:177 | the list loads, validates and matches --expect-list-checksum | (L-AM10's) | (L-AM10's) |
| G2 | (the bound/unbound split) | scripts/run_ptq.py:178-179 | artifact_status "registered" → src/quant/ptq_record.py, imported there only | e7_s43_exit_0, e4_s43_exit_0; unbound_run_never_imports_record_module | LQ06, LQ07, LQ93 |
| G2 | binding_needs_registered_list | scripts/run_ptq.py:182-186 | a binding flag on a list that is not a registered build | refuses_binding_flags_unregistered_list | LQ05 |
| G2 | calibration_list_not_committed | src/quant/ptq_record.py:129-147 (refusal :146) | the list as given is configs/calibration/ptq_calibration_seed<its seed>.json under the records root, tracked and byte-equal at HEAD (sweep_select.HeadSource.read) | refuses_list_not_committed, refuses_list_changed_since_head | LQ08 |
| G3 | seed_invalid | :380 | --seed in (42, 43, 44) | refuses_seed_invalid | LQ09 |
| G3 | sensitivity_seed | :382 | a 43/44/45 list only at --seed 42 (AM-16 item 5) | refuses_sensitivity_list_at_seed_43 | LQ10 |
| G3 | source_run_dir_required | :385 | a registered list takes --source-run-dir (R4) | refuses_source_run_dir_required | LQ11 |
| G3 | expect_head_format | :389 | --expect-head is 40 hex | refuses_expect_head_format | LQ12 |
| G3 | selection_not_applicable | :393 | E4 takes no λ or α flag | refuses_e4_selection_flags | LQ13 |
| G4 | out_dir_relative | :154 | --out-dir absolute | refuses_out_dir_relative | LQ14 |
| G4 | output_dir_inside_repo | :156-158 | runner.check_output_dir (E1's guard) | refuses_out_dir_inside_repo | LQ15 |
| G4 | out_dir_name | :161 | `<stage>_s<seed>_cal<S>_a<n>`, n ≥ 1, `_d3` exactly with --replicate-of | refuses_out_dir_name_stage, refuses_out_dir_name_list, refuses_d3_name_without_replicate, refuses_replicate_without_d3_name | LQ16 |
| G4 | out_dir_not_empty | :165 | absent or empty | refuses_out_dir_not_empty_bound | LQ17 |
| G5 | code_identity_unavailable | :182 | ptq.code_provenance(RECORD_CODE_PATHS)'s commit is not None (R7) | refuses_code_identity_unavailable | LQ18 |
| G5 | expect_head_mismatch | :186 | that commit = --expect-head (R7) | refuses_expect_head_mismatch | LQ19 |
| G5 | code_not_clean | :188 | RECORD_CODE_PATHS clean at that commit (R7) | refuses_code_not_clean | LQ20 |
| G5 | image_digest_not_of_record | :191 | PLANTSEG_IMAGE_DIGEST = qat.IMAGE_DIGEST_OF_RECORD (src/quant/qat.py:105; DL-21) | refuses_image_digest_unset, refuses_image_digest_not_of_record | LQ21 |
| G6 | data_root_has_test | :202 | src.data.isolation.assert_no_test_surfaces (src/data/isolation.py:31; existence only), on every bound run (R6) | refuses_data_root_with_test_surface | LQ22 |
| G7 | Q2's codes: lambda_selection_missing, alpha_selection_missing, selection_sha256_format, selection_path_not_repo_relative, selection_test_path, selection_missing_file, selection_not_tracked, selection_changed_since_head, selection_sha256_mismatch, selection_format | qat.read_tracked_selection (src/quant/qat.py:622) through `_q2` (src/quant/ptq_record.py:207-213; R9) | E7: the committed λ and α selections at the records root | refuses_e7_without_lambda_selection, refuses_selection_sha256_mismatch (its lead checked), refuses_selection_changed_since_head, refuses_selection_not_tracked, refuses_selection_format | LQ23, LQ24 |
| G8 | Q2's codes: parent_dir_test_path, parent_dir_missing, expect_source_sha256_format, parent_best_json_missing, parent_best_json_format, parent_checkpoint_test_path, parent_checkpoint_missing, parent_sha256_mismatch, parent_records_missing, parent_run_meta_rows, parent_run_meta_key, parent_mode_not_real, parent_seed_mismatch, parent_stage_mismatch, parent_incomplete | qat.resolve_parent (src/quant/qat.py:469) through `_q2` | the parent directory; its best.json checkpoint = --expect-source-sha256; its records complete | refuses_parent_dir_missing, refuses_parent_sha256_mismatch (its lead checked), refuses_parent_best_json_format, refuses_parent_two_run_meta_rows, refuses_parent_mode_dry, refuses_parent_seed_mismatch, refuses_e4_parent_without_final_val, refuses_e7_parent_stage_e2, refuses_e7_parent_run_abort, refuses_e7_parent_without_run_end, refuses_e7_parent_checks_failed | LQ23 (the wrapper) |
| G9 | parent_horizon | :246 (condition :245) | E1: max_iters = 80,000, and poly_horizon absent, null or 80,000 (R3) | refuses_e4_parent_160k, refuses_e4_parent_poly_horizon_160k, refuses_e4_parent_max_iters_160k (DL-24 D-2), accepts_e4_parent_without_poly_horizon | LQ26, LQ27, LQ28, LQ83 |
| G9 | parent_payload | :255 | the E1 payload: stage absent, null or E1; best_val_miou_all_class = best.json's | refuses_e4_parent_payload_best_mismatch, refuses_e4_parent_payload_stage_e3 | LQ29 |
| G10 | Q2's parent_stage_mismatch, parent_seed_mismatch, parent_lambda_mismatch, parent_alpha_mismatch, parent_run_id_mismatch (s42) | qat.e6_parent_binding (src/quant/qat.py:658) through `_q2` | λ and α = the selections' winners; at seed 42 the α winner's run_id | refuses_e7_parent_lambda_mismatch, refuses_e7_parent_alpha_mismatch, refuses_e7_s42_non_winner_run | LQ25 |
| G10 | parent_not_of_record | :278 (conditions :270-276) | parent_of_e4_e7 true, arm null, descriptive false, max_iters and poly_horizon 80,000, no target_construction: each refused alone (DL-24 D-6) | refuses_e7_parent_arm_run, refuses_e7_parent_160k, refuses_e7_parent_not_of_record_{arm, descriptive, parent_of_e4_e7, max_iters, poly_horizon, target_construction} | LQ30, LQ94-LQ99 |
| G10 | parent_teacher_not_of_record | :283 (conditions :281-282) | teacher_mock false; teacher_provenance an object with builder segnext_mscan_b_builder and ckpt_sha256 = expected_sha256 = 8c0e649a… (scripts/preflight_distill.py:93-96, imported lazily, R17): each refused alone (DL-24 D-6) | refuses_e7_parent_teacher_r1, refuses_e7_parent_teacher_mock, refuses_e7_parent_teacher_{builder, expected_sha256, not_an_object} | LQ31, LQ100-LQ104 |
| G10 | parent_checkpoint_not_selected | :289 | seed 42: the α winner's ckpt_sha256 = the parent checkpoint's | refuses_e7_s42_checkpoint_not_selected | LQ32 |
| G10 | parent_launch_log (5 sites) | :320, :324, :326, :328, :341 (fields :330-338) | reports/kd_launch_log.jsonl committed at HEAD (load_launch_log); a launch and a launched line for the run_id; no stopped line; the launched run_meta_sha256 = the directory's; sweep_select.py:1809-1815's fields (R8), each refused alone (DL-24 D-3) | refuses_e7_launch_log_uncommitted, refuses_e7_s43_not_launched, refuses_e7_s43_stopped_line, refuses_e7_s43_run_meta_not_launched, refuses_e7_s43_launch_line_fields (launch_time_utc), refuses_e7_s43_launch_field_{git_head, code_pin, records_commit, swept_key, stage, seed, horizon, lambda_logit, alpha_cwd} | LQ33-LQ38, LQ84-LQ92 |
| G10 | parent_payload | :302 | the E3 payload: stage E3, no arm or target_construction key, teacher_provenance = run_meta's, best_val_miou_all_class = best.json's | refuses_e7_payload_arm_key, refuses_e7_payload_target_construction_key (DL-24 D-6), refuses_e7_payload_teacher_mismatch, refuses_e7_payload_best_mismatch | LQ39, LQ105 |
| G11 | replicate_mismatch (2 sites) | :356, :363 | --replicate-of is an intact ptq_record/1 of role of_record or calibration_sensitivity with this stage, seed, list checksum and parent checkpoint, in <this out-dir minus _d3> | refuses_replicate_not_a_record, refuses_replicate_other_list, refuses_replicate_other_parent, refuses_replicate_of_replicate | LQ40, LQ41, LQ44 |
| G2-G11 | bound_input_unreadable (DL-24 A-1) | :427 (run_bound :417-428) | any other error a gate raises: an input it cannot read (a torn telemetry line, an unloadable checkpoint, no git binary) | refuses_e7_parent_torn_telemetry_line, refuses_e4_parent_checkpoint_unloadable, refuses_without_git_binary | LQ78 |

Then `src.quant.ptq.run_ptq(stage, source_ckpt=<best.json's checkpoint>, ...)` (src/quant/ptq.py:761; L-AM10, changed
only by C2), with its own refusals, each exit 2 before anything is written (src/quant/ptq.py at FINAL):
stage_unknown :773, stage_not_ptq :775, fresh_process_required :777, expected_hash_invalid :784, runner.check_output_dir's
codes (output_dir_inside_repo, ...) :788, output_exists :792, out_dir_not_empty :793-795 (C2), data_root_unset :797,
eval_split_path (refuse_eval_split_dir :226; the error :195), calibration_list_missing :318, calibration_list_invalid
:324, calibration_list_checksum_mismatch :326 (load_calibration_list), source_missing :802, source_sha256_mismatch :805,
backend_unavailable :810, threads_invalid :813, runner.check_source's codes :821, source_changed :823,
calibration_count :837, train_dir_missing :346, calibration_ids_missing (src/quant/runner.py:158, through :350),
calibration_order :356, wrong_engine :389, output_exists :450 (the writer); and its STOPs (exit 1,
`<stage>_ptq_STOP.json`, no run provenance): artifact_write_mismatch :868; per_tensor_weight,
undeclared_float_fallback, weights_unreadable, parity_mismatch :920-928. Then the record: STOP `identity_changed`
(src/quant/ptq_record.py:454: the run provenance's code commit, clean flag or CODE_PATHS sha256s, the RECORD_CODE_PATHS
re-read after the run, the source sha256 or the list file sha256 and checksum differ from what the gates bound) and
STOP `record_write_failed` (:492: `<stage>_ptq_record.json` created with open "xb"). Last line: `RESULT: PTQ OF RECORD
WRITTEN (<stage> s<seed>, role of_record, parent <run_id>, record <sha12>)`, or `RESULT: PTQ RECORD WRITTEN (...)` for
the descriptive roles. Exit 0 written, 1 STOP, 2 refused (scripts/run_ptq.py:192-197). `--compare A B` (d3) refuses
compare_same_run (src/quant/ptq.py:1043) and any unreadable input (scripts/run_ptq.py:139-141, exit 2).

### 4.3 ptq_record/1 (`<stage>_ptq_record.json`, strict JSON: sorted keys, indent 2, no NaN; written last, exclusively)

Top-level keys (src/quant/ptq_record.py RECORD_KEYS :83-85): format "ptq_record/1"; stage; seed; role (of_record |
calibration_sensitivity | d3_replicate); descriptive (false only for of_record, R6); replicate_of ({path, sha256} of the
original record, or null); authority; parent; calibration {list_path (repo-relative), list_file_sha256, checksum_sha256,
seed, role, list_artifact_status}; ptq_run {run_meta (name), sha256, ptq_run_role, schema}; artifacts {torchscript,
state_dict, x86_copy: {path, sha256, bytes}}; qconfig_fingerprint; observers; engine; environment {image_digest, torch,
torchvision, numpy, pillow, torch_num_threads}; isolation {data_root, method "src.data.isolation.assert_no_test_surfaces:
TEST-surface absence; TRAIN mounted for PTQ", result} (R6); writer {script {path scripts/run_ptq.py, sha256}, module
{path src/quant/ptq_record.py, sha256}, commit, clean_at_commit, code_sha256 (every RECORD_CODE_PATH)} (R7: writer.commit
= the identity reader's commit = the run provenance's code.commit); generated_utc.
parent (E1 rule "e1_run_of_record/1"): rule, stage E1, run_id, run_dir, best_json_sha256, best_val_miou_all_class,
checkpoint, checkpoint_sha256, checkpoint_bytes, run_meta {file, sha256}, telemetry {file, sha256}, mode, seed, max_iters,
poly_horizon (null when the run_meta row has none, R3). The E3 rule "e3_run_of_record/1" adds lambda_logit, alpha_cwd,
parent_of_e4_e7 (true), teacher_arch "segnext_mscan-b", teacher_ckpt_sha256, run_end {iter, checks_passed,
wall_clock_end, best_ckpt}, selections {lambda, alpha: {path, sha256, format, winner}}, launch_log {path, sha256 at
HEAD, launch, launched}.
RECORD_CODE_PATHS = ptq.CODE_PATHS (14 files, src/quant/ptq.py:147-151) + RECORD_EXTRA_CODE_PATHS (:75-77):
src/quant/ptq_record.py, src/quant/stages.py, src/quant/qat.py, src/training/sweep_select.py, src/data/isolation.py,
scripts/preflight_distill.py (R17), scripts/preflight_e1_trainval.py, configs/e1_student.py (the 80,000 horizon the
parent rules read; added in Phase B).

### 4.4 The manifest lane's interface (pure; no TEST path)

- `read_record(path, expect_sha256=None)` → (record, sha256): the format, the 18 keys, role and descriptive, the nested
  objects, and every listed file (three artifacts and the run provenance, beside the record) re-hashed; RecordInvalid or
  OSError otherwise.
- `manifest_entry(record)` → {status "made", stage, seed, role, parent_checkpoint_sha256, list_file_sha256,
  list_checksum_sha256, ptq_commit, ptq_script (R6), image_digest, qconfig_fingerprint, artifacts {k: {path, sha256}}}
  (AM-19 item 4(c)).
- `planned_entry(stage, seed, parent_checkpoint_sha256, list_file_sha256, list_checksum_sha256, ptq_commit,
  image_digest)` → the same keys with status "planned", role of_record, ptq_script scripts/run_ptq.py, artifacts null, and
  the registered PTQ qconfig's fingerprint (ptq.qconfig_summary(ptq_qconfig())); the smoke checks it equals the real E4
  s42 record's entry but for status and artifacts (e4_planned_entry_matches_record).

## 5. The mutation list and its killing checks

scripts/smoke_ptq_record_mutations.py at FINAL holds 108 edits, LQ01-LQ108 (REPOSITORY-PROVEN; the table is parsed
from its MUTATIONS list, never imported): LQ01-LQ77 from C5 (the plan's 58 edits, grown to 77 by every refusal C3,
C7 and C8 added), LQ78 (C9, DL-24 A-1), LQ79-LQ81 (C10, B-1) and LQ82-LQ108 (C12, D-1 to D-4 and D-6). Each is one
exact string, which must occur once in its file, replaced so that the edited file still parses; the case named runs
in a shadow of the repository (`--only`), and the edit is killed only when a named check that PASSED on the
unmutated shadow is reported FAIL. Every new refusal code of the lane has at least one edit (§4.2's last column).
The final run, at FINAL, is in §7.1.

| Id | File | Edit | Case (--only) | Killing check(s) |
|---|---|---|---|---|
| LQ01 | src/quant/runner.py | the old runner's refusal skipped: E4/E7 return 0 once parsed | s1_old_runners | old_runners_write_nothing |
| LQ02 | src/quant/ptq.py | ptq.run_ptq's out_dir_not_empty refusal removed | s5_out_dir | refuses_out_dir_not_empty_unbound |
| LQ03 | scripts/run_ptq.py | G0 source_exactly_one removed | s5_flags | refuses_source_exactly_one |
| LQ04 | scripts/run_ptq.py | G1 test_path removed | s5_flags | refuses_test_path_run_dir, refuses_test_path_out_dir |
| LQ05 | scripts/run_ptq.py | binding_needs_registered_list removed | s5_flags | refuses_binding_flags_unregistered_list |
| LQ06 | scripts/run_ptq.py | a registered list routed to the unbound path | s4_seeds_43 | e7_s43_exit_0, e4_s43_exit_0 |
| LQ07 | scripts/run_ptq.py | the record module imported at the top of run_ptq.py | s10_hygiene | unbound_run_never_imports_record_module |
| LQ08 | src/quant/ptq_record.py | G2 calibration_list_not_committed removed | s5_flags | refuses_list_not_committed, refuses_list_changed_since_head |
| LQ09 | src/quant/ptq_record.py | G3 seed_invalid removed | s5_flags | refuses_seed_invalid |
| LQ10 | src/quant/ptq_record.py | G3 sensitivity_seed removed | s5_flags | refuses_sensitivity_list_at_seed_43 |
| LQ11 | src/quant/ptq_record.py | G3 source_run_dir_required removed | s5_flags | refuses_source_run_dir_required |
| LQ12 | src/quant/ptq_record.py | G3 expect_head_format removed | s5_flags | refuses_expect_head_format |
| LQ13 | src/quant/ptq_record.py | G3 selection_not_applicable removed | s5_flags | refuses_e4_selection_flags |
| LQ14 | src/quant/ptq_record.py | G4 out_dir_relative removed | s5_out_dir | refuses_out_dir_relative |
| LQ15 | src/quant/ptq_record.py | G4 output_dir_inside_repo swallowed | s5_out_dir | refuses_out_dir_inside_repo |
| LQ16 | src/quant/ptq_record.py | G4 out_dir_name removed | s5_out_dir | refuses_out_dir_name_stage, refuses_out_dir_name_list, refuses_d3_name_without_replicate, refuses_replicate_without_d3_name |
| LQ17 | src/quant/ptq_record.py | G4 out_dir_not_empty removed (bound path) | s5_out_dir | refuses_out_dir_not_empty_bound |
| LQ18 | src/quant/ptq_record.py | G5 code_identity_unavailable removed (R7) | s5_identity | refuses_code_identity_unavailable |
| LQ19 | src/quant/ptq_record.py | G5 expect_head_mismatch removed (R7) | s5_identity | refuses_expect_head_mismatch |
| LQ20 | src/quant/ptq_record.py | G5 code_not_clean removed (R7) | s5_identity | refuses_code_not_clean |
| LQ21 | src/quant/ptq_record.py | G5 image_digest_not_of_record removed | s5_identity | refuses_image_digest_unset, refuses_image_digest_not_of_record |
| LQ22 | src/quant/ptq_record.py | G6 data_root_has_test swallowed (R6) | s5_identity | refuses_data_root_with_test_surface |
| LQ23 | src/quant/ptq_record.py | a Q2 refusal swallowed by the wrapper | s5_selections | refuses_selection_sha256_mismatch |
| LQ24 | src/quant/ptq_record.py | a Q2 refusal's message not led by the stage and rule (R9) | s5_selections | refuses_selection_sha256_mismatch |
| LQ25 | src/quant/ptq_record.py | G10 qat.e6_parent_binding dropped | s5_e3_rule | refuses_e7_parent_lambda_mismatch, refuses_e7_parent_alpha_mismatch, refuses_e7_s42_non_winner_run |
| LQ26 | src/quant/ptq_record.py | G9 parent_horizon removed | s5_e1_rule | refuses_e4_parent_160k, refuses_e4_parent_poly_horizon_160k |
| LQ27 | src/quant/ptq_record.py | G9 an absent poly_horizon refused (R3) | s5_e1_rule | accepts_e4_parent_without_poly_horizon |
| LQ28 | src/quant/ptq_record.py | G9 a present poly_horizon not compared (R3) | s5_e1_rule | refuses_e4_parent_poly_horizon_160k |
| LQ29 | src/quant/ptq_record.py | G9 parent_payload removed (E1) | s5_e1_rule | refuses_e4_parent_payload_best_mismatch, refuses_e4_parent_payload_stage_e3 |
| LQ30 | src/quant/ptq_record.py | G10 parent_not_of_record removed | s5_e3_rule | refuses_e7_parent_arm_run, refuses_e7_parent_160k |
| LQ31 | src/quant/ptq_record.py | G10 parent_teacher_not_of_record removed (R17) | s5_e3_rule | refuses_e7_parent_teacher_r1, refuses_e7_parent_teacher_mock |
| LQ32 | src/quant/ptq_record.py | G10 parent_checkpoint_not_selected removed | s5_e3_rule | refuses_e7_s42_checkpoint_not_selected |
| LQ33 | src/quant/ptq_record.py | G10 an unreadable launch log swallowed (R8) | s5_e3_rule | refuses_e7_launch_log_uncommitted |
| LQ34 | src/quant/ptq_record.py | G10 no launched line accepted (R8) | s5_e3_rule | refuses_e7_s43_not_launched |
| LQ35 | src/quant/ptq_record.py | G10 a stopped line accepted (R8) | s5_e3_rule | refuses_e7_s43_stopped_line |
| LQ36 | src/quant/ptq_record.py | G10 the launched run_meta sha256 not compared (R8) | s5_e3_rule | refuses_e7_s43_run_meta_not_launched |
| LQ37 | src/quant/ptq_record.py | G10 the launch-log fields not compared (R8) | s5_e3_rule | refuses_e7_s43_launch_line_fields |
| LQ38 | src/quant/ptq_record.py | G10 launch_time_utc dropped from the field set (R8, sweep_select.py:1815) | s5_e3_rule | refuses_e7_s43_launch_line_fields |
| LQ39 | src/quant/ptq_record.py | G10 parent_payload removed (E3) | s5_e3_rule | refuses_e7_payload_arm_key, refuses_e7_payload_teacher_mismatch, refuses_e7_payload_best_mismatch |
| LQ40 | src/quant/ptq_record.py | G11 an unreadable --replicate-of swallowed | s5_replicate | refuses_replicate_not_a_record |
| LQ41 | src/quant/ptq_record.py | G11 replicate_mismatch removed | s5_replicate | refuses_replicate_other_list, refuses_replicate_other_parent, refuses_replicate_of_replicate |
| LQ42 | src/quant/ptq_record.py | STOP identity_changed removed (R7) | s5_stops | stop_identity_changed |
| LQ43 | src/quant/ptq_record.py | STOP record_write_failed removed | s5_stops | stop_record_write_failed |
| LQ44 | src/quant/ptq_record.py | read_record's format and keys not checked | s5_replicate | refuses_replicate_not_a_record |
| LQ45 | src/quant/ptq_record.py | read_record's re-hash of the listed files dropped | s8_gate | official_refuses_tampered_artifact |
| LQ46 | src/quant/ptq_record.py | the parent's run_meta sha256 omitted | s4_seeds_43 | e4_s43_record_fields |
| LQ47 | src/quant/ptq_record.py | the list file sha256 omitted | s4_seeds_43 | e4_s43_record_fields |
| LQ48 | src/quant/ptq_record.py | another file's sha256 recorded for the run provenance | s4_seeds_43 | e4_s43_record_fields |
| LQ49 | src/quant/ptq_record.py | a sensitivity run labelled of_record | s7_sensitivity | e4_s42_cal43_role, e7_s42_cal44_role |
| LQ50 | src/quant/ptq_record.py | replicate_of omitted | s7_sensitivity | e4_d3_stub_replicate_role |
| LQ51 | src/quant/ptq_record.py | descriptive false for every role (R6) | s7_sensitivity | e4_s42_cal43_role |
| LQ52 | src/quant/ptq_record.py | the isolation block omitted (R6) | s4_seeds_43 | e4_s43_record_fields |
| LQ53 | src/quant/ptq_record.py | the writer's commit not the identity reader's (R7) | s4_seeds_43 | e4_s43_record_fields |
| LQ54 | src/quant/ptq_record.py | manifest_entry's ptq_script dropped (R6) | s4_seeds_43 | e4_s43_manifest_entry |
| LQ55 | src/eval/stage_artifacts.py | legacy_ptq_artifact removed | s8_gate | official_refuses_legacy_state_dict_ptq |
| LQ56 | src/eval/stage_artifacts.py | ptq_record_missing removed | s8_gate | official_refuses_missing_record, official_refuses_arm_directory |
| LQ57 | src/eval/stage_artifacts.py | ptq_record_invalid swallowed | s8_gate | official_refuses_malformed_record, official_refuses_tampered_artifact |
| LQ58 | src/eval/stage_artifacts.py | ptq_record_role removed | s8_gate | official_refuses_replicate_record |
| LQ59 | src/eval/stage_artifacts.py | ptq_record_mismatch removed | s8_gate | official_refuses_record_of_another_run, official_refuses_another_provenance |
| LQ60 | src/eval/stage_artifacts.py | the run provenance not compared with the record's | s8_gate | official_refuses_another_provenance |
| LQ61 | src/eval/stage_artifacts.py | the record gate called before the calibration gate | s8_gate | official_calibration_gate_runs_first |
| LQ62 | src/eval/stage_artifacts.py | QAT artifacts sent through the PTQ record gate | s8_gate | official_record_gate_passes_qat |
| LQ63 | scripts/evaluate_model.py | evaluate_model.py calls the calibration gate directly | s8_source | evaluate_model_calls_only_the_wrapper |
| LQ64 | scripts/evaluate_model.py | evaluate_model.py ignores the wrapper's refusal | s8_source | evaluate_model_calls_only_the_wrapper |
| LQ65 | scripts/ptq_calibration_range.py | output_dir_inside_repo swallowed | s9_range | range_refuses_out_dir_inside_repo |
| LQ66 | scripts/ptq_calibration_range.py | range_input_invalid not raised as a refusal | s9_range | range_refuses_input_invalid |
| LQ67 | scripts/ptq_calibration_range.py | range_stage removed | s9_range | range_refuses_stage |
| LQ68 | scripts/ptq_calibration_range.py | range_lists removed | s9_range | range_refuses_duplicate_list, range_refuses_missing_list, range_refuses_replicate_as_list_of_record |
| LQ69 | scripts/ptq_calibration_range.py | range_parent removed | s9_range | range_refuses_two_parents |
| LQ70 | scripts/ptq_calibration_range.py | range_artifact removed | s9_range | range_refuses_summary_of_another_artifact |
| LQ71 | scripts/ptq_calibration_range.py | range_status removed | s9_range | range_refuses_status |
| LQ72 | scripts/ptq_calibration_range.py | range_protocol removed | s9_range | range_refuses_mixed_split |
| LQ73 | scripts/ptq_calibration_range.py | range_rows removed | s9_range | range_refuses_rows |
| LQ74 | scripts/ptq_calibration_range.py | range_metric removed | s9_range | range_refuses_metric |
| LQ75 | scripts/ptq_calibration_range.py | range_output_exists swallowed | s9_range | range_refuses_existing_output |
| LQ76 | scripts/ptq_calibration_range.py | a summary's sha256 not recorded (R13) | s9_range | range_written |
| LQ77 | scripts/ptq_calibration_range.py | the disease-only range dropped (R13) | s9_range | range_written |
| LQ78 | src/quant/ptq_record.py | an input a gate cannot read escapes as a traceback (DL-24 A-1) | s5_unreadable | refuses_e7_parent_torn_telemetry_line, refuses_e4_parent_checkpoint_unloadable, refuses_without_git_binary |
| LQ79 | scripts/ptq_calibration_range.py | a summary's run, dataset and dataset_level not checked to be objects (DL-24 B-1) | s9_range | range_refuses_dataset_level_null |
| LQ80 | scripts/ptq_calibration_range.py | a summary's row counts not checked to be integers (DL-24 B-1) | s9_range | range_refuses_rows_not_integer |
| LQ81 | scripts/ptq_calibration_range.py | a record's list seed read leniently (DL-24 B-1) | s9_range | range_refuses_record_without_list_seed |
| LQ82 | scripts/run_ptq.py | run_ptq.py accepts an --arm flag (DL-24 D-1) | s5_flags | refuses_arm_flag |
| LQ83 | src/quant/ptq_record.py | G9 max_iters not compared, poly_horizon alone (R3; DL-24 D-2) | s5_e1_rule | refuses_e4_parent_max_iters_160k |
| LQ84 | src/quant/ptq_record.py | G10 git_head dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_git_head |
| LQ85 | src/quant/ptq_record.py | G10 code_pin dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_code_pin |
| LQ86 | src/quant/ptq_record.py | G10 records_commit dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_records_commit |
| LQ87 | src/quant/ptq_record.py | G10 stage dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_stage |
| LQ88 | src/quant/ptq_record.py | G10 seed dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_seed |
| LQ89 | src/quant/ptq_record.py | G10 horizon dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_horizon |
| LQ90 | src/quant/ptq_record.py | G10 the swept key dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_swept_key |
| LQ91 | src/quant/ptq_record.py | G10 lambda_logit dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_lambda_logit |
| LQ92 | src/quant/ptq_record.py | G10 alpha_cwd dropped from the field set (R8; DL-24 D-3) | s5_e3_rule | refuses_e7_s43_launch_field_alpha_cwd |
| LQ93 | scripts/run_ptq.py | run_ptq.py imports the record module once the core returns (DL-24 D-4) | s10_hygiene | unbound_run_never_imports_record_module |
| LQ94 | src/quant/ptq_record.py | G10 parent_of_e4_e7 not checked (DL-24 D-6) | s5_e3_rule | refuses_e7_parent_not_of_record_parent_of_e4_e7 |
| LQ95 | src/quant/ptq_record.py | G10 arm not checked (DL-24 D-6) | s5_e3_rule | refuses_e7_parent_not_of_record_arm |
| LQ96 | src/quant/ptq_record.py | G10 descriptive not checked (DL-24 D-6) | s5_e3_rule | refuses_e7_parent_not_of_record_descriptive |
| LQ97 | src/quant/ptq_record.py | G10 the E3 parent's max_iters not checked (DL-24 D-6) | s5_e3_rule | refuses_e7_parent_not_of_record_max_iters |
| LQ98 | src/quant/ptq_record.py | G10 the E3 parent's poly_horizon not checked (DL-24 D-6) | s5_e3_rule | refuses_e7_parent_not_of_record_poly_horizon |
| LQ99 | src/quant/ptq_record.py | G10 target_construction not checked in run_meta (DL-24 D-6) | s5_e3_rule | refuses_e7_parent_not_of_record_target_construction |
| LQ100 | src/quant/ptq_record.py | G10 the teacher's builder not checked (R17; DL-24 D-6) | s5_e3_rule | refuses_e7_parent_teacher_builder |
| LQ101 | src/quant/ptq_record.py | G10 the teacher's expected_sha256 not checked (R17; DL-24 D-6) | s5_e3_rule | refuses_e7_parent_teacher_expected_sha256 |
| LQ102 | src/quant/ptq_record.py | G10 a teacher_provenance that is not an object not refused (DL-24 D-6) | s5_e3_rule | refuses_e7_parent_teacher_not_an_object |
| LQ103 | src/quant/ptq_record.py | G10 the teacher's ckpt_sha256 not checked (R17) | s5_e3_rule | refuses_e7_parent_teacher_r1 |
| LQ104 | src/quant/ptq_record.py | G10 teacher_mock not checked (R17) | s5_e3_rule | refuses_e7_parent_teacher_mock |
| LQ105 | src/quant/ptq_record.py | G10 the payload's target_construction key not checked (DL-24 D-6) | s5_e3_rule | refuses_e7_payload_target_construction_key |
| LQ106 | src/eval/stage_artifacts.py | the record's TorchScript sha256 not compared (R2; DL-24 D-6) | s8_gate | official_refuses_record_of_another_torchscript |
| LQ107 | src/eval/stage_artifacts.py | the record's list checksum not compared (R2; DL-24 D-6) | s8_gate | official_refuses_record_of_another_list |
| LQ108 | src/eval/stage_artifacts.py | the record's source checkpoint not compared (R2) | s8_gate | official_refuses_record_of_another_run |

## 6. Local block L1 — draft (runs only under its own operator GO, after the merge)

L1 is a local block on the laptop, CPU only, in the student image, Docker `--network none`, at a pin P that is the
lane's merge commit or later. Nothing here has run; every command waits for an operator GO that names it.

### 6.1 Preconditions (ruling R7: one identity reader)

1. P checked out on the laptop; `git status --porcelain -- <RECORD_CODE_PATHS>` empty (the bound path refuses
   `code_not_clean` otherwise) and `git rev-parse HEAD` = P (passed as `--expect-head`).
2. `docker image inspect` shows the student image of record,
   sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf (DL-21; `qat.IMAGE_DIGEST_OF_RECORD`),
   passed as `-e PLANTSEG_IMAGE_DIGEST=...` (`image_digest_not_of_record` otherwise).
3. git answers inside the container for the read-only clone: `docker run --rm ... git --version`,
   `git -C /repo rev-parse HEAD` = P and the RECORD_CODE_PATHS status empty, run with the same `docker run` flags
   as the PTQ runs. A clone owned by another uid needs safe.directory; INFERRED fix without writing any config:
   `-e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/repo` (git >= 2.31). Without
   it, `ptq.code_provenance` answers a null commit and the run refuses `code_identity_unavailable` (R7); the λ/α
   selection and launch-log reads (git show / HeadSource) need the same.
4. The E1 seed-42 run folder: one run_meta row (mode real, seed 42, max_iters 80000), the VAL row at 80000,
   best.json with its two keys, checkpoint cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03 (DL-17);
   and, for the evidence (R3), the E1 log's sha256 and its RESULT line, read on the host.
5. The E3 α=25 seed-42 run folder e3_s42_lambda0.5_alpha25_80k_a1 from its keep tar (DL-120, the α sweep's run ends
   and copies), checkpoint 4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d (DL-121, the α selection of
   record), each folder checked against its manifest.
6. A fresh output root outside the repository; the TRAIN (and, for d4/d5, VAL) folders of the dataset; no TEST
   surface under the mounted data root (G6 checks `images/test`, `annotations/test`, `annotation_test.json` by
   existence only).

### 6.2 Mounts and environment

```
docker run --rm --network none --user <uid>:<gid> -w /repo \
  -v <clone>:/repo:ro \
  -v <dataset>/images/train:/data/plantseg/images/train:ro \
  -v <dataset>/annotations/train:/data/plantseg/annotations/train:ro \
  [-v <dataset>/images/val:/data/plantseg/images/val:ro -v <dataset>/annotations/val:/data/plantseg/annotations/val:ro]   # d4, d5 only
  -v <E1 s42 run folder>:/runs/<E1 run_id>:ro -v <E3 run folder>:/runs/e3_s42_lambda0.5_alpha25_80k_a1:ro \
  -v <output root>:/out \
  -e PLANTSEG_IMAGE_DIGEST=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf \
  -e PLANTSEG_DATA_ROOT=/data/plantseg -e PYTHONDONTWRITEBYTECODE=1 \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/repo \
  <student image> <command>
```

### 6.3 Commands, in order (each its own `docker run`, one PTQ run per process)

1. E4 s42 of record:
   `python -B /repo/scripts/run_ptq.py --real-run --confirm-real-run --stage E4 --seed 42
   --source-run-dir /runs/<E1 run_id> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03
   --calibration-list configs/calibration/ptq_calibration_seed42.json
   --expect-list-checksum 09320cbe7d5e8cee6d6a55c84679a65619fb52e3323fcdbae102c834a902357a
   --expect-head <P> --data-root /data/plantseg --out-dir /out/e4_s42_cal42_a1 --threads <N>`
   → `RESULT: PTQ OF RECORD WRITTEN (E4 s42, role of_record, parent <E1 run_id>, record <sha12>)`.
2. Its d3 replicate (R16): the same container image, the same `docker run` flags, the same `--threads <N>` and the same
   invocation, plus `--replicate-of /out/e4_s42_cal42_a1/e4_ptq_record.json --out-dir /out/e4_s42_cal42_a1_d3`
   → role d3_replicate; then `python -B /repo/scripts/run_ptq.py --compare /out/e4_s42_cal42_a1/e4_ptq_run_meta.json
   /out/e4_s42_cal42_a1_d3/e4_ptq_run_meta.json` → `RESULT: PTQ d3 PASS (bitwise identical)`.
3. E7 s42 of record:
   `python -B /repo/scripts/run_ptq.py --real-run --confirm-real-run --stage E7 --seed 42
   --source-run-dir /runs/e3_s42_lambda0.5_alpha25_80k_a1
   --expect-source-sha256 4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d
   --lambda-selection reports/derived/lambda_selection.json
   --lambda-selection-sha256 719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145
   --alpha-selection reports/derived/alpha_selection.json
   --alpha-selection-sha256 1399b897278db17a8876c724b89c8d8e7a7b42ffe1d9c4d67002578a889f2dd8
   --calibration-list configs/calibration/ptq_calibration_seed42.json
   --expect-list-checksum 09320cbe7d5e8cee6d6a55c84679a65619fb52e3323fcdbae102c834a902357a
   --expect-head <P> --data-root /data/plantseg --out-dir /out/e7_s42_cal42_a1 --threads <N>`;
   then its replicate into /out/e7_s42_cal42_a1_d3 and `--compare`, as in 2.
4. d2: `python -B /repo/scripts/smoke_qnnpack_ops.py --provenance /out/e4_s42_cal42_a1/e4_ptq_run_meta.json
   --source-ckpt <E1 checkpoint> --calibration-list configs/calibration/ptq_calibration_seed42.json
   --data-root /data/plantseg --out-dir /out/d2_e4`, and the same for E7 (/out/d2_e7).
5. d4: `python -B /repo/scripts/evaluate_model.py --stage E4 --precision int8_ptq
   --provenance /out/e4_s42_cal42_a1/e4_ptq_run_meta.json --split val --artifact-status provisional --device cpu
   --out-dir /out/d4_e4` (VAL mounted), and E7 the same, descriptive.
6. d5: `python -B /repo/scripts/ptq_val_checks.py backend --provenance /out/e4_s42_cal42_a1/e4_ptq_run_meta.json
   --out-dir /out/d5_e4` (full VAL) and `python -B /repo/scripts/ptq_val_checks.py parity --provenance ...`.
7. The six sensitivity runs (AM-16 item 5), E4 and E7 at seed 42 with the of-record flags except
   `--calibration-list configs/calibration/ptq_calibration_seed4X.json --expect-list-checksum <4X>
   --out-dir /out/e{4,7}_s42_cal4X_a1`: 43 1fbfc12a2d35d4e09db33aaed70fe558380ea9db49ae18857c0dc6323e24ffa9,
   44 e8e04425f0cac2b6f6aa42895c874e5d1176ce4319a44438a229d92bc920082b,
   45 02966ea8d619dd3d3384519ef34a36eb33a672441bead52cc1f0300a6b10af9d. Each prints
   `RESULT: PTQ RECORD WRITTEN (..., role calibration_sensitivity, ...)`. None is scored on VAL.
8. The range report's inputs (TEST session, after the unlock): per stage the four records
   (/out/e{4,7}_s42_cal4{2,3,4,5}_a1/<stage>_ptq_record.json) and the four evaluator summaries of their artifacts on
   clean TEST, canvas protocol (the seed-42 one `--artifact-status official`, the other three `provisional`); then
   `python -B /repo/scripts/ptq_calibration_range.py --stage E4 --pair <record> <summary> (×4) --out-dir /out/range`
   and the same for E7.
9. Evidence manifest: the sha256 of every file under /out, the number of paths whose name contains "test" (0), the
   E1 log's sha256 and RESULT line (R3), the docker image digest and P; copies to the laptop and to Drive; the
   docs session enters the DL row (no repository write in L1).

## 7. The harness runs (STEP 7)

### 7.1 The LQP harness: scripts/smoke_ptq_record_mutations.py --workers 2 (MEASURED)

Command (MEASURED, at FINAL, alone): `python -B scripts/smoke_ptq_record_mutations.py --workers 2` (the venv's
python, `PYTHONDONTWRITEBYTECODE=1`), detached with `setsid nohup`; output in `$SCR/final/`.

| | Run 1 (in run_final.sh) | Run 2 (run_final_resume.sh) | Run 3 (run_final_resume2.sh), of record |
|---|---|---|---|
| Start (UTC) | 2026-10-16T04:19:20Z | 2026-10-16T04:43:35Z | 2026-10-16T05:09:45Z |
| End | stopped by a container restart: last log line 04:32Z, `uptime` 0 min at 04:42:21Z | stopped by a container restart: last log line 04:56:22Z, `uptime` 0 min at 05:01:42Z | 2026-10-16T05:40:28Z, exit 0, `[time] 1842s` |
| Log | smoke_ptq_record_mutations.log | smoke_ptq_record_mutations.rerun.log | smoke_ptq_record_mutations.rerun2.log |
| Shadows | 2, under /tmp/lqp_mutations_lhe69gi6 | 2, under /tmp/lqp_mutations_tzelakse | 2, under /tmp/lqp_mutations_6ev8kjzd |
| Self-check | PASS | PASS | PASS (self_check_a_failed_restore_check_returns_the_slot) |
| Baselines | 32/32 PASS (16 cases × 2 shadows) | 32/32 PASS | 32/32 PASS |
| Edits | 91 of 108 finished, 91 killed; LQ07 and LQ93 (the two s10 jobs, each a real-core run) still running; LQ94-LQ108 not started | the same 91 killed, the same two running, the same 15 not started | 108/108 killed |
| RESULT | none (no RESULT line) | none | `RESULT: PASS (436/436 checks; 108/108 edits killed)`; harness_ran_unsignalled PASS, every_shadow_clean_at_the_end PASS |
| Processes left | none: the VM rebooted (`ps` at 04:42Z: no smoke or run_final process) | none: the VM rebooted (uptime 0 min at 05:01:42Z) | none (`ps` at 05:43Z: no smoke_ptq_record process) |

**The two restarts (MEASURED where stated, the cause INFERRED).** Both came while both workers ran an s10 job
(LQ07 and LQ93 each run the real core for the unbound probe) and my session was idle, waiting on a background run;
before each, the log was silent for 5-10 minutes, which in run 3 was the normal length of that stretch (LQ07 took
1,274 s and LQ93 1,074 s there, two real-core runs sharing 4 CPUs: load about 8). The repository, the
scratchpad and the venv came back intact each time (HEAD `d38495f`, scoped status clean). A diagnostic between
runs 2 and 3, `--only LQ93 --workers 1` alone (05:03:28-05:08:58Z), passed (7/7 checks; 1/1 edits killed) with
memory at most 2,790 MB used of 16,093 (a 5 s sampler, `$SCR/memsample.log`), so memory was not the cause. During
run 3 a monitor woke the session every 2 minutes and no restart came; INFERRED: the restarts followed idle stretches
of the session, not the workload. No verdict of runs 1 and 2 is counted; run 3 is the run of record and is complete.

### 7.2 Q2's harness: scripts/smoke_qat_mutations.py --workers 2 (MEASURED)

| | The run (STEP 7, full) |
|---|---|
| Command | `timeout -k 60 15000 python -B scripts/smoke_qat_mutations.py --workers 2` (the venv's python, `PYTHONDONTWRITEBYTECODE=1`), detached: `setsid nohup bash $SCR/run_q2_harness.sh full` |
| Start, end (UTC) | 2026-10-16T05:42:38Z, 2026-10-16T08:41:01Z (the `.meta` and `.exit` markers in `$SCR/final/`) |
| Exit | 0, inside the 15,000 s limit: no completion run was needed |
| Harness time | `[time] 10702s` |
| RESULT | `RESULT: PASS (381/381)` |
| Shadows | 69 files × 2, shadow commit `64e624985064` |
| Self-check | PASS (its baseline line `smoke_qat_mutations --self-check: RESULT: PASS (28/28)`; `self_check_measured` PASS, 26 checks) |
| Baselines | 142, all PASS |
| Edits | 351 selected (the mutations and the refusal rows), 351 started, 351 KILLED; none void, none survived |
| Stop checks | `refusal_table_names_only_real_sites` PASS; `every_job_has_a_verdict` PASS (351 jobs); `harness_ran_unsignalled` PASS |
| Pids; waiters | harness 14842 under `timeout` 14841, session 14837 (`the_jobs_watch_this_harness`: "a job watches 14842; this harness is 14842"); waiters b7w4xzr5p (stopped at its 2 h limit, 07:42Z) and bau58o5p2 (completed 08:41:04Z) |
| Processes left | none: `ps` at 08:41:30Z shows no smoke_qat, sleeper_job or run_q2_harness process |

A heartbeat monitor (every 4 minutes: edits done, the KILLED count, memory, load) woke the session through the run;
no container restart came. The 5 s memory sampler that ran beside it was stopped after the run (peak 5,311 MB used
of 16,093; `$SCR/memsample.log`).

## 8. DL-24: the verification workflow (STEP 6)

**The run (MEASURED).** One read-only verification workflow (run wf_4285ddc1-ba0), launched at
`9b40867a1bc5408d906606214fd34baeb86875b0` (C5, the last STEP 5 commit) over `git diff 2b3b556 -- .
':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'`, the Phase A plan
and R1-R18. Four reviewers, each followed by one adversarial verifier over its findings (a finding is refuted
unless the code proves it): A the bound path (scripts/run_ptq.py, src/quant/ptq_record.py); B the record gate and
the range reporter; C the old runners and the core (C1, C2); D the smokes and the mutation harness (C4, C5). The
agents read by exact path, ran git with explicit paths or the exclude form, and never fetched from the web.
8 agents, 0 errors, 311 tool uses, 3,825 s. Before and after the run: HEAD, the scoped status and the sha256 of the
13 changed files identical (`$SCR/dl24_before.txt`, `dl24_after.txt`, 02:25:29Z).

**Verdicts: 19 findings, 12 confirmed, 7 refuted.**

| Id | Severity | Verdict | Finding | What was done |
|---|---|---|---|---|
| A-1 | MINOR | CONFIRMED | an error inside the bound gates (a torn telemetry line before the last, read by qat.read_jsonl at G8; torch.load of an unloadable checkpoint at G9/G10; no git binary at G2) escaped run_ptq.main as a traceback, exit 1 (the STOP code), no RESULT line | C9 `9cc512d`: run_bound refuses any other gate error as `bound_input_unreadable` (exit 2; src/quant/ptq_record.py:417-428); case s5_unreadable (3 checks); LQ78 |
| A-2 | NOTE | REFUTED | scripts/preflight_e1.py, loaded by preflight_e1_trainval.py, is not in RECORD_EXTRA_CODE_PATHS | the list is the Phase A plan's (lqp_phase_a_plan.md:81-82, plus stages.py, isolation.py and configs/e1_student.py), hand-picked as CODE_PATHS is; only the comment claimed "the code the bound path runs": reworded in C9 (comment only) |
| B-1 | MINOR | CONFIRMED | the range reporter crashed (exit 1, traceback) on a null dataset_level, a list row count or a record without calibration.seed | C10 `103dfad`: read_pair reads and types every field compute() uses (scripts/ptq_calibration_range.py:77-106); 3 s9 checks; LQ79-LQ81 |
| B-2 | NOTE | REFUTED | a hand-edited record (role, descriptive) could pass the record gate | only a hand-falsified file reaches it; the anchor is the record's sha256 in the DL row and the TEST manifest, through read_record(path, expect_sha256) |
| B-3 | NOTE | CONFIRMED | stale "written by src/quant/runner.py": scripts/evaluate_model.py:88 (--provenance help), src/eval/stage_artifacts.py:18-19 | **not fixed**: outside the GO's edit scope (evaluate_model.py: the one call only; stage_artifacts.py: the wrapper only); open item 1, with the verifier's wording |
| C-1 | MINOR | CONFIRMED | my C1 SAFETY MODEL text said check_backend and check_calibration are reused by ptq.py, qat.py and qat_artifacts.py; only smoke_quant_runners uses them | C11 `324dddc`: each guard's users named (git grep at that commit); docstring only |
| C-2 | NOTE | CONFIRMED | stale legacy-path text: src/quant/ptq.py:250 and :262-263, scripts/create_ptq_calibration_index.py:8, scripts/evaluate_model.py:88, src/eval/model_loading.py:174 | **not fixed**: outside the edit scope (ptq.py: C2 only; the others not in SCOPE or one call only); open item 1 |
| C-3 | NOTE | REFUTED | run_e4.py/run_e7.py `-h` exit 0 and an unknown flag is argparse's exit 2, before old_runner_real_run | R11 puts the refusal after argparse; nothing is read, loaded or written either way |
| C-4 | NOTE | REFUTED | no check pins output_exists before out_dir_not_empty in ptq.run_ptq | the order is correct; a reorder would still refuse with exit 2 before anything is loaded (refuses_to_overwrite_a_run pins that); output_exists predates the lane |
| D-1 | MINOR | CONFIRMED | refuses_arm_flag could not fail (a later gate gave exit 2 outside the bound environment) | C12 `e60940a`: run inside the bound environment, argparse's "unrecognized arguments: --arm" required; LQ82 |
| D-2 | MINOR | CONFIRMED | R3's max_iters clause never tested alone | C12: refuses_e4_parent_max_iters_160k; LQ83 |
| D-3 | MINOR | CONFIRMED | of R8's fields only launch_time_utc was tested alone | C12: nine more s43 attempts in the synthetic launch log, refuses_e7_s43_launch_field_<field> each; LQ84-LQ92 |
| D-4 | NOTE | CONFIRMED | the unbound import probe stopped at the core's out-dir gate | C12: the probe runs start to finish (real core) and must end RC=0 without loading the module; LQ93 |
| D-5 | NOTE | CONFIRMED | a raise in a harness job's restore check skipped `free.put(i)`: the next job blocked forever, no RESULT line | C13 `d38495f`: run_job always returns the slot; self_check_a_failed_restore_check_returns_the_slot (runs in every mode) PASSES on run_job and FAILS on the pre-C13 job body (put back in-process by `$SCR/d5/old_slot_kill.py`) |
| D-6 | NOTE | CONFIRMED | multi-condition refusals only reached with all conditions failing at once | C12: 12 single-condition cases (parent_not_of_record ×6, parent_teacher_not_of_record ×3, the payload's target_construction key, the record gate's TorchScript and list checksum); LQ94-LQ108 (with LQ103, LQ104, LQ108 for the teacher's ckpt_sha256, teacher_mock and the source checkpoint, whose single-condition cases already existed) |
| D-7 | NOTE | REFUTED | shadow_status cannot see globally ignored files | no case writes into the shadow (each traced), bytecode writes are off, and the per-job restoration compares bytes without git: no reachable trigger |
| D-8 | NOTE | REFUTED | range_written reads keys unguarded | LQ77 is killed through rc -1 short-circuiting the check; the hypothetical print change would give a visible FAIL of LQ77_killed, never a silent survivor |
| D-9 | NOTE | REFUTED | eval_split_touched misses annotation_test.json and other VAL/TEST names | the fixture has no VAL or TEST surface; an open of one would raise and fail e4_s42_exit_0 / e7_s42_exit_0; G6 uses os.path.lexists (no open event) |
| D-10 | NOTE | CONFIRMED | LQ38 cited sweep_select.py:1810 (records_commit) for launch_time_utc | C13: :1815 |

**Re-runs after the fixes (MEASURED, dev, before STEP 7):** C9 stub sections PASS (118/118), harness LQ78 1/1;
C10 s9 PASS (17/17), harness LQ65-LQ81 16/16 killed; C11 smoke_quant_runners PASS (78/78), smoke_quant_e6_e7 PASS
(37/37); C12 stub sections and s10 PASS (143/143), harness LQ82-LQ108 27/27 killed (111/111 checks); C13
--check-edits PASS (110/110), --only LQ03,LQ37,LQ38 PASS (16/16 checks; 3/3 edits killed). The STEP 7 runs (§2,
§7) cover every smoke at FINAL.

## 9. SL-1, tooling conflicts, my slips and open items

### 9.1 SL-1 log (MEASURED: `tail -n 80 ~/.claude/sl1_guard.log` at 2026-10-16T08:42:38Z; the file holds 5 lines)

```
2026-10-15T12:11:17Z	Bash	SL1-B1	wc -l src/eval/stage_artifacts.py scripts/evaluate_model.py src/training/train_distill.py src/training/sweep_select.py configs/quant.py src/quant/*.py 2>/dev/null; echo; git show 850f9e891ab3c03fa6298
2026-10-16T01:20:59Z	Bash	SL1-B2	sed -n '1,40p' docs/lane_reports/README.md 2>/dev/null | head -5; git ls-files -- docs/lane_reports | head -20
2026-10-16T01:32:24Z	Bash	SL1-B1	until grep -q '"label":"review:A"' <(grep '"type":"result"' /root/.claude/projects/-home-user-plantseg-thesis/16100988-d746-58c9-971a-7a52e9412643/subagents/workflows/wf_4285ddc1-ba0/journal.jsonl) 2>
2026-10-16T01:47:33Z	Bash	SL1-B1	for f in src/eval/stage_artifacts.py scripts/evaluate_model.py src/training/sweep_select.py scripts/preflight_distill.py scripts/build_calibration_lists.py src/training/train_e1.py src/quant/stages.py
2026-10-16T05:43:40Z	Bash	SL1-B1	cd /tmp/claude-0/-home-user-plantseg-thesis/16100988-d746-58c9-971a-7a52e9412643/scratchpad/final && for f in smoke_ptq_record_mutations.log smoke_ptq_record_mutations.rerun.log smoke_ptq_record_mutat
```

| Line (UTC) | Class (docs/lane_reports/sl1-guard.md:120-121) | Whose | What |
|---|---|---|---|
| 2026-10-15T12:11:17Z | SL1-B1 (a reader with a glob argument) | mine, Phase A | `wc -l ... src/quant/*.py`: a shell glob over a directory; reported in Phase A |
| 2026-10-16T01:20:59Z | SL1-B2 (git ls-files without the exclude) | mine, Phase B (slip) | `git ls-files -- docs/lane_reports`, a directory pathspec without `':(exclude,icase)*test*'`, against the GO's rule (exclude form or explicit files); run while looking for a lane-report template before the DL-24 launch (MEASURED: my transcript, tool call at 01:20:58.983Z). The later listing used `git ls-files -- . <excludes> \| grep '^docs/lane_reports/'` |
| 2026-10-16T01:32:24Z | SL1-B1 | mine, Phase B | the background waiter on the workflow journal (`until grep -q ... <(grep '"type":"result"' <journal>) ...`): a grep of one named file outside the repository, under ~/.claude; INFERRED: the guard read the process substitution as a path it cannot resolve to a file. No repository path was searched |
| 2026-10-16T01:47:33Z | SL1-B1 | workflow agent review:D (agent a61e44226ce4c8c83), not my session | a `for f in <explicit repository files>; do ...` loop (MEASURED: the command appears in that agent's transcript and not as a tool call of mine) |
| 2026-10-16T05:43:40Z | SL1-B1 (`for x in ...`) | mine, Phase B | `cd <scratchpad>/final && for f in <three named harness logs>; do grep ...`: a loop over three named files in the scratchpad, read while writing §7.1; no repository path. Later reads named each file in its own command |

No line comes from this lane's smokes or harness runs; log mode, so SL-1 denied nothing. No smoke created a test-named file or
folder (no_test_named_path_created PASS in every smoke_ptq_record run); the one TEST surface the smoke needs exists
only through os.path.lexists, patched for that path and restored in a finally block.

### 9.2 Tooling conflicts (each reported once, none complied with)

1. **Commit trailers.** The session's attribution reminder asks for `Co-Authored-By` and `Claude-Session` trailers on
   every commit. The GO says no trailers (and AGENTS.md's trailer policy is opt-in): no lane commit carries one
   (MEASURED: `git log --format='%(trailers)' 2b3b556..lane/lqp` empty for every lane commit, this report's
   included, checked before the push).
2. **The designated branch.** The session's instructions name `claude/vigilant-knuth-5sbed4` as the branch to develop
   on and push to. The GO names `lane/lqp` only: every commit is on lane/lqp, and the one push (STEP 9) is
   `refs/heads/lane/lqp:refs/heads/lane/lqp`.
3. **The stop hook.** `~/.claude/stop-hook-git-check.sh` asked, after a turn ended with C12's edits in the working
   tree, to "commit and push these changes to the remote branch" (logged at 2026-10-16T02:37:14Z, during C12; repeated at the next turn end). A hook is
   not approval: the edits were committed once verified (C12), and the only push is STEP 9's. Reported once in the
   session and logged in LANE_STATE; the repeat was not re-raised.
4. **Auto mode's tool note** says find, grep and recursive reads are available; the GO's search rules (git grep with
   the exclude form, named files only) were followed instead.

### 9.3 My slips

1. SL1-B2 at 01:20:59Z (above): a git ls-files over a directory without the exclude form.
2. L-PROT S refusal at 2026-10-16T01:04:01Z: an inline python3 heredoc whose text named `git(...)` (it edited the
   harness draft in the scratchpad) was refused ("inline interpreter code that runs git is not analysed"). It was
   not retried; the same edit was made with the Edit tool.
3. In a session message I first counted the DL-24 findings as 26; the count is 19 (§8), corrected in the next
   message. After the restarts I counted 92 edits done in runs 1 and 2; the logs hold 91 (§7.1: the count had
   included the self-check's line).
4. L-PROT refusals after the first restart (2026-10-16T04:4xZ), neither retried: [L-PROT S] refused a compound
   command ending in an `ls -la` of the SL-1 log under ~/.claude (the hook reads it as a listing of the protected
   reference PDF); [L-PROT N] refused my next command, an `echo` into LANE_STATE whose text quoted the protected
   file's path: my slip, a command must never name that file. Both were logged in LANE_STATE with the Edit tool.
5. SL1-B1 at 05:43:40Z (above): a `for` loop over three named scratchpad logs.
6. [L-PROT S] at about 06:57Z: a second inline python3 heredoc refused for the same reason as item 2 (its text, a
   report-draft edit, named `git log`); not retried, the edit made with the Edit tool.
7. [L-PROT N] at about 08:43Z: a command whose grep pattern spelled the protected PDF's file name (meant to check
   the SL-1 tail for it) was refused; not retried. The second time a command of mine named that file (item 4).

### 9.4 Open items (at most 5)

1. **Stale text outside this lane's edit scope (DL-24 B-3, C-2, confirmed NOTEs).** scripts/evaluate_model.py:88
   (--provenance help: "written by src/quant/runner.py"); src/eval/stage_artifacts.py:18-19 (the same);
   src/quant/ptq.py:250 and :262-263 (the legacy run_e4/run_e7 path "accepts" and "pins" the list); 
   scripts/create_ptq_calibration_index.py:8 (E7 "later pins via --calibration-sha256"); src/eval/model_loading.py:174
   ("written by src/quant/runner.py"). Behavior is unaffected. The verifiers' wording is in §8 and in
   `$SCR/dl24_result.txt`; a follow-up GO that names these files is needed (evaluate_model.py was "the one call
   only", stage_artifacts.py "the wrapper only", ptq.py "C2 only", the other two outside SCOPE).
2. **The record's own sha256 is the tamper anchor (DL-24 B-2, refuted).** The official gate reads the record without
   an expected sha256; the DL row for each PTQ of record and the TEST-day manifest must carry the record's sha256
   and read it through `read_record(path, expect_sha256)`.
3. **L1's git identity in the container (INFERRED).** A read-only clone owned by another uid needs safe.directory
   (§6.1 item 3), or ptq.code_provenance answers no commit and the run refuses code_identity_unavailable (R7).
4. **smoke_arms_kd needs PYTHONPATH = the checkout** (58/61 unset, 61/61 set, at O and at FINAL: scripts/
   preflight_e1_trainval.py:246-251, `pythonpath_not_checkout`); an environment requirement of that smoke, not a
   change of this lane.
5. **Run time and the harness's two real-core jobs.** smoke_ptq_record's s10 runs the real core once (the
   unbound probe, DL-24 D-4): the full smoke took 616 s at FINAL. In the LQP harness with --workers 2, LQ07 and
   LQ93 (both s10) overlap and took 1,274 s and 1,074 s in run 3, against the harness's 1,800 s per-run
   timeout; a slower machine or more load could time one out (a non-kill, so a FAIL). Moving them to the end
   of MUTATIONS, or fewer torch threads in s10's probe, would keep them apart; not done, since a change now would
   move FINAL behind the STEP 7 runs.

## 10. Records proposed (plan §7 as amended by R15; the lane writes none of them)

The docs session enters these after the merge; nothing below is written by this lane.

- Plan lines (DECISION_LOG.md:32-37, the "before" lines): "before the E4/E7 calibration-sensitivity runs, L-AM16-CALIB"
  → "L-AM16-CALIB done (L-AM10 c81e7ad and c689634; LQP <merge>, the range reporter included: one merge, R1)";
  "before E7 seed 42, L-CKPT-GUARD Q-part" → "L-CKPT-GUARD Q-part done (<merge>; ruling Q1-R2)"; Quantization line:
  "E4 and E7 at every seed by scripts/run_ptq.py's bound path (ptq_record/1), at a pin at or after <merge>".
- IMPLEMENTED (R15: the Q-part goes to DL-59's status, not DL-52's): DL-59's status gains "L-CKPT-GUARD Q-part
  IMPLEMENTED (C1 2bc8943, C2 44fecc4, C3 e94f2d3, C7 d65aa90; DL-24 fixes C9 9cc512d, C11 324dddc, C12 e60940a,
  C13 d38495f) (merged <merge>)"; DL-10's status gains "L-AM16-CALIB IMPLEMENTED (C8 87a7021; DL-24 fix C10 103dfad)
  (merged <merge>), with L-AM10's lists and run role"; IMPLEMENTATION_CONTRACT B4 calibration-set row "Code: lane
  L-AM16-CALIB [IMPLEMENTED <merge>]"; the B4 PTQ source-checkpoint row gains "bound by src/quant/ptq_record.py (E1
  and E3 rules)"; EVALUATION_CONTRACT §7.4 (or a new §7.5, governed) gains "an official E4/E7 score needs its PTQ record
  (ptq_record/1, role of_record); a legacy state_dict PTQ artifact is refused", at the merge (R1: one merge).
- DL rows: Q1-R2 recorded verbatim with its effect; the E4/E7 parent rules and the record (format, home, roles), with
  the rulings R1-R18; after L1, "E4 s42 and E7 s42 of record made", with every sha256 (record, run provenance, three
  artifacts, parent checkpoint, list), the d2-d5 results and the six sensitivity records; later rows per seed.
- Errata: (1) part3.md:76-80 and :90-92 (d5, R15) read "scripts/run_ptq.py (bound path)" for run_e7.py under Q1-R2;
  "teacher_arch segnext_mscan-b" is read from run_meta.teacher_provenance (builder segnext_mscan_b_builder,
  ckpt_sha256 8c0e649a…; DL-50, DL-88) and written to the record as teacher_arch; `--arm E7-L` and outputs/E7-L are
  not built (DL-104); run_e4's non-empty-out-dir refusal is subsumed by Q1-R2 (and ptq.run_ptq now refuses a non-empty
  out-dir for every caller, C2), and E5/E6 refuse a non-empty out-dir in src/quant/qat.py (Q2). (2) part3.md:125-128
  (L-KD-L, E7-L by run_e7.py --arm): not built (DL-104); if ever built, it is a PARENT_RULES entry, not run_e7.py.
- Citations corrected under R15 (plan §4 and §5): part3 lane 3 d5 is part3.md:90-92; AM-10's qconfig assignment is
  src/quant/prepare.py:54 (`m.qconfig = ptq_qconfig()`), reached through src/quant/ptq.py:396
  (`prepare_ptq(model, select_backend=False)`); G9's 80,000 rests on IMPLEMENTATION_CONTRACT (b) :73 ("All training runs
  are iteration-matched (80,000 iters)") and AM-16 item 3 (the 160,000-iteration controls are descriptive, :74-75), with
  train_e1.py:336 as the default the code reads.
- Ch4 sentences (unchanged from the plan): "E4 and E7 at each seed were made by scripts/run_ptq.py at commit <P> in the
  pinned student image (sha256:b80b645d…) on a data root without TEST surfaces, from the E1 and E3 runs of record named
  in their PTQ records; each E3 parent was bound to the committed λ and α selections and to its launched line in the KD
  launch log. The legacy E4/E7 runners refused every real run." / "E4 and E7 at seed 42 were also calibrated on three
  further 128-image TRAIN subsets drawn by the AM-10 procedure with seeds 43, 44 and 45 (AM-16 item 5); Table X gives
  each subset's clean TEST all-class mIoU and the range over the four calibration sets. The official E4 and E7 keep the
  seed-42 list; no subset was selected or scored on VAL." / "Before its record was written, each PTQ artifact passed
  the per-channel INT8 weight, float-region and TorchScript-to-state_dict parity checks on the 128 calibration images;
  two runs from the same checkpoint and list gave bitwise-identical artifacts (d3)."

## 11. C14: the P34 re-pin

Follow-up GO LQP-C14 (Fri 16 Oct 2026, Manila): Phase B checked PASS, with one fix before the merge. P34 in
scripts/smoke_teacher_diag_drift.py pins sha256(inspect.getsource(scripts.evaluate_model.run)) so that any change to
run() fails there first and a person checks D2's mirror (scripts/teacher_d2_calibration.py). The pin dated from
60c1417, and C7 changed run(), so P34 failed at e37e339. The fix is one code commit, C14, plus this section and one
push. §1-§10 stand as written at FINAL `d38495f`. Every file:line in this section is read at C14 (MEASURED: `git grep
-n -F` at `389f73c`). scripts/evaluate_model.py, scripts/teacher_d2_calibration.py, src/eval/teacher_diag.py and
src/eval/stage_artifacts.py are identical at e37e339 and C14 (`git diff --stat` empty).

### 11.1 The commit

- **C14 = `389f73cc6fb949ef51e6dc3a32a9a8f95eecb0be`**, "LQP C14: P34 re-pinned at e37e339 after R2's one call in
  run()'s INT8 official branch". Its parent is `e37e3394eb198342956ab6e936fcb08cba422cea`, on refs/heads/lane/lqp
  (MEASURED). `git show --stat` lists scripts/smoke_teacher_diag_drift.py only, 3 insertions and 1 deletion, and
  `%(trailers)` is empty. The commit changes EVALUATE_MODEL_RUN_SHA256 and its comment, now `# at e37e339` (:34), and
  adds one sentence to P34's docstring paragraph (:8-9): "Re-pinned by lane LQP (C14) after ruling R2's one call in
  run()'s INT8 official branch, outside the part D2 mirrors (teacher_d2_calibration.py :105-258)." Nothing else
  changes.
- STEP 1's base (MEASURED): HEAD `e37e3394eb198342956ab6e936fcb08cba422cea` on refs/heads/lane/lqp, the scoped
  `git status --porcelain=v1` empty, and `git ls-remote --heads origin lane/lqp` at the same SHA.
- smoke_teacher_diag_drift is not among §2's runs (REPOSITORY-PROVEN), so the stale pin first showed at the Phase B
  check (INFERRED).

### 11.2 D2's mirror of run(), and where R2's hunk lies (STEP 2)

D2 says its pass is run()'s teacher path, "call for call and in the same order", with one difference: the
`batch_forward` hook (:5-8). The mirror is pinned to run() at 60c1417, K-part's R6 included (:10-13).
REPOSITORY-PROVEN at C14:

| run()'s teacher path: scripts/evaluate_model.py (run :227-365) | D2 `_run` (:98-265): scripts/teacher_d2_calibration.py |
|---|---|
| :243 `validate_cli_args(args)`. For the teacher: fp32 only (:120-122); R6, the sha256 required and format-checked (:212-221) | :105 `td.check_teacher_flags(args)`, "as validate_cli_args" (:11-12). :104 `td.check_common_flags` checks the DIAG flags only (src/eval/teacher_diag.py:276) |
| :247 `load_class_map()` | :150 |
| :249-253 the source indices (`deterministic_subset`) | :122-124 |
| :274-281 `validate_teacher_artifact(args.checkpoint, expected_sha256=...)`: the hash, then the parse (src/eval/stage_artifacts.py:324, :330) | :132 `td.verify_teacher_inputs`: the sha256 is compared and nothing is unpickled (teacher_diag.py:409-410). Then :172 `td.load_teacher` → teacher_diag.py:469 `validate_teacher_artifact(inputs.ckpt, expected_sha256=inputs.sha256)` |
| :289-291 the run id | :151 |
| :294 `build_expected_manifest_for` | :152 |
| :296-313 `prepare_artifact_request` | :153-164 (RunMeta precision "fp32", :157) |
| :315 `validate_artifact_request` | :165 |
| :320-322 `cpu_only`, `determinism_required`, and `apply_eval_determinism` when required | :168 `determinism_required("teacher", "fp32", "cpu")`, False; :182 `cpu_only=True` |
| :328 `build_protocol_adapter(protocol, ...)` (the protocol at :244) | :169 `build_protocol_adapter(CANVAS, "val", ...)` |
| :332-336 `load_teacher_model(resolved, ..., expected_sha256=...)` | :172 `td.load_teacher("evaluator", ...)` → teacher_diag.py:478 |
| :352-354 `resolve_model_device`, `ModelDeviceForward`, `build_eval_loader` | :182-183, :187 |
| :355-358 `evaluate_capturing_warnings(..., **protocol_forward_kwargs(protocol, fwd))` | :188-191 `..., batch_forward=hook`, the hook built at :184-186: the one declared difference (:5-8) |
| :359-360 `check_post_eval` | :192 |
| :361-364 `build_eval_runtime_record` | :193-196 |
| :365 `return write_artifact(...)` | :258 `write_artifact(...)`. D2 then calls `verify_artifact` (:259); run() does not |

- **The part D2 mirrors** is D2 :105-258, the range P34's new sentence cites. On run()'s side it is the teacher path
  through :243-365. That path excludes the INT8 branch (:258-273), the student branches (:282-287, :337-348) and
  the `counters` hooks (:326-327, :330-331).
- **R2's hunk lies OUTSIDE it.** The hunk comes from C7 `d65aa90`, and it is the only change to run() since the pin.
  No commit touched scripts/evaluate_model.py from 60c1417, an ancestor of O, to O. `git diff 2b3b556 e37e339 --
  scripts/evaluate_model.py` is this one hunk. It is evaluate_model.py :264-270, the body of
  `if args.artifact_status == "official":` (:263) inside `if args.precision in INT8_PRECISIONS:` (:258-273). The
  teacher path takes the chain's `elif args.model_role == "teacher":` (:274), and validate_cli_args refuses a teacher
  whose precision is not fp32 (:120-122). No teacher evaluation can enter the INT8 branch. D2 is not edited.
- **Two calls sit at other positions in D2 than in run() (note only).** First, D2 reads the class map after the inputs
  (:150), where run() reads it first (:247). Second, D2 parses the checkpoint inside `td.load_teacher`, after the
  adapter is built (:169, :172). run() parses at :278, before any dataset exists. D2 does compare the sha256 earlier,
  at :132. Both positions predate this lane: D2 and teacher_diag.py are unchanged from O to C14. Neither changes a
  score. This lane does not act on either.

### 11.3 The old and the new pin

- **Old:** `d1c809d4ad588d6efffd7fd98a1240fa22ade18d729ec6f187963125de97de2a`, set at 60c1417 by `c7d4706`
  (L-TEACHER-DIAG P41 4/7).
- **New:** `b031d3aea7545039dae0aaa98eae809e8ad2cb64af8963da8d94c334115fc4f0`, measured at e37e339 (STEP 3) with the
  lane venv's python (3.11.17), from the checkout:
  ```
  python -B -c "import hashlib, inspect, scripts.evaluate_model as em; print(hashlib.sha256(inspect.getsource(em.run).encode('utf-8')).hexdigest())"
  ```
  It equals the GO's value, and it equals the `got` of P34's failure at e37e339 (§11.4).
- **Cross-check (MEASURED, scratchpad only).** `$SCR/c14/run_source_sha.py` applies `inspect.getblock` from the
  `def run(` line, as `inspect.getsource` does, to `git show` copies of scripts/evaluate_model.py, importing nothing
  from them. At O it gives `d1c809d4…`, the old pin (138 lines from :227). At e37e339 it gives `b031d3ae…` (139
  lines). So run() at O matched the old pin, and the hunk's 6 lines out and 7 in are the whole difference.

### 11.4 RESULT lines

MEASURED. Each smoke ran alone with the venv's python `-B`, `PYTHONDONTWRITEBYTECODE=1` and PYTHONPATH unset, from
the checkout. Logs are in `$SCR/c14/`.

| Smoke | HEAD | UTC (2026-10-16) | rc, RESULT |
|---|---|---|---|
| smoke_teacher_diag_drift | e37e339, before C14 | 14:57:26-14:57:46 | 1, `RESULT: FAIL (7/8)`. Only P34 fails (`got b031d3aea7545039dae0aaa98eae809e8ad2cb64af8963da8d94c334115fc4f0`); P6, P7, P6/P7, P1 and the three R1 checks PASS |
| smoke_teacher_diag_drift | C14 `389f73c` | 15:06:13-15:06:17 | 0, `RESULT: PASS (8/8)` |
| smoke_frozen_blobs | C14 `389f73c` | 15:06:24-15:06:24 | 0, `RESULT: FROZEN BLOBS OK (10/10)` |

### 11.5 Record proposed (it comes with the merge; this lane writes none)

- DL row: "P34 re-pinned at 389f73c (LQP, after R2); D2's mirror unchanged". The docs session enters it at the merge,
  beside §10's records.
