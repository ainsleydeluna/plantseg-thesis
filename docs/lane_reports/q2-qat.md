# Lane Q2 · QAT runner (L-AM4 + L-AM1q): Phase B report

- GO: Phase B GO of Mon 5 Oct 2026 (18:15 Manila), the plan as amended by P1–P38, the OQ1–OQ9 answers and O1–O4.
- Branch `lane/q2-qat`, cut from `647d305274c439f30c1e087e9368100103ca1e7e` (PB-1a). Cloud lane session (DL-37).
- Tip before this report: `40c24e79cb500a14e6857ba59e3365b43c4563bd`; this report is the lane's last commit, and the push names its own SHA.
- Labels: MEASURED (run in this session), DOCUMENTED (the GO, the plan, the amendments), REPOSITORY-PROVEN
  (read in the code), INFERRED.

## 1. Commits and acceptance (g)

Commits on lane/q2-qat after the base, oldest first (MEASURED, `git log --format='%h %s' 647d305..HEAD`; no commit carries a trailer):

- `d1ee5cb Q2 C1: QAT trainer of record (AM-4/AM-4a recipe, L-AM1q seeding)`
- `6f7525e Convert and score every QAT epoch (lane 6, L-AM4 + L-AM1q, C2)`
- `f6bd582 Select the QAT epoch and the U4 clip of record (lane 6, L-AM4 + L-AM1q, C3)`
- `6a5fda9 Finalize the epoch of record and check a launch's run_meta (lane 6, L-AM4 + L-AM1q, C4)`
- `4614fbd Kill M1-M24 and every P8-P10, O2 and P23-P28 refusal with a named check (lane 6, L-AM4 + L-AM1q, C5)`
- `40c24e7 Resolve the DL-24 verification's confirmed findings (lane 6, L-AM4 + L-AM1q, C6)`

(g), MEASURED at `40c24e79cb500a14e6857ba59e3365b43c4563bd`:
```
git diff --stat 647d305274c439f30c1e087e9368100103ca1e7e -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
```
```
 configs/qat_selection_rules.json      |   46 ++
 configs/quant.py                      |   70 +-
 scripts/qat_epoch_eval.py             |  624 ++++++++++++++++
 scripts/run_e5.py                     |    3 +-
 scripts/run_e6.py                     |    3 +-
 scripts/select_clip.py                |   70 ++
 scripts/select_qat_epoch.py           |   72 ++
 scripts/smoke_qat_artifacts.py        |  776 ++++++++++++++++++++
 scripts/smoke_qat_mutations.py        |  861 ++++++++++++++++++++++
 scripts/smoke_qat_runner.py           | 1143 ++++++++++++++++++++++++++++++
 scripts/smoke_qat_seeding.py          |  188 +++++
 scripts/smoke_qat_selection.py        | 1258 +++++++++++++++++++++++++++++++++
 scripts/smoke_quant_e4_e5.py          |   13 +-
 scripts/smoke_quant_runners.py        |  208 +++---
 scripts/smoke_quant_x86_efficiency.py |   42 +-
 scripts/smoke_realrun_decisions.py    |  110 ++-
 scripts/synthetic_qat_fixtures.py     |  141 ++++
 src/quant/__init__.py                 |   12 +-
 src/quant/checkpoint.py               |   14 +-
 src/quant/prepare.py                  |   66 +-
 src/quant/ptq.py                      |   13 +-
 src/quant/qat.py                      | 1254 ++++++++++++++++++++++++++++++++
 src/quant/qat_artifacts.py            |  500 +++++++++++++
 src/quant/qat_select.py               |  483 +++++++++++++
 src/quant/runner.py                   |  245 +------
 src/quant/x86_latency.py              |   21 +-
 26 files changed, 7724 insertions(+), 512 deletions(-)
```
26 files, every one in SCOPE d (checked file by file against its list, §4); this report's own commit adds docs/lane_reports/q2-qat.md, also SCOPE d. None of evaluate_model.py, src/eval/, src/training/, src/data/, configs/distill.py, configs/e1_student.py, a K2 file, docs/ outside the report, AGENTS.md, CLAUDE.md, .claude/, requirements* or the 9 frozen files.

## 2. PB-1a, line references and the venv

### PB-1a (step 2) outputs
- Pre-checks: `git status --porcelain=v1 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'` printed nothing; `git ls-remote --heads origin lane/q2-qat` printed nothing.
- Recorded `git symbolic-ref HEAD`: `refs/heads/claude/lane-q2-qat-thesis-431dzr`.
- Chain `git branch lane/q2-qat 647d305274c439f30c1e087e9368100103ca1e7e && git symbolic-ref HEAD refs/heads/lane/q2-qat && git checkout --no-overlay HEAD -- . ':(exclude)docs/reference/reference.pdf'`: exit 0.
- Post-checks: `git rev-parse HEAD` → `647d305274c439f30c1e087e9368100103ca1e7e`; `git symbolic-ref HEAD` → `refs/heads/lane/q2-qat`; `git status --porcelain=v1 -- …` → nothing.

### Step 3: line shifts afd2d33 → 647d305 (MEASURED)
Q2's own files are byte-identical between afd2d33 and 647d305 (empty diff). Files the plan cites that moved:
- src/eval/stage_artifacts.py: :24→:34 INT8_ARTIFACT_FORMATS, :84→:95 validate_int8_artifact, :130→:141 projection flag, :135→:146 artifact_format, :146→:157 relative path, :161→:172 int8_official_calibration_error, :171→:182 PTQ-only test, :264→:293 resolve_evaluation_source.
- scripts/evaluate_model.py: :68→:69 build_parser, :100→:105 validate_cli_args, :150→:155 INT8 provenance + CPU, :197→:202, :206→:227 run(), :237→:258, :241→:262 resolve_evaluation_source, :242→:263 official check, :296→:319 cpu_only, :314→:338 load_int8_student, :343→:367 main.
- Unchanged: src/eval/model_loading.py :100, :153, :191, :210, :244, :255, :279 (teacher-only edits); src/eval/artifacts.py :131, :172, :234, :283, :565, :669 (STAGES gained A/F/G, no shift); src/quant/ptq.py 1079 lines with :429, :438, :441, :776, :853 as cited, compare_runs :1032–:1066, :1067–:1068 blank.
- scripts/select_lambda.py docstring :70–:74 (record at :69); scripts/select_alpha.py docstring :117–:125 (record at :115). E3's run_meta (train_distill :859+) carries stage, mode, seed, lambda_logit and alpha_cwd and no run_id key, so the parent's run_id is its run directory name (the sweep_select convention).
- New at the base: .claude/hooks/sl1_guard.{sh,py} (log mode); AGENTS.md rules 1 and 6 gained `':(exclude,icase)*test*'`.

### Step 4: the venv (MEASURED)
scratchpad/q2venv from /usr/bin/python3.11: Python 3.11.15, torch 2.1.0+cpu, torchvision 0.16.0+cpu, numpy 1.26.4, scipy 1.11.4, pillow 12.3.0 (the lock's pin; the lock pins torch/torchvision +cu121). Quantized engines: qnnpack, none, onednn, x86, fbgemm. 4 CPU threads. statsmodels is not installed (not in the GO's list), so smoke_quant_x86_efficiency's test_backend_parity cannot import src.stats.


## 3. Baseline (step 5) against the tip

MEASURED; each smoke run alone from the checkout root with the lane venv (`python -B scripts/<smoke>.py`), logs in the session scratchpad (`smoke_logs/baseline`, `smoke_logs/c6suite`). The tip column and the harness (§5) ran on C6's tree just before its commit, which holds the same bytes (MEASURED: the sha256 of C6's nine files, taken after the selection fix and before the harness and the selection re-run, equal the committed files, and `git diff --stat HEAD` over them is empty after the commit). The other smokes ran before that fix, which touched only scripts/smoke_qat_selection.py, a file none of them imports.

| smoke | baseline at 647d305 (step 5) | tip | note |
|---|---|---|---|
| smoke_frozen_blobs | RESULT: FROZEN BLOBS OK (10/10) | RESULT: FROZEN BLOBS OK (10/10) | unchanged |
| smoke_run_ptq | RESULT: FAIL (68/69) | RESULT: FAIL (68/69) | same count, same failing check (repository_untouched: configs/calibration is tracked since c689634); not Q2's, not repaired |
| smoke_calibration_lists | RESULT: FAIL (41/42) | RESULT: FAIL (41/42) | same count, same failing check (nothing_written_in_the_repository; same cause); not repaired |
| smoke_qnnpack_ops | RESULT: PASS (8/8) synthetic artifact | RESULT: PASS (8/8) synthetic artifact | unchanged |
| smoke_eval_int8 | RESULT: PASS (32/32) | RESULT: PASS (32/32) | unchanged |
| smoke_efficiency | RESULT: PASS (82/82) | RESULT: PASS (82/82) | unchanged |
| smoke_quant_e4_e5 | RESULT: PASS (58/58) | RESULT: PASS (58/58) | unchanged |
| smoke_quant_e6_e7 | RESULT: PASS (37/37) | RESULT: PASS (37/37) | unchanged |
| smoke_quant_runners | RESULT: PASS (77/77) | RESULT: PASS (77/77) | unchanged |
| smoke_realrun_decisions | RESULT: PASS (80/80) | RESULT: PASS (76/76) | P30's hunks: 80 → 76 (check names below) |
| smoke_quant_x86_efficiency | <no RESULT line> | <no RESULT line> | the full smoke cannot run in this venv: test_backend_parity imports src.stats, which needs statsmodels (not in the GO's install list); see the partial driver row |
| smoke_qat_runner | new | RESULT: PASS (196/196) | new (d1, d1 non-finite, gates, profile) |
| smoke_qat_seeding | new | RESULT: PASS (9/9) | new (d2) |
| smoke_qat_selection | new | RESULT: PASS (132/132) | new (d3, d5, finalize); the first C6 run gave 131/132, fixed and re-run (§9, note 12) |
| smoke_qat_artifacts | new | RESULT: PASS (68/68) | new (d4) |
| x86 partial driver (every section but test_backend_parity) | RESULT: PARTIAL PASS (72/72) without test_backend_parity | RESULT: PARTIAL PASS (73/73) without test_backend_parity | P30's test_runner_sidecar hunk |

## 4. Acceptance

| item | command | RESULT (MEASURED at the tip) | verdict |
|---|---|---|---|
| d1 (P1, P2, P5, P6, P12; O2's and P8–P10's gates; P14's profile) | `python -B scripts/smoke_qat_runner.py` | RESULT: PASS (196/196) | PASS |
| d2 (P13; num_workers 0 and 2) | `python -B scripts/smoke_qat_seeding.py` | RESULT: PASS (9/9) | PASS |
| d3, d5, finalize (P23–P28, O1) | `python -B scripts/smoke_qat_selection.py` | RESULT: PASS (132/132) | PASS |
| d4 (P15–P22, P34, P35) | `python -B scripts/smoke_qat_artifacts.py` | RESULT: PASS (68/68) | PASS |
| (g) | the diff --stat in §1 | 26 files; outside SCOPE d: none | PASS |
| (h) | `python -B scripts/smoke_qat_mutations.py --workers 2` | RESULT: PASS (233/233) | PASS |
| smoke_frozen_blobs 10/10 | `python -B scripts/smoke_frozen_blobs.py` | RESULT: FROZEN BLOBS OK (10/10) | PASS |
| smoke_run_ptq, smoke_calibration_lists, smoke_qnnpack_ops unchanged | the three smokes, as in §3 | RESULT: FAIL (68/69); RESULT: FAIL (41/42); RESULT: PASS (8/8) synthetic artifact (baseline: the same three lines) | PASS |
| every other baseline smoke passes, old and new counts (P30) | §3 and §8 (P30) | eval_int8, efficiency, e4_e5, e6_e7, quant_runners, realrun_decisions: PASS; x86: partial driver RESULT: PARTIAL PASS (73/73) without test_backend_parity | PASS |
| P38: the CUDA path exists, exercised on CPU where it can be | `python -B scripts/smoke_qat_runner.py --device cuda` on this CPU-only host | RESULT: ERROR --device cuda but this host has no CUDA device (the CPU run omits --device) (exit 4); the d1 sections run on the CPU through the same `run_case(device=...)` path | PASS |
| no "test" file or folder created; no script with "test" in its path run | §11 | Q2's code and smokes create none and no such script ran; smoke_run_ptq, which step 5 and this list require, creates `images/test` and `annotations/test` in its own temp tree, and smoke_quant_x86_efficiency imports src/stats/tests.py through src.stats until its statsmodels import fails (neither is Q2's file; both unchanged) | PASS, with the exception reported |
| nothing trains on real data; no checkpoint or dataset read | the environment of every smoke run | PLANTSEG_DATA_ROOT=unset; every smoke trains on synthetic fixtures in temp trees | PASS |

smoke_run_ptq and smoke_calibration_lists fail at the base already, with the same check (not Q2's; reported, not repaired): the GO's 69/69 and 42/42 read as "unchanged".

## 5. Mutations (acceptance (h)) and P36's refusal table

Command (MEASURED, at `40c24e79cb500a14e6857ba59e3365b43c4563bd`): `python -B scripts/smoke_qat_mutations.py --workers 2` →
`RESULT: PASS (233/233)` (4265s). The harness copies the 55 SHADOW_PATHS into temporary shadows committed at one
fixed commit, applies one edit at a time and runs the smoke holding the killing check in the shadow; a job is killed
only when that check reports FAIL, having passed on the unmutated shadow with the same arguments (DL-24's M1 fix). The
repository is never written. M1–M24 are the plan's and the GO's; X1–X17 undo the DL-24 fixes (X5 and X7 were
withdrawn with the refuted T7 and T5).

| # | mutation | file | killing check(s) | result |
|---|---|---|---|---|
| M1 | BN freeze one epoch late (epoch 11 -> 12) | src/quant/qat.py | d1_bn_frozen_from_step_21 | KILLED (d1_bn_frozen_from_step_21) |
| M2 | observer freeze one epoch late (epoch 13 -> 14) | src/quant/qat.py | d1_observers_off_from_step_25 | KILLED (d1_observers_off_from_step_25) |
| M3 | freeze steps as fractions round(0.65 T), round(0.70 T) | src/quant/prepare.py | d1_freeze_helper_335_gives_3350_4020 | KILLED (d1_freeze_helper_335_gives_3350_4020) |
| M4 | epoch tie goes to the later epoch (>=) | src/quant/qat_select.py | d3_tie_goes_to_earlier_epoch | KILLED (d3_tie_goes_to_earlier_epoch) |
| M5 | clip tie goes to 1.0 | src/quant/qat_select.py | d5_boundary_pair_ties_to_5 | KILLED (d5_boundary_pair_ties_to_5) |
| M6 | float arithmetic instead of Fraction | src/quant/qat_select.py | d5_exact_not_naive_float | KILLED (d5_exact_not_naive_float) |
| M7 | band 1/1000 -> 1/10000 | src/quant/qat_select.py | d5_boundary_pair_ties_to_5 | KILLED (d5_boundary_pair_ties_to_5) |
| M8 | select on the fake-quant VAL score | src/quant/qat_select.py | d3_decoy_converted_wins | KILLED (d3_decoy_converted_wins) |
| M9 | TRAIN loader built without seed=--seed | src/quant/qat.py | d2_seed43_different_order, d2_loader_seeded_with_run_seed | KILLED (d2_loader_seeded_with_run_seed) |
| M10 | set_seed removed | src/quant/qat.py | d2_set_seed_applied, d2_seed42_identical_4step_losses | KILLED (d2_set_seed_applied, d2_seed42_identical_4step_losses) |
| M11 | observers left on during VAL | src/quant/qat.py | val_leaves_state_unchanged | KILLED (val_leaves_state_unchanged) |
| M12 | T_max = steps_per_epoch | src/quant/qat.py | d1_lr_all_steps_match_closed_form | KILLED (d1_lr_all_steps_match_closed_form) |
| M13 | abort on a non-finite loss | src/quant/qat.py | nonfinite_loss_recorded_run_reaches_epoch_15_step27 | KILLED (nonfinite_loss_recorded_run_reaches_epoch_15_step27) |
| M14 | a patience path added | src/quant/qat.py | d1_no_patience_path_grep | KILLED (d1_no_patience_path_grep) |
| M15 | the VAL restore re-enables observers from epoch 13 | src/quant/qat.py | d1_observers_off_from_step_25 | KILLED (d1_observers_off_from_step_25) |
| M16 | the epoch checkpoint taken inside the VAL bracket | src/quant/qat.py | d1_checkpoint_observer_flags | KILLED (d1_checkpoint_observer_flags) |
| M17a | gradient clipping removed | src/quant/qat.py | d1_clip_applied_at_the_step | KILLED (d1_clip_applied_at_the_step) |
| M17b | gradient clipping after optimizer.step | src/quant/qat.py | d1_clip_applied_at_the_step | KILLED (d1_clip_applied_at_the_step) |
| M18a | momentum 0 | src/quant/qat.py | d1_optimizer_of_record | KILLED (d1_optimizer_of_record) |
| M18b | weight decay 0 | src/quant/qat.py | d1_optimizer_of_record | KILLED (d1_optimizer_of_record) |
| M19 | the checkpoint-to-provenance link dropped | src/quant/qat_select.py | d3_provenance_checkpoint_mismatch_refused | KILLED (d3_provenance_checkpoint_mismatch_refused) |
| M20a | a 64-sample summary accepted (a record summary's row-count and gt_support guards dropped) | src/quant/qat_select.py | d3_capped_summary_refused | KILLED (d3_capped_summary_refused) |
| M20b | an upstream-protocol summary accepted | src/quant/qat_select.py | d3_upstream_protocol_refused | KILLED (d3_upstream_protocol_refused) |
| M20c | a record summary's row counts not compared with the rules (one guard of M20a) | src/quant/qat_select.py | d3_capped_rows_alone_refused | KILLED (d3_capped_rows_alone_refused) |
| M20d | the gt_support sum not compared with the rules (the other guard of M20a) | src/quant/qat_select.py | d3_gt_support_sum_refused | KILLED (d3_gt_support_sum_refused) |
| M21 | the key-set comparison dropped | src/quant/qat_artifacts.py | d4_key_set_missing_key_refused | KILLED (d4_key_set_missing_key_refused) |
| M22a | the predicate reduced to 'all tensors finite' | src/quant/qat.py | d4_predicate_accepts_an_observed_state | KILLED (d4_predicate_accepts_an_observed_state) |
| M22b | the predicate reduced to parameters only | src/quant/qat.py | d4_predicate_flags_nonfinite_bn_with_finite_params | KILLED (d4_predicate_flags_nonfinite_bn_with_finite_params) |
| M23 | the AM-21 rejection removed | src/quant/qat_select.py | d5_nonfinite_candidate_rejected_a | KILLED (d5_nonfinite_candidate_rejected_a) |
| M24a | E6's λ/α binding removed at launch | src/quant/qat.py | gate_refuses_e6_parent_not_the_selected_run | KILLED (gate_refuses_e6_parent_not_the_selected_run) |
| M24b | E6's λ/α binding removed from check-run-meta | scripts/qat_epoch_eval.py | profile_stops_on_e6_parent_lambda | KILLED (profile_stops_on_e6_parent_lambda) |
| X1 | scheduler.step() moved before optimizer.step() (P1) | src/quant/qat.py | d1_lr_applied_at_each_optimizer_step | KILLED (d1_lr_applied_at_each_optimizer_step) |
| X2 | check-run-meta's repeat of the clip binding dropped (P10) | scripts/qat_epoch_eval.py | profile_pilot_checked_without_u4_pilot_stops, profile_stops_on_other_clip_selection | KILLED (profile_pilot_checked_without_u4_pilot_stops, profile_stops_on_other_clip_selection) |
| X3 | train mode not restored after a VAL pass that raised in a non-finite state (P12) | src/quant/qat.py | d1_val_error_recorded_and_train_mode_restored | KILLED (d1_val_error_recorded_and_train_mode_restored) |
| X4 | an exception before the run_meta row reported as an aborted run (P37) | src/quant/qat.py | main_error_before_telemetry_exit_4 | KILLED (main_error_before_telemetry_exit_4) |
| X6 | epoch_end rows not marked state_nonfinite (P12) | src/quant/qat.py | d1_nonfinite_rows_marked_step10 | KILLED (d1_nonfinite_rows_marked_step10) |
| X8 | a non-finite run rejected on its epoch_end rows only (AM-21 item 3) | src/quant/qat_select.py | d5_nonfinite_row_only_rejected | KILLED (d5_nonfinite_row_only_rejected) |
| X9 | a failed convert worker leaves its eval directory unspent (P26) | scripts/qat_epoch_eval.py | d4_convert_worker_refusal_spends_the_eval_dir | KILLED (d4_convert_worker_refusal_spends_the_eval_dir) |
| X10 | a failed evaluator leaves its eval directory unspent (P26) | scripts/qat_epoch_eval.py | d4_score_failure_spends_the_eval_dir | KILLED (d4_score_failure_spends_the_eval_dir) |
| X11 | P16's observer half dropped from the freeze cross-check | src/quant/qat_artifacts.py | d4_convert_stops_when_an_observer_moved_after_its_freeze | KILLED (d4_convert_stops_when_an_observer_moved_after_its_freeze) |
| X12 | the stored observer and fake-quant flags not checked (P15) | src/quant/qat_artifacts.py | d4_convert_epoch_stored_flags_stop | KILLED (d4_convert_epoch_stored_flags_stop) |
| X13 | the copy's dtype comparison dropped (P15; torch.equal ignores dtype) | src/quant/qat_artifacts.py | d4_convert_epoch_copy_dtype_stop | KILLED (d4_convert_epoch_copy_dtype_stop) |
| X14 | the checkpoint's identity not checked against the run (P15) | src/quant/qat_artifacts.py | d4_convert_epoch_identity_refused | KILLED (d4_convert_epoch_identity_refused) |
| X15 | the by-name copy's alias-key check dropped (P15) | src/quant/qat_artifacts.py | d4_alias_mismatch_stops | KILLED (d4_alias_mismatch_stops) |
| X16 | the run's qconfig fingerprint not compared at conversion (P15) | src/quant/qat_artifacts.py | d4_convert_epoch_qconfig_changed_stop | KILLED (d4_convert_epoch_qconfig_changed_stop) |
| X17 | select_clip compares the stored selection on the old eight keys only (P27) | src/quant/qat_select.py | d5_edited_selection_trace_refused | KILLED (d5_edited_selection_trace_refused) |

P36's refusal table: every refusal statement of P8–P10, O2 and P23–P28 (with the record helpers of convert, score,
finalize and check-run-meta they rely on), removed one at a time (the statement replaced by `pass`).

| refusal (file:function:code#k) | P item | killing check(s) | result |
|---|---|---|---|
| qat.py:resolve_parent:parent_dir_test_path#0 | P9 | refuses_parent_dir_test_path | KILLED (refuses_parent_dir_test_path) |
| qat.py:resolve_parent:parent_dir_missing#0 | P9 | refuses_parent_dir_missing | KILLED (refuses_parent_dir_missing) |
| qat.py:resolve_parent:expect_source_sha256_format#0 | P9 | refuses_expect_source_sha256_format | KILLED (refuses_expect_source_sha256_format) |
| qat.py:resolve_parent:parent_best_json_missing#0 | P9 | refuses_parent_best_json_missing | KILLED (refuses_parent_best_json_missing) |
| qat.py:resolve_parent:parent_best_json_format#0 | P9 | refuses_parent_best_json_not_json | KILLED (refuses_parent_best_json_not_json) |
| qat.py:resolve_parent:parent_best_json_format#1 | P9 | refuses_parent_best_json_format | KILLED (refuses_parent_best_json_format) |
| qat.py:resolve_parent:parent_checkpoint_test_path#0 | P9 | refuses_parent_checkpoint_test_name | KILLED (refuses_parent_checkpoint_test_name) |
| qat.py:resolve_parent:parent_checkpoint_missing#0 | P9 | refuses_parent_checkpoint_missing | KILLED (refuses_parent_checkpoint_missing) |
| qat.py:resolve_parent:parent_sha256_mismatch#0 | P9 | refuses_parent_sha256_mismatch | KILLED (refuses_parent_sha256_mismatch) |
| qat.py:resolve_parent:parent_records_missing#0 | P9 | refuses_parent_records_missing, refuses_e1_layout_for_e6 | KILLED (refuses_parent_records_missing, refuses_e1_layout_for_e6) |
| qat.py:resolve_parent:parent_run_meta_rows#0 | P9 | refuses_parent_run_meta_rows | KILLED (refuses_parent_run_meta_rows) |
| qat.py:resolve_parent:parent_run_meta_key#0 | P9 | refuses_parent_run_meta_without_mode | KILLED (refuses_parent_run_meta_without_mode) |
| qat.py:resolve_parent:parent_mode_not_real#0 | P9 | refuses_parent_mode_not_real | KILLED (refuses_parent_mode_not_real) |
| qat.py:resolve_parent:parent_seed_mismatch#0 | P9 | refuses_parent_seed_mismatch | KILLED (refuses_parent_seed_mismatch) |
| qat.py:resolve_parent:parent_stage_mismatch#0 | P9 | refuses_parent_stage_mismatch | KILLED (refuses_parent_stage_mismatch) |
| qat.py:resolve_parent:parent_incomplete#0 | P9 | refuses_parent_incomplete_e3_abort | KILLED (refuses_parent_incomplete_e3_abort) |
| qat.py:resolve_parent:parent_incomplete#1 | P9 | refuses_parent_incomplete_e3_no_run_end | KILLED (refuses_parent_incomplete_e3_no_run_end) |
| qat.py:resolve_parent:parent_stage_mismatch#1 | P9 | refuses_e1_parent_of_another_stage | KILLED (refuses_e1_parent_of_another_stage) |
| qat.py:resolve_parent:parent_run_meta_key#1 | P9 | refuses_parent_run_meta_key | KILLED (refuses_parent_run_meta_key) |
| qat.py:resolve_parent:parent_incomplete#2 | P9 | refuses_parent_incomplete_e1 | KILLED (refuses_parent_incomplete_e1) |
| qat.py:read_clip_selection:clip_selection_sha256_format#0 | P10 | refuses_clip_selection_sha_format | KILLED (refuses_clip_selection_sha_format) |
| qat.py:read_clip_selection:clip_selection_test_path#0 | P10 | refuses_clip_selection_test_path | KILLED (refuses_clip_selection_test_path) |
| qat.py:read_clip_selection:clip_selection_missing#0 | P10 | refuses_clip_selection_path_absent | KILLED (refuses_clip_selection_path_absent) |
| qat.py:read_clip_selection:clip_selection_missing#1 | P10 | refuses_clip_selection_file_missing | KILLED (refuses_clip_selection_file_missing) |
| qat.py:read_clip_selection:clip_selection_sha256_mismatch#0 | P10 | refuses_clip_selection_sha_mismatch | KILLED (refuses_clip_selection_sha_mismatch) |
| qat.py:read_clip_selection:clip_selection_format#0 | P10 | refuses_clip_selection_not_json | KILLED (refuses_clip_selection_not_json) |
| qat.py:read_clip_selection:clip_selection_format#1 | P10 | refuses_clip_selection_format | KILLED (refuses_clip_selection_format) |
| qat.py:read_clip_selection:clip_selection_format#2 | P10 | refuses_clip_selection_winner_not_a_candidate | KILLED (refuses_clip_selection_winner_not_a_candidate) |
| qat.py:clip_binding:grad_clip_norm_invalid#0 | P10 | refuses_clip_absent | KILLED (refuses_clip_absent) |
| qat.py:clip_binding:grad_clip_norm_not_candidate#0 | P10 | refuses_clip_not_candidate | KILLED (refuses_clip_not_candidate) |
| qat.py:clip_binding:u4_pilot_not_e5_s42#0 | P10 | refuses_u4_pilot_outside_e5_s42 | KILLED (refuses_u4_pilot_outside_e5_s42) |
| qat.py:clip_binding:u4_pilot_with_clip_selection#0 | P10 | refuses_u4_pilot_with_selection | KILLED (refuses_u4_pilot_with_selection) |
| qat.py:clip_binding:u4_pilot_required#0 | P10 | refuses_e5_s42_without_u4_pilot | KILLED (refuses_e5_s42_without_u4_pilot) |
| qat.py:clip_binding:clip_selection_required#0 | P10 | refuses_missing_clip_selection | KILLED (refuses_missing_clip_selection) |
| qat.py:clip_binding:clip_selection_winner_mismatch#0 | P10 | refuses_clip_not_the_winner | KILLED (refuses_clip_not_the_winner) |
| qat.py:read_tracked_selection:f'{what}_selection_missing'#0 | O2 | refuses_lambda_selection_missing, refuses_alpha_selection_missing | KILLED (refuses_lambda_selection_missing, refuses_alpha_selection_missing) |
| qat.py:read_tracked_selection:selection_sha256_format#0 | O2 | refuses_selection_sha_format | KILLED (refuses_selection_sha_format) |
| qat.py:read_tracked_selection:selection_path_not_repo_relative#0 | O2 | refuses_selection_absolute_path | KILLED (refuses_selection_absolute_path) |
| qat.py:read_tracked_selection:selection_test_path#0 | O2 | refuses_selection_test_path | KILLED (refuses_selection_test_path) |
| qat.py:read_tracked_selection:selection_missing_file#0 | O2 | refuses_selection_missing_file | KILLED (refuses_selection_missing_file) |
| qat.py:read_tracked_selection:selection_not_tracked#0 | O2 | refuses_selection_untracked | KILLED (refuses_selection_untracked) |
| qat.py:read_tracked_selection:selection_changed_since_head#0 | O2 | refuses_selection_changed_since_head | KILLED (refuses_selection_changed_since_head) |
| qat.py:read_tracked_selection:selection_sha256_mismatch#0 | O2 | refuses_selection_sha_mismatch | KILLED (refuses_selection_sha_mismatch) |
| qat.py:read_tracked_selection:selection_format#0 | O2 | refuses_selection_not_json | KILLED (refuses_selection_not_json) |
| qat.py:read_tracked_selection:selection_format#1 | O2 | refuses_selection_format, refuses_alpha_file_as_lambda | KILLED (refuses_selection_format, refuses_alpha_file_as_lambda) |
| qat.py:e6_parent_binding:parent_stage_mismatch#0 | O2 | refuses_e6_parent_not_e3 | KILLED (refuses_e6_parent_not_e3) |
| qat.py:e6_parent_binding:parent_seed_mismatch#0 | O2 | refuses_e6_parent_seed | KILLED (refuses_e6_parent_seed) |
| qat.py:e6_parent_binding:parent_lambda_mismatch#0 | O2 | refuses_e6_parent_lambda, gate_refuses_e6_parent_not_the_selected_run | KILLED (refuses_e6_parent_lambda, gate_refuses_e6_parent_not_the_selected_run) |
| qat.py:e6_parent_binding:parent_alpha_mismatch#0 | O2 | refuses_e6_parent_alpha | KILLED (refuses_e6_parent_alpha) |
| qat.py:e6_parent_binding:parent_run_id_mismatch#0 | O2 | refuses_e6_s42_parent_not_alpha_winner | KILLED (refuses_e6_s42_parent_not_alpha_winner) |
| qat.py:real_run_gates:real_run_flag#0 | P8 | gate_refuses_without_real_run | KILLED (gate_refuses_without_real_run) |
| qat.py:real_run_gates:confirm_real_run_flag#0 | P8 | gate_refuses_without_confirm | KILLED (gate_refuses_without_confirm) |
| qat.py:real_run_gates:config_pins#0 | P8 | gate_refuses_config_pins | KILLED (gate_refuses_config_pins) |
| qat.py:real_run_gates:expect_head_format#0 | P8 | gate_refuses_head_format | KILLED (gate_refuses_head_format) |
| qat.py:real_run_gates:seed#0 | P8 | gate_refuses_seed | KILLED (gate_refuses_seed) |
| qat.py:real_run_gates:num_workers#0 | P8 | gate_refuses_num_workers_0 | KILLED (gate_refuses_num_workers_0) |
| qat.py:real_run_gates:selection_not_applicable#0 | P8 | gate_refuses_selection_flags_for_e5 | KILLED (gate_refuses_selection_flags_for_e5) |
| qat.py:real_run_gates:out_dir#0 | P8 | gate_refuses_out_dir_in_repo, gate_refuses_out_dir_not_empty, gate_refuses_out_dir_test_path | KILLED (gate_refuses_out_dir_in_repo, gate_refuses_out_dir_not_empty, gate_refuses_out_dir_test_path) |
| qat.py:real_run_gates:cuda_required#0 | P8 | gate_refuses_cpu | KILLED (gate_refuses_cpu) |
| qat.py:real_run_gates:cuda_initialized_before_seed#0 | P8 | gate_refuses_cuda_initialised | KILLED (gate_refuses_cuda_initialised) |
| qat.py:real_run_gates:tf32_not_default#0 | P8 | gate_refuses_tf32_not_default | KILLED (gate_refuses_tf32_not_default) |
| qat.py:real_run_gates:backend_unavailable#0 | P8 | gate_refuses_backend_unavailable | KILLED (gate_refuses_backend_unavailable) |
| qat.py:real_run_gates:data_root_not_trainval_only#0 | P8 | gate_refuses_data_root_not_trainval_only | KILLED (gate_refuses_data_root_not_trainval_only) |
| qat.py:real_run_gates:expect_head_mismatch#0 | P8 | gate_refuses_head_mismatch | KILLED (gate_refuses_head_mismatch) |
| qat.py:real_run_gates:code_not_clean#0 | P8 | gate_refuses_code_not_clean | KILLED (gate_refuses_code_not_clean) |
| qat.py:load_source:e.code#0 | P8/P9 | load_source_refuses_projection_in_e6_student | KILLED (load_source_refuses_projection_in_e6_student) |
| qat.py:load_source:source_changed#0 | P8/P9 | load_source_refuses_changed_source | KILLED (load_source_refuses_changed_source) |
| qat.py:load_source:source_projection_keys#0 | P8/P9 | load_source_refuses_projection_keys | KILLED (load_source_refuses_projection_keys) |
| qat.py:run_qat:stage_not_qat#0 | P8 (+P2, P5 STOPs) | run_refuses_ptq_stage | KILLED (run_refuses_ptq_stage) |
| qat.py:run_qat:mode_invalid#0 | P8 (+P2, P5 STOPs) | run_refuses_mode | KILLED (run_refuses_mode) |
| qat.py:run_qat:real_run_test_hooks#0 | P8 (+P2, P5 STOPs) | run_refuses_test_hooks_in_real | KILLED (run_refuses_test_hooks_in_real) |
| qat.py:run_qat:clip_source#0 | P8 (+P2, P5 STOPs) | run_refuses_clip_source_in_real | KILLED (run_refuses_clip_source_in_real) |
| qat.py:run_qat:config_pins#0 | P8 (+P2, P5 STOPs) | run_refuses_config_pins | KILLED (run_refuses_config_pins) |
| qat.py:run_qat:grad_clip_norm_invalid#0 | P8 (+P2, P5 STOPs) | run_refuses_bad_clip | KILLED (run_refuses_bad_clip) |
| qat.py:run_qat:cuda_required#0 | P8 (+P2, P5 STOPs) | run_refuses_cpu_in_real | KILLED (run_refuses_cpu_in_real) |
| qat.py:run_qat:cuda_initialized_before_seed#0 | P8 (+P2, P5 STOPs) | run_refuses_cuda_initialised_in_real | KILLED (run_refuses_cuda_initialised_in_real) |
| qat.py:run_qat:out_dir_not_empty#0 | P8 (+P2, P5 STOPs) | run_refuses_nonempty_out_dir | KILLED (run_refuses_nonempty_out_dir) |
| qat.py:run_qat:steps_per_epoch#0 | P8 (+P2, P5 STOPs) | run_refuses_steps_per_epoch_in_real | KILLED (run_refuses_steps_per_epoch_in_real) |
| qat.py:run_qat:loader_of_record#0 | P8 (+P2, P5 STOPs) | run_refuses_loader_not_of_record_in_real | KILLED (run_refuses_loader_not_of_record_in_real) |
| qat.py:run_qat:empty_loader#0 | P8 (+P2, P5 STOPs) | run_refuses_empty_loader | KILLED (run_refuses_empty_loader) |
| qat.py:run_qat:backend_unavailable#0 | P8 (+P2, P5 STOPs) | run_refuses_unavailable_backend | KILLED (run_refuses_unavailable_backend) |
| qat.py:run_qat:unfused_batchnorm#0 | P8 (+P2, P5 STOPs) | run_refuses_unfused_bn_in_real | KILLED (run_refuses_unfused_bn_in_real) |
| qat.py:run_qat:optimizer_of_record#0 | P8 (+P2, P5 STOPs) | run_stops_on_an_optimizer_not_of_record | KILLED (run_stops_on_an_optimizer_not_of_record) |
| qat.py:run_qat:val_bracket_changed_state#0 | P8 (+P2, P5 STOPs) | run_stops_when_val_changes_the_state | KILLED (run_stops_when_val_changes_the_state) |
| qat_select.py:load_rules:rules_mismatch#0 | P23 | d3_rules_disagreeing_with_config_refused | KILLED (d3_rules_disagreeing_with_config_refused) |
| qat_select.py:_json:record_unreadable#0 | P24 | d3_unreadable_record_refused | KILLED (d3_unreadable_record_refused) |
| qat_select.py:epoch_selection:telemetry_sha256_mismatch#0 | P23-P26 | d3_telemetry_sha_mismatch_refused | KILLED (d3_telemetry_sha_mismatch_refused) |
| qat_select.py:epoch_selection:run_not_real#0 | P23-P26 | d3_smoke_run_refused | KILLED (d3_smoke_run_refused) |
| qat_select.py:epoch_selection:stage#0 | P23-P26 | d3_non_qat_stage_refused | KILLED (d3_non_qat_stage_refused) |
| qat_select.py:epoch_selection:stop_present#0 | P23-P26 | d3_stop_file_refused | KILLED (d3_stop_file_refused) |
| qat_select.py:epoch_selection:eval_record_missing#0 | P23-P26 | d3_missing_eval_record_exit_3 | KILLED (d3_missing_eval_record_exit_3) |
| qat_select.py:epoch_selection:eval_record_mismatch#0 | P23-P26 | d3_swapped_eval_dir_refused | KILLED (d3_swapped_eval_dir_refused) |
| qat_select.py:epoch_selection:purpose_not_record#0 | P23-P26 | d3_timing_purpose_refused | KILLED (d3_timing_purpose_refused) |
| qat_select.py:epoch_selection:smoke_inputs#0 | P23-P26 | d3_smoke_input_records_refused | KILLED (d3_smoke_input_records_refused) |
| qat_select.py:epoch_selection:eval_dir_mixed#0 | P23-P26 | d3_two_commits_refused | KILLED (d3_two_commits_refused) |
| qat_select.py:epoch_selection:eval_incomplete#0 | P23-P26 | d3_partial_eval_incomplete | KILLED (d3_partial_eval_incomplete) |
| qat_select.py:epoch_selection:chain_checkpoint#0 | P23-P26 | d3_row_checkpoint_mismatch_refused | KILLED (d3_row_checkpoint_mismatch_refused) |
| qat_select.py:epoch_selection:exclusion_inconsistent#0 | P23-P26 | d3_excluding_a_finite_epoch_refused | KILLED (d3_excluding_a_finite_epoch_refused) |
| qat_select.py:epoch_selection:score_missing#0 | P23-P26 | d3_unscored_epoch_exit_3 | KILLED (d3_unscored_epoch_exit_3) |
| qat_select.py:epoch_selection:scored_nonfinite_state#0 | P23-P26 | d3_scored_nonfinite_epoch_refused | KILLED (d3_scored_nonfinite_epoch_refused) |
| qat_select.py:epoch_selection:conversion_missing#0 | P23-P26 | d3_missing_provenance_exit_3 | KILLED (d3_missing_provenance_exit_3) |
| qat_select.py:epoch_selection:chain_provenance#0 | P23-P26 | d3_provenance_not_scored_one_refused | KILLED (d3_provenance_not_scored_one_refused) |
| qat_select.py:epoch_selection:conversion_missing#1 | P23-P26 | d3_missing_torchscript_exit_3 | KILLED (d3_missing_torchscript_exit_3) |
| qat_select.py:epoch_selection:eval_dir_mixed#1 | P23-P26 | d3_provenance_other_commit_refused | KILLED (d3_provenance_other_commit_refused) |
| qat_select.py:epoch_selection:chain_broken#0 | P23-P26 | d3_identity_sha_mismatch_refused, d3_provenance_checkpoint_mismatch_refused | KILLED (d3_identity_sha_mismatch_refused, d3_provenance_checkpoint_mismatch_refused) |
| qat_select.py:epoch_selection:score_missing#1 | P23-P26 | d3_missing_score_exit_3 | KILLED (d3_missing_score_exit_3) |
| qat_select.py:epoch_selection:score_artifact_invalid#0 | P23-P26 | d3_sha_mismatch_refused | KILLED (d3_sha_mismatch_refused) |
| qat_select.py:epoch_selection:chain_summary#0 | P23-P26 | d3_summary_not_scored_one_refused | KILLED (d3_summary_not_scored_one_refused) |
| qat_select.py:epoch_selection:eval_dir_mixed#2 | P23-P26 | d3_summary_commit_not_eval_commit_refused | KILLED (d3_summary_commit_not_eval_commit_refused) |
| qat_select.py:epoch_selection:chain_summary#1 | P23-P26 | d3_summary_identity_refused | KILLED (d3_summary_identity_refused) |
| qat_select.py:epoch_selection:evaluator_not_literal#0 | P23-P26 | d3_evaluator_not_literal_refused | KILLED (d3_evaluator_not_literal_refused) |
| qat_select.py:epoch_selection:summary_values#0 | P23-P26 | d3_capped_summary_refused, d3_non_val_split_refused, d3_upstream_protocol_refused, d3_smoke_status_refused | KILLED (d3_capped_summary_refused, d3_non_val_split_refused, d3_upstream_protocol_refused, d3_smoke_status_refused) |
| qat_select.py:epoch_selection:score_invalid#0 | P23-P26 | d3_score_outside_unit_interval_refused, d3_score_not_float_refused | KILLED (d3_score_outside_unit_interval_refused, d3_score_not_float_refused) |
| qat_select.py:epoch_selection:scores_dir_contents#0 | P23-P26 | d3_extra_scores_entry_refused | KILLED (d3_extra_scores_entry_refused) |
| qat_select.py:epoch_selection:no_convertible_epoch#0 | P23-P26 | d3_no_convertible_epoch_refused_exit_2 | KILLED (d3_no_convertible_epoch_refused_exit_2) |
| qat_select.py:epoch_selection:summaries_differ#0 | P23-P26 | d3_metric_impl_differs_across_summaries, d3_null_image_digest_everywhere_refused, d3_null_commit_everywhere_refused | KILLED (d3_metric_impl_differs_across_summaries, d3_null_image_digest_everywhere_refused, d3_null_commit_everywhere_refused) |
| qat_select.py:epoch_selection:winner_changed#0 | P23-P26 | d3_winner_changed_during_selection_refused | KILLED (d3_winner_changed_during_selection_refused) |
| qat_select.py:clip_selection:candidates#0 | P27/O1 | d5_one_candidate_refused | KILLED (d5_one_candidate_refused) |
| qat_select.py:clip_selection:not_a_pilot_run#0 | P27/O1 | d5_non_pilot_run_refused | KILLED (d5_non_pilot_run_refused) |
| qat_select.py:clip_selection:eval_record_missing#0 | P27/O1 | d5_rejected_candidate_without_conversion_record_exit_3 | KILLED (d5_rejected_candidate_without_conversion_record_exit_3) |
| qat_select.py:clip_selection:eval_record_mismatch#0 | P27/O1 | d5_rejected_candidate_record_mismatch_refused | KILLED (d5_rejected_candidate_record_mismatch_refused) |
| qat_select.py:clip_selection:purpose_not_record#0 | P27/O1 | d5_rejected_candidate_timing_record_refused | KILLED (d5_rejected_candidate_timing_record_refused) |
| qat_select.py:clip_selection:epoch_selection_missing#0 | P27/O1 | d5_missing_run_refused | KILLED (d5_missing_run_refused) |
| qat_select.py:clip_selection:epoch_selection_format#0 | P27/O1 | d5_selection_of_other_rules_refused | KILLED (d5_selection_of_other_rules_refused) |
| qat_select.py:clip_selection:selection_differs#0 | P27/O1 | d5_edited_epoch_selection_refused | KILLED (d5_edited_epoch_selection_refused) |
| qat_select.py:clip_selection:clip_values#0 | P27/O1 | d5_two_clip_1_runs_refused | KILLED (d5_two_clip_1_runs_refused) |
| qat_select.py:clip_selection:recipe_mismatch#0 | P27/O1 | d5_recipe_mismatch_refused, d5_cpu_model_difference_refused | KILLED (d5_recipe_mismatch_refused, d5_cpu_model_difference_refused) |
| qat_select.py:clip_selection:recipe_identity_null#0 | P27/O1 | d5_null_git_head_refused | KILLED (d5_null_git_head_refused) |
| qat_select.py:clip_selection:batch_order_differs#0 | P27/O1 | d5_batch_order_mismatch_refused | KILLED (d5_batch_order_mismatch_refused) |
| qat_select.py:clip_selection:pilot_evals_differ#0 | P27/O1 | d5_pilot_summary_commits_differ_refused, d5_pilot_host_labels_differ_refused | KILLED (d5_pilot_summary_commits_differ_refused, d5_pilot_host_labels_differ_refused) |
| qat_select.py:clip_selection:no_winner#0 | P27/O1 | d5_both_rejected_no_winner_exit_2_a | KILLED (d5_both_rejected_no_winner_exit_2_a) |
| qat_artifacts.py:refuse_test_path:f'{what}_test_path'#0 | SL-1 (P21) | d3_test_path_refused | KILLED (d3_test_path_refused) |
| qat_artifacts.py:read_run_record:telemetry_missing#0 | AM-19/P24 | d3_run_without_telemetry_refused | KILLED (d3_run_without_telemetry_refused) |
| qat_artifacts.py:read_run_record:telemetry_not_strict_json#0 | AM-19/P24 | d3_telemetry_not_strict_json_refused | KILLED (d3_telemetry_not_strict_json_refused) |
| qat_artifacts.py:read_run_record:run_meta_rows#0 | AM-19/P24 | d3_first_row_not_run_meta_refused | KILLED (d3_first_row_not_run_meta_refused) |
| qat_artifacts.py:read_run_record:epoch_end_duplicate#0 | AM-19/P24 | d3_duplicate_epoch_end_refused | KILLED (d3_duplicate_epoch_end_refused) |
| qat_artifacts.py:require_complete:run_incomplete#0 | AM-19/P24 | d3_incomplete_run_refused_exit_3 | KILLED (d3_incomplete_run_refused_exit_3) |
| qat_artifacts.py:verified_checkpoint:epoch_unknown#0 | P15/P24 | d4_convert_epoch_unknown_epoch_refused | KILLED (d4_convert_epoch_unknown_epoch_refused) |
| qat_artifacts.py:verified_checkpoint:checkpoint_name#0 | P15/P24 | d3_checkpoint_name_refused | KILLED (d3_checkpoint_name_refused) |
| qat_artifacts.py:verified_checkpoint:checkpoint_missing#0 | P15/P24 | d3_missing_checkpoint_exit_3 | KILLED (d3_missing_checkpoint_exit_3) |
| qat_artifacts.py:verified_checkpoint:checkpoint_sha256_mismatch#0 | P15/P24 | d3_checkpoint_bytes_changed_refused | KILLED (d3_checkpoint_bytes_changed_refused) |
| qat_artifacts.py:read_ts_identity:identity_missing#0 | P24 | d3_torchscript_without_identity_refused | KILLED (d3_torchscript_without_identity_refused) |
| select_qat_epoch.py:main:output_exists#0 | P25/P27 (one output) | d3_refuses_existing_output | KILLED (d3_refuses_existing_output) |
| select_clip.py:main:output_exists#0 | P25/P27 (one output) | d5_refuses_existing_output | KILLED (d5_refuses_existing_output) |
| qat_epoch_eval.py:cmd_convert_epoch:telemetry_sha256_mismatch#0 | P15/P24 | d4_convert_epoch_telemetry_sha_mismatch | KILLED (d4_convert_epoch_telemetry_sha_mismatch) |
| qat_epoch_eval.py:_epochs:epochs_with_record#0 | P21/P26 | d4_convert_record_takes_no_epochs, d4_score_record_takes_no_epochs | KILLED (d4_convert_record_takes_no_epochs, d4_score_record_takes_no_epochs) |
| qat_epoch_eval.py:_epochs:epochs_range#0 | P21/P26 | d4_convert_epochs_out_of_range | KILLED (d4_convert_epochs_out_of_range) |
| qat_epoch_eval.py:cmd_convert:expect_telemetry_sha256_required#0 | P24/P16 | d4_convert_record_requires_telemetry_sha | KILLED (d4_convert_record_requires_telemetry_sha) |
| qat_epoch_eval.py:cmd_convert:eval_dir_not_fresh#0 | P24/P16 | d4_convert_eval_dir_not_fresh | KILLED (d4_convert_eval_dir_not_fresh) |
| qat_epoch_eval.py:cmd_convert:telemetry_sha256_mismatch#0 | P24/P16 | d4_convert_telemetry_sha_mismatch | KILLED (d4_convert_telemetry_sha_mismatch) |
| qat_epoch_eval.py:cmd_convert:run_not_real#0 | P24/P16 | d4_convert_smoke_run_refused | KILLED (d4_convert_smoke_run_refused) |
| qat_epoch_eval.py:cmd_convert:freeze_cross_check#0 | P24/P16 | d4_convert_stops_when_a_freeze_did_not_take | KILLED (d4_convert_stops_when_a_freeze_did_not_take) |
| qat_epoch_eval.py:cmd_score:stop_present#0 | P22/P26 | d4_score_refuses_a_stop_file | KILLED (d4_score_refuses_a_stop_file) |
| qat_epoch_eval.py:cmd_score:convert_record_missing#0 | P22/P26 | d4_score_without_convert_record_exit_3 | KILLED (d4_score_without_convert_record_exit_3) |
| qat_epoch_eval.py:cmd_score:convert_record_mismatch#0 | P22/P26 | d4_score_convert_record_of_another_run | KILLED (d4_score_convert_record_of_another_run) |
| qat_epoch_eval.py:cmd_score:purpose_mismatch#0 | P22/P26 | d4_score_purpose_mismatch | KILLED (d4_score_purpose_mismatch) |
| qat_epoch_eval.py:cmd_score:convert_record_epochs#0 | P22/P26 | d4_score_record_needs_all_15_epochs | KILLED (d4_score_record_needs_all_15_epochs) |
| qat_epoch_eval.py:cmd_score:epoch_not_converted#0 | P22/P26 | d4_score_unconverted_epoch_refused | KILLED (d4_score_unconverted_epoch_refused) |
| qat_epoch_eval.py:cmd_score:scores_not_fresh#0 | P22/P26 | d4_score_scores_dir_not_fresh | KILLED (d4_score_scores_dir_not_fresh) |
| qat_epoch_eval.py:cmd_score:output_exists#0 | P22/P26 | d4_score_runs_once | KILLED (d4_score_runs_once) |
| qat_epoch_eval.py:_read_json:code#0 | P28 | finalize_selection_not_json | KILLED (finalize_selection_not_json) |
| qat_epoch_eval.py:_read_json:code#1 | P28 | finalize_selection_not_an_object | KILLED (finalize_selection_not_an_object) |
| qat_epoch_eval.py:_pilot_clip_winner:clip_selection_missing#0 | P28/P10 | finalize_clip_selection_file_missing | KILLED (finalize_clip_selection_file_missing) |
| qat_epoch_eval.py:_pilot_clip_winner:clip_selection_sha256_mismatch#0 | P28/P10 | finalize_checks_a_pinned_clip_selection_sha | KILLED (finalize_checks_a_pinned_clip_selection_sha) |
| qat_epoch_eval.py:_pilot_clip_winner:clip_selection_format#0 | P28/P10 | finalize_clip_selection_of_another_format | KILLED (finalize_clip_selection_of_another_format) |
| qat_epoch_eval.py:_pilot_clip_winner:clip_selection_winner_clip#0 | P28/P10 | finalize_refuses_an_edited_winner_clip | KILLED (finalize_refuses_an_edited_winner_clip) |
| qat_epoch_eval.py:_pilot_clip_winner:not_the_clip_winner#0 | P28/P10 | finalize_refuses_the_retained_loser | KILLED (finalize_refuses_the_retained_loser) |
| qat_epoch_eval.py:_pilot_clip_winner:clip_selection_stale#0 | P28/P10 | finalize_refuses_a_stale_clip_selection | KILLED (finalize_refuses_a_stale_clip_selection) |
| qat_epoch_eval.py:cmd_finalize:fresh_process_required#0 | P28 | finalize_fresh_process_guard | KILLED (finalize_fresh_process_guard) |
| qat_epoch_eval.py:cmd_finalize:run_not_real#0 | P28 | finalize_smoke_run_refused | KILLED (finalize_smoke_run_refused) |
| qat_epoch_eval.py:cmd_finalize:output_exists#0 | P28 | finalize_runs_once | KILLED (finalize_runs_once) |
| qat_epoch_eval.py:cmd_finalize:epoch_selection_missing#0 | P28 | finalize_without_epoch_selection_exit_3 | KILLED (finalize_without_epoch_selection_exit_3) |
| qat_epoch_eval.py:cmd_finalize:epoch_selection_format#0 | P28 | finalize_selection_of_other_rules_refused | KILLED (finalize_selection_of_other_rules_refused) |
| qat_epoch_eval.py:cmd_finalize:epoch_selection_mismatch#0 | P28 | finalize_selection_of_another_run_refused | KILLED (finalize_selection_of_another_run_refused) |
| qat_epoch_eval.py:cmd_finalize:smoke_inputs#0 | P28 | finalize_smoke_input_selection_refused | KILLED (finalize_smoke_input_selection_refused) |
| qat_epoch_eval.py:cmd_finalize:clip_selection_required#0 | P28 | finalize_pilot_requires_clip_selection | KILLED (finalize_pilot_requires_clip_selection) |
| qat_epoch_eval.py:cmd_finalize:clip_selection_not_applicable#0 | P28 | finalize_non_pilot_takes_no_clip_selection | KILLED (finalize_non_pilot_takes_no_clip_selection) |
| qat_epoch_eval.py:cmd_finalize:winner_checkpoint_changed#0 | P28 | finalize_winner_checkpoint_changed_refused | KILLED (finalize_winner_checkpoint_changed_refused) |
| qat_epoch_eval.py:cmd_finalize:winner_files_changed#0 | P28 | finalize_winner_artifact_changed_refused | KILLED (finalize_winner_artifact_changed_refused) |
| qat_epoch_eval.py:cmd_finalize:winner_files_changed#1 | P28 | finalize_state_dict_companion_changed_refused | KILLED (finalize_state_dict_companion_changed_refused) |
| qat_epoch_eval.py:cmd_finalize:x86_backend_unavailable#0 | P28 | finalize_refuses_a_non_x86_engine | KILLED (finalize_refuses_a_non_x86_engine) |
| qat_epoch_eval.py:cmd_finalize:x86_copy_checks#0 | P28 | finalize_stops_on_a_failed_x86_check | KILLED (finalize_stops_on_a_failed_x86_check) |
| qat_epoch_eval.py:cmd_check_run_meta:telemetry_missing#0 | P14/P10/O2 | profile_run_without_telemetry_exit_3 | KILLED (profile_run_without_telemetry_exit_3) |
| qat_epoch_eval.py:cmd_check_run_meta:run_meta_rows#0 | P14/P10/O2 | profile_first_row_not_json | KILLED (profile_first_row_not_json) |
| qat_epoch_eval.py:cmd_check_run_meta:run_meta_rows#1 | P14/P10/O2 | profile_first_row_not_run_meta | KILLED (profile_first_row_not_run_meta) |
| qat_epoch_eval.py:cmd_check_run_meta:run_meta_profile#0 | P14/P10/O2 | profile_stops_on_num_workers, profile_stops_on_lr | KILLED (profile_stops_on_num_workers, profile_stops_on_lr) |

## 6. DL-24 verification (step 7)

A read-only workflow (run `wf_547452fd-8f9`, 8 agents, 0 errors, 4,597 s): four reviewers (the trainer; the gates, the
harness and P30's hunks; conversion and scoring; selection and finalize) each read the lane's diff against 647d305 at
HEAD 6a5fda9 plus the uncommitted C5 smoke files, then four verifiers each tried to refute one reviewer's findings.
Hard limits passed to every agent: no reads under docs/reference/, no repository root or docs/ as a search path, the
SL-1 exclusion on every search over scripts or src, no script with "test" in its path, no writes.
Read-only (MEASURED, after the workflow): `sha256sum -c` of the 26 lane files hashed before it started: all OK;
`git status --porcelain=v1 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'`
identical to the status recorded before it; HEAD 6a5fda9 on refs/heads/lane/q2-qat.

35 findings, 26 confirmed by their verifier, 9 refuted. Every confirmed finding is fixed in C6; a refuted finding's
code change is not applied (four refuted findings, T3, T5, CS-9 and SEL-9, keep a test-only check). Labels are the
agents'.

| id | severity | item | verdict | resolution (C6) | check that fails without it |
|---|---|---|---|---|---|
| T1 | minor | P12 | confirmed | a VAL pass that raises in a non-finite state returns the model to train mode (qat.py `prepared.train()` in the VAL handler) | d1_val_error_recorded_and_train_mode_restored (X3) |
| T2 | minor | P37 | confirmed | an exception before the run_meta row prints `RESULT: ERROR`, exit 4; after it ABORTED, exit 3 (qat.py `main`) | main_error_before_telemetry_exit_4 (X4), main_abort_after_telemetry_exit_3 |
| T3 | minor | P1 | refuted (no defect) | test-only: the learning rate applied at each optimizer step is recorded and checked | d1_lr_applied_at_each_optimizer_step (X1) |
| T4 | nit | P12 | confirmed | epoch_end rows carry `state_nonfinite` | d1_nonfinite_rows_marked_step10 (X6) |
| T5 | nit | P5 | refuted | not applied (the rows keep the flags derived from the freeze epochs); test-only: d1 checks the rows' flags against the freeze epochs | d1_row_flags_match_the_freeze_epochs |
| T6 = G2 | minor | P7, P8 | confirmed | QAT_CODE_PATHS adds src/distill/export.py and the package `__init__` files the run imports | — (a constant; run_meta's code_files_sha256 carries every listed path) |
| T7 | nit | P37 | refuted | not applied (argparse's own usage error, exit 2, nothing written) | — |
| G1 | minor | P8 (SL-1) | confirmed | `names_test` reads the path as given and resolved; `--out-dir` is tested before it is listed; best.json's checkpoint name is tested | refuses_parent_checkpoint_test_name; the given-path half has no case (§11) |
| M1 | minor | P36, (h) | confirmed | the harness counts a kill only when the named check reports FAIL, each job baselined with its own argv; code_of and stop_of catch any exception | the harness's own `<id>_killed` and `<spec>_baseline` checks |
| M2 | minor | P36 (P24, P26, P10) | confirmed | the refusal table adds qat_artifacts' record helpers, convert-epoch and the selectors' mains; P10's profile repeat gets X2 | see the refusal table; X2 |
| B1 | nit | O2, P10 | refuted | not applied | — |
| P1 | nit | P14, P10 | confirmed | profile cases for the stage and for a clip_selection.json changed after launch | profile_stops_on_stage, profile_stops_on_clip_selection_file_changed |
| P2 | nit | P10 | confirmed | a clip-selection sha256 without its file is refused by name (`clip_selection_missing`) | refuses_clip_selection_path_absent, profile_stops_on_clip_selection_sha_without_file |
| CS-1 = SEL-5 | major | P34, plan §6 d4 | confirmed | d4 runs convert → score → select → finalize on its own records, as E5 (seed 43) and as E6; the evaluator stand-in takes SMOKE_GIT_HEAD | d4_e5_end_to_end_select_and_finalize, d4_e6_end_to_end_select_and_finalize |
| CS-2 | minor | P16 | confirmed | case for an observer buffer that moved after the epoch-13 freeze | d4_convert_stops_when_an_observer_moved_after_its_freeze (X11) |
| CS-3 | minor | P15 | confirmed | cases for the checkpoint's identity, its stored flags, a copy dtype change, the qconfig fingerprint and the alias-key check | d4_convert_epoch_identity_refused (X14), d4_convert_epoch_stored_flags_stop (X12), d4_convert_epoch_copy_dtype_stop (X13), d4_convert_epoch_qconfig_changed_stop (X16), d4_alias_mismatch_stops (X15) |
| CS-4 | minor | P26 | confirmed | a failed worker writes convert_STOP.json; a failed evaluator writes score_STOP.json; score refuses an existing scores/; both STOP files end selection | d4_convert_worker_refusal_spends_the_eval_dir (X9), d4_score_failure_spends_the_eval_dir (X10), d4_score_refuses_an_empty_scores_dir |
| CS-5 | minor | P26 | confirmed | workers and the evaluator get absolute paths; a worker that exits 0 without its output spends the eval directory, exit 4 | — (no case: it needs a worker that exits 0 without writing; the branch is the one CS-4's case runs) |
| CS-6 | minor | P26 | refuted | not applied (the GO asks no conversion-time cleanliness gate; Block B checks GOVERNED_DIRTY before every convert) | — |
| CS-7 | nit | P35 | confirmed | the controls are compared with `x is not None and x < 0.5` | d4_control_observers_not_carried_below_50pct, d4_control_bn_not_carried_below_50pct (the fix changes how 0.0 reads) |
| CS-8 | nit | P37 | confirmed | convert's RESULT word follows the worker's exit code | d4_convert_worker_refusal_spends_the_eval_dir |
| CS-9 | nit | P35 | refuted | test-only: d4 also prints the band-0 figures (below) | — |
| SEL-1 | minor | P28 (P10) | confirmed (the clip cross-check only) | finalize refuses a clip_selection.json whose winner clip is not the run's clip (`clip_selection_winner_clip`) | finalize_refuses_an_edited_winner_clip |
| SEL-2 | minor | O1 (P24) | confirmed | a rejected pilot run is anchored by its eval directory's qat_convert.json (run_id, telemetry sha256, purpose record) | d5_rejected_candidate_without_conversion_record_exit_3, d5_rejected_candidate_record_mismatch_refused, d5_rejected_candidate_timing_record_refused |
| SEL-3 | minor | O1 | confirmed | O1 cases: the rejected run holds the better selection, in both orientations, each clause alone | d5_nonfinite_candidate_rejected_a/b, d5_nonfinite_clip_1_rejected_a/b, d5_state_finite_false_alone_rejected, d5_nonfinite_since_step_alone_rejected, d5_nonfinite_row_only_rejected (X8) |
| SEL-4 | minor | P23 | confirmed | all-null digest and commit cases; one case per compared summary value | d3_null_image_digest_everywhere_refused, d3_null_commit_everywhere_refused, d3_precision_refused … d3_teacher_role_refused |
| SEL-6 | minor | P36 (P24) | confirmed | record and checkpoint cases; read_ts_identity in the table | d3_torchscript_without_identity_refused, d3_checkpoint_bytes_changed_refused and the d3 record cases |
| SEL-7 | nit | P23 | confirmed | the rules file lists the evaluation identity and purpose record; forward_batches is read from it | every selection case (load_rules refuses a rules file without the keys) |
| SEL-8 | nit | P27 | confirmed | host_identity's docstring says the CPU model is compared | — (text) |
| SEL-9 | nit | OQ5 (P27) | refuted | test-only: the no-tie mirror pair is registered | d5_just_above_not_tie_mirror |
| SEL-10 | nit | P28 | refuted | not applied | — |
| SEL-11 | nit | P28 | refuted | not applied | — |
| SEL-12 | nit | P27 | confirmed | select_clip compares the stored selection with its recomputation key by key (run_dir and eval_dir aside) | d5_edited_selection_trace_refused (X17) |


## 7. Where each patch item is implemented (file:line at `40c24e79cb500a14e6857ba59e3365b43c4563bd`)

MEASURED with a fixed list of code anchors (one literal string per line, resolved at the tip). P29 is an absence and P33
a report item; both are reported in §8.

```
P1   src/quant/qat.py:856                             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
P1   src/quant/qat.py:974                             scheduler.step()
P1   src/quant/qat.py:826                             _refuse("steps_per_epoch"
P1   scripts/smoke_qat_runner.py:205                  d1_lr_all_steps_match_closed_form
P2   src/quant/qat.py:854                             optimizer = torch.optim.SGD(
P2   src/quant/qat.py:958                             gnorm = torch.nn.utils.clip_grad_norm_(
P2   src/quant/qat.py:749                             def optimizer_record(
P2   src/quant/qat.py:761                             def optimizer_of_record_error(
P2   scripts/smoke_qat_runner.py:222                  d1_optimizer_of_record
P2   scripts/smoke_qat_runner.py:237                  d1_clip_applied_at_the_step
P3   src/quant/qat.py:233                             def config_pins_error(
P3   configs/quant.py:50                              "qat_real_run"
P3   src/quant/qat.py:1091                            p.add_argument("--grad-clip-norm"
P4   src/quant/qat.py:853                             criterion = CombinedCEDiceLoss(weight=weights
P4   src/quant/qat.py:890                             "objective": OBJECTIVE, "class_weights_sha256"
P5   src/quant/qat.py:932                             if epoch == BN_FREEZE_EPOCH + 1:
P5   src/quant/qat.py:934                             if epoch == OBS_FREEZE_EPOCH + 1:
P5   src/quant/qat.py:1005                            ck_path, ck_sha, ck_bytes = save_epoch_checkpoint(
P5   src/quant/qat.py:1016                            flags = observer_flags(prepared)
P5   src/quant/qat.py:1035                            raise QATStop("val_bracket_changed_state"
P5   src/quant/qat.py:1048                            tel.write(val_row)
P5   src/quant/qat.py:1059                            end_row = tel.write(end_row)
P6   scripts/smoke_qat_runner.py:246                  d1_bn_frozen_from_step_21
P6   scripts/smoke_qat_runner.py:248                  d1_observers_off_from_step_25
P6   scripts/smoke_qat_runner.py:267                  d1_checkpoint_observer_flags
P7   src/quant/qat.py:869                             run_meta = {
P7   src/quant/qat.py:895                             "never_observed_modules": never
P7   src/quant/qat.py:294                             def never_observed_modules(
P7   src/quant/qat.py:117                             "src/distill/__init__.py", "src/distill/export.py"
P8   src/quant/qat.py:1118                            def real_run_gates(
P8   src/quant/qat.py:1207                            def main(argv, stage_key
P8   src/quant/qat.py:147                             def names_test(
P9   src/quant/qat.py:463                             def resolve_parent(
P10  src/quant/qat.py:581                             def clip_binding(
P10  src/quant/qat.py:555                             def read_clip_selection(
P10  scripts/qat_epoch_eval.py:533                    # the clip and its source (P10)
P11  src/quant/qat.py:328                             def state_predicate(
P12  src/quant/qat.py:924                             def _mark_nonfinite(
P12  src/quant/qat.py:969                             "event": "step_error"
P12  src/quant/qat.py:1019                            val_error = None
P12  scripts/smoke_qat_runner.py:344                  nonfinite_loss_recorded_run_reaches_epoch_15_step
P12  src/quant/qat.py:1027                            prepared.train()                           # validate() returns
P12  src/quant/qat.py:994                             "state_nonfinite": nonfinite_since is not None
P13  src/quant/qat.py:693                             def batch_sha256(
P13  scripts/smoke_qat_seeding.py:8                   num_workers
P14  scripts/qat_epoch_eval.py:506                    def profile_mismatches(
P14  scripts/qat_epoch_eval.py:576                    def cmd_check_run_meta(
P15  src/quant/qat_artifacts.py:307                   def convert_epoch(
P15  src/quant/qat_artifacts.py:373                   def _convert_checked(
P15  src/quant/qat_artifacts.py:182                   def load_state_by_name(
P16  src/quant/qat_artifacts.py:231                   def freeze_cross_check(
P16  scripts/qat_epoch_eval.py:166                    cross = A.freeze_cross_check(rec)
P17  src/quant/qat_artifacts.py:262                   IDENTITY_KEYS = (
P18  src/quant/qat_artifacts.py:457                   prov = build_run_provenance(
P19  src/quant/ptq.py:1069                            def student_traced_in_process(
P20  src/quant/qat_artifacts.py:13                    synthetic_parity_inputs()
P21  scripts/qat_epoch_eval.py:70                     p.add_argument("--purpose"
P21  scripts/qat_epoch_eval.py:133                    def _host(
P22  scripts/qat_epoch_eval.py:228                    def evaluator_command(
P22  scripts/qat_epoch_eval.py:239                    def cmd_score(
P22  scripts/qat_epoch_eval.py:274                    sched_getaffinity
P23  configs/qat_selection_rules.json:9               "evaluator": {
P23  src/quant/qat_select.py:101                      def _check_summary(
P23  configs/qat_selection_rules.json:25              "equal_across_evaluations"
P24  src/quant/qat_select.py:129                      def epoch_selection(
P24  src/quant/qat_select.py:216                      "provenance qat_checkpoint_sha256"
P25  src/quant/qat_select.py:264                      raise QATRefused("scores_dir_contents"
P25  src/quant/qat_select.py:233                      s = verify_artifact(sdir)
P25  src/quant/qat_select.py:253                      raise QATRefused("score_invalid"
P26  src/quant/qat_artifacts.py:46                    NOT_CONVERTIBLE_RULE =
P26  src/quant/qat_select.py:266                      raise QATRefused("no_convertible_epoch"
P26  scripts/qat_epoch_eval.py:51                     SCORE_STOP = "score_STOP.json"
P26  scripts/qat_epoch_eval.py:200                    "code": "worker_failed"
P26  scripts/qat_epoch_eval.py:298                    write_exclusive({E / SCORE_STOP:
P27  src/quant/qat_select.py:360                      def clip_selection(
P27  src/quant/qat_select.py:330                      def _recipe(
P27  src/quant/qat_select.py:425                      raise QATRefused("batch_order_differs"
P27  src/quant/qat_select.py:353                      def clip_stats(
P27  src/quant/qat_select.py:86                       def selection_differences(
P28  scripts/qat_epoch_eval.py:372                    def cmd_finalize(
P28  scripts/qat_epoch_eval.py:341                    def _pilot_clip_winner(
P28  scripts/qat_epoch_eval.py:365                    raise QATRefused("clip_selection_winner_clip"
P30  scripts/smoke_realrun_decisions.py:14            import
P30  scripts/smoke_quant_x86_efficiency.py:406        def test_runner_sidecar(
P31  src/quant/checkpoint.py:1                        """
P32  src/quant/x86_latency.py:193                     _copy_trained_state =
P34  scripts/smoke_qat_artifacts.py:55                "e6"
P35  src/quant/qat_artifacts.py:269                   def argmax_agreement(
P35  src/quant/qat_artifacts.py:302                   def head_lsb(
P36  scripts/smoke_qat_mutations.py:137               MUTATIONS = [
P36  scripts/smoke_qat_mutations.py:327               KILLS = dict([
P37  src/quant/qat.py:1197                            def result_line(
P37  src/quant/qat.py:124                             EXIT_OK, EXIT_STOP, EXIT_REFUSED, EXIT_ABORTED, EXIT_ERROR = 0, 1, 2, 
P37  src/quant/qat.py:1237                            No run_meta row was written
P37  scripts/qat_epoch_eval.py:143                    def _result_word(
P38  scripts/smoke_qat_runner.py:683                  "--device"
O1   src/quant/qat_select.py:342                      def run_rejected(
O1   src/quant/qat_select.py:448                      raise QATRefused("no_winner", NO_WINNER)
O1   src/quant/qat_select.py:154                      raise QATIncomplete("eval_record_missing"
O2   src/quant/qat.py:616                             def read_tracked_selection(
O2   src/quant/qat.py:652                             def e6_parent_binding(
O2   scripts/qat_epoch_eval.py:554                    # E6: the λ and α selections and the parent they name (O2)
```

## 8. Item reports the GO asks for

- P19 (MEASURED): src/quant/ptq.py has 1079 lines at 647d305 and 1086 at the tip; lines 429 and 853 are byte-identical at both (True): `traced = torch.jit.trace(converted, example)` and `ts_b, ts_notes = torchscript_bytes(q_model, ts_meta)`. `git diff -U0 647d305 -- src/quant/ptq.py` has the hunks -438 +438; -441 +441; -776 +776; -1068,0 +1069,6; -1077,0 +1084: :438, :441 and :776 in place, six lines after compare_runs and one in `__all__`; above line 1067 no line was added or removed.
- P29 (REPOSITORY-PROVEN): no fake-quant-off driver, drift pin or "fp32" artifact exists in the lane; the trainer's
  per-epoch VAL is the fake-quant VAL with observers disabled, recorded and never used for selection.
- smoke_realrun_decisions: 80 → 76 checks (all PASS: True). Removed: locked_max_epochs, locked_early_stop_patience, locked_bn_freeze_pct, locked_observer_freeze_pct, bn_freeze_inside_registered_window, freeze_points_are_step_fractions, early_stop_eligibility_registered, patience_does_not_accrue_before_observer_freeze, patience_accrues_once_eligible, best_checkpoint_tracked_while_ineligible, runner_gates_early_stop_on_observer_freeze, qat_pilot_budget_shortened, gate_still_requires_grad-clip-norm, gate_still_requires_batch-size, gate_still_requires_weight-decay, gate_still_requires_epochs, gate_still_requires_early-stop-patience, gate_still_requires_bn-freeze-pct, gate_still_requires_observer-freeze-pct. Added: locked_epochs, locked_bn_freeze_epoch, locked_obs_freeze_epoch, freeze_points_are_epoch_boundaries, trainer_constants_equal_config, no_early_stop_controls_registered, qat_epochs_fixed_at_15, selection_on_converted_val, trainer_runs_every_epoch, qat_pilot_budget_full_15_epochs, gate_requires_expect_head, gate_still_requires_grad_clip_norm, gate_requires_u4_candidate, gate_binds_e5_seed42_to_u4_pilot, gate_takes_no_recipe_flag.
- x86 partial (smoke_quant_x86_efficiency): 72 → 73 checks (all PASS: True). Removed: official_converted_schema_unchanged, selection_semantics_preserved, sidecar_path_recorded_in_provenance. Added: epoch_checkpoint_loads_as_sidecar, no_selection_in_trainer, epoch_checkpoint_sha256_recorded, official_converted_schema_qat.
- P31 (MEASURED): `git diff -U0 647d305 -- src/quant/checkpoint.py` has 3 hunks (-17,2 +17,2; -134,2 +134,3; -137,3 +138,2), both inside docstrings; 202 lines before, 202 after.
- P32 (MEASURED): `git grep -n -E "^\s*(from|import) .*(train_distill|sweep_select|preflight_e1_trainval)" -- src/quant scripts/run_e5.py scripts/run_e6.py scripts/qat_epoch_eval.py scripts/select_qat_epoch.py scripts/select_clip.py scripts/synthetic_qat_fixtures.py ':(exclude,icase,glob)**/*test*'` prints nothing. `_copy_trained_state` stays an alias of `copy_qat_state_by_name`, which carries `@torch.no_grad()` (anchor P32 in §7).
- P33 (MEASURED at the tip): `git grep -n -E "qat_state_artifact|_qat_state\.pt|QAT_SIDECAR_SUFFIX|epochs_approx|early_stop|bn_freeze_pct|observer_freeze|max_epochs|EarlyStopper|unresolved_qat_values|bn_freeze_iteration|BN_FREEZE_PCT_RANGE|DRY_EPOCHS|official_max_epochs|freeze_rounding|_copy_trained_state" -- src scripts configs reports tools notebooks .github ':(exclude,icase,glob)**/*test*'` (explicit folders; never the repository root or docs/) prints:

```
reports/b58_teacher_runner_adjudication.md:396:68:        runner.message_hub.update_info('max_epochs', runner.max_epochs)
scripts/smoke_qat_runner.py:293:                 "never_observed_modules", "observer_freeze_scope", "tf32", "persistent_workers", "gpu_name",
scripts/smoke_qat_runner.py:302:          and meta.get("observer_freeze_scope") == "all FakeQuantize modules, weight and activation"
scripts/smoke_qat_runner.py:318:    pat = re.compile(r"patience|EarlyStopper|early_stop")
scripts/smoke_qat_runner.py:983:        ("bn_freeze_step", "bn_freeze_after_step", 3685), ("observer_freeze_step", "obs_freeze_after_step", 4355),
scripts/smoke_quant_runners.py:214:def test_no_early_stopping() -> None:
scripts/smoke_quant_runners.py:220:          not any(hasattr(_runner, n) for n in ("EarlyStopper", "run_qat", "unresolved_qat_values")),
scripts/smoke_quant_runners.py:222:    pat = re.compile(r"patience|EarlyStopper|early_stop")
scripts/smoke_quant_runners.py:301:               test_qat_real_run_gates, test_no_early_stopping, test_calibration_binding,
scripts/smoke_quant_x86_efficiency.py:230:              e.code == "qat_sidecar_required" and x86.QAT_SIDECAR_SUFFIX in str(e),
scripts/smoke_quant_x86_efficiency.py:387:    good = TMP / "e5_qat_state.pt"
scripts/smoke_realrun_decisions.py:60:    check("bn_freeze_precedes_observer_freeze",
scripts/smoke_realrun_decisions.py:72:def test_early_stop_after_observer_freeze() -> None:
scripts/smoke_realrun_decisions.py:75:    check("no_early_stop_controls_registered",
scripts/smoke_realrun_decisions.py:76:          not any("early_stop" in k or "patience" in k or "max_epochs" in k for k in keys),
scripts/smoke_realrun_decisions.py:85:          "for epoch in range(1, EPOCHS + 1):" in src and "EarlyStopper" not in src and "patience" not in src,
scripts/smoke_realrun_decisions.py:246:    for fn in (test_locked_qat_controls, test_early_stop_after_observer_freeze,
src/quant/__init__.py:25:                          QAT_SIDECAR_SUFFIX, X86_LATENCY_ARTIFACT_ROLE, X86_PTQ_STAGES,
src/quant/__init__.py:39:    "QAT_SIDECAR_KIND", "QAT_SIDECAR_ROLE", "QAT_SIDECAR_SUFFIX", "load_qat_sidecar",
src/quant/qat.py:896:        "observer_freeze_scope": OBSERVER_FREEZE_SCOPE,
src/quant/x86_latency.py:42:QAT_SIDECAR_SUFFIX = "_qat_state.pt"
src/quant/x86_latency.py:73:        f"pre-convert QAT sidecar ('*{QAT_SIDECAR_SUFFIX}', quantization={QAT_SIDECAR_KIND!r}). The "
src/quant/x86_latency.py:193:_copy_trained_state = copy_qat_state_by_name           # the pre-lane name, kept for its callers
src/quant/x86_latency.py:385:           "QAT_SIDECAR_SUFFIX", "X86_LATENCY_ARTIFACT_ROLE", "X86_PTQ_STAGES", "X86_QAT_STAGES",
```

Removed by the lane (REPOSITORY-PROVEN, `git diff 647d305 -- src/quant/runner.py src/quant/prepare.py configs/quant.py`):
runner.py `QAT`, `DRY_EPOCHS`, `unresolved_qat_values`, `EarlyStopper`, `run_qat` (now src/quant/qat.py `run_qat`, a new
signature); prepare.py `BN_FREEZE_PCT_RANGE`, `bn_freeze_iteration` (now `qat_freeze_steps`); configs/quant.py
`qat.epochs_approx`, `qat.early_stop`, `qat.bn_freeze_pct`, `qat.observer_freeze`, `qat_real_run.max_epochs`,
`.early_stop_patience`, `.bn_freeze_pct`, `.observer_freeze_pct`, `.freeze_rounding`, `.early_stop_eligible_after`,
`qat_grad_clip_pilot.official_max_epochs` (now `official_epochs`). x86_latency.py `_copy_trained_state` stays, as an
alias of `copy_qat_state_by_name` (P32).

Every match above is in a Q2 file except one, and none reads a removed name (REPOSITORY-PROVEN): the Q2 matches assert
that a removed name is gone (the patience greps of smoke_qat_runner and smoke_quant_runners, smoke_realrun_decisions'
no-early-stop checks), name a run_meta or profile key that is new (`observer_freeze_scope`, `bn_freeze_step`,
`observer_freeze_step`), or use a name that stays (`QAT_SIDECAR_SUFFIX` and `_copy_trained_state` in
src/quant/x86_latency.py and src/quant/__init__.py; smoke_quant_x86_efficiency's `e5_qat_state.pt` sidecar in
test_qat_translation, outside P30's hunk, unchanged and passing in the partial driver). The match outside Q2's files,
reports/b58_teacher_runner_adjudication.md, is a quoted mmengine teacher-runner line
(`runner.message_hub.update_info('max_epochs', runner.max_epochs)`), not a reader of configs/quant.py. No STOP item
for sequencing.

- P35 (MEASURED, the tip's d4 log; LSB = the scale of head.ff_add's output in the converted model; valid pixels are
  those outside the fixture's ±8 px ignore band at the 2×4 grid's cell borders, 8,960 of 16,384 per image; the
  band-0 line repeats the figures with no band, for DL-24's CS-9):
```
    [agreement] LSB 0.129241: overall 1.0, confident fraction 1.0, confident agreement 1.0, max |diff| 2.88 LSB, p99.9 1.81 LSB
    [agreement] by margin band: {"[0,1)": {"pixels": 0, "agreement": null}, "[1,2)": {"pixels": 0, "agreement": null}, "[2,4)": {"pixels": 0, "agreement": null}, "[4,8)": {"pixels": 0, "agreement": null}, "[8,inf)": {"pixels": 71680, "agreement": 1.0}}
    [agreement] controls: observers not carried 0.29880022321428573, BN not carried 0.125
    [agreement] band 0 (every pixel valid): overall 0.9991226196289062, confident fraction 0.974517822265625, confident agreement 1.0, max |diff| 2.88 LSB, p99.9 1.76 LSB
```
  P35's thresholds hold with the band and without it (MEASURED): (a) agreement 1.0 on every pixel with a margin of at
  least 8 LSB; (b) confident fraction 1.0 with the band and 0.9745 without it, both at least 0.90; (c) overall
  agreement 1.0 and 0.9991, both at least 0.99; (d) controls 0.299 and 0.125, both under 0.5.


## 9. Differences from the plan, and notes

1. **run_e5.py / run_e6.py entry point.** The plan had both import `src.quant.qat.main`. smoke_quant_e6_e7's
   `e6_e7_launchers_are_gated_and_stage_pinned` requires the text `src.quant.runner` in run_e6.py, so both import
   `src.quant.runner.main`, which delegates E5/E6 to `src.quant.qat.main` (QAT left runner.py). MEASURED: e6_e7 36/37
   before, 37/37 after.
2. **d2 canvas.** One QAT step at 16×3×512×512 on CPU needs about 15 GB (1.9 GB at batch 2, 3.6 GB at batch 4;
   MEASURED), so the d2 workers were OOM-killed on this 16 GB host. smoke_qat_seeding's workers set the canvas to 64
   in-process (`transforms.SIZE` and the `_random_crop_pad_512` default); no frozen file's bytes change.
3. **NaN injection** is multiplicative (`loss * NaN`), so the NaN reaches the gradients and parameters; an additive
   NaN left the gradients finite (MEASURED).
4. **finalize --clip-selection-sha256** is optional: block B passes none. finalize always records the file's sha256 and
   checks it when given.
5. **cpu_model** sits under run_meta `host` and is compared by select_clip: P27 exempts only the host name and pod id.
6. **The C4 reflow bug.** Wrapping lines over 120 characters dedented one `raise` in qat_select (UnboundLocalError on
   every excluded epoch). The C4 suite ran before the fix and found it (selection 99/101); the fixed file was
   validated by its one importer, smoke_qat_selection, at 101/101 (MEASURED), and the smoke files' reflow was checked
   AST-identical.
7. **A tensor storage defect found in C1** (MEASURED): `Tensor.numpy()` marks a storage non-resizable; hashing a live
   per-channel observer buffer before its first observation made `observer.resize_` raise, and the predicate then
   segfaulted on the half-resized tensor. `_hash_tensors` and `batch_sha256` hash copies.
8. **Fixture (P35)**: uniform-colour images with train-mode BN gave eval accuracy 0.25; pre-estimated frozen BN gave a
   QAT train/eval mismatch (fq margin ≥ 8 LSB only 0.62). The final fixture: 8 colours (the cube corners) in a 2×4 grid
   plus offset 3.0, trained in train mode, BN re-estimated (cumulative average over 40 batches).
9. **The harness's kill rule changed in C6** (DL-24's M1). C5's run (MEASURED: `RESULT: PASS (198/198)`, 3,310 s, 0
   survivors) still counted a killing check the smoke never reached as a kill: 18 of its 197 kills were of that kind.
   The C6 harness counts only a named check reported FAIL, baselines every job with its own argv, and its
   runner checks catch any exception, so those 18 jobs are now killed by an explicit FAIL (§5).
10. **The DL-24 workflow** ran four reviewers and four refuting verifiers (eight agents, under the session's
   guideline of fewer than ten), each reviewer with its dimension's smokes in its brief. Fixes for refuted findings
   were drafted in the scratchpad while the verifiers ran, then dropped: none entered a commit (T5, T7, B1, CS-6,
   SEL-10, SEL-11, and SEL-1's re-derived tie, rules and smoke checks).
11. **Mutations X1–X17** were added for the DL-24 fixes; they are not plan items. X5 and X7 were withdrawn with T7
   and T5.
12. **A defect in a new C6 case, found by the C6 suite** (MEASURED): the first full selection run gave
   `RESULT: FAIL (131/132)`: finalize_refuses_an_edited_winner_clip was refused `output_exists`, because the earlier
   case finalize_serves_the_winner had finalized the eval directory both shared. The case now builds its own pair of
   eval directories and its own clip_selection.json; alone it passes (`RESULT: PASS (1/1)`, refused
   `clip_selection_winner_clip`), and the selection smoke was re-run in full at the final tree (§3).


## 10. Tooling conflicts (each reported once, none complied with)

- The session's designated branch is `claude/lane-q2-qat-thesis-431dzr`; the GO names `lane/q2-qat` (DL-37). The lane
  works and pushes on `lane/q2-qat` only; the designated branch is not touched.
- A session attribution reminder asks for `Co-Authored-By` and `Claude-Session` trailers on commits. The GO says no
  trailers; no commit carries one.
- `~/.claude/stop-hook-git-check.sh` repeatedly demanded that the uncommitted changes be committed and pushed. Commits
  followed validation and the one push follows the report; the hook's demand released no gate (AGENTS.md).
- Not a conflict, for the record: the protected-file hook refused command lines whose paths it could not resolve
  (`ls` of a variable or `~` path, a `git show` without paths, inline interpreter code that mentions git). Each was
  rewritten with literal paths or a script file in the scratchpad; nothing under docs/reference/ was named or read.

## 11. SL-1

- No file or folder whose name contains "test" was created by the lane's smokes (they use `safe_tmpdir` temp trees),
  and no script whose path contains "test" ran. smoke_run_ptq (not Q2's, run as step 5 requires) creates
  `images/test` and `annotations/test` (3 synthetic pairs) inside its own temp tree through
  `synthetic_ptq_fixtures.make_tree(n_test=3)` (REPOSITORY-PROVEN); that is the existing smoke's behaviour, unchanged.
- smoke_quant_x86_efficiency (not Q2's, run as step 5 requires) imports `src.stats` in `test_backend_parity`, whose
  package imports `src/stats/tests.py`; that module runs until its line 21, `from statsmodels.stats.multitest import
  multipletests`, fails (MEASURED, the same traceback at the base and at the tip). A module whose path contains
  "test" was therefore loaded by a mandated smoke, at the base and at the tip; the partial driver skips that function.
- G1's given-path half and the `--out-dir` order have no exercising case: either needs a filesystem entry whose name
  contains "test", which the lane never creates.

What the lines are (read from the lines themselves; the guard is in log mode and denied nothing):
- Most are git commands whose pathspec carries the SL-1 exclusion (the word "test" in `':(exclude,icase,glob)**/*test*'`
  is what the guard matches), and a few git commands over lane folders without it (`git diff --stat 647d305 -- src
  scripts configs` at 17:29:43, a `git ls-files` of named Dockerfiles, a `git log -L` on a lane file).
- Some are shell `grep`/`sed` runs over named files or explicit globs in src/ or scripts/ without the exclusion (for
  example `grep ... scripts/evaluate_model.py src/eval/*.py` at 12:22:16, `grep -r ... src/quant src/distill` at
  18:09:54), and one Grep tool search over src/quant at 17:56:55 by a DL-24 review agent. That SL-1 rule was not kept
  on those lines. Whether a glob or a recursive grep matched a file whose name contains "test" was not checked (that
  needs a listing); the lane opened no such file by name (INFERRED from the lines).
- The `rm -f` at 12:15:14 removed bytecode that the lane's first runs wrote into `__pycache__` folders before every
  run used `python -B`; nothing tracked was removed.

`~/.claude/sl1_guard.log` (log mode; 54 lines at 2026-10-05T21:24Z), every line, cut at 200 characters:
```
2026-10-05T11:19:52Z	Bash	SL1-B2	echo "--- my files: diff afd2d33..647d305 (empty = byte-identical) ---" && git diff --stat afd2d3321ad122802bb05c8f62617b1df7847356 647d305274c439f30c1e087e9368100103c
2026-10-05T11:20:01Z	Bash	SL1-B2	git diff --stat afd2d3321ad122802bb05c8f62617b1df7847356 647d305274c439f30c1e087e9368100103ca1e7e -- AGENTS.md CLAUDE.md .claude/settings.json .claude/hooks/protect_re
2026-10-05T11:25:48Z	Bash	SL1-B2	git ls-files -- configs/calibration ':(exclude,icase,glob)**/*test*'; git log --oneline -1 -- configs/calibration
2026-10-05T11:38:06Z	Bash	SL1-B2	git ls-files -- configs/calibration && ls -d ptq_smoke_out 2>&1; git log --oneline -1 -- configs/calibration
2026-10-05T11:39:46Z	Bash	SL1-B2	git grep -n -E "qat_state_artifact|_qat_state\.pt|epochs_approx|early_stop|bn_freeze_pct|observer_freeze|max_epochs|EarlyStopper|unresolved_qat_values|bn_freeze_iterat
2026-10-05T11:40:07Z	Bash	SL1-B2	grep -n -E "^(def |class |[A-Z_]+ *=)|QAT_SIDECAR_SUFFIX" src/quant/x86_latency.py | head -n 50; git grep -n -E "qat_state_artifact|_qat_state\.pt|epochs_approx|early_
2026-10-05T11:44:56Z	Bash	SL1-B2	git grep -n -E "QUANT\[|QUANT\.get|qat_real_run|qat_grad_clip_pilot|checkpoint_selection|pilot_budget_epochs|official_max_epochs" -- src scripts configs reports tools 
2026-10-05T12:14:58Z	Bash	SL1-B1	git status -sb --untracked-files=all -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*' | head -n 30; git check-ignore -
2026-10-05T12:15:07Z	Bash	SL1-B2	git ls-files --others --ignored --exclude-standard -- src scripts configs ':(exclude,icase,glob)**/*test*' | head -n 40
2026-10-05T12:15:14Z	Bash	SL1-D2	rm -f configs/__pycache__/quant.cpython-311.pyc scripts/__pycache__/run_e5.cpython-311.pyc scripts/__pycache__/run_e6.cpython-311.pyc scripts/__pycache__/smoke_qat_run
2026-10-05T12:22:16Z	Bash	SL1-B1	sed -n '470,500p' src/eval/artifacts.py; grep -n -E "def build_eval_runtime_record" -A30 scripts/evaluate_model.py src/eval/*.py 2>/dev/null | grep -n -E "\"[a-z_]+\":
2026-10-05T12:24:26Z	Bash	SL1-B1	grep -n -E "^RECORD_KEYS|^EVAL_NUM_WORKERS|^def image_digest" -A3 src/eval/eval_runtime.py | head -n 20; grep -n -E "preprocess_protocol|PREPROCESS_PROTOCOL *=" src/ev
2026-10-05T13:28:14Z	Bash	SL1-B2	git grep -n "159279104\|159,279,104" -- src scripts configs reports ':(exclude,icase,glob)**/*test*' | head -5; echo ---; grep -n "inherited_by\|tie_band\|rules_file\|
2026-10-05T13:37:39Z	Bash	SL1-B1	sed -n 661,672p src/quant/qat.py; grep -n "def code_provenance" -A 30 src/quant/ptq.py | head -40; grep -n "CLASS_WEIGHTS_JSON =" -r src configs scripts --include=*.py
2026-10-05T13:43:38Z	Bash	SL1-B1	grep -n "^def build_student" -A 3 src/models/student.py; grep -n "build_student(" src/quant/*.py | head -5
2026-10-05T13:56:10Z	Bash	SL1-B2	git grep -n "QAT_SIDECAR_SUFFIX\|_qat_state\.pt\|qat_state_artifact" -- src scripts configs reports tools notebooks .github ':(exclude,icase,glob)**/*test*' | head -20
2026-10-05T14:40:15Z	Bash	SL1-B1	for f in pyproject.toml setup.cfg .flake8 ruff.toml tox.ini; do [ -f "$f" ] && grep -n "line-length\|max-line-length\|line_length" "$f" | sed "s|^|$f: |"; done; awk 'l
2026-10-05T14:40:28Z	Bash	SL1-B2	git add -- src/quant/qat_artifacts.py scripts/qat_epoch_eval.py scripts/synthetic_qat_fixtures.py scripts/smoke_qat_artifacts.py && git commit --only -F /tmp/claude-0/
2026-10-05T15:25:51Z	Bash	SL1-B1	grep -n "^def describe_qconfig" -A 25 src/quant/qconfig.py | head -40; sed -n 372,384p src/quant/x86_latency.py; grep -n "x86_artifact_provenance(" src/quant/*.py scri
2026-10-05T16:14:16Z	Bash	SL1-B2	git grep -n "qat_select" -- src scripts configs ':(exclude,icase,glob)**/*test*' | grep -v "^src/quant/qat_select.py" | cut -c1-150
2026-10-05T16:30:58Z	Bash	SL1-B2	cd /home/user/plantseg-thesis; git grep -n "allow_tf32\|matmul_precision\|cudnn.benchmark\|cudnn.deterministic\|use_deterministic" -- src scripts configs ':(exclude,ic
2026-10-05T17:08:31Z	Bash	SL1-B2	git diff --stat 647d305274c439f30c1e087e9368100103ca1e7e -- src configs scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T17:10:18Z	Bash	SL1-B2	git grep -n "def code_provenance\|def check_output_dir\|def check_source\|def _image_digest\|def set_seed\|def build_dataloader\|def assert_trainval_only_root\|def loa
2026-10-05T17:12:04Z	Bash	SL1-B2	git ls-files -- Dockerfile docker/Dockerfile Dockerfile.e1 .dockerignore | head; git cat-file -p HEAD:Dockerfile 2>/dev/null | head -40
2026-10-05T17:12:15Z	Bash	SL1-B2	git grep -n "train_distill\|sweep_select\|preflight_e1_trainval" -- src/quant configs/quant.py configs/qat_selection_rules.json scripts/qat_epoch_eval.py scripts/selec
2026-10-05T17:16:18Z	Bash	SL1-B2	git grep -n "checkpoint_sha256_mismatch\|run_incomplete\|epoch_end_duplicate\|telemetry_not_strict_json" -- src scripts ':(exclude,icase,glob)**/*test*'; grep -n "chec
2026-10-05T17:16:33Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "state_nonfinite|nonfinite_since_step|state_finite|step_error|run_stop|run_abort" -- src scripts ':(exclude,icase,glob)**/
2026-10-05T17:19:31Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "observers_enabled|bn_frozen|observers_disabled_during_val|state_unchanged|state_sha256" -- src scripts ':(exclude,icase,g
2026-10-05T17:27:01Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis diff --stat 647d305274c439f30c1e087e9368100103ca1e7e -- src configs scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T17:29:43Z	Bash	SL1-B2	git diff --stat 647d305274c439f30c1e087e9368100103ca1e7e -- src scripts configs
2026-10-05T17:30:59Z	Bash	SL1-B2	cd /home/user/plantseg-thesis; git grep -n -E "qat_state_artifact|_qat_state\.pt|QAT_SIDECAR_SUFFIX|epochs_approx|early_stop|bn_freeze_pct|observer_freeze|max_epochs|E
2026-10-05T17:38:23Z	Bash	SL1-B2	git grep -n -E "\"(governed_paths_clean|forward_batches|torch_num_threads|image_digest|repo_commit|metric_impl_sha256|config_sha256|split_manifest_sha256|class_map_sha
2026-10-05T17:41:01Z	Bash	SL1-B2	git diff --name-only 647d305274c439f30c1e087e9368100103ca1e7e -- src configs scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T17:44:01Z	Bash	SL1-B4	cd /home/user/plantseg-thesis; for c in telemetry_missing telemetry_not_strict_json run_meta_rows epoch_end_duplicate run_incomplete epoch_unknown checkpoint_name chec
2026-10-05T17:45:21Z	Bash	SL1-B2	git grep -n -E "epoch_selection|clip_selection\(|select_qat_epoch|read_ts_identity|cmd_finalize|\"finalize\"" -- scripts src ':(exclude,icase,glob)**/*test*'
2026-10-05T17:56:55Z	Grep	SL1-G1	/home/user/plantseg-thesis/src/quant
2026-10-05T17:57:29Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "scheduler|param_groups\[0\]\[.lr.\]|_step_count|get_last_lr" -- src scripts ':(exclude,icase,glob)**/*test*' | grep -v -E
2026-10-05T17:59:52Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n "checkpoint_sha256_mismatch" -- src scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T18:00:12Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -e "verified_checkpoint" -e "read_run_record" -e "require_complete" -- src scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T18:00:37Z	Bash	SL1-B2	sed -n 320,345p /home/user/plantseg-thesis/src/quant/qat_select.py; git -C /home/user/plantseg-thesis grep -n -E "state_nonfinite|bn_frozen|observers_enabled|observers
2026-10-05T18:04:52Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis diff --stat 647d305274c439f30c1e087e9368100103ca1e7e -- src configs scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T18:06:37Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -e "check-run-meta" -e "profile_mismatches" -- scripts src ':(exclude,icase,glob)**/*test*'
2026-10-05T18:08:33Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "select_qat_epoch|SE\.main|S\.epoch_selection|epoch_selection\(" -- src scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T18:08:44Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis diff --stat 647d305274c439f30c1e087e9368100103ca1e7e -- src configs scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T18:09:54Z	Bash	SL1-B1	cd /home/user/plantseg-thesis; grep -n "CWD_PROJECTION_KEY" scripts/smoke_qat_runner.py | head -n 3; grep -n "^CWD_PROJECTION_KEY\|CWD_PROJECTION_KEY =" -r src/quant s
2026-10-05T18:10:45Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "obs e1|observers_equal_to_e12|freeze_cross_check|first_differences" -- scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T18:10:48Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "checkpoint_sha256_mismatch|checkpoint_identity|stored_flags|state_alias_mismatch|state_copy_mismatch|qconfig_changed|iden
2026-10-05T18:11:13Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n "forward_batches" -- src scripts configs ':(exclude,icase,glob)**/*test*'
2026-10-05T18:11:42Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "\"finalize\"|finalize_cli|cmd_finalize" -- scripts ':(exclude,icase,glob)**/*test*'
2026-10-05T18:11:54Z	Bash	SL1-B6	git -C /home/user/plantseg-thesis log -L '/def color_task/,/return x.contiguous/:scripts/synthetic_qat_fixtures.py' --format="%h %s" 647d305274c439f30c1e087e9368100103
2026-10-05T18:12:44Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "image_digest|PLANTSEG_IMAGE_DIGEST" -- src/eval scripts/evaluate_model.py ':(exclude,icase,glob)**/*test*'
2026-10-05T18:15:32Z	Bash	SL1-B2	git -C /home/user/plantseg-thesis grep -n -E "translate_qat_state_to_x86\(\"E6\"|translate_qat_state_to_x86\('E6'|stage=\"E6\"|\"E6\", student" -- scripts ':(exclude,i
2026-10-05T18:32:02Z	Bash	SL1-B4	D=/tmp/claude-0/-home-user-plantseg-thesis/aa86210d-85b0-5686-8f2a-1875b4eb7c2f/scratchpad/draft/c6/scripts; for n in d1_val_error_recorded_and_train_mode_restored mai
2026-10-05T21:24:03Z	Bash	SL1-B2	cat > /tmp/claude-0/-home-user-plantseg-thesis/aa86210d-85b0-5686-8f2a-1875b4eb7c2f/scratchpad/drivers/assemble_fix5.py <<'EOF' from pathlib import Path p = Path("/tmp
```

## 12. Deferred commands (O4)

The audit's §3.3 blocks, as the GO quotes them. **Differences: none in flags, file names or RESULT strings.** Each
command's flags were checked against the final parsers (REPOSITORY-PROVEN: src/quant/qat.py `build_parser`;
scripts/qat_epoch_eval.py `build_parser`; scripts/select_qat_epoch.py and scripts/select_clip.py `add_argument`), and each
expected RESULT line against the code that prints it (`result_line` in src/quant/qat.py; the CONVERTED, SCORED, FINALIZED
and CHECK-RUN-META lines of scripts/qat_epoch_eval.py; SELECTED in scripts/select_qat_epoch.py; CLIP SELECTED in
scripts/select_clip.py). The one value the lane supplies:

1. Block A, GPU smoke: `<n>` = 196. `--device cuda` runs every section, as the CPU run does; no check is conditional on
   the device (REPOSITORY-PROVEN: scripts/smoke_qat_runner.py `main`, `test_d1`, `test_nonfinite`), so the CUDA count equals
   the CPU count, 196 (MEASURED on CPU; the CUDA run is deferred).

Notes on the blocks (not changes):
- check-run-meta reads the run_meta row, the first line of qat_telemetry.jsonl, which the trainer writes after every
  launch gate has passed and before the first step. Run before that, it ends `RESULT: INCOMPLETE [telemetry_missing]`,
  exit 3, which the blocks' rules make a STOP.
- epoch_ckpts/ holds only e01.pt … e15.pt (each written to a same-directory `.tmp` and renamed), so the `=15` count checks
  hold for a complete run (REPOSITORY-PROVEN: src/quant/qat.py `save_epoch_checkpoint`).
- finalize writes only into the eval directory (`final/`, qat_finalize.json, or finalize_STOP.json); the run directory may
  stay read-only (REPOSITORY-PROVEN: scripts/qat_epoch_eval.py `cmd_finalize`).

Rules for both blocks:
- Run one command at a time. Go on only when the expected lines appear.
- Replace every `<…>` first. `<PIN>` is the merged pin recorded in DL-37.
- On any other output or a non-zero exit: STOP. Never run the command again and delete nothing, containers included. A STOP never ends a running process; the ruling says what happens to it. A repeat follows a ruling and, for a run, AM-8a's written fault report; it uses a new directory.
- The RESULT strings are those of P37; the lane's report confirms them.
- Estimates are INFERRED, not measured: 2.2–3.1 h per pilot run on the pod (the plan's figure); on the laptop about 10–25 min of conversions and 1–3.6 h of scoring per run. One data point: the converted student takes 0.21–0.29 s per 512×512 image on a 2-core cloud CPU (MEASURED).

**Block A: pod (bash).** Needs G1, G2, G3, G5 and G6; AM-19's launch checklist done (no recorded divergence of E1 or E5 at seed 42 in the decision log, item 1(e)'s launch order, no waiting adviser reply that changes the schedule); a pod started from the pinned image by digest with /dev/shm for 12 workers (runbook §9.1); the DL-19 clone at `<PIN>` (§9.2); the TRAIN/VAL-only root (§9.3). `<EVID>` exists and lies outside the clone, the data root and /workspace/qat_runs. A new shell repeats the first block.

```bash
set -o noclobber
export PLANTSEG_IMAGE_DIGEST=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
unset PLANTSEG_GIT_COMMIT
export PYTHONPATH=<CLONE>
export PLANTSEG_DATA_ROOT=<TRAINVAL_ROOT>
cd <CLONE>
git rev-parse HEAD
```
Expected:
```
<PIN>
```
The pod's clock (AM-19's launch checklist). Write the laptop's UTC time at that moment beside it in the launch record; the difference is the pod's clock offset.
```bash
date -u +"%Y-%m-%dT%H:%M:%SZ" > <EVID>/pod_clock_e5_s42.txt
cat <EVID>/pod_clock_e5_s42.txt
```
Expected: one UTC time.
```bash
sha256sum <E1DIR>/e1_student_best_iter80000.pt
```
Expected first field:
```
cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03
```
GPU smoke (P38), synthetic data only. `<n>` is the count the lane's report gives for the CUDA run.
```bash
python -B scripts/smoke_qat_runner.py --device cuda > <EVID>/smoke_qat_runner_cuda.log 2>&1
tail -n 1 <EVID>/smoke_qat_runner_cuda.log
```
Expected:
```
RESULT: PASS (<n>/<n>)
```
Run 1 (clip 1.0):
```bash
nohup bash -c 'python -B scripts/run_e5.py --real-run --confirm-real-run --source-run-dir <E1DIR> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03 --expect-head <PIN> --seed 42 --grad-clip-norm 1.0 --u4-pilot --num-workers 12 --out-dir /workspace/qat_runs/e5_s42_clip1.0; echo "EXIT=$?"' > <EVID>/e5_s42_clip1.0.log 2>&1 &
echo $! > <EVID>/e5_s42_clip1.0.pid
```
Within 5 minutes:
```bash
python -B scripts/qat_epoch_eval.py check-run-meta --run-dir /workspace/qat_runs/e5_s42_clip1.0 --stage E5 --seed 42 --grad-clip-norm 1.0 --u4-pilot --expect-head <PIN> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03
```
Expected last line:
```
RESULT: CHECK-RUN-META PASS
```
Status, as often as wanted (it changes nothing):
```bash
kill -0 "$(cat <EVID>/e5_s42_clip1.0.pid)" 2>/dev/null && echo RUNNING || echo ENDED
```
Once it prints ENDED:
```bash
tail -n 2 <EVID>/e5_s42_clip1.0.log
```
Expected:
```
RESULT: QAT COMPLETE (E5, seed 42, clip 1.0, 15/15 epochs)
EXIT=0
```
```bash
(cd /workspace/qat_runs/e5_s42_clip1.0 && sha256sum qat_telemetry.jsonl epoch_ckpts/e*.pt) > <EVID>/e5_s42_clip1.0.sha256
wc -l < <EVID>/e5_s42_clip1.0.sha256
```
Expected:
```
16
```
Run 2 (clip 5.0), only after run 1's two lines:
```bash
nohup bash -c 'python -B scripts/run_e5.py --real-run --confirm-real-run --source-run-dir <E1DIR> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03 --expect-head <PIN> --seed 42 --grad-clip-norm 5.0 --u4-pilot --num-workers 12 --out-dir /workspace/qat_runs/e5_s42_clip5.0; echo "EXIT=$?"' > <EVID>/e5_s42_clip5.0.log 2>&1 &
echo $! > <EVID>/e5_s42_clip5.0.pid
```
Within 5 minutes:
```bash
python -B scripts/qat_epoch_eval.py check-run-meta --run-dir /workspace/qat_runs/e5_s42_clip5.0 --stage E5 --seed 42 --grad-clip-norm 5.0 --u4-pilot --expect-head <PIN> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03
```
Expected last line:
```
RESULT: CHECK-RUN-META PASS
```
Status:
```bash
kill -0 "$(cat <EVID>/e5_s42_clip5.0.pid)" 2>/dev/null && echo RUNNING || echo ENDED
```
Once it prints ENDED:
```bash
tail -n 2 <EVID>/e5_s42_clip5.0.log
```
Expected:
```
RESULT: QAT COMPLETE (E5, seed 42, clip 5.0, 15/15 epochs)
EXIT=0
```
```bash
(cd /workspace/qat_runs/e5_s42_clip5.0 && sha256sum qat_telemetry.jsonl epoch_ckpts/e*.pt) > <EVID>/e5_s42_clip5.0.sha256
wc -l < <EVID>/e5_s42_clip5.0.sha256
```
Expected:
```
16
```
Pod CPU timing (d6), only after both completion lines. It converts one epoch and scores 64 VAL images as a smoke artifact; it produces no VAL score. bash prints three `time` lines after each RESULT line; the `real` line of the second command is d6's pod figure for 64 images.
```bash
time python -B scripts/qat_epoch_eval.py convert --run-dir /workspace/qat_runs/e5_s42_clip1.0 --eval-dir /workspace/qat_timing/clip1.0 --epochs 15 --purpose timing --host-label pod
```
Expected, before the `time` lines:
```
RESULT: CONVERTED 1/1 (timing)
```
```bash
time python -B scripts/qat_epoch_eval.py score --run-dir /workspace/qat_runs/e5_s42_clip1.0 --eval-dir /workspace/qat_timing/clip1.0 --epochs 15 --purpose timing --host-label pod
```
Expected, before the `time` lines:
```
RESULT: SCORED 1/1 (timing, 64 samples)
```
A pod STOP report carries the two log lines, the run_abort row if there is one, and the first and last wall_clock of the telemetry.

**Block B: laptop (PowerShell).** Needs block A's outputs downloaded, and the live checkout at `<PIN>` (mounted whole and read-only; DL-62). `<RUNS>` holds e5_s42_clip1.0 and e5_s42_clip5.0 as downloaded. `<EVALS>` lies outside the checkout. `<TEL10>` and `<TEL50>` are the sha256 of each qat_telemetry.jsonl as captured on the pod. Nothing in the checkout is edited while the block runs. Long passes run with the laptop on its charger and sleep set to Never.

```powershell
Set-Location "<CHECKOUT>"
git status -sb -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
```
Expected first line (a tracking suffix may follow the name):
```
## claude/keen-curie-u4a8ig
```
```powershell
git rev-parse HEAD
```
Expected:
```
<PIN>
```
Setup:
```powershell
$IMG  = "ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
$PIN  = "<PIN>"
$REPO = "<CHECKOUT>"
$DS   = "<TRAINVAL_DS>"
$RUNS = "<RUNS>"
$EV   = "<EVALS>"
$ATT  = "a1"
function RO([string]$src, [string]$dst) { @("--mount", "type=bind,source=$src,target=$dst,readonly") }
function RW([string]$src, [string]$dst) {
    New-Item -ItemType Directory -Force -Path $src | Out-Null
    @("--mount", "type=bind,source=$src,target=$dst")
}
$ENVV = @("--network","none",
          "-e","PLANTSEG_IMAGE_DIGEST=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
          "-e","PLANTSEG_DATA_ROOT=/data","-e","PYTHONPATH=/work/repo","-e","PYTHONDONTWRITEBYTECODE=1",
          "-e","GIT_OPTIONAL_LOCKS=0","-e","GIT_CONFIG_COUNT=1","-e","GIT_CONFIG_KEY_0=safe.directory",
          "-e","GIT_CONFIG_VALUE_0=/work/repo","-e","EXPECT_P=$PIN","-w","/work/repo") + (RO $REPO "/work/repo")
$DATA = (RO "$DS\images\train" "/data/images/train") + (RO "$DS\images\val" "/data/images/val") +
        (RO "$DS\annotations\train" "/data/annotations/train") + (RO "$DS\annotations\val" "/data/annotations/val")
$R10  = RO "$RUNS\e5_s42_clip1.0" "/runs/e5_s42_clip1.0"
$R50  = RO "$RUNS\e5_s42_clip5.0" "/runs/e5_s42_clip5.0"
$E10  = RW "$EV\$ATT\e5_s42_clip1.0" "/evals/e5_s42_clip1.0"
$E50  = RW "$EV\$ATT\e5_s42_clip5.0" "/evals/e5_s42_clip5.0"
$T10  = RW "$EV\$ATT\timing_e5_s42_clip1.0" "/timing/e5_s42_clip1.0"
$SEL  = RW "$EV\$ATT\clip_selection" "/sel"
$STATE = "ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}}"
$CHECK = "import os,subprocess,sys;a=sys.argv[1:];g=lambda *x:subprocess.run(['git','-C','/work/repo']+list(x),capture_output=True,text=True);h=g('rev-parse','HEAD').stdout.strip();q=g('status','--porcelain=v1','--untracked-files=all','--','src','configs','scripts','requirements*','docs/EVALUATION_CONTRACT.md','docs/IMPLEMENTATION_CONTRACT.md');d=len(q.stdout.splitlines()) if q.returncode==0 else 'git failed';r=[('HEAD',h,os.environ.get('EXPECT_P')),('GOVERNED_DIRTY',d,0)];c=lambda p:sum(1 for e in os.scandir(p) if e.is_file()) if os.path.isdir(p) else 'missing';s=[x.rsplit('=',1) if '=' in x else [x,''] for x in a];r+=[(p,c(p),int(n)) if n else (p,os.path.exists(p),True) for p,n in s];[print(k,v,'OK' if v==w else 'STOP') for k,v,w in r];sys.exit(0 if all(v==w for k,v,w in r) else 1)"
```
How the lines below work:
- Each line names its container, runs it, then prints its STOP fields. After the command's own output the last line must read `ExitCode=0 OOMKilled=false StartedAt=<time> FinishedAt=<time>`.
- A check line runs in the same mounts as the command after it. It prints the HEAD, the count of dirty governed paths (the evaluator's own list) and one count or True per path, never a name. Every line must end in OK.
- docker refuses a container name that exists, so no line can run twice. If the window closes, the container keeps running: read it with `docker logs -f <name>`, then print its STOP fields.

Command 0. An empty HEAD or `GOVERNED_DIRTY git failed STOP` means git refused the mount: STOP and use the data session's recipe (DL-62).
```powershell
$NM = "q2_${ATT}_cmd0_git"; docker run --name $NM @ENVV $IMG git --version; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_cmd0_check"; docker run --name $NM @ENVV $IMG python -B -c $CHECK; docker inspect -f $STATE $NM
```
Expected: one git version line; then
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
```
Laptop timing (d6): one epoch converted, 64 VAL images scored as a smoke artifact; it produces no VAL score.
```powershell
$NM = "q2_${ATT}_t10_check1"; docker run --name $NM @ENVV @R10 @T10 $IMG python -B -c $CHECK /runs/e5_s42_clip1.0/epoch_ckpts=15 /runs/e5_s42_clip1.0/qat_telemetry.jsonl /timing/e5_s42_clip1.0; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip1.0/epoch_ckpts 15 OK
/runs/e5_s42_clip1.0/qat_telemetry.jsonl True OK
/timing/e5_s42_clip1.0 True OK
```
```powershell
$NM = "q2_${ATT}_t10_convert"; docker run --name $NM @ENVV @R10 @T10 $IMG python -B scripts/qat_epoch_eval.py convert --run-dir /runs/e5_s42_clip1.0 --eval-dir /timing/e5_s42_clip1.0 --epochs 15 --purpose timing --host-label laptop; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: CONVERTED 1/1 (timing)
```
```powershell
$NM = "q2_${ATT}_t10_check2"; docker run --name $NM @ENVV @DATA @R10 @T10 $IMG python -B -c $CHECK /data/images/train=5367 /data/images/val=846 /data/annotations/train /data/annotations/val /runs/e5_s42_clip1.0/epoch_ckpts=15 /timing/e5_s42_clip1.0; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/data/images/train 5367 OK
/data/images/val 846 OK
/data/annotations/train True OK
/data/annotations/val True OK
/runs/e5_s42_clip1.0/epoch_ckpts 15 OK
/timing/e5_s42_clip1.0 True OK
```
```powershell
$NM = "q2_${ATT}_t10_score"; docker run --name $NM @ENVV @DATA @R10 @T10 $IMG python -B scripts/qat_epoch_eval.py score --run-dir /runs/e5_s42_clip1.0 --eval-dir /timing/e5_s42_clip1.0 --epochs 15 --purpose timing --host-label laptop; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: SCORED 1/1 (timing, 64 samples)
```
FinishedAt minus StartedAt of that container is d6's laptop figure for 64 images.

Clip 1.0, the record pass. Check, then convert:
```powershell
$NM = "q2_${ATT}_c10_check1"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B -c $CHECK /runs/e5_s42_clip1.0/epoch_ckpts=15 /runs/e5_s42_clip1.0/qat_telemetry.jsonl /evals/e5_s42_clip1.0; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip1.0/epoch_ckpts 15 OK
/runs/e5_s42_clip1.0/qat_telemetry.jsonl True OK
/evals/e5_s42_clip1.0 True OK
```
```powershell
$NM = "q2_${ATT}_c10_convert"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B scripts/qat_epoch_eval.py convert --run-dir /runs/e5_s42_clip1.0 --eval-dir /evals/e5_s42_clip1.0 --purpose record --host-label laptop --expect-telemetry-sha256 <TEL10>; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: CONVERTED 15/15 (record)
```
Check, then score:
```powershell
$NM = "q2_${ATT}_c10_check2"; docker run --name $NM @ENVV @DATA @R10 @E10 $IMG python -B -c $CHECK /data/images/train=5367 /data/images/val=846 /data/annotations/train /data/annotations/val /runs/e5_s42_clip1.0/epoch_ckpts=15 /evals/e5_s42_clip1.0; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/data/images/train 5367 OK
/data/images/val 846 OK
/data/annotations/train True OK
/data/annotations/val True OK
/runs/e5_s42_clip1.0/epoch_ckpts 15 OK
/evals/e5_s42_clip1.0 True OK
```
```powershell
$NM = "q2_${ATT}_c10_score"; docker run --name $NM @ENVV @DATA @R10 @E10 $IMG python -B scripts/qat_epoch_eval.py score --run-dir /runs/e5_s42_clip1.0 --eval-dir /evals/e5_s42_clip1.0 --purpose record --host-label laptop; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: SCORED 15/15 (record)
```
Check, then select the epoch:
```powershell
$NM = "q2_${ATT}_c10_check3"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B -c $CHECK /runs/e5_s42_clip1.0/epoch_ckpts=15 /runs/e5_s42_clip1.0/qat_telemetry.jsonl /evals/e5_s42_clip1.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c10_select"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B scripts/select_qat_epoch.py --run-dir /runs/e5_s42_clip1.0 --eval-dir /evals/e5_s42_clip1.0 --expect-telemetry-sha256 <TEL10>; docker inspect -f $STATE $NM
```
Expected: the five OK lines of the first check; then, with the epoch the rule selects,
```
RESULT: SELECTED epoch <NN> (E5, seed 42, clip 1.0)
```
Clip 5.0, the same six lines, one at a time:
```powershell
$NM = "q2_${ATT}_c50_check1"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B -c $CHECK /runs/e5_s42_clip5.0/epoch_ckpts=15 /runs/e5_s42_clip5.0/qat_telemetry.jsonl /evals/e5_s42_clip5.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_convert"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B scripts/qat_epoch_eval.py convert --run-dir /runs/e5_s42_clip5.0 --eval-dir /evals/e5_s42_clip5.0 --purpose record --host-label laptop --expect-telemetry-sha256 <TEL50>; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_check2"; docker run --name $NM @ENVV @DATA @R50 @E50 $IMG python -B -c $CHECK /data/images/train=5367 /data/images/val=846 /data/annotations/train /data/annotations/val /runs/e5_s42_clip5.0/epoch_ckpts=15 /evals/e5_s42_clip5.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_score"; docker run --name $NM @ENVV @DATA @R50 @E50 $IMG python -B scripts/qat_epoch_eval.py score --run-dir /runs/e5_s42_clip5.0 --eval-dir /evals/e5_s42_clip5.0 --purpose record --host-label laptop; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_check3"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B -c $CHECK /runs/e5_s42_clip5.0/epoch_ckpts=15 /runs/e5_s42_clip5.0/qat_telemetry.jsonl /evals/e5_s42_clip5.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_select"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B scripts/select_qat_epoch.py --run-dir /runs/e5_s42_clip5.0 --eval-dir /evals/e5_s42_clip5.0 --expect-telemetry-sha256 <TEL50>; docker inspect -f $STATE $NM
```
Expected: the same OK lines as for clip 1.0, with the clip 5.0 paths (five, eight and five); and, after the second, fourth and sixth line,
```
RESULT: CONVERTED 15/15 (record)
RESULT: SCORED 15/15 (record)
RESULT: SELECTED epoch <NN> (E5, seed 42, clip 5.0)
```
Clip selection:
```powershell
$NM = "q2_${ATT}_sc_check"; docker run --name $NM @ENVV @R10 @R50 @E10 @E50 @SEL $IMG python -B -c $CHECK /runs/e5_s42_clip1.0/qat_telemetry.jsonl /runs/e5_s42_clip5.0/qat_telemetry.jsonl /evals/e5_s42_clip1.0/qat_selection.json /evals/e5_s42_clip5.0/qat_selection.json /sel; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip1.0/qat_telemetry.jsonl True OK
/runs/e5_s42_clip5.0/qat_telemetry.jsonl True OK
/evals/e5_s42_clip1.0/qat_selection.json True OK
/evals/e5_s42_clip5.0/qat_selection.json True OK
/sel True OK
```
```powershell
$NM = "q2_${ATT}_select_clip"; docker run --name $NM @ENVV @R10 @R50 @E10 @E50 @SEL $IMG python -B scripts/select_clip.py --candidate /runs/e5_s42_clip1.0 /evals/e5_s42_clip1.0 --candidate /runs/e5_s42_clip5.0 /evals/e5_s42_clip5.0 --out /sel/clip_selection.json; docker inspect -f $STATE $NM
```
Expected, with the winner the rule gives:
```
RESULT: CLIP SELECTED <1.0 or 5.0> (tie <true or false>)
```
Finalize, for the winner only. `<W>` is the value that line printed, 1.0 or 5.0.
```powershell
$W    = "<W>"
$RWIN = RO "$RUNS\e5_s42_clip$W" "/runs/e5_s42_clip$W"
$EWIN = RW "$EV\$ATT\e5_s42_clip$W" "/evals/e5_s42_clip$W"
$NM = "q2_${ATT}_fz_check"; docker run --name $NM @ENVV @RWIN @EWIN @SEL $IMG python -B -c $CHECK "/runs/e5_s42_clip$W/epoch_ckpts=15" "/evals/e5_s42_clip$W/qat_selection.json" /sel/clip_selection.json; docker inspect -f $STATE $NM
```
Expected, with the winner's paths:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip<W>/epoch_ckpts 15 OK
/evals/e5_s42_clip<W>/qat_selection.json True OK
/sel/clip_selection.json True OK
```
```powershell
$NM = "q2_${ATT}_finalize"; docker run --name $NM @ENVV @RWIN @EWIN @SEL $IMG python -B scripts/qat_epoch_eval.py finalize --run-dir "/runs/e5_s42_clip$W" --eval-dir "/evals/e5_s42_clip$W" --clip-selection /sel/clip_selection.json; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: FINALIZED (E5, seed 42, epoch <NN>)
```
The value for the decision-log row and for every later launch (P10):
```powershell
(Get-FileHash "$EV\$ATT\clip_selection\clip_selection.json" -Algorithm SHA256).Hash.ToLower()
```
Expected: 64 hex characters.

A local STOP report carries the container's name, its STOP-fields line and the last 20 lines of `docker logs` for it.


## 13. Proposed records

The audit's §4, as the GO quotes it (the docs session writes these; Q2 never does). `<tip>` is lane/q2-qat's tip after the
push (this report's commit); `<pin>` is the merge pin.

Decision log (the docs session writes these; Q2 never does):
- **DL-09** status: `RECORDED 8084671 (AM-4); L-AM4 code IMPLEMENTED <tip> (lane/q2-qat; merged <pin>)`.
- **DL-38** status: `RECORDED 826d851; AM-4a Status DRAFT, adviser approval pending; L-AM4 and L-AM1q code IMPLEMENTED <tip> (merged <pin>)`.
- **DL-02**: "QAT seeding is lane L-AM1q" gains `IMPLEMENTED <tip>`.
- **DL-37**: the new pin.
- **DL-17**: its status still reads "re-score pending (B66)". Record the verdict of the pair, or confirm that it is open.
- **DL-10**: "local d2–d5 pending" is closed by lane 8's own records before E5 launches.
- A new row (or a note on DL-58): plan R7's listing and search, with date, session and "no TEST data, no effect on any result".
- A new row at each pilot launch: stage, clip value, seed, run directory and the pod's clock offset. AM-19 asks for a launch-log entry only for sweep candidates and is silent on the pilot; this row is a precaution (INFERRED).
- A new row at selection time: clip_selection.json's sha256, the winner, both scores and both run directories.

IMPLEMENTATION_CONTRACT B4 notes (proposed; K-PART's edits have merged):
1. C:422, C:791–794 and C:816–817: L-AM4 has landed.
2. C:785–789: the AM-4a pins are read from configs/quant.py and checked against constants; the clip is the only launch value.
3. C:844: the pilot is two full 15-epoch E5 seed-42 runs (AM-4a item 3).
4. C:846–852: the selection sentence gives way to AM-4a item 3, with question 1's ruling for a non-finite candidate.
5. The schedule of record: 335 steps per epoch (drop_last), CosineAnnealingLR with T_max 5,025 and eta_min 0 stepped per optimizer step, BN statistics frozen after step 3,350 and observers after step 4,020.
6. The observer freeze covers every FakeQuantize module, weight and activation. Nine activation fake-quants have fixed qparams; five are never called.
7. Per-epoch fake-quant VAL runs with observers disabled. The pre-lane runner left them on (runner.py:389); no run was made with it.
8. Conversion: torch calls the weight fake-quant once at conversion, so a checkpoint of epochs 1–12 takes one more moving-average step of its per-channel weight ranges. Conversion is data-free and its parity inputs are synthetic.
9. C:772–776: the pre-convert state is the selected epoch's epoch_ckpts/eNN.pt, and the x86 copy is made for the run of record only.
10. Converted VAL scoring: CPU, canvas, batch size 1, one host. The artifact of record is the scored file.
11. Non-finite epochs, as ruled on question 1.

Errata (proposed):
- part2 lane 6 (d) d1: "step 20" and "step 24" are 0-based; the 21st and 25th steps are the first frozen ones.
- lane 6 (d) d5: no two float64 scores differ by exactly 0.1 pp. The pairs are 0.40099999999999997 vs 0.4 (tie) and 0.401 vs 0.4 (no tie), each in both orientations.
- lane 6 (d) d4: the 99% line stands with P35's margin rule and negative controls.
- lane 6 (d) d6: the wall time is measured on 64 VAL images as a smoke artifact, on the pod and on the laptop, and scaled to 846; no second full-VAL score of a pilot checkpoint is made.
- lane 6 (c): the converted artifacts, the scores and both selection files live in the eval directory.
- part2's order table: "≈ 2 × 5,040 steps" is 2 × 5,025.
- lane 6 (a), if question 2 is ruled so: the fake-quant-off score moves to its own lane, and EVALUATION_CONTRACT §5.3 gains the precision value there.

For lane L-LAT-THREADS (D:32): OMP_NUM_THREADS does not set QNNPACK's thread pool; torch.set_num_threads does (finding 51, MEASURED on a 2-core CPU).

Chapter 4 queue: the U4 disclosure (2 × 15 VAL evaluations, the clipped-step counts), the conversion note, and any non-finite epoch.

Decision-log rows for the orchestrator items:

- **New DL row, O1 (non-finite pilot runs; AM-21 item 3):** scripts/select_clip.py (src/quant/qat_select.py `clip_selection`) rejects a candidate run whose telemetry records `nonfinite_since_step` in any row, or whose epoch_end rows hold `state_finite: false`, and its rule_trace names "rejected: non-finite state (AM-21 item 3)". The rejection is anchored by the rejected run's own record conversion: its eval directory's qat_convert.json must name that run_id and telemetry sha256 with purpose record (else exit 3 `eval_record_missing`, or exit 2 `eval_record_mismatch` / `purpose_not_record`). The other run wins if it is complete with a selected epoch. With both rejected select_clip exits 2 with "no winner (AM-21 item 3): a new amendment decides the clip value" and writes nothing. Status: IMPLEMENTED `<tip>` (lane/q2-qat; merged `<pin>`). Smoke cases, each orientation where there are two: d5_nonfinite_candidate_rejected_a/b, d5_nonfinite_clip_1_rejected_a/b, d5_both_rejected_no_winner_exit_2_a/b, d5_state_finite_false_alone_rejected, d5_nonfinite_since_step_alone_rejected, d5_nonfinite_row_only_rejected, d5_rejected_candidate_without_conversion_record_exit_3, d5_rejected_candidate_record_mismatch_refused and d5_rejected_candidate_timing_record_refused; mutations M23 and X8 killed.
- **New DL row, O2 (E6's parent of record):** run_e6 requires `--lambda-selection <repo-relative path> --lambda-selection-sha256 <64 hex> --alpha-selection <repo-relative path> --alpha-selection-sha256 <64 hex>`; each file must be tracked and unchanged at HEAD (`git show HEAD:<path>`), match its sha256, and carry format lambda_selection/1 or alpha_selection/1; the parent's run_meta must have stage E3, the run's seed, lambda_logit = winner.lambda, alpha_cwd = winner.alpha, and at seed 42 the parent's run_id must be the α winner's run_id. With no α file run_e6 refuses `alpha_selection_missing`. The files are read with json.load only; nothing is imported from K2's files. check-run-meta repeats the comparison. Status: IMPLEMENTED `<tip>` (merged `<pin>`). The AM-16 cut path is added after K2 merges, before any E6 launch under a cut α. Mutations M24a (launch) and M24b (check-run-meta) killed.


## 14. Open questions

1. **A rejected pilot run and its conversion record (from DL-24's SEL-2).** select_clip now accepts a rejection
   (AM-21 item 3) only when the rejected run's eval directory holds the qat_convert.json of a record conversion of
   that telemetry (run_id, telemetry sha256, purpose record); without it select_clip exits 3 (`eval_record_missing`).
   Block A stops at a non-finite completion line (P37), so the ruling that follows decides whether the rejected run
   still gets its record `convert` pass before select_clip. A convert pass that passes P16's freeze cross-check writes
   qat_convert.json even when no epoch converts (REPOSITORY-PROVEN); whether a non-finite run passes that cross-check
   is not measured (INFERRED: frozen buffers keep their bits, NaN included, and the comparison is bitwise).
2. **Question 1 (OQ1) and question 2 stay as the GO records them:** the non-finite reading is recorded before the
   first QAT run; the fake-quant-off score has left this lane (P29) and, if question 2 is ruled so, moves to its own
   lane.
3. **smoke_quant_x86_efficiency cannot run whole in the GO's venv**: test_backend_parity imports src.stats, which needs
   statsmodels (not in step 4's list). Only the partial driver ran (§3). Whether lane venvs add statsmodels, or the
   smoke separates that section, is outside Q2's scope.
4. **smoke_run_ptq (68/69) and smoke_calibration_lists (41/42) fail at the base** on the same cause, configs/calibration
   tracked since c689634, which their "nothing written in the repository" checks read as a write. Not Q2's files; an
   owner and a lane are needed before those two smokes can gate anything.
5. **Refuted hardening, for the orchestrator's choice (not applied):** a conversion-time cleanliness gate (CS-6; today
   Block B's GOVERNED_DIRTY check before every convert does this job operationally), argparse usage errors with a
   RESULT line (T7), typed binding values (B1), and finalize re-deriving the selection (SEL-11). Each would be a new
   patch item.
6. **G1's given-path half** (a "latest" symlink, a directory listed before its name is tested) has no exercising case:
   SL-1 forbids creating the "test"-named entry such a case needs. Accept the code-review evidence, or name a
   fixture rule that allows it.


Lane pushed to lane/q2-qat.


## 15. Blocks A and B as revised by AM-21

AM-21 as committed (CP-007g, `2951dd4460faab03004fe203aa4e92fbd35b94e7`; DL-85) revises §12's blocks. §12 stays as
history; this section restates both blocks in full and replaces §12 for every run. The changes against §12, each with
its rule:

- **A non-finite completion is a REPORT, not a STOP** (Q2-4 ruling 1). Block A's P37 line for a run that completed with
  a non-finite state is pasted and recorded, and the run goes on to block B. Every other block A failure stays a STOP.
- **Record conversions wait for both runs and for the fault report** (CHECK ITEM 16; AM-21 items 1(f) and 3(e)). No
  record conversion of a pilot run starts before both pilot runs have ended. After a non-finite completion line, or a
  rejection that block B's rejection check prints, an AM-8a fault report (if there is a fault) is committed before any
  record conversion of either pilot run starts. The rejection check is added because item 1(f) also names a run whose
  logged loss or pre-clip gradient norm was non-finite, and such a run can complete with a finite state.
- **A rejected pilot run is converted and scored, never selected** (item 3(d)). It is still converted and scored as
  item 2(a) reads, and its records are kept. Its scores are reported and select nothing, and it gets no epoch selection.
  If its conversion or scoring fails, item 2(c) applies to it alone, and the other run's win does not wait.
- **select_clip checks each candidate's telemetry against the pod** (CHECK ITEM 12). It takes one
  `--expect-telemetry-sha256` per candidate, from the sha256 entries recorded on the pod, and prints one REPORT line per
  rejected run.
- **No convertible epoch is entered as "non-finite: no model"** (item 2(b)). For a run of record, the entry is written
  in the decision log that day.
- **Repeats** (items 2(c) and 3(e)). A re-made record conversion or scoring needs a written fault report, as a
  repeated run does, and two versions are never chosen between. Item 3(e) lifts a rejection once: a clip value whose
  repeat is rejected too is rejected for good.
- **Runs of record** (items 1(f) and 2(a)). Item 1(f)'s hold applies to them too: no record conversion before the
  AM-8a ruling. A selection with excluded epochs is a REPORT.
- **Block A's GPU smoke count** `<n>` is 209: the final count of scripts/smoke_qat_runner.py, at §16's last commit.
  `--device cuda` runs every section, as the CPU run does, and no check depends on the device
  (REPOSITORY-PROVEN: scripts/smoke_qat_runner.py `main`). The CUDA count is therefore the CPU count, 209 (MEASURED on
  CPU; the CUDA run is deferred).

§12's notes on the blocks still hold:
- check-run-meta reads the run_meta row before the first step.
- epoch_ckpts/ holds exactly e01.pt … e15.pt.
- finalize writes only into the eval directory.

check-run-meta now also compares the run's never-observed fake-quants with DL-85's five modules of record (CHECK
ITEM 10).

Rules for both blocks:
- Run one command at a time. Go on only when the expected lines appear.
- Replace every `<…>` first. `<PIN>` is the merged pin recorded in DL-37.
- **A REPORT outcome** is an expected output named as one below; no other output is one.
  - Paste it into the launch record, and enter it verbatim in the decision log that day.
  - It is not a STOP. The block goes on as its line says.
  - Where the line says so, a REPORT exits non-zero; its STOP-fields line then shows that exit code.
- **A warning is not an output.** torch 2.1 prints a UserWarning on stderr ("TypedStorage is deprecated") whenever a
  command loads a checkpoint, the rejection check's included, often between the expected lines (MEASURED with the
  pinned torch 2.1.0 on CPU; INFERRED for the image's CUDA build of the same version). It is neither a REPORT nor a
  STOP.
- **Anything else is a STOP**: any other output, or a non-zero exit outside a REPORT. Never run the command again, and
  delete nothing, containers included. A STOP never ends a running process; the ruling says what happens to it. A
  repeat follows a ruling and AM-8a's written fault report, for a run and for a failed record conversion or scoring
  alike (AM-21 item 2(c)). It uses a new directory, and two versions are never chosen between.
- The RESULT strings are those of P37 and AM-21 item 5; the lane's report confirms them (§16).
- **Estimates** (INFERRED, not measured):
  - on the pod, 2.2–3.1 h per pilot run (the plan's figure);
  - on the laptop, about 10–25 min of conversions and 1–3.6 h of scoring per run.

  One data point (MEASURED): the converted student takes 0.21–0.29 s per 512×512 image on a 2-core cloud CPU.

**Block A: pod (bash).** Needs:
- G1, G2, G3, G5 and G6;
- AM-19's launch checklist done: no recorded divergence of E1 or E5 at seed 42 in the decision log, item 1(e)'s launch
  order, and no waiting adviser reply that changes the schedule;
- the merged pin matching AM-21 (its header: no QAT run launches before then);
- a pod started from the pinned image by digest, with /dev/shm for 12 workers (runbook §9.1);
- the DL-19 clone at `<PIN>` (§9.2);
- the TRAIN/VAL-only root (§9.3).

`<EVID>` exists and lies outside the clone, the data root and /workspace/qat_runs. A new shell repeats the first block.

```bash
set -o noclobber
export PLANTSEG_IMAGE_DIGEST=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
unset PLANTSEG_GIT_COMMIT
export PYTHONPATH=<CLONE>
export PLANTSEG_DATA_ROOT=<TRAINVAL_ROOT>
cd <CLONE>
git rev-parse HEAD
```
Expected:
```
<PIN>
```
The pod's clock (AM-19's launch checklist). Write the laptop's UTC time at that moment beside it in the launch record;
the difference is the pod's clock offset.
```bash
date -u +"%Y-%m-%dT%H:%M:%SZ" > <EVID>/pod_clock_e5_s42.txt
cat <EVID>/pod_clock_e5_s42.txt
```
Expected: one UTC time.
```bash
sha256sum <E1DIR>/e1_student_best_iter80000.pt
```
Expected first field:
```
cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03
```
GPU smoke (P38), on synthetic data only:
```bash
python -B scripts/smoke_qat_runner.py --device cuda > <EVID>/smoke_qat_runner_cuda.log 2>&1
tail -n 1 <EVID>/smoke_qat_runner_cuda.log
```
Expected:
```
RESULT: PASS (209/209)
```
Run 1 (clip 1.0):
```bash
nohup bash -c 'python -B scripts/run_e5.py --real-run --confirm-real-run --source-run-dir <E1DIR> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03 --expect-head <PIN> --seed 42 --grad-clip-norm 1.0 --u4-pilot --num-workers 12 --out-dir /workspace/qat_runs/e5_s42_clip1.0; echo "EXIT=$?"' > <EVID>/e5_s42_clip1.0.log 2>&1 &
echo $! > <EVID>/e5_s42_clip1.0.pid
```
Within 5 minutes:
```bash
python -B scripts/qat_epoch_eval.py check-run-meta --run-dir /workspace/qat_runs/e5_s42_clip1.0 --stage E5 --seed 42 --grad-clip-norm 1.0 --u4-pilot --expect-head <PIN> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03
```
Expected last line:
```
RESULT: CHECK-RUN-META PASS
```
Status, as often as wanted (it changes nothing):
```bash
kill -0 "$(cat <EVID>/e5_s42_clip1.0.pid)" 2>/dev/null && echo RUNNING || echo ENDED
```
Once it prints ENDED:
```bash
tail -n 2 <EVID>/e5_s42_clip1.0.log
```
Expected, one of two pairs. The first is a run whose state stayed finite:
```
RESULT: QAT COMPLETE (E5, seed 42, clip 1.0, 15/15 epochs)
EXIT=0
```
The second is a REPORT. The run completed with a non-finite state, and `<N>` is the step at which the trainer first
found it (AM-21 items 1(c) and 5):
```
RESULT: QAT COMPLETE, STATE NON-FINITE first found at step <N> (E5, seed 42, clip 1.0, 15/15 epochs)
EXIT=0
```
The run is rejected (item 3(a)). It still goes on: the sha256 capture below, then run 2, then block B. Block B's gate
holds its record conversion.
```bash
(cd /workspace/qat_runs/e5_s42_clip1.0 && sha256sum qat_telemetry.jsonl epoch_ckpts/e*.pt) > <EVID>/e5_s42_clip1.0.sha256
wc -l < <EVID>/e5_s42_clip1.0.sha256
```
Expected:
```
16
```
Run 2 (clip 5.0). Start it only after run 1's two lines, either pair, and the capture:
```bash
nohup bash -c 'python -B scripts/run_e5.py --real-run --confirm-real-run --source-run-dir <E1DIR> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03 --expect-head <PIN> --seed 42 --grad-clip-norm 5.0 --u4-pilot --num-workers 12 --out-dir /workspace/qat_runs/e5_s42_clip5.0; echo "EXIT=$?"' > <EVID>/e5_s42_clip5.0.log 2>&1 &
echo $! > <EVID>/e5_s42_clip5.0.pid
```
Within 5 minutes:
```bash
python -B scripts/qat_epoch_eval.py check-run-meta --run-dir /workspace/qat_runs/e5_s42_clip5.0 --stage E5 --seed 42 --grad-clip-norm 5.0 --u4-pilot --expect-head <PIN> --expect-source-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03
```
Expected last line:
```
RESULT: CHECK-RUN-META PASS
```
Status:
```bash
kill -0 "$(cat <EVID>/e5_s42_clip5.0.pid)" 2>/dev/null && echo RUNNING || echo ENDED
```
Once it prints ENDED:
```bash
tail -n 2 <EVID>/e5_s42_clip5.0.log
```
Expected, as for run 1. The first pair:
```
RESULT: QAT COMPLETE (E5, seed 42, clip 5.0, 15/15 epochs)
EXIT=0
```
Or the REPORT pair, after which the run goes on as run 1's does:
```
RESULT: QAT COMPLETE, STATE NON-FINITE first found at step <N> (E5, seed 42, clip 5.0, 15/15 epochs)
EXIT=0
```
```bash
(cd /workspace/qat_runs/e5_s42_clip5.0 && sha256sum qat_telemetry.jsonl epoch_ckpts/e*.pt) > <EVID>/e5_s42_clip5.0.sha256
wc -l < <EVID>/e5_s42_clip5.0.sha256
```
Expected:
```
16
```
Pod CPU timing (d6), only after both runs' two lines.
- What it does: it converts one epoch and scores 64 VAL images as a smoke artifact. It produces no VAL score, and it is
  not a record conversion.
- `<TC>` is the clip of a run whose completion line is the first pair: 1.0 if run 1's is, else 5.0. A non-finite run's
  e15 is not convertible, so it cannot be timed.
- If both runs printed the REPORT pair, skip both timing commands. Record the REPORT line "d6 pod timing not run: both
  pilot runs completed non-finite".
- bash prints three `time` lines after each RESULT line. The `real` line of the second command is d6's pod figure for 64
  images.
```bash
time python -B scripts/qat_epoch_eval.py convert --run-dir /workspace/qat_runs/e5_s42_clip<TC> --eval-dir /workspace/qat_timing/clip<TC> --epochs 15 --purpose timing --host-label pod
```
Expected, before the `time` lines:
```
RESULT: CONVERTED 1/1 (timing)
```
```bash
time python -B scripts/qat_epoch_eval.py score --run-dir /workspace/qat_runs/e5_s42_clip<TC> --eval-dir /workspace/qat_timing/clip<TC> --epochs 15 --purpose timing --host-label pod
```
Expected, before the `time` lines:
```
RESULT: SCORED 1/1 (timing, 64 samples)
```
A pod STOP report carries the two log lines, the run_abort row if there is one, and the first and last wall_clock of
the telemetry.

**Block B: laptop (PowerShell).** Needs:
- block A's outputs, downloaded;
- both pilot runs ended, with each log's two lines recorded (either pair);
- the live checkout at `<PIN>`, mounted whole and read-only (DL-62).

The values:
- `<RUNS>` holds e5_s42_clip1.0 and e5_s42_clip5.0 as downloaded. `<EVALS>` lies outside the checkout.
- `<TEL10>` and `<TEL50>` are the telemetry sha256 values recorded on the pod: the first field of the
  `qat_telemetry.jsonl` line in `<EVID>/e5_s42_clip1.0.sha256` and in `<EVID>/e5_s42_clip5.0.sha256`.
- `<TC>` is block A's timing clip. With both runs non-finite it is 5.0 by block A's rule, and unused: the timing
  lines are skipped.

Nothing in the checkout is edited while the block runs. Long passes run with the laptop on its charger and sleep set to
Never.

```powershell
Set-Location "<CHECKOUT>"
git status -sb -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
```
Expected first line (a tracking suffix may follow the name):
```
## claude/keen-curie-u4a8ig
```
```powershell
git rev-parse HEAD
```
Expected:
```
<PIN>
```
Setup:
```powershell
$IMG  = "ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
$PIN  = "<PIN>"
$REPO = "<CHECKOUT>"
$DS   = "<TRAINVAL_DS>"
$RUNS = "<RUNS>"
$EV   = "<EVALS>"
$ATT  = "a1"
$TC   = "<TC>"
function RO([string]$src, [string]$dst) { @("--mount", "type=bind,source=$src,target=$dst,readonly") }
function RW([string]$src, [string]$dst) {
    New-Item -ItemType Directory -Force -Path $src | Out-Null
    @("--mount", "type=bind,source=$src,target=$dst")
}
$ENVV = @("--network","none",
          "-e","PLANTSEG_IMAGE_DIGEST=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
          "-e","PLANTSEG_DATA_ROOT=/data","-e","PYTHONPATH=/work/repo","-e","PYTHONDONTWRITEBYTECODE=1",
          "-e","GIT_OPTIONAL_LOCKS=0","-e","GIT_CONFIG_COUNT=1","-e","GIT_CONFIG_KEY_0=safe.directory",
          "-e","GIT_CONFIG_VALUE_0=/work/repo","-e","EXPECT_P=$PIN","-w","/work/repo") + (RO $REPO "/work/repo")
$DATA = (RO "$DS\images\train" "/data/images/train") + (RO "$DS\images\val" "/data/images/val") +
        (RO "$DS\annotations\train" "/data/annotations/train") + (RO "$DS\annotations\val" "/data/annotations/val")
$R10  = RO "$RUNS\e5_s42_clip1.0" "/runs/e5_s42_clip1.0"
$R50  = RO "$RUNS\e5_s42_clip5.0" "/runs/e5_s42_clip5.0"
$E10  = RW "$EV\$ATT\e5_s42_clip1.0" "/evals/e5_s42_clip1.0"
$E50  = RW "$EV\$ATT\e5_s42_clip5.0" "/evals/e5_s42_clip5.0"
$RT   = RO "$RUNS\e5_s42_clip$TC" "/runs/e5_s42_clip$TC"
$TT   = RW "$EV\$ATT\timing_e5_s42_clip$TC" "/timing/e5_s42_clip$TC"
$SEL  = RW "$EV\$ATT\clip_selection" "/sel"
$STATE = "ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}}"
$CHECK = "import os,subprocess,sys;a=sys.argv[1:];g=lambda *x:subprocess.run(['git','-C','/work/repo']+list(x),capture_output=True,text=True);h=g('rev-parse','HEAD').stdout.strip();q=g('status','--porcelain=v1','--untracked-files=all','--','src','configs','scripts','requirements*','docs/EVALUATION_CONTRACT.md','docs/IMPLEMENTATION_CONTRACT.md');d=len(q.stdout.splitlines()) if q.returncode==0 else 'git failed';r=[('HEAD',h,os.environ.get('EXPECT_P')),('GOVERNED_DIRTY',d,0)];c=lambda p:sum(1 for e in os.scandir(p) if e.is_file()) if os.path.isdir(p) else 'missing';s=[x.rsplit('=',1) if '=' in x else [x,''] for x in a];r+=[(p,c(p),int(n)) if n else (p,os.path.exists(p),True) for p,n in s];[print(k,v,'OK' if v==w else 'STOP') for k,v,w in r];sys.exit(0 if all(v==w for k,v,w in r) else 1)"
$REJ  = "import sys;from src.quant import qat_artifacts as A, qat_select as S;r=A.read_run_record(sys.argv[1]);A.require_complete(r);ok=r['telemetry_sha256']==sys.argv[2];print('TELEMETRY','OK' if ok else 'STOP');j=S.run_rejected(r) if ok else None;print(('REJECTED -- '+'; '.join(j['grounds']+['deviation: '+d for d in j['deviations']])) if j['rejected'] else 'NOT REJECTED') if j else None;sys.exit(0 if ok else 1)"
```
How the lines below work:
- Each line names its container, runs it, then prints its STOP fields. After the command's own output the last line
  must read `ExitCode=0 OOMKilled=false StartedAt=<time> FinishedAt=<time>`. The exception is a REPORT that names
  another exit code.
- A check line runs in the same mounts as the command after it. It prints the HEAD, the count of dirty governed paths
  (the evaluator's own list), and one count or True per path, never a name. Every line must end in OK.
- `$REJ` reads one run's own records with the lane's `run_rejected` (AM-21 item 3(a); src/quant/qat_select.py): its
  telemetry, checked against the pod's sha256, and every epoch checkpoint, checked against the sha256 that the
  telemetry records. It prints `TELEMETRY OK`, then `NOT REJECTED` or `REJECTED -- <grounds>`. The grounds name the
  step at which the telemetry first found the state non-finite, and the first step that logged a non-finite loss or
  pre-clip gradient norm. A `; deviation: …` clause names a checkpoint that fails item 1(a) with no state flag in
  the telemetry. It writes nothing.
- docker refuses a container name that exists, so no line can run twice. If the window closes, the container keeps
  running: read it with `docker logs -f <name>`, then print its STOP fields.

Command 0. An empty HEAD or `GOVERNED_DIRTY git failed STOP` means git refused the mount: STOP and use the data
session's recipe (DL-62).
```powershell
$NM = "q2_${ATT}_cmd0_git"; docker run --name $NM @ENVV $IMG git --version; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_cmd0_check"; docker run --name $NM @ENVV $IMG python -B -c $CHECK; docker inspect -f $STATE $NM
```
Expected: one git version line; then
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
```
**Rejection check** (AM-21 item 3(a)), read-only, both runs, before any conversion:
```powershell
$NM = "q2_${ATT}_rej10"; docker run --name $NM @ENVV @R10 $IMG python -B -c $REJ /runs/e5_s42_clip1.0 <TEL10>; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_rej50"; docker run --name $NM @ENVV @R50 $IMG python -B -c $REJ /runs/e5_s42_clip5.0 <TEL50>; docker inspect -f $STATE $NM
```
Expected, for each run:
```
TELEMETRY OK
NOT REJECTED
```
The second line can instead be a REPORT:
```
REJECTED -- <grounds>
```
A deviation clause in it goes into the record as a deviation of the trainer. A run whose completion line was the
non-finite pair must print REJECTED; NOT REJECTED after that pair is a STOP.

**The gate before any record conversion** (CHECK ITEM 16). It applies if either completion line was the non-finite pair,
or either check printed REJECTED. In that case, stop here: no record conversion of either pilot run starts until the
orchestrator's AM-8a ruling is recorded. The ruling is one of two:
- "no fault": record it, and go on;
- the commit of the fault report: record it. The repeat follows that ruling, in a new directory, and stands in the
  rejected run's place (item 3(e)). This block then waits for the repeat's block A.

Item 3(e) lifts a rejection once. If the repeat is rejected under item 3(a) too, that clip value is rejected for good.
No further repeat follows: the gate's ruling is recorded, and the block goes on with the repeat as a rejected run.

Laptop timing (d6), on clip `<TC>`. It converts one epoch and scores 64 VAL images as a smoke artifact, produces no VAL
score, and is not a record conversion. If block A skipped its timing (both runs non-finite), skip these four lines too
and record the REPORT line "d6 laptop timing not run: both pilot runs completed non-finite".
```powershell
$NM = "q2_${ATT}_t_check1"; docker run --name $NM @ENVV @RT @TT $IMG python -B -c $CHECK "/runs/e5_s42_clip$TC/epoch_ckpts=15" "/runs/e5_s42_clip$TC/qat_telemetry.jsonl" "/timing/e5_s42_clip$TC"; docker inspect -f $STATE $NM
```
Expected (with `<TC>` in the paths):
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip<TC>/epoch_ckpts 15 OK
/runs/e5_s42_clip<TC>/qat_telemetry.jsonl True OK
/timing/e5_s42_clip<TC> True OK
```
```powershell
$NM = "q2_${ATT}_t_convert"; docker run --name $NM @ENVV @RT @TT $IMG python -B scripts/qat_epoch_eval.py convert --run-dir "/runs/e5_s42_clip$TC" --eval-dir "/timing/e5_s42_clip$TC" --epochs 15 --purpose timing --host-label laptop; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: CONVERTED 1/1 (timing)
```
```powershell
$NM = "q2_${ATT}_t_check2"; docker run --name $NM @ENVV @DATA @RT @TT $IMG python -B -c $CHECK /data/images/train=5367 /data/images/val=846 /data/annotations/train /data/annotations/val "/runs/e5_s42_clip$TC/epoch_ckpts=15" "/timing/e5_s42_clip$TC"; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/data/images/train 5367 OK
/data/images/val 846 OK
/data/annotations/train True OK
/data/annotations/val True OK
/runs/e5_s42_clip<TC>/epoch_ckpts 15 OK
/timing/e5_s42_clip<TC> True OK
```
```powershell
$NM = "q2_${ATT}_t_score"; docker run --name $NM @ENVV @DATA @RT @TT $IMG python -B scripts/qat_epoch_eval.py score --run-dir "/runs/e5_s42_clip$TC" --eval-dir "/timing/e5_s42_clip$TC" --epochs 15 --purpose timing --host-label laptop; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: SCORED 1/1 (timing, 64 samples)
```
FinishedAt minus StartedAt of that container is d6's laptop figure for 64 images.

Clip 1.0, the record pass. Both runs get it; a rejected run's pass ends after its score line (item 3(d)).
- **A rejected run** printed REJECTED in the rejection check. Any non-zero exit of its convert or score line is a REPORT
  (item 2(c), for that run alone). Examples are a STOP, INCOMPLETE or ERROR line.
  - Paste it. That run's pass ends, and its eval directory is spent and kept.
  - A re-made conversion or scoring of that run follows only a written fault report (AM-8a), in a new evaluation
    directory (item 2(c)).
  - The block goes on with the other run and select_clip, which reports these records as stopped, failed or missing.
- **A run that is not rejected:** every failure stays a STOP.

Check, then convert:
```powershell
$NM = "q2_${ATT}_c10_check1"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B -c $CHECK /runs/e5_s42_clip1.0/epoch_ckpts=15 /runs/e5_s42_clip1.0/qat_telemetry.jsonl /evals/e5_s42_clip1.0; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip1.0/epoch_ckpts 15 OK
/runs/e5_s42_clip1.0/qat_telemetry.jsonl True OK
/evals/e5_s42_clip1.0 True OK
```
```powershell
$NM = "q2_${ATT}_c10_convert"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B scripts/qat_epoch_eval.py convert --run-dir /runs/e5_s42_clip1.0 --eval-dir /evals/e5_s42_clip1.0 --purpose record --host-label laptop --expect-telemetry-sha256 <TEL10>; docker inspect -f $STATE $NM
```
Expected, for a run whose state stayed finite:
```
RESULT: CONVERTED 15/15 (record)
```
For a run with a checkpoint that fails item 1(a), a REPORT instead: one that completed non-finite, or one whose
rejection check printed a deviation clause. `<k>` epochs converted, and the epochs whose checkpoints fail are
excluded (item 2(a); `<k>` can be 0):
```
RESULT: CONVERTED <k>/15 (record; not convertible: <eNN … e15>)
```
Check, then score:
```powershell
$NM = "q2_${ATT}_c10_check2"; docker run --name $NM @ENVV @DATA @R10 @E10 $IMG python -B -c $CHECK /data/images/train=5367 /data/images/val=846 /data/annotations/train /data/annotations/val /runs/e5_s42_clip1.0/epoch_ckpts=15 /evals/e5_s42_clip1.0; docker inspect -f $STATE $NM
```
Expected:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/data/images/train 5367 OK
/data/images/val 846 OK
/data/annotations/train True OK
/data/annotations/val True OK
/runs/e5_s42_clip1.0/epoch_ckpts 15 OK
/evals/e5_s42_clip1.0 True OK
```
```powershell
$NM = "q2_${ATT}_c10_score"; docker run --name $NM @ENVV @DATA @R10 @E10 $IMG python -B scripts/qat_epoch_eval.py score --run-dir /runs/e5_s42_clip1.0 --eval-dir /evals/e5_s42_clip1.0 --purpose record --host-label laptop; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: SCORED 15/15 (record)
```
For such a run, a REPORT instead, with the same `<k>` and epochs as its convert line. A rejected pilot run's scores
are reported and select nothing (item 3(d)):
```
RESULT: SCORED <k>/15 (record; excluded: <eNN … e15>)
```
Check, then select the epoch. This applies only to a run that is not rejected. A rejected run needs no selection
(item 3(d)): skip both lines for it. Run anyway, select_qat_epoch writes nothing. It refuses the run with
`[rejected_pilot_run]`, or earlier with `[stop_present]` or INCOMPLETE `[eval_record_missing]` when that run's
convert or score failed (item 2(c)).
```powershell
$NM = "q2_${ATT}_c10_check3"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B -c $CHECK /runs/e5_s42_clip1.0/epoch_ckpts=15 /runs/e5_s42_clip1.0/qat_telemetry.jsonl /evals/e5_s42_clip1.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c10_select"; docker run --name $NM @ENVV @R10 @E10 $IMG python -B scripts/select_qat_epoch.py --run-dir /runs/e5_s42_clip1.0 --eval-dir /evals/e5_s42_clip1.0 --expect-telemetry-sha256 <TEL10>; docker inspect -f $STATE $NM
```
Expected: the five OK lines of the first check; then, with the epoch the rule selects,
```
RESULT: SELECTED epoch <NN> (E5, seed 42, clip 1.0)
```
Clip 5.0: the same lines, one at a time, under the same rules for a rejected run:
```powershell
$NM = "q2_${ATT}_c50_check1"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B -c $CHECK /runs/e5_s42_clip5.0/epoch_ckpts=15 /runs/e5_s42_clip5.0/qat_telemetry.jsonl /evals/e5_s42_clip5.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_convert"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B scripts/qat_epoch_eval.py convert --run-dir /runs/e5_s42_clip5.0 --eval-dir /evals/e5_s42_clip5.0 --purpose record --host-label laptop --expect-telemetry-sha256 <TEL50>; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_check2"; docker run --name $NM @ENVV @DATA @R50 @E50 $IMG python -B -c $CHECK /data/images/train=5367 /data/images/val=846 /data/annotations/train /data/annotations/val /runs/e5_s42_clip5.0/epoch_ckpts=15 /evals/e5_s42_clip5.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_score"; docker run --name $NM @ENVV @DATA @R50 @E50 $IMG python -B scripts/qat_epoch_eval.py score --run-dir /runs/e5_s42_clip5.0 --eval-dir /evals/e5_s42_clip5.0 --purpose record --host-label laptop; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_check3"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B -c $CHECK /runs/e5_s42_clip5.0/epoch_ckpts=15 /runs/e5_s42_clip5.0/qat_telemetry.jsonl /evals/e5_s42_clip5.0; docker inspect -f $STATE $NM
$NM = "q2_${ATT}_c50_select"; docker run --name $NM @ENVV @R50 @E50 $IMG python -B scripts/select_qat_epoch.py --run-dir /runs/e5_s42_clip5.0 --eval-dir /evals/e5_s42_clip5.0 --expect-telemetry-sha256 <TEL50>; docker inspect -f $STATE $NM
```
Expected: the same OK lines as for clip 1.0, with the clip 5.0 paths (five, eight and five). After the second, fourth
and sixth lines:
```
RESULT: CONVERTED 15/15 (record)
RESULT: SCORED 15/15 (record)
RESULT: SELECTED epoch <NN> (E5, seed 42, clip 5.0)
```
A rejected run has no select line. One with a checkpoint that fails item 1(a) prints the REPORT forms of the first
two.

Clip selection. `$SELS` lists the qat_selection.json of each run that printed SELECTED. Drop the path of a run that did
not; with neither, `$SELS = @()`.
```powershell
$SELS = @("/evals/e5_s42_clip1.0/qat_selection.json", "/evals/e5_s42_clip5.0/qat_selection.json")
$NM = "q2_${ATT}_sc_check"; docker run --name $NM @ENVV @R10 @R50 @E10 @E50 @SEL $IMG python -B -c $CHECK /runs/e5_s42_clip1.0/qat_telemetry.jsonl /runs/e5_s42_clip5.0/qat_telemetry.jsonl @SELS /sel; docker inspect -f $STATE $NM
```
Expected, with one `True OK` line for each path in `$SELS`:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip1.0/qat_telemetry.jsonl True OK
/runs/e5_s42_clip5.0/qat_telemetry.jsonl True OK
</evals/e5_s42_clip<c>/qat_selection.json True OK>
/sel True OK
```
select_clip, each candidate with its telemetry sha256 as recorded on the pod (CHECK ITEM 12):
```powershell
$NM = "q2_${ATT}_select_clip"; docker run --name $NM @ENVV @R10 @R50 @E10 @E50 @SEL $IMG python -B scripts/select_clip.py --candidate /runs/e5_s42_clip1.0 /evals/e5_s42_clip1.0 --expect-telemetry-sha256 <TEL10> --candidate /runs/e5_s42_clip5.0 /evals/e5_s42_clip5.0 --expect-telemetry-sha256 <TEL50> --out /sel/clip_selection.json; docker inspect -f $STATE $NM
```
Expected, one of three outcomes.

- **Neither run rejected:** the winner the rule gives.
  ```
  RESULT: CLIP SELECTED <1.0 or 5.0> (tie <true or false>)
  ```
- **One run rejected:** a REPORT line for it, then the other run's win, which does not wait (items 3(b) and 3(d)).
  `<R>` is the rejected clip and `<W>` the other.
  ```
  REPORT: clip <R> rejected: non-finite state (AM-21 item 3) -- item 3(a): <grounds>; its records (item 3(d), reported, select nothing): conversion <state> (<detail>); scoring <state> (<detail>)
  RESULT: CLIP SELECTED <W> (tie false; clip <R> rejected: non-finite state (AM-21 item 3))
  ```
  The REPORT line enters the decision log verbatim.
  - The grounds name the step at which the state was first found non-finite (AM-21 item 5).
  - Each `<state>` is present, missing, stopped or failed.
  - A present scoring lists the scores, which select nothing.
  - A deviation clause (`; deviation: …`) is reported as a deviation of the trainer (item 3(a)).
- **Both runs rejected:** a REPORT line for each, then
  ```
  RESULT: REFUSED [no_winner] -- no winner (AM-21 item 3): a new amendment decides the clip value. Nothing was written.
  ```
  This is item 3(c)'s halt, a REPORT with ExitCode=2. Record it that day. The block ends here, with no finalize, and no
  later QAT run launches until a new amendment decides the clip value.

Finalize, for the winner only. `<W>` is the value the RESULT line printed, 1.0 or 5.0.
```powershell
$W    = "<W>"
$RWIN = RO "$RUNS\e5_s42_clip$W" "/runs/e5_s42_clip$W"
$EWIN = RW "$EV\$ATT\e5_s42_clip$W" "/evals/e5_s42_clip$W"
$NM = "q2_${ATT}_fz_check"; docker run --name $NM @ENVV @RWIN @EWIN @SEL $IMG python -B -c $CHECK "/runs/e5_s42_clip$W/epoch_ckpts=15" "/evals/e5_s42_clip$W/qat_selection.json" /sel/clip_selection.json; docker inspect -f $STATE $NM
```
Expected, with the winner's paths:
```
HEAD <PIN> OK
GOVERNED_DIRTY 0 OK
/runs/e5_s42_clip<W>/epoch_ckpts 15 OK
/evals/e5_s42_clip<W>/qat_selection.json True OK
/sel/clip_selection.json True OK
```
```powershell
$NM = "q2_${ATT}_finalize"; docker run --name $NM @ENVV @RWIN @EWIN @SEL $IMG python -B scripts/qat_epoch_eval.py finalize --run-dir "/runs/e5_s42_clip$W" --eval-dir "/evals/e5_s42_clip$W" --clip-selection /sel/clip_selection.json; docker inspect -f $STATE $NM
```
Expected:
```
RESULT: FINALIZED (E5, seed 42, epoch <NN>)
```
The value for the decision-log row and for every later launch (P10):
```powershell
(Get-FileHash "$EV\$ATT\clip_selection\clip_selection.json" -Algorithm SHA256).Hash.ToLower()
```
Expected: 64 hex characters.

**Runs of record.** E5 seeds 43 and 44, and E6, reuse block B's rejection check, convert, score and select lines with
their own paths.
- **The hold (item 1(f)).** After a non-finite completion line, or a rejection check that prints any ground, no record
  conversion of that run starts until the orchestrator's AM-8a ruling is recorded. A repeat under AM-8a needs its
  fault report before any record conversion (DL-85). For a run of record the check's REJECTED line rejects nothing,
  since item 3 governs the pilot runs; it marks the hold.
- **The select line** has two more REPORT outcomes. The first is a selection over the epochs that remain (item 2(a)),
  with the excluded epochs reported:
  ```
  RESULT: SELECTED epoch <NN> (<stage>, seed <s>, clip <c>; excluded: <eNN … e15>)
  ```
  The second is item 2(b)'s entry, below.

Item 2(b)'s outcome:
```
RESULT: REFUSED [no_convertible_epoch] -- non-finite: no model: every epoch of <run_id> is excluded (non-finite state, AM-21 item 2(a)); no epoch is selected and the run stays complete (item 2(b)): enter it in the decision log today as "non-finite: no model". Nothing was written.
```
That line exits 2.
- Enter the run in the decision log that day as "non-finite: no model".
- The run stays complete and does not move the freeze (AM-19 item 1(d)).
- Nothing that needs its model is computed or launched until a new amendment settles it (for E6 seed 42: the AM-3
  trigger and E6-KD).

For the two pilot runs neither outcome can occur. A pilot run that is not rejected has only finite checkpoints, and a
rejected one is refused first with `[rejected_pilot_run]`; item 2(b) does not apply to it (item 3(d)).

**The decision log that day** (Q2-4 ruling 1 and this section's REPORT rule; AM-21 item 2(b)) takes, verbatim:
- every REPORT line above;
- the non-finite completion lines, with the step at which each state was first found;
- the rejection checks' REJECTED lines;
- a rejected run's convert, score and select_clip REPORT lines;
- the AM-8a rulings of the gate and of a run of record's hold;
- a run of record's SELECTED line with excluded epochs;
- the no-winner halt;
- for a run of record, "non-finite: no model".

Chapter 4's disclosure (AM-21 item 5) draws on these entries.

A local STOP report carries the container's name, its STOP-fields line and the last 20 lines of `docker logs` for it.
