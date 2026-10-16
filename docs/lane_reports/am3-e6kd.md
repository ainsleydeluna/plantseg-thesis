# Lane report: L-AM3 (E6-KD), lane/am3-e6kd

## B-1: the AM-3 trigger

Cloud lane session (DL-37), Fri 16 Oct 2026, under the orchestrator's GO for sub-phase B-1 of the Phase A plan
(lam3_phase_a_plan_20261015.md; 78,551 B and sha256 91ca78a0…, as the GO gives them) as amended by the rulings
LB1-LB12.

Labels: MEASURED (observed in this session), DOCUMENTED (read in the named document), REPOSITORY-PROVEN (follows from
the code at the cited commit), INFERRED. Code lines are at 7cfcb48 (W5, the last code commit). This report commit
(C4) adds only this file, so they hold at C4. Doc lines are at O; the lane changes no document but this one.
Abbreviations: PA docs/PREREGISTRATION_AMENDMENTS.md, EC docs/EVALUATION_CONTRACT.md, IC
docs/IMPLEMENTATION_CONTRACT.md, LS docs/lane_specs/part2.md, DL docs/DECISION_LOG.md, "module"
src/quant/e6kd_trigger.py, "CLI" scripts/e6kd_trigger.py, "smoke" scripts/smoke_e6kd_trigger.py, "mutations"
scripts/smoke_e6kd_mutations.py.

**Acceptance, item by item:**
- STEP 4 reproduced the baselines (B-1.2).
- STEP 7 passed at W5: FROZEN BLOBS OK (10/10), the six Q2-era counts unchanged, smoke_e6kd_trigger at 101 with every
  change from the plan's 48 named (B-1.7), and every mutation and refusal row KILLED with Q2's self-checks passing
  (B-1.10, B-1.5).
- Every new code has a case and a killed mutation: 51 codes at 53 sites (B-1.5).
- No final run created a path whose name holds the token SL-1 forbids or ':'. One development run did, inside its own
  temp dir (B-1.14, slip 4; fixed in W4).
- One DL-24 workflow; every finding is fixed, recorded or refuted by name (B-1.11).
- No trailers. Pushes, of lane/am3-e6kd only: push 1 at C2, and push 2 after this commit (STEP 9).
- The scoped status after this commit is measured in the session and reported in its reply.

### B-1.1 Base, commits, scope and environment

- **O** = origin/master = `2b3b55651e8092fd7fff6e786b73d971096deb17`, Q2's merge (parents 31e12c4d…, 850f9e89…)
  (MEASURED, STEP 1, after `git fetch --unshallow origin`, exit 0). B-1 starts from O.
- **Branch** `lane/am3-e6kd`, made at O (MEASURED, STEP 2). Only this branch was committed to or pushed.
- **Commits.** Each was made by explicit path with `git commit --only -F <file> -- <paths>`, after the scoped
  `git diff --stat` and `git status -sb`. None has a trailer (MEASURED: `git log --format='%H %(trailers:only,unfold)'`
  over O..W5 prints no trailer line).

| Label | Commit | Files | What |
|---|---|---|---|
| C1 | `44e9ca8b3dcc6ae0dc452bbd207858d1fb09183b` | module (+649), CLI (+86) | score, decide, read_trigger_record, trigger_binding; the CLI |
| C2 | `fb9064c3c4c73fc15c98b5d2b62724dad8cb62fb` | smoke (+1219) | the smoke, 81 checks; push 1 |
| C2b | `263d0b2774d2b090f46f689f126ba0a31ca9d148` | module, smoke | LB7 in code; LB2's index-named temp dirs; one case made runnable alone; 83 checks |
| C3 | `578cd026539ae6ebb1b9e1a90920abd7bb8a2076` | mutations (+226) | MT1-MT14 and the refusal table, 53 sites (its message says 52: B-1.14) |
| W1 | `54f7a407765e8ffa229d5a9c9c1e14953d37b834` | module, smoke, mutations | DL-24 F1: decide binds its code to score's; MT15; 84 checks |
| W2 | `231371c97c805d35d56391786c9aa87a14ee76d7` | module, smoke, mutations | DL-24 R-F2: decide's RESULT line marks smoke inputs; MT16; 85 checks |
| W3 | `17a4486d5e426c541b3a6ab86b25c136989ad09f` | smoke, mutations | DL-24 S-F2: each condition of the two summary gates gets its own case; MT17-MT19; 101 checks |
| W4 | `a4b8151ad9d7508c049c034e8b485e24e8412d29` | smoke | DL-24 S-F5: g1 also fails a later gate, so a removed name check creates nothing |
| W5 | `7cfcb4835a3422e882dac60d3e5a0c5fa3e43574` | mutations | text only: "MT1-MT19" in the docstring, the help and the banner |
| C4 | this commit | docs/lane_reports/am3-e6kd.md | this report |

W1-W5 are the commits after C3. The session's notes called them C5-C9; they are renamed here because plan §7 numbers
B-2's commits C5-C11.

- **Scoped diff** (MEASURED at W5:
  `git diff --stat 2b3b556… -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'`):
  ```
   scripts/e6kd_trigger.py         |   86 ++++++
   scripts/smoke_e6kd_mutations.py |  257 ++++++++++++++++
   scripts/smoke_e6kd_trigger.py   | 1274 ++++++++++++++++++++++++++++
   src/quant/e6kd_trigger.py       |  665 ++++++++++++++++++++++
   4 files changed, 2282 insertions(+)
  ```
  C4 adds this file, so five new files and no edit to an existing file, as plan §0.1 and the GO require.
- **The venv against the lock** (MEASURED, STEP 3): `/usr/bin/python3.11 -m venv` (Python 3.11.17) in the session
  scratchpad. Every PyPI pin equals requirements.lock: numpy 1.26.4, scipy 1.11.4, pillow 12.3.0, statsmodels 0.14.6,
  patsy 1.0.2, pandas 3.0.3, python-dateutil 2.9.0.post0, six 1.17.0, tzdata 2026.2, packaging 24.2, filelock 3.14.0,
  typing_extensions 4.15.0, sympy 1.14.0, mpmath 1.3.0, networkx 3.6.1, Jinja2 3.1.6, MarkupSafe 3.0.3,
  fsspec 2026.4.0, requests 2.28.2, certifi 2026.6.17, charset-normalizer 3.4.7, idna 3.18, urllib3 1.26.20. Two
  differ by build: torch 2.1.0+cpu and torchvision 0.16.0+cpu (installed `--no-deps` from download.pytorch.org/whl/cpu)
  against the lock's +cu121 builds of the same versions; the session has no GPU. Two are not in the lock: pip 24.0 and
  setuptools 79.0.1 (ensurepip). `pip check`: "No broken requirements found". Quantized engines: qnnpack, none,
  onednn, x86, fbgemm; 4 threads.

### B-1.2 RESULT lines

| Smoke | Baseline: STEP 4 at O, 2026-10-16 14:32:00Z-14:51:22Z | Final: STEP 7 at W5, 2026-10-16T17:04:49Z-17:45:53Z | Seconds (final) |
|---|---|---|---|
| smoke_frozen_blobs | `RESULT: FROZEN BLOBS OK (10/10)` | `RESULT: FROZEN BLOBS OK (10/10)` | 0 |
| smoke_eval_int8 | `RESULT: PASS (32/32)` | `RESULT: PASS (32/32)` | 7 |
| smoke_eval_stage_artifacts | `RESULT: PASS (66/66)` | `RESULT: PASS (66/66)` | 5 |
| smoke_qat_runner | `RESULT: PASS (214/214)` | `RESULT: PASS (214/214)` | 132 |
| smoke_qat_selection | `RESULT: PASS (174/174)` | `RESULT: PASS (174/174)` | 919 |
| smoke_qat_artifacts `--sections records` | `RESULT: PASS (42/42)` | `RESULT: PASS (42/42)` | 45 |
| smoke_e6kd_trigger | new | `RESULT: PASS (101/101)` | 338 |
| smoke_e6kd_mutations `--workers 2` | new | `RESULT: PASS (102/102)` | 1018 |

Each smoke ran alone, one after the other, with the lane venv, `PYTHONDONTWRITEBYTECODE=1` and PLANTSEG_DATA_ROOT unset (the session's runner script). Nothing else was running (MEASURED with `ps` before the start). smoke_e6kd_trigger ran without a cache, so its fixtures were built fresh. Logs: the session scratchpad's runs_baseline/ and runs_final/.

Changed counts: none of the six Q2-era smokes changed (B-1 adds files only). Two smokes are new:
- **smoke_e6kd_trigger: 101 checks** against the plan's 48 (§4.1). B-1.7 names every case behind the difference.
- **smoke_e6kd_mutations: 102 checks.** Q2's harness checks: 28 (26 self-checks, `self_check_measured` ("26 checks") and `refusal_table_names_only_real_sites`), then 72 edits, each a `<id>_killed` check (MT1-MT19 and the 53 refusal rows), then `every_job_has_a_verdict` and `harness_ran_unsignalled`: 28 + 72 + 2 = 102. Plan §4.6 counted Q2's 28 checks, its 48 mutations and the refusal rows. B-1's share of those is MT1-MT14 and e6kd_trigger.py's rows. B-1 adds MT15-MT19 (DL-24), and B-2 brings the MK rows and qat_kd.py's.

### B-1.3 The rulings at the final commit

- **LB1** (F-03; AM-22 v6 item 1(a)2). The rule is text and an absence, so it has no code of its own.
  - §7's line becomes: "B-1 merged and pinned before E6 seed 42's block B ends. The trigger is scored, decided,
    committed alone with its DL row and entered by the end of the second day after the day E6 seed 42 is complete
    (AM-22 v6 item 1(a)2); a later entry is a deviation, reported in Chapter 4".
  - §3(xii)'s schedule check, restated against AM-22: D no later than 2026-11-02; the fallback freeze 2026-11-03.
  - No date enters the trigger JSON. Its 33 keys (module :552-578, listed in B-1.6) hold no clock and no date, and
    the docstring says so (module :13-14). d1_json_keys_exact (smoke :984) fixes the key set, and
    decide_twice_identical_bytes (smoke :1014) requires identical bytes from two decides. MT12, which adds a
    wall-clock key, is KILLED by it. trigger_eval.json and trigger_STOP.json do carry `created_wall_clock` (module
    :416, :405): they record the scoring, and the trigger JSON holds only trigger_eval.json's sha256.
  - The dates go into the DL row (B-1.13, row 2).
- **LB2** (F-07). Implemented.
  - The three renames: the refusal code `trigger_forbidden_path` (module :606); the checks
    `g1_forbidden_names_refused_before_any_probe` (smoke :649) and `no_forbidden_named_path_created` (smoke :1262).
  - No path is derived from a check name. The refusal loops name their temp dirs by index (`out_NN` smoke :883,
    `art_NN` smoke :1062), and the d1 cases use `dec_*` (smoke :911-977). C2 had derived them from the check names;
    C2b changed that (B-1.14, slip 2).
  - No check name, refusal code, file or folder holds the token (MEASURED over the final run's 101 check names, the 51
    codes of B-1.5, the four files and the mutation job ids). The token occurs in the lane's files only as:
    - prose (smoke :6, :52, :562);
    - zipfile's method `writestr` (smoke :473-474);
    - Q2's helper `names_test` (module :22, :211, :605; mutations :95). LB2 keeps Q2's helper names;
    - three path strings that are never created and must be refused: `nowhere_latest_dir` (smoke :653), the symlink
      target `attestations/run` (smoke :666; the link itself is `alias_run`) and `trig/latest.json` (smoke :1092).
      These are the GO's "never-created path".
    The smoke's own scan spells the token as a join (smoke :1216).
  - No final run created such a path: no_forbidden_named_path_created PASS in the final smoke run, which walks the
    smoke's whole temp dir. A development run did (B-1.14, slip 4; DL-24 S-F5; fixed in W4).
- **LB3.** `d1_e6_value_is_the_rescore` is dropped (absent from the 101; MEASURED). E6's value is the re-score by
  construction (module :521), and its equality with block B is refuses_e6_rescore_differs (smoke :1037; MT6 KILLED by
  it). The new count is 101 (B-1.7).
- **LB4.** Recorded in B-1.14.
- **LB5.** drop_decision (module :176-185) takes `Decimal(repr(x))` of both values (:180), the drop E3 − E6 (:181)
  and its percentage points (:182), and compares strictly with `THRESHOLD_PP = Decimal("1.0")` (:82, :183). The record
  carries the two floats `e3_val` and `e6_val`, the two decimal strings `e3_val_decimal` and `e6_val_decimal`, `drop`,
  `drop_pp`, the float `drop_pp_float`, `threshold` 1.0 and `threshold_decimal` "1.0" (module :553-555). Cases:
  d1_drop_1_01_triggered, d1_drop_1_00_not_triggered, d1_drop_0_50_not_triggered, d1_negative_drop_not_triggered,
  d1_decimal_not_float, d1_threshold_rule_recorded (smoke :910-952). MT1-MT4 are KILLED. The launch gate recomputes
  the decision from the record's own two values (module :629-633, `trigger_inconsistent`).
- **LB6.** score re-scores E6 in the trigger session (the E6 command, module :357-359, run at :394-415). decide refuses
  a re-score that differs from block B's winner value with `e6_rescore_differs` (module :524-527). The refusal's
  message calls it "a STOP for diagnosis and a ruling, never a choice between versions (AM-21 item 2(c))". Its case is
  refuses_e6_rescore_differs (smoke :1037); MT6 and the refusal row are KILLED. The operator block treats the refusal
  as a STOP (B-1.12). DL-24's D-F2 (refuted, B-1.11) asked whether the refusal should also spend the trigger eval dir.
- **LB7.** The image of record and the CPU at batch 1.
  - `image_not_of_record` (module :224-227) refuses unless PLANTSEG_IMAGE_DIGEST names `Q.IMAGE_DIGEST_OF_RECORD`
    (src/quant/qat.py:105, `sha256:b80b645d…`). Each summary's `run.eval_runtime.image_digest` must name it too (module
    :435-437), and the two summaries must share it, non-null (module :97-100).
  - The literal commands pass `--device cpu --batch-size 1` (module :188-193). decide checks `run.env.device` and
    `eval_runtime.batch_size` against the rules' evaluator values (module :426-432).
  - Cases: refuses_image_not_of_record (smoke :687), refuses_summary_not_the_student_image (smoke :1039),
    refuses_summary_not_cpu (smoke :1030), refuses_summary_batch_not_1 (smoke :1032),
    score_runs_the_literal_evaluator_commands (smoke :1173). MT13 (batch 16) is KILLED.
  - `allow_smoke_inputs` relaxes the image check for the smokes only (module :225, :435). The launch gate refuses a
    record that carries it (`trigger_smoke_inputs`, module :625-626).
- **LB8.** The sentence is verbatim at module :17-18. read_trigger_record compares the file with its bytes at HEAD
  through `Q._git_head_bytes` (module :610-615: `trigger_not_tracked`, `trigger_changed_since_head`). Cases:
  refuses_trigger_untracked and refuses_trigger_changed_since_head (smoke :1094, :1095, in the gate loop at
  :1088-1104).
- **LB9.** The four clause-map rows are in B-1.8.
- **LB10.** Not B-1's: no B-1 file reads projection.pt.
- **LB11.** For B-2: E6-KD's own rules loader, and the qat_select.py :448 choice. B-1 reads E6's own rules with Q2's
  loader (`S.load_rules()`, module :300), as block B does.
- **LB12.** Every other plan answer stands. B-1.9 names where the build differs from the plan. B-1.7 and B-1.10 give the
  final counts and the cases behind each change.

### B-1.4 The CLI

```
python -B scripts/e6kd_trigger.py score  IDENTITY --trigger-eval-dir NEWDIR --host-label TEXT [--allow-smoke-inputs]
python -B scripts/e6kd_trigger.py decide IDENTITY --trigger-eval-dir DIR --host-label TEXT --out NEWFILE [--allow-smoke-inputs]

IDENTITY = --e3-run-dir DIR --e3-checkpoint-sha256 H
           --e6-run-dir DIR --e6-eval-dir DIR --e6-telemetry-sha256 H --e6-selection-sha256 H
           --lambda-selection REL --lambda-selection-sha256 H --alpha-selection REL --alpha-selection-sha256 H
```
- Every flag is required (CLI :41-47). `--allow-smoke-inputs` is the smokes' (CLI :48-49).
- `score` checks steps 1-8 (module :202-348, listed in its docstring at :20-39). It then runs the two literal evaluator
  commands one after the other, from the repository root (module :394-415), and writes trigger_eval.json once (:417).
  It prints `[score] E3: all-class mIoU …`, `[score] E6: …`, `[score] <dir>` and `RESULT: SCORED 2/2 (record)`, or
  `(smoke inputs)` instead of `(record)` (CLI :73-77).
- `decide` checks steps 1-8 and the trigger eval dir again, then the two summaries (module :453-527). It writes `--out`
  once (:579) and prints the five rule-trace lines, `[decide] <out>`, and one of:
  - `RESULT: TRIGGERED -- drop D pp > 1.0 pp (E3 X, E6 Y; seed 42)`;
  - `RESULT: NOT TRIGGERED -- drop D pp <= 1.0 pp (E3 X, E6 Y; seed 42)`.
  ` (smoke inputs, not a record)` follows the verdict when smoke inputs were used (module :583-590; CLI :78-81).
- Exit codes (CLI :61-72; src/quant/qat.py:130): 0 done; 2 `RESULT: REFUSED [code] -- … Nothing was written.`; 3
  `RESULT: INCOMPLETE [code] -- …`; 1 `RESULT: STOP [code]` (no B-1 code raises it); 4 `RESULT: ERROR [type] -- …`.
- The two literal commands (module :188-193, :351-359; run ids :196-198):
  ```
  <python> -B scripts/evaluate_model.py --stage E3 --model-role student --precision fp32 --split val --protocol canvas
     --checkpoint <E3 run dir>/<best.json's file> --artifact-status provisional --device cpu --batch-size 1
     --out-dir <T>/scores/e3 --run-id e6kd_trigger_e3_<checkpoint sha256[:12]>
  <python> -B scripts/evaluate_model.py --stage E6 --model-role student --precision int8_qat --split val --protocol canvas
     --provenance <E6 eval dir>/<the winner's provenance> --artifact-status provisional --device cpu --batch-size 1
     --out-dir <T>/scores/e6 --run-id e6kd_trigger_e6_e<NN>_<epoch checkpoint sha256[:12]>
  ```
  No `--threads` and no thread variable: torch_num_threads is the evaluator's own record (module :101-102), as in
  block B. The two summaries must agree on it (module :97-99).

### B-1.5 Every refusal code, its line, and the cases that kill its removal

51 codes at 53 sites (`trigger_missing` and `trigger_format` have two each). The Exit column gives each site's exit
code (2 refused, 3 incomplete). Each row is also a row of the mutation smoke's refusal table (`KILLS`, mutations
:143-205): the statement is replaced by `pass`, and the named cases KILLED it at the final run (B-1.10).

| Function | Code | module line | Exit | Killed by (final run) |
|---|---|---|---|---|
| check_inputs | `forbidden_path` | :212 | 2 | g1_forbidden_names_refused_before_any_probe |
| check_inputs | `sha256_format` | :221 | 2 | refuses_sha256_format |
| check_inputs | `host_label` | :223 | 2 | refuses_host_label_empty |
| check_inputs | `image_not_of_record` | :226 | 2 | refuses_image_not_of_record |
| check_inputs | `trigger_eval_dir_not_outside_repo` | :234 | 2 | refuses_trigger_eval_dir_inside_repo |
| check_inputs | `trigger_eval_dir_not_fresh` | :236 | 2 | refuses_trigger_eval_dir_not_fresh |
| check_inputs | `stop_present` | :242 | 2 | refuses_stop_file_present |
| check_inputs | `trigger_eval_missing` | :244 | 3 | refuses_decide_before_score_incomplete |
| check_inputs | `out_not_outside_repo` | :248 | 2 | refuses_out_inside_repo |
| check_inputs | `out_exists` | :250 | 2 | refuses_out_exists |
| check_inputs | `e3_checkpoint_not_alpha_winner` | :261 | 2 | refuses_e3_checkpoint_not_alpha_winner_sha |
| check_inputs | `e6_telemetry_sha256_mismatch` | :268 | 2 | refuses_e6_telemetry_sha256_mismatch |
| check_inputs | `e6_stage_not_e6` | :272 | 2 | refuses_e6_stage_not_e6 |
| check_inputs | `e6_seed_not_42` | :274 | 2 | refuses_e6_not_seed_42 |
| check_inputs | `e6_mode_not_real` | :276 | 2 | refuses_e6_mode_not_real |
| check_inputs | `e6_parent_not_e3_checkpoint` | :281 | 2 | refuses_e6_parent_not_the_e3_checkpoint |
| check_inputs | `e6_selections_mismatch` | :288 | 2 | refuses_e6_selections_mismatch |
| check_inputs | `e6_clip_binding` | :296 | 2 | refuses_e6_clip_binding |
| check_inputs | `e6_no_model` | :309 | 2 | refuses_e6_no_model |
| check_inputs | `e6_selection_missing` | :314 | 3 | refuses_e6_selection_missing_incomplete |
| check_inputs | `e6_selection_sha256_mismatch` | :317 | 2 | refuses_e6_selection_sha256_mismatch |
| check_inputs | `e6_selection_format` | :325 | 2 | refuses_e6_selection_format |
| check_inputs | `e6_selection_differs` | :331 | 2 | refuses_e6_selection_recompute_differs |
| check_inputs | `data_root_not_trainval_only` | :340 | 2 | refuses_data_root_not_trainval |
| score | `evaluator_failed` | :406 | 3 | refuses_evaluator_failure_spends_the_dir |
| decide | `trigger_eval_format` | :467 | 2 | refuses_trigger_eval_format |
| decide | `trigger_eval_smoke_inputs` | :469 | 2 | smoke_inputs_refused_for_record |
| decide | `trigger_eval_mismatch` | :477 | 2 | refuses_eval_dir_mixed_commit, refuses_code_changed_since_score |
| decide | `evaluator_command_not_literal` | :483 | 2 | refuses_evaluator_command_not_literal |
| decide | `scores_dir_contents` | :489 | 2 | refuses_scores_dir_contents |
| decide | `summary_artifact_invalid` | :502 | 2 | refuses_summary_artifact_invalid |
| decide | `summary_sha256_mismatch` | :504 | 2 | refuses_summary_sha256_mismatch |
| decide | `summary_values` | :510 | 2 | refuses_e3_summary_not_val, refuses_e6_summary_not_val, refuses_e3_summary_not_fp32, refuses_e6_summary_not_int8_qat_qnnpack, refuses_summary_not_cpu, refuses_summary_not_canvas, refuses_summary_batch_not_1, refuses_rows_not_846_for_record, refuses_e6_summary_not_the_winner_artifact, refuses_e3_summary_not_the_checkpoint, refuses_summary_not_the_student_image, refuses_summary_governed_paths_dirty, refuses_summary_not_provisional, refuses_summary_workers_not_0, refuses_summary_run_id_not_the_trigger_run, refuses_summary_stage_mismatch, refuses_summary_role_not_student, refuses_summary_random_init, refuses_e3_summary_backend_not_null, refuses_e6_summary_not_int8_qat, refuses_summary_gt_support_sum, refuses_summaries_scored_at_another_commit |
| decide | `summaries_not_same_code_path` | :518 | 2 | refuses_summaries_not_same_code_path, refuses_summaries_differ_in_metric_code, refuses_summaries_differ_in_split_manifest, refuses_summaries_differ_in_class_map, refuses_summaries_differ_in_torch, refuses_summaries_differ_in_metric_protocol |
| decide | `score_invalid` | :523 | 2 | refuses_score_invalid |
| decide | `e6_rescore_differs` | :525 | 2 | refuses_e6_rescore_differs |
| read_trigger_record | `trigger_missing #0` | :598 | 2 | refuses_trigger_flag_missing |
| read_trigger_record | `trigger_sha256_format` | :600 | 2 | refuses_trigger_sha256_format |
| read_trigger_record | `trigger_path_not_repo_relative` | :603 | 2 | refuses_trigger_path_not_repo_relative |
| read_trigger_record | `trigger_forbidden_path` | :606 | 2 | refuses_trigger_forbidden_path |
| read_trigger_record | `trigger_missing #1` | :608 | 2 | refuses_trigger_file_missing |
| read_trigger_record | `trigger_not_tracked` | :612 | 2 | refuses_trigger_untracked |
| read_trigger_record | `trigger_changed_since_head` | :615 | 2 | refuses_trigger_changed_since_head |
| read_trigger_record | `trigger_sha256_mismatch` | :617 | 2 | refuses_trigger_sha256_mismatch |
| read_trigger_record | `trigger_format #0` | :622 | 2 | refuses_trigger_not_json |
| read_trigger_record | `trigger_format #1` | :624 | 2 | refuses_trigger_wrong_format |
| read_trigger_record | `trigger_smoke_inputs` | :626 | 2 | refuses_trigger_smoke_inputs |
| read_trigger_record | `trigger_seed` | :628 | 2 | refuses_trigger_seed |
| read_trigger_record | `trigger_inconsistent` | :632 | 2 | refuses_trigger_inconsistent |
| read_trigger_record | `trigger_not_triggered` | :635 | 2 | refuses_trigger_not_triggered |
| trigger_binding | `trigger_e3_mismatch` | :647 | 2 | refuses_trigger_e3_mismatch |
| trigger_binding | `trigger_clip_mismatch` | :651 | 2 | refuses_trigger_clip_mismatch |
| trigger_binding | `trigger_selection_mismatch` | :656 | 2 | refuses_trigger_selection_mismatch |

### B-1.6 The key lists

**e6kd_trigger.json, format `e6kd_trigger/1`** (module :552-578; sorted keys, A.json_bytes). 33 keys:
`format`, `lane`, `rule`, `seed`, `e3_val`, `e6_val`, `e3_val_decimal`, `e6_val_decimal`, `drop`, `drop_pp`,
`drop_pp_float`, `threshold`, `threshold_decimal`, `threshold_unit`, `decision_rule`, `arithmetic`, `triggered`,
`consequence`, `descriptive`, `holm_family`, `e3`, `e6`, `selections`, `same_code_path`, `input_sha256s`,
`trigger_eval`, `git_head`, `code_clean_at_head`, `code_files_sha256`, `host_label`, `cpu_model`, `smoke_inputs`,
`rule_trace`. Nested:
- `e3` (10): `run_id`, `checkpoint`, `checkpoint_sha256`, `best_json_sha256`, `run_meta_sha256`, `telemetry_sha256`,
  `best_json_value_not_used`, `summary` {`path`, `sha256`}, `eval_run_id`, `evaluator_command`.
- `e6` (17): `run_id`, `telemetry_sha256`, `clip_norm`, `clip_selection_sha256`, `qat_selection` {`path`, `sha256`},
  `rules_sha256`, `epoch`, `checkpoint_sha256`, `artifact_of_record`, `provenance`, `block_b_summary`,
  `selection_value`, `excluded_epochs`, `summary` {`path`, `sha256`}, `eval_run_id`, `evaluator_command`,
  `rescore_equals_selection`.
- `selections`: `lambda` and `alpha`, each {`path`, `sha256`, `winner`}.
- `same_code_path`: the ten SAME_CODE_PATH_KEYS (module :97-99) with E3's values: `run.repo_commit`,
  `run.metric_impl_sha256`, `dataset.split_manifest_sha256`, `dataset.class_map_sha256`, `run.env.torch`,
  `run.env.device`, `dataset.preprocess_protocol`, `run.eval_runtime.torch_num_threads`,
  `run.eval_runtime.image_digest`, `metric_protocol`.
- `input_sha256s` (17): `e3_checkpoint`, `e3_best_json`, `e3_run_meta`, `e3_telemetry`, `e6_telemetry`,
  `e6_clip_selection`, `e6_qat_selection`, `qat_selection_rules`, `e6_epoch_checkpoint`, `e6_artifact_of_record`,
  `e6_provenance`, `e6_block_b_summary`, `lambda_selection`, `alpha_selection`, `e3_trigger_summary`,
  `e6_trigger_summary`, `trigger_eval`.
- `trigger_eval`: {`path`, `sha256`}.

d1_json_keys_exact (smoke :984) and d1_input_sha256s_complete (smoke :992) check these sets.

**trigger_eval.json, format `e6kd_trigger_eval/1`** (module :386-390, :416). 16 keys: `format`, `lane`, `inputs`,
`host_label`, `cpu_model`, `git_head`, `code_clean_at_head`, `code_files_sha256`, `smoke_inputs`, `threads_note`,
`affinity_cpus`, `data_root`, `isolation_counts`, `status` ("scored"), `scores`, `created_wall_clock`. Nested:
- `inputs` (16; module :362-373): `e3_run_dir`, `e3_run_id`, `e3_checkpoint`, `e3_checkpoint_sha256`, `e6_run_dir`,
  `e6_run_id`, `e6_eval_dir`, `e6_telemetry_sha256`, `e6_selection_sha256`, `e6_epoch`, `e6_checkpoint_sha256`,
  `e6_artifact_of_record`, `e6_provenance`, `lambda_selection` {`path`, `sha256`}, `alpha_selection` {`path`,
  `sha256`}, `smoke_inputs`.
- `scores.e3` and `scores.e6` (9 each; module :410-414): `evaluator_command`, `evaluator_python`, `exit_code`,
  `seconds`, `run_id`, `score_dir`, `summary` {`path`, `sha256`}, `all_class_miou`, `torch_num_threads`.

trigger_STOP.json (module :401-405), written only on an evaluator failure, holds trigger_eval.json's first 13 keys
with `status` "STOP", `failed`, `evaluator_exit_code`, `stdout_tail`, `stderr_tail`, `scored` and
`created_wall_clock`.

decide requires its own `git_head`, `code_files_sha256`, `code_clean_at_head`, `host_label`, `cpu_model` and
`inputs` to equal trigger_eval.json's (module :471-478, W1).

### B-1.7 The smoke: 101 checks against the plan's 48

The final run's 101 checks (MEASURED, all PASS) against plan §4.1's 48: 48 − 1 + 15 + 19 + 2 + 1 + 1 + 16 = 101.

- **−1 (LB3):** `d1_e6_value_is_the_rescore` is dropped.
- **Renamed (LB2), counted once:** `g1_…named_paths_refused_before_any_probe` → `g1_forbidden_names_refused_before_any_probe`; `no_…named_path_created` → `no_forbidden_named_path_created`.
- **Kept:** the other 45 names of §4.1, unchanged.
- **+15, a code the plan described without a case (one case each):** `refuses_sha256_format`, `refuses_host_label_empty`, `refuses_trigger_eval_dir_inside_repo`, `refuses_e6_mode_not_real`, `refuses_e6_selections_mismatch`, `refuses_e6_clip_binding`, `refuses_e6_selection_format`, `refuses_decide_before_score_incomplete`, `refuses_evaluator_failure_spends_the_dir`, `refuses_trigger_eval_format`, `refuses_evaluator_command_not_literal`, `refuses_scores_dir_contents`, `refuses_summary_artifact_invalid`, `refuses_summary_sha256_mismatch`, `refuses_score_invalid`.
- **+19, the launch gate (B-1.9 item 6):** `trigger_record_accepted_when_committed`, `refuses_trigger_flag_missing`, `refuses_trigger_sha256_format`, `refuses_trigger_path_not_repo_relative`, `refuses_trigger_forbidden_path`, `refuses_trigger_file_missing`, `refuses_trigger_untracked`, `refuses_trigger_changed_since_head`, `refuses_trigger_sha256_mismatch`, `refuses_trigger_not_json`, `refuses_trigger_wrong_format`, `refuses_trigger_smoke_inputs`, `refuses_trigger_seed`, `refuses_trigger_inconsistent`, `refuses_trigger_not_triggered`, `trigger_binding_accepted`, `refuses_trigger_e3_mismatch`, `refuses_trigger_clip_mismatch`, `refuses_trigger_selection_mismatch`.
- **+2, LB7 (C2b):** `refuses_image_not_of_record`, `refuses_summary_not_the_student_image`.
- **+1, DL-24 F1 (W1):** `refuses_code_changed_since_score`.
- **+1, DL-24 R-F2 (W2):** `d1_result_line_marks_smoke_inputs`.
- **+16, DL-24 S-F2 (W3):** `refuses_summary_governed_paths_dirty`, `refuses_summary_not_provisional`, `refuses_summary_workers_not_0`, `refuses_summary_run_id_not_the_trigger_run`, `refuses_summary_stage_mismatch`, `refuses_summary_role_not_student`, `refuses_summary_random_init`, `refuses_e3_summary_backend_not_null`, `refuses_e6_summary_not_int8_qat`, `refuses_summary_gt_support_sum`, `refuses_summaries_scored_at_another_commit`, `refuses_summaries_differ_in_metric_code`, `refuses_summaries_differ_in_split_manifest`, `refuses_summaries_differ_in_class_map`, `refuses_summaries_differ_in_torch`, `refuses_summaries_differ_in_metric_protocol`.

Also changed, at the same count: refuses_rows_not_846_for_record varies the rows alone (W3), and g1 also carries a wrong `--e6-selection-sha256` (W4). The development counts were 81 (C2), 83 (C2b), 84 (W1), 85 (W2) and 101 (W3).

### B-1.8 The clause map: the rows LB9 adds

The plan's §2 rows stand (LB12). LB9 adds these rows, quoted at O (verbatim, bold markup dropped; "…" marks a cut).

| # | Clause | Source | Implemented by | Checked by |
|---|---|---|---|---|
| D-L | "Launched. A run is launched when its run_meta row with mode "real" exists; that row's wall_clock is its launch time." | PA:1371-1372 | The trigger reads E6 seed 42's run_meta row and requires mode "real" (`e6_mode_not_real`, module :275-277); it never reads a launch time. E6-KD's launch is B-2's | refuses_e6_mode_not_real |
| D-C | "Complete. A run is complete at a date when its trainer's end-of-run record meets that date: … the last epoch's record for a QAT run; …", with "[Extended 2026-10-07 — AM-21 item 1: an E5 or E6 run that reaches a non-finite state still writes its last epoch's record.]" | PA:1374-1381 | E6 seed 42 complete through Q2's `read_run_record` and `require_complete` (module :265-266): incomplete is exit 3 (`run_incomplete`, src/quant/qat_artifacts.py:149-151). The E3 parent complete through `resolve_parent` (module :258: run_end with checks_passed, no run_abort). The date E6 seed 42 is complete goes into row 2 (B-1.13), not into the record (LB1) | refuses_missing_e3_run_dir, and Q2's own cases of these functions |
| D-D | "Derived artifacts. The PTQ of a complete parent, and the conversions and the AM-4a item 2 selection of a complete QAT run, are not runs. They are made from inputs the manifest lists, before the TEST unlock (item 4(c)). Those of E6 seed 42 are made before the freeze, because the AM-3 trigger needs them." | PA:1382-1385 | The trigger needs E6 seed 42's derived artifacts and refuses without them: a missing selection is `e6_selection_missing` (exit 3), every epoch excluded is `e6_no_model` (module :300-314). The selection is recomputed from its conversions and scores and compared key by key (module :327-331) | refuses_e6_selection_missing_incomplete, refuses_e6_no_model, refuses_e6_selection_recompute_differs |
| EC-I | "[AM-3, AM-4, 2026-09-23] INT8 VAL scoring. QAT checkpoint selection (AM-4) and the E6-KD trigger (AM-3) score VAL on the converted INT8 (QNNPACK) model on CPU, not on the fake-quant model. Code: lanes L-AM4 and L-AM3." | EC:683-685 (§7.3) | E6's score is the QNNPACK TorchScript artifact of block B's winner, scored by scripts/evaluate_model.py with `--precision int8_qat --device cpu` (module :357-359). decide requires `run.precision` int8_qat, `run.quant_backend` qnnpack, `run.env.device` cpu and `run.checkpoint_sha256` equal to the artifact's sha256 (`summary_values`, module :492-510). The fake-quant VAL score is never read | refuses_e6_summary_not_int8_qat_qnnpack, refuses_summary_not_cpu, refuses_e6_summary_not_the_winner_artifact, score_runs_the_literal_evaluator_commands |

### B-1.9 Differences from the plan

Each is named; the rulings win where they differ.

1. **The same-code-path keys drop `run.config_sha256`; this was a plan defect.** Plan §3(i) required the two summaries
   to agree on all eight of Q2's `equal_across_summaries` fields (configs/qat_selection_rules.json:21-24),
   `run.config_sha256` among them. The evaluator hashes the
   stage, the precision and the backend into that field (src/eval/artifacts.py `build_config_payload`, :256-281 at O;
   the three at :265, :267, :268). E3's FP32 summary and E6's INT8 summary can therefore never share it, and the plan's
   check would refuse every trigger. `SAME_CODE_PATH_KEYS` (module :97-99) keeps the other seven of Q2's fields and
   adds `run.env.device`, `dataset.preprocess_protocol` and `metric_protocol`. `run.repo_commit` and
   `run.eval_runtime.image_digest` must be non-null (module :100).
2. **The five sha256 flags are format-checked first** (`sha256_format`, step 2), before any file is read.
   `resolve_parent` and `read_tracked_selection` keep their own format checks; with the flags checked first, this
   CLI's inputs no longer reach them.
3. **LB7 is enforced in code** (`image_not_of_record`; `summary_values` on each summary's image). The plan put the image
   in the procedure only.
4. **The launch gate adds `trigger_inconsistent` and `trigger_sha256_format`.** read_trigger_record recomputes the
   decision from the record's own `e3_val` and `e6_val` and refuses a record whose `drop_pp` or `triggered` does not
   follow. It checks the pin's format first, as `read_tracked_selection` does.
5. **The trigger record carries more than plan §3(ii) listed:** `e3_val_decimal`, `e6_val_decimal` and
   `threshold_decimal` (LB5); `lane`; `code_clean_at_head` and `code_files_sha256` (W1); and in `e6`, `block_b_summary`
   and `excluded_epochs`. Everything plan §3(ii) and lane 7(a) (LS:65) list is there.
6. **The launch gate's cases are B-1's.** Plan §4.2 put `gate_refuses_trigger_*` in B-2's runner smoke. The gate's code
   is B-1's (`read_trigger_record`, `trigger_binding`; plan §0.1), so its 19 cases are in smoke_e6kd_trigger. B-2's gate
   test calls them.
7. **The smoke has 101 checks, not §4.1's 48** (B-1.7).
8. **E6's no-model test is the recomputation.** With no qat_selection.json, the trigger recomputes the selection: Q2's
   `no_convertible_epoch` refusal becomes `e6_no_model`, and anything else is incomplete (`e6_selection_missing`, its
   message naming what the recomputation said; module :300-314). The plan read "qat_epoch_eval.json excludes every
   epoch". The recomputation applies item 1(a) to every checkpoint, as AM-21 item 2(a) asks.
9. **decide re-checks its own code (W1, DL-24 F1).** Plan §1.2 had decide repeat steps 1-6 and require the same
   git_head and host. It now also requires the same code files and clean state as score (module :471-478), and the
   record carries decide's values (module :576-577).
10. **More commits than §7's C1-C4:** C2b, between C2 and C3, and W1-W5 after C3 (B-1.1).
11. **§3(xii) is restated** (LB1, B-1.3): D no later than 2026-11-02; the fallback freeze 2026-11-03.

### B-1.10 The mutations and their killing cases

Every edit was KILLED by its named case at the final run, with Q2's harness self-checks passing (`RESULT: PASS (102/102)`). Each edit is made in a shadow copy, never in the repository, and its case must have PASSED on the unmutated shadow with the same argv (Q2's harness).

| ID | Mutation (mutations :45-120) | Killed by (final run) |
|---|---|---|
| MT1 | the threshold compared with >= (strict > undone) | d1_drop_1_00_not_triggered |
| MT2 | the drop taken in float, then made decimal | d1_decimal_not_float |
| MT3 | the drop taken as E6 - E3 | d1_drop_1_01_triggered |
| MT4 | the threshold read as a fraction (0.01) against the drop in pp | d1_drop_0_50_not_triggered |
| MT5 | E3's value taken from best.json | d1_e3_value_is_the_evaluator_score_not_best_json |
| MT6 | E6's re-score no longer compared with block B's value | refuses_e6_rescore_differs |
| MT7 | the summaries' split unchecked | refuses_e6_summary_not_val |
| MT8 | the thread setting dropped from the same-code-path keys | refuses_summaries_not_same_code_path |
| MT9 | E6's stored selection not compared with its recomputation | refuses_e6_selection_recompute_differs |
| MT10 | the α winner's ckpt_sha256 checked for presence only | refuses_e3_checkpoint_not_alpha_winner_sha |
| MT11 | E6 with no model read as a missing selection (incomplete) | refuses_e6_no_model |
| MT12 | a wall-clock time written into the trigger record | decide_twice_identical_bytes |
| MT13 | the evaluator commands run at batch size 16 | score_runs_the_literal_evaluator_commands |
| MT14 | the SL-1 name check made after a probe of the path (is_dir) | g1_forbidden_names_refused_before_any_probe |
| MT15 | decide no longer compares its code files with score's | refuses_code_changed_since_score |
| MT16 | decide's RESULT line no longer marks smoke inputs | d1_result_line_marks_smoke_inputs |
| MT17 | the summaries' governed_paths_clean unchecked | refuses_summary_governed_paths_dirty |
| MT18 | a summary scored at another commit than trigger_eval.json's accepted | refuses_summaries_scored_at_another_commit |
| MT19 | the sum of per_class.gt_support unchecked | refuses_summary_gt_support_sum |

Against plan §4.6's MT table: MT1-MT7 and MT9-MT12 are the plan's. MT4 divides the threshold by 100. MT8 is narrower than the plan's: it drops one same-code-path key (the thread setting), because the refusal row `summaries_not_same_code_path#0` already removes the whole check. MT13 runs both commands at batch 16, since one function builds both. MT14's case is renamed (LB2). MT15-MT19 come from DL-24 (W1-W3). The refusal rows are B-1.5's table, each KILLED by the cases named there.

### B-1.11 The DL-24 verification workflow

One read-only verification workflow, `wf_f01914b8-364`, ran at C3 (578cd02), before W1. It had eight agents in two
phases, each verifier told to refute its reviewer's findings:
- **Review:** `review:decision` (the decision and the two scores), `review:inputs` (steps 1-8), `review:records` (the
  two records and the launch gate), `review:smokes` (the smoke and the mutations). Each read the code at HEAD against
  the rules and reported findings with a severity.
- **Verify:** `verify:decision`, `verify:inputs`, `verify:records` and `verify:smokes`, one adversarial verifier per
  reviewer.
- Totals: 2,109,875 tokens and 341 tool calls (MEASURED, the workflow's result).

Before and after the workflow (MEASURED; the two records are equal): HEAD 578cd02; sha256 module eec9de21…, CLI
9040ca0e…, smoke 13bffa88…, mutations a01fae41…; scoped status empty. The workflow wrote nothing.

Ten findings, nine distinct (F1 was reported by all four reviewers); one refuted.

| Finding | Severity | Verdict | Resolution |
|---|---|---|---|
| **F1** (D-F1, I-F1, R-F1, S-F1): decide never checked its own code; the record copied score's `code_clean_at_head` and `code_files_sha256` | minor | confirmed ×4 | **W1.** decide compares `git_head`, `code_files_sha256`, `code_clean_at_head`, the host and the inputs with trigger_eval.json (module :471-478), and records its own values (:576-577). Case refuses_code_changed_since_score (smoke :876); MT15 KILLED; the `trigger_eval_mismatch` row names both its cases |
| **D-F2:** a re-score that differs from block B is caught only at decide, and the trigger eval dir is not spent | note | **refuted** | No change. The verifier found that LB6 names the outcome "the refusal e6_rescore_differs", and plan §1.2 and §3(i) put the equality in decide. decide refuses every time, from the sha-pinned summary, and score cannot reuse the non-empty dir. Spending the dir would not link a second attempt in a new dir; LB6 forbids that attempt procedurally, and the refusal's message says so |
| **R-F2:** a NOT TRIGGERED record made with `--allow-smoke-inputs` is never refused, and decide's RESULT line did not show the flag | minor | confirmed | **W2.** The RESULT line reads "(smoke inputs, not a record)" (module :583-590). Case d1_result_line_marks_smoke_inputs (smoke :963); MT16 KILLED. A TRIGGERED smoke record is refused at launch (`trigger_smoke_inputs`) |
| **R-F3:** `input_sha256s.e6_clip_selection` is copied from E6's run_meta, not hashed by the trigger | note | confirmed | No change; recorded (B-1.14, open item 2). 16 of the 17 input sha256 values are hashed from bytes in the session. The clip selection's is the value E6's run_meta holds. It is bound through the pinned telemetry sha256, since run_meta is the telemetry's first row, and E6's own launch gate compared it with the file (src/quant/qat.py:573) |
| **R-F4:** the records hold absolute container paths, so a determinism re-run must use score's mounts | note | confirmed | No change. The operator block reuses score's mounts and says so (B-1.12) |
| **S-F2:** several conditions of the two summary gates had no case of their own, so dropping one went unseen | minor | confirmed | **W3.** 16 cases (B-1.7); MT17-MT19 KILLED; both refusal rows name every case |
| **S-F3:** Q2's harness self-check measures Q2's `main()`, not the lane's | note | confirmed | No change; recorded (B-1.14, open item 1). The lane's `main()` follows Q2's order (mutations :208-253) |
| **S-F4:** C3's message says the refusal table has 52 sites; it has 53 | note | confirmed | No change: no history is rewritten. Recorded (B-1.14, slip 3) |
| **S-F5:** one mutation job made a forbidden-named folder in its own temp dir | note | confirmed | **W4.** g1's inputs also fail a later gate. Measured: with a clean-named stand-in for the removed check, the inputs end at `e6_selection_sha256_mismatch` and create nothing; g1, MT14 and the `forbidden_path` row are KILLED (B-1.14, slip 4) |

Refuted: D-F2. Every confirmed finding is either fixed (W1-W4, each with its case and a killed mutation) or recorded
above with its reason.

### B-1.12 The operator block

Text for the orchestrator. The laptop runs the trigger after E6 seed 42's block B has written its qat_selection.json and
that file's sha256 is in block B's DL row. It runs at the pin that holds B-1's merge: `score`, then `decide`, in the
student image by digest (LB7), on the CPU at batch 1 (the literal commands, B-1.4), with `--network none`. Every input
is a read-only `--mount`. The one writable `--mount` is the trigger eval dir, new and empty. No line uses `-v`. The
lines follow block B's conventions (docs/lane_reports/q2-qat.md:1430-1488: §15's setup, how its lines work, and its
Command 0): the `RO` and `RW` helpers, the environment, the STOP fields, the check line and container names that
cannot run twice.

Setup (PowerShell). `$CHECK` is block B's check line, verbatim (q2-qat.md:1460).
```powershell
$IMG  = "ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
$PIN  = "<PIN>"            # the master commit that holds B-1's merge
$REPO = "<CHECKOUT>"       # a clean checkout at $PIN (it holds reports/derived/lambda_selection.json and alpha_selection.json)
$DS   = "<TRAINVAL_DS>"    # block B's TRAIN/VAL staged data root (no TEST surface)
$E3   = "<E3_RUN>"         # the folder with best.json, e3_run_meta.jsonl, e3_telemetry.jsonl and
                           #   e3_student_best_iter76000.pt of e3_s42_lambda0.5_alpha25_80k_a1 (DL-120's first copy)
$E6R  = "<E6_RUN>"         # E6 seed 42's run directory as downloaded (qat_telemetry.jsonl, epoch_ckpts\)
$E6E  = "<E6_EVAL>"        # E6 seed 42's block B eval directory (it holds qat_selection.json)
$TRG  = "<TRIGGER_EVAL>"   # a new directory outside $REPO: the trigger eval dir
$ATT  = "a1"               # a new label with every new $TRG
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
$IN   = (RO $E3 "/e3/e3_s42_lambda0.5_alpha25_80k_a1") + (RO $E6R "/runs/e6_s42") + (RO $E6E "/evals/e6_s42")
$OUT  = RW $TRG "/trig"
$STATE = "ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}}"
$CHECK = "import os,subprocess,sys;a=sys.argv[1:];g=lambda *x:subprocess.run(['git','-C','/work/repo']+list(x),capture_output=True,text=True);h=g('rev-parse','HEAD').stdout.strip();q=g('status','--porcelain=v1','--untracked-files=all','--','src','configs','scripts','requirements*','docs/EVALUATION_CONTRACT.md','docs/IMPLEMENTATION_CONTRACT.md');d=len(q.stdout.splitlines()) if q.returncode==0 else 'git failed';r=[('HEAD',h,os.environ.get('EXPECT_P')),('GOVERNED_DIRTY',d,0)];c=lambda p:sum(1 for e in os.scandir(p) if e.is_file()) if os.path.isdir(p) else 'missing';s=[x.rsplit('=',1) if '=' in x else [x,''] for x in a];r+=[(p,c(p),int(n)) if n else (p,os.path.exists(p),True) for p,n in s];[print(k,v,'OK' if v==w else 'STOP') for k,v,w in r];sys.exit(0 if all(v==w for k,v,w in r) else 1)"
$ID = @("--e3-run-dir","/e3/e3_s42_lambda0.5_alpha25_80k_a1",
        "--e3-checkpoint-sha256","4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d",
        "--e6-run-dir","/runs/e6_s42","--e6-eval-dir","/evals/e6_s42",
        "--e6-telemetry-sha256","<TEL6>","--e6-selection-sha256","<SEL6>",
        "--lambda-selection","reports/derived/lambda_selection.json",
        "--lambda-selection-sha256","719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145",
        "--alpha-selection","reports/derived/alpha_selection.json",
        "--alpha-selection-sha256","1399b897278db17a8876c724b89c8d8e7a7b42ffe1d9c4d67002578a889f2dd8",
        "--trigger-eval-dir","/trig","--host-label","<HOST>")
```

Every flag and where each value comes from:
- `--e3-run-dir /e3/e3_s42_lambda0.5_alpha25_80k_a1`: the mount's last component is binding. The parent's run id is
  its directory's name (src/quant/qat.py:552), and it must be the α winner's run id (`e6_parent_binding`).
- `--e3-checkpoint-sha256 4baa5bad…`: `winner.ckpt_sha256` of reports/derived/alpha_selection.json, read at O in this
  session (MEASURED), and DL-121's. The trigger refuses any other checkpoint (`e3_checkpoint_not_alpha_winner`).
- `--e6-run-dir`, `--e6-eval-dir`: any mount names. The selection comparison leaves out `run_dir` and `eval_dir`
  (src/quant/qat_select.py:50, :93-97), E6's run id is read from its run_meta (src/quant/qat_artifacts.py:144), and the
  provenance names its artifact by a bare file name (qat_artifacts.py:72, :524; resolved beside the provenance file,
  src/eval/stage_artifacts.py:155-157).
- `--e6-telemetry-sha256 <TEL6>`: the sha256 of E6 seed 42's qat_telemetry.jsonl as printed on the pod at the run's end
  and recorded in E6 seed 42's DL row. The laptop copy must hash to it (`e6_telemetry_sha256_mismatch`).
- `--e6-selection-sha256 <SEL6>`: the sha256 of E6 seed 42's qat_selection.json, printed when block B wrote it and
  recorded in block B's DL row (`e6_selection_sha256_mismatch`).
- `--lambda-selection` and `--alpha-selection`: repo-relative, tracked and unchanged at HEAD
  (`read_tracked_selection`).
- `--lambda-selection-sha256 719eaa0a…`: DL-109's. `--alpha-selection-sha256 1399b897…`: DL-121's. Both were read at
  O in this session (MEASURED).
- `--trigger-eval-dir /trig`: the writable mount. score needs it new or empty (`trigger_eval_dir_not_fresh`, module
  :235-237); `RW` creates it empty, and the check line counts its files (`/trig=0`).
- `--host-label <HOST>`: the laptop's label. decide requires the same label as score (`trigger_eval_mismatch`).
- `--out /trig/e6kd_trigger.json` (decide only): new; written once.
- `PLANTSEG_IMAGE_DIGEST`: the student image of record (src/quant/qat.py:105). The trigger refuses another
  (`image_not_of_record`), and each summary records it.
- `--allow-smoke-inputs` is never passed.

The lines, in order. Each must end with `ExitCode=0 OOMKilled=false …`; any other line is a STOP.
```powershell
$NM = "am3_${ATT}_git";    docker run --name $NM @ENVV $IMG git --version; docker inspect -f $STATE $NM
$NM = "am3_${ATT}_check";  docker run --name $NM @ENVV @DATA @IN @OUT $IMG python -B -c $CHECK /data/images/train=5367 /data/images/val=846 /data/annotations/train /data/annotations/val /runs/e6_s42/qat_telemetry.jsonl /runs/e6_s42/epoch_ckpts=15 /e3/e3_s42_lambda0.5_alpha25_80k_a1/best.json /evals/e6_s42/qat_selection.json /trig=0; docker inspect -f $STATE $NM
$NM = "am3_${ATT}_score";  docker run --name $NM @ENVV @DATA @IN @OUT $IMG python -B scripts/e6kd_trigger.py score @ID; docker inspect -f $STATE $NM
$NM = "am3_${ATT}_check2"; docker run --name $NM @ENVV @DATA @IN @OUT $IMG python -B -c $CHECK /trig/trigger_eval.json /trig/scores/e3/summary.json /trig/scores/e6/summary.json; docker inspect -f $STATE $NM
$NM = "am3_${ATT}_decide"; docker run --name $NM @ENVV @DATA @IN @OUT $IMG python -B scripts/e6kd_trigger.py decide @ID --out /trig/e6kd_trigger.json; docker inspect -f $STATE $NM
Get-FileHash -Algorithm SHA256 "$TRG\e6kd_trigger.json", "$TRG\trigger_eval.json"
```
Expected:
- `git`: one git version line. An empty HEAD or `GOVERNED_DIRTY git failed STOP` on the next line means git refused
  the mount: STOP, as in block B (DL-62).
- `check`: `HEAD <PIN> OK`, `GOVERNED_DIRTY 0 OK`, then each path with `5367`, `846`, `True`, `True`, `True`, `15`,
  `True`, `True` and `0`, each ending in `OK`.
- `score`: `[score] E3: all-class mIoU …`, `[score] E6: …`, `[score] /trig`, then `RESULT: SCORED 2/2 (record)`.
  INFERRED time: about 15-40 min for the two VAL scorings on the CPU (plan §3(x)).
- `check2`: `HEAD <PIN> OK`, `GOVERNED_DIRTY 0 OK` and three `True … OK` lines.
- `decide`: five rule-trace lines, `[decide] /trig/e6kd_trigger.json`, then
  `RESULT: TRIGGERED -- drop D pp > 1.0 pp (E3 X, E6 Y; seed 42)` or
  `RESULT: NOT TRIGGERED -- drop D pp <= 1.0 pp (E3 X, E6 Y; seed 42)`. A line with `smoke inputs` is never the
  record.

Other outcomes:
- `RESULT: REFUSED [code]` (exit 2) wrote nothing: a STOP, reported with its line.
- `RESULT: REFUSED [e6_rescore_differs]` is a STOP for diagnosis and a ruling, never a choice between versions (LB6;
  AM-21 item 2(c)). No second scoring is made without that ruling.
- `RESULT: REFUSED [e6_no_model]`: AM-21 item 2(b)'s entry applies, and the trigger waits for the new amendment.
- `RESULT: INCOMPLETE [evaluator_failed]` (exit 3) spends `$TRG`, which then holds trigger_STOP.json. A new scoring
  uses a new `$TRG` and a new `$ATT`.
- decide may be repeated with the same mounts and another new `--out`; the bytes are identical
  (decide_twice_identical_bytes). The records hold the container paths (DL-24 R-F4), so moving any mount between score
  and decide is refused (`trigger_eval_mismatch` or `evaluator_command_not_literal`).

Then, in the checkout:
1. Copy `$TRG\e6kd_trigger.json` byte for byte to `reports/derived/e6kd_trigger.json`.
2. Check its sha256 against the `Get-FileHash` line.
3. Commit it alone and push.
4. Enter the DL row (B-1.13, row 2).

All four are done by the end of the second day after the day E6 seed 42 is complete (AM-22 v6 item 1(a)2; LB1). A
later entry is a deviation, reported in Chapter 4.

B-2's launch gate then reads that committed file: `--trigger reports/derived/e6kd_trigger.json` with
`--trigger-sha256` from row 2. E6-KD's pod checks out the commit that holds it, or a later master (LB8).

### B-1.13 Records proposed (the lane writes none)

These are plan §6's rows 1 and 2 and DL-07's status, as amended. LB1 puts the dates in the rows, never in the trigger
JSON. LB7 names the image. Plan §6's other rows and status texts are B-2's or later, and are unchanged.

**Row 1: B-1 implemented and merged.**
> L-AM3 B-1 (the AM-3 trigger of E6-KD) IMPLEMENTED 7cfcb48 (lane/am3-e6kd; report ⟨C4⟩; merged ⟨merge⟩, pin
> ⟨pin⟩): src/quant/e6kd_trigger.py (score, decide, read_trigger_record, trigger_binding) and
> scripts/e6kd_trigger.py. Smokes at 7cfcb48:
> - scripts/smoke_e6kd_trigger.py PASS (101/101);
> - scripts/smoke_e6kd_mutations.py PASS (102/102): MT1-MT19 and the 53 refusal rows, each KILLED by its named case,
>   with Q2's harness self-check passing;
> - the six Q2-era smokes unchanged.
>
> One DL-24 verification workflow (four reviewers, four adversarial verifiers); its confirmed findings were fixed in
> W1-W4 (docs/lane_reports/am3-e6kd.md §B-1, cloud lane session of 2026-10-16). The trigger runs on the laptop, by
> §B-1.12's operator block, after E6 seed 42's block B: in the student image b80b645d, on the CPU at batch 1 (LB7). It
> is scored, decided, committed alone with its DL row and entered by the end of the second day after the day E6 seed
> 42 is complete (AM-22 v6 item 1(a)2; LB1).

**Row 2: the trigger of record.** It is entered by the end of the second day after the day E6 seed 42 is complete
(AM-22 v6 item 1(a)2; LB1).
> AM-3 trigger evaluated: reports/derived/e6kd_trigger.json, sha256 ⟨S⟩ (⟨bytes⟩ B), committed alone at ⟨commit⟩.
> - e3_val ⟨X⟩: E3 seed 42, e3_s42_lambda0.5_alpha25_80k_a1, checkpoint 4baa5bad…, scored FP32 here.
> - e6_val ⟨Y⟩: E6 seed 42, epoch ⟨NN⟩, the converted QNNPACK artifact ⟨sha⟩, re-scored here and equal to block B's
>   value.
> - drop ⟨D⟩ pp (drop_pp, decimal on the recorded values) against the threshold 1.0 pp, strictly greater: triggered
>   ⟨true | false⟩.
> - Inputs: the file's input_sha256s: λ 719eaa0a…, α 1399b897…, E6 telemetry ⟨TEL6⟩, E6 qat_selection.json ⟨SEL6⟩,
>   the rules 76addc57….
> - Scored ⟨date, time +08:00⟩ on ⟨HOST⟩ (the laptop), in the student image b80b645d, at pin ⟨PIN⟩;
>   trigger_eval.json sha256 ⟨E⟩.
> - E6 seed 42 complete ⟨date⟩ (its epoch-15 epoch_end). This entry ⟨date⟩: ⟨within the second day after | late: a
>   deviation, reported in Chapter 4, changing nothing else⟩.
> - Consequence: ⟨E6-KD runs at seed 42 only (AM-3; DL-39, DL-51), and B-2 launches it only from this committed file
>   | E6-KD does not run⟩.
> - AM-22 item 1(a) condition 2: ⟨holds once this row is entered (not triggered) | holds once E6-KD is settled
>   (triggered)⟩.

If E6 seed 42 is entered as "non-finite: no model" (AM-21 item 2(b)), that entry replaces row 2. The trigger is then
refused (`e6_no_model`) and waits for the new amendment that item 2(b) requires.

**DL-07's status**, appended to "RECORDED 8084671 (AM-3)":
> ; L-AM3 B-1 trigger code IMPLEMENTED 7cfcb48 (lane/am3-e6kd; merged ⟨merge⟩, pin ⟨pin⟩): src/quant/e6kd_trigger.py,
> scripts/e6kd_trigger.py; the trigger of record: DL-⟨n⟩.

### B-1.14 The SL-1 log, refused commands, slips, tooling and open items

**The SL-1 log** (`tail -n 80 ~/.claude/sl1_guard.log`, run alone after the final runs):

The log holds three lines, all written during this session's STEP 7 (MEASURED; the file was absent at about 15:15Z):
```
2026-10-16T17:00:45Z	Bash	SL1-B1	S=/tmp/claude-0/-home-user-plantseg-thesis/566a2ec2-050a-5220-9ff2-5f75f02d3ff6/scratchpad; head -n 25 $S/LANE_STATE.md; echo "=== dev logs"; for f in run1 run2 run3 run4 mut1 mut2 mut3 mut4 mut5 mut6
2026-10-16T17:01:01Z	Bash	SL1-B1	head -n 25 /tmp/claude-0/-home-user-plantseg-thesis/566a2ec2-050a-5220-9ff2-5f75f02d3ff6/scratchpad/LANE_STATE.md; echo "=== dev logs"; cd /tmp/claude-0/-home-user-plantseg-thesis/566a2ec2-050a-5220-9
2026-10-16T17:19:57Z	Bash	SL1-B1	cd /tmp/claude-0/-home-user-plantseg-thesis/566a2ec2-050a-5220-9ff2-5f75f02d3ff6/scratchpad/dev && for f in mut1 mut2 mut3 mut4 mut5 mut6; do echo "$f: $(grep -c -E 'check_inputs:forbidden_path#0 +KIL
```
- **Class:** all three are `SL1-B1`, "Recursive, globbed or directory search or listing, or a bare listing, in a
  protected location" (.claude/hooks/sl1_guard.py:27; the fix it names: "pipe through grep -vi test, or name the
  file", :102). In log mode nothing was blocked (DL-79).
- **The cause, mine (slip 9):** each command read the session's own development logs in the scratchpad (run1-run4.log,
  mut1-mut6.log; none holds the token) through a shell loop's variable (`$f.log`), not by literal names. The first
  also ended in `ls $S`, which L-PROT refused, so it never ran (refusal 3). The second and third ran.

**LB4: Phase A's listing and its refused loop** (DOCUMENTED, plan Appendix A):
- Phase A ran one listing, `git ls-files -- src/distill src/training src/eval src/quant ':(exclude,icase)*test*'`
  ("Searches"). It was read-only and used to choose files to read. The form names directories and lacks this GO's
  exact exclude, so B-1 did not repeat it.
- Phase A's one refused command was a loop:
  `for p in src/quant/qat.py … ; do git show "$Q:$p" | wc -lc; done`. L-PROT G refused it: "git show $Q:$p shows
  docs/reference/reference.pdf or the tree that lists it." It was not retried; the counts were taken with literal
  paths, which is a different command.

**Refused commands in B-1** (each refused by the PreToolUse hook, not retried in the refused form):
1. STEP 1: `git cat-file -p 2b3b556…`. Refused by L-PROT G as a bare object id. The parents were read with
   `git rev-parse` instead.
2. About 15:15Z, during STEP 5: the GO's `tail -n 80 ~/.claude/sl1_guard.log` with an extra
   `ls -l ~/.claude/sl1_guard.log` appended. Refused by
   L-PROT S: "`ls ~/.claude/sl1_guard.log` lists docs/reference/reference.pdf". The GO's own tail then ran alone; the
   file was absent.
3. Before STEP 7: a compound read of scratchpad files ending in `ls $S`. Refused by L-PROT S: "`ls $S` lists
   docs/reference/reference.pdf". The reads ran again with literal paths and no `ls`.
4. Before STEP 7: a `python3` heredoc that edited the session's lane-state file and quoted refusal 3's message, which
   names the protected PDF. Refused by L-PROT N: "the command names docs/reference/reference.pdf; only the exact status
   probe may name it." The edit was made with the Edit tool instead.

**My slips:**
1. Refusals 1-4 above: each was a command outside the GO's forms. 1 used `cat-file`, 2 and 3 used `ls`, and 4 named
   the PDF inside a shell command.
2. **C2 (fb9064c, pushed) named temp dirs after check names** (`o_<check>`, `a_<check>`), against LB2. C2b replaced
   them with index names. C2 stays in the pushed history; nothing is rewritten.
3. **C3's message says "52 sites"; the refusal table has 53** (DL-24 S-F4). It is not amended.
4. **A development run created a forbidden-named folder.** The development mutation run on C3's files (mut1) ran the
   job that removes the SL-1 name check (`check_inputs:forbidden_path#0`). In that job, g1's `trigger_eval_dir`
   sub-case passed every later gate, so score created `nowhere_latest_dir` inside that job's own temp dir under /tmp.
   The smoke's end removed it, and the job was still KILLED. This went against the acceptance line "No run creates a
   path whose name contains 'test'". DL-24 found it (S-F5) and W4 fixed it. Of the later development runs, only mut6
   (after W4) ran that job, and it created nothing.
5. **C2's case refuses_e6_selections_mismatch failed when run alone**: it depended on an earlier case having built
   the temp repo. Found by the development mutation run's baseline; fixed in C2b.
6. **Stale text:** the mutation smoke said "MT1-MT14" after W1 and W3 had added MT15-MT19. W5 fixed it before the
   final runs.
7. **The first launch of the final runs** called the scratchpad runner without `bash`; the script is not executable,
   and the launch failed with "Permission denied" before any smoke ran. It was relaunched with `bash` (B-1.2's times
   are the relaunch's).
8. One `git grep` had `-A` after the pattern, so git read it as a revision and failed. It was rerun in the right order;
   no hook was involved.
9. **Three SL1-B1 log lines** (above): shell loops read my development logs through a variable name instead of
   naming each file.

**Tooling conflicts** (each reported once, not followed; AGENTS.md "What counts as approval" and the GO):
- the harness's designated branch `claude/vigilant-sagan-39fdg2` and its commit-and-push instructions: only
  `lane/am3-e6kd` is used;
- the harness's attribution reminder asking for `Co-Authored-By` and `Claude-Session` trailers. It came again during
  this session. No commit carries a trailer;
- a stop hook during STEP 7 (`~/.claude/stop-hook-git-check.sh`): "There are 7 unpushed commit(s) on branch
  'lane/am3-e6kd'. Please push these changes to the remote repository." Not followed. The GO's second push is STEP
  9's, after this report commit.

**Open items** (at most five):
1. **DL-24 S-F3.** Q2's harness self-check runs Q2's `main()` in its own processes, so the lane's `main()`, which
   mirrors it, has no self-check of its own. Accept, or ask B-2 to add one?
2. **DL-24 R-F3.** The trigger takes E6's clip selection sha256 from E6's run_meta. It is bound by the telemetry
   pin, and E6's launch gate compared it with the file. The trigger does not hash clip_selection.json itself. Accept,
   or add a `--e6-clip-selection` flag in B-2?
3. **D-F2 (refuted as a defect).** After `e6_rescore_differs`, no code stops a second scoring in a new trigger eval
   dir; only LB6 and the operator block forbid it. A ruling may want each scoring attempt in the DL row.
4. **C2 in the pushed history** (slip 2) and the development run's transient folder (slip 4): the merge carries C2's
   superseded smoke. The acceptance "no run creates" holds for every final run.
5. **The merge and the pin are Ice's block.** Under LB1, B-1 must be merged and pinned before E6 seed 42's block B
   ends.
