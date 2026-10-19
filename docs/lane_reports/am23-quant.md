# Lane report — L-AM23Q · AM-23's quantization code (`lane/am23-quant`)

## B-1 · the E6-X condition tool (AM-23 item 3(b))

GO: the orchestrator's B-1 GO of Sat 17 Oct 2026 (Manila), STEPS 1–9 under SCOPE, with rulings AQ1–AQ10 (the Phase A
plan, sha256 c49f4169…, as amended by them). Cloud lane (DL-37): no dataset, no checkpoint, no GPU; nothing here ran
on real data. Labels: MEASURED (run or read in this session), REPOSITORY-PROVEN (read at the commit named),
DOCUMENTED (the GO or the plan), INFERRED. File:line citations are at `b38dce30e720853e4a2cd5f48ff8c704942808c2`
(the last code commit; this report's commit adds only this file), unless another commit is named.

### B-1.1 Base, commits, diff and the venv

- **O** = `d32f501a289484991e6cdb005142d9af9a44ce78`, origin/master at STEP 1 (it holds none of tonight's merges;
  under src, scripts and configs it equals 2b3b556). `lane/am23-quant` starts at O: C1's parent is d32f501 (AQ1;
  MEASURED).
- **Commits**, each by explicit paths (`git add -- <new paths>`, `git commit --only -F <message> -- <paths>`), no
  trailer:

| commit | step | content |
|---|---|---|
| 524e778 | C1 | the module and the CLI |
| d94815d | C2 | the smoke; the first push |
| c6b122f | C3 | the mutation harness (the commit the DL-24 workflow reviewed) |
| 2d02ff5 | F1 | the smokes' coverage (DL-24 SMK-1, SMK-3, SMK-4, SMK-7, SMK-8) |
| 19fe67b | F2 | the deadline in whole seconds (NUM-1, NUM-3) |
| a55e859 | F3 | G10 takes only the last attempt of a grid point (PROV-5) |
| c7e56de | F4 | a failed write is STOP [write_failed] (GATE-1) |
| 30922f6 | F5 | print-dl reads the committed copy through read_condition (GATE-2) |
| 21bea2a | F6 | parse_record checks every field the text and B-2 use (GATE-3, SMK-5) |
| b38dce3 | F7 | G5 requires AM-23 committed at HEAD (PROV-1, the in-module half) |
| this commit | C4 | this report |

- **Scoped diff**, `git diff --stat d32f501 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference'
  ':(exclude,icase)*test*'` at b38dce3 (MEASURED); this commit adds docs/lane_reports/am23-quant.md:

```
 scripts/e6x_condition.py                 |   98 ++
 scripts/smoke_e6x_condition.py           | 1544 ++++++++++++++++++++++++++++++
 scripts/smoke_e6x_condition_mutations.py |  457 +++++++++
 src/quant/e6x_condition.py               |  739 ++++++++++++++
 4 files changed, 2838 insertions(+)
```

- **The venv** (`$SCR/am23venv`, Python 3.11.17) against requirements.lock (sha256 ebbf6782…; MEASURED): the 23 PyPI
  packages installed are each at the lock's pin; torch 2.1.0+cpu and torchvision 0.16.0+cpu are the lock's 2.1.0 and
  0.16.0 as CPU builds (the lock pins +cu121; this lane has no GPU); pip 24.0 and setuptools 79.0.1 are the venv's
  bootstrap, not in the lock; the lock's other 51 packages (the teacher stack: mmcv, mmsegmentation and their
  dependencies) are not installed, the student side only. `pip check`: "No broken requirements found." Quantized
  engines: qnnpack, none, onednn, x86, fbgemm.

### B-1.2 RESULT lines, and every changed count with its cases

| smoke (each run alone) | STEP 4 at O | STEP 7 at b38dce3 |
|---|---|---|
| smoke_frozen_blobs | `RESULT: FROZEN BLOBS OK (10/10)` | `RESULT: FROZEN BLOBS OK (10/10)` |
| smoke_qat_runner | `RESULT: PASS (214/214)` | `RESULT: PASS (214/214)` |
| smoke_arms_kd (`PYTHONPATH`, `--work-dir`) | `RESULT: PASS (61/61)` | `RESULT: PASS (61/61)` |
| smoke_e6x_condition | new | `RESULT: PASS (172/172)` |
| smoke_e6x_condition_mutations | new | `RESULT: PASS (428/428)`; `mutations killed: 141/141`; 170 (edit, case) pairs |

The three baseline counts are unchanged. `git diff --name-status d32f501 HEAD -- src scripts configs docs/lane_reports
':(exclude,icase)*test*'` at b38dce3 printed four `A` lines and nothing else; the scoped status was empty, also with
`--ignored` (MEASURED). Count history: smoke 133 (C2), 135 (C3), 146, 148, 149, 150, 153, 169, 172 (F1–F7); harness
107 edits and 326 checks (C3), then 115/350, 117/356, 119/362, 120/365, 122/371, 138/419, 141/428 (F1–F7).

**smoke_e6x_condition: the plan's 67 (§4.1) became 172.** The plan's c_ names became groups: w_ (written), r_
(refused), i_ (incomplete), s_ (the write's STOP), aq5_ (AQ5), d_ (the readers) and h_ (hygiene).

- Written, 12 → 18. The plan's twelve: w_yes_written, w_no_written, w_equal_is_no, w_one_ulp_above_is_yes,
  w_one_ulp_below_is_no, w_record_fields, w_deterministic, w_print_dl_matches (the plan's c_dl_text),
  w_deadline_before_manila_midnight, w_deadline_after_manila_midnight, w_r2_written, w_read_condition_accepts. Added:
  w_dl_text_one_line_aq3 (AQ3), w_dl_text_no_verdict, w_cli_subprocess_print_dl,
  w_deadline_fraction_before_manila_midnight (NUM-1), w_deadline_last_float_before_manila_midnight (NUM-3),
  w_print_dl_committed_copy (GATE-2).
- Reader refusals, 6 → 35. The plan's six: d_read_condition_sha256_mismatch, d_read_condition_not_tracked,
  d_read_condition_changed_since_head, d_condition_format_keys, d_condition_void_yes, d_condition_void_no. Added 29:
  print-dl's file reader d_condition_sha256_format, d_condition_missing_file, d_condition_sha256_mismatch,
  d_read_record_file_forbidden_path; read_condition's d_read_condition_sha256_format,
  d_read_condition_not_repo_relative, d_read_condition_forbidden_path, d_read_condition_missing_file; parse_record's
  d_condition_format_nan, d_condition_format_value, d_condition_format_threshold and, one per clause (F6),
  d_condition_format_format, _repr, _verdict, _arm, _stage, _parent, _seed, _run_id, _checkpoint, _sha256_best_json,
  _sha256_checkpoint, _commit, _deadline_utc, _deadline_manila, _deadline_day, _duplicate_key; print-dl on the
  committed copy (F5) d_print_dl_committed_other_record, d_print_dl_path_not_of_record.
- Tool refusals, 45 → r_ 106 (with s_ 2 and aq5_ 5 below). The plan's 45: r_missing_flags; r_forbidden_path_run_dir
  and r_forbidden_path_out (the plan's test_path, renamed by AQ6); r_out_relative, r_out_inside_repo, r_out_name,
  r_out_exists; r_expect_head_format, r_expect_head_mismatch, r_code_not_clean; r_arm_unknown;
  r_selection_sha256_mismatch (λ sha256), r_selection_not_tracked (α untracked); r_threshold_mismatch;
  r_teacher_record_missing, r_teacher_record_invalid; r_run_dir_inside_repo, r_run_dir_name; r_run_meta_rows; the 13
  r_run_meta_<field> cases; r_parent_aborted, r_parent_checks_failed, r_telemetry_torn; r_best_json_format_keys;
  r_checkpoint_missing; r_payload_stage, r_payload_arm, r_payload_best_val_miou_all_class; r_run_end_best;
  r_launch_log_launch, r_launch_log_stopped, r_launch_log_run_meta_sha256, r_launch_log_code_pin. Added 61, by gate:
  G0–G1 r_missing_flags_print_dl, r_no_command, r_forbidden_path_selection, r_forbidden_path_selection_alpha,
  r_forbidden_path_print_dl; G2 r_out_name_arm_form, r_out_exists_incomplete_run (SMK-1), r_out_parent_missing,
  r_out_created_before_write; G3 r_code_identity_unavailable; G4 r_arm_spec; G5 r_selection_not_of_record,
  r_selection_not_of_record_lambda (SMK-3), r_selection_changed_since_head, r_selection_format_winner,
  r_selection_format_winner_lambda (SMK-3), r_amendment_not_committed, r_amendment_uncommitted,
  r_amendment_without_threshold (PROV-1); G6 r_teacher_record_uncommitted, r_teacher_record_invalid_format, _arm,
  _sha256 (SMK-4); G7 r_run_dir_relative, r_run_records_missing, r_run_meta_rows_torn, r_run_meta_rows_event (SMK-4);
  G8 r_telemetry_format_line, r_telemetry_format_row, r_run_end_not_last, r_run_end_twice, r_run_end_format_iter,
  r_run_end_format_iter_float (SMK-4), r_run_end_format_clock; G9 r_best_json_format_parse (SMK-4),
  r_best_json_format_name, the seven AQ4 cases (r_value_invalid_nan_best_json, _infinity_run_end, _int_payload,
  _string_best_json, _above_one_best_json, _bool_run_end, _negative_payload), r_payload_unreadable,
  r_payload_not_dict, r_payload_target_construction, r_run_end_ckpt; G10 r_launch_log_missing, _uncommitted, _format,
  _launched, _attempt (PROV-5) and the nine other PL-14 fields (_git_head, _records_commit, _stage, _seed, _horizon,
  _sweep, _lambda_logit, _alpha_cwd, _launch_time_utc).
- The write and AQ5, 0 → 7: s_write_mismatch, s_write_failed (GATE-1); aq5_identity_clean_record,
  aq5_untracked_arm_file_not_clean, aq5_edited_arm_file_not_clean, aq5_edited_code_file_not_clean,
  aq5_tool_code_paths_cover.
- Incomplete, 1 → 1: i_run_end_missing.
- Hygiene, 3 → 5: h_no_forbidden_named_path_created (the plan's no test-named path, named per AQ6),
  h_repository_untouched, h_threshold_is_alpha_winner; added h_every_code_has_a_case, h_no_forbidden_path_opened.
- 18 + 35 + 106 + 7 + 1 + 5 = 172.

**smoke_e6x_condition_mutations: the plan's "about 32 edits, 3 checks each + 2" (§4.2) became 141 edits, 3 checks
each + 2 + 3 hygiene = 428.** Every edit the plan lists is present: `>` to `>=` (M001), the rounded compare (M002), the
verdict not derived (M003), the threshold not checked (M029), run_abort accepted (M061), a missing run_end read as
complete (M062), checks_passed ignored (M064), the 13 run_meta fields (M045–M057), payload best and run_end best
(M077, M078), the two wrong-file sha256 (M080, M102), `"xb"` to `"wb"` (M017), out inside the repo (M013), the
forbidden-name check after a file call (M141), no launched line (M082), the stopped line (M083), run_meta sha256
(M086), `--expect-head` not compared (M020), the reader's void, sha256 and tracked checks (M107, M138, M136), the DL
text without the record's sha256 (M103) and a clock time added (M101). The rest is one edit per refusing branch or
compared field (so every module code is the code of a killing case), the clause splits (F1, F6) and the fixes'
edits (F2–F7). The plan cites the copied `mutate` as smoke_teacher_diag_mutations.py ":55-60"; it is :56-81 there
(MEASURED), and the harness says so.

### B-1.3 Rulings AQ1–AQ10: where each is met

- **AQ1** (branch from origin/master): O = d32f501 = origin/master at STEP 1; C1's parent is O (B-1.1).
- **AQ2** (created exclusively at `--out`, outside the repository; Ice commits a byte-equal copy alone; the laptop in
  the student image with `--network none`, a pin at or after B-1's merge): `check_out`
  src/quant/e6x_condition.py:216-232 (absolute; E1's guard through `runner.check_output_dir` :221; the name; absent;
  its folder present), `write_record` :571-586 (`open(out, "xb")` :575, read back; a failure once the file exists
  is STOP [write_failed] :583). `RECORD_REL` :110 is the committed copy's path, which `dl_text` names and print-dl's
  committed mode checks (`read_committed_record` :730-739). The tool neither records nor enforces the image; the
  operator block (B-1.8) runs it there. Cases r_out_*, r_out_created_before_write, s_write_failed,
  w_print_dl_committed_copy, d_print_dl_path_not_of_record; edits M012–M017, M106, M139, M140.
- **AQ3** (one line, no `|`, no line break; the tokens B-2's row needs): `dl_text` :594-609 names
  "E6-<X> condition (AM-23 item 3(b)): yes" (or "no"), the value's repr, best.json's sha256, the checkpoint's sha256
  and the record's sha256; its guard :607 refuses `|`, a line break or non-ASCII; it runs before the write
  (`record_condition` :633-634). Case w_dl_text_one_line_aq3 (scripts/smoke_e6x_condition.py:518); edits M103,
  M104. The RECORDED-row gate is B-2's (AQ10).
- **AQ4** (value_invalid unless a float, finite, in [0, 1], for all three values): `value_error` :177-186;
  best.json's and run_end's values :440-444 and the payload's :455-457, inside G9 and before `verdict()` (:189-194,
  which checks again). One case per kind, smoke :980-1010: NaN, Infinity, int, string, above 1 (and a bool and a
  negative). The AQ4 edit M004 (value_error accepts every value) is killed by all seven; M005–M007 cover the type
  rule and each call site; the reader's own clause is M111 (d_condition_format_value).
- **AQ5** (TOOL_CODE_PATHS; code_provenance's whole output; clean required): `TOOL_CODE_PATHS` :126-130 (file by file
  in B-1.4); `tool_identity` :235-249 returns `{"code_paths": ..., **prov}` (:249) and refuses unless
  clean_at_commit is true (:246-247). Cases aq5_untracked_arm_file_not_clean, aq5_edited_arm_file_not_clean,
  aq5_edited_code_file_not_clean (smoke :1196-1211: the real `ptq.code_provenance` on a code repository built from
  this checkout's files) and aq5_tool_code_paths_cover (:1219); edit M021 is killed by r_code_not_clean and the three
  aq5_ cases.
- **AQ6** (no "test" in a new name): G1's code is `forbidden_path` (:211), the readers' `condition_forbidden_path`
  (:695, :715). The smoke's case registry refuses a case name containing "test"; h_no_forbidden_named_path_created
  (smoke :1447) and h_no_forbidden_path_opened (:1455) passed in both final runs. `qat.names_test`
  (src/quant/qat.py:153) and Q2's `selection_test_path` keep their names.
- **AQ7** (deterministic record; the deadline from run_end's time; stated in the docstring): the module docstring
  :22-27; `deadline` :519-530 takes run_end.wall_clock_end floored to whole seconds (:523) and gives the end of the
  Manila day after run_end's; `DEADLINE_RULE` :101-102. Cases w_deterministic (smoke :498), the four w_deadline_
  cases (:539, :550, :573, :574); edits M097–M101. The timing itself is open item 2.
- **AQ8** (the LQP boundary): B-1 adds four files and edits none (B-1.2's name-status); it calls
  `ptq.code_provenance` (:239) read-only. Recorded as a ruling in B-1.9.
- **AQ9** (the rest of the plan stands; the counts named): B-1.2, and the changes listed in B-1.4.
- **AQ10** (Q1, Q2, Q4–Q8 are B-2's): nothing of them is in B-1.

### B-1.4 The CLI, the refusal codes, TOOL_CODE_PATHS, the record and the changes from the plan

**The CLI**, scripts/e6x_condition.py (`build_parser` :39-48, `cmd_record` :51-58, `cmd_print_dl` :61-73, `main`
:76-94):

```
python -B scripts/e6x_condition.py record --arm R1|R2 --run-dir DIR \
    --lambda-selection reports/derived/lambda_selection.json --lambda-selection-sha256 HEX \
    --alpha-selection reports/derived/alpha_selection.json --alpha-selection-sha256 HEX \
    --expect-head COMMIT --out OUT/e6x_condition_<X>.json
python -B scripts/e6x_condition.py print-dl --record PATH --record-sha256 HEX
```

`record` prints `e6x_condition/1 written: <out>`, `sha256: <R>`, the line `decision-log text (one line; the
orchestrator enters it verbatim):` and the text, then `RESULT: E6-<X> CONDITION yes (<repr> > 0.36337557435035706)`
or `RESULT: E6-<X> CONDITION no (<repr> <= 0.36337557435035706)`. `print-dl` prints the same text from a record and
its sha256: a repo-relative PATH is the committed copy (read_condition: tracked and unchanged at HEAD, then the
sha256, at reports/arms/e6x_condition_<X>.json), an absolute PATH is read as a file (the record at `--out`); then
`RESULT: DL TEXT (E6-<X>, verdict <v>, record <R[:12]>; read from the committed copy at HEAD)` or `...; read from the
file at --record)`. Exits: 0 written or printed, 1 STOP, 2 refused, 3 E3-X incomplete, 4 error (nothing written).

**Every refusal code** (src/quant/e6x_condition.py unless named; exit 2 unless named):

| gate | code: line |
|---|---|
| G0 | missing_flags :202; no command: scripts/e6x_condition.py:79 |
| G1 | forbidden_path :211 |
| G2 | out_relative :219, out_inside_repo :223, out_name :227, out_exists :229, out_parent_missing :231 |
| G3 | expect_head_format :238, code_identity_unavailable :241, expect_head_mismatch :245, code_not_clean :247 |
| G4 | arm_spec :257, arm_unknown :260 |
| G5 | selection_not_of_record :269, selection_format :277, threshold_mismatch :280, amendment_not_committed :293; passed through from `qat.read_tracked_selection` (src/quant/qat.py:622): selection_sha256_format, selection_path_not_repo_relative, selection_test_path (Q2's name), selection_missing_file, selection_not_tracked, selection_changed_since_head, selection_sha256_mismatch, selection_format |
| G6 | teacher_record_missing :300 and teacher_record_uncommitted :301 (the names `sweep_select.HeadSource.read` refuses with), teacher_record_invalid :312 |
| G7 | run_dir_relative :339, run_dir_inside_repo :343, run_dir_name :346, run_records_missing :350, run_meta_rows :362, run_meta_mismatch :389 |
| G8 | telemetry_format :399, :403, telemetry_torn :401, parent_aborted :405, run_end_missing :409 (exit 3), run_end_not_last :412, parent_checks_failed :416, run_end_format :420 |
| G9 | best_json_format :432, :435, :438, value_invalid :444, :457, checkpoint_missing :447, payload_mismatch :452, :454, :463, run_end_mismatch :467 |
| G10 | launch_log :489 (launch:), :491 (launched:), :493 (stopped:), :496 (attempt:), :514 (each PL-14 field); passed through from `sweep_select.load_launch_log` (src/training/sweep_select.py:967): launch_log_missing, launch_log_uncommitted, launch_log_format |
| the write | out_exists :577; write_failed :583 (exit 1, STOP); write_mismatch :586 (exit 1, STOP) |
| parse_record | condition_format :660, :662, :681; condition_void :684 |
| read_record_file | condition_sha256_format :692, condition_forbidden_path :695, condition_missing_file :697, condition_sha256_mismatch :701 |
| read_condition | condition_sha256_format :709, condition_path_not_repo_relative :712, condition_forbidden_path :715, condition_missing_file :717, condition_not_tracked :721, condition_changed_since_head :723, condition_sha256_mismatch :726 |
| read_committed_record | condition_path_not_of_record :736 |

**TOOL_CODE_PATHS** (:126-130; AQ5), file by file: src/quant/e6x_condition.py (the module); scripts/e6x_condition.py
(the CLI); src/quant/qat.py (names_test, read_tracked_selection, sha256_bytes, _git_head_bytes); src/quant/runner.py
(check_output_dir); src/training/train_e1.py (_assert_outside_repo, which check_output_dir calls); src/quant/ptq.py
(code_provenance); src/training/sweep_select.py (HeadSource, load_launch_log, LaunchLog, _same, _finite_number,
MANILA, BEST_JSON_KEYS); src/training/kd_arms.py (load_arms); configs/kd_arms (the folder: git status names an
untracked arm file in it); configs/kd_arms/e3_flip.json, configs/kd_arms/e3_r1.json, configs/kd_arms/e3_r2.json (each
hashed). The records a gate reads (the selections, the teacher record, the launch log, the amendments file) are read
at HEAD of the records repository and bound by content, not listed here.

**e6x_condition/1** (`RECORD_KEYS` :131-133; `build_record` :533-560), 20 keys: `amendment` ("AM-23 item 3(b)"), `arm`
(`--arm`), `best_json` {file, sha256} (G9), `checkpoint` {file, sha256, bytes} (G9), `dl_deadline` {run_end_wall_clock_end,
run_end_utc, run_end_manila_date, enter_by_manila_date, enter_by_end_utc, timezone, rule} (AQ7), `format`
("e6x_condition/1"), `launch_log` {path, sha256, launch, launched} (G10), `parent_run_stage` ("E3-<X>"), `run_end`
{iter, checks_passed, best_val_miou_all_class, best_ckpt, wall_clock_end} (G8), `run_id` (the run directory's name),
`run_meta` {file, sha256} (G7), `seed` (42), `selections` {lambda, alpha: {path, sha256, winner}} (G5), `stage`
("E6-<X>"), `teacher_record` {path, sha256, checkpoint_name, checkpoint_sha256} (G6), `telemetry` {file, sha256} (G8),
`threshold` {repr, source, comparison}, `tool` {code_paths, commit, clean_at_commit, files_sha256} (G3), `value`
{best_val_miou_all_class, repr, source "best.json"} (G9), `verdict` ("yes" or "no", from `verdict()` alone). Written
as `json.dumps(indent=2, sort_keys=True, allow_nan=False)` plus a newline: no clock, host or local absolute path.

**Changes from the plan** (AQ9; each with its cases above):
1. Names (AQ6): test_path → forbidden_path; B-2's condition_test_path → condition_forbidden_path; case groups as in
   B-1.2.
2. Codes the plan did not list: out_parent_missing (G2); arm_spec (G4, the registry's own error); selection_not_of_record,
   selection_format and amendment_not_committed (G5); teacher_record_uncommitted (G6); run_records_missing (G7);
   telemetry_format, run_end_not_last and run_end_format (G8); value_invalid (AQ4); launch_log's attempt field (G10);
   write_failed; the readers' condition_* codes, with condition_path_not_of_record.
3. print-dl: a repo-relative `--record` reads through read_condition, as the plan says, and must be the arm's
   RECORD_REL; an absolute one (the record at `--out`, before it is committed) is read as a file (F5).
4. G8 reads the telemetry with its own reader on the bytes it hashed (qat.read_jsonl's reading), not through
   qat.read_jsonl.
5. G10 also requires the run to be the last attempt of its grid point (F3). PL-14's records-commit check
   (src/training/sweep_select.py:1820-1825) is not repeated (PROV-2, refuted; open item 3).
6. G5 also requires AM-23 committed at HEAD (F7; PROV-1).
7. The deadline is computed in whole seconds, floored (F2), as AM-19's date rule compares exact instants
   (docs/PREREGISTRATION_AMENDMENTS.md:1367-1368).
8. TOOL_CODE_PATHS adds src/training/train_e1.py to AQ5's minimum (check_output_dir calls its guard).
9. run_meta's selection_sha256 field is not compared: G5 binds the selections by path and sha256 at HEAD, and G7
   compares run_meta's λ and α values with their winners.
10. The smoke has no decision-log fixture (§4.1 s0 lists one): the decision-log row is B-2's gate's input.
11. The plan's forbidden-name-order edit (§4.2, "an audit-hook check") is killed by its code, not by the audit hook
    (open item 4).

### B-1.5 The mutations and their killing cases

The harness's final run (STEP 7, b38dce3), as printed. Kinds: accepted (the mutant wrote or accepted what the case
refuses), result (a written record or printed line is not the case's), code (another code or exit), error (the named
outcome degrades to exit 4 or an exception), text (the same code with another message or field). The three text
kills are M008 (with no command, print-dl's G0 refuses with the same code), M015 (also killed at the code level by
r_out_exists_incomplete_run) and M081 (a launched line never stands without its launch line, so the next check
refuses with the same code: the field differs). Totals: accepted 68, code 63, error 18, result 18, text 3.

| id | function | edit | killed by (kind) |
|---|---|---|---|
| M001 | verdict | the strict > made >= | w_equal_is_no (result) |
| M002 | verdict | the decimal comparison made one of floats rounded to six places | w_one_ulp_above_is_yes (result) |
| M003 | build_record | the record's verdict not derived from its value | w_no_written (result) |
| M004 | value_error | AQ4's check made to accept every value | r_value_invalid_nan_best_json (code), r_value_invalid_infinity_run_end (code), r_value_invalid_int_payload (code), r_value_invalid_string_best_json (code), r_value_invalid_above_one_best_json (code), r_value_invalid_bool_run_end (code), r_value_invalid_negative_payload (code) |
| M005 | value_error | AQ4's type rule reduced to 'not a string' | r_value_invalid_bool_run_end (code), r_value_invalid_int_payload (code) |
| M006 | best_checkpoint | best.json's and run_end's value checks removed | r_value_invalid_nan_best_json (code), r_value_invalid_infinity_run_end (code), r_value_invalid_string_best_json (code), r_value_invalid_above_one_best_json (code), r_value_invalid_bool_run_end (code) |
| M007 | best_checkpoint | the payload's value check removed | r_value_invalid_int_payload (code), r_value_invalid_negative_payload (code) |
| M008 | CLI.main | the command not required (print-dl's G0 then refuses with the same code) | r_no_command (text) |
| M009 | check_flags | G0 removed | r_missing_flags (error), r_missing_flags_print_dl (code) |
| M010 | check_names | G1 removed (SL-1, AQ6) | r_forbidden_path_run_dir (code), r_forbidden_path_out (code), r_forbidden_path_selection (code), r_forbidden_path_selection_alpha (code), r_forbidden_path_print_dl (code) |
| M011 | CLI.cmd_print_dl | print-dl's G1 call removed | r_forbidden_path_print_dl (code) |
| M012 | check_out | --out's absolute check removed | r_out_relative (accepted) |
| M013 | check_out | E1's outside-the-repository guard dropped for --out | r_out_inside_repo (code) |
| M014 | check_out | --out's name check removed | r_out_name (accepted), r_out_name_arm_form (code) |
| M015 | check_out | G2's existence check removed (the write's exclusive create still refuses, later) | r_out_exists_incomplete_run (code), r_out_exists (text) |
| M016 | check_out | --out's folder check removed | r_out_parent_missing (error) |
| M017 | write_record | the exclusive create made an overwrite | r_out_created_before_write (accepted) |
| M018 | tool_identity | --expect-head's form check removed | r_expect_head_format (code) |
| M019 | tool_identity | the missing-commit check removed | r_code_identity_unavailable (code) |
| M020 | tool_identity | --expect-head not compared | r_expect_head_mismatch (accepted) |
| M021 | tool_identity | cleanliness at the commit not required (AQ5) | r_code_not_clean (accepted), aq5_untracked_arm_file_not_clean (accepted), aq5_edited_arm_file_not_clean (accepted), aq5_edited_code_file_not_clean (accepted) |
| M022 | arm_spec | the registry's own error not caught | r_arm_spec (error) |
| M023 | arm_spec | the arm's count check removed | r_arm_unknown (error) |
| M024 | arm_spec | the arm teacher not matched | w_yes_written (result), w_r2_written (result) |
| M025 | selections | the α selection's path not checked | r_selection_not_of_record (accepted) |
| M026 | selections | the λ selection's path not checked | r_selection_not_of_record_lambda (accepted) |
| M027 | selections | the α winner's alpha not checked | r_selection_format_winner (code) |
| M028 | selections | the λ winner's lambda not checked | r_selection_format_winner_lambda (code) |
| M029 | selections | the threshold not checked against the α winner | r_threshold_mismatch (accepted) |
| M030 | amendment_on_record | AM-23's commit not checked | r_amendment_not_committed (accepted) |
| M031 | amendment_on_record | the amendments file read from the working tree, not at HEAD | r_amendment_uncommitted (accepted) |
| M032 | amendment_on_record | the AM-23 section not bounded by the next heading | r_amendment_without_threshold (accepted) |
| M033 | teacher_record | the teacher record read from the working tree, not at HEAD | r_teacher_record_uncommitted (accepted), r_teacher_record_missing (error) |
| M034 | teacher_record | the record's status not checked | r_teacher_record_invalid (accepted) |
| M035 | teacher_record | the record's format not checked | r_teacher_record_invalid_format (accepted) |
| M036 | teacher_record | the record's arm not checked | r_teacher_record_invalid_arm (accepted) |
| M037 | teacher_record | the selected checkpoint's sha256 form not checked | r_teacher_record_invalid_sha256 (code) |
| M038 | run_dir | --run-dir's absolute check removed | r_run_dir_relative (code) |
| M039 | run_dir | E1's outside-the-repository guard dropped for --run-dir | r_run_dir_inside_repo (code) |
| M040 | run_dir | the run directory's name not checked | r_run_dir_name (code) |
| M041 | run_dir | the record files' presence not checked | r_run_records_missing (error) |
| M042 | run_meta_row | a second run_meta row accepted | r_run_meta_rows (code) |
| M043 | run_meta_row | a torn run_meta line ignored | r_run_meta_rows_torn (code) |
| M044 | run_meta_row | the row's event not checked | r_run_meta_rows_event (code) |
| M045 | run_meta_checks | run_meta mode not compared | r_run_meta_mode (code) |
| M046 | run_meta_checks | run_meta stage not compared | r_run_meta_stage (code) |
| M047 | run_meta_checks | run_meta seed not compared | r_run_meta_seed (code) |
| M048 | run_meta_checks | run_meta arm not compared | r_run_meta_arm (code) |
| M049 | run_meta_checks | run_meta descriptive not compared | r_run_meta_descriptive (code) |
| M050 | run_meta_checks | run_meta parent_of_e4_e7 not compared | r_run_meta_parent_of_e4_e7 (code) |
| M051 | run_meta_checks | run_meta max_iters not compared | r_run_meta_max_iters (code) |
| M052 | run_meta_checks | run_meta poly_horizon not compared | r_run_meta_poly_horizon (code) |
| M053 | run_meta_checks | run_meta target_construction not compared | r_run_meta_target_construction (code) |
| M054 | run_meta_checks | run_meta lambda_logit not compared | r_run_meta_lambda_logit (code) |
| M055 | run_meta_checks | run_meta alpha_cwd not compared | r_run_meta_alpha_cwd (code) |
| M056 | run_meta_checks | run_meta teacher_mock not compared | r_run_meta_teacher_mock (code) |
| M057 | run_meta_checks | run_meta teacher_provenance.ckpt_sha256 not compared | r_run_meta_teacher_provenance_ckpt_sha256 (code) |
| M058 | completion | the telemetry's parse error not caught | r_telemetry_format_line (error) |
| M059 | completion | the row-type check removed | r_telemetry_format_row (error) |
| M060 | completion | the torn-line check removed | r_telemetry_torn (accepted) |
| M061 | completion | the run_abort check removed | r_parent_aborted (code) |
| M062 | completion | the incomplete exit removed | i_run_end_missing (code) |
| M063 | completion | rows after run_end accepted | r_run_end_not_last (accepted) |
| M064 | completion | checks_passed ignored | r_parent_checks_failed (accepted) |
| M065 | completion | run_end's iter not checked | r_run_end_format_iter (accepted) |
| M066 | completion | run_end's iter type not checked | r_run_end_format_iter_float (accepted) |
| M067 | completion | run_end's wall_clock_end not checked | r_run_end_format_clock (error) |
| M068 | best_checkpoint | best.json's parse error not caught | r_best_json_format_parse (error) |
| M069 | best_checkpoint | best.json's keys not checked | r_best_json_format_keys (accepted) |
| M070 | best_checkpoint | best.json's checkpoint name not checked | r_best_json_format_name (code) |
| M071 | best_checkpoint | the checkpoint's presence not checked | r_checkpoint_missing (error) |
| M072 | best_checkpoint | an unreadable checkpoint's error not caught | r_payload_unreadable (error) |
| M073 | best_checkpoint | the payload's type check removed | r_payload_not_dict (error) |
| M074 | best_checkpoint | the payload's stage not compared | r_payload_stage (accepted) |
| M075 | best_checkpoint | the payload's arm not compared | r_payload_arm (accepted) |
| M076 | best_checkpoint | the payload's target_construction not compared | r_payload_target_construction (accepted) |
| M077 | best_checkpoint | the payload's best_val_miou_all_class not compared | r_payload_best_val_miou_all_class (accepted) |
| M078 | best_checkpoint | run_end's best not compared | r_run_end_best (accepted) |
| M079 | best_checkpoint | run_end's best_ckpt not compared | r_run_end_ckpt (accepted) |
| M080 | best_checkpoint | the checkpoint's sha256 taken of best.json | w_record_fields (result) |
| M081 | launch_binding | the launch-line check removed (the launched-line check then refuses the same code) | r_launch_log_launch (text) |
| M082 | launch_binding | the launched-line check removed | r_launch_log_launched (error) |
| M083 | launch_binding | the stopped-line check removed | r_launch_log_stopped (accepted) |
| M084 | launch_binding | the last-attempt check removed | r_launch_log_attempt (accepted) |
| M085 | _launch_group | the stage left out of a launch line's group (E3-R2's later line then replaces E3-R1) | w_yes_written (result) |
| M086 | launch_binding | PL-14's run_meta_sha256 not compared | r_launch_log_run_meta_sha256 (accepted) |
| M087 | launch_binding | PL-14's git_head not compared | r_launch_log_git_head (accepted) |
| M088 | launch_binding | PL-14's code_pin not compared | r_launch_log_code_pin (accepted) |
| M089 | launch_binding | PL-14's records_commit not compared | r_launch_log_records_commit (accepted) |
| M090 | launch_binding | PL-14's stage not compared | r_launch_log_stage (accepted) |
| M091 | launch_binding | PL-14's seed not compared | r_launch_log_seed (accepted) |
| M092 | launch_binding | PL-14's horizon not compared | r_launch_log_horizon (accepted) |
| M093 | launch_binding | PL-14's sweep not compared | r_launch_log_sweep (accepted) |
| M094 | launch_binding | PL-14's lambda_logit not compared | r_launch_log_lambda_logit (accepted) |
| M095 | launch_binding | PL-14's alpha_cwd not compared | r_launch_log_alpha_cwd (accepted) |
| M096 | launch_binding | PL-14's launch_time_utc not compared | r_launch_log_launch_time_utc (accepted) |
| M097 | deadline | the deadline the day of run_end, not the day after | w_deadline_before_manila_midnight (result), w_deadline_after_manila_midnight (result) |
| M098 | deadline | run_end's day taken in UTC, not in Manila | w_deadline_after_manila_midnight (result) |
| M099 | deadline | run_end's time rounded to whole seconds, not floored | w_deadline_fraction_before_manila_midnight (result), w_deadline_last_float_before_manila_midnight (result) |
| M100 | deadline | run_end's time rounded to the microsecond, not floored | w_deadline_last_float_before_manila_midnight (result) |
| M101 | build_record | a clock time in the record | w_deterministic (result) |
| M102 | build_record | run_meta's sha256 taken of the telemetry | w_record_fields (result) |
| M103 | dl_text | the record's sha256 left out of the decision-log text | w_dl_text_one_line_aq3 (result) |
| M104 | dl_text | a '\|' in the decision-log text, its guard removed | w_dl_text_one_line_aq3 (result) |
| M105 | write_record | the read-back check removed | s_write_mismatch (accepted) |
| M106 | write_record | a failure after the create not caught | s_write_failed (error) |
| M107 | parse_record | the void check removed | d_condition_void_yes (accepted), d_condition_void_no (accepted) |
| M108 | parse_record | a record's keys not checked | d_condition_format_keys (accepted) |
| M109 | parse_record | NaN and Infinity accepted in a record | d_condition_format_nan (accepted) |
| M110 | parse_record | a record's threshold not checked | d_condition_format_threshold (accepted) |
| M111 | parse_record | a record's value not checked (AQ4) | d_condition_format_value (error) |
| M112 | parse_record | a record's format not checked | d_condition_format_format (accepted) |
| M113 | parse_record | a record's duplicate_key not checked | d_condition_format_duplicate_key (accepted) |
| M114 | parse_record | a record's repr not checked | d_condition_format_repr (accepted) |
| M115 | parse_record | a record's verdict not checked | d_condition_format_verdict (code) |
| M116 | parse_record | a record's arm not checked | d_condition_format_arm (accepted) |
| M117 | parse_record | a record's stage not checked | d_condition_format_stage (accepted) |
| M118 | parse_record | a record's parent not checked | d_condition_format_parent (accepted) |
| M119 | parse_record | a record's seed not checked | d_condition_format_seed (accepted) |
| M120 | parse_record | a record's run_id not checked | d_condition_format_run_id (accepted) |
| M121 | parse_record | a record's checkpoint not checked | d_condition_format_checkpoint (accepted) |
| M122 | parse_record | a record's sha256_best_json not checked | d_condition_format_sha256_best_json (accepted) |
| M123 | parse_record | a record's sha256_checkpoint not checked | d_condition_format_sha256_checkpoint (accepted) |
| M124 | parse_record | a record's commit not checked | d_condition_format_commit (accepted) |
| M125 | parse_record | a record's deadline_utc not checked | d_condition_format_deadline_utc (accepted) |
| M126 | parse_record | a record's deadline_manila not checked | d_condition_format_deadline_manila (accepted) |
| M127 | parse_record | a record's deadline_day not checked | d_condition_format_deadline_day (accepted) |
| M128 | read_record_file | print-dl's sha256 form not checked | d_condition_sha256_format (code) |
| M129 | read_record_file | print-dl's reader takes a forbidden name | d_read_record_file_forbidden_path (code) |
| M130 | read_record_file | print-dl's file presence not checked | d_condition_missing_file (error) |
| M131 | read_record_file | print-dl's sha256 not compared | d_condition_sha256_mismatch (accepted) |
| M132 | read_condition | read_condition's sha256 form not checked | d_read_condition_sha256_format (code) |
| M133 | read_condition | read_condition takes an absolute path | d_read_condition_not_repo_relative (code) |
| M134 | read_condition | read_condition takes a forbidden name | d_read_condition_forbidden_path (code) |
| M135 | read_condition | read_condition's file presence not checked | d_read_condition_missing_file (error) |
| M136 | read_condition | the tracked-at-HEAD check removed | d_read_condition_not_tracked (code) |
| M137 | read_condition | the unchanged-since-HEAD check removed | d_read_condition_changed_since_head (code) |
| M138 | read_condition | read_condition's sha256 not compared | d_read_condition_sha256_mismatch (accepted), d_print_dl_committed_other_record (accepted) |
| M139 | CLI.cmd_print_dl | print-dl reads a repo-relative --record as a file, not as the committed copy | w_print_dl_committed_copy (result), d_print_dl_committed_other_record (code) |
| M140 | read_committed_record | the committed copy's path not checked | d_print_dl_path_not_of_record (accepted) |
| M141 | record_condition | G1 run after the run directory's files are read | r_forbidden_path_run_dir (code) |

### B-1.6 The DL-24 verification workflow (STEP 6)

- **One read-only workflow**, run wf_159d8876-423, over `git diff d32f501 -- . <excludes>` at c6b122f (the four
  files) and the rulings. Eight agents: four reviewers (numerical; provenance; gates and record; smokes, harness and
  rulings), then one adversarial verifier per reviewer over all of its findings. Every prompt carried the GO's hard
  limits: read by exact path only, git with explicit paths or the exclude form, never a web fetch, never a write.
  The first run stopped at the usage limit after the numerical and provenance reviews (four agents: "You've hit your
  session limit · resets 9:40am (UTC)"); the same run was resumed at 09:42Z (the two finished reviews replayed from
  cache) and ended at 10:44Z with all 8 agents done.
- **Before, interim (09:41Z) and after (10:44Z)**: HEAD c6b122f, the four files' sha256 (module fa0f28f3…, CLI
  aa7f0dcf…, smoke 652e5ed9…, harness d88becc3…) and the scoped status were identical (MEASURED).
- **The numerical reviewer** recomputed the verdict by hand: the threshold itself (0.36337557435035706) gives no; one
  ulp above (0.3633755743503571) yes; one ulp below (0.363375574350357) no; the float32 neighbours a trainer can
  write (0.36337560415267944, 0.36337554454803467) yes and no. For each AQ4 value: NaN, Infinity, −Infinity, 0, 1,
  True, False, "0.37", 1.5, 1.0000000000000002 and −0.1 are value_invalid; −0.0, 0.0 and 1.0 are accepted (the bounds
  are attainable mIoUs); numpy.float64 is accepted and numpy.float32 refused. All agree with the module. Over
  1,000,001 floats, `Decimal(repr(x)) > Decimal(T)` never disagreed with `x > float(T)`. The trainer writes one Python
  float three ways (best.json, run_end, the payload), and each reads back bit-identical.
- **The provenance reviewer** mapped every gate to its clause and record. Condensed, with the module re-cited at
  b38dce3 (the callees are unchanged since O):

| gate | clause or ruling | record read | reader | at |
|---|---|---|---|---|
| G0 | plan §1.1 | argv | check_flags :198-202 | — |
| G1 | SL-1, AQ6 | none (path strings) | check_names :205-212 → qat.names_test qat.py:153 | — |
| G2 | AQ2 | none (stat of `--out`) | check_out :216-232 → runner.check_output_dir runner.py:66 → train_e1._assert_outside_repo train_e1.py:84 | — |
| G3 | AQ5 | git HEAD and status over TOOL_CODE_PATHS | tool_identity :235-249 → ptq.code_provenance ptq.py:724 | HEAD |
| G4 | 3(b) "E3-X"; AM-18 item 3(a) | configs/kd_arms/*.json | arm_spec :252-262 → kd_arms.load_arms kd_arms.py:278 | clean at HEAD via G3 |
| G5 | 3(b) "greater than E3 seed 42's, 0.36337557435035706"; DL-121; AM-23 in force from its commit | the two selections; the amendments file | selections :266-281 → qat.read_tracked_selection qat.py:622; amendment_on_record :284-294 | HEAD |
| G6 | AM-18 item 3(a) (E3-X distils from X) | reports/arms/arm_teacher_record_<X>.json | teacher_record :297-315 → sweep_select.HeadSource.read sweep_select.py:533 | HEAD |
| G7 | 3(b) "E3-X"; item 4's parent identity | e3_run_meta.jsonl | run_dir :336-351, run_meta_row :354-364, run_meta_checks :367-389 | run directory |
| G8 | 3(b) "complete (AM-19 Definitions)": AM-19 Complete (PREREGISTRATION_AMENDMENTS.md:1374-1375) | e3_telemetry.jsonl | completion :393-422 | run directory |
| G9 | 3(b) "the best VAL all-class mIoU that its best.json records"; the two sha256; AQ4 | best.json, the checkpoint | best_checkpoint :426-469 | run directory |
| G10 | PL-13, PL-14; AQ2; AM-8a (the last attempt) | reports/kd_launch_log.jsonl | launch_binding :480-515 → sweep_select.load_launch_log sweep_select.py:967 | HEAD |

  It found no AM-23 3(b) requirement without a gate except those that are not B-1's: entering the row by the
  deadline (the docs session and B-2), "before E6-X launches" and "a no rules E6-X out" (B-2), "if E3-X is cut" (no
  record can be written: no directory, no launched line or no run_end), and AM-23 being committed (PROV-1).
- **Confirmed (13)**, each fixed in a new commit with its cases and edits:

| finding | severity | fix |
|---|---|---|
| NUM-1: no case pins a fraction of a second before Manila midnight | nit | F2 19fe67b: w_deadline_fraction_before_manila_midnight; M099 |
| NUM-3: microsecond rounding dates the last half microsecond before Manila midnight to the next day | nit | F2: whole seconds, floored (:523); w_deadline_last_float_before_manila_midnight; M099, M100 |
| PROV-1: AM-23 has no committed text at O or HEAD | major | F7 b38dce3: amendment_not_committed (G5); r_amendment_not_committed, r_amendment_uncommitted, r_amendment_without_threshold; M030–M032. Committing AM-23 is open item 1 |
| PROV-5: G10 accepts an attempt that a later launch of its grid point replaced | minor | F3 a55e859: the "attempt:" check (:494-498); r_launch_log_attempt; M084, M085 |
| GATE-1: a failed write left a partial file while exit 4 said nothing was written | minor | F4 c7e56de: STOP [write_failed] (:583); s_write_failed; M106 |
| GATE-2: print-dl read any file, not the committed copy through read_condition | minor | F5 30922f6: read_committed_record (:730-739), condition_path_not_of_record; w_print_dl_committed_copy, d_print_dl_committed_other_record, d_print_dl_path_not_of_record; M139, M140 |
| GATE-3: parse_record left the sha256 fields, seed and run_id unchecked | nit | F6 21bea2a: the ok expression :666-677; the d_condition_format_ cases; M112–M127 |
| SMK-1: three kills only by message text | nit | F1 2d02ff5: r_out_exists_incomplete_run (a code-level kill of M015); the git_head case now differs in git_head alone; the remaining text kills are named in B-1.5 |
| SMK-3: M24 killed by an unrelated refusal; G5's λ halves without cases | nit | F1: committed byte copies; r_selection_not_of_record_lambda, r_selection_format_winner_lambda; M025–M028 split by clause |
| SMK-4: best_json_format's parse branch and several clauses without a case | nit | F1: r_best_json_format_parse, r_teacher_record_invalid_format/_arm/_sha256, r_run_meta_rows_torn/_event, r_run_end_format_iter_float, r_forbidden_path_selection_alpha, the field in r_missing_flags; M011, M035–M037, M043–M044, M066, M068 |
| SMK-5: most of parse_record's clauses without a case | nit | F6: one d_condition_format_ case and one edit per clause, with the repeated key (M113) |
| SMK-7: h_repository_untouched checked less than its docstring said | nit | F1: the project's scoped status over "." with ignored files, taken before the fixtures; no bytecode; the run works from a folder of its work dir |
| SMK-8: duplicate edit pairs; crash kills described as acceptances | nit | F1: one edit per fault with all its killing cases; each kill classified (B-1.5); edits described as edits |

- **Refuted (8)**, named with the verifier's reason:
  - NUM-2 (G8 accepts any finite wall_clock_end): G8 states and checks a finite number, as sweep_select's own reader
    does; only the trainer writes the field (time.time()); an out-of-range value fails safe (exit 4, nothing written).
  - PROV-2 (G10 omits PL-14's records-commit check): on the documented route it always holds (the arm gate requires
    the launch line at the records commit; a check-run-meta FAIL is closed by a stopped line, which G10 refuses); G10
    does what the plan specified (sweep_select.py:1809-1815). Kept as open item 3.
  - PROV-3 (the arm_go_rule.py citation): the cited :34-37 contains the passage (:35-37) and is the plan's own.
  - PROV-4 (parse_record's provenance fields): no record the tool writes reaches it (amendment and seed are constants;
    G3 requires clean). F6 nevertheless added seed and the sha256 forms for GATE-3.
  - PROV-6 (TOOL_CODE_PATHS without the import closure): nothing outside it decides a gate; it follows LQP's
    hand-picked pattern and exceeds AQ5's minimum.
  - SMK-2 (run_meta field edits killed by a later gate): each named case checks run_meta_mismatch and its field, and
    every edit fails it; a run failing those fields fails check-run-meta and is stopped.
  - SMK-6 (inherited G5 and G10 codes): they belong to Q2's and sweep_select's smokes, where they are covered.
  - SMK-9 (AQ9's counts not in the diff): they belong in this report (B-1.2, B-1.4).
- **The agents' conduct**: no command was refused by a hook or a permission check. They self-reported form deviations,
  all read-only and none on a test path or under docs/reference: `git ls-files` on explicit files without the exclude
  pathspec; `cat`, `wc`, `awk` or `sha256sum` instead of `sed -n`/`grep -n` for exact-path reads; one docs read by
  `awk` then re-read in the prescribed form; one command chaining single-file `grep -n` over six allowed docs files;
  `grep -c ''` on two allowed docs files; a `git ls-tree` with the exclude pathspec, which git 2.43 rejects
  ("pathspec magic not supported"), followed by `grep -vi 'test'`; `git rev-parse master origin/master` (no fetch);
  one Python parse of reports/kd_launch_log.jsonl from stdin. The SL-1 log shows one more that was not self-reported:
  `grep -rn` over src/training and src/eval (B-1.7).

### B-1.7 SL-1, tooling conflicts, slips and open items

**SL-1** (`tail -n 80 ~/.claude/sl1_guard.log`, 16 lines; log mode: nothing blocked). Classes as .claude/hooks/sl1_guard.py
:84-89 names them: B2 "git listing whose scope can include test paths" (11), G1 "Grep/Glob over a directory in a
protected location" (3), B1 "recursive, globbed or directory listing or search in a protected location" (1), B4
"test-named or NAMED_DENY path read, searched or listed" (1).

| time (UTC) | class | whose | what |
|---|---|---|---|
| 07:53:42 | B2 | mine (STEP 1) | the base checks' `git diff` name output over src, scripts, configs |
| 08:05:23 | B2 | mine | `git ls-tree -r --name-only HEAD -- configs/kd_arms` (ls-tree takes no exclude pathspec) |
| 08:05:32 | B2 | mine (slip 2) | `git ls-files -- configs/kd_arms` without the exclude, to pin the quirk of open item 5 |
| 08:47:21, 08:55:22, 09:03:22 | B2 | DL-24 agents | `git ls-tree -r --name-only HEAD -- configs/kd_arms` |
| 09:03:58 | B1 | a DL-24 agent | `grep -rn 'def miou_from_confusion' src/training src/eval`: against the hard limit "never grep -r" in its prompt; read-only, no test path |
| 09:04:34 | B2 | a DL-24 agent | `git ls-tree` on the launch log, reports/arms and the two selections |
| 09:10:40, 10:12:41, 10:12:42 | G1 | DL-24 agents (slip 4) | the Grep tool over src/quant, src and scripts with `!*[Tt][Ee][Ss][Tt]*` |
| 09:43:15, 09:43:16, 09:44:42, 10:18:42 | B2 | DL-24 agents | `git ls-tree` on configs/kd_arms, reports/arms and reports/derived |
| 10:31:24 | B4 | a DL-24 agent | `git ls-tree ... ':(exclude,icase)*test*'` (git rejects the magic) with a `git grep -l` |

**Tooling conflicts** (each reported once in the session, none complied with): the harness's designated branch
(claude/kind-brown-ljowe1) with its commit-and-push boilerplate and attribution trailers (no trailer was added and
only lane/am23-quant was used); the stop hook's push demands after C3 and later (no push beyond the GO's two).

**My slips**:
1. [L-PROT G] refused a `for` loop of `git ls-files -- "$spec"` (a variable pathspec) in STEP 5; not retried.
2. 08:05:32Z: `git ls-files -- configs/kd_arms` without `':(exclude,icase)*test*'` (logged B2), against the GO's
   search rule, while characterising open item 5.
3. [L-PROT S] refused `ls -la ~/.claude/sl1_guard.log` in STEP 8 (the hook reads an `ls` as listing the protected
   PDF); not retried; the GO's `tail -n 80` form was used.
4. The DL-24 prompts allowed the Grep tool on the src, scripts, configs and reports folders, wider than the GO's
   "named files": the three G1 lines.
5. Before C3 (never committed): a harness comment named a "ruling LB5" that exists nowhere, and the kill counter
   counted every_code_killed; both fixed before the commit.

**Open items**:
1. **Commit AM-23 (and AM-22) to docs/PREREGISTRATION_AMENDMENTS.md before the first `record` run** (PROV-1, major;
   an existing file, so not B-1's). Until an AM-23 section carrying 0.36337557435035706 is at the clone's HEAD,
   `record` refuses [amendment_not_committed]. Then re-read G8, the verdict and the deadline against the committed
   wording.
2. AQ7's timing: merge and pin B-1 before E3-R1 or E3-R2 completes (about Mon 19 to Tue 20 Oct). The record run
   needs a pin at or after the merge whose HEAD holds E3-X's launch and launched lines, both selections and AM-23.
3. PL-30 (check-run-meta PASS) is checked at launch only: the tool trusts a launch log without a stopped line and
   does not repeat PL-14's records-commit check (PROV-2, refuted; noted at the verifier's suggestion).
4. The plan's audit-hook kill of the forbidden-name order (§4.2) is a code-level kill (M141, run_records_missing):
   with never-created forbidden paths, as the GO requires, no forbidden open can happen.
5. git 2.43.0: `git ls-files -- configs/kd_arms ':(exclude,icase)*test*'` prints nothing although the folder holds
   three tracked files (`git ls-files -- configs` with the same exclude lists them), and `git ls-tree` rejects the
   exclude pathspec. Searches of that folder used `git ls-tree -r --name-only HEAD -- configs/kd_arms` (logged B2).

### B-1.8 The operator block (AQ2), as text for the orchestrator

Ice's laptop, the student image, `--network none`, `--mount` only (never `-v`), in q2-qat.md §15's Block B form
(PowerShell). Read-only binds: the clone and E3-X's folder; one writable bind: the out folder. One arm per block
(`R1` shown; `R2` alike). Every path is free of "test".

Setup:

```powershell
$IMG  = "ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
$PIN  = "<PIN_C>"       # the clone's HEAD: at or after B-1's merge, holding E3-X's launch and launched lines, both selections and AM-23
$REPO = "<CHECKOUT>"    # the clone at $PIN (a full clone)
$X    = "R1"
$RID  = "<RUN_ID>"      # E3-X's run directory name on the pod, kept: e3-r1_s42_lambda0.5_alpha25_80k_a<n>
$RUNS = "<RUNS>"        # the laptop folder holding the copy of $RID (from its keep tar)
$OUT  = "<OUT>"         # a folder outside the clone without e6x_condition_$X.json in it
$ATT  = "a1"
function RO([string]$src, [string]$dst) { @("--mount", "type=bind,source=$src,target=$dst,readonly") }
function RW([string]$src, [string]$dst) {
    New-Item -ItemType Directory -Force -Path $src | Out-Null
    @("--mount", "type=bind,source=$src,target=$dst")
}
$ENVV = @("--network","none","-e","PYTHONPATH=/work/repo","-e","PYTHONDONTWRITEBYTECODE=1",
          "-e","GIT_OPTIONAL_LOCKS=0","-e","GIT_CONFIG_COUNT=1","-e","GIT_CONFIG_KEY_0=safe.directory",
          "-e","GIT_CONFIG_VALUE_0=/work/repo","-w","/work/repo") + (RO $REPO "/work/repo")
$RUN   = RO "$RUNS\$RID" "/runs/$RID"
$STATE = "ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}}"
```

Command 0, git in the image reads the clone (prints `$PIN`):

```powershell
$NM = "e6x_${X}_${ATT}_head"; docker run --name $NM @ENVV $IMG git rev-parse HEAD; docker inspect -f $STATE $NM
```

Command 1, `record`:

```powershell
$NM = "e6x_${X}_${ATT}_record"; docker run --name $NM @ENVV @RUN (RW $OUT "/out") $IMG python -B scripts/e6x_condition.py record --arm $X --run-dir /runs/$RID --lambda-selection reports/derived/lambda_selection.json --lambda-selection-sha256 719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145 --alpha-selection reports/derived/alpha_selection.json --alpha-selection-sha256 1399b897278db17a8876c724b89c8d8e7a7b42ffe1d9c4d67002578a889f2dd8 --expect-head $PIN --out /out/e6x_condition_$X.json; docker inspect -f $STATE $NM
```

Command 2, `print-dl` on the file at `--out` (the out folder now read-only), and its sha256 on the laptop:

```powershell
$NM = "e6x_${X}_${ATT}_printdl"; docker run --name $NM @ENVV (RO $OUT "/out") $IMG python -B scripts/e6x_condition.py print-dl --record /out/e6x_condition_$X.json --record-sha256 <R>; docker inspect -f $STATE $NM
(Get-FileHash "$OUT\e6x_condition_$X.json" -Algorithm SHA256).Hash.ToLower()
```

Then Ice commits a byte-equal copy at `reports/arms/e6x_condition_R1.json`, alone in its commit, and updates the clone
to it (`$PIN2`). Command 3, `print-dl` on the committed copy (a repo-relative path):

```powershell
$NM = "e6x_${X}_${ATT}_printdl_committed"; docker run --name $NM @ENVV $IMG python -B scripts/e6x_condition.py print-dl --record reports/arms/e6x_condition_$X.json --record-sha256 <R>; docker inspect -f $STATE $NM
```

Where each value comes from:
- `--lambda-selection-sha256 719eaa0a…`: reports/derived/lambda_selection.json, the λ selection of record (DL-109).
- `--alpha-selection-sha256 1399b897…`: reports/derived/alpha_selection.json, the α selection of record (DL-121),
  whose winner.best_val is the threshold 0.36337557435035706. Both are REPOSITORY-PROVEN at O and at b38dce3
  (`git show HEAD:<path> | sha256sum`, MEASURED); the tool refuses unless each equals the file at the clone's HEAD.
- `--expect-head $PIN`: the clone's HEAD, as Command 0 prints it.
- `--record-sha256 <R>`: the `sha256:` line Command 1 prints; Get-FileHash must print the same.
- The record's own sha256 values (best.json, the checkpoint, e3_run_meta.jsonl, e3_telemetry.jsonl) are the tool's,
  computed from the copied run directory; they should equal the pod's capture of the same files where one exists.

Expected:
- Command 0: `$PIN`, then `ExitCode=0`. An empty HEAD means git refused the mount: STOP (q2-qat.md §15 Command 0).
- Command 1, exit 0: `e6x_condition/1 written: /out/e6x_condition_R1.json`, `sha256: <R>`, the decision-log header
  and its one line, then `RESULT: E6-R1 CONDITION yes (<repr> > 0.36337557435035706)` or
  `RESULT: E6-R1 CONDITION no (<repr> <= 0.36337557435035706)`. Exit 2: `RESULT: REFUSED [<code>] -- … Nothing was
  written.` (B-1.4 lists every code; `amendment_not_committed` means open item 1 is not done). Exit 3:
  `RESULT: INCOMPLETE [run_end_missing]`: E3-X has no run_end yet; run it again after E3-X completes. Exit 1:
  `RESULT: STOP [write_failed]` or `[write_mismatch]`: the file at `--out` is not a record; remove it and report.
  Exit 4: ERROR, nothing written.
- Command 2, exit 0: the same decision-log line, byte for byte, then
  `RESULT: DL TEXT (E6-R1, verdict <v>, record <R[:12]>; read from the file at --record)`.
- Command 3, exit 0: the same line again, then `RESULT: DL TEXT (...; read from the committed copy at HEAD)`. A
  refusal here (condition_sha256_mismatch, condition_not_tracked, condition_changed_since_head,
  condition_path_not_of_record) means the committed copy is not the record: do not enter the row.
- The docs session then enters the printed line verbatim, in a RECORDED row, by the record's
  `dl_deadline.enter_by_manila_date` (the end of the Manila day after run_end) and before E6-R1 launches (AQ7).

### B-1.9 Records proposed (the plan's §6 B-1 rows, as amended)

1. DL row: "L-AM23Q B-1 IMPLEMENTED (commits 524e778, d94815d, c6b122f, 2d02ff5, 19fe67b, a55e859, c7e56de, 30922f6,
   21bea2a, b38dce3 and the report commit; merged <m1>)".
2. DL-37's pin list gains <m1> (a governed-path merge: src/quant/e6x_condition.py, scripts/e6x_condition.py and the
   two smokes).
3. A RULINGS row: AQ1–AQ10 for L-AM23Q B-1, with AQ8 ("the LQP boundary is confirmed") recorded as a ruling.
4. Before the first `record` run: the docs commit of AM-23 (and AM-22) to docs/PREREGISTRATION_AMENDMENTS.md (open
   item 1; the tool refuses until it is there).
5. Each condition, per arm: the tool's printed decision-log text verbatim (one line, no `|`: AQ3), entered by the
   record's enter_by date (Asia/Manila) and before E6-X launches (AQ7), with the record committed alone at
   reports/arms/e6x_condition_<X>.json (AQ2) and its sha256 re-checked by print-dl on the committed copy (Command 3).
6. For Ch4, from the plan's §6: "E6-X ran only for an arm whose E3-X best VAL all-class mIoU exceeded
   0.36337557435035706, compared as exact decimals of the recorded values, as entered in DL-<n>."
