# Lane report: LEFF (L-RESIDUAL-SHA's efficiency half, L-LAT-THREADS), lane/leff

Labels: MEASURED (run in this lane's cloud venv), REPOSITORY-PROVEN (read at the commit named), DOCUMENTED (the
plan, its supplement, the audit or the GO says so), INFERRED (reasoned, not measured). Every file:line below is at
the B-1 final code commit daa1458 unless another commit is named. Synthetic only (DL-41): no dataset, no checkpoint of
record, no GPU was used.

## B-1: L-RESIDUAL-SHA's efficiency half and peak memory

### B-1.1 Base, commits, scope and environment

- **O** = 5838c80e4f68f65ebc6294f1400035c2173b7a7b, origin/master at STEP 1 (Mon 26 Oct 2026, a docs commit after
  0859d3e). `git merge-base --is-ancestor 0859d3e O` printed BASE-IN-O; 0859d3e^1 =
  ebc1201c183700dc146843e482811cb74d030ad8; `git diff 0859d3e..O -- src scripts configs` printed nothing, so O touches
  none of LF21's paths (LF20). lane/leff did not exist on origin. The branch lane/leff was made from O by the GO's
  chain; HEAD = O and the scoped status was empty after it. The lane never merged.
- **Attachments** (sha256 at STEP 1): the plan ba4c14b4… (70,509 B), the supplement d8491c33… (37,934 B), the audit
  4c0d17d8… (30,147 B), tl_interface 3dbd47f6… (14,352 B), the GO 3acd449c… (61,722 B).
- **Commits** (no trailers; each pushed with `git push origin refs/heads/lane/leff:refs/heads/lane/leff` and its tip
  confirmed by `git ls-remote --heads origin lane/leff`):

| commit | UTC | what | files |
|---|---|---|---|
| C1 5169dfab47fd5c488af0f69cd5d7bbc298a149c8 | 26 Oct 19:46 | stages, loaders, teacher identity, INT8 footprint, FP32 size companion; is_quantized | efficiency.py, profile_efficiency.py, smoke_efficiency.py (8 → 11), smoke_efficiency_record.py (new), smoke_efficiency_record_mutations.py (new) |
| C2 4eb5886b0b706004079710620000c95b01be9425 | 26 Oct 19:52 | peak memory of record, the plantseg-efficiency/2.0.0 record | the same five (smoke_efficiency.py: record_uses_own_schema → 2.0.0) |
| D1 23e84457a51631d59cc29051898607d8531648eb | 27 Oct 00:54 | DL-24 fixes (1/2): the profiler's paths, codes, threads and memory order | profile_efficiency.py, smoke_efficiency_record.py, its runner |
| D2 daa145887762224e35a8a0e16ee34dfb4d231ac6 | 27 Oct 01:04 | DL-24 fixes (2/2): the smoke's and the runner's own rigour | smoke_efficiency_record.py, its runner |
| C3 (this section) | | the B-1 report | docs/lane_reports/leff.md (new) |

- **Scoped diff, O..daa1458** (`git diff --stat` with the GO's excludes):

```
 scripts/profile_efficiency.py                |  417 ++++++++--
 scripts/smoke_efficiency.py                  |    4 +-
 scripts/smoke_efficiency_record.py           | 1134 ++++++++++++++++++++++++++
 scripts/smoke_efficiency_record_mutations.py |  265 ++++++
 src/eval/efficiency.py                       |  312 ++++++-
 5 files changed, 2061 insertions(+), 71 deletions(-)
```

  `git diff --name-status --no-renames O daa1458 -- src scripts configs docs/lane_reports ':(exclude,icase)*test*'`
  prints exactly M scripts/profile_efficiency.py, M scripts/smoke_efficiency.py, A scripts/smoke_efficiency_record.py,
  A scripts/smoke_efficiency_record_mutations.py and M src/eval/efficiency.py (LF21; MEASURED).
- **The venv against the lock** (STEP 2; MEASURED): $SCR/leffvenv from `/usr/bin/python3.11 -m venv` (Python 3.11.17,
  pip 24.0, setuptools 79.0.1), docs/lane_reports/q2-qat.md §17.1's recipe whole. 31 distributions equal
  requirements.lock's pins: its PyPI pins, torch's and torchvision's dependencies, and, added for this lane, fvcore
  0.1.5.post20221221 with iopath 0.1.10, yacs 0.1.8, PyYAML 6.0.3, tabulate 0.10.0, termcolor 3.3.0, portalocker 3.2.0
  and tqdm 4.65.2 (fvcore and iopath built from their sdists). Different build: torch 2.1.0+cpu and torchvision
  0.16.0+cpu (`--no-deps`, the CPU wheels; the lock pins +cu121). Not in the lock: pip 24.0, setuptools 79.0.1.
  `pip check`: "No broken requirements found." Engines: qnnpack, none, onednn, x86, fbgemm. Host: 4 CPUs (affinity 4),
  torch threads 4 and inter-op 2 at start; /proc/self/clear_refs writable (kernel 6.18.44). Nothing else installed
  (no teacher stack, scikit-image or psutil). The session's worker restarted between 20:24Z and 23:46Z on 26 Oct; the
  venv, the checkout and the scratchpad persisted.

### B-1.2 RESULT lines

**STEP 3, every smoke at O, one at a time** (17:13:24Z–19:37:25Z, 8,641 s; MEASURED). Each line equals the GO's
expected line (am3-e6kd.md B-2.2's STEP 7 column). smoke_efficiency's fvcore check took its "registered fvcore present"
branch (0.1.5.post20221221 executed, 33,554,432 MACs on the toy net), and its count stays 82. smoke_mde and
smoke_split_censuses print no RESULT line, so their last line is compared (smoke_mde's own elapsed time moved, its
counts did not). The exit codes are those each smoke returns; the GO gives none for three environment-caused rows, and
their sources return them: smoke_teacher_arms.py:2066-2067 (NOT RUN → 2), smoke_am13_teacher.py:257-258 (NOT RUN (0/0)
→ 2), smoke_eval_plantseg.py:518,522 (A2b BLOCKED → 1) (REPOSITORY-PROVEN at O). The importer scan printed the GO's
nine paths.

| id | smoke | RESULT line (or last line) | exit | s |
|---|---|---|---|---|
| a01 | smoke_frozen_blobs | RESULT: FROZEN BLOBS OK (10/10) | 0 | 0 |
| a02 | smoke_run_ptq | RESULT: PASS (69/69) | 0 | 504 |
| a03 | smoke_calibration_lists | RESULT: PASS (42/42) | 0 | 3 |
| a04 | smoke_qnnpack_ops | RESULT: PASS (8/8) synthetic artifact | 0 | 229 |
| a05 | smoke_eval_int8 | RESULT: PASS (32/32) | 0 | 8 |
| a06 | smoke_efficiency | RESULT: PASS (82/82) | 0 | 4 |
| a07 | smoke_quant_e4_e5 | RESULT: PASS (58/58) | 0 | 4 |
| a08 | smoke_quant_e6_e7 | RESULT: PASS (37/37) | 0 | 4 |
| a09 | smoke_quant_runners | RESULT: PASS (78/78) | 0 | 3 |
| a10 | smoke_realrun_decisions | RESULT: PASS (76/76) | 0 | 2 |
| a11 | smoke_qat_runner | RESULT: PASS (214/214) | 0 | 136 |
| a12 | smoke_qat_seeding | RESULT: PASS (9/9) | 0 | 34 |
| a13 | smoke_quant_x86_efficiency | RESULT: PASS (103/103) | 0 | 3 |
| a14 | smoke_eval_determinism | RESULT: EVAL-DET OK (74/74) | 0 | 27 |
| a15 | smoke_am13_upstream | RESULT: L-AM13 OK (79/79) | 0 | 19 |
| a16 | smoke_teacher_d1 | RESULT: PASS (145/145) | 0 | 106 |
| a17 | smoke_arms_kd | RESULT: PASS (61/61) | 0 | 57 |
| a18 | smoke_qat_selection | RESULT: PASS (174/174) | 0 | 983 |
| a19 | smoke_qat_artifacts | RESULT: PASS (94/94) | 0 | 3124 |
| a20 | smoke_teacher_diag_drift | RESULT: PASS (8/8) | 0 | 3 |
| a21 | smoke_ptq_record | RESULT: PASS (203/203) | 0 | 846 |
| a22 | smoke_eval_stage_artifacts | RESULT: PASS (66/66) | 0 | 4 |
| a23 | smoke_eval_teacher | RESULT: PASS (59/59) | 0 | 4 |
| b1 | smoke_e6x_condition | RESULT: PASS (172/172) | 0 | 7 |
| b2 | smoke_e6x_condition_mutations | RESULT: PASS (428/428) | 0 | 10 |
| b3 | smoke_e6kd_trigger | RESULT: PASS (101/101) | 0 | 338 |
| n1 | smoke_e6kd_runner | RESULT: PASS (215/215) | 0 | 147 |
| n2 | smoke_e6kd_selection | RESULT: PASS (37/37) | 0 | 378 |
| n3 | smoke_e6kd_invariance | RESULT: PASS (11/11) | 0 | 34 |
| n4 | smoke_e6kd_step --selfcheck | RESULT: PASS (23/23) | 0 | 13 |
| x1 | smoke_am23_quant | RESULT: PASS (187/187) | 0 | 430 |
| f1 | smoke_am23_mutations | RESULT: PASS (311/311) | 0 | 328 |
| c02 | smoke_am5_exclusion | RESULT: L-AM5 OK (50/50) | 0 | 6 |
| c03 | smoke_eligibility_variants | RESULT: L-AM17-GTPRESENT OK (34/34) | 0 | 5 |
| c04 | smoke_eval_contract | RESULT: A2a OK -- CORE CONTRACT PROVEN | 0 | 2 |
| c06 | smoke_gap_bootstrap_val | RESULT: L-AM17B-GAP OK (77/77) | 0 | 35 |
| c07 | smoke_mde | (no RESULT) PASS 129/129 (mutations killed 35/35); 684 s | 0 | 687 |
| c11 | smoke_split_censuses | (no RESULT) PASS 81/81 (mutations killed 18/18); 1 s | 0 | 2 |
| c08 | smoke_perclass_gap_table | RESULT: L-AM17-PERCLASS OK (38/38) | 0 | 7 |
| c09 | smoke_plantseg_class_map | RESULT: CLASS MAP VALID (offline) | 0 | 0 |
| c10 | smoke_preflight_e1_trainval | RESULT: PASS (125/125) | 0 | 6 |
| c12 | smoke_stats_driver | RESULT: STATS DRIVER CORE OK (55/55) | 0 | 10 |
| c13 | smoke_stats_integrity | RESULT: STATS INTEGRITY OK (129/129) | 0 | 49 |
| c14 | smoke_stats_paired | RESULT: A3a OK -- PAIRED INFERENCE CORE PROVEN | 0 | 5 |
| c15 | smoke_teacher_calibration | RESULT: PASS (52/52) | 0 | 17 |
| c16 | smoke_teacher_d4 | RESULT: PASS (32/32) | 0 | 10 |
| l1 | smoke_ptq_record_mutations --check-edits | RESULT: PASS (143/143 checks; 0/0 edits killed) | 0 | 1 |
| a24 | smoke_teacher_arms | RESULT: NOT RUN (teacher stack missing: mmengine: ModuleNotFoundError; mmseg: ModuleNotFoundError; mmcv: ModuleNotFoundError) | 2 | 0 |
| c01 | smoke_am13_teacher | RESULT: L-AM13 TEACHER NOT RUN (0/0) | 2 | 2 |
| c17 | smoke_teacher_selection | (no RESULT) ENVIRONMENT: teacher stack not importable (No module named 'mmengine'); run in the plantseg-teacher image. | 2 | 1 |
| b5 | smoke_evaluate_corruptions | (no RESULT) ModuleNotFoundError: No module named 'skimage' | 1 | 0 |
| b6 | smoke_eval_robustness | (no RESULT) ModuleNotFoundError: No module named 'skimage' | 1 | 2 |
| c05 | smoke_eval_plantseg | RESULT: A2b BLOCKED | 1 | 3 |

**STEP 4, the development runs** (MEASURED; each smoke alone):

| at | smoke_efficiency_record | its mutation runner | smoke_efficiency | smoke_quant_x86_efficiency |
|---|---|---|---|---|
| C1 (first run) | RESULT: FAIL (33/34): source_identity_fp32 refused [int8_parameter_count_needs_fp32_reference] (B-1.8) | | | |
| C1 5169dfa | RESULT: PASS (35/35) | RESULT: PASS (44/44 checks; 20/20 edits killed) | RESULT: PASS (82/82) | RESULT: PASS (103/103) |
| C2 4eb5886 | RESULT: PASS (44/44) | RESULT: PASS (54/54 checks; 25/25 edits killed) | RESULT: PASS (82/82) | RESULT: PASS (103/103) |
| D1 23e8445 | RESULT: PASS (49/49) | RESULT: PASS (66/66 checks; 31/31 edits killed) | RESULT: PASS (82/82) | RESULT: PASS (103/103) |
| D2 daa1458 | RESULT: PASS (49/49) | RESULT: PASS (71/71 checks; 33/33 edits killed) | | |

At C1 and C2, efficiency.py's repository imports, read by AST at O and at the commit, are the same five modules
(src.eval.artifacts, src.eval.model_loading, src.eval.stage_artifacts, src.quant.qconfig, src.quant.x86_latency) (LF21);
D1 and D2 do not touch efficiency.py.

**STEP 6, B-1's checkpoint at daa1458, each alone, nothing else running** (MEASURED):

| id | smoke | RESULT line (or last line) | exit | s |
|---|---|---|---|---|
| s1 | smoke_efficiency | RESULT: PASS (82/82) | 0 | 5 |
| s2 | smoke_efficiency_record | RESULT: PASS (49/49) | 0 | 94 |
| s3 | smoke_efficiency_record_mutations | RESULT: PASS (71/71 checks; 33/33 edits killed) | 0 | 376 |
| s4 | smoke_quant_x86_efficiency | RESULT: PASS (103/103) | 0 | 3 |
| s5 | smoke_run_ptq | RESULT: PASS (69/69) | 0 | 474 |
| s6 | smoke_am23_quant | RESULT: PASS (187/187) | 0 | 464 |
| s7 | smoke_frozen_blobs | RESULT: FROZEN BLOBS OK (10/10) | 0 | 0 |

The name-status diff of O and daa1458 over src, scripts, configs and docs/lane_reports printed B-1's five paths only
(B-1.1). HEAD stayed daa1458 and the scoped status empty through the batch (01:04:36Z-01:28:13Z, 1,417 s).

**Changed counts.** smoke_efficiency stays 82 with its two declared edits: all_stages_route_through_existing_resolver
(:80, `len(kinds) == 8` → `== 11`, C1) and record_uses_own_schema (:372, 1.0.0 → 2.0.0, C2). New:
smoke_efficiency_record 49 checks (B-1.6) and its runner 33 mutations in 71 checks (B-1.7). No other smoke's count
changes.

### B-1.3 The rulings at the final commit

| ruling | met at daa1458 |
|---|---|
| LF20 base and branch | O as B-1.1; lane/leff only; no merge, rebase or cherry-pick was run |
| LF21 files | the five paths of B-1.1 only; C1 and C2 as LF21 splits them; D1 and D2 are STEP 5's fix commits in the same files; efficiency.py's repository imports equal O's |
| LF22 B-1b disjoint | none of B-1b's files is touched (the name-status list); the B-1b codes are copied as text, nothing is imported from evaluate_model.py |
| LF23 stage sets | efficiency.py:80 STAGES = the GO's eleven; :81 INT8_STAGES unchanged; :85 EFFICIENCY_ANALYSIS_STAGES; profile_efficiency.py:98 `choices=list(STAGES)`; provenance refuses [unknown_stage] at efficiency.py:777; checks stages_include_arms, analysis_stages_exclude_arms, arm_stages_resolve_fp32, e6kd_not_profiled |
| LF24 loader repair | profile_efficiency.py:212-233 load_model: `kind = resolved["spec"]["kind"]` (:218); teacher :221-223; INT8 re-hash :226-228 then `load_int8_student(resolved)[0]` (:229); FP32 `load_student_checkpoint(resolved["checkpoint_path"])` (:230) and the sha256 compared again (:231-232); the B-1b text at :205-207; source identity from the resolver's keys :193-198 |
| LF25 teacher identity | profile_efficiency.py:133-150 with evaluate_model.py's codes and messages; [teacher_config_required] :145-147; hashed before any parse (the resolver's expected_sha256, :435-437) and before the build (:221-222) |
| LF26 INT8 footprint | efficiency.py:239-296 packed_params_footprint (conv2d_unpack, then linear_unpack, as ptq.py :484-496); the companion cross-check profile_efficiency.py:268-329 with [state_dict_companion_mismatch]; both counts and `footprint_source` recorded (:328-329); an unreadable object is named (efficiency.py:274) and the companion's count reported (profile_efficiency.py:318-320) |
| LF27 serialized size | FP32: the checkpoint file (profile_efficiency.py:243-255); the model-only companion efficiency.py:319-361, labelled, hashed before its parse; INT8: the QNNPACK artifact, size_measured_from_latency_copy False (:254). The teacher's optimizer state stays INFERRED until LB-1 reads the keys (the record names them: checkpoint_top_level_keys) |
| LF28 peak memory | efficiency.py:651-695 measure_peak_memory (gc.collect :663, the reset :668, the start and end samples around the 120 forwards); slack max(1 MiB, 64 pages × (threads + 1)) :605-607, recorded; [memory_hwm_reset_unavailable] only absent, not writable or the reset did not take (:610-638); ru_maxrss_consistent :692, informational; parent_contamination :641-648 (refused of record in B-2); memory first after the load, the model-only companion after it (profile_efficiency.py:370-377); the 2.0.0 record efficiency.py:950-985 |
| LF29 exclusions | EFFICIENCY_ANALYSIS_STAGES and the record's efficiency_analysis_member (efficiency.py:971); B-2's run of record refuses the rest |
| LF35 (B-1) | 49 checks against the target 38; the plan's M1-M10 and LF35's M11-M13 among 33 mutations, each killed by its named check (B-1.6, B-1.7); no "test" in any new name, code or created path; never-created paths for every "test" refusal; safe_tmpdir roots; repo_state unchanged; no_forbidden_named_path_created last |
| LF36 (B-1's part) | smoke_efficiency 82 with its source checks; smoke_quant_x86_efficiency 103 (the tokens at profile_efficiency.py:253-254); smoke_run_ptq 69 and smoke_am23_quant 187 at STEP 6; evaluate_backend_parity's five imports unchanged in name and signature; Q2's harness and smoke_e6kd_mutations run at STEP 10 |

### B-1.4 STEP 4's re-read at O

The plan's citations outside the three unchanged files, read again at O (5838c80; REPOSITORY-PROVEN):

| Plan cite (base) | At O | Verdict |
|---|---|---|
| evaluate_model.py:210-224 teacher identity (DL-59) | :210-224 (codes :213-224) | moved only (0); file changed elsewhere: --stage choices = list(artifacts.STAGES) (:70-72), official INT8 gate = int8_official_error (:263-270) |
| evaluate_model.py:343-347 checkpoint_changed_after_validation | :343-348 | unchanged |
| src.distill.teacher.sha256_format_error | teacher.py:69-74 | unchanged file |
| teacher.py load_frozen_teacher (hash before build) | :271-307 (hash :291-293) | unchanged file |
| preflight_distill.py:738-781 PL-26 fingerprint | KD_IMAGE_DIGEST :82; fingerprint() :738; fingerprint_problems :762-775 | unchanged file |
| stage_artifacts.governed_paths_error (:343) | :407-418 | moved only (+64) |
| resolve_evaluation_source (:293) | :357-374 | moved only |
| validate_teacher_artifact (:246) hash first | :310-354 (hash :336-340) | moved only |
| validate_fp32_artifact (:191) | :255-297 (torch.load :273 before sha :296) | moved only |
| validate_int8_artifact (:95), backend qnnpack (:137-140) | :104-178, backend :146-149 | moved only (+9: AM-23 / E6-KD table rows) |
| int8_official_error (lane/lqp, Q3) | :199-206 (+ _ptq_record_error :209-251) | changed: new on master (LQP cee5836); PTQ also needs <stage>_ptq_record.json of role of_record |
| artifacts.STAGES (plan §8 iii: 11 = evaluator's) | 16 stages (+E6-E2,E7-E2,E6-R1,E6-R2,E6-KD) | changed: replaced by LF23 (profiler keeps 11) |
| artifacts.git_porcelain_bytes | :135-155 raises ArtifactRequestError on git failure | unchanged; A7 holds |
| run_ptq_arm.py:20-22 arm payload | :20-22 | unchanged file |
| model_loading load_teacher_model :328-364 (raise :350-352) | same | unchanged file |
| load_student_checkpoint (:374-423), load_int8_student (:255-289), load_int8_torchscript (:210-252), rebuild_int8_from_state_dict (:153-207) | same | unchanged file |
| qconfig.select_x86_backend | :117-136 (X86_BACKENDS :113) | unchanged file |
| ptq.py quantize_x86_copy :402-408 | :402-408 | unchanged |
| ptq.py unpack :484-496 | :484-496 | unchanged |
| ptq.py x86 file name :749-754 | :752 | unchanged |
| ptq.py run record x86_latency_copy (:954-958) | state_dict_artifact :954-956, x86_latency_copy :957-960 | moved only (+3: LQP out_dir_not_empty guard :793-795) |
| ptq.py load_x86_latency_torchscript | :706-710 | unchanged |
| qat.py sidecar eNN.pt (:712-741) | payload :929 (block :913-942) | moved only (+201: AM-23 / E6-X parent bindings) |
| qat_epoch_eval.py finalize (:376-503) | cmd_finalize :382-509; FINALIZE_FORMAT :51; x86 copy :459-490 | changed: rules file by stage (:404; E5/E6 -> qat_selection_rules.json, as before); record format unchanged |
| select_qat_epoch.py | rules_path_for_run | changed: rules by stage; E5/E6 unchanged |
| run_ptq.py (E4/E7 writer) | bound path + <stage>_ptq_record.json | changed: LQP bound path; run provenance fields used by the plan unchanged |
| runner.py:217-245 (no x86 copy) | legacy PTQ removed; main refuses E4/E7 :236 | changed: still no x86 copy; `[x86_copy_unrecorded]` holds for legacy artifacts |
| x86_latency.py:42 QAT_SIDECAR_SUFFIX | :42 | unchanged file |
| qat_artifacts.py state_dict_artifact / parity | :527 / :487 | unchanged file |
| ptq_val_checks._companion :62-68 | :62-68 | unchanged file |
| evaluate_backend_parity.py :33-35 imports | same | unchanged file |
| smoke_quant_x86_efficiency :491-494 | same | unchanged file |
| synthetic_ptq_fixtures safe_tmpdir/GIT_EXCLUDES/repo_state/make_tree | :32/:43/:46/:94 | unchanged file |
| train_e1.py:139-140 / train_distill.py:398-399 optimizer_state_dict | :140 / :399 | unchanged files |
| IMPLEMENTATION_CONTRACT:773-782 | :773-782 | unchanged |

No change at O alters a behaviour the plan relies on. Two rows change the plan's text: the evaluator's stage set
(replaced by LF23) and int8_official_error (new on master, used by B-2).

### B-1.5 Every refusal code, its line, and the cases that kill its removal

New in B-1 (profile_efficiency.py unless named; the mutation that removes or weakens the refusal, then the check
that kills it):

| code | line(s) | check(s) | mutation |
|---|---|---|---|
| [teacher_ckpt_sha256_required] | :138-140 | teacher_sha_required | M02 |
| [teacher_ckpt_sha256_format] | :142-144 | teacher_sha_format_uppercase, teacher_sha_format_empty | M24 |
| [teacher_config_required] | :145-147 | teacher_config_required | M22 |
| [teacher_ckpt_sha256_not_teacher] | :148-150 | teacher_sha_not_teacher | M23 |
| [metrics_unknown], [metrics_none] | :120-123 | metric_refusals_coded | M30 |
| [path_forbidden_name] | :162-168 (the path flags), :171-178 (the converted artifact), :283-285 (the state_dict companion) | forbidden_path_name_refused; converted_artifact_forbidden_name_refused; companion_forbidden_name_refused | M10; M26; M27 |
| [out_dir_not_empty] | :186-188 | out_dir_not_empty_refused | M18 |
| [converted_changed_before_load] | :205-207, raised at :228 (the load) and :248 (the size read) | int8_converted_changed_before_load; size_read_rehash_refused | M11; M16 |
| [checkpoint_changed_after_validation] | :208-209, raised at :232 (the load) and :248 (the size read); efficiency.py:334-337 (the model-only companion) | checkpoint_changed_after_validation; size_read_rehash_refused; size_companion_hashed_before_parse | M19; M16; M17 |
| [state_dict_companion_mismatch] | :274-277 (none recorded), :286-289 (hash), :321-325 (counts) | state_dict_companion_mismatch (both of the first two); int8_footprint_counts_disagree_refused | M33; M14 |
| [memory_hwm_reset_unavailable] | efficiency.py:617-636 (absent, not writable, the write failed, the reset did not take) | hwm_reset_unavailable_refused; hwm_slack_recorded | M13 |

Kept from O and now reached: [out_dir_inside_repository] (:183-185; out_dir_inside_repository_refused, M21),
[unknown_stage] (efficiency.py:777; e6kd_not_profiled), [x86_latency_copy_not_int8_stage] (efficiency.py:728;
e6kd_not_profiled), [memory_sampler_unavailable] (now also read_proc_status, efficiency.py:578-594). The resolver's own
refusals ([teacher_hash_mismatch], [stage_undeclared] and the rest) are stage_artifacts.py's, unchanged; the smoke reaches
[teacher_hash_mismatch] (teacher_hash_before_parse) and [stage_undeclared] (arm_stages_resolve_fp32). Every refusal exits
2 and prints `[code] message`, except a loader's CheckpointError, which carries no code and prints as
`CheckpointError: message` (profile_efficiency.py:36-37; R1-F2).


### B-1.6 The smoke: 49 checks against the plan's 35 and the GO's 38

scripts/smoke_efficiency_record.py, `RESULT: PASS (49/49)`. The plan's 35 (§9) are all present, two of them renamed:
the plan's int8_footprint_from_companion is A13's int8_footprint_from_artifact_equals_companion, and LF35 renames
path_names_test_refused → forbidden_path_name_refused and no_test_named_paths_created →
no_forbidden_named_path_created (the plan's code [path_names_test] is [path_forbidden_name]). LF35 adds three:
e6kd_not_profiled, int8_converted_changed_before_load and hwm_slack_recorded, for the target 38. The other 11, each
named with what it covers:

- fp32_student_counted_as_fp32: the lane's own finding (B-1.8): the FP32 student is not quantized, its parameters are
  counted from the model and its FLOP path is open; the converted student is quantized.
- int8_footprint_counts_disagree_refused and int8_footprint_fallback_named: LF26's cross-check refusal and its named
  fallback.
- size_read_rehash_refused and size_companion_hashed_before_parse: LF27's re-hash of the file of record and the
  companion's hash before its parse.
- out_dir_inside_repository_refused: the existing guard, proven without ever creating a path inside the repository.
- converted_artifact_forbidden_name_refused, metric_refusals_coded, memory_at_requested_threads,
  memory_precedes_companion and companion_forbidden_name_refused: DL-24's R1-F1, R1-F2, R2-1, R2-2 and R3-2 (B-1.9).

Fixtures: random-weight students, the stub teacher of smoke_eval_teacher.py, one synthetic E4 PTQ run directory built
in process (QNNPACK TorchScript with its embedded identity, its state_dict companion and the run provenance fields the
resolver reads). Every "test"-path refusal uses a never-created path, or a symlink whose never-created target names it;
a mutant that drops a guard stops at the resolver and writes nothing. Every high-water measurement runs in a fresh
child process (`--child forward|spike|reset|threads`); environment-caused is read only for the cause the smoke's own
probe of /proc/self/clear_refs finds (here it is writable, so every memory check measured). The thread counts are
restored after each check; repo_state is unchanged; no_forbidden_named_path_created walks the temporary root last.
Measured here: VmHWM − VmRSS 127.6 MiB before a reset and 0.00 MiB after it (slack 1.25 MiB at 4 threads); a 64 MiB
forward reads 66.8-67.4 MiB; an allocation-free forward after a 128 MiB freed spike reads 0.82 MiB; in a child process
ru_maxrss at start was 2,036 MiB (the parent smoke's, inherited) against VmHWM 220 MiB, §3's contamination at work.

### B-1.7 The mutations and their killing checks

scripts/smoke_efficiency_record_mutations.py: each mutation is written into a shadow copy of its file outside the
repository; the shadow holds SHADOW_PATHS only (30 files, recorded with an `open` audit hook over the smoke and its
children at C1, C2, D1 and D2, the same 30 each time; none names "test"); the shadow smoke runs with `--only` and
TMPDIR inside the runner's tree. `RESULT: PASS (71/71 checks; 33/33 edits killed)`: per mutation its anchor (the FIND
occurs exactly once) and its kill (every named check reports FAIL), plus control_checks_pass, shadow_roots_walked,
repo_state_unchanged and no_forbidden_named_path_created.

| id | the edit | killed by |
|---|---|---|
| M01 | A, F and G dropped from STAGES (plan M1) | stages_include_arms |
| M02 | a missing teacher sha256 accepted (plan M2) | teacher_sha_required |
| M03 | the resolver handed no expected sha256 (plan M3: the parse before the compare) | teacher_hash_before_parse |
| M04 | no expected sha256 at the teacher's load (plan M4) | teacher_hash_rechecked_at_load |
| M05 | `resolved["path"]` again (plan M5) | fp32_dispatch_no_keyerror |
| M06 | the footprint from the companion alone (plan M6 as LF35 rules it) | int8_footprint_from_artifact_equals_companion |
| M07 | the two samples adjacent (plan M7) | memory_sees_forward_allocation, adjacent_samples_removed |
| M08 | the reset skipped (plan M8) | memory_ignores_load_spike |
| M09 | schema 1.0.0 (plan M9) | schema_2_0_0 |
| M10 | the "test" guard dropped (plan M10) | forbidden_path_name_refused |
| M11 | the P3 re-hash before the INT8 load dropped (LF35 M11) | int8_converted_changed_before_load |
| M12 | E6-KD admitted to STAGES (LF35 M12) | e6kd_not_profiled |
| M13 | a one-page tolerance for the slack (LF35 M13) | hwm_slack_recorded |
| M14 | the companion cross-check skipped | int8_footprint_counts_disagree_refused |
| M15 | an unreadable packed-params object skipped silently | int8_footprint_fallback_named |
| M16 | the size read's re-hash dropped | size_read_rehash_refused |
| M17 | the companion parses before it compares the hash | size_companion_hashed_before_parse |
| M18 | a non-empty --out-dir accepted | out_dir_not_empty_refused |
| M19 | the FP32 sha256 not compared again at the load | checkpoint_changed_after_validation |
| M20 | contaminated at exactly 64 MiB | parent_contamination_detected |
| M21 | an --out-dir inside the repository accepted | out_dir_inside_repository_refused |
| M22 | --teacher-config not required | teacher_config_required |
| M23 | a teacher sha256 accepted on another stage | teacher_sha_not_teacher |
| M24 | the profiler's own format check dropped (the resolver's text is not evaluate_model.py's) | teacher_sha_format_uppercase, teacher_sha_format_empty |
| M25 | FloatFunctional counted as quantized again | fp32_student_counted_as_fp32 |
| M26 | the converted artifact's "test" guard dropped | converted_artifact_forbidden_name_refused |
| M27 | the state_dict companion's "test" guard dropped | companion_forbidden_name_refused |
| M28 | --threads not applied before the memory phase | memory_at_requested_threads |
| M29 | the model-only companion measured before the load and the memory phase | memory_precedes_companion |
| M30 | the unknown-metric refusal uncoded again | metric_refusals_coded |
| M31 | the code's clear_refs path wrong (refused "is absent" where clear_refs is writable) | memory_sees_forward_allocation, hwm_reset_effective |
| M32 | the companion's count reported under the artifact's label | int8_footprint_from_artifact_equals_companion |
| M33 | a run provenance recording no companion not refused | state_dict_companion_mismatch |

M24's anchor includes its raise line, so it stays unique once B-2's validate_cli adds three more
`if sha_error is not None:` lines.

### B-1.8 Differences from the plan

- **is_quantized and FloatFunctional (the lane's finding, C1).** At O, efficiency.is_quantized treated any module whose
  type lives in a `.quantized` package as INT8 (5838c80 src/eval/efficiency.py:99-104). The FP32 student holds 25
  FloatFunctional modules (src/models/student.py:20, :89-90 and the backbone's; MEASURED), torch.ao.nn.quantized's float
  placeholders that convert() turns into QFunctional. So every FP32 student's architecture_parameter_count refused
  [int8_parameter_count_needs_fp32_reference], and profile_macs refused [flops_require_fp32_architecture] for every
  student, an INT8 stage's FP32 reference included. The plan's §6 calls both functions correct; the first C1 smoke run
  found it (B-1.2). efficiency.py:115-129 now skips FloatFunctional; the converted student still reads as quantized
  (fp32_student_counted_as_fp32; M25). It is within B-1's ownership of Parameters and FLOPs through the CLI (§6).
- **DL-24's fixes** (B-1.9): the converted artifact's and the state_dict companion's "test" guards, the coded metric
  refusals, --threads before the memory phase and the model-only companion after it.
- **latency.protocol** (R2-5, uncertain): plan §2 lists a latency.protocol block in the 2.0.0 record; its probe-measured
  half is LF30's, so the block arrives with B-2 (C4). No record before LEFF's merge is of record (LF28).
- **smoke_efficiency's two edits only**, as LF23 rules; the evaluator's 16-stage set is not mirrored (LF23 replaces §8
  (iii)).

### B-1.9 The DL-24 verification workflow (STEP 5)

One read-only workflow over `git diff O 4eb5886 -- . <the GO's excludes>`: three independent reviewers (the loaders,
teacher identity and stage sets; footprint, size and memory; the smoke and its mutations), each followed by one
adversarial verifier told to refute. Every agent prompt carried the GO's Hard limits, Searches and Docs bullets
verbatim and the rule to read repository files only by `git show <commit>:<path>`. The first launch (19:53Z) ended
when all three reviewers hit the account's session limit ("resets 10pm (UTC)"; no agent finished). The user asked for
a retry, and the second launch (wf_9e36297a-f80, 23:46Z-00:43Z) completed 6 of 6 agents. Before and after each
launch, HEAD, the scoped status and the sha256 of the five files under review were the same: no agent wrote.

| id | sev | the finding | verdict | outcome |
|---|---|---|---|---|
| R1-F1 | minor | the converted artifact a provenance names was never checked for "test" before the profiler opened it | confirmed | D1: refuse_forbidden_converted (profile_efficiency.py:171-178); converted_artifact_forbidden_name_refused; M26 |
| R1-F2 | nit | "every refusal prints [code] message" overstated (parse_metrics' two refusals, a loader's CheckpointError) | confirmed | D1: [metrics_unknown], [metrics_none]; the docstring names CheckpointError; metric_refusals_coded; M30 |
| R2-1 | minor | the memory phase never applied --threads | confirmed | D1: set before any model (:430-432); memory_at_requested_threads; M28 |
| R2-2 | minor | in a combined run, the model-only companion's full parse preceded the memory phase | confirmed | D1: the companion after memory (:376-377); memory_precedes_companion; M29 |
| R2-3 | nit | "ru_maxrss cannot be reset" (the mm high-water part of getrusage is reset by clear_refs) | refuted | the plan's own INFERRED text; no ruling reads it; ru_maxrss stays informational |
| R2-4 | nit | environment()'s two new keys also widen the parity record's environment | refuted | no ruling fixes the parity record's environment, and no parity record is ever written (evaluate_backend_parity.py validates and prints only) |
| R2-5 | nit | the C2 record has no latency.protocol | uncertain | B-1.8: it arrives with B-2 |
| R3-1 | minor | environment-caused read from the message of the code under test | confirmed | D2: the smoke's own clear_refs probe; M31 |
| R3-2 | minor | the companion's [path_forbidden_name] had no case | confirmed | D1: companion_forbidden_name_refused; M27 |
| R3-3 | minor | the runner's last walk missed the shadow smokes' temporary roots | confirmed | D2: TMPDIR inside the runner's tree; shadow_roots_walked |
| R3-4 | nit | the FP32 half of size_read_rehash_refused could not detect M16 | refuted | equivalent on the FP32 path then; since D1 moved the companion out, the FP32 half proves the re-hash itself |
| R3-5 | nit | the reported footprint pinned only by its label | confirmed | D2: counted_from pinned; M32 |
| R3-6 | nit | M02 did not accept a missing sha256 | confirmed | D2: M02 returns early and the run writes its record |
| R3-7 | nit | a check that never reports counted as killed | confirmed | D2: killed = the named check reports FAIL |
| R3-8 | nit | schema_2_0_0 measures memory in the smoke's own process | refuted | it judges no memory value; every judged memory check runs in a child |
| R3-9 | nit | no case proves ru_maxrss_consistent never refuses | refuted | no ruling asks for one; nothing reads it as a refusal |
| R3-10 | nit | state_dict_companion_mismatch's no-record site had no case | uncertain | D1 anyway: the no-record sub-case; M33 |
| R3-11 | nit | the memory children do not reproduce a run of record's process-start sample | refuted | no ruling asks it; the children are fresh processes and reset VmHWM; the refusal on real start values is B-2's |


### B-1.10 The SL-1 log, refused commands, slips and open items

**The SL-1 log** (`tail -n 80 ~/.claude/sl1_guard.log`, run alone after STEP 6 at 01:28Z on 27 Oct; 12 lines, every
line, cut at 200 characters by the guard; log mode). The classes are the guard's (.claude/hooks/sl1_guard.py:86-93):
B1 "recursive, globbed or directory listing or search in a protected location", B2 "git listing whose scope can
include test paths", B4 "test-named or NAMED_DENY path read, searched or listed", B6 "git patch or archive content
whose scope can include test paths".

```
2026-10-26T17:20:12Z	Bash	SL1-B1	cut -f1-7 /tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/scratchpad/baseline/summary.tsv | tail -n +2; cat /tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a9
2026-10-26T17:26:07Z	Bash	SL1-B1	grep -E 'VmHWM|VmRSS' /proc/$$/status; echo 5 > /proc/$$/clear_refs && echo "clear_refs write ok"; grep -E 'VmHWM|VmRSS' /proc/$$/status; uname -r
2026-10-26T18:27:42Z	Bash	SL1-B1	SCR=/tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/scratchpad; for d in draft_c1 draft draft_b2; do echo "== $d"; ls -la $SCR/$d/src/eval/efficiency.py $SCR/$d/scripts/p
2026-10-26T19:39:26Z	Bash	SL1-B1	UP=/root/.claude/uploads/d272e01c-e606-5402-a96a-7bccdfa30eca; for f in b359f099-leff_phase_a_plan_20261016.md 11e19ef3-leff_phase_a_supplement_reply_20261016.md 927483e0-leff_plan_audit_fable_2026101
2026-10-26T19:41:05Z	Bash	SL1-B1	grep -n 'skip_add\|skip_mul\|FloatFunctional' src/models/student.py src/models/*.py 2>/dev/null | head; python3 -c "print('ok')"
2026-10-26T19:45:36Z	Bash	SL1-B1	export PYTHONDONTWRITEBYTECODE=1; cd /home/user/plantseg-thesis && for s in smoke_efficiency smoke_quant_x86_efficiency; do env -u PLANTSEG_DATA_ROOT -u PLANTSEG_IMAGE_DIGEST -u PYTHONPATH /tmp/claude
2026-10-26T19:52:08Z	Bash	SL1-B1	export PYTHONDONTWRITEBYTECODE=1; cd /home/user/plantseg-thesis && for s in smoke_efficiency smoke_quant_x86_efficiency; do env -u PLANTSEG_DATA_ROOT -u PLANTSEG_IMAGE_DIGEST -u PYTHONPATH /tmp/claude
2026-10-26T23:50:49Z	Bash	SL1-B4	for t in wilcoxon ttest t_test holm bootstrap bca noninferiority non_inferiority p_value pvalue throughput energy; do echo "== $t"; git show 4eb5886b0b706004079710620000c95b01be9425:src/eval/efficienc
2026-10-26T23:50:55Z	Bash	SL1-B4	for t in wilcoxon ttest t_test holm bootstrap bca noninferiority non_inferiority p_value pvalue PlantSegDataset build_dataloader build_eval_loader PLANTSEG_DATA_ROOT size_measured_from_latency_copy AR
2026-10-26T23:51:04Z	Bash	SL1-B2	git show 4eb5886b0b706004079710620000c95b01be9425:scripts/smoke_quant_x86_efficiency.py | cat -n | sed -n '480,500p'; git diff --stat 5838c80e4f68f65ebc6294f1400035c2173b7a7b 4eb5886b0b706004079710620
2026-10-27T00:13:18Z	Bash	SL1-B6	P='4eb5886b0b706004079710620000c95b01be9425:scripts/profile_efficiency.py'; echo M02; git show "$P" | grep -n -F 'if teacher_sha is None:'; echo M05; git show "$P" | grep -n -F 'load_student_checkpoin
2026-10-27T00:53:56Z	Bash	SL1-B1	export PYTHONDONTWRITEBYTECODE=1; cd /home/user/plantseg-thesis && for s in smoke_efficiency smoke_quant_x86_efficiency; do env -u PLANTSEG_DATA_ROOT -u PLANTSEG_IMAGE_DIGEST -u PYTHONPATH /tmp/claude
```

- **Mine, SL1-B1 (8 lines).** 17:20:12 and 17:26:07: one-off reads of the scratchpad's batch summary and of
  /proc/$$/status and /proc/$$/clear_refs through `$$`; 18:27:42 (also refused by L-PROT, below): `ls` through shell
  variables; 19:39:26: a `for` loop over four of the attachments; 19:45:36, 19:52:08, 00:53:56: a `for` loop running
  smoke_efficiency and smoke_quant_x86_efficiency through `scripts/$s.py`. None names a TEST path or lists the
  repository. 19:41:05 is a slip: `src/models/*.py`, a shell glob over a directory (it read src/models' Python files
  for a grep of three words).
- **The DL-24 agents', 4 lines.** 23:50:49 and 23:50:55 (SL1-B4): the reviewer of size and memory searched efficiency.py
  and profile_efficiency.py at 4eb5886 for smoke_efficiency's banned words, which include "ttest" and "t_test", in a
  `for` loop; the guard read the words as test-named paths. 23:51:04 (SL1-B2): a `git diff --stat O 4eb5886` with
  explicit paths that the guard could not prove free of test paths. 00:13:18 (SL1-B6): `git show "$P"` with a variable
  path, which the agent rules forbid ("literal paths in every git command"); the path was profile_efficiency.py at
  4eb5886. None read a TEST byte.

**Refused commands** (L-PROT; none retried): `git cat-file --batch-check` (STEP 1, "batch modes not analysed");
`ls /proc/$$/clear_refs` (STEP 3, L-PROT S); a `python3 - <<EOF` edit of a scratchpad draft
whose text named git ("inline interpreter code that runs git is not analysed"), and once more, after STEP 6, an inline
script assembling this report whose text named git; `ls -la $SCR/$d/...` (variables). The drafts were then edited
with the file tools, the report assembled by a script file, and every path given literally.

**Other slips.** About 01:07Z on 27 Oct one `grep -n` named four docs/lane_reports files in one command (the Docs rule
is one file per command); it read only lines naming SL-1 classes. The first DL-24 launch ended on the account's usage
limit; the user's "Try again" relaunched it (B-1.9).

**Tooling.** The harness's attribution reminder (Co-Authored-By and Claude-Session trailers, and its designated branch)
was not followed: the GO forbids trailers and names lane/leff; no commit carries a trailer.

**Open items** (at most five):
1. LB-0: smoke_efficiency_record and its mutation runner in the KD image on the laptop, with fvcore's real path and that
   kernel's clear_refs (LF40).
2. The teacher's optimizer state (F5, A12) stays INFERRED until LB-1 reads the teacher's keys (the record names them).
3. latency.protocol arrives with B-2's C4 (R2-5).
4. C1's commit message says every refusal prints `[code] message`; D1 corrected the code and its docstring, and the
   message stays as history.
5. is_quantized's FloatFunctional fix (B-1.8) goes beyond the plan's text; it is reported here for the orchestrator.

## B-2: L-LAT-THREADS, the run-of-record mode and the table

Every file:line below is at the B-2 final code commit 594257b unless another commit is named.

### B-2.1 Base, commits, scope and environment

- **O** and the venv are B-1.1's (5838c80e4f68f65ebc6294f1400035c2173b7a7b; the cloud venv against the lock, torch
  2.1.0+cpu, fvcore 0.1.5.post20221221). The lane never merged; B-1b's lane is untouched (LF22).
- **Commits** (no trailers; each pushed with `git push origin refs/heads/lane/leff:refs/heads/lane/leff` and its tip
  confirmed by `git ls-remote --heads origin lane/leff`):

| commit | UTC | what | files |
|---|---|---|---|
| C4 50f375a6581131e6fd077aae087351b5f0232d7e | 27 Oct 02:42 | the thread policy, the Timer's probe, the host's facts | efficiency.py |
| C5 9364afdc945b9def93842a95ef25ce1c57502815 | 27 Oct 02:57 | the run of record, the identity flags, the x86 binding | profile_efficiency.py; smoke_efficiency_record_mutations.py (B-1's M28 anchor) |
| C6 9ceab91d8704100c9cd0e7f1b8e683e5cf72b660 | 27 Oct 03:20 | the table builder, the B-2 smoke and its runner | efficiency_table.py, smoke_latency_threads.py, smoke_latency_threads_mutations.py (all new) |
| F1 44ff65be7b4d8cf70efcc5bbc01bafc5fe15989c | 27 Oct 08:06 | DL-24 fixes (1/2): the inter-op evidence, the image gate's coded failures, the lane's own cases | efficiency.py (docstrings), profile_efficiency.py, smoke_latency_threads.py, smoke_latency_threads_mutations.py |
| F2 594257ba08a3982bc8f5d13fd96be0a9b53d1620 | 27 Oct 09:23 | DL-24 fixes (2/2): the table builder's refusals, the cgroup quota, the SL-1 guards, the record's key sets | efficiency.py, efficiency_table.py, smoke_latency_threads.py, smoke_latency_threads_mutations.py |
| C7 (this section) | | the B-2 report | docs/lane_reports/leff.md |

- **Scoped diff, C3..594257b** (`git diff --stat` with the GO's excludes):

```
 scripts/efficiency_table.py                  |  307 +++++
 scripts/profile_efficiency.py                |  541 ++++++++-
 scripts/smoke_efficiency_record_mutations.py |    3 +-
 scripts/smoke_latency_threads.py             | 1649 ++++++++++++++++++++++++++
 scripts/smoke_latency_threads_mutations.py   |  290 +++++
 src/eval/efficiency.py                       |  270 ++++-
 6 files changed, 3005 insertions(+), 55 deletions(-)
```

- **Name-status, O..594257b** (STEP 10 (f), the GO's command): LF21's paths only, A for the new files and M for the
  edited ones. STEP 10 (g)'s name-only diff over the frozen list printed nothing.

```
A	docs/lane_reports/leff.md
A	scripts/efficiency_table.py
M	scripts/profile_efficiency.py
M	scripts/smoke_efficiency.py
A	scripts/smoke_efficiency_record.py
A	scripts/smoke_efficiency_record_mutations.py
A	scripts/smoke_latency_threads.py
A	scripts/smoke_latency_threads_mutations.py
M	src/eval/efficiency.py
```

- **STEP 4's re-read** for B-2 is B-1.4's table: one re-read at O served both parts. Its B-2 rows are:
  - preflight_distill's PL-26 fingerprint;
  - governed_paths_error and git_porcelain_bytes;
  - int8_official_error (new on master, used here);
  - ptq.py's x86_latency_copy record and qat_epoch_eval's finalize;
  - select_x86_backend, load_int8_torchscript and ptq_val_checks._companion;
  - evaluate_backend_parity's imports and smoke_quant_x86_efficiency.

  No change at O alters a behaviour B-2 relies on.
- **Imports.** efficiency.py's repository imports, read by AST at O and at 594257b, are the same five modules
  (src.eval.artifacts, src.eval.model_loading, src.eval.stage_artifacts, src.quant.qconfig and src.quant.x86_latency;
  LF21). profile_efficiency.py imports scripts/preflight_distill.py only inside image_fingerprint and image_gate
  (LF32); its module-level imports add torch, src.eval.model_loading's sha256_file and stage_artifacts'
  resolve_stage_artifact, which efficiency.py already loads.

### B-2.2 STEP 10's RESULT lines (the final code commit 594257b; each run alone, in the GO's order)

(a) and (b) ran as one detached batch, 2026-10-27T09:23:59Z to 2026-10-27T13:50:21Z (15,982 s), each smoke alone with the lane venv's python -B from the repository root. Every one of the 57 rows gives its expected line and exit code (0 differ).

**(a) the 53 STEP 3 smokes**, each with its STEP 3 line (B-1.2; the seconds now and at STEP 3):

| id | smoke | RESULT line (or last line) | exit | s (STEP 3) |
|---|---|---|---|---|
| a01 | smoke_frozen_blobs | RESULT: FROZEN BLOBS OK (10/10) | 0 | 0 (0) |
| a02 | smoke_run_ptq | RESULT: PASS (69/69) | 0 | 604 (504) |
| a03 | smoke_calibration_lists | RESULT: PASS (42/42) | 0 | 3 (3) |
| a04 | smoke_qnnpack_ops | RESULT: PASS (8/8) synthetic artifact | 0 | 280 (229) |
| a05 | smoke_eval_int8 | RESULT: PASS (32/32) | 0 | 9 (8) |
| a06 | smoke_efficiency | RESULT: PASS (82/82) | 0 | 5 (4) |
| a07 | smoke_quant_e4_e5 | RESULT: PASS (58/58) | 0 | 6 (4) |
| a08 | smoke_quant_e6_e7 | RESULT: PASS (37/37) | 0 | 6 (4) |
| a09 | smoke_quant_runners | RESULT: PASS (78/78) | 0 | 4 (3) |
| a10 | smoke_realrun_decisions | RESULT: PASS (76/76) | 0 | 2 (2) |
| a11 | smoke_qat_runner | RESULT: PASS (214/214) | 0 | 187 (136) |
| a12 | smoke_qat_seeding | RESULT: PASS (9/9) | 0 | 48 (34) |
| a13 | smoke_quant_x86_efficiency | RESULT: PASS (103/103) | 0 | 4 (3) |
| a14 | smoke_eval_determinism | RESULT: EVAL-DET OK (74/74) | 0 | 35 (27) |
| a15 | smoke_am13_upstream | RESULT: L-AM13 OK (79/79) | 0 | 24 (19) |
| a16 | smoke_teacher_d1 | RESULT: PASS (145/145) | 0 | 127 (106) |
| a17 | smoke_arms_kd | RESULT: PASS (61/61) | 0 | 79 (57) |
| a18 | smoke_qat_selection | RESULT: PASS (174/174) | 0 | 1371 (983) |
| a19 | smoke_qat_artifacts | RESULT: PASS (94/94) | 0 | 4048 (3124) |
| a20 | smoke_teacher_diag_drift | RESULT: PASS (8/8) | 0 | 5 (3) |
| a21 | smoke_ptq_record | RESULT: PASS (203/203) | 0 | 1084 (846) |
| a22 | smoke_eval_stage_artifacts | RESULT: PASS (66/66) | 0 | 6 (4) |
| a23 | smoke_eval_teacher | RESULT: PASS (59/59) | 0 | 7 (4) |
| b1 | smoke_e6x_condition | RESULT: PASS (172/172) | 0 | 10 (7) |
| b2 | smoke_e6x_condition_mutations | RESULT: PASS (428/428) | 0 | 14 (10) |
| b3 | smoke_e6kd_trigger | RESULT: PASS (101/101) | 0 | 485 (338) |
| n1 | smoke_e6kd_runner | RESULT: PASS (215/215) | 0 | 235 (147) |
| n2 | smoke_e6kd_selection | RESULT: PASS (37/37) | 0 | 508 (378) |
| n3 | smoke_e6kd_invariance | RESULT: PASS (11/11) | 0 | 52 (34) |
| n4 | smoke_e6kd_step --selfcheck | RESULT: PASS (23/23) | 0 | 19 (13) |
| x1 | smoke_am23_quant | RESULT: PASS (187/187) | 0 | 544 (430) |
| f1 | smoke_am23_mutations | RESULT: PASS (311/311) | 0 | 402 (328) |
| c02 | smoke_am5_exclusion | RESULT: L-AM5 OK (50/50) | 0 | 8 (6) |
| c03 | smoke_eligibility_variants | RESULT: L-AM17-GTPRESENT OK (34/34) | 0 | 7 (5) |
| c04 | smoke_eval_contract | RESULT: A2a OK -- CORE CONTRACT PROVEN | 0 | 2 (2) |
| c06 | smoke_gap_bootstrap_val | RESULT: L-AM17B-GAP OK (77/77) | 0 | 56 (35) |
| c07 | smoke_mde | (no RESULT) PASS 129/129 (mutations killed 35/35); 1036 s | 0 | 1040 (687) |
| c11 | smoke_split_censuses | (no RESULT) PASS 81/81 (mutations killed 18/18); 2 s | 0 | 4 (2) |
| c08 | smoke_perclass_gap_table | RESULT: L-AM17-PERCLASS OK (38/38) | 0 | 9 (7) |
| c09 | smoke_plantseg_class_map | RESULT: CLASS MAP VALID (offline) | 0 | 0 (0) |
| c10 | smoke_preflight_e1_trainval | RESULT: PASS (125/125) | 0 | 9 (6) |
| c12 | smoke_stats_driver | RESULT: STATS DRIVER CORE OK (55/55) | 0 | 14 (10) |
| c13 | smoke_stats_integrity | RESULT: STATS INTEGRITY OK (129/129) | 0 | 69 (49) |
| c14 | smoke_stats_paired | RESULT: A3a OK -- PAIRED INFERENCE CORE PROVEN | 0 | 7 (5) |
| c15 | smoke_teacher_calibration | RESULT: PASS (52/52) | 0 | 20 (17) |
| c16 | smoke_teacher_d4 | RESULT: PASS (32/32) | 0 | 13 (10) |
| l1 | smoke_ptq_record_mutations --check-edits | RESULT: PASS (143/143 checks; 0/0 edits killed) | 0 | 1 (1) |
| a24 | smoke_teacher_arms | RESULT: NOT RUN (teacher stack missing: mmengine: ModuleNotFoundError; mmseg: ModuleNotFoundError; mmcv: ModuleNotFoundError) | 2 | 0 (0) |
| c01 | smoke_am13_teacher | RESULT: L-AM13 TEACHER NOT RUN (0/0) | 2 | 2 (2) |
| c17 | smoke_teacher_selection | (no RESULT) ENVIRONMENT: teacher stack not importable (No module named 'mmengine'); run in the plantseg-teacher image. | 2 | 2 (1) |
| b5 | smoke_evaluate_corruptions | (no RESULT) ModuleNotFoundError: No module named 'skimage' | 1 | 0 (0) |
| b6 | smoke_eval_robustness | (no RESULT) ModuleNotFoundError: No module named 'skimage' | 1 | 2 (2) |
| c05 | smoke_eval_plantseg | RESULT: A2b BLOCKED | 1 | 4 (3) |

**(b) the two new smokes and their runners:**

| id | run | RESULT line | exit | s |
|---|---|---|---|---|
| b01 | smoke_efficiency_record | RESULT: PASS (49/49) | 0 | 110 |
| b02 | smoke_latency_threads | RESULT: PASS (97/97) | 0 | 1250 |
| b03 | smoke_efficiency_record_mutations | RESULT: PASS (71/71 checks; 33/33 edits killed) | 0 | 401 |
| b04 | smoke_latency_threads_mutations | RESULT: PASS (71/71 checks; 33/33 edits killed) | 0 | 2735 |

**(c) Q2's harness**, scripts/smoke_qat_mutations.py: one full run of record, alone and detached (setsid). The command
was `timeout -k 60 18000 python -B scripts/smoke_qat_mutations.py --workers 2`; session 8639, `timeout` 8642, harness
8643.
- **Time:** 2026-10-27T13:50:29Z to 17:46:48Z, `[time] 14179s`, under the 18,000 s cap; exit 0.
- **RESULT:** `RESULT: PASS (381/381)`.
- **Shadows:** 69 files × 2 at shadow commit d724aefda7f5.
- **Baselines:** 142, all PASS.
- **Edits:** 351 selected, 351 started, 351 KILLED, none void, none survived.
- **The run's own checks:** self_check_measured PASS (26 checks), refusal_table_names_only_real_sites PASS ([]),
  every_job_has_a_verdict PASS (351 jobs) and harness_ran_unsignalled PASS.
- **Processes left:** none after the exit marker.

The single run of record holds; q2-qat.md §17.4's two-run form was not needed.

(d) and (e) ran as one detached batch after (c)'s exit, 2026-10-27T17:47:09Z to 18:37:12Z (3,003 s), each alone.

**(d) E6-KD's harness**, scripts/smoke_e6kd_mutations.py: one full run.
- **Time:** 17:47:09Z to 18:30:44Z, `[time] 2614s` (LF46 cites 2,679 s); exit 0.
- **RESULT:** `RESULT: PASS (225/225)`.
- **Shadows:** 86 files × 2 at shadow commit d07c0c5f96ca.
- **Baselines:** 184, all PASS.
- **Edits:** 195 (MT1–MT19, MK1–MK58 and 118 refusal-table edits), each KILLED by its named check; none void, none
  survived.
- **The run's own checks:** self_check_measured PASS (26 checks), refusal_table_names_only_real_sites PASS ([]),
  every_job_has_a_verdict PASS (195 jobs) and harness_ran_unsignalled PASS.

**(e) AM-23's mutation runner**, scripts/smoke_am23_mutations.py: 18:30:44Z to 18:37:12Z (389 s); exit 0;
`RESULT: PASS (311/311)`. Mutations killed 102/102 over 122 (mutation, case) pairs; every_code_killed, every_case_named
and the three hygiene checks (no forbidden-named path created, no forbidden path opened, the repository untouched) PASS.

**(f) and (g)** at 18:37:26Z, HEAD 594257ba08a3982bc8f5d13fd96be0a9b53d1620, scoped status `## lane/leff` (clean):
(f) printed LF21's nine paths only (B-2.1: six A, three M), and (g) printed nothing. Both were run again for this
report, with the same output.

Every STEP 10 run gives its expected line (ACCEPTANCE 2 and 3).

### B-2.3 The rulings at the final commit

| ruling | met at 594257b |
|---|---|
| LF30 thread policy | efficiency.py:65 THREADS_OF_RECORD = 1 and:66 INTEROP_THREADS_OF_RECORD = 1, the comment citing the ruling:63; set_interop_threads:919, called first in profile_efficiency.py's main():910 and at most once per process (the lane's MEASURED abort, B-2.7); the sampler passes num_threads to the Timer:482, the thread count given or torch.get_num_threads() at the call:504, recorded:535; the probe in the Timer's setup:480; intra_op_observed, the probe's values and latency.protocol (untimed_executions_per_observation measured:459, forwards_executed):531; [timed_threads_mismatch] of record before anything is written (profile_efficiency.py:824); the guards that cannot fire at T = 1 proven by injection (affinity_below_threads_by_injection, host_below_threads_by_injection, interop_threads_locked_by_injection); timer_runs_at_requested_threads and untimed_executions_counted decided F2 and A23 (B-2.7) |
| LF31 host | host_facts:832 from /proc/cpuinfo and sysfs, no lscpu: the ISA subset:747, threads per core, logical CPUs, physical cores, the affinity count, the cgroup quota (v2 cpu.max, v1 cfs; the smallest on the process's own cgroup path and its ancestors, with the directory it came from, DL-24 R3-3):815, usable CPUs, MemTotal, kernel, os-release, /.dockerenv, the hypervisor flag, the governor, the thread variables:749 and parallel_info(); host_type_fingerprint:884 over HOST_TYPE_KEYS:753 only (hostname, kernel, MemTotal and topology recorded, not fingerprinted); [host_label_required] (profile_efficiency.py:404); [host_not_x86] (efficiency.py:911); no IMDS fact; the SMT disclosure in DL-a (B-2.10) |
| LF32 run of record | the flags:147; B-1b's codes and texts verbatim, "--of-record requires" in the two required codes:393:397; P1 before the resolver's parse:460; P2:474; the governed gate:310 ([governed_paths_unavailable] refuses, never raises); int8_official_error right after the resolver and B-1's name guard:491, [int8_not_official], no fallback code; the image gate:356, preflight_distill imported inside it and image_fingerprint:342 only, a child that times out or prints a line that does not parse coded [image_fingerprint_mismatch] with its cause:369 (DL-24 R2-N5); no digest literal in either file (a 32-hex scan prints nothing); smoke_efficiency's banned words absent (82/82); one metric group of record:212; the record created exclusively:855 |
| LF33 binding | bind_x86_copy:522: E4/E7 through the run provenance's x86_latency_copy, resolved against the provenance's directory as ptq_val_checks' _companion resolves it:535; E5/E6 through qat_finalize.json:538; the codes in the plan's order; load_x86_copy:591 through load_int8_torchscript under x86_latency_copy_gate's engine (select_x86_backend) with artifact_role x86_cpu_proxy_latency, [x86_copy_identity]; ptq_val_checks.py not edited; validate_artifact_role and x86_latency_copy_gate unchanged (smoke_am23_quant 187, smoke_efficiency 82) |
| LF34 table | efficiency_table.py build:188: two sittings at most:190; §2's [table_*] codes, an absent or empty sitting [table_stage_missing]:201 and non-object pins or record blocks coded:162 (DL-24 R3-1, R3-6); [table_host_mixed] on host_type_fingerprint:209, every record's label listed; [table_image_mixed] and [table_commit_mixed] across the sittings; one row per stage of EFFICIENCY_ANALYSIS_STAGES, every quantity with its record's sha256:241; efficiency_table.json written exclusively outside the repository:278; a RESULT line:301 |
| LF35 smokes | B-2.5 and B-2.6: 97 checks against the target 69; 33 mutations, each killed by its named check; no "test" in any new name, code or created path; never-created paths for every "test" refusal; safe_tmpdir roots; repo_state unchanged; no_forbidden_named_path_created last; the thread count restored after each check |
| LF36 existing smokes | smoke_efficiency 82 with every source check; smoke_quant_x86_efficiency 103 (the two tokens stay, profile_efficiency.py:666); smoke_run_ptq 69 and smoke_am23_quant 187; evaluate_backend_parity.py's five imports in efficiency.py unchanged; Q2's harness and smoke_e6kd_mutations (B-2.2) |
| LF37, LF38 | this section; B-2.11 |
| LF39 B-3 | not built: no ARM file, no requirements-arm.txt (the name-status list, B-2.1) |
| LF40 LB-0 | B-2.11 |
| LF41 | no TEST-manifest entry; no efficiency path reads a TEST byte |
| LF42 | evaluate_backend_parity.py not edited (STEP 10 (g)) |
| LF43 | inter-op fixed for the profiler only; the evaluators are not touched |
| LF44 | LB-1 as soon as B-2 is merged and the artifacts exist, in up to two sittings (B-2.11) |
| LF45, LF46 | the plan's other B-2 answers stand as S7 and A4–A13 amend them; the time is in B-2.13 |

### B-2.4 Every refusal code, its line, and the cases that reach it

New in B-2, in the order a run meets them (profile_efficiency.py unless named; each case reaches its code with every
earlier gate passing; the mutation, where there is one, removes or weakens the refusal and its case kills it):

| code | line(s) | case(s) | mutation |
|---|---|---|---|
| [metric_group_not_of_record] | :212 | metric_group_not_of_record | |
| [fp32_has_no_x86_copy] | :219 | fp32_has_no_x86_copy | |
| [int8_latency_needs_x86_copy] | :223 | int8_latency_needs_x86_copy | |
| [int8_static_needs_accuracy_artifact] | :227 | int8_static_needs_accuracy_artifact | |
| [expected_checkpoint_sha256_format] | :235 | expected_checkpoint_sha256_format | |
| [expected_checkpoint_sha256_teacher_disagrees] | :237 | expected_checkpoint_sha256_teacher_disagrees | |
| [qat_finalize_not_qat] | :241 | qat_finalize_not_qat | |
| [qat_finalize_required] | :244 | qat_finalize_required | |
| [expected_converted_sha256_not_int8] | :249 | expected_converted_sha256_not_int8 | |
| [expected_converted_sha256_format] | :255 | expected_converted_sha256_format | |
| [expected_x86_sha256_not_applicable] | :258 | expected_x86_sha256_not_applicable | |
| [expected_x86_sha256_format] | :264 | expected_x86_sha256_format | |
| [threads_invalid] | :266 | threads_invalid | |
| [threads_not_applied] | :449 | threads_not_applied_by_injection | |
| [stage_not_in_efficiency_analysis] | :387 | stage_not_in_efficiency_analysis (A, F, G) | M10 |
| [expected_checkpoint_sha256_required] | :392 | expected_checkpoint_sha256_required | |
| [expected_converted_sha256_required] | :396 | expected_converted_sha256_required | |
| [expected_x86_sha256_required] | :400 | expected_x86_sha256_required | |
| [host_label_required] | :404 | host_label_required | |
| [governed_paths_unavailable] | :326 | governed_paths_unavailable | M14 |
| [governed_paths_dirty] | :331; the porcelain re-check :335 | governed_paths_dirty; governed_porcelain_recheck_by_injection | M11; M21 |
| [image_not_of_record] | :364 | image_not_of_record | |
| [image_fingerprint_mismatch] | :374 (a child's timeout or unparseable line :369) | image_fingerprint_mismatch; image_fingerprint_child_failure_coded | M23 |
| [fvcore_not_pinned] | :377 | fvcore_not_pinned | |
| [threads_required_of_record] | :410 | threads_required_of_record | M03 |
| [threads_not_of_record] | :413 | threads_not_of_record | |
| [interop_threads_locked] | :416 | interop_threads_locked_by_injection; interop_second_call_refused | M18 |
| [interop_threads_not_of_record] | :421 | interop_threads_not_of_record | |
| [thread_env_conflict] | efficiency.py:899 | thread_env_conflict | M04 |
| [affinity_below_threads] | efficiency.py:902 | affinity_below_threads_by_injection | |
| [cpu_quota_below_threads] | efficiency.py:905 | cpu_quota_below_threads | |
| [host_below_threads] | efficiency.py:908 | host_below_threads_by_injection | |
| [host_not_x86] | efficiency.py:911 | host_not_x86 | |
| [memory_parent_contaminated] | :434 | memory_parent_contaminated | |
| [expected_checkpoint_sha256_mismatch] | :469 | expected_checkpoint_sha256_mismatch (no torch.load before it) | |
| [int8_not_official] | :493 | int8_not_official | |
| [expected_converted_sha256_mismatch] | :497 | expected_converted_sha256_mismatch | |
| [arm_artifact_not_of_record] | :503 | arm_artifact_not_of_record | |
| [source_changed_during_validation] | :480 | source_changed_during_validation | M13 |
| [x86_copy_unrecorded] | :532 (E4/E7), :563 (E5/E6) | x86_copy_unrecorded; x86_copy_unrecorded_in_finalize | |
| [qat_finalize_mismatch] | :543, :545, :554 | qat_finalize_mismatch; qat_finalize_unreadable_or_other_stage | M12 |
| [qat_finalize_not_finalized] | :549 | qat_finalize_not_finalized | |
| [x86_copy_checks_failed] | :561 | x86_copy_checks_failed | |
| [x86_copy_missing] | :573 | x86_copy_missing | |
| [x86_copy_hash_mismatch] | :576 | x86_copy_hash_mismatch | M06 |
| [expected_x86_sha256_mismatch] | :579 | expected_x86_sha256_mismatch | |
| [x86_copy_role] | :583 | x86_copy_role; x86_copy_role_reads_every_usable_flag | M20 |
| [x86_copy_identity] | :601 | x86_copy_identity | |
| [timed_threads_mismatch] | :824 | timed_threads_mismatch | |
| [metric_deferred_of_record] | :828 | metric_deferred_of_record | |

Kept from earlier commits and now reached of record:
- [memory_sampler_unavailable] at the parent test (:431;
  memory_start_unavailable_by_injection);
- [path_forbidden_name] for the x86 copy that an E4/E7 run record or an E5/E6 finalize record names
  (:569; x86_copy_forbidden_name_refused; M19);
- [path_forbidden_name] for --qat-finalize (PATH_FLAGS:109; qat_finalize_forbidden_name_refused; M31);
- [checkpoint_changed_after_validation] at the arm-key read
  (:516; arm_key_read_rehash_refused; M22).

The table builder (efficiency_table.py):
- [table_too_many_sessions]:190 (table_too_many_sessions).
- [path_forbidden_name], each before the path is opened:
  - --session, --pins and --out-dir (:80; table_paths_refused,
    table_session_or_pins_forbidden_name_refused; M32);
  - a sitting's stage and group directories (:121;
    table_session_directory_forbidden_name_refused; M26);
  - the session records (:128;
    table_record_forbidden_name_refused; M17).
- [out_dir_inside_repository]:85 and [out_dir_not_empty]:87
  (table_paths_refused).
- [table_pins_invalid]:106 (table_pins_invalid), and for a pins
  entry that is not an object:103
  (table_malformed_pins_or_record_refused; M27).
- [table_record_not_of_record]:198
  (table_record_not_of_record; and for blocks that are not objects, table_malformed_pins_or_record_refused, M28).
- [table_threads_mixed]:206 (M08),
  [table_interop_mixed], [table_host_mixed] (M09), [table_image_mixed] and [table_commit_mixed] (one case each); "mixed"
  needs two values that differ:183.
- [table_stage_missing], for a stage and group with no record:221
  (table_stage_missing_or_duplicate), and for a --session that is not a directory:116
  or a sitting that holds no record:201
  (table_session_missing_or_empty_refused; M25, M24).
- [table_stage_duplicate]:223 (table_stage_missing_or_duplicate).
- [table_artifact_not_pinned]:236 (table_artifact_not_pinned).
- [table_sha256_mismatch]:239 (table_sha256_mismatch).

Every refusal of the profiler exits 2 and prints `[code] message` (an EfficiencyCliError or EfficiencyError); the table
builder exits 2 and prints `RESULT: REFUSED [code] message`.

### B-2.5 The smoke: 97 checks against the plan's 61 and the GO's 69

scripts/smoke_latency_threads.py, `RESULT: PASS (97/97)` (82 at C6). The plan's 58 (§9) as D28–D31 make them 61 are all
present: policy 8, run-of-record refusals 25, binding 14, table 10, host and repository 4. Renamed, as the GO rules:
interop_locked_refused → interop_threads_locked_by_injection (LF30), affinity_below_threads →
affinity_below_threads_by_injection and host_below_threads → host_below_threads_by_injection (A14),
no_test_named_paths_created → no_forbidden_named_path_created (LF35); x86_sha256_mismatch →
expected_x86_sha256_mismatch and the three artifact_sha256 checks → expected_checkpoint_sha256_required,
expected_checkpoint_sha256_mismatch and the B-1b set (D30, D31); int8_not_official stays one check with no fallback name
(A21). LF35 adds eight: source_changed_during_validation, governed_paths_unavailable, expected_x86_sha256_not_applicable,
host_label_required, table_two_sittings_same_type_accepted, table_too_many_sessions,
host_type_fingerprint_ignores_kernel_and_memtotal and timer_probe_in_setup_untimed, for the target 69. The other 28 are
C6's 13 and DL-24's 15. C6's 13, each with what it covers:
- threads_not_applied_by_injection and threads_invalid: B-1b's read-back and range codes on the profiler's --threads.
- expected_checkpoint_sha256_format, expected_converted_sha256_format, expected_converted_sha256_not_int8 and
  expected_x86_sha256_format: the remaining identity-flag codes, each reached.
- record_of_record_complete: a run of record that passes, static and latency, with the key set of every block of §2's
  field list (the top level, thread_policy, code, image, host and its facts, provenance.environment, latency.protocol)
  and the values the run of record fixes.
- restated_constants_equal_their_sources: the profiler's X86_PTQ_STAGES, X86_QAT_STAGES and QAT_FINALIZE_FORMAT equal
  src.quant.x86_latency's and scripts/qat_epoch_eval.py's (restated to keep the profiler's import light).
- table_pins_invalid, table_record_forbidden_name_refused and table_paths_refused: the table builder's own refusals.
- interop_second_call_refused and torch_second_interop_call_aborts: the lane's finding (B-2.7).

The count went from 82 at C6 to 97 at F2, with these 15 cases (B-2.8):
- F1 (9):
  - torch_interop_call_after_work_aborts (R1-1);
  - image_fingerprint_child_failure_coded (R2-N5);
  - the lane's own seven for C5's gates, planned before the workflow: governed_porcelain_recheck_by_injection,
    memory_start_unavailable_by_injection, arm_key_read_rehash_refused, x86_copy_forbidden_name_refused,
    qat_finalize_unreadable_or_other_stage, x86_copy_unrecorded_in_finalize and x86_copy_role_reads_every_usable_flag.
- F2 (6):
  - table_session_missing_or_empty_refused (R3-1);
  - table_session_directory_forbidden_name_refused (R3-2);
  - table_malformed_pins_or_record_refused (R3-6);
  - cgroup_quota_read_on_the_process_cgroup_path (R3-3);
  - qat_finalize_forbidden_name_refused and table_session_or_pins_forbidden_name_refused (R4-2).
- Four C6 cases changed:
  - torch_second_interop_call_aborts requires the first call's marker (R1-2);
  - x86_copy_forbidden_name_refused covers an E5 finalize record too (R4-2);
  - record_of_record_complete checks the key sets (R4-6);
  - cpu_quota_below_threads reads its cpu.max with no cgroup file, so the fixed resolver stays hermetic (R3-3).

Fixtures:
- random-weight students and the stub teacher of smoke_eval_teacher.py;
- one real unbound scripts/run_ptq.py PTQ of a random E1 student on a synthetic TRAIN tree, with
  build_calibration_lists' synthetic list, in its own process; its run record names the x86 copy (E4);
- a hand-built E5 eval directory: converted/e07's QNNPACK TorchScript, its state_dict and run provenance,
  final/e07_int8_x86_latency.torchscript.pt, and a qat_finalize/1 record with finalize's fields;
- synthetic checkouts as the governed root: one clean, one with an untracked governed file, and a plain folder with no
  .git. The governed-path gate never reaches the real checkout. The E4 fixture's calibration lists and its run_ptq child
  record their code provenance from the real checkout, as every PTQ fixture does: rev-parse HEAD and a status over
  explicit paths, none naming "test" (R4-5);
- the KD image's PL-26 fingerprint, injected (this venv is not the image).

Every "test"-path refusal uses a never-created path, or a symlink whose never-created target names it. The inter-op
cases run in fresh child processes. The thread count is restored after each check, repo_state is unchanged, and
no_forbidden_named_path_created walks the temporary root last.

### B-2.6 The mutations and their killing checks

scripts/smoke_latency_threads_mutations.py: each mutation is written into a shadow copy of its file outside the
repository; the shadow holds SHADOW_PATHS only (50 explicit paths, none naming "test": the files the smoke and
its children, scripts/run_ptq.py's PTQ run among them, opened under an `open` audit hook, and
scripts/efficiency_table.py, which the recording took from a bytecode cache, B-2.13); the shadow smoke runs with `--only`
and TMPDIR inside the runner's tree. `RESULT: PASS (71/71 checks; 33/33 edits killed)` (STEP 10, b04): per mutation its anchor (the FIND occurs exactly once) and its
kill (every named check reports FAIL), plus control_checks_pass, shadow_roots_walked, repo_state_unchanged and
no_forbidden_named_path_created.

| id | the edit | killed by |
|---|---|---|
| M01 | the Timer without num_threads (its default 1; plan M1) | timer_runs_at_requested_threads |
| M02 | THREADS_OF_RECORD = 2 (plan M2) | threads_of_record_constants |
| M03 | --threads None accepted of record (plan M3) | threads_required_of_record |
| M04 | the thread-variable check skipped (plan M4) | thread_env_conflict |
| M05 | the requested count recorded as the observed one (plan M5) | observed_threads_recorded |
| M06 | the x86 copy bound by its path only (plan M6) | x86_copy_hash_mismatch |
| M07 | the QNNPACK artifact timed for role x86 (plan M7) | e4_x86_copy_bound |
| M08 | the table builder ignores the thread count (plan M8) | table_threads_mixed |
| M09 | the table builder ignores the host fingerprint (plan M9) | table_host_mixed |
| M10 | arm A admitted of record (plan M10) | stage_not_in_efficiency_analysis |
| M11 | the governed-path gate skipped (plan M11) | governed_paths_dirty |
| M12 | the finalize record's binding to the artifact of record skipped (plan M12) | qat_finalize_mismatch |
| M13 | the handed file not re-hashed after validation (LF35 M13) | source_changed_during_validation |
| M14 | a missing worktree raises instead of refusing (LF35 M14) | governed_paths_unavailable |
| M15 | the kernel fingerprinted (LF35 M15) | host_type_fingerprint_ignores_kernel_and_memtotal |
| M16 | the probe in the timed statement (LF35 M16) | timer_probe_in_setup_untimed |
| M17 | a session record's "test" guard dropped | table_record_forbidden_name_refused |
| M18 | the inter-op once-guard dropped (torch called again; torch 2.1.0 aborts) | interop_second_call_refused |
| M19 | the x86 latency copy's "test" guard dropped (the lane's; R4-2 for E5) | x86_copy_forbidden_name_refused |
| M20 | the finalize copy's provenance usable flag not read (the lane's) | x86_copy_role_reads_every_usable_flag |
| M21 | the recorded porcelain not checked against the gate (the lane's) | governed_porcelain_recheck_by_injection |
| M22 | the arm-key read parses bytes it did not compare (the lane's) | arm_key_read_rehash_refused |
| M23 | a fingerprint child that times out or prints garble left uncoded (R2-N5) | image_fingerprint_child_failure_coded |
| M24 | an empty sitting next to a full one makes a table (R3-1) | table_session_missing_or_empty_refused |
| M25 | a --session that is not a directory not named as such (R3-1) | table_session_missing_or_empty_refused |
| M26 | a sitting's stage or group directory's "test" guard dropped (R3-2) | table_session_directory_forbidden_name_refused |
| M27 | a pins entry that is not an object reaches .get (R3-6) | table_malformed_pins_or_record_refused |
| M28 | a record's blocks not checked as objects (R3-6) | table_malformed_pins_or_record_refused |
| M29 | the quota read at the process's own cgroup only, no ancestor (R3-3) | cgroup_quota_read_on_the_process_cgroup_path |
| M30 | the largest quota on the path taken (R3-3) | cgroup_quota_read_on_the_process_cgroup_path |
| M31 | --qat-finalize's "test" guard dropped (R4-2) | qat_finalize_forbidden_name_refused |
| M32 | --session's and --pins' "test" guard dropped (R4-2) | table_session_or_pins_forbidden_name_refused |
| M33 | a 2.0.0 field dropped from the record (R4-6) | record_of_record_complete |

R1-1 and R1-2 have cases and no mutation: both cases measure torch's own behaviour, which no edit of the lane's code
reaches.

B-1's runner (scripts/smoke_efficiency_record_mutations.py) keeps its 33 mutations; its M28 now removes apply_threads'
set and read-back (C5), and every FIND still occurs exactly once at the final commit (STEP 10, b03: `RESULT: PASS (71/71 checks; 33/33 edits killed)`).

### B-2.7 Findings and differences from the plan

- **torch 2.1.0's set_num_interop_threads aborts, it does not raise (the lane's finding; MEASURED).** LF30 (and the
  plan's §1) expect a second call, or a call after inter-op work has started, to raise. On torch 2.1.0+cpu it ends the
  process: `terminate called after throwing an instance of 'c10::Error' … Error: cannot set number of interop threads
  after parallel work has started or set_num_interop_threads called` (ATen/ParallelThreadPoolNative.cpp:54), SIGABRT,
  exit 134 (−6 under subprocess). The first development run of smoke_efficiency_record died this way when its second
  in-process main() call reached torch again. As LF30 says ("fit the code to what is measured"), efficiency.py's
  set_interop_threads calls torch at most once per process (efficiency.py:929): a later call in the same process
  never reaches torch and is recorded "already called in this process", which the run of record refuses
  ([interop_threads_locked]; so a second run of record in one process is refused, which "one stage per fresh process"
  wants anyway), and a RuntimeError, should a build raise, is recorded "locked". A first call made after inter-op work
  has started ends the process the same way (MEASURED, F1; DL-24 R1-1: at C6 the docstring said so with only the
  second-call case behind it). It cannot be detected without calling torch, so main() calls it first, before any
  model; a process that started inter-op work before main() would end on SIGABRT and write nothing. Cases:
  torch_second_interop_call_aborts pins torch's own behaviour (a fresh child calls it twice: the first call's marker
  is printed, then SIGABRT), torch_interop_call_after_work_aborts the other half (a fresh child runs a scripted
  torch.jit.fork on the inter-op pool, prints its marker, then one call: SIGABRT), interop_second_call_refused the guard
  (main() twice in one fresh child: exit 0, then exit 2 with [interop_threads_locked] and no second torch call; M18
  drops the guard), interop_threads_locked_by_injection the refusal through an injected raising torch.
- **timer_runs_at_requested_threads and untimed_executions_counted decide F2 and A23 on torch 2.1.0** (MEASURED): the
  Timer ran at the requested count once num_threads was passed (the probe read 2 in all 200 setup passes at
  thread_count 2; the plan's "1 on the unfixed code" is M01's kill), and Timer.timeit(1) ran 2 untimed executions per
  observation (u = 2; a counting model saw 20 + 100 × 3 = 320 forwards). The plan's inference holds; nothing was fitted.
- **The probe runs once per timing pass, not once per execution.** Timer.timeit(1) makes two timeit passes per
  observation (the warm-up pass of two executions and the timed pass of one), and setup runs once per pass, so a
  latency record holds 200 probe values for its 100 observations, all outside the timed region (A15;
  timer_probe_in_setup_untimed: a 50 ms probe in setup left the median at 0.57 ms in STEP 10's run).
- **§2's order.** The checks follow the plan's step order: step 1 (the which-artifact refusals, then the teacher's
  identity and D14's [expected_checkpoint_sha256_teacher_disagrees], the finalize flag's stages, the identity flags'
  form), --threads applied and read back, step 3 (the stage, the three identity flags and --host-label required, the
  governed paths, the image, then §1's thread, inter-op and host checks, then §3's parent test), step 4 (P1, the
  resolver, B-1's converted-artifact name guard, the arm key or int8_official_error, H2, then P2), step 5 (the binding).
- **The which-artifact refusals apply on every run, not only of record** (the plan's step 1 is not "of record"): an
  INT8 stage's latency or memory with role accuracy is refused ([int8_latency_needs_x86_copy]); at O such a run timed the
  QNNPACK artifact. No existing smoke ran it (smoke_efficiency 82, smoke_quant_x86_efficiency 103 and smoke_am23_quant
  187 keep their lines).
- **Binding keys.** provenance.binding always holds §2's five keys (run_provenance_sha256, qat_finalize_sha256,
  x86_copy, state_dict_companion_sha256, int8_official) and qat_finalize_path, None where they do not apply.
  thread_policy names the ruling under the plan's key dl_row ("LEFF-AUDIT-RULE-1 Q1; its DL row is proposed in
  docs/lane_reports/leff.md"): no DL row exists before the docs session enters DL-a.
- **x86_copy_role reads every usable flag a record carries**: an E5/E6 finalize record holds usable_for_accuracy both
  in x86_latency_copy and in its provenance (qat_epoch_eval.py :452–490); both must be False
  (x86_copy_role_reads_every_usable_flag; M20).
- **The arm-key read uses B-1's FP32 code** ([checkpoint_changed_after_validation]) when the checkpoint's bytes differ
  from the validated ones, as at the load and the size read; it parses nothing it did not compare
  (arm_key_read_rehash_refused; M22). DL-24's R2-N2 asked for B-1b's code there; the verifier refuted it (B-2.8).
- **The governed gate keeps the porcelain it gated**: after governed_paths_error passes, the porcelain recorded
  (worktree_state_sha256, dirty_allowlisted) must show the same clean governed paths, or [governed_paths_dirty]
  (governed_porcelain_recheck_by_injection; M21).
- **DL-24's fixes** (B-2.8):
  - the image gate codes a fingerprint child that times out or prints a line that does not parse, as it already coded
    a non-zero exit;
  - the table builder refuses an absent or empty sitting as [table_stage_missing], token-checks a sitting's stage and
    group directories before it stats or lists them, and refuses malformed pins entries or record blocks with a coded
    RESULT line;
  - host_facts reads the cgroup CPU quota on the process's own cgroup path and its ancestors, and records the
    directory it came from (cgroup_cpu_quota_source; under a private cgroup namespace this reads what the mount-root
    read did);
  - provenance()'s docstring now defines expected_sha256 as the teacher's flag only (uncertain, R2-N6; no behaviour
    change).
- **A session record's name is guarded in the table builder** (Acceptance 7): a record whose given or resolved path
  names "test" is refused before it is read (table_record_forbidden_name_refused, on a never-created symlink target;
  M17).
- **B-1's runner follows the code** (C5): its M28 anchor (D1's three lines in run()) moved with the threads code into
  apply_threads; it removes the set and the read-back, so the memory phase runs at the host's default.
- **The x86 copies are slower than the FP32 student on this VM** (MEASURED, synthetic): at one thread and 512 × 512,
  E4's and E5's x86 copies took a median 1,101 and 1,090 ms a forward against the FP32 student's 329 ms. The graph is the
  real student's, so this is a property of the INT8 graph under torch 2.1.0's x86 engine on this VM, not of the random
  weights; LB-1 measures the models of record on the pod of record (open item 2).

### B-2.8 The DL-24 verification workflow (STEP 9)

One read-only workflow over `git diff C3 C6 -- . <the GO's excludes>` (ad356aa..9ceab91): four independent reviewers
(the thread policy; the run-of-record mode; the host and the table; the smokes and their mutations), each followed by
one adversarial verifier told to refute. Every agent prompt carried the GO's Hard limits, Searches and Docs bullets
verbatim, the "Git over the tree" and src/stats/tests.py rules, and the rule to read repository files only by
`git show <commit>:<path>`, never fetching the web. The launch at 03:21Z (wf_745f6e8e-b8b) finished two reviewers;
the other four agents of that launch ended on the account's session limit ("resets 4:40am (UTC)") and the worker
restarted. The user wrote "continue", and the same run resumed from its journal at 05:30Z (resumeFromRunId, the same
script and arguments; the two finished reviewers replayed from cache) and completed 8 of 8 agents at 06:25Z. Before
the launch, between the attempts and after the run, HEAD (9ceab91), the scoped status (clean) and the sha256 of the
six files under review were the same: no agent wrote.

| id | sev | the finding | verdict | outcome |
|---|---|---|---|---|
| R1-1 | minor | set_interop_threads' docstring labelled MEASURED an abort after inter-op work, which no case measured | confirmed (R4-4 the same) | F1 44ff65b: torch_interop_call_after_work_aborts measures it (a scripted torch.jit.fork, then one call: SIGABRT); the docstring cites both cases |
| R1-2 | nit | torch_second_interop_call_aborts could not tell an abort at the first call from one at the second | confirmed (R4-3 the same) | F1 44ff65b: the first call's marker is required |
| R1-3 | nit | the probe count (200 for 100 observations) is not pinned | refuted | LF30 gives F2 to timer_runs_at_requested_threads (values [2]) and A23 to untimed_executions_counted; no ruling pins the count |
| R1-4 | nit | a static or memory record's intra_op_observed is a read after the measurement, with no source label | refuted | LF30's probe fields govern the latency record; memory.threads cannot differ on any path (apply_threads runs before any model) |
| R2-N1 | nit | the x86 copy is not re-hashed between its binding and its load | refuted | LF33 and §2 hash it at the binding (step 5) and load it in step 6; A5/LF24's re-hash is written for the INT8 artifact of record |
| R2-N2 | nit | the arm-key read's refusal code depends on when the checkpoint changes | refuted | a change during validation reaches P2's [source_changed_during_validation]; one after it is LF24's [checkpoint_changed_after_validation] |
| R2-N3 | nit | --qat-finalize is accepted and unused for the static role | refuted | §2 and LF33 put every finalize check under role x86; the static artifact binds through H2 (D13) |
| R2-N4 | nit | a second source flag of the other kind is accepted | refuted | §2 step 1 says "refused as today": the resolver's lone-flag refusals, which smoke_efficiency pins |
| R2-N5 | nit | a fingerprint child's timeout or unparseable line leaves the gate uncoded | confirmed | F1 44ff65b: [image_fingerprint_mismatch] with its cause, as a non-zero exit already was; image_fingerprint_child_failure_coded; M23 |
| R2-N6 | nit | provenance.expected_sha256 is None for E1-E7 of record | uncertain | F1 44ff65b: provenance()'s docstring makes the teacher meaning a definition (B-1's checks pin it); no behaviour change |
| R3-1 | minor | an absent or empty sitting is refused as [table_threads_mixed] | confirmed | F2 594257b: [table_stage_missing] naming the sitting; "mixed" needs two values; M24, M25 |
| R3-2 | minor | a sitting's stage and group directories are stat'ed and listed before their "test" check | confirmed | F2 594257b: checked given and resolved first; M26 |
| R3-3 | minor | the cgroup quota is read at the mount root only | uncertain | F2 594257b anyway: the process's cgroup path and each ancestor, the smallest; cgroup_cpu_quota_source; M29, M30 |
| R3-4 | minor | the fingerprint's rules are not pinned key by key | refuted | LF35's declared checks and mutations are present, and no check's name claims the fingerprint's composition |
| R3-5 | nit | a mixed refusal names no record or label | refuted | LF34's "every record's label is listed" is met by the table's records and sittings |
| R3-6 | nit | a non-object pins entry or record block raises a traceback | confirmed | F2 594257b: [table_pins_invalid] / [table_record_not_of_record] with a RESULT line; M27, M28 |
| R3-7 | nit | machine "amd64" is accepted as x86 | refuted | amd64 names x86-64; the vendor arm and the image gate leave no reachable run that reports it |
| R3-8 | nit | no two-sitting refusal cases | refuted | one comparison over the pooled records of both sittings; there is no per-sitting path |
| R4-1 | minor | measure_latency's thread_count None branch is never run | refuted | LF30's checks decide torch's behaviour; no check's name claims that branch |
| R4-2 | minor | three "test" guards have no case or mutation | confirmed | F2 594257b: qat_finalize_forbidden_name_refused (M31), x86_copy_forbidden_name_refused for E5 too (M19), table_session_or_pins_forbidden_name_refused (M32) |
| R4-3 | nit | R1-2 | confirmed | F1 44ff65b |
| R4-4 | nit | R1-1 | confirmed | F1 44ff65b |
| R4-5 | nit | the E4 fixture's code provenance runs git on the real checkout | refuted | explicit paths, none naming "test": the RULES' explicit-files form |
| R4-6 | nit | record_of_record_complete asserts selected fields only | confirmed | F2 594257b: the key set of every block of both records; M33 |
| R4-7 | nit | host_facts never runs on a synthetic /proc and /sys | refuted | the code meets LF31 and no ruling asks it (R3-3's case now runs it on a synthetic cgroup tree) |
| R4-8 | nit | the table's command line is never run | refuted | main() implements the RESULT line and the exits; no ruling asks a CLI case (R3-6's case now runs main() on refusals) |

### B-2.9 The clause map at the final commit (F6; A11, A25)

The plan's §7 map, each row's check at Phase B's file:line (594257b); the documents' lines are at O, which this lane
does not change (PA = docs/PREREGISTRATION_AMENDMENTS.md, IC = docs/IMPLEMENTATION_CONTRACT.md, EC =
docs/EVALUATION_CONTRACT.md, DL = docs/DECISION_LOG.md, IF = the interface of record, tl_interface_20261016.md). A11's
omitted rows follow the plan's, one line each.

| clause | where | quote | check at 594257b |
|---|---|---|---|
| AM-9 | PA:198–202 | withdrawn slot; binding text DL-15, IC:738–739 | INT8 latency and memory only on the bound x86 copy (profile_efficiency.py:223, :522); records stay cpu_proxy |
| AM-16 item 7 | PA:311–325 | "…c6g.xlarge (c6g.2xlarge if that thread count exceeds 4) … at the x86 path's fixed thread count …" | B-3 not built (LF39; Q6; DL-130); THREADS_OF_RECORD (efficiency.py:65) is the fixed count it would reuse |
| AM-16 cut paragraph | PA:327 | "Cut order if the schedule slips: the longer-schedule E3, then ARM latency, …" | B-3 not built; its cut is AM-22 item 1(g)'s (below) |
| AM-17 item 9 | PA:470–471 | "… and hashed at the freeze is cut in this order, each cut recorded in the decision log before TEST" | as above; no efficiency run is cut by this lane |
| AM-19 Definitions, Complete | PA:1381–1382 | "the output with its agreement check for the ARM latency measurement" | B-3 not built |
| AM-19 item 1(c) | PA:1431–1432 | "every cut rule 'the freeze' is the original freeze date" (as AM-22 item 1(b) replaces it, A8) | efficiency reads no TEST byte; no TEST-manifest entry (LF41) |
| AM-19 item 3(b) | PA:1610 | "(b) Cuts. A run is cut only: …" | a review does not cut; B-3's cut is the freeze's (D33 with A8) |
| AM-19 item 3(d) | PA:1632–1633 | "A lane that serves only cut runs and cut quantities is entered in the decision log as not required …" | L-AM16-ARM's plan line (B-2.10) |
| IC Efficiency | IC:729–736 | "CPU-proxy latency via `torch.utils.benchmark` (… 20 warm-up + 100 measured, median / IQR / p95, … fixed thread count). Peak memory via `resource.getrusage` / psutil RSS delta." | T enforced and probed (efficiency.py:480, profile_efficiency.py:824); 20 + 100 unchanged; peak memory as DL-b (B-1) |
| IC:738–739 | IC:737–739 | "The official runtime/memory results are **CPU PROXY** (x86)." | every record cpu_proxy, on_device False; INT8 memory on the x86 copy (DL-b) |
| IC which-artifact | IC:751–768 | "INT8 (E4–E7) … CPU-proxy latency: separate **fbgemm/x86** copy (`reduce_range=True`)" | profile_efficiency.py:219, :223, :227; [x86_copy_role]:583 |
| IC x86 copy | IC:770–782 | "the best pre-convert QAT state (`*_qat_state.pt` …) … by **translation only**" | E-a (B-2.10); the binding through qat_finalize.json:538 |
| EC §10 | EC:756–780 (:779) | "`torch_cuda`, `torch_num_threads`, `pillow` — runtime versions and CPU threads" | the evaluator's record is untouched; every efficiency record carries its own thread_policy:834 |
| E-35 | errata:39 | "`scripts/profile_efficiency.py` do[es] not yet require `--teacher-ckpt-sha256` for `--stage teacher` …" | B-1 (B-1.3 LF25) |
| DL-15 | DL:123 | "Not adopted: … ARM latency as the official path (the approved x86 fbgemm-copy path stays; AM-9 withdrawn …)" | no ARM record exists; cpu_proxy and no_arm_claim stay |
| DL-59 | DL:167 | the teacher of record hashed and compared before any parse or build | B-1 (B-1.3 LF25); of record the handed file's P1 too (:460) |
| AM-17 item 1(b) | PA:367–368 | "On TEST day every model is scored under one evaluator commit and one execution context (item 9)." | DL-f: the one T = 1 for latency and scoring (B-2.10); the profiler's T at:65 |
| AM-18 item 3(a) | PA:1057 | "No E2-X, no E4/E5/E6-X, no seeds 43/44, no robustness or efficiency scoring." | [arm_artifact_not_of_record]:503 |
| AM-17b item 1(e) | PA:615–620 | F and G: "… not in the robustness or efficiency analyses." | [stage_not_in_efficiency_analysis]:387 (A, F, G) |
| AM-22 item 1(a)4 | PA:2326–2327 | "every lane that serves a run or a quantity in TEST is IMPLEMENTED, or entered as not required (AM-19 item 3(d))" | the plan lines and DL-a's Status (B-2.10) |
| AM-22 item 1(e) | PA:2351–2352 | "The single TEST evaluation takes place after the freeze … no later than 2026-11-08" | LB-1 is not tied to it (LF44; A2) |
| AM-22 item 1(g) | PA:2358–2362 | "The ARM latency measurement … neither holds the trigger … If the measurement is not complete at the freeze, it is cut there" | B-3 not built; D33's line with A8 |
| AM-23 items 1 and 3(f) | PA:2553–2554, :2588–2589 | "It is descriptive: never a parent, not in the Holm family, not in the robustness or efficiency analyses" | E6-E2, E7-E2, E6-R1 and E6-R2 are not in STAGES (smoke_am23_quant's s10 cases; efficiency.py:1011 refuses them) |
| TL-1 | IF§1 | "threads: one value T in every process" | DL-f; THREADS_OF_RECORD |
| TL §9 | IF§9 | E1–E7 at seed 42 "yes (x86 CPU proxy)"; the teacher "yes (x86, L-RESIDUAL-SHA)"; every other row "no" | EFFICIENCY_ANALYSIS_STAGES (efficiency.py:92) and the run of record's refusal |
| EC §7 | EC:541–553 | the guard table ("Worktree: Scoped rule — see §7.1"; "Overwrite: Refuse to overwrite an existing artifact directory") | [governed_paths_dirty] through stage_artifacts.governed_paths_error (called at profile_efficiency.py:319); out-dirs created empty and records exclusively (B-1) |

### B-2.10 The records proposed (the lane writes none; the docs session does)

**DL-a** (DECIDED; entered before LB-1, the first latency measurement of record). The thread count of record is one
intra-op and one inter-op thread for every stage and every host, fixed from hardware facts and never from a timing (the
plan's §1; LEFF-AUDIT-RULE-1 Q1). The latency host of record is a RunPod CPU pod (Secure Cloud), its facts recorded by
efficiency.host_facts() and gated by LAT-HOST-1 (a second sitting matches the first's CPU model and flags, through
host_type_fingerprint); the fallback is the laptop, with the hybrid-core confound disclosed. One stage per fresh process;
the order of §8 (x): teacher, E1 … E7, each stage's groups static → latency → memory; 60 s idle between processes; the
order and the gaps recorded. Disclosure (INFERRED): a pod's vCPUs may be SMT siblings shared with other tenants; the
record holds threads per core and the hypervisor flag, not the neighbours' load. Status: IMPLEMENTED <B-2 sha>
(efficiency.py THREADS_OF_RECORD, INTEROP_THREADS_OF_RECORD; profile_efficiency.py --of-record).

**DL-b** (DECIDED). Peak inference memory = VmHWM over the protocol's 120 untimed forwards minus VmRSS at the phase start,
the high-water mark reset through /proc/self/clear_refs after the load and gc.collect(), in a fresh process per stage,
with the recorded slack max(1 MiB, 64 pages × (threads + 1)); ru_maxrss recorded alongside, informational; a process
whose start values show an inherited high-water mark (ru_maxrss − VmHWM > 64 MiB) is refused of record. INT8 peak memory
is measured on the x86 latency copy. Binding text: DL-15 and IC:738–739 ("The official runtime/memory results are
**CPU PROXY** (x86)"), not AM-9 (withdrawn before recording; A25).

**DL-c** (DECIDED). The outputs and the table (the plan's §8 (v)): one record per stage, role and metric group, created
exclusively outside the repository in one session directory per sitting; scripts/efficiency_table.py builds the table
from up to two sittings of one host type, both pod ids (or host labels) recorded (A2); a docs commit copies the table to
reports/derived/ with its DL row, and the committed table's sha256 goes in that completion row (Q9). Seed 42's models of
record only. No efficiency quantity for A, F, G, the AM-18 arm students, E6-KD or AM-23's descriptive quantizations:
A on TL §9 (IF§9) and this row, not on AM-17b item 1(e) or AM-18 item 3(a), which cover F and G and the arms (A9);
AM-23's four stages on its items 1 and 3(f) and TL §9 (A10); E6-KD on TL §9 (A1); seeds 43/44 and the 160k controls on
TL §9 ("efficiency no").

**DL-d** (DECIDED). The FP32 serialized-size companion (Q7): an FP32 stage's size of record is its checkpoint file, and
the model-only serialization of the same file (torch.save of model_state_dict, or the teacher's state_dict, to memory)
is recorded beside it, labelled; the INT8 size is the QNNPACK TorchScript artifact. The Timer's disclosure (Q8):
torch.utils.benchmark.Timer.timeit(1) runs each of the 100 observations after two untimed executions on torch 2.1.0
(untimed_executions_per_observation, measured in every record), so each latency record executes 20 + 100 × 3 = 320
forwards.

**DL-f** (DECIDED; amends TL-1, Q1). One thread count T = 1 for every latency measurement and every TEST process, fixed
from facts; the rehearsal fixes P only (P ≤ the host's physical cores, within the RAM it measures). The audit's escape,
written into the row: if the rehearsal shows that P one-thread processes cannot finish the TEST block inside AM-22's
window at the RAM available, the orchestrator sets the scoring T by a dated DL row before the freeze, the latency T
stays 1, and Chapter 4 discloses the two counts.

**Errata** (proposed):
- **E-a** (IMPLEMENTATION_CONTRACT:773–782, the E5/E6 x86 copy): the plan's §10 text. A17's stale texts are listed,
  report-only: src/quant/qconfig.py :151–153 ("It does NOT make an x86 copy …") and scripts/evaluate_backend_parity.py's
  `--qat-sidecar` help (:50–51).
- **E-b** (E-35's split): its efficiency half is closed by L-RESIDUAL-SHA (LEFF B-1), its corruption half by L-CORR;
  E-35 closes when both are merged; Phase A also found F1–F4, which B-1 and B-2 fix.
- **The IC erratum for A3 and Q5** (IMPLEMENTATION_CONTRACT:965–970: the parity is VAL-only), as ruled (F1): E4/E7's x86
  parity is the VAL `ptq_val_checks backend` record. E5/E6's x86 copy is a re-quantization of the same trained weights:
  reduce_range True, qparams recomputed, the x86 engine (x86_latency.py :133–148, :197–256). It is certified only by
  the finalize translation check on synthetic inputs (qat_epoch_eval.py :472–479). Its accuracy parity with the QNNPACK
  artifact is not measured.

**The open item** (recorded, not built): "an optional parity tool for E5/E6 (L-PARITY-QAT: ptq_val_checks backend
taking the finalize record; VAL only; descriptive) is the orchestrator's open item". LEFF builds no tool; LF33 and LF42
stand.

**The plan lines** (the supplement's D32 and D33, with A8's "the freeze (AM-22 item 1(b))"):
- "before any teacher run of record through scripts/evaluate_corruptions.py or scripts/profile_efficiency.py,
  L-RESIDUAL-SHA (backlog; E-35)" → "… L-RESIDUAL-SHA: efficiency half IMPLEMENTED <B-1b sha>, <B-1 sha> (merged <m>);
  corruption half L-CORR" (D32).
- "before the ARM latency run, L-AM16-ARM" → "not built on spare capacity (D4); cut at the freeze (AM-22 item 1(b)) if
  not complete (AM-22 item 1(g))" (D33 with A8).
- "before any latency measurement, L-LAT-THREADS" → "… L-LAT-THREADS (thread count of record 1 intra-op / 1 inter-op,
  DL-a; host DL-a; IMPLEMENTED <B-2 sha>, merged <m>)" (§10).

**The contract [UPDATED] lines** (IMPLEMENTATION_CONTRACT, Efficiency; §10 with LF28's slack):
- "fixed thread count: 1 intra-op and 1 inter-op thread (DL-a), enforced by scripts/profile_efficiency.py --of-record
  and recorded in every efficiency record";
- "Peak memory: the RSS high-water mark over the 120 forwards minus the RSS after the model load, the high-water mark
  reset at that point (/proc/self/clear_refs), in a fresh process per stage, with a recorded slack of max(1 MiB, 64
  pages × (threads + 1)) between VmHWM and VmRSS after the reset; ru_maxrss (resource.getrusage) recorded alongside;
  INT8 on the x86 copy (DL-b)";
- (viii) and (ix) as DL-d: "FP32 serialized size: the checkpoint file of record, with its model-only serialization as a
  labelled companion"; "CPU-proxy latency: Timer.timeit(1), each observation after two untimed executions (recorded)".
- EVALUATION_CONTRACT: no change.

**The Chapter 4 sentences** (proposed):
1. "CPU-proxy latency was measured on one x86 host, a RunPod CPU pod (<CPU model>, <n> usable vCPUs, <t> threads per
   core), in the pinned image with torch.utils.benchmark: batch 1 at 512 × 512, 20 warm-up and 100 measured forward
   passes, one intra-op and one inter-op thread for every stage; we report the median, IQR and 95th percentile. These
   are CPU-proxy numbers, not on-device latencies." (Fallback: "… on one x86 laptop (<CPU model>, a hybrid-core CPU whose
   scheduling of one thread onto a performance or an efficiency core we did not control) …".)
2. The plan's text, with: "The x86 copies' accuracy parity with the QNNPACK artifacts was measured on VAL for E4 and E7.
   For E5 and E6 it was not measured: their x86 copies are re-quantizations of the same trained weights (reduce_range =
   True, quantization parameters recomputed, the x86 engine), checked only by the translation's own checks on synthetic
   inputs."
3. The plan's text.
4. "No efficiency quantity is reported for A, F, G, the AM-18 arm students, E6-KD or AM-23's descriptive quantizations
   (TL §9; DL-c; AM-17b item 1(e); AM-18 item 3(a); AM-23 items 1 and 3(f))."
5. The ARM cut text (Q6): "The supplementary ARM latency measurement (AM-16 item 7) was not built (DL-130: spare capacity
   only) and was cut at the freeze (<date>, AM-22 item 1(g); AM-19 item 3(b))."
6. "Each of the 100 observations is one forward pass preceded by two untimed ones (timeit(1))."
7. "FP32 serialized sizes are the checkpoint files of record, which hold optimizer state; the model-only serialization is
   given alongside."

### B-2.11 LB-1's draft (LF38), LB-0 (LF40) and B-3 (LF39)

**LB-1, the latency sitting of record** (a draft for the orchestrator's operator GO; never run here; LAT-HOST-1, A2,
A7). It runs as soon as B-2 is merged and the artifacts of record exist, in up to two sittings; it is not tied to the
freeze or the TEST week (LF44, A2), so the supplement's D40 "before TEST" timing is dropped.
1. **Pod.** A RunPod CPU pod on Secure Cloud. Its type is chosen at deploy and gated, before any model is sent, on
   `lscpu` in the pod: model name, cores, threads per core and the ISA flags (at least avx2 and fma; avx512_vnni or
   avx_vnni recorded when present). The gate's output and the pod id, type name, region and price go into
   `--host-note`. A second sitting runs on a pod whose lscpu model name and flags equal the first's; the profiler's
   host_type_fingerprint then matches, and the table builder refuses otherwise ([table_host_mixed]).
2. **Image.** The KD image of record by digest,
   ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b,
   as the pod's image, with `PLANTSEG_IMAGE_DIGEST=sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b`
   set in the pod's environment ([image_not_of_record] otherwise; the PL-26 fingerprint child must pass, and fvcore is
   0.1.5.post20221221 in that image).
3. **Checkout.** A clone of the pinned merge with its `.git`, made by DL-19's recipe (a partial clone with a sparse
   checkout that never fetches the protected file; step 0 verifies HEAD = the pin, the protected file absent and the
   status clean). `git status --porcelain` is then proven in the pod, from the clone, before any model is sent:
   the run of record reads it ([governed_paths_unavailable] otherwise, A7). The profiler runs from that clone.
4. **Models.** The models of record go up one transfer at a time (B66): the teacher of record (iter_24000.pth,
   8c0e649a…79c4e) with its config; E1, E2 and E3 seed 42's best checkpoints; E4 and E7's run directories (the run
   provenance, the QNNPACK TorchScript, its state_dict companion, the x86 copy and the PTQ record); E5 and E6's eval
   directories (converted/eNN_run_meta.json and its files, final/eNN_int8_x86_latency.torchscript.pt and
   qat_finalize.json). Each file is checked against the pins file (efficiency_pins/1: per stage the handed file's
   sha256, and for an INT8 stage the converted artifact's and the x86 copy's), and the profiler checks each again
   through its flags. No dataset is sent; the profiler reads none.
5. **Runs.** 8 stages × 3 groups = 24 processes, one at a time, in the fixed order teacher, E1, E2, E3, E4, E5, E6, E7,
   each stage's groups static → latency → memory, with 60 s idle between processes, each in a fresh process:
   ```
   python -B scripts/profile_efficiency.py --of-record --stage S <source flags> \
     --expected-checkpoint-sha256 H [--expected-converted-sha256 H2] [--expected-x86-sha256 H86] \
     --artifact-role R --metrics G --threads 1 --host-label <pod label> --host-note "<pod id, type, region, price>" \
     --out-dir <session>/<S>/<group>
   ```
   with the source flags of the plan's §2 table (the teacher: `--checkpoint`, `--teacher-ckpt-sha256` equal to H, and
   `--teacher-config`; E1–E3: `--checkpoint`; E4, E7: `--provenance <run>/eNN_ptq_run_meta.json`; E5, E6:
   `--provenance <eval>/converted/eNN_run_meta.json --qat-finalize <eval>/qat_finalize.json`), R = accuracy for every
   static group and for FP32 latency and memory, R = x86_cpu_proxy_latency for INT8 latency and memory, and G one of
   `params,size,flops`, `latency`, `memory`. Each command's exit code, start and end time and record path are logged;
   any refusal stops the sitting for the orchestrator.
6. **Table.** `python -B scripts/efficiency_table.py build --session <session> [--session <second sitting>] --pins
   <pins> --out-dir <table dir>`; its RESULT line and the table's sha256 are kept.
7. **Copy-back and termination.** The session directories, the pins file and the table are copied back, and a second
   copy goes to a separate medium within 24 h, checked by sha256 (DL-110's rule); then the pod is terminated and the
   termination time recorded.

**The laptop fallback** (disclosed): the plan's §1 form in the KD image by digest, `docker run --network none` with
the clone and the models mounted read-only and the session directory the only writable mount, the same 24 commands,
and `--host-note` naming the Windows build, the power mode (on AC, no other container running) and the hybrid-core
confound (a one-thread process can be scheduled onto a performance or an efficiency core, and host_facts() cannot see
which).

**Expected time.** MEASURED on this lane's cloud VM (a 4-vCPU Xeon @ 2.80 GHz with AVX-512 VNNI), random-weight
students at one thread, 512 × 512, in the B-2 smoke's development run: the FP32 student's latency group took 107 s
(median 329 ms a forward, 320 forwards), the x86 INT8 copies' 355 s and 350 s (E4 and E5; median 1,101 and 1,090 ms),
and the static group's measurement 1 s. INFERRED from these: the teacher's latency group about 10–20 min and its memory
group about 4–8 min; each FP32 student about 4 min and each INT8 stage about 9 min over its three groups; with the 23
gaps of 60 s, about 1.5–2 h for the 24 processes on a comparable pod, plus the transfers. The x86 copies' median above
the FP32 student's on this VM is a synthetic, random-weight observation, not a result.

**LB-0** (LF40; the KD-image acceptance on the laptop before LEFF's last merge; Q10). In the KD image by digest, from a
clone of lane/leff's tip, each alone, with fvcore's real path:
- scripts/smoke_efficiency_record.py: `RESULT: PASS (49/49)` (its memory checks need a writable /proc/self/clear_refs;
  Docker Desktop's kernel decides it, and an environment-caused deferral is read from the smoke's own probe);
- scripts/smoke_latency_threads.py: `RESULT: PASS (97/97)` (its torch_second_interop_call_aborts and
  torch_interop_call_after_work_aborts re-measure torch's abort in the image's build, torch 2.1.0+cu121);
- scripts/smoke_efficiency.py: `RESULT: PASS (82/82)`;
- scripts/smoke_quant_x86_efficiency.py: `RESULT: PASS (103/103)`;
- scripts/smoke_run_ptq.py: `RESULT: PASS (69/69)`;
- scripts/smoke_am23_quant.py: `RESULT: PASS (187/187)`.

**B-3 is not built** (LF39; Q6). The plan line for L-AM16-ARM reads "not built on spare capacity (D4); cut at the freeze
if not complete (AM-22 item 1(g))".

**No TEST-manifest entry** for efficiency (LF41; Q9): no efficiency path reads a TEST byte, and the table's sha256 enters
its DL completion row instead.

### B-2.12 The Phase A departures (A24; recorded here only, leff-b1b.md does not carry them)

From the plan's report (its "SL-1 log", "Other departures", "Refused commands" and "Other" paragraphs) and the
supplement's report (its reads list, SL-1 log and refusals), as A24 rules: all on allowed files, none on a forbidden
path, no DL row.
- **Two `for` loops over named, allowed doc files** (Phase A, 2026-10-16 05:50:09Z over docs/lane_specs/part1.md,
  part2.md, part3.md and errata.md; 05:53:01Z over DECISION_LOG.md, PREREGISTRATION_AMENDMENTS.md and
  EVALUATION_CONTRACT.md), both SL1-B1, both breaking "one file per command"; no loop after them.
- **Command-form departures on the allowed files only:** three `awk` line-range locators on
  PREREGISTRATION_AMENDMENTS.md; two multi-range `sed` commands (DECISION_LOG.md and IMPLEMENTATION_CONTRACT.md);
  `grep` with `-o`/`-i` and pipes; `git log -1 -- <doc>` for commit ids (metadata only).
- **Two refusals, neither retried:** `git diff --name-status 2b6000e… HEAD -- . <X>; git diff --name-only 2b6000e… HEAD |
  wc -l` ([L-PROT G], a diff without a pathspec); `ls -la ~/.claude/ …; tail -n 40 ~/.claude/sl1_guard.log` ([L-PROT S]).
  One `git grep` failed on its option order (a usage error, re-run with the options first).
- **The supplement's session:** one SL1-B2 line (13:38:51Z, `git fetch origin refs/heads/lane/l-corr && … git diff
  --stat …`), logged, not refused; the lane/l-corr and lqp fetches created remote-tracking refs only; the read of the
  session's own transcript under ~/.claude/projects/ after a compaction (the harness's file, not a repository claude/
  file; it could carry no forbidden content, and none was read); a Python AttributeError corrected (not retried); two
  missed Edit calls on its own scratchpad file.

### B-2.13 The SL-1 log, refused commands, slips, pauses, open items and the time

**The SL-1 log** (`tail -n 80 ~/.claude/sl1_guard.log`, run before C7's commit; log mode, so nothing was denied):
18 lines, every line, each cut at 200 characters by the guard. Lines 1–12 are B-1.10's twelve, unchanged,
and are classed there. Lines 13–18 are new:

```
2026-10-27T01:50:33Z	Bash	SL1-B1	cat >> /tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/scratchpad/LANE_STATE.md <<'EOF' - SLIP (01:58Z): `python -B -m py_compile` on the four B-2 files wrote bytecode ca
2026-10-27T03:36:47Z	Bash	SL1-B1	cd /root/.claude/projects/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/subagents/workflows/wf_745f6e8e-b8b/ && for f in agent-a656f2907de8d4db1.jsonl agent-ad089579db8967582.jsonl; 
2026-10-27T06:26:47Z	Bash	SL1-B1	D=/tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/scratchpad/dev_b2; for f in r3b_smoke_full.txt f1_smoke_full.txt; do printf '%s: ' $f; grep -c ': PASS' $D/$f | tr '\n' 
2026-10-27T06:28:50Z	Bash	SL1-B1	D=/tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/scratchpad/dev_b2; cp $D/f1/smoke_latency_threads_mutations.py $D/shadow_f1/scripts/smoke_latency_threads_mutations.py &
2026-10-27T06:30:32Z	Bash	SL1-B1	D=/tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/scratchpad/dev_b2; for f in f1_smoke_full.txt f2_smoke_full.txt; do printf '%s: ' $f; grep -c ': PASS' $D/$f | tr '\n' '
2026-10-27T17:47:18Z	Monitor	SL1-B1	O=/tmp/claude-0/-home-user-plantseg-thesis/d272e01c-e606-5402-a96a-7bccdfa30eca/scratchpad/step10; SUM=$O/de/summary.tsv; seen=0; t=0; while true; do n=$(grep -c -v '^#' "$SUM" 2>/dev/null); n=${n:-0}
```

The classes are the guard's (.claude/hooks/sl1_guard.py:86-93; B-1.10 quotes them). All six new lines are mine and
SL1-B1 ("recursive, globbed or directory listing or search in a protected location"). DL-24 B-2's agents added none.
- 01:50:33: one command appended the py_compile slip to LANE_STATE.md, then ran `ls -la` on src/eval/__pycache__ and
  scripts/__pycache__. That is a listing inside scripts/, the second slip below. It printed the four .pyc files and
  nothing else.
- 03:36:47: a `for` loop over two named agent transcripts of DL-24 B-2's run (under ~/.claude/projects), reading the
  commands they ran.
- 06:26:47 and 06:30:32: `for` loops counting PASS and FAIL lines in two development logs in the scratchpad.
- 06:28:50: a `cp` of the two runner drafts into their development shadows, through a variable scratchpad path.
- 17:47:18 (Monitor): STEP 10 (d)'s first watch command, which L-PROT refused (below). The redone command, with
  literal log paths, was not logged.

Apart from the 01:50:33 listing, none lists or searches the repository, and none names a TEST path.

**Refused commands** (each refused before it ran; none retried in the refused form; times from the session's
transcript):
- [the tool] 01:56Z and 06:03Z: a foreground `sleep` before reading a development log, which the tool blocks. The
  first was redone without the sleep, the second as a Monitor watch.
- [L-PROT S] 05:52Z: an `ls -la` of a scratchpad folder through a variable path. It was redone with the literal path.
- [L-PROT G] 05:59Z: a `git show 9ceab91:$f` inside a `for` loop, a variable path. It was redone with literal paths.
- [L-PROT S] two inline-interpreter edits of scratchpad drafts whose text named git: one in the DL-24 fixes' drafting
  (06:09Z), one in the report's (09:25Z). Both were redone through the Write and Edit tools or a script file.
- [L-PROT S] 17:47Z: STEP 10 (d)'s first watch command (a Monitor) ran `ls` on a variable path with a glob over a
  scratchpad folder, which also breaks the GO's "no shell globs over a directory". It was redone with the two literal
  log paths.

**Slips** (each ran; none touched a forbidden path):
- 01:50Z `python -B -m py_compile` on the four B-2 files. py_compile ignores -B and PYTHONDONTWRITEBYTECODE, so it wrote
  bytecode caches: src/eval/__pycache__/ and scripts/__pycache__/. Both are git-ignored, their names are free of the
  token, and no commit carries them. They were not deleted, under the GO's no-delete rule. Compile checks since then are
  ast.parse.
  - One consequence: the open-audit recording of SHADOW_PATHS missed scripts/efficiency_table.py, whose import was served
    from that cache. It was added by hand, with a comment (B-2.6).
- 01:50Z an `ls -la` of those two folders, in the command that logged the first slip (SL-1 01:50:33): a listing
  inside scripts/, which the GO does not allow. (The STATUS-MD of 07:37Z gave its time as 01:59Z; the transcript says
  01:50:33Z.)
- 06:49Z fix 1's first shadow smoke ran with no timeout set, so the tool's 30-minute background limit stopped it at
  61/91 (0 FAIL). Every later run sets its limit explicitly.
- Two development smokes run at once oversubscribed the 4 vCPUs: each PTQ child ran about 6× slower. One was stopped
  (06:39Z), and every run since is alone.

**Pauses and tooling.**
- The account's session limit stopped the session three times (MEASURED from the transcript):
  - 26 Oct 20:02Z to 23:45Z, during B-1's DL-24 ("resets 10pm (UTC)"; B-1.9), until the user's "Try again";
  - 27 Oct 03:52Z to 05:29Z, during DL-24 B-2's first launch (03:21Z; "resets 4:40am (UTC)"). The worker restarted,
    the user wrote "continue", and the same run resumed from its journal at 05:30Z;
  - 27 Oct 10:13Z to 10:20Z ("resets 10:20am (UTC)"), during STEP 10 (a). The detached batch ran on, and the session
    resumed when the limit reset. No run was affected.
- The stop hook (~/.claude/stop-hook-git-check.sh) asked many times to commit and push uncommitted development files. I
  reported it once and did not comply (the GO's RULES); each commit followed the GO's procedure after its runs.
- The harness's attribution reminder asked for trailers. No commit carries one.

**Open items** (at most five):
1. **DL-a before LB-1.** LB-1 is a draft here (B-2.11). The docs session enters DL-a before the first latency
   measurement of record (LF29's order).
2. **The x86 copies' speed.** On this VM the x86 INT8 copies ran slower than the FP32 student (synthetic, random
   weights; B-2.7). LB-1 measures the models of record on the pod of record.
3. **torch's abort in the image's build.** The abort of a second, or post-work, set_num_interop_threads is MEASURED on
   torch 2.1.0+cpu. LB-0 re-measures it in the KD image (torch 2.1.0+cu121) through the two cases.
4. **The cgroup quota** (DL-24 R3-3, uncertain, fixed) is proven on synthetic /proc and /sys trees. LB-1's pod records
   cgroup_cpu_quota_source beside the quota, for the operator to read against `--host-note`.
5. **provenance.expected_sha256** (DL-24 R2-N6, uncertain). It keeps the teacher meaning B-1 pinned. The docs session
   may rule whether 2.0.0's field list meant H and H2 there. H, H2 and H86 are in provenance.expected either way.

**The time** (LF46; the times MEASURED from the transcript and the commits, the comparison INFERRED):
- From the GO (26 Oct 17:03:44Z) to STEP 10's end (27 Oct 18:37:26Z): 25 h 34 min of wall clock. Without the first two
  limit pauses (3 h 43 min and 1 h 37 min) that is 20 h 13 min of work. The third pause fell inside STEP 10's detached
  batch and is not subtracted. The report (STEP 11) adds about 1 h; C7's own time is in git.
- By part:
  - the start to B-1's checkpoint (C3, 01:32Z): 8 h 29 min, or 4 h 46 min net of the first pause (LF46: 13–15 h).
    STEP 3's batch took 2 h 24 min (17:13Z to 19:37Z); the re-read and the first B-1 and B-2 drafts were written while
    it ran;
  - B-2, C3 to F2 (09:23Z): 7 h 51 min, or 6 h 14 min net of the second pause (LF46: 10–12 h). The code took 1 h 48 min
    to C6 (03:20Z), DL-24 B-2 ran to 06:25Z, and its two fixes, each with its smoke and runner, ran to 09:23Z;
  - STEP 10: 9 h 13 min (LF46: 8–9 h). Q2's harness took 14,179 s (LF46 cites 13,479 s) and E6-KD's 2,614 s
    (2,679 s).
- In all, about 21 h of work against LF46's 32–37 h. The main saving is that the drafts were written while STEP 3's
  batch ran (INFERRED).
