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
