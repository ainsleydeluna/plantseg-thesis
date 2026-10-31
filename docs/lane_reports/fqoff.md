# Lane report: L-FQOFF (AM-4's fake-quant-off score), lane/fqoff

## Phase 1 checkpoint

Phase 1 of L-FQOFF Phase B, under the orchestrator's GO of Mon 26 Oct 2026 (Manila) and its rulings FQR1-FQR16,
FQR7a and FQR15a (LFQOFF-RULE-1), on the plan "L-FQOFF plan (AM-4's fake-quant-off score)" (lfqoff_plan_20261017.md,
F0-F8). Labels: MEASURED, DOCUMENTED, REPOSITORY-PROVEN, INFERRED. Every file:line is at the tip named in its row.
Phase 2 (evaluate_model.py's branch, the end-to-end cases, the full mutation run, DL-24 and the final runs) follows
the merge-in ADDENDUM, after LEFF B-1b merges.

### 1.1 Base, branch and the venv (STEP 1, STEP 2; MEASURED)

- **O** = `0859d3e2356c392fc672805a2d8d821897a4060a`. **OM** (`git rev-parse origin/master` after
  `git fetch origin +refs/heads/master:refs/remotes/origin/master`) = `5838c80e4f68f65ebc6294f1400035c2173b7a7b`.
- The clone was shallow (`git rev-parse --is-shallow-repository`: true); `git fetch --unshallow origin` exited 0.
- `git merge-base --is-ancestor 0859d3e… 5838c80…` printed `O-IN-MASTER`; `git ls-remote --heads origin lane/fqoff`
  printed nothing.
- OM is not O, so (FQR7a) the records for the orchestrator:
  - `git log --first-parent --format='%h %s' O..OM -- …` printed one commit: `5838c80 Docs: records catch-up,
    DL-142 to DL-168 (docs #3 re-based, run and lane records for AM-22 item 1(a)); M-2's supplement`;
  - `git diff --name-only O OM -- src scripts configs ':(exclude,icase)*test*'` printed nothing.
- The branch: the scoped `git status --porcelain=v1` printed nothing; `refs/heads/lane/fqoff` did not exist;
  `git symbolic-ref HEAD` was `refs/heads/claude/nifty-sagan-m48uj1` (the session's starting branch, at OM). The
  chain `git branch lane/fqoff 0859d3e… && git symbolic-ref HEAD refs/heads/lane/fqoff && git checkout --no-overlay
  HEAD -- . ':(exclude)docs/reference/reference.pdf'` exited 0. Post-checks: HEAD `0859d3e…`,
  `refs/heads/lane/fqoff`, the scoped status empty.
- The attachments: lfqoff_plan_20261017.md 19,108 B `64c01093…b39a`; tl_interface_20261016.md 14,352 B
  `3dbd47f6…df563`; tl_interface_lrep_20261016.md 8,607 B `7bb8697a…6c321`: each as the GO gives it.
- **The venv** `$SCR/fqoffvenv`, by docs/lane_reports/q2-qat.md §17.1's recipe: `/usr/bin/python3.11 -m venv`
  (Python 3.11.17); requirements.lock's 23 PyPI pins, each equal to the lock: numpy 1.26.4, scipy 1.11.4,
  pillow 12.3.0, statsmodels 0.14.6, patsy 1.0.2, pandas 3.0.3, python-dateutil 2.9.0.post0, six 1.17.0,
  tzdata 2026.2, packaging 24.2, filelock 3.14.0, typing_extensions 4.15.0, sympy 1.14.0, mpmath 1.3.0,
  networkx 3.6.1, Jinja2 3.1.6, MarkupSafe 3.0.3, fsspec 2026.4.0, requests 2.28.2, certifi 2026.6.17,
  charset-normalizer 3.4.7, idna 3.18, urllib3 1.26.20. Two differ by build: torch 2.1.0+cpu and torchvision
  0.16.0+cpu, installed `--no-deps` from https://download.pytorch.org/whl/cpu, against the lock's +cu121. Two are not
  in the lock: pip 24.0 and setuptools 79.0.1 (ensurepip). Nothing else is installed: no teacher stack (mmcv,
  mmengine, mmsegmentation), no scikit-image. `pip check`: "No broken requirements found."
  torch.backends.quantized.supported_engines: qnnpack, none, onednn, x86, fbgemm; 4 threads; no CUDA.

### 1.2 Commits (MEASURED)

Each commit was made by explicit path (`git add --` for a new file, then `git commit --only -F <message> --
<paths>`), without trailers (`git log -1 --format='%(trailers)'` printed nothing after each), after the scoped
`git diff --stat`, `git status -sb` and `TZ=Asia/Manila date`; each was pushed with `git push origin
refs/heads/lane/fqoff:refs/heads/lane/fqoff`, and `git ls-remote --heads origin lane/fqoff` printed the commit.

| Commit | Files | Subject |
|---|---|---|
| C1a `8b2935665ed924949dfb93b2be5ac72ed3e9a06f` | src/eval/artifacts.py, src/eval/stage_artifacts.py | Add precision qat_fakequant_off and bind its source through the selection of record (L-FQOFF phase 1, FQR1-FQR3) |
| C1b `9f12f13239ed417b603f2ae1116b593a67164802` | src/eval/model_loading.py | Load the fake-quant-off student and build its FP32 reference (L-FQOFF phase 1, FQR9, FQR14) |
| C2a `fea600286ec53351de4e3869505d375b3148812c` | scripts/smoke_eval_qat_fakequant_off.py (new) | Smoke the fake-quant-off binding, loader and FP32 reference on synthetic QAT runs (L-FQOFF phase 1) |
| C3a `11a88a2ce196efa62e650aa870b279481228b5b6` | scripts/smoke_eval_qat_fakequant_off_mutations.py (new), scripts/smoke_eval_qat_fakequant_off.py | Kill every fake-quant-off mutation and refusal through Q2's harness (L-FQOFF phase 1, FQR12) |

C4a adds this file. The phase-1 diff from O at C3a:

```
$ git diff --stat 0859d3e2356c392fc672805a2d8d821897a4060a 11a88a2 -- . \
    ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
 scripts/smoke_eval_qat_fakequant_off.py           | 1283 +++++++++++++++++++++
 scripts/smoke_eval_qat_fakequant_off_mutations.py |  245 ++++
 src/eval/artifacts.py                             |   15 +-
 src/eval/model_loading.py                         |  143 +++
 src/eval/stage_artifacts.py                       |  228 +++-
 5 files changed, 1910 insertions(+), 4 deletions(-)
```

### 1.3 The baseline at O (STEP 3; MEASURED at 0859d3e)

Each smoke ran alone with the venv's `python -B`, `PYTHONDONTWRITEBYTECODE=1`, from the checkout's root, through a
runner that checked before each part that no other process of the venv ran ("other venv pythons: none" at all five
parts) and set PLANTSEG_DATA_ROOT, PYTHONPATH and TMPDIR unset except where a row names its environment. Part A began
at 17:20:02Z and part E ended at 19:45:04Z (26 Oct 2026, UTC): 58 runs, before any edit.

| id | smoke | rc | s | RESULT line (the last line where there is none) |
|---|---|---|---|---|
| a01 | smoke_frozen_blobs | 0 | 0 | RESULT: FROZEN BLOBS OK (10/10) |
| a02 | smoke_run_ptq | 0 | 502 | RESULT: PASS (69/69) |
| a03 | smoke_calibration_lists | 0 | 3 | RESULT: PASS (42/42) |
| a04 | smoke_qnnpack_ops | 0 | 221 | RESULT: PASS (8/8) synthetic artifact |
| a05 | smoke_eval_int8 | 0 | 7 | RESULT: PASS (32/32) |
| a06 | smoke_efficiency | 0 | 5 | RESULT: PASS (82/82) |
| a07 | smoke_quant_e4_e5 | 0 | 4 | RESULT: PASS (58/58) |
| a08 | smoke_quant_e6_e7 | 0 | 4 | RESULT: PASS (37/37) |
| a09 | smoke_quant_runners | 0 | 3 | RESULT: PASS (78/78) |
| a10 | smoke_realrun_decisions | 0 | 2 | RESULT: PASS (76/76) |
| a11 | smoke_qat_runner | 0 | 137 | RESULT: PASS (214/214) |
| a12 | smoke_qat_seeding | 0 | 32 | RESULT: PASS (9/9) |
| a13 | smoke_quant_x86_efficiency | 0 | 3 | RESULT: PASS (103/103) |
| a14 | smoke_eval_determinism | 0 | 26 | RESULT: EVAL-DET OK (74/74) |
| a15 | smoke_am13_upstream | 0 | 19 | RESULT: L-AM13 OK (79/79) |
| a16 | smoke_teacher_d1 | 0 | 113 | RESULT: PASS (145/145) |
| a17 | smoke_arms_kd | 1 | 60 | RESULT: FAIL (58/61) |
| a18 | smoke_qat_selection | 0 | 908 | RESULT: PASS (174/174) |
| a19 | smoke_qat_artifacts | 0 | 3197 | RESULT: PASS (94/94) |
| a17p | smoke_arms_kd (PYTHONPATH=<checkout>) | 0 | 49 | RESULT: PASS (61/61) |
| a20 | smoke_teacher_diag_drift | 0 | 3 | RESULT: PASS (8/8) |
| a21 | smoke_ptq_record | 0 | 794 | RESULT: PASS (203/203) |
| a22 | smoke_eval_stage_artifacts | 0 | 4 | RESULT: PASS (66/66) |
| a23 | smoke_eval_teacher | 0 | 4 | RESULT: PASS (59/59) |
| a24 | smoke_teacher_arms | 2 | 0 | RESULT: NOT RUN (teacher stack missing: mmengine: ModuleNotFoundError; mmseg: ModuleNotFoundError; mmcv: ModuleNotFoundError) |
| b1 | smoke_e6x_condition | 0 | 7 | RESULT: PASS (172/172) |
| b2 | smoke_e6x_condition_mutations | 0 | 9 | RESULT: PASS (428/428) |
| b3 | smoke_e6kd_trigger | 0 | 325 | RESULT: PASS (101/101) |
| b5 | smoke_evaluate_corruptions | 1 | 0 | (no RESULT line) ModuleNotFoundError: No module named 'skimage' |
| b6 | smoke_eval_robustness | 1 | 2 | (no RESULT line) ModuleNotFoundError: No module named 'skimage' |
| c01 | smoke_am13_teacher | 2 | 2 | RESULT: L-AM13 TEACHER NOT RUN (0/0) |
| c02 | smoke_am5_exclusion | 0 | 6 | RESULT: L-AM5 OK (50/50) |
| c03 | smoke_eligibility_variants | 0 | 5 | RESULT: L-AM17-GTPRESENT OK (34/34) |
| c04 | smoke_eval_contract | 0 | 2 | RESULT: A2a OK -- CORE CONTRACT PROVEN |
| c05 | smoke_eval_plantseg | 1 | 3 | RESULT: A2b BLOCKED |
| c06 | smoke_gap_bootstrap_val | 0 | 34 | RESULT: L-AM17B-GAP OK (77/77) |
| c07 | smoke_mde (TMPDIR=/tmp, PYTHONPATH=<checkout>) | 0 | 684 | (no RESULT line) PASS 129/129 (mutations killed 35/35); 682 s |
| c08 | smoke_perclass_gap_table | 0 | 7 | RESULT: L-AM17-PERCLASS OK (38/38) |
| c09 | smoke_plantseg_class_map | 0 | 0 | RESULT: CLASS MAP VALID (offline) |
| c10 | smoke_preflight_e1_trainval | 0 | 6 | RESULT: PASS (125/125) |
| c11 | smoke_split_censuses (TMPDIR=/tmp, PYTHONPATH=<checkout>) | 0 | 2 | (no RESULT line) PASS 81/81 (mutations killed 18/18); 1 s |
| c12 | smoke_stats_driver | 0 | 10 | RESULT: STATS DRIVER CORE OK (55/55) |
| c13 | smoke_stats_integrity | 0 | 49 | RESULT: STATS INTEGRITY OK (129/129) |
| c14 | smoke_stats_paired | 0 | 5 | RESULT: A3a OK -- PAIRED INFERENCE CORE PROVEN |
| c15 | smoke_teacher_calibration | 0 | 16 | RESULT: PASS (52/52) |
| c16 | smoke_teacher_d4 | 0 | 10 | RESULT: PASS (32/32) |
| c17 | smoke_teacher_selection | 2 | 2 | (no RESULT line) ENVIRONMENT: teacher stack not importable (No module named 'mmengine'); run in the plantseg-teacher image. |
| x1 | smoke_am23_quant | 0 | 404 | RESULT: PASS (187/187) |
| n1 | smoke_e6kd_runner | 0 | 143 | RESULT: PASS (215/215) |
| n2 | smoke_e6kd_selection | 0 | 367 | RESULT: PASS (37/37) |
| n3 | smoke_e6kd_invariance | 0 | 33 | RESULT: PASS (11/11) |
| n4 | smoke_e6kd_step | 2 | 2 | (no RESULT line) smoke_e6kd_step.py: error: give run or --selfcheck |
| l1 | smoke_ptq_record_mutations `--check-edits` | 0 | 1 | RESULT: PASS (143/143 checks; 0/0 edits killed) |
| f1 | smoke_am23_mutations | 0 | 314 | RESULT: PASS (311/311) |
| j1 | smoke_probe_e1_grad | 0 | 9 | RESULT: PASS (24/24) |
| j2 | smoke_score_teacher_train | 0 | 8 | RESULT: PASS (13/13) |
| j3 | smoke_teacher_strict_load | 0 | 9 | RESULT: PASS (21/21, 21 skipped) |
| n4s | smoke_e6kd_step `--selfcheck` | 0 | 13 | RESULT: PASS (23/23) |

- **(a)**, B-2.2's STEP 7 column (docs/lane_reports/am3-e6kd.md :818-875; 47 rows): each row gave B-2.2's line
  (c07 and c11 as their last lines' counts, below), except two rows' first runs. Each of those differed for a named
  cause that is not a code difference (the GO's rule for a different count), and its rerun gave B-2.2's line:
  - a17 smoke_arms_kd: FAIL (58/61) with PYTHONPATH unset. Its three failing checks,
    rf_record_teacher_arm_with_arm_teacher and rf_existing_output_gate_record (`arguments:pythonpath_not_checkout`)
    and mut_record_teacher_arm_accepts_arm_teacher (its case fails unmutated), are the three that
    docs/lane_reports/lqp.md :127-130 and :639 record as that smoke's environment requirement (DOCUMENTED). a17p,
    with PYTHONPATH set to the checkout: PASS (61/61).
  - n4 smoke_e6kd_step: exit 2, "give run or --selfcheck": my run list gave it no argument (a slip, 1.10). n4s, with
    `--selfcheck` (docs/lane_reports/am3-e6kd.md :911): PASS (23/23).
- **c07 and c11** (smoke_mde, smoke_split_censuses) ran once each, with TMPDIR=/tmp and PYTHONPATH set to the
  checkout, their documented invocation (docs/lane_reports/am23-quant.md :715). They print no RESULT line; their last
  lines' counts, "PASS 129/129 (mutations killed 35/35)" and "PASS 81/81 (mutations killed 18/18)", are B-2.2's.
- **The environment-caused rows** have B-2.2's causes, and nothing was installed for them: a24 smoke_teacher_arms
  NOT RUN (the teacher stack: mmengine, mmseg, mmcv); b5 smoke_evaluate_corruptions and b6 smoke_eval_robustness
  ModuleNotFoundError skimage; c01 smoke_am13_teacher L-AM13 TEACHER NOT RUN (0/0); c05 smoke_eval_plantseg A2b
  BLOCKED (AdapterError: the split's image directory is not under the default root images/val); c17
  smoke_teacher_selection ENVIRONMENT (no mmengine).
- **The importer scan** (`git grep -l -E '<the GO's pattern>' -- scripts ':(exclude,icase)*test*'`, once): 69 paths,
  42 of them `scripts/smoke_*.py`. All are in (a) or among the environment rows except four: smoke_probe_e1_grad,
  smoke_score_teacher_train and smoke_teacher_strict_load join (a) and phase 2's final runs (the ADDENDUM's A7),
  measured here as j1 PASS (24/24), j2 PASS (13/13) and j3 PASS (21/21, 21 skipped: the F and P cases need MMSeg);
  and smoke_qat_mutations is Q2's harness, which runs in full at phase 2's final runs (A7).
- **The harnesses at baseline:** l1 (`--check-edits`) and f1 only, both B-2.2's lines.

### 1.4 The re-read at O (STEP 4; read-only, before any edit; REPOSITORY-PROVEN at 0859d3e)

Every citation of the plan and FACTS, re-read at O by exact path. "As FACTS say" means the line holds the cited text at
O and the semantics this lane relies on are as FACTS state them.

| Citation | At O | Reading |
|---|---|---|
| PA, AM-4's sentence | docs/PREREGISTRATION_AMENDMENTS.md:119-120 | as FACTS say |
| IC :424 (Supplementary diagnostic (AM-4), "lane L-AM4") and :428 (fusion) | docs/IMPLEMENTATION_CONTRACT.md:424, :428 | as FACTS say |
| EC :403 (`"precision": "fp32 \| int8_ptq \| int8_qat"`) | docs/EVALUATION_CONTRACT.md:403 | as FACTS say |
| PA :2314, :2326-2327 (D; AM-22 item 1(a)4) | docs/PREREGISTRATION_AMENDMENTS.md | as FACTS say |
| DL plan line naming L-FQOFF; DL-137, DL-138 (E6-R1, E6-R2 cut) | docs/DECISION_LOG.md:44-47, :219, :220 | as FACTS say |
| E-66 the last erratum | docs/lane_specs/errata.md:70 (70 lines) | as FACTS say; E-67 is free |
| am3-e6kd.md :1623 (bind E6-KD through qat_selection.json's winner) | docs/lane_reports/am3-e6kd.md:1623 | as FACTS say |
| save_epoch_checkpoint, payload | src/quant/qat.py:913, :929-938 | as FACTS say |
| the freezes; BN_FREEZE_EPOCH, OBS_FREEZE_EPOCH | src/quant/qat.py:1149-1152; :83, :84 | as FACTS say |
| the VAL bracket; fake_quant_val_all_class_miou | src/quant/qat.py:1231-1253; :1274 | as FACTS say |
| flag_summary; fake_quant_modules | src/quant/qat.py:430-439; :284-286 | as FACTS say |
| QAT_CODE_PATHS | src/quant/qat.py:120-135 | as FACTS say: src/eval/__init__.py and metrics.py, none of the four evaluator files |
| the selection record and its winner | src/quant/qat_select.py:531-542, :535-537 | as FACTS say, with one note: the record also holds `excluded_despite_record` (after `excluded_epochs`), which FACTS' field list leaves out. It dates from 1220ebf (2026-10-07, Q2-F), before 2b3b556; this lane never reads it |
| format; RULES_FILES; rules_path_for_stage; _check_summary, run.precision | src/quant/qat_select.py:47; :41-45; :78-82; :293-318, :297 | as FACTS say; load_rules' sha256 is the rules file's raw bytes |
| cmd_finalize; its re-hash; artifact_of_record.checkpoint; the record; qat_selection.sha256; status | scripts/qat_epoch_eval.py:382-504; :430-433; :445-448; :485-494; :487; :499 | as FACTS say |
| FINALIZE_FORMAT; SELECTION_RECORD; FINALIZE_RECORD | scripts/qat_epoch_eval.py:51; :56; :57 | as FACTS say |
| PRECISIONS; STAGES; the precision gate; precision hashed; build_config_payload; canonical_json_bytes | src/eval/artifacts.py:61; :58-59; :299-300; :271; :260-284; :97 | as FACTS say |
| STAGE_ARTIFACTS (QAT stages :47-58); PROJECTION_FREE_STAGES and the comment after it | src/eval/stage_artifacts.py:37-65; :66-67 | as FACTS say |
| StageArtifactError, _fail; _ptq_record_error | src/eval/stage_artifacts.py:79-88; :209-251 | as FACTS say |
| validate_fp32_artifact, its quantization-state refusal; resolve_evaluation_source | src/eval/stage_artifacts.py:255-297, :278-280; :357-374 | as FACTS say |
| the source rules (INT8 CPU-only); the TEST guards; the source validation (INT8 sha); RunMeta; cpu_only; the policy; the load | scripts/evaluate_model.py:151-183 (:168-170); :190-208; :255-287 (:272-273); :300-303; :320; :321; :337-339 | as FACTS say (phase 2 re-reads them after B-1b) |
| CliError; "[code]" text; `--provenance` read defensively; checkpoint_changed_after_validation | scripts/evaluate_model.py:58; :215; :152-154; :344-348 | as FACTS say |
| STAGE_PRECISION; `.tests` imported at module level | src/stats/driver.py:71-72; :46 | as FACTS say (read as text with sed; never imported) |
| the precision subset checks | scripts/smoke_eval_int8.py:102-104; scripts/smoke_stats_driver.py:104-105 | as FACTS say |
| read_run_record; require_complete; verified_checkpoint | src/quant/qat_artifacts.py:94-146; :149-151; :159-172 | as FACTS say |
| prepared_skeleton; alias_keys; load_state_by_name; fake_quant_model | src/quant/qat_artifacts.py:201-204; :207-211; :214-232; :235-241 | as FACTS say |
| convert_epoch, its identity check; qat_checkpoint in the provenance; CONVERT_CODE_PATHS | src/quant/qat_artifacts.py:350-421, :384-388; :398; :54-59 | as FACTS say |
| kind and role; validate_qat_sidecar; load_qat_sidecar | src/quant/x86_latency.py:40-41; :96-118; :121-126 | as FACTS say: validate_qat_sidecar checks the kind, the state and the official/deployment flags, not `artifact_role` (FQR12's code 18 checks the role itself) |
| copy_qat_state_by_name (under `@torch.no_grad()`) | src/quant/x86_latency.py:159-190 | as read: a parameter shape mismatch surfaces as a RuntimeError of `resize_`, not as a QATStop |
| _fused_copy; prepare_qat_model | src/quant/prepare.py:34-45; :89-102 | as FACTS say |
| fuse(): each Conv2dNormActivation's ["0", "1"] (+ "2" when exactly nn.ReLU); each SE's ["fc1", "activation"]; the head's high_proj | src/models/student.py:151-225; :197-198; :201; :214 | as FACTS say (the ReLU rule's expression is :197, applied at :198) |
| rebuild_int8_from_state_dict; load_int8_student; CheckpointError | src/eval/model_loading.py:153-207; :255-289; :44 | as FACTS say (the plan's :153-209 counts the two blank lines after it) |
| EVAL_NUM_WORKERS; determinism_required; torch_num_threads; image_digest; EvalRuntimeError | src/eval/eval_runtime.py:44; :88-91; :290; :235-237, :294; :62-67 | as FACTS say |
| E6-KD's eNN.pt (Q2's schema, stage E6-KD); its save call; E6KD_CODE_PATHS (artifacts.py, model_loading.py); the pin | src/quant/qat_kd.py:33-36; :829-832; :105-124 (:119, :121); :989 | as FACTS say |
| TRIGGER_CODE_PATHS | src/quant/e6kd_trigger.py:106-113 | as FACTS say (all four evaluator files) |
| QUANT_STAGES' E6-KD method | src/quant/stages.py:48 | as FACTS say ("qat_kd") |
| Q2's SHADOW_PATHS and the evaluator files in it; CHECK_LINE; run_smoke; refusal_sites; jobs_of; edited | scripts/smoke_qat_mutations.py:485-555 (:496, :529, :531, :534, :536); :82; :1610-1630; :1637-1674; :1689-1709; :1712-1723 | as FACTS say |
| LQP's GATE and EVAL; LQ63; LQ64 | scripts/smoke_ptq_record_mutations.py:122; :308-313; :314-317 | as FACTS say |
| AM-23's EM row (build_parser's `--stage` line) and ART row (STAGES) | scripts/smoke_am23_mutations.py:306-308; :309-310 | as FACTS say (F rows rewrite a function from inspect.getsource in-process; D rows swap a module table at run time) |
| MK31; E6-KD's SHADOW_PATHS; main's assignment | scripts/smoke_e6kd_mutations.py:236-238; :44-48; :520 | as FACTS say |
| P34 | scripts/smoke_teacher_diag_drift.py:34, :86-87; scripts/smoke_am23_quant.py:83, :1854 | as FACTS say |
| run_qat's pin | scripts/smoke_am23_quant.py:2046 | as FACTS say |
| the three substring checks | scripts/smoke_e6kd_selection.py:475-477; scripts/smoke_eval_stage_artifacts.py:248-250, :271-275 | as FACTS say |
| smoke_qat_selection's import side effects; relabel; fabricate | scripts/smoke_qat_selection.py:39-40; :80; :164-234 | as FACTS say (fabricate ends at :234) |
| evaluator_driver | scripts/synthetic_qat_fixtures.py:138 | as FACTS say |
| (not in FACTS) smoke_eval_stage_artifacts reads stage_artifacts.py's text | scripts/smoke_eval_stage_artifacts.py:276-287 | a constraint on this lane's edits: no `if`/`elif`/`assert`/`raise`/`return` line of stage_artifacts.py names `actual_rows`, and the file holds none of `PlantSegEvalDataset`, `build_eval_loader`, `DataLoader` (comments included) |
| (not in FACTS) smoke_frozen_blobs' pins | scripts/smoke_frozen_blobs.py:29-41 | none of the lane's files is frozen |

No difference on a path this lane relies on: no STOP.

### 1.5 The census at O (STEP 4; MEASURED at 0859d3e)

**(1) Every branch on a precision value.** This printed 149 lines in 46 files:

```
git grep -n -E 'PRECISIONS|"fp32"|"int8_ptq"|"int8_qat"|\.precision\b' -- src scripts configs ':(exclude,icase)*test*'
```

What qat_fakequant_off gets at each site:

| Sites | qat_fakequant_off there | Why |
|---|---|---|
| configs/qat_selection_rules.json:12, configs/qat_selection_rules_am23.json:12, configs/e6kd_selection_rules.json:12 | refused | each rules file's evaluator block requires int8_qat, and _check_summary (src/quant/qat_select.py:297) compares every summary's run.precision with it: no qat_fakequant_off summary enters a selection (never selects; :30 of each names the fake-quant VAL in never_selects) |
| src/eval/artifacts.py:61 (PRECISIONS) | accepted | C1a adds the value |
| src/eval/artifacts.py:271 (build_config_payload) | accepted, hashed | precision is in the config payload, so config_sha256 differs from the int8_qat row's (FQR3) |
| src/eval/artifacts.py:299-300 (the precision gate) | accepted, then constrained | C1a's rule after the gate: a stage in FQOFF_STAGES, model_role student, quant_backend None, random_init false (FQR3) |
| src/eval/artifacts.py:439 (build_summary) | accepted | recorded as summary run.precision |
| src/eval/efficiency.py:267 | never reached | the efficiency record's constant `profiled_precision` "fp32"; no fqoff path profiles anything |
| src/eval/eval_runtime.py:90 (determinism_required) | never applies | true only for an fp32 student on CUDA; this precision is CPU only (FQR11) |
| src/eval/stage_artifacts.py:38-40, :43-49, :52-55, :58, :63 (16 sites, STAGE_ARTIFACTS rows) | never reached as a value | the stage table is unchanged (FQR2): E5, E6, E6-E2 and E6-KD keep precision int8_qat; the fqoff route reads a stage's kind and method, never spec["precision"]; FQR1's case cross-checks FQOFF_STAGES against it |
| src/eval/teacher_diag.py:750 | refused | the teacher-diagnostic reader requires (teacher, teacher, fp32) |
| src/quant/e6kd_trigger.py:354, :357, :429, :492, :494 | refused / never reached | the trigger scores E3 at fp32 and E6 at int8_qat only and compares each summary's run.precision with those (:429); its decision is made (DL-132) |
| src/quant/qat_select.py:297 | refused | as the rules-file row: a selection never reads this precision |
| src/stats/driver.py:70-72 (STAGE_PRECISION; :70 its comment) | never reached | E5 and E6 map to int8_qat; no statistic reads a qat_fakequant_off artifact (read as text, never imported) |
| src/stats/rehearsal.py:159; src/stats/val_artifacts.py:121 | refused | require_role compares the artifact's precision with fp32 (or the caller's value) |
| scripts/evaluate_model.py:51 (SUPPORTED_PRECISIONS), :74 (`--precision` choices), :134, :136 | refused in phase 1 | argparse and validate_cli_args refuse the value until phase 2 adds it (FQR11); phase 1 never edits this file |
| scripts/evaluate_model.py:52 (INT8_PRECISIONS) | never reached | never gains the value (FQR11) |
| scripts/evaluate_model.py:121-122 | refused | the teacher is fp32 only |
| scripts/evaluate_model.py:155, :158, :162, :202, :258 | never reached | the INT8 branches (source rules, TEST guard, source validation) |
| scripts/evaluate_model.py:174 | phase 2 | the non-INT8 branch refuses `--provenance`; phase 2 refuses it for this precision first (fqoff_provenance_forbidden, code 2) |
| scripts/evaluate_model.py:300 (RunMeta), :320 (cpu_only) | phase 2 | RunMeta records the precision with quant_backend None; cpu_only gains the precision (FQR11) |
| scripts/evaluate_model.py:321 | never applies | determinism_required is false for it |
| scripts/smoke_eval_int8.py:24, :88, :103-104, :107-108, :111, :113, :115, :117, :119, :121-122, :126, :129, :131, :135, :138, :141, :144, :218, :226, :232, :238, :244 (25) | never reached | each builds fp32 or INT8 arguments for its own cases; :103-104 is a subset check (INT8_PRECISIONS within SUPPORTED_PRECISIONS, fp32 in it), true with an added value |
| scripts/smoke_stats_driver.py:38, :105 | never reached | a membership check (every STAGE_PRECISION value in PRECISIONS), true with an added value |
| scripts/gap_bootstrap_val.py:138; scripts/mde_entry.py:229 | refused | role checks requiring an fp32 student artifact |
| scripts/profile_efficiency.py:76, :79 | never reached | its two load paths (teacher, FP32 student) return "fp32" |
| scripts/qat_epoch_eval.py:241 | never reached | score's evaluator command for int8_qat |
| scripts/score_teacher_train.py:169; scripts/teacher_d2_calibration.py:157, :168 | never reached | the teacher's fp32 RunMeta and policy |
| scripts/smoke_eval_stage_artifacts.py:85-90, :108 (7); scripts/smoke_e6kd_runner.py:2171; scripts/smoke_e6kd_selection.py:483 | never reached | checks of STAGE_ARTIFACTS rows, which do not change |
| scripts/smoke_teacher_diag_mutations.py:944 | never reached | a mutation anchor in src/eval/teacher_diag.py:750, unchanged |
| scripts/smoke_am13_teacher.py:202; smoke_am13_upstream.py:467, :483, :669; smoke_am23_quant.py:1822, :1823, :1860, :1916; smoke_am5_exclusion.py:169, :297; smoke_e6kd_selection.py:222, :675; smoke_e6kd_trigger.py:489, :583, :585, :1028, :1050, :1180, :1183; smoke_efficiency.py:94, :182, :366; smoke_eligibility_variants.py:132; smoke_eval_contract.py:160, :566; smoke_eval_determinism.py:250, :274, :400-402; smoke_eval_plantseg.py:165, :232, :233, :421; smoke_eval_teacher.py:129, :149, :296; smoke_gap_bootstrap_val.py:232; smoke_perclass_gap_table.py:190; smoke_ptq_record.py:1240; smoke_qat_artifacts.py:1065, :1069; smoke_qat_selection.py:109, :723; smoke_run_ptq.py:344, :363, :428; smoke_stats_integrity.py:745; smoke_stats_paired.py:136; teacher_diag_fixtures.py:377 | never reached | each is a smoke's or fixture's own fp32 or INT8 metadata, arguments, summary or expected value; none compares a whole tuple this lane extends (FQR15: no STOP) |

Count: 3 (configs) + 35 (src) + 16 (evaluate_model.py) + 25 (smoke_eval_int8.py) + 70 (the other scripts) = 149.

**(2) Every source check**: `git grep -n -e 'getsource' -- src scripts ':(exclude,icase)*test*'` printed 15 lines.

| Site | Kind | This lane |
|---|---|---|
| scripts/smoke_teacher_diag_drift.py:4 (docstring), :86, :87 | pin (P34): sha256 of scripts.evaluate_model.run | phase 2 re-pins it (FQR15a); phase 1 never edits run() |
| scripts/smoke_am23_quant.py:1854 | pin (P34) | as above |
| scripts/smoke_am23_quant.py:2046 | pin: sha256 of src.quant.qat.run_qat | never edited (no src/quant file changes) |
| scripts/smoke_e6kd_selection.py:475 | substring: "req.run.stage not in STAGES" in validate_artifact_request | kept: C1a's rule follows the precision gate and leaves that line as it is |
| scripts/smoke_eval_stage_artifacts.py:248 | substring: no "actual_rows" in official_launch_error's code | official_launch_error is not edited |
| scripts/smoke_eval_stage_artifacts.py:271 | substring: artifacts.py's module source holds `d["actual_rows"] != d["expected_rows"]` and `"actual_rows": len(result.rows)` | both lines are untouched |
| scripts/smoke_preflight_e1_trainval.py:323 | substring of scripts/preflight_e1_trainval.py's check_class_weights | not this lane's file |
| scripts/smoke_quant_runners.py:234 | substring of src/quant/qat.py's build_qat_loaders | not this lane's file |
| scripts/smoke_am23_mutations.py:353, :376 | mutation machinery (a function rewritten from its own source) | its F rows target stages, checkpoint, qat, the selection, qat_epoch_eval, the runner, select_qat_epoch and evaluate_model.build_parser (its `--stage` line: phase 2 keeps it once); its D rows swap artifacts.STAGES and stage_artifacts' STAGE_ARTIFACTS, PROJECTION_FREE_STAGES and DESCRIPTIVE_ONLY_STAGES at run time, all of which stay module tables |
| scripts/smoke_e6x_condition_mutations.py:333, :357 | mutation machinery | its owners are scripts/smoke_e6x_condition's C and CLI (src/quant/e6x_condition.py and its entry point): not this lane's files |
| scripts/smoke_teacher_diag_mutations.py:62 | mutation machinery | its targets are teacher_diag, the calibration, fixture and harness modules: not this lane's files |

Beyond getsource, scripts/smoke_eval_stage_artifacts.py:276-287 reads src/eval/stage_artifacts.py's text: no
`if`/`elif`/`assert`/`raise`/`return` line may name `actual_rows`, and `PlantSegEvalDataset`,
`build_eval_loader` and `DataLoader` may not appear. Phase 1's edits keep both.

### 1.6 What phase 1 adds (REPOSITORY-PROVEN at 11a88a2)

- **src/eval/artifacts.py** (FQR1, FQR3): `PRECISIONS = ("fp32", "int8_ptq", "int8_qat", "qat_fakequant_off")` (:63);
  `FQOFF_STAGES = ("E5", "E6", "E6-E2", "E6-KD")` (:66), defined there only (E6-R1 and E6-R2 were cut, DL-137 and
  DL-138); after the precision gate, :306-313, the precision requires a stage in FQOFF_STAGES, model_role student,
  quant_backend None and random_init false (an ArtifactRequestError; no code of its own, FQR3). The precision is in
  the config payload, so the config_sha256 of a qat_fakequant_off row differs from its int8_qat row's (the case
  stages_config_payload_hashes_precision).
- **The validator** (FQR2), src/eval/stage_artifacts.py :327:

  ```
  validate_qat_fakequant_off_source(stage: str, checkpoint: str | Path | None, *, run_dir: str | Path | None,
                                    selection: str | Path | None, selection_sha256: str | None,
                                    telemetry_sha256: str | None, allow_smoke_inputs: bool = False) -> dict
  ```

  It binds `<run>/epoch_ckpts/eNN.pt` through `<eval>/qat_selection.json`, whose sha256 is pinned; it checks a
  `qat_finalize.json` beside the selection whenever one exists and never requires it (E6-KD has none); src/quant is
  imported inside it only. It returns the carrier {spec, checkpoint_path, checkpoint_sha256, epoch, run_id,
  selection_sha256, telemetry_sha256, finalize_sha256 (None when absent)}.
  `resolve_evaluation_source(stage, *, checkpoint=None, provenance=None, expected_sha256=None, qat_run_dir=None,
  qat_selection=None, qat_selection_sha256=None, qat_telemetry_sha256=None, allow_smoke_inputs=False)` (:566) routes
  to it when any qat_* argument is given; a call without them routes as before (the case
  bind_resolve_old_routes_unchanged).
- **The loader** (FQR9), src/eval/model_loading.py :306, `load_qat_fakequant_off_student(fqoff: dict)`: the eNN.pt's
  bytes read once and their sha256 compared with the carrier's (`checkpoint_changed_after_validation`, the FP32
  path's code); `torch.load` of those bytes with weights_only; Q2's `prepared_skeleton()`; `load_state_by_name`
  (exact, the 18 FixedQParams aliases included); `disable_fake_quant` and `disable_observer` applied to every module;
  eval mode; the census (`flag_summary` {observer_enabled [0], fake_quant_enabled [0]}, at least one fake-quant
  module, no module in training mode). Its refusals go through one module-private helper, `_fqoff_refuse` (:300),
  an EvalRuntimeError whose message reads "[code] ...". It writes no model file.
- **The FP32 reference** (FQR14), src/eval/model_loading.py :352, `fp32_reference_from_qat_state(state: dict)` ->
  (model, report): the unfused FP32 student, its keys copied through the fusion groups `fuse()` makes (reached
  through src/quant/prepare.py `_fused_copy`), never by name, and loaded strictly. The loader never calls it.
- **The 17 phase-1 codes, in FQR12's order** (the finder at 11a88a2; each site a `_fail("fqoff_...", ...)` in
  stage_artifacts.py's validate_qat_fakequant_off_source or an `_fqoff_refuse("fqoff_...", ...)` in model_loading.py's
  load_qat_fakequant_off_student; 27 sites, the codes first met in this order):

  6. `fqoff_forbidden_path`: stage_artifacts.py:381
  7. `fqoff_selection_sha256_mismatch`: stage_artifacts.py:388
  8. `fqoff_selection_not_of_record`: stage_artifacts.py:391, :398, :402, :406, :412, :439
  9. `fqoff_stage_mismatch`: stage_artifacts.py:415, :431, :494
  10. `fqoff_run_record`: stage_artifacts.py:422
  11. `fqoff_run_incomplete`: stage_artifacts.py:426
  12. `fqoff_run_not_real`: stage_artifacts.py:429
  13. `fqoff_telemetry_sha256_mismatch`: stage_artifacts.py:435
  14. `fqoff_finalize_disagrees`: stage_artifacts.py:461
  15. `fqoff_not_the_winner`: stage_artifacts.py:466, :472
  16. `fqoff_checkpoint_missing`: stage_artifacts.py:475
  17. `fqoff_checkpoint_sha256_mismatch`: stage_artifacts.py:478
  18. `fqoff_not_qat_train_state`: stage_artifacts.py:483, :488, :491
  19. `fqoff_checkpoint_identity`: stage_artifacts.py:500
  20. `fqoff_state_not_finite`: stage_artifacts.py:503
  21. `fqoff_load_not_exact`: model_loading.py:338
  22. `fqoff_fake_quant_enabled`: model_loading.py:347

- **FQR14's mapping** (MEASURED on the smoke's E5 run, epoch 7, `parity_mapping_accounts_every_key`): the state's
  1593 keys are 314 mapped and 1279 left out, and the 314 targets are the FP32 student's 314 keys, each once.
  Mapped: convbn_convolution 46 (each fused ConvBn's weight; their convolutions have no bias), convbn_bn.weight 46,
  convbn_bn.bias 46, convbn_bn.running_mean 46, convbn_bn.running_var 46, convbn_bn.num_batches_tracked 46 (each to
  the group's BN sibling), same_name 38 (the ConvReLU convolutions' and the other unfused modules' entries). Left
  out: fake_quant_or_observer 1261, fixedqparams_alias 18. Fusion groups: ConvBn2d 34, ConvBnReLU2d 12, ConvReLU2d 8.
- **The parity** (MEASURED, `parity_fp32_reference_logits`, the smoke's `color_task(4, 64, seed=4242)` input on the
  same E5 state): max |logit difference| between the fake-quant-off student and the FP32 reference 8.583e-06, the
  same in all six runs that printed it (five dev runs and STEP 5's s01). Four times that is 3.43e-05, so FQR14's
  rule pins **PARITY_TOL = 1e-4** (the smallest of 1e-6, 1e-5, 1e-4 and 1e-3 at least that; at most 2.5e-4, so no
  1e-3 pin and no STOP). The argmax agrees on 100% of the pixels whose top-two margin exceeds 2 x PARITY_TOL, which
  here is every pixel (share 1.0000). The toggle bites (`parity_toggle_bites`): with fake quantization on (Q2's
  fake_quant_model) the max |logit difference| from the fake-quant-off student is 6.986e-01, above 10 x PARITY_TOL.

### 1.7 The smoke and the harness (phase 1)

- **scripts/smoke_eval_qat_fakequant_off.py**: 85 cases in the sections stages (11), binding (56: the binding's
  positives and a case for each refusal path of codes 6-20, every earlier gate passing), loader (9, with codes 21
  and 22), parity (3), finder (1), never_selects (3) and hygiene (2), on synthetic 15-epoch smoke runs (E5 and
  E6-E2 through make_qat_run; E6-KD through run_qat_kd with a stub teacher) and selections made by Q2's
  select_qat_epoch.py under each stage's committed rules file. `--only NAME,...` and `--cache-dir DIR`; it prints
  Q2's "[CHECKS]" block and ends with `RESULT: PASS (n/n)`. It reads src/stats/driver.py and the rules files as
  text and never imports src.stats; it creates no path whose name holds the token SL-1 forbids (its code-6 cases
  name never-created paths, and hygiene_no_forbidden_named_path_created checks that none appears).
- **scripts/smoke_eval_qat_fakequant_off_mutations.py**: Q2's harness (scripts/smoke_qat_mutations.py) run
  unchanged, as smoke_e6kd_mutations.py runs it, with this lane's SHADOW_PATHS, MUTATIONS, SCOPE and KILLS and its
  own site finder `fqoff_refusal_sites`, assigned over `H.refusal_sites` in its main. SHADOW_PATHS: Q2's 69, the two
  lane files and 19 more that a cold run of the smoke opens (MEASURED with an audit hook on `open`, every imported
  module's file included): configs/qat_selection_rules_am23.json, configs/e6kd_selection_rules.json,
  src/stats/driver.py, configs/distill.py, configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py,
  scripts/e6kd_eval.py, scripts/invariance_harness.py, scripts/preflight_distill.py, scripts/preflight_e1.py,
  scripts/preflight_e1_trainval.py, scripts/run_e6_e2.py, scripts/run_e6_r1.py, scripts/run_e6_r2.py,
  scripts/run_e6kd.py, src/quant/e6kd_trigger.py, src/quant/e6x_condition.py, src/quant/qat_kd.py,
  src/training/teacher_components.py and src/training/train_distill.py: 90 paths, none naming the token. 38 edits:
  the 11 phase-1 mutations (FQ1-FQ7, FQ9, FQ10, FQ12, FQ15) and the 27 refusal sites, each killed by a named check.
- FQR15 (MEASURED after every edit of an evaluator file, and at 11a88a2): every M or MM row's `old` string in
  the five merged harnesses (Q2's, LQP's, E6-KD's, AM-23's, E6-X's) that targets one of the four evaluator files,
  read with ast, occurs exactly once in its file: 15 strings (LQP's LQ55-LQ64, LQ106-LQ108 and LQ117; E6-KD's
  MK31), none otherwise. AM-23's rows on these files are F and D rows: evaluate_model.build_parser's `--stage` line
  (phase 1 leaves the file alone) and the module tables artifacts.STAGES and stage_artifacts' STAGE_ARTIFACTS,
  PROJECTION_FREE_STAGES and DESCRIPTIVE_ONLY_STAGES, still module tables; STEP 5 runs them (M091-M095). No merged
  smoke was edited.

### 1.8 The checkpoint runs on C3a (STEP 5; MEASURED at 11a88a2)

Through the baseline's runner, one after the other, on `11a88a2` (HEAD at the start): part S5 ran 20:17:07Z-20:43:24Z,
"other venv pythons: none" at its start, and no other process of mine ran while it did. The merged smokes' lines
are set beside their STEP 3 lines.

| id | run | rc | s | RESULT line | STEP 3 (at O) |
|---|---|---|---|---|---|
| s01 | smoke_eval_qat_fakequant_off | 0 | 85 | RESULT: PASS (85/85) | new in phase 1 |
| s02 | smoke_eval_qat_fakequant_off_mutations | 0 | 268 | RESULT: PASS (68/68) | new in phase 1 |
| s03 | smoke_eval_stage_artifacts | 0 | 4 | RESULT: PASS (66/66) | a22: the same |
| s04 | smoke_eval_int8 | 0 | 7 | RESULT: PASS (32/32) | a05: the same |
| s05 | smoke_eval_teacher | 0 | 4 | RESULT: PASS (59/59) | a23: the same |
| s06 | smoke_eval_determinism | 0 | 25 | RESULT: EVAL-DET OK (74/74) | a14: the same |
| s07 | smoke_am13_upstream | 0 | 21 | RESULT: L-AM13 OK (79/79) | a15: the same |
| s08 | smoke_eval_contract | 0 | 2 | RESULT: A2a OK -- CORE CONTRACT PROVEN | c04: the same |
| s09 | smoke_ptq_record | 0 | 813 | RESULT: PASS (203/203) | a21: the same |
| s10 | smoke_ptq_record_mutations `--check-edits` | 0 | 1 | RESULT: PASS (143/143 checks; 0/0 edits killed) | l1: the same |
| s11 | smoke_am23_mutations `--only M091,M092` | 0 | 2 | RESULT: PASS (9/9) | f1 ran every row at O: PASS (311/311) |
| s12 | smoke_am23_mutations `--only M093,M094,M095` | 0 | 345 | RESULT: PASS (12/12) | f1 ran every row at O: PASS (311/311) |

- s01, the new smoke, cold (no cache): every phase-1 case PASS, by section stages 11, binding 56, loader 9, parity 3,
  finder 1 (`finder_finds_phase1_sites`: 27 sites of 17 codes; 27 raise forms in the files), never_selects 3 and
  hygiene 2.
- s02, the harness: 38 of 38 edits KILLED and none surviving, the 11 phase-1 mutations and the 27 refusal sites;
  refusal_table_names_only_real_sites PASS (nothing stale), every_job_has_a_verdict PASS (38 jobs),
  self_check_measured PASS (Q2's 26 self-checks, each PASS) and harness_ran_unsignalled PASS; 90 shadow files x 2.
- s11, AM-23's EM and ART rows: M091 (EM.build_parser's `--stage` choices) KILLED by s10_em_choices_are_the_stages
  and s10_em_parses_each; M092 (artifacts.STAGES) KILLED by s10_stages_appended and s10_em_choices_are_the_stages;
  each row's anchor and control PASS. s12, beyond the GO's list because C1a edits stage_artifacts.py: M093-M095,
  its three table rows (STAGE_ARTIFACTS, PROJECTION_FREE_STAGES, DESCRIPTIVE_ONLY_STAGES), each KILLED (3/3).
- MK31's `old` string (scripts/smoke_e6kd_mutations.py :236-238), read with ast, occurs once in
  src/eval/stage_artifacts.py at 11a88a2, as do the other 14 anchors of 1.7.

### 1.9 Disjointness, the rules files and SL-1 (MEASURED at 11a88a2)

- `git diff --name-only 0859d3e2356c392fc672805a2d8d821897a4060a HEAD -- .
  ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'` printed only
  phase-1 files, none in LF2 (scripts/evaluate_model.py; scripts/smoke_eval_teacher.py, smoke_run_ptq.py,
  smoke_ptq_record.py, smoke_teacher_diag_drift.py, smoke_am23_quant.py; scripts/smoke_evaluator_identity.py and
  smoke_evaluator_identity_mutations.py; docs/lane_reports/leff-b1b.md):

  ```
  scripts/smoke_eval_qat_fakequant_off.py
  scripts/smoke_eval_qat_fakequant_off_mutations.py
  src/eval/artifacts.py
  src/eval/model_loading.py
  src/eval/stage_artifacts.py
  ```

- The rules files' sha256 at 11a88a2, each its value at O:

  - configs/qat_selection_rules.json: `76addc57d0367dda9a1163f749ec32c9c240b20736104dc12aa77488344f9568`
  - configs/qat_selection_rules_am23.json: `e299941524968f143359d154566930819cd064152691395fb3647f4de0d98f3d`
  - configs/e6kd_selection_rules.json: `5acad295ffd71ea7630bf10184d0ae0f959a395ed7a27306b23f7dc86c847304`

- SL-1: `~/.claude/sl1_guard.log` is absent (checked at 2026-10-26T20:43:53Z, as at ~18:05Z):
  the guard logged nothing in this session, so there are no lines to give.

### 1.10 Slips and tooling (MEASURED)

- **Slips (mine).**
  - ~18:03Z: one shell glob over a scratch directory (an awk long-line scan of `$SCR/draft/*.py`), against the rule
    against shell globs over a directory; every file was named one by one after it. Nothing in the repository was
    read through it.
  - n4: my baseline run list gave smoke_e6kd_step no argument (exit 2, "give run or --selfcheck"); n4s ran its
    documented `--selfcheck` (1.3).
  - ~20:07Z: an `ls -la` of the session transcript's path through a shell variable. The L-PROT hook refused it (it
    cannot resolve the variable, so it treats the command as a listing that may name docs/reference/reference.pdf);
    the read was redone as a grep of the literal path, with no listing.
  - C2a's `c09_selection_stage` did not isolate refusal site #0 of fqoff_stage_mismatch (the selection's stage): it
    requested E6 against E5's binding, so with that site removed the run_meta check (#1) refused with the same code.
    The harness's first dev run, before C3a (20:04:51Z-20:09:22Z, RESULT: FAIL (67/68), 37 of 38 edits killed),
    found it. C3a's case forges the selection's stage alone (E6, whose rules file is E5's) and keeps the request
    check; the second dev run (20:11:47Z-20:16:24Z) and STEP 5's s02 each gave RESULT: PASS (68/68), all 38 killed.
  - Scratch only: my lane state file miscounted the baseline as 59 runs (58; its summary has a header line) and
    noted two times wrongly; both were corrected there, and this report gives the runner's records.
- **Tooling, not followed** (AGENTS.md, "What counts as approval"): the stop hook
  `~/.claude/stop-hook-git-check.sh` asked, at ~19:48Z and once more, to commit and push the uncommitted changes. I
  reported it once and did not act on it; the commits and pushes followed the GO's order, after its checks. The
  session's attribution reminder (Co-Authored-By and Claude-Session trailers, and the designated branch
  claude/nifty-sagan-m48uj1) was not followed either: no commit has a trailer, and every push went to lane/fqoff.
- **Refused by a hook:** the one L-PROT refusal above; no other command was refused.

### 1.11 Open items

1. **Phase 2**, after LEFF B-1b merges and the merge-in ADDENDUM: scripts/evaluate_model.py's branch (codes 1-5),
   P34's re-pin (FQR15a), the end-to-end cases, the remaining FQ rows, the full mutation runs (A7) and DL-24.
2. **PARITY_TOL = 1e-4 rests on one synthetic measurement** (8.583e-06, the same in all six runs that printed it)
   of the smoke's E5 student; it binds the smoke's parity case only.
3. **The selection record's `excluded_despite_record`** field (since 1220ebf) is not in FACTS' field list. The lane
   never reads it (1.4).
4. **E6-KD has no qat_finalize.json**, so its binding rests on its selection and its run alone (FQR2: checked when
   present, never required); bind_e6kd_rules_e6kd covers it.
5. **SHADOW_PATHS were measured on phase 1's smoke.** Phase 2's cases (evaluate_model.py's path end to end) may open
   more files, so phase 2 measures them again.

## Phase 2

Phase 2 of L-FQOFF Phase B, under ADDENDUM 1 to the GO v2 (received 27 Oct 2026, ~11:36Z; FQR7a), on LEFF B-1b's
merge. The GO v2's rulings stand as the ADDENDUM amends them. Labels as in phase 1. Every file:line is at the final
code commit 759440f (`759440ffa8ef0208832a6734d8b6b79962b8cd6f`) unless its row names another commit.

### 2.1 The move to the merge-in (A1; MEASURED)

- `git fetch origin +refs/heads/master:refs/remotes/origin/master
  +refs/heads/lane/fqoff:refs/remotes/origin/lane/fqoff` exited 0 (master 5838c80..879cab5, lane/fqoff
  5c6a8aa..d211f08). **OM2** = `879cab53770e4a761f5d3b98ce1b222158a344d7`. The environment had not been reset:
  refs/heads/lane/fqoff stood at 5c6a8aa and `$SCR/fqoffvenv` was in place, so neither TAKEOVER branch applied.
- The pre-checks printed what A1 expects: origin/lane/fqoff `d211f083cc52812665b443645ff95a46facba7f7`; its parents
  `5c6a8aadcf70f0cf6e004907ed73926f1973a2e0` and `879cab53770e4a761f5d3b98ce1b222158a344d7`; HEAD `5c6a8aa…`;
  `refs/heads/lane/fqoff`; `O2-IN-MASTER`; the scoped `git status --porcelain=v1` empty.
- The chain `git update-ref refs/heads/lane/fqoff d211f08… 5c6a8aa… && git checkout --no-overlay HEAD -- .
  ':(exclude)docs/reference/reference.pdf'` exited 0. Post-checks: HEAD `d211f083…`, the scoped status empty.
- **The merge-in brings master and nothing else.** `git diff --name-only 5c6a8aa d211f08` and `git diff
  --name-only 0859d3e 879cab5`, each with the scoped pathspec, printed the same 10 lines: docs/DECISION_LOG.md,
  docs/lane_reports/leff-b1b.md, reports/kd_f_s42_alpha25_80k_a1_host_fault.md, scripts/evaluate_model.py,
  scripts/smoke_am23_quant.py, scripts/smoke_eval_teacher.py, scripts/smoke_evaluator_identity.py,
  scripts/smoke_ptq_record.py, scripts/smoke_run_ptq.py and scripts/smoke_teacher_diag_drift.py. `git diff
  --name-only 879cab5 d211f08` printed phase 1's six files: docs/lane_reports/fqoff.md, the two lane smokes,
  src/eval/artifacts.py, src/eval/model_loading.py and src/eval/stage_artifacts.py.
- **The flags as merged** (`git show d211f08:scripts/evaluate_model.py | grep -n …`):
  `"--expected-checkpoint-sha256"` :94, `"--expected-converted-sha256"` :99 and `"--threads"` :103. Both identity
  flags exist: no STOP.
- `git cat-file -e d211f08:docs/lane_reports/leff-b1b.md && echo B1B-REPORT` printed `B1B-REPORT`. Master's change
  holds no scripts/smoke_evaluator_identity_mutations.py: smoke_evaluator_identity runs its own 21 mutations, so A2
  and A7 (h) have no separate runner.

### 2.2 Commits (MEASURED)

As in phase 1 (1.2): each commit by explicit path (`git commit --only -F <message> -- <paths>`), without trailers
(`git log -1 --format='%(trailers)'` printed nothing after each), after the scoped `git diff --stat`, `git status
-sb` and `TZ=Asia/Manila date`; each pushed with `git push origin refs/heads/lane/fqoff:refs/heads/lane/fqoff`, with
`git ls-remote --heads origin lane/fqoff` printing the commit.

| Commit | Files | Subject |
|---|---|---|
| C1c `b43ab55e08733a3437be44f1ece44e3e7f604b35` | scripts/evaluate_model.py | Score precision qat_fakequant_off in scripts/evaluate_model.py (L-FQOFF phase 2, FQR11, FQR13) |
| C1d `13d2e1bff10bf2710a88c8a96314b4ec51bb2873` | scripts/smoke_am23_quant.py, scripts/smoke_teacher_diag_drift.py | Re-pin P34 to run()'s source after L-FQOFF's hunks (L-FQOFF phase 2, FQR15a) |
| C2b `851f4c5183d49d39cdf0bc92ad3bba3f0c321fc2` | scripts/smoke_eval_qat_fakequant_off.py | Smoke the fake-quant-off CLI and run end to end (L-FQOFF phase 2, FQR11-FQR13) |
| C3b `521e3df29970b828420f4dcf6159d57b0eeed3da` | scripts/smoke_eval_qat_fakequant_off_mutations.py | Kill the phase-2 mutations and the CLI's refusals through Q2's harness (L-FQOFF phase 2, FQR12) |
| C5a `43a074050ca7263dc0c4bce57d575ba045e0327f` | src/eval/stage_artifacts.py, scripts/smoke_eval_qat_fakequant_off.py, scripts/smoke_eval_qat_fakequant_off_mutations.py | Read the eNN.pt once and refuse an unreadable finalize record as code 14 (L-FQOFF phase 2, DL-24) |
| C5b `c256d1bc3cb1048426132e6a4af60c756957c9e4` | scripts/smoke_eval_qat_fakequant_off.py, scripts/smoke_eval_qat_fakequant_off_mutations.py | Remove the fake-quant-off smoke's temp directory when it ends (L-FQOFF phase 2, DL-24) |
| C5c `0c50388a782efe163b94c3310377bc5bc5c46460` | scripts/evaluate_model.py, scripts/smoke_eval_qat_fakequant_off.py, scripts/smoke_eval_qat_fakequant_off_mutations.py | Re-hash the eNN.pt at P2 for qat_fakequant_off (L-FQOFF phase 2, DL-24 R3-F1, LFQOFF-RULE-3) |
| C5d `759440ffa8ef0208832a6734d8b6b79962b8cd6f` | scripts/smoke_am23_quant.py, scripts/smoke_teacher_diag_drift.py | Re-pin P34 to run()'s source after the P2 re-hash (L-FQOFF phase 2, FQR15a, LFQOFF-RULE-3) |

C1c, C1d, C2b and C3b were pushed in A4, each after its dev checks (27 Oct). C5a and C5b were held at A5's STOP
and pushed together at 04:25Z on 28 Oct under LFQOFF-RULE-3 Q3 (`521e3df..c256d1b`); C5c (`c256d1b..0c50388`) and
C5d (`0c50388..759440f`) were pushed after their dev checks (2.7), the remote tip printed after each push.

C4b adds this section. The lane's whole change over master, at the final code commit:

```
$ git diff --stat 879cab53770e4a761f5d3b98ce1b222158a344d7 759440f -- . \
    ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
 docs/lane_reports/fqoff.md                        |  475 ++++++
 scripts/evaluate_model.py                         |  112 +-
 scripts/smoke_am23_quant.py                       |    2 +-
 scripts/smoke_eval_qat_fakequant_off.py           | 1870 +++++++++++++++++++++
 scripts/smoke_eval_qat_fakequant_off_mutations.py |  300 ++++
 scripts/smoke_teacher_diag_drift.py               |    3 +-
 src/eval/artifacts.py                             |   15 +-
 src/eval/model_loading.py                         |  143 ++
 src/eval/stage_artifacts.py                       |  241 ++-
 9 files changed, 3149 insertions(+), 12 deletions(-)
```

### 2.3 The new baseline at d211f08 (A2; MEASURED)

Each run alone through phase 1's runner (stdin /dev/null, PLANTSEG_DATA_ROOT, PLANTSEG_IMAGE_DIGEST, PYTHONPATH and
TMPDIR unset; "other venv pythons: none" at its start), at HEAD `d211f083…`, from 11:41:30Z to 12:27:04Z
(27 Oct 2026, UTC). Every line is its expected line, so these are the new baseline.

| id | run | rc | s | RESULT line | Expected (source) |
|---|---|---|---|---|---|
| b01 | smoke_eval_teacher | 0 | 12 | RESULT: PASS (59/59) | B-1b's final line (leff-b1b.md) |
| b02 | smoke_run_ptq | 0 | 493 | RESULT: PASS (69/69) | B-1b's |
| b03 | smoke_ptq_record | 0 | 888 | RESULT: PASS (203/203) | B-1b's |
| b04 | smoke_teacher_diag_drift | 0 | 3 | RESULT: PASS (8/8) | B-1b's (its pin d41964f5…) |
| b05 | smoke_am23_quant | 0 | 491 | RESULT: PASS (187/187) | B-1b's (its pin d41964f5…) |
| b06 | smoke_evaluator_identity | 0 | 44 | RESULT: PASS (38/38; 21/21 edits killed) | B-1b's |
| b07 | smoke_ptq_record_mutations `--check-edits` | 0 | 1 | RESULT: PASS (143/143 checks; 0/0 edits killed) | l1 (STEP 3) |
| b08 | smoke_am23_mutations | 0 | 367 | RESULT: PASS (311/311) | f1 (STEP 3) |
| b09 | smoke_eval_qat_fakequant_off | 0 | 93 | RESULT: PASS (85/85) | s01 (STEP 5) |
| b10 | smoke_eval_qat_fakequant_off_mutations | 0 | 276 | RESULT: PASS (68/68) | s02 (STEP 5) |
| b11 | smoke_eval_stage_artifacts | 0 | 3 | RESULT: PASS (66/66) | s03 |
| b12 | smoke_eval_int8 | 0 | 7 | RESULT: PASS (32/32) | s04 |
| b13 | smoke_eval_determinism | 0 | 26 | RESULT: EVAL-DET OK (74/74) | s06 |
| b14 | smoke_am13_upstream | 0 | 28 | RESULT: L-AM13 OK (79/79) | s07 |
| b15 | smoke_eval_contract | 0 | 2 | RESULT: A2a OK -- CORE CONTRACT PROVEN | s08 |
| b16 | smoke_am23_mutations `--only M091,M092` | 0 | 2 | RESULT: PASS (9/9) | s11 |

B-1b's six smokes give B-1b's final lines; s05, s09 and s10 of STEP 5 are b01, b03 and b07 here. A2 found no
difference, so nothing in it needed an environment cause.

### 2.4 The re-read at d211f08 (A3; read-only, before any edit; REPOSITORY-PROVEN at d211f08)

**Master's change to the four files** (`git diff 0859d3e 879cab5 -- <file>`, one file per command):

| File | Master's change | Reading |
|---|---|---|
| src/eval/artifacts.py | none (empty diff) | — |
| src/eval/stage_artifacts.py | none | — |
| src/eval/model_loading.py | none | — |
| scripts/evaluate_model.py | 8 hunks, 150 insertions, no deletion (LEFF B-1b, IF-T1) | read in full, below |

B-1b's evaluate_model.py, hunk by hunk (line numbers at d211f08):

1. The module docstring gains IF-T1's paragraph (the three flags; required under R, optional elsewhere, inert when
   absent).
2. Three parser flags after `--teacher-ckpt-sha256`: `--expected-checkpoint-sha256` (:94), `--expected-converted-sha256`
   (:99) and `--threads` (:103), each default None.
3. validate_cli_args gains IF-T1's block after every earlier guard: `in_r = args.split == "test" and
   args.artifact_status == "official"`; H refused with `--random-init`, format-checked, required under R and equal to
   `--teacher-ckpt-sha256` for the teacher; H2 refused off INT8 or with `--random-init`
   (`[expected_converted_sha256_not_int8]`), format-checked and required under R for INT8; T required under R
   (`[threads_required]`). Then five new functions: parse_threads, apply_threads, handed_file (`--provenance` for an
   INT8 stage, else `--checkpoint`: "The one place that names it"), identity_sha256 and hash_before_parse (P1).
4. run(): `threads = apply_threads(args)` right after `validate_cli_args(args)`, before the class map, the source
   checks and any build.
5. run(): `handed = hash_before_parse(args)` (P1) after `ckpt_path = ckpt_sha = None`, before the source branches.
6. run(): after the branches, the P2 block (`after = ckpt_sha if resolved is None else identity_sha256(handed)`,
   refused as `[source_changed_during_validation]`) and H2's comparison with `resolved["artifact_sha256"]`.
7. run(): P3 in the INT8 load branch (`converted_changed_before_load`).
8. run(): the drift check before write_artifact (`threads_drifted`).

No line of the source rules, the TEST guards, the source validation, RunMeta, cpu_only, the determinism policy or the
load dispatch changed, and no path this lane relies on outside the file changed (no src/quant file, nor
scripts/qat_epoch_eval.py or scripts/select_qat_epoch.py): no STOP. smoke_evaluator_identity's 21 mutations anchor lines
of evaluate_model.py, and its cli_block() slices validate_cli_args from B-1b's IF-T1 comment to `"\n\n\ndef
parse_threads"`; so this lane's CLI rules sit in a helper called early in validate_cli_args
(scripts/evaluate_model.py:205), never between B-1b's block and parse_threads. It compares no whole precision tuple
(FQR15: no STOP).

**The flag table** (how qat_fakequant_off meets B-1b's three flags; file:line at the final commit):

| Flag | What qat_fakequant_off does with it | file:line | Case |
|---|---|---|---|
| H `--expected-checkpoint-sha256` (P1) | B-1b's handed_file gains this precision's case: the handed file is `--checkpoint`, the eNN.pt. hash_before_parse (P1) hashes it before any source branch, so the validator opens its binding files only after that, and none before code 6 | scripts/evaluate_model.py:391-392 (the case), :453 (P1), :470 (the source branch) | run_h_mismatch_refused_before_parse (FQ14a); run_h_matching_accepted |
| H (P2), as LFQOFF-RULE-3 Q1 amends A3 (2.8) | the P2 block compares the carrier's `fqoff["checkpoint_sha256"]`, the sha256 of the bytes the validator read once and judged (C5a), with H; then it re-hashes the handed file and compares that with H, as B-1b does for a hash recorded before validation ends (C5c). Either mismatch is B-1b's `[source_changed_during_validation]`, before any dataset or model exists | scripts/evaluate_model.py:511-513 | run_h_changed_during_validation (FQ14b: both comparisons skipped); run_h_changed_after_validation (FQ19: the re-hash skipped) |
| H under R | required, by B-1b's rule, unchanged | scripts/evaluate_model.py:332 | cli_guards_of_record_kept |
| H2 `--expected-converted-sha256` | does not apply (not INT8): B-1b's `[expected_converted_sha256_not_int8]` refuses it, unchanged | scripts/evaluate_model.py:341 | cli_h2_refused |
| T `--threads` | set by B-1b's apply_threads right after validate_cli_args, before any precision branch (so no FQ13 row: FQ13 and FQ14 apply only where the fqoff path applies a flag itself); checked for drift before the write; required under R; all unchanged | scripts/evaluate_model.py:437 (set), :610 (drift), :354 (R) | run_threads_recorded (run.eval_runtime.torch_num_threads = T in this precision's summary); cli_guards_of_record_kept |

**The census at d211f08** (MEASURED; the GO's two greps):

- **Precision branches**: 224 lines in 49 files (149 in 46 at O). The difference is master's and phase 1's:
  - scripts/evaluate_model.py 16 → 20 lines: B-1b's :270 and :272 (H2's off-INT8 refusal), :279 (H2 required for
    INT8) and :320 (handed_file). qat_fakequant_off is refused at :270 (H2 off INT8, case cli_h2_refused), never
    reaches :279 (INT8 only) and reaches :320's replacement through its own case of handed_file (above);
  - scripts/smoke_evaluator_identity.py 0 → 60 lines (new): B-1b's smoke, its own fp32, INT8 and teacher fixtures and
    its mutations M15 and M21; it compares no whole tuple this lane extends (FQR15: no STOP);
  - phase 1's own: src/eval/artifacts.py 5 → 6 (:307, the precision's rule), the lane smoke 9 and the harness 1.
  - Every other site reads as at O (1.5): no new branch on a precision value.
- **Source checks** (`getsource`): 15 lines, the same sites as at O, smoke_teacher_diag_drift's at :4, :87 and :88
  (B-1b's docstring line moved them by one). P34's two pins held run()'s source sha256 as B-1b set it,
  `d41964f5ebf692526d75399a05e888d68632b43cb4da760718ca6e3e17ad783a` (smoke_teacher_diag_drift.py:35 "# at
  ea22241"; smoke_am23_quant.py:83), equal to the value measured at d211f08. No pin of another function this phase
  edits exists: no STOP.
- At the final commit the precision grep prints 241 lines in 49 files (224 in 49 at d211f08). The difference is this
  phase's own lines: scripts/evaluate_model.py 20 → 26 (validate_qat_fakequant_off_cli's precision test :155 and its
  other-precision message :191, handed_file's case :391, the source branch :470, cpu_only :555 and the load dispatch
  :572), the lane smoke 9 → 19 (its cli and run cases) and the harness 1 → 2 (FQ11's anchor, :110).

### 2.5 What phase 2 adds (REPOSITORY-PROVEN at 759440f)

**scripts/evaluate_model.py** (C1c; FQR11, FQR13, the flag table):

- `SUPPORTED_PRECISIONS = ("fp32", "int8_ptq", "int8_qat", "qat_fakequant_off")` (:63) and the
  `--precision` choices (:88) gain the precision; `INT8_PRECISIONS` (:64) does not.
  `QAT_FLAGS` (:66) names the four binding flags.
- **The flags** follow `--provenance` (:115), each default None: `--qat-run-dir`
  (:117), `--qat-selection` (:120), `--qat-selection-sha256` (:122)
  and `--qat-telemetry-sha256` (:125). `--checkpoint` names the eNN.pt; there is no finalize
  flag (FQR2). Every read is `getattr(args, …, None)`, so an args object without them validates as before
  (cli_callers_without_the_flags).
- **The CLI rules**: `validate_qat_fakequant_off_cli(args)` (:150-191), pure, called in
  validate_cli_args right after the role check (:205), before the teacher, precision, source, TEST and
  IF-T1 guards. Its refusals go through one module-private helper, `_fqoff_cli(code, message)`
  (:144-147), which raises the file's CliError with "[code] …" in its message. For this precision, in
  order: the stage in FQOFF_STAGES and the role student (code 1); no `--provenance` (2); `--checkpoint` and the four
  flags given, no `--random-init`, each sha256 flag 64 lowercase hex (3); `--device cpu` (4); `--protocol canvas`
  (5, FQR13). For every other precision a `--qat-*` flag is code 3. The TEST guards keep their substance for this
  precision (the confirm flag, no cap, not smoke, `--checkpoint`; :284-300), and the source is
  `--checkpoint` plus the four flags.
- **run()**, with one carrier:
  - `fqoff = None` (:452) beside `resolved`, `quant_backend`, `ckpt_path` and `ckpt_sha`;
  - the precision's own source branch (:470), after the INT8 branch and before the teacher's and FP32's:
    `fqoff = validate_qat_fakequant_off_source(args.stage, args.checkpoint, run_dir=…, selection=…,
    selection_sha256=…, telemetry_sha256=…, allow_smoke_inputs=args.artifact_status == "smoke")`
    (:475-480); `ckpt_path = str(fqoff["checkpoint_path"])` and
    `ckpt_sha = fqoff["checkpoint_sha256"]` (:481, :482); quant_backend and resolved stay
    None, so the INT8 load can never take this source;
  - one stdout line after validation (:483): `fqoff binding: stage <S>, run <run_id>, epoch <NN>,
    checkpoint <sha256>, selection <sha256>, telemetry <sha256>, finalize <sha256 | absent>` (run_binding_line,
    run_finalize_line);
  - B-1b's P2 block gains this precision's case (:511-513; C1c, then C5c under LFQOFF-RULE-3
    Q1): with H given, the carrier's checkpoint_sha256 must equal H, and then the handed file, re-hashed with B-1b's
    identity_sha256, must equal H too; either mismatch is `[source_changed_during_validation]`, raised before the
    adapter, the dataset or a model exists (run_h_changed_during_validation, run_h_changed_after_validation);
  - RunMeta (:534) records precision qat_fakequant_off, quant_backend None, the eNN.pt and its sha256;
  - `cpu_only` gains the precision (:554-555); the determinism policy is unchanged
    (:556: an FP32 student on CUDA only), so it applies nothing here;
  - the load dispatch tests this precision after the teacher and before `elif resolved is not None`
    (:572-574) and calls `load_qat_fakequant_off_student(fqoff)`; ck_info stays None.
- The module docstring's list of construction paths gains the precision (:18).

**P34 and D2** (FQR15a). scripts/teacher_d2_calibration.py mirrors "run()'s teacher path, call for call and in the
same order" (its docstring, :5), in :105-258 (RunMeta :157, validate_artifact_request :165, the policy :168, the
adapter :169, its own load :172, `resolve_model_device("cpu", cpu_only=True)` :182, write_artifact :258; at
759440f). run()'s five hunks add, remove or reorder no call on that path and change none of its arguments:
`fqoff = None` binds a name the teacher path never reads; the source branch's test (precision qat_fakequant_off) is
evaluated and false for the teacher (precision fp32), and the P2 block's `fqoff is not None` is false wherever it is
reached (only with H, which D2 never passes; C5c's lines sit inside it); cpu_only's added
disjunct is never evaluated for the teacher, whose first disjunct, `args.model_role == "teacher"`, is true, so
cpu_only stays True, the value D2 passes; and the load dispatch's branch comes after the teacher's. So the hunks lie
outside the part D2 mirrors, as the docstring line says (DL-24 R4-4 and its verifier, 2.8).

| | smoke_teacher_diag_drift.py | smoke_am23_quant.py |
|---|---|---|
| before (B-1b, at d211f08) | :35 `d41964f5ebf692526d75399a05e888d68632b43cb4da760718ca6e3e17ad783a` "# at ea22241" | :83 the same value, "# P34 (teacher_diag_drift)" |
| C1d `13d2e1b` (after C1c's hunks) | :36 `c3460683307f4e59d58ac0379fe682a532c93f33179f2b3c2583b8f50c0a0bfb` "# at b43ab55" | :83 the same value, the comment kept |
| C5d `759440f` (after C5c's P2 hunk; final) | :36 `07e9da16fb7d81c7c3494f120082b6c4fd257862a86d1785b3aba80bef3fb9af` "# at 0c50388" | :83 the same value, the comment kept |

Each value is `sha256(inspect.getsource(scripts.evaluate_model.run).encode("utf-8"))` at the commit named, computed as
the two smokes compute it (MEASURED; at 0c50388 run() spans scripts/evaluate_model.py:420-615, 196 lines). C1d's
docstring line (scripts/smoke_teacher_diag_drift.py:11, "Re-pinned by lane L-FQOFF after its run()
hunks, outside the part D2 mirrors") covers C5c's hunk too, so C5d changes the two values and the "# at" comment
only. At C5c, before the re-pin, smoke_teacher_diag_drift gave `RESULT: FAIL (7/8)`, P34 alone failing with the
value above (by design, as at C1c); at C5d both smokes keep their counts and every case name (8/8, 187/187; D5 and
A7 (a), 2.7).

**The 22 refusal codes, in FQR12's order** (the finder at 759440f: 22 codes, 37 sites; each site a call of its
file's raise form with the literal code first: `_fqoff_cli` in scripts/evaluate_model.py, `_fail` in
src/eval/stage_artifacts.py, `_fqoff_refuse` in src/eval/model_loading.py, whose error is EvalRuntimeError):

1. `fqoff_stage_not_in_scope`: scripts/evaluate_model.py:158, :161
2. `fqoff_provenance_forbidden`: scripts/evaluate_model.py:164
3. `fqoff_source_flags`: scripts/evaluate_model.py:167, :171, :173, :179, :190
4. `fqoff_not_cpu`: scripts/evaluate_model.py:181
5. `fqoff_protocol`: scripts/evaluate_model.py:185
6. `fqoff_forbidden_path`: src/eval/stage_artifacts.py:387
7. `fqoff_selection_sha256_mismatch`: src/eval/stage_artifacts.py:394
8. `fqoff_selection_not_of_record`: src/eval/stage_artifacts.py:397, :404, :408, :412, :418, :445
9. `fqoff_stage_mismatch`: src/eval/stage_artifacts.py:421, :437, :507
10. `fqoff_run_record`: src/eval/stage_artifacts.py:428
11. `fqoff_run_incomplete`: src/eval/stage_artifacts.py:432
12. `fqoff_run_not_real`: src/eval/stage_artifacts.py:435
13. `fqoff_telemetry_sha256_mismatch`: src/eval/stage_artifacts.py:441
14. `fqoff_finalize_disagrees`: src/eval/stage_artifacts.py:470
15. `fqoff_not_the_winner`: src/eval/stage_artifacts.py:475, :481
16. `fqoff_checkpoint_missing`: src/eval/stage_artifacts.py:484
17. `fqoff_checkpoint_sha256_mismatch`: src/eval/stage_artifacts.py:490
18. `fqoff_not_qat_train_state`: src/eval/stage_artifacts.py:495, :501, :504
19. `fqoff_checkpoint_identity`: src/eval/stage_artifacts.py:513
20. `fqoff_state_not_finite`: src/eval/stage_artifacts.py:516
21. `fqoff_load_not_exact`: src/eval/model_loading.py:338
22. `fqoff_fake_quant_enabled`: src/eval/model_loading.py:347

**The summary's fields for the precision** (FQR3; the cases summary_precision_and_backend, summary_checkpoint_fields,
summary_runtime_record, summary_config_sha256, run_threads_recorded): run.stage S; run.model_role student;
run.precision `qat_fakequant_off`; run.quant_backend null; run.checkpoint_path the eNN.pt, `<run>/epoch_ckpts/eNN.pt`;
run.checkpoint_sha256 its sha256 (the selection's winner.checkpoint_sha256); run.random_init false; run.env.device
cpu; run.config_sha256 = sha256(canonical_json_bytes(build_config_payload(request))), precision in the payload
(src/eval/artifacts.py:276), so it differs from the same request's at int8_qat;
run.eval_runtime with model_device cpu, determinism_policy_applied false, checkpoint_iteration and
checkpoint_best_val_miou_all_class null (ck_info None), torch_num_threads (T when `--threads` is given) and the image
digest as for every artifact; dataset.condition clean; dataset.preprocess_protocol `core_preprocess/1.0.0` (canvas).
A selection's own summary check refuses such a summary (summary_never_selected).

**FQR14** (from phase 1, 1.6; MEASURED on the smoke's E5 run, epoch 7): the state's 1593 keys are 314 mapped and 1279
left out (convbn_convolution 46, convbn_bn.weight 46, convbn_bn.bias 46, convbn_bn.running_mean 46,
convbn_bn.running_var 46, convbn_bn.num_batches_tracked 46, same_name 38; left out fake_quant_or_observer 1261 and
fixedqparams_alias 18), the 314 targets the FP32 student's 314 keys, each once. The parity: max |logit difference|
8.583e-06, so **PARITY_TOL = 1e-4** (scripts/smoke_eval_qat_fakequant_off.py:102); argmax agreement on every pixel whose
top-two margin exceeds 2 × PARITY_TOL (share 1.0000); the toggle bites at 6.986e-01. In A7 (b) at 759440f the same three
cases gave the same values: max |logit difference| 8.583e-06 <= 1e-4, argmax agreement 1.0000 on the 1.0000 of pixels
whose top-two margin exceeds 2e-4, the toggle 6.986e-01 (> 1e-3), and the mapping's 314 FP32 keys with 1261 fake-quant
or observer keys and 18 FixedQParams aliases left out.

**FQR12's mapping of Q2's exceptions to the fqoff codes** (DL-24 R1-F4; src/eval/stage_artifacts.py's validator at
759440f; each message keeps Q2's code, or the exception's type, in "[…]"):

| Q2 call in the chain | What it raises | fqoff code (site) |
|---|---|---|
| the validator's own check that `--qat-run-dir` is given | QATRefused `run_dir_missing` | 10 `fqoff_run_record` (:428) |
| src/quant/qat_artifacts.py read_run_record | QATRefused `run_dir_test_path` (its refuse_test_path), `telemetry_missing`, `telemetry_not_strict_json`, `run_meta_rows`, `epoch_end_duplicate`; OSError; ValueError (a UnicodeDecodeError among them). A JSON line that is not an object (AttributeError) or an epoch that is not an int (TypeError) is none of these and escapes uncoded, failing closed (DL-24 R1-F3, refuted) | 10 `fqoff_run_record` (:428) |
| qat_artifacts.py require_complete | QATIncomplete `run_incomplete` | 11 `fqoff_run_incomplete` (:432) |
| src/quant/x86_latency.py validate_qat_sidecar, on `torch.load(io.BytesIO(<the bytes read once>), map_location="cpu", weights_only=True)` (load_qat_sidecar's load and checks, on those bytes since C5a) | X86LatencyCopyError `qat_sidecar_malformed`, `qat_sidecar_wrong_kind`, `qat_sidecar_no_state`, `qat_sidecar_role_conflict`; any exception of the load | 18 `fqoff_not_qat_train_state` (:501) |
| src/quant/qat_select.py rules_path_for_stage | never raises (a stage no rules file lists gets Q2's file; code 9 then refuses a stage other than the requested one) | — |
| src/quant/qat.py names_test, sha256_bytes | pure | — |

The validator's own reads (DL-24 R1-F2): an OSError on reading the finalize record is code 14 `[unreadable]` since C5a
(:451-454), as FQR12's code 14 names "unreadable". An OSError on reading the selection (:390),
the committed rules file (`Q.sha256_file`, :406) or the eNN.pt (:487) escapes uncoded and
fails closed; FQR12 names no "unreadable" cause for codes 7, 8 or 17, so the verifier confirmed R1-F2 for code 14's
site only.

**Two corrections to phase 1's text** (phase 1's sections are not edited; these lines supersede them):

- **1.6, the loader** (DL-24 R2-N1, confirmed). 1.6 says the loader's refusals go through one module-private helper,
  `_fqoff_refuse`. Codes 21 and 22 do (src/eval/model_loading.py:338 and :347). FQR9 step 1's re-hash does not: when
  the bytes it reads differ from the carrier's sha256 it raises `EvalRuntimeError("checkpoint_changed_after_validation",
  …)` directly (:330), the FP32 path's code by FQR9, which is none of FQR12's 22 codes. model_loading.py is unchanged
  since C1b, so these lines hold at 11a88a2 and at 759440f.
- **1.5, the census** (DL-24 R5-1, confirmed). The census grep matches a precision written as a literal, a PRECISIONS
  name or `.precision`; it misses reads by subscript or `.get`. Five such sites read run.precision (REPOSITORY-PROVEN
  at 759440f; neither file set changed in this lane):

  | Site | qat_fakequant_off there | Why |
  |---|---|---|
  | src/stats/artifact.py:704-706 (check_input_provenance) | refused | the stats writer gate requires (student, `STAGE_PRECISION[stage]`), int8_qat for E5 and E6, before anything is written; src/stats/driver.py's load_input does not check the precision, so this is the one precision gate on the official driver path |
  | src/stats/ingest.py:259 | recorded | EvaluationRunIdentity copies the value; nothing branches on it there |
  | src/stats/val_artifacts.py:68 | recorded | the artifact's row copies the value |
  | src/stats/val_artifacts.py:124 (require_role) | refused | the (stage, role, precision) triple must equal the caller's, fp32 by default (1.5 lists its :121 def line) |
  | scripts/gap_bootstrap_val.py:133 (pairing_checks) | refused | the role's pairing check compares the triple with the role's (stage, role, precision) |

  1.5's row for src/stats/driver.py:70-72 says "no statistic reads a qat_fakequant_off artifact": true, and the
  reason is artifact.py:704-706's gate, which refuses the precision, not STAGE_PRECISION's mapping alone.

### 2.6 The smoke and the harness (phase 2)

- **scripts/smoke_eval_qat_fakequant_off.py** (C2b; C5a, C5b and C5c add DL-24's cases): 121 cases. Phase 1's sections
  stages 11, binding 56, loader 9, parity 3, finder 1, never_selects 3 and hygiene 2 (85); C2b's cli 14, run 9, summary
  5, order 3 and imports 1 (117 at C2b); DL-24's binding 2 (C5a: bind_reads_the_checkpoint_once,
  c14_finalize_unreadable_file), hygiene 1 (C5b: hygiene_temp_removed) and run 1 (C5c: run_h_changed_after_validation).
  At 759440f: stages 11, binding 58, loader 9, parity 3, finder 1, never_selects 3, cli 14, run 10, summary 5, order 3,
  imports 1, hygiene 3 (121 cases). The run cases score E5's (and E6-KD's) selected epoch end to end through
  `scripts.evaluate_model.run` on a synthetic VAL split: make_tree's 846 VAL stems under a PLANTSEG_DATA_ROOT the smoke
  sets in its own process before configs/data.py is imported, `--max-samples 4`, the artifact writer's two
  whole-repository git calls stubbed as scripts/synthetic_qat_fixtures.py's evaluator driver stubs them. The finder case
  now checks all 22 codes in FQR12's order and the table's keys against the 37 sites found.
- **Codes 1-5's cases** reach each refusal with every earlier gate passing: cli_base is a valid fqoff run's arguments,
  and each case changes one value (c01_stage_not_in_scope: E1, E4, E6-R1, E6-R2 and E7; c01_role_not_student;
  c02_provenance_forbidden; c03_checkpoint_missing, c03_binding_flag_missing (each of the four), c03_random_init,
  c03_sha256_malformed (upper case and 63 characters, each flag), c03_qat_flag_off_precision (an fp32 and an INT8 run
  with a `--qat-*` flag); c04_not_cpu; c05_protocol_upstream). order_cli_refusals_construct_nothing drives codes 1-5
  through run(): each refused before the adapter or a model exists. order_source_refusals_before_adapter drives codes
  6-20 through run(), each refused in the source branch with nothing constructed (15/15), and
  order_loader_refusal_after_adapter shows code 21 after the adapter and the model, with no summary written.
- **SHADOW_PATHS, measured again** (phase 1's open item 5): a cold run of the phase-2 smoke under an audit hook on
  `open` (imports included) opened 78 repository files, every one among the harness's 90 SHADOW_PATHS, so phase 2 adds
  none (the in-process run cases load what the imports case's fresh interpreter loads, and no killing case is the
  imports case). C5a-C5c add no repository import (MEASURED: A7 (b)'s imports case lists the same 39 modules as at
  C3b, 2.14): the validator's `io` and torch, and src/quant/x86_latency.py's validate_qat_sidecar beside the
  load_qat_sidecar it replaces, were loaded before.
- **scripts/smoke_eval_qat_fakequant_off_mutations.py** (C3b; C5a, C5b and C5c add DL-24's rows): at C3b MUTATIONS
  gain FQ8, FQ11, FQ14a and FQ14b (15 rows), SCOPE gains scripts/evaluate_model.py's validate_qat_fakequant_off_cli,
  listed first so the finder meets the codes in FQR12's order, and KILLS gains codes 1-5's 10 sites (37 rows). DL-24
  adds FQ16 and FQ17 (C5a), FQ18 (C5b) and FQ19 (C5c), and C5c redefines FQ14b (both P2 comparisons skipped): 19
  MUTATIONS rows and 37 KILLS rows, 56 edits in all. Q2's file is unchanged.
- **FQR15** (MEASURED with ast at 759440f, no job run): the 15 `old` strings of the merged harnesses that target the
  four files (LQP's LQ55-LQ64, LQ106-LQ108 and LQ117; E6-KD's MK31) and the 27 `old` strings of
  smoke_evaluator_identity's 21 mutations (all in scripts/evaluate_model.py) each occur exactly once: `ANCHORS: 42 old
  strings in the four files, 0 not exactly once` at 759440f (check_anchors2.py), and the lane harness's own 19 MUTATIONS
  anchors each once (`LANE ANCHORS: 19 rows, 0 not exactly once`). build_parser's `--stage` line (AM-23's M091) is
  unchanged; FQ13 needs no anchor.

**The mutation list** (both phases; each edit with the case that kills it; MEASURED in A7 (c)):

| Id | Commit | The edit | File | Killed by |
|---|---|---|---|---|
| FQ1 | C3a | fake quantization left on (disable_fake_quant not applied) | src/eval/model_loading.py | load_fake_quant_off_model |
| FQ2 | C3a | observers left on (disable_observer not applied) | src/eval/model_loading.py | load_fake_quant_off_model |
| FQ3 | C3a | a non-strict load: Q2's copy by name without load_state_by_name's exactness checks | src/eval/model_loading.py | c21_state_key_missing |
| FQ4 | C3a | the selection's sha256 not compared with its pin | src/eval/stage_artifacts.py | c07_selection_pin_differs |
| FQ5 | C3a | the winner's epoch file unchecked: any checkpoint of the run accepted | src/eval/stage_artifacts.py | c15_checkpoint_other_epoch |
| FQ6 | C3a | a converted TorchScript archive passed to the payload load as if it were the eNN.pt | src/eval/stage_artifacts.py | c18_torchscript_archive |
| FQ7 | C3a | the stored state_finite unchecked | src/eval/stage_artifacts.py | c20_state_not_finite |
| FQ9 | C3a | the census skipped | src/eval/model_loading.py | c22_toggle_not_taken |
| FQ10 | C3a | artifacts.py's precision rule removed | src/eval/artifacts.py | stages_rule_refuses_cut_stage |
| FQ12 | C3a | the finalize record ignored when present | src/eval/stage_artifacts.py | c14_finalize_selection_sha256 |
| FQ15 | C3a | the loader's re-hash of the eNN.pt's bytes before it loads them removed | src/eval/model_loading.py | load_file_changed_after_validation |
| FQ8 | C3b | a quantization backend recorded for this precision | scripts/evaluate_model.py | summary_precision_and_backend |
| FQ11 | C3b | the load dispatch's branch of this precision removed: the eNN.pt reaches load_student_checkpoint | scripts/evaluate_model.py | run_loads_through_fqoff_loader |
| FQ14a | C3b | LEFF B-1b's identity flag skipped at P1 for this precision: its handed-file case names the run dir | scripts/evaluate_model.py | run_h_mismatch_refused_before_parse |
| FQ14b | C3b, redefined in C5c | LEFF B-1b's identity flag skipped at P2 for this precision: neither the carrier's sha256 nor the re-hashed file compared | scripts/evaluate_model.py | run_h_changed_during_validation |
| FQ16 | C5a | the payload loaded from the path again, not from the bytes the validator hashed (DL-24 R3-F1) | src/eval/stage_artifacts.py | bind_reads_the_checkpoint_once |
| FQ17 | C5a | an unreadable finalize record's OSError escapes code 14 (DL-24 R1-F2) | src/eval/stage_artifacts.py | c14_finalize_unreadable_file |
| FQ18 | C5b | the smoke leaves its temp directory behind (DL-24 R4-1) | scripts/smoke_eval_qat_fakequant_off.py | hygiene_temp_removed |
| FQ19 | C5c | P2's re-hash of the handed file skipped for this precision: a file replaced after the validator returns passes P2 (DL-24 R3-F1, LFQOFF-RULE-3) | scripts/evaluate_model.py | run_h_changed_after_validation |

FQ6's anchor is `    if _is_torchscript_archive(ck_raw):` since C5a. The refusal rows, one per site (Q2's key: file,
function, code#k; the function is each file's SCOPE function), each with the cases that kill its removal:

| Site | Killed by |
|---|---|
| scripts/evaluate_model.py, fqoff_stage_not_in_scope#0 | c01_stage_not_in_scope |
| scripts/evaluate_model.py, fqoff_stage_not_in_scope#1 | c01_role_not_student |
| scripts/evaluate_model.py, fqoff_provenance_forbidden#0 | c02_provenance_forbidden |
| scripts/evaluate_model.py, fqoff_source_flags#0 | c03_checkpoint_missing |
| scripts/evaluate_model.py, fqoff_source_flags#1 | c03_binding_flag_missing |
| scripts/evaluate_model.py, fqoff_source_flags#2 | c03_random_init |
| scripts/evaluate_model.py, fqoff_source_flags#3 | c03_sha256_malformed |
| scripts/evaluate_model.py, fqoff_not_cpu#0 | c04_not_cpu |
| scripts/evaluate_model.py, fqoff_protocol#0 | c05_protocol_upstream |
| scripts/evaluate_model.py, fqoff_source_flags#4 | c03_qat_flag_off_precision |
| src/eval/stage_artifacts.py, fqoff_forbidden_path#0 | c06_forbidden_checkpoint, c06_forbidden_run_dir, c06_forbidden_selection |
| src/eval/stage_artifacts.py, fqoff_selection_sha256_mismatch#0 | c07_selection_absent, c07_selection_pin_differs |
| src/eval/stage_artifacts.py, fqoff_selection_not_of_record#0 | c08_selection_not_named_of_record |
| src/eval/stage_artifacts.py, fqoff_selection_not_of_record#1 | c08_selection_not_json, c08_selection_format |
| src/eval/stage_artifacts.py, fqoff_selection_not_of_record#2 | c08_selection_rules_sha256 |
| src/eval/stage_artifacts.py, fqoff_selection_not_of_record#3 | c08_selection_smoke_inputs |
| src/eval/stage_artifacts.py, fqoff_selection_not_of_record#4 | c08_winner_epoch_not_int, c08_winner_epoch_out_of_range, c08_winner_sha_malformed |
| src/eval/stage_artifacts.py, fqoff_stage_mismatch#0 | c09_selection_stage |
| src/eval/stage_artifacts.py, fqoff_run_record#0 | c10_run_record_missing, c10_run_dir_none |
| src/eval/stage_artifacts.py, fqoff_run_incomplete#0 | c11_run_incomplete |
| src/eval/stage_artifacts.py, fqoff_run_not_real#0 | c12_run_not_real |
| src/eval/stage_artifacts.py, fqoff_stage_mismatch#1 | c09_run_meta_stage |
| src/eval/stage_artifacts.py, fqoff_telemetry_sha256_mismatch#0 | c13_telemetry_pin, c13_telemetry_selection |
| src/eval/stage_artifacts.py, fqoff_selection_not_of_record#5 | c08_selection_run_id |
| src/eval/stage_artifacts.py, fqoff_finalize_disagrees#0 | c14_finalize_unreadable, c14_finalize_format, c14_finalize_status, c14_finalize_selection_sha256, c14_finalize_checkpoint, c14_finalize_run_id, c14_finalize_telemetry, c14_finalize_stage |
| src/eval/stage_artifacts.py, fqoff_not_the_winner#0 | c15_winner_checkpoint_not_its_epoch |
| src/eval/stage_artifacts.py, fqoff_not_the_winner#1 | c15_checkpoint_other_epoch, c15_row_names_other_file |
| src/eval/stage_artifacts.py, fqoff_checkpoint_missing#0 | c16_checkpoint_missing |
| src/eval/stage_artifacts.py, fqoff_checkpoint_sha256_mismatch#0 | c17_bytes_differ_from_row, c17_winner_sha_differs |
| src/eval/stage_artifacts.py, fqoff_not_qat_train_state#0 | c18_torchscript_archive |
| src/eval/stage_artifacts.py, fqoff_not_qat_train_state#1 | c18_wrong_kind, c18_unloadable |
| src/eval/stage_artifacts.py, fqoff_not_qat_train_state#2 | c18_wrong_role |
| src/eval/stage_artifacts.py, fqoff_stage_mismatch#2 | c09_payload_stage |
| src/eval/stage_artifacts.py, fqoff_checkpoint_identity#0 | c19_identity_epoch, c19_identity_run_id, c19_identity_seed, c19_identity_source |
| src/eval/stage_artifacts.py, fqoff_state_not_finite#0 | c20_state_not_finite, c20_state_flag_missing |
| src/eval/model_loading.py, fqoff_load_not_exact#0 | c21_state_key_missing, c21_state_shape |
| src/eval/model_loading.py, fqoff_fake_quant_enabled#0 | c22_toggle_not_taken |

### 2.7 The final runs (A7; MEASURED on 759440f)

A7 ran on the last code commit 759440f from 05:10:04Z to 12:34:16Z on 28 Oct 2026 (7 h 24 min; at its start I
reported the time and an expected end ~8.0-8.7 h later, INFERRED from A2's, STEP 3's and B-1b's times, against the
ADDENDUM's 10-11 h). It ran detached in a session of its own (`setsid bash $SCR/a7_driver.sh`), each run alone and in
the ADDENDUM's order: (a), (b) and (d)'s `--check-edits` through phase 1's runner (stdin /dev/null;
PLANTSEG_DATA_ROOT, PLANTSEG_IMAGE_DIGEST, PYTHONPATH and TMPDIR unset; "other venv pythons: none" at each part's
start), and each mutation harness under `setsid --wait timeout -k 60 <cap>`, never nohup: 3,600 s for (c) and (g),
21,600 s for (d), 18,000 s for (e), 10,800 s for (f). Nothing else ran beside it, and the repository was not written.
/tmp's filesystem had 18.27 GiB free at the start (LFQOFF-RULE-3 Q2's floor: 15 GiB) and never fell below 9.0 GiB.
Each row is compared with A2's line where A2 set one and STEP 3's otherwise; c07 and c11, which print no RESULT
line, by their last lines' counts; s01 with its final count; the harnesses with the ADDENDUM's lines.

| id | run | rc | s | RESULT line | baseline |
|---|---|---|---|---|---|
| a01 | smoke_frozen_blobs | 0 | 0 | RESULT: FROZEN BLOBS OK (10/10) | STEP 3 a01: same |
| a02 | smoke_run_ptq | 0 | 439 | RESULT: PASS (69/69) | A2 b02: same |
| a03 | smoke_calibration_lists | 0 | 3 | RESULT: PASS (42/42) | STEP 3 a03: same |
| a04 | smoke_qnnpack_ops | 0 | 216 | RESULT: PASS (8/8) synthetic artifact | STEP 3 a04: same |
| a05 | smoke_eval_int8 | 0 | 7 | RESULT: PASS (32/32) | A2 b12: same |
| a06 | smoke_efficiency | 0 | 4 | RESULT: PASS (82/82) | STEP 3 a06: same |
| a07 | smoke_quant_e4_e5 | 0 | 4 | RESULT: PASS (58/58) | STEP 3 a07: same |
| a08 | smoke_quant_e6_e7 | 0 | 4 | RESULT: PASS (37/37) | STEP 3 a08: same |
| a09 | smoke_quant_runners | 0 | 3 | RESULT: PASS (78/78) | STEP 3 a09: same |
| a10 | smoke_realrun_decisions | 0 | 2 | RESULT: PASS (76/76) | STEP 3 a10: same |
| a11 | smoke_qat_runner | 0 | 124 | RESULT: PASS (214/214) | STEP 3 a11: same |
| a12 | smoke_qat_seeding | 0 | 31 | RESULT: PASS (9/9) | STEP 3 a12: same |
| a13 | smoke_quant_x86_efficiency | 0 | 3 | RESULT: PASS (103/103) | STEP 3 a13: same |
| a14 | smoke_eval_determinism | 0 | 26 | RESULT: EVAL-DET OK (74/74) | A2 b13: same |
| a15 | smoke_am13_upstream | 0 | 18 | RESULT: L-AM13 OK (79/79) | A2 b14: same |
| a16 | smoke_teacher_d1 | 0 | 117 | RESULT: PASS (145/145) | STEP 3 a16: same |
| a18 | smoke_qat_selection | 0 | 861 | RESULT: PASS (174/174) | STEP 3 a18: same |
| a19 | smoke_qat_artifacts | 0 | 3038 | RESULT: PASS (94/94) | STEP 3 a19: same |
| a17 | smoke_arms_kd | 0 | 52 | RESULT: PASS (61/61) | STEP 3 a17p: same |
| a20 | smoke_teacher_diag_drift | 0 | 4 | RESULT: PASS (8/8) | A2 b04: same |
| a21 | smoke_ptq_record | 0 | 840 | RESULT: PASS (203/203) | A2 b03: same |
| a22 | smoke_eval_stage_artifacts | 0 | 4 | RESULT: PASS (66/66) | A2 b11: same |
| a23 | smoke_eval_teacher | 0 | 4 | RESULT: PASS (59/59) | A2 b01: same |
| a24 | smoke_teacher_arms | 2 | 0 | RESULT: NOT RUN (teacher stack missing: mmengine: ModuleNotFoundError; mmseg: ModuleNotFoundError; mmcv: ModuleNotFoundError) | STEP 3 a24: same (environment) |
| b1 | smoke_e6x_condition | 0 | 8 | RESULT: PASS (172/172) | STEP 3 b1: same |
| b2 | smoke_e6x_condition_mutations | 0 | 10 | RESULT: PASS (428/428) | STEP 3 b2: same |
| b3 | smoke_e6kd_trigger | 0 | 329 | RESULT: PASS (101/101) | STEP 3 b3: same |
| b5 | smoke_evaluate_corruptions | 1 | 0 | (no RESULT line) ModuleNotFoundError: No module named 'skimage' | STEP 3 b5: same (environment) |
| b6 | smoke_eval_robustness | 1 | 2 | (no RESULT line) ModuleNotFoundError: No module named 'skimage' | STEP 3 b6: same (environment) |
| c01 | smoke_am13_teacher | 2 | 2 | RESULT: L-AM13 TEACHER NOT RUN (0/0) | STEP 3 c01: same (environment) |
| c02 | smoke_am5_exclusion | 0 | 6 | RESULT: L-AM5 OK (50/50) | STEP 3 c02: same |
| c03 | smoke_eligibility_variants | 0 | 5 | RESULT: L-AM17-GTPRESENT OK (34/34) | STEP 3 c03: same |
| c04 | smoke_eval_contract | 0 | 2 | RESULT: A2a OK -- CORE CONTRACT PROVEN | A2 b15: same |
| c05 | smoke_eval_plantseg | 1 | 3 | RESULT: A2b BLOCKED | STEP 3 c05: same (environment) |
| c06 | smoke_gap_bootstrap_val | 0 | 44 | RESULT: L-AM17B-GAP OK (77/77) | STEP 3 c06: same |
| c07 | smoke_mde | 0 | 672 | (no RESULT line) PASS 129/129 (mutations killed 35/35) | STEP 3 c07: same |
| c08 | smoke_perclass_gap_table | 0 | 7 | RESULT: L-AM17-PERCLASS OK (38/38) | STEP 3 c08: same |
| c09 | smoke_plantseg_class_map | 0 | 0 | RESULT: CLASS MAP VALID (offline) | STEP 3 c09: same |
| c10 | smoke_preflight_e1_trainval | 0 | 6 | RESULT: PASS (125/125) | STEP 3 c10: same |
| c11 | smoke_split_censuses | 0 | 3 | (no RESULT line) PASS 81/81 (mutations killed 18/18) | STEP 3 c11: same |
| c12 | smoke_stats_driver | 0 | 10 | RESULT: STATS DRIVER CORE OK (55/55) | STEP 3 c12: same |
| c13 | smoke_stats_integrity | 0 | 47 | RESULT: STATS INTEGRITY OK (129/129) | STEP 3 c13: same |
| c14 | smoke_stats_paired | 0 | 5 | RESULT: A3a OK -- PAIRED INFERENCE CORE PROVEN | STEP 3 c14: same |
| c15 | smoke_teacher_calibration | 0 | 16 | RESULT: PASS (52/52) | STEP 3 c15: same |
| c16 | smoke_teacher_d4 | 0 | 11 | RESULT: PASS (32/32) | STEP 3 c16: same |
| c17 | smoke_teacher_selection | 2 | 2 | (no RESULT line) ENVIRONMENT: teacher stack not importable (No module named 'mmengine'); run in the plantseg-teacher image. | STEP 3 c17: same (environment) |
| x1 | smoke_am23_quant | 0 | 440 | RESULT: PASS (187/187) | A2 b05: same |
| n1 | smoke_e6kd_runner | 0 | 175 | RESULT: PASS (215/215) | STEP 3 n1: same |
| n2 | smoke_e6kd_selection | 0 | 395 | RESULT: PASS (37/37) | STEP 3 n2: same |
| n3 | smoke_e6kd_invariance | 0 | 38 | RESULT: PASS (11/11) | STEP 3 n3: same |
| j1 | smoke_probe_e1_grad | 0 | 10 | RESULT: PASS (24/24) | STEP 3 j1: same |
| j2 | smoke_score_teacher_train | 0 | 12 | RESULT: PASS (13/13) | STEP 3 j2: same |
| j3 | smoke_teacher_strict_load | 0 | 10 | RESULT: PASS (21/21, 21 skipped) | STEP 3 j3: same |
| n4 | smoke_e6kd_step `--selfcheck` | 0 | 15 | RESULT: PASS (23/23) | STEP 3 n4s: same |
| n01 | smoke_evaluator_identity | 0 | 50 | RESULT: PASS (38/38; 21/21 edits killed) | A2 b06: same |
| s01 | smoke_eval_qat_fakequant_off | 0 | 116 | RESULT: PASS (121/121) | declared count: same |
| l1 | smoke_ptq_record_mutations `--check-edits` | 0 | 1 | RESULT: PASS (143/143 checks; 0/0 edits killed) | A2 b07: same |
| c_fqoff_mutations | smoke_eval_qat_fakequant_off_mutations | 0 | 354 | RESULT: PASS (86/86) | expected |
| d_ptq_record_mutations | smoke_ptq_record_mutations (full run) | 0 | 4578 | RESULT: PASS (427/427 checks; 141/141 edits killed) | expected |
| e_qat_mutations | smoke_qat_mutations `--workers 2` | 0 | 11043 | RESULT: PASS (381/381) | expected |
| f_e6kd_mutations | smoke_e6kd_mutations `--workers 2` | 0 | 2102 | RESULT: PASS (225/225) | expected |
| g_am23_mutations | smoke_am23_mutations | 0 | 321 | RESULT: PASS (311/311) | expected |

The comparison script printed `DIFFERENT ROWS: 0` over the 62 rows.

- **Changed counts: none.** Every (a) row gives its baseline line. The two P34 smokes keep their counts with the new
  pin (a20 8/8, x1 187/187). c07 and c11 keep `129/129 (mutations killed 35/35)` and `81/81 (mutations killed
  18/18)`. a17 runs with PYTHONPATH set to the checkout, its documented form (lqp.md:127-130), and n4 with
  `--selfcheck` (am3-e6kd.md:911), as STEP 3's a17p and n4s did.
- **Environment rows** (the same lines and causes as at STEP 3 and in B-1b's report): a24, c01 and c17, no teacher
  stack (mmengine, mmseg, mmcv); b5 and b6, no skimage; c05, `A2b BLOCKED`, no VAL data, as PLANTSEG_DATA_ROOT is
  never set (leff-b1b.md:270).
- **(b)** s01, the lane smoke, cold: `RESULT: PASS (121/121)` in 116 s, its twelve sections each PASS (stages 11,
  binding 58, loader 9, parity 3, finder 1, never_selects 3, cli 14, run 10, summary 5, order 3, imports 1, hygiene 3);
  its temp folder was removed at its end.
- **(c)** the lane harness: `RESULT: PASS (86/86)` in 354 s. Its 56 edits, the 19 FQ rows and the 37 refusal rows,
  are each KILLED by a case that 2.6's tables name (checked line by line); refusal_table_names_only_real_sites `[]`,
  every_job_has_a_verdict (56 jobs), harness_ran_unsignalled and the 26 self-checks PASS. Its header's "at commit
  b9935d6cc38a" is the shadow's own commit of the 90 SHADOW_PATHS copied from the checkout (Q2's build_shadow), not
  the checkout's HEAD.
- **(d)** `--check-edits` 143/143, then the full run with the harness's default one shadow, as B-1b's and AM-23's runs
  of record: `RESULT: PASS (427/427 checks; 141/141 edits killed)` in 4,578 s; harness_ran_unsignalled and
  every_shadow_clean_at_the_end PASS.
- **(e)** Q2's harness: `RESULT: PASS (381/381)` in 11,043 s, one run inside its 18,000 s cap, so q2-qat.md §17.4's
  two-run form was not needed. All 351 edits KILLED; every_job_has_a_verdict (351 jobs) and
  refusal_table_names_only_real_sites `[]` PASS, so its KILLS keys equal the sites: every key a real site, and every
  site's job killed by its case. harness_ran_unsignalled PASS.
- **(f)** `RESULT: PASS (225/225)` in 2,102 s: all 195 edits KILLED; the same three checks PASS.
- **(g)** `RESULT: PASS (311/311)` in 321 s.
- **(h)** none: master's change brings no mutation runner of B-1b's own; smoke_evaluator_identity (n01) runs its 21
  mutations inside (`38/38; 21/21 edits killed`).
- **(i) and (j)**: 2.9.

### 2.8 DL-24 (A5)

**The run** (MEASURED). The read-only verification workflow ran on C3b `521e3df` from ~13:04Z to 14:40Z on 27 Oct 2026:
10 agents, 0 errors, 671 tool calls, 5,735 s of agent time, at most 2 at a time (4 CPUs). Five reviewers each read one
dimension at C3b: model (the loader, the FP32 reference, the parity), binding (the validator, codes 6-20), cli_run
(scripts/evaluate_model.py's CLI rules, run() and LEFF B-1b's flags), smoke_harness (the two lane smokes), invariance
(what must not change: src/quant, the selection path, the census, P34 and D2). Five verifiers then each tried to refute
one reviewer's findings, reading the code again. The run: workflow `wf_e92200c0-26c`, script `$SCR/p2/dl24_workflow.js`;
the agents (label, id): review:model `ad29c7ec0593014e9`, review:binding `acff28d07b3516aa0`, review:cli_run
`ac4fe31366fdd1154`, review:smoke_harness `a2b75d1d326aa3d27`, review:invariance `a2a79e371069fc02d`, verify:model
`a6951497ee490e8d3`, verify:binding `a39b10c59547a6f6d`, verify:cli_run `aac0deecd8ee7e23e`, verify:smoke_harness
`aa643263ca879003f`, verify:invariance `a14875e6ad57d0b32`. Before and after (`$SCR/p2/dl24_before.txt`,
`dl24_after.txt`): HEAD `521e3df`, the scoped `git status --porcelain=v1` empty, and the sha256 of `git show
521e3df:<path>` of the lane's nine files equal; `git ls-remote --heads origin lane/fqoff` printed `521e3df` both times.

**What the agents did** (MEASURED from their transcripts with two scratch scripts, one listing every write-capable
tool call, the other every shell command against the RULES). Writes: none (three flags, each a false positive: two
`printf` loops printing "=>" and one grep pattern containing "chmod"). Read-only deviations from the RULES' search
forms: review:binding's `git grep -n -l 'E6-KD' 521e3df -- src scripts configs ':(exclude,icase)*test*'` (a file
listing over scripts); reads at commits other than C3b by literal paths (O, d211f08, 11a88a2, 879cab5 and the record
pins 2b3b556, 3373139 and ebc1201); review:invariance's Grep-tool read of docs/lane_reports/fqoff.md in the working
tree; review:cli_run's two `git grep` commands on one explicit file each without the exclude pathspec; `git ls-files`
on configs and on explicit files; `cd`, `echo`, `cat -n`, `awk`, `tr`, `sort`, `uniq` and `printf` as filters; and
`for` loops over patterns or commits, every path literal. The scan flagged no command or tool call naming
docs/reference, or naming the token outside the exclude pathspec (so no TEST path and not src/stats/tests.py).

**The findings** (14 from the reviewers; each verifier's verdict; the resolution, with the orchestrator's rulings
LFQOFF-RULE-3 of 28 Oct 2026 where they apply):

| Id | Reviewer | Severity | Finding | Verdict | Resolution |
|---|---|---|---|---|---|
| R1-F1 | binding | minor | The binding accepts E5 seed 42's losing U4 pilot run: it has no qat_finalize.json, since finalize refuses the loser, and FQR2 never requires one | REFUTED: FQR2 and FQR12 list the binding's causes; AM-4a item 3 keeps the loser "retained and reported"; the scenario needs the loser's inputs pinned, and DL-139 records neither pilot's selection sha256; a guard would need a new ruling | No lane change (LFQOFF-RULE-3). Binding E5 seed 42's fake-quant-off score to the U4 winner only goes to TMAN's T-FQ items as NOTE-FQ-U4-1 (2.12) |
| R1-F2 | binding | nit | An OSError on reading qat_finalize.json, the selection or the eNN.pt escapes as a raw exception, not its code (fails closed; the code is lost) | CONFIRMED for code 14's site only: FQR12's code 14 names "unreadable", codes 7 and 17 do not | C5a: the finalize read in a try, so an unreadable record is code 14 `[unreadable]` (:451-454); case c14_finalize_unreadable_file; mutation FQ17 |
| R1-F3 | binding | nit | read_run_record's AttributeError (a JSON line that is not an object) or TypeError (an epoch that is not an int) escapes code 10's except | REFUTED: code 10 maps Q2's coded refusals, and a hand-edited row's AttributeError is not one; it fails closed | None |
| R1-F4 | binding | nit | Phase 1's report does not list FQR12's mapping of Q2's exceptions to the fqoff codes | CONFIRMED | Report: the mapping, 2.5 |
| R2-N1 | model | nit | 1.6 says the loader's refusals all go through `_fqoff_refuse`; FQR9 step 1's re-hash raises `EvalRuntimeError("checkpoint_changed_after_validation", …)` directly | CONFIRMED: the sentence is false at its own tip 11a88a2, and model_loading.py is unchanged since | Report: the correction, 2.5 (phase 1's text is not edited) |
| R3-F1 | cli_run | minor | P2 compares the carrier's sha256, which the validator took before it parsed the eNN.pt: a pre-parse hash, the teacher's kind that B-1b's P2 re-hashes, not FP32's kind. A swap during the parse passes P2 | CONFIRMED, with a broader path: a swap right after the validator returns also passes P2 and is refused only at the loader's re-hash, after the adapter; B-1b's fp32 and teacher cases require `source_changed_during_validation` with no dataset | C5a: the validator reads the eNN.pt once and hashes, scans and loads those bytes (:487-498); case bind_reads_the_checkpoint_once; FQ16. C5c (LFQOFF-RULE-3 Q1 (a)): P2 also re-hashes the handed file (:511-513); case run_h_changed_after_validation; FQ19; FQ14b redefined |
| R3-F2 | cli_run | nit | An empty `--qat-run-dir` or `--qat-selection` passes code 3 (`is None`), unlike `--checkpoint` (truthiness) | REFUTED: "missing" is None by FQR11 and FQR12; an empty value is refused at codes 6, 7 or 10; no wrong binding | None |
| R4-1 | smoke_harness | minor | The lane smoke never removes its temp directory, unlike Q2's smokes (smoke_qat_selection, smoke_e6kd_selection, smoke_qat_artifacts) | CONFIRMED | C5b: main() removes TMP after its checks (:1863); case hygiene_temp_removed; FQ18. The 433 directories earlier runs left were removed under LFQOFF-RULE-3 Q2 (2.15) |
| R4-2 | smoke_harness | nit | The finder case's name, docstring and section header still say phase 1 | REFUTED: its detail line prints "37 sites of 22 codes"; the name is the one phase 1's report cites | None |
| R4-3 | smoke_harness | nit | The smoke's relabel sets u4_pilot false and clip_source clip_selection, unlike Q2's relabel | REFUTED: that meta is Q2's own run-of-record form (smoke_qat_selection.py:440-444, :1450-1453, :1623-1627, :1374-1375) | None |
| R4-4 | smoke_harness | nit | The cpu_only hunk is a statement D2 mirrors, made teacher-inert by short-circuit, not a hunk outside the mirrored part | REFUTED: D2 mirrors "run()'s teacher path, call for call" (teacher_d2_calibration.py:5); no hunk adds, removes or reorders a teacher-path call or changes an argument | 2.5's P34 paragraph gives the verifier's reading |
| R5-1 | invariance | minor | The census regex misses subscript and `.get` reads of the precision: src/stats/artifact.py:704-706 (check_input_provenance, the stats writer gate), src/stats/ingest.py:259, src/stats/val_artifacts.py:68 and :124, scripts/gap_bootstrap_val.py:133 | CONFIRMED: artifact.py:704 is the only precision gate on the official driver path | Report: the census correction, 2.5 |
| R5-2 | invariance | minor | = R3-F1; either way, the comment "as FP32's ckpt_sha" on P2's line is wrong | CONFIRMED (a duplicate of R3-F1) | C5c: the comment now says what the block compares |
| R5-3 | invariance | nit | Code 6's message, comment and docstring say "nothing was opened", false when P1 hashed `--checkpoint` | CONFIRMED (the message false on the CLI path) | C5a: the text says the validator opened nothing (:387-388) |

Seven confirmed (R1-F2, R1-F4, R2-N1, R3-F1, R4-1, R5-1, R5-3), one confirmed duplicate (R5-2), six refuted (R1-F1,
R1-F3, R3-F2, R4-2, R4-3, R4-4).

**The STOP and the rulings.** At 15:08Z (27 Oct) A5 stopped with three questions: Q1, whether P2 re-hashes the
handed file for this precision (an A3 amendment) or keeps the carrier comparison with its comment corrected; Q2,
whether to remove the lane smoke's leftover temp directories before A7; Q3, whether to push C5a and C5b at once. The
rulings (LFQOFF-RULE-3, received ~04:24Z on 28 Oct): Q1 (a), Q2 remove, Q3 push now; R1-F1 no lane change
(NOTE-FQ-U4-1); R1-F4, R2-N1 and R5-1 into this report; no new merge-in.

**A3 as amended** (LFQOFF-RULE-3 Q1): A3's "compares fqoff["checkpoint_sha256"] with H, as it compares FP32's
ckpt_sha" now reads "re-hashes the handed file at P2 and compares it with H, as B-1b does for a pre-parse recorded
hash, and also compares the carrier's checkpoint_sha256 with H". 2.4's flag table gives the amended row.

### 2.9 Independence and scope (A6, A7 (i) and (j); MEASURED at 759440f)

**A6** (after A5's commits, at 759440f; MEASURED 2026-10-28T05:08:41Z):

```
$ git diff --name-only 879cab53770e4a761f5d3b98ce1b222158a344d7 HEAD -- src/quant ':(exclude,icase)*test*'
(no output)
$ git grep -n -e qat_fakequant_off -- src scripts configs ':(exclude,icase)*test*'   (per file: lines)
     32 scripts/evaluate_model.py
     26 scripts/smoke_eval_qat_fakequant_off.py
     20 scripts/smoke_eval_qat_fakequant_off_mutations.py
      4 src/eval/artifacts.py
      3 src/eval/model_loading.py
      8 src/eval/stage_artifacts.py
(93 lines in all)
```

The first prints nothing: no src/quant file changed. The second lists only the four edited files and the two lane
smokes.

**A7 (i)** (MEASURED 2026-10-28T12:34:36Z, on 759440f, before the report commit):

```
$ git diff --name-status --no-renames 879cab53770e4a761f5d3b98ce1b222158a344d7 HEAD -- src scripts configs \
    docs/lane_reports ':(exclude,icase)*test*'
A  docs/lane_reports/fqoff.md
M  scripts/evaluate_model.py
M  scripts/smoke_am23_quant.py
A  scripts/smoke_eval_qat_fakequant_off.py
A  scripts/smoke_eval_qat_fakequant_off_mutations.py
M  scripts/smoke_teacher_diag_drift.py
M  src/eval/artifacts.py
M  src/eval/model_loading.py
M  src/eval/stage_artifacts.py
```

**A7 (j)** (the same time):

```
$ git diff --name-only 879cab53770e4a761f5d3b98ce1b222158a344d7 HEAD -- src/quant src/training src/distill \
    src/models src/data src/stats src/eval/__init__.py src/eval/metrics.py src/eval/protocols.py \
    src/eval/evaluate.py src/eval/eval_runtime.py src/eval/adapters.py configs scripts/qat_epoch_eval.py \
    scripts/select_qat_epoch.py scripts/select_clip.py scripts/evaluate_corruptions.py scripts/e6kd_eval.py \
    scripts/run_e6kd.py scripts/synthetic_qat_fixtures.py scripts/synthetic_ptq_fixtures.py \
    scripts/smoke_qat_mutations.py scripts/smoke_ptq_record_mutations.py scripts/smoke_am23_mutations.py \
    scripts/smoke_e6kd_mutations.py scripts/smoke_evaluator_identity.py ':(exclude,icase)*test*'
(no output)
```

It prints nothing.

**The rules files' sha256** (of `git show 759440f:<path>`, MEASURED ~05:19Z on 28 Oct), each its value at O (1.9):

- configs/qat_selection_rules.json: `76addc57d0367dda9a1163f749ec32c9c240b20736104dc12aa77488344f9568`
- configs/qat_selection_rules_am23.json: `e299941524968f143359d154566930819cd064152691395fb3647f4de0d98f3d`
- configs/e6kd_selection_rules.json: `5acad295ffd71ea7630bf10184d0ae0f959a395ed7a27306b23f7dc86c847304`

**ACCEPTANCE 7** (REPOSITORY-PROVEN at 759440f): no selection, rule, trigger or gate reads a qat_fakequant_off
summary. The three rules files require int8_qat (1.5), src/quant/qat_select.py's _check_summary compares every
summary's run.precision with it, src/quant/e6kd_trigger.py scores E3 at fp32 and E6 at int8_qat only, and
src/stats/artifact.py:704-706 refuses the precision before the stats driver writes anything (2.5's census
correction); the smoke's never_selects section (3 cases) shows each refusal, and summary_never_selected shows a
selection's own summary check refusing such a summary. No src/quant file changed (A6).

**ACCEPTANCE 8**: no run created a path containing "test" or ":" (MEASURED: the lane smoke's
hygiene_no_forbidden_named_path_created PASS in A7 (b), over its temp folder and cache; /tmp held 393 names holding the
token before A7 and 393 after, every one under the session's own folder /tmp/claude-0 (the venv's package folders), and
no name holding ':' either time ($SCR/p2/tmp_usage.py, which counts such names and never prints or opens them); A7's
output folder holds only run ids and smoke names); the scoped status after the report commit: before it, only this file,
docs/lane_reports/fqoff.md (MEASURED at A8); after it, empty (A9's check).

### 2.10 The rulings at 759440f (REPOSITORY-PROVEN; MEASURED where a run is named)

| Ruling | Where it is met | Shown by |
|---|---|---|
| FQR1 scope | `FQOFF_STAGES = ("E5", "E6", "E6-E2", "E6-KD")`, src/eval/artifacts.py:66, its only definition (`git grep -n -e 'FQOFF_STAGES =' -- src scripts configs ':(exclude,icase)*test*'`: two lines, the definition src/eval/artifacts.py:66 and the smoke's equality check scripts/smoke_eval_qat_fakequant_off.py:531); imported by scripts/evaluate_model.py's code 1 (:156); E6-R1 and E6-R2 get no row (cut, DL-137, DL-138) | stages_fqoff_stages_defined_once, stages_fqoff_stages_are_qat_int8, stages_cut_stages_left_out, c01_stage_not_in_scope |
| FQR2 the binding | validate_qat_fakequant_off_source, src/eval/stage_artifacts.py:328-519, returning the carrier; resolve_evaluation_source's route by keyword arguments only, :579-609; src/quant imported inside the functions only; `allow_smoke_inputs=args.artifact_status == "smoke"`, scripts/evaluate_model.py:480 | bind_* (phase 1), bind_resolve_old_routes_unchanged, codes 6-20's cases, run_e5_end_to_end, run_e6kd_end_to_end |
| FQR3 the summary | the request rule, src/eval/artifacts.py:307; RunMeta through run(), scripts/evaluate_model.py:534; precision hashed in the config payload, artifacts.py:276 | stages_rule_* (FQ10), stages_config_payload_hashes_precision, summary_* |
| FQR4 TMAN owns the row kind | this lane writes no inventory row and no TMAN file (2.9, A7 (i)) | A7 (i) |
| FQR5 who scores, and where | the run paths, 2.11; this lane builds neither the TEST runner nor the VAL block | 2.11 |
| FQR6 the records | proposed verbatim, 2.13; no governed document written (A7 (i) lists none) | A7 (i) |
| FQR7, FQR7a the base, the phases and the files | phase 1 from O; phase 2 on the merge-in d211f08 (2.1); only the lane's files differ from master (A7 (i)); no src/quant file, nor src/eval/__init__.py or metrics.py (A6, A7 (j)); the session never merged (L-PROT denies it) | A6, A7 (i), (j) |
| FQR8 the session | one cloud session on lane/fqoff, pushing lane/fqoff only (2.2); the harness rows FQ1-FQ15 bar FQ13 (FQ14 as FQ14a and FQ14b) and DL-24's FQ16-FQ19 (2.6), and every refusal row; DL-24 (2.8) | 2.2, 2.6, 2.8 |
| FQR9 the model | load_qat_fakequant_off_student, src/eval/model_loading.py:306-349: the bytes' re-hash :329-332, `torch.load(…, weights_only=True)` :333, prepared_skeleton() :334, load_state_by_name :336, disable_fake_quant and disable_observer :340-341, eval() :342, the census :346; called by run() only (:574), ck_info None | load_*, c21_*, c22_toggle_not_taken, run_loads_through_fqoff_loader, summary_runtime_record |
| FQR10 TMAN's list | 2.12, with the final flag names | 2.12 |
| FQR11 the flags and the CLI | scripts/evaluate_model.py: SUPPORTED_PRECISIONS :63, the choices :88, the four flags :117-127, validate_qat_fakequant_off_cli :150-191, its call :205, the source branch :470-485, cpu_only :555, the load dispatch :572-574; handed_file's case :391-392 and P2's line :511-513 (the flag table) | cli_*, c01-c05, run_*, summary_*, FQ8, FQ11, FQ14a, FQ14b, FQ19 |
| FQR12 the 22 codes | 2.5's list, in order; the raise forms `_fqoff_cli` (scripts/evaluate_model.py:144-147), `_fail` and `_fqoff_refuse` (src/eval/model_loading.py:300-303, EvalRuntimeError); Q2's harness unchanged, the lane's finder assigned over H.refusal_sites (scripts/smoke_eval_qat_fakequant_off_mutations.py:260) | finder_finds_phase1_sites (22 codes, 37 sites); the harness, every site KILLED (2.6, A7 (c)) |
| FQR13 clean and canvas | code 5, scripts/evaluate_model.py:185 | c05_protocol_upstream, summary_runtime_record (canvas) |
| FQR14 the FP32 reference and the parity | fp32_reference_from_qat_state, src/eval/model_loading.py:352-432; PARITY_TOL = 1e-4, scripts/smoke_eval_qat_fakequant_off.py:102 | parity_fp32_reference_logits, parity_mapping_accounts_every_key, parity_toggle_bites |
| FQR15 the merged harnesses | every `old` string of LQP's, AM-23's, E6-KD's and Q2's harnesses that targets the four files occurs once (`ANCHORS: 42 old strings in the four files, 0 not exactly once` at 759440f (check_anchors2.py), and the lane harness's own 19 MUTATIONS anchors each once (`LANE ANCHORS: 19 rows, 0 not exactly once`)); no merged smoke edited but by FQR15a; none compares a whole tuple this lane extends (2.4) | A7 (d) `--check-edits`, (e), (f), (g) |
| FQR15a P34's re-pin | scripts/smoke_teacher_diag_drift.py:36 and its docstring :11; scripts/smoke_am23_quant.py:83 (2.5) | A7 (a) a20 8/8, x1 187/187 |
| FQR16 hours and merge | phase 2 took about 25.1 h elapsed, from the ADDENDUM's receipt (27 Oct ~11:36Z) to the report commit (28 Oct ~12:40Z): A1-A5 ~3.5 h (to A5's STOP at 15:08Z), the STOP's wait for LFQOFF-RULE-3 ~13.3 h (to ~04:24Z), the rulings, C5c, C5d and A6 ~0.8 h, A7 7.4 h and A8 ~0.1 h (MEASURED from the session's timestamps, rounded); the final merge is Ice's block | — |

### 2.11 The run paths (FQR5, FQR10; text for the orchestrator)

One process per row. Neither command is run by this lane; the orchestrator builds the VAL block and the TEST-day
plan from them (FQR5). `<S>` is the stage (E5 or E6 at seed 42, 43 or 44, E6-E2, E6-KD; FQR1); `<run>` the QAT run
directory, `<eval>` its eval directory and `<NN>` the selection's winner epoch.

**VAL (provisional; the L-FQOFF VAL block on the laptop, at a clean clone of the merged commit, before the freeze):**

```
python -B scripts/evaluate_model.py --stage <S> --model-role student --precision qat_fakequant_off \
  --split val --condition clean --protocol canvas --device cpu --batch-size 1 \
  --checkpoint <run>/epoch_ckpts/e<NN>.pt --qat-run-dir <run> \
  --qat-selection <eval>/qat_selection.json --qat-selection-sha256 <selection sha256> \
  --qat-telemetry-sha256 <telemetry sha256> \
  [--expected-checkpoint-sha256 <winner checkpoint sha256>] [--threads <T>] \
  --artifact-status provisional --run-id <run id> --out-dir <out dir>
```

(H and T are optional off R; given, they bind the eNN.pt's bytes at P1 and P2 and record T.)

**TEST (official; the single TEST session, in TL-1's context):**

```
python -B scripts/evaluate_model.py --stage <S> --model-role student --precision qat_fakequant_off \
  --split test --confirm-test-split --artifact-status official \
  --condition clean --protocol canvas --device cpu --batch-size 1 \
  --checkpoint <input root>/<run>/epoch_ckpts/e<NN>.pt --qat-run-dir <input root>/<run> \
  --qat-selection <input root>/<eval>/qat_selection.json --qat-selection-sha256 <selection sha256> \
  --qat-telemetry-sha256 <telemetry sha256> \
  --expected-checkpoint-sha256 <source.sha256> --threads <T> \
  --run-id <run id> --out-dir tday_fqoff_<file_id>
```

| Flag | Value | Where it comes from |
|---|---|---|
| `--stage` | E5, E6, E6-E2 or E6-KD | FQR1's 8 models; any other stage is code 1 |
| `--model-role` | student | code 1 refuses any other role |
| `--precision` | qat_fakequant_off | EC §5.3's value (2.13) |
| `--split`, `--confirm-test-split` | val; or test with the confirm flag | FQR5; the TEST guards (2.5) |
| `--condition`, `--protocol` | clean, canvas | FQR13 (code 5); clean is the only supported condition |
| `--device`, `--batch-size` | cpu, 1 | FQR5 (TL-1: CPU, batch 1); code 4 refuses any other device. Workers are 0 by EVAL_NUM_WORKERS (src/eval/eval_runtime.py:44), not a flag |
| `--checkpoint` | `<run>/epoch_ckpts/e<NN>.pt` | the selection's winner.checkpoint under the run directory; TEST: T-FQ2's source.path, under the input root |
| `--qat-run-dir` | `<run>` | the run directory holding qat_telemetry.jsonl and epoch_ckpts/ (T-FQ3, T-FQ6) |
| `--qat-selection` | `<eval>/qat_selection.json` | the selection of record (FQR2; T-FQ3) |
| `--qat-selection-sha256` | 64 lowercase hex | the selection's sha256 in its decision-log row (T-FQ3) |
| `--qat-telemetry-sha256` | 64 lowercase hex | the pod's copy-back value of the run's qat_telemetry.jsonl (T-FQ3) |
| `--expected-checkpoint-sha256` | 64 lowercase hex | the selection's winner.checkpoint_sha256; TEST: T-FQ2's source.sha256. Required under R (B-1b) |
| `--threads` | T | TL-1's one thread count; required under R (B-1b); VAL may give the laptop's |
| `--artifact-status` | provisional (VAL), official (TEST) | FQR5; either binds only a real run and a selection with smoke_inputs false (FQR2) |
| `--run-id`, `--out-dir` | the block's id; TEST `tday_fqoff_<file_id>` | FQR5, IF§2 |
| not given | `--provenance`, `--random-init`, `--max-samples`, `--expected-converted-sha256` | codes 2 and 3; VAL scores all 846 rows; H2 is INT8 only |

No path given to `--checkpoint`, `--qat-run-dir` or `--qat-selection` may name "test" in any case (code 6, before
any of them is opened).

### 2.12 What TMAN must add (FQR10, with the final flag names)

- **T-FQ1.** TMAN writes every qat_fakequant_off row; this lane writes none.
- **T-FQ2.** One row per FQR1 model that has a model, beside its int8_qat row under the same slot (IF v2 §11.1), rows
  unique by (file_id, precision) and by (slot, precision); precision "qat_fakequant_off"; source {kind "checkpoint",
  path "<run dir>/epoch_ckpts/eNN.pt" relative to the input root, sha256 = the selection's
  winner.checkpoint_sha256}; no converted_artifact_sha256; `cells` [].
- **T-FQ3.** A score-blind binding key on each such row, named by TMAN, holding the run directory (relative), the
  selection {path, sha256 of its DL row}, the telemetry sha256 (the pod's copy-back value) and the winner epoch. They
  feed `--qat-run-dir`, `--qat-selection`, `--qat-selection-sha256` and `--qat-telemetry-sha256`; source.path feeds
  `--checkpoint`, and source.sha256 feeds `--expected-checkpoint-sha256`.
- **T-FQ4.** A validator check that a model's int8_qat and qat_fakequant_off rows name the same run directory,
  selection sha256 and epoch.
- **T-FQ5.** E6-R1 and E6-R2 carry their cut entries and no row; TB5's fq_off requirement is met by the cut (a
  terminal kind, TB10) and never refused. E6-KD's row exists only after its selection's DL row.
- **T-FQ6.** For each row the input root holds the run's qat_telemetry.jsonl, the winner's eNN.pt under
  epoch_ckpts/, and the eval directory's qat_selection.json, plus qat_finalize.json where one exists (FQR2 checks
  it when present).
- **T-FQ7.** tday_binding/1 gives each such artifact a record with scope `fq_off`, cell null, protocol canvas and
  checkpoint_sha256 = source.sha256 (IF v2 §11.4); TB17's post-check compares the row's run.checkpoint_sha256 with
  source.sha256; execution_context_split covers these artifacts (TB1).
- **T-FQ8.** IF§10's script list gains nothing. The src/quant modules this path imports inside functions must be in
  TMAN's evaluator file set, through its import closure if that follows function-level imports, otherwise by name
  from the imports list (2.14).
- **NOTE-FQ-U4-1** (LFQOFF-RULE-3; DL-24 R1-F1, refuted as a lane defect, 2.8). The binding checks the run, the
  selection and the pins it is given; it does not know which of E5 seed 42's two U4 pilot runs won. The loser is
  retained and reported, never served (AM-4a item 3), and has no qat_finalize.json, since finalize refuses it, which
  FQR2 never requires. TMAN's T-FQ items carry the rule: E5 seed 42's qat_fakequant_off row binds the U4 winner's run
  only (T-FQ2's source and T-FQ3's key name the winner's run directory, selection and telemetry, and T-FQ4's check
  holds them to the winner's int8_qat row). The lane changes nothing.

### 2.13 The records proposed (FQR6; verbatim; the docs session writes them, this lane none)

- **EC §5.3** (docs/EVALUATION_CONTRACT.md:403), the precision line becomes:

  ```
  "precision": "fp32 | int8_ptq | int8_qat | qat_fakequant_off"
  ```

  with the note: "qat_fakequant_off: the selected epoch's pre-convert QAT state, every fake-quant and observer
  disabled (AM-4's supplementary score); descriptive; never an inferential comparator".
- **EC §7.6** (a new note; EC's §7 ends at §7.4, :687, at 759440f, so the docs session numbers it):

  > **7.6 The fake-quant-off score (AM-4; lane L-FQOFF).** Precision `qat_fakequant_off` scores a selected QAT
  > model (E5, E6, E6-E2, E6-KD) on its selected epoch's pre-convert state, `<run>/epoch_ckpts/eNN.pt`, with every
  > fake-quant and observer disabled. The source is bound through the selection of record,
  > `<eval>/qat_selection.json`, whose sha256 is pinned to its decision-log row, with the run's telemetry sha256; a
  > `qat_finalize.json` beside the selection is checked when it exists and is never required (E6-KD has none). Every
  > binding refusal comes before any dataset or model exists. It runs on the CPU only, under the canvas protocol on
  > the clean condition, and records run.quant_backend null. It is descriptive: it never selects an epoch or a
  > model, and it is never an inferential comparator (outside Holm).
- **Erratum E-67** (E-66 is the last, docs/lane_specs/errata.md:70; the plan's E-65 is taken): "IC :424, Supplementary
  diagnostic (AM-4): 'lane L-AM4' → 'lane L-FQOFF' (Q2 P29; q2-qat.md:1143, :1165)".
- **DL rows:**
  1. L-FQOFF opened (TL-12): the precision value, the scope and the TEST context;
  2. L-FQOFF implemented and merged (pin), before D;
  3. one row per scoring block (the VAL and TEST fqoff artifacts, their sha256 values, the evaluator commit, T, the
     host and the image).
- **Lane line** (the docs session places it; the plan's line with E6-R1 and E6-R2 removed, FQR1): "L-FQOFF — AM-4's
  fake-quant-off score: every selected QAT model of record (E5 and E6 at each seed, E6-KD if run, AM-23's E6-E2)
  scored on its selected epoch's pre-convert state with every fake-quant and observer disabled, on TEST (clean,
  canvas, TL-1) and VAL; descriptive, never selects, outside Holm; precision qat_fakequant_off (EC §5.3). STOP: a
  selection reading the score, or a converted artifact scored as fqoff."
- **Chapter 4 sentence:** "Each selected QAT model was also scored with fake quantization disabled, on its weights
  after QAT in FP32 (AM-4; descriptive): on clean TEST, E5 X and E6 Y against their converted INT8 scores X′ and Y′,
  which separates the gain from extra fine-tuning from the INT8 adaptation."

### 2.14 The imports list (T-FQ8; MEASURED, imports_fqoff_run_modules in A7 (b))

The repository modules one fqoff run of `scripts/evaluate_model.py`'s main() loads after the module itself is imported
(a sys.modules difference in a fresh interpreter, the writer's git calls stubbed): `configs`, `configs.augment`,
`configs.data`, `configs.model`, `configs.quant`, `src`, `src.data`, `src.data.dataset`, `src.data.original_resolution`,
`src.data.transforms`, `src.distill`, `src.distill.cwd_projection`, `src.distill.export`, `src.distill.features`,
`src.distill.nmf_stream`, `src.distill.segnext_teacher`, `src.distill.teacher`, `src.eval`, `src.eval.adapters`,
`src.eval.artifacts`, `src.eval.eval_runtime`, `src.eval.evaluate`, `src.eval.metrics`, `src.eval.model_loading`,
`src.eval.protocols`, `src.eval.stage_artifacts`, `src.models`, `src.models.student`, `src.quant`,
`src.quant.calibration`, `src.quant.checkpoint`, `src.quant.prepare`, `src.quant.qat`, `src.quant.qat_artifacts`,
`src.quant.qat_select`, `src.quant.qconfig`, `src.quant.stages`, `src.quant.x86_latency`, `src.seeds` (39 modules; the
same list as at C3b). Of src/quant, the run loads `src.quant.calibration`, `src.quant.checkpoint`, `src.quant.prepare`,
`src.quant.qat`, `src.quant.qat_artifacts`, `src.quant.qat_select`, `src.quant.qconfig`, `src.quant.stages`,
`src.quant.x86_latency` (9 modules, with the package `src.quant`), each only through imports made while main() runs (the
validator's and the loader's function-level imports and what those modules import); importing scripts/evaluate_model.py
loads none of them. No src.stats module is loaded.

### 2.15 Slips and tooling (MEASURED)

**SL-1** (`~/.claude/sl1_guard.log`, the guard in log mode; MEASURED 2026-10-28T12:37Z): 4 lines, each a
command I ran: time, tool, class and the command's start as the guard cut it (233 characters a line; the log's tabs
shown as two spaces). No line was added after 05:14:09Z on 28 Oct.

```
2026-10-26T20:46:40Z  Bash  SL1-B1  SCR=/tmp/claude-0/-home-user-plantseg-thesis/a55fc176-d7bc-506e-8948-54f760dc550f/scratchpad; for f in dev/run1.log dev/run2_parity.log dev/run3_full.log dev/measure_cold.log dev/run6_full.log runs_st
2026-10-27T07:41:52Z  Bash  SL1-B1  SCR=/tmp/claude-0/-home-user-plantseg-thesis/a55fc176-d7bc-506e-8948-54f760dc550f/scratchpad; for f in dev/report_at_tip.md draft/status_head.md draft/status_last_reply.md; do printf '%s: ' "$f"; grep
2026-10-27T14:08:10Z  Bash  SL1-B2  git show 521e3df29970b828420f4dcf6159d57b0eeed3da:scripts/smoke_qat_mutations.py | grep -o -E '"smoke_[a-z0-9_]+"' | sort | uniq -c; for f in scripts/smoke_qat_runner.py scripts/smoke_qat_selection.py
2026-10-28T05:14:09Z  Bash  SL1-B2  git show --stat --format='%H %s' 43a074050ca7263dc0c4bce57d575ba045e0327f -- src scripts | cut -c1-200; git show --stat --format='%H %s' c256d1bc3cb1048426132e6a4af60c756957c9e4 -- src scripts | cut -
```

- 2026-10-26T20:46:40Z, SL1-B1: phase 1's report-filling `for` loop over scratch log files with a grep (after phase 1's
  report recorded the file absent at 20:43:53Z).
- 2026-10-27T07:41:52Z, SL1-B1: the 07:42Z STATUS-MD's `for` loop over three scratch files with a grep.
- 2026-10-27T14:08:10Z, SL1-B2: a `git show 521e3df…:scripts/smoke_qat_mutations.py | grep …` followed by a `for`
  loop over three smoke paths with a `git grep … -- $f` (a variable path); L-PROT refused the command and I ran it
  again with literal paths.
- 2026-10-28T05:14:09Z, SL1-B2: slip 8 below.

**Slips** (phase 2; MEASURED):

1. ~13:03Z (27 Oct): a `git show 521e3df:"$f"` in a loop (a variable path) while making DL-24's before-record,
   refused by L-PROT; redone with literal paths.
2. ~13:09Z: `ls -la "$W"` (a variable path, the workflow's transcript folder), refused by L-PROT; redone on the
   literal path.
3. ~14:07Z: `du -sh /tmp`, a traversal of all of /tmp (it may have stat-ed entries other runs created; it read no
   file's content).
4. ~14:08Z: the SL1-B2 command above.
5. ~15:18Z: an inline Python heredoc assembling the 15:15Z STATUS-MD, refused by L-PROT because its text named the
   version-control tool; redone as a script file.
6. A4 (~12:50Z): a cold `measure_opens` run beside smoke_am23_quant (two 4-thread torch processes on 4 cores) slowed
   to its 590 s timeout (rc 124, nothing written); redone alone: 111 s, PASS (117/117).
7. D4 (28 Oct, 04:37Z): the runner's "other venv pythons" check listed the shell that launched it (its command line
   names the venv); no second python ran. A7 was launched from a command line that does not name the venv.
8. 05:14:09Z (28 Oct, during A7): `git show --stat --format='%H %s' <commit> -- src scripts` for C5a-C5d, two
   directories without the exclude pathspec the RULES require, logged by SL-1 as SL1-B2 (above). It printed only
   each commit's own stat lines (the lane's files, none token-named). Every later command named explicit files or
   carried the exclude.
9. ~05:19Z (28 Oct, during A7): an inline Python heredoc editing a scratch draft of this report, refused by L-PROT
   because its text named the version-control tool (as slip 5); redone as a script file. Nothing in the repository
   was touched.

**Found in development and fixed before its commit was pushed:** C5a's c14_finalize_unreadable_file failed under a
long TMPDIR, because the smoke's `refused()` cuts a detail at 240 characters and the case read that detail; C5b reads
the code's full message (`code_of`). C5a and C5b were pushed together, after C5b's dev checks.

**Tooling** (AGENTS.md "What counts as approval"):

- The stop hook (`~/.claude/stop-hook-git-check.sh`) asked again and again to commit and push (uncommitted changes;
  unpushed commits). It is tooling, not approval: reported once in phase 1, not re-raised, and never followed on its
  account. C5a and C5b were pushed under LFQOFF-RULE-3 Q3, and each later commit after its dev checks.
- The session's attribution reminder (commit trailers, another branch): ignored, as the GO says; no commit carries a
  trailer, and only lane/fqoff was pushed.
- **The container restarted** before 04:24:44Z on 28 Oct (MEASURED: `uptime` 0 min; kernel 6.18.44-fc-v180, was
  -v165). The repository (HEAD c256d1b, C5a and C5b local, the scoped status clean), the scratchpad, the venv and
  /tmp survived; no run was in flight.

**The Q2 removal** (LFQOFF-RULE-3 Q2; `$SCR/p2/q2_remove_tmp.py`, a dry run first; its log, verbatim):

```
2026-10-28T04:27:06Z Q2 BEFORE: 433 directories /tmp/smoke_fqoff_* to remove (directories, not links, uid 0, changed 2026-10-26T19:48:31Z to 2026-10-27T14:58:49Z), 93108 files, 10.52 GiB (unique inodes); free 10.00 GiB
2026-10-28T04:27:06Z Q2 DRY RUN: nothing removed
2026-10-28T04:27:17Z Q2 BEFORE: 433 directories /tmp/smoke_fqoff_* to remove (directories, not links, uid 0, changed 2026-10-26T19:48:31Z to 2026-10-27T14:58:49Z), 93108 files, 10.52 GiB (unique inodes); free 10.00 GiB
2026-10-28T04:27:25Z Q2 AFTER: removed 433 of 433; 0 failed; /tmp/smoke_fqoff_* left 0; free 20.05 GiB
```

Only /tmp/smoke_fqoff_* entries were candidates: directly under /tmp, a directory and not a link (lstat), owned by this
user (uid 0), changed after this session began (2026-10-26T17:00:00Z), no name inside holding the token. Nothing else
was touched. Before A7 /tmp held 10.99 GiB in 108 entries with 18.27 GiB free; after it, 14.67 GiB in 316 entries with
14.43 GiB free (MEASURED, $SCR/p2/tmp_usage.log). The merged smokes and harnesses left 208 entries, about 3.7 GiB, among
them 128 more smoke_ptq_record folders from LQP's harness jobs (132 in all, 1.49 GiB). Two /tmp/smoke_fqoff_* folders
remain, one from each run of the lane harness since the removal (C5c's dev run and A7 (c)), FQ18's mutant job by design.
Nothing was removed after the Q2 removal.

### 2.16 Open items

1. **NOTE-FQ-U4-1 (TMAN).** E5 seed 42's qat_fakequant_off row must bind the U4 winner's run only (2.12). The lane's
   binding cannot tell the pilots apart; TMAN's T-FQ2-T-FQ4 carry the rule (LFQOFF-RULE-3).
2. **Uncoded refusals that fail closed.** An OSError on reading the selection (:390), the committed rules
   file (:406) or the eNN.pt (:487) of src/eval/stage_artifacts.py, and a hand-edited
   telemetry row's AttributeError or TypeError inside read_run_record, escape without an fqoff code. Each stops the
   run before any dataset or model exists. A code for them needs a ruling: FQR12 names "unreadable" for code 14 alone
   (DL-24 R1-F2, R1-F3; 2.5's mapping).
3. **Temp directories other smokes leave.** The merged smokes and harnesses that A7 runs leave theirs under /tmp
   (2.15 gives what A7 added), and FQ18's mutant job leaves one /tmp/smoke_fqoff_* per run of the lane harness by
   design. Q2's approval covered the 433 directories removed before A7 only, so nothing else was removed.
4. **The scoring runs** (FQR5): this lane ran neither the VAL block nor the TEST scoring; 2.11 gives both commands for
   the orchestrator, with TMAN's T-FQ1-T-FQ8 (2.12).
5. **The records** (FQR6): EC §5.3, the EC §7.6 note, erratum E-67, the DL rows, the lane line and the Chapter 4
   sentence are proposed verbatim (2.13) and written by the docs session; the final merge is Ice's block (FQR16).
