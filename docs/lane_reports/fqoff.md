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
