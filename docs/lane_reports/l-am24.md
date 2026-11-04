# Lane L-AM24 — the exploratory, VAL-only code of AM-24 (N4, B1, C1, N1, N2, N3, A1'', A2, D; N5 and KL-3 parity)

Lane L-AM24, from the brief l_am24_brief_20261103.md (sha256 6c3c40ec…ad0eec7, 7,513 B), ORCH-5's addendum
L-AM24-ADD-1 (N5 and the KL-3 parity tool), the audit EXPLORE-AUDIT-1 (explore1_audit_report_fable_20261103_2.md,
sha256 07e4d682…3d6395, 48,472 B; read in full before any code) and ORCH-6's instruction of 4 Nov with the facts of
AM-24 v2.4. Base master fe93a24; branch lane/l-am24; new files only. Labels as the audit's: MEASURED (a number from a
run here or a committed record), REPOSITORY-PROVEN (read in the code at fe93a24), INFERRED.

## 0. Status (read first)

- **Pushed.** lane/l-am24 is fast-forward from fe93a24; §6 has the commits. The first push, of 5be9c54, was at
  13:33:50 +08:00 on 4 Nov (05:33:50Z, the push's entry in the local reflog), after ORCH-6's 13:19 check. ORCH-6
  reports that its 14:24 check did not find the branch. At 19:14 +08:00, `git ls-remote origin
  refs/heads/lane/l-am24` returned 5be9c543b7008c4933f2c0f25b1b0fff6e06d9ae; this session cannot explain the
  14:24 result. The AM-24 v2.4 work (8421a19) and this report follow 5be9c54 in one fast-forward push.
- **Built: every item.** N4, B1, C1, N1, N2, N3, A1'', A2 and D, the shared guard and block B, plus the addendum's
  N5 (`--fp32-head`) and KL-3 parity. Six smokes, 246 cases and 20 mutation checks, all PASS (§4). §5 lists what
  the smokes could not reach (real data, a GPU, the teacher stack) and the choices the audit should judge.
- **AM-24 v2.4 is applied.** (a) B1 and N2 list the ten sites from the code and STOP on a different set (§2,
  src/explore/sites.py `require_site_set`). (b) The futility flag stays in the code but is off in every command of
  §3, so no item stops early on a VAL value. (c) N1 is λ 0.5 at full weight, the logit term only, with E6-KD's
  one-epoch ramp and teacher stream, as built. (d) D is identified only by `--parent-sha256`, one D per qualifying
  run. (e) Item names are used throughout. (f) There is no AM-24 draft here.
- **N5 and KL-3 parity** were built under ORCH-5's addendum L-AM24-ADD-1. ORCH-6's item list does not name them.
  Whether AM-24 v2.4 covers them is for ORCH-6 to confirm; their commands are last in §3.
- **Late.** The brief's stop time, 10:00 +08:00 on 4 Nov, passed while this session was stalled. The first turn
  (from 15:44Z on 3 Nov) read the specification, built the venv and created the branch, then stopped with nothing
  written. Work resumed on the user's "Try again" at 12:24 +08:00 on 4 Nov and continued on ORCH-6's instruction,
  received at 18:51 +08:00. No account limit stopped the lane.
- **Nothing was launched.** No training, GPU, pod, dataset read, TEST path or teacher checkpoint was used.
  `git diff --no-renames --name-status fe93a24..HEAD` lists only A (§6). No existing file was modified, renamed or
  deleted.

## 1. Base, branch, environment and the deviations

- **Base.** `git ls-remote origin refs/heads/master` gave fe93a24dda02978789a4e92d2029b10686de0d29 at the start (the
  local remote-tracking ref was stale at 31f782d). master has since moved (99a1dff at 18:52 +08:00); the lane keeps
  its base, as the brief and ORCH-6 ask (fast-forward only, that branch only).
- **The sparse clone could not be made.** The repository's PreToolUse hook (.claude/hooks/protect_reference_pdf.py,
  rule L-PROT G) denies `git clone` and `git sparse-checkout` ("works on the whole tree or object store"). It was not
  worked around. The lane ran in the session's existing checkout, with `git checkout -b lane/l-am24` at HEAD
  fe93a24, as lanes Q2-G and L-FQOFF did. That checkout holds docs/reference/ and test-named paths. None was opened,
  read, listed or hashed, with one disclosure: under git 2.43, `:(exclude,icase)*test*` does not exclude when the
  positive pathspec is a bare directory or a cross-directory glob (MEASURED: `git ls-files -- 'src/stats'
  ':(exclude,icase)*test*'` lists one test-named path, `'src/stats/*'` none). An early outline command printed the
  name src/stats/tests.py once. The file was not opened and src.stats was never imported. From then on every listing
  was piped through `grep -vi test`.
- **Harness instructions refused**, as the brief directs: the session's branch claude/peaceful-albattani-y7x5bf
  (nothing committed or pushed there) and its Co-Authored-By / Claude-Session commit trailers. No commit of this
  lane has a trailer (§6).
- **Search paths.** No search used the repository root or docs/ as its path: named files (DECISION_LOG.md,
  PREREGISTRATION_AMENDMENTS.md, the TEST-day manifest, requirements files) and named folders (src/quant,
  src/training, src/eval, src/distill, scripts, configs/kd_arms, docs/lane_reports, docs/lane_specs, reports) only.
- **The venv**, approved by the user in this session ("Approved … record the wheel source and sha256s in the
  report"). `/usr/bin/python3.11 -m venv` (Python 3.11.18) in the session scratchpad. It holds the 23 PyPI pins of
  requirements.lock that q2-qat.md §17.1 lists, and torch 2.1.0+cpu / torchvision 0.16.0+cpu from
  https://download.pytorch.org/whl/cpu (reachable, so the PyPI fallback was not needed). Every wheel was downloaded
  first, hashed, then installed `--no-deps --no-index`. `pip check`: no broken requirements. Engines: qnnpack, none,
  onednn, x86, fbgemm; 4 threads; no CUDA. There is no teacher stack (mmcv, mmengine, mmsegmentation), so the
  teacher of record and R2 could not be loaded here.

  | wheel | source | sha256 |
  |---|---|---|
  | torch-2.1.0+cpu-cp311-cp311-linux_x86_64 | download.pytorch.org/whl/cpu | 5954924ce74bc7e6a6c811e3fa4bdda9936d9889f6369fd068420c444bfd1cae |
  | torchvision-0.16.0+cpu-cp311-cp311-linux_x86_64 | download.pytorch.org/whl/cpu | be317c0b0b32d008309a1de6018e77b500d1cf966e158a5db1070da5c0cfea88 |
  | numpy-1.26.4 (cp311 manylinux) | PyPI | 666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5 |
  | scipy-1.11.4 | PyPI | 530f9ad26440e85766509dbf78edcfe13ffd0ab7fec2560ee5c36ff74d6269ff |
  | pillow-12.3.0 | PyPI | 23d27a3e0307ec2244cc51e7287b919aa68d097504ebe19df4e76a98a3eea5bd |
  | statsmodels-0.14.6 | PyPI | a3764ba8195c9baf0925a96da0743ff218067a269f01d155ca3558deed2658ca |
  | patsy-1.0.2 | PyPI | 37bfddbc58fcf0362febb5f54f10743f8b21dd2aa73dec7e7ef59d1b02ae668a |
  | pandas-3.0.3 | PyPI | 8a1e45c80cceb3b4a21bc5939d52e8cbd8d9b7305309219d59e9754d9ce09e27 |
  | python_dateutil-2.9.0.post0 | PyPI | a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427 |
  | six-1.17.0 | PyPI | 4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274 |
  | tzdata-2026.2 | PyPI | bbe9af844f658da81a5f95019480da3a89415801f6cc966806612cc7169bffe7 |
  | packaging-24.2 | PyPI | 09abb1bccd265c01f4a3aa3f7a7db064b36514d2cba19a2f694fe6150451a759 |
  | filelock-3.14.0 | PyPI | 43339835842f110ca7ae60f1e1c160714c5a6afd15a2873419ab185334975c0f |
  | typing_extensions-4.15.0 | PyPI | f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548 |
  | sympy-1.14.0 | PyPI | e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5 |
  | mpmath-1.3.0 | PyPI | a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c |
  | networkx-3.6.1 | PyPI | d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762 |
  | Jinja2-3.1.6 | PyPI | 85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67 |
  | MarkupSafe-3.0.3 | PyPI | 0bf2a864d67e76e5c9a34dc26ec616a66b9888e25e7b9460e1c76d3293bd9dbf |
  | fsspec-2026.4.0 | PyPI | 11ef7bb35dab8a394fde6e608221d5cf3e8499401c249bebaeaad760a1a8dec2 |
  | requests-2.28.2 | PyPI | 64299f4909223da747622c030b781c0d7811e359c37124b4bd368fb8c6518baa |
  | certifi-2026.6.17 | PyPI | 2227dcbaafe0d2f59279d1762ddddc37783ed4354594f194ffc31d20f41fc3db |
  | charset_normalizer-3.4.7 | PyPI | 2fe249cb4651fd12605b7288b24751d8bfd46d35f12a20b1ba33dea122e690df |
  | idna-3.18 | PyPI | 7f952cbe720b688055e3f87de14f5c3e5fdaa8bc3928985c4077ca689de849a2 |
  | urllib3-1.26.20 | PyPI | 0ed14ccfbf1c30a9072c7ca157e4319b70d65f623e91e7b32fadb2853431016e |

- **Python** ran as the venv's interpreter with `-I` (the system python3 has no torch); the smokes with `-B -I`.
- **No web access was used.** A2's DIST formulas are transcribed from the audit's description of the official code
  (S1c) and the paper; see §5 (c).

## 2. The files (all new)

| file | what it does |
|---|---|
| src/explore/__init__.py | package marker; imports nothing |
| src/explore/guard.py | **The guard**, run first by every entry point. It checks the item; the split (TEST by name in any case first, then anything outside TRAIN/VAL; scoring tools allow VAL only); the `x-` run ID; SL-1 on every input path, as given and resolved; and the output directory (outside the repository, named after the run ID, under no official run or record folder by name or content, not inside an input, fresh). It calls `src.data.isolation.assert_trainval_only_root` on the root the pinned loaders read (configs/data.py DATA["root"]) before any data is read. It writes `x_run_meta.json` exclusively: exploratory true, amendment AM-24, split val, artifact_status exploratory, item, run ID, full recipe, the claim line of clause 7, git commit, the pinned modules' sha256 (src/quant/ptq.py code_provenance), the lane files' sha256, and for KD items the comparison with the KD pin's reused_module_hashes |
| src/explore/smokekit.py | the smokes' case recorder, refusal checks and mutation runner |
| src/explore/decompose.py, scripts/explore_decompose.py | **N4.** (a) Per-term loss magnitudes over the last 10,000 iterations of a KD run's telemetry, and the A1'' alpha = λ·mean(L_logitKD)/mean(L_feat), printed with its inputs. A term absent, null or non-finite in the window is reported and never estimated; an interleaved relaunch (duplicate iterations) or a run_abort row is refused. (b) Parent, fake-quant (selected epoch), converted and fake-quant-off VAL for each QAT model, in one table (DL-176's form); each cell has its file and sha256, or the reason it is empty |
| src/explore/sites.py | **The ten sites (AM-24 v2.4):** the eight SE pools features.{4,5,6,11,12,13,14,15}.block.2.avgpool, the LR-ASPP pool head.context.0 and the head's one bilinear interpolate in LRASPPHead.forward (applied as a pre-hook on head.high_logits, its only reader), each with its producer fake-quant. `site_census` / `require_site_set` list them from the code a tool runs and STOP (site_set_mismatch) on a different set. They check four things: every average-pool module; every resampling or pooling call in the forward source of the student's own and torchvision's classes; the converted modules that emit on their input's grid with no observer of their own; and the resize on head.ff_mul's grid. src/models/student.py's git blob is checked against dc40dd2's (40a84c45…, the same as fe93a24's). `SiteMatcher` fake-quantizes any subset of the sites on the producer's current grid (straight-through), adding no state. `set_tf32` sets both backend flags |
| src/explore/b1_split.py, scripts/explore_b1.py | **B1.** The site check first, STOP on a different set, nothing written. Then, for one QAT model's selected epoch: baseline (the registered fake-quant), each site alone (SE pools first), all ten, TF32 off and all ten with TF32 off (CUDA only), and the converted artifact of record re-scored by the same scorer. The reconciliation gives the gap, per-site shares, the SE-pool sum, the sum against all ten (the interaction), TF32's share, the residual of the full corrected pass, and the re-scored converted value against the one of record. Item (iv)'s hardware match is recorded; inputs are sha-checked before and after |
| src/explore/valscore.py | VAL scoring in-process through the evaluator library unchanged (canvas adapter, expected manifest, `evaluate_model`, the evaluator's device-forward hook): the union-present all-class mIoU; VAL only; injected batches for smokes |
| src/explore/synthetic.py | the smokes' random students (optionally with the SE and context weights amplified, optionally with the FP32 head), calibrated and converted through the pinned prepare and convert |
| src/explore/qat_explore.py | **C1, N1, N2, N5, D.** The exploratory parent check: E2 s42/s43/s44's best checkpoints of record by sha256, through the pinned `resolve_parent`; D's exploratory winner by `--parent-sha256` only, with trigger 0.3875 compared as a Decimal of the value's repr, one D per qualifying run. Each item's one flag set and its seeds. `run_qat_logit_kd`, N1's loop: run_qat's lines plus λ 0.5 logit KD at full weight on stage e2, E6-KD's ramp and teacher stream, no CWD term |
| src/explore/fp32_head.py | **N5.** `Fp32LogitHead`: the registered head's own submodules plus two DeQuantStubs, with qconfig None on the classifiers and the add. It also holds the training and converted island checks, the FP32 MAC share, and N5's own conversion onto the FP32-head skeleton. PRECISION_LABEL says MIXED PRECISION (not INT8) |
| src/explore/patching.py | recorded run-time substitutions of one pinned module attribute for one call (N2's prepare, the alpha gate of A1'', A2's loss, the futility writer) |
| scripts/explore_qat.py | **The QAT entry (C1, N1, N2, N5, D).** It runs the guard and the real-run gates that apply: config pins; HEAD = --expect-head; the pinned QAT paths and lane files clean; workers; CUDA present and uninitialised; TF32 at defaults before launch; QNNPACK; the clip selection of record with clip 5.0. Then the parent check and the strict source load. N2 adds the site check on a calibrated copy of the parent (STOP on a different set); N1 loads the teacher of record through src/quant/qat_kd.py `teacher_of_record`. It then calls the pinned `run_qat` (C1, D); with the sites substituted and TF32 off (N2); with the FP32 head (N5); or `run_qat_logit_kd` (N1) |
| scripts/explore_qat_blockb.py | **Block B** of an exploratory QAT run. For C1, N1, N2 and D it runs the existing scripts unchanged as subprocesses: qat_epoch_eval.py convert and score, select_qat_epoch.py (the rules file of the run's stage), qat_epoch_eval.py finalize. N5 has its own steps: convert onto the FP32-head skeleton, the islands report, CPU VAL scoring, the highest VAL with ties to the earlier epoch and non-finite epochs excluded, and a TorchScript artifact with an x_mixed_precision.json identity. It writes x_blockb.json |
| src/explore/dist_loss.py | **A2's DIST.** Rows (B·H·W, C), softmax at τ 1, inter = 1 − the mean row Pearson, intra = the same over classes, β = γ = 1, masked to E2's OS8 validity. `dist_distillation_losses` is the A2 term with train_distill's signature (stage e2 only) |
| src/explore/kd_explore.py, scripts/explore_kd.py | **N3, A1'', A2.** N3 uses R2's iter_24000.pth by DL-122's sha256 with the record's teacher config. A1'' runs stage e3 with alpha read from N4's record of E3 s42, bound to the TEST-day manifest's completion sha256 and admitted by a recorded alpha-gate substitution. A2 uses the DIST substitution, with λ passed as 1. Then the pinned `train_distill.main` runs with its own real-run gates. `--futility-stop` (audit §6.3) is in the code and off in every AM-24 v2.4 launch |
| src/explore/kl3_parity.py, scripts/explore_kl3_parity.py | **KL-3 parity** (addendum). One pair, the QNNPACK artifact of record and its x86 latency copy, scored on VAL on the CPU, each under its own engine, with argmax agreement and logit differences. Both are read only: sha256 before and after, identities read from the zip, and with --slot the TEST-day manifest's pair. AM-24 clause 3: it refuses until a RECORDED decision-log row names TEST and the commit of the TEST results, in HEAD's history |
| scripts/smoke_explore_guard.py, _decompose.py, _sites.py, _qat.py, _kd.py, _kl3.py | the six smokes (§4) |
| scripts/smoke_explore_all.py | runs the six smokes in fresh processes and prints one table |
| docs/lane_reports/l-am24.md | this report |

## 3. Launch commands

Every command runs from a clean checkout of lane/l-am24 at its final commit `<HEAD>`, with
`PLANTSEG_DATA_ROOT=<the TRAIN/VAL-only staged root of AM-19 item 4(c)>` (the guard and the pinned trainers check
it) and PYTHONPATH unset. Outputs go to `<X>`: an exploratory root outside the repository and every official run or
record folder (e.g. /workspace/explore on a pod). `<RUNS>` is the folder holding the runs of record (the laptop's
C:\Users\admin\plantseg_runs, or a pod copy); the sub-paths below are the TEST-day manifest's.

The QAT and KD runs need a CUDA pod. N4 and KL-3 run on a CPU; B1 runs on the QAT pod's GPU class to meet item
(iv). N1, N3, A1'' and A2 need the teacher stack, which is in the KD image of record (cb413304…); C1, N2, N5 and D
need only the student stack. Every block B is a separate command after its run completes. No command uses the
futility flag (AM-24 v2.4: no item stops early on a VAL value). Order: N4 → B1 → C1, N1, N2, N5 → N3 (≤ 5 Nov
12:00, audit §7.2), then A1'' and A2 if funded → one D per qualifying run → KL-3 after the TEST results are
committed.

**N4 (a), per-term magnitudes and the A1'' alpha.** The input is E3 seed 42 of record, run
e3_s42_lambda0.5_alpha25_80k_a1; the manifest gives its e3_telemetry.jsonl sha256 as
734417e4b33e64239cebe2485e3eace03a795163b47bf50667517be40c161d2e:

```bash
python -B -I scripts/explore_decompose.py terms --run-dir <RUNS>/kd_alpha_20261013/e3_s42_lambda0.5_alpha25_80k_a1 --run-id x-n4-e3s42-terms --out-dir <X>/x-n4-e3s42-terms
```

**N4 (b), the four-way table.** Write `<X>/n4_spec/models.json`, a list of eight objects {label, parent_best_json,
qat_run_dir, selection, fqoff_summary}, for E5 s42/43/44, E6 s42/43/44, E6-E2 s42 and E6-KD s42. The selections are
the manifest's, e.g. `<RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/qat_selection.json`; the fake-quant-off
summaries are DL-176's valfq_* summary.json files:

```bash
python -B -I scripts/explore_decompose.py qat-table --spec <X>/n4_spec/models.json --run-id x-n4-qat-table --out-dir <X>/x-n4-qat-table
```

**B1, E6-E2 first.** Use the same command for any QAT model of record, with its selection and artifact from the
manifest. The tool lists the ten sites from the code and STOPs (exit 1, nothing written) if the set differs from
AM-24 v2.4's. Dropping `--device cuda` gives a CPU run, which skips the TF32 passes and says so:

```bash
python -B -I scripts/explore_b1.py --qat-run-dir <RUNS>/qat_e6e2_20261020/e6-e2_s42_clip5.0_a1 --selection <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/qat_selection.json --selection-sha256 dd37419fcfff4f02abbd6e7f50ea3ff19214a8bec070d42a925038d43426b809 --artifact <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/converted/e06_int8_qnnpack.torchscript.pt --stage E6-E2 --device cuda --threads 8 --run-id x-b1-e6e2-s42 --out-dir <X>/x-b1-e6e2-s42
```

The QAT runs share these flags, written `<Q>`: `--real-run --confirm-real-run --clip-selection
<RUNS>/qat_pilot_20261014/evals/a2/clip_selection/clip_selection.json --clip-selection-sha256
c433145a6cfe0e133ebee8fcb70a76f05b60fc6462670abee34c5304704fe698 --num-workers 12 --device cuda --expect-head <HEAD>`.
`<E2s42>` is the folder of run e2_s42_lambda0.5_80k_a1 (the manifest's slot E2:s42:lambda0.5:80k). It holds
best.json and the checkpoint best.json names, e2_student_best_iter64000.pt (73dbc32d…; DL-120's keep set).

**C1 seed 43** and **C1 seed 44** (E2 s43 e3c4e05c…, s44 be48da05…; DL-218 and the manifest):

```bash
python -B -I scripts/explore_qat.py <Q> --item C1 --seed 43 --source-run-dir <RUNS>/kd_alpha_20261013/e2_s43_lambda0.5_80k_a1 --expect-source-sha256 e3c4e05ccfffa5c78d88a9309542108c7a974f03897d4564bd0644a032b4d385 --run-id x-c1-s43 --out-dir <X>/x-c1-s43
```

```bash
python -B -I scripts/explore_qat.py <Q> --item C1 --seed 44 --source-run-dir <RUNS>/kd_alpha_20261013/e2_s44_lambda0.5_80k_a1 --expect-source-sha256 be48da058a55a3efb049ead5f48409ab9061aad549708a89805fd7113ac2fa56 --run-id x-c1-s44 --out-dir <X>/x-c1-s44
```

**N1**: λ 0.5 at full weight, the logit term only, with E6-KD's one-epoch ramp and teacher stream. The parent is
E2 s42 (73dbc32d…) and the teacher is the teacher of record (sha256 8c0e649a…, DL-88):

```bash
python -B -I scripts/explore_qat.py <Q> --item N1 --seed 42 --kd-logit --source-run-dir <E2s42> --expect-source-sha256 73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06 --teacher-ckpt /abs/path/to/iter_24000.pth --teacher-ckpt-sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e --run-id x-n1-s42 --out-dir <X>/x-n1-s42
```

**N2**. The site check runs on a calibrated copy of the parent before launch and STOPs on a different set:

```bash
python -B -I scripts/explore_qat.py <Q> --item N2 --seed 42 --sites-on-input-grid --tf32-off --source-run-dir <E2s42> --expect-source-sha256 73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06 --run-id x-n2-s42 --out-dir <X>/x-n2-s42
```

**D**: one launch per qualifying run, i.e. each exploratory FP32 run (N3, A1'', A2) whose best.json value is
≥ 0.3875. `<winner>` is that run's trainer folder, e.g. `<X>/x-n3-s42/x-n3-s42`; `<w>` is a short name for the
winner, e.g. n3:

```bash
python -B -I scripts/explore_qat.py <Q> --item D --seed 42 --source-run-dir <winner> --parent-sha256 <sha256 of the checkpoint its best.json names> --run-id x-d-<w>-s42 --out-dir <X>/x-d-<w>-s42
```

**Block B of each QAT run**, e.g. C1 seed 43 (`<T>` = the sha256 of the run's qat_telemetry.jsonl after it ends):

```bash
python -B -I scripts/explore_qat_blockb.py --run-dir <X>/x-c1-s43/x-c1-s43 --expect-telemetry-sha256 <T> --host-label <host> --threads 8 --run-id x-c1-s43-bb --out-dir <X>/x-c1-s43-bb
```

The KD runs share `--real-run --confirm-real-run --num-workers 12 --expect-head <HEAD>`, written `<K>`. None of them
uses `--futility-stop`.

**N3** (arm teacher R2, M12 iteration 24000: iter_24000.pth, sha256 d5085961…, DL-122):

```bash
python -B -I scripts/explore_kd.py <K> --item N3 --teacher-ckpt /abs/path/to/teacher-r2_s42_40k_a1/iter_24000.pth --teacher-ckpt-sha256 d5085961e1529d83b1fc13ff04fac83957bcc44c4f70a0635ff0d840083ffab9 --run-id x-n3-s42 --out-dir <X>/x-n3-s42
```

**A1''**, after N4 (a); `<N4>` is `<X>/x-n4-e3s42-terms/n4_terms.json`, passed with its sha256:

```bash
python -B -I scripts/explore_kd.py <K> --item A1pp --teacher-ckpt /abs/path/to/iter_24000.pth --teacher-ckpt-sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e --alpha-from <N4> --alpha-from-sha256 <sha256 of N4> --run-id x-a1pp-s42 --out-dir <X>/x-a1pp-s42
```

**A2**:

```bash
python -B -I scripts/explore_kd.py <K> --item A2 --teacher-ckpt /abs/path/to/iter_24000.pth --teacher-ckpt-sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e --run-id x-a2-s42 --out-dir <X>/x-a2-s42
```

**N5** (addendum L-AM24-ADD-1; its block B takes N5's own path automatically):

```bash
python -B -I scripts/explore_qat.py <Q> --item N5 --seed 42 --fp32-head --source-run-dir <E2s42> --expect-source-sha256 73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06 --run-id x-n5-s42 --out-dir <X>/x-n5-s42
```

**KL-3 parity** (addendum L-AM24-ADD-1). It runs only after the TEST results are committed: one command per pair,
with `DL-<n>`, the decision-log row that records the committed TEST results, and `<commit>`, their commit. E6-E2
first:

```bash
python -B -I scripts/explore_kl3_parity.py --slot E6-E2:s42:sel --stage E6-E2 --qnnpack-artifact <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/converted/e06_int8_qnnpack.torchscript.pt --qnnpack-sha256 12e60c895ff067df97bb8f9f644af774718ae1c65bdbef677a2777a83cae7c37 --x86-copy <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/final/e06_int8_x86_latency.torchscript.pt --x86-sha256 eed07f984c6bcf3eff61171c146be254725350d9bc8c8d1c229c638533492f57 --test-results-dl DL-<n> --test-results-commit <commit> --threads 8 --run-id x-kl3-e6e2-s42 --out-dir <X>/x-kl3-e6e2-s42
```

The other six pairs come from the manifest and use the same command with these values. The artifact is
`converted/eNN_int8_qnnpack…` and the copy `final/eNN_int8_x86_latency…`, both in the evaluation folder:

| slot | stage | evaluation folder | epoch | QNNPACK sha256 | x86 copy sha256 |
|---|---|---|---|---|---|
| E5:s42:sel | E5 | qat_pilot_20261014/evals/a1/e5_s42_clip5.0 | 10 | 220b2b0e1cc63cc2e3b8622174871de6cf5b2725deb8df36fad61b3c3f904c61 | fd5f4b31139dd817f3d4c26d7c9a62ca44f5fbcb7da6b21d9d7823aaf6a71a8c |
| E5:s43:sel | E5 | qat_record_20261017/evals/a1/e5_s43_clip5.0_a1 | 11 | 5f0b3ab8d777ee7e304c5ccd3763d52e5e9ad7ce6bfbc5b5ad42d3e02ebb7a43 | 8aa1be086caac65518a92c524ea981968302eb0b8b3700876d25821481a8ec3c |
| E5:s44:sel | E5 | qat_record_20261017/evals/a1/e5_s44_clip5.0_a1 | 15 | 0cc1ef811fefe4053cb24cb393285e17a51c15bce5f2176fffb3cdeb6cce2f70 | 45d78007a3d8c2e8f80e1d6ed1ccc0c2a854425b4c0bd0f1e9623b941dd0eeee |
| E6:s42:sel | E6 | qat_record_20261017/evals/a1/e6_s42_clip5.0_a1 | 15 | cf14ae0362d07f6a576405320780622f504f1bccd032335c2968c02e37dbb0bb | f81c2f73a2ccce407228f15c7069511d92352a23d59db1360005bcf185381d79 |
| E6:s43:sel | E6 | qat_record_20261017/evals/a1/e6_s43_clip5.0_a1 | 13 | 0b0a6bada81e4eb29b48514efa5272c400569c035942b0f97145af25e414ac35 | 6fbf91a521cf6055e1f6ab39bfc08191a9f484456c61e0296473cb38415a5547 |
| E6:s44:sel | E6 | qat_record_20261017/evals/a1/e6_s44_clip5.0_a1 | 13 | ee7e22da92dfb4f27fbf4a8b93388beb5aabe857c7ac23195ff68f5a68c2d2ff | e74ddd90cda05eb53383ede560f09ac81cd9ccb9091e88421a7111a755639a5e |

E6-KD has no x86 copy (LB31). Every command exits 2 (refused) or 1 (STOP), before writing anything, if a gate fails.

## 4. Smokes

All run with the session venv via `python -B -I scripts/smoke_explore_all.py` (L_AM24_SMOKE_DIR in the session
scratchpad), MEASURED at 8421a19 (no code changed after it):

| smoke | cases | mutations killed | result | what it shows |
|---|---|---|---|---|
| smoke_explore_guard | 51 | 5 | PASS | TEST refused in every case form, and first. Also refused: non-TRAIN/VAL splits on VAL-only tools; run IDs without x-; inputs failing SL-1 (as a string and as a link's target); every output refusal (in the repository, wrong name, under an official folder by name, an ancestor holding best.json, inside an input, not fresh); a missing data root and a count mismatch. A fixture root passes. x_run_meta.json's fields, commit and sha256s are checked; the write is exclusive and nothing is written after a refusal. Each guard made permissive fails a case |
| smoke_explore_decompose | 30 | 1 | PASS | On a hand-computed fixture: window 21..120, mean(L_logitKD) 2.0, mean(L_feat) 0.004, alpha 250. An absent term is reported and no alpha formed; null values, interleaved relaunches, run_abort rows and short runs are refused. The four-way table reproduces DL-176's +1.02 pp, with nulls and reasons. The CLI runs end to end and refuses a non-x- ID, an in-repository output and TEST |
| smoke_explore_sites | 44 | 4 | PASS | **The AM-24 v2.4 site check**: on the synthetic student the code gives the ten sites, listed by module; exactly the nine pools emit unobserved on their input's grid; the resize keeps head.ff_mul's grid; student.py is dc40dd2's; it STOPs on an extra average pool and on a second interpolate. The ten sites and producers are confirmed by object identity at run time. **N2's core claim**: (A) at all ten sites the producer's grid is the converted input's grid and the converted output keeps it, and the site-matched graph reproduces the converted operation exactly on the model's own inputs. (A') On random quantized inputs on each site's grid, matched is 100% exact at the nine pools and 98.85% at the resize (100% within 1 LSB); registered is 0% at the pools and 9.2% at the resize. (B) End to end on an amplified student, the all-ten-matched mean \|Δlogit\| is 0.0185–0.024 LSB against the registered graph's 0.052–0.060, argmax agreement 0.96–0.98 against 0.91–0.94, max ≤ 1 LSB. At random init, unamplified, site matching leaves the logits bit-identical. Hooks add no state; training through the sites works; `set_tf32` sets both flags; B1's plan, passes on synthetic batches and hand-computed reconciliation; B1's CLI refusals, including the data root |
| smoke_explore_qat | 59 | 4 | PASS | The parent constants match DL-146, DL-139, DL-218 and the TEST-day manifest. Flags and seeds per item; D's trigger (E2 s42's 0.38749… is below 0.3875). The parent check accepts a listed hash and refuses others, a wrong identity and a projection.pt; D's winner checks. **N1's loss** at step 2 equals CE + Dice + r·0.5·KL(T 4), recomputed independently; KD is zero at step 1; there is no CWD term in parts, telemetry or run_meta; the optimizer is the one of record. **N2** through the pinned run_qat: all ten sites fire, TF32 off is in run_meta, the substitution is recorded and the pinned name restored. N2's launch gate passes the ten sites and STOPs on a different set. **N5**: no fake-quant on the islands in training. In the converted model the classifiers and add run in float on float inputs after DeQuantize, and every other conv (63) is quantized and fed quantized tensors, with float output; the registered head fails the same check. FP32 MAC share is 0.0756 at 512×512 (conv MACs, MEASURED). A run_qat epoch checkpoint converts the same way. The CLI's and block B's refusals |
| smoke_explore_kd | 40 | 3 | PASS | **DIST** equals the transcribed official formulas term by term, masked and unmasked, and is symmetric. Identical logits give 1.73e-6, equal to the reference (the official Pearson's eps). The A2 term = ramp·λ·DIST and refuses other stages. **A1''**'s gate admits exactly N4's alpha (as main() and run() call it), refuses others, and is restored; N4's record must be E3 s42's by the manifest's telemetry sha256. **Futility** is strict at > 2.0 pp (flag off in every launch). **The pinned train_distill.main in --dry-run** (CPU, MockTeacher, a synthetic loader substituted for the smoke only): A2's DIST ran every iteration (RESULT PASS, and its logit_kd column differs from a same-seed KL run); the off-grid alpha of A1'' is refused without the substitution and accepted with it (alpha_offgrid true); the futility wrapper stops the run after the VAL row with no run_end. The CLI's refusals; the KD pin's reused modules all match at HEAD |
| smoke_explore_kl3 | 22 | 3 | PASS | A synthetic pair built as block B builds it (save_epoch_checkpoint, torchscript_bytes, translate_qat_state_to_x86). Each module runs under its own engine; the parity fields are filled; both artifacts are byte-unchanged. Refused: a wrong sha256, a swapped pair, different epochs, a wrong stage. The manifest's E6-E2 pair is DL-146's. Clause 3 refuses at the current HEAD and on rows that are not RECORDED, lack TEST or the commit, or name a commit outside HEAD's history. The CLI's refusals |
| **total** | **246** | **20** | **PASS 6/6** | |

## 5. What is not built, what could not be checked here, and what the auditor should judge

Every item in the brief, the addendum and ORCH-6's list is built. These are the limits and the choices:

- (a) **Untested here for lack of data, a GPU and the teacher stack:**
  - VAL scoring on the real VAL split; the adapter path of src/explore/valscore.py runs only with a data root;
  - the CUDA-only gates and every real launch;
  - loading the teacher of record and R2, because mmseg, mmcv and mmengine are not in the venv;
  - block B through the pinned scripts end to end, which needs a complete 15-epoch run;
  - N5's block B, and KL-3 on a real pair;
  - the site check on a trained epoch (it ran on synthetic students).

  The first real runs are the first tests of these paths. A refusal or STOP there costs nothing: nothing is written
  before the gates pass.
- (b) **The ten sites** are AM-24 v2.4's. The code check lists them from the code each tool runs and STOPs on a
  different set. Its runtime part is structural, not data-dependent: a converted module counts if it emits on its
  input's grid, has no observer of its own in the prepared graph, and is not an exact op such as identity, ReLU or
  max-pool. On a random-init student most pooled means land exactly on the grid, so a data-dependent test would miss
  sites (MEASURED). src/models/student.py's blob is the same at dc40dd2, fe93a24 and the lane's HEAD (40a84c45…).
- (c) **DIST is transcribed, not copied.** This session had no web access. The formulas follow the audit's S1c
  description of hunto/DIST_KD segmentation/losses/dist_kd.py and the paper. The smoke's reference was typed
  separately from src/explore/dist_loss.py, so the two agree with each other, not necessarily with the official file.
  Diff `_ref_*` in scripts/smoke_explore_kd.py against the official file before A2 launches.
- (d) **Run-time substitutions inside pinned modules** (src/explore/patching.py):
  - N2 binds src/quant/qat.py's `prepare_qat_model` to a wrapper that calls it and installs the sites;
  - A1'' binds train_distill's `alpha_gate_error` to one that admits exactly N4's alpha;
  - A2 binds `distillation_losses` to DIST's;
  - the futility flag, off in every launch, binds `_jsonl`.

  Each pinned file stays byte-unchanged; its sha256 is in x_run_meta.json and in run_meta's x_substitution. A
  binding lasts one call in one exploratory process, and the smokes show the restoration. The audit should rule
  whether this meets "imports the pinned trainers unchanged"; the alternative is a copied trainer, which would drift
  from the record.
- (e) **N1 is a new loop**, not the pinned run_qat. run_qat has no loss hook, and run_qat_kd is hard-wired to E3's
  three terms and a projection, while N1 has neither. `run_qat_logit_kd` copies run_qat's lines and adds the teacher
  forward and stage e2's term through train_distill.distillation_losses, at λ 0.5 full weight with E6-KD's ramp and
  stream. Its non-finite handling is run_qat's (AM-21 item 1: recorded, the run continues), not E6-KD's abort. Its
  stage label is E6-E2, so the registered block B accepts it, and its run_meta carries distillation true and the KD
  fields.
- (f) **The evaluator has no exploratory status.** src/eval/artifacts.py ARTIFACT_STATUSES is (official, provisional,
  smoke). The registered block B therefore writes "provisional" artifacts for C1, N1, N2 and D. The exploratory status
  is in x_run_meta.json and x_blockb.json, and the x- run ID is in every artifact name and identity. These artifacts
  carry stage E6-E2 (E6 for D on an E3-stage winner) and must never enter the manifest (clause 2).
- (g) **D on A1''**, an E3-stage winner, runs stage E6, whose recipe is E6-E2's; only the parent loader differs. Its
  block B uses E6's rules file through select_qat_epoch.py's routing by stage, and that file states the same epoch
  rule (highest converted VAL, strict, ties to the earlier epoch). The audit should confirm that the two rules files
  agree in everything D needs.
- (h) **N5's artifact** is a TorchScript trace with an x_mixed_precision.json identity, not plantseg_int8.json. The
  evaluator of record cannot score it, so its VAL values come from the exploratory scorer only. Its FP32 MAC share,
  0.0756 at 512×512, counts conv MACs only.
- (i) **B1 on a CPU** cannot measure TF32's share and does not meet the audit's item (iv). A CUDA run on the QAT pod's
  GPU class does, and the record says which it was.
- (j) **KL-3's clause-3 gate** depends on a future decision-log row that names TEST and the commit of the TEST
  results. If the records lane uses another form, the gate needs a matching change before KL-3 can run.
- (k) **Not built, by the brief:** B3, A1, A3, B2. The Status cells of DL-180 (and DL-181 for KL-3) are left to the
  records lane, because this lane edits no existing file.

## 6. Verification, commits and push

- Every commit used `git commit --only -F <message> -- <explicit paths>`, after the smokes and the scoped status and
  diff. No commit message carries a trailer: `git log --format=%B fe93a24..HEAD` holds no Co-Authored-By,
  Claude-Session or Signed-off-by line. The subjects of the first six commits use the build labels E0–E5 from before
  AM-24 v2.4; history is not rewritten (fast-forward only).
- Commits, oldest first:

  | commit | contents |
  |---|---|
  | 05d1db9 | the guard and its smoke |
  | 9ef3b2d | N4 and its smoke |
  | 2c6c3b3 | B1 and the ten sites, the VAL scorer, synthetic students, their smoke |
  | 743ef40 | the QAT entry for C1, N1, N2, N5 and D, block B, their smoke |
  | 906beb7 | the KD entry for N3, A1'' and A2 (and the futility flag), their smoke |
  | 99191f6 | KL-3 parity and its smoke |
  | 5be9c54 | the first version of this report and the smoke runner (pushed 13:34 +08:00) |
  | 8421a19 | AM-24 v2.4: the ten sites listed from the code or STOP (B1, N2); item names throughout |
  | (this commit) | this report, rewritten to ORCH-6's instruction |

- The scoped diff (`git diff --no-renames --name-status fe93a24..HEAD -- . ':(exclude)docs/reference/reference.pdf'
  ':(exclude)docs/reference' ':(exclude,icase)*test*' | grep -vi test`): 28 lines, every one `A`.
- Push: `git push -u origin lane/l-am24`, fast-forward, that branch only.
