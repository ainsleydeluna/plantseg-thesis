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

## Addendum L-AM24-ADD-2 — N2R2 (AM-29's E6-R2-M) and block B's card fix 1

Session L-AM24-ADD-2 (lane role: implementation, smokes, push of lane/l-am24 only), from the prompt
l_am24_add2_prompt_20261115.md (sha256 164ea422baba17d64eb60e6fecdeff64fbbf1b0397dc864a36399fc35e65ef33, 6,762 B),
in a Claude Code cloud session on lane/l-am24 at 60dc29497accf2ff0ccd9a38857b9e53435a4dba. Labels: MEASURED (output
of a command, a run or a sha256 read in this session), DOCUMENTED (a committed record, or the prompt, says it),
REPOSITORY-PROVEN (read in the code at the commit named), INFERRED (a conclusion this session did not observe).
Times are the session clock's (`date`), given in +08:00.

### A0. Status (read first)

- **Push.** lane/l-am24 is pushed after this report's commit: the code commit a8718d7, then this report's commit, on 60dc294. Fast-forward from 60dc294, lane/l-am24 only, never forced; master untouched. Commits in
  A9.
- **Built.** Item N2R2 in the exploratory QAT entry (seed 42 only, N2's two flags exactly, run ID x-n2r2-s42, E3-R2
  seed 42's checkpoint da70d82f… by sha256 and by DL-136's identity, then E6-R2's own parent loader, the site census,
  N2's substitution and TF32 off on the pinned run_qat at stage e6-r2). Block B accepts N2R2. Card fix 1 (DL-251):
  block B records the selected epoch's fake-quant VAL and fake-quant-off VAL for N2, the latter through the pinned
  scripts/evaluate_model.py, unchanged, as a subprocess. N2's code is unchanged. MEASURED: main()'s N2 branch,
  exploratory_parent, item_flag_error, run_qat_logit_kd, the E2 table, sites.py and patching.py are byte-identical
  to 60dc294's. The gate lines N2 shares with N2R2 changed only by N2R2 conditionals that give N2 its 60dc294
  values (REPOSITORY-PROVEN). Its existing smoke cases pass unchanged.
- **One STOP, not worked around (A3).** Card fix 1 for N2R2 needs a change to a non-lane file: src/eval/artifacts.py
  line 66, `FQOFF_STAGES = ("E5", "E6", "E6-E2", "E6-KD")`, holds no "E6-R2", so the pinned evaluator refuses N2R2's
  fake-quant-off VAL (`[fqoff_stage_not_in_scope]`, MEASURED by the script itself in the smoke). Block B records that
  refusal and ends INCOMPLETE (exit 3) for N2R2. That file was not edited.
- **N2R2 cannot pass its parent check yet, as built (A2, A8).** E6-R2's parent loader requires AM-23 item 3(b)'s
  condition record, reports/arms/e6x_condition_R2.json, committed with a RECORDED decision-log row. DL-136
  (DOCUMENTED): "no reports/arms/e6x_condition_R2.json exists: no E6-R2 launch can follow (DL-138)". MEASURED: no
  such file in the trees of fe93a24 or of master aa8cac2 (`git ls-tree`). N2R2 takes those inputs as required
  arguments with their sha256, and the checkout that holds them as `--records-root` at `--records-commit`. It
  refuses until they exist. So does E6-R2.
- **Smokes all pass (A5), MEASURED.** Before, at 60dc294: 6/6, 246 rows, 20 mutations killed.
  After, at a8718d7: 7/7, 334 rows, 25 mutations killed. The six existing smokes have the
  same counts as before; the new scripts/smoke_explore_n2r2.py has 88 rows (83 cases and
  5 mutations).
- **Nothing was launched.** No experiment, pod, GPU, dataset read, real-data scoring or TEST. Inputs were synthetic.
- **Lane files only (A4).** `git diff --name-only fe93a24..HEAD` lists lane files only. src/, scripts/ and configs/
  outside them are byte-identical to fe93a24, and fe93a24's src/, scripts/ and configs/ trees equal master aa8cac2's
  (MEASURED: equal git tree ids cc891886…, 7968da84…, 47117626…).

### A1. Base, inputs and environment

- **Clock.** Session start 19:29:52 +08:00 on 15 Nov (11:29:52Z).
  The final smokes ran from 20:19:42 to 20:28:12 +08:00. The code commit a8718d7 was made at
  20:28:35 +08:00 (12:28:35Z), and this report after it.
- **Base.** lane/l-am24 at 60dc29497accf2ff0ccd9a38857b9e53435a4dba, scoped status clean (MEASURED). master was
  fetched by name (`git fetch origin master`): aa8cac283c1d7bda7aceb0844b47f6c3cf1ab1b7 (MEASURED). It was fetched to
  read DL-250 and DL-251, which lane/l-am24's decision log does not hold (its last row is DL-228; DL-250 and DL-251
  are master's), and AM-24's text (master only).
- **The venv** (approved by the user in this session: "Approve venv"). `/usr/bin/python3.11 -m venv` (Python 3.11.18)
  in the session scratchpad. The same 25 wheels and sources as §1's table: torch 2.1.0+cpu and torchvision 0.16.0+cpu
  from download.pytorch.org/whl/cpu, and the 23 PyPI pins. Each was downloaded with `--no-deps --only-binary=:all:`,
  then hashed; all 25 sha256 equal §1's table (MEASURED, compared by script). They were installed with
  `--no-deps --no-index`. `pip check`: no broken requirements. Engines: qnnpack, none, onednn, x86, fbgemm; 4 threads;
  no CUDA. Nothing was installed outside the scratchpad, and nothing of it is in the repository.
- **What was read.** The files the prompt lists: src/quant/qat_select.py in part (rules_path_for_stage,
  epoch_selection), the others whole. Then the pinned code they call:
  - src/quant/stages.py, runner.py and checkpoint.py, whole;
  - qat_artifacts.py (read_run_record) and x86_latency.py (translate_qat_state_to_x86), in part;
  - src/eval/artifacts.py lines 54-67 and the qat_fakequant_off check;
  - src/eval/stage_artifacts.py: the fqoff validator's head and `_fail`;
  - src/distill/export.py (deployment_student_state);
  - src/explore/decompose.py (qat_row, DL-176's form);
  - scripts/smoke_am23_quant.py lines 60-460 (its fixture records repository). The two docs were read by grep or line
  range only, inside each file. docs/DECISION_LOG.md: DL-136, DL-138, DL-139, DL-146, DL-250, DL-251 in full. To find
  the AM-23 rows on E6-R2, the ids of the rows naming E6-R2 were listed, and the first 330 characters of DL-127,
  DL-164, DL-222, DL-227, DL-234, DL-237 and DL-269 were printed (A7). docs/PREREGISTRATION_AMENDMENTS.md (master;
  AM-23's section is byte-identical on the lane): AM-23 items 1, 3 and 4; AM-24 items 2, 4 and 5, item 10 whole,
  item 11's lead and items 11(b) and 11(e).

### A2. What was built

**N2R2's identity of record** (src/explore/qat_explore.py `E3R2_PARENT_OF_RECORD`, DOCUMENTED by DL-136): run
e3-r2_s42_lambda0.5_alpha25_80k_a2, seed 42, arm E3-R2, best.json 9cb73b2b… naming e3_student_best_iter80000.pt
(da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4), best VAL 0.3842073380947113,
e3_run_meta.jsonl f68e4ec0…, e3_telemetry.jsonl 8c8465b4…. The smoke checks that each value stands in DL-136's row.

**The parent check** (`n2r2_parent`), in this order:
1. D's `--parent-sha256` is refused; seed 42 only; `--expect-source-sha256` must be da70d82f…
   (`parent_not_of_record` otherwise, so N2's E2 parent 73dbc32d… is refused).
2. `records_root_record`: `--records-root` must be a git checkout's top level, with no "test" in its path (checked
   before git runs), whose HEAD is `--records-commit`. git runs with the environment of src/quant/qat.py
   `_git_head_bytes`, so it names the HEAD the pinned readers then read.
3. E6-R2's own loader, unchanged, in src/quant/qat.py `real_run_gates`' order (REPOSITORY-PROVEN):
   - `read_tracked_selection` reads the λ and α selections at that HEAD;
   - `resolve_parent` runs on stage E6-R2: best.json, the checkpoint's sha256, e3_run_meta.jsonl's one run_meta row,
     mode real, seed 42, stage E3, and a run_end with checks_passed and no run_abort;
   - then DL-136's identity (`n2r2_identity_differ`): run ID, checkpoint name, best.json, run_meta, telemetry,
     seed, stage and the value's repr;
   - then `am23_parent_binding` with `repo_root` = the records checkout: run_meta's arm E3-R2, descriptive true,
     parent_of_e4_e7 false, max_iters and poly_horizon 80,000, target_construction set; the λ and α winners;
     `--condition-record` at the path of record for arm R2, tracked and unchanged at HEAD, with its
     `--condition-sha256`; the record's run, best.json, checkpoint, run_meta, telemetry and value equal the
     parent's; the verdict recomputed as yes; and `--condition-dl` a RECORDED row of that checkout's
     docs/DECISION_LOG.md carrying the record's tokens.
4. `load_source`: the strict E3 loader, as E6-R2's `main` calls it.

The smoke shows the two agree: E6-R2's own gates (`resolve_parent` + `am23_parent_binding`) give the parent record
and the binding N2R2 returns, equal as dicts (MEASURED).

**Why a records checkout.** The pinned loader reads those committed inputs at the HEAD of a checkout (`repo_root`,
default this one; REPOSITORY-PROVEN). lane/l-am24 may hold no file but fe93a24's outside the lane's, and neither
fe93a24 nor master aa8cac2 holds the condition record or its row (MEASURED). With `repo_root` = the lane checkout,
N2R2 would refuse `[condition_missing_file]` (MEASURED in the smoke), and making it pass would take a non-lane file
in the lane. That is the STOP of the hard limit, so it was not done. The binding's own `repo_root` parameter takes a
second checkout instead. The pinned scripts/smoke_am23_quant.py passes it the same way, to a synthetic records
repository (REPOSITORY-PROVEN). The code that runs is still the lane checkout at `--expect-head`. No input has a
default; each sha256 is a required argument (prompt item 1).

**The run** (`run_n2r2`). The pinned run_qat runs on stage e6-r2 with N2's change made exactly as scripts/explore_qat.py's
N2 branch makes it on e6-e2: `set_tf32(False)` (both flags), then `prepare_qat_model` bound for that call to a wrapper
that calls the pinned function and installs `SiteMatcher(prepared, SITE_NAMES)`. The binding is recorded
(x_substitution) and restored. `sites=False` runs the same call untouched (E6-R2's path); it exists for the smoke and
is refused in a real run.

**scripts/explore_qat.py for N2R2.**
- It requires the nine records flags and run ID x-n2r2-s42; another item given any of those flags is refused.
- It binds the clip as E6-R2's (`clip_binding("E6-R2", …)`; the result equals E6-E2's).
- It runs N2's site census on a calibrated copy of the loaded parent and STOPs on a different set.
- x_run_meta.json adds `am23_parent`, `x_parent_of_record` and `x_records_root`.
- run_meta (the trainer's first row) records what E6-R2's `main` records (`am23_parent`, `e6_selections` null,
  `cwd_projection_loaded` false), plus the lane's x_ fields: x_item N2R2, x_substitution, x_tf32_set,
  x_parent_of_record, x_records_root, x_run_meta_sha256. run_qat itself records tf32 (both flags false), the parent
  and the code provenance (git_head, code_files_sha256).

**Block B** (scripts/explore_qat_blockb.py).
- N2R2 is a registered item; the existing scripts run unchanged as subprocesses. select_qat_epoch.py picks AM-23's
  rules file by stage (configs/qat_selection_rules_am23.json lists E6-R2; REPOSITORY-PROVEN).
- For N2 and N2R2 only:
  - the run's stage must be its item's (N2 E6-E2, N2R2 E6-R2), refused `[run_stage]` before any step;
  - the selection's rules_sha256 is checked against AM-23's file;
  - after finalize comes card fix 1: the selected epoch's fake-quant VAL (its epoch_end row's
    fake_quant_val_all_class_miou, with the telemetry's sha256) and its fake-quant-off VAL. The latter is
    `scripts/evaluate_model.py --precision qat_fakequant_off`, run from the checkout as qat_epoch_eval.py's score step
    runs the evaluator: VAL, canvas, CPU, batch size 1, artifact status provisional, no thread flag. It is bound by
    the selection's sha256, the telemetry's and `--expected-checkpoint-sha256` (the winner's). Its summary is
    checked (split val, precision, stage, the winner's checkpoint, 846 rows, the value in [0, 1]) and recorded with
    its sha256, i.e. DL-176's form as src/explore/decompose.py reads it.
- C1, N1, N5 and D take exactly their 60dc294 path (every new branch is gated on N2/N2R2).

### A3. STOP: card fix 1 for N2R2 (src/eval/artifacts.py:66)

- **File and line.** src/eval/artifacts.py:66, `FQOFF_STAGES = ("E5", "E6", "E6-E2", "E6-KD")`. Its comment,
  lines 64-65: "E6-R1 and E6-R2 were cut and never launched (DL-137, DL-138), so they have no model. Defined here only;
  other modules import it." (REPOSITORY-PROVEN.)
- **Reason.** scripts/evaluate_model.py refuses precision qat_fakequant_off for a stage outside it, in
  `validate_qat_fakequant_off_cli` (lines 157-159), before any file or dataset is read. src/eval/artifacts.py
  `validate_artifact_request` (lines 308-313) refuses it as well. The validator binds `--stage` to the selection's,
  the run's and the payload's stage (`fqoff_stage_mismatch`), so no other stage name can stand in for E6-R2. N2R2's
  fake-quant-off VAL therefore cannot be made through the pinned evaluator at this code (fe93a24 = master aa8cac2)
  without that change. Making it is outside the lane.
- **Effect.** N2R2's block B runs every registered step, then the fake-quant-off command. The evaluator refuses it
  (exit 1, `CliError: [fqoff_stage_not_in_scope] …`; MEASURED by the script itself in the smoke). Block B records
  that and ends `RESULT: X BLOCK B INCOMPLETE -- see x_blockb.json (fake-quant-off VAL: failed, CliError:
  [fqoff_stage_not_in_scope] …)`, exit 3. The selected epoch and its fake-quant VAL are still recorded. The fix,
  adding "E6-R2" to FQOFF_STAGES, is for that file's owner, outside this lane. Whether it is wanted is for the audit
  and AM-29 to rule (A8).

### A4. Files, sha256 and diff

The report itself (docs/lane_reports/l-am24.md, sha256 before 5045e9880248c2d5525d204d79cbdd2b96291ed608596a0d5b533f36767cfa7c)
gains this section in its own commit. A file cannot carry its own sha256 or commit, so both are in the download copy
and the chat reply. The code commit is a8718d776c068cbe60e685c52afe545d8bc4fa36:

| file | sha256 at 60dc294 | sha256 at a8718d7 |
|---|---|---|
| scripts/explore_qat.py | 3c8143a96e24d9a6786f8b1c112848ba2ecfb6cd0a6c162d35041ea3550a499f | 6c6cf0d78da54bf937491e108e9996b97de680847c367a75a6de2a1dd01413de |
| scripts/explore_qat_blockb.py | 3bd5f50100c29b484bc9ebf036d08ed711e58d0ad2458e4a3f12ca1f1476b736 | 7f92351516b2eeb5c0ed195ab310fde1f75ff95f3171aa624ff6a4897dfe7dfe |
| scripts/smoke_explore_all.py | da82fd37b0a355176cfb32ede7e54ef82ab31e0631dee36ca54a0b7fd7971fea | 823ae35c9fa8ffcfdd53a43dc034708125037e352d38554bb036017170786fc6 |
| scripts/smoke_explore_n2r2.py | (new file) | 95e2435e75b11f81634d4bfbd9a12f38afec7988547890fe57e20522965ef1b0 |
| src/explore/guard.py | 8b5fadd695c6a7766961aa42bcd6f506840e9cd2b33c455595b2b14d9507e258 | ada4cf2498b0f657c5a33adce6af73d35d8c6820455f5638d8d8c2c4866e29cd |
| src/explore/qat_explore.py | 5229fa2b2015f5ba583b4c36cdb519025101e17b2ac63bf42050977d017390ac | 2b353c2edd25da107cbe6b823296f7ae3cdb86b248af515851b2fd3aebb5dcee |

`git diff --name-only fe93a24..a8718d7 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference'`
with test-named paths filtered, 29 lines: every one a lane file (src/explore/*, scripts/explore_*.py,
scripts/smoke_explore_*.py) or the report. The report commit adds nothing to the list.

```text
docs/lane_reports/l-am24.md
scripts/explore_b1.py
scripts/explore_decompose.py
scripts/explore_kd.py
scripts/explore_kl3_parity.py
scripts/explore_qat.py
scripts/explore_qat_blockb.py
scripts/smoke_explore_all.py
scripts/smoke_explore_decompose.py
scripts/smoke_explore_guard.py
scripts/smoke_explore_kd.py
scripts/smoke_explore_kl3.py
scripts/smoke_explore_n2r2.py
scripts/smoke_explore_qat.py
scripts/smoke_explore_sites.py
src/explore/__init__.py
src/explore/b1_split.py
src/explore/decompose.py
src/explore/dist_loss.py
src/explore/fp32_head.py
src/explore/guard.py
src/explore/kd_explore.py
src/explore/kl3_parity.py
src/explore/patching.py
src/explore/qat_explore.py
src/explore/sites.py
src/explore/smokekit.py
src/explore/synthetic.py
src/explore/valscore.py
```

The full diff of the code commit against 60dc294:

<details><summary>git diff 60dc294 a8718d7 (1453 lines)</summary>

```diff
diff --git a/scripts/explore_qat.py b/scripts/explore_qat.py
index a74b2a6..ee41151 100644
--- a/scripts/explore_qat.py
+++ b/scripts/explore_qat.py
@@ -1,22 +1,32 @@
 #!/usr/bin/env python3
-"""C1, N1, N2, N5 and D (lane L-AM24): the exploratory QAT entry (E6-E2's recipe of record, one named change).
+"""C1, N1, N2, N2R2, N5 and D (lane L-AM24): the exploratory QAT entry (E6-E2's recipe of record, one named change).
 
-    python -B -I scripts/explore_qat.py --real-run --confirm-real-run --item C1|N1|N2|N5|D --seed S
-        --source-run-dir <E2 run dir | D: the exploratory winner's trainer dir>
-        (--expect-source-sha256 <E2 of record's checkpoint sha256> | D: --parent-sha256 <its checkpoint sha256>)
+    python -B -I scripts/explore_qat.py --real-run --confirm-real-run --item C1|N1|N2|N2R2|N5|D --seed S
+        --source-run-dir <E2 run dir | D: the exploratory winner's trainer dir | N2R2: E3-R2 seed 42's run dir>
+        (--expect-source-sha256 <E2 of record's checkpoint sha256 | N2R2: da70d82f...> |
+         D: --parent-sha256 <its checkpoint sha256>)
         --clip-selection <clip_selection.json of record> --clip-selection-sha256 c433145a...
         --num-workers 12 --device cuda --expect-head <40-hex HEAD> --run-id x-... --out-dir <root>/<run-id>
         [N1: --kd-logit --teacher-ckpt <.../iter_24000.pth> --teacher-ckpt-sha256 8c0e649a...]
         [N2: --sites-on-input-grid --tf32-off]   [N5: --fp32-head]
+        [N2R2: --sites-on-input-grid --tf32-off --run-id x-n2r2-s42 --records-root <checkout> --records-commit <40-hex>
+               --lambda-selection reports/derived/lambda_selection.json --lambda-selection-sha256 <sha>
+               --alpha-selection reports/derived/alpha_selection.json --alpha-selection-sha256 <sha>
+               --condition-record reports/arms/e6x_condition_R2.json --condition-sha256 <sha> --condition-dl DL-<n>]
 
-Order: the item's flags and seed; the guard (split VAL, the data root's TRAIN/VAL-only check, x- run ID, output
-outside the repository and official folders); the real-run gates of src/quant/qat.py real_run_gates that apply (the
-config pins, HEAD = --expect-head, the pinned QAT paths and the lane's files clean at HEAD, workers, CUDA present and
-uninitialised, TF32 at torch defaults before launch, QNNPACK, the clip selection of record with clip 5.0); the
-exploratory parent check; the strict source load; for N1 the teacher of record through src/quant/qat_kd.py
-teacher_of_record. Nothing is written before all pass. Then <out-dir>/x_run_meta.json, and the pinned trainer's run
-directory <out-dir>/<run-id>. Block B: scripts/explore_qat_blockb.py. Exit codes as src/quant/qat.py main: 0 done,
-1 STOP, 2 refused, 3 aborted or incomplete, 4 error.
+N2R2 (addendum L-AM24-ADD-2) is N2's change on E6-R2: E6-R2's stage, clip binding and parent loader (src/quant/qat.py
+resolve_parent and am23_parent_binding, which read the selections, AM-23 item 3(b)'s condition record and its RECORDED
+decision-log row at the HEAD of --records-root), then the pinned run_qat with N2's substitution and TF32 off.
+
+Order: the item's flags and seed (N2R2: also its nine records flags and its run ID); the guard (split VAL, the data
+root's TRAIN/VAL-only check, x- run ID, output outside the repository and official folders); the real-run gates of
+src/quant/qat.py real_run_gates that apply (the config pins, HEAD = --expect-head, the pinned QAT paths and the lane's
+files clean at HEAD, workers, CUDA present and uninitialised, TF32 at torch defaults before launch, QNNPACK, the clip
+selection of record with clip 5.0); the exploratory parent check (N2R2: src/explore/qat_explore.py n2r2_parent, E6-R2's
+parent loader); the strict source load; for N2 and N2R2 the site census; for N1 the teacher of record through
+src/quant/qat_kd.py teacher_of_record. Nothing is written before all pass. Then <out-dir>/x_run_meta.json, and the
+pinned trainer's run directory <out-dir>/<run-id>. Block B: scripts/explore_qat_blockb.py. Exit codes as
+src/quant/qat.py main: 0 done, 1 STOP, 2 refused, 3 aborted or incomplete, 4 error.
 """
 from __future__ import annotations
 
@@ -58,6 +68,8 @@ def build_parser() -> argparse.ArgumentParser:
     p.add_argument("--split", default="val")
     for f in FLAG_NAMES:
         p.add_argument("--" + f.replace("_", "-"), dest=f, action="store_true")
+    for f in X.N2R2_RECORD_FLAGS:                 # N2R2 only: E6-R2's committed inputs and their checkout, no default
+        p.add_argument("--" + f.replace("_", "-"), dest=f, default=None)
     return p
 
 
@@ -84,17 +96,34 @@ def gates(args, flags: set) -> dict:
         raise X.ExploreQATError("teacher_required", "N1 needs --teacher-ckpt and --teacher-ckpt-sha256 (of record)")
     if args.item != "N1" and (args.teacher_ckpt or args.teacher_ckpt_sha256):
         raise X.ExploreQATError("teacher_not_applicable", f"{args.item} runs no teacher")
+    n2r2_given = {f: getattr(args, f, None) for f in X.N2R2_RECORD_FLAGS}
+    if args.item == "N2R2":
+        missing = ["--" + f.replace("_", "-") for f, v in n2r2_given.items() if v in (None, "")]
+        if missing:
+            raise X.ExploreQATError("n2r2_records_required", f"N2R2 reads E6-R2's committed inputs through its parent "
+                                                              f"loader: {', '.join(missing)} required (no default)")
+        if args.run_id != X.N2R2_RUN_ID:
+            raise X.ExploreQATError("n2r2_run_id", f"N2R2's run ID is {X.N2R2_RUN_ID}, not {args.run_id!r}")
+    elif any(v is not None for v in n2r2_given.values()):
+        given = sorted(k for k, v in n2r2_given.items() if v is not None)
+        raise X.ExploreQATError("n2r2_flags_not_applicable", f"{args.item} takes none of N2R2's flags {given}")
     from src.explore.fp32_head import PRECISION_LABEL
+    recipe_extra = ({"parent_of_record": X.E3R2_PARENT_OF_RECORD, "matched_run": X.N2R2_MATCHED_RUN,
+                     "stage": "E6-R2", "registered_by": "AM-29 (E6-R2-M; drafted in parallel, not committed at this "
+                                                         "commit); lane addendum L-AM24-ADD-2"}
+                    if args.item == "N2R2" else {})
     st = G.guarded_start(
         item=args.item, run_id=args.run_id, out_dir=args.out_dir, split=args.split, allowed_splits=G.VAL_ONLY,
         input_paths={"--source-run-dir": args.source_run_dir, "--clip-selection": args.clip_selection,
-                     "--teacher-ckpt": args.teacher_ckpt},
+                     "--teacher-ckpt": args.teacher_ckpt, "--records-root": args.records_root},
         seeds=[args.seed], pinned_paths=pinned_paths(args.item),
-        precision_label=PRECISION_LABEL if args.item == "N5" else "int8 QNNPACK (as E6-E2)",
-        recipe={"recipe_of_record": X.RECIPE_OF_RECORD, "change": X.CHANGES[args.item], "flags": sorted(flags),
+        precision_label=(PRECISION_LABEL if args.item == "N5" else
+                         "int8 QNNPACK (as E6-R2)" if args.item == "N2R2" else "int8 QNNPACK (as E6-E2)"),
+        recipe={"recipe_of_record": X.RECIPE_OF_RECORD_N2R2 if args.item == "N2R2" else X.RECIPE_OF_RECORD,
+                "change": X.CHANGES[args.item], "flags": sorted(flags),
                 "seed": args.seed, "clip": X.CLIP_OF_RECORD, "comparators": X.COMPARATORS,
                 "lambda_logit": X.N1_LAMBDA if args.item == "N1" else None,
-                "d_trigger": str(X.D_TRIGGER) if args.item == "D" else None},
+                "d_trigger": str(X.D_TRIGGER) if args.item == "D" else None, **recipe_extra},
         write=False)
     pins = Q.config_pins_error()
     if pins:
@@ -122,22 +151,38 @@ def gates(args, flags: set) -> dict:
     if args.clip_selection_sha256 != X.CLIP_SELECTION_SHA256:
         raise X.ExploreQATError("clip_selection_not_of_record", f"--clip-selection-sha256 is not DL-139's "
                                                                 f"{X.CLIP_SELECTION_SHA256}")
-    clip = Q.clip_binding("E6-E2", args.seed, X.CLIP_OF_RECORD, u4_pilot=False, clip_selection=args.clip_selection,
-                          clip_selection_sha256=args.clip_selection_sha256)
-    stage, parent = X.exploratory_parent(args.item, st["inputs"]["--source-run-dir"], seed=args.seed,
-                                         expect_source_sha256=args.expect_source_sha256,
-                                         parent_sha256=args.parent_sha256)
+    clip = Q.clip_binding("E6-R2" if args.item == "N2R2" else "E6-E2", args.seed, X.CLIP_OF_RECORD, u4_pilot=False,
+                          clip_selection=args.clip_selection, clip_selection_sha256=args.clip_selection_sha256)
+    am23 = records = None
+    if args.item == "N2R2":                       # E6-R2's parent loader, its committed inputs at --records-root's HEAD
+        got = X.n2r2_parent(st["inputs"]["--source-run-dir"], seed=args.seed,
+                            expect_source_sha256=args.expect_source_sha256, parent_sha256=args.parent_sha256,
+                            records_root=st["inputs"]["--records-root"], records_commit=args.records_commit,
+                            lambda_selection=args.lambda_selection,
+                            lambda_selection_sha256=args.lambda_selection_sha256,
+                            alpha_selection=args.alpha_selection, alpha_selection_sha256=args.alpha_selection_sha256,
+                            condition_record=args.condition_record, condition_sha256=args.condition_sha256,
+                            condition_dl=args.condition_dl)
+        stage, parent, am23, records = got["stage"], got["parent"], got["am23_binding"], got["records_root"]
+    else:
+        stage, parent = X.exploratory_parent(args.item, st["inputs"]["--source-run-dir"], seed=args.seed,
+                                             expect_source_sha256=args.expect_source_sha256,
+                                             parent_sha256=args.parent_sha256)
     model, source_meta = Q.load_source(stage, parent)
     census = None
     if args.item == "N2":                         # AM-24 v2.4: the ten sites from the code, or STOP before launch
         from src.explore.sites import calibrated_copy, require_site_set
         census = require_site_set(calibrated_copy(model))
+    if args.item == "N2R2":                       # N2's launch gate, on a calibrated copy of E6-R2's parent
+        from src.explore.sites import calibrated_copy, require_site_set
+        census = require_site_set(calibrated_copy(model))
     teacher = teacher_rec = None
     if args.item == "N1":
         from src.quant import qat_kd as K
         teacher, teacher_rec = K.teacher_of_record(args.teacher_ckpt, args.teacher_ckpt_sha256)
     return {"guard": st, "clip": clip, "stage": stage, "parent": parent, "model": model, "source_meta": source_meta,
-            "teacher": teacher, "teacher_record": teacher_rec, "site_census": census}
+            "teacher": teacher, "teacher_record": teacher_rec, "site_census": census, "am23_binding": am23,
+            "records_root": records}
 
 
 def main(argv=None) -> int:
@@ -167,9 +212,14 @@ def main(argv=None) -> int:
         extra_x["x_mac_share"] = mac_share(copy.deepcopy(model))
         apply_fp32_head(model)
         extra_x["x_precision_label"] = PRECISION_LABEL
+    if args.item == "N2R2":
+        extra_x["x_parent_of_record"] = X.E3R2_PARENT_OF_RECORD
+        extra_x["x_records_root"] = g["records_root"]
     doc = dict(st["run_meta"])
     doc.update({"parent_check": g["parent"], "clip_binding": g["clip"], "quant_stage": g["stage"]["name"],
                 "teacher_record": g["teacher_record"], "site_census": g["site_census"], **extra_x})
+    if args.item == "N2R2":
+        doc["am23_parent"] = g["am23_binding"]
     path = G.write_run_meta(out, doc)
     inner = out / args.run_id
     extra = {"clip_selection": g["clip"]["clip_selection"], "u4_pilot": False, "e6_selections": None,
@@ -178,6 +228,8 @@ def main(argv=None) -> int:
              "cwd_projection_loaded": False, "x_exploratory": True, "x_amendment": G.AMENDMENT, "x_item": args.item,
              "x_flags": sorted(flags), "x_change": X.CHANGES[args.item], "x_run_meta_sha256": G.sha256_file(path),
              "x_artifact_status": G.ARTIFACT_STATUS, **extra_x}
+    if args.item == "N2R2":                           # as src/quant/qat.py main records it for an AM-23 stage
+        extra["am23_parent"] = g["am23_binding"]
     try:
         if args.item == "N1":
             summary = X.run_qat_logit_kd(mode="real", model=model, teacher=g["teacher"], source_meta=g["source_meta"],
@@ -205,6 +257,12 @@ def main(argv=None) -> int:
                                     clip_source=g["clip"]["clip_source"], device=args.device,
                                     num_workers=args.num_workers, parent=g["parent"], extra_meta=extra)
             print(f"[n2] site hook calls: {holder['matcher'].calls}")
+        elif args.item == "N2R2":                     # N2's change on E6-R2 (src/explore/qat_explore.py run_n2r2)
+            summary, substitution, matcher = X.run_n2r2(
+                mode="real", model=model, source_meta=g["source_meta"], out_dir=inner, seed=args.seed,
+                clip_norm=X.CLIP_OF_RECORD, clip_source=g["clip"]["clip_source"], device=args.device,
+                num_workers=args.num_workers, parent=g["parent"], extra_meta=extra)
+            print(f"[n2r2] site hook calls: {matcher.calls}")
         else:
             summary = Q.run_qat(stage=g["stage"]["key"], mode="real", model=model, source_meta=g["source_meta"],
                                 out_dir=inner, seed=args.seed, clip_norm=X.CLIP_OF_RECORD,
diff --git a/scripts/explore_qat_blockb.py b/scripts/explore_qat_blockb.py
index 5478d92..4da1dcc 100644
--- a/scripts/explore_qat_blockb.py
+++ b/scripts/explore_qat_blockb.py
@@ -1,19 +1,27 @@
 #!/usr/bin/env python3
-"""Block B of an exploratory QAT run (lane L-AM24; C1, N1, N2, N5, D): 15 converted epochs scored on the CPU, AM-23's selection
-rules, finalize.
+"""Block B of an exploratory QAT run (lane L-AM24; C1, N1, N2, N2R2, N5, D): 15 converted epochs scored on the CPU,
+AM-23's selection rules, finalize.
 
     python -B -I scripts/explore_qat_blockb.py --run-dir <item dir>/<run-id> --expect-telemetry-sha256 <sha>
         --host-label TEXT --run-id x-...-bb --out-dir <root>/x-...-bb [--threads 8]
 
 The guard runs first (VAL, the data root's TRAIN/VAL-only check, x- run ID, the output outside the repository,
 every official folder and the run directory). The run is identified from its x_run_meta.json (the item folder that
-holds the trainer's run directory): it must be an exploratory C1, N1, N2, N5 or D run.
+holds the trainer's run directory): it must be an exploratory C1, N1, N2, N2R2, N5 or D run.
 
-  C1, N1, N2, D: the existing scripts, unchanged, as subprocesses from the checkout, into <out-dir>/eval:
+  C1, N1, N2, N2R2, D: the existing scripts, unchanged, as subprocesses from the checkout, into <out-dir>/eval:
                  qat_epoch_eval.py convert --purpose record, score (the evaluator at --artifact-status provisional:
                  the evaluator has no exploratory status), select_qat_epoch.py (the rules file of the run's stage:
-                 AM-23's for E6-E2), qat_epoch_eval.py finalize. They accept the run because its telemetry is the
-                 pinned trainer's own (stage E6-E2 or E6, mode real); the x- run ID travels in every artifact name.
+                 AM-23's for E6-E2 and E6-R2), qat_epoch_eval.py finalize. They accept the run because its telemetry
+                 is the pinned trainer's own (stage E6-E2, E6-R2 or E6, mode real); the x- run ID travels in every
+                 artifact name.
+  N2, N2R2 also: the run's stage is its item's (N2 E6-E2, N2R2 E6-R2), the selection's rules are AM-23's file, and,
+                 after finalize, card fix 1 of the lane's audit (DL-251; AM-24 item 10(b)): the selected epoch's
+                 fake-quant VAL (its epoch_end row) and fake-quant-off VAL, the latter by the pinned
+                 scripts/evaluate_model.py at precision qat_fakequant_off (lane L-FQOFF), unchanged, as a subprocess
+                 into <out-dir>/fqoff/eNN, each value recorded with its file and sha256 (DL-176's form). The
+                 evaluator's own stage list (src/eval/artifacts.py FQOFF_STAGES) decides: it holds E6-E2, not E6-R2,
+                 so for N2R2 the evaluator refuses and block B ends INCOMPLETE with that refusal recorded.
   N5:            its own steps (the registered converter refuses the FP32 head's key set and its float fallback):
                  each epoch's state, sha-checked against its epoch_end row, converted through the pinned copy and
                  convert onto the FP32-head skeleton (src/explore/fp32_head.py), the islands report, VAL scored on the
@@ -29,6 +37,7 @@ from __future__ import annotations
 
 import argparse
 import json
+import math
 import subprocess
 import sys
 import time
@@ -41,8 +50,14 @@ if str(REPO) not in sys.path:
 
 from src.explore import guard as G                                   # noqa: E402
 
-REGISTERED_ITEMS = ("C1", "N1", "N2", "D")
+REGISTERED_ITEMS = ("C1", "N1", "N2", "N2R2", "D")
 FORMAT = "x_qat_blockb/1"
+# Card fix 1 (DL-251; AM-24 item 10(b)), for N2 and N2R2: the selected epoch's fake-quant and fake-quant-off VAL.
+FQOFF_ITEMS = ("N2", "N2R2")
+ITEM_STAGE = {"N2": "E6-E2", "N2R2": "E6-R2"}
+AM23_RULES_REL = "configs/qat_selection_rules_am23.json"
+EVALUATOR_SCRIPT = "scripts/evaluate_model.py"          # run from the checkout, unchanged (as qat_epoch_eval.py does)
+FQOFF_PINNED = ("src/eval/artifacts.py", "src/eval/stage_artifacts.py", "src/eval/model_loading.py")
 
 
 def build_parser() -> argparse.ArgumentParser:
@@ -63,10 +78,98 @@ def run_item(run_dir: Path) -> dict:
         raise G.GuardError("run_not_exploratory", f"{run_dir} is not the trainer folder of an x- item folder")
     doc = json.loads(meta.read_text(encoding="utf-8"))
     if doc.get("exploratory") is not True or doc.get("item") not in (*REGISTERED_ITEMS, "N5"):
-        raise G.GuardError("run_item", f"{meta} records item {doc.get('item')!r}; block B takes C1, N1, N2, N5, D")
+        raise G.GuardError("run_item", f"{meta} records item {doc.get('item')!r}; block B takes C1, N1, N2, N2R2, N5, "
+                                       "D")
     return doc
 
 
+def run_stage(run_dir: Path) -> str | None:
+    """The trainer's stage, from its telemetry's first row (the run_meta row)."""
+    with open(run_dir / "qat_telemetry.jsonl", encoding="utf-8") as fh:
+        first = json.loads(fh.readline())
+    return first.get("stage") if isinstance(first, dict) and first.get("event") == "run_meta" else None
+
+
+def selection_rules_record(eval_dir: Path, stage: str) -> dict:
+    """The rules file select_qat_epoch.py read for this stage (src/quant/qat_select.py rules_path_for_stage), its
+    sha256 against the selection's rules_sha256: AM-23's file for E6-E2 and E6-R2."""
+    from src.quant import qat_select as S
+    path = Path(S.rules_path_for_stage(stage))
+    rel = path.resolve().relative_to(REPO.resolve()).as_posix()
+    sha = G.sha256_file(path)
+    sel = json.loads((eval_dir / "qat_selection.json").read_text(encoding="utf-8"))
+    return {"rules_file": rel, "rules_sha256": sha, "selection_rules_sha256": sel.get("rules_sha256"),
+            "ok": rel == AM23_RULES_REL and sel.get("rules_sha256") == sha}
+
+
+def fake_quant_off_argv(*, stage: str, run_dir: Path, winner: dict, selection: Path, selection_sha256: str,
+                        telemetry_sha256: str, out_dir: Path, run_id: str) -> list[str]:
+    """The pinned evaluator's command for the selected epoch's fake-quant-off VAL (precision qat_fakequant_off, lane
+    L-FQOFF), as qat_epoch_eval.py's score step runs the evaluator: from the checkout, on the CPU, canvas protocol,
+    batch size 1, artifact status provisional (the evaluator has no exploratory status) and no thread flag (the thread
+    count is the environment's, which the summary records). The selection, its sha256 and the telemetry's bind the
+    selected epoch's eNN.pt before any dataset or model exists; --expected-checkpoint-sha256 is the winner's."""
+    return [sys.executable, "-B", EVALUATOR_SCRIPT, "--stage", stage, "--model-role", "student",
+            "--precision", "qat_fakequant_off", "--split", "val", "--protocol", "canvas",
+            "--checkpoint", str(Path(run_dir) / str(winner.get("checkpoint"))), "--qat-run-dir", str(run_dir),
+            "--qat-selection", str(selection), "--qat-selection-sha256", selection_sha256,
+            "--qat-telemetry-sha256", telemetry_sha256,
+            "--expected-checkpoint-sha256", str(winner.get("checkpoint_sha256")),
+            "--artifact-status", "provisional", "--device", "cpu", "--batch-size", "1",
+            "--out-dir", str(out_dir), "--run-id", run_id]
+
+
+def selected_epoch_vals(run_dir: Path, eval_dir: Path, out: Path, args, *, stage: str, runner=subprocess.run) -> dict:
+    """Card fix 1 (DL-251; AM-24 item 10(b)): the selected epoch's fake-quant VAL, the value its epoch_end row
+    records, and its fake-quant-off VAL, scored by the pinned scripts/evaluate_model.py, as DL-176 records them: each
+    value with its file and sha256. Returns {"fake_quant", "fake_quant_off", "ok"}."""
+    from src.quant import qat_artifacts as A
+    sel_p = eval_dir / "qat_selection.json"
+    sel = json.loads(sel_p.read_text(encoding="utf-8"))
+    sel_sha = G.sha256_file(sel_p)
+    win = sel.get("winner") or {}
+    epoch = win.get("epoch")
+    rec = A.read_run_record(run_dir)
+    end = rec["ends"].get(epoch) or {}
+    fq_v = end.get("fake_quant_val_all_class_miou")
+    fq = {"epoch": epoch, "value": fq_v, "source": "the selected epoch's epoch_end row, fake_quant_val_all_class_miou",
+          "telemetry": {"path": str(run_dir / "qat_telemetry.jsonl"), "sha256": rec["telemetry_sha256"]},
+          "checkpoint_sha256": end.get("checkpoint_sha256")}
+    fq_ok = isinstance(fq_v, float) and math.isfinite(fq_v)
+    if type(epoch) is not int or end.get("checkpoint_sha256") != win.get("checkpoint_sha256"):
+        return {"fake_quant": fq, "ok": False,
+                "fake_quant_off": {"status": "failed", "message": f"the selection's winner (epoch {epoch!r}) is not an "
+                                                                  "epoch_end row of the run"}}
+    fo_dir = out / "fqoff" / f"e{epoch:02d}"
+    run_id = f"{rec['run_id']}_fqoff_e{epoch:02d}_{str(win.get('checkpoint_sha256'))[:12]}"
+    argv = fake_quant_off_argv(stage=stage, run_dir=run_dir, winner=win, selection=sel_p, selection_sha256=sel_sha,
+                               telemetry_sha256=args.expect_telemetry_sha256, out_dir=fo_dir, run_id=run_id)
+    fo_dir.parent.mkdir(parents=True, exist_ok=True)
+    t0 = time.time()
+    p = runner(argv, cwd=str(REPO), capture_output=True, text=True)
+    row = {"step": "fake_quant_off", "argv": argv[1:], "python": argv[0], "exit": p.returncode,
+           "seconds": time.time() - t0, "out_dir": str(fo_dir), "selection": {"path": str(sel_p), "sha256": sel_sha},
+           "evaluator_status_note": "artifact_status provisional: the evaluator has no exploratory status"}
+    summ = fo_dir / "summary.json"
+    if p.returncode != 0 or not summ.is_file():
+        lines = [ln for ln in (p.stderr or "").splitlines() + (p.stdout or "").splitlines() if ln.strip()]
+        row.update({"status": "failed", "message": lines[-1][:500] if lines else None})
+        return {"fake_quant": fq, "fake_quant_off": row, "ok": False}
+    s = json.loads(summ.read_text(encoding="utf-8"))
+    run, ds = s.get("run") or {}, s.get("dataset") or {}
+    v = (s.get("dataset_level") or {}).get("all_class_miou")
+    checks = {"split_val": ds.get("split") == "val",
+              "precision_qat_fakequant_off": run.get("precision") == "qat_fakequant_off",
+              "stage": run.get("stage") == stage,
+              "checkpoint_is_the_winner": run.get("checkpoint_sha256") == win.get("checkpoint_sha256"),
+              "rows_846": ds.get("expected_rows") == ds.get("actual_rows") == G.VAL_IMAGES,
+              "value_in_0_1": isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0}
+    row.update({"status": "scored" if all(checks.values()) else "refused", "value": v, "checks": checks,
+                "summary": {"path": str(summ), "sha256": G.sha256_file(summ)},
+                "torch_num_threads": (run.get("eval_runtime") or {}).get("torch_num_threads")})
+    return {"fake_quant": fq, "fake_quant_off": row, "ok": all(checks.values()) and fq_ok}
+
+
 def registered_block_b(run_dir: Path, eval_dir: Path, args) -> list[dict]:
     py = [sys.executable, "-B"]
     steps = [
@@ -157,13 +260,15 @@ def main(argv=None) -> int:
         G.check_input_paths({"--run-dir": args.run_dir})
         run_dir = Path(args.run_dir).resolve()
         item_doc = run_item(run_dir)
+        fqoff_item = item_doc["item"] in FQOFF_ITEMS
         st = G.guarded_start(item=item_doc["item"], run_id=args.run_id, out_dir=args.out_dir, split=args.split,
                              allowed_splits=G.VAL_ONLY, input_paths={"--run-dir": str(run_dir.parent)},
                              seeds=item_doc.get("seeds"), pinned_paths=("scripts/qat_epoch_eval.py",
                                                                         "scripts/select_qat_epoch.py",
                                                                         "src/quant/qat_artifacts.py",
                                                                         "src/quant/qat_select.py",
-                                                                        "scripts/evaluate_model.py"),
+                                                                        "scripts/evaluate_model.py")
+                             + (FQOFF_PINNED if fqoff_item else ()),
                              precision_label=item_doc.get("precision_label"),
                              recipe={"what": "block B of an exploratory QAT run", "run_dir": str(run_dir),
                                      "run_item": item_doc["item"], "run_x_run_meta": item_doc.get("run_id"),
@@ -173,6 +278,12 @@ def main(argv=None) -> int:
         tel_before = G.sha256_file(run_dir / "qat_telemetry.jsonl")
         if tel_before != args.expect_telemetry_sha256:
             raise G.GuardError("telemetry_sha256_mismatch", f"the run's telemetry is {tel_before}")
+        stage = None
+        if fqoff_item:                                    # N2 is E6-E2's run, N2R2 E6-R2's
+            stage = run_stage(run_dir)
+            if stage != ITEM_STAGE[item_doc["item"]]:
+                raise G.GuardError("run_stage", f"{item_doc['item']}'s run is stage {ITEM_STAGE[item_doc['item']]}; "
+                                                f"this run's run_meta records {stage!r}")
         if item_doc["item"] == "N5":
             res = n5_block_b(run_dir, eval_dir, args)
             ok = True
@@ -183,6 +294,18 @@ def main(argv=None) -> int:
             res = {"steps": steps, "selection": (json.loads(sel.read_text(encoding="utf-8")) if sel.is_file() else None),
                    "evaluator_status_note": ("the evaluator wrote artifact_status 'provisional' (src/eval/artifacts.py "
                                              "ARTIFACT_STATUSES has no exploratory value); this record is exploratory")}
+            if fqoff_item and ok:                         # card fix 1 (DL-251; AM-24 item 10(b))
+                res["selection_rules"] = selection_rules_record(eval_dir, stage)
+                vals = selected_epoch_vals(run_dir, eval_dir, out, args, stage=stage)
+                res["selected_epoch_fake_quant_val"] = vals["fake_quant"]
+                res["selected_epoch_fake_quant_off_val"] = vals["fake_quant_off"]
+                ok = ok and vals["ok"] and res["selection_rules"]["ok"]
+                fo = vals["fake_quant_off"]
+                what = repr(fo.get("value")) if fo.get("status") == "scored" else fo.get("message")
+                print(f"[blockb] fake_quant_off: exit {fo.get('exit')} {fo.get('status')} {what}")
+            elif fqoff_item:
+                res["selected_epoch_fake_quant_off_val"] = {"status": "not run",
+                                                            "reason": "a registered step did not finish"}
         tel_after = G.sha256_file(run_dir / "qat_telemetry.jsonl")
         doc = {"format": FORMAT, "exploratory": True, "amendment": G.AMENDMENT, "artifact_status": G.ARTIFACT_STATUS,
                "split": "val", "item": item_doc["item"], "run_dir": str(run_dir), "telemetry_sha256": tel_before,
@@ -190,10 +313,22 @@ def main(argv=None) -> int:
                "claim_line": item_doc.get("claim_line"), **res}
         G.write_exclusive(out / "x_blockb.json", G.json_bytes(doc))
         if not ok or tel_after != tel_before:
-            print("RESULT: X BLOCK B INCOMPLETE -- see x_blockb.json")
+            fo_txt = ""
+            if fqoff_item:
+                fo = res.get("selected_epoch_fake_quant_off_val") or {}
+                why = fo.get("message") or fo.get("reason")
+                fo_txt = f" (fake-quant-off VAL: {fo.get('status')}{', ' + why if why else ''})"
+                rules = res.get("selection_rules")
+                if rules is not None and not rules["ok"]:
+                    fo_txt += f" (selection rules {rules['rules_file']}, not {AM23_RULES_REL} as selected)"
+            print("RESULT: X BLOCK B INCOMPLETE -- see x_blockb.json" + fo_txt)
             return G.EXIT_ABORTED
         w = res.get("winner") or ((res.get("selection") or {}).get("winner") or {})
-        print(f"RESULT: X BLOCK B DONE ({item_doc['item']}): epoch {w.get('epoch')}, VAL {w.get('value')!r}"
+        vals_txt = ""
+        if fqoff_item:
+            vals_txt = (f"; selected epoch fake-quant VAL {res['selected_epoch_fake_quant_val']['value']!r}, "
+                        f"fake-quant-off VAL {res['selected_epoch_fake_quant_off_val']['value']!r}")
+        print(f"RESULT: X BLOCK B DONE ({item_doc['item']}): epoch {w.get('epoch')}, VAL {w.get('value')!r}" + vals_txt
               + (" -- MIXED PRECISION, not INT8" if item_doc["item"] == "N5" else ""))
         return G.EXIT_OK
     except G.GuardError as e:
diff --git a/scripts/smoke_explore_all.py b/scripts/smoke_explore_all.py
index 12e10f6..f4b2c7a 100644
--- a/scripts/smoke_explore_all.py
+++ b/scripts/smoke_explore_all.py
@@ -15,7 +15,8 @@ from pathlib import Path
 
 REPO = Path(__file__).resolve().parents[1]
 SMOKES = ("smoke_explore_guard", "smoke_explore_decompose", "smoke_explore_sites", "smoke_explore_qat",
-          "smoke_explore_kd", "smoke_explore_kl3")
+          "smoke_explore_kd", "smoke_explore_kl3",
+          "smoke_explore_n2r2")                    # addendum L-AM24-ADD-2: N2R2 and block B's card fix 1
 
 
 def main() -> int:
diff --git a/scripts/smoke_explore_n2r2.py b/scripts/smoke_explore_n2r2.py
new file mode 100644
index 0000000..a4af258
--- /dev/null
+++ b/scripts/smoke_explore_n2r2.py
@@ -0,0 +1,760 @@
+#!/usr/bin/env python3
+"""Smoke for N2R2 (addendum L-AM24-ADD-2; AM-29's E6-R2-M): src/explore/qat_explore.py n2r2_parent,
+records_root_record and run_n2r2, scripts/explore_qat.py's N2R2 gates and scripts/explore_qat_blockb.py's N2R2 route
+and card fix 1. Synthetic students, loaders, run folders and a synthetic records checkout (git, in the scratch folder;
+the condition record made by the pinned writer, as scripts/smoke_am23_quant.py makes it); no dataset, no GPU, no TEST.
+
+  * the constants against the records: E3R2_PARENT_OF_RECORD is DL-136's row and the addendum's da70d82f...; E6-R2's
+    stage, parent arm and AM-23 rules file; the committed λ and α selections read through the pinned reader with the
+    sha256s the launch command gives; the condition record's path of record, absent at this checkout;
+  * flags and seeds: N2's flag set exactly (the same object), seed 42 only; N2's own unchanged;
+  * the parent check: it accepts the listed hash and identity through E6-R2's loader (the same resolve_parent record
+    and AM-23 binding as E6-R2's own gates give) and refuses other hashes (N2's E2 parent included), a wrong identity
+    (run ID, best.json, run_meta, telemetry, value), D's flag, another seed and an E2 folder; the pinned binding refuses
+    a condition record that is missing, changed after its commit or of another sha256, a decision-log row that is not
+    RECORDED or absent, another arm, a wrong records commit or folder, and this checkout as the records root (no record
+    here); N2 refuses E3-R2's parent;
+  * through the pinned run_qat on a synthetic student: all ten sites fire, TF32 off is in run_meta, the substitution is
+    recorded and the pinned name restored, run_meta is E6-R2's; a real run refuses the change removed and injected
+    loaders;
+  * with the substitution removed, N2R2's first-step loss (and CE, Dice, gradient norm, batch, epoch state, recipe
+    fields) equals E6-R2's path, the pinned loader and run_qat, on one synthetic batch; with it, the loss differs;
+  * N2's launch gate passes on a calibrated copy of the parent E6-R2's loader built;
+  * block B: an N2R2 run is accepted and runs the pinned scripts (convert refuses the 2-epoch smoke run, card fix 1
+    recorded as not run), an unknown item and an N2R2 item whose run is not E6-R2 are refused; card fix 1's command
+    passes the pinned evaluator's own CLI checks for N2 (stage E6-E2) and is refused for N2R2 (E6-R2 is not in
+    src/eval/artifacts.py FQOFF_STAGES), in process and by the script itself; the record of a scored, a failed and a
+    mismatched step;
+  * the CLI: N2R2 refuses another seed, a missing records flag, another run ID, TEST and a data root that fails the
+    TRAIN/VAL-only check, writing nothing; N2 refuses N2R2's flags; block B's CLI accepts N2R2 up to the guard.
+Mutations: a permissive identity check, N2R2 on E6-E2's stage, E6-R2's binding skipped, run_n2r2 without the sites
+and a block B without card fix 1 each fail a case.
+
+    python -I scripts/smoke_explore_n2r2.py        (exit 0 PASS, 1 FAIL)
+"""
+from __future__ import annotations
+
+import contextlib
+import hashlib
+import importlib.util
+import io
+import json
+import os
+import subprocess
+import sys
+from pathlib import Path
+from types import SimpleNamespace
+
+sys.dont_write_bytecode = True
+REPO = Path(__file__).resolve().parents[1]
+if str(REPO) not in sys.path:
+    sys.path.insert(0, str(REPO))
+
+import torch                                                          # noqa: E402
+from torch.utils.data import DataLoader, TensorDataset              # noqa: E402
+
+from src.explore import guard as G                                   # noqa: E402
+from src.explore import qat_explore as X                             # noqa: E402
+from src.explore.sites import SITE_NAMES, tf32_flags                 # noqa: E402
+from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402
+from src.explore.synthetic import random_student                    # noqa: E402
+from src.quant import e6x_condition as XC                            # noqa: E402
+from src.quant import qat as Q                                       # noqa: E402
+from src.quant.prepare import prepare_qat_model as PINNED_PREPARE   # noqa: E402
+
+LAM_REL, ALPHA_REL, DL_REL = ("reports/derived/lambda_selection.json", "reports/derived/alpha_selection.json",
+                              "docs/DECISION_LOG.md")
+COND_REL = XC.RECORD_REL.format(arm="R2")
+# The λ and α selections of record: whole-file sha256 of the committed files (the same blobs at fe93a24 and at master
+# aa8cac2), the values the launch command passes.
+LAM_SHA = "719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145"
+ALPHA_SHA = "1399b897278db17a8876c724b89c8d8e7a7b42ffe1d9c4d67002578a889f2dd8"
+REC = X.E3R2_PARENT_OF_RECORD
+DL_ID = "DL-950"
+N2_FLAGS = {"sites_on_input_grid", "tf32_off"}
+_CACHE: dict = {}
+
+
+def quiet(*_a, **_k):
+    return None
+
+
+def _sha(p) -> str:
+    return G.sha256_file(p)
+
+
+def _jl(rows) -> str:
+    return "".join(json.dumps(r) + "\n" for r in rows)
+
+
+def _module(path: Path):
+    """A script imported as a module: its top level only (the scripts used here run nothing on import)."""
+    key = ("module", str(path))
+    if key not in _CACHE:
+        spec = importlib.util.spec_from_file_location(f"x_smoke_{path.stem}", path)
+        m = importlib.util.module_from_spec(spec)
+        sys.modules[spec.name] = m                 # a dataclass in the module looks itself up there
+        spec.loader.exec_module(m)
+        _CACHE[key] = m
+    return _CACHE[key]
+
+
+def blockb():
+    return _module(REPO / "scripts" / "explore_qat_blockb.py")
+
+
+def evaluator():
+    return _module(REPO / "scripts" / "evaluate_model.py")
+
+
+def _git(repo: Path, *args: str) -> str:
+    """git in a synthetic records checkout, as scripts/smoke_am23_quant.py runs it: a fixed identity and date, no global
+    or system configuration."""
+    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
+    env.update({"GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
+                "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
+                "GIT_AUTHOR_DATE": "2026-11-15T00:00:00Z", "GIT_COMMITTER_DATE": "2026-11-15T00:00:00Z",
+                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"})
+    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout
+
+
+# ------------------------------------------------------------------ fixtures
+def student_state(seed: int = 7) -> dict:
+    key = ("state", seed)
+    if key not in _CACHE:
+        _CACHE[key] = {k: v.detach().clone() for k, v in random_student(seed=seed).state_dict().items()}
+    return _CACHE[key]
+
+
+def e3r2_meta(**over) -> dict:
+    """E3-R2's run_meta row with the fields AM-23 item 4's binding reads (scripts/smoke_am23_quant.py e3_meta)."""
+    row = {"event": "run_meta", "stage": "E3", "mode": "real", "seed": 42,
+           "terms": {"logit_kd": True, "cwd_feat": True, "cwd_logit": True}, "lambda_logit": 0.5, "alpha_cwd": 25.0,
+           "max_iters": 80000, "num_classes": 116, "poly_horizon": 80000, "teacher_mock": False, "arm": "E3-R2",
+           "descriptive": True, "parent_of_e4_e7": False, "records_commit": "b" * 40,
+           "target_construction": {"K": 2, "views": 2}}
+    row.update(over)
+    return row
+
+
+def e3r2_run(base: Path, *, name: str = REC["run_id"], meta: dict | None = None) -> Path:
+    """A fixture E3-R2 run folder: e3_run_meta.jsonl, e3_telemetry.jsonl ending in run_end, best.json (DL-136's value)
+    and a checkpoint holding a full random student (stage E3), so the strict E3 loader builds it."""
+    v, ck = REC["best_val"], REC["checkpoint"]
+    d = base / name
+    d.mkdir(parents=True)
+    (d / "e3_run_meta.jsonl").write_text(_jl([meta or e3r2_meta()]), encoding="utf-8")
+    end = {"event": "run_end", "iter": 80000, "best_val_miou_all_class": v, "best_ckpt": ck, "checks_passed": True,
+           "wall_clock_end": 1792665391.5086315}
+    (d / "e3_telemetry.jsonl").write_text(_jl([{"event": "val", "iter": 80000, "all_class_miou": v}, end]),
+                                          encoding="utf-8")
+    (d / "best.json").write_text(json.dumps({"best_ckpt": f"/workspace/kd_runs/{name}/{ck}",
+                                             "best_val_miou_all_class": v}, indent=2), encoding="utf-8")
+    torch.save({"stage": "E3", "model_state_dict": student_state(), "num_classes": 116, "best_val_miou_all_class": v,
+                "arm": "E3-R2", "target_construction": {"K": 2, "views": 2}}, d / ck)
+    return d
+
+
+def e2_folder(base: Path) -> Path:
+    """An E2-layout folder (e2_ records, an E2 checkpoint name) named as E3-R2's run."""
+    d = base / REC["run_id"]
+    d.mkdir(parents=True)
+    ck = "e2_student_best_iter64000.pt"
+    (d / "e2_run_meta.jsonl").write_text(_jl([{"event": "run_meta", "stage": "E2", "mode": "real", "seed": 42}]))
+    (d / "e2_telemetry.jsonl").write_text(_jl([{"event": "run_end", "iter": 80000, "checks_passed": True}]))
+    (d / "best.json").write_text(json.dumps({"best_ckpt": ck, "best_val_miou_all_class": 0.38}))
+    (d / ck).write_bytes(b"e2 fixture checkpoint")
+    return d
+
+
+def table_of(d: Path, **over) -> dict:
+    """E3R2_PARENT_OF_RECORD with a fixture folder's hashes (the real values name files that exist only on the laptop
+    and a pod)."""
+    t = {**REC, "sha256": _sha(d / REC["checkpoint"]), "best_json_sha256": _sha(d / "best.json"),
+         "run_meta_sha256": _sha(d / "e3_run_meta.jsonl"), "telemetry_sha256": _sha(d / "e3_telemetry.jsonl")}
+    t.update(over)
+    return t
+
+
+def selection(rel: str, sha: str) -> dict:
+    doc = json.loads((REPO / rel).read_bytes().decode("utf-8"))
+    return {"path": rel, "sha256": sha, "format": doc["format"], "winner": doc["winner"]}
+
+
+def condition_record(d: Path) -> dict:
+    """AM-23 item 3(b)'s record of a fixture run, by the pinned writer's build_record from the folder's own files (as
+    scripts/smoke_am23_quant.py condition() makes it)."""
+    meta_data, tel_data = (d / XC.META_NAME).read_bytes(), (d / XC.TELEMETRY_NAME).read_bytes()
+    best_doc = json.loads((d / XC.BEST_JSON_NAME).read_text(encoding="utf-8"))
+    ck = d / REC["checkpoint"]
+    best = {"value": best_doc["best_val_miou_all_class"], "best_json_sha256": _sha(d / XC.BEST_JSON_NAME),
+            "checkpoint": REC["checkpoint"], "checkpoint_sha256": _sha(ck), "checkpoint_bytes": ck.stat().st_size}
+    end = json.loads(tel_data.decode("utf-8").splitlines()[-1])
+    return XC.build_record(arm="R2", spec=SimpleNamespace(run_stage="E3-R2"), run_id=d.name, best=best,
+                           meta_data=meta_data, tel_data=tel_data, end=end, lam=selection(LAM_REL, LAM_SHA),
+                           alpha=selection(ALPHA_REL, ALPHA_SHA),
+                           teacher={"path": "reports/arms/arm_teacher_record_R2.json", "sha256": "8" * 64,
+                                    "checkpoint_sha256": "9" * 64},
+                           launch={"path": "reports/kd_launch_log.jsonl", "sha256": "a" * 64},
+                           tool={"script": "scripts/e6x_condition.py", "module": "src/quant/e6x_condition.py",
+                                 "commit": "6" * 40, "clean_at_commit": True})
+
+
+def records_checkout(root: Path, d: Path, *, record: bool = True, status: str = "RECORDED (synthetic)",
+                     after: bytes | None = None) -> tuple[Path, str, str]:
+    """A synthetic records checkout committed once: the λ and α selections of record (their committed bytes), E3-R2's
+    condition record at its path of record and a decision log whose DL-950 carries the record's text. `record` False
+    leaves the record out; `after` rewrites it after the commit. Returns (root, HEAD, the record's sha256)."""
+    root.mkdir(parents=True)
+    for rel in (LAM_REL, ALPHA_REL):
+        (root / rel).parent.mkdir(parents=True, exist_ok=True)
+        (root / rel).write_bytes((REPO / rel).read_bytes())
+    rec = condition_record(d)
+    data = XC.record_bytes(rec)
+    rec_sha = hashlib.sha256(data).hexdigest()
+    files = [LAM_REL, ALPHA_REL, DL_REL]
+    if record:
+        (root / COND_REL).parent.mkdir(parents=True, exist_ok=True)
+        XC.write_record(root / COND_REL, data)
+        files.append(COND_REL)
+    (root / DL_REL).parent.mkdir(parents=True, exist_ok=True)
+    (root / DL_REL).write_text("# Decision log (synthetic)\n\n| ID | Status | Entry |\n|---|---|---|\n"
+                               f"| {DL_ID} | {status} | {XC.dl_text(rec, rec_sha)} |\n", encoding="utf-8")
+    _git(root, "init", "-q")
+    _git(root, "add", "--", *files)
+    _git(root, "commit", "-q", "-m", "synthetic records")
+    if after is not None:
+        (root / COND_REL).write_bytes(after)
+    return root, _git(root, "rev-parse", "HEAD").strip(), rec_sha
+
+
+def fixtures() -> dict:
+    """Built once per process; the mutation passes reuse them."""
+    if "fx" in _CACHE:
+        return _CACHE["fx"]
+    base = scratch_dir("x_n2r2_fx_")
+    fx = {"base": base, "d": e3r2_run(base / "runs")}
+    fx["root"], fx["head"], fx["rec_sha"] = records_checkout(base / "recs", fx["d"])
+    fx["root_norec"], fx["head_norec"], _ = records_checkout(base / "recs_norec", fx["d"], record=False)
+    fx["root_decided"], fx["head_decided"], _ = records_checkout(base / "recs_decided", fx["d"],
+                                                                 status="DECIDED (synthetic)")
+    fx["root_changed"], fx["head_changed"], _ = records_checkout(base / "recs_changed", fx["d"], after=b"{}\n")
+    fx["d_a1"] = e3r2_run(base / "runs_a1", name="e3-r2_s42_lambda0.5_alpha25_80k_a1")
+    fx["d_arm"] = e3r2_run(base / "runs_arm", meta=e3r2_meta(arm="E3-R1"))
+    fx["d_e2"] = e2_folder(base / "runs_e2")
+    fx["repo_head"] = _git(REPO, "rev-parse", "HEAD").strip()
+    _CACHE["fx"] = fx
+    return fx
+
+
+def n2r2_kw(fx: dict, **over) -> dict:
+    kw = {"seed": 42, "expect_source_sha256": _sha(fx["d"] / REC["checkpoint"]), "records_root": str(fx["root"]),
+          "records_commit": fx["head"], "lambda_selection": LAM_REL, "lambda_selection_sha256": LAM_SHA,
+          "alpha_selection": ALPHA_REL, "alpha_selection_sha256": ALPHA_SHA, "condition_record": COND_REL,
+          "condition_sha256": fx["rec_sha"], "condition_dl": DL_ID}
+    kw.update(over)
+    return kw
+
+
+def one_batch(n: int = 2, size: int = 64, seed: int = 0):
+    """One synthetic TRAIN batch (and the same images as the VAL loader)."""
+    g = torch.Generator().manual_seed(seed)
+    imgs = torch.randn(n, 3, size, size, generator=g)
+    masks = torch.randint(0, 116, (n, size, size), generator=g)
+    masks[:, :4, :] = 255
+    ds = TensorDataset(imgs, masks)
+    return (DataLoader(ds, batch_size=n, shuffle=False, drop_last=True),
+            DataLoader(ds, batch_size=n, shuffle=False, drop_last=False))
+
+
+def rows_of(out: Path) -> list[dict]:
+    return [json.loads(ln) for ln in (out / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()]
+
+
+def first(rows: list[dict], event: str) -> dict:
+    return next((r for r in rows if r.get("event") == event), {})
+
+
+def tf32_restore(before: dict) -> None:
+    torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
+    torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
+
+
+def smoke_qat(model, out: Path, *, stage: str = "e6-r2", parent=None, extra=None, steps: int = 1, meta=None):
+    """The pinned run_qat in smoke mode on one batch (E6-R2's path when stage is e6-r2)."""
+    return Q.run_qat(stage=stage, mode="smoke", model=model, source_meta=meta or {"sha256": "0" * 64}, out_dir=out,
+                     seed=42, clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0, loaders=one_batch(),
+                     parent=parent, extra_meta=extra, max_steps=steps, max_val_batches=1, log=quiet)
+
+
+def n2r2_smoke(model, out: Path, *, sites: bool = True, parent=None, extra=None, steps: int = 1, meta=None):
+    return X.run_n2r2(mode="smoke", model=model, source_meta=meta or {"sha256": "0" * 64}, out_dir=out, seed=42,
+                      clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0, parent=parent,
+                      extra_meta=extra, sites=sites, loaders=one_batch(), max_steps=steps, max_val_batches=1,
+                      log=quiet)
+
+
+class Done:
+    """What subprocess.run returns, for the injected runners."""
+
+    def __init__(self, argv, code: int, out: str = "", err: str = ""):
+        self.args, self.returncode, self.stdout, self.stderr = argv, code, out, err
+
+
+def evaluator_cli_runner(argv, **_kw):
+    """The pinned evaluator's own CLI checks (build_parser, validate_cli_args) on block B's command, in process; when
+    they pass, a summary.json with the fields block B reads, as the evaluator's artifact holds them (synthetic: no
+    dataset here, so no score is made)."""
+    em = evaluator()
+    if argv[1:3] != ["-B", "scripts/evaluate_model.py"]:
+        return Done(argv, 9, "", f"not the evaluator: {argv[1:3]}")
+    a = em.build_parser().parse_args(argv[3:])
+    try:
+        em.validate_cli_args(a)
+    except em.CliError as e:
+        return Done(argv, 1, "", f"CliError: {e}")
+    out = Path(a.out_dir)
+    out.mkdir(parents=True)
+    (out / "summary.json").write_text(json.dumps({
+        "run": {"stage": a.stage, "precision": a.precision, "checkpoint_sha256": a.expected_checkpoint_sha256,
+                "eval_runtime": {"torch_num_threads": 8}},
+        "dataset": {"split": a.split, "expected_rows": 846, "actual_rows": 846},
+        "dataset_level": {"all_class_miou": 0.3412}}), encoding="utf-8")
+    return Done(argv, 0, f"artifact written: {out}")
+
+
+def script_runner(argv, **kw):
+    """The script itself, with no data root in its environment."""
+    env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
+    return subprocess.run(argv, env=env, **kw)
+
+
+def selection_doc(run: Path, epoch: int) -> dict:
+    """A synthetic qat_selection.json naming `epoch` of a smoke run (no rules_sha256: no rules selected it)."""
+    from src.quant.qat_select import EPOCH_SELECTION_FORMAT
+    ends = {r["epoch"]: r for r in rows_of(run) if r.get("event") == "epoch_end"}
+    return {"format": EPOCH_SELECTION_FORMAT, "run_id": run.name,
+            "winner": {"epoch": epoch, "checkpoint": f"epoch_ckpts/e{epoch:02d}.pt",
+                       "checkpoint_sha256": ends[epoch]["checkpoint_sha256"], "value": 0.35}}
+
+
+def blockb_main(argv) -> tuple[int, str]:
+    """Block B's main in process with the guard's data-root step given a fixture record (this session has no data
+    root; the guard smoke covers that step), stdout captured."""
+    iso = {"checked": True, "data_root": "<synthetic>", "counts": {"train": 5367, "val": 846},
+           "reason": "smoke: the TRAIN/VAL-only check is smoke_explore_guard's"}
+    buf = io.StringIO()
+    with patched(G, "check_data_root", lambda records_only=False: dict(iso)), contextlib.redirect_stdout(buf):
+        code = blockb().main(argv)
+    return code, buf.getvalue()
+
+
+# ------------------------------------------------------------------ the cases
+def run_cases(verbose: bool = False, full: bool = True) -> Cases:
+    c = Cases("SMOKE_EXPLORE_N2R2", verbose=verbose)
+    fx = fixtures()
+    work = scratch_dir("x_n2r2_run_")
+    d, d_ck = fx["d"], fx["d"] / REC["checkpoint"]
+    st = Q.resolve_quant_stage("e6-r2")
+    # ---- 1. the constants against the records
+    dl = (REPO / DL_REL).read_text(encoding="utf-8")
+    row136 = next((ln for ln in dl.splitlines() if ln.startswith("| DL-136 |")), "")
+    c.check("E3-R2 seed 42's checkpoint sha256 is the addendum's da70d82f...",
+            REC["sha256"] == "da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4")
+    vals = [REC[k] for k in ("run_id", "checkpoint", "sha256", "best_json_sha256", "run_meta_sha256",
+                             "telemetry_sha256")] + [repr(REC["best_val"])]
+    c.check("the parent record is DL-136's: run ID, checkpoint, its sha256, best.json, run_meta, telemetry, value",
+            bool(row136) and all(v in row136 for v in vals) and "E6-R2 condition (AM-23 item 3(b)): yes" in row136,
+            str([v for v in vals if v not in row136]))
+    from src.quant import qat_select as S
+    c.check("N2R2's stage is E6-R2: source E3, parent arm E3-R2, an AM-23 stage, scripts/run_e6_r2.py's STAGE",
+            X.N2R2_STAGE == "e6-r2" and st["name"] == "E6-R2" and st["source_stage"] == "E3"
+            and st.get("parent_arm") == "E3-R2" and "E6-R2" in Q.AM23_STAGES
+            and _module(REPO / "scripts" / "run_e6_r2.py").STAGE == X.N2R2_STAGE)
+    c.check("E6-R2's epoch rules are AM-23's file (qat_select.rules_path_for_stage)",
+            Path(S.rules_path_for_stage("E6-R2")).resolve()
+            == (REPO / "configs" / "qat_selection_rules_am23.json").resolve())
+    lam = c.passes("the λ selection of record reads through the pinned reader with the command's sha256",
+                   Q.read_tracked_selection, LAM_REL, LAM_SHA, Q.LAMBDA_SELECTION_FORMAT, "lambda", REPO)
+    alpha = c.passes("the α selection of record reads through the pinned reader with the command's sha256",
+                     Q.read_tracked_selection, ALPHA_REL, ALPHA_SHA, Q.ALPHA_SELECTION_FORMAT, "alpha", REPO)
+    c.check("their winners: λ 0.5; α 25.0 with best_val 0.36337557435035706 (AM-23 item 3(b)'s threshold)",
+            bool(lam) and bool(alpha) and lam["winner"]["lambda"] == 0.5 and alpha["winner"]["alpha"] == 25.0
+            and repr(alpha["winner"]["best_val"]) == XC.THRESHOLD)
+    c.check("the condition record's path of record is reports/arms/e6x_condition_R2.json; it is absent here",
+            COND_REL == "reports/arms/e6x_condition_R2.json" and not (REPO / COND_REL).exists())
+    c.check("N2R2's run ID is x-n2r2-s42 and its records flags are E6-R2's flag names",
+            X.N2R2_RUN_ID == "x-n2r2-s42"
+            and {"lambda_selection", "alpha_selection", "condition_record", "condition_sha256",
+                 "condition_dl"} <= set(X.N2R2_RECORD_FLAGS))
+    # ---- 2. flags and seeds
+    c.check("N2R2's flag set is N2's (the same object)", X.ITEM_FLAGS["N2R2"] is X.ITEM_FLAGS["N2"])
+    c.check("N2R2 with N2's two flags at seed 42 accepted", X.item_flag_error("N2R2", N2_FLAGS, 42) is None)
+    c.check("N2R2 with one flag, none, an added flag or N5's refused (item_flags)",
+            all((X.item_flag_error("N2R2", f, 42) or ("",))[0] == "item_flags"
+                for f in ({"tf32_off"}, {"sites_on_input_grid"}, set(), N2_FLAGS | {"kd_logit"}, {"fp32_head"})))
+    c.check("N2R2 at seed 43 and 44 refused (item_seed)",
+            all((X.item_flag_error("N2R2", N2_FLAGS, s) or ("",))[0] == "item_seed" for s in (43, 44)))
+    c.check("N2 unchanged: its two flags at seed 42 accepted, seed 43 refused",
+            X.item_flag_error("N2", N2_FLAGS, 42) is None
+            and (X.item_flag_error("N2", N2_FLAGS, 43) or ("",))[0] == "item_seed")
+    # ---- 3. the parent check
+    got = parent_e = binding_e = None
+    with patched(X, "E3R2_PARENT_OF_RECORD", table_of(d)):
+        got = c.passes("the parent check accepts the listed hash and identity, through E6-R2's loader",
+                       X.n2r2_parent, d, **n2r2_kw(fx))
+        b = (got or {}).get("am23_binding") or {}
+        c.check("... stage E6-R2, binding arm E3-R2, condition yes from DL-950, the records commit recorded",
+                bool(got) and got["stage"]["name"] == "E6-R2" and b.get("arm") == "E3-R2"
+                and (b.get("condition") or {}).get("verdict") == "yes" and (b.get("condition") or {}).get("dl") == DL_ID
+                and got["records_root"]["commit"] == fx["head"]
+                and got["parent"]["of_record"]["run_id"] == REC["run_id"])
+        parent_e = Q.resolve_parent(st, d, _sha(d_ck), 42)
+        lam_r = Q.read_tracked_selection(LAM_REL, LAM_SHA, Q.LAMBDA_SELECTION_FORMAT, "lambda", fx["root"])
+        alpha_r = Q.read_tracked_selection(ALPHA_REL, ALPHA_SHA, Q.ALPHA_SELECTION_FORMAT, "alpha", fx["root"])
+        binding_e = Q.am23_parent_binding(st, parent_e, SimpleNamespace(seed=42, condition_record=COND_REL,
+                                                                        condition_sha256=fx["rec_sha"],
+                                                                        condition_dl=DL_ID),
+                                          (lam_r, alpha_r), repo_root=fx["root"])
+        c.check("N2R2's parent record and binding are E6-R2's own (resolve_parent, am23_parent_binding)",
+                bool(got) and {k: v for k, v in got["parent"].items() if k != "of_record"} == parent_e
+                and got["am23_binding"] == binding_e)
+        for case, code, over in (
+                ("another hash refused", "parent_not_of_record", {"expect_source_sha256": "0" * 64}),
+                ("N2's E2 parent (73dbc32d...) refused", "parent_not_of_record",
+                 {"expect_source_sha256": X.E2_PARENTS_OF_RECORD[42]["sha256"]}),
+                ("D's --parent-sha256 refused", "parent_sha256_not_d", {"parent_sha256": _sha(d_ck)}),
+                ("seed 43 refused", "item_seed", {"seed": 43}),
+                ("a records commit that is not the checkout's HEAD refused", "records_commit_mismatch",
+                 {"records_commit": "0" * 40}),
+                ("a malformed records commit refused", "records_commit_format", {"records_commit": "abc"}),
+                ("a records folder that is not a checkout refused", "records_root_not_checkout",
+                 {"records_root": str(fx["base"] / "runs")}),
+                ("a missing records folder refused", "records_root_missing",
+                 {"records_root": str(fx["base"] / "absent")}),
+                ("the λ selection of another sha256 refused (pinned reader)", "selection_sha256_mismatch",
+                 {"lambda_selection_sha256": "0" * 64}),
+                ("a condition record of another sha256 refused (pinned binding)", "condition_sha256_mismatch",
+                 {"condition_sha256": "0" * 64}),
+                ("a decision-log row that does not exist refused (pinned binding)", "condition_dl_row",
+                 {"condition_dl": "DL-951"}),
+                ("an uncommitted condition record refused: none in the checkout (pinned binding)",
+                 "condition_missing_file", {"records_root": str(fx["root_norec"]), "records_commit": fx["head_norec"]}),
+                ("a decision-log row that is not RECORDED refused (pinned binding)", "condition_dl_row",
+                 {"records_root": str(fx["root_decided"]), "records_commit": fx["head_decided"]}),
+                ("a condition record changed after its commit refused (pinned binding)",
+                 "condition_changed_since_head",
+                 {"records_root": str(fx["root_changed"]), "records_commit": fx["head_changed"]}),
+                ("this checkout as the records root refused: no condition record at this HEAD",
+                 "condition_missing_file", {"records_root": str(REPO), "records_commit": fx["repo_head"]})):
+            c.refuses(f"N2R2 parent: {case}", code, X.n2r2_parent, d, **n2r2_kw(fx, **over))
+    for case, folder, table in (
+            ("another run's folder (attempt a1) under the record's hash", fx["d_a1"],
+             table_of(fx["d_a1"], run_id=REC["run_id"])),
+            ("a best.json other than DL-136's", d, table_of(d, best_json_sha256="0" * 64)),
+            ("a run_meta other than DL-136's", d, table_of(d, run_meta_sha256="0" * 64)),
+            ("a telemetry other than DL-136's", d, table_of(d, telemetry_sha256="0" * 64)),
+            ("a best value other than DL-136's", d, table_of(d, best_val=0.38))):
+        with patched(X, "E3R2_PARENT_OF_RECORD", table):
+            c.refuses(f"N2R2 parent: {case} refused", "parent_identity", X.n2r2_parent, folder,
+                      **n2r2_kw(fx, expect_source_sha256=table["sha256"]))
+    with patched(X, "E3R2_PARENT_OF_RECORD", table_of(fx["d_arm"])):
+        c.refuses("N2R2 parent: a run of arm E3-R1 refused (pinned binding)", "parent_arm_mismatch", X.n2r2_parent,
+                  fx["d_arm"], **n2r2_kw(fx, expect_source_sha256=_sha(fx["d_arm"] / REC["checkpoint"])))
+    e2ck = fx["d_e2"] / "e2_student_best_iter64000.pt"
+    with patched(X, "E3R2_PARENT_OF_RECORD", {**REC, "sha256": _sha(e2ck)}):
+        c.refuses("N2R2 parent: an E2 folder refused by E6-R2's loader (no e3 records)", "parent_records_missing",
+                  X.n2r2_parent, fx["d_e2"], **n2r2_kw(fx, expect_source_sha256=_sha(e2ck)))
+    c.refuses("N2R2 parent with the real table: the fixture's hash refused", "parent_not_of_record", X.n2r2_parent, d,
+              **n2r2_kw(fx))
+    c.refuses("N2 refuses E3-R2's parent: da70d82f... is not E2 seed 42's", "parent_not_of_record",
+              X.exploratory_parent, "N2", d, seed=42, expect_source_sha256=REC["sha256"])
+    with patched(X, "E2_PARENTS_OF_RECORD", {**X.E2_PARENTS_OF_RECORD,
+                                             42: {**X.E2_PARENTS_OF_RECORD[42], "sha256": _sha(d_ck)}}):
+        c.refuses("N2 refuses an E3-R2 folder even under its hash: E6-E2's loader needs e2 records",
+                  "parent_records_missing", X.exploratory_parent, "N2", d, seed=42, expect_source_sha256=_sha(d_ck))
+    # ---- 4. through the pinned run_qat
+    item = work / "x-n2r2-s42"
+    out_s = item / "x-n2r2-s42"
+    item.mkdir()
+    res = None
+    if got:
+        model, meta = Q.load_source(got["stage"], got["parent"])
+        before = tf32_flags()
+        try:
+            res = c.passes("run_n2r2 runs the pinned run_qat on E6-R2's parent for 2 smoke steps", n2r2_smoke, model,
+                           out_s, parent=got["parent"], extra={"am23_parent": got["am23_binding"], "x_item": "N2R2"},
+                           steps=2, meta=meta)
+        finally:
+            tf32_restore(before)
+    _summary, sub, matcher = res or (None, None, None)
+    c.check("all ten sites were installed and fired", matcher is not None and tuple(matcher.calls) == SITE_NAMES
+            and all(v > 0 for v in matcher.calls.values()), str(None if matcher is None else matcher.calls))
+    m0 = rows_of(out_s)[0] if (out_s / Q.TELEMETRY_NAME).is_file() else {}
+    c.check("run_meta records TF32 off (cuDNN and matmul), as read and as set",
+            (m0.get("tf32") or {}).get("cudnn_allow_tf32") is False
+            and (m0.get("tf32") or {}).get("cuda_matmul_allow_tf32") is False
+            and (m0.get("x_tf32_set") or {}).get("cudnn_allow_tf32") is False, str(m0.get("tf32")))
+    c.check("run_meta records the substitution: qat.py's prepare_qat_model, the file's sha256, item N2R2",
+            sub is not None and m0.get("x_substitution") == sub and sub["pinned_file"] == "src/quant/qat.py"
+            and sub["attribute"] == "prepare_qat_model" and sub["item"] == "N2R2"
+            and sub["pinned_file_sha256"] == _sha(REPO / "src" / "quant" / "qat.py"))
+    c.check("the pinned prepare_qat_model is restored afterwards", Q.prepare_qat_model is PINNED_PREPARE)
+    c.check("run_meta is E6-R2's: stage, source stage, no projection, the parent's sha256, run ID and AM-23 binding",
+            bool(got) and m0.get("stage") == "E6-R2" and m0.get("source_stage") == "E3"
+            and m0.get("cwd_projection_loaded") is False
+            and m0.get("source_checkpoint_sha256") == got["parent"]["checkpoint_sha256"]
+            and (m0.get("parent") or {}).get("run_id") == REC["run_id"] and m0.get("x_item") == "N2R2"
+            and json.dumps(m0.get("am23_parent"), sort_keys=True) == json.dumps(got["am23_binding"], sort_keys=True))
+    before = tf32_flags()
+    try:
+        c.refuses("a real N2R2 run refuses the change removed", "n2r2_change_required", X.run_n2r2, mode="real",
+                  model=random_student(seed=5), source_meta={"sha256": "0" * 64}, out_dir=work / "r1" / "x-n2r2-r",
+                  seed=42, clip_norm=5.0, clip_source="clip_selection", device="cpu", num_workers=1, parent=None,
+                  sites=False)
+        c.refuses("a real N2R2 run refuses injected loaders (run_qat's own refusal)", "real_run_test_hooks",
+                  X.run_n2r2, mode="real", model=random_student(seed=5), source_meta={"sha256": "0" * 64},
+                  out_dir=work / "r2" / "x-n2r2-r", seed=42, clip_norm=5.0, clip_source="clip_selection",
+                  device="cpu", num_workers=1, parent=None, loaders=one_batch())
+    finally:
+        tf32_restore(before)
+    c.check("... and the pinned name is restored after the refusal", Q.prepare_qat_model is PINNED_PREPARE)
+    # ---- 5. the first step: N2R2 with the substitution removed = E6-R2's path
+    te = tr = tx = {}
+    if got and parent_e is not None:
+        model_e, meta_e = Q.load_source(st, parent_e)                 # E6-R2's own loader and parent record
+        model_r, meta_r = Q.load_source(got["stage"], got["parent"])   # N2R2's
+        c.check("the same source: both loads read one checkpoint into equal states",
+                meta_e["sha256"] == meta_r["sha256"] == _sha(d_ck)
+                and Q.state_digest(model_e) == Q.state_digest(model_r))
+        out_e, out_r, out_x = work / "e6r2" / "x-e6r2-path", work / "nosites" / "x-n2r2-nosites", \
+            work / "sites1" / "x-n2r2-sites1"
+        smoke_qat(model_e, out_e, parent=parent_e, extra={"am23_parent": binding_e}, meta=meta_e)
+        n2r2_smoke(model_r, out_r, sites=False, parent=got["parent"], extra={"am23_parent": got["am23_binding"]},
+                   meta=meta_r)
+        re_, rr_ = rows_of(out_e), rows_of(out_r)
+        te, tr = first(re_, "train"), first(rr_, "train")
+        c.check("with the substitution removed, N2R2's first-step loss equals E6-R2's path (one synthetic batch)",
+                isinstance(te.get("loss"), float) and te.get("loss") == tr.get("loss"),
+                f"{te.get('loss')!r} vs {tr.get('loss')!r}")
+        c.check("... and its CE, Dice, gradient norm, lr and batch fingerprint, and the state after the step",
+                all(te.get(k) == tr.get(k) for k in ("ce", "dice", "grad_norm", "lr", "batch_sha256"))
+                and first(re_, "epoch_end").get("state_sha256") == first(rr_, "epoch_end").get("state_sha256"))
+        keys = ("stage", "source_stage", "seed", "clip_norm", "epochs", "batch_size", "drop_last", "steps_per_epoch",
+                "total_steps", "scheduler", "optimizer", "lr", "momentum", "weight_decay", "nesterov", "dampening",
+                "bn_freeze_after_step", "obs_freeze_after_step", "objective", "class_weights_sha256", "engine",
+                "qconfig", "observer_census", "fake_quant_modules", "never_observed_modules", "fused_convbn",
+                "unfused_batchnorm", "tf32", "source_checkpoint_sha256", "cwd_projection_loaded", "am23_parent",
+                "code_files_sha256")
+        c.check("... and run_meta's recipe, census, TF32, source and binding fields",
+                all(re_[0].get(k) == rr_[0].get(k) for k in keys),
+                str([k for k in keys if re_[0].get(k) != rr_[0].get(k)]))
+        before = tf32_flags()
+        try:
+            model_x, meta_x = Q.load_source(st, parent_e)
+            n2r2_smoke(model_x, out_x, sites=True, parent=got["parent"], meta=meta_x)
+        finally:
+            tf32_restore(before)
+        tx = first(rows_of(out_x), "train")
+    c.check("with the substitution, the first-step loss differs: the comparison discriminates",
+            isinstance(tx.get("loss"), float) and tx.get("loss") != te.get("loss"),
+            f"{tx.get('loss')!r} vs {te.get('loss')!r}")
+    if full and parent_e is not None:
+        from src.explore.sites import calibrated_copy, require_site_set
+        c.passes("N2's launch gate on N2R2's parent: a calibrated copy of the loaded E3-R2 student gives the ten sites",
+                 require_site_set, calibrated_copy(Q.load_source(st, parent_e)[0]))
+    # ---- 6. block B
+    BB = blockb()
+    (item / G.RUN_META_NAME).write_text(json.dumps({
+        "exploratory": True, "item": "N2R2", "run_id": "x-n2r2-s42", "seeds": [42], "amendment": G.AMENDMENT,
+        "precision_label": "int8 QNNPACK (as E6-R2)", "claim_line": G.CLAIM_LINE.format(seeds=42)}), encoding="utf-8")
+    doc = c.passes("block B accepts an N2R2 run (run_item)", BB.run_item, out_s)
+    c.check("... a registered item with card fix 1, its run of stage E6-R2",
+            bool(doc) and doc["item"] == "N2R2" and "N2R2" in BB.REGISTERED_ITEMS and "N2R2" in BB.FQOFF_ITEMS
+            and BB.ITEM_STAGE.get("N2R2") == "E6-R2" and (out_s / Q.TELEMETRY_NAME).is_file()
+            and BB.run_stage(out_s) == "E6-R2")
+    c.check("block B gives card fix 1 and the stage check to N2 and N2R2 only (C1, N1, D as at 60dc294)",
+            tuple(BB.FQOFF_ITEMS) == ("N2", "N2R2") and all(i not in BB.FQOFF_ITEMS and i not in BB.ITEM_STAGE
+                                                             for i in ("C1", "N1", "D", "N5")))
+    unknown = work / "x-z9"
+    (unknown / "r").mkdir(parents=True)
+    (unknown / G.RUN_META_NAME).write_text(json.dumps({"exploratory": True, "item": "Z9"}), encoding="utf-8")
+    c.refuses("block B refuses an unknown item (run_item)", "run_item", BB.run_item, unknown / "r")
+    tel_sha = _sha(out_s / Q.TELEMETRY_NAME) if (out_s / Q.TELEMETRY_NAME).is_file() else "0" * 64
+    bbo = work / "bbo" / "x-n2r2-bb"
+    code, text = blockb_main(["--run-dir", str(out_s), "--expect-telemetry-sha256", tel_sha, "--host-label", "smoke",
+                              "--run-id", "x-n2r2-bb", "--out-dir", str(bbo)])
+    xb = json.loads((bbo / "x_blockb.json").read_text(encoding="utf-8")) if (bbo / "x_blockb.json").is_file() else {}
+    steps = xb.get("steps") or [{}]
+    c.check("block B on an N2R2 run runs the pinned scripts: convert, from the checkout, refuses the 2-epoch smoke run "
+            "as incomplete (exit 3); card fix 1 recorded as not run; INCOMPLETE",
+            code == 3 and xb.get("item") == "N2R2" and steps[0].get("step") == "convert"
+            and steps[0].get("argv", ["", "", ""])[1:3] == ["scripts/qat_epoch_eval.py", "convert"]
+            and steps[0].get("exit") == 3
+            and (xb.get("selected_epoch_fake_quant_off_val") or {}).get("status") == "not run"
+            and "RESULT: X BLOCK B INCOMPLETE" in text, text.strip()[-200:])
+    bad_item = work / "bbs" / "x-n2r2-s42"
+    (bad_item / "r").mkdir(parents=True)
+    (bad_item / G.RUN_META_NAME).write_text(json.dumps({"exploratory": True, "item": "N2R2", "seeds": [42]}))
+    (bad_item / "r" / Q.TELEMETRY_NAME).write_text(json.dumps({"event": "run_meta", "stage": "E6-E2"}) + "\n")
+    bso = work / "bbs_o" / "x-n2r2-bbs"
+    code2, text2 = blockb_main(["--run-dir", str(bad_item / "r"), "--expect-telemetry-sha256",
+                                _sha(bad_item / "r" / Q.TELEMETRY_NAME), "--host-label", "smoke", "--run-id",
+                                "x-n2r2-bbs", "--out-dir", str(bso)])
+    c.check("block B refuses an N2R2 item whose run is stage E6-E2 (run_stage), before any step",
+            code2 == 2 and "[run_stage]" in text2 and not (bso / "eval").exists()
+            and not (bso / "x_blockb.json").exists(), text2.strip()[-200:])
+    # card fix 1's step: the command, judged by the pinned evaluator's own CLI checks
+    ev_r2 = work / "fq_eval_r2"
+    ev_r2.mkdir()
+    if (out_s / Q.TELEMETRY_NAME).is_file():
+        (ev_r2 / "qat_selection.json").write_text(json.dumps(selection_doc(out_s, 2)), encoding="utf-8")
+    r_r2 = c.passes("card fix 1 runs for N2R2's selection", BB.selected_epoch_vals, out_s, ev_r2, work / "fq_o_r2",
+                    SimpleNamespace(expect_telemetry_sha256=tel_sha), stage="E6-R2", runner=evaluator_cli_runner) or {}
+    fo = r_r2.get("fake_quant_off") or {}
+    c.check("card fix 1 for N2R2: the pinned evaluator's CLI refuses stage E6-R2 (FQOFF_STAGES); recorded, not ok",
+            r_r2.get("ok") is False and fo.get("status") == "failed" and fo.get("exit") == 1
+            and "fqoff_stage_not_in_scope" in str(fo.get("message")), str(fo.get("message"))[:160])
+    ends_s = {r["epoch"]: r for r in rows_of(out_s) if r.get("event") == "epoch_end"} if out_s.is_dir() else {}
+    c.check("... the selected epoch's fake-quant VAL is recorded from its epoch_end row all the same",
+            (r_r2.get("fake_quant") or {}).get("epoch") == 2 and 2 in ends_s
+            and r_r2["fake_quant"]["value"] == ends_s[2]["fake_quant_val_all_class_miou"]
+            and r_r2["fake_quant"]["telemetry"]["sha256"] == tel_sha)
+    from src.eval.artifacts import FQOFF_STAGES
+    c.check("the evaluator's stage list, read here: E6-E2 in FQOFF_STAGES, E6-R2 not",
+            "E6-E2" in FQOFF_STAGES and "E6-R2" not in FQOFF_STAGES, str(FQOFF_STAGES))
+    out_n2 = work / "n2run" / "x-n2-s42"
+    smoke_qat(random_student(seed=4), out_n2, stage="e6-e2", steps=2)
+    ev_n2 = work / "fq_eval_n2"
+    ev_n2.mkdir()
+    (ev_n2 / "qat_selection.json").write_text(json.dumps(selection_doc(out_n2, 2)), encoding="utf-8")
+    tel_n2 = _sha(out_n2 / Q.TELEMETRY_NAME)
+    r_n2 = c.passes("card fix 1 runs for N2's selection", BB.selected_epoch_vals, out_n2, ev_n2, work / "fq_o_n2",
+                    SimpleNamespace(expect_telemetry_sha256=tel_n2), stage="E6-E2", runner=evaluator_cli_runner) or {}
+    fo2 = r_n2.get("fake_quant_off") or {}
+    c.check("card fix 1 for N2: the command passes the pinned evaluator's CLI checks; the value recorded with its "
+            "summary's sha256, the checks all true",
+            r_n2.get("ok") is True and fo2.get("status") == "scored" and fo2.get("value") == 0.3412
+            and len((fo2.get("summary") or {}).get("sha256", "")) == 64
+            and all((fo2.get("checks") or {"x": 0}).values()))
+    a = evaluator().build_parser().parse_args(fo2.get("argv", ["-B", "x"])[2:]) if fo2.get("argv") else None
+    sel_sha = _sha(ev_n2 / "qat_selection.json")
+    win = selection_doc(out_n2, 2)["winner"]
+    c.check("... the command: qat_fakequant_off on VAL, canvas, CPU, batch 1, provisional; the winner's eNN.pt and "
+            "sha256; the selection's and the telemetry's sha256",
+            a is not None and a.precision == "qat_fakequant_off" and a.split == "val" and a.protocol == "canvas"
+            and a.device == "cpu" and a.batch_size == 1 and a.artifact_status == "provisional" and a.stage == "E6-E2"
+            and Path(a.checkpoint) == out_n2 / "epoch_ckpts" / "e02.pt" and a.qat_run_dir == str(out_n2)
+            and a.expected_checkpoint_sha256 == win["checkpoint_sha256"] and a.qat_selection_sha256 == sel_sha
+            and a.qat_telemetry_sha256 == tel_n2 and a.max_samples is None and a.threads is None)
+
+    def wrong_stage_runner(argv, **kw):
+        p = evaluator_cli_runner(argv, **kw)
+        out = Path(argv[argv.index("--out-dir") + 1]) / "summary.json"
+        s = json.loads(out.read_text(encoding="utf-8"))
+        s["run"]["stage"] = "E6"
+        out.write_text(json.dumps(s), encoding="utf-8")
+        return p
+    r_w = BB.selected_epoch_vals(out_n2, ev_n2, work / "fq_o_n2w", SimpleNamespace(expect_telemetry_sha256=tel_n2),
+                                 stage="E6-E2", runner=wrong_stage_runner)
+    c.check("card fix 1: a summary of another stage is refused, not recorded as the value",
+            r_w["ok"] is False and r_w["fake_quant_off"]["status"] == "refused"
+            and r_w["fake_quant_off"]["checks"]["stage"] is False)
+    r_f = BB.selected_epoch_vals(out_n2, ev_n2, work / "fq_o_n2f", SimpleNamespace(expect_telemetry_sha256=tel_n2),
+                                 stage="E6-E2", runner=lambda argv, **kw: Done(argv, 1, "", "CliError: boom"))
+    c.check("card fix 1: an evaluator that exits 1 is recorded as failed with its message",
+            r_f["ok"] is False and r_f["fake_quant_off"]["status"] == "failed"
+            and r_f["fake_quant_off"]["message"] == "CliError: boom")
+    if full:
+        r_s = BB.selected_epoch_vals(out_s, ev_r2, work / "fq_o_r2s", SimpleNamespace(expect_telemetry_sha256=tel_sha),
+                                     stage="E6-R2", runner=script_runner)
+        fs = r_s["fake_quant_off"]
+        c.check("card fix 1 for N2R2 through the script itself: scripts/evaluate_model.py exits 1, "
+                "[fqoff_stage_not_in_scope], writes nothing",
+                fs.get("exit") == 1 and "fqoff_stage_not_in_scope" in str(fs.get("message"))
+                and not Path(fs["out_dir"]).exists(), str(fs.get("message"))[:200])
+        r_s2 = BB.selected_epoch_vals(out_n2, ev_n2, work / "fq_o_n2s", SimpleNamespace(expect_telemetry_sha256=tel_n2),
+                                      stage="E6-E2", runner=script_runner)
+        fs2 = r_s2["fake_quant_off"]
+        c.check("card fix 1 for N2 through the script itself: past its CLI checks (no CliError), refused at the source "
+                "binding on the synthetic selection (StageArtifactError on qat_selection.json; no dataset read)",
+                fs2.get("exit") == 1 and str(fs2.get("message")).startswith("StageArtifactError:")
+                and "qat_selection.json" in str(fs2.get("message")) and "CliError" not in str(fs2.get("message"))
+                and not Path(fs2["out_dir"]).exists(), str(fs2.get("message"))[:200])
+    # ---- 7. the CLI (subprocesses: no data root, no CUDA in this session)
+    if full:
+        py, cli = sys.executable, str(REPO / "scripts" / "explore_qat.py")
+        env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
+        (work / "clipdir").mkdir()
+        common = ["--real-run", "--confirm-real-run", "--source-run-dir", str(d), "--expect-source-sha256",
+                  REC["sha256"], "--clip-selection", str(work / "clipdir" / "clip.json"), "--clip-selection-sha256",
+                  X.CLIP_SELECTION_SHA256, "--num-workers", "1", "--expect-head", "0" * 40, "--sites-on-input-grid",
+                  "--tf32-off"]
+        recs = ["--records-root", str(fx["root"]), "--records-commit", fx["head"], "--lambda-selection", LAM_REL,
+                "--lambda-selection-sha256", LAM_SHA, "--alpha-selection", ALPHA_REL, "--alpha-selection-sha256",
+                ALPHA_SHA, "--condition-record", COND_REL, "--condition-sha256", fx["rec_sha"], "--condition-dl",
+                DL_ID]
+
+        def cli_run(*extra):
+            return subprocess.run([py, "-I", cli, *common, *extra], capture_output=True, text=True, env=env)
+        r = cli_run("--item", "N2R2", "--seed", "43", *recs, "--run-id", "x-n2r2-s42",
+                    "--out-dir", str(work / "o1" / "x-n2r2-s42"))
+        c.check("CLI: N2R2 at seed 43 refused (exit 2)", r.returncode == 2 and "item_seed" in r.stdout)
+        r = cli_run("--item", "N2R2", "--seed", "42", "--run-id", "x-n2r2-s42",
+                    "--out-dir", str(work / "o2" / "x-n2r2-s42"))
+        c.check("CLI: N2R2 without its records flags refused (exit 2), each named",
+                r.returncode == 2 and "n2r2_records_required" in r.stdout and "--condition-record" in r.stdout
+                and "--records-root" in r.stdout, r.stdout.strip()[-160:])
+        r = cli_run("--item", "N2R2", "--seed", "42", *recs[:-2], "--run-id", "x-n2r2-s42",
+                    "--out-dir", str(work / "o3" / "x-n2r2-s42"))
+        c.check("CLI: N2R2 without --condition-dl refused (exit 2)", r.returncode == 2
+                and "n2r2_records_required" in r.stdout and "--condition-dl" in r.stdout)
+        r = cli_run("--item", "N2R2", "--seed", "42", *recs, "--run-id", "x-n2r2-s42-b",
+                    "--out-dir", str(work / "o4" / "x-n2r2-s42-b"))
+        c.check("CLI: N2R2 under another run ID refused (exit 2)", r.returncode == 2 and "n2r2_run_id" in r.stdout)
+        r = cli_run("--item", "N2", "--seed", "42", "--records-root", str(fx["root"]), "--run-id", "x-n2-s42",
+                    "--out-dir", str(work / "o5" / "x-n2-s42"))
+        c.check("CLI: N2 given N2R2's records flag refused (exit 2)",
+                r.returncode == 2 and "n2r2_flags_not_applicable" in r.stdout)
+        r = cli_run("--item", "N2R2", "--seed", "42", *recs, "--split", "TE" + "ST", "--run-id", "x-n2r2-s42",
+                    "--out-dir", str(work / "o6" / "x-n2r2-s42"))
+        c.check("CLI: N2R2 with --split TEST refused first, by name (exit 2)",
+                r.returncode == 2 and "split_test" in r.stdout)
+        r = cli_run("--item", "N2R2", "--seed", "42", *recs, "--run-id", "x-n2r2-s42",
+                    "--out-dir", str(work / "o7" / "x-n2r2-s42"))
+        c.check("CLI: N2R2 with a data root that fails the TRAIN/VAL-only check refused, nothing written",
+                r.returncode == 2 and "data_root_not_trainval_only" in r.stdout
+                and not (work / "o7" / "x-n2r2-s42").exists(), r.stdout.strip()[-160:])
+        bbs = str(REPO / "scripts" / "explore_qat_blockb.py")
+        r = subprocess.run([py, "-I", bbs, "--run-dir", str(out_s), "--expect-telemetry-sha256", tel_sha,
+                            "--host-label", "smoke", "--run-id", "x-n2r2-bbc", "--out-dir",
+                            str(work / "bbc" / "x-n2r2-bbc")], capture_output=True, text=True, env=env)
+        c.check("block B CLI: an N2R2 run passes run_item and reaches the guard's data-root check (exit 2 there)",
+                r.returncode == 2 and "data_root_not_trainval_only" in r.stdout and "run_item" not in r.stdout)
+        r = subprocess.run([py, "-I", bbs, "--run-dir", str(unknown / "r"), "--expect-telemetry-sha256", "0" * 64,
+                            "--host-label", "smoke", "--run-id", "x-z9-bb", "--out-dir", str(work / "bbz" / "x-z9-bb")],
+                           capture_output=True, text=True, env=env)
+        c.check("block B CLI: an unknown item refused (exit 2, run_item)", r.returncode == 2 and "run_item" in r.stdout)
+    return c
+
+
+def main() -> int:
+    print("[smoke_explore_n2r2] cases")
+    c = run_cases(verbose=True, full=True)
+    print("[smoke_explore_n2r2] mutations")
+    bb = blockb()
+    orig_run = X.run_n2r2
+    for name, mut in {
+        "a permissive identity check": lambda: patched(X, "n2r2_identity_differ", lambda parent, rec: []),
+        "N2R2 on E6-E2's stage": lambda: patched(X, "N2R2_STAGE", "e6-e2"),
+        "E6-R2's binding skipped": lambda: patched(Q, "am23_parent_binding", lambda *a, **k: {"stage": "E3"}),
+        "run_n2r2 without the sites": lambda: patched(X, "run_n2r2", lambda **kw: orig_run(**{**kw, "sites": False})),
+        "block B without card fix 1": lambda: patched(bb, "FQOFF_ITEMS", ()),
+    }.items():
+        killed, failed = mutation_killed(lambda: run_cases(False, full=False), mut)
+        c.check(f"mutation KILLED: {name}", killed, f"e.g. {failed[:2]}")
+    return c.result()
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/src/explore/guard.py b/src/explore/guard.py
index 5a1b132..578ceb8 100644
--- a/src/explore/guard.py
+++ b/src/explore/guard.py
@@ -56,13 +56,16 @@ RUN_META_FORMAT = "x_run_meta/1"
 EXPECTED_COUNTS = {"train": 5367, "val": 846}          # configs/data.py SPLIT_SIZES (TRAIN, VAL)
 VAL_IMAGES = 846
 
-# The items of AM-24 this lane builds (audit §2; addendum L-AM24-ADD-1). "A1pp" is A1''.
+# The items of AM-24 this lane builds (audit §2; addendum L-AM24-ADD-1), and N2R2 (addendum L-AM24-ADD-2, the item
+# AM-29 registers). "A1pp" is A1''.
 ITEMS = {
     "N4": "free decompositions from existing records (CPU; no dataset, no model)",
     "B1": "D-1 conversion-gap split: the ten sites in turn, all ten, TF32 off, reconciled (CPU or GPU, VAL)",
     "C1": "E6-E2 at seeds 43 and 44 (E6-E2's recipe of record, QAT seed = source seed)",
     "N1": "E6-E2-KD: E6-E2 with logit KD in the QAT loss (lambda 0.5, T 4; no CWD term)",
     "N2": "E6-E2-M: E6-E2 with the ten sites fake-quantized on their input's grid and TF32 off",
+    "N2R2": ("E6-R2-M (AM-29; addendum L-AM24-ADD-2): E6-R2 with N2's change, the ten sites fake-quantized on their "
+             "input's grid and TF32 off, on E3-R2 seed 42's best checkpoint through E6-R2's parent loader"),
     "N5": "E6-E2 with an FP32 LR-ASPP logit head: a MIXED-PRECISION artifact, never called INT8",
     "D": "E6-E2's recipe of record on an exploratory FP32 winner (trigger: best VAL >= 0.3875)",
     "N3": "E2-R2: stage e2 (logit KD, lambda 0.5, T 4) from arm teacher R2 at its M12 iteration 24000",
diff --git a/src/explore/qat_explore.py b/src/explore/qat_explore.py
index 77b0dff..444521d 100644
--- a/src/explore/qat_explore.py
+++ b/src/explore/qat_explore.py
@@ -1,4 +1,5 @@
-"""The exploratory QAT entry for C1, N1, N2, N5 and D (audit EXPLORE-AUDIT-1 §6.2; addendum L-AM24-ADD-1).
+"""The exploratory QAT entry for C1, N1, N2, N2R2, N5 and D (audit EXPLORE-AUDIT-1 §6.2; addenda L-AM24-ADD-1 and
+L-AM24-ADD-2).
 
 Every run is E6-E2's recipe of record (AM-23 item 1 = AM-4a item 1 + clip 5.0, DL-139/DL-146), run by the pinned
 src/quant/qat.py run_qat in mode "real", except the one named change of its item:
@@ -15,6 +16,18 @@ src/quant/qat.py run_qat in mode "real", except the one named change of its item
        built inside run_qat by the name prepare_qat_model of src/quant/qat.py; for this one run that name is bound to
        a wrapper that calls the pinned function and installs the sites (src/explore/patching.py), recorded in
        run_meta (x_substitution). The file is unchanged and its sha256 recorded.
+  N2R2 --sites-on-input-grid --tf32-off (addendum L-AM24-ADD-2; AM-29's E6-R2-M): N2's change exactly, on E6-R2 (AM-23
+       items 3(a) and 4: E6-E2's recipe, only the parent differs; scripts/run_e6_r2.py). The parent is E3-R2 seed 42's
+       best checkpoint as DL-136 records it (E3R2_PARENT_OF_RECORD), seed 42, run ID x-n2r2-s42, checked by sha256 and
+       identity and then through E6-R2's own parent loader: src/quant/qat.py read_tracked_selection (the λ and α
+       selections), resolve_parent (stage E6-R2) and am23_parent_binding (AM-23 item 3(b)'s condition record and its
+       RECORDED decision-log row), and load_source (the strict E3 loader). That loader reads those committed inputs at
+       the HEAD of a checkout. lane/l-am24 holds no other file than fe93a24's, and no condition record exists at
+       fe93a24 (DL-136: none was made) or at master aa8cac2 (its tree, 15 Nov), so N2R2 takes them as required
+       arguments with their sha256, and the
+       checkout that holds them as --records-root at --records-commit: the binding's own repo_root parameter, as the
+       pinned scripts/smoke_am23_quant.py passes it. `run_n2r2` runs the pinned run_qat on stage e6-r2 with N2's
+       substituted prepare_qat_model and TF32 off, exactly as N2's branch of scripts/explore_qat.py does on e6-e2.
   N5   --fp32-head: the LR-ASPP logit head in FP32 (src/explore/fp32_head.py): a MIXED-PRECISION model, never INT8.
   D    --parent-sha256 only (AM-24 v2.4): an exploratory FP32 winner (an x- run of N3, A1'' or A2) whose best VAL
        all-class mIoU is at least D_TRIGGER = 0.3875 (compared as Decimal of the value's repr); one D per qualifying
@@ -24,11 +37,16 @@ src/quant/qat.py run_qat in mode "real", except the one named change of its item
 
 The exploratory parent check accepts, for C1, N1, N2 and N5, exactly E2 s42 / s43 / s44's best checkpoints of record
 by their recorded sha256 (E2_PARENTS_OF_RECORD), through the pinned resolve_parent (best.json, run_meta, a complete
-run, the seed); D's parent through resolve_parent with --parent-sha256, which must match the file.
+run, the seed); D's parent through resolve_parent with --parent-sha256, which must match the file. N2R2's parent check
+(`n2r2_parent`) accepts E3-R2 seed 42's checkpoint only, so N2's E2 parent is refused there, and N2's (the E2 table
+above) refuses E3-R2's.
 """
 from __future__ import annotations
 
 import math
+import os
+import re
+import subprocess
 import time
 from decimal import Decimal
 from pathlib import Path
@@ -54,10 +72,32 @@ CLIP_SELECTION_SHA256 = "c433145a6cfe0e133ebee8fcb70a76f05b60fc6462670abee34c530
 D_TRIGGER = Decimal("0.3875")               # audit §2 D / §6.3: exploratory FP32 best VAL >= 0.3875
 D_PARENT_ITEMS = ("N3", "A1pp", "A2")
 N1_LAMBDA = 0.5
-ITEMS = ("C1", "N1", "N2", "N5", "D")
-ITEM_SEEDS = {"C1": (43, 44), "N1": (42,), "N2": (42,), "N5": (42,)}
+ITEMS = ("C1", "N1", "N2", "N2R2", "N5", "D")
+ITEM_SEEDS = {"C1": (43, 44), "N1": (42,), "N2": (42,), "N2R2": (42,), "N5": (42,)}
 ITEM_FLAGS = {"C1": frozenset(), "N1": frozenset({"kd_logit"}), "N2": frozenset({"sites_on_input_grid", "tf32_off"}),
               "N5": frozenset({"fp32_head"}), "D": frozenset()}
+ITEM_FLAGS["N2R2"] = ITEM_FLAGS["N2"]       # N2R2: exactly N2's flag set (the same object)
+
+# N2R2 (addendum L-AM24-ADD-2; AM-29's E6-R2-M). E3-R2 seed 42's best checkpoint and its records as DL-136 records them
+# (AM-23 item 3(b)'s entry: E3-R2 complete, best VAL 0.3842073380947113 > 0.36337557435035706, "yes"; DL-138: E6-R2
+# cut, never launched). The values were MEASURED there on the laptop's copy of the run's keep.
+E3R2_PARENT_OF_RECORD = {
+    "seed": 42, "arm": "E3-R2", "run_id": "e3-r2_s42_lambda0.5_alpha25_80k_a2",
+    "checkpoint": "e3_student_best_iter80000.pt",
+    "sha256": "da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4",
+    "best_json_sha256": "9cb73b2bab64f1c9941295de87c1541ec198dfa3b85298e5b37b7b38704e0f52",
+    "best_val": 0.3842073380947113,
+    "run_meta_sha256": "f68e4ec0889cbca427a9a559293c8d3f3147f5d017a3ad0a79967f5c871c5b39",
+    "telemetry_sha256": "8c8465b4001cd089a39c21e9f3731644210f814b0fcf7adc39e47411edfe9c24",
+    "records": "DL-136 (AM-23 item 3(b)'s condition entry); DL-138 (E6-R2 cut, never launched)"}
+N2R2_STAGE = "e6-r2"                        # E6-R2's stage key (src/quant/stages.py; scripts/run_e6_r2.py STAGE)
+N2R2_RUN_ID = "x-n2r2-s42"
+# The committed inputs E6-R2's parent loader reads (src/quant/qat.py build_parser's E6-R2 flags), and the checkout that
+# holds them at its HEAD: each a required argument of N2R2, with no default (the sha256 flags pin the files).
+N2R2_RECORD_FLAGS = ("records_root", "records_commit", "lambda_selection", "lambda_selection_sha256", "alpha_selection",
+                     "alpha_selection_sha256", "condition_record", "condition_sha256", "condition_dl")
+N2R2_MATCHED_RUN = ("E6-R2 seed 42 (AM-28; scripts/run_e6_r2.py): the same parent, recipe, parent loader and clip; "
+                    "N2's substitution (the ten sites on their input's grid, TF32 off) is the only change")
 COMPARATORS = {"E6-E2 s42": 0.35132896900177, "E6 s42/43/44": [0.3308, 0.3490, 0.3197],
                "E5 s42/43/44": [0.3343, 0.3192, 0.3328], "E6-KD s42": 0.3321,
                "source": "audit §6.2 (values of record: DL-131, DL-139, DL-142 to DL-146)"}
@@ -65,9 +105,19 @@ RECIPE_OF_RECORD = ("E6-E2 (AM-23 item 1 = AM-4a item 1 + clip 5.0): SGD lr 3e-4
                     "T_max 5,025, batch 16 drop_last, 15 epochs, clip 5.0, fake-quant from step 0, BN frozen top of "
                     "epoch 11, observers off top of epoch 13, the VAL bracket, the QNNPACK qconfig of record, TF32 at "
                     "torch defaults; block B: 15 converted epochs scored on CPU, AM-23's selection rules, finalize")
+RECIPE_OF_RECORD_N2R2 = ("E6-R2 (AM-23 items 3(a) and 4; scripts/run_e6_r2.py): E6-E2's recipe of record (AM-23 "
+                         "item 1 = AM-4a item 1 + clip 5.0) on E3-R2 seed 42's best checkpoint, read through E6-R2's "
+                         "parent loader (src/quant/qat.py read_tracked_selection, resolve_parent, "
+                         "am23_parent_binding, load_source): SGD lr 3e-4, momentum 0.9, wd 1e-4, cosine T_max 5,025, "
+                         "batch 16 drop_last, 15 epochs, clip 5.0, fake-quant from step 0, BN frozen top of epoch 11, "
+                         "observers off top of epoch 13, the VAL bracket, the QNNPACK qconfig of record, TF32 at "
+                         "torch defaults; block B: 15 converted epochs scored on CPU, AM-23's selection rules, "
+                         "finalize")
 CHANGES = {"C1": "none (seeds 43/44, which AM-23 item 1 excludes)",
            "N1": "logit KD in the QAT loss: lambda 0.5, T 4, E6-KD's ramp and teacher stream; no CWD term",
            "N2": "the ten sites fake-quantized on their input's grid, and TF32 off (cuDNN and matmul)",
+           "N2R2": ("N2's change on E6-R2: the ten sites fake-quantized on their input's grid, and TF32 off (cuDNN and "
+                    "matmul); the parent E3-R2 seed 42 (DL-136) through E6-R2's parent loader"),
            "N5": "the LR-ASPP logit head in FP32 (mixed precision; never INT8)",
            "D": "none: the parent is an exploratory FP32 winner (best VAL >= 0.3875)"}
 
@@ -145,6 +195,132 @@ def exploratory_parent(item: str, run_dir, *, seed: int, expect_source_sha256: s
     return stage, {**parent, "of_record": rec}
 
 
+# ------------------------------------------------------------------ N2R2 (addendum L-AM24-ADD-2; AM-29's E6-R2-M)
+_HEX40 = re.compile(r"[0-9a-f]{40}")
+
+
+def records_root_record(root, commit) -> dict:
+    """The checkout whose HEAD holds the committed inputs E6-R2's parent loader reads (the λ and α selections, AM-23
+    item 3(b)'s condition record and docs/DECISION_LOG.md). Refused unless `root` is a git checkout's top level,
+    holding no "test" (SL-1) and checked before git runs on it, whose HEAD is `commit`. git runs as
+    src/quant/qat.py _git_head_bytes runs it, so it names the HEAD the pinned readers then read."""
+    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
+        raise ExploreQATError("records_commit_format", "--records-commit must be the 40-hex HEAD of --records-root")
+    if not root or G.names_test(root) or G.names_test(os.path.realpath(str(root))):
+        raise ExploreQATError("records_root_path", f"--records-root {root!r} is not given or holds 'test' (SL-1)")
+    p = Path(root).resolve()
+    if not p.is_dir():
+        raise ExploreQATError("records_root_missing", f"--records-root {root!r} is not a directory")
+    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
+
+    def _git(*argv):
+        try:
+            r = subprocess.run(["git", "-C", str(p), *argv], capture_output=True, text=True, timeout=60, env=env)
+        except (OSError, subprocess.SubprocessError):
+            return None
+        return r.stdout.strip() if r.returncode == 0 else None
+    top, head = _git("rev-parse", "--show-toplevel"), _git("rev-parse", "HEAD")
+    if top is None or Path(top).resolve() != p:
+        raise ExploreQATError("records_root_not_checkout", f"--records-root {p} is not the top level of a git checkout "
+                                                           f"(git gives {top!r})")
+    if head != commit:
+        raise ExploreQATError("records_commit_mismatch", f"the records checkout's HEAD is {head!r}, --records-commit "
+                                                         f"{commit}")
+    return {"path": str(p), "commit": head,
+            "reads": ("the λ and α selections, the condition record and docs/DECISION_LOG.md at this HEAD, through "
+                      "src/quant/qat.py read_tracked_selection and am23_parent_binding (repo_root)")}
+
+
+def n2r2_identity_differ(parent: dict, rec: dict) -> list[str]:
+    """The fields of resolve_parent's record that differ from E3-R2 seed 42 as DL-136 records it (`rec`)."""
+    want = {"run_id": rec["run_id"], "checkpoint": rec["checkpoint"], "checkpoint_sha256": rec["sha256"],
+            "best_json_sha256": rec["best_json_sha256"], "run_meta_sha256": rec["run_meta_sha256"],
+            "telemetry_sha256": rec["telemetry_sha256"], "seed": rec["seed"], "stage": "E3"}
+    differ = [k for k, v in want.items() if parent.get(k) != v]
+    if repr(parent.get("best_val_miou_all_class")) != repr(rec["best_val"]):
+        differ.append("best_val_miou_all_class")
+    return differ
+
+
+def n2r2_parent(run_dir, *, seed: int, expect_source_sha256: str | None, parent_sha256: str | None = None,
+                records_root, records_commit, lambda_selection, lambda_selection_sha256, alpha_selection,
+                alpha_selection_sha256, condition_record, condition_sha256, condition_dl) -> dict:
+    """N2R2's parent check: E3-R2 seed 42's best checkpoint by sha256 and by its identity as DL-136 records it (run ID,
+    checkpoint file, best.json, run_meta, telemetry, best value), then E6-R2's own parent loader unchanged, in the order
+    src/quant/qat.py real_run_gates runs it: the λ and α selections (read_tracked_selection), the parent
+    (resolve_parent on stage E6-R2) and AM-23 item 4's binding (am23_parent_binding: arm E3-R2, descriptive, never a
+    parent of E4-E7, the 80,000-iteration horizon, the selections' λ and α, and item 3(b)'s condition record with its
+    RECORDED decision-log row), each committed input read at --records-root's HEAD. Returns {"stage", "parent",
+    "am23_binding", "records_root"}; raises ExploreQATError or the pinned QATRefused (the condition reader's refusals
+    are QATRefused too)."""
+    from types import SimpleNamespace
+
+    from src.quant import qat as Q
+    from src.quant.stages import resolve_quant_stage
+    rec = E3R2_PARENT_OF_RECORD
+    if parent_sha256 is not None:
+        raise ExploreQATError("parent_sha256_not_d", "--parent-sha256 is D's flag")
+    if type(seed) is not int or seed != rec["seed"]:
+        raise ExploreQATError("item_seed", f"N2R2 runs seed {rec['seed']} only (E6-R2, AM-23 items 3 and 4), not "
+                                           f"{seed!r}")
+    if expect_source_sha256 != rec["sha256"]:
+        raise ExploreQATError("parent_not_of_record", f"--expect-source-sha256 {expect_source_sha256!r} is not E3-R2 "
+                                                      f"seed 42's best checkpoint ({rec['sha256']}, DL-136)")
+    root = records_root_record(records_root, records_commit)
+    root_p = Path(root["path"])
+    stage = resolve_quant_stage(N2R2_STAGE)
+    lam = Q.read_tracked_selection(lambda_selection, lambda_selection_sha256, Q.LAMBDA_SELECTION_FORMAT, "lambda",
+                                   root_p)
+    alpha = Q.read_tracked_selection(alpha_selection, alpha_selection_sha256, Q.ALPHA_SELECTION_FORMAT, "alpha", root_p)
+    parent = Q.resolve_parent(stage, run_dir, expect_source_sha256, seed)
+    differ = n2r2_identity_differ(parent, rec)
+    if differ:
+        raise ExploreQATError("parent_identity", f"the parent is not E3-R2 seed 42 as DL-136 records it: {differ} "
+                                                 f"differ ({parent.get('run_id')}/{parent.get('checkpoint')})")
+    flags = SimpleNamespace(seed=seed, condition_record=condition_record, condition_sha256=condition_sha256,
+                            condition_dl=condition_dl)
+    binding = Q.am23_parent_binding(stage, parent, flags, (lam, alpha), repo_root=root_p)
+    return {"stage": stage, "parent": {**parent, "of_record": rec}, "am23_binding": binding, "records_root": root}
+
+
+def run_n2r2(*, mode: str, model: nn.Module, source_meta: dict, out_dir, seed: int, clip_norm: float,
+             clip_source: str, device: str, num_workers: int, parent: dict | None, extra_meta: dict | None = None,
+             sites: bool = True, loaders=None, max_steps: int | None = None, max_val_batches: int | None = None,
+             log=print):
+    """N2R2's run: the pinned src/quant/qat.py run_qat on stage e6-r2 with N2's change, as scripts/explore_qat.py's N2
+    branch makes it on e6-e2: TF32 off (both flags, set_tf32), then prepare_qat_model bound for this call to a wrapper
+    that calls the pinned function and installs the ten site fake-quants (SiteMatcher), the binding recorded in
+    run_meta (x_substitution) and restored after. Returns (summary, substitution record, matcher). `sites` False runs
+    the same call with neither TF32 nor the binding touched, i.e. E6-R2's own path; it is refused in a real run (one
+    named change per run). `mode` "smoke" passes injected loaders, a step cap and a VAL cap to run_qat, which refuses
+    them in a real run."""
+    from src.quant import qat as Q
+    from .patching import substituted
+    from .sites import SITE_NAMES, SiteMatcher, set_tf32
+    if mode == "real" and sites is not True:
+        raise ExploreQATError("n2r2_change_required", "a real N2R2 run carries N2's change: the ten sites and TF32 off")
+    kw = dict(stage=N2R2_STAGE, mode=mode, model=model, source_meta=source_meta, out_dir=out_dir, seed=seed,
+              clip_norm=clip_norm, clip_source=clip_source, device=device, num_workers=num_workers, parent=parent,
+              loaders=loaders, max_steps=max_steps, max_val_batches=max_val_batches, log=log)
+    extra = dict(extra_meta or {})
+    if sites is not True:
+        return Q.run_qat(extra_meta=extra, **kw), None, None
+    holder = {}
+    pinned_prepare = Q.prepare_qat_model
+
+    def prepare_with_sites(m, *, select_backend=True):
+        prepared = pinned_prepare(m, select_backend=select_backend)
+        holder["matcher"] = SiteMatcher(prepared, SITE_NAMES)
+        return prepared
+
+    extra["x_tf32_set"] = set_tf32(False)
+    with substituted(Q, "prepare_qat_model", prepare_with_sites, item="N2R2",
+                     why="install the ten input-grid site fake-quants on the prepared student") as rec:
+        extra["x_substitution"] = rec
+        summary = Q.run_qat(extra_meta=extra, **kw)
+    return summary, rec, holder.get("matcher")
+
+
 # ------------------------------------------------------------------ N1: run_qat's loop with logit KD
 def run_qat_logit_kd(*, mode: str, model: nn.Module, teacher, source_meta: dict, out_dir, seed: int,
                      clip_norm: float, clip_source: str, device: str, num_workers: int, lambda_logit: float = N1_LAMBDA,
```

</details>

### A5. Smokes

Every smoke ran with the venv: `python -B -I`, and `L_AM24_SMOKE_DIR` in the scratchpad. The runner's RESULT count
of each smoke includes its mutation rows. One torch warning is printed to stderr by the pinned QAT path's fake-quants
that a smoke forward never observes: "torch/ao/quantization/utils.py:317: UserWarning: must run observer before
calling calculate_qparams. Returning default values." It is not a failure (MEASURED).

**Before**, at 60dc294, `python -B -I scripts/smoke_explore_all.py` (MEASURED):

```text
HEAD 60dc29497accf2ff0ccd9a38857b9e53435a4dba
start 2026-11-15T11:33:43Z
smoke_explore_guard        RESULT: SMOKE_EXPLORE_GUARD PASS (51/51)         mutations killed  5     1.9s
smoke_explore_decompose    RESULT: SMOKE_EXPLORE_DECOMPOSE PASS (30/30)     mutations killed  1     3.4s
smoke_explore_sites        RESULT: SMOKE_EXPLORE_SITES PASS (44/44)         mutations killed  4    48.4s
smoke_explore_qat          RESULT: SMOKE_EXPLORE_QAT PASS (59/59)           mutations killed  4    51.9s
smoke_explore_kd           RESULT: SMOKE_EXPLORE_KD PASS (40/40)            mutations killed  3    59.0s
smoke_explore_kl3          RESULT: SMOKE_EXPLORE_KL3 PASS (22/22)           mutations killed  3    34.4s
RESULT: L-AM24 SMOKES PASS (6/6)
exit 0
end 2026-11-15T11:37:02Z
```

**After**, at a8718d7, the same command (MEASURED):

```text
HEAD 60dc29497accf2ff0ccd9a38857b9e53435a4dba (working tree: this session's edits, uncommitted)
start 2026-11-15T12:19:42Z
$ python -B -I scripts/smoke_explore_all.py
smoke_explore_guard        RESULT: SMOKE_EXPLORE_GUARD PASS (51/51)         mutations killed  5     1.6s
smoke_explore_decompose    RESULT: SMOKE_EXPLORE_DECOMPOSE PASS (30/30)     mutations killed  1     3.0s
smoke_explore_sites        RESULT: SMOKE_EXPLORE_SITES PASS (44/44)         mutations killed  4    48.5s
smoke_explore_qat          RESULT: SMOKE_EXPLORE_QAT PASS (59/59)           mutations killed  4    52.6s
smoke_explore_kd           RESULT: SMOKE_EXPLORE_KD PASS (40/40)            mutations killed  3    55.3s
smoke_explore_kl3          RESULT: SMOKE_EXPLORE_KL3 PASS (22/22)           mutations killed  3    36.3s
smoke_explore_n2r2         RESULT: SMOKE_EXPLORE_N2R2 PASS (88/88)          mutations killed  5    56.9s
RESULT: L-AM24 SMOKES PASS (7/7)
exit 0
end 2026-11-15T12:23:56Z
each done 2026-11-15T12:28:12Z
```

| smoke | rows before | rows after | mutations killed before → after |
|---|---|---|---|
| smoke_explore_guard | 51 | 51 | 5 → 5 |
| smoke_explore_decompose | 30 | 30 | 1 → 1 |
| smoke_explore_sites | 44 | 44 | 4 → 4 |
| smoke_explore_qat | 59 | 59 | 4 → 4 |
| smoke_explore_kd | 40 | 40 | 3 → 3 |
| smoke_explore_kl3 | 22 | 22 | 3 → 3 |
| smoke_explore_n2r2 (new) | — | 88 | — → 5 |
| **total** | **246 (6 smokes)** | **334 (7 smokes)** | **20 → 25** |

**What the new smoke shows** (scripts/smoke_explore_n2r2.py, 83 cases and 5 mutations; the
fixtures are synthetic: an E3-R2 run folder with a full random student, and a records checkout committed by git in the
scratch folder, its condition record made by the pinned writer `build_record` / `write_record` and its row's text by
`dl_text`):
- **Constants.**
  - The parent record is DL-136's, value by value.
  - E6-R2's stage, parent arm, scripts/run_e6_r2.py's STAGE and its AM-23 rules file.
  - The committed λ and α selections read through the pinned reader with the commands' sha256s: λ 0.5, α 25.0,
    and α's best_val 0.36337557435035706, AM-23 item 3(b)'s threshold.
  - The condition record's path of record, absent at this checkout.
- **Flags and seeds.** N2R2 takes N2's two flags (the same object) at seed 42 only, and refuses one flag, none, an
  added flag, N5's flag and seeds 43 and 44. N2's own behaviour is unchanged.
- **Parent check: accepted.** It accepts the fixture's hash and identity through E6-R2's loader. Its parent record
  and binding equal E6-R2's own gates'.
- **Parent check: refused.**
  - Hashes: another hash; N2's E2 parent; with the real table, the fixture's hash.
  - Flags and seed: D's flag; seed 43.
  - The records checkout: a wrong, malformed or missing records commit or folder.
  - The pinned binding:
    - a λ selection of another sha256;
    - a condition record of another sha256, not committed, or changed after its commit;
    - a decision-log row that is absent or not RECORDED;
    - an E3-R1-arm run;
    - this checkout as the records root (`[condition_missing_file]`, MEASURED).
  - DL-136's identity: the a1 attempt's folder, and a best.json, run_meta, telemetry or value other than DL-136's.
  - An E2 folder.
  - N2 refuses E3-R2's parent, by hash and, under a patched table, by E6-E2's loader.
- **Through the pinned run_qat.**
  - All ten sites fire.
  - TF32 off is recorded, as read and as set.
  - The substitution is recorded with qat.py's sha256, and the pinned name is restored, after a refusal too.
  - run_meta is E6-R2's: stage, source stage, projection flag, parent and binding.
  - A real run refuses the change removed and injected loaders.
- **First step.** With the substitution removed, N2R2's first-step loss equals E6-R2's path on one synthetic batch:
  5.981044769287109 both (MEASURED). So do its CE, Dice, gradient norm, lr, batch fingerprint and the state after
  the step, and run_meta's recipe, census, TF32, source and binding fields. With the substitution the loss is
  5.980941295623779, so the comparison discriminates (MEASURED).
- **Launch gate.** N2's launch gate passes on a calibrated copy of the student E6-R2's loader built.
- **Block B.**
  - It accepts an N2R2 run and runs the pinned convert on it from the checkout. Convert refuses the 2-epoch smoke
    run as incomplete; card fix 1 is recorded as not run; INCOMPLETE, exit 3.
  - It refuses an unknown item, and an N2R2 item whose run is E6-E2.
  - Card fix 1's command passes the pinned evaluator's own CLI checks for N2 and is refused for N2R2, in process and
    by the script itself. For N2, the script gets past its CLI checks and refuses the synthetic selection at the
    source binding; no dataset is read.
  - A scored, a failed and a mismatched summary are recorded as such.
- **CLI.**
  - N2R2 refuses: seed 43, a missing records flag (each named), another run ID, TEST, and a data root that fails
    the TRAIN/VAL-only check, writing nothing.
  - N2 refuses N2R2's flag.
  - Block B's CLI takes N2R2 up to the guard, and refuses an unknown item.
- **Mutations, each KILLED.** A permissive identity check; N2R2 on E6-E2's stage; E6-R2's binding skipped; run_n2r2
  without the sites; block B without card fix 1.

**Each smoke alone** at a8718d7 (`python -B -I scripts/<smoke>.py`, full stdout and stderr's distinct lines):

<details><summary>smoke_explore_guard (stdout; stderr: 0 lines, 0 distinct)</summary>

```text
$ python -B -I scripts/smoke_explore_guard.py
[smoke_explore_guard] cases
  [PASS] split 'test' refused as split_test -- raised [split_test] [split_test] --split 'test': TEST is never read by exploratory code (AM-24 clause 2)
  [PASS] split 'TEST' refused as split_test -- raised [split_test] [split_test] --split 'TEST': TEST is never read by exploratory code (AM-24 clause 2)
  [PASS] split 'Test' refused as split_test -- raised [split_test] [split_test] --split 'Test': TEST is never read by exploratory code (AM-24 clause 2)
  [PASS] split 'val_test' refused as split_test -- raised [split_test] [split_test] --split 'val_test': TEST is never read by exploratory code (AM-24 clause 2)
  [PASS] split None refused -- raised [split_test] [split_test] --split None: TEST is never read by exploratory code (AM-24 clause 2)
  [PASS] split 'trainval' refused -- raised [split_not_trainval] [split_not_trainval] --split 'trainval' is not one of ('train', 'val')
  [PASS] split 'TRAIN' (case) refused -- raised [split_not_trainval] [split_not_trainval] --split 'TRAIN' is not one of ('train', 'val')
  [PASS] split 'val' accepted
  [PASS] split 'train' accepted by a train/val tool
  [PASS] split 'train' refused by a VAL-only tool -- raised [split_not_trainval] [split_not_trainval] --split 'train' is not one of ('val',)
  [PASS] run ID without x- refused -- raised [run_id_prefix] [run_id_prefix] run ID 'e6-e2_s43_clip5.0_a1' does not start with 'x-' (AM-24 clause 7)
  [PASS] run ID 'X-' (case) refused -- raised [run_id_prefix] [run_id_prefix] run ID 'X-c1-s43' does not start with 'x-' (AM-24 clause 7)
  [PASS] run ID naming the forbidden word refused -- raised [run_id_test] [run_id_test] run ID 'x-c1-test' holds 'test' (SL-1)
  [PASS] run ID with a slash refused -- raised [run_id_format] [run_id_format] run ID 'x-c1/s43': x- then letters, digits, '.', '_' or '-' only
  [PASS] bare 'x-' refused -- raised [run_id_format] [run_id_format] run ID 'x-': x- then letters, digits, '.', '_' or '-' only
  [PASS] run ID x-c1-s43 accepted
  [PASS] input path naming the forbidden word refused -- raised [input_path_test] [input_path_test] --run-dir '/data/test/run' holds 'test' (SL-1); refused before any read
  [PASS] input link resolving to a forbidden name refused -- raised [input_path_test] [input_path_test] --run-dir '/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb395t13
  [PASS] plain input path accepted
  [PASS] out dir inside the repository refused -- raised [out_dir_in_repo] [out_dir_in_repo] --out-dir /home/user/plantseg-thesis/runs/x-c1-s43 is inside the repository /home/user/plantseg-thesis: run outputs never live there
  [PASS] out dir not named after the run ID refused -- raised [out_dir_not_run_id] [out_dir_not_run_id] --out-dir /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb395t
  [PASS] out dir under an official run folder refused -- raised [out_dir_official] [out_dir_official] --out-dir /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb395t13
  [PASS] out dir under qat_* records folder refused -- raised [out_dir_official] [out_dir_official] --out-dir /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb395t13
  [PASS] out dir under a folder holding best.json refused -- raised [out_dir_official] [out_dir_official] --out-dir /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb395t13
  [PASS] out dir inside an input refused -- raised [out_dir_inside_input] [out_dir_inside_input] --out-dir /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb39
  [PASS] non-empty out dir refused -- raised [out_dir_not_fresh] [out_dir_not_fresh] --out-dir /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb395t1
  [PASS] out dir under an x- folder with records accepted
  [PASS] fresh out dir accepted
  [PASS] missing data root refused -- raised [data_root_not_trainval_only] [data_root_not_trainval_only] [data_root_missing] data root does not resolve: /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scra
  [PASS] TRAIN/VAL count mismatch refused -- raised [data_root_not_trainval_only] [data_root_not_trainval_only] [split_count_mismatch] split train: 2 images / 2 masks, expected 3 of each
  [PASS] fixture TRAIN/VAL-only root accepted
  [PASS] isolation record names the counts
  [PASS] records-only tool skips the data root and says so
  [PASS] guarded_start writes x_run_meta.json
  [PASS] run_meta carries the exploratory fields -- {'exploratory': True, 'amendment': 'AM-24', 'split': 'val', 'item': 'C1', 'artifact_status': 'exploratory', 'run_id': 'x-c1-s43', 'registered': False, 'test_scored': False}
  [PASS] run_meta carries the full recipe
  [PASS] run_meta carries the git commit
  [PASS] run_meta carries the pinned modules' sha256
  [PASS] run_meta carries the lane files' sha256
  [PASS] run_meta claim line
  [PASS] run_meta records the isolation result
  [PASS] second guarded_start into the same dir refused -- raised [out_dir_not_fresh] [out_dir_not_fresh] --out-dir /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_guard_bb395t1
  [PASS] x_run_meta.json is never overwritten
  [PASS] guarded_start refuses split TEST before writing -- raised [split_test] [split_test] --split 'test': TEST is never read by exploratory code (AM-24 clause 2)
  [PASS] nothing written after the TEST refusal
  [PASS] guarded_start refuses an unknown item -- raised [item_unknown] [item_unknown] item 'Z9' is not one of ['A1pp', 'A2', 'B1', 'C1', 'D', 'KL3', 'N1', 'N2', 'N2R2', 'N3', 'N4', 'N5']
[smoke_explore_guard] mutations (each guard made permissive must fail a case)
  [PASS] mutation KILLED: refuse_split -> identity -- 10 case(s) failed, e.g. ["split 'test' refused as split_test", "split 'TEST' refused as split_test"]
  [PASS] mutation KILLED: check_run_id -> identity -- 5 case(s) failed, e.g. ['run ID without x- refused', "run ID 'X-' (case) refused"]
  [PASS] mutation KILLED: check_input_paths -> no SL-1 -- 2 case(s) failed, e.g. ['input path naming the forbidden word refused', 'input link resolving to a forbidden name refused']
  [PASS] mutation KILLED: out_dir_error -> None -- 8 case(s) failed, e.g. ['out dir inside the repository refused', 'out dir not named after the run ID refused']
  [PASS] mutation KILLED: check_data_root -> unchecked -- 4 case(s) failed, e.g. ['missing data root refused', 'TRAIN/VAL count mismatch refused']
RESULT: SMOKE_EXPLORE_GUARD PASS (51/51)
exit 0
```

stderr, distinct lines:

```text
(none)
```

</details>

<details><summary>smoke_explore_decompose (stdout; stderr: 0 lines, 0 distinct)</summary>

```text
$ python -B -I scripts/smoke_explore_decompose.py
[smoke_explore_decompose] cases
  [PASS] terms: fixture E3 run
  [PASS] window is 21..120, 100 rows -- {'last': 100, 'first_iter': 21, 'last_iter': 120, 'rows': 100}
  [PASS] mean(L_logitKD) = 2.0 (hand) -- 2.0
  [PASS] mean(L_feat) = 0.004 (hand) -- 0.004
  [PASS] mean(L_cwd_logit) = 0.25 (hand)
  [PASS] alpha = 0.5 * 2.0 / 0.004 = 250 (hand) -- 250.0
  [PASS] alpha printed with its inputs
  [PASS] weighted contributions: 0.5*2, 25*0.004, 3*0.25
  [PASS] records are hashed
  [PASS] an absent term is said so, never estimated
  [PASS] alpha not formed without L_feat, with the reason
  [PASS] a null value in the window: incomplete, no mean
  [PASS] duplicate iterations (an interleaved relaunch) refused -- raised [window_rows] [window_rows] the window 21..120 holds 101 train rows over 100 distinct iterations; exactly one row per iteration is required (a relaunch interleaved into one f
  [PASS] a run_abort row refused -- raised [run_aborted] [run_aborted] e3_telemetry.jsonl holds a run_abort row: not a completed run
  [PASS] a window longer than the run refused -- raised [window_exceeds_run] [window_exceeds_run] run_end iter 120 is shorter than the window 121
  [PASS] A: parent / fq / converted / fq-off read
  [PASS] A: parent - fq = 1.7491 pp (hand)
  [PASS] A: fq-off - converted = +1.02 pp (DL-176) -- 1.023411750793457
  [PASS] B: missing parent and run dir leave nulls with reasons
  [PASS] B: a summary of another split is not used
  [PASS] table text renders
  [PASS] CLI terms: exit 0 -- RESULT: N4 TERMS DONE -> /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n4_fb5zrwec/explore/x-n4-terms-fixture/n4_terms.json
  [PASS] CLI terms: x_run_meta exploratory, AM-24, item N4, split val, records-only
  [PASS] CLI terms: n4_terms.json alpha 250
  [PASS] CLI qat-table: exit 0 -- RESULT: N4 QAT TABLE DONE (2 models) -> /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n4_fb5zrwec/explore/x-n4-table-fixture/n4_qat_table.json
  [PASS] CLI qat-table: n4_qat_table.json written
  [PASS] CLI: run ID without x- refused (exit 2)
  [PASS] CLI: output inside the repository refused (exit 2)
  [PASS] CLI: --split TEST refused (exit 2)
[smoke_explore_decompose] mutations
  [PASS] mutation KILLED: a finiteness check that accepts null -- e.g. ["crash: TypeError: float() argument must be a string or a real number, not 'NoneType'"]
RESULT: SMOKE_EXPLORE_DECOMPOSE PASS (30/30)
exit 0
```

stderr, distinct lines:

```text
(none)
```

</details>

<details><summary>smoke_explore_sites (stdout; stderr: 2 lines, 2 distinct)</summary>

```text
$ python -B -I scripts/smoke_explore_sites.py
[smoke_explore_sites] cases
  [PASS] ten sites resolve in order, SE pools first
  [PASS] producers: ReLU blocks block.1.0, Hardswish blocks block.1.2, head pool skip_add, resize ff_mul
  [PASS] each pool site's input IS its producer's output (object identity, one forward)
  [PASS] AM-24 v2.4 site census: the code gives the ten sites
  [PASS] census lists the eight SE pools and head.context.0 by module, then the head's interpolate
  [PASS] census: exactly these nine converted modules round onto their input's grid -- ['features.11.block.2.avgpool', 'features.12.block.2.avgpool', 'features.13.block.2.avgpool', 'features.14.block.2.avgpool', 'features.15.block.2.avgpool', 'features.4.block.2.avgpool', 'features.5.block.2.avgpool', 'features.6.block.2.avgpool', 'head.context.0']
  [PASS] census: the head's resize keeps head.ff_mul's grid in the converted model -- fake-quant off-grid fraction 0.0
  [PASS] census: no other converted module keeps its input's grid unobserved
  [PASS] census: src/models/student.py is dc40dd2's (git blob)
  [PASS] census STOPs on an extra average pool in the head -- raised [site_set_mismatch] [site_set_mismatch] the code gives a different set of sites than AM-24 v2.4's: average-pool modules ['features.11.block.2.avgpool', 'features.12.block.2.avgpool
  [PASS] census STOPs on a second interpolate in the head's forward -- raised [site_set_mismatch] [site_set_mismatch] the code gives a different set of sites than AM-24 v2.4's: the head's forward (__main__._student_with_two_interpolates.<locals>.TwoResizeHea
  [PASS] (A) the producer's grid is the converted input's grid, at all ten sites -- {'se4': True, 'se5': True, 'se6': True, 'se11': True, 'se12': True, 'se13': True, 'se14': True, 'se15': True, 'head_pool': True, 'head_resize': True}
  [PASS] (A) the converted output carries its input's grid, at all ten sites
  [PASS] (A) site-matched reproduces the converted op: >= 99.9% exact, 100% within 1 LSB, all ten -- {'se4': 1.0, 'se5': 1.0, 'se6': 1.0, 'se11': 1.0, 'se12': 1.0, 'se13': 1.0, 'se14': 1.0, 'se15': 1.0, 'head_pool': 1.0, 'head_resize': 1.0}
  [PASS] (A) in the random student, where the registered graph is off the converted op, the matched one is on -- registered exact fractions {'se4': 0.3056, 'se5': 1.0, 'se6': 0.9667, 'se11': 1.0, 'se12': 1.0, 'se13': 1.0, 'se14': 1.0, 'se15': 1.0, 'head_pool': 1.0, 'head_resize': 1.0}
  [PASS] (A') op level, random quantized inputs on each site's own grid: matched >= 99.9% exact at the nine pools, >= 98% at the resize, 100% within 1 LSB at all ten -- {'se4': (1.0, 1.0), 'se5': (1.0, 1.0), 'se6': (1.0, 1.0), 'se11': (1.0, 1.0), 'se12': (1.0, 1.0), 'se13': (1.0, 1.0), 'se14': (1.0, 1.0), 'se15': (1.0, 1.0), 'head_pool': (1.0, 1.0), 'head_resize': (0.9885, 1.0)}
  [PASS] (A') op level: the quantized operations emit on their input's grid
  [PASS] (A') op level: the registered graph is exact in under half the elements at every site -- {'se4': 0.0, 'se5': 0.0, 'se6': 0.0, 'se11': 0.0, 'se12': 0.0, 'se13': 0.0, 'se14': 0.0, 'se15': 0.0, 'head_pool': 0.0, 'head_resize': 0.0922}
  [PASS] (B) all-ten-matched mean |diff| <= 0.6 x registered, every image -- [(0.024, 0.0599), (0.0192, 0.0562), (0.0185, 0.0519)]
  [PASS] (B) matched argmax agreement >= registered, every image -- [(0.9781, 0.9401), (0.9608, 0.9101), (0.9791, 0.9406)]
  [PASS] (B) matched max |diff| <= 1 LSB (+1e-4), every image
  [PASS] the site hooks add nothing to the state_dict
  [PASS] remove() restores the registered graph bit for bit
  [PASS] every site hook fired once per forward -- {'se4': 1, 'se5': 1, 'se6': 1, 'se11': 1, 'se12': 1, 'se13': 1, 'se14': 1, 'se15': 1, 'head_pool': 1, 'head_resize': 1}
  [PASS] training through the sites: finite loss and gradients
  [PASS] a site's output lies on its producer's current grid (train mode, observers on)
  [PASS] set_tf32(False) sets both backend flags off
  [PASS] set_tf32(True) sets both on
  [PASS] B1 CPU plan: 13 passes, no TF32 pass
  [PASS] B1 CUDA plan: all 15 passes
  [PASS] B1 passes on synthetic batches: 13 summaries with a value each
  [PASS] B1 site passes record their hook calls
  [PASS] reconcile: gap 0.5 pp, se4 0.2, se5 0.1 (hand)
  [PASS] reconcile: sum of shares 0.3, all ten 0.4, interaction +0.1 (hand)
  [PASS] reconcile: TF32 share 0.02, residual +0.05, explained 0.9 (hand)
  [PASS] reconcile on a CPU: no TF32 share, residual from all10, said so
  [PASS] B1 table lists the SE-pool sites first
  [PASS] B1 CLI refuses --split TEST (exit 2)
  [PASS] B1 CLI refuses a run ID without x- (exit 2)
  [PASS] B1 CLI refuses a data root that fails the TRAIN/VAL-only check (exit 2) -- RESULT: REFUSED [data_root_not_trainval_only] [data_root_missing] data root does not resolve: C:\Users\admin\plantseg_data\plantseg
[smoke_explore_sites] mutations
  [PASS] mutation KILLED: site fake-quant as identity -- e.g. ['(A) site-matched reproduces the converted op: >= 99.9% exact, 100% within 1 LSB, all ten', "(A') op level, random quantized inputs on each site's own grid: matched >= 99.9% exact at the nine pools, >= 98% at the resize, 100% within 1 LSB at all ten"]
  [PASS] mutation KILLED: every site reads head.ff_add's grid -- e.g. ['producers: ReLU blocks block.1.0, Hardswish blocks block.1.2, head pool skip_add, resize ff_mul', "each pool site's input IS its producer's output (object identity, one forward)"]
  [PASS] mutation KILLED: TF32 switch sets cuDNN only -- e.g. ['set_tf32(True) sets both on']
  [PASS] mutation KILLED: a site check that never STOPs -- e.g. ["crash: KeyError: 'keeps_ff_mul_grid'"]
RESULT: SMOKE_EXPLORE_SITES PASS (44/44)
exit 0
```

stderr, distinct lines:

```text
/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/venv/lib/python3.11/site-packages/torch/ao/quantization/utils.py:317: UserWarning: must run observer before calling calculate_qparams. Returning default values.
warnings.warn(
```

</details>

<details><summary>smoke_explore_qat (stdout; stderr: 16 lines, 4 distinct)</summary>

```text
$ python -B -I scripts/smoke_explore_qat.py
[smoke_explore_qat] cases
  [PASS] E2 s42's parent hash is DL-146's
  [PASS] E2 s43/s44's parent hashes are the TEST-day manifest's
  [PASS] the clip selection's hash is DL-139's
  [PASS] the E6-E2 comparator is DL-146's value
  [PASS] C1 with --kd-logit refused
  [PASS] N2 with --tf32-off alone refused
  [PASS] N2 with both flags accepted
  [PASS] C1 at seed 42 refused
  [PASS] C1 at seed 43 and 44 accepted
  [PASS] N1 at seed 43 refused
  [PASS] N5 with --fp32-head accepted
  [PASS] D trigger: E2 s42's 0.38749080896377563 is below 0.3875
  [PASS] D trigger: 0.3875 and 0.39 meet it; NaN does not
  [PASS] C1 parent: a listed hash is accepted
  [PASS] C1 parent: stage E6-E2 from E2
  [PASS] C1 parent: an unlisted hash refused -- raised [parent_not_of_record] [parent_not_of_record] --expect-source-sha256 '0000000000000000000000000000000000000000000000000000000000000000' is not E2 seed 43's best checkpoint of record (
  [PASS] C1 parent: a listed hash under another run's name refused -- raised [parent_identity] [parent_identity] the parent is e2_s43_other_run/e2_student_best_iter72000.pt, the record's E2 seed 43 is e2_s43_lambda0.5_80k_a1/e2_student_best_iter72000.pt
  [PASS] C1 parent: a projection.pt in the E2 folder refused -- raised [parent_projection_present] [parent_projection_present] an E2 run directory holds no projection.pt
  [PASS] C1 parent with the real table: a fixture hash refused -- raised [parent_not_of_record] [parent_not_of_record] --expect-source-sha256 '6513e65502dcd552b7ea06b82e0b6aa9328db656f420b214e26789a01155f1c4' is not E2 seed 43's best checkpoint of record (
  [PASS] D parent: an x- N3 winner at 0.391 accepted
  [PASS] D parent: stage E6-E2 for an E2-stage winner
  [PASS] D parent below the trigger refused -- raised [d_trigger_not_met] [d_trigger_not_met] the parent's best VAL 0.3874 is below D's trigger 0.3875 (a number, never a judgment)
  [PASS] D parent of another item refused -- raised [d_parent_item] [d_parent_item] D's parent must be an exploratory ('N3', 'A1pp', 'A2') run; x-c1-s43 records item 'C1'
  [PASS] D parent outside an x- item folder refused -- raised [d_parent_not_exploratory] [d_parent_not_exploratory] /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_qat_nluga4sg/p1/
  [PASS] D with an E2 of record's hash refused -- raised [d_parent_of_record] [d_parent_of_record] D's parent is an exploratory winner, not an E2 of record
  [PASS] N1 loop runs 3 smoke steps
  [PASS] N1 step 1: zero distillation (ramp 0)
  [PASS] N1 step 2: loss = CE + Dice + r * 0.5 * KL(T 4), recomputed -- 6.114292621612549 vs 6.114292621612549
  [PASS] N1: the only term is logit_kd
  [PASS] N1 run_meta: terms logit_kd only, lambda 0.5, T 4, no projection
  [PASS] N1 telemetry: no CWD column
  [PASS] N1: the optimizer holds the student only (one group of record)
  [PASS] N1: an epoch checkpoint and a fake-quant VAL row were written
  [PASS] N1 real mode refuses injected loaders -- raised [real_run_overrides] [real_run_overrides] a real run takes no injected loaders, step cap, VAL cap or hook
  [PASS] N2: the ten sites were installed and fired -- {'se4': 4, 'se5': 4, 'se6': 4, 'se11': 4, 'se12': 4, 'se13': 4, 'se14': 4, 'se15': 4, 'head_pool': 4, 'head_resize': 4}
  [PASS] N2: run_meta records TF32 off (cuDNN and matmul) -- {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False, 'float32_matmul_precision': 'highest', 'NVIDIA_TF32_OVERRIDE': None, 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': None}
  [PASS] N2: run_meta records the substitution and the pinned file's sha256
  [PASS] N2: the pinned prepare_qat_model is restored afterwards
  [PASS] N2 launch gate (AM-24 v2.4): the parent's calibrated copy gives the ten sites
  [PASS] N2 launch gate STOPs when the code gives a different site set -- raised [site_set_mismatch] [site_set_mismatch] the code gives a different set of sites than AM-24 v2.4's: average-pool modules ['features.11.block.2.avgpool', 'features.12.block.2.avgpool
  [PASS] N5 training: the islands carry no fake-quant and qconfig None -- {'head.high_logits': {'type': 'Conv2d', 'fake_quant_modules': [], 'qconfig_none': True}, 'head.low_logits': {'type': 'Conv2d', 'fake_quant_modules': [], 'qconfig_none': True}, 'head.ff_add': {'type': 'FloatFunctional', 'fake_quant_modules': [], 'qconfig_none': True}}
  [PASS] N5 converted: classifiers and add in float on float inputs, DeQuantize before them -- {'island_types': {'head.high_logits': 'torch.nn.modules.conv.Conv2d', 'head.low_logits': 'torch.nn.modules.conv.Conv2d', 'head.ff_add': 'torch.ao.nn.quantized.modules.functional_modules.FloatFunctional'}, 'dequant_types': {'head.dequant_high': 'DeQuantize', 'head.dequant_low': 'DeQuantize'}}
  [PASS] N5 converted: every other conv quantized, fed quantized tensors -- 63
  [PASS] N5 converted: the output is float
  [PASS] N5 check discriminates: the registered head fails it
  [PASS] N5 FP32 MAC share in (0, 0.2) -- 0.0756
  [PASS] N5 precision label says mixed and not INT8
  [PASS] N5: an epoch checkpoint of a run_qat smoke converts with the islands in float
  [PASS] N5: the registered converter would refuse the key set (fake-quant entries absent)
  [PASS] CLI: C1 at seed 42 refused (exit 2)
  [PASS] CLI: a run ID without x- refused (exit 2)
  [PASS] CLI: --split TEST refused (exit 2)
  [PASS] CLI: a data root failing the TRAIN/VAL-only check refused, nothing written -- RESULT: REFUSED [data_root_not_trainval_only] [data_root_missing] data root does not resolve: C:\Users\admin\plantseg_data\plantseg. Nothing was written.
  [PASS] block B: a run outside an x- item folder refused (exit 2)
  [PASS] block B: --split TEST refused first, by name (exit 2)
[smoke_explore_qat] mutations
  [PASS] mutation KILLED: item flags unchecked -- e.g. ["crash: TypeError: 'NoneType' object is not subscriptable"]
  [PASS] mutation KILLED: D trigger always met -- e.g. ["D trigger: E2 s42's 0.38749080896377563 is below 0.3875", 'D trigger: 0.3875 and 0.39 meet it; NaN does not']
  [PASS] mutation KILLED: FP32 head a no-op -- e.g. ["crash: KeyError: 'head.dequant_high'"]
  [PASS] mutation KILLED: N1's KD term dropped -- e.g. ['N1 step 2: loss = CE + Dice + r * 0.5 * KL(T 4), recomputed']
RESULT: SMOKE_EXPLORE_QAT PASS (59/59)
exit 0
```

stderr, distinct lines:

```text
/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/venv/lib/python3.11/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/venv/lib/python3.11/site-packages/torch/ao/quantization/utils.py:317: UserWarning: must run observer before calling calculate_qparams. Returning default values.
return self.fget.__get__(instance, owner)()
warnings.warn(
```

</details>

<details><summary>smoke_explore_kd (stdout; stderr: 1 lines, 1 distinct)</summary>

```text
$ python -B -I scripts/smoke_explore_kd.py
[smoke_explore_kd] cases
  [PASS] DIST equals the official formulas on random tensors -- 1.9955079555511475 vs 1.9955079555511475
  [PASS] DIST inter and intra terms equal the official ones
  [PASS] masked DIST equals the official formulas on the valid rows
  [PASS] DIST is symmetric
  [PASS] DIST is zero for identical logits up to the official Pearson's eps (bound 1e-4; equal to the reference) -- 1.7285346984863281e-06 (reference 1.7285346984863281e-06)
  [PASS] A2 term = ramp * lambda * DIST on E2's OS8 validity mask
  [PASS] A2 term refuses stage e3
  [PASS] A2 term refuses a projection
  [PASS] A2 term refuses arm targets
  [PASS] A1'' gate admits the N4 alpha
  [PASS] A1'' gate admits it as run() calls it (allow_offgrid true)
  [PASS] A1'' gate still refuses another off-grid alpha
  [PASS] A1'' gate keeps the grid values
  [PASS] the pinned alpha gate is restored and refuses the N4 alpha again
  [PASS] the manifest's E3 s42 is the alpha-25 run of record
  [PASS] A1'' alpha read from N4's record of E3 s42
  [PASS] A1'' alpha is N4's value, unrounded
  [PASS] N4 record of another telemetry refused -- raised [alpha_source] [alpha_source] /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/n4/n4_other.json
  [PASS] N4 record with a wrong sha256 refused -- raised [alpha_source_sha256] [alpha_source_sha256] /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/n4/n4_ter
  [PASS] futility: 2.01 pp below stops
  [PASS] futility: exactly 2.0 pp below does not stop
  [PASS] futility: 1.9 pp below does not stop
  [PASS] futility: a non-finite VAL is not futility
  [PASS] futility wrapper raises at 40k when futile
  [PASS] futility wrapper wrote the VAL row, then x_futility_stop
[teacher] DRY-RUN uses an EXPLICIT MockTeacher (random, weight-free). This is a smoke substitute and is refused by the real-run path, which requires --teacher-ckpt.
[stage] E2 | objective: L_CE + L_Dice + lambda_logit*L_LogitKD | terms: logit_kd=on cwd_feat=off cwd_logit=off
[mode] DRY | torch 2.1.0+cpu | device=cpu | cuda_available=False | num_classes=116
[budget] max_iters=4 batch_size=2 val_interval=2 max_val_batches=2 num_workers=0 seed=42
[student] params=2,933,688 used_pretrained=False
[teacher] NMF control: none (MockTeacher has no NMF)
[teacher] frozen=True eval=True params=38,080 trainable_params=0 mock=True provenance=None
[data] train_index=6 val_index=4 (train+val only; the TEST split is never built here)
[loss] sup=CombinedCEDiceLoss | lambda_logit=1.0 T_logit=4
[opt] SGD lr=0.01 scheduler=PolynomialLR (total_iters=80000, power=0.9) trainable_tensors=176 param_groups=['student'] teacher_params_in_optimizer=0
[ramp] linear 0 -> target over first epoch -> ramp_iters=3 (factor = clamp((it-1)/(ramp_iters-1), 0, 1)) for each instantiated term: logit_kd; targets {'logit_kd': 1.0}
[grad-clip] none: AM-7 (DL-04) makes E1, E2 and E3 unclipped; a real run refuses the flag
[ckpt] dir=/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-a2-dry/train (verified OUTSIDE repo)
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=None override=False -> e2_run_meta.jsonl
[val     2/4] cm_batches=2 all_class_miou=0.00007 disease_only_miou(PROVISIONAL)=0.00007
[ckpt    2/4] new best all_class_miou=0.00007 -> /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-a2-dry/train/e2_student_best_iter2.pt
[iter    4/4] loss=7.9881 sup=5.9895 logit_kd=1.9986 ramp=1.0000 lr=9.99955000e-03
[val     4/4] cm_batches=2 all_class_miou=0.00007 disease_only_miou(PROVISIONAL)=0.00007

[CHECKS]
  logits_shape          : PASS
  c5_channels           : PASS
  head_logits_64x64     : PASS
  c5_32x32              : PASS
  teacher_logits_shape  : PASS
  teacher_feat_shape    : PASS
  loss_finite           : PASS
  has_expected_terms    : PASS
  optimizer_step        : PASS
  teacher_stayed_frozen : PASS
  teacher_params_frozen : PASS
  optimizer_excludes_teacher: PASS
  teacher_same_augmented_input: PASS
  distill_ramp_starts_at_zero: PASS
  distill_zero_at_step1 : PASS
  sup_added_unscaled    : PASS
  sup_positive_finite   : PASS
  grad_clip_applied     : PASS
  val_cm_accumulated    : PASS
  lr_non_increasing     : PASS
[summary] best_all_class_val_miou=0.00007 best_ckpt=/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-a2-dry/train/e2_student_best_iter2.pt

RESULT: PASS
  [PASS] dry A2 run inside the pinned trainer: RESULT PASS
  [PASS] dry A2: DIST was the distillation term every iteration
[teacher] DRY-RUN uses an EXPLICIT MockTeacher (random, weight-free). This is a smoke substitute and is refused by the real-run path, which requires --teacher-ckpt.
[stage] E2 | objective: L_CE + L_Dice + lambda_logit*L_LogitKD | terms: logit_kd=on cwd_feat=off cwd_logit=off
[mode] DRY | torch 2.1.0+cpu | device=cpu | cuda_available=False | num_classes=116
[budget] max_iters=4 batch_size=2 val_interval=2 max_val_batches=2 num_workers=0 seed=42
[student] params=2,933,688 used_pretrained=False
[teacher] NMF control: none (MockTeacher has no NMF)
[teacher] frozen=True eval=True params=38,080 trainable_params=0 mock=True provenance=None
[data] train_index=6 val_index=4 (train+val only; the TEST split is never built here)
[loss] sup=CombinedCEDiceLoss | lambda_logit=1.0 T_logit=4
[opt] SGD lr=0.01 scheduler=PolynomialLR (total_iters=80000, power=0.9) trainable_tensors=176 param_groups=['student'] teacher_params_in_optimizer=0
[ramp] linear 0 -> target over first epoch -> ramp_iters=3 (factor = clamp((it-1)/(ramp_iters-1), 0, 1)) for each instantiated term: logit_kd; targets {'logit_kd': 1.0}
[grad-clip] none: AM-7 (DL-04) makes E1, E2 and E3 unclipped; a real run refuses the flag
[ckpt] dir=/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-kl-dry/train (verified OUTSIDE repo)
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=None override=False -> e2_run_meta.jsonl
[val     2/4] cm_batches=2 all_class_miou=0.00007 disease_only_miou(PROVISIONAL)=0.00007
[ckpt    2/4] new best all_class_miou=0.00007 -> /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-kl-dry/train/e2_student_best_iter2.pt
[iter    4/4] loss=6.4983 sup=5.9868 logit_kd=0.5115 ramp=1.0000 lr=9.99955000e-03
[val     4/4] cm_batches=2 all_class_miou=0.00007 disease_only_miou(PROVISIONAL)=0.00007

[CHECKS]
  logits_shape          : PASS
  c5_channels           : PASS
  head_logits_64x64     : PASS
  c5_32x32              : PASS
  teacher_logits_shape  : PASS
  teacher_feat_shape    : PASS
  loss_finite           : PASS
  has_expected_terms    : PASS
  optimizer_step        : PASS
  teacher_stayed_frozen : PASS
  teacher_params_frozen : PASS
  optimizer_excludes_teacher: PASS
  teacher_same_augmented_input: PASS
  distill_ramp_starts_at_zero: PASS
  distill_zero_at_step1 : PASS
  sup_added_unscaled    : PASS
  sup_positive_finite   : PASS
  grad_clip_applied     : PASS
  val_cm_accumulated    : PASS
  lr_non_increasing     : PASS
[summary] best_all_class_val_miou=0.00007 best_ckpt=/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-kl-dry/train/e2_student_best_iter2.pt

RESULT: PASS
  [PASS] dry A2: the logit_kd column holds DIST, not the KL of a same-seed run -- [1.9986073970794678, 1.9988815784454346, 2.0035476684570312, 1.9986066818237305]
  [PASS] dry A1'': the pinned trainer refuses the off-grid alpha alone (exit 2)
[teacher] DRY-RUN uses an EXPLICIT MockTeacher (random, weight-free). This is a smoke substitute and is refused by the real-run path, which requires --teacher-ckpt.
[stage] E3 | objective: L_CE + L_Dice + lambda_logit*L_LogitKD + alpha_cwd*L_CWD_feat + 3*L_CWD_logit | terms: logit_kd=on cwd_feat=on cwd_logit=on
[mode] DRY | torch 2.1.0+cpu | device=cpu | cuda_available=False | num_classes=116
[budget] max_iters=4 batch_size=2 val_interval=2 max_val_batches=2 num_workers=0 seed=42
[student] params=2,933,688 used_pretrained=False
[teacher] NMF control: none (MockTeacher has no NMF)
[teacher] frozen=True eval=True params=38,080 trainable_params=0 mock=True provenance=None
[cwd] projection 1x1 conv student 160-ch C5 -> teacher 320-ch params=51,200 (TRAINING-ONLY; written with its optimizer group to projection.pt, never into the student checkpoint)
[data] train_index=6 val_index=4 (train+val only; the TEST split is never built here)
[loss] sup=CombinedCEDiceLoss | lambda_logit=1.0 T_logit=4 | alpha_cwd=3.2171 T_cwd=4 C_feat=320 | beta_cwd=3 T_cwd=4
[opt] SGD lr=0.01 scheduler=PolynomialLR (total_iters=80000, power=0.9) trainable_tensors=177 param_groups=['student', 'cwd_projection'] teacher_params_in_optimizer=0
[ramp] linear 0 -> target over first epoch -> ramp_iters=3 (factor = clamp((it-1)/(ramp_iters-1), 0, 1)) for each instantiated term: logit_kd, cwd_feat, cwd_logit; targets {'logit_kd': 1.0, 'cwd_feat': 3.2171, 'cwd_logit': 3}
[grad-clip] none: AM-7 (DL-04) makes E1, E2 and E3 unclipped; a real run refuses the flag
[ckpt] dir=/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-a1pp-dry/train (verified OUTSIDE repo)
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=None override=False -> e3_run_meta.jsonl
[val     2/4] cm_batches=2 all_class_miou=0.00007 disease_only_miou(PROVISIONAL)=0.00007
[ckpt    2/4] new best all_class_miou=0.00007 -> /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-a1pp-dry/train/e3_student_best_iter2.pt
[iter    4/4] loss=9.5757 sup=5.9864 logit_kd=0.5070 cwd_feat=0.4908 cwd_logit=0.5011 ramp=1.0000 lr=9.99955000e-03
[val     4/4] cm_batches=2 all_class_miou=0.00007 disease_only_miou(PROVISIONAL)=0.00007

[CHECKS]
  logits_shape          : PASS
  c5_channels           : PASS
  head_logits_64x64     : PASS
  c5_32x32              : PASS
  teacher_logits_shape  : PASS
  teacher_feat_shape    : PASS
  loss_finite           : PASS
  has_expected_terms    : PASS
  optimizer_step        : PASS
  teacher_stayed_frozen : PASS
  teacher_params_frozen : PASS
  optimizer_excludes_teacher: PASS
  teacher_same_augmented_input: PASS
  distill_ramp_starts_at_zero: PASS
  distill_zero_at_step1 : PASS
  sup_added_unscaled    : PASS
  sup_positive_finite   : PASS
  grad_clip_applied     : PASS
  val_cm_accumulated    : PASS
  lr_non_increasing     : PASS
[summary] best_all_class_val_miou=0.00007 best_ckpt=/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-a1pp-dry/train/e3_student_best_iter2.pt

RESULT: PASS
  [PASS] dry A1'': accepted with the substitution, alpha in run_meta, alpha_offgrid true
[teacher] DRY-RUN uses an EXPLICIT MockTeacher (random, weight-free). This is a smoke substitute and is refused by the real-run path, which requires --teacher-ckpt.
[stage] E2 | objective: L_CE + L_Dice + lambda_logit*L_LogitKD | terms: logit_kd=on cwd_feat=off cwd_logit=off
[mode] DRY | torch 2.1.0+cpu | device=cpu | cuda_available=False | num_classes=116
[budget] max_iters=4 batch_size=2 val_interval=2 max_val_batches=2 num_workers=0 seed=42
[student] params=2,933,688 used_pretrained=False
[teacher] NMF control: none (MockTeacher has no NMF)
[teacher] frozen=True eval=True params=38,080 trainable_params=0 mock=True provenance=None
[data] train_index=6 val_index=4 (train+val only; the TEST split is never built here)
[loss] sup=CombinedCEDiceLoss | lambda_logit=0.5 T_logit=4
[opt] SGD lr=0.01 scheduler=PolynomialLR (total_iters=80000, power=0.9) trainable_tensors=176 param_groups=['student'] teacher_params_in_optimizer=0
[ramp] linear 0 -> target over first epoch -> ramp_iters=3 (factor = clamp((it-1)/(ramp_iters-1), 0, 1)) for each instantiated term: logit_kd; targets {'logit_kd': 0.5}
[grad-clip] none: AM-7 (DL-04) makes E1, E2 and E3 unclipped; a real run refuses the flag
[ckpt] dir=/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kd_sv705oop/x-fut-dry/train (verified OUTSIDE repo)
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=None override=False -> e2_run_meta.jsonl
  [PASS] dry futility: the pinned trainer stops at the rule's iteration
  [PASS] dry futility: the VAL row then x_futility_stop end the telemetry, no run_end
  [PASS] CLI: --split TEST refused (exit 2)
  [PASS] CLI: a run ID without x- refused (exit 2)
  [PASS] CLI: N3 with the teacher of record's sha256 refused (exit 2)
  [PASS] CLI: a data root failing the TRAIN/VAL-only check refused, nothing written -- RESULT: REFUSED [data_root_not_trainval_only] [data_root_missing] data root does not resolve: C:\Users\admin\plantseg_data\plantseg. Nothing was written.

  [PASS] the KD pin's reused-module check runs and lists each module -- all_match True, differ []
[smoke_explore_kd] mutations (unit cases only)
  [PASS] mutation KILLED: DIST without its intra term -- e.g. ['DIST equals the official formulas on random tensors', 'DIST inter and intra terms equal the official ones']
  [PASS] mutation KILLED: futility margin ignored -- e.g. ['futility: 2.01 pp below stops', 'futility wrapper raises at 40k when futile']
  [PASS] mutation KILLED: a gate that admits every alpha -- e.g. ["A1'' gate still refuses another off-grid alpha"]
RESULT: SMOKE_EXPLORE_KD PASS (40/40)
exit 0
```

stderr, distinct lines:

```text
REFUSING to start the dry E3 run: --alpha 3.2171 is not in the AM-16 item 2 grid (25, 50, 100) (default 50); --allow-offgrid admits other values in dry runs only
```

</details>

<details><summary>smoke_explore_kl3 (stdout; stderr: 14 lines, 5 distinct)</summary>

```text
$ python -B -I scripts/smoke_explore_kl3.py
[smoke_explore_kl3] cases
  [PASS] the synthetic pair loads (sha-checked, identities checked)
  [PASS] identities: QNNPACK accuracy artifact and x86 latency copy
  [PASS] each module runs under its own engine -- x86, qnnpack
  [PASS] parity: both mIoUs and their difference
  [PASS] parity: argmax agreement over every pixel, and the logit differences -- agreement 0.2879, max |diff| 0.1517
  [PASS] both artifacts are byte-unchanged (read only)
  [PASS] a wrong sha256 refused -- raised [artifact_sha256_mismatch] [artifact_sha256_mismatch] the qnnpack artifact has sha256 243cf3e2461d2ca457b4e6a74476a152876d9cd30130e111d3ba197bb0ab4809, expected 00000000000000000000000000
  [PASS] a swapped pair refused (the x86 copy as the QNNPACK artifact) -- raised [qnnpack_identity] [qnnpack_identity] /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_kl3_kx6r8ee8/eval_a/fina
  [PASS] a pair of different epochs refused -- raised [pair_mismatch] [pair_mismatch] the two artifacts differ in epoch: 6 vs 7
  [PASS] a wrong stage refused -- raised [stage_mismatch] [stage_mismatch] the artifacts are stage 'E6', --stage 'E5'
  [PASS] the manifest's E6-E2 pair is DL-146's
  [PASS] clause 3 at the current HEAD: no TEST results on record -- raised [clause3_not_met] [clause3_not_met] DL-999 is not a row of docs/DECISION_LOG.md at HEAD: the TEST results are not on record, so KL-3 parity does not run yet (AM-24 clause 3)
  [PASS] clause 3: a row that is not RECORDED -- raised [clause3_not_met] [clause3_not_met] DL-900 has status 'DRAFT', not RECORDED
  [PASS] clause 3: a row without TEST -- raised [clause3_not_met] [clause3_not_met] DL-900 does not name TEST and the commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  [PASS] clause 3: a row without the commit -- raised [clause3_not_met] [clause3_not_met] DL-900 does not name TEST and the commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  [PASS] clause 3: a commit outside HEAD's history -- raised [clause3_not_met] [clause3_not_met] aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa is not in HEAD's history
  [PASS] clause 3: a RECORDED row naming TEST and an ancestor commit passes
  [PASS] CLI: --split TEST refused first (exit 2)
  [PASS] CLI: clause 3 unmet at HEAD refused, nothing written
[smoke_explore_kl3] mutations
  [PASS] mutation KILLED: clause 3 always met -- e.g. ['clause 3 at the current HEAD: no TEST results on record', 'clause 3: a row that is not RECORDED']
  [PASS] mutation KILLED: identity reader that hides roles -- e.g. ['the synthetic pair loads (sha-checked, identities checked)', 'a swapped pair refused (the x86 copy as the QNNPACK artifact)']
  [PASS] mutation KILLED: engine wrapper that does not set the engine -- e.g. ['each module runs under its own engine']
RESULT: SMOKE_EXPLORE_KL3 PASS (22/22)
exit 0
```

stderr, distinct lines:

```text
/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/venv/lib/python3.11/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/venv/lib/python3.11/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/venv/lib/python3.11/site-packages/torch/ao/quantization/utils.py:317: UserWarning: must run observer before calling calculate_qparams. Returning default values.
return self.fget.__get__(instance, owner)()
warnings.warn(
```

</details>

<details><summary>smoke_explore_n2r2 (stdout; stderr: 8 lines, 8 distinct)</summary>

```text
$ python -B -I scripts/smoke_explore_n2r2.py
[smoke_explore_n2r2] cases
  [PASS] E3-R2 seed 42's checkpoint sha256 is the addendum's da70d82f...
  [PASS] the parent record is DL-136's: run ID, checkpoint, its sha256, best.json, run_meta, telemetry, value -- []
  [PASS] N2R2's stage is E6-R2: source E3, parent arm E3-R2, an AM-23 stage, scripts/run_e6_r2.py's STAGE
  [PASS] E6-R2's epoch rules are AM-23's file (qat_select.rules_path_for_stage)
  [PASS] the λ selection of record reads through the pinned reader with the command's sha256
  [PASS] the α selection of record reads through the pinned reader with the command's sha256
  [PASS] their winners: λ 0.5; α 25.0 with best_val 0.36337557435035706 (AM-23 item 3(b)'s threshold)
  [PASS] the condition record's path of record is reports/arms/e6x_condition_R2.json; it is absent here
  [PASS] N2R2's run ID is x-n2r2-s42 and its records flags are E6-R2's flag names
  [PASS] N2R2's flag set is N2's (the same object)
  [PASS] N2R2 with N2's two flags at seed 42 accepted
  [PASS] N2R2 with one flag, none, an added flag or N5's refused (item_flags)
  [PASS] N2R2 at seed 43 and 44 refused (item_seed)
  [PASS] N2 unchanged: its two flags at seed 42 accepted, seed 43 refused
  [PASS] the parent check accepts the listed hash and identity, through E6-R2's loader
  [PASS] ... stage E6-R2, binding arm E3-R2, condition yes from DL-950, the records commit recorded
  [PASS] N2R2's parent record and binding are E6-R2's own (resolve_parent, am23_parent_binding)
  [PASS] N2R2 parent: another hash refused -- raised [parent_not_of_record] [parent_not_of_record] --expect-source-sha256 '0000000000000000000000000000000000000000000000000000000000000000' is not E3-R2 seed 42's best checkpoint (9b5f9ac
  [PASS] N2R2 parent: N2's E2 parent (73dbc32d...) refused -- raised [parent_not_of_record] [parent_not_of_record] --expect-source-sha256 '73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06' is not E3-R2 seed 42's best checkpoint (9b5f9ac
  [PASS] N2R2 parent: D's --parent-sha256 refused -- raised [parent_sha256_not_d] [parent_sha256_not_d] --parent-sha256 is D's flag
  [PASS] N2R2 parent: seed 43 refused -- raised [item_seed] [item_seed] N2R2 runs seed 42 only (E6-R2, AM-23 items 3 and 4), not 43
  [PASS] N2R2 parent: a records commit that is not the checkout's HEAD refused -- raised [records_commit_mismatch] [records_commit_mismatch] the records checkout's HEAD is 'a75900f1ff1b8a6920c4f553e5b341c5b21f8763', --records-commit 0000000000000000000000000000000000000000
  [PASS] N2R2 parent: a malformed records commit refused -- raised [records_commit_format] [records_commit_format] --records-commit must be the 40-hex HEAD of --records-root
  [PASS] N2R2 parent: a records folder that is not a checkout refused -- raised [records_root_not_checkout] [records_root_not_checkout] --records-root /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_
  [PASS] N2R2 parent: a missing records folder refused -- raised [records_root_missing] [records_root_missing] --records-root '/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2
  [PASS] N2R2 parent: the λ selection of another sha256 refused (pinned reader) -- raised [selection_sha256_mismatch] reports/derived/lambda_selection.json has sha256 719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145, expected 000000000000000000000000000000000000
  [PASS] N2R2 parent: a condition record of another sha256 refused (pinned binding) -- raised [condition_sha256_mismatch] reports/arms/e6x_condition_R2.json has sha256 8b0def9f4900caa4b59595e8559fcbcde77247af1b49fdeb04eac9260084c8a4, expected 000000000000000000000000000000000000000
  [PASS] N2R2 parent: a decision-log row that does not exist refused (pinned binding) -- raised [condition_dl_row] DL-951 is not a row of docs/DECISION_LOG.md at HEAD
  [PASS] N2R2 parent: an uncommitted condition record refused: none in the checkout (pinned binding) -- raised [condition_missing_file] /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_fx_fv2n_7kd/recs_norec/reports/arms/e6
  [PASS] N2R2 parent: a decision-log row that is not RECORDED refused (pinned binding) -- raised [condition_dl_row] DL-950 has status 'DECIDED', not RECORDED
  [PASS] N2R2 parent: a condition record changed after its commit refused (pinned binding) -- raised [condition_changed_since_head] reports/arms/e6x_condition_R2.json differs from its bytes at HEAD
  [PASS] N2R2 parent: this checkout as the records root refused: no condition record at this HEAD -- raised [condition_missing_file] /home/user/plantseg-thesis/reports/arms/e6x_condition_R2.json is not a file
  [PASS] N2R2 parent: another run's folder (attempt a1) under the record's hash refused -- raised [parent_identity] [parent_identity] the parent is not E3-R2 seed 42 as DL-136 records it: ['run_id'] differ (e3-r2_s42_lambda0.5_alpha25_80k_a1/e3_student_best_iter80000.pt)
  [PASS] N2R2 parent: a best.json other than DL-136's refused -- raised [parent_identity] [parent_identity] the parent is not E3-R2 seed 42 as DL-136 records it: ['best_json_sha256'] differ (e3-r2_s42_lambda0.5_alpha25_80k_a2/e3_student_best_iter8000
  [PASS] N2R2 parent: a run_meta other than DL-136's refused -- raised [parent_identity] [parent_identity] the parent is not E3-R2 seed 42 as DL-136 records it: ['run_meta_sha256'] differ (e3-r2_s42_lambda0.5_alpha25_80k_a2/e3_student_best_iter80000
  [PASS] N2R2 parent: a telemetry other than DL-136's refused -- raised [parent_identity] [parent_identity] the parent is not E3-R2 seed 42 as DL-136 records it: ['telemetry_sha256'] differ (e3-r2_s42_lambda0.5_alpha25_80k_a2/e3_student_best_iter8000
  [PASS] N2R2 parent: a best value other than DL-136's refused -- raised [parent_identity] [parent_identity] the parent is not E3-R2 seed 42 as DL-136 records it: ['best_val_miou_all_class'] differ (e3-r2_s42_lambda0.5_alpha25_80k_a2/e3_student_best_i
  [PASS] N2R2 parent: a run of arm E3-R1 refused (pinned binding) -- raised [parent_arm_mismatch] E6-R2's parent is E3-R2's run; its run_meta names arm 'E3-R1'
  [PASS] N2R2 parent: an E2 folder refused by E6-R2's loader (no e3 records) -- raised [parent_records_missing] the parent's run_meta file e3_run_meta.jsonl is missing
  [PASS] N2R2 parent with the real table: the fixture's hash refused -- raised [parent_not_of_record] [parent_not_of_record] --expect-source-sha256 '9b5f9ac87d00c890ccce6be585517326f44b88d56c836e94328505b3ef57cd5b' is not E3-R2 seed 42's best checkpoint (da70d82
  [PASS] N2 refuses E3-R2's parent: da70d82f... is not E2 seed 42's -- raised [parent_not_of_record] [parent_not_of_record] --expect-source-sha256 'da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4' is not E2 seed 42's best checkpoint of record (
  [PASS] N2 refuses an E3-R2 folder even under its hash: E6-E2's loader needs e2 records -- raised [parent_records_missing] the parent's run_meta file e2_run_meta.jsonl is missing
  [PASS] run_n2r2 runs the pinned run_qat on E6-R2's parent for 2 smoke steps
  [PASS] all ten sites were installed and fired -- {'se4': 5, 'se5': 5, 'se6': 5, 'se11': 5, 'se12': 5, 'se13': 5, 'se14': 5, 'se15': 5, 'head_pool': 5, 'head_resize': 5}
  [PASS] run_meta records TF32 off (cuDNN and matmul), as read and as set -- {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False, 'float32_matmul_precision': 'highest', 'NVIDIA_TF32_OVERRIDE': None, 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': None}
  [PASS] run_meta records the substitution: qat.py's prepare_qat_model, the file's sha256, item N2R2
  [PASS] the pinned prepare_qat_model is restored afterwards
  [PASS] run_meta is E6-R2's: stage, source stage, no projection, the parent's sha256, run ID and AM-23 binding
  [PASS] a real N2R2 run refuses the change removed -- raised [n2r2_change_required] [n2r2_change_required] a real N2R2 run carries N2's change: the ten sites and TF32 off
  [PASS] a real N2R2 run refuses injected loaders (run_qat's own refusal) -- raised [real_run_test_hooks] a real run takes no injected loaders, hooks, NaN injection, step cap or VAL cap
  [PASS] ... and the pinned name is restored after the refusal
  [PASS] the same source: both loads read one checkpoint into equal states
  [PASS] with the substitution removed, N2R2's first-step loss equals E6-R2's path (one synthetic batch) -- 5.981044769287109 vs 5.981044769287109
  [PASS] ... and its CE, Dice, gradient norm, lr and batch fingerprint, and the state after the step
  [PASS] ... and run_meta's recipe, census, TF32, source and binding fields -- []
  [PASS] with the substitution, the first-step loss differs: the comparison discriminates -- 5.980941295623779 vs 5.981044769287109
  [PASS] N2's launch gate on N2R2's parent: a calibrated copy of the loaded E3-R2 student gives the ten sites
  [PASS] block B accepts an N2R2 run (run_item)
  [PASS] ... a registered item with card fix 1, its run of stage E6-R2
  [PASS] block B gives card fix 1 and the stage check to N2 and N2R2 only (C1, N1, D as at 60dc294)
  [PASS] block B refuses an unknown item (run_item) -- raised [run_item] [run_item] /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run__0ie8mdm/x-z9/x_run_met
  [PASS] block B on an N2R2 run runs the pinned scripts: convert, from the checkout, refuses the 2-epoch smoke run as incomplete (exit 3); card fix 1 recorded as not run; INCOMPLETE --  epochs [1, 2], not 1..15 in order; the last row is not the epoch-15 completion record
RESULT: X BLOCK B INCOMPLETE -- see x_blockb.json (fake-quant-off VAL: not run, a registered step did not finish)
  [PASS] block B refuses an N2R2 item whose run is stage E6-E2 (run_stage), before any step -- RESULT: REFUSED [run_stage] N2R2's run is stage E6-R2; this run's run_meta records 'E6-E2'
  [PASS] card fix 1 runs for N2R2's selection
  [PASS] card fix 1 for N2R2: the pinned evaluator's CLI refuses stage E6-R2 (FQOFF_STAGES); recorded, not ok -- CliError: [fqoff_stage_not_in_scope] precision qat_fakequant_off scores the selected QAT models of ('E5', 'E6', 'E6-E2', 'E6-KD'); got stage 'E6-R2'
  [PASS] ... the selected epoch's fake-quant VAL is recorded from its epoch_end row all the same
  [PASS] the evaluator's stage list, read here: E6-E2 in FQOFF_STAGES, E6-R2 not -- ('E5', 'E6', 'E6-E2', 'E6-KD')
  [PASS] card fix 1 runs for N2's selection
  [PASS] card fix 1 for N2: the command passes the pinned evaluator's CLI checks; the value recorded with its summary's sha256, the checks all true
  [PASS] ... the command: qat_fakequant_off on VAL, canvas, CPU, batch 1, provisional; the winner's eNN.pt and sha256; the selection's and the telemetry's sha256
  [PASS] card fix 1: a summary of another stage is refused, not recorded as the value
  [PASS] card fix 1: an evaluator that exits 1 is recorded as failed with its message
  [PASS] card fix 1 for N2R2 through the script itself: scripts/evaluate_model.py exits 1, [fqoff_stage_not_in_scope], writes nothing -- CliError: [fqoff_stage_not_in_scope] precision qat_fakequant_off scores the selected QAT models of ('E5', 'E6', 'E6-E2', 'E6-KD'); got stage 'E6-R2'
  [PASS] card fix 1 for N2 through the script itself: past its CLI checks (no CliError), refused at the source binding on the synthetic selection (StageArtifactError on qat_selection.json; no dataset read) -- StageArtifactError: /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run__0ie8mdm/fq_eval_n2/qat_selection.json records rules_s
  [PASS] CLI: N2R2 at seed 43 refused (exit 2)
  [PASS] CLI: N2R2 without its records flags refused (exit 2), each named -- election-sha256, --alpha-selection, --alpha-selection-sha256, --condition-record, --condition-sha256, --condition-dl required (no default). Nothing was written.
  [PASS] CLI: N2R2 without --condition-dl refused (exit 2)
  [PASS] CLI: N2R2 under another run ID refused (exit 2)
  [PASS] CLI: N2 given N2R2's records flag refused (exit 2)
  [PASS] CLI: N2R2 with --split TEST refused first, by name (exit 2)
  [PASS] CLI: N2R2 with a data root that fails the TRAIN/VAL-only check refused, nothing written -- RESULT: REFUSED [data_root_not_trainval_only] [data_root_missing] data root does not resolve: C:\Users\admin\plantseg_data\plantseg. Nothing was written.
  [PASS] block B CLI: an N2R2 run passes run_item and reaches the guard's data-root check (exit 2 there)
  [PASS] block B CLI: an unknown item refused (exit 2, run_item)
[smoke_explore_n2r2] mutations
  [PASS] mutation KILLED: a permissive identity check -- e.g. ["N2R2 parent: another run's folder (attempt a1) under the record's hash refused", "N2R2 parent: a best.json other than DL-136's refused"]
  [PASS] mutation KILLED: N2R2 on E6-E2's stage -- e.g. ["N2R2's stage is E6-R2: source E3, parent arm E3-R2, an AM-23 stage, scripts/run_e6_r2.py's STAGE", "the parent check accepts the listed hash and identity, through E6-R2's loader"]
  [PASS] mutation KILLED: E6-R2's binding skipped -- e.g. ['... stage E6-R2, binding arm E3-R2, condition yes from DL-950, the records commit recorded', 'N2R2 parent: a condition record of another sha256 refused (pinned binding)']
  [PASS] mutation KILLED: run_n2r2 without the sites -- e.g. ['all ten sites were installed and fired', 'run_meta records TF32 off (cuDNN and matmul), as read and as set']
  [PASS] mutation KILLED: block B without card fix 1 -- e.g. ['... a registered item with card fix 1, its run of stage E6-R2', 'block B gives card fix 1 and the stage check to N2 and N2R2 only (C1, N1, D as at 60dc294)']
RESULT: SMOKE_EXPLORE_N2R2 PASS (88/88)
exit 0
```

stderr, distinct lines:

```text
/tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/venv/lib/python3.11/site-packages/torch/ao/quantization/utils.py:317: UserWarning: must run observer before calling calculate_qparams. Returning default values.
RESULT: INCOMPLETE [run_incomplete] -- /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run_81oz9xfl/x-n2r2-s42/x-n2r2-s42: epoch_end rows for epochs [1, 2], not 1..15 in order; the last row is not the epoch-15 completion record
RESULT: INCOMPLETE [run_incomplete] -- /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run__0ie8mdm/x-n2r2-s42/x-n2r2-s42: epoch_end rows for epochs [1, 2], not 1..15 in order; the last row is not the epoch-15 completion record
RESULT: INCOMPLETE [run_incomplete] -- /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run_a725lz3i/x-n2r2-s42/x-n2r2-s42: epoch_end rows for epochs [1, 2], not 1..15 in order; the last row is not the epoch-15 completion record
RESULT: INCOMPLETE [run_incomplete] -- /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run_m1s1scno/x-n2r2-s42/x-n2r2-s42: epoch_end rows for epochs [1, 2], not 1..15 in order; the last row is not the epoch-15 completion record
RESULT: INCOMPLETE [run_incomplete] -- /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run_qlz3yb_w/bbs/x-n2r2-s42/r: epoch_end rows for epochs [], not 1..15 in order; the last row is not the epoch-15 completion record
RESULT: INCOMPLETE [run_incomplete] -- /tmp/claude-0/-home-user-plantseg-thesis/4d5f454f-359b-5d5f-8e3d-856fc7643fa2/scratchpad/smokes/final/scratch_each/x_n2r2_run_qlz3yb_w/x-n2r2-s42/x-n2r2-s42: epoch_end rows for epochs [1, 2], not 1..15 in order; the last row is not the epoch-15 completion record
warnings.warn(
```

</details>


### A6. The commands (prompt item 5), at a8718d776c068cbe60e685c52afe545d8bc4fa36 or a later lane head

Every command runs from a clean checkout of lane/l-am24 at `<HEAD>`, PYTHONPATH unset, with
`PLANTSEG_DATA_ROOT=<the TRAIN/VAL-only staged root of AM-19 item 4(c)>`. Every argument in `<>` is the kit's to
fill:

| argument | what |
|---|---|
| `<HEAD>` | the 40-hex lane commit the pod checks out |
| `<X>` | the exploratory output root, outside the repository and every official folder |
| `<RUNS>` | the folder of the runs of record (§3) |
| `<E2s42>` | the folder of run e2_s42_lambda0.5_80k_a1 (§3) |
| `<E3R2>` | the folder of run e3-r2_s42_lambda0.5_alpha25_80k_a2, holding best.json, e3_run_meta.jsonl, e3_telemetry.jsonl and e3_student_best_iter80000.pt: DL-136's run, from keep_e3-r2_s42_lambda0.5_alpha25_80k_a2.tar (sha256 f1e1941e…) |
| `<REC>`, `<REC_HEAD>` | a clean clone of this repository at the master commit `<REC_HEAD>` that holds reports/arms/e6x_condition_R2.json and its RECORDED decision-log row; neither exists yet (A0) |
| `<COND_SHA>`, `<COND_DL>` | that record's sha256 and its decision-log row's id, as the row records them |
| `<T_N2>`, `<T_N2R2>` | the sha256 of each run's qat_telemetry.jsonl after it ends |
| `<host>` | the block B host label |

`<Q>` stands for `--real-run --confirm-real-run --clip-selection
<RUNS>/qat_pilot_20261014/evals/a2/clip_selection/clip_selection.json --clip-selection-sha256
c433145a6cfe0e133ebee8fcb70a76f05b60fc6462670abee34c5304704fe698 --num-workers 12 --device cuda --expect-head <HEAD>`
(§3; DL-139). DOCUMENTED preconditions: the Fable audit of the lane passes first (the prompt), and AM-24 item 11(e)
says N2 launches after item 11(b)'s record is entered. N2R2's launch also needs the condition record and its row
(A0). Neither run was launched here.

**N2** (unchanged from §3):

```bash
python -B -I scripts/explore_qat.py <Q> --item N2 --seed 42 --sites-on-input-grid --tf32-off --source-run-dir <E2s42> --expect-source-sha256 73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06 --run-id x-n2-s42 --out-dir <X>/x-n2-s42
```

**N2R2**:

```bash
python -B -I scripts/explore_qat.py <Q> --item N2R2 --seed 42 --sites-on-input-grid --tf32-off --source-run-dir <E3R2> --expect-source-sha256 da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4 --records-root <REC> --records-commit <REC_HEAD> --lambda-selection reports/derived/lambda_selection.json --lambda-selection-sha256 719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145 --alpha-selection reports/derived/alpha_selection.json --alpha-selection-sha256 1399b897278db17a8876c724b89c8d8e7a7b42ffe1d9c4d67002578a889f2dd8 --condition-record reports/arms/e6x_condition_R2.json --condition-sha256 <COND_SHA> --condition-dl <COND_DL> --run-id x-n2r2-s42 --out-dir <X>/x-n2r2-s42
```

The λ and α sha256 are MEASURED: the committed files at fe93a24, the same blobs at master aa8cac2. The pinned reader
checks them again in `<REC>`.

**Block B**, each after its run ends, on the CPU in the student image of record (AM-24 item 10(b)), with
`OMP_NUM_THREADS=8` (DL-251's card fix 5):

```bash
OMP_NUM_THREADS=8 python -B -I scripts/explore_qat_blockb.py --run-dir <X>/x-n2-s42/x-n2-s42 --expect-telemetry-sha256 <T_N2> --host-label <host> --threads 8 --run-id x-n2-s42-bb --out-dir <X>/x-n2-s42-bb
```

```bash
OMP_NUM_THREADS=8 python -B -I scripts/explore_qat_blockb.py --run-dir <X>/x-n2r2-s42/x-n2r2-s42 --expect-telemetry-sha256 <T_N2R2> --host-label <host> --threads 8 --run-id x-n2r2-s42-bb --out-dir <X>/x-n2r2-s42-bb
```

N2's block B ends `RESULT: X BLOCK B DONE (N2): epoch NN, VAL …; selected epoch fake-quant VAL …, fake-quant-off VAL
…`. Card fix 1 is made inside it, by this command, which block B builds and records in x_blockb.json (`<NN>`, `<W>`,
`<S>`: the selected epoch, its checkpoint's sha256 and qat_selection.json's sha256; nothing for the kit to run):

```bash
python -B scripts/evaluate_model.py --stage E6-E2 --model-role student --precision qat_fakequant_off --split val --protocol canvas --checkpoint <X>/x-n2-s42/x-n2-s42/epoch_ckpts/e<NN>.pt --qat-run-dir <X>/x-n2-s42/x-n2-s42 --qat-selection <X>/x-n2-s42-bb/eval/qat_selection.json --qat-selection-sha256 <S> --qat-telemetry-sha256 <T_N2> --expected-checkpoint-sha256 <W> --artifact-status provisional --device cpu --batch-size 1 --out-dir <X>/x-n2-s42-bb/fqoff/e<NN> --run-id x-n2-s42_fqoff_e<NN>_<first 12 of W>
```

N2R2's block B ends INCOMPLETE, exit 3, at that step (A3). The same command with `--stage E6-R2` and N2R2's paths is
the kit's command for N2R2's fake-quant-off VAL. It exits 1 `[fqoff_stage_not_in_scope]` at fe93a24's code (= master
aa8cac2's). It can run only from a checkout whose src/eval/artifacts.py holds "E6-R2" in FQOFF_STAGES, which no
commit does today. That is the STOP of A3; this session gives no command that works around it.

### A7. Deviations

- **(a) Hard limit 1: one search opened a test-named file.**
  - `git grep -l -I "RECORDS_ROOT" -- 'scripts/*.py' 'src/quant/*.py' | grep -vi test` had no test exclude, so git
    read every scripts/*.py file. MEASURED by count: that includes the one test-named file there; src/quant/ holds
    none. Its name and content were never displayed, because the output was filtered by `grep -vi test`.
  - An earlier `git grep -n -I "qat_fakequant_off" -- 'scripts/*.py' 'src/explore/*.py' ':(exclude,icase)*test*'`
    may have read it too. Whether git 2.43 applies that exclude inside `git grep` is unverified.
  - Neither was repeated. No path under docs/reference/ was touched; no TEST data, issue #11 CSV or dataset archive
    was read.
- **(b) The prompt's listing form lists nothing under git 2.43.** `git ls-files -- 'scripts/*.py'
  ':(exclude,icase)*test*'` and `… ':(exclude)*test*'` list 0 files, and the glob form `:(exclude,glob,icase)**/*test*`
  excludes nothing (MEASURED counts: 0, 0, 241 of 241). Listings were therefore `git ls-files -- '<dir>/*.py' | grep -vi
  test`, as §1 records. The fe93a24 = aa8cac2 claim rests on git tree ids, not on such a listing.
- **(c) More of the two docs was printed than the prompt names.**
  - The heads (first 330 characters) of DL-127, DL-164, DL-222, DL-227, DL-234, DL-237 and DL-269 were printed
    while finding "the AM-23 rows that record E6-R2's parent and condition". Of them only DL-138 (read in full) and
    DL-136 are used; DL-237's head names E6-R2's AM-23 rules file, which the code also gives. DL-269's head mentions
    E6-R1 and E6-R2 "restored as exploratory runs", which nothing here relies on.
  - AM-24 item 10 was printed whole, (a) to (c), and item 11's three-line lead too; only 10(b) is used.
- **(d) DL-250 and DL-251 come from master.** lane/l-am24 does not hold them. master was fetched by name; nothing
  was merged or pushed to it.
- **(e) Stricter than the prompt's identity list.** N2R2's identity check includes DL-136's telemetry sha256 and
  best value as well as the run ID, checkpoint, best.json and run_meta the prompt names.
- **(f) N2R2's run ID is enforced** as x-n2r2-s42 (refused otherwise); N2's is not.
- **(g) Card fix 1 is given to N2 and N2R2 only**, as the prompt says. DL-251 names C1, N1 and D as well; their
  block B is unchanged here (A8).
- **(h) No decision-log Status was updated** (CLAUDE.md's rule) because the lane edits no file but its own.
- **(i) N2R2's x_run_meta.json says amendment AM-24**, as every lane item's does (src/explore/guard.py AMENDMENT).
  Its recipe records `registered_by`: AM-29, drafted and not committed at this commit.

### A8. Open questions for the audit

1. **The records checkout.** Is a second checkout at a named commit, given to the pinned binding's `repo_root`, the
   right way for N2R2 to read E6-R2's committed inputs? The alternative is for lane/l-am24 to carry the condition
   record and its row, which needs non-lane files (the STOP rule).
2. **The condition record does not exist.** E6-R2 (AM-28) and N2R2 both need reports/arms/e6x_condition_R2.json
   (scripts/e6x_condition.py `record`, its G0-G10 gates) and a RECORDED row with its text, on master, before either
   launches. DL-136 says the condition is "yes" (0.3842073380947113 > 0.36337557435035706); the record was never made.
3. **FQOFF_STAGES (A3).** Should "E6-R2" join src/eval/artifacts.py:66, so that N2R2 (and AM-28's E6-R2, if AM-24
   item 10(b)'s form applies to it) can carry its fake-quant-off VAL? If not, N2R2's block B stays INCOMPLETE at
   that step by design.
4. **Card fix 1 for C1, N1 and D.** DL-251 asks it for them too; the prompt limits it to N2 and N2R2.
5. **Labels and comparators.** Should N2R2's records name AM-29 (they say AM-24, with `registered_by` AM-29)? What
   are its comparators? The recipe names E6-R2 seed 42 (AM-28) as the matched run, from the prompt; AM-29's text was
   not available here.
6. **OMP_NUM_THREADS=8** (DL-251's card fix 5) is set in the commands, not in code: block B's `--threads` sets only
   its own process, and the evaluator subprocesses inherit the environment.

### A9. Commits and push

- `git status` and `git diff` were checked with the scoped pathspec before each commit (AGENTS.md rule 6). The new smoke was staged by its path (`git add -- scripts/smoke_explore_n2r2.py`). Both commits used `git commit --only -F <message> -- <explicit paths>`.
- **No trailers.** Neither commit message carries a Co-Authored-By, Claude-Session or Signed-off-by line, or any other trailer (`git log --format='%(trailers)'` is empty for a8718d7), against the session's attribution reminder, as hard limit 2 directs. The author is the checkout's configured identity, as for every lane commit.

  | commit | contents |
  |---|---|
  | a8718d7 | N2R2, block B's N2R2 route and card fix 1, scripts/smoke_explore_n2r2.py, the runner |
  | (this commit) | this addendum |

- `git diff --name-only fe93a24..HEAD` lists 29 files, all lane files (A4). A count without names shows 0 test-named paths changed since fe93a24 (MEASURED). No file under src/ or configs/ changed outside src/explore.
- **Push.** `git push -u origin lane/l-am24`, a fast-forward from 60dc294 (the remote tip was checked first), that branch only, never forced; master was not pushed or merged. The pushed head is this commit; its id and this file's sha256 are in the download copy and the chat reply.
