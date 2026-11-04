# Lane L-AM24 — the exploratory, VAL-only code of ORCH-5's EXPLORE-2 (and addendum L-AM24-ADD-1)

Lane L-AM24, from the brief l_am24_brief_20261103.md (sha256 6c3c40ec…ad0eec7, 7,513 B), its addendum L-AM24-ADD-1
(N5 and E5, in the session's first message) and the audit EXPLORE-AUDIT-1 (explore1_audit_report_fable_20261103_2.md,
sha256 07e4d682…3d6395, 48,472 B; read in full before any code). Base master fe93a24; branch lane/l-am24; new files
only. Labels as the audit's: MEASURED (a number from a run here or a committed record), REPOSITORY-PROVEN (read in the
code at fe93a24), INFERRED.

## 0. Status (read first)

- **Built and smoked: E0, E1 (N4), E2 (B1, with N2's site machinery), E3 (C1, N1, N2, N5 `--fp32-head`, D, and
  their block B), E4 (N3, A1'', A2, the 40k futility flag), E5 (KL-3 parity).** Six smokes, 235 cases, 19 mutations
  killed, all PASS (§4). Nothing in the brief's or the addendum's list is unbuilt; §5 lists what the smokes could not
  reach (real data, a GPU, the teacher stack) and every design choice the audit should judge.
- **Late.** The brief's stop time, 10:00 +08:00 on 4 Nov, passed while this session was stalled. The first turn
  (from 2026-11-03 15:44Z) read the specification, built the venv (15:54–15:55Z) and created the branch. It then
  stopped in the reading phase with nothing written, and the worker restarted. No work happened until the user's
  "Try again" at 2026-11-04 04:24Z (12:24 +08:00). The whole build, the smokes, this report and the push came after
  that, on the user's instruction. No account limit stopped the lane.
- **Nothing was launched**: no training, no GPU, no pod, no dataset read, no TEST path, no teacher checkpoint.
  `git diff --no-renames --name-status fe93a24..HEAD` lists only A (§7). No existing file was modified, renamed or
  deleted.

## 1. Base, branch, environment and the deviations

- **Base.** `git ls-remote origin refs/heads/master` gave fe93a24dda02978789a4e92d2029b10686de0d29 (the local
  remote-tracking ref was stale at 31f782d). `refs/heads/lane/l-am24` did not exist on the remote.
- **The sparse clone could not be made.** The repository's PreToolUse hook (.claude/hooks/protect_reference_pdf.py,
  rule L-PROT G) denies `git clone` and `git sparse-checkout` ("works on the whole tree or object store"). It was not
  worked around. The lane ran in the session's existing checkout, with `git checkout -b lane/l-am24` at HEAD
  fe93a24, as lanes Q2-G and L-FQOFF did. That checkout holds docs/reference/ and test-named paths. None was opened,
  read, listed or hashed, with one disclosure: under git 2.43, `:(exclude,icase)*test*` does not exclude when the
  positive pathspec is a bare directory or a cross-directory glob (MEASURED: `git ls-files -- 'src/stats'
  ':(exclude,icase)*test*'` lists one test-named path, `'src/stats/*'` none). An early outline command
  (`git ls-files -- … 'src/*.py' ':(exclude,icase)*test*'`) therefore printed the name src/stats/tests.py once. The
  file was not opened; src.stats was never imported. From then on every listing was piped through `grep -vi test`.
- **Harness instructions refused**, as the brief directs: the session's branch claude/peaceful-albattani-y7x5bf
  (nothing was committed or pushed to it) and its Co-Authored-By / Claude-Session commit trailers. No commit of this
  lane has a trailer (§7).
- **Search paths.** No search used the repository root or docs/ as its path: named files (DECISION_LOG.md,
  PREREGISTRATION_AMENDMENTS.md, the TEST-day manifest, requirements files) and named folders (src/quant, src/training,
  src/eval, src/distill, scripts, configs/kd_arms, docs/lane_reports, docs/lane_specs, reports) only.
- **The venv** (approved by the user in this session: "Approved … record the wheel source and sha256s in the report").
  `/usr/bin/python3.11 -m venv` (Python 3.11.18) in the session scratchpad; the 23 PyPI pins of requirements.lock that
  q2-qat.md §17.1 lists, and torch 2.1.0+cpu / torchvision 0.16.0+cpu from https://download.pytorch.org/whl/cpu
  (reachable; the PyPI fallback was not needed), every wheel downloaded first, hashed, then installed `--no-deps
  --no-index`. `pip check`: no broken requirements. Engines: qnnpack, none, onednn, x86, fbgemm; 4 threads; no CUDA.
  No teacher stack (mmcv, mmengine, mmsegmentation): the teacher of record and R2 could not be loaded here.

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

- **Python** ran as the venv's interpreter with `-I` (the system python3 has no torch), and the smokes with `-B -I`.
- **No web access was used.** A2's DIST formulas are transcribed from the audit's description of the official code
  (S1c) and the paper; see §5 (c).

## 2. The files (all new)

| file | what it does |
|---|---|
| src/explore/__init__.py | package marker; imports nothing |
| src/explore/guard.py | **E0.** The guard every entry point runs first: the item; the split (TEST by name in any case first, then anything outside TRAIN/VAL; scoring tools VAL only); `x-` run IDs; SL-1 on every input path (as given and resolved); the output directory (outside the repository, named after the run ID, under no official run or record folder by name or by content, not inside an input, fresh); `src.data.isolation.assert_trainval_only_root` on the root the pinned loaders read (configs/data.py DATA["root"]) before any data is read; `x_run_meta.json` (exploratory true, amendment AM-24, split val, artifact_status exploratory, item, run ID, full recipe, claim line of clause 7, git commit, the pinned modules' sha256 through src/quant/ptq.py code_provenance, the lane files' sha256, and for KD items the KD pin's reused_module_hashes comparison), written exclusively |
| src/explore/smokekit.py | the smokes' case recorder, refusal checks and mutation runner |
| src/explore/decompose.py, scripts/explore_decompose.py | **E1, N4.** (a) per-term loss magnitudes over the last 10,000 iterations of a KD run's telemetry and the A1'' alpha = λ·mean(L_logitKD)/mean(L_feat), printed with its inputs; a term absent, null or non-finite in the window is reported and never estimated; an interleaved relaunch (duplicate iterations) or a run_abort is refused. (b) parent / fake-quant (selected epoch) / converted / fake-quant-off VAL per QAT model, one table (DL-176's form), each cell with its file and sha256 or the reason it is empty |
| src/explore/sites.py | the ten D-1 sites, derived from the code (§5 (b)): the eight SE pools features.{4,5,6,11,12,13,14,15}.block.2.avgpool, the LR-ASPP pool head.context.0 and the head's OS8 resize (applied as a pre-hook on head.high_logits, its only consumer); each with its producer fake-quant; `SiteMatcher` fake-quantizes any subset on the producer's current grid (straight-through), adding no state; `set_tf32` sets both backend flags |
| src/explore/b1_split.py, scripts/explore_b1.py | **E2, B1.** For one QAT model's selected epoch: baseline (registered fake-quant), each site alone (SE pools first), all ten, TF32 off and all ten + TF32 off (CUDA only), and the converted artifact of record re-scored by the same scorer; the reconciliation: gap, per-site shares, SE-pool sum, sum vs all-ten (interaction), TF32's share, residual of the full corrected pass, converted re-scored vs of record; item (iv)'s hardware match recorded; inputs sha-checked before and after |
| src/explore/valscore.py | VAL scoring in-process through the evaluator library unchanged (canvas adapter, expected manifest, `evaluate_model`, the evaluator's device-forward hook): the union-present all-class mIoU; VAL only; injected batches for smokes |
| src/explore/synthetic.py | the smokes' random students (optionally with SE/context weights amplified, optionally the FP32 head), calibrated and converted through the pinned prepare/convert |
| src/explore/qat_explore.py | **E3.** The exploratory parent check (E2 s42/s43/s44's best checkpoints of record by sha256, through the pinned `resolve_parent`; D's exploratory winner by `--parent-sha256`, trigger 0.3875 as a Decimal of the value's repr); each item's one flag set and seeds; `run_qat_logit_kd`, N1's loop (run_qat's lines plus λ 0.5 logit KD on stage e2, E6-KD's ramp and teacher stream, no CWD term) |
| src/explore/fp32_head.py | **N5.** `Fp32LogitHead` (the registered head's own submodules plus two DeQuantStubs; the classifiers and the add carry qconfig None), the training and converted island checks, the FP32 MAC share, and N5's own conversion onto the FP32-head skeleton; PRECISION_LABEL says MIXED PRECISION (not INT8) |
| src/explore/patching.py | recorded run-time substitutions of one pinned module attribute for one call (N2's prepare, A1'''s alpha gate, A2's loss, the futility writer) |
| scripts/explore_qat.py | **E3's entry.** The guard and the real-run gates that apply (config pins, HEAD = --expect-head, pinned QAT paths and lane files clean, workers, CUDA present and uninitialised, TF32 at defaults before launch, QNNPACK, the clip selection of record with clip 5.0), the parent check, the strict source load, N1's teacher of record through src/quant/qat_kd.py `teacher_of_record`; then the pinned `run_qat` (C1, D), with the sites substituted and TF32 off (N2), with the FP32 head (N5), or `run_qat_logit_kd` (N1) |
| scripts/explore_qat_blockb.py | **Block B** of an exploratory QAT run: for C1, N1, N2 and D the existing scripts unchanged as subprocesses (qat_epoch_eval.py convert/score, select_qat_epoch.py with the rules file of the run's stage, qat_epoch_eval.py finalize); for N5 its own steps (convert onto the FP32-head skeleton, the islands report, CPU VAL scoring, highest VAL with ties to the earlier epoch and non-finite epochs excluded, a TorchScript artifact with an x_mixed_precision.json identity); x_blockb.json |
| src/explore/dist_loss.py | **A2's DIST**: rows (B·H·W, C), softmax at τ 1, inter = 1 − mean row Pearson, intra = the same over classes, β = γ = 1; masked to E2's OS8 validity; `dist_distillation_losses`, the A2 term with train_distill's signature (stage e2 only) |
| src/explore/kd_explore.py, scripts/explore_kd.py | **E4.** N3 (R2's iter_24000.pth by DL-122's sha256, the record's teacher config), A1'' (stage e3, alpha from N4's record bound to E3 s42's telemetry sha256 in the TEST-day manifest, admitted by a recorded alpha-gate substitution), A2 (the DIST substitution, λ passed as 1); `--futility-stop` (iteration 40,000, more than 2.0 pp below the comparator's VAL at 40,000; the comparator's telemetry bound to the manifest's completion sha256; records an x_futility_stop row and stops); then the pinned `train_distill.main` with its own real-run gates |
| src/explore/kl3_parity.py, scripts/explore_kl3_parity.py | **E5, KL-3.** One pair (the QNNPACK artifact of record and its x86 latency copy) scored on VAL on the CPU, each under its own engine, with argmax agreement and logit differences; read only (sha256 before and after; identities read from the zip; with --slot, the TEST-day manifest's pair); AM-24 clause 3: refuses until a RECORDED decision-log row names TEST and the TEST results' commit, in HEAD's history |
| scripts/smoke_explore_guard.py, _decompose.py, _sites.py, _qat.py, _kd.py, _kl3.py | the six smokes (§4) |
| scripts/smoke_explore_all.py | runs the six smokes in fresh processes and prints one table |
| docs/lane_reports/l-am24.md | this report |

## 3. Launch commands

Every command runs from a clean checkout of lane/l-am24 at its final commit `<HEAD>`, with
`PLANTSEG_DATA_ROOT=<the TRAIN/VAL-only staged root of AM-19 item 4(c)>` (the guard and the pinned trainers check
it), PYTHONPATH unset, and writes into `<X>`: an exploratory root outside the repository and outside every official
run or record folder (e.g. /workspace/explore on a pod). `<RUNS>` is the folder holding the runs of record (the
laptop's C:\Users\admin\plantseg_runs, or a pod copy); the sub-paths below are the TEST-day manifest's. The QAT and
KD runs need a CUDA pod; KL-3, N4 and B1 on a CPU (B1 on the QAT pod's GPU class to meet item (iv)). N1, N3, A1''
and A2 need the teacher stack: the KD image of record (cb413304…). C1, N2, N5 and D need only the student stack.
Every block B is a separate command after its run completes. Order: N4 → B1 → C1, N1, N2, N5 → N3 (≤ 5 Nov 12:00,
audit §7.2) and A1'' / A2 if funded → D if a trigger is met → KL-3 after the TEST results are committed.

**N4 (a), the per-term magnitudes and the A1'' alpha** (E3 seed 42 of record: run e3_s42_lambda0.5_alpha25_80k_a1,
its e3_telemetry.jsonl sha256 734417e4b33e64239cebe2485e3eace03a795163b47bf50667517be40c161d2e per the manifest):

```bash
python -B -I scripts/explore_decompose.py terms --run-dir <RUNS>/kd_alpha_20261013/e3_s42_lambda0.5_alpha25_80k_a1 --run-id x-n4-e3s42-terms --out-dir <X>/x-n4-e3s42-terms
```

**N4 (b), the four-way table**: write `<X>/n4_spec/models.json`, a list of eight objects {label, parent_best_json,
qat_run_dir, selection, fqoff_summary} for E5 s42/43/44, E6 s42/43/44, E6-E2 s42 and E6-KD s42 (the selections are
the manifest's: e.g. `<RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/qat_selection.json`; the fake-quant-off
summaries are DL-176's valfq_* artifacts' summary.json):

```bash
python -B -I scripts/explore_decompose.py qat-table --spec <X>/n4_spec/models.json --run-id x-n4-qat-table --out-dir <X>/x-n4-qat-table
```

**B1, E6-E2 first** (the same command for any QAT model of record, with its selection and artifact from the
manifest; drop `--device cuda` for a CPU run, which skips the TF32 passes and says so):

```bash
python -B -I scripts/explore_b1.py --qat-run-dir <RUNS>/qat_e6e2_20261020/e6-e2_s42_clip5.0_a1 --selection <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/qat_selection.json --selection-sha256 dd37419fcfff4f02abbd6e7f50ea3ff19214a8bec070d42a925038d43426b809 --artifact <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/converted/e06_int8_qnnpack.torchscript.pt --stage E6-E2 --device cuda --threads 8 --run-id x-b1-e6e2-s42 --out-dir <X>/x-b1-e6e2-s42
```

The QAT runs share these flags, written `<Q>` below:
`--real-run --confirm-real-run --clip-selection <RUNS>/qat_pilot_20261014/evals/a2/clip_selection/clip_selection.json
--clip-selection-sha256 c433145a6cfe0e133ebee8fcb70a76f05b60fc6462670abee34c5304704fe698 --num-workers 12
--device cuda --expect-head <HEAD>`. E2's run folders (`<E2s42>` etc.) hold best.json and the checkpoint it names
(DL-120's keep set).

**C1 seed 43** and **C1 seed 44** (E2 s43 e3c4e05c…, s44 be48da05…; DL-218 and the manifest):

```bash
python -B -I scripts/explore_qat.py <Q> --item C1 --seed 43 --source-run-dir <RUNS>/kd_alpha_20261013/e2_s43_lambda0.5_80k_a1 --expect-source-sha256 e3c4e05ccfffa5c78d88a9309542108c7a974f03897d4564bd0644a032b4d385 --run-id x-c1-s43 --out-dir <X>/x-c1-s43
```

```bash
python -B -I scripts/explore_qat.py <Q> --item C1 --seed 44 --source-run-dir <RUNS>/kd_alpha_20261013/e2_s44_lambda0.5_80k_a1 --expect-source-sha256 be48da058a55a3efb049ead5f48409ab9061aad549708a89805fd7113ac2fa56 --run-id x-c1-s44 --out-dir <X>/x-c1-s44
```

**N1** (E2 s42 73dbc32d…; the teacher of record, sha256 8c0e649a…, DL-88):

```bash
python -B -I scripts/explore_qat.py <Q> --item N1 --seed 42 --kd-logit --source-run-dir <E2s42> --expect-source-sha256 73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06 --teacher-ckpt /abs/path/to/iter_24000.pth --teacher-ckpt-sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e --run-id x-n1-s42 --out-dir <X>/x-n1-s42
```

**N2**:

```bash
python -B -I scripts/explore_qat.py <Q> --item N2 --seed 42 --sites-on-input-grid --tf32-off --source-run-dir <E2s42> --expect-source-sha256 73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06 --run-id x-n2-s42 --out-dir <X>/x-n2-s42
```

**N5** (addendum; its block B is N5's own path, automatically):

```bash
python -B -I scripts/explore_qat.py <Q> --item N5 --seed 42 --fp32-head --source-run-dir <E2s42> --expect-source-sha256 73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06 --run-id x-n5-s42 --out-dir <X>/x-n5-s42
```

**D** (only if an exploratory FP32 run's best.json value is ≥ 0.3875; on the higher one only if two qualify;
`<winner>` is N3's, A1'''s or A2's trainer folder, e.g. `<X>/x-n3-s42/x-n3-s42`):

```bash
python -B -I scripts/explore_qat.py <Q> --item D --seed 42 --source-run-dir <winner> --parent-sha256 <sha256 of the checkpoint its best.json names> --run-id x-d-s42 --out-dir <X>/x-d-s42
```

**Block B of each QAT run**, e.g. C1 seed 43 (`<T>` = the sha256 of the run's qat_telemetry.jsonl after it ends):

```bash
python -B -I scripts/explore_qat_blockb.py --run-dir <X>/x-c1-s43/x-c1-s43 --expect-telemetry-sha256 <T> --host-label <host> --threads 8 --run-id x-c1-s43-bb --out-dir <X>/x-c1-s43-bb
```

The KD runs share `--real-run --confirm-real-run --num-workers 12 --expect-head <HEAD>`, written `<K>`. With the
optional futility flag, N3 and A2 take `--futility-stop --futility-comparator-telemetry <E2s42>/e2_telemetry.jsonl
--futility-comparator-sha256 4f8400f2b2ba70c33d01b5c981b1c928a3ec25d5ae18d90f30a741cac701f895`; A1'' takes
`--futility-stop --futility-comparator-telemetry <RUNS>/kd_alpha_20261013/e3_s42_lambda0.5_alpha25_80k_a1/e3_telemetry.jsonl
--futility-comparator-sha256 734417e4b33e64239cebe2485e3eace03a795163b47bf50667517be40c161d2e`.

**N3** (arm teacher R2, M12 iteration 24000: iter_24000.pth, sha256 d5085961…, DL-122):

```bash
python -B -I scripts/explore_kd.py <K> --item N3 --teacher-ckpt /abs/path/to/teacher-r2_s42_40k_a1/iter_24000.pth --teacher-ckpt-sha256 d5085961e1529d83b1fc13ff04fac83957bcc44c4f70a0635ff0d840083ffab9 --run-id x-n3-s42 --out-dir <X>/x-n3-s42
```

**A1''** (after N4 (a); `<N4>` = `<X>/x-n4-e3s42-terms/n4_terms.json` and its sha256):

```bash
python -B -I scripts/explore_kd.py <K> --item A1pp --teacher-ckpt /abs/path/to/iter_24000.pth --teacher-ckpt-sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e --alpha-from <N4> --alpha-from-sha256 <sha256 of N4> --run-id x-a1pp-s42 --out-dir <X>/x-a1pp-s42
```

**A2**:

```bash
python -B -I scripts/explore_kd.py <K> --item A2 --teacher-ckpt /abs/path/to/iter_24000.pth --teacher-ckpt-sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e --run-id x-a2-s42 --out-dir <X>/x-a2-s42
```

**KL-3 (E5; only after the TEST results are committed)**: one command per pair, with the decision-log row
`DL-<n>` that records the committed TEST results and `<commit>` its commit. E6-E2 first:

```bash
python -B -I scripts/explore_kl3_parity.py --slot E6-E2:s42:sel --stage E6-E2 --qnnpack-artifact <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/converted/e06_int8_qnnpack.torchscript.pt --qnnpack-sha256 12e60c895ff067df97bb8f9f644af774718ae1c65bdbef677a2777a83cae7c37 --x86-copy <RUNS>/qat_e6e2_20261020/evals/a1/e6-e2_s42_clip5.0_a1/final/e06_int8_x86_latency.torchscript.pt --x86-sha256 eed07f984c6bcf3eff61171c146be254725350d9bc8c8d1c229c638533492f57 --test-results-dl DL-<n> --test-results-commit <commit> --threads 8 --run-id x-kl3-e6e2-s42 --out-dir <X>/x-kl3-e6e2-s42
```

The other six pairs (the manifest's; same command with these values; artifact `converted/eNN_int8_qnnpack…`, copy
`final/eNN_int8_x86_latency…` in the evaluation folder):

| slot | stage | evaluation folder | epoch | QNNPACK sha256 | x86 copy sha256 |
|---|---|---|---|---|---|
| E5:s42:sel | E5 | qat_pilot_20261014/evals/a1/e5_s42_clip5.0 | 10 | 220b2b0e1cc63cc2e3b8622174871de6cf5b2725deb8df36fad61b3c3f904c61 | fd5f4b31139dd817f3d4c26d7c9a62ca44f5fbcb7da6b21d9d7823aaf6a71a8c |
| E5:s43:sel | E5 | qat_record_20261017/evals/a1/e5_s43_clip5.0_a1 | 11 | 5f0b3ab8d777ee7e304c5ccd3763d52e5e9ad7ce6bfbc5b5ad42d3e02ebb7a43 | 8aa1be086caac65518a92c524ea981968302eb0b8b3700876d25821481a8ec3c |
| E5:s44:sel | E5 | qat_record_20261017/evals/a1/e5_s44_clip5.0_a1 | 15 | 0cc1ef811fefe4053cb24cb393285e17a51c15bce5f2176fffb3cdeb6cce2f70 | 45d78007a3d8c2e8f80e1d6ed1ccc0c2a854425b4c0bd0f1e9623b941dd0eeee |
| E6:s42:sel | E6 | qat_record_20261017/evals/a1/e6_s42_clip5.0_a1 | 15 | cf14ae0362d07f6a576405320780622f504f1bccd032335c2968c02e37dbb0bb | f81c2f73a2ccce407228f15c7069511d92352a23d59db1360005bcf185381d79 |
| E6:s43:sel | E6 | qat_record_20261017/evals/a1/e6_s43_clip5.0_a1 | 13 | 0b0a6bada81e4eb29b48514efa5272c400569c035942b0f97145af25e414ac35 | 6fbf91a521cf6055e1f6ab39bfc08191a9f484456c61e0296473cb38415a5547 |
| E6:s44:sel | E6 | qat_record_20261017/evals/a1/e6_s44_clip5.0_a1 | 13 | ee7e22da92dfb4f27fbf4a8b93388beb5aabe857c7ac23195ff68f5a68c2d2ff | e74ddd90cda05eb53383ede560f09ac81cd9ccb9091e88421a7111a755639a5e |

E6-KD has no x86 copy (LB31). Every command refuses with exit 2, before writing anything, if any gate fails.

## 4. Smokes

All with the session venv: `python -B -I scripts/smoke_explore_all.py` (L_AM24_SMOKE_DIR in the session scratchpad),
MEASURED at the final code commit:

| smoke | cases | mutations killed | result | what it shows |
|---|---|---|---|---|
| smoke_explore_guard | 51 | 5 | PASS | TEST refused in every case form and first; non-TRAIN/VAL splits; VAL-only tools; x- run IDs; SL-1 on inputs (string and a link's target); each output refusal (repository, name, official folder by name, an ancestor holding best.json, inside an input, not fresh); the isolation call on a missing root and a count mismatch, a passing fixture root; x_run_meta.json's fields, commit and sha256s; exclusive write; nothing written after a refusal; each guard made permissive fails a case |
| smoke_explore_decompose | 30 | 1 | PASS | hand-computed fixture: window 21..120, mean(L_logitKD) 2.0, mean(L_feat) 0.004, alpha 250; an absent term said so and no alpha formed; null values, interleaved relaunches, run_abort and short runs refused; the four-way table with DL-176's +1.02 pp; nulls with reasons; the CLI end to end and its refusals (non-x-, in-repository, TEST) |
| smoke_explore_sites | 35 | 3 | PASS | the ten sites and producers (object identity at run time); **N2's core claim**: (A) at all ten sites the producer's grid is the converted input's grid and the converted output keeps it; the site-matched graph reproduces the converted operation exactly on the model's own inputs; (A') on random quantized inputs on each site's grid: matched 100% exact at the nine pools and 98.85% at the resize (100% within 1 LSB), registered 0% at the pools and 9.2% at the resize; (B) end to end on an amplified student, all-ten-matched mean \|Δlogit\| 0.0185–0.024 LSB against the registered graph's 0.052–0.060 on three images, argmax agreement 0.96–0.98 against 0.91–0.94, max ≤ 1 LSB; at random init (unamplified) site matching leaves the logits bit-identical (MEASURED during the build: the gates absorb the change); hooks add no state; training through the sites; `set_tf32` sets both flags; B1's plan, passes on synthetic batches and hand-computed reconciliation; the B1 CLI's refusals including the data root |
| smoke_explore_qat | 57 | 4 | PASS | the parent constants against DL-146, DL-139, DL-218 and the TEST-day manifest; flags and seeds per item; D's trigger (E2 s42's 0.38749… is below 0.3875); the parent check accepts a listed hash and refuses others, a wrong identity, a projection.pt; D's winner checks; **N1's loss** at step 2 equals CE + Dice + r·0.5·KL(T 4) recomputed independently, zero KD at step 1, no CWD term in parts, telemetry or run_meta, the optimizer of record; **N2** through the pinned run_qat: all ten sites firing, TF32 off in run_meta, the substitution recorded, the pinned name restored; **N5**: no fake-quant on the islands in training; in the converted model the classifiers and add run in float on float inputs after DeQuantize, every other conv (63) quantized and fed quantized tensors, float output; the registered head fails the same check; FP32 MAC share 0.0756 at 512×512 (MEASURED, conv MACs); a run_qat epoch checkpoint converts the same way; the CLI's and block B's refusals |
| smoke_explore_kd | 40 | 3 | PASS | **DIST** equals the transcribed official formulas term by term, masked and unmasked; symmetric; identical logits give 1.73e-6, equal to the reference (the official Pearson's eps); the A2 term = ramp·λ·DIST and refuses other stages; **A1''**'s gate admits exactly N4's alpha (as main() and run() call it), refuses others, and is restored; N4's record must be E3 s42's by the manifest's telemetry sha256; **futility** strict at > 2.0 pp; and **the pinned train_distill.main in --dry-run** (CPU, MockTeacher, a synthetic loader substituted for the smoke only): A2's DIST ran every iteration (RESULT PASS; its logit_kd column differs from a same-seed KL run), A1'''s off-grid alpha refused without the substitution and accepted with it (alpha_offgrid true), the futility wrapper stopped the run after the VAL row with no run_end; the CLI's refusals; the KD pin's reused modules all match at HEAD |
| smoke_explore_kl3 | 22 | 3 | PASS | a synthetic pair built as block B builds it (save_epoch_checkpoint, torchscript_bytes, translate_qat_state_to_x86); each module under its own engine; parity fields; both artifacts byte-unchanged; refusals of a wrong sha256, a swapped pair, different epochs, a wrong stage; the manifest's E6-E2 pair is DL-146's; clause 3 refuses at the current HEAD and on rows that are not RECORDED or lack TEST or the commit or name a commit outside HEAD's history; the CLI's refusals |
| **total** | **235** | **19** | **PASS 6/6** | |

## 5. What could not be built or checked here, and what the auditor should judge

Everything listed in the brief and the addendum is built. These are the limits and choices:

- (a) **Untested here for lack of data, a GPU and the teacher stack:** VAL scoring on the real VAL split (the
  adapter path of src/explore/valscore.py runs only with a data root); the CUDA-only gates and every real launch; the
  teacher of record and R2 loads (no mmseg/mmcv/mmengine in the venv); block B through the pinned scripts end to end
  (it needs a complete 15-epoch run); N5's block B; KL-3 on a real pair. The first real runs are the first tests of
  those paths; a refusal there costs nothing (nothing is written before the gates pass).
- (b) **The ten sites come from the code, not from a D-1 record.** DL-180 and everything it cites (AMENDMENTS AM-22
  item 2, AM-23 item 1, DL-127, DL-146) say only "the ten sites fake-quantized on their input's grid". No repository
  file names them (searched: the decision log, the amendments, docs/lane_reports, docs/lane_specs, reports). The
  derivation (src/explore/sites.py header) agrees with the audit's F10 inference and is MEASURED on the synthetic
  converted model (each of the ten outputs carries its input's scale and zero point; no other operation does). If the
  orchestrator's D-1 text exists outside the repository, compare it with SITE_NAMES.
- (c) **DIST is transcribed, not copied.** This session had no web access. The formulas follow the audit's S1c
  description of hunto/DIST_KD segmentation/losses/dist_kd.py and the paper. The smoke's reference was typed
  independently of src/explore/dist_loss.py, so the two agree with each other, not necessarily with the official file.
  Diff `_ref_*` in scripts/smoke_explore_kd.py against the official file before A2 launches.
- (d) **Run-time substitutions inside pinned modules** (src/explore/patching.py): N2 binds src/quant/qat.py's
  `prepare_qat_model` to a wrapper that calls it and installs the sites; A1'' binds train_distill's
  `alpha_gate_error` to one admitting exactly N4's alpha; A2 binds `distillation_losses` to DIST's; the futility flag
  binds `_jsonl` to a writer that stops after the 40k VAL row. Each pinned file is byte-unchanged (its sha256 is in
  x_run_meta.json and in run_meta's x_substitution), the binding lasts one call in one exploratory process, and the
  smokes show the restoration. The audit should rule whether this meets "imports the pinned trainers unchanged". The
  alternative is a copied trainer, which would drift from the record.
- (e) **N1 is a new loop**, not the pinned run_qat: run_qat has no loss hook, and run_qat_kd is hard-wired to E3's
  three terms and a projection, while N1 has neither. `run_qat_logit_kd` copies run_qat's lines and adds the teacher
  forward and stage e2's term through train_distill.distillation_losses. Its non-finite handling is run_qat's
  (AM-21 item 1: recorded, the run continues), not E6-KD's abort. Its stage label is E6-E2, so the registered
  block B accepts it, and its run_meta carries distillation true and the KD fields.
- (f) **The evaluator has no exploratory status.** src/eval/artifacts.py ARTIFACT_STATUSES is (official, provisional,
  smoke). The registered block B therefore writes "provisional" artifacts for C1, N1, N2 and D. The exploratory status
  lives in x_run_meta.json and x_blockb.json, and the x- run ID is in every artifact name and identity. These
  artifacts carry stage E6-E2 (E6 for D on an E3-stage winner) and must never enter the manifest (clause 2).
- (g) **D on A1''** (an E3-stage winner) runs stage E6, whose recipe is E6-E2's; only the parent loader differs.
  Its block B uses E6's rules file, configs/qat_selection_rules.json, through select_qat_epoch.py's routing by
  stage. That file states the same epoch rule (highest converted VAL, strict, ties to the earlier epoch; DL-139,
  DL-146). The audit should confirm that the two rules files agree in everything D needs.
- (h) **N5's artifact** is a TorchScript trace with an x_mixed_precision.json identity, not plantseg_int8.json. The
  evaluator of record cannot score it, and its VAL values come from the exploratory scorer only. Its FP32 MAC share,
  0.0756 at 512×512, counts conv MACs only.
- (i) **B1 on a CPU** cannot measure TF32's share and does not meet the audit's item (iv); a CUDA run on the QAT pod's
  GPU class does. The record says which it was.
- (j) **KL-3's clause-3 gate** depends on a future decision-log row that names TEST and the commit of the TEST
  results. If the records lane uses another form, the gate needs a matching change before KL-3 can run.
- (k) **Not in this lane:** B3, A1, A3, B2 (brief). The Status cells of DL-180 (and DL-181 for KL-3) are left to
  ORCH-5's records lane, because this lane edits no existing file.

## 6. Draft of AM-24, in the register of docs/PREREGISTRATION_AMENDMENTS.md

*A draft for ORCH-5's records lane. It is not an amendment, it is not in docs/PREREGISTRATION_AMENDMENTS.md, and
every value in its State is to be re-checked there.*

> ## AM-24 — Exploratory VAL-only runs of ORCH-5's EXPLORE-2 (C1, N1, N2, N5, D, N3, A1'', A2, B1, N4, KL-3) (registers no run, test, family member or derived artifact; amends nothing of record)
>
> Dated <date>. Status: DRAFT (<ruling>; the audit EXPLORE-AUDIT-1, sha256 07e4d682…, and the addendum L-AM24-ADD-1; DL-<n>).
> - **Fixed text.** From its commit this text is changed only by a new amendment, which lists the results known when
>   it is written. A dated bracket note only points to a later record or records an outcome.
> - **In force** from the time its commit reaches the remote. Approval gates nothing here.
> - **A rejection** is entered verbatim in the decision log on the day it arrives. It never reopens a run already
>   launched or a TEST evaluation.
> - **Written after results.** This amendment was written after every result its State lists and after the TEST-day
>   manifest was frozen (fe93a24), before the TEST evaluation. It adds exploratory quantities only.
>
> State at amendment. The TEST-day manifest is frozen (reports/derived/tday_manifest.json, body_sha256 6c00cc4b…,
> fe93a24); TEST has not been evaluated. Known (MEASURED, of record): E2 seeds 42/43/44 best VAL 0.38749080896377563,
> 0.3716566562652588, 0.3703170716762543 (DL-109, DL-120); E3 seed 42 0.36337557435035706 (DL-121); the arms A, G, F
> 0.3509, 0.3659, 0.1747 (DL-165); R2 0.3971906304359436 (DL-122) and E3-R2 0.3842073380947113 (DL-136); E6-E2
> 0.35132896900177 at epoch 06 (DL-146); E5, E6 and E6-KD's converted values (AM-22's State) and the eight
> fake-quant-off values (DL-176); the teacher of record 0.38577 (DL-88). D-1 was shelved (DL-180).
>
> 1. **Scope.** This amendment registers no run, no test, no family member and no derived artifact. Every item is
>    exploratory, descriptive, VAL-only (the 846-image VAL split, canvas protocol core_preprocess/1.0.0, union-present
>    all-class mIoU), and single-seed unless the item names seeds. Nothing here touches TEST, the TEST-day manifest, a
>    run or derived artifact of record, a registered selection or a registered analysis. AM-23's E6-E2 (seed 42 only)
>    is unchanged by C1.
> 2. **TEST.** No exploratory checkpoint, converted artifact, loss variant, fake-quant path or PTQ variant is scored
>    on TEST, listed in the manifest or present in the TEST session. The exploratory entry points (lane L-AM24) refuse
>    `--split test`, any split but TRAIN/VAL, and any data root that fails src/data/isolation.py's TRAIN/VAL-only
>    check, before any data is read. They run only on the TRAIN/VAL-only staged root of AM-19 item 4(c). Their records
>    carry exploratory true, split val, amendment AM-24 and artifact_status exploratory. Where a pinned script writes
>    an evaluator artifact, its status is the evaluator's non-official "provisional", and the exploratory record
>    beside it governs.
> 3. **Ordering.** (a) The TEST session, the creation of every E4, E7 and E7-E2 artifact and the Chapter 5 data work
>    take precedence. An exploratory launch or block-B scoring that would delay them is not started. (b) No B3 variant
>    and no KL-3 parity run (item 10, KL-3) runs before every E4, E7 and E7-E2 artifact exists with its sha256 in the
>    manifest and the TEST session's results are committed. (c) Exploratory pods are separate pods. The laptop runs no
>    exploratory scoring between the TEST unlock and the commit of the TEST results.
> 4. **Code.** Exploratory code lives on one branch (lane L-AM24, `lane/l-am24`, at <commit>). It adds new modules and
>    entry points only, imports the pinned trainers, quantizers and evaluator unchanged, and is merged to master only
>    after the TEST results are committed. No file the manifest or a derived-artifact pin names is modified before
>    that. Where an item's one change sits inside a pinned module's loop, the pinned attribute is bound for that one
>    call in the exploratory process only, recorded with the pinned file's sha256 (N2, A1'', A2, the futility rule).
> 5. **No feedback.** No exploratory result selects, re-selects, re-scores or amends anything of record: no epoch,
>    checkpoint, λ, α, clip, calibration list, qconfig, evaluator setting or teacher of record (N3's use of R2 argues
>    nothing about the teacher of record). The corrected fake-quant path (N2), the KD-in-QAT path (N1) and the FP32
>    head (N5) are never applied to E5, E6, E6-KD or E6-E2 of record, and no model of record is re-scored through them.
> 6. **One shot.** Each item is one run per named configuration, its recipe written here before launch. No value
>    changes after any VAL number of the item is seen. A run stopped by an infrastructure cause (host fault, OOM, pod
>    loss) is relaunched once, identical, from iteration or epoch 0 in a fresh directory; the stopped attempt's rows
>    are evidence, never a result. A run that completes, diverges or ends non-finite is the item's result and is not
>    repeated. The only early stop is the futility rule of item 10.
> 7. **Labelling and claims.** Run IDs carry the prefix `x-` and run_meta exploratory true. Chapter 5 reports them in
>    a separate section after the confirmatory results, never in a table with a Holm-family contrast, each with the
>    words "exploratory; VAL (846 images); seed 42 [or seeds]; not pre-registered; not evaluated on TEST". A
>    single-seed result may state its value, its difference from the matched run of record at the same seed, and the
>    confirmatory seed spread as context (E2 0.3703–0.3875; E3 0.3505–0.3820; E6 0.3197–0.3490). It may say a
>    hypothesis is "consistent with" or "not supported by" the run. It may not say "significantly", "outperforms",
>    "confirms", or generalize to TEST or deployment. A difference smaller than the comparator's seed spread is
>    reported as "within the observed seed spread". E2 seed 42's VAL value is cited as a post-selection value (best
>    of five λ, best of 20 checkpoints), and every INT8 value of record as the best of 15 converted epochs on VAL.
>    N5's model is called mixed precision and never INT8. D's trigger and every comparator are numbers written here.
> 8. **MVA 2027.** Exploratory runs enter the paper only as development-set ablations so labelled. A claim about a
>    method needs at least three seeds. No exploratory run is evaluated on TEST for the paper (TEST is single-use,
>    AM-19 item 9); any other held-out protocol is a matter for a new pre-registration.
> 9. **Records.** Each item's launch, end, VAL value(s), GPU-hours and the sha256 of its checkpoint or artifact are
>    entered in the decision log by the end of the day after its end, under the heading "exploratory (AM-24)", as for a
>    run of record. The entry names the comparator of record and its value.
> 10. **The items.** Each is the named recipe of record except the one named change (audit §6). Comparators: E2 s42
>    0.38749080896377563 (and 0.3765 over seeds), E3 s42 0.36337557435035706, the arms A/G/F; for QAT, E6-E2
>    0.35132896900177 (epoch 06), E6 0.3308/0.3490/0.3197, E5 0.3343/0.3192/0.3328, E6-KD 0.3321.
>    (a) **C1** — E6-E2 at seeds 43 and 44: QAT of E2 s43's and s44's best.json checkpoints of record (sha256
>        e3c4e05c… and be48da05…, DL-218), QAT seed = source seed, E6-E2's recipe (AM-23 item 1 = AM-4a item 1 +
>        clip 5.0, the clip selection of record c433145a…), block B as AM-23's (15 converted epochs on the CPU,
>        configs/qat_selection_rules_am23.json, finalize).
>    (b) **N1** — as C1 at seed 42 from E2 s42 (73dbc32d…), with λ 0.5 · L_LogitKD (T 4, OS8, the validity mask of
>        record) added to the QAT loss, ramped linearly over the first epoch (E6-KD's ramp), the teacher of record one
>        frozen forward per step on its M4-KD stream (E6-KD's stream); no CWD term and no projection.
>    (c) **N2** — as C1 at seed 42 from E2 s42, with a fake-quant sharing its input's scale and zero point after each
>        of the ten sites (the eight SE pools, the LR-ASPP pool, the head's OS8 resize: src/explore/sites.py) and TF32
>        off for cuDNN and matmul.
>    (d) **N5** — as C1 at seed 42 from E2 s42, with the LR-ASPP logit head (both 1×1 classifiers, their add and the
>        final upsample) in float: no fake-quant there in training, a dequant before it in the converted model. A
>        mixed-precision artifact (FP32 conv MAC share 0.0756 at 512×512); its block B converts, scores and selects as
>        AM-23's rule, on the FP32-head skeleton.
>    (e) **D** — E6-E2's recipe of record (not the corrected one) on the exploratory FP32 winner, only if its best VAL
>        all-class mIoU is at least 0.3875 (Decimal of the value's repr), at its seed; if two qualify, the higher one
>        only, and the record says so.
>    (f) **N3** — E2 seed 42's recipe with arm teacher R2 at its M12 iteration 24000 (iter_24000.pth, sha256 d5085961…,
>        DL-122) in place of the teacher of record, loaded with the record's teacher configuration.
>    (g) **A1''** — E3 seed 42's recipe with α = λ · mean(L_logitKD) / mean(L_feat) over E3 seed 42's last 10,000
>        iterations, read from N4's record before launch (recorded, never chosen).
>    (h) **A2** — E2 seed 42's recipe with DIST (β = γ = 1, τ = 1, rows (B·H·W, C) on the OS8 grid within the validity
>        mask, weight 1 against CE) in place of λ · KL only. A null result is the expected outcome under a parity
>        teacher.
>    (i) **Futility** (optional, N3, A1'' and A2): at iteration 40,000, if the run's VAL is more than 2.0 pp below the
>        comparator's logged VAL at 40,000 (E2 seed 42's telemetry for N3 and A2, E3 seed 42's for A1''), stop and
>        record "stopped: futility rule, 40k VAL v vs comparator u". Its rows are evidence, not a result, and it is not
>        a comparator for anything.
>    (j) **B1** — for each QAT model's selected epoch, VAL through the fake-quant graph with each site in turn on its
>        input's grid, all ten, TF32 off, all ten with TF32 off, reconciled against the converted artifact re-scored
>        by the same scorer (interactions reported, the SE-pool sites first). On the hardware class of the recorded
>        fake-quant VAL for item (iv).
>    (k) **N4** — from existing records only: E3 seed 42's per-term magnitudes (and the A1'' α), and for the eight QAT
>        models parent → fake-quant (selected epoch) → converted → fake-quant-off.
>    (l) **KL-3** — for each of the seven x86 latency copies (E5 and E6 seeds 42–44, E6-E2 seed 42), VAL of the copy
>        against its QNNPACK artifact of record on the CPU, both read only; after the TEST results are committed
>        (item 3(b)).
> 11. **Disclosure.** Chapter 4 states that this amendment was written after the results its State lists and after
>    the manifest was frozen, and that its items select nothing of record.

## 7. Verification, commits and push

- Every commit was made with `git commit --only -F <message> -- <explicit paths>`, after the smokes and the scoped
  status and diff. No commit message carries a trailer.
- Commits, oldest first:

  | commit | contents |
  |---|---|
  | 05d1db9 | E0: the guard and its smoke |
  | 9ef3b2d | E1 (N4): decompositions and their smoke |
  | 2c6c3b3 | E2 (B1) and N2's sites: sites, B1, the VAL scorer, synthetic students, their smoke |
  | 743ef40 | E3: the exploratory QAT entry (C1, N1, N2, N5, D), block B, their smoke |
  | 906beb7 | E4: the exploratory KD entry (N3, A1'', A2, futility), their smoke |
  | 99191f6 | E5: KL-3 parity and its smoke |
  | (this commit) | the smoke runner and this report |

- The scoped diff at the last code commit (`git diff --no-renames --name-status fe93a24..HEAD -- .
  ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*' | grep -vi test`):
  26 lines, every one `A`; this commit adds two more A lines (scripts/smoke_explore_all.py, this report).
- Push: `git push -u origin lane/l-am24`, fast-forward, the only push of the lane.
