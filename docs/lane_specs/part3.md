Source: Fable 5.1 lane-spec audit, 2026-09-28 (part 3: L-AM17-L, the SegNeXt-L descriptive arm).
Authoritative acceptance criteria. Read with part1.md, part2.md and errata.md (E-5 blob check, E-7
smoke-script convention apply). Authority: AM-17 item 11 (with the 11(a)/11(b) texts revised at
CP-007e), DL-30, DL-32 and the two decision-log rows recorded with this file.

## Cross-lane rules (part 3)

- No L pod before adviser approval of AM-17 and this file's commit (DL-32). The fine-tune phase
  (lanes 1, 2, 4) depends on no K- or Q-track lane; lane 3's K-part merges with K-track lanes 1+2 and
  gates both W1's first KD run and L's R3; lane 5 follows K-track 1+2 → 4 → 3 → 5 and Q-track 8; lane 6
  follows part-1 lanes 2 and 3.
- The nine frozen teacher runtime files stay frozen. L is reached by a new config file and a new
  launcher wrapper that imports launch_teacher_finetune unchanged; the two architecture-dependent
  runtime files are re-hashed under the L pin and the hashes recorded in the launch record.
- Every L artifact carries `arm` ("L-teacher", "E2-L", "E3-L", "E7-L"), `teacher_arch: "segnext_mscan-l"`,
  `teacher_config_sha256` and `teacher_ckpt_sha256`; every B-teacher artifact carries the same
  fields with "segnext_mscan-b". Nothing L-tagged is a parent of any run of record (11(e)).
- All L outputs are `artifact_status: provisional`; TEST day writes `official`.

## 1. L-TEACHER-CFG-L — the L configuration

(a) AM-17 11(a). New configs/teacher/segnext_mscan-l_1xb16-adamw-40k_plantseg116-512x512.py: the B
    file with `_base_` = mmseg::segnext/segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512.py and the
    init identity replaced (config name, filename …19b14b63.pth, EXPECTED_ADE20K_SHA256 = the
    recorded L sha, classifier_only_mismatch unchanged). Every other statement is byte-identical to
    the B file (same M2/M3 transforms, optimizer, schedule with end = MAX_ITERS, hooks, M4 seed,
    randomness, provenance markers; PROTOCOL_CLASSIFICATION 'thesis-derived';
    PUBLISHED_MSCAN_L_REFERENCE = dict(miou=44.52, macc=59.95, params_m=49,
    role='contextual-reference-only')).
(b) configs/teacher/ (new file), scripts/smoke_teacher_config.py (extended to take a config path);
    docs/teacher_init_source.md (L row). Governed: configs, scripts.
(c) Out: the config; reports/derived/teacher_config_diff_L_vs_B.json (mmengine-resolved diff).
(d) d1: the resolved-config diff L vs B (mmengine Config.fromfile, both files, dumped and diffed) is
    exactly {backbone.depths, backbone.drop_path_rate, decode_head.channels, decode_head.ham_channels,
    load_from, ADE20K_INIT_IDENTITY.*, EXPECTED_ADE20K_SHA256, PUBLISHED_*_REFERENCE}; embed_dims,
    num_classes 116, all pipelines, optimizer, param_scheduler, hooks, randomness identical. d2: the
    stock L ADE20K checkpoint loads under TeacherInitCompatibilityHook's rule: missing 0, unexpected 0,
    shape mismatches exactly {decode_head.conv_seg.weight, decode_head.conv_seg.bias} (150 → 116);
    parameter count recorded (expected ≈ 48.9 M backbone+head at 150 classes per Guo et al. 2022
    Table 2; the 116-class count is the recorded number). d3: the source-text diff of the two config
    files touches only the lines named in (a).
(e) No dependency. First L lane.
(f) STOP: d1 shows any field outside the named set.

## 2. L-TEACHER-LAUNCH-L — launcher wrapper, pod chain, memory record

(a) AM-17 11(a). scripts/launch_teacher_finetune_l.py imports launch_teacher_finetune unchanged and
    overrides only DEFAULT_CONFIG, ADE20K_INIT_CONFIG, ADE20K_INIT_FILENAME, EXPECTED_ADE20K_SHA256
    (L values); all gates, the CUDA probe, the frozen-runtime-file check and the launch record are
    reused. The pod chain is the B66 harness pattern (step 0, uploads with sha, gate, GO-file hold,
    watcher, watchdog, in-chain R3 (lane 4), hashes, bundle, pre-stop retrieval hold, self-stop) on an
    A100 PCIe 80GB Secure pod with the pinned teacher image. The first-iteration peak memory
    (torch.cuda.max_memory_allocated and reserved) is written to the launch record; the batch size is
    never reduced.
(b) scripts/ (new wrapper), reports/ (runbook and records). Governed: scripts.
(c) Out: launch record with config sha, init sha, L pin, gpu_name "NVIDIA A100 80GB PCIe", driver,
    peak memory; the ten 4k checkpoints + last + the M12 selection record; telemetry.
(d) d1: the wrapper refuses the B checkpoint (sha mismatch) and the B config (name check) and
    accepts the L pair; the B launcher still refuses the L checkpoint. d2: the frozen-runtime check
    passes with the two architecture-dependent files re-hashed and the seven others unchanged. d3
    (pod, first minute): gpu_name exact, peak memory recorded, loss finite at iteration 50; if peak
    reserved > 76 GiB → STOP (projection wrong by > 50%). d4: chain tests from the B66 harness reused
    (hold timeout, stall kill, pre-stop hold).
(e) After lane 1; after DL-32's gate. Cost: plan's 15–20 h, ≈ $21–28, pessimistic $38 (INFERRED).
(f) STOP: any edit to launch_teacher_finetune.py or the frozen runtime files; a smaller batch.

## 3. L-CKPT-GUARD — strict teacher load and E7 provenance

(a) DL-30's "no silent teacher substitution" made mechanical. K-part: build_segnext_teacher gains a
    strict load: after init_model, the model's state_dict keys and shapes must equal the checkpoint's
    (missing 0, unexpected 0, no shape mismatch) — mmseg's strict=False warnings are turned into a
    TeacherCheckpointInvalid; an architecture signature (stage depths from backbone.block{1..4}.*
    key counts, embed dims from the first conv of each stage, head channels from decode_head.*)
    is inferred from the state_dict and must equal the config's; `--teacher-ckpt-sha256` is required
    for --real-run in train_distill and evaluate_model and must equal the file's sha; run_meta and
    the evaluator summary record teacher_arch, teacher_config_sha256, teacher_ckpt_sha256. Q-part:
    run_e7.py requires the source checkpoint's run_meta to show stage E3 and teacher_arch
    segnext_mscan-b unless `--arm E7-L` is given (then E3-L and segnext_mscan-l), writes to
    outputs/E7-L/ for the L arm, and refuses a non-empty out-dir. The same non-empty-out-dir refusal
    is added to run_e4/e5/e6.
(b) src/distill/segnext_teacher.py, src/training/train_distill.py, scripts/evaluate_model.py,
    scripts/teacher_readiness_r3.py, src/quant runners, scripts/run_e{4,5,6,7}.py. Governed.
    metrics.py untouched.
(c) New run_meta/summary fields as in (a); E7-L out-dir.
(d) d1: L checkpoint into B config → TeacherCheckpointInvalid naming the first unexpected key and
    the head shape mismatch; B into L likewise; B into B and L into L pass. d2: signature inference
    on both real checkpoints returns [3,3,12,3]/512 and [3,5,27,3]/1024. d3: sha mismatch refused;
    missing --teacher-ckpt-sha256 refused for --real-run, allowed for smokes. d4 invariance: with the
    B pair, the part-2 lane-1 harness output is byte-identical before and after this lane (the guard
    adds no forward call and no RNG consumption). d5: run_e7 refuses an E3-L checkpoint without
    --arm E7-L, refuses a non-empty out-dir, and writes E7-L artifacts to the L out-dir with arm
    "E7-L". d6: the evaluator summary of a B-teacher VAL re-score carries the three new fields and
    the all-class value is unchanged bitwise (regression on the R3 checkpoint, CPU).
(e) K-part merges with K-track lanes 1+2 and must land before W1's first KD run and before L's R3
    (lane 4); Q-part before E7 s42 (W3).
(f) STOP: d4 differs; any guard implemented as a warning.

## 4. L-R3-L — readiness, go/no-go, decision-log entry

(a) AM-17 11(b) as revised; the go/no-go decision-log row exists before this lane runs. Steps, on one
    host (the L pod's CPU in-chain, or the laptop after retrieval; recorded): (1) re-score the teacher
    of record (8c0e649a…) with teacher_readiness_r3.py at the L-lane pin → Δ from 0.38576993346214294
    recorded, |Δ| ≤ 1e-4 else STOP; (2) run R1–R4 verbatim for L with the L config (best-VAL
    checkpoint per M12); (3) write reports/derived/segnext_l_go_nogo.json = {l_r3, b_constant,
    b_rescore, b_delta, l_in_training_best (recorded only), host, evaluator_commit, rule_trace,
    go: bool, tie_case: bool}; (4) decision-log entry. Also: L's per-class VAL IoU is scored into the
    part-1 lane 4 table as a third column (descriptive).
(b) scripts/teacher_readiness_r3.py (config parameter, no rule text change), scripts/ (go/no-go
    script). Governed: scripts.
(c) The JSON above; the L R3 artifact (4 files) with arm "L-teacher".
(d) d1 unit: rule_trace on synthetic values covers go, no-go, the tie case (both outcomes), and the
    STOP on |b_delta| > 1e-4. d2 real: the B re-score's per-class arrays equal the R3 bundle's within
    1e-6 (same evaluator, different host); the L artifact has 846 rows and n_eligible recorded.
(e) After lanes 2 and 3 (K-part); before any E2-L launch.
(f) STOP: any L score read before step (1) passes; any use of the in-training value in the decision.

## 5. L-KD-L — E2-L, E3-L, E7-L

(a) AM-17 11(c). Launch blocks from the distill gate profile with `--teacher-config <L config>
    --teacher-ckpt <L best> --teacher-ckpt-sha256 <sha> --arm E2-L|E3-L`, λ from
    lambda_selection.json (E2-L, E3-L) and α from alpha_selection.json (E3-L); β = 3, T = 4; seed 42;
    80,000 iterations; the same switches as E2/E3 (part-2 lane 1); out-dirs outputs/E2-L, outputs/E3-L.
    Before E2-L: the A40 KD smoke (part-2 lane 5's script with the L teacher, stages E2 and E3, 25
    steps) must show peak reserved ≤ 40 GB and finite losses; else the KD arm is not run and this is
    recorded. E7-L: run_e7.py --arm E7-L from E3-L's best checkpoint with the AM-10 list (calibration
    pass only). No E4/E5/E6-L, no seeds 43/44, no robustness scoring.
(b) configs/distill.py (arm entries), preflight profile, scripts/smoke_kd_step.py (teacher
    argument), run_e7.py (lane 3 Q-part). Governed.
(c) run_meta: arm, terms, lambda/alpha/beta/T, selection sha256s, teacher fields; best.json per arm.
(d) d1: the gate refuses E2-L/E3-L without the L teacher triple or without the selection files;
    refuses seeds ≠ 42; refuses an out-dir under outputs/E2 or outputs/E3. d2: with the B triple and
    arm "E2", the launch block is byte-identical to the run-of-record block (the arm machinery does
    not touch B launches). d3 (W3 pod): the L KD smoke JSON with peak, s/iter, teacher-forward share,
    shapes [16,116,64,64] and [16,320,32,32]. d4: E7-L artifacts carry arm "E7-L" and the E3-L
    parent's sha.
(e) After K-track 1+2 → 4 → 3 → 5, Q-track 8 (for E7-L), lane 4's go = true, and the λ (E2-L) and α
    (E3-L, E7-L) selections. Cut first (AM-17 item 9).
(f) STOP: an L arm launched with go ≠ true; any L-tagged checkpoint consumed by a run-of-record
    launcher (lane 3 guards).

## 6. L-REPORT-L — descriptive reporting

(a) AM-17 11(d), 11(f). One report section, produced by the part-1 stats lanes: VAL and TEST
    dataset-level mIoU (union-present, GT-present, disease-only) for L-teacher, E2-L, E3-L, E7-L and
    their B counterparts; per-image mean ΔmIoU with BCa 95% CIs (B = 10,000, seed 42) for E2-L − E2,
    E3-L − E3, E7-L − E7; the upstream-protocol TEST score for L (part-1 lane 2 with the L config)
    against 44.52% with the item 10(f) caveats; the 11(f) reading rendered as text from the Holm
    results and the E3-L − E3 / E2-L − E2 values, with no other consequence.
(b) src/stats/ report module; no evaluator change.
(c) reports/derived/segnext_l_arm_<UTC>.json and the Ch4 table.
(d) d1: synthetic inputs → the three contrasts, CIs containing points, the 11(f) text for each of
    the four cases (H1a/H1b rejected or not × L gain ≥ 1.0 pp or not). d2: the report refuses to run
    before the TEST manifest is frozen for TEST values and runs on VAL values earlier.
(e) After part-1 lanes 2 and 3; TEST values on TEST day only.
(f) STOP: any path by which this report feeds a selection.

## Order relative to the K and Q tracks

Fine-tune phase: lanes 1 → 2 (cloud-writable: config, wrapper, smokes on synthetic configs; the
init-load test needs the L ADE20K checkpoint locally) and lane 3 (with K-track 1+2), then the L pod.
KD phase: lane 4 (end of the fine-tune) → lane 5 after K-track 5 and Q-track 8 → lane 6 on TEST day.
Nothing in part 3 gates the λ sweep except lane 3's K-part, which is a one-day lane merged with
the K-track pair.
