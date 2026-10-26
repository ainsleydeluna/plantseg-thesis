# KD arm E3-R2 (AM-18 item 3): a host failure stopped attempt 1 (AM-8a fault report)

Written 2026-10-20 by the orchestrator (rulings HOST-FAULT-1, ARMS-STOP-1 and W4R-AUDIT-1), before the repeat.
Labels: MEASURED (a command's output that Ice De Luna pasted, or a file of a keep read by sha256), DOCUMENTED (a screenshot
of the RunPod console, or the ledger), REPOSITORY-PROVEN (read in the code at the arms pin P_K or in a file committed
at 3373139), INFERRED. Times are +08:00 unless they end in Z.

## 1. What failed

- **Run:** e3-r2_s42_lambda0.5_alpha25_80k_a1, E3-R2 (AM-18 item 3(a)): E3's recipe of record distilled from R2's X, iter_24000.pth (sha256
  d5085961e1529d83b1fc13ff04fac83957bcc44c4f70a0635ff0d840083ffab9; DL-122; go rule DL-125), seed 42, λ 0.5, α 25, 80,000 iterations, attempt 1. Code pin P_K
  572ab226aff8f9ef63cbf8d1fe212187afafdd67 (DL-115); records commit c3341eb6e2a8d88c2ebeb0d988af6c1b707ae3ea.
- **Its launch (reports/kd_launch_log.jsonl at 3373139, lines 39 and 43; REPOSITORY-PROVEN):**
  launched Mon 19 Oct 23:12:48 (launch_time_utc 1792422768.4012723, 2026-10-19T15:12:48.401272Z); run_meta sha256 6b1627981683a403190c880bfabb1bd421212d4a51aef2980bd3b50ac330fb37; launched line
  sha256 ce0aac17a9ba1c4f272697f3993a04e3182a38e461f057e4ac34fd1551ebaa7d. The arm gate printed VERDICT: GO and the pod kit printed RESULT: LAUNCHED (DOCUMENTED, the arms
  session's report); the kit prints that line only after check-run-meta passes (kd_pod_arms_20261019.sh, d503be97...).
- **The pod:** kd-e3arm-r2 (RunPod id 7pasvhyp2q7ssd), Secure Cloud, On-Demand, 1 × NVIDIA A40, data centre CA-MTL-1, the
  plantseg-kd template (DOCUMENTED, ledger). The trainer's pid file names pid 8156.
- **The RunPod console (DOCUMENTED):** Ice's screenshot at 09:51 Tue 20 Oct showed kd-e3arm-r2 with an Alert badge and GPU 0 %, while kd-e3arm-w showed GPU 99 % (ruling PODS-ALERT-1).
- **The provider's Alert (DOCUMENTED, Ice's screenshot of the RunPod console at 09:52 Tue 20 Oct, the text read on kd-e3arm-r2 itself (rulings PODS-ALERT-1 and HOST-FAULT-1)):** "We have detected a critical error on this machine which may affect some pods. We are looking into the root cause and apologize for any inconvenience. We would recommend backing up your data and creating a new pod in the meantime."
- **The run's end, as found (MEASURED, read-only, check 1 at 2026-10-20T04:27:11Z):**
  - The stdout log (stdout and stderr) was last written at 2026-10-19 20:40:56.485727765 +0000 (Tue 20 Oct 04:40:56 +08:00), 42,655 bytes. It ends with
    `[iter 15600/80000] loss=7.5146 sup=0.7950 logit_kd=1.9636 cwd_feat=0.0985 cwd_logit=1.0919 ramp=1.0000 lr=8.22652255e-03`, and it holds 0 Traceback lines.
  - The telemetry (e3_telemetry.jsonl, sha256 b69a493c0baece61caeeba2cd40d245c70267f9af708c7f92a3e0da423444457, 7,879,861 B) holds 15,635 rows:
    15,632 train rows and 3 VAL rows; run_abort 0, run_end 0, rows with a
    `nonfinite` map 0; a torn last line: no. Its last row is a train row at iteration 15,632, written at
    wall_clock 1792442464.0348275 (2026-10-19T20:41:04Z; Tue 20 Oct 04:41:04).
  - The run directory /workspace/kd/e3-r2_s42_lambda0.5_alpha25_80k_a1 holds best.json, e3_run_meta.jsonl, e3_telemetry.jsonl, projection.pt (411,708 B) and 3 best checkpoints, at iterations 4,000, 8,000 and 12,000 (check 1's listing).
  - best.json: e3_student_best_iter12000.pt, best_val_miou_all_class 0.22997714579105377 (written 2026-10-19 19:25:01Z).
  - The volume: mounted from ca-mtl-1-vast.runpod.net:/runpodfs/podvolumes/t3s7iufjr6vy/7pasvhyp2q7ssd, 19 G, 2.6 G used (14 %), so not full.

## 2. Cause: the provider reported a critical error on the pod's machine

- **MEASURED (read-only, check 2 at 2026-10-20T04:27:20Z):**
  - `/proc/uptime` read 27672.88 s, so the machine booted at about Tue 20 Oct 04:46:07 (2026-10-19T20:46:07Z).
  - The container's PID 1: not read: the image has no ps (bash: ps: command not found), and no read of /proc/1 was ruled for this pod.
  - The trainer's process is absent (pid 8156).
  - The memory cgroup: cgroup v1: memory.oom_control oom_kill_disable 0, under_oom 0, oom_kill 0; memory.failcnt 0; memory.max_usage_in_bytes 47,284,224 B against memory.limit_in_bytes 49,999,998,976 B (INFERRED: these counters cover only the time since the container restarted after the boot, so they say nothing about what stopped the run).
  - MemTotal 527,972,732 kB (MemAvailable 497,781,768 kB); 96 CPUs.
  - The GPU: NVIDIA A40, ECC enabled, 0 uncorrected errors, volatile and aggregate (the volatile count resets at a reboot); retired pages pending [N/A].
  - `dmesg`: refused in the container (dmesg: read kernel buffer failed: Operation not permitted).
- **DOCUMENTED:** the provider's Alert above names a critical error on the machine and recommends backing up the data
  and creating a new pod (ruling HOST-FAULT-1). It is the documented host failure that AM-8a requires before a repeat.
- **INFERRED:** the machine under kd-e3arm-r2 failed between the telemetry's last write (20:41:04Z on 19 Oct) and its boot (about 20:46:07Z), which is the critical error the provider's Alert reports, and that failure stopped the trainer. The trainer wrote no Python error, no abort row and no non-finite row; its process is gone and was not restarted. No written rule and no operator stopped the run. It was a different machine from kd-e3arm-r1's and kd-w4-e3l's (their hosts booted at 19:05:20Z and 21:57:48Z, and their MemTotal values differ); all three pods were in CA-MTL-1 and showed the same Alert. What failed inside the machine cannot be seen from the pod (dmesg was refused).
- The run cannot be resumed: the runbook's launch block never passes `--resume` (REPOSITORY-PROVEN, KD.9), and AM-8a
  repeats the run.

## 3. How the rules apply

- **AM-8a:** "A run is repeated with the same seed only after a documented infrastructure fault (crash, host or pod
  failure, corrupted artifact, wrong configuration)." As amended by AM-19 item 3(a), a run that stops in any other way
  than under a written rule (operator, pod, host) "is repeated under AM-8a from the first iteration, in a new run
  directory, on the first card item 1(e) gives it", after this report, which states who or what stopped the run and
  the VAL rows written by then, by the run and by its comparator (§4).
- **The runbook, KD.11:** "Killed, or the pod lost: AM-8a report, then relaunch on the first card."
- **Not AM-7 and not an abort:** no run_abort row exists, and no row carries a `nonfinite` map (§1). So this is
  neither a divergence (AM-18 item 3(e); ruling R4(a): an arm that diverged is never rerun) nor KD.11's "non-finite
  row with no abort record". The arm gate reads attempt 1's directory with --previous-run and refuses a repeat after a
  divergence (REPOSITORY-PROVEN, scripts/preflight_arms.py at P_K, check_previous_runs).
- **AM-22 item 2(b):** the repeat is not a launch under item 2(a); it launches on the first card AM-19 item 1(e) gives
  it, subject to AM-19 item 3(b)'s arithmetic. A repeat of a cuttable run that has not launched by the end of the
  third day after the day of the last row the attempt before it wrote is cut on that date. Attempt 1's last row is on
  2026-10-20 (Asia/Manila; AM-19 Definitions: a date ends at 24:00 Asia/Manila, 16:00:00 UTC), so the repeat is
  cut if it has not launched by the end of 2026-10-23 (2026-10-23T16:00:00Z). The repeat's pod script and launch-log
  block compute the same date from the same file and refuse after it.
- **The repeat's precondition** (KD.11, "Before any repeat confirm that the attempt before it has ended") holds: the
  trainer's process is absent, and its last row is earlier than any launch of the repeat.

## 4. The VAL rows written by then (AM-19 item 3(a))

- **By the run** (MEASURED, from the kept telemetry, sha256 b69a493c0baece61caeeba2cd40d245c70267f9af708c7f92a3e0da423444457): 3 VAL rows.

| VAL | iteration | all-class mIoU | written (+08:00) |
|---|---|---|---|
| 1 | 4,000 | 0.11340745538473129 | Tue 20 Oct 00:36:57 |
| 2 | 8,000 | 0.20404556393623352 | Tue 20 Oct 02:01:00 |
| 3 | 12,000 | 0.22997714579105377 | Tue 20 Oct 03:25:01 |

- **By its comparator.** AM-19 item 3(a) asks for "the VAL rows written by then by the run and by the other runs of
  its sweep, or by its comparator" (docs/PREREGISTRATION_AMENDMENTS.md at 3373139, lines 1593–1594). An arm differs
  from E3 only in what its definition names (AM-18 item 3), so its comparator is E3 seed 42 at λ 0.5 and α 25,
  e3_s42_lambda0.5_alpha25_80k_a1 (the α selection's winner of record).
  - **Its identity (REPOSITORY-PROVEN at 3373139):** its launch and launched lines are reports/kd_launch_log.jsonl
    lines 13 and 17 (launched Mon 12 Oct 17:03:56, 2026-10-12T09:03:56Z). Its run_meta sha256,
    ab73e8279bbb2796b4deb5973c9e10f59192254f2c6ada2a3e9bab48a2edd6a3, is the launched line's and the one
    reports/derived/alpha_selection.json (sha256 1399b897278db17a8876c724b89c8d8e7a7b42ffe1d9c4d67002578a889f2dd8)
    lists for it (inputs[0]). Its e3_run_meta.jsonl, read from its keep, hashes to that value (MEASURED).
  - **It ended before the arms launched** (its run_end row, Tue 13 Oct 21:06:56, 2026-10-13T13:06:56Z), so every VAL row it wrote was written
    by then.
  - **Its kept telemetry** (MEASURED: kd/e3_s42_lambda0.5_alpha25_80k_a1/e3_telemetry.jsonl, 40,523,892 B,
    sha256 734417e4b33e64239cebe2485e3eace03a795163b47bf50667517be40c161d2e, equal to its keep manifest's line) holds 80,021 rows, among them
    20 VAL rows and one run_end row, with no run_abort row and no `nonfinite` map. Its VAL rows, in the
    order written:

| VAL | iteration | all-class mIoU | written (+08:00) |
|---|---|---|---|
| 1 | 4,000 | 0.08035974949598312 | Mon 12 Oct 18:28:12 |
| 2 | 8,000 | 0.1981126070022583 | Mon 12 Oct 19:52:16 |
| 3 | 12,000 | 0.24441805481910706 | Mon 12 Oct 21:16:27 |
| 4 | 16,000 | 0.23171493411064148 | Mon 12 Oct 22:40:41 |
| 5 | 20,000 | 0.2613939046859741 | Tue 13 Oct 00:04:58 |
| 6 | 24,000 | 0.2735271155834198 | Tue 13 Oct 01:29:18 |
| 7 | 28,000 | 0.3077707290649414 | Tue 13 Oct 02:53:48 |
| 8 | 32,000 | 0.3111237585544586 | Tue 13 Oct 04:18:18 |
| 9 | 36,000 | 0.33465683460235596 | Tue 13 Oct 05:42:34 |
| 10 | 40,000 | 0.30759426951408386 | Tue 13 Oct 07:06:34 |
| 11 | 44,000 | 0.3311440050601959 | Tue 13 Oct 08:30:40 |
| 12 | 48,000 | 0.3530866205692291 | Tue 13 Oct 09:54:49 |
| 13 | 52,000 | 0.35441237688064575 | Tue 13 Oct 11:18:52 |
| 14 | 56,000 | 0.34778642654418945 | Tue 13 Oct 12:42:54 |
| 15 | 60,000 | 0.35122615098953247 | Tue 13 Oct 14:06:56 |
| 16 | 64,000 | 0.3444356620311737 | Tue 13 Oct 15:30:55 |
| 17 | 68,000 | 0.35341838002204895 | Tue 13 Oct 16:54:54 |
| 18 | 72,000 | 0.3604889214038849 | Tue 13 Oct 18:18:56 |
| 19 | 76,000 | 0.36337557435035706 | Tue 13 Oct 19:42:57 |
| 20 | 80,000 | 0.35811886191368103 | Tue 13 Oct 21:06:56 |

- **The comparator's best VAL:** 0.36337557435035706 at iteration 76,000 (VAL row 19 of 20, written Tue 13 Oct 19:42:57; the earliest row on a tie). It equals reports/derived/alpha_selection.json's best_val for its
  winner, 0.36337557435035706 at e3_student_best_iter76000.pt (REPOSITORY-PROVEN at 3373139). The comparator's rows
  are given for comparison only.
- Attempt 1's VAL rows are evidence of the stop only; the repeat replaces the run and no selection is made between the
  attempts (AM-8a). If the repeat is cut under AM-22 item 2(b), these rows are reported under AM-19 item 3(c).

## 5. What is kept

- AM-8a: "The failed run's logs and partial checkpoints are kept." Everything of attempt 1 on kd-e3arm-r2 was copied to
  the laptop before any repeat work (ARMS-STOP-1): the whole run directory (every checkpoint, the telemetry, run_meta,
  best.json, projection.pt where present), the whole evidence folder /workspace/ev_e3-r2_s42_lambda0.5_alpha25_80k_a1, the records folder,
  kd_env.sh and the operator's op_*.out files.
- They are in one keep tar, keep_e3-r2_s42_lambda0.5_alpha25_80k_a1.tar (MEASURED): sha256 8131ba03e2cd3dc9c3ddba54496d3c67a4f6cd0f9157b20d13588ae06e45befb, 80670720 B, 43
  members, with a sha256 manifest of 42 files; on the laptop every manifest line checks (KEPT-OK,
  42 lines). The manifest lists e3_run_meta.jsonl with sha256 6b1627981683a403190c880bfabb1bd421212d4a51aef2980bd3b50ac330fb37 (the launched line's run_meta_sha256)
  and e3_telemetry.jsonl with sha256 b69a493c0baece61caeeba2cd40d245c70267f9af708c7f92a3e0da423444457.
- The run's durable evidence folder is C:\Users\admin\plantseg_runs\kd_arms_20261019\fault_e3-r2_s42_lambda0.5_alpha25_80k_a1. It holds the
  keep tar, its extracted files and this report. Its SHA256SUMS lists this report and the keep tar, and it is
  written before the repeat launches (AM-8a).
- kd-e3arm-r2 is terminated only after the copy is checked and the orchestrator confirms.

## 6. The repeat

- **One repeat: attempt 2, e3-r2_s42_lambda0.5_alpha25_80k_a2.** It has the same arm, stage, seed, λ, α, horizon, code pin P_K and arguments, a
  fresh run directory /workspace/kd/e3-r2_s42_lambda0.5_alpha25_80k_a2, and a new pod, kd-e3arm-r2b (the first free card; KD.11). It starts from
  iteration 0. The same inputs tar as attempt 1 carries R2's X and AM-19 item 3(b)'s references.
- **Its launch line names this report** (`am8a_report`, path and sha256; KD.8), and the launch log records attempt
  1's stop in a `stopped` line that names this report (AM-19a reading 6). The arm gate takes `--attempt 2`,
  `--am8a-report` and `--previous-run` (attempt 1's run_meta and telemetry from the keep).
- **AM-19 item 3(b):** the gate computes the arithmetic at the launch: 80,000 iterations times the larger of the
  references' medians against the time left to the original freeze date 2026-11-03 (end 2026-11-03T16:00:00Z). With
  the references of attempt 1's gate (1.2698394060134888 s/iter, about 101,600 s) it fits by far (INFERRED until the
  gate prints it).
- **The repeat replaces the failed run.** No selection is made between them, and attempt 1's partial results are
  never used as a result (AM-8a), unless the repeat is cut under AM-22 item 2(b), in which case its VAL rows (§4) are
  reported under AM-19 item 3(c).
- **If attempt 2 fails,** whatever the cause, it is a STOP for the orchestrator before anything else runs.

## 7. Timeline

| Time (+08:00) | Event | Label |
|---|---|---|
| Mon 19 Oct 23:12:48 | attempt 1 launched on kd-e3arm-r2 | REPOSITORY-PROVEN, launch log |
| Tue 20 Oct 03:25:01 | the last VAL row (iteration 12,000) | MEASURED, telemetry |
| Tue 20 Oct 04:41:04 | the telemetry's last row (train, iteration 15632) | MEASURED, telemetry |
| Tue 20 Oct 04:46:07 | the machine's boot, by /proc/uptime | MEASURED |
| Tue 20 Oct 09:51:00 | Alert and 0 % GPU in the RunPod console | DOCUMENTED, screenshot |
| Tue 20 Oct 12:27:11 | check 1 (read-only) | MEASURED |
| Tue 20 Oct 12:27:20 | check 2 (read-only) | MEASURED |
| Fri 23 Oct 24:00 | AM-22 item 2(b): the repeat is cut if it has not launched by now | REPOSITORY-PROVEN rule |
