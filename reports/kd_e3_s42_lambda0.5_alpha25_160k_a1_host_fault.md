# KD wave 4, E3 seed 42 at 160k: attempt 1 stopped before its end (AM-8a fault report)

Written Tue 20 Oct 2026, 13:13 +08:00 by the orchestrator (rulings E3L-STOP-1, E3L-STOP-2, W4R-1 and W4R-AUDIT-1), before the
repeat. Labels: MEASURED (a command's output that Ice De Luna pasted, or a file read from the run's fault keep or from
its comparator's keep), DOCUMENTED (his screenshots of the RunPod console and his notes from it), REPOSITORY-PROVEN
(read in the code at the KD code pin 8487551, or in the amendments, runbook, launch log or derived records at master),
INFERRED. Times are +08:00 unless they end in Z.

## 1. What stopped

- **Run:** e3_s42_lambda0.5_alpha25_160k_a1, stage E3, seed 42, λ = 0.5, α = 25, 160,000 iterations, attempt 1
  (AM-16 item 3's longer-schedule E3; wave 4). Code pin 8487551ca67bd50ffbe911d1f3e93244841cbc9f; records commit
  1dad9550bbeb82eb6a87d1e7c20f36d030afe0ef.
- **Its launch (reports/kd_launch_log.jsonl at 3373139179e6e27ecbe25f280be7b2b7889c38f2, lines 27 and 31):** launched
  2026-10-17 23:50:26 (launch_time_utc 1792252226.4753544); run_meta sha256
  590f6e872a4c8288f0305c4b36a0d555cb5ae80f3e83c40cfc7fc1e31be7e1b5; launched line sha256
  6bcb869ed016ed517abaff9f409eae71a02edd485e6e4fa604ee2838fe38e931. The gate printed VERDICT: GO with every row PASS,
  and check-run-meta passed (the wave's launch report). Its daily copies on 18 and 19 Oct showed it running, with no
  abort and no non-finite row.
- **The pod:** kd-w4-e3l (RunPod id m0z26zjgpaa93q), Secure Cloud, On-Demand, 1 × NVIDIA A40, the plantseg-kd
  template. Trainer pid 20271. On Mon 19 Oct the pod also relayed, by runpodctl send, the payload three times and the
  teacher once to the AM-18 arms' pods; those sends had ended by 2026-10-19T15:00Z, with their logs under /tmp.
- **The run's end, as found** (MEASURED, read-only, 2026-10-20T03:08:59Z; the W4-END session's B1 at 09:31 Tue 20 Oct and
  ruling E3L-STOP-1's check 1):
  - The stdout log /workspace/ev_e3_s42_lambda0.5_alpha25_160k_a1/e3_s42_lambda0.5_alpha25_160k_a1.stdout.log (386,863 bytes) was last written 2026-10-19 21:51:53.91Z. Its last three lines are progress lines, the last `[iter 154300/160000] loss=4.7458 sup=0.3075 logit_kd=0.6415 cwd_feat=0.0704 cwd_logit=0.7859 ramp=1.0000 lr=4.97255260e-04`. It holds no Traceback (count 0).
  - The telemetry (78,309,942 bytes) was last written 2026-10-19 21:52:37.01Z. Its last two rows are finite train rows, at iterations 154,336 and 154,337 (grad_norm 5.04 and 6.15). It holds 38 VAL rows, the last at iteration 152,000 (all-class mIoU 0.36540040373802185), and run_abort 0, run_end 0 and nonfinite 0.
  - best.json (written 2026-10-19 15:26:53Z) names e3_student_best_iter136000.pt with best_val_miou_all_class 0.376208633184433. The run directory holds e3_run_meta.jsonl (3,785 bytes), e3_telemetry.jsonl, best.json, projection.pt (411,708 bytes) and 11 best checkpoints, at iterations 4,000, 8,000, 12,000, 24,000, 32,000, 36,000, 48,000, 60,000, 88,000, 128,000 and 136,000.
  - The volume /workspace (mounted from ca-mtl-1-vast.runpod.net:/runpodfs/podvolumes/…/m0z26zjgpaa93q, 19 G) was 14 % used, so it was not full.
  - Part B's B2F (read-only, after check 2b): the stdout log holds no end-of-run check line (the form `  <name> : PASS`, count 0) and no `[summary]` line (count 0), so the trainer printed no end-of-run checks or summary; there is one records folder, records_1dad955, and 13 op_*.out files.
- **The telemetry, read from the fault keep** (MEASURED: kd/e3_s42_lambda0.5_alpha25_160k_a1/e3_telemetry.jsonl,
  78,309,942 bytes, sha256 1accbe1a41c28b509efa0888edd3092e4cfec239084141174f843a203feca999, equal to the keep manifest's line; e3_run_meta.jsonl beside it hashes to
  590f6e87…, the launched line's run_meta_sha256): 154,375 rows, 154,337 train and 38 VAL;
  run_abort 0, run_end 0, rows with a `nonfinite` map 0; a torn last line: no. Its last row is a train row at iteration 154,337,
  written Tue 20 Oct 05:52:37 +08:00 (2026-10-19T21:52:37Z).

## 2. The VAL rows written by then (AM-19 item 3(a))

The VAL rows attempt 1 wrote before it stopped, in the order written (MEASURED, the telemetry above; all-class mIoU as
the trainer wrote it):

| VAL | iteration | all-class mIoU | written (+08:00) |
|---|---|---|---|
| 1 | 4,000 | 0.10159627348184586 | Sun 18 Oct 01:14:28 +08:00 |
| 2 | 8,000 | 0.18287082016468048 | Sun 18 Oct 02:38:29 +08:00 |
| 3 | 12,000 | 0.24109721183776855 | Sun 18 Oct 04:02:28 +08:00 |
| 4 | 16,000 | 0.23683711886405945 | Sun 18 Oct 05:26:25 +08:00 |
| 5 | 20,000 | 0.20466622710227966 | Sun 18 Oct 06:50:26 +08:00 |
| 6 | 24,000 | 0.27291837334632874 | Sun 18 Oct 08:14:25 +08:00 |
| 7 | 28,000 | 0.26441818475723267 | Sun 18 Oct 09:38:40 +08:00 |
| 8 | 32,000 | 0.3065701127052307 | Sun 18 Oct 11:02:46 +08:00 |
| 9 | 36,000 | 0.31779858469963074 | Sun 18 Oct 12:26:43 +08:00 |
| 10 | 40,000 | 0.28705132007598877 | Sun 18 Oct 13:50:41 +08:00 |
| 11 | 44,000 | 0.2990975081920624 | Sun 18 Oct 15:14:40 +08:00 |
| 12 | 48,000 | 0.3200051486492157 | Sun 18 Oct 16:38:39 +08:00 |
| 13 | 52,000 | 0.2978285849094391 | Sun 18 Oct 18:02:37 +08:00 |
| 14 | 56,000 | 0.31862640380859375 | Sun 18 Oct 19:26:35 +08:00 |
| 15 | 60,000 | 0.35513073205947876 | Sun 18 Oct 20:50:46 +08:00 |
| 16 | 64,000 | 0.33539238572120667 | Sun 18 Oct 22:14:47 +08:00 |
| 17 | 68,000 | 0.32167455554008484 | Sun 18 Oct 23:38:48 +08:00 |
| 18 | 72,000 | 0.3235018253326416 | Mon 19 Oct 01:02:47 +08:00 |
| 19 | 76,000 | 0.35124364495277405 | Mon 19 Oct 02:26:42 +08:00 |
| 20 | 80,000 | 0.34954574704170227 | Mon 19 Oct 03:50:43 +08:00 |
| 21 | 84,000 | 0.35207054018974304 | Mon 19 Oct 05:14:46 +08:00 |
| 22 | 88,000 | 0.36340054869651794 | Mon 19 Oct 06:38:45 +08:00 |
| 23 | 92,000 | 0.3543545603752136 | Mon 19 Oct 08:02:45 +08:00 |
| 24 | 96,000 | 0.3386482298374176 | Mon 19 Oct 09:26:46 +08:00 |
| 25 | 100,000 | 0.3177410662174225 | Mon 19 Oct 10:50:48 +08:00 |
| 26 | 104,000 | 0.33814895153045654 | Mon 19 Oct 12:14:44 +08:00 |
| 27 | 108,000 | 0.34505921602249146 | Mon 19 Oct 13:38:43 +08:00 |
| 28 | 112,000 | 0.3622128367424011 | Mon 19 Oct 15:02:43 +08:00 |
| 29 | 116,000 | 0.3616790175437927 | Mon 19 Oct 16:26:46 +08:00 |
| 30 | 120,000 | 0.3470127284526825 | Mon 19 Oct 17:50:45 +08:00 |
| 31 | 124,000 | 0.3574497103691101 | Mon 19 Oct 19:14:46 +08:00 |
| 32 | 128,000 | 0.3693986237049103 | Mon 19 Oct 20:38:57 +08:00 |
| 33 | 132,000 | 0.36419937014579773 | Mon 19 Oct 22:02:59 +08:00 |
| 34 | 136,000 | 0.376208633184433 | Mon 19 Oct 23:26:53 +08:00 |
| 35 | 140,000 | 0.37058234214782715 | Tue 20 Oct 00:50:50 +08:00 |
| 36 | 144,000 | 0.357247918844223 | Tue 20 Oct 02:14:57 +08:00 |
| 37 | 148,000 | 0.36660000681877136 | Tue 20 Oct 03:39:06 +08:00 |
| 38 | 152,000 | 0.36540040373802185 | Tue 20 Oct 05:03:34 +08:00 |

- **Best VAL:** 0.376208633184433 at iteration 136,000 (VAL row 34 of 38, written Mon 19 Oct 23:26:53 +08:00; the earliest row on a tie).
- **By its comparator.** AM-19 item 3(a) asks for "the VAL rows written by then by the run and by the other runs of
  its sweep, or by its comparator" (docs/PREREGISTRATION_AMENDMENTS.md at 3373139, lines 1593–1594). E3 at 160k
  belongs to no sweep, and AM-16 item 3 sets "each 160,000-iteration run against its 80,000-iteration run" (the same
  file, lines 292–293). Its comparator is therefore E3 seed 42 at 80k, e3_s42_lambda0.5_alpha25_80k_a1 (λ 0.5, α 25:
  the α selection's winner of record).
  - **Its identity (REPOSITORY-PROVEN at 3373139):** its launch and launched lines are reports/kd_launch_log.jsonl
    lines 13 and 17 (launched Mon 12 Oct 17:03:56 +08:00, 2026-10-12T09:03:56Z). Its run_meta sha256,
    ab73e8279bbb2796b4deb5973c9e10f59192254f2c6ada2a3e9bab48a2edd6a3, is the launched line's and the one
    reports/derived/alpha_selection.json lists for it (inputs[0]). Its e3_run_meta.jsonl, read from its keep, hashes to
    that value (MEASURED).
  - **It ended before attempt 1 launched** (its run_end row, Tue 13 Oct 21:06:56 +08:00, 2026-10-13T13:06:56Z), so every VAL row it wrote was written
    by then.
  - **Its kept telemetry** (MEASURED: kd/e3_s42_lambda0.5_alpha25_80k_a1/e3_telemetry.jsonl, 40,523,892 bytes,
    sha256 734417e4b33e64239cebe2485e3eace03a795163b47bf50667517be40c161d2e, equal to its keep manifest's line) holds 80,021 rows, among them
    20 VAL rows and one run_end row, with no run_abort row and no `nonfinite` map. Its VAL rows, in the
    order written:

| VAL | iteration | all-class mIoU | written (+08:00) |
|---|---|---|---|
| 1 | 4,000 | 0.08035974949598312 | Mon 12 Oct 18:28:12 +08:00 |
| 2 | 8,000 | 0.1981126070022583 | Mon 12 Oct 19:52:16 +08:00 |
| 3 | 12,000 | 0.24441805481910706 | Mon 12 Oct 21:16:27 +08:00 |
| 4 | 16,000 | 0.23171493411064148 | Mon 12 Oct 22:40:41 +08:00 |
| 5 | 20,000 | 0.2613939046859741 | Tue 13 Oct 00:04:58 +08:00 |
| 6 | 24,000 | 0.2735271155834198 | Tue 13 Oct 01:29:18 +08:00 |
| 7 | 28,000 | 0.3077707290649414 | Tue 13 Oct 02:53:48 +08:00 |
| 8 | 32,000 | 0.3111237585544586 | Tue 13 Oct 04:18:18 +08:00 |
| 9 | 36,000 | 0.33465683460235596 | Tue 13 Oct 05:42:34 +08:00 |
| 10 | 40,000 | 0.30759426951408386 | Tue 13 Oct 07:06:34 +08:00 |
| 11 | 44,000 | 0.3311440050601959 | Tue 13 Oct 08:30:40 +08:00 |
| 12 | 48,000 | 0.3530866205692291 | Tue 13 Oct 09:54:49 +08:00 |
| 13 | 52,000 | 0.35441237688064575 | Tue 13 Oct 11:18:52 +08:00 |
| 14 | 56,000 | 0.34778642654418945 | Tue 13 Oct 12:42:54 +08:00 |
| 15 | 60,000 | 0.35122615098953247 | Tue 13 Oct 14:06:56 +08:00 |
| 16 | 64,000 | 0.3444356620311737 | Tue 13 Oct 15:30:55 +08:00 |
| 17 | 68,000 | 0.35341838002204895 | Tue 13 Oct 16:54:54 +08:00 |
| 18 | 72,000 | 0.3604889214038849 | Tue 13 Oct 18:18:56 +08:00 |
| 19 | 76,000 | 0.36337557435035706 | Tue 13 Oct 19:42:57 +08:00 |
| 20 | 80,000 | 0.35811886191368103 | Tue 13 Oct 21:06:56 +08:00 |

- **The comparator's best VAL:** 0.36337557435035706 at iteration 76,000 (VAL row 19 of 20, written Tue 13 Oct 19:42:57 +08:00; the earliest row on a tie). It equals reports/derived/alpha_selection.json's best_val for its
  winner, 0.36337557435035706 at iteration 76,000 (REPOSITORY-PROVEN at 3373139).
- Attempt 1's rows are evidence of the stop only. The repeat replaces the run (AM-8a): they are never used as a result,
  unless the repeat is cut under AM-22 item 2(b), in which case they are reported under AM-19 item 3(c). The
  comparator's rows are given for comparison only.

## 3. Cause: a host failure (the critical error on the pod's machine that RunPod's Alert reports)

- **MEASURED (read-only, 2026-10-20T03:09:16Z; check 2):**
  - PROC 20271 ABSENT: the trainer's process no longer exists.
  - /proc/uptime 18,687.87 s (idle 1,728,077.36 s) at 03:09:16Z, so the machine had booted after the telemetry's last write (the boot below).
  - The container's cgroup: memory.events low 0, high 0, max 0, oom 0, oom_kill 0, oom_group_kill 0; memory.peak 120,823,808 bytes against memory.max 49,999,998,976 bytes. (INFERRED: these counters cover only the time since the container's start after the boot, so they say nothing about what stopped the run.)
  - The host as seen from the pod: MemTotal 527,972,304 kB, MemAvailable 420,106,428 kB, nproc 96; the GPU an NVIDIA A40 with ECC enabled and 0 uncorrected ECC errors, volatile and aggregate (the volatile count resets at a reboot); retired pages pending [N/A].
  - dmesg was refused in the container (`read kernel buffer failed: Operation not permitted`), so the kernel's log could not be read; the image has no `ps` (`bash: ps: command not found`).
  - Check 2b (read-only, 2026-10-20 03:12:56Z, ruling E3L-STOP-2): /proc/1/stat gives `1 (docker-init) 10436` with CLK_TCK 100, so PID 1 started 104.36 s after the boot; its command line is `/sbin/docker-init -- sleep infinity`; /proc/uptime then read 18,908.13 s. So the container started at about 2026-10-19 21:59:32Z, the same second as the console's System Log line below.
- **The machine's boot** (check 2's /proc/uptime subtracted from its time): Tue 20 Oct 05:57:48 +08:00 (2026-10-19T21:57:48Z), 0 h 05 min after the last row.
- **The RunPod console's Alert (DOCUMENTED, Ice's screenshots, 09:51 and 09:52 Tue 20 Oct; ruling HOST-FAULT-1):**
  kd-w4-e3l showed a yellow Alert with GPU 0 % (volume 13 %); kd-e3arm-r1 and kd-e3arm-r2 showed an Alert with GPU 0 %
  too. The Alert's text, read on kd-e3arm-r2, begins "We have detected a critical error on this machine which may affect some pods." It
  recommends backing up the data and creating a new pod.
- **The RunPod console (DOCUMENTED, Ice's notes, 11:10 Tue 20 Oct; ruling E3L-STOP-2):** the pod's uptime reads "2d 12h" (its age since it was created on Sat 17 Oct); its System Logs read "Tue Oct 20 2026 05:59:32 GMT+0800 (Philippine Standard Time) start container for ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b: begin", so the container was started again at 05:59:32 (2026-10-19 21:59:32Z), about 7 min after the telemetry's last row.
- **INFERRED:** the machine under kd-w4-e3l failed between the telemetry's last write (21:52:37Z) and its boot (about 21:57:48Z) on 19 Oct, which is the critical error RunPod's Alert reports, and that failure stopped the trainer. The trainer wrote no Python error, no abort and no non-finite row. After the boot RunPod started the container again (21:59:32Z) with its start command, `sleep infinity` (check 2b's PID 1), so the trainer was not restarted. No written rule and no operator stopped the run. What failed inside the machine cannot be seen from the pod (dmesg was refused).
- The run cannot be resumed. The process has ended, and the runbook's launch block never passes `--resume`
  (REPOSITORY-PROVEN, KD.9).

## 4. How the rules apply

- **AM-8a:** "A run is repeated with the same seed only after a documented infrastructure fault (crash, host or
  pod failure, corrupted artifact, wrong configuration)." As amended by AM-19 item 3(a): "A run that stops in any
  other way (operator, pod, host) is repeated under AM-8a from the first iteration, in a new run directory, on the
  first card item 1(e) gives it."
- **The runbook, KD.11:** "Killed, or the pod lost: AM-8a report, then relaunch on the first card."
- **Not AM-7 and not an abort.** The last row is finite and carries no `nonfinite` map, and no run_abort row exists.
  So this is neither a divergence nor KD.11's "non-finite row with no abort record". No written rule stopped the run
  (AM-19 item 3(a)'s first bullet), and no operator stopped it.
- **The repeat's precondition** (KD.11, "Before any repeat confirm that the attempt before it has ended") holds. The
  process is gone, and its last row (Tue 20 Oct 05:52:37 +08:00) is earlier than any launch of the repeat.
- **AM-22 item 2(b):** the repeat "launches on the first card AM-19 item 1(e) gives it, subject to AM-19 item 3(b)'s
  arithmetic", and it is cut if it has not launched by the end of the third day after the day of the last row
  attempt 1 wrote. AM-19's Definitions read a date in Asia/Manila: "A date ends at 24:00 Asia/Manila of that day
  (16:00:00 UTC). A timestamp meets a date when it is earlier than that instant." The last row was written
  Tue 20 Oct 05:52:37 +08:00 (2026-10-19T21:52:37Z), on the Asia/Manila day Tue 20 Oct (2026-10-20). The cut of record is therefore
  the end of Fri 23 Oct Asia/Manila, 2026-10-23 24:00 +08:00 (2026-10-23T16:00:00Z): the repeat's launch time (its run_meta row's wall_clock) must be earlier than that instant.
- **AM-19 item 3(b)'s arithmetic:** 160,000 iterations × 1.2698394060134888 s/iter (the larger of two reference medians, both recorded in DL-134 (docs/DECISION_LOG.md at 3373139; REPOSITORY-PROVEN): e3_s44's telemetry 1.2486631870269775 s/iter, the most recently finished E3 run, and the KD.6 smoke's 1.2698394060134888 s/iter; the larger is conservative) = 203,174 s, about 56.4 h. Launched at the latest, just before the cut of record (2026-10-23T16:00:00Z), it would end about Mon 26 Oct 08:26:14 +08:00, 208 h before the end of the freeze date, Tue 3 Nov 24:00 +08:00 (2026-11-03T16:00:00Z; AM-22 item 1(c)). It fits. (The rule is REPOSITORY-PROVEN; the end time is INFERRED.)

## 5. What is kept

- AM-8a: "The failed run's logs and partial checkpoints are kept." Everything of attempt 1 on kd-w4-e3l is copied to
  the laptop before the pod is terminated: the whole run directory (every checkpoint, the telemetry, run_meta,
  best.json, projection.pt), the whole evidence folder /workspace/ev_e3_s42_lambda0.5_alpha25_160k_a1, the records
  folder records_1dad955, kd_env.sh and the operator's op_*.out files.
- They are in one keep tar, keep_e3_s42_lambda0.5_alpha25_160k_a1.tar (sha256 81bd859cf6d433ee3cb6c1e802c07be80534c919d1d565fc2c9022b6a315daf3, 344,483,840 bytes), with a
  sha256 manifest of 160 lines; the tar holds 161 files, and every kept file checked OK on the laptop
  (KEPT-OK, 160).
- The run's durable evidence folder is C:\Users\admin\plantseg_runs\kd_w4_20261015\fault_e3_s42_lambda0.5_alpha25_160k_a1.
  It holds the keep tar, its extracted files and this report. Its SHA256SUMS lists this report and the keep tar, and
  it is written before the repeat launches (AM-8a).
- kd-w4-e3l is terminated only after the copy is checked and the orchestrator confirms.

## 6. The repeat

- **One repeat: attempt 2, e3_s42_lambda0.5_alpha25_160k_a2.** It has the same stage, seed, λ, α, horizon, code pin and
  arguments, a fresh run directory /workspace/kd/e3_s42_lambda0.5_alpha25_160k_a2, and a new pod kd-w4-e3l2 (the first
  free card; KD.11). It starts from iteration 0. The payload and the teacher of record are sent from the laptop.
- **Its launch line names this report** (`am8a_report`, path and sha256; KD.8). The launch log records attempt 1's stop
  in a `stopped` line that names this report (KD.10).
- **The repeat replaces the stopped run.** No selection is made between them (AM-8a), and attempt 1's partial results
  are never used as a result. If the repeat is cut under AM-22 item 2(b), attempt 1's VAL rows (section 2) are reported
  under AM-19 item 3(c).
- **If attempt 2 fails,** whatever the cause, it is a STOP for the orchestrator before anything else runs.

## 7. Timeline

| Time (+08:00) | Event | Label |
|---|---|---|
| Sat 17 Oct 23:50:26 +08:00 | attempt 1 launched on kd-w4-e3l | REPOSITORY-PROVEN, launch log |
| Tue 20 Oct 05:03:34 +08:00 | the last VAL row (iteration 152,000) | MEASURED, telemetry |
| Tue 20 Oct 05:51:53 +08:00 | the stdout's last write (the progress line for iteration 154,300) | MEASURED, file time |
| Tue 20 Oct 05:52:37 +08:00 | the telemetry's last row (train, iteration 154,337) | MEASURED, telemetry |
| Tue 20 Oct 05:57:48 +08:00 | the machine's boot, by check 2's /proc/uptime | MEASURED |
| Tue 20 Oct 05:59:32 +08:00 | the container started again (the console's System Log; check 2b: PID 1 started 104.36 s after the boot) | DOCUMENTED (console notes) and MEASURED (/proc) |
| Tue 20 Oct 09:31 +08:00 | found ended (W4-END, Part B's B1) | MEASURED |
| Tue 20 Oct 09:51 +08:00 | kd-w4-e3l: Alert and GPU 0 % in the RunPod console | DOCUMENTED, screenshot |
| Tue 20 Oct 11:08:59 +08:00 | check 1 (read-only) | MEASURED |
| Tue 20 Oct 11:09:16 +08:00 | check 2 (read-only) | MEASURED |
| Tue 20 Oct 11:10 +08:00 | Ice's look at the RunPod console (the uptime and the System Logs) | DOCUMENTED, Ice's notes |
| Tue 20 Oct 11:12:56 +08:00 | check 2b (read-only): PID 1 is /sbin/docker-init -- sleep infinity | MEASURED |
| Tue 20 Oct 11:24:10 +08:00 | the keep tar received on the laptop (44 s; its sha256 equal to the pod's), then KEPT-OK with 160 files | MEASURED |
| Fri 23 Oct 24:00 | AM-22 item 2(b): the repeat is cut if it has not launched by now (2026-10-23T16:00:00Z) | REPOSITORY-PROVEN rule |
