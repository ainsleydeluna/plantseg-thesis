# KD wave 4, F seed 42: a host failure ended attempt 1 (AM-8a fault report)

Written 2026-10-18 by the orchestrator (ruling W4F-1), before the repeat. Labels: MEASURED (a command's output that
Ice De Luna pasted, or his screenshot of the RunPod console), REPOSITORY-PROVEN (read in the code at the KD code pin
8487551 or in the runbook at master), INFERRED. Times are +08:00 unless they end in Z.

## 1. What failed

- **Run:** f_s42_alpha25_80k_a1, stage F, seed 42, α = 25, 80,000 iterations, attempt 1 (AM-17b item 1(a); wave 4).
  Code pin 8487551ca67bd50ffbe911d1f3e93244841cbc9f; records commit 1dad9550bbeb82eb6a87d1e7c20f36d030afe0ef.
- **Its launch (reports/kd_launch_log.jsonl at e837ba5d14d577faf0d974b336d34ded0d4d4a25):** launched 2026-10-18 00:24:19
  (launch_time_utc 1792254259.2933776); run_meta sha256 32e432c54963d1640ce112777bb02c4f92331f7fea632b95ffff8ba7087fc70c;
  launched line sha256 8e6c9ccbd7a5f7e4afcd54b108fb860a742c98f6c396a094701830dbe598e0fa. The gate printed VERDICT: GO
  with every row PASS, and check-run-meta returned RESULT: PASS.
- **The pod:** kd-w4-f (RunPod id 1ya2wxmnu26kxt), Secure Cloud, On-Demand, 1 × NVIDIA A40 (46,068 MiB), the
  plantseg-kd template; /workspace an nfs4 pod volume from ca-mtl-1-vast.runpod.net. Trainer pid 7885.
- **The run's end, as found (MEASURED, read-only, 2026-10-18T02:33:20Z):**
  - The GPU was empty: `NVIDIA A40, 0 MiB, 0 %`. At 10:31 the RunPod console showed kd-w4-f running at 0% CPU,
    0% GPU, 0% memory and 1% GPU memory, while the project's other four pods ran at 99–100% GPU.
  - The stdout log (stdout and stderr; REPOSITORY-PROVEN, scripts/preflight_distill.py's launch block) was last
    written at 2026-10-17T22:59:03Z. It ends with the ordinary progress line
    `[iter 18850/80000] loss=4.4424 sup=2.0944 cwd_feat=0.0939 ramp=1.0000 lr=7.85191929e-03`, and it holds no
    Traceback (count 0), no error line and no end line.
  - The telemetry's last row is iteration 18884, a finite train row (loss 5.7017, grad_norm 13.0316) written at
    wall_clock 1792277982.4184346 (2026-10-18 06:59:42). It has no `nonfinite` map, and no run_abort or run_end row
    follows it.
  - The run directory /workspace/kd/f_s42_alpha25_80k_a1 holds f_run_meta.jsonl, f_telemetry.jsonl (8,303,749 B),
    best.json, projection.pt and the best checkpoints of iterations 4000, 8000, 12000 and 16000 (the last written
    2026-10-17T21:59:31Z).
  - The disk: 2.1 GB of 19 GB used (11%).

## 2. Cause: the pod's machine restarted

- **MEASURED (read-only, 2026-10-18T02:38:37Z):**
  - `/proc/uptime` read 12,844.19 s, so the machine (96 CPUs; MemTotal 527,972,720 kB) booted at about 07:04
    (2026-10-17T23:04:32Z).
  - The container's PID 1, `/sbin/docker-init -- sleep infinity`, started 91.67 s after that boot (about 07:06).
  - The trainer's process is gone (`/proc/7885` does not exist).
  - The container's memory cgroup reads `oom_kill 0` and `failcnt 0`, with a limit of 49,999,998,976 B. These
    counters belong to the container started after the boot.
  - The GPU's ECC is enabled, and every uncorrectable counter is 0, volatile and aggregate (SRAM parity, SRAM
    SEC-DED and DRAM). No channel or TPC repair is pending.
  - `dmesg` is not permitted in the container.
- **INFERRED:** the machine under kd-w4-f failed or was restarted on RunPod's side. The trainer wrote its last row
  at 06:59:42, about 290 s before the boot, so the failure began at or before then: the run hung or was being shut
  down. Nothing points at the run itself:
  - no Python error;
  - a finite last row;
  - no memory-limit kill recorded;
  - no GPU memory error;
  - the disk nearly empty;
  - the same code running normally on kd-w4-a, kd-w4-g and kd-w4-e3l.
- The run cannot be resumed. The machine's restart ended the process, and the runbook's launch block never passes
  `--resume` (REPOSITORY-PROVEN, KD.9).

## 3. How the rules apply

- **AM-8a:** "A run is repeated with the same seed only after a documented infrastructure fault (crash, host or
  pod failure, corrupted artifact, wrong configuration)." As amended by AM-19 item 3(a), "a run stopped by the
  operator, the pod or the host is repeated under this rule".
- **The runbook, KD.11:** "Killed, or the pod lost: AM-8a report, then relaunch on the first card."
- **Not AM-7 and not an abort.** The last row is finite and carries no `nonfinite` map, and no run_abort row
  exists. So this is neither a divergence nor KD.11's "non-finite row with no abort record".
- **The repeat's precondition** (KD.11, "Before any repeat confirm that the attempt before it has ended") holds.
  The process is gone, and its last row (06:59:42) is earlier than any launch of the repeat.

## 4. What is kept

- AM-8a: "The failed run's logs and partial checkpoints are kept." Everything of attempt 1 on kd-w4-f is copied to
  the laptop before the pod is terminated. That is: the whole run directory (every checkpoint, the telemetry,
  run_meta, best.json, projection.pt), the whole evidence folder /workspace/ev_f_s42_alpha25_80k_a1 (the gate
  record and log, the launch block, the stdout log, the pid, check-run-meta's log, the launched line), kd_env.sh and
  the operator's op_*.out files.
- They go into one keep tar, keep_f_s42_alpha25_80k_a1.tar, with a sha256 manifest of every file.
- The run's durable evidence folder is C:\Users\admin\kd_wave4_20261015\fault_f_s42_alpha25_80k_a1. It holds the keep
  tar, its extracted files and this report. Its SHA256SUMS lists this report and the keep tar, and it is written
  before the repeat launches (AM-8a).
- kd-w4-f is terminated only after the copy is checked and the orchestrator confirms.

## 5. The repeat

- **One repeat: attempt 2, f_s42_alpha25_80k_a2.** It has the same stage, seed, α, horizon, code pin and
  arguments, a fresh run directory /workspace/kd/f_s42_alpha25_80k_a2, and a new pod (the first free card;
  KD.11). It starts from iteration 0. It is planned to launch on 2026-10-18, inside AM-22 item 2(a)'s window (a run
  is cut if it has not launched by the end of 2026-10-20).
- **Its launch line names this report** (`am8a_report`, path and sha256; KD.8). The launch log records attempt 1's
  stop in a `stopped` line that names this report (KD.10).
- **The repeat replaces the failed run.** No selection is made between them (AM-8a), and attempt 1's partial
  results are never used as a result.
- **If attempt 2 fails,** whatever the cause, it is a STOP for the orchestrator before anything else runs.

## 6. Timeline

| Time (+08:00) | Event | Label |
|---|---|---|
| Sun 18 Oct 00:24:19 | attempt 1 launched on kd-w4-f | MEASURED, launch log |
| 01:48, 03:11, 04:35, 05:59 | best checkpoints of iterations 4000, 8000, 12000, 16000 | MEASURED, file times |
| 06:59:03 | the stdout log's last write (iteration 18850) | MEASURED |
| 06:59:42 | the telemetry's last row (iteration 18884) | MEASURED |
| about 07:04 | the pod's machine boots | MEASURED, uptime |
| about 07:06 | the container starts again (sleep infinity) | MEASURED, PID 1 |
| 10:31 | found idle in the RunPod console | MEASURED, screenshot |
| 10:33, 10:38 | the read-only checks of §1 and §2 | MEASURED |
