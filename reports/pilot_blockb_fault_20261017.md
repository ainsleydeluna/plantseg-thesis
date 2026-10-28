# U4 pilot, block B: a host fault in clip 5.0's record scoring (AM-8a fault report)

Written 2026-10-17 by the orchestrator (ruling FAULT-B1), before the repeat. Labels: MEASURED (read from the
laptop's records, or from a command's output that Ice De Luna pasted), REPOSITORY-PROVEN (read in the code at
2b3b556), INFERRED. Times are +08:00 unless they end in Z.

## 1. What failed

- **Run:** e5_s42_clip5.0, E5 seed 42 at clip 5.0, one of the two U4 pilot runs (AM-4a item 3). Its telemetry
  sha256 as recorded on the pod: c7b2023943224008e7106fdc771fe8c29b2324a90a6b581aa9a0f618408a5898. The run itself,
  trained on a pod and complete and finite (15/15 epochs), is untouched.
- **Step:** block B's record scoring of clip 5.0 (qat_pilot_blockb_20261014.sh, step `score 5.0`; container
  q2_a1_c50_score; evaluation directory C:\Users\admin\plantseg_runs\qat_pilot_20261014\evals\a1\e5_s42_clip5.0).
- **The container (MEASURED, docker inspect):** started 2026-10-16T14:12:47.883Z, finished 2026-10-16T16:10:32.190Z
  (2026-10-17 00:10:32, by the Docker VM's clock), exit code 4, OOMKilled false.
- **Its log (MEASURED, docker logs, as pasted):** the scores of epochs 1 to 7, then two error lines:

  ```
  [score] e01: all-class mIoU 0.29566916823387146 (639.9s)
  [score] e02: all-class mIoU 0.3269672095775604 (500.1s)
  [score] e03: all-class mIoU 0.32098138332366943 (654.9s)
  [score] e04: all-class mIoU 0.33026283979415894 (732.8s)
  [score] e05: all-class mIoU 0.3313092887401581 (987.7s)
  [score] e06: all-class mIoU 0.3297959566116333 (1602.6s)
  [score] e07: all-class mIoU 0.3192453384399414 (1901.3s)
  OSError: [Errno 5] Input/output error: '/data/images/val/apple_rust_google_0111.jpg'
  RESULT: ERROR [OSError] -- [Errno 5] Input/output error: '/work/repo/src/quant'
  ```

- **The block's own capture of that container (MEASURED, blockb_state\q2_a1_c50_score.out, 492 B, last written
  2026-10-16T16:10:29.8Z):** the same seven score lines, then the docker client's line

  ```
  time="2026-10-17T00:10:29+08:00" level=error msg="error waiting for container: unexpected EOF"
  ```

  The docker client lost its connection to Docker Desktop at 00:10:29, before the container's error lines reached it.
- **Reading (REPOSITORY-PROVEN, scripts/qat_epoch_eval.py at 2b3b556, cmd_score and main):** epoch 8's evaluator, a
  subprocess, failed reading a VAL image from the container's read-only data mount. The score command's failure
  branch, which writes score_STOP.json, then raised an OSError of its own naming /work/repo/src/quant (INFERRED: its
  import of src.quant.ptq, the branch's next statement, read the read-only checkout mount), so no score_STOP.json was
  written, and main printed RESULT: ERROR (exit 4).
- **On disk (MEASURED):** scores\e01 to e07 are complete (each holds MANIFEST.sha256, per_image.jsonl,
  sufficient_stats.npz and summary.json); there is no e08, no qat_epoch_eval.json and no STOP file. The block's step
  log blockb_logs\10_score50.out (549 B) ends with check2's lines, last written 2026-10-16T14:12:47Z, and holds
  nothing after them: the block B process printed no RESULT or STOP line (INFERRED: it ended with the same fault).

## 2. Cause

- The docker client's connection and both read-only bind mounts of the container, the laptop's data root and its
  checkout, failed within seconds of each other (00:10:29 to 00:10:32), while the Docker engine went on and recorded
  the container's exit.
  INFERRED from these MEASURED lines: a failure of Docker Desktop's host side (its connection and its file sharing),
  not of the code, the data or the run. The same commit, image, host and data root scored clip 1.0's 15 epochs and
  clip 5.0's first 7 without an error (MEASURED).
- Windows did not restart (LastBootUpTime 2026-10-14 17:40:22, MEASURED). INFERRED: the laptop went to sleep at about
  00:10, and Docker Desktop's file sharing failed with it. The recovery session reads the System event log's power
  events of that night, and the decision log records what they show.
- **Precursor (MEASURED):** clip 5.0's per-epoch scoring times rose from 500–733 s (epochs 1–4) to 988, 1603 and
  1901 s (epochs 5–7). Clip 1.0's epochs took 478–716 s, apart from epoch 5 (9662 s).

## 3. Class

A host failure (AM-8a: "crash, host or pod failure"): an infrastructure fault.

## 4. Consequence (AM-8a; AM-21 item 2(c); the Q2 lane report §15)

- The a1 evaluation directory of clip 5.0 is spent and kept unchanged. Its seven partial scores are reported here and
  are never used.
- Clip 5.0's record conversion and scoring are re-made once, in a new evaluation directory
  (C:\Users\admin\plantseg_runs\qat_pilot_20261014\evals\a2\e5_s42_clip5.0): an evaluation directory is scored once,
  and its scoring needs that directory's own conversion record (REPOSITORY-PROVEN, cmd_score). The re-made pass
  replaces a1's for clip 5.0, and two versions are never chosen between.
- Clip 1.0's a1 records stand: its conversion (15/15), its scoring (15/15) and its selection (epoch 08, converted VAL
  all-class mIoU 0.33294206857681274) were complete before the fault.
- The repeat runs from a clean clone of the repository at 2b3b556 (ruling QEV-1), with the same image, host and data
  root. The clip selection then compares clip 1.0 (a1) with clip 5.0 (a2), and the winner is finalized.
- **Prevention:** before the repeat, the laptop's sleep and hibernate timeouts on AC power are set to Never and its
  lid-close action on AC power to Do nothing (the recovery session checks all three), Windows updates are paused, and
  the laptop stays on its charger until block B ends.

## 5. Evidence kept

- evals\a1\e5_s42_clip5.0, blockb_state and blockb_logs, unchanged; the container q2_a1_c50_score, not removed.
- This report, written to qat_pilot_20261014\faults\ and sha256-listed in its SHA256SUMS before the repeat, and
  committed with the next docs commit (AM-8a).
