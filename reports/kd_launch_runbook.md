# KD launch runbook — E2, E3, A, F and G (lane 4(b), K2)

Added 2026-10-07 by lane K2 (Session 2, commit C4). Not a governed file. It applies to every KD launch: the
λ sweep (E2, seed 42, 80,000 iterations), the α sweep (E3, seed 42), E2 and E3 at seeds 43 and 44, the arms A, F
and G, and the 160,000-iteration controls. Authority: AM-19 with AM-19a (readings 1–21), DL-87 (the KD image),
DL-88 (the KD values of record), DL-89 (one code pin per sweep), the K2 patch list PL-25 to PL-32 and CHECK ITEMS 3,
5 and 6. The E1 runbook's §9 (`reports/e1_launch_runbook_v2.md`) still governs E1; where a step here cites it, its
text applies unchanged. Nothing in this runbook is run from a cloud lane.

Placeholders: `<pin>` the sweep's code pin (DL-89); `<H>` the records commit (a later commit that holds the launch
line, PL-28); `<D>` the run's checkpoint directory, named by its run_id; `<E>` the evidence folder; `<F>` the records
folder; `<T>` the teacher file. A run_id is `<stage>_s<seed>[_lambda<v>][_alpha<a>]_<80|160>k_a<n>`.

## Flow per wave (PL-32)

1. **Local, in the KD image:** print each launch line with `entry` (KD.8), append it to `reports/kd_launch_log.jsonl`,
   commit and push. Press **Sync now** on both GitHub sources.
2. **Pod:** clone, check out the pin, fetch `<H>`, write the records folder (KD.2), stage the data, the backbone and
   the teacher (KD.3–KD.5), run the pod smoke on the first pod (KD.6), record the clock offset and run the gate
   (KD.7), launch (KD.9), and run check-run-meta within 5 minutes (KD.10).
3. **Local:** append the launched line that check-run-meta printed, commit and push. Press **Sync now**.

A non-default candidate's records commit holds the default's launched line: launch the default, push its launched
line, and only then print, push and gate the next candidate (AM-19 item 1(e); the gate's `launch_order` stage).

## Where commands run (PL-32)

`entry`, `select_lambda.py`, `select_alpha.py`, `--write-decision-record`, `push-evidence` and the laptop's clock
offset run in the KD image, never with the host's Python. `entry` mounts the clone read-only and `--out` on a separate
writable mount:

```bash
docker run --rm -v <clone>:/repo:ro -v <out>:/out -w /repo -e PYTHONPATH=/repo \
  ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
  python -B scripts/<script> <arguments>
```

The selections and the record writer also mount the run directories (the copies off the pod, KD.14) read-only at
`/runs`, and name them there: `--runs /runs/<run_id> ...` (a path the container cannot see is a run not supplied,
AM-19a reading 20):

```bash
docker run --rm -v <clone>:/repo:ro -v <runs>:/runs:ro -v <out>:/out -w /repo -e PYTHONPATH=/repo \
  ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
  python -B scripts/select_lambda.py --runs /runs/<run_id> /runs/<run_id> ... <arguments>
```

`push-evidence` fetches from the remote. It runs on a full (not shallow) clone mounted writable, whose remote it can
read: for a private repository, a URL with a read-only token (`https://<token>@github.com/...`); the list records the
URL and the remote's answers without it. It writes only git objects into that clone.

The files these commands write are copied from `<out>` into the clone at the path each RESULT line names, then
committed.

## KD.1 Pod and shell

Start the pod from the KD image by digest (DL-87), with enough `/dev/shm` for 12 DataLoader workers. In the one
shell that will run the gate and the launch:

```bash
export PLANTSEG_IMAGE_DIGEST=sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b
unset PLANTSEG_GIT_COMMIT
```

## KD.2 Checkout, records commit and records folder (PL-28)

The pod checks out the pin itself, as E1's pods did (§9.2: a DL-19 partial clone, `--filter=blob:none
--no-checkout`, sparse patterns `/*` and `!/docs/reference/`, `checkout --detach <pin>`; never `--depth`: the gate
refuses a shallow clone, PL-12). Step 0 of §9.2 holds: HEAD equals the pin, the protected reference file is absent,
the scoped status is clean. Then, before the remote URL is invalidated:

```bash
export PYTHONPATH=<clone>
git fetch --filter=blob:none origin <H>
python -B scripts/preflight_distill.py records --records-commit <H> --out <F>
```

`<F>` is a new folder outside the clone. The command writes the launch log, the selection files, the decision records
and the corrected ones with the void records and fault reports they name, the band file, the push evidence and its
list, the AM-7a clipping value, the decision log and every report the launch log names, each as `<H>` holds it, with
a manifest of blob ids. The pin must be an ancestor of `<H>`. On the first pod, also export commit 73fd4d7 (the pod
smoke's `--old-root`, PL-34) before the remote is invalidated.

A new records commit on a running pod needs a fetch and a new records folder; HEAD stays at the pin.

## KD.3 Data

As §9.3. Record the payload's sha256 line in `<E>` before extracting (PL-32).

## KD.4 ImageNet backbone

As §9.4.

## KD.5 Teacher

Stage the teacher of record (DL-88) outside the clone, the ckpt dir and the data root, and record its sha256 line in
`<E>`:

```bash
sha256sum <T> | tee <E>/teacher_sha256.txt
```

It must read `8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e`, the file named `iter_24000.pth`,
335,949,080 bytes. A different file is a STOP.

## KD.6 Pod smoke (first pod of the wave)

`scripts/smoke_kd_step.py` on the pod, as its usage text gives it, in KD.1's shell; its verdict and the throughput
it measures are recorded in `<E>`. A STOP from it is a STOP for the wave. A `REFUSED` line (`RESULT: REFUSED`, exit
2) comes before any child runs and is no STOP: correct the argument, the shell (KD.1) or the checkout (KD.2: HEAD is
the pin) and run it again.

## KD.7 Clock offset and gate

Record the pod's offset from a network time source first (AM-19a reading 21; §9.13): more than 60 seconds is a STOP.

```bash
python -B -c "import email.utils, time, urllib.request as u; r = u.urlopen(u.Request('https://api.github.com', method='HEAD'), timeout=10); print(round(time.time() - email.utils.parsedate_to_datetime(r.headers['Date']).timestamp(), 1))" | tee <E>/clock_offset_<run_id>.txt
```

On the laptop (before a selection with a cut, KD.13, and before a decision record is written, KD.12) the machine that
runs the step is the KD-image container, so the offset is taken in the same image, immediately before the step:

```bash
docker run --rm ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
  python -B -c "import email.utils, time, urllib.request as u; r = u.urlopen(u.Request('https://api.github.com', method='HEAD'), timeout=10); print(round(time.time() - email.utils.parsedate_to_datetime(r.headers['Date']).timestamp(), 1))" | tee <out>/clock_offset_<sweep>_<step>.txt
```

More than 60 seconds is a STOP there too.

Then the gate, with every value of the launch as its launch line holds it:

```bash
python -B scripts/preflight_distill.py gate --profile distill_80k --stage <S> --seed <N> \
  [--lambda-logit <λ>] [--alpha <α>] [--attempt <n> --am8a-report <report>] \
  [--previous-run <dir1> --previous-run <dir2> ...] \
  [--decision-record reports/derived/<sweep>_decision_record_corrected.json] \
  --ckpt-dir <D> --evidence <E> --expect-head <pin> --records-commit <H> --records <F> --teacher-ckpt <T> \
  --clock-offset-seconds <offset> --record <E>/gate_<run_id>.json 2>&1 | tee <E>/gate_<run_id>.log
```

Every further gate run of the same run_id, after a NO-GO or after a GO whose launch block was not run, keeps the
earlier record and log and writes `--record <E>/gate_<run_id>_<k>.json` and `tee <E>/gate_<run_id>_<k>.log`, with k =
2, 3, …: the gate refuses an existing `--record` (`record_exists`) and `tee` would overwrite the log, and the E1
runbook's §9.5 gives every gate run new evidence names for the same reasons. The GO record `<G>` that KD.10 names is
the `--record` of the gate run whose launch block was run (KD.9).

`--previous-run` is given once per directory: a non-default value's attempt after its decision date (the on-course
repeat) names every stopped earlier attempt of the value, from the copies of their run_meta and telemetry (KD.14)
placed on the pod.

`--profile distill_160k` for the 160,000-iteration controls (E2 and E3 at seed 42 only). It must print
`VERDICT: GO`. The stages, first FAIL stops: `arguments` → `data_isolation` → `repo_state` → `module_provenance` →
`records` → `selection_inputs` → `launch_order` → `class_weights` → `smoke_loss` → `kd_image` → `teacher_identity`
→ `cuda` → `imagenet_backbone` → `smoke_loader_seed` → `smoke_dataloader` → `seed_sequence_R5` → `kd_dry_run` →
`repo_unchanged`. It prints the decision date and the launch-order position (the §9.13 launch note) and warns when
more than 6 hours have passed since the sweep default's first launch. `--rehearsal` exercises the gate off-pod and
never prints a launch block.

Where λ and α come from (the gate's `selection_inputs`): a λ-sweep launch passes its λ; once
`reports/derived/lambda_selection.json` exists only its winner launches. E2 at seeds 43/44 and the 160k controls take
λ from that file; the α sweep runs at its λ and passes its α; E3 at seeds 43/44, A and F take α from
`alpha_selection.json`, or α = 50 from the committed α-cut record (select_alpha has exited 5 and printed that record's
sha256; keep its log, PL-32); G takes neither. A sweep's corrected decision record is passed with `--decision-record`,
and must be once `<H>` holds one (CHECK ITEM 5).

## KD.8 Launch-log entry (local, before the pod)

In the KD image (see "Where commands run"):

```bash
python -B scripts/preflight_distill.py entry --profile distill_80k --stage <S> --seed <N> [--lambda-logit <λ>] \
  [--alpha <α>] [--am8a-report <report>] --ckpt-dir <D> --code-pin <pin>
```

It prints the launch line, the decision date and the launch-order position. Append the line to
`reports/kd_launch_log.jsonl` as one line, exactly as printed, commit and push; that commit or a later one is `<H>`.
The §9.13 checklist applies: no adviser reply is waiting to be entered (AM-19 item 1(b)); before E2 or E3 at seed
43/44, check the decision log for a recorded divergence of the same stage and seed. Before the first KD run (AM-18's
Gate): the decision log holds the AM-18 item 1(a) output that holds F and the item 1(d) output (margin_dedup), each
hashed, with the item 6 branches they select, and margin_dedup > 0 (item 6(b): ≤ 0 halts); otherwise no KD run
launches: STOP and inform the adviser. Neither the gate nor `entry` checks these entries; this line does. An attempt
that follows a launched attempt of its group names its committed AM-8a report (`--am8a-report`), also when
not_launched starts lie between them (PL-9(c)); `entry` and the gate refuse it otherwise. Every launch line of a sweep
carries the sweep's one code pin (DL-89): `entry` refuses another.

## KD.9 Launch

Save the printed block as `<E>/<run_id>.sh` and run it with `bash` from the KD.1 shell. It refuses a non-empty
`<D>` or an existing `<E>/<run_id>.stdout.log`, writes stdout and the pid file to `<E>` (nothing is written into
`<D>` before the trainer), pins `PYTHONPATH`, `PLANTSEG_DATA_ROOT`, `PLANTSEG_IMAGE_DIGEST` and `TORCH_HOME`,
passes the records folder's copies to `--lambda-selection` and `--alpha-selection`, `--records-commit <H>`,
`--num-workers 12` and `--log-every 50`, and never passes `--lambda`, `--grad-clip-norm`, `--resume`, `--max-iters`,
`--batch-size`, `--val-interval`, `--max-val-batches`, `--device`, `--teacher-config`, `--lambda-semantics`,
`--allow-semantics-mismatch`, `--allow-offgrid` or `--dry-run`. Its last line names the next steps.

## KD.10 check-run-meta (within 5 minutes)

```bash
python -B scripts/preflight_distill.py check-run-meta --gate-record <G> --ckpt-dir <D> \
  2>&1 | tee <E>/check_run_meta_<run_id>.log
```

`<G>` is the GO record: the `--record` of the gate run whose launch block was run (KD.7, KD.9),
`<E>/gate_<run_id>.json` when the gate ran once. It checks the run's one run_meta row against one rule per key
(PL-30) and prints the launched line on PASS and on FAIL. Append that line to the launch log, commit and push, then
press **Sync now**. Three refusals exit 2 (`RESULT: REFUSED`) with no launched line, and none is a FAIL: a `<D>` and
a gate record of different runs (another name or folder; a trailing slash is the same directory): rerun with this
run's `<D>` and its `<G>`; a gate record that cannot be read or is not a complete GO record (a NO-GO record
included): rerun with `<G>`, and if that file itself is the cause, fix its access and rerun within the 5 minutes,
otherwise STOP and report; a file of the run's own `<D>` that cannot be read: fix its access and rerun within the 5
minutes, otherwise STOP and report (the run has no launched line). On FAIL, stop the run: a stopped line with its
AM-8a report ("wrong configuration") is committed, the stop is reported as a deviation outside AM-19 item 3(a), and
the repeat launches into a fresh `<D>` with the next attempt number. A start that wrote no run_meta row is not
launched: append a `not_launched` line with its evidence (AM-19a reading 6).

No command prints the stopped and not_launched lines; write each as one line with exactly these keys (the launch log
refuses any other form, PL-13), after committing the file it names (its sha256 is the committed file's):

```text
{"event": "stopped", "run_id": "<run_id>", "report": {"path": "<repository path of the AM-8a report>", "sha256": "<sha256>"}}
{"event": "not_launched", "run_id": "<run_id>", "reason": "<what refused the start>", "evidence": {"path": "<repository path of the evidence>", "sha256": "<sha256>"}}
```

## KD.11 Exit codes

train_distill.py:
- **0:** hash the checkpoints on the pod (KD.14); list the evidence within 24 hours.
- **1:** with a run_end whose checks failed: AM-8a report, then one repeat in a new directory; if it fails the same
  way as any earlier attempt of the value (AM-19a reading 9), it is recorded "aborted (rule, cause)": a non-default
  value is cut at its decision date, the default is a STOP. From a traceback: a stop, handled under AM-8a, unless the
  telemetry ends in a row the trainer flagged non-finite (its `nonfinite` map) with no run_abort after it (below).
- **2:** not launched: append a `not_launched` line, fix, start a new attempt.
- **3:** `student_divergence` is never repeated: a non-default sweep value is excluded; the seed-42 default gets
  AM-7 in full (STOP); E2/E3 at seeds 43/44, A, F, G and the 160k controls follow the arm rule (AM-19 item 5). Any
  other abort: AM-8a report, then one repeat in a new directory; the same ending twice is "aborted (rule, cause)".
- **Killed, or the pod lost:** AM-8a report, then relaunch on the first card. A lost run directory of a launched run
  is a STOP for the orchestrator (PL-32).
- **A non-finite row with no abort record** (a traceback or a kill whose telemetry ends in a flagged non-finite row and
  no run_abort): an abort with (rule, cause) = (non-finite row, abort record missing), never a stop and never a
  divergence (AM-19a reading 13). Write the AM-8a report and make one repeat; the same ending twice is "aborted
  (rule, cause)".
- **Before any repeat** confirm that the attempt before it has ended (the pod terminated or the process gone): a repeat
  that launches no later than the previous attempt's largest timestamp refuses the selection (repeat_overlap, AM-19a
  reading 12).

select_lambda.py and select_alpha.py: **0** selected; **2** STOP and investigate; **3** wait, or write the decision
record (KD.12); **4** error; **5** α = 50 from the record (select_alpha only).

## KD.12 Decision dates and the decision record

The λ decision date is 2026-10-19 under Schedule T (2026-12-07 under R); the α date is the later of 2026-10-22 under
T (2026-12-10 under R) and the third day after the Asia/Manila day on which the commit that added
`reports/derived/lambda_selection.json` reached the remote (AM-19a reading 10; KD.13). A date ends at 24:00
Asia/Manila (16:00 UTC). §9.13 applies.

After the end of a decision date, run the selection first (KD.13): a run it reports as "the on-course repeat of ..."
is waited for as the default is, with no time limit, and is never stopped (stopping it cuts the value: AM-19 item
2(b); AM-19a reading 7). Stop the sweep's other unfinished non-default candidates (a stopped line and its report
each), copy every run's run_meta and telemetry off the pod (the copies match the pod-side sha256, PL-32), then write
the decision record. **The `--write-decision-record` step:**
- the record may be written while the default runs (AM-19a reading 19);
- every non-default run has ended or was stopped first, except a running on-course repeat, which the record lists as
  "running (on-course repeat, item 2(b))" (AM-19a reading 7);
- a late record is a deviation from AM-19 item 2(h): its 24 hours count from the later of the end of the decision
  date and the default's last ending, and the record is written only once the on-course repeat has launched and the
  stop has its stopped line (readings 7 and 19);
- the machine's clock offset from a network time source is recorded first (KD.7's laptop command, in the KD image),
  and an offset above 60 seconds stops the step (AM-19a reading 21).

```bash
python -B scripts/select_lambda.py --runs <every launched run dir of the sweep> \
  --write-decision-record /out/lambda_decision_record.json
```

(`select_alpha.py` likewise, with `alpha_decision_record.json`.) Review the record, commit it as
`reports/derived/<sweep>_decision_record.json`, push it within the 24 hours, and enter its sha256 in the decision log.
Never push the record, or a selection file with a cut, before the end of the decision date: one whose commit reached
the remote before C is void (AM-19a reading 21). No code checks when it reached the remote, so after each such push
save GitHub's activity response verbatim in `<E>` (`gh api "repos/ainsleydeluna/plantseg-thesis/activity?per_page=100"
> <E>/activity_after_<file>.json`): the push time is then on record. A committed record is never deleted or edited:
the selection and the gate refuse a deleted one (decision_record_rewritten, a STOP).

**A void record (AM-19a reading 19; CHECK ITEM 5).** A record the selection's checks refuse for a reason other than a
later non-default launch line is void. It is kept and never edited. Write the AM-8a fault report, commit it, enter its
sha256 and the void record's sha256 in the decision log, then write the one corrected record:

```bash
python -B scripts/select_lambda.py --runs <...> --write-decision-record /out/lambda_decision_record_corrected.json \
  --corrects reports/derived/lambda_decision_record.json --fault-report <committed report>
```

Commit it as `reports/derived/<sweep>_decision_record_corrected.json`, enter its sha256 in the decision log beside the
other two, and pass `--decision-record reports/derived/<sweep>_decision_record_corrected.json` to every later
selection and gate run. A corrected record is written once; a record refused for a later non-default launch line is
not corrected (the selection refuses: a STOP for a ruling).

## KD.13 Selections and the push evidence

Before a selection with a cut, record the machine's clock offset (KD.7's laptop command, in the KD image); more than
60 seconds is a STOP (AM-19a reading 21; PL-32).

```bash
python -B scripts/select_lambda.py --runs <run dirs> [--decision-record <path>] --out /out/lambda_selection.json
python -B scripts/select_alpha.py --runs <run dirs> --band reports/derived/dl27_band.json \
  --lambda-selection reports/derived/lambda_selection.json [--decision-record <path>] --out /out/alpha_selection.json
```

**On the day `lambda_selection.json` is pushed** (PL-32; CHECK ITEM 3; AM-19a reading 10), whether or not the
evidence is needed later:

```bash
gh api "repos/ainsleydeluna/plantseg-thesis/activity?per_page=100" > <out>/activity_response.json
python -B scripts/preflight_distill.py push-evidence --response <out>/activity_response.json --out <out>/evidence
```

The response is saved verbatim. `push-evidence` runs on a full clone (see "Where commands run"; it refuses a shallow
one). It asks the remote, for every push after T_lo (the λ selection's last input timestamp less 60 s), whether it
still serves the pushed commit, from an empty repository so that no local object answers for the remote, and records
the answer. A pushed commit the remote does not serve counts as carrying the file; it is named in the α decision's
basis and in Chapter 4. If the response does not reach back before T_lo, the step refuses: STOP and ask (a response
of more than one page is not one verbatim file). If no push in it carries the λ selection yet, the step refuses and
writes nothing: save the activity again later the same day. A write that fails (`REFUSED [io_error]`) leaves neither
file, so the step runs again into the same `--out`; if the refusal names a file it could not remove, run the step
into a fresh `--out`. Copy both files it writes to `reports/derived/lambda_selection_push.json` and
`reports/derived/lambda_selection_push_list.json` and commit them **together in ONE commit**: the evidence is settled
once, and a list committed separately, edited later or written again is refused. Once both files are written, never
run the step again for the same push day. Push, then press **Sync now**. Commit and push the evidence before printing
the α default's launch line: until the α default has launched there is no upper bound, so its `entry` and gate derive
the α decision date from the evidence (basis (c)) and refuse without it.

## KD.14 After the run

As §9.9: capture every checkpoint's sha256 on the pod before download. Copy each run's run_meta and telemetry off the
pod at least daily; the copies match the pod-side sha256 (PL-32). Before any pod is terminated, §9.12 applies.

## Sync now

After every push of a launch line, a launched line, a stopped or not_launched line, a decision record, a selection
file or the push evidence, press **Sync now** on both GitHub sources before the next step reads them.
