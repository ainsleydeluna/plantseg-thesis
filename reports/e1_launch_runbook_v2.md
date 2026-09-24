# E1 Launch Runbook v2 (B31c) — supersedes [e1_runpod_launch_runbook.md](e1_runpod_launch_runbook.md)

Operational runbook for launching the **real E1 FP32 training run** on a RunPod GPU box. **[UPDATED 2026-09-24 — B66-prep S2, DL-21]** TRAIN/VAL-only pods (B66 on): follow §9; §3's `preflight_e1.py` gate is the seed-42 path.
**Instructions only** — producing this document involved no training, GPU use, download, or install.

**Why v2:** v1 (B28) predates `--resume`, `--ckpt-interval`, `--num-workers`, `--jsonl-name`,
`--keep-ckpts`, the CUDA compute-capability gate, the dataset check in `verify_env.py`, and
`scripts/preflight_e1.py`. Launching from v1 would launch the wrong command and leave you with no
preemption playbook.

Every flag below is **quoted from `parse_args` in `src/training/train_e1.py`**, not recalled.

---

## 0. The short version

```bash
export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg
cd /workspace/plantseg-thesis
python scripts/preflight_e1.py            # must print VERDICT: GO
python src/training/train_e1.py --real-run --confirm-real-run --init imagenet --ckpt-dir /workspace/e1_ckpts
```

If preflight says **NO-GO**, stop. Do not pass it flags to make it quiet.

---

## 1. GPU selection — hard gate at sm ≤ 9.0

`requirements.lock` pins **torch 2.1.0+cu121**, which ships kernels for **sm_50 … sm_90 only**. A
Blackwell-class device has no compatible kernel and dies at the *first* CUDA launch — after the pod
is provisioned and billing.

`scripts/verify_env.py` now enforces this (`MAX_SM = (9, 0)`) and **FAILS**, not warns.

### DENY — will not run on the pinned stack

| GPU | Compute capability |
|---|---|
| RTX 5090 | sm_120 |
| RTX Pro 6000 (Blackwell) | sm_120 |
| B200 | sm_100 |
| B300 | sm_100 |

### ALLOW

| GPU | Compute capability | Note |
|---|---|---|
| RTX 4090 | sm_89 | v1's reference pod (~$0.34/hr) |
| L40S | sm_89 | |
| RTX 6000 Ada | sm_89 | |
| A6000 | sm_86 | |
| A40 | sm_86 | |
| A100 | sm_80 | |
| H100 | sm_90 | at the ceiling — allowed |

If a pod's GPU is not on either list, run `preflight_e1.py` and believe its verdict rather than
guessing from the marketing name.

---

## 2. Network Volume layout — everything out of the repo

`train_e1.py` **hard-guards** against a checkpoint directory inside the repo
(`_assert_outside_repo`). The repo stays clean; all run output lives on the Network Volume.

```
/workspace/
├── plantseg-thesis/                 # the git checkout — NOTHING is written here at runtime
├── plantseg_data/plantseg/          # dataset root (PLANTSEG_DATA_ROOT)
│   ├── images/{train,val,test}/
│   └── annotations/{train,val,test}/
└── e1_ckpts/                        # --ckpt-dir  (ALL run output)
    ├── last.pt                      # periodic resume point, every --ckpt-interval (2000) iters
    ├── best.json                    # {"best_ckpt": ..., "best_val_miou_all_class": ...}
    ├── e1_student_best_iter*.pt     # rolling best checkpoints, --keep-ckpts (3) retained
    └── e1_telemetry.jsonl           # append-mode telemetry, resume-safe
```

**Disk budget [INFERRED from a MEASURED 24 MB checkpoint]:** `last.pt` ≈ 24 MB, each best ≈ 24 MB,
retention 3 → **≈ 100 MB steady state**. Telemetry: 80,000 train rows + 20 val rows (each val row
carries a 116-element IoU array) → **≈ 15–25 MB**. Budget 1 GB and you will not think about it again.

---

## 3. Launch gate

### 3.1 Code-gated — `scripts/preflight_e1.py` must print `VERDICT: GO`

Runs in order, stops at first hard failure:

| # | Stage | What it proves |
|---|---|---|
| 1 | `verify_env.py` | platform, pinned versions, **sm ≤ 9.0 capability gate**, **dataset root + all six split dirs + per-split counts vs `SPLIT_SIZES`** |
| 2 | `smoke_dataloader.py` | one train + one val batch: shapes, dtypes, label range |
| 3 | `smoke_aug_stochasticity.py` | augmentation varies across epochs · val bit-identical · seeds reproduce across processes · **R5 pinned-stack seed comparison** |
| 4 | `train_e1.py --dry-run` | end-to-end scaffold, asserting **6/6 checks exercised, 0 skipped** |

Stage 4 checks the **coverage count**, not just the exit code: a `NOOP` or a reduced-coverage
dry-run also exits 0 and also says `PASS`, and neither is a launch gate.

**Stage 3 is the R5 re-measurement.** It compares the observed per-sample augmentation seed sequence
against the B30 §9 development-stack baseline:

```
[[1608637542, 1273642419], [1935803228, 787846414]]
```

`MATCH` → the pinned stack draws the development sequence and every seed-dependent artifact under
`reports/` stays valid. `DEVIATION` → **NO-GO**: those artifacts describe a schedule this pod will
not run and must be regenerated here before they can be cited.

### 3.2 NOT code-gated — the operator's responsibility

- **SegNeXt teacher checkpoint is absent.** `weights/` is empty and `docs/teacher_init_source.md`
  still carries `NEED_TO_CONFIRM` for URL / SHA256 / date / init-test result. **Blocks E2/E3, not
  E1** — E1 has no teacher.
- **ImageNet backbone may not be cached.** `--init imagenet` loads
  `MobileNet_V3_Large_Weights.IMAGENET1K_V2` from the torch-hub cache, else downloads. If the pod is
  offline, pre-stage it. `verify_env.py` reports `imagenet_cached`.

---

## 4. The real-run command

```bash
export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg
cd /workspace/plantseg-thesis

python src/training/train_e1.py \
  --real-run \
  --confirm-real-run \
  --init imagenet \
  --ckpt-dir /workspace/e1_ckpts \
  --seed 42 \
  --log-every 50
```

Everything else is left at its default **on purpose** — the defaults are the locked recipe:

| Flag | Default when omitted | Source |
|---|---|---|
| `--batch-size` | **16** | `E1_STUDENT["batch_size"]` |
| `--max-iters` | **80000** **[UPDATED 2026-09-24 — B66-prep S3]** = `--iterations`; a real run refuses any other value | `E1_STUDENT["iterations"]` |
| `--iterations` **[UPDATED 2026-09-24 — B66-prep S3]** | **80000**; 160000 only for the AM-16 item-3 seed-42 control | sets the poly horizon and the real-run length |
| `--val-interval` | **4000** | `E1_STUDENT["val_interval"]` |
| `--max-val-batches` | `None` → **full val set** (846 imgs, 53 batches) | real-run path |
| `--num-workers` | **`min(cpu_count-2, 12)`** **[UPDATED 2026-09-24 — B66-prep S3]** B66: pass `--num-workers 12` explicitly (seed 42 resolved to 12, and the augmentation stream depends on it) | B31-5 |
| `--ckpt-interval` | **2000** | B31-2 |
| `--keep-ckpts` | **3** | B31-9 |
| `--jsonl-name` | **`e1_telemetry.jsonl`** | B31-7 |
| `--grad-clip-norm` | `None` → **unclipped** | D2 / D-A documented deviation |
| `--device` | `cuda` if available, else **hard refusal** | real run aborts on CPU |
| `--seed` | 42 | shown explicitly above for the log |

**Do not pass `--max-iters`, `--batch-size` or `--val-interval`.** They are locked; passing them
invites a typo that silently changes the recipe. **[UPDATED 2026-09-24 — B66-prep S3]** A real run refuses a `--max-iters` that differs from its schedule. The 160,000-iteration control is `--iterations 160000`; `--max-iters 160000` alone is refused (before S3 it would have trained iterations 80,001–160,000 at LR 0).

**Record `num_workers` with the seed.** It is reproducibility-relevant — see
[IMPLEMENTATION_CONTRACT.md](../docs/IMPLEMENTATION_CONTRACT.md) §B6. It is written into the
`run_meta` line of the JSONL automatically; do not rely on memory.

### Run it detached

An 80k run outlives an SSH session.

```bash
mkdir -p /workspace/e1_ckpts
nohup python src/training/train_e1.py \
  --real-run --confirm-real-run --init imagenet \
  --ckpt-dir /workspace/e1_ckpts --seed 42 --log-every 50 \
  > /workspace/e1_ckpts/e1_stdout.log 2>&1 &
echo $! > /workspace/e1_ckpts/e1.pid
```

---

## 5. Preemption playbook — read this one at 3am

**Symptom:** pod reclaimed, container restarted, SSH died, process killed. Checkpoints and telemetry
survive on the Network Volume.

### Step 1 — find where it stopped

```bash
cd /workspace/plantseg-thesis
export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg
python -c "import torch; ck=torch.load('/workspace/e1_ckpts/last.pt', map_location='cpu', weights_only=False); print('iter', ck['iter'], '| best', ck['best_val_miou_all_class'])"
cat /workspace/e1_ckpts/best.json
```

`last.pt` is written atomically (`.tmp` + `os.replace` in the same directory), so a kill mid-write
**cannot** corrupt it. If you see a stray `last.pt.tmp`, that is the interrupted write — delete it;
`last.pt` itself is the previous good state.

For an **official** run this step is now forensics, not a prelude to resuming: it records how far the
attempt got before you discard it. Step 2 explains why.

### Step 2 — discard the partial run and relaunch from iteration 0

**Official runs are never resumed** (`AGENTS.md` § E1 invariants). Move the dead run's directory
aside first. This is the step that is easy to skip at 3am and the one that silently corrupts the
record:

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
mv /workspace/e1_ckpts /workspace/e1_ckpts.aborted-$ts
mkdir -p /workspace/e1_ckpts
[ -z "$(ls -A /workspace/e1_ckpts)" ] || { echo "STOP: not empty"; exit 1; }
echo "clear"
```

That guard must print `clear`, and it exits non-zero if it does not — a guard that reports failure
without signalling it is no guard at all. `train_e1.py:205-209` opens the telemetry JSONL in
**append** mode — deliberately, so a *resumed* run continues one file — which means relaunching into
the dead run's directory appends the new run's rows onto the old with iterations restarting at 1, in
a single file, with nothing to signal it. `prune_checkpoints` (`keep_ckpts=3`) would likewise mix
the two runs' best-checkpoints.

`mkdir -p` is required, not tidiness: the shell opens the `>>` redirect target below **before**
Python starts, so `resolve_ckpt_dir()` creating the directory comes too late. The old resume block
worked only because resuming implies the directory already exists — exactly the assumption
discard-and-relaunch breaks.

Then relaunch with the original command, byte-identical, with no `--resume`:

```bash
nohup python src/training/train_e1.py \
  --real-run --confirm-real-run --init imagenet \
  --ckpt-dir /workspace/e1_ckpts \
  --seed 42 --log-every 50 \
  >> /workspace/e1_ckpts/e1_stdout.log 2>&1 &
```

Before launching, confirm `--seed` matches the aborted run's `run_meta` row rather than your memory
of it: `head -1 /workspace/e1_ckpts.aborted-$ts/e1_telemetry.jsonl`. Relaunching the same stage under
a different seed produces a run that looks official and is not: it will pass every gate, write a
well-formed telemetry record, and silently duplicate or omit a seed in the three-seed set. Nothing
downstream detects it — the seed identity check in `preflight_e1.py` ~~verifies the RNG sequence for
whatever seed it is given~~ **[UPDATED 2026-09-24 — B66-prep S2]** is seed-42-specific (`smoke_aug_stochasticity.py` hard-codes seed 42): a pinned-stack regression check, not a per-seed check, and it never checks that the seed is the one the run plan called for.

Keep the aborted directory. It holds the telemetry and checkpoints of a run that happened and is the
evidence for how far the attempt got. Do not commit it (`ai_guardrails.md` §2).

### Step 3 — why official runs don't resume, and when `--resume` is still right

```
[resume] WARNING: data-order continuity is NOT restored. ...
         A resumed run is NOT bitwise-identical to an uninterrupted one.
```

**What IS restored:** the LR curve exactly (~~checkpoint LR matches the analytic 80,000-iteration
`PolynomialLR` curve at full float64~~ **[UPDATED 2026-09-24 — B66-prep S3]** the restored scheduler continues the chained `PolynomialLR` recursion over the checkpoint's own horizon; `--resume` refuses a checkpoint whose horizon differs from `--iterations`), all RNG streams, best-mIoU tracking, and LR-monotonicity
checking **across** the resume boundary (`last.pt` carries `prev_lr`, so a *k*-segment run leaves
zero unverified transitions).

**What is NOT restored:** position within the current epoch. The loop consumes an infinite
`cycle(train_loader)` and the sampler position is not persisted, so post-resume sample order differs
from an uninterrupted run.

**Why that settles it for an official run:** E1 is the baseline every later stage is measured
against, and `scripts/preflight_e1.py` hard-gates on seed-sequence identity. A resumed official run
would carry a permanent asterisk to save compute worth about **$4.40** at Community rates — the full
13-hour run at ~$0.34/hr `[INFERRED]`. Relaunching costs only that.

**`--resume` remains correct for non-official work** — debugging, rehearsals, and the *k*-segment
resume verification behind `docs/open_questions.md` D27. There D26's disclosure requirement still
applies: report a resumed run as resumed. `--resume` is not removed from `train_e1.py`.

### Common preemption mistakes

| Mistake | Consequence |
|---|---|
| **Resuming an official run** with `--resume` | not bitwise-identical to an uninterrupted run; carries a permanent asterisk and is no longer a clean baseline. Forbidden — `AGENTS.md` § E1 invariants |
| **Relaunching into the dead run's `--ckpt-dir`** | `train_e1.py:205-209` appends telemetry, so two runs interleave in one JSONL with iterations restarting at 1, silently. Move the directory aside first (Step 2) |
| **Relaunching under a different `--seed`** | produces a run that looks official and is not; passes every gate and silently duplicates or omits a seed in the three-seed set. Confirm the seed from the aborted run's `run_meta` row, not memory |
| **Omitting `--resume` on a NON-official run** (debugging, rehearsal) | starts from iteration 1 silently — you lose the partial run. For official runs this is the required behaviour; see Step 2 |
| Deleting the aborted directory | destroys the record of how far the attempt got; keep it as `e1_ckpts.aborted-<ts>` |
| Resuming from `e1_student_best_iter*.pt` | carries no RNG state; for a **non-official** resume use `last.pt` |
| Changing `--num-workers` on relaunch | changes the realized augmentation sequence; keep every flag identical to the original launch |
| Overwriting the log with `>` instead of `>>` | destroys the pre-preemption stdout |
| `RESULT: NOOP (already complete at iter N)` | `last.pt` is already at/past `--max-iters`; nothing ran. Not an error |

---

## 6. Live monitoring — the first 500 iterations

Telemetry is one JSON object per line at `/workspace/e1_ckpts/e1_telemetry.jsonl`.

```bash
# throughput, once ~200 iters exist
python - <<'EOF'
import json
rows=[json.loads(l) for l in open('/workspace/e1_ckpts/e1_telemetry.jsonl')]
tr=[r for r in rows if r['event']=='train'][10:210]   # skip warm-up
sps=[r['samples_per_sec'] for r in tr if r['samples_per_sec']]
sps.sort()
print('samples/sec  median', round(sps[len(sps)//2],2), ' p05', round(sps[len(sps)//20],2))
print('it/s median', round(sps[len(sps)//2]/16,3))
print('projected hours for 80k iters:', round(80000/(sps[len(sps)//2]/16)/3600,2))
EOF

# loss sanity
tail -5 /workspace/e1_ckpts/e1_stdout.log
```

### What to watch, and when to abort

| Signal | Healthy | Abort threshold |
|---|---|---|
| `loss` finite | always | any `NaN`/`Inf` → the real run **already self-aborts** (`train_e1.py` raises on non-finite loss in real mode) |
| `loss` trend | falls from ~6 over the first few thousand iters | flat at its initial value after 4,000 iters → stop and investigate the LR/init |
| `lr` | monotonically non-increasing, starts `9.99988750e-03` | any increase → scheduler wiring broken |
| `samples_per_sec` | stable after ~50 iters | sustained < 50% of the first-200-iter median → check whether workers are starving |
| first `val` at iter 4000 | `all_class_miou > 0`, `n_eligible_classes` in the tens | `all_class_miou == 0.0` at iter 8000 → stop |
| GPU utilisation | high, steady | persistently low with high CPU → data-bound; raise `--num-workers` **on a fresh run**, not mid-run |

### Throughput — MEASURED vs INFERRED

- **88.3 ms/sample is a MEASURED number from the Windows development box** (B30 P4), on a CPU-only
  machine with a different core count, different storage, and a different NumPy/torch build.
- **It is NOT a pod prediction.** Any wall-clock estimate derived from it — including the ~7.9 h
  figure quoted during the B30/B31 review — is **INFERRED and cross-platform**, and must not be used
  for scheduling.
- **Re-measure on the pod** from `samples_per_sec` in the first 200 JSONL rows (snippet above)
  before trusting any duration estimate or committing to a multi-seed schedule.
- For reference: **335 iterations/epoch** (`drop_last=True`, 5,367 / 16) and **238.806 epochs** at
  80,000 iterations. Both MEASURED, both platform-independent.

---

## 7. Pre-launch checklist

- [ ] Pod GPU is on the **ALLOW** list (§1)
- [ ] `git rev-parse HEAD` matches the remote `master` tip read *at this moment*
- [ ] `export PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg`
- [ ] `pip install -r requirements-e1.txt` (student-only; **no** mmcv/mmseg)
- [ ] `mkdir -p /workspace/e1_ckpts` on the Network Volume
- [ ] `python scripts/preflight_e1.py` → **`VERDICT: GO`**
- [ ] Stage 3 reported **`seed sequence MATCH`**
- [ ] ImageNet backbone cached, or the pod has network
- [ ] Launch detached (§4), PID recorded
- [ ] After ~200 iters: re-measure throughput from the JSONL (§6)
- [ ] `docs/reference/reference.pdf` still unstaged — never commit it

---

## 8. Related documents

| Document | Role |
|---|---|
| [pre_e1_launch_audit.md](pre_e1_launch_audit.md) | B30 forensic audit — the evidence base, incl. §9 cross-process reproducibility |
| [b31_e1_fixes.md](b31_e1_fixes.md) | B31 fixes, acceptance tests, measured timings |
| [b31c_closeout.md](b31c_closeout.md) | B31c — V1 verification, commits, preflight |
| [../docs/IMPLEMENTATION_CONTRACT.md](../docs/IMPLEMENTATION_CONTRACT.md) | §B6 seeds, `num_workers`, resume deviation, B31 defaults |
| [../docs/open_questions.md](../docs/open_questions.md) | D25 `num_workers` · D26 resume non-identity · D27 check coverage |
| [e1_runpod_launch_runbook.md](e1_runpod_launch_runbook.md) | **v1, SUPERSEDED** — history only |

---

## 9. B66 — TRAIN/VAL-only launch path (DL-21) [added 2026-09-24, B66-prep S2]

Applies to every real E1 run from B66 on (seeds 43/44 and the longer-schedule E1). §1–§8 remain the
seed-42 record; where they conflict with this section, this section wins. From this commit on,
`train_e1.py` refuses a real run on any root that holds a TEST surface, so the §1–§8 launch path (the
seed-42 volume root) cannot start a real run.

**9.0 Scope.** On these pods never run `verify_env.py`, `preflight_e1.py`, `verify_plantseg_dataset.py`,
`smoke_metrics.py` or `verify_class_weights_pod.py`, nor steps 6, 7, 10 or 12 of the RunPod pre-flight
template: they list TEST or write into the repository. The gate in 9.5 replaces them.

**9.1 Pod and shell.** Start the pod from the pinned image by digest (DL-21), with enough `/dev/shm` for 12
DataLoader workers (Docker's 64 MB default crashes them). In the one shell that will run the gate and the
launch:

```bash
export PLANTSEG_IMAGE_DIGEST=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
unset PLANTSEG_GIT_COMMIT     # the image bakes f77d05d7, and train_e1.py prefers it over the checkout
```

**9.2 Checkout.** A DL-19 partial clone at the pushed pin: `--filter=blob:none --no-checkout`;
`core.sparseCheckout true`, `core.sparseCheckoutCone false`, patterns `/*` and `!/docs/reference/`;
`checkout --detach <pin>`; remote URL invalidated afterwards. Step 0: HEAD equals the pin, the protected
reference file is absent, and the scoped status is clean. Then `export PYTHONPATH=<clone>` (the image's
baked `PYTHONPATH` points at a stale source copy).

**9.3 Data.** Stage the TRAIN/VAL payload at a NEW path and check its sha256 before extracting. Never use
or list the seed-42 volume root, which contains TEST. `export PLANTSEG_DATA_ROOT=<absolute staged root>`.

**9.4 ImageNet backbone.** Pre-stage `mobilenet_v3_large-5c1a4163.pth` (22,132,113 B, sha256
`5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`) into
`$(python -B -c 'import torch; print(torch.hub.get_dir())')/checkpoints/` by a verified fetch or transfer,
each separately approved. There is no in-run download: the gate refuses a missing or different file. Keep
`TORCH_HOME` unchanged from here on; the launch block pins it.

**9.5 Gate.** `<evidence>` is an existing directory outside the clone, the ckpt dir and the data root;
`<D>` is absent or empty. `<n>` numbers the attempt (1, 2, …): every gate run gets new evidence names,
because the gate refuses an existing `--record` and `tee` would overwrite an earlier log.

```bash
python -B scripts/preflight_e1_trainval.py gate --seed <S> --ckpt-dir <D> --expect-head <pin> \
  --record <evidence>/preflight_<S>_<n>.json 2>&1 | tee <evidence>/preflight_<S>_<n>.log
```

It must print `VERDICT: GO`. Stages, first FAIL stops: `arguments` → `data_isolation` (TEST refused by
exact path and never listed; TRAIN/VAL counts and stem pairing) → `repo_state` (pin, `885523a` floor,
scoped status, `PLANTSEG_GIT_COMMIT` unset, sparse config) → `module_provenance` → `class_weights` →
`smoke_loss` → `image` (`preflight_environment.py --mode gpu`) → `cuda` → `imagenet_backbone` →
`smoke_loader_seed` → `smoke_dataloader` → `seed_sequence_R5` → `dry_run` → `repo_unchanged`.
`--rehearsal` exercises the gate off-pod and never prints a launch block.

**9.6 Launch.** Save the printed block as `<evidence>/launch_<S>_<n>.sh` and run it with `bash` from the
same shell. It unsets `PLANTSEG_GIT_COMMIT`, pins `PYTHONPATH`, `PLANTSEG_DATA_ROOT`, `PLANTSEG_IMAGE_DIGEST`
and `TORCH_HOME`, and passes `--num-workers 12` (the augmentation stream depends on it; seed 42 ran 12)
and `--log-every 50`. It never passes `--resume`, `--max-iters`, `--batch-size`, `--val-interval`,
`--max-val-batches` or `--device`. `train_e1.py` itself refuses a real run on a root that fails the
TRAIN/VAL-only check.

**9.7 Verify the launch.** Within 5 minutes:
`python -B scripts/preflight_e1_trainval.py check-run-meta --ckpt-dir <D> --seed <S> --expect-head <pin>`.
It compares the `run_meta` row with the seed-42 row (exempt: seed, git_head, wall_clock, gpu_name as a
warning, and the profile's max_iters/poly_horizon). On FAIL, kill the run and relaunch into a fresh `<D>`
from 9.5.

**9.8 Preemption.** Move the dead `<D>` aside as in §5 Step 2 (do not run §5 Step 1 or §5's relaunch
command: they name the seed-42 paths), then repeat from 9.5 with a FRESH `<D>` and the next `<n>`. If the
container restarted, the 9.1 shell is gone: redo 9.1–9.3 in the new shell (and 9.4 if the backbone was not on
the volume; a re-fetch needs its own approval) before 9.5.

**9.9 After the run.** Capture every checkpoint's sha256 on the pod before download (B52 N9).

**9.10 Longer-schedule control (AM-16 item 3, DL-27) [added 2026-09-24, B66-prep S3].** E1 at seed 42 with
160,000 iterations: pass `--profile e1_160k` to the gate (9.5) and to `check-run-meta` (9.7). Its launch
block adds `--iterations 160000` (the poly horizon and the run length); it never passes `--max-iters`, which a
real run refuses. `check-run-meta` then requires `max_iters` == `poly_horizon` == 160000, so a launch that
forgot `--iterations` (a valid-looking duplicate 80k run) is caught. Budget: about 47 h at the seed-42 rate
(1.0567 s/iter), 40 validations, 80 `last.pt` writes. No resume (AGENTS.md): an interruption forfeits the
attempt, up to about 47 h; relaunch from 9.5 into a fresh `<D>`.
