# E1 Launch Runbook v2 (B31c) — supersedes [e1_runpod_launch_runbook.md](e1_runpod_launch_runbook.md)

Operational runbook for launching the **real E1 FP32 training run** on a RunPod GPU box.
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
| `--max-iters` | **80000** | `E1_STUDENT["iterations"]` |
| `--val-interval` | **4000** | `E1_STUDENT["val_interval"]` |
| `--max-val-batches` | `None` → **full val set** (846 imgs, 53 batches) | real-run path |
| `--num-workers` | **`min(cpu_count-2, 12)`** | B31-5 |
| `--ckpt-interval` | **2000** | B31-2 |
| `--keep-ckpts` | **3** | B31-9 |
| `--jsonl-name` | **`e1_telemetry.jsonl`** | B31-7 |
| `--grad-clip-norm` | `None` → **unclipped** | D2 / D-A documented deviation |
| `--device` | `cuda` if available, else **hard refusal** | real run aborts on CPU |
| `--seed` | 42 | shown explicitly above for the log |

**Do not pass `--max-iters`, `--batch-size` or `--val-interval`.** They are locked; passing them
invites a typo that silently changes the recipe.

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

### Step 2 — resume

```bash
nohup python src/training/train_e1.py \
  --real-run --confirm-real-run --init imagenet \
  --ckpt-dir /workspace/e1_ckpts \
  --resume /workspace/e1_ckpts/last.pt \
  --seed 42 --log-every 50 \
  >> /workspace/e1_ckpts/e1_stdout.log 2>&1 &
```

Keep every other flag identical to the original launch. `--resume` restores model, optimizer,
scheduler, and all RNG state (`torch`, `torch.cuda`, `numpy`, python `random`, DataLoader
generator), and continues from `iter + 1`.

### Step 3 — read the resume warning and know what it means

```
[resume] WARNING: data-order continuity is NOT restored. ...
         A resumed run is NOT bitwise-identical to an uninterrupted one.
```

**What IS restored:** the LR curve exactly (checkpoint LR matches the analytic 80,000-iteration
`PolynomialLR` curve at full float64), all RNG streams, best-mIoU tracking, and LR-monotonicity
checking **across** the resume boundary (`last.pt` carries `prev_lr`, so a *k*-segment run leaves
zero unverified transitions).

**What is NOT restored:** position within the current epoch. The loop consumes an infinite
`cycle(train_loader)` and the sampler position is not persisted, so post-resume sample order differs
from an uninterrupted run.

**What that means for the thesis:** a resumed run is a valid E1 run, but it is not a byte
reproduction of an uninterrupted one. **If a resumed run produces a headline result, report it as
resumed.** See `docs/open_questions.md` D26.

### Common preemption mistakes

| Mistake | Consequence |
|---|---|
| Omitting `--resume` | starts from iteration 1, silently — you lose everything and won't notice for hours |
| Resuming from `e1_student_best_iter*.pt` | those carry no RNG state; **resume from `last.pt`** |
| Changing `--num-workers` on resume | changes the realized augmentation sequence mid-run |
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
