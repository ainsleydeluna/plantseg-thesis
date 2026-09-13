# B52 — E1 seed 42: completion record

**Type:** run completion record + provenance archive. **Status:** run **COMPLETE**, 80,000/80,000
iterations. **Date:** 2026-09-12.
**Run:** E1 FP32 MobileNetV3-Large + LR-ASPP student, `seed=42`, the first of the three planned
E1 seeds.
**Source tree:** `f77d05d7b35187bf0da7e7b94a629549fe2e1c05`, resolved at runtime by
`git rev-parse` (`source=git_checkout`), image `sha256:b80b645d…866aaf`.
**Pod:** RunPod Secure, NVIDIA A40, `CA-MTL-1`, terminated 2026-09-12. Every on-pod artifact
quoted here was retrieved before termination; the pod-side originals no longer exist.

> ## ⚠ VALIDATION ONLY — the test split has not been evaluated
>
> Every mIoU figure in this report is **validation-split**. `docs/EVALUATION_CONTRACT.md` permits
> exactly one test-set evaluation per stage, and E1's has not been spent. No number here is a
> test result, and none may be reported as one. §5.3 pre-registers, before any test read, what
> the test number is expected to do relative to the validation number and why.

---

## 1. The result

| Field | Value | Source |
|---|---|---|
| Best all-class val mIoU | **0.36314016580581665** | `best.json`, checkpoint `best_val_miou_all_class` |
| Iteration of best | **80000** (the final validation) | checkpoint `iter` |
| Disease-only mIoU at 80k *(PROVISIONAL)* | 0.35861 | telemetry `disease_only_miou_PROVISIONAL` |
| Checkpoint | `e1_student_best_iter80000.pt`, 23,734,564 B | laptop `ls -l` |
| Student parameters | 2,933,688 | `[student]` line |
| Telemetry rows | 80,021 = 1 `run_meta` + 80,000 `train` + 20 `val` | laptop read-back |

`best.json`, verbatim (119 B):

```json
{
  "best_ckpt": "/workspace/e1_ckpts/e1_student_best_iter80000.pt",
  "best_val_miou_all_class": 0.36314016580581665
}
```

Checkpoint keys, read back on the laptop after transfer:
`['best_val_miou_all_class', 'iter', 'model_state_dict', 'num_classes', 'optimizer_state_dict',
'scheduler', 'scheduler_state_dict']`. The file loads, `iter == 80000`, and its recorded mIoU is
bitwise equal to `best.json`'s and to `max(all_class_miou)` over the 20 telemetry validation rows.
Three independent records agree. `[MEASURED]`

The checkpoint criterion was **all-class mIoU**, as the E1 invariants require. Disease-only mIoU
was computed and logged at every validation but never selected on. `[MEASURED]`

---

## 2. Provenance chain

| Link | Value | How established |
|---|---|---|
| Source commit | `f77d05d7b35187bf0da7e7b94a629549fe2e1c05` | `[provenance] … (source=git_checkout)` in `e1_stdout.log` |
| Provenance source | `git_checkout` — a live `git rev-parse HEAD`, **not** the baked env var | same line; see §2.1 |
| Working tree | clean at launch | `git status --porcelain \| head` printed nothing (step0.log) |
| Image | `sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf` | `PLANTSEG_IMAGE_DIGEST`, echoed in step0.log and in `[provenance]` |
| Dataset | PlantSeg from Zenodo, md5 `9358a66dff88cdd15c4fe009763c40a3` | verified on-pod against Zenodo's published record |
| Data root | `/workspace/plantseg_data/plantseg` | `PLANTSEG_DATA_ROOT`, echoed in step0.log |
| Index sizes | `train_index=5367 val_index=846` | `[data]` line |
| ImageNet init | `mobilenet_v3_large-5c1a4163.pth`, 22,132,113 B, sha256 `5c1a4163…7997` | `imagenet_capture.json`; `url_hash_prefix` `5c1a4163` matches the file's own leading hash digits |
| Pre-flight | `VERDICT: GO` — 5/5 stages | `preflight.log` |
| Seed sequence | `MATCH` against the B30 §9 baseline `[[1608637542, 1273642419], [1935803228, 787846414]]` | `preflight.log` stage 4 |
| `--resume` | never used; `resumed_from: null` | launch command; `run_meta` |

Launch command, verbatim:

```bash
nohup python src/training/train_e1.py \
  --real-run --confirm-real-run --init imagenet \
  --ckpt-dir /workspace/e1_ckpts --seed 42 --log-every 50 \
  > /workspace/e1_ckpts/e1_stdout.log 2>&1 &
```

`--ckpt-dir` is outside the repository, as `train_e1.py`'s hard guard requires; the log confirms
`[ckpt] dir=/workspace/e1_ckpts (verified OUTSIDE repo)`. `--num-workers` was **not** passed, so
the B31-5 default `min(max(cpu_count - 2, 1), 12)` applied and resolved to **12** (§3.3).

### 2.1 `PLANTSEG_GIT_COMMIT` — read by the training path, and deliberately neutralised

**The training path does read it.** `src/training/train_e1.py:245`, inside `_git_provenance`:

```python
    env = os.environ.get("PLANTSEG_GIT_COMMIT", "").strip()
    if env:
        return env, "image_env"
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, …)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip(), "git_checkout"
```

step0.log records `PLANTSEG_GIT_COMMIT=[<unset>]` in both passes of the checkout block, because
the block was run with a preceding `unset PLANTSEG_GIT_COMMIT` placed deliberately outside the
brace group. That was the intended configuration, and **this is not a provenance gap — it is the
stronger of the two available outcomes.** The function's own docstring (`:233-237`) says why:
`image_env` "proves what was BUILT… It cannot detect a working tree modified after the build",
whereas `git_checkout` "proves both what was built and the tree the code is actually running
from." Unsetting the variable forces the fall-through to the live `git rev-parse`, and the run
duly recorded `source=git_checkout`. The complementary check — that the live tree was unmodified
— is satisfied by `git status --porcelain` printing nothing at launch. Closed. `[MEASURED]`

For contrast: the failed launch-1 run on the 4090 (B48) recorded `source=image_env`, because the
`unset` was not in place there. Same commit, weaker attestation.

`src/eval/artifacts.py:153` documents that the evaluation-artifact path deliberately does **not**
fall back to this variable. Nothing else in the training path reads it.

### 2.2 FLAG — the checkout block ran twice, and the second clone failed

step0.log contains the checkout sequence **twice**, with identical results both times
(`HEAD=f77d05d7…`, `CHECKOUT OK`, `smoke_loss refs: 0`, `stage fns: 5`,
`PLANTSEG_GIT_COMMIT=[<unset>]`). The second pass is preceded by:

```text
already a git checkout — leaving alone
fatal: destination path '/workspace/plantseg-thesis' already exists and is not an empty directory.
CLONE FAILED
```

**The trained commit is not in doubt.** It is corroborated independently of step0.log by the
`[provenance]` line in `e1_stdout.log`, which the training process resolved for itself at launch.

**Was the duplication an idempotent retry by design, or unintended re-entry?** Partly
determinable from the block, and the honest answer is *neither, exactly*:

- **From the script — the clone is not idempotent.** The block guards only the *move* that
  preserves the image's B41 source copy (`if [ -d plantseg-thesis/.git ]; then echo "already a
  git checkout — leaving alone"; else mv … fi`). The `git clone` that follows is unconditional
  and unguarded. So on any second execution inside the same container the guard fires, the clone
  then hits a non-empty directory, and `fatal:` / `CLONE FAILED` is the *expected* output of
  re-running the block — not a fault, but not designed-for either. `[MEASURED — from the script]`
- **From the script — the verification is idempotent, which is why it was harmless.** The
  `git checkout --detach <literal sha>` → `git rev-parse HEAD` → string-compare-against-literal
  sequence is unconditional and re-entrant. It ran and passed on both passes regardless of the
  clone's outcome. The outcome was therefore correct by verification, not by luck.
  `[MEASURED — from the script]`
- **From the script — the redirect is `tee -a`**, which is why the duplication is visible at all;
  a second run appends rather than overwrites. `[MEASURED — from the script]`
- **The script contains no retry construct** — no loop, no re-exec, no conditional re-run. The
  duplication was therefore produced at operator level, not by the block. `[MEASURED]`
- **Why it was re-run is not determinable** from the script or from any saved artifact. The pass
  captured in the session transcript already shows the guard message and `CLONE FAILED`, i.e. it
  was itself the *second* pass; the first is not recorded anywhere that survives. Whether the
  re-run was a deliberate retry (there was at least one SSH drop during this pod's session) or an
  accidental repeat cannot be recovered. **Stated as undetermined rather than guessed.**
  `[UNVERIFIED]`

Queued, non-governed: guard the `git clone` with the same `.git` test that guards the `mv`, so a
re-run prints "already a checkout" and skips the clone instead of emitting a `fatal:` line that
reads like a failure. Also note the block wrote to `step0.log` via `tee -a` while the runbook text
names `setup.log`; `setup.log` was never created on this pod, and the tarball build later failed
on `tar: setup.log: Cannot stat` for that reason (the tarball is otherwise complete — §8).

---

## 3. The pod — ch3 RunPod field set

Quoted from `step0.log`, the capture made at pod start. Fields absent from that file are marked
absent rather than filled from memory.

```text
start_time_utc : 2026-09-11T11:59:01Z
pod_host       : 1166491d9cf4
RUNPOD_POD_ID  = 6x7mmai2986tjt
RUNPOD_DC_ID   = CA-MTL-1
RUNPOD_GPU_NAME = NVIDIA+A40
RUNPOD_GPU_COUNT = 1
RUNPOD_CPU_COUNT = 9
RUNPOD_MEM_GB  = 50
RUNPOD_VOLUME_ID =
gpu            : NVIDIA A40, 46068 MiB, 8.6, 570.195.03
image_digest   : sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
data_root      : /workspace/plantseg_data/plantseg
rate           : $0.49/hr Secure A40 (operator-recorded)
Filesystem                    Size  Used Avail Use% Mounted on
overlay                        25G   16M   25G   1% /
mfs#ca-mtl-1.runpod.net:9421  965T  752T  214T  78% /workspace

=== STEP 0 — UVM + CUDA gate ===
open /dev/nvidiactl OK
open /dev/nvidia-uvm OK
open /dev/nvidia-uvm-tools OK
cuInit rc       0
is_available    True
gpu             NVIDIA A40 44.4 GiB sm_86
alloc           OK
cpu_count       96 | cgroup quota unknown | torch_threads 48
num_workers     12
VERDICT: GPU USABLE — proceed
```

`RUNPOD_VOLUME_ID` is empty: `/workspace` was the pod's own network-backed scratch, not a
persistent volume, so everything on it died with the pod. That is why §8's rescue matters.

Three fields are reported twice, with different values. None is smoothed over.

### 3.1 VRAM — 46068 MiB and 44.4 GiB are different quantities, and the code says so

The discrepancy is **explained**, not unexplained, and the step-0 script settles it:

- `46068 MiB` comes from `nvidia-smi --query-gpu=…,memory.total,…` — the driver's total
  framebuffer. 46068 MiB = **44.988 GiB**.
- `44.4 GiB` comes from `print(… f'{pr.total_memory/2**30:.1f} GiB' …)` where
  `pr = torch.cuda.get_device_properties(0)` — CUDA's `totalGlobalMem`, i.e. what a process can
  actually allocate, formatted to one decimal.

So it is the **total-vs-torch-visible** distinction, from two different APIs in the same file.
The ~572 MiB difference is driver/context reserve. `[MEASURED]`

**On the `44.43 GiB` figure carried in conversation:** it is *not* unevidenced, and it should not
simply be dropped — it is the same `torch.cuda.get_device_properties().total_memory` quantity at
two decimals, printed by `grad_probe.log`'s P0 header on this same pod (`gpu=NVIDIA A40
total=44.43 GiB`). It is absent from *step0.log*, which is what the field set above quotes, and
that is the correct reason to keep it out of §3's field set — but the number itself has a source.
Recording it as unevidenced would have been wrong. `[MEASURED]`

For the record: 44.43 GiB → 45,496 MiB; 46,068 − 45,496 = 572 MiB reserve.

### 3.2 Billed rate — operator-recorded, and the tier differs from the infra record

`$0.49/hr` is an **operator-recorded** figure — the step-0 script echoes a hard-coded string, it
does not query RunPod — and the line names **Secure** A40. `docs/runpod_environment.md`'s ~$0.35/hr
A40 figure refers to **Community**. The gap is therefore tier, most likely not an anomaly, but
neither figure is machine-captured for this pod. `[MEASURED — that the string is hard-coded;
INFERRED — that tier explains the gap]`

Measured cost of the training window alone: 23.02 h × $0.49 = **$11.28**. Pod lifetime was longer
— pod start 11:59:01Z on 2026-09-11, last recorded pod command 13:09 UTC on 2026-09-12, i.e.
**≥25.2 h → ≥ ~$12.33** — and the exact termination time was not recorded.
`[MEASURED / partly UNVERIFIED]`

**Queued: the Secure-vs-Community decision for the remaining stages.** It is a real decision and
it needs a restated basis. The figures carried in conversation — "~$3.2k Secure versus ~$2.3k
Community" for twelve remaining runs at ~23 h — do **not** follow from that premise: 12 × 23.02 h
at $0.49 is ~$135, at $0.35 is ~$97. A campaign total in the thousands may well be right once the
SegNeXt teacher fine-tune and the E2/E3 KD runs are costed (they are far heavier than E1), but
those assumptions are not written down anywhere, so the estimate must be rebuilt from the measured
per-run cost rather than carried forward. Flagged, not resolved here.

### 3.3 CPU — 9 allocated, 96 seen, 48 threads, 12 workers

| Reading | Value | Source |
|---|---|---|
| Allocation | `RUNPOD_CPU_COUNT = 9` | RunPod env var |
| What Python saw | `cpu_count 96` | `os.cpu_count()` — the host's cores |
| cgroup quota | `unknown` | the step-0 reader could not read `/sys/fs/cgroup/cpu.max` |
| torch intra-op threads | `48` | `torch.get_num_threads()` — half of 96 |
| DataLoader workers | `12` | `min(max(96 - 2, 1), 12)` |

`os.cpu_count()` reports the host's cores, not the cgroup quota, so torch set **48 intra-op
threads on a 9-vCPU allocation** — roughly 5× oversubscription — and `num_workers` saturated at
its cap of 12 regardless of the allocation. `[MEASURED]`

This is a **throughput caveat on §4's 1.036 s/iter**: that figure was measured under
oversubscription, and a correctly-sized pod could be faster or slower. It is **not** a correctness
concern — `num_workers` is recorded with the seed as reproducibility-relevant and was not
overridden — and nothing here is fixed in this report. Queued as a configuration item for every
future pod: read the allocation from `RUNPOD_CPU_COUNT` and set threads from it, and carry the
value into the run record.

---

## 4. Run configuration and throughput

`[budget] max_iters=80000 batch_size=16 val_interval=4000 max_val_batches=None num_workers=12 seed=42`

| Setting | Value |
|---|---|
| Optimizer | SGD, lr 0.01, momentum 0.9, weight_decay 1e-4 |
| Schedule | `PolynomialLR(total_iters=80000, power=0.9)`, final lr `0.00000000e+00` |
| Loss | `CombinedCEDiceLoss(weight=len116, ignore_index=255)`, weight buffer on `cuda:0` |
| Grad clip | disabled — no concrete E1 `max_norm` |
| Classes | 116, background 0, disease 1..115, `ignore_index=255`, `reduce_zero_label=False` |
| Validation | every 4,000 iters over the full val split (`max_val_batches=None`) |
| Checkpointing | best-on-all-class-mIoU; `ckpt_interval=2000`; `keep_ckpts=3` |

### 4.1 Throughput, derived from `wall_clock` deltas

Not asserted — computed from the four 4,000-iteration intervals spanning the last five
validations:

| Interval | Δ wall_clock (s) | s/iter |
|---|---|---|
| 64k → 68k | 4155 | 1.03875 |
| 68k → 72k | 4144 | 1.03600 |
| 72k → 76k | 4137 | 1.03425 |
| 76k → 80k | 4140 | 1.03500 |
| **mean** | **16576 / 16000** | **1.0360** |

80,000 iterations × 1.036 s = 82,880 s = **23.02 h**. `[MEASURED]`

Range **1.03425 – 1.03875 s/iter**. (The "1.0341" lower bound carried in conversation is a
rounding slip of the same 4137 s delta — 4137/4000 = 1.03425. Recorded because this repo records
those, not because it changes anything.)

This is the number every future schedule estimate is built on, subject to §3.3's oversubscription
caveat.

---

## 5. The validation series — and what may not be concluded from it

### 5.1 The last five validations

| iter | all-class mIoU | disease-only (PROVISIONAL) |
|---|---|---|
| 64000 | 0.36178 | 0.35728 |
| 68000 | 0.35709 | 0.35254 |
| 72000 | 0.35599 | 0.35143 |
| 76000 | 0.36042 | 0.35587 |
| 80000 | **0.36314** | 0.35861 |

`cm_batches=53`, `cm_total_px=159279104`, `n_eligible_classes=116` at every one of the twenty
validations. 846 val images at batch 16 → 53 batches. `159,279,104` non-ignored pixels against
`846 × 512 × 512 = 221,773,824` total means **28.2 % of val pixels are `ignore_index=255`** and
are excluded from the confusion matrix — consistent with letterbox padding at 512². `[MEASURED]`

### 5.2 No sufficiency conclusion is drawn

**The curve is not monotonic.** It falls from 0.36178 at 64k to 0.35599 at 72k, then recovers to
0.36314 at 80k. Across those five points the span is **0.00715** (0.72 pp). The net 64k → 80k
change is **+0.00136** — **5.3× smaller than the oscillation in the same window**.

The final 16,000 iterations therefore show oscillation of roughly 0.7 pp with no monotonic trend.
The budget is locked at 80,000 regardless. **No conclusion is drawn about whether 80,000 was
sufficient, and none may be read into the fact that the best checkpoint happens to fall at the
last validation.** An earlier draft of this report claimed the budget "wasn't cutting it short";
that claim was wrong and is retracted here. `[MEASURED]`

### 5.3 Pre-registered expectation for the test split — dated 2026-09-12

Recorded **before** any test-set evaluation, which is what makes it a prediction rather than a
post-hoc explanation.

The reported 0.36314 is the **maximum of 20 validations** on a curve oscillating ~0.7 pp.
Best-of-N selection on a noisy curve is optimistically biased: the selected value overstates the
checkpoint's expected performance by roughly the order of that oscillation.

**Consequence, stated explicitly: E1 test mIoU is expected to come in below 36.31 % even with zero
distribution shift, purely from selection on the maximum.** A test number below the validation
number is therefore the predicted outcome, not evidence of a problem.

This selection bias is the **only** pre-registered reason to expect test below 36.31 %. See §7.3
for why the zero-GT-class penalty is explicitly *not* a second one.

---

## 6. Determinism — the finding, the verdict, and its scope

### 6.1 What the probe compared

A throwaway two-arm probe (`grad_probe.log`, run on this pod before launch; the script was never
added to the repository) compared **repeat passes over fixed inputs within one process**, not
across runs:

- **Probe A** — weighted mean-reduced CE on a **fixed `logits0` tensor** and fixed target, 10
  passes. Isolates the loss computation; the backbone forward is not in the loop.
- **Probe B** — a full E1 step (CE + Dice, backward, no optimizer step), 10 passes. Includes the
  forward.

Because the input and target are fixed, the non-ignored pixel set is fixed, so `total_weight` — the
normaliser of the weighted mean reduction — is a sum over a **fixed multiset**. Any variation in it
can only come from **summation order**.

`total_weight` is not computed in repository source. `src/training/losses.py:52-53` is a thin
wrapper:

```python
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index)
```

so the reduction — and the normaliser — belong to ATen, not to this repository. The probe
recovered it as `ce_sum / ce_mean`.

### 6.2 The PyTorch warning, verbatim — the primary-source evidence

From `e1_stdout.log:14`, **in the production seed-42 run**:

```text
/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:3053: UserWarning:
nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but you set
'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at
https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for
this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
  return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
```

The same warning appears twice in `grad_probe.log`, at `:8` (Probe A) and `:19` (Probe B), with
`deterministic=True` recorded at `:4`.

### 6.3 Verdict

**Determinism was requested in warn-only mode, exactly as ch3 §D specifies; a kernel with no
deterministic implementation was therefore permitted to execute rather than raising; and that
kernel is the cross-entropy forward reduction.**

Three candidate causes were on the table. The warning settles all three:

| Candidate | Status |
|---|---|
| (i) Expected per-batch variation | **Excluded.** Inputs and targets were held fixed; there is no batch-to-batch variation to observe. |
| (ii) CUDA kernel nondeterminism | **Confirmed, and named.** `nll_loss2d_forward_out_cuda_template`. |
| (iii) An unset `use_deterministic_algorithms` | **Definitively excluded.** The warning's own text quotes the flag as set. |

The framing matters and an earlier draft got it wrong. `torch.are_deterministic_algorithms_enabled()`
returns `True` under warn-only mode too; the distinguishing call is
`torch.is_deterministic_algorithms_warn_only_enabled()`. So the probe's gate confirmed *the flag
was on*, not *that determinism was enforced*. With the flag enabled, ops use deterministic
algorithms where available and raise `RuntimeError` where only nondeterministic ones exist;
`warn_only=True` downgrades that error to a warning and lets the nondeterministic kernel run. That
is what happened.

This is stronger evidence than the probe alone in three ways:

1. **PyTorch names the operation and states that no deterministic implementation exists.** That is
   a primary-source statement from torch 2.1.0 itself, not an inference from observed variation.
2. **It fired in the actual training run**, so the nondeterminism is attested in production, not
   extrapolated from a separate probe.
3. `nll_loss2d_forward` **is** the CE-loss forward reduction — exactly where `total_weight` is
   computed — which mechanistically explains Probe A, the arm that held logits fixed and still
   varied.

### 6.4 Localisation — from positive evidence only

**Probe A held `logits0` fixed and still varied. The loss forward is therefore sufficient to
explain the observed variation.**

Whether any upstream operation *also* contributes is **unresolved**, and it does not affect any
conclusion in this report. In particular: **the backbone forward is not exonerated.** No other
nondeterminism warning appeared in the logs, but the deduplication regime for these warnings is
unverified, and reading an unverified absence as evidence is a failure mode this repository has
now hit repeatedly — the ledger of instances is kept in B53. It was declined here. `[UNVERIFIED —
deliberately]`

### 6.5 Magnitude — the datatype noise floor

| Quantity | Observed values (of those printed) | Spread | ULP at that magnitude | ULPs | Relative |
|---|---|---|---|---|---|
| `total_weight` (implied) | 5177098.0, 5177098.5, 5177099.5 | 1.5 | 0.5 | **3** | **2.90e-7** |
| CE mean (Probe A) | 5.245862007141113, 5.24586296081543, 5.245863437652588 | 1.4305e-6 | 4.7684e-7 | **3** | 2.73e-7 |
| Total loss (Probe B) | 6.004361152648926, 6.004362106323242 | 9.5367e-7 | 4.7684e-7 | **2** | 1.59e-7 |

Every observed value lies **exactly on the float32 grid** at its magnitude — the `total_weight`
values are exact multiples of 0.5, the ULP there. The float64 reference computed by the probe was
`5177098.95762077`, whose correctly-rounded float32 is `5177099.0`; none of the three *printed*
passes equals it.

**Verdict on magnitude: this is the float32 noise floor, not a numerical defect.** Variation of
2–3 ULP over a fixed multiset is exactly what non-associative summation in a different order
produces. `[MEASURED]`

**The fourth `total_weight` value is DROPPED, not inferred.** The probe reported
`distinct over 10: mean=4 sum=4 total_weight=4` but printed only the first three passes — a
`rows[:3]` truncation in the probe's own reporting. A fourth distinct value therefore existed and
is unrecoverable, the pod being gone. **It is not guessed at here**, not even from the arithmetic
above. This is an **instrumentation defect in the probe, recorded as such**, recoverable at zero
marginal cost by printing all repeats when the probe is re-run during the teacher pod session.

Gradient hashes: Probe A produced 3 distinct gradient hashes over 10 passes (passes 1 vs 2
`DIFFER`); Probe B produced 3 distinct all-parameter gradient hashes over 10 (`VERDICT B: NOT
bitwise reproducible`). Probe B peak VRAM: 22.480 GiB of 44.43 GiB.

### 6.6 ch3 §D — all four measures are implemented; no deviation exists

ch3 §D's reproducibility row (tools-and-libraries table, **pp. 121–122**, the clause spanning the
page break) prescribes four measures. All four are implemented:

| ch3 §D measure | Implementation |
|---|---|
| `cudnn.deterministic = True` | `src/seeds.py:35` |
| `cudnn.benchmark = False` | `src/seeds.py:36` |
| `torch.use_deterministic_algorithms(True, warn_only=True)` | `src/seeds.py:37` |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8`, before CUDA init | `src/seeds.py:26` |

Bare grep output, for the record:

```text
$ grep -rn "use_deterministic_algorithms\|cudnn.deterministic\|cudnn.benchmark" src/ scripts/ configs/
src/seeds.py:35:    torch.backends.cudnn.deterministic = True
src/seeds.py:36:    torch.backends.cudnn.benchmark = False
src/seeds.py:37:    torch.use_deterministic_algorithms(True, warn_only=True)  # warn (not error) on non-deterministic ops

$ grep -rn "CUBLAS_WORKSPACE_CONFIG" src/ scripts/ configs/
src/seeds.py:21:    CUBLAS_WORKSPACE_CONFIG. Returns the seed used so callers can log it.
src/seeds.py:26:    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

The "before CUDA initialization" condition is satisfied by ordering: `set_seed(seed)` is the
**first statement** of `run()` at `train_e1.py:301`, ahead of `torch.device(...)` at `:302` and the
first device transfer at `:309` (`build_student(...).to(dev)`).

**An earlier draft recorded a `CUBLAS_WORKSPACE_CONFIG` deviation. That was wrong** — it came from
a grep whose pattern never contained the string — and it is retracted. There is no
implementation-vs-manuscript deviation in the determinism configuration.

**Seed-42 attestation is code-level only.** A grep of `e1_stdout.log` and every tarball member for
`cublas` returned **zero hits**; the matches were only on `deterministic` and `warn_only`. Nothing
in the run echoed the environment variable at runtime, so the claim rests on the source ordering
above, not on a runtime capture. The 4090 VRAM probe *did* echo
`CUBLAS_WORKSPACE_CONFIG=':4096:8'`, but that was different hardware on a different day and is
**not** presented as evidence for this run — it corroborates the mechanism only.
Queued, non-governed: have step 0 echo the effective determinism environment.

### 6.7 ch3 §D's caveat — partially accommodating, with a minimal extension proposed

ch3 §D, verbatim:

> "floating-point variation may remain across GPU classes and compiled CUDA kernels. Three-seed
> validation is planned for E1 and E3; E4 and E7 are recomputed per seed from their corresponding
> FP32 checkpoints, while E5 and E6 are fine-tuned per seed where compute permits. Primary
> inference uses one pre-registered seed, with mean ± SD reported across completed seeds."

The caveat covers variation **across GPU classes and compiled CUDA kernels**. The finding here is
**same-device, same-process, same-input repeat variation**. "Compiled CUDA kernels" arguably
reaches it, and a panel member could as reasonably read the sentence as addressing cross-hardware
variation only.

**Recorded as partially accommodating. Full coverage is not asserted.**

A minimal-change extension to the clause is queued as a governed manuscript edit (G6). **No
wording is drafted here**, deliberately — see the blocking dependency below. What the extension
must satisfy:

- **Scope** — same-device, same-process, run-to-run variation, stated as distinct from the
  existing caveat's cross-GPU-class framing rather than folded into it.
- **Cause** — a named training-path operation with no deterministic CUDA implementation in torch
  2.1.0, on the authority of PyTorch's own warning text rather than on observed variation.
- **Must NOT claim** — that evaluation is affected. That is **unresolved** (§6.8(b)), and an
  extension asserting it would over-reach in the opposite direction from the current clause.
- **Evidence it must cite** — `e1_stdout.log:14`; `grad_probe.log:8,19`; the 3 ULP / 2.9e-7
  relative / on-grid magnitude (§6.5); ch3 §D pp. 121–122 as the text being extended.

**Blocking dependency — do not draft this early.** The final wording depends on the
twice-evaluate-a-fixed-checkpoint attestation queued for the val rehearsal (§6.8(c), N12). Until
that attestation returns, it is not known whether the extension should say that reported metrics
*are* reproducible from the released checkpoint or must stay silent on evaluation. Wording written
before the result would have to be rewritten after it, and a draft sitting in the repository
invites exactly that. The specification above is the deliverable until N12 lands.

Note that ch3 §D already makes **no bitwise claim** about results, and its mean ± SD across
completed seeds is the reporting mechanism into which this variation is subsumed: 2–3 ULP is orders
of magnitude below any plausible three-seed standard deviation.

### 6.8 Scope of the limitation — pipeline-wide, training-only, with one attestation queued

**(a) PIPELINE-WIDE, NOT E1-SPECIFIC.** Every stage uses cross-entropy: the teacher fine-tune, E1,
E2, E3 (CE + KD terms), and E5/E6 (supervised-only CE + Dice). The same operation is in all of
them. **The limitation is written once, at pipeline scope, so it is not re-litigated per stage.**

**(b) TRAINING-PATH ONLY.** `nll_loss2d_forward` is not on the evaluation path — evaluation is
`argmax` plus confusion-matrix accumulation, with no loss computed (`validate()`,
`train_e1.py:271-284`). The **expectation** is therefore that reported metrics are bitwise
reproducible **from a fixed checkpoint**, and that only the *training trajectory* is
non-reproducible. **Framed as an expectation, not a fact**, pending (c).

**(c) ATTESTATION TASK — QUEUED, NOT DONE.** During the eval-harness rehearsal on val: evaluate
`e1_student_best_iter80000.pt` **twice** and assert bitwise-identical `all_class_miou`; also grep
the rehearsal stdout for any nondeterminism warning on the eval path. If identical, Chapter 4 can
say that **the checkpoint is not regenerable but every reported number is reproducible from the
released checkpoint** — which is the claim that actually protects the eight hypothesis tests.

**Consequence for this run, stated plainly:** `e1_student_best_iter80000.pt` is **not
regenerable**. Re-running the same command with the same seed on the same hardware will not
reproduce it bitwise. The released checkpoint is the artifact of record, which is why §8's
integrity chain matters.

---

## 7. Metric convention and eligibility

### 7.1 Eligibility is union-present, and `n_eligible_classes = 116` is correct

`reports/e1_training_loop_readiness.md` expects an absent-class skip via `present = row > 0`, and
`open_questions` NTC-8 carries absent-class eval support as OPEN (class 41 absent from test, class
68 absent from val). Against that, telemetry reports `n_eligible_classes = 116` with
`per_class_eligible` all-`True` at every validation. The tension is resolved by reading the code:

```python
def _cm_parts(cm):                      # src/eval/metrics.py:91
    tp = torch.diag(cm).float()          # :94  intersection
    gt = cm.sum(1).float()               # :95  GT support
    pr = cm.sum(0).float()               # :96  prediction support
    return tp, gt, pr, gt + pr - tp      # :97  ... and the union

def miou_from_confusion(cm, class_indices=None) -> float:   # :116
    tp, gt, pr, un = _cm_parts(cm)                           # :128
    eligible = _restrict(un > 0, class_indices)              # :129
    return _macro(tp / un.clamp_min(1e-9), eligible)         # :130
```

Eligibility is **union-present** (`UN_c = GT_c + PR_c − TP_c > 0`), not GT-present. The docstring
at `:119-121` says so explicitly — "a prediction-only class contributes IoU 0 and is counted" — and
`:125` notes that per-image metrics keep the GT-present rule via `_miou_from_cm`. So a class with
zero GT can still be eligible if it was ever predicted. **`n_eligible_classes = 116` is the correct
output of the implemented rule, not a failure to skip.** `[MEASURED]`

**Indexing correction to the audits:** `reports/trainval_mask_value_audit.md:48,59` names the
val-absent class "68" and the test-absent class "41" as **COCO `category_id`s**. Mask/classifier
index = `category_id + 1`, so they are **index 69** (val) and **index 42** (test). The audits'
wording is category_id-space; anything reasoning about eligibility must be index-space.

**NTC-8's val half is resolved:** val has exactly one zero-GT class, index 69, and it is counted
anyway because it was predicted somewhere. Its test half stands, and the same code path handles
index 42 on test, so it must be settled **before** the single test campaign, not during it.

**Quantified:** one wrongly-included zero-support class costs `0.36314 / 115 = 0.3158 pp` on the
headline. Excluding index 69 would give `0.36314 × 116 / 115 = 0.36630`. **That recomputation is an
internal note only and is not a reported number.**

**Three classes report exactly 0.0 IoU at all five of the last validations: indices 32, 69, 95.**
Index 54 is 0.0 at four of five but 0.00084632 at 64k, proving it has GT support; index 17 is
near-zero and fluctuating, so it has support too.

- **Index 69** — zero val GT by the audit, `pr > 0` by the eligibility rule. Scoped precisely:
  this establishes that **class 69 was predicted somewhere across the 846 val images at each of the
  five validations** — not consistently, and not in any identified image.
- **Indices 32 and 95** — the mechanism is **not determinable from the saved artifacts**. Both
  `pr = 0` (never predicted) and `pr > 0` with zero overlap yield exactly 0.0, and telemetry stores
  per-class IoU but not per-class `gt`/`pr`. "Never predicted across 846 images at five
  validations" would be a class-collapse observation worth recording; whether that is what
  happened is **unresolved**. `[UNVERIFIED]`
  Recorded as an **instrumentation limit in the `rows[:3]` family** — the same pattern as §6.5's
  dropped fourth value: the measurement was taken but not saved. Recoverable at zero marginal cost
  by **saving the confusion matrix** during the eval-harness rehearsal.

### 7.2 The convention matches the benchmark's — the offset claim is retracted

An earlier draft recorded a "systematic ~0.32 pp offset vs published numbers" arising from a
union-present-vs-GT-present convention mismatch. **That claim was backwards and is retracted.**

`docs/EVALUATION_CONTRACT.md` §2.2 quotes MMSeg verbatim:
`area_union = area_pred_label + area_label - area_intersect`, reduced with `np.nanmean`. §2.3
records that `nan_to_num` is **not** configured — "this is the decisive finding". MMSeg's
`IoUMetric` therefore excludes a class only when its **union** is zero: **MMSeg is union-present
too.**

Stated as **positive justification, not merely as a retraction:** union-present eligibility is
precisely what makes this pipeline's all-class mIoU **commensurable with the benchmark's**, and
GT-present would have *introduced* a divergence that does not currently exist. That is the
affirmative reason `EVALUATION_CONTRACT` §(f)'s "match PlantSeg benchmark reporting" is satisfied.
The retracted offset is the lesser half of the finding.

The numbers nevertheless remain **non-comparable** to published results for the reasons already in
contract §(a), and **no comparison to published tables appears in this report**.

### 7.3 The zero-GT penalty is symmetric — explicitly NOT a second downward pressure

Val has exactly one zero-GT class (index 69). Test has exactly one (index 42). Same count,
comparable magnitude.

**The penalty is therefore a wash between the splits.** It is pre-registered as what it is: a fixed
≈0.32 pp deduction present on **both** splits and on **every stage**, cancelling out of all eight
paired comparisons. It is **not** a second reason to expect test below validation, and must not be
stacked alongside §5.3's selection bias. §5.3 remains the only pre-registered reason.

---

## 8. Artifacts and integrity

Everything below was transferred off the pod with `runpodctl` before termination and now lives at
`~/plantseg_runs/e1_seed42` on the operator's laptop, with an off-machine copy. The pod is gone;
these files are the only copies.

### 8.1 Five loose files — SHA256 captured 2026-09-12 on the laptop, post-transfer

| sha256 | file | bytes | attests |
|---|---|---|---|
| `8b4e61e8bbd8df0cce3e97d4995e166856aa99135aaf4a6fec44a3e69c2a0d2c` | `best.json` | 119 | post-run |
| `09285efbef31f0ec6cd99cfaa7185c3c97fbe7ba88454d7d113565b8ff19a0c8` | `e1_seed42_logs.tar.gz` | — | post-run |
| `fa183481140147102342606237ff3d702b7d971abb2ded71e7336d977439044b` | `e1_stdout.log` | 124,276 | whole run |
| `cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03` | `e1_student_best_iter80000.pt` | 23,734,564 | post-run |
| `3ca9fd7998e7734efa414615e906236f04babc890861e852bade066bb97ff19d` | `e1_telemetry.jsonl` | 20,341,737 | whole run |

### 8.2 Five tarball members, hashed individually

A tarball hash proves transfer integrity but is useless for citing pre-flight or step-0 evidence
later, so the members are hashed too.

| sha256 | member | bytes | mtime | attests |
|---|---|---|---|---|
| `ec8f27edc6112996c134abb2d21acee1b2bd14713be0ee2e48d7b67f713116c1` | `preflight.log` | 11,638 | 2026-09-11 20:39 | **pre-run** |
| `5d2b497537f7a1cddbc498b459129b2fccb7734ab6f0f3a67f47c78d4a299a11` | `step0.log` | 1,566 | 2026-09-11 20:01 | **pre-run** |
| `c4a0214a0ba6030a47dadd6b58fbfa50194bbc94fb4773442a5de33c5bf3eb1b` | `grad_probe.log` | 2,500 | 2026-09-11 20:46 | **pre-run** |
| `5ac1a941e52877cf9be2f87742b9800927960ef0186eba25ff2dfc5134a702fa` | `imagenet_capture.json` | 547 | 2026-09-12 21:09 | **post-run** |
| `cc05aa5603643a11f0cae47de0348dab41409548a2bf72ed8b8dcbd29e3a2fab` | `check_hash.txt` | 146 | 2026-09-12 21:09 | **post-run** |

The pre-run group attests the state the run *started from*: the environment gate, the 5/5 pre-flight
with seed-sequence `MATCH`, and the determinism probe. The post-run group was captured after the
run finished and attests only what was on disk at that time. The ImageNet capture is nonetheless
valid provenance for the initialisation, because it hashes the cached weight file the run itself
downloaded and torchvision's own URL-embedded hash prefix (`5c1a4163`) independently corroborates it.

One caveat on the ImageNet capture, stated rather than glossed: the values quoted in §2 come from
the capture **printed to the terminal** at `2026-09-11T13:00:56Z` (its own `captured_utc` field,
ten minutes after launch). The archived `imagenet_capture.json` is a **re-run** of the same block
with `> /workspace/imagenet_capture.json` at tarball-build time, so its `captured_utc` will read
2026-09-12 and **its content has not been read back**. The hashed file is the same cached weight
file in both cases, so the hash is expected to be identical — but "expected", not verified.
`[MEASURED for the 09-11 capture; UNVERIFIED for the archived file's content]`

### 8.3 `check_hash.txt` does NOT contain a checkpoint hash

Its 146 bytes are two lines of torchvision source — `inspect.getsource` of
`WeightsEnum.get_state_dict`:

```python
    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)
```

It was captured to document how torchvision resolves weight URLs, alongside `imagenet_capture.json`.
No reconciliation against a checkpoint hash is possible, and none is attempted.

**Recorded on its own terms: no pod-side SHA256 of `e1_student_best_iter80000.pt` was ever
captured.** The laptop hash `cf0879f7…6a03` therefore **stands alone**, with no pod-side
counterpart to verify the transfer against. The ImageNet *input* weights received provenance
capture; the trained *output* checkpoint did not. `[MEASURED]`

Mitigating, not equivalent: the checkpoint loads cleanly and its internal `iter` and
`best_val_miou_all_class` agree bitwise with `best.json` and with the telemetry maximum (§1). That
makes silent corruption in transfer implausible, but it is not a hash.

**Queued, non-governed: hash checkpoints pod-side before download**, for the teacher run and every
subsequent seed.

### 8.4 Never captured

**`e1_pip_freeze.txt` and `e1_nvidia_smi.txt` do not exist** — in neither the five loose files nor
the tarball. They were never captured, and the pod is gone, so they cannot be recovered.

Consequences, and why this is survivable:

- **The version authority is the lock files** — `requirements.lock` and `requirements-e1.txt`, as
  baked into image `sha256:b80b645d…866aaf` under `--require-hashes`. An as-installed freeze would
  have been a convenience, not the authority.
- **The hardware authority is `step0.log`** (§3), which carries GPU model, driver, compute
  capability, memory, CPU and RAM.

Queued, non-governed: add both captures to step 0 so the next run has them.

---

## 9. Deviations and open items surfaced by this run

### 9.1 Albumentations — a factual deviation, pending a governed decision

`docs/IMPLEMENTATION_CONTRACT.md` B2 names **Albumentations** as the augmentation library, tagged
`[ch3 §D]`, and ch3 names it. `requirements-e1.txt` states that E1 imports neither OpenCV nor
Albumentations and uses NumPy/PIL transforms only.
`reports/claude_web_alignment_handoff.md` flagged this gap and deferred the decision.

**Recorded here as the factual finding only: E1 seed 42 trained without Albumentations, deviating
from the library named in ch3 §D.** Every augmentation *semantic* in the B2 table is implemented as
specified; only the named library differs.

This is **not** a pending version string — it is a governed decision (G5), and it is **not taken in
this report**. It needs the governed-path workflow (`docs/IMPLEMENTATION_CONTRACT.md` and a ch3 edit
with a Ctrl+F table). The facts a decision-maker needs, with no conclusion attached:

- E1 seed 42 was trained with the hand-written NumPy/PIL transform stack at commit `f77d05d7`.
- Any change to the augmentation implementation makes subsequent stages **non-comparable** to E1
  seed 42 unless E1 is re-run.
- An E1 re-run costs **~23.0 h of A40 time per seed** (§4.1).

Note on wording, because an earlier draft of this report got it wrong: a change to the augmentation
stack would **not** *invalidate* the completed run. E1 seed 42 remains a valid record of what it
was — trained under the transform stack at `f77d05d7`, fully provenanced. The exposure is
**cross-stage comparability**, not validity.

Also worth stating plainly: the teacher / E2–E3 manifest **does not exist yet**, so albumentations
there is not `NEED_TO_CONFIRM` in the "we don't know" sense — it is simply **not yet specified**.

### 9.2 statsmodels — resolved

`statsmodels 0.14.6`, from `requirements.lock`. `open_questions` #5's own stated resolution names
the lock / Dockerfile as the authority, and no pip freeze is needed to settle it. The
albumentations half of #5 is **not** closed by this — it is escalated to §9.1.

### 9.3 Instrumentation defects (measurement taken, not saved)

| Defect | Effect | Recovery |
|---|---|---|
| Probe printed `rows[:3]` of 10 | 4th distinct `total_weight` unrecoverable (§6.5) | print all repeats; re-run during the teacher pod session |
| Telemetry stores per-class IoU, not per-class `gt`/`pr` | indices 32/95 mechanism undeterminable (§7.1) | save the confusion matrix during the eval-harness rehearsal |
| No pod-side checkpoint hash | laptop hash stands alone (§8.3) | hash pod-side before download |
| No `pip freeze`, no `nvidia-smi` dump | version/hardware records rest on lock + step0.log (§8.4) | add both to step 0 |
| Step 0 does not echo the determinism environment | `CUBLAS_WORKSPACE_CONFIG` attestation is code-level only (§6.6) | echo it in step 0 |
| Unguarded `git clone`; `tee -a` to `step0.log` not `setup.log` | duplicated block, spurious `fatal:` line (§2.2) | guard the clone; align the log name |
| `os.cpu_count()` ignores the cgroup quota | 48 threads on 9 vCPUs (§3.3) | size threads from `RUNPOD_CPU_COUNT` |

None of these affects the validity of the run or of any number in §1. All are cheap to close before
the teacher run.

---

## 10. What this run establishes, and what it does not

**Establishes:**

- A complete, un-resumed, seed-42 E1 baseline at 80,000 iterations, with **0.36314 all-class
  validation mIoU**, selected on the contract-mandated criterion.
- Full provenance from commit through image, dataset md5, ImageNet init hash, pre-flight `GO`, and
  seed-sequence `MATCH`, with the *stronger* `git_checkout` attestation and a verified-clean tree.
- A measured throughput of **1.036 s/iter → 23.02 h** on an A40, the basis for every future
  schedule estimate.
- That the CE forward reduction has no deterministic CUDA implementation in torch 2.1.0, attested
  in production by PyTorch itself, at a magnitude of 2–3 ULP — the float32 noise floor.
- That the eligibility rule is union-present and **matches** the benchmark's convention.

**Does not establish:**

- Anything about the **test split**, which has not been evaluated.
- Whether 80,000 iterations was **sufficient** — §5.2.
- Whether the backbone forward contributes to the nondeterminism — §6.4.
- Whether reported metrics are bitwise reproducible from a fixed checkpoint — **expected**, queued
  for attestation at §6.8(c).
- Whether indices 32 and 95 were ever predicted — §7.1.

**Next, in order:** B53 (the UVM host fault across three Community pods); the governed-path queue;
the eval-harness rehearsal on val carrying its two queued attestations; then E1 seeds 2 and 3. The
full queue is §11 — nothing in it is decided here.

---

## 11. Queue register

Every item this session queued, in one place. **Nothing below is decided in this report.** Governed
paths per [AGENTS.md](../AGENTS.md) rule 8 — `src/**` · `configs/**` · `scripts/**` ·
`requirements*` · `docs/EVALUATION_CONTRACT.md` · `docs/IMPLEMENTATION_CONTRACT.md` — plus
manuscript edits, which are the operator's half and plan-gated the same way.

### 11.1 Governed — needs a plan and an explicit "go" before any edit

| # | Item | Where |
|---|---|---|
| G1 | D29 — the no-resume rule's contract-level home is not settled | `docs/open_questions.md:731` |
| G2 | D30 — the INFERRED 11–14 GB VRAM band is falsified for every student stage under determinism | `docs/open_questions.md:747` |
| G3 | D31 — `write_report`'s generation date is hardcoded | `docs/open_questions.md:776`; `scripts/smoke_loss.py:36` |
| G4 | D32 — `IMPLEMENTATION_CONTRACT.md:327-330` asserts more than its evidence supports | `docs/open_questions.md:789` |
| G5 | Albumentations — B2 names the library, E1 ran NumPy/PIL; decide library-name edit vs code retrofit | §9.1; `docs/IMPLEMENTATION_CONTRACT.md` B2 |
| G6 | ch3 §D pp. 121–122 — extend the floating-point caveat to same-device run-to-run variation | §6.7 (manuscript) |
| G7 | NTC-8 test half — index 42 has zero test GT under union-present eligibility; settle **before** the single test campaign, not during it | §7.1; `src/eval/metrics.py`; `docs/EVALUATION_CONTRACT.md` |
| G8 *(raised as N18)* | Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` as a Dockerfile `ENV` so the value is attested by the image, not only by source ordering — **default is DON'T**, see §11.3 | `Dockerfile` (no `ENTRYPOINT`; `ENV` at `:79`/`:97`) |
| G9 *(raised as N19)* | Assert the ordering in `set_seed` instead of depending on it, so a caller that touches CUDA before seeding fails loudly — see §11.3 | `src/seeds.py` |

**How G8 and G9 are classified — the path list is a floor, not a ceiling.**

> Anything on rule 8's path list is governed **regardless of consequence**. Consequence can only
> **ADD** items to the governed set; it can never remove one. Applying consequence-classification
> to something already on the list would invert the rule into an escape hatch.

Applied here:

- **G9 is governed by the path list alone.** `src/seeds.py` is inside `src/**`, which rule 8 names
  directly. That settles it, and no consequence argument is needed or admissible. G9 was raised as
  non-governed and is reclassified on that basis.
- **G8 is governed by consequence, because the path list does not reach it.** `Dockerfile` is
  **not** on rule 8's list — and that read must not be inverted later into "the Dockerfile is a
  governed path." G8 clears the bar on effect: the image digest `sha256:b80b645d…866aaf` is
  attested in ways-of-working, in `step0.log`, and in ch3's reproducibility record, so rebuilding
  it mid-campaign means E1 seed 42 and every subsequent stage ran under **different attested
  environments** — the same comparability hazard as G5.

### 11.2 Non-governed — `reports/`, runbooks, `docs/open_questions.md`, `docs/runpod_environment.md`

| # | Item | Where |
|---|---|---|
| N1 | **B53** — the UVM host-fault record across three Community pods, carrying the methodological pattern ledger | new `reports/` entry |
| N2 | `open_questions` #5 — record statsmodels resolved from the lock; escalate the albumentations half to G5 | `docs/open_questions.md:855-859` |
| N3 | Step 0 — guard `git clone` with the same `.git` test that guards the `mv`; align the `tee -a` target with the log name the runbook uses | §2.2 |
| N4 | Step 0 — echo the effective determinism environment (`CUBLAS_WORKSPACE_CONFIG`, warn-only state) so the attestation is runtime, not code-level | §6.6 |
| N5 | Step 0 — capture `pip freeze` and `nvidia-smi` | §8.4 |
| N6 | Step 0 — read the CPU allocation from `RUNPOD_CPU_COUNT`, size torch threads from it, and record allocated-vs-seen | §3.3 |
| N7 | Step 0 — make the CUDA + UVM device-**open** gate permanent in the runbook (existence ≠ usability) | runbook |
| N8 | Step 0 — capture RunPod env by explicit variable name, never `env \| grep` | runbook |
| N9 | Hash checkpoints **pod-side** before download, for the teacher run and every subsequent seed | §8.3 |
| N10 | Gradient probe — print all repeats, not `rows[:3]`; re-run during the teacher pod session | §6.5 |
| N11 | Eval rehearsal — save the confusion matrix so per-class `gt`/`pr` survive | §7.1 |
| N12 | Eval rehearsal — evaluate `e1_student_best_iter80000.pt` **twice**, assert bitwise-identical `all_class_miou`, and grep stdout for any eval-path nondeterminism warning | §6.8(c) |
| N13 | Settle whether torchvision 0.16.0 honours `check_hash=True`; currently `[UNVERIFIED]` | `docs/runpod_environment.md:217-220`; B46 §2 |
| N14 | Replace "Download provenance is manual (no URL/SHA/date captured)" with the Zenodo URL and md5 `9358a66d…`, captured 2026-09-11 | `reports/dataset_download_log.md:78` |
| N15 | §3.1's table lists **4** pre-flight stages; `preflight_e1.py` at `f77d05d7` runs **5** (`class_weights` is missing) | `reports/e1_launch_runbook_v2.md:88-97` |
| N16 | Rebuild the campaign cost estimate from the measured per-run cost, then decide Secure vs Community for the remaining stages | §3.2 |
| N17 | **Verification discipline** — never adjudicate an absence with an instrument whose coverage of the adjudicated string was not established; rule text and the three instances at §11.4 | methodology; ledger in B53 |
| N20 | Recover pattern-ledger instances 1, 2 and 4 from repo history, at the commits where those corrections landed — B53 §5 carries them as gaps rather than reconstructions | `git log`; B53 §5 |

### 11.3 The cuBLAS ordering dependency — G8 and G9 are two answers to one problem

`CUBLAS_WORKSPACE_CONFIG` is set in-process at `src/seeds.py:26`, and `set_seed(seed)` is the first
statement of `run()` at `train_e1.py:301`, ahead of `torch.device(...)` at `:302` and the first
device transfer at `:309`. The configuration is therefore correct **by ordering**, and nothing in
the current training path violates it. The exposure is that the ordering is a convention, not an
enforced invariant: a future caller that touches CUDA before seeding would silently lose cuBLAS
determinism, with no error and no log line.

**G8 — bake it into the image. Default is DON'T.** An `ENV` line removes the ordering dependency
outright, but buys only that, and costs an image rebuild and a new digest mid-campaign (see the
classification note in §11.1). **Recorded default: no change.** Revisit only if the campaign needs
a rebuild for some other reason, in which case the `ENV` line rides along at zero marginal cost.

**G9 — assert the ordering instead. The cheaper answer, and the one to prefer.** In `set_seed`:
`assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"` and
`assert torch.cuda.is_initialized() is False`. No rebuild, no digest change, and no behavioural
change on the current correct path — only a loud failure on an incorrect one. Implementation note:
the CUDA-initialised assertion must sit **before** `:32-33`'s `torch.cuda.manual_seed*` calls
rather than after.

**G5, G8 and G9 are one family: mid-campaign attestation changes.** Each leaves the *numbers*
alone and moves what the run is *attested against* — G5 the augmentation stack, G8 the image
digest, G9 the provenance commit. G9 belongs here despite being assertion-only: editing
`src/seeds.py` changes the commit recorded in `run_meta`, so seeds 2 and 3 would run at a different
`git_head` from seed 42's `f77d05d7` even though numerical behaviour is identical.

**Shared default for all three: do not land mid-campaign.** Sequence them together, and take any
of them only when another change has already forced a commit, a rebuild, or a re-run — at which
point it rides along at zero marginal cost.

### 11.4 N17 — the rule, and the three instances that produced it

**Rule.** Before recording an absence, state which instrument was used and why its coverage
includes the adjudicated string. **If coverage cannot be established, record "not determinable
from X" rather than "absent."**

The failure class is broader than grep: *adjudicating an absence using an instrument whose coverage
of the adjudicated string was never established.* Three shapes, all seen this session:

| Shape | Instance |
|---|---|
| A pattern that never contained the adjudicated string | the `CUBLAS_WORKSPACE_CONFIG` grep — it reported "not present in `src/`" while the value sat at `src/seeds.py:26` (§6.6) |
| An instrument that truncated its own output | the gradient probe's `rows[:3]` — a fourth distinct `total_weight` existed and was not printed (§6.5) |
| A field read from one artifact when a second artifact held it | `44.43 GiB` — absent from `step0.log`, present in `grad_probe.log`'s P0 header (§3.1) |

A fourth shape was **avoided** rather than committed: reading the absence of any further
nondeterminism warning as exoneration of the backbone forward, where the warning deduplication
regime is unverified (§6.4). That one was caught before publication, which is the difference worth
recording.
