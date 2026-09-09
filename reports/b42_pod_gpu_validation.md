# B42 — first GPU validation of the official image on a RunPod pod

**Type:** operational validation + doc corrections. **Status:** COMPLETE — `VERDICT: GO`.
**Date:** 2026-09-09. **Closes:** the capability-gate and arch-list halves of B30's pre-E1 audit
finding (see [pre_e1_launch_audit.md](pre_e1_launch_audit.md) addendum).

**Image commit:** `f77d05d7b35187bf0da7e7b94a629549fe2e1c05`
**Image digest:** `sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf`
**Pod:** RunPod **Community Cloud**, 1× RTX 4090, id `ept6347ehe96ew` — terminated after the verdict.
**Real E1 training was NOT launched.**

---

## 1. What this establishes

`scripts/preflight_environment.py --mode image` has passed since B38, but on a **CPU builder**. The
GPU path had never run anywhere. This is the first `VERDICT: GO` from `scripts/preflight_e1.py` on
real GPU hardware, against the authoritative image **provisioned by digest**.

The load-bearing result is the R5 seed re-measurement in §4: the pinned stack draws the
development-stack sequence, so every seed-dependent artifact under `reports/` remains valid and
needs no regeneration on the pod.

---

## 2. Environment as measured

| Field | Value |
|---|---|
| `PLANTSEG_GIT_COMMIT` | `f77d05d7b35187bf0da7e7b94a629549fe2e1c05` |
| `PLANTSEG_IMAGE_DIGEST` | `sha256:b80b645d…866aaf` (supplied via template env var) |
| `PLANTSEG_DATA_ROOT` | `/workspace/plantseg_data/plantseg` |
| `PYTHONPATH` | `/workspace/plantseg-thesis` |
| Platform | `Linux-6.8.0-58-generic-x86_64-with-glibc2.36` |
| Python | 3.11.16 |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, 1 device |
| Compute capability | `(8, 9)` — sm_89 |
| Driver / host CUDA | 570.133.20 / 12.8 |
| torch | 2.1.0+cu121, `cuda_build` 12.1, cuDNN 8902 |
| Quant backends | `['qnnpack', 'none', 'onednn', 'x86', 'fbgemm']` |
| vCPU allocated / seen | 32 / **256** |
| RAM allocated / seen | 50 GB / **1056595676 kB ≈ 1008 GiB** |
| `torch_num_threads` | 128 |
| Volume | 20 GB at `/workspace`, 2.2 GB used after dataset staging |
| Dataset | 5367 / 846 / 1561 = 7774 pairs, images **and** annotations both matching |

**The container reads host CPU and memory, not RunPod's allocation.** Both figures are recorded
because §3 of the run-provenance requirement wants the hardware facts, and because the
`num_workers` default derives from the container's view (§5.4).

---

## 3. Three RunPod platform behaviours that cost pod time

None is an image defect — `docker run -it` yields a shell directly — but each was met for the first
time on a provisioned, paid pod. Now documented at
[docs/runpod_environment.md](../docs/runpod_environment.md) §4c.

### 3.1 The container restart-loops without an explicit Start command

RunPod starts the container non-interactively. The image ends in `CMD ["/bin/bash"]`, so PID 1 reads
EOF on stdin and exits 0 at once; RunPod's supervisor restarts it. The pod log showed:

```
15:10:06  start container … : begin
15:10:24  start container … : begin      ← +18s
15:10:40  start container … : begin      ← +16s
…twelve times, no container stdout
```

**Fix:** set the pod template's **Start command** to `sleep infinity`. A healthy pod then logs that
line **once** and goes silent — silence is the pass signal, because `sleep` emits nothing and there
are no further lifecycle events. Verified: the corrected pod logged `start container … : begin` at
`15:32:56` and nothing after.

This touches no package, no config value, and nothing the run records: `run_meta` reads
`PLANTSEG_GIT_COMMIT` (baked) and `PLANTSEG_IMAGE_DIGEST` (supplied), neither of which depends on
`CMD`.

### 3.2 The web terminal does not attach to this image

Enabling RunPod's web terminal silently reverted to off, both before and after 3.1 was fixed. The
`python:3.11-slim-bookworm` base installs only `libgl1`, `libglib2.0-0`, `libgomp1` and `git` — no
`openssh-server`, none of the startup tooling RunPod's own templates carry.

**Fix:** RunPod's **proxied SSH**, which needs neither a public IP nor an exposed TCP port:

```bash
ssh-keygen -t ed25519            # on the workstation; register the .pub on the pod
ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

That proxy carries **no SCP or SFTP**. Retrieve artifacts with `runpodctl send`, or print small text
files and copy them out. `[INFERRED]` — the missing-`sshd` explanation fits the evidence but was not
isolated directly; the workaround is `[MEASURED]`.

### 3.3 `PLANTSEG_IMAGE_DIGEST` must be a template environment variable

There is no `docker run -e` on RunPod. Supplied through the template's **Environment variables**
field (value including the `sha256:` prefix), the digest resolved correctly. Omitted, it is recorded
absent on a pod that otherwise looks healthy — exactly the B40 #19 failure mode B41 fixed.

### 3.4 Two Community Cloud facts, learned the expensive way

- **Stopping a pod releases the GPU.** A stop taken in order to apply 3.1 lost the 4090 to another
  tenant; `Automatically migrate your Pod data` then failed repeatedly with *"no instances currently
  available"*, because migration is constrained to identical GPUs. Recovery meant terminating and
  redeploying. **Do not stop Community pods — leave them running or terminate them.**
- **Community 4090 capacity churns in minutes.** Two deploys were lost to
  *"no longer any instances available with the requested specifications"* while the panel was being
  reviewed. Have the template fully correct beforehand so deploy is one click.

Switching cloud type in the GPU filter **silently clears the template's advanced compatibility
filters** (min vRAM, min RAM, CUDA versions all reset). Re-verify specs by eye on every deploy.

---

## 4. Preflight results

`python -B scripts/preflight_environment.py --mode gpu` → **`RESULT: PASS (37/37)`**, exit 0,
`"verdict": "pass"` written to `/workspace/preflight_gpu.json`. Every version pin exact; `mmcv 2.1.0`
compiled ops present; registry agrees with the hashed lock across 78 distributions.

`python -B scripts/preflight_e1.py` → **`VERDICT: GO`**, exit 0:

| Stage | Result | Secs | Detail |
|---|---|---|---|
| 1/5 CE class-weight artifact | PASS | 4.0 | 116 weights, ratio 259.551929, max deviation 0.000e+00 |
| 2/5 `verify_env.py` | PASS | 231.1 | verdict=PASS, `can_run_real_E1: True` |
| 3/5 `smoke_dataloader.py` | PASS | 8.0 | exit=0 |
| 4/5 `smoke_aug_stochasticity.py` | PASS | 17.0 | **seed sequence MATCH** |
| 5/5 `train_e1.py --dry-run` | PASS | 460.6 | 6/6 checks exercised, 0 skipped |

### R5 — pinned-stack seed comparison

```
B30 §9 baseline : [[1608637542, 1273642419], [1935803228, 787846414]]
observed here   : [[1608637542, 1273642419], [1935803228, 787846414]]
VERDICT         : MATCH — the pinned stack draws the development-stack sequence.
```

Processes A and B produced identical sequences and all eight sub-checks passed: train varies across
passes at both `workers=0` and `workers=2 persistent`, val is bit-identical as the control, and
seeds and digests reproduce across processes. **Seed-dependent artifacts under `reports/` remain
valid.**

---

## 5. Measured facts worth citing

### 5.1 `torch.cuda.get_arch_list()` — the pinned wheel's architecture list

```
['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

Exactly the set the environment docs claim, now **MEASURED** rather than asserted. Two consequences:

- **No `sm_89`.** The RTX 4090 runs on the `sm_86` cubin via CUDA minor-version binary
  compatibility, as predicted. Ada cards work.
- **No `compute_*` entry**, so the build embeds **no PTX** and has no JIT fallback. A Blackwell-class
  device (sm_100/sm_120) has neither a matching cubin nor PTX to compile from. The DENY list is
  evidence-based, not inferred.

### 5.2 The capability gate exists and fires

`scripts/verify_env.py:32` defines `MAX_SM = (9, 0)`; `:133-146` hard-**FAIL**s above it and feeds
the `[18] VERDICT` block. Observed: `gpu_capabilities : ['sm_89'] (max supported by the pinned
stack: sm_90)` / `capability_gate : PASS`. B30's finding that capability was ungated was accurate at
`148af2c` and was closed by B31-3/4 (`bc5f644`).

### 5.3 B41's runtime provenance, confirmed on a pod

```
[provenance] git_head=f77d05d7b35187bf0da7e7b94a629549fe2e1c05 (source=image_env)
             image_digest=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
```

`git_head_source=image_env` as designed, digest from the template env var. Checkout clean
(`dirty=False`, 0 modified, 0 untracked). `repo_.pt_files: []` and `no_repo_checkpoint: True` — the
out-of-repo checkpoint guard holds; the dry-run wrote to `/tmp/e1_dryrun_ckpt_*`.

### 5.4 Resolved `num_workers` = 12

`src/training/train_e1.py:636` — `min(max((os.cpu_count() or 4) - 2, 1), 12)` with
`os.cpu_count() = 256` (host cores, not the 32 vCPU allocated) → clamped to the **cap of 12**. Not
overridden. The dry-run logged `num_workers=0`, which is the dry branch at `:616`; **12** is the
real-run value.

The cap is what keeps this sane and stable: any host with ≥14 cores yields 12, so the recorded value
does not drift across pods even though the vCPU allocation does.

### 5.5 Registry pull facts

Package is **public** on GHCR (anonymous pull token issued), so no registry authentication is
needed. The image is a single `linux/amd64` manifest plus a build attestation, **2.85 GB
compressed**, pulling in ~85 s and expanding to the ~8–12 GB that makes the 25 GB container disk
correct. The 4090 reports **24564 MiB**, which is why a minimum-vRAM filter of 20 GB rather than 24
avoids a boundary-rounding exclusion.

---

## 6. Residuals — neither blocks E1

1. **ImageNet backbone not cached** (`imagenet_cached: False`;
   `/root/.cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth` absent). The real run
   defaults to `--init imagenet` and will fetch it on launch. The cache dies with the pod, so this
   recurs on every fresh pod. Not code-gated; the operator's responsibility, as the GO note says.
2. **SegNeXt teacher checkpoint absent** — blocks E2/E3, not E1.
3. **`albumentations` not installed** — teacher/E2-E3 stack, not an E1 dependency.

---

## 7. Community Cloud is not suitable for the real E1 runs

Resume **is** supported (`--resume`, with `last.pt` written every `--ckpt-interval 2000`), and it
restores model, optimizer, scheduler and RNG state. But `src/training/train_e1.py:395` prints its own
warning:

> data-order continuity is NOT restored. The training loop consumes an infinite
> `cycle(train_loader)`; the position within the current epoch is not recoverable, so the post-resume
> sample order differs from an uninterrupted run. RNG streams ARE restored, so augmentation remains
> reproducible from this point onward. **A resumed run is NOT bitwise-identical to an uninterrupted
> one.**

The lost compute is trivial — an interruption costs ≤2000 of 80,000 iterations, 2.5%. The cost is
that the resumed run carries a permanent asterisk, in a programme whose pre-flight **hard-gates** on
seed-sequence identity. E1 is the baseline every later stage is measured against.

Against that, §3.4 records Community taking the GPU during a routine stop on day one. E1 ×3 is
roughly 30 GPU-hours of exposure. At ~$0.34/hr Community versus ~$0.74/hr Secure the difference is
about **$12 for three uninterrupted runs** — cheap relative to an interrupted baseline.

**Recommendation:** run E1 ×3 (and E3 ×3) on **Secure Cloud**. Community remains right for
preflights and probes, where an interruption costs minutes. `[operational judgement — not a ch3
requirement]`

---

## 8. Doc corrections in this commit

| File | Change |
|---|---|
| `docs/runpod_environment.md` §3 | `docker run` invocation was missing `-it`; without a TTY the `CMD ["/bin/bash"]` exits immediately instead of giving a shell. Step 4's example always had it — the two now agree. |
| `docs/runpod_environment.md` §4c | New: the three RunPod platform behaviours of §3 above. |
| `reports/e1_runpod_launch_runbook.md` §3 | Pointer to §4c before the checklist, plus a note that 3.2–3.3 (conda/venv install) are superseded when provisioning from the official image. |
| `reports/pre_e1_launch_audit.md` | Dated addendum closing both halves of the capability finding. Original text left as written. |

---

## Provenance / guardrails honored

- No training launched; no package installed, upgraded or downgraded; no config value, batch size or
  seed altered; no test loosened.
- `docs/reference/reference.pdf` never opened, read, hashed, copied, staged or modified.
- Run outputs written to `/workspace` only (`preflight_gpu.json`, `preflight_e1_output.txt`); the
  dry-run's checkpoints went to `/tmp`. Nothing written into the repo tree.
- `--num-workers` not overridden; the B31-5 default was allowed to resolve and is recorded in §5.4.
- Pod terminated after the verdict.
