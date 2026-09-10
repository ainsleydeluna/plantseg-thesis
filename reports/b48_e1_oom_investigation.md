# B48 — E1 launch attempt 1: why it OOM'd, and what F7 actually predicted

**Type:** incident record + root-cause investigation. **Status:** root cause **CONFIRMED** by
two-arm on-pod measurement; relaunch **decided — Option A** (§7). **Date:** 2026-09-10.
**Corrects:** [pre_e1_launch_audit.md](pre_e1_launch_audit.md) §7.1's consequence prediction for
F7, and resolves that finding's `CANNOT-VERIFY-LOCALLY` marker.
**Source:** RTX 4090 Community pod, image `sha256:b80b645d…866aaf`, repo
`f77d05d7b35187bf0da7e7b94a629549fe2e1c05`. **The pod was terminated on 2026-09-10 and the
artifacts quoted in §1 exist nowhere else** — this report is their only copy.

---

## 1. What happened

The first official E1 launch (real run, `seed=42`, `max_iters=80000`, `batch_size=16`,
`num_workers=12`) started cleanly, **completed iteration 1**, and died in iteration 2's forward
with a CUDA OOM. No checkpoint was written — `ckpt_interval=2000` was never reached — so there is
no partial run to discard under the no-resume invariant ([AGENTS.md](../AGENTS.md), E1 invariants).

### 1.1 `/workspace/e1_ckpts/e1_stdout.log` — complete

```text
nohup: ignoring input
[mode] REAL | torch 2.1.0+cu121 | device=cuda | cuda_available=True | num_classes=116
[budget] max_iters=80000 batch_size=16 val_interval=4000 max_val_batches=None num_workers=12 seed=42
Downloading: "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth
100%|██████████| 21.1M/21.1M [00:00<00:00, 140MB/s]
[student] params=2,933,688 used_pretrained=True (pretrained arg='torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2')
[data] train_index=5367 val_index=846 (index globbed; only the batches pulled below are decoded)
[loss] CombinedCEDiceLoss(weight=len116, ignore_index=255) weight_on=cuda:0
[opt] SGD lr=0.01 momentum=0.9 weight_decay=0.0001 | scheduler=PolynomialLR (total_iters=80000, power=0.9)
[grad-clip] disabled (no concrete E1 max_norm)
[ckpt] dir=/workspace/e1_ckpts (verified OUTSIDE repo)
[provenance] git_head=f77d05d7b35187bf0da7e7b94a629549fe2e1c05 (source=image_env) image_digest=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
[jsonl] telemetry -> /workspace/e1_ckpts/e1_telemetry.jsonl
/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:3053: UserWarning: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
  return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
Traceback (most recent call last):
  File "/workspace/plantseg-thesis/src/training/train_e1.py", line 648, in <module>
    raise SystemExit(main())
  File "/workspace/plantseg-thesis/src/training/train_e1.py", line 639, in main
    return run(mode=mode, device=device, pretrained=pretrained, batch_size=batch_size,
  File "/workspace/plantseg-thesis/src/training/train_e1.py", line 425, in run
    logits = student(img)
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspace/plantseg-thesis/src/models/student.py", line 149, in forward
    return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
  File "/usr/local/lib/python3.11/site-packages/torch/nn/functional.py", line 4018, in interpolate
    return importlib.import_module('torch._decomp.decompositions').upsample_bilinear2d_vec(
  File "/usr/local/lib/python3.11/site-packages/torch/_decomp/decompositions.py", line 2976, in upsample_bilinear2d_vec
    return upsample_bilinear2d(input, osize, align_corners, scale_h, scale_w)
  File "/usr/local/lib/python3.11/site-packages/torch/_decomp/decompositions.py", line 70, in inner
    r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
  File "/usr/local/lib/python3.11/site-packages/torch/_decomp/decompositions.py", line 3045, in upsample_bilinear2d
    result = torch.mul(q1, yscale1) + torch.mul(q2, yscale2)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.81 GiB. GPU 0 has a total capacty of 23.53 GiB of which 361.69 MiB is free. Including non-PyTorch memory, this process has 23.16 GiB memory in use. Of the allocated memory 20.68 GiB is allocated by PyTorch, and 1.97 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

_(Caret/underline marker lines from the traceback are omitted; every source line is verbatim.)_

### 1.2 `/workspace/e1_ckpts/e1_telemetry.jsonl` — complete, both rows

```json
{"event": "run_meta", "wall_clock": 1789049098.1387577, "mode": "real", "seed": 42, "git_head": "f77d05d7b35187bf0da7e7b94a629549fe2e1c05", "git_head_source": "image_env", "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf", "torch": "2.1.0+cu121", "numpy": "1.26.4", "device": "cuda", "cuda_available": true, "gpu_name": "NVIDIA GeForce RTX 4090", "num_workers": 12, "batch_size": 16, "max_iters": 80000, "val_interval": 4000, "max_val_batches": null, "ckpt_interval": 2000, "resumed_from": null, "num_classes": 116, "ignore_index": 255, "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0001, "lr_power": 0.9, "poly_horizon": 80000, "grad_clip_norm": null, "used_pretrained": true, "params": 2933688}
{"event": "train", "iter": 1, "loss": 9.220895767211914, "ce": 8.22772216796875, "dice": 0.9931731820106506, "lr": 0.009999887499929687, "wall_clock": 1789049105.42647, "iter_seconds": 7.287464380264282, "samples_per_sec": 2.195551040130059}
```

**The run was numerically healthy.** CE 8.228 against a 116-class weighted objective and Dice
0.993 at near-zero overlap are both what an ImageNet-initialised student should produce at
iteration 1. It died on memory, not on the model, the data, or the loss.

## 2. Root cause `[MEASURED — two-arm probe, RTX 4090, 2026-09-10]`

A throwaway probe (`/tmp/vram_probe.py`, not committed) ran the same step twice, flipping exactly
one variable: `torch.use_deterministic_algorithms`, which [src/seeds.py](../src/seeds.py)`:37`
sets to `True, warn_only=True`. cuDNN stayed `deterministic=True` / `benchmark=False` in both
arms. Batch, resolution, `num_classes`, and every locked flag were unchanged; the probe used
synthetic tensors, no dataloader, no download, and wrote nothing.

| | det **off** | det **on** | delta |
|---|---|---|---|
| Final-upsample `grad_fn` | `UpsampleBilinear2DBackward0` | `AddBackward0` → 2× `MulBackward0` | native → **decomposed** |
| That op alone at `[16,116,64,64]→[16,116,512,512]`, fwd+bwd, peak | **3.684 GiB** | **16.342 GiB** | **+12.658 GiB** |
| Full E1 step, peak | **17.014 GiB** | **20.667 GiB** | +3.653 GiB |
| Full E1 step, *allocated* after forward | 6.139 GiB | 6.139 GiB | **0.000 GiB** |

**The last row is the finding.** Resident memory after the forward is identical to the
milligibibyte across arms, and every later phase matches too (CE 7.951 / Dice 11.592 /
post-backward 1.913 in both). The extra 12.658 GiB is *transient*, allocated and freed entirely
inside one call to `F.interpolate`.

The arithmetic closes: activations before the final upsample are `6.139 − 1.812 = 4.327 GiB`;
adding the isolated op's 16.342 GiB gives **20.669**, against a measured full-step peak of
**20.667**. One op accounts for the entire difference between fitting and not.

**This was never an inference.** `functional.py:4018` in the traceback names the substitution
outright, and `decompositions.py:3045` — `torch.mul(q1, yscale1) + torch.mul(q2, yscale2)` — is
literally the graph the probe printed: `AddBackward0` with two `MulBackward0` children. The crash
and the probe reached the same expression by different routes.

**The decomposition is in the backward too, and it does not raise the requirement.** Because the
autograd graph is built from decomposition primitives rather than one fused node, `backward()`
replays them. But in the on arm `peak_alloc` reads 20.667 after forward and is *still* 20.667
after CE, after Dice, after backward, and after `optimizer.step` — the replay is cheaper, since by
then only the incoming gradient is large. **The forward is the binding constraint.**

## 3. Why iteration 2, not iteration 1

Peak demand (20.667 GiB) *fits* on a 23.53 GiB card. The run died on fragmentation stacked on top
of it. From the traceback's own numbers:

| | GiB |
|---|---|
| Card total | 23.53 |
| Non-PyTorch (CUDA context): `23.16 − 20.68 − 1.97` | 0.51 |
| **Available to PyTorch** | **23.02** |
| **Peak demand** (measured, clean allocator) | **20.67** |
| **Margin** | **2.35** |
| **"Reserved but unallocated" at failure** | **1.97** |
| Free at failure | 0.35 |

Fragmentation consumed **84% of the margin**, leaving 361.69 MiB against a 1.81 GiB request. The
mechanism follows from §2: iteration 1 allocates and frees a 16.3 GiB transient, leaving the
reserved pool split into blocks that iteration 2's identically-shaped request cannot reuse. The
probe never saw this because it ran one step from a clean allocator.

**A retry without a change fails identically.** Every tensor shape in the step is fixed, so the
allocation sequence is deterministic: iteration 1 succeeds, iteration 2 OOMs, every time.

## 4. F7 — resolved on-pod, and not in the predicted direction

[b31_e1_fixes.md](b31_e1_fixes.md)`:1547` (R4) states the claim and its verification method:

> **F7 CUDA determinism.** `F.interpolate(bilinear)` and `AdaptiveAvgPool2d` backward have no
> deterministic CUDA kernel. […] run E1 with `PYTHONWARNINGS=always` and grep for
> `does not have a deterministic implementation`

The launch executed that grep by accident. **Neither op F7 names produced the warning.** The op
that did — `nll_loss2d_forward_out_cuda_template`, the CE forward — F7 never mentions.

The audit marked this `CANNOT-VERIFY-LOCALLY` at [pre_e1_launch_audit.md](pre_e1_launch_audit.md)
`:58` and `:358`, and predicted the consequence at `:4006`:

> Either way the E1 run is **unaffected**, because E1 uses `warn_only=True` per ch3 — this probe
> only tells you whether the warning list will be noisy.

**That prediction is refuted.** The run was not unaffected; it died. The error is that `:4006`
treats one mechanism where there are two:

- **An op with a deterministic alternative is silently rerouted.** `F.interpolate` did not warn
  because it did not need to — `functional.py:4018` substituted the decomposition. `warn_only` is
  irrelevant on this path. The cost is not noise, it is **12.658 GiB** (§2).
- **An op without one is warned about and runs non-deterministically anyway.** That is CE (§5).

F7's structural premise ("no deterministic CUDA kernel") is therefore right about the *native*
kernel and wrong about the consequence: PyTorch substitutes rather than giving up, and the price
is memory and time, not determinism.

**Two limits on this, stated rather than glossed:**

1. `PYTHONWARNINGS=always` was **not** set, so Python's default first-occurrence filter applied. A
   warning from a different op and line would still have appeared once, so `AdaptiveAvgPool2d`'s
   silence across a *completed* forward and backward is suggestive — but the run reached only
   iteration 2, and this is not the controlled probe.
2. The audit's own §7.1 probe at `:3984` was **never run**, and would not have caught the OOM if it
   had been: it uses `batch 2`, where the decomposition costs roughly 2 GiB and is invisible. It
   was designed to answer a determinism question and cannot surface a memory consequence at that
   batch.

## 5. The determinism gap that is actually there `[MEASURED]`

Reproduced both in the launch (§1.1) and in the probe's on arm:

```text
torch/nn/functional.py:3053: UserWarning: nll_loss2d_forward_out_cuda_template does not have a
deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'
```

`warn_only=True` at [src/seeds.py](../src/seeds.py)`:37` permits it, so **the CE forward runs a
non-deterministic CUDA kernel in every E1 run on GPU.** This predates today and is independent of
the OOM — it is a gap in what contract B6's determinism protocol actually delivers, not a
consequence of the crash. It fires only on CUDA, which is why no local run surfaced it.

No action taken: `src/seeds.py` is a governed path and the response is a methodology decision.
Recorded here so it is known rather than rediscovered.

## 6. Dice sets the deterministic-off floor `[MEASURED]`

The off-arm peak is 17.014 GiB, not the ~11–14 GiB that had been inferred before measurement. The
difference is the Dice term, and it is unrelated to determinism — it is present in both arms.

In the off arm, allocation rises `7.951 → 11.592 GiB` across Dice (**+3.641 retained**) while peak
rises `7.951 → 15.248` (**+7.297 transient**). [src/training/losses.py](../src/training/losses.py)
`:62-77` materialises repeated `[16,116,512,512]` tensors — `:64` softmax, `:68` one-hot, `:69`
and `:70` the validity masks, `:72` the product. At this shape fp32 is 1.812 GiB per tensor, and
the retained 3.641 GiB matches `probs` + `onehot` held for backward.

The single largest allocation is `:68`: `F.one_hot(tgt, num_classes)` produces **int64** before
`.float()`, so that one expression allocates **3.625 GiB** — twice the fp32 size — and then 1.812
more on the cast. No line-by-line budget for the full 7.297 GiB transient is asserted here; only
the measured totals and the largest identified contributor.

This is recorded as an observation, not a proposal. Changing the loss implementation is a governed
path and a methodology question.

## 7. Resolution — **Option A, decided 2026-09-10**

**Rent a 48 GB card and change nothing else.** Determinism stays on, `src/seeds.py` is untouched,
and every E1 invariant holds. The basis is §2's measurement: the deterministic decomposition costs
**16.342 GiB against the native kernel's 3.684 GiB** at the real shape, and no option that keeps
determinism fits inside 24 GB.

RTX A6000 (~$0.33/hr) and A40 (~$0.35/hr) were the candidates at decision time. Both are sm_86,
inside the `MAX_SM = (9, 0)` ceiling at [scripts/verify_env.py](../scripts/verify_env.py)`:32`, so
the capability gate at `:133-146` passes — and is re-exercised as preflight stage 2 on the new pod
regardless. **Rates are time-varying and `[UNVERIFIED]` here**; confirm at rental. The property
that decided it is that a 48 GB card was available at or below the rate of the 24 GB card that
just failed, so doubling VRAM cost nothing.

The measured options, retained as the basis for the decision:

| | Change | Peak vs available | Cost |
|---|---|---|---|
| **A** | 48 GB card inside the sm_90 ceiling, nothing else | 20.67 of ~45+ | New rental; no code, config, or methodology change. E1 invariants untouched. |
| **B** | Disable or narrow `use_deterministic_algorithms` | 17.01 of 23.02 → **6.0 GiB margin** | Governed edit to `src/seeds.py:37`. Trades the deterministic upsample backward that contract B6 assumes — but see §5 on what the flag is currently delivering. Also removes the decomposition's per-iteration time cost (§8). A narrower variant (scoping the flag around the final upsample only) is a `src/models/student.py` edit. |
| **C** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 20.67 of 23.02 → **2.35 GiB margin** | Environment only, no code change. Targets exactly the 1.97 GiB of §3. Thinnest margin: ~10% at a hard floor, held for 80,000 iterations, with the val↔train transition re-shuffling the pool every 4,000. |

**Why B was rejected.** It would trade a contract-B6 guarantee for a card that costs the same. The
6.0 GiB margin is real, and §5 shows the flag is not currently delivering full determinism — but
neither makes the trade worth taking when the alternative is free. §5's gap is a separate question
that this decision does not settle. B would also have required a session explicitly approved to
edit governed paths; A does not.

**Why C was rejected.** At failure the process held 23.16 of 23.53 GiB — there is no headroom to
recover, only fragmentation. And validation is a sustained stressor on precisely the mechanism C
targets: `max_val_batches=None` over `val_index=846` at batch 16 is **53 consecutive
full-resolution forwards**, each carrying the decomposition transient, every 4,000 iterations.
`[MEASURED]` validation's *peak* is below training's, because `validate` is `@torch.no_grad()`
([src/training/train_e1.py](../src/training/train_e1.py)`:271`) and moves predictions to CPU before
the confusion matrix (`:282`); the objection is fragmentation across 53 allocations, not peak.

## 8. Not measured — steady-state throughput

Iteration 1 took **7.287 s** (2.196 samples/s). Multiplying that by 80,000 gives 162 hours, but
**that number is an artifact and is not a projection**: iteration 1 carries DataLoader worker
fork and prefetch fill, the first batch decoded cold, CUDA kernel and cuDNN handle load, and a
cold allocator issuing large `cudaMalloc`s for a transient it has never allocated.

`[ESTIMATE — not measured, arithmetic shown so it can be checked]` The decomposition writes
16.342 GiB per call (§2); counting reads, order 46 GiB of traffic per forward and similar in
backward, against the 4090's ~1 TB/s, is order **0.15–0.2 s per iteration**, or roughly 3.5–4.5
hours across the run — against roughly 3 ms per iteration for the native kernel. For the run to
take days it would need ~7 s/iteration *sustained*, which nothing in the log supports.

**This is unresolved and should be settled before the next 80,000-iteration commitment.** It is a
two-minute probe on any GPU pod: time the isolated upsample in both arms at the real shape (both
fit — the on arm peaks at 16.342 GiB) plus one deterministic-off full step. It was offered on the
source pod and not run before termination; taking it on the next pod costs nothing extra, since a
relaunch requires a pod regardless.

## 9. What this commit changes

| File | Change |
|---|---|
| `reports/b48_e1_oom_investigation.md` | This report. |
| `reports/pre_e1_launch_audit.md` | F7's status line (`:58`) gains a resolution pointer; §7.1's interpretation (`:4006`) is marked superseded by §4 above. The original text is preserved, not rewritten. |

## Provenance / guardrails honored

- **No governed path modified.** `src/**`, `configs/**`, `scripts/**`, `requirements*`,
  `docs/EVALUATION_CONTRACT.md`, `docs/IMPLEMENTATION_CONTRACT.md` untouched. §5, §6 and option B
  in §7 describe changes and stop there.
- **No training, download, install, or GPU use from this session.** All measurements were taken on
  the pod before its termination and are transcribed here.
- **No config value changed to make anything pass.** Batch size, resolution, seed, `num_workers`
  and every locked flag are as launched; the probe reproduced them rather than adjusting them.
- **The failed run is discarded, not resumed.** No checkpoint was written, so the no-resume
  invariant for official runs is satisfied trivially.
- `docs/reference/reference.pdf` untouched.
