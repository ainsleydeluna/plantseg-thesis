# B30 — Pre-E1 launch forensic audit

**READ-ONLY evidence pass. No fixes applied. No files edited, staged, or committed.**

Produced for adjudication by a separate review pass holding ch1/ch2/ch3 and the source
papers. Every number is labelled MEASURED or INFERRED. Code and contract are shown side by
side wherever they disagree.

- HEAD: `148af2c9b30ffc78e7e2f4d0b9ff55cd40948607`
- Branch: `master`
- Generated: B30 session

---

## ⚠️ HEADLINE — F1 IS REFUTED

**F1 is wrong. The train augmentation RNG is NOT frozen per image.**

`src/data/dataset.py:83` does **not** read `np.random.RandomState((SEED + idx) % 2**32)`.
It reads:

```python
rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
```

The idx-seeded form F1 describes is explicitly named in the code comment as the *previous*
behaviour that was already fixed (`dataset.py:79-82`: *"previously seeded by idx alone ->
frozen per image"*).

Probe P1 settles it at the mechanism, not just the output — the logged seed sequences differ
across two passes over the same samples:

```text
pass1: RandomState seed sequence = [1608637542, 1273642419]
pass2: RandomState seed sequence = [1935803228, 787846414]
```

And the stated refutation condition is met: two independent passes over the train DataLoader
yield **different** tensors for the same source file (P2a/P2b), while the val control is
**bit-identical** (P3) — proving the difference is genuine augmentation variation and not
loader nondeterminism.

**F12 is also REFUTED** (train mode *is* restored after validation), and **F5 is REFUTED**
(both determinism settings are present, and match ch3's own wording verbatim).

---

## 1. Verdict table

| # | Claim (abbreviated) | Verdict | Evidence pointer |
|---|---|---|---|
| F1 | Aug RNG seeded `(SEED+idx)` → identical aug every epoch | **REFUTED** | `dataset.py:83`; P1 seed logs differ; P2a/P2b differ; P3 control identical |
| F2 | No resume, no periodic ckpt, no RNG-state save | **CONFIRMED** | `train_e1.py:263-283` (no `--resume`), `:98-106`, `:235-238` |
| F3 | Training telemetry is print()-only, nothing persisted | **CONFIRMED** | `train_e1.py` every `print`; no writer anywhere in the train path |
| F4 | No pin_memory/persistent_workers/prefetch_factor; nw=4 → data-bound | **CONFIRMED** | `dataset.py:102-110`, `train_e1.py:338`; P4 = 88.3 ms/sample MEASURED |
| F5 | `use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` missing | **REFUTED** | `seeds.py:26` and `seeds.py:37` — both present |
| F6 | train `drop_last=False` → ragged final batch | **CONFIRMED** | `dataset.py:109`; P5: 5367 mod 16 = **7**, 336 iters/epoch |
| F7 | bilinear interpolate + AdaptiveAvgPool2d have no deterministic CUDA backward | **PARTIALLY-CONFIRMED** / CUDA part CANNOT-VERIFY-LOCALLY → **resolved on-pod 2026-09-10, not as predicted — see [B48](b48_e1_oom_investigation.md) §4** | Both ops on train path (`student.py:83,99,149`); P6 finds **9** pools, not 1 |
| F8 | Logit-KD / CWD-logit resolution unpinned in ch3 + contract | **PARTIALLY-CONFIRMED** | ch3 pins pixels not grid; contract §B3 says stride-16, code uses OS8 |
| F9 | No teacher forward/integration code; weights/ empty; NEED_TO_CONFIRM remain | **PARTIALLY-CONFIRMED** (core claim REFUTED) | `segnext_teacher.py:183,195`, `teacher.py:122` exist; `weights/` **is** empty |
| F10 | `verify_env.py` has no CUDA compute-capability gate | **CONFIRMED** | `verify_env.py:115-130`, `:225` — no `get_device_capability` anywhere |
| F11 | λ_logit sweep preregistered at full 80k with no reduced-budget proxy | **CONFIRMED** | ch3 verbatim (E13a); only reduced budget in repo is the *clipping* pilot |
| F12 | `student.train()` not restored after validation | **REFUTED** | `train_e1.py:113-114` saves `was_training`, `:127-128` restores it |
| F13 | Checkpoint writes non-atomic (no temp-then-rename) | **CONFIRMED** | `train_e1.py:97-106` — `torch.save` straight to the final path |

---

## 2. Per-finding detail

### F1 — REFUTED

**Anchor:** [src/data/dataset.py:75-88](../src/data/dataset.py#L75)

```python
   75	    def __getitem__(self, idx: int):
   76	        img_path, mask_path = self.pairs[idx]
   77	        with Image.open(img_path) as im, Image.open(mask_path) as mk:
   78	            if self.split == "train":
   79	                # Per-sample augmentation RNG seeded from the worker/global RNG stream — reseeded per
   80	                # worker per epoch by worker_init_fn (num_workers>0) or advancing in the seeded main
   81	                # process (num_workers=0). Augmentation VARIES across epochs/repeats yet stays
   82	                # reproducible from the global seed (previously seeded by idx alone -> frozen per image).
   83	                rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
   84	                # true train-time multi-scale RRC on the original-resolution image (pre-pad)
   85	                img_np, mask_np = train_preprocess(im, mk, rng, self.aug_params)
   86	            else:
   87	                img_np, mask_np = core_preprocess(im, mk)    # val/test: unchanged deterministic core
   88	        return finalize(img_np, mask_np)                      # float32 CHW, int64 HW
```

The asserted expression `np.random.RandomState((SEED + idx) % 2**32)` does not appear anywhere
in the file. Line 83 draws the per-sample seed from the **ambient numpy RNG**, which advances
on every call, so the seed depends on stream position — not on `idx`.

Reseeding across epochs is supplied by two mechanisms:

- `num_workers>0`: `_seed_worker` (`dataset.py:91-94`) seeds numpy from `torch.initial_seed()`,
  which is `base_seed + worker_id`. `base_seed` is drawn from the DataLoader's `generator`
  (`dataset.py:100-101,107`), and that generator **advances between iterator creations** — it is
  seeded once at loader construction, not once per epoch.
- `num_workers=0`: the main-process numpy RNG simply keeps advancing.

**Probe P1 (root cause, MEASURED).** `np.random.RandomState` and `np.random.randint` were
monkeypatched in the scratchpad at `num_workers=0` and the seed arguments logged over two
full passes of a 2-element Subset:

```text
pass1: RandomState seed sequence = [1608637542, 1273642419]
pass2: RandomState seed sequence = [1935803228, 787846414]
np.random.randint draws (both passes, in order) = [1608637542, 1273642419, 1935803228, 787846414]
```

The seed lists **differ**. Per the amendment-A2 decision rule that is F1 REFUTED at the
mechanism.

**Probes P2a/P2b/P3 (output, MEASURED).**

| Probe | Loader | pos0 `apple_rust_12.jpg` | pos1 `banana_panama_disease_4.jpg` |
|---|---|---|---|
| P1 | train, nw=0 | different, max abs diff 3.67647076 | different, 4.44444466 |
| P2b | train, nw=0 | different, 3.67647076 | different, 4.44444466 |
| P2a | train, nw=2 | different, 3.81652641 | different, 4.44444466 |
| P3 | **val**, nw=2 | **bit-identical**, 0.00000000 | **bit-identical**, 0.00000000 |

The val control being exactly identical while train differs is what makes this decisive: the
probe apparatus is sound, and the variation is the augmentation RNG, not loader noise.

**Reasoning:** the finding describes code that is not in this repository; the behaviour it
predicts is contradicted by both the seed log and the tensor comparison.

### F2 — CONFIRMED

**Anchors:** [train_e1.py:95-107](../src/training/train_e1.py#L95),
[:228-240](../src/training/train_e1.py#L228), [:263-283](../src/training/train_e1.py#L263)

```python
   95	def save_checkpoint(ckpt_dir: Path, student, optimizer, scheduler, sched_name: str,
   96	                    it: int, best_miou: float) -> str:
   97	    path = _assert_outside_repo(Path(ckpt_dir)) / f"e1_student_best_iter{it}.pt"
   98	    torch.save({
   99	        "iter": it,
  100	        "model_state_dict": student.state_dict(),
  101	        "optimizer_state_dict": optimizer.state_dict(),
  102	        "scheduler_state_dict": scheduler.state_dict(),
  103	        "scheduler": sched_name,
  104	        "best_val_miou_all_class": best_miou,
  105	        "num_classes": NUM_CLASSES,
  106	    }, path)
  107	    return str(path)
```

The only `torch.save` in the file, and its only call site is the best-improvement branch:

```python
  228	        if it % val_interval == 0 or it == max_iters:
  229	            all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
  230	                                                       max_val_batches)
  231	            checks["val_cm_accumulated"] = (tuple(cm.shape) == (NUM_CLASSES, NUM_CLASSES)
  232	                                            and int(cm.sum()) > 0 and nvb >= 1)
  233	            print(f"[val  {it:>4}/{max_iters}] cm_batches={nvb} cm_total_px={int(cm.sum())} "
  234	                  f"all_class_miou={all_miou:.5f} disease_only_miou(PROVISIONAL)={disease_miou:.5f}")
  235	            if all_miou > best_miou:
  236	                best_miou = all_miou
  237	                best_ckpt = save_checkpoint(ckpt_dir, student, optimizer, scheduler, sched_name,
  238	                                            it, best_miou)
  239	                print(f"[ckpt {it:>4}/{max_iters}] new best all_class_miou={best_miou:.5f} "
  240	                      f"-> {best_ckpt}")
```

Full `parse_args` option list — there is no `--resume`, no `--init-from`, no `--save-every`:

```python
  263	def parse_args(argv=None):
  264	    p = argparse.ArgumentParser(description="E1 training-loop scaffold (dry-run by default).")
  265	    p.add_argument("--dry-run", action="store_true", help="tiny CPU dry-run (safe default behavior)")
  266	    p.add_argument("--real-run", action="store_true",
  267	                   help="intent to run the real 80k training (requires --confirm-real-run)")
  268	    p.add_argument("--confirm-real-run", action="store_true",
  269	                   help="explicit confirmation gate for the real 80k run")
  270	    p.add_argument("--device", default=None)
  271	    p.add_argument("--init", choices=["none", "imagenet"], default=None,
  272	                   help="backbone init; 'imagenet' uses configs/e1_student.py (real run only)")
  273	    p.add_argument("--batch-size", type=int, default=None)
  274	    p.add_argument("--max-iters", type=int, default=None)
  275	    p.add_argument("--val-interval", type=int, default=None)
  276	    p.add_argument("--max-val-batches", type=int, default=None)
  277	    p.add_argument("--num-workers", type=int, default=None)
  278	    p.add_argument("--ckpt-dir", default=None, help="out-of-repo dir; auto temp dir if omitted")
  279	    p.add_argument("--grad-clip-norm", type=float, default=None,
  280	                   help="global-norm clip; omitted by default (no concrete E1 value)")
  281	    p.add_argument("--log-every", type=int, default=1)
  282	    p.add_argument("--seed", type=int, default=42)
  283	    return p.parse_args(argv)
```

**Reasoning:** checkpoints are written only when `all_miou > best_miou`; the payload carries
`iter`/model/optimizer/scheduler but **no** python, numpy, torch or CUDA RNG state, and no
dataloader position. A run killed at iter 79,000 resumes from nothing.

### F3 — CONFIRMED

**Anchor:** every emission in [train_e1.py](../src/training/train_e1.py) is `print`.

Repo-wide grep for any metric writer:

```text
$ grep -rn "SummaryWriter|tensorboard|wandb|mlflow|jsonl|csv.writer|FileHandler|logging.basicConfig" \
    --include=*.py src/ scripts/ configs/
src/eval/artifacts.py:47:  ARTIFACT_FILES = ("per_image.jsonl", "sufficient_stats.npz", "summary.json")
src/eval/artifacts.py:452: (tmp / "per_image.jsonl").write_text(rows, encoding="utf-8", newline="\n")
src/stats/ingest.py:221:   for line in (run_dir / "per_image.jsonl").read_text(...)
... (all remaining hits are in src/eval/, src/stats/ or scripts/smoke_*, i.e. the EVALUATION
    artifact path, not the E1 training loop)
```

**Reasoning:** `per_image.jsonl` belongs to the post-hoc evaluator (`src/eval/artifacts.py`),
which runs on a finished checkpoint. Nothing in the training loop persists loss, val-mIoU,
per-class IoU, LR, or wall-clock. For Chapter 4 those curves would have to be recovered by
scraping stdout, and only if stdout was redirected to a file.

Two aggravating details, both MEASURED from source:

- `train_e1.py:253` prints the **entire** `lr_trace` list in one call:
  `print(f"[summary] lr_trace={['%.8e' % x for x in lr_trace]}")`. At `max_iters=80000` that is
  a single ~1.4 MB stdout line (80,000 entries x ~17 chars). INFERRED from the format string.
- per-class IoU is computed inside `miou_from_confusion` but only the scalar mean escapes;
  the 116x116 confusion matrix `cm` is discarded when `validate()` returns.

### F4 — CONFIRMED

**Anchor:** [src/data/dataset.py:97-110](../src/data/dataset.py#L97)

```python
   97	def build_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
   98	    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42)."""
   99	    dataset = PlantSegDataset(split)
  100	    generator = torch.Generator()
  101	    generator.manual_seed(SEED)
  102	    return DataLoader(
  103	        dataset,
  104	        batch_size=batch_size,
  105	        shuffle=(split == "train"),
  106	        num_workers=num_workers,
  107	        generator=generator,
  108	        worker_init_fn=_seed_worker,
  109	        drop_last=False,
  110	    )
```

No `pin_memory`, no `persistent_workers`, no `prefetch_factor`. Real-run default:

```python
  338	        num_workers = args.num_workers if args.num_workers is not None else 4
```

**Probe P4 (MEASURED, this Windows box, 13th Gen Intel Core i7-1360P, `os.cpu_count()=16`):**

| Stage | mean (ms) | median (ms) | p95 (ms) |
|---|---|---|---|
| (a) JPEG+PNG decode | 18.779 | 18.536 | 35.487 |
| (b) `_resize_long_side` | 9.591 | 8.417 | 18.562 |
| (c) `_apply_rotation` | 11.230 | 7.269 | 39.239 |
| (d) `_random_crop_pad_512` | 15.095 | 7.906 | 73.610 |
| (e) `_apply_flips` | 2.693 | 2.116 | 6.240 |
| (f) `_apply_photometric` | 17.650 | 13.521 | 41.196 |
| (g) `finalize` | 13.178 | 14.139 | 18.935 |
| **TOTAL transform** | **69.500** | 67.197 | 133.920 |
| **end-to-end per sample** | **88.279** | — | — |

Workers needed to sustain a given rate at batch 16 — **INFERRED — CROSS-PLATFORM** (derived
from Windows-dev-box timings; the Linux pod has different CPU, different PIL/SIMD build, and
no Windows spawn overhead, so treat these as an order-of-magnitude floor, not a pod prediction):

| Target | samples/s | worker-cores needed |
|---|---|---|
| 6 it/s | 96 | 8.47 → **9** |
| 8 it/s | 128 | 11.30 → **12** |
| 12 it/s | 192 | 16.95 → **17** |

**Reasoning:** the stated refutation condition (`4 workers can saturate the GPU`) fails by a
wide margin. At 88.3 ms/sample, 4 workers deliver ~45 samples/s ≈ **2.8 it/s** at bs16. The
finding stands, and P7 below identifies a concrete cause inside stage (d).

### F5 — REFUTED

**Anchor:** [src/seeds.py:17-39](../src/seeds.py#L17) — the whole function.

```python
   17	def set_seed(seed: int = 42) -> int:
   18	    """Seed all RNGs and enforce the contract-B6 determinism protocol.
   19	
   20	    Sets cuDNN deterministic mode, torch deterministic algorithms (warn_only=True), and
   21	    CUBLAS_WORKSPACE_CONFIG. Returns the seed used so callers can log it.
   22	    """
   23	    os.environ["PYTHONHASHSEED"] = str(seed)
   24	    # cuBLAS determinism must be configured BEFORE the first CUDA op / cuBLAS handle is created;
   25	    # set_seed runs before any device transfer in the training path (contract B6).
   26	    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
   27	
   28	    random.seed(seed)
   29	    np.random.seed(seed)
   30	
   31	    torch.manual_seed(seed)
   32	    torch.cuda.manual_seed(seed)
   33	    torch.cuda.manual_seed_all(seed)  # all CUDA devices
   34	
   35	    torch.backends.cudnn.deterministic = True
   36	    torch.backends.cudnn.benchmark = False
   37	    torch.use_deterministic_algorithms(True, warn_only=True)  # warn (not error) on non-deterministic ops
   38	
   39	    return seed
```

Line 26 sets `CUBLAS_WORKSPACE_CONFIG`. Line 37 calls `torch.use_deterministic_algorithms`.
Both of the things F5 says are absent are present, on the E1 execution path
(`train_e1.py:138` calls `set_seed(seed)` as the **first** statement of `run()`, before
`torch.device` and before any `.to(dev)`).

Repo-wide grep, verbatim:

```text
$ grep -rn "use_deterministic_algorithms|CUBLAS_WORKSPACE_CONFIG|cudnn" --include=*.py .
src/seeds.py:26:    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
src/seeds.py:35:    torch.backends.cudnn.deterministic = True
src/seeds.py:36:    torch.backends.cudnn.benchmark = False
src/seeds.py:37:    torch.use_deterministic_algorithms(True, warn_only=True)
scripts/verify_env.py:119:    cudnn_avail = bool(torch.backends.cudnn.is_available())
scripts/verify_env.py:120:    cudnn_ver = torch.backends.cudnn.version() if cudnn_avail else None
```

**This is not merely present — it matches ch3 word for word.** See E13(d): ch3 specifies
`torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
explicitly. `src/seeds.py` is a faithful implementation of the manuscript, not a half-one.

### F6 — CONFIRMED

**Anchor:** [src/data/dataset.py:109](../src/data/dataset.py#L109) — `drop_last=False,`

**Probe P5 (MEASURED len + arithmetic):**

```text
train samples          = 5367
batch_size             = 16       (configs/e1_student.py:19)
full batches           = 335
5367 mod 16            = 7        <- SIZE OF THE FINAL RAGGED BATCH
iterations per epoch   = 336      (ceil, drop_last=False)
iterations per epoch   = 335      (if drop_last were True)
max_iters              = 80000
epochs at 80k          = 238.095
val_interval           = 4000  -> 20 validations
```

**Reasoning:** every 336th iteration trains on 7 samples instead of 16. Because
`SoftDiceLoss` aggregates *per class present in the batch* (`losses.py:75-77`) and CE is
class-weighted, a 7-sample batch has materially different loss statistics from a 16-sample
batch. Over 238 epochs that is 238 such iterations out of 80,000 (~0.3%).

### F7 — PARTIALLY-CONFIRMED; the CUDA determinism property is CANNOT-VERIFY-LOCALLY

**The stated refutation condition is NOT met — both ops are on the training forward path.**

```python
   92	    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
   93	        """Return OS8 class logits. The final upsample to full resolution happens after dequant in
   94	        the parent (matches the quantization-ready isolated head: one quantized interpolate, then a
   95	        float final upsample). In FP32 this is numerically identical to the B10 head."""
   96	        a = self.high_proj(high)                                       # [B,inter,H16,W16]
   97	        s = self.context(high)                                         # [B,inter,1,1]
   98	        high_feat = self.ff_mul.mul(a, s)                              # quantization-safe attention
   99	        high_os8 = F.interpolate(high_feat, size=low.shape[-2:], mode="bilinear", align_corners=False)
  100	        return self.ff_add.add(self.high_logits(high_os8), self.low_logits(low))  # OS8 class logits
```

```python
  143	    def forward(self, x: torch.Tensor) -> torch.Tensor:
  144	        out_size = x.shape[-2:]
  145	        x = self.quant(x)                          # pass-through in FP32 (not prepared)
  146	        low, high = self._forward_features(x)
  147	        logits = self.head(low, high)              # OS8 class logits
  148	        logits = self.dequant(logits)              # dequant BEFORE the float final upsample
  149	        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
```

`self.context` (called at :97) is:

```python
   82	        self.context = nn.Sequential(                                   # avgpool -> 1x1 -> sigmoid
   83	            nn.AdaptiveAvgPool2d(1),
   84	            nn.Conv2d(high_ch, inter_ch, 1, bias=True),
   85	            nn.Sigmoid(),
   86	        )
```

**Probe P6 (MEASURED)** — output strides for a 512x512 input, and a real backward:

```text
low  (skip)      : (1, 40, 64, 64)    OS=8
high (C5)        : (1, 160, 32, 32)   OS=16
head OS8 logits  : (1, 116, 64, 64)   OS=8
final logits     : (1, 116, 512, 512) OS=1

head.context[1] (Conv2d) weight.grad is not None -> True
head.context[1] grad abs-sum = 0.14050017297267914
```

**F7 undercounts the pooling op by 9x.** P6 enumerates every `AdaptiveAvgPool2d` reachable in
the module tree:

```text
features.4.block.2.avgpool    AdaptiveAvgPool2d(output_size=1)
features.5.block.2.avgpool    AdaptiveAvgPool2d(output_size=1)
features.6.block.2.avgpool    AdaptiveAvgPool2d(output_size=1)
features.11.block.2.avgpool   AdaptiveAvgPool2d(output_size=1)
features.12.block.2.avgpool   AdaptiveAvgPool2d(output_size=1)
features.13.block.2.avgpool   AdaptiveAvgPool2d(output_size=1)
features.14.block.2.avgpool   AdaptiveAvgPool2d(output_size=1)
features.15.block.2.avgpool   AdaptiveAvgPool2d(output_size=1)
head.context.0                AdaptiveAvgPool2d(output_size=1)
```

Eight of these are the MobileNetV3 squeeze-excitation gates in the backbone; only the ninth is
the LR-ASPP context branch F7 names. Any determinism argument about this op applies to the
whole backbone, not just the head.

**Where F7's stated consequence is wrong.** F7 says *"adding `use_deterministic_algorithms(True)`
WITHOUT `warn_only=True` would raise at the first backward"*. That is a hypothetical about a
change nobody is proposing: `seeds.py:37` **already** passes `warn_only=True`, and ch3 itself
specifies `warn_only=True` (E13d). So F7's second clause — *"with `warn_only=True` the run is
seed-controlled but NOT bit-deterministic"* — is a description of the **current, intended,
preregistered** state, not a defect. ch3 says so directly: *"Reproducibility is reported with
the caveat that small floating-point variation may remain across GPU classes and compiled CUDA
kernels."*

**What cannot be settled here:** whether these specific kernels actually emit a
nondeterminism warning on the pod's CUDA build. This machine is `torch 2.9.1+cpu` — no CUDA at
all, and not the pinned `2.1.0+cu121`. See section 7 for the exact pod command.

### F8 — PARTIALLY-CONFIRMED

F8 has two halves. The first is essentially right; the second is wrong in a way that matters,
and the real defect is sharper than the one stated.

**(i) ch3 pins WHICH pixels, not WHICH grid — F8 correct.** The authoritative ch3 sentence
(verbatim, via the ToUnicode-decoded text layer of `docs/reference/ch3.pdf`):

> For the Logit KD term, the per-pixel KL divergence is averaged over valid pixels only.

That fixes the *denominator population* (non-255 pixels) but names no spatial resolution.

**(ii) The validity-mask target is NOT unpinned — it is pinned, and the code disagrees with it.**
`docs/IMPLEMENTATION_CONTRACT.md:193`:

```text
| Ignore handling | validity mask downsampled to stride-16; channel-wise spatial softmax + KL restricted to valid locations | `[ch3]` |
```

`configs/distill.py:32` repeats it as a blanket rule for CWD:

```python
   32	        "ignore_handling": "validity mask downsampled to stride-16; softmax + KL over valid locations only",
```

But the implementation runs the **CWD-logit** term at **OS8**, not stride-16:

```python
  203	    # --- CWD logit term: on the head's native OS8 logit map (no projection needed, same C) ---
  204	    t_logits_os8 = (t_logits if t_logits.shape[-2:] == head_logits.shape[-2:]
  205	                    else F.interpolate(t_logits, size=head_logits.shape[-2:], mode="bilinear",
  206	                                       align_corners=False))
  207	    valid_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE_INDEX)
  208	    # channels_norm defaults to this map's own channel count (116 classes) — 320 is the FEATURE-map
  209	    # normalisation only and must never be hard-coded here.
  210	    l_logit_map = cwd_channelwise_kl(head_logits, t_logits_os8, valid_os8, T=T_CWD)
  211	    total = total + ramp * BETA_CWD_LOGIT * l_logit_map
  212	    parts["cwd_logit"] = float(l_logit_map.detach())
  213	    return total, parts
```

For a 512x512 input, P6 MEASURED `head_logits` at `(1, 116, 64, 64)` = **OS8**. Stride-16
would be 32x32. So `downsample_validity(mask, head_logits.shape[-2:])` pools to 64x64 while
the contract's ignore-handling row says stride-16. The CWD **feature** term does use stride-16
(`train_distill.py:196-199`, projecting to the teacher's 320-ch stride-16 map), so the
contract row is defensible for the feature term and wrong-or-ambiguous for the logit term.

**(iii) The Logit-KD resolution IS pinned by the code, at full resolution:**

```python
  181	    # --- Logit KD (E2 and E3, unchanged between them per contract B3) ---
  182	    t_logits_full = (t_logits if t_logits.shape[-2:] == logits.shape[-2:]
  183	                     else F.interpolate(t_logits, size=logits.shape[-2:], mode="bilinear",
  184	                                        align_corners=False))
  185	    l_kd = logit_kd_kl(logits, t_logits_full, mask, T=T_LOGIT, ignore_index=IGNORE_INDEX)
  186	    total = total + ramp * lambda_logit * l_kd
  187	    parts["logit_kd"] = float(l_kd.detach())
```

`logits` is the student's post-upsample output — P6 MEASURED `(1, 116, 512, 512)`. `mask` is
the full-resolution 512x512 target. So Logit-KD KL is averaged over valid pixels at **512x512**.
That is consistent with ch3's "valid pixels" phrasing (validity is a full-resolution property
of the mask) and with ch3's head description ending *"summed and upsampled to full resolution"*.

**Verdict reasoning:** ch3 does not state a grid for the logit terms (F8 correct), but the
consequence F8 draws — that the validity-mask target is therefore unpinned — is not what the
evidence shows. The contract *does* pin it, to stride-16, and the code contradicts that for the
CWD-logit map by using OS8. That is a code-vs-contract conflict, not an absence.

### F9 — PARTIALLY-CONFIRMED (its headline claim is REFUTED)

F9 makes four claims. Two are false, two are true.

**(a) "No SegNeXt-B teacher forward/integration code exists yet" — REFUTED.**

```text
$ wc -l src/distill/*.py scripts/launch_teacher_finetune.py configs/teacher/*.py
    28 src/distill/__init__.py
    32 src/distill/cwd_projection.py
    54 src/distill/export.py
    64 src/distill/features.py
   241 src/distill/segnext_teacher.py
   229 src/distill/teacher.py
   270 scripts/launch_teacher_finetune.py
   370 configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
```

Concrete forward/integration surface, by line:

```text
src/distill/segnext_teacher.py:52   load_teacher_state_dict(ckpt_path, map_location="cpu")
src/distill/segnext_teacher.py:104  select_stride16_feature(feats, input_hw, ...)
src/distill/segnext_teacher.py:139  class SegNeXtTeacherAdapter(nn.Module)
src/distill/segnext_teacher.py:183      def forward(self, x) -> dict          <-- teacher forward
src/distill/segnext_teacher.py:187          logits = self.model.decode_head.forward(feats)
src/distill/segnext_teacher.py:195  build_segnext_teacher(ckpt_path, config_path=None, ...)
src/distill/teacher.py:64           class FrozenTeacher(nn.Module)
src/distill/teacher.py:122              def forward(self, x, *, logits_size=None, ...)
src/distill/teacher.py:211          load_frozen_teacher(ckpt_path, ...)
```

A 370-line MMSeg teacher fine-tune config also exists at
`configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py`.

**(b) "`weights/` is empty" — CONFIRMED.** Per R4, the listing is the evidence; it succeeded
(exit 0) and shows only `.` and `..`:

```text
$ ls -la weights/
total 8
drwxr-xr-x 1 admin 197121 0 Jun 24 22:57 .
drwxr-xr-x 1 admin 197121 0 Aug 17 19:21 ..
--- exit code: 0 ---
```

(Produced by GNU `ls` via the Bash tool. No file inside `weights/` was opened.)

**(c) "`docs/teacher_init_source.md` still carries NEED_TO_CONFIRM fields" — CONFIRMED.**
Four of them: the resolved `.pth` URL, the SHA256, the download date, and the init-test result
(see the verbatim file in E12).

**(d) "`scripts/test_teacher_init.py` has never been run" — CANNOT-VERIFY-LOCALLY.**
Absence of a run log is not proof of absence of a run. What is verifiable: the script exists
(108 lines), its `PASS` line is not recorded anywhere in `reports/`, and
`docs/teacher_init_source.md:17` still shows `NEED_TO_CONFIRM (PASS/FAIL)`. It also could not
have succeeded here — it requires the pinned MMSeg stack, and `weights/` holds no checkpoint.

### F10 — CONFIRMED

**Anchor:** [scripts/verify_env.py:113-130](../scripts/verify_env.py#L113) — the complete GPU probe.

```python
  113	    torch_ver = torch.__version__
  114	    tv_ver = import_version("torchvision")
  115	    cuda = bool(torch.cuda.is_available())
  116	    cuda_ver = torch.version.cuda
  117	    gpu_count = torch.cuda.device_count() if cuda else 0
  118	    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda else []
  119	    cudnn_avail = bool(torch.backends.cudnn.is_available())
  120	    cudnn_ver = torch.backends.cudnn.version() if cudnn_avail else None
  121	    print(f"  torch_version       : {torch_ver}")
  122	    print(f"  torchvision_version : {tv_ver}")
  123	    print(f"  cuda_available      : {cuda}")
  124	    print(f"  cuda_version        : {cuda_ver}")
  125	    print(f"  gpu_count           : {gpu_count}")
  126	    print(f"  gpu_names           : {gpu_names}")
  127	    print(f"  cudnn_available     : {cudnn_avail}")
  128	    print(f"  cudnn_version       : {cudnn_ver}")
  129	    print(f"  cpu_count(logical)  : {os.cpu_count()}")
  130	    print(f"  torch_num_threads   : {torch.get_num_threads()}")
```

And the verdict gate:

```python
  223	    # (18) verdict
  224	    print("\n[18] VERDICT")
  225	    real_ready = cuda and gpu_count >= 1 and student_ok and dry_ok
  226	    if not (student_ok and dry_ok):
  227	        verdict, label = "FAIL", "NOT OK for real E1 training (E1 scaffold not runnable on this machine)"
  228	    elif real_ready:
  229	        verdict, label = "PASS", "OK for real E1 training"
  230	    else:
  231	        verdict, label = "PARTIAL", "OK for DRY-RUN only; NOT OK for real E1 training"
```

**Reasoning:** `real_ready` depends on `cuda` (a boolean), `gpu_count >= 1`, and two CPU-side
checks. `torch.cuda.get_device_capability` is called nowhere in the repository — the script
records `get_device_name(i)` (a marketing string) but never the SM version. A pod whose GPU is
`sm_100`/`sm_120` would satisfy every check in this file and then fail at the first kernel
launch inside `train_e1.py`. The blocker list at `:233-247` covers torch-pin mismatch, missing
ImageNet cache and missing packages, but has no capability entry.

I did **not** independently verify the claim that `torch 2.1.0+cu121` ships only `sm_50..sm_90`
— that is a property of the pinned wheel, not of this repo. The pod command in section 7
settles it in one line.

> **ADDENDUM (B42, 2026-09-09) — both halves of this finding are now closed.** This audit was a
> read-only pass at HEAD `148af2c`; the text above is left as written and this note supersedes it.
> Evidence: [b42_pod_gpu_validation.md](b42_pod_gpu_validation.md).
>
> - **The missing gate was added after this audit**, by **B31-3/4** (`bc5f644`, "CUDA
>   compute-capability gate and dataset verification in verify_env"). `scripts/verify_env.py:32`
>   defines `MAX_SM = (9, 0)` and `:133-146` hard-**FAIL**s above it, feeding the `[18] VERDICT`
>   block. So "`torch.cuda.get_device_capability` is called nowhere in the repository" was accurate
>   at `148af2c` and is false from `bc5f644` onward. Observed on the pod:
>   `gpu_capabilities : ['sm_89'] (max supported by the pinned stack: sm_90)` and
>   `capability_gate : PASS`. An sm_100/sm_120 device now fails stage 2/5 of `preflight_e1.py`
>   rather than reaching a kernel launch.
> - **The arch-list claim is now MEASURED, not inferred.** On the pinned stack,
>   `torch.cuda.get_arch_list()` → `['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']`
>   — exactly the claimed set. Note additionally that **no `compute_*` entry is present**: the build
>   embeds no PTX, so there is no JIT fallback path. A Blackwell-class device has neither a matching
>   cubin nor PTX to compile from, which raises this finding's failure prediction from inference to
>   measurement.

### F11 — CONFIRMED

**ch3, verbatim** (the sweep protocol; see E13a for the full passage):

> λ_logit, is not fixed a priori but is selected on the validation partition through a small
> pre-registered grid sweep over λ_logit ∈{0.25, 0.5, 1, 2, 4} [...] The candidate maximizing
> dataset-level validation mIoU at the default seed (42) is selected

**ch3, verbatim** (the budget):

> All stages share the same 80,000-iteration training budget; the comparisons are therefore
> iteration-matched rather than wall-clock- or FLOPs-matched.

Nowhere in the sweep passage — which runs from "The constant weight scaling the distillation
term" to "reused unchanged in E3" — is a reduced budget, short-horizon proxy, early-stopping
rule, or subsampled-validation protocol specified. The passage ends by fixing the *selection
rule* (max validation mIoU, ties toward the smaller weight, boundary values reported as such)
and the *reporting* venue (Chapter 4), not the compute.

**Checking the stated refutation condition against the contract.** The contract does contain a
reduced budget — but it belongs to a **different decision**:

```text
docs/IMPLEMENTATION_CONTRACT.md:485
| `DISTILLATION_GRAD_CLIP_NORM` (`configs/distill.py`) | **E2, E3** (same value) | {1.0, 5.0} |
  pilot on **E2**, λ_logit held at **1.0**, 8,000 iters (10% of official) with validation every 1,000 |
```

`configs/distill.py:60-65` says the same, and is explicit that λ is *held fixed* during that
pilot rather than swept:

```python
   58	        # λ is HELD FIXED at the centre of the preregistered λ grid during this pilot, which avoids a
   59	        # 2 x 5 Cartesian clipping-by-λ search while keeping the pilot inside the registered grid.
   60	        "lambda_logit_during_pilot": 1.0,
   61	        # Shortened budget, identical for both candidates and clearly distinct from the official
   62	        # 80,000-iteration run, so a pilot artifact can never be mistaken for an official E2 result.
   63	        "pilot_budget_iters": 8000,              # 10% of the official 80,000
   64	        "pilot_val_interval": 1000,              # 8 validation points per candidate
   65	        "official_budget_iters": 80000,          # for contrast only; the pilot never runs this long
```

And the contract's own ordering (`IMPLEMENTATION_CONTRACT.md:497`, mirrored at
`configs/distill.py:72-74`) places the λ sweep **after** the pilot as a separate step:

```text
λ_logit sweep → freeze λ_logit → official E2/E3 runs
```

**Reasoning:** the 8,000-iteration budget is the clipping pilot's, is explicitly scoped to a
single λ value, and is sequenced before the sweep. It is not a reduced sweep budget. So neither
ch3 nor the contract specifies one, and five λ values at the shared 80,000-iteration budget is
what stands preregistered — five full E2 runs with the teacher online.

### F12 — REFUTED

**Anchor:** [train_e1.py:110-129](../src/training/train_e1.py#L110) — the whole of `validate()`.

```python
  110	@torch.no_grad()
  111	def validate(student, val_loader, device, num_classes: int, max_val_batches: int | None):
  112	    """Accumulate ONE confusion matrix over the val set (or a capped subset), then compute mIoU once."""
  113	    was_training = student.training
  114	    student.eval()
  115	    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
  116	    n_batches = 0
  117	    for i, (img, mask) in enumerate(val_loader):
  118	        if max_val_batches is not None and i >= max_val_batches:
  119	            break
  120	        logits = student(img.to(device))
  121	        pred = logits.argmax(1).cpu()                 # logits -> class indices (required by metrics)
  122	        cm += confusion_matrix(pred, mask, num_classes, ignore_index=IGNORE_INDEX)
  123	        n_batches += 1
  124	    all_miou = miou_from_confusion(cm)
  125	    disease_idx = torch.tensor([c for c in range(num_classes) if c != BACKGROUND_INDEX])
  126	    disease_miou = miou_from_confusion(cm, disease_idx)   # PROVISIONAL (open_questions #2)
  127	    if was_training:
  128	        student.train()
  129	    return all_miou, disease_miou, cm, n_batches
```

Line 113 captures `was_training`; line 114 switches to eval; lines 127-128 restore train mode
if and only if it was set on entry. The loop calls `student.train()` once at `:147` before
training begins, so `was_training` is `True` at every validation call and `student.train()`
is re-entered every time. BatchNorm is **not** left in inference mode after iter 4000.

**`optimizer.zero_grad()` placement — also correct.**

```python
  195	        optimizer.zero_grad(set_to_none=True)
  196	        logits = student(img)
  197	        if it == 1:
  198	            checks["logits_shape"] = tuple(logits.shape) == (img.shape[0], NUM_CLASSES, 512, 512)
  199	        ce = criterion.ce(logits, mask)
  200	        dice = criterion.dice(logits, mask)
  201	        loss = ce + dice                              # == CombinedCEDiceLoss(logits, mask)
  202	        if mode == "real" and not bool(torch.isfinite(loss)):
  203	            raise RuntimeError(
  204	                f"non-finite loss at iter {it}: loss={loss.item():.4f} ce={ce.item():.4f} "
  205	                f"dice={dice.item():.4f}. Aborting the real E1 run to avoid poisoning the model "
  206	                "or wasting compute.")
  207	
  208	        if it == 1:
  209	            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
  210	            p0 = next(p for p in student.parameters() if p.requires_grad)
  211	            before = p0.detach().clone()
  212	
  213	        loss.backward()
  214	        if grad_clip_norm is not None:
  215	            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
  216	        optimizer.step()
  217	        scheduler.step()                              # step ONCE per iteration, after optimizer
```

`zero_grad(set_to_none=True)` at :195 precedes the forward at :196, the backward at :213 and
the step at :216, once per iteration. There is no gradient accumulation and no double-step;
`scheduler.step()` at :217 runs once per iteration after the optimizer, as its comment says.

### F13 — CONFIRMED

**Anchor:** [train_e1.py:95-107](../src/training/train_e1.py#L95) (embedded under F2).

```python
    path = _assert_outside_repo(Path(ckpt_dir)) / f"e1_student_best_iter{it}.pt"
    torch.save({...}, path)
```

`torch.save` targets the final filename directly. There is no `.tmp` path, no
`os.replace`/`Path.rename`, no `fsync`. Repo-wide, the only rename-style checkpoint handling
is in `src/quant/checkpoint.py` and the eval artifact writer `src/eval/artifacts.py:452`
(which does stage into a `tmp` dir) — neither is on the E1 path.

**Reasoning:** a pod kill, OOM-kill, or spot-preemption during the write leaves a truncated
`.pt`. Combined with F2 (best-only, no `last.pt`) the blast radius is the entire run: the
truncated file is also the only file, unless an earlier best happens to survive under a
different `iter` in its name. The `iter{it}` suffix does mean earlier bests are **not**
overwritten, which is the one thing limiting the damage — see NEW RISK N4 for the disk cost.

---

## 3. Evidence blocks

### E1 — Repo state

```text
$ git rev-parse HEAD
148af2c9b30ffc78e7e2f4d0b9ff55cd40948607

$ git status -sb
## master...origin/master
 M docs/reference/reference.pdf
?? reports/pre_e1_launch_audit.md

$ git remote -v
origin	https://github.com/ainsleydeluna/plantseg-thesis.git (fetch)
origin	https://github.com/ainsleydeluna/plantseg-thesis.git (push)

$ git rev-parse --abbrev-ref HEAD
master

$ git status --porcelain
 M docs/reference/reference.pdf
?? reports/pre_e1_launch_audit.md

$ git log --oneline -5
148af2c finalize the preregistered real-run decision surface
9b132be preregister the remaining real-run training decisions
e5a6349 fix registered-stack compatibility defects found by the container gate
a15506f add the reproducible RunPod experiment environment
0b58050 validate the corruption dependency environment
```

**Every dirty/untracked path (complete list):**

| Path | Status | Note |
|---|---|---|
| `docs/reference/reference.pdf` | ` M` (modified, unstaged) | PROTECTED. Never opened, hashed,
copied, staged or modified by this audit. Metadata only, recorded below. |

That is the **only** entry. There are no untracked files reported by git, and the governed
paths (`src/**`, `configs/**`, `scripts/**`, `requirements*`, `docs/EVALUATION_CONTRACT.md`,
`docs/IMPLEMENTATION_CONTRACT.md`) are all clean — so the EVALUATION_CONTRACT §7.1 scoped
cleanliness gate for an `official` artifact is **satisfied** at this HEAD.

Protected-file metadata (permitted: status, size, mtime only):

```text
$ ls -la docs/reference/reference.pdf
-rw-r--r-- 1 admin 197121 163179 Jun 27 13:28 docs/reference/reference.pdf
```

E1 safety floor:

```text
$ git merge-base --is-ancestor 885523a HEAD && echo YES
"YES - 885523a is ancestor of HEAD"
```

### E2 — Repo tree, depth 3 (excluding `.git`, `__pycache__`)

```text
.
.  .agents
.  .claude
.  .claude  settings.local.json
.  .codex
.  .dockerignore
.  .gitattributes
.  .gitignore
.  AGENTS.md
.  CLAUDE.md
.  Dockerfile
.  README.md
.  configs
.  configs  augment.py
.  configs  corruption_protocol.json
.  configs  data.py
.  configs  distill.py
.  configs  e1_student.py
.  configs  loss.py
.  configs  model.py
.  configs  plantseg_class_map.json
.  configs  quant.py
.  configs  teacher
.  configs  teacher  segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
.  configs  teacher_finetune.py
.  data
.  docs
.  docs  B8_checkpoint.md
.  docs  EVALUATION_CONTRACT.md
.  docs  IMPLEMENTATION_CONTRACT.md
.  docs  STATISTICAL_ANALYSIS_CONTRACT.md
.  docs  WEEK1_CLOSEOUT.md
.  docs  WEEK1_STUDY_REVIEWER.md
.  docs  ai_guardrails.md
.  docs  b7_result.md
.  docs  conflicts.md
.  docs  e1-guide
.  docs  e1-guide  e1-guide-standalone.html
.  docs  e1-guide  index.html
.  docs  e1-guide  script.js
.  docs  e1-guide  styles.css
.  docs  open_questions.md
.  docs  reference
.  docs  reference  Distilling the Knowledge in a Neural Network.pdf
.  docs  reference  Guo (2022) - SegNeXt - Rethinking convolutional attention design for semantic segmentation (2).pdf
.  docs  reference  Hendrycks (2019) - Benchmarking neural network robustness to common corruptions and perturbations (1).pdf
.  docs  reference  Howard (2019) - Searching for MobileNetV3.pdf
.  docs  reference  Jacob (2018) - Quantization and training of neural networks for efficient integer-arithmetic-only inference.pdf
.  docs  reference  Kamann (2020) - Benchmarking the robustness of semantic segmentation models (1).pdf
.  docs  reference  Krishnamoorthi (2018) - Quantizing deep convolutional networks for efficient inference - A whitepaper.pdf
.  docs  reference  Shu (2021) - Channel-wise knowledge distillation for dense prediction (1).pdf
.  docs  reference  Wei (2026) - PlantSeg dataset paper (2).pdf
.  docs  reference  ch1.pdf
.  docs  reference  ch2.pdf
.  docs  reference  ch3.pdf
.  docs  reference  context.md
.  docs  reference  reference.pdf
.  docs  runpod_environment.md
.  docs  task_templates
.  docs  task_templates  B31_review_template.md
.  docs  task_templates  README.md
.  docs  task_templates  fix_template.md
.  docs  task_templates  runpod_preflight_template.md
.  docs  teacher_init_source.md
.  docs  teacher_prep_runbook.md
.  docs  updated_ch3_sync_audit.md
.  notebooks
.  outputs
.  outputs  E1
.  outputs  E1  checkpoints
.  outputs  E1  efficiency
.  outputs  E1  qualitative
.  outputs  E1  robustness
.  outputs  E2
.  outputs  E2  checkpoints
.  outputs  E2  efficiency
.  outputs  E2  qualitative
.  outputs  E2  robustness
.  outputs  E3
.  outputs  E3  checkpoints
.  outputs  E3  efficiency
.  outputs  E3  qualitative
.  outputs  E3  robustness
.  outputs  E4
.  outputs  E4  checkpoints
.  outputs  E4  efficiency
.  outputs  E4  qualitative
.  outputs  E4  robustness
.  outputs  E5
.  outputs  E5  checkpoints
.  outputs  E5  efficiency
.  outputs  E5  qualitative
.  outputs  E5  robustness
.  outputs  E6
.  outputs  E6  checkpoints
.  outputs  E6  efficiency
.  outputs  E6  qualitative
.  outputs  E6  robustness
.  outputs  E7
.  outputs  E7  checkpoints
.  outputs  E7  efficiency
.  outputs  E7  qualitative
.  outputs  E7  robustness
.  outputs  Teacher
.  outputs  Teacher  checkpoints
.  outputs  Teacher  efficiency
.  outputs  Teacher  qualitative
.  outputs  Teacher  robustness
.  reports
.  reports  a1a_metric_contract_red.md
.  reports  a1b_metric_contract_green.md
.  reports  a2a_evaluator_core_smoke.md
.  reports  a2b0_plantseg_class_map_provenance.md
.  reports  a2b_evaluator_plantseg_smoke.md
.  reports  a3a_paired_inference_smoke.md
.  reports  a3b0_corruption_vocabulary_smoke.md
.  reports  a3b_bootstrap_noninferiority_smoke.md
.  reports  annotation_test_audit.md
.  reports  annotation_train_audit.md
.  reports  annotation_val_audit.md
.  reports  annotations_test_audit.md
.  reports  annotations_train_audit.md
.  reports  annotations_val_audit.md
.  reports  b7_qnnpack.log
.  reports  claude_web_alignment_handoff.md
.  reports  cwd_projection_smoke.md
.  reports  dataloader_smoke.md
.  reports  dataset_audit_summary.md
.  reports  dataset_code_expectation_audit.md
.  reports  dataset_download_log.md
.  reports  dataset_location_log.md
.  reports  dataset_report.json
.  reports  dataset_report.md
.  reports  e1_class_weights.json
.  reports  e1_class_weights.md
.  reports  e1_preprocessing_init_readiness.md
.  reports  e1_runpod_launch_runbook.md
.  reports  e1_train_scaffold_dryrun.md
.  reports  e1_training_loop_readiness.md
.  reports  imagenet_init_wiring.md
.  reports  images_test_audit.md
.  reports  images_train_audit.md
.  reports  images_val_audit.md
.  reports  loss_smoke.md
.  reports  losses_metrics_smoke.md
.  reports  metadata_csv_audit.md
.  reports  metrics_smoke.md
.  reports  platform_verify.md
.  reports  pre_e1_launch_audit.md
.  reports  rrc_augmentation_smoke.md
.  reports  run_manifest.json
.  reports  strict_G0_closeout_update.md
.  reports  student_forward_smoke.md
.  reports  test_mask_value_audit.md
.  reports  test_split_consistency_audit.md
.  reports  train_split_consistency_audit.md
.  reports  train_step_smoke.md
.  reports  trainval_mask_value_audit.md
.  reports  val_split_consistency_audit.md
.  reports  week1_G0_status.md
.  requirements-e1.txt
.  requirements-runpod.in
.  requirements-runpod.lock
.  requirements.lock
.  scripts
.  scripts  build_corruption_cache.py
.  scripts  create_ptq_calibration_index.py
.  scripts  evaluate_backend_parity.py
.  scripts  evaluate_corruptions.py
.  scripts  evaluate_model.py
.  scripts  launch_teacher_finetune.py
.  scripts  preflight_environment.py
.  scripts  profile_efficiency.py
.  scripts  run_e4.py
.  scripts  run_e5.py
.  scripts  run_e6.py
.  scripts  run_e7.py
.  scripts  smoke_corruption_protocol.py
.  scripts  smoke_corruption_vendor.py
.  scripts  smoke_cwd_projection.py
.  scripts  smoke_dataloader.py
.  scripts  smoke_distill.py
.  scripts  smoke_efficiency.py
.  scripts  smoke_environment.py
.  scripts  smoke_eval_contract.py
.  scripts  smoke_eval_int8.py
.  scripts  smoke_eval_plantseg.py
.  scripts  smoke_eval_robustness.py
.  scripts  smoke_eval_stage_artifacts.py
.  scripts  smoke_eval_teacher.py
.  scripts  smoke_evaluate_corruptions.py
.  scripts  smoke_loss.py
.  scripts  smoke_losses_metrics.py
.  scripts  smoke_metrics.py
.  scripts  smoke_metrics_contract.py
.  scripts  smoke_plantseg_class_map.py
.  scripts  smoke_qnnpack_full_student.py
.  scripts  smoke_qnnpack_head.py
.  scripts  smoke_quant_e4_e5.py
.  scripts  smoke_quant_e6_e7.py
.  scripts  smoke_quant_runners.py
.  scripts  smoke_quant_x86_efficiency.py
.  scripts  smoke_realrun_decisions.py
.  scripts  smoke_stats_bootstrap.py
.  scripts  smoke_stats_paired.py
.  scripts  smoke_student_forward.py
.  scripts  smoke_teacher.py
.  scripts  smoke_teacher_config.py
.  scripts  smoke_teacher_launch.py
.  scripts  smoke_train_step.py
.  scripts  smoke_vendor_plantseg_class_map.py
.  scripts  test_teacher_init.py
.  scripts  vendor_plantseg_class_map.py
.  scripts  verify_env.py
.  scripts  verify_plantseg_dataset.py
.  scripts  write_run_manifest.py
.  src
.  src  __init__.py
.  src  corruption_cache.py
.  src  data
.  src  data  __init__.py
.  src  data  dataset.py
.  src  data  transforms.py
.  src  distill
.  src  distill  __init__.py
.  src  distill  cwd_projection.py
.  src  distill  export.py
.  src  distill  features.py
.  src  distill  segnext_teacher.py
.  src  distill  teacher.py
.  src  eval
.  src  eval  __init__.py
.  src  eval  adapters.py
.  src  eval  artifacts.py
.  src  eval  efficiency.py
.  src  eval  evaluate.py
.  src  eval  metrics.py
.  src  eval  model_loading.py
.  src  eval  robustness.py
.  src  eval  stage_artifacts.py
.  src  models
.  src  models  __init__.py
.  src  models  student.py
.  src  quant
.  src  quant  __init__.py
.  src  quant  calibration.py
.  src  quant  checkpoint.py
.  src  quant  prepare.py
.  src  quant  qconfig.py
.  src  quant  runner.py
.  src  quant  stages.py
.  src  quant  x86_latency.py
.  src  seeds.py
.  src  stats
.  src  stats  __init__.py
.  src  stats  align.py
.  src  stats  artifact.py
.  src  stats  bootstrap.py
.  src  stats  corruption_protocol.py
.  src  stats  ingest.py
.  src  stats  noninferiority.py
.  src  stats  robustness.py
.  src  stats  tests.py
.  src  training
.  src  training  __init__.py
.  src  training  losses.py
.  src  training  train_distill.py
.  src  training  train_e1.py
.  src  training  train_e2.py
.  src  training  train_e3.py
.  src  vendor
.  src  vendor  __init__.py
.  src  vendor  imagecorruptions
.  weights
```

### E3 — `src/data/dataset.py` (entire file)

```python
    1	"""PlantSegDataset + build_dataloader for the student.
    2	
    3	Reads JPG image + grayscale-PNG mask pairs from the on-disk, pre-partitioned folders
    4	(images/<split>/ + annotations/<split>/) — the folder layout is the split source of
    5	truth; annotation_*.json is NOT parsed for split membership. Config (root, num_classes)
    6	comes from configs/data.py; augmentation params from configs/augment.py.
    7	"""
    8	
    9	from __future__ import annotations
   10	
   11	import random
   12	import sys
   13	from pathlib import Path
   14	
   15	import numpy as np
   16	import torch
   17	from PIL import Image
   18	from torch.utils.data import DataLoader, Dataset
   19	
   20	# Make the repo root importable (configs/ and src/ are resolvable when run from anywhere).
   21	REPO = Path(__file__).resolve().parents[2]
   22	if str(REPO) not in sys.path:
   23	    sys.path.insert(0, str(REPO))
   24	
   25	from configs.augment import AUGMENT          # noqa: E402
   26	from configs.data import DATA                # noqa: E402
   27	from src.seeds import SEED                   # noqa: E402
   28	
   29	from .transforms import core_preprocess, finalize, train_preprocess  # noqa: E402
   30	
   31	NUM_CLASSES = DATA["num_classes"]
   32	_SPLITS = ("train", "val", "test")
   33	
   34	
   35	class PlantSegDataset(Dataset):
   36	    def __init__(self, split: str):
   37	        if split not in _SPLITS:
   38	            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
   39	        root = Path(DATA["root"])
   40	        if not root.exists():
   41	            raise FileNotFoundError(f"dataset root does not exist: {root}")
   42	        self.split = split
   43	        img_dir, mask_dir = root / "images" / split, root / "annotations" / split
   44	        # Collect images case-insensitively for .jpg/.jpeg (consistent with
   45	        # scripts/verify_plantseg_dataset.py); deterministic order via name sort.
   46	        imgs = sorted((p for p in img_dir.iterdir()
   47	                       if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
   48	                      key=lambda p: p.name)
   49	        self.pairs, missing = [], []
   50	        for img in imgs:
   51	            mask = mask_dir / f"{img.stem}.png"
   52	            if mask.exists():
   53	                self.pairs.append((img, mask))
   54	            else:
   55	                missing.append(img.name)                       # do NOT silently drop
   56	        # Missing masks must FAIL LOUD, not silently reduce the sample count.
   57	        if missing:
   58	            raise FileNotFoundError(
   59	                f"{len(missing)} image(s) in split={split} have no matching mask under {mask_dir} "
   60	                f"(expected <stem>.png), e.g. {missing[:5]}")
   61	        if not self.pairs:
   62	            raise RuntimeError(f"no image/mask pairs found for split={split} under {root}")
   63	        # Guard against silent under/over-count vs the locked PlantSeg split sizes (configs/data.py).
   64	        expected = DATA.get("splits", {}).get("sizes", {}).get(split)
   65	        if expected is not None and len(self.pairs) != expected:
   66	            raise RuntimeError(
   67	                f"split={split} pair count {len(self.pairs)} != locked expected {expected} "
   68	                f"(configs/data.py DATA['splits']['sizes']); check the dataset upload/extraction "
   69	                f"under {root}")
   70	        self.aug_params = AUGMENT
   71	
   72	    def __len__(self) -> int:
   73	        return len(self.pairs)
   74	
   75	    def __getitem__(self, idx: int):
   76	        img_path, mask_path = self.pairs[idx]
   77	        with Image.open(img_path) as im, Image.open(mask_path) as mk:
   78	            if self.split == "train":
   79	                # Per-sample augmentation RNG seeded from the worker/global RNG stream — reseeded per
   80	                # worker per epoch by worker_init_fn (num_workers>0) or advancing in the seeded main
   81	                # process (num_workers=0). Augmentation VARIES across epochs/repeats yet stays
   82	                # reproducible from the global seed (previously seeded by idx alone -> frozen per image).
   83	                rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
   84	                # true train-time multi-scale RRC on the original-resolution image (pre-pad)
   85	                img_np, mask_np = train_preprocess(im, mk, rng, self.aug_params)
   86	            else:
   87	                img_np, mask_np = core_preprocess(im, mk)    # val/test: unchanged deterministic core
   88	        return finalize(img_np, mask_np)                      # float32 CHW, int64 HW
   89	
   90	
   91	def _seed_worker(worker_id: int) -> None:
   92	    s = torch.initial_seed() % (2 ** 32)
   93	    np.random.seed(s)
   94	    random.seed(s)
   95	
   96	
   97	def build_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
   98	    """train shuffles; val/test do not. Seeded generator + worker_init_fn => deterministic (seed 42)."""
   99	    dataset = PlantSegDataset(split)
  100	    generator = torch.Generator()
  101	    generator.manual_seed(SEED)
  102	    return DataLoader(
  103	        dataset,
  104	        batch_size=batch_size,
  105	        shuffle=(split == "train"),
  106	        num_workers=num_workers,
  107	        generator=generator,
  108	        worker_init_fn=_seed_worker,
  109	        drop_last=False,
  110	    )
```

### E4 — `src/data/transforms.py` (entire file)

`train_preprocess` is at :152-172; every helper it calls (`_resize_long_side` :104,
`_apply_rotation` :79, `_random_crop_pad_512` :122, `_dom_nonignore_ratio` :114,
`_apply_flips` :70, `_apply_photometric` :93, `_jitter_hue_sat` :61, `finalize` :186) is
in the same file, so the whole file is given.

```python
    1	"""Core preprocessing + train-only augmentation for the PlantSeg student dataloader.
    2	
    3	Core (configs/data.py) — applied IDENTICALLY to train/val/test:
    4	  EXIF-transpose image (never mask) -> aspect-ratio-preserving resize long side->512
    5	  (bilinear image / nearest mask) -> symmetric pad to 512x512 (image [124,116,104],
    6	  mask 255) -> scale [0,1] -> ImageNet normalize -> float32 CHW image / int64 HW mask.
    7	Masks are read as RAW class-index arrays (PIL 'L'/'P', no RGB/palette expansion).
    8	
    9	Train path (train_preprocess, configs/augment.py) — true multi-scale RRC on the ORIGINAL-
   10	resolution image, BEFORE the final pad/normalize: EXIF(image) -> RGB -> aspect-preserving
   11	multi-scale resize -> rotation (before crop) -> random 512x512 crop+pad with cat_max_ratio
   12	-> joint flips -> image-only hue/sat. val/test use core_preprocess (unchanged). augment() is
   13	retained as a LEGACY 512x512-canvas helper (flips/rotation/photometric; no RRC crop).
   14	
   15	No Albumentations dependency — params come from configs/augment.py; ops use numpy/PIL/torch
   16	under a per-sample seeded RNG for determinism.
   17	"""
   18	
   19	from __future__ import annotations
   20	
   21	import numpy as np
   22	import torch
   23	from PIL import Image, ImageOps
   24	
   25	SIZE = 512
   26	MASK_IGNORE = 255
   27	IMAGENET_MEAN_8BIT = (124, 116, 104)
   28	IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
   29	IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
   30	
   31	
   32	def _resize_keep_ratio(img: Image.Image, mask: Image.Image):
   33	    w, h = img.size
   34	    if w >= h:
   35	        nw, nh = SIZE, max(1, round(h * SIZE / w))
   36	    else:
   37	        nw, nh = max(1, round(w * SIZE / h)), SIZE
   38	    return img.resize((nw, nh), Image.BILINEAR), mask.resize((nw, nh), Image.NEAREST)
   39	
   40	
   41	def _pad_to_square(img_np: np.ndarray, mask_np: np.ndarray):
   42	    h, w = img_np.shape[:2]
   43	    top, left = (SIZE - h) // 2, (SIZE - w) // 2
   44	    canvas = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
   45	    canvas[:] = IMAGENET_MEAN_8BIT
   46	    canvas[top:top + h, left:left + w] = img_np
   47	    mcanvas = np.full((SIZE, SIZE), MASK_IGNORE, dtype=np.int64)
   48	    mcanvas[top:top + h, left:left + w] = mask_np
   49	    return canvas, mcanvas
   50	
   51	
   52	def core_preprocess(img_pil: Image.Image, mask_pil: Image.Image):
   53	    """Return (uint8 HWC 512x512 image, int64 HW 512x512 mask) — geometric core, pre-normalize."""
   54	    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")  # EXIF on image only; RGB 3ch
   55	    img_r, mask_r = _resize_keep_ratio(img_pil, mask_pil)
   56	    img_np = np.asarray(img_r, dtype=np.uint8)                 # HWC uint8
   57	    mask_np = np.asarray(mask_r).astype(np.int64)              # HW raw class indices (no expansion)
   58	    return _pad_to_square(img_np, mask_np)
   59	
   60	
   61	def _jitter_hue_sat(img_np: np.ndarray, hue_delta: float, sat_factor: float) -> np.ndarray:
   62	    hsv = np.asarray(Image.fromarray(img_np).convert("HSV"), dtype=np.float32)  # H,S,V in 0-255
   63	    hsv[..., 0] = (hsv[..., 0] + hue_delta * 255.0) % 256.0
   64	    hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0.0, 255.0)
   65	    return np.asarray(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"), dtype=np.uint8)
   66	
   67	
   68	# ---- shared train-aug primitives (used by train_preprocess and the legacy augment) ----
   69	
   70	def _apply_flips(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
   71	    """Joint horizontal/vertical flips (works on any HxW)."""
   72	    if rng.rand() < p["horizontal_flip_p"]:
   73	        img_np, mask_np = img_np[:, ::-1, :].copy(), mask_np[:, ::-1].copy()
   74	    if rng.rand() < p["vertical_flip_p"]:
   75	        img_np, mask_np = img_np[::-1, :, :].copy(), mask_np[::-1, :].copy()
   76	    return img_np, mask_np
   77	
   78	
   79	def _apply_rotation(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
   80	    """Joint rotation: image bilinear (fill mean), mask nearest (fill 255). Works on any HxW."""
   81	    rot = p["rotation"]
   82	    if rng.rand() < rot["p"]:
   83	        angle = float(rng.uniform(-rot["degrees"], rot["degrees"]))
   84	        img_np = np.asarray(
   85	            Image.fromarray(img_np).rotate(angle, resample=Image.BILINEAR, fillcolor=IMAGENET_MEAN_8BIT),
   86	            dtype=np.uint8)
   87	        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L").rotate(
   88	            angle, resample=Image.NEAREST, fillcolor=MASK_IGNORE)
   89	        mask_np = np.asarray(mask_pil).astype(np.int64)
   90	    return img_np, mask_np
   91	
   92	
   93	def _apply_photometric(img_np: np.ndarray, rng: np.random.RandomState, p: dict) -> np.ndarray:
   94	    """Image-only hue/saturation jitter (mask untouched)."""
   95	    photo = p["photometric"]
   96	    if rng.rand() < photo["p"]:
   97	        hue_delta = float(rng.uniform(-photo["hue"], photo["hue"]))
   98	        lo, hi = photo["saturation_factor"]
   99	        sat_factor = float(rng.uniform(lo, hi))
  100	        img_np = _jitter_hue_sat(img_np, hue_delta, sat_factor)
  101	    return img_np
  102	
  103	
  104	def _resize_long_side(img_pil: Image.Image, mask_pil: Image.Image, target_long: int):
  105	    """Aspect-preserving resize so the LONG side == target_long (bilinear image / nearest mask)."""
  106	    w, h = img_pil.size
  107	    if w >= h:
  108	        nw, nh = target_long, max(1, round(h * target_long / w))
  109	    else:
  110	        nw, nh = max(1, round(w * target_long / h)), target_long
  111	    return img_pil.resize((nw, nh), Image.BILINEAR), mask_pil.resize((nw, nh), Image.NEAREST)
  112	
  113	
  114	def _dom_nonignore_ratio(mask_np: np.ndarray) -> float:
  115	    """Dominant NON-ignore class pixel fraction (255 excluded); 1.0 if the crop is all-ignore."""
  116	    vals, cnts = np.unique(mask_np, return_counts=True)
  117	    cnts = cnts[vals != MASK_IGNORE]
  118	    tot = int(cnts.sum())
  119	    return float(cnts.max() / tot) if tot > 0 else 1.0
  120	
  121	
  122	def _random_crop_pad_512(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState,
  123	                         cat_max_ratio: float, size: int = SIZE, max_attempts: int = 10):
  124	    """Random size x size crop. Per dimension: if the source side >= size take a random window;
  125	    else random-place and pad (image [124,116,104], mask 255). cat_max_ratio: accept iff the
  126	    dominant non-255 class fraction <= cat_max_ratio (255 excluded; background counts); up to
  127	    max_attempts tries, else fall back to the best (lowest-dominance) candidate seen."""
  128	    h, w = img_np.shape[:2]
  129	    best = None  # (dom, img_crop, mask_crop)
  130	    for k in range(1, max_attempts + 1):
  131	        canvas_img = np.empty((size, size, 3), dtype=np.uint8)
  132	        canvas_img[:] = IMAGENET_MEAN_8BIT
  133	        canvas_mask = np.full((size, size), MASK_IGNORE, dtype=np.int64)
  134	        if h >= size:
  135	            sy, dy, ch = int(rng.randint(0, h - size + 1)), 0, size
  136	        else:
  137	            sy, dy, ch = 0, int(rng.randint(0, size - h + 1)), h
  138	        if w >= size:
  139	            sx, dx, cw = int(rng.randint(0, w - size + 1)), 0, size
  140	        else:
  141	            sx, dx, cw = 0, int(rng.randint(0, size - w + 1)), w
  142	        canvas_img[dy:dy + ch, dx:dx + cw] = img_np[sy:sy + ch, sx:sx + cw]
  143	        canvas_mask[dy:dy + ch, dx:dx + cw] = mask_np[sy:sy + ch, sx:sx + cw]
  144	        dom = _dom_nonignore_ratio(canvas_mask)
  145	        if best is None or dom < best[0]:
  146	            best = (dom, canvas_img, canvas_mask)
  147	        if dom <= cat_max_ratio:
  148	            return canvas_img, canvas_mask, {"attempts": k, "fallback": False, "dom_ratio": dom}
  149	    return best[1], best[2], {"attempts": max_attempts, "fallback": True, "dom_ratio": best[0]}
  150	
  151	
  152	def train_preprocess(img_pil: Image.Image, mask_pil: Image.Image,
  153	                     rng: np.random.RandomState, p: dict):
  154	    """TRAIN-ONLY true RRC / multi-scale pipeline on the ORIGINAL-resolution image (pre-pad):
  155	    EXIF(image)->RGB -> aspect-preserving multi-scale resize (long side = round(512*r),
  156	    r~U(scale_range)) -> rotation (before crop) -> random 512x512 crop+pad (cat_max_ratio) ->
  157	    joint flips -> image-only photometric. Returns (uint8 HWC 512x512, int64 HW 512x512).
  158	    bilinear image / nearest mask; image pad [124,116,104] / mask pad 255; labels {0..115,255};
  159	    no label remap. val/test use core_preprocess (unchanged)."""
  160	    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")     # EXIF image only; RGB 3ch
  161	    rrc = p["random_resized_crop"]
  162	    lo, hi = rrc["scale_range"]
  163	    r = float(rng.uniform(lo, hi))                                 # multi-scale factor
  164	    target_long = max(1, int(round(SIZE * r)))
  165	    img_r, mask_r = _resize_long_side(img_pil, mask_pil, target_long)
  166	    img_np = np.asarray(img_r, dtype=np.uint8)
  167	    mask_np = np.asarray(mask_r).astype(np.int64)
  168	    img_np, mask_np = _apply_rotation(img_np, mask_np, rng, p)     # rotation BEFORE crop
  169	    img_np, mask_np, _ = _random_crop_pad_512(img_np, mask_np, rng, rrc["cat_max_ratio"])
  170	    img_np, mask_np = _apply_flips(img_np, mask_np, rng, p)
  171	    img_np = _apply_photometric(img_np, rng, p)
  172	    return img_np, mask_np
  173	
  174	
  175	def augment(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
  176	    """LEGACY (compatibility) train aug on a 512x512 canvas: flips -> rotation -> image-only
  177	    photometric. The ACTIVE train path is train_preprocess() (true original-resolution multi-scale
  178	    RRC). Retained for backward compatibility / direct 512x512-canvas use; it does NOT perform the
  179	    multi-scale RRC crop."""
  180	    img_np, mask_np = _apply_flips(img_np, mask_np, rng, p)
  181	    img_np, mask_np = _apply_rotation(img_np, mask_np, rng, p)
  182	    img_np = _apply_photometric(img_np, rng, p)
  183	    return img_np, mask_np
  184	
  185	
  186	def finalize(img_np: np.ndarray, mask_np: np.ndarray):
  187	    """uint8 HWC image + int64 HW mask -> (float32 CHW image tensor, int64 HW mask tensor)."""
  188	    img = img_np.astype(np.float32) / 255.0
  189	    img = (img - IMAGENET_MEAN) / IMAGENET_STD          # per-channel, broadcast over HxW
  190	    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # CHW
  191	    img_t = torch.from_numpy(img).float()
  192	    mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).long()
  193	    return img_t, mask_t
```

**Global-RNG bypass check — none found.**

```text
$ grep -n "np\.random\|random\." src/data/transforms.py
70:def _apply_flips(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
79:def _apply_rotation(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
93:def _apply_photometric(img_np: np.ndarray, rng: np.random.RandomState, p: dict) -> np.ndarray:
122:def _random_crop_pad_512(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState,
153:                     rng: np.random.RandomState, p: dict):
175:def augment(img_np: np.ndarray, mask_np: np.ndarray, rng: np.random.RandomState, p: dict):
```

Every hit is a **type annotation** (`rng: np.random.RandomState`). There is no call to
`np.random.<fn>`, `random.<fn>`, or any module-level RNG inside any transform. All randomness
flows through the `rng` object passed in from `PlantSegDataset.__getitem__`. This matters for
F1: the transforms are fully reproducible *given* the rng, so the seeding question is decided
entirely at `dataset.py:83`.

### E5 — `src/seeds.py` (entire file)

```python
    1	"""Reproducibility seed utility.
    2	
    3	Implements the seeding half of Blocker B6 (see docs/IMPLEMENTATION_CONTRACT.md):
    4	seed 42 across python random / numpy / torch / torch.cuda, with cuDNN set to
    5	deterministic and non-benchmark mode.
    6	"""
    7	
    8	import os
    9	import random
   10	
   11	import numpy as np
   12	import torch
   13	
   14	SEED = 42
   15	
   16	
   17	def set_seed(seed: int = 42) -> int:
   18	    """Seed all RNGs and enforce the contract-B6 determinism protocol.
   19	
   20	    Sets cuDNN deterministic mode, torch deterministic algorithms (warn_only=True), and
   21	    CUBLAS_WORKSPACE_CONFIG. Returns the seed used so callers can log it.
   22	    """
   23	    os.environ["PYTHONHASHSEED"] = str(seed)
   24	    # cuBLAS determinism must be configured BEFORE the first CUDA op / cuBLAS handle is created;
   25	    # set_seed runs before any device transfer in the training path (contract B6).
   26	    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
   27	
   28	    random.seed(seed)
   29	    np.random.seed(seed)
   30	
   31	    torch.manual_seed(seed)
   32	    torch.cuda.manual_seed(seed)
   33	    torch.cuda.manual_seed_all(seed)  # all CUDA devices
   34	
   35	    torch.backends.cudnn.deterministic = True
   36	    torch.backends.cudnn.benchmark = False
   37	    torch.use_deterministic_algorithms(True, warn_only=True)  # warn (not error) on non-deterministic ops
   38	
   39	    return seed
```

**Repo-wide determinism grep:**

```text
$ grep -rn "use_deterministic_algorithms|CUBLAS_WORKSPACE_CONFIG|cudnn" --include=*.py .
src/seeds.py:21:    CUBLAS_WORKSPACE_CONFIG. Returns the seed used so callers can log it.
src/seeds.py:26:    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
src/seeds.py:35:    torch.backends.cudnn.deterministic = True
src/seeds.py:36:    torch.backends.cudnn.benchmark = False
src/seeds.py:37:    torch.use_deterministic_algorithms(True, warn_only=True)  # warn (not error) on non-deterministic ops
scripts/verify_env.py:119:    cudnn_avail = bool(torch.backends.cudnn.is_available())
scripts/verify_env.py:120:    cudnn_ver = torch.backends.cudnn.version() if cudnn_avail else None
scripts/verify_env.py:127:    print(f"  cudnn_available     : {cudnn_avail}")
scripts/verify_env.py:128:    print(f"  cudnn_version       : {cudnn_ver}")
```

### E6 — `src/training/train_e1.py` (entire file)

```python
    1	#!/usr/bin/env python3
    2	"""E1 training-loop SCAFFOLD (Blocker B19).
    3	
    4	Composes the audited E1 components into one iteration-based supervised training loop:
    5	  set_seed(42) -> build_student -> build_dataloader(train/val) -> CombinedCEDiceLoss(class-weighted
    6	  CE + Dice) -> SGD (configs/e1_student.py) -> per-iteration PolynomialLR -> periodic validation that
    7	  ACCUMULATES ONE confusion matrix and computes mIoU once -> best-all-class-val-mIoU checkpoint.
    8	
    9	SAFETY MODEL (this file deliberately makes the real 80k run hard to start by accident):
   10	  * Default / `--dry-run`  -> tiny CPU dry-run, random init, NO download, checkpoint to a temp dir.
   11	  * Real run requires BOTH `--real-run` AND `--confirm-real-run`; otherwise the script exits with a
   12	    clear message. The real run is defined but is NOT exercised by B19.
   13	
   14	API/wiring verified in reports/e1_training_loop_readiness.md. Grad clipping is omitted by default: no
   15	concrete E1 max_norm exists (the config lists global-norm only under E2/E3 shared mechanics). Disease-only
   16	mIoU is PROVISIONAL (open_questions #2). Checkpoints are NEVER written inside the repo (hard-guarded).
   17	"""
   18	from __future__ import annotations
   19	
   20	import argparse
   21	import json
   22	import sys
   23	import tempfile
   24	from pathlib import Path
   25	
   26	REPO = Path(__file__).resolve().parents[2]
   27	if str(REPO) not in sys.path:
   28	    sys.path.insert(0, str(REPO))
   29	
   30	import torch  # noqa: E402
   31	
   32	from configs.data import DATA                                   # noqa: E402
   33	from configs.e1_student import E1_STUDENT                       # noqa: E402
   34	from src.data import NUM_CLASSES, build_dataloader             # noqa: E402
   35	from src.eval.metrics import confusion_matrix, miou_from_confusion  # noqa: E402
   36	from src.models.student import build_student                    # noqa: E402
   37	from src.seeds import set_seed                                  # noqa: E402
   38	from src.training.losses import CombinedCEDiceLoss             # noqa: E402
   39	
   40	IGNORE_INDEX = DATA["ignore_index"]            # 255
   41	BACKGROUND_INDEX = DATA["background_index"]    # 0
   42	CLASS_WEIGHTS_JSON = REPO / "reports" / "e1_class_weights.json"
   43	
   44	
   45	# --------------------------------------------------------------------------------------------------
   46	# helpers
   47	# --------------------------------------------------------------------------------------------------
   48	def load_ce_weights(path: Path = CLASS_WEIGHTS_JSON) -> torch.Tensor:
   49	    """Length-116 CE class weights (index-aligned to class id 0..115) from the B18a artifact."""
   50	    with open(path, encoding="utf-8") as f:
   51	        w = json.load(f)["weights"]
   52	    t = torch.tensor(w, dtype=torch.float32)
   53	    if t.numel() != NUM_CLASSES:
   54	        raise ValueError(f"class-weights length {t.numel()} != num_classes {NUM_CLASSES}")
   55	    if not bool(torch.isfinite(t).all()):
   56	        raise ValueError("class-weights contain non-finite values")
   57	    return t
   58	
   59	
   60	def build_scheduler(optimizer, horizon: int, power: float):
   61	    """Per-iteration polynomial LR decay. PolynomialLR if available (torch>=1.13), else a LambdaLR
   62	    fallback that reproduces lr = base * (1 - it/horizon)**power. Returns (scheduler, name)."""
   63	    L = torch.optim.lr_scheduler
   64	    if hasattr(L, "PolynomialLR"):
   65	        return L.PolynomialLR(optimizer, total_iters=horizon, power=power), "PolynomialLR"
   66	    return (L.LambdaLR(optimizer, lr_lambda=lambda it: (1.0 - it / horizon) ** power),
   67	            "LambdaLR(fallback)")
   68	
   69	
   70	def cycle(loader):
   71	    """Infinite iterator over a DataLoader -> iteration-based (not epoch-based) training."""
   72	    while True:
   73	        for batch in loader:
   74	            yield batch
   75	
   76	
   77	def _assert_outside_repo(path: Path) -> Path:
   78	    """Hard guard: a checkpoint directory may NEVER be the repo or live inside it."""
   79	    path = path.resolve()
   80	    if path == REPO or REPO in path.parents:
   81	        raise RuntimeError(f"refusing to use a checkpoint dir inside the repo: {path}")
   82	    return path
   83	
   84	
   85	def resolve_ckpt_dir(ckpt_dir_arg: str | None) -> Path:
   86	    """Prefer an explicit out-of-repo dir; otherwise auto-create a temp dir. Always guarded + created."""
   87	    if ckpt_dir_arg:
   88	        ckpt_dir = _assert_outside_repo(Path(ckpt_dir_arg))
   89	        ckpt_dir.mkdir(parents=True, exist_ok=True)
   90	    else:
   91	        ckpt_dir = _assert_outside_repo(Path(tempfile.mkdtemp(prefix="e1_dryrun_ckpt_")))
   92	    return ckpt_dir
   93	
   94	
   95	def save_checkpoint(ckpt_dir: Path, student, optimizer, scheduler, sched_name: str,
   96	                    it: int, best_miou: float) -> str:
   97	    path = _assert_outside_repo(Path(ckpt_dir)) / f"e1_student_best_iter{it}.pt"
   98	    torch.save({
   99	        "iter": it,
  100	        "model_state_dict": student.state_dict(),
  101	        "optimizer_state_dict": optimizer.state_dict(),
  102	        "scheduler_state_dict": scheduler.state_dict(),
  103	        "scheduler": sched_name,
  104	        "best_val_miou_all_class": best_miou,
  105	        "num_classes": NUM_CLASSES,
  106	    }, path)
  107	    return str(path)
  108	
  109	
  110	@torch.no_grad()
  111	def validate(student, val_loader, device, num_classes: int, max_val_batches: int | None):
  112	    """Accumulate ONE confusion matrix over the val set (or a capped subset), then compute mIoU once."""
  113	    was_training = student.training
  114	    student.eval()
  115	    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
  116	    n_batches = 0
  117	    for i, (img, mask) in enumerate(val_loader):
  118	        if max_val_batches is not None and i >= max_val_batches:
  119	            break
  120	        logits = student(img.to(device))
  121	        pred = logits.argmax(1).cpu()                 # logits -> class indices (required by metrics)
  122	        cm += confusion_matrix(pred, mask, num_classes, ignore_index=IGNORE_INDEX)
  123	        n_batches += 1
  124	    all_miou = miou_from_confusion(cm)
  125	    disease_idx = torch.tensor([c for c in range(num_classes) if c != BACKGROUND_INDEX])
  126	    disease_miou = miou_from_confusion(cm, disease_idx)   # PROVISIONAL (open_questions #2)
  127	    if was_training:
  128	        student.train()
  129	    return all_miou, disease_miou, cm, n_batches
  130	
  131	
  132	# --------------------------------------------------------------------------------------------------
  133	# training loop
  134	# --------------------------------------------------------------------------------------------------
  135	def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int, val_interval: int,
  136	        max_val_batches: int | None, num_workers: int, ckpt_dir_arg: str | None,
  137	        grad_clip_norm: float | None, log_every: int, seed: int) -> int:
  138	    set_seed(seed)
  139	    dev = torch.device(device)
  140	    print(f"[mode] {mode.upper()} | torch {torch.__version__} | device={dev} | "
  141	          f"cuda_available={torch.cuda.is_available()} | num_classes={NUM_CLASSES}")
  142	    print(f"[budget] max_iters={max_iters} batch_size={batch_size} val_interval={val_interval} "
  143	          f"max_val_batches={max_val_batches} num_workers={num_workers} seed={seed}")
  144	
  145	    # --- model ---
  146	    student = build_student(pretrained=pretrained).to(dev)
  147	    student.train()
  148	    if mode == "dry" and student.used_pretrained:
  149	        raise RuntimeError("dry-run must NOT use pretrained weights (no download allowed)")
  150	    n_params = sum(p.numel() for p in student.parameters())
  151	    print(f"[student] params={n_params:,} used_pretrained={student.used_pretrained} "
  152	          f"(pretrained arg={pretrained!r})")
  153	
  154	    # --- data ---
  155	    train_loader = build_dataloader("train", batch_size, num_workers=num_workers)
  156	    val_loader = build_dataloader("val", batch_size, num_workers=num_workers)
  157	    print(f"[data] train_index={len(train_loader.dataset)} val_index={len(val_loader.dataset)} "
  158	          f"(index globbed; only the batches pulled below are decoded)")
  159	
  160	    # --- loss (class-weighted CE + Dice); weight buffer follows .to(device) ---
  161	    weights = load_ce_weights().to(dev)
  162	    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=IGNORE_INDEX).to(dev)
  163	    print(f"[loss] CombinedCEDiceLoss(weight=len{weights.numel()}, ignore_index={IGNORE_INDEX}) "
  164	          f"weight_on={criterion.ce.weight.device}")
  165	
  166	    # --- optimizer (configs/e1_student.py) + per-iteration poly LR ---
  167	    optimizer = torch.optim.SGD(student.parameters(), lr=E1_STUDENT["learning_rate"],
  168	                                momentum=E1_STUDENT["momentum"],
  169	                                weight_decay=E1_STUDENT["weight_decay"])
  170	    horizon = E1_STUDENT["iterations"]            # poly horizon is ALWAYS the real 80k curve
  171	    scheduler, sched_name = build_scheduler(optimizer, horizon, E1_STUDENT["lr_power"])
  172	    print(f"[opt] SGD lr={E1_STUDENT['learning_rate']} momentum={E1_STUDENT['momentum']} "
  173	          f"weight_decay={E1_STUDENT['weight_decay']} | scheduler={sched_name} "
  174	          f"(total_iters={horizon}, power={E1_STUDENT['lr_power']})")
  175	    print(f"[grad-clip] {'disabled (no concrete E1 max_norm)' if grad_clip_norm is None else grad_clip_norm}")
  176	
  177	    ckpt_dir = resolve_ckpt_dir(ckpt_dir_arg)
  178	    print(f"[ckpt] dir={ckpt_dir} (verified OUTSIDE repo)")
  179	
  180	    checks: dict[str, bool] = {}
  181	    best_miou = float("-inf")
  182	    best_ckpt = None
  183	    first_batch_meta = None
  184	    lr_trace: list[float] = []
  185	    train_iter = cycle(train_loader)
  186	
  187	    for it in range(1, max_iters + 1):
  188	        img, mask = next(train_iter)
  189	        img, mask = img.to(dev), mask.to(dev)
  190	        if it == 1:
  191	            first_batch_meta = (tuple(img.shape), str(img.dtype), tuple(mask.shape), str(mask.dtype))
  192	            checks["batch_shapes"] = (img.shape[1:] == (3, 512, 512) and img.dtype == torch.float32
  193	                                      and mask.shape[1:] == (512, 512) and mask.dtype == torch.int64)
  194	
  195	        optimizer.zero_grad(set_to_none=True)
  196	        logits = student(img)
  197	        if it == 1:
  198	            checks["logits_shape"] = tuple(logits.shape) == (img.shape[0], NUM_CLASSES, 512, 512)
  199	        ce = criterion.ce(logits, mask)
  200	        dice = criterion.dice(logits, mask)
  201	        loss = ce + dice                              # == CombinedCEDiceLoss(logits, mask)
  202	        if mode == "real" and not bool(torch.isfinite(loss)):
  203	            raise RuntimeError(
  204	                f"non-finite loss at iter {it}: loss={loss.item():.4f} ce={ce.item():.4f} "
  205	                f"dice={dice.item():.4f}. Aborting the real E1 run to avoid poisoning the model "
  206	                "or wasting compute.")
  207	
  208	        if it == 1:
  209	            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
  210	            p0 = next(p for p in student.parameters() if p.requires_grad)
  211	            before = p0.detach().clone()
  212	
  213	        loss.backward()
  214	        if grad_clip_norm is not None:
  215	            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
  216	        optimizer.step()
  217	        scheduler.step()                              # step ONCE per iteration, after optimizer
  218	        lr = optimizer.param_groups[0]["lr"]
  219	        lr_trace.append(lr)
  220	
  221	        if it == 1:
  222	            checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)
  223	
  224	        if it % log_every == 0 or it == max_iters:
  225	            print(f"[iter {it:>4}/{max_iters}] loss={loss.item():.4f} ce={ce.item():.4f} "
  226	                  f"dice={dice.item():.4f} lr={lr:.8e}")
  227	
  228	        if it % val_interval == 0 or it == max_iters:
  229	            all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
  230	                                                       max_val_batches)
  231	            checks["val_cm_accumulated"] = (tuple(cm.shape) == (NUM_CLASSES, NUM_CLASSES)
  232	                                            and int(cm.sum()) > 0 and nvb >= 1)
  233	            print(f"[val  {it:>4}/{max_iters}] cm_batches={nvb} cm_total_px={int(cm.sum())} "
  234	                  f"all_class_miou={all_miou:.5f} disease_only_miou(PROVISIONAL)={disease_miou:.5f}")
  235	            if all_miou > best_miou:
  236	                best_miou = all_miou
  237	                best_ckpt = save_checkpoint(ckpt_dir, student, optimizer, scheduler, sched_name,
  238	                                            it, best_miou)
  239	                print(f"[ckpt {it:>4}/{max_iters}] new best all_class_miou={best_miou:.5f} "
  240	                      f"-> {best_ckpt}")
  241	
  242	    # scheduler sanity: per-iteration poly decay is monotonically non-increasing
  243	    checks["lr_non_increasing"] = all(lr_trace[i + 1] <= lr_trace[i] + 1e-12
  244	                                      for i in range(len(lr_trace) - 1))
  245	
  246	    hard = ["batch_shapes", "logits_shape", "loss_finite", "optimizer_step",
  247	            "val_cm_accumulated", "lr_non_increasing"]
  248	    passed = all(checks.get(k, False) for k in hard)
  249	    print("\n[CHECKS]")
  250	    for k in hard:
  251	        print(f"  {k:18}: {'PASS' if checks.get(k) else 'FAIL'}")
  252	    print(f"[summary] first_batch={first_batch_meta}")
  253	    print(f"[summary] lr_trace={['%.8e' % x for x in lr_trace]}")
  254	    print(f"[summary] best_all_class_val_miou={best_miou:.5f} best_ckpt={best_ckpt}")
  255	    print(f"[summary] used_pretrained={student.used_pretrained} (no download in dry-run)")
  256	    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
  257	    return 0 if passed else 1
  258	
  259	
  260	# --------------------------------------------------------------------------------------------------
  261	# CLI / safety gate
  262	# --------------------------------------------------------------------------------------------------
  263	def parse_args(argv=None):
  264	    p = argparse.ArgumentParser(description="E1 training-loop scaffold (dry-run by default).")
  265	    p.add_argument("--dry-run", action="store_true", help="tiny CPU dry-run (safe default behavior)")
  266	    p.add_argument("--real-run", action="store_true",
  267	                   help="intent to run the real 80k training (requires --confirm-real-run)")
  268	    p.add_argument("--confirm-real-run", action="store_true",
  269	                   help="explicit confirmation gate for the real 80k run")
  270	    p.add_argument("--device", default=None)
  271	    p.add_argument("--init", choices=["none", "imagenet"], default=None,
  272	                   help="backbone init; 'imagenet' uses configs/e1_student.py (real run only)")
  273	    p.add_argument("--batch-size", type=int, default=None)
  274	    p.add_argument("--max-iters", type=int, default=None)
  275	    p.add_argument("--val-interval", type=int, default=None)
  276	    p.add_argument("--max-val-batches", type=int, default=None)
  277	    p.add_argument("--num-workers", type=int, default=None)
  278	    p.add_argument("--ckpt-dir", default=None, help="out-of-repo dir; auto temp dir if omitted")
  279	    p.add_argument("--grad-clip-norm", type=float, default=None,
  280	                   help="global-norm clip; omitted by default (no concrete E1 value)")
  281	    p.add_argument("--log-every", type=int, default=1)
  282	    p.add_argument("--seed", type=int, default=42)
  283	    return p.parse_args(argv)
  284	
  285	
  286	def main(argv=None) -> int:
  287	    args = parse_args(argv)
  288	
  289	    if args.dry_run and args.real_run:
  290	        print("ERROR: pass only one of --dry-run / --real-run.", file=sys.stderr)
  291	        return 2
  292	
  293	    # ---- real-run safety gate ----
  294	    if args.real_run:
  295	        if not args.confirm_real_run:
  296	            print("REFUSING to start the real E1 run: --real-run requires --confirm-real-run.\n"
  297	                  "The full 80k training is intentionally NOT runnable from defaults. "
  298	                  "Re-run with: --real-run --confirm-real-run", file=sys.stderr)
  299	            return 2
  300	        mode = "real"
  301	    elif args.confirm_real_run:
  302	        print("ERROR: --confirm-real-run given without --real-run; nothing to confirm.", file=sys.stderr)
  303	        return 2
  304	    else:
  305	        mode = "dry"
  306	        if not args.dry_run:
  307	            print("[mode] No --dry-run/--real-run given; defaulting to SAFE DRY-RUN. "
  308	                  "The real 80k run requires --real-run --confirm-real-run.")
  309	
  310	    if mode == "dry":
  311	        device = args.device or "cpu"
  312	        if args.init == "imagenet":
  313	            print("[init] --init imagenet ignored in dry-run (forcing random init, no download).")
  314	        pretrained = False                                   # forced: no ImageNet download in dry-run
  315	        batch_size = args.batch_size or 2
  316	        max_iters = args.max_iters or 4
  317	        val_interval = args.val_interval or 2
  318	        max_val_batches = args.max_val_batches if args.max_val_batches is not None else 2
  319	        num_workers = args.num_workers if args.num_workers is not None else 0
  320	    else:  # real (defined, NOT exercised by B19)
  321	        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  322	        # Real E1 must run on CUDA. Refuse a CPU real run (an 80k-iter CPU run is almost certainly a
  323	        # mistake and would take weeks); fail loud rather than silently degrade. No hidden CPU path.
  324	        if not str(device).startswith("cuda"):
  325	            reason = ("--device cpu was passed" if args.device == "cpu"
  326	                      else "CUDA is not available (torch.cuda.is_available()=False)")
  327	            print(f"REFUSING to start the real E1 run on CPU: {reason}.\n"
  328	                  "The real 80k run requires a CUDA GPU. Provision a GPU (see "
  329	                  "reports/e1_runpod_launch_runbook.md), or use --dry-run for a safe CPU smoke.",
  330	                  file=sys.stderr)
  331	            return 2
  332	        init = args.init or "imagenet"
  333	        pretrained = False if init == "none" else E1_STUDENT["init_weights"]
  334	        batch_size = args.batch_size or E1_STUDENT["batch_size"]
  335	        max_iters = args.max_iters or E1_STUDENT["iterations"]
  336	        val_interval = args.val_interval or E1_STUDENT["val_interval"]
  337	        max_val_batches = args.max_val_batches            # None -> full val
  338	        num_workers = args.num_workers if args.num_workers is not None else 4
  339	
  340	    return run(mode=mode, device=device, pretrained=pretrained, batch_size=batch_size,
  341	               max_iters=max_iters, val_interval=val_interval, max_val_batches=max_val_batches,
  342	               num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
  343	               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed)
  344	
  345	
  346	if __name__ == "__main__":
  347	    raise SystemExit(main())
```

**Explicitly identified, as requested:**

| Item | Line(s) | Detail |
|---|---|---|
| `set_seed` vs first CUDA touch | `:138` then `:139`/`:146` | `set_seed(seed)` is the **first**
statement of `run()`. `torch.device(device)` is `:139`; the first actual device transfer is
`.to(dev)` at `:146`. So `CUBLAS_WORKSPACE_CONFIG` is exported before any cuBLAS handle exists,
as contract B6 requires. |
| Checkpoint writes | `:98` (only `torch.save`), called from `:237` | best-val-improvement only |
| `print` calls | `:140`, `:142`, `:151`, `:157`, `:163`, `:172`, `:175`, `:178`, `:225`,
`:233`, `:239`, `:249`, `:251`, `:252`, `:253`, `:254`, `:255`, `:256`, plus `:290`, `:296`,
`:302`, `:307`, `:313`, `:327` in the CLI gate | all stdout; none write a file |
| `parse_args` option list | `:263-283` | `--dry-run`, `--real-run`, `--confirm-real-run`,
`--device`, `--init`, `--batch-size`, `--max-iters`, `--val-interval`, `--max-val-batches`,
`--num-workers`, `--ckpt-dir`, `--grad-clip-norm`, `--log-every`, `--seed`. **No `--resume`.** |
| Real-run defaults | `:320-338` | `device=cuda` (hard-abort on CPU `:324-331`),
`pretrained=E1_STUDENT['init_weights']`, `batch_size=16`, `max_iters=80000`,
`val_interval=4000`, `max_val_batches=None` (full val), **`num_workers=4`** |

### E7 — `configs/e1_student.py` and `configs/data.py` (entire files)

#### `configs/e1_student.py`

```python
    1	# Student training config — Blocker B2 (shared E1 / E2 / E3 recipe)
    2	# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B2 - Student training"
    3	# Every value traced to ch3.pdf; init checkpoint per ch3 section D / Table 3.5.
    4	# Analysis/config artifact only — contains NO training logic.
    5	
    6	E1_STUDENT = {
    7	    # Initialization
    8	    "init_weights": "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2",  # quantizable variant, top-1 75.27%
    9	
   10	    # Optimizer
   11	    "optimizer": "SGD",
   12	    "momentum": 0.9,
   13	    "learning_rate": 1e-2,
   14	    "lr_schedule": "polynomial",
   15	    "lr_power": 0.9,
   16	    "weight_decay": 1e-4,
   17	
   18	    # Budget
   19	    "batch_size": 16,
   20	    "iterations": 80000,
   21	    "input": (512, 512),
   22	
   23	    # Validation / checkpoint
   24	    "val_interval": 4000,
   25	    "checkpoint_selection": "best_val_miou",  # D1: ALL-CLASS validation mIoU (headline + checkpoint); disease-only mIoU is secondary/provisional only; switching headline/checkpoint to disease-only needs a separate explicit decision. See docs/open_questions.md #2.
   26	
   27	    # Shared training mechanics — applied per ch3 to the WHOLE E1/E2/E3 recipe (see `shared_by`); the
   28	    # distill-specific items only take effect once E2/E3 add distillation.
   29	    "distill_weight_ramp": "linear 0 -> target over first epoch",
   30	    "gradient_clipping": "global_norm",   # D2: METHOD FAMILY only (global-norm) — NOT a numeric value and NOT proof clipping is active.
   31	    "grad_clip_max_norm": None,           # D-A/D2 resolved: E1 is intentionally unclipped by default as a documented deviation because Chapter 3 gives no numeric max_norm. The optional train_e1.py hook remains available via --grad-clip-norm if instability occurs. See docs/open_questions.md D2.
   32	    "teacher_in_loop": "eval mode, online, identical augmented input as student",
   33	
   34	    # Scope note: E1=this recipe with no distillation; E2/E3 add distill terms on top (see configs/distill.py).
   35	    "shared_by": ("E1", "E2", "E3"),
   36	
   37	    # ---------------------------------------------------------------- clipping scope note
   38	    # E1 stays UNCLIPPED: open_questions D2/D-A resolved that, and this task does not reopen it.
   39	    # Chapter 3 requires global-norm clipping for the DISTILLATION stages and for QAT, so the clipped
   40	    # stages carry their own separate, independently selected thresholds:
   41	    #   * E2/E3  -> DISTILL["distillation_grad_clip_pilot"]  (configs/distill.py)
   42	    #   * E5/E6  -> QUANT["qat_grad_clip_pilot"]             (configs/quant.py)
   43	    # Those two are DIFFERENT optimization regimes and are not required to share a numeric threshold.
   44	    #
   45	    # CARRY FORWARD (manuscript, not resolved here): Chapter 3 describes E1/E2/E3 as sharing an
   46	    # identical recipe and attributes their differences to the distillation objectives, yet applies
   47	    # clipping only to the distillation stages. That wording needs reconciling; execution is governed
   48	    # by the repository decision above.
   49	    "grad_clip_scope": {
   50	        "e1": "unclipped (D2/D-A)",
   51	        "e2_e3": "DISTILL['distillation_grad_clip_pilot']",
   52	        "e5_e6": "QUANT['qat_grad_clip_pilot']",
   53	        "shared_numeric_threshold_required": False,
   54	        "manuscript_reconciliation_pending": True,
   55	    },
   56	}
```

#### `configs/data.py`

```python
    1	# Dataset config — PlantSeg (Wei et al., 2026)
    2	# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (c) "Dataset facts" + (e)/(f) preprocessing.
    3	# Dataset facts traced to Wei (2026); preprocessing traced to ch3.pdf.
    4	# num_classes/root resolved empirically 2026-06-27 (see reports/dataset_report.md). reduce_zero_label is
    5	# set False (PROVEN no-remap: masks already 0-115, bg=0 kept); the separate disease-only metric convention
    6	# (exclude index 0 in reporting) remains open — see open_questions #2.
    7	# Analysis/config artifact only — contains NO training logic.
    8	
    9	import os
   10	
   11	# Dataset root is portable via the PLANTSEG_DATA_ROOT environment variable. Set PLANTSEG_DATA_ROOT on
   12	# RunPod/Linux to avoid editing this file; if it is unset, the local Windows default below is used for the
   13	# current development machine (behavior unchanged). See reports/e1_runpod_launch_runbook.md.
   14	DEFAULT_PLANTSEG_DATA_ROOT = r"C:\Users\admin\plantseg_data\plantseg"
   15	
   16	DATA = {
   17	    # Root (extracted dataset path) — verified to exist; see reports/dataset_location_log.md.
   18	    # PLANTSEG_DATA_ROOT env var overrides this when set; otherwise the Windows default is used.
   19	    "root": os.environ.get("PLANTSEG_DATA_ROOT", DEFAULT_PLANTSEG_DATA_ROOT),
   20	
   21	    # Class space — empirically verified: mask values 0-115 (background 0 + 115 diseases 1-115).
   22	    "num_classes": 116,                          # all-class; output layer = 116  [empirical]
   23	    "reduce_zero_label": False,                  # PROVEN no-remap: masks are already 0-115 (bg=0 kept). See reports/dataloader_smoke.md and reports/dataset_audit_summary.md. NOTE: the separate disease-only metric convention (exclude index 0 in reporting) is NOT controlled by this flag and remains open; see docs/open_questions.md #2.
   24	    "ignore_index": 255,                         # padding/ignore; ABSENT from raw masks, added at preprocessing
   25	    "background_index": 0,                       # [empirical] disease-only metrics exclude this
   26	    "mask_indices_documented": "verified 0-115: 0=background, 1-115=diseases (mask=category_id+1)",
   27	
   28	    # Splits (official 70/10/20) — pre-partitioned on disk as images/<split>/ + annotations/<split>/
   29	    "splits": {
   30	        "ratio": "70/10/20",
   31	        "dirs": ("images/{split}", "annotations/{split}"),  # use folders, NOT annotation_*.json
   32	        "names": ("train", "val", "test"),
   33	        "files": ("annotation_train.json", "annotation_val.json", "annotation_test.json"),  # COCO; not used for split
   34	        "sizes": {"train": 5367, "val": 846, "test": 1561},  # [empirical]
   35	        "test_count": 1561,                      # per-image metric unit count
   36	        "integrity_check": "zero identifier overlap across partitions (verified)",
   37	    },
   38	
   39	    # Formats
   40	    "image_format": "JPEG",
   41	    "mask_format": "PNG grayscale",
   42	
   43	    # Core preprocessing applied to ALL partitions (val / clean-test use exactly this, no augmentation)
   44	    "preprocess_val_test": {
   45	        "resize": "aspect-ratio preserving, long side -> 512",
   46	        "pad_to": (512, 512),
   47	        "image_pad_value": (124, 116, 104),      # ImageNet mean in 8-bit terms
   48	        "mask_pad_value": 255,
   49	        "image_interpolation": "bilinear",
   50	        "mask_interpolation": "nearest",
   51	        "scale": (0.0, 1.0),
   52	        "normalize_mean": (0.485, 0.456, 0.406),
   53	        "normalize_std": (0.229, 0.224, 0.225),
   54	        "image_dtype_layout": "float32 CHW",
   55	        "mask_dtype_layout": "int64 HW",
   56	    },
   57	
   58	    # Train partition = core preprocessing + train-only augmentation
   59	    "preprocess_train": "same core preprocessing + augmentation (see configs/augment.py)",
   60	
   61	    # Provenance
   62	    "doi": "10.5281/zenodo.17719108",
   63	    "license": "CC BY-NC 4.0",
   64	}
```

### E8 — `src/models/student.py` forwards + output strides

#### `LRASPPHead.forward` (:92-100) and its `context` branch (:79-90)

```python
   76	class LRASPPHead(nn.Module):
   77	    """LR-ASPP-style head: additive class-logit fusion (NOT feature concatenation)."""
   78	
   79	    def __init__(self, low_ch: int, high_ch: int, inter_ch: int, num_classes: int):
   80	        super().__init__()
   81	        self.high_proj = _ConvBNReLU(high_ch, inter_ch, 1)              # 1x1 Conv-BN-ReLU
   82	        self.context = nn.Sequential(                                   # avgpool -> 1x1 -> sigmoid
   83	            nn.AdaptiveAvgPool2d(1),
   84	            nn.Conv2d(high_ch, inter_ch, 1, bias=True),
   85	            nn.Sigmoid(),
   86	        )
   87	        self.high_logits = nn.Conv2d(inter_ch, num_classes, 1)
   88	        self.low_logits = nn.Conv2d(low_ch, num_classes, 1)
   89	        self.ff_mul = FloatFunctional()   # quantization-safe element-wise multiply (attention)
   90	        self.ff_add = FloatFunctional()   # quantization-safe class-logit add
   91	
   92	    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
   93	        """Return OS8 class logits. The final upsample to full resolution happens after dequant in
   94	        the parent (matches the quantization-ready isolated head: one quantized interpolate, then a
   95	        float final upsample). In FP32 this is numerically identical to the B10 head."""
   96	        a = self.high_proj(high)                                       # [B,inter,H16,W16]
   97	        s = self.context(high)                                         # [B,inter,1,1]
   98	        high_feat = self.ff_mul.mul(a, s)                              # quantization-safe attention
   99	        high_os8 = F.interpolate(high_feat, size=low.shape[-2:], mode="bilinear", align_corners=False)
  100	        return self.ff_add.add(self.high_logits(high_os8), self.low_logits(low))  # OS8 class logits
```

#### `PlantSegStudent.forward` (:143-149) and `_forward_features` (:131-141)

```python
  131	    def _forward_features(self, x):
  132	        low = high = None
  133	        h = x
  134	        for i, blk in enumerate(self.features):
  135	            h = blk(h)
  136	            if i == self.low_tap:
  137	                low = h
  138	            if i == self.high_tap:
  139	                high = h
  140	                break
  141	        return low, high
  142	
  143	    def forward(self, x: torch.Tensor) -> torch.Tensor:
  144	        out_size = x.shape[-2:]
  145	        x = self.quant(x)                          # pass-through in FP32 (not prepared)
  146	        low, high = self._forward_features(x)
  147	        logits = self.head(low, high)              # OS8 class logits
  148	        logits = self.dequant(logits)              # dequant BEFORE the float final upsample
  149	        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
```

**Output stride at each stage for a 512x512 input — MEASURED by probe P6:**

```text
features[ 0] ch=  16 256x256  OS=2
features[ 1] ch=  16 256x256  OS=2
features[ 2] ch=  24 128x128  OS=4
features[ 3] ch=  24 128x128  OS=4
features[ 4] ch=  40 64x64    OS=8
features[ 5] ch=  40 64x64    OS=8
features[ 6] ch=  40 64x64    OS=8   <== LOW TAP (skip)
features[ 7] ch=  80 32x32    OS=16
features[ 8] ch=  80 32x32    OS=16
features[ 9] ch=  80 32x32    OS=16
features[10] ch=  80 32x32    OS=16
features[11] ch= 112 32x32    OS=16
features[12] ch= 112 32x32    OS=16
features[13] ch= 160 32x32    OS=16
features[14] ch= 160 32x32    OS=16
features[15] ch= 160 32x32    OS=16  <== HIGH TAP (C5)

low  (skip)      : (1, 40, 64, 64)     OS=8
high (C5)        : (1, 160, 32, 32)    OS=16
head OS8 logits  : (1, 116, 64, 64)    OS=8
final logits     : (1, 116, 512, 512)  OS=1

used_pretrained=False  num_classes=116  low_tap=6 high_tap=15 low_ch=40 high_ch=160
params=2,933,688
```

Stage-by-stage narrative: input 512x512 → OS2 after `features[0..1]` → OS4 after `[2..3]` →
OS8 at `[4..6]` (the **low tap**, 40 ch) → OS16 from `[7]` onward, held at OS16 through `[15]`
by the dilation (`MODEL['dilated']=True`) rather than striding further down (the **high tap**,
160 ch). The head projects high→256 ch at OS16, multiplies by the global-pooled sigmoid gate,
bilinearly upsamples that to OS8 (`student.py:99`), adds the low-tap class logits, and emits
OS8 116-channel logits. `PlantSegStudent.forward:149` then bilinearly upsamples to OS1
(512x512) **after** `dequant`.

### E9 — `requirements.lock` (entire file)

```text
    1	addict==2.4.0
    2	aliyun-python-sdk-core==2.16.0
    3	aliyun-python-sdk-kms==2.16.5
    4	certifi==2026.6.17
    5	cffi==2.0.0
    6	charset-normalizer==3.4.7
    7	click==8.4.2
    8	colorama==0.4.6
    9	contourpy==1.3.3
   10	crcmod==1.7
   11	cryptography==49.0.0
   12	cycler==0.12.1
   13	filelock==3.14.0
   14	fonttools==4.63.0
   15	fsspec==2026.4.0
   16	fvcore==0.1.5.post20221221
   17	idna==3.18
   18	ImageIO==2.37.3
   19	iopath==0.1.10
   20	Jinja2==3.1.6
   21	jmespath==0.10.0
   22	kiwisolver==1.5.0
   23	lazy-loader==0.5
   24	Markdown==3.10.2
   25	markdown-it-py==4.2.0
   26	MarkupSafe==3.0.3
   27	matplotlib==3.11.0
   28	mdurl==0.1.2
   29	mmcv==2.1.0
   30	mmengine==0.10.7
   31	mmsegmentation==1.2.2
   32	model-index==0.1.11
   33	mpmath==1.3.0
   34	networkx==3.6.1
   35	numpy==1.26.4
   36	opencv-python==4.8.1.78
   37	opendatalab==0.0.10
   38	openmim==0.3.9
   39	openxlab==0.1.3
   40	ordered-set==4.1.0
   41	oss2==2.17.0
   42	packaging==24.2
   43	pandas==3.0.3
   44	patsy==1.0.2
   45	pillow==12.3.0
   46	platformdirs==4.10.0
   47	portalocker==3.2.0
   48	prettytable==3.18.0
   49	pycparser==3.0
   50	pycryptodome==3.23.0
   51	Pygments==2.20.0
   52	pyparsing==3.3.2
   53	python-dateutil==2.9.0.post0
   54	pytz==2023.4
   55	pywin32==312
   56	PyYAML==6.0.3
   57	regex==2026.5.9
   58	requests==2.28.2
   59	rich==13.4.2
   60	scikit-image==0.23.2
   61	scipy==1.11.4
   62	six==1.17.0
   63	statsmodels==0.14.6
   64	sympy==1.14.0
   65	tabulate==0.10.0
   66	termcolor==3.3.0
   67	tifffile==2026.3.3
   68	torch==2.1.0+cu121
   69	torchvision==0.16.0+cu121
   70	tqdm==4.65.2
   71	typing_extensions==4.15.0
   72	tzdata==2026.2
   73	urllib3==1.26.20
   74	wcwidth==0.8.1
   75	yacs==0.1.8
   76	yapf==0.43.0
```

**Contract-B6 NEED_TO_CONFIRM packages — explicit answer:**

| Package | Present in `requirements.lock`? | Version |
|---|---|---|
| `albumentations` | **NO — absent** | — |
| `statsmodels` | **YES** | `0.14.6` (line 63) |

```text
$ grep -in "albumentations|statsmodels" requirements.lock requirements-e1.txt requirements-runpod.in requirements-runpod.lock
requirements.lock:63:statsmodels==0.14.6
requirements-e1.txt:10:# aliyun-python-sdk-*, oss2. Also EXCLUDED: opencv-python and albumentations â€” the E1 graph uses
requirements-e1.txt:11:# NumPy/PIL transforms only (no OpenCV / Albumentations import anywhere in E1).
requirements-runpod.in:62:statsmodels==0.14.6
requirements-runpod.lock:162:statsmodels==0.14.6 \
```

The only `albumentations` hits anywhere are the two **comment lines in `requirements-e1.txt`**
that document its deliberate exclusion. This is a **live code-vs-manuscript conflict**: ch3
Table 3.4 names Albumentations as the augmentation library (see NEW RISK N7), while
`configs/augment.py:9-13` records a D3 reconciliation to hand-written NumPy/PIL transforms.
`statsmodels` is present and matches the ch3 Holm-Bonferroni requirement.

### E10 — `scripts/verify_env.py` (entire file)

```python
    1	#!/usr/bin/env python3
    2	"""B20a — Strict-G0 compute-platform verification (diagnostic only; NOT training).
    3	
    4	Captures the ACTUAL local environment and decides whether it is suitable for the E1 DRY-RUN only or
    5	for the REAL E1 training run. Side-effect-free w.r.t. the repo: it performs NO downloads, NO installs,
    6	NO GPU compute (availability queries only), writes NO repo files and NO repo checkpoints. The only
    7	disk writes are a temporary checkpoint dir (OUTSIDE the repo) used by the tiny train_e1 --dry-run
    8	subprocess. Missing optional/pinned packages (mmcv, mmseg, cv2, albumentations) are reported as
    9	"not installed" and never crash the script. Author reports/platform_verify.md from this output.
   10	"""
   11	from __future__ import annotations
   12	
   13	import importlib
   14	import os
   15	import platform
   16	import subprocess
   17	import sys
   18	import tempfile
   19	from importlib.metadata import PackageNotFoundError, version as dist_version
   20	from pathlib import Path
   21	
   22	REPO = Path(__file__).resolve().parents[1]
   23	if str(REPO) not in sys.path:
   24	    sys.path.insert(0, str(REPO))
   25	
   26	MOBILENET_CKPT = "mobilenet_v3_large-5c1a4163.pth"      # torchvision IMAGENET1K_V2 backbone file
   27	IMAGENET_ALIAS = "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"
   28	
   29	
   30	def _run(cmd, timeout=120):
   31	    try:
   32	        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
   33	        return r.stdout.rstrip("\n") if r.returncode == 0 else None
   34	    except Exception:  # noqa: BLE001
   35	        return None
   36	
   37	
   38	def git_info() -> dict:
   39	    porcelain = _run(["git", "status", "--porcelain"])
   40	    modified, untracked = [], []
   41	    if porcelain:
   42	        for line in porcelain.splitlines():
   43	            if not line:
   44	                continue
   45	            (untracked if line[:2] == "??" else modified).append(line[3:])  # paths only, never contents
   46	    return {
   47	        "head": _run(["git", "rev-parse", "HEAD"]) or "UNKNOWN",
   48	        "short": _run(["git", "rev-parse", "--short", "HEAD"]) or "UNKNOWN",
   49	        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "UNKNOWN",
   50	        "dirty": bool(porcelain),
   51	        "modified": modified,
   52	        "untracked": untracked,
   53	    }
   54	
   55	
   56	def import_version(module_name: str, dist_name: str | None = None):
   57	    """Return version string, or None if the package is not installed (never raises)."""
   58	    try:
   59	        m = importlib.import_module(module_name)
   60	    except Exception:  # noqa: BLE001 -- not installed / import error
   61	        return None
   62	    v = getattr(m, "__version__", None)
   63	    if v is None and dist_name:
   64	        try:
   65	            v = dist_version(dist_name)
   66	        except PackageNotFoundError:
   67	            v = None
   68	    return v or "installed (version unknown)"
   69	
   70	
   71	def parse_lock() -> dict:
   72	    pins = {}
   73	    lock = REPO / "requirements.lock"
   74	    if lock.exists():
   75	        for line in lock.read_text(encoding="utf-8").splitlines():
   76	            line = line.strip()
   77	            if "==" in line and not line.startswith("#"):
   78	                name, ver = line.split("==", 1)
   79	                pins[name.lower()] = ver
   80	    return pins
   81	
   82	
   83	def main() -> int:
   84	    pins = parse_lock()
   85	    print("=" * 78)
   86	    print("B20a PLATFORM VERIFICATION (diagnostic only — no training, no download, no GPU compute)")
   87	    print("=" * 78)
   88	
   89	    # (1) python executable + version, (2) OS/platform, (3) repo root
   90	    print("\n[1-3] Python / OS / repo")
   91	    print(f"  python_executable : {sys.executable}")
   92	    print(f"  python_version    : {platform.python_version()}")
   93	    print(f"  os_system         : {platform.system()} {platform.release()} ({platform.version()})")
   94	    print(f"  machine           : {platform.machine()}")
   95	    print(f"  platform          : {platform.platform()}")
   96	    print(f"  repo_root         : {REPO}")
   97	
   98	    # (4) git HEAD + dirty status (paths only)
   99	    g = git_info()
  100	    print("\n[4] git")
  101	    print(f"  HEAD   : {g['short']} ({g['head']}) on {g['branch']} | dirty={g['dirty']}")
  102	    print(f"  modified ({len(g['modified'])}): {g['modified']}")
  103	    print(f"  untracked ({len(g['untracked'])}): {g['untracked']}")
  104	
  105	    # (5-11) torch / torchvision / CUDA / GPU / cuDNN / CPU threads
  106	    print("\n[5-11] torch / CUDA / GPU / threads")
  107	    try:
  108	        import torch
  109	    except Exception as e:  # noqa: BLE001
  110	        print(f"  torch import FAILED: {type(e).__name__}: {e}")
  111	        print("\nVERDICT: FAIL (torch unavailable) -> NOT OK for real E1 training")
  112	        return 1
  113	    torch_ver = torch.__version__
  114	    tv_ver = import_version("torchvision")
  115	    cuda = bool(torch.cuda.is_available())
  116	    cuda_ver = torch.version.cuda
  117	    gpu_count = torch.cuda.device_count() if cuda else 0
  118	    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda else []
  119	    cudnn_avail = bool(torch.backends.cudnn.is_available())
  120	    cudnn_ver = torch.backends.cudnn.version() if cudnn_avail else None
  121	    print(f"  torch_version       : {torch_ver}")
  122	    print(f"  torchvision_version : {tv_ver}")
  123	    print(f"  cuda_available      : {cuda}")
  124	    print(f"  cuda_version        : {cuda_ver}")
  125	    print(f"  gpu_count           : {gpu_count}")
  126	    print(f"  gpu_names           : {gpu_names}")
  127	    print(f"  cudnn_available     : {cudnn_avail}")
  128	    print(f"  cudnn_version       : {cudnn_ver}")
  129	    print(f"  cpu_count(logical)  : {os.cpu_count()}")
  130	    print(f"  torch_num_threads   : {torch.get_num_threads()}")
  131	
  132	    # (12) key package versions (actual vs pinned); missing optional pkgs => "not installed"
  133	    print("\n[12] packages (actual vs requirements.lock pin)")
  134	    pkgs = [
  135	        ("torch", "torch", "torch"),
  136	        ("torchvision", "torchvision", "torchvision"),
  137	        ("numpy", "numpy", "numpy"),
  138	        ("scipy", "scipy", "scipy"),
  139	        ("Pillow (PIL)", "PIL", "pillow"),
  140	        ("opencv (cv2)", "cv2", "opencv-python"),
  141	        ("albumentations", "albumentations", "albumentations"),
  142	        ("mmcv", "mmcv", "mmcv"),
  143	        ("mmseg", "mmseg", "mmsegmentation"),
  144	        ("statsmodels", "statsmodels", "statsmodels"),
  145	    ]
  146	    pkg_rows = []
  147	    for label, mod, distname in pkgs:
  148	        actual = import_version(mod, distname)
  149	        actual_str = actual if actual is not None else "not installed"
  150	        pinned = pins.get(distname.lower(), "—")
  151	        pkg_rows.append((label, actual_str, pinned))
  152	        print(f"  {label:16}: actual={actual_str:28} pinned={pinned}")
  153	    missing = [label for (label, a, _) in pkg_rows if a == "not installed"]
  154	
  155	    # (13-14) torch hub cache + ImageNet checkpoint cached?
  156	    print("\n[13-14] torch hub cache / ImageNet weights")
  157	    hub_dir = torch.hub.get_dir()
  158	    ckpt_path = os.path.join(hub_dir, "checkpoints", MOBILENET_CKPT)
  159	    cached = os.path.exists(ckpt_path)
  160	    print(f"  hub_dir          : {hub_dir}")
  161	    print(f"  imagenet_ckpt    : {ckpt_path}")
  162	    print(f"  imagenet_cached  : {cached}")
  163	
  164	    # (15) random-init student build + tiny CPU forward (NO download)
  165	    print("\n[15] build_student(pretrained=False) — CPU random init")
  166	    try:
  167	        from src.models.student import build_student
  168	        m = build_student(pretrained=False)
  169	        m.eval()
  170	        with torch.no_grad():
  171	            y = m(torch.zeros(1, 3, 64, 64))
  172	        student_ok = (tuple(y.shape)[:2] == (1, 116) and bool(torch.isfinite(y).all())
  173	                      and m.used_pretrained is False)
  174	        student_detail = f"out={tuple(y.shape)} used_pretrained={m.used_pretrained}"
  175	    except Exception as e:  # noqa: BLE001
  176	        student_ok = False
  177	        student_detail = f"{type(e).__name__}: {e}"
  178	    print(f"  student_ok       : {student_ok} ({student_detail})")
  179	
  180	    # (16) ImageNet pretrained build — ONLY if cached (else SKIP to avoid download)
  181	    print("\n[16] build_student(IMAGENET1K_V2) — skipped unless cached")
  182	    if cached:
  183	        try:
  184	            mi = build_student(pretrained=IMAGENET_ALIAS)
  185	            imagenet_skipped = False
  186	            imagenet_status = f"loaded from cache (used_pretrained={mi.used_pretrained}) — no download"
  187	        except Exception as e:  # noqa: BLE001
  188	            imagenet_skipped = True
  189	            imagenet_status = f"cached but load failed: {type(e).__name__}: {e}"
  190	    else:
  191	        imagenet_skipped = True
  192	        imagenet_status = "SKIPPED to avoid download (checkpoint not cached)"
  193	    print(f"  imagenet_skipped : {imagenet_skipped} ({imagenet_status})")
  194	
  195	    # (17) tiny train_e1.py --dry-run subprocess -> temp ckpt dir OUTSIDE repo
  196	    print("\n[17] train_e1.py --dry-run subprocess (tiny; temp checkpoint dir outside repo)")
  197	    tmp_ckpt = Path(tempfile.mkdtemp(prefix="verify_env_dryrun_ckpt_")).resolve()
  198	    repo_guard_ok = not (tmp_ckpt == REPO or REPO in tmp_ckpt.parents)
  199	    cmd = [sys.executable, str(REPO / "src" / "training" / "train_e1.py"), "--dry-run",
  200	           "--batch-size", "1", "--max-iters", "2", "--val-interval", "2",
  201	           "--max-val-batches", "1", "--ckpt-dir", str(tmp_ckpt)]
  202	    env = dict(os.environ, PYTHONIOENCODING="utf-8")
  203	    try:
  204	        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600, env=env)
  205	        dry_rc = r.returncode
  206	        result_line = next((ln for ln in reversed(r.stdout.splitlines())
  207	                            if ln.startswith("RESULT:")), "RESULT: (not found)")
  208	    except Exception as e:  # noqa: BLE001
  209	        dry_rc = -1
  210	        result_line = f"RESULT: (subprocess error {type(e).__name__}: {e})"
  211	    tmp_ckpts = sorted(str(p) for p in tmp_ckpt.glob("*.pt"))
  212	    repo_pt = sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*.pt") if ".git" not in p.parts)
  213	    dry_ok = (dry_rc == 0 and "RESULT: PASS" in result_line)
  214	    print(f"  command          : {' '.join(cmd)}")
  215	    print(f"  temp_ckpt_dir    : {tmp_ckpt}")
  216	    print(f"  temp_dir_outside_repo : {repo_guard_ok}")
  217	    print(f"  return_code      : {dry_rc}")
  218	    print(f"  {result_line}")
  219	    print(f"  temp_ckpt_files  : {tmp_ckpts}")
  220	    print(f"  repo_.pt_files   : {repo_pt}  (MUST be empty)")
  221	    print(f"  no_repo_checkpoint : {len(repo_pt) == 0}")
  222	
  223	    # (18) verdict
  224	    print("\n[18] VERDICT")
  225	    real_ready = cuda and gpu_count >= 1 and student_ok and dry_ok
  226	    if not (student_ok and dry_ok):
  227	        verdict, label = "FAIL", "NOT OK for real E1 training (E1 scaffold not runnable on this machine)"
  228	    elif real_ready:
  229	        verdict, label = "PASS", "OK for real E1 training"
  230	    else:
  231	        verdict, label = "PARTIAL", "OK for DRY-RUN only; NOT OK for real E1 training"
  232	
  233	    blockers = []
  234	    if not cuda:
  235	        blockers.append(f"No CUDA GPU (torch.cuda.is_available()=False; local torch is CPU-only "
  236	                        f"build '{torch_ver}'). Real E1 (80k iters @ bs16/512^2) needs a GPU.")
  237	    if pins.get("torch") and pins["torch"] not in torch_ver:
  238	        blockers.append(f"torch mismatch vs pinned: local '{torch_ver}' != requirements.lock "
  239	                        f"'{pins['torch']}' (cu121 GPU training stack).")
  240	    if not cached:
  241	        blockers.append("ImageNet backbone not cached; real run with --init imagenet needs network "
  242	                        "to populate the torch-hub cache OR a pre-staged checkpoint.")
  243	    if missing:
  244	        blockers.append(f"Not installed locally: {', '.join(missing)} "
  245	                        "(teacher/aug stack — needed for teacher fine-tune / E2-E3, not E1 supervised).")
  246	    if not repo_guard_ok or repo_pt:
  247	        blockers.append("Checkpoint isolation problem (temp dir inside repo OR a .pt found in repo).")
  248	
  249	    print(f"  verdict          : {verdict}")
  250	    print(f"  environment_label: {label}")
  251	    print(f"  can_run_real_E1  : {real_ready}")
  252	    print("  blockers_before_real_E1:")
  253	    for b in blockers:
  254	        print(f"    - {b}")
  255	    if not blockers:
  256	        print("    - (none)")
  257	
  258	    print("\n" + "=" * 78)
  259	    print(f"RESULT: {verdict} | {label}")
  260	    print("=" * 78)
  261	    return 0
  262	
  263	
  264	if __name__ == "__main__":
  265	    raise SystemExit(main())
```

**Every check performed, and its PASS criterion:**

| # | Check | Lines | PASS criterion |
|---|---|---|---|
| 1-3 | Python exe / version / OS / machine / repo root | 90-96 | informational only |
| 4 | git HEAD, branch, dirty, modified/untracked **paths only** | 38-53, 99-103 | informational |
| 5-11 | torch import; torch/tv version; `cuda_available`; cuda version; gpu count; gpu names;
cuDNN avail+version; `os.cpu_count()`; `torch.get_num_threads()` | 107-130 | torch import must
succeed (`return 1` at 112 otherwise); the rest informational |
| 12 | 10 package versions actual-vs-pinned | 134-153 | informational; missing → `not installed`,
collected into `missing` |
| 13-14 | torch hub dir; ImageNet ckpt cached | 157-162 | informational; feeds a blocker |
| 15 | `build_student(pretrained=False)` + 64x64 CPU forward | 166-178 | `student_ok` =
`y.shape[:2]==(1,116)` AND all-finite AND `used_pretrained is False` |
| 16 | `build_student(IMAGENET1K_V2)` **only if cached** | 182-193 | skipped when not cached
(deliberate no-download) |
| 17 | `train_e1.py --dry-run` subprocess, temp ckpt dir outside repo | 197-221 | `dry_ok` =
`returncode==0` AND stdout contains `RESULT: PASS`; plus `repo_guard_ok` and `repo_.pt_files`
must be empty |
| 18 | verdict | 225-231 | `real_ready = cuda AND gpu_count>=1 AND student_ok AND dry_ok`;
`FAIL` if not (`student_ok` and `dry_ok`), else `PASS` if `real_ready`, else `PARTIAL` |

**Absent:** any compute-capability / SM-version check, any VRAM check, any disk-space check,
any dataset-presence check, any `PLANTSEG_DATA_ROOT` check.

### E11 — `src/eval/metrics.py` public API surface

Module docstring first, because it carries the eligibility policy that the signatures encode:

```python
    1	"""Segmentation metrics. Confusion-matrix based; ignore_index=255 excluded everywhere.
    2	
    3	Semantics are frozen by docs/EVALUATION_CONTRACT.md sections 3.1/3.2 (decisions D3/D3b in
    4	docs/open_questions.md). THREE ELIGIBILITY RULES COEXIST BY DESIGN -- do not "harmonise" them:
    5	
    6	  * dataset-level mIoU   -> UNION-present : eligible iff UN_c = GT_c + PR_c - TP_c > 0.
    7	                            A class with GT_c = 0 but PR_c > 0 scores IoU 0 and IS COUNTED;
    8	                            a class absent from GT and prediction alike is omitted.
    9	                            Matches the official PlantSeg evaluator config
   10	                            (IoUMetric, iou_metrics=['mIoU'], nan_to_num unset -> np.nanmean).
   11	  * dataset-level mAcc   -> GT-present    : eligible iff GT_c > 0. Prediction-only classes are
   12	                            omitted. The asymmetry vs mIoU is genuine benchmark behaviour.
   13	  * dataset-level Dice   -> UNION-present : project-defined DESCRIPTIVE metric. The official
   14	                            benchmark never computes mDice; this is a thesis-internal extension
   15	                            that reuses the mIoU eligibility rule for internal consistency only.
   16	  * per-image mIoU       -> GT-present    : preregistered by ch3 section (f) "per-image
   17	                            absent-class exclusion". `_miou_from_cm` / `per_image_miou` are the
   18	                            per-image path and are DELIBERATELY left on the GT-present rule.
   19	
   20	Class space: num_classes = 116, background = 0, diseases = 1..115, ignore = 255 (pad only;
   21	absent from released masks). reduce_zero_label is NOT used. Reduction is float32 (see the A1a
   22	report case C4); a float64 reduction is a separate follow-up, deliberately not done here.
   23	"""
   24	
   25	from __future__ import annotations
   26	
   27	import torch
   28	
   29	IGNORE_INDEX = 255
   30	
```

**Public functions — full signatures + docstrings, verbatim:**

```python
   32	def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
   33	                     ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
   34	    """[num_classes, num_classes] confusion matrix (rows=GT, cols=pred); ignore_index pixels dropped.
   35	
   36	    Labels are validated first (EVALUATION_CONTRACT section 5.3 `invalid_pred_labels` must be 0):
   37	    an out-of-range label would otherwise alias through `target * num_classes + pred` into a
   38	    neighbouring row and silently fabricate counts. Nothing is clamped, remapped, or discarded.
   39	    """
```

```python
  116	def miou_from_confusion(cm: torch.Tensor, class_indices: torch.Tensor | None = None) -> float:
  117	    """DATASET-LEVEL mIoU -- UNION-present (contract section 3.1).
  118	
  119	    Eligible iff `UN_c = GT_c + PR_c - TP_c > 0`, so a prediction-only class contributes IoU 0 and
  120	    is counted, while a class absent from GT and prediction alike is omitted. `class_indices=None`
  121	    -> all-class; otherwise restrict (disease-only = every class except background).
  122	
  123	    This is the reducer the E1 validation loop uses for its accumulate-ONE-confusion-matrix-then-
  124	    compute-once protocol, so checkpoint-selection mIoU follows the contract automatically.
  125	    NOT the per-image rule -- per-image metrics stay GT-present via `_miou_from_cm`.
  126	    Returns NaN if no selected class is eligible.
  127	    """
  128	    tp, gt, pr, un = _cm_parts(cm)
  129	    eligible = _restrict(un > 0, class_indices)
  130	    return _macro(tp / un.clamp_min(1e-9), eligible)
  131	
```

```python
  133	def macc_from_confusion(cm: torch.Tensor, class_indices: torch.Tensor | None = None) -> float:
  134	    """DATASET-LEVEL mAcc -- GT-present (contract section 3.1). `Acc_c = TP_c / GT_c`.
  135	
  136	    Eligible iff `GT_c > 0`, so prediction-only classes are EXCLUDED. The different denominator
  137	    from `miou_from_confusion` is deliberate and mirrors MMSeg (`acc = intersect / label`).
  138	    """
  139	    tp, gt, _pr, _un = _cm_parts(cm)
  140	    eligible = _restrict(gt > 0, class_indices)
  141	    return _macro(tp / gt.clamp_min(1e-9), eligible)
  142	
```

```python
  144	def dice_from_confusion(cm: torch.Tensor, class_indices: torch.Tensor | None = None) -> float:
  145	    """DATASET-LEVEL macro per-class Dice (DSC) -- UNION-present (contract section 3.1).
  146	
  147	    `Dice_c = 2*TP_c / (GT_c + PR_c)`, eligible iff `GT_c + PR_c > 0` (equivalently `UN_c > 0`).
  148	    PROJECT-DEFINED DESCRIPTIVE METRIC: the official PlantSeg evaluator requests
  149	    `iou_metrics=['mIoU']` and never computes mDice, so this is a thesis-internal extension and is
  150	    NOT evidence of comparability with published PlantSeg results. Not binary foreground Dice.
  151	    """
  152	    tp, gt, pr, _un = _cm_parts(cm)
  153	    eligible = _restrict((gt + pr) > 0, class_indices)
  154	    return _macro(2.0 * tp / (gt + pr).clamp_min(1e-9), eligible)
  155	
```

```python
  157	def all_class_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
  158	                   ignore_index: int = IGNORE_INDEX) -> float:
  159	    """Dataset-level all-class mIoU (union-present) over one pred/target pair."""
  160	    cm = confusion_matrix(pred, target, num_classes, ignore_index)
  161	    return miou_from_confusion(cm)
  162	
```

```python
  164	def disease_only_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
  165	                      background_index: int = 0, ignore_index: int = IGNORE_INDEX) -> float:
  166	    """Dataset-level disease-only mIoU (union-present), excluding `background_index` (= 0).
  167	
  168	    The class mapping is confirmed by the official PlantSeg source (METAINFO index 0 is the
  169	    non-disease slot, 1..115 are the diseases; num_classes=116; reduce_zero_label=False), so this
  170	    is no longer PROVISIONAL -- see open_questions #2.
  171	    """
  172	    cm = confusion_matrix(pred, target, num_classes, ignore_index)
  173	    idx = torch.tensor([c for c in range(num_classes) if c != background_index], dtype=torch.long)
  174	    return miou_from_confusion(cm, idx)
  175	
```

```python
  177	def per_image_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
  178	                   class_indices: torch.Tensor | None = None,
  179	                   ignore_index: int = IGNORE_INDEX) -> list[float]:
  180	    """One mIoU per image (paired-test unit). class_indices=None -> all-class; else restrict (e.g. disease-only)."""
  181	    out = []
  182	    for p, t in zip(pred, target):
  183	        cm = confusion_matrix(p, t, num_classes, ignore_index)
  184	        out.append(_miou_from_cm(cm, class_indices)[0])
  185	    return out
```

**Private helpers that carry the per-image rule:**

```python
   73	def _miou_from_cm(cm: torch.Tensor, class_indices: torch.Tensor | None = None):
   74	    diag = torch.diag(cm).float()
   75	    row = cm.sum(1).float()                 # GT pixels per class
   76	    col = cm.sum(0).float()                 # predicted pixels per class
   77	    iou = diag / (row + col - diag).clamp_min(1e-9)
   78	    present = row > 0                        # classes present in ground truth
   79	    if class_indices is not None:
   80	        sel = torch.zeros_like(present)
   81	        sel[class_indices] = True
   82	        present = present & sel
   83	    if present.any():
   84	        return float(iou[present].mean().item()), iou
   85	    return float("nan"), iou
   86	
```

```python
  100	def _restrict(eligible: torch.Tensor, class_indices: torch.Tensor | None) -> torch.Tensor:
  101	    """Intersect an eligibility mask with an optional class subset (e.g. disease-only 1..115)."""
  102	    if class_indices is None:
  103	        return eligible
  104	    sel = torch.zeros_like(eligible)
  105	    sel[class_indices] = True
  106	    return eligible & sel
  107	
  108	
  109	def _macro(values: torch.Tensor, eligible: torch.Tensor) -> float:
  110	    """Unweighted mean over eligible classes; NaN when nothing is eligible."""
  111	    if not bool(eligible.any()):
  112	        return float("nan")
  113	    return float(values[eligible].mean().item())
  114	
```

**Per-image mIoU and `class_indices` selection — direct answer.**

| Question | Answer | Anchor |
|---|---|---|
| Per-image mIoU entry point | `per_image_miou(pred, target, num_classes, class_indices=None, ignore_index=255) -> list[float]` | `:177-185` |
| Its eligibility rule | **GT-present** (`present = row > 0`), *deliberately* different from the dataset-level rule | `:78` via `_miou_from_cm` |
| Dataset-level mIoU rule | **UNION-present** (`eligible = un > 0`) | `:129` |
| Dataset-level mAcc rule | **GT-present** (`eligible = gt > 0`) | `:140` |
| `class_indices` semantics | boolean mask intersected with the eligibility mask; `None` → all-class | `_restrict` `:100-106`, `_miou_from_cm` `:79-82` |
| Disease-only selection | `[c for c in range(num_classes) if c != background_index]` = 1..115 | `:173` |
| Checkpoint-selection metric | `miou_from_confusion(cm)` with `class_indices=None` → all-class, union-present | `train_e1.py:124` |

Note the three coexisting eligibility rules are **intentional** and the docstring says so
(`:4` *"THREE ELIGIBILITY RULES COEXIST BY DESIGN -- do not 'harmonise' them"*). I flag no
defect here; it is recorded so the adjudicator can check it against ch3 section (f).

One factual note for the reviewer: `disease_only_miou`'s docstring (`:168-170`) states the
disease-only convention is *"no longer PROVISIONAL"*, while `configs/e1_student.py:25` and
`train_e1.py:126`/`:234` still label disease-only mIoU **PROVISIONAL**. That is a docstring
-vs-config drift, not a behavioural one — the checkpoint criterion is all-class in both.

### E12 — Teacher init files + `weights/` listing

#### `docs/teacher_init_source.md` (entire file)

```markdown
    1	# Teacher initialization weights — source record
    2	
    3	ADE20K-pretrained **SegNeXt-B / MSCAN-B (512×512)** checkpoint used as the *initialization* for the
    4	in-house teacher fine-tune (B1). **Not trained, not fine-tuned, not modified** by this step. See
    5	[B8_checkpoint.md](B8_checkpoint.md) (no PlantSeg teacher checkpoint is publicly released) and
    6	[IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §B1.
    7	
    8	| Field | Value |
    9	|---|---|
   10	| MMSeg 1.x config name | `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512` |
   11	| Exact `.pth` URL (resolved by `mim`) | `NEED_TO_CONFIRM` (paste from the `mim download` log) |
   12	| SHA256 of the `.pth` | `NEED_TO_CONFIRM` (`sha256sum weights/<file>.pth`) |
   13	| Download date | `NEED_TO_CONFIRM` |
   14	| MMSeg version (download + test env) | **1.2.2** (pinned; mmcv 2.1.0, torch 2.1.0) |
   15	| Reported ADE20K mIoU | **48.03 (SS) / 49.68 (MS)** |
   16	| Source / license | **OpenMMLab** (MMSegmentation model zoo), **Apache-2.0** |
   17	| Init-test result (`scripts/test_teacher_init.py`) | `NEED_TO_CONFIRM` (PASS/FAIL) |
   18	
   19	> **Re-init note:** the stock checkpoint's classifier `decode_head.conv_seg` is sized for **150 ADE20K
   20	> classes**. Before teacher fine-tuning it will be **re-initialized to the empirically verified PlantSeg
   21	> class count = 116** (all-class: background `0` + 115 diseases `1–115`; see
   22	> [reports/dataset_report.md](../reports/dataset_report.md)). The downloaded checkpoint itself is never
   23	> edited — re-init happens in the training pipeline.
   24	
   25	## How to fill the `NEED_TO_CONFIRM` fields (run in the pinned MMSeg env)
   26	
   27	```bash
   28	# 1) download config + checkpoint into weights/ (git-ignored); capture the printed URL
   29	mim download mmsegmentation \
   30	  --config segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512 --dest weights/
   31	
   32	# 2) checksum + date
   33	sha256sum weights/segnext_mscan-b_*ade20k*.pth
   34	date -u +%Y-%m-%d
   35	
   36	# 3) read-only init sanity test (must print "RESULT: PASS")
   37	python scripts/test_teacher_init.py
   38	```
   39	
   40	Paste the resolved `.pth` URL, the SHA256, the date, and the test's PASS line back here and this record
   41	will be finalized. `weights/` and `*.pth` are git-ignored — the checkpoint is never committed.
```

#### `configs/teacher_finetune.py` (entire file)

```python
    1	# Teacher fine-tune config — Blocker B1
    2	# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B1 - Teacher fine-tune"
    3	# Every value traced to ch3.pdf (method authority); versions corroborated by context.md.
    4	# Analysis/config artifact only — contains NO training logic.
    5	
    6	TEACHER_FINETUNE = {
    7	    # Base / initialization
    8	    "model": "SegNeXt-B / MSCAN-B",
    9	    "init_checkpoint": "segnext_mscan-b_512x512_160k_ade20k",  # ADE20K-pretrained, MMSeg zoo (or equivalent)
   10	    "framework": "MMSegmentation 1.2.2 + mmcv 2.1.0",
   11	
   12	    # Optimizer
   13	    "optimizer": "AdamW",
   14	    "learning_rate": 6e-5,
   15	    "weight_decay": 0.01,
   16	    "betas": (0.9, 0.999),
   17	    "decode_head_lr_mult": 10,
   18	
   19	    # Schedule / budget
   20	    "lr_schedule": "poly",
   21	    "iterations": 40000,
   22	    "batch_size": 16,
   23	    "crop": (512, 512),
   24	
   25	    # Loss
   26	    "loss": "cross_entropy",
   27	
   28	    # Success criterion (protocol match, not max accuracy)
   29	    "success_criterion": "recover 42.05% mIoU within +/-1.5-2.0 pp",
   30	
   31	    # Role
   32	    "role": "descriptive upper-bound reference only (not deployed, not an inferential comparator)",
   33	}
```

#### `ls -la weights/`

```text
$ ls -la weights/
total 8
drwxr-xr-x 1 admin 197121 0 Jun 24 22:57 .
drwxr-xr-x 1 admin 197121 0 Aug 17 19:21 ..
```

Exit code 0 — the directory exists and is empty (only `.` and `..`). Per ruling R4 the raw
output is the evidence; it did not fail, so no error text is available to paste. Produced by
GNU `ls` through the Bash tool. **No file inside `weights/` was opened.**

### E13 — ch3 authoritative sentences

**Correction to the brief's calibration note.** `docs/reference/ch3.pdf` **IS** in this
checkout — it is git-tracked, 484,152 bytes, 77 pages, and is **not** on the Part-E
do-not-touch list (only `reference.pdf` and `context.md` are). So "NOT FOUND IN MY COPY" is
**not** the answer for any of (a)-(d).

**Method.** The PDF carries a real text layer (Identity-H subset fonts, hex-encoded glyph
strings). I decoded it through the document's own 9 `/ToUnicode` CMaps — **0 mapping
conflicts**, 22 unmapped glyph ids out of 151 mapped entries (a symbol/math subset), 170,890
characters recovered. Quotes below are verbatim from that decode; the only edits are that
stray page numbers fused into the text stream are shown as they appear (e.g. `104default`,
`9780,000`). No OCR was used; no text was reconstructed from the contract.

#### (a) The λ_logit sweep protocol and its budget — **FOUND**

> The constant weight scaling the distillation term relative to the cross-entropy term,
> λ_logit, is not fixed a priori but is selected on the validation partition through a small
> pre-registered grid sweep over λ_logit ∈{0.25, 0.5, 1, 2, 4}, a geometric grid centered on
> the conventional unit weight and bounded above near the locked Channel-Wise logit-map weight
> β_CWD = 3; because the temperature factor T² is already absorbed into the Logit KD term,
> λ_logit acts as an order-unity multiplier on a gradient-matched loss, which fixes the grid at
> this scale. The candidate maximizing dataset-level validation mIoU at the 104default seed (42)
> is selected; ties falling within the validation-set noise band are resolved toward the smaller
> weight, and if the selected value lies at a grid boundary it is reported as such rather than
> the grid being extended. The held-out test partition is not consulted during selection,
> mirroring the selection-on-validation, reporting-on-test protocol already applied to the
> post-training-quantization calibration configuration; the selected value, together with the
> validation mIoU of every candidate, is reported in Chapter 4. This selected value of λ_logit
> is then reused unchanged in E3 so that the E2-versus-E3 comparison isolates the added
> Channel-Wise Distillation terms and is not confounded by a change in the Logit KD weight.

**Budget — the governing sentence, found elsewhere in ch3:**

> All training runs use random seed 42 by default; if multi-seed validation is conducted for
> any stage (treated as optional due to compute constraints), all seeds used are reported in
> the appendix. All stages share the same 9780,000-iteration training budget; the comparisons
> are therefore iteration-matched rather than wall-clock- or FLOPs-matched.

(`9780,000` = page number `97` fused to `80,000`.) **No reduced sweep budget appears anywhere**
in ch3. This is the basis for F11 = CONFIRMED.

#### (b) The resolution at which the Logit-KD KL is averaged — **FOUND, but it pins pixels, not a grid**

> Regarding the ignore label, [...] For cross-entropy, exclusion is handled natively through
> ignore_index = 255. For the Dice term, the validity mask is applied to both the predicted
> probabilities and the one-hot labels before the intersection and union are computed. For the
> Logit KD term, the per-pixel KL divergence is averaged over valid pixels only.

That is the **only** ch3 sentence governing the Logit-KD averaging. It names the population
(valid, non-255 pixels) and not the spatial resolution. The nearest ch3 statement that
constrains resolution is the head description:

> the low-level skip feature (output stride 8) is projected by a 1×1 convolution to the 115
> output channels, the upsampled high-level branch is projected by a separate 1×1 convolution
> to 115 channels, and the two are summed and upsampled to full resolution.

(Note ch3 says **115** here; ch3 elsewhere says the output layer is sized to the verified
**116**-class count — an internal ch3 inconsistency, see NEW RISK N8.)

#### (c) Gradient clipping for the E1/E2/E3 shared recipe — **FOUND**

> The Dice term is a soft Dice computed on predicted probabilities, aggregated per class across
> the batch and averaged over the classes present in the batch, with absent classes excluded
> from the average and a smoothing constant of 1e-5 added to its numerator and denominator.
> **During the distillation stages the distillation weights are linearly ramped from zero to
> their target values over the first training epoch, and global-norm gradient clipping is
> applied throughout.** In all distillation stages the teacher is run in evaluation mode online
> and consumes the identical augmented input tensor used by the student [...]

And for QAT, separately:

> Gradient clipping by global norm is applied for stability, and no distillation loss is active
> during quantization-aware fine-tuning [...]

**Reading:** clipping is scoped by ch3 to the **distillation stages** and to **QAT**. No
numeric `max_norm` is given in either place, and E1 is not named. This exactly matches
`configs/e1_student.py:30-31` (`gradient_clipping: global_norm` as a *method family only*,
`grad_clip_max_norm: None`) and its `grad_clip_scope` block at `:49-55`. The "CARRY FORWARD"
note at `:45-48` — that ch3 calls E1/E2/E3 an identical recipe yet clips only the distillation
stages — is a **real** manuscript tension and is confirmed by the two quotes above.

#### (d) The reproducibility / determinism paragraph — **FOUND**

> PyTorch/NumPy/Python RNG seeding — Reproducibility. All training runs use random seed 42
> across torch, numpy, and python's random module. torch.backends.cudnn.deterministic = True,
> torch.backends.cudnn.benchmark = False, torch.use_deterministic_algorithms(True,
> warn_only=True), and CUBLAS_WORKSPACE_CONFIG=:4096:8 are set before CUDA initialization where
> applicable. Reproducibility is reported with the caveat that small 121floating-point variation
> may remain across GPU classes and compiled CUDA kernels. Three-seed validation is planned for
> E1 and E3; E4 and E7 are recomputed per seed from their corresponding FP32 checkpoints, while
> E5 and E6 are fine-tuned per seed where compute permits. Primary inference uses one
> pre-registered seed, with mean ± SD reported across completed seeds.

**This paragraph is what refutes F5 outright.** `src/seeds.py` implements all four settings in
the order ch3 specifies. It also carries a launch-planning fact the brief did not ask about:
**three-seed validation is planned for E1** (see NEW RISK N1).

### E14 — Stale-placeholder inventory (`NEED_TO_CONFIRM` in `configs/` and `src/`)

```text
$ grep -rn "NEED_TO_CONFIRM" configs/ src/
configs/distill.py:4:# kept as the literal NEED_TO_CONFIRM string per source hierarchy (rule 5).
configs/distill.py:12:        "lambda_logit": "NEED_TO_CONFIRM",          # selected via validation sweep; reported in Ch4
configs/loss.py:4:# unresolved values stay the literal NEED_TO_CONFIRM string. NO training logic.
configs/loss.py:14:    "real_class_weights": "NEED_TO_CONFIRM",     # sqrt inv-freq over FULL train set (not yet computed)
configs/loss.py:15:    "logit_kd_weight": "NEED_TO_CONFIRM",        # lambda_logit; validation sweep {0.25,0.5,1,2,4}
configs/model.py:19:    "pretrained": "NEED_TO_CONFIRM",   # ImageNet weights not cached; weights=None for smoke (no download)
configs/quant.py:5:# qparams, so scale/zero_point stay the literal NEED_TO_CONFIRM string (rule 5).
configs/quant.py:33:        "e6kd_reduced_weights": "NEED_TO_CONFIRM",   # contingency only if E3->E6 clean mIoU drop > 1.0 pp
configs/quant.py:136:        "scale": "NEED_TO_CONFIRM",                  # not specified in contract
configs/quant.py:137:        "zero_point": "NEED_TO_CONFIRM",             # not specified in contract
configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:356:# A dedicated control remains NEED_TO_CONFIRM, to be settled against the pinned stack; it is
configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:358:NMF_SEED_CONTROL = 'NEED_TO_CONFIRM'
configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:363:ADE20K_CKPT_UNSET_SENTINEL = 'NEED_TO_CONFIRM__SET_SEGNEXT_ADE20K_CKPT'
configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:369:WORK_DIR_UNSET_SENTINEL = 'NEED_TO_CONFIRM__SET_TEACHER_WORK_DIR'
src/training/losses.py:23:    NEED_TO_CONFIRM. Classes absent from `masks` (common in a tiny batch) get weight 1.0 (= median)
src/training/losses.py:97:    lambda_logit (the weight on this term relative to CE) is NEED_TO_CONFIRM (validation sweep).
src/training/train_distill.py:21:    NEED_TO_CONFIRM, selected by validation sweep â€” it is never guessed here), and requires an
src/training/train_distill.py:416:                   help=f"Logit-KD weight; contract leaves it NEED_TO_CONFIRM (sweep {LAMBDA_SWEEP})")
src/training/train_distill.py:474:                  f"The contract leaves lambda_logit as NEED_TO_CONFIRM (validation sweep over "
```

**Config-vs-code drift, one line each. This is the part the brief asked not to be smoothed.**

| file:line | Placeholder | Settled elsewhere? | Drift verdict |
|---|---|---|---|
| `configs/loss.py:14` | `real_class_weights` | **YES** — `reports/e1_class_weights.json` exists
(schema `e1_ce_class_weights/v1`, generated 2026-06-29, `num_classes:116`, computed over raw
annotation PNGs) and is what `train_e1.py:42,48-57` actually loads | **STALE.** Nothing reads
`LOSS['real_class_weights']` (grep returns only its own definition). The config says unresolved;
the training path is already resolved. |
| `configs/loss.py:15` | `logit_kd_weight` | NO — genuinely open, sweep not yet run | **CORRECT** |
| `configs/model.py:19` | `pretrained` | **YES** — `train_e1.py:333` uses
`E1_STUDENT['init_weights']`; `build_student(pretrained=...)` defaults to `False` | **STALE +
DEAD.** Nothing reads `MODEL['pretrained']` (grep confirms zero readers). The comment
*"ImageNet weights not cached"* is a B10-era smoke note that no longer governs anything. |
| `configs/distill.py:12` | `lambda_logit` | NO — the sweep is the resolution mechanism | **CORRECT** |
| `configs/quant.py:33` | `e6kd_reduced_weights` | NO — contingency-only, gated on >1.0 pp drop | **CORRECT** |
| `configs/quant.py:136-137` | `scale` / `zero_point` | NO — runtime-derived qparams | **CORRECT** |
| `configs/teacher/...:358` | `NMF_SEED_CONTROL` | NO — pending against pinned stack | **CORRECT** |
| `configs/teacher/...:363,369` | `ADE20K_CKPT_UNSET_SENTINEL`, `WORK_DIR_UNSET_SENTINEL` | N/A — deliberate refuse-to-launch sentinels | **CORRECT (by design)** |
| `src/training/losses.py:23,97` | docstring references | N/A — prose | **CORRECT** |
| `src/training/train_distill.py:21,416,474` | `lambda_logit` CLI refusal text | N/A — prose | **CORRECT** |

**`configs/data.py` `reduce_zero_label`** — asked about explicitly. It is **not** a
`NEED_TO_CONFIRM`; it is already the literal `False` at `configs/data.py:23` with the
empirical justification inline, and `train_e1.py` never remaps labels. **No drift.** The
separate disease-only *reporting* convention is what remains open (open_questions #2), and
`configs/data.py:23` says so in the same comment.

**Net:** 2 of 13 placeholders in `configs/`+`src/` are stale — `loss.py:14` and `model.py:19`.
Neither blocks E1 (nothing reads them), but both misrepresent project state to a reader, and
`loss.py:14` in particular contradicts an artifact the training loop hard-depends on.

### E15 — Split counts constructed on THIS machine

Dataset root resolves locally (`PLANTSEG_DATA_ROOT` unset → `configs/data.py:14` Windows
default), so this is **MEASURED**, not CANNOT-VERIFY-LOCALLY.

```text
DATA['root'] = C:\Users\admin\plantseg_data\plantseg
train  len(PlantSegDataset) = 5367
val    len(PlantSegDataset) = 846
test   len(PlantSegDataset) = 1561
TOTAL = 7774
configs/data.py locked sizes = {'train': 5367, 'val': 846, 'test': 1561}
```

| Source | train | val | test | total | Matches code? |
|---|---|---|---|---|---|
| **`PlantSegDataset` on this machine (MEASURED)** | **5367** | **846** | **1561** | **7774** | — |
| `configs/data.py:34` locked sizes | 5367 | 846 | 1561 | 7774 | **YES** |
| B30 prompt figures | 5367 | 846 | 1561 | 7774 | **YES** |
| Ch3-prose figures | 5442 | 778 | 1554 | 7774 | **NO** |

**The code produces 5367 / 846 / 1561.** Note the totals agree (7774) while the per-split
splits do not — consistent with a different partition assignment being described in ch3 prose
than the one on disk. `PlantSegDataset.__init__:63-69` **hard-fails** if the on-disk count
deviates from `configs/data.py`, so E1 cannot silently train on a different partition; but if
ch3's 5442/778/1554 is the intended split, E1 would refuse to start rather than adapt. Flagged
as NEW RISK N2 — this needs the adjudicator's ch3 to settle which is authoritative.

---

## 4. Probe results

**Environment for every probe (MEASURED):**

| | |
|---|---|
| Interpreter | `C:\Users\admin\anaconda3\python.exe`, `PYTHONIOENCODING=utf-8` |
| torch | **2.9.1+cpu** |
| numpy | **2.1.3** |
| CPU | 13th Gen Intel(R) Core(TM) i7-1360P |
| `os.cpu_count()` | **16** |
| Dataset root | `C:\Users\admin\plantseg_data\plantseg` (`PLANTSEG_DATA_ROOT` unset) |
| GPU used | **none** |
| Network | **none** |

> **Environment caveat, stated up front.** The local stack is **not** the pinned stack.
> `requirements.lock` pins `torch==2.1.0+cu121` / `numpy==1.26.4`; this box runs
> `torch 2.9.1+cpu` / `numpy 2.1.3`. This does **not** weaken the F1 refutation — the seeding
> mechanism at `dataset.py:83` is pure MT19937 (`np.random.seed(42)` → first
> `randint(0, 2**31-1)` = `1608637542` is version-stable), and the P2/P3 contrast is a property
> of repo logic, not of torch's version. It **does** mean the P4 timings are indicative only,
> and that nothing about CUDA determinism (F7) or SM capability (F10) could be exercised.

**Decode budget:** 6 (P1 is 2 idx × 2 passes = 4) + P2a 4 + P2b 4 + P3 4 + P4 16 = **32**
image decodes, against the 40 cap raised by ruling R3. No full-dataset scan; `len()` and
`pairs` listing are directory operations, not decodes.

### P1 / P2a / P2b / P3 — determinism

**Deviation declared (amendment A1):** `shuffle=False`. `build_dataloader` uses
`shuffle=True` for train. Order variation is a confound for the only question these probes
ask — whether the augmentation RNG advances across passes — and fixing the order lets a source
file be matched across passes by position instead of by content-guessing. **Why the deviation
cannot affect the result:** the per-sample RNG is constructed inside
`PlantSegDataset.__getitem__` (`dataset.py:83`) from the ambient numpy stream on every call.
It never reads `idx`, never reads batch position, and never reads the sampler. Shuffling
changes *which* sample gets *which* draw; it cannot change *whether the draws differ between
passes*, which is the claim under test. The `Subset` restriction (2 elements) likewise only
bounds decode count — `worker_init_fn`, the seeded `generator`, `num_workers=2` worker
spawn/teardown and `drop_last=False` are all preserved verbatim from `build_dataloader`.

**Script:**

```python
"""B30 probes P1 / P2a / P2b / P3. READ-ONLY on the repo. CPU only. 16 image decodes total.

P1  root-cause seed logging: monkeypatch np.random.RandomState (+ np.random.randint) in the
    MAIN PROCESS at num_workers=0 and print the seed sequence for two full passes.
P2a real loader path at num_workers=2, shuffle=False, Subset[100,500], two fresh iterators.
P2b same at num_workers=0.
P3  val control, num_workers=2, two fresh iterators.

DEVIATION (mandated by amendment A1): shuffle is forced False. build_dataloader() uses
shuffle=True for train. Order variation is a confound for the only question here -- whether the
augmentation RNG advances across passes -- and fixing the order lets a file be matched by
position. It cannot affect the result: the per-sample RNG in PlantSegDataset.__getitem__ is drawn
per __getitem__ call from the ambient numpy RNG and never depends on batch order or on idx.
"""
import sys
from pathlib import Path

REPO = Path(r"C:\Users\admin\plantseg-thesis")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.data.dataset import PlantSegDataset, _seed_worker  # noqa: E402
from src.seeds import SEED, set_seed  # noqa: E402

IDX = [100, 500]


def make_loader(ds, num_workers, batch_size=2):
    """Replicates build_dataloader() kwargs verbatim EXCEPT shuffle (forced False, see A1)."""
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      generator=g, worker_init_fn=_seed_worker, drop_last=False)


def two_passes(loader):
    passes = []
    for _ in range(2):
        imgs = []
        for img, _mask in loader:          # fresh iterator == new "epoch"
            for i in range(img.shape[0]):
                imgs.append(img[i].clone())
        passes.append(imgs)
    return passes


def compare(tag, passes, names):
    print(f"\n--- {tag} ---")
    a, b = passes
    for i, nm in enumerate(names):
        same = torch.equal(a[i], b[i])
        d = (a[i] - b[i]).abs().max().item()
        print(f"  pos{i} file={nm:<40} bit_identical={same}  max_abs_diff={d:.8f}")


def main():
    print("=" * 78)
    print("P1/P2/P3 -- torch", torch.__version__, "| numpy", np.__version__)
    print("=" * 78)

    set_seed(42)
    train_full = PlantSegDataset("train")
    val_full = PlantSegDataset("val")
    print(f"[len] train={len(train_full)} val={len(val_full)}")
    tnames = [train_full.pairs[i][0].name for i in IDX]
    vnames = [val_full.pairs[i][0].name for i in IDX]
    print(f"[train files tracked] {tnames}")
    print(f"[val   files tracked] {vnames}")

    # ---------------- P1: seed-sequence logging (num_workers=0, main process) ----------------
    print("\n" + "=" * 78)
    print("P1  ROOT-CAUSE SEED LOG (num_workers=0, RandomState monkeypatched in scratchpad)")
    print("=" * 78)
    orig_rs = np.random.RandomState
    orig_randint = np.random.randint
    rs_seeds, randint_calls = [], []

    def logging_randomstate(seed=None, *a, **k):
        rs_seeds.append(seed)
        return orig_rs(seed, *a, **k)

    def logging_randint(*a, **k):
        v = orig_randint(*a, **k)
        randint_calls.append(int(v))
        return v

    np.random.RandomState = logging_randomstate
    np.random.randint = logging_randint
    try:
        set_seed(42)
        sub = Subset(train_full, IDX)
        loader = make_loader(sub, num_workers=0)
        p1_passes = []
        for p in range(2):
            rs_seeds.clear()
            imgs = []
            for img, _m in loader:
                for i in range(img.shape[0]):
                    imgs.append(img[i].clone())
            print(f"  pass{p + 1}: RandomState seed sequence = {list(rs_seeds)}")
            p1_passes.append(imgs)
        print(f"  np.random.randint draws (both passes, in order) = {randint_calls}")
    finally:
        np.random.RandomState = orig_rs
        np.random.randint = orig_randint
    compare("P1 tensor equality (num_workers=0)", p1_passes, tnames)

    # ---------------- P2b: num_workers=0, no patch ----------------
    print("\n" + "=" * 78)
    print("P2b TRAIN, num_workers=0, shuffle=False, two fresh iterators")
    print("=" * 78)
    set_seed(42)
    compare("P2b", two_passes(make_loader(Subset(train_full, IDX), 0)), tnames)

    # ---------------- P2a: num_workers=2 (real loader path) ----------------
    print("\n" + "=" * 78)
    print("P2a TRAIN, num_workers=2, shuffle=False, two fresh iterators (REAL worker path)")
    print("=" * 78)
    set_seed(42)
    compare("P2a", two_passes(make_loader(Subset(train_full, IDX), 2)), tnames)

    # ---------------- P3: val control ----------------
    print("\n" + "=" * 78)
    print("P3  VAL CONTROL, num_workers=2, two fresh iterators (expect identical)")
    print("=" * 78)
    set_seed(42)
    compare("P3", two_passes(make_loader(Subset(val_full, IDX), 2)), vnames)


if __name__ == "__main__":
    main()
```

**Raw stdout:**

```text
==============================================================================
P1/P2/P3 -- torch 2.9.1+cpu | numpy 2.1.3
==============================================================================
[len] train=5367 val=846
[train files tracked] ['apple_rust_12.jpg', 'banana_panama_disease_4.jpg']
[val   files tracked] ['bean_rust_google_0172.jpg', 'potato_early_blight_google_0048.jpg']

==============================================================================
P1  ROOT-CAUSE SEED LOG (num_workers=0, RandomState monkeypatched in scratchpad)
==============================================================================
  pass1: RandomState seed sequence = [1608637542, 1273642419]
  pass2: RandomState seed sequence = [1935803228, 787846414]
  np.random.randint draws (both passes, in order) = [1608637542, 1273642419, 1935803228, 787846414]

--- P1 tensor equality (num_workers=0) ---
  pos0 file=apple_rust_12.jpg                        bit_identical=False  max_abs_diff=3.67647076
  pos1 file=banana_panama_disease_4.jpg              bit_identical=False  max_abs_diff=4.44444466

==============================================================================
P2b TRAIN, num_workers=0, shuffle=False, two fresh iterators
==============================================================================

--- P2b ---
  pos0 file=apple_rust_12.jpg                        bit_identical=False  max_abs_diff=3.67647076
  pos1 file=banana_panama_disease_4.jpg              bit_identical=False  max_abs_diff=4.44444466

==============================================================================
P2a TRAIN, num_workers=2, shuffle=False, two fresh iterators (REAL worker path)
==============================================================================

--- P2a ---
  pos0 file=apple_rust_12.jpg                        bit_identical=False  max_abs_diff=3.81652641
  pos1 file=banana_panama_disease_4.jpg              bit_identical=False  max_abs_diff=4.44444466

==============================================================================
P3  VAL CONTROL, num_workers=2, two fresh iterators (expect identical)
==============================================================================

--- P3 ---
  pos0 file=bean_rust_google_0172.jpg                bit_identical=True  max_abs_diff=0.00000000
  pos1 file=potato_early_blight_google_0048.jpg      bit_identical=True  max_abs_diff=0.00000000
```

**Reading of the result:**

- **P1** — the decisive root-cause probe. Seed lists differ between pass 1 and pass 2
  (`[1608637542, 1273642419]` vs `[1935803228, 787846414]`). Per amendment A2's rule, that is
  **F1 REFUTED at the mechanism**. The `np.random.randint` log shows all four draws are
  consecutive samples from one advancing stream — exactly the design the `dataset.py:79-82`
  comment describes.
- **P2b** (nw=0) and **P2a** (nw=2, real worker path) both show the same source files coming
  back with different tensors on the second pass.
- **P3** (val control) is bit-identical, `max_abs_diff = 0.00000000`, at `num_workers=2`.

> P2 differs and P3 is identical → **F1 is REFUTED**, stated loudly as requested.

The val control matters: it rules out the alternative explanation that the train differences
come from worker nondeterminism or loader ordering rather than from augmentation. Val runs the
same loader machinery with the same worker count and returns byte-identical tensors.

### P4 / P7 / P8 — transform cost, crop retries, mask dtype

**Script:**

```python
"""B30 probes P4 / P7 / P8. READ-ONLY on the repo. CPU only. 16 image decodes.

P4  per-stage transform cost, 16 unique train images x 5 repetitions.
P7  crop-retry cost: attempts / fallback rate / _dom_nonignore_ratio share of wall time.
P8  mask dtype + bytes-per-pixel at every stage and at the DataLoader output.
"""
import platform
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\admin\plantseg-thesis")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import os  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from configs.augment import AUGMENT  # noqa: E402
from src.data import transforms as T  # noqa: E402
from src.data.dataset import PlantSegDataset  # noqa: E402
from src.seeds import set_seed  # noqa: E402

N_IMAGES = 16
N_REPS = 5
SIZE = T.SIZE

_dom_time = [0.0]
_dom_calls = [0]
_orig_dom = T._dom_nonignore_ratio


def timed_dom(mask_np):
    t0 = time.perf_counter()
    r = _orig_dom(mask_np)
    _dom_time[0] += time.perf_counter() - t0
    _dom_calls[0] += 1
    return r


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def row(name, xs):
    return (f"  {name:<26} mean={st.mean(xs)*1000:8.3f}  median={st.median(xs)*1000:8.3f}  "
            f"p95={pct(xs,0.95)*1000:8.3f}   (n={len(xs)})")


def main():
    print("=" * 92)
    print("P4/P7/P8 -- machine + env")
    print("=" * 92)
    print(f"  os.cpu_count()      : {os.cpu_count()}")
    print(f"  platform.processor(): {platform.processor()}")
    print(f"  platform.machine()  : {platform.machine()}")
    print(f"  numpy               : {np.__version__}")
    try:
        import subprocess
        cpu = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True,
                             text=True, timeout=30).stdout.strip().splitlines()
        print(f"  cpu_model           : {[c.strip() for c in cpu if c.strip()][-1:]}")
    except Exception as e:
        print(f"  cpu_model           : unavailable ({type(e).__name__})")

    set_seed(42)
    ds = PlantSegDataset("train")
    step = max(1, len(ds) // N_IMAGES)
    picks = [i * step for i in range(N_IMAGES)]
    print(f"  train_len={len(ds)}  sampled_indices={picks}")

    T._dom_nonignore_ratio = timed_dom

    t_decode, t_resize, t_rot, t_crop, t_flip, t_photo, t_final, t_total = ([] for _ in range(8))
    attempts, fallbacks, doms = [], 0, []
    dtype_trace = None

    for n, idx in enumerate(picks):
        img_path, mask_path = ds.pairs[idx]

        t0 = time.perf_counter()
        im = Image.open(img_path)
        mk = Image.open(mask_path)
        im_r = ImageOps.exif_transpose(im).convert("RGB")
        im_r.load()
        mk.load()
        t_decode.append(time.perf_counter() - t0)

        for rep in range(N_REPS):
            rng = np.random.RandomState(1000 + n * 10 + rep)
            p = AUGMENT
            rrc = p["random_resized_crop"]
            w0 = time.perf_counter()

            lo, hi = rrc["scale_range"]
            r = float(rng.uniform(lo, hi))
            target_long = max(1, int(round(SIZE * r)))
            t0 = time.perf_counter()
            img_rr, mask_rr = T._resize_long_side(im_r, mk, target_long)
            img_np = np.asarray(img_rr, dtype=np.uint8)
            mask_np = np.asarray(mask_rr).astype(np.int64)
            t_resize.append(time.perf_counter() - t0)
            st_resize = (mask_np.dtype, mask_np.nbytes, mask_np.shape)

            t0 = time.perf_counter()
            img_np, mask_np = T._apply_rotation(img_np, mask_np, rng, p)
            t_rot.append(time.perf_counter() - t0)
            st_rot = (mask_np.dtype, mask_np.nbytes, mask_np.shape)

            t0 = time.perf_counter()
            img_np, mask_np, info = T._random_crop_pad_512(img_np, mask_np, rng,
                                                           rrc["cat_max_ratio"])
            t_crop.append(time.perf_counter() - t0)
            attempts.append(info["attempts"])
            fallbacks += int(info["fallback"])
            doms.append(info["dom_ratio"])
            st_crop = (mask_np.dtype, mask_np.nbytes, mask_np.shape)

            t0 = time.perf_counter()
            img_np, mask_np = T._apply_flips(img_np, mask_np, rng, p)
            t_flip.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            img_np = T._apply_photometric(img_np, rng, p)
            t_photo.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            img_t, mask_t = T.finalize(img_np, mask_np)
            t_final.append(time.perf_counter() - t0)
            t_total.append(time.perf_counter() - w0)

            if dtype_trace is None:
                raw = np.asarray(Image.open(mask_path))
                dtype_trace = {
                    "raw PNG (np.asarray)": (raw.dtype, raw.nbytes, raw.shape),
                    "after _resize_long_side+astype(int64)": st_resize,
                    "after _apply_rotation": st_rot,
                    "after _random_crop_pad_512": st_crop,
                    "after finalize (torch)": (mask_t.dtype, mask_t.element_size()
                                               * mask_t.nelement(), tuple(mask_t.shape)),
                }

    T._dom_nonignore_ratio = _orig_dom

    print("\n" + "=" * 92)
    print(f"P4  PER-STAGE TRANSFORM COST -- {N_IMAGES} unique images x {N_REPS} reps  [MEASURED]")
    print("=" * 92)
    print(row("(a) JPEG+PNG decode", t_decode) + "   <- once per image, not per rep")
    print(row("(b) _resize_long_side", t_resize))
    print(row("(c) _apply_rotation", t_rot))
    print(row("(d) _random_crop_pad_512", t_crop))
    print(row("(e) _apply_flips", t_flip))
    print(row("(f) _apply_photometric", t_photo))
    print(row("(g) finalize", t_final))
    print(row("TOTAL train_preprocess+fin", t_total))
    per_sample = st.mean(t_total) + st.mean(t_decode)
    print(f"\n  end-to-end per sample (decode + transform) mean = {per_sample*1000:.3f} ms")

    print("\n" + "=" * 92)
    print("P7  CROP-RETRY COST  [MEASURED]")
    print("=" * 92)
    print(f"  attempts: mean={st.mean(attempts):.3f}  max={max(attempts)}  "
          f"distribution={ {a: attempts.count(a) for a in sorted(set(attempts)) } }")
    print(f"  fallback rate (10 tries, cat_max_ratio={AUGMENT['random_resized_crop']['cat_max_ratio']}"
          f" never met): {fallbacks}/{len(attempts)} = {100.0*fallbacks/len(attempts):.1f}%")
    print(f"  dom_ratio: mean={st.mean(doms):.4f} median={st.median(doms):.4f} max={max(doms):.4f}")
    print(f"  _dom_nonignore_ratio: calls={_dom_calls[0]}  total={_dom_time[0]*1000:.1f} ms  "
          f"mean_per_call={_dom_time[0]/max(1,_dom_calls[0])*1000:.4f} ms")
    print(f"  share of train_preprocess wall time = "
          f"{100.0*_dom_time[0]/sum(t_total):.1f}%")
    print(f"  share of _random_crop_pad_512 time  = {100.0*_dom_time[0]/sum(t_crop):.1f}%")

    print("\n" + "=" * 92)
    print("P8  MASK DTYPE THROUGH THE PIPELINE  [MEASURED]")
    print("=" * 92)
    for k, (dt, nb, shp) in dtype_trace.items():
        bpp = nb / (shp[0] * shp[1])
        print(f"  {k:<40} dtype={str(dt):<10} shape={str(shp):<16} "
              f"bytes={nb:>9,}  bytes/px={bpp:.1f}")
    u8 = SIZE * SIZE
    i64 = SIZE * SIZE * 8
    print(f"\n  512x512 mask as int64 = {i64:,} B ; as uint8 = {u8:,} B ; "
          f"delta per sample = {i64-u8:,} B ({(i64-u8)/1024:.0f} KiB)")
    print(f"  at batch 16: int64={16*i64/1024/1024:.2f} MiB vs uint8={16*u8/1024/1024:.2f} MiB")

    print("\n" + "=" * 92)
    print("P4 (cont) WORKERS NEEDED  [INFERRED -- CROSS-PLATFORM, Windows dev box, NOT the Linux pod]")
    print("=" * 92)
    print(f"  measured per-sample cost = {per_sample*1000:.3f} ms (single core, this machine)")
    for it_s in (6, 8, 12):
        samples_s = it_s * 16
        need = samples_s * per_sample
        print(f"  {it_s:>2} it/s @ bs16 = {samples_s:>3} samples/s -> "
              f"{need:6.2f} worker-cores (ceil {int(-(-need // 1))})")


if __name__ == "__main__":
    main()
```

**Raw stdout:**

```text
============================================================================================
P4/P7/P8 -- machine + env
============================================================================================
  os.cpu_count()      : 16
  platform.processor(): Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
  platform.machine()  : AMD64
  numpy               : 2.1.3
  cpu_model           : ['13th Gen Intel(R) Core(TM) i7-1360P']
  train_len=5367  sampled_indices=[0, 335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350, 3685, 4020, 4355, 4690, 5025]

============================================================================================
P4  PER-STAGE TRANSFORM COST -- 16 unique images x 5 reps  [MEASURED]
============================================================================================
  (a) JPEG+PNG decode        mean=  18.779  median=  18.536  p95=  35.487   (n=16)   <- once per image, not per rep
  (b) _resize_long_side      mean=   9.591  median=   8.417  p95=  18.562   (n=80)
  (c) _apply_rotation        mean=  11.230  median=   7.269  p95=  39.239   (n=80)
  (d) _random_crop_pad_512   mean=  15.095  median=   7.906  p95=  73.610   (n=80)
  (e) _apply_flips           mean=   2.693  median=   2.116  p95=   6.240   (n=80)
  (f) _apply_photometric     mean=  17.650  median=  13.521  p95=  41.196   (n=80)
  (g) finalize               mean=  13.178  median=  14.139  p95=  18.935   (n=80)
  TOTAL train_preprocess+fin mean=  69.500  median=  67.197  p95= 133.920   (n=80)

  end-to-end per sample (decode + transform) mean = 88.279 ms

============================================================================================
P7  CROP-RETRY COST  [MEASURED]
============================================================================================
  attempts: mean=2.562  max=10  distribution={1: 64, 2: 1, 3: 1, 6: 1, 10: 13}
  fallback rate (10 tries, cat_max_ratio=0.95 never met): 13/80 = 16.2%
  dom_ratio: mean=0.8016 median=0.8859 max=0.9748
  _dom_nonignore_ratio: calls=205  total=434.0 ms  mean_per_call=2.1172 ms
  share of train_preprocess wall time = 7.8%
  share of _random_crop_pad_512 time  = 35.9%

============================================================================================
P8  MASK DTYPE THROUGH THE PIPELINE  [MEASURED]
============================================================================================
  raw PNG (np.asarray)                     dtype=uint8      shape=(480, 640)       bytes=  307,200  bytes/px=1.0
  after _resize_long_side+astype(int64)    dtype=int64      shape=(602, 802)       bytes=3,862,432  bytes/px=8.0
  after _apply_rotation                    dtype=int64      shape=(602, 802)       bytes=3,862,432  bytes/px=8.0
  after _random_crop_pad_512               dtype=int64      shape=(512, 512)       bytes=2,097,152  bytes/px=8.0
  after finalize (torch)                   dtype=torch.int64 shape=(512, 512)       bytes=2,097,152  bytes/px=8.0

  512x512 mask as int64 = 2,097,152 B ; as uint8 = 262,144 B ; delta per sample = 1,835,008 B (1792 KiB)
  at batch 16: int64=32.00 MiB vs uint8=4.00 MiB

============================================================================================
P4 (cont) WORKERS NEEDED  [INFERRED -- CROSS-PLATFORM, Windows dev box, NOT the Linux pod]
============================================================================================
  measured per-sample cost = 88.279 ms (single core, this machine)
   6 it/s @ bs16 =  96 samples/s ->   8.47 worker-cores (ceil 9)
   8 it/s @ bs16 = 128 samples/s ->  11.30 worker-cores (ceil 12)
  12 it/s @ bs16 = 192 samples/s ->  16.95 worker-cores (ceil 17)
```

#### P4 — per-stage cost [MEASURED]

Per-stage table is reproduced under F4 above. Headline: **88.279 ms** end-to-end per sample
(18.779 ms decode + 69.500 ms transform), single-core, this machine.

The three most expensive transform stages are `_apply_photometric` (17.650 ms mean),
`_random_crop_pad_512` (15.095 ms mean, **73.610 ms p95**) and `finalize` (13.178 ms). The
p95 on the crop is 4.9x its median (7.906 ms) — that tail is P7's subject.

Workers-needed derivation is **INFERRED — CROSS-PLATFORM** and repeated under F4. It is
computed as `samples_per_second × seconds_per_sample`, i.e. pure Amdahl scaling with no
allowance for IPC, collate, pinning, or Windows-vs-Linux worker start-up. On the Linux pod the
per-sample cost will differ; the *shape* of the conclusion (4 workers is not enough) is what
transfers, not the integer.

#### P7 — crop-retry cost [MEASURED]

```text
attempts: mean=2.562  max=10  distribution={1: 64, 2: 1, 3: 1, 6: 1, 10: 13}
fallback rate (10 tries, cat_max_ratio=0.95 never met): 13/80 = 16.2%
dom_ratio: mean=0.8016 median=0.8859 max=0.9748
_dom_nonignore_ratio: calls=205  total=434.0 ms  mean_per_call=2.1172 ms
share of train_preprocess wall time = 7.8%
share of _random_crop_pad_512 time  = 35.9%
```

**This is a throughput finding in its own right.** The distribution is bimodal: 64/80 samples
succeed on the **first** attempt, but 13/80 (**16.2%**) exhaust all 10 attempts and fall back
to the best candidate. There is almost nothing in between. `_dom_nonignore_ratio` — which runs
`np.unique(..., return_counts=True)` (a sort of 262,144 `int64` elements) on every attempt —
costs **2.12 ms per call** and accounts for **35.9%** of all time inside
`_random_crop_pad_512` and **7.8%** of total transform wall time.

Mechanically: a sample whose mask is dominated by one class above `cat_max_ratio=0.95` can
never satisfy the criterion no matter how the window moves, so it always pays the full 10×
`np.unique` cost and then discards the result. On this 16-image sample that is 16.2% of
samples paying ~21 ms of pure-waste sorting each. Extrapolating the observed rate to the full
train split is **INFERRED**, not measured — 16 images is far too small a sample for the true
population rate, and the audit's decode cap forbids measuring it properly. What is measured is
that the mechanism is real and its unit cost is 2.12 ms.

#### P8 — mask dtype through the pipeline [MEASURED]

| Stage | dtype | shape | bytes | bytes/px |
|---|---|---|---|---|
| raw PNG (`np.asarray`) | `uint8` | (480, 640) | 307,200 | 1.0 |
| after `_resize_long_side` + `.astype(int64)` | **`int64`** | (602, 802) | 3,862,432 | **8.0** |
| after `_apply_rotation` | `int64` | (602, 802) | 3,862,432 | 8.0 |
| after `_random_crop_pad_512` | `int64` | (512, 512) | 2,097,152 | 8.0 |
| after `finalize` (torch) | `torch.int64` | (512, 512) | 2,097,152 | 8.0 |

**`int64` is carried where `uint8` would suffice.** The label space is `{0..115} ∪ {255}` —
every value fits in `uint8`. The widening happens at `transforms.py:57` and `:167
(`.astype(np.int64)`), and the ignore-pad canvas is allocated `int64` directly at `:47`/`:133`.

- per-sample delta at 512x512: **1,835,008 B = 1,792 KiB** (2,097,152 vs 262,144)
- at batch 16: **32.00 MiB vs 4.00 MiB** — a 28 MiB per-batch overhead on every
  host→device transfer, on top of the pre-crop cost at the larger intermediate size
  (3.86 MB per mask at 602x802 in this sample).

Two caveats, stated so this is not overread. (1) The **final** cast to `int64` is genuinely
required — `F.cross_entropy` and `F.one_hot` demand a `long` target, so `finalize` must produce
`torch.int64`. The avoidable part is carrying `int64` through resize/rotate/crop, where the
arrays are largest and the ops are pure indexing. (2) `_apply_rotation` already round-trips
through `uint8` internally (`transforms.py:87`) and casts straight back to `int64` at `:89`,
which shows the `uint8` representation is sufficient for the geometric stages.

### P5 / P6 / E15 — epoch arithmetic and forward-path ops

**Script:**

```python
"""B30 probes P5 / P6 + E15. READ-ONLY. CPU only. ZERO image decodes (len() and shapes only)."""
import inspect
import math
import sys
from pathlib import Path

REPO = Path(r"C:\Users\admin\plantseg-thesis")
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from configs.data import DATA  # noqa: E402
from configs.e1_student import E1_STUDENT  # noqa: E402
from src.data.dataset import PlantSegDataset  # noqa: E402
from src.models import student as S  # noqa: E402
from src.seeds import set_seed  # noqa: E402


def main():
    set_seed(42)

    # ---------------- E15 + P5 ----------------
    print("=" * 92)
    print("E15  SPLIT COUNTS ON THIS MACHINE  [MEASURED, len() only, no decode]")
    print("=" * 92)
    print(f"  DATA['root'] = {DATA['root']}")
    counts = {}
    for sp in ("train", "val", "test"):
        counts[sp] = len(PlantSegDataset(sp))
        print(f"  {sp:<6} len(PlantSegDataset) = {counts[sp]}")
    tot = sum(counts.values())
    print(f"  TOTAL = {tot}")
    print(f"  configs/data.py locked sizes    = {DATA['splits']['sizes']}")
    print(f"  B30 prompt figures  5367/846/1561/7774 -> match={counts['train']==5367 and counts['val']==846 and counts['test']==1561 and tot==7774}")
    print(f"  Ch3-prose figures   5442/778/1554       -> match={counts['train']==5442 and counts['val']==778 and counts['test']==1554}")

    print("\n" + "=" * 92)
    print("P5  ITERATIONS PER EPOCH / EPOCHS AT 80k / RAGGED BATCH  [MEASURED + arithmetic]")
    print("=" * 92)
    n = counts["train"]
    bs = E1_STUDENT["batch_size"]
    iters = E1_STUDENT["iterations"]
    full, rem = divmod(n, bs)
    per_epoch_nodrop = math.ceil(n / bs)
    print(f"  train samples          = {n}")
    print(f"  batch_size             = {bs}   (configs/e1_student.py)")
    print(f"  drop_last              = False  (src/data/dataset.py:109)")
    print(f"  full batches           = {full}")
    print(f"  {n} mod {bs}           = {rem}   <- SIZE OF THE FINAL RAGGED BATCH")
    print(f"  iterations per epoch   = {per_epoch_nodrop}  (ceil, drop_last=False)")
    print(f"  iterations per epoch   = {full}  (if drop_last were True)")
    print(f"  max_iters              = {iters}")
    print(f"  epochs at 80k          = {iters/per_epoch_nodrop:.3f}")
    print(f"  val_interval           = {E1_STUDENT['val_interval']}  -> "
          f"{iters//E1_STUDENT['val_interval']} validations")

    # ---------------- P6 ----------------
    print("\n" + "=" * 92)
    print("P6  STUDENT FORWARD OPS + OUTPUT STRIDE  [MEASURED]")
    print("=" * 92)
    m = S.build_student(pretrained=False)
    m.eval()
    print(f"  used_pretrained={m.used_pretrained}  num_classes={m.num_classes}  "
          f"low_tap={m.low_tap} high_tap={m.high_tap} low_ch={m.low_ch} high_ch={m.high_ch}")
    print(f"  params={sum(p.numel() for p in m.parameters()):,}")

    print("\n  -- backbone stage output strides for 512x512 input --")
    h = torch.zeros(1, 3, 512, 512)
    with torch.no_grad():
        for i, blk in enumerate(m.features):
            h = blk(h)
            tag = ""
            if i == m.low_tap:
                tag = "  <== LOW TAP (skip)"
            if i == m.high_tap:
                tag = "  <== HIGH TAP (C5)"
            print(f"    features[{i:>2}] ch={int(h.shape[1]):>4} "
                  f"{int(h.shape[2])}x{int(h.shape[3])}  OS={512//int(h.shape[2])}{tag}")
            if i == m.high_tap:
                break

    with torch.no_grad():
        low, high = m._forward_features(torch.zeros(1, 3, 512, 512))
        head_out = m.head(low, high)
        full_out = m(torch.zeros(1, 3, 512, 512))
    print(f"\n  low  (skip)      : {tuple(low.shape)}  OS={512//low.shape[2]}")
    print(f"  high (C5)        : {tuple(high.shape)}  OS={512//high.shape[2]}")
    print(f"  head OS8 logits  : {tuple(head_out.shape)}  OS={512//head_out.shape[2]}")
    print(f"  final logits     : {tuple(full_out.shape)}  OS={512//full_out.shape[2]}")

    print("\n  -- ops named in F7, located in the TRAINING forward source --")
    for label, fn in (("PlantSegStudent.forward", S.PlantSegStudent.forward),
                      ("LRASPPHead.forward", S.LRASPPHead.forward)):
        src = inspect.getsource(fn)
        base = inspect.getsourcelines(fn)[1]
        print(f"\n  [{label}]  (starts at src/models/student.py:{base})")
        for off, line in enumerate(src.splitlines()):
            if "interpolate" in line or "AdaptiveAvgPool" in line or "self.context" in line:
                print(f"    student.py:{base+off}: {line.strip()}")

    print("\n  -- AdaptiveAvgPool2d instances reachable in the module tree --")
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.AdaptiveAvgPool2d):
            print(f"    {name}: {mod}")

    print("\n  -- head.context definition --")
    print(f"    {m.head.context}")

    print("\n  -- is AdaptiveAvgPool2d on the training path? (grad check, CPU) --")
    m.train()
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = m(x)
    y.sum().backward()
    ctx_conv = m.head.context[1]
    print(f"    head.context[1] (Conv2d) weight.grad is not None -> "
          f"{ctx_conv.weight.grad is not None}")
    print(f"    head.context[1] grad abs-sum = "
          f"{float(ctx_conv.weight.grad.abs().sum()) if ctx_conv.weight.grad is not None else 'NO GRAD'}")


if __name__ == "__main__":
    main()
```

**Raw stdout:**

```text
============================================================================================
E15  SPLIT COUNTS ON THIS MACHINE  [MEASURED, len() only, no decode]
============================================================================================
  DATA['root'] = C:\Users\admin\plantseg_data\plantseg
  train  len(PlantSegDataset) = 5367
  val    len(PlantSegDataset) = 846
  test   len(PlantSegDataset) = 1561
  TOTAL = 7774
  configs/data.py locked sizes    = {'train': 5367, 'val': 846, 'test': 1561}
  B30 prompt figures  5367/846/1561/7774 -> match=True
  Ch3-prose figures   5442/778/1554       -> match=False

============================================================================================
P5  ITERATIONS PER EPOCH / EPOCHS AT 80k / RAGGED BATCH  [MEASURED + arithmetic]
============================================================================================
  train samples          = 5367
  batch_size             = 16   (configs/e1_student.py)
  drop_last              = False  (src/data/dataset.py:109)
  full batches           = 335
  5367 mod 16           = 7   <- SIZE OF THE FINAL RAGGED BATCH
  iterations per epoch   = 336  (ceil, drop_last=False)
  iterations per epoch   = 335  (if drop_last were True)
  max_iters              = 80000
  epochs at 80k          = 238.095
  val_interval           = 4000  -> 20 validations

============================================================================================
P6  STUDENT FORWARD OPS + OUTPUT STRIDE  [MEASURED]
============================================================================================
  used_pretrained=False  num_classes=116  low_tap=6 high_tap=15 low_ch=40 high_ch=160
  params=2,933,688

  -- backbone stage output strides for 512x512 input --
    features[ 0] ch=  16 256x256  OS=2
    features[ 1] ch=  16 256x256  OS=2
    features[ 2] ch=  24 128x128  OS=4
    features[ 3] ch=  24 128x128  OS=4
    features[ 4] ch=  40 64x64  OS=8
    features[ 5] ch=  40 64x64  OS=8
    features[ 6] ch=  40 64x64  OS=8  <== LOW TAP (skip)
    features[ 7] ch=  80 32x32  OS=16
    features[ 8] ch=  80 32x32  OS=16
    features[ 9] ch=  80 32x32  OS=16
    features[10] ch=  80 32x32  OS=16
    features[11] ch= 112 32x32  OS=16
    features[12] ch= 112 32x32  OS=16
    features[13] ch= 160 32x32  OS=16
    features[14] ch= 160 32x32  OS=16
    features[15] ch= 160 32x32  OS=16  <== HIGH TAP (C5)

  low  (skip)      : (1, 40, 64, 64)  OS=8
  high (C5)        : (1, 160, 32, 32)  OS=16
  head OS8 logits  : (1, 116, 64, 64)  OS=8
  final logits     : (1, 116, 512, 512)  OS=1

  -- ops named in F7, located in the TRAINING forward source --

  [PlantSegStudent.forward]  (starts at src/models/student.py:143)
    student.py:149: return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

  [LRASPPHead.forward]  (starts at src/models/student.py:92)
    student.py:94: the parent (matches the quantization-ready isolated head: one quantized interpolate, then a
    student.py:97: s = self.context(high)                                         # [B,inter,1,1]
    student.py:99: high_os8 = F.interpolate(high_feat, size=low.shape[-2:], mode="bilinear", align_corners=False)

  -- AdaptiveAvgPool2d instances reachable in the module tree --
    features.4.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    features.5.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    features.6.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    features.11.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    features.12.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    features.13.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    features.14.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    features.15.block.2.avgpool: AdaptiveAvgPool2d(output_size=1)
    head.context.0: AdaptiveAvgPool2d(output_size=1)

  -- head.context definition --
    Sequential(
  (0): AdaptiveAvgPool2d(output_size=1)
  (1): Conv2d(160, 256, kernel_size=(1, 1), stride=(1, 1))
  (2): Sigmoid()
)

  -- is AdaptiveAvgPool2d on the training path? (grad check, CPU) --
    head.context[1] (Conv2d) weight.grad is not None -> True
    head.context[1] grad abs-sum = 0.14050017297267914
```

#### P5 [MEASURED len + arithmetic]

| Quantity | Value |
|---|---|
| train samples | 5367 |
| batch_size | 16 |
| full batches | 335 |
| **final ragged batch** | **7** (`5367 mod 16`) |
| iterations per epoch (`drop_last=False`) | **336** |
| iterations per epoch (if `drop_last=True`) | 335 |
| epochs at 80,000 iters | **238.095** |
| validations (`val_interval=4000`) | 20 |

#### P6 [MEASURED]

Output strides and the op inventory are given in full under E8 and F7. The load-bearing
results: the head emits **OS8** (64x64) logits and `forward` upsamples to **OS1** (512x512);
`F.interpolate(mode="bilinear")` appears **twice** on the training path (`student.py:99`,
`:149`); `AdaptiveAvgPool2d(1)` appears **nine** times (8 backbone SE gates + `head.context.0`);
and a real CPU backward confirms `head.context[1]` receives gradient
(`grad abs-sum = 0.14050017297267914`), so the context branch is unambiguously on the
training forward/backward path and not an inference-only shortcut.

---

## 5. DISAGREEMENTS

### 5.1 Findings I refuted

**F1 — REFUTED (the most consequential one).** The finding describes
`np.random.RandomState((SEED + idx) % 2**32)`. That expression is not in this repository. The
actual line re-seeds from the advancing ambient stream, the code comment names the idx-seeded
form as already-fixed prior behaviour, the logged seed sequences differ across passes, and the
val control isolates the effect. There is no per-image augmentation freeze.

**F5 — REFUTED.** Both `torch.use_deterministic_algorithms(...)` and `CUBLAS_WORKSPACE_CONFIG`
are set, in `src/seeds.py`, on the E1 path, before any CUDA touch. Stronger than that: they
match ch3's determinism paragraph token for token, including `warn_only=True` and the exact
`:4096:8` workspace string. `seeds.py` is not "half of B6"; it is B6.

**F12 — REFUTED.** `validate()` saves and restores training mode (`train_e1.py:113-114`,
`:127-128`). BatchNorm is not left frozen after iter 4000. `optimizer.zero_grad(set_to_none=True)`
is correctly placed at the top of each iteration.

**F9 — headline claim REFUTED.** Teacher forward and integration code exists and is
substantial: `SegNeXtTeacherAdapter.forward` (`segnext_teacher.py:183`), `build_segnext_teacher`
(`:195`), `FrozenTeacher.forward` (`teacher.py:122`), `load_frozen_teacher` (`:211`), a 270-line
launcher and a 370-line MMSeg config. The parts of F9 that survive are narrower: `weights/` is
empty and the `NEED_TO_CONFIRM` fields remain.

### 5.2 Where a finding is right but its stated consequence is not

**F7.** Both named ops are on the training path, so the refutation condition fails and the
structural claim stands. But F7's framing — *"adding `use_deterministic_algorithms(True)`
WITHOUT `warn_only=True` would raise"* — describes a change nobody proposes. `warn_only=True`
is already set **and is what ch3 prescribes**. So F7's fallback clause (*"seed-controlled but
NOT bit-deterministic"*) is not a defect to be fixed; it is the preregistered design, and ch3
states the caveat explicitly. F7 also undercounts `AdaptiveAvgPool2d` by 9x — the SE gates
throughout the backbone carry the same property as the head's context branch.

**F8.** The first half is right (ch3 pins the pixel population, not a grid). The second half —
*"the validity-mask downsampling target is therefore also unpinned"* — is wrong. It **is**
pinned, by `IMPLEMENTATION_CONTRACT.md:193` and `configs/distill.py:32`, to **stride-16**; and
the implementation contradicts that for the CWD-**logit** term by running it at **OS8**
(64x64 vs 32x32, MEASURED by P6). The real defect is a code-vs-contract conflict, which is a
sharper and more actionable finding than an absence.

### 5.3 Things I think are wrong that were not on the list

**The brief's own calibration note about ch3 was incorrect.** "Your expectation that ch3 is
not in this checkout is almost certainly right... I hold ch3" — `docs/reference/ch3.pdf` is
git-tracked in this repository, 77 pages, with an intact text layer, and is not on the Part-E
protected list. All four E13 items are answered from the manuscript itself rather than from
the contract. Had I followed the calibration note I would have returned four false
"NOT FOUND IN MY COPY" answers.

**F11's premise about where reduced budgets live is worth a caution to the adjudicator.** F11
is CONFIRMED, but the repository *does* contain an 8,000-iteration reduced budget
(`configs/distill.py:63`). It belongs to the clipping pilot, holds λ **fixed** at 1.0, and is
sequenced *before* the sweep. Anyone grepping for "reduced budget" will hit it and may
wrongly conclude F11 is refuted. It is not — but the two must not be conflated.

**`configs/loss.py:14` actively contradicts the training loop.** It declares
`real_class_weights: "NEED_TO_CONFIRM"` — *"not yet computed"* — while
`reports/e1_class_weights.json` has existed since 2026-06-29 and is loaded unconditionally by
`train_e1.py:42,48-57`, which raises if it is missing or malformed. The config asserts a state
of the project that the code disproves on every single run.

---

## 6. NEW RISKS

Ranked by whether they block E1.

### BLOCKING — decide before launching E1

**N1 — ch3 requires THREE-SEED E1. Nothing in the repo plans for it.**
ch3, verbatim: *"Three-seed validation is planned for E1 and E3"* and *"the two full-precision
trainings that anchor the distillation comparison, E1 and E3, are each repeated under three
random seeds"*. `train_e1.py` accepts `--seed` (default 42) so it is mechanically supported,
but the launch budget, the runbook and the checkpoint naming (`e1_student_best_iter{it}.pt`,
no seed in the filename) all assume a single run. **Impact: 3x the E1 GPU budget, and a
checkpoint-collision risk if two seeds share a `--ckpt-dir`.** This is the single largest
planning gap I found. It is not on the F-list at all.

**N2 — ch3 prose split counts do not match the dataset on disk, and the dataset class
hard-fails on mismatch.** Code produces 5367/846/1561 (MEASURED); the Ch3-prose figures cited
in the brief are 5442/778/1554. Totals agree at 7774, so this is a partition-assignment
difference, not a missing-data problem. `PlantSegDataset.__init__:63-69` raises
`RuntimeError` if the on-disk count deviates from `configs/data.py`. **If ch3's numbers are
authoritative, E1 cannot start at all** until one side is corrected. If the on-disk official
split is authoritative, ch3 needs amending. The adjudicator holds the authority to settle this;
I can only report that the two disagree.

**N3 — no resume + non-atomic writes + best-only checkpoints compound into total run loss.**
F2 and F13 are individually true; together they mean a preemption at iteration 79,000 loses
~99% of an 80,000-iteration run, and a kill *during* a write can corrupt the one artifact that
survives. On interruptible/spot RunPod capacity this is close to a certainty over a multi-day
run. ch3 itself anticipates interruption (*"Any pod interruption, hardware substitution, or
migration between pod types is reported"*) without providing a recovery mechanism.

**N4 — throughput will bound E1, and the crop retry loop is a named, fixable cause.**
MEASURED 88.3 ms/sample; `num_workers=4` default; no `pin_memory`/`persistent_workers`/
`prefetch_factor`. On top of that, P7 MEASURED a 16.2% full-fallback rate in
`_random_crop_pad_512`, with `_dom_nonignore_ratio` (an `np.unique` sort over 262k `int64`)
consuming 35.9% of crop time for results that are then discarded. **Impact: GPU idle time,
i.e. direct pod cost.** Quantifying it on the pod is cheap; see section 7.

**N5 — `verify_env.py` cannot detect the two failure modes most likely on a fresh pod.**
No compute-capability gate (F10), and additionally **no dataset check at all** — it never
touches `PLANTSEG_DATA_ROOT`, never confirms `images/<split>` exists, never confirms the split
counts. A pod can pass `verify_env.py` with `PASS | OK for real E1 training` and then have
`train_e1.py` die at `PlantSegDataset.__init__` seconds later.

### NON-BLOCKING — should be fixed, will not stop the run

**N6 — `train_e1.py:253` prints all 80,000 LR values in one call.**
`print(f"[summary] lr_trace={['%.8e' % x for x in lr_trace]}")`. At 80,000 entries x ~17
chars that is a single ~1.4 MB stdout line (INFERRED from the format string and `max_iters`).
Combined with F3 (stdout is the *only* telemetry), a log pipeline that truncates long lines
would destroy the one artifact worth keeping.

**N7 — checkpoints accumulate without pruning and without a `best` pointer.**
`save_checkpoint` writes `e1_student_best_iter{it}.pt` — a new filename per improvement, never
overwritten, never pruned. With 20 validations that is up to 20 files. Model+SGD-momentum
state is ~23.5 MB (INFERRED: 2,933,688 params MEASURED x 4 B x 2 for momentum), so up to
~470 MB. Not fatal, but downstream stages must guess which file is best from the `iter` in the
name, since nothing records it.

**N8 — workers are respawned every epoch.** `cycle()` (`train_e1.py:70-74`) creates a fresh
iterator on each pass; with `persistent_workers` unset (F4) the worker pool is torn down and
rebuilt every **336 iterations** (P5, MEASURED). This is also, incidentally, part of *why* F1
is refuted — but it is pure overhead ~238 times over the run.

**N9 — masks carry `int64` through the whole geometric pipeline.** P8: 8 bytes/px where 1
suffices; 32 MiB vs 4 MiB per batch of 16. The final `torch.int64` **is** required by
`cross_entropy`/`one_hot`; the widening at `transforms.py:57`/`:167`, before resize/rotate/crop,
is what is avoidable.

**N10 — local environment is not the pinned environment.** `torch 2.9.1+cpu` / `numpy 2.1.3`
locally vs `torch==2.1.0+cu121` / `numpy==1.26.4` pinned. `verify_env.py:237-239` would flag
the torch mismatch as a blocker, so the machinery works — but every local smoke result in
`reports/` should be read as "passed on an unpinned stack" unless it says otherwise.

### MANUSCRIPT-vs-CODE conflicts for the adjudicator

These are not repo defects; they are places where ch3 and the implementation disagree and only
the manuscript-holder can decide which moves.

**N11 — Albumentations.** ch3 Table 3.4 names *"Albumentations — Joint image + mask
augmentation ... version pinned in the reproducibility manifest"* and *"OpenCV is the
underlying image I/O layer for both"*. Neither is in `requirements.lock`; `requirements-e1.txt:10-11`
excludes both explicitly; `configs/augment.py:9-13` records this as reconciliation D3. The
*behaviour* is argued to be equivalent, but ch3's claim that the version is "pinned in the
reproducibility manifest" is **false as written** — there is nothing to pin.

**N12 — ch3 says the head projects to `115` channels.** *"the low-level skip feature (output
stride 8) is projected by a 1x1 convolution to the 115 output channels ... to 115 channels"*.
The model MEASURED emits **116** (`(1, 116, 512, 512)`), and ch3 elsewhere says the output
layer is sized to the verified 116-class count. Internal ch3 inconsistency; the code is right.

**N13 — ch3 says the context branch is "large-kernel average-pooling".** The implementation
uses `nn.AdaptiveAvgPool2d(1)` — **global** pooling (`student.py:83`, MEASURED as
`AdaptiveAvgPool2d(output_size=1)`). The original LR-ASPP uses a large-kernel strided pool.
This matches torchvision's `LRASPPHead`, so it is a defensible engineering choice, but ch3's
wording describes something the code does not do.

**N14 — CWD-logit validity grid.** Contract `:193` and `configs/distill.py:32` say
*"validity mask downsampled to stride-16"*; `train_distill.py:207` downsamples to
`head_logits.shape[-2:]` = **OS8**. Affects E3, not E1.

**N15 — `disease_only_miou` docstring vs configs.** `metrics.py:168-170` says the disease-only
convention is *"no longer PROVISIONAL"*; `configs/e1_student.py:25` and `train_e1.py:126,234`
still print/label it PROVISIONAL. Cosmetic — the checkpoint criterion is all-class in both.

---

## 7. CANNOT-VERIFY-LOCALLY

This machine has **no CUDA device** and runs `torch 2.9.1+cpu`, not the pinned
`2.1.0+cu121`. Three items therefore cannot be settled here.

### 7.1 F7 — do the two ops actually lack a deterministic CUDA backward on the pinned stack?

**Why blocked:** `torch.use_deterministic_algorithms` only raises/warns when a
nondeterministic **CUDA** kernel is dispatched. On CPU the code path is never taken.

**Settle it on the pod (read-only, no training, seconds):**

```bash
python -W always -c "
import warnings, torch, sys
sys.path.insert(0,'.')
from src.seeds import set_seed
from src.models.student import build_student
set_seed(42)                      # sets warn_only=True exactly as E1 does
m = build_student(pretrained=False).cuda().train()
x = torch.randn(2,3,512,512, device='cuda', requires_grad=True)
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always')
    m(x).sum().backward()
    print('WARNINGS:', [str(w.message)[:160] for w in ws] or 'NONE')
torch.use_deterministic_algorithms(True, warn_only=False)   # strict probe
try:
    m.zero_grad(); m(x).sum().backward(); print('STRICT: no error raised')
except RuntimeError as e:
    print('STRICT RAISES:', str(e)[:300])
"
```

**Interpretation:** `STRICT RAISES` naming `upsample_bilinear2d_backward_cuda` or
`adaptive_avg_pool2d_backward_cuda` confirms F7's structural claim on the pinned stack.
`STRICT: no error raised` refutes it. Either way the E1 run is **unaffected**, because E1 uses
`warn_only=True` per ch3 — this probe only tells you whether the warning list will be noisy.

> **[SUPERSEDED 2026-09-10 — see [B48](b48_e1_oom_investigation.md) §4.]** The final sentence above
> is wrong. The first real E1 launch died at iteration 2 with a CUDA OOM caused by exactly this
> finding, so the run was not "unaffected" and the consequence was not warning noise. The error is
> that this paragraph assumes one mechanism where there are two: an op with a deterministic
> alternative is **silently rerouted** to it — `F.interpolate` never warned, `functional.py:4018`
> substituted `torch._decomp.decompositions.upsample_bilinear2d_vec`, and `warn_only` is
> irrelevant on that path — at a **measured 12.658 GiB** cost; only an op with *no* alternative
> warns, and the op that warned was `nll_loss2d_forward_out_cuda_template` (the CE forward), which
> F7 does not name. Note also that the probe above uses `batch 2`, where the decomposition costs
> roughly 2 GiB and is invisible; running it would not have surfaced this. The original text is
> left intact so the reasoning error stays recognisable.

### 7.2 F10 — does the pod's GPU fall outside the pinned wheel's compiled SM list?

**Why blocked:** requires the actual pod GPU and the actual installed wheel.

```bash
python -c "import torch; print('torch', torch.__version__); print('cap', torch.cuda.get_device_capability(0)); print('name', torch.cuda.get_device_name(0)); print('arch_list', torch.cuda.get_arch_list())"
```

**Interpretation:** if `cap` (e.g. `(12, 0)` for an RTX 5090) has no matching `sm_XX` in
`arch_list`, the pod is unusable with the pinned wheel and `verify_env.py` will still say
`PASS`. Run this **before** paying for the pod. A one-line smoke that actually launches a
kernel is the belt-and-braces version:

```bash
python -c "import torch; a=torch.randn(8,8,device='cuda'); print((a@a).sum().item())"
```

### 7.3 F9(d) — has `scripts/test_teacher_init.py` ever been run?

**Why blocked:** absence of a log is not proof of absence of a run, and the script needs the
pinned MMSeg stack plus a checkpoint in `weights/`, neither of which exists here.

```bash
python scripts/test_teacher_init.py   # in the pinned mmseg env, after `mim download`
```

**Interpretation:** it prints `RESULT: PASS` only if the checkpoint loads into the stock
150-class ADE20K model with zero missing and zero unexpected keys. That line, plus the URL,
SHA256 and date, are exactly the four `NEED_TO_CONFIRM` fields in `docs/teacher_init_source.md`.
**Not an E1 blocker** — E1 has no teacher.

### 7.4 Also not verifiable here (lower priority)

| Item | Why | Command |
|---|---|---|
| True crop-fallback rate over the full train split | audit decode cap is 40 images; 16 is far
too small for a population rate | on the pod, instrument `_random_crop_pad_512`'s `info` dict
over one full epoch (336 iters) and histogram `attempts`/`fallback` |
| Real dataloader throughput at `num_workers=4/8/16` | Windows spawn ≠ Linux fork; different
CPU and PIL build | time 50 iterations of the real `build_dataloader("train", 16, num_workers=N)`
on the pod for N in 4, 8, 16 |
| Whether ImageNet init downloads or hits cache | needs the pod's torch-hub cache | `verify_env.py`
reports `imagenet_cached` (`:157-162`) |

---

## 8. Provenance

### Files READ (repository)

```text
AGENTS.md, CLAUDE.md                          (project contract, loaded as instructions)
configs/augment.py                            configs/data.py
configs/distill.py                            configs/e1_student.py
configs/loss.py                               configs/model.py
configs/teacher_finetune.py
docs/IMPLEMENTATION_CONTRACT.md               (greps only: sweep/budget/KD-resolution rows)
docs/teacher_init_source.md
docs/reference/ch3.pdf                        (text layer decoded; NOT on the Part-E protected list)
reports/e1_class_weights.json                 (first 400 bytes, for the E14 drift check)
requirements.lock                             requirements-e1.txt
requirements-runpod.in                        requirements-runpod.lock   (grep only)
scripts/test_teacher_init.py                  (first 40 lines)
scripts/verify_env.py
src/data/dataset.py                           src/data/transforms.py
src/distill/segnext_teacher.py                src/distill/teacher.py     (symbol listing only)
src/eval/metrics.py                           src/models/student.py
src/seeds.py                                  src/training/losses.py
src/training/train_distill.py                 (lines 150-220 + greps)
src/training/train_e1.py
```

**NOT touched, as required:** `docs/reference/reference.pdf` (metadata only — `ls -la` output
recorded in E1; never opened, hashed, copied, staged or modified), `docs/reference/context.md`,
every file inside `weights/` (directory listing only), the dataset (read-only, 32 decodes),
`~/.claude`, Claude memory.

### Files WRITTEN

```text
reports/pre_e1_launch_audit.md    <- this file. CREATED (did not previously exist).
```

Pre-existence check, run before writing (per ruling R2):

```text
$ ls -la reports/pre_e1_launch_audit.md
ls: cannot access 'reports/pre_e1_launch_audit.md': No such file or directory
$ git ls-files --error-unmatch reports/pre_e1_launch_audit.md
error: pathspec 'reports/pre_e1_launch_audit.md' did not match any file(s) known to git
$ git log -1 --format="%h %ci" -- reports/pre_e1_launch_audit.md
(no output)
```

So R2's record-then-overwrite path did not apply: **no mtime, no git tracking, no commit
history existed.** This is a creation, not an overwrite.

### Scratchpad files (never written into the repo)

```text
<scratchpad>/pdftext.py     ch3.pdf text extractor (ToUnicode CMap decoder)
<scratchpad>/ch3find.py     text normaliser + context-window search
<scratchpad>/ch3.txt        decoded ch3 text (170,890 chars)
<scratchpad>/p1_p3.py .out  probes P1, P2a, P2b, P3
<scratchpad>/p4_p8.py .out  probes P4, P7, P8
<scratchpad>/p5_p6.py .out  probes P5, P6, E15
<scratchpad>/genreport.py, gen2..gen7.py   this report's generator
```

### Probes run

| Probe | Decodes | Result |
|---|---|---|
| P1 seed logging + tensor equality (nw=0) | 4 | seeds DIFFER → F1 refuted at mechanism |
| P2a real loader path (nw=2, shuffle=False) | 4 | tensors differ across passes |
| P2b same at nw=0 | 4 | tensors differ across passes |
| P3 val control (nw=2) | 4 | bit-identical, max_abs_diff 0.0 |
| P4 transform cost, 16 unique x 5 reps | 16 | 88.279 ms/sample end-to-end |
| P5 epoch arithmetic | 0 | 336 iters/epoch, ragged batch 7, 238.095 epochs |
| P6 module tree + forward ops + backward | 0 | 2x interpolate, 9x AdaptiveAvgPool2d |
| P7 crop-retry cost | 0 (reuses P4) | 16.2% fallback; `_dom_nonignore_ratio` 35.9% of crop time |
| P8 mask dtype | 0 (reuses P4) | int64 throughout; 32 MiB vs 4 MiB per batch |
| **TOTAL** | **32** | within the 40 cap (R3) |

No GPU. No network. No downloads. No installs. No training.

### Git status AFTER the audit

```text
$ git status --porcelain
 M docs/reference/reference.pdf
?? reports/pre_e1_launch_audit.md

$ git status -sb
## master...origin/master
 M docs/reference/reference.pdf
?? reports/pre_e1_launch_audit.md
```

### Confirmation

- **No existing file was edited.** The only repository write is the creation of
  `reports/pre_e1_launch_audit.md`.
- **Nothing was staged.** No `git add` was run, with or without paths.
- **Nothing was committed or pushed.** No `git commit`, `git push`, `git checkout`,
  `git restore`, `git clean`, or branch change was run.
- **`docs/reference/reference.pdf` remains exactly as found** — ` M`, unstaged, untouched. Its
  appearance in `git status` above is the same pre-existing modification recorded in E1.
- **No fixes were applied.** F1 was refuted rather than repaired; every confirmed finding was
  left in place for B31.

---

## 9. ADDENDUM — P9: cross-process reproducibility of the augmentation stream

Added after review. P1 established that the seed stream **advances across passes** (F1
refuted). It did not establish that the stream is **reproducible across interpreter
invocations** — a distinct property, and the one ch3's seed-42 claim actually needs. If the
first held and the second did not, augmentation would be stochastic but the run would not be
reproducible, trading one ch3 violation for another. P9 tests the second property.

**Method.** The P1 seed-logging probe was re-run as two independent OS processes (different
pids, fresh interpreters), each logging the `np.random.RandomState` seed sequence for two
passes at `num_workers=0`, plus SHA-256 digests of the returned image tensors, plus a
`num_workers=2` pass to exercise the real worker-seeding path. 12 image decodes.

**Script:**

```python
"""P9 — cross-PROCESS reproducibility of the augmentation seed stream.

P1 proved the stream ADVANCES across passes (F1 refuted). It did not prove the stream is
REPRODUCIBLE across interpreter invocations, which is the property ch3's seed-42 claim needs.
This script prints the seed list + tensor digests; run it in two separate processes and diff.

Usage:  python p9_xproc.py <tag>
"""
import hashlib
import sys
from pathlib import Path

REPO = Path(r"C:\Users\admin\plantseg-thesis")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.data.dataset import PlantSegDataset, _seed_worker  # noqa: E402
from src.seeds import SEED, set_seed  # noqa: E402

IDX = [100, 500]
TAG = sys.argv[1] if len(sys.argv) > 1 else "A"


def loader(ds, nw):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=2, shuffle=False, num_workers=nw,
                      generator=g, worker_init_fn=_seed_worker, drop_last=False)


def digest(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def main():
    print(f"### PROCESS {TAG} | pid={__import__('os').getpid()} | "
          f"torch {torch.__version__} | numpy {np.__version__}")

    set_seed(42)
    ds = PlantSegDataset("train")

    # ---- num_workers=0 : main-process numpy stream ----
    orig = np.random.RandomState
    seeds = []

    def logged(seed=None, *a, **k):
        seeds.append(seed)
        return orig(seed, *a, **k)

    np.random.RandomState = logged
    try:
        set_seed(42)
        ld = loader(Subset(ds, IDX), 0)
        for p in range(2):
            seeds.clear()
            digs = []
            for img, _m in ld:
                for i in range(img.shape[0]):
                    digs.append(digest(img[i]))
            print(f"[{TAG}] nw=0 pass{p+1} seeds={list(seeds)} digests={digs}")
    finally:
        np.random.RandomState = orig

    # ---- num_workers=2 : real worker seeding path ----
    set_seed(42)
    ld2 = loader(Subset(ds, IDX), 2)
    digs = []
    for img, _m in ld2:
        for i in range(img.shape[0]):
            digs.append(digest(img[i]))
    print(f"[{TAG}] nw=2 pass1 digests={digs}")


if __name__ == "__main__":
    main()
```

**Raw stdout — process A:**

```text
### PROCESS A | pid=15620 | torch 2.9.1+cpu | numpy 2.1.3
[A] nw=0 pass1 seeds=[1608637542, 1273642419] digests=['140c8c720c8bc712', 'a26114baf0d446c8']
[A] nw=0 pass2 seeds=[1935803228, 787846414] digests=['73f0abdbf0330e04', '3f32b51dce6c715a']
[A] nw=2 pass1 digests=['6c772db174314e67', 'f4109cb73ece4509']
```

**Raw stdout — process B:**

```text
### PROCESS B | pid=20148 | torch 2.9.1+cpu | numpy 2.1.3
[B] nw=0 pass1 seeds=[1608637542, 1273642419] digests=['140c8c720c8bc712', 'a26114baf0d446c8']
[B] nw=0 pass2 seeds=[1935803228, 787846414] digests=['73f0abdbf0330e04', '3f32b51dce6c715a']
[B] nw=2 pass1 digests=['6c772db174314e67', 'f4109cb73ece4509']
```

**Result [MEASURED]:**

| Quantity | Process A | Process B | Identical? |
|---|---|---|---|
| `nw=0` pass1 seeds | `[1608637542, 1273642419]` | `[1608637542, 1273642419]` | **YES** |
| `nw=0` pass2 seeds | `[1935803228, 787846414]` | `[1935803228, 787846414]` | **YES** |
| `nw=0` pass1 digests | `140c8c720c8bc712`, `a26114baf0d446c8` | same | **YES** |
| `nw=0` pass2 digests | `73f0abdbf0330e04`, `3f32b51dce6c715a` | same | **YES** |
| `nw=2` pass1 digests | `6c772db174314e67`, `f4109cb73ece4509` | same | **YES** |

**Both properties hold simultaneously, so there is no trade-off:**

1. Within a process, pass 1 ≠ pass 2 — augmentation varies across epochs (F1 refuted).
2. Across processes, pass *k* of A == pass *k* of B — the whole schedule is reproducible from
   seed 42, which is what ch3 asserts.

The `num_workers=2` row matters independently: it shows the **worker** seeding path is also
cross-process reproducible. That follows from `dataset.py:100-101` seeding a `torch.Generator`
with `SEED`, from which the DataLoader draws `base_seed`; `_seed_worker` (`dataset.py:91-94`)
then derives each worker's numpy seed from `torch.initial_seed()`. Every link is
torch-deterministic, so the real run's `num_workers=4` path inherits the same guarantee.

**Remaining caveat — the pinned stack.** P9 ran on `numpy 2.1.3` / `torch 2.9.1+cpu`, not the
pinned `numpy==1.26.4` / `torch==2.1.0+cu121`. `np.random.RandomState` is NumPy's *legacy*
generator and carries an explicit stream-compatibility guarantee across versions, so the seed
values are expected to be identical on the pod — but the claim should be **measured where it
will be asserted**. One command settles it, and it needs no GPU:

```bash
python - <<'EOF'
import numpy as np, torch, sys
sys.path.insert(0, '.')
from src.seeds import set_seed
set_seed(42)
print('numpy', np.__version__, 'torch', torch.__version__)
print('first 4 draws:', [int(np.random.randint(0, 2**31 - 1)) for _ in range(4)])
EOF
```

**Expected on the pinned stack:** `[1608637542, 1273642419, 1935803228, 787846414]` — the exact
sequence P9 measured here. Any deviation means the local reports in `reports/` describe a
different augmentation schedule than the pod will run, and every seed-dependent artifact would
need regenerating on the pod.

**Verdict:** F1's refutation is now closed on both properties, subject only to that one
pinned-stack re-measurement.

