# B36 — E1 checkpoint resume-identity drill

**Type:** empirical drill, CPU only. **Status:** COMPLETE. **Date:** 2026-09-05.
**Precondition:** B34c committed at `281d765`, governed paths verified clean before starting.

**Not committed.** `src/**`, `configs/**`, and every existing script are unmodified — one new script
(`scripts/smoke_resume_identity.py`) and this report are the only additions. No GPU, no downloads.
All checkpoints and telemetry were written to `tempfile.mkdtemp()` directories **outside the
repository**; `train_e1.py` hard-guards in-repo `--ckpt-dir` and that guard was honored, not
circumvented. `docs/reference/reference.pdf` untouched.

---

## 1. Method, and why N = 20

Four runs, four temp dirs, all `--dry-run --batch-size 2 --num-workers 0 --seed 42 --val-interval 10
--max-val-batches 1`:

| run | invocation | purpose |
|---|---|---|
| **P** | `--max-iters 10 --ckpt-interval 10` | produces the N/2 resume point (`last.pt`) |
| **A** | `--max-iters 20 --ckpt-interval 20` | uninterrupted reference |
| **B** | `--resume P/last.pt --max-iters 20` | the resumed run |
| **B2** | identical to B, separate dir | resume-to-resume determinism (§6) |

### N justification

Measured on this box before choosing N: **3.10 s/iter** steady-state (iter 1 = 4.02 s including
warm-up), 0.89 s per validation batch, ~12 s fixed overhead. Four runs ≈ 3.5 min.

The binding argument is not cost, it is that **a larger N adds no evidence**. At `batch_size 2` over
5,367 train samples with `drop_last=True`, one epoch is **2,683 iterations**, so N=20 sits deep inside
epoch 1 and no epoch boundary is ever crossed. Divergence nonetheless appears at the **first
post-resume iteration** (§4), because the mechanism is a *fresh* `cycle()` drawing a *new*
permutation — not an epoch rollover. N=200 would reproduce the same mechanism at ten times the cost.

### Confounds controlled

- **`--val-interval 10` on every run.** Validation fires at iteration 10 in both P and A. The
  validation block precedes the checkpoint block in the loop
  ([train_e1.py:437](../src/training/train_e1.py#L437) vs
  [:466](../src/training/train_e1.py#L466)), so P's RNG state saved at iteration 10 already includes
  validation's effect and matches A's state there. A mismatched interval would desynchronise the
  streams and produce a divergence that would then be **misattributed to data order**.
- **`--num-workers 0`.** With workers, `_seed_worker` reseeds from `torch.initial_seed()` per worker
  per epoch ([dataset.py:100-103](../src/data/dataset.py#L100)); the restored main-process state
  would not reproduce worker streams.

### Premise: is P@10 the same state as A@10?

Required, because run B inherits P's trajectory rather than A's. Verified in the **strong form**:
at `--val-interval 10` both P and A validate at iteration 10 and therefore both write
`e1_student_best_iter10.pt`, so the comparison is an **ELEMENTWISE `state_dict` comparison of real
tensors** (plus optimizer and scheduler) — not a comparison of losses.

```
PREMISE (strong): P@10 vs A@10 — do the first 10 iterations coincide?
  MODEL   tensors=314  differing=0  global_max_abs_dev=0.000000e+00
  OPTIM   momentum_buffers=176  max_abs_dev=0.000000e+00  param_groups_equal=True  missing_keys=0
  -> IDENTICAL

  premise check used: ELEMENTWISE state_dict comparison
```

Full-precision JSONL losses for iterations 1–10 agree as a secondary cross-check (`all equal = True`).
**The premise is PROVEN, not merely evidenced** — the fallback loss-only path (which would have been
marked EVIDENCED-NOT-PROVEN) was not needed.

---

## 2. Comparison table

### A (uninterrupted, 20 iters) vs B (resumed at 10, run to 20)

```
  MODEL   tensors=314  differing=268  global_max_abs_dev=1.407546e+00
  tensor                                              shape                 max|dev|
    head.high_proj.1.running_var                      (256,)                1.407546e+00
    features.1.block.0.0.weight                       (16, 1, 3, 3)         2.925178e-01
    features.2.block.1.0.weight                       (64, 1, 3, 3)         1.593720e-01
    features.6.block.0.1.running_var                  (120,)                1.493078e-01
    features.0.0.weight                               (16, 3, 3, 3)         1.407492e-01
    head.low_logits.weight                            (116, 40, 1, 1)       1.225046e-01
    features.1.block.1.1.running_var                  (16,)                 1.146140e-01
    features.3.block.1.0.weight                       (72, 1, 3, 3)         9.979707e-02

  OPTIM   momentum_buffers=176  max_abs_dev=2.290673e+00  param_groups_equal=True  missing_keys=0

  SCHED
    _get_lr_called_within_step  False   ==  False
    _is_initial                 False   ==  False
    _last_lr   [0.009997749971872423]   ==  [0.009997749971872423]
    _step_count                    21   ==  21
    base_lrs                   [0.01]   ==  [0.01]
    last_epoch                     20   ==  20
    power                         0.9   ==  0.9
    total_iters                 80000   ==  80000
```

| quantity | A vs B | B vs B2 |
|---|---|---|
| model tensors compared | 314 | 314 |
| model tensors differing | **268** | **0** |
| model global max abs deviation | **1.407546e+00** | **0.000000e+00** |
| optimizer momentum buffers | 176 | 176 |
| optimizer max abs deviation | **2.290673e+00** | **0.000000e+00** |
| optimizer param-group scalars equal | yes | yes |
| scheduler keys (8/8) equal | **yes** | yes |
| first iteration with differing loss | **11** | none |

The largest model deviation sits in a BatchNorm `running_var`, which is expected: BN running
statistics integrate over whatever samples the batch contained, so they register a different sample
order most directly.

**The scheduler is bit-identical across the resume.** Every one of the eight scheduler keys matches,
and the LR is equal at every compared iteration:

```
  iter  A lr                    B lr                    equal
  11    0.009998762491491758    0.009998762491491758    True
  12    0.009998649989874443    0.009998649989874443    True
  13    0.00999853748811648     0.00999853748811648     True
  20    0.009997749971872423    0.009997749971872423    True
```

---

## 3. VERDICT

> # **NON-IDENTICAL**
>
> Global max absolute deviation **1.407546e+00** across 268 of 314 model tensors; optimizer momentum
> deviates by up to **2.290673e+00**. Divergence begins at **iteration 11** — the first post-resume
> iteration.

This confirms rather than discovers: [train_e1.py:355-360](../src/training/train_e1.py#L355) already
warns that *"data-order continuity is NOT restored"* and that *"A resumed run is NOT bitwise-identical
to an uninterrupted one."* The drill's contribution is §4–§6, not this line.

---

## 4. Attribution inventory

Every candidate mechanism, with the code that does or does not restore it.

| # | mechanism | status | evidence |
|---|---|---|---|
| 1 | model weights | **RESTORED** | [`:342`](../src/training/train_e1.py#L342) `student.load_state_dict(ck["model_state_dict"])` |
| 2 | optimizer state (SGD momentum) | **RESTORED** | [`:343`](../src/training/train_e1.py#L343) `optimizer.load_state_dict(...)`; drill: 176 buffers present in both, `missing_keys=0` |
| 3 | scheduler position | **RESTORED** | [`:344`](../src/training/train_e1.py#L344) `scheduler.load_state_dict(...)`; drill: all 8 keys equal, `last_epoch=20` |
| 4 | python `random` | **RESTORED** | [`:126`](../src/training/train_e1.py#L126) `random.setstate(state["python"])` |
| 5 | numpy global RNG | **RESTORED** | [`:127`](../src/training/train_e1.py#L127) `np.random.set_state(state["numpy"])` |
| 6 | torch global RNG | **RESTORED** | [`:128`](../src/training/train_e1.py#L128) `torch.set_rng_state(state["torch"])` |
| 7 | torch.cuda RNG | **RESTORED** (no-op on CPU) | [`:129-130`](../src/training/train_e1.py#L129) |
| 8 | **DataLoader generator** | **RESTORED** | [`:131`](../src/training/train_e1.py#L131) `train_loader.generator.set_state(state["loader_generator"])` |
| 9 | augmentation RNG | **RESTORED** (derived) | [dataset.py:89](../src/data/dataset.py#L89) draws `np.random.RandomState(np.random.randint(...))` from the global numpy stream, which #5 restores |
| 10 | prev-segment LR (boundary check) | **RESTORED** | [`:351`](../src/training/train_e1.py#L351) `prev_lr = ck.get("prev_lr")` |
| 11 | **position within the epoch's permutation** | **NOT RESTORED** | see below |
| 12 | criterion / CE class weights | **not checkpointed — contributes nothing** | see below |

### Why #8 being RESTORED does **not** restore #11

This is the whole finding, and it is counter-intuitive: **a restored generator does not imply a
restored stream position.**

`DataLoader(shuffle=True)` draws a **fresh permutation from `generator` on every `__iter__` call**.
`cycle()` holds **one** iterator alive for the entire run:

```python
def cycle(loader):
    """Infinite iterator over a DataLoader -> iteration-based (not epoch-based) training."""
    while True:
        for batch in loader:
            yield batch
```
— [train_e1.py:77-82](../src/training/train_e1.py#L77)

An uninterrupted run calls `__iter__` once at iteration 1 and stays inside permutation #1 for 2,683
iterations. A resumed run builds a **new** `DataLoader` and a **new** `cycle()` at
[`:373`](../src/training/train_e1.py#L373) — created unconditionally, *after* the resume block — so
its first `next()` triggers a fresh `__iter__` and draws permutation **#2**, from the correctly
restored generator state. The generator is in the right place in the stream; the *iterator* is not.

Nothing persists the position. Proof of absence — command and its actual result:

```
$ grep -n "epoch\|train_iter\|cycle(" src/training/train_e1.py
77:def cycle(loader):
78:    """Infinite iterator over a DataLoader -> iteration-based (not epoch-based) training."""
357:              "an infinite `cycle(train_loader)`; the position within the current epoch is not "
373:    train_iter = cycle(train_loader)
378:        img, mask = next(train_iter)
```

Five matches: a definition, its docstring, the warning text, the unconditional re-creation, and the
consumption. **No save, no restore, no offset.**

Demonstrated directly, without training anything:

```
  consumption model reproduces the checkpoint's saved generator state: True
  train samples=5367  batch_size=2  one epoch = 2683 iterations
  iteration 11 is deep inside epoch 1 -- no epoch boundary is reached.

  run A  iteration 11 batch = perm1[20:22] = [1027, 3140]
  run B  iteration 11 batch = perm2[0:2]  = [152, 4785]
  same batch: False
```

> **Modelling correction, recorded because it changed a stated conclusion.** The first version of this
> drill replayed the stream as `randperm` alone and reported that B's permutation was *not* A's
> next-epoch permutation. That was wrong. `_BaseDataLoaderIter.__init__` draws a `base_seed` from
> `loader.generator` **before** the sampler's permutation, and does so even at `num_workers=0`.
> Verified against the checkpoint's own saved state:
>
> ```
> model A (randperm only)              matches restored: False
> model B (base_seed draw + randperm)  matches restored: True
> ```
>
> The script now **asserts the consumption model against the saved state before drawing any
> conclusion** and prints nothing if the model fails to reproduce it.

### Why #12 contributes nothing

B34 established the criterion is not checkpointed. Confirmed — command and actual result:

```
$ grep -n "criterion" src/training/train_e1.py
294:    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=IGNORE_INDEX).to(dev)
296:          f"weight_on={criterion.ce.weight.device}")
389:        ce = criterion.ce(logits, mask)
390:        dice = criterion.dice(logits, mask)
```

Construction and use only; it never appears in a save path, and a filtered grep for
`criterion|GradScaler|amp` intersected with `save|state_dict` returned no lines at all. This is
**harmless**: the criterion carries no learned state, and its weight buffer is reloaded
deterministically from `reports/e1_class_weights.json` on every run
([`:293-294`](../src/training/train_e1.py#L293)) — the artifact whose content B34b's gate stage now
asserts. There is no AMP `GradScaler` in this path to lose.

### Inventory of what the checkpoint omits

`save_last` persists `iter`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`,
`scheduler`, `best_val_miou_all_class`, `best_ckpt`, `num_classes`, `rng_state`, `prev_lr`
([`:147-169`](../src/training/train_e1.py#L147)). Omitted: **the DataLoader iterator position** (the
only consequential omission), the criterion (stateless, reloaded), and any AMP scaler (not used).

---

## 5. BOUNDED vs UNBOUNDED

> # **BOUNDED**

Every failure mode that would make it unbounded was tested and ruled out:

| unbounded failure mode | ruled out by |
|---|---|
| scheduler restarting from zero | all 8 scheduler keys equal; `last_epoch=20`, `_step_count=21` in both; LR identical at iterations 11, 12, 13, 20 |
| optimizer momentum lost | `param_groups_equal=True`, `missing_keys=0`, 176 momentum buffers present in both |
| augmentation collapsing to a fixed seed | augmentation draws from the global numpy stream ([dataset.py:89](../src/data/dataset.py#L89)), which is restored ([`:127`](../src/training/train_e1.py#L127)); B2 reproduces B exactly, so the augmentation stream is live and deterministic, not frozen |
| sampling from a different distribution | **`B's permutation == A's epoch-2 permutation: True`** |

That last line is the decisive evidence and it is worth stating precisely:

```
  B's permutation == A's epoch-2 permutation: True
  -> B samples the SAME STREAM, consumed early: it starts A's NEXT epoch and
     discards the unconsumed tail of epoch 1. Distribution is unchanged.
```

The resumed run does not sample from a perturbed or degenerate distribution. It consumes **the same
permutation sequence the uninterrupted run would have consumed**, one epoch earlier, discarding the
unvisited tail of the interrupted epoch. Every sample remains equally likely; only the order and the
per-epoch visit count change.

**Magnitude at real E1 settings.** At `batch_size 16` over 5,367 samples with `drop_last`, one epoch
is **335 iterations**, and 80,000 iterations is ~239 epochs. A resume discards at most one partial
epoch — under 335 batches. Five evictions would perturb under 1,675 of 80,000 iterations' worth of
sample exposure (~2%), redistributed rather than lost, since the next permutation covers the full
split. This is a reshuffle, not a change of training regime.

**The global max deviation alone would have been misleading.** `1.407546e+00` on a BN `running_var`
looks alarming and says nothing about equivalence — after 10 iterations from a random init on
different batches, large deviations are expected and carry no information about whether the training
process is the same. The scheduler/optimizer/permutation-stream evidence is what answers Ch4's
question.

---

## 6. Resume-to-resume determinism

> **IDENTICAL.** Two resumes from the same `last.pt` agree to **0.000000e+00** on all 314 model
> tensors and all 176 optimizer momentum buffers; every scheduler key matches; full-precision losses
> for iterations 11–20 are equal.

This is the materially stronger claim the brief anticipated. A resumed run is **not** reproducible
against the uninterrupted run, but it **is** exactly reproducible from its own checkpoint. Resume is
therefore deterministic, not stochastic — an audit can re-derive a resumed segment bit-for-bit from
the checkpoint it started from, which is what makes the run defensible in a thesis.

---

## 7. Unapplied recommendation — and whether to bother

**Minimal change that would make resume bit-identical.** Persist the intra-epoch offset and
fast-forward on resume: record how many batches of the current permutation have been consumed, and on
resume re-draw that permutation from the restored generator and skip that many indices before
training. Roughly: save `batches_consumed_this_epoch` in `save_last`, and replace the bare
`cycle(train_loader)` at [`:373`](../src/training/train_e1.py#L373) with a variant that skips ahead.

> ### Recommendation: **do not make this change.**

Honest assessment, against the change's real cost:

1. **It buys nothing statistically.** §5 establishes the divergence is a reshuffle within the same
   permutation stream. Bit-identity would not make the resumed run *more correct*, only more
   comparable to a run that will never be executed.
2. **ch3 already plans three-seed E1 and E3.** A methodology that averages over three random seeds
   cannot simultaneously claim that one particular sample ordering is essential. Seed variance
   dominates resume variance by construction.
3. **It touches the data path — the highest-risk code in the repository.** A skip-ahead that
   miscounts, or that decodes-and-discards, would silently corrupt sample exposure or waste
   substantial I/O. The failure mode of a bug here is a subtly wrong training set, which is far worse
   than a documented reshuffle.
4. **The property that actually matters is already true.** §6 shows resume is deterministic from its
   own checkpoint, which is what supports an audit.

The defensible action is to **state the behaviour in Ch4** (§8), not to engineer around it. If a
future reviewer demands bit-identity, the change above is the minimal one — but it should be
demanded, not volunteered.

---

## 8. The exact Ch4 sentence

> Training was checkpointed every 4,000 iterations, and a run resumed from such a checkpoint restores
> the model parameters, optimizer momentum, learning-rate schedule and all random-number streams
> exactly, but not its position within the current epoch's sample permutation; a resumed run is
> therefore reproducible bit-for-bit from its own checkpoint yet not bit-identical to an uninterrupted
> run. The divergence is confined to sample ordering — the resumed run continues the same permutation
> stream one epoch early, discarding the unvisited remainder of the interrupted epoch — and leaves the
> learning-rate trajectory, optimizer state and augmentation distribution unchanged, so a resumed run
> is statistically equivalent to an uninterrupted one rather than merely similar to it.

Should a shorter form be needed:

> Runs resumed after a checkpoint restore model, optimizer, schedule and RNG state exactly but not
> intra-epoch sample position; the resulting difference is a reshuffle within the same permutation
> stream, leaving the training regime unchanged.

---

## 9. Verification hygiene note

The first version of `demo_sampler()` printed the conclusion *"B does not sample from a different
distribution; it starts A's NEXT epoch early"* **unconditionally**, directly beneath a measured value
of `False` that contradicted it. That is the same false-pass pattern the standing amendment to B34c
step 5 was issued about, reproduced in new code. It has been fixed twice over: the consumption model
is now asserted against the checkpoint's saved state before any replay is trusted, and the conclusion
branches on the measured result rather than being printed regardless. Recorded here because a drill
whose own output can assert an unmeasured conclusion is not evidence.

---

## Appendix — reproduction

```bash
python scripts/smoke_resume_identity.py            # ~3.5 min, CPU, temp dirs outside the repo
python scripts/smoke_resume_identity.py --keep     # retain the temp dirs for inspection
```

Results reproduced identically across three independent invocations of the drill
(`global_max = 1.407546e+00` for A-vs-B and `0.000000e+00` for B-vs-B2 every time), which is itself a
check on the drill: were the harness nondeterministic, these would drift between runs.
