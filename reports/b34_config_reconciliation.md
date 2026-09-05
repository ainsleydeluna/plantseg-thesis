# B34 — Pre-launch config↔report reconciliation sweep

**Type:** read-only audit. **Status:** COMPLETE. **Date:** 2026-09-04.
**Scope:** prove whether the real E1 training path loads the computed CE class weights, and reconcile
every literal `NEED_TO_CONFIRM` in `configs/` against the reports that may already have closed it.

**No file was edited, created, staged, committed, or pushed by this task except this report.**
`docs/reference/reference.pdf` was not opened, read, hashed, copied, or staged. No training, no
downloads, no installs, no GPU. Every claim below is quoted from a file read during this task.

---

## 1. Part A — class-weight wiring trace

### A1. Where the supervised loss object is constructed

[`src/training/train_e1.py:292-296`](../src/training/train_e1.py#L292):

```python
    # --- loss (class-weighted CE + Dice); weight buffer follows .to(device) ---
    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=IGNORE_INDEX).to(dev)
    print(f"[loss] CombinedCEDiceLoss(weight=len{weights.numel()}, ignore_index={IGNORE_INDEX}) "
          f"weight_on={criterion.ce.weight.device}")
```

`weight=` receives **a tensor**. Not `None`, not a config lookup.

### A2. Provenance of that value

**(a) — it loads `reports/e1_class_weights.json`.**

[`train_e1.py:49`](../src/training/train_e1.py#L49):

```python
CLASS_WEIGHTS_JSON = REPO / "reports" / "e1_class_weights.json"
```

Explicitly **not** (b), (c), (d), or (e):

| candidate | status | evidence |
|---|---|---|
| (b) reads `configs/loss.py["real_class_weights"]` | **NO** | `grep -rn "LOSS\[" --include=*.py .` returns four hits, all `num_classes` / `ignore_index`, in `scripts/smoke_loss.py:24-25` and `scripts/smoke_metrics.py:23-24`. **`real_class_weights` has zero readers in the repository.** |
| (c) calls `losses.compute_class_weights()` on a batch | **NO** | that function is defined at [`losses.py:19`](../src/training/losses.py#L19) and is never called from `train_e1.py` (grep: no call site). |
| (d) passes `None` | **NO** | see A1. |
| (e) other | **NO** | — |

The artifact is git-tracked, so it ships with a clone: `git ls-files --error-unmatch
reports/e1_class_weights.json` succeeds; introduced by commit `4eb5644` *"add E1 CE class weights
(B18a)"*; `git check-ignore` returns nothing.

### A3. REAL-RUN path vs `--dry-run` path

**They do not diverge.** `main()` computes mode-specific values, then makes a **single** call to
`run()` for both modes — [`train_e1.py:600-605`](../src/training/train_e1.py#L600):

```python
    return run(mode=mode, device=device, pretrained=pretrained, batch_size=batch_size,
               max_iters=max_iters, val_interval=val_interval, max_val_batches=max_val_batches,
               num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed,
               resume=args.resume, ckpt_interval=args.ckpt_interval,
               jsonl_name=args.jsonl_name, keep_ckpts=args.keep_ckpts)
```

`mode` is consumed inside `run()` in exactly two places — the pretrained guard
([`:277`](../src/training/train_e1.py#L277)) and the non-finite-loss abort
([`:392`](../src/training/train_e1.py#L392)). Neither touches the loss weights. The dry-run therefore
**does** exercise `load_ce_weights()`, so it cannot hide this code path.

**Resume does not reintroduce a stale vector.** The checkpoint payload carries only
`model_state_dict` / `optimizer_state_dict` / `scheduler_state_dict`
([`:139-141`](../src/training/train_e1.py#L139), [`:159-161`](../src/training/train_e1.py#L159)), and
the resume branch restores only those ([`:342-344`](../src/training/train_e1.py#L342)). The criterion
is rebuilt from the JSON on every `run()`.

### A4. Verification of the loaded vector

| requirement | result | evidence |
|---|---|---|
| length 116 | **PASS** | `len(json["weights"]) == 116`; enforced at [`train_e1.py:60-61`](../src/training/train_e1.py#L60). |
| index-aligned 0–115 | **PASS — independently reproduced** | see below. |
| dtype float32 | **PASS** | `t = torch.tensor(w, dtype=torch.float32)` at [`train_e1.py:59`](../src/training/train_e1.py#L59). |
| moved to training device | **PASS** | `.to(dev)` at [`:293`](../src/training/train_e1.py#L293), and again on the module at [`:294`](../src/training/train_e1.py#L294). `weight` is a registered buffer ([`losses.py:49`](../src/training/losses.py#L49)), so it follows `.to()`. Printed at runtime by [`:295-296`](../src/training/train_e1.py#L295). |
| index 0 is the small background weight (~0.0370), not a 1.0 placeholder | **PASS** | `w[0] = 0.036954205368266338`. |

**Index alignment was not taken on trust.** The JSON asserts alignment, but the assertion was verified
by recomputing the documented formula from the artifact's own `pixel_counts` and comparing elementwise:

```
w_c = sqrt(total_nonignore_pixels / (num_classes * pixels_in_class)), then median-normalized
max |recomputed - stored| over all 116 indices = 0.0    (exact, not tolerance-passing)
sum(pixel_counts) == total_nonignore_pixels == 4,582,656,621   (True)
```

Background is 3,703,512,045 px = **80.82%** of non-ignore pixels, which is why index 0 carries the
smallest weight. This is the natural inverse-frequency result, not a placeholder.

### A5. Does any gate assert A1–A4?

| script | asserts A1–A4? | evidence |
|---|---|---|
| `scripts/verify_env.py` | **NO** | grep for `class_weight` / `e1_class_weights` / `load_ce_weights` returns one unrelated hit, [`:182`](../scripts/verify_env.py#L182) `print("\n[13-14] torch hub cache / ImageNet weights")` — the torch-hub cache check, not CE weights. |
| `scripts/preflight_e1.py` (the B31c-3 gate) | **NO direct assertion; PARTIAL indirect coverage** | grep returns zero class-weight hits. But stage 4/4 runs `train_e1.py --dry-run` ([`:110-111`](../scripts/preflight_e1.py#L110)), which executes `load_ce_weights()`. |

The B31c gate is `scripts/preflight_e1.py` — identified from its own docstring,
[`:1-4`](../scripts/preflight_e1.py#L1): *"Single pre-flight gate for the real E1 run (B31c-3)."*

**What the indirect coverage actually buys:** because `load_ce_weights()` is fail-closed (A7), a
missing, malformed, wrong-length, or non-finite JSON **fails the gate**. What is *not* asserted:
index alignment, the index-0 value, dtype, and device placement. The six hard checks inside
`train_e1.py` ([`:382`](../src/training/train_e1.py#L382), `:388`, `:399`, `:422`, `:452`, `:473`)
contain no weight check.

### A6. VERDICT

> ## **WIRED**

**The single line of evidence that decides it —** [`train_e1.py:293`](../src/training/train_e1.py#L293):

```python
    weights = load_ce_weights().to(dev)
```

`load_ce_weights` resolves to `reports/e1_class_weights.json` and to nothing else. The config value is
never consulted.

**The task brief's stated failure mode is disproven.** The brief warned: *"If the training loop reads
the config value rather than the JSON, E1 launches with an unweighted CE and the run is wasted."* The
loop reads the JSON; `LOSS["real_class_weights"]` has **zero readers repository-wide**. `configs/loss.py:14`
is a stale **record**, not a live defect. E1 is not blocked by it.

### A7. Fail-closed analysis of `load_ce_weights()`

Quoted in full, [`train_e1.py:55-64`](../src/training/train_e1.py#L55):

```python
def load_ce_weights(path: Path = CLASS_WEIGHTS_JSON) -> torch.Tensor:
    """Length-116 CE class weights (index-aligned to class id 0..115) from the B18a artifact."""
    with open(path, encoding="utf-8") as f:
        w = json.load(f)["weights"]
    t = torch.tensor(w, dtype=torch.float32)
    if t.numel() != NUM_CLASSES:
        raise ValueError(f"class-weights length {t.numel()} != num_classes {NUM_CLASSES}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("class-weights contain non-finite values")
    return t
```

> ### **FAIL-CLOSED. No fallback exists.**

Five distinct failure paths, all raising:

| failure | raises |
|---|---|
| file missing | `FileNotFoundError` (`open`) |
| file not valid JSON | `json.JSONDecodeError` |
| `"weights"` key absent | `KeyError` |
| length ≠ 116 | `ValueError` (explicit) |
| any non-finite value | `ValueError` (explicit) |

There is **no** fallback to `None`, to `torch.ones`, or to a batch-derived vector, and no default
argument that could silently substitute one. **No silent-degradation path exists** — this is a finding
in the repository's favour and is recorded as such.

The function contains no `try`/`except`. The only two `try`/`except` blocks in `train_e1.py` are
unrelated: [`:197-200`](../src/training/train_e1.py#L197) wraps `q.unlink()` during checkpoint
rotation, and [`:229-233`](../src/training/train_e1.py#L229) wraps `_git_head()`. All three call sites
across the repository are bare — [`train_e1.py:293`](../src/training/train_e1.py#L293),
[`train_distill.py:301`](../src/training/train_distill.py#L301),
[`quant/runner.py:363`](../src/quant/runner.py#L363) — so an unreadable artifact aborts the run rather
than degrading it.

---

## 2. Part B — `NEED_TO_CONFIRM` sweep across `configs/`

### B1 + B5. Hits, and how the arithmetic closes

`grep -rn "NEED_TO_CONFIRM" configs/` returns **14 text hits + 4 binary matches**.

**Exclusion rule (stated so the arithmetic closes):**

- **4 binary `.pyc` matches excluded** — `configs/__pycache__/{distill,loss,model,quant}.cpython-313.pyc`
  are compiled copies of the same source lines already counted. Counting them would double-count.
- **4 prose comments excluded from bucketing** — `distill.py:4`, `loss.py:4`, `quant.py:5`,
  `teacher/…:356`. These are file-header narration *describing* the convention ("unresolved values stay
  the literal NEED_TO_CONFIRM string"), not assignable values. No code can read them, so no bucket applies.

> **14 text hits = 10 assignable keys + 4 prose comments.** The 10 keys are bucketed below.

### B6. OPEN vs DEFERRED are not disjoint — the tie-break, stated

As specified, `STALE` / `OPEN` / `DEFERRED` overlap: a key can simultaneously "need a run or a decision"
**and** "belong to a later stage." **Six of the ten keys satisfy both criteria:**

| key | file:line | needs a run/decision? | later stage? |
|---|---|---|---|
| `lambda_logit` | `configs/distill.py:29` | yes — validation sweep | yes — E2 |
| `logit_kd_weight` | `configs/loss.py:15` | yes — same sweep | yes — E2 |
| `e6kd_reduced_weights` | `configs/quant.py:33` | yes — contingency decision | yes — E6 |
| `scale` | `configs/quant.py:136` | yes — runtime-derived | yes — E4 |
| `zero_point` | `configs/quant.py:137` | yes — runtime-derived | yes — E4 |
| `NMF_SEED_CONTROL` | `configs/teacher/…:358` | yes — settle vs pinned stack | yes — teacher |

**Tie-break applied:** *DEFERRED wins whenever the resolution mechanism is calendared to a stage after
E1.* `OPEN` is reserved for keys that are unresolved **and** required at or before E1 — the only
category that could bear on this launch.

The two teacher sentinels (`:363`, `:369`) are **DEFERRED-only**: their values are final by design
(refuse-to-launch sentinels overridden by environment variables), so they need no run or decision.

> **`OPEN = 0` is a consequence of that tie-break, not evidence that nothing is unresolved.**
> Six of the eight DEFERRED keys are genuinely unresolved work. They are simply not E1's work.

### B2 + B3. The bucketed table

| file:line | key | bucket | resolving artifact / disposition | E1-blocking |
|---|---|---|---|---|
| `configs/loss.py:14` | `real_class_weights` | **STALE** | `reports/e1_class_weights.json` (B18a, commit `4eb5644`, generated 2026-06-29) + `reports/e1_class_weights.md`. Resolved value: length-116 √inv-freq vector, median-normalized, no cap. | **NO** — see note |
| `configs/model.py:19` | `pretrained` | **STALE (and dead)** | `configs/e1_student.py:8` `init_weights`, consumed via `build_student(pretrained=…)`. | **NO** |
| `configs/loss.py:15` | `logit_kd_weight` | DEFERRED | E2 validation sweep `{0.25,0.5,1,2,4}`; `docs/open_questions.md:716` item 3. | no |
| `configs/distill.py:29` | `lambda_logit` | DEFERRED | same sweep. Literal is **load-bearing** — asserted at `scripts/smoke_realrun_decisions.py:151`. | no |
| `configs/quant.py:33` | `e6kd_reduced_weights` | DEFERRED | contingency only if E3→E6 clean mIoU drop > 1.0 pp; `open_questions.md:722` item 4. | no |
| `configs/quant.py:136` | `scale` | DEFERRED | E4 runtime-derived qparams; gated on QNNPACK INT8 Sigmoid support. | no |
| `configs/quant.py:137` | `zero_point` | DEFERRED | as above. | no |
| `configs/teacher/…:358` | `NMF_SEED_CONTROL` | DEFERRED | teacher stage; to be settled against the pinned stack. Literal is **load-bearing** — asserted at `scripts/smoke_teacher_config.py:208`. | no |
| `configs/teacher/…:363` | `ADE20K_CKPT_UNSET_SENTINEL` | DEFERRED (by design) | deliberate refuse-to-launch sentinel; overridden by `$SEGNEXT_ADE20K_CKPT`. | no |
| `configs/teacher/…:369` | `WORK_DIR_UNSET_SENTINEL` | DEFERRED (by design) | deliberate refuse-to-launch sentinel; overridden by `$TEACHER_WORK_DIR`. | no |

**Counts: STALE 2 · OPEN 0 · DEFERRED 8.**

### B3 (cont.) — why the two STALE items are *not* flagged E1-BLOCKING

The brief instructed: *"Flag every STALE item that is on the E1 path as E1-BLOCKING."* Applied
mechanically that would tag `configs/loss.py:14`. **That label would be wrong, and inflating it would
misrepresent launch risk.**

1. **Nothing reads either key.** `LOSS["real_class_weights"]`: zero readers. `MODEL["pretrained"]`:
   zero readers — `MODEL[...]` is read only at
   [`src/models/student.py:248-253`](../src/models/student.py#L248) for `low_tap_index`,
   `high_tap_index`, `head_inter_channels`, `dilated`, `backbone_truncate`. `pretrained` arrives as a
   **function argument** ([`student.py:235`](../src/models/student.py#L235),
   [`:252`](../src/models/student.py#L252)), supplied by
   [`train_e1.py:591`](../src/training/train_e1.py#L591) from `E1_STUDENT["init_weights"]`.
2. **`configs/loss.py` is not even imported by the E1 training path** — only by
   `scripts/smoke_loss.py:21` and `scripts/smoke_metrics.py:20`.

> **Verdict: documentation defect, fix-before-launch, NOT run-blocking.**
> `configs/loss.py:14` still matters — it asserts *"not yet computed"* about an artifact the training
> loop hard-depends on and would abort without. It misleads any reader, including the thesis. But it
> cannot change what E1 computes.

### B4. Prior-art disclosure

`reports/pre_e1_launch_audit.md:2898` **already ran this sweep** and reached the same two STALE items.
B34 is a re-verification, not a discovery. Every finding above was independently re-derived from the
files during this task; the prior audit is cited for provenance, not relied upon. That report also
states the contradiction directly at
[`:3839-3843`](pre_e1_launch_audit.md): *"`configs/loss.py:14` actively contradicts the training loop."*

### B7. Adjudication of `reduce_zero_label` (asked by name)

**This key is not a `NEED_TO_CONFIRM` and never appeared in the B1 grep.** It is already the literal
`False` at [`configs/loss.py:16`](../configs/loss.py#L16).

> ### Verdict: **CORRECTLY LITERAL — the value is not stale. The trailing comment is.**

**Value — agrees with every artifact:**

| source | value |
|---|---|
| `configs/loss.py:16` | `False` |
| `configs/data.py:37` | `False` |
| `reports/e1_class_weights.json` (`reduce_zero_label` field) | `false` |
| official PlantSeg `configs/_base_/datasets/plantseg115.py` (quoted at `docs/open_questions.md:44`) | `reduce_zero_label=False` |

No contradiction with any artifact the training loop depends on. The training loop performs no label
remap.

**Comment — stale pointer.** `configs/loss.py:16` says the disease-only reporting convention is
*"separate & still open — see docs/open_questions.md #2"*, and `configs/data.py:37` says it *"remains
open; see docs/open_questions.md #2."* But [`docs/open_questions.md:20`](../docs/open_questions.md#L20)
reads:

> `### 2. ✅ RESOLVED (D1 + A0-FIX 2026-07-26) — disease-only / background convention (reduce_zero_label)`

and [`:47-48`](../docs/open_questions.md#L47): *"disease-only metric definitions are **no longer marked
PROVISIONAL** on account of the class mapping."*

Both comments point at a question that closed on 2026-07-26. **Documentation-only, not E1-blocking.**
Offered as an optional edit in §3, deliberately *not* bundled with the two STALE fixes.

---

## 3. Proposed minimal edits — **UNAPPLIED**

Nothing in this section has been applied. Tiers are separated so nothing gets bundled silently.

**Safety pre-check performed:** no test pins either STALE literal. `grep -rn
"real_class_weights\|logit_kd_weight" --include=*.py .` returns only the two `configs/loss.py`
definitions themselves. This contrasts with `configs/distill.py:29` and `configs/teacher/…:358`, whose
literals **are** asserted by `scripts/smoke_realrun_decisions.py:151` and
`scripts/smoke_teacher_config.py:208` — confirming those are correctly DEFERRED and must not be touched.

### Tier 1 — the two STALE records (recommended)

**Edit 1 — `configs/loss.py:14`**

```diff
-    "real_class_weights": "NEED_TO_CONFIRM",     # sqrt inv-freq over FULL train set (not yet computed)
+    # RESOLVED B18a (commit 4eb5644, 2026-06-29). Pointer, not a copy: duplicating 116 floats here
+    # would create a second source of truth that can drift from the artifact the loop actually loads.
+    "real_class_weights": "reports/e1_class_weights.json",   # loaded by src/training/train_e1.py:55,293
```

*Alternative considered and rejected:* inlining the 116 floats. Rejected — it duplicates a load-bearing
artifact and invites drift. The pointer form is the minimal drift-safe edit.

**Edit 2 — `configs/model.py:19`**

```diff
-    "pretrained": "NEED_TO_CONFIRM",   # ImageNet weights not cached; weights=None for smoke (no download)
+    # NOT READ by any code path. Initialization is governed by configs/e1_student.py["init_weights"],
+    # passed as build_student(pretrained=...) (src/models/student.py:235,252) from train_e1.py:591.
+    "pretrained": "governed by configs/e1_student.py['init_weights'] (this key is not read)",
```

*Alternative for B34b to decide:* delete the key entirely. `pre_e1_launch_audit.md:2898` classifies it
**"STALE + DEAD."** Deletion is cleaner but is a larger change than this pass recommends; the
descriptive string is chosen over `None` because `None` could be misread as "no pretraining."

### Tier 2 — optional documentation fix (B7)

Retarget the two stale comment pointers at `configs/loss.py:16` and `configs/data.py:37` so they no
longer describe `open_questions.md` #2 as open. **Values unchanged — comment text only.** Listed
separately because it is unrelated to the class-weight question.

### Tier 3 — gate hardening (C1). Separate decision; adopt or reject independently

Closes the A5 gap. **Neither tier changes any computed metric or loss value; both are pure assertions.**

#### Tier 3a — cheap, every preflight run

Add to `scripts/preflight_e1.py` as a fifth stage, or to `scripts/verify_env.py`:

```python
# --- provenance + correctness assertions on the B18a CE class weights (B34/C1 Tier 1) ---
import json, math, torch
d = json.load(open(REPO / "reports" / "e1_class_weights.json", encoding="utf-8"))
w = d["weights"]
t = torch.tensor(w, dtype=torch.float32)

# --- correctness: these bind the vector to the dataset and to the CE contract ---
assert t.numel() == 116,                     "class-weights length != 116"
assert bool(torch.isfinite(t).all()),        "class-weights contain non-finite values"
assert t.dtype is torch.float32,             "class-weights dtype != float32"
assert int(t.argmin()) == 0,                 "argmin != 0 (background must carry the smallest weight)"
assert int(t.argmax()) == 42,                "argmax != 42"
# w[42]/w[0] == sqrt(N_0/N_42) exactly: scale-invariant, but a pure function of the pixel
# histogram, so it survives any renormalization yet breaks if the dataset changes.
assert math.isclose(w[42] / w[0], 259.551929472253, rel_tol=1e-9), "weight ratio drift"
# device placement, checked on the object the loop actually builds:
#   crit = CombinedCEDiceLoss(weight=load_ce_weights().to(dev), ignore_index=255).to(dev)
#   assert crit.ce.weight.device == dev

# --- PROVENANCE ONLY -- NOT a correctness check. See the note below before changing this. ---
srt = sorted(w)
assert math.isclose((srt[57] + srt[58]) / 2, 1.0, abs_tol=1e-12), "median-normalization fingerprint"
```

> #### ⚠ Why the median assertion must be written this way
>
> **116 is even, so the two median conventions disagree**, and the obvious formulation would **NO-GO a
> perfectly correct artifact**:
>
> | convention | value | as an assertion |
> |---|---|---|
> | average of the two middles (`sorted[57]+sorted[58])/2`, numpy) | `1.000000000000` | ✅ passes |
> | lower middle (`torch.median`) | `0.996273398399` | ❌ **fails** |
>
> The B18a artifact was normalized with the average-of-middles convention.
>
> **It is labelled PROVENANCE, not correctness, because the median value cannot affect training** —
> see C3 below. It fingerprints *which script generated the artifact*. Nothing more.

**By contrast, `w[42]/w[0]` is correctness-relevant *and* dataset-binding.** Verified:

```
w[42]/w[0]     = 259.551929472253
sqrt(N_0/N_42) = 259.551929472253      (N_0 = 3,703,512,045   N_42 = 54,975)
agreement      = 0.00e+00 (exact)
```

Because `w_c ∝ 1/sqrt(N_c)`, the ratio is a pure function of the pixel histogram. It is invariant to
any renormalization — so it cannot be broken by a convention change — yet it breaks immediately if the
underlying class distribution changes. It is the strongest single scalar in the artifact.

#### Tier 3b — one-time, pod-side, after the dataset lands and before E1

Recompute the full train-mask pixel histogram **on the pod** and assert it matches the JSON's
`pixel_counts` and `total_nonignore_pixels` **exactly**, emitting
`reports/e1_class_weights_pod_verify.md`.

**Estimated runtime: ~1–2 minutes single-threaded; budget ~5 minutes on a pod network volume (cold
cache).** Grounded, not guessed: a 60-mask sample was timed locally at 0.009 s/mask; that sample skewed
small on pixels (3.34e9 projected vs 4.58e9 recorded), so the pixel-scaled extrapolation is ~65–70 s.

**Why split-count matching is insufficient to bind the weights to the pod dataset.**
[`scripts/verify_env.py:249-284`](../scripts/verify_env.py#L249) compares only the **number** of
image/mask pairs per split against `configs/data.py` `SPLIT_SIZES` (`train: 5367`, matching the JSON's
`train_mask_count: 5367`). That is a **cardinality** check. Identical counts are fully compatible with
different pixel content: a re-exported, re-rasterized, differently-compressed, or partially-corrupted
mask set of the same cardinality passes unchanged. **The weights are a function of the per-class pixel
histogram over 4,582,656,621 pixels, not of file count.** Count equality is necessary, not sufficient.

---

## 4. C3 — median convention, gradient clipping, and the real E5 hazard

### The measurement

A reviewer hypothesis was put forward during this task: *scaled weights → scaled loss → shifted
gradient-clipping decisions*, implying the numpy/torch median conventions would be **not**
interchangeable at the QAT stages, since `src/quant/runner.py:363` is a `load_ce_weights` call site and
E5/E6 clip.

**The two factual premises check out.** `quant/runner.py:363` is a call site, and E5/E6 really do clip:
[`runner.py:379`](../src/quant/runner.py#L379) `torch.nn.utils.clip_grad_norm_(prepared.parameters(),
args.grad_clip_norm)`, gated at [`:142`](../src/quant/runner.py#L142), with
`QUANT["qat_grad_clip_pilot"]` still `"status": "PILOT_REQUIRED"` / `"selected_value": None`
([`configs/quant.py:89-92`](../configs/quant.py#L89)). The general principle is also sound: a uniform
**loss** scale does move clip decisions.

**The hypothesis is nevertheless refuted, because the median convention never produces a scaled loss.**
It scales the *weight vector*, and `F.cross_entropy(reduction='mean')` normalizes by the sum of
per-pixel weights — so the loss is not scaled, it is **unchanged**. An unchanged loss produces
unchanged gradients, which cannot move a clip decision.

Measured end-to-end through `CombinedCEDiceLoss` with `backward()`, comparing the stored vector against
the same vector scaled by the numpy↔torch normalizer ratio `1.003726590848`:

| weights | loss | gradient norm |
|---|---|---|
| as stored (average-of-middles normalized) | 5.9677643776 | 0.1524708569 |
| × 1.003726590848 (lower-middle convention) | 5.9677653313 | 0.1524708718 |

```
absolute gradient-norm difference : 1.490e-08
relative gradient-norm difference : 9.773e-08     (float32 eps = 1.19e-07)
```

The residual is **below float32 epsilon** — rounding, not signal. A clip threshold of 1.0 or 5.0 sees
the same gradient norm.

> ### Finding, recorded as MEASURED
>
> **The numpy and torch median conventions are interchangeable for E1 *and* for E5/E6, for the same
> reason:** `F.cross_entropy(reduction='mean')` normalizes by the sum of per-pixel weights, so scaling
> the weight vector leaves the loss **unchanged rather than scaled**, and an unchanged loss cannot move
> a gradient-clipping decision.
>
> The refuted hypothesis is attributed to the reviewer and recorded here so the reasoning is auditable.

**Corollary — the JSON's `matches_repo_fn` claim is imprecise but harmless.** The artifact records
`"matches_repo_fn": "src/training/losses.compute_class_weights"`, but that function uses
`w_present.median()` ([`losses.py:35`](../src/training/losses.py#L35)) — the torch lower-middle
convention — so for an even class count it would emit a vector ~1.0037× the artifact's. By the result
above this changes no loss and no gradient. **Documentation imprecision; training unaffected.**

### The real E5-stage hazard (replaces the refuted mechanism)

> ### ⚠ The danger in `compute_class_weights()` is **not** its median convention.

It is that the function derives weights **from a single batch**
([`losses.py:19-43`](../src/training/losses.py#L19)), where classes absent from that batch are assigned
`1.0`:

```python
    weights = torch.ones(num_classes, dtype=torch.float32)
    if present.any():
        w_present = torch.sqrt(total / (num_classes * counts[present]))
        med = w_present.median()
        weights[present] = w_present / med  # median-normalized
```

Its own docstring says so — [`losses.py:22-24`](../src/training/losses.py#L22): *"SCAFFOLD /
SMOKE-ONLY: real thesis weights are computed over the FULL train set… Classes absent from `masks`
(common in a tiny batch) get weight 1.0 (= median)."*

This perturbs the **relative structure** of the vector, not its scale. **Relative changes do not
cancel.** Under `reduction='mean'` a relative change alters the loss, hence the gradients, hence — at
E5/E6 — which steps clip. This is a strictly stronger warning than the scale hypothesis it replaces.

**Status: LATENT, not live.** All three call sites currently load the JSON —
`train_e1.py:293`, `train_distill.py:301`, `quant/runner.py:363`. The hazard is that a future author
substitutes `compute_class_weights()` at the E5 stage. The Tier-3a assertions (`argmin == 0`,
`argmax == 42`, the ratio) would catch such a substitution immediately, since a batch-derived vector
would not reproduce them.

---

## 5. D1 — staleness cross-check (report only; no edits)

Input for the next task's scope, as requested.

### `open_questions.md` #2 — disease-only / `reduce_zero_label` convention

Current status line, verbatim — [`docs/open_questions.md:20`](../docs/open_questions.md#L20):

```
### 2. ✅ RESOLVED (D1 + A0-FIX 2026-07-26) — disease-only / background convention (`reduce_zero_label`)
```

> **RESOLVED.** Resolving artifact: the A0-FIX check against the official `tqwei05/PlantSeg` source,
> **2026-07-26**, quoted at `open_questions.md:38-46` (`METAINFO['classes']`,
> `decode_head.num_classes = 116`, `reduce_zero_label=False`). Consequence recorded at `:47-48`:
> disease-only definitions are *"no longer marked PROVISIONAL."*

### NTC-8 — absent-class eval support (class 41 no test, class 68 no val)

Three copies, all still reading Open, all from commit `e73d712` *"add B16 dataset audit reports"*,
**2026-06-29**:

| location | verbatim status |
|---|---|
| [`reports/metadata_csv_audit.md:118`](metadata_csv_audit.md) | `**Open (metrics):** per-class val/test IoU undefined for these; metric code must handle NaN. Both present in train.` |
| [`reports/trainval_mask_value_audit.md:99`](trainval_mask_value_audit.md) | `**Open** — per-class val/test IoU undefined for these; metrics step must handle NaN.` |
| [`reports/test_mask_value_audit.md:109`](test_mask_value_audit.md) | `Per-class test IoU for class 41 undefined; metrics step must handle absent-class NaN.` |

> **SUBSTANTIVELY RESOLVED — label never flipped.** Closed by `docs/EVALUATION_CONTRACT.md` §2.3/§3.1
> (**A0, 2026-07-26**), which records both absences explicitly at
> [`:30-31`](../docs/EVALUATION_CONTRACT.md#L30) and fixes union-present eligibility, implemented at
> [`src/eval/metrics.py:129-131`](../src/eval/metrics.py#L129).

**Direct question — is dataset-level macro-mIoU defined over 116 classes or over present-classes-only?**

> **Present-classes-only (union-present), *not* a fixed 116.**

[`src/eval/metrics.py:129-131`](../src/eval/metrics.py#L129):

```python
    tp, gt, pr, un = _cm_parts(cm)
    eligible = _restrict(un > 0, class_indices)
    return _macro(tp / un.clamp_min(1e-9), eligible)
```

A class absent from GT **and** prediction is dropped; a prediction-only class contributes `IoU = 0` and
**is counted**. `_macro` returns `NaN` only when nothing is eligible
([`:109-113`](../src/eval/metrics.py#L109)). The contract states the consequence directly —
[`docs/EVALUATION_CONTRACT.md:188-190`](../docs/EVALUATION_CONTRACT.md#L188): *"The eligible-class count
is therefore model-dependent and MUST be recorded per run."*

### Process observation

`configs/loss.py:14`, NTC-8 (×3), and the two `reduce_zero_label` comment pointers are the **same
defect class**: a status label that outlived its own resolution. Six instances found in this pass. Each
causes a reader — human or agent — to treat a closed question as open. Recorded here as input for the
next task; **no fix applied.**

---

## 6. What could NOT be determined from the files

Stated as unknown rather than inferred.

1. **Whether the pod-side dataset has the same label distribution as the one the weights were computed
   over.** The artifact records generation against `C:\Users\admin\plantseg_data\plantseg` at git head
   `122ba8d`; the pod mounts `/workspace/plantseg_data/plantseg`. The preflight matches split
   **counts** only (§3, Tier 3b). Content identity is **unverifiable from the repository** and is
   exactly what Tier 3b would close. **Unknown until run on the pod.**

2. **Whether the ~1.0037× `matches_repo_fn` convention mismatch was deliberate or incidental.** The
   B18a generator was *"`scripts`-style scratch compute"* (`reports/e1_class_weights.md:6`) and is not
   committed, so its source cannot be read. Its numerical output is fully reproducible from the
   artifact (§1, A4), but the author's intent is **unknown**.

3. **Whether `MODEL["pretrained"]` should be corrected or deleted.** Both are defensible; this is a
   project decision, not a fact recoverable from the files. Deferred to B34b.

4. **Whether three-seed E1 changes anything here.** `pre_e1_launch_audit.md:3849` raises a separate
   planning gap (ch3 requires three-seed E1; the runbook and checkpoint naming assume one run). It does
   not bear on class-weight wiring — the same JSON would be loaded by every seed — but it was **not
   investigated** in this pass and is **out of scope**.

5. **Runtime behaviour of any assertion in §3.** Nothing in Tier 3 was executed; no gate was run. The
   Tier-3a values were verified against the artifact, but the code block itself is **untested** as
   written.

---

## Appendix — commands run (all read-only)

```
git status -sb ; git log --oneline -5
git ls-files --error-unmatch reports/e1_class_weights.json
git check-ignore -v reports/e1_class_weights.json
grep -rn "NEED_TO_CONFIRM" configs/
grep -rn "LOSS\[" --include=*.py .        # readers of configs/loss.py
grep -rn "MODEL\[" --include=*.py .       # readers of configs/model.py
grep -rn "e1_class_weights\|load_ce_weights" --include=*.py --include=*.md .
```

Numerical verification used `C:\Users\admin\anaconda3\python.exe` (read-only: JSON parsing, a 60-mask
timing sample, and synthetic-tensor loss/gradient comparisons). **No dataset file was modified and no
model was trained.**
