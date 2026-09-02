# B32c — commit B32, λ semantics guard, and the E2/E3 VRAM question

**Base:** `fa3e8a0` · **HEAD:** see §1 · pushed to `origin/master`

---

## ⚑ HEADLINE — E2/E3 plausibly fit on a 4090, and the A6000 was probably never needed

| | E2 | E3 |
|---|---|---|
| activations + static, **pre-B32** | 13.63 GB | 13.78 GB |
| activations + static, **post-B32** | **10.04 GB** | **10.20 GB** |
| INFERRED peak on CUDA, batch 16 | **11–14 GB** | **11–14 GB** |

**Both stories you asked me to distinguish resolve the same way, and it is the second one.** This is *not* "E2 needed 31 GB and now needs 9". It is **"E2 needed 13.6 GB and now needs 10.0 GB"** — the pre-B32 figure already fit inside 24 GB with ~10 GB of headroom. **B32 widened an already-sufficient margin; it did not retire a genuine 24 GB overflow.**

So if the 24 GB → 48 GB pod switch was justified on VRAM grounds, **that justification looks to have been wrong even before B32**. Neither of the two things one would blame was the driver:

- **The teacher retains nothing.** `FrozenTeacher.forward` is `@torch.no_grad()`, so it contributes **zero** retained activations — MEASURED, not assumed. Its cost is a transient working set bounded at **~0.3–0.55 GB**.
- **The KD term was 3.7 GB**, real but not decisive at a 24 GB budget.

The dominant costs are the **student's own retained activations (4.47 GB)** and the **CE+Dice term (5.62 GB)** — both at full 512×512, both untouched by B32, and both inherent to the locked batch-16 / 512² recipe. **E2 and E3 therefore cost only marginally more than E1 itself.** If E1 fits on a 4090, E2/E3 fit on a 4090.

**This is INFERRED and is NOT a GO.** See §3 for the method, what it misses, and the one command that settles it.

---

## 1. Commit log

| # | Hash | Contents | Dry-run at that commit |
|---|---|---|---|
| D1 | `6a6113b` | B32/F8 core — `losses.py`, `train_distill.py`, `__init__.py`, 2 smokes | `RESULT: PASS (6/6 checks exercised, 0 skipped)` |
| D2 | `1529a54` | `scripts/smoke_kd_resolution.py` (new gate) | `RESULT: PASS (6/6 checks exercised, 0 skipped)` |
| D3 | `23bce3e` | `reports/b32_kd_resolution.md` | `RESULT: PASS (6/6 checks exercised, 0 skipped)` |
| D4 | `0d437c6` | B32c-2 λ semantics guard — `configs/distill.py`, `train_distill.py` | `RESULT: PASS (6/6 checks exercised, 0 skipped)` |
| D5 | `99108e1` | B32c-4 contract + open-questions | `RESULT: PASS (6/6 checks exercised, 0 skipped)` |
| D6 | (this report) | `reports/b32c_closeout.md` | docs-only; verified at final HEAD |

**D1 is atomic by necessity.** A signature change and its five call sites cannot split without an intermediate commit where `smoke_distill.py` and `train_distill.py` are broken — which would defeat the bisectability the previous series paid to establish.

### Bisectability — MEASURED in a throwaway worktree outside the repo

```text
commit   label                                     train_e1 --dry-run
6a6113b  B32/F8: compute Logit-KD at the head's    RESULT: PASS (6/6 checks exercised, 0 skipped)
1529a54  B32-6: standing gate for KD spatial re    RESULT: PASS (6/6 checks exercised, 0 skipped)
23bce3e  B32 audit trail: KD resolution report     RESULT: PASS (6/6 checks exercised, 0 skipped)
0d437c6  B32c-2: lambda_logit semantics tag —    RESULT: PASS (6/6 checks exercised, 0 skipped)
99108e1  B32c-4: record the OS8 pin, the lambda    RESULT: PASS (6/6 checks exercised, 0 skipped)
```

**5/5 green.** Worktree removed; `git worktree list` afterwards:
```text
C:/Users/admin/plantseg-thesis 99108e1 [master]
```

### Push
```text
To https://github.com/ainsleydeluna/plantseg-thesis.git
   fa3e8a0..99108e1  master -> master
```

---

## 2. B32c-2 — the λ semantics guard

### Premise correction, accepted before building

The item said *"the JSONL run_meta line already carries resolved config — extend it"*. **`train_distill.py` had no JSONL at all** — B31-7 was E1-scoped. So there was nothing to extend, and the guard adds **one `run_meta` line**, not the E1 telemetry stack.

Also: **no λ-selection artifact exists yet** — the sweep hasn't run, and λ enters only as a CLI float. The guard therefore sits on the **consumption** path.

### Shape (B — stamp plus opt-in assertion)

| Element | Behaviour |
|---|---|
| `LOGIT_KD_SEMANTICS` | `"logitkd@os8-64x64-of-512"` — self-describing: the Logit-KD KL was computed at output stride 8, on a 64×64 grid, from a 512×512 input. No decoder ring. |
| `LOGIT_KD_SEMANTICS_SUPERSEDED` | `("logitkd@full-512x512-upsampled",)` — named so a stale tag is diagnosed, not merely rejected |
| stamp (always) | checkpoint payload **and** `<stage>_run_meta.jsonl`: actual tag, declared tag, override flag, every term's grid, locked constants |
| `--lambda-semantics` | optional; when declared it is checked, and a mismatch **refuses** a real run |
| `--allow-semantics-mismatch` | proceeds — and the override is **stamped into both artifacts** |

Declaring is optional by design: **the sweep runs produce the tag rather than consume it**, so requiring it there would add friction at the point of lowest risk. The risk is downstream.

### Acceptance

```text
==============================================================================
B32c-2 GUARD ACCEPTANCE
==============================================================================
--- [1] no tag declared (the sweep runs) -> proceeds, records declared=None ---
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=None override=False -> e2_run_meta.jsonl
RESULT: PASS

--- [2] matching tag declared -> proceeds ---
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=logitkd@os8-64x64-of-512 override=False -> e2_run_meta.jsonl
RESULT: PASS

--- [3] SUPERSEDED pre-B32 tag on a REAL run -> must REFUSE ---
REFUSING to start the real E2 run on CPU: CUDA is not available (torch.cuda.is_available()=False).
   exit=0

--- [4] override -> proceeds AND is stamped durably ---
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=logitkd@full-512x512-upsampled override=True -> e2_run_meta.jsonl
[semantics] *** OVERRIDE ACTIVE: --allow-semantics-mismatch was used. This run's lambda_logit was selected under DIFFERENT Logit-KD semantics and its results are NOT comparable to runs without the override. Recorded in the checkpoint payload and in e2_run_meta.jsonl. ***
RESULT: PASS

--- [5] durability: the artifacts, not the terminal ---
  run_meta lines:
    declared=None                                     override=False  actual=logitkd@os8-64x64-of-512
    declared='logitkd@os8-64x64-of-512'               override=False  actual=logitkd@os8-64x64-of-512
    declared='logitkd@full-512x512-upsampled'         override=True  actual=logitkd@os8-64x64-of-512

  full run_meta line 1:
    {
        "event": "run_meta",
        "stage": "E2",
        "mode": "dry",
        "seed": 42,
        "lambda_logit": 1.0,
        "logit_kd_semantics": "logitkd@os8-64x64-of-512",
        "logit_kd_semantics_declared": null,
        "logit_kd_semantics_override_used": false,
        "logit_kd_grid": "os8 64x64 (head-native, no upsample)",
        "cwd_logit_grid": "os8 64x64 (shared validity mask)",
        "cwd_feat_grid": "stride-16 32x32",
        "supervised_grid": "full 512x512",
        "T_logit": 4,
        "T_cwd": 4,
        "alpha_cwd": 50,
        "beta_cwd": 3,
        "cwd_C": 320,
        "lambda_sweep_grid": [
            0.25,
            0.5,
            1,
            2,
            4
        ],
        "batch_size": 1,
        "max_iters": 2,
        "num_classes": 116
    }

  checkpoint payload semantics keys (1 ckpt):
    logit_kd_semantics = 'logitkd@os8-64x64-of-512'
    logit_kd_semantics_declared = 'logitkd@full-512x512-upsampled'
    logit_kd_semantics_override_used = True

--- [6] E3 path also stamped ---
[semantics] logit_kd=logitkd@os8-64x64-of-512 declared=None override=False -> e3_run_meta.jsonl
RESULT: PASS

--- [3b] semantics gate, tested DIRECTLY ---
  (test [3] above could not reach it: on this CPU box the real-run CUDA refusal fires
   first in main(). The gate sits after the CUDA and grad-clip gates by design —
   cheapest/most fundamental refusals first. End-to-end ordering on a GPU is
   CANNOT-VERIFY-LOCALLY; the pod command is in the report.)

  PROCEEDS  None declared (sweep runs)   declared=None override=False
  PROCEEDS  matching tag                 declared='logitkd@os8-64x64-of-512' override=False
  REFUSES   SUPERSEDED pre-B32 tag       declared='logitkd@full-512x512-upsampled' override=False
            message: --lambda-semantics 'logitkd@full-512x512-upsampled' (a SUPERSEDED pre-B32 tag) does not match the semantics this code implements, 'logitkd@os8-64x64-o...
  REFUSES   unknown/typo tag             declared='logitkd@os8' override=False
            message: --lambda-semantics 'logitkd@os8' does not match the semantics this code implements, 'logitkd@os8-64x64-of-512'. lambda_logit is not transferable acros...
  PROCEEDS  SUPERSEDED tag + override    declared='logitkd@full-512x512-upsampled' override=True

  all gate paths behave as specified -> True

  tag is self-describing without a decoder ring: 'logitkd@os8-64x64-of-512'
  superseded tags: ('logitkd@full-512x512-upsampled',)
```

### One honest limitation

Test **[3]** could not reach the semantics gate: on this CPU box the real-run **CUDA** refusal fires first in `main()`. The gate deliberately sits after the CUDA and grad-clip gates so the cheapest, most fundamental refusals come first. The gate function is therefore unit-tested directly across all five paths (**[3b]**, all correct), but **end-to-end ordering on a GPU is CANNOT-VERIFY-LOCALLY**:
```bash
python src/training/train_e2.py --real-run --confirm-real-run \
  --teacher-ckpt weights/<ckpt>.pth --lambda-logit 1.0 --grad-clip-norm 1.0 \
  --lambda-semantics 'logitkd@full-512x512-upsampled'
# expected: REFUSING to start the real E2 run: --lambda-semantics ... (a SUPERSEDED pre-B32 tag)
```

---

## 3. B32c-3 — VRAM re-measurement

### Method, stated plainly

| | |
|---|---|
| **How** | `torch.autograd.graph.saved_tensors_hooks` intercepts every tensor saved for backward and sums **unique storages**. Parameters / gradients / SGD momentum are exact arithmetic. |
| **Scaling** | MEASURED at batch 2, scaled ×8. The graph is fixed-shape and every retained tensor carries the batch dimension, so retained bytes are **exactly linear** in batch. |
| **Captures** | the activation set held across forward → backward — the dominant term. A measurement, not an estimate. |
| **Misses** | CUDA allocator caching and fragmentation (typically +5–15%), transient forward peaks not saved for backward, cuDNN workspace, and the backward pass's own temporaries. |
| **Teacher** | synthetic. The real SegNeXt-B was never built (MMSeg absent). Bounded analytically below. |

### Raw output

```text
==============================================================================
B32c-3 — E2/E3 peak-memory re-measurement
==============================================================================
MEASURED at batch 2, scaled x8 to batch 16. Retained-activation bytes via saved_tensors_hooks.

==============================================================================
E2
==============================================================================
  component                                  batch2 MB   batch16 MB
  student fwd (retained for bwd)                 558.7       4469.2
  CE + Dice term                                 702.0       5616.0
  teacher fwd (retained)                           0.0          0.0
  Logit-KD term (NEW, OS8)                         7.3         58.3
  Logit-KD term (OLD, 512 upsampled)             466.0       3728.0

  ACTIVATIONS, NEW path (total)                             10143.5
  ACTIVATIONS, OLD path (total)                             13813.3

  parameters (batch-independent):
    student params                            11.2 MB
    student grads                             11.2 MB
    SGD momentum buffer                       11.2 MB
    teacher params (frozen, no grad)         105.3 MB   [INFERRED ~27.6M params]
    static subtotal                          138.9 MB

  >>> E2 NEW total (activations + static) :   10282.4 MB = 10.04 GB
  >>> E2 OLD total (activations + static) :   13952.1 MB = 13.63 GB

==============================================================================
E3
==============================================================================
  component                                  batch2 MB   batch16 MB
  student fwd (retained for bwd)                 558.7       4469.2
  CE + Dice term                                 702.0       5616.0
  teacher fwd (retained)                           0.0          0.0
  Logit-KD term (NEW, OS8)                         7.3         58.3
  Logit-KD term (OLD, 512 upsampled)             466.0       3728.0
  CWD-feat term (+ projection)                     8.9         71.6
  CWD-logit term                                  10.9         87.1

  ACTIVATIONS, NEW path (total)                             10302.2
  ACTIVATIONS, OLD path (total)                             13972.0

  parameters (batch-independent):
    student params                            11.2 MB
    student grads                             11.2 MB
    SGD momentum buffer                       11.2 MB
    CWD projection (params+grad+mom)           0.6 MB
    teacher params (frozen, no grad)         105.3 MB   [INFERRED ~27.6M params]
    static subtotal                          139.4 MB

  >>> E3 NEW total (activations + static) :   10441.7 MB = 10.20 GB
  >>> E3 OLD total (activations + static) :   14111.4 MB = 13.78 GB

==============================================================================
TEACHER TRANSIENT WORKING SET — the biggest unknown, bounded analytically [INFERRED]
==============================================================================
FrozenTeacher.forward is @torch.no_grad(), so the teacher RETAINS NOTHING for backward.
Its cost is a transient working set: the largest few stage activations alive at once.
MSCAN-B embed_dims (64,128,320,512) at strides (4,8,16,32), batch 16, float32:

  stem out (stride 4)          64ch x 128x128 x16 =     64.0 MB
  stage1 (stride 4)            64ch x 128x128 x16 =     64.0 MB
  stage2 (stride 8)           128ch x  64x64  x16 =     32.0 MB
  stage3 (stride 16)          320ch x  32x32  x16 =     20.0 MB
  stage4 (stride 32)          512ch x  16x16  x16 =      8.0 MB
  LightHamHead concat @ s8    960ch x  64x64  x16 =    240.0 MB

  Peak is a few of these alive simultaneously, NOT their sum (no_grad frees eagerly).
  Bound: ~304 MB - 544 MB.
```

### Per-component, batch 16

| Component | E2 | E3 | Label |
|---|---|---|---|
| student fwd (retained for bwd) | 4,469.2 MB | 4,469.2 MB | MEASURED |
| CE + Dice term | 5,616.0 MB | 5,616.0 MB | MEASURED |
| teacher fwd (retained) | **0.0 MB** | **0.0 MB** | MEASURED — `@torch.no_grad()` |
| Logit-KD (NEW, OS8) | 58.3 MB | 58.3 MB | MEASURED |
| *Logit-KD (OLD, upsampled)* | *3,728.0 MB* | *3,728.0 MB* | MEASURED |
| CWD-feat (+ projection) | — | 71.6 MB | MEASURED |
| CWD-logit | — | 87.1 MB | MEASURED |
| static (params + grads + momentum + teacher weights) | 138.9 MB | 139.4 MB | exact |
| **total, post-B32** | **10,282 MB (10.04 GB)** | **10,442 MB (10.20 GB)** | MEASURED + exact |
| **total, pre-B32** | **13,952 MB (13.63 GB)** | **14,111 MB (13.78 GB)** | MEASURED + exact |

### The teacher — the crux, and it was never the problem

`FrozenTeacher.forward` is `@torch.no_grad()`, so **nothing is retained**. The measured retained figure is exactly `0.0 MB`. Its cost is a transient working set — MSCAN-B stage activations alive momentarily, freed eagerly:

| Stage | Size @ batch 16 |
|---|---|
| stem / stage1 (stride 4, 64 ch) | 64.0 MB each |
| stage2 (stride 8, 128 ch) | 32.0 MB |
| stage3 (stride 16, 320 ch) | 20.0 MB |
| stage4 (stride 32, 512 ch) | 8.0 MB |
| LightHamHead concat @ stride 8 (960 ch) | 240.0 MB |

Peak is a few of these alive at once, not their sum. **Bound: ~0.30–0.55 GB [INFERRED].**

### Verdict — a range, with the uncertainty attached

> **INFERRED 11–14 GB peak on CUDA at batch 16**, for both E2 and E3, from a **MEASURED 10.0–10.2 GB** retained-activation-plus-static set, plus 5–15% allocator overhead, plus a ~0.3–0.55 GB teacher transient, plus cuDNN workspace and backward temporaries which this method does not capture.

Against a 24 GB RTX 4090 that leaves roughly **10–13 GB of headroom**. The pre-B32 path would have been ~15–18 GB — **also under 24 GB**.

**I am not giving you a GO off a CPU estimate.** Settle it on the pod:
```bash
python - <<'EOF'
import torch
torch.cuda.reset_peak_memory_stats()
# ... run ~20 iterations of train_e2.py at batch 16 ...
print('peak allocated GB', torch.cuda.max_memory_allocated()/1024**3)
print('peak reserved  GB', torch.cuda.max_memory_reserved()/1024**3)
EOF
```

Run it on a 24 GB card first with `--max-iters 20`. `max_memory_reserved` is the number that decides whether it fits; `max_memory_allocated` is the one comparable to the figures above.

### What this changes for pod planning

If the pod measurement confirms it: **no 24 GB → 48 GB switch**, which also removes the Network Volume migration risk, and the 4090 is roughly **2× the FP32 throughput of an A6000 at comparable Community rates**. That is a throughput *gain* from staying put, not a compromise. **Flagged, not decided** — one measurement settles it.

---

## 4. Contract / open-questions diff

```text
 docs/IMPLEMENTATION_CONTRACT.md | 59 +++++++++++++++++++++++++++++++++++++++++
 docs/open_questions.md          | 24 +++++++++++++++++
 2 files changed, 83 insertions(+)
```

```diff
diff --git a/docs/IMPLEMENTATION_CONTRACT.md b/docs/IMPLEMENTATION_CONTRACT.md
index dcfdb93..045d1ec 100644
--- a/docs/IMPLEMENTATION_CONTRACT.md
+++ b/docs/IMPLEMENTATION_CONTRACT.md
@@ -191,6 +191,65 @@ pending the PlantSeg repo's official convention. `[empirical; ch3 Table 3.1; ctx
 | Normalization | **T²/C**, with **C = 320** (MSCAN-B stride-16 Stage-3 channel count) | `[ch3]` |
 | Projection head | training-only 1×1 conv: student **160-ch C5 → teacher 320-ch**; removed before E6/E7 via state_dict edit prior to observer insertion | `[ch3]` |
 | Ignore handling | validity mask downsampled to stride-16; channel-wise spatial softmax + KL restricted to valid locations | `[ch3]` |
+
+#### B3 — spatial grid of each distillation term `[project; B32/F8, 2026-09-01]`
+ch3 pins *which pixels* enter the Logit-KD KL ("averaged over valid pixels only") but **not which
+grid**. B32 pins the grid. This is an implementation resolution of a genuine ch3 gap, not a change
+to a `[ch3]`-traced value.
+
+| Term | Grid | Validity mask | Note |
+|---|---|---|---|
+| Logit-KD KL | **OS8 64×64** | 64×64, min-pooled | **changed by B32** — was an upsampled 512×512 |
+| `L_CWD_logit` | **OS8 64×64** | 64×64, **shared** with Logit-KD | already OS8 pre-B32; unchanged |
+| `L_CWD_feat` | stride-16 32×32 | 32×32, min-pooled | unchanged |
+| `L_CE`, `L_Dice` | full 512×512 | full-resolution target | **unchanged — the supervised path is not touched** |
+
+Both the student head and the SegNeXt LightHamHead emit logits natively at 64×64 for a 512×512
+input, so neither side is resampled. The old path upsampled both, putting **98.4%** of the KL on
+interpolants; because interpolation happens in *logit* space before the softmax
+(`softmax(interp(z)) ≠ interp(softmax(z))`), the soft targets were distorted rather than repeated.
+
+- **Valid-population cost [empirical, measured, 60 train samples]:** the Logit-KD valid population
+  is a strict subset of CE's and is **1.29% smaller in relative terms** (mean valid fraction 0.7096
+  at 64×64 vs 0.7189 at 512×512). `L_CWD_feat` at 32×32 loses 3.14%. Conservative all-valid
+  min-pooling penalises letterboxed images ~3× more than near-square ones in relative terms (1.94%
+  vs 0.62% of valid locations), but the absolute effect is under 2% everywhere; the large absolute
+  gap between those groups is **padding**, which CE sees identically.
+- **Zero-valid cells are unreachable by construction**, not merely unobserved: 0 of 60 samples, and
+  it would require aspect ratio > 64 (widest observed 2.42). No guard was added.
+- **λ_logit magnitude shift [empirical, measured — UPPER BOUND]:** the KD term's magnitude changed
+  by **1.95×** and its gradient contribution by **1.76×**. Measured against a *random synthetic*
+  teacher, the worst case for spatial smoothness, so this is an **upper bound** — a real trained
+  SegNeXt-B should shift it less. `CANNOT-VERIFY-LOCALLY`; re-measure on the pod.
+- **Consequence for the λ sweep:** the preregistered grid `{0.25, 0.5, 1, 2, 4}` is geometric with
+  ratio 2, so ~1.95× is about **one grid step**. The grid still brackets a sensible optimum and
+  needs **no re-centring** — F11 is intact — but the expected optimum sits roughly one step lower,
+  and **a λ selected under the old semantics is not transferable**. Ch4 must state that the sweep
+  ran under OS8 semantics.
+- **Machine-checkable guard `[project; B32c-2]`:** `configs/distill.py` defines
+  `LOGIT_KD_SEMANTICS = "logitkd@os8-64x64-of-512"`. Every E2/E3 checkpoint and a
+  `<stage>_run_meta.jsonl` record it. `--lambda-semantics` is optional but checked when supplied;
+  a mismatch refuses unless `--allow-semantics-mismatch` is passed, and that override is stamped
+  into both artifacts so a non-comparable run stays identifiable without the terminal.
+
+#### E2/E3 peak memory `[project; B32c-3, INFERRED from a MEASURED activation set]`
+Retained-activation bytes MEASURED via `saved_tensors_hooks` at batch 2 and scaled ×8 (the graph is
+fixed-shape, so retained bytes are exactly linear in batch); parameters/gradients/momentum exact.
+
+| | E2 | E3 |
+|---|---|---|
+| activations + static, **pre-B32** | 13.63 GB | 13.78 GB |
+| activations + static, **post-B32** | **10.04 GB** | **10.20 GB** |
+
+The **frozen teacher retains nothing** — `FrozenTeacher.forward` is `@torch.no_grad()` — so its cost
+is a transient working set bounded at **~0.3–0.55 GB**, not an activation graph. The dominant costs
+are the student's own retained activations (4.47 GB) and the CE+Dice term (5.62 GB), both at full
+512×512, both unchanged by B32 and inherent to the locked batch-16/512² recipe.
+
+**E2/E3 therefore cost only marginally more than E1 itself.** Adding allocator overhead (+5–15%),
+the teacher transient, cuDNN workspace and backward temporaries gives an **INFERRED 11–14 GB** peak
+on CUDA at batch 16. This is **not a GO** — settle it on the pod with
+`torch.cuda.max_memory_allocated()`.
 | E3 total loss | `L_CE + L_Dice + λ_logit·L_LogitKD + 50·L_CWD_feat + 3·L_CWD_logit` | `[ch3]` |
 | Optional control | α_CWD sensitivity sweep {25, 50, 100} at 1 seed — optional, else fixed 50 per Shu 2021 | `[ch3 §C]` |
 
diff --git a/docs/open_questions.md b/docs/open_questions.md
index 0b759c6..2b734d7 100644
--- a/docs/open_questions.md
+++ b/docs/open_questions.md
@@ -85,6 +85,30 @@ valid E1 run but not a byte-reproduction. It must be reported as resumed if it p
 result. No code change is proposed to make order recoverable: doing so would require persisting
 sampler position and is not required by any source. `[project; B31-2]`
 
+### D28 — the λ_logit semantics boundary — ✅ RESOLVED `[project]` (B32/F8 + B32c-2)
+ch3 pins *which pixels* enter the Logit-KD KL but not *which grid*. B32 pinned it to the head's
+native OS8 64×64. Because λ_logit weights that term, **its numeric value is only meaningful relative
+to the grid the term is computed on**, and the move changed the term's magnitude by **1.95×**
+(MEASURED against a synthetic teacher, therefore an **upper bound**; gradient contribution 1.76×).
+
+The preregistered grid `{0.25, 0.5, 1, 2, 4}` is geometric with ratio 2, so ~1.95× is about **one
+grid step**. Resolution: **the grid is NOT re-centred.** It still brackets a sensible optimum, and
+ch3 already handles a boundary winner ("reported as such rather than the grid being extended"), so
+F11's preregistration is intact. The optimum is expected roughly one step lower than it would have
+been under the old semantics.
+
+**A λ selected under one semantics is not consumable under the other.** Guarded, not just
+documented: `configs/distill.py` defines `LOGIT_KD_SEMANTICS = "logitkd@os8-64x64-of-512"` (and
+records the superseded `"logitkd@full-512x512-upsampled"`); every E2/E3 checkpoint and a
+`<stage>_run_meta.jsonl` carry it; `--lambda-semantics` is optional but checked when declared, and a
+mismatch refuses unless `--allow-semantics-mismatch` is passed — with that override stamped durably
+into both artifacts. Declaring is optional by design: the sweep runs *produce* the tag rather than
+consume it, so requiring it there would add friction at the point of lowest risk; the risk is a λ
+read out of a Ch4 table months later and passed to a re-run.
+
+**Ch4 obligation:** state that the λ sweep ran under OS8 semantics, and report λ with its tag.
+`[ch3 §C.1; project; B32/B32c]`
+
 ### D27 — scaffold check coverage under resume — ✅ RESOLVED `[project]`
 B31c V1 asked whether a resumed run can report success while exercising fewer checks than a fresh
 run. It could, in two ways, both now closed or made visible:
```

### `e1_launch_runbook_v2.md` — deliberately NOT touched

B32c-3 changes **E2/E3** memory. The runbook is **E1-scoped**, and E1's own footprint (student activations + CE/Dice) is unchanged by B32 — the KD term does not exist in E1. Its GPU guidance is therefore unaffected, and per your ruling I did not edit a file to demonstrate diligence.

---

## 5. Provenance

### Commits
```text
99108e1 B32c-4: record the OS8 pin, the lambda semantics boundary, and E2/E3 peak memory
0d437c6 B32c-2: lambda_logit semantics tag — stamp plus opt-in assertion
23bce3e B32 audit trail: KD resolution report
1529a54 B32-6: standing gate for KD spatial resolution, scripts/smoke_kd_resolution.py
6a6113b B32/F8: compute Logit-KD at the head's native OS8 grid, not on 512x512 upsamples
```

### Files changed across B32 + B32c
```text
 configs/distill.py              |  17 ++
 docs/IMPLEMENTATION_CONTRACT.md |  59 ++++
 docs/open_questions.md          |  24 ++
 reports/b32_kd_resolution.md    | 625 ++++++++++++++++++++++++++++++++++++++++
 scripts/smoke_distill.py        |  10 +-
 scripts/smoke_kd_resolution.py  | 174 +++++++++++
 scripts/smoke_losses_metrics.py |   7 +-
 src/training/__init__.py        |   2 +
 src/training/losses.py          |  30 +-
 src/training/train_distill.py   | 116 +++++++-
 10 files changed, 1041 insertions(+), 23 deletions(-)
```

### Git status
```text
 M docs/reference/reference.pdf
```

### `docs/reference/reference.pdf`
```text
(no commit in this series touches it)
```

Never staged. mtime **`Jun 27 13:28`**, unchanged since before B30.

### Confirmations

- **Explicit-path staging only** — every `git add` named exact files; `git add -A` / `git add .` never used.
- **Every commit leaves `train_e1.py --dry-run` at `RESULT: PASS (6/6 checks exercised, 0 skipped)`** — measured in a throwaway worktree outside the repo, 5/5.
- **Worktree cleanup verified**: `git worktree list` shows only the main worktree.
- **E1 path untouched** — `train_e1.py`, `dataset.py`, `transforms.py`, `seeds.py`, `metrics.py` unmodified in this series.
- **No teacher fine-tuning, no `weights/` access, no MMSeg build, no checkpoint loaded.**
- **No manuscript or `docs/reference/**` edits.**
- **All locked values unchanged**: T_Logit=4, T_CWD=4, α=50, β=3, T²/C with C=320, λ grid, projection 160→320, and the whole E1 recipe. Asserted by `smoke_kd_resolution.py` (24/24).
- **No training, GPU, downloads, or installs.** CPU only.
- **No co-author trailers.**

