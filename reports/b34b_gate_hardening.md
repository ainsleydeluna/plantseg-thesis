# B34b — E1 pre-flight gate hardening (CE class-weight assertions)

**Type:** code change (first in the B34/B35 series). **Status:** APPLIED and VERIFIED. **Date:** 2026-09-05.
**Closes:** the [B34](b34_config_reconciliation.md) §A5 gap.

**Not committed.** `docs/reference/reference.pdf` was not opened, read, hashed, copied, or staged.
No training, no GPU, no downloads. `src/**`, `configs/**`, `requirements*`, and both contracts are
untouched. `reports/e1_class_weights.json` is **unmodified** (`git status --porcelain` on it returns
nothing).

**Precondition check.** Every asserted value was **re-derived from `reports/e1_class_weights.json`**,
not copied from the B34 report or from the task prompt. **No discrepancy was found**; the artifact
agrees with both. Had it disagreed, the artifact would have won and this task would have stopped.

---

## 1. Which file was hardened, and why

> ### `scripts/preflight_e1.py` — Tier 1 added there, **not** in `verify_env.py`.

The requirement is *"must NO-GO the gate … do not warn-and-continue."* That decides it:

**`verify_env.py` structurally cannot NO-GO.** Its `main()` ends in an unconditional `return 0`, and
its own docstring calls it *"B20a — Strict-G0 compute-platform verification (**diagnostic only; NOT
training**)"*. `preflight_e1.py` already documents the consequence at
[`:225`](../scripts/preflight_e1.py#L225): *"verify_env returns 0 regardless of verdict, so the RESULT
line is the authority."* A hard assertion placed in a script that never fails would be
warn-and-continue **by construction** — exactly what the brief forbids. It is also off-theme:
`verify_env.py` verifies the compute platform, not repository artifacts.

`preflight_e1.py` has native NO-GO semantics: a stage returns `False` → `first_failure` is set → the
loop breaks → `main()` returns `1`. **One file was chosen; the work was not split across both.**

**Docstring honesty.** The file previously claimed it *"ORCHESTRATES existing scripts and
re-implements none of their checks."* That is now amended rather than left quietly false. The new
stage re-implements **no other script's check** — no script anywhere asserts anything about the class
weights, which is precisely the gap B34 §A5 identified — but it is genuinely in-process, and the
docstring now says so.

**Stage position: 1 of 5.** Cheapest check in the gate (one JSON read), zero dependencies, and it
guards an artifact `train_e1.py` hard-depends on. All four existing stages keep their relative order
and their labels were renumbered `/4 → /5` per the brief's instruction about the hard-coded count.

*Alternative rejected:* appending as 5/5. Stage 5 (`--dry-run`) already calls `load_ce_weights()`, so
appending would place the millisecond check **after** the expensive one for no benefit.

---

## 2. Assertions added — each value re-derived from the artifact

| id | assertion | value re-derived from `e1_class_weights.json` |
|---|---|---|
| **T1a** | `len(weights) == 116`, `len(pixel_counts) == 116` | `116` |
| **T1b** | all finite; `dtype` is `float32` after `torch.tensor(w, dtype=torch.float32)` | finite `True`; `torch.float32` |
| **T1c** | **PROVENANCE fingerprint** — `(sorted[57]+sorted[58])/2 == 1.0` | `sorted[57] = 0.996273409152`, `sorted[58] = 1.003726590848` → **`1.000000000000`** |
| **T1d** | `argmin == 0` | `0` |
| **T1d** | `argmax == 42` | `42` |
| **T1d** | `argmax(weights) == argmin(pixel_counts)` (re-derived cross-check, not a constant) | rarest class `42`, `N = 54,975` |
| **T1d** | `w[42]/w[0] == sqrt(N_0/N_42)`, ratio computed **at runtime** from `pixel_counts` | `259.551929472253 == 259.551929472253` (`N_0 = 3,703,512,045`, `N_42 = 54,975`) |
| **T1e** | `w[0]` is the small background weight, not a `1.0` placeholder | `w[0] = 0.036954205368266338` (background = 80.82% of pixels) |
| *(added)* | **index alignment** — every weight reproduces from `pixel_counts` | max deviation **0.000e+00** over all 116 |

**T1c uses the explicit two-element form.** Not `np.median`, not `torch.median`. The inline comment
records why, so a future reader does not "fix" it:

```python
    # DO NOT "SIMPLIFY" THIS TO np.median / torch.median. num_classes is 116, i.e. EVEN, and the
    # conventions disagree: torch.median returns the LOWER middle (0.996273398399 here) and would
    # NO-GO a perfectly correct artifact. The artifact was median-normalized with the
    # average-of-the-two-middles convention, so the two-element form below is the only correct one.
```

It is labelled **PROVENANCE, not correctness**, in the code, in the comment, and in the PASS line —
because B34 §4 measured that a uniform scale on CE weights cancels in
`F.cross_entropy(reduction='mean')` (gradient-norm delta `9.77e-08`, below float32 eps). The median
value pins *which generator produced the artifact*; it cannot affect training.

**The ratio is not hardcoded.** `expected_ratio = math.sqrt(counts[0] / counts[argmax])` is computed
from the artifact's own histogram each run. It is scale-invariant — surviving any renormalization —
yet a pure function of the pixel distribution, so it breaks immediately if the class distribution
changes. It is the strongest single scalar in the artifact.

**Index alignment was added beyond T1a–T1e.** It re-derives all 116 weights from `pixel_counts` and
compares elementwise. This closes the last unasserted item from B34 §A5 and costs microseconds.

**Failure behaviour: NO-GO, never warn-and-continue.** Every branch returns through one `fail()`
helper that prints the failing assertion *and what the artifact actually contained*, then returns
`False`, which stops the gate and exits `1`. `check_class_weights()` **never raises** — a malformed
artifact is a NO-GO verdict, not a traceback.

---

## 3. Full diff

### 3.1 `scripts/preflight_e1.py` (modified)

```diff
@@ -3,16 +3,30 @@
 
 ONE command, ONE verdict. Run this on the pod immediately before launching E1.
 
-It ORCHESTRATES existing scripts and re-implements none of their checks:
-
-  1. scripts/verify_env.py            — platform, pinned versions, the B31-3 CUDA compute-capability
+It ORCHESTRATES existing scripts and re-implements none of their checks, with ONE exception noted
+at stage 1 below:
+
+  1. CE class-weight artifact       — IN-PROCESS (B34b). The only stage that is not a delegation.
+                                        No other script asserts anything about
+                                        reports/e1_class_weights.json, which is exactly the gap the
+                                        B34 audit found (its section A5): train_e1.py loads the file
+                                        unconditionally and aborts without it, but nothing verified
+                                        its CONTENT. Runs first because it is the cheapest check
+                                        here (one JSON read) and guards an artifact E1 hard-depends
+                                        on.
+  2. scripts/verify_env.py            — platform, pinned versions, the B31-3 CUDA compute-capability
                                         gate (sm<=9.0) and the B31-4 dataset verification.
-  2. scripts/smoke_dataloader.py      — one train + one val batch: shapes, dtypes, label range.
-  3. scripts/smoke_aug_stochasticity.py — augmentation varies across epochs, val is bit-identical,
+  3. scripts/smoke_dataloader.py      — one train + one val batch: shapes, dtypes, label range.
+  4. scripts/smoke_aug_stochasticity.py — augmentation varies across epochs, val is bit-identical,
                                         seeds reproduce across processes. On the pod this doubles as
                                         the R5 PINNED-STACK re-measurement: the observed seed
                                         sequence is compared against the B30 §9 baseline.
-  4. src/training/train_e1.py --dry-run — the end-to-end scaffold floor.
+  5. src/training/train_e1.py --dry-run — the end-to-end scaffold floor.
+
+Stage 1 is a REPO-ARTIFACT check, not a dataset check: it proves the committed weight vector is the
+one B18a computed. It does NOT prove the pod's dataset matches the histogram those weights came from
+— identical split counts are compatible with a different pixel histogram. That binding is the job of
+scripts/verify_class_weights_pod.py, which is ADVISORY and deliberately NOT a stage here.
 
 Stops at the first hard failure. Exits non-zero on NO-GO.
 
@@ -29,6 +43,8 @@ from __future__ import annotations
 
 import argparse
 import ast
+import json
+import math
 import os
 import re
 import subprocess
@@ -39,6 +55,11 @@ from pathlib import Path
 REPO = Path(__file__).resolve().parents[1]
 PY = sys.executable
 
+# B34b: the CE class-weight artifact the E1 loop loads unconditionally (train_e1.py:49,55,293).
+CLASS_WEIGHTS_JSON = REPO / "reports" / "e1_class_weights.json"
+EXPECTED_NUM_CLASSES = 116          # configs/data.py DATA["num_classes"]; background = index 0
+BACKGROUND_INDEX = 0
+
 # B30 §9 / B31-5: the first two per-sample augmentation seeds of pass 1 and pass 2 over the fixed
 # probe subset, measured on the development stack. smoke_aug_stochasticity.py prints these.
 B30_SEED_BASELINE = [[1608637542, 1273642419], [1935803228, 787846414]]
@@ -63,8 +84,138 @@ def _run(cmd, label):
     return r, time.time() - t0
 
 
+def _banner(label):
+    """Stage banner for an IN-PROCESS stage. Mirrors _run()'s header so the transcript reads the
+    same whether a stage delegates to a subprocess or not."""
+    print("\n" + "=" * 78)
+    print(f"STAGE {label}")
+    print("=" * 78)
+    print(f"# in-process check of {CLASS_WEIGHTS_JSON.relative_to(REPO).as_posix()}\n", flush=True)
+    return time.time()
+
+
+def check_class_weights(path=CLASS_WEIGHTS_JSON):
+    """B34b Tier 1. Assert the CE class-weight artifact is the one B18a computed.
+
+    Returns (ok, detail). NEVER raises: a malformed artifact is a NO-GO, not a traceback.
+    `path` is a parameter so the check can be exercised against a deliberately corrupted copy
+    without touching the real artifact.
+    """
+    import torch                       # lazy: keeps the gate's own import cost near zero
+
+    def fail(assertion, found):
+        print(f"  [FAIL] {assertion}")
+        print(f"         artifact contained: {found}")
+        return False, f"{assertion} (artifact contained: {found})"
+
+    try:
+        with open(path, encoding="utf-8") as f:
+            doc = json.load(f)
+    except Exception as e:                                            # noqa: BLE001
+        return fail("artifact unreadable", f"{type(e).__name__}: {e}")
+
+    for key in ("weights", "pixel_counts", "total_nonignore_pixels"):
+        if key not in doc:
+            return fail(f"artifact key {key!r} present", f"keys={sorted(doc)}")
+    w, counts = doc["weights"], doc["pixel_counts"]
+
+    # --- T1a: length ------------------------------------------------------------------------
+    if len(w) != EXPECTED_NUM_CLASSES:
+        return fail(f"T1a len(weights) == {EXPECTED_NUM_CLASSES}", f"len={len(w)}")
+    if len(counts) != EXPECTED_NUM_CLASSES:
+        return fail(f"T1a len(pixel_counts) == {EXPECTED_NUM_CLASSES}", f"len={len(counts)}")
+
+    # --- T1b: finite + float32 after the conversion train_e1.load_ce_weights performs ---------
+    t = torch.tensor(w, dtype=torch.float32)
+    if not bool(torch.isfinite(t).all()):
+        bad = [i for i, v in enumerate(w) if not math.isfinite(v)]
+        return fail("T1b all weights finite", f"non-finite at indices {bad[:8]}")
+    if t.dtype is not torch.float32:
+        return fail("T1b dtype is float32", f"dtype={t.dtype}")
+
+    # --- T1c: PROVENANCE fingerprint -- NOT a correctness check --------------------------------
+    # This records WHICH generator produced the artifact; it cannot affect training. A uniform
+    # scale on CE weights cancels in F.cross_entropy(reduction='mean'), which normalizes by the
+    # sum of per-pixel weights (measured in B34 section 4: gradient-norm delta 9.77e-08, below
+    # float32 eps). Kept because it pins provenance, not because the number matters numerically.
+    #
+    # DO NOT "SIMPLIFY" THIS TO np.median / torch.median. num_classes is 116, i.e. EVEN, and the
+    # conventions disagree: torch.median returns the LOWER middle (0.996273398399 here) and would
+    # NO-GO a perfectly correct artifact. The artifact was median-normalized with the
+    # average-of-the-two-middles convention, so the two-element form below is the only correct one.
+    srt = sorted(w)
+    lo, hi = srt[EXPECTED_NUM_CLASSES // 2 - 1], srt[EXPECTED_NUM_CLASSES // 2]
+    median_two_element = (lo + hi) / 2
+    if not math.isclose(median_two_element, 1.0, abs_tol=1e-9):
+        return fail("T1c median-normalization fingerprint == 1.0 (two-element form)",
+                    f"(sorted[57]+sorted[58])/2 = {median_two_element!r}")
+
+    # --- T1d: CORRECTNESS + dataset binding ----------------------------------------------------
+    argmin, argmax = int(t.argmin()), int(t.argmax())
+    if argmin != BACKGROUND_INDEX:
+        return fail(f"T1d argmin == {BACKGROUND_INDEX} (background is the most frequent class)",
+                    f"argmin={argmin}, w[{argmin}]={w[argmin]!r}")
+    if argmax != 42:
+        return fail("T1d argmax == 42", f"argmax={argmax}, w[{argmax}]={w[argmax]!r}")
+    # Cross-check the recorded argmax against the histogram rather than trusting the constant:
+    # w_c is proportional to 1/sqrt(N_c), so the largest weight must be the rarest class.
+    rarest = min(range(EXPECTED_NUM_CLASSES), key=lambda c: counts[c])
+    if rarest != argmax:
+        return fail("T1d argmax(weights) == argmin(pixel_counts)",
+                    f"argmax={argmax} but rarest class is {rarest} (N={counts[rarest]})")
+    # The ratio is scale-invariant (it survives ANY renormalization) yet is a pure function of the
+    # pixel histogram, so it breaks the moment the class distribution changes. Re-derived here from
+    # the artifact's own pixel_counts -- deliberately NOT hardcoded.
+    expected_ratio = math.sqrt(counts[BACKGROUND_INDEX] / counts[argmax])
+    actual_ratio = w[argmax] / w[BACKGROUND_INDEX]
+    if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-9):
+        return fail("T1d w[42]/w[0] == sqrt(N_0/N_42)",
+                    f"ratio={actual_ratio!r} vs sqrt({counts[BACKGROUND_INDEX]}/"
+                    f"{counts[argmax]})={expected_ratio!r}")
+
+    # --- T1e: index 0 is the small background weight, not a 1.0 placeholder --------------------
+    # 1.0 is exactly what an absent/placeholder class receives (it equals the median), so a
+    # background weight at or near 1.0 means the vector was NOT computed over the real train set.
+    if not w[BACKGROUND_INDEX] < 0.5:
+        return fail("T1e w[0] is the small background weight, not a 1.0 placeholder",
+                    f"w[0]={w[BACKGROUND_INDEX]!r}")
+
+    # --- index alignment: reproduce every weight from the artifact's own histogram --------------
+    # Closes the remaining B34 A5 item. w_c = sqrt(total / (num_classes * N_c)), median-normalized.
+    total = doc["total_nonignore_pixels"]
+    if sum(counts) != total:
+        return fail("sum(pixel_counts) == total_nonignore_pixels",
+                    f"sum={sum(counts)} vs total={total}")
+    raw = [math.sqrt(total / (EXPECTED_NUM_CLASSES * c)) if c else float("inf") for c in counts]
+    rs = sorted(raw)
+    rmed = (rs[EXPECTED_NUM_CLASSES // 2 - 1] + rs[EXPECTED_NUM_CLASSES // 2]) / 2
+    worst = max(range(EXPECTED_NUM_CLASSES), key=lambda c: abs(raw[c] / rmed - w[c]))
+    worst_err = abs(raw[worst] / rmed - w[worst])
+    if worst_err > 1e-9:
+        return fail("index alignment: weights reproduce from pixel_counts",
+                    f"largest deviation {worst_err:.3e} at class {worst}")
+
+    print(f"  [PASS] T1a  len(weights) == {EXPECTED_NUM_CLASSES}")
+    print(f"  [PASS] T1b  all finite; dtype={t.dtype}")
+    print(f"  [PASS] T1c  provenance fingerprint (sorted[57]+sorted[58])/2 == "
+          f"{median_two_element:.12f}  [NOT a correctness check]")
+    print(f"  [PASS] T1d  argmin={argmin} argmax={argmax}; "
+          f"w[{argmax}]/w[{BACKGROUND_INDEX}] = {actual_ratio:.12f} == "
+          f"sqrt({counts[BACKGROUND_INDEX]}/{counts[argmax]})")
+    print(f"  [PASS] T1e  w[0] = {w[BACKGROUND_INDEX]:.12f} (background, {100.0 * counts[0] / total:.2f}% of pixels)")
+    print(f"  [PASS] index alignment: all {EXPECTED_NUM_CLASSES} weights reproduce "
+          f"(max deviation {worst_err:.3e})")
+    return True, f"{EXPECTED_NUM_CLASSES} weights verified; ratio {actual_ratio:.6f}"
+
+
+def stage_class_weights(_dev):
+    t0 = _banner("1/5 — CE class-weight artifact (B34b Tier 1)")
+    ok, detail = check_class_weights()
+    return ok, detail, time.time() - t0
+
+
 def stage_verify_env(dev_rehearsal):
-    r, secs = _run([PY, str(REPO / "scripts" / "verify_env.py")], "1/4 — verify_env.py")
+    r, secs = _run([PY, str(REPO / "scripts" / "verify_env.py")], "2/5 — verify_env.py")
     # verify_env returns 0 regardless of verdict, so the RESULT line is the authority.
     line = next((x for x in reversed(r.stdout.splitlines()) if x.startswith("RESULT:")), "")
     verdict = line.split("RESULT:", 1)[1].split("|")[0].strip() if line else "UNKNOWN"
@@ -77,14 +228,14 @@ def stage_verify_env(dev_rehearsal):
 
 def stage_dataloader(_dev):
     r, secs = _run([PY, str(REPO / "scripts" / "smoke_dataloader.py")],
-                   "2/4 — smoke_dataloader.py")
+                   "3/5 — smoke_dataloader.py")
     ok = r.returncode == 0 and "[PASS] dataloader smoke test" in r.stdout
     return ok, f"exit={r.returncode}", secs
 
 
 def stage_stochasticity(_dev):
     r, secs = _run([PY, str(REPO / "scripts" / "smoke_aug_stochasticity.py")],
-                   "3/4 — smoke_aug_stochasticity.py  (R5 pinned-stack seed re-measurement)")
+                   "4/5 — smoke_aug_stochasticity.py  (R5 pinned-stack seed re-measurement)")
     if r.returncode != 0 or "RESULT: PASS" not in r.stdout:
         return False, f"exit={r.returncode}, gate did not pass", secs
 
@@ -109,7 +260,7 @@ def stage_stochasticity(_dev):
 
 def stage_dryrun(_dev):
     r, secs = _run([PY, str(REPO / "src" / "training" / "train_e1.py"), "--dry-run"],
-                   "4/4 — train_e1.py --dry-run")
+                   "5/5 — train_e1.py --dry-run")
     line = next((x for x in reversed(r.stdout.splitlines()) if x.startswith("RESULT:")), "")
     print(f"\n--- final line: {line!r} ---")
     if r.returncode != 0:
@@ -130,6 +281,7 @@ def stage_dryrun(_dev):
 
 
 STAGES = [
+    ("class_weights", stage_class_weights),
     ("verify_env", stage_verify_env),
     ("smoke_dataloader", stage_dataloader),
     ("smoke_aug_stochasticity", stage_stochasticity),
```

**Additive-only confirmation.** The diff contains **no deletion of any assertion**. The only removed
lines are (a) docstring lines replaced by an expanded version and (b) four stage-label string
literals renumbered `/4 → /5`. Every pre-existing check keeps its logic, its order relative to the
others, and its pass/fail criteria.

### 3.2 `scripts/verify_class_weights_pod.py` (new, 261 lines)

New file — see §5 for the design. Not reproduced in full here; it is committed alongside this report
and is the only other file added.

---

## 4. Verification

### 4.1 Run 1 — the gate on this CPU box

`python scripts/preflight_e1.py` → **exit 1 (NO-GO)**, as predicted.

```
==============================================================================
STAGE 1/5 — CE class-weight artifact (B34b Tier 1)
==============================================================================
# in-process check of reports/e1_class_weights.json

  [PASS] T1a  len(weights) == 116
  [PASS] T1b  all finite; dtype=torch.float32
  [PASS] T1c  provenance fingerprint (sorted[57]+sorted[58])/2 == 1.000000000000  [NOT a correctness check]
  [PASS] T1d  argmin=0 argmax=42; w[42]/w[0] = 259.551929472253 == sqrt(3703512045/54975)
  [PASS] T1e  w[0] = 0.036954205368 (background, 80.82% of pixels)
  [PASS] index alignment: all 116 weights reproduce (max deviation 0.000e+00)
```

Final summary:

```
  stage                     result       secs  detail
  class_weights             PASS          6.8  116 weights verified; ratio 259.551929
  verify_env                FAIL         28.6  verdict=PARTIAL — needs PASS (CUDA GPU, sm<=9.0, dataset OK)
  smoke_dataloader          SKIPPED         -  not reached
  smoke_aug_stochasticity   SKIPPED         -  not reached
  train_e1 --dry-run        SKIPPED         -  not reached

VERDICT: NO-GO — first failing stage: verify_env.
```

> **The new class-weight stage PASSES before the CUDA stop**, which is exactly the required outcome:
> the gate reaches stage 2 and NO-GOs on platform grounds (`torch 2.9.1+cpu`, `cuda_available: False`),
> not on the artifact. The 6.8 s is almost entirely the lazy `import torch`; the assertions themselves
> are sub-millisecond.

### 4.2 Run 2 — deliberately corrupted temp copy

A copy with `weights` truncated to 115 entries was written **to the session scratchpad**, outside the
repository. The real artifact was not modified — confirmed by `git status --porcelain
reports/e1_class_weights.json` returning nothing, before and after.

```
temp copy written to scratchpad (real artifact untouched):
  …/scratchpad/corrupt_wrong_length.json

--- check_class_weights(corrupted copy) ---
  [FAIL] T1a len(weights) == 116
         artifact contained: len=115

returned ok    : False
returned detail: T1a len(weights) == 116 (artifact contained: len=115)

>>> the check FIRED correctly on a wrong-length artifact
```

The failure message names **which assertion failed** and **what the artifact contained**, as required.
Because `stage_class_weights` returns that `False` straight into the stage loop, this would set
`first_failure` and NO-GO the gate at stage 1 with exit 1.

### 4.3 Constraint check

| constraint | status |
|---|---|
| Additive only; no assertion weakened, reordered, removed | ✅ §3.1 |
| No computed metric or training value changes | ✅ `src/**` and `configs/**` untouched; nothing in the training or metric path was edited |
| Stage numbering / output format / exit codes match existing style | ✅ `_banner` mirrors `_run`'s header; stage tuple signature `(ok, detail, secs)` unchanged; exit codes unchanged |
| Hard-coded stage count updated | ✅ four labels `/4 → /5` |
| Tier 2 implemented but **not run** | ✅ compile-checked only (`py_compile`) |

---

## 5. Tier 2 design and gating recommendation

**`scripts/verify_class_weights_pod.py`** — recomputes the full train-split mask pixel histogram and
asserts **exact** equality against the artifact's `pixel_counts` (116 classes) and
`total_nonignore_pixels` (`4,582,656,621` over `5,367` masks). Emits
`reports/e1_class_weights_pod_verify.md`. Invocable standalone; **not** in `STAGES`.

**Runtime: ~1–2 min single-threaded warm; budget ~5 min cold on a network volume.** Basis stated in
the script's own docstring: a 60-mask sample timed at 0.009 s/mask on the development box extrapolates
to ~49 s over 5,367 masks; that sample skewed small on pixel count (3.34e9 projected vs 4.58e9
recorded), so the pixel-scaled estimate is ~65–70 s, with headroom for cold first reads.

### Why split-count matching is insufficient (stated in the script and in its report)

`verify_env.py` §17b compares the **number** of image/mask pairs per split against
`configs/data.py` `SPLIT_SIZES`. That is a **cardinality** check. Identical file counts are fully
compatible with a different pixel histogram — a re-exported, re-rasterized, differently-compressed, or
partially-corrupted mask set of the same cardinality passes it unchanged. The CE weights are a
function of the per-class **pixel histogram** over ~4.58e9 pixels, not of file count. **Count equality
is necessary but not sufficient.** This script is the only check in the repository that establishes the
binding.

### Recommendation: ADVISORY, bounded — as ruled

Gating was rejected for the stated reason: the check needs the dataset mounted and reads every train
mask, so on a cold network volume a hard gate risks a **false NO-GO on a correct launch** from storage
flakiness. But advisory-by-default fails open, so all three bounds are implemented:

| bound | implementation |
|---|---|
| **(a)** named runbook step | recommended wording below — **runbook NOT edited** |
| **(b)** non-zero exit + `MISMATCH` on mismatch | `print("MISMATCH: …", file=sys.stderr)` then `return 1`; missing artifact/dataset returns `2`. Advisory means a human may override, **not** that the script shrugs. |
| **(c)** skip is legible | a skipped script emits no report, so the emitted report carries: *"If this report is absent from a run's artifact set, E1 was launched WITHOUT class-weight histogram verification. Absence is not a pass — it is the record of a skip."* |

`--limit N` exists for plumbing smoke only and refuses to produce a verdict (`RESULT: SMOKE ONLY`), so
a partial pass can never be mistaken for a MATCH.

### Recommended runbook line — **wording only, not applied**

Insert into [`reports/e1_runpod_launch_runbook.md`](e1_runpod_launch_runbook.md) §9 *"Final launch
approval checklist"*, immediately after the existing `train_e1.py --dry-run` checkbox:

```markdown
- [ ] `scripts/verify_class_weights_pod.py` → **MATCH** (advisory: on `MISMATCH` do not launch
      without an explicit recorded override). If skipped, record that E1 was launched **without**
      class-weight histogram verification.
```

Placed there because §9 already sequences `verify_env.py → PASS` and `train_e1.py --dry-run → RESULT:
PASS`, and this check logically follows both: it presumes the dataset is mounted and verified by
cardinality, then binds the weights to it.

---

## 6. Dropped, and why

**T1f — device placement. DROPPED, per the pre-ruling.** Recorded here rather than silently omitted.

The check would have asserted that the weight tensor handed to the criterion is on the training
device. To observe `criterion.ce.weight.device` the gate would have to construct
`CombinedCEDiceLoss(weight=load_ce_weights().to(dev), …)`, and therefore resolve a device and import
the training stack — **exceeding a pre-flight's remit**, which is to check preconditions cheaply, not
to instantiate training objects.

It is also already covered at runtime. [`train_e1.py:295-296`](../src/training/train_e1.py#L295) prints
the device of the actual buffer on every run:

```python
    print(f"[loss] CombinedCEDiceLoss(weight=len{weights.numel()}, ignore_index={IGNORE_INDEX}) "
          f"weight_on={criterion.ce.weight.device}")
```

and the mechanism is structural rather than incidental: `weight` is a registered buffer
([`losses.py:49`](../src/training/losses.py#L49)), so it follows `.to(device)` by construction. The
gate asserts the artifact's **content**; the loop reports the tensor's **placement**. Faking the latter
from the gate would have added confidence without adding coverage.

**Nothing else was dropped.** T1a–T1e are all implemented, plus one addition (index alignment) beyond
the brief.

---

## 7. Commit plan (E1) — **NOT executed**

Nothing is staged. `git status` shows the working tree exactly as this task left it.

### Recommendation: **ONE commit, not two**

The two scripts and the three reports are a **single logical change**: B34 identified the gap, B35
adjudicated the surrounding label state, and B34b closes the gap in code. Splitting `scripts/` from
`reports/` would produce a commit whose code has no rationale in the tree and a commit whose rationale
describes code that is not there — and would leave an intermediate HEAD where the gate references
`reports/e1_class_weights_pod_verify.md` semantics that no committed document explains. A reviewer
bisecting to either half would see an incomplete change.

The counter-argument — that reports are documentation and code is behaviour, so they should be
separable — does not apply here, because these reports **are** the change's specification and
verification record, not commentary added afterwards.

### Exact staging — five paths, explicitly, no wildcards

```bash
git add scripts/preflight_e1.py scripts/verify_class_weights_pod.py reports/b34_config_reconciliation.md reports/b35_eligibility_and_stale_labels.md reports/b34b_gate_hardening.md
```

> **`docs/reference/reference.pdf` is NOT staged and must never be.** It is listed in `git status` as
> modified, is pre-existing dirt unrelated to this work, and is explicitly protected by `AGENTS.md`.
> The command above names five paths and uses **no** `-A`, no `.`, and no wildcard, so the PDF cannot
> be swept in. Verify with `git status --short` before committing: the PDF must still show as ` M`
> (unstaged), never `M ` or `A `.

### Proposed commit message

```text
B34b: harden the E1 pre-flight gate with CE class-weight assertions

Closes the gap found by the B34 audit (its section A5): train_e1.py loads
reports/e1_class_weights.json unconditionally and aborts without it, but nothing
verified the file's CONTENT. Neither verify_env.py nor preflight_e1.py asserted
anything about the class weights by name.

Adds stage 1/5 to scripts/preflight_e1.py, in-process and fail-closed:
  - length 116, all finite, float32 after tensor conversion
  - argmin == 0, argmax == 42, and w[42]/w[0] == sqrt(N_0/N_42) with the ratio
    re-derived at runtime from the artifact's own pixel_counts, not hardcoded
  - w[0] is the small background weight (0.036954), not a 1.0 placeholder
  - index alignment: all 116 weights reproduce from pixel_counts (deviation 0.0)
  - a provenance fingerprint on the median-normalization, labelled as such

The median assertion uses the explicit (sorted[57]+sorted[58])/2 form. 116 is
even and torch.median takes the lower middle (0.996273398399), which would NO-GO
a correct artifact; the inline comment records this so it is not "simplified"
later. It is provenance, not correctness: a uniform scale on CE weights cancels
in F.cross_entropy(reduction='mean'), measured at a gradient-norm delta of
9.77e-08, below float32 eps.

Adds scripts/verify_class_weights_pod.py (ADVISORY, not a gate stage): recomputes
the full train-split pixel histogram on the pod and asserts exact equality against
the artifact. Split-count matching is necessary but not sufficient — identical
file cardinality is compatible with a different pixel histogram. Exits non-zero
and prints MISMATCH on failure; its report records that absence of the report
means E1 was launched without histogram verification.

T1f (device placement) was deliberately dropped: asserting it would require the
gate to construct the criterion, and train_e1.py:295-296 already prints the
buffer's device at runtime.

Additive only. No existing assertion weakened, reordered, or removed; no computed
metric or training value changes. src/, configs/, requirements* and both contracts
are untouched.

Reports: B34 (config reconciliation), B35 (eligibility + stale labels), B34b (this).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

## 8. Post-commit governed-path status (E2)

`EVALUATION_CONTRACT.md` §7.1 governs: `src/**` · `configs/**` · `scripts/**` · `requirements*` ·
`docs/EVALUATION_CONTRACT.md` · `docs/IMPLEMENTATION_CONTRACT.md`.

**Current state (before the commit): governed paths are DIRTY.** `scripts/preflight_e1.py` is
modified and `scripts/verify_class_weights_pod.py` is untracked. **No `official` artifact may be
produced right now.**

**After the commit above: governed paths are CLEAN, and official artifacts may be produced again.**
Both `scripts/` entries become committed; `src/**`, `configs/**`, `requirements*`, and both contracts
were never touched.

**What remains dirty afterwards — and why it does not matter:**

| path | state after commit | governed? |
|---|---|---|
| `docs/reference/reference.pdf` | still ` M` (unstaged, pre-existing) | **No** — `docs/reference/**` is not in the §7.1 list |

That is the only residual. It is protected, deliberately never staged, and outside the governed set,
so it does not block official-artifact production. This is the scoped rule working as intended:
`AGENTS.md` is explicit that the repository *"is intentionally never globally clean"* and that a
blanket cleanliness requirement must never be introduced.

**Verify after committing:**

```bash
git status --porcelain src/ configs/ scripts/ requirements* docs/EVALUATION_CONTRACT.md docs/IMPLEMENTATION_CONTRACT.md
```

Empty output ⇒ governed paths clean ⇒ official artifacts permitted.
