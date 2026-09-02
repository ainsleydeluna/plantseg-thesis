# B32 / F8 — pin Logit-KD and CWD-logit at OS8

**Base:** `fa3e8a0` · **Scope:** E2/E3 distillation losses only · **Uncommitted** (no commit until asked)  
**Untouched:** `L_CE` / `L_Dice` (full 512×512), all E1 code, teacher fine-tuning, `weights/`, every manuscript file, `docs/reference/**`.

**Decision implemented (settled in the B30 adjudication, not re-litigated):**

| Term | Grid | Validity mask | Changed? |
|---|---|---|---|
| Logit-KD KL | **OS8 64×64** | 64×64 | **YES** — was 512×512 |
| `L_CWD_logit` | **OS8 64×64** | 64×64 (shared) | no — already OS8 (B32-5 REFUTED) |
| `L_CWD_feat` | stride-16 32×32 | 32×32 | no |
| `L_CE`, `L_Dice` | full 512×512 | full | no |

---

## 0. Scope correction: the substantive change is on the STUDENT side

The item text frames B32-2 around teacher resampling. That framing is incomplete, and the report is written around the corrected version.

`logits_size` was **never passed** to `FrozenTeacher`, so the teacher's logits already arrived at their native 64×64 — `train_distill.py:182-184` was the *only* thing upsampling them. The substantive change is that `logit_kd_kl`'s **student** argument moves from `logits` (512², itself a bilinear upsample of `head_logits`) to `head_logits` (64², native).

So the old path was computing the KL between **two bilinear upsamples of two natively-64×64 fields** — 4,096 native positions inflated to 262,144, of which 98.4% were interpolants.

### Precondition, verified rather than assumed: gradients reach the head

Moving the KD term onto `head_logits` is only sound if that tensor is on the autograd graph. `src/distill/features.py:41-45`:

```python
        def _c5_hook(_module, _inp, out):
            self.c5 = out

        def _head_hook(_module, _inp, out):
            self.head_logits = out
```

Plain forward hooks, no `detach()`, no `torch.no_grad()`. The module docstring states it explicitly: *"The captured tensors stay attached to the autograd graph, so gradients flow back into the student exactly as they would through `logits`."* Confirmed empirically — `smoke_kd_resolution.py` asserts `head_logits.requires_grad and head_logits.grad_fn is not None`, and B32-4 measures a non-zero gradient norm through it on 20 real samples.

---

## B32-1 — `logit_kd_kl` signature

```diff
diff --git a/src/training/losses.py b/src/training/losses.py
index 9ef0440..8f77b66 100644
--- a/src/training/losses.py
+++ b/src/training/losses.py
@@ -91,15 +91,37 @@ class CombinedCEDiceLoss(nn.Module):
 
 
 def logit_kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
-                target: torch.Tensor, T: float = 4.0, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
-    """Hinton-style KL on temperature-softened outputs, averaged over valid (non-ignore) pixels (E2/E3).
+                valid_mask: torch.Tensor, T: float = 4.0) -> torch.Tensor:
+    """Hinton-style KL on temperature-softened outputs, averaged over valid locations (E2/E3).
+
+    Computed at the LOGIT MAP'S OWN grid — OS8 (64x64 for a 512x512 input), the resolution both the
+    student head and the SegNeXt LightHamHead emit natively (B32/F8). It is NOT computed on
+    upsampled copies: bilinear interpolation happens in LOGIT space, and
+    `softmax(interp(z)) != interp(softmax(z))`, so an upsampled KL evaluates ~98% of its positions
+    on interpolants that distort the soft targets rather than merely repeating them.
+
+    `valid_mask` is a bool [B,h,w] mask ALREADY at the logits' resolution — build it with
+    `downsample_validity(target, student_logits.shape[-2:])`. It is passed in rather than derived
+    from the full-resolution target because the previous signature silently accepted a 512x512
+    target regardless of the logits' grid, which is exactly the ambiguity that let the resolutions
+    drift apart. The shape is checked below so a mismatch fails loudly instead of broadcasting.
+
+    The mean reduction is UNCHANGED from the pre-B32 implementation: the term is a mean over valid
+    locations, so its magnitude is preserved when the valid population drops from ~262k to ~4k, and
+    the preregistered lambda_logit grid {0.25, 0.5, 1, 2, 4} stays on-scale.
 
     lambda_logit (the weight on this term relative to CE) is NEED_TO_CONFIRM (validation sweep).
     """
-    valid = (target != ignore_index).float()                  # [B,H,W]
+    expected = (student_logits.shape[0],) + tuple(student_logits.shape[-2:])
+    if tuple(valid_mask.shape) != expected:
+        raise ValueError(
+            f"valid_mask shape {tuple(valid_mask.shape)} does not match the logits grid {expected}; "
+            f"pass downsample_validity(target, student_logits.shape[-2:]) — the mask must be at the "
+            f"SAME resolution as the logits, not the full-resolution target")
+    valid = valid_mask.float()                                # [B,h,w]
     s_logp = F.log_softmax(student_logits / T, dim=1)
     t_prob = F.softmax(teacher_logits / T, dim=1)
-    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=1)  # [B,H,W]
+    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=1)  # [B,h,w]
     kl = kl * (T * T)                                          # T^2 absorbed into the term
     denom = valid.sum().clamp_min(1.0)
     return (kl * valid).sum() / denom
```

### The reduction is UNCHANGED — verbatim, both sides

| | pre-B32 | post-B32 |
|---|---|---|
| denominator | `denom = valid.sum().clamp_min(1.0)` | `denom = valid.sum().clamp_min(1.0)` |
| return | `(kl * valid).sum() / denom` | `(kl * valid).sum() / denom` |
| `T²` absorption | `kl = kl * (T * T)` | `kl = kl * (T * T)` |

Byte-identical. The term remains a **mean over valid locations**, so its magnitude is not mechanically rescaled when the valid population drops from ~262k to ~4k. That is the property F11 depends on. (The magnitude nonetheless changes for a *different* and more interesting reason — see B32-4.)

A shape check was added so a mask at the wrong resolution raises a named `ValueError` instead of failing on a broadcast, and `ignore_index` was dropped from the signature since the mask is now pre-computed and the parameter would be inert.

## B32-1b — the 5 external call sites, before and after

Required by the ruling: each must now pass a mask **at the resolution matching the logits it is paired with**, not one that merely happens to broadcast.

| # | File:line | Logits grid | BEFORE (mask arg) | AFTER (mask arg) | Matched? |
|---|---|---|---|---|---|
| 1 | `smoke_distill.py:63` | 8×8 | `target` (8×8) | `valid = downsample_validity(target, student.shape[-2:])` → 8×8 | ✅ |
| 2 | `smoke_distill.py:76` | 8×8 | `target` (8×8) | `valid` → 8×8 | ✅ |
| 3 | `smoke_distill.py:77` | 8×8 | `target` (8×8) | `valid` → 8×8 | ✅ |
| 4 | `smoke_distill.py:84` | 8×8 | `target` (8×8) | `valid` → 8×8 | ✅ |
| 5 | `smoke_losses_metrics.py:63` | 512×512 | `mask` (512×512) | `kd_valid = downsample_validity(mask, logits.shape[-2:])` → 512×512 | ✅ |
| — | `train_distill.py:185` **(production)** | **512×512 ← the defect** | `mask` (512×512) | `valid_os8` → **64×64**, paired with `head_logits` | ✅ |

### Was either smoke passing a full-res mask against OS8 logits?

**No.** Both were internally consistent: `smoke_distill.py` used `b,h,w = 2,8,8` for student, teacher and target alike, and `smoke_losses_metrics.py` used 512×512 logits with a 512×512 mask. **There is no second latent defect in the smokes.**

That is worth stating precisely, because it identifies what the production defect actually *was*. `train_distill.py` also passed a mask that matched its logits — both at 512×512. The bug was never a mask/logits **mismatch**; it was that the matched pair sat on the **wrong grid**, one manufactured by upsampling. A mismatch check alone would never have caught it, which is why `smoke_kd_resolution.py` asserts the absolute grid (64×64), not merely agreement between two tensors.

Also required: `downsample_validity` had to be re-exported from `src/training/__init__.py`, which `smoke_losses_metrics.py` imports from.

```diff
diff --git a/scripts/smoke_distill.py b/scripts/smoke_distill.py
index 5dd1172..9b4b3e8 100644
--- a/scripts/smoke_distill.py
+++ b/scripts/smoke_distill.py
@@ -60,7 +60,9 @@ def test_logit_kd() -> None:
     target = torch.randint(0, NC, (b, h, w))
     target[0, 0, :] = IGNORE                               # an ignored row
 
-    loss = logit_kd_kl(student, teacher, target, T=4.0, ignore_index=IGNORE)
+    # B32: the mask is downsampled to the LOGITS' grid and passed in explicitly.
+    valid = downsample_validity(target, student.shape[-2:], ignore_index=IGNORE)
+    loss = logit_kd_kl(student, teacher, valid, T=4.0)
     check("kd_scalar_finite", loss.dim() == 0 and bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")
 
     loss.backward()
@@ -73,15 +75,15 @@ def test_logit_kd() -> None:
     t2 = teacher.clone()
     s2[0, :, 0, :] += 12.0
     t2[0, :, 0, :] -= 7.0
-    base = logit_kd_kl(student.detach(), teacher, target, T=4.0, ignore_index=IGNORE)
-    pert = logit_kd_kl(s2, t2, target, T=4.0, ignore_index=IGNORE)
+    base = logit_kd_kl(student.detach(), teacher, valid, T=4.0)
+    pert = logit_kd_kl(s2, t2, valid, T=4.0)
     check("kd_ignore_index_invariant", torch.allclose(base, pert, atol=1e-6),
           f"base={base.item():.6f} perturbed={pert.item():.6f}")
 
     # spatial alignment is explicit: a coarser teacher is resampled before the loss
     t_small = torch.randn(b, NC, h // 2, w // 2)
     t_up = F.interpolate(t_small, size=(h, w), mode="bilinear", align_corners=False)
-    aligned = logit_kd_kl(student.detach(), t_up, target, T=4.0, ignore_index=IGNORE)
+    aligned = logit_kd_kl(student.detach(), t_up, valid, T=4.0)
     check("kd_spatial_alignment", aligned.dim() == 0 and bool(torch.isfinite(aligned)),
           f"teacher {tuple(t_small.shape[-2:])} -> {tuple(t_up.shape[-2:])}")
 
diff --git a/scripts/smoke_losses_metrics.py b/scripts/smoke_losses_metrics.py
index 74902cf..8950d22 100644
--- a/scripts/smoke_losses_metrics.py
+++ b/scripts/smoke_losses_metrics.py
@@ -21,7 +21,8 @@ from src.seeds import set_seed  # noqa: E402
 from src.data import NUM_CLASSES, build_dataloader  # noqa: E402
 from src.models import build_student  # noqa: E402
 from src.training import (  # noqa: E402
-    CombinedCEDiceLoss, SoftDiceLoss, WeightedCrossEntropyLoss, compute_class_weights, logit_kd_kl,
+    CombinedCEDiceLoss, SoftDiceLoss, WeightedCrossEntropyLoss, compute_class_weights,
+    downsample_validity, logit_kd_kl,
 )
 from src.eval import all_class_miou, confusion_matrix, disease_only_miou, per_image_miou  # noqa: E402
 
@@ -60,7 +61,9 @@ def main() -> int:
         dice = SoftDiceLoss()(logits, mask)
         combined = CombinedCEDiceLoss(weight=weights)(logits, mask)
         teacher = torch.randn_like(logits)  # dummy teacher (smoke only)
-        kd = logit_kd_kl(logits, teacher, mask, T=4.0)
+        # B32: mask downsampled to the logits' grid (here they already match at 512x512).
+        kd_valid = downsample_validity(mask, logits.shape[-2:])
+        kd = logit_kd_kl(logits, teacher, kd_valid, T=4.0)
     print(f"[loss] CE={ce.item():.4f}  Dice={dice.item():.4f}  CE+Dice={combined.item():.4f}  "
           f"logitKD-KL(T=4, dummy teacher)={kd.item():.4f}")
     for name, v in [("CE", ce), ("Dice", dice), ("CE+Dice", combined), ("logitKD-KL", kd)]:
diff --git a/src/training/__init__.py b/src/training/__init__.py
index 21c259f..889e10b 100644
--- a/src/training/__init__.py
+++ b/src/training/__init__.py
@@ -6,6 +6,7 @@ from .losses import (
     WeightedCrossEntropyLoss,
     compute_class_weights,
     cwd_channelwise_kl,
+    downsample_validity,
     logit_kd_kl,
 )
 
@@ -16,4 +17,5 @@ __all__ = [
     "compute_class_weights",
     "logit_kd_kl",
     "cwd_channelwise_kl",
+    "downsample_validity",
 ]
```

## B32-2 — call-site rewiring

```diff
diff --git a/src/training/train_distill.py b/src/training/train_distill.py
index 82cf6cc..d7d15b9 100644
--- a/src/training/train_distill.py
+++ b/src/training/train_distill.py
@@ -179,10 +179,15 @@ def distillation_losses(*, stage: dict, logits, head_logits, c5, mask, teacher_o
                          f"{logits.shape[1]}")
 
     # --- Logit KD (E2 and E3, unchanged between them per contract B3) ---
-    t_logits_full = (t_logits if t_logits.shape[-2:] == logits.shape[-2:]
-                     else F.interpolate(t_logits, size=logits.shape[-2:], mode="bilinear",
-                                        align_corners=False))
-    l_kd = logit_kd_kl(logits, t_logits_full, mask, T=T_LOGIT, ignore_index=IGNORE_INDEX)
+    # B32/F8: computed on the head's NATIVE OS8 map, not on upsampled copies. The student's
+    # `head_logits` and the teacher's LightHamHead output are both 64x64 for a 512x512 input, so
+    # neither side is resampled and no interpolation artifact enters the soft targets. The validity
+    # mask is downsampled to that same grid and SHARED with the CWD logit term below.
+    valid_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE_INDEX)
+    t_logits_os8 = (t_logits if t_logits.shape[-2:] == head_logits.shape[-2:]
+                    else F.interpolate(t_logits, size=head_logits.shape[-2:], mode="bilinear",
+                                       align_corners=False))
+    l_kd = logit_kd_kl(head_logits, t_logits_os8, valid_os8, T=T_LOGIT)
     total = total + ramp * lambda_logit * l_kd
     parts["logit_kd"] = float(l_kd.detach())
 
@@ -201,10 +206,9 @@ def distillation_losses(*, stage: dict, logits, head_logits, c5, mask, teacher_o
     parts["cwd_feat"] = float(l_feat.detach())
 
     # --- CWD logit term: on the head's native OS8 logit map (no projection needed, same C) ---
-    t_logits_os8 = (t_logits if t_logits.shape[-2:] == head_logits.shape[-2:]
-                    else F.interpolate(t_logits, size=head_logits.shape[-2:], mode="bilinear",
-                                       align_corners=False))
-    valid_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE_INDEX)
+    # This term was ALREADY at OS8 before B32 (B32-5: REFUTED, unchanged). It now reuses the
+    # `t_logits_os8` and `valid_os8` computed once for the Logit-KD term above — same grid, same
+    # mask, one min-pool instead of two.
     # channels_norm defaults to this map's own channel count (116 classes) — 320 is the FEATURE-map
     # normalisation only and must never be hard-coded here.
     l_logit_map = cwd_channelwise_kl(head_logits, t_logits_os8, valid_os8, T=T_CWD)
@@ -320,7 +324,12 @@ def run(*, stage: dict, mode: str, device: str, pretrained, teacher: FrozenTeach
                                                          for p in g["params"]}
                                                         & teacher.parameter_ids())
 
-        teacher_out = teacher(model_input, feat_size=c5.shape[-2:])
+        # B32/F8: request the teacher's logits on the student head's NATIVE OS8 grid. The SegNeXt
+        # LightHamHead already emits 64x64 for a 512x512 input (stock in_index=[1,2,3], resized to
+        # inputs[0] = stride-8), so this is currently a no-op — but it makes the resolution contract
+        # explicit and load-bearing if the teacher config ever changes.
+        teacher_out = teacher(model_input, logits_size=head_logits.shape[-2:],
+                              feat_size=c5.shape[-2:])
         if it == 1:
             checks["teacher_same_augmented_input"] = model_input is img
 
```

### Consumers of `teacher_out.logits`, confirmed before deleting the upsample

| Line | Consumer | Depends on 512×512? |
|---|---|---|
| `:176` | `t_logits = teacher_out.logits` | no — binding only |
| `:177` | class-count check `t_logits.shape[1]` | no — channel dim, resolution-independent |
| `:182-184` | `t_logits_full` (Logit-KD) | **the only one** — deleted |
| `:204-206` | `t_logits_os8` (CWD-logit) | no — already targets `head_logits.shape[-2:]` |

No other consumer exists, so deleting the upsample is safe.

`logits_size=head_logits.shape[-2:]` is now passed to `FrozenTeacher`. It is **currently a no-op** — the teacher already emits 64×64 — but it makes the contract explicit and load-bearing if the teacher config ever changes. **CANNOT-VERIFY-LOCALLY:** MMSeg is not installed, so the real teacher was never built. Settle on the pod with:
```bash
python -c "import torch,sys; sys.path.insert(0,'.'); from src.distill.segnext_teacher import build_segnext_teacher; m=build_segnext_teacher('weights/<ckpt>.pth'); o=m(torch.zeros(1,3,512,512)); print('teacher logits', tuple(o['logits'].shape), 'feat_s16', tuple(o['feat_s16'].shape))"
```

**Expected:** `teacher logits (1, 116, 64, 64)  feat_s16 (1, 320, 32, 32)`. If the logits come back at any other size, `logits_size` stops being a no-op and starts doing real work — which is precisely why it was added.

## B32-5 — REFUTED, nothing changed

`L_CWD_logit` was **already** computed at OS8 before B32. Pre-existing code, `train_distill.py:204-210`:
```python
    # --- CWD logit term: on the head's native OS8 logit map (no projection needed, same C) ---
    t_logits_os8 = (t_logits if t_logits.shape[-2:] == head_logits.shape[-2:]
                    else F.interpolate(t_logits, size=head_logits.shape[-2:], mode="bilinear",
                                       align_corners=False))
    valid_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE_INDEX)
    l_logit_map = cwd_channelwise_kl(head_logits, t_logits_os8, valid_os8, T=T_CWD)
```

Both the teacher resample target and `downsample_validity` already used `head_logits.shape[-2:]`. The only edit was removing the now-duplicated `valid_os8` / `t_logits_os8` computation, since the Logit-KD block above computes them once and shares them (approved).

**The ternary at `:204` is retained as a defensive no-op.** After B32-2 it is **dead-by-construction** — `t_logits` always arrives at 64×64, so the `F.interpolate` branch can never execute. Recording that explicitly so nobody later reads its presence as evidence that the teacher arrives at 512×512. It costs nothing and guards a future teacher-config change.

## B32-3 — validity-mask statistics [MEASURED, 60 real train samples]

```text
==============================================================================
B32-3 — validity-mask statistics, 60 real train samples [MEASURED]
==============================================================================
downsample_validity uses CONSERVATIVE ALL-VALID min-pooling: an OS8 cell is valid iff
ALL 64 contributing full-resolution pixels are valid (non-255).

grid                         mean   median      min      max   valid FRACTION
512x512  (CE / Dice)       0.7189   0.7227   0.4141   1.0000
64x64    (Logit-KD, CWD-logit)   0.7096   0.7188   0.4062   1.0000
32x32    (CWD-feat)        0.6964   0.6875   0.3750   1.0000

grid                         mean   median      min      max   valid CELL COUNT
64x64  (of 4096)           2906.7   2944.0     1664     4096
32x32  (of 1024)            713.1    704.0      384     1024

--- A2: zero-valid and near-degenerate tail ---
  samples with ZERO valid 64x64 cells : 0 of 60  (0.0%)
  samples with ZERO valid 32x32 cells : 0 of 60  (0.0%)
  minimum NON-ZERO valid 64x64 count  : 1664 of 4096  (40.62%)
  minimum NON-ZERO valid 32x32 count  : 384 of 1024  (37.50%)
  5 smallest 64x64 counts             : [1664, 1792, 1792, 2048, 2048]
  5 smallest 32x32 counts             : [384, 448, 448, 512, 512]

--- Logit-KD valid population vs CE's (the ch3 number) ---
  CE       valid fraction (512x512) : 0.7189
  Logit-KD valid fraction (64x64)   : 0.7096
  CWD-feat valid fraction (32x32)   : 0.6964
  Logit-KD retains 0.9871 of CE's valid fraction (loses 1.29% relative)
  CWD-feat retains 0.9686 of CE's valid fraction (loses 3.14% relative)

--- A3: letterboxed vs near-square source images ---
  aspect ratio range observed: 1.00 .. 2.42

  group                          n     v512      f64      f32  f64/v512
  near-square (AR < 1.10)        7   0.9883   0.9821   0.9732    0.9938
  moderate (1.10 <= AR < 1.40)  24   0.7549   0.7480   0.7383    0.9909
  letterboxed (AR >= 1.40)      29   0.6242   0.6121   0.5948    0.9806

  near-square minus letterboxed, 64x64 valid fraction: +0.3701 (+37.01 percentage points)

--- most letterboxed samples ---
  idx 4093  src (716, 296)   AR 2.42  v512=0.4141 f64=0.4062 n64=1664
  idx 3638  src (600, 276)   AR 2.17  v512=0.4609 f64=0.4375 n64=1792
  idx 182   src (640, 1387)  AR 2.17  v512=0.4609 f64=0.4375 n64=1792
  idx 4547  src (250, 478)   AR 1.91  v512=0.5234 f64=0.5000 n64=2048
  idx 4638  src (224, 425)   AR 1.90  v512=0.5273 f64=0.5000 n64=2048
```

### The number for the ch3 amendment

| Grid | Term | Mean valid fraction | Retained vs CE |
|---|---|---|---|
| 512×512 | `L_CE`, `L_Dice` | **0.7189** | — |
| 64×64 | Logit-KD, `L_CWD_logit` | **0.7096** | **98.71%** (loses 1.29% relative) |
| 32×32 | `L_CWD_feat` | **0.6964** | 96.86% (loses 3.14% relative) |

**The Logit-KD valid population is a strict subset of CE's, and it is 1.29% smaller in relative terms.** That is the sentence for ch3. It is far smaller than I flagged as a risk when signing off F8 — conservative all-valid min-pooling costs almost nothing here, because `core_preprocess` pads only one axis and the padding boundary is a thin band relative to the content.

### A2 — zero-valid and the near-degenerate tail

| Quantity | 64×64 | 32×32 |
|---|---|---|
| samples with ZERO valid cells | **0 of 60 (0.0%)** | **0 of 60 (0.0%)** |
| minimum non-zero valid count | **1,664 of 4,096 (40.62%)** | **384 of 1,024 (37.50%)** |
| five smallest counts | 1664, 1792, 1792, 2048, 2048 | 384, 448, 448, 512, 512 |

**No near-degenerate tail exists.** The worst sample retains 40.6% of its OS8 cells — three orders of magnitude away from the "3 valid cells out of 4,096" scenario. The distribution has no thin left tail at all; the minimum is close to the bulk.

### Zero-valid guard: UNREACHABLE-BY-CONSTRUCTION, no guard added

Per the ruling, no guard was added for an undemonstrated case. The reasoning, beyond the 0/60 measurement:

`core_preprocess` resizes the **long** side to 512 and pads only the short side, so the content region is always `512 × ceil(512/AR)` pixels. Zero valid OS8 cells would require no 8×8-aligned block to be fully inside the content region — i.e. a content strip under ~8 px, meaning **AR > 64**. The widest aspect ratio in the 60-sample probe is **2.42**, and PlantSeg is photographic. The case is not merely unobserved; it is geometrically excluded.

`denom = valid.sum().clamp_min(1.0)` remains as the arithmetic backstop: were it ever reached, the term contributes `0.0` rather than `NaN`, which is the correct degenerate behaviour anyway.

### A3 — letterboxed vs near-square

| Group | n | valid @512² | valid @64² | **retention f64/v512** |
|---|---|---|---|---|
| near-square (AR < 1.10) | 7 | 0.9883 | 0.9821 | **0.9938** |
| moderate (1.10 ≤ AR < 1.40) | 24 | 0.7549 | 0.7480 | **0.9909** |
| letterboxed (AR ≥ 1.40) | 29 | 0.6242 | 0.6121 | **0.9806** |

**The effect is real but small, and the headline number is misleading unless decomposed.**

Absolute OS8 valid fraction differs enormously between groups — **+37.01 percentage points** for near-square over letterboxed. But that gap is caused by **padding**, which `L_CE` sees equally; it is not attributable to the OS8 downsampling.

The quantity that isolates min-pooling's cost is the **retention ratio** `f64/v512`, and it is remarkably stable: **0.9938 → 0.9909 → 0.9806**. Conservative min-pooling costs a near-square image 0.62% of its valid population and a letterboxed one 1.94% — a differential of only **1.3 percentage points**.

**Ch4 sentence, if you want one:** conservative all-valid min-pooling penalises letterboxed samples about three times as heavily as near-square ones in relative terms (1.94% vs 0.62% of valid locations lost), but the absolute effect is under 2% everywhere and is dwarfed by the padding itself, which the supervised terms see identically. **This is not a Ch4 alarm.**

## B32-4 — numerical equivalence and impact [MEASURED, 20 real samples]

Synthetic teacher: random logits at 64×64. MMSeg is absent locally; a random teacher characterises the **operator**, which is what this item asks for. See the caveat below on how a real teacher would differ.

```text
C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-plantseg-thesis\524239b6-5425-4558-b126-688ba3ba9e9c\scratchpad\b32_4_impact.py:85: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\autograd\generated\python_variable_methods.cpp:837.)
  kl_old.append(float(lo))
==============================================================================
B32-4 — OLD (both sides upsampled to 512) vs NEW (both native 64x64), 20 real samples
==============================================================================
  [ 1/20] idx 0     KL old=0.565660 new=1.109456 ratio=1.9613 | grad-norm old=1.451e-03 new=2.564e-03
  [ 2/20] idx 282   KL old=0.588738 new=1.171258 ratio=1.9894 | grad-norm old=1.525e-03 new=2.734e-03
  [ 3/20] idx 565   KL old=0.756736 new=1.403643 ratio=1.8549 | grad-norm old=2.128e-03 new=3.567e-03
  [ 4/20] idx 847   KL old=0.599112 new=1.144526 ratio=1.9104 | grad-norm old=1.528e-03 new=2.616e-03
  [ 5/20] idx 1130  KL old=0.547311 new=1.081148 ratio=1.9754 | grad-norm old=1.448e-03 new=2.569e-03
  [ 6/20] idx 1412  KL old=0.541005 new=1.060606 ratio=1.9604 | grad-norm old=1.410e-03 new=2.490e-03
  [ 7/20] idx 1695  KL old=0.593824 new=1.156474 ratio=1.9475 | grad-norm old=1.579e-03 new=2.786e-03
  [ 8/20] idx 1977  KL old=0.536141 new=1.095074 ratio=2.0425 | grad-norm old=1.375e-03 new=2.528e-03
  [ 9/20] idx 2259  KL old=0.555441 new=1.075453 ratio=1.9362 | grad-norm old=1.356e-03 new=2.373e-03
  [10/20] idx 2542  KL old=0.638855 new=1.205437 ratio=1.8869 | grad-norm old=1.693e-03 new=2.933e-03
  [11/20] idx 2824  KL old=0.554693 new=1.102610 ratio=1.9878 | grad-norm old=1.426e-03 new=2.542e-03
  [12/20] idx 3107  KL old=0.506630 new=1.016820 ratio=2.0070 | grad-norm old=1.205e-03 new=2.182e-03
  [13/20] idx 3389  KL old=0.585146 new=1.123075 ratio=1.9193 | grad-norm old=1.566e-03 new=2.751e-03
  [14/20] idx 3671  KL old=0.525667 new=0.991276 ratio=1.8857 | grad-norm old=1.344e-03 new=2.304e-03
  [15/20] idx 3954  KL old=0.522911 new=1.027738 ratio=1.9654 | grad-norm old=1.392e-03 new=2.462e-03
  [16/20] idx 4236  KL old=0.602226 new=1.196018 ratio=1.9860 | grad-norm old=1.626e-03 new=2.940e-03
  [17/20] idx 4519  KL old=0.572728 new=1.128229 ratio=1.9699 | grad-norm old=1.465e-03 new=2.586e-03
  [18/20] idx 4801  KL old=0.508349 new=1.006982 ratio=1.9809 | grad-norm old=1.208e-03 new=2.180e-03
  [19/20] idx 5084  KL old=0.682292 new=1.296363 ratio=1.9000 | grad-norm old=1.886e-03 new=3.251e-03
  [20/20] idx 5366  KL old=0.556384 new=1.104584 ratio=1.9853 | grad-norm old=1.433e-03 new=2.556e-03

------------------------------------------------------------------------------
KL MAGNITUDE [MEASURED]
------------------------------------------------------------------------------
  OLD (upsampled 512x512)      mean=0.576992  median=0.561022  min=0.506630  max=0.756736
  NEW (native 64x64)           mean=1.124838  median=1.107020  min=0.991276  max=1.403643

  per-sample ratio NEW/OLD      mean=1.9526  median=1.9634  min=1.8549  max=2.0425
  ratio of means                1.9495

------------------------------------------------------------------------------
GRADIENT NORM w.r.t. student head_logits [MEASURED]
------------------------------------------------------------------------------
  OLD                          mean=1.5023e-03  median=1.4494e-03  min=1.2054e-03  max=2.1278e-03
  NEW                          mean=2.6457e-03  median=2.5663e-03  min=2.1801e-03  max=3.5674e-03

  per-sample grad-norm ratio NEW/OLD  mean=1.7645  median=1.7669  min=1.6766  max=1.8383

------------------------------------------------------------------------------
WALL CLOCK per KD-term evaluation, batch=1 [MEASURED, CPU]
------------------------------------------------------------------------------
  OLD  mean=209.69 ms   median=205.78 ms
  NEW  mean=97.47 ms   median=110.22 ms
  speedup  mean=2.2x   median=1.9x

------------------------------------------------------------------------------
PEAK TENSOR FOOTPRINT of the KD term [INFERRED — exact arithmetic, not a measurement]
------------------------------------------------------------------------------
  at the real batch size 16, float32:

  tensor                          OLD (MB)    NEW (MB)
  teacher upsample t_up             1856.0         0.0
  log_softmax(student/T)            1856.0        29.0
  softmax(teacher/T)                1856.0        29.0
  kl_div elementwise                1856.0        29.0
  kl summed over C                    16.0         0.2
  valid mask                          16.0         0.2
  TOTAL                             7456.0        87.5

  reduction: 7368.5 MB  (85x smaller)
  Forward activations only; the backward pass holds comparable tensors, so the realised
  saving on GPU is roughly double this. MEASURE IT ON THE POD with
  torch.cuda.max_memory_allocated() around the KD term.
```

### Summary

| Quantity | OLD (upsampled 512²) | NEW (native 64²) | Ratio NEW/OLD | Label |
|---|---|---|---|---|
| KL magnitude, mean | 0.576992 | 1.124838 | **1.9495** | MEASURED |
| KL magnitude, per-sample median ratio | — | — | **1.9634** (range 1.855–2.043) | MEASURED |
| grad norm w.r.t. `head_logits`, mean | 1.5023e-03 | 2.6457e-03 | **1.7645** | MEASURED |
| wall clock per KD term, batch 1, CPU | 209.69 ms | 97.47 ms | **2.2× faster** | MEASURED |
| peak forward tensor footprint, batch 16 | **7,456 MB** | **87.5 MB** | **85× smaller** | INFERRED (exact arithmetic) |

### Why the magnitude nearly doubles — this is the artifact, quantified

The reduction did not change, so this is not a rescaling. Under the OLD path both sides were bilinear upsamples of the *same* 64×64 grid, so at the 63/64 interpolated positions the student and teacher were each blends of the same neighbourhoods — they **agreed more than their native values do**. The upsampled KL therefore *understated* the true divergence by roughly half. The NEW value is the honest one.

`smoke_kd_resolution.py` reproduces this independently with a different random teacher (ratio 2.165), so it is a property of the operator, not of one sample of noise.

### A1 — DOES λ STAY ON-SCALE?

**YES — but the expected optimum shifts down by one grid step, and that must be stated in Ch4.**

- The KD term's magnitude rises by **1.95×** and its gradient contribution by **1.76×**.
- The preregistered grid `{0.25, 0.5, 1, 2, 4}` is **geometric with ratio exactly 2**, spanning 16×. A 1.95× magnitude change is therefore **almost exactly one grid step**.
- Consequence: whatever λ would have been optimal under the old semantics, the optimum under OS8 semantics sits approximately **one step lower** (an old optimum of 1 → about 0.5).
- **The grid still brackets a sensible optimum.** It only fails to bracket if the old-path optimum sat at the bottom boundary 0.25, which would push the new optimum to ~0.125. ch3 already handles that: *"if the selected value lies at a grid boundary it is reported as such rather than the grid being extended."*
- **No re-centring is needed and none is proposed.** A one-step shift inside a 16×-wide geometric grid is well within its span. F11's preregistration is intact.

**The one thing Ch4 must say:** the λ sweep was run under **OS8** Logit-KD semantics. A λ value selected under the old upsampled semantics is not transferable, because the term it weights is ~2× larger.

### Caveat on the 1.95× figure — it is likely an UPPER BOUND [INFERRED]

The synthetic teacher is **random** logits, the worst case for spatial smoothness: adjacent positions are uncorrelated, so bilinear interpolation destroys the most information and the OLD/NEW gap is maximal. A **real trained SegNeXt teacher** produces spatially smooth logit fields, where interpolation is closer to faithful and the ratio should be nearer 1. **CANNOT-VERIFY-LOCALLY.** Re-measure on the pod once the teacher checkpoint exists, before committing to a λ interpretation. The direction of the effect is certain; only its size is teacher-dependent.

### The memory result may be load-bearing for E2/E3 feasibility

At batch 16, the OLD path allocated **four** `[16, 116, 512, 512]` float32 tensors — 1,856 MB each, **7.4 GB** of forward activations for the KD term alone, before the backward pass, the student, the frozen teacher, or the CWD terms. On a 24 GB RTX 4090 that is very likely the difference between E2/E3 fitting at the locked batch size 16 and not. **INFERRED** from exact tensor arithmetic; measure on the pod with `torch.cuda.max_memory_allocated()` around the KD term.

## B32-6 — `scripts/smoke_kd_resolution.py` (new standing gate)

```text
C:\Users\admin\plantseg-thesis\scripts\smoke_kd_resolution.py:102: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\autograd\generated\python_variable_methods.cpp:837.)
  l_kd.dim() == 0 and bool(torch.isfinite(l_kd)), f"loss={float(l_kd):.6f}")
==============================================================================
B32 — KD spatial-resolution gate
torch 2.9.1+cpu | input 512x512 -> OS8 64x64, s16 32x32
==============================================================================

  PASS  student full-res logits are 512x512   [(2, 116, 512, 512)]
  PASS  student head_logits are OS8 64x64   [(2, 116, 64, 64)]
  PASS  student c5 is stride-16 32x32   [(2, 160, 32, 32)]
  PASS  head_logits stays on the autograd graph   [forward hooks must not detach, or KD gradients never reach the head]
  PASS  validity mask for the logit terms is 64x64   [(2, 64, 64)]
  PASS  validity mask for the feature term is 32x32   [(2, 32, 32)]
  PASS  mask is bool   [torch.bool]
  PASS  min-pool is conservative (all-valid), not any-valid   [first 8 OS8 rows fully invalid, remainder valid]
  PASS  Logit-KD evaluates at OS8 and returns a finite scalar   [loss=1.017978]
  PASS  Logit-KD gradient reaches head_logits
  PASS  logit_kd_kl rejects a full-res mask against OS8 logits   [ValueError]
  PASS  reduction is a MEAN over valid cells, not a sum   [full=1.0180 half-population=1.0190 — a SUM would roughly halve]
  PASS  CWD-logit evaluates at OS8 on head_logits   [loss=1.015847]
  PASS  CWD-feat operates at stride-16 with C=320   [(2, 320, 32, 32)]
  PASS  CWD-feat returns a finite scalar   [loss=1.019863]
  PASS  CWD-feat mask is NOT the OS8 mask   [32x32 vs 64x64 — distinct grids by design]
  PASS  CE+Dice still evaluate at FULL 512x512 on the upsampled logits   [loss=5.990883]
  PASS  T_logit == 4
  PASS  T_cwd == 4
  PASS  alpha_CWD == 50
  PASS  beta_CWD == 3
  PASS  CWD C == 320
  PASS  lambda grid == (0.25, 0.5, 1, 2, 4)
  PASS  the OLD upsampled path gives a MATERIALLY different value   [os8=1.0180 upsampled=0.4702 ratio=2.165 — they are not interchangeable]

RESULT: PASS (24/24)
```

**24/24 in 7 s wall, CPU, synthetic teacher, no checkpoint.**

The design point: it asserts the **absolute grid** (64×64, 32×32, 512×512), not merely that a mask agrees with its logits. The original defect had a mask that *did* agree with its logits — both were at 512×512 — so an agreement-only check would have passed it. The final assertion pins the two paths apart deliberately (`ratio=2.165 — they are not interchangeable`), so a silent revert to the upsampled path fails the gate.

## Refused / flagged

**Nothing refused.** Four items flagged:

1. **`src/training/__init__.py` was edited** (+2 lines) to re-export `downsample_validity`. Not named in the item list; `smoke_losses_metrics.py` imports from the package, not the module, so the B32-1b update could not work without it.
2. **B32-5 REFUTED** — `L_CWD_logit` was already at OS8. No change beyond removing the now-duplicated mask computation.
3. **The `:204` ternary is dead-by-construction** after B32-2 and retained deliberately.
4. **No zero-valid guard added** — unreachable by geometry, 0/60 measured, per the ruling.

**Locked values verified unchanged:** `T_Logit=4`, `T_CWD=4`, `α_CWD=50`, `β_CWD=3`, `T²/C` with `C=320`, λ grid `{0.25, 0.5, 1, 2, 4}`, projection 160→320. All asserted by `smoke_kd_resolution.py`. `L_CE`/`L_Dice` path untouched.

## Residual risks

| # | Risk | Status |
|---|---|---|
| 1 | The real teacher's output grid was never observed — MMSeg absent locally | **CANNOT-VERIFY-LOCALLY**; pod command in B32-2. `logits_size` makes a surprise here safe rather than silent |
| 2 | The 1.95× ratio is measured against a *random* teacher and is probably an upper bound | Re-measure with the real teacher before interpreting λ |
| 3 | λ's optimum shifts ~1 grid step down; a λ chosen under old semantics is not transferable | Ch4 must state the sweep ran under OS8 semantics |
| 4 | The 85× memory reduction is INFERRED arithmetic, not a GPU measurement | `torch.cuda.max_memory_allocated()` on the pod |
| 5 | ch3 pins *which pixels* ("valid pixels only") but not *which grid* | Manuscript amendment — Logit-KD must be split out of the CE/Dice ignore-handling sentence, since those stay at 512² while Logit-KD moves to OS8, and the valid population differs by 1.29% |
| 6 | E2/E3 remain blocked on the absent teacher checkpoint | Unchanged by B32 |

## Provenance

### Isolation — E1 and every neighbouring smoke, after all items
```text
########## E1 ISOLATION — FINAL, after all B32 items ##########
--- train_e1.py --dry-run ---

RESULT: PASS (6/6 checks exercised, 0 skipped)
--- smoke_aug_stochasticity.py ---

RESULT: PASS
--- smoke_distill.py ---

RESULT: PASS (89/89)
--- smoke_losses_metrics.py ---

[PASS] losses and metrics smoke test
--- smoke_kd_resolution.py ---

RESULT: PASS (24/24)
--- smoke_cwd_projection.py (CWD unaffected?) ---

[PASS] CWD projection stub smoke test
```

E1 was re-verified after **every** item, not only at the end: `train_e1.py --dry-run` and `smoke_aug_stochasticity.py` both green after B32-1/1b/2 and again after B32-6. Neither E1 path imports `train_distill.py`, and `logit_kd_kl` is not on the E1 graph.

### Files changed
```text
 docs/reference/reference.pdf    | Bin 129010 -> 163179 bytes
 scripts/smoke_distill.py        |  10 ++++++----
 scripts/smoke_losses_metrics.py |   7 +++++--
 src/training/__init__.py        |   2 ++
 src/training/losses.py          |  30 ++++++++++++++++++++++++++----
 src/training/train_distill.py   |  27 ++++++++++++++++++---------
 6 files changed, 57 insertions(+), 19 deletions(-)
```

### Git status
```text
 M docs/reference/reference.pdf
 M scripts/smoke_distill.py
 M scripts/smoke_losses_metrics.py
 M src/training/__init__.py
 M src/training/losses.py
 M src/training/train_distill.py
?? scripts/smoke_kd_resolution.py
```

### Confirmations

- **Nothing committed, nothing staged** — no commit until asked.
- **`docs/reference/reference.pdf`** untouched; ` M` and mtime `Jun 27 13:28` predate B30.
- **No teacher fine-tuning, no `weights/` access, no MMSeg build, no checkpoint loaded.**
- **No manuscript or `docs/reference/**` edit.**
- **`L_CE` / `L_Dice` untouched**; `CombinedCEDiceLoss` not modified.
- **No training, GPU, downloads, or installs.** CPU only.

