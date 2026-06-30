# ImageNet Student-Init Wiring (B18d)

**RESULT: ✅ PASS** — `build_student()` now honors the configured ImageNet init, with a safe no-download
default. **No download / no network was attempted** in this step. Not training.

_Generated: 2026-06-28 (read-only verification; Anaconda Python torch 2.9.1+cpu, CPU). HEAD `666ef38`._

---

## What was implemented (`src/models/student.py` only)
`_build_backbone(dilated, pretrained)` now resolves the `pretrained` argument instead of forcing
`weights=None`:

| `pretrained` value | Behavior |
|---|---|
| `False` / `None` (**default**) | random init, `weights=None`, **NO download** (smoke/offline-safe) |
| `True` | `torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2` |
| `"IMAGENET1K_V2"` | same (short alias) |
| `"torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"` | same (**exact `configs/e1_student.py` value**) |
| anything else | **`ValueError`** (clear message) |

- ImageNet path uses the **float** enum `torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2`
  on the quantizable `mobilenet_v3_large(quantize=False)`; loads from the torch-hub cache if present,
  else downloads (**real E1 run env only**).
- If ImageNet is requested but unobtainable (offline + not cached), it raises a **clear RuntimeError with
  guidance** — **no silent random fallback**.
- `build_student(num_classes=None, pretrained=False)` signature/default **unchanged** → smokes never
  download. **No architecture change; output stays `[B,116,H,W]`.**
- `scripts/smoke_train_step.py` **not modified** (its `build_student()` default is already random/no-download).

## Verification (commands + results)
**Command 1 — wiring verification (scratch, no network):**
`python <scratch>/imagenet_init_verify.py` (Anaconda Python, `PYTHONIOENCODING=utf-8`)
| Check | Result |
|---|---|
| 1. Default build (`pretrained=False`) | `used_pretrained=False`, 2,933,688 params, forward **`(2,116,512,512)`** finite ✅ (no download) |
| 2. Enum resolves (no download) | `MobileNet_V3_Large_Weights.IMAGENET1K_V2`, url `…/mobilenet_v3_large-5c1a4163.pth` ✅ |
| 3. Exact config string routes | `E1_STUDENT["init_weights"]` = `"torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"` ∈ aliases ✅; `"IMAGENET1K_V2"`/`True` ∈ aliases ✅; unknown → `ValueError` ✅ |
| 4. Cache check (no download) | hub dir `C:\Users\admin/.cache\torch\hub`; file `mobilenet_v3_large-5c1a4163.pth` → **cached = False** |
| 5. Pretrained-from-cache | **SKIPPED** (not cached) — would download in the real-run env; **no download attempted** |
| **download_attempted** | **False** |

**Command 2 — train-step regression (default random path):**
`python scripts/smoke_train_step.py` → **RESULT: PASS** — image `(2,3,512,512)` f32 / mask `(2,512,512)` i64 /
logits `(2,116,512,512)` finite; CE 5.1293 · Dice 0.9820 · CE+Dice 6.1112; grads 176/176 finite;
`features.0.0.weight` updated; trajectory `6.1112 → 4.7847`. **Identical to the B18c run → the default path
is unchanged by this edit** (`used_pretrained=False`).

## Downloads avoided?
**Yes.** No network/download was attempted. The IMAGENET1K_V2 checkpoint is **not currently cached**
locally, so the pretrained constructor was deliberately **not** invoked (cache-check-then-skip). Smokes use
`pretrained=False` and never download.

## Env-gating note
Actual ImageNet weight **download/load is environment-gated**. The wiring is complete and verified
offline; the **real E1 run** must run where the torch-hub cache is populated or network is available, then
call `build_student(pretrained=E1_STUDENT["init_weights"])` (or `pretrained=True`). The expected cache file
is `…/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth`.

## Files changed / created
- `src/models/student.py` — `_build_backbone` resolves `pretrained` (incl. exact config string) with
  graceful offline failure; `build_student`/`PlantSegStudent` docstrings/annotations updated. No arch change.
- `reports/imagenet_init_wiring.md` — this report.

_Not training. No download. `docs/reference/reference.pdf` untouched (deferred). Nothing committed._
