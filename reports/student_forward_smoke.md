# Student forward-pass smoke test — B10

**RESULT: PASS**  
_Generated: 2026-06-27 by `scripts/smoke_student_forward.py` (forward pass only — no training, no quantization)._

## Model summary
MobileNetV3-Large (quantizable TorchVision variant, `dilated=True` → output stride 16) truncated to
`features[0:16]` (the 960-ch conv at idx 16 is dropped — unused by the head), feeding an **LR-ASPP-style
head** with **additive class-logit fusion**:
- high-level C5 (160ch, OS16) → `1×1 Conv-BN-ReLU → 256` **⊙** `sigmoid(avgpool → 1×1 conv → 256)`
- product → bilinear upsample to OS8 → `1×1 → num_classes`
- low-level skip (40ch, OS8) → `1×1 → num_classes`
- **sum** the two logit maps → bilinear upsample to 512×512
- `QuantStub`/`DeQuantStub` wrappers present (FP32 pass-through; **no prepare/convert run**).
- Native Hard-Swish/ReLU retained (**no ReLU6 substitution**).

| Property | Value |
|---|---|
| `num_classes` used | **116** (from `configs/data.py`) |
| Backbone source | `torchvision.models.quantization.mobilenet_v3_large` (torchvision 0.24.1+cpu), `dilated=True`, `quantize=False` |
| ImageNet-pretrained init | **`NEED_TO_CONFIRM`** — `weights=None` (hub cache empty; no download performed) |
| Output shape (dummy & real) | **`[2, 116, 512, 512]`**, dtype `torch.float32`, finite (no NaN/Inf) |
| Parameter count | **2,933,688 (~2.93M)** |

## Backbone stage probe (`[1,3,512,512]`, `dilated=True`)
| idx | out_ch | size | stride | note |
|--:|--:|:--|:--|:--|
| 0 | 16 | 256×256 | OS2 | |
| 1 | 16 | 256×256 | OS2 | |
| 2 | 24 | 128×128 | OS4 | |
| 3 | 24 | 128×128 | OS4 | |
| 4 | 40 | 64×64 | OS8 | |
| 5 | 40 | 64×64 | OS8 | |
| **6** | **40** | 64×64 | **OS8** | **← low-level skip tap** |
| 7 | 80 | 32×32 | OS16 | |
| 8 | 80 | 32×32 | OS16 | |
| 9 | 80 | 32×32 | OS16 | |
| 10 | 80 | 32×32 | OS16 | |
| 11 | 112 | 32×32 | OS16 | |
| 12 | 112 | 32×32 | OS16 | |
| 13 | 160 | 32×32 | OS16 | C5 (160-ch) begins |
| 14 | 160 | 32×32 | OS16 | |
| **15** | **160** | 32×32 | **OS16** | **← high-level C5 tap** |
| 16 | 960 | 32×32 | OS16 | dropped from the student (`features[0:16]`) |

## Selected taps
| Tap | Index | Channels | Output stride |
|---|--:|--:|:--|
| OS8 low-level skip | **6** | **40** | OS8 |
| OS16 high-level C5 | **15** | **160** | OS16 |

- OS16 C5 has **160 channels** — confirmed (matches the contract's CWD 160-ch C5; no `NEED_TO_CONFIRM` on taps).

## `NEED_TO_CONFIRM` items
1. **ImageNet-pretrained initialization** — built with `weights=None` for the smoke (torch-hub cache empty; no
   download per the no-download rule). Wire `MobileNet_V3_Large_Weights.IMAGENET1K_V2` once weights are
   available in the pinned env, then re-confirm.

## Deferred (out of B10 scope; not run here)
Quantization (`QuantStub`/`DeQuantStub` are present but inert): `FloatFunctional` for the `⊙`/`+` ops,
Conv-BN-ReLU fusion, and PTQ/QAT prepare/convert are deferred to the quantization blockers. No loss /
distillation code written.
