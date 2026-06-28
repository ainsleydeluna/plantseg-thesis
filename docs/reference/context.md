# CLAUDE.md — Resource-Constrained Plant Lesion Segmentation (KD + INT8 QAT)

Controlled computer-vision evaluation: compress a SegNeXt-B teacher into a mobile-deployable
MobileNetV3-Large + LR-ASPP student via knowledge distillation (Logit KD + CWD) and INT8
quantization, evaluated on PlantSeg. This is a **controlled evaluation, not a deployment study** —
no on-device/ARM claims. Single-seed vertical slice (E1→E7 + full eval) before scaling to multi-seed.

<!-- MAINTAINER NOTE (stripped from Claude's context): update the paths in "Reference docs"
     to match this repo's actual layout. The chapters currently live as ch1/ch2/ch3.pdf +
     reference.pdf; convert/point to whatever markdown or PDF lives in the repo. -->

## Reference docs — read the relevant one BEFORE acting, do not duplicate it here
Pointers only (no `@` import, so they load on demand, not every session):
- `docs/ch3-methodology.md` — **single source of truth** for method/protocol. When ch3 conflicts
  with anything (including this file or older notes), ch3 wins. Read it before any design decision.
- `docs/ch1-2-background.md` — rationale, hypotheses (H1a–H1d), alternatives-rejected.
- `docs/references.md` — citation list (two divergent lists exist historically; treat as
  needs-verification, never fabricate a DOI/venue/author-year).
- `papers/` — primary sources: Shu 2021 (CWD), Hinton 2015 (logit KD), Jacob 2018 +
  Krishnamoorthi 2018 (quant), Nagel 2019 (segmentation INT8 anchor), Howard 2019 (MobileNetV3),
  Guo 2022 (SegNeXt), Wei 2026 (PlantSeg), Hendrycks 2019 + Kamann 2020 (robustness).

## Golden rules — these are load-bearing; violating one invalidates the study
- **IMPORTANT — `num_classes` is NOT settled. NEVER hardcode it.** Masks are documented as indices
  0–114, but ch3 is internally inconsistent (115 vs 116, background-as-separate-index unresolved).
  Run `np.unique()` on the real annotation PNGs, then size the final classifier to the **verified**
  count. Settle this before E1.
- **IMPORTANT — Table 3.1 verification is mandatory before ANY training**: mask class count,
  image↔mask pairing (name/readability/dims), split-integrity (zero ID overlap), mask value range
  (⊆ class set ∪ {255}), ignore-label unit tests. Store verification logs as JSON next to each run.
- **`ignore_index = 255` everywhere** — CE, Dice, Logit KD, CWD, and all metrics exclude 255.
- **Seed 42** across `torch`/`numpy`/`random`; set `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA.
- **NEVER bump the locked stack.** `torch==2.1.0+cu121` is a hard ceiling from the mmcv prebuilt-wheel
  constraint needed for the MMSeg teacher. Newer torch breaks the teacher bridge.
- **CWD 1×1 projection head is training-only** — remove it before E6 and E7 (and it is absent from
  every evaluated model).
- **NO distillation loss during QAT.** E5/E6 are supervised-only; adding teacher signal conflates
  quantization with continued KD and breaks ablation attribution.
- **NEVER touch val / clean-test / corrupted-test sets.** Augmentation is train-only; corruptions are
  applied to test images (uint8 RGB, before normalization) and **never to masks**.
- **Keep MobileNetV3 native activations** (Hard-Swish/ReLU) via the quantizable TorchVision variant;
  the global ReLU6 substitution is **rejected**.
- **Locked decisions are fixed.** If something seems better, FLAG it for approval — do not silently
  override. Read the actual source file before recommending; flag version ambiguity, don't assume.

## Environment & commands
Stack (pinned; mirror in `requirements.lock` + Dockerfile): Python 3.11 · torch 2.1.0+cu121 ·
torchvision 0.16.0 · MMSeg 1.2.2 · mmcv 2.1.0 · numpy 1.26.4 · scipy 1.11.4 · scikit-image 0.20.0 ·
OpenCV 4.8.1 · Albumentations · fvcore 0.1.5.post20221221 · statsmodels · vendored imagecorruptions
1.1.2 (`np.float_`→`np.float64` patched). Backend: eager-mode `torch.ao.quantization`, QNNPACK.
- `pip install --break-system-packages` if installing into the system env.
- Smoke test (Kaggle free tier OK): a few iters to confirm shapes/losses, **plus** verify QNNPACK
  supports INT8 Sigmoid in the LR-ASPP global-pool branch — gate this before E4.
- Pilot before any full run: **2,000-iteration E1 pilot on RunPod** (do not commit to full runs without it).
- Full runs on **RunPod RTX 4090** (~$0.34/hr); log GPU-hours + cost per stage.

## Pipeline (E1–E7 + Teacher)
| Stage | From | Adds |
|---|---|---|
| Teacher | ADE20K-pretrained SegNeXt-B/MSCAN-B | Fine-tune on PlantSeg → ~42.05% mIoU ref (not deployed) |
| E1 | ImageNet-pretrained quantizable MobileNetV3-L | FP32 baseline (no KD, no quant) |
| E2 | independent, E1 init | + Logit KD |
| E3 | independent, E1 init | + Logit KD + CWD (**proposed FP32 student**) |
| E4 | E1 ckpt | INT8 PTQ |
| E5 | E1 ckpt | INT8 QAT |
| **E6** | **E3 ckpt (head removed)** | **INT8 QAT — final proposed pipeline** |
| E7 | E3 ckpt (head removed) | INT8 PTQ |

## Locked training config (reference card — do not re-derive)
**E1/E2/E3 share an identical recipe:** SGD momentum 0.9, lr 1e-2, poly decay (power 0.9),
weight-decay 1e-4, batch 16, **80,000 iters @ 512×512**, validate every 4,000, keep best-val-mIoU ckpt.
- Supervised loss (all stages): `L_sup = L_CE + L_Dice` (equal weight). CE uses **√ inverse-frequency**
  class weights (over non-255 pixels, normalized by median); Dice is unweighted, soft, smoothing 1e-5.
- Distillation weights ramp 0→target over the first epoch; global-norm grad clipping throughout.
- Teacher runs in eval mode online on the identical augmented input as the student.

**E2 Logit KD:** CE + KL on softened outputs, `T_Logit=4`. `λ_logit` chosen by validation sweep over
`{0.25, 0.5, 1, 2, 4}` (seed 42, report value in Ch4); the selected λ is **reused unchanged in E3**.

**E3 CWD** (Shu 2021): `T_CWD=4`, `α_CWD=50` (feature map, stride-16 C5), `β_CWD=3` (logit map),
`T²/C` normalization with `C=320` (MSCAN-B stride-16 Stage-3). Training-only 1×1 head maps student
160-ch C5 → teacher 320-ch. Loss: `L_E3 = L_CE + L_Dice + λ_logit·L_LogitKD + 50·L_CWD_feat + 3·L_CWD_logit`.

**E5/E6 QAT:** SGD momentum 0.9, lr 3e-4 cosine decay, ~15 epochs, early-stop on val mIoU. Activation
quant on from step 0 (moving-avg observers); freeze BN stats after ~65–70%, freeze observers shortly
after; global-norm clip; **no EMA of weights**; best-val-mIoU ckpt. E6 starts from E3 (head removed), E5 from E1.

**E4/E7 PTQ:** static INT8, ~128 calibration images. E4 from E1, E7 from E3 (head removed).

**Teacher:** AdamW lr 6e-5, wd 0.01, betas (0.9,0.999), decode-head `lr_mult=10`, poly schedule,
40,000 iters, batch 16, 512×512, CE loss. Success = recover 42.05% within ±1.5–2.0 pp (protocol match, not max accuracy).

**Quantization detail:** per-channel **symmetric INT8** weights (all conv); per-tensor **asymmetric
UINT8** activations. Fuse Conv-BN-ReLU before observer insertion; Hard-Swish/Hardsigmoid as standalone
quantized ops. Report full INT8 only after backend conversion + operator-support verification. Any
unsupported op → FP32 fallback, reported.

## Architecture
MobileNetV3-Large via `torchvision.models.quantization.mobilenet_v3_large` (quantizable variant) +
LR-ASPP head. Output stride 16 via dilation rate 2 in the C4–C5 depthwise convs (stage stride→1; no
parallel atrous pyramid). LR-ASPP: high-level C5 → [1×1 Conv-BN-ReLU 256ch] ⊙ [avgpool→1×1 conv→
sigmoid→bilinear], upsampled to OS8; low-level OS8 skip → 1×1 conv to `num_classes`; high branch → its
own 1×1 conv to `num_classes`; the two summed and upsampled to full res. Output channels = **verified** count.

## Data & preprocessing
PlantSeg (Wei 2026): 7,774 imgs, 115 disease classes, 34 hosts, official **70/10/20** (~1,554 test);
JPEG images + PNG grayscale masks (indices 0–114, 255 = ignore). Identical preprocessing on **all**
partitions: aspect-ratio-preserving resize (long side→512), symmetric pad short side to 512 (image pad =
ImageNet mean [124,116,104]; mask pad = 255), bilinear (image) / nearest (mask), scale [0,1], ImageNet
norm mean [0.485,0.456,0.406] std [0.229,0.224,0.225]; image→float32 CHW, mask→int64 HW. Train-only aug
(Albumentations): hflip p=0.5, vflip p=0.5, rotation ±10° p=0.5 (before crop, fill 255 on masks),
hue/saturation (image only).

## Evaluation & statistics
- **Accuracy:** per-image **disease-only mIoU** (background excluded — the paired unit for tests) +
  **dataset-level all-class mIoU** (basis for non-inferiority + bootstrap). Dice = dataset-level
  secondary (no separate test). mAcc descriptive. Absent classes excluded per-image; 255 excluded always.
- **Robustness mIoU-C:** mean over **5 corruptions** (motion blur, Gaussian noise, JPEG, brightness, fog),
  severities **1–3** for summary/inferential; sev 4 descriptive-only; sev 5 excluded. Plus RPD/rCD.
- **8 inferential tests** (one-tailed **Wilcoxon signed-rank**, `scipy.stats.wilcoxon(zero_method='pratt',
  alternative='greater')`; paired t-test sensitivity; **Holm-Bonferroni** α=0.05 via statsmodels):
  E1vE2, E2vE3, E4vE5, E7vE6, E4vE7, E5vE6, E1vE6 (mIoU), E1vE6 (mIoU-C).
- **Non-inferiority E3 vs E6 — SEPARATE from the 8**: pass iff BCa-95% CI lower bound on ΔmIoU > **−2.0 pp**.
- Effect sizes: rank-biserial r_rb, Cohen's dz, Hodges-Lehmann shift, mean ΔmIoU, **BCa 95% CI, B=10,000**
  (`scipy.stats.bootstrap(method='BCa')`; documented percentile/basic fallback for degenerate zero-inflated
  medians). HL BCa at B=10,000 ≈ 7–8 min (O(n²) Walsh averages) — plan for it.
- **Efficiency = descriptive only:** params, serialized model size, FLOPs=2×MACs via fvcore @512×512
  (from FP32 arch; QAT does not cut FLOPs), CPU-proxy latency via `torch.utils.benchmark` (CPU, batch 1,
  20 warm-up + 100 measured, report median/IQR/p95), peak memory via `resource.getrusage`/psutil RSS delta.
  x86 CPU proxy → benchmark a separate fbgemm/x86 INT8 copy (`reduce_range=True`); QNNPACK copy stays the
  reported accuracy/size artifact.

## Working conventions
Be decisive (single verdict, not option menus); minimal-change edits preserving humanized prose (passive
voice, no first person); citations use `;` between co-citations and `et al.,` (no space before comma);
"needs verification" discipline on all reference details — no fabricated DOIs/venues/years. Conflicts with
locked decisions are documented and flagged, never silently applied. Verify source files before claims.
