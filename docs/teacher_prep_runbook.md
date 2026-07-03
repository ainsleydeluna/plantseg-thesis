# Teacher Preparation Runbook — B1 (SegNeXt-B / MSCAN-B fine-tune)

> **Status:** Preparation/reference artifact (TEACHER-PREP-B1). **No training, no download, no
> install, no GPU** performed to author this document. It captures the *verified* facts, the
> *derived-config spec*, and the *operating procedure* for the in-house teacher fine-tune so the run
> can be executed later in the pinned MMSegmentation environment on the correct GPU.
> This runbook **does not author** the SegNeXt config `.py`, does not edit any config/code, and does
> not modify the pinned stack. Each of those is a separate, explicitly-approved step.
>
> **Authorities:** method = `docs/reference/ch3.pdf` `[ch3]`; reconciliation = [context.md](reference/context.md)
> `[ctx]`; locked configs = [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §B1/§B6; dataset =
> `[Wei]`/`[empirical]`. Verified init facts corroborated against the OpenMMLab SegNeXt model zoo
> (`[zoo]`). Where a value is not yet in hand it is recorded as the literal `NEED_TO_CONFIRM` — never guessed.

---

## 1. Scope & role

- The teacher is a **descriptive upper-bound reference only** — **never** an inferential comparator, and
  **not deployed** `[ch3 §A; IMPLEMENTATION_CONTRACT.md:39]`. It exists to (a) produce teacher
  logits/features for E2/E3 distillation and (b) provide a descriptive accuracy ceiling.
- **Budget = 40,000 iterations, and this is intentional.** The teacher is **exempt** from the
  "all training runs iteration-matched at 80,000 iters" rule `[IMPLEMENTATION_CONTRACT.md:66]`. That rule
  governs the **E1–E7 comparison set**; the teacher is descriptive and is *not* iteration-matched to the
  students. Do **not** "fix" 40k to 80k.
- Success = **protocol match, not maximum accuracy**: recover **42.05% mIoU within ±1.5–2.0 pp**
  `[teacher_finetune.py:29; IMPLEMENTATION_CONTRACT.md:135]`. The 42.05% is the Wei et al. (2026)
  contextual SegNeXt-B benchmark number — a **descriptive target**, not an inferential comparator.

---

## 2. Verified initialization source `[zoo; teacher_init_source.md]`

| Field | Value | State |
|---|---|---|
| Init type | **FULL MMSeg ADE20K model** (decode head + backbone), **not** an IN-1K backbone-only load | verified |
| MMSeg 1.x config name (canonical) | `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512` | verified `[teacher_init_source.md:10; test_teacher_init.py:32]` |
| Checkpoint `.pth` filename | `segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth` | verified `[zoo]` |
| Checkpoint URL host/path | `https://download.openmmlab.com/mmsegmentation/v0.5/segnext/…` (resolved by `mim`) | verified host `[zoo]` |
| IN-1K backbone (**fallback only**) | `mscan_b_20230227-3ab7d230.pth` — used *only* if the full ADE20K model is unavailable | verified `[zoo]` |
| Stock `num_classes` | **150** (ADE20K) — re-headed to 116 at load (see §5) | verified |
| Params | **~27.6M** (contract benchmark rounds to 28M) | verified `[test_teacher_init.py:34]` |
| Reported ADE20K mIoU | **48.03 (SS) / 49.68 (MS)** | verified `[teacher_init_source.md:15]` |
| Source / license | OpenMMLab MMSegmentation model zoo, **Apache-2.0** | verified |
| SHA256 of the `.pth` | `NEED_TO_CONFIRM` — `sha256sum weights/<file>.pth` at download | pending |
| Download date | `NEED_TO_CONFIRM` — `date -u +%Y-%m-%d` at download | pending |

> **Shorthand note.** `configs/teacher_finetune.py:9` and `IMPLEMENTATION_CONTRACT.md:123` use the loose
> alias `segnext_mscan-b_512x512_160k_ade20k` ("or equivalent"). That is an **alias**, not a valid MMSeg
> 1.x config id. The canonical name above (already correct in `scripts/test_teacher_init.py` and
> `docs/teacher_init_source.md`) is what `mim download` resolves. Do not pass the alias to `mim`.

> **The downloaded checkpoint is never edited.** Re-heading 150→116 happens at load time in the runner
> (§5); `weights/*.pth` are treated as read-only, git-ignored inputs.

---

## 3. Derived-config spec — `segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py`

> **Specification only — this runbook does NOT author the file.** Authoring the `.py` is a separate,
> explicitly-approved step. Below is exactly what that derived config must contain, expressed as deltas
> from the stock ADE20K config named in §2. Values trace to `[ch3]` §B1 and the verified facts.

**Base:** inherit the stock `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512` config, then override:

| Group | Key | Value | Source |
|---|---|---|---|
| Classes | `decode_head.num_classes` | **116** (150→116) | `[data.py:22; empirical]` |
| Classes | dataset `reduce_zero_label` | **False** (masks already 0–115, bg=0 kept) | `[data.py:23; empirical]` |
| Norm | `norm_cfg` | **`dict(type='BN', requires_grad=True)`** — **override the stock SyncBN** (see §6) | `[ch3 §D; verified]` |
| Backbone | `backbone.init_cfg` | **`None`** — do **not** re-pull the IN-1K backbone; the full ADE20K weights arrive via `load_from` (§5) | verified fact |
| Backbone (inherited, for verification) | `embed_dims / depths / drop_path_rate` | `[64,128,320,512] / [3,3,12,3] / 0.1` | verified `[zoo]` |
| Head (inherited, for verification) | LightHamHead `channels / ham_channels` | `512 / 512` (MSCAN-B) | verified `[zoo]` |
| Head — determinism | NMF / Hamburger (`ham_kwargs`) seed | **set explicitly** (LightHamHead NMF is random-init + eval-seed-sensitive) — see §6 | `[ch3 §D; verified]` |
| Optim | optimizer | **AdamW**, `lr=6e-5`, `weight_decay=0.01`, `betas=(0.9,0.999)` | `[ch3; teacher_finetune.py:13-16]` |
| Optim | paramwise | **decode-head `lr_mult=10`** | `[ch3; teacher_finetune.py:17]` |
| Sched | policy / iters | **poly**, **40,000** iters | `[ch3; teacher_finetune.py:20-21]` |
| Data | `batch_size` (train dataloader) | **16** (single GPU) | `[ch3; teacher_finetune.py:22]` |
| Data | crop | **512×512** | `[ch3; teacher_finetune.py:23]` |
| Loss | `decode_head.loss_decode` | **cross-entropy** (teacher recipe is CE-only; unlike the student's B5 CE+Dice) | `[ch3; teacher_finetune.py:26]` |
| Runtime | seed | **42** + determinism flags (§11) | `[ch3 §D; ctx]` |

**CWD tap (for downstream E3, recorded here so the teacher config exposes it):** the teacher **stride-16
Stage-3 feature = 320 channels** is the CWD `C_t` source `[IMPLEMENTATION_CONTRACT.md:191; verified]`. A
pre-train gate verifies the teacher hook returns this 320-ch stride-16 feature (§8).

---

## 4. Environment & install order (pinned env; run later, not now)

Pinned stack `[IMPLEMENTATION_CONTRACT.md §B6; requirements.lock]`: Python 3.11 · torch 2.1.0+cu121 ·
torchvision 0.16.0 · **MMSegmentation 1.2.2** · **mmcv 2.1.0** · numpy 1.26.4 · mmengine 0.10.7.

Order matters because of the compat blocker in step 4:

1. Create the Python 3.11 env.
2. Install the pinned training stack:
   `pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu121`
   → expect `torch 2.1.0+cu121`, `torch.cuda.is_available() == True`.
3. Install the **mmcv 2.1.0 prebuilt wheel** from the OpenMMLab index (matches torch 2.1.0 / cu121):
   `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html`
4. Install `mmsegmentation==1.2.2` (+ `mmengine`, `openmim`).
5. **COMPAT BLOCKER — apply the mmseg assertion fix BEFORE any mmseg import** (including `mim` and the
   init test):
   - mmseg 1.2.2's `mmseg/__init__.py` upper-bounds mmcv **below 2.1.0**, so it **rejects** the pinned
     mmcv **2.1.0** at import.
   - **Fix:** open the installed `mmseg/__init__.py`, read the actual constant, and relax the upper bound
     `MMCV_MAX` from `'2.1.0'` → `'2.2.0'` (single-line edit).
   - > **This is NOT a stack bump.** mmcv itself stays **2.1.0** (the pinned, wheel-available version). Only
     > mmseg's *upper-bound check* is relaxed so it accepts 2.1.0. This does not violate
     > `[context.md:34]` "NEVER bump the locked stack." Read the file first and confirm the exact constant
     > name/value before editing (`[context.md:45]` — read the source, don't assume).
6. Download the init config + checkpoint into git-ignored `weights/`:
   `mim download mmsegmentation --config segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512 --dest weights/`
   Capture the printed `.pth` URL.
7. Run the read-only init sanity gate: `python scripts/test_teacher_init.py` → must print **`RESULT: PASS`**
   (see §8).
8. Record the resolved URL, `sha256sum` of the `.pth`, and the UTC date into
   [teacher_init_source.md](teacher_init_source.md) (fills its `NEED_TO_CONFIRM` rows).

---

## 5. Head-swap procedure (150 → 116)

The re-head is a **runner-level weight load**, not a resume and not backbone `init_cfg`:

- Set **`load_from = weights/<ADE20K .pth>`** at the runner/config top level. This loads the full ADE20K
  model as the fine-tune starting point.
- Do **not** use `resume` (that would also restore optimizer/iteration state) and do **not** route the
  ADE20K weights through `backbone.init_cfg` (which loads the backbone only). Set
  **`backbone.init_cfg = None`** (§3) so the IN-1K backbone is not re-pulled.
- With `decode_head.num_classes = 116`, the classifier `decode_head.conv_seg` has a **shape mismatch**
  vs the stock 150-class weight, so MMSeg **auto-drops** that tensor on load and **re-initializes** it to
  116 outputs. Everything else (backbone + LightHamHead body) loads intact.
- Net effect: backbone + neck/head body = ADE20K-pretrained; only the 116-way classifier starts fresh.

> Contrast with §8's init test, which loads the **stock** checkpoint into the **stock 150-class** model and
> requires **zero** missing/unexpected keys. That test validates the *download*; the re-head above is the
> *fine-tune start* and intentionally drops `conv_seg`.

---

## 6. Single-GPU / no-SyncBN + NMF-seed requirements

- **Single GPU only.** Launch non-distributed (one process, one GPU). The stock ADE20K config assumes a
  distributed launch and **SyncBN**.
- **Plain BN, never SyncBN.** Override `norm_cfg = dict(type='BN', requires_grad=True)` (§3). SyncBN on a
  single GPU is at best a no-op wrapper and at worst a launch/eval hazard; the contract mandates plain BN.
  If a merged/inherited config still yields SyncBN modules, convert with
  `mmengine`'s `revert_sync_batchnorm` before training.
- **Set the NMF (Hamburger) seed.** LightHamHead's Hamburger/NMF block is **random-initialized** and its
  evaluation is **seed-sensitive**; without a fixed NMF seed, eval mIoU is non-reproducible across runs.
  Fix it alongside the global seed 42 (§11).

---

## 7. GPU sizing rationale — A6000, not 4090, for the teacher

- ADE20K MSCAN-B at **batch-16 / 512²** peaks around **~31 GB**, which **exceeds the RTX 4090's 24 GB**.
- **The teacher fine-tune therefore requires an A6000 (48 GB)** (or equivalent ≥ ~32 GB card).
- **E1 (and the rest of the student pipeline) is unaffected** — student-only, fits the 4090, and **stays on
  the RTX 4090** `[e1_runpod_launch_runbook.md]`.
- The repo's blanket "RunPod RTX 4090" line `[IMPLEMENTATION_CONTRACT.md:265; context.md:56]` is
  **student/E1-scoped**; it does not cover the teacher. Provision the **A6000** for B1 specifically.
- Reducing `batch_size` or `crop` to fit 24 GB would change the **locked** B1 recipe — do **not** do that
  without separate approval; treat any such change as a flagged deviation.

---

## 8. Init sanity gate + pre-train verification

**Init gate — `scripts/test_teacher_init.py` (read-only; PASS required before fine-tune):**
loads the **stock** config + checkpoint, runs one dummy 512×512×3 forward, and asserts the state_dict
loads into the **stock num_classes=150** model with **zero missing AND zero unexpected keys**, and
`decode_head.conv_seg.out_channels == 150`. This validates that the *download* is the exact stock ADE20K
model before any re-heading.

**Table 3.1 pre-train gates (mandatory before any training) `[ch3; ctx]`:** mask class count via
`np.unique`; image↔mask pairing; split integrity (zero ID overlap); mask values ⊆ class set ∪ {255};
ignore-label unit tests; **teacher hook returns the stride-16 MSCAN-B Stage-3 320-ch feature** (the CWD
`C_t` tap, §3). Store verification logs as JSON next to the run.

---

## 9. Expected mIoU trajectory & success criterion

- **Start:** stock ADE20K SegNeXt-B ≈ **48.03% mIoU (SS)** / 49.68% (MS) `[teacher_init_source.md:15]`.
- **Land:** **~42.05% mIoU ± 1.5–2.0 pp** (all-class PlantSeg) `[teacher_finetune.py:29]`.
- The drop from ADE20K→PlantSeg is expected: 150-class scene parsing → 116-class in-the-wild disease
  segmentation with a large small-lesion regime `[Wei; ch3]`.
- **Success = protocol match, not peak accuracy.** Landing inside ±1.5–2.0 pp of 42.05% confirms the recipe
  reproduces the descriptive upper-bound. The 42.05% is Wei's contextual benchmark, used as a **target**,
  never as an inferential comparator `[ch3 §A]`.

---

## 10. Out-of-repo checkpoint export discipline

- **No public PlantSeg SegNeXt-B checkpoint exists** (`[B8_checkpoint.md]`, checked 2026-06-27) — the teacher
  **must** be fine-tuned in-house.
- `weights/` and `*.pth` are **git-ignored**; **no checkpoint is ever committed** (init ckpt or trained
  teacher).
- Write the **trained teacher checkpoint to an out-of-repo path** (same discipline as the E1 `--ckpt-dir`
  outside the repo `[verify_env.py; strict_G0_closeout_update.md]`).
- Record provenance for **both** checkpoints (init + trained teacher): SHA256, UTC date, resolved URL (for
  the init), producing config name, and the init-test/`RESULT` line. Init-ckpt provenance goes in
  [teacher_init_source.md](teacher_init_source.md).
- The trained teacher is consumed **online, in eval mode**, feeding identical augmented inputs to the
  student during E2/E3 `[IMPLEMENTATION_CONTRACT.md:151]`, and for descriptive eval — **not deployed**.

---

## 11. Shared-protocol invariants (teacher ≡ student)

The teacher and every student stage **must** share these, or distillation/comparison is invalid `[ch3 §C]`:

- **Class space:** `num_classes = 116`, **background = index 0**, **`reduce_zero_label = False`**
  `[data.py:22-25; empirical]`.
- **Dataset:** the same **7,774-image** PlantSeg set (Zenodo **10.5281/zenodo.17719108**, CC BY-NC 4.0),
  **70/10/20** split = **train 5,367 / val 846 / test 1,561** `[Wei; empirical]`.
- **Preprocessing:** 512×512, aspect-preserving resize + pad, ImageNet mean/std normalization,
  **`ignore_index = 255`** excluded from loss & metrics `[data.py; ch3]`.
- **Determinism:** **seed 42** across `torch`/`numpy`/`random`; before CUDA init set
  `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` `[ch3 §D; ctx]` — **plus the NMF seed** (§6).

---

## 12. Cross-reference map

| Topic | Authority in repo |
|---|---|
| Locked B1 recipe / B6 versions | [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §(d) B1, §B6 |
| B1 config artifact (analysis only) | [configs/teacher_finetune.py](../configs/teacher_finetune.py) |
| Class space / split / `reduce_zero_label` | [configs/data.py](../configs/data.py) |
| Init-checkpoint source record | [docs/teacher_init_source.md](teacher_init_source.md) |
| No public checkpoint → in-house plan | [docs/B8_checkpoint.md](B8_checkpoint.md) |
| Init sanity gate (read-only) | [scripts/test_teacher_init.py](../scripts/test_teacher_init.py) |
| Env/GPU verification | [scripts/verify_env.py](../scripts/verify_env.py) |
| Pinned-stack rules ("never bump") | [docs/reference/context.md](reference/context.md) |
| Disease-only / `reduce_zero_label` reporting (open) | [docs/open_questions.md](open_questions.md) #2 |

---

### Open items carried by this runbook (`NEED_TO_CONFIRM` — fill at execution, never guess)
- Init `.pth` **SHA256** + **download date** + exact resolved URL (from the `mim download` log).
- `scripts/test_teacher_init.py` **PASS/FAIL** result in the pinned env.
- Exact mmseg `MMCV_MAX` constant value on disk (read before editing to 2.2.0).
- Final RunPod **A6000** pod type / CUDA image (reported in Ch4).
- Recovered teacher mIoU (produced by the fine-tune run; out of preparation scope).
