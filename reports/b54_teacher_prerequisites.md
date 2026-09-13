# B54 — teacher (B1) prerequisites: one blocker, not two

**Type:** readiness audit + resolution plan. **Status:** **ONE blocker** — the ADE20K init
checkpoint. The dependency/image blocker **does not exist**; it was a premise, and the repository
refutes it.
**Date:** 2026-09-13. **Scope:** what must be true before the SegNeXt-B teacher fine-tune (B1) can
start. B1 blocks E2 and E3.
**Read-only.** Nothing was downloaded, no GPU was used, no image was rebuilt, and no governed path
was edited. Sibling records: [B52](b52_e1_seed42_completion.md) (E1 seed 42),
[B53](b53_uvm_host_fault.md) (the pods), [B8](../docs/B8_checkpoint.md) (no released checkpoint).

---

## 1. Correcting the premise

This audit was commissioned to resolve **two** blockers: the teacher checkpoint, and a
teacher-stack dependency gap in the container image. **The second does not exist.** The teacher
stack — mmcv, mmengine, mmsegmentation, opencv — is already installed in the image that ran E1, and
was confirmed present at runtime on the pod.

Recording the correction rather than quietly dropping it, because the reasoning that produced it is
reusable: the mistake was reading `requirements-e1.txt`'s student-only exclusion as a statement
about the **image**, when it is a statement about a **file**. §3 sets out why.

| Blocker | Status |
|---|---|
| ADE20K SegNeXt-B / MSCAN-B init checkpoint | **OPEN** — §2 |
| Teacher-stack dependencies in the image | **NOT A BLOCKER** — §3 |
| Teacher VRAM / pod sizing | **NOT DETERMINABLE**, and it gates provisioning — §4 |

---

## 2. The one real blocker — the init checkpoint

### 2.1 Why it must be downloaded rather than found

[B8](../docs/B8_checkpoint.md) settled this on 2026-06-27: **no publicly released trained SegNeXt-B
PlantSeg checkpoint exists.** The PlantSeg repo's README mentions no weights, its GitHub Releases
page is empty, every linked URL points at the dataset, the paper, or MMSegmentation install docs,
and the Zenodo record holds `plantseg.zip` — dataset only. Reproduction is documented as
train-it-yourself.

So the teacher must be fine-tuned in-house from the **ADE20K-pretrained** MSCAN-B init, targeting
42.05 % mIoU ± 1.5–2.0 pp. B8 carries a standing re-check trigger: if a checkpoint is later
published, switch to download-and-verify instead of fine-tuning.

### 2.2 The four unresolved fields

From [`docs/teacher_init_source.md`](../docs/teacher_init_source.md), verbatim:

| Line | Field | Value |
|---|---|---|
| `:11` | Exact `.pth` URL (resolved by `mim`) | `NEED_TO_CONFIRM` (paste from the `mim download` log) |
| `:12` | SHA256 of the `.pth` | `NEED_TO_CONFIRM` (`sha256sum weights/<file>.pth`) |
| `:13` | Download date | `NEED_TO_CONFIRM` |
| `:17` | Init-test result (`scripts/test_teacher_init.py`) | `NEED_TO_CONFIRM` (PASS/FAIL) |

**All four remain `NEED_TO_CONFIRM`.** Nothing was downloaded in this session, so no URL, no hash
and no date could be obtained, and none is written into the record. §2.4 explains why `:17` could
not be closed either, contrary to plan.

Already known and **not** in doubt: MMSeg version 1.2.2 (mmcv 2.1.0, torch 2.1.0), reported ADE20K
mIoU 48.03 SS / 49.68 MS, source OpenMMLab model zoo under Apache-2.0.

### 2.3 What the init test actually asserts

`scripts/test_teacher_init.py` is genuinely read-only — it loads, runs one dummy forward, and never
trains, fine-tunes or modifies the checkpoint. It **PASSes iff all three** hold (`:99-101`):

- `len(missing_keys) == 0`
- `len(unexpected_keys) == 0`
- `decode_head.conv_seg.out_channels == 150`

It needs a config `.py` and a `.pth` under `weights/`, auto-globbed at `:48-49`, and runs **on CPU**
— `device="cpu"` at both `:71` and `:81`. No GPU is required at any point. `weights/` and `*.pth`
are git-ignored; the checkpoint is never committed.

The 150-class assertion is deliberate: it verifies the **stock** ADE20K architecture loads cleanly
*before* anything is changed. The 150 → 116 re-initialisation of `decode_head.conv_seg` happens
later, in the training pipeline, and the downloaded checkpoint is never edited
(`teacher_init_source.md:19-23`).

### 2.4 The harness check was attempted and could not be completed here

The plan was to run the script as-is, confirm the MM environment imports and that execution reaches
the file check, then set `:17` to "harness verified, awaiting checkpoint". **That is not what
happened, so `:17` stays `NEED_TO_CONFIRM`.**

```text
$ python scripts/test_teacher_init.py
Traceback (most recent call last):
  File "/home/user/plantseg-thesis/scripts/test_teacher_init.py", line 28, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
EXIT CODE: 1
```

**Cause: the environment, not the script's logic.** This repository checkout runs in a bare
Python 3.11.15 container with no numpy, no torch and no MM stack — `pip list` shows conan,
cryptography, Jinja2 and similar tooling only. The pinned MM environment lives in the RunPod image,
which is not what this session executes in. Installing numpy to get past it is forbidden and would
prove nothing about the pinned environment anyway.

**A real defect surfaced on the way, worth recording `[MEASURED]`:** the script's import guard at
`:54-60` catches `mmseg` / `mmengine` import failure and exits **2** with actionable guidance
("Run inside the pinned env: torch 2.1.0, mmcv 2.1.0, mmseg 1.2.2"). But `import numpy as np` sits
at **module scope, line 28 — outside that guard**. In any environment missing numpy the script
therefore produces a bare traceback and exit **1**, not the clean exit 2 it was designed to give.
The guard does not cover its own hardest failure case. Queued as **G11** *(raised as N21)* — `scripts/**` is on rule 8's path list, so it is governed
regardless of how small the fix is; see §5.1.

**`:17` is closable in one CPU-only command** the next time the pinned environment is available —
before any download, since exit 2 with "config/checkpoint not found" is itself the confirmation
that the harness runs.

---

## 3. The image is not a blocker — evidence

### 3.1 The teacher stack is in the lock the image actually installs

| Package | `requirements-runpod.lock` | Pinned |
|---|---|---|
| mmcv | `:92` | `2.1.0` |
| mmengine | `:94` | `0.10.7` |
| mmsegmentation | `:96` | `1.2.2` |
| opencv-python | `:106` | `4.8.1.78` |
| **albumentations** | **absent from every lock file** | — |

albumentations is the one genuine absence, and it is **not a teacher blocker** — it is the subject
of the G5 governed decision ([B52 §9.1](b52_e1_seed42_completion.md),
[open_questions #5](../docs/open_questions.md)).

### 3.2 That lock is what gets installed

`Dockerfile:59-62`:

```dockerfile
RUN python -m pip install --require-hashes --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html \
      -r requirements-runpod.lock
```

The `--find-links` index is OpenMMLab's prebuilt-wheel index for **exactly** cu121 / torch 2.1.0.

### 3.3 Confirmed at runtime, not only from the Dockerfile

`verify_env` on the E1 pod, inside image `sha256:b80b645d…866aaf`:

```text
mmcv            : actual=2.1.0    pinned=2.1.0     (compiled ops present)
mmseg           : actual=1.2.2    pinned=1.2.2
mmengine                             : PASS
mmcv                                 : PASS
```

Dockerfile intent and runtime fact agree. The digest is the one attested in `step0.log` and in the
E1 `[provenance]` line on all five pods of 2026-09-11.

### 3.4 Why the premise looked true — the file/image distinction

`requirements-e1.txt:10-11` excludes opencv-python and albumentations, and
`docs/ai_guardrails.md:30` says the E1 install carries "NO mmcv/mmseg/teacher". Both are correct
**about that file**. But the Dockerfile never installs it:

- `:53` `COPY requirements-runpod.lock requirements-runpod.in ./` → installed at `:59-62`.
- `:71` `COPY requirements.lock requirements-e1.txt ./` → **copied only**, as the comment there says:
  "readable version registry, kept for provenance reporting inside the container".

**`requirements-e1.txt` is a registry, not an installation manifest.** The student-only invariant
describes what a bare-metal E1 install needs; it does not describe the image, which is deliberately
a superset so that one image serves every stage.

### 3.5 mmcv is not a long-lead item

The contract anticipated this. `IMPLEMENTATION_CONTRACT.md:379` pins torch 2.1.0+cu121 as a **hard
ceiling — "mmcv prebuilt-wheel constraint"**, and `:651-653` records that
`mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl` exists on the OpenMMLab cu121/torch2.1.0 index for
exactly this combination, so **"MMCV is never compiled"**. The pod confirms it: compiled ops
present. There is no CUDA-matched build to wait on.

### 3.6 The second-image plan is dropped, and that is the better outcome

A second image was the preferred shape *on the assumption that something had to be added*. Nothing
does. **Doing nothing is strictly better than adding an image:**

- E1 seed 42 is attested against `sha256:b80b645d…866aaf`. Seeds 2 and 3 must run against the same
  digest, and they will, because no rebuild is proposed.
- A second image would create a second attested environment to track, reconcile and cite, for zero
  capability gain.
- It keeps G8 (the Dockerfile `ENV` proposal) at its recorded default of **don't**: no rebuild is
  forced by anything here, so nothing rides along.

---

## 4. G2 — teacher VRAM is not determinable, and it gates provisioning

### 4.1 What D30 established, and its stated limit

[`open_questions.md:747`](../docs/open_questions.md) (D30) falsified `IMPLEMENTATION_CONTRACT.md`
`:249-252`'s **INFERRED 11–14 GB** band — a figure the contract itself flagged as "not a GO — settle
it on the pod".

**Measured** (B48 §2; RTX 4090, batch 16, 512², 116 classes):

| Condition | Peak |
|---|---|
| `use_deterministic_algorithms(True)` — as production runs | **20.667 GiB** |
| without | **17.014 GiB** |
| **determinism penalty** | **+3.653 GiB, ×1.2147 — a ~21.5 % increase** |

The omitted term is the 12.658 GiB forward transient from rerouting `F.interpolate` to
`torch._decomp.decompositions.upsample_bilinear2d_vec`.

D30 is **student-scoped**, and says so: it generalises across E1/E2/E3/E5/E6 because they share the
116-class 512² head, and it explicitly leaves *"whether E2/E3 add materially on top of E1's
20.667 GiB"* as INFERRED until measured. **It says nothing about the teacher.**

### 4.2 The only teacher number in the repository, and why it cannot be used

`docs/teacher_prep_runbook.md:166`: *"ADE20K MSCAN-B at batch-16 / 512² peaks around **~31 GB**"*,
which drove the A6000 recommendation at `:167` (≥ ~32 GB) and the `:172` prohibition on shrinking
batch or crop to fit 24 GB.

Three defects make it unusable as a sizing basis:

1. **Unsourced and unmeasured.** No citation, no probe, no `max_memory_allocated()` behind it.
2. **Wrong class count.** It is an **ADE20K (150-class)** figure. The teacher fine-tune carries a
   **150 → 116 decode-head swap** (`teacher_init_source.md:19-23`) that is not in it. The direction
   of that change is not obvious enough to hand-wave: fewer classes shrink the final logits, but
   the swap is a structural difference the quoted figure never modelled.
3. **It predates the determinism finding.** If the teacher's decode path hits the same deterministic
   upsample at 512², its true peak is higher than quoted — by how much, nobody has measured.

### 4.3 An illustrative scaling, which is not an estimate

> **`[ILLUSTRATIVE ONLY — DO NOT SIZE AGAINST THIS]`** Applying the student's measured ×1.2147
> determinism penalty to the unsourced ~31 GB gives **~37.7 GB**.

This arithmetic is recorded for exactly one purpose: **to show why the probe is the gate.** It
multiplies a measured ratio by an unsourced number, from a different architecture at a different
class count, and inherits every defect in §4.2. It is not an estimate, it is not a requirement, and
it must not appear in any provisioning decision or in Chapter 4. It is here because ~31 GB and
~37.7 GB fall on opposite sides of enough card boundaries to make the difference between a pod that
works and one that OOMs at iteration 2 — which is precisely how E1's launch-1 failed (B48).

### 4.4 Verdict and the gating measurement

**Teacher VRAM: NOT DETERMINABLE from any artifact available to this repository.** No number is
estimated into the record.

**The gate is a memory probe, same shape as B48 §2**, run on whatever card is rented first and
**before** committing to a full teacher fine-tune:

| Probe parameter | Value |
|---|---|
| Model | SegNeXt-B / MSCAN-B, teacher fine-tune config, **116 classes** (post decode-head swap) |
| Batch / crop | **16** / **512×512** — the locked B1 recipe, unmodified |
| Optimizer | AdamW, as B1 specifies |
| Work | **one batch: forward + backward.** No optimizer step, no training, no checkpoint |
| Determinism | **enabled exactly as production** — `set_seed()` as `src/seeds.py:35-37` configures it |
| Report | `torch.cuda.max_memory_allocated()` **and** `torch.cuda.max_memory_reserved()`, both in GiB |
| Cost | minutes; a fraction of one pod-hour |

Report both allocated and reserved: B48's OOM was a fragmentation failure at iteration 2, where
reserved-but-unallocated memory was the difference, so allocated alone would understate the card
needed.

**This probe is the item that unblocks provisioning.** Until it returns, the A6000 recommendation at
`teacher_prep_runbook.md:167` rests on an unsourced figure, and no card can be chosen on evidence.
Run it on the first teacher pod, before the fine-tune, and record the result against D30.

---

## 5. Config-name reconciliation

Two spellings exist. The executable path is consistent; the prose lags.

| Site | Spelling | Era |
|---|---|---|
| `src/distill/segnext_teacher.py:35` | `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512` | MMSeg 1.x |
| `scripts/test_teacher_init.py:32` | same | MMSeg 1.x |
| `docs/teacher_init_source.md:10` | same | MMSeg 1.x |
| `configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:44` | `_base_ = ['mmseg::segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py']` | MMSeg 1.x |
| `docs/IMPLEMENTATION_CONTRACT.md:123` | `segnext_mscan-b_512x512_160k_ade20k` **(or equivalent)** | MMSeg 0.x |
| `configs/teacher_finetune.py:9` | `"init_checkpoint": "segnext_mscan-b_512x512_160k_ade20k"` | MMSeg 0.x |

**Operative name: `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512`** — it is what `mim` resolves and
what every executable site uses.

**This is a wording lag, not a conflict.** The contract's "(or equivalent)" already admits the 1.x
spelling, so nothing is being executed against a name the contract forbids, and no behaviour is
wrong today.

### 5.1 Classification of the wording fix — governed, and here is why the shortcut fails

The fix touches `docs/IMPLEMENTATION_CONTRACT.md:123` and `configs/teacher_finetune.py:9`. **Both are
governed**, and the reasoning matters more than the answer:

> **The path list is a floor, not a ceiling** ([B52 §11.1](b52_e1_seed42_completion.md)). Anything on
> rule 8's list is governed **regardless of consequence**. Consequence can only ADD items; it can
> never remove one.

`docs/IMPLEMENTATION_CONTRACT.md` and `configs/**` are both named directly by rule 8. The tempting
argument — *"the contract's phrasing already permits the name, so the fix is cosmetic and therefore
non-governed"* — is consequence-classification applied to something already on the list, which is
the exact inversion the floor-not-ceiling rule exists to prevent. **Harmlessness is not a route off
the list.** Queued as **G10**, governed.

The `test_teacher_init.py:28` import-guard defect (§2.4) is queued as **G11** *(raised as N21)* —
governed by the same rule, since `scripts/**` is on the list. Both carry their originally-raised
names in parentheses so earlier references still resolve.

---

## 6. What is needed before B1 can start

| # | Item | State | Cost |
|---|---|---|---|
| 1 | Download the ADE20K init via `mim`, capture URL + SHA256 + date | **OPEN** — §2.2 | minutes, CPU, no GPU |
| 2 | Run `scripts/test_teacher_init.py` → `RESULT: PASS` | **OPEN** — §2.4; harness unverified | seconds, CPU |
| 3 | Teacher VRAM probe → card choice | **OPEN, gating** — §4.4 | minutes, one GPU pod |
| 4 | Teacher-stack dependencies | **DONE** — already in the image, §3 | zero |
| 5 | A pod | Secure tier, per [B53](b53_uvm_host_fault.md) §4; size from item 3 | — |

Items 1 and 2 need no GPU and can be done on any machine with the pinned environment. **Item 3 is
the only one that requires renting anything**, and it should be the first thing that pod does.

**Not in scope of this report, and not decided here:** G5 (albumentations), G8 (Dockerfile `ENV`),
G9 (`set_seed` assertions), G10 (config-name wording), G11 (the import guard). All are governed, all need a plan and an
explicit go, and G5/G8/G9 carry the shared "do not land mid-campaign" default recorded at
[B52 §11.3](b52_e1_seed42_completion.md).
