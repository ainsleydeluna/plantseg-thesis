# Week 1 — Formal Closeout

**Verdict: Week-1 CORE = ✅ PASS** (Week 2 may begin) · **strict original G0 gate = ⚠ NOT fully closed**
(open items in §7). Core setup, validation, dataset proof, smoke evidence, and documentation readiness are
committed with passing evidence and no remaining item blocks B17 — but the stricter original Week-1 / G0
guide has missing / deferred / env-gated items that are **not** hidden (see §7).

**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD at closeout:** `e73d712` (`add B16 dataset audit
reports`). _Synthesis of the Week-1 completeness audit; no Week-1 work was redone._

---

## 1. Verdict
- **Week-1 CORE: ✅ PASS** — every core Week-1 task is committed with passing evidence; the ch3 sync audit
  reports **0 conflicts**; the dataset is proven valid end-to-end (B16, `e73d712`).
- **Week 2 may begin** — no remaining item blocks **B17** (E1 train-step smoke); see §6.
- **Strict original Week-1 / G0 gate: ⚠ NOT fully closed** — stricter original-guide items remain missing /
  deferred / env-gated (platform verification, teacher-init weights + load test, the formal
  `reports/week1_G0_status.md`, GitHub remote/push, a persisted dataloader smoke report). These are **not**
  Week-2 blockers, so Week 1 is *practically* closed but **not** *formally / strictly* closed. Full
  reconciliation in **§7**.

## 2. Timeline boundary
- **Week 1 (this closeout) — setup / validation / dataset proof / smoke evidence / docs readiness.** DONE.
- **Week 2 — implementation wiring**, starting with **B17 — E1 train-step smoke**, then the E1 training loop.
- **Later — teacher fine-tune (B1, env-gated), distillation (E2/E3), quantization (E4–E7), corruption suite,
  statistics, Chapter 4 artifacts.**

> **B17 is implementation wiring, not setup validation — it belongs to Week 2, not Week 1.**

## 3. Completed Week-1 evidence
| Item | Evidence | Commit / file |
|---|---|---|
| Environment lock (pinned training stack) | torch 2.1.0+cu121, mmcv 2.1.0, mmseg 1.2.2, mmengine 0.10.7, openmim, numpy 1.26.4, statsmodels 0.14.6 … | `requirements.lock` (`a7cefef`) |
| Implementation contract / configs / seed | contract + `configs/{data,augment,model,loss,e1_student,teacher_finetune,distill,quant}.py` + `src/seeds.py` | `f9c5a45`, `2b94a77`, `f458c81` |
| Dataset location register / `.gitignore` harden | `reports/dataset_location_log.md` (root external; raw data git-safe) | `c4048c4` |
| **B16 dataset audit** (filename + decode level, all splits) | `reports/dataset_audit_summary.md` + 16 B16 reports | `e73d712` |
| Dataloader smoke | `scripts/smoke_dataloader.py` → `[PASS]` (`[2,3,512,512]`/`[2,512,512]`) | `1bbbeee` |
| Student forward smoke (B10) | `reports/student_forward_smoke.md` — PASS; `[2,116,512,512]`, ~2.93M params; taps idx6/40ch (OS8), idx15/160ch (OS16) | `764093e`, `82b0c6c` |
| Losses / metrics / loss-backward smokes (B11/B11b) | `reports/losses_metrics_smoke.md`, `loss_smoke.md`, `metrics_smoke.md` — PASS; finite losses+grads; 255 ignore-invariance verified; mIoU sanity | `84fa62a`, `56149a9` |
| CWD projection stub smoke (B12) | `reports/cwd_projection_smoke.md` — PASS; bias-free 1×1 160→320, 51,200 params (320 LOCKED) | `0527823` |
| QNNPACK proxy smoke (B7/B13/B7b) | `docs/b7_result.md` + `reports/b7_qnnpack.log` — onednn-proxy `PASS_CLEAN`; QNNPACK `BLOCKED_ENV` | `1d082b2`, `82b0c6c` |
| Run manifest | `reports/run_manifest.json` (local dev env capture) | `a94af20` |
| Docs / ch3 sync audit | `docs/{IMPLEMENTATION_CONTRACT,conflicts,open_questions,B8_checkpoint}.md`; `docs/updated_ch3_sync_audit.md` — **PASS, 0 conflicts** | `f9c5a45`, `b46176c` |

**Pre-training gates (Table 3.1):** mask class-count ✅, image↔mask pairing ✅, split integrity ✅, mask
value-range ✅ — all satisfied by `dataset_report` + B16. (Gate 5 CWD-softmax + teacher-320-hook and gate 6
QNNPACK Sigmoid are E3/E4/teacher gates, not E1 gates — see §5.)

## 4. Proven dataset facts (B16)
- **7,774** complete image–mask pairs (train **5,367** / val **846** / test **1,561**); **0 silent drops**.
- **`num_classes = 116`**; **background = raw mask label 0**; **disease labels = raw mask labels 1–115**.
- **`mask = category_id + 1`** — proven **100%** against split-local COCO (all annotated images).
- **`reduce_zero_label = False`** (masks already encode bg=0 + diseases 1–115).
- Raw masks contain **0–115 only**; **255 is preprocessing pad/ignore only** (absent from released masks).
- _(Non-blocking)_ `Metadata.csv` has **12** wrong `Index` values (0.15%) — masks/COCO authoritative; no
  pipeline impact.

## 5. Carry-forward items — **NOT Week-1 blockers**
None of these block closing Week 1 or starting Week 2 (B17). Listed with when each is actually needed:
- **Before the E1 *training run*** (not the B17 smoke):
  - real **CE class weights** computed over the full train set (current weights are smoke/batch-derived);
  - **multi-scale / random-resized-crop** augmentation wired into the dataloader (declared in
    `configs/augment.py`, not yet in `transforms.py`);
  - **ImageNet student init** (`MobileNet_V3_Large_Weights.IMAGENET1K_V2`; currently `weights=None`).
- **Later / environment-gated:**
  - **teacher-init B1** (SegNeXt-B/MSCAN-B) setup in the pinned MMSeg env (`mim download` + init test);
  - **authoritative QNNPACK** run in the pinned torch-2.1 + QNNPACK env (local is onednn proxy only).
- **Before final evaluation / Chapter 4:**
  - **disease-only mIoU convention** (exclude background class 0) — data side already settled
    (`reduce_zero_label=False`); only the reporting convention remains (`open_questions` #2).

## 6. Week-2 readiness statement
- **No remaining blocker prevents Week 2 from starting.** B17's prerequisites (student, dataloader, CE+Dice
  loss, metrics, seed util, local CPU torch) are all present and individually smoke-verified; B17 needs no
  teacher and no downloads (build student with `weights=None`).
- **Recommended Week 2 Task 1: B17 — E1 train-step smoke** (single CPU train step; wiring + gradient-flow
  proof; no real training run, no GPU, no download, no checkpoint).

## 7. Original Week 1 Guide Reconciliation
Reconciliation of this closeout against the **stricter original Week-1 / G0 setup guide**. The core is PASS
and Week 2 (B17) is unblocked, but the rows below show the strict G0 gate is **not fully closed**. Nothing
here is hidden.

| Original G0 guide item | Current repo evidence | Status | Blocks W2 / B17? | When to resolve |
|---|---|---|---|---|
| No training during Week 1 (G0 rule) | all smokes are forward / single-step gradient checks; **no training run executed** | **DONE** | N/A | — |
| Dataset download log vs location log | `reports/dataset_location_log.md` registers the **manual** download ("nothing downloaded by task"); **no** separate `reports/dataset_download_log.md` | **PARTIAL** | NO | optional — location-log serves the purpose, or add an explicit download log |
| B8 teacher-checkpoint check | `docs/B8_checkpoint.md` — none public → in-house fine-tune | **DONE** | NO | — |
| Teacher init weights + load test | `weights/` **empty**; `scripts/test_teacher_init.py` present but **unrun**; `docs/teacher_init_source.md` fields = `NEED_TO_CONFIRM` | **ENV-GATED / NOT DONE** | NO | teacher phase — pinned MMSeg env (`mim download` + `test_teacher_init.py`) |
| Compute-platform verification | `scripts/verify_env.py` **missing**; `reports/platform_verify.md` **missing**; only `reports/run_manifest.json` (LOCAL dev env, **not** the RunPod/compute target) | **NOT DONE** (local manifest only) | NO | before E1 *real run* / when provisioning compute |
| Final `reports/week1_G0_status.md` | **present** (`reports/week1_G0_status.md`) | **DONE** | NO | — |
| Persisted dataloader smoke report | `scripts/smoke_dataloader.py` + commit `1bbbeee` + `[PASS]` stdout; **no** `reports/dataloader_smoke.md` | **PARTIAL** | NO | optional — persist a report `.md` |
| QNNPACK: proxy vs authoritative | head + full-student smokes present; `docs/b7_result.md` + `reports/b7_qnnpack.log` = onednn **proxy `PASS_CLEAN`**; QNNPACK backend unavailable locally | **DONE (proxy) / ENV-GATED (authoritative)** | NO | pinned torch-2.1 + QNNPACK env |
| GitHub remote / push status | `git remote -v` **empty**; `master` has **no upstream**; **not pushed** | **NOT DONE / UNKNOWN** (no remote) | NO | when a remote is configured |

**Reconciliation summary:** the original Week-1 / G0 guide is **only *practically* satisfied for Week-2
entry**, **not *fully / formally* closed**. The unclosed strict items — **compute-platform verification**,
**teacher-init weights + load test**, the formal **`reports/week1_G0_status.md`**, **GitHub remote/push**,
and the **persisted dataloader smoke report** — are documented above and are **not** B17 blockers, but they
should be completed before the formal G0 sign-off (and before the E1 real run / compute provisioning, where
noted).

---

_Week-1 **core** closed at `e73d712` (strict G0 gate partially open — see §7). `docs/reference/reference.pdf`
remains a pre-existing deferred dirty file and was not touched. This closeout introduces no
code/config/dataset changes._
