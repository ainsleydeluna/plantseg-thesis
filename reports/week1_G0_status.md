# Week 1 — G0 Gate Status

Formal G0 (pre-training) gate artifact. Reconciles the **practical Week-1 core** with the **stricter
original Week-1 / G0 guide**. Companion to [docs/WEEK1_CLOSEOUT.md](../docs/WEEK1_CLOSEOUT.md) §7 (single
source of truth — this artifact mirrors it; if they ever diverge, the closeout wins). **Doc-only synthesis;
no Week-1 work redone; no training started.**

**Generated:** 2026-06-28 · **Branch:** `master` · **HEAD:** `e73d712` · **Remote:** none configured.

> **⏩ Post-Week-1 update (2026-06-30 · B20c · HEAD `73aa06d`):** the rows below are the **Week-1
> snapshot** (`e73d712`). Since then **two strict-G0 items have CLOSED** — **compute-platform
> verification** (B20a `7452b89`) and the **persisted dataloader smoke report** (B20b `73aa06d`) — and
> the **B19 E1 training-loop scaffold** (`93e57bb`, CPU dry-run PASS) landed. Those two rows in §3/§4 are
> updated in place. **Authoritative current status: [`reports/strict_G0_closeout_update.md`](strict_G0_closeout_update.md).**
> `docs/WEEK1_CLOSEOUT.md` §7 **predates** B20a/B20b and is superseded for those two items by the
> addendum (the closeout itself is intentionally left unedited). **Strict-G0 overall remains ⚠ PARTIAL**
> — GitHub remote/push deferred (no URL yet), teacher-init env-gated, optional download log deferred.
> **B19 scaffold is complete and its dry-run passed, but the real E1 run is still blocked on a GPU/CUDA
> stack and the ImageNet cache/download.**

---

## 1. G0 verdict
- **Week-1 CORE: ✅ PASS** — setup, validation, dataset proof (B16), smoke evidence, and docs readiness are
  committed with passing evidence; ch3 sync audit = 0 conflicts.
- **Strict original G0 gate: ⚠ NOT fully closed** — several stricter original-guide items remain missing /
  deferred / env-gated (see §3, §4).
- **Week 2 / B17 may begin** — **no** missing strict-G0 item blocks **B17** (E1 train-step smoke). B17's
  prerequisites are all present and individually smoke-verified; B17 needs no teacher and no downloads.

> **Bottom line:** Week 1 is *practically* closed (core PASS, Week 2 unblocked) but **not** *formally /
> strictly* closed until the §4 items are done.

## 2. Table 3.1 / pre-training gate checklist
| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Mask class count | **✅ PASS** | `np.unique` over all masks → 0–115; `num_classes=116`. `reports/dataset_report.{md,json}`, `reports/dataset_audit_summary.md` |
| 2 | Image↔mask pairing | **✅ PASS** | 7,774 complete pairs, **0 drops**, 0 unpaired/dim-mismatch. B16 split-consistency audits (train/val/test) |
| 3 | Split integrity | **✅ PASS** | zero cross-split identifier overlap; per-split stem sets equal (images=masks=JSON). `dataset_report.json`, B16-4/8/12 |
| 4 | Mask value range | **✅ PASS** | masks ∈ 0–115 only, **255 absent** (pad/ignore only). `reports/test_mask_value_audit.md`, `reports/trainval_mask_value_audit.md` |
| 5 | Ignore-label behavior | **◑ PARTIAL** | CE/Dice **255-invariance** verified (`loss_smoke.md`); metrics **255-exclusion** verified (`metrics_smoke.md`). **Deferred:** CWD spatial-softmax-sums-to-1 over valid locations + teacher stride-16 320-ch forward-hook — CWD loss is a placeholder and no teacher is built (E3 / teacher phase, not an E1 gate). |
| 6 | QNNPACK / quantization smoke | **✅ DONE (proxy) / ⛓ ENV-GATED (authoritative)** | onednn-proxy `PASS_CLEAN` for head + full student. `docs/b7_result.md`, `reports/b7_qnnpack.log`. Authoritative QNNPACK + INT8-Sigmoid gate is an **E4** gate (pinned torch-2.1+QNNPACK), not an E1 gate. |

**E1-relevant gates (1–4) are fully PASS.** Gate 5's open parts and gate 6's authoritative run are E3/E4/teacher gates, not E1/B17 blockers.

## 3. Original Week-1 guide reconciliation
Consistent with `docs/WEEK1_CLOSEOUT.md` §7.

| Original G0 guide item | Current repo evidence | Status | Blocks W2 / B17? | When to resolve |
|---|---|---|---|---|
| No training during Week 1 (G0 rule) | all smokes are forward / single-step gradient checks; **no training run executed** | **DONE** | N/A | — |
| Dataset download log vs location log | `reports/dataset_location_log.md` registers the **manual** download ("nothing downloaded by task"); **no** separate `reports/dataset_download_log.md` | **PARTIAL** | NO | optional — location-log serves the purpose, or add an explicit download log |
| B8 teacher-checkpoint check | `docs/B8_checkpoint.md` — none public → in-house fine-tune | **DONE** | NO | — |
| Teacher init weights + load test | `weights/` **empty**; `scripts/test_teacher_init.py` present but **unrun**; `docs/teacher_init_source.md` fields = `NEED_TO_CONFIRM` | **ENV-GATED / NOT DONE** | NO | teacher phase — pinned MMSeg env (`mim download` + `test_teacher_init.py`) |
| Compute-platform verification | **✅ DONE (B20a, `7452b89`)** — `scripts/verify_env.py` + `reports/platform_verify.md` capture this machine; verdict **PARTIAL** (OK for dry-run only, **NOT** real E1). Reusable script re-runs on the real GPU target. | **✅ DONE** | NO | a PASS verdict on the actual GPU target is pending provisioning |
| Final `reports/week1_G0_status.md` | **this file** | **DONE (now)** | NO | — |
| Persisted dataloader smoke report | **✅ DONE (B20b, `73aa06d`)** — `reports/dataloader_smoke.md`: committed smoke PASS + scratchpad probe 7/7 (train/val/test valid; labels ⊆ {0..115,255}; val deterministic; train RRC) | **✅ DONE** | NO | — |
| QNNPACK: proxy vs authoritative | head + full-student smokes; `docs/b7_result.md` + `reports/b7_qnnpack.log` = onednn **proxy `PASS_CLEAN`**; QNNPACK backend unavailable locally | **DONE (proxy) / ENV-GATED (authoritative)** | NO | pinned torch-2.1 + QNNPACK env |
| GitHub remote / push status | `git remote -v` **empty**; `master` has **no upstream**; **not pushed** | **NOT DONE / UNKNOWN** (no remote) | NO | when a remote is configured |

## 4. What remains to FORMALLY close strict G0
_Updated 2026-06-30 (B20c). **Closed since the Week-1 snapshot:** ~~compute-platform verification~~ (B20a `7452b89`) and ~~persisted dataloader smoke report~~ (B20b `73aa06d`). Remaining items:_
1. **Teacher-init weights + teacher load test** — in the pinned MMSeg env: `mim download … --dest weights/`, then `python scripts/test_teacher_init.py` → `RESULT: PASS`; fill the 4 `NEED_TO_CONFIRM` fields in `docs/teacher_init_source.md`. **(env-gated / deferred)**
2. **GitHub remote / push** — configure a remote and push `master` (so the work is backed up / verifiable). **(deferred — no remote URL provided yet)**
3. **(Optional) explicit dataset download log** — `reports/dataset_download_log.md`, or formally accept `reports/dataset_location_log.md` as serving this purpose. **(optional / deferred)**

> Teacher-init also gates the teacher phase / E1 *real-run* prerequisites; GitHub push + download log are hygiene/provenance. **None gate B17.** **Strict-G0 overall remains ⚠ PARTIAL** — see [`reports/strict_G0_closeout_update.md`](strict_G0_closeout_update.md).

## 5. Statement
- **None of the remaining strict-G0 items blocks B17.**
- **B17 — E1 train-step smoke remains Week 2, Task 1** (implementation wiring, not setup validation).
- **No training has been started** — all Week-1 evidence is forward-pass / single-step gradient smokes and
  read-only audits; the G0 "no training in Week 1" rule is intact.

---

_Doc-only artifact. `docs/reference/reference.pdf` remains a pre-existing deferred dirty file and was not
touched. No code/config/dataset changes; nothing committed._
