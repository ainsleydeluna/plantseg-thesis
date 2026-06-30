# Strict-G0 Closeout Update (B20c)

Post-Week-1 addendum recording strict-G0 progress since the Week-1 snapshot. **Docs-only; no training,
no code/config/dataset changes, nothing pushed.** This is the **authoritative current strict-G0 status**;
it supersedes the affected rows of [`reports/week1_G0_status.md`](week1_G0_status.md) §3/§4 and
`docs/WEEK1_CLOSEOUT.md` §7 (which **predates** B20a/B20b and is intentionally left unedited).

**Generated:** 2026-06-30 · **Branch:** `master` · **HEAD:** `73aa06d` · **Remote:** none configured.
**Updated:** 2026-06-30 (B20e · HEAD `c5e5d16`) — the **optional dataset download/location log** is now **DONE** via **B20d `c5e5d16`** (`reports/dataset_download_log.md`, a provenance/location log — **not** a fresh audit); moved from §2 to §1 below.

**Strict-G0 overall: ⚠ PARTIAL** (improved — **three items now closed** since the Week-1 snapshot; not fully closed: only **GitHub remote/push** and **teacher-init weights/load test** remain). **This update task: ✅ PASS.**

---

## 0. Why this addendum exists
`reports/week1_G0_status.md` names `docs/WEEK1_CLOSEOUT.md` §7 as the single source of truth. Because the
closeout is intentionally **not** edited in this step, this dated addendum is the post-Week-1 record of
record for strict-G0 status; `week1_G0_status.md` carries a forward-pointer banner to it and has its two
now-closed rows updated in place.

## 1. Strict-G0 items NOW CLOSED since the Week-1 snapshot (`e73d712`)
| Item | Status | Commit | Evidence |
|---|---|---|---|
| **Compute-platform verification** | **✅ DONE** | **B20a `7452b89`** | `scripts/verify_env.py` + `reports/platform_verify.md`. Verdict **PARTIAL**: this machine is **OK for dry-run only, NOT for real E1**. The script is reusable and re-runs on the actual GPU target for a PASS there. |
| **Persisted dataloader smoke report** | **✅ DONE** | **B20b `73aa06d`** | `reports/dataloader_smoke.md`: committed `scripts/smoke_dataloader.py` PASS (train+val) + scratchpad probe **7/7** (test batch, normalization sanity, effective `reduce_zero_label=False`, val determinism, train RRC label-safety). |
| **(Optional) dataset download / location log** | **✅ DONE** | **B20d `c5e5d16`** | `reports/dataset_download_log.md` — provenance/location log (manual acquisition; DOI 10.5281/zenodo.17719108, CC BY-NC 4.0; external root + depth-1 layout + all six split dirs; reuses audit counts 7,774 = 5,367/846/1,561; bg=0, diseases 1–115, num_classes=116, 255=pad/ignore, effective `reduce_zero_label=False`). **Not a fresh audit.** |

**Also landed (Week-2 implementation, not an original Week-1 row):**
| Work | Status | Commit | Evidence |
|---|---|---|---|
| **B19 — E1 training-loop scaffold** | **✅ complete (scaffold); CPU dry-run PASS** | **`93e57bb`** | `src/training/train_e1.py` + `reports/e1_train_scaffold_dryrun.md` (4 CPU iters, finite loss, optimizer step, per-iteration PolynomialLR, accumulate-one-CM validation, all-class-mIoU checkpoint selection; checkpoint to temp only; real 80k gated behind `--real-run --confirm-real-run`). Readiness audited in `reports/e1_training_loop_readiness.md`. |

## 2. Strict-G0 items STILL DEFERRED (why strict-G0 is not fully closed)
| Item | Status | Why deferred |
|---|---|---|
| **GitHub remote / push** | **deferred** | No remote URL provided yet; `git remote -v` empty, `master` has no upstream, nothing pushed. Outward-facing — awaits an explicit remote URL + visibility. |
| **Teacher-init weights + load test** | **env-gated / deferred** | Needs the pinned MMSeg env (`mim download … --dest weights/` → `python scripts/test_teacher_init.py` → PASS; fill the 4 `NEED_TO_CONFIRM` fields in `docs/teacher_init_source.md`). Teacher phase (E2/E3), not E1. |

> Because these **two** remain open (GitHub remote/push; teacher-init weights/load test), **strict-G0 overall stays ⚠ PARTIAL** — not fully/formally closed. _(The optional dataset download/location log moved to §1 — DONE via B20d `c5e5d16`.)_

## 3. Is the B19 scaffold complete?
**Yes — the scaffold is complete and its CPU dry-run PASSED** (`93e57bb`; `reports/e1_train_scaffold_dryrun.md`).
Only the **real E1 run** is outstanding, and it is environment-gated, not code-gated.

## 4. Can real E1 training run on this machine?
**No.** This is a CPU-only host (`torch 2.9.1+cpu`, `cuda_available=False`). The 80k-iter run at batch 16 /
512² requires a GPU and the pinned `cu121` stack.

## 5. Exact blocker list for the real E1 run
(from `reports/platform_verify.md`)
1. **No CUDA GPU** — `cuda_available=False`; local torch is the CPU-only build `2.9.1+cpu`.
2. **torch/torchvision mismatch vs pinned** — local `2.9.1+cpu` / `0.24.1+cpu` ≠ pinned `2.1.0+cu121` / `0.16.0+cu121` (`requirements.lock`).
3. **ImageNet backbone not cached** — `mobilenet_v3_large-5c1a4163.pth` absent; `--init imagenet` needs network (one-time) or a pre-staged checkpoint.
4. **Teacher/aug stack absent locally** — `cv2`, `albumentations`, `mmcv`, `mmseg` not installed (teacher fine-tune / E2–E3, **not** E1 supervised; listed for completeness).

> E1-specific hard blockers: #1 (and matching #2). #3 only if using ImageNet init. #4 is later-phase.

## 6. Exact next choices
- **(a) GitHub remote/push** — provide a remote URL + visibility (public/private); then add the remote and push `master` (no force, no `--no-verify`). Closes the last environment-independent strict-G0 hygiene item.
- **(b) Provision compute for real E1** — stand up a GPU target with the pinned stack (`pip install -r requirements.lock` → `torch 2.1.0+cu121`), re-run `scripts/verify_env.py` there for a PASS, populate the ImageNet cache, then launch `python src/training/train_e1.py --real-run --confirm-real-run --init imagenet --ckpt-dir <out-of-repo path>`.
- **(c) Continue local docs/review work** — e.g. pre-E2/E3 (teacher / distillation) planning that doesn't need GPU. _(The optional dataset download/location log is now DONE — B20d `c5e5d16`.)_

---

## Provenance / guardrails honored
Docs-only. Edited exactly two approved files (this addendum + a surgical `reports/week1_G0_status.md`
update). **`docs/WEEK1_CLOSEOUT.md` intentionally NOT edited**; `docs/reference/reference.pdf`,
`docs/reference/context.md`, and teacher-init files untouched; no code/config/model/data/loss/training
changes; no training, download, GPU, or push; nothing staged or committed; Claude memory and `~/.claude`
untouched.
