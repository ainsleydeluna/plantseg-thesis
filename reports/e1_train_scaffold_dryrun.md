# B19 — E1 Training-Loop Scaffold Dry-Run

**RESULT: ✅ PASS** (6/6 hard checks, exit 0). **This is a wiring/dry-run scaffold — NOT real E1 training.**

The E1 components compose into one iteration-based supervised loop with per-iteration polynomial LR,
accumulate-one-confusion-matrix validation, all-class-mIoU checkpoint selection, and a provisional
disease-only mIoU. Run on CPU, random init (no download), 4 iterations, checkpoint to a temp dir only.

_Generated: 2026-06-30. Interpreter `C:\Users\admin\anaconda3\python.exe` (torch 2.9.1+cpu), CPU._

---

## 1. PASS/FAIL — **PASS**
| Check | Result |
|---|---|
| `batch_shapes` (img `[2,3,512,512]` f32, mask `[2,512,512]` i64) | PASS |
| `logits_shape` (`[2,116,512,512]`) | PASS |
| `loss_finite` (scalar, finite) | PASS |
| `optimizer_step` (a tracked param moved) | PASS |
| `val_cm_accumulated` (one `[116,116]` CM, >0 px, ≥1 batch) | PASS |
| `lr_non_increasing` (per-iteration poly decay monotonic) | PASS |

## 2. Command used
```
PYTHONIOENCODING=utf-8 "C:/Users/admin/anaconda3/python.exe" src/training/train_e1.py --dry-run
```
(No `--ckpt-dir` given ⇒ the script auto-created a temp dir via `tempfile.mkdtemp`.)

## 3. Repo HEAD
`9539e744ed34865f9f6db965e97715f429b51f20` (`9539e74` — *add Pre-B19 E1 training-loop readiness audit*).

## 4. Files changed / created
- **Created:** `src/training/train_e1.py` (the scaffold), `reports/e1_train_scaffold_dryrun.md` (this report).
- **Modified:** `src/eval/metrics.py` — added the public helper `miou_from_confusion(cm, class_indices=None)` (delegates to `_miou_from_cm`; no behavior change to existing functions).
- **Not touched:** configs, dataset, losses, model, data files; `docs/reference/reference.pdf`; `docs/reference/context.md`; teacher-init files.

## 5. Dry-run iterations completed
**4 / 4** training iterations; validation fired at iter 2 and iter 4 (`--val-interval 2`).

## 6. Batch shapes and dtypes
First train batch: `img = (2, 3, 512, 512) torch.float32`, `mask = (2, 512, 512) torch.int64`.
Student forward logits: `(2, 116, 512, 512)`.

## 7. Loss finite check
Finite scalar every iteration. Per-iter `loss = ce + dice` (equals `CombinedCEDiceLoss`):
`it1 5.9640 (ce 4.9820 / dice 0.9820)`, `it2 6.8961`, `it3 5.9594`, `it4 6.1681` — all finite.
(4 iters at lr 1e-2 is a plumbing check, not convergence; no trend is expected or required.)

## 8. Optimizer step check
`optimizer_step = PASS` — a tracked trainable parameter changed after `optimizer.step()` (SGD lr 0.01,
momentum 0.9, weight_decay 1e-4 from `configs/e1_student.py`).

## 9. Scheduler / LR behavior
`scheduler = PolynomialLR (total_iters=80000, power=0.9)` (torch 2.9.1 has it; LambdaLR fallback wired
but unused). Stepped **once per iteration, after `optimizer.step()`**. LR trace (monotonic ↓, matches
`0.01·(1−it/80000)^0.9`):
`9.99988750e-03, 9.99977500e-03, 9.99966250e-03, 9.99955000e-03`.
The horizon is fixed at the real **80 000**-iter curve regardless of dry-run length, so these are the
true early-E1 LR values.

## 10. Validation confusion-matrix accumulation check
`val_cm_accumulated = PASS`. Each validation **accumulates a single `[116,116]` confusion matrix** over
the val batches, then computes mIoU **once** from it. At each val: `cm_batches=2`, `cm_total_px=700928`
(of 2×2×512² = 1,048,576 pixels; the 347,648 padding pixels at index 255 are excluded — confirming
ignore-handling and cross-batch accumulation). `argmax` is applied to logits before counting.

## 11. All-class mIoU and disease-only provisional mIoU (dry-run subset)
- all-class val mIoU = **0.00000**
- disease-only val mIoU (**PROVISIONAL**, excludes background class 0) = **0.00000**

Both are **0.0 as expected for an untrained random-init student** over a 4-image val subset (argmax
collapses to a few classes ⇒ ~zero overlap with GT-present classes). The meaningful result is that the
metric returns a **finite** value (not NaN) and the GT-absent-class skip path works; accuracy is not a
dry-run objective.

## 12. Checkpoint path + outside-repo confirmation
- Best (and only) checkpoint: `C:\Users\admin\AppData\Local\Temp\e1_dryrun_ckpt_32inmqz0\e1_student_best_iter2.pt` (≈23.7 MB).
- **Outside the repo: confirmed.** The dir was auto-created by `tempfile.mkdtemp` in the system temp
  area and passed the hard guard `_assert_outside_repo` (the script raises if a checkpoint dir is the
  repo or inside it). Selection used **all-class val mIoU**: iter-2 (0.0 > −inf) saved; iter-4 (0.0, not
  an improvement) did **not** re-save — best-only selection verified. `git status` shows no `.pt` in the
  repo, and `find . -name '*.pt'` (excluding `.git`) returns nothing.

## 13. No ImageNet download attempted — confirmed
`build_student(pretrained=False)` ⇒ `used_pretrained=False`; the dry-run additionally **raises** if
`used_pretrained` were ever true. `_build_backbone(pretrained=False)` calls
`mobilenet_v3_large(weights=None)` — no torch-hub fetch. (`--init imagenet` is ignored in dry-run.)

## 14. No real training performed — confirmed
Dry-run only: CPU, 4 iterations (vs the real 80 000), batch size 2 (vs 16), `--max-val-batches 2`
(no full val pass). The real 80k path is **gated**: it requires both `--real-run` and
`--confirm-real-run` (verified — `--real-run` alone exits 2 with a refusal; `--confirm-real-run` alone
exits 2). The full run is intentionally not reachable from defaults.

## 15. Remaining blockers before the real E1 run
- **Compute/GPU** — host is CPU-only (`torch 2.9.1+cpu`, `cuda_available=False`); 80k iters at bs 16 /
  512² is infeasible on CPU. Real run needs a GPU.
- **ImageNet cache / download** — `…/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth` is absent;
  the real run's `--init imagenet` must populate the cache (network) or have it pre-staged.
- **Real-run checkpoint destination** — pick a real (out-of-repo) checkpoint dir via `--ckpt-dir`.
- **Open/secondary (non-blocking):** E1 grad-clip `max_norm` (currently omitted — confirm if E1 clips);
  disease-only mIoU exclusion convention (open_questions #2, reported PROVISIONAL); strict-G0 items
  (platform verify, teacher-init for E2/E3, GitHub remote/push).

---

## Provenance / guardrails honored
Approved outputs only: `src/training/train_e1.py`, the `miou_from_confusion` helper in
`src/eval/metrics.py`, and this report. No real training; no GPU; no download; no full dataset scan
(only the few batches pulled were decoded; the split index is globbed by `build_dataloader`); no repo
checkpoint (code-enforced, temp dir only); configs/dataset/losses/model/data unchanged;
`docs/reference/reference.pdf` and `docs/reference/context.md` and teacher-init files untouched; nothing
staged or committed; Claude memory and `~/.claude` untouched.
