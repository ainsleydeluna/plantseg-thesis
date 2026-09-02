#!/usr/bin/env python3
"""E1 training-loop SCAFFOLD (Blocker B19).

Composes the audited E1 components into one iteration-based supervised training loop:
  set_seed(42) -> build_student -> build_dataloader(train/val) -> CombinedCEDiceLoss(class-weighted
  CE + Dice) -> SGD (configs/e1_student.py) -> per-iteration PolynomialLR -> periodic validation that
  ACCUMULATES ONE confusion matrix and computes mIoU once -> best-all-class-val-mIoU checkpoint.

SAFETY MODEL (this file deliberately makes the real 80k run hard to start by accident):
  * Default / `--dry-run`  -> tiny CPU dry-run, random init, NO download, checkpoint to a temp dir.
  * Real run requires BOTH `--real-run` AND `--confirm-real-run`; otherwise the script exits with a
    clear message. The real run is defined but is NOT exercised by B19.

API/wiring verified in reports/e1_training_loop_readiness.md. Grad clipping is omitted by default: no
concrete E1 max_norm exists (the config lists global-norm only under E2/E3 shared mechanics). Disease-only
mIoU is PROVISIONAL (open_questions #2). Checkpoints are NEVER written inside the repo (hard-guarded).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from configs.data import DATA                                   # noqa: E402
from configs.e1_student import E1_STUDENT                       # noqa: E402
from src.data import NUM_CLASSES, build_dataloader             # noqa: E402
from src.eval.metrics import confusion_matrix, miou_from_confusion  # noqa: E402
from src.models.student import build_student                    # noqa: E402
from src.seeds import set_seed                                  # noqa: E402
from src.training.losses import CombinedCEDiceLoss             # noqa: E402

IGNORE_INDEX = DATA["ignore_index"]            # 255
BACKGROUND_INDEX = DATA["background_index"]    # 0
CLASS_WEIGHTS_JSON = REPO / "reports" / "e1_class_weights.json"


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def load_ce_weights(path: Path = CLASS_WEIGHTS_JSON) -> torch.Tensor:
    """Length-116 CE class weights (index-aligned to class id 0..115) from the B18a artifact."""
    with open(path, encoding="utf-8") as f:
        w = json.load(f)["weights"]
    t = torch.tensor(w, dtype=torch.float32)
    if t.numel() != NUM_CLASSES:
        raise ValueError(f"class-weights length {t.numel()} != num_classes {NUM_CLASSES}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("class-weights contain non-finite values")
    return t


def build_scheduler(optimizer, horizon: int, power: float):
    """Per-iteration polynomial LR decay. PolynomialLR if available (torch>=1.13), else a LambdaLR
    fallback that reproduces lr = base * (1 - it/horizon)**power. Returns (scheduler, name)."""
    L = torch.optim.lr_scheduler
    if hasattr(L, "PolynomialLR"):
        return L.PolynomialLR(optimizer, total_iters=horizon, power=power), "PolynomialLR"
    return (L.LambdaLR(optimizer, lr_lambda=lambda it: (1.0 - it / horizon) ** power),
            "LambdaLR(fallback)")


def cycle(loader):
    """Infinite iterator over a DataLoader -> iteration-based (not epoch-based) training."""
    while True:
        for batch in loader:
            yield batch


def _assert_outside_repo(path: Path) -> Path:
    """Hard guard: a checkpoint directory may NEVER be the repo or live inside it."""
    path = path.resolve()
    if path == REPO or REPO in path.parents:
        raise RuntimeError(f"refusing to use a checkpoint dir inside the repo: {path}")
    return path


def resolve_ckpt_dir(ckpt_dir_arg: str | None) -> Path:
    """Prefer an explicit out-of-repo dir; otherwise auto-create a temp dir. Always guarded + created."""
    if ckpt_dir_arg:
        ckpt_dir = _assert_outside_repo(Path(ckpt_dir_arg))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = _assert_outside_repo(Path(tempfile.mkdtemp(prefix="e1_dryrun_ckpt_")))
    return ckpt_dir


def _atomic_save(payload: dict, path: Path) -> str:
    """Write to `<final>.tmp` in the SAME directory, then os.replace().

    Same-directory is required: os.replace is only atomic within one filesystem. A kill mid-write
    destroys the .tmp and leaves the previous good checkpoint at `path` untouched.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic on POSIX and on NTFS
    return str(path)


def _rng_state(train_loader) -> dict:
    """Full RNG snapshot: python / numpy / torch / torch.cuda + the DataLoader generator."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": train_loader.generator.get_state(),
    }


def _restore_rng(state: dict, train_loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_loader.generator.set_state(state["loader_generator"])


def save_checkpoint(ckpt_dir: Path, student, optimizer, scheduler, sched_name: str,
                    it: int, best_miou: float) -> str:
    path = _assert_outside_repo(Path(ckpt_dir)) / f"e1_student_best_iter{it}.pt"
    return _atomic_save({
        "iter": it,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler": sched_name,
        "best_val_miou_all_class": best_miou,
        "num_classes": NUM_CLASSES,
    }, path)


def save_last(ckpt_dir: Path, student, optimizer, scheduler, sched_name: str,
              it: int, best_miou: float, best_ckpt, train_loader, prev_lr) -> str:
    """Periodic resume point. Same fields as `best` PLUS the RNG state needed to continue.

    `prev_lr` is the LR of iteration `it`. Carrying it lets a resumed run compare its FIRST
    iteration against the last iteration of the previous segment, so a k-segment run leaves zero
    unverified LR transitions (V1).
    """
    path = _assert_outside_repo(Path(ckpt_dir)) / "last.pt"
    return _atomic_save({
        "iter": it,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler": sched_name,
        "best_val_miou_all_class": best_miou,
        "best_ckpt": best_ckpt,
        "num_classes": NUM_CLASSES,
        "rng_state": _rng_state(train_loader),
        "prev_lr": prev_lr,
    }, path)


TRACE_KEEP = 50          # lr values retained at each end for the stdout summary (B31-9)


def write_best_pointer(ckpt_dir: Path, best_ckpt, best_miou: float) -> str:
    """Record the current best so downstream tooling never has to glob/parse filenames."""
    path = _assert_outside_repo(Path(ckpt_dir)) / "best.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"best_ckpt": best_ckpt,
                               "best_val_miou_all_class": best_miou}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return str(path)


def prune_checkpoints(ckpt_dir: Path, keep: int, best_ckpt) -> list:
    """Keep the `keep` newest best-checkpoints; never touch last.pt, best.json, or the current best.

    Without this, every val improvement leaves a ~24 MB file behind for the whole 80k run.
    """
    d = _assert_outside_repo(Path(ckpt_dir))
    cks = sorted(d.glob("e1_student_best_iter*.pt"), key=lambda q: q.stat().st_mtime)
    protect = {Path(best_ckpt).name} if best_ckpt else set()
    removed = []
    for q in cks[:-keep] if keep > 0 else []:
        if q.name in protect:
            continue
        try:
            q.unlink()
            removed.append(q.name)
        except OSError:
            pass
    return removed


def _jsonl(path: Path, rec: dict) -> None:
    """Append ONE JSON object as a line. Opened per write so a pod kill cannot lose buffered rows;
    append mode makes it resume-safe (a resumed run continues the same file)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def per_class_iou(cm: torch.Tensor):
    """Per-class IoU from the SAME confusion matrix `validate()` already returned.

    Mirrors `miou_from_confusion`'s UNION-present rule (EVALUATION_CONTRACT 3.1) WITHOUT touching
    src/eval/metrics.py: eligible iff `UN_c = GT_c + PR_c - TP_c > 0`. Nothing is re-accumulated and
    no inference is re-run. The macro mean over eligible classes must equal miou_from_confusion(cm)
    exactly — asserted in the B31-7 acceptance test, which is the guard against this helper drifting
    away from the frozen metric.
    """
    tp = torch.diag(cm).float()
    gt = cm.sum(1).float()
    pr = cm.sum(0).float()
    un = gt + pr - tp
    return tp / un.clamp_min(1e-9), un > 0


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                           text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


@torch.no_grad()
def validate(student, val_loader, device, num_classes: int, max_val_batches: int | None):
    """Accumulate ONE confusion matrix over the val set (or a capped subset), then compute mIoU once."""
    was_training = student.training
    student.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    n_batches = 0
    for i, (img, mask) in enumerate(val_loader):
        if max_val_batches is not None and i >= max_val_batches:
            break
        logits = student(img.to(device))
        pred = logits.argmax(1).cpu()                 # logits -> class indices (required by metrics)
        cm += confusion_matrix(pred, mask, num_classes, ignore_index=IGNORE_INDEX)
        n_batches += 1
    all_miou = miou_from_confusion(cm)
    disease_idx = torch.tensor([c for c in range(num_classes) if c != BACKGROUND_INDEX])
    disease_miou = miou_from_confusion(cm, disease_idx)   # PROVISIONAL (open_questions #2)
    if was_training:
        student.train()
    return all_miou, disease_miou, cm, n_batches


# --------------------------------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------------------------------
def run(*, mode: str, device: str, pretrained, batch_size: int, max_iters: int, val_interval: int,
        max_val_batches: int | None, num_workers: int, ckpt_dir_arg: str | None,
        grad_clip_norm: float | None, log_every: int, seed: int,
        resume: str | None = None, ckpt_interval: int = 2000,
        jsonl_name: str = "e1_telemetry.jsonl", keep_ckpts: int = 3) -> int:
    set_seed(seed)
    dev = torch.device(device)
    print(f"[mode] {mode.upper()} | torch {torch.__version__} | device={dev} | "
          f"cuda_available={torch.cuda.is_available()} | num_classes={NUM_CLASSES}")
    print(f"[budget] max_iters={max_iters} batch_size={batch_size} val_interval={val_interval} "
          f"max_val_batches={max_val_batches} num_workers={num_workers} seed={seed}")

    # --- model ---
    student = build_student(pretrained=pretrained).to(dev)
    student.train()
    if mode == "dry" and student.used_pretrained:
        raise RuntimeError("dry-run must NOT use pretrained weights (no download allowed)")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[student] params={n_params:,} used_pretrained={student.used_pretrained} "
          f"(pretrained arg={pretrained!r})")

    # --- data ---
    # persistent workers on TRAIN only: the pool lives for all 80k iters, whereas val runs ~20 times
    # and the respawn cost there is noise against holding a second worker pool resident (B31-5 Q1).
    train_loader = build_dataloader("train", batch_size, num_workers=num_workers,
                                    persistent_workers=num_workers > 0)
    val_loader = build_dataloader("val", batch_size, num_workers=num_workers)
    print(f"[data] train_index={len(train_loader.dataset)} val_index={len(val_loader.dataset)} "
          f"(index globbed; only the batches pulled below are decoded)")

    # --- loss (class-weighted CE + Dice); weight buffer follows .to(device) ---
    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=IGNORE_INDEX).to(dev)
    print(f"[loss] CombinedCEDiceLoss(weight=len{weights.numel()}, ignore_index={IGNORE_INDEX}) "
          f"weight_on={criterion.ce.weight.device}")

    # --- optimizer (configs/e1_student.py) + per-iteration poly LR ---
    optimizer = torch.optim.SGD(student.parameters(), lr=E1_STUDENT["learning_rate"],
                                momentum=E1_STUDENT["momentum"],
                                weight_decay=E1_STUDENT["weight_decay"])
    horizon = E1_STUDENT["iterations"]            # poly horizon is ALWAYS the real 80k curve
    scheduler, sched_name = build_scheduler(optimizer, horizon, E1_STUDENT["lr_power"])
    print(f"[opt] SGD lr={E1_STUDENT['learning_rate']} momentum={E1_STUDENT['momentum']} "
          f"weight_decay={E1_STUDENT['weight_decay']} | scheduler={sched_name} "
          f"(total_iters={horizon}, power={E1_STUDENT['lr_power']})")
    print(f"[grad-clip] {'disabled (no concrete E1 max_norm)' if grad_clip_norm is None else grad_clip_norm}")

    ckpt_dir = resolve_ckpt_dir(ckpt_dir_arg)
    print(f"[ckpt] dir={ckpt_dir} (verified OUTSIDE repo)")

    # --- persistent telemetry (B31-7). Lives beside the checkpoints, NEVER inside the repo. ---
    jsonl_path = ckpt_dir / jsonl_name
    _jsonl(jsonl_path, {
        "event": "run_meta", "wall_clock": time.time(), "mode": mode, "seed": seed,
        "git_head": _git_head(), "torch": torch.__version__, "numpy": np.__version__,
        "device": str(dev), "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_workers": num_workers, "batch_size": batch_size, "max_iters": max_iters,
        "val_interval": val_interval, "max_val_batches": max_val_batches,
        "ckpt_interval": ckpt_interval, "resumed_from": resume,
        "num_classes": NUM_CLASSES, "ignore_index": IGNORE_INDEX,
        "learning_rate": E1_STUDENT["learning_rate"], "momentum": E1_STUDENT["momentum"],
        "weight_decay": E1_STUDENT["weight_decay"], "lr_power": E1_STUDENT["lr_power"],
        "poly_horizon": E1_STUDENT["iterations"], "grad_clip_norm": grad_clip_norm,
        "used_pretrained": student.used_pretrained, "params": n_params,
    })
    print(f"[jsonl] telemetry -> {jsonl_path}")

    checks: dict[str, bool] = {}
    best_miou = float("-inf")
    best_ckpt = None
    first_batch_meta = None
    lr_head: list = []                  # first TRACE_KEEP lr values
    lr_tail: deque = deque(maxlen=TRACE_KEEP)   # last TRACE_KEEP lr values
    lr_monotonic, prev_lr, n_lr, n_lr_pairs = True, None, 0, 0

    # --- resume (optional) ---
    start_iter = 1
    if resume:
        ck = torch.load(resume, map_location=dev, weights_only=False)
        student.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        best_miou = ck.get("best_val_miou_all_class", float("-inf"))
        best_ckpt = ck.get("best_ckpt")
        if "rng_state" in ck:
            _restore_rng(ck["rng_state"], train_loader)
        # Carry the previous segment's final LR so the monotonicity check spans the resume
        # boundary. Without this a k-segment run leaves k-1 transitions unverified (V1).
        prev_lr = ck.get("prev_lr")
        start_iter = int(ck["iter"]) + 1
        print(f"[resume] from {resume} | resuming at iter={start_iter} "
              f"lr={optimizer.param_groups[0]['lr']:.8e} best_all_class_miou={best_miou:.5f} "
              f"prev_lr={'None (pre-B31c checkpoint)' if prev_lr is None else '%.17g' % prev_lr}")
        print("[resume] WARNING: data-order continuity is NOT restored. The training loop consumes "
              "an infinite `cycle(train_loader)`; the position within the current epoch is not "
              "recoverable, so the post-resume sample order differs from an uninterrupted run. RNG "
              "streams ARE restored, so augmentation remains reproducible from this point onward. "
              "A resumed run is NOT bitwise-identical to an uninterrupted one.")

    if start_iter > max_iters:
        print(f"[resume] nothing to do: the checkpoint is already at iter {start_iter - 1}, which "
              f"meets or exceeds --max-iters {max_iters}. Raise --max-iters to continue training, "
              f"or resume from an earlier checkpoint. No iterations were run and no checkpoint was "
              f"written.")
        # Distinct token: zero checks were exercised, so this must not read as a checked PASS.
        # Exit 0 because "nothing to do" is not an error.
        print("\n[CHECKS] 0/6 exercised, 6 skipped (no iterations ran)")
        print(f"RESULT: NOOP (already complete at iter {start_iter - 1})")
        return 0

    train_iter = cycle(train_loader)
    first_it = start_iter
    t_prev = time.time()

    for it in range(start_iter, max_iters + 1):
        img, mask = next(train_iter)
        img, mask = img.to(dev), mask.to(dev)
        if it == first_it:
            first_batch_meta = (tuple(img.shape), str(img.dtype), tuple(mask.shape), str(mask.dtype))
            checks["batch_shapes"] = (img.shape[1:] == (3, 512, 512) and img.dtype == torch.float32
                                      and mask.shape[1:] == (512, 512) and mask.dtype == torch.int64)

        optimizer.zero_grad(set_to_none=True)
        logits = student(img)
        if it == first_it:
            checks["logits_shape"] = tuple(logits.shape) == (img.shape[0], NUM_CLASSES, 512, 512)
        ce = criterion.ce(logits, mask)
        dice = criterion.dice(logits, mask)
        loss = ce + dice                              # == CombinedCEDiceLoss(logits, mask)
        if mode == "real" and not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"non-finite loss at iter {it}: loss={loss.item():.4f} ce={ce.item():.4f} "
                f"dice={dice.item():.4f}. Aborting the real E1 run to avoid poisoning the model "
                "or wasting compute.")

        if it == first_it:
            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
            p0 = next(p for p in student.parameters() if p.requires_grad)
            before = p0.detach().clone()

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()                              # step ONCE per iteration, after optimizer
        lr = optimizer.param_groups[0]["lr"]
        # B31-9: bounded retention. The monotonicity check is done STREAMING so it still
        # covers every consecutive pair -- only the stdout summary is truncated. The full
        # curve lives in the JSONL, not in a 1.4 MB print.
        if len(lr_head) < TRACE_KEEP:
            lr_head.append(lr)
        lr_tail.append(lr)
        if prev_lr is not None:
            n_lr_pairs += 1                    # count of transitions ACTUALLY compared
            if lr > prev_lr + 1e-12:
                lr_monotonic = False
        prev_lr, n_lr = lr, n_lr + 1

        if it == first_it:
            checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)

        now = time.time()
        _jsonl(jsonl_path, {
            "event": "train", "iter": it, "loss": float(loss.item()), "ce": float(ce.item()),
            "dice": float(dice.item()), "lr": lr, "wall_clock": now,
            "iter_seconds": now - t_prev,
            "samples_per_sec": (batch_size / (now - t_prev)) if now > t_prev else None,
        })
        t_prev = now

        if it % log_every == 0 or it == max_iters:
            print(f"[iter {it:>4}/{max_iters}] loss={loss.item():.4f} ce={ce.item():.4f} "
                  f"dice={dice.item():.4f} lr={lr:.8e}")

        if it % val_interval == 0 or it == max_iters:
            t_val0 = time.time()
            all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
                                                       max_val_batches)
            val_seconds = time.time() - t_val0
            iou_vec, eligible = per_class_iou(cm)
            _jsonl(jsonl_path, {
                "event": "val", "iter": it, "all_class_miou": all_miou,
                "disease_only_miou_PROVISIONAL": disease_miou,
                "per_class_iou": [round(float(x), 8) for x in iou_vec.tolist()],
                "per_class_eligible": [bool(x) for x in eligible.tolist()],
                "n_eligible_classes": int(eligible.sum()), "val_batches": nvb,
                "val_total_px": int(cm.sum()), "val_seconds": val_seconds,
                "wall_clock": time.time(),
            })
            checks["val_cm_accumulated"] = (tuple(cm.shape) == (NUM_CLASSES, NUM_CLASSES)
                                            and int(cm.sum()) > 0 and nvb >= 1)
            print(f"[val  {it:>4}/{max_iters}] cm_batches={nvb} cm_total_px={int(cm.sum())} "
                  f"all_class_miou={all_miou:.5f} disease_only_miou(PROVISIONAL)={disease_miou:.5f}")
            if all_miou > best_miou:
                best_miou = all_miou
                best_ckpt = save_checkpoint(ckpt_dir, student, optimizer, scheduler, sched_name,
                                            it, best_miou)
                write_best_pointer(ckpt_dir, best_ckpt, best_miou)
                pruned = prune_checkpoints(ckpt_dir, keep_ckpts, best_ckpt)
                print(f"[ckpt {it:>4}/{max_iters}] new best all_class_miou={best_miou:.5f} "
                      f"-> {best_ckpt}"
                      + (f" | pruned {len(pruned)} old ckpt(s)" if pruned else ""))

        # periodic resume point (atomic; carries RNG state). Independent of best-val improvement.
        if it % ckpt_interval == 0 or it == max_iters:
            last_path = save_last(ckpt_dir, student, optimizer, scheduler, sched_name,
                                  it, best_miou, best_ckpt, train_loader, prev_lr)
            print(f"[last {it:>4}/{max_iters}] resume point -> {last_path}")

    # scheduler sanity: per-iteration poly decay is monotonically non-increasing (streamed above)
    checks["lr_non_increasing"] = lr_monotonic

    hard = ["batch_shapes", "logits_shape", "loss_finite", "optimizer_step",
            "val_cm_accumulated", "lr_non_increasing"]
    # `all([]) is True` must never print as PASS: with no transition actually compared the
    # monotonicity check is vacuous, so report it SKIPPED. Not fatal -- a legitimate one-iteration
    # resume is not a defect -- but it must be visible (V1).
    skipped = {"lr_non_increasing"} if n_lr_pairs == 0 else set()
    exercised = [k for k in hard if k not in skipped]
    passed = all(checks.get(k, False) for k in exercised)
    print("\n[CHECKS]")
    for k in hard:
        if k in skipped:
            print(f"  {k:18}: SKIPPED (0 LR transitions compared; {n_lr} LR value(s) seen)")
        else:
            print(f"  {k:18}: {'PASS' if checks.get(k) else 'FAIL'}")
    print(f"[CHECKS] {len(exercised)}/{len(hard)} exercised, {len(skipped)} skipped")
    print(f"[summary] lr_transitions_compared={n_lr_pairs} "
          f"(spans the resume boundary when resuming from a checkpoint carrying prev_lr)")
    print(f"[summary] first_batch={first_batch_meta}")
    _h = ['%.8e' % x for x in lr_head]
    _t = ['%.8e' % x for x in lr_tail]
    print(f"[summary] lr_trace n={n_lr} (bounded print: first {len(_h)} / last {len(_t)}; "
          f"full curve in {jsonl_path.name})")
    print(f"[summary] lr_head={_h}")
    if n_lr > len(_h):
        print(f"[summary] lr_tail={_t}")
    print(f"[summary] best_all_class_val_miou={best_miou:.5f} best_ckpt={best_ckpt}")
    print(f"[summary] used_pretrained={student.used_pretrained} (no download in dry-run)")
    # The exercised/skipped counts ride on the RESULT line itself so that a full run, a
    # reduced-coverage run and a NOOP are all distinguishable from the LAST line alone (V1).
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'} "
          f"({len(exercised)}/{len(hard)} checks exercised, {len(skipped)} skipped)")
    return 0 if passed else 1


# --------------------------------------------------------------------------------------------------
# CLI / safety gate
# --------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E1 training-loop scaffold (dry-run by default).")
    p.add_argument("--dry-run", action="store_true", help="tiny CPU dry-run (safe default behavior)")
    p.add_argument("--real-run", action="store_true",
                   help="intent to run the real 80k training (requires --confirm-real-run)")
    p.add_argument("--confirm-real-run", action="store_true",
                   help="explicit confirmation gate for the real 80k run")
    p.add_argument("--device", default=None)
    p.add_argument("--init", choices=["none", "imagenet"], default=None,
                   help="backbone init; 'imagenet' uses configs/e1_student.py (real run only)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--val-interval", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--ckpt-dir", default=None, help="out-of-repo dir; auto temp dir if omitted")
    p.add_argument("--resume", default=None,
                   help="path to a last.pt resume point; restores model/optimizer/scheduler/RNG "
                        "and continues from the saved iter (data ORDER is not restored)")
    p.add_argument("--ckpt-interval", type=int, default=2000,
                   help="write a periodic last.pt resume point every N iters (default 2000)")
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="global-norm clip; omitted by default (no concrete E1 value)")
    p.add_argument("--jsonl-name", default="e1_telemetry.jsonl",
                   help="telemetry filename written inside --ckpt-dir (never inside the repo)")
    p.add_argument("--keep-ckpts", type=int, default=3,
                   help="rolling best-checkpoint retention; best + last.pt are always kept")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.dry_run and args.real_run:
        print("ERROR: pass only one of --dry-run / --real-run.", file=sys.stderr)
        return 2

    # ---- real-run safety gate ----
    if args.real_run:
        if not args.confirm_real_run:
            print("REFUSING to start the real E1 run: --real-run requires --confirm-real-run.\n"
                  "The full 80k training is intentionally NOT runnable from defaults. "
                  "Re-run with: --real-run --confirm-real-run", file=sys.stderr)
            return 2
        mode = "real"
    elif args.confirm_real_run:
        print("ERROR: --confirm-real-run given without --real-run; nothing to confirm.", file=sys.stderr)
        return 2
    else:
        mode = "dry"
        if not args.dry_run:
            print("[mode] No --dry-run/--real-run given; defaulting to SAFE DRY-RUN. "
                  "The real 80k run requires --real-run --confirm-real-run.")

    if mode == "dry":
        device = args.device or "cpu"
        if args.init == "imagenet":
            print("[init] --init imagenet ignored in dry-run (forcing random init, no download).")
        pretrained = False                                   # forced: no ImageNet download in dry-run
        batch_size = args.batch_size or 2
        max_iters = args.max_iters or 4
        val_interval = args.val_interval or 2
        max_val_batches = args.max_val_batches if args.max_val_batches is not None else 2
        num_workers = args.num_workers if args.num_workers is not None else 0
    else:  # real (defined, NOT exercised by B19)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Real E1 must run on CUDA. Refuse a CPU real run (an 80k-iter CPU run is almost certainly a
        # mistake and would take weeks); fail loud rather than silently degrade. No hidden CPU path.
        if not str(device).startswith("cuda"):
            reason = ("--device cpu was passed" if args.device == "cpu"
                      else "CUDA is not available (torch.cuda.is_available()=False)")
            print(f"REFUSING to start the real E1 run on CPU: {reason}.\n"
                  "The real 80k run requires a CUDA GPU. Provision a GPU (see "
                  "reports/e1_runpod_launch_runbook.md), or use --dry-run for a safe CPU smoke.",
                  file=sys.stderr)
            return 2
        init = args.init or "imagenet"
        pretrained = False if init == "none" else E1_STUDENT["init_weights"]
        batch_size = args.batch_size or E1_STUDENT["batch_size"]
        max_iters = args.max_iters or E1_STUDENT["iterations"]
        val_interval = args.val_interval or E1_STUDENT["val_interval"]
        max_val_batches = args.max_val_batches            # None -> full val
        # B31-5: data loading, not the GPU, bounded E1 throughput at the old default of 4.
        default_workers = min(max((os.cpu_count() or 4) - 2, 1), 12)
        num_workers = args.num_workers if args.num_workers is not None else default_workers

    return run(mode=mode, device=device, pretrained=pretrained, batch_size=batch_size,
               max_iters=max_iters, val_interval=val_interval, max_val_batches=max_val_batches,
               num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed,
               resume=args.resume, ckpt_interval=args.ckpt_interval,
               jsonl_name=args.jsonl_name, keep_ckpts=args.keep_ckpts)


if __name__ == "__main__":
    raise SystemExit(main())
