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
import sys
import tempfile
from pathlib import Path

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


def save_checkpoint(ckpt_dir: Path, student, optimizer, scheduler, sched_name: str,
                    it: int, best_miou: float) -> str:
    path = _assert_outside_repo(Path(ckpt_dir)) / f"e1_student_best_iter{it}.pt"
    torch.save({
        "iter": it,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler": sched_name,
        "best_val_miou_all_class": best_miou,
        "num_classes": NUM_CLASSES,
    }, path)
    return str(path)


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
        grad_clip_norm: float | None, log_every: int, seed: int) -> int:
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
    train_loader = build_dataloader("train", batch_size, num_workers=num_workers)
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

    checks: dict[str, bool] = {}
    best_miou = float("-inf")
    best_ckpt = None
    first_batch_meta = None
    lr_trace: list[float] = []
    train_iter = cycle(train_loader)

    for it in range(1, max_iters + 1):
        img, mask = next(train_iter)
        img, mask = img.to(dev), mask.to(dev)
        if it == 1:
            first_batch_meta = (tuple(img.shape), str(img.dtype), tuple(mask.shape), str(mask.dtype))
            checks["batch_shapes"] = (img.shape[1:] == (3, 512, 512) and img.dtype == torch.float32
                                      and mask.shape[1:] == (512, 512) and mask.dtype == torch.int64)

        optimizer.zero_grad(set_to_none=True)
        logits = student(img)
        if it == 1:
            checks["logits_shape"] = tuple(logits.shape) == (img.shape[0], NUM_CLASSES, 512, 512)
        ce = criterion.ce(logits, mask)
        dice = criterion.dice(logits, mask)
        loss = ce + dice                              # == CombinedCEDiceLoss(logits, mask)
        if mode == "real" and not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"non-finite loss at iter {it}: loss={loss.item():.4f} ce={ce.item():.4f} "
                f"dice={dice.item():.4f}. Aborting the real E1 run to avoid poisoning the model "
                "or wasting compute.")

        if it == 1:
            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
            p0 = next(p for p in student.parameters() if p.requires_grad)
            before = p0.detach().clone()

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()                              # step ONCE per iteration, after optimizer
        lr = optimizer.param_groups[0]["lr"]
        lr_trace.append(lr)

        if it == 1:
            checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)

        if it % log_every == 0 or it == max_iters:
            print(f"[iter {it:>4}/{max_iters}] loss={loss.item():.4f} ce={ce.item():.4f} "
                  f"dice={dice.item():.4f} lr={lr:.8e}")

        if it % val_interval == 0 or it == max_iters:
            all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
                                                       max_val_batches)
            checks["val_cm_accumulated"] = (tuple(cm.shape) == (NUM_CLASSES, NUM_CLASSES)
                                            and int(cm.sum()) > 0 and nvb >= 1)
            print(f"[val  {it:>4}/{max_iters}] cm_batches={nvb} cm_total_px={int(cm.sum())} "
                  f"all_class_miou={all_miou:.5f} disease_only_miou(PROVISIONAL)={disease_miou:.5f}")
            if all_miou > best_miou:
                best_miou = all_miou
                best_ckpt = save_checkpoint(ckpt_dir, student, optimizer, scheduler, sched_name,
                                            it, best_miou)
                print(f"[ckpt {it:>4}/{max_iters}] new best all_class_miou={best_miou:.5f} "
                      f"-> {best_ckpt}")

    # scheduler sanity: per-iteration poly decay is monotonically non-increasing
    checks["lr_non_increasing"] = all(lr_trace[i + 1] <= lr_trace[i] + 1e-12
                                      for i in range(len(lr_trace) - 1))

    hard = ["batch_shapes", "logits_shape", "loss_finite", "optimizer_step",
            "val_cm_accumulated", "lr_non_increasing"]
    passed = all(checks.get(k, False) for k in hard)
    print("\n[CHECKS]")
    for k in hard:
        print(f"  {k:18}: {'PASS' if checks.get(k) else 'FAIL'}")
    print(f"[summary] first_batch={first_batch_meta}")
    print(f"[summary] lr_trace={['%.8e' % x for x in lr_trace]}")
    print(f"[summary] best_all_class_val_miou={best_miou:.5f} best_ckpt={best_ckpt}")
    print(f"[summary] used_pretrained={student.used_pretrained} (no download in dry-run)")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
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
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="global-norm clip; omitted by default (no concrete E1 value)")
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
        num_workers = args.num_workers if args.num_workers is not None else 4

    return run(mode=mode, device=device, pretrained=pretrained, batch_size=batch_size,
               max_iters=max_iters, val_interval=val_interval, max_val_batches=max_val_batches,
               num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
