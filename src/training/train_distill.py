#!/usr/bin/env python3
"""Shared distillation training core for E2 (Logit KD) and E3 (Logit KD + CWD).

Both stages are the SAME student recipe as E1 — MobileNetV3-Large + LR-ASPP, FP32, ImageNet init,
identical preprocessing/augmentation/splits, all-class validation mIoU for checkpoint selection —
with a frozen SegNeXt-B/MSCAN-B teacher added on top:

  E2  L = L_CE + L_Dice + lambda_logit * L_LogitKD
  E3  L = L_CE + L_Dice + lambda_logit * L_LogitKD + 50 * L_CWD_feat + 3 * L_CWD_logit

Both start from **E1 init (ImageNet), trained independently** (IMPLEMENTATION_CONTRACT.md stage
table): E2 does NOT resume from a trained E1 checkpoint and E3 does NOT resume from E2. They are
controlled, independently reproducible configurations that happen to share this implementation.
`train_e2.py` / `train_e3.py` are the identifiable per-stage entry points.

SAFETY MODEL (mirrors train_e1.py, plus two distillation-specific gates):
  * Default / `--dry-run` -> tiny CPU run, random student init, NO download, EXPLICIT MockTeacher,
    checkpoint to a temp dir.
  * Real run requires BOTH `--real-run` AND `--confirm-real-run`, requires CUDA, requires an existing
    `--teacher-ckpt`, requires an explicit `--lambda-logit` (the contract leaves lambda_logit as
    NEED_TO_CONFIRM, selected by validation sweep — it is never guessed here), and requires an
    explicit positive finite `--grad-clip-norm` (methodology mandates global-norm clipping but fixes
    no numeric threshold, so it stays a recorded experiment-level decision).
  * All gates return before any dataloader or teacher is constructed.
  * Checkpoints are NEVER written inside the repo, and the E3 training-only CWD projection is stored
    under a separate key so `model_state_dict` is already the clean E6/E7 deployment student.

No quantization path exists in this file: no QuantStub prepare/convert, no QAT, no PTQ.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from configs.data import DATA                                      # noqa: E402
from configs.distill import DISTILL                                # noqa: E402
from configs.e1_student import E1_STUDENT                          # noqa: E402
from src.data import NUM_CLASSES, build_dataloader                 # noqa: E402
from src.distill.cwd_projection import build_cwd_projection        # noqa: E402
from src.distill.export import CWD_PROJECTION_KEY, assert_clean_student_state  # noqa: E402
from src.distill.features import StudentTaps                       # noqa: E402
from src.distill.teacher import (FrozenTeacher, MockTeacher,       # noqa: E402
                                 TeacherCheckpointMissing, load_frozen_teacher,
                                 require_teacher_checkpoint)
from src.models.student import build_student                       # noqa: E402
from src.seeds import set_seed                                     # noqa: E402
from src.training.losses import (CombinedCEDiceLoss, cwd_channelwise_kl,  # noqa: E402
                                 downsample_validity, logit_kd_kl)
# Reuse the audited E1 mechanics verbatim rather than re-implementing them.
from src.training.train_e1 import (build_scheduler, cycle, load_ce_weights,  # noqa: E402
                                   resolve_ckpt_dir, validate, _assert_outside_repo)

IGNORE_INDEX = DATA["ignore_index"]            # 255
T_LOGIT = DISTILL["logit_kd"]["T_logit"]       # 4
T_CWD = DISTILL["cwd"]["T_cwd"]                # 4
ALPHA_CWD_FEAT = DISTILL["cwd"]["alpha_cwd_feature_map"]   # 50
BETA_CWD_LOGIT = DISTILL["cwd"]["beta_cwd_logit_map"]      # 3
CWD_C_FEAT = DISTILL["cwd"]["C"]               # 320 (MSCAN-B stride-16 Stage-3)
LAMBDA_SWEEP = DISTILL["logit_kd"]["lambda_logit_sweep_grid"]

STAGES: dict[str, dict] = {
    "e2": {"name": "E2", "logit_kd": True, "cwd": False,
           "objective": "L_CE + L_Dice + lambda_logit*L_LogitKD"},
    "e3": {"name": "E3", "logit_kd": True, "cwd": True,
           "objective": DISTILL["e3_total_loss"]},
}


def grad_clip_gate_error(value: float | None) -> str | None:
    """Validate `--grad-clip-norm` for a REAL E2/E3 launch. Returns an error string, or None if OK.

    Methodology requires global-norm gradient clipping THROUGHOUT distillation training
    (IMPLEMENTATION_CONTRACT B2), but no numeric `max_norm` is locked anywhere authoritative — D-A/D2
    records that Chapter 3 gives no value, and `configs/e1_student.py` keeps `grad_clip_max_norm=None`
    rather than inventing one. Rather than guessing a default, a real E2/E3 run REQUIRES the value to
    be supplied explicitly on the command line, so the threshold stays a recorded experiment-level
    decision. Dry-runs are unaffected. E1 is untouched by this gate.
    """
    if value is None:
        return ("--grad-clip-norm is required. The methodology mandates global-norm gradient "
                "clipping throughout distillation training, but the numeric max_norm is NOT fixed "
                "by any authoritative source (IMPLEMENTATION_CONTRACT D-A/D2; open_questions D2; "
                "configs/e1_student.py grad_clip_max_norm=None). It therefore remains an explicit "
                "experiment-level decision: re-run with --grad-clip-norm <positive finite value> "
                "and record the chosen threshold with the run.")
    if not math.isfinite(value) or value <= 0.0:
        return (f"--grad-clip-norm must be a positive finite value, got {value!r}. Zero, negative, "
                "NaN and Inf are rejected.")
    return None


def resolve_stage(stage: str) -> dict:
    """Map a stage key to its distillation composition. E2 = Logit KD only; E3 = Logit KD + CWD."""
    key = str(stage).lower()
    if key not in STAGES:
        raise ValueError(f"unknown distillation stage {stage!r}; expected one of {sorted(STAGES)}")
    return {"key": key, **STAGES[key]}


def save_distill_checkpoint(ckpt_dir: Path, stage: dict, student, projection, optimizer, scheduler,
                            sched_name: str, it: int, best_miou: float, teacher_provenance) -> str:
    """Write an E2/E3 checkpoint.

    `model_state_dict` holds the student ONLY — the training-only CWD projection goes under
    `cwd_projection_state_dict`, so the checkpoint E6/E7 consume is already projection-free
    (contract B3/B4). The split is asserted before writing.
    """
    path = _assert_outside_repo(Path(ckpt_dir)) / f"{stage['key']}_student_best_iter{it}.pt"
    student_state = student.state_dict()
    assert_clean_student_state(student_state)
    payload = {
        "stage": stage["name"],
        "iter": it,
        "model_state_dict": student_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler": sched_name,
        "best_val_miou_all_class": best_miou,
        "num_classes": NUM_CLASSES,
        "teacher_provenance": None if teacher_provenance is None else teacher_provenance.as_dict(),
        CWD_PROJECTION_KEY: None if projection is None else projection.state_dict(),
    }
    torch.save(payload, path)
    return str(path)


def distill_ramp(it: int, ramp_iters: int) -> float:
    """Linear distillation-weight ramp: EXACTLY 0 -> EXACTLY target over the FIRST EPOCH.

    IMPLEMENTATION_CONTRACT.md B2 ("Distillation-weight ramp (E2/E3) | linear 0 -> target over first
    epoch") and `configs/e1_student.py["distill_weight_ramp"]`. `ramp_iters` is one epoch expressed
    in iterations — taken from the actual train DataLoader length, never a hand-picked constant.

    The training loop is 1-indexed (`for it in range(1, max_iters + 1)`), so the first iteration is
    `it = 1` and the LAST iteration of the first epoch is `it = ramp_iters`. The factor is therefore

        ramp(it) = clamp((it - 1) / (ramp_iters - 1), 0, 1)

    giving ramp(1) = 0.0 exactly, ramp(ramp_iters) = 1.0 exactly, and 1.0 for every later iteration.
    A degenerate epoch of one iteration (or fewer) is that epoch's first AND last step, so the ramp
    is already complete and returns 1.0 rather than dividing by zero.

    Applied to EVERY distillation term (Logit KD and both CWD terms); the supervised CE+Dice term is
    never ramped.
    """
    if ramp_iters <= 1:
        return 1.0
    if it <= 1:
        return 0.0
    if it >= ramp_iters:
        return 1.0
    return float(it - 1) / float(ramp_iters - 1)


def distillation_losses(*, stage: dict, logits, head_logits, c5, mask, teacher_out, projection,
                        lambda_logit: float, ramp: float = 1.0):
    """Compute the stage's distillation terms. Returns (total_distill, parts dict).

    Teacher tensors arrive detached from `FrozenTeacher`, so nothing here can push gradient into the
    teacher; the student and (E3) the projection are the only things on the graph. `ramp` scales
    every distillation term per the first-epoch ramp; `parts` reports the UNRAMPED loss values so the
    log stays diagnostic.
    """
    parts: dict[str, float] = {}
    total = logits.new_zeros(())

    t_logits = teacher_out.logits
    if t_logits.shape[1] != logits.shape[1]:
        raise ValueError(f"teacher logits have {t_logits.shape[1]} classes, student has "
                         f"{logits.shape[1]}")

    # --- Logit KD (E2 and E3, unchanged between them per contract B3) ---
    t_logits_full = (t_logits if t_logits.shape[-2:] == logits.shape[-2:]
                     else F.interpolate(t_logits, size=logits.shape[-2:], mode="bilinear",
                                        align_corners=False))
    l_kd = logit_kd_kl(logits, t_logits_full, mask, T=T_LOGIT, ignore_index=IGNORE_INDEX)
    total = total + ramp * lambda_logit * l_kd
    parts["logit_kd"] = float(l_kd.detach())

    if not stage["cwd"]:
        return total, parts

    # --- CWD feature term: student C5 (160ch) -> 1x1 projection -> teacher stride-16 (320ch) ---
    if teacher_out.feat_s16 is None:
        raise RuntimeError("E3 requires the teacher stride-16 Stage-3 feature, but the teacher "
                           "returned none — check the teacher feature hook")
    s_feat = projection(c5)
    t_feat = teacher_out.feat_s16
    valid_s16 = downsample_validity(mask, s_feat.shape[-2:], ignore_index=IGNORE_INDEX)
    l_feat = cwd_channelwise_kl(s_feat, t_feat, valid_s16, T=T_CWD, channels_norm=CWD_C_FEAT)
    total = total + ramp * ALPHA_CWD_FEAT * l_feat
    parts["cwd_feat"] = float(l_feat.detach())

    # --- CWD logit term: on the head's native OS8 logit map (no projection needed, same C) ---
    t_logits_os8 = (t_logits if t_logits.shape[-2:] == head_logits.shape[-2:]
                    else F.interpolate(t_logits, size=head_logits.shape[-2:], mode="bilinear",
                                       align_corners=False))
    valid_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE_INDEX)
    # channels_norm defaults to this map's own channel count (116 classes) — 320 is the FEATURE-map
    # normalisation only and must never be hard-coded here.
    l_logit_map = cwd_channelwise_kl(head_logits, t_logits_os8, valid_os8, T=T_CWD)
    total = total + ramp * BETA_CWD_LOGIT * l_logit_map
    parts["cwd_logit"] = float(l_logit_map.detach())
    return total, parts


def run(*, stage: dict, mode: str, device: str, pretrained, teacher: FrozenTeacher,
        lambda_logit: float, batch_size: int, max_iters: int, val_interval: int,
        max_val_batches: int | None, num_workers: int, ckpt_dir_arg: str | None,
        grad_clip_norm: float | None, log_every: int, seed: int) -> int:
    set_seed(seed)
    dev = torch.device(device)
    print(f"[stage] {stage['name']} | objective: {stage['objective']}")
    print(f"[mode] {mode.upper()} | torch {torch.__version__} | device={dev} | "
          f"cuda_available={torch.cuda.is_available()} | num_classes={NUM_CLASSES}")
    print(f"[budget] max_iters={max_iters} batch_size={batch_size} val_interval={val_interval} "
          f"max_val_batches={max_val_batches} num_workers={num_workers} seed={seed}")

    # --- student (same architecture/init basis as E1) ---
    student = build_student(pretrained=pretrained).to(dev)
    student.train()
    if mode == "dry" and student.used_pretrained:
        raise RuntimeError("dry-run must NOT use pretrained weights (no download allowed)")
    print(f"[student] params={sum(p.numel() for p in student.parameters()):,} "
          f"used_pretrained={student.used_pretrained}")

    # --- frozen teacher ---
    teacher = teacher.to(dev)
    teacher.eval()
    if teacher.trainable_parameters():
        raise RuntimeError("teacher has trainable parameters — it must be frozen for distillation")
    is_mock = getattr(teacher.teacher, "is_mock", False)
    n_teacher_params = sum(p.numel() for p in teacher.teacher.parameters())
    print(f"[teacher] frozen=True eval=True params={n_teacher_params:,} trainable_params=0 "
          f"mock={is_mock} "
          f"provenance={None if teacher.provenance is None else teacher.provenance.as_dict()}")

    # --- training-only CWD projection (E3 only) ---
    projection = None
    if stage["cwd"]:
        projection = build_cwd_projection().to(dev)
        projection.train()
        print(f"[cwd] projection 1x1 conv {DISTILL['cwd']['projection_head']['maps']} "
              f"params={sum(p.numel() for p in projection.parameters()):,} "
              f"(TRAINING-ONLY; stored separately, absent from model_state_dict)")

    train_loader = build_dataloader("train", batch_size, num_workers=num_workers)
    val_loader = build_dataloader("val", batch_size, num_workers=num_workers)
    print(f"[data] train_index={len(train_loader.dataset)} val_index={len(val_loader.dataset)} "
          "(train+val only; the TEST split is never built here)")

    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=IGNORE_INDEX).to(dev)
    print(f"[loss] sup=CombinedCEDiceLoss | lambda_logit={lambda_logit} T_logit={T_LOGIT}"
          + (f" | alpha_cwd={ALPHA_CWD_FEAT} beta_cwd={BETA_CWD_LOGIT} T_cwd={T_CWD} "
             f"C_feat={CWD_C_FEAT}" if stage["cwd"] else ""))

    # Optimizer is built EXPLICITLY from the student (+ E3 projection) — never from the teacher.
    trainable = list(student.parameters()) + (list(projection.parameters()) if projection else [])
    overlap = {id(p) for p in trainable} & teacher.parameter_ids()
    if overlap:
        raise RuntimeError(f"{len(overlap)} teacher parameter(s) reached the optimizer parameter "
                           "list — the teacher must never be optimized")
    optimizer = torch.optim.SGD(trainable, lr=E1_STUDENT["learning_rate"],
                                momentum=E1_STUDENT["momentum"],
                                weight_decay=E1_STUDENT["weight_decay"])
    horizon = E1_STUDENT["iterations"]
    scheduler, sched_name = build_scheduler(optimizer, horizon, E1_STUDENT["lr_power"])
    print(f"[opt] SGD lr={E1_STUDENT['learning_rate']} scheduler={sched_name} "
          f"(total_iters={horizon}, power={E1_STUDENT['lr_power']}) trainable_tensors={len(trainable)} "
          f"teacher_params_in_optimizer=0")

    # First-epoch distillation ramp: one epoch expressed in iterations, from the real loader.
    ramp_iters = len(train_loader)
    print(f"[ramp] {E1_STUDENT['distill_weight_ramp']} -> ramp_iters={ramp_iters} "
          f"(= len(train_loader); factor = min(it/ramp_iters, 1.0), applied to every distill term)")
    if grad_clip_norm is None:
        clip_msg = ("DISABLED — no numeric max_norm is specified anywhere authoritative "
                    "(IMPLEMENTATION_CONTRACT D-A/D2, open_questions D2); pass "
                    "--grad-clip-norm <value> to enable the global-norm path")
    else:
        clip_msg = f"global-norm, max_norm={grad_clip_norm}, applied every iteration"
    print(f"[grad-clip] {clip_msg}")

    ckpt_dir = resolve_ckpt_dir(ckpt_dir_arg)
    print(f"[ckpt] dir={ckpt_dir} (verified OUTSIDE repo)")

    checks: dict[str, bool] = {}
    best_miou = float("-inf")
    best_ckpt = None
    lr_trace: list[float] = []
    train_iter = cycle(train_loader)

    for it in range(1, max_iters + 1):
        img, mask = next(train_iter)
        img, mask = img.to(dev), mask.to(dev)

        # ONE already-augmented tensor is shared by student and teacher — the teacher must see the
        # exact same augmented input as the student for every training sample.
        model_input = img

        optimizer.zero_grad(set_to_none=True)
        with StudentTaps(student) as taps:
            logits = student(model_input)
            c5, head_logits = taps.require()
        if it == 1:
            checks["logits_shape"] = tuple(logits.shape) == (img.shape[0], NUM_CLASSES, 512, 512)
            checks["c5_channels"] = c5.shape[1] == 160
            checks["teacher_params_frozen"] = not teacher.trainable_parameters()
            checks["optimizer_excludes_teacher"] = not ({id(p) for g in optimizer.param_groups
                                                         for p in g["params"]}
                                                        & teacher.parameter_ids())

        teacher_out = teacher(model_input, feat_size=c5.shape[-2:])
        if it == 1:
            checks["teacher_same_augmented_input"] = model_input is img

        ramp = distill_ramp(it, ramp_iters)
        sup = criterion(logits, mask)
        distill, parts = distillation_losses(
            stage=stage, logits=logits, head_logits=head_logits, c5=c5, mask=mask,
            teacher_out=teacher_out, projection=projection, lambda_logit=lambda_logit, ramp=ramp)
        loss = sup + distill
        if it == 1:
            checks["distill_ramp_starts_at_zero"] = (ramp == 0.0 if ramp_iters > 1
                                                     else ramp == 1.0)
            checks["supervised_never_ramped"] = bool(torch.equal(sup, criterion(logits, mask)))

        if mode == "real" and not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"non-finite loss at iter {it}: total={loss.item():.4f} sup={sup.item():.4f} "
                f"parts={parts}. Aborting the real {stage['name']} run.")
        if it == 1:
            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
            checks["has_expected_terms"] = (
                set(parts) == ({"logit_kd", "cwd_feat", "cwd_logit"} if stage["cwd"]
                               else {"logit_kd"}))
            p0 = next(p for p in student.parameters() if p.requires_grad)
            before = p0.detach().clone()

        loss.backward()
        if grad_clip_norm is not None:
            # global-norm clipping over the student (+ E3 projection), every iteration ("throughout")
            total_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            if it == 1:
                checks["grad_clip_applied"] = bool(torch.isfinite(torch.as_tensor(total_norm)))
        elif it == 1:
            checks["grad_clip_applied"] = True   # path present, intentionally disabled (D-A/D2)
        optimizer.step()
        scheduler.step()
        lr_trace.append(optimizer.param_groups[0]["lr"])

        if it == 1:
            checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)
            checks["teacher_stayed_frozen"] = all(
                p.grad is None for p in teacher.teacher.parameters())

        if it % log_every == 0 or it == max_iters:
            extra = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"[iter {it:>4}/{max_iters}] loss={loss.item():.4f} sup={sup.item():.4f} "
                  f"{extra} ramp={ramp:.4f} lr={lr_trace[-1]:.8e}")

        if it % val_interval == 0 or it == max_iters:
            all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
                                                       max_val_batches)
            checks["val_cm_accumulated"] = (tuple(cm.shape) == (NUM_CLASSES, NUM_CLASSES)
                                            and int(cm.sum()) > 0 and nvb >= 1)
            print(f"[val  {it:>4}/{max_iters}] cm_batches={nvb} all_class_miou={all_miou:.5f} "
                  f"disease_only_miou(PROVISIONAL)={disease_miou:.5f}")
            if all_miou > best_miou:                       # selection = ALL-CLASS val mIoU (D1)
                best_miou = all_miou
                best_ckpt = save_distill_checkpoint(ckpt_dir, stage, student, projection, optimizer,
                                                    scheduler, sched_name, it, best_miou,
                                                    teacher.provenance)
                print(f"[ckpt {it:>4}/{max_iters}] new best all_class_miou={best_miou:.5f} "
                      f"-> {best_ckpt}")

    checks["lr_non_increasing"] = all(lr_trace[i + 1] <= lr_trace[i] + 1e-12
                                      for i in range(len(lr_trace) - 1))

    hard = ["logits_shape", "c5_channels", "loss_finite", "has_expected_terms", "optimizer_step",
            "teacher_stayed_frozen", "teacher_params_frozen", "optimizer_excludes_teacher",
            "teacher_same_augmented_input", "distill_ramp_starts_at_zero", "supervised_never_ramped",
            "grad_clip_applied", "val_cm_accumulated", "lr_non_increasing"]
    passed = all(checks.get(k, False) for k in hard)
    print("\n[CHECKS]")
    for k in hard:
        print(f"  {k:22}: {'PASS' if checks.get(k) else 'FAIL'}")
    print(f"[summary] best_all_class_val_miou={best_miou:.5f} best_ckpt={best_ckpt}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def parse_args(argv=None, stage_default: str | None = None):
    p = argparse.ArgumentParser(
        description="Shared E2/E3 distillation training core (dry-run by default).")
    if stage_default is None:
        p.add_argument("--stage", choices=sorted(STAGES), required=True,
                       help="e2 = Logit KD only; e3 = Logit KD + CWD")
    p.add_argument("--dry-run", action="store_true", help="tiny CPU dry-run (safe default)")
    p.add_argument("--real-run", action="store_true", help="intent to run real training")
    p.add_argument("--confirm-real-run", action="store_true", help="explicit confirmation gate")
    p.add_argument("--teacher-ckpt", default=None,
                   help="path to the fine-tuned SegNeXt-B teacher checkpoint (required for a real run)")
    p.add_argument("--teacher-config", default=None, help="optional mmseg config path for the teacher")
    p.add_argument("--lambda-logit", type=float, default=None,
                   help=f"Logit-KD weight; contract leaves it NEED_TO_CONFIRM (sweep {LAMBDA_SWEEP})")
    p.add_argument("--device", default=None)
    p.add_argument("--init", choices=["none", "imagenet"], default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--val-interval", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--ckpt-dir", default=None, help="out-of-repo dir; auto temp dir if omitted")
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if stage_default is not None:
        args.stage = stage_default
    return args


def main(argv=None, stage_default: str | None = None) -> int:
    args = parse_args(argv, stage_default)
    stage = resolve_stage(args.stage)

    if args.dry_run and args.real_run:
        print("ERROR: pass only one of --dry-run / --real-run.", file=sys.stderr)
        return 2

    # ---- real-run safety gates (all BEFORE any data/teacher construction) ----
    if args.real_run:
        if not args.confirm_real_run:
            print(f"REFUSING to start the real {stage['name']} run: --real-run requires "
                  "--confirm-real-run.", file=sys.stderr)
            return 2
        mode = "real"
    elif args.confirm_real_run:
        print("ERROR: --confirm-real-run given without --real-run; nothing to confirm.",
              file=sys.stderr)
        return 2
    else:
        mode = "dry"
        if not args.dry_run:
            print(f"[mode] No --dry-run/--real-run given; defaulting to SAFE DRY-RUN. The real "
                  f"{stage['name']} run requires --real-run --confirm-real-run.")

    if mode == "real":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not str(device).startswith("cuda"):
            reason = ("--device cpu was passed" if args.device == "cpu"
                      else "CUDA is not available (torch.cuda.is_available()=False)")
            print(f"REFUSING to start the real {stage['name']} run on CPU: {reason}.",
                  file=sys.stderr)
            return 2
        try:
            require_teacher_checkpoint(args.teacher_ckpt)
        except TeacherCheckpointMissing as e:
            print(f"REFUSING to start the real {stage['name']} run: {e}", file=sys.stderr)
            return 2
        if args.lambda_logit is None:
            print(f"REFUSING to start the real {stage['name']} run: --lambda-logit is required. "
                  f"The contract leaves lambda_logit as NEED_TO_CONFIRM (validation sweep over "
                  f"{LAMBDA_SWEEP} at seed 42); it is never guessed.", file=sys.stderr)
            return 2
        clip_error = grad_clip_gate_error(args.grad_clip_norm)
        if clip_error is not None:
            print(f"REFUSING to start the real {stage['name']} run: {clip_error}", file=sys.stderr)
            return 2
        teacher = load_frozen_teacher(args.teacher_ckpt, config_path=args.teacher_config)
        init = args.init or "imagenet"
        pretrained = False if init == "none" else E1_STUDENT["init_weights"]
        batch_size = args.batch_size or E1_STUDENT["batch_size"]
        max_iters = args.max_iters or E1_STUDENT["iterations"]
        val_interval = args.val_interval or E1_STUDENT["val_interval"]
        max_val_batches = args.max_val_batches
        num_workers = args.num_workers if args.num_workers is not None else 4
        lambda_logit = args.lambda_logit
    else:
        device = args.device or "cpu"
        if args.init == "imagenet":
            print("[init] --init imagenet ignored in dry-run (forcing random init, no download).")
        pretrained = False
        if args.teacher_ckpt:
            teacher = load_frozen_teacher(args.teacher_ckpt, config_path=args.teacher_config)
        else:
            print("[teacher] DRY-RUN uses an EXPLICIT MockTeacher (random, weight-free). This is a "
                  "smoke substitute and is refused by the real-run path, which requires "
                  "--teacher-ckpt.")
            teacher = FrozenTeacher(MockTeacher(NUM_CLASSES))
        batch_size = args.batch_size or 2
        max_iters = args.max_iters or 4
        val_interval = args.val_interval or 2
        max_val_batches = args.max_val_batches if args.max_val_batches is not None else 2
        num_workers = args.num_workers if args.num_workers is not None else 0
        lambda_logit = args.lambda_logit if args.lambda_logit is not None else 1.0

    return run(stage=stage, mode=mode, device=device, pretrained=pretrained, teacher=teacher,
               lambda_logit=lambda_logit, batch_size=batch_size, max_iters=max_iters,
               val_interval=val_interval, max_val_batches=max_val_batches,
               num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
