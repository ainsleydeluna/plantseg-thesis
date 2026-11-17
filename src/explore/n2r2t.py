"""N2R2T (addendum L-AM24-ADD-3; AM-29a's E6-R2-MT, run ID x-n2r2t-s42): this lane's N2R2 exactly, with ONE term
added to the QAT loss: AM-29's T1 term (AM-29 item 3(a)), as lane/l-am29 builds it in src/quant/am29.py at 2a94019f
(SelfTeacher, t_logit_kd, run_am29_qat).

N2R2 (src/explore/qat_explore.py run_n2r2; AM-29's E6-R2-M) runs the pinned src/quant/qat.py run_qat on stage e6-r2
from E3-R2 seed 42's checkpoint of record (sha256 da70d82f..., DL-136) with N2's change: the ten sites of
src/explore/sites.py fake-quantized on their input's grid, and TF32 off for cuDNN and matmul. N2R2T keeps all of that
and adds one term to the QAT loss of record (CE + Dice):

  factor 0.5 x E3's lambda_logit 0.5 = weight 0.25, times losses.logit_kd_kl(student head logits, teacher head logits)
  at T = 4 (train_distill.T_LOGIT; the function's own T^2, none added), over losses.downsample_validity(target, that
  grid): train_distill.distillation_losses on stage e2's switches with ramp 1.0 (t_logit_kd). The student's logits are
  the prepared student's head_logits tap (StudentTaps), a fake-quant tap. The teacher is a frozen FP32 copy of the same
  parent checkpoint (SelfTeacher: eval mode, no gradient, outside the optimizer), fed the student's input batch;
  building it draws nothing from the global random stream. The weight is constant from the first step (no ramp). The
  rate is 3e-4, the QAT rate of record, with the cosine of record from it.

Copied from src/quant/am29.py at 2a94019f4dc11b8c0b543107e026ecb5ecc108c5 (file sha256 3fbf35b3...; COPIED below
records each copy's path, commit, line range and the sha256 of its source text, i.e. of the UTF-8 bytes of those
lines; run_meta's x_term block carries COPIED). Bodies unchanged:
  E3_LAMBDA_OF_RECORD             am29.py:71
  T_LOGIT, E2_SWITCHES, T_RAMP    am29.py:73-75
  _refuse                         am29.py:134-135
  _is_float                       am29.py:138-139
  SelfTeacher                     am29.py:187-219
  t_logit_kd                      am29.py:222-227
Copied with changes: run_n2r2t_qat is run_am29_qat's loop (am29.py:231-634; src/quant/qat.py run_qat line for line
with the term added). Its changes, each marked "N2R2T:" at its line:
  1. the stage: this lane's e6-r2 (qat_explore.N2R2_STAGE through src/quant/stages.py), as N2R2 binds it, with
     run_qat's method check, in place of an AM-29 arm's stage;
  2. the rate: src/quant/qat.py LEARNING_RATE (3e-4) in the optimizer and so in the cosine, checked by run_qat's
     optimizer_of_record_error (am29's optimizer_error is that rule with the arm's rate; at 3e-4 they are equal);
  3. the family: the T branch only (the teacher and the term are always on; am29's LR branches are dropped), and in a
     real run the item's values (seed 42, factor 0.5; values_error) in place of the arm allow-list, with am29's
     E3-lambda refusal kept;
  4. prepare_qat_model is called by its src/quant/qat.py name, so that N2's substitution, which run_n2r2t binds as
     run_n2r2 binds it, reaches the call;
  5. run_meta: E6-R2-M's fields (run_qat's schema, lane, scheduler record and code identity), this item's objective,
     "distillation" true and one x_term block (the item, the run ID and the term's fields) in place of am29's am29
     block and its four teacher keys; the log prefix and the refusal texts name N2R2T.
`run_n2r2t` is run_n2r2 with run_n2r2t_qat in place of run_qat: TF32 off (sites.set_tf32), then prepare_qat_model of
src/quant/qat.py bound, for the call, to the wrapper that installs the ten site fake-quants, recorded in run_meta
(x_substitution) and restored after.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from configs.distill import LOGIT_KD_SEMANTICS
from src.distill import teacher as TT
from src.distill.features import StudentTaps
from src.quant import qat as Q
from src.quant.prepare import disable_observers, freeze_bn_stats, qat_freeze_steps, unfused_batchnorm
from src.quant.qat import QATRefused, QATStop
from src.quant.qconfig import QuantBackendUnavailable, select_qnnpack_backend
from src.quant.stages import resolve_quant_stage
from src.training import train_distill as TD

from . import qat_explore as X

ITEM = "N2R2T"
RUN_ID = X.N2R2T_RUN_ID                     # x-n2r2t-s42
SEED = 42                                   # E6-R2-M's seed (qat_explore.ITEM_SEEDS["N2R2T"])
FACTOR = 0.5                                # AM-29 item 3(a): T1's factor; weight 0.5 x E3's lambda 0.5 = 0.25

AM29_PATH = "src/quant/am29.py"
AM29_COMMIT = "2a94019f4dc11b8c0b543107e026ecb5ecc108c5"
AM29_FILE_SHA256 = "3fbf35b3011aaaab74f0d97c85cbb03e9f3f0151bddd38b9f1767507dd8ddf39"
AM29_FILE_BLOB = "f2abe98a57c3ca74229463519100937a58b009c3"   # git rev-parse 2a94019f:src/quant/am29.py


def _copied(name: str, first: int, last: int, sha256: str, **more) -> dict:
    return {"name": name, "path": AM29_PATH, "commit": AM29_COMMIT, "lines": [first, last], "sha256": sha256, **more}


# The copies: the sha256 is that of the UTF-8 bytes of the lines first..last of src/quant/am29.py at AM29_COMMIT, each
# line with its newline. scripts/smoke_explore_n2r2t.py recomputes each from the file and from this module's source.
COPIED = {
    "E3_LAMBDA_OF_RECORD": _copied("E3_LAMBDA_OF_RECORD", 71, 71,
                                   "03735a9eedc9b149199adbad060917876d1c772dc402285e7c5e5b175451a4c0", body="unchanged"),
    "T_LOGIT, E2_SWITCHES, T_RAMP": _copied("T_LOGIT, E2_SWITCHES, T_RAMP", 73, 75,
                                            "bdcd4275f4df5000aaadf06f0e8f734161f13f007e0fdf407c55f54a79da24ab",
                                            body="unchanged"),
    "_refuse": _copied("_refuse", 134, 135, "ac47f0784d60bf8d797c99ca126d6b5d0a0256c97b8c5120b99b6b189ba37f6f",
                       body="unchanged"),
    "_is_float": _copied("_is_float", 138, 139, "d9d04f5cfb1669e2d7348baaf9a63898fbb0c1410b2ce8de6dc13bc24a1de902",
                         body="unchanged"),
    "SelfTeacher": _copied("SelfTeacher", 187, 219,
                           "c809f7f8a8916360e7fbbe33bb3c41e2da57cd2b3d70f596a2e6731d26e0729e", body="unchanged"),
    "t_logit_kd": _copied("t_logit_kd", 222, 227,
                          "30f4276ead4d86c637393e365f1b76baf1a16963b44498757e05976861fc133e", body="unchanged"),
    "run_am29_qat": _copied("run_am29_qat", 231, 634,
                            "aacc513833fa59322413f597be8cfd9c1f540fa8e8d0f1821a8fb56f5507ccd5",
                            body="changed: copied as run_n2r2t_qat with the five changes of this module's docstring"),
}

# ------------------------------------------------------------------ copied from src/quant/am29.py:71 at 2a94019f
E3_LAMBDA_OF_RECORD = 0.5                            # DL-121: e3_s42_lambda0.5_alpha25_80k_a1 (the lambda selection's)
# ------------------------------------------------------------------ copied from src/quant/am29.py:73-75 at 2a94019f
T_LOGIT = TD.T_LOGIT                                 # E3's constant (train_distill.py:142); never a new value
E2_SWITCHES = TD.resolve_stage("e2")                 # logit KD only: E3's logit-KD term, alone (no CWD, no projection)
T_RAMP = None                                        # the T term's weight is constant: the spec names no ramp
# ------------------------------------------------------------------ end of the copied constants

# The files the term's code comes from, hashed into run_meta's x_term (the QAT code paths are run_qat's own record).
TERM_CODE_PATHS = ("src/explore/n2r2t.py", "src/training/train_distill.py", "src/training/losses.py",
                   "src/distill/features.py", "src/distill/teacher.py", "configs/distill.py")
OBJECTIVE = ("supervised CE (E1 class weights) + Dice + factor x E3's lambda_logit x E3's logit-KD term (KL at T = 4, "
             "T^2 in the term, on the head's OS8 logits over the downsampled validity mask), the teacher a frozen FP32 "
             "copy of the parent checkpoint; no CWD term, no projection, no ramp; N2R2T: factor 0.5, lambda 0.5, weight "
             "0.25 -- lane L-AM24 N2R2T (AM-29's T1 term on E6-R2-M)")
TERM_TEXT = ("AM-29's T1 term (AM-29 item 3(a); src/quant/am29.py t_logit_kd at 2a94019f): factor x E3's lambda_logit "
             "x losses.logit_kd_kl(student head logits, teacher head logits, downsample_validity(target, that grid), "
             "T=4), through train_distill.distillation_losses on stage e2's switches, ramp 1.0")
MATCHED_RUNS = ("N2R2 (x-n2r2-s42; AM-29's E6-R2-M): the same parent, parent loader, recipe, clip, seed and N2's "
                "change, the term the one difference; AM-29's T1 (lane/l-am29 run_am29_qat, arm T1): the same term, "
                "parent, seed and rate, without N2's change")


# ------------------------------------------------------------------ copied from src/quant/am29.py:134-135 at 2a94019f
def _refuse(code: str, message: str):
    raise QATRefused(code, message)


# ------------------------------------------------------------------ copied from src/quant/am29.py:138-139 at 2a94019f
def _is_float(v) -> bool:
    return isinstance(v, float) and not isinstance(v, bool) and math.isfinite(v)


# ------------------------------------------------------------------ copied from src/quant/am29.py:187-219 at 2a94019f
class SelfTeacher(nn.Module):
    """T1-T4's teacher: a frozen FP32 copy of the parent checkpoint's student, in eval mode, no grad, outside the
    optimizer. Its output is the head's OS8 logits (StudentTaps' head_logits), the grid E3's logit-KD term is computed
    on; it has no stride-16 feature (no CWD term)."""

    def __init__(self, student: nn.Module, *, checkpoint_sha256: str | None = None):
        super().__init__()
        self.student = copy.deepcopy(student)
        self.checkpoint_sha256 = checkpoint_sha256
        self.student.eval()
        for p in self.student.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "SelfTeacher":
        super().train(mode)
        self.student.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.student.parameters() if p.requires_grad]

    def parameter_ids(self) -> set[int]:
        return {id(p) for p in self.student.parameters()}

    def state_sha256(self) -> str:
        return Q.state_digest(self.student)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> TT.TeacherOutput:
        with StudentTaps(self.student) as taps:
            self.student(x)
            _c5, head_logits = taps.require()
        return TT.TeacherOutput(logits=head_logits.detach(), feat_s16=None)


# ------------------------------------------------------------------ copied from src/quant/am29.py:222-227 at 2a94019f
def t_logit_kd(*, logits, head_logits, c5, mask, teacher_out, weight: float):
    """E3's own logit-KD term at `weight` (factor x E3's lambda_logit): train_distill.distillation_losses on stage e2's
    switches, i.e. weight x losses.logit_kd_kl(head_logits, teacher OS8 logits, downsample_validity(mask), T=T_LOGIT).
    Returns (weight x term, {"logit_kd": the unweighted term})."""
    return TD.distillation_losses(stage=E2_SWITCHES, logits=logits, head_logits=head_logits, c5=c5, mask=mask,
                                  teacher_out=teacher_out, projection=None, lambda_logit=weight, ramp=1.0)


# ------------------------------------------------------------------ end of the unchanged copies
def values_error(*, seed, factor) -> tuple[str, str] | None:
    """A real run's values (in place of am29's arm allow-list): E6-R2-M's seed 42 and T1's factor 0.5. Returns (code,
    text) or None."""
    if isinstance(seed, bool) or seed != SEED:
        return "n2r2t_seed", f"N2R2T runs seed {SEED} (E6-R2-M's), not {seed!r}"
    if not _is_float(factor) or factor != FACTOR:
        return "n2r2t_factor", (f"N2R2T's factor is {FACTOR} (AM-29's T1: 0.5 x E3's lambda_logit 0.5 = weight 0.25), "
                                f"not {factor!r}")
    return None


def teacher_record(teacher: SelfTeacher) -> dict:
    """The teacher as x_run_meta.json and run_meta record it."""
    return {"kind": "frozen FP32 copy of the parent checkpoint (SelfTeacher; src/quant/am29.py:187-219 at 2a94019f)",
            "checkpoint_sha256": teacher.checkpoint_sha256, "state_sha256": teacher.state_sha256(),
            "eval_mode": not teacher.student.training, "requires_grad": bool(teacher.trainable_parameters()),
            "in_optimizer": False, "input": "the student's input batch, after the student's forward"}


def term_record(*, run_id: str, factor: float, e3_lambda: float, kd_weight: float, teacher_state: str,
                teacher: SelfTeacher, lr: float) -> dict:
    """run_meta's x_term: the item, the run ID and the term's fields."""
    return {"item": ITEM, "run_id": run_id, "term": TERM_TEXT, "factor": factor, "e3_lambda_logit": e3_lambda,
            "kd_weight": kd_weight, "T_logit": T_LOGIT, "t_squared": "the function's own (losses.py:125); none added",
            "ramp": T_RAMP, "weight_rule": "constant from the first step (no ramp)",
            "terms": {t: bool(E2_SWITCHES[t]) for t in TD.TERMS}, "logit_kd_semantics": LOGIT_KD_SEMANTICS,
            "student_logits": "the prepared student's head_logits tap (StudentTaps: the head's output), a fake-quant tap",
            "teacher_logits": "the same tap of the frozen FP32 copy, on the same grid",
            "mask": "losses.downsample_validity(target, the head logits' grid)",
            "teacher": {"kind": "frozen FP32 copy of the parent checkpoint (SelfTeacher)",
                        "checkpoint_sha256": teacher.checkpoint_sha256, "state_sha256": teacher_state,
                        "eval_mode": True, "requires_grad": False, "in_optimizer": False,
                        "input": "the student's input batch"},
            "lr": lr, "schedule": "the cosine of record from lr (CosineAnnealingLR, T_max = 15 epochs, eta_min 0)",
            "matched_runs": MATCHED_RUNS, "copied": COPIED,
            "code_files_sha256": {p: Q.sha256_file(Q.REPO / p) for p in TERM_CODE_PATHS if (Q.REPO / p).is_file()}}


# ------------------------------------------------------------------ run_am29_qat's loop (am29.py:231-634), changed
def run_n2r2t_qat(*, mode: str, model: nn.Module, source_meta: dict, out_dir, seed: int, clip_norm: float,
                  clip_source: str, device: str, num_workers: int, factor: float | None = None,
                  e3_lambda: float | None = None, teacher: SelfTeacher | None = None, loaders=None,
                  parent: dict | None = None, extra_meta: dict | None = None, hooks: Q.QATHooks | None = None,
                  inject_nan_at_steps=(), max_steps: int | None = None, max_val_batches: int | None = None,
                  log=print) -> dict:
    """N2R2T's run: run_am29_qat's T loop (src/quant/qat.py run_qat line for line, with the teacher forward and the one
    term) on this lane's stage e6-r2 at the QAT rate of record. Returns a summary; raises QATRefused, QATStop.

    `mode` "real" is the launch (scripts/explore_qat.py has passed every gate): seed 42, factor 0.5 and E3's lambda 0.5
    only. `mode` "smoke" accepts injected loaders, hooks, NaN injection, a step cap, a VAL cap and any finite factor (a
    factor 0 run has N2R2's loss)."""
    from src.seeds import set_seed
    from src.training.losses import CombinedCEDiceLoss
    from src.training.train_e1 import CLASS_WEIGHTS_JSON, _image_digest, load_ce_weights, validate

    st = resolve_quant_stage(X.N2R2_STAGE)                  # N2R2T: this lane's stage e6-r2, as N2R2 binds it
    if st["method"] != "qat":                               # N2R2T: run_qat's check, in place of the arm's family
        _refuse("stage_not_qat", f"{st['name']} is not a QAT stage")
    lr = Q.LEARNING_RATE                                    # N2R2T: the QAT rate of record, 3e-4 (am29: the arm's)
    if mode not in Q.MODES:
        _refuse("mode_invalid", f"mode {mode!r} not in {Q.MODES}")
    if mode == "real" and (loaders is not None or hooks is not None or inject_nan_at_steps or max_steps
                           or max_val_batches):
        _refuse("real_run_test_hooks", "a real run takes no injected loaders, hooks, NaN injection, step cap "
                                       "or VAL cap")
    if mode == "real" and clip_source != "clip_selection":
        _refuse("clip_source", f"an N2R2T run's clip is the U4 winner bound at launch (DL-139), not {clip_source!r}")
    pins = Q.config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    if isinstance(clip_norm, bool) or not isinstance(clip_norm, float) or not math.isfinite(clip_norm) \
            or clip_norm <= 0:
        _refuse("grad_clip_norm_invalid", f"clip norm must be a positive finite float, got {clip_norm!r}")
    if not _is_float(lr) or lr <= 0:
        _refuse("lr_invalid", f"lr must be a positive finite float, got {lr!r}")
    if not isinstance(teacher, SelfTeacher) or teacher.trainable_parameters():    # N2R2T: always the T branch
        _refuse("teacher_not_frozen", "N2R2T's teacher is a SelfTeacher with no trainable parameter")
    if not _is_float(factor) or not _is_float(e3_lambda):
        _refuse("weight_invalid", f"factor {factor!r} and e3_lambda {e3_lambda!r} must be finite floats")
    kd_weight = factor * e3_lambda
    if mode == "real":
        err = values_error(seed=seed, factor=factor)        # N2R2T: the item's values, in place of the arm's
        if err:
            _refuse(*err)
        if e3_lambda != E3_LAMBDA_OF_RECORD:
            _refuse("e3_lambda", f"E3's lambda_logit is {E3_LAMBDA_OF_RECORD} (DL-121), not {e3_lambda!r}")
    dev = torch.device(device)
    if mode == "real" and dev.type != "cuda":
        _refuse("cuda_required", f"a real QAT run needs a CUDA device, got {device!r}")
    cuda_initialized_at_seed = bool(torch.cuda.is_initialized())
    if mode == "real" and cuda_initialized_at_seed:
        _refuse("cuda_initialized_before_seed", "CUDA was initialised before set_seed; launch in a fresh process")
    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _refuse("out_dir_not_empty", f"{out} exists and is not an empty directory; a run starts in a fresh "
                                     "directory (the telemetry is append-only)")
    run_id = out.name

    set_seed(seed)                                           # AM-4a item 4: before CUDA initialises
    if loaders is None:
        train_loader, val_loader = Q.build_qat_loaders(seed, num_workers)
        loader_source = "E1 build_dataloader (train_e1.py:320-322), seed=--seed"
    else:
        train_loader, val_loader = loaders
        loader_source = "injected (smoke)"
    steps_per_epoch = len(train_loader)
    t_max = Q.EPOCHS * steps_per_epoch
    loader_batch = getattr(train_loader, "batch_size", None)
    loader_drop_last = getattr(train_loader, "drop_last", None)
    if mode == "real" and (steps_per_epoch != Q.REAL_STEPS_PER_EPOCH or t_max != Q.REAL_T_MAX):
        _refuse("steps_per_epoch", f"steps_per_epoch {steps_per_epoch} / T_max {t_max}; the recipe of record is "
                                   f"{Q.REAL_STEPS_PER_EPOCH} / {Q.REAL_T_MAX} (5,367 TRAIN images, batch 16, drop_last)")
    if mode == "real" and (loader_batch != Q.BATCH_SIZE or loader_drop_last is not True):
        _refuse("loader_of_record", f"the TRAIN loader has batch_size {loader_batch!r}, drop_last "
                                    f"{loader_drop_last!r}; the recipe of record is {Q.BATCH_SIZE}, True")
    if steps_per_epoch < 1:
        _refuse("empty_loader", "the TRAIN loader yields no batch")
    bn_after_step, obs_after_step = qat_freeze_steps(steps_per_epoch, Q.BN_FREEZE_EPOCH, Q.OBS_FREEZE_EPOCH)

    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        _refuse("backend_unavailable", str(e))
    prepared = Q.prepare_qat_model(model, select_backend=False)     # N2R2T: qat.py's name (N2's substitution binds it)
    unfused = unfused_batchnorm(prepared)
    if unfused and mode == "real":
        _refuse("unfused_batchnorm", f"{len(unfused)} BatchNorm module(s) sit outside a fused QAT ConvBn: "
                                     f"{unfused[:5]}; the epoch-11 freeze would miss them")
    census = Q.observer_census(prepared)
    fq_count = len(Q.fake_quant_modules(prepared))
    never = Q.never_observed_modules(prepared)
    kinds = Q.state_kinds(prepared)
    convbn = Q.fused_convbn_census(prepared)
    prepared = prepared.to(dev)
    prepared.train()
    teacher = teacher.to(dev)                               # N2R2T: always the T branch
    teacher.eval()
    teacher_state = teacher.state_sha256()

    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=Q.IGNORE_INDEX).to(dev)
    optimizer = torch.optim.SGD(prepared.parameters(), lr=lr, momentum=Q.MOMENTUM,
                                weight_decay=Q.WEIGHT_DECAY, dampening=Q.DAMPENING, nesterov=Q.NESTEROV)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=Q.ETA_MIN)
    opt_rec = Q.optimizer_record(optimizer, prepared)
    opt_err = Q.optimizer_of_record_error(opt_rec)          # N2R2T: run_qat's rule (am29's optimizer_error at 3e-4)
    if opt_err:
        raise QATStop("optimizer_of_record", opt_err)
    if teacher.parameter_ids() & {id(p) for g in optimizer.param_groups for p in g["params"]}:
        raise QATStop("teacher_in_optimizer", "a teacher parameter reached the optimizer")

    from src.quant.ptq import qconfig_summary
    from src.quant.qconfig import qat_qconfig
    qsum = qconfig_summary(qat_qconfig())
    identity = Q.code_identity()                            # N2R2T: E6-R2-M's code identity (QAT_CODE_PATHS)
    import numpy
    import torchvision
    tel_path = out / Q.TELEMETRY_NAME
    run_meta = {                                            # N2R2T: E6-R2-M's fields (run_qat's), then x_term
        "event": "run_meta", "wall_clock": time.time(), "schema": Q.TELEMETRY_SCHEMA, "lane": Q.LANE,
        "mode": mode, "stage": st["name"], "source_stage": st["source_stage"], "seed": seed,
        "run_id": run_id, "out_dir": str(out.resolve()),
        "parent": parent, "source_checkpoint_sha256": source_meta["sha256"],
        "source_checkpoint": source_meta.get("path"), "source_checkpoint_bytes": source_meta.get("bytes"),
        "clip_norm": clip_norm, "clip_source": clip_source,
        "epochs": Q.EPOCHS, "batch_size": loader_batch, "drop_last": loader_drop_last,
        "gradient_accumulation": None, "ema": False, "distillation": True,
        "steps_per_epoch": steps_per_epoch, "total_steps": t_max,
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "loader": loader_source, "num_workers": num_workers, "val_batches_cap": max_val_batches,
        "persistent_workers": bool(getattr(train_loader, "persistent_workers", False)),
        "scheduler": {"class": type(scheduler).__name__, "T_max": scheduler.T_max,
                      "eta_min": scheduler.eta_min, "step": "once after every optimizer.step"},
        "optimizer": opt_rec, "lr": opt_rec["lr"], "momentum": opt_rec["momentum"],
        "weight_decay": opt_rec["weight_decay"], "nesterov": opt_rec["nesterov"],
        "dampening": opt_rec["dampening"],
        "bn_freeze_epoch": Q.BN_FREEZE_EPOCH, "obs_freeze_epoch": Q.OBS_FREEZE_EPOCH,
        "bn_frozen_from_epoch": Q.BN_FREEZE_EPOCH + 1, "obs_frozen_from_epoch": Q.OBS_FREEZE_EPOCH + 1,
        "bn_freeze_after_step": bn_after_step, "obs_freeze_after_step": obs_after_step,
        "objective": OBJECTIVE,
        "class_weights_sha256": Q.sha256_file(CLASS_WEIGHTS_JSON),
        "ignore_index": Q.IGNORE_INDEX,
        "engine": torch.backends.quantized.engine, "qconfig": qsum,
        "observers": {"activation": qsum["activation"]["observer_class"],
                      "weight": qsum["weight"]["observer_class"]},
        "observer_census": census, "fake_quant_modules": fq_count, "never_observed_modules": never,
        "observer_freeze_scope": Q.OBSERVER_FREEZE_SCOPE,
        "fused_convbn": convbn, "unfused_batchnorm": len(unfused),
        "tf32": Q.tf32_state(), "determinism": Q.determinism_state(),
        "cuda_initialized_at_seed": cuda_initialized_at_seed,
        "device": str(dev), "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "python": sys.version.split()[0], "image_digest": _image_digest(), "host": Q.host_identity(),
        "git_head": identity["git_head"], "code_clean_at_head": identity["code_clean_at_head"],
        "code_files_sha256": identity["code_files_sha256"],
        "x_term": term_record(run_id=run_id, factor=factor, e3_lambda=e3_lambda, kd_weight=kd_weight,
                              teacher_state=teacher_state, teacher=teacher, lr=lr),
    }
    if st["source_stage"] == "E3":
        run_meta["cwd_projection_loaded"] = False
    if extra_meta:
        run_meta.update(extra_meta)
    out.mkdir(parents=True, exist_ok=True)
    tel = Q.Telemetry(tel_path)
    tel.write(run_meta)
    if hooks is not None:
        hooks.on_start(prepared, optimizer, scheduler, run_meta)
    log(f"[n2r2t] {st['name']} seed {seed} clip {clip_norm} lr {lr} kd {kd_weight} | {steps_per_epoch} steps/epoch, "
        f"T_max {t_max} | BN frozen after step {bn_after_step}, observers after {obs_after_step} | {run_id}")

    # nonfinite_since (the telemetry's nonfinite_since_step): the step at which the state was first found non-finite
    step = 0
    nonfinite_since = None
    nan_steps = {int(s) for s in inject_nan_at_steps}
    epoch = 0

    def _mark_nonfinite(at_step: int, fails: list[str]) -> None:
        nonlocal nonfinite_since
        nonfinite_since = at_step
        tel.write({"event": "state_nonfinite", "wall_clock": time.time(), "epoch": epoch, "step": at_step,
                   "nonfinite_since_step": at_step, "failing": fails})

    try:
        for epoch in range(1, Q.EPOCHS + 1):
            if epoch == Q.BN_FREEZE_EPOCH + 1:
                freeze_bn_stats(prepared)
            if epoch == Q.OBS_FREEZE_EPOCH + 1:
                disable_observers(prepared)
            ep = {"nonfinite_loss": 0, "nonfinite_grad": 0, "step_errors": 0, "clipped": 0,
                  "max_pre_clip_norm": None, "steps": 0}
            for img, mask in train_loader:
                if max_steps is not None and step >= max_steps:
                    break
                step += 1
                ep["steps"] += 1
                t0 = time.time()
                fp = Q.batch_sha256(mask)
                lr_now = optimizer.param_groups[0]["lr"]
                bn_frozen = epoch > Q.BN_FREEZE_EPOCH
                obs_on = epoch <= Q.OBS_FREEZE_EPOCH
                parts: dict = {}
                sup = None
                try:
                    img_d, mask_d = img.to(dev), mask.to(dev)
                    optimizer.zero_grad(set_to_none=True)
                    seen: dict = {}                         # N2R2T: always the T branch
                    probes = ([prepared.register_forward_pre_hook(TD.input_probe(seen, "student")),
                               teacher.student.register_forward_pre_hook(TD.input_probe(seen, "teacher"))]
                              if step == 1 else [])
                    try:
                        with StudentTaps(prepared) as taps:
                            logits = prepared(img_d)
                            c5, head_logits = taps.require()
                        teacher_out = teacher(img_d)     # one frozen forward per step, after the student's
                    finally:
                        for h in probes:
                            h.remove()
                    ce = criterion.ce(logits, mask_d)
                    dice = criterion.dice(logits, mask_d)
                    sup = ce + dice
                    distill, parts = t_logit_kd(logits=logits, head_logits=head_logits, c5=c5, mask=mask_d,
                                                teacher_out=teacher_out, weight=kd_weight)
                    loss = sup + distill
                    if step == 1:
                        h, w = img_d.shape[-2:]
                        checks = {"terms_logit_kd_only": set(parts) == {"logit_kd"},
                                  "head_logits_os8": tuple(head_logits.shape[-2:]) == (h // 8, w // 8),
                                  "teacher_logits_on_head_grid": tuple(teacher_out.logits.shape)
                                  == tuple(head_logits.shape),
                                  "teacher_no_feature": teacher_out.feat_s16 is None,
                                  "teacher_frozen": not teacher.trainable_parameters()
                                  and not teacher.student.training,
                                  "teacher_outside_optimizer": not (teacher.parameter_ids() & {
                                      id(p) for g in optimizer.param_groups for p in g["params"]}),
                                  "teacher_same_input": (seen.get("student") == seen.get("teacher")
                                                         == (img_d.data_ptr(), tuple(img_d.shape),
                                                             tuple(img_d.stride()))),
                                  "T_is_E3s": T_LOGIT == TD.T_LOGIT == 4}
                        tel.write({"event": "step1_checks", "wall_clock": time.time(), "epoch": epoch,
                                   "step": step, "checks": checks, "passed": all(checks.values())})
                        if mode == "real" and not all(checks.values()):
                            raise QATStop("step1_checks", f"N2R2T step-1 checks failed: {checks}")
                    if step in nan_steps:            # smoke only: a NaN loss whose gradient is NaN too
                        loss = loss * loss.new_tensor(float("nan"))
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,
                                                           norm_type=2.0, error_if_nonfinite=False)
                    optimizer.step()
                except QATStop:
                    raise
                except Exception as e:                     # noqa: BLE001 -- recorded in a non-finite state
                    # AM-21 item 1(b): evaluated at every raise, on the state a checkpoint would hold now
                    step_ok, step_fails = Q.state_predicate(Q.cpu_state(prepared), kinds, never)
                    if step_ok:
                        raise                              # item 1(e): a raise while the state is finite aborts
                    if nonfinite_since is None:
                        _mark_nonfinite(step, step_fails)
                    scheduler.step()
                    ep["step_errors"] += 1
                    tel.write({"event": "step_error", "wall_clock": time.time(), "epoch": epoch,
                               "step": step, "epoch_step": ep["steps"], "lr": lr_now,
                               "error": f"{type(e).__name__}: {str(e)[:300]}", "batch_sha256": fp,
                               "state_nonfinite": True})
                    continue
                scheduler.step()
                loss_v, ce_v, dice_v, gn = (float(loss.item()), float(ce.item()), float(dice.item()),
                                            float(gnorm.item()))
                if not math.isfinite(loss_v):
                    ep["nonfinite_loss"] += 1
                if not math.isfinite(gn):
                    ep["nonfinite_grad"] += 1
                clipped = bool(math.isfinite(gn) and gn > clip_norm)
                ep["clipped"] += int(clipped)
                if math.isfinite(gn):
                    prev = ep["max_pre_clip_norm"]
                    ep["max_pre_clip_norm"] = gn if prev is None else max(prev, gn)
                if nonfinite_since is None and (not math.isfinite(loss_v) or not math.isfinite(gn)):
                    ok, fails = Q.state_predicate(prepared.state_dict(), kinds, never)
                    if not ok:
                        _mark_nonfinite(step, fails)
                row = {"event": "train", "wall_clock": time.time(), "epoch": epoch, "step": step,
                       "epoch_step": ep["steps"], "lr": lr_now, "loss": loss_v, "ce": ce_v, "dice": dice_v,
                       "grad_norm": gn, "clipped": clipped, "bn_frozen": bn_frozen,
                       "observers_enabled": obs_on, "batch_sha256": fp,
                       "state_nonfinite": nonfinite_since is not None,
                       "iter_seconds": time.time() - t0}
                row.update({"sup": float(sup.item()), "logit_kd": parts.get("logit_kd"), "kd_weight": kd_weight})
                row = tel.write(row)
                if hooks is not None:
                    hooks.after_step(step, prepared, row)
            if max_steps is not None and step >= max_steps and ep["steps"] == 0:
                break

            # the state the epoch checkpoint holds: item 1(a) read on it and a first finding recorded (AM-21 item
            # 1(b)), then the checkpoint written before the VAL bracket, carrying that state flag (item 1(c))
            state = Q.cpu_state(prepared)
            state_sha = Q.state_digest(state)
            finite, fails = Q.state_predicate(state, kinds, never)
            if not finite and nonfinite_since is None:
                _mark_nonfinite(step, fails)
            ck_path, ck_sha, ck_bytes = Q.save_epoch_checkpoint(
                out, stage_name=st["name"], source_stage=st["source_stage"], epoch=epoch, step=step,
                state=state, state_sha256=state_sha, seed=seed, clip_norm=clip_norm, run_id=run_id,
                source_sha256=source_meta["sha256"], state_finite=bool(finite), nonfinite_since_step=nonfinite_since)
            # the VAL bracket: observers off for the pass, their snapshot restored after it
            if hooks is not None:
                hooks.before_val(epoch, prepared)
            flags = Q.observer_flags(prepared)
            disable_observers(prepared)
            t_val = time.time()
            val_error = None
            try:
                all_miou, disease_miou, cm, n_val_batches = validate(prepared, val_loader, dev, Q.NUM_CLASSES,
                                                                     max_val_batches)
            except Exception as e:                         # noqa: BLE001 -- recorded in a non-finite state
                # AM-21 item 1(b): evaluated at every raise, on the state the epoch's checkpoint holds
                val_ok, _ = Q.state_predicate(state, kinds, never)
                if val_ok:
                    Q.restore_observer_flags(prepared, flags)
                    raise                                  # item 1(e): a raise while the state is finite aborts
                # not finite: the epoch-end evaluation of this same state has already recorded it (item 1(c))
                prepared.train()                           # validate() returns the model to train mode only on success
                all_miou = disease_miou = None
                cm, n_val_batches = None, 0
                val_error = f"{type(e).__name__}: {str(e)[:300]}"
            del state
            Q.restore_observer_flags(prepared, flags)
            val_seconds = time.time() - t_val
            # the bracket left the state untouched
            if Q.state_digest(prepared) != state_sha:
                raise QATStop("val_bracket_changed_state",
                              f"epoch {epoch}: the state after the VAL bracket differs from the checkpoint's")
            if teacher.state_sha256() != teacher_state:
                raise QATStop("teacher_changed", f"epoch {epoch}: the frozen teacher's state changed")
            if hooks is not None:
                hooks.after_val(epoch, prepared)
            val_row = {"event": "val", "wall_clock": time.time(), "epoch": epoch, "step": step,
                       "kind": "fake_quant", "all_class_miou": all_miou,
                       "disease_only_miou_PROVISIONAL": disease_miou,
                       "val_batches": n_val_batches, "val_total_px": int(cm.sum()) if cm is not None else None,
                       "val_seconds": val_seconds, "observers_disabled_during_val": True,
                       "state_unchanged": True, "selects": False,
                       "state_nonfinite": nonfinite_since is not None}
            if val_error is not None:
                val_row["error"] = val_error
            tel.write(val_row)
            end_row = {"event": "epoch_end", "wall_clock": time.time(), "epoch": epoch, "step": step,
                       "steps_in_epoch": ep["steps"], "checkpoint": f"{Q.EPOCH_DIR}/{ck_path.name}",
                       "checkpoint_sha256": ck_sha, "checkpoint_bytes": ck_bytes, "state_sha256": state_sha,
                       "fake_quant_val_all_class_miou": all_miou,
                       "bn_frozen": epoch > Q.BN_FREEZE_EPOCH, "observers_enabled": epoch <= Q.OBS_FREEZE_EPOCH,
                       "nonfinite_loss_steps": ep["nonfinite_loss"], "nonfinite_grad_steps": ep["nonfinite_grad"],
                       "step_errors": ep["step_errors"], "clipped_steps": ep["clipped"],
                       "max_pre_clip_norm": ep["max_pre_clip_norm"], "state_finite": bool(finite),
                       "nonfinite_since_step": nonfinite_since, "state_nonfinite": nonfinite_since is not None,
                       "run_complete": epoch == Q.EPOCHS and max_steps is None}
            end_row["teacher_state_sha256"] = teacher_state
            end_row = tel.write(end_row)
            if hooks is not None:
                hooks.after_epoch(epoch, prepared, ck_path, end_row)
            log(f"[n2r2t] epoch {epoch:02d}/{Q.EPOCHS} step {step} fake-quant VAL {all_miou}"
                f"{' (non-finite state)' if nonfinite_since is not None else ''}")
            if max_steps is not None and step >= max_steps:
                break
    except QATStop as e:
        tel.write({"event": "run_stop", "wall_clock": time.time(), "epoch": epoch, "step": step,
                   "code": e.code, "message": str(e)[:500], "state_nonfinite": nonfinite_since is not None})
        raise
    except BaseException as e:
        tel.write({"event": "run_abort", "wall_clock": time.time(), "epoch": epoch, "step": step,
                   "cause": "exception", "exception_type": type(e).__name__, "message": str(e)[:500],
                   "state_nonfinite": nonfinite_since is not None})
        raise
    complete = epoch == Q.EPOCHS and max_steps is None
    return {"run_id": run_id, "stage": st["name"], "seed": seed, "clip_norm": clip_norm, "lr": lr,
            "telemetry": str(tel_path), "steps": step, "epochs_completed": epoch,
            "complete": complete, "nonfinite_since_step": nonfinite_since}


# ------------------------------------------------------------------ N2's change, as run_n2r2 applies it
def run_n2r2t(*, mode: str, model: nn.Module, teacher: SelfTeacher, source_meta: dict, out_dir, seed: int,
              clip_norm: float, clip_source: str, device: str, num_workers: int, parent: dict | None,
              factor: float = FACTOR, e3_lambda: float = E3_LAMBDA_OF_RECORD, extra_meta: dict | None = None,
              sites: bool = True, loaders=None, hooks: Q.QATHooks | None = None, inject_nan_at_steps=(),
              max_steps: int | None = None, max_val_batches: int | None = None, log=print):
    """N2R2T's run: run_n2r2t_qat on stage e6-r2 with N2's change, made as src/explore/qat_explore.py run_n2r2 makes
    it: TF32 off (both flags, set_tf32), then prepare_qat_model of src/quant/qat.py bound for this call to a wrapper that
    calls the pinned function and installs the ten site fake-quants (SiteMatcher), the binding recorded in run_meta
    (x_substitution) and restored after. Returns (summary, substitution record, matcher). `sites` False runs the same
    call with neither TF32 nor the binding touched (AM-29's T1 term on E6-R2 without N2's change); it is refused in a
    real run. `mode` "smoke" passes injected loaders, hooks, NaN injection, a step cap and a VAL cap to the loop,
    which refuses them in a real run, as it refuses a factor other than 0.5."""
    from .patching import substituted
    from .sites import SITE_NAMES, SiteMatcher, set_tf32
    if mode == "real" and sites is not True:
        raise X.ExploreQATError("n2r2t_change_required", "a real N2R2T run carries N2's change: the ten sites and "
                                                         "TF32 off")
    kw = dict(mode=mode, model=model, teacher=teacher, source_meta=source_meta, out_dir=out_dir, seed=seed,
              clip_norm=clip_norm, clip_source=clip_source, device=device, num_workers=num_workers, parent=parent,
              factor=factor, e3_lambda=e3_lambda, loaders=loaders, hooks=hooks,
              inject_nan_at_steps=inject_nan_at_steps, max_steps=max_steps, max_val_batches=max_val_batches, log=log)
    extra = dict(extra_meta or {})
    if sites is not True:
        return run_n2r2t_qat(extra_meta=extra, **kw), None, None
    holder = {}
    pinned_prepare = Q.prepare_qat_model

    def prepare_with_sites(m, *, select_backend=True):
        prepared = pinned_prepare(m, select_backend=select_backend)
        holder["matcher"] = SiteMatcher(prepared, SITE_NAMES)
        return prepared

    extra["x_tf32_set"] = set_tf32(False)
    with substituted(Q, "prepare_qat_model", prepare_with_sites, item="N2R2T",
                     why="install the ten input-grid site fake-quants on the prepared student") as rec:
        extra["x_substitution"] = rec
        summary = run_n2r2t_qat(extra_meta=extra, **kw)
    return summary, rec, holder.get("matcher")


def recipe_extra() -> dict:
    """scripts/explore_qat.py's recipe fields for N2R2T (x_run_meta.json's recipe)."""
    return {"parent_of_record": X.E3R2_PARENT_OF_RECORD, "matched_run": MATCHED_RUNS, "stage": "E6-R2",
            "registered_by": "AM-29a (E6-R2-MT); lane addendum L-AM24-ADD-3",
            "term": {"text": TERM_TEXT, "factor": FACTOR, "e3_lambda_logit": E3_LAMBDA_OF_RECORD,
                     "kd_weight": FACTOR * E3_LAMBDA_OF_RECORD, "T_logit": T_LOGIT, "ramp": T_RAMP,
                     "lr": Q.LEARNING_RATE, "copied": COPIED}}


__all__ = ["AM29_COMMIT", "AM29_FILE_SHA256", "AM29_PATH", "COPIED", "E2_SWITCHES", "E3_LAMBDA_OF_RECORD", "FACTOR",
           "ITEM", "MATCHED_RUNS", "OBJECTIVE", "RUN_ID", "SEED", "SelfTeacher", "TERM_CODE_PATHS", "TERM_TEXT",
           "T_LOGIT", "T_RAMP", "recipe_extra", "run_n2r2t", "run_n2r2t_qat", "t_logit_kd", "teacher_record",
           "term_record", "values_error"]
