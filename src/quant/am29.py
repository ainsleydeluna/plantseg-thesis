"""AM-29's QAT runner (lane L-AM29, P1): E6's QAT recipe of record with three parameters changed, for an allow-list.

Every arm runs E6's QAT recipe of record (AM-4a item 1), imported from the code of record unchanged: the qconfig
(src/quant/qconfig.py), prepare (src/quant/prepare.py), 15 epochs x 335 steps, batch 16 (E1's TRAIN loader, drop_last),
SGD momentum 0.9 / weight decay 1e-4 / dampening 0 / no Nesterov, CosineAnnealingLR to 0 stepped after every step, BN
statistics frozen at the top of epoch 11 and every observer at the top of epoch 13, the VAL bracket, the state
predicate, the epoch checkpoint and the telemetry of src/quant/qat.py, and the clip of record bound through
clip_selection.json (DL-139: 5.0, sha256 c433145a...). Its only changes are three parameters, accepted for ARMS only:

  T1-T4    parent E3-R2 seed 42 (best.json checkpoint, sha256 da70d82f...), seed 42. Loss: the QAT loss of record
           (CE + Dice) plus factor x E3's logit-KD weight x E3's own logit-KD term (train_distill.distillation_losses
           on stage e2's switches, i.e. losses.logit_kd_kl on the head's OS8 logits, the validity mask downsampled to
           that grid, T = train_distill.T_LOGIT = 4, T^2 inside the term). Its teacher is SelfTeacher: a frozen FP32
           copy of the same E3-R2 checkpoint (eval mode, no grad, outside the optimizer). No CWD term, no projection,
           no SegNeXt-B. (factor, lr): T1 (0.5, 3e-4), T2 (0.5, 6.25e-5), T3 (1.0, 3e-4), T4 (1.0, 6.25e-5).
  A3-s43/44  E6-KD's recipe exactly, as src/quant/qat_kd.py builds it (E3's three terms at 0.5x, its T, the teacher of
           record, the parent's own CWD projection), at seed 43 / 44, from E3 seed 43's / 44's best checkpoint paired
           with its own projection.pt by sha256; rate 3e-4. The A3 gates replace qat_kd.py's seed-42-only refusal
           (:937), its trigger (:951, :1000) and its alpha-winner parent (:998) for these two arms only.
  LR-s42/43/44  plain QAT (E6's recipe) from E3 seed 42 / 43 / 44 at lr 6.25e-5.

The loops are run_qat's (T, LR) and run_qat_kd's (A3) line for line, with the arm's stage (AM29-<ARM>, registered by
src/quant/am29_registry.py), the arm's rate in every parameter group and the scheduler, and for T the teacher forward
and the one term added: run_qat and run_qat_kd hold their stage and rate as constants, so they cannot be called with
another. scripts/smoke_am29_runner.py runs both copies against the originals on synthetic batches.

Not decided by the spec, and so recorded, never invented: the T arms' term has no warm-up ramp (the weight is the
constant factor x lambda); T and LR follow run_qat's non-finite handling (AM-21 item 1), A3 run_qat_kd's generic abort.

    python -B scripts/run_am29.py --arm T1 --seed 42 --factor 0.5 --lr 3e-4 --real-run --confirm-real-run ...
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.distill import LOGIT_KD_SEMANTICS                                 # noqa: E402
from src.distill import teacher as TT                                          # noqa: E402
from src.distill.cwd_projection import STUDENT_C5_CH, TEACHER_STRIDE16_CH       # noqa: E402
from src.distill.features import StudentTaps                                   # noqa: E402
from src.distill.nmf_stream import M4_NMF_SEED                                 # noqa: E402
from src.training import train_distill as TD                                   # noqa: E402

from . import am29_registry as R                                               # noqa: E402
from . import qat as Q                                                         # noqa: E402
from . import qat_kd as KD                                                     # noqa: E402
from .prepare import disable_observers, freeze_bn_stats, prepare_qat_model, qat_freeze_steps, unfused_batchnorm  # noqa: E402
from .qat import QATRefused, QATStop                                           # noqa: E402
from .qconfig import QuantBackendUnavailable, select_qnnpack_backend           # noqa: E402

LANE = "L-AM29 (AM-29's new code; P1)"
TELEMETRY_SCHEMA = "plantseg-am29-telemetry/1.0.0"
LR_OF_RECORD = Q.LEARNING_RATE                       # 3e-4, AM-4a item 1
LR_AM29 = 6.25e-5                                    # AM-29: the reduced rate of T2, T4 and LR-s42/43/44
CLIP_OF_RECORD = 5.0                                 # DL-139: the U4 winner
CLIP_SELECTION_SHA256 = "c433145a6cfe0e133ebee8fcb70a76f05b60fc6462670abee34c5304704fe698"   # DL-139
E3_LAMBDA_OF_RECORD = 0.5                            # DL-121: e3_s42_lambda0.5_alpha25_80k_a1 (the lambda selection's)
E3_ALPHA_OF_RECORD = 25.0                            # DL-121: RESULT: SELECTED alpha_cwd = 25
T_LOGIT = TD.T_LOGIT                                 # E3's constant (train_distill.py:142); never a new value
E2_SWITCHES = TD.resolve_stage("e2")                 # logit KD only: E3's logit-KD term, alone (no CWD, no projection)
T_RAMP = None                                        # the T term's weight is constant: the spec names no ramp
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# ------------------------------------------------------------------ the allow-list (AM-29 P1)
ARMS = {
    "T1": {"family": "T", "seed": 42, "factor": 0.5, "lr": LR_OF_RECORD, "parent": "E3-R2-s42"},
    "T2": {"family": "T", "seed": 42, "factor": 0.5, "lr": LR_AM29, "parent": "E3-R2-s42"},
    "T3": {"family": "T", "seed": 42, "factor": 1.0, "lr": LR_OF_RECORD, "parent": "E3-R2-s42"},
    "T4": {"family": "T", "seed": 42, "factor": 1.0, "lr": LR_AM29, "parent": "E3-R2-s42"},
    "A3-s43": {"family": "A3", "seed": 43, "factor": KD.WEIGHT_FACTOR, "lr": LR_OF_RECORD, "parent": "E3-s43"},
    "A3-s44": {"family": "A3", "seed": 44, "factor": KD.WEIGHT_FACTOR, "lr": LR_OF_RECORD, "parent": "E3-s44"},
    "LR-s42": {"family": "LR", "seed": 42, "factor": None, "lr": LR_AM29, "parent": "E3-s42"},
    "LR-s43": {"family": "LR", "seed": 43, "factor": None, "lr": LR_AM29, "parent": "E3-s43"},
    "LR-s44": {"family": "LR", "seed": 44, "factor": None, "lr": LR_AM29, "parent": "E3-s44"},
}
if set(ARMS) != set(R.ARM_STAGES):
    raise RuntimeError("src/quant/am29.py ARMS and src/quant/am29_registry.py ARM_STAGES disagree")

# The parents of record, by the sha256 the decision log records (a value it does not record stays None and is a
# required launch argument, never a guess).
PARENTS_OF_RECORD = {
    "E3-R2-s42": {"seed": 42, "arm": "E3-R2", "run_id": None, "checkpoint": None,
                  "sha256": "da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4",
                  "projection_sha256": None, "record": "the L-AM29 prompt (DECISION_LOG DL-136 holds the value)"},
    "E3-s42": {"seed": 42, "arm": None, "run_id": "e3_s42_lambda0.5_alpha25_80k_a1",
               "checkpoint": "e3_student_best_iter76000.pt",
               "sha256": "4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d",
               "projection_sha256": "2e93ebe05e8c7d55b9ce7305a9066aab2e1777a710555c18ff9ea0cb84726751",
               "record": "DL-121 (checkpoint); DL-120 (projection.pt of alpha 25)"},
    "E3-s43": {"seed": 43, "arm": None, "run_id": "e3_s43_lambda0.5_alpha25_80k_a1",
               "checkpoint": "e3_student_best_iter72000.pt",
               "sha256": "f4d6f689d5545fb222fc9cb706c8143e5c04dff2082e86135dfdcabab4fc9a0e",
               "projection_sha256": None,
               "record": "DL-218 (checkpoint); projection.pt: not recorded (DL-158 records the keep tar only)"},
    "E3-s44": {"seed": 44, "arm": None, "run_id": "e3_s44_lambda0.5_alpha25_80k_a1",
               "checkpoint": "e3_student_best_iter60000.pt",
               "sha256": "6585ed09398ec8225a2d49304e1e5d8611b585456ac3e7c111eb1744c313363b",
               "projection_sha256": None,
               "record": "DL-218 (checkpoint); projection.pt: not recorded (DL-158 records the keep tar only)"},
}
# projection.pt files of record of other runs (E3 seed 42's alpha 50 and alpha 100 candidates; DL-120)
OTHER_PROJECTIONS_OF_RECORD = {
    "e95989e2e8788ee228c3e4639e850cf8723c5c007603467fafb07ef913c3aecd": "E3 seed 42, alpha 50 (DL-120)",
    "d40107c92461489b93e5d4c02f7e83080c1ab324badc263b26373a2df92c0365": "E3 seed 42, alpha 100 (DL-120)",
}

AM29_FILES = ("src/quant/am29.py", "src/quant/am29_registry.py", "src/quant/am29_islands.py", "src/eval/am29_score.py",
              "scripts/run_am29.py", "scripts/am29_blockb.py", "scripts/am29_islands.py", "scripts/am29_score.py",
              "configs/qat_selection_rules_am29.json")
# The code paths whose cleanliness at HEAD a launch requires and whose sha256 run_meta records: E6-KD's (a superset of
# E6's QAT_CODE_PATHS that holds train_distill and the distill modules the T and A3 arms load), plus AM-29's files.
AM29_CODE_PATHS = tuple(dict.fromkeys((*KD.E6KD_CODE_PATHS, *AM29_FILES)))

OBJECTIVE_T = ("supervised CE (E1 class weights) + Dice + factor x E3's lambda_logit x E3's logit-KD term (KL at T = 4, "
               "T^2 in the term, on the head's OS8 logits over the downsampled validity mask), the teacher a frozen FP32 "
               "copy of the parent checkpoint; no CWD term, no projection, no ramp (AM-29 T1-T4)")


def _refuse(code: str, message: str):
    raise QATRefused(code, message)


def _is_float(v) -> bool:
    return isinstance(v, float) and not isinstance(v, bool) and math.isfinite(v)


def arm_spec(arm: str) -> dict:
    if arm not in ARMS:
        _refuse("arm_unknown", f"--arm {arm!r} is not on AM-29's allow-list {sorted(ARMS)}")
    return {"arm": arm, "stage": R.stage_of_arm(arm), **ARMS[arm]}


def arm_values_error(arm: str, *, seed, factor, lr) -> tuple[str, str] | None:
    """The allow-list: the arm and exactly its seed, factor and rate. Returns (code, text) or None."""
    if arm not in ARMS:
        return "arm_unknown", f"--arm {arm!r} is not on AM-29's allow-list {sorted(ARMS)}"
    spec = ARMS[arm]
    if isinstance(seed, bool) or seed != spec["seed"]:
        return "arm_seed", f"--arm {arm} runs seed {spec['seed']}, not {seed!r}"
    if spec["factor"] is None:
        if factor is not None:
            return "arm_factor", f"--arm {arm} is plain QAT: it takes no --factor (got {factor!r})"
    elif not _is_float(factor) or factor != spec["factor"]:
        return "arm_factor", f"--arm {arm} runs factor {spec['factor']}, not {factor!r}"
    if not _is_float(lr) or lr != spec["lr"]:
        return "arm_lr", f"--arm {arm} runs lr {spec['lr']!r}, not {lr!r}"
    return None


def am29_stage(arm: str) -> dict:
    """The arm's stage dict, as resolve_quant_stage returns one (key, name, method, source_stage, objective)."""
    stage = R.stage_of_arm(arm)
    return {"key": stage.lower(), **R.QUANT_ENTRIES[stage.lower()]}


def code_identity() -> dict:
    from .ptq import code_provenance
    prov = code_provenance(AM29_CODE_PATHS)
    return {"git_head": prov["commit"], "code_clean_at_head": prov["clean_at_commit"],
            "code_files_sha256": prov["files_sha256"]}


def optimizer_error(rec: dict, lr: float) -> str | None:
    """src/quant/qat.py optimizer_of_record_error with the arm's rate in place of 3e-4 (equal to it at 3e-4)."""
    want = {"class": "SGD", "param_groups": 1, "group_holds_every_model_param": True, "lr": lr, "initial_lr": lr,
            "momentum": Q.MOMENTUM, "weight_decay": Q.WEIGHT_DECAY, "dampening": Q.DAMPENING, "nesterov": Q.NESTEROV}
    wrong = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v or type(rec.get(k)) is not type(v)}
    return None if not wrong else f"the optimizer differs from the AM-29 recipe (AM-4a item 1, lr {lr!r}): {wrong}"


# ------------------------------------------------------------------ T1-T4: the teacher and the term
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


def t_logit_kd(*, logits, head_logits, c5, mask, teacher_out, weight: float):
    """E3's own logit-KD term at `weight` (factor x E3's lambda_logit): train_distill.distillation_losses on stage e2's
    switches, i.e. weight x losses.logit_kd_kl(head_logits, teacher OS8 logits, downsample_validity(mask), T=T_LOGIT).
    Returns (weight x term, {"logit_kd": the unweighted term})."""
    return TD.distillation_losses(stage=E2_SWITCHES, logits=logits, head_logits=head_logits, c5=c5, mask=mask,
                                  teacher_out=teacher_out, projection=None, lambda_logit=weight, ramp=1.0)


# ------------------------------------------------------------------ the T and LR run (run_qat's loop)
def run_am29_qat(*, arm: str, mode: str, model: nn.Module, source_meta: dict, out_dir, seed: int,
                 clip_norm: float, clip_source: str, device: str, num_workers: int, lr: float,
                 factor: float | None = None, e3_lambda: float | None = None, teacher: SelfTeacher | None = None,
                 loaders=None, parent: dict | None = None, extra_meta: dict | None = None,
                 hooks: Q.QATHooks | None = None, inject_nan_at_steps=(), max_steps: int | None = None,
                 max_val_batches: int | None = None, log=print) -> dict:
    """Train one T or LR arm: src/quant/qat.py run_qat line for line, with the arm's stage and rate and, for T, the
    teacher forward and the one term. Returns a summary; raises QATRefused, QATStop.

    `mode` "real" is the launch (main() has passed every gate): the arm's values only. `mode` "smoke" accepts injected
    loaders, hooks, NaN injection, a step cap, a VAL cap and any rate and factor (a factor 0 run is plain QAT)."""
    from src.seeds import set_seed
    from src.training.losses import CombinedCEDiceLoss
    from src.training.train_e1 import CLASS_WEIGHTS_JSON, _image_digest, load_ce_weights, validate

    spec = arm_spec(arm)
    if spec["family"] not in ("T", "LR"):
        _refuse("arm_family", f"{arm} is not a T or LR arm; A3 runs run_am29_a3")
    st = am29_stage(arm)
    if mode not in Q.MODES:
        _refuse("mode_invalid", f"mode {mode!r} not in {Q.MODES}")
    if mode == "real" and (loaders is not None or hooks is not None or inject_nan_at_steps or max_steps
                           or max_val_batches):
        _refuse("real_run_test_hooks", "a real run takes no injected loaders, hooks, NaN injection, step cap "
                                       "or VAL cap")
    if mode == "real" and clip_source != "clip_selection":
        _refuse("clip_source", f"an AM-29 run's clip is the U4 winner bound at launch (DL-139), not {clip_source!r}")
    pins = Q.config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    if isinstance(clip_norm, bool) or not isinstance(clip_norm, float) or not math.isfinite(clip_norm) \
            or clip_norm <= 0:
        _refuse("grad_clip_norm_invalid", f"clip norm must be a positive finite float, got {clip_norm!r}")
    if not _is_float(lr) or lr <= 0:
        _refuse("lr_invalid", f"lr must be a positive finite float, got {lr!r}")
    if spec["family"] == "T":
        if not isinstance(teacher, SelfTeacher) or teacher.trainable_parameters():
            _refuse("teacher_not_frozen", "a T arm's teacher is a SelfTeacher with no trainable parameter")
        if not _is_float(factor) or not _is_float(e3_lambda):
            _refuse("weight_invalid", f"factor {factor!r} and e3_lambda {e3_lambda!r} must be finite floats")
        kd_weight = factor * e3_lambda
    else:
        if teacher is not None or factor is not None or e3_lambda is not None:
            _refuse("teacher_not_applicable", f"{arm} is plain QAT: no teacher, no factor, no lambda")
        kd_weight = None
    if mode == "real":
        err = arm_values_error(arm, seed=seed, factor=factor, lr=lr)
        if err:
            _refuse(*err)
        if spec["family"] == "T" and e3_lambda != E3_LAMBDA_OF_RECORD:
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
    prepared = prepare_qat_model(model, select_backend=False)
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
    teacher_state = None
    if teacher is not None:
        teacher = teacher.to(dev)
        teacher.eval()
        teacher_state = teacher.state_sha256()

    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=Q.IGNORE_INDEX).to(dev)
    optimizer = torch.optim.SGD(prepared.parameters(), lr=lr, momentum=Q.MOMENTUM,
                                weight_decay=Q.WEIGHT_DECAY, dampening=Q.DAMPENING, nesterov=Q.NESTEROV)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=Q.ETA_MIN)
    opt_rec = Q.optimizer_record(optimizer, prepared)
    opt_err = optimizer_error(opt_rec, lr)
    if opt_err:
        raise QATStop("optimizer_of_record", opt_err)
    if teacher is not None and teacher.parameter_ids() & {id(p) for g in optimizer.param_groups for p in g["params"]}:
        raise QATStop("teacher_in_optimizer", "a teacher parameter reached the optimizer")

    from .ptq import qconfig_summary
    from .qconfig import qat_qconfig
    qsum = qconfig_summary(qat_qconfig())
    identity = code_identity()
    import numpy
    import torchvision
    tel_path = out / Q.TELEMETRY_NAME
    run_meta = {
        "event": "run_meta", "wall_clock": time.time(), "schema": TELEMETRY_SCHEMA, "lane": LANE,
        "mode": mode, "stage": st["name"], "source_stage": st["source_stage"], "seed": seed,
        "run_id": run_id, "out_dir": str(out.resolve()),
        "parent": parent, "source_checkpoint_sha256": source_meta["sha256"],
        "source_checkpoint": source_meta.get("path"), "source_checkpoint_bytes": source_meta.get("bytes"),
        "clip_norm": clip_norm, "clip_source": clip_source,
        "epochs": Q.EPOCHS, "batch_size": loader_batch, "drop_last": loader_drop_last,
        "gradient_accumulation": None, "ema": False, "distillation": teacher is not None,
        "steps_per_epoch": steps_per_epoch, "total_steps": t_max,
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "loader": loader_source, "num_workers": num_workers, "val_batches_cap": max_val_batches,
        "persistent_workers": bool(getattr(train_loader, "persistent_workers", False)),
        "scheduler": {"class": type(scheduler).__name__, "T_max": scheduler.T_max,
                      "eta_min": scheduler.eta_min, "step": "once after every optimizer.step",
                      "base_lrs": list(scheduler.base_lrs)},
        "optimizer": opt_rec, "lr": opt_rec["lr"], "momentum": opt_rec["momentum"],
        "weight_decay": opt_rec["weight_decay"], "nesterov": opt_rec["nesterov"],
        "dampening": opt_rec["dampening"],
        "bn_freeze_epoch": Q.BN_FREEZE_EPOCH, "obs_freeze_epoch": Q.OBS_FREEZE_EPOCH,
        "bn_frozen_from_epoch": Q.BN_FREEZE_EPOCH + 1, "obs_frozen_from_epoch": Q.OBS_FREEZE_EPOCH + 1,
        "bn_freeze_after_step": bn_after_step, "obs_freeze_after_step": obs_after_step,
        "objective": OBJECTIVE_T if teacher is not None else Q.OBJECTIVE,
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
        "am29": {"arm": arm, "family": spec["family"], "factor": factor, "lr": lr, "seed": seed,
                 "parent_sha256": source_meta["sha256"],
                 "teacher_sha256": None if teacher is None else teacher.checkpoint_sha256,
                 "teacher_state_sha256": teacher_state, "T_logit": T_LOGIT if teacher is not None else None,
                 "e3_lambda_logit": e3_lambda, "kd_weight": kd_weight, "ramp": T_RAMP,
                 "code_paths": list(AM29_CODE_PATHS), "new_files": list(AM29_FILES)},
    }
    if teacher is not None:
        run_meta.update({"terms": {t: bool(E2_SWITCHES[t]) for t in TD.TERMS}, "T_logit": T_LOGIT,
                         "logit_kd_semantics": LOGIT_KD_SEMANTICS,
                         "teacher": {"kind": "frozen FP32 copy of the parent checkpoint (SelfTeacher)",
                                     "checkpoint_sha256": teacher.checkpoint_sha256, "state_sha256": teacher_state,
                                     "eval_mode": True, "requires_grad": False, "in_optimizer": False}})
    if st["source_stage"] == "E3":
        run_meta["cwd_projection_loaded"] = False
    if extra_meta:
        run_meta.update(extra_meta)
    out.mkdir(parents=True, exist_ok=True)
    tel = Q.Telemetry(tel_path)
    tel.write(run_meta)
    if hooks is not None:
        hooks.on_start(prepared, optimizer, scheduler, run_meta)
    log(f"[am29] {st['name']} seed {seed} clip {clip_norm} lr {lr} kd {kd_weight} | {steps_per_epoch} steps/epoch, "
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
                    if teacher is None:
                        logits = prepared(img_d)
                        ce = criterion.ce(logits, mask_d)
                        dice = criterion.dice(logits, mask_d)
                        loss = ce + dice
                    else:
                        seen: dict = {}
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
                                raise QATStop("step1_checks", f"T-arm step-1 checks failed: {checks}")
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
                if teacher is not None:
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
            if teacher is not None and teacher.state_sha256() != teacher_state:
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
            if teacher is not None:
                end_row["teacher_state_sha256"] = teacher_state
            end_row = tel.write(end_row)
            if hooks is not None:
                hooks.after_epoch(epoch, prepared, ck_path, end_row)
            log(f"[am29] epoch {epoch:02d}/{Q.EPOCHS} step {step} fake-quant VAL {all_miou}"
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
    return {"run_id": run_id, "stage": st["name"], "arm": arm, "seed": seed, "clip_norm": clip_norm, "lr": lr,
            "telemetry": str(tel_path), "steps": step, "epochs_completed": epoch,
            "complete": complete, "nonfinite_since_step": nonfinite_since}


# ------------------------------------------------------------------ A3: run_qat_kd's loop
def run_am29_a3(*, arm: str, mode: str, model: nn.Module, teacher, projection: nn.Module, weights: dict, e3: dict,
                source_meta: dict, out_dir, seed: int, clip_norm: float, clip_source: str, device: str,
                num_workers: int, projection_record: dict | None = None, loaders=None, parent: dict | None = None,
                extra_meta: dict | None = None, hooks: Q.QATHooks | None = None, inject: dict | None = None,
                max_steps: int | None = None, max_val_batches: int | None = None, log=print) -> dict:
    """Train A3-s43 or A3-s44: src/quant/qat_kd.py run_qat_kd line for line, with the arm's stage. Returns a summary;
    raises QATRefused (nothing written), QATStop, or qat_kd.E6KDAborted after the run_abort record.

    `mode` "real": the arm's seed, and `weights` exactly 0.5x `e3`'s (AM-3, as E6-KD). `mode` "smoke": injected
    loaders, hooks, the injections of qat_kd.INJECTIONS, a step cap, a VAL cap, any seed and any weights."""
    from src.seeds import set_seed
    from src.training.losses import CombinedCEDiceLoss
    from src.training.train_e1 import CLASS_WEIGHTS_JSON, _image_digest, load_ce_weights, validate

    spec = arm_spec(arm)
    if spec["family"] != "A3":
        _refuse("arm_family", f"{arm} is not an A3 arm")
    st = am29_stage(arm)
    if st["method"] != KD.METHOD:
        _refuse("stage_not_qat_kd", f"{st['name']} is registered with method {st['method']!r}, not {KD.METHOD!r}")
    if mode not in Q.MODES:
        _refuse("mode_invalid", f"mode {mode!r} not in {Q.MODES}")
    unknown = sorted(set(inject or {}) - set(KD.INJECTIONS))
    if unknown:
        raise ValueError(f"unknown injections {unknown}; the smokes' injections are {KD.INJECTIONS}")
    inj = {k: {int(s) for s in (inject or {}).get(k, ())} for k in KD.INJECTIONS}
    if mode == "real" and (loaders is not None or hooks is not None or any(inj.values()) or max_steps is not None
                           or max_val_batches is not None):
        _refuse("real_run_overrides", "a real run takes no injected loaders, hooks, injections, step cap or VAL cap")
    if mode == "real" and clip_source != "clip_selection":
        _refuse("clip_source", f"A3's clip is the U4 winner bound at launch (AM-4a item 3), not {clip_source!r}")
    if mode == "real" and seed != spec["seed"]:
        _refuse("arm_seed", f"{arm} runs seed {spec['seed']}, not {seed!r}")
    pins = Q.config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    if isinstance(clip_norm, bool) or not isinstance(clip_norm, float) or not math.isfinite(clip_norm) \
            or clip_norm <= 0:
        _refuse("grad_clip_norm_invalid", f"clip norm must be a positive finite float, got {clip_norm!r}")
    if sorted(weights) != sorted(KD.TERMS) or any(isinstance(v, bool) or not isinstance(v, float)
                                                  or not math.isfinite(v) for v in weights.values()):
        _refuse("weights_invalid", f"weights {weights!r}: one finite float per term {KD.TERMS}")
    if mode == "real" and (weights != KD.e6kd_weights(e3["e3_weights"])
                           or any(2.0 * weights[t] != e3["e3_weights"][t] for t in KD.TERMS)):
        _refuse("weights_not_half", f"weights {weights!r} are not exactly 0.5x E3's {e3['e3_weights']!r} (AM-3)")
    if not isinstance(teacher, TT.FrozenTeacher) or teacher.trainable_parameters():
        _refuse("teacher_not_frozen", "the teacher must be a FrozenTeacher with no trainable parameter")
    proj_fq = [n for n, m in projection.named_modules() if isinstance(m, Q._fq_base()) or getattr(m, "qconfig", None)]
    proj_ids = {id(p) for p in projection.parameters()}
    if (not isinstance(projection, nn.Conv2d) or proj_fq or projection.weight.dtype != torch.float32
            or proj_ids & {id(p) for p in model.parameters()}):
        _refuse("projection_not_of_record", "the projection must be an FP32 nn.Conv2d with no qconfig or fake-quant "
                                            f"module, outside the student (DL-51){f': {proj_fq}' if proj_fq else ''}")
    dev = torch.device(device)
    if mode == "real" and dev.type != "cuda":
        _refuse("cuda_required", f"a real A3 run needs a CUDA device, got {device!r}")
    cuda_initialized_at_seed = bool(torch.cuda.is_initialized())
    if mode == "real" and cuda_initialized_at_seed:
        _refuse("cuda_initialized_before_seed", "CUDA was initialised before set_seed; launch in a fresh process")
    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _refuse("out_dir_not_empty", f"{out} exists and is not an empty directory; a run starts in a fresh directory "
                                     "(the telemetry is append-only)")
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
                                   f"{Q.REAL_STEPS_PER_EPOCH} / {Q.REAL_T_MAX}")
    if mode == "real" and (loader_batch != Q.BATCH_SIZE or loader_drop_last is not True):
        _refuse("loader_of_record", f"the TRAIN loader has batch_size {loader_batch!r}, drop_last "
                                    f"{loader_drop_last!r}; the recipe of record is {Q.BATCH_SIZE}, True")
    if steps_per_epoch < 1:
        _refuse("empty_loader", "the TRAIN loader yields no batch")
    bn_after_step, obs_after_step = qat_freeze_steps(steps_per_epoch, Q.BN_FREEZE_EPOCH, Q.OBS_FREEZE_EPOCH)
    ramp_iters = steps_per_epoch                             # one epoch of the TRAIN loader (DL-39; 335 of record)

    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        _refuse("backend_unavailable", str(e))
    prepared = prepare_qat_model(model, select_backend=False)
    unfused = unfused_batchnorm(prepared)
    if unfused and mode == "real":
        _refuse("unfused_batchnorm", f"{len(unfused)} BatchNorm module(s) sit outside a fused QAT ConvBn: "
                                     f"{unfused[:5]}; the epoch-11 freeze would miss them")
    census = Q.observer_census(prepared)
    fq_count = len(Q.fake_quant_modules(prepared))
    never = Q.never_observed_modules(prepared)
    kinds = Q.state_kinds(prepared)
    convbn = Q.fused_convbn_census(prepared)
    # the teacher's private M4-KD stream, begun once after set_seed (train_distill.py:926), before any device move so
    # that every refusal precedes one (the stream is a CPU generator state; a device move does not touch it)
    teacher.eval()
    teacher_nmf = teacher.begin_nmf_stream(KD.NMF_POLICY, M4_NMF_SEED)
    if mode == "real" and teacher_nmf is None:
        _refuse("teacher_nmf_missing", "the teacher of record exposes no isolated NMF stream (M4-KD); build it from "
                                       "the thesis teacher config (IsolatedNMFLightHamHead)")
    teacher_mock = bool(getattr(teacher.teacher, "is_mock", False))
    prepared = prepared.to(dev)
    prepared.train()
    teacher = teacher.to(dev)
    teacher.eval()
    projection = projection.to(dev)
    projection.train()
    student_params = list(prepared.parameters())
    projection_params = list(projection.parameters())

    ce_weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=ce_weights, ignore_index=Q.IGNORE_INDEX).to(dev)
    optimizer = KD.kd_optimizer(prepared, projection)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=Q.ETA_MIN)
    opt_rec = KD.kd_optimizer_record(optimizer, prepared, projection)
    opt_err = KD.kd_optimizer_error(opt_rec)
    if opt_err:
        raise QATStop("optimizer_of_record", opt_err)
    ramps = TD.build_term_ramps(KD.E3_STAGE, ramp_iters)

    from .ptq import qconfig_summary
    from .qconfig import qat_qconfig
    qsum = qconfig_summary(qat_qconfig())
    identity = code_identity()
    import numpy
    import torchvision
    tel_path = out / Q.TELEMETRY_NAME
    run_meta = {
        "event": "run_meta", "wall_clock": time.time(), "schema": TELEMETRY_SCHEMA, "lane": LANE,
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
                      "eta_min": scheduler.eta_min, "step": "once after every optimizer.step",
                      "base_lrs": list(scheduler.base_lrs)},
        "optimizer": opt_rec, "lr": opt_rec["lr"], "momentum": opt_rec["momentum"],
        "weight_decay": opt_rec["weight_decay"], "nesterov": opt_rec["nesterov"],
        "dampening": opt_rec["dampening"],
        "bn_freeze_epoch": Q.BN_FREEZE_EPOCH, "obs_freeze_epoch": Q.OBS_FREEZE_EPOCH,
        "bn_frozen_from_epoch": Q.BN_FREEZE_EPOCH + 1, "obs_frozen_from_epoch": Q.OBS_FREEZE_EPOCH + 1,
        "bn_freeze_after_step": bn_after_step, "obs_freeze_after_step": obs_after_step,
        "objective": KD.OBJECTIVE, "class_weights_sha256": Q.sha256_file(CLASS_WEIGHTS_JSON),
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
        # E6-KD's recipe (AM-3; DL-07, DL-39, DL-51); no cwd_projection_loaded: this run trains the projection
        "e6kd_recipe": True, "weight_factor": KD.WEIGHT_FACTOR, "descriptive": True, "holm_family": False,
        "terms": {t: True for t in KD.TERMS}, "e3_weights": dict(e3["e3_weights"]), "weights": dict(weights),
        "T_logit": TD.T_LOGIT, "T_cwd": TD.T_CWD, "cwd_C": TD.CWD_C_FEAT,
        "logit_kd_semantics": LOGIT_KD_SEMANTICS, "grids": dict(e3.get("grids") or {}),
        "ramp_iters": ramp_iters, "ramp_rule": KD.RAMP_RULE,
        "projection_source": projection_record, "projection_params": sum(p.numel() for p in projection_params),
        "projection_fake_quant_modules": len(proj_fq), "projection_in_student": False,
        "teacher_provenance": None if teacher.provenance is None else teacher.provenance.as_dict(),
        "teacher_nmf": teacher_nmf, "teacher_mock": teacher_mock,
        "abort_rule": KD.ABORT_RULE_TEXT,
        "am29": {"arm": arm, "family": "A3", "factor": KD.WEIGHT_FACTOR, "lr": opt_rec["lr"], "seed": seed,
                 "parent_sha256": source_meta["sha256"],
                 "teacher_sha256": None if teacher.provenance is None else teacher.provenance.ckpt_sha256,
                 "projection_sha256": (projection_record or {}).get("sha256"),
                 "T_logit": TD.T_LOGIT, "T_cwd": TD.T_CWD, "code_paths": list(AM29_CODE_PATHS),
                 "new_files": list(AM29_FILES),
                 "replaces": "qat_kd.py's seed-42-only refusal (:937), trigger (:951, :1000) and alpha-winner "
                             "parent (:998), for A3-s43 and A3-s44 only"},
    }
    if extra_meta:
        run_meta.update(extra_meta)
    out.mkdir(parents=True, exist_ok=True)
    tel = Q.Telemetry(tel_path)
    tel.write(run_meta)
    if hooks is not None:
        hooks.on_start(prepared, optimizer, scheduler, run_meta)
    log(f"[am29-a3] {st['name']} seed {seed} clip {clip_norm} | weights {weights} | {steps_per_epoch} steps/epoch, "
        f"T_max {t_max}, ramp {ramp_iters} | {run_id}")

    step = 0
    epoch = 0
    nonfinite_since = None
    cur: dict = {"img": None, "teacher_out": None}

    def _mark_nonfinite(at_step: int, fails: list[str]) -> None:
        nonlocal nonfinite_since
        nonfinite_since = at_step
        tel.write({"event": "state_nonfinite", "wall_clock": time.time(), "epoch": epoch, "step": at_step,
                   "nonfinite_since_step": at_step, "failing": fails})

    def reading(fn, guarded: bool):
        try:
            return fn()
        except Exception:                                    # noqa: BLE001 -- a device that cannot answer: None
            if not guarded:
                raise
            return None

    def abort_record(rule: str, *, quantity, detail: dict, failed=None, exc: BaseException | None = None) -> dict:
        """The run_abort record (DL-04 (a); LB16), written as the telemetry's last row (qat_kd's, unchanged)."""
        img, to = cur["img"], cur["teacher_out"]
        guarded = rule == KD.EXCEPTION_RULE
        input_finite = None if img is None else reading(lambda: KD._finite(img), guarded)
        teacher_finite = None if to is None else reading(lambda: KD._finite(to.logits)
                                                         and (to.feat_s16 is None or KD._finite(to.feat_s16)), guarded)
        cause = ("exception" if rule == KD.EXCEPTION_RULE else
                 KD.abort_cause(rule, input_finite=bool(input_finite), teacher_finite=bool(teacher_finite)))
        try:
            state_ok, state_fails = Q.state_predicate(Q.cpu_state(prepared), kinds, never)
        except Exception:                                    # noqa: BLE001 -- the reading is recorded as missing
            state_ok, state_fails = None, []
        record = {"event": "run_abort", "wall_clock": time.time(), "epoch": epoch, "step": step, "rule": rule,
                  "cause": cause, "nonfinite_quantity": quantity,
                  "detail": {k: TD.record_number(detail.get(k)) for k in
                             ("loss", "grad_norm", "grad_norm_student", "grad_norm_projection")},
                  "input_finite": input_finite, "teacher_finite": teacher_finite,
                  "params_finite": {"student": reading(lambda: all(KD._finite(p) for p in student_params), guarded),
                                    "projection": reading(lambda: all(KD._finite(p) for p in projection_params),
                                                          guarded)},
                  "state_finite": state_ok, "state_failing": state_fails[:5],
                  "state_nonfinite": nonfinite_since is not None or state_ok is False,
                  "nonfinite_since_step": nonfinite_since, "abort_kind": KD.ABORT_KIND}
        if failed:
            record["failed_checks"] = failed
        if exc is not None:
            record.update({"exception_type": type(exc).__name__, "message": str(exc)[:500]})
        tel.write(record)
        return record

    def abort(rule: str, **kw) -> None:
        raise KD.E6KDAborted(abort_record(rule, **kw))

    def train_row(*, lr, epoch_step, loss_v, ce_v, dice_v, sup_v, parts, ramp, norms, clipped, fp, t0) -> dict:
        row = {"event": "train", "wall_clock": time.time(), "epoch": epoch, "step": step, "epoch_step": epoch_step,
               "lr": lr, "loss": loss_v, "ce": ce_v, "dice": dice_v, "sup": sup_v,
               **{t: parts.get(t) for t in KD.TERMS}, "ramp": ramp,
               "grad_norm": norms[0], "grad_norm_student": norms[1], "grad_norm_projection": norms[2],
               "clipped": clipped, "bn_frozen": epoch > Q.BN_FREEZE_EPOCH,
               "observers_enabled": epoch <= Q.OBS_FREEZE_EPOCH, "batch_sha256": fp,
               "state_nonfinite": nonfinite_since is not None, "iter_seconds": time.time() - t0}
        return tel.write(row)

    try:
        for epoch in range(1, Q.EPOCHS + 1):
            if epoch == Q.BN_FREEZE_EPOCH + 1:
                freeze_bn_stats(prepared)
            if epoch == Q.OBS_FREEZE_EPOCH + 1:
                disable_observers(prepared)
            ep = {"clipped": 0, "max_pre_clip_norm": None, "steps": 0}
            for img, mask in train_loader:
                if max_steps is not None and step >= max_steps:
                    break
                step += 1
                ep["steps"] += 1
                t0 = time.time()
                cur["img"] = cur["teacher_out"] = None
                fp = Q.batch_sha256(mask)
                lr = optimizer.param_groups[0]["lr"]
                if step in inj["raise_at"]:                  # smoke only: an exception inside a step
                    raise RuntimeError(f"injected exception at step {step} (smoke)")
                img_d, mask_d = img.to(dev), mask.to(dev)
                if step in inj["input_nan"]:                 # smoke only: a non-finite input
                    img_d = img_d * img_d.new_tensor(float("nan"))
                cur["img"] = img_d
                optimizer.zero_grad(set_to_none=True)
                seen: dict = {}
                probes = ([prepared.register_forward_pre_hook(TD.input_probe(seen, "student")),
                           teacher.teacher.register_forward_pre_hook(TD.input_probe(seen, "teacher"))]
                          if step == 1 else [])
                try:
                    with StudentTaps(prepared) as taps:
                        logits = prepared(img_d)
                        c5, head_logits = taps.require()
                    teacher_out = teacher(img_d)             # one frozen forward per step, after the student's
                finally:
                    for h in probes:
                        h.remove()
                if step in inj["teacher_nan"]:               # smoke only: a non-finite teacher output
                    teacher_out = TT.TeacherOutput(logits=teacher_out.logits * float("nan"),
                                                   feat_s16=teacher_out.feat_s16)
                cur["teacher_out"] = teacher_out
                ramp_by_term = {t: r(step) for t, r in ramps.items()}
                ramp = ramp_by_term[KD.TERMS[0]]            # every term runs the same first-epoch schedule
                ce = criterion.ce(logits, mask_d)
                dice = criterion.dice(logits, mask_d)
                sup = ce + dice
                distill, parts = TD.distillation_losses(
                    stage=KD.E3_STAGE, logits=logits, head_logits=head_logits, c5=c5, mask=mask_d,
                    teacher_out=teacher_out, projection=projection, lambda_logit=weights["logit_kd"],
                    ramp=ramp_by_term, alpha=weights["cwd_feat"], beta=weights["cwd_logit"])
                loss = sup + distill
                if step in inj["loss_nan"]:                  # smoke only: a NaN loss whose gradient is NaN too
                    loss = loss * loss.new_tensor(float("nan"))
                checks: dict = {}
                if step == 1:
                    h, w = img_d.shape[-2:]
                    b = img_d.shape[0]
                    checks.update({
                        "logits_shape": tuple(logits.shape) == (b, Q.NUM_CLASSES, h, w),
                        "c5_channels": c5.shape[1] == STUDENT_C5_CH,
                        "head_logits_os8": tuple(head_logits.shape[-2:]) == (h // 8, w // 8),
                        "c5_os16": tuple(c5.shape[-2:]) == (h // 16, w // 16),
                        "teacher_params_frozen": not teacher.trainable_parameters(),
                        "optimizer_excludes_teacher": not ({id(p) for g in optimizer.param_groups
                                                            for p in g["params"]} & teacher.parameter_ids()),
                        "teacher_same_augmented_input": (seen.get("student") == seen.get("teacher")
                                                         == (img_d.data_ptr(), tuple(img_d.shape),
                                                             tuple(img_d.stride()))),
                        "teacher_logits_shape": tuple(teacher_out.logits.shape) == tuple(head_logits.shape),
                        "teacher_feat_shape": (teacher_out.feat_s16 is not None
                                               and tuple(teacher_out.feat_s16.shape)
                                               == (c5.shape[0], TEACHER_STRIDE16_CH, *c5.shape[-2:])),
                        "projection_fp32_outside_student": (not proj_fq and projection.weight.dtype == torch.float32
                                                            and not (proj_ids & {id(p) for p in student_params})),
                        "has_expected_terms": set(parts) == set(KD.TERMS),
                        "loss_finite": KD._finite(loss) and loss.dim() == 0,
                    })
                    if ramp_iters > 1:
                        checks.update({"distill_ramp_starts_at_zero": all(v == 0.0 for v in ramp_by_term.values()),
                                       "distill_zero_at_step1": float(distill) == 0.0,
                                       "sup_added_unscaled": bool(torch.equal(loss, sup)),
                                       "sup_positive_finite": KD._finite(sup) and float(sup) > 0.0})
                    else:                                   # one step per epoch: step 1 carries the full weights
                        checks.update({"distill_ramp_starts_at_zero": all(v == 1.0 for v in ramp_by_term.values()),
                                       "distill_zero_at_step1": True, "sup_added_unscaled": True,
                                       "sup_positive_finite": True})
                    p0 = student_params[0]
                    before = p0.detach().clone()
                loss_v, ce_v, dice_v, sup_v = (float(loss.item()), float(ce.item()), float(dice.item()),
                                               float(sup.item()))
                row_args = dict(lr=lr, epoch_step=ep["steps"], loss_v=loss_v, ce_v=ce_v, dice_v=dice_v, sup_v=sup_v,
                                parts=parts, ramp=ramp, fp=fp, t0=t0)
                # DL-04 (a): a non-finite total loss stops the run before backward (step 1: step1_checks, R8-1)
                if not math.isfinite(loss_v):
                    train_row(**row_args, norms=(None, None, None), clipped=None)
                    abort(KD.nonfinite_rule(step), quantity="loss", detail={"loss": loss_v})
                loss.backward()
                if step in inj["grad_inf"]:                  # smoke only: a non-finite gradient, finite loss
                    student_params[0].grad.view(-1)[0] = float("inf")
                total, n_s, n_p = KD.grad_norms(student_params, projection_params)
                norms = (float(total.item()), float(n_s.item()), float(n_p.item()))
                # DL-04 (a): a non-finite pre-clip grad_norm stops the run before optimizer.step
                if not math.isfinite(norms[0]):
                    train_row(**row_args, norms=norms, clipped=None)
                    abort(KD.nonfinite_rule(step), quantity="grad_norm",
                          detail={"loss": loss_v, "grad_norm": norms[0], "grad_norm_student": norms[1],
                                  "grad_norm_projection": norms[2]})
                KD.apply_clip(student_params, projection_params, total, clip_norm)
                optimizer.step()
                scheduler.step()
                clipped = bool(norms[0] > clip_norm)
                ep["clipped"] += int(clipped)
                prev = ep["max_pre_clip_norm"]
                ep["max_pre_clip_norm"] = norms[0] if prev is None else max(prev, norms[0])
                row = train_row(**row_args, norms=norms, clipped=clipped)
                if step in inj["state_nan"]:                 # smoke only: a non-finite state, finite loss and norm
                    with torch.no_grad():
                        mod = dict(prepared.named_modules())[never[0]]
                        mod.activation_post_process.min_val.fill_(float("nan"))
                if step == 1:
                    checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)
                    checks["teacher_stayed_frozen"] = all(p.grad is None for p in teacher.teacher.parameters())
                    checks["clip_applied"] = math.isfinite(norms[0])
                    if inj["check_fail"]:                    # smoke only: a failed step-1 check
                        checks["injected_check"] = False
                    failed = [k for k, ok in checks.items() if not ok]
                    tel.write({"event": "step1_checks", "wall_clock": time.time(), "epoch": epoch, "step": step,
                               "checks": checks, "passed": not failed})
                    if failed:
                        abort(KD.STEP1_RULE, quantity=None, failed=failed,
                              detail={"loss": loss_v, "grad_norm": norms[0], "grad_norm_student": norms[1],
                                      "grad_norm_projection": norms[2]})
                if hooks is not None:
                    hooks.after_step(step, prepared, row)
            if max_steps is not None and step >= max_steps and ep["steps"] == 0:
                break

            # the state the epoch checkpoint holds (LB17: a non-finite reading is recorded and the run continues)
            state = Q.cpu_state(prepared)
            state_sha = Q.state_digest(state)
            finite, fails = Q.state_predicate(state, kinds, never)
            if not finite and nonfinite_since is None:
                _mark_nonfinite(step, fails)
            ck_path, ck_sha, ck_bytes = Q.save_epoch_checkpoint(
                out, stage_name=st["name"], source_stage=st["source_stage"], epoch=epoch, step=step,
                state=state, state_sha256=state_sha, seed=seed, clip_norm=clip_norm, run_id=run_id,
                source_sha256=source_meta["sha256"], state_finite=bool(finite), nonfinite_since_step=nonfinite_since)
            # the VAL bracket: observers off for the pass, their snapshot restored after it; the teacher never runs
            if hooks is not None:
                hooks.before_val(epoch, prepared)
            flags = Q.observer_flags(prepared)
            disable_observers(prepared)
            t_val = time.time()
            all_miou, disease_miou, cm, n_val_batches = validate(prepared, val_loader, dev, Q.NUM_CLASSES,
                                                                 max_val_batches)
            del state
            Q.restore_observer_flags(prepared, flags)
            val_seconds = time.time() - t_val
            if Q.state_digest(prepared) != state_sha:
                raise QATStop("val_bracket_changed_state",
                              f"epoch {epoch}: the state after the VAL bracket differs from the checkpoint's")
            if hooks is not None:
                hooks.after_val(epoch, prepared)
            tel.write({"event": "val", "wall_clock": time.time(), "epoch": epoch, "step": step,
                       "kind": "fake_quant", "all_class_miou": all_miou,
                       "disease_only_miou_PROVISIONAL": disease_miou,
                       "val_batches": n_val_batches, "val_total_px": int(cm.sum()),
                       "val_seconds": val_seconds, "observers_disabled_during_val": True,
                       "state_unchanged": True, "selects": False,
                       "state_nonfinite": nonfinite_since is not None})
            nmf = teacher.nmf_stream_state()
            end_row = {"event": "epoch_end", "wall_clock": time.time(), "epoch": epoch, "step": step,
                       "steps_in_epoch": ep["steps"], "checkpoint": f"{Q.EPOCH_DIR}/{ck_path.name}",
                       "checkpoint_sha256": ck_sha, "checkpoint_bytes": ck_bytes, "state_sha256": state_sha,
                       "fake_quant_val_all_class_miou": all_miou,
                       "bn_frozen": epoch > Q.BN_FREEZE_EPOCH, "observers_enabled": epoch <= Q.OBS_FREEZE_EPOCH,
                       "nonfinite_loss_steps": 0, "nonfinite_grad_steps": 0, "step_errors": 0,
                       "clipped_steps": ep["clipped"], "max_pre_clip_norm": ep["max_pre_clip_norm"],
                       "state_finite": bool(finite), "nonfinite_since_step": nonfinite_since,
                       "state_nonfinite": nonfinite_since is not None,
                       "projection_sha256": Q.state_digest(projection),
                       "teacher_nmf_draws": None if nmf is None else nmf["draws"],
                       "run_complete": epoch == Q.EPOCHS and max_steps is None}
            end_row = tel.write(end_row)
            if hooks is not None:
                hooks.after_epoch(epoch, prepared, ck_path, end_row)
            log(f"[am29-a3] epoch {epoch:02d}/{Q.EPOCHS} step {step} fake-quant VAL {all_miou}"
                f"{' (non-finite state)' if nonfinite_since is not None else ''}")
            if max_steps is not None and step >= max_steps:
                break
    except (KD.E6KDAborted, QATRefused):
        raise
    except QATStop as e:
        tel.write({"event": "run_stop", "wall_clock": time.time(), "epoch": epoch, "step": step,
                   "code": e.code, "message": str(e)[:500], "state_nonfinite": nonfinite_since is not None})
        raise
    except BaseException as e:                              # the run_abort record, then re-raised (Q2's form)
        abort_record(KD.EXCEPTION_RULE, quantity=None, detail={}, exc=e)
        raise
    complete = epoch == Q.EPOCHS and max_steps is None
    nmf = teacher.nmf_stream_state()
    return {"run_id": run_id, "stage": st["name"], "arm": arm, "seed": seed, "clip_norm": clip_norm,
            "telemetry": str(tel_path), "steps": step, "epochs_completed": epoch, "complete": complete,
            "nonfinite_since_step": nonfinite_since, "teacher_nmf_draws": None if nmf is None else nmf["draws"]}


# ------------------------------------------------------------------ the parents, the pairing and the teachers
def parent_of_record_error(arm: str, parent: dict, parents: dict = PARENTS_OF_RECORD) -> str | None:
    """The parent of the arm by the sha256 of record (and its run ID, checkpoint name and arm where recorded)."""
    want = parents[ARMS[arm]["parent"]]
    if parent.get("checkpoint_sha256") != want["sha256"]:
        return (f"{arm}'s parent is {ARMS[arm]['parent']} (sha256 {want['sha256']}, {want['record']}); the directory's "
                f"best.json checkpoint has {parent.get('checkpoint_sha256')}")
    for key in ("run_id", "checkpoint"):
        if want.get(key) is not None and parent.get(key) != want[key]:
            return f"{arm}'s parent {key} is {want[key]!r}; the directory has {parent.get(key)!r}"
    if parent.get("seed") != want["seed"] or parent.get("stage") != "E3":
        return f"{arm}'s parent runs stage E3 at seed {want['seed']}; it records {parent.get('stage')!r}, {parent.get('seed')!r}"
    return None


def projection_pairing_error(arm: str, parent: dict, projection_sha256, *, parents: dict = PARENTS_OF_RECORD,
                             others: dict = OTHER_PROJECTIONS_OF_RECORD) -> str | None:
    """AM-29 P1 pairing: an A3 arm's parent and projection.pt belong together by their sha256 (per-seed pairing).

    Refused: a parent that is not the arm's; a projection sha256 of record of another parent or run; a projection sha256
    that is a checkpoint's; and, where the seed's projection sha256 is recorded, any other value. The file's own pairing
    (its checkpoint name and iter against best.json's, stage E3) is qat_kd.projection_of_record's, run after this."""
    if ARMS.get(arm, {}).get("family") != "A3":
        return f"{arm} is not an A3 arm: it takes no projection"
    if not isinstance(projection_sha256, str) or not _HEX64.match(projection_sha256):
        return "--projection-sha256 must be 64 lowercase hex characters"
    key = ARMS[arm]["parent"]
    mine = parents[key]
    bad = parent_of_record_error(arm, parent, parents)
    if bad:
        return bad
    for k, p in parents.items():
        if k != key and p.get("projection_sha256") == projection_sha256:
            return f"projection.pt {projection_sha256} is {k}'s ({p['record']}), not {key}'s"
    if projection_sha256 in others:
        return f"projection.pt {projection_sha256} is {others[projection_sha256]}'s, not {key}'s"
    if any(projection_sha256 == p["sha256"] for p in parents.values()):
        return f"{projection_sha256} is a checkpoint's sha256, not a projection.pt's"
    if mine.get("projection_sha256") is not None and projection_sha256 != mine["projection_sha256"]:
        return f"{key}'s projection.pt of record is {mine['projection_sha256']}, not {projection_sha256}"
    return None


def t_parent_binding(parent: dict, lam: dict, alpha: dict) -> dict:
    """T1-T4: the parent is E3-R2's run (its run_meta names arm E3-R2), at the lambda and alpha of the selections,
    as AM-23's e6x_parent_binding reads it (qat.py:788-808) without AM-23's condition record."""
    rows = Q._run_meta_rows(Path(parent["run_dir"]) / parent["run_meta_file"])
    meta = rows[0] if len(rows) == 1 else {}
    if meta.get("arm") != "E3-R2":
        _refuse("parent_arm_mismatch", f"T1-T4's parent is E3-R2's run; its run_meta names arm {meta.get('arm')!r}")
    if parent.get("lambda_logit") != lam["winner"].get("lambda"):
        _refuse("parent_lambda_mismatch", f"the parent's lambda_logit {parent.get('lambda_logit')!r} != the λ "
                                          f"selection's winner {lam['winner'].get('lambda')!r}")
    if parent.get("alpha_cwd") != alpha["winner"].get("alpha"):
        _refuse("parent_alpha_mismatch", f"the parent's alpha_cwd {parent.get('alpha_cwd')!r} != the α selection's "
                                         f"winner {alpha['winner'].get('alpha')!r}")
    return {"stage": "E3", "arm": "E3-R2", "run_id": parent["run_id"], "seed": parent["seed"],
            "lambda_logit": parent["lambda_logit"], "alpha_cwd": parent["alpha_cwd"],
            "checkpoint": parent["checkpoint"], "checkpoint_sha256": parent["checkpoint_sha256"],
            "lambda_selection": {k: lam[k] for k in ("path", "sha256", "winner")},
            "alpha_selection": {k: alpha[k] for k in ("path", "sha256", "winner")}}


def selections_of_record_error(lam: dict, alpha: dict) -> str | None:
    """The selections' winners are E3's of record (DL-121): lambda 0.5, alpha 25."""
    wl, wa = lam["winner"].get("lambda"), alpha["winner"].get("alpha")
    if wl != E3_LAMBDA_OF_RECORD or wa != E3_ALPHA_OF_RECORD:
        return (f"the selections name lambda {wl!r} and alpha {wa!r}; E3's of record are {E3_LAMBDA_OF_RECORD} and "
                f"{E3_ALPHA_OF_RECORD} (DL-121)")
    return None


def t_teacher_error(arm: str, teacher_sha256, parent: dict, parents: dict = PARENTS_OF_RECORD) -> str | None:
    """T1-T4's teacher is the parent checkpoint itself: --teacher-sha256 must be its sha256 of record and the file's."""
    want = parents[ARMS[arm]["parent"]]["sha256"]
    if not isinstance(teacher_sha256, str) or not _HEX64.match(teacher_sha256):
        return "--teacher-sha256 must be 64 lowercase hex characters"
    if teacher_sha256 != want or parent.get("checkpoint_sha256") != want:
        return (f"the T teacher is a frozen FP32 copy of {ARMS[arm]['parent']} (sha256 {want}); --teacher-sha256 is "
                f"{teacher_sha256}, the parent file {parent.get('checkpoint_sha256')}")
    return None


def flags_error(args) -> tuple[str, str] | None:
    """The flags each family takes: T --teacher-sha256 only; A3 --teacher-ckpt, --teacher-ckpt-sha256 and
    --projection-sha256; LR none of them."""
    fam = ARMS[args.arm]["family"]
    given = {k for k in ("teacher_sha256", "teacher_ckpt", "teacher_ckpt_sha256", "projection_sha256")
             if getattr(args, k, None) is not None}
    allowed = {"T": {"teacher_sha256"}, "A3": {"teacher_ckpt", "teacher_ckpt_sha256", "projection_sha256"},
               "LR": set()}[fam]
    required = allowed
    extra, missing = sorted(given - allowed), sorted(required - given)
    if extra:
        return "flag_not_applicable", f"--arm {args.arm} ({fam}) takes no {['--' + e.replace('_', '-') for e in extra]}"
    if missing:
        return "flag_required", f"--arm {args.arm} ({fam}) requires {['--' + m.replace('_', '-') for m in missing]}"
    return None


# ------------------------------------------------------------------ the launch gates
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AM-29 QAT runner (lane L-AM29, P1): E6's QAT recipe of record with the "
                                            "allow-list's factor, rate and seed")
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--factor", type=float, default=None)
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--source-run-dir", default=None)
    p.add_argument("--expect-source-sha256", default=None)
    p.add_argument("--expect-head", default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--clip-selection", default=None)
    p.add_argument("--clip-selection-sha256", default=None)
    p.add_argument("--lambda-selection", default=None)
    p.add_argument("--lambda-selection-sha256", default=None)
    p.add_argument("--alpha-selection", default=None)
    p.add_argument("--alpha-selection-sha256", default=None)
    p.add_argument("--teacher-sha256", default=None, help="T1-T4: the parent checkpoint's sha256 (the frozen copy)")
    p.add_argument("--teacher-ckpt", default=None, help="A3: the teacher of record (DL-88)")
    p.add_argument("--teacher-ckpt-sha256", default=None, help="A3: the teacher of record's sha256")
    p.add_argument("--projection-sha256", default=None, help="A3: the parent's own projection.pt sha256")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda")
    return p


def real_run_gates(args, *, repo_root: Path = REPO, teacher_builder=None, parents: dict = PARENTS_OF_RECORD,
                   others: dict = OTHER_PROJECTIONS_OF_RECORD) -> dict:
    """Every gate of an AM-29 launch, in order; none writes anything. T and LR follow src/quant/qat.py real_run_gates'
    order, A3 src/quant/qat_kd.py's with its :937, :951, :998 and :1000 replaced. `parents`/`others` and
    `teacher_builder` exist for the smokes' fixtures; a launch passes none."""
    # 1. the flags, the config pins, the head's form
    if not args.real_run:
        _refuse("real_run_flag", "an AM-29 run requires --real-run --confirm-real-run")
    if not args.confirm_real_run:
        _refuse("confirm_real_run_flag", "--real-run requires --confirm-real-run")
    pins = Q.config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    if not args.expect_head or not _HEX40.match(args.expect_head):
        _refuse("expect_head_format", "--expect-head must be the 40-hex commit of the checkout")
    # 2. the allow-list: the arm, its seed, its factor and its rate (replaces qat_kd.py:937 for A3)
    err = arm_values_error(args.arm, seed=args.seed, factor=args.factor, lr=args.lr)
    if err:
        _refuse(*err)
    spec = arm_spec(args.arm)
    stage = am29_stage(args.arm)
    fam = spec["family"]
    err = flags_error(args)
    if err:
        _refuse(*err)
    # 3. loader workers: E6's rule for T and LR, E6-KD's value of record for A3
    if fam == "A3":
        if args.num_workers != KD.preflight_distill().NUM_WORKERS:
            _refuse("num_workers", f"--num-workers {args.num_workers!r}: A3's loader workers are the KD value of "
                                   f"record, {KD.preflight_distill().NUM_WORKERS} (DL-88)")
    elif args.num_workers is None or args.num_workers < 1:
        _refuse("num_workers", "--num-workers must be given and at least 1")
    # 4. the clip of record: clip_selection.json's U4 winner, bound by its sha256 (DL-139)
    if args.clip_selection_sha256 != CLIP_SELECTION_SHA256 or args.grad_clip_norm != CLIP_OF_RECORD:
        _refuse("clip_not_of_record", f"the clip of record is {CLIP_OF_RECORD} from clip_selection.json sha256 "
                                      f"{CLIP_SELECTION_SHA256} (DL-139); got {args.grad_clip_norm!r}, "
                                      f"{args.clip_selection_sha256!r}")
    clip = Q.clip_binding(stage["name"], args.seed, args.grad_clip_norm, u4_pilot=False,
                          clip_selection=args.clip_selection, clip_selection_sha256=args.clip_selection_sha256)
    # 5. the lambda and alpha selections, tracked and unchanged at HEAD; their winners E3's of record
    lam = Q.read_tracked_selection(args.lambda_selection, args.lambda_selection_sha256, Q.LAMBDA_SELECTION_FORMAT,
                                   "lambda", repo_root)
    alpha = Q.read_tracked_selection(args.alpha_selection, args.alpha_selection_sha256, Q.ALPHA_SELECTION_FORMAT,
                                     "alpha", repo_root)
    err = selections_of_record_error(lam, alpha)
    if err:
        _refuse("selections_not_of_record", err)
    # 6. a fresh out dir outside the repository
    err = Q._out_dir_error(args.out_dir)
    if err:
        _refuse("out_dir", err)
    # 7. CUDA available and not yet initialised; TF32 at the torch defaults; QNNPACK present
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        _refuse("cuda_required", f"a real AM-29 run needs a CUDA device (--device {args.device!r}, "
                                 f"available={torch.cuda.is_available()})")
    if torch.cuda.is_initialized():
        _refuse("cuda_initialized_before_seed", "CUDA is already initialised; set_seed must come first")
    tf = Q.tf32_error()
    if tf:
        _refuse("tf32_not_default", tf)
    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        _refuse("backend_unavailable", str(e))
    # 8. A3: the KD image of record and its fingerprint (qat_kd.py gate 9)
    image = None
    if fam == "A3":
        pd = KD.preflight_distill()
        digest = (os.environ.get("PLANTSEG_IMAGE_DIGEST") or "").strip()
        if digest != pd.KD_IMAGE_DIGEST:
            _refuse("image_not_kd", f"PLANTSEG_IMAGE_DIGEST is {digest or None!r}; A3 trains in the KD image "
                                    f"{pd.KD_IMAGE_DIGEST} (DL-87), as E6-KD does")
        rc, fp, fp_out = pd._child_json([REPO / "scripts" / "preflight_distill.py", "_fingerprint"],
                                        "FINGERPRINT_JSON", 600)
        fp_bad = (["the fingerprint child failed: rc=" + str(rc) + " " + fp_out[-200:]] if rc != 0 or fp is None
                  else pd.fingerprint_problems(fp))
        if fp_bad:
            _refuse("fingerprint_mismatch", f"the image differs from the KD fingerprint of record (PL-26): {fp_bad[:6]}")
        image = {"image_digest": digest, "fingerprint": fp}
    # 9. the data root holds TRAIN and VAL only
    from configs.data import DATA, SPLIT_SIZES
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    try:
        iso = assert_trainval_only_root(Path(DATA["root"]), {"train": SPLIT_SIZES["train"],
                                                             "val": SPLIT_SIZES["val"]})
    except TrainValIsolationError as e:
        _refuse("data_root_not_trainval_only", f"[{e.code}] {e}")
    # 10. HEAD is the pin, and AM29_CODE_PATHS (E6-KD's, with AM-29's files) are clean at it
    ident = code_identity()
    if ident["git_head"] != args.expect_head:
        _refuse("expect_head_mismatch", f"the checkout's HEAD is {ident['git_head']!r}, --expect-head "
                                        f"{args.expect_head}")
    if ident["code_clean_at_head"] is not True:
        _refuse("code_not_clean", "the AM-29 code paths are not clean at HEAD")
    # 11. the parent: best.json's checkpoint (AM-19 item 4(a)), the arm's parent of record by sha256
    parent = Q.resolve_parent(stage, args.source_run_dir, args.expect_source_sha256, args.seed)
    err = parent_of_record_error(args.arm, parent, parents)
    if err:
        _refuse("parent_not_of_record", err)
    out = {"clip": clip, "parent": parent, "isolation": iso, "data_root": str(DATA["root"]), "identity": ident,
           "image": image, "lambda_selection": lam, "alpha_selection": alpha}
    # 12. the family's binding, teacher and terms
    if fam == "LR":
        out["e6_binding"] = Q.e6_parent_binding(parent, args.seed, lam, alpha)
    elif fam == "T":
        out["e6_binding"] = None
        out["t_binding"] = t_parent_binding(parent, lam, alpha)
        err = t_teacher_error(args.arm, args.teacher_sha256, parent, parents)
        if err:
            _refuse("teacher_not_of_record", err)
        out["e3_lambda"] = lam["winner"]["lambda"]
    else:
        # replaces qat_kd.py:998 (the alpha-winner parent) and :951/:1000 (the trigger): the sha256 parent of record
        # above, then e6_parent_binding at seed 43/44, whose run-id check is seed 42's only (qat.py:679)
        out["e6_binding"] = Q.e6_parent_binding(parent, args.seed, lam, alpha)
        e3 = KD.e3_record(parent)
        err = projection_pairing_error(args.arm, parent, args.projection_sha256, parents=parents, others=others)
        if err:
            _refuse("projection_unpaired", err)
        projection, projection_rec = KD.projection_of_record(parent, args.projection_sha256)
        teacher, teacher_rec = KD.teacher_of_record(args.teacher_ckpt, args.teacher_ckpt_sha256,
                                                    builder=teacher_builder)
        out.update({"e3": e3, "weights": KD.e6kd_weights(e3["e3_weights"]), "projection": projection,
                    "projection_record": projection_rec, "teacher": teacher, "teacher_record": teacher_rec})
    return out


def result_line(summary: dict) -> str:
    tag = (f"{summary['stage']}, seed {summary['seed']}, clip {summary['clip_norm']}, "
           f"{summary['epochs_completed']}/{Q.EPOCHS} epochs")
    if not summary["complete"]:
        return f"RESULT: AM-29 QAT INCOMPLETE ({tag})"
    if summary["nonfinite_since_step"] is not None:
        return f"RESULT: AM-29 QAT COMPLETE, STATE NON-FINITE first found at step {summary['nonfinite_since_step']} ({tag})"
    return f"RESULT: AM-29 QAT COMPLETE ({tag})"


def main(argv=None, *, teacher_builder=None) -> int:
    R.register()                                         # AM-29's stages, in this process only
    args = build_parser().parse_args(argv)
    try:
        gates = real_run_gates(args, teacher_builder=teacher_builder)
        stage = am29_stage(args.arm)
        model, source_meta = Q.load_source(stage, gates["parent"])
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return Q.EXIT_REFUSED
    except Exception as e:                                 # noqa: BLE001 -- before the run: nothing written
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. Nothing was written.")
        return Q.EXIT_ERROR
    fam = ARMS[args.arm]["family"]
    extra = {"clip_selection": gates["clip"]["clip_selection"], "u4_pilot": False,
             "e6_selections": gates.get("e6_binding"), "data_root": gates["data_root"],
             "isolation_counts": gates["isolation"].get("counts"),
             "expect_head": args.expect_head, "expect_source_sha256": args.expect_source_sha256}
    try:
        if fam == "A3":
            extra.update({"teacher_identity": gates["teacher_record"], "kd_image": gates["image"]})
            summary = run_am29_a3(arm=args.arm, mode="real", model=model, teacher=gates["teacher"],
                                  projection=gates["projection"], weights=gates["weights"], e3=gates["e3"],
                                  source_meta=source_meta, out_dir=args.out_dir, seed=args.seed,
                                  clip_norm=args.grad_clip_norm, clip_source=gates["clip"]["clip_source"],
                                  device=args.device, num_workers=args.num_workers,
                                  projection_record=gates["projection_record"], parent=gates["parent"],
                                  extra_meta=extra)
        else:
            teacher = None
            if fam == "T":
                extra["t_binding"] = gates["t_binding"]
                teacher = SelfTeacher(model, checkpoint_sha256=source_meta["sha256"])
            summary = run_am29_qat(arm=args.arm, mode="real", model=model, source_meta=source_meta,
                                   out_dir=args.out_dir, seed=args.seed, clip_norm=args.grad_clip_norm,
                                   clip_source=gates["clip"]["clip_source"], device=args.device,
                                   num_workers=args.num_workers, lr=args.lr, factor=args.factor,
                                   e3_lambda=gates.get("e3_lambda"), teacher=teacher, parent=gates["parent"],
                                   extra_meta=extra)
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return Q.EXIT_REFUSED
    except QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return Q.EXIT_STOP
    except KD.E6KDAborted as e:
        print(KD.aborted_line(e.record))
        return Q.EXIT_ABORTED
    except Exception as e:                                 # noqa: BLE001 -- after the run_meta row: run_abort
        tel_p = Path(args.out_dir) / Q.TELEMETRY_NAME
        if not (tel_p.is_file() and tel_p.stat().st_size > 0):
            print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. No run_meta row was written: nothing was launched.")
            return Q.EXIT_ERROR
        print(f"RESULT: ABORTED [exception] -- {type(e).__name__}: {e}")
        return Q.EXIT_ABORTED
    print(result_line(summary))
    return Q.EXIT_OK if summary["complete"] else Q.EXIT_ABORTED


__all__ = ["AM29_CODE_PATHS", "AM29_FILES", "ARMS", "CLIP_OF_RECORD", "CLIP_SELECTION_SHA256", "E2_SWITCHES",
           "E3_ALPHA_OF_RECORD", "E3_LAMBDA_OF_RECORD", "LR_AM29", "LR_OF_RECORD", "OTHER_PROJECTIONS_OF_RECORD",
           "PARENTS_OF_RECORD", "SelfTeacher", "T_LOGIT", "am29_stage", "arm_spec", "arm_values_error",
           "build_parser", "code_identity", "flags_error", "main", "optimizer_error", "parent_of_record_error",
           "projection_pairing_error", "real_run_gates", "result_line", "run_am29_a3", "run_am29_qat",
           "selections_of_record_error", "t_logit_kd", "t_parent_binding", "t_teacher_error"]
