#!/usr/bin/env python3
"""Shared distillation training core for E2, E3 and the exploratory arms A, F and G.

Every stage is the SAME student recipe as E1 — MobileNetV3-Large + LR-ASPP, FP32, ImageNet init,
identical preprocessing/augmentation/splits, all-class validation mIoU for checkpoint selection —
with a frozen SegNeXt-B/MSCAN-B teacher added on top and a set of three independently switched
distillation terms (L-AM17B-FG; AM-17 item 7, AM-17b item 1):

  E2  L = L_CE + L_Dice + lambda_logit * L_LogitKD
  E3  L = L_CE + L_Dice + lambda_logit * L_LogitKD + alpha_cwd * L_CWD_feat + 3 * L_CWD_logit
  A   L = L_CE + L_Dice + alpha_cwd * L_CWD_feat + 3 * L_CWD_logit        (AM-17 item 7)
  F   L = L_CE + L_Dice + alpha_cwd * L_CWD_feat                          (AM-17b item 1(a))
  G   L = L_CE + L_Dice + 3 * L_CWD_logit                                 (AM-17b item 1(b))

alpha_cwd is 50 unless `--alpha` picks another value from the AM-16 item 2 grid {25, 50, 100}. A term
that is off is not instantiated: no projection module (cwd_feat), no optimizer param group, no ramp,
no loss call and no telemetry column. The teacher still runs one full forward per step (features and
logits) in every stage, so the M4-KD NMF stream and the per-step teacher cost are the same for all.

All stages start from **E1 init (ImageNet), trained independently** (IMPLEMENTATION_CONTRACT.md stage
table): E2 does NOT resume from a trained E1 checkpoint and E3 does NOT resume from E2. They are
controlled, independently reproducible configurations that happen to share this implementation.
`train_e2.py` / `train_e3.py` are the identifiable per-stage entry points; A, F and G run through
`--stage a|f|g`.

SAFETY MODEL (mirrors train_e1.py, plus the distillation-specific gates):
  * Default / `--dry-run` -> tiny CPU run, random student init, NO download, EXPLICIT MockTeacher,
    checkpoint to a temp dir.
  * Real run requires BOTH `--real-run` AND `--confirm-real-run`, requires CUDA, requires an existing
    `--teacher-ckpt`, requires an explicit `--lambda-logit` for a stage with Logit KD (the contract
    leaves lambda_logit as NEED_TO_CONFIRM, selected by validation sweep — it is never guessed here),
    and requires an explicit positive finite `--grad-clip-norm` (methodology mandates global-norm
    clipping but fixes no numeric threshold, so it stays a recorded experiment-level decision). A
    stage with the feature-map CWD term takes `--alpha` from {25, 50, 100} only, and its `--ckpt-dir`
    name must carry the token `alpha<value>`. An argument for a term the stage does not instantiate
    is refused, not ignored.
  * All gates return before any dataloader or teacher is constructed.
  * Checkpoints are NEVER written inside the repo. The training-only CWD projection and its optimizer
    group are written to `projection.pt` beside the checkpoint, never into it, so `model_state_dict`
    is already the clean E6/E7 deployment student and the checkpoint carries nothing to strip.

No quantization path exists in this file: no QuantStub prepare/convert, no QAT, no PTQ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs.data import DATA                                      # noqa: E402
from configs.distill import (DISTILL, DISTILL_STAGES, DISTILL_TERMS,  # noqa: E402
                             LOGIT_KD_SEMANTICS, LOGIT_KD_SEMANTICS_SUPERSEDED)
from configs.e1_student import E1_STUDENT                          # noqa: E402
from src.data import NUM_CLASSES, build_dataloader                 # noqa: E402
from src.data.isolation import TrainValIsolationError, assert_trainval_only_root  # noqa: E402
from src.distill.cwd_projection import (PROJECTION_FILE, PROJECTION_GROUP,  # noqa: E402
                                        build_cwd_projection_for, save_projection,
                                        split_optimizer_state)
from src.distill.export import assert_clean_student_state          # noqa: E402
from src.distill.features import StudentTaps                       # noqa: E402
from src.distill.nmf_stream import M4_NMF_SEED                     # noqa: E402
from src.distill.teacher import (FrozenTeacher, MockTeacher,       # noqa: E402
                                 TeacherCheckpointMissing, load_frozen_teacher,
                                 require_teacher_checkpoint)
from src.models.student import build_student                       # noqa: E402
from src.seeds import set_seed                                     # noqa: E402
from src.training.losses import (CombinedCEDiceLoss, cwd_channelwise_kl,  # noqa: E402
                                 downsample_validity, logit_kd_kl)
# Reuse the audited E1 mechanics verbatim rather than re-implementing them.
from src.training.train_e1 import (CLASS_WEIGHTS_JSON, build_scheduler, cycle,  # noqa: E402
                                   load_ce_weights, per_class_iou, resolve_ckpt_dir,
                                   total_grad_norm, validate, write_best_pointer,
                                   _assert_outside_repo, _atomic_save, _git_provenance,
                                   _image_digest, _jsonl)

IGNORE_INDEX = DATA["ignore_index"]            # 255
T_LOGIT = DISTILL["logit_kd"]["T_logit"]       # 4
T_CWD = DISTILL["cwd"]["T_cwd"]                # 4
ALPHA_CWD_FEAT = DISTILL["cwd"]["alpha_cwd_feature_map"]   # 50: the default and AM-16's fallback
ALPHA_GRID = tuple(DISTILL["cwd"]["alpha_cwd_grid"])       # (25, 50, 100): AM-16 item 2
BETA_CWD_LOGIT = DISTILL["cwd"]["beta_cwd_logit_map"]      # 3
CWD_C_FEAT = DISTILL["cwd"]["C"]               # 320 (MSCAN-B stride-16 Stage-3)
LAMBDA_SWEEP = DISTILL["logit_kd"]["lambda_logit_sweep_grid"]
TERMS = DISTILL_TERMS                          # logit_kd, cwd_feat, cwd_logit: the loss addition order
WEIGHT_KEYS = {"logit_kd": "lambda_logit", "cwd_feat": "alpha_cwd", "cwd_logit": "beta_cwd"}
# The thesis teacher config (B62): its IsolatedNMFLightHamHead is what lets the frozen teacher honour M4-KD.
DEFAULT_TEACHER_CONFIG = (REPO / "configs" / "teacher"
                          / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py")


class StageConfigError(ValueError):
    """A distillation stage config that breaks the per-term switch rules (L-AM17B-FG)."""


def validate_stage_config(key: str, cfg) -> dict:
    """One stage config: a display name plus exactly the three term booleans.

    Refused: the pre-lane combined `cwd` key (it switched the projection and both channel-wise terms
    together, DL-35 G1); a missing, extra or non-bool term; all three terms off (that is E1, which is
    train_e1.py, not a distillation stage).
    """
    if not isinstance(cfg, Mapping):
        raise StageConfigError(f"stage {key!r}: config must be a mapping, got {type(cfg).__name__}")
    if "cwd" in cfg:
        raise StageConfigError(f"stage {key!r}: the combined 'cwd' key is refused; switch 'cwd_feat' "
                               "and 'cwd_logit' separately (L-AM17B-FG, DL-35 G1)")
    unknown = sorted(set(cfg) - {"name", *TERMS})
    missing = [t for t in TERMS if t not in cfg]
    if unknown or missing:
        raise StageConfigError(f"stage {key!r}: unknown keys {unknown}, missing terms {missing}; a "
                               f"stage is 'name' plus {list(TERMS)}")
    not_bool = [t for t in TERMS if not isinstance(cfg[t], bool)]
    if not_bool:
        raise StageConfigError(f"stage {key!r}: terms {not_bool} must be True or False")
    if not any(cfg[t] for t in TERMS):
        raise StageConfigError(f"stage {key!r}: every term is off — that is E1, which is train_e1.py, "
                               "not a distillation stage")
    if not isinstance(cfg.get("name"), str) or not cfg["name"]:
        raise StageConfigError(f"stage {key!r}: 'name' must be a non-empty string")
    return {"name": cfg["name"], **{t: cfg[t] for t in TERMS}}


def instantiated_terms(stage) -> tuple[str, ...]:
    """The stage's switched-on terms, in the addition order of the total loss."""
    return tuple(t for t in TERMS if stage[t])


def stage_objective(stage) -> str:
    parts = ["L_CE + L_Dice"]
    if stage["logit_kd"]:
        parts.append("lambda_logit*L_LogitKD")
    if stage["cwd_feat"]:
        parts.append("alpha_cwd*L_CWD_feat")
    if stage["cwd_logit"]:
        parts.append(f"{BETA_CWD_LOGIT}*L_CWD_logit")
    return " + ".join(parts)


def _build_stage_table(table) -> dict[str, dict]:
    stages = {}
    for key, cfg in table.items():
        stage = validate_stage_config(key, cfg)
        stages[key] = {**stage, "objective": stage_objective(stage)}
    return stages


STAGES: dict[str, dict] = _build_stage_table(DISTILL_STAGES)


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


def lambda_semantics_gate_error(declared, override: bool) -> str | None:
    """Validate a DECLARED lambda_logit semantics tag against the one this code implements (B32c-2).

    lambda_logit weights the Logit-KD term, so its numeric value only means anything relative to the
    SPATIAL GRID that term is computed on. B32/F8 moved that grid from an upsampled 512x512 to the
    head's native OS8 64x64 and the term's magnitude changed by ~1.95x — about one step of the
    preregistered geometric grid {0.25, 0.5, 1, 2, 4}. A lambda read out of a Ch4 table months from
    now and passed to a re-run under different semantics is silently wrong, not loudly wrong.

    Declaring the tag is OPTIONAL: the sweep runs PRODUCE the tag rather than consume it, so
    requiring it there would add friction at the point of lowest risk. But once declared it is
    checked, and a mismatch refuses unless explicitly overridden.
    """
    if declared is None or override:
        return None
    if declared == LOGIT_KD_SEMANTICS:
        return None
    superseded = " (a SUPERSEDED pre-B32 tag)" if declared in LOGIT_KD_SEMANTICS_SUPERSEDED else ""
    return (f"--lambda-semantics {declared!r}{superseded} does not match the semantics this code "
            f"implements, {LOGIT_KD_SEMANTICS!r}. lambda_logit is not transferable across Logit-KD "
            f"resolutions: B32/F8 moved the KL from an upsampled 512x512 grid to the head's native "
            f"OS8 64x64 and the term's magnitude changed by ~1.95x (MEASURED, synthetic-teacher "
            f"UPPER BOUND), which is about one step of the {LAMBDA_SWEEP} grid. Re-select lambda "
            f"under the current semantics, or pass --allow-semantics-mismatch to proceed anyway "
            f"(the override is recorded in the checkpoint and the run_meta file).")


def alpha_token(alpha: float) -> str:
    """The checkpoint-dir token that carries alpha (L-AM16-ALPHA): alpha25, alpha50, alpha100."""
    return f"alpha{float(alpha):g}"


def alpha_gate_error(alpha, *, allow_offgrid: bool, mode: str) -> str | None:
    """Validate `--alpha` (None = the default, 50). Returns an error string, or None if OK.

    AM-16 item 2 registers the grid {25, 50, 100}. `--allow-offgrid` admits another positive finite
    value in dry runs only (tests); a real run refuses it.
    """
    if allow_offgrid and mode == "real":
        return ("--allow-offgrid is for tests only; a real run takes alpha from the AM-16 item 2 "
                f"grid {ALPHA_GRID}")
    if alpha is None:
        return None
    if not math.isfinite(alpha) or alpha <= 0.0:
        return f"--alpha must be a positive finite value, got {alpha!r}"
    if alpha not in ALPHA_GRID and not allow_offgrid:
        return (f"--alpha {alpha:g} is not in the AM-16 item 2 grid {ALPHA_GRID} (default "
                f"{ALPHA_CWD_FEAT}); --allow-offgrid admits other values in dry runs only")
    return None


def ckpt_dir_alpha_error(ckpt_dir, alpha: float) -> str | None:
    """A run with the feature-map term names its checkpoint dir after its alpha (L-AM16-ALPHA): the
    last path component must contain the token `alpha<value>`, e.g. /workspace/e3_s42_alpha50, with
    no digit following it. Returns an error string, or None if OK."""
    tok = alpha_token(alpha)
    name = Path(ckpt_dir).name
    if re.search(rf"(?<![0-9A-Za-z]){re.escape(tok)}(?![0-9]|\.[0-9])", name, flags=re.IGNORECASE):
        return None
    return (f"--ckpt-dir {ckpt_dir} does not carry the alpha token {tok!r} in its last path "
            f"component; name it after the run's alpha, e.g. .../e3_s42_{tok}")


def term_args_error(stage, *, lambda_logit, lambda_semantics, allow_semantics_mismatch, alpha,
                    allow_offgrid, mode) -> str | None:
    """Arguments that belong to a term the stage does not instantiate are refused, not ignored;
    `--alpha` is checked against the AM-16 grid. Returns an error string, or None if OK."""
    on = ", ".join(instantiated_terms(stage))
    if not stage["logit_kd"]:
        for flag, given in (("--lambda-logit", lambda_logit is not None),
                            ("--lambda-semantics", lambda_semantics is not None),
                            ("--allow-semantics-mismatch", bool(allow_semantics_mismatch))):
            if given:
                return (f"{flag} was given, but stage {stage['name']} has no Logit-KD term "
                        f"(terms on: {on})")
    if not stage["cwd_feat"]:
        for flag, given in (("--alpha", alpha is not None), ("--allow-offgrid", bool(allow_offgrid))):
            if given:
                return (f"{flag} was given, but stage {stage['name']} has no feature-map CWD term "
                        f"(terms on: {on})")
        return None
    return alpha_gate_error(alpha, allow_offgrid=allow_offgrid, mode=mode)


def resolve_stage(stage: str) -> dict:
    """Map a stage key to its term switches: e2 = Logit KD; e3 = Logit KD + both CWD terms; a = both
    CWD terms; f = the feature-map CWD term; g = the logit-map CWD term."""
    key = str(stage).lower()
    if key not in STAGES:
        raise ValueError(f"unknown distillation stage {stage!r}; expected one of {sorted(STAGES)}")
    return {"key": key, **STAGES[key]}


def term_weights(stage, *, lambda_logit, alpha, beta=BETA_CWD_LOGIT) -> dict[str, float]:
    """Target weight of each INSTANTIATED term (the value its ramp reaches); an off term has none."""
    weights = {}
    if stage["logit_kd"]:
        weights["logit_kd"] = lambda_logit
    if stage["cwd_feat"]:
        weights["cwd_feat"] = alpha
    if stage["cwd_logit"]:
        weights["cwd_logit"] = beta
    return weights


def build_optimizer(student, projection, teacher: FrozenTeacher | None = None):
    """SGD over the student, plus the projection's own param group when cwd_feat is instantiated.

    Returns (optimizer, trainable). `trainable` keeps the pre-lane order (student, then projection)
    for the gradient-norm telemetry and clipping. SGD updates every parameter independently, so the
    two groups are numerically the pre-lane single group; the projection group exists only when the
    projection does (L-AM17B-FG). The optimizer is never built from the teacher.
    """
    student_params = list(student.parameters())
    groups = [{"params": student_params, "name": "student"}]
    trainable = list(student_params)
    if projection is not None:
        projection_params = list(projection.parameters())
        groups.append({"params": projection_params, "name": PROJECTION_GROUP})
        trainable += projection_params
    if teacher is not None:
        overlap = {id(p) for p in trainable} & teacher.parameter_ids()
        if overlap:
            raise RuntimeError(f"{len(overlap)} teacher parameter(s) reached the optimizer parameter "
                               "list — the teacher must never be optimized")
    optimizer = torch.optim.SGD(groups, lr=E1_STUDENT["learning_rate"],
                                momentum=E1_STUDENT["momentum"],
                                weight_decay=E1_STUDENT["weight_decay"])
    return optimizer, trainable


def save_distill_checkpoint(ckpt_dir: Path, stage: dict, student, projection, optimizer, scheduler,
                            sched_name: str, it: int, best_miou: float, teacher_provenance,
                            semantics_declared=None, semantics_override: bool = False, *,
                            lambda_logit=None, alpha=None) -> str:
    """Write a distillation checkpoint, and projection.pt beside it when cwd_feat is instantiated.

    `model_state_dict` holds the student ONLY and `optimizer_state_dict` only the student's param
    group: the training-only projection and its optimizer group go to projection.pt (L-AM17B-FG), so
    the checkpoint E6/E7 consume carries nothing to strip (contract B3/B4). The split is asserted
    before writing. The payload records the stage's term switches and its instantiated weights.
    """
    path = _assert_outside_repo(Path(ckpt_dir)) / f"{stage['key']}_student_best_iter{it}.pt"
    student_state = student.state_dict()
    assert_clean_student_state(student_state)
    student_opt, projection_opt = split_optimizer_state(optimizer.state_dict())
    if (projection is None) != (not projection_opt["param_groups"]):
        raise RuntimeError("the optimizer's projection group does not match the projection module")
    weights = term_weights(stage, lambda_logit=lambda_logit, alpha=alpha)
    payload = {
        "stage": stage["name"],
        "iter": it,
        "model_state_dict": student_state,
        "optimizer_state_dict": student_opt,
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler": sched_name,
        "best_val_miou_all_class": best_miou,
        "num_classes": NUM_CLASSES,
        "teacher_provenance": None if teacher_provenance is None else teacher_provenance.as_dict(),
        "terms": {t: bool(stage[t]) for t in TERMS},
        **{WEIGHT_KEYS[t]: w for t, w in weights.items()},
    }
    if stage["logit_kd"]:
        # B32c-2: the semantics lambda_logit was measured under travels WITH the artifact, so a
        # mismatched run stays identifiable from its checkpoint alone once the terminal is gone.
        payload.update({"logit_kd_semantics": LOGIT_KD_SEMANTICS,
                        "logit_kd_semantics_declared": semantics_declared,
                        "logit_kd_semantics_override_used": bool(semantics_override)})
    _atomic_save(payload, path)          # train_e1's same-directory .tmp + os.replace (item 7)
    if projection is not None:
        save_projection(Path(ckpt_dir) / PROJECTION_FILE, projection, projection_opt,
                        stage=stage["name"], it=it, checkpoint=path.name)
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

    Applied, through one `TermRamp` per instantiated term, to every distillation term (Logit KD and
    both CWD terms); the supervised CE+Dice term is never ramped.
    """
    if ramp_iters <= 1:
        return 1.0
    if it <= 1:
        return 0.0
    if it >= ramp_iters:
        return 1.0
    return float(it - 1) / float(ramp_iters - 1)


class TermRamp:
    """The first-epoch linear ramp of ONE instantiated distillation term (F6; contract B2).

    One object per instantiated term and none for a term that is off. Every instance applies
    `distill_ramp`, so a term's weight at iteration `it` is `distill_ramp(it, ramp_iters) * weight`,
    exactly the pre-lane value.
    """

    __slots__ = ("term", "ramp_iters")

    def __init__(self, term: str, ramp_iters: int):
        self.term = term
        self.ramp_iters = int(ramp_iters)

    def __call__(self, it: int) -> float:
        return distill_ramp(it, self.ramp_iters)


def build_term_ramps(stage, ramp_iters: int) -> dict[str, TermRamp]:
    """One ramp object per instantiated term, in the addition order; none for an off term."""
    return {t: TermRamp(t, ramp_iters) for t in instantiated_terms(stage)}


def require_same_grid(what: str, tensor, target_what: str, target) -> None:
    """A teacher map must already be on the student's grid: it is never resampled (L-KD-HARDEN item 8)."""
    if tuple(tensor.shape[-2:]) != tuple(target.shape[-2:]):
        raise RuntimeError(f"{what} is on a {tuple(tensor.shape[-2:])} grid but {target_what} is on "
                           f"{tuple(target.shape[-2:])}: a teacher map is never resampled onto the "
                           "student's grid; check the teacher config (B32/F8)")


def input_probe(seen: dict, name: str):
    """A forward pre-hook recording the (data_ptr, shape) of the tensor a model receives (item 1c)."""
    def hook(module, inputs):
        seen[name] = (inputs[0].data_ptr(), tuple(inputs[0].shape))
    return hook


def distillation_losses(*, stage: dict, logits, head_logits, c5, mask, teacher_out, projection,
                        lambda_logit: float | None, ramp=1.0, alpha: float | None = None,
                        beta: float | None = None):
    """Compute the stage's INSTANTIATED distillation terms. Returns (total_distill, parts dict).

    Teacher tensors arrive detached from `FrozenTeacher`, so nothing here can push gradient into the
    teacher; the student and (cwd_feat) the projection are the only things on the graph. A term that
    is off is never computed (no loss call) and has no entry in `parts`. `ramp` is either one factor
    for every instantiated term or a per-term mapping from the per-term ramp objects. Terms are added
    in the pre-lane order logit_kd -> cwd_feat -> cwd_logit, each as `ramp * weight * loss`, so a
    stage's total is bitwise the pre-lane total restricted to its terms, and equals the three-term
    total with the missing terms weighted 0.0. `parts` reports the UNRAMPED loss values so the log
    stays diagnostic. alpha and beta default to the config values (50 and 3).
    """
    alpha = ALPHA_CWD_FEAT if alpha is None else alpha
    beta = BETA_CWD_LOGIT if beta is None else beta

    def factor(term: str) -> float:
        return ramp[term] if isinstance(ramp, Mapping) else ramp

    parts: dict[str, float] = {}
    total = logits.new_zeros(())

    t_logits = teacher_out.logits
    if t_logits.shape[1] != logits.shape[1]:
        raise ValueError(f"teacher logits have {t_logits.shape[1]} classes, student has "
                         f"{logits.shape[1]}")
    if projection is not None and not stage["cwd_feat"]:
        raise RuntimeError(f"stage {stage['name']} has no feature-map CWD term, but a projection was "
                           "passed — a term that is off is not instantiated")

    if stage["logit_kd"] or stage["cwd_logit"]:
        # B32/F8: both logit-map terms are computed on the head's NATIVE OS8 map, not on upsampled
        # copies. The student's `head_logits` and the teacher's LightHamHead output are both 64x64 for
        # a 512x512 input, so neither side is resampled and no interpolation artifact enters the soft
        # targets; a teacher map on any other grid is refused, never resampled (L-KD-HARDEN item 8).
        # The validity mask is downsampled to that same grid once and shared.
        require_same_grid("the teacher logits", t_logits, "the student head logits", head_logits)
        valid_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE_INDEX)
        t_logits_os8 = t_logits

    # --- Logit KD (E2, E3; unchanged between them per contract B3) ---
    if stage["logit_kd"]:
        if lambda_logit is None:
            raise ValueError(f"stage {stage['name']} instantiates Logit KD but lambda_logit is None")
        l_kd = logit_kd_kl(head_logits, t_logits_os8, valid_os8, T=T_LOGIT)
        total = total + factor("logit_kd") * lambda_logit * l_kd
        parts["logit_kd"] = float(l_kd.detach())

    # --- CWD feature term: student C5 (160ch) -> 1x1 projection -> teacher stride-16 (320ch) ---
    if stage["cwd_feat"]:
        if teacher_out.feat_s16 is None:
            raise RuntimeError(f"stage {stage['name']} requires the teacher stride-16 Stage-3 feature, "
                               "but the teacher returned none — check the teacher feature hook")
        if projection is None:
            raise RuntimeError(f"stage {stage['name']} instantiates the feature-map CWD term but no "
                               "projection was built")
        t_feat = teacher_out.feat_s16
        require_same_grid("the teacher stride-16 feature", t_feat, "the student C5 map", c5)
        s_feat = projection(c5)
        valid_s16 = downsample_validity(mask, s_feat.shape[-2:], ignore_index=IGNORE_INDEX)
        l_feat = cwd_channelwise_kl(s_feat, t_feat, valid_s16, T=T_CWD, channels_norm=CWD_C_FEAT)
        total = total + factor("cwd_feat") * alpha * l_feat
        parts["cwd_feat"] = float(l_feat.detach())

    # --- CWD logit term: on the head's native OS8 logit map (no projection needed, same C) ---
    # This term was ALREADY at OS8 before B32 (B32-5: REFUTED, unchanged). It shares `t_logits_os8`
    # and `valid_os8` with the Logit-KD term — same grid, same mask, one min-pool instead of two.
    # channels_norm defaults to this map's own channel count (116 classes) — 320 is the FEATURE-map
    # normalisation only and must never be hard-coded here.
    if stage["cwd_logit"]:
        l_logit_map = cwd_channelwise_kl(head_logits, t_logits_os8, valid_os8, T=T_CWD)
        total = total + factor("cwd_logit") * beta * l_logit_map
        parts["cwd_logit"] = float(l_logit_map.detach())
    return total, parts


def tf32_state() -> dict:
    """The TF32 state a run executes under (run_meta, item 5). Read only: no trainer sets these."""
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(*, stage: dict, mode: str, device: str, pretrained, teacher: FrozenTeacher,
        lambda_logit: float | None, batch_size: int, max_iters: int, val_interval: int,
        max_val_batches: int | None, num_workers: int, ckpt_dir_arg: str | None,
        semantics_declared=None, semantics_override: bool = False,
        grad_clip_norm: float | None, log_every: int, seed: int,
        alpha: float | None = None, alpha_offgrid: bool = False) -> int:
    wall_clock_start = time.time()
    set_seed(seed)
    dev = torch.device(device)
    terms = instantiated_terms(stage)
    if stage["logit_kd"] != (lambda_logit is not None):
        raise ValueError(f"stage {stage['name']}: lambda_logit={lambda_logit!r} with logit_kd="
                         f"{stage['logit_kd']} — a weight exists exactly when its term does")
    if not stage["cwd_feat"] and alpha is not None:
        raise ValueError(f"stage {stage['name']} has no feature-map CWD term; alpha must be None")
    if stage["cwd_feat"]:
        alpha = float(ALPHA_CWD_FEAT if alpha is None else alpha)
    print(f"[stage] {stage['name']} | objective: {stage['objective']} | terms: "
          + " ".join(f"{t}={'on' if stage[t] else 'off'}" for t in TERMS))
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
    # M4-KD (B61 §4): one private NMF stream, seeded 42 ONCE for the whole run and advancing across
    # teacher calls. Only the NMF basis draw consumes it, so the student's CPU RNG is never perturbed;
    # every stage sees the same NMF sequence because every stage makes one full teacher call per step.
    teacher_nmf = teacher.begin_nmf_stream("M4-KD", M4_NMF_SEED)
    if mode == "real" and teacher_nmf is None:
        raise RuntimeError("M4-KD: the real teacher exposes no isolated NMF stream; build it from the "
                           "thesis teacher config (IsolatedNMFLightHamHead)")
    print(f"[teacher] NMF control: {teacher_nmf if teacher_nmf is not None else 'none (MockTeacher has no NMF)'}")
    n_teacher_params = sum(p.numel() for p in teacher.teacher.parameters())
    print(f"[teacher] frozen=True eval=True params={n_teacher_params:,} trainable_params=0 "
          f"mock={is_mock} "
          f"provenance={None if teacher.provenance is None else teacher.provenance.as_dict()}")

    # --- training-only CWD projection: instantiated only with the cwd_feat term (L-AM17B-FG) ---
    projection = build_cwd_projection_for(stage)
    if projection is not None:
        projection = projection.to(dev)
        projection.train()
        print(f"[cwd] projection 1x1 conv {DISTILL['cwd']['projection_head']['maps']} "
              f"params={sum(p.numel() for p in projection.parameters()):,} "
              f"(TRAINING-ONLY; written with its optimizer group to {PROJECTION_FILE}, never into the "
              f"student checkpoint)")

    # Item 6: E1's TRAIN-loader call verbatim (train_e1.run), so at equal seed and num_workers a KD
    # stage consumes E1's realized sample stream for the whole run, not only its first epoch.
    train_loader = build_dataloader("train", batch_size, num_workers=num_workers,
                                    persistent_workers=num_workers > 0, seed=seed)
    val_loader = build_dataloader("val", batch_size, num_workers=num_workers, seed=seed)
    print(f"[data] train_index={len(train_loader.dataset)} val_index={len(val_loader.dataset)} "
          "(train+val only; the TEST split is never built here)")

    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=IGNORE_INDEX).to(dev)
    print("[loss] sup=CombinedCEDiceLoss"
          + (f" | lambda_logit={lambda_logit} T_logit={T_LOGIT}" if stage["logit_kd"] else "")
          + (f" | alpha_cwd={alpha:g} T_cwd={T_CWD} C_feat={CWD_C_FEAT}" if stage["cwd_feat"] else "")
          + (f" | beta_cwd={BETA_CWD_LOGIT} T_cwd={T_CWD}" if stage["cwd_logit"] else ""))

    # Optimizer is built EXPLICITLY from the student (+ the projection's group) — never from the teacher.
    optimizer, trainable = build_optimizer(student, projection, teacher)
    horizon = E1_STUDENT["iterations"]
    scheduler, sched_name = build_scheduler(optimizer, horizon, E1_STUDENT["lr_power"])
    print(f"[opt] SGD lr={E1_STUDENT['learning_rate']} scheduler={sched_name} "
          f"(total_iters={horizon}, power={E1_STUDENT['lr_power']}) trainable_tensors={len(trainable)} "
          f"param_groups={[g['name'] for g in optimizer.param_groups]} teacher_params_in_optimizer=0")

    # First-epoch distillation ramp: one epoch expressed in iterations, from the real loader. One ramp
    # object per instantiated term (F6); a term that is off has none.
    ramp_iters = len(train_loader)
    ramps = build_term_ramps(stage, ramp_iters)
    print(f"[ramp] {E1_STUDENT['distill_weight_ramp']} -> ramp_iters={ramp_iters} "
          f"(factor = clamp((it-1)/(ramp_iters-1), 0, 1)) for each instantiated term: "
          f"{', '.join(ramps)}; targets "
          f"{term_weights(stage, lambda_logit=lambda_logit, alpha=alpha)}")
    if grad_clip_norm is None:
        clip_msg = ("DISABLED — no numeric max_norm is specified anywhere authoritative "
                    "(IMPLEMENTATION_CONTRACT D-A/D2, open_questions D2); pass "
                    "--grad-clip-norm <value> to enable the global-norm path")
    else:
        clip_msg = f"global-norm, max_norm={grad_clip_norm}, applied every iteration"
    print(f"[grad-clip] {clip_msg}")

    if ckpt_dir_arg is None and stage["cwd_feat"]:
        # L-AM16-ALPHA: the checkpoint dir of a run with the feature-map term carries its alpha.
        ckpt_dir_arg = tempfile.mkdtemp(prefix=f"{stage['key']}_{alpha_token(alpha)}_dryrun_")
    ckpt_dir = resolve_ckpt_dir(ckpt_dir_arg)
    if mode == "real" and stage["cwd_feat"]:
        dir_error = ckpt_dir_alpha_error(ckpt_dir, alpha)
        if dir_error is not None:
            raise RuntimeError(dir_error)
    print(f"[ckpt] dir={ckpt_dir} (verified OUTSIDE repo)")

    # B32c-2: one run_meta line recording the semantics lambda_logit is being used under, so a
    # mismatched run stays identifiable from its artifacts alone long after the terminal is gone.
    # L-AM17B-FG: the term switches and projection size, and the weight, temperature and grid of each
    # INSTANTIATED term only. L-KD-HARDEN item 5: then E1's run_meta keys and the KD carriers.
    meta_path = _assert_outside_repo(Path(ckpt_dir)) / f"{stage['key']}_run_meta.jsonl"
    # Per-iteration telemetry beside the checkpoints, never inside the repo (append mode, like E1's).
    telemetry_path = meta_path.with_name(f"{stage['key']}_telemetry.jsonl")
    meta = {"event": "run_meta", "stage": stage["name"], "mode": mode, "seed": seed,
            "terms": {t: bool(stage[t]) for t in TERMS},
            "projection_params": (0 if projection is None
                                  else sum(p.numel() for p in projection.parameters()))}
    if stage["logit_kd"]:
        meta.update({"lambda_logit": lambda_logit,
                     "logit_kd_semantics": LOGIT_KD_SEMANTICS,
                     "logit_kd_semantics_declared": semantics_declared,
                     "logit_kd_semantics_override_used": bool(semantics_override),
                     "logit_kd_grid": "os8 64x64 (head-native, no upsample)",
                     "T_logit": T_LOGIT, "lambda_sweep_grid": list(LAMBDA_SWEEP)})
    if stage["cwd_feat"]:
        meta.update({"alpha_cwd": alpha, "alpha_offgrid": bool(alpha_offgrid),
                     "alpha_grid": list(ALPHA_GRID), "cwd_feat_grid": "stride-16 32x32",
                     "T_cwd": T_CWD, "cwd_C": CWD_C_FEAT})
    if stage["cwd_logit"]:
        meta.update({"beta_cwd": BETA_CWD_LOGIT, "T_cwd": T_CWD,
                     "cwd_logit_grid": "os8 64x64 (validity mask shared with Logit-KD when on)"})
    meta.update({"supervised_grid": "full 512x512", "batch_size": batch_size,
                 "max_iters": max_iters, "num_classes": NUM_CLASSES, "teacher_nmf": teacher_nmf})
    # L-KD-HARDEN item 5: E1's run_meta keys, through train_e1's provenance helpers, plus
    # persistent_workers (E1's TRAIN-loader argument); then the KD carriers (L-CKPT-GUARD adds further
    # provenance fields later).
    git_head, git_head_source = _git_provenance()
    meta.update({"wall_clock": time.time(), "git_head": git_head, "git_head_source": git_head_source,
                 "image_digest": _image_digest(), "torch": torch.__version__, "numpy": np.__version__,
                 "device": str(dev), "cuda_available": torch.cuda.is_available(),
                 "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "num_workers": num_workers, "persistent_workers": num_workers > 0,
                 "val_interval": val_interval, "max_val_batches": max_val_batches,
                 "learning_rate": E1_STUDENT["learning_rate"], "momentum": E1_STUDENT["momentum"],
                 "weight_decay": E1_STUDENT["weight_decay"], "lr_power": E1_STUDENT["lr_power"],
                 "poly_horizon": horizon, "grad_clip_norm": grad_clip_norm,
                 "used_pretrained": student.used_pretrained,
                 "params": sum(p.numel() for p in student.parameters()), "ignore_index": IGNORE_INDEX,
                 "ramp_iters": ramp_iters, "class_weights_sha256": sha256_file(CLASS_WEIGHTS_JSON),
                 "tf32": tf32_state(),
                 "teacher_provenance": (None if teacher.provenance is None
                                        else teacher.provenance.as_dict()),
                 "teacher_mock": bool(is_mock)})
    with open(meta_path, "a", encoding="utf-8") as _f:
        _f.write(json.dumps(meta) + "\n")
    if stage["logit_kd"]:
        print(f"[semantics] logit_kd={LOGIT_KD_SEMANTICS} declared={semantics_declared} "
              f"override={bool(semantics_override)} -> {meta_path.name}")
    if semantics_override:
        print("[semantics] *** OVERRIDE ACTIVE: --allow-semantics-mismatch was used. This run's "
              "lambda_logit was selected under DIFFERENT Logit-KD semantics and its results are "
              "NOT comparable to runs without the override. Recorded in the checkpoint payload and "
              f"in {meta_path.name}. ***")

    checks: dict[str, bool] = {}
    best_miou = float("-inf")
    best_ckpt = None
    lr_trace: list[float] = []
    train_iter = cycle(train_loader)
    t_prev = time.time()
    train_seconds = 0.0       # top of each iteration to its train row: no setup, validation or checkpoint

    for it in range(1, max_iters + 1):
        t_iter = time.time()
        img, mask = next(train_iter)
        img, mask = img.to(dev), mask.to(dev)

        # ONE already-augmented tensor is shared by student and teacher — the teacher must see the
        # exact same augmented input as the student for every training sample.
        model_input = img

        optimizer.zero_grad(set_to_none=True)
        # Item 1c: at step 1, pre-hooks on the student and on the wrapped teacher record the tensor each
        # model actually receives, so the check compares the inputs, not two names for one variable.
        seen: dict = {}
        hooks = ([student.register_forward_pre_hook(input_probe(seen, "student")),
                  teacher.teacher.register_forward_pre_hook(input_probe(seen, "teacher"))]
                 if it == 1 else [])
        try:
            with StudentTaps(student) as taps:
                logits = student(model_input)
                c5, head_logits = taps.require()
            if it == 1:
                checks["logits_shape"] = tuple(logits.shape) == (img.shape[0], NUM_CLASSES, 512, 512)
                checks["c5_channels"] = c5.shape[1] == 160
                checks["head_logits_64x64"] = tuple(head_logits.shape[-2:]) == (64, 64)
                checks["c5_32x32"] = tuple(c5.shape[-2:]) == (32, 32)
                checks["teacher_params_frozen"] = not teacher.trainable_parameters()
                checks["optimizer_excludes_teacher"] = not ({id(p) for g in optimizer.param_groups
                                                             for p in g["params"]}
                                                            & teacher.parameter_ids())

            # B32/F8 and L-KD-HARDEN item 8: the teacher's maps are used on their NATIVE grids. The
            # SegNeXt LightHamHead emits 64x64 logits for a 512x512 input (stock in_index=[1,2,3],
            # resized to inputs[0] = stride-8) and the MSCAN-B Stage-3 feature is 32x32: the student
            # head's and C5's grids. No size is requested and nothing is resampled: distillation_losses
            # refuses another grid for a map an instantiated term uses, and the step-1 checks
            # teacher_logits_shape and teacher_feat_shape cover both maps in every stage. L-AM17B-FG:
            # one full forward (features AND logits) in every stage.
            teacher_out = teacher(model_input)
        finally:
            for h in hooks:
                h.remove()
        if it == 1:
            checks["teacher_same_augmented_input"] = (
                seen.get("student") == seen.get("teacher") == (img.data_ptr(), tuple(img.shape)))
            checks["teacher_logits_shape"] = tuple(teacher_out.logits.shape) == tuple(head_logits.shape)
            checks["teacher_feat_shape"] = (teacher_out.feat_s16 is not None
                                            and tuple(teacher_out.feat_s16.shape)
                                            == (c5.shape[0], CWD_C_FEAT, *c5.shape[-2:]))

        ramp_by_term = {t: r(it) for t, r in ramps.items()}
        ramp = ramp_by_term[terms[0]]      # every instantiated term runs the same first-epoch schedule
        # The supervised term is CombinedCEDiceLoss's own sum, from one call of each of its two parts.
        ce = criterion.ce(logits, mask)
        dice = criterion.dice(logits, mask)
        sup = ce + dice
        distill, parts = distillation_losses(
            stage=stage, logits=logits, head_logits=head_logits, c5=c5, mask=mask,
            teacher_out=teacher_out, projection=projection, lambda_logit=lambda_logit,
            ramp=ramp_by_term, alpha=alpha)
        loss = sup + distill
        if it == 1:
            checks["distill_ramp_starts_at_zero"] = all(
                (v == 0.0) if ramp_iters > 1 else (v == 1.0) for v in ramp_by_term.values())
            # Item 1a (G-F1): every ramp is exactly 0 at step 1, so the distillation total is exactly 0
            # and the loss is exactly the unscaled supervised term, which must be positive and finite.
            if ramp_iters > 1:
                checks["distill_zero_at_step1"] = float(distill) == 0.0
                checks["sup_added_unscaled"] = bool(torch.equal(loss, sup))
                checks["sup_positive_finite"] = bool(torch.isfinite(sup)) and float(sup) > 0.0
            else:
                print("[checks] ramp_iters <= 1: step 1 already carries the full distillation weight, "
                      "so distill_zero_at_step1, sup_added_unscaled and sup_positive_finite are set "
                      "True without a test")
                for k in ("distill_zero_at_step1", "sup_added_unscaled", "sup_positive_finite"):
                    checks[k] = True

        if mode == "real" and not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"non-finite loss at iter {it}: total={loss.item():.4f} sup={sup.item():.4f} "
                f"parts={parts}. Aborting the real {stage['name']} run.")
        if it == 1:
            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
            checks["has_expected_terms"] = set(parts) == set(terms)
            p0 = next(p for p in student.parameters() if p.requires_grad)
            before = p0.detach().clone()

        loss.backward()
        # Telemetry only, measured BEFORE any clipping: grad_norm over every trainable parameter (the
        # AM-7 (a) quantity), and its student and projection parts (item 4b).
        grad_norm = total_grad_norm(trainable)
        grad_norm_student = total_grad_norm(student.parameters())
        grad_norm_projection = None if projection is None else total_grad_norm(projection.parameters())
        if grad_clip_norm is not None:
            # global-norm clipping over the student (+ projection), every iteration ("throughout")
            total_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            if it == 1:
                checks["grad_clip_applied"] = bool(torch.isfinite(torch.as_tensor(total_norm)))
        elif it == 1:
            checks["grad_clip_applied"] = True   # path present, intentionally disabled (D-A/D2)
        optimizer.step()
        scheduler.step()
        lr_trace.append(optimizer.param_groups[0]["lr"])
        # Item 4a/4b: the 44c05dc keys in their order, ce and dice after sup, the grad-norm split after
        # grad_norm, and E1's wall-clock fields (train_e1's definitions) last.
        now = time.time()
        row = {"event": "train", "iter": it, "loss": float(loss.item()), "sup": float(sup.item()),
               "ce": float(ce.item()), "dice": float(dice.item()), **parts, "ramp": ramp,
               "lr": lr_trace[-1], "grad_norm": grad_norm, "grad_norm_student": grad_norm_student}
        if projection is not None:
            row["grad_norm_projection"] = grad_norm_projection
        row.update({"wall_clock": now, "iter_seconds": now - t_prev,
                    "samples_per_sec": (batch_size / (now - t_prev)) if now > t_prev else None})
        _jsonl(telemetry_path, row)
        train_seconds += now - t_iter
        t_prev = now

        if it == 1:
            checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)
            checks["teacher_stayed_frozen"] = all(
                p.grad is None for p in teacher.teacher.parameters())

        if it % log_every == 0 or it == max_iters:
            extra = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"[iter {it:>4}/{max_iters}] loss={loss.item():.4f} sup={sup.item():.4f} "
                  f"{extra} ramp={ramp:.4f} lr={lr_trace[-1]:.8e}")

        if it % val_interval == 0 or it == max_iters:
            t_val0 = time.time()
            all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
                                                       max_val_batches)
            val_seconds = time.time() - t_val0
            iou_vec, eligible = per_class_iou(cm)
            # Item 4c: train_e1's val row, written before the best-checkpoint save.
            _jsonl(telemetry_path, {
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
            print(f"[val  {it:>4}/{max_iters}] cm_batches={nvb} all_class_miou={all_miou:.5f} "
                  f"disease_only_miou(PROVISIONAL)={disease_miou:.5f}")
            if all_miou > best_miou:                       # selection = ALL-CLASS val mIoU (D1)
                best_miou = all_miou
                best_ckpt = save_distill_checkpoint(ckpt_dir, stage, student, projection, optimizer,
                                                    scheduler, sched_name, it, best_miou,
                                                    teacher.provenance, semantics_declared,
                                                    semantics_override, lambda_logit=lambda_logit,
                                                    alpha=alpha)
                write_best_pointer(ckpt_dir, best_ckpt, best_miou)   # E1's best.json schema
                print(f"[ckpt {it:>4}/{max_iters}] new best all_class_miou={best_miou:.5f} "
                      f"-> {best_ckpt}")

    checks["lr_non_increasing"] = all(lr_trace[i + 1] <= lr_trace[i] + 1e-12
                                      for i in range(len(lr_trace) - 1))

    hard = ["logits_shape", "c5_channels", "head_logits_64x64", "c5_32x32", "teacher_logits_shape",
            "teacher_feat_shape", "loss_finite", "has_expected_terms", "optimizer_step",
            "teacher_stayed_frozen", "teacher_params_frozen", "optimizer_excludes_teacher",
            "teacher_same_augmented_input", "distill_ramp_starts_at_zero", "distill_zero_at_step1",
            "sup_added_unscaled", "sup_positive_finite", "grad_clip_applied", "val_cm_accumulated",
            "lr_non_increasing"]
    passed = all(checks.get(k, False) for k in hard)
    # The run's last telemetry row, written after the final validation and checkpoint save, so a
    # selection (scripts/select_*.py) can tell a finished run from one that died during its last
    # validation. Not a train row. Item 4d (L-AM16-GPUH): the run's wall-clock span from run() entry,
    # its train-step time, and gpu_hours = wall_seconds / 3600 (validations included).
    wall_clock_end = time.time()
    wall_seconds = wall_clock_end - wall_clock_start
    _jsonl(telemetry_path, {"event": "run_end", "iter": max_iters,
                            "best_val_miou_all_class": best_miou,
                            "best_ckpt": None if best_ckpt is None else Path(best_ckpt).name,
                            "checks_passed": passed, "wall_clock_start": wall_clock_start,
                            "wall_clock_end": wall_clock_end, "wall_seconds": wall_seconds,
                            "train_seconds": train_seconds, "gpu_hours": wall_seconds / 3600.0})
    print("\n[CHECKS]")
    for k in hard:
        print(f"  {k:22}: {'PASS' if checks.get(k) else 'FAIL'}")
    print(f"[summary] best_all_class_val_miou={best_miou:.5f} best_ckpt={best_ckpt}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def parse_args(argv=None, stage_default: str | None = None):
    p = argparse.ArgumentParser(
        description="Shared distillation training core for E2/E3/A/F/G (dry-run by default).")
    if stage_default is None:
        p.add_argument("--stage", choices=sorted(STAGES), required=True,
                       help="e2 = Logit KD; e3 = Logit KD + both CWD terms; a = both CWD terms "
                            "(AM-17 item 7); f = feature-map CWD only; g = logit-map CWD only "
                            "(AM-17b item 1)")
    p.add_argument("--dry-run", action="store_true", help="tiny CPU dry-run (safe default)")
    p.add_argument("--real-run", action="store_true", help="intent to run real training")
    p.add_argument("--confirm-real-run", action="store_true", help="explicit confirmation gate")
    p.add_argument("--teacher-ckpt", default=None,
                   help="path to the fine-tuned SegNeXt-B teacher checkpoint (required for a real run)")
    p.add_argument("--teacher-config", default=None,
                   help="teacher mmseg config (default: the thesis teacher config, whose "
                        "IsolatedNMFLightHamHead implements M4-KD)")
    p.add_argument("--lambda-semantics", default=None,
                   help="OPTIONAL: the Logit-KD semantics tag the supplied --lambda-logit was "
                        "SELECTED under. If given it must match this code's tag; a mismatch "
                        "refuses unless --allow-semantics-mismatch is also passed. Declaring it is "
                        "how a lambda read out of a table months later gets checked, not trusted.")
    p.add_argument("--allow-semantics-mismatch", action="store_true",
                   help="proceed despite a --lambda-semantics mismatch. The override is STAMPED "
                        "into the checkpoint payload and the run_meta file, so the run stays "
                        "identifiable as non-comparable from its artifacts alone.")
    p.add_argument("--lambda-logit", type=float, default=None,
                   help=f"Logit-KD weight (stages with Logit KD only); contract leaves it "
                        f"NEED_TO_CONFIRM (sweep {LAMBDA_SWEEP})")
    p.add_argument("--alpha", type=float, default=None,
                   help=f"feature-map CWD weight alpha_cwd (stages with that term only): one of "
                        f"{ALPHA_GRID}; default {ALPHA_CWD_FEAT} (AM-16 item 2)")
    p.add_argument("--allow-offgrid", action="store_true",
                   help="dry runs only (tests): accept an --alpha outside the AM-16 grid; recorded "
                        "in run_meta as alpha_offgrid; a real run refuses it")
    p.add_argument("--device", default=None)
    p.add_argument("--init", choices=["none", "imagenet"], default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--val-interval", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--ckpt-dir", default=None, help="out-of-repo dir; auto temp dir if omitted; a stage "
                                                    "with the feature-map term needs alpha<value> in "
                                                    "its name")
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

    # ---- L-AM17B-FG / L-AM16-ALPHA: arguments must match the stage's instantiated terms ----
    term_error = term_args_error(stage, lambda_logit=args.lambda_logit,
                                 lambda_semantics=args.lambda_semantics,
                                 allow_semantics_mismatch=args.allow_semantics_mismatch,
                                 alpha=args.alpha, allow_offgrid=args.allow_offgrid, mode=mode)
    if term_error is not None:
        print(f"REFUSING to start the {mode} {stage['name']} run: {term_error}", file=sys.stderr)
        return 2
    alpha = (None if not stage["cwd_feat"]
             else float(ALPHA_CWD_FEAT if args.alpha is None else args.alpha))
    alpha_offgrid = alpha is not None and alpha not in ALPHA_GRID

    if mode == "real":
        # M11 (B60 §5): the E2/E3 data root must be staged with TRAIN and VAL only. TEST surfaces are
        # checked for existence only — never opened, listed or counted.
        try:
            isolation = assert_trainval_only_root(Path(DATA["root"]))
        except TrainValIsolationError as e:
            print(f"REFUSING to start the real {stage['name']} run: [{e.code}] {e}", file=sys.stderr)
            return 2
        print(f"[data] M11 TRAIN/VAL-only root verified: {isolation['counts']}")
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
        if stage["logit_kd"] and args.lambda_logit is None:
            print(f"REFUSING to start the real {stage['name']} run: --lambda-logit is required. "
                  f"The contract leaves lambda_logit as NEED_TO_CONFIRM (validation sweep over "
                  f"{LAMBDA_SWEEP} at seed 42); it is never guessed.", file=sys.stderr)
            return 2
        clip_error = grad_clip_gate_error(args.grad_clip_norm)
        if clip_error is not None:
            print(f"REFUSING to start the real {stage['name']} run: {clip_error}", file=sys.stderr)
            return 2
        sem_error = lambda_semantics_gate_error(args.lambda_semantics,
                                                args.allow_semantics_mismatch)
        if sem_error is not None:
            print(f"REFUSING to start the real {stage['name']} run: {sem_error}", file=sys.stderr)
            return 2
        if stage["cwd_feat"] and args.ckpt_dir is not None:
            dir_error = ckpt_dir_alpha_error(args.ckpt_dir, alpha)
            if dir_error is not None:
                print(f"REFUSING to start the real {stage['name']} run: {dir_error}", file=sys.stderr)
                return 2
        teacher = load_frozen_teacher(args.teacher_ckpt,
                                      config_path=args.teacher_config or str(DEFAULT_TEACHER_CONFIG))
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
            teacher = load_frozen_teacher(
                args.teacher_ckpt, config_path=args.teacher_config or str(DEFAULT_TEACHER_CONFIG))
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
        lambda_logit = (args.lambda_logit if args.lambda_logit is not None
                        else (1.0 if stage["logit_kd"] else None))

    return run(stage=stage, mode=mode, device=device, pretrained=pretrained, teacher=teacher,
               lambda_logit=lambda_logit, batch_size=batch_size, max_iters=max_iters,
               val_interval=val_interval, max_val_batches=max_val_batches,
               num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
               grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed,
               semantics_declared=args.lambda_semantics,
               semantics_override=bool(args.allow_semantics_mismatch),
               alpha=alpha, alpha_offgrid=alpha_offgrid)


if __name__ == "__main__":
    raise SystemExit(main())
