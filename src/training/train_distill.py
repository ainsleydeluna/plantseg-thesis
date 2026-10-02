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
    `--teacher-ckpt`, and requires an explicit `--lambda-logit` from the AM-2 grid for a stage with
    Logit KD (the contract leaves lambda_logit as NEED_TO_CONFIRM, selected by validation sweep — it
    is never guessed here). It runs the registered recipe only (L-KD-HARDEN item 2): a fresh, explicit
    `--ckpt-dir`; the whole 80,000-iteration schedule; VAL on the full set every 4,000 iterations; an
    explicit `--num-workers` >= 1; seed 42, 43 or 44 (42 only for A, F and G); batch 16; ImageNet
    init; the default TF32 state; and NO gradient clipping: AM-7 makes E1, E2 and E3 unclipped, so
    `--grad-clip-norm` is refused (dry runs accept it). A stage with the feature-map CWD term takes
    `--alpha` from {25, 50, 100} only, and its `--ckpt-dir` name must carry exactly one token
    `alpha<value>`. An argument for a term the stage does not instantiate is refused, not ignored.
  * main()'s gates return before any dataloader or teacher is constructed; run() repeats the schedule
    and CUDA-order checks at its entry, after main() has loaded the teacher, and in a real run it also
    refuses a missing --ckpt-dir, a clipping value and an off-grid alpha (direct calls).
  * A real run stops (L-KD-HARDEN items 1b and 3; AM-7, DL-04) on AM-7 (a), a non-finite total loss
    before backward or a non-finite pre-clip gradient norm before the step, from iteration 2 on; on
    AM-7 (b), the post-ramp rolling-mean rule of configs/distill.py AM7_DIVERGENCE; on a non-finite VAL
    mIoU; and on a step-1 failure (rule step1_checks): a failed step-1 check after iteration 1's block,
    or a non-finite loss or gradient norm AT iteration 1 (orchestrator ruling R8-1, not yet in DL-04 or
    AM-7: before the first update nothing can diverge). It writes that iteration's row, then a
    run_abort record (never run_end), and raises RunAborted; main() reports it and exits 3. Telemetry
    is strict JSON: a row writes a non-finite value as null and lists it in its `nonfinite` map; the
    run_abort record writes "nan", "inf" or "-inf".
  * Checkpoints are NEVER written inside the repo. The training-only CWD projection and its optimizer
    group are written to `projection.pt` beside the checkpoint, never into it, so `model_state_dict`
    is already the clean E6/E7 deployment student and the checkpoint carries nothing to strip.

Exit codes of main() (and of train_e2.py / train_e3.py):
  0  the run finished and every hard check passed (RESULT: PASS); argparse's --help also exits 0
  1  the run finished with a failed hard check (RESULT: FAIL); also an uncaught exception (a traceback)
  2  refused before training: a REFUSING or ERROR line (most name a [code]), or an argparse usage error;
     run() returns 2 itself for [cuda_initialized_before_seed]
  3  a real run aborted (RunAborted: AM-7 (a) or (b), val_nonfinite or step1_checks): RESULT: ABORTED;
     the run_abort record is the telemetry's last row, and the run is never relaunched (AM-7a; a fault
     follows AM-8a)

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
from collections import deque
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs.data import DATA                                      # noqa: E402
from configs.distill import (AM7_DIVERGENCE, DISTILL, DISTILL_STAGES,  # noqa: E402
                             DISTILL_TERMS, LOGIT_KD_SEMANTICS, LOGIT_KD_SEMANTICS_SUPERSEDED)
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
    """Validate `--grad-clip-norm` for a REAL E2/E3/A/F/G launch. Returns an error string, or None if OK.

    AM-7 (docs/PREREGISTRATION_AMENDMENTS.md; DL-04) makes E1, E2 and E3 unclipped under one shared
    rule, so a real run of any distillation stage takes no clipping value and ANY value is refused
    (lane L-AM7, carried out by L-KD-HARDEN item 2g); divergence is governed by AM-7's rules (a) and
    (b) instead. The pre-amendment pilot (configs/distill.py `distillation_grad_clip_pilot`) is
    WITHDRAWN_AM7. Dry runs never call this gate, so they still accept a value.
    """
    if value is None:
        return None
    return (f"--grad-clip-norm {value!r} was given, but AM-7 (DL-04) makes E1, E2 and E3 unclipped: "
            "a real run of any distillation stage takes no clipping value (divergence is governed by "
            "AM-7's rules (a) and (b)). Dry runs still accept the flag.")


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
        return (f"--alpha {alpha!r} is not in the AM-16 item 2 grid {ALPHA_GRID} (default "
                f"{ALPHA_CWD_FEAT}); --allow-offgrid admits other values in dry runs only")
    return None


# An alpha<value> token in a directory name (L-KD-HARDEN item 2j): not preceded by a letter or digit,
# not followed by a digit or a decimal part. ALPHA_VALUE_RE counts every alpha<value>, glued to other
# text or not, so a second value refuses the name even when glued on (e3_s42_alpha25alpha50).
ALPHA_TOKEN_RE = re.compile(r"(?<![0-9A-Za-z])alpha[0-9]+(?:\.[0-9]+)?(?![0-9]|\.[0-9])",
                            re.IGNORECASE)
ALPHA_VALUE_RE = re.compile(r"alpha[0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def ckpt_dir_alpha_error(ckpt_dir, alpha: float) -> str | None:
    """A run with the feature-map term names its checkpoint dir after its alpha (L-AM16-ALPHA): the
    last path component must carry exactly one alpha<value> (item 2j), as a token, and it must be the
    run's, e.g. /workspace/e3_s42_alpha50, with no digit following it. Returns an error string, or
    None if OK."""
    tok = alpha_token(alpha)
    name = Path(ckpt_dir).name
    values = ALPHA_VALUE_RE.findall(name)
    if len(values) > 1:
        return (f"--ckpt-dir {ckpt_dir} carries {len(values)} alpha values {values} in its last path "
                f"component; exactly one, {tok!r}, names the run's alpha, e.g. .../e3_s42_{tok}")
    found = ALPHA_TOKEN_RE.findall(name)
    if len(found) == 1 and found[0].lower() == tok.lower():
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
    """A forward pre-hook recording the (data_ptr, shape, stride) of the tensor a model receives (item
    1c). The stride catches a layout-changing view such as a transpose, which keeps a square input's
    pointer and shape."""
    def hook(module, inputs):
        seen[name] = (inputs[0].data_ptr(), tuple(inputs[0].shape), tuple(inputs[0].stride()))
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


# ------------------------------------------------------------------ real-run gates (L-KD-HARDEN item 2)
# Item 2h: the pinned torch 2.1.0 defaults (cuDNN convolutions may use TF32 on Ampere; matmul TF32 off;
# float32_matmul_precision "highest") with NVIDIA_TF32_OVERRIDE unset. Contract B6.
TF32_DEFAULTS = {"cudnn_allow_tf32": True, "matmul_allow_tf32": False,
                 "float32_matmul_precision": "highest", "NVIDIA_TF32_OVERRIDE": None}
# Item 2f: E2 and E3 run seeds 42, 43 and 44; the exploratory arms A, F and G run seed 42 only.
REAL_RUN_SEEDS = {"e2": (42, 43, 44), "e3": (42, 43, 44), "a": (42,), "f": (42,), "g": (42,)}
if set(REAL_RUN_SEEDS) != set(STAGES):
    raise StageConfigError(f"REAL_RUN_SEEDS covers {sorted(REAL_RUN_SEEDS)}, the stage table "
                           f"{sorted(STAGES)}")


def tf32_gate_error(state: dict) -> str | None:
    """Item 2h: a real run starts only under TF32_DEFAULTS. Returns an error string, or None if OK."""
    bad = {k: (state.get(k), want) for k, want in TF32_DEFAULTS.items() if state.get(k) != want}
    if not bad:
        return None
    return (f"the TF32 state differs from the pinned torch 2.1.0 defaults: {bad} (got, want). No "
            "trainer sets these flags, so the environment did; restore the defaults (unset "
            "NVIDIA_TF32_OVERRIDE) and relaunch")


def schedule_gate_error(mode: str, max_iters: int, *, horizon: int | None = None) -> str | None:
    """Item 2c. None = admissible, else the refusal text, led by its bracketed code.

    The KD stages train on E1's fixed poly horizon (E1_STUDENT["iterations"]). PolynomialLR holds the
    LR at 0.0 past it, so no mode may run longer, and a real run trains exactly the whole schedule.
    `horizon` is keyword-only: train_e1.schedule_gate_error takes (mode, poly_horizon, max_iters).
    """
    horizon = E1_STUDENT["iterations"] if horizon is None else horizon
    if max_iters > horizon:
        return (f"[max_iters_above_horizon] --max-iters {max_iters} exceeds the poly horizon {horizon}: "
                f"PolynomialLR holds lr at 0.0 after it, so iterations {horizon + 1}..{max_iters} "
                "would not train")
    if mode == "real" and max_iters != horizon:
        return (f"[max_iters_not_horizon] a real run trains its whole schedule: --max-iters {max_iters} "
                f"!= the poly horizon {horizon}; do not pass --max-iters to a real run")
    return None


def ckpt_dir_fresh_error(ckpt_dir) -> str | None:
    """Item 2b: a real run starts in a fresh checkpoint directory, absent or empty (the check of
    scripts/preflight_e1_trainval.py). Its run_meta and telemetry files are appended to, so a reused
    directory would interleave two runs. Returns an error string, or None if OK."""
    p = Path(ckpt_dir)
    if p.exists() and (not p.is_dir() or any(p.iterdir())):
        return (f"--ckpt-dir {ckpt_dir} exists and is not an empty directory; a real run starts in a "
                "fresh directory (its run_meta and telemetry files are appended to)")
    return None


# ---------------------------------------------------------------- AM-7 aborts (L-KD-HARDEN item 3)
# AM-7 (b)'s pre-registered parameters (DL-04). configs/distill.py AM7_DIVERGENCE must carry them:
# another window, factor or ramp rule is a new amendment, not a config edit, so the import refuses it.
AM7_REGISTERED = {"window": 100, "factor": 5.0, "post_ramp_only": True}


def am7_divergence_error(cfg) -> str | None:
    """None when `cfg` carries AM7_REGISTERED's values (a bool never stands in for a number, nor a
    number for a bool), else the refusal text."""
    bad = {k: (cfg.get(k), want) for k, want in AM7_REGISTERED.items()
           if cfg.get(k) != want or isinstance(cfg.get(k), bool) != isinstance(want, bool)}
    if not bad:
        return None
    return (f"configs/distill.py AM7_DIVERGENCE departs from AM-7 (b)'s registered parameters: {bad} "
            "(got, want)")


if am7_divergence_error(AM7_DIVERGENCE) is not None:
    raise ValueError(am7_divergence_error(AM7_DIVERGENCE))
AM7_RULES = ("AM-7(a)", "AM-7(b)")


class RunAborted(RuntimeError):
    """A real run stopped by AM-7 (a) or (b), a non-finite VAL mIoU or failed step-1 checks. `record`
    is the run_abort row, the telemetry's last line; no run_end follows it."""

    def __init__(self, record: dict):
        super().__init__(f"{record['rule']} at iter {record['iter']} (cause {record['cause']})")
        self.record = record

    def __reduce__(self):                 # pickling and copying rebuild it from the record
        return (RunAborted, (self.record,))


ABORTED_EXIT = 3                          # main()'s exit code for a RunAborted run (module docstring)


class AM7bMonitor:
    """AM-7 (b), pure (DL-04; item 3b): a divergence monitor on the logged total loss.

    Only post-ramp iterations (it > ramp_iters) enter a rolling window of `window` losses; ramp
    iterations never enter a window or the minimum. From it = ramp_iters + window on, every iteration
    evaluates the window mean: the run diverges when the mean is STRICTLY greater than factor x the
    running minimum of the earlier window means; otherwise the mean joins the minimum. update() returns
    None, or on divergence {window_mean, running_min, ratio, threshold} (threshold = factor x
    running_min; ratio = window_mean / running_min, inf when the minimum is 0).
    """

    def __init__(self, ramp_iters: int, window: int = 100, factor: float = 5.0):
        self.ramp_iters, self.window, self.factor = int(ramp_iters), int(window), float(factor)
        self._losses: deque = deque(maxlen=self.window)
        self.running_min: float | None = None
        self.n_windows = 0

    def update(self, it: int, loss: float) -> dict | None:
        if it <= self.ramp_iters:
            return None
        self._losses.append(float(loss))
        if len(self._losses) < self.window:
            return None
        mean = math.fsum(self._losses) / self.window
        self.n_windows += 1
        if self.running_min is not None and mean > self.factor * self.running_min:
            return {"window_mean": mean, "running_min": self.running_min,
                    "ratio": mean / self.running_min if self.running_min > 0 else math.inf,
                    "threshold": self.factor * self.running_min}
        self.running_min = mean if self.running_min is None else min(self.running_min, mean)
        return None


def _nonfinite_tag(x: float) -> str:
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


def strict_row(row: dict) -> dict:
    """A telemetry row as strict JSON (item 3a): each non-finite float becomes null and is listed, as
    "nan", "inf" or "-inf", in a trailing `nonfinite` map. A finite row is returned unchanged."""
    bad = {k: _nonfinite_tag(v) for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)}
    if not bad:
        return row
    return {**{k: (None if k in bad else v) for k, v in row.items()}, "nonfinite": bad}


def record_number(x) -> float | str | None:
    """A run_abort detail value (AM-7a): a finite float as written, a non-finite one as "nan", "inf" or
    "-inf", and null when it was not computed at the abort point."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else _nonfinite_tag(x)


def nonfinite_rule(it: int) -> str:
    """The rule of a non-finite total loss or gradient norm at iteration `it` (orchestrator ruling
    R8-1): AM-7 (a) from iteration 2 on. At iteration 1 it is a step-1 failure: before the first update
    nothing can diverge, and with every ramp at 0 the loss does not depend on lambda or alpha, so AM-7a
    item 4 sends it to STOP (it must never read as a default candidate's divergence)."""
    return "AM-7(a)" if it >= 2 else "step1_checks"


def abort_cause(rule: str, input_finite: bool, teacher_finite: bool) -> str:
    """AM-7a: an AM-7 (a)/(b) stop is a STUDENT divergence only when the step's input and teacher
    outputs were finite; otherwise it is a fault of the input or of the teacher. step1_checks (a failed
    step-1 check, or a non-finite loss or gradient norm at iteration 1, R8-1) is checks_failed whatever
    the input and teacher were; its record still carries input_finite and teacher_finite."""
    if rule in AM7_RULES:
        if not input_finite:
            return "input_nonfinite"
        if not teacher_finite:
            return "teacher_nonfinite"
        return "student_divergence"
    return {"val_nonfinite": "val_nonfinite", "step1_checks": "checks_failed"}[rule]


def run(*, stage: dict, mode: str, device: str, pretrained, teacher: FrozenTeacher,
        lambda_logit: float | None, batch_size: int, max_iters: int, val_interval: int,
        max_val_batches: int | None, num_workers: int, ckpt_dir_arg: str | None,
        semantics_declared=None, semantics_override: bool = False,
        grad_clip_norm: float | None, log_every: int, seed: int,
        alpha: float | None = None, alpha_offgrid: bool = False) -> int:
    wall_clock_start = time.time()
    # Item 2i: set_seed exports CUBLAS_WORKSPACE_CONFIG and the determinism settings, which must precede
    # the first CUDA op (contract B6), so CUDA must still be uninitialised here; main() checks the same
    # before the teacher load. Nothing that touches CUDA or data comes before these entry checks.
    if mode == "real" and torch.cuda.is_initialized():
        print(f"REFUSING to start the real {stage['name']} run: [cuda_initialized_before_seed] CUDA was "
              "initialised before run() seeded the run; find what touched CUDA first",
              file=sys.stderr)
        return 2
    sched_error = schedule_gate_error(mode, max_iters)            # item 2c, repeated for direct calls
    if sched_error is not None:
        raise RuntimeError(sched_error)
    if mode == "real":
        # Item 2 for direct calls: the real-run gates of main() that run() can check itself.
        if not ckpt_dir_arg:                      # None or "": resolve_ckpt_dir would take a temp dir
            raise ValueError("[ckpt_dir_required] a real run needs an explicit, fresh --ckpt-dir; the "
                             "temporary-directory fallback is for dry runs only")
        clip_error = grad_clip_gate_error(grad_clip_norm)
        if clip_error is not None:
            raise ValueError(f"[grad_clip_am7] {clip_error}")
        if stage["cwd_feat"]:
            alpha_error = alpha_gate_error(alpha, allow_offgrid=alpha_offgrid, mode="real")
            if alpha_error is not None:
                raise ValueError(f"[alpha] {alpha_error}")
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
        clip_msg = "none: AM-7 (DL-04) makes E1, E2 and E3 unclipped; a real run refuses the flag"
    else:
        clip_msg = (f"global-norm, max_norm={grad_clip_norm}, applied every iteration (dry run only: a "
                    "real run refuses --grad-clip-norm under AM-7)")
    print(f"[grad-clip] {clip_msg}")

    if ckpt_dir_arg is None and stage["cwd_feat"]:
        # L-AM16-ALPHA: the checkpoint dir of a run with the feature-map term carries its alpha. Dry
        # runs only: a real run has already refused a missing --ckpt-dir (item 2b).
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
    n_val = 0
    # AM-7 (b), real runs only (item 3b). window and factor are read from AM7_DIVERGENCE here, at run
    # time (item 3d); the import has pinned the config's values to AM7_REGISTERED.
    am7b = (AM7bMonitor(ramp_iters, window=AM7_DIVERGENCE["window"], factor=AM7_DIVERGENCE["factor"])
            if mode == "real" else None)

    def write_train_row(it: int, *, lr=None, norms=(None, None, None)) -> None:
        """The iteration's train row (items 3a, 4a, 4b): the 44c05dc keys in their order, ce and dice
        after sup, the grad-norm split after grad_norm, E1's wall-clock fields last. A value not computed
        when a run aborts is null; a non-finite float is null and listed in `nonfinite`."""
        nonlocal t_prev, train_seconds
        now = time.time()
        row = {"event": "train", "iter": it, "loss": float(loss.item()), "sup": float(sup.item()),
               "ce": float(ce.item()), "dice": float(dice.item()), **parts, "ramp": ramp,
               "lr": lr, "grad_norm": norms[0], "grad_norm_student": norms[1]}
        if projection is not None:
            row["grad_norm_projection"] = norms[2]
        row.update({"wall_clock": now, "iter_seconds": now - t_prev,
                    "samples_per_sec": (batch_size / (now - t_prev)) if now > t_prev else None})
        _jsonl(telemetry_path, strict_row(row))
        train_seconds += now - t_iter
        t_prev = now

    def abort(rule: str, it: int, *, loss_value=None, grad_norm_value=None, hit=None) -> None:
        """Write the run_abort record (AM-7a section A) as the telemetry's last row, then raise. The
        window fields are AM-7 (b)'s; another rule records only the monitor's running minimum."""
        input_finite = bool(torch.isfinite(img).all())
        teacher_finite = bool(torch.isfinite(teacher_out.logits).all()) and (
            teacher_out.feat_s16 is None or bool(torch.isfinite(teacher_out.feat_s16).all()))
        hit = hit or {}
        running_min = hit.get("running_min", None if am7b is None else am7b.running_min)
        record = {"event": "run_abort", "iter": it, "rule": rule,
                  "cause": abort_cause(rule, input_finite, teacher_finite),
                  "detail": {"loss": record_number(loss_value), "grad_norm": record_number(grad_norm_value),
                             "window_mean": record_number(hit.get("window_mean")),
                             "running_min": record_number(running_min),
                             "ratio": record_number(hit.get("ratio")),
                             "threshold": record_number(hit.get("threshold"))},
                  "input_finite": input_finite, "teacher_finite": teacher_finite,
                  "params_finite": all(bool(torch.isfinite(p).all()) for p in trainable),
                  "n_val": n_val, "wall_clock": time.time()}
        _jsonl(telemetry_path, record)
        print(f"[abort] {rule} at iter {it} (cause {record['cause']}): {json.dumps(record['detail'])}. "
              f"The real {stage['name']} run stops here; no run_end is written.", file=sys.stderr)
        raise RunAborted(record)

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
                seen.get("student") == seen.get("teacher")
                == (img.data_ptr(), tuple(img.shape), tuple(img.stride())))
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

        # AM-7 (a), real runs (item 3a): a non-finite total loss stops the run BEFORE backward; its row
        # has no lr and no gradient norms (not computed), and the weights are not stepped. At iteration
        # 1 the same stop is a step-1 failure, rule step1_checks (R8-1; nonfinite_rule).
        if mode == "real" and not bool(torch.isfinite(loss)):
            write_train_row(it)
            abort(nonfinite_rule(it), it, loss_value=float(loss.item()))
        if it == 1:
            checks["loss_finite"] = bool(torch.isfinite(loss)) and loss.dim() == 0
            checks["has_expected_terms"] = set(parts) == set(terms)
            p0 = next(p for p in student.parameters() if p.requires_grad)
            before = p0.detach().clone()

        loss.backward()
        # Measured BEFORE any clipping, and never used to scale a gradient: grad_norm over every
        # trainable parameter (the AM-7 (a) quantity: logged, and tested below in a real run), and its
        # student and projection parts (telemetry, item 4b).
        grad_norm = total_grad_norm(trainable)
        grad_norm_student = total_grad_norm(student.parameters())
        grad_norm_projection = None if projection is None else total_grad_norm(projection.parameters())
        norms = (grad_norm, grad_norm_student, grad_norm_projection)
        # AM-7 (a), real runs: a non-finite pre-clip gradient norm stops the run before optimizer.step
        # (step1_checks at iteration 1, R8-1).
        if mode == "real" and not math.isfinite(grad_norm):
            write_train_row(it, norms=norms)
            abort(nonfinite_rule(it), it, loss_value=float(loss.item()), grad_norm_value=grad_norm)
        if grad_clip_norm is not None:
            # global-norm clipping over the student (+ projection), every iteration ("throughout")
            total_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            if it == 1:
                checks["grad_clip_applied"] = bool(torch.isfinite(torch.as_tensor(total_norm)))
        elif it == 1:
            checks["grad_clip_applied"] = True   # no clipping: AM-7 (the dry-run clip path is above)
        optimizer.step()
        scheduler.step()
        lr_trace.append(optimizer.param_groups[0]["lr"])
        write_train_row(it, lr=lr_trace[-1], norms=norms)

        if it == 1:
            checks["optimizer_step"] = bool((p0.detach() - before).abs().sum().item() > 0.0)
            checks["teacher_stayed_frozen"] = all(
                p.grad is None for p in teacher.teacher.parameters())
            # Item 1b: a real run whose step-1 checks failed stops now, not 80,000 iterations later.
            failed = [k for k, ok in checks.items() if not ok]
            if mode == "real" and failed:
                print(f"[checks] step-1 checks FAILED: {failed}", file=sys.stderr)
                abort("step1_checks", it, loss_value=float(loss.item()), grad_norm_value=grad_norm)

        # AM-7 (b), real runs (item 3b): the post-ramp rolling mean of the logged total loss.
        if am7b is not None:
            hit = am7b.update(it, float(loss.item()))
            if hit is not None:
                abort("AM-7(b)", it, loss_value=float(loss.item()), grad_norm_value=grad_norm, hit=hit)

        if it % log_every == 0 or it == max_iters:
            extra = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"[iter {it:>4}/{max_iters}] loss={loss.item():.4f} sup={sup.item():.4f} "
                  f"{extra} ramp={ramp:.4f} lr={lr_trace[-1]:.8e}")

        if it % val_interval == 0 or it == max_iters:
            t_val0 = time.time()
            all_miou, disease_miou, cm, nvb = validate(student, val_loader, dev, NUM_CLASSES,
                                                       max_val_batches)
            val_seconds = time.time() - t_val0
            n_val += 1
            iou_vec, eligible = per_class_iou(cm)
            # Item 4c: train_e1's val row, written before the best-checkpoint save (strict JSON, 3a).
            _jsonl(telemetry_path, strict_row({
                "event": "val", "iter": it, "all_class_miou": all_miou,
                "disease_only_miou_PROVISIONAL": disease_miou,
                "per_class_iou": [round(float(x), 8) for x in iou_vec.tolist()],
                "per_class_eligible": [bool(x) for x in eligible.tolist()],
                "n_eligible_classes": int(eligible.sum()), "val_batches": nvb,
                "val_total_px": int(cm.sum()), "val_seconds": val_seconds,
                "wall_clock": time.time(),
            }))
            # Item 3c: a non-finite VAL all-class mIoU stops a real run, after its val row.
            if mode == "real" and not math.isfinite(all_miou):
                abort("val_nonfinite", it, loss_value=float(loss.item()), grad_norm_value=grad_norm)
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
    _jsonl(telemetry_path, strict_row({"event": "run_end", "iter": max_iters,
                                       "best_val_miou_all_class": best_miou,
                                       "best_ckpt": None if best_ckpt is None else Path(best_ckpt).name,
                                       "checks_passed": passed, "wall_clock_start": wall_clock_start,
                                       "wall_clock_end": wall_clock_end, "wall_seconds": wall_seconds,
                                       "train_seconds": train_seconds,
                                       "gpu_hours": wall_seconds / 3600.0}))
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
    p.add_argument("--num-workers", type=int, default=None,
                   help="TRAIN/VAL loader workers; a real run needs an explicit value >= 1 (official 12)")
    p.add_argument("--ckpt-dir", default=None, help="out-of-repo dir, absent or empty; required for a "
                                                    "real run (a dry run gets a temp dir if omitted); a "
                                                    "stage with the feature-map term needs exactly one "
                                                    "alpha<value> token in its name")
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="dry runs only: a real run refuses it (AM-7: E1, E2 and E3 are unclipped)")
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

    def refuse(code: str, message: str) -> int:
        print(f"REFUSING to start the {mode} {stage['name']} run: [{code}] {message}", file=sys.stderr)
        return 2

    max_iters = ((args.max_iters if args.max_iters is not None else E1_STUDENT["iterations"])
                 if mode == "real" else (args.max_iters or 4))

    def both_mode_gates() -> int | None:
        """L-KD-HARDEN items 2j and 2c and the log cadence (Q10), in both modes; a real run checks them
        after M11 and the device."""
        sem_error = lambda_semantics_gate_error(args.lambda_semantics, args.allow_semantics_mismatch)
        if sem_error is not None:
            return refuse("lambda_semantics", sem_error)
        sched_error = schedule_gate_error(mode, max_iters)
        if sched_error is not None:
            print(f"REFUSING to start the {mode} {stage['name']} run: {sched_error}", file=sys.stderr)
            return 2
        if args.log_every < 1:
            return refuse("log_every", f"--log-every {args.log_every}: the train log is printed every "
                                       "--log-every iterations, a whole number >= 1")
        return None

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
        gate_rc = both_mode_gates()
        if gate_rc is not None:
            return gate_rc
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
        # ---- L-KD-HARDEN item 2: the registered recipe, refused by name, before the teacher load ----
        lam = args.lambda_logit
        if stage["logit_kd"] and (not math.isfinite(lam) or lam <= 0.0 or lam not in LAMBDA_SWEEP):
            return refuse("lambda_grid", f"--lambda-logit {lam!r} is not in the AM-2 grid "
                                         f"{LAMBDA_SWEEP}; a real run takes lambda from the grid, "
                                         "with no override")
        clip_error = grad_clip_gate_error(args.grad_clip_norm)
        if clip_error is not None:
            return refuse("grad_clip_am7", clip_error)
        if not args.ckpt_dir:                     # None or "": resolve_ckpt_dir would take a temp dir
            return refuse("ckpt_dir_required", "--ckpt-dir is required: a real run writes into an "
                                               "explicit, fresh, out-of-repo directory; the "
                                               "temporary-directory fallback is for dry runs only")
        # Q10: train_e1's guard, now before anything is built. The path is resolved first, so an error of
        # the resolution itself (pathlib's RuntimeError for a symlink loop) raises as it does in train_e1
        # and is never refused as an in-repo directory.
        ckpt_resolved = Path(args.ckpt_dir).resolve()
        try:
            _assert_outside_repo(ckpt_resolved)
        except RuntimeError as e:
            return refuse("ckpt_dir_in_repo", f"{e}; checkpoints live outside the repository")
        fresh_error = ckpt_dir_fresh_error(args.ckpt_dir)
        if fresh_error is not None:
            return refuse("ckpt_dir_not_fresh", fresh_error)
        if stage["cwd_feat"]:
            dir_error = ckpt_dir_alpha_error(args.ckpt_dir, alpha)
            if dir_error is not None:
                return refuse("ckpt_dir_alpha", dir_error)
        if args.max_val_batches is not None:
            return refuse("max_val_batches", f"--max-val-batches {args.max_val_batches} was given; "
                                             "a real run validates on the full VAL set")
        if args.val_interval is not None and args.val_interval != E1_STUDENT["val_interval"]:
            return refuse("val_interval", f"--val-interval {args.val_interval} != "
                                          f"{E1_STUDENT['val_interval']}; a real run validates every "
                                          f"{E1_STUDENT['val_interval']} iterations, as E1")
        if args.num_workers is None or args.num_workers < 1:
            return refuse("num_workers", f"--num-workers {args.num_workers!r}: a real run needs an "
                                         "explicit value >= 1 (the official value is 12, E1's), "
                                         "recorded in run_meta")
        if args.seed not in REAL_RUN_SEEDS[stage["key"]]:
            return refuse("seed", f"--seed {args.seed} is not a registered seed of stage "
                                  f"{stage['name']}: {REAL_RUN_SEEDS[stage['key']]}")
        if args.batch_size is not None and args.batch_size != E1_STUDENT["batch_size"]:
            return refuse("batch_size", f"--batch-size {args.batch_size} != "
                                        f"{E1_STUDENT['batch_size']}, the E1 recipe's batch")
        if args.init is not None and args.init != "imagenet":
            return refuse("init", f"--init {args.init}: a real run starts from E1's ImageNet init")
        tf32_error = tf32_gate_error(tf32_state())
        if tf32_error is not None:
            return refuse("tf32", tf32_error)
        # Item 2i (decision C5): CUDA must still be uninitialised when run() calls set_seed, which
        # exports CUBLAS_WORKSPACE_CONFIG and the determinism settings that must precede the first
        # CUDA op (contract B6); run() checks the same at its entry, after the teacher load.
        if torch.cuda.is_initialized():
            return refuse("cuda_initialized_before_teacher", "CUDA is already initialised before the "
                                                             "teacher load and the run's seeding")
        teacher = load_frozen_teacher(args.teacher_ckpt,
                                      config_path=args.teacher_config or str(DEFAULT_TEACHER_CONFIG))
        pretrained = E1_STUDENT["init_weights"]
        batch_size = E1_STUDENT["batch_size"]
        val_interval = E1_STUDENT["val_interval"]
        max_val_batches = None
        num_workers = args.num_workers
        lambda_logit = args.lambda_logit
    else:
        gate_rc = both_mode_gates()
        if gate_rc is not None:
            return gate_rc
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
        val_interval = args.val_interval or 2
        max_val_batches = args.max_val_batches if args.max_val_batches is not None else 2
        num_workers = args.num_workers if args.num_workers is not None else 0
        lambda_logit = (args.lambda_logit if args.lambda_logit is not None
                        else (1.0 if stage["logit_kd"] else None))

    try:
        return run(stage=stage, mode=mode, device=device, pretrained=pretrained, teacher=teacher,
                   lambda_logit=lambda_logit, batch_size=batch_size, max_iters=max_iters,
                   val_interval=val_interval, max_val_batches=max_val_batches,
                   num_workers=num_workers, ckpt_dir_arg=args.ckpt_dir,
                   grad_clip_norm=args.grad_clip_norm, log_every=args.log_every, seed=args.seed,
                   semantics_declared=args.lambda_semantics,
                   semantics_override=bool(args.allow_semantics_mismatch),
                   alpha=alpha, alpha_offgrid=alpha_offgrid)
    except RunAborted as e:                       # run() raises; the process exits cleanly with code 3
        r = e.record
        print(f"RESULT: ABORTED rule={r['rule']} iter={r['iter']} cause={r['cause']}; the run_abort record "
              "is the telemetry's last row; do not relaunch this run (AM-7a; a fault follows AM-8a)")
        return ABORTED_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
