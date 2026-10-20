"""E6-KD: E6's INT8 QAT with the teacher of record in the loop (lane L-AM3, B-2; AM-3, DL-39, DL-51).

AM-3 runs E6-KD at seed 42 only, when the committed trigger fired: E6's recipe of record (AM-4a; the constants, the
loaders, the freezes, the VAL bracket, the state predicate and the epoch checkpoint are src/quant/qat.py's, reused
unchanged), plus E3's three distillation terms at exactly 0.5x their E3 weights, T unchanged (AM-3; DL-07, DL-39):

    loss = CE + Dice + r(step) * (0.5 * lambda_logit * L_LogitKD + 0.5 * alpha_cwd * L_CWD_feat + 0.5 * beta_cwd * L_CWD_logit)

  * the terms are src/training/train_distill.py's distillation_losses on E3's stage (all three on), on the fake-quant
    taps of the prepared student (c5 = features[15], head_logits = the head's ff_add), in its addition order;
  * r(step) is E3's one-epoch linear ramp (train_distill.distill_ramp), one per term, over the global QAT step, with
    ramp_iters = len(TRAIN loader) = 335; CE + Dice is never ramped (DL-39);
  * the teacher of record (DL-88) runs one frozen forward per step on the student's input, after the student's
    forward, on its private M4-KD NMF stream begun once after set_seed (src/distill/segnext_teacher.py);
  * the CWD projection starts from E3's projection.pt, paired with E3's best.json checkpoint (DL-51; LB14: the file's
    checkpoint is best.json's, its iter the checkpoint's, its stage E3, its sha256 pinned). It is built without a
    global RNG draw (skip_init) and loaded strictly; it stays FP32, outside the prepared student and out of every
    epoch checkpoint; its optimizer state is not loaded: it gets a fresh "cwd_projection" group under the QAT
    optimizer (DL-51), with the student's lr, momentum, weight decay and cosine;
  * the clip is the U4 winner (AM-4a item 3) over student and projection, in two levels: each group's norm as torch
    2.1's clip_grad_norm_ computes one list's, then the 2-norm of the two (DL-04 (a)'s "every trainable parameter");
    grad_norm, grad_norm_student and grad_norm_projection are logged.

Aborts (DL-04 (a), DL-51; AM-21's header keeps E6-KD's generic abort, and its items 2, 4 and 5 apply). A non-finite
total loss before backward, or a non-finite pre-clip grad_norm before optimizer.step, writes the step's row, then a
run_abort record, the telemetry's last row. At step 1 the rule is step1_checks, cause checks_failed (R8-1, DL-04), as
for a failed step-1 check; from step 2 it is GENERIC_RULE, with the cause read from the finiteness of the input and
the teacher outputs: input_nonfinite, teacher_nonfinite, or student_nonfinite when both were finite (never a
divergence cause: E6-KD has no divergence rule; LB16). An exception writes a run_abort with cause exception. There
is no AM-7 (b) monitor. A state found non-finite at an epoch end while the loss and norm stayed finite is recorded in
that epoch's checkpoint and epoch_end row, and the run continues (LB17; AM-21 item 2(a) excludes the epoch).

Records: qat_telemetry.jsonl (Q2's name, so src/quant/qat_artifacts.py reads it), schema TELEMETRY_SCHEMA, Q2's rows
with the KD columns; epoch_ckpts/eNN.pt in Q2's sidecar schema with stage E6-KD. Conversion, scoring and selection
are Q2's code under the E6-KD rules file (scripts/qat_epoch_eval.py, scripts/select_qat_epoch.py); finalize and the
x86 copy do not apply (LB31). The launch gates, in order, are real_run_gates'; none writes anything.

    python -B scripts/run_e6kd.py --real-run --confirm-real-run ...      (exit 0 done, 1 STOP, 2 refused,
                                                                          3 aborted or incomplete, 4 error)
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
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

from configs.distill import LOGIT_KD_SEMANTICS                               # noqa: E402
from src.distill import teacher as TT                                        # noqa: E402
from src.distill.cwd_projection import (PROJECTION_FILE, PROJECTION_FORMAT,  # noqa: E402
                                        PROJECTION_GROUP, PROJECTION_PARAMS, STUDENT_C5_CH, TEACHER_STRIDE16_CH)
from src.distill.features import StudentTaps                                 # noqa: E402
from src.distill.nmf_stream import M4_NMF_SEED                               # noqa: E402
from src.distill.segnext_teacher import TeacherStateDictMismatch            # noqa: E402
from src.training import train_distill as TD                                 # noqa: E402

from . import qat as Q                                                       # noqa: E402
from .e6kd_trigger import read_trigger_record, trigger_binding               # noqa: E402
from .prepare import disable_observers, freeze_bn_stats, prepare_qat_model, qat_freeze_steps, unfused_batchnorm  # noqa: E402
from .qat import QATRefused, QATStop                                         # noqa: E402
from .qconfig import QuantBackendUnavailable, select_qnnpack_backend         # noqa: E402
from .stages import resolve_quant_stage                                      # noqa: E402

LANE = "L-AM3 (docs/lane_specs/part2.md lane 7), B-2"
STAGE_KEY, STAGE_NAME, METHOD = "e6-kd", "E6-KD", "qat_kd"
TELEMETRY_SCHEMA = "plantseg-e6kd-telemetry/1.0.0"
SEED = 42                                       # AM-3: seed 42 only
WEIGHT_FACTOR = 0.5                             # AM-3: the Logit-KD and CWD weights are 0.5x E3's
E3_STAGE = TD.resolve_stage("e3")               # E3's instantiated terms: all three (DL-39)
TERMS = TD.TERMS                                # logit_kd, cwd_feat, cwd_logit: the addition order
E3_WEIGHT_KEYS = {"logit_kd": "lambda_logit", "cwd_feat": "alpha_cwd", "cwd_logit": "beta_cwd"}
NMF_POLICY = "M4-KD"
OBJECTIVE = ("supervised CE (E1 class weights) + Dice + E3's three distillation terms at 0.5x their E3 weights, T "
             "unchanged, each ramped linearly over the first epoch (AM-3; DL-39)")
RAMP_RULE = ("r(step) = clamp((step - 1)/(ramp_iters - 1), 0, 1) over the global QAT step, ramp_iters = len(TRAIN "
             "loader), one ramp per term; CE + Dice never ramped (DL-39; train_distill.distill_ramp)")
STEP1_RULE = "step1_checks"
GENERIC_RULE = "generic_abort (DL-04 (a); DL-51)"
EXCEPTION_RULE = "exception"
ABORT_KIND = "generic (DL-04 (a); DL-51): AM-7 (b) not applicable; no AM-7 consequence"
ABORT_RULE_TEXT = ("DL-04 (a), DL-51: a non-finite total loss before backward, or a non-finite pre-clip grad_norm "
                   "before optimizer.step, writes the step's row and a run_abort record; step 1 is step1_checks "
                   "(R8-1); cause from the input and teacher finiteness: input_nonfinite, teacher_nonfinite, "
                   "student_nonfinite; an exception is a run_abort with cause exception; no AM-7 (b) monitor")
PROJECTION_SOURCE_NOTE = "E3's trained projection.pt weights, paired with E3's best.json checkpoint (DL-51)"
INJECTIONS = ("loss_nan", "input_nan", "teacher_nan", "grad_inf", "raise_at", "state_nan", "check_fail")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The code paths whose cleanliness at HEAD the launch requires and whose sha256 run_meta records (LB26).
E6KD_CODE_PATHS = (
    *Q.QAT_CODE_PATHS,
    # the lane's files
    "src/quant/qat_kd.py", "scripts/run_e6kd.py", "scripts/e6kd_eval.py", "configs/e6kd_selection_rules.json",
    # the trigger gate (L-AM3 B-1)
    "src/quant/e6kd_trigger.py",
    # train_distill and the modules it loads at import that QAT_CODE_PATHS does not hold
    "src/training/train_distill.py", "configs/distill.py", "src/eval/evaluate.py",
    # the src/distill modules E6-KD imports
    "src/distill/teacher.py", "src/distill/segnext_teacher.py", "src/distill/nmf_stream.py",
    "src/distill/cwd_projection.py", "src/distill/features.py",
    # the teacher config of record and the module it imports (custom_imports), with that module's at-load imports
    # that QAT_CODE_PATHS does not hold (DL-24 G1)
    "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py", "src/training/teacher_components.py",
    "src/eval/artifacts.py", "src/eval/protocols.py",
    # code_identity's src/quant/ptq.py loads it at import, and QAT_CODE_PATHS does not hold it (DL-24 G1)
    "src/eval/model_loading.py",
    # gates 9 and 16: the values of record (DL-87, DL-88), loaded by file location
    "scripts/preflight_distill.py", "scripts/preflight_e1_trainval.py", "scripts/preflight_e1.py",
)


class E6KDAborted(RuntimeError):
    """A run stopped under DL-04 (a), R8-1 or by an exception; `record` is the run_abort row, the telemetry's last."""

    def __init__(self, record: dict):
        super().__init__(f"{record['rule']} at step {record['step']} (cause {record['cause']})")
        self.record = record


def _refuse(code: str, message: str):
    raise QATRefused(code, message)


_PD: dict = {}


def preflight_distill():
    """scripts/preflight_distill.py, loaded once by file location (its own pattern for the E1 gate): the KD image and
    the teacher of record with their values (DL-87, DL-88). Only its definitions run."""
    if "mod" not in _PD:
        spec = importlib.util.spec_from_file_location("preflight_distill", str(REPO / "scripts" / "preflight_distill.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PD["mod"] = mod
    return _PD["mod"]


# ------------------------------------------------------------------ the weights, the cause and the clip
def e6kd_weights(e3_weights: dict) -> dict:
    """AM-3: each E3 weight times 0.5, exactly (the factor is a power of two)."""
    return {t: WEIGHT_FACTOR * float(e3_weights[t]) for t in TERMS}


def nonfinite_rule(step: int) -> str:
    """R8-1 (DL-04): a non-finite total loss or grad_norm at step 1 is a step-1 check failure; from step 2 it is
    DL-04 (a)'s generic abort (DL-51), never AM-7 (a) (LB16)."""
    return GENERIC_RULE if step >= 2 else STEP1_RULE


def abort_cause(rule: str, *, input_finite: bool, teacher_finite: bool) -> str:
    """DL-04's cause rule, for E6-KD (LB16): step1_checks is checks_failed; the generic abort reads the input and the
    teacher outputs, and a non-finite step with both finite is student_nonfinite. E6-KD has no divergence rule."""
    if rule == STEP1_RULE:
        return "checks_failed"
    if not input_finite:
        return "input_nonfinite"
    if not teacher_finite:
        return "teacher_nonfinite"
    return "student_nonfinite"


def _grouped(params) -> tuple[list, dict]:
    from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
    grads = [p.grad for p in params if p.grad is not None]
    return grads, (_group_tensors_by_device_and_dtype([[g.detach() for g in grads]]) if grads else {})


def group_norm(params, device=None) -> torch.Tensor:
    """One group's gradient norm, as torch 2.1's clip_grad_norm_ computes its total over one parameter list
    (torch/nn/utils/clip_grad.py): the 2-norms by device and dtype, foreach where supported, then the 2-norm of
    their stack. A group without gradients has norm +0.0."""
    from torch.utils._foreach_utils import _has_foreach_support
    grads, grouped = _grouped(params)
    if not grads:
        return torch.zeros((), device=device)
    norms = []
    for (dev, _), ([gs], _) in grouped.items():
        if _has_foreach_support(gs, device=dev):
            norms.extend(torch._foreach_norm(gs, 2.0))
        else:
            norms.extend([torch.linalg.vector_norm(g, 2.0) for g in gs])
    return torch.linalg.vector_norm(torch.stack([n.to(grads[0].device) for n in norms]), 2.0)


def grad_norms(student_params, projection_params) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pre-clip global norm over every trainable parameter (DL-04 (a)), in two levels: (total, student,
    projection). With zero projection gradients the total is the student's norm, bitwise."""
    n_s = group_norm(student_params)
    n_p = group_norm(projection_params, device=n_s.device)
    return torch.linalg.vector_norm(torch.stack([n_s, n_p.to(n_s.device)]), 2.0), n_s, n_p


def apply_clip(student_params, projection_params, total: torch.Tensor, max_norm: float) -> None:
    """torch's coefficient, clamp(max_norm / (total + 1e-6), max=1), applied to both groups' gradients."""
    from torch.utils._foreach_utils import _has_foreach_support
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for params in (student_params, projection_params):
        _grads, grouped = _grouped(params)
        for (dev, _), ([gs], _) in grouped.items():
            if _has_foreach_support(gs, device=dev):
                torch._foreach_mul_(gs, coef.to(dev))
            else:
                c = coef.to(dev)
                for g in gs:
                    g.detach().mul_(c)


# ------------------------------------------------------------------ the optimizer
def kd_optimizer(prepared: nn.Module, projection: nn.Module) -> torch.optim.Optimizer:
    """The QAT optimizer of record (AM-4a item 1) over the student, plus the projection's fresh group under the same
    values (DL-51). SGD steps each group on its own, so the student's group is E6's single group."""
    common = dict(lr=Q.LEARNING_RATE, momentum=Q.MOMENTUM, weight_decay=Q.WEIGHT_DECAY, dampening=Q.DAMPENING,
                  nesterov=Q.NESTEROV)
    return torch.optim.SGD([{"params": list(prepared.parameters()), "name": "student"},
                            {"params": list(projection.parameters()), "name": PROJECTION_GROUP}], **common)


def kd_optimizer_record(optimizer, prepared: nn.Module, projection: nn.Module) -> dict:
    """Q2's optimizer record of the student's group, plus both groups and the projection group's freshness."""
    groups = optimizer.param_groups
    rec = Q.optimizer_record(optimizer, prepared)
    rec["param_groups"] = len(groups)
    rec["groups"] = [{"name": g.get("name"), "params": len(g["params"]), "lr": g["lr"], "initial_lr": g.get("initial_lr"),
                      "momentum": g["momentum"], "weight_decay": g["weight_decay"], "dampening": g["dampening"],
                      "nesterov": g["nesterov"]} for g in groups]
    proj_ids = {id(p) for p in projection.parameters()}
    rec["projection_group_holds_the_projection"] = (len(groups) == 2 and {id(p) for p in groups[1]["params"]}
                                                    == proj_ids)
    rec["projection_state_loaded"] = any(id(p) in proj_ids and st for p, st in optimizer.state.items())
    return rec


def kd_optimizer_error(rec: dict) -> str | None:
    want_group = {"lr": Q.LEARNING_RATE, "initial_lr": Q.LEARNING_RATE, "momentum": Q.MOMENTUM,
                  "weight_decay": Q.WEIGHT_DECAY, "dampening": Q.DAMPENING, "nesterov": Q.NESTEROV}
    wrong = {}
    for k, v in {"class": "SGD", "param_groups": 2, "group_holds_every_model_param": True,
                 "projection_group_holds_the_projection": True, "projection_state_loaded": False}.items():
        if rec.get(k) != v or type(rec.get(k)) is not type(v):
            wrong[k] = rec.get(k)
    names = [g.get("name") for g in rec.get("groups") or []]
    if names != ["student", PROJECTION_GROUP]:
        wrong["groups.name"] = names
    for g in rec.get("groups") or []:
        bad = {k: g.get(k) for k, v in want_group.items() if g.get(k) != v or type(g.get(k)) is not type(v)}
        if bad:
            wrong[f"groups.{g.get('name')}"] = bad
    return None if not wrong else f"the optimizer differs from the recipe of record (AM-4a item 1; DL-51): {wrong}"


# ------------------------------------------------------------------ the launch gates' parts
def _run_meta_row(raw: bytes) -> dict:
    """The run_meta row of the bytes resolve_parent hashed, read as Q._run_meta_rows reads that file (exactly one row
    there): the lines holding '"run_meta"' that decode, the one whose event is run_meta (DL-24 T3)."""
    rows = []
    for ln in io.StringIO(raw.decode("utf-8"), newline=None):
        if '"run_meta"' not in ln:
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "run_meta":
            rows.append(row)
    return rows[0]


def e3_record(parent: dict) -> dict:
    """Gate 14: E3's terms and weights from the parent's run_meta row, the file resolve_parent hashed. All three terms
    on; λ and α are the selections' (e6_parent_binding); β, both T and C are configs/distill.py's; the Logit-KD
    semantics is the code's."""
    p = Path(parent["run_dir"]) / parent["run_meta_file"]
    raw = p.read_bytes()
    if Q.sha256_bytes(raw) != parent["run_meta_sha256"]:
        _refuse("parent_run_meta_changed", f"{p.name} changed after the parent was read: sha256 "
                                           f"{Q.sha256_bytes(raw)}, read as {parent['run_meta_sha256']}")
    meta = _run_meta_row(raw)
    terms = {t: True for t in TERMS}
    if meta.get("terms") != terms:
        _refuse("parent_terms", f"the parent's run_meta has terms {meta.get('terms')!r}; E6-KD takes E3's instantiated "
                                f"terms, {terms} (DL-39)")
    e3w = {t: meta.get(E3_WEIGHT_KEYS[t]) for t in TERMS}
    temps = {"T_logit": meta.get("T_logit"), "T_cwd": meta.get("T_cwd"), "cwd_C": meta.get("cwd_C")}
    want = {"beta_cwd": TD.BETA_CWD_LOGIT, "T_logit": TD.T_LOGIT, "T_cwd": TD.T_CWD, "cwd_C": TD.CWD_C_FEAT,
            "logit_kd_semantics": LOGIT_KD_SEMANTICS}
    got = {"beta_cwd": e3w["cwd_logit"], **temps, "logit_kd_semantics": meta.get("logit_kd_semantics")}
    bad = {k: got[k] for k, v in want.items() if got[k] != v or isinstance(got[k], bool)}
    bad.update({E3_WEIGHT_KEYS[t]: v for t, v in e3w.items()
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0})
    if bad:
        _refuse("parent_weights", f"the parent's run_meta has {bad}; E3's of record: β {TD.BETA_CWD_LOGIT}, T_logit "
                                  f"{TD.T_LOGIT}, T_cwd {TD.T_CWD}, C {TD.CWD_C_FEAT}, semantics {LOGIT_KD_SEMANTICS}")
    return {"run_meta_sha256": parent["run_meta_sha256"], "terms": terms, "e3_weights": e3w, **temps,
            "logit_kd_semantics": meta["logit_kd_semantics"], "ramp_iters": meta.get("ramp_iters"),
            "grids": {k: meta.get(k) for k in ("logit_kd_grid", "cwd_feat_grid", "cwd_logit_grid")}}


def projection_of_record(parent: dict, expect_sha256) -> tuple[nn.Module, dict]:
    """Gate 15 (DL-51; LB14): <the parent's run dir>/projection.pt with the pinned sha256, read weights_only from the
    hashed bytes; format cwd_projection/1, stage E3, its checkpoint best.json's file and its iter that checkpoint's;
    160 -> 320, 51,200 parameters, training_only. Built without a global RNG draw (skip_init) and loaded strictly; its
    optimizer state is not read."""
    from .checkpoint import SourceCheckpointInvalid, load_e3_source_checkpoint
    if not isinstance(expect_sha256, str) or not _HEX64.match(expect_sha256):
        _refuse("projection_sha256_format", "--projection-sha256 must be 64 lowercase hex characters")
    path = Path(parent["run_dir"]) / PROJECTION_FILE
    if not path.is_file():
        _refuse("projection_missing", f"{path} is missing: E6-KD's projection is E3's projection.pt (DL-51)")
    raw = path.read_bytes()
    if Q.sha256_bytes(raw) != expect_sha256:
        _refuse("projection_sha256_mismatch", f"{path} has sha256 {Q.sha256_bytes(raw)}, expected {expect_sha256}")
    try:
        doc = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as e:                                   # noqa: BLE001 -- a refusal, named
        _refuse("projection_unreadable", f"{path} does not load with weights_only: {type(e).__name__}: {e}")
    if not isinstance(doc, dict) or doc.get("format") != PROJECTION_FORMAT:
        _refuse("projection_format", f"{path} is not a {PROJECTION_FORMAT} file")
    if doc.get("stage") != "E3":
        _refuse("projection_stage", f"{path} was saved by stage {doc.get('stage')!r}, not E3")
    if doc.get("checkpoint") != parent["checkpoint"]:
        _refuse("projection_unpaired_checkpoint", f"{path} pairs with {doc.get('checkpoint')!r}; best.json names "
                                                  f"{parent['checkpoint']!r} (DL-51: pairing asserted)")
    try:
        ck = load_e3_source_checkpoint(parent["checkpoint_path"])
    except SourceCheckpointInvalid as e:
        _refuse("parent_checkpoint_invalid", f"the E3 checkpoint {parent['checkpoint']}: {e}")
    if type(doc.get("iter")) is not int or type(ck.get("iter")) is not int or doc["iter"] != ck["iter"]:
        _refuse("projection_unpaired_iter", f"{path} was saved at iter {doc.get('iter')!r}; the checkpoint "
                                            f"{parent['checkpoint']} holds iter {ck.get('iter')!r}")
    sd = doc.get("projection_state_dict")
    w = sd.get("weight") if isinstance(sd, dict) else None
    shape_ok = (isinstance(sd, dict) and sorted(sd) == ["weight"] and torch.is_tensor(w)
                and tuple(w.shape) == (TEACHER_STRIDE16_CH, STUDENT_C5_CH, 1, 1) and w.dtype == torch.float32
                and doc.get("in_channels") == STUDENT_C5_CH and doc.get("out_channels") == TEACHER_STRIDE16_CH
                and doc.get("params") == PROJECTION_PARAMS and doc.get("training_only") is True)
    if not shape_ok:
        _refuse("projection_shape", f"{path} is not E3's 1x1 projection {STUDENT_C5_CH} -> {TEACHER_STRIDE16_CH} "
                                    f"({PROJECTION_PARAMS} parameters, training_only)")
    projection = torch.nn.utils.skip_init(nn.Conv2d, STUDENT_C5_CH, TEACHER_STRIDE16_CH, 1, bias=False)
    projection.load_state_dict(sd, strict=True)
    return projection, {"path": str(path.resolve()), "sha256": expect_sha256, "format": PROJECTION_FORMAT,
                        "stage": "E3", "iter": doc["iter"], "checkpoint": doc["checkpoint"], "paired": True,
                        "optimizer_state_loaded": False, "in_channels": STUDENT_C5_CH,
                        "out_channels": TEACHER_STRIDE16_CH, "params": PROJECTION_PARAMS, "note": PROJECTION_SOURCE_NOTE}


def teacher_record(frozen, ckpt: str) -> dict:
    """The teacher of record's identity, as scripts/preflight_distill.py teacher_identity_child records it."""
    model = frozen.teacher
    return {"ok": True, "sha256": frozen.provenance.ckpt_sha256, "bytes": os.path.getsize(ckpt),
            "provenance": frozen.provenance.as_dict(), "params": int(sum(p.numel() for p in model.parameters())),
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "training_modules": sorted(n for n, m in model.named_modules() if m.training)[:10],
            "state_entries": len(model.state_dict())}


def teacher_of_record(path, expect_sha256, *, builder=None):
    """Gate 16 (DL-88): the teacher file named, its sha256 the teacher of record's, compared before the file is read,
    built from the default teacher config and loaded strictly on the CPU (src/distill/teacher.py
    load_frozen_teacher), then its identity checked as scripts/preflight_distill.py checks it. The path has E-31's
    form, as the KD gate checks it (scripts/preflight_distill.py): absolute, outside the repository and named
    iter_24000.pth, as given and as resolved (the loader records the resolved path, whose name check-run-meta reads)."""
    pd = preflight_distill()
    if path and (Q.names_test(path) or Q.names_test(Path(path).expanduser())):
        _refuse("teacher_forbidden_path", f"--teacher-ckpt {path!r}: its given, expanded or resolved form holds a name "
                                          "SL-1 forbids; refused before any filesystem call")
    if not isinstance(expect_sha256, str) or not _HEX64.match(expect_sha256):
        _refuse("teacher_sha256_format", "--teacher-ckpt-sha256 must be 64 lowercase hex characters")
    if expect_sha256 != pd.TEACHER_SHA256:
        _refuse("teacher_not_of_record", f"--teacher-ckpt-sha256 {expect_sha256} is not the teacher of record's "
                                         f"{pd.TEACHER_SHA256} (DL-88)")
    p = Path(path) if path else None
    resolved = p.resolve() if p is not None else None
    if (p is None or not os.path.isabs(path) or pd._under(p, REPO)
            or p.name != pd.TEACHER_FILE or resolved.name != pd.TEACHER_FILE):
        _refuse("teacher_ckpt_path", f"--teacher-ckpt {path!r} (resolved {str(resolved)!r}) must be an absolute path "
                                     f"outside the repository named {pd.TEACHER_FILE} (E-31; DL-88)")
    if not p.is_file():
        _refuse("teacher_missing", f"--teacher-ckpt {path!r} is not a file (DL-39: the teacher of record in the loop)")
    try:
        frozen = TT.load_frozen_teacher(str(p), builder=builder, config_path=str(TD.DEFAULT_TEACHER_CONFIG),
                                        expected_sha256=expect_sha256)
    except TT.TeacherChecksumMismatch as e:
        _refuse("teacher_sha256_mismatch", str(e))
    except TeacherStateDictMismatch as e:
        _refuse("teacher_state_dict_mismatch", str(e))
    except TT.TeacherStackMissing as e:
        _refuse("teacher_stack_missing", str(e))
    ckpt = str(p.resolve())
    rec = teacher_record(frozen, ckpt)
    bad = pd.teacher_identity_problems(rec, ckpt)
    if bad:
        _refuse("teacher_identity", f"the teacher differs from the teacher of record (DL-88): {bad[:6]}")
    return frozen, rec


def code_identity() -> dict:
    from .ptq import code_provenance
    prov = code_provenance(E6KD_CODE_PATHS)
    return {"git_head": prov["commit"], "code_clean_at_head": prov["clean_at_commit"],
            "code_files_sha256": prov["files_sha256"]}


# ------------------------------------------------------------------ the run
class E6KDHooks(Q.QATHooks):
    """Smoke-only observation points (src/quant/qat.py QATHooks); a real run refuses any."""


def _finite(t) -> bool:
    return bool(torch.isfinite(t).all())


def run_qat_kd(*, mode: str, model: nn.Module, teacher, projection: nn.Module, weights: dict, e3: dict,
               source_meta: dict, out_dir, seed: int, clip_norm: float, clip_source: str, device: str,
               num_workers: int, projection_record: dict | None = None, loaders=None, parent: dict | None = None,
               extra_meta: dict | None = None, hooks: Q.QATHooks | None = None, inject: dict | None = None,
               max_steps: int | None = None, max_val_batches: int | None = None, log=print) -> dict:
    """Train E6-KD under E6's recipe (AM-4a) with E3's terms at `weights`. Returns a summary; raises QATRefused (nothing
    written), QATStop, or E6KDAborted after the run_abort record.

    `mode` "real" is the launch of record (main() has passed every gate): `weights` must be 0.5x `e3`'s. `mode` "smoke"
    is the smokes' path: injected loaders, hooks, the injections of INJECTIONS, a step cap and a VAL cap, and any
    weights (d4 runs them at 0), none of which a real run accepts.
    """
    from src.seeds import set_seed
    from src.training.losses import CombinedCEDiceLoss
    from src.training.train_e1 import CLASS_WEIGHTS_JSON, _image_digest, load_ce_weights, validate

    st = resolve_quant_stage(STAGE_KEY)
    if st["method"] != METHOD:
        _refuse("stage_not_qat_kd", f"{st['name']} is registered with method {st['method']!r}, not {METHOD!r}")
    if mode not in Q.MODES:
        _refuse("mode_invalid", f"mode {mode!r} not in {Q.MODES}")
    unknown = sorted(set(inject or {}) - set(INJECTIONS))
    if unknown:
        raise ValueError(f"unknown injections {unknown}; the smokes' injections are {INJECTIONS}")
    inj = {k: {int(s) for s in (inject or {}).get(k, ())} for k in INJECTIONS}
    if mode == "real" and (loaders is not None or hooks is not None or any(inj.values()) or max_steps is not None
                           or max_val_batches is not None):
        _refuse("real_run_overrides", "a real run takes no injected loaders, hooks, injections, step cap or VAL cap")
    if mode == "real" and clip_source != "clip_selection":
        _refuse("clip_source", f"E6-KD's clip is the U4 winner bound at launch (AM-4a item 3), not {clip_source!r}")
    pins = Q.config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    if isinstance(clip_norm, bool) or not isinstance(clip_norm, float) or not math.isfinite(clip_norm) \
            or clip_norm <= 0:
        _refuse("grad_clip_norm_invalid", f"clip norm must be a positive finite float, got {clip_norm!r}")
    if sorted(weights) != sorted(TERMS) or any(isinstance(v, bool) or not isinstance(v, float)
                                               or not math.isfinite(v) for v in weights.values()):
        _refuse("weights_invalid", f"weights {weights!r}: one finite float per term {TERMS}")
    if mode == "real" and (weights != e6kd_weights(e3["e3_weights"])
                           or any(2.0 * weights[t] != e3["e3_weights"][t] for t in TERMS)):
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
        _refuse("cuda_required", f"a real E6-KD run needs a CUDA device, got {device!r}")
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
    teacher_nmf = teacher.begin_nmf_stream(NMF_POLICY, M4_NMF_SEED)
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
    optimizer = kd_optimizer(prepared, projection)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=Q.ETA_MIN)
    opt_rec = kd_optimizer_record(optimizer, prepared, projection)
    opt_err = kd_optimizer_error(opt_rec)
    if opt_err:
        raise QATStop("optimizer_of_record", opt_err)
    ramps = TD.build_term_ramps(E3_STAGE, ramp_iters)

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
                      "eta_min": scheduler.eta_min, "step": "once after every optimizer.step"},
        "optimizer": opt_rec, "lr": opt_rec["lr"], "momentum": opt_rec["momentum"],
        "weight_decay": opt_rec["weight_decay"], "nesterov": opt_rec["nesterov"],
        "dampening": opt_rec["dampening"],
        "bn_freeze_epoch": Q.BN_FREEZE_EPOCH, "obs_freeze_epoch": Q.OBS_FREEZE_EPOCH,
        "bn_frozen_from_epoch": Q.BN_FREEZE_EPOCH + 1, "obs_frozen_from_epoch": Q.OBS_FREEZE_EPOCH + 1,
        "bn_freeze_after_step": bn_after_step, "obs_freeze_after_step": obs_after_step,
        "objective": OBJECTIVE, "class_weights_sha256": Q.sha256_file(CLASS_WEIGHTS_JSON),
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
        # E6-KD (AM-3; DL-07, DL-39, DL-51; lane 7(c)); no cwd_projection_loaded: this run trains the projection
        "e6kd": True, "weight_factor": WEIGHT_FACTOR, "descriptive": True, "holm_family": False,
        "terms": {t: True for t in TERMS}, "e3_weights": dict(e3["e3_weights"]), "weights": dict(weights),
        "T_logit": TD.T_LOGIT, "T_cwd": TD.T_CWD, "cwd_C": TD.CWD_C_FEAT,
        "logit_kd_semantics": LOGIT_KD_SEMANTICS, "grids": dict(e3.get("grids") or {}),
        "ramp_iters": ramp_iters, "ramp_rule": RAMP_RULE,
        "projection_source": projection_record, "projection_params": sum(p.numel() for p in projection_params),
        "projection_fake_quant_modules": len(proj_fq), "projection_in_student": False,
        "teacher_provenance": None if teacher.provenance is None else teacher.provenance.as_dict(),
        "teacher_nmf": teacher_nmf, "teacher_mock": teacher_mock,
        "abort_rule": ABORT_RULE_TEXT,
    }
    if extra_meta:
        run_meta.update(extra_meta)
    out.mkdir(parents=True, exist_ok=True)
    tel = Q.Telemetry(tel_path)
    tel.write(run_meta)
    if hooks is not None:
        hooks.on_start(prepared, optimizer, scheduler, run_meta)
    log(f"[e6kd] {st['name']} seed {seed} clip {clip_norm} | weights {weights} | {steps_per_epoch} steps/epoch, "
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
        """One finiteness reading for the run_abort record. On the exception path (guarded) a reading the device
        cannot answer, after a sticky CUDA fault, is None, so that the record is still written; a generic abort's
        readings stay unguarded, since abort_cause reads a None as non-finite (DL-24 T1)."""
        try:
            return fn()
        except Exception:                                    # noqa: BLE001 -- a device that cannot answer: None
            if not guarded:
                raise
            return None

    def abort_record(rule: str, *, quantity, detail: dict, failed=None, exc: BaseException | None = None) -> dict:
        """The run_abort record (DL-04 (a); LB16), written as the telemetry's last row."""
        img, to = cur["img"], cur["teacher_out"]
        guarded = rule == EXCEPTION_RULE
        input_finite = None if img is None else reading(lambda: _finite(img), guarded)
        teacher_finite = None if to is None else reading(lambda: _finite(to.logits)
                                                         and (to.feat_s16 is None or _finite(to.feat_s16)), guarded)
        cause = ("exception" if rule == EXCEPTION_RULE else
                 abort_cause(rule, input_finite=bool(input_finite), teacher_finite=bool(teacher_finite)))
        try:
            state_ok, state_fails = Q.state_predicate(Q.cpu_state(prepared), kinds, never)
        except Exception:                                    # noqa: BLE001 -- the reading is recorded as missing
            state_ok, state_fails = None, []
        record = {"event": "run_abort", "wall_clock": time.time(), "epoch": epoch, "step": step, "rule": rule,
                  "cause": cause, "nonfinite_quantity": quantity,
                  "detail": {k: TD.record_number(detail.get(k)) for k in
                             ("loss", "grad_norm", "grad_norm_student", "grad_norm_projection")},
                  "input_finite": input_finite, "teacher_finite": teacher_finite,
                  "params_finite": {"student": reading(lambda: all(_finite(p) for p in student_params), guarded),
                                    "projection": reading(lambda: all(_finite(p) for p in projection_params),
                                                          guarded)},
                  "state_finite": state_ok, "state_failing": state_fails[:5],
                  "state_nonfinite": nonfinite_since is not None or state_ok is False,
                  "nonfinite_since_step": nonfinite_since, "abort_kind": ABORT_KIND}
        if failed:
            record["failed_checks"] = failed
        if exc is not None:
            record.update({"exception_type": type(exc).__name__, "message": str(exc)[:500]})
        tel.write(record)
        return record

    def abort(rule: str, **kw) -> None:
        raise E6KDAborted(abort_record(rule, **kw))

    def train_row(*, lr, epoch_step, loss_v, ce_v, dice_v, sup_v, parts, ramp, norms, clipped, fp, t0) -> dict:
        row = {"event": "train", "wall_clock": time.time(), "epoch": epoch, "step": step, "epoch_step": epoch_step,
               "lr": lr, "loss": loss_v, "ce": ce_v, "dice": dice_v, "sup": sup_v,
               **{t: parts.get(t) for t in TERMS}, "ramp": ramp,
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
                ramp = ramp_by_term[TERMS[0]]               # every term runs the same first-epoch schedule
                ce = criterion.ce(logits, mask_d)
                dice = criterion.dice(logits, mask_d)
                sup = ce + dice
                distill, parts = TD.distillation_losses(
                    stage=E3_STAGE, logits=logits, head_logits=head_logits, c5=c5, mask=mask_d,
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
                        "has_expected_terms": set(parts) == set(TERMS),
                        "loss_finite": _finite(loss) and loss.dim() == 0,
                    })
                    if ramp_iters > 1:
                        checks.update({"distill_ramp_starts_at_zero": all(v == 0.0 for v in ramp_by_term.values()),
                                       "distill_zero_at_step1": float(distill) == 0.0,
                                       "sup_added_unscaled": bool(torch.equal(loss, sup)),
                                       "sup_positive_finite": _finite(sup) and float(sup) > 0.0})
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
                    abort(nonfinite_rule(step), quantity="loss", detail={"loss": loss_v})
                loss.backward()
                if step in inj["grad_inf"]:                  # smoke only: a non-finite gradient, finite loss
                    student_params[0].grad.view(-1)[0] = float("inf")
                total, n_s, n_p = grad_norms(student_params, projection_params)
                norms = (float(total.item()), float(n_s.item()), float(n_p.item()))
                # DL-04 (a): a non-finite pre-clip grad_norm stops the run before optimizer.step
                if not math.isfinite(norms[0]):
                    train_row(**row_args, norms=norms, clipped=None)
                    abort(nonfinite_rule(step), quantity="grad_norm",
                          detail={"loss": loss_v, "grad_norm": norms[0], "grad_norm_student": norms[1],
                                  "grad_norm_projection": norms[2]})
                apply_clip(student_params, projection_params, total, clip_norm)
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
                        abort(STEP1_RULE, quantity=None, failed=failed,
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
            log(f"[e6kd] epoch {epoch:02d}/{Q.EPOCHS} step {step} fake-quant VAL {all_miou}"
                f"{' (non-finite state)' if nonfinite_since is not None else ''}")
            if max_steps is not None and step >= max_steps:
                break
    except (E6KDAborted, QATRefused):
        raise
    except QATStop as e:
        tel.write({"event": "run_stop", "wall_clock": time.time(), "epoch": epoch, "step": step,
                   "code": e.code, "message": str(e)[:500], "state_nonfinite": nonfinite_since is not None})
        raise
    except BaseException as e:                              # the run_abort record, then re-raised (Q2's form)
        abort_record(EXCEPTION_RULE, quantity=None, detail={}, exc=e)
        raise
    complete = epoch == Q.EPOCHS and max_steps is None
    nmf = teacher.nmf_stream_state()
    return {"run_id": run_id, "stage": st["name"], "seed": seed, "clip_norm": clip_norm,
            "telemetry": str(tel_path), "steps": step, "epochs_completed": epoch, "complete": complete,
            "nonfinite_since_step": nonfinite_since, "teacher_nmf_draws": None if nmf is None else nmf["draws"]}


# ------------------------------------------------------------------ CLI (scripts/run_e6kd.py)
def build_parser(stage: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{stage['name']}: {stage['objective']} (AM-3; DL-39, DL-51; the stage "
                                            "is pinned by the entry point)")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--source-run-dir", default=None)
    p.add_argument("--expect-source-sha256", default=None)
    p.add_argument("--expect-head", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--clip-selection", default=None)
    p.add_argument("--clip-selection-sha256", default=None)
    p.add_argument("--lambda-selection", default=None)
    p.add_argument("--lambda-selection-sha256", default=None)
    p.add_argument("--alpha-selection", default=None)
    p.add_argument("--alpha-selection-sha256", default=None)
    p.add_argument("--trigger", default=None)
    p.add_argument("--trigger-sha256", default=None)
    p.add_argument("--projection-sha256", default=None)
    p.add_argument("--teacher-ckpt", default=None)
    p.add_argument("--teacher-ckpt-sha256", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda")
    return p


def real_run_gates(args, stage: dict, *, repo_root: Path = REPO, teacher_builder=None) -> dict:
    """Gates 1-16 of the E6-KD launch, in order (plan §1.3 as LB14-LB38 amend it); main() then loads the source
    (Q.load_source, the rest of gate 16) and run_qat_kd makes its own checks (gate 17). None writes anything.
    `teacher_builder` is the smokes' stub (src/distill/teacher.py load_frozen_teacher's builder); a launch passes
    none, so the teacher is built from the thesis teacher config."""
    # 1. the flags
    if not args.real_run:
        _refuse("real_run_flag", f"{stage['name']} requires --real-run --confirm-real-run")
    if not args.confirm_real_run:
        _refuse("confirm_real_run_flag", "--real-run requires --confirm-real-run")
    # 2. configs/quant.py holds AM-4a's recipe, which E6-KD repeats (LB22)
    pins = Q.config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    # 3. the head's form, seed 42 only (AM-3), loader workers
    if not args.expect_head or not _HEX40.match(args.expect_head):
        _refuse("expect_head_format", "--expect-head must be the 40-hex commit of the checkout")
    if args.seed != SEED:
        _refuse("seed_not_42", f"--seed {args.seed!r}: E6-KD runs at seed {SEED} only (AM-3)")
    if args.num_workers is None or args.num_workers < 1:
        _refuse("num_workers", "--num-workers must be given and at least 1")
    # 4. the clip: clip_selection.json's U4 winner (AM-4a item 3)
    clip = Q.clip_binding(stage["name"], args.seed, args.grad_clip_norm, u4_pilot=False,
                          clip_selection=args.clip_selection, clip_selection_sha256=args.clip_selection_sha256)
    # 5. the λ and α selections, tracked and unchanged at HEAD
    lam = Q.read_tracked_selection(args.lambda_selection, args.lambda_selection_sha256, Q.LAMBDA_SELECTION_FORMAT,
                                   "lambda", repo_root)
    alpha = Q.read_tracked_selection(args.alpha_selection, args.alpha_selection_sha256, Q.ALPHA_SELECTION_FORMAT,
                                     "alpha", repo_root)
    # 6. the committed trigger record, triggered (DL-39; lane 7's STOP condition)
    trigger = read_trigger_record(args.trigger, args.trigger_sha256, repo_root=repo_root)
    # 7. a fresh out dir outside the repository
    err = Q._out_dir_error(args.out_dir)
    if err:
        _refuse("out_dir", err)
    # 8. CUDA available and not yet initialised; TF32 at the torch defaults; QNNPACK present
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        _refuse("cuda_required", f"a real E6-KD run needs a CUDA device (--device {args.device!r}, "
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
    # 9. the KD image of record (DL-87): its digest, and a child's fingerprint equal to the record (PL-26)
    pd = preflight_distill()
    digest = (os.environ.get("PLANTSEG_IMAGE_DIGEST") or "").strip()
    if digest != pd.KD_IMAGE_DIGEST:
        _refuse("image_not_kd", f"PLANTSEG_IMAGE_DIGEST is {digest or None!r}; E6-KD trains in the KD image "
                                f"{pd.KD_IMAGE_DIGEST} (DL-87; LB28)")
    rc, fp, fp_out = pd._child_json([REPO / "scripts" / "preflight_distill.py", "_fingerprint"], "FINGERPRINT_JSON",
                                    600)
    fp_bad = (["the fingerprint child failed: rc=" + str(rc) + " " + fp_out[-200:]] if rc != 0 or fp is None
              else pd.fingerprint_problems(fp))
    if fp_bad:
        _refuse("fingerprint_mismatch", f"the image differs from the KD fingerprint of record (PL-26): {fp_bad[:6]}")
    # 10. the data root holds TRAIN and VAL only
    from configs.data import DATA, SPLIT_SIZES
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    try:
        iso = assert_trainval_only_root(Path(DATA["root"]), {"train": SPLIT_SIZES["train"],
                                                             "val": SPLIT_SIZES["val"]})
    except TrainValIsolationError as e:
        _refuse("data_root_not_trainval_only", f"[{e.code}] {e}")
    # 11. HEAD is the pin, and E6KD_CODE_PATHS are clean at it (LB26)
    ident = code_identity()
    if ident["git_head"] != args.expect_head:
        _refuse("expect_head_mismatch", f"the checkout's HEAD is {ident['git_head']!r}, --expect-head "
                                        f"{args.expect_head}")
    if ident["code_clean_at_head"] is not True:
        _refuse("code_not_clean", "the E6-KD code paths are not clean at HEAD")
    # 12. the parent: E3 seed 42's α winner, best.json's checkpoint (AM-19 item 4(a)), the run the selections name
    parent = Q.resolve_parent(stage, args.source_run_dir, args.expect_source_sha256, args.seed)
    binding = Q.e6_parent_binding(parent, args.seed, lam, alpha)
    # 13. the run the trigger read: that E3 checkpoint, E6's clip of record, the same selections
    bound = trigger_binding(trigger, parent=parent, clip_norm=args.grad_clip_norm,
                            clip_selection_sha256=clip["clip_selection"]["sha256"], lam=lam, alpha=alpha)
    # 14. E3's terms and weights; E6-KD's are half of them
    e3 = e3_record(parent)
    # 15. the projection of record (DL-51; LB14)
    projection, projection_rec = projection_of_record(parent, args.projection_sha256)
    # 16. the teacher of record (DL-88), on the CPU
    teacher, teacher_rec = teacher_of_record(args.teacher_ckpt, args.teacher_ckpt_sha256, builder=teacher_builder)
    return {"clip": clip, "parent": parent, "e6_binding": binding, "trigger": trigger, "trigger_binding": bound,
            "e3": e3, "weights": e6kd_weights(e3["e3_weights"]), "projection": projection,
            "projection_record": projection_rec, "teacher": teacher, "teacher_record": teacher_rec,
            "isolation": iso, "data_root": str(DATA["root"]), "identity": ident, "image_digest": digest,
            "fingerprint": fp}


def trigger_meta(trigger: dict) -> dict:
    doc = trigger["record"]
    return {"path": trigger["path"], "sha256": trigger["sha256"], "e3_val": doc.get("e3_val"),
            "e6_val": doc.get("e6_val"), "drop_pp": doc.get("drop_pp"), "triggered": doc.get("triggered")}


def result_line(summary: dict) -> str:
    tag = (f"{summary['stage']}, seed {summary['seed']}, clip {summary['clip_norm']}, "
           f"{summary['epochs_completed']}/{Q.EPOCHS} epochs")
    if not summary["complete"]:
        return f"RESULT: E6-KD INCOMPLETE ({tag})"
    if summary["nonfinite_since_step"] is not None:            # LB17: recorded, the run continued
        return f"RESULT: E6-KD COMPLETE, STATE NON-FINITE first found at step {summary['nonfinite_since_step']} ({tag})"
    return f"RESULT: E6-KD COMPLETE ({tag})"


def aborted_line(record: dict) -> str:
    if record["rule"] == EXCEPTION_RULE:
        return f"RESULT: ABORTED [exception] -- {record.get('exception_type')}: {record.get('message')}"
    return (f"RESULT: ABORTED rule={record['rule']} step={record['step']} cause={record['cause']} -- "
            f"{record['nonfinite_quantity'] or 'failed step-1 checks ' + str(record.get('failed_checks'))}; the "
            "run_abort record is the telemetry's last row (AM-19 item 3(a) governs any repeat)")


def main(argv, stage_key: str = STAGE_KEY, *, teacher_builder=None) -> int:
    stage = resolve_quant_stage(stage_key)
    args = build_parser(stage).parse_args(argv)
    try:
        gates = real_run_gates(args, stage, teacher_builder=teacher_builder)
        model, source_meta = Q.load_source(stage, gates["parent"])
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return Q.EXIT_REFUSED
    except Exception as e:                                 # noqa: BLE001 -- before the run: nothing written
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. Nothing was written.")
        return Q.EXIT_ERROR
    extra = {"clip_selection": gates["clip"]["clip_selection"], "u4_pilot": False,
             "e6_selections": gates["e6_binding"], "data_root": gates["data_root"],
             "isolation_counts": gates["isolation"].get("counts"),
             "expect_head": args.expect_head, "expect_source_sha256": args.expect_source_sha256,
             "trigger": trigger_meta(gates["trigger"]), "trigger_sha256": gates["trigger"]["sha256"],
             "trigger_binding": gates["trigger_binding"], "teacher_identity": gates["teacher_record"],
             "kd_image": {"image_digest": gates["image_digest"], "fingerprint": gates["fingerprint"]}}
    try:
        summary = run_qat_kd(mode="real", model=model, teacher=gates["teacher"], projection=gates["projection"],
                             weights=gates["weights"], e3=gates["e3"], source_meta=source_meta,
                             out_dir=args.out_dir, seed=args.seed, clip_norm=args.grad_clip_norm,
                             clip_source=gates["clip"]["clip_source"], device=args.device,
                             num_workers=args.num_workers, projection_record=gates["projection_record"],
                             parent=gates["parent"], extra_meta=extra)
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return Q.EXIT_REFUSED
    except QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return Q.EXIT_STOP
    except E6KDAborted as e:
        print(aborted_line(e.record))
        return Q.EXIT_ABORTED
    except Exception as e:                                 # noqa: BLE001 -- after the run_meta row: run_abort
        tel_p = Path(args.out_dir) / Q.TELEMETRY_NAME
        if not (tel_p.is_file() and tel_p.stat().st_size > 0):  # the run_meta row is the first one written
            print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. No run_meta row was written: nothing was launched.")
            return Q.EXIT_ERROR
        print(f"RESULT: ABORTED [exception] -- {type(e).__name__}: {e}")
        return Q.EXIT_ABORTED
    print(result_line(summary))
    return Q.EXIT_OK if summary["complete"] else Q.EXIT_ABORTED


__all__ = ["ABORT_KIND", "E3_STAGE", "E6KDAborted", "E6KD_CODE_PATHS", "E6KDHooks", "EXCEPTION_RULE", "GENERIC_RULE",
           "INJECTIONS", "METHOD", "SEED", "STAGE_KEY", "STAGE_NAME", "STEP1_RULE", "TELEMETRY_SCHEMA", "TERMS",
           "WEIGHT_FACTOR", "abort_cause", "aborted_line", "apply_clip", "build_parser", "code_identity", "e3_record",
           "e6kd_weights", "grad_norms", "group_norm", "kd_optimizer", "kd_optimizer_error", "kd_optimizer_record",
           "main", "nonfinite_rule", "preflight_distill", "projection_of_record", "real_run_gates", "result_line",
           "run_qat_kd", "teacher_of_record", "teacher_record", "trigger_meta"]
