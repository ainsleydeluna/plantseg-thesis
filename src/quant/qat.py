"""E5/E6 INT8 quantization-aware training of record (lane 6: L-AM4 + L-AM1q).

AM-4 and AM-4a item 1 fix the recipe. This module holds those values as constants and refuses a
configs/quant.py that disagrees; no launch flag sets any of them. The one launch value is the
global-norm clipping threshold, which the U4 pilot selects (AM-4a item 3).

    E5 <- E1's best.json checkpoint of the same seed        E6 <- E3's, projection-free (AM-4a item 5)

Recipe (AM-4a item 1)
  * SGD, one parameter group: lr 3e-4, momentum 0.9, weight decay 1e-4, dampening 0, no Nesterov;
    CosineAnnealingLR(T_max = 15 x steps_per_epoch, eta_min 0), stepped once after every optimizer step.
  * physical batch 16 (E1's TRAIN loader, drop_last: 5,367 images -> 335 steps per epoch), no
    accumulation, no EMA; 15 fixed epochs, no early stopping.
  * fake quantization and moving-average observers from the first step; BN statistics frozen at the
    top of epoch 11 and every observer, weight and activation, disabled at the top of epoch 13.
  * supervised CE (E1's class weights) + Dice; global-norm clipping between backward and the step.

Seeding (AM-4a item 4, L-AM1q): set_seed(--seed), before CUDA is initialised; both loaders are E1's
build_dataloader calls with seed=--seed.

Per epoch, in this order: the epoch's steps; epoch_ckpts/eNN.pt written atomically (the pre-convert
QAT state, in the x86 sidecar schema) and hashed; the fake-quant VAL bracket (every FakeQuantize's
observer flag snapshotted, all disabled, train_e1.validate, the snapshot restored), so a VAL image
never moves a range; the state is asserted unchanged; the `val` row, then the `epoch_end` row. The
epoch-15 `epoch_end` row is the AM-19 completion record.

Non-finite states (AM-19 item 3(a)). There is no divergence rule. A non-finite loss or gradient norm
is recorded. Once the state predicate (`state_predicate`) fails, the run keeps drawing batches and
stepping the schedule, marks every later row as non-finite and still writes every checkpoint, VAL row
and epoch record. An exception while the state is finite ends the run with a run_abort row.

Selection never happens here: the conversion and converted VAL scoring of every epoch are
scripts/qat_epoch_eval.py's, and the epoch is chosen by scripts/select_qat_epoch.py.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.quant import QUANT                                             # noqa: E402
from .prepare import (disable_observers, freeze_bn_stats, prepare_qat_model,  # noqa: E402
                      qat_freeze_steps, unfused_batchnorm)
from .qconfig import QUANT_BACKEND, QuantBackendUnavailable, select_qnnpack_backend  # noqa: E402
from .stages import resolve_quant_stage                                       # noqa: E402
from .x86_latency import QAT_SIDECAR_KIND, QAT_SIDECAR_ROLE                   # noqa: E402

LANE = "L-AM4 + L-AM1q (docs/lane_specs/part2.md lane 6)"
TELEMETRY_SCHEMA = "plantseg-qat-telemetry/1.0.0"
TELEMETRY_NAME = "qat_telemetry.jsonl"
EPOCH_DIR = "epoch_ckpts"
NUM_CLASSES = 116
IGNORE_INDEX = 255

# ------------------------------------------------------------------ AM-4 / AM-4a item 1
EPOCHS = 15
BATCH_SIZE = 16
BN_FREEZE_EPOCH = 10             # BN statistics are frozen at the top of epoch 11
OBS_FREEZE_EPOCH = 12            # every observer is disabled at the top of epoch 13
LEARNING_RATE = 3e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
DAMPENING = 0.0
NESTEROV = False
ETA_MIN = 0.0
TRAIN_IMAGES = 5367
REAL_STEPS_PER_EPOCH = 335       # floor(5,367 / 16): E1's TRAIN loader drops the last partial batch
REAL_T_MAX = EPOCHS * REAL_STEPS_PER_EPOCH          # 5,025
REAL_SEEDS = (42, 43, 44)
CLIP_CANDIDATES = (1.0, 5.0)     # configs/quant.py qat_grad_clip_pilot (AM-4a item 3)
U4_STAGE, U4_SEED = "E5", 42
MODES = ("real", "smoke")
CLIP_SOURCES = ("u4_pilot", "clip_selection")
OBSERVER_FREEZE_SCOPE = "all FakeQuantize modules, weight and activation"
OBJECTIVE = "supervised CE (E1 class weights) + Dice; no distillation (E5/E6)"
SCHEDULER_CLASS = "CosineAnnealingLR"
CLIP_SELECTION_FORMAT = "qat_clip_selection/1"
LAMBDA_SELECTION_FORMAT = "lambda_selection/1"
ALPHA_SELECTION_FORMAT = "alpha_selection/1"
IMAGE_DIGEST_OF_RECORD = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
TORCH_OF_RECORD = "2.1.0"
TF32_ENV_KEYS = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
TF32_DEFAULTS = {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                 "float32_matmul_precision": "highest",
                 "NVIDIA_TF32_OVERRIDE": None, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The code paths whose cleanliness at HEAD a real launch requires and whose sha256 its run_meta
# records, through src/quant/ptq.py code_provenance.
QAT_CODE_PATHS = (
    "src/quant/qat.py", "src/quant/qat_artifacts.py", "src/quant/qat_select.py",
    "src/quant/prepare.py", "src/quant/qconfig.py", "src/quant/stages.py", "src/quant/checkpoint.py",
    "src/quant/calibration.py", "src/quant/x86_latency.py", "src/quant/ptq.py", "src/quant/runner.py",
    "src/quant/__init__.py", "src/models/student.py", "src/data/__init__.py", "src/data/dataset.py",
    "src/data/transforms.py", "src/data/isolation.py", "src/training/train_e1.py",
    "src/training/losses.py", "src/seeds.py", "src/eval/metrics.py",
    "configs/quant.py", "configs/qat_selection_rules.json", "configs/data.py", "configs/augment.py",
    "configs/model.py", "configs/e1_student.py", "reports/e1_class_weights.json",
    "scripts/run_e5.py", "scripts/run_e6.py", "scripts/qat_epoch_eval.py",
    "scripts/select_qat_epoch.py", "scripts/select_clip.py",
)

EXIT_OK, EXIT_STOP, EXIT_REFUSED, EXIT_ABORTED, EXIT_ERROR = 0, 1, 2, 3, 4


class QATRefused(RuntimeError):
    """A named refusal before anything was written (exit 2)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class QATStop(RuntimeError):
    """A named STOP during a run (exit 1): an invariant of the recipe failed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _refuse(code: str, message: str):
    raise QATRefused(code, message)


# ------------------------------------------------------------------ small helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _nonfinite_tag(x: float) -> str:
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


def strict_row(row: dict) -> dict:
    """Strict JSON: a top-level non-finite float is written null and named in a `nonfinite` map."""
    bad = {k: _nonfinite_tag(v) for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)}
    if not bad:
        return row
    return {**{k: (None if k in bad else v) for k, v in row.items()}, "nonfinite": bad}


class Telemetry:
    """Append-only JSONL beside the checkpoints. Opened per row, so a killed pod loses no buffered row."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, row: dict) -> dict:
        row = strict_row(row)
        line = json.dumps(row, allow_nan=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return row


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def host_identity() -> dict:
    """Host identifiers: recorded, and never compared across the two pilot runs (two pods of the same
    GPU type may differ in CPU model; gpu_name stays a compared run_meta key)."""
    return {"hostname": socket.gethostname(), "pod_id": os.environ.get("RUNPOD_POD_ID") or None,
            "cpu_model": cpu_model()}


def tf32_state() -> dict:
    """TF32 as torch reports it, plus the two environment overrides. Read here, never set."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                 "float32_matmul_precision": torch.get_float32_matmul_precision()}
    for key in TF32_ENV_KEYS:
        v = os.environ.get(key)
        state[key] = v if (v is not None and v.strip()) else None
    return state


def determinism_state() -> dict:
    return {"deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_algorithms_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def config_pins_error() -> str | None:
    """configs/quant.py must hold exactly the AM-4a values this module holds."""
    want = {("qat", "epochs"): EPOCHS, ("qat", "bn_freeze_epoch"): BN_FREEZE_EPOCH,
            ("qat", "obs_freeze_epoch"): OBS_FREEZE_EPOCH, ("qat", "learning_rate"): LEARNING_RATE,
            ("qat", "momentum"): MOMENTUM,
            ("qat_real_run", "epochs"): EPOCHS, ("qat_real_run", "batch_size_physical"): BATCH_SIZE,
            ("qat_real_run", "bn_freeze_epoch"): BN_FREEZE_EPOCH,
            ("qat_real_run", "obs_freeze_epoch"): OBS_FREEZE_EPOCH,
            ("qat_real_run", "weight_decay"): WEIGHT_DECAY,
            ("qat_real_run", "drop_last_train"): True,
            ("qat_grad_clip_pilot", "candidates"): CLIP_CANDIDATES,
            ("qat_grad_clip_pilot", "pilot_budget_epochs"): EPOCHS}
    bad = []
    for (section, key), value in want.items():
        got = QUANT.get(section, {}).get(key)
        if type(got) is not type(value) or got != value:
            bad.append(f"configs/quant.py {section}.{key} = {got!r}, AM-4a: {value!r}")
    if QUANT.get("qat_real_run", {}).get("gradient_accumulation") is not None:
        bad.append("configs/quant.py qat_real_run.gradient_accumulation must be None")
    if QUANT.get("qat", {}).get("weight_ema") is not False:
        bad.append("configs/quant.py qat.weight_ema must be False")
    if QUANT.get("qat", {}).get("distillation_during_qat") is not False:
        bad.append("configs/quant.py qat.distillation_during_qat must be False")
    return "; ".join(bad) if bad else None


# ------------------------------------------------------------------ module census and the predicate
def _fq_base():
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    return FakeQuantizeBase


def _moving_average_types():
    from torch.ao.quantization.observer import (MovingAverageMinMaxObserver,
                                                MovingAveragePerChannelMinMaxObserver)
    return (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver)


def fake_quant_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    base = _fq_base()
    return [(n, m) for n, m in model.named_modules() if isinstance(m, base)]


def observer_census(model: nn.Module) -> dict:
    out: dict[str, int] = {}
    for _, m in fake_quant_modules(model):
        name = type(m.activation_post_process).__name__
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


def fused_convbn_census(model: nn.Module) -> dict:
    from torch.ao.nn.intrinsic.qat.modules.conv_fused import _ConvBnNd
    out: dict[str, int] = {}
    for _, m in model.named_modules():
        if isinstance(m, _ConvBnNd):
            out[type(m).__name__] = out.get(type(m).__name__, 0) + 1
    return dict(sorted(out.items()))


@torch.no_grad()
def never_observed_modules(prepared: nn.Module, example: torch.Tensor | None = None) -> list[str]:
    """The fake-quant modules a training forward never calls, from forward hooks on a CPU copy."""
    probe = copy.deepcopy(prepared).cpu().train()
    called: set[str] = set()
    hooks = [m.register_forward_hook(lambda mod, inp, out, _n=n: called.add(_n))
             for n, m in fake_quant_modules(probe)]
    try:
        x = example if example is not None else torch.randn(
            2, 3, 64, 64, generator=torch.Generator().manual_seed(20261005))
        probe(x)
    finally:
        for h in hooks:
            h.remove()
    return sorted(n for n, _ in fake_quant_modules(probe) if n not in called)


def state_kinds(prepared: nn.Module) -> dict[str, str]:
    """Classify the state entries the predicate inspects, from the prepared module's structure."""
    ma_types = _moving_average_types()
    fq_base = _fq_base()
    kinds = {name: "parameter" for name, _ in prepared.named_parameters()}
    for mname, mod in prepared.named_modules():
        prefix = f"{mname}." if mname else ""
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            kinds[prefix + "running_mean"] = "bn_buffer"
            kinds[prefix + "running_var"] = "bn_buffer"
        elif isinstance(mod, fq_base):
            kinds[prefix + "scale"] = "fq_scale"
        elif isinstance(mod, ma_types):
            kinds[prefix + "min_val"] = "observer_min"
            kinds[prefix + "max_val"] = "observer_max"
    return kinds


def state_predicate(state: Mapping[str, torch.Tensor], kinds: Mapping[str, str],
                    never_observed: Iterable[str], limit: int = 20) -> tuple[bool, list[str]]:
    """The one state predicate of the trainer, the conversion worker and both selectors.

    Finite iff every parameter and every BN buffer is finite, every fake-quant scale is finite, and every
    moving-average observer holds finite min <= max in every element, or exactly (+inf, -inf) while its
    fake-quant module is one of run_meta's never_observed_modules.
    """
    never = set(never_observed)
    fails: list[str] = []
    for key, kind in kinds.items():
        t = state.get(key)
        if t is None:
            fails.append(f"{key}: missing ({kind})")
            continue
        if kind in ("parameter", "bn_buffer", "fq_scale") and not bool(torch.isfinite(t).all()):
            fails.append(f"{key}: non-finite {kind}")
    for key, kind in kinds.items():
        if kind != "observer_min":
            continue
        obs = key[: -len(".min_val")]
        mn, mx = state.get(key), state.get(obs + ".max_val")
        if mn is None or mx is None:
            continue                                   # reported above as missing
        fq = obs[: -len(".activation_post_process")] if obs.endswith(".activation_post_process") else obs
        finite = bool(torch.isfinite(mn).all()) and bool(torch.isfinite(mx).all())
        if finite and mn.shape == mx.shape and bool((mn <= mx).all()):
            continue
        unobserved = (mn.numel() > 0 and bool((mn == float("inf")).all())
                      and bool((mx == float("-inf")).all()))
        if unobserved and fq in never:
            continue
        fails.append(f"{obs}: observer range not finite min <= max"
                     + (" (never observed, but not a never_observed module)" if unobserved else ""))
    return (not fails), fails[:limit] + ([f"... {len(fails) - limit} more"] if len(fails) > limit else [])


def _hash_tensors(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    """sha256 over (name, dtype, shape, bytes). Hashes a copy: Tensor.numpy() marks a storage
    non-resizable, and an observer buffer that has not yet been sized must stay resizable."""
    h = hashlib.sha256()
    for name, t in items:
        t = t.detach().to("cpu", copy=True).contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


def state_digest(model_or_state) -> str:
    """sha256 over every state entry (name, dtype, shape, bytes), in state_dict order."""
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    return _hash_tensors(state.items())


def bn_buffer_digest(model: nn.Module) -> str:
    items = []
    for mname, mod in model.named_modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            for b in ("running_mean", "running_var", "num_batches_tracked"):
                items.append((f"{mname}.{b}", getattr(mod, b)))
    return _hash_tensors(items)


def observer_buffer_digest(model: nn.Module) -> str:
    """sha256 over every observer buffer and every fake-quant scale and zero_point."""
    items = []
    for mname, fq in fake_quant_modules(model):
        items.append((f"{mname}.scale", fq.scale))
        items.append((f"{mname}.zero_point", fq.zero_point))
        for bname, buf in fq.activation_post_process.named_buffers():
            items.append((f"{mname}.activation_post_process.{bname}", buf))
    return _hash_tensors(items)


def observer_flags(model: nn.Module) -> list[tuple[str, int]]:
    return [(n, int(m.observer_enabled.item())) for n, m in fake_quant_modules(model)]


def restore_observer_flags(model: nn.Module, flags: list[tuple[str, int]]) -> None:
    """Restore a snapshot exactly; the VAL bracket never calls enable_observers."""
    mods = dict(fake_quant_modules(model))
    if set(mods) != {n for n, _ in flags}:
        raise QATStop("observer_snapshot_mismatch", "the fake-quant modules changed inside the VAL bracket")
    for name, value in flags:
        mods[name].observer_enabled[0] = value


def flag_summary(model_or_state) -> dict:
    """The distinct observer_enabled / fake_quant_enabled values over every fake-quant module."""
    if isinstance(model_or_state, nn.Module):
        fq = fake_quant_modules(model_or_state)
        obs = {int(m.observer_enabled.item()) for _, m in fq}
        fqe = {int(m.fake_quant_enabled.item()) for _, m in fq}
    else:
        obs = {int(v.item()) for k, v in model_or_state.items() if k.endswith(".observer_enabled")}
        fqe = {int(v.item()) for k, v in model_or_state.items() if k.endswith(".fake_quant_enabled")}
    return {"observer_enabled": sorted(obs), "fake_quant_enabled": sorted(fqe)}


# ------------------------------------------------------------------ parent and bindings
def read_jsonl(path: Path) -> tuple[list[dict], bool]:
    """Rows of a JSONL file and whether its last line is torn (any other bad line raises)."""
    rows, torn = [], False
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                torn = True
            else:
                raise
    return rows, torn


def _run_meta_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if '"run_meta"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "run_meta":
                rows.append(obj)
    return rows


def resolve_parent(stage: dict, run_dir, expect_sha256: str, seed: int, *,
                   require_real: bool = True) -> dict:
    """AM-19 item 4(a): the parent run directory, its best.json checkpoint and its records.

    E1 (train_e1): e1_telemetry.jsonl holds exactly one run_meta row (mode, seed; no stage key) and the
    VAL row at its max_iters. E3 (train_distill): e3_run_meta.jsonl holds exactly one run_meta row with
    stage "E3", and e3_telemetry.jsonl a run_end record with checks_passed and no run_abort row. In both,
    best.json names a file in the directory whose sha256 equals --expect-source-sha256.
    """
    d = Path(run_dir) if run_dir else None
    if d is not None and "test" in str(d.resolve()).lower():
        _refuse("parent_dir_test_path", f"--source-run-dir {d} contains 'test'")
    if d is None or not d.is_dir():
        _refuse("parent_dir_missing", f"--source-run-dir {run_dir!r} is not a directory")
    if not isinstance(expect_sha256, str) or not _HEX64.match(expect_sha256):
        _refuse("expect_source_sha256_format", "--expect-source-sha256 must be 64 lowercase hex characters")
    best_p = d / "best.json"
    if not best_p.is_file():
        _refuse("parent_best_json_missing", f"{best_p} is missing")
    try:
        best = json.loads(best_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _refuse("parent_best_json_format", f"{best_p} is not JSON ({e})")
    if not isinstance(best, dict) or sorted(best) != ["best_ckpt", "best_val_miou_all_class"]:
        _refuse("parent_best_json_format", f"{best_p} keys {sorted(best) if isinstance(best, dict) else best!r} "
                                           "!= ['best_ckpt', 'best_val_miou_all_class']")
    ck = d / Path(str(best["best_ckpt"])).name
    if not ck.is_file():
        _refuse("parent_checkpoint_missing", f"best.json names {Path(str(best['best_ckpt'])).name}, absent from {d}")
    ck_sha = sha256_file(ck)
    if ck_sha != expect_sha256:
        _refuse("parent_sha256_mismatch", f"{ck.name} has sha256 {ck_sha}, expected {expect_sha256}")

    if stage["source_stage"] == "E1":
        meta_p = tel_p = d / "e1_telemetry.jsonl"
    else:
        meta_p, tel_p = d / "e3_run_meta.jsonl", d / "e3_telemetry.jsonl"
    for p, what in ((meta_p, "run_meta"), (tel_p, "telemetry")):
        if not p.is_file():
            _refuse("parent_records_missing", f"the parent's {what} file {p.name} is missing")
    metas = _run_meta_rows(meta_p)
    if len(metas) != 1:
        _refuse("parent_run_meta_rows", f"{meta_p.name} holds {len(metas)} run_meta rows; exactly 1 is required")
    meta = metas[0]
    for key in ("mode", "seed"):
        if key not in meta:
            _refuse("parent_run_meta_key", f"the parent's run_meta row has no {key!r} key")
    if require_real and meta["mode"] != "real":
        _refuse("parent_mode_not_real", f"the parent's run_meta row has mode {meta['mode']!r}, not 'real'")
    if meta["seed"] != seed:
        _refuse("parent_seed_mismatch", f"the parent ran seed {meta['seed']!r}; --seed is {seed} (AM-1: E5/E6 "
                                        "take the seed of their FP32 parent)")
    if stage["source_stage"] == "E3":
        if meta.get("stage") != "E3":
            _refuse("parent_stage_mismatch", f"the parent's run_meta row has stage {meta.get('stage')!r}; E6 "
                                             "takes an E3 parent")
        rows, _torn = read_jsonl(tel_p)
        if any(r.get("event") == "run_abort" for r in rows):
            _refuse("parent_incomplete", f"{tel_p.name} holds a run_abort row")
        ends = [r for r in rows if r.get("event") == "run_end"]
        if not ends or ends[-1].get("checks_passed") is not True:
            _refuse("parent_incomplete", f"{tel_p.name} has no run_end record with checks_passed true")
    else:
        if "stage" in meta and meta["stage"] not in (None, "E1"):
            _refuse("parent_stage_mismatch", f"the parent's run_meta row has stage {meta['stage']!r}; E5 takes E1")
        if "max_iters" not in meta:
            _refuse("parent_run_meta_key", "the parent's run_meta row has no 'max_iters' key")
        found = False
        with open(tel_p, encoding="utf-8") as fh:
            for line in fh:
                if '"val"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "val" and r.get("iter") == meta["max_iters"]:
                    found = True
        if not found:
            _refuse("parent_incomplete", f"{tel_p.name} has no VAL row at max_iters {meta['max_iters']}")
    return {"run_dir": str(d.resolve()), "run_id": d.resolve().name, "best_json_sha256": sha256_file(best_p),
            "best_val_miou_all_class": best["best_val_miou_all_class"], "checkpoint": ck.name,
            "checkpoint_path": str(ck.resolve()), "checkpoint_sha256": ck_sha,
            "telemetry": tel_p.name, "telemetry_sha256": sha256_file(tel_p),
            "run_meta_file": meta_p.name, "run_meta_sha256": sha256_file(meta_p),
            "seed": meta["seed"], "mode": meta["mode"], "stage": meta.get("stage"),
            "lambda_logit": meta.get("lambda_logit"), "alpha_cwd": meta.get("alpha_cwd")}


def read_clip_selection(path, expect_sha256: str | None) -> dict:
    """A clip_selection.json (scripts/select_clip.py), checked against its pinned sha256."""
    if not expect_sha256 or not _HEX64.match(expect_sha256):
        _refuse("clip_selection_sha256_format", "--clip-selection-sha256 must be 64 lowercase hex characters")
    p = Path(path)
    if not p.is_file():
        _refuse("clip_selection_missing", f"{p} is not a file")
    raw = p.read_bytes()
    if sha256_bytes(raw) != expect_sha256:
        _refuse("clip_selection_sha256_mismatch", f"{p} has sha256 {sha256_bytes(raw)}, expected {expect_sha256}")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        _refuse("clip_selection_format", f"{p} is not JSON ({e})")
    if not isinstance(doc, dict) or doc.get("format") != CLIP_SELECTION_FORMAT:
        _refuse("clip_selection_format", f"{p} is not a {CLIP_SELECTION_FORMAT} file")
    winner = (doc.get("winner") or {}).get("clip_norm")
    if winner not in CLIP_CANDIDATES:
        _refuse("clip_selection_format", f"{p} names no winner among {CLIP_CANDIDATES} (got {winner!r})")
    return {"path": str(p.resolve()), "sha256": expect_sha256, "winner": winner}


def clip_binding(stage_name: str, seed: int, clip, *, u4_pilot: bool,
                 clip_selection: str | None, clip_selection_sha256: str | None) -> dict:
    """The pilot's two runs pass --u4-pilot; every other real run binds to clip_selection.json."""
    if clip is None or not isinstance(clip, float) or not math.isfinite(clip) or clip <= 0:
        _refuse("grad_clip_norm_invalid", f"--grad-clip-norm must be a positive finite value, got {clip!r}")
    if clip not in CLIP_CANDIDATES:
        _refuse("grad_clip_norm_not_candidate", f"--grad-clip-norm {clip!r} is not a U4 candidate {CLIP_CANDIDATES}")
    is_u4 = stage_name == U4_STAGE and seed == U4_SEED
    if u4_pilot:
        if not is_u4:
            _refuse("u4_pilot_not_e5_s42", f"--u4-pilot is accepted for E5 seed 42 only, not {stage_name} seed {seed}")
        if clip_selection is not None or clip_selection_sha256 is not None:
            _refuse("u4_pilot_with_clip_selection", "--u4-pilot and --clip-selection are exclusive")
        return {"clip_source": "u4_pilot", "u4_pilot": True, "clip_selection": None}
    if is_u4:
        _refuse("u4_pilot_required", "a real E5 seed-42 launch is one of the two U4 pilot runs: pass --u4-pilot")
    if not clip_selection or not clip_selection_sha256:
        _refuse("clip_selection_required", f"{stage_name} seed {seed} takes the U4 winner: pass --clip-selection "
                                           "<clip_selection.json> --clip-selection-sha256 <64 hex>")
    sel = read_clip_selection(clip_selection, clip_selection_sha256)
    if sel["winner"] != clip:
        _refuse("clip_selection_winner_mismatch", f"clip_selection.json's winner is {sel['winner']!r}; "
                                                  f"--grad-clip-norm is {clip!r}")
    return {"clip_source": "clip_selection", "u4_pilot": False, "clip_selection": sel}


def _git_head_bytes(repo_root: Path, rel: str) -> bytes | None:
    try:
        p = subprocess.run(["git", "-C", str(repo_root), "show", f"HEAD:{rel}"], capture_output=True,
                           timeout=60, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def read_tracked_selection(rel_path: str | None, expect_sha256: str | None, fmt: str, what: str,
                           repo_root: Path = REPO) -> dict:
    """A λ or α selection file: repo-relative, tracked and unchanged at HEAD, its sha256 and format checked.

    Read with json.load only; nothing is imported from the lane that writes these files.
    """
    if not rel_path:
        _refuse(f"{what}_selection_missing", f"E6 requires --{what}-selection and --{what}-selection-sha256")
    if not expect_sha256 or not _HEX64.match(expect_sha256):
        _refuse("selection_sha256_format", f"--{what}-selection-sha256 must be 64 lowercase hex characters")
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        _refuse("selection_path_not_repo_relative", f"--{what}-selection {rel_path!r} must be repo-relative")
    if "test" in rel.as_posix().lower():
        _refuse("selection_test_path", f"--{what}-selection {rel_path!r} contains 'test'")
    full = Path(repo_root) / rel
    if not full.is_file():
        _refuse("selection_missing_file", f"{full} is not a file")
    data = full.read_bytes()
    at_head = _git_head_bytes(Path(repo_root), rel.as_posix())
    if at_head is None:
        _refuse("selection_not_tracked", f"{rel.as_posix()} is not tracked at HEAD")
    if at_head != data:
        _refuse("selection_changed_since_head", f"{rel.as_posix()} differs from its bytes at HEAD")
    if sha256_bytes(data) != expect_sha256:
        _refuse("selection_sha256_mismatch", f"{rel.as_posix()} has sha256 {sha256_bytes(data)}, expected "
                                             f"{expect_sha256}")
    try:
        doc = json.loads(data.decode("utf-8"))
    except ValueError as e:
        _refuse("selection_format", f"{rel.as_posix()} is not JSON ({e})")
    if not isinstance(doc, dict) or doc.get("format") != fmt or not isinstance(doc.get("winner"), dict):
        _refuse("selection_format", f"{rel.as_posix()} is not a {fmt} file with a winner")
    return {"path": rel.as_posix(), "sha256": expect_sha256, "format": fmt, "winner": doc["winner"]}


def e6_parent_binding(parent: dict, seed: int, lam: dict, alpha: dict) -> dict:
    """E6's parent is the E3 run that the λ and α selections name."""
    if parent.get("stage") != "E3":
        _refuse("parent_stage_mismatch", f"E6's parent run_meta has stage {parent.get('stage')!r}, not 'E3'")
    if parent.get("seed") != seed:
        _refuse("parent_seed_mismatch", f"E6's parent ran seed {parent.get('seed')!r}, --seed is {seed}")
    if parent.get("lambda_logit") != lam["winner"].get("lambda"):
        _refuse("parent_lambda_mismatch", f"the parent's lambda_logit {parent.get('lambda_logit')!r} != the λ "
                                          f"selection's winner {lam['winner'].get('lambda')!r}")
    if parent.get("alpha_cwd") != alpha["winner"].get("alpha"):
        _refuse("parent_alpha_mismatch", f"the parent's alpha_cwd {parent.get('alpha_cwd')!r} != the α selection's "
                                         f"winner {alpha['winner'].get('alpha')!r}")
    if seed == 42 and parent.get("run_id") != alpha["winner"].get("run_id"):
        _refuse("parent_run_id_mismatch", f"at seed 42 E6's parent must be the α winner run "
                                          f"{alpha['winner'].get('run_id')!r}, not {parent.get('run_id')!r}")
    return {"lambda_selection": lam, "alpha_selection": alpha}


def tf32_error(state: dict | None = None) -> str | None:
    state = tf32_state() if state is None else state
    wrong = {k: state.get(k) for k, v in TF32_DEFAULTS.items() if state.get(k) != v}
    return None if not wrong else f"TF32 is not at the torch defaults {TF32_DEFAULTS}: {wrong}"


def code_identity() -> dict:
    from .ptq import code_provenance
    prov = code_provenance(QAT_CODE_PATHS)
    return {"git_head": prov["commit"], "code_clean_at_head": prov["clean_at_commit"],
            "code_files_sha256": prov["files_sha256"]}


# ------------------------------------------------------------------ loaders
def build_qat_loaders(seed: int, num_workers: int):
    """E1's TRAIN and VAL loader calls, verbatim (train_e1.py:320-322), with seed=--seed (L-AM1q)."""
    from src.data import build_dataloader
    train = build_dataloader("train", BATCH_SIZE, num_workers=num_workers,
                             persistent_workers=num_workers > 0, seed=seed)
    val = build_dataloader("val", BATCH_SIZE, num_workers=num_workers, seed=seed)
    return train, val


def batch_sha256(mask: torch.Tensor) -> str:
    """The batch fingerprint: exact, on the CPU, from the integer masks as the loader returned them."""
    m = mask.detach()
    if m.device.type != "cpu":
        raise QATStop("batch_fingerprint_device", "the batch fingerprint is taken on the CPU, before .to(device)")
    return hashlib.sha256(m.clone().contiguous().numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ checkpoints
def cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_epoch_checkpoint(out_dir: Path, *, stage_name: str, source_stage: str, epoch: int, step: int,
                          state: dict, state_sha256: str, seed: int, clip_norm: float, run_id: str,
                          source_sha256: str) -> tuple[Path, str, int]:
    """eNN.pt, written atomically (same-directory .tmp + os.replace), in the x86 sidecar schema."""
    d = Path(out_dir) / EPOCH_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"e{epoch:02d}.pt"
    if path.exists():
        raise QATStop("epoch_checkpoint_exists", f"{path} already exists; a run never overwrites")
    payload = {"stage": stage_name, "quantization": QAT_SIDECAR_KIND, "artifact_role": QAT_SIDECAR_ROLE,
               "is_official_accuracy_artifact": False, "is_deployment_artifact": False,
               "epoch": epoch, "step": step, "model_state_dict": state, "num_classes": NUM_CLASSES,
               "training_quant_backend": QUANT_BACKEND, "source_stage": source_stage,
               "source_checkpoint_sha256": source_sha256, "seed": seed, "clip_norm": clip_norm,
               "run_id": run_id, "state_sha256": state_sha256,
               "purpose": ("pre-convert QAT state of one epoch, converted and scored after training "
                           "(AM-4a item 2); never an accuracy, robustness, size or deployment artifact")}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path, sha256_file(path), path.stat().st_size


# ------------------------------------------------------------------ the run
class QATHooks:
    """Smoke-only observation points inside the trainer's own loop; a real run refuses any."""

    def on_start(self, prepared, optimizer, scheduler, run_meta):
        pass

    def after_step(self, step, prepared, row):
        pass

    def before_val(self, epoch, prepared):
        pass

    def after_val(self, epoch, prepared):
        pass

    def after_epoch(self, epoch, prepared, checkpoint_path, row):
        pass


def optimizer_record(optimizer, model: nn.Module) -> dict:
    groups = optimizer.param_groups
    g = groups[0]
    group_ids = {id(p) for p in g["params"]}
    model_ids = {id(p) for p in model.parameters()}
    return {"class": type(optimizer).__name__, "param_groups": len(groups),
            "params_in_group": len(g["params"]), "model_params": len(model_ids),
            "group_holds_every_model_param": group_ids == model_ids,
            "lr": g["lr"], "initial_lr": g.get("initial_lr"), "momentum": g["momentum"],
            "weight_decay": g["weight_decay"], "dampening": g["dampening"], "nesterov": g["nesterov"]}


def optimizer_of_record_error(rec: dict) -> str | None:
    want = {"class": "SGD", "param_groups": 1, "group_holds_every_model_param": True, "lr": LEARNING_RATE,
            "initial_lr": LEARNING_RATE, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
            "dampening": DAMPENING, "nesterov": NESTEROV}
    wrong = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v or type(rec.get(k)) is not type(v)}
    return None if not wrong else f"the optimizer differs from the recipe of record: {wrong}"


def run_qat(*, stage: str, mode: str, model: nn.Module, source_meta: dict, out_dir, seed: int,
            clip_norm: float, clip_source: str, device: str, num_workers: int, loaders=None,
            parent: dict | None = None, extra_meta: dict | None = None, hooks: QATHooks | None = None,
            inject_nan_at_steps=(), max_steps: int | None = None, max_val_batches: int | None = None,
            log=print) -> dict:
    """Train one E5/E6 run under the AM-4a recipe. Returns a summary; raises QATRefused, QATStop.

    `mode` "real" is the launch of record (main() has already passed every launch gate). `mode` "smoke"
    is the synthetic path of the smokes: it accepts injected loaders, hooks, NaN injection, an early
    stop at `max_steps` and a VAL cap, none of which a real run accepts. Nothing is written before every
    check below has passed.
    """
    from src.seeds import set_seed
    from src.training.losses import CombinedCEDiceLoss
    from src.training.train_e1 import CLASS_WEIGHTS_JSON, _image_digest, load_ce_weights, validate

    st = resolve_quant_stage(stage)
    if st["method"] != "qat":
        _refuse("stage_not_qat", f"{st['name']} is not a QAT stage")
    if mode not in MODES:
        _refuse("mode_invalid", f"mode {mode!r} not in {MODES}")
    if mode == "real" and (loaders is not None or hooks is not None or inject_nan_at_steps or max_steps
                           or max_val_batches):
        _refuse("real_run_test_hooks", "a real run takes no injected loaders, hooks, NaN injection, step cap "
                                       "or VAL cap")
    if mode == "real" and clip_source not in CLIP_SOURCES:
        _refuse("clip_source", f"a real run's clip comes from {CLIP_SOURCES}, not {clip_source!r}")
    pins = config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    if isinstance(clip_norm, bool) or not isinstance(clip_norm, float) or not math.isfinite(clip_norm) \
            or clip_norm <= 0:
        _refuse("grad_clip_norm_invalid", f"clip norm must be a positive finite float, got {clip_norm!r}")
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
        train_loader, val_loader = build_qat_loaders(seed, num_workers)
        loader_source = "E1 build_dataloader (train_e1.py:320-322), seed=--seed"
    else:
        train_loader, val_loader = loaders
        loader_source = "injected (smoke)"
    steps_per_epoch = len(train_loader)
    t_max = EPOCHS * steps_per_epoch
    loader_batch = getattr(train_loader, "batch_size", None)
    loader_drop_last = getattr(train_loader, "drop_last", None)
    if mode == "real" and (steps_per_epoch != REAL_STEPS_PER_EPOCH or t_max != REAL_T_MAX):
        _refuse("steps_per_epoch", f"steps_per_epoch {steps_per_epoch} / T_max {t_max}; the recipe of record is "
                                   f"{REAL_STEPS_PER_EPOCH} / {REAL_T_MAX} (5,367 TRAIN images, batch 16, drop_last)")
    if mode == "real" and (loader_batch != BATCH_SIZE or loader_drop_last is not True):
        _refuse("loader_of_record", f"the TRAIN loader has batch_size {loader_batch!r}, drop_last "
                                    f"{loader_drop_last!r}; the recipe of record is {BATCH_SIZE}, True")
    if steps_per_epoch < 1:
        _refuse("empty_loader", "the TRAIN loader yields no batch")
    bn_after_step, obs_after_step = qat_freeze_steps(steps_per_epoch, BN_FREEZE_EPOCH, OBS_FREEZE_EPOCH)

    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        _refuse("backend_unavailable", str(e))
    prepared = prepare_qat_model(model, select_backend=False)
    unfused = unfused_batchnorm(prepared)
    if unfused and mode == "real":
        _refuse("unfused_batchnorm", f"{len(unfused)} BatchNorm module(s) sit outside a fused QAT ConvBn: "
                                     f"{unfused[:5]}; the epoch-11 freeze would miss them")
    census = observer_census(prepared)
    fq_count = len(fake_quant_modules(prepared))
    never = never_observed_modules(prepared)
    kinds = state_kinds(prepared)
    convbn = fused_convbn_census(prepared)
    prepared = prepared.to(dev)
    prepared.train()

    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=IGNORE_INDEX).to(dev)
    optimizer = torch.optim.SGD(prepared.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, dampening=DAMPENING, nesterov=NESTEROV)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=ETA_MIN)
    opt_rec = optimizer_record(optimizer, prepared)
    opt_err = optimizer_of_record_error(opt_rec)
    if opt_err:
        raise QATStop("optimizer_of_record", opt_err)

    from .ptq import qconfig_summary
    from .qconfig import qat_qconfig
    qsum = qconfig_summary(qat_qconfig())
    identity = code_identity()
    import numpy
    import torchvision
    tel_path = out / TELEMETRY_NAME
    run_meta = {
        "event": "run_meta", "wall_clock": time.time(), "schema": TELEMETRY_SCHEMA, "lane": LANE,
        "mode": mode, "stage": st["name"], "source_stage": st["source_stage"], "seed": seed,
        "run_id": run_id, "out_dir": str(out.resolve()),
        "parent": parent, "source_checkpoint_sha256": source_meta["sha256"],
        "source_checkpoint": source_meta.get("path"), "source_checkpoint_bytes": source_meta.get("bytes"),
        "clip_norm": clip_norm, "clip_source": clip_source,
        "epochs": EPOCHS, "batch_size": loader_batch, "drop_last": loader_drop_last,
        "gradient_accumulation": None, "ema": False, "distillation": False,
        "steps_per_epoch": steps_per_epoch, "total_steps": t_max,
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "loader": loader_source, "num_workers": num_workers, "val_batches_cap": max_val_batches,
        "persistent_workers": bool(getattr(train_loader, "persistent_workers", False)),
        "scheduler": {"class": type(scheduler).__name__, "T_max": scheduler.T_max,
                      "eta_min": scheduler.eta_min, "step": "once after every optimizer.step"},
        "optimizer": opt_rec, "lr": opt_rec["lr"], "momentum": opt_rec["momentum"],
        "weight_decay": opt_rec["weight_decay"], "nesterov": opt_rec["nesterov"],
        "dampening": opt_rec["dampening"],
        "bn_freeze_epoch": BN_FREEZE_EPOCH, "obs_freeze_epoch": OBS_FREEZE_EPOCH,
        "bn_frozen_from_epoch": BN_FREEZE_EPOCH + 1, "obs_frozen_from_epoch": OBS_FREEZE_EPOCH + 1,
        "bn_freeze_after_step": bn_after_step, "obs_freeze_after_step": obs_after_step,
        "objective": OBJECTIVE, "class_weights_sha256": sha256_file(CLASS_WEIGHTS_JSON),
        "ignore_index": IGNORE_INDEX,
        "engine": torch.backends.quantized.engine, "qconfig": qsum,
        "observers": {"activation": qsum["activation"]["observer_class"],
                      "weight": qsum["weight"]["observer_class"]},
        "observer_census": census, "fake_quant_modules": fq_count, "never_observed_modules": never,
        "observer_freeze_scope": OBSERVER_FREEZE_SCOPE,
        "fused_convbn": convbn, "unfused_batchnorm": len(unfused),
        "tf32": tf32_state(), "determinism": determinism_state(),
        "cuda_initialized_at_seed": cuda_initialized_at_seed,
        "device": str(dev), "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "python": sys.version.split()[0], "image_digest": _image_digest(), "host": host_identity(),
        "git_head": identity["git_head"], "code_clean_at_head": identity["code_clean_at_head"],
        "code_files_sha256": identity["code_files_sha256"],
    }
    if st["source_stage"] == "E3":
        run_meta["cwd_projection_loaded"] = False
    if extra_meta:
        run_meta.update(extra_meta)
    out.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(tel_path)
    tel.write(run_meta)
    if hooks is not None:
        hooks.on_start(prepared, optimizer, scheduler, run_meta)
    log(f"[qat] {st['name']} seed {seed} clip {clip_norm} | {steps_per_epoch} steps/epoch, T_max {t_max} | "
        f"BN frozen after step {bn_after_step}, observers after {obs_after_step} | {run_id}")

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
        for epoch in range(1, EPOCHS + 1):
            if epoch == BN_FREEZE_EPOCH + 1:
                freeze_bn_stats(prepared)
            if epoch == OBS_FREEZE_EPOCH + 1:
                disable_observers(prepared)
            ep = {"nonfinite_loss": 0, "nonfinite_grad": 0, "step_errors": 0, "clipped": 0,
                  "max_pre_clip_norm": None, "steps": 0}
            for img, mask in train_loader:
                if max_steps is not None and step >= max_steps:
                    break
                step += 1
                ep["steps"] += 1
                t0 = time.time()
                fp = batch_sha256(mask)
                lr = optimizer.param_groups[0]["lr"]
                bn_frozen = epoch > BN_FREEZE_EPOCH
                obs_on = epoch <= OBS_FREEZE_EPOCH
                try:
                    img_d, mask_d = img.to(dev), mask.to(dev)
                    optimizer.zero_grad(set_to_none=True)
                    logits = prepared(img_d)
                    ce = criterion.ce(logits, mask_d)
                    dice = criterion.dice(logits, mask_d)
                    loss = ce + dice
                    if step in nan_steps:            # smoke only: a NaN loss whose gradient is NaN too
                        loss = loss * loss.new_tensor(float("nan"))
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,
                                                           norm_type=2.0, error_if_nonfinite=False)
                    optimizer.step()
                except Exception as e:                     # noqa: BLE001 -- recorded in a non-finite state
                    if nonfinite_since is None:
                        ok, fails = state_predicate(prepared.state_dict(), kinds, never)
                        if ok:
                            raise
                        _mark_nonfinite(step, fails)
                    scheduler.step()
                    ep["step_errors"] += 1
                    tel.write({"event": "step_error", "wall_clock": time.time(), "epoch": epoch,
                               "step": step, "epoch_step": ep["steps"], "lr": lr,
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
                    ep["max_pre_clip_norm"] = gn if ep["max_pre_clip_norm"] is None else max(ep["max_pre_clip_norm"], gn)
                if nonfinite_since is None and (not math.isfinite(loss_v) or not math.isfinite(gn)):
                    ok, fails = state_predicate(prepared.state_dict(), kinds, never)
                    if not ok:
                        _mark_nonfinite(step, fails)
                row = {"event": "train", "wall_clock": time.time(), "epoch": epoch, "step": step,
                       "epoch_step": ep["steps"], "lr": lr, "loss": loss_v, "ce": ce_v, "dice": dice_v,
                       "grad_norm": gn, "clipped": clipped, "bn_frozen": bn_frozen,
                       "observers_enabled": obs_on, "batch_sha256": fp,
                       "state_nonfinite": nonfinite_since is not None,
                       "iter_seconds": time.time() - t0}
                row = tel.write(row)
                if hooks is not None:
                    hooks.after_step(step, prepared, row)
            if max_steps is not None and step >= max_steps and ep["steps"] == 0:
                break

            # the epoch checkpoint, written before the VAL bracket, and the state it holds
            state = cpu_state(prepared)
            state_sha = state_digest(state)
            ck_path, ck_sha, ck_bytes = save_epoch_checkpoint(
                out, stage_name=st["name"], source_stage=st["source_stage"], epoch=epoch, step=step,
                state=state, state_sha256=state_sha, seed=seed, clip_norm=clip_norm, run_id=run_id,
                source_sha256=source_meta["sha256"])
            finite, fails = state_predicate(state, kinds, never)
            del state
            if not finite and nonfinite_since is None:
                _mark_nonfinite(step, fails)
            # the VAL bracket: observers off for the pass, their snapshot restored after it
            if hooks is not None:
                hooks.before_val(epoch, prepared)
            flags = observer_flags(prepared)
            disable_observers(prepared)
            t_val = time.time()
            val_error = None
            try:
                all_miou, disease_miou, cm, n_val_batches = validate(prepared, val_loader, dev, NUM_CLASSES,
                                                                     max_val_batches)
            except Exception as e:                         # noqa: BLE001 -- recorded in a non-finite state
                if nonfinite_since is None:
                    restore_observer_flags(prepared, flags)
                    raise
                all_miou = disease_miou = None
                cm, n_val_batches = None, 0
                val_error = f"{type(e).__name__}: {str(e)[:300]}"
            restore_observer_flags(prepared, flags)
            val_seconds = time.time() - t_val
            # the bracket left the state untouched
            if state_digest(prepared) != state_sha:
                raise QATStop("val_bracket_changed_state",
                              f"epoch {epoch}: the state after the VAL bracket differs from the checkpoint's")
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
                       "steps_in_epoch": ep["steps"], "checkpoint": f"{EPOCH_DIR}/{ck_path.name}",
                       "checkpoint_sha256": ck_sha, "checkpoint_bytes": ck_bytes, "state_sha256": state_sha,
                       "fake_quant_val_all_class_miou": all_miou,
                       "bn_frozen": epoch > BN_FREEZE_EPOCH, "observers_enabled": epoch <= OBS_FREEZE_EPOCH,
                       "nonfinite_loss_steps": ep["nonfinite_loss"], "nonfinite_grad_steps": ep["nonfinite_grad"],
                       "step_errors": ep["step_errors"], "clipped_steps": ep["clipped"],
                       "max_pre_clip_norm": ep["max_pre_clip_norm"], "state_finite": bool(finite),
                       "nonfinite_since_step": nonfinite_since,
                       "run_complete": epoch == EPOCHS and max_steps is None}
            end_row = tel.write(end_row)
            if hooks is not None:
                hooks.after_epoch(epoch, prepared, ck_path, end_row)
            log(f"[qat] epoch {epoch:02d}/{EPOCHS} step {step} fake-quant VAL {all_miou}"
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
    complete = epoch == EPOCHS and max_steps is None
    return {"run_id": run_id, "stage": st["name"], "seed": seed, "clip_norm": clip_norm,
            "telemetry": str(tel_path), "steps": step, "epochs_completed": epoch,
            "complete": complete, "nonfinite_since_step": nonfinite_since}


# ------------------------------------------------------------------ CLI (scripts/run_e5.py, run_e6.py)
def build_parser(stage: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{stage['name']}: {stage['objective']} (AM-4/AM-4a; the stage is "
                                            "pinned by the entry point)")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--source-run-dir", default=None)
    p.add_argument("--expect-source-sha256", default=None)
    p.add_argument("--expect-head", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--u4-pilot", action="store_true")
    p.add_argument("--clip-selection", default=None)
    p.add_argument("--clip-selection-sha256", default=None)
    p.add_argument("--lambda-selection", default=None)
    p.add_argument("--lambda-selection-sha256", default=None)
    p.add_argument("--alpha-selection", default=None)
    p.add_argument("--alpha-selection-sha256", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda")
    return p


def _out_dir_error(value) -> str | None:
    from .runner import QuantRunError, check_output_dir
    try:
        out = check_output_dir(value, create=False)
    except QuantRunError as e:
        return str(e)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        return f"--out-dir {out} exists and is not an empty directory; a run starts in a fresh directory"
    if "test" in str(out).lower():
        return f"--out-dir {out} contains 'test'"
    return None


def real_run_gates(args, stage: dict, *, repo_root: Path = REPO) -> dict:
    """Every gate of a real launch, in order. None of them writes anything."""
    if not args.real_run:
        _refuse("real_run_flag", f"{stage['name']} requires --real-run --confirm-real-run")
    if not args.confirm_real_run:
        _refuse("confirm_real_run_flag", "--real-run requires --confirm-real-run")
    pins = config_pins_error()
    if pins:
        _refuse("config_pins", pins)
    if not args.expect_head or not _HEX40.match(args.expect_head):
        _refuse("expect_head_format", "--expect-head must be the 40-hex commit of the checkout")
    if args.seed not in REAL_SEEDS:
        _refuse("seed", f"--seed {args.seed!r} is not one of {REAL_SEEDS}")
    if args.num_workers is None or args.num_workers < 1:
        _refuse("num_workers", "--num-workers must be given and at least 1")
    clip = clip_binding(stage["name"], args.seed, args.grad_clip_norm, u4_pilot=args.u4_pilot,
                        clip_selection=args.clip_selection, clip_selection_sha256=args.clip_selection_sha256)
    sel = None
    e6_flags = (args.lambda_selection, args.lambda_selection_sha256, args.alpha_selection,
                args.alpha_selection_sha256)
    if stage["source_stage"] == "E3":
        lam = read_tracked_selection(args.lambda_selection, args.lambda_selection_sha256,
                                     LAMBDA_SELECTION_FORMAT, "lambda", repo_root)
        alpha = read_tracked_selection(args.alpha_selection, args.alpha_selection_sha256,
                                       ALPHA_SELECTION_FORMAT, "alpha", repo_root)
        sel = (lam, alpha)
    elif any(v is not None for v in e6_flags):
        _refuse("selection_not_applicable", f"{stage['name']} takes no λ or α selection file")
    err = _out_dir_error(args.out_dir)
    if err:
        _refuse("out_dir", err)
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        _refuse("cuda_required", f"a real QAT run needs a CUDA device (--device {args.device!r}, "
                                 f"available={torch.cuda.is_available()})")
    if torch.cuda.is_initialized():
        _refuse("cuda_initialized_before_seed", "CUDA is already initialised; set_seed must come first")
    tf = tf32_error()
    if tf:
        _refuse("tf32_not_default", tf)
    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        _refuse("backend_unavailable", str(e))
    from configs.data import DATA, SPLIT_SIZES
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    try:
        iso = assert_trainval_only_root(Path(DATA["root"]), {"train": SPLIT_SIZES["train"],
                                                             "val": SPLIT_SIZES["val"]})
    except TrainValIsolationError as e:
        _refuse("data_root_not_trainval_only", f"[{e.code}] {e}")
    ident = code_identity()
    if ident["git_head"] != args.expect_head:
        _refuse("expect_head_mismatch", f"the checkout's HEAD is {ident['git_head']!r}, --expect-head "
                                        f"{args.expect_head}")
    if ident["code_clean_at_head"] is not True:
        _refuse("code_not_clean", "the QAT code paths are not clean at HEAD")
    parent = resolve_parent(stage, args.source_run_dir, args.expect_source_sha256, args.seed)
    binding = None
    if sel is not None:
        binding = e6_parent_binding(parent, args.seed, *sel)
    return {"clip": clip, "parent": parent, "e6_binding": binding, "isolation": iso,
            "data_root": str(DATA["root"]), "identity": ident}


def load_source(stage: dict, parent: dict):
    """The parent's best.json checkpoint through the strict E1/E3 loaders (src/quant/checkpoint.py)."""
    from .runner import QuantRunError, check_source
    try:
        model, _ckpt, meta = check_source(stage, parent["checkpoint_path"])
    except QuantRunError as e:
        _refuse(e.code, str(e))
    if meta["sha256"] != parent["checkpoint_sha256"]:
        _refuse("source_changed", "the source checkpoint changed between the gate and the load")
    if stage["source_stage"] == "E3" and any("cwd" in k.lower() or "projection" in k.lower()
                                             for k in model.state_dict()):
        _refuse("source_projection_keys", "the E3 student carries projection keys (AM-4a item 5)")
    return model, meta


def result_line(summary: dict) -> str:
    tag = (f"{summary['stage']}, seed {summary['seed']}, clip {summary['clip_norm']}, "
           f"{summary['epochs_completed']}/{EPOCHS} epochs")
    if not summary["complete"]:
        return f"RESULT: QAT INCOMPLETE ({tag})"
    if summary["nonfinite_since_step"] is not None:
        return f"RESULT: QAT COMPLETE, STATE NON-FINITE since step {summary['nonfinite_since_step']} ({tag})"
    return f"RESULT: QAT COMPLETE ({tag})"


def main(argv, stage_key: str) -> int:
    stage = resolve_quant_stage(stage_key)
    args = build_parser(stage).parse_args(argv)
    try:
        gates = real_run_gates(args, stage)
        model, source_meta = load_source(stage, gates["parent"])
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return EXIT_REFUSED
    except Exception as e:                                 # noqa: BLE001 -- before the run: nothing written
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}. Nothing was written.")
        return EXIT_ERROR
    extra = {"clip_selection": gates["clip"]["clip_selection"], "u4_pilot": gates["clip"]["u4_pilot"],
             "e6_selections": gates["e6_binding"], "data_root": gates["data_root"],
             "isolation_counts": gates["isolation"].get("counts"),
             "expect_head": args.expect_head, "expect_source_sha256": args.expect_source_sha256}
    try:
        summary = run_qat(stage=stage["key"], mode="real", model=model, source_meta=source_meta,
                          out_dir=args.out_dir, seed=args.seed, clip_norm=args.grad_clip_norm,
                          clip_source=gates["clip"]["clip_source"], device=args.device,
                          num_workers=args.num_workers, parent=gates["parent"], extra_meta=extra)
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return EXIT_REFUSED
    except QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    except Exception as e:                                 # noqa: BLE001 -- recorded as run_abort
        print(f"RESULT: ABORTED [exception] -- {type(e).__name__}: {e}")
        return EXIT_ABORTED
    print(result_line(summary))
    return EXIT_OK if summary["complete"] else EXIT_ABORTED


__all__ = [
    "BATCH_SIZE", "BN_FREEZE_EPOCH", "CLIP_CANDIDATES", "EPOCHS", "EPOCH_DIR", "LEARNING_RATE", "MOMENTUM",
    "OBS_FREEZE_EPOCH", "QAT_CODE_PATHS", "QATHooks", "QATRefused", "QATStop", "REAL_SEEDS",
    "REAL_STEPS_PER_EPOCH", "REAL_T_MAX", "TELEMETRY_NAME", "WEIGHT_DECAY", "batch_sha256",
    "bn_buffer_digest", "build_parser", "build_qat_loaders", "clip_binding", "config_pins_error",
    "e6_parent_binding", "fake_quant_modules", "flag_summary", "main", "never_observed_modules",
    "observer_buffer_digest", "observer_census", "read_clip_selection", "read_jsonl",
    "read_tracked_selection", "real_run_gates", "resolve_parent", "run_qat", "save_epoch_checkpoint",
    "state_digest", "state_kinds", "state_predicate", "strict_row",
]
