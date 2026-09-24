"""Evaluator runtime policy and record (L-EVAL-DET, DL-17; docs/EVALUATION_CONTRACT.md section 10).

What this module owns, and nothing else:

  * the determinism POLICY the evaluator applies before an FP32 student is evaluated on CUDA -- the
    four IMPLEMENTATION_CONTRACT B6 settings `src/seeds.py:set_seed` applies for E1 (seeds.py:26 and
    :35-37), WITHOUT the reseed (evaluation draws no random numbers, and a reseed would change the
    process RNG state every CPU caller relies on);
  * the forward hook that moves every input batch to the model's construction device, exactly as
    `train_e1.validate` does (`img.to(device)`), through the core's existing `forward=` parameter;
  * warning capture around the core call (N12: eval-path nondeterminism alerts are recorded, not lost);
  * the post-evaluation device and policy checks;
  * the `run.eval_runtime` record (version EVAL_RUNTIME_VERSION) that `write_artifact` stores.

Scope: the policy applies only to model_role=student, precision=fp32 on a CUDA device. Teacher, INT8
and CPU evaluations initialise no CUDA, apply nothing and record the inherited state with
`determinism_policy_applied=false`.

The policy key names mirror the frozen teacher runtime's (src/training/teacher_runner.py:41-48); they are
copied, never imported (that module imports mmengine). Import-time behaviour is side-effect free: no
torch state is read or written until a function is called.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import torch

from .artifacts import EVAL_RUNTIME_VERSION

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_REQUIRED = ":4096:8"                                   # = src/seeds.py:26
EXPECTED_POLICY = {
    "deterministic_algorithms": True,
    "deterministic_algorithms_warn_only": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    CUBLAS_ENV: CUBLAS_REQUIRED,
}
TF32_ENV_KEYS = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
EVAL_NUM_WORKERS = 0
NONDETERMINISTIC_ALERT = "does not have a deterministic implementation"   # e1_stdout.log:14 format
MAX_EVAL_WARNINGS = 50
MAX_WARNING_CHARS = 300

RECORD_KEYS = (
    "eval_runtime_version",
    "model_device", "model_tensor_devices", "input_devices", "forward_batches", "batch_size",
    "num_workers",
    "determinism_policy_applied", "cuda_initialized_before_policy",
    "inherited_cublas_workspace_config", "determinism", "fill_uninitialized_memory",
    "cudnn_enabled", "tf32",
    "gpu_name", "gpu_capability", "cudnn_version", "torch_cuda", "torch_num_threads", "pillow",
    "checkpoint_iteration", "checkpoint_best_val_miou_all_class", "image_digest",
    "eval_warnings", "eval_warnings_truncated", "nondeterministic_alert_count",
)


class EvalRuntimeError(RuntimeError):
    """A fail-closed runtime refusal. Raised before any artifact is written."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _env_or_none(name: str) -> str | None:
    """Environment value, stripped; unset or blank -> None (contract section 5.2: never "")."""
    v = os.environ.get(name, "").strip()
    return v or None


def determinism_state() -> dict:
    """The live five-state. Reads only; torch is consulted through module attributes at call time."""
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only":
            bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        CUBLAS_ENV: os.environ.get(CUBLAS_ENV),
    }


def determinism_required(model_role: str, precision: str, device: str) -> bool:
    """True exactly for an FP32 student evaluated on a CUDA device (EVALUATION_CONTRACT section 10)."""
    return (model_role == "student" and precision == "fp32"
            and torch.device(device).type == "cuda")


def apply_eval_determinism() -> dict:
    """Apply the B6 settings (seeds.py:26, :35-37) with no reseed. Fail-closed around CUDA init.

    One CUDA evaluation per process: CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle is
    created, so the policy is refused once CUDA is already initialised rather than applied too late.
    """
    if torch.cuda.is_initialized():
        raise EvalRuntimeError(
            "cuda_initialized_before_policy",
            "CUDA was initialised before the evaluation determinism policy could be applied; "
            "run one CUDA evaluation per fresh process (EVALUATION_CONTRACT section 10)")
    inherited = _env_or_none(CUBLAS_ENV)
    os.environ[CUBLAS_ENV] = CUBLAS_REQUIRED                  # seeds.py:26
    torch.backends.cudnn.deterministic = True                 # seeds.py:35
    torch.backends.cudnn.benchmark = False                    # seeds.py:36
    torch.use_deterministic_algorithms(True, warn_only=True)  # seeds.py:37
    if torch.cuda.is_initialized():
        raise EvalRuntimeError(
            "cuda_initialized_during_policy",
            "CUDA became initialised while the evaluation determinism policy was being applied")
    state = determinism_state()
    if state != EXPECTED_POLICY:
        raise EvalRuntimeError(
            "policy_not_applied", f"live determinism state {state} != policy {EXPECTED_POLICY}")
    return {"cuda_initialized_before_policy": False, "inherited_cublas_workspace_config": inherited}


def resolve_model_device(requested: str, *, cpu_only: bool) -> torch.device:
    """The model's construction device. Call AFTER the model is built.

    Teacher and INT8 models are built on the CPU whatever `--device` says; a CUDA request without an
    index resolves to the current CUDA device, which is where `.to("cuda")` placed the model.
    """
    if cpu_only:
        return torch.device("cpu")
    d = torch.device(requested)
    if d.type != "cuda":
        return torch.device(d.type)          # tensors report "cpu", never "cpu:0"
    if d.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return d


def model_tensor_devices(model) -> list[str]:
    """Sorted distinct devices of every parameter and buffer; [] for a tensor-free module."""
    devs = {str(t.device) for t in model.parameters()} | {str(t.device) for t in model.buffers()}
    return sorted(devs)


class ModelDeviceForward:
    """`evaluate_model(forward=...)` hook: move each input batch to the model device, then call it.

    Mirrors train_e1.validate (`student(img.to(device))`). Records the device every input was on.
    """

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.input_devices: set[str] = set()

    def __call__(self, model, images: torch.Tensor):
        x = images.to(self.device)
        self.input_devices.add(str(x.device))
        return model(x)


def _warning_line(w) -> str:
    text = str(w.message).splitlines()[0] if str(w.message) else ""
    return f"{w.category.__name__}: {text}"[:MAX_WARNING_CHARS]


def evaluate_capturing_warnings(fn, *args, **kwargs):
    """Run `fn`, capturing every warning it emits (simplefilter "always").

    Returns (result, summary) where summary = {eval_warnings, eval_warnings_truncated,
    nondeterministic_alert_count}. Each unique warning is re-emitted once to stderr as
    "[eval-warning xN] <Category>: <first line>" so log greps keep working.
    """
    counts: dict[str, int] = {}
    alerts = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn(*args, **kwargs)
        finally:
            # Echoed on success AND on failure, so a failing run keeps its diagnostics on stderr.
            for w in caught:
                line = _warning_line(w)
                counts[line] = counts.get(line, 0) + 1
                if NONDETERMINISTIC_ALERT in str(w.message):
                    alerts += 1
            for line in sorted(counts):
                print(f"[eval-warning x{counts[line]}] {line}", file=sys.stderr)
    unique = sorted(counts)
    summary = {
        "eval_warnings": unique[:MAX_EVAL_WARNINGS],
        "eval_warnings_truncated": len(unique) > MAX_EVAL_WARNINGS,
        "nondeterministic_alert_count": alerts,
    }
    return result, summary


def check_post_eval(*, model, model_device: torch.device, fwd: ModelDeviceForward,
                    policy_applied: bool) -> None:
    """Refuse an artifact whose model, inputs or policy did not stay where they were put."""
    md = str(torch.device(model_device))
    devs = model_tensor_devices(model)
    if devs and devs != [md]:
        raise EvalRuntimeError("model_device_mismatch",
                               f"model tensors on {devs}, model device {md}")
    if fwd.input_devices != {md}:
        raise EvalRuntimeError("input_device_mismatch",
                               f"inputs on {sorted(fwd.input_devices)}, model device {md}")
    if policy_applied and determinism_state() != EXPECTED_POLICY:
        raise EvalRuntimeError("policy_drift",
                               f"determinism state after evaluation {determinism_state()} "
                               f"!= policy {EXPECTED_POLICY}")


def fill_uninitialized_memory_state():
    """torch.utils.deterministic.fill_uninitialized_memory, or None where torch lacks it."""
    try:
        import torch.utils.deterministic as _det
        return bool(_det.fill_uninitialized_memory)
    except (ImportError, AttributeError):
        return None


def tf32_state() -> dict:
    """TF32 is RECORDED, never set (E1 never set it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
    for key in TF32_ENV_KEYS:
        state[key] = _env_or_none(key)
    return state


def image_digest() -> str | None:
    """PLANTSEG_IMAGE_DIGEST, stripped; absent or blank -> None (train_e1._image_digest semantics)."""
    return _env_or_none("PLANTSEG_IMAGE_DIGEST")


def _pillow_version() -> str | None:
    try:
        import PIL
        return PIL.__version__
    except Exception:                                         # noqa: BLE001 -- recorded as absent
        return None


def build_eval_runtime_record(*, model, model_device, fwd: ModelDeviceForward, batch_size: int,
                              forward_batches: int, policy_applied: bool, policy_info,
                              ckpt_info, warn_summary: dict) -> dict:
    """The fixed RECORD_KEYS record stored as summary.run.eval_runtime. null = not applicable."""
    md = torch.device(model_device)
    on_cuda = md.type == "cuda"
    if on_cuda:
        idx = md.index if md.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(idx)
        gpu_name = torch.cuda.get_device_name(idx)
        gpu_capability = f"{major}.{minor}"
        cudnn_version = torch.backends.cudnn.version()
    else:
        gpu_name = gpu_capability = cudnn_version = None
    if policy_applied:
        before = policy_info["cuda_initialized_before_policy"]
        inherited = policy_info["inherited_cublas_workspace_config"]
    else:
        before = None
        inherited = _env_or_none(CUBLAS_ENV)
    best = getattr(ckpt_info, "best_val_miou_all_class", None)
    if best is not None and not math.isfinite(best):
        best = None                          # section 5.2: null, never a non-finite number
    record = {
        "eval_runtime_version": EVAL_RUNTIME_VERSION,
        "model_device": str(md),
        "model_tensor_devices": model_tensor_devices(model),
        "input_devices": sorted(fwd.input_devices),
        "forward_batches": int(forward_batches),
        "batch_size": int(batch_size),
        "num_workers": EVAL_NUM_WORKERS,
        "determinism_policy_applied": bool(policy_applied),
        "cuda_initialized_before_policy": before,
        "inherited_cublas_workspace_config": inherited,
        "determinism": {**determinism_state(), CUBLAS_ENV: _env_or_none(CUBLAS_ENV)},
        "fill_uninitialized_memory": fill_uninitialized_memory_state(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "tf32": tf32_state(),
        "gpu_name": gpu_name,
        "gpu_capability": gpu_capability,
        "cudnn_version": cudnn_version,
        "torch_cuda": torch.version.cuda,
        "torch_num_threads": int(torch.get_num_threads()),
        "pillow": _pillow_version(),
        "checkpoint_iteration": getattr(ckpt_info, "iteration", None),
        "checkpoint_best_val_miou_all_class": best,
        "image_digest": image_digest(),
        "eval_warnings": list(warn_summary["eval_warnings"]),
        "eval_warnings_truncated": bool(warn_summary["eval_warnings_truncated"]),
        "nondeterministic_alert_count": int(warn_summary["nondeterministic_alert_count"]),
    }
    if tuple(record) != RECORD_KEYS:
        raise EvalRuntimeError("record_keys", f"record keys {tuple(record)} != {RECORD_KEYS}")
    return record
