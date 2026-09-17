"""G18 — the teacher fine-tune's MMEngine Runner and its determinism attestation (B58 §9–§11).

`TeacherRunner` is MMEngine's `Runner` with one override, `set_randomness`. MMEngine 0.10.7 calls it
from `Runner.__init__` (runner.py:376). That call comes after `setup_env` (:372), which applies
`env_cfg.cudnn_benchmark`, and before `_log_env` (:403), whose `collect_env()` initializes CUDA on a GPU
host. `resume()` (:2060) calls it again when a checkpoint's seed differs.

The override seeds through MMEngine's `set_random_seed` with the strict branch off. That branch calls
`torch.use_deterministic_algorithms(True)` and so discards `warn_only` (B56 §3.4). The override then
establishes contract B6's policy itself. `CUBLAS_WORKSPACE_CONFIG` is asserted and never created: the G8
image supplies it at process start.

`TeacherDeterminismAttestationHook` records the effective policy once at the first training iteration and
once at the first validation iteration. It writes to the run log and to the message hub, which every later
checkpoint carries, and fails closed on a deviation.

Build the runner with `TeacherRunner.from_cfg(cfg)`. The caller is `scripts/launch_teacher_finetune.py
--launch`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Imported-code attestation (B58 §11): hash the bytes this import resolved to, before torch or MMEngine
# load. A checkout hash cannot stand in for it: a mounted volume can shadow the image's source tree (B55 §4).
_MODULE_PATH = Path(__file__).resolve()
MODULE_PROVENANCE = {"path": str(_MODULE_PATH),
                     "sha256": hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()}

import torch  # noqa: E402
from mmengine.hooks import Hook  # noqa: E402
from mmengine.runner import Runner, set_random_seed  # noqa: E402

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_REQUIRED = ":4096:8"
CUBLAS_MESSAGE = "CUBLAS_WORKSPACE_CONFIG must already equal ':4096:8' before CUDA initialization."  # B58 §10

# Contract B6's effective policy, in the attestation's order.
EXPECTED_POLICY = {
    "deterministic_algorithms": True,
    "deterministic_algorithms_warn_only": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    CUBLAS_ENV: CUBLAS_REQUIRED,
}

_ENTERED = "_g18_set_randomness_entered"
_PRE_CUDA_MARKER = "_g18_pre_cuda_policy_established"


def effective_policy() -> dict:
    """The five-state policy as this process sees it now. Reads only: no RNG draw, no CUDA call."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        CUBLAS_ENV: os.environ.get(CUBLAS_ENV),
    }


def pre_cuda_policy_established(runner) -> bool:
    """True only after a first `TeacherRunner.set_randomness` found CUDA uninitialized both before and after
    establishing the policy (B58 §9.2 steps 1 and 5)."""
    return getattr(runner, _PRE_CUDA_MARKER, False) is True


class TeacherRunner(Runner):
    """MMEngine `Runner` whose only override is `set_randomness` (B58 §9.1)."""

    def set_randomness(self, seed, diff_rank_seed: bool = False, deterministic: bool = False) -> None:
        first_call = not getattr(self, _ENTERED, False)
        setattr(self, _ENTERED, True)
        # B58 §9.3: a later call (MMEngine's differing-seed resume) requires a completed first call.
        if not first_call and not pre_cuda_policy_established(self):
            raise RuntimeError("TeacherRunner.set_randomness was called again, but its first call never "
                               "established the pre-CUDA policy (G18).")
        if deterministic is not True:
            raise ValueError("TeacherRunner requires randomness.deterministic=True (contract B6); got "
                             f"{deterministic!r}.")
        # B58 §9.2 steps 1-6. A failure before step 6 leaves the marker unset.
        if first_call and torch.cuda.is_initialized():                                  # 1
            raise RuntimeError("CUDA was initialized before TeacherRunner.set_randomness ran (G18).")
        inherited = os.environ.get(CUBLAS_ENV)                                          # 2
        if inherited != CUBLAS_REQUIRED:
            raise RuntimeError(f"{CUBLAS_MESSAGE} Observed: {inherited!r}.")
        self._deterministic = deterministic
        self._seed = set_random_seed(seed=seed, deterministic=False,                   # 3
                                     diff_rank_seed=diff_rank_seed)
        torch.backends.cudnn.deterministic = True                                       # 4
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        if first_call:
            if torch.cuda.is_initialized():                                             # 5
                raise RuntimeError("CUDA became initialized inside TeacherRunner.set_randomness (G18).")
            setattr(self, _PRE_CUDA_MARKER, True)                                       # 6


class TeacherDeterminismAttestationHook(Hook):
    """Record the effective policy at the first `before_train_iter` and the first `before_val_iter`
    (B58 §11), then fail closed on a deviation.

    LOWEST priority runs it after every other hook at those points, immediately before `train_step` and
    `val_step` (loops.py:309-313, :401-404). Each record is logged and stored in the message hub before it is
    validated, and only a record that passes marks its point as attested. The hook reads state only, and
    never touches `runner.iter`, because that property can build the train loop (runner.py:538-592).
    """

    priority = "LOWEST"

    def __init__(self) -> None:
        self._attested: set[str] = set()

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._attest(runner, "first_train_iter", batch_idx)

    def before_val_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._attest(runner, "first_val_iter", batch_idx)

    def _attest(self, runner, point: str, batch_idx: int) -> None:
        if point in self._attested:
            return
        policy = effective_policy()
        record = {
            "point": point,
            "batch_idx": batch_idx,
            "policy": policy,
            "runner_class": f"{type(runner).__module__}.{type(runner).__qualname__}",
            "pre_cuda_policy_established": pre_cuda_policy_established(runner),
            "seed": runner.seed,                                   # context
            "cuda_initialized": torch.cuda.is_initialized(),       # context
            "teacher_runner_module": MODULE_PROVENANCE,
        }
        runner.logger.info(f"G18 determinism attestation: {json.dumps(record, sort_keys=True)}")
        runner.message_hub.update_info(f"g18_attestation/{point}", record)
        deviations = {key: value for key, value in policy.items() if value != EXPECTED_POLICY[key]}
        if deviations or type(runner) is not TeacherRunner or not record["pre_cuda_policy_established"]:
            raise RuntimeError(f"G18 determinism attestation failed at {point}: deviations {deviations}, "
                               f"runner {record['runner_class']}, pre-CUDA marker "
                               f"{record['pre_cuda_policy_established']}.")
        self._attested.add(point)          # only a record that passed validation completes its point
