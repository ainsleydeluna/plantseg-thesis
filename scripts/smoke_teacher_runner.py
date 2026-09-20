#!/usr/bin/env python3
"""CPU verification of the G18 determinism seam — `TeacherRunner` + `TeacherDeterminismAttestationHook`.

WHY THIS IS A SEPARATE FILE, not an extension of an existing teacher smoke. The two candidates each
assert an invariant that hosting these checks would falsify:

  * `smoke_teacher_launch.py` asserts `provenance_versions_not_fabricated` — that the `mmsegmentation`
    and `mmcv` distributions are ABSENT — so it is a student-stack smoke and cannot pass inside the
    teacher image, which is exactly where `teacher_runner.py` must be exercised (it imports mmengine
    at module scope).
  * `smoke_teacher_config.py` states in its own docstring that "nothing here builds a model". G18
    cannot be proven without constructing a `Runner`, which builds one.

Extending either would mean weakening a check that already holds. So the seam gets its own file.

WHAT IS PROVEN HERE, and what is NOT. Every assertion below is CPU-only. The runner is constructed
through the real `TeacherRunner.from_cfg` path against a minimal in-memory config carrying a two-
parameter probe model — the B58 §6 "M arm" shape — so `setup_env` (which applies
`env_cfg.cudnn_benchmark=True`) and `set_randomness` run in their real order, with no dataset, no
checkpoint, no network and no CUDA. What this CANNOT show is the effective state at a real CUDA
forward: G20 stays open, and a passing run here is not a substitute for the official run's own
first-train and first-validation attestation.

Needs torch + mmengine 0.10.7 (the teacher stack), and `CUBLAS_WORKSPACE_CONFIG=:4096:8` inherited
from the environment, as the G8 image supplies it.

Exit codes: 0 all checks pass · 1 a check failed · 2 the environment cannot host the smoke.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import os
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_REQUIRED = ":4096:8"

results: list[tuple[str, bool, str]] = []
opened_paths: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect_raises(name: str, exc_type, needle: str, fn, *a, **kw) -> None:
    """The call must raise `exc_type` and say `needle`. Anything else — including success — fails."""
    try:
        fn(*a, **kw)
    except exc_type as e:
        check(name, needle in str(e), f"{type(e).__name__}: {str(e)[:90]}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")
    else:
        check(name, False, "no exception raised — the gate did not fail closed")


def guarded(section: str, fn, *a):
    """Run a section; an unexpected exception becomes a FAIL row so the table still prints."""
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f"{section}_section_raised_unexpectedly", False, f"{type(e).__name__}: {str(e)[:110]}")
        return None


class _Env:
    """Temporarily set or delete CUBLAS_WORKSPACE_CONFIG, then restore it exactly."""

    def __init__(self, value: str | None) -> None:
        self.value = value

    def __enter__(self) -> None:
        self.saved = os.environ.get(CUBLAS_ENV)
        if self.value is None:
            os.environ.pop(CUBLAS_ENV, None)
        else:
            os.environ[CUBLAS_ENV] = self.value

    def __exit__(self, *exc) -> None:
        if self.saved is None:
            os.environ.pop(CUBLAS_ENV, None)
        else:
            os.environ[CUBLAS_ENV] = self.saved


def rng_states():
    import numpy as np
    import torch
    return (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())


def states_equal(a, b) -> list[bool]:
    import numpy as np
    import torch
    return [torch.equal(a[0], b[0]),
            a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and tuple(a[1][2:]) == tuple(b[1][2:]),
            a[2] == b[2]]


def seeded_states(seed: int):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return rng_states()


def perturb(seed: int):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.rand(8), np.random.rand(8), random.random()


# ---------------------------------------------------------------- 1. module provenance + policy shape
def test_module_provenance(tr) -> None:
    on_disk = Path(tr.MODULE_PROVENANCE["path"])
    check("module_provenance_path_is_the_imported_file",
          on_disk == Path(tr.__file__).resolve(), str(on_disk))
    check("module_provenance_sha256_matches_bytes_on_disk",
          tr.MODULE_PROVENANCE["sha256"] == hashlib.sha256(on_disk.read_bytes()).hexdigest(),
          tr.MODULE_PROVENANCE["sha256"][:16] + "…")
    check("expected_policy_is_contract_b6_five_states",
          set(tr.EXPECTED_POLICY) == {"deterministic_algorithms", "deterministic_algorithms_warn_only",
                                      "cudnn_deterministic", "cudnn_benchmark", CUBLAS_ENV})
    check("expected_policy_warn_only_true", tr.EXPECTED_POLICY["deterministic_algorithms_warn_only"] is True,
          "contract B6 warn_only=True, NOT mmengine's strict form")
    check("expected_policy_cudnn_benchmark_false", tr.EXPECTED_POLICY["cudnn_benchmark"] is False)


# ---------------------------------------------------------------- 2. accept path, through from_cfg
def build_runner(tr, cfg_overrides: dict | None = None):
    from mmengine.config import Config
    cfg = dict(model=dict(type="G18ProbeNet"),
               work_dir=tempfile.mkdtemp(prefix="smoke_teacher_runner_"),
               randomness=dict(seed=42, deterministic=True),
               env_cfg=dict(cudnn_benchmark=True),     # the teacher config's real value: setup_env
               log_level="WARNING")                     # applies it BEFORE set_randomness runs
    cfg.update(cfg_overrides or {})
    return tr.TeacherRunner.from_cfg(Config(cfg))


def test_accept_path(tr, refs) -> dict:
    import torch
    perturb(1234)                                        # prove construction reseeds, not inherits
    runner = build_runner(tr)
    post = rng_states()

    check("from_cfg_returns_TeacherRunner", type(runner) is tr.TeacherRunner, type(runner).__name__)
    check("runner_seed_42", runner.seed == 42, repr(runner.seed))
    check("runner_deterministic_true", runner.deterministic is True, repr(runner.deterministic))
    policy = tr.effective_policy()
    check("effective_policy_equals_contract_b6", policy == tr.EXPECTED_POLICY,
          str({k: v for k, v in policy.items() if v != tr.EXPECTED_POLICY[k]} or "exact match"))
    check("cudnn_benchmark_override_beats_env_cfg", policy["cudnn_benchmark"] is False,
          "env_cfg.cudnn_benchmark=True was applied by setup_env, then overridden")
    check("pre_cuda_marker_set", tr.pre_cuda_policy_established(runner) is True)
    check("cuda_never_initialized_by_construction", torch.cuda.is_initialized() is False)

    same = states_equal(post, refs[42])
    check("construction_reseeds_numpy_to_seed_42", same[1])
    check("construction_reseeds_python_random_to_seed_42", same[2])
    check("construction_does_not_match_seed_43", states_equal(post, refs[43]) == [False, False, False],
          "negative control")
    return runner


# ---------------------------------------------------------------- 3. fail-closed inputs
def test_fail_closed(tr) -> None:
    import torch
    bare = lambda: tr.TeacherRunner.__new__(tr.TeacherRunner)   # noqa: E731 — no full construction needed

    with _Env(None):
        b = bare()
        expect_raises("missing_cublas_fails_closed", RuntimeError, tr.CUBLAS_MESSAGE,
                      b.set_randomness, 42, deterministic=True)
        check("missing_cublas_leaves_marker_unset", tr.pre_cuda_policy_established(b) is False)
        expect_raises("missing_cublas_fails_closed_through_from_cfg", RuntimeError, tr.CUBLAS_MESSAGE,
                      build_runner, tr)

    with _Env(":16:8"):
        b = bare()
        expect_raises("wrong_cublas_fails_closed", RuntimeError, "':16:8'",
                      b.set_randomness, 42, deterministic=True)
        check("wrong_cublas_leaves_marker_unset", tr.pre_cuda_policy_established(b) is False)

    expect_raises("deterministic_false_rejected", ValueError, "contract B6",
                  bare().set_randomness, 42, deterministic=False)
    expect_raises("deterministic_false_rejected_through_from_cfg", ValueError, "contract B6",
                  build_runner, tr, {"randomness": dict(seed=42, deterministic=False)})
    check("cuda_still_uninitialized_after_fail_closed_paths", torch.cuda.is_initialized() is False)


# ---------------------------------------------------------------- 4. the repeat-call guard
def test_repeat_call_guard(tr, runner, refs) -> None:
    b = tr.TeacherRunner.__new__(tr.TeacherRunner)
    with _Env(":16:8"):                                  # first call fails after the entered-flag is set
        expect_raises("first_call_fails_on_wrong_cublas", RuntimeError, "':16:8'",
                      b.set_randomness, 42, deterministic=True)
    expect_raises("second_call_without_marker_fails_closed", RuntimeError, "called again",
                  b.set_randomness, 42, deterministic=True)   # correct cuBLAS now — still refused

    # The resume path MMEngine really takes (runner.py:2060): a completed first call, then a re-entry.
    import torch
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    perturb(777)
    runner.set_randomness(**runner._randomness_cfg)
    check("resume_reentry_restores_contract_b6", tr.effective_policy() == tr.EXPECTED_POLICY)
    check("resume_reentry_restores_seed_42_rng", states_equal(rng_states(), refs[42]) == [True, True, True],
          "torch, numpy and python all back to a fresh seed 42")


# ---------------------------------------------------------------- 5. the attestation hook
class _NotATeacherRunner:
    """Carries everything the hook reads, but is not a TeacherRunner. Isolates the type check."""

    def __init__(self, runner, marker_attr: str) -> None:
        self.logger, self.message_hub, self.seed = runner.logger, runner.message_hub, runner.seed
        setattr(self, marker_attr, True)


def test_hook(tr, runner) -> None:
    import torch
    from src.training.teacher_runner import _PRE_CUDA_MARKER   # attribute name the runner really sets

    hook = tr.TeacherDeterminismAttestationHook()
    hook.before_train_iter(runner, 0)
    hook.before_val_iter(runner, 0)
    check("hook_accepts_valid_train_state", "first_train_iter" in hook._attested)
    check("hook_accepts_valid_val_state", "first_val_iter" in hook._attested)

    rec = runner.message_hub.get_info("g18_attestation/first_train_iter")
    check("record_policy_is_contract_b6", rec["policy"] == tr.EXPECTED_POLICY)
    check("record_names_TeacherRunner", rec["runner_class"].endswith(".TeacherRunner"), rec["runner_class"])
    check("record_carries_pre_cuda_marker", rec["pre_cuda_policy_established"] is True)
    check("record_cuda_not_initialized", rec["cuda_initialized"] is False)
    check("record_carries_module_provenance", rec["teacher_runner_module"] == tr.MODULE_PROVENANCE)
    check("record_seed_42", rec["seed"] == 42)

    hook.before_train_iter(runner, 99)                   # already attested -> no-op, not a re-record
    check("hook_attests_each_point_once",
          runner.message_hub.get_info("g18_attestation/first_train_iter")["batch_idx"] == 0)

    # (a) deviating policy — a fresh hook, so the point is unattested and the check can fire.
    deviated = tr.TeacherDeterminismAttestationHook()
    torch.backends.cudnn.benchmark = True
    expect_raises("hook_rejects_policy_deviation", RuntimeError, "cudnn_benchmark",
                  deviated.before_train_iter, runner, 0)
    check("failed_attestation_does_not_latch", "first_train_iter" not in deviated._attested,
          "fail-closed must not mark the point attested")
    check("failed_attestation_still_recorded_in_message_hub",
          runner.message_hub.get_info("g18_attestation/first_train_iter")["policy"]["cudnn_benchmark"] is True,
          "the record is written before it is validated — the deviation stays on the forensic trail")
    runner.set_randomness(**runner._randomness_cfg)      # restore contract B6 for the rest of the run
    check("policy_restored_after_deviation_test", tr.effective_policy() == tr.EXPECTED_POLICY)

    # (b) wrong runner type, with the marker present so ONLY the type check can fail it.
    expect_raises("hook_rejects_non_TeacherRunner", RuntimeError, "_NotATeacherRunner",
                  tr.TeacherDeterminismAttestationHook().before_train_iter,
                  _NotATeacherRunner(runner, _PRE_CUDA_MARKER), 0)

    # (c) missing pre-CUDA marker on an otherwise valid runner.
    delattr(runner, _PRE_CUDA_MARKER)
    try:
        check("marker_removal_is_visible", tr.pre_cuda_policy_established(runner) is False)
        expect_raises("hook_rejects_missing_pre_cuda_marker", RuntimeError, "pre-CUDA marker",
                      tr.TeacherDeterminismAttestationHook().before_train_iter, runner, 0)
    finally:
        setattr(runner, _PRE_CUDA_MARKER, True)
    check("marker_restored", tr.pre_cuda_policy_established(runner) is True)


# ---------------------------------------------------------------- 6. hygiene
def test_hygiene() -> None:
    import torch
    check("cuda_never_initialized_anywhere_in_this_smoke", torch.cuda.is_initialized() is False)
    data_root = os.environ.get("PLANTSEG_DATA_ROOT", "/workspace/plantseg_data/plantseg")
    touched = [p for p in opened_paths if data_root in p]
    check("dataset_root_never_opened", touched == [], f"{len(touched)} path(s) under {data_root}")
    split = [p for p in opened_paths if f"{os.sep}test{os.sep}" in p or "/test/" in p]
    check("held_out_test_split_never_opened", split == [], f"{len(split)} test-split path(s)")
    inside_repo = [p for p in opened_paths
                   if p.startswith(str(REPO)) and Path(p).suffix in (".pth", ".pt", ".ckpt")]
    check("no_checkpoint_opened", inside_repo == [])


def main() -> int:
    if os.environ.get(CUBLAS_ENV) != CUBLAS_REQUIRED:
        print(f"ENVIRONMENT: {CUBLAS_ENV} must be inherited as {CUBLAS_REQUIRED!r} before the "
              f"interpreter starts (G8); got {os.environ.get(CUBLAS_ENV)!r}. Run inside the "
              "plantseg-teacher image, which declares it.")
        return 2
    try:
        import torch  # noqa: F401
        import src.training.teacher_runner as tr
    except ImportError as e:
        print(f"ENVIRONMENT: the teacher stack is not importable ({e}). Run inside the teacher "
              "stack (torch 2.1.0 + mmengine 0.10.7), e.g. the plantseg-teacher image.")
        return 2

    import torch
    from mmengine.registry import MODELS

    @MODELS.register_module()
    class G18ProbeNet(torch.nn.Module):
        """Two parameters. Enough for Runner to build a model; nothing to do with the teacher graph."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

    real_open = builtins.open

    def watched_open(file, *a, **kw):
        opened_paths.append(str(file))
        return real_open(file, *a, **kw)

    print("=" * 78)
    print("G18 TEACHER RUNNER SMOKE — determinism seam + attestation hook (CPU only, no dataset)")
    print(f"torch {torch.__version__} | cuda available {torch.cuda.is_available()} | "
          f"{CUBLAS_ENV}={os.environ.get(CUBLAS_ENV)!r}")
    print(f"teacher_runner.py sha256 {tr.MODULE_PROVENANCE['sha256'][:16]}…")
    print("=" * 78)

    refs = {42: seeded_states(42), 43: seeded_states(43)}
    builtins.open = watched_open
    try:
        # A section that raises is recorded as a FAIL rather than aborting the run: a real regression
        # must still print the table below, so the reader sees WHICH invariant broke and not merely
        # where the process stopped.
        runner = guarded("accept_path", test_accept_path, tr, refs)
        guarded("module_provenance", test_module_provenance, tr)
        if runner is not None:
            guarded("fail_closed", test_fail_closed, tr)
            guarded("repeat_call_guard", test_repeat_call_guard, tr, runner, refs)
            guarded("attestation_hook", test_hook, tr, runner)
        else:
            check("accept_path_produced_a_runner", False, "later sections could not run")
    finally:
        builtins.open = real_open
    test_hygiene()

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    print("SCOPE: CPU only. G20 (first-forward attestation on CUDA) is NOT addressed by this smoke.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
