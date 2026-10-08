#!/usr/bin/env python3
"""Real-run launch gates of the distillation trainer (lane L-KD-HARDEN item 2). CPU, no dataset, no GPU.

Every real-run gate of item 2 refuses with exit code 2 and its bracketed reason, the legal launch of
each stage (E2, E3, A, F, G) passes every gate, and no dataloader or teacher is built before a refusal.

Isolation: DATA["root"] points at a synthetic TRAIN/VAL-only staged root (two empty named files per
split, with isolation.DEFAULT_EXPECTED_COUNTS set to match: M11 counts the TRAIN/VAL files and checks
the TEST surfaces for existence only), `--device cuda` passes the CUDA-device gate on this CPU box, the
teacher checkpoint is a throwaway file (only its existence is checked before loading), and
`load_frozen_teacher`, `build_dataloader` and `run` are replaced by counting stubs. The checks that
run() repeats at its entry for direct calls (item 2i, CUDA initialised before seeding; item 2c's
schedule; in real mode a missing --ckpt-dir, any clip value and an off-grid alpha) are exercised on the
REAL run(); `build_student` is replaced by a stub that raises, so no student (and no ImageNet download)
can be built even if one of them failed.

  2a  lambda: NaN, inf, <= 0 or off the AM-2 grid -> [lambda_grid]; missing -> refused (existing gate)
  2b  --ckpt-dir: missing or empty -> [ckpt_dir_required] (main, and run() for a direct real call); an
      existing non-empty directory or a file -> [ckpt_dir_not_fresh]; an existing empty directory is fresh
  2c  --max-iters above the 80,000 horizon -> [max_iters_above_horizon] in both modes; a real run with
      any other value -> [max_iters_not_horizon]; an explicit 80000 is legal; run() repeats the check
  2d  --max-val-batches -> [max_val_batches]; --val-interval other than 4000 -> [val_interval]
  2e  --num-workers missing or < 1 -> [num_workers]
  2f  seed not in {42, 43, 44}, or not 42 for A/F/G -> [seed]; batch size other than 16 -> [batch_size];
      --init none -> [init]
  2g  --grad-clip-norm in a real run (every stage) -> [grad_clip_am7], in main and in run(); a dry run
      accepts it
  2h  cuDNN TF32 off, matmul TF32 on, float32_matmul_precision not "highest" or NVIDIA_TF32_OVERRIDE set
      -> [tf32]; nothing is set by the trainer
  2i  CUDA initialised before the teacher load -> [cuda_initialized_before_teacher] (main, C5); CUDA
      initialised by the time run() starts -> [cuda_initialized_before_seed] (run() entry, before
      set_seed; the teacher already exists by construction, no dataloader or student is built)
  2j  more than one alpha<v> in --ckpt-dir (glued on or not) -> [ckpt_dir_alpha]; run() refuses an
      off-grid alpha in real mode; the lambda-semantics gate refuses in dry mode too; the grid error
      prints repr(alpha)
  Q10 a --ckpt-dir that resolves inside the repository (the repository itself, a path under it, a
      relative path and a path through a symlink that resolve into it; the test of train_e1's
      _assert_outside_repo) -> [ckpt_dir_in_repo] in a real run, before the fresh-directory gate, and
      nothing is created; a path whose resolution fails (pathlib's symlink-loop RuntimeError, simulated)
      is refused with [ckpt_dir_unresolvable] before any teacher or loader and never as in-repo
      (K8-2(d): also a simulated OSError and a real NUL-byte path); --log-every < 1 ->
      [log_every] in both modes (the dry cases pass --teacher-ckpt, so a gate after the teacher load
      would show as a load). A host without symlink rights (Windows) SKIPs the symlink case, and a
      working directory on another drive SKIPs the relative one: a named SKIP, never a PASS.
  R6   --teacher-ckpt-sha256: absent in a real run -> [teacher_ckpt_sha256_required] (after the TF32
      gate, before the CUDA-order gate); "", 63 characters or uppercase -> [teacher_ckpt_sha256_format]
      and without --teacher-ckpt -> [teacher_ckpt_sha256_without_ckpt], in both modes; the legal launch
      hands the value to the teacher load; the REAL loader refuses a wrong value with
      [teacher_ckpt_sha256_mismatch] before any torch.load or builder call; a loader raising
      TeacherStateDictMismatch or TeacherChecksumMismatch exits 2 in both modes; a dry run with
      --teacher-ckpt and no value stays legal
  abort  main() turns run()'s RunAborted into exit code 3 (td.ABORTED_EXIT, distinct from 0, 1 and 2) and
      one RESULT: ABORTED line on stdout (a stub run() that raises)
  legal  each stage's legal launch reaches run() with the recipe values (mode real, 80,000 iterations,
      batch 16, ImageNet init, VAL every 4,000 over the full set, explicit workers, no clipping)
  order  M11 and the CUDA-device refusals keep their text and fire before every item-2 gate
  L4   lane 4(a) and PL-28 (K2), in both modes: --lambda-selection on a stage without Logit KD and
      --alpha-selection on one without the feature-map CWD term -> [selection_term]; a path that is not an
      existing file (never created, a directory) or an existing file that cannot be read ->
      [selection_file_missing], before any teacher or loader; --records-commit other than 40 lowercase hex
      -> [records_commit_format]; the legal E2/E3 seed-43 launches, the arms A and F at seed 42 with
      --alpha-selection, and a dry run hand run() the resolved path and sha256 of each file (a relative path
      and a path through a symlink are recorded resolved; a named SKIP where the host has neither), and no
      flag hands it nulls; launch_fields() per stage and horizon, and its refusals for direct calls; the
      REAL run(), stubbed up to its training loop, writes the six launch fields last in run_meta, after
      teacher_mock

Fixtures (RULING 2, SL-1): every path the smoke creates is under one scratch directory named
kdh_gates_<pid>_<n> (no random suffix); the Q10 cases name in-repository paths that are never created, and a
path with a "test" component is only ever named, never created.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import src.data.isolation as isolation  # noqa: E402
import src.training.train_distill as td  # noqa: E402
from src.distill.segnext_teacher import TeacherCheckpointInvalid, TeacherStateDictMismatch  # noqa: E402
from src.distill.teacher import TeacherChecksumMismatch  # noqa: E402
from configs.e1_student import E1_STUDENT  # noqa: E402
from src.distill import FrozenTeacher, MockTeacher  # noqa: E402

NC = 116
results: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []
COUNT = {"teacher": 0, "loader": 0, "run": 0, "student": 0}
SEEN: list[dict] = []
LOAD_KW: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def skip(name: str, reason: str) -> None:
    """A case this host cannot run: printed with its reason, never counted as a PASS."""
    SKIPPED.append((name, reason))


# ------------------------------------------------------------------------------------ stubs
class _WouldBuildStudent(Exception):
    pass


def stub_load_frozen_teacher(*_a, **k):
    COUNT["teacher"] += 1
    LOAD_KW.append(dict(k))
    return FrozenTeacher(MockTeacher(NC))


def stub_build_dataloader(*_a, **_k):
    COUNT["loader"] += 1
    raise RuntimeError("a dataloader was requested")


def stub_run(**kwargs):
    COUNT["run"] += 1
    SEEN.append(kwargs)
    return 0


def stub_build_student(*_a, **_k):
    COUNT["student"] += 1
    raise _WouldBuildStudent()


def reset() -> None:
    for k in COUNT:
        COUNT[k] = 0
    SEEN.clear()
    LOAD_KW.clear()


def call_main(argv: list[str]) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        try:
            rc = td.main(argv)
        except SystemExit as e:                       # argparse errors
            rc = int(e.code) if e.code is not None else 0
    return rc, err.getvalue()


# ------------------------------------------------------------------------------- fixtures
def scratch_dir(prefix: str) -> Path:
    """A new directory under the system temp dir, named <prefix>_<pid>_<n> as in the selection smoke (RULING
    2): no random suffix can spell 'test' (SL-1), and it lies outside the repository."""
    n = 0
    while True:
        n += 1
        p = Path(tempfile.gettempdir()) / f"{prefix}_{os.getpid()}_{n}"
        if "test" in str(p).lower():
            raise SystemExit(f"refusing to create {p}: its path contains 'test' (SL-1); set TMPDIR elsewhere")
        if REPO == p.resolve() or REPO in p.resolve().parents:
            raise SystemExit(f"refusing to create {p} inside the repository")
        try:
            p.mkdir()
        except FileExistsError:
            continue
        return p


TMP = scratch_dir("kdh_gates")


def staged_root(name: str = "staged") -> Path:
    root = TMP / name
    for split in ("train", "val"):
        for kind, ext in (("images", ".jpg"), ("annotations", ".png")):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (root / kind / split / f"s{i}{ext}").touch()
    return root


_n = [0]


def fresh(name: str) -> str:
    """A --ckpt-dir path that does not exist yet (a fresh run directory)."""
    _n[0] += 1
    return str(TMP / "ckpt" / f"{_n[0]:03d}" / name)


TEACHER = TMP / "teacher.pth"
TEACHER.write_bytes(b"not read before the teacher load")
TEACHER_SHA256 = hashlib.sha256(TEACHER.read_bytes()).hexdigest()
BASE = ["--real-run", "--confirm-real-run", "--device", "cuda", "--teacher-ckpt", str(TEACHER),
        "--teacher-ckpt-sha256", TEACHER_SHA256, "--num-workers", "12"]


def legal(stage: str, **over) -> list[str]:
    """The legal real launch of `stage`: seed 42, --num-workers 12, lambda 1 (E2, E3), alpha 50 (E3, A;
    F takes the default 50) and a fresh --ckpt-dir carrying one alpha token where the stage has the
    feature-map term."""
    argv = BASE + ["--stage", stage]
    if stage in ("e2", "e3"):
        argv += ["--lambda-logit", "1"]
    if stage in ("e3", "a"):
        argv += ["--alpha", "50"]
    name = {"e2": "e2_s42", "e3": "e3_s42_alpha50", "a": "a_s42_alpha50", "f": "f_s42_alpha50",
            "g": "g_s42"}[stage]
    argv += ["--ckpt-dir", over.pop("ckpt_dir", fresh(name))]
    for flag, value in over.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return argv


def with_value(argv: list[str], flag: str, value: str) -> list[str]:
    """`argv` with the value that follows `flag` replaced."""
    i = argv.index(flag)
    return argv[:i + 1] + [value] + argv[i + 2:]


def without(argv: list[str], flag: str) -> list[str]:
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok == flag:
            skip = True
            continue
        out.append(tok)
    return out


def refused(label: str, argv: list[str], code: str, *, teacher: int = 0) -> None:
    reset()
    try:
        rc, err = call_main(argv)
    except Exception as e:  # noqa: BLE001 - a raise instead of a refusal is a FAIL of this check
        rc, err = None, f"raised {type(e).__name__}: {e}"
    check(f"{label}_refused_{code}", rc == 2 and f"[{code}]" in err and COUNT["loader"] == 0
          and COUNT["teacher"] == teacher and COUNT["run"] == 0,
          f"rc={rc} teacher={COUNT['teacher']} loader={COUNT['loader']} run={COUNT['run']} "
          f"err={err.strip()[-160:]}")


# ---------------------------------------------------------------------------------- tests
def test_legal_launches() -> None:
    for stage in ("e2", "e3", "a", "f", "g"):
        reset()
        rc, err = call_main(legal(stage))
        kw = SEEN[-1] if SEEN else {}
        want_lambda = 1.0 if stage in ("e2", "e3") else None
        want_alpha = 50.0 if stage in ("e3", "a", "f") else None
        ok = (rc == 0 and COUNT["run"] == 1 and COUNT["teacher"] == 1 and COUNT["loader"] == 0
              and kw.get("mode") == "real" and kw.get("device") == "cuda"
              and kw.get("pretrained") == E1_STUDENT["init_weights"] and kw.get("batch_size") == 16
              and kw.get("max_iters") == 80000 and kw.get("val_interval") == 4000
              and kw.get("max_val_batches") is None and kw.get("num_workers") == 12
              and kw.get("grad_clip_norm") is None and kw.get("seed") == 42
              and kw.get("lambda_logit") == want_lambda and kw.get("alpha") == want_alpha)
        check(f"legal_{stage}_passes_every_gate", ok, f"rc={rc} {err.strip()[-160:]} kw={kw}")
    for stage in ("e2", "e3"):
        for seed in (43, 44):
            reset()
            rc, err = call_main(legal(stage, seed=seed))
            check(f"legal_{stage}_seed{seed}", rc == 0 and SEEN and SEEN[-1].get("seed") == seed,
                  err.strip()[-160:])
    reset()
    empty = Path(fresh("e2_s42_existing_empty"))
    empty.mkdir(parents=True)
    rc, err = call_main(legal("e2", ckpt_dir=str(empty), max_iters=80000, val_interval=4000,
                              batch_size=16, init="imagenet"))
    check("legal_explicit_recipe_values_and_empty_existing_dir", rc == 0 and COUNT["run"] == 1,
          err.strip()[-160:])


def test_lambda() -> None:
    for bad in ("nan", "inf", "0", "-1", "3", "0.3"):
        refused(f"2a_lambda_{bad}", with_value(legal("e2"), "--lambda-logit", bad), "lambda_grid")
    reset()
    rc, err = call_main(without(legal("e2"), "--lambda-logit"))
    check("2a_lambda_missing_refused", rc == 2 and "--lambda-logit is required" in err
          and COUNT["teacher"] == 0, err.strip()[-120:])
    for lam in ("0.25", "0.5", "2", "4"):
        reset()
        rc, _ = call_main(with_value(legal("e2"), "--lambda-logit", lam))
        check(f"2a_lambda_{lam}_on_grid_accepted", rc == 0 and bool(SEEN)
              and SEEN[-1]["lambda_logit"] == float(lam))


def test_ckpt_dir() -> None:
    refused("2b_missing", without(legal("g"), "--ckpt-dir"), "ckpt_dir_required")
    refused("2b_empty_string", legal("g", ckpt_dir=""), "ckpt_dir_required")
    used = Path(fresh("g_s42_used"))
    used.mkdir(parents=True)
    (used / "g_telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    refused("2b_nonempty_dir", legal("g", ckpt_dir=str(used)), "ckpt_dir_not_fresh")
    afile = Path(fresh("g_s42_file"))
    afile.parent.mkdir(parents=True, exist_ok=True)
    afile.write_text("x", encoding="utf-8")
    refused("2b_path_is_a_file", legal("g", ckpt_dir=str(afile)), "ckpt_dir_not_fresh")


def test_ckpt_dir_in_repo_and_log_every() -> None:
    """Q10: an in-repository --ckpt-dir and a log cadence below 1 are refused by name before any loader
    or teacher is built."""
    refused("q10_repo_itself", legal("g", ckpt_dir=str(REPO)), "ckpt_dir_in_repo")
    under = REPO / "kdh_ckpt_never_created"
    refused("q10_under_repo", legal("g", ckpt_dir=str(under / "g_s42")), "ckpt_dir_in_repo")
    try:
        rel = os.path.relpath(REPO / "kdh_ckpt_never_created_rel", start=os.getcwd())
    except ValueError as e:                       # Windows: the working directory is on another drive
        rel = None
        skip("q10_relative_into_repo_refused_ckpt_dir_in_repo", f"no relative path from here: {e}")
    if rel is not None:
        refused("q10_relative_into_repo", legal("g", ckpt_dir=rel), "ckpt_dir_in_repo")
    link = TMP / "link_into_repo"
    try:
        link.symlink_to(REPO, target_is_directory=True)
    except (OSError, NotImplementedError) as e:   # Windows without the symlink privilege
        skip("q10_symlink_into_repo_refused_ckpt_dir_in_repo", f"os.symlink not permitted on this host: {e}")
    else:
        try:
            refused("q10_symlink_into_repo", legal("g", ckpt_dir=str(link / "kdh_ckpt_never_created_link")),
                    "ckpt_dir_in_repo")
        finally:
            link.unlink()
    created = [p for p in (under, REPO / "kdh_ckpt_never_created_rel", REPO / "kdh_ckpt_never_created_link")
               if p.exists()]
    check("q10_nothing_created_in_repo", not created, str(created))
    # K8-2(d): a --ckpt-dir whose resolution fails is refused by name, exit 2, never as in-repo
    marker = "kdh_unresolvable_ckpt_dir"
    real_resolve = Path.resolve
    for label, error in (("symlink_loop", RuntimeError), ("oserror", OSError)):
        def resolve(self, strict=False, _error=error):   # pathlib's symlink-loop error (Python <= 3.12)
            if marker in str(self):
                raise _error(f"Symlink loop from {str(self)!r}")
            return real_resolve(self, strict=strict)
        reset()
        Path.resolve = resolve
        try:
            rc, err = call_main(legal("g", ckpt_dir=str(TMP / marker / "g_s42")))
            outcome = f"rc={rc} {err.strip()[-120:]}"
        except Exception as e:  # noqa: BLE001 - a raise is the failure this check detects
            rc, err, outcome = None, "", f"raised {type(e).__name__}: {e}"
        finally:
            Path.resolve = real_resolve
        check(f"q10_unresolvable_ckpt_dir_{label}_refused_ckpt_dir_unresolvable",
              rc == 2 and "[ckpt_dir_unresolvable]" in err and "[ckpt_dir_in_repo]" not in err
              and COUNT["teacher"] == 0 and COUNT["loader"] == 0 and COUNT["run"] == 0, outcome)
    refused("q10_nul_byte_ckpt_dir", legal("g", ckpt_dir=str(TMP / "kdh_nul\x00byte" / "g_s42")),
            "ckpt_dir_unresolvable")
    for bad in ("0", "-1"):
        refused(f"q10_real_log_every_{bad}", legal("g", log_every=bad), "log_every")
        refused(f"q10_dry_log_every_{bad}", ["--stage", "e2", "--dry-run", "--teacher-ckpt", str(TEACHER),
                                             "--log-every", bad], "log_every")
    reset()
    rc, err = call_main(["--stage", "e2", "--dry-run", "--log-every", "1"])
    check("q10_dry_log_every_1_accepted", rc == 0 and COUNT["run"] == 1 and SEEN[-1].get("log_every") == 1,
          err.strip()[-120:])


def test_teacher_sha256() -> None:
    """R6 (L-CKPT-GUARD): the --teacher-ckpt-sha256 gates, the pass-through and the two load refusals."""
    refused("r6_real_without_sha256", without(legal("g"), "--teacher-ckpt-sha256"),
            "teacher_ckpt_sha256_required")
    # placement (patch 7): after the TF32 gate, before the CUDA-order gate of the teacher load
    torch.backends.cudnn.allow_tf32 = False
    try:
        refused("r6_tf32_gate_precedes_the_sha256_gate", without(legal("g"), "--teacher-ckpt-sha256"), "tf32")
    finally:
        torch.backends.cudnn.allow_tf32 = True
    real_is_init = torch.cuda.is_initialized
    torch.cuda.is_initialized = lambda: True
    try:
        refused("r6_sha256_gate_precedes_the_cuda_order_gate", without(legal("g"), "--teacher-ckpt-sha256"),
                "teacher_ckpt_sha256_required")
    finally:
        torch.cuda.is_initialized = real_is_init
    for label, value in (("empty", ""), ("63_chars", TEACHER_SHA256[:63]),
                         ("uppercase", TEACHER_SHA256.upper())):
        refused(f"r6_real_sha256_{label}", with_value(legal("g"), "--teacher-ckpt-sha256", value),
                "teacher_ckpt_sha256_format")
        refused(f"r6_dry_sha256_{label}", ["--stage", "e2", "--dry-run", "--teacher-ckpt", str(TEACHER),
                                           "--teacher-ckpt-sha256", value], "teacher_ckpt_sha256_format")
    refused("r6_real_sha256_without_ckpt", without(legal("g"), "--teacher-ckpt"),
            "teacher_ckpt_sha256_without_ckpt")
    refused("r6_dry_sha256_without_ckpt", ["--stage", "e2", "--dry-run", "--teacher-ckpt-sha256",
                                           TEACHER_SHA256], "teacher_ckpt_sha256_without_ckpt")
    reset()
    rc, err = call_main(legal("e3"))
    check("r6_legal_launch_passes_sha256_to_the_teacher_load",
          rc == 0 and LOAD_KW and LOAD_KW[-1].get("expected_sha256") == TEACHER_SHA256,
          f"rc={rc} {LOAD_KW} {err.strip()[-120:]}")
    reset()
    rc, err = call_main(["--stage", "e2", "--dry-run", "--teacher-ckpt", str(TEACHER)])
    check("r6_dry_with_ckpt_and_no_sha256_stays_legal", rc == 0 and COUNT["run"] == 1
          and LOAD_KW and LOAD_KW[-1].get("expected_sha256") is None, f"rc={rc} {err.strip()[-120:]}")
    # the REAL loader: a wrong value is refused before any torch.load or builder call
    import src.distill.segnext_teacher as st
    calls = {"torch_load": 0, "builder": 0}
    real_torch_load, real_builder = torch.load, st.segnext_builder

    def counting_load(*a, **k):
        calls["torch_load"] += 1
        return real_torch_load(*a, **k)

    def counting_builder(*a, **k):
        calls["builder"] += 1
        return real_builder(*a, **k)
    torch.load, st.segnext_builder = counting_load, counting_builder
    td.load_frozen_teacher = REAL_LOAD
    reset()
    try:
        rc, err = call_main(with_value(legal("g"), "--teacher-ckpt-sha256", "0" * 64))
    finally:
        torch.load, st.segnext_builder = real_torch_load, real_builder
        td.load_frozen_teacher = stub_load_frozen_teacher
    check("r6_wrong_sha256_refused_before_any_load", rc == 2 and "[teacher_ckpt_sha256_mismatch]" in err
          and calls == {"torch_load": 0, "builder": 0} and COUNT["run"] == 0,
          f"rc={rc} {calls} {err.strip()[-160:]}")
    # a loader raising either load refusal exits 2 in both modes; any other loader error is not mapped
    for exc in (TeacherStateDictMismatch(missing=["backbone.x"], source="fixture"),
                TeacherChecksumMismatch("0" * 64, "1" * 64, "fixture")):
        def raising(*_a, _exc=exc, **_k):
            COUNT["teacher"] += 1
            raise _exc
        td.load_frozen_teacher = raising
        try:
            for mode, argv in (("real", legal("g")),
                               ("dry", ["--stage", "e2", "--dry-run", "--teacher-ckpt", str(TEACHER)])):
                reset()
                rc, err = call_main(argv)
                check(f"r6_{mode}_{type(exc).__name__}_exits_2", rc == 2 and f"[{exc.code}]" in err
                      and COUNT["run"] == 0, f"rc={rc} {err.strip()[-160:]}")
        finally:
            td.load_frozen_teacher = stub_load_frozen_teacher

    def invalid(*_a, **_k):
        raise TeacherCheckpointInvalid("an unrelated loader error")
    td.load_frozen_teacher = invalid
    reset()
    try:
        call_main(legal("g"))
        outcome = "returned"
    except TeacherCheckpointInvalid:
        outcome = "raised"
    finally:
        td.load_frozen_teacher = stub_load_frozen_teacher
    check("r6_other_loader_errors_are_not_mapped", outcome == "raised", outcome)


def test_abort_exit() -> None:
    """Item 3: run() raises RunAborted; main() prints one RESULT: ABORTED line and returns 3."""
    record = {"event": "run_abort", "iter": 1234, "rule": "AM-7(a)", "cause": "student_divergence",
              "detail": {"loss": "nan", "grad_norm": None, "window_mean": None, "running_min": 0.8,
                         "ratio": None, "threshold": None},
              "input_finite": True, "teacher_finite": True, "params_finite": True, "n_val": 0,
              "wall_clock": 0.0}

    def aborting_run(**kwargs):
        COUNT["run"] += 1
        raise td.RunAborted(record)
    reset()
    out, err = io.StringIO(), io.StringIO()
    td.run = aborting_run
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = td.main(legal("e2"))
    except td.RunAborted as e:                    # main() let it through: a FAIL, not a traceback
        rc = f"raised {e}"
    finally:
        td.run = stub_run
    want = ("RESULT: ABORTED rule=AM-7(a) iter=1234 cause=student_divergence; the run_abort record is the "
            "telemetry's last row; do not relaunch this run (AM-7a; a fault follows AM-8a)")
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("RESULT")]
    check("abort_main_exits_3_with_one_result_line", rc == 3 and td.ABORTED_EXIT == 3 and lines == [want]
          and COUNT["run"] == 1 and "RESULT" not in err.getvalue(),
          f"rc={rc} lines={lines} err={err.getvalue().strip()[-120:]}")


def test_schedule() -> None:
    refused("2c_real_above", legal("g", max_iters=80001), "max_iters_above_horizon")
    refused("2c_real_not_horizon", legal("g", max_iters=100), "max_iters_not_horizon")
    refused("2c_dry_above", ["--stage", "g", "--dry-run", "--max-iters", "80001"], "max_iters_above_horizon")


def test_val() -> None:
    refused("2d_max_val_batches", legal("g", max_val_batches=2), "max_val_batches")
    refused("2d_val_interval_2000", legal("g", val_interval=2000), "val_interval")
    refused("2d_val_interval_0", legal("g", val_interval=0), "val_interval")


def test_workers() -> None:
    refused("2e_missing", without(legal("g"), "--num-workers"), "num_workers")
    refused("2e_zero", with_value(legal("g"), "--num-workers", "0"), "num_workers")


def test_seed_batch_init() -> None:
    refused("2f_seed45_e2", legal("e2", seed=45), "seed")
    for stage in ("a", "f", "g"):
        refused(f"2f_seed43_{stage}", legal(stage, seed=43), "seed")
    refused("2f_batch8", legal("g", batch_size=8), "batch_size")
    refused("2f_batch0", legal("g", batch_size=0), "batch_size")
    refused("2f_init_none", legal("g", init="none"), "init")


def test_clip() -> None:
    for stage in ("e2", "e3", "a", "f", "g"):
        refused(f"2g_clip_{stage}", legal(stage, grad_clip_norm=1.0), "grad_clip_am7")
    check("2g_gate_function_semantics", td.grad_clip_gate_error(None) is None
          and td.grad_clip_gate_error(1.0) is not None and "AM-7" in td.grad_clip_gate_error(1.0))
    reset()
    rc, err = call_main(["--stage", "e2", "--dry-run", "--grad-clip-norm", "1.0"])
    check("2g_dry_run_accepts_clip", rc == 0 and SEEN and SEEN[-1].get("grad_clip_norm") == 1.0
          and SEEN[-1].get("mode") == "dry", err.strip()[-120:])


def test_tf32() -> None:
    cases = (("cudnn_off", lambda: setattr(torch.backends.cudnn, "allow_tf32", False),
              lambda: setattr(torch.backends.cudnn, "allow_tf32", True)),
             ("matmul_on", lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", True),
              lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", False)),
             ("precision_high", lambda: torch.set_float32_matmul_precision("high"),
              lambda: torch.set_float32_matmul_precision("highest")),
             ("override_env", lambda: os.environ.__setitem__("NVIDIA_TF32_OVERRIDE", "0"),
              lambda: os.environ.pop("NVIDIA_TF32_OVERRIDE", None)))
    check("2h_defaults_hold_here", td.tf32_gate_error(td.tf32_state()) is None, str(td.tf32_state()))
    for label, do, undo in cases:
        do()
        try:
            refused(f"2h_{label}", legal("g"), "tf32")
        finally:
            undo()
    check("2h_defaults_restored", td.tf32_gate_error(td.tf32_state()) is None, str(td.tf32_state()))


def test_cuda_order() -> None:
    real_is_init = torch.cuda.is_initialized
    torch.cuda.is_initialized = lambda: True
    try:
        refused("2i_main_before_teacher", legal("g"), "cuda_initialized_before_teacher")
    finally:
        torch.cuda.is_initialized = real_is_init
    # run() entry: CUDA "becomes" initialised during the teacher load; the REAL run() must refuse
    # before set_seed, with no dataloader and no student built.
    state = {"cuda": False}

    def loader_that_inits(*a, **k):
        state["cuda"] = True
        return stub_load_frozen_teacher(*a, **k)

    td.run, td.load_frozen_teacher = REAL_RUN, loader_that_inits
    torch.cuda.is_initialized = lambda: state["cuda"]
    seeded = []
    real_set_seed = td.set_seed
    td.set_seed = lambda s: seeded.append(s)
    reset()
    try:
        rc, err = call_main(legal("g"))
    except _WouldBuildStudent:
        rc, err = "student-built", ""
    finally:
        torch.cuda.is_initialized = real_is_init
        td.run, td.load_frozen_teacher, td.set_seed = stub_run, stub_load_frozen_teacher, real_set_seed
    check("2i_run_entry_refused_cuda_initialized_before_seed",
          rc == 2 and "[cuda_initialized_before_seed]" in err and COUNT["loader"] == 0
          and COUNT["student"] == 0 and COUNT["teacher"] == 1 and seeded == [],
          f"rc={rc} seeded={seeded} loader={COUNT['loader']} student={COUNT['student']} {err.strip()[-160:]}")


def test_alpha_and_semantics() -> None:
    refused("2j_two_alpha_tokens", legal("e3", ckpt_dir=fresh("e3_alpha50_alpha25")), "ckpt_dir_alpha")
    refused("2j_same_token_twice", legal("e3", ckpt_dir=fresh("e3_alpha50_s42_alpha50")), "ckpt_dir_alpha")
    refused("2j_no_token", legal("e3", ckpt_dir=fresh("e3_s42")), "ckpt_dir_alpha")
    refused("2j_glued_second_value", legal("e3", ckpt_dir=fresh("e3_s42_alpha50alpha25")), "ckpt_dir_alpha")
    check("2j_glued_second_value_refused_for_either_alpha",
          td.ckpt_dir_alpha_error("/w/e3_s42_alpha25alpha50", 25.0) is not None
          and td.ckpt_dir_alpha_error("/w/e3_s42_alpha25alpha50", 50.0) is not None)
    check("2j_single_token_accepted", td.ckpt_dir_alpha_error("/w/e3_s42_alpha50", 50.0) is None)
    msg = td.alpha_gate_error(50.0000001, allow_offgrid=False, mode="dry") or ""
    check("2j_grid_error_prints_repr", "50.0000001" in msg, msg[:120])
    reset()
    rc, err = call_main(["--stage", "e2", "--dry-run", "--lambda-semantics", "logitkd@elsewhere"])
    check("2j_dry_run_refuses_semantics_mismatch", rc == 2 and "[lambda_semantics]" in err
          and COUNT["run"] == 0, err.strip()[-120:])
    reset()
    rc, err = call_main(["--stage", "e2", "--dry-run", "--lambda-semantics", td.LOGIT_KD_SEMANTICS])
    check("2j_dry_run_accepts_matching_semantics", rc == 0 and COUNT["run"] == 1, err.strip()[-120:])
    refused("2j_real_semantics_mismatch", legal("e2") + ["--lambda-semantics", "logitkd@elsewhere"],
            "lambda_semantics")
    # run() itself refuses an off-grid alpha in real mode (a direct call; no data, no student)
    common = dict(mode="real", device="cpu", pretrained=False, teacher=None, batch_size=16,
                  max_iters=80000, val_interval=4000, max_val_batches=None, num_workers=12,
                  ckpt_dir_arg=fresh("e3_s42_alpha30"), grad_clip_norm=None, log_every=50, seed=42)
    for label, kw, exc in (("alpha30", dict(alpha=30.0), ValueError),
                           ("alpha50_offgrid_flag", dict(alpha=50.0, alpha_offgrid=True), ValueError)):
        reset()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                REAL_RUN(stage=td.resolve_stage("e3"), lambda_logit=1.0, **common, **kw)
            outcome = "no refusal"
        except exc as e:
            outcome = f"refused: {e}"
        except _WouldBuildStudent:
            outcome = "student built"
        check(f"2j_run_real_refuses_{label}", outcome.startswith("refused") and COUNT["student"] == 0,
              outcome[:160])
    reset()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            REAL_RUN(stage=td.resolve_stage("g"), lambda_logit=None,
                     **{**common, "ckpt_dir_arg": None})
        outcome = "no refusal"
    except ValueError as e:
        outcome = f"refused: {e}"
    except _WouldBuildStudent:
        outcome = "student built"
    check("2b_run_real_refuses_missing_ckpt_dir", outcome.startswith("refused") and COUNT["student"] == 0,
          outcome[:160])


def test_run_entry_repeats() -> None:
    """Direct calls of the REAL run(): the schedule check in both modes and, in a real run, a missing
    or empty --ckpt-dir and any clip value refuse before set_seed, with no student built."""
    seeded: list[int] = []
    real_set_seed = td.set_seed
    td.set_seed = lambda s: seeded.append(s)
    common = dict(stage=td.resolve_stage("g"), mode="real", device="cpu", pretrained=False, teacher=None,
                  lambda_logit=None, batch_size=16, max_iters=80000, val_interval=4000,
                  max_val_batches=None, num_workers=12, ckpt_dir_arg=fresh("g_s42"),
                  grad_clip_norm=None, log_every=50, seed=42)
    cases = (("2c_run_real_above", dict(max_iters=80001), RuntimeError, "max_iters_above_horizon"),
             ("2c_run_real_not_horizon", dict(max_iters=100), RuntimeError, "max_iters_not_horizon"),
             ("2c_run_dry_above", dict(mode="dry", max_iters=80001, ckpt_dir_arg=None), RuntimeError,
              "max_iters_above_horizon"),
             ("2b_run_real_empty_ckpt_dir", dict(ckpt_dir_arg=""), ValueError, "ckpt_dir_required"),
             ("2g_run_real_clip", dict(grad_clip_norm=1.0), ValueError, "grad_clip_am7"))
    try:
        for label, over, exc, code in cases:
            reset()
            seeded.clear()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    REAL_RUN(**{**common, **over})
                outcome = "no refusal"
            except exc as e:
                outcome = f"refused: {e}"
            except _WouldBuildStudent:
                outcome = "student built"
            check(f"{label}_refused_{code}", outcome.startswith(f"refused: [{code}]") and seeded == []
                  and COUNT["student"] == 0, outcome[:160])
        # a dry run below the horizon is not refused: it seeds and reaches the (stub) student build
        reset()
        seeded.clear()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                REAL_RUN(**{**common, "mode": "dry", "max_iters": 100, "ckpt_dir_arg": None})
            outcome = "no student build"
        except _WouldBuildStudent:
            outcome = "student built"
        except Exception as e:                    # noqa: BLE001 - any other outcome is a FAIL detail
            outcome = f"{type(e).__name__}: {e}"
        check("2c_run_dry_below_horizon_proceeds", outcome == "student built" and seeded == [42],
              f"{outcome[:160]} seeded={seeded}")
    finally:
        td.set_seed = real_set_seed


def test_gate_order() -> None:
    """M11 and the CUDA-device refusals keep their text and fire before every item-2 gate."""
    reset()
    rc, err = call_main(with_value(legal("g", max_iters=100), "--device", "cpu"))
    check("order_device_refusal_before_schedule", rc == 2 and "on CPU" in err
          and "[max_iters_not_horizon]" not in err and COUNT["teacher"] == 0, err.strip()[-160:])
    reset()
    rc, err = call_main(with_value(legal("e2") + ["--lambda-semantics", "logitkd@elsewhere"],
                                   "--device", "cpu"))
    check("order_device_refusal_before_lambda_semantics", rc == 2 and "on CPU" in err
          and "[lambda_semantics]" not in err and COUNT["teacher"] == 0, err.strip()[-160:])
    saved_root = td.DATA["root"]
    m11_root = staged_root("staged_m11")
    td.DATA["root"] = str(m11_root)
    # PL-4 (SL-1): no TEST folder is created. os.path.lexists answers True for this root's first TEST
    # surface only (isolation.TEST_SURFACES[0]) and asks the real function about every other path.
    surface = os.path.join(str(m11_root), *isolation.TEST_SURFACES[0].split("/"))
    real_lexists = os.path.lexists

    def lexists(path, _real=real_lexists):
        return True if os.fspath(path) == surface else _real(path)
    os.path.lexists = lexists
    try:
        refused("order_m11_before_schedule", legal("g", max_iters=100), "test_split_present")
        refused("order_m11_before_clip", legal("g", grad_clip_norm=1.0), "test_split_present")
    finally:
        os.path.lexists = real_lexists
        td.DATA["root"] = saved_root


# ------------------------------------------------------------------ lane 4(a), PL-28 (K2)
LAUNCH_KEYS = ["arm", "descriptive", "parent_of_e4_e7", "selection_sha256", "selection_files", "records_commit"]
NULL_SELECTIONS = {"lambda_logit": None, "alpha_cwd": None}
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class _StopAtLoop(Exception):
    pass


class _TinyStudent(torch.nn.Module):
    used_pretrained = False

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


class _Loader:
    """Three batches over six samples: run() reads only len() and the dataset's len() before its loop."""
    dataset = [0] * 6

    def __len__(self):
        return 3


def run_meta_of_direct_run(stage: str, **kw):
    """The REAL run() in dry mode up to its training loop: a tiny student, length-only loaders, a fixed
    git_head and a no-op set_seed (no global determinism state changes), and the loop's first call
    (cycle) stops it. Returns its one run_meta row, or the reason it has none."""
    ckpt = Path(fresh(f"{stage}_s42_meta"))
    saved = (td.build_student, td.build_dataloader, td.cycle, td._git_provenance, td.set_seed)

    def stop(*_a, **_k):
        raise _StopAtLoop()
    td.build_student, td.build_dataloader = (lambda *a, **k: _TinyStudent()), (lambda *a, **k: _Loader())
    td.cycle, td._git_provenance, td.set_seed = stop, (lambda: ("0" * 40, "smoke_stub")), (lambda s: None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            REAL_RUN(stage=td.resolve_stage(stage), mode="dry", device="cpu", pretrained=False,
                     teacher=FrozenTeacher(MockTeacher(NC)),
                     lambda_logit=1.0 if stage in ("e2", "e3") else None, batch_size=2, max_iters=2,
                     val_interval=2, max_val_batches=1, num_workers=0, ckpt_dir_arg=str(ckpt),
                     grad_clip_norm=None, log_every=1, seed=42, **kw)
        return "run() did not reach its training loop"
    except _StopAtLoop:
        pass
    except Exception as e:  # noqa: BLE001 - the reason is the check's detail
        return f"{type(e).__name__}: {e}"
    finally:
        td.build_student, td.build_dataloader, td.cycle, td._git_provenance, td.set_seed = saved
    lines = (ckpt / f"{stage}_run_meta.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]) if len(lines) == 1 else f"{len(lines)} run_meta lines"


def test_launch_fields() -> None:
    """Lane 4(a), PL-28: the selection files and the records commit, in both modes, through to run_meta."""
    sel = TMP / "records"                         # the records folder's copies (PL-28), outside the repository
    sel.mkdir()
    lam, alp = sel / "lambda_selection.json", sel / "alpha_selection.json"
    lam.write_text('{"format": "lambda_selection/1"}\n', encoding="utf-8")
    alp.write_text('{"format": "alpha_selection/1"}\n', encoding="utf-8")
    want = {"lambda_logit": {"path": str(lam.resolve()), "sha256": hashlib.sha256(lam.read_bytes()).hexdigest()},
            "alpha_cwd": {"path": str(alp.resolve()), "sha256": hashlib.sha256(alp.read_bytes()).hexdigest()}}
    for stage in ("a", "f", "g"):
        refused(f"l4_lambda_selection_on_{stage}", legal(stage, lambda_selection=lam), "selection_term")
    for stage in ("e2", "g"):
        refused(f"l4_alpha_selection_on_{stage}", legal(stage, alpha_selection=alp), "selection_term")
    refused("l4_dry_lambda_selection_on_g", ["--stage", "g", "--dry-run", "--lambda-selection", str(lam)],
            "selection_term")
    never = sel / "never_created.json"
    phantom = sel / "test" / "lambda_selection.json"          # SL-1: named, never created (C0's method)
    refused("l4_missing_selection_file", legal("e2", seed=43, lambda_selection=never), "selection_file_missing")
    refused("l4_never_created_path_with_a_test_component", legal("e2", seed=43, lambda_selection=phantom),
            "selection_file_missing")
    refused("l4_directory_as_selection_file", legal("e3", seed=43, alpha_selection=sel), "selection_file_missing")
    refused("l4_dry_missing_selection_file", ["--stage", "e3", "--dry-run", "--alpha-selection", str(never)],
            "selection_file_missing")
    check("l4_refused_paths_were_not_created", not os.path.lexists(never) and not os.path.lexists(phantom.parent))
    # C2 workflow c2trainer-1 / c2smokes-3: a file that exists and cannot be read is [selection_file_missing], in
    # both modes, before any teacher or loader (root ignores chmod 000: the hash of that one file is made to fail)
    unreadable = sel / "unreadable_selection.json"
    unreadable.write_text('{"format": "lambda_selection/1"}\n', encoding="utf-8")
    saved_sha = td.sha256_file

    def sha_or_denied(path):
        if Path(path).resolve() == unreadable.resolve():
            raise PermissionError(13, "Permission denied", str(path))
        return saved_sha(path)
    td.sha256_file = sha_or_denied
    try:
        for label, argv in (("l4_unreadable_selection_file", legal("e2", seed=43, lambda_selection=unreadable)),
                            ("l4_dry_unreadable_selection_file",
                             ["--stage", "e2", "--dry-run", "--lambda-selection", str(unreadable)])):
            reset()
            try:
                rc, err = call_main(argv)
            except Exception as e:  # noqa: BLE001 - a raise instead of a refusal is a FAIL of this check
                rc, err = None, f"raised {type(e).__name__}: {e}"
            check(f"{label}_refused_selection_file_missing", rc == 2 and "[selection_file_missing]" in err
                  and "cannot be read" in err and COUNT["teacher"] == 0 and COUNT["loader"] == 0 and COUNT["run"] == 0,
                  f"rc={rc} teacher={COUNT['teacher']} run={COUNT['run']} err={err.strip()[-160:]}")
    finally:
        td.sha256_file = saved_sha
    lam_only = {"lambda_logit": want["lambda_logit"], "alpha_cwd": None}
    cases = [("l4_legal_e2_s43_lambda_selection_reaches_run", legal("e2", seed=43, lambda_selection=lam), lam_only),
             ("l4_legal_e3_s43_both_selections_reach_run",
              legal("e3", seed=43, lambda_selection=lam, alpha_selection=alp), want),
             ("l4_dry_run_accepts_both_selections",
              ["--stage", "e3", "--dry-run", "--lambda-selection", str(lam), "--alpha-selection", str(alp)], want),
             ("l4_no_selection_flag_reaches_run_as_nulls", legal("e3", seed=43), NULL_SELECTIONS),
             # C2 workflow c2trainer-4: the arms A and F take --alpha-selection
             ("l4_legal_a_s42_alpha_selection_reaches_run", legal("a", alpha_selection=alp),
              {"lambda_logit": None, "alpha_cwd": want["alpha_cwd"]}),
             ("l4_legal_f_s42_alpha_selection_reaches_run", legal("f", alpha_selection=alp),
              {"lambda_logit": None, "alpha_cwd": want["alpha_cwd"]})]
    # c2trainer-4: a relative path and a path through a symlink are both recorded resolved (a named SKIP where the
    # host gives no relative path or no symlink, as in Q10)
    try:
        rel_lam = os.path.relpath(lam)
    except ValueError as e:                       # Windows: the working directory is on another drive
        skip("l4_relative_selection_path_is_recorded_resolved", f"no relative path from here: {e}")
    else:
        cases.append(("l4_relative_selection_path_is_recorded_resolved",
                      legal("e2", seed=43, lambda_selection=rel_lam), lam_only))
    link = TMP / "records_link"
    try:
        link.symlink_to(sel, target_is_directory=True)
    except (OSError, NotImplementedError) as e:   # Windows without the symlink privilege
        skip("l4_selection_path_through_a_symlink_is_recorded_resolved", f"os.symlink not permitted on this host: {e}")
    else:
        cases.append(("l4_selection_path_through_a_symlink_is_recorded_resolved",
                      legal("e2", seed=43, lambda_selection=link / lam.name), lam_only))
    for label, argv, selections in cases:
        reset()
        rc, err = call_main(argv)
        got = SEEN[-1] if SEEN else {}
        check(label, rc == 0 and got.get("selections") == selections and got.get("records_commit") is None,
              f"rc={rc} selections={got.get('selections')} {err.strip()[-120:]}")
    reset()
    rc, err = call_main(legal("e2", records_commit=COMMIT))
    check("l4_records_commit_reaches_run", rc == 0 and bool(SEEN) and SEEN[-1].get("records_commit") == COMMIT,
          f"rc={rc} {err.strip()[-120:]}")
    for label, bad in (("39_chars", COMMIT[:39]), ("41_chars", COMMIT + "0"), ("uppercase", COMMIT.upper()),
                       ("not_hex", "g" * 40), ("short_id", COMMIT[:7]), ("empty", "")):
        refused(f"l4_records_commit_{label}", legal("e2", records_commit=bad), "records_commit_format")
    refused("l4_dry_records_commit_uppercase", ["--stage", "e2", "--dry-run", "--records-commit", COMMIT.upper()],
            "records_commit_format")
    # launch_fields(): per stage and horizon, and its refusals for direct calls
    for stage, horizon, arm, descriptive, parent in (
            ("e2", 80000, None, False, False), ("e3", 80000, None, False, True), ("e2", 160000, None, True, False),
            ("e3", 160000, None, True, False), ("a", 80000, "A", True, False), ("f", 80000, "F", True, False),
            ("g", 80000, "G", True, False)):
        f = td.launch_fields(td.resolve_stage(stage), horizon=horizon)
        check(f"l4_launch_fields_{stage}_{horizon // 1000}k", list(f) == LAUNCH_KEYS
              and (f["arm"], f["descriptive"], f["parent_of_e4_e7"]) == (arm, descriptive, parent)
              and f["selection_sha256"] == f["selection_files"] == NULL_SELECTIONS and f["records_commit"] is None,
              json.dumps(f))
    for label, stage, kw, code in (
            ("alpha_selection_on_e2", "e2", {"selections": {**NULL_SELECTIONS, "alpha_cwd": want["alpha_cwd"]}},
             "selection_term"),
            ("lambda_selection_on_g", "g", {"selections": {**NULL_SELECTIONS, "lambda_logit": want["lambda_logit"]}},
             "selection_term"),
            ("a_missing_key", "e3", {"selections": {"lambda_logit": None}}, "selection_term"),
            ("records_commit_short_id", "e2", {"records_commit": COMMIT[:7]}, "records_commit_format")):
        try:
            td.launch_fields(td.resolve_stage(stage), horizon=80000, **kw)
            outcome = "no refusal"
        except ValueError as e:
            outcome = str(e)
        check(f"l4_launch_fields_refuses_{label}", outcome.startswith(f"[{code}]"), outcome[:160])
    # the REAL run(): a direct call refuses before any student is built; a dry run records the fields
    saved_seed = td.set_seed
    td.set_seed = lambda s: None
    reset()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            REAL_RUN(stage=td.resolve_stage("e2"), mode="dry", device="cpu", pretrained=False, teacher=None,
                     lambda_logit=1.0, batch_size=2, max_iters=2, val_interval=2, max_val_batches=1,
                     num_workers=0, ckpt_dir_arg=fresh("e2_s42_direct"), grad_clip_norm=None, log_every=1,
                     seed=42, selections={**NULL_SELECTIONS, "alpha_cwd": want["alpha_cwd"]})
        outcome = "no refusal"
    except ValueError as e:
        outcome = f"refused: {e}"
    except _WouldBuildStudent:
        outcome = "student built"
    finally:
        td.set_seed = saved_seed
    check("l4_run_direct_call_refuses_selection_term", outcome.startswith("refused: [selection_term]")
          and COUNT["student"] == 0, outcome[:160])
    meta = run_meta_of_direct_run("e3", selections=want, records_commit=COMMIT)
    tail = dict(list(meta.items())[-6:]) if isinstance(meta, dict) else meta
    check("l4_run_meta_e3_launch_fields_recorded_last", isinstance(meta, dict)
          and list(meta)[-7:] == ["teacher_mock"] + LAUNCH_KEYS
          and tail == {"arm": None, "descriptive": False, "parent_of_e4_e7": True,
                       "selection_sha256": {k: v["sha256"] for k, v in want.items()},
                       "selection_files": {k: v["path"] for k, v in want.items()}, "records_commit": COMMIT},
          json.dumps(tail)[:400])
    meta = run_meta_of_direct_run("g")
    tail = dict(list(meta.items())[-6:]) if isinstance(meta, dict) else meta
    check("l4_run_meta_g_launch_fields_with_no_flag", isinstance(meta, dict)
          and list(meta)[-7:] == ["teacher_mock"] + LAUNCH_KEYS
          and tail == {"arm": "G", "descriptive": True, "parent_of_e4_e7": False,
                       "selection_sha256": NULL_SELECTIONS, "selection_files": NULL_SELECTIONS,
                       "records_commit": None}, json.dumps(tail)[:400])


REAL_RUN = td.run
REAL_LOAD = td.load_frozen_teacher


def main() -> int:
    print("=" * 78)
    print("DISTILL REAL-RUN GATES SMOKE (L-KD-HARDEN item 2) - CPU, synthetic staged root, stubs")
    print(f"torch {torch.__version__}")
    print("=" * 78)
    saved = (td.DATA["root"], dict(isolation.DEFAULT_EXPECTED_COUNTS), td.load_frozen_teacher,
             td.build_dataloader, td.run, td.build_student)
    td.DATA["root"] = str(staged_root())
    isolation.DEFAULT_EXPECTED_COUNTS.update({"train": 2, "val": 2})
    td.load_frozen_teacher, td.build_dataloader = stub_load_frozen_teacher, stub_build_dataloader
    td.run, td.build_student = stub_run, stub_build_student
    try:
        for fn in (test_legal_launches, test_lambda, test_ckpt_dir, test_ckpt_dir_in_repo_and_log_every,
                   test_abort_exit, test_schedule, test_val, test_workers, test_seed_batch_init, test_clip,
                   test_tf32, test_cuda_order, test_alpha_and_semantics, test_run_entry_repeats,
                   test_gate_order, test_teacher_sha256, test_launch_fields):
            fn()
    finally:
        td.DATA["root"] = saved[0]
        isolation.DEFAULT_EXPECTED_COUNTS.clear()
        isolation.DEFAULT_EXPECTED_COUNTS.update(saved[1])
        td.load_frozen_teacher, td.build_dataloader, td.run, td.build_student = saved[2:]
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:58}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    for name, reason in SKIPPED:
        print(f"  {name:58}: SKIP  {reason}")
    passed = sum(1 for _, ok, _ in results if ok)
    skipped = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)}{skipped})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
