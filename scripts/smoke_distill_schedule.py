#!/usr/bin/env python3
"""Smoke for lane 3 (K2; the KD part of L-AM16-ITERS, AM-16 item 3 / DL-27): train_distill --iterations.

Data-free and CPU-only: PlantSegDataset is replaced by a synthetic in-memory dataset and the teacher is an
explicit MockTeacher, so no data root, teacher checkpoint or ImageNet weight is read. Everything is written
under one scratch directory named k2_schedule_<pid>_<n> (no random suffix: RULING 2, SL-1), removed at the
end.

Sections:
  G  the pure schedule gate: the registered horizons (train_e1.REGISTERED_POLY_HORIZONS) admitted, an
     unregistered one refused ([poly_horizon_unregistered]), the refusals at 160,000, and the default texts
     byte-identical to the pre-K2 ones (MEASURED at 647d305)
  K  main() -> run() kwargs through a recorder (run(), the teacher load and the M11 root check stubbed):
     --iterations 120000 is an argparse exit 2; E2 and E3 at seed 42 with --iterations 160000, alone or with
     --max-iters 160000, give max_iters == poly_horizon == 160000 and change nothing else; any other stage
     or seed at 160,000 is refused ([iterations]); --max-iters 160000 alone and 160,000 with --max-iters
     80000 are refused; every refusal comes before the teacher load; a dry 160k run has max_iters 4; the
     default real launch equals the pre-K2 golden (MEASURED at 647d305) plus the K2 kwargs at their
     defaults, and --iterations 80000 equals it (PL-23); an explicit --iterations 80000 is accepted with
     max_iters and poly_horizon 80000 for A at seed 42 and E2 at seed 43 (K11); run() is keyword-only and
     its K2 parameters default to None
  R  dry runs through the real run(): the scheduler built over the horizon (160000 or 80000, power 0.9),
     run_meta's max_iters, poly_horizon and descriptive, the telemetry LRs equal to the curve's prefix; an
     injected total_iters mismatch -> [poly_horizon_mismatch]; run(poly_horizon=120000) ->
     [poly_horizon_unregistered] before set_seed; the ramp identical in both profiles
  L  LR curves replayed with the values run() used: the distill curve equals E1's at 80,000 and 160,000 and
     GOLDEN_LR160_DISTILL equals E1's 160k golden (same process, every stack); the MEASURED goldens only on
     the pinned stack (linux + torch 2.1.0+cu121 + CPython 3.11; elsewhere SKIP, a FAIL with
     --require-goldens); the 160k curve has 160,000 values, one 0.0 (the last) and never increases

Run:  python -B scripts/smoke_distill_schedule.py [--require-goldens]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse      # noqa: E402
import contextlib    # noqa: E402
import hashlib       # noqa: E402
import inspect       # noqa: E402
import io            # noqa: E402
import json          # noqa: E402
import os            # noqa: E402
import re            # noqa: E402
import shutil        # noqa: E402
import struct        # noqa: E402
import tempfile      # noqa: E402
import traceback     # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def scratch_dir(prefix: str) -> Path:
    """A new directory under the system temp dir, named <prefix>_<pid>_<n> as in the selection smoke (RULING
    2): no random suffix can spell 'test' (SL-1), and it lies outside the repository."""
    n = 0
    while True:
        n += 1
        p = Path(tempfile.gettempdir()).resolve() / f"{prefix}_{os.getpid()}_{n}"
        if "test" in str(p).lower():
            raise SystemExit(f"refusing to create {p}: its path contains 'test' (SL-1); set TMPDIR elsewhere")
        if p == REPO or REPO in p.parents:
            raise SystemExit(f"refusing to create {p} inside the repository")
        try:
            p.mkdir()
        except FileExistsError:
            continue
        return p


TMP = scratch_dir("k2_schedule")
os.environ["PLANTSEG_DATA_ROOT"] = str(TMP / "no_such_data_root")     # nothing may reach a real root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                         # noqa: E402

import src.data.dataset as ds_mod                    # noqa: E402
import src.training.train_distill as td              # noqa: E402
import src.training.train_e1 as T                    # noqa: E402
from configs.e1_student import E1_STUDENT            # noqa: E402
from src.distill import FrozenTeacher, MockTeacher   # noqa: E402

CHECKS: list[tuple[str, str, str]] = []
PINNED = (sys.platform == "linux" and torch.__version__ == "2.1.0+cu121"
          and sys.version_info[:2] == (3, 11))


def e1_golden(name: str) -> str | None:
    """smoke_e1_schedule's literal `name`, read as text from that file (importing it would run its module-level
    mkdtemp, whose random suffix is an SL-1 risk), so L02 and L03 compare with E1's goldens themselves (C3
    workflow c3-2)."""
    m = re.search(rf'^{name} = "([0-9a-f]{{64}})"', (REPO / "scripts" / "smoke_e1_schedule.py")
                  .read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


GOLDEN_LR80 = e1_golden("GOLDEN_LR80")
GOLDEN_LR160 = e1_golden("GOLDEN_LR160")
# train_distill builds E1's scheduler (train_e1.build_scheduler) with E1_STUDENT's lr and power, so its
# 160k curve is E1's: L03 checks the equality on every stack, L04 the literal on the pinned one.
GOLDEN_LR160_DISTILL = "dbe6d363704692590d22f0c5219d8fcaed5c94521b8ab55e89b3220d101bc54c"
# The schedule gate's texts at 647d305 (pre-K2), MEASURED: the default horizon keeps them byte for byte.
PRE_K2_NOT_HORIZON = ("[max_iters_not_horizon] a real run trains its whole schedule: --max-iters 100 != the "
                      "poly horizon 80000; do not pass --max-iters to a real run")
PRE_K2_ABOVE_HORIZON = ("[max_iters_above_horizon] --max-iters 80001 exceeds the poly horizon 80000: "
                        "PolynomialLR holds lr at 0.0 after it, so iterations 80001..80001 would not train")
# K09: main()'s run() kwargs for the default real E2 / E3 seed-42 launch at 647d305 (pre-K2), MEASURED with
# run(), the teacher load and the M11 root check stubbed; the teacher object and ckpt_dir_arg left out.
_PRE_K2_COMMON = {"mode": "real", "device": "cuda", "pretrained": "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2",
                  "lambda_logit": 1.0, "batch_size": 16, "max_iters": 80000, "val_interval": 4000,
                  "max_val_batches": None, "num_workers": 12, "grad_clip_norm": None, "log_every": 1,
                  "seed": 42, "semantics_declared": None, "semantics_override": False, "alpha_offgrid": False}
PRE_K2_KWARGS = {
    "e2": {**_PRE_K2_COMMON, "alpha": None,
           "stage": {"key": "e2", "name": "E2", "logit_kd": True, "cwd_feat": False, "cwd_logit": False,
                     "objective": "L_CE + L_Dice + lambda_logit*L_LogitKD"}},
    "e3": {**_PRE_K2_COMMON, "alpha": 50.0,
           "stage": {"key": "e3", "name": "E3", "logit_kd": True, "cwd_feat": True, "cwd_logit": True,
                     "objective": "L_CE + L_Dice + lambda_logit*L_LogitKD + alpha_cwd*L_CWD_feat + "
                                  "3*L_CWD_logit"}}}
K2_KWARG_DEFAULTS = {"selections": {"lambda_logit": None, "alpha_cwd": None}, "records_commit": None,
                     "poly_horizon": 80000}
REQUIRE_GOLDENS = False


def check(name, ok, detail=""):
    CHECKS.append((name, "PASS" if ok else "FAIL", " | ".join(str(detail).split("\n"))[:240]))


def golden(name, fn):
    """A check that needs the pinned stack's float behaviour; SKIP elsewhere (FAIL with --require-goldens)."""
    if PINNED:
        ok, detail = fn()
        check(name, ok, detail)
    elif REQUIRE_GOLDENS:
        check(name, False, f"--require-goldens: not the pinned stack ({sys.platform}, torch {torch.__version__})")
    else:
        CHECKS.append((name, "SKIP", f"pinned-stack golden; here {sys.platform} torch {torch.__version__}"))


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def lr_seq(horizon, n=None, power=None, lr=None, momentum=None, wd=None):
    """smoke_e1_schedule's probe: SGD on 4 zero params, train_e1.build_scheduler, step, record the post-step
    lr (the value a train row logs). Hyper-parameters default to E1_STUDENT's."""
    n = horizon if n is None else n
    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([p], lr=E1_STUDENT["learning_rate"] if lr is None else lr,
                          momentum=E1_STUDENT["momentum"] if momentum is None else momentum,
                          weight_decay=E1_STUDENT["weight_decay"] if wd is None else wd)
    sch, _ = T.build_scheduler(opt, horizon, E1_STUDENT["lr_power"] if power is None else power)
    out = []
    for _ in range(n):
        p.grad = torch.zeros_like(p)
        opt.step()
        sch.step()
        out.append(opt.param_groups[0]["lr"])
    return out, sch


def sha(seq):
    return hashlib.sha256(struct.pack("<%dd" % len(seq), *seq)).hexdigest()


class SynthDS(torch.utils.data.Dataset):
    """Stands in for PlantSegDataset (smoke_e1_schedule's): train 4 / val 2 samples, 512x512, labels 0..115
    with 255 rows."""

    def __init__(self, split):
        self.split = split
        self.n = {"train": 4, "val": 2}[split]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(1000 * (self.split == "val") + i)
        img = torch.rand(3, 512, 512, generator=g)
        mask = torch.randint(0, 116, (512, 512), generator=g)
        mask[:16] = 255
        return img, mask


TEACHER = TMP / "teacher.pth"
TEACHER.write_bytes(b"read by no one: the teacher load is stubbed")
TEACHER_SHA256 = hashlib.sha256(TEACHER.read_bytes()).hexdigest()
_n = [0]


def fresh(name: str) -> Path:
    """A --ckpt-dir path that does not exist yet."""
    _n[0] += 1
    return TMP / "ckpt" / f"{_n[0]:03d}" / name


def launch(stage: str, *extra: str, seed: int = 42) -> list[str]:
    """The legal real launch of `stage` (smoke_distill_realrun_gates' shape): --device cuda passes the device
    gate on this CPU box; lambda 1 for E2 and E3, alpha 50 for E3 and A, a fresh --ckpt-dir."""
    argv = ["--stage", stage, "--real-run", "--confirm-real-run", "--device", "cuda", "--teacher-ckpt",
            str(TEACHER), "--teacher-ckpt-sha256", TEACHER_SHA256, "--num-workers", "12", "--seed", str(seed)]
    if stage in ("e2", "e3"):
        argv += ["--lambda-logit", "1"]
    if stage in ("e3", "a"):
        argv += ["--alpha", "50"]
    tok = "_alpha50" if stage in ("e3", "a", "f") else ""
    return argv + ["--ckpt-dir", str(fresh(f"{stage}_s{seed}{tok}")), *extra]


TEACHER_LOADS: list = []                             # the stubbed teacher loads of the last main_rec call


def main_rec(argv):
    """train_distill.main with run() replaced by a recorder, the teacher load (counted in TEACHER_LOADS) and
    the M11 root check stubbed."""
    calls = []
    TEACHER_LOADS.clear()

    def rec(**kw):
        calls.append(kw)
        return 0

    def load(*a, **k):
        TEACHER_LOADS.append(a)
        return FrozenTeacher(MockTeacher(116))
    err = io.StringIO()
    with patched(td, "run", rec), patched(td, "load_frozen_teacher", load), \
            patched(td, "assert_trainval_only_root", lambda *a, **k: {"counts": {"train": 5367, "val": 846}}), \
            contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        try:
            rc = td.main(argv)
        except SystemExit as e:                       # argparse errors
            rc = e.code
    return rc, calls, err.getvalue()


def kwargs_of(calls) -> dict:
    """The last run() call's kwargs without the teacher object and the per-call ckpt_dir_arg."""
    return {k: v for k, v in calls[-1].items() if k not in ("teacher", "ckpt_dir_arg")} if calls else {}


def refused(label, argv, code):
    """Refused with `code`, before run() and before the teacher load (the module docstring's gate order; C3
    workflow c3-1)."""
    rc, calls, err = main_rec(argv)
    check(f"{label}_refused_{code}", rc == 2 and calls == [] and TEACHER_LOADS == [] and f"[{code}]" in err,
          f"teacher_loads={len(TEACHER_LOADS)} {err.strip()[-160:]}")


def dry_run(stage, ckpt: Path, spy: list, *, poly_horizon=None, scheduler_builder=None):
    """The REAL run() in dry mode on the synthetic set: batch 2, 2 iterations, VAL at iteration 2 over one
    batch, CPU, an explicit MockTeacher, no ImageNet download. build_scheduler is spied (horizon, power and
    the optimizer's lr, momentum and weight decay), or replaced by `scheduler_builder`."""
    real_bs = td.build_scheduler

    def bs_spy(opt, horizon, power):
        g = opt.param_groups[0]
        spy.append((horizon, power, g["lr"], g["momentum"], g["weight_decay"]))
        return (scheduler_builder or real_bs)(opt, horizon, power)
    kw = dict(stage=td.resolve_stage(stage), mode="dry", device="cpu", pretrained=False,
              teacher=FrozenTeacher(MockTeacher(116)), lambda_logit=1.0 if stage in ("e2", "e3") else None,
              batch_size=2, max_iters=2, val_interval=2, max_val_batches=1, num_workers=0,
              ckpt_dir_arg=str(ckpt), grad_clip_norm=None, log_every=1, seed=42)
    if poly_horizon is not None:
        kw["poly_horizon"] = poly_horizon
    out = io.StringIO()
    with patched(ds_mod, "PlantSegDataset", SynthDS), patched(td, "build_scheduler", bs_spy), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = td.run(**kw)
    return rc, out.getvalue()


def jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ------------------------------------------------------------------------------------------ G
def section_g():
    admitted = (("dry", 4, None), ("real", 80000, None), ("dry", 4, 160000), ("real", 160000, 160000))
    bad = [a for a in admitted if td.schedule_gate_error(a[0], a[1], horizon=a[2]) is not None]
    check("G01_gate_admits_dry_and_real_at_both_registered_horizons",
          not bad and T.REGISTERED_POLY_HORIZONS == (80000, 160000), str(bad))
    refusals = {("dry", 4, 120000): "[poly_horizon_unregistered]", ("real", 80000, 160000): "[max_iters_not_horizon]",
                ("real", 160000, 80000): "[max_iters_above_horizon]",
                ("dry", 160001, 160000): "[max_iters_above_horizon]"}
    got = {a: td.schedule_gate_error(a[0], a[1], horizon=a[2]) for a in refusals}
    check("G02_gate_refuses_an_unregistered_horizon_and_the_160k_misfits",
          all(isinstance(g, str) and g.startswith(refusals[a]) for a, g in got.items()), str(got))
    check("G03_default_texts_byte_identical_to_pre_k2",
          td.schedule_gate_error("real", 100) == PRE_K2_NOT_HORIZON
          and td.schedule_gate_error("dry", 80001) == PRE_K2_ABOVE_HORIZON
          and td.schedule_gate_error("real", 80000) is None and td.schedule_gate_error("dry", 4) is None)


# ------------------------------------------------------------------------------------------ K
def section_k():
    rc, calls, err = main_rec(launch("e2", "--iterations", "120000"))
    check("K01_iterations_120000_is_an_argparse_exit_2", rc == 2 and calls == [] and "invalid choice" in err,
          f"rc={rc} {err.strip()[-120:]}")
    base = {}
    for stage in ("e2", "e3"):
        rc, calls, err = main_rec(launch(stage))
        base[stage] = kw = kwargs_of(calls)
        check(f"K09_default_real_{stage}_launch_equals_the_pre_k2_golden",
              rc == 0 and {k: v for k, v in kw.items() if k not in K2_KWARG_DEFAULTS} == PRE_K2_KWARGS[stage]
              and {k: kw.get(k) for k in K2_KWARG_DEFAULTS} == K2_KWARG_DEFAULTS,
              f"rc={rc} {json.dumps(kw, default=str)[:160]} {err.strip()[-80:]}")
        rc, calls, err = main_rec(launch(stage, "--iterations", "80000"))
        check(f"K09_{stage}_explicit_iterations_80000_equals_the_default_launch",
              rc == 0 and kwargs_of(calls) == kw, f"rc={rc} {err.strip()[-120:]}")
        for label, extra in (("", ()), ("_with_max_iters_160000", ("--max-iters", "160000"))):
            rc, calls, err = main_rec(launch(stage, "--iterations", "160000", *extra))
            check(f"K02_{stage}_s42_160k{label}_gives_max_iters_and_poly_horizon_160000",
                  rc == 0 and kwargs_of(calls) == {**kw, "max_iters": 160000, "poly_horizon": 160000},
                  f"rc={rc} {err.strip()[-120:]}")
    for stage, seed in (("e2", 43), ("e3", 43), ("e2", 44)):
        refused(f"K04_{stage}_s{seed}_at_160k", launch(stage, "--iterations", "160000", seed=seed), "iterations")
    for stage in ("a", "f", "g"):
        refused(f"K05_{stage}_at_160k", launch(stage, "--iterations", "160000"), "iterations")
    # C3 workflow c3-4: an explicit --iterations 80000 is E1's horizon, accepted for every stage and seed
    for stage, seed in (("a", 42), ("e2", 43)):
        rc, calls, err = main_rec(launch(stage, "--iterations", "80000", seed=seed))
        kw = kwargs_of(calls)
        check(f"K11_{stage}_s{seed}_explicit_iterations_80000_is_accepted",
              rc == 0 and kw.get("max_iters") == 80000 and kw.get("poly_horizon") == 80000,
              f"rc={rc} {err.strip()[-120:]}")
    refused("K06_max_iters_160000_alone", launch("e2", "--max-iters", "160000"), "max_iters_above_horizon")
    refused("K07_160k_with_max_iters_80000", launch("e2", "--iterations", "160000", "--max-iters", "80000"),
            "max_iters_not_horizon")
    rc, calls, err = main_rec(["--stage", "e2", "--dry-run", "--iterations", "160000"])
    kw = kwargs_of(calls)
    check("K08_dry_160k_gives_max_iters_4_and_poly_horizon_160000",
          rc == 0 and kw.get("max_iters") == 4 and kw.get("poly_horizon") == 160000 and kw.get("mode") == "dry",
          f"rc={rc} {err.strip()[-120:]}")
    params = inspect.signature(td.run).parameters
    check("K10_run_is_keyword_only_and_its_k2_parameters_default_to_none",
          all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
          and all(k in params and params[k].default is None
                  for k in ("selections", "records_commit", "poly_horizon")), str(list(params)))


# ------------------------------------------------------------------------------------------ R
def guarded_dry_run(*a, **kw):
    """dry_run(), with a raise reported as the calling check's failure (rc None, the error as output)."""
    try:
        return dry_run(*a, **kw)
    except Exception as e:                            # noqa: BLE001 -- the check's FAIL detail
        return None, f"{type(e).__name__}: {e}"


def section_r(state):
    spy160, ck160 = [], fresh("e2_s42_160k")
    rc, out = guarded_dry_run("e2", ck160, spy160, poly_horizon=160000)
    state["hp160"] = spy160[0] if spy160 else None
    meta = (jsonl(ck160 / "e2_run_meta.jsonl") or [{}])[0] if (ck160 / "e2_run_meta.jsonl").exists() else {}
    tel = jsonl(ck160 / "e2_telemetry.jsonl") if (ck160 / "e2_telemetry.jsonl").exists() else []
    tr160 = [r for r in tel if r.get("event") == "train"]
    head160, _ = lr_seq(160000, 2)
    check("R01_dry_run_at_160k_builds_the_160k_scheduler_and_records_it",
          rc == 0 and [s[:2] for s in spy160] == [(160000, 0.9)] and meta.get("poly_horizon") == 160000
          and meta.get("max_iters") == 2 and meta.get("descriptive") is True
          and meta.get("parent_of_e4_e7") is False and [r["lr"] for r in tr160] == head160,
          f"rc={rc} spy={spy160} meta={ {k: meta.get(k) for k in ('poly_horizon', 'max_iters', 'descriptive')} } "
          f"{out.strip()[-120:]}")
    spy80, ck80 = [], fresh("e2_s42_80k")
    rc, out = guarded_dry_run("e2", ck80, spy80)
    state["hp80"] = spy80[0] if spy80 else None
    meta80 = (jsonl(ck80 / "e2_run_meta.jsonl") or [{}])[0] if (ck80 / "e2_run_meta.jsonl").exists() else {}
    tel80 = jsonl(ck80 / "e2_telemetry.jsonl") if (ck80 / "e2_telemetry.jsonl").exists() else []
    tr80 = [r for r in tel80 if r.get("event") == "train"]
    head80, _ = lr_seq(80000, 2)
    check("R02_dry_run_by_default_builds_the_80k_scheduler",
          rc == 0 and [s[:2] for s in spy80] == [(80000, 0.9)] and meta80.get("poly_horizon") == 80000
          and meta80.get("descriptive") is False and [r["lr"] for r in tr80] == head80,
          f"rc={rc} spy={spy80} {out.strip()[-120:]}")

    def builds_80k(opt, horizon, power):                # an injected mismatch: always E1's 80k horizon
        return T.build_scheduler(opt, 80000, power)
    try:
        dry_run("e2", fresh("e2_s42_mismatch"), [], poly_horizon=160000, scheduler_builder=builds_80k)
        outcome = "no refusal"
    except RuntimeError as e:
        outcome = str(e)
    check("R03_injected_total_iters_mismatch_refused_poly_horizon_mismatch",
          outcome.startswith("[poly_horizon_mismatch]"), outcome[:160])
    seeded = []
    try:
        with patched(td, "set_seed", lambda s: seeded.append(s)):
            dry_run("e2", fresh("e2_s42_unregistered"), [], poly_horizon=120000)
        outcome = "no refusal"
    except RuntimeError as e:
        outcome = str(e)
    check("R04_run_poly_horizon_120000_refused_before_set_seed",
          outcome.startswith("[poly_horizon_unregistered]") and seeded == [], f"{outcome[:160]} seeded={seeded}")
    check("R05_ramp_identical_in_both_profiles",
          bool(tr80) and meta.get("ramp_iters") == meta80.get("ramp_iters") == 2
          and [r["ramp"] for r in tr160] == [r["ramp"] for r in tr80],
          f"{meta.get('ramp_iters')} {meta80.get('ramp_iters')} {[r.get('ramp') for r in tr160]}")


# ------------------------------------------------------------------------------------------ L
def section_l(state):
    e1_hp = (E1_STUDENT["lr_power"], E1_STUDENT["learning_rate"], E1_STUDENT["momentum"],
             E1_STUDENT["weight_decay"])
    curve80, _ = lr_seq(E1_STUDENT["iterations"])
    hp80 = state.get("hp80")
    via80 = lr_seq(hp80[0], 80000, *hp80[1:])[0] if hp80 else []
    check("L01_distill_80k_curve_equals_e1s",
          bool(hp80) and hp80[0] == 80000 and tuple(hp80[1:]) == e1_hp and via80 == curve80
          and len(curve80) == 80000, str(hp80))
    golden("L02_distill_80k_curve_is_golden_lr80", lambda: (sha(via80) == GOLDEN_LR80, sha(via80)[:16]))
    curve160, _ = lr_seq(T.LONGER_SCHEDULE_ITERS)
    hp160 = state.get("hp160")
    via160 = lr_seq(hp160[0], 160000, *hp160[1:])[0] if hp160 else []
    check("L03_distill_160k_curve_equals_e1s_and_its_golden_is_e1s",
          bool(hp160) and hp160[0] == 160000 and tuple(hp160[1:]) == e1_hp and via160 == curve160
          and GOLDEN_LR160_DISTILL == GOLDEN_LR160, str(hp160))
    golden("L04_distill_160k_curve_is_golden_lr160_distill",
           lambda: (sha(via160) == GOLDEN_LR160_DISTILL, sha(via160)[:16]))
    check("L05_160k_curve_has_160000_values_one_zero_at_the_end_never_increasing",
          len(via160) == 160000 and via160.count(0.0) == 1 and via160[-1] == 0.0
          and all(a >= b for a, b in zip(via160, via160[1:])))


def main() -> int:
    global REQUIRE_GOLDENS
    state = {}
    try:                                              # TMP is removed on every exit, argparse's included (c3-5)
        ap = argparse.ArgumentParser()
        ap.add_argument("--require-goldens", action="store_true")
        REQUIRE_GOLDENS = ap.parse_args().require_goldens
        print(f"torch {torch.__version__}  python {sys.version.split()[0]}  platform {sys.platform}  "
              f"pinned={PINNED}  tmp={TMP}")
        for label, fn in (("G", section_g), ("K", section_k), ("R", lambda: section_r(state)),
                          ("L", lambda: section_l(state))):
            try:
                fn()
            except Exception as e:                        # noqa: BLE001 -- a section crash is a FAIL
                check(f"{label}_section_completed", False,
                      f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-300:]}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n[CHECKS]")
    for name, verdict, detail in CHECKS:
        print(f"  {name:72}: {verdict}{('  ' + detail) if detail and verdict != 'PASS' else ''}")
    n_pass = sum(1 for _, v, _ in CHECKS if v == "PASS")
    n_fail = sum(1 for _, v, _ in CHECKS if v == "FAIL")
    n_skip = sum(1 for _, v, _ in CHECKS if v == "SKIP")
    print(f"\nRESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({n_pass}/{n_pass + n_fail}, {n_skip} skipped)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
