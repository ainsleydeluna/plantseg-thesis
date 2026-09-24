#!/usr/bin/env python3
"""Smoke for L-AM16-ITERS, E1 part (AM-16 item 3 / DL-27; B66-prep S3): train_e1 --iterations.

Data-free and CPU-only: PlantSegDataset is replaced by a synthetic in-memory dataset, so no data root is
read. Writes only under tempfile.mkdtemp() (removed in `finally`).

Sections:
  G  the pure schedule gate and its constants
  K  main() -> run() kwargs through a recorder (P4 goldens from B66-PREP Phase A, MEASURED)
  R  in-process runs through the real run(): scheduler horizon, run_meta, LR, checkpoints, resume
     refusals, the run() defence guard, R11 (run_meta value equivalence with seed 42, X9) and R12
     (ruling R-X10: the resume pre-check reads total_iters from a real dry-run last.pt and a best
     checkpoint, CPU-mapped with weights_only=False in the helper and at main()'s call site, with CUDA
     never initialised)
  L  LR curves: same-process equality everywhere; the MEASURED goldens only in the pinned stack
     (linux + torch 2.1.0+cu121 + CPython 3.11). --require-goldens turns a golden SKIP into a FAIL.

Run:  python -B scripts/smoke_e1_schedule.py [--require-goldens]
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
import shutil        # noqa: E402
import struct        # noqa: E402
import subprocess    # noqa: E402
import tempfile      # noqa: E402
import textwrap      # noqa: E402
import traceback     # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="smoke_e1_schedule_")).resolve()
os.environ["PLANTSEG_DATA_ROOT"] = str(TMP / "no_such_data_root")     # nothing may reach a real root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                   # noqa: E402

import src.data.dataset as ds_mod              # noqa: E402
import src.data.isolation as ISO               # noqa: E402
import src.training.train_e1 as T              # noqa: E402
from configs.e1_student import E1_STUDENT      # noqa: E402

CHECKS: list[tuple[str, str, str]] = []
PINNED = (sys.platform == "linux" and torch.__version__ == "2.1.0+cu121"
          and sys.version_info[:2] == (3, 11))
GOLDEN_LR80 = "562a1873e3808ed42fa750567a6addd43654d83e52a7b1cf6c97a17fbe32f323"   # = seed-42 A40 telemetry
GOLDEN_LR160 = "dbe6d363704692590d22f0c5219d8fcaed5c94521b8ab55e89b3220d101bc54c"
P4_REAL42 = {"batch_size": 16, "ckpt_dir_arg": None, "ckpt_interval": 2000, "device": "cuda",
             "grad_clip_norm": None, "jsonl_name": "e1_telemetry.jsonl", "keep_ckpts": 3,
             "log_every": 1, "max_iters": 80000, "max_val_batches": None, "mode": "real",
             "num_workers": 12, "pretrained": "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2",
             "resume": None, "seed": 42, "val_interval": 4000}
P4_DRY = {"batch_size": 2, "ckpt_dir_arg": None, "ckpt_interval": 2000, "device": "cpu",
          "grad_clip_norm": None, "jsonl_name": "e1_telemetry.jsonl", "keep_ckpts": 3, "log_every": 1,
          "max_iters": 4, "max_val_batches": 2, "mode": "dry", "num_workers": 0, "pretrained": False,
          "resume": None, "seed": 42, "val_interval": 2}
P4_DESTS = ["batch_size", "ckpt_dir", "ckpt_interval", "confirm_real_run", "device", "dry_run",
            "grad_clip_norm", "init", "jsonl_name", "keep_ckpts", "log_every", "max_iters",
            "max_val_batches", "num_workers", "real_run", "resume", "seed", "val_interval"]
SEED42_RUN_META = {
    "event": "run_meta", "wall_clock": 1789131059.3474903, "mode": "real", "seed": 42,
    "git_head": "f77d05d7b35187bf0da7e7b94a629549fe2e1c05", "git_head_source": "git_checkout",
    "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
    "torch": "2.1.0+cu121", "numpy": "1.26.4", "device": "cuda", "cuda_available": True,
    "gpu_name": "NVIDIA A40", "num_workers": 12, "batch_size": 16, "max_iters": 80000,
    "val_interval": 4000, "max_val_batches": None, "ckpt_interval": 2000, "resumed_from": None,
    "num_classes": 116, "ignore_index": 255, "learning_rate": 0.01, "momentum": 0.9,
    "weight_decay": 0.0001, "lr_power": 0.9, "poly_horizon": 80000, "grad_clip_norm": None,
    "used_pretrained": True, "params": 2933688}
TRAIN_ROW_KEYS = {"event", "iter", "loss", "ce", "dice", "lr", "grad_norm", "wall_clock", "iter_seconds",
                  "samples_per_sec"}
VAL_ROW_KEYS = {"event", "iter", "all_class_miou", "disease_only_miou_PROVISIONAL", "per_class_iou",
                "per_class_eligible", "n_eligible_classes", "val_batches", "val_total_px", "val_seconds",
                "wall_clock"}
BEST_CKPT_KEYS = {"iter", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scheduler",
                  "best_val_miou_all_class", "num_classes"}
LAST_CKPT_KEYS = BEST_CKPT_KEYS | {"rng_state", "prev_lr", "best_ckpt"}
SEED42_SCHED = {"total_iters": 80000, "power": 0.9, "base_lrs": [0.01], "last_epoch": 80000,
                "verbose": False, "_step_count": 80001, "_get_lr_called_within_step": False,
                "_last_lr": [0.0]}                    # MEASURED from the seed-42 checkpoint (Phase A)
LR80_ENDPOINTS = {1: 0.009999887499929687, 40000: 0.005358867312681493, 80000: 0.0}
LR160_ENDPOINTS = {1: 0.009999943749982422, 4000: 0.009774716137503494, 80000: 0.005358867312681469,
                   160000: 0.0}
# R11 (X9): fields that legitimately differ on a CPU, no-download, max_iters=0 replay; torch and numpy
# are exempt on the host only (in the pinned image they must equal seed 42's).
R11_EXEMPT = {"wall_clock", "git_head", "git_head_source", "image_digest", "gpu_name", "cuda_available",
              "device", "used_pretrained", "max_iters"}
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


SPY_HP: list = []      # (horizon, power, lr, momentum, weight_decay) of every scheduler run() built


def lr_seq(horizon, n=None, power=None, lr=None, momentum=None, wd=None):
    """Exactly probe P1's method: SGD on 4 zero params, T.build_scheduler, step, record post-step lr.
    Hyper-parameters default to E1_STUDENT; L01/L04 replay the values run() actually used."""
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
    """Stands in for PlantSegDataset: train 4 / val 2 samples, 512x512, labels 0..115 with 255 rows."""

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


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def main_rec(argv, *, iso_calls=None):
    """train_e1.main with run() replaced by a recorder and the M11 check stubbed (no data root)."""
    calls = []

    def rec(**kw):
        calls.append(kw)
        return 0

    def iso_stub(root, expected=None):
        if iso_calls is not None:
            iso_calls.append(str(root))
        return {"counts": {"train": {"images": 5367, "masks": 5367}, "val": {"images": 846, "masks": 846}},
                "test_surfaces": {}, "method": "stub"}
    err, out = io.StringIO(), io.StringIO()
    with patched(T, "run", rec), patched(ISO, "assert_trainval_only_root", iso_stub), \
            contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        try:
            rc = T.main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, calls, err.getvalue()


def run_real(argv, spy):
    """train_e1.main through the REAL run() on the synthetic dataset, with a build_scheduler spy."""
    real_bs = T.build_scheduler

    def bs_spy(opt, horizon, power):
        spy.append((horizon, power))
        g = opt.param_groups[0]
        SPY_HP.append((horizon, power, g["lr"], g["momentum"], g["weight_decay"]))
        return real_bs(opt, horizon, power)
    out, err = io.StringIO(), io.StringIO()
    with patched(ds_mod, "PlantSegDataset", SynthDS), patched(T, "build_scheduler", bs_spy), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = T.main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


def rows(d: Path):
    with open(d / "e1_telemetry.jsonl", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ------------------------------------------------------------------------------------------ G
def section_g():
    ok_none = all(T.schedule_gate_error(*a) is None for a in
                  (("dry", 80000, 4), ("dry", 80000, 80000), ("dry", 160000, 4), ("real", 80000, 80000),
                   ("real", 160000, 160000)))
    check("G01 schedule_gate_error admits (dry,80000,4) (dry,80000,80000) (dry,160000,4) (real,80000,80000) "
          "(real,160000,160000)", ok_none)
    bad = [a for a in (("dry", 80000, 80001), ("real", 80000, 160000), ("real", 160000, 80000),
                       ("real", 80000, 79999), ("dry", 120000, 4))
           if not isinstance(T.schedule_gate_error(*a), str)]
    check("G02 it refuses (dry,80000,80001) (real,80000,160000) (real,160000,80000) (real,80000,79999) "
          "(dry,120000,4)", bad == [], str(bad))
    check("G03 REGISTERED_POLY_HORIZONS (80000,160000); LONGER_SCHEDULE_ITERS 160000; E1_STUDENT unchanged "
          "(iterations 80000, val_interval 4000)",
          T.REGISTERED_POLY_HORIZONS == (80000, 160000) and T.LONGER_SCHEDULE_ITERS == 160000
          and E1_STUDENT["iterations"] == 80000 and E1_STUDENT["val_interval"] == 4000)


# ------------------------------------------------------------------------------------------ K
def section_k():
    check("K01 argparse dests == the P4 18 + iterations",
          sorted(vars(T.parse_args([]))) == sorted(P4_DESTS + ["iterations"]))
    base = ["--real-run", "--confirm-real-run", "--device", "cuda"]
    rc, c, _ = main_rec(base + ["--num-workers", "12"])
    k02 = c[-1] if c else {}
    check("K02 real default -> kwargs == P4 real golden + poly_horizon 80000",
          rc == 0 and {k: v for k, v in k02.items() if k != "poly_horizon"} == P4_REAL42
          and k02.get("poly_horizon") == 80000, json.dumps(k02, default=str)[:160])
    rc, c, _ = main_rec(base)
    exp_nw = min(max((os.cpu_count() or 4) - 2, 1), 12)
    check("K03 without --num-workers: the host-dependent default, everything else equal",
          rc == 0 and c[-1]["num_workers"] == exp_nw and dict(c[-1], num_workers=12) == k02, f"nw={exp_nw}")
    rc, c, _ = main_rec(base + ["--num-workers", "12", "--seed", "43"])
    check("K04 --seed 43 changes only seed", rc == 0 and c[-1] == dict(k02, seed=43))
    rc, c, _ = main_rec(base + ["--num-workers", "12", "--iterations", "80000"])
    check("K05 --iterations 80000 == the default", rc == 0 and c[-1] == k02)
    rc, c, _ = main_rec(base + ["--num-workers", "12", "--max-iters", "80000"])
    check("K06 --max-iters 80000 == the default", rc == 0 and c[-1] == k02)
    rc, c, _ = main_rec(base + ["--num-workers", "12", "--iterations", "160000"])
    check("K07 --iterations 160000 -> max_iters == poly_horizon == 160000, nothing else changes",
          rc == 0 and c[-1] == dict(k02, max_iters=160000, poly_horizon=160000))
    rc, c, err = main_rec(base + ["--max-iters", "160000"])
    check("K08 real --max-iters 160000 alone (LR 0 after 80k before S3) -> exit 2, hint --iterations 160000",
          rc == 2 and c == [] and "--iterations 160000" in err, err[-160:])
    rc, c, err = main_rec(base + ["--iterations", "160000", "--max-iters", "80000"])
    check("K09 real --iterations 160000 --max-iters 80000 -> exit 2 (the schedule gate)",
          rc == 2 and c == [] and "--max-iters 80000 != --iterations 160000" in err, err[-120:])
    rc, c, err = main_rec(base + ["--max-iters", "1000"])
    check("K10 real --max-iters 1000 (a truncated real run) -> exit 2 (the schedule gate)",
          rc == 2 and c == [] and "--max-iters 1000 != --iterations 80000" in err, err[-120:])
    rc, c, _ = main_rec(base + ["--iterations", "120000"])
    check("K11 --iterations 120000 (unregistered) -> argparse exit 2", rc == 2 and c == [])
    rc, c, _ = main_rec(["--dry-run"])
    check("K12 --dry-run -> P4 dry golden + poly_horizon 80000", rc == 0 and c[-1] == dict(P4_DRY, poly_horizon=80000))
    rc, c, _ = main_rec(["--dry-run", "--batch-size", "1", "--max-iters", "2", "--val-interval", "2",
                         "--max-val-batches", "1", "--ckpt-dir", str(TMP / "k13")])
    check("K13 the verify_env [17] arguments -> max_iters 2, poly_horizon 80000",
          rc == 0 and c[-1]["max_iters"] == 2 and c[-1]["poly_horizon"] == 80000)
    rc, c, _ = main_rec(["--dry-run", "--iterations", "160000"])
    check("K14 --dry-run --iterations 160000 -> max_iters 4, poly_horizon 160000",
          rc == 0 and c[-1]["max_iters"] == 4 and c[-1]["poly_horizon"] == 160000)
    rc, c, err = main_rec(["--dry-run", "--max-iters", "80001"])
    check("K15 --dry-run --max-iters 80001 -> exit 2 (the schedule gate)",
          rc == 2 and c == [] and "--max-iters 80001 exceeds the poly horizon 80000" in err, err[-120:])
    ck80 = TMP / "k_ck80.pt"
    torch.save({"scheduler_state_dict": {"total_iters": 80000}}, ck80)
    rc, c, err = main_rec(["--dry-run", "--iterations", "160000", "--resume", str(ck80)])
    check("K16 resuming an 80000-horizon checkpoint with --iterations 160000 -> exit 2 (a horizon mismatch, not "
          "a read failure), run() not called",
          rc == 2 and c == [] and "is 80000, this run's is 160000" in err and "Pass --iterations 80000" in err
          and "cannot read" not in err, err[-120:])
    ck_none = TMP / "k_cknone.pt"
    torch.save({"iter": 5}, ck_none)
    rc, c, err = main_rec(["--dry-run", "--resume", str(ck_none)])
    check("K17 a checkpoint without scheduler_state_dict -> exit 2 (fail closed); the hint never suggests an "
          "unregistered --iterations value",
          rc == 2 and c == [] and "cannot resume it" in err and "--iterations None" not in err, err[-160:])
    rc, c, _ = main_rec(["--dry-run", "--resume", str(ck80)])
    check("K18 a matching resume -> run() called with poly_horizon 80000 and resume == path",
          rc == 0 and c[-1]["poly_horizon"] == 80000 and c[-1]["resume"] == str(ck80))
    sig = inspect.signature(T.run)
    params = sig.parameters
    check("K19 run() is keyword-only; names == P4 kwargs + poly_horizon; defaults unchanged + poly_horizon None",
          all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
          and set(params) == set(P4_REAL42) | {"poly_horizon"}
          and params["resume"].default is None and params["ckpt_interval"].default == 2000
          and params["jsonl_name"].default == "e1_telemetry.jsonl" and params["keep_ckpts"].default == 3
          and params["poly_horizon"].default is None)
    iso_calls = []
    rc, c, err = main_rec(base + ["--max-iters", "160000"], iso_calls=iso_calls)
    rc_cpu, c2, err2 = main_rec(["--real-run", "--confirm-real-run", "--device", "cpu", "--max-iters", "160000"],
                                iso_calls=iso_calls)
    check("K20 order: the schedule gate refuses before the CPU refusal and before any data-root check (S2b)",
          rc == 2 and rc_cpu == 2 and iso_calls == [] and "poly horizon" in err2 and "on CPU" not in err2,
          err2[-120:])
    rc, c, err = main_rec(["--dry-run", "--resume", str(TMP / "no_such_ckpt.pt")])
    check("K21 an unreadable --resume path -> exit 2 before run()", rc == 2 and c == [] and "cannot read" in err)


# ------------------------------------------------------------------------------------------ R
def section_r(state):
    d1, d2 = TMP / "r_d1", TMP / "r_d2"
    spy = []
    args = ["--dry-run", "--batch-size", "1", "--max-iters", "2", "--val-interval", "2", "--max-val-batches", "1"]
    SPY_HP.clear()
    rc, out, err = run_real(args + ["--ckpt-dir", str(d1)], spy)
    state["hp80"] = SPY_HP[0] if SPY_HP else None
    last = [ln for ln in out.splitlines() if ln.strip()][-1] if out.strip() else ""
    check("R01 verify_env-style dry run through the real run() -> PASS 6/6",
          rc == 0 and last == "RESULT: PASS (6/6 checks exercised, 0 skipped)", (last or err[-200:]))
    check("R02 run() built exactly one scheduler: (80000, 0.9)", spy == [(80000, 0.9)], str(spy))
    r1 = rows(d1) if (d1 / "e1_telemetry.jsonl").exists() else []
    meta = next((r for r in r1 if r["event"] == "run_meta"), {})
    check("R03 run_meta: the 29 seed-42 keys; poly_horizon 80000; max_iters 2",
          set(meta) == set(SEED42_RUN_META) and meta.get("poly_horizon") == 80000 and meta.get("max_iters") == 2,
          str(sorted(set(meta) ^ set(SEED42_RUN_META))))
    curve80_head, _ = lr_seq(80000, 3)
    tr = [r for r in r1 if r["event"] == "train"]
    va = [r for r in r1 if r["event"] == "val"]
    check("R04 telemetry lrs == the 80k curve prefix bitwise; train rows 10 keys, val rows 11 keys",
          [r["lr"] for r in tr] == curve80_head[:2] and all(set(r) == TRAIN_ROW_KEYS for r in tr)
          and bool(va) and all(set(r) == VAL_ROW_KEYS for r in va))
    best = torch.load(d1 / "e1_student_best_iter2.pt", map_location="cpu", weights_only=False)
    lastck = torch.load(d1 / "last.pt", map_location="cpu", weights_only=False)
    check("R05 checkpoint key sets unchanged (best 7, last.pt 10)",
          set(best) == BEST_CKPT_KEYS and set(lastck) == LAST_CKPT_KEYS)
    p = torch.nn.Parameter(torch.zeros(4))
    ref_opt = torch.optim.SGD([p], lr=0.01)
    ref, _ = T.build_scheduler(ref_opt, 80000, 0.9)
    for _ in range(2):
        ref_opt.step()
        ref.step()
    check("R06 best scheduler_state_dict == a reference PolynomialLR(80000, 0.9) stepped twice",
          best["scheduler_state_dict"] == ref.state_dict() and best["scheduler_state_dict"]["total_iters"] == 80000)
    spy2 = []
    SPY_HP.clear()
    rc, out, err = run_real(args + ["--iterations", "160000", "--ckpt-dir", str(d2)], spy2)
    state["hp160"] = SPY_HP[0] if SPY_HP else None
    r2 = rows(d2) if (d2 / "e1_telemetry.jsonl").exists() else []
    meta2 = next((r for r in r2 if r["event"] == "run_meta"), {})
    curve160_head, _ = lr_seq(160000, 3)
    ck2 = torch.load(d2 / "last.pt", map_location="cpu", weights_only=False) if (d2 / "last.pt").exists() else {}
    check("R07 --iterations 160000: PASS 6/6, scheduler (160000, 0.9), poly_horizon 160000, lrs == 160k prefix, "
          "checkpoint total_iters 160000",
          rc == 0 and "RESULT: PASS (6/6 checks exercised, 0 skipped)" in out and spy2 == [(160000, 0.9)]
          and meta2.get("poly_horizon") == 160000
          and [r["lr"] for r in r2 if r["event"] == "train"] == curve160_head[:2]
          and ck2.get("scheduler_state_dict", {}).get("total_iters") == 160000, str(spy2))
    d3, d4 = TMP / "r_d3", TMP / "r_d4"
    rc3, _, e3 = run_real(args + ["--iterations", "160000", "--resume", str(d1 / "last.pt"), "--ckpt-dir", str(d3)], [])
    rc4, _, e4 = run_real(args + ["--resume", str(d2 / "last.pt"), "--ckpt-dir", str(d4)], [])
    check("R08 cross-horizon resumes of real dry-run last.pt files are refused as horizon mismatches (exit 2, "
          "never 'cannot read') and their ckpt dirs are never created",
          rc3 == 2 and rc4 == 2 and not d3.exists() and not d4.exists()
          and "is 80000, this run's is 160000" in e3 and "is 160000, this run's is 80000" in e4
          and "cannot read" not in e3 + e4, f"{rc3}/{rc4} {e3[-80:]} {e4[-80:]}")
    rc13, out13, _ = run_real(args + ["--resume", str(d1 / "last.pt"), "--ckpt-dir", str(d1)], [])
    nonempty13 = [ln for ln in out13.splitlines() if ln.strip()]
    check("R13 a finished resume (last.pt at iter 2, --max-iters 2) passes the defence guard and prints the "
          "unchanged NOOP tokens",
          rc13 == 0 and bool(nonempty13) and nonempty13[-1] == "RESULT: NOOP (already complete at iter 2)"
          and "[CHECKS] 0/6 exercised, 6 skipped (no iterations ran)" in out13,
          (nonempty13[-1] if nonempty13 else out13[-120:]))
    spy9 = []
    rc9, out9, _ = run_real(["--dry-run", "--batch-size", "1", "--max-iters", "3", "--val-interval", "2",
                             "--max-val-batches", "1", "--resume", str(d1 / "last.pt"), "--ckpt-dir", str(d1)], spy9)
    lr3 = [r["lr"] for r in rows(d1) if r["event"] == "train" and r["iter"] == 3]
    check("R09 a same-horizon resume of a real dry-run last.pt continues the 80k curve (lr@3 == curve80[2])",
          rc9 == 0 and "RESULT: PASS (6/6 checks exercised, 0 skipped)" in out9 and lr3 == [curve80_head[2]],
          f"rc={rc9} lr3={lr3}")
    try:
        with patched(ds_mod, "PlantSegDataset", SynthDS), contextlib.redirect_stdout(io.StringIO()):
            T.run(mode="dry", device="cpu", pretrained=False, batch_size=1, max_iters=4, val_interval=2,
                  max_val_batches=1, num_workers=0, ckpt_dir_arg=str(TMP / "r_d5"), grad_clip_norm=None,
                  log_every=1, seed=42, resume=str(d1 / "last.pt"), poly_horizon=160000)
        e10 = None
    except RuntimeError as e:
        e10 = str(e)
    check("R10 run() defence guard: poly_horizon 160000 with an 80000 checkpoint -> 'E1 schedule guard'",
          e10 is not None and "E1 schedule guard" in e10, str(e10)[:120])
    d6 = TMP / "r_d6"
    try:
        with patched(ds_mod, "PlantSegDataset", SynthDS), contextlib.redirect_stdout(io.StringIO()):
            T.run(**dict(P4_REAL42, device="cpu", pretrained=False, max_iters=0, ckpt_dir_arg=str(d6)),
                  poly_horizon=80000)
        e11 = None
    except RuntimeError as e:
        e11 = str(e)
    m6 = next((r for r in rows(d6) if r["event"] == "run_meta"), {}) if (d6 / "e1_telemetry.jsonl").exists() else {}
    exempt = set(R11_EXEMPT) | (set() if PINNED else {"torch", "numpy"})
    diff = sorted(k for k in SEED42_RUN_META if k not in exempt and m6.get(k) != SEED42_RUN_META[k])
    check("R11 X9: an 80k real-mode run_meta (CPU replay, guard stops before any iteration) equals seed 42's in "
          f"every non-exempt field{' incl. torch/numpy (pinned stack)' if PINNED else ' (torch/numpy exempt on this host)'}",
          e11 is not None and "E1 schedule guard" in e11 and set(m6) == set(SEED42_RUN_META) and diff == [],
          f"diff={diff} exempt={sorted(exempt)}")
    child = textwrap.dedent(f'''
        import json, sys
        sys.path.insert(0, {str(REPO)!r})
        import torch
        calls = {{"lazy_init": 0}}
        real_lazy = torch.cuda._lazy_init
        def spy():
            calls["lazy_init"] += 1
            return real_lazy()
        torch.cuda._lazy_init = spy
        import src.training.train_e1 as T
        out = {{}}
        loads = []
        real_load = torch.load
        def load_spy(*a, **k):
            loads.append({{"map_location": k.get("map_location"), "weights_only": k.get("weights_only")}})
            return real_load(*a, **k)
        torch.load = load_spy
        for label, p in (("last80", {str(d1 / "last.pt")!r}), ("best80", {str(d1 / "e1_student_best_iter2.pt")!r}),
                         ("last160", {str(d2 / "last.pt")!r})):
            out[label] = T.checkpoint_poly_horizon(p)
        out["loads"] = list(loads)
        del loads[:]
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            out["main_rc"] = T.main(["--dry-run", "--iterations", "160000", "--resume", {str(d1 / "last.pt")!r},
                                     "--ckpt-dir", {str(TMP / "r_d7")!r}])
        out["main_loads"] = list(loads)
        out["main_mismatch_text"] = "is 80000, this run's is 160000" in err.getvalue()
        torch.load = real_load
        try:
            torch.load({str(d1 / "last.pt")!r}, map_location="cpu", weights_only=True)
            out["weights_only_true_last"] = "loaded"
        except Exception as e:
            out["weights_only_true_last"] = type(e).__name__
        out["lazy_init_calls"] = calls["lazy_init"]
        out["cuda_initialized"] = torch.cuda.is_initialized()
        print("RESULT_JSON:" + json.dumps(out))
    ''')
    r = subprocess.run([sys.executable, "-B", "-c", child], cwd=str(TMP), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT_JSON:")]
    res = json.loads(lines[-1][12:]) if lines else {}
    check("R12 R-X10: the pre-check reads total_iters from a real dry-run last.pt (80000/160000) and a best "
          "checkpoint (80000), every load mapped to 'cpu' with weights_only=False, both in the helper and at "
          "main()'s pre-check call site (one load, then the mismatch refusal before run()); CUDA never initialised "
          "(0 lazy-init calls; on a CPU-only stack the load spy is the discriminating part); weights_only=True "
          "cannot load that last.pt",
          res.get("last80") == 80000 and res.get("best80") == 80000 and res.get("last160") == 160000
          and res.get("lazy_init_calls") == 0 and res.get("cuda_initialized") is False
          and res.get("loads") == [{"map_location": "cpu", "weights_only": False}] * 3
          and res.get("main_rc") == 2 and res.get("main_mismatch_text") is True
          and res.get("main_loads") == [{"map_location": "cpu", "weights_only": False}]
          and not (TMP / "r_d7").exists()
          and res.get("weights_only_true_last") == "UnpicklingError",
          json.dumps(res) if res else (r.stdout + r.stderr)[-240:])
    state["spy_horizon"] = spy[0][0] if spy else None
    state["spy2_horizon"] = spy2[0][0] if spy2 else None


# ------------------------------------------------------------------------------------------ L
def section_l(state):
    curve80, sch80 = lr_seq(E1_STUDENT["iterations"])
    hp80 = state.get("hp80")
    via_run = lr_seq(hp80[0], 80000, *hp80[1:])[0] if hp80 else []
    e1_hp = (E1_STUDENT["lr_power"], E1_STUDENT["learning_rate"], E1_STUDENT["momentum"],
             E1_STUDENT["weight_decay"])
    check("L01 run()'s optimizer/scheduler hyper-parameters are E1_STUDENT's (power, lr, momentum, weight "
          "decay), and replaying its scheduler (horizon, power, lr) gives the canonical 80,000-value curve",
          bool(hp80) and tuple(hp80[1:]) == e1_hp and via_run == curve80 and len(curve80) == 80000, str(hp80))
    golden("L02 sha256('<80000d') == the MEASURED seed-42 A40 telemetry golden",
           lambda: (sha(curve80) == GOLDEN_LR80, sha(curve80)[:16]))
    check("L03 the 80k curve ends at 0.0, is > 0 before, and never increases",
          curve80[-1] == 0.0 and all(v > 0 for v in curve80[:-1])
          and all(a >= b for a, b in zip(curve80, curve80[1:])))
    curve160, _ = lr_seq(160000)
    hp160 = state.get("hp160")
    via_run160 = lr_seq(hp160[0], 160000, *hp160[1:])[0] if hp160 else []
    check("L04 run()'s 160k hyper-parameters are E1_STUDENT's; replaying its scheduler gives the canonical "
          "160k curve; it starts above the 80k curve",
          bool(hp160) and tuple(hp160[1:]) == e1_hp and via_run160 == curve160 and curve160[0] > curve80[0],
          str(hp160))
    golden("L05 sha256('<160000d') == the MEASURED 160k golden",
           lambda: (sha(curve160) == GOLDEN_LR160, sha(curve160)[:16]))
    check("L06 the 160k curve: 160,000 values, exactly one 0.0 (the last), never increasing (no LR-0 tail)",
          len(curve160) == 160000 and curve160.count(0.0) == 1 and curve160[-1] == 0.0
          and all(a >= b for a, b in zip(curve160, curve160[1:])))
    golden("L07 endpoint literals (80k: lr[1], lr[40000], lr[80000]; 160k: lr[1], lr[4000], lr[80000], lr[160000])",
           lambda: (all(curve80[k - 1] == v for k, v in LR80_ENDPOINTS.items())
                    and all(curve160[k - 1] == v for k, v in LR160_ENDPOINTS.items()), ""))
    golden("L08 the scheduler state after 80,000 steps == the MEASURED seed-42 checkpoint's",
           lambda: (sch80.state_dict() == SEED42_SCHED, json.dumps(sch80.state_dict(), default=str)[:160]))


def main() -> int:
    global REQUIRE_GOLDENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-goldens", action="store_true")
    REQUIRE_GOLDENS = ap.parse_args().require_goldens
    print(f"torch {torch.__version__}  python {sys.version.split()[0]}  platform {sys.platform}  "
          f"pinned={PINNED}  tmp={TMP}")
    state = {}
    try:
        for label, fn in (("G", section_g), ("K", section_k), ("R", lambda: section_r(state)),
                          ("L", lambda: section_l(state))):
            try:
                fn()
            except Exception as e:                        # noqa: BLE001 -- a section crash is a FAIL
                check(f"{label} section completed", False,
                      f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-300:]}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    width = max(len(n) for n, _, _ in CHECKS)
    for name, verdict, detail in CHECKS:
        print(f"  [{verdict}] {name:<{width}}  {detail}")
    n_pass = sum(1 for _, v, _ in CHECKS if v == "PASS")
    n_fail = sum(1 for _, v, _ in CHECKS if v == "FAIL")
    n_skip = sum(1 for _, v, _ in CHECKS if v == "SKIP")
    print(f"SUMMARY  {n_pass}/{n_pass + n_fail} checks passed ({n_skip} skipped)")
    print(f"RESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({n_pass}/{n_pass + n_fail}, {n_skip} skipped)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
