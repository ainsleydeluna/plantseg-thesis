#!/usr/bin/env python3
"""AM-7 divergence aborts of the distillation trainer (lane L-KD-HARDEN item 3, L-AM7; AM-7a record A).

Part 1 - AM7bMonitor units (pure python, ramp_iters = 335 as at batch 16, window 100, factor 5.0):
  B1  E3's ramp pattern never fires: the total loss rises with the first-epoch ramp (distillation
      weights 0 -> target), peaks at the ramp end and then decays over 80,000 iterations; the SAME trace
      fires when the ramp iterations are allowed into the windows (ramp_iters=0), so the exclusion is
      what keeps it quiet
  B2  a post-ramp step to 5.01 x the running minimum fires at the first iteration whose 100-iteration
      mean exceeds 5 x the minimum; the result carries window_mean, running_min, ratio > 5 and
      threshold = 5 x running_min
  B3  a step to exactly 5.0 x never fires (strictly greater)
  B4  pre-ramp values never enter: NaN, inf and huge ramp-phase losses leave every later decision
      identical to B2's
  B5  no window is evaluated before ramp_iters + 100; the first is evaluated exactly there, and the
      ramp's last iteration (it = ramp_iters) never enters one
  P1  configs/distill.py AM7_DIVERGENCE carries the registered window 100, factor 5.0 and
      post_ramp_only True; a drifted value is refused (train_distill refuses to import it)
Part 2 - real-mode CPU integration: synthetic 512x512 TRAIN/VAL set (4 + 2 images, batch 2, so
  ramp_iters = 2), stage E2, the invariance harness's stub SegNeXt teacher (a real run needs its
  M4-KD NMF stream), pretrained=False (no download), E1_STUDENT["iterations"] patched to a short
  horizon so run(mode="real") is legal on CPU. Faults are injected through the real code path:
  A1  non-finite loss at iteration 3 (distillation_losses wrapper) -> iteration 3's train row with the
      loss null and listed in `nonfinite`, lr and grad norms null (not computed), then the run_abort
      record {rule AM-7(a), cause student_divergence, detail.loss "nan", input/teacher/params finite},
      then RunAborted, which pickles with its record; no run_end; strict JSON; the weights are not
      stepped at iteration 3
  A2  finite loss, non-finite gradient at iteration 3 (sqrt(0) in the graph) -> abort after backward and
      before optimizer.step: the row carries the (null) grad norms, the record detail.grad_norm "nan"
  A3  a NaN input pixel at iteration 3 -> rule AM-7(a), cause input_nonfinite, input_finite false
  A4  a NaN teacher output at iteration 3 -> rule AM-7(a), cause teacher_nonfinite, teacher_finite false
  A5  R8-1: a non-finite loss AT iteration 1 is a step-1 failure, not AM-7 (a) (before the first update
      nothing can diverge) -> iteration 1's train row (loss null and listed in `nonfinite`, lr and grad
      norms null), then run_abort {rule step1_checks, cause checks_failed, detail.loss "nan",
      detail.grad_norm null}, and nothing else in the telemetry
  A6  R8-1: a finite loss and a non-finite gradient at iteration 1 (the A2 fault) -> run_abort {rule
      step1_checks, cause checks_failed, detail.grad_norm non-finite, detail.loss finite}
  B6  AM-7(b) with the window patched to 2: a +1000 loss offset from iteration 5 on (after the ramp)
      -> rule AM-7(b), cause student_divergence, ratio > 5, iter >= ramp_iters + window
  V1  a non-finite VAL all-class mIoU -> the val row (strict JSON), then run_abort {rule val_nonfinite,
      cause val_nonfinite}
  S1  item 1b: a distillation total that is not exactly 0 at step 1 (distill_zero_at_step1 and
      sup_added_unscaled fail) -> iteration 1's train row, then run_abort {rule step1_checks, cause
      checks_failed}; smoke_distill_stepchecks.py F1 covers the same rule with an input-identity mutant
  D1  dry mode ignores the same non-finite loss (AM-7 aborts are real-run only) and still ends with
      run_end; its telemetry stays strict JSON
GPUs are hidden (CUDA_VISIBLE_DEVICES="" before torch loads). On a GPU host a CPU run touches CUDA
(run_meta's gpu_name calls torch.cuda.get_device_name, which initialises it; the loaders pin memory), and
run()'s CUDA-order gate would then refuse the next real run in this process. A refused run is a FAIL with
its reason, never a traceback.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only smoke: hide GPUs before torch loads (above)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import configs.data as cdata  # noqa: E402
import src.training.train_distill as td  # noqa: E402
from configs.e1_student import E1_STUDENT  # noqa: E402
from scripts.invariance_harness import build_stub_segnext, make_synthetic_dataset  # noqa: E402
from src.distill.segnext_teacher import SegNeXtTeacherAdapter  # noqa: E402
from src.distill.teacher import FrozenTeacher  # noqa: E402

RAMP = 335
ABORT_KEYS = ["event", "iter", "rule", "cause", "detail", "input_finite", "teacher_finite", "params_finite",
              "n_val", "wall_clock"]
DETAIL_KEYS = ["loss", "grad_norm", "window_mean", "running_min", "ratio", "threshold"]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ================================================================================== Part 1
def feed(monitor, trace, start=1):
    """Feed trace values from iteration `start`; return (iteration, result) of the first firing."""
    for i, v in enumerate(trace, start=start):
        hit = monitor.update(i, v)
        if hit is not None:
            return i, hit
    return None, None


def e3_ramp_trace(n: int = 80000, sup0: float = 4.0, dmax: float = 100.0) -> list[float]:
    """A total loss shaped like E3's: supervised ~4 decaying slowly; distillation weighted by the
    first-epoch linear ramp, then decaying to a fifth of its peak; a small deterministic wiggle."""
    out = []
    for it in range(1, n + 1):
        ramp = min(max((it - 1) / (RAMP - 1), 0.0), 1.0)
        decay = 0.2 + 0.8 * math.exp(-it / 15000.0)
        wiggle = 1.0 + 0.03 * math.sin(it * 0.7)
        out.append((sup0 * (0.5 + 0.5 * math.exp(-it / 30000.0)) + dmax * ramp * decay) * wiggle)
    return out


def test_monitor_units() -> None:
    trace = e3_ramp_trace()
    at, hit = feed(td.AM7bMonitor(RAMP, window=100, factor=5.0), trace)
    check("B1_e3_ramp_pattern_never_fires", at is None, f"fired at {at}: {hit}")
    at0, hit0 = feed(td.AM7bMonitor(0, window=100, factor=5.0), trace)
    check("B1_same_trace_fires_if_ramp_entered", at0 is not None and at0 <= RAMP + 100,
          f"ramp_iters=0 fired at {at0}: {hit0}")

    step = [1.0] * 200 + [5.01] * 400
    at, hit = feed(td.AM7bMonitor(RAMP), step, start=RAMP + 1)
    check("B2_step_to_5_01x_fires_when_window_is_all_5_01", at == RAMP + 300 and hit is not None
          and hit["ratio"] > 5.0 and hit["running_min"] == 1.0 and hit["threshold"] == 5.0
          and hit["window_mean"] > 5.0 and list(hit) == ["window_mean", "running_min", "ratio", "threshold"],
          f"at={at} hit={hit}")
    at5, hit5 = feed(td.AM7bMonitor(RAMP), [1.0] * 200 + [5.0] * 2000, start=RAMP + 1)
    check("B3_exactly_5x_never_fires", at5 is None, f"fired at {at5}: {hit5}")
    garbage = [float("nan"), float("inf"), 1e30, -5.0] * (RAMP // 4) + [7.0] * (RAMP % 4)
    atg, hitg = feed(td.AM7bMonitor(RAMP), garbage + step)
    check("B4_pre_ramp_values_never_enter", atg == at and hitg == hit, f"at={atg} hit={hitg}")
    m = td.AM7bMonitor(RAMP)
    m.update(RAMP, 1e9)                           # the ramp's last iteration (it = ramp_iters) stays out
    feed(m, [1.0] * 99, start=RAMP + 1)
    before = m.n_windows
    m.update(RAMP + 100, 1.0)
    check("B5_first_window_at_ramp_plus_100", before == 0 and m.n_windows == 1 and m.running_min == 1.0,
          f"windows before={before} after={m.n_windows} running_min={m.running_min}")

    cfg = dict(td.AM7_DIVERGENCE)
    drifts = ({"factor": 4.0}, {"window": 50}, {"post_ramp_only": False}, {"window": True},
              {"post_ramp_only": 1})
    check("P1_registered_parameters_pinned", td.am7_divergence_error(cfg) is None
          and all(td.am7_divergence_error({**cfg, **d}) is not None for d in drifts)
          and td.AM7_REGISTERED == {"window": 100, "factor": 5.0, "post_ramp_only": True},
          f"config={cfg} registered={td.AM7_REGISTERED}")


# ================================================================================== Part 2
def no_nan_constants(tok):
    raise ValueError(f"non-strict JSON token {tok}")


def rows(path: Path) -> list[dict]:
    """The telemetry rows, strict JSON; none when run() refused before writing any (a FAIL, not a
    traceback)."""
    if not path.is_file():
        return []
    return [json.loads(ln, parse_constant=no_nan_constants)
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def params(model) -> dict:
    """The trainable weights only: BatchNorm running statistics are buffers, which every train-mode
    forward updates, including the forward of an iteration that then aborts."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


class Capture:
    """Captures the student run() builds and its weights after each written train row."""

    def __init__(self):
        self.student = None
        self.after = {}

    def install(self):
        self._build, self._jsonl = td.build_student, td._jsonl

        def build_student(*a, **k):
            self.student = self._build(*a, **k)
            return self.student

        def jsonl(path, rec):
            self._jsonl(path, rec)
            if rec.get("event") == "train" and self.student is not None:
                self.after[rec["iter"]] = params(self.student)

        td.build_student, td._jsonl = build_student, jsonl

    def uninstall(self):
        td.build_student, td._jsonl = self._build, self._jsonl


def stub_teacher():
    return FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext()))


def real_run(tag: str, *, mode="real", max_iters=6, val_interval=6, teacher=None) -> dict:
    """run() on the synthetic set; returns {rc|exc, rows, capture}."""
    ckpt = Path(tempfile.mkdtemp(prefix=f"kdh_am7_{tag}_"))
    cap = Capture()
    cap.install()
    saved_h = E1_STUDENT["iterations"]
    E1_STUDENT["iterations"] = max_iters
    out = {"exc": None, "rc": None}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out["rc"] = td.run(stage=td.resolve_stage("e2"), mode=mode, device="cpu", pretrained=False,
                               teacher=teacher or stub_teacher(), lambda_logit=1.0, batch_size=2,
                               max_iters=max_iters, val_interval=val_interval, max_val_batches=None,
                               num_workers=0, ckpt_dir_arg=str(ckpt / "run"), grad_clip_norm=None,
                               log_every=1, seed=42)
    except td.RunAborted as e:
        out["exc"] = e
    finally:
        E1_STUDENT["iterations"] = saved_h
        cap.uninstall()
    out["rows"] = rows(ckpt / "run" / "e2_telemetry.jsonl")
    out["cap"] = cap
    return out


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def distill_wrapper(fn_for_call):
    """Wrap td.distillation_losses; fn_for_call(call_no, total, kwargs) -> new total."""
    real = td.distillation_losses
    calls = {"n": 0}

    def wrapped(**kw):
        calls["n"] += 1
        total, parts = real(**kw)
        return fn_for_call(calls["n"], total, kw), parts
    return wrapped


def tail(r: dict) -> tuple[dict, dict]:
    """(the last telemetry row, the row before it); {} where absent."""
    rs = r["rows"]
    return (rs[-1] if rs else {}), (rs[-2] if len(rs) > 1 else {})


def check_safe(name: str, cond, detail: str = "") -> None:
    """check() on a condition evaluated here: an exception (a run refused before its rows, a row
    without an expected key) is a FAIL with its message, so the smoke always reaches its RESULT."""
    try:
        ok = bool(cond())
    except Exception as e:                        # noqa: BLE001 - reported as the FAIL detail
        ok, detail = False, f"{type(e).__name__}: {e}"
    check(name, ok, detail)


def abort_ok(r: dict, *, rule: str, cause: str, it: int) -> tuple[bool, str]:
    rs = r["rows"]
    if len(rs) < 2:
        return False, f"rc={r['rc']} exc={r['exc']} rows={rs}"
    last = rs[-1]
    ok = (isinstance(r["exc"], td.RunAborted) and last.get("event") == "run_abort"
          and list(last) == ABORT_KEYS and list(last.get("detail", {})) == DETAIL_KEYS
          and last["rule"] == rule and last["cause"] == cause and last["iter"] == it
          and not any(x.get("event") == "run_end" for x in rs)
          and rs[-2].get("event") in ("train", "val") and rs[-2].get("iter") == it)
    return ok, json.dumps(last)[:300]


def test_real_mode() -> None:
    nan_at_3 = distill_wrapper(lambda n, total, kw: total + float("nan") if n == 3 else total)
    with patched(td, "distillation_losses", nan_at_3):
        r = real_run("a1")
    ok, det = abort_ok(r, rule="AM-7(a)", cause="student_divergence", it=3)
    last, row = tail(r)
    check_safe("A1_nonfinite_loss_aborts_before_backward", lambda: ok and last["detail"]["loss"] == "nan"
               and last["detail"]["grad_norm"] is None and last["input_finite"] is True
               and last["teacher_finite"] is True and last["params_finite"] is True
               and last["n_val"] == 0, det)
    exc = r["exc"]
    check_safe("A1_run_aborted_pickles_with_its_record",
               lambda: isinstance(exc, td.RunAborted)
               and pickle.loads(pickle.dumps(exc)).record == exc.record, str(exc))
    check_safe("A1_train_row_nulls_and_nonfinite_map", lambda: row["loss"] is None
               and row.get("nonfinite", {}).get("loss") == "nan"
               and row["lr"] is None and row["grad_norm"] is None and row["grad_norm_student"] is None
               and "wall_clock" in row, json.dumps(row)[:300])
    cap = r["cap"]
    check_safe("A1_weights_not_stepped_at_abort",
               lambda: cap.student is not None and all(i in cap.after for i in (1, 2, 3))
               and all(torch.equal(v, cap.after[2][k]) for k, v in cap.after[3].items())
               and all(torch.equal(v, cap.after[2][k]) for k, v in params(cap.student).items())
               and not all(torch.equal(v, cap.after[1][k]) for k, v in cap.after[2].items()),
               "weights at the abort row and at the end vs after iteration 2 (which did step them)")

    def bad_grad(n, total, kw):
        if n != 3:
            return total
        z = kw["logits"].sum() * 0.0
        return total + torch.sqrt(z)                                  # forward 0, backward inf*0 = NaN
    with patched(td, "distillation_losses", distill_wrapper(bad_grad)):
        r = real_run("a2")
    ok, det = abort_ok(r, rule="AM-7(a)", cause="student_divergence", it=3)
    last, row = tail(r)
    check_safe("A2_nonfinite_grad_norm_aborts_before_step",
               lambda: ok and isinstance(last["detail"]["loss"], float)
               and last["detail"]["grad_norm"] == "nan"
               and row.get("nonfinite", {}).get("grad_norm") == "nan" and row["lr"] is None, det)
    cap2 = r["cap"]
    check_safe("A2_weights_not_stepped_at_abort",
               lambda: cap2.student is not None and 2 in cap2.after
               and all(torch.equal(v, cap2.after[2][k]) for k, v in params(cap2.student).items()),
               "weights at the end vs after iteration 2")

    real_cycle = td.cycle

    def poisoned_cycle(loader):
        for i, (img, mask) in enumerate(real_cycle(loader), start=1):
            if i == 3:
                img = img.clone()
                img[0, 0, 0, 0] = float("nan")
            yield img, mask
    with patched(td, "cycle", poisoned_cycle):
        r = real_run("a3")
    ok, det = abort_ok(r, rule="AM-7(a)", cause="input_nonfinite", it=3)
    last3 = tail(r)[0]
    check_safe("A3_nonfinite_input_is_not_a_divergence", lambda: ok and last3["input_finite"] is False, det)

    teacher = stub_teacher()
    calls = {"n": 0}

    def poison(_m, _i, out):
        calls["n"] += 1
        if calls["n"] == 3:
            out = dict(out)
            out["logits"] = out["logits"].clone()
            out["logits"][0, 0, 0, 0] = float("nan")
        return out
    teacher.teacher.register_forward_hook(poison)
    r = real_run("a4", teacher=teacher)
    ok, det = abort_ok(r, rule="AM-7(a)", cause="teacher_nonfinite", it=3)
    last4 = tail(r)[0]
    check_safe("A4_nonfinite_teacher_is_not_a_divergence", lambda: ok and last4["teacher_finite"] is False
               and last4["input_finite"] is True, det)

    with patched(td, "distillation_losses",
                 distill_wrapper(lambda n, total, kw: total + float("nan") if n == 1 else total)):
        r = real_run("a5")
    ok, det = abort_ok(r, rule="step1_checks", cause="checks_failed", it=1)
    last5, row5 = tail(r)
    check_safe("A5_nonfinite_loss_at_iter1_is_step1_checks",
               lambda: ok and [x.get("event") for x in r["rows"]] == ["train", "run_abort"]
               and last5["detail"]["loss"] == "nan" and last5["detail"]["grad_norm"] is None
               and last5["input_finite"] is True and last5["teacher_finite"] is True, det)
    check_safe("A5_train_row_nulls_and_nonfinite_map", lambda: row5["loss"] is None
               and row5.get("nonfinite", {}).get("loss") == "nan" and row5["lr"] is None
               and row5["grad_norm"] is None and row5["grad_norm_student"] is None, json.dumps(row5)[:300])

    def bad_grad_at_1(n, total, kw):
        if n != 1:
            return total
        z = kw["logits"].sum() * 0.0
        return total + torch.sqrt(z)                                  # forward 0, backward inf*0 = NaN
    with patched(td, "distillation_losses", distill_wrapper(bad_grad_at_1)):
        r = real_run("a6")
    ok, det = abort_ok(r, rule="step1_checks", cause="checks_failed", it=1)
    last6, row6 = tail(r)
    check_safe("A6_nonfinite_grad_norm_at_iter1_is_step1_checks",
               lambda: ok and [x.get("event") for x in r["rows"]] == ["train", "run_abort"]
               and last6["detail"]["grad_norm"] in ("nan", "inf", "-inf")
               and isinstance(last6["detail"]["loss"], float) and math.isfinite(last6["detail"]["loss"])
               and row6.get("nonfinite", {}).get("grad_norm") == last6["detail"]["grad_norm"]
               and row6["lr"] is None, det)

    blowup = distill_wrapper(lambda n, total, kw: total + 1000.0 if n >= 5 else total)
    with patched(td, "distillation_losses", blowup), \
            patched(td, "AM7_DIVERGENCE", {"window": 2, "factor": 5.0, "post_ramp_only": True}):
        r = real_run("b6", max_iters=8, val_interval=8)
    ok, det = abort_ok(r, rule="AM-7(b)", cause="student_divergence", it=5)
    d = tail(r)[0].get("detail", {})
    check_safe("B6_am7b_aborts_after_the_ramp", lambda: ok and d.get("ratio", 0) > 5.0
               and d.get("threshold") == 5.0 * d.get("running_min", -1)
               and d.get("window_mean", 0) > d.get("threshold", 1e9)
               and isinstance(d.get("grad_norm"), float), det)

    def nan_validate(*a, **k):
        all_miou, disease, cm, nvb = REAL_VALIDATE(*a, **k)
        return float("nan"), disease, cm, nvb
    with patched(td, "validate", nan_validate):
        r = real_run("v1", max_iters=4, val_interval=2)
    ok, det = abort_ok(r, rule="val_nonfinite", cause="val_nonfinite", it=2)
    vlast, vrow = tail(r)
    check_safe("V1_nonfinite_val_miou_aborts", lambda: ok and vrow.get("event") == "val"
               and vrow["all_class_miou"] is None
               and vrow.get("nonfinite", {}).get("all_class_miou") == "nan" and vlast["n_val"] == 1, det)

    with patched(td, "distillation_losses", distill_wrapper(lambda n, t, kw: t + 1.0 if n == 1 else t)):
        r = real_run("s1")
    ok, det = abort_ok(r, rule="step1_checks", cause="checks_failed", it=1)
    check("S1_failed_step1_check_aborts_after_iteration_1", ok
          and [x.get("event") for x in r["rows"]] == ["train", "run_abort"], det)

    nan_at_3_again = distill_wrapper(lambda n, t, kw: t + float("nan") if n == 3 else t)
    with patched(td, "distillation_losses", nan_at_3_again):
        r = real_run("d1", mode="dry", max_iters=4, val_interval=4)
    rs = r["rows"]
    check_safe("D1_dry_mode_does_not_abort", lambda: r["exc"] is None and rs[-1].get("event") == "run_end"
               and not any(x.get("event") == "run_abort" for x in rs)
               and any("nonfinite" in x for x in rs if x.get("event") == "train"),
               f"last={rs[-1] if rs else None}")


REAL_VALIDATE = td.validate


def main() -> int:
    print("=" * 78)
    print("AM-7 DIVERGENCE SMOKE (L-KD-HARDEN item 3; L-AM7; AM-7a record) - CPU, synthetic")
    print(f"torch {torch.__version__}")
    print("=" * 78)
    test_monitor_units()
    root = Path(tempfile.mkdtemp(prefix="kdh_am7_data_"))
    make_synthetic_dataset(root / "data", n_train=4, n_val=2)
    saved = (cdata.DATA["root"], dict(cdata.SPLIT_SIZES))
    cdata.DATA["root"] = str(root / "data")
    cdata.SPLIT_SIZES.update({"train": 4, "val": 2})
    try:
        test_real_mode()
    finally:
        cdata.DATA["root"] = saved[0]
        cdata.SPLIT_SIZES.clear()
        cdata.SPLIT_SIZES.update(saved[1])
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
