#!/usr/bin/env python3
"""Step-1 checks and native teacher grids of the distillation trainer (lane L-KD-HARDEN items 1 and 8).

CPU, a synthetic 512x512 TRAIN/VAL set (4 + 2 images, batch 2, so the first-epoch ramp spans 2
iterations), the explicit MockTeacher unless stated; no dataset, checkpoint or GPU. Mutants are made at
SOURCE level: the production train_distill.py text is edited in memory (each edit must match exactly
once) and compiled into a separate module, so the production module is never modified.

  P    production dry run: RESULT PASS and every hard check PASS; per step exactly one CE and one Dice
       evaluation (call counters on the two loss modules)
  G1   G-F1 mutant `loss = 0.5 * sup + distill` -> sup_added_unscaled FAIL, RESULT FAIL
  G2   G-F1 mutant sup scaled by 0.0 -> sup_positive_finite FAIL, RESULT FAIL
  G0   the 44c05dc second criterion call re-inserted -> one extra CE and Dice evaluation at step 1
       (proves the counter sees it)
  I1   F4/N1 mutant: the teacher is handed `model_input.clone()` -> teacher_same_augmented_input FAIL
       (the 44c05dc check `model_input is img` passed this mutant)
  D1   teacher Stage-3 on a 64x64 grid, stage E2 (no feature term) -> teacher_feat_shape FAIL
  D2   item 8: teacher logits on a 32x32 grid -> distillation_losses raises (never resampled), in an E2
       run and in a direct E2 call; a teacher Stage-3 grid different from C5 raises in a direct call for
       a feature-term stage (F)
  F0   item 1b: a REAL-mode production run (3 iterations, the harness's stub teacher) is not stopped
  F1   item 1b: the I1 mutant in REAL mode stops right after iteration 1: its train row, then a run_abort
       record {rule step1_checks, cause checks_failed}, no run_end, RunAborted raised
GPUs are hidden (CUDA_VISIBLE_DEVICES="" before torch loads). On a GPU host the earlier CPU runs touch
CUDA (run_meta's gpu_name calls torch.cuda.get_device_name, which initialises it; the loaders pin
memory), and run()'s CUDA-order gate would then refuse F0/F1's real runs in this process.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only smoke: hide GPUs before torch loads (above)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import configs.data as cdata  # noqa: E402
import src.training.losses as losses  # noqa: E402
import src.training.train_distill as td  # noqa: E402
from configs.e1_student import E1_STUDENT  # noqa: E402
from scripts.invariance_harness import build_stub_segnext, make_synthetic_dataset  # noqa: E402
from src.distill import FrozenTeacher, MockTeacher  # noqa: E402
from src.distill.segnext_teacher import SegNeXtTeacherAdapter  # noqa: E402

NC = 116
TD_SOURCE = Path(td.__file__).read_text(encoding="utf-8")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def mutant(edits: list[tuple[str, str]], name: str) -> types.ModuleType:
    """train_distill compiled from its source with `edits` applied; each `old` must occur once."""
    src = TD_SOURCE
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"mutant {name}: {old!r} occurs {n} times in train_distill.py")
        src = src.replace(old, new)
    mod = types.ModuleType(f"td_mutant_{name}")
    mod.__file__ = td.__file__
    exec(compile(src, td.__file__, "exec"), mod.__dict__)
    return mod


class Counter:
    """Counts forward calls of the CE and Dice modules (class-level wrappers, restored on exit)."""

    def __init__(self):
        self.ce = self.dice = 0

    def __enter__(self):
        self._ce, self._dice = losses.WeightedCrossEntropyLoss.forward, losses.SoftDiceLoss.forward
        real_ce, real_dice = self._ce, self._dice

        def ce(mod, *a, **k):
            self.ce += 1
            return real_ce(mod, *a, **k)

        def dice(mod, *a, **k):
            self.dice += 1
            return real_dice(mod, *a, **k)

        losses.WeightedCrossEntropyLoss.forward, losses.SoftDiceLoss.forward = ce, dice
        return self

    def __exit__(self, *exc):
        losses.WeightedCrossEntropyLoss.forward, losses.SoftDiceLoss.forward = self._ce, self._dice
        return False


def dry_run(module, *, stage="e3", teacher=None, max_iters=2) -> tuple[int, str]:
    """A dry run of `module.run` on the synthetic set; returns (rc, stdout). Checkpoints go to a temp dir."""
    teacher = teacher if teacher is not None else FrozenTeacher(MockTeacher(NC))
    st = module.resolve_stage(stage)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = module.run(stage=st, mode="dry", device="cpu", pretrained=False, teacher=teacher,
                        lambda_logit=1.0 if st["logit_kd"] else None, batch_size=2, max_iters=max_iters,
                        val_interval=max_iters, max_val_batches=None, num_workers=0,
                        ckpt_dir_arg=tempfile.mkdtemp(prefix=f"kdh_step_{stage}_alpha50_"),
                        grad_clip_norm=None, log_every=1, seed=42)
    return rc, out.getvalue()


def check_line(stdout: str, name: str) -> str | None:
    for ln in stdout.splitlines():
        if ln.strip().startswith(name + " ") or ln.strip().startswith(name + ":"):
            return "PASS" if ln.rstrip().endswith("PASS") else "FAIL"
    return None


def result_line(stdout: str) -> str:
    return next((ln.strip() for ln in reversed(stdout.splitlines()) if ln.strip().startswith("RESULT")), "")


# ----------------------------------------------------------------------------------- tests
SUP_LINE = "        sup = ce + dice"                  # the supervised composition in run() (item 4a)


def test_production() -> None:
    with Counter() as c:
        rc, out = dry_run(td, max_iters=2)
    hard = ["logits_shape", "c5_channels", "head_logits_64x64", "c5_32x32", "teacher_logits_shape",
            "teacher_feat_shape", "teacher_same_augmented_input", "distill_zero_at_step1",
            "sup_added_unscaled", "sup_positive_finite"]
    states = {k: check_line(out, k) for k in hard}
    check("P_production_passes", rc == 0 and result_line(out).startswith("RESULT: PASS")
          and all(v == "PASS" for v in states.values()), f"rc={rc} {states} {result_line(out)}")
    check("P_one_ce_and_one_dice_per_step", c.ce == 2 and c.dice == 2, f"ce={c.ce} dice={c.dice} over 2 steps")
    check("P_supervised_never_ramped_removed", "supervised_never_ramped" not in TD_SOURCE)


def test_g_f1_mutants() -> None:
    m = mutant([("        loss = sup + distill\n", "        loss = 0.5 * sup + distill\n")], "half_sup")
    rc, out = dry_run(m)
    check("G1_half_sup_fails_sup_added_unscaled", rc == 1 and check_line(out, "sup_added_unscaled") == "FAIL"
          and result_line(out).startswith("RESULT: FAIL"), f"rc={rc} {result_line(out)}")
    m = mutant([(SUP_LINE + "\n", SUP_LINE.replace("ce + dice", "0.0 * (ce + dice)") + "\n")], "zero_sup")
    rc, out = dry_run(m)
    check("G2_zero_sup_fails_sup_positive_finite", rc == 1 and check_line(out, "sup_positive_finite") == "FAIL"
          and result_line(out).startswith("RESULT: FAIL"), f"rc={rc} {result_line(out)}")
    m = mutant([("        loss = sup + distill\n",
                 "        loss = sup + distill\n"
                 "        if it == 1:\n"
                 "            _ = bool(torch.equal(sup, criterion(logits, mask)))\n")], "old_second_call")
    with Counter() as c:
        dry_run(m, max_iters=2)
    check("G0_reinserted_44c05dc_call_is_counted", c.ce == 3 and c.dice == 3, f"ce={c.ce} dice={c.dice}")


def test_input_identity() -> None:
    m = mutant([("teacher_out = teacher(model_input)", "teacher_out = teacher(model_input.clone())")], "clone")
    rc, out = dry_run(m)
    check("I1_teacher_given_a_copy_fails", rc == 1
          and check_line(out, "teacher_same_augmented_input") == "FAIL", f"rc={rc} {result_line(out)}")


def test_grids() -> None:
    rc, out = dry_run(td, stage="e2", teacher=FrozenTeacher(MockTeacher(NC, feat_stride=8)))
    check("D1_feat_grid_mismatch_fails_step1_check", rc == 1 and check_line(out, "teacher_feat_shape") == "FAIL",
          f"rc={rc} {result_line(out)}")
    try:
        dry_run(td, stage="e2", teacher=FrozenTeacher(MockTeacher(NC, logits_stride=16)))
        check("D2_logit_grid_mismatch_raises_in_run", False, "no exception")
    except RuntimeError as e:
        check("D2_logit_grid_mismatch_raises_in_run", "never resampled" in str(e), str(e)[:160])
    kw = dict(logits=torch.randn(1, NC, 512, 512), head_logits=torch.randn(1, NC, 64, 64),
              c5=torch.randn(1, 160, 32, 32), mask=torch.zeros(1, 512, 512, dtype=torch.long))
    t = FrozenTeacher(MockTeacher(NC, logits_stride=16))(torch.randn(1, 3, 512, 512))
    try:
        td.distillation_losses(stage=td.resolve_stage("e2"), projection=None, teacher_out=t,
                               lambda_logit=1.0, **kw)
        check("D2_logit_grid_mismatch_raises_in_direct_call", False, "no exception")
    except RuntimeError as e:
        check("D2_logit_grid_mismatch_raises_in_direct_call", "never resampled" in str(e), str(e)[:160])
    t = FrozenTeacher(MockTeacher(NC, feat_stride=8))(torch.randn(1, 3, 512, 512))
    try:
        td.distillation_losses(stage=td.resolve_stage("f"), teacher_out=t, lambda_logit=None,
                               projection=td.build_cwd_projection_for(td.resolve_stage("f")), **kw)
        check("D2_feat_grid_mismatch_raises_for_cwd_feat", False, "no exception")
    except RuntimeError as e:
        check("D2_feat_grid_mismatch_raises_for_cwd_feat", "never resampled" in str(e), str(e)[:160])


def real_run(module, tag: str, max_iters: int = 3) -> tuple[object, list[dict]]:
    """A REAL-mode run of `module.run` on the synthetic set (the invariance harness's stub teacher, which
    has the M4-KD NMF stream a real run requires; E1_STUDENT["iterations"] patched to `max_iters` so the
    schedule gate admits it on CPU). Returns (the RunAborted record, or the return code; the rows)."""
    ckpt = Path(tempfile.mkdtemp(prefix=f"kdh_step_real_{tag}_")) / "run"
    saved = E1_STUDENT["iterations"]
    E1_STUDENT["iterations"] = max_iters
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            outcome = module.run(stage=module.resolve_stage("e2"), mode="real", device="cpu",
                                 pretrained=False, teacher=FrozenTeacher(SegNeXtTeacherAdapter(
                                     build_stub_segnext())), lambda_logit=1.0, batch_size=2,
                                 max_iters=max_iters, val_interval=max_iters, max_val_batches=None,
                                 num_workers=0, ckpt_dir_arg=str(ckpt), grad_clip_norm=None, log_every=1,
                                 seed=42)
    except module.RunAborted as e:
        outcome = e.record
    finally:
        E1_STUDENT["iterations"] = saved
    tel = ckpt / "e2_telemetry.jsonl"
    rows = ([json.loads(ln) for ln in tel.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if tel.is_file() else [])             # none when run() refused first: a FAIL, not a traceback
    return outcome, rows


def test_real_fail_fast() -> None:
    """Item 1b: in a real run, a failed step-1 check stops the run right after iteration 1."""
    rc, rows = real_run(td, "production")
    check("F0_real_production_run_is_not_stopped", rc == 0 and bool(rows)
          and rows[-1].get("event") == "run_end" and not any(r.get("event") == "run_abort" for r in rows),
          f"rc={rc} last={rows[-1] if rows else None}")
    m = mutant([("teacher_out = teacher(model_input)", "teacher_out = teacher(model_input.clone())")],
               "clone_real")
    rec, rows = real_run(m, "clone")
    events = [(r.get("event"), r.get("iter")) for r in rows]
    check("F1_failed_step1_check_aborts_after_iteration_1", isinstance(rec, dict)
          and rec.get("rule") == "step1_checks" and rec.get("cause") == "checks_failed"
          and rec.get("iter") == 1 and events == [("train", 1), ("run_abort", 1)], f"{rec} {events}")


def main() -> int:
    print("=" * 78)
    print("DISTILL STEP-1 CHECKS SMOKE (L-KD-HARDEN items 1, 8) - CPU, synthetic 512x512 set")
    print(f"torch {torch.__version__}")
    print("=" * 78)
    root = Path(tempfile.mkdtemp(prefix="kdh_stepchecks_"))
    make_synthetic_dataset(root / "data", n_train=4, n_val=2)
    saved = (cdata.DATA["root"], dict(cdata.SPLIT_SIZES))
    cdata.DATA["root"] = str(root / "data")
    cdata.SPLIT_SIZES.update({"train": 4, "val": 2})
    try:
        for fn in (test_production, test_g_f1_mutants, test_input_identity, test_grids, test_real_fail_fast):
            fn()
    finally:
        cdata.DATA["root"] = saved[0]
        cdata.SPLIT_SIZES.clear()
        cdata.SPLIT_SIZES.update(saved[1])
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
