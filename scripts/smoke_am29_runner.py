#!/usr/bin/env python3
"""L-AM29 P1 smoke: src/quant/am29.py on synthetic data, on the CPU. No dataset, no checkpoint of record, no download.

Sections (all by default):
  term      the T arms' logit-KD term: the teacher's sha256 is checked; with the quantizers off and student = teacher the
            step-0 term is 0 within 1e-6; T^2 is in the term at T = 4 and T is train_distill's constant
            (L-AM29-FIX, AUDIT-LAM29-1 F-07: on T1 and T3 runs every step's loss is sup + factor x 0.5 x logit_kd
            within 2.4e-7)
  weight0   a T arm at factor 0 against src/quant/qat.py run_qat: every train row and every epoch state bitwise equal
  lr        the rate: in every parameter group, the scheduler's base and its cosine, and run_meta; at 3e-4 the LR loop
            equals run_qat bitwise
  a3        A3 at seed 42's settings against src/quant/qat_kd.py run_qat_kd: every train row (all three terms, the
            ramp, both norms) and every epoch state bitwise equal
  allow     the allow-list refuses every other arm, seed, factor, rate and teacher or projection flag
  pairing   the A3 arms refuse a parent and projection.pt whose sha256 do not belong together
  gates     the launch gates' order on the CPU, the parent of record by sha256, the T parent binding
  meta      run_meta: the arm, factor, rate, seed, the parent's and the teacher's sha256, T, the code with the new files

    python -B scripts/smoke_am29_runner.py [--sections term,weight0,...]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import smoke_am29_fixtures as FX  # noqa: E402

TMP = FX.tmpdir("smoke_am29_runner_")
os.environ["PLANTSEG_DATA_ROOT"] = str(TMP / "absent_data_root")      # before configs.data is imported

import torch  # noqa: E402

from src.distill import teacher as TT  # noqa: E402
from src.distill.features import StudentTaps  # noqa: E402
from src.quant import am29 as M  # noqa: E402
from src.quant import am29_registry as R  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_kd as KD  # noqa: E402
from src.quant.prepare import disable_observers, prepare_qat_model  # noqa: E402
from src.training import train_distill as TD  # noqa: E402
from src.training.losses import downsample_validity, logit_kd_kl  # noqa: E402

SECTIONS = ("term", "weight0", "lr", "a3", "allow", "pairing", "gates", "meta")
SRC = {"sha256": "0" * 64, "path": None}
TRAIN_KEYS = ("loss", "ce", "dice", "grad_norm", "lr", "batch_sha256", "clipped")
A3_KEYS = ("loss", "ce", "dice", "sup", "logit_kd", "cwd_feat", "cwd_logit", "ramp", "grad_norm", "grad_norm_student",
           "grad_norm_projection", "lr", "batch_sha256", "clipped")
LOSS_IDENTITY_TOL = 2.4e-7       # AUDIT-LAM29-1 section 8 (F-07): loss = sup + factor x 0.5 x logit_kd within this
check = FX.Checks("L-AM29 P1 (src/quant/am29.py)")
R.register()


def refused(fn, code: str) -> tuple[bool, str]:
    try:
        fn()
    except Q.QATRefused as e:
        return e.code == code, f"[{e.code}] {str(e)[:160]}"
    except Exception as e:                                   # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return False, "not refused"


def run_t(out: str, *, factor: float, lr: float = 3e-4, max_steps=4, arm: str = "T1", hooks=None):
    m = FX.student()
    teacher = M.SelfTeacher(m, checkpoint_sha256="1" * 64)
    return M.run_am29_qat(arm=arm, mode="smoke", model=m, source_meta=SRC, out_dir=TMP / out, seed=42, clip_norm=5.0,
                          clip_source="clip_selection", device="cpu", num_workers=0, lr=lr, factor=factor,
                          e3_lambda=0.5, teacher=teacher, loaders=FX.loaders(), max_steps=max_steps,
                          max_val_batches=1, hooks=hooks, log=FX.quiet)


def run_e6(out: str, max_steps=4):
    return Q.run_qat(stage="e6", mode="smoke", model=FX.student(), source_meta=SRC, out_dir=TMP / out, seed=42,
                     clip_norm=5.0, clip_source="clip_selection", device="cpu", num_workers=0, loaders=FX.loaders(),
                     max_steps=max_steps, max_val_batches=1, log=FX.quiet)


# ------------------------------------------------------------------ term
def sec_term() -> None:
    torch.manual_seed(5)
    m = FX.student()
    teacher = M.SelfTeacher(m, checkpoint_sha256="1" * 64)
    prepared = prepare_qat_model(m)
    prepared.apply(torch.ao.quantization.disable_fake_quant)
    disable_observers(prepared)
    prepared.eval()
    g = torch.Generator().manual_seed(9)
    x = torch.randn(2, 3, 64, 64, generator=g)
    mask = torch.randint(0, FX.NC, (2, 64, 64), generator=g)
    mask[:, :8] = 255
    with StudentTaps(prepared) as taps:
        logits = prepared(x)
        c5, head = taps.require()
    tout = teacher(x)
    total, parts = M.t_logit_kd(logits=logits, head_logits=head, c5=c5, mask=mask, teacher_out=tout, weight=0.25)
    check("term_zero_student_equals_teacher", abs(parts["logit_kd"]) <= 1e-6 and abs(float(total)) <= 1e-6,
          f"step-0 logit-KD {parts['logit_kd']!r} (weighted {float(total)!r}); quantizers off, student = teacher")
    check("term_teacher_grid_os8", tuple(tout.logits.shape) == tuple(head.shape) == (2, FX.NC, 8, 8)
          and tout.feat_s16 is None, f"teacher {tuple(tout.logits.shape)}, head {tuple(head.shape)}, no feature")
    check("term_teacher_frozen", not teacher.trainable_parameters() and not teacher.student.training
          and all(not p.requires_grad for p in teacher.parameters()), "eval mode, requires_grad False")
    # T^2 at T = 4, and T read from train_distill's constant
    s = torch.randn(2, FX.NC, 8, 8, generator=g)
    t = torch.randn(2, FX.NC, 8, 8, generator=g)
    valid = downsample_validity(mask, (8, 8))
    T = 4.0
    p_t = torch.softmax(t / T, 1)
    manual = ((p_t * (torch.log_softmax(t / T, 1) - torch.log_softmax(s / T, 1))).sum(1) * valid.float()).sum() \
        / valid.float().sum()
    term = logit_kd_kl(s, t, valid, T=T)
    check("term_T2_scaling_T4", math.isclose(float(term), 16.0 * float(manual), rel_tol=1e-5),
          f"logit_kd_kl {float(term):.7f} = T^2 (16) x plain KL {float(manual):.7f}")
    check("term_T_is_E3s", M.T_LOGIT == TD.T_LOGIT == 4 and M.E2_SWITCHES == TD.resolve_stage("e2"),
          f"am29.T_LOGIT {M.T_LOGIT}, train_distill.T_LOGIT {TD.T_LOGIT}; stage e2 switches {M.E2_SWITCHES}")
    tout2 = TT.TeacherOutput(logits=t, feat_s16=None)
    _, a = M.t_logit_kd(logits=logits, head_logits=s, c5=c5, mask=mask, teacher_out=tout2, weight=1.0)
    saved = TD.T_LOGIT
    try:
        TD.T_LOGIT = 2.0
        _, b = M.t_logit_kd(logits=logits, head_logits=s, c5=c5, mask=mask, teacher_out=tout2, weight=1.0)
    finally:
        TD.T_LOGIT = saved
    want_b = float(logit_kd_kl(s, t, downsample_validity(mask, (8, 8)), T=2.0))
    check("term_T_read_from_train_distill", math.isclose(a["logit_kd"], float(term), rel_tol=1e-6)
          and math.isclose(b["logit_kd"], want_b, rel_tol=1e-6),
          f"T=4: {a['logit_kd']:.6f}; with train_distill.T_LOGIT set to 2 the term follows it: {b['logit_kd']:.6f}")
    # the teacher's sha256 is checked
    parent = {"checkpoint_sha256": M.PARENTS_OF_RECORD["E3-R2-s42"]["sha256"]}
    good = M.PARENTS_OF_RECORD["E3-R2-s42"]["sha256"]
    check("term_teacher_sha256_of_record", M.t_teacher_error("T1", good, parent) is None, "T1 teacher = da70d82f...")
    e1 = M.t_teacher_error("T1", "ab" * 32, parent)
    e2 = M.t_teacher_error("T1", good, {"checkpoint_sha256": "cd" * 32})
    e3 = M.t_teacher_error("T2", "zz", parent)
    check("term_teacher_sha256_refused", all((e1, e2, e3)), f"wrong value: {e1[:60]}; wrong file: {e2[:60]}; form: {e3}")
    term_loss_identity()                                     # AUDIT-LAM29-1 F-07


def term_loss_identity(tag: str = "") -> dict:
    """AUDIT-LAM29-1 F-07 (its section 8): on a T1 run (factor 0.5) and a T3 run (factor 1.0), every train row's loss is
    sup + factor x 0.5 x logit_kd within 2.4e-7, so the term's weight is 0.25 and 0.5 at every step. The weight comes
    from the arm's factor and E3's lambda 0.5, never from the row's own kd_weight, so that a ramp or a weight without
    lambda fails the check; the term must be non-zero at every step, so that the check is not vacuous."""
    ok_by_arm = {}
    for arm, factor in (("T1", 0.5), ("T3", 1.0)):
        out = f"loss_identity_{arm.lower()}{tag}"
        run_t(out, factor=factor, arm=arm)
        rows = FX.rows(TMP / out, "train")
        weight = factor * 0.5
        diffs = [abs(r["loss"] - (r["sup"] + weight * r["logit_kd"])) for r in rows]
        ok = (len(rows) == 4 and all(d <= LOSS_IDENTITY_TOL for d in diffs)
              and all(abs(r["logit_kd"]) > 1e-3 for r in rows))
        ok_by_arm[arm] = check(f"term_loss_identity_{arm}", ok,
                               f"weight {weight} (factor {factor} x lambda 0.5), {len(rows)} steps: |loss - (sup + "
                               f"weight x logit_kd)| {[float(f'{d:.3g}') for d in diffs]} <= {LOSS_IDENTITY_TOL}; "
                               f"logit_kd {[round(r['logit_kd'], 4) for r in rows]}")
    return ok_by_arm


# ------------------------------------------------------------------ weight 0
def sec_weight0() -> None:
    run_e6("w0_e6")
    run_t("w0_t1", factor=0.0)
    ra, rb = FX.rows(TMP / "w0_e6", "train"), FX.rows(TMP / "w0_t1", "train")
    same = len(ra) == len(rb) == 4 and all(all(x[k] == y[k] for k in TRAIN_KEYS) for x, y in zip(ra, rb))
    check("weight0_rows_equal_run_qat", same, f"4 steps: loss {[r['loss'] for r in rb]}; logit_kd computed "
                                              f"{[round(r['logit_kd'], 4) for r in rb]} at weight {rb[0]['kd_weight']}")
    ea, eb = FX.rows(TMP / "w0_e6", "epoch_end"), FX.rows(TMP / "w0_t1", "epoch_end")
    check("weight0_states_equal_run_qat", len(ea) == len(eb) == 2 and all(x["state_sha256"] == y["state_sha256"]
                                                                        for x, y in zip(ea, eb)),
          f"epoch states {[r['state_sha256'][:12] for r in eb]}")
    s1 = FX.rows(TMP / "w0_t1", "step1_checks")
    check("weight0_step1_checks", len(s1) == 1 and s1[0]["passed"], f"{s1[0]['checks'] if s1 else None}")


# ------------------------------------------------------------------ lr
class Grab(Q.QATHooks):
    def on_start(self, prepared, optimizer, scheduler, run_meta):
        self.groups = [(g["lr"], g["initial_lr"]) for g in optimizer.param_groups]
        self.base = list(scheduler.base_lrs)
        self.t_max = scheduler.T_max
        self.meta = run_meta


def sec_lr() -> None:
    for out, lr in (("lr_am29", M.LR_AM29),):
        h = Grab()
        M.run_am29_qat(arm="LR-s42", mode="smoke", model=FX.student(), source_meta=SRC, out_dir=TMP / out, seed=42,
                       clip_norm=5.0, clip_source="clip_selection", device="cpu", num_workers=0, lr=lr,
                       loaders=FX.loaders(), max_steps=6, max_val_batches=1, hooks=h, log=FX.quiet)
        check("lr_every_param_group", h.groups == [(lr, lr)], f"groups (lr, initial_lr) {h.groups}")
        check("lr_scheduler_base", h.base == [lr] and h.t_max == 30, f"base_lrs {h.base}, T_max {h.t_max}")
        rows = FX.rows(TMP / out, "train")
        want = [lr * (1 + math.cos(math.pi * k / h.t_max)) / 2 for k in range(len(rows))]
        got = [r["lr"] for r in rows]
        check("lr_cosine", all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15) for a, b in zip(got, want)),
              f"rows {got[:3]}... vs lr(1+cos(pi k/T_max))/2 {want[:3]}...")
        meta = h.meta
        check("lr_in_run_meta", meta["lr"] == lr and meta["am29"]["lr"] == lr and meta["optimizer"]["initial_lr"] == lr
              and meta["scheduler"]["base_lrs"] == [lr], f"run_meta lr {meta['lr']}, am29.lr {meta['am29']['lr']}")
        check("lr_optimizer_error_rule", M.optimizer_error(meta["optimizer"], lr) is None
              and M.optimizer_error(meta["optimizer"], M.LR_OF_RECORD) is not None
              and Q.optimizer_of_record_error(meta["optimizer"]) is not None,
              "6.25e-5 passes the AM-29 rule and fails the record's 3e-4 rule")
    run_e6("lr_e6")
    M.run_am29_qat(arm="LR-s42", mode="smoke", model=FX.student(), source_meta=SRC, out_dir=TMP / "lr_record", seed=42,
                   clip_norm=5.0, clip_source="clip_selection", device="cpu", num_workers=0, lr=M.LR_OF_RECORD,
                   loaders=FX.loaders(), max_steps=4, max_val_batches=1, log=FX.quiet)
    ra, rb = FX.rows(TMP / "lr_e6", "train"), FX.rows(TMP / "lr_record", "train")
    ea, eb = FX.rows(TMP / "lr_e6", "epoch_end"), FX.rows(TMP / "lr_record", "epoch_end")
    check("lr_record_loop_equals_run_qat", all(all(x[k] == y[k] for k in TRAIN_KEYS) for x, y in zip(ra, rb))
          and all(x["state_sha256"] == y["state_sha256"] for x, y in zip(ea, eb)) and len(rb) == 4,
          "the T/LR loop at 3e-4 without a teacher is run_qat, bitwise")
    meta_a, meta_b = FX.rows(TMP / "lr_e6", "run_meta")[0], FX.rows(TMP / "lr_record", "run_meta")[0]
    differ = sorted(k for k in set(meta_a) | set(meta_b) if meta_a.get(k) != meta_b.get(k))
    check("lr_record_run_meta_keys", set(differ) <= {"wall_clock", "schema", "lane", "stage", "run_id", "out_dir",
                                                     "scheduler", "git_head", "code_clean_at_head", "code_files_sha256",
                                                     "am29"},
          f"the run_meta rows differ only in labels, identity and the am29 block: {differ}")


# ------------------------------------------------------------------ a3
def sec_a3() -> None:
    e3 = {"e3_weights": {"logit_kd": 0.5, "cwd_feat": 25.0, "cwd_logit": 3.0}, "grids": {}}
    w = KD.e6kd_weights(e3["e3_weights"])
    KD.run_qat_kd(mode="smoke", model=FX.student(), teacher=FX.stub_teacher(), projection=FX.projection(), weights=w,
                  e3=e3, source_meta=SRC, out_dir=TMP / "a3_kd", seed=42, clip_norm=5.0, clip_source="clip_selection",
                  device="cpu", num_workers=0, loaders=FX.loaders(), max_steps=4, max_val_batches=1, log=FX.quiet)
    h = Grab()
    M.run_am29_a3(arm="A3-s43", mode="smoke", model=FX.student(), teacher=FX.stub_teacher(), projection=FX.projection(),
                  weights=w, e3=e3, source_meta=SRC, out_dir=TMP / "a3_am29", seed=42, clip_norm=5.0,
                  clip_source="clip_selection", device="cpu", num_workers=0, loaders=FX.loaders(), max_steps=4,
                  max_val_batches=1, hooks=h, log=FX.quiet)
    ra, rb = FX.rows(TMP / "a3_kd", "train"), FX.rows(TMP / "a3_am29", "train")
    eq = [all(x[k] == y[k] for k in A3_KEYS) for x, y in zip(ra, rb)]
    check("a3_loss_equals_qat_kd_first_batch", bool(eq) and eq[0], f"step 1: loss {rb[0]['loss']!r} == {ra[0]['loss']!r}")
    check("a3_rows_equal_qat_kd", len(eq) == 4 and all(eq) and any(r["cwd_feat"] and r["ramp"] > 0 for r in rb),
          f"4 steps, the terms on from step 2: {[(round(r['loss'], 4), r['ramp']) for r in rb]}")
    ea, eb = FX.rows(TMP / "a3_kd", "epoch_end"), FX.rows(TMP / "a3_am29", "epoch_end")
    check("a3_states_equal_qat_kd", all(x["state_sha256"] == y["state_sha256"] and x["projection_sha256"]
                                        == y["projection_sha256"] for x, y in zip(ea, eb)) and len(eb) == 2,
          "student state and projection, both epoch ends")
    check("a3_rate_every_group", h.groups == [(3e-4, 3e-4), (3e-4, 3e-4)] and h.base == [3e-4, 3e-4],
          f"student + projection groups {h.groups}, scheduler {h.base}")
    check("a3_run_meta", h.meta["stage"] == "AM29-A3-S43" and h.meta["am29"]["arm"] == "A3-s43"
          and h.meta["weights"] == w and h.meta["T_logit"] == TD.T_LOGIT == 4, f"am29 {h.meta['am29']['replaces'][:60]}")
    ok, d = refused(lambda: M.run_am29_a3(arm="A3-s43", mode="real", model=FX.student(), teacher=FX.stub_teacher(),
                                          projection=FX.projection(), weights=w, e3=e3, source_meta=SRC,
                                          out_dir=TMP / "a3_real", seed=42, clip_norm=5.0, clip_source="clip_selection",
                                          device="cpu", num_workers=0), "arm_seed")
    check("a3_real_seed_42_refused", ok, d)


# ------------------------------------------------------------------ allow
def sec_allow() -> None:
    cases = [("T5", 42, 0.5, 3e-4, "arm_unknown"), ("T1", 43, 0.5, 3e-4, "arm_seed"), ("T1", 42, 1.0, 3e-4, "arm_factor"),
             ("T1", 42, 0.5, 6.25e-5, "arm_lr"), ("T2", 42, 0.5, 3e-4, "arm_lr"), ("T3", 42, 0.5, 3e-4, "arm_factor"),
             ("T4", 42, 1.0, 1e-4, "arm_lr"), ("A3-s43", 42, 0.5, 3e-4, "arm_seed"), ("A3-s43", 43, 1.0, 3e-4, "arm_factor"),
             ("A3-s44", 44, 0.5, 6.25e-5, "arm_lr"), ("LR-s42", 42, 0.5, 6.25e-5, "arm_factor"),
             ("LR-s43", 43, None, 3e-4, "arm_lr"), ("LR-s44", 42, None, 6.25e-5, "arm_seed"),
             ("A3-s42", 42, 0.5, 3e-4, "arm_unknown"), ("E6", 42, None, 3e-4, "arm_unknown")]
    bad = []
    for arm, seed, fac, lr, code in cases:
        err = M.arm_values_error(arm, seed=seed, factor=fac, lr=lr)
        if err is None or err[0] != code:
            bad.append((arm, seed, fac, lr, err))
    check("allow_refuses_other_values", not bad, f"{len(cases)} cases refused with their code; misses {bad}")
    good = [(a, s["seed"], s["factor"], s["lr"]) for a, s in M.ARMS.items()]
    check("allow_accepts_the_nine", all(M.arm_values_error(a, seed=s, factor=f, lr=lr) is None for a, s, f, lr in good)
          and len(good) == 9, f"{[g[0] for g in good]}")
    check("allow_values_of_the_spec", {a: (s["factor"], s["lr"]) for a, s in M.ARMS.items() if a.startswith("T")}
          == {"T1": (0.5, 3e-4), "T2": (0.5, 6.25e-5), "T3": (1.0, 3e-4), "T4": (1.0, 6.25e-5)}
          and all(M.ARMS[a]["lr"] == 6.25e-5 and M.ARMS[a]["factor"] is None for a in ("LR-s42", "LR-s43", "LR-s44"))
          and all(M.ARMS[a]["lr"] == 3e-4 and M.ARMS[a]["factor"] == 0.5 for a in ("A3-s43", "A3-s44")),
          "T1-T4 (factor, lr), A3 (0.5, 3e-4), LR (none, 6.25e-5)")
    # the CLI refuses before anything is read (the allow-list is gate 2)
    import contextlib
    import io
    base = ["--real-run", "--confirm-real-run", "--expect-head", "a" * 40]
    cli = [(["--arm", "T1", "--seed", "44", "--factor", "0.5", "--lr", "3e-4"], "arm_seed"),
           (["--arm", "LR-s43", "--seed", "43", "--lr", "3e-4"], "arm_lr"),
           (["--arm", "LR-s42", "--seed", "42", "--factor", "0.5", "--lr", "6.25e-5"], "arm_factor"),
           (["--arm", "LR-s42", "--seed", "42", "--lr", "6.25e-5", "--teacher-sha256", "a" * 64], "flag_not_applicable"),
           (["--arm", "T1", "--seed", "42", "--factor", "0.5", "--lr", "3e-4", "--teacher-ckpt", "/x/iter_24000.pth"],
            "flag_not_applicable"),
           (["--arm", "T1", "--seed", "42", "--factor", "0.5", "--lr", "3e-4"], "flag_required"),
           (["--arm", "A3-s43", "--seed", "43", "--factor", "0.5", "--lr", "3e-4", "--teacher-ckpt", "/x/iter_24000.pth",
             "--teacher-ckpt-sha256", "a" * 64], "flag_required"),
           (["--arm", "A3-s44", "--seed", "44", "--factor", "0.5", "--lr", "3e-4", "--teacher-sha256", "a" * 64],
            "flag_not_applicable")]
    miss = []
    for argv, code in cli:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(argv + base)
        if rc != Q.EXIT_REFUSED or f"[{code}]" not in buf.getvalue():
            miss.append((argv[:2], code, rc, buf.getvalue()[-120:]))
    check("allow_cli_refusals", not miss, f"{len(cli)} launches refused with exit 2 and their code; misses {miss}")
    ok, d = refused(lambda: M.run_am29_qat(arm="T1", mode="real", model=FX.student(), source_meta=SRC,
                                           out_dir=TMP / "real_t1", seed=42, clip_norm=5.0, clip_source="clip_selection",
                                           device="cpu", num_workers=1, lr=6.25e-5, factor=0.5, e3_lambda=0.5,
                                           teacher=M.SelfTeacher(FX.student())), "arm_lr")
    check("allow_loop_real_refuses_rate", ok, d)
    ok, d = refused(lambda: M.run_am29_qat(arm="A3-s43", mode="smoke", model=FX.student(), source_meta=SRC,
                                           out_dir=TMP / "a3_wrong_loop", seed=43, clip_norm=5.0,
                                           clip_source="clip_selection", device="cpu", num_workers=0, lr=3e-4),
                    "arm_family")
    check("allow_a3_not_in_plain_loop", ok, d)
    ok, d = refused(lambda: M.run_am29_qat(arm="LR-s42", mode="smoke", model=FX.student(), source_meta=SRC,
                                           out_dir=TMP / "lr_teacher", seed=42, clip_norm=5.0,
                                           clip_source="clip_selection", device="cpu", num_workers=0, lr=6.25e-5,
                                           teacher=M.SelfTeacher(FX.student())), "teacher_not_applicable")
    check("allow_lr_takes_no_teacher", ok, d)


# ------------------------------------------------------------------ pairing
def _parent_rec(d: Path, seed: int) -> dict:
    st = M.am29_stage({42: "LR-s42", 43: "A3-s43", 44: "A3-s44"}[seed])
    return Q.resolve_parent(st, d, FX.sha_file(next(d.glob("e3_student_best_iter*.pt"))), seed)


def sec_pairing() -> dict:
    root = TMP / "parents"
    fx = {}
    for seed, it in ((42, 76000), (43, 72000), (44, 60000)):
        d = root / f"e3_s{seed}_lambda0.5_alpha25_80k_a1"
        ck, pj = FX.write_e3_parent(d, seed=seed, ckpt_name=f"e3_student_best_iter{it}.pt", it=it, student_seed=seed)
        fx[seed] = {"dir": d, "ck": ck, "proj": pj, "ckpt": f"e3_student_best_iter{it}.pt"}
    parents = {k: dict(v) for k, v in M.PARENTS_OF_RECORD.items()}
    for seed, key in ((42, "E3-s42"), (43, "E3-s43"), (44, "E3-s44")):
        parents[key].update({"sha256": fx[seed]["ck"], "run_id": fx[seed]["dir"].name, "checkpoint": fx[seed]["ckpt"]})
    parents["E3-s42"]["projection_sha256"] = fx[42]["proj"]
    p43, p44 = _parent_rec(fx[43]["dir"], 43), _parent_rec(fx[44]["dir"], 44)
    ok43 = M.projection_pairing_error("A3-s43", p43, fx[43]["proj"], parents=parents)
    ok44 = M.projection_pairing_error("A3-s44", p44, fx[44]["proj"], parents=parents)
    check("pairing_own_pair_accepted", ok43 is None and ok44 is None, "s43 and s44 each with its own projection.pt")
    other = dict(M.OTHER_PROJECTIONS_OF_RECORD)
    cases = {"s42's projection of record": M.projection_pairing_error("A3-s43", p43, fx[42]["proj"], parents=parents),
             "alpha 50's projection (DL-120)": M.projection_pairing_error(
                 "A3-s44", p44, next(iter(other)), parents=parents),
             "a checkpoint's sha256": M.projection_pairing_error("A3-s43", p43, fx[43]["ck"], parents=parents),
             "s44's parent for A3-s43": M.projection_pairing_error("A3-s43", p44, fx[44]["proj"], parents=parents),
             "s43's parent for A3-s44": M.projection_pairing_error("A3-s44", p43, fx[43]["proj"], parents=parents),
             "malformed sha256": M.projection_pairing_error("A3-s43", p43, "xyz", parents=parents),
             "a T arm": M.projection_pairing_error("T1", p43, fx[43]["proj"], parents=parents)}
    recorded = {k: dict(v) for k, v in parents.items()}
    recorded["E3-s43"]["projection_sha256"] = "ef" * 32
    cases["s43's recorded value differs"] = M.projection_pairing_error("A3-s43", p43, fx[43]["proj"], parents=recorded)
    check("pairing_refuses_unpaired", all(v is not None for v in cases.values()),
          "; ".join(f"{k}: {'refused' if v else 'ACCEPTED'}" for k, v in cases.items()))
    # the file's own pairing (qat_kd.projection_of_record): s44's projection.pt copied into s43's directory
    swapped = root / "swapped" / "e3_s43_lambda0.5_alpha25_80k_a1"
    import shutil
    shutil.copytree(fx[43]["dir"], swapped)
    shutil.copy2(fx[44]["dir"] / "projection.pt", swapped / "projection.pt")
    ps = _parent_rec(swapped, 43)
    sw_sha = FX.sha_file(swapped / "projection.pt")
    first = M.projection_pairing_error("A3-s43", ps, sw_sha, parents=parents)
    ok, d = refused(lambda: KD.projection_of_record(ps, sw_sha), "projection_unpaired_checkpoint")
    check("pairing_file_level_refused", first is None and ok,
          f"s44's projection.pt (not of record by sha256) in s43's directory: the file pairs with "
          f"{fx[44]['ckpt']}: {d}")
    check("pairing_s43_s44_projection_required", M.PARENTS_OF_RECORD["E3-s43"]["projection_sha256"] is None
          and M.PARENTS_OF_RECORD["E3-s44"]["projection_sha256"] is None
          and M.PARENTS_OF_RECORD["E3-s42"]["projection_sha256"] == "2e93ebe05e8c7d55b9ce7305a9066aab2e1777a710555c18ff9ea0cb84726751",
          "the decision log records no projection.pt sha256 for E3 s43/s44 (DL-158: keep tars only): a required argument")
    return {"fx": fx, "parents": parents}


# ------------------------------------------------------------------ gates
def sec_gates(pfx: dict) -> None:
    fx, parents = pfx["fx"], pfx["parents"]
    sel = FX.selections_repo(TMP / "selrepo")
    p42 = _parent_rec(fx[42]["dir"], 42)
    check("gates_parent_of_record", M.parent_of_record_error("LR-s42", p42, parents) is None
          and M.parent_of_record_error("LR-s43", p42, parents) is not None
          and M.parent_of_record_error("LR-s42", p42) is not None,
          "by sha256 (and run ID, checkpoint name): accepted under the fixture table, refused for another arm and "
          "under the table of record")
    lam = Q.read_tracked_selection(sel["lambda"], sel["lambda_sha256"], Q.LAMBDA_SELECTION_FORMAT, "lambda", sel["repo"])
    alpha = Q.read_tracked_selection(sel["alpha"], sel["alpha_sha256"], Q.ALPHA_SELECTION_FORMAT, "alpha", sel["repo"])
    check("gates_selections_of_record", M.selections_of_record_error(lam, alpha) is None
          and M.selections_of_record_error({"winner": {"lambda": 1.0}}, alpha) is not None, "lambda 0.5, alpha 25")
    r2 = TMP / "parents" / "e3-r2_s42_x"
    ck_r2, _ = FX.write_e3_parent(r2, seed=42, ckpt_name="e3_student_best_iter64000.pt", it=64000, arm="E3-R2",
                                  student_seed=99, with_projection=True)
    pr2 = Q.resolve_parent(M.am29_stage("T1"), r2, ck_r2, 42)
    b = M.t_parent_binding(pr2, lam, alpha)
    ok_r2, d = refused(lambda: M.t_parent_binding(p42, lam, alpha), "parent_arm_mismatch")
    check("gates_t_parent_binding", b["arm"] == "E3-R2" and ok_r2, f"E3-R2's run accepted; E3 s42's refused: {d}")
    # the real gates' order, on the CPU, under the fixture table and a fixture clip record
    clip_p = TMP / "clip_selection.json"
    csha = FX.write_json(clip_p, {"format": Q.CLIP_SELECTION_FORMAT, "winner": {"clip_norm": 5.0}})

    def args_for(**over):
        ns = M.build_parser().parse_args(["--arm", "LR-s42", "--seed", "42", "--lr", "6.25e-5"])
        for k, v in {"real_run": True, "confirm_real_run": True, "expect_head": "a" * 40, "num_workers": 12,
                     "grad_clip_norm": 5.0, "clip_selection": str(clip_p), "clip_selection_sha256": csha,
                     "lambda_selection": sel["lambda"], "lambda_selection_sha256": sel["lambda_sha256"],
                     "alpha_selection": sel["alpha"], "alpha_selection_sha256": sel["alpha_sha256"],
                     "source_run_dir": str(fx[42]["dir"]), "expect_source_sha256": fx[42]["ck"],
                     "out_dir": str(TMP / "fresh_out"), "device": "cuda", **over}.items():
            setattr(ns, k, v)
        return ns

    def code(ns, patch_clip=True):
        saved = M.CLIP_SELECTION_SHA256
        try:
            if patch_clip:
                M.CLIP_SELECTION_SHA256 = csha
            M.real_run_gates(ns, repo_root=sel["repo"], parents=parents)
        except Q.QATRefused as e:
            return e.code
        finally:
            M.CLIP_SELECTION_SHA256 = saved
        return None
    order = [("real_run_flag", dict(real_run=False)), ("confirm_real_run_flag", dict(confirm_real_run=False)),
             ("expect_head_format", dict(expect_head="abc")), ("arm_lr", dict(lr=3e-4)),
             ("num_workers", dict(num_workers=0)), ("clip_not_of_record", dict(grad_clip_norm=1.0)),
             ("selection_sha256_mismatch", dict(lambda_selection_sha256="ab" * 32)),
             ("out_dir", dict(out_dir=str(REPO / "x"))), ("cuda_required", {})]
    got = [(want, code(args_for(**over))) for want, over in order]
    check("gates_order_on_cpu", all(w == g for w, g in got), f"{got}")
    check("gates_clip_of_record_pinned", code(args_for(), patch_clip=False) == "clip_not_of_record"
          and M.CLIP_SELECTION_SHA256 == "c433145a6cfe0e133ebee8fcb70a76f05b60fc6462670abee34c5304704fe698",
          "a clip record other than DL-139's sha256 is refused")


# ------------------------------------------------------------------ meta
def sec_meta() -> None:
    m = FX.student()
    teacher = M.SelfTeacher(m, checkpoint_sha256=M.PARENTS_OF_RECORD["E3-R2-s42"]["sha256"])
    M.run_am29_qat(arm="T3", mode="smoke", model=m, source_meta={"sha256": "da70d82f" + "0" * 56, "path": None},
                   out_dir=TMP / "meta_t3", seed=42, clip_norm=5.0, clip_source="clip_selection", device="cpu",
                   num_workers=0, lr=3e-4, factor=1.0, e3_lambda=0.5, teacher=teacher, loaders=FX.loaders(),
                   max_steps=2, max_val_batches=1, log=FX.quiet)
    meta = FX.rows(TMP / "meta_t3", "run_meta")[0]
    a = meta["am29"]
    check("meta_am29_block", a["arm"] == "T3" and a["factor"] == 1.0 and a["lr"] == 3e-4 and a["seed"] == 42
          and a["parent_sha256"].startswith("da70d82f") and a["teacher_sha256"] == M.PARENTS_OF_RECORD["E3-R2-s42"]["sha256"]
          and a["T_logit"] == 4 and a["kd_weight"] == 0.5 and a["e3_lambda_logit"] == 0.5 and a["ramp"] is None,
          f"{ {k: a[k] for k in ('arm', 'factor', 'lr', 'seed', 'T_logit', 'kd_weight', 'ramp')} }")
    files = meta["code_files_sha256"]
    present = [f for f in M.AM29_FILES if (REPO / f).is_file()]
    check("meta_code_with_new_files", all(f in files for f in present) and len(present) == len(M.AM29_FILES)
          and all(p in files for p in Q.QAT_CODE_PATHS if (REPO / p).is_file()),
          f"{len(files)} files hashed: QAT_CODE_PATHS, E6-KD's and AM-29's {list(M.AM29_FILES)}")
    check("meta_stage_and_teacher", meta["stage"] == "AM29-T3" and meta["distillation"] is True
          and meta["teacher"]["in_optimizer"] is False and meta["cwd_projection_loaded"] is False
          and meta["T_logit"] == 4, f"stage {meta['stage']}, teacher {meta['teacher']['kind']}")
    ck = Q.read_jsonl(TMP / "meta_t3" / "qat_telemetry.jsonl")[0]
    pay = torch.load(TMP / "meta_t3" / "epoch_ckpts" / "e01.pt", map_location="cpu", weights_only=True)
    check("meta_checkpoint_stage", pay["stage"] == "AM29-T3" and pay["quantization"] == "qat-train-state"
          and len(ck) > 3, f"e01.pt stage {pay['stage']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args(argv)
    want = [s for s in a.sections.split(",") if s]
    pfx = None
    for s in want:
        if s not in SECTIONS:
            print(f"unknown section {s}")
            return 2
    try:
        for s in SECTIONS:
            if s not in want and not (s == "pairing" and "gates" in want):
                continue
            if s == "pairing":
                pfx = sec_pairing()
            elif s == "gates":
                sec_gates(pfx)
            else:
                globals()[f"sec_{s}"]()
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("no_unexpected_exception", False, f"{type(e).__name__}: {e}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
