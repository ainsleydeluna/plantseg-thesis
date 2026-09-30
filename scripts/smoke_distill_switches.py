#!/usr/bin/env python3
"""Unit smoke for the per-term distillation switches (L-AM17B-FG) and the alpha grid (L-AM16-ALPHA).

Synthetic CPU tensors and an explicit MockTeacher; no dataset, teacher checkpoint or GPU. The
run-level checks (d4 cross-commit invariance, d6 telemetry, d7 teacher stream, d8 checkpoints) are in
scripts/smoke_invariance_distill.py. Lane checks (docs/lane_specs/part2.md; errata E-7):

  d1  the stage table has E2/E3/A/F/G with the specified switches; the combined `cwd` key, an all-off
      stage and a missing, extra or non-bool term are refused; each stage instantiates exactly its
      modules (a projection iff cwd_feat, one ramp and one weight per instantiated term) and calls
      exactly its loss functions.
  d2  optimizer parameters = the student (E2, G) or the student + 51,200 (E3, A, F); the projection in
      its own param group; never a teacher parameter.
  d3  on a synthetic batch, total(term off) == total(term weight 0.0) bitwise, and both equal the sum
      of the instantiated terms in the pre-lane addition order (an independent copy of the 73fd4d7
      code path).
  d5  per-step weights of every instantiated term equal the pre-lane values for E2/E3; A/F/G have ramp
      objects for their own terms only.
  L2 d1  --alpha 30 refused, 25/50/100 accepted, off-grid only with --allow-offgrid in dry runs; the
      feature-term weight at the ramp's end equals alpha exactly; a real cwd_feat run refuses a
      --ckpt-dir without alpha<value> and passes every gate with it.
  gates  an argument for a term the stage does not instantiate is refused before any data access.
  2j (L-KD-HARDEN)  a --ckpt-dir with more than one alpha<value> token is refused; run() refuses an
      off-grid alpha in real mode itself; the lambda-semantics gate refuses in dry mode too; the grid
      error prints repr(alpha). The real launches here carry --num-workers and no --grad-clip-norm
      (item 2; AM-7).
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import src.training.train_distill as td  # noqa: E402
from configs.distill import DISTILL, DISTILL_STAGES, DISTILL_TERMS  # noqa: E402
from src.distill import FrozenTeacher, MockTeacher  # noqa: E402
from src.distill.cwd_projection import (PROJECTION_GROUP, PROJECTION_PARAMS,  # noqa: E402
                                        build_cwd_projection, build_cwd_projection_for,
                                        merge_optimizer_state, split_optimizer_state)
from src.models.student import build_student  # noqa: E402
from src.training.losses import cwd_channelwise_kl, downsample_validity, logit_kd_kl  # noqa: E402

NC, IGNORE = 116, 255
EXPECT = {"e2": ("E2", True, False, False), "e3": ("E3", True, True, True),
          "a": ("A", False, True, True), "f": ("F", False, True, False),
          "g": ("G", False, False, True)}
STUDENT_PARAMS = 2_933_688
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def raises(fn, exc, *args, **kwargs) -> tuple[bool, str]:
    try:
        fn(*args, **kwargs)
    except exc as e:           # noqa: PERF203
        return True, str(e)[:120]
    except Exception as e:     # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:120]
    return False, "no exception"


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and a.detach().contiguous().numpy().tobytes() == b.detach().contiguous().numpy().tobytes())


# ------------------------------------------------------------------------ pre-lane oracles
def pre_lane_ramp(it: int, n: int) -> float:
    """distill_ramp at 73fd4d7, copied: clamp((it-1)/(n-1), 0, 1)."""
    if n <= 1:
        return 1.0
    if it <= 1:
        return 0.0
    if it >= n:
        return 1.0
    return float(it - 1) / float(n - 1)


def pre_lane_total(flags, *, logits, head_logits, c5, mask, t_out, projection, lam, alpha, beta, ramp):
    """The 73fd4d7 distillation_losses code path, restricted to the switched-on terms, in its order."""
    lk, cf, cl = flags
    total = logits.new_zeros(())
    valid_os8 = downsample_validity(mask, head_logits.shape[-2:], ignore_index=IGNORE)
    t_os8 = t_out.logits
    if lk:
        total = total + ramp * lam * logit_kd_kl(head_logits, t_os8, valid_os8, T=4)
    if cf:
        s_feat = projection(c5)
        valid_s16 = downsample_validity(mask, s_feat.shape[-2:], ignore_index=IGNORE)
        total = total + ramp * alpha * cwd_channelwise_kl(s_feat, t_out.feat_s16, valid_s16, T=4,
                                                          channels_norm=320)
    if cl:
        total = total + ramp * beta * cwd_channelwise_kl(head_logits, t_os8, valid_os8, T=4)
    return total


def synthetic_batch():
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(2, NC, 64, 64, generator=g)
    head_logits = torch.randn(2, NC, 8, 8, generator=g)
    c5 = torch.randn(2, 160, 4, 4, generator=g)
    mask = torch.randint(0, NC, (2, 64, 64), generator=g)
    mask[0, :16, :] = IGNORE                          # a padded band, so validity masks matter
    torch.manual_seed(11)
    teacher = FrozenTeacher(MockTeacher(NC))
    t_out = teacher(torch.randn(2, 3, 64, 64, generator=g), logits_size=(8, 8), feat_size=(4, 4))
    torch.manual_seed(13)
    projection = build_cwd_projection()
    return dict(logits=logits, head_logits=head_logits, c5=c5, mask=mask, t_out=t_out,
                projection=projection)


def losses(stage_key, b, *, lam=1.0, alpha=50.0, beta=3.0, ramp=0.5, projection="auto"):
    stage = td.resolve_stage(stage_key)
    proj = (b["projection"] if stage["cwd_feat"] else None) if projection == "auto" else projection
    return td.distillation_losses(stage=stage, logits=b["logits"], head_logits=b["head_logits"],
                                  c5=b["c5"], mask=b["mask"], teacher_out=b["t_out"], projection=proj,
                                  lambda_logit=lam if stage["logit_kd"] else None, ramp=ramp,
                                  alpha=alpha, beta=beta)


# ------------------------------------------------------------------------------------- d1
def test_d1_stage_table() -> None:
    check("d1_config_table_is_the_trainer_table",
          sorted(DISTILL_STAGES) == sorted(td.STAGES) == ["a", "e2", "e3", "f", "g"]
          and DISTILL_TERMS == ("logit_kd", "cwd_feat", "cwd_logit"), str(sorted(td.STAGES)))
    for key, (name, lk, cf, cl) in EXPECT.items():
        s = td.resolve_stage(key)
        check(f"d1_{key}_switches", (s["name"], s["logit_kd"], s["cwd_feat"], s["cwd_logit"])
              == (name, lk, cf, cl) and "cwd" not in s, f"{s}")
    base = {"name": "X", "logit_kd": True, "cwd_feat": True, "cwd_logit": True}
    bad = {"combined_cwd_key": {"name": "X", "logit_kd": True, "cwd": True},
           "cwd_key_next_to_new_keys": {**base, "cwd": False},
           "all_terms_off": {"name": "X", "logit_kd": False, "cwd_feat": False, "cwd_logit": False},
           "missing_term": {"name": "X", "logit_kd": True, "cwd_feat": True},
           "extra_key": {**base, "lambda": 1.0},
           "non_bool_term": {**base, "cwd_logit": 1},
           "missing_name": {"logit_kd": True, "cwd_feat": False, "cwd_logit": False}}
    for label, cfg in bad.items():
        ok, msg = raises(td.validate_stage_config, td.StageConfigError, "x", cfg)
        check(f"d1_refuses_{label}", ok, msg)
    ok, msg = raises(td.validate_stage_config, td.StageConfigError, "x",
                     {"name": "X", "logit_kd": True, "cwd": True})
    check("d1_cwd_refusal_names_the_split", ok and "cwd_feat" in msg and "cwd_logit" in msg, msg)
    ok, msg = raises(td.validate_stage_config, td.StageConfigError, "x", bad["all_terms_off"])
    check("d1_all_off_refusal_names_e1", ok and "E1" in msg and "train_e1" in msg, msg)

    b = synthetic_batch()
    for key, (_, lk, cf, cl) in EXPECT.items():
        stage = td.resolve_stage(key)
        torch.manual_seed(0)
        proj = build_cwd_projection_for(stage)
        ramps = td.build_term_ramps(stage, 8)
        weights = td.term_weights(stage, lambda_logit=1.0 if lk else None, alpha=50.0 if cf else None)
        on = [t for t, f in zip(DISTILL_TERMS, (lk, cf, cl)) if f]
        check(f"d1_{key}_instantiates_exactly_its_modules",
              (proj is not None) == cf and list(ramps) == on and list(weights) == on
              and all(isinstance(r, td.TermRamp) and r.term == t for t, r in ramps.items()),
              f"projection={proj is not None} ramps={list(ramps)} weights={list(weights)}")
        calls = {"logit_kd_kl": 0, "cwd_feat": 0, "cwd_logit": 0, "projection": 0}
        real_kd, real_cwd = td.logit_kd_kl, td.cwd_channelwise_kl

        def kd_spy(*a, **k):
            calls["logit_kd_kl"] += 1
            return real_kd(*a, **k)

        def cwd_spy(s_map, *a, **k):
            calls["cwd_feat" if s_map.shape[1] == 320 else "cwd_logit"] += 1
            return real_cwd(s_map, *a, **k)

        hook = b["projection"].register_forward_hook(
            lambda *_: calls.__setitem__("projection", calls["projection"] + 1))
        td.logit_kd_kl, td.cwd_channelwise_kl = kd_spy, cwd_spy
        try:
            _, parts = losses(key, b)
        finally:
            td.logit_kd_kl, td.cwd_channelwise_kl = real_kd, real_cwd
            hook.remove()
        want = {"logit_kd_kl": int(lk), "cwd_feat": int(cf), "cwd_logit": int(cl), "projection": int(cf)}
        check(f"d1_{key}_calls_exactly_its_losses", calls == want and list(parts) == on,
              f"calls={calls} parts={list(parts)}")
    ok, msg = raises(losses, RuntimeError, "g", b, projection=b["projection"])
    check("d1_projection_refused_for_a_stage_without_cwd_feat", ok, msg)
    ok, msg = raises(losses, RuntimeError, "f", b, projection=None)
    check("d1_cwd_feat_without_projection_refused", ok, msg)


# ------------------------------------------------------------------------------------- d2
def test_d2_optimizer() -> None:
    torch.manual_seed(0)
    student = build_student(pretrained=False)
    n_student = sum(p.numel() for p in student.parameters())
    check("d2_student_params_2933688", n_student == STUDENT_PARAMS, f"{n_student:,}")
    teacher = FrozenTeacher(MockTeacher(NC))
    for key, (_, _, cf, _) in EXPECT.items():
        stage = td.resolve_stage(key)
        proj = build_cwd_projection_for(stage)
        opt, trainable = td.build_optimizer(student, proj, teacher)
        names = [g["name"] for g in opt.param_groups]
        count = sum(p.numel() for g in opt.param_groups for p in g["params"])
        ids = {id(p) for g in opt.param_groups for p in g["params"]}
        order_ok = trainable == list(student.parameters()) + (list(proj.parameters()) if proj else [])
        check(f"d2_{key}_param_count", count == n_student + (PROJECTION_PARAMS if cf else 0)
              and names == ["student"] + ([PROJECTION_GROUP] if cf else []) and order_ok
              and not (ids & teacher.parameter_ids()),
              f"count={count:,} groups={names}")

    class LeakyTeacher:                                    # shares a parameter with the student
        def parameter_ids(self):
            return {id(next(student.parameters()))}
    ok, msg = raises(td.build_optimizer, RuntimeError, student, None, LeakyTeacher())
    check("d2_teacher_parameter_in_optimizer_refused", ok, msg)

    # the checkpoint/projection.pt split of the optimizer state is lossless
    proj = build_cwd_projection_for(td.resolve_stage("e3"))
    opt, trainable = td.build_optimizer(student, proj)
    g = torch.Generator().manual_seed(3)
    for p in trainable:
        p.grad = torch.randn(p.shape, generator=g)
    opt.step()
    full = opt.state_dict()
    rest, part = split_optimizer_state(full)
    merged = merge_optimizer_state(rest, part)
    same = (merged["param_groups"] == full["param_groups"] and list(merged["state"]) == list(full["state"])
            and all(same_bits(merged["state"][i]["momentum_buffer"], full["state"][i]["momentum_buffer"])
                    for i in full["state"]))
    check("d2_optimizer_state_split_merge_lossless",
          same and [x["name"] for x in rest["param_groups"]] == ["student"]
          and [x["name"] for x in part["param_groups"]] == [PROJECTION_GROUP]
          and len(part["state"]) == 1, f"student entries={len(rest['state'])} "
                                       f"projection entries={len(part['state'])}")


# ------------------------------------------------------------------------------------- d3
def test_d3_composition() -> None:
    b = synthetic_batch()
    for ramp in (0.0, 0.5, 1.0):
        for key, (_, lk, cf, cl) in EXPECT.items():
            t_off, _ = losses(key, b, ramp=ramp)
            t_zero, _ = losses("e3", b, lam=1.0 if lk else 0.0, alpha=50.0 if cf else 0.0,
                               beta=3.0 if cl else 0.0, ramp=ramp)
            t_pre = pre_lane_total((lk, cf, cl), logits=b["logits"], head_logits=b["head_logits"],
                                   c5=b["c5"], mask=b["mask"], t_out=b["t_out"],
                                   projection=b["projection"], lam=1.0, alpha=50.0, beta=3.0,
                                   ramp=ramp)
            check(f"d3_{key}_ramp{ramp}_off_equals_zero_weight_equals_pre_lane_sum",
                  same_bits(t_off, t_zero) and same_bits(t_off, t_pre),
                  f"off={t_off.item()!r} zero={t_zero.item()!r} pre={t_pre.item()!r}")
    mapped, _ = losses("e3", b, ramp={"logit_kd": 0.5, "cwd_feat": 0.5, "cwd_logit": 0.5})
    scalar, _ = losses("e3", b, ramp=0.5)
    check("d3_per_term_ramp_mapping_equals_scalar", same_bits(mapped, scalar))
    grads = []
    for key in ("f", "e3"):
        proj = b["projection"]
        proj.zero_grad(set_to_none=True)
        lam, beta = (1.0, 3.0) if key == "f" else (0.0, 0.0)
        t, _ = losses(key, b, lam=lam, beta=beta, ramp=1.0)
        t.backward()
        grads.append(proj.weight.grad.clone())
    check("d3_projection_grad_off_equals_zero_weight", same_bits(grads[0], grads[1]))


# ------------------------------------------------------------------------------------- d5
def test_d5_ramps() -> None:
    n = 8
    for key, (_, lk, cf, cl) in EXPECT.items():
        stage = td.resolve_stage(key)
        ramps = td.build_term_ramps(stage, n)
        on = [t for t, f in zip(DISTILL_TERMS, (lk, cf, cl)) if f]
        check(f"d5_{key}_ramps_only_for_present_terms", list(ramps) == on, str(list(ramps)))
        if key in ("e2", "e3"):
            weights = td.term_weights(stage, lambda_logit=0.5, alpha=50.0)
            ok = all(ramps[t](it) * w == pre_lane_ramp(it, n) * w
                     for it in range(1, n + 4) for t, w in weights.items())
            check(f"d5_{key}_per_step_weights_equal_pre_lane", ok,
                  f"{[round(ramps[on[0]](it), 6) for it in range(1, n + 2)]}")
    check("d5_ramp_endpoints", td.distill_ramp(1, n) == 0.0 and td.distill_ramp(n, n) == 1.0
          and td.distill_ramp(n + 5, n) == 1.0 and td.distill_ramp(1, 1) == 1.0)


# -------------------------------------------------------------------------------- lane 2 d1
def test_l2_alpha_grid() -> None:
    check("l2_grid_is_am16", td.ALPHA_GRID == (25, 50, 100) and td.ALPHA_CWD_FEAT == 50
          and DISTILL["cwd"]["alpha_cwd_feature_map"] == 50)
    for a in (25.0, 50.0, 100.0, None):
        check(f"l2d1_alpha_{a}_accepted", td.alpha_gate_error(a, allow_offgrid=False, mode="real") is None)
    for a in (30.0, 0.0, -25.0, float("nan"), float("inf")):
        check(f"l2d1_alpha_{a}_refused", td.alpha_gate_error(a, allow_offgrid=False, mode="dry")
              is not None)
    check("l2d1_offgrid_dry_accepted", td.alpha_gate_error(30.0, allow_offgrid=True, mode="dry") is None)
    check("l2d1_offgrid_real_refused", td.alpha_gate_error(50.0, allow_offgrid=True, mode="real")
          is not None)
    check("l2d1_offgrid_still_needs_positive_finite",
          td.alpha_gate_error(float("nan"), allow_offgrid=True, mode="dry") is not None)
    stage = td.resolve_stage("e3")
    n = 8
    for a in (25.0, 50.0, 100.0):
        r = td.build_term_ramps(stage, n)["cwd_feat"]
        w = td.term_weights(stage, lambda_logit=1.0, alpha=a)["cwd_feat"]
        check(f"l2d1_feature_weight_at_ramp_end_equals_alpha_{a:g}",
              r(n) * w == a and r(n + 100) * w == a and r(1) * w == 0.0)
    cases = {("/w/e3_s42_alpha50", 50.0): True, ("/w/e3_run", 50.0): False,
             ("/w/e3_alpha500", 50.0): False, ("/w/e3_alpha5", 50.0): False,
             ("/w/alpha50", 50.0): True, ("/w/E3_ALPHA50", 50.0): True,
             ("/w/e3_alpha25_s42", 25.0): True, ("/w/e3_s42_alpha100", 100.0): True,
             ("/w/e3_alpha50.5", 50.0): False, ("/w/alpha50/e3_run", 50.0): False}
    for (d, a), ok in cases.items():
        check(f"l2d1_ckpt_dir_token {Path(d).name}@{a:g}",
              (td.ckpt_dir_alpha_error(d, a) is None) == ok)


# ------------------------------------------------------------------------------ CLI gates
class _Reached(Exception):
    pass


def _main(argv, stage=None):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            rc = td.main(argv, stage_default=stage)
        except SystemExit as e:
            rc = int(e.code) if e.code is not None else 0
    return rc, err.getvalue()


def test_cli_gates() -> None:
    seen: list[dict] = []
    real_run, real_load = td.run, td.load_frozen_teacher

    def fake_run(**kwargs):
        seen.append(kwargs)
        return 99

    def fake_load(*_a, **_k):
        raise _Reached()

    td.run = fake_run
    try:
        refused = {"e3_alpha30": ["--stage", "e3", "--alpha", "30"],
                   "e2_alpha": ["--stage", "e2", "--alpha", "50"],
                   "g_alpha": ["--stage", "g", "--alpha", "50"],
                   "g_lambda": ["--stage", "g", "--lambda-logit", "1.0"],
                   "a_lambda_semantics": ["--stage", "a", "--lambda-semantics", "x"],
                   "e2_dry_lambda_semantics_mismatch": ["--stage", "e2", "--lambda-semantics",
                                                        "logitkd@elsewhere"],
                   "f_allow_semantics_mismatch": ["--stage", "f", "--allow-semantics-mismatch"],
                   "e2_allow_offgrid": ["--stage", "e2", "--allow-offgrid"],
                   "e3_offgrid_real": ["--stage", "e3", "--real-run", "--confirm-real-run",
                                       "--alpha", "30", "--allow-offgrid"]}
        for label, argv in refused.items():
            n0 = len(seen)
            rc, err = _main(argv)
            check(f"gate_refuses_{label}", rc == 2 and len(seen) == n0 and "REFUSING" in err,
                  err.strip()[-110:])
        for label, argv, want in (("e3_alpha25", ["--stage", "e3", "--alpha", "25"], (25.0, False)),
                                  ("e3_default", ["--stage", "e3"], (50.0, False)),
                                  ("f_offgrid_dry", ["--stage", "f", "--alpha", "30", "--allow-offgrid"],
                                   (30.0, True)),
                                  ("g_plain", ["--stage", "g"], (None, False))):
            rc, err = _main(argv)
            got = (seen[-1].get("alpha"), seen[-1].get("alpha_offgrid")) if seen else None
            check(f"gate_accepts_{label}", rc == 99 and got == want, f"rc={rc} got={got}")
        lam = [k.get("lambda_logit") for k in seen[-4:]]
        check("gate_dry_lambda_only_for_logit_kd", lam == [1.0, 1.0, None, None], str(lam))
    finally:
        td.run = real_run

    # direct run() calls refuse mismatched weights before any data access
    common = dict(mode="dry", device="cpu", pretrained=False, teacher=None, batch_size=2, max_iters=1,
                  val_interval=1, max_val_batches=1, num_workers=0, ckpt_dir_arg=None,
                  grad_clip_norm=None, log_every=1, seed=42)
    for label, kw in (("g_with_lambda", dict(stage=td.resolve_stage("g"), lambda_logit=1.0)),
                      ("e2_without_lambda", dict(stage=td.resolve_stage("e2"), lambda_logit=None)),
                      ("e2_with_alpha", dict(stage=td.resolve_stage("e2"), lambda_logit=1.0,
                                             alpha=50.0))):
        ok, msg = raises(td.run, ValueError, **common, **kw)
        check(f"run_refuses_{label}", ok, msg)
    # L-KD-HARDEN item 2j: run() itself applies the alpha gate in real mode (a direct call; it refuses
    # before any seeding, data access or model build)
    real = dict(common, mode="real", max_iters=80000, num_workers=12,
                ckpt_dir_arg=str(Path(tempfile.mkdtemp(prefix="k1_switch_run_")) / "e3_s42_alpha30"))
    for label, kw in (("e3_real_alpha30", dict(alpha=30.0)),
                      ("e3_real_offgrid_flag", dict(alpha=50.0, alpha_offgrid=True))):
        ok, msg = raises(td.run, ValueError, **real, stage=td.resolve_stage("e3"), lambda_logit=1.0, **kw)
        check(f"run_refuses_{label}", ok, msg)
    # the grid error prints repr(alpha), so a near-grid value is not shown as a grid value
    msg = td.alpha_gate_error(50.0000001, allow_offgrid=False, mode="dry") or ""
    check("alpha_grid_error_prints_repr", "50.0000001" in msg, msg[:110])

    # real-run gate order on a synthetic TRAIN/VAL-only root; the CUDA gate is passed with
    # `--device cuda`, and load_frozen_teacher is replaced so reaching it proves every gate passed.
    import src.data.isolation as isolation
    root = Path(tempfile.mkdtemp(prefix="k1_switch_root_"))
    for split in ("train", "val"):
        for kind, ext in (("images", ".jpg"), ("annotations", ".png")):
            (root / kind / split).mkdir(parents=True)
            (root / kind / split / f"s0{ext}").touch()
    ckpt = Path(tempfile.mkdtemp(prefix="k1_switch_teacher_")) / "teacher.pth"
    torch.save({"state_dict": {"backbone.w": torch.zeros(1), "decode_head.w": torch.zeros(1)}}, ckpt)
    work = Path(tempfile.mkdtemp(prefix="k1_switch_ckpt_"))
    saved_root, saved_counts = td.DATA["root"], dict(isolation.DEFAULT_EXPECTED_COUNTS)
    td.DATA["root"] = str(root)
    isolation.DEFAULT_EXPECTED_COUNTS.update({"train": 1, "val": 1})
    td.load_frozen_teacher = fake_load
    # L-KD-HARDEN item 2: no --grad-clip-norm (AM-7) and an explicit --num-workers in a real launch.
    base = ["--real-run", "--confirm-real-run", "--device", "cuda", "--teacher-ckpt", str(ckpt),
            "--num-workers", "12"]
    try:
        for label, argv, want in (
                ("e3_dir_without_token", ["--stage", "e3", "--lambda-logit", "1", "--alpha", "50",
                                          "--ckpt-dir", str(work / "e3_run")], "refused"),
                ("e3_dir_wrong_alpha", ["--stage", "e3", "--lambda-logit", "1", "--alpha", "25",
                                        "--ckpt-dir", str(work / "e3_s42_alpha50")], "refused"),
                ("e3_dir_two_alpha_tokens", ["--stage", "e3", "--lambda-logit", "1", "--alpha", "50",
                                             "--ckpt-dir", str(work / "e3_alpha50_alpha25")], "refused"),
                ("e3_dir_same_token_twice", ["--stage", "e3", "--lambda-logit", "1", "--alpha", "50",
                                             "--ckpt-dir", str(work / "e3_alpha50_s42_alpha50")],
                 "refused"),
                ("f_dir_two_alpha_tokens", ["--stage", "f", "--ckpt-dir",
                                            str(work / "f_s42_alpha50_alpha100")], "refused"),
                ("e3_dir_with_token", ["--stage", "e3", "--lambda-logit", "1", "--alpha", "50",
                                       "--ckpt-dir", str(work / "e3_s42_alpha50")], "reached"),
                ("a_no_lambda", ["--stage", "a", "--alpha", "50",
                                 "--ckpt-dir", str(work / "a_s42_alpha50")], "reached"),
                ("f_default_alpha_dir", ["--stage", "f", "--ckpt-dir", str(work / "f_s42_alpha50")],
                 "reached"),
                ("g_any_dir", ["--stage", "g", "--ckpt-dir", str(work / "g_s42")], "reached"),
                ("e2_needs_lambda", ["--stage", "e2", "--ckpt-dir", str(work / "e2_s42")], "refused")):
            try:
                rc, err = _main(base + argv)
                outcome = "refused" if rc == 2 else f"rc={rc}"
            except _Reached:
                outcome, err = "reached", ""
            check(f"real_gate_{label}", outcome == want, f"{outcome} {err.strip()[-110:]}")
    finally:
        td.DATA["root"] = saved_root
        isolation.DEFAULT_EXPECTED_COUNTS.clear()
        isolation.DEFAULT_EXPECTED_COUNTS.update(saved_counts)
        td.load_frozen_teacher = real_load


def main() -> int:
    print("=" * 78)
    print("DISTILL SWITCHES + ALPHA GRID SMOKE — synthetic CPU only; no dataset, teacher ckpt or GPU")
    print(f"torch {torch.__version__}")
    print("=" * 78)
    for fn in (test_d1_stage_table, test_d2_optimizer, test_d3_composition, test_d5_ramps,
               test_l2_alpha_grid, test_cli_gates):
        print(f"--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:62}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\nRESULT: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
