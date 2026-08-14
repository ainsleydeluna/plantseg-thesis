#!/usr/bin/env python3
"""Synthetic CPU smoke for the E2/E3 distillation implementation. No training, no GPU, no dataset.

Deliberately requires NEITHER a RunPod pod NOR the real teacher checkpoint: everything runs on tiny
synthetic tensors and an explicit MockTeacher. The TEST split is never touched — no dataloader is
built at all. Uses only APIs that are stable between the local CPU torch and the pinned
torch 2.1.0+cu121 stack, so the result is meaningful before the pod exists.

Covers: Logit-KD correctness/gradients/ignore-index/spatial alignment · CWD correctness/shapes/
projection/gradients/valid-only softmax · student feature taps · E2 vs E3 stage composition ·
checkpoint projection isolation · real-run safety gates · absence of any quantization path.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.distill import (CWD_PROJECTION_KEY, CWDProjectionLeak, FrozenTeacher, MockTeacher,  # noqa: E402
                         StudentTaps, TeacherCheckpointMissing, assert_clean_student_state,
                         build_cwd_projection, find_projection_keys, require_teacher_checkpoint,
                         strip_cwd_projection)
from src.models.student import build_student  # noqa: E402
from src.seeds import set_seed  # noqa: E402
from src.training.losses import cwd_channelwise_kl, downsample_validity, logit_kd_kl  # noqa: E402
from src.training.train_distill import (STAGES, distill_ramp, distillation_losses,  # noqa: E402
                                        main as distill_main, resolve_stage)
from configs.distill import DISTILL          # noqa: E402
from configs.e1_student import E1_STUDENT    # noqa: E402

IGNORE = 255
NC = 116
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------- 1. Logit KD
def test_logit_kd() -> None:
    set_seed(42)
    b, h, w = 2, 8, 8
    student = torch.randn(b, NC, h, w, requires_grad=True)
    teacher = torch.randn(b, NC, h, w)                     # detached, as FrozenTeacher returns
    target = torch.randint(0, NC, (b, h, w))
    target[0, 0, :] = IGNORE                               # an ignored row

    loss = logit_kd_kl(student, teacher, target, T=4.0, ignore_index=IGNORE)
    check("kd_scalar_finite", loss.dim() == 0 and bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")

    loss.backward()
    check("kd_grad_reaches_student",
          student.grad is not None and float(student.grad.abs().sum()) > 0)
    check("kd_no_grad_on_teacher", teacher.grad is None and not teacher.requires_grad)

    # ignore-index: perturbing ONLY ignored positions must not change the loss
    s2 = student.detach().clone()
    t2 = teacher.clone()
    s2[0, :, 0, :] += 12.0
    t2[0, :, 0, :] -= 7.0
    base = logit_kd_kl(student.detach(), teacher, target, T=4.0, ignore_index=IGNORE)
    pert = logit_kd_kl(s2, t2, target, T=4.0, ignore_index=IGNORE)
    check("kd_ignore_index_invariant", torch.allclose(base, pert, atol=1e-6),
          f"base={base.item():.6f} perturbed={pert.item():.6f}")

    # spatial alignment is explicit: a coarser teacher is resampled before the loss
    t_small = torch.randn(b, NC, h // 2, w // 2)
    t_up = F.interpolate(t_small, size=(h, w), mode="bilinear", align_corners=False)
    aligned = logit_kd_kl(student.detach(), t_up, target, T=4.0, ignore_index=IGNORE)
    check("kd_spatial_alignment", aligned.dim() == 0 and bool(torch.isfinite(aligned)),
          f"teacher {tuple(t_small.shape[-2:])} -> {tuple(t_up.shape[-2:])}")


# ---------------------------------------------------------------- 2. CWD
def test_cwd() -> None:
    set_seed(42)
    b, c, h, w = 2, 320, 4, 4
    s = torch.randn(b, c, h, w, requires_grad=True)
    t = torch.randn(b, c, h, w)
    valid = torch.ones(b, h, w, dtype=torch.bool)
    valid[0, 0, 0] = False

    loss = cwd_channelwise_kl(s, t, valid, T=4.0, channels_norm=320)
    check("cwd_scalar_finite", loss.dim() == 0 and bool(torch.isfinite(loss)),
          f"loss={loss.item():.4f}")
    loss.backward()
    check("cwd_grad_reaches_student", s.grad is not None and float(s.grad.abs().sum()) > 0)
    check("cwd_no_grad_on_teacher", t.grad is None and not t.requires_grad)

    # identical maps -> KL 0
    z = torch.randn(b, c, h, w)
    check("cwd_zero_when_identical",
          abs(float(cwd_channelwise_kl(z, z, valid, T=4.0, channels_norm=320))) < 1e-5)

    # channel-wise spatial softmax sums to 1 over VALID locations only (Table 3.1 gate)
    m = valid.reshape(b, 1, h * w)
    x = (t.reshape(b, c, h * w) / 4.0).masked_fill(~m, torch.finfo(t.dtype).min)
    p = F.softmax(x, dim=-1)
    check("cwd_softmax_sums_to_one_over_valid",
          torch.allclose(p.sum(-1), torch.ones(b, c), atol=1e-5))
    check("cwd_softmax_zero_on_invalid", float(p[0, :, 0].abs().max()) < 1e-12)

    # a sample with no valid location contributes 0 instead of NaN
    none_valid = torch.zeros(b, h, w, dtype=torch.bool)
    check("cwd_all_invalid_is_zero_not_nan",
          float(cwd_channelwise_kl(t, t.clone(), none_valid, T=4.0)) == 0.0)

    # --- CONSERVATIVE ALL-VALID mask reduction (each cell covers 16x16 px of a 64x64 target) ---
    # 1 fully valid cell -> valid ; 2 fully ignored cell -> invalid
    full = torch.full((1, 64, 64), 3, dtype=torch.long)
    full[0, :16, :] = IGNORE                         # the ENTIRE first stride-16 cell row
    ds_full = downsample_validity(full, (4, 4), ignore_index=IGNORE)
    check("mask_shape", tuple(ds_full.shape) == (1, 4, 4), f"{tuple(ds_full.shape)}")
    check("mask_fully_valid_cell_is_valid", bool(ds_full[0, 1:].all()))
    check("mask_fully_ignored_cell_is_invalid", not bool(ds_full[0, 0].any()))

    # 3 partially IGNORED cell -> INVALID (only half the first cell row is padded)
    partial = torch.full((1, 64, 64), 3, dtype=torch.long)
    partial[0, :8, :] = IGNORE
    ds_partial = downsample_validity(partial, (4, 4), ignore_index=IGNORE)
    check("mask_partially_ignored_cell_is_invalid", not bool(ds_partial[0, 0].any()),
          "all-valid rule: a mixed cell must NOT participate in CWD")
    check("mask_untouched_cells_stay_valid", bool(ds_partial[0, 1:].all()))

    # 4 partially VALID cell -> INVALID (a single valid pixel in an otherwise ignored cell)
    speck = torch.full((1, 64, 64), IGNORE, dtype=torch.long)
    speck[0, 0, 0] = 3
    ds_speck = downsample_validity(speck, (4, 4), ignore_index=IGNORE)
    check("mask_partially_valid_cell_is_invalid", not bool(ds_speck.any()),
          "one valid pixel must NOT make a padded cell valid")

    # non-divisible target grids must still reduce exactly
    odd = torch.full((1, 30, 30), 3, dtype=torch.long)
    odd[0, 0, 0] = IGNORE
    ds_odd = downsample_validity(odd, (4, 4), ignore_index=IGNORE)
    check("mask_non_divisible_grid_ok",
          tuple(ds_odd.shape) == (1, 4, 4) and not bool(ds_odd[0, 0, 0]),
          "arbitrary HxW -> hxw")


# ---------------------------------------------------------------- 3. projection + taps
def test_projection_and_taps() -> None:
    set_seed(42)
    proj = build_cwd_projection()
    x = torch.randn(2, 160, 4, 4)
    y = proj(x)
    check("proj_shape_160_to_320", tuple(y.shape) == (2, 320, 4, 4), f"{tuple(y.shape)}")

    student = build_student(pretrained=False)
    img = torch.randn(1, 3, 64, 64)
    with StudentTaps(student) as taps:
        logits = student(img)
        c5, head_logits = taps.require()
    check("taps_logits_full_res", tuple(logits.shape) == (1, NC, 64, 64), f"{tuple(logits.shape)}")
    check("taps_c5_is_160ch_os16", tuple(c5.shape) == (1, 160, 4, 4), f"{tuple(c5.shape)}")
    check("taps_head_logits_os8", tuple(head_logits.shape) == (1, NC, 8, 8),
          f"{tuple(head_logits.shape)}")

    # gradients flow through the projection AND back into the student
    s_feat = proj(c5)
    t_feat = torch.randn_like(s_feat)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    cwd_channelwise_kl(s_feat, t_feat, valid, T=4.0, channels_norm=320).backward()
    check("proj_grad_present", proj.weight.grad is not None
          and float(proj.weight.grad.abs().sum()) > 0)
    check("cwd_grad_reaches_student_backbone",
          any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in student.parameters()))

    # hooks removed on exit -> the E1 forward path is restored exactly
    check("taps_hooks_removed",
          len(student.head._forward_hooks) == 0
          and len(student.features[student.high_tap]._forward_hooks) == 0)


# ---------------------------------------------------------------- 4. frozen teacher
def test_frozen_teacher() -> None:
    set_seed(42)
    teacher = FrozenTeacher(MockTeacher(NC))
    img = torch.randn(1, 3, 64, 64, requires_grad=True)
    out = teacher(img, feat_size=(4, 4))
    check("teacher_feat_320ch_at_requested_grid", tuple(out.feat_s16.shape) == (1, 320, 4, 4),
          f"{tuple(out.feat_s16.shape)}")
    check("teacher_outputs_detached",
          not out.logits.requires_grad and not out.feat_s16.requires_grad)
    check("teacher_trainable_parameters_empty", teacher.trainable_parameters() == [])
    teacher.train(True)
    check("teacher_stays_eval_after_train_call", not teacher.teacher.training)

    # standard nn.Module semantics: parameters stay present and introspectable, but frozen
    t_params = list(teacher.teacher.parameters())
    check("teacher_has_real_parameters", len(t_params) > 0 and sum(p.numel() for p in t_params) > 0,
          f"{sum(p.numel() for p in t_params):,} params")
    check("teacher_all_requires_grad_false", all(not p.requires_grad for p in t_params))
    check("teacher_parameters_introspectable",
          len(list(teacher.parameters())) == len(t_params) and len(teacher.state_dict()) > 0)

    # gradients never appear on the teacher after a student-side backward
    student = build_student(pretrained=False)
    img2 = torch.randn(1, 3, 64, 64)
    out2 = teacher(img2, feat_size=(4, 4))
    (student(img2).mean() + out2.logits.mean()).backward()
    check("teacher_grads_none_after_backward", all(p.grad is None for p in t_params))

    # the optimizer is built from student (+ projection) ONLY and shares nothing with the teacher
    proj = build_cwd_projection()
    trainable = list(student.parameters()) + list(proj.parameters())
    opt = torch.optim.SGD(trainable, lr=1e-2, momentum=0.9)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    check("optimizer_disjoint_from_teacher", not (opt_ids & teacher.parameter_ids()),
          f"|optimizer|={len(opt_ids)} |teacher|={len(teacher.parameter_ids())}")
    check("optimizer_contains_student_and_projection",
          all(id(p) in opt_ids for p in list(student.parameters()) + list(proj.parameters())))

    # a real run cannot silently get a random teacher
    try:
        require_teacher_checkpoint(None)
        check("teacher_missing_path_fails_loud", False, "no exception raised")
    except TeacherCheckpointMissing:
        check("teacher_missing_path_fails_loud", True)
    try:
        require_teacher_checkpoint(str(REPO / "does_not_exist_teacher.pth"))
        check("teacher_absent_file_fails_loud", False, "no exception raised")
    except TeacherCheckpointMissing:
        check("teacher_absent_file_fails_loud", True)


# ---------------------------------------------------------------- 4b. training schedule
def test_training_schedule() -> None:
    set_seed(42)
    # --- first-epoch linear ramp (contract B2: "linear 0 -> target over first epoch") ---
    n = 336                                     # one epoch at train=5367, batch=16
    check("ramp_starts_near_zero", 0.0 < distill_ramp(1, n) < 0.01, f"{distill_ramp(1, n):.6f}")
    check("ramp_is_linear_midway", abs(distill_ramp(n // 2, n) - 0.5) < 1e-6)
    check("ramp_reaches_target_at_epoch_end", distill_ramp(n, n) == 1.0)
    check("ramp_stays_at_target_after", distill_ramp(n * 7, n) == 1.0)
    check("ramp_degenerate_epoch_is_target", distill_ramp(1, 1) == 1.0)
    check("ramp_config_source_is_contract",
          E1_STUDENT["distill_weight_ramp"] == "linear 0 -> target over first epoch",
          E1_STUDENT["distill_weight_ramp"])

    # --- the ramp actually scales every distillation term ---
    b, h8, h16 = 1, 8, 4
    logits = torch.randn(b, NC, 64, 64, requires_grad=True)
    head_logits = torch.randn(b, NC, h8, h8, requires_grad=True)
    c5 = torch.randn(b, 160, h16, h16, requires_grad=True)
    mask = torch.randint(0, NC, (b, 64, 64))
    proj = build_cwd_projection()
    t_out = FrozenTeacher(MockTeacher(NC))(torch.randn(b, 3, 64, 64), feat_size=(h16, h16))
    e3 = resolve_stage("e3")
    kw = dict(stage=e3, logits=logits, head_logits=head_logits, c5=c5, mask=mask,
              teacher_out=t_out, projection=proj, lambda_logit=1.0)
    full_total, parts_full = distillation_losses(**kw, ramp=1.0)
    half_total, parts_half = distillation_losses(**kw, ramp=0.5)
    check("ramp_scales_distillation_total",
          torch.allclose(half_total, 0.5 * full_total, rtol=1e-5, atol=1e-7),
          f"full={float(full_total):.5f} half={float(half_total):.5f}")
    check("ramp_leaves_reported_parts_unramped",
          all(abs(parts_full[k] - parts_half[k]) < 1e-9 for k in parts_full))

    # --- CWD normalisation: feature map C=320, logit map C=116, never hard-coded ---
    # NOTE: the teacher map must differ NON-CONSTANTLY from the student's — a constant offset leaves
    # a spatial softmax unchanged, which would make every KL zero and these checks vacuous.
    feat_s, feat_t = torch.randn(1, 320, 4, 4), torch.randn(1, 320, 4, 4)
    valid_f = torch.ones(1, 4, 4, dtype=torch.bool)
    feat_default = cwd_channelwise_kl(feat_s, feat_t, valid_f, T=4.0)
    check("cwd_feature_norm_is_320",
          torch.allclose(feat_default,
                         cwd_channelwise_kl(feat_s, feat_t, valid_f, T=4.0, channels_norm=320))
          and float(feat_default) > 1e-3, f"L_CWD_feat={float(feat_default):.5f}")

    lg_s, lg_t = torch.randn(1, NC, 8, 8), torch.randn(1, NC, 8, 8)
    valid_l = torch.ones(1, 8, 8, dtype=torch.bool)
    default_116 = cwd_channelwise_kl(lg_s, lg_t, valid_l, T=4.0)
    explicit_116 = cwd_channelwise_kl(lg_s, lg_t, valid_l, T=4.0, channels_norm=116)
    wrong_320 = cwd_channelwise_kl(lg_s, lg_t, valid_l, T=4.0, channels_norm=320)
    check("cwd_logit_term_is_non_degenerate", float(default_116) > 1e-3,
          f"L_CWD_logit={float(default_116):.5f} (guards against a vacuous comparison)")
    check("cwd_logit_norm_is_116_by_default", torch.allclose(default_116, explicit_116))
    check("cwd_logit_norm_not_hardcoded_320", not torch.allclose(default_116, wrong_320),
          f"C=116 -> {float(default_116):.5f} vs C=320 -> {float(wrong_320):.5f}")
    check("cwd_norm_ratio_matches_channel_counts",
          torch.allclose(default_116 / wrong_320, torch.tensor(320.0 / 116.0), rtol=1e-4),
          f"ratio={float(default_116 / wrong_320):.5f} expected={320 / 116:.5f}")
    check("cwd_T_squared_over_C",
          torch.allclose(cwd_channelwise_kl(lg_s, lg_t, valid_l, T=4.0, channels_norm=1),
                         default_116 * 116, rtol=1e-4))

    # --- CWD logit term runs on the OS8 head map, not the upsampled 512-res logits ---
    check("cwd_logit_uses_os8_head_map",
          set(parts_full) == {"logit_kd", "cwd_feat", "cwd_logit"}
          and tuple(head_logits.shape[-2:]) == (h8, h8),
          f"head map {tuple(head_logits.shape)} vs full logits {tuple(logits.shape)}")

    # --- global-norm gradient clipping path (value itself intentionally unset: D-A/D2) ---
    student = build_student(pretrained=False)
    student(torch.randn(1, 3, 64, 64)).mean().backward()
    trainable = list(student.parameters()) + list(proj.parameters())
    total_norm = torch.nn.utils.clip_grad_norm_(trainable, 0.1)
    after = torch.norm(torch.stack([p.grad.norm() for p in trainable if p.grad is not None]))
    check("grad_clip_global_norm_path", float(after) <= 0.1 + 1e-5,
          f"pre={float(total_norm):.4f} post={float(after):.4f} max_norm=0.1")
    check("grad_clip_default_is_unset_per_D2", E1_STUDENT["grad_clip_max_norm"] is None,
          "no numeric max_norm exists in any authoritative source")

    # --- teacher receives the EXACT same already-augmented tensor as the student ---
    seen: dict[str, int] = {}
    teacher = FrozenTeacher(MockTeacher(NC))
    hs = student.register_forward_pre_hook(lambda m, i: seen.__setitem__("student", id(i[0])))
    ht = teacher.teacher.register_forward_pre_hook(lambda m, i: seen.__setitem__("teacher", id(i[0])))
    augmented = torch.randn(1, 3, 64, 64)
    student(augmented)
    teacher(augmented, feat_size=(4, 4))
    hs.remove(); ht.remove()
    check("teacher_sees_identical_augmented_input",
          seen.get("student") == seen.get("teacher") == id(augmented),
          "same tensor object reaches both models")

    # --- lambda_logit is reused unchanged in E3 (contract flag), and neither stage can resume ---
    check("lambda_logit_reused_in_e3", DISTILL["logit_kd"]["reused_unchanged_in_e3"] is True)
    src = (REPO / "src/training/train_distill.py").read_text(encoding="utf-8")
    check("no_resume_from_other_stage",
          all(tok not in src for tok in ("--resume", "--init-from", "load_state_dict(")),
          "E2/E3 always start from ImageNet init, never from each other")


# ---------------------------------------------------------------- 5. stage composition
def test_stage_composition() -> None:
    e2, e3 = resolve_stage("e2"), resolve_stage("e3")
    check("e2_is_logit_kd_only", e2["logit_kd"] and not e2["cwd"], str(e2["objective"]))
    check("e3_is_logit_kd_plus_cwd", e3["logit_kd"] and e3["cwd"], str(e3["objective"]))
    check("only_two_distill_stages", sorted(STAGES) == ["e2", "e3"])
    try:
        resolve_stage("e4")
        check("unknown_stage_rejected", False, "no exception raised")
    except ValueError:
        check("unknown_stage_rejected", True)


# ---------------------------------------------------------------- 6. checkpoint isolation
def test_checkpoint_isolation() -> None:
    student = build_student(pretrained=False)
    proj = build_cwd_projection()
    payload = {"model_state_dict": student.state_dict(),
               CWD_PROJECTION_KEY: proj.state_dict()}
    assert_clean_student_state(payload["model_state_dict"])
    check("student_state_has_no_projection",
          find_projection_keys(payload["model_state_dict"]) == [])
    check("projection_stored_separately", payload[CWD_PROJECTION_KEY] is not None
          and "weight" in payload[CWD_PROJECTION_KEY])

    leaked = dict(payload["model_state_dict"])
    leaked["cwd_projection.weight"] = proj.weight.detach().clone()
    try:
        assert_clean_student_state(leaked)
        check("leak_detected", False, "no exception raised")
    except CWDProjectionLeak:
        check("leak_detected", True)
    check("leak_strippable", find_projection_keys(strip_cwd_projection(leaked)) == [])

    # removing/never-attaching the projection cannot change student logits
    set_seed(7)
    x = torch.randn(1, 3, 64, 64)
    student.eval()
    with torch.no_grad():
        before = student(x).clone()
    keys_before = list(student.state_dict())
    _ = build_cwd_projection()            # exists alongside, never inside, the student
    with torch.no_grad():
        after = student(x)
    check("projection_removal_logit_neutral", torch.equal(before, after))
    check("student_state_keys_unchanged", list(student.state_dict()) == keys_before)
    check("projection_trainable_in_e3", all(p.requires_grad for p in proj.parameters()))


# ---------------------------------------------------------------- 7. safety gates
def test_safety_gates() -> None:
    # Every gate must return BEFORE any dataloader/teacher is built (no dataset access here).
    check("gate_real_without_confirm",
          distill_main(["--real-run"], stage_default="e2") == 2)
    check("gate_confirm_without_real",
          distill_main(["--confirm-real-run"], stage_default="e2") == 2)
    check("gate_dry_and_real_conflict",
          distill_main(["--dry-run", "--real-run"], stage_default="e3") == 2)
    # No CUDA locally -> the real run must refuse before touching data or the teacher.
    if not torch.cuda.is_available():
        check("gate_real_refuses_cpu",
              distill_main(["--real-run", "--confirm-real-run"], stage_default="e2") == 2)
    else:
        check("gate_real_refuses_cpu",
              distill_main(["--real-run", "--confirm-real-run", "--device", "cpu"],
                           stage_default="e2") == 2)

    # No quantization path is reachable from the distillation entry points.
    forbidden = ("quantize_dynamic", "prepare_qat", "convert(", "prepare(", ".fuse(",
                 "QuantWrapper", "get_default_qconfig")
    hits = []
    for f in ("src/training/train_distill.py", "src/training/train_e2.py",
              "src/training/train_e3.py"):
        text = (REPO / f).read_text(encoding="utf-8")
        hits += [f"{f}:{tok}" for tok in forbidden if tok in text]
    check("no_quantization_path_in_e2_e3", hits == [], str(hits))


def main() -> int:
    print("=" * 78)
    print("E2/E3 DISTILLATION SMOKE — synthetic CPU only; no dataset, no teacher ckpt, no GPU")
    print(f"torch {torch.__version__} (local stack; pinned run stack is 2.1.0+cu121)")
    print("=" * 78)
    for fn in (test_logit_kd, test_cwd, test_projection_and_taps, test_frozen_teacher,
               test_training_schedule, test_stage_composition, test_checkpoint_isolation,
               test_safety_gates):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:38}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\nRESULT: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
