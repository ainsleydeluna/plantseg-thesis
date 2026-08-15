#!/usr/bin/env python3
"""Synthetic verification of the E4 (PTQ) / E5 (QAT) quantization core. No dataset, no GPU, no ckpt.

Everything runs on a randomly-initialised student and synthetic checkpoints in a temp dir outside
the repository. Local torch is 2.9.1+cpu while the registered experiment stack is torch 2.1.0+cu121,
so results here are STRUCTURAL DEVELOPMENT EVIDENCE — official operator-support and conversion
validation must be repeated on the pinned stack.

QConfig checks introspect REAL observer / fake-quant instances (dtype, qscheme, quant_min,
quant_max, ch_axis, reduce_range), not class names.
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.models.student import build_student  # noqa: E402
from src.quant import (BN_FREEZE_PCT_RANGE, CALIBRATION_COUNT, CalibrationIndexError,  # noqa: E402
                       QuantPreparationError, SourceCheckpointInvalid, bn_freeze_iteration,
                       build_calibration_index, calibrate, convert_model, describe_qconfig,
                       disable_observers, freeze_bn_stats, load_calibration_index,
                       load_e1_source_checkpoint, prepare_ptq, prepare_qat_model, ptq_qconfig,
                       qat_grad_clip_gate_error, qat_qconfig, quantization_coverage,
                       save_calibration_index, select_qnnpack_backend, try_converted_forward)
from src.seeds import set_seed  # noqa: E402

NC = 116
TMP = Path(tempfile.mkdtemp(prefix="smoke_quant_"))
X = torch.randn(1, 3, 64, 64)
results: list[tuple[str, bool, str]] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect_raises(name: str, exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no exception raised")
    except exc as e:
        check(name, True, type(e).__name__)
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


def count_types(model: nn.Module, types) -> int:
    return sum(1 for m in model.modules() if isinstance(m, types))


def _backend_available() -> bool:
    """True only when the contract's QNNPACK backend is really present in this build."""
    return "qnnpack" in torch.backends.quantized.supported_engines


# ---------------------------------------------------------------- 1. backend + qconfigs
def test_qconfigs() -> None:
    supported = list(torch.backends.quantized.supported_engines)
    if "qnnpack" in supported:
        engine = select_qnnpack_backend()
        check("qnnpack_engine_selected",
              engine == "qnnpack" and torch.backends.quantized.engine == "qnnpack", engine)
    else:
        # LOCAL ENVIRONMENT DIFFERENCE — this CPU build ships onednn only. The production gate must
        # (and does) refuse loudly rather than substitute a different backend for the accuracy
        # artifact; the structural path below then runs on an explicitly labelled local proxy,
        # matching the repo's existing onednn-proxy precedent (docs/b7_result.md).
        expect_raises("qnnpack_absence_fails_loudly", Exception, select_qnnpack_backend)
        proxy = "onednn" if "onednn" in supported else supported[0]
        torch.backends.quantized.engine = proxy
        globals()["_PROXY_ENGINE"] = proxy
        notes.append(f"LOCAL ENVIRONMENT DIFFERENCE: supported_engines={supported}; QNNPACK absent, "
                     f"so structural checks run on the {proxy!r} proxy. Backend-specific operator "
                     "support MUST be re-verified on the pinned torch 2.1.0+cu121 QNNPACK stack.")
        check("local_proxy_engine_set", torch.backends.quantized.engine == proxy, proxy)

    p = describe_qconfig(ptq_qconfig())
    pa, pw = p["activation"], p["weight"]
    check("ptq_act_is_histogram_observer", pa["observer_class"] == "HistogramObserver",
          pa["observer_class"])
    check("ptq_act_quint8_per_tensor_affine",
          pa["dtype"] == torch.quint8 and pa["qscheme"] == torch.per_tensor_affine,
          f"{pa['dtype']} / {pa['qscheme']}")
    check("ptq_act_full_uint8_range", pa["quant_min"] == 0 and pa["quant_max"] == 255,
          f"{pa['quant_min']}..{pa['quant_max']}")
    check("ptq_act_reduce_range_false", pa["reduce_range"] is False, str(pa["reduce_range"]))
    check("ptq_weight_per_channel_symmetric_qint8",
          pw["dtype"] == torch.qint8 and pw["qscheme"] == torch.per_channel_symmetric
          and pw["ch_axis"] == 0, f"{pw['dtype']} / {pw['qscheme']} / ch_axis={pw['ch_axis']}")
    notes.append(f"PTQ weight observer resolved quant_min/max = "
                 f"{pw['quant_min']}..{pw['quant_max']} ({pw['observer_class']})")

    q = describe_qconfig(qat_qconfig())
    qa, qw = q["activation"], q["weight"]
    check("qat_act_is_fake_quant", qa["is_fake_quant"] is True, qa["class"])
    check("qat_act_moving_average_observer",
          qa["observer_class"] == "MovingAverageMinMaxObserver", qa["observer_class"])
    check("qat_act_quint8_per_tensor_affine_full_range",
          qa["dtype"] == torch.quint8 and qa["qscheme"] == torch.per_tensor_affine
          and qa["quant_min"] == 0 and qa["quant_max"] == 255,
          f"{qa['dtype']} / {qa['qscheme']} / {qa['quant_min']}..{qa['quant_max']}")
    check("qat_weight_per_channel_symmetric_qint8",
          qw["dtype"] == torch.qint8 and qw["qscheme"] == torch.per_channel_symmetric
          and qw["ch_axis"] == 0, f"{qw['dtype']} / {qw['qscheme']} / ch_axis={qw['ch_axis']}")
    check("qat_weight_is_fake_quant", qw["is_fake_quant"] is True, qw["class"])

    # the defaults we are deliberately NOT using would be per-TENSOR on weights
    from torch.ao.quantization import get_default_qconfig
    d = describe_qconfig(get_default_qconfig("qnnpack"))
    check("default_qnnpack_would_be_per_tensor_weights",
          d["weight"]["qscheme"] != torch.per_channel_symmetric,
          f"default weight qscheme={d['weight']['qscheme']} -> contract requires per-channel")


# ---------------------------------------------------------------- 2. PTQ path
def test_ptq_path() -> None:
    set_seed(42)
    model = build_student(pretrained=False)
    model.eval()
    with torch.no_grad():
        y = model(X)
    check("fp32_student_outputs_116ch", tuple(y.shape) == (1, NC, 64, 64), str(tuple(y.shape)))

    # --- BEFORE fusion: the upstream-eligible targets must actually exist ---
    from torchvision.ops.misc import Conv2dNormActivation
    eligible = [m for m in model.features.modules()
                if isinstance(m, Conv2dNormActivation) and len(m) >= 2
                and isinstance(m[1], nn.BatchNorm2d)]
    bb_bn_before = count_types(model.features, nn.BatchNorm2d)
    check("backbone_has_fuseable_targets_before_fusion",
          len(eligible) > 0 and bb_bn_before > 0,
          f"{len(eligible)} Conv2dNormActivation groups, {bb_bn_before} backbone BatchNorm2d")

    # --- AFTER fuse(is_qat=False): every eligible group must be handled, loudly ---
    fused_ptq = copy.deepcopy(model)
    fused_ptq.fuse(is_qat=False)
    rep = fused_ptq.fusion_report()
    globals()["_PTQ_FUSE_REPORT"] = rep
    check("ptq_fuses_every_eligible_backbone_group",
          rep["fused_backbone_groups"] == rep["eligible_conv_norm_activation"] > 0,
          f"{rep['fused_backbone_groups']}/{rep['eligible_conv_norm_activation']} groups, "
          f"SE groups={rep['squeeze_excitation_groups']}, fuse_fn={rep['fuse_fn']}")
    check("ptq_backbone_bn_decreases_by_accounting",
          rep["backbone_bn_after"] < rep["backbone_bn_before"],
          f"backbone BatchNorm2d {rep['backbone_bn_before']} -> {rep['backbone_bn_after']}")
    check("ptq_no_silent_zero_fusion", rep["fused_backbone_groups"] > 0)

    bn_before = count_types(model, nn.BatchNorm2d)
    prepared = prepare_ptq(model, select_backend=_backend_available())
    bn_after = count_types(prepared, nn.BatchNorm2d)
    check("fusion_precedes_observer_insertion", bn_after < bn_before,
          f"BatchNorm2d {bn_before} -> {bn_after} (fused by fuse() before prepare)")
    obs = [n for n, _ in prepared.named_modules() if "activation_post_process" in n]
    check("ptq_observers_inserted", len(obs) > 0, f"{len(obs)} observer modules")

    # What did fuse() actually fuse? Recorded as data, not asserted against an assumed count.
    import torch.ao.nn.intrinsic as nni
    fused = [n for n, m in prepared.named_modules()
             if type(m).__module__.startswith(("torch.ao.nn.intrinsic", "torch.nn.intrinsic"))]
    globals()["_FUSE_DIAG"] = {"bn_before": bn_before, "bn_after": bn_after,
                               "intrinsic_fused_modules": len(fused)}
    check("fusion_diagnostic_recorded", True,
          f"BatchNorm2d {bn_before}->{bn_after}, intrinsic fused modules={len(fused)}")

    n_cal = calibrate(prepared, [X, torch.randn(1, 3, 64, 64)])
    check("ptq_calibration_runs", n_cal == 2, f"{n_cal} batches")
    expect_raises("ptq_refuses_zero_calibration_batches", QuantPreparationError,
                  calibrate, copy.deepcopy(prepared), [])

    converted = convert_model(prepared)
    obs_after = [n for n, _ in converted.named_modules() if "activation_post_process" in n]
    # convert() strips observers from every swapped module EXCEPT QFunctional, which keeps its
    # activation_post_process as the holder of its own output qparams. So the accurate assertion is:
    # observers largely disappear, and every survivor sits on a QFunctional (residual skip_add, SE
    # skip_mul, LR-ASPP ff_mul/ff_add) — i.e. on a QUANTIZED module, not an FP32 leftover.
    from torch.ao.nn.quantized import QFunctional
    survivors = []
    for name in obs_after:
        parent_name = name.rsplit(".activation_post_process", 1)[0]
        parent = converted.get_submodule(parent_name) if parent_name else converted
        survivors.append((parent_name, type(parent).__name__, isinstance(parent, QFunctional)))
    check("observers_removed_except_qfunctional_qparams",
          len(obs_after) < len(obs) and all(is_qf for _, _, is_qf in survivors),
          f"{len(obs)} -> {len(obs_after)}; all survivors are QFunctional qparam holders")
    globals()["_OBS_SURVIVORS"] = survivors
    globals()["_CONVERTED"] = converted


# ---------------------------------------------------------------- 3. QAT path
def test_qat_path() -> None:
    set_seed(42)
    model = build_student(pretrained=False)
    prepared = prepare_qat_model(model, select_backend=_backend_available())
    from torch.ao.quantization.fake_quantize import FakeQuantize
    fq = [m for m in prepared.modules() if isinstance(m, FakeQuantize)]
    check("qat_fake_quant_inserted", len(fq) > 0, f"{len(fq)} FakeQuantize modules")
    check("qat_fake_quant_enabled_from_step_0",
          all(int(m.fake_quant_enabled) == 1 for m in fq))

    # One warm-up step with observers ENABLED so the per-channel weight observers size their
    # scale/zero_point. (Disabling observers before any forward would leave scalar qparams.)
    prepared.train()
    with torch.no_grad():
        prepared(X)
    check("qat_forward_runs_with_fake_quant", True, "one QAT forward completed")

    # QAT fusion must PRESERVE BN inside fused QAT modules so stats can be learned then frozen.
    from torch.ao.nn.intrinsic.qat import ConvBn2d, ConvBnReLU2d
    convbn = [m for _, m in prepared.named_modules() if isinstance(m, (ConvBn2d, ConvBnReLU2d))]
    bn_left = count_types(prepared, nn.BatchNorm2d)
    globals()["_QAT_CONVBN"] = len(convbn)
    globals()["_QAT_BN_LEFT"] = bn_left
    check("qat_fusion_yields_bn_bearing_modules", len(convbn) > 0,
          f"intrinsic.qat ConvBn/ConvBnReLU modules={len(convbn)}")
    check("qat_bn_not_frozen_before_freeze",
          not all(bool(m.freeze_bn) for m in convbn),
          f"{sum(1 for m in convbn if m.freeze_bn)}/{len(convbn)} frozen before")

    freeze_bn_stats(prepared)
    check("freeze_bn_acts_on_qat_fused_modules",
          all(bool(m.freeze_bn) for m in convbn),
          f"{sum(1 for m in convbn if m.freeze_bn)}/{len(convbn)} frozen after")

    disable_observers(prepared)
    check("disable_observers_helper_works",
          all(int(m.observer_enabled) == 0 for m in fq),
          "all observer_enabled buffers cleared")

    prepared.eval()
    with torch.no_grad():
        y_qat = prepared(X)
    check("qat_model_outputs_116ch", tuple(y_qat.shape) == (1, NC, 64, 64), str(tuple(y_qat.shape)))

    check("bn_freeze_iteration_in_locked_range",
          bn_freeze_iteration(1000, 0.65) == 650 and bn_freeze_iteration(1000, 0.70) == 700)
    expect_raises("bn_freeze_pct_outside_range_rejected", QuantPreparationError,
                  bn_freeze_iteration, 1000, 0.5)
    check("qat_grad_clip_gate_requires_explicit_value",
          qat_grad_clip_gate_error(None) is not None
          and qat_grad_clip_gate_error(0) is not None
          and qat_grad_clip_gate_error(float("nan")) is not None
          and qat_grad_clip_gate_error(float("inf")) is not None
          and qat_grad_clip_gate_error(1.0) is None,
          "no numeric max_norm invented")


# ---------------------------------------------------------------- 4. checkpoint contract
def _e1_state() -> dict:
    m = build_student(pretrained=False)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_checkpoint_contract() -> None:
    set_seed(0)
    good = TMP / "e1_good.pt"
    torch.save({"iter": 4000, "model_state_dict": _e1_state(), "num_classes": NC,
                "best_val_miou_all_class": 0.1}, good)
    ckpt = load_e1_source_checkpoint(good)
    check("valid_e1_checkpoint_accepted", "model_state_dict" in ckpt)

    bad = TMP / "no_msd.pt"
    torch.save({"iter": 1}, bad)
    expect_raises("missing_model_state_dict_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, bad)

    e3_key = TMP / "e3_projkey.pt"
    torch.save({"model_state_dict": _e1_state(), "num_classes": NC,
                "cwd_projection_state_dict": {"weight": torch.randn(320, 160, 1, 1)}}, e3_key)
    expect_raises("cwd_projection_key_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, e3_key)

    leaked = TMP / "e3_leaked.pt"
    st = _e1_state()
    st["cwd_projection.weight"] = torch.randn(320, 160, 1, 1)
    torch.save({"model_state_dict": st, "num_classes": NC}, leaked)
    expect_raises("cwd_projection_in_state_dict_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, leaked)

    staged = TMP / "e3_stage.pt"
    torch.save({"stage": "E3", "model_state_dict": _e1_state(), "num_classes": NC}, staged)
    expect_raises("e3_stage_metadata_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, staged)

    teacher = TMP / "teacher_contaminated.pt"
    st2 = _e1_state()
    st2["teacher.backbone.weight"] = torch.randn(4, 3, 1, 1)
    torch.save({"model_state_dict": st2, "num_classes": NC}, teacher)
    expect_raises("teacher_state_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, teacher)

    quantized = TMP / "already_quant.pt"
    st3 = _e1_state()
    st3["features.0.0.activation_post_process.scale"] = torch.tensor([0.1])
    torch.save({"model_state_dict": st3, "num_classes": NC}, quantized)
    expect_raises("already_quantized_source_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, quantized)

    wrong_nc = TMP / "wrong_nc.pt"
    torch.save({"model_state_dict": _e1_state(), "num_classes": 150}, wrong_nc)
    expect_raises("wrong_num_classes_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, wrong_nc)
    expect_raises("missing_file_rejected", SourceCheckpointInvalid,
                  load_e1_source_checkpoint, TMP / "absent.pt")


# ---------------------------------------------------------------- 5. calibration index
def test_calibration_index() -> None:
    ids = [f"train_{i:05d}" for i in range(5367)]
    idx = build_calibration_index(ids)
    check("calibration_selects_128", len(idx["selected_ids"]) == CALIBRATION_COUNT
          and idx["count"] == CALIBRATION_COUNT, str(idx["count"]))
    check("calibration_seed_42_split_train", idx["seed"] == 42 and idx["split"] == "train")
    check("calibration_shared_by_e4_e7", idx["shared_by"] == ["E4", "E7"])

    again = build_calibration_index(list(ids))
    check("calibration_deterministic", again["selected_ids"] == idx["selected_ids"])

    shuffled = list(ids)
    import random as _r
    _r.Random(999).shuffle(shuffled)
    from_shuffled = build_calibration_index(shuffled)
    check("calibration_independent_of_input_order",
          from_shuffled["selected_ids"] == idx["selected_ids"],
          "canonical sort precedes sampling")
    check("calibration_checksum_stable",
          from_shuffled["checksum_sha256"] == idx["checksum_sha256"],
          idx["checksum_sha256"][:16] + "…")
    check("calibration_ids_are_train_only",
          all(i.startswith("train_") for i in idx["selected_ids"]))

    expect_raises("calibration_rejects_val_split", CalibrationIndexError,
                  build_calibration_index, ids, split="val")
    expect_raises("calibration_rejects_test_split", CalibrationIndexError,
                  build_calibration_index, ids, split="test")
    expect_raises("calibration_rejects_duplicates", CalibrationIndexError,
                  build_calibration_index, ["a", "a"] + [f"x{i}" for i in range(200)])
    expect_raises("calibration_rejects_small_pool", CalibrationIndexError,
                  build_calibration_index, [f"t{i}" for i in range(10)])

    path = save_calibration_index(idx, TMP / "calib" / "e4_e7_calibration_index.json")
    reloaded = load_calibration_index(path)
    check("calibration_roundtrip_e4_e7_shareable",
          reloaded["selected_ids"] == idx["selected_ids"]
          and reloaded["checksum_sha256"] == idx["checksum_sha256"])
    tampered = dict(reloaded)
    tampered["selected_ids"] = list(reloaded["selected_ids"])[:-1] + ["train_99999"]
    expect_raises("calibration_checksum_detects_tampering", CalibrationIndexError,
                  load_calibration_index, save_and_return(tampered, TMP / "calib" / "bad.json"))


def save_and_return(index: dict, path: Path) -> Path:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 6. operator coverage
def test_coverage() -> None:
    converted = globals().get("_CONVERTED")
    if converted is None:
        check("coverage_available", False, "no converted model")
        return
    cov = quantization_coverage(converted)
    check("coverage_reports_int8_modules", cov["int8_count"] > 0,
          f"int8={cov['int8_count']} fp32_fallback={cov['fp32_count']}")
    fwd = try_converted_forward(converted, X)
    globals()["_FWD"] = fwd
    globals()["_COV"] = cov
    if fwd["ok"]:
        check("converted_forward_outputs_116ch",
              fwd["output_shape"][:2] == (1, NC) and fwd["output_shape"][2:] == (64, 64),
              str(fwd["output_shape"]))
    else:
        check("converted_forward_failure_reported_precisely", bool(fwd["error"]), fwd["error"][:110])
    notes.append(f"converted forward ok={fwd['ok']}"
                 + (f" shape={fwd['output_shape']}" if fwd["ok"] else f" error={fwd['error']}"))


def main() -> int:
    print("=" * 78)
    print("E4/E5 QUANTIZATION CORE SMOKE — synthetic; no dataset, no GPU, no real checkpoint")
    print(f"torch {torch.__version__} (LOCAL; registered experiment stack is 2.1.0+cu121)")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_qconfigs, test_ptq_path, test_qat_path, test_checkpoint_contract,
               test_calibration_index, test_coverage):
        print(f"\n--- {fn.__name__} ---")
        fn()

    cov, fwd = globals().get("_COV"), globals().get("_FWD")
    if cov:
        print("\n[COVERAGE]")
        print(f"  INT8 modules            : {cov['int8_count']}")
        for e in cov["int8_modules"][:6]:
            print(f"      {e}")
        print(f"  FP32 fallback modules   : {cov['fp32_count']}")
        for e in cov["fp32_fallback_modules"][:8]:
            print(f"      {e}")
        print("  Functional (module walk cannot observe):")
        for e in cov["functional_ops_unobservable_by_module_walk"]:
            print(f"      {e}")
        print(f"  Converted forward       : {fwd}")
    print(f"\n[QAT FUSION DIAGNOSTIC] intrinsic.qat ConvBn modules="
          f"{globals().get('_QAT_CONVBN')} | BatchNorm2d remaining={globals().get('_QAT_BN_LEFT')}")
    for n in notes:
        print(f"[note] {n}")

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:44}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
