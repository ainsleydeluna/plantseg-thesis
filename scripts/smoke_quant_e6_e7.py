#!/usr/bin/env python3
"""Synthetic verification of the E6 (QAT from E3) / E7 (PTQ from E3) scaffold. No GPU, no dataset.

E6 and E7 are the E5 and E4 mechanisms pointed at a different FP32 source, so this suite checks the
ROUTING and the E3 source contract rather than re-testing PTQ/QAT internals (covered by
scripts/smoke_quant_e4_e5.py). All checkpoints are synthetic and written to a temp dir outside the
repository; no real E3 checkpoint exists yet, which blocks real E6/E7 execution but not this.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.distill.export import CWD_PROJECTION_KEY  # noqa: E402
from src.models.student import build_student  # noqa: E402
from src.quant import (CalibrationIndexError, QuantStageError, SourceCheckpointInvalid,  # noqa: E402
                       build_calibration_index, describe_qconfig, load_source_for_stage,
                       load_student_from_e3, prepare_for_stage, ptq_qconfig, qat_qconfig,
                       require_shared_calibration_index, resolve_quant_stage,
                       save_calibration_index)
from src.seeds import set_seed  # noqa: E402

NC = 116
TMP = Path(tempfile.mkdtemp(prefix="smoke_quant_e6e7_"))
X = torch.randn(1, 3, 64, 64)
results: list[tuple[str, bool, str]] = []


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


def _student_state() -> dict:
    return {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}


def _backend_available() -> bool:
    return "qnnpack" in torch.backends.quantized.supported_engines


def write(name: str, payload) -> Path:
    p = TMP / name
    torch.save(payload, p)
    return p


# ---------------------------------------------------------------- 1. stage routing
def test_stage_routing() -> None:
    e4, e5, e6, e7 = (resolve_quant_stage(s) for s in ("e4", "e5", "e6", "e7"))
    check("e4_is_ptq_from_e1", e4["method"] == "ptq" and e4["source_stage"] == "E1")
    check("e5_is_qat_from_e1", e5["method"] == "qat" and e5["source_stage"] == "E1")
    check("e6_is_qat_from_e3", e6["method"] == "qat" and e6["source_stage"] == "E3",
          e6["objective"])
    check("e7_is_ptq_from_e3", e7["method"] == "ptq" and e7["source_stage"] == "E3",
          e7["objective"])
    check("e6_mirrors_e5_method", e6["method"] == e5["method"])
    check("e7_mirrors_e4_method", e7["method"] == e4["method"])
    expect_raises("unknown_stage_rejected", QuantStageError, resolve_quant_stage, "e8")

    # E6 must use the SAME explicit QAT qconfig as E5; E7 the SAME PTQ qconfig as E4.
    qat_a, qat_b = describe_qconfig(qat_qconfig()), describe_qconfig(qat_qconfig())
    ptq_a, ptq_b = describe_qconfig(ptq_qconfig()), describe_qconfig(ptq_qconfig())
    check("e6_uses_same_qat_qconfig_as_e5", qat_a == qat_b
          and qat_a["weight"]["qscheme"] == torch.per_channel_symmetric
          and qat_a["activation"]["quant_max"] == 255)
    check("e7_uses_same_ptq_qconfig_as_e4", ptq_a == ptq_b
          and ptq_a["activation"]["observer_class"] == "HistogramObserver"
          and ptq_a["weight"]["qscheme"] == torch.per_channel_symmetric)

    # no distillation machinery may be reachable from the quantization stages
    src = "".join((REPO / "src" / "quant" / f).read_text(encoding="utf-8")
                  for f in ("stages.py", "prepare.py", "qconfig.py"))
    for tok in ("logit_kd", "cwd_channelwise", "FrozenTeacher", "lambda_logit", "T_logit"):
        check(f"no_distillation_in_quant_{tok}", tok not in src)


# ---------------------------------------------------------------- 2. E3 source contract
def test_e3_source_contract() -> None:
    set_seed(0)
    good = write("e3_good.pt", {"stage": "E3", "iter": 4000, "num_classes": NC,
                                "model_state_dict": _student_state(),
                                CWD_PROJECTION_KEY: {"weight": torch.randn(320, 160, 1, 1)}})
    model, ckpt = load_student_from_e3(good)
    check("e3_projection_free_source_accepted",
          isinstance(model, nn.Module) and ckpt["stage"] == "E3")
    check("training_only_projection_present_but_ignored",
          ckpt[CWD_PROJECTION_KEY] is not None
          and not any("cwd" in k for k in model.state_dict()),
          "separate field validated then never loaded into the student")
    with torch.no_grad():
        model.eval()
        y = model(X)
    check("e3_source_student_is_116_class", tuple(y.shape) == (1, NC, 64, 64), str(tuple(y.shape)))

    e1 = write("e1_for_e6.pt", {"iter": 80000, "num_classes": NC,
                                "model_state_dict": _student_state()})
    expect_raises("e1_source_rejected_for_e6_e7", SourceCheckpointInvalid,
                  load_student_from_e3, e1)
    e2 = write("e2_for_e6.pt", {"stage": "E2", "num_classes": NC,
                                "model_state_dict": _student_state()})
    expect_raises("e2_source_rejected_for_e6_e7", SourceCheckpointInvalid,
                  load_student_from_e3, e2)

    leaked_state = _student_state()
    leaked_state["cwd_projection.weight"] = torch.randn(320, 160, 1, 1)
    leaked = write("e3_leaked.pt", {"stage": "E3", "num_classes": NC,
                                    "model_state_dict": leaked_state})
    expect_raises("projection_inside_model_state_dict_rejected", SourceCheckpointInvalid,
                  load_student_from_e3, leaked)

    tstate = _student_state()
    tstate["teacher.backbone.weight"] = torch.randn(4, 3, 1, 1)
    expect_raises("teacher_state_rejected", SourceCheckpointInvalid, load_student_from_e3,
                  write("e3_teacher.pt", {"stage": "E3", "num_classes": NC,
                                          "model_state_dict": tstate}))
    qstate = _student_state()
    qstate["features.0.0.activation_post_process.scale"] = torch.tensor([0.1])
    expect_raises("already_quantized_rejected", SourceCheckpointInvalid, load_student_from_e3,
                  write("e3_quant.pt", {"stage": "E3", "num_classes": NC,
                                        "model_state_dict": qstate}))
    expect_raises("wrong_num_classes_rejected", SourceCheckpointInvalid, load_student_from_e3,
                  write("e3_nc.pt", {"stage": "E3", "num_classes": 150,
                                     "model_state_dict": _student_state()}))
    expect_raises("missing_stage_metadata_rejected", SourceCheckpointInvalid, load_student_from_e3,
                  write("e3_nostage.pt", {"num_classes": NC,
                                          "model_state_dict": _student_state()}))
    expect_raises("missing_file_rejected", SourceCheckpointInvalid,
                  load_student_from_e3, TMP / "absent.pt")


# ---------------------------------------------------------------- 3. E7 shared calibration
def test_shared_calibration() -> None:
    pool = [f"train_{i:05d}" for i in range(5367)]
    e4_index = build_calibration_index(pool)
    e4_path = save_calibration_index(e4_index, TMP / "calib" / "e4_calibration_index.json")

    reused = require_shared_calibration_index(e4_path,
                                              expected_checksum=e4_index["checksum_sha256"])
    check("e7_consumes_exact_e4_index",
          reused["selected_ids"] == e4_index["selected_ids"]
          and reused["checksum_sha256"] == e4_index["checksum_sha256"],
          e4_index["checksum_sha256"][:16] + "…")
    check("index_is_marked_shared_by_e4_and_e7", set(reused["shared_by"]) == {"E4", "E7"})

    # an INDEPENDENTLY resampled list (different candidate pool) must not be silently accepted
    other = build_calibration_index([f"train_{i:05d}" for i in range(1000, 6367)])
    other_path = save_calibration_index(other, TMP / "calib" / "resampled_index.json")
    check("resampled_index_differs", other["checksum_sha256"] != e4_index["checksum_sha256"])
    expect_raises("independently_resampled_index_rejected", CalibrationIndexError,
                  require_shared_calibration_index, other_path,
                  expected_checksum=e4_index["checksum_sha256"])

    # a tampered artifact fails its own checksum
    import json
    tampered = dict(e4_index)
    tampered["selected_ids"] = list(e4_index["selected_ids"])[:-1] + ["train_99999"]
    tpath = TMP / "calib" / "tampered.json"
    tpath.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")
    expect_raises("altered_index_rejected", CalibrationIndexError,
                  require_shared_calibration_index, tpath)

    # an index not declaring the shared stages is refused for E7
    narrowed = dict(e4_index)
    narrowed["shared_by"] = ["E4"]
    npath = TMP / "calib" / "not_shared.json"
    npath.write_text(json.dumps(narrowed, indent=2, sort_keys=True), encoding="utf-8")
    expect_raises("non_shared_index_rejected", CalibrationIndexError,
                  require_shared_calibration_index, npath)
    check("calibration_never_uses_val_or_test",
          all(i.startswith("train_") for i in reused["selected_ids"]))


# ---------------------------------------------------------------- 4. E6/E7 preparation routing
def test_stage_preparation() -> None:
    set_seed(42)
    good = TMP / "e3_good.pt"
    model, _ = load_source_for_stage("e6", good)
    prep_e6 = prepare_for_stage("e6", model, select_backend=_backend_available())
    from torch.ao.nn.intrinsic.qat import ConvBn2d, ConvBnReLU2d
    from torch.ao.quantization.fake_quantize import FakeQuantize
    convbn = [m for m in prep_e6.modules() if isinstance(m, (ConvBn2d, ConvBnReLU2d))]
    fq = [m for m in prep_e6.modules() if isinstance(m, FakeQuantize)]
    check("e6_prepares_as_qat", len(fq) > 0 and len(convbn) > 0,
          f"{len(fq)} FakeQuantize, {len(convbn)} QAT ConvBn modules")
    check("e6_student_has_no_projection",
          not any("cwd" in k for k in prep_e6.state_dict()))

    model2, _ = load_source_for_stage("e7", good)
    prep_e7 = prepare_for_stage("e7", model2, select_backend=_backend_available())
    obs = [n for n, _ in prep_e7.named_modules() if "activation_post_process" in n]
    check("e7_prepares_as_ptq",
          len(obs) > 0 and not any(isinstance(m, FakeQuantize) for m in prep_e7.modules()),
          f"{len(obs)} observers, no FakeQuantize")

    # real-run launch stays gated: no entry point here can start training or calibration
    from src.quant import qat_grad_clip_gate_error
    check("e6_real_run_requires_explicit_clip",
          qat_grad_clip_gate_error(None) is not None and qat_grad_clip_gate_error(1.0) is None)
    # Any non-smoke e6/e7 script must be a stage-PINNED runner behind the real-run gates. (Before
    # the runners existed this asserted that none existed at all; that is now superseded.)
    scripts = sorted(set((REPO / "scripts").glob("*e6*")) | set((REPO / "scripts").glob("*e7*")))
    launchers = [p for p in scripts if "smoke" not in p.name]
    # The wrappers are thin: they pin STAGE and delegate to src/quant/runner.py, which owns the
    # --real-run/--confirm-real-run gates (verified behaviourally in scripts/smoke_quant_runners.py).
    guarded = []
    for p in launchers:
        text = p.read_text(encoding="utf-8")
        if "STAGE =" in text and "--stage" not in text and "src.quant.runner" in text:
            guarded.append(p)
    check("e6_e7_launchers_are_gated_and_stage_pinned",
          len(guarded) == len(launchers),
          f"{[p.name for p in launchers]} — {len(guarded)} pinned to the gated shared runner, "
          "stage not selectable by CLI")


def main() -> int:
    print("=" * 78)
    print("E6/E7 SCAFFOLD SMOKE — synthetic; no GPU, no dataset, no real E3 checkpoint")
    print(f"torch {torch.__version__} | temp {TMP}")
    print("=" * 78)
    for fn in (test_stage_routing, test_e3_source_contract, test_shared_calibration,
               test_stage_preparation):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:44}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
