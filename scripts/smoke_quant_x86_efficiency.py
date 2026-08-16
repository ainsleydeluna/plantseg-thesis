#!/usr/bin/env python3
"""Synthetic verification of the x86/fbgemm INT8 copy used ONLY for CPU-proxy latency.

No real backend, no dataset, no checkpoint, no timing, no install. Observers and fake-quant modules
are INSTANTIATED and introspected, so nothing is asserted from a helper's name.

This host exposes only `onednn`, so every real x86 path must refuse loudly — that refusal is the
expected result here, not a failure.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from torch.ao.quantization.fake_quantize import FakeQuantize  # noqa: E402

from src.eval.efficiency import EfficiencyError, x86_latency_copy_gate  # noqa: E402
from src.quant import qconfig as qc  # noqa: E402
from src.quant import x86_latency as x86  # noqa: E402
from src.quant.prepare import prepare_qat_model  # noqa: E402
from src.quant.calibration import (CALIBRATION_COUNT, CALIBRATION_SEED,  # noqa: E402
                                   CALIBRATION_SPLIT, build_calibration_index)
from src.quant.stages import QUANT_STAGES, SHARED_CALIBRATION_STAGES  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="smoke_x86_eff_"))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect(name: str, exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no exception raised")
    except exc as e:
        check(name, True, str(getattr(e, "code", type(e).__name__)))
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


class TinyStudent(torch.nn.Module):
    """Stand-in fusable module; the real student's quantization logic is never duplicated here.

    It exposes `fuse(is_qat=...)` because `src.quant.prepare._fused_copy` mandates it — fusion must
    precede observer insertion (contract B4), and that invariant is exercised here rather than
    bypassed.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self, is_qat: bool = False):
        from torch.ao.quantization import fuse_modules, fuse_modules_qat
        self.train(is_qat)
        fuser = fuse_modules_qat if is_qat else fuse_modules
        fuser(self, [["conv", "bn", "relu"]], inplace=True)
        return self


# ---------------------------------------------------------------- 1. qconfig introspection
def test_qconfig() -> None:
    ptq = qc.describe_qconfig(qc.x86_ptq_qconfig())
    qat = qc.describe_qconfig(qc.x86_qat_qconfig())

    check("x86_ptq_activation_reduce_range", ptq["activation"]["reduce_range"] is True,
          f"resolved quant_min/max={ptq['activation']['quant_min']}/"
          f"{ptq['activation']['quant_max']}")
    check("x86_ptq_activation_histogram_quint8",
          ptq["activation"]["observer_class"] == "HistogramObserver"
          and ptq["activation"]["dtype"] == torch.quint8
          and ptq["activation"]["qscheme"] == torch.per_tensor_affine)
    check("x86_ptq_weight_per_channel_symmetric_int8",
          ptq["weight"]["dtype"] == torch.qint8
          and ptq["weight"]["qscheme"] == torch.per_channel_symmetric
          and ptq["weight"]["ch_axis"] == 0,
          "thesis weight scheme preserved on the latency copy")

    check("x86_qat_activation_reduce_range", qat["activation"]["reduce_range"] is True)
    check("x86_qat_activation_is_moving_average_fake_quant",
          qat["activation"]["is_fake_quant"]
          and "MovingAverage" in qat["activation"]["observer_class"]
          and qat["activation"]["dtype"] == torch.quint8
          and qat["activation"]["qscheme"] == torch.per_tensor_affine,
          qat["activation"]["observer_class"])
    check("x86_qat_weight_per_channel_symmetric_int8",
          qat["weight"]["is_fake_quant"] and qat["weight"]["dtype"] == torch.qint8
          and qat["weight"]["qscheme"] == torch.per_channel_symmetric
          and qat["weight"]["ch_axis"] == 0)

    # the accuracy QConfigs are untouched
    acc_ptq = qc.describe_qconfig(qc.ptq_qconfig())
    acc_qat = qc.describe_qconfig(qc.qat_qconfig())
    check("qnnpack_ptq_still_reduce_range_false",
          acc_ptq["activation"]["reduce_range"] is False
          and acc_ptq["activation"]["quant_min"] == 0
          and acc_ptq["activation"]["quant_max"] == 255,
          "accuracy artifact keeps the full 8-bit range")
    check("qnnpack_qat_still_reduce_range_false",
          acc_qat["activation"]["reduce_range"] is False)
    check("qnnpack_weight_scheme_unchanged",
          acc_ptq["weight"]["qscheme"] == torch.per_channel_symmetric
          and acc_qat["weight"]["ch_axis"] == 0)
    check("accuracy_backend_constant_unchanged", qc.QUANT_BACKEND == "qnnpack")
    check("reduce_range_is_the_only_activation_difference",
          acc_ptq["activation"]["observer_class"] == ptq["activation"]["observer_class"]
          and acc_ptq["activation"]["dtype"] == ptq["activation"]["dtype"]
          and acc_ptq["activation"]["qscheme"] == ptq["activation"]["qscheme"]
          and acc_ptq["activation"]["reduce_range"] != ptq["activation"]["reduce_range"])


# ---------------------------------------------------------------- 2. backend selector
def test_backend() -> None:
    supported = list(torch.backends.quantized.supported_engines)
    check("approved_x86_backends", qc.X86_BACKENDS == ("x86", "fbgemm"),
          "contract wording 'fbgemm/x86'; deterministic preference")
    check("onednn_not_approved", "onednn" not in qc.X86_BACKENDS)
    check("qnnpack_not_approved_for_latency", "qnnpack" not in qc.X86_BACKENDS)

    if not ({"x86", "fbgemm"} & set(supported)):
        expect("unavailable_x86_backend_refused", qc.QuantBackendUnavailable, qc.select_x86_backend)
        check("local_host_has_no_x86_backend", True, f"supported_engines={supported}")
    else:  # pragma: no cover - not this host
        engine = qc.select_x86_backend()
        check("selected_engine_takes_effect",
              torch.backends.quantized.engine == engine and engine in qc.X86_BACKENDS, engine)
        check("local_host_has_no_x86_backend", False, "host unexpectedly exposes x86/fbgemm")

    try:
        qc.select_x86_backend()
    except qc.QuantBackendUnavailable as e:
        check("refusal_names_reason",
              "onednn" in str(e) and "qnnpack" in str(e) and "refused" in str(e).lower(),
              "explains why neither substitute is accepted")


# ---------------------------------------------------------------- 3. artifact identity
def test_identity() -> None:
    check("two_roles_distinct",
          x86.ACCURACY_ARTIFACT_ROLE == "accuracy"
          and x86.X86_LATENCY_ARTIFACT_ROLE == "x86_cpu_proxy_latency"
          and x86.ACCURACY_ARTIFACT_ROLE != x86.X86_LATENCY_ARTIFACT_ROLE)

    prov = x86.x86_artifact_provenance(stage="E4", engine="fbgemm", source_sha256="a" * 64,
                                       artifact_path=TMP / "e4_x86.pt",
                                       calibration_checksum="b" * 64)
    check("x86_provenance_is_latency_only",
          prov["artifact_role"] == "x86_cpu_proxy_latency" and prov["usable_for_latency"]
          and prov["usable_for_accuracy"] is False
          and prov["usable_for_size_reporting"] is False,
          "size stays with the QNNPACK artifact (contract line 412)")
    check("x86_provenance_records_reduce_range", prov["reduce_range"] is True)
    check("x86_provenance_records_real_engine", prov["backend"] == "fbgemm")
    check("x86_provenance_has_own_identity",
          prov["source_checkpoint_sha256"] == "a" * 64
          and prov["calibration_checksum_sha256"] == "b" * 64
          and prov["artifact_path"].endswith("e4_x86.pt"))
    check("x86_provenance_no_device_claim",
          prov["cpu_proxy"] is True and prov["on_device"] is False)
    expect("x86_provenance_rejects_qnnpack_engine", x86.X86LatencyCopyError,
           x86.x86_artifact_provenance, stage="E4", engine="qnnpack")
    expect("x86_provenance_rejects_onednn_engine", x86.X86LatencyCopyError,
           x86.x86_artifact_provenance, stage="E4", engine="onednn")


# ---------------------------------------------------------------- 4. stage routing
def test_stages() -> None:
    check("all_int8_stages_reconstructible",
          set(x86.X86_RECONSTRUCTIBLE_STAGES) == {"E4", "E5", "E6", "E7"})
    check("routing_matches_committed_stage_table",
          all(QUANT_STAGES[s.lower()]["method"] == "ptq" for s in x86.X86_PTQ_STAGES)
          and all(QUANT_STAGES[s.lower()]["method"] == "qat" for s in x86.X86_QAT_STAGES),
          "E4/E7 PTQ, E5/E6 QAT")
    check("source_relationships_unchanged",
          QUANT_STAGES["e4"]["source_stage"] == "E1" and QUANT_STAGES["e5"]["source_stage"] == "E1"
          and QUANT_STAGES["e6"]["source_stage"] == "E3"
          and QUANT_STAGES["e7"]["source_stage"] == "E3",
          "E1/E3 sources untouched")

    net = TinyStudent()
    # PTQ stages route to the x86 PTQ configuration (backend selection refused on this host)
    for stage in ("E4", "E7"):
        expect(f"{stage.lower()}_ptq_routes_to_x86_backend", qc.QuantBackendUnavailable,
               x86.build_x86_latency_copy, stage, net)
    # QAT stages refuse without the sidecar, BEFORE any backend question, and never fabricate one
    for stage in ("E5", "E6"):
        expect(f"{stage.lower()}_qat_copy_refused_without_sidecar", x86.X86LatencyCopyError,
               x86.build_x86_latency_copy, stage, net)
    try:
        x86.build_x86_latency_copy("E5", net)
    except x86.X86LatencyCopyError as e:
        check("qat_refusal_names_sidecar_requirement",
              e.code == "qat_sidecar_required" and x86.QAT_SIDECAR_SUFFIX in str(e),
              "sidecar absence fails loudly")

    # the x86 PTQ preparation path itself is real: with backend selection skipped it inserts
    # observers carrying the x86 (reduce_range=True) configuration
    prepared = x86.prepare_x86_ptq(net, select_backend=False)
    obs = [m for m in prepared.modules() if hasattr(m, "reduce_range")]
    check("x86_prepare_inserts_reduce_range_observers",
          bool(obs) and all(bool(m.reduce_range) for m in obs),
          f"{len(obs)} observers, all reduce_range=True")
    check("x86_prepare_fused_not_duplicated",
          any("Conv" in type(m).__name__ and "Bn" in type(m).__name__ for m in prepared.modules())
          or not any(isinstance(m, torch.nn.BatchNorm2d) for m in prepared.modules()),
          "fusion came from src.quant.prepare, not a reimplementation")


# ---------------------------------------------------------------- 4b. E5/E6 QAT translation
def _fake_quants(model):
    """Split FakeQuantize modules into activation vs weight quantizers."""
    act, wt = {}, {}
    for name, mod in model.named_modules():
        if isinstance(mod, FakeQuantize):
            (wt if name.endswith("weight_fake_quant") else act)[name] = mod
    return act, wt


def _synthetic_sidecar(stage: str = "E5"):
    """A realistic pre-convert QAT state: prepared under the ACCURACY qconfig, observers populated."""
    torch.manual_seed(0)
    source = prepare_qat_model(TinyStudent(), select_backend=False)
    source.train()
    with torch.no_grad():
        source(torch.randn(1, 3, 32, 32))      # populate observer ranges + qparams
    state = {k: v.clone() for k, v in source.state_dict().items()}
    sidecar = {"stage": stage, "quantization": x86.QAT_SIDECAR_KIND, "iter": 100,
               "best_val_miou_all_class": 0.1234, "model_state_dict": state,
               "num_classes": 116, "artifact_role": x86.QAT_SIDECAR_ROLE,
               "is_official_accuracy_artifact": False, "is_deployment_artifact": False,
               "training_quant_backend": "qnnpack", "source_stage": "E1"}
    return source, sidecar


def test_qat_translation() -> None:
    source, sidecar = _synthetic_sidecar("E5")
    prepared, report = x86.translate_qat_state_to_x86("E5", TinyStudent(), sidecar,
                                                      select_backend=False)

    # A. SAME TRAINED WEIGHTS — verified before conversion
    check("translated_weights_identical_to_source",
          x86.assert_trained_state_preserved(prepared, sidecar["model_state_dict"]) == [],
          "every non-qparam trained tensor matches the sidecar")
    src_w = sidecar["model_state_dict"]
    live = prepared.state_dict()
    weight_keys = [k for k in src_w if k.endswith("weight") and src_w[k].dim() > 1]
    check("conv_weight_tensors_bitwise_equal",
          bool(weight_keys) and all(torch.equal(live[k], src_w[k]) for k in weight_keys),
          f"{len(weight_keys)} weight tensor(s)")

    # B/F. zero training
    check("zero_optimizer_steps",
          report["optimizer_steps_for_translation"] == 0 and report["gradient_updates"] == 0
          and report["retrained"] is False)
    # AST call analysis, not substring matching: the function's docstring explains that it performs
    # no optimizer step, and its report legitimately contains the KEY
    # "optimizer_steps_for_translation" -- both would defeat a raw text scan.
    tree = ast.parse((REPO / "src/quant/x86_latency.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "translate_qat_state_to_x86")
    called = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = node.func
            called.add(target.attr if isinstance(target, ast.Attribute) else
                       getattr(target, "id", ""))
    training_calls = {"backward", "step", "zero_grad", "SGD", "Adam", "AdamW", "train"}
    touches_optim = any(isinstance(n, ast.Attribute) and n.attr == "optim" for n in ast.walk(fn))
    check("translation_has_no_training_calls",
          not (training_calls & called) and not touches_optim,
          f"no optimizer/backward/step call; calls={sorted(c for c in called if c)[:6]}")
    check("translation_runs_under_no_grad",
          any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "no_grad"
              for d in fn.decorator_list),
          "@torch.no_grad() makes 'no gradient updates' structural")

    # C. QNNPACK activation qparams are NOT reused
    src_act, src_wt = _fake_quants(source)
    new_act, new_wt = _fake_quants(prepared)
    check("activation_qparams_recomputed_not_reused",
          bool(new_act) and all(not torch.equal(new_act[k].scale, src_act[k].scale)
                                for k in new_act),
          f"{len(new_act)} activation quantizer(s) re-derived")
    ratios = [float(new_act[k].scale.max() / src_act[k].scale.max()) for k in new_act]
    check("activation_scale_reflects_reduced_range",
          all(abs(r - 2.0) < 0.05 for r in ratios),
          f"x86 scale / qnnpack scale ~= {ratios[0]:.3f} (255->127 range)")
    check("activation_range_evidence_carried_over",
          all(torch.equal(new_act[k].activation_post_process.min_val,
                          src_act[k].activation_post_process.min_val)
              and torch.equal(new_act[k].activation_post_process.max_val,
                              src_act[k].activation_post_process.max_val) for k in new_act),
          "same learned min/max, different quantizer parameters")

    # 8. reduced x86 activation range under the current semantics
    check("x86_activation_range_is_reduced",
          all((int(m.quant_min), int(m.quant_max)) == (0, 127) for m in new_act.values()),
          "quant_min/quant_max = 0/127")

    # D. weight granularity unchanged, and weight qparams identical (reduce_range is activation-only)
    check("weight_scheme_unchanged_after_translation",
          all(m.qscheme == torch.per_channel_symmetric and m.dtype == torch.qint8
              and int(m.ch_axis) == 0 for m in new_wt.values()),
          f"{len(new_wt)} weight quantizer(s) per-channel symmetric qint8 ch_axis=0")
    check("weight_qparams_unaffected_by_reduce_range",
          all(torch.allclose(new_wt[k].scale, src_wt[k].scale) for k in new_wt),
          "reduce_range governs activations only")

    # refusals
    converted_only = {"stage": "E5", "quantization": "qat", "num_classes": 116, "model": {}}
    expect("converted_only_artifact_rejected", x86.X86LatencyCopyError,
           x86.translate_qat_state_to_x86, "E5", TinyStudent(), converted_only)
    expect("ptq_stage_rejected_by_qat_translator", x86.X86LatencyCopyError,
           x86.translate_qat_state_to_x86, "E4", TinyStudent(), sidecar)
    mismatched = dict(sidecar, stage="E6")
    expect("sidecar_stage_mismatch_rejected", x86.X86LatencyCopyError,
           x86.translate_qat_state_to_x86, "E5", TinyStudent(), mismatched)

    # E6 translates through the same path
    _, sidecar6 = _synthetic_sidecar("E6")
    prepared6, report6 = x86.translate_qat_state_to_x86("E6", TinyStudent(), sidecar6,
                                                        select_backend=False)
    check("e6_translates_from_sidecar",
          x86.assert_trained_state_preserved(prepared6, sidecar6["model_state_dict"]) == []
          and report6["optimizer_steps_for_translation"] == 0)

    # conversion still demands a real x86 engine, so an onednn-packed model can never be emitted
    expect("convert_requires_x86_engine", x86.X86LatencyCopyError, x86.build_x86_latency_copy,
           "E5", TinyStudent(), sidecar=sidecar, select_backend=False)

    # sidecar loading validation
    good = TMP / "e5_qat_state.pt"
    torch.save(sidecar, good)
    loaded = x86.load_qat_sidecar(good)
    check("sidecar_roundtrip", loaded["quantization"] == x86.QAT_SIDECAR_KIND
          and loaded["artifact_role"] == x86.QAT_SIDECAR_ROLE
          and loaded["is_official_accuracy_artifact"] is False)
    bad = TMP / "e5_int8_student.pt"
    torch.save(converted_only, bad)
    expect("converted_artifact_not_loadable_as_sidecar", x86.X86LatencyCopyError,
           x86.load_qat_sidecar, bad)
    expect("absent_sidecar_fails_loudly", x86.X86LatencyCopyError, x86.load_qat_sidecar,
           TMP / "nope.pt")
    conflict = TMP / "conflict.pt"
    torch.save(dict(sidecar, is_deployment_artifact=True), conflict)
    expect("sidecar_role_conflict_rejected", x86.X86LatencyCopyError, x86.load_qat_sidecar,
           conflict)


# ---------------------------------------------------------------- 4c. runner sidecar contract
def test_runner_sidecar() -> None:
    src = (REPO / "src/quant/runner.py").read_text(encoding="utf-8")
    check("runner_writes_sidecar",
          "QAT_SIDECAR_SUFFIX" in src and "QAT_SIDECAR_KIND" in src
          and "model_state_dict" in src)
    check("sidecar_marked_auxiliary",
          '"is_official_accuracy_artifact": False' in src
          and '"is_deployment_artifact": False' in src
          and "QAT_SIDECAR_ROLE" in src)
    check("official_converted_schema_unchanged",
          '"quantization": "qat", "num_classes": NUM_CLASSES,' in src
          and '"model": converted.state_dict()' in src,
          "existing E5/E6 consumers keep the same primary artifact")
    check("selection_semantics_preserved",
          "if stopper.update(all_miou, it):" in src
          and "best_state = copy.deepcopy(prepared.state_dict())" in src
          and "prepared.load_state_dict(best_state)" in src,
          "best all-class validation mIoU selection untouched")
    check("sidecar_path_recorded_in_provenance", '"qat_state_artifact": str(qat_path)' in src,
          "pre-existing provenance field already carries it; no duplicate key added")


# ---------------------------------------------------------------- 5. calibration identity
def test_calibration() -> None:
    index = build_calibration_index([f"train_{i:04d}" for i in range(500)])
    path = TMP / "calibration_index.json"
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")

    loaded = x86.require_x86_calibration_identity(path, index["checksum_sha256"])
    check("x86_reuses_frozen_calibration",
          loaded["split"] == CALIBRATION_SPLIT and loaded["count"] == CALIBRATION_COUNT
          and loaded["seed"] == CALIBRATION_SEED,
          f"split={CALIBRATION_SPLIT} count={CALIBRATION_COUNT} seed={CALIBRATION_SEED}")
    check("calibration_shared_by_e4_and_e7",
          set(SHARED_CALIBRATION_STAGES) <= set(loaded.get("shared_by", [])),
          str(loaded.get("shared_by")))
    check("calibration_checksum_preserved",
          loaded["checksum_sha256"] == index["checksum_sha256"], "identity frozen, never resampled")
    expect("wrong_calibration_checksum_refused", Exception,
           x86.require_x86_calibration_identity, path, "0" * 64)

    # the x86 copy must not resample: the selected identifiers are byte-identical
    again = build_calibration_index([f"train_{i:04d}" for i in range(500)])
    check("calibration_selection_deterministic",
          again["selected_ids"] == index["selected_ids"]
          and again["checksum_sha256"] == index["checksum_sha256"])


# ---------------------------------------------------------------- 6. efficiency integration
def test_efficiency_integration() -> None:
    # QNNPACK timing can never be reported as the registered x86 measurement
    expect("qnnpack_timing_not_reportable_as_x86", EfficiencyError, x86_latency_copy_gate, "E4")
    try:
        x86_latency_copy_gate("E4")
    except EfficiencyError as e:
        check("x86_gate_refuses_without_backend", e.code == "x86_backend_unavailable",
              "no approved x86 engine on this host")
    # SUPERSEDED: E5/E6 used to be categorically blocked at this gate. They are now reconstructible
    # by sidecar translation, so the gate must let them through the STAGE check and refuse only on
    # the backend — the sidecar requirement is enforced where the copy is actually built.
    expect("x86_gate_refuses_qat_stage_without_backend", EfficiencyError,
           x86_latency_copy_gate, "E5")
    try:
        x86_latency_copy_gate("E6")
    except EfficiencyError as e:
        check("x86_gate_admits_qat_stage_then_checks_backend",
              e.code == "x86_backend_unavailable", e.code)
    expect("x86_gate_refuses_non_int8_stage", EfficiencyError, x86_latency_copy_gate, "E1")

    check("onednn_never_official_x86_evidence",
          "onednn" not in qc.X86_BACKENDS
          and torch.backends.quantized.engine != "x86",
          f"local engines={list(torch.backends.quantized.supported_engines)}")

    runner = (REPO / "scripts/profile_efficiency.py").read_text(encoding="utf-8")
    check("size_not_switched_to_latency_copy",
          "size_measured_from_latency_copy" in runner and "ARTIFACT_ROLE_ACCURACY" in runner,
          "serialized size stays with the QNNPACK accuracy artifact")


def main() -> int:
    print("=" * 78)
    print("X86 EFFICIENCY QUANT SMOKE — synthetic; no backend, no dataset, no checkpoint, no timing")
    print(f"torch {torch.__version__} | engines {list(torch.backends.quantized.supported_engines)}")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_qconfig, test_backend, test_identity, test_stages, test_qat_translation,
               test_runner_sidecar, test_calibration, test_efficiency_integration):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:46}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    print("\nNOTE: this host exposes only 'onednn', so every real x86 conversion refuses by design. "
          "E4/E7 rebuild from FP32 + the frozen calibration subset; E5/E6 translate from the "
          "pre-convert QAT sidecar with zero optimizer steps. Real x86 timing needs an x86 host.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
