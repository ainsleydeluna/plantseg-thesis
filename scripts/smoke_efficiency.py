#!/usr/bin/env python3
"""Synthetic verification of the descriptive efficiency/profiling surface.

No PlantSeg dataset, no GPU, no real checkpoint, no install. Statistics are verified as PURE
REDUCERS against known numbers, so nothing here depends on real wall-clock timing — the protocol
(20 warm-up + exactly 100 measured forwards) is asserted, the timings themselves are not.
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.profile_efficiency import METRICS, build_parser, parse_metrics  # noqa: E402
from src.eval import efficiency as eff  # noqa: E402
from src.eval.efficiency import EfficiencyError  # noqa: E402
from src.eval.stage_artifacts import StageArtifactError, resolve_stage_artifact  # noqa: E402
from src.eval.stage_artifacts import resolve_evaluation_source  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="smoke_efficiency_"))
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


def code_only(src: str) -> str:
    """Source minus docstrings/comments — prose must not satisfy or break a code-level claim."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return ast.unparse(tree)


class TinyNet(torch.nn.Module):
    """A cheap stand-in architecture: real nn.Module, trivial cost at 512x512."""

    def __init__(self, ch: int = 4):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, ch, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class FakePackedNet(torch.nn.Module):
    """Simulates a converted INT8 module: packed params, almost no visible nn.Parameters."""

    def __init__(self):
        super().__init__()
        self._packed_params = object()


# ---------------------------------------------------------------- 1. resolution
def test_resolution() -> None:
    kinds = {s: resolve_stage_artifact(s)["kind"] for s in eff.STAGES}
    check("all_stages_route_through_existing_resolver", len(kinds) == 8,
          f"{sorted(set(kinds.values()))}")
    check("int8_stages_identified",
          all(kinds[s] == "int8_artifact" for s in eff.INT8_STAGES),
          "E4-E7 are INT8 artifacts")
    check("fp32_and_teacher_kinds_distinct",
          kinds["E1"] == "fp32_checkpoint" and kinds["teacher"] == "teacher_checkpoint")

    # source TYPES are not confused: an INT8 stage refuses a raw checkpoint, FP32 refuses none
    expect("int8_stage_refuses_raw_checkpoint", StageArtifactError, resolve_evaluation_source,
           "E4", checkpoint=str(TMP / "e4.pt"))
    expect("fp32_stage_requires_checkpoint", StageArtifactError, resolve_evaluation_source, "E1")
    expect("teacher_requires_checkpoint", StageArtifactError, resolve_evaluation_source, "teacher")
    expect("unknown_stage_rejected", EfficiencyError, eff.provenance, stage="E9",
           model_role="student", precision="fp32", artifact_role=eff.ARTIFACT_ROLE_ACCURACY,
           backend=None)


# ---------------------------------------------------------------- 2. parameters
def test_parameters() -> None:
    net = TinyNet()
    counts = eff.architecture_parameter_count(net)
    manual = sum(p.numel() for p in net.parameters())
    check("parameter_count_matches_module", counts["total"] == manual and manual > 0, str(manual))
    check("parameter_count_source_recorded", counts["counted_from"] == "model")

    packed = FakePackedNet()
    check("quantized_model_detected", eff.is_quantized(packed))
    check("visible_params_would_undercount", sum(p.numel() for p in packed.parameters()) == 0,
          "packed INT8 exposes no nn.Parameters")
    expect("int8_count_refuses_without_fp32_reference", EfficiencyError,
           eff.architecture_parameter_count, packed)
    from_ref = eff.architecture_parameter_count(packed, fp32_reference=net)
    check("int8_count_uses_fp32_architecture",
          from_ref["total"] == manual and from_ref["counted_from"] == "fp32_reference_architecture",
          "quantization changes storage, not the learned parameter set")

    # the same student architecture yields the same count for every student stage
    counts_by_stage = {s: eff.architecture_parameter_count(TinyNet())["total"]
                       for s in ("E1", "E2", "E3", "E4", "E5", "E6", "E7")}
    check("student_parameter_count_stable_across_stages",
          len(set(counts_by_stage.values())) == 1, f"{manual} for all 7 student stages")


# ---------------------------------------------------------------- 3. size / byte accounting
def test_size() -> None:
    # pure reducer against hand-computed numbers
    f = eff.footprint_from_counts(1000, 0, 10)
    check("int8_byte_rule", f["int8_bytes"] == 1000 and f["metadata_bytes"] == 40
          and f["theoretical_weight_bytes"] == 1040, str(f["theoretical_weight_bytes"]))
    check("fully_quantized_flagged", f["fully_quantized"] and not f["mixed_precision_remains"])

    fp32 = eff.footprint_from_counts(0, 1000, 10)
    check("fp32_byte_rule", fp32["theoretical_weight_bytes"] == 4000 + 40)

    mixed = eff.footprint_from_counts(600, 400, 10)
    check("mixed_precision_partition_reported",
          mixed["int8_bytes"] == 600 and mixed["fp32_bytes"] == 1600
          and mixed["mixed_precision_remains"] and not mixed["fully_quantized"])
    check("no_flat_4x_claim",
          abs(mixed["compression_vs_fp32"] - 4.0) > 0.1
          and abs(f["compression_vs_fp32"] - 4.0) < 0.25,
          f"mixed={mixed['compression_vs_fp32']:.3f}x, fully-int8={f['compression_vs_fp32']:.3f}x")
    expect("negative_counts_refused", EfficiencyError, eff.footprint_from_counts, -1, 0, 0)

    # live FP32 module partitions into weights + bias metadata
    live = eff.weight_footprint(TinyNet())
    check("live_fp32_footprint", live["fp32_weight_elements"] > 0
          and live["int8_weight_elements"] == 0 and live["metadata_elements"] > 0)

    # physical serialized size of both a synthetic FP32 and a synthetic INT8 artifact
    fp32_path, int8_path = TMP / "fp32.pt", TMP / "int8.pt"
    torch.save({"model_state_dict": TinyNet().state_dict(), "num_classes": 116}, fp32_path)
    torch.save({"stage": "E4", "quantization": "ptq", "num_classes": 116,
                "model": {"w": torch.zeros(64, dtype=torch.qint8 if False else torch.int8)}},
               int8_path)
    a_fp32, a_int8 = eff.serialized_artifact(fp32_path), eff.serialized_artifact(int8_path)
    check("serialized_sizes_measured",
          a_fp32["serialized_artifact_bytes"] > 0 and a_int8["serialized_artifact_bytes"] > 0,
          f"fp32={a_fp32['serialized_artifact_bytes']}B int8={a_int8['serialized_artifact_bytes']}B")
    check("artifact_identity_recorded",
          len(a_fp32["sha256"]) == 64 and a_fp32["sha256"] != a_int8["sha256"])
    check("size_units_explicit", "bytes canonical" in a_fp32["size_convention"])
    expect("missing_artifact_refused", EfficiencyError, eff.serialized_artifact, TMP / "nope.pt")


# ---------------------------------------------------------------- 4. MACs / FLOPs
def test_flops() -> None:
    net = TinyNet()

    def fake_counter(model, inputs):
        return 1_000_000, {"aten::sigmoid": 4, "aten::add": 2}, "synthetic-0.0"

    out = eff.profile_macs(net, counter=fake_counter)
    check("macs_and_flops_separated",
          out["macs"] == 1_000_000 and out["flops_2x_mac"] == 2_000_000,
          "FLOPs = 2 x MACs, fvcore total IS the MAC count")
    check("flop_convention_documented", "one fused multiply-add" in out["convention"])
    check("unsupported_ops_retained",
          out["unsupported_flop_ops"] == {"aten::sigmoid": 4, "aten::add": 2},
          "elementwise ops reported, not absorbed by custom handlers")
    check("flops_input_shape_fixed", out["input_shape"] == [1, 3, 512, 512])
    check("flops_profiled_from_fp32", out["profiled_precision"] == "fp32")

    expect("wrong_input_shape_refused", EfficiencyError, eff.profile_macs, net,
           input_shape=(1, 3, 256, 256), counter=fake_counter)
    expect("quantized_model_refused_for_flops", EfficiencyError, eff.profile_macs,
           FakePackedNet(), counter=fake_counter)

    # real counter path: fvcore is absent here and must defer, never install
    try:
        eff.profile_macs(net)
        check("fvcore_absence_defers_cleanly", False, "fvcore unexpectedly present")
    except EfficiencyError as e:
        check("fvcore_absence_defers_cleanly", e.code == "fvcore_unavailable"
              and "NEEDS OFFICIAL PROFILING ENVIRONMENT" in str(e), e.code)


# ---------------------------------------------------------------- 5. latency
def test_latency() -> None:
    # reducer against known observations: 1..100 ms
    samples = [i / 1000.0 for i in range(1, 101)]
    st = eff.latency_statistics(samples)
    check("median_correct", abs(st["latency_median_ms"] - 50.5) < 1e-9, str(st["latency_median_ms"]))
    check("quartiles_correct",
          abs(st["latency_q1_ms"] - 25.75) < 1e-9 and abs(st["latency_q3_ms"] - 75.25) < 1e-9)
    check("iqr_correct", abs(st["latency_iqr_ms"] - 49.5) < 1e-9, str(st["latency_iqr_ms"]))
    check("p95_correct", abs(st["latency_p95_ms"] - 95.05) < 1e-9, str(st["latency_p95_ms"]))
    check("percentile_method_documented", "linear" in st["percentile_method"])
    check("timing_not_claimed_deterministic", st["deterministic"] is False)
    check("cpu_proxy_label_mandatory", st["cpu_proxy"] is True and st["on_device"] is False)
    expect("wrong_sample_count_refused", EfficiencyError, eff.latency_statistics, samples[:99])
    expect("negative_sample_refused", EfficiencyError, eff.latency_statistics,
           [-1.0] + samples[1:])

    # protocol enforcement + real measurement path with an injected sampler
    seen = {}

    def sampler(model, inputs, measured):
        seen["measured"] = measured
        seen["eval"] = not model.training
        seen["shape"] = tuple(inputs.shape)
        return [0.001] * measured

    net = TinyNet()
    net.train()
    out = eff.measure_latency(net, sampler=sampler, thread_count=2)
    check("exactly_100_measured", seen["measured"] == 100 and out["latency_samples"] == 100)
    check("exactly_20_warmups", out["warmup_forwards"] == 20)
    check("eval_mode_forced", seen["eval"] and out["eval_mode"], "model.eval() before measurement")
    check("inference_mode_and_amp_off", out["inference_mode"] and out["amp"] is False)
    check("batch_and_shape_fixed",
          seen["shape"] == (1, 3, 512, 512) and out["batch_size"] == 1)
    check("thread_count_captured", out["thread_count"] == 2, str(out["thread_count"]))
    expect("protocol_violation_refused", EfficiencyError, eff.measure_latency, net,
           measured=10, sampler=sampler)

    # the default torch.utils.benchmark path really runs
    real = eff.measure_latency(TinyNet(), sampler=None, thread_count=2)
    check("torch_benchmark_path_runs",
          real["timer"] == "torch.utils.benchmark" and real["latency_samples"] == 100
          and real["latency_median_ms"] > 0,
          f"median={real['latency_median_ms']:.3f} ms (development evidence only)")


# ---------------------------------------------------------------- 6. INT8 artifact roles
def test_int8_roles() -> None:
    check("two_roles_registered",
          eff.ARTIFACT_ROLES == ("accuracy", "x86_cpu_proxy_latency"))
    check("accuracy_artifact_is_qnnpack",
          eff.ACCURACY_BACKEND == "qnnpack" and eff.ACCURACY_REDUCE_RANGE is False)
    check("x86_latency_copy_is_separate",
          eff.X86_LATENCY_BACKEND == "fbgemm" and eff.X86_LATENCY_REDUCE_RANGE is True,
          "contract line 412: reduce_range=True for the x86 proxy")

    eff.validate_artifact_role("accuracy", "qnnpack")
    expect("accuracy_role_rejects_other_backend", EfficiencyError, eff.validate_artifact_role,
           "accuracy", "fbgemm")
    expect("latency_role_rejects_qnnpack_copy", EfficiencyError, eff.validate_artifact_role,
           "x86_cpu_proxy_latency", "qnnpack")
    expect("unknown_role_rejected", EfficiencyError, eff.validate_artifact_role, "whatever",
           "fbgemm")

    # the x86 latency copy is NOT improvised, and says exactly what is missing
    try:
        eff.x86_latency_copy_gate()
        check("x86_copy_not_improvised", False, "gate did not refuse")
    except EfficiencyError as e:
        check("x86_copy_not_improvised",
              e.code == "x86_latency_copy_unavailable" and "src.quant.qconfig" in str(e),
              "names the missing src.quant interface")

    # local ONEDNN can never stand in for an official backend
    supported = list(torch.backends.quantized.supported_engines)
    check("local_backends_recorded", isinstance(supported, list), str(supported))
    if "qnnpack" not in supported:
        expect("unavailable_backend_refused_loudly", EfficiencyError,
               eff.require_quantized_backend, "qnnpack")
    if "fbgemm" not in supported:
        expect("x86_backend_unavailable_refused", EfficiencyError,
               eff.require_quantized_backend, "fbgemm")
    check("onednn_is_not_official_x86_evidence",
          "onednn" not in (eff.ACCURACY_BACKEND, eff.X86_LATENCY_BACKEND),
          "onednn is neither the accuracy nor the registered x86 proxy backend")


# ---------------------------------------------------------------- 7. memory
def test_memory() -> None:
    m = eff.memory_delta(1000, 1500)
    check("memory_delta_arithmetic", m["peak_rss_delta_bytes"] == 500 and not m["negative_delta"])
    check("memory_units_explicit", m["units"] == "bytes" and "rss" in m["source"].lower())
    check("memory_platform_recorded", isinstance(m["platform"], str) and m["platform"])

    neg = eff.memory_delta(2000, 1500)
    check("negative_delta_reported_not_clamped",
          neg["peak_rss_delta_bytes"] == -500 and neg["negative_delta"] and neg["clamped"] is False,
          "explicit design: report and flag, never silently clamp to zero")
    expect("negative_absolute_refused", EfficiencyError, eff.memory_delta, -1, 10)

    check("injected_sampler_used", eff.rss_bytes(sampler=lambda: 4242) == 4242)
    check("real_rss_available", eff.rss_bytes() > 0, "psutil RSS (development evidence)")


# ---------------------------------------------------------------- 8. scientific role
def test_scientific_role() -> None:
    src = code_only((REPO / "src/eval/efficiency.py").read_text(encoding="utf-8"))
    runner = code_only((REPO / "scripts/profile_efficiency.py").read_text(encoding="utf-8"))
    banned = ("wilcoxon", "ttest", "t_test", "holm", "bootstrap", "bca", "noninferiority",
              "non_inferiority", "p_value", "pvalue")
    check("no_statistical_test_reachable",
          all(b not in src.lower() for b in banned) and all(b not in runner.lower() for b in banned),
          "efficiency never enters an inferential family")
    check("no_dataset_access",
          all(t not in runner for t in ("PlantSegDataset", "build_dataloader", "build_eval_loader",
                                        "PLANTSEG_DATA_ROOT")),
          "efficiency needs no PlantSeg image")

    rec = eff.efficiency_record(
        provenance_block=eff.provenance(stage="E1", model_role="student", precision="fp32",
                                        artifact_role="accuracy", backend=None),
        parameters={"total": 1}, deferred={"flops": "[fvcore_unavailable] deferred"})
    check("record_labelled_descriptive",
          rec["measurement_role"] == "descriptive" and rec["cpu_proxy"] is True
          and rec["on_device"] is False and rec["no_arm_claim"] is True)
    check("record_uses_own_schema", rec["schema"] == "plantseg-efficiency/1.0.0",
          "frozen evaluation/statistics schemas are not widened")
    check("provenance_identifies_artifact",
          {"stage", "model_role", "precision", "artifact_role", "backend",
           "source_artifact_path", "source_artifact_sha256", "input_shape", "batch_size",
           "environment"} <= set(rec["provenance"]),
          "no pooling of two unidentified artifacts")
    env = rec["provenance"]["environment"]
    check("environment_captured",
          {"torch", "torchvision", "python", "platform", "thread_count",
           "quant_backends_available"} <= set(env),
          f"torch {env['torch']} (NOT the pinned official stack)")
    check("deferred_reasons_visible", "fvcore_unavailable" in rec["deferred"]["flops"])
    check("throughput_and_energy_absent",
          "throughput" not in src.lower() and "energy" not in src.lower(),
          "only contract-registered metrics implemented")


# ---------------------------------------------------------------- 9. CLI
def test_cli() -> None:
    p = build_parser()
    ns = p.parse_args(["--stage", "E1", "--out-dir", str(TMP / "out")])
    check("cli_defaults_all_metrics", parse_metrics(ns.metrics) == METRICS,
          str(parse_metrics(ns.metrics)))
    check("cli_default_role_is_accuracy", ns.artifact_role == "accuracy")
    expect("cli_unknown_metric_refused", Exception, parse_metrics, "params,bogus")
    try:
        p.parse_args(["--stage", "E9", "--out-dir", str(TMP / "out")])
        check("cli_unknown_stage_refused", False, "parser accepted E9")
    except SystemExit:
        check("cli_unknown_stage_refused", True, "argparse choices")

    from scripts.profile_efficiency import main as profile_main
    rc = profile_main(["--stage", "E1", "--out-dir", str(REPO / "inside")])
    check("cli_refuses_output_inside_repo", rc == 2, "artifacts never written into the repository")
    check("no_output_written_inside_repo", not (REPO / "inside").exists())


def main() -> int:
    print("=" * 78)
    print("EFFICIENCY SMOKE — synthetic only; no dataset, no checkpoint, no GPU, no install")
    print(f"torch {torch.__version__} | backends {list(torch.backends.quantized.supported_engines)}")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_resolution, test_parameters, test_size, test_flops, test_latency,
               test_int8_roles, test_memory, test_scientific_role, test_cli):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:46}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    print("\nNOTE: all timings/backends here are LOCAL DEVELOPMENT EVIDENCE on an unpinned stack; "
          "official efficiency numbers require the pinned environment and real artifacts.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
