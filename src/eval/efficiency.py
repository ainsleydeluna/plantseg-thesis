"""Descriptive efficiency/profiling surface for the teacher and E1-E7.

DESCRIPTIVE ONLY, BY CONSTRUCTION. IMPLEMENTATION_CONTRACT (g)/"Efficiency (descriptive only)" and
line 384 keep efficiency out of every inferential family: nothing here computes or imports a
Wilcoxon, paired t-test, Holm correction, bootstrap, BCa or non-inferiority decision, and every
emitted record carries `cpu_proxy=True` / `on_device=False`. No ARM or on-device latency may be
claimed from these numbers — the wording is CPU PROXY throughout.

Registered metrics (IMPLEMENTATION_CONTRACT lines 405-413):
  * parameter count;
  * model size = deterministic parameter-BYTE footprint (INT8 = 1 B/weight, FP32 = 4 B/weight,
    bias and per-channel scale/zero-point = 4 B each) PLUS the serialized on-disk size of the same
    artifact used for latency;
  * FLOPs = 2 x MACs via fvcore @ 512x512, profiled from the FP32 architecture (QAT does not cut
    FLOPs);
  * CPU-proxy latency via `torch.utils.benchmark` (CPU, batch 1, 20 warm-up + 100 measured,
    median / IQR / p95, `eval()` + `inference_mode()`, AMP off, fixed thread count);
  * peak memory via RSS delta.
Throughput and energy are NOT registered and are deliberately absent.

MEASUREMENT vs REDUCTION. Every statistic is a pure function of observations
(`latency_statistics`, `footprint_from_counts`, `memory_delta`), so the arithmetic is verifiable
with known synthetic numbers and never depends on real wall-clock timing. The measuring functions
take injectable `counter` / `sampler` hooks for the same reason.

TWO INT8 ARTIFACTS, NEVER POOLED. The QNNPACK copy is the accuracy/size artifact; a SEPARATE
fbgemm/x86 copy with `reduce_range=True` is the x86 CPU-proxy latency artifact (contract line 412).
`artifact_role` distinguishes them and `x86_latency_copy_gate()` refuses to improvise the second one
— see that function for the exact interface `src.quant` would have to grow first.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.model_loading import sha256_file  # noqa: E402  (reuse; no second loader here)

EFFICIENCY_SCHEMA = "plantseg-efficiency/1.0.0"

# ---- frozen protocol -------------------------------------------------------------------------
INPUT_SHAPE = (1, 3, 512, 512)
BATCH_SIZE = 1
WARMUP_FORWARDS = 20
MEASURED_FORWARDS = 100
PERCENTILE_METHOD = "linear"          # numpy method='linear'; stated so the reducer is unambiguous
FVCORE_PINNED = "0.1.5.post20221221"  # IMPLEMENTATION_CONTRACT pinned-stack table

# ---- byte accounting (contract line 406-407) -------------------------------------------------
INT8_WEIGHT_BYTES = 1
FP32_WEIGHT_BYTES = 4
METADATA_BYTES = 4                    # bias, per-channel scale, per-channel zero-point

# ---- INT8 artifact roles (contract line 412) --------------------------------------------------
ARTIFACT_ROLE_ACCURACY = "accuracy"
ARTIFACT_ROLE_X86_LATENCY = "x86_cpu_proxy_latency"
ARTIFACT_ROLES = (ARTIFACT_ROLE_ACCURACY, ARTIFACT_ROLE_X86_LATENCY)
ACCURACY_BACKEND = "qnnpack"
ACCURACY_REDUCE_RANGE = False
X86_LATENCY_BACKEND = "fbgemm"
X86_LATENCY_REDUCE_RANGE = True

CPU_PROXY = True
ON_DEVICE = False
MEASUREMENT_ROLE = "descriptive"

STAGES = ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
INT8_STAGES = ("E4", "E5", "E6", "E7")


class EfficiencyError(RuntimeError):
    """A refused efficiency measurement. Carries a stable `code` for tests and logs."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str):
    raise EfficiencyError(code, message)


# ------------------------------------------------------------------ parameters
def count_parameters(model) -> dict:
    """Raw parameter counts of whatever module is handed in."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def is_quantized(model) -> bool:
    """True when any submodule holds packed INT8 parameters."""
    for mod in model.modules():
        if hasattr(mod, "_packed_params") or ".quantized" in type(mod).__module__:
            return True
    return False


def architecture_parameter_count(model, *, fp32_reference=None) -> dict:
    """Architectural parameter count — quantization changes STORAGE, not the learned parameter set.

    After `convert`, INT8 weights live inside opaque packed params and `model.parameters()` is
    largely empty. Reporting that number would silently claim an INT8 model has almost no
    parameters, so a quantized model must be counted against its FP32 architecture instead.
    """
    if is_quantized(model):
        if fp32_reference is None:
            _fail("int8_parameter_count_needs_fp32_reference",
                  "a converted INT8 model exposes packed parameters, so its architectural parameter "
                  "count must come from the FP32 architecture; pass fp32_reference=")
        counts = count_parameters(fp32_reference)
        counts["counted_from"] = "fp32_reference_architecture"
        return counts
    counts = count_parameters(model)
    counts["counted_from"] = "model"
    return counts


# ------------------------------------------------------------------ byte footprint
def footprint_from_counts(int8_weight_elements: int, fp32_weight_elements: int,
                          metadata_elements: int) -> dict:
    """Deterministic parameter-byte footprint. Pure reducer — the unit under synthetic test.

    Never asserts a flat 4x saving: mixed precision and metadata are reported explicitly so a
    partially quantized model cannot be described as "exactly 4x smaller".
    """
    for name, value in (("int8_weight_elements", int8_weight_elements),
                        ("fp32_weight_elements", fp32_weight_elements),
                        ("metadata_elements", metadata_elements)):
        if not isinstance(value, int) or value < 0:
            _fail("footprint_negative_count", f"{name} must be a non-negative int, got {value!r}")

    int8_bytes = int8_weight_elements * INT8_WEIGHT_BYTES
    fp32_bytes = fp32_weight_elements * FP32_WEIGHT_BYTES
    metadata_bytes = metadata_elements * METADATA_BYTES
    total = int8_bytes + fp32_bytes + metadata_bytes

    all_weights = int8_weight_elements + fp32_weight_elements
    fp32_equivalent = all_weights * FP32_WEIGHT_BYTES + metadata_bytes
    return {
        "int8_weight_elements": int8_weight_elements,
        "fp32_weight_elements": fp32_weight_elements,
        "metadata_elements": metadata_elements,
        "int8_bytes": int8_bytes,
        "fp32_bytes": fp32_bytes,
        "metadata_bytes": metadata_bytes,
        "theoretical_weight_bytes": total,
        "fp32_equivalent_bytes": fp32_equivalent,
        "compression_vs_fp32": (fp32_equivalent / total) if total else None,
        "mixed_precision_remains": fp32_weight_elements > 0,
        "fully_quantized": int8_weight_elements > 0 and fp32_weight_elements == 0,
        "byte_rule": {"int8_weight": INT8_WEIGHT_BYTES, "fp32_weight": FP32_WEIGHT_BYTES,
                      "bias_scale_zero_point": METADATA_BYTES},
    }


def weight_footprint(model) -> dict:
    """Partition a live module into INT8 weights, retained FP32 weights and metadata elements."""
    int8_elements = fp32_elements = metadata_elements = 0
    seen = set()

    for mod in model.modules():
        packed = hasattr(mod, "_packed_params") or ".quantized" in type(mod).__module__
        weight = getattr(mod, "weight", None)
        if packed:
            try:
                tensor = weight() if callable(weight) else weight
            except Exception:                      # noqa: BLE001 - opaque packed param
                tensor = None
            if torch.is_tensor(tensor):
                int8_elements += tensor.numel()
                # per-channel scale + zero-point, or one pair for per-tensor
                channels = tensor.shape[0] if tensor.dim() > 0 else 1
                per_channel = getattr(tensor, "qscheme", lambda: None)() in (
                    torch.per_channel_affine, torch.per_channel_symmetric)
                metadata_elements += 2 * (channels if per_channel else 1)
            bias = getattr(mod, "bias", None)
            try:
                bias = bias() if callable(bias) else bias
            except Exception:                      # noqa: BLE001
                bias = None
            if torch.is_tensor(bias):
                metadata_elements += bias.numel()
            continue

        for name, param in mod.named_parameters(recurse=False):
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            if name == "bias":
                metadata_elements += param.numel()
            else:
                fp32_elements += param.numel()

    return footprint_from_counts(int(int8_elements), int(fp32_elements), int(metadata_elements))


def serialized_artifact(path) -> dict:
    """Physical on-disk bytes of the deployable artifact actually being measured."""
    p = Path(path)
    if not p.is_file():
        _fail("artifact_missing", f"serialized artifact not found: {p}")
    size = p.stat().st_size
    return {
        "path": str(p),
        "serialized_artifact_bytes": int(size),
        "serialized_artifact_mib": size / (1024 ** 2),   # MiB = 1024^2, stated explicitly
        "size_convention": "bytes canonical; MiB = bytes / 1024^2",
        "sha256": sha256_file(p),
    }


# ------------------------------------------------------------------ MACs / FLOPs
def _fvcore_counter(model, inputs):
    """Default MAC counter. fvcore counts one fused multiply-add as ONE operation."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        _fail("fvcore_unavailable",
              "NEEDS OFFICIAL PROFILING ENVIRONMENT: fvcore is not installed here and must not be "
              f"installed as a side effect; the pinned stack records fvcore {FVCORE_PINNED}")
    import fvcore
    analysis = FlopCountAnalysis(model, inputs)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    return int(analysis.total()), dict(analysis.unsupported_ops()), getattr(
        fvcore, "__version__", "unknown")


def profile_macs(model, *, input_shape=INPUT_SHAPE, counter=None) -> dict:
    """Theoretical architecture compute at 1x3x512x512, reported as MACs and FLOPs = 2 x MACs.

    Profiles the FP32 architecture: an INT8 model performs the same multiply-accumulates, so
    quantization must not appear to reduce theoretical compute.

    Elementwise operations that the registered count excludes (squeeze-and-excitation gating,
    Hardsigmoid/Sigmoid, residual adds, bilinear upsampling) are REPORTED via
    `unsupported_flop_ops` rather than absorbed by custom handlers that would inflate the total.
    """
    if tuple(input_shape) != INPUT_SHAPE:
        _fail("flops_input_shape", f"registered FLOP input is {INPUT_SHAPE}, got {tuple(input_shape)}")
    if is_quantized(model):
        _fail("flops_require_fp32_architecture",
              "theoretical MAC/FLOP counting is defined on the FP32 architecture; QAT/PTQ does not "
              "cut FLOPs, so profile the FP32 student/teacher instead of a converted INT8 model")

    inputs = (torch.zeros(*input_shape),)
    macs, unsupported, version = (counter or _fvcore_counter)(model, inputs)
    if not isinstance(macs, int) or macs < 0:
        _fail("flops_counter_invalid", f"MAC counter returned {macs!r}")
    return {
        "macs": macs,
        "flops_2x_mac": 2 * macs,
        "unsupported_flop_ops": unsupported,
        "profiler": "fvcore",
        "profiler_version": version,
        "input_shape": list(input_shape),
        "profiled_precision": "fp32",
        "convention": ("fvcore counts one fused multiply-add as one operation, so its total IS the "
                       "MAC count; the thesis reports FLOPs = 2 x MACs"),
    }


# ------------------------------------------------------------------ latency
def latency_statistics(samples_seconds, *, expected: int | None = MEASURED_FORWARDS) -> dict:
    """Median / Q1 / Q3 / IQR / p95 over raw per-forward observations. Pure reducer.

    Descriptive only: no confidence interval and no inferential test is derived from these.
    """
    samples = [float(s) for s in samples_seconds]
    if expected is not None and len(samples) != expected:
        _fail("latency_sample_count",
              f"the registered protocol measures exactly {expected} forwards, got {len(samples)}")
    if not samples:
        _fail("latency_no_samples", "no latency observations")
    if any(s < 0 for s in samples):
        _fail("latency_negative_sample", "a negative forward time is not measurable")

    arr = np.asarray(samples, dtype=float) * 1000.0        # -> milliseconds
    q1, median, q3 = (float(np.percentile(arr, q, method=PERCENTILE_METHOD)) for q in (25, 50, 75))
    return {
        "latency_median_ms": median,
        "latency_q1_ms": q1,
        "latency_q3_ms": q3,
        "latency_iqr_ms": q3 - q1,
        "latency_p95_ms": float(np.percentile(arr, 95, method=PERCENTILE_METHOD)),
        "latency_samples": len(samples),
        "percentile_method": f"numpy method='{PERCENTILE_METHOD}'",
        "units": "milliseconds",
        "cpu_proxy": CPU_PROXY,
        "on_device": ON_DEVICE,
        "deterministic": False,      # timing varies; the protocol is fixed, the numbers are not
    }


def _benchmark_sampler(model, inputs, measured: int):
    """Default sampler: `torch.utils.benchmark`, one observation per measured forward."""
    import torch.utils.benchmark as benchmark

    timer = benchmark.Timer(
        stmt="with torch.inference_mode():\n    model(x)",
        globals={"model": model, "x": inputs, "torch": torch})
    return [timer.timeit(1).median for _ in range(measured)]


def measure_latency(model, *, input_shape=INPUT_SHAPE, warmup: int = WARMUP_FORWARDS,
                    measured: int = MEASURED_FORWARDS, sampler=None,
                    thread_count: int | None = None) -> dict:
    """CPU-proxy latency under the registered protocol. Never a device/ARM claim."""
    if warmup != WARMUP_FORWARDS or measured != MEASURED_FORWARDS:
        _fail("latency_protocol_violation",
              f"the registered protocol is {WARMUP_FORWARDS} warm-up + {MEASURED_FORWARDS} "
              f"measured forwards, got {warmup} + {measured}")
    if input_shape[0] != BATCH_SIZE:
        _fail("latency_batch_size", f"CPU-proxy latency is batch {BATCH_SIZE}, got {input_shape[0]}")

    if thread_count is not None:
        torch.set_num_threads(int(thread_count))

    model.eval()                       # required by the protocol; AMP is never enabled here
    inputs = torch.zeros(*input_shape)

    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)

    samples = (sampler or _benchmark_sampler)(model, inputs, measured)
    stats = latency_statistics(samples, expected=measured)
    stats.update({
        "warmup_forwards": warmup,
        "batch_size": input_shape[0],
        "input_shape": list(input_shape),
        "thread_count": torch.get_num_threads(),
        "interop_thread_count": torch.get_num_interop_threads(),
        "timer": "torch.utils.benchmark" if sampler is None else "injected",
        "amp": False,
        "inference_mode": True,
        "eval_mode": not model.training,
    })
    return stats


# ------------------------------------------------------------------ memory
def rss_bytes(sampler=None) -> int:
    """Resident set size in BYTES.

    `resource.getrusage` is Unix-only (and its `ru_maxrss` unit differs across platforms: KiB on
    Linux, bytes on macOS), so psutil RSS is used where available because it is bytes everywhere.
    """
    if sampler is not None:
        return int(sampler())
    try:
        import psutil
    except ImportError:
        _fail("memory_sampler_unavailable",
              "NEEDS OFFICIAL PROFILING ENVIRONMENT: psutil is unavailable and must not be "
              "installed as a side effect; pass sampler= to inject one")
    return int(psutil.Process().memory_info().rss)


def memory_delta(baseline_bytes: int, peak_bytes: int) -> dict:
    """Peak RSS delta over the post-model-load baseline. Pure reducer.

    EXPLICIT DESIGN: a negative delta is REPORTED with `negative_delta=True`, never silently
    clamped to zero. Allocator behaviour can genuinely release memory during a run, and clamping
    would disguise that as "no growth".
    """
    for name, value in (("baseline_bytes", baseline_bytes), ("peak_bytes", peak_bytes)):
        if not isinstance(value, int) or value < 0:
            _fail("memory_negative_absolute", f"{name} must be a non-negative int, got {value!r}")
    delta = peak_bytes - baseline_bytes
    return {
        "baseline_rss_bytes": baseline_bytes,
        "peak_rss_bytes": peak_bytes,
        "peak_rss_delta_bytes": delta,
        "negative_delta": delta < 0,
        "clamped": False,
        "units": "bytes",
        "source": "psutil.Process().memory_info().rss",
        "platform": platform.platform(),
    }


# ------------------------------------------------------------------ INT8 artifact roles
def require_quantized_backend(name: str) -> str:
    """Select a quantization engine or refuse. Never silently substitutes another backend."""
    supported = list(torch.backends.quantized.supported_engines)
    if name not in supported:
        _fail("quant_backend_unavailable",
              f"quantization backend {name!r} is unavailable in this build "
              f"(supported_engines={supported}); substituting another backend would change the "
              "artifact and is refused")
    torch.backends.quantized.engine = name
    return name


def x86_latency_copy_gate(stage: str | None = None) -> str:
    """Route to the real x86 latency-copy support in `src.quant`; refuse loudly when it cannot run.

    Contract line 412 requires the x86 proxy to be a SEPARATE fbgemm/x86 INT8 copy with
    `reduce_range=True`, while the QNNPACK copy (`reduce_range=False`) stays the reported
    accuracy/size artifact. `src.quant.x86_latency` now owns that construction; this stays a gate
    so the profiler can never time the QNNPACK copy and label it the x86 result.

    Returns the engine actually selected. All four INT8 stages are reconstructible — E4/E7 from
    their FP32 source plus the frozen calibration subset, E5/E6 by zero-training translation of the
    auxiliary pre-convert QAT sidecar — so this refuses only for a non-INT8 stage or a host without
    an approved x86 engine. The sidecar requirement itself is enforced where the copy is built.
    """
    from src.quant.qconfig import QuantBackendUnavailable, select_x86_backend
    from src.quant.x86_latency import X86_RECONSTRUCTIBLE_STAGES

    if stage is not None and stage not in X86_RECONSTRUCTIBLE_STAGES:
        _fail("x86_latency_copy_not_int8_stage",
              f"{stage} is not an INT8 stage; the x86 CPU-proxy latency copy exists only for "
              f"{list(X86_RECONSTRUCTIBLE_STAGES)}")
    try:
        return select_x86_backend()
    except QuantBackendUnavailable as e:
        _fail("x86_backend_unavailable", str(e))


def validate_artifact_role(role: str, backend: str) -> None:
    """Keep the two INT8 artifacts distinguishable so their numbers are never pooled."""
    if role not in ARTIFACT_ROLES:
        _fail("artifact_role_unknown", f"artifact_role must be one of {ARTIFACT_ROLES}, got {role!r}")
    if role == ARTIFACT_ROLE_ACCURACY and backend != ACCURACY_BACKEND:
        _fail("accuracy_artifact_backend",
              f"the accuracy/size INT8 artifact is {ACCURACY_BACKEND!r}, got {backend!r}")
    if role == ARTIFACT_ROLE_X86_LATENCY and backend == ACCURACY_BACKEND:
        _fail("latency_artifact_is_not_qnnpack",
              "benchmarking the QNNPACK artifact and labelling it the x86 CPU-proxy latency result "
              "conflates two different quantization configurations")


# ------------------------------------------------------------------ provenance / record
def environment() -> dict:
    import torchvision
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "quant_backends_available": list(torch.backends.quantized.supported_engines),
        "thread_count": torch.get_num_threads(),
        "interop_thread_count": torch.get_num_interop_threads(),
    }


def provenance(*, stage: str, model_role: str, precision: str, artifact_role: str,
               backend: str | None, source_path=None, source_sha256: str | None = None) -> dict:
    """Identify EXACTLY which artifact produced a measurement. Never pool two unidentified ones."""
    if stage not in STAGES:
        _fail("unknown_stage", f"stage must be one of {STAGES}, got {stage!r}")
    return {
        "stage": stage,
        "model_role": model_role,
        "precision": precision,
        "artifact_role": artifact_role,
        "backend": backend,
        "source_artifact_path": str(source_path) if source_path else None,
        "source_artifact_sha256": source_sha256,
        "input_shape": list(INPUT_SHAPE),
        "batch_size": BATCH_SIZE,
        "environment": environment(),
    }


def efficiency_record(*, provenance_block: dict, parameters: dict | None = None,
                      footprint: dict | None = None, artifact: dict | None = None,
                      compute: dict | None = None, latency: dict | None = None,
                      memory: dict | None = None, deferred: dict | None = None) -> dict:
    """One structured efficiency result for one stage/artifact.

    Written to its OWN schema: the frozen evaluation/statistics artifact schemas are not widened to
    carry efficiency fields.
    """
    return {
        "schema": EFFICIENCY_SCHEMA,
        "measurement_role": MEASUREMENT_ROLE,
        "cpu_proxy": CPU_PROXY,
        "on_device": ON_DEVICE,
        "no_arm_claim": True,
        "provenance": provenance_block,
        "parameters": parameters,
        "footprint": footprint,
        "artifact": artifact,
        "compute": compute,
        "latency": latency,
        "memory": memory,
        "deferred": deferred or {},
    }
