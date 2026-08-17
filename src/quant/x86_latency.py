"""The SEPARATE x86/fbgemm INT8 copy used only for descriptive CPU-proxy latency.

IMPLEMENTATION_CONTRACT line 412: "x86 CPU proxy = separate fbgemm/x86 INT8 copy
(reduce_range=True); the QNNPACK copy stays the reported accuracy/size artifact. No ARM /
on-device latency claimed."

THIS IS NOT A NEW EXPERIMENT STAGE. It is an alternate backend representation of an existing
quantized stage, built solely so x86 latency can be timed on an x86 host. It never produces an
accuracy number, is never substituted during evaluation, and never overwrites the QNNPACK artifact.
Its provenance carries `artifact_role="x86_cpu_proxy_latency"` and its own SHA-256.

REUSE, NOT DUPLICATION. Fusion, calibration and conversion come from `src.quant.prepare`
(including its private `_fused_copy`, imported deliberately so the fusion traversal is not
reimplemented). Only the QConfig and the backend differ from the accuracy path.

E4/E7 (PTQ) rebuild from their FP32 source plus the frozen shared calibration subset. E5/E6 (QAT)
rebuild from the AUXILIARY pre-convert QAT sidecar the runner writes alongside the official
converted artifact, by TRANSLATION ONLY — same trained weights, same learned activation-range
evidence, backend-appropriate x86 quantizer parameters, zero optimizer steps. E5/E6 are never
re-trained under x86 to obtain a latency artifact: that would be a different model.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization.fake_quantize import FakeQuantize

from .prepare import _fused_copy, calibrate, convert_model
from .qconfig import (QUANT_BACKEND, X86_ACT_REDUCE_RANGE, X86_BACKENDS, describe_qconfig,
                      select_x86_backend, x86_ptq_qconfig, x86_qat_qconfig)
from .stages import require_shared_calibration_index, resolve_quant_stage

ACCURACY_ARTIFACT_ROLE = "accuracy"
X86_LATENCY_ARTIFACT_ROLE = "x86_cpu_proxy_latency"
ARTIFACT_ROLES = (ACCURACY_ARTIFACT_ROLE, X86_LATENCY_ARTIFACT_ROLE)

# Auxiliary pre-convert QAT companion artifact written by `src.quant.runner.run_qat`.
QAT_SIDECAR_KIND = "qat-train-state"
QAT_SIDECAR_ROLE = "preconvert_qat_state"
QAT_SIDECAR_SUFFIX = "_qat_state.pt"

# State-dict key suffixes. Quantizer PARAMETERS are backend-specific and must be recomputed;
# observed RANGE evidence is the training result and is carried over unchanged.
QPARAM_SUFFIXES = (".scale", ".zero_point")
RANGE_SUFFIXES = ("min_val", "max_val", "min_vals", "max_vals")

# PTQ stages rebuild from FP32 + calibration; QAT stages rebuild from the sidecar.
X86_PTQ_STAGES = ("E4", "E7")
X86_QAT_STAGES = ("E5", "E6")
X86_RECONSTRUCTIBLE_STAGES = X86_PTQ_STAGES + X86_QAT_STAGES


class X86LatencyCopyError(RuntimeError):
    """A refused x86 latency-copy construction. `code` keeps the gate testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def qat_sidecar_missing_error(stage_name: str) -> X86LatencyCopyError:
    """A QAT stage without its pre-convert sidecar cannot yield an x86 copy. Fails loudly.

    The official converted artifact holds only post-convert INT8 state whose activation qparams were
    computed under the accuracy configuration (reduce_range=False). Rebuilding from that alone would
    mean fabricating training evidence, so the auxiliary sidecar is REQUIRED rather than optional.
    """
    return X86LatencyCopyError(
        "qat_sidecar_required",
        f"{stage_name} is a QAT stage: its x86 latency copy must be translated from the auxiliary "
        f"pre-convert QAT sidecar ('*{QAT_SIDECAR_SUFFIX}', quantization={QAT_SIDECAR_KIND!r}). The "
        "official converted artifact contains only post-convert INT8 state whose activation qparams "
        "were computed with reduce_range=False, so reconstructing from it would fabricate training "
        "evidence. Pass sidecar= with that companion artifact; nothing is invented here.")


def prepare_x86_qat(model: nn.Module, *, select_backend: bool = True) -> nn.Module:
    """Fuse (QAT) -> assign the x86 QAT QConfig -> insert fake-quant. Mirrors `prepare_qat_model`.

    Structurally identical to the accuracy QAT preparation, so the sidecar's state-dict keys line up
    exactly; only the QConfig (reduce_range=True) differs.
    """
    if select_backend:
        select_x86_backend()
    m = _fused_copy(model, is_qat=True)        # QAT fusion: keeps ConvBn/ConvBnReLU structures
    m.train()
    m.qconfig = x86_qat_qconfig()
    from torch.ao.quantization import prepare_qat
    prepared = prepare_qat(m, inplace=False)
    prepared.train()
    return prepared


def validate_qat_sidecar(obj, *, origin: str = "sidecar") -> dict:
    """Validate a pre-convert QAT companion artifact. Used for both loaded files and in-memory dicts.

    Applied by `translate_qat_state_to_x86` too, so passing a converted-only artifact straight to the
    translator is refused with a stated reason rather than an incidental KeyError.
    """
    if not isinstance(obj, dict):
        raise X86LatencyCopyError("qat_sidecar_malformed", f"{origin} is not a checkpoint dict")
    if obj.get("quantization") != QAT_SIDECAR_KIND:
        raise X86LatencyCopyError(
            "qat_sidecar_wrong_kind",
            f"{origin} has quantization={obj.get('quantization')!r}; the x86 QAT translation "
            f"requires the pre-convert sidecar ({QAT_SIDECAR_KIND!r}). A converted INT8 artifact "
            "cannot be translated: its activation qparams were already computed under "
            "reduce_range=False, so rebuilding from it would fabricate training evidence.")
    if "model_state_dict" not in obj:
        raise X86LatencyCopyError("qat_sidecar_no_state", f"{origin} carries no 'model_state_dict'")
    if obj.get("is_official_accuracy_artifact") or obj.get("is_deployment_artifact"):
        raise X86LatencyCopyError(
            "qat_sidecar_role_conflict",
            f"{origin} claims to be an official accuracy/deployment artifact; the pre-convert QAT "
            "state is auxiliary and must never be either")
    return obj


def load_qat_sidecar(path: str | Path) -> dict:
    """Load and validate the auxiliary pre-convert QAT companion artifact."""
    p = Path(path)
    if not p.is_file():
        raise X86LatencyCopyError("qat_sidecar_missing", f"QAT sidecar not found: {p}")
    return validate_qat_sidecar(torch.load(p, map_location="cpu", weights_only=True), origin=str(p))


def _is_qparam(key: str) -> bool:
    return key.endswith(QPARAM_SUFFIXES)


def _recompute_x86_qparams(prepared: nn.Module) -> int:
    """Derive x86 quantizer parameters from the CARRIED-OVER observed ranges.

    This is the heart of the translation. The sidecar's `scale`/`zero_point` were computed for the
    accuracy configuration (reduce_range=False, activations over 0..255) and are deliberately NOT
    loaded. Each x86 FakeQuantize instead recomputes its qparams from the range evidence it just
    received, using its OWN quant_min/quant_max (0..127 under reduce_range=True).

    The update mirrors `torch.ao.quantization.FakeQuantize.forward`'s own qparam refresh, so the
    resulting buffers are exactly what a forward pass would have produced — no fabricated state.
    """
    updated = 0
    for mod in prepared.modules():
        if not isinstance(mod, FakeQuantize):
            continue
        scale, zero_point = mod.calculate_qparams()
        scale, zero_point = scale.to(mod.scale.device), zero_point.to(mod.zero_point.device)
        if mod.scale.shape != scale.shape:
            mod.scale.resize_(scale.shape)
            mod.zero_point.resize_(zero_point.shape)
        mod.scale.copy_(scale)
        mod.zero_point.copy_(zero_point)
        updated += 1
    return updated


def _copy_trained_state(prepared: nn.Module, incoming: dict) -> tuple[int, list[str]]:
    """Copy trained tensors into the prepared model BY NAME, bypassing `load_state_dict`.

    WHY NOT `load_state_dict`. A freshly `prepare_qat`-ed model has EMPTY per-channel observer
    buffers (`min_val`/`max_val` start at shape [0] and are sized on first observation), while the
    trained sidecar carries them at [out_channels]. PyTorch's per-channel observer reconciles that
    inside `_load_from_state_dict`, but the path is version-gated on the state dict's `_metadata` —
    and `_metadata` is an ATTRIBUTE of the returned OrderedDict, not an entry, so it does not survive
    `torch.save` + `torch.load(weights_only=True)`. On the registered torch 2.1 the observer then
    falls back to the LEGACY `min_vals`/`max_vals` names, reporting the modern buffers as both size
    mismatches and missing keys. torch 2.9 happened to tolerate it; torch 2.1 does not.

    Copying by name is version-independent and strictly more explicit: shapes are aligned where the
    target buffer is resizable, and every value still comes from the trained state. Nothing is
    synthesised, and `assert_trained_state_preserved` re-verifies the result afterwards.
    """
    targets = dict(prepared.named_parameters())
    targets.update(dict(prepared.named_buffers()))
    copied, unexpected = 0, []
    for key, value in incoming.items():
        target = targets.get(key)
        if target is None:
            unexpected.append(key)
            continue
        if not torch.is_tensor(value) or not torch.is_tensor(target):
            continue
        if target.shape != value.shape:
            target.resize_(value.shape)
        target.copy_(value)
        copied += 1
    return copied, unexpected


@torch.no_grad()
def translate_qat_state_to_x86(stage: str, model: nn.Module, sidecar: dict, *,
                               select_backend: bool = True) -> tuple[nn.Module, dict]:
    """Translate a trained QAT state into an x86-PREPARED model. ZERO optimizer steps.

    Same trained weights, same learned activation-range evidence, backend-appropriate x86 qparams.
    Decorated `@torch.no_grad()` and containing no optimizer, loss or backward call, so "no
    retraining" is a structural property rather than a promise.
    """
    st = resolve_quant_stage(stage)
    if st["method"] != "qat":
        raise X86LatencyCopyError(
            "not_a_qat_stage",
            f"{st['name']} is a {st['method'].upper()} stage; use the PTQ path with the frozen "
            "calibration subset instead")
    validate_qat_sidecar(sidecar, origin=f"{st['name']} sidecar")
    if sidecar.get("stage") not in (None, st["name"]):
        raise X86LatencyCopyError(
            "qat_sidecar_stage_mismatch",
            f"sidecar was written for {sidecar.get('stage')!r}, translating {st['name']!r}")

    source_state = sidecar["model_state_dict"]
    prepared = prepare_x86_qat(model, select_backend=select_backend)

    carried = {k: v for k, v in source_state.items() if not _is_qparam(k)}
    skipped = sorted(k for k in source_state if _is_qparam(k))
    copied, unexpected = _copy_trained_state(prepared, carried)
    if unexpected:
        raise X86LatencyCopyError(
            "qat_sidecar_key_mismatch",
            f"sidecar carries {len(unexpected)} key(s) the x86-prepared model does not define, "
            f"e.g. {unexpected[:3]}; the two preparations are not structurally identical")

    # every non-qparam tensor the x86 model defines must have been supplied by the sidecar
    expected = {k for k in prepared.state_dict() if not _is_qparam(k)}
    unexplained = sorted(expected - set(carried))
    if unexplained:
        raise X86LatencyCopyError(
            "qat_sidecar_incomplete",
            f"the x86-prepared model needs {len(unexplained)} non-qparam tensor(s) the sidecar does "
            f"not provide, e.g. {unexplained[:3]}")

    updated = _recompute_x86_qparams(prepared)
    mismatched = assert_trained_state_preserved(prepared, source_state)
    if mismatched:
        raise X86LatencyCopyError(
            "trained_weights_altered",
            f"translation changed {len(mismatched)} trained tensor(s), e.g. {mismatched[:3]}; the "
            "x86 latency copy must carry the exact QAT-trained weights")

    report = {
        "translated_from": QAT_SIDECAR_ROLE,
        "trained_tensors_copied": copied,
        "qparams_recomputed": updated,
        "qparams_skipped_from_source": len(skipped),
        "carried_tensors": len(carried),
        "optimizer_steps_for_translation": 0,
        "gradient_updates": 0,
        "retrained": False,
        "training_quant_backend": sidecar.get("training_quant_backend", QUANT_BACKEND),
        "reduce_range": X86_ACT_REDUCE_RANGE,
        "source_best_val_miou_all_class": sidecar.get("best_val_miou_all_class"),
    }
    return prepared, report


def assert_trained_state_preserved(prepared: nn.Module, source_state: dict) -> list[str]:
    """Return the trained tensors that differ from the sidecar. Qparams are excluded by design."""
    live = prepared.state_dict()
    mismatched = []
    for key, value in source_state.items():
        if _is_qparam(key):
            continue                       # backend-specific, intentionally recomputed
        current = live.get(key)
        if current is None or current.shape != value.shape or not torch.equal(current, value):
            mismatched.append(key)
    return mismatched


def _require_x86_engine() -> str:
    """Conversion must happen under a real x86 engine, or the packed model is not an x86 artifact."""
    engine = torch.backends.quantized.engine
    if engine not in X86_BACKENDS:
        raise X86LatencyCopyError(
            "x86_engine_not_selected",
            f"conversion requires an approved x86 engine {list(X86_BACKENDS)}, but the active "
            f"engine is {engine!r}. Converting under {engine!r} would produce a differently packed "
            "model that must never be reported as the x86 CPU-proxy result.")
    return engine


def prepare_x86_ptq(model: nn.Module, *, select_backend: bool = True) -> nn.Module:
    """Fuse -> assign the x86 PTQ QConfig -> insert observers. Mirrors `prepare_ptq`, x86 QConfig."""
    if select_backend:
        select_x86_backend()
    m = _fused_copy(model, is_qat=False)      # same fusion traversal as the accuracy path
    m.eval()
    m.qconfig = x86_ptq_qconfig()
    from torch.ao.quantization import prepare
    prepared = prepare(m, inplace=False)
    prepared.eval()
    return prepared


def build_x86_latency_copy(stage: str, model: nn.Module, *, calibration_batches=None,
                           sidecar: dict | None = None,
                           select_backend: bool = True) -> nn.Module:
    """Build the converted x86 INT8 latency copy for any INT8 stage.

    PTQ (E4/E7): rebuild from the FP32 source and the SAME frozen shared calibration subset — see
    `require_x86_calibration_identity`. QAT (E5/E6): translate the auxiliary pre-convert sidecar,
    which is REQUIRED; without it this refuses rather than fabricating one.

    In both cases the x86 activation scales differ from the QNNPACK artifact's because reduce_range
    differs. That is the point of a backend-specific latency representation, and it is acceptable
    precisely because this copy is never used for accuracy, robustness or size.
    """
    st = resolve_quant_stage(stage)

    if st["method"] == "ptq":
        prepared = prepare_x86_ptq(model, select_backend=select_backend)
        if calibration_batches is not None:
            calibrate(prepared, calibration_batches)
    else:
        if sidecar is None:
            raise qat_sidecar_missing_error(st["name"])
        prepared, _report = translate_qat_state_to_x86(st["name"], model, sidecar,
                                                       select_backend=select_backend)
        prepared.eval()

    _require_x86_engine()          # never emit a differently-packed model as the x86 artifact
    return convert_model(prepared)


def require_x86_calibration_identity(path: str | Path, expected_checksum: str | None = None) -> dict:
    """The x86 PTQ copy reuses the EXACT frozen E4/E7 calibration subset. Never resamples.

    Delegates to the committed shared-calibration validator (schema, split=train, seed=42,
    count=128, shared_by, checksum) rather than restating those rules.
    """
    return require_shared_calibration_index(path, expected_checksum)


def x86_artifact_provenance(*, stage: str, engine: str, source_sha256: str | None = None,
                            artifact_path: str | Path | None = None,
                            calibration_checksum: str | None = None,
                            translation: dict | None = None) -> dict:
    """Provenance that cannot be confused with the accuracy artifact's.

    For QAT stages `translation` carries the sidecar report, so the record states plainly that the
    weights were TRAINED under QNNPACK and only re-quantized for x86 — it never claims the artifact
    was trained with x86 QAT.
    """
    if engine not in X86_BACKENDS:
        raise X86LatencyCopyError(
            "x86_provenance_backend",
            f"an x86 latency artifact must record an approved engine {list(X86_BACKENDS)}, "
            f"got {engine!r}")
    st = resolve_quant_stage(stage)
    is_qat = st["method"] == "qat"
    return {
        "stage": st["name"],
        "source_stage": st["source_stage"],
        "source_role": QAT_SIDECAR_ROLE if is_qat else "fp32_checkpoint_plus_frozen_calibration",
        "artifact_role": X86_LATENCY_ARTIFACT_ROLE,
        "accuracy_artifact_role": ACCURACY_ARTIFACT_ROLE,
        "size_artifact_role": ACCURACY_ARTIFACT_ROLE,
        "training_quant_backend": QUANT_BACKEND,      # weights were TRAINED under QNNPACK
        "latency_quant_backend": engine,
        "trained_with_x86_qat": False,                # never claim otherwise
        "optimizer_steps_for_translation": (translation or {}).get(
            "optimizer_steps_for_translation", 0),
        "translation": translation,
        "backend": engine,
        "reduce_range": X86_ACT_REDUCE_RANGE,
        "qconfig": describe_qconfig(x86_qat_qconfig() if is_qat else x86_ptq_qconfig()),
        "source_checkpoint_sha256": source_sha256,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "calibration_checksum_sha256": calibration_checksum,
        "usable_for_accuracy": False,
        "usable_for_robustness": False,
        "usable_for_size_reporting": False,   # the QNNPACK copy stays the reported size artifact
        "usable_for_latency": True,
        "on_device": False,
        "cpu_proxy": True,
    }


__all__ = ["ACCURACY_ARTIFACT_ROLE", "ARTIFACT_ROLES", "QAT_SIDECAR_KIND", "QAT_SIDECAR_ROLE",
           "QAT_SIDECAR_SUFFIX", "X86_LATENCY_ARTIFACT_ROLE", "X86_PTQ_STAGES", "X86_QAT_STAGES",
           "X86_RECONSTRUCTIBLE_STAGES", "X86LatencyCopyError", "assert_trained_state_preserved",
           "build_x86_latency_copy", "load_qat_sidecar", "prepare_x86_ptq", "prepare_x86_qat",
           "qat_sidecar_missing_error", "require_x86_calibration_identity",
           "translate_qat_state_to_x86", "validate_qat_sidecar", "x86_artifact_provenance"]
