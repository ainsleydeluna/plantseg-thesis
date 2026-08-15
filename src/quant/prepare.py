"""E4 (PTQ) and E5 (QAT) structural paths over the existing FP32 `PlantSegStudent`.

Required ordering, enforced by the helpers below (contract B4: "fusion via `fuse_modules` BEFORE
observer insertion"):

    FP32 E1 checkpoint -> eval/train state -> fuse -> explicit qconfig -> prepare / prepare_qat
                       -> calibrate (E4) or QAT train (E5) -> convert

`src/models/student.py` is NOT modified: the student already carries QuantStub/DeQuantStub at the
boundary, `FloatFunctional` for the head's attention-multiply and class-logit add, and a `fuse()`
that fuses the head's Conv-BN-ReLU plus each quantizable backbone block. This module only drives it.

E4/E5 both take the **E1** FP32 checkpoint (contract stage table). E6/E7 derive from E3 and are not
implemented here.
"""

from __future__ import annotations

import copy
import math
from typing import Iterable

import torch
import torch.nn as nn
from torch.ao.quantization import convert, disable_observer, enable_observer, prepare, prepare_qat

from .qconfig import ptq_qconfig, qat_qconfig, select_qnnpack_backend

# Contract B4 pins a RANGE for the BN-stat freeze ("after ~65-70% of training"), not a value.
BN_FREEZE_PCT_RANGE = (0.65, 0.70)


class QuantPreparationError(RuntimeError):
    """Raised when a model cannot be prepared for quantization as the contract requires."""


def _fused_copy(model: nn.Module, *, is_qat: bool = False) -> nn.Module:
    """Deep-copy then fuse. Fusion always precedes observer insertion (contract B4).

    `fuse()` selects eval/train mode and the eager vs QAT fusion function itself, so PTQ gets
    BN folded into Conv while QAT keeps ConvBn/ConvBnReLU structures for BN-stat learning.
    """
    m = copy.deepcopy(model)
    if not hasattr(m, "fuse"):
        raise QuantPreparationError("model exposes no fuse(); Conv-BN(-ReLU) fusion is mandatory "
                                    "before observer insertion (contract B4)")
    m.fuse(is_qat=is_qat)
    return m


def prepare_ptq(model: nn.Module, *, select_backend: bool = True) -> nn.Module:
    """E4: fuse -> assign the explicit PTQ qconfig -> insert observers. Returns a prepared copy."""
    if select_backend:
        select_qnnpack_backend()
    m = _fused_copy(model, is_qat=False)
    m.eval()
    m.qconfig = ptq_qconfig()
    prepared = prepare(m, inplace=False)
    prepared.eval()
    return prepared


@torch.no_grad()
def calibrate(prepared: nn.Module, batches: Iterable[torch.Tensor]) -> int:
    """Run calibration forwards to populate the observers. Returns the number of batches seen.

    The caller supplies batches drawn from the frozen 128-image TRAIN calibration subset with
    clean-test preprocessing and NO augmentation (contract B4 / `src.quant.calibration`).
    """
    prepared.eval()
    n = 0
    for batch in batches:
        prepared(batch)
        n += 1
    if n == 0:
        raise QuantPreparationError("no calibration batches were supplied; converting an "
                                    "uncalibrated PTQ model would produce meaningless qparams")
    return n


def prepare_qat_model(model: nn.Module, *, select_backend: bool = True) -> nn.Module:
    """E5: fuse -> assign the explicit QAT qconfig -> insert fake-quant. Returns a prepared copy.

    Fake quantization is active from step 0 (contract B4). The returned module is left in train()
    mode, ready for the supervised QAT loop.
    """
    if select_backend:
        select_qnnpack_backend()
    m = _fused_copy(model, is_qat=True)      # QAT fusion: keeps ConvBn/ConvBnReLU for BN learning
    m.train()
    m.qconfig = qat_qconfig()
    prepared = prepare_qat(m, inplace=False)
    prepared.train()
    return prepared


def convert_model(prepared: nn.Module) -> nn.Module:
    """Convert a calibrated PTQ model or a trained QAT model to its INT8 form."""
    m = copy.deepcopy(prepared)
    m.eval()
    return convert(m, inplace=False)


# ------------------------------------------------------------------ QAT schedule helpers
def freeze_bn_stats(model: nn.Module) -> nn.Module:
    """Freeze BN running statistics (contract B4: after ~65-70% of training)."""
    from torch.ao.nn.intrinsic.qat import freeze_bn_stats as _freeze
    model.apply(_freeze)
    return model


def disable_observers(model: nn.Module) -> nn.Module:
    """Freeze activation ranges (contract B4: "shortly after BN freeze")."""
    model.apply(disable_observer)
    return model


def enable_observers(model: nn.Module) -> nn.Module:
    model.apply(enable_observer)
    return model


def bn_freeze_iteration(total_iters: int, pct: float) -> int:
    """Convert a BN-freeze percentage into an iteration index.

    The contract pins only the RANGE 65-70%, so `pct` must be supplied explicitly at launch — this
    helper exposes the mechanism and refuses a value outside the locked range rather than picking
    one silently.
    """
    lo, hi = BN_FREEZE_PCT_RANGE
    if not isinstance(pct, (int, float)) or not math.isfinite(pct) or not (lo <= pct <= hi):
        raise QuantPreparationError(
            f"BN-freeze percentage must be an explicit value inside the locked range "
            f"[{lo}, {hi}], got {pct!r}. Contract B4 fixes the range, not the value.")
    return max(1, int(round(total_iters * pct)))


def qat_grad_clip_gate_error(value: float | None) -> str | None:
    """Validate `--grad-clip-norm` for a REAL E5 launch. Returns an error string, or None if OK.

    Contract B4 requires global-norm gradient clipping for QAT but fixes no numeric `max_norm`
    (`configs/quant.py` records only the method family "global_norm"). Rather than inventing a
    threshold, a real E5 run must supply it explicitly — the same discipline as the E2/E3 gate.
    """
    if value is None:
        return ("--grad-clip-norm is required for a real E5 run. Contract B4 mandates global-norm "
                "gradient clipping for QAT but fixes no numeric max_norm (configs/quant.py records "
                "the method family only), so the threshold stays an explicit experiment-level "
                "decision: re-run with --grad-clip-norm <positive finite value>.")
    if not math.isfinite(value) or value <= 0.0:
        return (f"--grad-clip-norm must be a positive finite value, got {value!r}. Zero, negative, "
                "NaN and Inf are rejected.")
    return None


# ------------------------------------------------------------------ operator coverage
_QUANTIZED_MODULE_ROOTS = ("torch.ao.nn.quantized", "torch.ao.nn.intrinsic.quantized",
                           "torch.nn.quantized", "torch.nn.intrinsic.quantized")

# Ops the model applies FUNCTIONALLY, so a module walk cannot see them. Listed explicitly rather
# than silently omitted. NOTE: `FloatFunctional.mul/add` (LR-ASPP attention, class-logit add, the
# backbone residual `skip_add` and SE `skip_mul`) are NOT in this list — they are real modules and
# convert() turns them into `QFunctional`, so they ARE counted as INT8 below. QFunctional keeps its
# `activation_post_process` after conversion because that is where its output qparams live.
FUNCTIONAL_OPS = ("F.interpolate (head OS8 resize)", "F.interpolate (final float upsample)")


def _is_quantized_module(m: nn.Module) -> bool:
    return type(m).__module__.startswith(_QUANTIZED_MODULE_ROOTS)


def quantization_coverage(converted: nn.Module) -> dict:
    """Classify every leaf module of a CONVERTED model as INT8 or an FP32 fallback.

    Conversion succeeding is not proof of INT8 execution, so this reports what actually became
    quantized. Functional ops are reported separately as STRUCTURAL because a module walk cannot
    observe them.
    """
    int8, fp32, other = [], [], []
    for name, m in converted.named_modules():
        if list(m.children()):
            continue                                    # leaves only
        entry = f"{name or '<root>'}: {type(m).__module__}.{type(m).__name__}"
        if _is_quantized_module(m):
            int8.append(entry)
        elif isinstance(m, (nn.Identity,)) or type(m).__name__ in ("QuantStub", "DeQuantStub"):
            other.append(entry)
        elif isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ReLU, nn.Hardswish,
                            nn.Hardsigmoid, nn.Sigmoid, nn.AdaptiveAvgPool2d)):
            fp32.append(entry)
        else:
            other.append(entry)
    return {
        "int8_modules": int8,
        "fp32_fallback_modules": fp32,
        "structural_or_boundary_modules": other,
        "functional_ops_unobservable_by_module_walk": list(FUNCTIONAL_OPS),
        "int8_count": len(int8),
        "fp32_count": len(fp32),
    }


def try_converted_forward(converted: nn.Module, x: torch.Tensor) -> dict:
    """Attempt a forward pass on the converted model and report precisely what happened.

    Full INT8 may only be claimed after real backend conversion AND a successful forward, so this
    returns the failure verbatim instead of swallowing it.
    """
    converted.eval()
    try:
        with torch.no_grad():
            y = converted(x)
        return {"ok": True, "output_shape": tuple(y.shape), "dtype": str(y.dtype), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "output_shape": None, "dtype": None,
                "error": f"{type(e).__name__}: {e}"}
