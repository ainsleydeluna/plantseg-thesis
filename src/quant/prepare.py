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


# AM-10: one image per calibration mini-batch (configs/quant.py `calibration_batch_size`).
CALIBRATION_BATCH_SIZE = 1


@torch.no_grad()
def calibrate(prepared: nn.Module, batches: Iterable[torch.Tensor]) -> int:
    """Run calibration forwards to populate the observers. Returns the number of batches seen.

    The caller supplies batches drawn from the frozen 128-image TRAIN calibration subset with
    clean-test preprocessing and NO augmentation (contract B4 / `src.quant.calibration`).
    AM-10 fixes ONE image per mini-batch: a batch that is not a single [1, C, H, W] image is refused
    before it reaches an observer, so no calibration path can use another batch size.
    """
    prepared.eval()
    n = 0
    for batch in batches:
        if not torch.is_tensor(batch) or batch.dim() != 4 or batch.shape[0] != CALIBRATION_BATCH_SIZE:
            shape = tuple(batch.shape) if torch.is_tensor(batch) else type(batch).__name__
            raise QuantPreparationError(
                f"calibration batch {n} has shape {shape}; AM-10 fixes one image per mini-batch "
                f"([{CALIBRATION_BATCH_SIZE}, C, H, W]) and any other batch size is refused")
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
    """Freeze BN running statistics in every fused QAT ConvBn module (AM-4a item 1: top of epoch 11)."""
    from torch.ao.nn.intrinsic.qat import freeze_bn_stats as _freeze
    model.apply(_freeze)
    return model


def disable_observers(model: nn.Module) -> nn.Module:
    """Disable every FakeQuantize observer, weight and activation (AM-4a item 1: top of epoch 13)."""
    model.apply(disable_observer)
    return model


def enable_observers(model: nn.Module) -> nn.Module:
    model.apply(enable_observer)
    return model


def qat_freeze_steps(steps_per_epoch: int, bn_freeze_epoch: int | None = None,
                     obs_freeze_epoch: int | None = None) -> tuple[int, int]:
    """AM-4a item 1: the completed optimizer steps after which BN statistics, then every observer, freeze.

    BN statistics freeze at the top of epoch bn_freeze_epoch + 1 and every observer at the top of epoch
    obs_freeze_epoch + 1, so both counts are whole epochs: (10 x spe, 12 x spe), i.e. (3,350, 4,020) at
    the 335 steps per epoch of record and (20, 24) in the smokes' 2-step geometry. The epochs default to
    configs/quant.py qat_real_run.
    """
    if bn_freeze_epoch is None or obs_freeze_epoch is None:
        from configs.quant import QUANT
        run = QUANT["qat_real_run"]
        bn_freeze_epoch = run["bn_freeze_epoch"] if bn_freeze_epoch is None else bn_freeze_epoch
        obs_freeze_epoch = run["obs_freeze_epoch"] if obs_freeze_epoch is None else obs_freeze_epoch
    for name, v in (("steps_per_epoch", steps_per_epoch), ("bn_freeze_epoch", bn_freeze_epoch),
                    ("obs_freeze_epoch", obs_freeze_epoch)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise QuantPreparationError(f"{name} must be a positive integer, got {v!r}")
    if not bn_freeze_epoch < obs_freeze_epoch:
        raise QuantPreparationError(f"the BN freeze (after epoch {bn_freeze_epoch}) must precede the observer "
                                    f"freeze (after epoch {obs_freeze_epoch})")
    return bn_freeze_epoch * steps_per_epoch, obs_freeze_epoch * steps_per_epoch


def unfused_batchnorm(model: nn.Module) -> list[str]:
    """BatchNorm modules outside a fused QAT ConvBn / ConvBnReLU module.

    freeze_bn_stats reaches a BN only through its fused module, so the epoch-11 freeze would miss any
    BN listed here; a real QAT run requires the list to be empty.
    """
    from torch.ao.nn.intrinsic.qat.modules.conv_fused import _ConvBnNd
    fused = {id(m.bn) for m in model.modules() if isinstance(m, _ConvBnNd)}
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and id(m) not in fused]


def qat_grad_clip_gate_error(value: float | None) -> str | None:
    """Validate a global-norm clipping threshold for E5/E6 QAT. Returns an error string, or None if OK.

    Contract B4 requires global-norm gradient clipping for QAT and AM-4a item 3 selects the threshold
    from the U4 candidates (configs/quant.py qat_grad_clip_pilot). This primitive accepts any positive
    finite value; candidate membership and the binding to the U4 selection are src/quant/qat.py's.
    """
    if value is None:
        return ("--grad-clip-norm is required for a real E5/E6 run. Contract B4 mandates global-norm "
                "gradient clipping for QAT and AM-4a item 3 selects the threshold from the U4 "
                "candidates (configs/quant.py qat_grad_clip_pilot): re-run with --grad-clip-norm "
                "<positive finite value>.")
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
