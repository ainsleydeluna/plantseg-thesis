"""Explicit INT8 QConfigs for E4 (PTQ) and E5 (QAT), built to the locked B4 contract.

WHY NOT `get_default_qconfig("qnnpack")` / `get_default_qat_qconfig("qnnpack")`:
those helpers quantize WEIGHTS PER-TENSOR, while contract B4 requires **per-channel symmetric INT8
for all conv layers** ("per-channel depthwise essential"). A helper's name is not a guarantee of its
qscheme, so both QConfigs below are constructed explicitly and are introspected by the smoke —
dtype, qscheme, quant_min, quant_max and ch_axis are asserted on the real observer instances rather
than by class name.

Contract B4:
  * activations — per-tensor **asymmetric UINT8**, full 8-bit range (0..255), reduce_range=False
  * weights     — per-channel **symmetric INT8**, all conv
  * PTQ activations use a **histogram** observer; PTQ weights a per-channel min/max observer
  * QAT activations use **moving-average** range observers, fake quant on from step 0
  * backend/toolchain — **QNNPACK**, eager-mode `torch.ao.quantization`

The experiment is frozen to the registered PyTorch 2.1 eager API; this module deliberately does NOT
migrate to torchao / PT2E even though newer PyTorch deprecates eager quantization.
"""

from __future__ import annotations

import torch
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (HistogramObserver, MovingAverageMinMaxObserver,
                                            MovingAveragePerChannelMinMaxObserver,
                                            PerChannelMinMaxObserver)

QUANT_BACKEND = "qnnpack"

# Activation range — locked by the contract, so stated explicitly.
ACT_DTYPE = torch.quint8
ACT_QSCHEME = torch.per_tensor_affine        # asymmetric
ACT_QUANT_MIN = 0
ACT_QUANT_MAX = 255                          # full 8-bit range for QNNPACK/ARM
ACT_REDUCE_RANGE = False

# Weight scheme — locked. quant_min/quant_max are deliberately NOT pinned here: the contract fixes
# only "per-channel symmetric INT8", so the observer derives the range from the dtype and the smoke
# reports whatever it actually resolves to (see the pinned-stack confirmation note in the report).
WEIGHT_DTYPE = torch.qint8
WEIGHT_QSCHEME = torch.per_channel_symmetric
WEIGHT_CH_AXIS = 0                           # Conv2d weight layout [out_ch, in_ch, kH, kW]


class QuantBackendUnavailable(RuntimeError):
    """Raised when the QNNPACK backend required by the accuracy artifact is not present."""


def select_qnnpack_backend() -> str:
    """Require and select QNNPACK. Never silently falls back to x86/fbgemm.

    A separate fbgemm/x86 build is a later EFFICIENCY-benchmark concern; the accuracy/deployment
    artifact is QNNPACK, so an unavailable backend is a loud failure here.
    """
    supported = list(torch.backends.quantized.supported_engines)
    if QUANT_BACKEND not in supported:
        raise QuantBackendUnavailable(
            f"quantization backend {QUANT_BACKEND!r} is not available in this build "
            f"(supported_engines={supported}). The E4/E5 accuracy artifact requires QNNPACK; "
            "falling back to x86/fbgemm would change the artifact and is refused.")
    torch.backends.quantized.engine = QUANT_BACKEND
    if torch.backends.quantized.engine != QUANT_BACKEND:
        raise QuantBackendUnavailable(
            f"setting torch.backends.quantized.engine to {QUANT_BACKEND!r} did not take effect "
            f"(engine is {torch.backends.quantized.engine!r})")
    return torch.backends.quantized.engine


def ptq_qconfig() -> QConfig:
    """E4 static-PTQ QConfig: histogram activations (UINT8 asym 0..255) + per-channel INT8 weights."""
    activation = HistogramObserver.with_args(
        dtype=ACT_DTYPE, qscheme=ACT_QSCHEME,
        quant_min=ACT_QUANT_MIN, quant_max=ACT_QUANT_MAX, reduce_range=ACT_REDUCE_RANGE)
    weight = PerChannelMinMaxObserver.with_args(
        dtype=WEIGHT_DTYPE, qscheme=WEIGHT_QSCHEME, ch_axis=WEIGHT_CH_AXIS)
    return QConfig(activation=activation, weight=weight)


def qat_qconfig() -> QConfig:
    """E5 QAT QConfig: moving-average fake-quant activations + per-channel INT8 weight fake-quant.

    Fake quantization is active from step 0 (contract B4 "activation quant start: step 0"); the
    caller controls later observer/BN freezing via the helpers in `src.quant.prepare`.
    """
    activation = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=ACT_DTYPE, qscheme=ACT_QSCHEME,
        quant_min=ACT_QUANT_MIN, quant_max=ACT_QUANT_MAX, reduce_range=ACT_REDUCE_RANGE)
    weight = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        dtype=WEIGHT_DTYPE, qscheme=WEIGHT_QSCHEME, ch_axis=WEIGHT_CH_AXIS)
    return QConfig(activation=activation, weight=weight)


def describe_qconfig(qconfig: QConfig) -> dict:
    """Instantiate both halves and report their ACTUAL resolved quantization semantics.

    Used by the smoke so verification inspects real observer/fake-quant state instead of trusting
    class names or constructor arguments.
    """
    def _describe(factory) -> dict:
        obj = factory()
        inner = getattr(obj, "activation_post_process", None)   # FakeQuantize wraps an observer
        info = {
            "class": type(obj).__name__,
            "observer_class": type(inner).__name__ if inner is not None else type(obj).__name__,
            "dtype": getattr(obj, "dtype", None),
            "qscheme": getattr(obj, "qscheme", None),
            "quant_min": getattr(obj, "quant_min", None),
            "quant_max": getattr(obj, "quant_max", None),
            "ch_axis": getattr(obj, "ch_axis", None),
            "reduce_range": getattr(inner if inner is not None else obj, "reduce_range", None),
            "is_fake_quant": isinstance(obj, FakeQuantize),
        }
        return info

    return {"activation": _describe(qconfig.activation), "weight": _describe(qconfig.weight)}
