#!/usr/bin/env python3
"""B7/B13 — eager-mode QNNPACK INT8 operator-support smoke: LR-ASPP head in ISOLATION.

Operator-support + conversion smoke ONLY (not a latency benchmark, not training, no backward).
Targets QNNPACK; if QNNPACK is absent from this torch build it runs the SAME flow under the
available engine (e.g. onednn) as a CLEARLY-LABELED non-QNNPACK PROXY. The model boundary receives
normal float tensors; QuantStub quantizes internally.

Sigmoid fallback ladder (recorded exactly): clean default -> FixedQParams(0,1; scale 1/256, zp 0)
-> DeQuant->Sigmoid->Quant float branch.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.ao.quantization as tq  # noqa: E402
from torch.ao.nn.quantized import FloatFunctional  # noqa: E402

from src.seeds import set_seed  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*reduce_range.*")

HIGH_CH, LOW_CH, INTER, NUM_CLASSES = 160, 40, 256, 116


def select_engine():
    eng = list(torch.backends.quantized.supported_engines)
    if "qnnpack" in eng:
        torch.backends.quantized.engine = "qnnpack"
        return "qnnpack", False
    proxy = "onednn" if "onednn" in eng else ("fbgemm" if "fbgemm" in eng else eng[0])
    torch.backends.quantized.engine = proxy
    return proxy, True


class QHead(nn.Module):
    """Quantization-ready LR-ASPP head matching the student head (FloatFunctional mul/add)."""

    def __init__(self, sigmoid_mode: str = "default"):
        super().__init__()
        self.sigmoid_mode = sigmoid_mode
        self.quant_high = tq.QuantStub()
        self.quant_low = tq.QuantStub()
        self.high_proj = nn.Sequential(
            nn.Conv2d(HIGH_CH, INTER, 1, bias=False), nn.BatchNorm2d(INTER), nn.ReLU())
        self.ctx_pool = nn.AdaptiveAvgPool2d(1)
        self.ctx_conv = nn.Conv2d(HIGH_CH, INTER, 1)
        self.ctx_sigmoid = nn.Sigmoid()
        if sigmoid_mode == "float_branch":
            self.ctx_dq = tq.DeQuantStub()
            self.ctx_q = tq.QuantStub()
        self.ff_mul = FloatFunctional()
        self.high_cls = nn.Conv2d(INTER, NUM_CLASSES, 1)
        self.low_cls = nn.Conv2d(LOW_CH, NUM_CLASSES, 1)
        self.ff_add = FloatFunctional()
        self.dequant = tq.DeQuantStub()

    def _context(self, high_q):
        x = self.ctx_conv(self.ctx_pool(high_q))
        if self.sigmoid_mode == "float_branch":
            return self.ctx_q(self.ctx_sigmoid(self.ctx_dq(x)))
        return self.ctx_sigmoid(x)

    def forward(self, high, low):
        high_q, low_q = self.quant_high(high), self.quant_low(low)
        a = self.high_proj(high_q)
        s = self._context(high_q)
        hi = self.ff_mul.mul(a, s)                                          # broadcast attention
        hi8 = F.interpolate(hi, size=low_q.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.ff_add.add(self.high_cls(hi8), self.low_cls(low_q))  # add class logits
        out = self.dequant(logits)
        return F.interpolate(out, size=(512, 512), mode="bilinear", align_corners=False)

    def fuse(self):
        tq.fuse_modules(self, [["high_proj.0", "high_proj.1", "high_proj.2"]], inplace=True)


def _dummy():
    return torch.randn(2, HIGH_CH, 32, 32), torch.randn(2, LOW_CH, 64, 64)


def try_quantize(engine: str, sigmoid_mode: str, fixed_sigmoid: bool = False):
    head = QHead(sigmoid_mode=sigmoid_mode).eval()
    head.fuse()
    head.qconfig = tq.get_default_qconfig(engine)
    if fixed_sigmoid:
        from torch.ao.quantization.qconfig import QConfig
        from torch.ao.quantization.observer import (
            default_fixed_qparams_range_0to1_observer, default_weight_observer)
        head.ctx_sigmoid.qconfig = QConfig(
            activation=default_fixed_qparams_range_0to1_observer, weight=default_weight_observer)
    tq.prepare(head, inplace=True)
    with torch.no_grad():                                # calibrate with a few dummy floats
        for _ in range(3):
            head(*_dummy())
    tq.convert(head, inplace=True)
    with torch.no_grad():
        out = head(*_dummy())
    return tuple(out.shape)


def main() -> int:
    set_seed()
    engine, is_proxy = select_engine()
    tag = f"{engine}{' (PROXY - NOT QNNPACK)' if is_proxy else ''}"
    print(f"[engine] {tag} | supported={list(torch.backends.quantized.supported_engines)}")
    print(f"[note] eager-mode QNNPACK INT8 OPERATOR-SUPPORT smoke (not an x86 latency benchmark)")

    ladder = [
        ("CLEAN", dict(sigmoid_mode="default")),
        ("FIXEDQPARAMS", dict(sigmoid_mode="default", fixed_sigmoid=True)),
        ("DEQUANT_SIGMOID_QUANT_FALLBACK", dict(sigmoid_mode="float_branch")),
    ]
    outcome, shape, errors = "FAIL", None, []
    for name, kw in ladder:
        try:
            shape = try_quantize(engine, **kw)
            outcome = name
            print(f"[attempt {name}] OK -> output {shape}")
            break
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:160]}"
            errors.append(f"{name}: {msg}")
            print(f"[attempt {name}] FAIL -> {msg}")

    ok = outcome != "FAIL" and shape == (2, NUM_CLASSES, 512, 512)
    print(f"[HEAD_OUTCOME] engine={engine} proxy={is_proxy} result={outcome} shape={shape}")
    print(f"[{'PASS' if ok else 'FAIL'}] qnnpack head smoke ({'PROXY' if is_proxy else 'QNNPACK'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
