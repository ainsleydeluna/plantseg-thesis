"""N5: E6-E2 with the LR-ASPP logit head in FP32 (addendum L-AM24-ADD-1, from MVA-AUDIT-1; lane L-AM24, the QAT entry's --fp32-head).

The head's two 1x1 classifiers (head.high_logits, head.low_logits), their add (head.ff_add) and the final upsample
stay in float: in training no fake-quant is attached to them (each carries qconfig None, which prepare_qat's qconfig
propagation keeps), and in the converted model a DeQuantize precedes each classifier (head.dequant_high after the
OS8 resize, head.dequant_low on the low tap), so they run as float nn.Conv2d and a float add. The final upsample was
already float, after the student's own DeQuantStub (src/models/student.py), which receives float logits here (a
dequantize of a float tensor returns it). Everything before the dequant points keeps the registered quantization:
the backbone, head.high_proj, the context branch and head.ff_mul are INT8 (QNNPACK) as in E6-E2.

THE ARTIFACT IS MIXED PRECISION. Its records say so (PRECISION_LABEL) and give the share of multiply-accumulates left
in FP32; it is never called INT8, never an "int8_qat" artifact, and never scored as one by the evaluator of record.

src/models/student.py is not modified: `apply_fp32_head` replaces the student's head with `Fp32LogitHead`, which
holds the SAME submodules (same names, same parameters) plus two DeQuantStubs, so a checkpoint's head weights load
unchanged and the state_dict differs from the registered prepared student only where the FP32 islands remove
fake-quant entries.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import DeQuantStub

PRECISION_LABEL = ("MIXED PRECISION (not INT8): INT8 QNNPACK backbone, head.high_proj, context branch and ff_mul; "
                   "FP32 LR-ASPP logit head (head.high_logits, head.low_logits, head.ff_add) and the final upsample")
FP32_ISLANDS = ("head.high_logits", "head.low_logits", "head.ff_add")
DEQUANT_POINTS = ("head.dequant_high", "head.dequant_low")


class Fp32HeadError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


class Fp32LogitHead(nn.Module):
    """LRASPPHead's forward with a dequant before each classifier; the classifiers and the add carry qconfig None."""

    def __init__(self, head: nn.Module):
        super().__init__()
        for name in ("high_proj", "context", "high_logits", "low_logits", "ff_mul", "ff_add"):
            setattr(self, name, getattr(head, name))
        self.dequant_high = DeQuantStub()
        self.dequant_low = DeQuantStub()
        for name in ("high_logits", "low_logits", "ff_add"):
            getattr(self, name).qconfig = None

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        a = self.high_proj(high)
        s = self.context(high)
        high_feat = self.ff_mul.mul(a, s)
        high_os8 = F.interpolate(high_feat, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.ff_add.add(self.high_logits(self.dequant_high(high_os8)), self.low_logits(self.dequant_low(low)))


def apply_fp32_head(student: nn.Module) -> nn.Module:
    """Replace `student.head` in place (before prepare_qat_model). Refuses a head already replaced."""
    if isinstance(student.head, Fp32LogitHead):
        raise Fp32HeadError("already_applied", "the student already carries the FP32 logit head")
    student.head = Fp32LogitHead(student.head)
    return student


def _is_quantized_module(m: nn.Module) -> bool:
    return type(m).__module__.startswith(("torch.ao.nn.quantized", "torch.ao.nn.intrinsic.quantized",
                                          "torch.nn.quantized", "torch.nn.intrinsic.quantized"))


def training_islands_report(prepared: nn.Module) -> dict:
    """In the prepared (QAT) model: the islands carry no fake-quant (no weight fake-quant, no output observer)."""
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    mods = dict(prepared.named_modules())
    fq_in = {k: sorted(n for n, m in prepared.named_modules() if n.startswith(k + ".") and
                       isinstance(m, FakeQuantizeBase)) for k in FP32_ISLANDS}
    return {"islands": {k: {"type": type(mods[k]).__name__, "fake_quant_modules": fq_in[k],
                            "qconfig_none": getattr(mods[k], "qconfig", "absent") is None} for k in FP32_ISLANDS},
            "ok": all(not v for v in fq_in.values())
            and all(type(mods[k]) in (nn.Conv2d,) for k in ("head.high_logits", "head.low_logits"))}


@torch.no_grad()
def converted_islands_report(converted: nn.Module, x: torch.Tensor) -> dict:
    """In the converted model: the islands are float modules fed float tensors; every other conv is quantized and fed
    quantized tensors; the dequant points are DeQuantize modules."""
    mods = dict(converted.named_modules())
    seen: dict[str, dict] = {}
    hooks = []
    convs = [n for n, m in converted.named_modules() if not list(m.children())
             and (isinstance(m, nn.Conv2d) or _is_quantized_module(m)) and "conv" in type(m).__name__.lower()]
    for n in convs + [k for k in FP32_ISLANDS if k != "head.ff_add"]:
        hooks.append(mods[n].register_forward_hook(
            lambda m, i, o, n=n: seen.__setitem__(n, {"in_quantized": bool(i[0].is_quantized),
                                                      "out_quantized": bool(o.is_quantized)})))
    y = converted(x)
    for h in hooks:
        h.remove()
    island_convs = ("head.high_logits", "head.low_logits")
    rest = [n for n in convs if n not in island_convs]
    report = {
        "island_types": {k: f"{type(mods[k]).__module__}.{type(mods[k]).__name__}" for k in FP32_ISLANDS},
        "dequant_types": {k: type(mods[k]).__name__ for k in DEQUANT_POINTS},
        "islands_float_in_float_out": all(seen.get(k) == {"in_quantized": False, "out_quantized": False}
                                          for k in island_convs),
        "islands_are_float_modules": (all(type(mods[k]) is nn.Conv2d for k in island_convs)
                                      and type(mods["head.ff_add"]).__name__ == "FloatFunctional"),
        "dequant_points_converted": all(type(mods[k]).__name__ == "DeQuantize" for k in DEQUANT_POINTS),
        "quantized_convs": len(rest),
        "rest_quantized_in_and_out": all(_is_quantized_module(mods[n]) and seen.get(n, {}).get("in_quantized")
                                         and seen.get(n, {}).get("out_quantized") for n in rest),
        "rest_not_quantized": [n for n in rest if not _is_quantized_module(mods[n])],
        "output_float": not y.is_quantized, "output_shape": tuple(y.shape)}
    report["ok"] = all(report[k] for k in ("islands_float_in_float_out", "islands_are_float_modules",
                                           "dequant_points_converted", "rest_quantized_in_and_out", "output_float"))
    return report


@torch.no_grad()
def mac_share(float_student: nn.Module, size=(512, 512)) -> dict:
    """Multiply-accumulates of every conv in the float student at `size`, and the share in the FP32 islands."""
    macs: dict[str, int] = {}
    hooks = []
    for n, m in float_student.named_modules():
        if isinstance(m, nn.Conv2d):
            def h(mod, i, o, n=n):
                k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
                macs[n] = int(o.numel() // o.shape[0]) * k
            hooks.append(m.register_forward_hook(h))
    was = float_student.training
    float_student.eval()
    float_student(torch.zeros(1, 3, *size))
    float_student.train(was)
    for hk in hooks:
        hk.remove()
    total = sum(macs.values())
    island = sum(v for k, v in macs.items() if k in ("head.high_logits", "head.low_logits"))
    return {"input": list(size), "conv_macs_total": total, "fp32_island_conv_macs": island,
            "fp32_mac_share": island / total if total else None,
            "note": "conv multiply-accumulates only; the float add and the two bilinear resizes are not counted"}


# ------------------------------------------------------------------ N5's own conversion (its block B)
def prepared_skeleton_fp32head() -> nn.Module:
    from src.models.student import build_student
    from src.quant.prepare import prepare_qat_model
    return prepare_qat_model(apply_fp32_head(build_student(num_classes=116, pretrained=False)), select_backend=False)


def convert_state(state: dict) -> tuple[nn.Module, dict]:
    """An N5 epoch's pre-convert state -> the converted mixed-precision model, through the pinned copy and convert."""
    from src.quant import qat_artifacts as A
    from src.quant.prepare import convert_model, disable_observers
    from src.quant.qconfig import select_qnnpack_backend
    skel = prepared_skeleton_fp32head()
    copy = A.load_state_by_name(skel, state)
    disable_observers(skel)
    select_qnnpack_backend()
    conv = convert_model(skel)
    return conv.eval(), copy
