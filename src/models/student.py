"""PlantSeg student scaffold — MobileNetV3-Large (quantizable) + LR-ASPP head.

Blockers B10 (FP32 forward scaffold) + B7b (quantization-readiness):
  - QuantStub/DeQuantStub at the boundary; head uses FloatFunctional for the attention-multiply and
    class-logit add (quantization-safe); final upsample is in float after dequant.
  - fuse() fuses Conv-BN(-ReLU) for eager-mode quantization (head + backbone blocks).
  - Still NO prepare/convert here, NO training, NO loss/distillation, NO PTQ/QAT.
Architecture per docs/IMPLEMENTATION_CONTRACT.md (e). Native Hard-Swish/ReLU retained
(no ReLU6 substitution). Output stride 16 via dilation; head taps OS8 (40ch) + OS16 C5 (160ch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.data import DATA      # noqa: E402
from configs.model import MODEL    # noqa: E402


def _build_backbone(dilated: bool, pretrained: bool):
    """Return (features Sequential, used_pretrained). For the smoke we never download weights."""
    from torchvision.models.quantization import mobilenet_v3_large
    weights = None
    used_pretrained = False
    # NOTE: pretrained download intentionally NOT performed here (cache empty). If/when ImageNet
    # weights are available, wire MobileNet_V3_Large_Weights.IMAGENET1K_V2 in and set the flag.
    model = mobilenet_v3_large(weights=weights, quantize=False, dilated=dilated)
    return model.features, used_pretrained


class _ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class LRASPPHead(nn.Module):
    """LR-ASPP-style head: additive class-logit fusion (NOT feature concatenation)."""

    def __init__(self, low_ch: int, high_ch: int, inter_ch: int, num_classes: int):
        super().__init__()
        self.high_proj = _ConvBNReLU(high_ch, inter_ch, 1)              # 1x1 Conv-BN-ReLU
        self.context = nn.Sequential(                                   # avgpool -> 1x1 -> sigmoid
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_ch, inter_ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.high_logits = nn.Conv2d(inter_ch, num_classes, 1)
        self.low_logits = nn.Conv2d(low_ch, num_classes, 1)
        self.ff_mul = FloatFunctional()   # quantization-safe element-wise multiply (attention)
        self.ff_add = FloatFunctional()   # quantization-safe class-logit add

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Return OS8 class logits. The final upsample to full resolution happens after dequant in
        the parent (matches the quantization-ready isolated head: one quantized interpolate, then a
        float final upsample). In FP32 this is numerically identical to the B10 head."""
        a = self.high_proj(high)                                       # [B,inter,H16,W16]
        s = self.context(high)                                         # [B,inter,1,1]
        high_feat = self.ff_mul.mul(a, s)                              # quantization-safe attention
        high_os8 = F.interpolate(high_feat, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.ff_add.add(self.high_logits(high_os8), self.low_logits(low))  # OS8 class logits


class PlantSegStudent(nn.Module):
    def __init__(self, num_classes: int, low_tap: int = 6, high_tap: int = 15,
                 inter_ch: int = 256, dilated: bool = True, pretrained: bool = False,
                 truncate: int = 16):
        super().__init__()
        features, used_pretrained = _build_backbone(dilated, pretrained)
        self.features = nn.Sequential(*list(features.children())[:truncate])  # drop 960-ch conv
        self.low_tap, self.high_tap = low_tap, high_tap
        self.used_pretrained = used_pretrained
        self.num_classes = num_classes
        self.low_ch, self.high_ch = self._tap_channels()
        self.head = LRASPPHead(self.low_ch, self.high_ch, inter_ch, num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    @torch.no_grad()
    def _tap_channels(self):
        h = torch.zeros(1, 3, 64, 64)
        low_ch = high_ch = None
        for i, blk in enumerate(self.features):
            h = blk(h)
            if i == self.low_tap:
                low_ch = h.shape[1]
            if i == self.high_tap:
                high_ch = h.shape[1]
                break
        return int(low_ch), int(high_ch)

    def _forward_features(self, x):
        low = high = None
        h = x
        for i, blk in enumerate(self.features):
            h = blk(h)
            if i == self.low_tap:
                low = h
            if i == self.high_tap:
                high = h
                break
        return low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        x = self.quant(x)                          # pass-through in FP32 (not prepared)
        low, high = self._forward_features(x)
        logits = self.head(low, high)              # OS8 class logits
        logits = self.dequant(logits)              # dequant BEFORE the float final upsample
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

    def fuse(self) -> "PlantSegStudent":
        """Fuse Conv-BN(-ReLU) for eager-mode quantization. Call in eval() before prepare.

        Fuses the head's 1x1 Conv-BN-ReLU and the quantizable backbone blocks via their own
        fuse_model(). FP32 behaviour is unchanged (BN folds into Conv in eval). This is NOT called
        by the FP32 forward path; it is only used ahead of quantization prepare/convert.
        """
        from torch.ao.quantization import fuse_modules
        self.eval()
        fuse_modules(self.head.high_proj, ["0", "1", "2"], inplace=True)
        for block in self.features:                # quantizable backbone blocks self-fuse
            if hasattr(block, "fuse_model"):
                block.fuse_model()
        return self


def build_student(num_classes: int | None = None, pretrained: bool = False) -> PlantSegStudent:
    """Build the FP32 student. num_classes defaults to configs/data.py (116)."""
    if num_classes is None:
        num_classes = DATA["num_classes"]
    return PlantSegStudent(
        num_classes=num_classes,
        low_tap=MODEL["low_tap_index"],
        high_tap=MODEL["high_tap_index"],
        inter_ch=MODEL["head_inter_channels"],
        dilated=MODEL["dilated"],
        pretrained=pretrained,
        truncate=MODEL["backbone_truncate"],
    )


def probe_backbone_stages(input_size=(512, 512), dilated: bool = True) -> list[dict]:
    """Run [1,3,H,W] through the full backbone and report per-stage ch / size / output stride."""
    from torchvision.models.quantization import mobilenet_v3_large
    model = mobilenet_v3_large(weights=None, quantize=False, dilated=dilated)
    model.eval()
    rows = []
    h = torch.zeros(1, 3, *input_size)
    with torch.no_grad():
        for i, blk in enumerate(model.features):
            h = blk(h)
            rows.append({
                "idx": i,
                "channels": int(h.shape[1]),
                "size": f"{int(h.shape[2])}x{int(h.shape[3])}",
                "stride": int(input_size[0] // int(h.shape[2])),
            })
    return rows
