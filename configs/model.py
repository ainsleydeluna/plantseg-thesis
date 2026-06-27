# Student model config — Blocker B10 scaffold (forward-smoke only)
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md (e) Architecture.
# Forward scaffold ONLY — NO quantization prepare/convert, NO training, NO loss/distillation.
# Tap indices confirmed by probing torchvision quantizable MobileNetV3-Large (dilated=True).

from configs.data import DATA

MODEL = {
    "backbone": "torchvision.models.quantization.mobilenet_v3_large",
    "quantizable": True,
    "quantize_now": False,             # QuantStub/DeQuantStub present; prepare/convert NOT run
    "dilated": True,                   # output stride 16 via C4-C5 dilation rate 2
    "low_tap_index": 6,                # OS8 low-level skip  -> 40 channels (probe-confirmed)
    "high_tap_index": 15,              # OS16 high-level C5  -> 160 channels (probe-confirmed)
    "backbone_truncate": 16,           # use features[0:16]; drop the 960-ch conv (idx 16), unused by head
    "head_inter_channels": 256,        # high-level branch projection
    "num_classes": DATA["num_classes"],   # 116 (empirically verified; see reports/dataset_report.md)
    "native_activations": "Hard-Swish/ReLU retained via quantizable TorchVision impl (NO ReLU6 substitution)",
    "pretrained": "NEED_TO_CONFIRM",   # ImageNet weights not cached; weights=None for smoke (no download)
    "input_size": (512, 512),
}
