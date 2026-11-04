"""Synthetic students for the lane's smokes (no dataset, no download): a random-init PlantSegStudent, prepared for QAT
by the pinned prepare_qat_model, its observers calibrated on random inputs in train mode, BN statistics then frozen
and observers disabled (the state a converted epoch is taken from), and its QNNPACK conversion.

`amplify` scales every squeeze-excitation fc1 weight and the LR-ASPP context conv weight by that factor and zeroes
their biases. At random init the pooled vectors sit far below one step of their input's grid and the gates absorb
any change (MEASURED: site matching leaves the logits bit-identical), unlike a trained student whose gates respond to
the pooled vector; the amplified student makes the sites' rounding reach the logits, so N2's claim can be shown end
to end on random inputs.
"""
from __future__ import annotations

import torch

SE_BLOCKS = (4, 5, 6, 11, 12, 13, 14, 15)


def random_student(*, seed: int = 0, amplify: float | None = None, fp32_head: bool = False):
    from src.models.student import build_student
    torch.manual_seed(seed)
    m = build_student(num_classes=116, pretrained=False)
    if amplify is not None:
        with torch.no_grad():
            for i in SE_BLOCKS:
                se = m.features[i].block[2]
                se.fc1.weight.mul_(float(amplify))
                se.fc1.bias.zero_()
            m.head.context[1].weight.mul_(float(amplify))
            m.head.context[1].bias.zero_()
    if fp32_head:
        from .fp32_head import apply_fp32_head
        apply_fp32_head(m)
    return m


def calibrated_prepared(model, *, steps: int = 6, batch: int = 4, size: int = 128, seed: int = 1):
    """prepare_qat_model, `steps` train-mode forwards on random inputs (observers move), BN frozen, observers off."""
    from src.quant.prepare import disable_observers, freeze_bn_stats, prepare_qat_model
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    p = prepare_qat_model(model, select_backend=False)
    p.train()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(steps):
            p(torch.randn(batch, 3, size, size, generator=g))
    freeze_bn_stats(p)
    disable_observers(p)
    return p.eval()


def converted_of(prepared):
    from src.quant.prepare import convert_model
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    return convert_model(prepared).eval()
