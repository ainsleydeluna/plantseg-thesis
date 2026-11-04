"""The ten D-1 sites and their input-grid fake-quant (lane L-AM24: N2's correction, B1's passes).

The repository's D-1 record does not name the sites: DL-180 (and AMENDMENTS AM-22 item 2, AM-23 item 1, DL-127,
DL-146, which it cites) say only "the ten sites fake-quantized on their input's grid" (searched in this lane; see the
lane report). They are therefore derived here from the code and checked against the converted model:

  REPOSITORY-PROVEN (src/models/student.py; torchvision 0.16's quantizable MobileNetV3; torch 2.1 eager quantization)
  and MEASURED on a synthetic student (scripts/smoke_explore_sites.py): in the prepared (QAT) student, ten operations
  receive a fake-quantized tensor and emit a FLOAT result that no fake-quant follows, while the converted QNNPACK
  model computes the same operation on the quantized tensor and emits it ON ITS INPUT'S scale and zero point:
    1-8  the squeeze-excitation global average pools, features.{4,5,6,11,12,13,14,15}.block.2.avgpool
         (input: the depthwise block's output, the activation fake-quant of features.N.block.1);
    9    the LR-ASPP pool branch, head.context.0 (input: features.15's output, features.15.skip_add's fake-quant);
    10   the head's OS8 bilinear resize, F.interpolate in LRASPPHead.forward (input: head.ff_mul's output); its result
         is read only by head.high_logits, so the site is applied as a forward pre-hook on that module.
  The sigmoid and hardsigmoid outputs are NOT sites: eager prepare_qat gives each a FixedQParams fake-quant (scale
  1/256, zero point 0) in training mode, the grid the quantized kernels emit on (the observer census of record holds
  9 FixedQParamsObserver).

`SiteMatcher` installs, for any subset of the ten sites, a fake-quant that reads the producer's CURRENT scale and zero
point (so it follows the moving-average observers during training) and applies
torch.fake_quantize_per_tensor_affine with the producer's quant range: the straight-through estimator, as every
eager fake-quant. A site is skipped while its producer's fake quantization is disabled. Hooks hold no state and add
nothing to the state_dict: an epoch checkpoint of a site-matched run has the registered key set.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SE_BLOCKS = (4, 5, 6, 11, 12, 13, 14, 15)
SITE_NAMES = tuple([f"se{i}" for i in SE_BLOCKS] + ["head_pool", "head_resize"])
SE_SITES = tuple(f"se{i}" for i in SE_BLOCKS)


class SiteError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _fq_types():
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    return FakeQuantizeBase


def _last_activation_fq(module: nn.Module) -> tuple[str, nn.Module] | None:
    """The last activation fake-quant inside `module`, in module order (its output's fake-quant)."""
    fq = _fq_types()
    found = None
    for n, m in module.named_modules():
        if isinstance(m, fq) and n.split(".")[-1] == "activation_post_process":
            found = (n, m)
    return found


def resolve_sites(prepared: nn.Module) -> dict[str, dict]:
    """The ten sites of a prepared (QAT) student: {name: {kind, module_path, hook, producer_path, producer}}."""
    fq = _fq_types()
    mods = dict(prepared.named_modules())
    sites: dict[str, dict] = {}
    for i in SE_BLOCKS:
        pool_path = f"features.{i}.block.2.avgpool"
        pool = mods.get(pool_path)
        if not isinstance(pool, nn.AdaptiveAvgPool2d):
            raise SiteError("site_missing", f"{pool_path} is not an AdaptiveAvgPool2d in the prepared student")
        dw = mods.get(f"features.{i}.block.1")
        hit = None if dw is None else _last_activation_fq(dw)
        if hit is None:
            raise SiteError("producer_missing", f"features.{i}.block.1 holds no activation fake-quant")
        sites[f"se{i}"] = {"kind": "se_pool", "module_path": pool_path, "hook": "forward",
                           "producer_path": f"features.{i}.block.1.{hit[0]}", "producer": hit[1]}
    high_tap = int(getattr(prepared, "high_tap", 15))
    tap = mods.get(f"features.{high_tap}")
    if tap is None:
        raise SiteError("site_missing", f"features.{high_tap} (the high tap) is missing")
    if hasattr(tap, "skip_add") and getattr(tap, "use_res_connect", True):
        prod_path = f"features.{high_tap}.skip_add.activation_post_process"
    else:
        hit = _last_activation_fq(tap)
        prod_path = f"features.{high_tap}.{hit[0]}" if hit else None
    prod = mods.get(prod_path) if prod_path else None
    if not isinstance(prod, fq):
        raise SiteError("producer_missing", f"the high tap's output fake-quant {prod_path!r} is missing")
    pool = mods.get("head.context.0")
    if not isinstance(pool, nn.AdaptiveAvgPool2d):
        raise SiteError("site_missing", "head.context.0 is not an AdaptiveAvgPool2d")
    sites["head_pool"] = {"kind": "head_pool", "module_path": "head.context.0", "hook": "forward",
                          "producer_path": prod_path, "producer": prod}
    ffm = mods.get("head.ff_mul.activation_post_process")
    if not isinstance(ffm, fq) or mods.get("head.high_logits") is None:
        raise SiteError("producer_missing", "head.ff_mul's fake-quant or head.high_logits is missing")
    sites["head_resize"] = {"kind": "head_resize", "module_path": "head.high_logits", "hook": "forward_pre",
                            "producer_path": "head.ff_mul.activation_post_process", "producer": ffm}
    if tuple(sites) != SITE_NAMES:
        raise SiteError("site_order", f"resolved {tuple(sites)} != {SITE_NAMES}")
    return sites


def fake_quant_like(x: torch.Tensor, producer) -> torch.Tensor:
    """`x` on `producer`'s current grid (its scale, zero point and quant range), straight-through gradient."""
    if not bool(producer.fake_quant_enabled[0]):
        return x
    scale = producer.scale.detach().to(device=x.device, dtype=torch.float32).reshape(1)
    zp = producer.zero_point.detach().to(device=x.device, dtype=torch.int32).reshape(1)
    return torch.fake_quantize_per_tensor_affine(x, scale, zp, int(producer.quant_min), int(producer.quant_max))


class SiteMatcher:
    """Install input-grid fake-quants at `names` (default: all ten) on a prepared student; remove() undoes it."""

    def __init__(self, prepared: nn.Module, names=SITE_NAMES):
        unknown = sorted(set(names) - set(SITE_NAMES))
        if unknown:
            raise SiteError("site_unknown", f"unknown sites {unknown}; the sites are {SITE_NAMES}")
        self.sites = resolve_sites(prepared)
        self.names = tuple(n for n in SITE_NAMES if n in set(names))
        self.handles = []
        self.calls = {n: 0 for n in self.names}
        mods = dict(prepared.named_modules())
        for n in self.names:
            s = self.sites[n]
            mod = mods[s["module_path"]]
            if s["hook"] == "forward":
                self.handles.append(mod.register_forward_hook(self._post(n, s["producer"])))
            else:
                self.handles.append(mod.register_forward_pre_hook(self._pre(n, s["producer"])))

    def _post(self, name, producer):
        def hook(module, inputs, output):
            self.calls[name] += 1
            return fake_quant_like(output, producer)
        return hook

    def _pre(self, name, producer):
        def hook(module, inputs):
            self.calls[name] += 1
            return (fake_quant_like(inputs[0], producer),) + tuple(inputs[1:])
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def describe(self) -> list[dict]:
        return [{"site": n, "kind": self.sites[n]["kind"], "module": self.sites[n]["module_path"],
                 "hook": self.sites[n]["hook"], "input_grid_from": self.sites[n]["producer_path"]}
                for n in self.names]


def site_table(prepared: nn.Module) -> list[dict]:
    """The ten sites, SE-pool sites first, for records (no hooks installed)."""
    sites = resolve_sites(prepared)
    return [{"site": n, "kind": sites[n]["kind"], "module": sites[n]["module_path"],
             "input_grid_from": sites[n]["producer_path"]} for n in SITE_NAMES]


# ------------------------------------------------------------------ TF32 (N2; B1's TF32 pass)
def tf32_flags() -> dict:
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def set_tf32(enabled: bool) -> dict:
    """Set BOTH TF32 backend flags (cuDNN convolutions and CUDA matmul); returns the flags read back."""
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    got = tf32_flags()
    if got["cudnn_allow_tf32"] != bool(enabled) or got["cuda_matmul_allow_tf32"] != bool(enabled):
        raise SiteError("tf32_not_applied", f"setting TF32 {enabled} left {got}")
    return got
