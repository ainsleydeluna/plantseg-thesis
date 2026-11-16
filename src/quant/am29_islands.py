"""AM-29 Step 1's FP32 islands (lane L-AM29, P2): QNNPACK INT8 conversion with named module groups kept in FP32.

A variant is the all-INT8 conversion of record (G0) with the modules of one group swapped for their FP32 forms, each
kept region entered through a DeQuantize and left through a Quantize carrying the qparams the G0 model uses at that
point (its stored qparams: re-entry changes nothing downstream). Two sources:

  (a) a QAT state (an epoch checkpoint eNN.pt): src/quant/qat_artifacts.py prepared_skeleton + load_state_by_name,
      then convert_model, exactly as _convert_checked converts an epoch (G0). An island's float module is the QAT
      module's to_float(): its weight unquantized, a ConvBn's BN folded from the frozen running statistics; no
      fake-quant module survives inside an island.
  (b) an FP32 checkpoint through the PTQ of record: prepare_ptq, calibrate (one image per batch, the list's order),
      convert_model, exactly as src/quant/ptq.py quantize_qnnpack (G0). An island's float module is the calibrated
      FP32 module (BN folded at fusion, from the checkpoint's statistics), copied without its observer.

The groups, by module path of src/models/student.py (the SE blocks found by a module walk, SE_PATHS):

  G0  none (all-INT8)
  G1  the logit head: head.high_logits, head.low_logits (each dequantized before), head.ff_add (a float add); the
      region ends at the model's float output, so it has no re-entry (the output DeQuantize passes a float tensor)
  G2  the eight SE gates (AM-24 item 11(j)(ii): pool, fc, fc, hardsigmoid, mul): features.{4,5,6,11,12,13,14,15}.block.2
  G3  the LR-ASPP context branch: head.context (pool, 1x1 conv, sigmoid)
  G4  AM-24 item 11(b)'s ten sites: the eight SE global average pools features.{i}.block.2.avgpool, the context pool
      head.context.0, and the one quantized bilinear interpolate (LRASPPHead.forward, src/models/student.py:99, between
      head.ff_mul and head.high_logits: ff_mul's output is dequantized and high_logits' input re-quantized)

A variant is eligible only if at least 90% of its parameters are INT8. Shares: parameters are the weights and biases of
every convolution of the converted model (BN is folded); MACs are convolutions (output elements x in/groups x kh x kw),
element-wise add and mul (one per output element), average pools (one per input element) and the head interpolate
(four per output element), at the 1x3x512x512 canvas. The final float upsample after the output DeQuantize is the
declared float boundary of every variant, G0 included: reported apart, and in a second share that counts it.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import DeQuantize, FloatFunctional, QFunctional, Quantize

REPO = Path(__file__).resolve().parents[2]

SE_BLOCKS = (4, 5, 6, 11, 12, 13, 14, 15)
SE_PATHS = tuple(f"features.{i}.block.2" for i in SE_BLOCKS)
INTERP_SITE = "head.interpolate"
GROUPS = {
    "G0": (),
    "G1": ("head.high_logits", "head.low_logits", "head.ff_add"),
    "G2": SE_PATHS,
    "G3": ("head.context",),
    "G4": tuple(f"{p}.avgpool" for p in SE_PATHS) + ("head.context.0", INTERP_SITE),
}
GROUP_TEXT = {
    "G0": "none (all-INT8)",
    "G1": "the logit head (head.high_logits, head.low_logits, head.ff_add)",
    "G2": "the eight SE gates (features.{4,5,6,11,12,13,14,15}.block.2: pool, fc1, fc2, hardsigmoid, mul)",
    "G3": "the LR-ASPP context branch (head.context: pool, 1x1 conv, sigmoid)",
    "G4": "AM-24 item 11(b)'s ten sites (eight SE pools, the context pool, the head's quantized interpolate)",
}
ELIGIBLE_PARAM_SHARE = 0.90
EQUIVALENCE_TOL = 1e-5
LABEL_INT8 = "int8"
LABEL_MIXED = "mixed precision"
CANVAS = (1, 3, 512, 512)
_QCONV_TYPES = ("torch.ao.nn.quantized.modules.conv.Conv2d", "torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d")


class IslandError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def label_of(group: str) -> str:
    """G0 is the INT8 artifact; every other variant is mixed precision, never INT8."""
    return LABEL_INT8 if group == "G0" else LABEL_MIXED


class Island(nn.Module):
    """One kept FP32 region: DeQuantize at its entry, the float body, and a Quantize at its exit with the stored
    qparams of that point (none where the region ends at the model's float output)."""

    def __init__(self, body: nn.Module, *, exit_qparams: tuple[float, int] | None):
        super().__init__()
        self.dequant = DeQuantize()
        self.body = body
        self.quant = None if exit_qparams is None else Quantize(float(exit_qparams[0]), int(exit_qparams[1]),
                                                                torch.quint8)

    def forward(self, x):
        y = self.body(self.dequant(x))
        return y if self.quant is None else self.quant(y)


# ------------------------------------------------------------------ the module walk
def se_paths_of(model: nn.Module) -> tuple[str, ...]:
    return tuple(n for n, m in model.named_modules() if type(m).__name__.endswith("SqueezeExcitation"))


def check_sites(model: nn.Module) -> dict:
    """The groups' paths exist in this model: the SE blocks are exactly SE_PATHS, the head is LR-ASPP's."""
    found = se_paths_of(model)
    mods = dict(model.named_modules())
    missing = [p for g in GROUPS.values() for p in g if p != INTERP_SITE and p not in mods]
    ok = found == SE_PATHS and not missing
    if not ok:
        raise IslandError("sites_differ", f"SE blocks {found} (expected {SE_PATHS}); missing paths {missing}")
    return {"se_paths": list(found), "groups": {g: list(p) for g, p in GROUPS.items()}, "ok": True}


def _set(model: nn.Module, path: str, module: nn.Module) -> None:
    parent, _, name = path.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, name, module)


# ------------------------------------------------------------------ float modules from the two sources
def float_from_qat(m: nn.Module) -> nn.Module:
    """A QAT conv's FP32 form: to_float() (weight unquantized; a ConvBn's BN folded from its running statistics)."""
    import torch.ao.nn.intrinsic.qat as nniqat
    import torch.ao.nn.qat as nnqat
    if not isinstance(m, (nnqat.Conv2d, nniqat.ConvReLU2d, nniqat.ConvBn2d, nniqat.ConvBnReLU2d)):
        raise IslandError("not_a_qat_conv", f"{type(m).__name__} is not a QAT conv module")
    out = m.to_float()
    out.eval()
    return out


def float_from_ptq(m: nn.Module) -> nn.Module:
    """A calibrated PTQ conv's FP32 form: a fresh conv of the same shape holding its (BN-folded) weights, no observer."""
    import torch.ao.nn.intrinsic as nni

    def fresh(c: nn.Conv2d) -> nn.Conv2d:
        n = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, c.dilation, c.groups,
                      c.bias is not None, c.padding_mode)
        with torch.no_grad():
            n.weight.copy_(c.weight.detach())
            if c.bias is not None:
                n.bias.copy_(c.bias.detach())
        return n.eval()
    if isinstance(m, nni.ConvReLU2d):
        return nni.ConvReLU2d(fresh(m[0]), nn.ReLU()).eval()
    if type(m) is nn.Conv2d:
        return fresh(m)
    raise IslandError("not_a_ptq_conv", f"{type(m).__name__} is not a calibrated float conv")


# ------------------------------------------------------------------ G0 and the qparams it stores
@torch.no_grad()
def stored_qparams(g0: nn.Module, x: torch.Tensor | None = None) -> dict:
    """The qparams the all-INT8 model carries at each re-entry point, read from its quantized tensors (static: a
    module's scale and zero point, never the input's values)."""
    x = torch.zeros(1, 3, 64, 64) if x is None else x
    out, hooks = {}, []
    mods = dict(g0.named_modules())

    def post(name):
        def h(_m, _i, y):
            out[name] = (float(y.q_scale()), int(y.q_zero_point()))
        return h
    for p in SE_PATHS:
        hooks.append(mods[p].register_forward_hook(post(p)))
        hooks.append(mods[f"{p}.avgpool"].register_forward_hook(post(f"{p}.avgpool")))
    hooks.append(mods["head.context"].register_forward_hook(post("head.context")))
    hooks.append(mods["head.context.0"].register_forward_hook(post("head.context.0")))
    hooks.append(mods["head.high_logits"].register_forward_pre_hook(
        lambda _m, inp: out.__setitem__(INTERP_SITE, (float(inp[0].q_scale()), int(inp[0].q_zero_point())))))
    try:
        g0.eval()
        g0(x)
    finally:
        for h in hooks:
            h.remove()
    return out


def build_variant(g0: nn.Module, group: str, float_conv_of) -> tuple[nn.Module, dict]:
    """Variant `group` of the converted model `g0`; `float_conv_of(path)` gives the FP32 form of the conv at `path`."""
    if group not in GROUPS:
        raise IslandError("group_unknown", f"group {group!r} is not one of {sorted(GROUPS)}")
    check_sites(g0)
    model = copy.deepcopy(g0).eval()
    if group == "G0":
        return model, {"group": group, "islands": [], "reentry_qparams": {}}
    qp = stored_qparams(g0)
    islands, reentry = [], {}
    if group == "G1":
        _set(model, "head.high_logits", Island(float_conv_of("head.high_logits"), exit_qparams=None))
        _set(model, "head.low_logits", Island(float_conv_of("head.low_logits"), exit_qparams=None))
        _set(model, "head.ff_add", FloatFunctional())
        islands = ["head.high_logits", "head.low_logits", "head.ff_add"]
    elif group == "G2":
        for p in SE_PATHS:
            se = copy.deepcopy(g0.get_submodule(p))
            se.fc1 = float_conv_of(f"{p}.fc1")
            se.fc2 = float_conv_of(f"{p}.fc2")
            se.skip_mul = FloatFunctional()
            _set(model, p, Island(se, exit_qparams=qp[p]))
            islands.append(p)
            reentry[p] = qp[p]
    elif group == "G3":
        ctx = nn.Sequential(copy.deepcopy(g0.head.context[0]), float_conv_of("head.context.1"),
                            copy.deepcopy(g0.head.context[2]))
        _set(model, "head.context", Island(ctx, exit_qparams=qp["head.context"]))
        islands, reentry = ["head.context"], {"head.context": qp["head.context"]}
    elif group == "G4":
        for p in (*(f"{s}.avgpool" for s in SE_PATHS), "head.context.0"):
            _set(model, p, Island(copy.deepcopy(g0.get_submodule(p)), exit_qparams=qp[p]))
            islands.append(p)
            reentry[p] = qp[p]
        model.head.ff_mul.activation_post_process = DeQuantize()
        s, z = qp[INTERP_SITE]
        _set(model, "head.high_logits", nn.Sequential(Quantize(s, z, torch.quint8), copy.deepcopy(g0.head.high_logits)))
        islands.append(INTERP_SITE)
        reentry[INTERP_SITE] = qp[INTERP_SITE]
    return model.eval(), {"group": group, "islands": islands, "reentry_qparams": {k: list(v) for k, v in reentry.items()}}


# ------------------------------------------------------------------ the two sources
def convert_qat_state(state: dict, groups=tuple(GROUPS)) -> dict:
    """(a) A QAT state to every requested variant. Returns {"prepared": the fake-quant skeleton holding the state,
    "g0": the converted model, "copy": the exact-copy report, "variants": {group: (model, report)}}."""
    from . import qat_artifacts as A
    from .prepare import convert_model
    from .qconfig import select_qnnpack_backend
    skeleton = A.prepared_skeleton()
    copy_rep = A.load_state_by_name(skeleton, state)
    select_qnnpack_backend()
    g0 = convert_model(skeleton)                    # qat_artifacts._convert_checked's conversion, unchanged
    variants = {g: build_variant(g0, g, lambda p: float_from_qat(skeleton.get_submodule(p))) for g in groups}
    return {"prepared": skeleton, "g0": g0, "copy": copy_rep, "variants": variants, "source": "qat"}


def calibrate_fp32(model: nn.Module, images) -> tuple[nn.Module, int]:
    """(b) prepare_ptq + calibrate under QNNPACK, as src/quant/ptq.py quantize_qnnpack does before its convert."""
    from .prepare import calibrate, prepare_ptq
    from .qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    prepared = prepare_ptq(model, select_backend=False)
    n = calibrate(prepared, images)
    return prepared, n


def convert_ptq_prepared(prepared: nn.Module, groups=tuple(GROUPS)) -> dict:
    from .prepare import convert_model
    if torch.backends.quantized.engine != "qnnpack":
        raise IslandError("wrong_engine", f"the quantized engine is {torch.backends.quantized.engine!r}")
    g0 = convert_model(prepared).eval()            # src/quant/ptq.py quantize_qnnpack's conversion, unchanged
    variants = {g: build_variant(g0, g, lambda p: float_from_ptq(prepared.get_submodule(p))) for g in groups}
    return {"prepared": prepared, "g0": g0, "variants": variants, "source": "ptq"}


# ------------------------------------------------------------------ the checks
def fake_quant_report(model: nn.Module) -> dict:
    """No FakeQuantize anywhere in the converted variant (so none inside an island), and no observer."""
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    from torch.ao.quantization.observer import ObserverBase
    fq = [n for n, m in model.named_modules() if isinstance(m, FakeQuantizeBase)]
    obs = [n for n, m in model.named_modules() if isinstance(m, ObserverBase)]
    bn = [n for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    return {"fake_quant_modules": fq, "observers": obs, "batchnorm_modules": bn, "ok": not fq and not obs and not bn}


def _island_bodies(model: nn.Module) -> dict:
    return {n: m for n, m in model.named_modules() if isinstance(m, Island)}


def bn_folded_report(prepared_qat: nn.Module, variant: nn.Module, report: dict) -> dict:
    """Every conv inside an island holds its QAT module's weights with BN folded from the frozen running statistics
    (fuse_conv_bn_weights on the QAT module's bn buffers), or the QAT weights unchanged where the conv has no BN."""
    import torch.ao.nn.intrinsic.qat as nniqat
    from torch.nn.utils.fusion import fuse_conv_bn_weights
    rows = []
    for name, isl in _island_bodies(variant).items():
        for sub, m in isl.body.named_modules():
            if type(m) is not nn.Conv2d:
                continue
            src_path = _qat_path_of(name, sub)
            q = prepared_qat.get_submodule(src_path)
            if isinstance(q, (nniqat.ConvBn2d, nniqat.ConvBnReLU2d)):
                w, b = fuse_conv_bn_weights(q.weight, q.bias, q.bn.running_mean, q.bn.running_var, q.bn.eps,
                                            q.bn.weight, q.bn.bias)
                kind = "convbn: folded from running statistics"
            else:
                w, b = q.weight, q.bias
                kind = "conv without BN: weights unchanged"
            same = torch.equal(m.weight, w.detach()) and ((m.bias is None and b is None)
                                                          or torch.equal(m.bias, b.detach()))
            rows.append({"island": name, "module": src_path, "kind": kind, "equal": bool(same)})
    return {"convs": rows, "ok": all(r["equal"] for r in rows)}


def _qat_path_of(island: str, sub: str) -> str:
    """The prepared module a conv inside an island came from (G1: the island path; G2: <se>.fc1/fc2; G3: head.context.1)."""
    if island.startswith("features.") and island.endswith(".block.2"):
        return f"{island}.{sub.split('.')[0]}"
    if island == "head.context":
        return "head.context.1"
    return island


@torch.no_grad()
def equivalence_report(prepared_qat: nn.Module, variant: nn.Module, report: dict,
                       x: torch.Tensor | None = None) -> dict:
    """Each island of the variant against its QAT module with fake quantization and observers off (eval mode): one
    forward of the variant records each island body's float input and output, and the fake-quant-off module is run on
    that same input; max |difference| <= 1e-5. The interpolate site is compared with the float bilinear call of
    LRASPPHead.forward on the tensor the variant dequantized; head.ff_add with the fake-quant-off add on the two float
    logit maps the variant produced."""
    from torch.ao.quantization import disable_fake_quant, disable_observer
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(29)) if x is None else x
    fqoff = copy.deepcopy(prepared_qat)
    fqoff.apply(disable_fake_quant)
    fqoff.apply(disable_observer)
    fqoff.eval()
    vm = dict(variant.named_modules())
    cap, hooks = {}, []
    for p in report["islands"]:
        if p == INTERP_SITE:
            hooks.append(variant.head.ff_mul.activation_post_process.register_forward_hook(
                lambda _m, _i, y: cap.__setitem__("interp_in", y)))
            hooks.append(vm["head.high_logits"][0].register_forward_pre_hook(
                lambda _m, inp: cap.__setitem__("interp_out", inp[0])))
        elif p != "head.ff_add":
            hooks.append(vm[p].body.register_forward_hook(
                lambda _m, inp, y, _p=p: cap.__setitem__(_p, (inp[0], y))))
    try:
        variant.eval()
        variant(x)
    finally:
        for h in hooks:
            h.remove()
    rows = []
    for p in report["islands"]:
        if p == INTERP_SITE:
            src, got = cap["interp_in"], cap["interp_out"]
            ref = F.interpolate(src, size=got.shape[-2:], mode="bilinear", align_corners=False)
            rows.append({"island": p, "input_float": not src.is_quantized, "max_abs_diff": float((got - ref).abs().max()),
                         "reference": "F.interpolate(bilinear, align_corners=False), LRASPPHead.forward's call"})
        elif p == "head.ff_add":
            h, lo = cap["head.high_logits"][1], cap["head.low_logits"][1]
            a = vm["head.ff_add"].add(h, lo)
            b = fqoff.head.ff_add.add(h, lo)
            rows.append({"island": p, "input_float": True, "max_abs_diff": float((a - b).abs().max()),
                         "reference": "the fake-quant-off head.ff_add"})
        else:
            inp, y = cap[p]
            ref = fqoff.get_submodule(p)(inp)
            rows.append({"island": p, "input_float": not inp.is_quantized, "max_abs_diff": float((y - ref).abs().max()),
                         "reference": f"the fake-quant-off {p}"})
    worst = max((r["max_abs_diff"] for r in rows), default=0.0)
    return {"rows": rows, "max_abs_diff": worst, "tolerance": EQUIVALENCE_TOL,
            "ok": worst <= EQUIVALENCE_TOL and all(r["input_float"] for r in rows)}


@torch.no_grad()
def reentry_report(g0: nn.Module, variant: nn.Module, report: dict) -> dict:
    """Every exit Quantize carries the qparams G0 holds at that point, and, where a module stores them, those too
    (an SE block's skip_mul and the head's ff_mul as QFunctional scale/zero_point; the sigmoid's fixed 1/256, 0)."""
    stored = stored_qparams(g0)
    rows = []
    vm = dict(variant.named_modules())
    for p, (s, z) in report["reentry_qparams"].items():
        q = vm["head.high_logits"][0] if p == INTERP_SITE else vm[p].quant
        got = (float(q.scale), int(q.zero_point))
        module_qp = None
        if p in SE_PATHS:
            m = g0.get_submodule(f"{p}.skip_mul")
            module_qp = (float(m.scale), int(m.zero_point))
        elif p == INTERP_SITE:
            module_qp = (float(g0.head.ff_mul.scale), int(g0.head.ff_mul.zero_point))
        elif p == "head.context":
            module_qp = (1.0 / 256.0, 0)
        ok = got == tuple(stored[p]) == (s, z) and (module_qp is None or got == module_qp)
        rows.append({"site": p, "quantize": list(got), "g0_tensor": list(stored[p]),
                     "stored_in_module": None if module_qp is None else list(module_qp), "equal": ok})
    return {"rows": rows, "ok": all(r["equal"] for r in rows)}


@torch.no_grad()
def mixing_report(model: nn.Module, x: torch.Tensor | None = None) -> dict:
    """No add or mul mixes quantized and float tensors: every QFunctional add/mul gets two quantized tensors and every
    FloatFunctional add/mul two float tensors (the instances' methods are wrapped for one forward, then restored)."""
    x = torch.zeros(1, 3, 64, 64) if x is None else x
    calls, restore = [], []
    for name, m in model.named_modules():
        if not isinstance(m, (QFunctional, FloatFunctional)):
            continue
        for op in ("add", "mul"):
            orig = getattr(m, op)

            def wrap(a, b, _orig=orig, _n=name, _op=op, _cls=type(m).__name__):
                calls.append({"module": _n, "op": _op, "class": _cls,
                              "inputs_quantized": [bool(a.is_quantized), bool(b.is_quantized)]})
                return _orig(a, b)
            setattr(m, op, wrap)
            restore.append((m, op))
    try:
        model.eval()
        model(x)
    finally:
        for m, op in restore:
            delattr(m, op)
    bad = [c for c in calls if c["inputs_quantized"] != ([True, True] if c["class"] == "QFunctional" else [False, False])]
    return {"calls": len(calls), "mixed": bad, "ok": bool(calls) and not bad}


@torch.no_grad()
def shares(model: nn.Module, x_shape=CANVAS) -> dict:
    """INT8 shares of parameters and MACs of a converted variant (the conventions are the module docstring's)."""
    params = {"int8": 0, "fp32": 0}
    for _n, m in model.named_modules():
        tname = f"{type(m).__module__}.{type(m).__name__}"
        if tname in _QCONV_TYPES:
            n = m.weight().numel() + (m.bias().numel() if m.bias() is not None else 0)
            params["int8"] += n
        elif type(m) is nn.Conv2d:
            params["fp32"] += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    macs = {"int8": 0, "fp32": 0, "boundary_fp32": 0}
    by_kind: dict = {}
    hooks, restore = [], []

    def add(kind, precision, n):
        macs[precision] += int(n)
        key = f"{kind} ({precision})"
        by_kind[key] = by_kind.get(key, 0) + int(n)

    for _name, m in model.named_modules():
        tname = f"{type(m).__module__}.{type(m).__name__}"
        if tname in _QCONV_TYPES or type(m) is nn.Conv2d:
            def conv_hook(mod, _inp, y, _q=tname in _QCONV_TYPES):
                kh, kw = mod.kernel_size
                add("conv", "int8" if _q else "fp32", y.numel() * (mod.in_channels // mod.groups) * kh * kw)
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            hooks.append(m.register_forward_hook(
                lambda _m, inp, _y: add("avgpool", "int8" if inp[0].is_quantized else "fp32", inp[0].numel())))
        elif isinstance(m, (QFunctional, FloatFunctional)):
            for op in ("add", "mul"):
                orig = getattr(m, op)

                def wrap(a, b, _orig=orig, _op=op, _q=isinstance(m, QFunctional)):
                    y = _orig(a, b)
                    add(_op, "int8" if _q else "fp32", y.numel())
                    return y
                setattr(m, op, wrap)
                restore.append((m, op))
    hl = model.get_submodule("head.high_logits")
    hooks.append(hl.register_forward_pre_hook(
        lambda _m, inp: add("head interpolate", "int8" if inp[0].is_quantized else "fp32", 4 * inp[0].numel())))
    try:
        model.eval()
        y = model(torch.zeros(*x_shape))
    finally:
        for h in hooks:
            h.remove()
        for m, op in restore:
            delattr(m, op)
    macs["boundary_fp32"] = 4 * y.numel()
    by_kind["final float upsample (declared boundary, fp32)"] = macs["boundary_fp32"]
    total_p = params["int8"] + params["fp32"]
    total_m = macs["int8"] + macs["fp32"]
    return {"params": params, "params_total": total_p,
            "int8_param_share": params["int8"] / total_p if total_p else None,
            "macs": macs, "macs_by_kind": dict(sorted(by_kind.items())), "macs_total_network": total_m,
            "int8_mac_share": macs["int8"] / total_m if total_m else None,
            "int8_mac_share_with_boundary": macs["int8"] / (total_m + macs["boundary_fp32"]) if total_m else None,
            "canvas": list(x_shape)}


def eligible(share_report: dict) -> bool:
    s = share_report.get("int8_param_share")
    return isinstance(s, float) and math.isfinite(s) and s >= ELIGIBLE_PARAM_SHARE


def variant_record(group: str, model: nn.Module, report: dict, *, g0: nn.Module, prepared_qat: nn.Module | None) -> dict:
    """The per-variant record: label, members, shares, eligibility and the checks of AM-29 P2."""
    sh = shares(model)
    rec = {"group": group, "text": GROUP_TEXT[group], "members": list(GROUPS[group]), "label": label_of(group),
           "islands": report["islands"], "reentry_qparams": report["reentry_qparams"], "shares": sh,
           "eligible": eligible(sh), "eligibility_rule": f"int8_param_share >= {ELIGIBLE_PARAM_SHARE}",
           "checks": {"no_fake_quant_observer_or_bn": fake_quant_report(model), "no_mixed_add_mul": mixing_report(model)}}
    if group != "G0":
        rec["checks"]["reentry_reuses_stored_qparams"] = reentry_report(g0, model, report)
        if prepared_qat is not None:
            rec["checks"]["bn_folded_from_frozen_statistics"] = bn_folded_report(prepared_qat, model, report)
            rec["checks"]["equals_fake_quant_off_module"] = equivalence_report(prepared_qat, model, report)
    rec["checks_ok"] = all(c.get("ok") for c in rec["checks"].values())
    return rec


__all__ = ["CANVAS", "ELIGIBLE_PARAM_SHARE", "EQUIVALENCE_TOL", "GROUPS", "GROUP_TEXT", "INTERP_SITE", "Island",
           "IslandError", "LABEL_INT8", "LABEL_MIXED", "SE_BLOCKS", "SE_PATHS", "bn_folded_report", "build_variant",
           "calibrate_fp32", "check_sites", "convert_ptq_prepared", "convert_qat_state", "eligible",
           "equivalence_report", "fake_quant_report", "float_from_ptq", "float_from_qat", "label_of", "mixing_report",
           "reentry_report", "se_paths_of", "shares", "stored_qparams", "variant_record"]
