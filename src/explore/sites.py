"""The ten D-1 sites and their input-grid fake-quant (lane L-AM24: N2's correction, B1's passes).

AM-24 v2.4 (approved 4 Nov) names them: the eight squeeze-excitation blocks' global average pools, the LR-ASPP head's
context pool (head.context.0) and the head's one quantized bilinear interpolate (LRASPPHead.forward), as
src/models/student.py builds them at dc40dd2. Before v2.4 no repository record named them (DL-180 and what it cites
say only "the ten sites fake-quantized on their input's grid"), so this lane derived them from the code; the
derivation is v2.4's set. `require_site_set` lists them from the code each tool runs and STOPs on a different set:

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


# ------------------------------------------------------------------ AM-24 v2.4: the ten sites, from the code, or STOP
# AM-24 v2.4 (approved 4 Nov): B1's ten sites are the eight squeeze-excitation blocks' global average pools, the LR-ASPP
# head's context pool (head.context.0) and the head's one quantized bilinear interpolate (LRASPPHead.forward), as
# src/models/student.py builds them at dc40dd2. The tools list them by module from the code they run and STOP when the
# code gives a different set (`require_site_set`). src/models/student.py's git blob at dc40dd2 is the one at fe93a24.
EXPECTED_POOLS = tuple(f"features.{i}.block.2.avgpool" for i in SE_BLOCKS) + ("head.context.0",)
EXPECTED_RESIZE = "head: one bilinear F.interpolate in the head's forward (LRASPPHead.forward), read by head.high_logits"
STUDENT_REL = "src/models/student.py"
STUDENT_BLOB_DC40DD2 = "40a84c45a35fd7af8f39ca7e111b1a3d9ff9e2ef"     # git rev-parse dc40dd2:src/models/student.py
POOL_TYPES = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AvgPool1d, nn.AvgPool2d,
              nn.AvgPool3d)
RESAMPLE_MODULE_TYPES = (nn.Upsample, nn.UpsamplingBilinear2d, nn.UpsamplingNearest2d)
RESAMPLE_CALLS = ("interpolate", "upsample", "upsample_bilinear", "upsample_nearest", "grid_sample", "avg_pool1d",
                  "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d")
PASS_THROUGH_TYPES = (nn.Identity, nn.Dropout)
# Ops that keep their input's grid EXACTLY (no rounding): never a site, whatever their qparams.
EXACT_TYPES = (nn.Identity, nn.Dropout, nn.ReLU, nn.ReLU6, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
               nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, nn.Flatten)


def git_blob_sha1(path) -> str:
    """The git blob id of a file (`git rev-parse <rev>:<path>` form), computed from its bytes."""
    import hashlib
    from pathlib import Path
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _calls_in_source(cls) -> list[dict]:
    """Resampling and pooling CALLS in one class's own source (functional ops a module walk cannot see)."""
    import ast
    import inspect
    import textwrap
    try:
        src = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return [{"call": "<source unavailable>", "mode": None, "line": None}]
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name in RESAMPLE_CALLS:
                mode = next((kw.value.value for kw in node.keywords
                             if kw.arg == "mode" and isinstance(kw.value, ast.Constant)), None)
                out.append({"call": name, "mode": mode, "line": node.lineno})
    return out


def functional_census(model: nn.Module) -> dict:
    """{class: [calls]} for every class in the model's modules' MROs that the repository or torchvision defines."""
    seen: dict[str, list] = {}
    for m in model.modules():
        for cls in type(m).__mro__:
            mod = getattr(cls, "__module__", "") or ""
            if not mod.startswith(("src.", "torchvision.")):
                continue
            key = f"{mod}.{cls.__qualname__}"
            if key not in seen:
                seen[key] = _calls_in_source(cls)
    return {k: v for k, v in seen.items() if v}


def _off_grid_fraction(x: torch.Tensor, scale: float, zp: int) -> float:
    q = x.detach().float() / scale + zp
    return float(((q - q.round()).abs() > 1e-3).float().mean())


@torch.no_grad()
def runtime_census(prepared: nn.Module, example: torch.Tensor) -> dict:
    """The modules of an eager QNNPACK conversion of `prepared` (data-free, as block B converts) whose quantized output
    carries its quantized input's scale and zero point, and whether the fake-quant graph's output of the same module
    lies off that grid (a rounding site) or on it (an exact op); and the head's resize: the input of the module that
    reads it carries head.ff_mul's grid in the converted model and lies off it in the fake-quant graph."""
    import copy
    from src.quant.prepare import convert_model
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    was_training = prepared.training
    pm = copy.deepcopy(prepared).eval()
    conv = convert_model(pm)
    cmods, pmods = dict(conv.named_modules()), dict(pm.named_modules())
    reader = "head.dequant_high" if "head.dequant_high" in cmods else "head.high_logits"
    seen_c, seen_p, hooks = {}, {}, []

    def recorder(store: dict, name: str):
        def hook(mod, i, o):                     # returns None: a hook's return value would replace the output
            if name not in store:
                store[name] = (i[0] if i else None, o)
        return hook
    for n, m in cmods.items():
        if list(m.children()) or isinstance(m, PASS_THROUGH_TYPES):
            continue
        hooks.append(m.register_forward_hook(recorder(seen_c, n)))
    for n, m in pmods.items():
        if not list(m.children()):
            hooks.append(m.register_forward_hook(recorder(seen_p, n)))
    hooks.append(cmods[reader].register_forward_pre_hook(lambda mod, i: seen_c.__setitem__("<resize>", (i[0], None))))
    hooks.append(pmods[reader].register_forward_pre_hook(lambda mod, i: seen_p.__setitem__("<resize>", (i[0], None))))
    conv(example)
    pm(example)
    for h in hooks:
        h.remove()
    prepared.train(was_training)
    keep = []
    for n, (i, o) in seen_c.items():
        if n == "<resize>" or not (torch.is_tensor(i) and torch.is_tensor(o) and i.is_quantized and o.is_quantized):
            continue
        if i.qscheme() != torch.per_tensor_affine or o.qscheme() != torch.per_tensor_affine:
            continue
        if i.q_scale() != o.q_scale() or i.q_zero_point() != o.q_zero_point():
            continue
        po = seen_p.get(n, (None, None))[1]
        frac = None if not torch.is_tensor(po) else _off_grid_fraction(po, i.q_scale(), i.q_zero_point())
        keep.append({"module": n, "type": type(cmods[n]).__name__, "scale": i.q_scale(), "zero_point": i.q_zero_point(),
                     "observed_in_prepared": hasattr(pmods.get(n), "activation_post_process"),
                     "exact_type": isinstance(cmods[n], EXACT_TYPES),
                     "fake_quant_off_grid_fraction": frac})
    # Structural, not data-dependent: the converted module emits on its input's grid, the prepared module has no
    # output observer of its own (so the training graph leaves its float result unrounded), and the op is not one that
    # keeps the grid exactly. The off-grid fraction on the census input is recorded for information only.
    rounding = sorted(k["module"] for k in keep if not k["observed_in_prepared"] and not k["exact_type"])
    ff = conv.head.ff_mul
    ci = seen_c.get("<resize>", (None, None))[0]
    pi = seen_p.get("<resize>", (None, None))[0]
    resize = {"reader": reader, "converted_input_quantized": bool(torch.is_tensor(ci) and ci.is_quantized)}
    if resize["converted_input_quantized"]:
        resize.update({"input_scale": ci.q_scale(), "input_zero_point": ci.q_zero_point(),
                       "ff_mul_scale": float(ff.scale), "ff_mul_zero_point": int(ff.zero_point),
                       "keeps_ff_mul_grid": ci.q_scale() == float(ff.scale) and ci.q_zero_point() == int(ff.zero_point),
                       "fake_quant_off_grid_fraction": (None if not torch.is_tensor(pi) else
                                                        _off_grid_fraction(pi, float(ff.scale), int(ff.zero_point)))})
    return {"grid_keeping_modules": keep, "rounding_modules": rounding, "resize": resize}


def site_census(prepared: nn.Module, *, example: torch.Tensor | None = None) -> dict:
    """The ten sites as the code gives them: the average-pool modules, the functional resamples by class, the runtime
    census and the student file's git blob; `matches` says whether it is AM-24 v2.4's set, with every difference."""
    from pathlib import Path
    mods = dict(prepared.named_modules())
    pools = sorted(n for n, m in mods.items() if isinstance(m, POOL_TYPES))
    resample_modules = sorted(n for n, m in mods.items() if isinstance(m, RESAMPLE_MODULE_TYPES))
    calls = functional_census(prepared)
    head_cls = type(prepared.head)
    head_key = f"{head_cls.__module__}.{head_cls.__qualname__}"
    root_cls = type(prepared)
    root_key = f"{root_cls.__module__}.{root_cls.__qualname__}"
    if example is None:
        example = torch.randn(1, 3, 128, 128, generator=torch.Generator().manual_seed(0))
    rt = runtime_census(prepared, example)
    repo = Path(__file__).resolve().parents[2]
    blob = git_blob_sha1(repo / STUDENT_REL)
    problems = []
    if tuple(sorted(EXPECTED_POOLS)) != tuple(pools):
        problems.append(f"average-pool modules {pools} != {sorted(EXPECTED_POOLS)}")
    if tuple(sorted(EXPECTED_POOLS)) != tuple(rt["rounding_modules"]):
        problems.append(f"converted modules that round onto their input's grid {rt['rounding_modules']} != "
                        f"{sorted(EXPECTED_POOLS)}")
    if resample_modules:
        problems.append(f"resampling modules {resample_modules}")
    head_calls = _calls_in_source(head_cls)                # wherever the head class is defined
    if [(c["call"], c["mode"]) for c in head_calls] != [("interpolate", "bilinear")]:
        problems.append(f"the head's forward ({head_key}) calls {head_calls}, not one bilinear interpolate")
    root_calls = _calls_in_source(root_cls)
    if [(c["call"], c["mode"]) for c in root_calls] != [("interpolate", "bilinear")]:
        problems.append(f"the student's forward ({root_key}) calls {root_calls}, not one (the final float upsample)")
    others = {k: v for k, v in calls.items() if k not in (head_key, root_key)}
    if others:
        problems.append(f"other classes call resampling or pooling functions: {others}")
    r = rt["resize"]
    if not (r.get("converted_input_quantized") and r.get("keeps_ff_mul_grid")):
        problems.append(f"the head's resize does not emit on head.ff_mul's grid in the converted model: {r}")
    listed = [{"site": n, "module": EXPECTED_POOLS[k] if k < len(EXPECTED_POOLS) else EXPECTED_RESIZE}
              for k, n in enumerate(SITE_NAMES)]
    return {"rule": "AM-24 v2.4: the eight SE pools, head.context.0 and the head's one bilinear interpolate, as "
                    "src/models/student.py builds them at dc40dd2; a different set STOPs",
            "sites_by_module": listed, "average_pool_modules": pools, "resample_modules": resample_modules,
            "functional_calls": calls, "runtime": rt, "student_py_blob": blob,
            "student_py_blob_dc40dd2": STUDENT_BLOB_DC40DD2, "student_py_is_dc40dd2s": blob == STUDENT_BLOB_DC40DD2,
            "problems": problems, "matches": not problems}


def require_site_set(prepared: nn.Module, *, example: torch.Tensor | None = None) -> dict:
    """STOP (SiteError site_set_mismatch) unless the code gives AM-24 v2.4's ten sites; else the census."""
    census = site_census(prepared, example=example)
    if not census["matches"]:
        raise SiteError("site_set_mismatch", "the code gives a different set of sites than AM-24 v2.4's: "
                                             + "; ".join(census["problems"]))
    return census


def calibrated_copy(model: nn.Module, *, size: int = 128, seed: int = 0) -> nn.Module:
    """A prepared (QAT) copy of an FP32 student, observers moved by one train-mode forward on random inputs, then
    frozen: the census's input when no trained epoch exists yet (N2's launch). The model itself is not touched."""
    import copy
    from src.quant.prepare import disable_observers, prepare_qat_model
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    p = prepare_qat_model(copy.deepcopy(model), select_backend=False)
    p.train()
    with torch.no_grad():
        p(torch.randn(2, 3, size, size, generator=torch.Generator().manual_seed(seed)))
    disable_observers(p)
    return p.eval()
