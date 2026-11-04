#!/usr/bin/env python3
"""Export the student to ONNX for in-browser inference (ONNX Runtime Web) and measure parity.

    python3 -I demo/ondevice/export_onnx.py [--model INT8.pt] [--fp32-state STATE.pt] [--no-fp32]

Every exported graph takes float32 [1, 3, 512, 512] (the canvas of record; static, which the WebGPU
execution provider prefers) and returns "logits" [1, 116, 512, 512] and "mask" uint8 [1, 512, 512] (an
in-graph argmax, so a browser fetches 0.26 MB instead of 121 MB of logits).

  FP32  build_student(pretrained=False) + the FP32 weights (default: the stand-in's random weights,
        demo/models/stand_in_int8_qnnpack_fp32_state.pt; for the artifact of record, the FP32 source
        checkpoint's weights) -> torch.onnx.export, opset 17
  INT8  the INT8 TorchScript file itself (default: demo/config.py MODEL_PATH) -> torch.onnx.export,
        opset 17; the quantized ops become QuantizeLinear/DequantizeLinear (QDQ). If that export
        fails, the converted eager module (rebuilt by the repository's PTQ path from the FP32 weights,
        as the stand-in was built) is tried instead, and the facts say which one produced the file.

Parity: onnxruntime (CPU) against the PyTorch module the graph came from, on random inputs and on
synthetic images through the evaluator's preprocessing: max/mean |logit difference|, the difference in
output quantization steps (INT8), and argmax agreement. Writes demo/models/student_fp32.onnx,
student_int8_qdq.onnx and onnx_export_facts.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402

OPSET = 17
SHAPE = (1, 3, 512, 512)


def parity_inputs(n_random: int = 3, n_images: int = 3):
    import torch

    from demo.server.preprocess import evaluator_preprocess
    from demo.synthetic import calibration_images

    g = torch.Generator().manual_seed(42)
    xs = [("random_normal_%d" % i, torch.randn(*SHAPE, generator=g)) for i in range(n_random)]
    for i, im in enumerate(calibration_images(n_images, 5000)):
        xs.append((f"synthetic_image_{i}_{im.size[0]}x{im.size[1]}", evaluator_preprocess(im).tensor))
    return xs


class WithMask:
    """Wraps a module so the exported graph also returns the uint8 argmax mask."""

    @staticmethod
    def make(module):
        import torch

        class _W(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                y = self.m(x)
                return y, torch.argmax(y, dim=1).to(torch.uint8)

        return _W(module).eval()


def load_fp32(path: Path):
    """FP32 weights: the repository's nested checkpoint (src/eval/model_loading.py
    load_student_checkpoint, e.g. the E1 source checkpoint), else the stand-in's bare state_dict."""
    import torch

    from src.eval.model_loading import CheckpointError, load_student_checkpoint
    from src.models.student import build_student

    try:
        model, info = load_student_checkpoint(path)
        return model.eval(), f"load_student_checkpoint({path.name}) sha256 {info.sha256}"
    except CheckpointError as e:
        if "BARE state_dict" not in str(e):
            raise
    model = build_student(pretrained=False).eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model, f"build_student(pretrained=False) + bare state_dict {path.name} sha256 {common.sha256_file(path)}"


def export(module, path: Path) -> dict:
    import torch

    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(WithMask.make(module), torch.zeros(SHAPE), str(path), opset_version=OPSET,
                          input_names=["input"], output_names=["logits", "mask"],
                          do_constant_folding=True)
    return {"seconds": round(time.perf_counter() - t0, 2), "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path)}


def drop_int32_zero_points(path: Path) -> int:
    """Remove the all-zero zero-point input of every int32 DequantizeLinear (the quantized biases).

    For int32 the ONNX operator treats zero_point as 0 when it is absent, so this does not change the
    graph's outputs; ONNX Runtime Web 1.20.1's WebGPU kernel refuses int32 dequantization with a zero
    point (measured), while the WASM/CPU kernels accept both forms. Returns the number of nodes changed.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    from onnx.helper import tensor_dtype_to_np_dtype

    m = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    memo: dict = {}

    def value(name):
        """The constant value of `name`, through Constant, ConstantOfShape and Cast; else None."""
        if name in memo:
            return memo[name]
        v, n = inits.get(name), prod.get(name)
        if v is None and n is not None:
            attrs = {a.name: a for a in n.attribute}
            if n.op_type == "Constant" and "value" in attrs:
                v = numpy_helper.to_array(attrs["value"].t)
            elif n.op_type == "ConstantOfShape":
                shape = value(n.input[0])
                fill = (numpy_helper.to_array(attrs["value"].t) if "value" in attrs
                        else np.zeros(1, np.float32))
                v = None if shape is None else np.full(tuple(int(s) for s in shape), fill.ravel()[0], fill.dtype)
            elif n.op_type == "Cast":
                src = value(n.input[0])
                v = None if src is None else src.astype(tensor_dtype_to_np_dtype(attrs["to"].i))
        memo[name] = v
        return v

    changed = 0
    for n in m.graph.node:
        if n.op_type != "DequantizeLinear" or len(n.input) < 3:
            continue
        x, zp = value(n.input[0]), value(n.input[2])
        if x is None or zp is None or x.dtype != np.int32 or np.any(zp != 0):
            continue
        del n.input[2]
        changed += 1
    if changed:                                    # drop the now-unused zero-point producers
        while True:
            used = {i for n in m.graph.node for i in n.input} | {o.name for o in m.graph.output}
            keep_nodes = [n for n in m.graph.node if any(o in used for o in n.output)]
            if len(keep_nodes) == len(m.graph.node):
                break
            del m.graph.node[:]
            m.graph.node.extend(keep_nodes)
        keep = [i for i in m.graph.initializer if i.name in used]
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep)
        onnx.checker.check_model(m)
        onnx.save(m, str(path))
    return changed


def onnx_summary(path: Path) -> dict:
    import collections

    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    return {"opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import],
            "nodes": len(m.graph.node), "op_types": dict(sorted(ops.items())),
            "inputs": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
            "outputs": [o.name for o in m.graph.output], "checker": "ok"}


def output_quant_step(ts) -> float | None:
    """Scale of the quantized tensor the graph dequantizes (the head's OS8 logits), from the graph."""
    for n in ts.inlined_graph.nodes():
        if n.kind() == "aten::dequantize":
            src = n.inputsAt(0).node()
            if src.kind().startswith("quantized::add"):
                return float(src.inputsAt(2).toIValue())
    return None


def parity(torch_fn, onnx_path: Path, inputs, quant_step: float | None = None) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch

    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    rows, times = [], []
    for name, x in inputs:
        with torch.no_grad():
            ref = torch_fn(x)
        t0 = time.perf_counter()
        logits, mask = sess.run(["logits", "mask"], {"input": x.numpy()})
        times.append((time.perf_counter() - t0) * 1000)
        ref_np = ref.numpy()
        d = np.abs(logits - ref_np)
        ref_mask = ref.argmax(1).to(torch.uint8).numpy()
        row = {"input": name, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
               "ref_logit_range": [float(ref_np.min()), float(ref_np.max())],
               "argmax_agreement": float((mask == ref_mask).mean()),
               "mask_equals_argmax_of_its_logits": bool((mask == logits.argmax(1)).all()),
               "ref_distinct_argmax_classes": int(len(np.unique(ref_mask)))}
        if quant_step:
            row["max_abs_diff_in_output_quant_steps"] = float(d.max() / quant_step)
        rows.append(row)
    return {"rows": rows, "max_abs_diff": max(r["max_abs_diff"] for r in rows),
            "min_argmax_agreement": min(r["argmax_agreement"] for r in rows),
            "ort_cpu_ms_per_image": {"median": float(np.median(times)), "all": [round(t, 1) for t in times]},
            "onnxruntime": ort.__version__}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ONNX exports for the in-browser demo")
    ap.add_argument("--model", default=None, help="INT8 TorchScript (default: demo/config.py MODEL_PATH)")
    ap.add_argument("--fp32-state", default=None,
                    help="FP32 weights for the FP32 export: the repository's nested FP32 checkpoint "
                         "(e.g. the E1 source checkpoint) or, by default, the stand-in's random weights")
    ap.add_argument("--no-fp32", action="store_true")
    a = ap.parse_args(argv)

    import torch

    from src.eval.model_loading import select_int8_backend

    torch.manual_seed(0)
    int8_path = config.resolve_model_path(a.model)
    fp32_state = Path(a.fp32_state) if a.fp32_state else int8_path.with_name(int8_path.stem + "_fp32_state.pt")
    inputs = parity_inputs()
    facts = {"torch": torch.__version__, "python": platform.python_version(), "opset": OPSET,
             "input_shape": list(SHAPE), "parity_inputs": [n for n, _ in inputs]}

    fp32 = None
    if not a.no_fp32:
        if not fp32_state.is_file():
            facts["fp32"] = {"ok": False, "error": f"no FP32 state at {fp32_state}"}
        else:
            fp32, fp32_source = load_fp32(fp32_state)
            try:
                info = export(fp32, config.ONNX_FP32_PATH)
                info.update(onnx_summary(config.ONNX_FP32_PATH))
                info["parity_vs_pytorch_fp32"] = parity(fp32, config.ONNX_FP32_PATH, inputs)
                facts["fp32"] = {"ok": True, "file": config.ONNX_FP32_PATH.name,
                                 "source": fp32_source, **info}
            except Exception as e:  # noqa: BLE001 -- recorded verbatim
                facts["fp32"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()[-2000:]}

    # ---------------------------------------------------------------- INT8
    select_int8_backend(require_qnnpack=True)
    from demo.server.inference import load_model

    loaded = load_model(int8_path)
    ts = loaded.module
    try:
        step = output_quant_step(ts)                       # the OS8 logits' quantization step
    except Exception:  # noqa: BLE001 -- descriptive only
        step = None
    attempts = []
    for label, make in (("torchscript_file", lambda: ts), ("converted_eager_module", None)):
        if make is None:
            if fp32 is None:
                attempts.append({"source": label, "ok": False, "error": "no FP32 weights to rebuild from"})
                continue
            from demo.model.build_stand_in import CALIBRATION_IMAGES, CALIBRATION_SEED
            from demo.server.preprocess import evaluator_preprocess
            from demo.synthetic import calibration_images
            from src.quant.ptq import quantize_qnnpack

            imgs = [evaluator_preprocess(im).tensor
                    for im in calibration_images(CALIBRATION_IMAGES, CALIBRATION_SEED)]
            module, _ = quantize_qnnpack(fp32, imgs)
        else:
            module = make()
        try:
            info = export(module, config.ONNX_INT8_PATH)
            info["int32_zero_points_dropped"] = drop_int32_zero_points(config.ONNX_INT8_PATH)
            info["size_bytes"] = config.ONNX_INT8_PATH.stat().st_size
            info["sha256"] = common.sha256_file(config.ONNX_INT8_PATH)
            info.update(onnx_summary(config.ONNX_INT8_PATH))
            info["parity_vs_int8_torchscript"] = parity(ts, config.ONNX_INT8_PATH, inputs, step)
            attempts.append({"source": label, "ok": True, **info})
            break
        except Exception as e:  # noqa: BLE001 -- recorded verbatim
            attempts.append({"source": label, "ok": False, "error": f"{type(e).__name__}: {str(e)[:1500]}",
                             "trace": traceback.format_exc()[-2500:]})
    facts["int8"] = {"model_file": int8_path.name, "model_sha256": loaded.sha256,
                     "output_quant_step": step, "attempts": attempts,
                     "ok": any(x.get("ok") for x in attempts)}
    if fp32 is not None and facts.get("fp32", {}).get("ok"):
        # INT8 TorchScript vs the FP32 PyTorch model (quantization error of this stand-in)
        import numpy as np
        rows = []
        for name, x in inputs:
            with torch.no_grad():
                yq, yf = ts(x), fp32(x)
            rows.append({"input": name, "max_abs_diff": float((yq - yf).abs().max()),
                         "argmax_agreement": float((yq.argmax(1) == yf.argmax(1)).float().mean())})
        facts["int8_torchscript_vs_fp32_pytorch"] = rows
    common.assert_no_stats()
    out = config.MODELS_DIR / "onnx_export_facts.json"
    out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps({"fp32_ok": facts.get("fp32", {}).get("ok"),
                      "fp32_size": facts.get("fp32", {}).get("size_bytes"),
                      "fp32_max_abs_diff": facts.get("fp32", {}).get("parity_vs_pytorch_fp32", {}).get("max_abs_diff"),
                      "int8_ok": facts["int8"]["ok"],
                      "int8_attempts": [(x["source"], x["ok"], x.get("error", "")[:300]) for x in attempts]},
                     indent=2))
    print(f"facts: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
