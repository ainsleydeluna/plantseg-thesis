#!/usr/bin/env python3
"""DL-18 — QNNPACK PTQ probe: registered qconfig, per-channel conv weights, LR-ASPP Sigmoid at runtime.

Runs the E4 structural path exactly as `src.quant.prepare` defines it — QNNPACK backend, fuse, the
registered PTQ qconfig (`src.quant.qconfig.ptq_qconfig`: histogram UINT8 activations, per-channel
symmetric INT8 weights), observer insertion, calibration, convert — on the FP32 student, then:

  * asserts every converted convolution weight is per-channel INT8 (qint8, axis 0, zero points all 0);
  * records, through a runtime forward hook on the LR-ASPP context Sigmoid (`head.context[2]`,
    src/models/student.py:82-86), whether it runs quantized or through a float fallback, with the
    tensors' dtype, scale and zero_point (U6);
  * records whether the head's bilinear interpolate runs on quantized tensors (a forward pre-hook on
    `head.high_logits`, whose input is the interpolate output);
  * records `quantization_coverage()` verbatim. That walk labels `nn.Sigmoid` an FP32 fallback by
    module type alone (src/quant/prepare.py:188-190); the runtime hook is the evidence for U6.

Calibration uses SYNTHETIC inputs only: this is not E4 and not the AM-10 calibration list (lane
L-AM10). CPU only (QNNPACK is a CPU engine). Nothing is written inside the repository.

Exit status (DL-18) — non-zero ONLY on:
  1  a converted conv weight that is not per-channel INT8
  2  a quantized engine other than qnnpack
  3  a conversion error: prepare, calibration, convert or the converted forward fails, the output
     shape is wrong, or an FP32 nn.Conv2d survives conversion
  4  a run that does not complete (bad arguments, checkpoint refused, any other exception)
A float fallback of the Sigmoid is NOT a failure: it resolves U6 and is recorded for the E4-E7
quantization coverage and Ch4.

Example (pinned student image, CPU, no network):
    python -B scripts/probe_qnnpack_ptq.py --e1-checkpoint /ckpt/e1_student_best_iter80000.pt \\
        --expect-sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03 \\
        --out /out/dl18_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.ao.nn.quantized.modules.conv import _ConvNd as QuantizedConvNd  # noqa: E402

from src.quant import prepare as P  # noqa: E402
from src.quant import qconfig as Q  # noqa: E402

EXIT_OK, EXIT_NOT_PER_CHANNEL, EXIT_ENGINE, EXIT_CONVERSION, EXIT_INCOMPLETE = 0, 1, 2, 3, 4
N_CALIBRATION = 8
INPUT_SHAPE = (1, 3, 512, 512)              # one image per mini-batch, as AM-10 fixes for E4
OUTPUT_SHAPE = (1, 116, 512, 512)
SIGMOID_PATH = "head.context.2"             # AdaptiveAvgPool2d -> Conv2d -> Sigmoid
INTERPOLATE_PROBE_PATH = "head.high_logits"


class ProbeExit(Exception):
    """Stop the probe with a DL-18 exit status."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _tensor_info(t) -> dict:
    if not isinstance(t, torch.Tensor):
        return {"type": type(t).__name__}
    info = {"is_quantized": bool(t.is_quantized), "dtype": str(t.dtype), "shape": list(t.shape)}
    if t.is_quantized:
        qs = t.qscheme()
        info["qscheme"] = str(qs)
        if qs in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            info["scale"] = float(t.q_scale())
            info["zero_point"] = int(t.q_zero_point())
    return info


def _require_qnnpack(stage: str) -> str:
    engine = torch.backends.quantized.engine
    if engine != Q.QUANT_BACKEND:
        raise ProbeExit(EXIT_ENGINE, f"quantized engine is {engine!r} at {stage}; DL-18 requires "
                                     f"{Q.QUANT_BACKEND!r}")
    return engine


def conv_weight_report(converted: nn.Module) -> tuple[list[dict], list[str], list[str]]:
    """Every converted conv's weight quantization, the per-channel violations, and any FP32 conv."""
    rows, violations, float_convs = [], [], []
    for name, m in converted.named_modules():
        if isinstance(m, nn.Conv2d):
            float_convs.append(name)
            continue
        if not isinstance(m, QuantizedConvNd):
            continue
        w = m.weight()
        qs = w.qscheme()
        row = {"module": name, "type": f"{type(m).__module__}.{type(m).__name__}",
               "qscheme": str(qs), "dtype": str(w.dtype), "out_channels": int(w.shape[0]),
               "groups": int(m.groups)}
        if qs in (torch.per_channel_affine, torch.per_channel_symmetric):
            scales, zero_points = w.q_per_channel_scales(), w.q_per_channel_zero_points()
            row.update(axis=int(w.q_per_channel_axis()), n_scales=int(scales.numel()),
                       zero_points_all_zero=bool((zero_points == 0).all()),
                       scale_min=float(scales.min()), scale_max=float(scales.max()))
        row["per_channel_int8"] = bool(
            qs in (torch.per_channel_affine, torch.per_channel_symmetric) and w.dtype == torch.qint8
            and row.get("axis") == 0 and row.get("zero_points_all_zero") is True
            and row.get("n_scales") == int(w.shape[0]))
        if not row["per_channel_int8"]:
            violations.append(name)
        rows.append(row)
    return rows, violations, float_convs


def run(args, report: dict) -> int:
    from src.models.student import build_student
    from src.quant.checkpoint import load_student_from_e1
    from src.seeds import set_seed

    report["supported_engines"] = list(torch.backends.quantized.supported_engines)
    try:
        Q.select_qnnpack_backend()
    except Q.QuantBackendUnavailable as e:
        raise ProbeExit(EXIT_ENGINE, str(e)) from e
    report["engine"] = _require_qnnpack("backend selection")
    set_seed(args.seed)

    if args.random_init:
        model = build_student(pretrained=False)
        report["source"] = {"kind": "random_init", "seed": args.seed}
    else:
        path = Path(args.e1_checkpoint)
        if not path.is_file():
            raise ProbeExit(EXIT_INCOMPLETE, f"E1 checkpoint not found: {path}")
        sha = _sha256(path)
        if args.expect_sha256 and sha != args.expect_sha256:
            raise ProbeExit(EXIT_INCOMPLETE, f"E1 checkpoint sha256 {sha} != expected {args.expect_sha256}")
        model, ckpt = load_student_from_e1(path)
        report["source"] = {"kind": "E1", "path": str(path), "sha256": sha, "iter": ckpt.get("iter"),
                            "best_val_miou_all_class": ckpt.get("best_val_miou_all_class")}
    model.eval()
    report["qconfig"] = Q.describe_qconfig(Q.ptq_qconfig())

    stage = "prepare"
    try:
        prepared = P.prepare_ptq(model)           # fuse -> registered PTQ qconfig -> observers
        report["fusion"] = prepared.fusion_report() if hasattr(prepared, "fusion_report") else None
        stage = "calibrate"
        g = torch.Generator().manual_seed(args.seed)
        report["calibration"] = {
            "kind": "synthetic torch.randn, NOT the AM-10 TRAIN list", "batches": N_CALIBRATION,
            "shape": list(INPUT_SHAPE), "seen": P.calibrate(
                prepared, [torch.randn(*INPUT_SHAPE, generator=g) for _ in range(N_CALIBRATION)])}
        stage = "convert"
        converted = P.convert_model(prepared)
    except ProbeExit:
        raise
    except Exception as e:  # noqa: BLE001 -- a conversion error is a DL-18 failure, reported verbatim
        raise ProbeExit(EXIT_CONVERSION, f"{stage}: {type(e).__name__}: {e}") from e
    _require_qnnpack("after convert")

    rows, violations, float_convs = conv_weight_report(converted)
    report["conv_weights"] = {"converted_conv_modules": len(rows),
                              "per_channel_int8": sum(r["per_channel_int8"] for r in rows),
                              "violations": violations, "fp32_conv_modules": float_convs, "modules": rows}
    if float_convs:
        raise ProbeExit(EXIT_CONVERSION, f"FP32 nn.Conv2d survived conversion: {float_convs}")

    modules = dict(converted.named_modules())
    if SIGMOID_PATH not in modules or INTERPOLATE_PROBE_PATH not in modules:
        raise ProbeExit(EXIT_INCOMPLETE, f"{SIGMOID_PATH} or {INTERPOLATE_PROBE_PATH} not found")
    sigmoid_module = modules[SIGMOID_PATH]
    seen: dict[str, list] = {"sigmoid": [], "interpolate": []}
    hooks = [
        sigmoid_module.register_forward_hook(
            lambda m, i, o: seen["sigmoid"].append({"input": _tensor_info(i[0]), "output": _tensor_info(o)})),
        modules[INTERPOLATE_PROBE_PATH].register_forward_pre_hook(
            lambda m, i: seen["interpolate"].append({"input": _tensor_info(i[0])})),
    ]
    g = torch.Generator().manual_seed(args.seed + 1)
    forward = P.try_converted_forward(converted, torch.randn(*INPUT_SHAPE, generator=g))
    for h in hooks:
        h.remove()
    report["forward"] = forward
    if not forward["ok"]:
        raise ProbeExit(EXIT_CONVERSION, f"converted forward failed: {forward['error']}")
    if tuple(forward["output_shape"]) != OUTPUT_SHAPE:
        raise ProbeExit(EXIT_CONVERSION, f"converted output shape {forward['output_shape']} != {OUTPUT_SHAPE}")
    if not seen["sigmoid"] or not seen["interpolate"]:
        raise ProbeExit(EXIT_INCOMPLETE, "a runtime hook did not fire during the converted forward")
    _require_qnnpack("after converted forward")

    s_in, s_out = seen["sigmoid"][0]["input"], seen["sigmoid"][0]["output"]
    if s_in.get("is_quantized") and s_out.get("is_quantized"):
        verdict = "RUNS_QUANTIZED"
    elif not s_in.get("is_quantized") and not s_out.get("is_quantized"):
        verdict = "FLOAT_FALLBACK"
    else:
        verdict = "MIXED"
    report["sigmoid"] = {
        "module": SIGMOID_PATH, "module_type": f"{type(sigmoid_module).__module__}.{type(sigmoid_module).__name__}",
        "verdict": verdict, "input": s_in, "output": s_out,
        "u6_values": ({"scale": s_out.get("scale"), "zero_point": s_out.get("zero_point")}
                      if verdict == "RUNS_QUANTIZED" else None)}
    report["interpolate"] = {"probe": f"forward pre-hook on {INTERPOLATE_PROBE_PATH}",
                             "head_os8_interpolate_quantized": bool(seen["interpolate"][0]["input"].get("is_quantized")),
                             "input": seen["interpolate"][0]["input"],
                             "final_upsample": "float after dequant by design (src/models/student.py:148-149)"}
    report["coverage"] = P.quantization_coverage(converted)
    report["coverage_note"] = ("quantization_coverage classifies nn.Sigmoid by module type only "
                               "(src/quant/prepare.py:188-190); the runtime hook above is the U6 evidence")
    return EXIT_NOT_PER_CHANNEL if violations else EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DL-18 QNNPACK PTQ probe (registered qconfig).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--e1-checkpoint", help="FP32 E1 checkpoint (loaded with src.quant.checkpoint)")
    src.add_argument("--random-init", action="store_true", help="random FP32 student instead of E1")
    ap.add_argument("--expect-sha256", default=None, help="required SHA-256 of --e1-checkpoint")
    ap.add_argument("--out", required=True, help="JSON report path, OUTSIDE the repository")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    report = {"probe": "DL-18", "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "torch": torch.__version__, "probe_sha256": _sha256(Path(__file__)),
              "argv": list(sys.argv[1:] if argv is None else argv)}
    if out == REPO or REPO in out.parents:
        print(f"[DL18_RESULT] exit={EXIT_INCOMPLETE} refusing to write inside the repository: {out}")
        return EXIT_INCOMPLETE
    try:
        import torchvision
        report["torchvision"] = torchvision.__version__
        code = run(args, report)
        report["message"] = ("every converted conv weight is per-channel INT8" if code == EXIT_OK
                             else f"conv weights not per-channel INT8: {report['conv_weights']['violations']}")
    except ProbeExit as e:
        code = e.code
        report["message"] = str(e)
    except Exception as e:  # noqa: BLE001 -- an incomplete run is DL-18 exit 4, reported verbatim
        code = EXIT_INCOMPLETE
        report["message"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
    report["exit_code"] = code
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    conv = report.get("conv_weights", {})
    sig = report.get("sigmoid", {})
    print(f"[DL18] engine={report.get('engine')} supported={report.get('supported_engines')}")
    print(f"[DL18] source={report.get('source', {}).get('kind')} "
          f"sha256={report.get('source', {}).get('sha256')}")
    print(f"[DL18] converted conv modules={conv.get('converted_conv_modules')} "
          f"per_channel_int8={conv.get('per_channel_int8')} violations={len(conv.get('violations', []))} "
          f"fp32_conv={len(conv.get('fp32_conv_modules', []))}")
    if sig:
        print(f"[DL18] sigmoid({SIGMOID_PATH}) verdict={sig['verdict']} input={sig['input'].get('dtype')} "
              f"output={sig['output'].get('dtype')} u6={sig['u6_values']}")
    if "interpolate" in report:
        print(f"[DL18] head OS8 interpolate quantized={report['interpolate']['head_os8_interpolate_quantized']}")
    print(f"[DL18_RESULT] exit={code} {report.get('message', '')} report={out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
