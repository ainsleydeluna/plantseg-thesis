#!/usr/bin/env python3
"""Build the demo's STAND-IN model: the thesis student with RANDOM weights, quantized through the
repository's own QNNPACK eager path and serialized the way the artifact of record is.

    python3 -I demo/model/build_stand_in.py [--out PATH] [--force]

  * model     src/models/student.py build_student(pretrained=False): random init, no download
  * quantize  src/quant/ptq.py quantize_qnnpack: QNNPACK engine -> fuse -> the registered PTQ qconfig
              (src/quant/qconfig.py) -> calibrate one image per batch (src/quant/prepare.py) -> convert
  * calibrate synthetic self-made images (demo/synthetic.py) through the evaluator's preprocessing;
              no dataset exists here
  * serialize src/quant/ptq.py torchscript_bytes: torch.jit.trace at 1x3x512x512 + plantseg_int8.json
  * verify    load back through src/eval/model_loading.py load_int8_torchscript (the evaluator's loader),
              src/quant/ptq.py graph_census and output_parity (converted module vs TorchScript)

Writes <out> (default demo/models/stand_in_int8_qnnpack.pt), <out stem>_fp32_state.pt (the random FP32
weights, for the ONNX exports) and <out stem>.facts.json. Refuses to overwrite without --force.
The stand-in's output is NOT a prediction: its weights are random.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402

STAND_IN_SEED = 20261103          # torch seed for the random init
CALIBRATION_SEED = 1103           # synthetic calibration images
CALIBRATION_IMAGES = 8
STAND_IN_STAGE = "DEMO_STAND_IN"
STAND_IN_ROLE = "demo_stand_in"


def build(out: Path, force: bool = False) -> dict:
    import torch

    from demo.server.preprocess import PROTOCOL_ID, evaluator_preprocess
    from demo.synthetic import calibration_images
    from src.eval.model_loading import (INT8_TORCHSCRIPT_SCHEMA, load_int8_torchscript,
                                        select_int8_backend)
    from src.models.student import build_student
    from src.quant.ptq import (graph_census, output_parity, quantize_qnnpack, student_traced_in_process,
                               synthetic_parity_inputs, torchscript_bytes)

    fp32_out = out.with_name(out.stem + "_fp32_state.pt")
    facts_out = out.with_name(out.stem + ".facts.json")
    for p in (out, fp32_out, facts_out):
        if p.exists() and not force:
            raise SystemExit(f"refusing to overwrite {p} (pass --force)")
    if student_traced_in_process():
        raise SystemExit("the student was already traced in this process; run in a fresh process")

    t0 = time.perf_counter()
    torch.manual_seed(STAND_IN_SEED)
    model = build_student(pretrained=False).eval()
    if model.used_pretrained:
        raise SystemExit("build_student loaded pretrained weights; the stand-in must be random")
    fp32_state = {k: v.clone() for k, v in model.state_dict().items()}

    images = [evaluator_preprocess(im).tensor for im in calibration_images(CALIBRATION_IMAGES,
                                                                           CALIBRATION_SEED)]
    converted, n_cal = quantize_qnnpack(model, images)
    meta = {
        "schema": INT8_TORCHSCRIPT_SCHEMA, "stage": STAND_IN_STAGE, "quantization": "ptq",
        "num_classes": int(model.num_classes), "engine": "qnnpack", "artifact_role": STAND_IN_ROLE,
        "stand_in": True,
        "warning": "random weights: the output is not a prediction",
        "weights": f"build_student(pretrained=False) after torch.manual_seed({STAND_IN_SEED})",
        "calibration": (f"{n_cal} synthetic images (demo/synthetic.py, seed {CALIBRATION_SEED}) "
                        f"through {PROTOCOL_ID}; no dataset"),
        "builder": "demo/model/build_stand_in.py",
        "torch": torch.__version__,
    }
    blob, trace_notes = torchscript_bytes(converted, meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    buf = io.BytesIO()
    torch.save(fp32_state, buf)
    fp32_out.write_bytes(buf.getvalue())
    t_build = time.perf_counter() - t0

    # ---- verify through the evaluator's own loader and the PTQ lane's own checks ----
    select_int8_backend(require_qnnpack=True)
    ts, meta_back = load_int8_torchscript(out, stage=STAND_IN_STAGE, method="ptq",
                                          artifact_role=STAND_IN_ROLE)
    census = graph_census(ts)
    opnames = sorted(torch.jit.export_opnames(ts))
    parity = output_parity({"converted": converted, "torchscript": ts}, synthetic_parity_inputs(),
                           [("converted", "torchscript")])
    with torch.no_grad():
        y = ts(torch.zeros(1, 3, 512, 512))
    common.assert_no_stats()
    facts = {
        "file": out.name, "size_bytes": out.stat().st_size, "sha256": common.sha256_file(out),
        "fp32_state_file": fp32_out.name, "fp32_state_sha256": common.sha256_file(fp32_out),
        "embedded_meta": meta_back,
        "output_shape_for_1x3x512x512": list(y.shape), "output_dtype": str(y.dtype),
        "calibration_images": n_cal,
        "graph_census": census,
        "quantized_ops": sorted(op for op in opnames if op.startswith("quantized::")),
        "export_opnames": opnames,
        "parity_converted_vs_torchscript": parity,
        "trace_warnings": trace_notes,
        "build_seconds": round(t_build, 2),
        "torch": torch.__version__, "python": platform.python_version(),
        "machine": platform.machine(), "engine": torch.backends.quantized.engine,
        "src_stats_imported": common.stats_modules_loaded(),
    }
    facts_out.write_text(json.dumps(facts, indent=2, sort_keys=True), encoding="utf-8")
    return facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(config.STAND_IN_PATH))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    facts = build(config.resolve_model_path(a.out), force=a.force)
    print(json.dumps({k: facts[k] for k in ("file", "size_bytes", "sha256", "output_shape_for_1x3x512x512",
                                             "quantized_ops", "build_seconds")}, indent=2))
    print(f"graph_census ok={facts['graph_census']['ok']}  parity all_equal="
          f"{facts['parity_converted_vs_torchscript']['all_equal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
