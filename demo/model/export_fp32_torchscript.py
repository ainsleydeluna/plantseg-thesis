#!/usr/bin/env python3
"""Export an FP32 student checkpoint as the TorchScript file the demo's e3 slot serves.

    python3 -I demo/model/export_fp32_torchscript.py CHECKPOINT --source-sha256 HEX --stage E3 --seed 42 \
        --out OUT.pt [--e6 E6_INT8.pt]

  check   the checkpoint's sha256 must equal --source-sha256 before the file is read; with --e6, the E6
          file's embedded plantseg_int8.json (read from the zip; the model is not loaded) must name this
          checkpoint as its source_checkpoint_sha256, i.e. this is the checkpoint E6 was quantized from
  build   the repository's own model code: src/quant/checkpoint.py load_student_from_e3, the E3 loader E6
          was quantized through (stage must be E3; no CWD projection, teacher or quantization state;
          strict load into src/models/student.py build_student(pretrained=False)), in eval mode
  trace   torch.jit.trace at 1x3x512x512 (src/quant/ptq.py EXAMPLE_SHAPE), with plantseg_fp32.json
          embedded: stage, seed, the source checkpoint's sha256, 116 classes and the preprocessing id
          the INT8 path names (core_preprocess/1.0.0 = src.eval.adapters.PREPROCESS_PROTOCOL)
  verify  the bytes are written to OUT.partial, loaded back through demo/server/inference.py
          load_fp32_model (identity and 116 classes checked), and compared with the eager model on fixed
          synthetic tensors (src/quant/ptq.py synthetic_parity_inputs: a batch of 2 and 384x512 among
          them) within demo/model/parity_fp32.py's limits; only then renamed to OUT
Run in a fresh process (a second trace in one process renames the traced class). Refuses an existing
OUT. Every run applies PHONE-FIX-1 (mkldnn off). It is tested here only on a random-weights stand-in
(demo/model/build_stand_in.py --e3-checkpoint); E3's own checkpoint is exported on the laptop, after AM-30.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402

STAGES = ("E3",)                 # the registry's FP32 slot is e3


def e6_source_sha256(e6_path: Path) -> tuple[str | None, dict]:
    from demo.server.inference import read_embedded_meta

    meta = read_embedded_meta(e6_path)          # the zip's JSON entry only; the module is not loaded
    return meta.get("source_checkpoint_sha256"), meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FP32 student checkpoint -> TorchScript with plantseg_fp32.json")
    ap.add_argument("checkpoint")
    ap.add_argument("--source-sha256", required=True, help="the checkpoint's expected sha256")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--seed", required=True, type=int, help="the training seed of the checkpoint (E3: 42)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--e6", default=None, help="E6's INT8 TorchScript file, to cross-check the source checkpoint")
    a = ap.parse_args(argv)
    src, out = config.resolve_model_path(a.checkpoint), config.resolve_model_path(a.out)
    partial = out.with_name(out.name + ".partial")
    for p in (out, partial):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    for p, what in ((src, "checkpoint"), (out, "output")):
        common.refuse_test_named(p, what)

    phone_fix = common.phone_fix()
    import torch

    from demo.model.parity_fp32 import compare_logits, within_limits
    from demo.server.inference import (FP32_ARTIFACT_ROLE, FP32_META_NAME, FP32_PRECISION, FP32_SCHEMA,
                                       NUM_CLASSES, load_fp32_model)
    from demo.server.preprocess import PROTOCOL_ID
    from src.eval.adapters import PREPROCESS_PROTOCOL
    from src.quant.checkpoint import load_student_from_e3
    from src.quant.ptq import EXAMPLE_SHAPE, student_traced_in_process, synthetic_parity_inputs

    if PROTOCOL_ID != PREPROCESS_PROTOCOL:
        raise SystemExit(f"the demo's preprocessing {PROTOCOL_ID} is not the evaluator's {PREPROCESS_PROTOCOL}")
    if student_traced_in_process():
        raise SystemExit("the student was already traced in this process; run in a fresh process")
    want = a.source_sha256.strip().lower()
    got = common.sha256_file(src)
    if got != want:
        raise SystemExit(f"refused: {src.name} has sha256 {got}, not --source-sha256 {want}; it was not read")
    e6_check = "not run (no --e6)"
    if a.e6:
        e6_path = config.resolve_model_path(a.e6)
        e6_src, e6_meta = e6_source_sha256(e6_path)
        if e6_src is None:
            e6_check = f"not checked: {e6_path.name}'s identity names no source_checkpoint_sha256"
        elif e6_src != got:
            raise SystemExit(f"refused: {e6_path.name} (stage {e6_meta.get('stage')}) was quantized from {e6_src}, "
                             f"not from {src.name} ({got})")
        else:
            e6_check = f"{e6_path.name} (stage {e6_meta.get('stage')}) names this checkpoint as its source"

    t0 = time.perf_counter()
    model, ckpt = load_student_from_e3(src)
    model = model.eval()
    if int(model.num_classes) != NUM_CLASSES:
        raise SystemExit(f"the student has {model.num_classes} classes, not {NUM_CLASSES}")
    if "seed" in ckpt and int(ckpt["seed"]) != a.seed:
        raise SystemExit(f"the checkpoint records seed {ckpt['seed']}, not --seed {a.seed}")
    identity = {
        "schema": FP32_SCHEMA, "precision": FP32_PRECISION, "stage": a.stage, "seed": a.seed,
        "num_classes": NUM_CLASSES, "source_checkpoint_sha256": got, "source_checkpoint_file": src.name,
        "source_checkpoint_iter": ckpt.get("iter") if isinstance(ckpt.get("iter"), int) else None,
        "preprocess_protocol": PROTOCOL_ID,
        "preprocess_protocol_source": "the INT8 path's id: demo/server/preprocess.py PROTOCOL_ID = "
                                      "src.eval.adapters.PREPROCESS_PROTOCOL",
        "artifact_role": FP32_ARTIFACT_ROLE, "stand_in": bool(ckpt.get("stand_in", False)),
        "traced_example_shape": list(EXAMPLE_SHAPE), "input": "float32 [N, 3, H, W]; the 512x512 canvas",
        "output": f"float32 [N, {NUM_CLASSES}, H, W] logits",
        "model_code": "src/quant/checkpoint.py load_student_from_e3 -> src/models/student.py build_student",
        "mkldnn_enabled_at_trace": phone_fix["mkldnn_enabled"], "torch": torch.__version__,
        "writer": "demo/model/export_fp32_torchscript.py",
    }
    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        traced = torch.jit.trace(model, torch.zeros(EXAMPLE_SHAPE))
    trace_notes = sorted({f"{Path(w.filename).name}:{w.lineno}: {w.category.__name__}: "
                          f"{str(w.message).splitlines()[0][:160]}" for w in caught})
    buf = io.BytesIO()
    torch.jit.save(traced, buf, _extra_files={FP32_META_NAME: json.dumps(identity, sort_keys=True,
                                                                          separators=(",", ":"))})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "xb") as fh:
        fh.write(buf.getvalue())

    loaded = load_fp32_model(partial, a.stage)
    rows = []
    with torch.inference_mode():
        for x in synthetic_parity_inputs():
            rows.append(compare_logits(model(x), loaded.module(x)))
    ok = all(within_limits(r) for r in rows)
    if not ok:
        print(json.dumps(rows, indent=2))
        raise SystemExit(f"STOP: the traced file differs from the eager model beyond the parity limits; "
                         f"{partial} is kept for inspection and {out.name} was not written")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    os.replace(partial, out)
    common.assert_no_stats()
    print(json.dumps({"out": str(out), "size_bytes": out.stat().st_size, "sha256": common.sha256_file(out),
                      "identity": identity, "e6_cross_check": e6_check,
                      "round_trip": {"output_shapes": [r["shape"] for r in rows],
                                     "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                                     "min_argmax_agreement": min(r["argmax_agreement"] for r in rows),
                                     "bitwise_equal": sum(r["bitwise_equal"] for r in rows)},
                      "trace_warnings": trace_notes, "seconds": round(time.perf_counter() - t0, 2)}, indent=2))
    print(f"[export] wrote {out.name} sha256 {common.sha256_file(out)}; set it as e3's expected sha256 "
          f"(demo/config.py E3_SHA256 or PLANTSEG_DEMO_E3_SHA256) once its parity check has passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
