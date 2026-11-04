#!/usr/bin/env python3
"""Only if the INT8 artifact of record is a runner-schema state_dict (src/quant/runner.py) rather than
the TorchScript file src/quant/ptq.py writes: make the TorchScript file the demo loads, with the
repository's own functions, and prove the two give bitwise-equal outputs.

    python3 -I demo/model/torchscript_from_state_dict.py STATE.pt --stage E5 --method qat --out OUT.pt

  rebuild  src/eval/model_loading.py rebuild_int8_from_state_dict (the evaluator's rebuild: the converted
           skeleton from the committed quantization helpers, strict load; nothing is recalibrated)
  trace    src/quant/ptq.py torchscript_bytes (1x3x512x512 trace + plantseg_int8.json)
  check    src/quant/ptq.py output_parity, converted module vs the written TorchScript, bitwise
The embedded identity names the source file and its sha256, with artifact_role "demo_conversion": the
file is a conversion for the demo, never an artifact of record. Run it in a fresh process; it refuses
to overwrite OUT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_dict")
    ap.add_argument("--stage", required=True, help="the artifact's stage, e.g. E5")
    ap.add_argument("--method", required=True, choices=("ptq", "qat"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src, out = Path(a.state_dict), Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    from src.eval.model_loading import (INT8_TORCHSCRIPT_SCHEMA, load_int8_torchscript,
                                        rebuild_int8_from_state_dict, select_int8_backend)
    from src.quant.ptq import (output_parity, student_traced_in_process, synthetic_parity_inputs,
                               torchscript_bytes)

    if student_traced_in_process():
        raise SystemExit("run in a fresh process")
    select_int8_backend(require_qnnpack=True)
    converted = rebuild_int8_from_state_dict(src, stage=a.stage.upper(), method=a.method)
    meta = {"schema": INT8_TORCHSCRIPT_SCHEMA, "stage": a.stage.upper(), "quantization": a.method,
            "num_classes": 116, "engine": "qnnpack", "artifact_role": "demo_conversion",
            "converted_from": src.name, "converted_from_sha256": common.sha256_file(src),
            "converter": "demo/model/torchscript_from_state_dict.py"}
    blob, notes = torchscript_bytes(converted, meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as fh:
        fh.write(blob)
    ts, _ = load_int8_torchscript(out, stage=meta["stage"], method=a.method, artifact_role="demo_conversion")
    parity = output_parity({"converted": converted, "torchscript": ts}, synthetic_parity_inputs(),
                           [("converted", "torchscript")])
    common.assert_no_stats()
    print(json.dumps({"out": str(out), "sha256": common.sha256_file(out), "bitwise_parity": parity["all_equal"],
                      "trace_warnings": len(notes)}, indent=2))
    return 0 if parity["all_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
