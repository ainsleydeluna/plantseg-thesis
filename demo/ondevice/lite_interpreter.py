#!/usr/bin/env python3
"""Can the INT8 QNNPACK TorchScript artifact be saved for the PyTorch lite interpreter (LibTorch-Lite)?

    python3 -I demo/ondevice/lite_interpreter.py [--model INT8.pt]

Loads the artifact through the evaluator's loader (QNNPACK), then tries, recording each outcome:
  1 module._save_for_lite_interpreter(...)               -> .ptl, reloaded with
    torch.jit.mobile._load_for_lite_interpreter and compared with the TorchScript output (torch.equal)
  2 torch.utils.mobile_optimizer.optimize_for_mobile(...) -> .ptl, the same comparison
  3 torch.jit.mobile._backport_for_mobile to each older bytecode version, the same comparison
Records file sizes, bytecode versions and the operator lists the iOS runtime must contain.
Writes demo/models/lite_*.ptl and lite_interpreter_facts.json. Runs on Linux/Windows; no Mac needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402


def inputs():
    import torch

    from demo.server.preprocess import evaluator_preprocess
    from demo.synthetic import calibration_images

    g = torch.Generator().manual_seed(7)
    xs = [torch.randn(1, 3, 512, 512, generator=g) for _ in range(2)]
    xs += [evaluator_preprocess(im).tensor for im in calibration_images(2, 7000)]
    return xs


def compare(ref_module, path: Path, xs) -> dict:
    import torch
    from torch.jit.mobile import _get_model_bytecode_version, _load_for_lite_interpreter

    lite = _load_for_lite_interpreter(str(path))
    equal, max_diff = 0, 0.0
    with torch.no_grad():
        for x in xs:
            a, b = ref_module(x), lite(x)
            if a.shape == b.shape and torch.equal(a, b):
                equal += 1
            else:
                max_diff = max(max_diff, float((a - b).abs().max()) if a.shape == b.shape else float("inf"))
    return {"file": path.name, "size_bytes": path.stat().st_size, "sha256": common.sha256_file(path),
            "bytecode_version": int(_get_model_bytecode_version(str(path))),
            "bitwise_equal_outputs": f"{equal}/{len(xs)}", "max_abs_diff_if_unequal": max_diff}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="lite-interpreter feasibility")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)

    import torch
    from torch.jit.mobile import _backport_for_mobile, _get_model_bytecode_version

    from demo.server.inference import load_model

    src = config.resolve_model_path(a.model)
    loaded = load_model(src)
    ts = loaded.module
    xs = inputs()
    facts = {"source_model": src.name, "source_sha256": loaded.sha256, "torch": torch.__version__,
             "engine": torch.backends.quantized.engine, "attempts": {}}
    out_dir = config.MODELS_DIR

    # 1 plain lite save
    plain = out_dir / "lite_int8_qnnpack.ptl"
    try:
        ts._save_for_lite_interpreter(str(plain))
        facts["attempts"]["save_for_lite_interpreter"] = {"ok": True, **compare(ts, plain, xs)}
    except Exception as e:  # noqa: BLE001
        facts["attempts"]["save_for_lite_interpreter"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                                          "trace": traceback.format_exc()[-1500:]}

    # 2 optimize_for_mobile then lite save
    opt = out_dir / "lite_int8_qnnpack_optimized.ptl"
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile

        om = optimize_for_mobile(ts)
        om._save_for_lite_interpreter(str(opt))
        facts["attempts"]["optimize_for_mobile"] = {"ok": True, **compare(ts, opt, xs),
                                                    "root_ops": sorted(torch.jit.export_opnames(om))}
    except Exception as e:  # noqa: BLE001
        facts["attempts"]["optimize_for_mobile"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:1500]}",
                                                    "trace": traceback.format_exc()[-1500:]}

    # operator list the runtime needs (from the plain lite file)
    if plain.is_file():
        try:
            from torch.jit.mobile import _export_operator_list, _load_for_lite_interpreter

            facts["operator_list"] = sorted(_export_operator_list(_load_for_lite_interpreter(str(plain))))
        except Exception as e:  # noqa: BLE001
            facts["operator_list_error"] = f"{type(e).__name__}: {e}"
        # 3 backports to older bytecode versions (older LibTorch-Lite runtimes read older bytecode)
        current = int(_get_model_bytecode_version(str(plain)))
        backports = {}
        for v in range(current - 1, 3, -1):
            dst = out_dir / f"lite_int8_qnnpack_bytecode_v{v}.ptl"
            try:
                ok = bool(_backport_for_mobile(str(plain), str(dst), v))
                backports[v] = {"ok": ok, **(compare(ts, dst, xs) if ok and dst.is_file() else {})}
            except Exception as e:  # noqa: BLE001
                backports[v] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        facts["backports"] = backports
    common.assert_no_stats()
    out = out_dir / "lite_interpreter_facts.json"
    out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "trace"} if isinstance(v, dict) else v)
                      for k, v in facts["attempts"].items()}, indent=2))
    print("operator_list:", facts.get("operator_list"))
    print("backports:", {k: (v.get("ok"), v.get("bitwise_equal_outputs"), v.get("error", "")[:120])
                         for k, v in facts.get("backports", {}).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
