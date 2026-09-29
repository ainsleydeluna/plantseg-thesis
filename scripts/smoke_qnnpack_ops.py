#!/usr/bin/env python3
"""Lane 8 QNNPACK operator census of a PTQ artifact -- a regression check only (errata E-4).

    python -B scripts/smoke_qnnpack_ops.py                                        # synthetic (cloud)
    python -B scripts/smoke_qnnpack_ops.py --provenance RUN_META --source-ckpt CKPT \\
        --calibration-list LIST --data-root ROOT --out-dir OUT                    # real (local, d2)

DL-18 measured QNNPACK support for the LR-ASPP Sigmoid (d44cce9) and the cloud smoke passed
smoke_qnnpack_head and smoke_qnnpack_full_student; this census re-checks that state on the PTQ artifact
scripts/run_ptq.py writes and does not reopen the question. With no arguments it builds a synthetic
artifact first (a random-weight E1-schema student, 140 synthetic TRAIN images, the seed-42 list drawn by
scripts/build_calibration_lists.py, all in a temp dir) through scripts/run_ptq.py itself.

  C1  engine qnnpack; the artifact passes validate_int8_artifact and loads as the evaluator loads it
  C2  module census (the eager rebuild of the artifact's state_dict): quantized Conv2d, ConvReLU2d,
      Hardswish and QFunctional present
  C3  run time: Sigmoid, Hardsigmoid, adaptive avg pool and the head interpolate ran on quantized
      tensors; the Sigmoid's output qparams are recorded (U6)
  C4  no float module other than the declared input Quantize and output DeQuantize
  C5  TorchScript graph: one quantize_per_tensor fed by the input, one dequantize feeding only the final
      upsample, no float-only op, every required quantized op present
  C6  every conv weight per-channel INT8, in the eager rebuild and inside the TorchScript file
  C7  forward at 512x512 on CPU: [1, 116, 512, 512] float; the TorchScript artifact and the eager rebuild
      give bitwise-equal outputs, so C2-C4 describe the artifact of record
  C8  recorded, descriptive: FP32-vs-INT8 argmax agreement on the calibration images (valid pixels)

Writes qnnpack_smoke_<UTC>.json {torch, engine, module_census, float_modules, sigmoid_quantized,
forward_ok, fp32_vs_int8_pixel_agreement_on_calib, ...} to OUT, outside the repository. Exit codes:
0 PASS; 1 FAIL (STOP: an undeclared float fallback or a per-tensor weight); 2 unreadable input.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_PASS, EXIT_FAIL, EXIT_UNREADABLE = 0, 1, 2
SMOKE_SCHEMA = "plantseg-qnnpack-ops-census/1.0.0"


def build_synthetic() -> dict:
    """A synthetic PTQ run through scripts/run_ptq.py; returns the census arguments."""
    from scripts.synthetic_ptq_fixtures import make_tree, safe_tmpdir
    base = safe_tmpdir("smoke_qnnpack_ops_")
    root = base / "data"
    make_tree(root, n_train=140, seed=8)
    os.environ.setdefault("PLANTSEG_DATA_ROOT", str(root))
    from scripts.build_calibration_lists import FILENAME, main as build_lists
    from scripts.run_ptq import main as run_ptq
    from scripts.synthetic_ptq_fixtures import make_e1_checkpoint

    ckpt = make_e1_checkpoint(base / "e1.pt")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc_l = build_lists(["--data-root", str(root), "--out-dir", str(base / "lists"),
                            "--expected-images", "140", "--generated-utc", "2026-10-01T00:00:00Z"])
        lst = base / "lists" / FILENAME.format(seed=42)
        ck = json.loads(lst.read_text())["checksum_sha256"] if lst.exists() else "0" * 64
        rc_r = run_ptq(["--real-run", "--confirm-real-run", "--stage", "E4", "--source-ckpt", str(ckpt),
                        "--expect-source-sha256", hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                        "--calibration-list", str(lst), "--expect-list-checksum", ck,
                        "--data-root", str(root), "--out-dir", str(base / "run")])
    if rc_l != 0 or rc_r != 0:
        print(buf.getvalue()[-2000:])
        raise RuntimeError(f"synthetic PTQ build failed (lists exit {rc_l}, run_ptq exit {rc_r})")
    return {"provenance": base / "run" / "e4_ptq_run_meta.json", "source_ckpt": ckpt,
            "calibration_list": lst, "data_root": root, "out_dir": base / "census", "synthetic": True}


def census(provenance, source_ckpt, calibration_list, data_root) -> tuple[list, dict]:
    """[(check, passed, detail)] and the census record."""
    import torch

    from src.eval.model_loading import load_int8_student, rebuild_int8_from_state_dict, sha256_file
    from src.eval.stage_artifacts import validate_int8_artifact
    from src.quant import ptq
    from src.quant.checkpoint import load_student_from_e1, load_student_from_e3

    out: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), str(detail)))

    p = Path(provenance)
    prov = json.loads(p.read_text(encoding="utf-8"))
    stage = str(prov["stage"])
    resolved = validate_int8_artifact(stage, p)
    ts, info = load_int8_student(resolved, require_qnnpack=True)
    check("C1 engine qnnpack; artifact validated and loaded as the evaluator loads it",
          torch.backends.quantized.engine == "qnnpack" and resolved["artifact_format"] == "torchscript"
          and isinstance(ts, torch.jit.ScriptModule),
          f"{stage} {info.sha256[:16]} engine={torch.backends.quantized.engine}")
    sd_rec = prov["state_dict_artifact"]
    sd_path = p.resolve().parent / sd_rec["path"]
    if sha256_file(sd_path) != sd_rec["sha256"]:
        raise RuntimeError(f"{sd_path} does not match the sha256 its run provenance records")
    eager = rebuild_int8_from_state_dict(sd_path, stage=stage, method="ptq")

    x = torch.randn(1, 3, 512, 512, generator=torch.Generator().manual_seed(ptq.PARITY_SEED))
    run = ptq.runtime_census(eager, x)
    graph = ptq.graph_census(ts)
    weights = {"eager": ptq.weight_scheme_report(eager), "torchscript": ptq.weight_scheme_report(ts)}
    req = run["required"]
    check("C2 module census: quantized Conv2d, ConvReLU2d, Hardswish, QFunctional",
          all(req[k] for k in ptq.REQUIRED_MODULE_TYPES),
          {k.split(".")[-1]: v for k, v in run["module_census"].items()
           if k in ptq.REQUIRED_MODULE_TYPES.values()})
    runtime_keys = [k for k in req if k not in ptq.REQUIRED_MODULE_TYPES]
    check("C3 Sigmoid, Hardsigmoid, adaptive avg pool and the head interpolate ran quantized",
          all(req[k] for k in runtime_keys) and run["sigmoid"]["quantized"],
          f"sigmoid u6={run['sigmoid']['u6_values']}")
    check("C4 no float module besides the declared input Quantize / output DeQuantize",
          not run["float_modules"] and all(b["ok"] for b in run["declared_float_boundary"].values())
          and run["quantize_modules"] == run["dequantize_modules"] == 1,
          f"float_modules={run['float_modules']} unused={len(run['unused_leaf_modules'])}")
    check("C5 TorchScript graph: one quantize (input), one dequantize (final upsample only), no float op",
          graph["ok"], f"quantize={graph['quantize_per_tensor']} dequantize={graph['dequantize']} "
                       f"float_only_ops={graph['float_only_ops']}")
    check("C6 every conv weight per-channel INT8 (eager rebuild and TorchScript file)",
          all(w["ok"] for w in weights.values()),
          {k: f"{w['per_channel_int8']}/{w['quantized_convs']} {w['qschemes']}" for k, w in weights.items()})
    with torch.no_grad():
        y_ts, y_eager = ts(x), eager(x)
    forward_ok = tuple(y_ts.shape) == (1, 116, 512, 512) and not y_ts.is_quantized
    check("C7 forward 512x512 on CPU; TorchScript == eager rebuild bitwise",
          forward_ok and torch.equal(y_ts, y_eager), f"shape={tuple(y_ts.shape)} dtype={y_ts.dtype}")

    # C8 (descriptive): agreement with the FP32 parent on the calibration images
    src = Path(source_ckpt)
    if sha256_file(src) != prov["source_checkpoint_sha256"]:
        raise RuntimeError(f"{src} is not the source checkpoint this artifact records")
    fp32, _ = (load_student_from_e1 if prov["source_stage"] == "E1" else load_student_from_e3)(src)
    cal, _ = ptq.load_calibration_list(calibration_list,
                                       expected_checksum=prov["calibration"]["checksum_sha256"])
    ds = ptq.calibration_dataset(data_root, cal["selected_ids"])
    agreement = ptq.pixel_agreement(fp32.eval(), ts, ptq.iter_calibration(ds))
    check("C8 FP32-vs-INT8 agreement on the calibration images recorded (descriptive)",
          agreement["images"] == len(cal["selected_ids"]),
          f"valid-pixel agreement {agreement['agreement_valid_pixels']} over {agreement['images']} images")

    record = {
        "schema": SMOKE_SCHEMA, "regression_only": "errata E-4 (DL-18 measured Sigmoid support)",
        "torch": torch.__version__, "engine": torch.backends.quantized.engine,
        "stage": stage, "artifact": {"path": str(resolved["artifact_path"]), "sha256": info.sha256,
                                     "format": resolved["artifact_format"]},
        "source_checkpoint_sha256": prov["source_checkpoint_sha256"],
        "calibration_checksum_sha256": prov["calibration"]["checksum_sha256"],
        "module_census": run["module_census"], "float_modules": run["float_modules"],
        "declared_float_boundary": run["declared_float_boundary"],
        "unused_leaf_modules": run["unused_leaf_modules"],
        "sigmoid_quantized": run["sigmoid"]["quantized"], "sigmoid": run["sigmoid"],
        "head_interpolate": run["head_interpolate"], "final_upsample": run["final_upsample"],
        "graph": graph, "weights": weights, "forward_ok": forward_ok,
        "fp32_vs_int8_pixel_agreement_on_calib": agreement,
        "checks": [{"check": n, "passed": ok, "detail": d} for n, ok, d in out],
    }
    return out, record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lane 8 QNNPACK operator census (regression only, E-4).")
    ap.add_argument("--provenance", default=None, help="a scripts/run_ptq.py run provenance")
    ap.add_argument("--source-ckpt", default=None, help="the FP32 source checkpoint it records")
    ap.add_argument("--calibration-list", default=None, help="the calibration list it records")
    ap.add_argument("--data-root", default=None, help="the TRAIN images of that list")
    ap.add_argument("--out-dir", default=None, help="where qnnpack_smoke_<UTC>.json goes (outside the repo)")
    args = ap.parse_args(argv)
    real = [args.provenance, args.source_ckpt, args.calibration_list, args.data_root, args.out_dir]
    print("=" * 78)
    print("QNNPACK OPERATOR CENSUS (lane 8, regression only per errata E-4)")
    print("=" * 78)
    try:
        if any(real) and not all(real):
            raise RuntimeError("a real census needs --provenance, --source-ckpt, --calibration-list, "
                               "--data-root and --out-dir together")
        cfg = ({"provenance": args.provenance, "source_ckpt": args.source_ckpt,
                "calibration_list": args.calibration_list, "data_root": args.data_root,
                "out_dir": args.out_dir, "synthetic": False} if all(real) else build_synthetic())
        out_dir = Path(cfg["out_dir"]).resolve()
        if out_dir == REPO or REPO in out_dir.parents:
            raise RuntimeError("the census record is never written inside the repository")
        results, record = census(cfg["provenance"], cfg["source_ckpt"], cfg["calibration_list"],
                                 cfg["data_root"])
    except Exception as e:                            # noqa: BLE001 -- unreadable, never a verdict
        print(f"RESULT: ERROR -- {type(e).__name__}: {e}")
        return EXIT_UNREADABLE
    record["synthetic"] = cfg["synthetic"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record["created_utc"] = stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"qnnpack_smoke_{stamp}.json"
    with open(target, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(record, indent=2, allow_nan=False, default=str) + "\n")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nrecord: {target}")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL -- STOP'} ({passed}/{len(results)})"
          f"{' synthetic artifact' if cfg['synthetic'] else ''}")
    return EXIT_PASS if passed == len(results) else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
