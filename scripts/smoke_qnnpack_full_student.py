#!/usr/bin/env python3
"""B7/B13 — eager-mode QNNPACK INT8 operator-support smoke: FULL student prepare/convert/forward.

Operator-support + conversion smoke ONLY (not latency, not training, no backward). Uses the existing
student from src/models/student.py as-is (NO hacks). Targets QNNPACK; runs the SAME flow under the
available engine as a CLEARLY-LABELED PROXY if QNNPACK is absent. Captures the EXACT exception on
failure. Acceptable outcomes: PASS_CLEAN / PASS_WITH_DOCUMENTED_FALLBACK /
HEAD_PASS_FULL_STUDENT_FAILS_DOCUMENTED.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.ao.quantization as tq  # noqa: E402

from src.seeds import set_seed  # noqa: E402
from src.models import build_student  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*reduce_range.*")


def select_engine():
    eng = list(torch.backends.quantized.supported_engines)
    if "qnnpack" in eng:
        torch.backends.quantized.engine = "qnnpack"
        return "qnnpack", False
    proxy = "onednn" if "onednn" in eng else ("fbgemm" if "fbgemm" in eng else eng[0])
    torch.backends.quantized.engine = proxy
    return proxy, True


def main() -> int:
    set_seed()
    engine, is_proxy = select_engine()
    tag = f"{engine}{' (PROXY - NOT QNNPACK)' if is_proxy else ''}"
    print(f"[engine] {tag} | supported={list(torch.backends.quantized.supported_engines)}")
    print(f"[note] eager-mode QNNPACK INT8 OPERATOR-SUPPORT smoke (not an x86 latency benchmark)")
    print(f"[note] student used as-is from src/models/student.py (no quantization-readiness hacks)")

    model = build_student(num_classes=116, pretrained=False).eval()

    fused = False
    try:
        model.fuse()                       # B7b: fuse Conv-BN-ReLU (head + backbone blocks)
        fused = True
        print("[fuse] model.fuse() called -> head Conv-BN-ReLU + backbone blocks fused")
    except Exception as e:  # noqa: BLE001
        print(f"[fuse] model.fuse() FAILED (continuing UNFUSED): {type(e).__name__}: {str(e)[:160]}")

    model.qconfig = tq.get_default_qconfig(engine)

    stage, outcome, shape, err = "init", "FULL_STUDENT_FAILS", None, None
    try:
        stage = "prepare"
        tq.prepare(model, inplace=True)
        stage = "calibrate"
        with torch.no_grad():
            for _ in range(2):
                model(torch.randn(1, 3, 512, 512))
        stage = "convert"
        tq.convert(model, inplace=True)
        stage = "forward(converted)"
        with torch.no_grad():
            out = model(torch.randn(1, 3, 512, 512))
        shape = tuple(out.shape)
        outcome = "PASS_CLEAN" if shape == (1, 116, 512, 512) else "FULL_STUDENT_FAILS"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:300]}"
        outcome = "FULL_STUDENT_FAILS"
        print(f"[fail-stage] {stage}")
        print(f"[exact-error] {err}")

    print(f"[FULL_STUDENT_OUTCOME] engine={engine} proxy={is_proxy} fusion_called={fused} "
          f"result={outcome} shape={shape} fail_stage={stage if err else 'n/a'}")
    # This script never asserts PASS for the full student unless it converts+forwards cleanly with
    # no fallback. A documented failure is an acceptable B7 outcome (exit 0) so the log/doc are produced.
    clean = (outcome == "PASS_CLEAN" and not is_proxy)
    print(f"[{'PASS_CLEAN' if clean else 'DOCUMENTED'}] full-student qnnpack smoke "
          f"({'PROXY' if is_proxy else 'QNNPACK'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
