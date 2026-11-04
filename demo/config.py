"""Demo configuration (lane DEMO-1).

SWAPPING IN THE ARTIFACT OF RECORD (after the TEST session) is the one line marked below: point
MODEL_PATH at the INT8 QNNPACK TorchScript file of record (absolute path, or relative to the
repository root). Nothing else changes: the server reads the file's name, sha256 and embedded
identity (plantseg_int8.json) at start-up and shows them on the page. A container or a host can
instead set the environment variable PLANTSEG_DEMO_MODEL to the same path.
See docs/lane_reports/demo.md section 7.
"""
from __future__ import annotations

import os
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
MODELS_DIR = DEMO_DIR / "models"          # generated files only (demo/.gitignore); never committed

STAND_IN_PATH = MODELS_DIR / "stand_in_int8_qnnpack.pt"

# ---- THE ONE LINE TO CHANGE AFTER TEST: the model file the demo serves ---------------------------
MODEL_PATH = Path(os.environ.get("PLANTSEG_DEMO_MODEL") or STAND_IN_PATH)
# --------------------------------------------------------------------------------------------------

# In-browser (ONNX Runtime Web) files written by demo/ondevice/export_onnx.py.
ONNX_FP32_PATH = MODELS_DIR / "student_fp32.onnx"
ONNX_INT8_PATH = MODELS_DIR / "student_int8_qdq.onnx"

HOST = "0.0.0.0"        # reachable from a phone on the same network; use 127.0.0.1 for local-only
PORT = 8000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TOP_K = 5               # how many disease classes the page lists by pixel share


def resolve_model_path(path: str | os.PathLike | None = None) -> Path:
    """An explicit path, else MODEL_PATH; a relative path is taken from the repository root."""
    p = Path(path) if path is not None else MODEL_PATH
    return p if p.is_absolute() else (REPO_ROOT / p)
