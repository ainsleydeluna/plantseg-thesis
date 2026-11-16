"""Demo configuration (lane DEMO-1; the model registry is DEMO-3).

THE MODEL REGISTRY. The server has three slots, each named by rule and never by a score:

  e6     "E6 · INT8 (QAT) · the demo model"                       always required
  e3     "E3 · FP32 · the model E6 was quantized from"            optional
  shown  "AM-29's shown student · INT8 · the highest of the candidates on VAL (exploratory)"  optional

Each slot has a model file (an environment variable, else its marked line below) and an expected
sha256 (an environment variable, else its marked line below; empty until the orchestrator sets it).
The server loads a slot only if its expected sha256 is set and the file's sha256 equals it, and it
loads no other model. A slot whose sha256 is not set is not loaded and its option is hidden. e6 must
load, or the server refuses to start; a set sha256 that the file does not match also stops start-up.

e6 keeps today's route: PLANTSEG_DEMO_MODEL names its file, and MODEL_PATH is its marked line.
See docs/lane_reports/demo.md section 7 and docs/lane_reports/demo3.md.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
MODELS_DIR = DEMO_DIR / "models"          # generated files only (demo/.gitignore); never committed

STAND_IN_PATH = MODELS_DIR / "stand_in_int8_qnnpack.pt"

# ---- THE MARKED LINES: each slot's model file and its expected sha256 -----------------------------
# e6, the demo model: E6 seed 42, INT8 QNNPACK TorchScript (ruling DEMO-MODEL-2; decision-log DL-264).
MODEL_PATH = Path(os.environ.get("PLANTSEG_DEMO_MODEL") or STAND_IN_PATH)
E6_SHA256 = os.environ.get("PLANTSEG_DEMO_E6_SHA256") or \
    "cf14ae0362d07f6a576405320780622f504f1bccd032335c2968c02e37dbb0bb"
# e3: the FP32 TorchScript file demo/model/export_fp32_torchscript.py writes from E3 seed 42 (after AM-30).
E3_PATH = os.environ.get("PLANTSEG_DEMO_E3_MODEL") or ""
E3_SHA256 = os.environ.get("PLANTSEG_DEMO_E3_SHA256") or ""
# shown: AM-29's shown student, an all-INT8 QNNPACK TorchScript file of the same form as E6's.
SHOWN_PATH = os.environ.get("PLANTSEG_DEMO_SHOWN_MODEL") or ""
SHOWN_SHA256 = os.environ.get("PLANTSEG_DEMO_SHOWN_SHA256") or ""
# --------------------------------------------------------------------------------------------------

# In-browser (ONNX Runtime Web) files written by demo/ondevice/export_onnx.py.
ONNX_FP32_PATH = MODELS_DIR / "student_fp32.onnx"
ONNX_INT8_PATH = MODELS_DIR / "student_int8_qdq.onnx"

HOST = "0.0.0.0"        # reachable from a phone on the same network; use 127.0.0.1 for local-only
PORT = 8000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TOP_K = 5               # how many disease classes the page lists by pixel share

KIND_INT8 = "int8"      # INT8 QNNPACK TorchScript carrying plantseg_int8.json (src/quant writers)
KIND_FP32 = "fp32"      # FP32 TorchScript carrying plantseg_fp32.json (demo/model/export_fp32_torchscript.py)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SlotSpec:
    """One registry slot. The label, description and option text are fixed: never derived from a score."""
    key: str
    kind: str
    label: str
    description: str
    option: str           # the page's "Compare with" text; "" for e6, whose result is always shown
    required: bool
    path_env: str
    sha256_env: str
    stage: str = ""       # the stage the file's embedded identity must name; "" = whatever it names


SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("e6", KIND_INT8, "E6 · INT8 (QAT)", "the demo model", "", True,
             "PLANTSEG_DEMO_MODEL", "PLANTSEG_DEMO_E6_SHA256"),
    SlotSpec("e3", KIND_FP32, "E3 · FP32", "the model E6 was quantized from", "E3 (FP32)", False,
             "PLANTSEG_DEMO_E3_MODEL", "PLANTSEG_DEMO_E3_SHA256", stage="E3"),
    SlotSpec("shown", KIND_INT8, "AM-29's shown student · INT8",
             "the highest of the candidates on VAL (exploratory)", "shown student (INT8, exploratory)", False,
             "PLANTSEG_DEMO_SHOWN_MODEL", "PLANTSEG_DEMO_SHOWN_SHA256"),
)
SLOT_KEYS = tuple(s.key for s in SLOTS)
_MARKED = {"e6": (MODEL_PATH, E6_SHA256), "e3": (E3_PATH, E3_SHA256), "shown": (SHOWN_PATH, SHOWN_SHA256)}


class SlotConfigError(RuntimeError):
    """A registry setting the server refuses to start with."""


@dataclass(frozen=True)
class ResolvedSlot:
    spec: SlotSpec
    path: Path | None             # absolute, or None when no file is named
    expected_sha256: str          # "" when not set
    path_source: str              # where the path came from (an environment variable, the marked line, ...)
    sha256_source: str

    @property
    def configured(self) -> bool:
        """A slot is loaded only when its expected sha256 is set."""
        return bool(self.expected_sha256)


def resolve_model_path(path: str | os.PathLike | None = None) -> Path:
    """An explicit path, else MODEL_PATH (e6); a relative path is taken from the repository root."""
    p = Path(path) if path is not None else MODEL_PATH
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_slots(paths: dict | None = None, sha256s: dict | None = None) -> list[ResolvedSlot]:
    """The three slots in order (e6, e3, shown). `paths` / `sha256s` override a slot for one run.

    Raises SlotConfigError for a malformed sha256 or a set sha256 with no file named. Whether the
    file exists and matches is checked by the loader (demo/server/inference.py), before it is read.
    """
    paths, sha256s = dict(paths or {}), dict(sha256s or {})
    unknown = sorted((set(paths) | set(sha256s)) - set(SLOT_KEYS))
    if unknown:
        raise SlotConfigError(f"unknown slot(s) {unknown}; the slots are {list(SLOT_KEYS)}")
    out = []
    for spec in SLOTS:
        marked_path, marked_sha = _MARKED[spec.key]
        if paths.get(spec.key):
            raw_path, path_src = paths[spec.key], "command line"
        else:
            raw_path = marked_path
            path_src = (f"environment {spec.path_env}" if os.environ.get(spec.path_env)
                        else "demo/config.py marked line")
        if sha256s.get(spec.key):
            sha, sha_src = sha256s[spec.key], "command line"
        else:
            sha = marked_sha
            sha_src = (f"environment {spec.sha256_env}" if os.environ.get(spec.sha256_env)
                       else "demo/config.py marked line")
        sha = str(sha).strip().lower()
        if sha and not _HEX64.match(sha):
            raise SlotConfigError(f"slot {spec.key}: the expected sha256 {sha!r} ({sha_src}) is not 64 hex digits")
        path = resolve_model_path(raw_path) if str(raw_path) else None
        if sha and path is None:
            raise SlotConfigError(f"slot {spec.key}: an expected sha256 is set ({sha_src}) but no model file "
                                  f"is named; set {spec.path_env} or its marked line in demo/config.py")
        out.append(ResolvedSlot(spec, path, sha, path_src if path else "not set", sha_src if sha else "not set"))
    return out
