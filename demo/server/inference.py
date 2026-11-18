"""Model loading and one segmentation request, on CPU, for the demo server.

The registry (demo/config.py) names up to three models. A slot is loaded only when its expected sha256
is set and the file's sha256, computed before the file is deserialized, equals it; e6 must load. Each
file is then loaded by the loader for its kind:

  INT8  the evaluator's own loader (src/eval/model_loading.py load_int8_torchscript) under the QNNPACK
        engine it requires (select_int8_backend). The identity the loader verifies (stage, method,
        artifact role) is read from the file's own embedded plantseg_int8.json; the loader still
        enforces the schema, 116 classes and the qnnpack engine.
  FP32  load_fp32_model below: the TorchScript file demo/model/export_fp32_torchscript.py writes, with
        its embedded plantseg_fp32.json checked the same way (schema, precision, stage, seed, source
        checkpoint sha256, 116 classes, the preprocessing id) before and after torch.jit.load.

Every process that imports this module runs with PHONE-FIX-1 (demo/common.py phone_fix: mkldnn off).

Prediction = argmax over the 116 logits on the 512x512 canvas (the canvas protocol's prediction),
cropped to the photo's region; shares are over that region only, never over the padding. A comparison
runs the second model on the same input tensor as E6, so both panels show the same photo.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from demo import common, config
from demo.server.preprocess import CANVAS, PROTOCOL_ID, evaluator_preprocess

PHONE_FIX = common.phone_fix()          # PHONE-FIX-1, before any model is loaded in this process

META_NAME = "plantseg_int8.json"        # = src.eval.model_loading.INT8_TORCHSCRIPT_META (smoke-checked)
FP32_META_NAME = "plantseg_fp32.json"   # written by demo/model/export_fp32_torchscript.py
FP32_SCHEMA = "plantseg-fp32-torchscript/1.0.0"
FP32_PRECISION = "fp32"
FP32_ARTIFACT_ROLE = "demo_fp32_export"
FP32_META_KEYS = ("schema", "precision", "stage", "seed", "num_classes", "source_checkpoint_sha256",
                  "preprocess_protocol", "artifact_role")
FP32_ENGINE = "fp32 CPU (mkldnn off)"
NUM_CLASSES = 116                       # = src.eval.model_loading.FROZEN_NUM_CLASSES (smoke-checked)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class DemoError(ValueError):
    """A request the demo refuses (bad upload); the message is shown to the user."""


class SlotRefused(RuntimeError):
    """A registry slot the server refuses to start with (a wrong sha256, a missing file, a bad identity)."""


def read_embedded_meta(path: Path, name: str = META_NAME) -> dict:
    """The embedded identity file `name` from the TorchScript zip, without loading the module."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith("/extra/" + name)]
        if len(names) != 1:
            writer = ("demo/model/export_fp32_torchscript.py" if name == FP32_META_NAME
                      else "src/quant/ptq.py or src/quant/qat_artifacts.py")
            raise RuntimeError(f"{path.name} carries {len(names)} {name} entries; expected exactly 1 "
                               f"(only TorchScript files written by {writer} are accepted)")
        return json.loads(zf.read(names[0]).decode("utf-8"))


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def jpeg_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class LoadedModel:
    module: torch.jit.ScriptModule
    meta: dict
    path: Path
    sha256: str
    size_bytes: int
    engine: str
    num_classes: int
    load_ms: float
    kind: str = config.KIND_INT8
    slot: str = "e6"
    label: str = ""
    description: str = ""
    option: str = ""
    sha256_source: str = ""
    notes: list = field(default_factory=list)

    @property
    def stage(self) -> str:
        return str(self.meta.get("stage"))

    @property
    def stand_in(self) -> bool:
        return bool(self.meta.get("stand_in", False))


def _warm_up(module, path: Path) -> int:
    with torch.inference_mode():
        y = module(torch.zeros(1, 3, CANVAS, CANVAS))          # warm-up and shape check
    if y.dim() != 4 or tuple(y.shape[2:]) != (CANVAS, CANVAS) or y.dtype != torch.float32:
        raise RuntimeError(f"{path.name}: unexpected model output {tuple(y.shape)} {y.dtype}")
    return int(y.shape[1])


def load_model(path: Path) -> LoadedModel:
    """An INT8 QNNPACK TorchScript file through the evaluator's loader (no registry check: tools use it)."""
    from src.eval.model_loading import load_int8_torchscript, select_int8_backend

    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path} (build the stand-in with "
                                "python3 -I demo/model/build_stand_in.py, or set MODEL_PATH)")
    t0 = time.perf_counter()
    engine = select_int8_backend(require_qnnpack=True)
    peek = read_embedded_meta(path)
    module, meta = load_int8_torchscript(path, stage=str(peek.get("stage")),
                                         method=str(peek.get("quantization")),
                                         artifact_role=str(peek.get("artifact_role")))
    n = _warm_up(module, path)
    return LoadedModel(module=module, meta=meta, path=path, sha256=common.sha256_file(path),
                       size_bytes=path.stat().st_size, engine=engine, num_classes=n,
                       load_ms=(time.perf_counter() - t0) * 1000.0)


def check_fp32_identity(meta: dict, path: Path, expected_stage: str = "") -> None:
    """The embedded plantseg_fp32.json must carry every key with the values the demo serves."""
    missing = [k for k in FP32_META_KEYS if k not in meta]
    if missing:
        raise SlotRefused(f"{path.name}: {FP32_META_NAME} lacks {missing}")
    expected = {"schema": FP32_SCHEMA, "precision": FP32_PRECISION, "num_classes": NUM_CLASSES,
                "preprocess_protocol": PROTOCOL_ID, "artifact_role": FP32_ARTIFACT_ROLE}
    if expected_stage:
        expected["stage"] = expected_stage
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise SlotRefused(f"{path.name}: embedded identity {wrong} does not match the expected "
                          f"{ {k: expected[k] for k in wrong} }")
    if not isinstance(meta["seed"], int) or isinstance(meta["seed"], bool):
        raise SlotRefused(f"{path.name}: {FP32_META_NAME} seed {meta['seed']!r} is not an integer")
    if not (isinstance(meta["stage"], str) and meta["stage"]):
        raise SlotRefused(f"{path.name}: {FP32_META_NAME} stage {meta['stage']!r} is empty")
    if not _HEX64.match(str(meta["source_checkpoint_sha256"])):
        raise SlotRefused(f"{path.name}: {FP32_META_NAME} source_checkpoint_sha256 is not 64 hex digits")


def load_fp32_model(path: Path, expected_stage: str = "") -> LoadedModel:
    """An FP32 TorchScript file written by demo/model/export_fp32_torchscript.py, identity checked."""
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    t0 = time.perf_counter()
    peek = read_embedded_meta(path, FP32_META_NAME)
    check_fp32_identity(peek, path, expected_stage)
    extra = {FP32_META_NAME: ""}
    try:
        module = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
    except Exception as e:  # noqa: BLE001 -- surfaced verbatim
        raise SlotRefused(f"FP32 TorchScript load failed for {path}: {type(e).__name__}: {str(e)[:300]}") from e
    raw = extra[FP32_META_NAME]
    meta = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    if meta != peek:
        raise SlotRefused(f"{path.name}: the identity torch.jit.load read differs from the zip's {FP32_META_NAME}")
    check_fp32_identity(meta, path, expected_stage)
    module = module.eval()
    n = _warm_up(module, path)
    if n != NUM_CLASSES:
        raise SlotRefused(f"{path.name}: the model outputs {n} classes; its identity says {NUM_CLASSES}")
    return LoadedModel(module=module, meta=meta, path=path, sha256=common.sha256_file(path),
                       size_bytes=path.stat().st_size, engine=FP32_ENGINE, num_classes=n,
                       load_ms=(time.perf_counter() - t0) * 1000.0, kind=config.KIND_FP32)


def check_e3_source(e6_meta: dict, e3_meta: dict) -> str:
    """The e3 slot says "the model E6 was quantized from": when E6's identity names its source checkpoint
    (src/quant/qat_artifacts.py writes source_checkpoint_sha256), the e3 file must have been exported
    from that same checkpoint. Returns a note; raises SlotRefused on a mismatch."""
    e6_src = e6_meta.get("source_checkpoint_sha256")
    e3_src = e3_meta.get("source_checkpoint_sha256")
    if not e6_src:
        return "e3 source checkpoint not cross-checked: E6's identity names no source_checkpoint_sha256"
    if e6_src != e3_src:
        raise SlotRefused(f"e3 was exported from checkpoint {e3_src}, but E6's identity says E6 was quantized "
                          f"from {e6_src}; the e3 label would be false")
    return f"e3 source checkpoint {e3_src[:12]}... equals the one E6's identity names"


def load_registry(slots: list) -> tuple[list[LoadedModel], list[str]]:
    """Load the configured slots (config.resolve_slots order: e6, e3, shown) and nothing else.

    Every configured file is hashed and compared before any file is deserialized, so a wrong sha256
    stops start-up without reading a model. Returns (loaded models, start-up notes).
    """
    notes: list[str] = []
    todo = []
    for rs in slots:
        key = rs.spec.key
        if not rs.configured:
            if rs.spec.required:
                raise SlotRefused(f"slot {key} is required and has no expected sha256; set "
                                  f"{rs.spec.sha256_env} or its marked line in demo/config.py")
            named = f" (a file is named, {rs.path}, but it is not read)" if rs.path else ""
            notes.append(f"{key}: not loaded: no expected sha256 is set{named}; its option is hidden")
            continue
        if rs.path is None or not rs.path.is_file():
            raise SlotRefused(f"slot {key}: model file not found: {rs.path} ({rs.path_source})")
        actual = common.sha256_file(rs.path)
        if actual != rs.expected_sha256:
            raise SlotRefused(f"slot {key}: {rs.path.name} has sha256 {actual}, but the expected sha256 is "
                              f"{rs.expected_sha256} ({rs.sha256_source}); refused, the file was not loaded")
        todo.append((rs, actual))
    shas = [a for _, a in todo]
    if len(set(shas)) != len(shas):
        raise SlotRefused("two slots name the same file (equal sha256); each slot must hold its own model")
    loaded = []
    for rs, actual in todo:
        spec = rs.spec
        m = (load_fp32_model(rs.path, spec.stage) if spec.kind == config.KIND_FP32 else load_model(rs.path))
        if m.sha256 != actual:
            raise SlotRefused(f"slot {spec.key}: {rs.path.name} changed while it was loaded")
        if m.kind != spec.kind:
            raise SlotRefused(f"slot {spec.key} takes a {spec.kind} file; {rs.path.name} is {m.kind}")
        m.slot, m.label, m.description, m.option = spec.key, spec.label, spec.description, spec.option
        m.sha256_source = rs.sha256_source
        loaded.append(m)
    by = {m.slot: m for m in loaded}
    if "e3" in by:
        notes.append(check_e3_source(by["e6"].meta, by["e3"].meta))
    return loaded, notes


class Segmenter:
    """The loaded models + the repository's class list. Requests are serialised (one CPU)."""

    def __init__(self, models: list[LoadedModel], threads: int | None = None, notes: list[str] | None = None):
        if threads:
            torch.set_num_threads(int(threads))
        self.threads = torch.get_num_threads()
        self.classes, self.class_provenance = common.load_classes()
        self.data = common.data_facts()
        self.models = {m.slot: m for m in models}
        if "e6" not in self.models:
            raise SlotRefused("slot e6 is required")
        self.model = self.models["e6"]
        self.notes = list(notes or [])
        for m in models:
            n = m.num_classes
            if not (n == len(self.classes) == self.data["num_classes"]):
                raise RuntimeError(f"class count mismatch: {m.slot} outputs {n}, class map has "
                                   f"{len(self.classes)}, configs/data.py says {self.data['num_classes']}")
        n = self.model.num_classes
        self._lut = np.zeros((n, 4), dtype=np.uint8)            # class -> RGBA overlay colour
        for c in self.classes:
            if c.role != "background_or_non_disease":
                self._lut[c.class_id] = (*c.color, 255)
        self._lock = threading.Lock()
        self.runtime = {**common.phone_fix(), "phone_fix": "PHONE-FIX-1"}
        common.assert_no_stats()

    @classmethod
    def from_registry(cls, slots: list, threads: int | None = None) -> "Segmenter":
        """The server's route: only the configured slots whose file's sha256 equals the configured value."""
        if threads:
            torch.set_num_threads(int(threads))
        models, notes = load_registry(slots)
        return cls(models, threads, notes)

    @classmethod
    def for_file(cls, path: Path, threads: int | None = None) -> "Segmenter":
        """Tools only (demo/tools/bench_server.py): one INT8 file as e6, without the registry."""
        if threads:
            torch.set_num_threads(int(threads))
        m = load_model(Path(path))
        spec = config.SLOTS[0]
        m.label, m.description = spec.label, spec.description
        return cls([m], threads)

    # -------------------------------------------------------------------------------- metadata
    def model_info(self, slot: str = "e6") -> dict:
        m = self.models[slot]
        fmt = ("TorchScript (torch.jit.trace of the converted eager INT8 module)" if m.kind == config.KIND_INT8
               else "TorchScript (torch.jit.trace of the eager FP32 student)")
        return {"slot": m.slot, "label": m.label, "description": m.description, "option": m.option,
                "file": m.path.name, "sha256": m.sha256, "sha256_short": common.short_sha(m.sha256),
                "sha256_source": m.sha256_source, "size_bytes": m.size_bytes, "format": fmt, "kind": m.kind,
                "engine": m.engine, "stage": m.stage, "num_classes": m.num_classes, "embedded_meta": m.meta,
                "stand_in": m.stand_in, "load_ms": round(m.load_ms, 1), "torch": torch.__version__,
                "threads": self.threads}

    def models_info(self) -> list[dict]:
        return [self.model_info(k) for k in config.SLOT_KEYS if k in self.models]

    def compare_options(self) -> list[dict]:
        """The page's "Compare with" options: loaded optional slots only (an unloaded slot is hidden)."""
        return [{"slot": m.slot, "option": m.option, "label": m.label, "description": m.description}
                for k in config.SLOT_KEYS[1:] if (m := self.models.get(k)) is not None]

    def classes_info(self) -> list[dict]:
        return [{"class_id": c.class_id, "name": c.name, "label": c.label, "role": c.role,
                 "color": "#%02x%02x%02x" % c.color} for c in self.classes]

    # -------------------------------------------------------------------------------- request
    def _panel(self, m: LoadedModel, pred: np.ndarray, logits_shape: list, box, top_k: int, inf_ms: float) -> dict:
        top, left, h, w = box
        crop = pred[top:top + h, left:left + w]
        counts = np.bincount(crop.ravel(), minlength=m.num_classes)
        total = int(counts.sum())
        present = [{"class_id": int(i), "label": self.classes[i].label, "role": self.classes[i].role,
                    "color": "#%02x%02x%02x" % self.classes[i].color, "pixels": int(counts[i]),
                    "share": counts[i] / total} for i in np.argsort(-counts, kind="stable") if counts[i] > 0]
        diseases = [p for p in present if p["role"] == "disease"]
        background = next((p for p in present if p["class_id"] == self.data["background_index"]), None)
        return {"slot": m.slot, "label": m.label, "description": m.description, "file": m.path.name,
                "sha256": m.sha256, "sha256_short": common.short_sha(m.sha256), "engine": m.engine,
                "kind": m.kind, "stage": m.stage, "stand_in": m.stand_in,
                "latency_ms": {"inference": round(inf_ms, 2)},
                "output": {"logits_shape": logits_shape, "mask_shape_hw": [h, w],
                           "class_min": int(crop.min()), "class_max": int(crop.max()), "pixels": total},
                "classes": {"top": diseases[:top_k], "background": background,
                            "present_count": len(present), "present": present},
                "images": {"overlay_png": png_data_url(Image.fromarray(self._lut[crop], "RGBA")),
                           "mask_png": png_data_url(Image.fromarray(crop, "L"))}}

    def segment(self, data: bytes, top_k: int = 5, compare: str | None = None) -> dict:
        compare = None if compare in (None, "", "none") else str(compare)
        if compare is not None and (compare == "e6" or compare not in self.models):
            raise DemoError(f"compare={compare!r} is not a loaded comparison model; options: "
                            f"{['none'] + [o['slot'] for o in self.compare_options()]}")
        t_start = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as e:  # noqa: BLE001 -- any decode failure is a bad upload
            raise DemoError(f"could not decode the image ({type(e).__name__}): send a JPEG or PNG") from e
        fmt, mode, original_size = img.format, img.mode, img.size
        t_dec = time.perf_counter()
        pre = evaluator_preprocess(img)
        t_pre = time.perf_counter()
        runs = [self.model] + ([self.models[compare]] if compare else [])
        outs = []
        with self._lock, torch.inference_mode():
            t_lock = time.perf_counter()                       # a queue wait before this is in no part, as in build 2
            for m in runs:                                     # the same input tensor for every model
                t0 = time.perf_counter()
                logits = m.module(pre.tensor)
                t1 = time.perf_counter()
                outs.append((m, logits.argmax(dim=1)[0].to(torch.uint8).numpy(), list(logits.shape),
                             (t1 - t0) * 1e3))
                del logits
        panels = [self._panel(m, pred, shape, pre.box, top_k, ms) for m, pred, shape, ms in outs]
        top, left, h, w = pre.box
        photo = Image.fromarray(pre.canvas[top:top + h, left:left + w], "RGB")
        e6 = panels[0]
        t_end = time.perf_counter()
        latency = {"decode": round((t_dec - t_start) * 1e3, 2),
                   "preprocess": round((t_pre - t_dec) * 1e3, 2),
                   "inference": e6["latency_ms"]["inference"],
                   "postprocess": round((t_end - t_lock) * 1e3 - sum(o[3] for o in outs), 2),
                   "total": round((t_end - t_start) * 1e3, 2)}
        if compare:
            latency["inference_compare"] = panels[1]["latency_ms"]["inference"]
        return {
            "latency_ms": latency,
            "threads": self.threads,
            "input": {"format": fmt, "mode": mode, "original_size_wh": list(original_size),
                      "model_input_shape": list(pre.tensor.shape), "protocol": PROTOCOL_ID,
                      "photo_box_on_canvas": {"top": top, "left": left, "height": h, "width": w},
                      "input_tensor_sha256": hashlib.sha256(pre.tensor.numpy().tobytes()).hexdigest()},
            "output": e6["output"],
            "classes": e6["classes"],
            "images": {"photo_jpeg": jpeg_data_url(photo),        # what the models saw (display only)
                       "overlay_png": e6["images"]["overlay_png"],  # RGBA, transparent where class 0
                       "mask_png": e6["images"]["mask_png"]},       # 8-bit class ids 0..115
            "model": {"file": self.model.path.name, "sha256": self.model.sha256, "stand_in": self.model.stand_in,
                      "stage": self.model.stage, "engine": self.model.engine},
            "compare": compare,
            "panels": panels,
        }


def default_threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))
