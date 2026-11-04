"""Model loading and one segmentation request, on CPU, for the demo server.

The model file is loaded by the evaluator's own INT8 loader (src/eval/model_loading.py
load_int8_torchscript) under the QNNPACK engine it requires (select_int8_backend). The identity the
loader is asked to verify (stage, method, artifact role) is read from the file's own embedded
plantseg_int8.json, so the artifact of record swaps in by changing demo/config.py MODEL_PATH only; the
loader still enforces the schema, 116 classes and the qnnpack engine.

Prediction = argmax over the 116 logits on the 512x512 canvas (the canvas protocol's prediction),
cropped to the photo's region; shares are over that region only, never over the padding.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from demo import common
from demo.server.preprocess import CANVAS, PROTOCOL_ID, evaluator_preprocess

META_NAME = "plantseg_int8.json"       # = src.eval.model_loading.INT8_TORCHSCRIPT_META (smoke-checked)


class DemoError(ValueError):
    """A request the demo refuses (bad upload); the message is shown to the user."""


def read_embedded_meta(path: Path) -> dict:
    """plantseg_int8.json from the TorchScript zip, without loading the module."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith("/extra/" + META_NAME)]
        if len(names) != 1:
            raise RuntimeError(f"{path.name} carries {len(names)} {META_NAME} entries; expected exactly 1 "
                               "(only TorchScript files written by src/quant/ptq.py are accepted)")
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


def load_model(path: Path) -> LoadedModel:
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
    with torch.inference_mode():
        y = module(torch.zeros(1, 3, CANVAS, CANVAS))          # warm-up and shape check
    if tuple(y.shape[2:]) != (CANVAS, CANVAS) or y.dim() != 4:
        raise RuntimeError(f"unexpected model output shape {tuple(y.shape)}")
    return LoadedModel(module=module, meta=meta, path=path, sha256=common.sha256_file(path),
                       size_bytes=path.stat().st_size, engine=engine, num_classes=int(y.shape[1]),
                       load_ms=(time.perf_counter() - t0) * 1000.0)


class Segmenter:
    """One loaded model + the repository's class list. Requests are serialised (one CPU model)."""

    def __init__(self, model_path: Path, threads: int | None = None):
        if threads:
            torch.set_num_threads(int(threads))
        self.threads = torch.get_num_threads()
        self.classes, self.class_provenance = common.load_classes()
        self.data = common.data_facts()
        self.model = load_model(model_path)
        n = self.model.num_classes
        if not (n == len(self.classes) == self.data["num_classes"]):
            raise RuntimeError(f"class count mismatch: model outputs {n}, class map has "
                               f"{len(self.classes)}, configs/data.py says {self.data['num_classes']}")
        self._lut = np.zeros((n, 4), dtype=np.uint8)            # class -> RGBA overlay colour
        for c in self.classes:
            if c.role != "background_or_non_disease":
                self._lut[c.class_id] = (*c.color, 255)
        self._lock = threading.Lock()
        common.assert_no_stats()

    # -------------------------------------------------------------------------------- metadata
    def model_info(self) -> dict:
        m = self.model
        return {"file": m.path.name, "sha256": m.sha256, "size_bytes": m.size_bytes,
                "format": "TorchScript (torch.jit.trace of the converted eager INT8 module)",
                "engine": m.engine, "num_classes": m.num_classes, "embedded_meta": m.meta,
                "stand_in": bool(m.meta.get("stand_in", False)), "load_ms": round(m.load_ms, 1),
                "torch": torch.__version__, "threads": self.threads}

    def classes_info(self) -> list[dict]:
        return [{"class_id": c.class_id, "name": c.name, "label": c.label, "role": c.role,
                 "color": "#%02x%02x%02x" % c.color} for c in self.classes]

    # -------------------------------------------------------------------------------- request
    def segment(self, data: bytes, top_k: int = 5) -> dict:
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
        with self._lock, torch.inference_mode():
            t_inf0 = time.perf_counter()
            logits = self.model.module(pre.tensor)
            t_inf1 = time.perf_counter()
        pred = logits.argmax(dim=1)[0].to(torch.uint8).numpy()   # [512, 512] class ids 0..115
        top, left, h, w = pre.box
        crop = pred[top:top + h, left:left + w]
        counts = np.bincount(crop.ravel(), minlength=self.model.num_classes)
        total = int(counts.sum())
        present = [{"class_id": int(i), "label": self.classes[i].label, "role": self.classes[i].role,
                    "color": "#%02x%02x%02x" % self.classes[i].color, "pixels": int(counts[i]),
                    "share": counts[i] / total} for i in np.argsort(-counts, kind="stable") if counts[i] > 0]
        diseases = [p for p in present if p["role"] == "disease"]
        background = next((p for p in present if p["class_id"] == self.data["background_index"]), None)
        photo = Image.fromarray(pre.canvas[top:top + h, left:left + w], "RGB")
        overlay = Image.fromarray(self._lut[crop], "RGBA")
        mask = Image.fromarray(crop, "L")
        images = {"photo_jpeg": jpeg_data_url(photo),        # what the model saw (display only)
                  "overlay_png": png_data_url(overlay),      # RGBA, transparent where class 0
                  "mask_png": png_data_url(mask)}            # 8-bit class ids 0..115
        t_end = time.perf_counter()
        return {
            "latency_ms": {"decode": round((t_dec - t_start) * 1e3, 2),
                           "preprocess": round((t_pre - t_dec) * 1e3, 2),
                           "inference": round((t_inf1 - t_inf0) * 1e3, 2),
                           "postprocess": round((t_end - t_inf1) * 1e3, 2),
                           "total": round((t_end - t_start) * 1e3, 2)},
            "threads": self.threads,
            "input": {"format": fmt, "mode": mode, "original_size_wh": list(original_size),
                      "model_input_shape": list(pre.tensor.shape), "protocol": PROTOCOL_ID,
                      "photo_box_on_canvas": {"top": top, "left": left, "height": h, "width": w},
                      "input_tensor_sha256": hashlib.sha256(pre.tensor.numpy().tobytes()).hexdigest()},
            "output": {"logits_shape": list(logits.shape), "mask_shape_hw": [h, w],
                       "class_min": int(crop.min()), "class_max": int(crop.max()),
                       "pixels": total},
            "classes": {"top": diseases[:top_k], "background": background,
                        "present_count": len(present), "present": present},
            "images": images,
            "model": {"file": self.model.path.name, "sha256": self.model.sha256,
                      "stand_in": bool(self.model.meta.get("stand_in", False)),
                      "stage": self.model.meta.get("stage"), "engine": self.model.engine},
        }


def default_threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))
