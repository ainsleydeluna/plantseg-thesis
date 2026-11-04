"""FastAPI app for the mobile web demo (research prototype; CPU only; no data is stored).

Routes
  GET  /                 the phone page (demo/web/index.html)
  GET  /api/info         model file name, sha256 and embedded identity; the repository's class list;
                         the notices; the preprocessing protocol
  POST /api/segment      multipart field "file": one image -> overlay PNG, class shares, top classes,
                         latency (ms), model name + sha256
  GET  /healthz          liveness
  GET  /ondevice/        the in-browser (ONNX Runtime Web) page; /ondevice/models/<name>.onnx and
                         /ondevice/vendor/... when those generated files exist
Uploads are decoded in memory and discarded after the response.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from demo import common, config
from demo.server.inference import DemoError, Segmenter, default_threads
from demo.server.preprocess import PROTOCOL_ID

WEB_DIR = config.DEMO_DIR / "web"
ONDEVICE_DIR = config.DEMO_DIR / "ondevice"
NOTICE_PROTOTYPE = "Research prototype; not for agronomic decisions."

mimetypes.add_type("text/javascript", ".mjs")      # ONNX Runtime Web loads .mjs workers
mimetypes.add_type("application/wasm", ".wasm")    # needed for WebAssembly streaming compile


def licence_notice(data: dict) -> str:
    return (f"The model was trained on the PlantSeg dataset (DOI {data['doi']}), licensed "
            f"{data['license']}: non-commercial use only, with attribution.")


def create_app(model_path: Path | None = None, threads: int | None = None) -> FastAPI:
    seg = Segmenter(config.resolve_model_path(model_path), threads or default_threads())
    app = FastAPI(title="PlantSeg lesion segmentation demo (research prototype)",
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.segmenter = seg

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model_sha256": seg.model.sha256}

    @app.get("/api/info")
    def info():
        return {"model": seg.model_info(),
                "classes": seg.classes_info(),
                "class_map": seg.class_provenance,
                "data": seg.data,
                "notices": {"prototype": NOTICE_PROTOTYPE, "licence": licence_notice(seg.data)},
                "preprocessing": {"protocol": PROTOCOL_ID,
                                  "functions": ["src/data/transforms.py core_preprocess",
                                                "src/data/transforms.py finalize"]},
                "src_stats_imported": common.stats_modules_loaded()}

    @app.post("/api/segment")
    async def segment(file: UploadFile = File(...)):
        data = await file.read(config.MAX_UPLOAD_BYTES + 1)
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"image larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            result = await run_in_threadpool(seg.segment, data, config.TOP_K)
        except DemoError as e:
            raise HTTPException(400, str(e)) from e
        result["notices"] = {"prototype": NOTICE_PROTOTYPE, "licence": licence_notice(seg.data)}
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/ondevice/")
    def ondevice():
        return FileResponse(ONDEVICE_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/ondevice/models/{name}")
    def onnx_model(name: str):
        allowed = {config.ONNX_FP32_PATH.name: config.ONNX_FP32_PATH,
                   config.ONNX_INT8_PATH.name: config.ONNX_INT8_PATH}
        p = allowed.get(name)
        if p is None or not p.is_file():
            raise HTTPException(404, f"{name} not found; run python3 -I demo/ondevice/export_onnx.py")
        return FileResponse(p, media_type="application/octet-stream")

    vendor = ONDEVICE_DIR / "vendor"
    if vendor.is_dir():
        app.mount("/ondevice/vendor", StaticFiles(directory=vendor), name="vendor")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
