# PlantSeg demo (lane DEMO-1): research prototype, not for agronomic decisions

The model was trained on PlantSeg (CC BY-NC 4.0: non-commercial use only, with attribution). The full
lane report, with measurements, the on-device decision memo, hosting options and the swap steps, is
[docs/lane_reports/demo.md](../docs/lane_reports/demo.md).

| what | command (Python 3.11, Linux, macOS or WSL2) |
|---|---|
| install | `pip install -r demo/requirements-demo.txt` |
| run the server (builds the stand-in on first run) | `python -I demo/run_demo.py`, then open `http://<this-machine>:8000` on the phone |
| smokes | `python -I demo/smoke_demo.py [--model PATH]` |
| on-device (in-browser) files | `python -I demo/ondevice/export_onnx.py` and `python -I demo/ondevice/fetch_ort_web.py`, then `/ondevice/` |
| static on-device bundle (no PyTorch needed to serve it) | `python -I demo/ondevice/make_static_bundle.py OUT`, then `python -m http.server 8000 --directory OUT` |
| lite-interpreter (.ptl) check | `python -I demo/ondevice/lite_interpreter.py` |
| Docker | `python -I demo/docker/build_context.py ../ctx --with-ondevice`, then `docker build -t plantseg-demo ../ctx` |

- **Swapping in the artifact of record:** change the one marked line in `demo/config.py`
  (`MODEL_PATH`), or set `PLANTSEG_DEMO_MODEL`.
- **Windows:** the official Windows torch 2.1 wheels lack QNNPACK, so run the server in WSL2 or
  Docker, or serve the static bundle (report section 2).
- **Generated files** (`demo/models/`, `demo/ondevice/vendor/`) are never committed.
