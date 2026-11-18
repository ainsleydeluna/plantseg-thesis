# PlantSeg demo (lanes DEMO-1 and DEMO-3): research prototype, not for agronomic decisions

The model was trained on PlantSeg (CC BY-NC 4.0: non-commercial use only, with attribution). The full
lane report, with measurements, the on-device decision memo, hosting options and the swap steps, is
[docs/lane_reports/demo.md](../docs/lane_reports/demo.md); the model registry, the FP32 path and the
comparison switch are in [docs/lane_reports/demo3.md](../docs/lane_reports/demo3.md).

| what | command (Python 3.11, Linux, macOS or WSL2) |
|---|---|
| install | `pip install -r demo/requirements-demo.txt` |
| run the server | `PLANTSEG_DEMO_MODEL=/path/to/E6.pt python -I demo/run_demo.py --no-build`, then open `http://<this-machine>:8000` on the phone; with no E6 file, `python -I demo/run_demo.py` builds the stand-in if needed, refuses it and prints the `PLANTSEG_DEMO_E6_SHA256=…` setting that serves it |
| smokes | `python -I demo/smoke_demo.py [--model PATH]` |
| FP32 export of E3 (after AM-30) | `python -I demo/model/export_fp32_torchscript.py E3.pt --source-sha256 HEX --stage E3 --seed 42 --out OUT.pt --e6 E6.pt` |
| its parity check (VAL images) | `python -I demo/model/parity_fp32.py --model OUT.pt --checkpoint E3.pt --source-sha256 HEX --images DIR --n 32` |
| on-device (in-browser) files | `python -I demo/ondevice/export_onnx.py` and `python -I demo/ondevice/fetch_ort_web.py`, then `/ondevice/` |
| static on-device bundle (no PyTorch needed to serve it) | `python -I demo/ondevice/make_static_bundle.py OUT`, then `python -m http.server 8000 --directory OUT` |
| lite-interpreter (.ptl) check | `python -I demo/ondevice/lite_interpreter.py` |
| Docker | `python -I demo/docker/build_context.py ../ctx --model /path/to/E6.pt --with-ondevice`, then `docker build -t plantseg-demo ../ctx` and `docker run --env-file ../ctx/demo.env ...` |

- **The model registry** (`demo/config.py`): slots `e6` (required), `e3` and `shown`. Each has a file
  (`PLANTSEG_DEMO_MODEL`, `PLANTSEG_DEMO_E3_MODEL`, `PLANTSEG_DEMO_SHOWN_MODEL`, or its marked line) and an
  expected sha256 (`PLANTSEG_DEMO_E6_SHA256`, `..._E3_SHA256`, `..._SHOWN_SHA256`, or its marked line). A
  slot loads only when the file's sha256 equals it; a slot without one is hidden.
- **Swapping in the artifact of record:** set e6's file (`MODEL_PATH` or `PLANTSEG_DEMO_MODEL`); its
  expected sha256 is E6's.
- **Windows:** the official Windows torch 2.1 wheels lack QNNPACK, so run the server in WSL2 or
  Docker, or serve the static bundle (report section 2).
- **Generated files** (`demo/models/`, `demo/ondevice/vendor/`) are never committed.
