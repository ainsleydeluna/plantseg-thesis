# Lane DEMO — mobile web demo and on-device feasibility: lane report

Session DEMO-1 (Claude Code cloud session), under the brief `demo1_brief_20261103.md` of 3 Nov 2026.
Branch `lane/demo`, created at master `fe93a24`; new files only, under `demo/` plus this report.
Labels: **MEASURED** (run in this session, numbers from the output), **DOCUMENTED** (the page or file
named was fetched in this session), **INFERRED** (reasoning or prior knowledge, not checked here).

## 0. Status

- **Done:** the stand-in model (task 1), the server and phone page with a Dockerfile and hosting list
  (task 2), the on-device feasibility memo (task 3), the smokes (task 4) and this report (task 5).
  Every smoke passes (section 8).
- **Time box missed.** The brief's box was 10:00 +08:00 on 4 Nov. The session stopped after about
  00:10 +08:00 on 4 Nov. No error and no account-limit message reached the session, so the cause is not
  visible to me. Work resumed at 12:24 +08:00 on your "Try again". The worker process restarted once
  (about 13:12 +08:00) and resumed on "continue". The work was pushed in small commits, so nothing was
  lost.
- **The predictions are meaningless until the swap.** The stand-in has random weights. The page says
  so in a red banner, which disappears when the artifact of record is swapped in (section 7).

## 1. Base, commits and the scoped diff

- Base: `fe93a24` = origin/master (MEASURED with `git ls-remote` at the start). `lane/demo` did not
  exist on the remote; it was created there by the first push. Every push was a fast-forward.
- Clone: the brief asks for a clone without `docs/reference/` and test-named paths. The repository's
  L-PROT hook refuses `git clone` and `git sparse-checkout` (DL-37 records the latter), so the sparse
  partial clone was built only from forms the hook allows:
  - `git init`, `git remote add`, `git fetch --filter=blob:none`, `update-ref` and `symbolic-ref`;
  - then `git checkout lane/demo -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'`.

  As a result, the protected file's and the test-named files' blobs were never fetched (MEASURED: the
  index holds 0 test-named entries, and `docs/` has no `reference/`).

  The excluded paths are absent from the index. An unscoped `git status` would therefore list them as
  deletions, and it was never run. Every commit used `git commit --only -- <paths>`, so no deletion can
  enter a commit. After each commit, the number of changed paths since `fe93a24` (every path except
  the protected file) equalled the number of new files (MEASURED).
- Commits (author Claude, no trailers, each `git commit --only` with explicit paths):

| | commit | subject |
|---|---|---|
| C1 | e6cc9ad | stand-in INT8 QNNPACK TorchScript builder and the demo's shared base |
| C2 | 5905950 | FastAPI server, the phone page and the one-command runner |
| C3 | 08aed77 | demo smokes |
| C4 | a955ee7 | on-device feasibility scripts (ONNX FP32 and INT8 QDQ exports, lite interpreter) |
| C5 | ecb1951 | in-browser (ONNX Runtime Web) page, its verified runtime fetch and the headless check |
| C6 | f0900c8 | Dockerfile with an allowlisted build context, and a static on-device bundle |
| C7 | 8e16949 | panel-view screenshots and the host benchmark |
| C8 | dcf55be | swap-path tools for the artifact of record |
| C9 | (this commit) | this report and demo/README.md |

- Scoped diff, `git diff --stat fe93a24 HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'`
  at C8: 31 files, every one added (status `A`) under `demo/`, 2,718 insertions, 0 deletions
  (MEASURED). C9 adds 2 more. No existing file was modified, renamed or deleted.
- Generated files are never committed (`demo/.gitignore`: `models/`, `ondevice/vendor/`):
  - the stand-in `.pt` and its FP32 state;
  - the ONNX and `.ptl` files;
  - the vendored ONNX Runtime Web.

  Each is rebuilt by one script (section 2).

## 2. How to run it

You need Python 3.11, because torch 2.1 has no wheels for 3.12+. The pins are in
`demo/requirements-demo.txt`: `torch==2.1.0+cpu`, `torchvision==0.16.0+cpu` (the repository's pins,
as CPU builds), `numpy==1.26.4`, `pillow==12.3.0`, `fastapi==0.115.12`, `uvicorn==0.34.2`,
`python-multipart==0.0.20`, plus `onnx==1.17.0` and `onnxruntime==1.20.1` for the exports.

**Linux or macOS (or WSL2 on Windows, see below):**

```
python3.11 -m venv .venv-demo
.venv-demo/bin/pip install -r demo/requirements-demo.txt
.venv-demo/bin/python -I demo/run_demo.py
```

- Open `http://localhost:8000`.
- On the iPhone, on the same Wi-Fi or the laptop's hotspot, open the printed `http://<laptop-IP>:8000`.
- The first run builds the stand-in in about 15 s (MEASURED: 14.4 s).
- `--host 127.0.0.1` keeps it local, `--threads 2` sets the CPU threads, and `--model PATH` overrides
  the model for one run.
- Smokes: `.venv-demo/bin/python -I demo/smoke_demo.py`.

**Windows laptops: the server cannot load the artifact natively.**

The official Windows wheel `torch-2.1.0+cpu-cp311-cp311-win_amd64.whl` (sha256 `c2ec46e3…bd2b9`) has no
QNNPACK compiled in (MEASURED by inspecting its `torch_cpu.dll`, without running it):

- Its compiled build flags read `/DWIN32 /D_WINDOWS … -DUSE_FBGEMM -DUSE_XNNPACK …`.
- `USE_PYTORCH_QNNPACK`, `init_qnnpack.cpp` and QNNPACK's error strings occur 0 times.
- The Linux wheel's flags carry `-DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK`.

The evaluator's loader requires QNNPACK and refuses other engines by design
(`src/eval/model_loading.py:132-150`), so on Windows it will stop at start-up (INFERRED from the flags;
not run on Windows). Three routes work instead:

1. **WSL2 (Ubuntu).** Run the Linux commands inside WSL. Ubuntu 24.04 ships Python 3.12, so install
   3.11 first (deadsnakes, or `uv python install 3.11`) (INFERRED).
   - WSL2 uses NAT by default; "mirrored" mode, or `netsh interface portproxy`, exposes the port to
     the LAN; bind to `0.0.0.0` (DOCUMENTED: MicrosoftDocs/WSL `WSL/networking.md`).
   - Allow port 8000 in Windows Firewall (INFERRED).
2. **Docker Desktop (WSL 2 backend).**
   ```
   python -I demo/docker/build_context.py ..\plantseg-demo-context --with-ondevice
   docker build -t plantseg-demo ..\plantseg-demo-context
   docker run --rm -p 8000:8000 plantseg-demo
   ```
   - Docker Desktop needs Windows 10 22H2 or Windows 11 23H2 (Enterprise, Pro or Education) and WSL 2.
   - It is free except for commercial use in large enterprises (DOCUMENTED: docker/docs
     `windows-install.md`).
   - The build context script runs where the repository and model are (WSL or Linux), and the context
     is never the repository root (section 6).
3. **The static on-device bundle** (no PyTorch on the laptop at all).
   - Make it once on Linux or WSL: `python -I demo/ondevice/make_static_bundle.py OUT`.
   - Then, on Windows: `python -m http.server 8000 --directory OUT`.
   - The model runs on the phone (MEASURED served by `http.server`, section 5).

**On-device (in-browser) page:** run these once, with internet, then open `/ondevice/` on the running
server, or use the static bundle.

```
python -I demo/ondevice/export_onnx.py
python -I demo/ondevice/fetch_ort_web.py
```

`fetch_ort_web.py` vendors `onnxruntime-web@1.20.1` after checking the npm registry's sha512 integrity.

**What was not run verbatim here.** I installed the same pins with uv in two steps: PyPI packages,
then torch and torchvision from the PyTorch CPU index. The auto-mode classifier refused my first,
mixed-index install, so `pip install -r demo/requirements-demo.txt` was not run here. The file uses
the repository's own `--extra-index-url` pattern from `requirements-e1.txt` (INFERRED to resolve).

## 3. What the panel will see

These screenshots come from headless Chromium 153 emulating the iPhone 15 Plus viewport (430×739 CSS
px, DPR 3; Playwright's device descriptor) with a self-made synthetic leaf image (MEASURED). Chromium is
not Safari: they show layout, not Safari compatibility (section 9).

- [`demo/screenshots/01_server_page.jpg`](../../demo/screenshots/01_server_page.jpg) shows the start page:
  - the notice "Research prototype; not for agronomic decisions.";
  - the stand-in banner;
  - **Take photo** (`<input type=file accept="image/*" capture="environment">`) and **Upload image**;
  - the model card: file name, size, sha256, embedded identity, runtime (`torch 2.1.0+cpu`, engine
    `qnnpack`) and input protocol `core_preprocess/1.0.0`;
  - the 116-class list, with names from `configs/plantseg_class_map.json` (class 0 is unnamed there,
    so the page shows its role, "background / non-disease");
  - the licence notice built from `configs/data.py`: "…PlantSeg dataset (DOI 10.5281/zenodo.17719108),
    licensed CC BY-NC 4.0: non-commercial use only, with attribution."
- [`demo/screenshots/02_server_result.jpg`](../../demo/screenshots/02_server_result.jpg) shows a result:
  - the photo as the model saw it (long side 512), with the colour overlay (PNG, transparent for
    class 0) and an opacity slider;
  - the top classes by pixel share, as a legend with colours;
  - latency: model inference in ms (156 ms here), server total with its parts, and the round trip.

  With the stand-in the overlay is one class everywhere ("tomato leaf mold", 100%). That is what random
  weights do, and the red banner says so.
- [`demo/screenshots/03_ondevice_result.jpg`](../../demo/screenshots/03_ondevice_result.jpg) is the
  in-browser page:
  - it uses the same layout;
  - the ONNX file's sha256 is verified in the browser;
  - the runtime line reads `onnxruntime-web 1.20.1, wasm, threads 1`;
  - inference took 644 ms on its first run, which includes warm-up.

A suggested description for the manuscript, true as of this lane:

> "The demo runs the INT8 QNNPACK model of record on a CPU server (FastAPI) and is used from a phone's
> browser, with the evaluator's preprocessing. An on-device variant runs the same model, exported to
> ONNX (QDQ), inside the phone's browser with ONNX Runtime Web."

Quote phone latencies only after measuring them on the iPhone 15 Plus (section 9).

## 4. The stand-in model (task 1)

- Built by `demo/model/build_stand_in.py` exactly the way the repository builds the student:
  - `build_student(pretrained=False)` (`src/models/student.py:235`): random init after
    `torch.manual_seed(20261103)`, no download;
  - `src/quant/ptq.py quantize_qnnpack`: QNNPACK engine, `fuse`, the registered PTQ qconfig from
    `src/quant/qconfig.py`, batch-1 calibration, `convert`. Calibration used 8 self-made synthetic
    images through the evaluator's preprocessing, because no data exists here;
  - `src/quant/ptq.py torchscript_bytes`: `torch.jit.trace` at 1×3×512×512 with `plantseg_int8.json`
    embedded, the same format as the artifact of record.

  It was then loaded back through the evaluator's `load_int8_torchscript` (MEASURED).
- The file (MEASURED):
  - `stand_in_int8_qnnpack.pt`, **3,821,061 bytes**, sha256
    `a5c699b5e79a9dedbbafc6f59f46764aa8c9c5f27a0d259f9ccc29b28160fc32`;
  - two builds in fresh processes at the same checkout path gave identical bytes;
  - the bytes depend on the checkout path, because TorchScript debug records keep source paths
    (`src/quant/ptq.py:102-107`), so another machine's stand-in will hash differently (INFERRED);
  - embedded identity: `stage DEMO_STAND_IN`, `quantization ptq`, `engine qnnpack`, `artifact_role
    demo_stand_in`, `stand_in: true`.
- Outputs [1, 116, 512, 512] float32. The class count is 116 everywhere: model output channels =
  `configs/data.py num_classes` = 116 class-map entries, with ids 0..115 in order (MEASURED, smoke S2).
- Quantized op set:
  - `src/quant/ptq.py graph_census` reports ok: one `quantize_per_tensor` fed by the input, one
    `dequantize` feeding only the final upsample, no float-only ops and 2 upsamples (MEASURED).
  - Op counts: `quantized::conv2d` 45, `quantized::conv2d_relu` 20, `quantized::hardswish` 19,
    `quantized::add` 11, `quantized::mul` 9, `aten::adaptive_avg_pool2d` 9, `aten::hardsigmoid` 8,
    `aten::sigmoid` 1, `aten::upsample_bilinear2d` 2.
  - Root operators (`torch.jit.export_opnames`): `quantized::{add, conv2d.new, conv2d_prepack,
    conv2d_relu.new, hardswish, mul}`, `aten::{adaptive_avg_pool2d, dequantize.self, hardsigmoid,
    quantize_per_tensor, sigmoid, size.int, upsample_bilinear2d.vec, Int.Tensor, eq.str}`,
    `prim::TupleUnpack`.
- Parity (MEASURED with `src/quant/ptq.py output_parity` on its synthetic shapes, which include a batch
  of 2 and 384×512): the converted module and the TorchScript file agree bitwise, 4/4.
- One benign warning appears at convert: "must run observer before calling calculate_qparams". It
  comes from the 5 unused `skip_add` observers of torchvision's non-residual blocks (`features.2, 4, 7,
  11, 13`; MEASURED). A real PTQ run would show the same warning (INFERRED).

## 5. Decision memo: on-device feasibility (task 3)

### Recommendation

1. **The defense demo: the server-backed mobile web app.**
   - It runs the INT8 QNNPACK artifact of record itself.
   - It applies the evaluator's own preprocessing, byte for byte (smoke S5).
   - It needs no Mac and no Apple account.
   - It runs on a Linux host: the team laptop via WSL2 or Docker Desktop, or a 2 GB Linux VM.
2. **"Phone deployment": in-browser inference with ONNX Runtime Web (WASM) in Safari.**
   - The model is the INT8 ONNX exported from the TorchScript artifact itself.
   - It is not QNNPACK running on the phone. It is the same quantized graph as QDQ, executed by ONNX
     Runtime, with outputs within one quantization step on synthetic inputs.
   - No Mac and no app are needed.
   - Measure it on the iPhone 15 Plus before quoting any phone latency.
3. **Native iOS (LibTorch-Lite):** only if someone can use a Mac with Xcode. The artifact converts
   bitwise, but PyTorch Mobile is no longer actively supported.
4. **ExecuTorch:** not for this artifact. It would be a different INT8 model, and it also needs a Mac.

### (a) In-browser inference: ONNX Runtime Web, WASM or WebGPU

**Exports**, from `demo/ondevice/export_onnx.py`, all MEASURED:

| model | source | file | sha256 | parity on CPU (onnxruntime 1.20.1) |
|---|---|---|---|---|
| FP32 | `build_student` + the stand-in's FP32 weights, `torch.onnx.export` opset 17, static 1×3×512×512 | 11,678,309 B, 204 nodes | `64110c1c…a10c115` | max \|Δlogit\| 1.3e-7 vs PyTorch FP32 |
| INT8 QDQ | **the INT8 TorchScript file itself**, `torch.onnx.export` opset 17 | 3,335,960 B, 1,543 nodes (QuantizeLinear 124, DequantizeLinear 274, Conv 65) | `9efcb76c…265fd1f6` | max \|Δlogit\| 0.00143 = **0.94 output quantization steps** (step 0.0015268), mean ≤ 1.4e-7, vs the INT8 TorchScript |

- Both graphs return `logits` and an in-graph uint8 argmax `mask`, so the phone fetches 0.26 MB
  instead of 121 MB.
- The parity inputs were 3 random-normal tensors and 3 synthetic images through the evaluator's
  preprocessing.
- With random weights the argmax is one class everywhere, so argmax agreement (100%) tells you nothing
  here. The logit differences are the informative measure.
- After export, the 65 all-zero int32 bias zero points are dropped. The ONNX operator treats a missing
  int32 zero point as 0, and CPU parity is unchanged (MEASURED).
- For the stand-in, the INT8 TorchScript differs from its own FP32 model by at most 0.002–0.008 in the
  logits (MEASURED). That is quantization error with random weights, not a property of the real model.

**The browser page** (`demo/ondevice/index.html`) is a JavaScript port of
`core_preprocess`/`finalize`:

- Pillow 12.3.0's BILINEAR resample (`src/libImaging/Resample.c`, DOCUMENTED from
  `raw.githubusercontent.com/python-pillow/Pillow/12.3.0/...`);
- Python's round-half-even;
- the centred pad;
- float32 normalisation.

**Headless Chromium 153, WASM, 1 thread, on this 2.1 GHz Xeon** (MEASURED with
`demo/tools/browser_check.cjs`):

- **Input:** byte-identical input tensors vs the evaluator's Python function on **7/7** fixtures:
  5 PNG (including 1024×765, where Python's half-to-even round matters, and 512×512, no resize) and
  2 JPEG (Chromium's JPEG decode matched Pillow's on these).
- **Output:** masks equal on 7/7. Logits on a stride-4 subsample (1,900,544 values): INT8 max
  \|Δ\| 0.00134, mean 4e-8 (vs the INT8 TorchScript); FP32 max 1.2e-7.
- **Model integrity:** the browser computed the model's sha256 and it matched.
- **Speed:** inference about 370–920 ms (INT8) and 280–560 ms (FP32) per image, and the session
  starts in 0.5–2.7 s. On this CPU, INT8 QDQ under WASM is not faster than FP32.
- **Static bundle:** served by stdlib `python -m http.server` (no PyTorch, no FastAPI), the same
  results hold, 7/7.

**WebGPU** (MEASURED in headless Chromium on SwiftShader, a software GPU, so its timings, 10–14 s per
image, mean nothing):

- **FP32** runs correctly: max \|Δ\| 1.2e-7 and masks 7/7.
- **INT8 QDQ** first failed with `[WebGPU] Kernel "[DequantizeLinear]" failed. … In the case of
  dequantizing int32 there is no zero point`. After the zero points were dropped it runs, but its
  output is **wrong**: max \|Δ\| 0.389, mean 0.166, masks 0/7.
- So the WebGPU route is **FP32 only** with onnxruntime-web 1.20.1. Whether a later release fixes the
  INT8 kernels was not tested.

**Safari constraints:**

- WebGPU is in Safari 26, and iOS mirrors it (DOCUMENTED: MDN browser-compat-data `api/GPU.json`). It
  exists only in a secure context (DOCUMENTED: MDN `webgpu_api/index.md`), so it needs HTTPS. A
  laptop on the LAN serving plain `http://` gets no WebGPU.
- WASM multi-threading needs `crossOriginIsolated` (DOCUMENTED: ONNX Runtime
  `env-flags-and-session-options.md`). That needs a secure context plus COOP/COEP (DOCUMENTED: MDN
  `SharedArrayBuffer`). So over LAN HTTP the phone runs single-threaded WASM.
- `<input capture>` is supported since iOS Safari 10 (DOCUMENTED: MDN browser-compat-data
  `html/elements/input.json`).
- iOS Safari caps canvas area at about 16.7 MP, so the page pre-scales larger photos and flags them
  "not byte-identical" (INFERRED limit; the branch was not exercised on a device).
- Safari converts HEIC to JPEG for `accept="image/*"`, and the server page re-encodes HEIC or large
  files on the phone anyway (INFERRED).
- Safari's JPEG decoder and colour management may differ from Pillow's, so byte equality is measured
  only for Chromium (INFERRED risk).

### (b) Native iOS: LibTorch-Lite or ExecuTorch

**Lite interpreter**, all MEASURED with `demo/ondevice/lite_interpreter.py`:

- The QNNPACK TorchScript artifact *can* be saved for the lite interpreter.
- `_save_for_lite_interpreter` gives 3,976,417 bytes with bytecode v8. Reloaded with
  `_load_for_lite_interpreter`, its outputs are bitwise equal to the TorchScript's, 4/4.
- After `optimize_for_mobile` it is 3,395,177 bytes, also bitwise equal, 4/4.
- Backports to bytecode v7, v6, v5 and v4 all succeed and stay bitwise equal.
- The operator list the iOS runtime must contain is recorded in `lite_interpreter_facts.json`; it is
  the root-op list in section 4.

**LibTorch-Lite (PyTorch Mobile):**

- CocoaPods podspecs exist for `LibTorch-Lite` 2.1.0 (iOS ≥ 12.0, `libtorch_lite_ios_2.1.0.zip`) and
  1.13.0.1. Version strings 1.13.1, 2.0.0, 2.0.1 and 2.2.0 do not exist (DOCUMENTED: CocoaPods/Specs on
  GitHub, `Specs/c/c/3/LibTorch-Lite/<version>/LibTorch-Lite.podspec.json`). The full version listing
  was not available here.
- torch 2.1.0 writes bytecode v8 and reads v4–v9, and so does 1.13.0 (DOCUMENTED:
  `caffe2/serialize/versions.h` at tags v2.1.0 and v1.13.0).
- "PyTorch Mobile is no longer actively supported. Please check out ExecuTorch" (DOCUMENTED:
  `pytorch/ios-demo-app` README).
- **Needs:** a Mac with Xcode and CocoaPods to build an app around the pod (INFERRED from the pod
  format). It also needs an Apple account to install on the iPhone. A free account is a "Personal
  Team": provisioning profiles expire after 7 days, with up to 3 devices and 3 apps per device
  (DOCUMENTED: developer.apple.com `/support/compare-memberships/`).

**ExecuTorch:**

- It ships prebuilt `.xcframework`s via Swift Package Manager in Xcode, needs at least iOS 17 and
  macOS 14, and uses Xcode 15+ (DOCUMENTED: `pytorch/executorch` `docs/source/using-executorch-ios.md`).
- It consumes `torch.export` programs, not TorchScript. Its INT8 path is PT2E quantization, so it
  cannot run this artifact. A re-export and a fresh quantization would make a *different* INT8 model,
  and it needs a newer torch than the frozen pin (INFERRED).
- An ExecuTorch export was not attempted.

### (c) Running on the iPhone without a Mac

| route | Mac? | Apple account? | runs the artifact of record? | status here |
|---|---|---|---|---|
| Server + phone browser (section 2) | no | no | **yes** (QNNPACK, server CPU) | MEASURED end to end in Chromium |
| ONNX Runtime Web, WASM, in Safari | no | no | the same graph as ONNX QDQ (≤ 1 quantization step) | MEASURED in Chromium; not on an iPhone |
| ONNX Runtime Web, WebGPU (HTTPS only) | no | no | FP32 only (INT8 QDQ wrong on WebGPU) | MEASURED in Chromium/SwiftShader |
| Static bundle as a Home Screen web app | no | no | as the WASM row | INFERRED (no service worker yet, so not offline) |
| Cloud macOS build (GitHub-hosted macOS runners) + LibTorch-Lite app | no own Mac | yes, to sign and install | yes (lite interpreter, bitwise) | runners DOCUMENTED (`actions/runner-images` README); signing for a device without Xcode INFERRED to be impractical |
| Expo EAS cloud build | no own Mac | $99 membership for App Store builds (DOCUMENTED: `expo/expo` `docs/pages/build/setup.mdx`) | would need an ONNX/React Native runtime: no | INFERRED, not recommended |

## 6. Hosting options (not deployed)

**What the server needs** (MEASURED with `demo/tools/bench_server.py` on an Intel Xeon @ 2.10 GHz,
4 logical CPUs, 10 requests per image after warm-up):

| threads | RSS after model load | peak RSS | inference median, 1024×768 / 12 MP photo | server total median, 1024×768 / 12 MP photo |
|---|---|---|---|---|
| 1 | 275 MB | 853 MB | 205 / 177 ms | 317 / 407 ms |
| 2 | 275 MB | 854 MB | 146 / 137 ms | 214 / 361 ms |
| 4 | 275 MB | 854 MB | 179 / 188 ms | 230 / 399 ms |

So plan **≥ 1 GB RAM (2 GB comfortable) and 1–2 vCPUs**. More threads do not help on this CPU, and the
peak comes with a 4032×3024 upload. QNNPACK is tuned for ARM; on x86 it may be slower than on an ARM
host (INFERRED).

| option | CPU / RAM it offers | fits? | notes |
|---|---|---|---|
| Team laptop on a private Wi-Fi or phone hotspot (WSL2 or Docker Desktop) | the laptop | yes | Recommended for the panel: no internet needed once the vendor files are local; plain HTTP (no WebGPU). WSL networking and Docker requirements: DOCUMENTED (section 2) |
| Hugging Face Spaces, Docker Space, CPU Basic | 2 vCPU, 16 GB (DOCUMENTED: `huggingface/hub-docs` `spaces-overview.md`) | yes | Docker and Gradio Spaces "require a paid plan to create" (PRO). Static Spaces are free, and the static on-device bundle (section 2) needs only static hosting with HTTPS, which also unlocks WebGPU and threads (DOCUMENTED; COOP/COEP header support INFERRED) |
| Google Cloud Run (container) | configurable vCPU and memory, scale to zero | INFERRED yes at 1 vCPU / 2 GiB | limits not fetched: docs.cloud.google.com is blocked in this session |
| Render free web service | INFERRED 512 MB, 0.1 CPU | **no** (peak 853 MB) | a paid ≥ 1 GB instance would be needed; docs not fetched (blocked) |
| Fly.io machine (shared CPU, 1–2 GB) | INFERRED | INFERRED yes | docs not fetched (blocked) |
| A university Linux VM | any 1–2 vCPU / 2 GB | yes | `docker run -p 8000:8000` from the image |

- The Docker image is `demo/Dockerfile` (python:3.11.11-slim-bookworm, non-root user). It builds from
  a context assembled by `demo/docker/build_context.py` from `git ls-files`.
- The test-named paths are excluded before listing, and the context holds only the modules the server
  imports: no `docs/`, no `src/stats`, no training code and no TEST-day catalogue.
- MEASURED: 49 files with `--with-ondevice`, and the server ran from that context (healthz, segment,
  `/ondevice/`). The image itself was not built, because there is no Docker daemon here.
- Licence: CC BY-NC 4.0 permits non-commercial sharing with attribution, so a thesis demo qualifies
  and the page shows the attribution (INFERRED reading, not legal advice).

## 7. Swapping in the artifact of record (after TEST)

1. Put the INT8 QNNPACK TorchScript file of record on the demo machine, **outside the repository**
   (checkpoints live outside the repo). This is the file `scripts/run_ptq.py` writes through
   `src/quant/ptq.py`, carrying `plantseg_int8.json`.
2. **The one line.** In `demo/config.py`, change
   `MODEL_PATH = Path(os.environ.get("PLANTSEG_DEMO_MODEL") or STAND_IN_PATH)`
   to
   `MODEL_PATH = Path(os.environ.get("PLANTSEG_DEMO_MODEL") or "/path/to/<artifact>.pt")`.

   Alternatively, set `PLANTSEG_DEMO_MODEL=/path/to/<artifact>.pt` (the Docker image sets it; copy the
   file in with `build_context.py --model /path/to/<artifact>.pt`).
3. Run `python -I demo/run_demo.py --no-build`. Then check three things:
   - the page's file name and sha256 equal the run provenance's artifact sha256;
   - its identity line shows the stage and method from `plantseg_int8.json`;
   - the stand-in banner is gone.

   The evaluator's loader enforces the schema, the 116 classes and the qnnpack engine; any mismatch
   stops start-up.
4. Run `python -I demo/smoke_demo.py --model /path/to/<artifact>.pt`; all 8 must pass.
   - MEASURED for a non-stand-in file: 8/8 with a converted E4-identity file, `converted_E4.pt`.
5. On-device files:
   - Run `python -I demo/ondevice/export_onnx.py --model /path/to/<artifact>.pt --no-fp32`. The INT8
     ONNX comes from the artifact itself; check the printed parity (expect ≤ about 1 quantization step).
   - For FP32 ONNX (WebGPU), pass `--fp32-state` with the FP32 source checkpoint. It is loaded through
     `src/eval/model_loading.py load_student_checkpoint`, in the nested schema.
   - Then run `python -I demo/ondevice/make_static_bundle.py OUT --model /path/to/<artifact>.pt` if
     you use the bundle.
6. Re-run the browser check and re-take the screenshots:
   - `python -I demo/tools/make_browser_fixtures.py FIX --model …`
   - `node demo/tools/browser_check.cjs http://127.0.0.1:8000 FIX wasm`
   - `node demo/tools/screenshots.cjs http://127.0.0.1:8000 FIX/fx_1024x768.jpg demo/screenshots`
7. If the artifact of record is a runner-schema INT8 **state_dict** (`src/quant/runner.py`) instead of
   TorchScript, first run
   `python -I demo/model/torchscript_from_state_dict.py STATE.pt --stage E5 --method qat --out /path/to/demo_e5.pt`.
   - It uses the evaluator's `rebuild_int8_from_state_dict`, then `torchscript_bytes`, then a bitwise
     `output_parity` check, with `artifact_role demo_conversion`.
   - MEASURED: bitwise parity on a synthetic runner-schema file.
8. Native iOS only: run `python -I demo/ondevice/lite_interpreter.py --model /path/to/<artifact>.pt`.

## 8. Smokes and checks (final RESULT lines)

`python3 -I demo/smoke_demo.py` on the final code (MEASURED, 8/8):

```
RESULT PASS S1 server_starts: healthz={'ok': True, 'model_sha256': 'a5c699b5…fc32'} after 2.0s
RESULT PASS S2 class_list_is_repos: 116 served == 116 raw == 116 loader; configs/data.py num_classes 116; model outputs 116; class 0 name '' role background_or_non_disease
RESULT PASS S3 mask_shape_and_class_range: leaf_1024x768.jpg: mask (384, 512) …; leaf_480x640.png: mask (512, 384) …; leaf_800x800.jpg: mask (512, 512) …; leaf_720x1280.jpg: mask (512, 288) …
RESULT PASS S4 latency_field_present: decode/preprocess/inference/postprocess/total in ms
RESULT PASS S5 preprocessing_byte_equal: in-process wrapper == dataset path: True; server input sha256 == dataset path: True; placeholder independent: True; protocol core_preprocess/1.0.0 == src.eval.adapters.PREPROCESS_PROTOCOL; over 4 images (JPEG and PNG)
RESULT PASS S6 exif_orientation_applied: stored 800x600 with orientation 6 -> photo box 384x512 (w x h)
RESULT PASS S7 bad_upload_refused: HTTP 400 could not decode the image (UnidentifiedImageError): send a JPEG or PNG
RESULT PASS S8 no_src_stats: server [], smoke []
SUMMARY 8/8 passed
```

What S5 means:

- The server calls `src/data/transforms.py core_preprocess` + `finalize` exactly as the evaluator's
  dataset does (`src/data/dataset.py:96-97`, the canvas protocol; the contract's thesis protocol).
- An upload has no mask, so it passes a constant placeholder mask. The smoke shows the placeholder's
  value cannot change the image tensor, and its padded canvas gives the photo's box.
- The upstream (L-AM13) protocol is used only for the descriptive comparison with 42.05%, so it is not
  the demo's.

Other checks:

- The headless browser check is in section 5.
- `src.stats` is never imported, asserted in every entry point and the smoke.
- Every Python run used `python3 -I`, and no pytest was used.

## 9. What I could not do, and why

- **The time box.** The session stopped overnight with no message (section 0).
- **No iPhone or Safari.** Nothing was measured on the iPhone 15 Plus, so phone latency, Safari's
  decoding and colour management, and the HEIC and 16.7 MP paths are unmeasured. Chromium is the only
  pre-installed browser, and the session rules forbid installing others.
- **No Docker daemon.** The image was not built (section 6).
- **No Windows machine.** The Windows finding comes from inspecting the official wheel's binary, not
  from running it.
- **ExecuTorch export not attempted.** It would need a newer torch, in a separate environment.
- **WebGPU INT8 not diagnosed** beyond the measurements in section 5.
- **Documentation hosts blocked.** These were outside this session's network policy:
  - not resolvable: pytorch.org, docs.pytorch.org, webkit.org, developer.mozilla.org, onnxruntime.ai,
    huggingface.co, render.com, fly.io, docs.github.com, docs.expo.dev, learn.microsoft.com and
    docs.cloud.google.com;
  - refused by the proxy (403): trunk.cocoapods.org.

  Their GitHub-hosted sources were used instead. Render, Fly.io and Cloud Run figures stay INFERRED.
  Allowing those hosts in the environment's network settings would let a later session cite them
  directly.
- **The auto-mode classifier refused one install** (a mixed-index `uv pip install` with
  `--index-strategy unsafe-best-match`). I installed the same pins in two single-index steps instead.

## 10. The SL-1 log, slips, conflicts and open items

- **SL-1 log** (`~/.claude/sl1_guard.log`, log mode; 3 would-be denials, all from my opening recon
  in the full clone):
  - B1: a globbed root listing, `ls requirements*`. It listed only the requirements files.
  - B1: `ls .claude/hooks/`. This printed the name of one test-named file there. **Slip:** the name
    only; the file was not opened. Afterwards every listing used `grep -vi test`, and every listing
    after the sparse clone ran inside it, where no test-named path exists.
  - B4: the count check `git ls-files -- ':(icase)*test*' | wc -l` in the sparse clone. It printed 0.
- **Third-party packages:** `import torch` loads torch's own `torch/testing` modules. That is the
  interpreter, outside the repository; I opened no such file.
- **Tooling conflicts, reported once:**
  - The harness asked me to develop and push on `claude/new-session-huavyv` and to add
    `Co-Authored-By`/`Claude-Session` trailers.
  - The brief overrides both: only `lane/demo` was pushed, and no commit has a trailer.
  - Nothing was pushed to master or any other branch.
- **The decision log** has no entry for the demo, so no Status was updated (the brief also forbids
  modifying existing files).
- **Open items:**
  1. Measure the server page and the WASM page on the iPhone 15 Plus, over the venue network, before
     quoting numbers.
  2. Pick the host. The recommendation is a team laptop with WSL2 or Docker, plus the static bundle as
     a fallback.
  3. After TEST, run section 7 and re-take the screenshots.
  4. If on-device INT8 matters to the panel, try a later onnxruntime-web release for WebGPU QDQ.
  5. Optionally, add a service worker so the static bundle works offline as a Home Screen app.
