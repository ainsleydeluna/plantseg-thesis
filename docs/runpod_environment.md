# Official RunPod experiment environment — build, verify, run, record

The reproducibility envelope for every official PlantSeg experiment. Four artifacts make it up:

| artifact | role |
|---|---|
| `requirements.lock` | human-readable **exact-version registry** (the version authority; consumed by `src/corruption_cache.py` and the smokes) |
| `requirements-runpod.in` | resolver **input**, derived verbatim from the registry minus Windows-only wheels |
| `requirements-runpod.lock` | **hash-verified** Linux x86-64 / CPython 3.11 installation lock (78 distributions, 78 sha256 constraints) |
| `Dockerfile` + `.dockerignore` | the **container specification**, base pinned by immutable digest |

## Two states, never conflated

- **CONTAINER SPECIFICATION COMPLETE** — the four artifacts above are coherent and statically
  verified by `scripts/smoke_environment.py`. This is the current state.
- **FULL OFFICIAL EXPERIMENT ENVIRONMENT EXECUTABLY VALIDATED** — the image has actually been built
  **and** `scripts/preflight_environment.py --mode gpu` has passed on the real accelerator. This has
  **not** happened, so `full_experiment_environment_validated` remains `false`. A Dockerfile that
  merely exists is not validation.

Separately validated already: the **corruption dependency closure** (Python 3.11 · numpy 1.26.4 ·
Pillow 12.3.0 · scikit-image 0.23.2), executably verified in `0b58050`. That is a strictly smaller
claim than the full environment, and the container preflight re-asserts it inside the image.

## Registered stack

Python 3.11 · torch 2.1.0+cu121 · torchvision 0.16.0+cu121 · mmcv 2.1.0 · mmsegmentation 1.2.2 ·
mmengine 0.10.7 · numpy 1.26.4 · scipy 1.11.4 · Pillow 12.3.0 · scikit-image 0.23.2 ·
opencv-python 4.8.1.78 · statsmodels 0.14.6 · fvcore 0.1.5.post20221221 · triton 2.1.0.

Binary artifacts, not source builds: the cu121 torch/torchvision wheels and
`mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl` (from the `cu121/torch2.1.0` OpenMMLab index) all
exist for exactly this combination, so **MMCV is never compiled**.

**Base image.** `python:3.11-slim-bookworm`, pinned by digest
`sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91` (Python 3.11.16,
glibc 2.36). The historical `pytorch/pytorch:2.1.0-cuda12.1-*` images are **rejected**: the upstream
v2.1.0 Dockerfile defaults to `ARG PYTHON_VERSION=3.8` and the published tags do not ship Python
3.11, so they would silently violate the registered interpreter. The `+cu121` wheels bundle the CUDA
runtime they need; the GPU itself comes from the host runtime.

## 1 — Build the image

```bash
scripts/build_and_push_image.sh              # build, verify, push
scripts/build_and_push_image.sh --build-only # build + verify, no push
```

**This script is the only supported build path.** It derives the commit from `git rev-parse HEAD`,
refuses to build when any governed path is dirty or when HEAD is not on `origin`, verifies the
revision label and the class-weight gate stage before pushing, and never handles credentials. A raw
`docker build` is what produced the superseded image in §3: hand-built, hand-tagged, with its revision
label applied on the CLI and no record of the procedure, it drifted 19 commits behind HEAD while still
looking authoritative.

The build fails if the installed stack drifts: `pip install --require-hashes` rejects any unpinned or
altered artifact, and the final layer runs the image-mode preflight.

## 2 — Verify the image (no GPU needed)

```bash
docker run --rm plantseg-thesis:official python -B scripts/preflight_environment.py --mode image
```

Proves **software identity only**: Linux x86-64, Python 3.11, every registered version, the torch
cu121 build identity, and the load-bearing imports. It cannot establish the GPU environment.

## 3 — Record the image identity

```bash
docker image inspect plantseg-thesis:official --format '{{.Id}}'
docker image inspect plantseg-thesis:official --format '{{json .RepoDigests}}'
```

Put the resulting image ID/digest in the run provenance of every official artifact.

### Authoritative image — provision BY DIGEST

**Every official run provisions from this digest:**

    ghcr.io/ainsleydeluna/plantseg-thesis
      @sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6

Built from repo commit `28d1038a4b36caf02736b190dc3e032f181b7bb5`. Verification:
`reports/b38_docker_image.md` §8.

**Provision by digest, never by tag.** `:official` is a moving pointer that the next official build
reassigns, so a run recorded against `:official` cannot be reproduced later. A digest is
content-addressed and cannot change meaning:

```bash
docker pull ghcr.io/ainsleydeluna/plantseg-thesis@sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6
```

> **SUPERSEDED — do not use.** `sha256:5f5dba46b8668949dc166bfbe7acc6db0fc90beba746b425d5e057183adbf9d2`
> (tags `:148af2c`, `:runpod-preflight-2026-08-17`) is still on the registry and still pullable, but
> **must not provision any run.** It predates `scripts/preflight_e1.py` entirely — the E1 pre-flight
> gate did not exist when it was built — along with the B31/B31c training-loop hardening and the
> B31-5/A1 augmentation-RNG fix. It is left in place deliberately so an old note citing it resolves to
> something clearly marked superseded rather than to nothing.

## 4 — Start on a GPU-capable runtime, dataset mounted externally

The dataset is **never** baked into the image (`.dockerignore` denies by default and re-excludes
`datasets/`, `plantseg_data/`, `data/`, checkpoints, archives, keys and the protected
`docs/reference/`). Mount it instead:

```bash
docker run --rm -it --gpus all -v /host/plantseg_data:/workspace/plantseg_data:ro plantseg-thesis:official bash
```

On RunPod, attach the volume containing `plantseg_data/plantseg` and keep
`PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg`. Checkpoints are written **outside** the
repository (`--ckpt-dir /workspace/e1_ckpts`), which `train_e1.py` hard-guards.

## 4b — Two run-time prerequisites the image deliberately does not carry

The image ships **software only**. Two committed surfaces need more than that, and both were
confirmed by running the suite inside the container:

1. **A git checkout is required for artifact provenance.** `src/eval/artifacts.py` records the commit
   and the governed-path porcelain in every official artifact, so it shells out to `git`. The binary
   is installed in the image, but the build context deliberately excludes `.git`, so the image's
   copied source is **not** a repository. `scripts/smoke_eval_stage_artifacts.py` and
   `scripts/smoke_eval_contract.py` therefore fail inside the bare image with
   `fatal: not a git repository` and pass on a checkout. For official runs, provide the code as a
   **git checkout** — mount the repository (or clone it) at `/workspace/plantseg-thesis` — rather than
   relying on the baked copy.
2. **The dataset must be mounted for anything that builds a dataloader.** `PlantSegDataset` requires
   the split directories to exist, so even `train_e1.py --dry-run` (whose weights and compute are
   synthetic) constructs real dataloaders and refuses with `dataset root does not exist` in a
   dataset-free container. Mount `plantseg_data` as in step 4 before running it.

Neither is an implementation defect; both are properties of a deliberately dataset-free,
history-free image.

## 5 — Official GPU preflight

```bash
python -B scripts/preflight_environment.py --mode gpu --json /workspace/preflight_gpu.json
```

Fails unless `torch.cuda.is_available()`. On success it records device name, device count, compute
capability and the CUDA build — the run-specific hardware facts.

## 6 — Record run provenance (hardware, not image)

The image defines **software**; the run record defines **hardware**. No GPU model is baked into the
Dockerfile. For each official run capture: image ID/digest · GPU model and count · driver / CUDA
runtime · CPU model and core count · RAM · RunPod pod type and region · torch thread settings ·
the `--mode gpu` preflight JSON.

## Regenerating the hashed lock

Run on Linux x86-64 (hashes come from the actual resolved artifacts — never hand-written):

```bash
docker run --rm -v "$PWD:/work" python@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91 bash -lc 'python -m pip install --dry-run --ignore-installed --report /work/report.json --extra-index-url https://download.pytorch.org/whl/cu121 --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html -r /work/requirements-runpod.in'
```

Then convert `report.json` into `requirements-runpod.lock`, taking `download_info.archive_info.hashes.sha256`
per distribution. Two artifacts (`Jinja2`, `MarkupSafe`) are served by the PyTorch index without hash
fragments; those were downloaded and hashed directly, and their digests were confirmed identical to
the PyPI-served copies of the same versions.

## Local build evidence (NOT official validation)

The image has been built once locally to prove the specification actually works — Docker 29.7.2,
`linux/amd64` (WSL2 backend), **no GPU, no dataset**:

| observation | result |
|---|---|
| `docker build` | **succeeded**; final image ≈ 8.6 GB |
| hash-verified install | every one of the 78 pinned artifacts accepted under `--require-hashes` |
| `--mode image` preflight | **PASS 36/36** inside the container |
| interpreter | Linux x86_64, **Python 3.11.16**, glibc 2.36 |
| framework identity | torch 2.1.0+cu121 · torchvision 0.16.0+cu121 · `torch.version.cuda = 12.1` |
| OpenMMLab | mmcv 2.1.0 with **compiled ops importable** · mmseg 1.2.2 · mmengine 0.10.7 (no source build) |
| corruption closure | reproduced: Python 3.11 · numpy 1.26.4 · Pillow 12.3.0 · scikit-image 0.23.2 |
| `--mode gpu` preflight | **correctly FAILED** (35/36, `cuda_available`) — a CPU host cannot fake the official preflight |

**Significant finding for the deferred INT8 work.** Inside the official image
`torch.backends.quantized.supported_engines` reports
**`['qnnpack', 'none', 'onednn', 'x86', 'fbgemm']`** — both the **QNNPACK** accuracy/deployment
backend and the **fbgemm/x86** CPU-proxy latency backend are present. Locally only `onednn` was
available, which is what blocked the E4–E7 INT8 artifacts and the x86 latency copy. Those blockers
are environmental, not architectural, and this image resolves them.

**Registered-stack CPU integration.** The committed synthetic/dry-run suites were then executed
*inside* this image (source mounted from the checkout, pinned upstream corruption reference mounted,
no GPU, no dataset, no training). Every code-level suite passes —
`smoke_quant_x86_efficiency` 102/102 · `smoke_efficiency` 82/82 (registered fvcore actually
executing) · `smoke_quant_runners` 77/77 · `smoke_quant_e4_e5` 58/58 · `smoke_quant_e6_e7` 37/37 ·
`smoke_eval_int8` 32/32 · `smoke_corruption_vendor` **87/87 including the 40/40 zero-tolerance
corruption grid** · `smoke_evaluate_corruptions` 25/25 · `smoke_eval_robustness` 59/59 ·
`smoke_eval_teacher` 38/38. The only non-passing items are the two prerequisites in §4b.

On this stack the backend-dependent paths run for real instead of being skipped: the E4/E7 x86 PTQ
copies and the E5/E6 zero-training sidecar translation are **built and converted** under an approved
`x86`/`fbgemm` engine. A few suite counts differ from the Windows development box because those
branches now take the real-backend path rather than the unavailable-backend path.

This is **build and CPU-integration evidence only**. It is not the official environment validation: no
GPU was involved, so `full_experiment_environment_validated` stays `false`. Local Docker image IDs are
not portable identifiers — record your own via step 3 for each official run.

## Known residual gaps

- **apt package versions** are not individually pinned; they follow the pinned base digest. Full
  determinism would need a Debian snapshot mirror.
- **No GPU run has been performed.** The image builds and passes image-mode preflight, but the
  official environment requires a passing `--mode gpu` on a real accelerator.
- fvcore is specified but its real profiling execution is still pending.
- The teacher (MMSeg) stack is installed but teacher fine-tuning has not been run in this image.
