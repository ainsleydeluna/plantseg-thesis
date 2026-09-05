# Official PlantSeg experiment image — Linux x86-64, CPython 3.11, torch 2.1.0+cu121.
#
# BASE CHOICE (and why not the historical PyTorch image). The registered stack requires
# **Python 3.11**. The `pytorch/pytorch:2.1.0-cuda12.1-*` images do not deliver it: the upstream
# v2.1.0 Dockerfile defaults to `ARG PYTHON_VERSION=3.8` and the published 2.1.0 tags ship a
# conda Python that is not 3.11. Using them would silently violate the registered interpreter, so
# this image starts from the official Python 3.11 image pinned by immutable digest and installs the
# exact registered cu121 PyTorch and OpenMMLab BINARY artifacts explicitly. Nothing is compiled from
# source: the cu121 torch/torchvision wheels and the mmcv 2.1.0 cp311 manylinux wheel all exist for
# this exact combination, and the +cu121 wheels bundle the CUDA runtime they need.
#
# GPU access comes from the host at run time (nvidia-container-runtime / RunPod). No host driver,
# CUDA driver version, or GPU MODEL is baked in: the image defines SOFTWARE, while the run record
# defines the actual hardware.
#
# Base: python:3.11-slim-bookworm (Debian bookworm, glibc 2.36 -> manylinux1/2014 compatible)
FROM python@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91

# Deterministic, non-interactive, no bytecode, unbuffered logs (RunPod streams stdout).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Native libraries the registered wheels dlopen at import time:
#   libgl1 + libglib2.0-0  -> opencv-python 4.8.1.78 (imported by mmcv/mmsegmentation)
#   libgomp1               -> OpenMP runtime used by torch and scikit-image
#   git                    -> LOAD-BEARING, not convenience: src/eval/artifacts.py records the commit
#                             and the governed-path porcelain in every official artifact, so official
#                             evaluation cannot be finalised without it.
# NOTE: apt package versions follow the pinned base digest rather than being individually pinned;
# that residual non-determinism is recorded in docs/runpod_environment.md.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/plantseg-thesis

# ---- dependency layer first: the lock changes far less often than the source ----
COPY requirements-runpod.lock requirements-runpod.in ./

# --require-hashes makes pip refuse ANY unpinned requirement or altered artifact, which is what
# prevents the failure modes this stack is sensitive to: a CPU-only torch, a newer torch, a NumPy
# upgrade, an MMCV source build, or a silent "latest" resolution. Every one of those produces a
# different artifact hash and therefore a hard install failure rather than a wrong experiment.
RUN python -m pip install --require-hashes --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html \
      -r requirements-runpod.lock

# ---- source layer ----
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
# the only docs the runtime reads (artifact provenance + contract hashing)
COPY docs/IMPLEMENTATION_CONTRACT.md docs/EVALUATION_CONTRACT.md docs/STATISTICAL_ANALYSIS_CONTRACT.md ./docs/
# readable version registry, kept for provenance reporting inside the container
COPY requirements.lock requirements-e1.txt ./
# The ONE reports/ artifact the runtime hard-depends on: the B18a CE class-weight vector that
# train_e1.py:49,55,293 loads UNCONDITIONALLY through a fail-closed loader. Without it the real E1 run
# aborts at loss construction and preflight_e1.py NO-GOs at stage 1/5. Copied by exact path, never
# reports/ wholesale; this also creates reports/ for verify_class_weights_pod.py to write into.
COPY reports/e1_class_weights.json ./reports/

# Dataset root is MOUNTED, never baked. No dataset, checkpoint, key or secret is in this image.
ENV PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg \
    PYTHONPATH=/workspace/plantseg-thesis

# Fail the BUILD if the installed stack does not match the registered versions. Image mode does not
# require a GPU, so this works on a CPU-only builder; the GPU preflight is a separate run-time mode.
RUN python -B scripts/preflight_environment.py --mode image

# Commit provenance. Declared LAST on purpose: an ARG invalidates every layer after it, so placing
# this above the pip layer would force a full ~8.6 GB reinstall on every new commit. Ch4 cites the
# image DIGEST, and a digest with no revision label cannot be traced back to a tree without external
# notes — this makes the image self-identifying.
ARG GIT_COMMIT
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"

CMD ["/bin/bash"]
