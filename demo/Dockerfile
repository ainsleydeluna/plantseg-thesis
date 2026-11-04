# PlantSeg mobile web demo (research prototype; CPU only; not deployed by this lane).
#
# Linux image, so the QNNPACK engine the INT8 artifact needs is available (the official Windows torch 2.1
# wheels are built without it; see docs/lane_reports/demo.md). Build from a context made by
# demo/docker/build_context.py, never from the repository root: the script copies only src/ (without
# src/stats), configs/, demo/ and the model file, so docs/ and test-named paths never enter the context.
#
#   python3 -I demo/docker/build_context.py ../plantseg-demo-context [--with-ondevice]
#   docker build -t plantseg-demo ../plantseg-demo-context
#   docker run --rm -p 8000:8000 plantseg-demo            # then http://<host>:8000 on the phone
FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# The server's runtime only (the ONNX packages are for the exports, which run elsewhere).
COPY demo/requirements-demo.txt /tmp/requirements-demo.txt
RUN grep -vE '^(onnx|onnxruntime)==' /tmp/requirements-demo.txt > /tmp/requirements-server.txt \
 && pip install -r /tmp/requirements-server.txt && rm /tmp/requirements-*.txt

COPY src /app/src
COPY configs /app/configs
COPY demo /app/demo

# build_context.py places the model (MODEL_PATH at build time) here
ENV PLANTSEG_DEMO_MODEL=/app/demo/models/model.pt
RUN useradd --uid 10001 --no-create-home demo
USER demo
EXPOSE 8000
CMD ["python", "-I", "demo/run_demo.py", "--host", "0.0.0.0", "--port", "8000", "--no-build"]
