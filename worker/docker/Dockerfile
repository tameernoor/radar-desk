# GPU pull worker image. No weights inside; the entrypoint fetches or checks them at start.
#
# Build from the repo root:
#   docker buildx build --platform linux/amd64 -f worker/docker/Dockerfile -t radar-worker .
# Run:
#   docker run --gpus all -v /workspace:/workspace -e RADAR_DESK_URL=... -e RADAR_WORKER_TOKEN=... radar-worker
#
# The pins mirror the Modal image in worker/modal_app.py; keep the two in step.
# The base is plain Python: torch pulls its CUDA libraries as nvidia-* pip packages and the driver
# comes from the host, so a CUDA base image mostly adds size. The digest is python:3.10-slim as
# built and smoke-tested on 2026-10-02; move it on purpose, not by accident.
FROM python:3.10-slim@sha256:c1aaf3d03e14944a039a1647e0b3f6f34c6bee517bac6ff380215ee099c4e808

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# the pip bundled with the base rejects the torch index's dependency wheels over name
# spelling (Jinja2 against jinja2), so a newer pip comes first
RUN pip install --no-cache-dir pip==24.3.1

# torch alone first, from the CUDA 12.4 index; 2.5.1 because torch.load defaults change in 2.6
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    "numpy<2" \
    monai==1.4.0 \
    nibabel \
    SimpleITK \
    transformers==4.25.1 \
    huggingface_hub==0.16.4 \
    pandas \
    tqdm \
    scipy \
    requests

# NVIDIA_*: what the CUDA base image used to set; the container runtime reads them to expose the GPU
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/worker \
    RADAR_VENDOR_DIR=/app/worker/vendor/damo-radar/RADAR_inference \
    RADAR_APP_DIR=/app \
    RADAR_DEVICE=auto \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

COPY worker/vendor/damo-radar /app/worker/vendor/damo-radar
COPY worker/weights.json /app/worker/weights.json
COPY scripts/ensure_weights.py /app/scripts/ensure_weights.py
COPY --chmod=755 worker/docker/entrypoint.sh /app/worker/docker/entrypoint.sh
# the package last, so a code change only rebuilds this layer
COPY worker/radar_worker /app/worker/radar_worker

ENTRYPOINT ["/app/worker/docker/entrypoint.sh"]
CMD ["python3", "-m", "radar_worker.pull"]
