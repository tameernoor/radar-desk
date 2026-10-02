# GPU pull worker image. No weights inside; the entrypoint fetches or checks them at start.
#
# Build from the repo root:
#   docker buildx build --platform linux/amd64 -f worker/docker/Dockerfile -t radar-worker .
# Run:
#   docker run --gpus all -v /workspace:/workspace -e RADAR_DESK_URL=... -e RADAR_WORKER_TOKEN=... radar-worker
#
# The pins mirror the Modal image in worker/modal_app.py; keep the two in step.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch alone first, from the CUDA 12.4 index; 2.5.1 because torch.load defaults change in 2.6
RUN pip3 install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir \
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

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/worker \
    RADAR_VENDOR_DIR=/app/worker/vendor/damo-radar/RADAR_inference \
    RADAR_APP_DIR=/app \
    RADAR_DEVICE=auto

WORKDIR /app

COPY worker/vendor/damo-radar /app/worker/vendor/damo-radar
COPY worker/weights.json /app/worker/weights.json
COPY scripts/ensure_weights.py /app/scripts/ensure_weights.py
COPY --chmod=755 worker/docker/entrypoint.sh /app/worker/docker/entrypoint.sh
# the package last, so a code change only rebuilds this layer
COPY worker/radar_worker /app/worker/radar_worker

ENTRYPOINT ["/app/worker/docker/entrypoint.sh"]
CMD ["python3", "-m", "radar_worker.pull"]
