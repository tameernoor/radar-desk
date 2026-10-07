# GPU pull worker image. No weights inside; the entrypoint fetches or checks them at start.
#
# Build from the repo root:
#   docker buildx build --platform linux/amd64 -f worker/docker/Dockerfile -t radar-worker .
# Run:
#   docker run --gpus all -v radar-weights:/workspace -e RADAR_DESK_URL=... -e RADAR_WORKER_TOKEN=... radar-worker
#
# The pins (worker/docker/requirements.in) mirror the Modal image in worker/modal_app.py; keep the two in step.
# The base is plain Python: torch pulls its CUDA libraries as nvidia-* pip packages and the driver
# comes from the host, so a CUDA base image mostly adds size. The digest is python:3.10-slim as
# published 2026-10-06 (worker 0.6); move it on purpose, not by accident.
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# The worker runs as radar; the entrypoint starts as root only to hand it the weights directory.
# /workspace belongs to radar so a fresh named volume mounted there does too.
RUN useradd --create-home --uid 1000 --user-group radar \
    && mkdir -p /workspace && chown radar:radar /workspace

# Every package at an exact version with its sha256 (requirements.in says how to recompile).
# A newer pip first, hash-checked like the rest.
# setuptools and wheel go afterwards: the scan flags them and eager inference never imports them.
# triton does (runtime/build.py), so torch.compile does not work in this image.
COPY worker/docker/requirements-pip.txt worker/docker/requirements.txt /tmp/
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements-pip.txt \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements-pip.txt /tmp/requirements.txt \
    && pip uninstall --yes setuptools wheel

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
