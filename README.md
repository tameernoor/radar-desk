# radar-desk

radar-desk runs RADAR, the abdominal CT model Alibaba DAMO Academy published in Science in 2026, on a CT scan and shows its 146 finding scores next to the image, with RADAR's own organ outlines on top. A chat panel answers questions about the scores and moves the viewer.

Research use only. Public and research scans only. This is not a medical device.

## What you can do in it

- Upload a scan (`.nii` or `.nii.gz`) and score it on a GPU.
- Click a finding and the viewer jumps to that organ with RADAR's outline isolated.
- Press `w` for "What am I looking at?": the slice, every organ RADAR outlined on it with its scores, and what is under the crosshair. No model call.
- Press `m` for one main view plus two reference views. Click a reference view to make it the main one.
- Press `f` for focus mode, the image over the whole window with a Light panel for window width and level in HU, gamma, invert and colour map. `+` and `-` or ctrl-wheel zoom, shift-drag pans.
- Click "View for {organ}" (or ask the chat) to see an organ the way it is usually read: its window from a cited table (Angio, key `5`, for the aorta), centred and zoomed, with RADAR's scoring box.
- Ask the chat ("show me the liver", "what did RADAR flag above 50%?", "what am I looking at?"). It can move the viewer.
- Export scores as CSV or JSON, and the organ mask as NIfTI.

A score is how close an organ looks to a finding's text, not a calibrated probability. The 50% line is for display.

## How it works

A FastAPI server takes the upload, and the scan goes to a shared S3-compatible bucket (or a volume). A GPU runs DAMO's own PyTorch code and checkpoint, unedited, and writes the scores and the organ mask back to the same storage. The GPU is one of four, chosen on the jobs page: a Modal function, your own NVIDIA machine running the pull worker, a RunPod pod the app starts, or a RunPod serverless endpoint. The viewer (NiiVue) reads them through the API. The chat panel is persona, backed by any OpenAI-compatible model. `LLM_PROVIDER` picks OpenRouter (the default), a local Ollama or another endpoint.

## Results

| Check | Result |
|---|---|
| Our wrapper against DAMO's own script, same GPU | identical, all 146 scores |
| DAMO's published demo scores | max difference 0.0064, all in the oesophagus |
| A Merlin scan against the browser port (radar-web) | same positives, 84.8 / 62.1 / 60.0% |
| Time on an L4 | 16 to 28 s per scan, about 15 s model load when cold |
| Cost on Modal | about 3 cents per scan, including the idle minutes |
| RunPod pod, RTX 4090 | about 2.5 min from Score to a ready pod, 2.4 s scoring |
| RunPod serverless, L4 | 20 s per scan, about 2.5 cents |

## Run it

You need `uv`, Node and a Modal account. The checkpoint goes on a Modal Volume called `radar-weights`, laid out as in `worker/weights.json`.

```sh
uv sync
cd web && npm install && npm run build && cd ..
cp .env.example .env    # set OWNER_TOKEN, SESSION_SECRET, the Modal token and, for chat, LLM_API_KEY and CHAT_MODEL
set -a && source .env && set +a && uv run modal deploy worker/modal_app.py
uv run python -m radar_desk
```

Open http://127.0.0.1:8000 and log in with your `OWNER_TOKEN`. With `GPU_BACKEND=fake` it runs without a GPU and makes up clearly marked results. `.env.example` explains every setting.

The recommended setup is one S3-compatible bucket for scans and results, which every compute mode can use. Set `S3_BUCKET` and the four `S3_*` and `AWS_*` keys from `.env.example`, leave `STORAGE_BACKEND` unset, and allow the app's origin in the bucket's CORS rules, because the browser uploads to it directly. The jobs page says which storage is in use under the Compute choice. To move existing scans and results off a volume, run `uv run python scripts/migrate_storage.py --from modal_volume --to s3 --dry-run`, then again without `--dry-run`.

## Run on your own GPU

Pick Own GPU workers on the jobs page (or set `GPU_BACKEND=worker` before the first start). The app then only queues jobs and waits for a worker; it never starts a RunPod pod. Create a worker token on the jobs page, then on the GPU machine run `docker run --gpus all -v <weights dir>:/workspace -e RADAR_DESK_URL=<app url> -e RADAR_WORKER_TOKEN=<token> <image>`. The jobs page shows the exact line with this app's URL. The image is `WORKER_IMAGE`, built from `worker/docker/Dockerfile`, and the weights are fetched into `<weights dir>/radar-weights` on first start. The worker must be able to reach the app, on the same network, through a tunnel, or at `WORKER_PUBLIC_URL`.

What is tested. The worker image and code are the same that ran on RunPod RTX 4090 and L4 pods, with scores identical to Modal, but they have not been run on a home or on-prem NVIDIA machine. It needs a 24 GB NVIDIA card (measured peak 15 to 20 GB). On an Apple GPU, `scripts/score_local.py --device mps` gives correct scores but needs about 32 GB of unified memory in practice (a 16 GB M5 took 16 minutes, swapping). CPU works and is slow.

## Compute

The Compute choice at the top of the jobs page moves scoring between Modal, your own workers, a RunPod pod and RunPod serverless without a restart. In RunPod pod mode the app starts one pod when a job is queued and the pod deletes itself after 10 idle minutes; it needs `WORKER_IMAGE` and the `RUNPOD_*` keys from `.env.example`. `RUNPOD_GPUS` sets which GPU types to try, in order.

The pod reaches the app through `WORKER_PUBLIC_URL`, which the app only checks, or, when that is unset, through a Cloudflare quick tunnel the app starts and stops itself. `uv run python -m radar_desk.compute workers|runpod [--start]|modal|serverless|stop|status` does the same from a terminal.

RunPod serverless submits each job to an endpoint that scales to zero, with no tunnel and no pod to manage; the scans and artefacts sit on the RunPod network volume (`STORAGE_BACKEND=runpod_volume`) or in the S3 bucket. Pin `WORKER_IMAGE` to a `worker-vX.Y` tag, create the endpoint once with `uv run python scripts/runpod_endpoint.py create`, put the printed `RUNPOD_ENDPOINT_ID` in `.env`, and pick `RunPod serverless` on the jobs page.

To roll out a worker change, merge it (CI builds the image), tag the commit `worker-vX.Y` and push the tag (CI publishes `:X.Y`), and set `WORKER_IMAGE` to that tag. Then restart the app for pods, run `scripts/runpod_endpoint.py update` for serverless, and run `modal deploy worker/modal_app.py` for Modal. Never pin `:latest`.

Tests are `uv run pytest` and, in `web/`, `npm test`.

## Licence

The whole repository is CC BY-NC-SA 4.0, because it includes and adapts RADAR's code. Model and weights by Alibaba DAMO Academy ([huggingface.co/radar-generalist/RADAR](https://huggingface.co/radar-generalist/RADAR)). Paper: "An expert-level generalist AI for abdominal CT diagnosis", Science 393(6817), eaec6129, doi [10.1126/science.aec6129](https://doi.org/10.1126/science.aec6129). See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
