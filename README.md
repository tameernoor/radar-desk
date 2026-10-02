# radar-desk

**Part of aotn**, a series of small example projects. This one goes with the article "What the Reports Already Knew", about RADAR, the abdominal CT model Alibaba DAMO Academy published in Science in 2026.

radar-desk runs RADAR on a CT scan and shows its 146 finding scores next to the image, with RADAR's own organ outlines on top. A chat panel answers questions about the scores and moves the viewer.

Research use only. Public and research scans only. This is not a medical device.

## What you can do in it

- Upload a scan (`.nii` or `.nii.gz`) and score it on a GPU.
- Click a finding and the viewer jumps to that organ with RADAR's outline isolated.
- Press `w` for "What am I looking at?": the slice, every organ RADAR outlined on it with its scores, and what is under the crosshair. No model call.
- Press `m` for one main view plus two reference views. Click a reference view to make it the main one.
- Ask the chat ("show me the liver", "what did RADAR flag above 50%?", "what am I looking at?"). It can move the viewer.
- Export scores as CSV or JSON, and the organ mask as NIfTI.

A score is how close an organ looks to a finding's text, not a calibrated probability. The 50% line is for display.

## How it works

The browser uploads to a FastAPI server, which stores the scan on a Modal Volume. A Modal function on an L4 GPU runs DAMO's own PyTorch code and checkpoint, unedited, and writes the scores and the organ mask back to the Volume. The viewer (NiiVue) reads them through the API. The chat panel is persona, backed by any OpenAI-compatible model. `LLM_PROVIDER` picks OpenRouter (the default), a local Ollama or another endpoint.

## Results

| Check | Result |
|---|---|
| Our wrapper against DAMO's own script, same GPU | identical, all 146 scores |
| DAMO's published demo scores | max difference 0.0064, all in the oesophagus |
| A Merlin scan against the browser port (radar-web) | same positives, 84.8 / 62.1 / 60.0% |
| Time on an L4 | 16 to 28 s per scan, about 15 s model load when cold |
| Cost | about 3 cents per scan, including the idle minutes |

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

Any NVIDIA machine can score instead of Modal. Set `GPU_BACKEND=worker`, create a worker token on the jobs page, build `worker/docker/Dockerfile` and run `docker run --gpus all -v /workspace:/workspace -e RADAR_DESK_URL=<app url> -e RADAR_WORKER_TOKEN=<token> radar-worker`; the weights are fetched to `/workspace/radar-weights` on first start.

`uv run python -m radar_desk.compute runpod|modal|status|stop` switches scoring between Modal and a RunPod pod. It edits `GPU_BACKEND` in `.env`, you restart the app, and `runpod` then starts the tunnel, the worker token and the pod. It needs `WORKER_IMAGE` and the `RUNPOD_*` keys from `.env.example`.

Tests are `uv run pytest` and, in `web/`, `npm test`.

## Licence

The whole repository is CC BY-NC-SA 4.0, because it includes and adapts RADAR's code. Model and weights by Alibaba DAMO Academy ([huggingface.co/radar-generalist/RADAR](https://huggingface.co/radar-generalist/RADAR)). Paper: "An expert-level generalist AI for abdominal CT diagnosis", Science 393(6817), eaec6129, doi [10.1126/science.aec6129](https://doi.org/10.1126/science.aec6129). See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
