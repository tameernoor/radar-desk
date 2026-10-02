# radar-desk

radar-desk runs Alibaba DAMO Academy's RADAR abdominal CT model on uploaded research scans and shows its 146 finding scores next to the image, with RADAR's own organ mask on top. A chat panel can answer questions about the scores and move the viewer.

The Radar model is currently intended for research purposes only. Use public and research scans only; this is not a medical device.

Model: RADAR by Alibaba DAMO Academy, CC BY-NC-SA 4.0. Paper: "An expert-level generalist AI for abdominal CT diagnosis", Science 393(6817), eaec6129, doi [10.1126/science.aec6129](https://doi.org/10.1126/science.aec6129). Weights: [huggingface.co/radar-generalist/RADAR](https://huggingface.co/radar-generalist/RADAR).

The whole repository is CC BY-NC-SA 4.0 because it vendors and adapts RADAR's code. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.

## Run it locally

You need `uv` and Node. From the repo root, run these.

```sh
uv sync
cd web && npm install && npm run build && cd ..
cp .env.example .env        # then set OWNER_TOKEN and SESSION_SECRET to long random strings
uv run python -m radar_desk
```

Open http://127.0.0.1:8000 and log in with your `OWNER_TOKEN`.

The GPU backend defaults to `fake`. It makes up a result from the scan header (blob-shaped organs, scores derived from the file hash) so the viewer and chat work without a GPU. Fake results are marked as fake in the page, carry `gpu: fake` in their versions, and are left out of the CSV export once a real backend is set.

For a demo scan without uploading anything, run this while the server is stopped (the fake backend keeps its calls in memory, so a running server would fail the job).

```sh
uv run python scripts/seed_dev.py
```

`uv run python scripts/export_all.py [out.csv]` writes the scores of every finished job to one CSV (default `exports/scores-<date>.csv` under `DATA_DIR`).

### Run against Modal from your Mac

Real scoring from a laptop needs no bucket. Scans and artefacts go on a Modal Volume, the GPU function reads and writes it directly, and the browser reaches it through the API. Deploy the worker with the Volume's name once, then start the server with these variables.

```sh
RADAR_GPU=L4,L40S MODAL_DATA_VOLUME=radar-data uv run modal deploy worker/modal_app.py

STORAGE_BACKEND=modal_volume GPU_BACKEND=modal MODAL_DATA_VOLUME=radar-data \
MODAL_TOKEN_ID=ak-... MODAL_TOKEN_SECRET=as-... uv run python -m radar_desk
```

The server variables can go in `.env` instead, but `modal deploy` reads `MODAL_DATA_VOLUME` and `RADAR_GPU` from the shell only, so set them on the deploy line. `MODAL_DATA_VOLUME` must match on both sides; the Volume is created on first use.

## Tests

```sh
uv run pytest
cd web && npm test
```

`npm test` runs the Playwright smoke in your installed Google Chrome. It seeds a scan into a data folder under the OS temp dir and starts the API with the fake backend on port 8000, or reuses a server already listening there (that one needs a finished job, so seed it first). Tests that need the real scans skip unless they are in `RADAR_FIXTURE_DIR`; see `fixtures/README.md`.

## Real scoring on Modal

Scoring runs on a Modal GPU function that scales to zero. The weights live on a Modal Volume called `radar-weights`, laid out as in `worker/weights.json`. Three things go on it. `checkpoint_radar_pretrain.pth` and the `bert-base-chinese/` folder come from [huggingface.co/radar-generalist/RADAR](https://huggingface.co/radar-generalist/RADAR), and `infer_text_embedding_radar.pt` is vendored at `worker/vendor/damo-radar/ckpt/`.

With a Modal token set up (`uv run modal token new`), run these from the repo root.

```sh
uv run modal volume create radar-weights
uv run modal volume put radar-weights checkpoint_radar_pretrain.pth /checkpoint_radar_pretrain.pth
uv run modal volume put radar-weights bert-base-chinese /bert-base-chinese
uv run modal volume put radar-weights worker/vendor/damo-radar/ckpt/infer_text_embedding_radar.pt /infer_text_embedding_radar.pt

uv run modal run scripts/weights_check.py                  # sizes and sha256 must match weights.json
uv run modal run scripts/modal_spike.py --imports-only     # the image installs and imports, CPU only
uv run modal run scripts/parity.py --path AC423ccbe.nii.gz # our wrapper against upstream, on a GPU
RADAR_GPU=L4 uv run modal deploy worker/modal_app.py
```

`AC423ccbe.nii.gz` is DAMO's demo case from `data/demo_cases/` in the upstream repo. Then set `GPU_BACKEND=modal`, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the server's environment. The server holds a job when its worst-case cost would take the month's estimated spend past `GPU_MONTHLY_BUDGET_USD`.

## Chat

Set `LLM_API_KEY` and `CHAT_MODEL` (a model name on the provider). Any OpenAI-compatible provider works; `LLM_BASE_URL` defaults to OpenRouter. Without them the app runs normally and the chat panel answers that chat is not configured.

## Deploy

The API runs on Fly.io as one machine with a volume for SQLite, built from `Dockerfile` and configured in `fly.toml`; the GPU stays on Modal and scans live in a Tigris bucket. Set the secrets with `fly secrets set` for `OWNER_TOKEN`, `SESSION_SECRET`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `LLM_API_KEY`, `CHAT_MODEL`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (the Tigris bucket keys; `S3_BUCKET`, `S3_ENDPOINT_URL` and `S3_REGION` are already in `fly.toml`), then run `fly deploy`. The bucket stays private and needs a CORS rule allowing PUT and GET from `https://radar-desk.fly.dev`, because the browser uploads straight to it.

## Environment variables

| Variable | Default | What it does |
|---|---|---|
| `OWNER_TOKEN` | required | The login token for the one owner. |
| `SESSION_SECRET` | required | Signs the session cookie. |
| `PUBLIC_BASE_URL` | `http://127.0.0.1:8000` | Where the browser reaches the API; used in local storage URLs. |
| `APP_HOSTNAME` | unset | When set, only this Host header (and localhost) is accepted. |
| `DATA_DIR` | `./data` | SQLite database and the local object store. |
| `STORAGE_BACKEND` | unset | `local`, `s3` or `modal_volume`; unset means `s3` when `S3_BUCKET` is set, else `local`. |
| `MODAL_DATA_VOLUME` | `radar-data` | Modal Volume for scans and artefacts with `STORAGE_BACKEND=modal_volume`; the worker must be deployed with the same name. |
| `S3_BUCKET` | unset | Bucket name; with no `STORAGE_BACKEND`, unset means the local storage adapter. |
| `S3_ENDPOINT_URL` | unset | S3 endpoint, for example Tigris. |
| `S3_REGION` | unset | Bucket region. |
| `AWS_ACCESS_KEY_ID` | unset | Bucket access key. |
| `AWS_SECRET_ACCESS_KEY` | unset | Bucket secret key. |
| `GPU_BACKEND` | `fake` | `fake` for synthetic results, `modal` for real scoring. |
| `MODAL_TOKEN_ID` | unset | Modal token id, needed with `GPU_BACKEND=modal`. |
| `MODAL_TOKEN_SECRET` | unset | Modal token secret. |
| `MODAL_APP_NAME` | `radar-desk` | Name of the deployed Modal app. |
| `MODAL_FUNCTION_NAME` | `score` | Name of the scoring function in that app. |
| `RADAR_GPU` | `L4` | GPU type, or an ordered comma-separated list Modal falls back through. |
| `GPU_TIMEOUT_S` | `1800` | The scoring function's timeout as the server counts it, for the budget check and for cancelling a stuck call. |
| `GPU_SCALEDOWN_WINDOW_S` | `120` | How long an idle container stays up, counted in the cost estimate. |
| `GPU_MONTHLY_BUDGET_USD` | `10` | Monthly GPU budget; a job is held when its worst case would go past it. |
| `GPU_POLL_INTERVAL_S` | `10` | How often the poller checks queued and running jobs. |
| `LLM_API_KEY` | unset | Turns the chat on. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint for the chat model. |
| `CHAT_MODEL` | unset | Model name for the chat. |
| `MAX_UPLOAD_BYTES` | `314572800` | Largest accepted upload (300 MB). |

## Layout

- `src/radar_desk/` is the API server, poller, chat agent and storage adapters.
- `web/` is the front end (vite, NiiVue viewer, persona chat) and the Playwright smoke.
- `worker/` is the Modal app, the scoring wrapper and the vendored DAMO code (`worker/vendor/`, unedited).
- `scripts/` holds the seed, export, weights check, spike and parity scripts.
- `fixtures/` has the scan manifest and the reference results.
- `tests/` is the Python test suite.
- `docs/` holds the design and the plan.

## Design and plan

[docs/design.md](docs/design.md) explains what the app does and why. [docs/plan.md](docs/plan.md) is the build plan with task status.
