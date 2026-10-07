"""Create, show, update or delete the RunPod Serverless endpoint the app submits to (plan.md, The endpoint).

The app never creates or edits endpoints; this script does it once, by hand, over RunPod's v2 REST API.
Settings come from the environment or `.env` in the working directory, like the app:

    uv run python scripts/runpod_endpoint.py create [--force]  # prints RUNPOD_ENDPOINT_ID=<id> last
    uv run python scripts/runpod_endpoint.py show
    uv run python scripts/runpod_endpoint.py update            # image, workers, timeout and env
    uv run python scripts/runpod_endpoint.py delete [--force]

Every command needs RUNPOD_API_KEY. `create` also needs WORKER_IMAGE, RUNPOD_REGISTRY_AUTH_ID and
RUNPOD_VOLUME_ID; `show`, `update` and `delete` need RUNPOD_ENDPOINT_ID, and `update` needs WORKER_IMAGE.
WORKER_IMAGE must carry a pinned tag (such as `ghcr.io/<owner>/radar-worker:0.3`), never `:latest`, because
RunPod hosts keep serving a cached `:latest` until their workers are replaced.

`create` reads RunPod's GPU catalog, keeps the cards in the RUNPOD_SERVERLESS_GPUS pools and excludes the
Blackwell ones (the worker's torch 2.5.1/cu124 cannot drive them), with the same rule as compute/runpod.py
applied to each card's id and name. It refuses while RUNPOD_ENDPOINT_ID is set unless `--force`.
`delete` also cancels queued and running jobs on RunPod, so it refuses while any are in flight unless
`--force`. Management calls go to api.runpod.io, `/health` to the jobs host api.runpod.ai.

The worker runs as an unprivileged user; with STORAGE_BACKEND=runpod_volume the env also carries
RADAR_RUN_AS_ROOT=1, because results go into job folders the app made on the volume. Run `update` after
changing STORAGE_BACKEND.

Exit codes are 0 ok, 1 refused or a RunPod error, 2 invalid settings. The API key is never printed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from radar_desk.compute.runpod import BLACKWELL, RunPodApi, RunPodError
from radar_desk.config import ConfigError, Settings, load_settings
from radar_desk.storage import storage_backend

JOBS_URL = "https://api.runpod.ai/v2"
NAME = "radar-desk"


def _print_out(line: str) -> None:
    print(line, flush=True)


def _print_err(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


@dataclass
class Deps:
    settings: Settings | None = None  # None means load_settings()
    client: httpx.Client | None = None
    out: Callable[[str], None] = field(default=_print_out)
    err: Callable[[str], None] = field(default=_print_err)


class Refused(Exception):
    """The script will not go on; the message says why and names the setting when one is at fault."""


def is_blackwell(gpu_id: str) -> bool:
    return any(b in "".join(gpu_id.lower().split()) for b in BLACKWELL)


def need(settings: Settings, *names: str) -> None:
    missing = [n.upper() for n in names if not getattr(settings, n)]
    if missing:
        raise Refused(f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set")


def pinned_image(settings: Settings) -> str:
    image = settings.worker_image or ""
    last = image.rsplit("/", 1)[-1]
    if "@" not in last and (":" not in last or last.endswith(":latest")):
        raise Refused(f"WORKER_IMAGE {image} has no pinned tag; use a worker-vX.Y tag such as "
                      "ghcr.io/<owner>/radar-worker:0.3 (RunPod serves a cached :latest until workers are replaced)")
    return image


def worker_env(image: str, settings: Settings) -> dict[str, str]:
    env = {"RADAR_WEIGHTS_DIR": "/runpod-volume/radar-weights", "RADAR_DATA_ROOT": "/runpod-volume",
           "RADAR_IMAGE": image, "RADAR_DEVICE": "auto"}
    if storage_backend(settings) == "runpod_volume":
        env["RADAR_RUN_AS_ROOT"] = "1"  # results go into job folders the app made on the volume
    return env


def workers(settings: Settings) -> dict:
    return {"min": 0, "max": 1, "idleTimeout": settings.runpod_serverless_idle_s}


def pool_members(api: RunPodApi, pools: list[str], out: Callable[[str], None]) -> list[str]:
    """The Blackwell ids in `pools`, after printing each pool's kept and excluded ids.

    Refuses a pool the catalog does not know and a pool left empty by the exclusion.
    """
    catalog = (api.call("GET", "/catalog/gpus") or {}).get("gpus") or []
    excluded: list[str] = []
    problems: list[str] = []
    for pool in pools:
        cards = [g for g in catalog if isinstance(g, dict) and g.get("pool") == pool and g.get("id")]
        if not cards:
            problems.append(f"RUNPOD_SERVERLESS_GPUS: pool {pool} is not in RunPod's GPU catalog")
            continue
        dropped = [g["id"] for g in cards if is_blackwell(g["id"]) or is_blackwell(str(g.get("name") or ""))]
        kept = [g["id"] for g in cards if g["id"] not in dropped]
        out(f"pool {pool}: kept {', '.join(kept) or 'none'}; excluded {', '.join(dropped) or 'none'}")
        if not kept:
            problems.append(f"RUNPOD_SERVERLESS_GPUS: pool {pool} holds no usable GPU after the Blackwell exclusion")
        excluded += [i for i in dropped if i not in excluded]
    if problems:
        raise Refused("; ".join(problems))
    return excluded


def cmd_create(api: RunPodApi, settings: Settings, args: argparse.Namespace, deps: Deps) -> int:
    need(settings, "worker_image", "runpod_registry_auth_id", "runpod_volume_id")
    image = pinned_image(settings)
    if settings.runpod_endpoint_id and not args.force:
        raise Refused(f"RUNPOD_ENDPOINT_ID is already set ({settings.runpod_endpoint_id}); delete that endpoint first, "
                      "or pass --force to create another")
    pools = settings.runpod_serverless_gpu_list
    excluded = pool_members(api, pools, deps.out)
    gpu: dict = {"pools": pools, "excludedTypes": excluded, "count": 1, "minCudaVersion": "12.4"}
    if not excluded:
        del gpu["excludedTypes"]
    body = {"name": NAME, "type": "QUEUE",
            "image": image, "registry": settings.runpod_registry_auth_id, "disk": 20,
            "cmd": ["python3", "-u", "-m", "radar_worker.serverless"],
            "env": worker_env(image, settings),
            "gpu": gpu,
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
            "workers": workers(settings),
            "timeout": settings.gpu_timeout_s * 1000, "flashboot": "FLASHBOOT",
            "dataCenterIds": [settings.runpod_datacenter], "networkVolumes": [settings.runpod_volume_id]}
    created = api.call("POST", "/serverless", json=body)
    if not isinstance(created, dict) or not created.get("id"):
        raise RunPodError("RunPod answered the create without an endpoint id")
    deps.out(f"created endpoint {created['id']} ({created.get('name') or NAME})")
    deps.out(f"RUNPOD_ENDPOINT_ID={created['id']}")
    return 0


def health(settings: Settings, client: httpx.Client | None) -> dict:
    key = settings.runpod_api_key.get_secret_value()
    jobs_api = RunPodApi(key, client, base_url=f"{JOBS_URL}/{settings.runpod_endpoint_id}")
    body = jobs_api.call("GET", "/health")
    return body if isinstance(body, dict) else {}


def _counts(d: dict | None, names: tuple[str, ...]) -> str:
    d = d if isinstance(d, dict) else {}
    return ", ".join(f"{n} {d.get(n, '?')}" for n in names)


def cmd_show(api: RunPodApi, settings: Settings, args: argparse.Namespace, deps: Deps) -> int:
    need(settings, "runpod_endpoint_id")
    eid = settings.runpod_endpoint_id
    ep = api.call("GET", f"/serverless/{eid}") or {}
    w = ep.get("workers") or {}
    g = ep.get("gpu") or {}
    deps.out(f"endpoint {ep.get('id', eid)} ({ep.get('name', '?')}), image {ep.get('image', '?')}")
    deps.out(f"workers: min {w.get('min', '?')}, max {w.get('max', '?')}, idleTimeout {w.get('idleTimeout', '?')} s; "
             f"timeout {ep.get('timeout', '?')} ms; flashboot {ep.get('flashboot', '?')}")
    deps.out(f"gpu: pools {', '.join(g.get('pools') or []) or 'none'}; "
             f"excluded {', '.join(g.get('excludedTypes') or []) or 'none'}")
    deps.out(f"dataCenterIds {', '.join(ep.get('dataCenterIds') or []) or 'none'}; "
             f"networkVolumes {', '.join(ep.get('networkVolumes') or []) or 'none'}")
    summary = (api.call("GET", f"/serverless/{eid}/workers") or {}).get("summary")
    deps.out("worker summary: " + _counts(summary, ("running", "idle", "initializing", "throttled", "unhealthy",
                                                    "total")))
    h = health(settings, deps.client)
    deps.out("health workers: " + _counts(h.get("workers"), ("idle", "running")))
    deps.out("health jobs: " + _counts(h.get("jobs"), ("inQueue", "inProgress", "completed", "failed", "retried")))
    return 0


def cmd_update(api: RunPodApi, settings: Settings, args: argparse.Namespace, deps: Deps) -> int:
    need(settings, "runpod_endpoint_id", "worker_image")
    image = pinned_image(settings)
    body = {"image": image, "workers": workers(settings), "timeout": settings.gpu_timeout_s * 1000,
            "env": worker_env(image, settings)}
    ep = api.call("PATCH", f"/serverless/{settings.runpod_endpoint_id}", json=body) or {}
    deps.out(f"updated endpoint {ep.get('id') or settings.runpod_endpoint_id}: image {image}, "
             f"workers max 1, idleTimeout {settings.runpod_serverless_idle_s} s, timeout {body['timeout']} ms")
    return 0


def cmd_delete(api: RunPodApi, settings: Settings, args: argparse.Namespace, deps: Deps) -> int:
    need(settings, "runpod_endpoint_id")
    eid = settings.runpod_endpoint_id
    jobs = health(settings, deps.client).get("jobs") or {}
    in_flight = int(jobs.get("inQueue") or 0) + int(jobs.get("inProgress") or 0)
    if in_flight and not args.force:
        raise Refused(f"endpoint {eid} has {in_flight} job(s) queued or running; a delete cancels them. "
                      "Wait for them, or pass --force")
    api.call("DELETE", f"/serverless/{eid}")
    deps.out(f"deleted endpoint {eid}; remove RUNPOD_ENDPOINT_ID from .env")
    return 0


COMMANDS = {"create": cmd_create, "show": cmd_show, "update": cmd_update, "delete": cmd_delete}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts/runpod_endpoint.py",
                                description="Manage the RunPod Serverless endpoint the app submits to.")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create", help="create the endpoint and print RUNPOD_ENDPOINT_ID=<id>")
    c.add_argument("--force", action="store_true", help="create even when RUNPOD_ENDPOINT_ID is already set")
    sub.add_parser("show", help="print the endpoint, its workers and its /health")
    sub.add_parser("update", help="set the image, workers, timeout and env from the settings")
    d = sub.add_parser("delete", help="delete the endpoint; refused while jobs are in flight")
    d.add_argument("--force", action="store_true", help="delete even with jobs queued or running (cancels them)")
    return p


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parser().parse_args(argv)
    deps = deps or Deps()
    try:
        settings = deps.settings or load_settings()
    except ConfigError as exc:
        deps.err(str(exc))
        return 2
    try:
        need(settings, "runpod_api_key")
        api = RunPodApi(settings.runpod_api_key.get_secret_value(), deps.client)
        return COMMANDS[args.command](api, settings, args, deps)
    except (Refused, RunPodError) as exc:
        deps.err(f"refused: {exc}" if isinstance(exc, Refused) else str(exc))
        return 1
    except httpx.HTTPError as exc:
        deps.err(f"RunPod does not answer ({type(exc).__name__})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
