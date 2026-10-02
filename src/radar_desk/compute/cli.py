"""A front end to the app's Compute choice (plan.md, Compute switch, B, and RunPod Serverless).

The app starts and stops the pod and its tunnel itself; this command only calls the owner routes. OWNER_TOKEN
comes from the environment or `.env` in the working directory (the environment wins) and is never printed.
Exit codes are 0 ok and 1 when the app refuses (the detail is printed) or does not answer.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from radar_desk.compute import envfile
from radar_desk.compute.desk import Desk, DeskError

DEFAULT_APP = "http://127.0.0.1:8000"


@dataclass
class Deps:
    env_path: Path = field(default_factory=lambda: Path(".env"))
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    app_url: str = DEFAULT_APP
    app_client: httpx.Client | None = None


def out(line: str) -> None:
    print(line, flush=True)


def err(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def compute_lines(c: dict) -> list[str]:
    fixed = "" if c["changeable"] else ", fixed at start-up"
    url = f", {c['public_url']}" if c.get("public_url") else ""
    lines = [f"compute: mode {c['mode']}{fixed}, tunnel {c['tunnel_mode']}{url}"]
    runpod = c.get("runpod") or {}
    if isinstance(runpod.get("gpus"), list) and runpod["gpus"]:
        where = f" in {runpod['datacenter']}" if runpod.get("datacenter") else ""
        lines.append(f"gpus: {', '.join(runpod['gpus'])}{where}")
    pod = c.get("pod")
    if pod:
        cost = "?" if pod.get("cost_per_hr") is None else f"{pod['cost_per_hr']:.3f}"
        up = "not deployed yet" if pod.get("up_s") is None else f"up {int(pod['up_s']) // 60} min"
        lines.append(f"pod: {pod.get('runpod_id') or 'none yet'}, {pod['phase']}, {pod.get('gpu') or '?'}, "
                     f"${cost}/h, image {pod.get('image') or '?'}, {up}")
    else:
        lines.append("pod: none")
    serverless = c.get("serverless")
    if serverless is not None:
        lines.append(serverless_line(serverless))
    if c.get("problem"):
        lines.append(f"problem: {c['problem']}")
    lines.append(f"spend: ${c['spend_month_usd']:.2f} of ${c['budget_usd']:.2f} in {c['month']}")
    storage = c.get("storage")
    if storage:
        modes = ", ".join(f"{m} {'yes' if v['available'] else 'no'}" for m, v in storage["modes"].items())
        lines.append(f"storage: {storage['name']}, {storage['backend']}; {modes}")
    return lines


def serverless_line(s: dict) -> str:
    if not s.get("configured"):
        return "serverless: not configured"
    head = f"serverless: endpoint {s.get('endpoint_id')}, {', '.join(s.get('gpus') or [])}"
    job = s.get("job")
    if not job:
        return f"{head}, no job"
    return f"{head}, job {job['job_id'][:8]} {job.get('status') or 'status pending'} since {job.get('submitted_at')}"


def cmd_runpod(desk: Desk, args: argparse.Namespace) -> int:
    c = desk.set_mode("worker")
    if args.start:
        c = desk.pod_start()
    for line in compute_lines(c):
        out(line)
    return 0


def cmd_modal(desk: Desk, args: argparse.Namespace) -> int:
    for line in compute_lines(desk.set_mode("modal")):
        out(line)
    return 0


def cmd_serverless(desk: Desk, args: argparse.Namespace) -> int:
    for line in compute_lines(desk.set_mode("serverless")):
        out(line)
    return 0


def cmd_stop(desk: Desk, args: argparse.Namespace) -> int:
    for line in compute_lines(desk.pod_stop()):
        out(line)
    return 0


def cmd_status(desk: Desk, args: argparse.Namespace) -> int:
    h = desk.health()
    if not h:
        err(f"app: does not answer at {desk.base}")
        return 1
    out(f"app: backend {h.get('backend')}, version {h.get('version')}")
    for line in compute_lines(desk.compute()):
        out(line)
    workers = desk.workers()
    if not workers:
        out("workers: none")
    for w in workers:
        out(f"worker: {w.get('id')}, {'online' if w.get('online') else 'offline'}, job {w.get('job_id') or 'none'}")
    return 0


COMMANDS = {"runpod": cmd_runpod, "modal": cmd_modal, "serverless": cmd_serverless, "status": cmd_status,
            "stop": cmd_stop}


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--app", help=f"the app's URL (default {DEFAULT_APP})")
    p = argparse.ArgumentParser(prog="python -m radar_desk.compute",
                                description="Choose where the app scores: Modal, a RunPod pod or RunPod serverless.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("runpod", parents=[common], help="score on RunPod; the app starts a pod when work is queued")
    r.add_argument("--start", action="store_true", help="start the pod now")
    sub.add_parser("modal", parents=[common], help="score on Modal; an idle pod is stopped by the app")
    sub.add_parser("serverless", parents=[common],
                   help="score on the RunPod serverless endpoint; an idle pod is stopped by the app")
    sub.add_parser("stop", parents=[common], help="stop the pod now")
    sub.add_parser("status", parents=[common], help="show the app, the Compute choice, the pod and the workers")
    return p


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parser().parse_args(argv)
    deps = deps or Deps()
    app_url = args.app or deps.app_url
    token = deps.environ.get("OWNER_TOKEN") or envfile.read_env(deps.env_path).get("OWNER_TOKEN")
    if not token:
        err("OWNER_TOKEN is not set in .env or the environment")
        return 1
    try:
        return COMMANDS[args.command](Desk(app_url, token, deps.app_client), args)
    except DeskError as exc:
        err(f"app: {exc}")
        return 1
    except httpx.HTTPError as exc:
        err(f"app: does not answer at {app_url} ({type(exc).__name__})")
        return 1
