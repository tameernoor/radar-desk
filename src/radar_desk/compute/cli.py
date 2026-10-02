"""The compute switch: move scoring between Modal and one RunPod pod (plan.md, Compute switch).

Reads `.env` in the working directory (the process environment wins) and never prints a value from it.
State is DATA_DIR/compute.json; every `runpod` step is idempotent, so a rerun continues where it stopped.
Exit codes are 0 ok, 1 error, 2 the app is not ready.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from radar_desk.compute import envfile, state, tunnel
from radar_desk.compute.desk import Desk, DeskError
from radar_desk.compute.runpod import Pod, PodSpec, RunPod, RunPodError

DEFAULT_APP = "http://127.0.0.1:8000"
DEFAULTS = {"RUNPOD_DATACENTER": "EU-RO-1", "DATA_DIR": "./data"}
KEYS = ("OWNER_TOKEN", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID", "RUNPOD_REGISTRY_AUTH_ID", "RUNPOD_DATACENTER",
        "WORKER_IMAGE", "DATA_DIR")
RUNPOD_REQUIRED = ("OWNER_TOKEN", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID", "RUNPOD_REGISTRY_AUTH_ID", "WORKER_IMAGE")
POLL_S = 2
TUNNEL_WAIT_S = 90
GONE_CHECKS = 10
RESTART = "uv run python -m radar_desk"
RERUN = "uv run python -m radar_desk.compute runpod"
STOP = "uv run python -m radar_desk.compute stop"
POD_NAME = PodSpec.name


class Fail(Exception):
    """End the command with this message on stderr and this exit code."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Deps:
    env_path: Path = field(default_factory=lambda: Path(".env"))
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    app_url: str = DEFAULT_APP
    app_client: httpx.Client | None = None
    runpod_client: httpx.Client | None = None
    cloudflared: str = "cloudflared"
    probe: Callable[[str], dict | None] = tunnel.public_health
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time


def out(line: str) -> None:
    print(line, flush=True)


def err(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class Ctx:
    """Settings, paths and clients for one run."""

    def __init__(self, deps: Deps, app_url: str) -> None:
        self.deps = deps
        self.env_path = Path(deps.env_path)
        cfg = {k: v for k, v in envfile.read_env(self.env_path).items() if k in KEYS and v}
        cfg.update({k: v for k, v in deps.environ.items() if k in KEYS and v})
        self.cfg = {**DEFAULTS, **cfg}
        data_dir = Path(self.cfg["DATA_DIR"])
        if not data_dir.is_absolute():
            data_dir = self.env_path.parent / data_dir
        self.state_path = data_dir / "compute.json"
        self.log_path = data_dir / "cloudflared.log"
        self.state = state.load(self.state_path)
        self.app_url = app_url
        self.desk = Desk(app_url, self.cfg.get("OWNER_TOKEN", ""), deps.app_client)
        self._runpod: RunPod | None = None
        self._lock = None

    def lock(self) -> None:
        """Hold an exclusive lock on the state for the rest of the run."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.state_path.parent / "compute.lock", "w")  # noqa: SIM115
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Fail("another compute command is running") from None

    def unlock(self) -> None:
        if self._lock:
            self._lock.close()
            self._lock = None

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if not self.cfg.get(k)]
        if missing:
            raise Fail(f"not set in .env or the environment: {', '.join(missing)}")

    @property
    def runpod(self) -> RunPod:
        if self._runpod is None:
            self._runpod = RunPod(self.cfg["RUNPOD_API_KEY"], self.deps.runpod_client, self.deps.sleep)
        return self._runpod

    def save(self) -> None:
        state.save(self.state_path, self.state)

    def set_backend(self, backend: str) -> None:
        changed = envfile.set_key(self.env_path, "GPU_BACKEND", backend)
        out(f".env: GPU_BACKEND set to {backend}" if changed else f".env: GPU_BACKEND already {backend}")

    def wait_for(self, check: Callable[[], bool], timeout_s: float) -> bool:
        deadline = self.deps.clock() + timeout_s
        while not check():
            if self.deps.clock() >= deadline:
                return False
            self.deps.sleep(POLL_S)
        return True


def fmt_pod(pod) -> str:
    uptime = "not running yet" if pod.uptime_s is None else f"up {pod.uptime_s} s"
    return f"{pod.id} ({pod.name}) {pod.gpu}, {pod.desired_status}, {uptime}, ${pod.cost_per_hr:.3f}/h, image {pod.image or 'unknown'}"


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# runpod


def adopt(ctx: Ctx, listed: list[Pod]) -> bool:
    """Take a radar-worker pod that RunPod lists into the state, so this tool never starts a second one."""
    pod = next((p for p in listed if p.name == POD_NAME), None)
    if not pod:
        return False
    ctx.state["pod"] = {"id": pod.id, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr,
                        "started_at": utc(ctx.deps.clock())}
    ctx.save()
    out(f"pod: adopted {pod.id}, started outside this state file; stop it with {STOP}")
    return True


def cmd_runpod(ctx: Ctx, args: argparse.Namespace) -> int:
    ctx.require(*RUNPOD_REQUIRED)

    # 1. .env
    ctx.set_backend("worker")

    # 2. one pod from this tool, never a second
    listed = ctx.runpod.pods()
    mine = ctx.state["pod"]
    if mine:
        pod = next((p for p in listed if p.id == mine["id"]), None)
        if pod:
            out(f"pod already running: {fmt_pod(pod)}")
            t = ctx.state["tunnel"]
            if pod.desired_status != "RUNNING" or not (t and tunnel.alive(t["pid"], t["binary"])):
                raise Fail(f"the tunnel it was given is gone; run: {STOP}, then rerun")
            return 0
        ctx.state["pod"] = None
        ctx.save()
    if adopt(ctx, listed):
        return 0

    # 3. the app must be on the worker backend
    def app_on_worker() -> bool:
        h = ctx.desk.health()
        return bool(h) and h.get("backend") == "worker"

    if not (app_on_worker() if not args.wait else ctx.wait_for(app_on_worker, args.wait_timeout)):
        h = ctx.desk.health()
        now = f"is on backend {h.get('backend')}" if h else "does not answer /health"
        raise Fail(f"The app at {ctx.app_url} {now}, it must be on worker.\n"
                   f"1. Restart the app: {RESTART}\n"
                   f"2. Then rerun: {RERUN} (or add --wait to wait for it)", code=2)

    # 4. tunnel
    t = ctx.state["tunnel"]
    if t and tunnel.alive(t["pid"], t["binary"]):
        out(f"tunnel: reusing {t['url']} (pid {t['pid']})")
    else:
        port = urlparse(ctx.app_url).port or 80
        new = tunnel.start(port, ctx.log_path, ctx.deps.cloudflared)
        ctx.state["tunnel"] = t = asdict(new)
        ctx.save()
        out(f"tunnel: started {t['url']} (pid {t['pid']}, log {t['log']})")

    def tunnel_on_worker() -> bool:
        h = ctx.deps.probe(t["url"])
        return bool(h) and h.get("backend") == "worker"

    if not ctx.wait_for(tunnel_on_worker, TUNNEL_WAIT_S):
        raise Fail(f"{t['url']}/health did not answer backend worker in {TUNNEL_WAIT_S} s; the tunnel is "
                   f"still up, see {t['log']}, then rerun")
    out(f"tunnel: {t['url']} answers backend worker")

    # 5. pods this tool did not start
    others = [p for p in listed if p.name != POD_NAME]
    if others and not args.allow_other_pods:
        for p in others:
            err(f"other pod: {fmt_pod(p)}")
        raise Fail("RunPod lists pods this tool did not start; stop them or pass --allow-other-pods")

    # 6. worker token, the plaintext only goes into the pod's env
    if ctx.state["token_id"]:
        ctx.desk.revoke_token(ctx.state["token_id"])
        out(f"token: revoked {ctx.state['token_id']} from an earlier attempt")
        ctx.state["token_id"] = None
        ctx.save()
    token_id, plaintext = ctx.desk.create_token(f"runpod {utc(ctx.deps.clock())}")
    worker_id = f"runpod-{secrets.token_hex(3)}"
    ctx.state.update(token_id=token_id, worker_id=worker_id)
    ctx.save()
    out(f"token: created {token_id} for worker {worker_id}")

    # 7. pod
    image = ctx.cfg["WORKER_IMAGE"]
    spec = PodSpec(image=image, registry_auth_id=ctx.cfg["RUNPOD_REGISTRY_AUTH_ID"],
                   datacenter=ctx.cfg["RUNPOD_DATACENTER"], volume_id=ctx.cfg["RUNPOD_VOLUME_ID"],
                   env={"RADAR_DESK_URL": t["url"], "RADAR_WORKER_TOKEN": plaintext,
                        "RADAR_WORKER_ID": worker_id, "RADAR_IMAGE": image})
    if adopt(ctx, ctx.runpod.pods()):
        return 0
    try:
        pod = ctx.runpod.deploy(spec)
    except Exception as exc:  # noqa: BLE001, any error text may echo the pod's input
        text = str(exc).replace(plaintext, "<worker token>")
        for key in RUNPOD_REQUIRED:
            text = text.replace(ctx.cfg[key], f"<{key}>")
        raise Fail(f"error: {text}") from None
    ctx.state["pod"] = {"id": pod.id, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr,
                        "started_at": utc(ctx.deps.clock())}
    ctx.save()
    out(f"pod: deployed {pod.id} on {pod.gpu} at ${pod.cost_per_hr:.3f}/h, image {pod.image or image}")
    out("stop it with: uv run python -m radar_desk.compute stop")
    return 0


# stop and modal


def stop_all(ctx: Ctx, to_modal: bool) -> int:
    s = ctx.state
    if s["pod"]:
        ctx.require("RUNPOD_API_KEY")
    if s["token_id"]:
        ctx.require("OWNER_TOKEN")
    if to_modal:
        ctx.set_backend("modal")
    if not (s["pod"] or s["tunnel"] or s["token_id"]):
        out("nothing to stop")
        print_spend(ctx)
        return 0
    code = 0
    if s["pod"]:
        pod_id = s["pod"]["id"]
        deleted = ctx.runpod.delete(pod_id)
        out(f"pod: deleted {pod_id}" if deleted else f"pod: {pod_id} was already gone")
        for i in range(GONE_CHECKS):
            if ctx.runpod.pod(pod_id) is None:
                break
            if i == GONE_CHECKS - 1:
                raise Fail(f"pod {pod_id} is still listed by RunPod; check the console and rerun stop")
            ctx.deps.sleep(POLL_S)
        s["pod"] = None
        ctx.save()
    if s["tunnel"]:
        t = s["tunnel"]
        if tunnel.alive(t["pid"], t["binary"]):
            tunnel.stop(t["pid"], t["binary"])
            out(f"tunnel: stopped pid {t['pid']}")
        else:
            out(f"tunnel: pid {t['pid']} was not running")
        s["tunnel"] = None
        ctx.save()
    if s["token_id"]:
        try:
            revoked = ctx.desk.revoke_token(s["token_id"])
        except (httpx.HTTPError, DeskError) as exc:
            err(f"token: could not revoke {s['token_id']} ({exc}); revoke it on the jobs page, or rerun stop "
                "once the app is up")
            code = 1
        else:
            out(f"token: revoked {s['token_id']}" if revoked else f"token: {s['token_id']} was already gone")
            s["token_id"] = None
            s["worker_id"] = None
            ctx.save()
    print_spend(ctx)
    return code


def print_spend(ctx: Ctx) -> None:
    if ctx.cfg.get("RUNPOD_API_KEY"):
        try:
            out(f"runpod spend: ${ctx.runpod.spend_per_hr():.3f}/h")
        except (httpx.HTTPError, RunPodError) as exc:
            out(f"runpod spend: unknown ({exc})")


def cmd_stop(ctx: Ctx, args: argparse.Namespace) -> int:
    code = stop_all(ctx, args.modal)
    if args.modal:
        out(f"restart the app so it picks up GPU_BACKEND=modal: {RESTART}")
    return code


def cmd_modal(ctx: Ctx, args: argparse.Namespace) -> int:
    code = stop_all(ctx, True)
    out(f"restart the app so it picks up GPU_BACKEND=modal: {RESTART}")
    return code


# status


def cmd_status(ctx: Ctx, args: argparse.Namespace) -> int:
    env = envfile.read_env(ctx.env_path)
    out(f".env: GPU_BACKEND={env.get('GPU_BACKEND', '(not set)')}, "
        f"WORKER_IMAGE={env.get('WORKER_IMAGE') or 'not set'}")

    h = ctx.desk.health()
    out(f"app: backend {h.get('backend')}, version {h.get('version')}" if h
        else f"app: unreachable at {ctx.app_url}")

    t = ctx.state["tunnel"]
    if t:
        up = tunnel.alive(t["pid"], t["binary"])
        public = ctx.deps.probe(t["url"]) if up else None
        reach = f"reachable, backend {public.get('backend')}" if public else "not reachable"
        out(f"tunnel: {t['url']}, pid {t['pid']}, {'alive' if up else 'not running'}, {reach}")
    else:
        out("tunnel: none")

    mine = ctx.state["pod"]
    if not ctx.cfg.get("RUNPOD_API_KEY"):
        out("pod: unknown, RUNPOD_API_KEY is not set")
    else:
        try:
            listed = ctx.runpod.pods()
            pod = next((p for p in listed if mine and p.id == mine["id"]), None)
            if pod:
                out(f"pod: {fmt_pod(pod)}")
            else:
                out(f"pod: none (state named {mine['id']}, RunPod no longer lists it)" if mine else "pod: none")
            others = [p for p in listed if not mine or p.id != mine["id"]]
            out("other pods: none" if not others else "other pods: " + "; ".join(fmt_pod(p) for p in others))
            out(f"runpod spend: ${ctx.runpod.spend_per_hr():.3f}/h")
        except (httpx.HTTPError, RunPodError) as exc:
            out(f"pod: RunPod did not answer ({exc})")

    if not ctx.cfg.get("OWNER_TOKEN"):
        out("workers: unknown, OWNER_TOKEN is not set")
    elif not h:
        out("workers: unknown, the app is unreachable")
    else:
        try:
            workers = ctx.desk.workers()
        except (httpx.HTTPError, DeskError) as exc:
            out(f"workers: unknown ({exc})")
        else:
            if not workers:
                out("workers: none")
            for w in workers:
                state_ = "online" if w.get("online") else "offline"
                out(f"worker: {w.get('id')}, {state_}, job {w.get('job_id') or 'none'}")
    return 0


COMMANDS = {"runpod": cmd_runpod, "modal": cmd_modal, "status": cmd_status, "stop": cmd_stop}


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--app", help=f"the app's URL (default {DEFAULT_APP})")
    p = argparse.ArgumentParser(prog="python -m radar_desk.compute",
                                description="Switch scoring between Modal and one RunPod pod.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("modal", parents=[common], help="set GPU_BACKEND=modal and stop the pod, tunnel and token")
    r = sub.add_parser("runpod", parents=[common], help="set GPU_BACKEND=worker, start the tunnel, token and pod")
    r.add_argument("--wait", action="store_true", help="wait for the app to come back on the worker backend")
    r.add_argument("--wait-timeout", type=float, default=300, help="seconds for --wait (default 300)")
    r.add_argument("--allow-other-pods", action="store_true", help="deploy even when RunPod lists other pods")
    sub.add_parser("status", parents=[common], help="show the backend, app, tunnel, pod, workers and spend")
    s = sub.add_parser("stop", parents=[common], help="delete the pod, stop the tunnel, revoke the token")
    s.add_argument("--modal", action="store_true", help="also set GPU_BACKEND=modal")
    return p


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parser().parse_args(argv)
    deps = deps or Deps()
    ctx = None
    try:
        ctx = Ctx(deps, args.app or deps.app_url)
        if args.command != "status":
            ctx.lock()
        return COMMANDS[args.command](ctx, args)
    except Fail as exc:
        err(str(exc))
        return exc.code
    except (RunPodError, DeskError, tunnel.TunnelError, httpx.HTTPError, OSError, ValueError) as exc:
        err(f"error: {exc}")
        return 1
    finally:
        if ctx:
            ctx.unlock()
