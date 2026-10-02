"""RunPod over its v2 REST API: deploy one worker pod, list pods and their hourly cost, delete a pod.

RunPodApi is the bare v2 client (base URL, auth, problem JSON errors) and takes another base URL so other
RunPod products can reuse it. RunPod is the pod layer the compute service drives. A create that RunPod
answers with 400 counts as no stock for that GPU type, because v2 gives capacity exhaustion no code of its
own; the deploy loop then retries and moves on to the next GPU type.

Every request carries a named User-Agent; RunPod answers a default Python one with 403.
"""

from __future__ import annotations

import logging
import shlex
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

API_URL = "https://api.runpod.io/v2"
DEFAULT_GPUS = "NVIDIA L4,NVIDIA GeForce RTX 4090"
GPU_PREFERENCE = tuple(DEFAULT_GPUS.split(","))
# The image's torch 2.5.1/cu124 cannot drive Blackwell parts; parse_gpus refuses any id holding one of these,
# matched with the spaces removed (so "RTX PRO 6000", "RTX5090" and "GB200" all count). "rtxpro" is the whole
# RTX PRO family; no Ada, Ampere or Hopper id contains it, and "RTX 5000 Ada" does not match any "rtx50x0".
BLACKWELL = ("blackwell", "rtxpro", "pro6000", "b200", "b300", "rtx5090", "rtx5080", "rtx5070", "rtx5060")
# RunPod ids the image is known or expected to run on. The list is from memory of RunPod's ids and only
# drives a warning; RunPod itself decides whether an id exists.
KNOWN_GPUS = frozenset({
    "NVIDIA L4", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A4000", "NVIDIA RTX A4500",
    "NVIDIA RTX A5000", "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40", "NVIDIA L40S",
    "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX 6000 Ada Generation", "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3",
})

try:
    USER_AGENT = f"radar-desk-compute/{version('radar-desk')}"
except PackageNotFoundError:
    USER_AGENT = "radar-desk-compute/unknown"

log = logging.getLogger(__name__)


def parse_gpus(value: str, warn: bool = False) -> list[str]:
    """RUNPOD_GPUS as an ordered list without duplicates. Refuses an empty list and any Blackwell part.

    With `warn`, an id outside KNOWN_GPUS is logged; the settings do that once at load, not on every read.
    """
    gpus: list[str] = []
    for name in (g.strip() for g in value.split(",")):
        if name and name not in gpus:
            gpus.append(name)
    if not gpus:
        raise ValueError("RUNPOD_GPUS names no GPU type")
    for name in gpus:
        if any(b in "".join(name.lower().split()) for b in BLACKWELL):
            raise ValueError(f"RUNPOD_GPUS: {name} is a Blackwell part; "
                             "the worker image's torch 2.5.1/cu124 cannot drive it")
        if warn and name not in KNOWN_GPUS:
            log.warning("RUNPOD_GPUS: %s is not a GPU type this app knows; RunPod decides whether it exists", name)
    return gpus


class RunPodError(RuntimeError):
    """RunPod refused a call. `status` is the HTTP status when there was one, `detail` RunPod's explanation."""

    def __init__(self, message: str, status: int | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = message if detail is None else detail


class StockError(RunPodError):
    """No machine with the requested GPU type is free right now."""


class RunPodApi:
    """RunPod's v2 REST API: bearer auth, a named User-Agent, problem JSON turned into RunPodError."""

    def __init__(self, api_key: str, client: httpx.Client | None = None, base_url: str = API_URL) -> None:
        self.client = client or httpx.Client(timeout=30)
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}

    def call(self, method: str, path: str, json: Any = None, params: dict | None = None) -> Any:
        """The parsed JSON body of a 2xx answer, None when it is empty. Any other answer raises RunPodError.

        A transport failure raises httpx.HTTPError as it is. Messages never carry the key or the headers.
        """
        r = self.client.request(method, f"{self.base_url}{path}", json=json, params=params, headers=self.headers)
        try:
            body = r.json() if r.content else None
        except ValueError:
            body = ValueError
        if r.is_success:
            if body is ValueError:
                raise RunPodError(f"RunPod answered {r.status_code} without JSON", r.status_code)
            return body
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, str) and detail:
            errors = body.get("errors")
            if isinstance(errors, list) and errors:
                detail = f"{detail}: {'; '.join(str(e) for e in errors)}"
            raise RunPodError(f"RunPod answered {r.status_code}: {detail}", r.status_code, detail)
        raise RunPodError(f"RunPod answered {r.status_code}", r.status_code)


@dataclass
class Pod:
    id: str
    name: str
    gpu: str
    cost_per_hr: float
    status: str
    uptime_s: int | None = None
    image: str = ""


@dataclass
class PodSpec:
    image: str
    registry_auth_id: str
    datacenter: str
    volume_id: str
    env: dict[str, str] = field(default_factory=dict)
    name: str = "radar-worker"
    docker_args: str = "python3 -m radar_worker.pull"
    container_disk_gb: int = 20
    mount_path: str = "/workspace"
    cloud_type: str = "SECURE"
    gpu_count: int = 1

    def body(self, gpu_type: str) -> dict:
        """The POST /v2/pods body. `cmd` is the argument list for the image's ENTRYPOINT."""
        return {
            "name": self.name, "image": self.image, "registry": self.registry_auth_id, "cloud": self.cloud_type,
            "gpu": {"id": gpu_type, "count": self.gpu_count}, "dataCenterIds": [self.datacenter],
            "mounts": {"network": [{"volumeId": self.volume_id, "path": self.mount_path}]},
            "disk": self.container_disk_gb, "cmd": shlex.split(self.docker_args), "env": dict(self.env),
        }


def _pod(raw: dict, gpu: str = "") -> Pod:
    found = raw.get("gpu") or {}
    runtime = raw.get("runtime") or {}
    return Pod(id=raw["id"], name=raw.get("name") or "", gpu=found.get("id") or gpu,
               cost_per_hr=float(raw.get("cost") or 0), status=raw.get("status") or "",
               uptime_s=runtime.get("uptime"), image=raw.get("image") or "")


class RunPod:
    def __init__(self, api_key: str, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.api = RunPodApi(api_key, client)
        self.sleep = sleep

    def deploy(self, spec: PodSpec, gpus: Sequence[str] = GPU_PREFERENCE, attempts: int = 3,
               wait_s: float = 20) -> Pod:
        """Deploy on the first GPU type in `gpus` that has stock, retrying a 400 `attempts` times per type."""
        last = ""
        for gpu in gpus:
            for attempt in range(attempts):
                try:
                    raw = self.api.call("POST", "/pods", json=spec.body(gpu))
                except RunPodError as exc:
                    if exc.status != 400:
                        raise
                    last = exc.detail
                    log.warning("RunPod has no %s: %s", gpu, last)
                    if attempt < attempts - 1:
                        self.sleep(wait_s)
                    continue
                if not isinstance(raw, dict) or not raw.get("id"):
                    raise RunPodError("deploy returned no pod")
                return _pod(raw, gpu)
        raise StockError(f"no stock for {', '.join(gpus)}: {last}", 400, last)

    def pods(self) -> list[Pod]:
        found: list[Pod] = []
        params: dict = {"limit": 1000}
        while True:
            page = self.api.call("GET", "/pods", params=params) or {}
            found += [_pod(p) for p in page.get("pods") or []]
            more = page.get("pagination") or {}
            if not (more.get("hasNextPage") and more.get("nextCursor")):
                return found
            params = {"limit": 1000, "cursor": more["nextCursor"]}

    def pod(self, pod_id: str) -> Pod | None:
        try:
            raw = self.api.call("GET", f"/pods/{pod_id}")
        except RunPodError as exc:
            if exc.status == 404:
                return None
            raise
        return _pod(raw) if isinstance(raw, dict) and raw.get("id") else None

    def spend_per_hr(self) -> float:
        """The summed hourly cost of this account's pods. v2 has no live account rate, so serverless
        workers and network volumes are not in it."""
        return sum(p.cost_per_hr for p in self.pods())

    def delete(self, pod_id: str) -> bool:
        """Delete a pod; False when RunPod no longer has it, also after a failed delete."""
        try:
            self.api.call("DELETE", f"/pods/{pod_id}")
        except RunPodError as exc:
            if exc.status is not None and 200 <= exc.status < 300:  # deleted, with a body that is not JSON
                return True
            if exc.status == 404 or (exc.status is not None and self.pod(pod_id) is None):
                return False
            raise
        return True
