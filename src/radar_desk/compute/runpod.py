"""RunPod over its GraphQL and REST APIs: deploy one worker pod, list pods and spend, delete a pod.

Every request carries a named User-Agent; RunPod answers a default Python one with 403.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

import httpx

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
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
STOCK_MESSAGES = ("SUPPLY_CONSTRAINT", "no longer any instances available")

try:
    USER_AGENT = f"radar-desk-compute/{version('radar-desk')}"
except PackageNotFoundError:
    USER_AGENT = "radar-desk-compute/unknown"

DEPLOY = """mutation($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) { id name imageName costPerHr desiredStatus
    machine { gpuDisplayName } }
}"""
MYSELF = """query { myself { currentSpendPerHr pods { id name imageName desiredStatus costPerHr
  runtime { uptimeInSeconds } machine { gpuDisplayName } } } }"""


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
    """RunPod refused a call. The message is RunPod's first error message."""


class StockError(RunPodError):
    """No machine with the requested GPU type is free right now."""


@dataclass
class Pod:
    id: str
    name: str
    gpu: str
    cost_per_hr: float
    desired_status: str
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

    def input(self, gpu_type: str) -> dict:
        return {
            "cloudType": self.cloud_type, "gpuCount": self.gpu_count, "gpuTypeId": gpu_type,
            "dataCenterId": self.datacenter, "networkVolumeId": self.volume_id,
            "volumeMountPath": self.mount_path, "containerDiskInGb": self.container_disk_gb, "volumeInGb": 0,
            "imageName": self.image, "containerRegistryAuthId": self.registry_auth_id,
            "dockerArgs": self.docker_args, "name": self.name,
            "env": [{"key": k, "value": v} for k, v in self.env.items()],
        }


def _pod(raw: dict, gpu: str = "") -> Pod:
    machine = raw.get("machine") or {}
    runtime = raw.get("runtime") or {}
    return Pod(id=raw["id"], name=raw.get("name") or "", gpu=machine.get("gpuDisplayName") or gpu,
               cost_per_hr=float(raw.get("costPerHr") or 0), desired_status=raw.get("desiredStatus") or "",
               uptime_s=runtime.get("uptimeInSeconds"), image=raw.get("imageName") or "")


class RunPod:
    def __init__(self, api_key: str, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client or httpx.Client(timeout=30)
        self.headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}
        self.sleep = sleep

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        r = self.client.post(GRAPHQL_URL, json={"query": query, "variables": variables or {}},
                             headers=self.headers)
        try:
            body = r.json()
        except ValueError:
            raise RunPodError(f"RunPod GraphQL answered {r.status_code}") from None
        errors = body.get("errors") if isinstance(body, dict) else None
        if errors:
            message = str(errors[0].get("message", errors[0]))
            raise (StockError if any(s in message for s in STOCK_MESSAGES) else RunPodError)(message)
        if r.status_code != 200 or not isinstance(body, dict) or "data" not in body:
            raise RunPodError(f"RunPod GraphQL answered {r.status_code}")
        return body["data"]

    def deploy(self, spec: PodSpec, gpus: Sequence[str] = GPU_PREFERENCE, attempts: int = 3,
               wait_s: float = 20) -> Pod:
        """Deploy on the first GPU type in `gpus` that has stock, retrying a stock error `attempts` times."""
        for gpu in gpus:
            for attempt in range(attempts):
                try:
                    raw = self.graphql(DEPLOY, {"input": spec.input(gpu)})["podFindAndDeployOnDemand"]
                    if not raw:
                        raise RunPodError("deploy returned no pod")
                    return _pod(raw, gpu)
                except StockError:
                    if attempt < attempts - 1:
                        self.sleep(wait_s)
        raise StockError(f"no stock for {', '.join(gpus)}")

    def _myself(self) -> dict:
        return self.graphql(MYSELF)["myself"]

    def pods(self) -> list[Pod]:
        return [_pod(p) for p in self._myself().get("pods") or []]

    def pod(self, pod_id: str) -> Pod | None:
        return next((p for p in self.pods() if p.id == pod_id), None)

    def spend_per_hr(self) -> float:
        return float(self._myself().get("currentSpendPerHr") or 0)

    def delete(self, pod_id: str) -> bool:
        """Delete a pod; False when RunPod no longer has it, also after a failed delete."""
        r = self.client.delete(f"{REST_URL}/pods/{pod_id}", headers=self.headers)
        if r.status_code == 404:
            return False
        if r.status_code not in (200, 204):
            if self.pod(pod_id) is None:
                return False
            raise RunPodError(f"DELETE pod {pod_id} answered {r.status_code}")
        return True
