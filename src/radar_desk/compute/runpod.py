"""RunPod over its GraphQL and REST APIs: deploy one worker pod, list pods and spend, delete a pod.

Every request carries a named User-Agent; RunPod answers a default Python one with 403.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

import httpx

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
# The image's torch 2.5.1/cu124 cannot drive Blackwell parts (RTX PRO, B200, RTX 5090); never list one.
GPU_PREFERENCE = ("NVIDIA L4", "NVIDIA GeForce RTX 4090")
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
