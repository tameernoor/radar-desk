"""The RunPod client against httpx.MockTransport: headers, stock retries, pods, spend and delete."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from radar_desk.compute.runpod import (
    DEFAULT_GPUS,
    GPU_PREFERENCE,
    KNOWN_GPUS,
    USER_AGENT,
    PodSpec,
    RunPod,
    RunPodError,
    StockError,
    parse_gpus,
)

STOCK = "There are no longer any instances available with the requested specifications."
PODS = {"data": {"myself": {"currentSpendPerHr": 0.39, "pods": [
    {"id": "p1", "name": "radar-worker", "desiredStatus": "RUNNING", "costPerHr": 0.39,
     "runtime": {"uptimeInSeconds": 120}, "machine": {"gpuDisplayName": "L4"}},
    {"id": "p2", "name": "other", "desiredStatus": "EXITED", "costPerHr": "0.69", "runtime": None,
     "machine": None}]}}}


def spec() -> PodSpec:
    return PodSpec(image="img", registry_auth_id="reg", datacenter="EU-RO-1", volume_id="vol",
                   env={"RADAR_DESK_URL": "https://x.trycloudflare.com"})


def client(handler, seen: list):
    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(wrapped))


def test_stock_errors_retry_then_next_gpu_with_the_user_agent():
    seen, sleeps = [], []

    def handler(request):
        gpu = json.loads(request.content)["variables"]["input"]["gpuTypeId"]
        if gpu == "NVIDIA L4":
            return httpx.Response(200, json={"errors": [{"message": STOCK}]})
        return httpx.Response(200, json={"data": {"podFindAndDeployOnDemand": {
            "id": "pod9", "name": "radar-worker", "costPerHr": 0.69, "desiredStatus": "RUNNING",
            "machine": {"gpuDisplayName": "RTX 4090"}}}})

    pod = RunPod("key", client(handler, seen), sleep=sleeps.append).deploy(spec())
    assert (pod.id, pod.gpu, pod.cost_per_hr) == ("pod9", "RTX 4090", 0.69)
    inputs = [json.loads(r.content)["variables"]["input"] for r in seen]
    assert [i["gpuTypeId"] for i in inputs] == ["NVIDIA L4"] * 3 + ["NVIDIA GeForce RTX 4090"]
    assert sleeps == [20, 20]
    first = inputs[0]
    assert first["volumeMountPath"] == "/workspace" and first["dockerArgs"] == "python3 -m radar_worker.pull"
    assert first["cloudType"] == "SECURE" and first["gpuCount"] == 1 and first["containerDiskInGb"] == 20
    assert first["env"] == [{"key": "RADAR_DESK_URL", "value": "https://x.trycloudflare.com"}]
    for r in seen:
        assert r.headers["user-agent"] == USER_AGENT and USER_AGENT.startswith("radar-desk-compute/")
        assert r.headers["authorization"] == "Bearer key"


def test_no_stock_anywhere_raises_stock_error():
    seen, sleeps = [], []
    handler = lambda r: httpx.Response(200, json={"errors": [{"message": "SUPPLY_CONSTRAINT"}]})
    with pytest.raises(StockError):
        RunPod("key", client(handler, seen), sleep=sleeps.append).deploy(spec())
    assert len(seen) == 6 and sleeps == [20] * 4


def test_other_error_raises_at_once():
    seen, sleeps = [], []
    handler = lambda r: httpx.Response(200, json={"errors": [{"message": "bad volume"}]})
    with pytest.raises(RunPodError, match="bad volume") as info:
        RunPod("key", client(handler, seen), sleep=sleeps.append).deploy(spec())
    assert not isinstance(info.value, StockError)
    assert len(seen) == 1 and sleeps == []


def test_pods_and_spend():
    seen = []
    rp = RunPod("key", client(lambda r: httpx.Response(200, json=PODS), seen))
    pods = rp.pods()
    assert [(p.id, p.gpu, p.cost_per_hr, p.desired_status, p.uptime_s) for p in pods] == [
        ("p1", "L4", 0.39, "RUNNING", 120), ("p2", "", 0.69, "EXITED", None)]
    assert rp.pod("p2").name == "other" and rp.pod("nope") is None
    assert rp.spend_per_hr() == 0.39
    assert all(r.headers["user-agent"] == USER_AGENT for r in seen)


@pytest.mark.parametrize(("status", "result"), [(200, True), (204, True), (404, False)])
def test_delete(status, result):
    seen = []
    rp = RunPod("key", client(lambda r: httpx.Response(status), seen))
    assert rp.delete("p1") is result
    assert seen[0].method == "DELETE" and str(seen[0].url) == "https://rest.runpod.io/v1/pods/p1"
    assert seen[0].headers["user-agent"] == USER_AGENT


def test_delete_other_status_raises():
    with pytest.raises(RunPodError):
        RunPod("key", client(lambda r: httpx.Response(500), [])).delete("p1")


def test_default_gpus_parse_to_the_preference():
    assert parse_gpus(DEFAULT_GPUS) == list(GPU_PREFERENCE)


@pytest.mark.parametrize("name", [
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 4000 Blackwell", "NVIDIA B200",
    "NVIDIA B300", "NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5080", "NVIDIA GeForce RTX 5070",
    "NVIDIA RTX PRO 4500", "nvidia pro 6000", "NVIDIA RTX PRO 5000", "NVIDIA RTX PRO 2000",
    "NVIDIA GeForce RTX 5060 Ti", "RTX5090", "NVIDIA RTX  PRO  6000", "NVIDIA GB200"])
def test_parse_gpus_refuses_blackwell(name):
    with pytest.raises(ValueError, match="Blackwell"):
        parse_gpus(f"NVIDIA L4,{name}")


@pytest.mark.parametrize("name", sorted(KNOWN_GPUS) + [
    "NVIDIA RTX 5000 Ada Generation", "NVIDIA RTX 2000 Ada Generation", "NVIDIA H200", "NVIDIA H100 NVL",
    "NVIDIA GeForce RTX 4080 SUPER", "NVIDIA GeForce RTX 3080 Ti"])
def test_parse_gpus_keeps_every_pre_blackwell_part(name):
    assert parse_gpus(name) == [name]


def test_parse_gpus_strips_and_drops_duplicates():
    assert parse_gpus(" NVIDIA GeForce RTX 4090 ,NVIDIA L4,,NVIDIA GeForce RTX 4090") == [
        "NVIDIA GeForce RTX 4090", "NVIDIA L4"]
    with pytest.raises(ValueError, match="names no GPU type"):
        parse_gpus(" , ")


def test_parse_gpus_warns_on_an_unknown_id_only_when_asked(caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_gpus("NVIDIA L4,NVIDIA RTX A2000") == ["NVIDIA L4", "NVIDIA RTX A2000"]
        assert caplog.text == ""  # a plain read, as status() does every few seconds, stays quiet
        assert parse_gpus("NVIDIA L4,NVIDIA RTX A2000", warn=True) == ["NVIDIA L4", "NVIDIA RTX A2000"]
    assert "NVIDIA RTX A2000 is not a GPU type" in caplog.text and "NVIDIA L4 is" not in caplog.text


def test_deploy_tries_a_custom_order():
    seen, sleeps = [], []

    def handler(request):
        gpu = json.loads(request.content)["variables"]["input"]["gpuTypeId"]
        if gpu == "NVIDIA GeForce RTX 4090":
            return httpx.Response(200, json={"errors": [{"message": STOCK}]})
        return httpx.Response(200, json={"data": {"podFindAndDeployOnDemand": {
            "id": "pod9", "name": "radar-worker", "costPerHr": 0.39, "desiredStatus": "RUNNING",
            "machine": {"gpuDisplayName": "L4"}}}})

    gpus = ["NVIDIA GeForce RTX 4090", "NVIDIA L4"]
    pod = RunPod("key", client(handler, seen), sleep=sleeps.append).deploy(spec(), gpus=gpus, attempts=1)
    assert pod.gpu == "L4"
    assert [json.loads(r.content)["variables"]["input"]["gpuTypeId"] for r in seen] == gpus
    assert sleeps == []


def test_failed_delete_of_a_pod_that_is_gone_is_false():
    def handler(request):
        if request.method == "DELETE":
            return httpx.Response(500)
        return httpx.Response(200, json={"data": {"myself": {"currentSpendPerHr": 0, "pods": []}}})

    assert RunPod("key", client(handler, [])).delete("p1") is False


def test_deploy_without_a_pod_raises():
    handler = lambda r: httpx.Response(200, json={"data": {"podFindAndDeployOnDemand": None}})
    with pytest.raises(RunPodError, match="no pod"):
        RunPod("key", client(handler, []), sleep=lambda s: None).deploy(spec())
