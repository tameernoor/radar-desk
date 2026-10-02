"""The RunPod v2 client against httpx.MockTransport: the create body, stock retries, pods, spend and delete."""

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
    RunPodApi,
    RunPodError,
    StockError,
    parse_gpus,
)

NO_STOCK = {"title": "Bad Request", "status": 400,
            "detail": "There are no longer any instances available with the requested specifications."}
P1 = {"id": "p1", "name": "radar-worker", "image": "img", "status": "RUNNING", "cost": 0.39,
      "runtime": {"uptime": 120}, "gpu": {"id": "NVIDIA L4", "count": 1, "vcpuCount": 4, "memory": 24}}
P2 = {"id": "p2", "name": "other", "status": "EXITED", "cost": "0.69", "runtime": None}


def spec() -> PodSpec:
    return PodSpec(image="img", registry_auth_id="reg", datacenter="EU-RO-1", volume_id="vol",
                   env={"RADAR_DESK_URL": "https://x.trycloudflare.com"})


def client(handler, seen: list):
    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(wrapped))


def created(gpu: str, pod_id: str = "pod9", cost: float = 0.69) -> httpx.Response:
    return httpx.Response(201, json={"id": pod_id, "name": "radar-worker", "image": "img", "status": "RUNNING",
                                     "cost": cost, "runtime": {}, "gpu": {"id": gpu, "count": 1}})


def assert_headers(seen: list) -> None:
    assert USER_AGENT.startswith("radar-desk-compute/")
    for r in seen:
        assert r.headers["user-agent"] == USER_AGENT
        assert r.headers["authorization"] == "Bearer key"


def test_the_create_body():
    seen = []
    pod = RunPod("key", client(lambda r: created("NVIDIA L4"), seen)).deploy(spec())
    assert (pod.id, pod.gpu, pod.cost_per_hr, pod.status, pod.image) == ("pod9", "NVIDIA L4", 0.69, "RUNNING", "img")
    [r] = seen
    assert r.method == "POST" and str(r.url) == "https://api.runpod.io/v2/pods"
    assert json.loads(r.content) == {
        "name": "radar-worker", "image": "img", "registry": "reg", "cloud": "SECURE",
        "gpu": {"id": "NVIDIA L4", "count": 1}, "dataCenterIds": ["EU-RO-1"],
        "mounts": {"network": [{"volumeId": "vol", "path": "/workspace"}]}, "disk": 20,
        "cmd": ["python3", "-m", "radar_worker.pull"], "env": {"RADAR_DESK_URL": "https://x.trycloudflare.com"}}
    assert_headers(seen)


def test_stock_errors_retry_then_next_gpu(caplog):
    seen, sleeps = [], []

    def handler(request):
        gpu = json.loads(request.content)["gpu"]["id"]
        return httpx.Response(400, json=NO_STOCK) if gpu == "NVIDIA L4" else created(gpu)

    with caplog.at_level(logging.WARNING):
        pod = RunPod("key", client(handler, seen), sleep=sleeps.append).deploy(spec())
    assert (pod.id, pod.gpu, pod.cost_per_hr) == ("pod9", "NVIDIA GeForce RTX 4090", 0.69)
    assert [json.loads(r.content)["gpu"]["id"] for r in seen] == ["NVIDIA L4"] * 3 + ["NVIDIA GeForce RTX 4090"]
    assert sleeps == [20, 20]
    assert "NVIDIA L4" in caplog.text and "no longer any instances" in caplog.text
    assert_headers(seen)


def test_a_400_on_every_candidate_raises_stock_error():
    seen, sleeps = [], []
    with pytest.raises(StockError, match="no stock for NVIDIA L4, NVIDIA GeForce RTX 4090: There are no") as info:
        RunPod("key", client(lambda r: httpx.Response(400, json=NO_STOCK), seen), sleep=sleeps.append).deploy(spec())
    assert len(seen) == 6 and sleeps == [20] * 4
    assert info.value.status == 400


@pytest.mark.parametrize("status", [401, 402, 403, 404, 413, 422, 429, 500, 503])
def test_other_create_errors_raise_at_once_with_the_detail(status, caplog):
    seen, sleeps = [], []
    problem = {"title": "x", "status": status, "detail": "bad volume",
               "errors": ["$.disk: must be >= 1", "$.bogus: not allowed"]}
    handler = lambda r: httpx.Response(status, json=problem, headers={"content-type": "application/problem+json"})
    with caplog.at_level(logging.DEBUG), pytest.raises(RunPodError, match="bad volume") as info:
        RunPod("rpk_s3cr3t_7777", client(handler, seen), sleep=sleeps.append).deploy(spec())
    assert not isinstance(info.value, StockError)
    assert info.value.status == status
    assert info.value.detail == "bad volume: $.disk: must be >= 1; $.bogus: not allowed"
    assert "rpk_s3cr3t_7777" not in str(info.value) and "Bearer" not in str(info.value)
    assert "rpk_s3cr3t_7777" not in caplog.text
    assert len(seen) == 1 and sleeps == []


def test_an_error_without_problem_json_names_the_status():
    with pytest.raises(RunPodError, match="^RunPod answered 502$") as info:
        RunPod("key", client(lambda r: httpx.Response(502, text="bad gateway"), [])).deploy(spec())
    assert info.value.status == 502


def test_deploy_without_a_pod_raises():
    for answer in (httpx.Response(201), httpx.Response(201, json={"name": "radar-worker"})):
        with pytest.raises(RunPodError, match="no pod"):
            RunPod("key", client(lambda r, a=answer: a, []), sleep=lambda s: None).deploy(spec())


def test_deploy_tries_a_custom_order():
    seen, sleeps = [], []

    def handler(request):
        gpu = json.loads(request.content)["gpu"]["id"]
        return httpx.Response(400, json=NO_STOCK) if gpu == "NVIDIA GeForce RTX 4090" else created(gpu, cost=0.39)

    gpus = ["NVIDIA GeForce RTX 4090", "NVIDIA L4"]
    pod = RunPod("key", client(handler, seen), sleep=sleeps.append).deploy(spec(), gpus=gpus, attempts=1)
    assert pod.gpu == "NVIDIA L4"
    assert [json.loads(r.content)["gpu"]["id"] for r in seen] == gpus
    assert sleeps == []


def test_pods_follow_the_cursor_and_spend_is_the_sum():
    seen = []

    def handler(request):
        if request.url.params.get("cursor") == "c2":
            return httpx.Response(200, json={"pods": [P2], "pagination": {"nextCursor": None, "hasNextPage": False}})
        return httpx.Response(200, json={"pods": [P1], "pagination": {"nextCursor": "c2", "hasNextPage": True}})

    rp = RunPod("key", client(handler, seen))
    pods = rp.pods()
    assert [(p.id, p.gpu, p.cost_per_hr, p.status, p.uptime_s) for p in pods] == [
        ("p1", "NVIDIA L4", 0.39, "RUNNING", 120), ("p2", "", 0.69, "EXITED", None)]
    assert [dict(r.url.params) for r in seen] == [{"limit": "1000"}, {"limit": "1000", "cursor": "c2"}]
    assert all(r.method == "GET" and r.url.path == "/v2/pods" for r in seen)
    assert rp.spend_per_hr() == pytest.approx(0.39 + 0.69)
    assert_headers(seen)


def test_pod_by_id():
    seen = []

    def handler(request):
        if request.url.path == "/v2/pods/p1":
            return httpx.Response(200, json=P1)
        return httpx.Response(404, json={"title": "Not Found", "status": 404, "detail": "pod not found"})

    rp = RunPod("key", client(handler, seen))
    assert rp.pod("p1").uptime_s == 120 and rp.pod("nope") is None
    assert [str(r.url) for r in seen] == ["https://api.runpod.io/v2/pods/p1", "https://api.runpod.io/v2/pods/nope"]
    assert_headers(seen)


@pytest.mark.parametrize(("status", "result"), [(204, True), (200, True), (404, False)])
def test_delete(status, result):
    seen = []
    rp = RunPod("key", client(lambda r: httpx.Response(status), seen))
    assert rp.delete("p1") is result
    assert seen[0].method == "DELETE" and str(seen[0].url) == "https://api.runpod.io/v2/pods/p1"
    assert len(seen) == 1
    assert_headers(seen)


def test_failed_delete_of_a_pod_that_is_gone_is_false():
    seen = []

    def handler(request):
        return httpx.Response(500) if request.method == "DELETE" else httpx.Response(404)

    assert RunPod("key", client(handler, seen)).delete("p1") is False
    assert [(r.method, r.url.path) for r in seen] == [("DELETE", "/v2/pods/p1"), ("GET", "/v2/pods/p1")]


def test_failed_delete_of_a_pod_that_is_still_there_raises():
    def handler(request):
        return httpx.Response(500) if request.method == "DELETE" else httpx.Response(200, json=P1)

    with pytest.raises(RunPodError) as info:
        RunPod("key", client(handler, [])).delete("p1")
    assert info.value.status == 500


def test_the_api_takes_another_base_url():
    seen = []
    api = RunPodApi("key", client(lambda r: httpx.Response(200, json={"ok": True}), seen),
                    base_url="https://api.runpod.ai/v2/")
    assert api.call("GET", "/endpoints", params={"limit": 5}) == {"ok": True}
    assert str(seen[0].url) == "https://api.runpod.ai/v2/endpoints?limit=5"
    assert_headers(seen)


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
