"""The reading radiologist's reads: stored with their inputs, redone only when an input changes."""

from __future__ import annotations

from api_client import api_client
from radar_desk.services.blobs import read_json
from radar_desk.services.readers import readers_key
from readers_seed import NV_READ, REPORT_READ, SYNTHETIC_REPORT


def run_jobs(world, ticks: int = 3) -> None:
    world.svc.jobs.create(world.scan.id)
    for _ in range(ticks):
        world.poller.tick()


def test_attaching_a_report_reads_it_once(readers_world):
    w = readers_world([REPORT_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    stored = read_json(w.svc.storage, readers_key(w.scan.id, "report_read"))
    assert stored["state"] == "ready" and len(stored["input_hash"]) == 64
    assert set(stored["inputs"]) == {"text", "labels", "prompt", "model"}
    assert stored["read"]["labels"]["ascites"] == {"state": "present", "quote": "Small volume ascites in the pelvis."}
    assert len(w.llm.calls) == 1
    assert w.svc.readers.kick(w.scan.id) is False
    status = w.svc.readers.status(w.scan.id)
    assert status["report_read"] == {"status": "ready", "error": None}
    assert status["nv_read"] == {"status": "none", "error": None}


def test_replacing_the_report_reads_again_and_removing_it_deletes_the_read(readers_world):
    w = readers_world([REPORT_READ, REPORT_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT + "\nAddendum: no change.")
    assert len(w.llm.calls) == 2
    w.svc.scans.delete_report(w.scan.id)
    assert read_json(w.svc.storage, readers_key(w.scan.id, "report_read")) is None
    assert w.svc.readers.status(w.scan.id)["report_read"]["status"] == "none"


def test_the_nv_report_is_read_when_its_job_is_done(readers_world):
    w = readers_world([NV_READ])
    run_jobs(w)
    stored = read_json(w.svc.storage, readers_key(w.scan.id, "nv_read"))
    assert stored["state"] == "ready"
    assert stored["read"]["labels"]["renal_cyst"] == {"state": "present", "quote": "A simple cyst in the left kidney."}
    assert "Fake NV-Reason-CT report" in w.llm.calls[0][1]["content"]


def test_a_failed_read_is_kept_until_retry(readers_world):
    w = readers_world(["not json", "still not json", REPORT_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    status = w.svc.readers.status(w.scan.id)["report_read"]
    assert status["status"] == "failed" and "JSON" in status["error"]
    assert w.svc.readers.kick(w.scan.id) is False
    assert len(w.llm.calls) == 2
    assert w.svc.readers.kick(w.scan.id, force=True) is True
    assert w.svc.readers.status(w.scan.id)["report_read"]["status"] == "ready"


def test_without_a_chat_model_the_reads_wait(readers_world):
    w = readers_world(llm=False)
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    assert w.svc.readers.status(w.scan.id)["report_read"] == {"status": "no_llm", "error": None}
    assert w.svc.readers.kick(w.scan.id) is False


def test_refresh_route(readers_world):
    w = readers_world(["bad", "bad", REPORT_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    client = api_client(w.svc)
    r = client.post(f"/scans/{w.scan.id}/readers/refresh")
    assert r.status_code == 200, r.text
    assert r.json()["started"] is True and r.json()["report_read"]["status"] == "ready"
    assert client.post("/scans/scan_missing/readers/refresh").status_code == 404


def test_a_listener_call_with_unchanged_inputs_does_not_read_again(readers_world):
    w = readers_world([REPORT_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    for listener in w.svc.scans.report_listeners:
        listener(w.scan.id)
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    assert len(w.llm.calls) == 1


def test_deleting_the_scan_removes_the_reads(readers_world):
    w = readers_world([REPORT_READ, NV_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    run_jobs(w)
    w.svc.storage.put_bytes(readers_key(w.scan.id, "queries"), b"{}")
    for name in ("report_read", "nv_read", "queries"):
        assert w.svc.storage.exists(readers_key(w.scan.id, name))
    w.svc.scans.delete(w.scan.id)
    for name in ("report_read", "nv_read", "queries"):
        assert not w.svc.storage.exists(readers_key(w.scan.id, name))


def test_removing_the_report_without_a_chat_model_still_deletes_the_read(readers_world):
    w = readers_world([REPORT_READ])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    w.svc.readers.llm = None
    w.svc.scans.delete_report(w.scan.id)
    assert read_json(w.svc.storage, readers_key(w.scan.id, "report_read")) is None


def test_a_fake_nv_result_missing_optional_keys_still_reads(readers_world):
    w = readers_world([NV_READ])
    run_jobs(w)
    job = w.svc.nvreason.latest_done(w.scan.id)
    from radar_desk.services.blobs import write_json
    from radar_desk.services.nvreason import nv_keys
    write_json(w.svc.storage, nv_keys(job.id)["nv_report"], {})
    assert w.svc.readers.plan(w.scan.id)["nv_read"]["status"] == "none"


class HookLLM:
    """Wraps the scripted model; the hook runs once, during the first call (a change landing mid-flight)."""

    def __init__(self, inner, hook) -> None:
        self.inner, self.hook = inner, hook

    @property
    def calls(self):
        return self.inner.calls

    async def stream_chat(self, messages, tools):
        hook, self.hook = self.hook, None
        if hook:
            hook()
        async for event in self.inner.stream_chat(messages, tools):
            yield event


def hooked(w, hook):
    w.svc.readers.llm = HookLLM(w.llm, hook)


def test_a_report_replaced_mid_flight_is_read_again(readers_world):
    w = readers_world([REPORT_READ, REPORT_READ])
    new = SYNTHETIC_REPORT + "\nAddendum: no change."
    hooked(w, lambda: w.svc.scans.put_report(w.scan.id, new))
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    plan = w.svc.readers.plan(w.scan.id)
    assert plan["report_read"]["status"] == "ready" and plan["report_read"]["text"] == new
    assert w.svc.readers.status(w.scan.id)["report_read"]["status"] == "ready"
    assert len(w.llm.calls) == 2


def test_a_report_removed_mid_flight_leaves_no_read(readers_world):
    w = readers_world([REPORT_READ])
    hooked(w, lambda: w.svc.scans.delete_report(w.scan.id))
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    assert not w.svc.storage.exists(readers_key(w.scan.id, "report_read"))


def test_a_scan_deleted_mid_flight_leaves_nothing(readers_world):
    w = readers_world([REPORT_READ])
    hooked(w, lambda: w.svc.scans.delete(w.scan.id))
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    for name in ("report_read", "nv_read", "queries"):
        assert not w.svc.storage.exists(readers_key(w.scan.id, name))


def test_a_retry_mid_flight_is_not_lost(readers_world):
    w = readers_world(["bad", "bad"])
    w.svc.scans.put_report(w.scan.id, SYNTHETIC_REPORT)
    assert w.svc.readers.status(w.scan.id)["report_read"]["status"] == "failed"
    w.llm.replies[:] = ["bad", "bad", REPORT_READ]
    hooked(w, lambda: w.svc.readers.kick(w.scan.id, force=True))
    assert w.svc.readers.kick(w.scan.id, force=True) is True
    assert w.svc.readers.status(w.scan.id)["report_read"]["status"] == "ready"
