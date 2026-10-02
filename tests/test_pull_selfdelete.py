"""The pull worker deletes its own RunPod pod when idle or cut off from the app, never while holding a job."""

from __future__ import annotations

import http.client
import http.server
import socket
import threading

import pytest

from radar_worker import pull
from test_pull_worker import INFO, FakeScorer, make_env  # noqa: F401  (the fixture)

POD = "pod-abc123"
KEY = "rpa_SECRETKEY0123456789"
FALLBACK = ("if this is a 401 or 403 the pod-scoped RUNPOD_API_KEY may not be allowed to delete pods; "
            "the app's hour cap or Stop now ends this pod")


class FakeRunPod:
    """A fake `request`: records each call and answers the scripted statuses (204 once the script runs out)."""

    def __init__(self, *statuses, clock=None, on_call=None):
        self.statuses = list(statuses)
        self.calls: list = []
        self.clock = clock
        self.on_call = on_call

    def __call__(self, method, url, headers):
        self.calls.append((method, url, dict(headers), self.clock() if self.clock else None))
        if self.on_call is not None:
            self.on_call()
        status = self.statuses.pop(0) if self.statuses else 204
        if isinstance(status, Exception):
            raise status
        return status, '{"title": "Forbidden"}' if status == 403 else ""


class Clock:
    """A fake clock that only moves when the worker sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list = []
        self.on_sleep = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(self.now)


def worker_for(client, clock: Clock, rp: FakeRunPod, idle_s=600.0, app_lost_s=600.0, **kw) -> pull.Worker:
    kw.setdefault("poll_s", 10)
    sd = pull.PodSelfDelete(POD, KEY, idle_s, app_lost_s, request=rp, sleep=clock.sleep)
    return pull.Worker(client, FakeScorer(), INFO, clock=clock, sleep=clock.sleep, self_delete=sd, **kw)


def claims(env) -> int:
    return sum(1 for _, p, _ in env.seen if p == "/worker/claim")


def dead_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def test_idle_pod_deletes_itself_after_the_last_claim(make_env, capsys):  # noqa: F811
    env = make_env()
    clock = Clock()
    rp = FakeRunPod(clock=clock)
    times = []

    class Timed(pull.DeskClient):
        def claim(self, info):
            times.append(clock.now)
            return super().claim(info)

    worker = worker_for(Timed(env.url, env.token), clock, rp)
    assert worker.run() == 0
    assert len(rp.calls) == 1
    method, url, headers, at = rp.calls[0]
    assert method == "DELETE" and url == f"https://api.runpod.io/v2/pods/{POD}"
    assert headers == {"Authorization": f"Bearer {KEY}", "User-Agent": "radar-worker"}
    assert at == 600 and times[-1] == 600 and max(clock.sleeps) == 10
    out = capsys.readouterr().out
    assert f"deleting pod {POD}: idle for 600 s with no job" in out and f"pod {POD} deleted" in out
    assert KEY not in out


def test_app_that_never_answers_deletes_the_pod(capsys):
    clock = Clock()
    rp = FakeRunPod(clock=clock)
    worker = worker_for(pull.DeskClient(dead_url(), "rdw_x", timeout=2), clock, rp, idle_s=3600)
    assert worker.run() == 0
    assert len(rp.calls) == 1 and rp.calls[0][3] == 600
    assert f"deleting pod {POD}: no answer from the app for 600 s" in capsys.readouterr().out


def test_app_lost_while_waiting_for_the_desk(capsys):
    clock = Clock()
    rp = FakeRunPod(clock=clock)
    worker = worker_for(pull.DeskClient(dead_url(), "rdw_x", timeout=2), clock, rp)
    assert worker.wait_for_desk() == 0
    assert len(rp.calls) == 1 and rp.calls[0][3] == 600
    assert "no answer from the app for 600 s" in capsys.readouterr().out


def test_html_503_from_the_start_counts_as_no_answer(make_env):  # noqa: F811
    env = make_env()
    for _ in range(300):
        env.fault("/worker/", 503)
    clock = Clock()
    rp = FakeRunPod(clock=clock)
    worker = worker_for(env.client(), clock, rp)
    assert worker.wait_for_desk() == 0
    assert len(rp.calls) == 1 and rp.calls[0][3] == 600
    assert all(p == "/worker/me" for _, p, _ in env.seen)

    clock2 = Clock()
    rp2 = FakeRunPod(clock=clock2)
    worker = worker_for(env.client(), clock2, rp2, idle_s=3600)
    assert worker.run() == 0
    assert len(rp2.calls) == 1 and rp2.calls[0][3] == 600 and claims(env) > 0


class SilentDuringJob(pull.DeskClient):
    """The first `failing_completes` completes fail; claims after the first fail when `vanish`.

    Heartbeats would fail too, but with a 3600 s lease the heartbeat thread waits 1200 s of real
    time, so in these tests it never sends one.
    """

    def __init__(self, *a, failing_completes=2, vanish=False, **kw):
        super().__init__(*a, **kw)
        self.failing_completes = failing_completes
        self.vanish = vanish
        self.claimed = 0

    def heartbeat(self, job_id, lease, progress):
        raise pull.DeskError(0, "connection refused")

    def complete(self, job_id, body):
        if self.failing_completes:
            self.failing_completes -= 1
            raise pull.DeskError(0, "connection refused")
        return super().complete(job_id, body)

    def claim(self, info):
        self.claimed += 1
        if self.vanish and self.claimed > 1:
            raise pull.DeskError(0, "connection refused")
        return super().claim(info)


def test_no_delete_while_holding_a_job_and_none_when_the_app_answers_again(make_env):  # noqa: F811
    """A 700 s job (fake clock) passes both deadlines; once the app answers the complete, nothing is deleted."""
    env = make_env(worker_lease_s=3600)
    job_id = env.queue()
    clock = Clock()
    rp = FakeRunPod(clock=clock)
    client = SilentDuringJob(env.url, env.token)
    worker = worker_for(client, clock, rp, idle_exit=30)
    worker.scorer = FakeScorer(on_score=lambda path: clock.sleep(700))
    assert worker.run() == 0
    assert env.job(job_id)["state"] == "done"
    assert rp.calls == []


def test_app_gone_after_a_long_job_deletes_once_back_to_claiming(make_env):  # noqa: F811
    env = make_env(worker_lease_s=3600)
    env.queue()
    clock = Clock()
    holder = {}
    rp = FakeRunPod(clock=clock, on_call=lambda: holder.setdefault("current", holder["worker"].current))
    client = SilentDuringJob(env.url, env.token, failing_completes=10**9, vanish=True)
    worker = worker_for(client, clock, rp)
    holder["worker"] = worker
    worker.scorer = FakeScorer(on_score=lambda path: clock.sleep(700))
    assert worker.run() == 0
    assert len(rp.calls) == 1 and holder["current"] is None
    assert client.claimed == 2  # the delete came only after the loop was back to claiming


def test_failed_claim_at_the_idle_deadline_does_not_delete(make_env):  # noqa: F811
    env = make_env()
    clock = Clock()
    outcomes = []

    def fault_at_deadline(now):
        if now >= 600 and not env.faults and not any(o == "error" for _, o in outcomes):
            env.fault("/worker/claim", 503)

    clock.on_sleep = fault_at_deadline
    rp = FakeRunPod(clock=clock, on_call=lambda: outcomes.append((clock.now, "delete")))

    class Timed(pull.DeskClient):
        def claim(self, info):
            try:
                answer = super().claim(info)
            except pull.DeskError:
                outcomes.append((clock.now, "error"))
                raise
            outcomes.append((clock.now, answer))
            return answer

    worker = worker_for(Timed(env.url, env.token), clock, rp)
    assert worker.run() == 0
    assert len(rp.calls) == 1 and rp.calls[0][3] == 600
    assert outcomes[-3:] == [(600, "error"), (600, None), (600, "delete")]


def test_job_claimed_at_the_deadline_is_run_and_the_idle_clock_restarts(make_env, capsys):  # noqa: F811
    env = make_env()
    clock = Clock()
    queued = []

    def queue_at_deadline(now):
        if now >= 600 and not queued:
            queued.append(env.queue())

    clock.on_sleep = queue_at_deadline
    rp = FakeRunPod(clock=clock)
    worker = worker_for(env.client(), clock, rp)
    assert worker.run() == 0
    assert env.job(queued[0])["state"] == "done"
    assert len(rp.calls) == 1 and rp.calls[0][3] == 1200
    assert "idle for 600 s with no job" in capsys.readouterr().out


def test_403_logs_the_fallback_disarms_and_the_loop_goes_on(make_env, capsys):  # noqa: F811
    env = make_env()
    clock = Clock()
    rp = FakeRunPod(403, clock=clock)
    worker = worker_for(env.client(), clock, rp, idle_exit=700)
    assert worker.run() == 0
    assert len(rp.calls) == 1 and worker.self_delete is None and clock.now >= 700
    out = capsys.readouterr().out
    assert (f'deleting pod {POD} failed (HTTP 403: {{"title": "Forbidden"}}); {FALLBACK}') in out
    assert "idle for 700 s; exiting" in out and KEY not in out


def test_transient_failures_are_retried_twice():
    sleeps = []
    rp = FakeRunPod(503, 204)
    assert pull.PodSelfDelete(POD, KEY, request=rp, sleep=sleeps.append).delete("test") is True
    assert len(rp.calls) == 2 and sleeps == [5]

    sleeps.clear()
    rp = FakeRunPod(429, OSError("reset"), 502)
    assert pull.PodSelfDelete(POD, KEY, request=rp, sleep=sleeps.append).delete("test") is False
    assert len(rp.calls) == 3 and sleeps == [5, 5]

    rp = FakeRunPod(404)
    assert pull.PodSelfDelete(POD, KEY, request=rp, sleep=sleeps.append).delete("test") is True


def test_an_incomplete_read_is_retried():
    sleeps = []
    rp = FakeRunPod(http.client.IncompleteRead(b""), 204)
    assert pull.PodSelfDelete(POD, KEY, request=rp, sleep=sleeps.append).delete("test") is True
    assert len(rp.calls) == 2 and sleeps == [5]


def test_main_without_a_pod_id_builds_no_self_delete(make_env, monkeypatch):  # noqa: F811
    env = make_env()
    built = []
    monkeypatch.setattr(pull, "PodSelfDelete", lambda *a, **kw: built.append(a))
    monkeypatch.setenv("RADAR_DESK_URL", env.url)
    monkeypatch.setenv("RADAR_WORKER_TOKEN", env.token)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    assert pull.main(["--idle-exit", "0", "--poll", "0.2"]) == 0
    assert built == []


def test_main_with_a_pod_id(make_env, monkeypatch, capsys):  # noqa: F811
    env = make_env()
    runpod = FakeRunPod()
    monkeypatch.setattr(pull, "_urllib_request", runpod)
    monkeypatch.setenv("RADAR_DESK_URL", env.url)
    monkeypatch.setenv("RADAR_WORKER_TOKEN", env.token)
    monkeypatch.setenv("RUNPOD_POD_ID", POD)
    monkeypatch.setenv("RUNPOD_API_KEY", "")
    assert pull.main(["--idle-exit", "0", "--poll", "0.2"]) == 0
    assert "RUNPOD_POD_ID is set but RUNPOD_API_KEY is empty; the pod cannot delete itself" in capsys.readouterr().out

    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RADAR_POD_IDLE_DELETE_S", "300")
    assert pull.main(["--idle-exit", "0", "--poll", "0.2"]) == 0
    out = capsys.readouterr().out
    assert f"pod {POD} deletes itself after 300 s without a job or 600 s without the app" in out
    assert KEY not in out
    assert runpod.calls == []  # --idle-exit 0 ends the loop before any delete


@pytest.mark.parametrize("bad", ["soon", "0", "-5"])
def test_main_with_bad_deadlines_runs_without_self_delete(make_env, monkeypatch, capsys, bad):  # noqa: F811
    env = make_env()
    built = []
    monkeypatch.setattr(pull, "PodSelfDelete", lambda *a, **kw: built.append(a))
    monkeypatch.setenv("RADAR_DESK_URL", env.url)
    monkeypatch.setenv("RADAR_WORKER_TOKEN", env.token)
    monkeypatch.setenv("RUNPOD_POD_ID", POD)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RADAR_POD_APP_LOST_DELETE_S", bad)
    assert pull.main(["--idle-exit", "0", "--poll", "0.2"]) == 0
    assert built == []
    out = capsys.readouterr().out
    assert ("RADAR_POD_IDLE_DELETE_S and RADAR_POD_APP_LOST_DELETE_S must be numbers of seconds above 0; "
            "the pod cannot delete itself") in out
    assert "RUNPOD_API_KEY is empty" not in out


@pytest.fixture
def runpod_server():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_DELETE(self):
            seen.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v2", seen
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_default_request_sends_delete_with_both_headers(runpod_server, monkeypatch, capsys):
    base, seen = runpod_server
    monkeypatch.setattr(pull, "RUNPOD_API_URL", base)
    assert pull.PodSelfDelete(POD, KEY).delete("idle for 600 s with no job") is True
    assert len(seen) == 1
    method, path, headers = seen[0]
    assert method == "DELETE" and path == f"/v2/pods/{POD}"
    assert headers["authorization"] == f"Bearer {KEY}" and headers["user-agent"] == "radar-worker"
    out = capsys.readouterr().out
    assert f"pod {POD} deleted" in out and KEY not in out
