"""The pull worker: claim a job from the app's job desk, score it, upload the artefacts, report back.

    python -m radar_worker.pull [--once] [--idle-exit SECONDS] [--poll SECONDS]

Env: RADAR_DESK_URL and RADAR_WORKER_TOKEN (both required), RADAR_WORKER_ID (default the host
name plus 4 hex), RADAR_DEVICE (auto), RADAR_WEIGHTS_RESOLVED else RADAR_WEIGHTS_DIR (default
/workspace/radar-weights), RADAR_VENDOR_DIR, RADAR_IMAGE (reported as versions.image_id). On a
RunPod pod also RUNPOD_POD_ID and RUNPOD_API_KEY (both set by RunPod), RADAR_POD_IDLE_DELETE_S
(default 600) and RADAR_POD_APP_LOST_DELETE_S (default 600).

Every request carries the worker token, `ngrok-skip-browser-warning: 1` and a User-Agent. An
empty claim backs off 1, 2, 4 .. seconds up to --poll. Only the app's own answers count: a
network error, a 5xx or a 4xx whose body is not the app's JSON {"detail": ...} (a tunnel's page
while the app is down) is transient, backs off the same way and never stops the loop. The
download and the complete, fail and release calls retry transient errors for about one lease.
A thread heartbeats every `heartbeat_s` while a job runs; an app 409 there means the lease is
lost, and the job's upload and completion are skipped. A model that does not load releases the
job and ends the worker with 1, so one bad machine does not fail the queue. SIGTERM or SIGINT
finishes a job that is scoring or uploading, releases one that has not started scoring (a
download in progress is cut short), and exits 0; a second signal releases the job and exits at
once. Exit 0 after --once or --idle-exit, 2 on a config error (missing env, token refused at
start), 1 otherwise.

On a RunPod pod (RUNPOD_POD_ID set) the worker deletes its own pod through RunPod's API, because
RunPod restarts a container whose process ends. It does so when RADAR_POD_IDLE_DELETE_S pass
without a job (the claim at that moment is the last chance: a job it returns is run, and a claim the
app did not answer only counts toward the next rule) or when
RADAR_POD_APP_LOST_DELETE_S pass without an answer from the app itself, also while waiting for
the desk at start. It never deletes while a job is in hand. After a successful delete the worker
exits 0. A delete that fails is logged once and the worker carries on without self-delete; the
app's hour cap or Stop then ends the pod. --idle-exit is checked before either rule.

Standard library and radar_worker only at module level; torch and the model are imported by
`RealScorer` when it loads.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import re
import secrets
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from radar_worker import io as wio
from radar_worker import job as jobmod
from radar_worker.weights import check_weights, code_commit

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_DIR = PACKAGE_PARENT / "vendor" / "damo-radar" / "RADAR_inference"
MANIFEST = PACKAGE_PARENT / "weights.json"
DEFAULT_WEIGHTS_DIR = "/workspace/radar-weights"
USER_AGENT = "radar-worker"
RUNPOD_API_URL = "https://api.runpod.io/v2"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}Z {msg}", flush=True)


# ---------------------------------------------------------------- the desk


class DeskError(Exception):
    """A non-2xx answer from the desk; status 0 for a network failure.

    `from_app` is False when a 4xx body was not the app's JSON {"detail": ...}, such as a
    tunnel's HTML page while the app behind it is down.
    """

    def __init__(self, status: int, detail: str, from_app: bool = True):
        self.status = int(status)
        self.detail = detail
        self.from_app = from_app
        super().__init__(f"HTTP {status}: {detail}")

    @property
    def transient(self) -> bool:
        """True when trying again later may work: no answer, a 5xx, or a 4xx that is not the app's."""
        return self.status == 0 or self.status >= 500 or not self.from_app


def transient_transfer(err: wio.TransferError) -> bool:
    """The same rule for a download: no answer, a 5xx, or a 4xx whose body is not the app's JSON."""
    return err.status == 0 or err.status >= 500 or not _app_json(err.body_snippet)


def _app_json(body) -> bool:
    try:
        data = json.loads(body)
    except ValueError:
        return False
    return isinstance(data, dict) and "detail" in data


class DeskClient:
    """The worker API of the app at `base_url`, over urllib."""

    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.contacted = None  # called whenever the app itself answered

    def _answered(self) -> None:
        if self.contacted is not None:
            self.contacted()

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "ngrok-skip-browser-warning": "1",
                "User-Agent": USER_AGENT, "Accept": "application/json"}

    def url(self, path: str) -> str:
        """`path` joined to the base URL, always; the token never goes to another host."""
        return self.base_url + "/" + path.lstrip("/")

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        headers = self.headers()
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url(path), data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
                if resp.status == 204 or not raw:
                    self._answered()
                    return None
                parsed = json.loads(raw)
                self._answered()
                return parsed
        except urllib.error.HTTPError as err:
            try:
                raw = err.read()
            except OSError:
                raw = b""
            if _app_json(raw):
                if err.code < 500:
                    self._answered()
                raise DeskError(err.code, str(json.loads(raw)["detail"])) from None
            raise DeskError(err.code, raw[:300].decode("utf-8", "replace"), from_app=False) from None
        except (urllib.error.URLError, OSError, ValueError) as err:
            raise DeskError(0, str(getattr(err, "reason", err))) from None

    def me(self) -> dict:
        return self._call("GET", "/worker/me")

    def claim(self, worker_info: dict) -> dict | None:
        """The claimed job, or None when nothing is queued."""
        return self._call("POST", "/worker/claim", {"worker": worker_info})

    def heartbeat(self, job_id: str, lease: str, progress: str | None) -> dict:
        return self._call("POST", f"/worker/jobs/{job_id}/heartbeat", {"lease": lease, "progress": progress})

    def download_source(self, url: str, dest, on_chunk=None) -> int:
        return wio.download(self.url(url), dest, headers=self.headers(), on_chunk=on_chunk)

    def upload_artefact(self, url: str, path, content_type: str) -> None:
        wio.upload(path, self.url(url), content_type, headers=self.headers())

    def complete(self, job_id: str, body: dict) -> dict:
        return self._call("POST", f"/worker/jobs/{job_id}/complete", body)

    def fail(self, job_id: str, lease: str, error: dict) -> dict:
        return self._call("POST", f"/worker/jobs/{job_id}/fail", {"lease": lease, "error": error})

    def release(self, job_id: str, lease: str, timeout: float | None = None) -> dict:
        return self._call("POST", f"/worker/jobs/{job_id}/release", {"lease": lease}, timeout=timeout)


# ---------------------------------------------------------------- the pod


def _urllib_request(method: str, url: str, headers: dict) -> tuple[int, str]:
    """(status, body text) for one request over urllib; a transport error is raised."""
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        try:
            body = err.read().decode("utf-8", "replace")
        except OSError:
            body = ""
        return err.code, body


class PodSelfDelete:
    """Deletes this pod through RunPod's v2 API. Built only when RUNPOD_POD_ID is set.

    `request(method, url, headers)` returns (status, body text); the default goes over urllib.
    The API key goes in the Authorization header only and is never logged.
    """

    def __init__(self, pod_id: str, api_key: str, idle_s: float = 600.0, app_lost_s: float = 600.0,
                 request=None, sleep=time.sleep):
        self.pod_id = pod_id
        self._api_key = api_key
        self.idle_s = float(idle_s)
        self.app_lost_s = float(app_lost_s)
        self.request = request or _urllib_request
        self.sleep = sleep

    def delete(self, reason: str) -> bool:
        """True when RunPod deleted the pod (or no longer knows it); a transient failure is tried 3 times."""
        log(f"deleting pod {self.pod_id}: {reason}")
        url = f"{RUNPOD_API_URL}/pods/{urllib.parse.quote(self.pod_id, safe='')}"
        headers = {"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT}
        status, body = 0, ""
        for attempt in range(3):
            if attempt:
                self.sleep(5)
            try:
                status, body = self.request("DELETE", url, headers)
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as err:
                status, body = 0, str(getattr(err, "reason", err))
                continue
            if 200 <= status < 300 or status == 404:
                log(f"pod {self.pod_id} deleted")
                return True
            if status != 429 and status < 500:
                break
        log(f"deleting pod {self.pod_id} failed (HTTP {status}: {body[:200]}); if this is a 401 or 403 the "
            "pod-scoped RUNPOD_API_KEY may not be allowed to delete pods; the app's hour cap or Stop now "
            "ends this pod")
        return False


# ---------------------------------------------------------------- the loop


class _Release(Exception):
    """Stop was asked for before scoring started: give the job back."""


class _Lost(Exception):
    """The lease is gone: skip whatever is left of the job."""


class _LoadFailed(Exception):
    """The model did not load: give the job back and stop claiming."""


class _Running:
    """The job in hand: its claim, its phase and whether its lease was lost."""

    def __init__(self, claim: dict):
        self.claim = claim
        self.job_id = claim["job_id"]
        self.lease = claim["lease"]
        self.progress = "claimed"
        self.lease_s = float(claim["lease_s"])
        self.renewed_at = 0.0  # clock time of the claim or the last accepted heartbeat
        self.lost = threading.Event()
        self.done = threading.Event()


class Worker:
    """Claims and runs jobs until stopped, `once` is satisfied or `idle_exit` seconds pass without a job.

    With `self_delete` it also deletes its pod when idle or cut off from the app for too long.
    `worker_info` is the dict sent with each claim, or a callable that returns it. `sleep` waits
    between empty claims and download retries; the default wakes early when a stop is requested.
    Retries of complete, fail and release wait with `sleep` when given, else `time.sleep`, so a
    stop does not hurry them.
    """

    def __init__(self, client: DeskClient, scorer, worker_info, *, poll_s: float = 10.0, once: bool = False,
                 idle_exit: float | None = None, clock=time.monotonic, sleep=None,
                 self_delete: PodSelfDelete | None = None):
        self.client = client
        self.scorer = scorer
        self.worker_info = worker_info
        self.poll_s = poll_s
        self.once = once
        self.idle_exit = idle_exit
        self.clock = clock
        self._stop = threading.Event()
        self.sleep = sleep or self._stop.wait
        self._settle_sleep = sleep or time.sleep
        self.current: _Running | None = None
        self.load_failed = False
        self.self_delete = self_delete
        self.last_contact = clock()
        client.contacted = self._contacted

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    def abort(self) -> None:
        """Release the job in hand, if any; for a second signal."""
        running = self.current
        if running is None or running.lost.is_set():
            return
        try:
            self.client.release(running.job_id, running.lease, timeout=5)
            log(f"released {running.job_id}")
        except DeskError as err:
            log(f"release of {running.job_id} failed: {err}")

    def _info(self) -> dict:
        return self.worker_info() if callable(self.worker_info) else self.worker_info

    def _contacted(self) -> None:
        self.last_contact = self.clock()

    def _backoff(self, delay: float, idle_since: float | None = None, between_claims: bool = False) -> float:
        """Sleep `delay`, cut short at the next self-delete deadline when waiting between claims."""
        wait = delay
        if between_claims and self.self_delete is not None and self.current is None:
            due = self.last_contact + self.self_delete.app_lost_s
            if idle_since is not None:
                due = min(due, idle_since + self.self_delete.idle_s)
            wait = min(delay, max(due - self.clock(), 0.0))
        self.sleep(wait)
        return min(delay * 2, self.poll_s)

    def _delete_pod_if_due(self, idle_since: float | None) -> bool:
        """Delete the pod when no job is in hand and a deadline has passed; True once it is deleted.

        A failed delete turns self-delete off for the rest of the process.
        """
        if self.self_delete is None or self.current is not None:
            return False
        now = self.clock()
        if now - self.last_contact >= self.self_delete.app_lost_s:
            reason = f"no answer from the app for {int(now - self.last_contact)} s"
        elif idle_since is not None and now - idle_since >= self.self_delete.idle_s:
            reason = f"idle for {int(now - idle_since)} s with no job"
        else:
            return False
        if self.self_delete.delete(reason):
            return True
        self.self_delete = None
        return False

    def wait_for_desk(self) -> int | None:
        """Check the token with /worker/me, retrying while the app is unreachable.

        None when the desk accepted the token, 2 when it refused it, 0 when a stop came first.
        """
        delay = min(1.0, self.poll_s)
        while not self.stopping:
            try:
                me = self.client.me()
            except DeskError as err:
                if err.transient:
                    if self._delete_pod_if_due(None):
                        return 0
                    log(f"desk not reachable ({err}); retrying in {delay:g} s")
                    delay = self._backoff(delay, between_claims=True)
                    continue
                log(f"the desk refused this worker ({err}); check RADAR_DESK_URL and RADAR_WORKER_TOKEN")
                return 2
            log(f"connected to {self.client.base_url} as token {me['token']['name']} ({me['token']['id']})")
            return None
        return 0

    def run(self) -> int:
        delay = min(1.0, self.poll_s)
        idle_since = self.clock()
        while not self.stopping:
            answered = False
            try:
                claim = self.client.claim(self._info())
                answered = True
            except DeskError as err:
                if not err.transient and err.status == 401:
                    log("the desk refused the worker token; stopping")
                    return 2
                if not err.transient:
                    log(f"claim refused: {err}; stopping")
                    return 1
                log(f"claim failed ({err}); retrying in {delay:g} s")
                claim = None
            if claim is None:
                if self.idle_exit is not None and self.clock() - idle_since >= self.idle_exit:
                    log(f"idle for {self.idle_exit:g} s; exiting")
                    return 0
                if self._delete_pod_if_due(idle_since if answered else None):
                    return 0
                delay = self._backoff(delay, idle_since, between_claims=True)
                continue
            self._run_one(claim)
            if self.load_failed:
                return 1
            delay = min(1.0, self.poll_s)
            idle_since = self.clock()
            if self.once:
                return 0
        log("stopped")
        return 0

    # One job

    def _heartbeats(self, running: _Running, every: float) -> None:
        while not running.done.wait(every):
            try:
                self.client.heartbeat(running.job_id, running.lease, running.progress)
                running.renewed_at = self.clock()
            except DeskError as err:
                if not err.transient and err.status in (404, 409):
                    if not running.lost.is_set():
                        log(f"lease on {running.job_id} lost ({err}); the job will not be uploaded or completed")
                    running.lost.set()
                    return
                log(f"heartbeat for {running.job_id} failed: {err}")

    def _run_one(self, claim: dict) -> None:
        running = _Running(claim)
        running.renewed_at = self.clock()
        self.current = running
        log(f"claimed {running.job_id} ({claim['scan']['filename']}, {claim['scan']['size_bytes']} bytes)")
        beat = threading.Thread(target=self._heartbeats, args=(running, max(float(claim["heartbeat_s"]), 0.5)),
                                daemon=True, name=f"heartbeat-{running.job_id}")
        beat.start()
        try:
            self._score(running)
        finally:
            running.done.set()
            self.current = None
            beat.join(timeout=5)

    def _check(self, running: _Running) -> None:
        if running.lost.is_set():
            raise _Lost
        if self.stopping:
            raise _Release

    def _score(self, running: _Running) -> None:
        claim = running.claim
        worker, scorer = self, self.scorer

        class Phased:
            """The scorer with the job's phase recorded for heartbeats and stop checks before scoring."""

            def load(self, log):
                running.progress = "loading model"
                try:
                    refused = scorer.load(log)
                except Exception as exc:  # any load failure stops the worker
                    raise _LoadFailed(f"{type(exc).__name__}: {exc}") from exc
                if refused:
                    raise _LoadFailed(f"{refused.get('class')}: {refused.get('message')}")

            def score(self, path, log):
                worker._check(running)
                running.progress = "scoring"
                return scorer.score(path, log)

            def versions(self):
                return scorer.versions()

        def on_chunk(length: int) -> None:
            worker._check(running)

        def fetch(work: Path, log) -> Path:
            worker._check(running)
            running.progress = "downloading"
            dest = Path(work) / source_name(claim["scan"].get("filename"))
            delay = min(1.0, worker.poll_s)
            give_up = worker.clock() + running.lease_s
            while True:
                try:
                    worker.client.download_source(claim["source_url"], dest, on_chunk=on_chunk)
                    break
                except wio.TransferError as err:
                    if not transient_transfer(err):
                        if err.status == 409:
                            raise _Lost from None
                        raise
                    if worker.clock() >= give_up:
                        raise
                    log(f"download failed ({err}); retrying in {delay:g} s")
                    delay = worker._backoff(delay)
                    worker._check(running)
            worker._check(running)
            return dest

        def publish(name: str, path: Path, content_type: str) -> None:
            if running.lost.is_set():
                raise _Lost
            running.progress = "uploading"
            self.client.upload_artefact(claim["artefact_urls"][name], path, content_type)

        try:
            result = jobmod.run_job(running.job_id, fetch, publish, claim.get("artefact_keys"), Phased())
        except _Release:
            self._settle("release", running, lambda: self.client.release(running.job_id, running.lease))
            return
        except _LoadFailed as exc:
            log(f"{running.job_id}: the model did not load ({exc}); releasing the job and stopping")
            self.load_failed = True
            self._settle("release", running, lambda: self.client.release(running.job_id, running.lease))
            return
        except _Lost:
            log(f"{running.job_id}: lease lost, upload and completion skipped")
            return
        except Exception as exc:  # noqa: BLE001  every failure is reported to the desk
            traceback.print_exc()
            error = {"class": type(exc).__name__, "message": str(exc)[:2000]}
            self._settle("fail", running, lambda: self.client.fail(running.job_id, running.lease, error))
            return
        body = dict(jobmod.plain(result), lease=running.lease)
        self._settle("complete", running, lambda: self.client.complete(running.job_id, body))

    def _settle(self, what: str, running: _Running, call) -> None:
        """Make the call, retrying transient errors until the lease would have run out."""
        delay = min(1.0, self.poll_s)
        while True:
            if running.lost.is_set():
                log(f"{running.job_id}: lease lost, {what} skipped")
                return
            try:
                job = call()
                break
            except DeskError as err:
                if not err.transient:
                    log(f"{running.job_id}: {what} refused ({err}); moving on")
                    return
                if self.clock() >= running.renewed_at + running.lease_s:
                    log(f"{running.job_id}: {what} failed ({err}); the lease has run out, moving on")
                    return
                log(f"{running.job_id}: {what} failed ({err}); retrying in {delay:g} s")
                self._settle_sleep(delay)
                delay = min(delay * 2, self.poll_s)
        state = job.get("state") if isinstance(job, dict) else None
        log(f"{running.job_id}: {what} -> {state}")


def source_name(filename: str | None) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename or "").name)
    return name.lstrip(".") or "source.nii.gz"


# ---------------------------------------------------------------- the real scorer


class RealScorer:
    """RADAR on this machine: weights checked once against the manifest, the model loaded on `device`."""

    def __init__(self, weights_dir, device: str = "auto", manifest_path=MANIFEST, vendor_dir=DEFAULT_VENDOR_DIR):
        self.weights_dir = str(weights_dir)
        self.device = device
        self.manifest_path = Path(manifest_path)
        self.vendor_dir = str(vendor_dir)
        self._weights: dict | None = None
        self._device = None
        self.loaded = None

    def load(self, log):
        if self._weights is None:
            manifest = json.loads(self.manifest_path.read_text())
            t = time.perf_counter()
            self._weights = check_weights(self.weights_dir, manifest)
            ok = self._weights["ok"]
            log(f"weights check {'ok' if ok else 'FAILED'} in {time.perf_counter() - t:.1f}s"
                + ("" if ok else ": " + "; ".join(self._weights["problems"])))
        if not self._weights["ok"]:
            return {"class": "weights_mismatch", "message": "; ".join(self._weights["problems"])}
        os.environ["RADAR_VENDOR_DIR"] = self.vendor_dir
        from radar_worker import infer

        self._device = infer.resolve_device(self.device)
        self.device = str(self._device)
        self.loaded = infer.load_model(self.weights_dir, self._device)
        return None

    def score(self, path: str, log) -> dict:
        from radar_worker import infer

        return infer.score_file(path, self.loaded, log)

    def versions(self) -> dict:
        import torch

        from radar_worker import infer

        return {
            "code_commit": code_commit(Path(self.vendor_dir).parent / "VENDORED.md"),
            "checkpoint_sha256": (self._weights or {}).get("checkpoint_sha256"),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda) if torch.version.cuda else None,
            "gpu": infer.device_name(self._device) if self._device is not None else None,
            "image_id": os.environ.get("RADAR_IMAGE"),
        }

    def gpu_name(self) -> str | None:
        """The CUDA card's name once the model is loaded on CUDA, else None."""
        if self._device is None or self._device.type != "cuda":
            return None
        import torch

        return str(torch.cuda.get_device_name(self._device))


def default_worker_id() -> str:
    return os.environ.get("RADAR_WORKER_ID") or f"{socket.gethostname()}-{secrets.token_hex(2)}"


def worker_info(scorer, worker_id: str) -> dict:
    """What the worker tells the desk with each claim; torch fields stay None until the model is loaded."""
    torch = sys.modules.get("torch")
    gpu_name = getattr(scorer, "gpu_name", None)
    return {
        "id": worker_id,
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name() if callable(gpu_name) else None,
        "device": getattr(scorer, "device", None),
        "versions": {
            "torch": str(torch.__version__) if torch is not None else None,
            "cuda": str(torch.version.cuda) if torch is not None and torch.version.cuda else None,
            "python": platform.python_version(),
            "image": os.environ.get("RADAR_IMAGE"),
        },
    }


# ---------------------------------------------------------------- CLI


def install_signals(worker: Worker) -> dict:
    """SIGTERM and SIGINT ask the worker to stop; a second one releases the job and exits at once.

    Returns the previous handlers so they can be put back.
    """
    def handler(signum, frame):
        if worker.stopping:
            log("second signal: releasing and exiting")
            worker.abort()
            os._exit(0)
        log(f"signal {signum}: finishing up, then exiting (send again to exit at once)")
        worker.request_stop()

    return {sig: signal.signal(sig, handler) for sig in (signal.SIGTERM, signal.SIGINT)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m radar_worker.pull",
                                     description="Pull scoring jobs from the radar-desk app and run them.")
    parser.add_argument("--once", action="store_true", help="exit after the first job")
    parser.add_argument("--idle-exit", type=float, default=None, metavar="SECONDS",
                        help="exit after this many seconds without a job")
    parser.add_argument("--poll", type=float, default=10.0, metavar="SECONDS",
                        help="longest wait between empty claims (default 10)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base_url = os.environ.get("RADAR_DESK_URL", "").strip()
    token = os.environ.get("RADAR_WORKER_TOKEN", "").strip()
    missing = [n for n, v in (("RADAR_DESK_URL", base_url), ("RADAR_WORKER_TOKEN", token)) if not v]
    if missing:
        log(f"missing environment: {', '.join(missing)}")
        return 2
    weights_dir = (os.environ.get("RADAR_WEIGHTS_RESOLVED") or os.environ.get("RADAR_WEIGHTS_DIR")
                   or DEFAULT_WEIGHTS_DIR)
    scorer = RealScorer(weights_dir, os.environ.get("RADAR_DEVICE", "auto").strip() or "auto", MANIFEST,
                        os.environ.get("RADAR_VENDOR_DIR") or DEFAULT_VENDOR_DIR)
    self_delete = None
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    if pod_id:
        api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
        try:
            idle_s = float(os.environ.get("RADAR_POD_IDLE_DELETE_S", "").strip() or 600)
            app_lost_s = float(os.environ.get("RADAR_POD_APP_LOST_DELETE_S", "").strip() or 600)
            deadlines_ok = idle_s > 0 and app_lost_s > 0
        except ValueError:
            deadlines_ok = False
        if not deadlines_ok:
            log("RADAR_POD_IDLE_DELETE_S and RADAR_POD_APP_LOST_DELETE_S must be numbers of seconds above 0; "
                "the pod cannot delete itself")
        elif api_key:
            self_delete = PodSelfDelete(pod_id, api_key, idle_s, app_lost_s)
    worker_id = default_worker_id()
    worker = Worker(DeskClient(base_url, token), scorer, lambda: worker_info(scorer, worker_id),
                    poll_s=args.poll, once=args.once, idle_exit=args.idle_exit, self_delete=self_delete)
    log(f"worker {worker_id}, weights in {weights_dir}, device {scorer.device}")
    if self_delete is not None:
        log(f"pod {pod_id} deletes itself after {idle_s:g} s without a job or {app_lost_s:g} s without the app")
    elif pod_id and deadlines_ok:
        log("RUNPOD_POD_ID is set but RUNPOD_API_KEY is empty; the pod cannot delete itself")
    previous = install_signals(worker) if threading.current_thread() is threading.main_thread() else {}
    try:
        code = worker.wait_for_desk()
        if code is not None:
            return code
        return worker.run()
    except Exception:  # noqa: BLE001  anything unexpected ends the worker with 1
        traceback.print_exc()
        return 1
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


if __name__ == "__main__":
    sys.exit(main())
