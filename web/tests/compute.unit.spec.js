// Unit tests for src/compute.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import { computePhrase, fromGpuStatus, modeControl, workerUrl } from "../src/compute.js";

const NOW = Date.parse("2026-10-02T12:00:00Z");
const at = (secondsAgo) => new Date(NOW - secondsAgo * 1000).toISOString();
const base = { mode: "runpod", changeable: true, runpod: { configured: true }, pod: null, in_flight: null, queued: 0, held: [], problem: null };
const pod = (extra) => ({ id: "pod_1", runpod_id: "rp123", gpu: "NVIDIA L4", cost_per_hr: 0.39, created_at: at(200), started_at: at(150), job_id: null, idle_s: null, idle_delete_s: 600, ...extra });
const phrase = (s) => computePhrase({ ...base, ...s }, NOW);

test("modal names the requested GPUs", () => {
  expect(phrase({ mode: "modal", gpus: ["L4", "L40S"] })).toEqual({ text: "Modal, L4 or L40S, scales to zero", tone: "idle" });
});

test("own workers: none online, some online, count unknown", () => {
  const worker = { mode: "worker", runpod: { configured: false } };
  expect(phrase({ ...worker, workers_online: 0 })).toEqual({ text: "Waiting for your workers; start one with the docker run line below", tone: "idle" });
  expect(phrase({ ...worker, workers_online: 2 })).toEqual({ text: "Own GPU workers, 2 online", tone: "idle" });
  expect(phrase(worker)).toEqual({ text: "Own GPU workers", tone: "idle" });
});

test("own workers scoring get the generic suffix", () => {
  const s = { mode: "worker", workers_online: 1, in_flight: { job_id: "job_77777777x", backend: "worker" }, scan: "b.nii" };
  expect(phrase(s)).toEqual({ text: "Own GPU workers, 1 online, scoring b.nii", tone: "busy" });
});

test("RunPod pod mode without a pod", () => {
  expect(phrase({})).toEqual({ text: "RunPod pod, none running, starts when a job is queued", tone: "idle" });
});

test("pod waiting for the tunnel", () => {
  expect(phrase({ pod: pod({ phase: "tunnel", runpod_id: null, gpu: null, cost_per_hr: null, started_at: null }) })).toEqual({
    text: "Starting pod, waiting for the tunnel, 3:20",
    tone: "busy",
  });
});

test("pod starting", () => {
  expect(phrase({ pod: pod({ phase: "starting" }) })).toEqual({
    text: "Starting pod rp123, NVIDIA L4 at $0.39/h, about 4 min (image pull, then weights), up 2:30",
    tone: "busy",
  });
});

test("pod ready and scoring names the scan", () => {
  const s = { pod: pod({ phase: "ready", job_id: "job_abcdef123" }), in_flight: { job_id: "job_abcdef123", backend: "worker" }, scan: "case-7.nii.gz" };
  expect(phrase(s)).toEqual({ text: "Pod rp123, NVIDIA L4, $0.39/h, up 2:30, scoring case-7.nii.gz", tone: "busy" });
  expect(phrase({ ...s, scan: null }).text).toBe("Pod rp123, NVIDIA L4, $0.39/h, up 2:30, scoring job_abcd");
});

test("pod ready and idle says how long and when it deletes itself", () => {
  expect(phrase({ pod: pod({ phase: "ready", idle_s: 179 }) })).toEqual({ text: "Pod rp123, NVIDIA L4, $0.39/h, idle 2 min, deletes itself after 10 idle min", tone: "busy" });
  expect(phrase({ pod: pod({ phase: "ready" }) }).text).toBe("Pod rp123, NVIDIA L4, $0.39/h, idle, deletes itself after 10 idle min");
  expect(phrase({ pod: pod({ phase: "ready", idle_s: 0, idle_delete_s: undefined }) }).text).toBe("Pod rp123, NVIDIA L4, $0.39/h, idle 0 min");
});

test("a job running elsewhere is named", () => {
  const s = { mode: "runpod", in_flight: { job_id: "job_99999999x", backend: "modal" }, scan: "a.nii" };
  expect(phrase(s)).toEqual({ text: "RunPod pod, none running, starts when a job is queued, scoring a.nii", tone: "busy" });
});

test("a problem is its own sentence and wins the tone", () => {
  const held = [{ job_id: "j1", hold_reason: "no_gpu" }];
  expect(phrase({ held, problem: "No L4 or RTX 4090 in EU-RO-1, retrying in 60 s" })).toEqual({
    text: "RunPod pod, none running, starts when a job is queued. No L4 or RTX 4090 in EU-RO-1, retrying in 60 s.",
    tone: "problem",
  });
  expect(phrase({ problem: "Budget reached." }).text).toBe("RunPod pod, none running, starts when a job is queued. Budget reached.");
});

test("held jobs without a problem are counted", () => {
  const held = [{ job_id: "j1", hold_reason: "budget" }, { job_id: "j2", hold_reason: "budget" }];
  expect(phrase({ held })).toEqual({ text: "RunPod pod, none running, starts when a job is queued. 2 job(s) held (budget).", tone: "held" });
});

test("fake backend", () => {
  expect(phrase({ mode: "fake", changeable: false })).toEqual({ text: "Fake backend", tone: "idle" });
});

test("the scans strip's gpu_status maps onto the same phrase", () => {
  const g = {
    backend: "worker",
    compute_mode: "runpod",
    runpod_configured: true,
    gpu_requested: ["L4"],
    pod: pod({ phase: "ready", job_id: "job_1" }),
    in_flight: { id: "job_1", scan_id: "scan_1", backend: "worker" },
    queued: [{ id: "job_2", hold_reason: null }],
    held: [],
  };
  const s = fromGpuStatus(g, (id) => `name of ${id}`);
  expect(s).toMatchObject({ mode: "runpod", in_flight: { job_id: "job_1", backend: "worker" }, queued: 1, held: [], scan: "name of scan_1" });
  expect(computePhrase(s, NOW)).toEqual({ text: "Pod rp123, NVIDIA L4, $0.39/h, up 2:30, scoring name of scan_1", tone: "busy" });
  const idle = { backend: "worker", compute_mode: "runpod", runpod_configured: true, gpu_requested: [], pod: null, in_flight: null, queued: [], held: [] };
  expect(computePhrase(fromGpuStatus(idle), NOW).text).toBe("RunPod pod, none running, starts when a job is queued");
  // The strip has no worker count.
  expect(computePhrase(fromGpuStatus({ ...idle, compute_mode: "worker", runpod_configured: false }), NOW).text).toBe("Own GPU workers");
  const sg = (status) => ({
    backend: "serverless",
    compute_mode: "serverless",
    runpod_configured: true,
    gpu_requested: [],
    pod: null,
    in_flight: { id: "job_5555aaaa9", scan_id: "scan_1", backend: "serverless" },
    queued: [],
    held: [],
    serverless: sl(sjob(status), { health }).serverless,
  });
  const queued = computePhrase(fromGpuStatus(sg("IN_QUEUE"), (id) => `name of ${id}`), NOW);
  expect(queued).toEqual({ text: "RunPod serverless, job queued 1:35, waiting for a worker (1 running, 0 idle)", tone: "busy" });
  const scoring = computePhrase(fromGpuStatus(sg("IN_PROGRESS"), (id) => `name of ${id}`), NOW);
  expect(scoring).toEqual({ text: "RunPod serverless, scoring name of scan_1, 1:35", tone: "busy" });
});

test("mode modal with a pod still draining names the pod", () => {
  const g = { backend: "modal", compute_mode: "modal", runpod_configured: true, gpu_requested: ["L4"], pod: pod({ phase: "ready", idle_s: 0 }), in_flight: null, queued: [], held: [] };
  expect(computePhrase(fromGpuStatus(g), NOW)).toEqual({ text: "Pod rp123, NVIDIA L4, $0.39/h, idle 0 min, deletes itself after 10 idle min", tone: "busy" });
});

const sl = (job, extra) => ({
  mode: "serverless",
  in_flight: job ? { job_id: job.job_id, backend: "serverless" } : null,
  serverless: { configured: true, endpoint_id: "abc123", gpus: ["AMPERE_24", "ADA_24"], idle_s: 60, price_per_s: 0.00031, health: null, job, ...extra },
});
const sjob = (status) => ({ job_id: "job_5555aaaa9", status, submitted_at: at(95), status_at: at(5) });
const health = { workers: { idle: 0, running: 1 }, jobs: { completed: 1, failed: 0, inProgress: 0, inQueue: 1, retried: 0 }, at: at(5) };

test("serverless queued without health", () => {
  expect(phrase(sl(sjob("IN_QUEUE")))).toEqual({ text: "RunPod serverless, job queued 1:35, waiting for a worker", tone: "busy" });
});

test("serverless queued with health counts the workers", () => {
  expect(phrase(sl(sjob("IN_QUEUE"), { health }))).toEqual({
    text: "RunPod serverless, job queued 1:35, waiting for a worker (1 running, 0 idle)",
    tone: "busy",
  });
});

test("serverless queued with health but no worker counts", () => {
  const queued = "RunPod serverless, job queued 1:35, waiting for a worker";
  expect(phrase(sl(sjob("IN_QUEUE"), { health: { ...health, workers: null } })).text).toBe(queued);
  expect(phrase(sl(sjob("IN_QUEUE"), { health: { ...health, workers: {} } })).text).toBe(`${queued} (0 running, 0 idle)`);
});

test("serverless scoring names the scan, or the job id without one", () => {
  expect(phrase({ ...sl(sjob("IN_PROGRESS")), scan: "case-7.nii.gz" })).toEqual({ text: "RunPod serverless, scoring case-7.nii.gz, 1:35", tone: "busy" });
  expect(phrase(sl(sjob("RUNNING"))).text).toBe("RunPod serverless, scoring job_5555, 1:35");
});

test("serverless job whose status is not known yet", () => {
  expect(phrase(sl(sjob(null)))).toEqual({ text: "RunPod serverless, job submitted, status pending", tone: "busy" });
});

test("serverless job in another status", () => {
  expect(phrase(sl(sjob("COMPLETED")))).toEqual({ text: "RunPod serverless, job completed, 1:35", tone: "busy" });
});

test("serverless with nothing submitted scales to zero", () => {
  expect(phrase(sl(null))).toEqual({ text: "RunPod serverless, AMPERE_24 or ADA_24, scales to zero", tone: "idle" });
});

test("serverless not configured, or the block missing", () => {
  const text = "RunPod serverless, not configured (needs RUNPOD_ENDPOINT_ID)";
  expect(phrase({ mode: "serverless", serverless: { configured: false, job: null, health: null } })).toEqual({ text, tone: "idle" });
  expect(phrase({ mode: "serverless" })).toEqual({ text, tone: "idle" });
});

test("a serverless job in flight while the mode is runpod gets the generic suffix", () => {
  const s = { ...sl(sjob("IN_PROGRESS")), mode: "runpod", scan: "a.nii" };
  expect(phrase(s)).toEqual({ text: "RunPod pod, none running, starts when a job is queued, scoring a.nii", tone: "busy" });
});

test("a problem sentence still wins the tone in serverless", () => {
  const problem = "No worker has started in 5 min; EU-RO-1 stock or a slow image pull. Cancel the job to switch.";
  expect(phrase({ ...sl(sjob("IN_QUEUE")), problem })).toEqual({ text: `RunPod serverless, job queued 1:35, waiting for a worker. ${problem}`, tone: "problem" });
});

test("worker URL: managed tunnel of the pod, fixed URL, nothing on modal", () => {
  // base is mode runpod; mode worker shows the same rows (a pod may drain there after a switch).
  const managed = { ...base, tunnel_mode: "managed", pod: pod({ tunnel_url: "https://a-b-c.trycloudflare.com", tunnel_alive: true }) };
  expect(workerUrl(managed)).toEqual({ url: "https://a-b-c.trycloudflare.com", label: "Tunnel", note: "managed" });
  expect(workerUrl({ ...managed, pod: { ...managed.pod, tunnel_alive: false } }).note).toBe("down");
  expect(workerUrl({ ...base, tunnel_mode: "managed", pod: null })).toBeNull();
  expect(workerUrl({ ...base, tunnel_mode: "external", public_url: "https://radar.example.org" })).toEqual({ url: "https://radar.example.org", label: "Worker URL", note: "fixed" });
  expect(workerUrl({ ...managed, mode: "modal" })).toBeNull();
  const own = { ...base, mode: "worker", runpod: { configured: false } };
  expect(workerUrl({ ...own, tunnel_mode: "external", public_url: "https://radar.example.org" })).toEqual({ url: "https://radar.example.org", label: "Worker URL", note: "fixed" });
  expect(workerUrl({ ...own, tunnel_mode: "managed", pod: managed.pod }).label).toBe("Tunnel");
  expect(workerUrl({ ...own, tunnel_mode: "managed", pod: null })).toBeNull();
});

const open = { available: true, reason: null, note: null };
const localReason = "Modal cannot reach a local folder; this app stores scans under DATA_DIR";
const storage = (modal, serverless) => ({ backend: "local", name: "Local folder", modes: { modal, worker: open, runpod: open, serverless } });

test("mode control: an available mode has no note and no title", () => {
  const c = { ...base, storage: storage(open, open) };
  expect(modeControl(c, "modal")).toEqual({ label: "Modal", disabled: false, title: null });
  expect(modeControl(c, "serverless")).toEqual({ label: "RunPod serverless", disabled: false, title: null });
});

test("mode control: an unavailable mode is disabled, with its note and reason", () => {
  const c = { ...base, storage: storage({ available: false, reason: localReason, note: "not on local storage" }, open) };
  expect(modeControl(c, "modal")).toEqual({ label: "Modal (not on local storage)", disabled: true, title: localReason });
  expect(modeControl(c, "worker")).toEqual({ label: "Own GPU workers", disabled: false, title: null });
});

test("mode control: an unavailable runpod carries the backend's note and reason", () => {
  const keys = "RunPod keys missing; set RUNPOD_API_KEY";
  const c = { ...base, storage: { ...storage(open, open), modes: { ...storage(open, open).modes, runpod: { available: false, reason: keys, note: "no RunPod key" } } } };
  expect(modeControl(c, "runpod")).toEqual({ label: "RunPod pod (no RunPod key)", disabled: true, title: keys });
  const image = "A pod needs RUNPOD_VOLUME_ID, RUNPOD_REGISTRY_AUTH_ID and WORKER_IMAGE";
  c.storage.modes.runpod = { available: false, reason: image, note: "not configured" };
  expect(modeControl(c, "runpod")).toEqual({ label: "RunPod pod (not configured)", disabled: true, title: image });
});

test("mode control: not changeable disables every mode but adds no note", () => {
  const c = { ...base, mode: "fake", changeable: false, storage: storage(open, open) };
  expect(modeControl(c, "modal")).toEqual({ label: "Modal", disabled: true, title: null });
  expect(modeControl(c, "worker")).toEqual({ label: "Own GPU workers", disabled: true, title: null });
});

test("mode control: a missing storage block reads as every mode available", () => {
  expect(modeControl(base, "modal")).toEqual({ label: "Modal", disabled: false, title: null });
  expect(modeControl(base, "serverless")).toEqual({ label: "RunPod serverless", disabled: false, title: null });
});

test("mode control: the four names do not depend on the RunPod keys", () => {
  const names = { modal: "Modal", worker: "Own GPU workers", runpod: "RunPod pod", serverless: "RunPod serverless" };
  for (const runpod of [{ configured: true }, { configured: false }, undefined]) {
    for (const [value, name] of Object.entries(names)) expect(modeControl({ ...base, runpod }, value).label).toBe(name);
  }
  expect(modeControl(base, "other").label).toBe("other");
});
