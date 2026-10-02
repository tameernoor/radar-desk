// Unit tests for src/compute.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import { computePhrase, fromGpuStatus, workerUrl } from "../src/compute.js";

const NOW = Date.parse("2026-10-02T12:00:00Z");
const at = (secondsAgo) => new Date(NOW - secondsAgo * 1000).toISOString();
const base = { mode: "worker", changeable: true, runpod: { configured: true }, pod: null, in_flight: null, queued: 0, held: [], problem: null };
const pod = (extra) => ({ id: "pod_1", runpod_id: "rp123", gpu: "NVIDIA L4", cost_per_hr: 0.39, created_at: at(200), started_at: at(150), job_id: null, stops_in_s: null, ...extra });
const phrase = (s) => computePhrase({ ...base, ...s }, NOW);

test("modal names the requested GPUs", () => {
  expect(phrase({ mode: "modal", gpus: ["L4", "L40S"] })).toEqual({ text: "Modal, L4 or L40S, scales to zero", tone: "idle" });
});

test("worker without RunPod keys", () => {
  expect(phrase({ runpod: { configured: false } })).toEqual({ text: "Worker backend, no RunPod keys, workers are started by hand", tone: "idle" });
});

test("RunPod without a pod", () => {
  expect(phrase({})).toEqual({ text: "RunPod, no pod, starts when a job is queued", tone: "idle" });
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

test("pod ready and idle says when it stops", () => {
  expect(phrase({ pod: pod({ phase: "ready", stops_in_s: 361 }) })).toEqual({ text: "Pod rp123, NVIDIA L4, $0.39/h, idle, stops in 7 min", tone: "busy" });
  expect(phrase({ pod: pod({ phase: "ready" }) }).text).toBe("Pod rp123, NVIDIA L4, $0.39/h, idle");
});

test("a job running elsewhere is named", () => {
  const s = { mode: "worker", in_flight: { job_id: "job_99999999x", backend: "modal" }, scan: "a.nii" };
  expect(phrase(s)).toEqual({ text: "RunPod, no pod, starts when a job is queued, scoring a.nii", tone: "busy" });
});

test("a problem is its own sentence and wins the tone", () => {
  const held = [{ job_id: "j1", hold_reason: "no_gpu" }];
  expect(phrase({ held, problem: "No L4 or RTX 4090 in EU-RO-1, retrying in 60 s" })).toEqual({
    text: "RunPod, no pod, starts when a job is queued. No L4 or RTX 4090 in EU-RO-1, retrying in 60 s.",
    tone: "problem",
  });
  expect(phrase({ problem: "Budget reached." }).text).toBe("RunPod, no pod, starts when a job is queued. Budget reached.");
});

test("held jobs without a problem are counted", () => {
  const held = [{ job_id: "j1", hold_reason: "budget" }, { job_id: "j2", hold_reason: "budget" }];
  expect(phrase({ held })).toEqual({ text: "RunPod, no pod, starts when a job is queued. 2 job(s) held (budget).", tone: "held" });
});

test("fake backend", () => {
  expect(phrase({ mode: "fake", changeable: false })).toEqual({ text: "Fake backend", tone: "idle" });
});

test("the scans strip's gpu_status maps onto the same phrase", () => {
  const g = {
    backend: "worker",
    compute_mode: "worker",
    runpod_configured: true,
    gpu_requested: ["L4"],
    pod: pod({ phase: "ready", job_id: "job_1" }),
    in_flight: { id: "job_1", scan_id: "scan_1", backend: "worker" },
    queued: [{ id: "job_2", hold_reason: null }],
    held: [],
  };
  const s = fromGpuStatus(g, (id) => `name of ${id}`);
  expect(s).toMatchObject({ mode: "worker", in_flight: { job_id: "job_1", backend: "worker" }, queued: 1, held: [], scan: "name of scan_1" });
  expect(computePhrase(s, NOW)).toEqual({ text: "Pod rp123, NVIDIA L4, $0.39/h, up 2:30, scoring name of scan_1", tone: "busy" });
  const idle = fromGpuStatus({ backend: "worker", compute_mode: "worker", runpod_configured: false, gpu_requested: [], pod: null, in_flight: null, queued: [], held: [] });
  expect(computePhrase(idle, NOW).text).toBe("Worker backend, no RunPod keys, workers are started by hand");
});

test("mode modal with a pod still draining names the pod", () => {
  const g = { backend: "modal", compute_mode: "modal", runpod_configured: true, gpu_requested: ["L4"], pod: pod({ phase: "ready", stops_in_s: 0 }), in_flight: null, queued: [], held: [] };
  expect(computePhrase(fromGpuStatus(g), NOW)).toEqual({ text: "Pod rp123, NVIDIA L4, $0.39/h, idle, stops in 0 min", tone: "busy" });
});

test("worker URL: managed tunnel of the pod, fixed URL, nothing on modal", () => {
  const managed = { ...base, tunnel_mode: "managed", pod: pod({ tunnel_url: "https://a-b-c.trycloudflare.com", tunnel_alive: true }) };
  expect(workerUrl(managed)).toEqual({ url: "https://a-b-c.trycloudflare.com", label: "Tunnel", note: "managed" });
  expect(workerUrl({ ...managed, pod: { ...managed.pod, tunnel_alive: false } }).note).toBe("down");
  expect(workerUrl({ ...base, tunnel_mode: "managed", pod: null })).toBeNull();
  expect(workerUrl({ ...base, tunnel_mode: "external", public_url: "https://radar.example.org" })).toEqual({ url: "https://radar.example.org", label: "Worker URL", note: "fixed" });
  expect(workerUrl({ ...managed, mode: "modal" })).toBeNull();
});
