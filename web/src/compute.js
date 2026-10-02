// The compute phrase shared by the jobs page (GET /compute) and the scans strip (GET /gpu/status).
// Pure: no DOM, the clock comes in as nowMs.
import { elapsed, shortId } from "./format.js";

const rate = (x) => (x == null ? "$–/h" : `$${x.toFixed(2)}/h`);

// s is the GET /compute body plus `gpus` (the requested Modal GPUs) and `scan` (the in-flight scan's name).
// Its `serverless` block may be missing (an older backend, the scans strip); that reads as not configured.
export function computePhrase(s, nowMs) {
  const now = new Date(nowMs).toISOString();
  const pod = s.pod;
  const held = s.held || [];
  let text;
  let podScoring = false;
  let ownPhrase = false; // a serverless job phrase already names the job
  const sjob = s.serverless?.job;
  if (pod?.phase === "tunnel") text = `Starting pod, waiting for the tunnel, ${elapsed(pod.created_at, now)}`;
  else if (pod?.phase === "starting") {
    text = `Starting pod ${pod.runpod_id}, ${pod.gpu} at ${rate(pod.cost_per_hr)}, about 4 min (image pull, then weights), up ${elapsed(pod.started_at, now)}`;
  } else if (pod && pod.job_id) {
    podScoring = true;
    text = `Pod ${pod.runpod_id}, ${pod.gpu}, ${rate(pod.cost_per_hr)}, up ${elapsed(pod.started_at, now)}, scoring ${s.scan || shortId(pod.job_id)}`;
  } else if (pod) {
    const idle = pod.idle_s == null ? "" : ` ${Math.floor(pod.idle_s / 60)} min`;
    const deletes = pod.idle_delete_s == null ? "" : `, deletes itself after ${Math.round(pod.idle_delete_s / 60)} idle min`;
    text = `Pod ${pod.runpod_id}, ${pod.gpu}, ${rate(pod.cost_per_hr)}, idle${idle}${deletes}`;
  } else if (s.mode === "serverless" && sjob) {
    ownPhrase = true;
    const since = elapsed(sjob.submitted_at, now);
    const w = s.serverless.health?.workers;
    if (sjob.status === "IN_QUEUE") {
      const counts = w && typeof w === "object" ? ` (${w.running ?? 0} running, ${w.idle ?? 0} idle)` : "";
      text = `RunPod serverless, job queued ${since}, waiting for a worker${counts}`;
    } else if (sjob.status === "IN_PROGRESS" || sjob.status === "RUNNING") {
      text = `RunPod serverless, scoring ${s.scan || shortId(sjob.job_id)}, ${since}`;
    } else if (sjob.status == null) text = "RunPod serverless, job submitted, status pending";
    else text = `RunPod serverless, job ${sjob.status.toLowerCase()}, ${since}`;
  } else if (s.mode === "serverless" && s.serverless?.configured) {
    text = `RunPod serverless, ${(s.serverless.gpus || []).join(" or ")}, scales to zero`;
  } else if (s.mode === "serverless") text = "RunPod serverless, not configured (needs RUNPOD_ENDPOINT_ID)";
  else if (s.mode === "modal") text = `Modal, ${(s.gpus || []).join(" or ") || "GPU"}, scales to zero`;
  else if (s.mode === "worker" && !s.runpod?.configured) text = "Worker backend, no RunPod keys, workers are started by hand";
  else if (s.mode === "worker") text = "RunPod, no pod, starts when a job is queued";
  else text = "Fake backend";
  // A job still running elsewhere (Modal, or a worker started by hand) is named too.
  if (s.in_flight && !podScoring && !ownPhrase) text += `, scoring ${s.scan || shortId(s.in_flight.job_id)}`;
  if (s.problem) text += `. ${s.problem}${/[.!?]$/.test(s.problem) ? "" : "."}`;
  else if (held.length) text += `. ${held.length} job(s) held (${[...new Set(held.map((j) => j.hold_reason))].join(", ")}).`;
  const tone = s.problem ? "problem" : held.length ? "held" : pod || s.in_flight || ownPhrase ? "busy" : "idle";
  return { text, tone };
}

// The address workers use to reach this app: the managed quick tunnel of the open pod, or the fixed
// WORKER_PUBLIC_URL. Null when there is none to show.
export function workerUrl(s) {
  if (s.mode !== "worker") return null;
  if (s.tunnel_mode === "external" && s.public_url) return { url: s.public_url, label: "Worker URL", note: "fixed" };
  if (s.pod?.tunnel_url) return { url: s.pod.tunnel_url, label: "Tunnel", note: s.pod.tunnel_alive === false ? "down" : "managed" };
  return null;
}

// The scans strip's GET /gpu/status body in the shape computePhrase reads.
export function fromGpuStatus(g, scanName = shortId) {
  return {
    mode: g.compute_mode ?? g.backend,
    runpod: { configured: g.runpod_configured },
    pod: g.pod ?? null,
    in_flight: g.in_flight ? { job_id: g.in_flight.id, backend: g.in_flight.backend } : null,
    queued: (g.queued || []).length,
    held: (g.held || []).map((j) => ({ job_id: j.id, hold_reason: j.hold_reason })),
    problem: g.problem ?? null,
    gpus: g.gpu_requested,
    scan: g.in_flight ? scanName(g.in_flight.scan_id) : null,
    serverless: g.serverless ?? null,
  };
}
