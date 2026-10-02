// The compute phrase shared by the jobs page (GET /compute) and the scans strip (GET /gpu/status).
// Pure: no DOM, the clock comes in as nowMs.
import { elapsed, shortId } from "./format.js";

const rate = (x) => (x == null ? "$–/h" : `$${x.toFixed(2)}/h`);

// s is the GET /compute body plus `gpus` (the requested Modal GPUs) and `scan` (the in-flight scan's name).
export function computePhrase(s, nowMs) {
  const now = new Date(nowMs).toISOString();
  const pod = s.pod;
  const held = s.held || [];
  let text;
  let podScoring = false;
  if (pod?.phase === "tunnel") text = `Starting pod, waiting for the tunnel, ${elapsed(pod.created_at, now)}`;
  else if (pod?.phase === "starting") {
    text = `Starting pod ${pod.runpod_id}, ${pod.gpu} at ${rate(pod.cost_per_hr)}, about 4 min (image pull, then weights), up ${elapsed(pod.started_at, now)}`;
  } else if (pod && pod.job_id) {
    podScoring = true;
    text = `Pod ${pod.runpod_id}, ${pod.gpu}, ${rate(pod.cost_per_hr)}, up ${elapsed(pod.started_at, now)}, scoring ${s.scan || shortId(pod.job_id)}`;
  } else if (pod) {
    const stops = pod.stops_in_s == null ? "" : `, stops in ${Math.max(0, Math.ceil(pod.stops_in_s / 60))} min`;
    text = `Pod ${pod.runpod_id}, ${pod.gpu}, ${rate(pod.cost_per_hr)}, idle${stops}`;
  } else if (s.mode === "modal") text = `Modal, ${(s.gpus || []).join(" or ") || "GPU"}, scales to zero`;
  else if (s.mode === "worker" && !s.runpod?.configured) text = "Worker backend, no RunPod keys, workers are started by hand";
  else if (s.mode === "worker") text = "RunPod, no pod, starts when a job is queued";
  else text = "Fake backend";
  // A job still running elsewhere (Modal, or a worker started by hand) is named too.
  if (s.in_flight && !podScoring) text += `, scoring ${s.scan || shortId(s.in_flight.job_id)}`;
  if (s.problem) text += `. ${s.problem}${/[.!?]$/.test(s.problem) ? "" : "."}`;
  else if (held.length) text += `. ${held.length} job(s) held (${[...new Set(held.map((j) => j.hold_reason))].join(", ")}).`;
  const tone = s.problem ? "problem" : held.length ? "held" : pod || s.in_flight ? "busy" : "idle";
  return { text, tone };
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
  };
}
