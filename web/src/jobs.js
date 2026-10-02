import "./styles.css";
import { ApiError, api, del, get, post } from "./api.js";
import { computePhrase, workerUrl } from "./compute.js";
import { el, elapsed, shortId, usd, when } from "./format.js";
import { wireLogout } from "./login.js";

const ACTIVE = new Set(["queued", "submitted"]);
let scanNames = new Map();
let openLogs = new Set();
let jobs = [];

function timings(job) {
  const t = job.timings;
  if (t) return `infer ${t.infer_s?.toFixed(1)} s, total ${t.total_s?.toFixed(1)} s`;
  if (job.state === "submitted") return `running ${elapsed(job.submitted_at)}`;
  return "–";
}

function exportsFor(job) {
  if (job.state !== "done") return [];
  const base = `/jobs/${job.id}`;
  return ["scores.csv", "scores.json", "mask.nii.gz", "trace.json"].map((f) => el("a", { href: `${base}/${f}`, download: "" }, f));
}

async function act(promise) {
  const status = document.getElementById("jobs-status");
  status.textContent = "";
  try {
    await promise;
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 401)) status.textContent = e.message;
  }
  load();
}

function rerun(job) {
  // retry re-queues held or failed jobs; a done or cancelled one gets a fresh job.
  if (job.state === "failed" || job.hold_reason) return act(post(`/jobs/${job.id}/retry`));
  return act(post(`/scans/${job.scan_id}/jobs`));
}

function actions(job) {
  const cell = el("td", { class: "actions" });
  cell.append(el("a", { class: "button", href: `/workspace.html?scan=${encodeURIComponent(job.scan_id)}` }, "Open"));
  if (ACTIVE.has(job.state)) cell.append(el("button", { type: "button", onclick: () => act(post(`/jobs/${job.id}/cancel`)) }, "Cancel"));
  else cell.append(el("button", { type: "button", onclick: () => rerun(job) }, "Rerun"));
  cell.append(
    el(
      "button",
      {
        type: "button",
        "aria-expanded": String(openLogs.has(job.id)),
        onclick: () => {
          openLogs.has(job.id) ? openLogs.delete(job.id) : openLogs.add(job.id);
          render();
        },
      },
      "Log",
    ),
  );
  cell.append(...exportsFor(job));
  return cell;
}

function jobRow(job) {
  const error = job.error ? el("div", { class: "muted small" }, `${job.error.class}: ${job.error.message}`) : null;
  return el(
    "tr",
    { "data-job": job.id },
    el("td", { title: job.id, class: "mono" }, shortId(job.id)),
    el("td", { class: "file" }, scanNames.get(job.scan_id) || shortId(job.scan_id)),
    el("td", {}, el("span", { class: `chip chip-${job.state === "submitted" ? "scoring" : job.state}` }, job.state === "submitted" ? "scoring" : job.state), error),
    el("td", {}, job.hold_reason || "–"),
    el("td", {}, when(job.queued_at)),
    el("td", {}, when(job.finished_at)),
    el("td", {}, timings(job)),
    el(
      "td",
      {},
      job.gpu_used || (job.gpu_requested || []).join(" or ") || "–",
      job.lease ? el("div", { class: "muted small mono" }, job.lease.worker_id) : null,
    ),
    el("td", { class: "num" }, usd(job.cost_estimate_usd)),
    el("td", { class: "mono small" }, job.model_version || "–"),
    actions(job),
  );
}

function logRow(job) {
  const pre = el("pre", { class: "log" }, "Loading log…");
  get(`/jobs/${job.id}/logs`)
    .then((r) => (pre.textContent = r.text || "No log lines."))
    .catch((e) => (pre.textContent = `No log: ${e.message}`));
  return el("tr", { class: "log-row" }, el("td", { colspan: "11" }, pre));
}

function render() {
  const rows = [];
  for (const job of jobs) {
    rows.push(jobRow(job));
    if (openLogs.has(job.id)) rows.push(logRow(job));
  }
  document.querySelector("#jobs-table tbody").replaceChildren(...rows);
  document.getElementById("jobs-empty").hidden = jobs.length > 0;
}

async function load() {
  const [j, s] = await Promise.all([get("/jobs"), get("/scans")]);
  jobs = j.jobs;
  scanNames = new Map(s.scans.map((x) => [x.id, x.filename]));
  render();
}

// ---------- compute ----------

let compute = null; // the last GET /compute body
let gpus = []; // the Modal GPUs requested, for the phrase

const modeLabel = (mode) =>
  mode === "modal" ? "Modal" : mode === "serverless" ? "RunPod serverless" : compute?.runpod?.configured ? "RunPod pod" : "Worker";

async function computeAct(promise) {
  const status = document.getElementById("compute-status");
  status.textContent = "";
  try {
    const body = await promise;
    if (body) renderCompute(body);
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 401)) status.textContent = e.message;
  }
}

function renderCompute(c) {
  compute = c;
  const jobId = c.pod?.job_id || c.in_flight?.job_id;
  const job = jobId && jobs.find((j) => j.id === jobId);
  const { text, tone } = computePhrase({ ...c, gpus, scan: job && scanNames.get(job.scan_id) }, Date.now());
  const line = document.getElementById("compute-line");
  line.textContent = text;
  line.dataset.tone = tone;
  const w = workerUrl(c);
  const tunnel = document.getElementById("compute-tunnel");
  tunnel.hidden = !w;
  if (w) {
    tunnel.replaceChildren(`${w.label}: `, Object.assign(document.createElement("a"), { href: w.url, textContent: w.url, target: "_blank", rel: "noopener" }), ` (${w.note})`);
  }
  for (const input of document.querySelectorAll('input[name="compute-mode"]')) {
    input.checked = input.value === c.mode; // also snaps a radio back while a switch waits for its confirmation
    const unconfigured = input.value === "serverless" && !c.serverless?.configured;
    input.disabled = !c.changeable || unconfigured;
    const label = modeLabel(input.value) + (unconfigured ? " (needs RUNPOD_ENDPOINT_ID)" : "");
    if (!input.dataset.armed) input.parentElement.querySelector("[data-label]").textContent = label;
  }
  document.getElementById("compute-fixed").hidden = c.changeable;
  document.getElementById("pod-start").hidden = !(c.mode === "worker" && c.runpod.configured && !c.pod);
  document.getElementById("pod-stop").hidden = !c.pod;
}

function chooseMode(input) {
  // With a job in flight a switch needs a second click on the same radio; no modal dialogs.
  const job = compute?.in_flight;
  if (job && !input.dataset.armed) {
    input.dataset.armed = "1";
    const where = { modal: "Modal", worker: modeLabel("worker"), serverless: "RunPod serverless", fake: "the fake backend" }[job.backend] || job.backend;
    input.parentElement.querySelector("[data-label]").textContent =
      `Switch to ${modeLabel(input.value)}? Job ${shortId(job.job_id)} finishes on ${where}`;
    setTimeout(() => {
      delete input.dataset.armed;
      renderCompute(compute);
    }, 4000);
    renderCompute(compute);
    return;
  }
  delete input.dataset.armed;
  computeAct(api("/compute", { method: "PUT", body: { mode: input.value } }));
}

const loadCompute = () => computeAct(get("/compute"));

function wireCompute() {
  for (const input of document.querySelectorAll('input[name="compute-mode"]')) {
    input.addEventListener("change", () => chooseMode(input));
  }
  document.getElementById("pod-start").addEventListener("click", () => computeAct(post("/compute/pod/start")));
  document.getElementById("pod-stop").addEventListener("click", () => computeAct(post("/compute/pod/stop")));
}

// ---------- workers ----------

let newToken = null; // plaintext of the token created in this page load, shown once
let runTarget = { app_url: "", image: "" };

function copy(node) {
  const text = node.textContent;
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text).catch(() => select(node));
  select(node);
}

function select(node) {
  const range = document.createRange();
  range.selectNodeContents(node);
  getSelection().removeAllRanges();
  getSelection().addRange(range);
}

async function workerAct(promise) {
  const status = document.getElementById("workers-status");
  status.textContent = "";
  try {
    return await promise;
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 401)) status.textContent = e.message;
  }
}

function renderRunLine() {
  // The pod needs the URL this browser is using (a tunnel URL when the app is opened through one).
  const image = runTarget.image || "radar-worker";
  document.getElementById("run-line").textContent =
    `docker run --gpus all -v /workspace:/workspace -e RADAR_DESK_URL=${location.origin} -e RADAR_WORKER_TOKEN=${newToken || "<token>"} ${image}`;
}

async function revoke(token, button) {
  // A second click confirms; no modal dialogs.
  if (!button.dataset.armed) {
    button.dataset.armed = "1";
    button.textContent = "Revoke token?";
    setTimeout(() => {
      delete button.dataset.armed;
      button.textContent = "Revoke";
    }, 4000);
    return;
  }
  await workerAct(del(`/workers/tokens/${token.id}`));
  loadTokens();
}

function tokenRow(t) {
  const state = t.revoked_at ? "revoked" : "active";
  return el(
    "tr",
    { "data-token": t.id },
    el("td", {}, t.name),
    el("td", {}, when(t.created_at)),
    el("td", {}, when(t.last_used_at)),
    el("td", {}, el("span", { class: `chip chip-${t.revoked_at ? "cancelled" : "done"}` }, state)),
    el("td", { class: "actions" }, t.revoked_at ? null : el("button", { type: "button", class: "danger", onclick: (e) => revoke(t, e.currentTarget) }, "Revoke")),
  );
}

async function loadTokens() {
  const r = await workerAct(get("/workers/tokens"));
  if (r) document.querySelector("#tokens-table tbody").replaceChildren(...r.tokens.map(tokenRow));
}

function workerRow(w) {
  return el(
    "tr",
    { "data-worker": w.id },
    el("td", { class: "mono" }, w.id),
    el("td", {}, w.hostname || "–"),
    el("td", {}, w.gpu_name || w.device || "–"),
    el("td", {}, w.online ? el("span", { class: "chip chip-done" }, "online") : `last seen ${when(w.last_seen_at)}`),
    el("td", { class: "mono", title: w.job_id || null }, w.job_id ? shortId(w.job_id) : "–"),
  );
}

async function loadWorkers() {
  const r = await workerAct(get("/workers"));
  if (!r) return;
  runTarget = r;
  renderRunLine();
  document.querySelector("#workers-table tbody").replaceChildren(...r.workers.map(workerRow));
  document.getElementById("workers-empty").hidden = r.workers.length > 0;
}

function wireWorkers() {
  document.getElementById("token-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("token-name");
    const made = await workerAct(post("/workers/tokens", { name: input.value }));
    if (!made) return;
    newToken = made.token;
    input.value = "";
    document.getElementById("token-plain").textContent = made.token;
    document.getElementById("token-new").hidden = false;
    renderRunLine();
    loadTokens();
  });
  document.getElementById("token-copy").addEventListener("click", () => copy(document.getElementById("token-plain")));
  document.getElementById("run-copy").addEventListener("click", () => copy(document.getElementById("run-line")));
}

async function main() {
  wireLogout();
  try {
    await get("/auth/me");
  } catch {
    return;
  }
  wireCompute();
  wireWorkers();
  const loadGpus = () => get("/gpu/status").then((g) => (gpus = g.gpu_requested), () => {});
  await Promise.all([load(), loadTokens(), loadWorkers(), loadGpus()]);
  await loadCompute();
  setInterval(loadCompute, 5000);
  setInterval(loadWorkers, 5000);
  setInterval(() => {
    // Refresh while anything is moving, but keep open logs stable.
    if (jobs.some((x) => ACTIVE.has(x.state)) && openLogs.size === 0) load();
  }, 5000);
}

main();
