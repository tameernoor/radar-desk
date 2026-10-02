import "./styles.css";
import { ApiError, get, post } from "./api.js";
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
    el("td", {}, job.gpu_used || (job.gpu_requested || []).join(" or ") || "–"),
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

async function main() {
  wireLogout();
  try {
    await get("/auth/me");
  } catch {
    return;
  }
  await load();
  setInterval(() => {
    // Refresh while anything is moving, but keep open logs stable.
    if (jobs.some((x) => ACTIVE.has(x.state)) && openLogs.size === 0) load();
  }, 5000);
}

main();
