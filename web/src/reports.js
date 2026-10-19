// The reports window: the radiologist's report and NV-Reason-CT's report in full, side by side, next to the workspace.
// Every word from a report or an answer goes in as text (el() uses text nodes), never as HTML.
import "./styles.css";
import { ApiError, get } from "./api.js";
import { el } from "./format.js";
import { wireLogout } from "./login.js";
import { finePrint, HOLDS, isBusy, shown, splitReport } from "./readers.js";

const $ = (id) => document.getElementById(id);
const SUB = "The radiologist's report and NV-Reason-CT's report of the same scan, side by side.";
const FROM_WORKSPACE = "Run or retry it from the workspace.";

let scanId = null;
let view = null;
let lastJson = "";
let thinkingOpen = false; // survives re-renders
let pollTimer = null;
let seq = 0; // a response older than the one on screen is dropped

const note = (text) => el("p", { class: "muted small" }, text);
const para = (t, extra = "") => el("p", { class: `reports-text${extra}` }, t);

function radiologistColumn(v) {
  const body = v.has_report ? [para(v.report?.text || "")] : [el("p", { class: "muted" }, "No report attached. Add one from the workspace.")];
  return el("section", { class: "reports-col", "data-col": "radiologist" }, el("h2", {}, "Radiologist's report"), ...body);
}

function nvBody(nv) {
  const job = nv.job;
  if (!job) return [el("p", { class: "muted" }, nv.unavailable ? `Not available: ${nv.unavailable}.` : `Not run yet. ${FROM_WORKSPACE}`)];
  if (job.state === "failed" || job.state === "cancelled") return [el("p", { class: "rretry" }, `It did not finish${job.error?.message ? `: ${job.error.message}` : "."} ${FROM_WORKSPACE}`)];
  if (job.state === "queued" && job.hold_reason) return [note(`Waiting, ${HOLDS[job.hold_reason] || job.hold_reason}.`)];
  if (job.state === "queued" || job.state === "submitted") return [el("p", { class: "muted" }, "Reading the scan, about 2 minutes.")];
  if (!nv.report) return [note("Its report is not available.")];
  const { findings, impression } = splitReport(nv.report.answer);
  const body = [el("h3", {}, "Impression"), para(impression || "NV wrote no impression.")];
  if (findings) body.push(el("h3", {}, "Findings"), para(findings));
  if (nv.report.thinking) {
    body.push(el("details", { "data-part": "thinking", open: thinkingOpen, ontoggle: (e) => (thinkingOpen = e.target.open) }, el("summary", {}, "How it reasoned"), para(nv.report.thinking, " muted")));
  }
  const answers = nv.answers || [];
  if (answers.length) {
    body.push(el("h3", {}, "Answers to questions asked"));
    for (const a of answers) {
      body.push(
        el("p", { class: "reports-text reports-q" }, shown(a.question)),
        para(a.answer || "no answer"),
        a.differs ? el("p", { class: "rdiffers small" }, "NV's answer differs from its report") : null,
      );
    }
  }
  body.push(note(finePrint(nv)));
  return body;
}

function nvColumn(v) {
  return el("section", { class: "reports-col", "data-col": "nv" }, el("h2", {}, "NV-Reason-CT's report"), ...nvBody(v.nv || {}));
}

// Both columns are rebuilt, then each one's scrollTop is put back; the <details> state lives in thinkingOpen.
function render() {
  const host = $("reports-cols");
  const old = [...host.children].map((c) => c.scrollTop);
  host.replaceChildren(radiologistColumn(view), nvColumn(view));
  [...host.children].forEach((c, i) => (c.scrollTop = old[i] || 0));
}

function show(next) {
  const json = JSON.stringify(next);
  if (json === lastJson) return; // nothing changed, so nothing moves
  lastJson = json;
  view = next;
  render();
}

const failed = (what, e) => $("reports-cols").replaceChildren(el("p", { class: "rretry reports-col" }, `Could not load ${what}: ${e.message}`));

function poll() {
  clearTimeout(pollTimer);
  if (view && isBusy(view)) pollTimer = setTimeout(load, 5000);
}

async function load() {
  clearTimeout(pollTimer);
  const mine = ++seq;
  try {
    const next = await get(`/scans/${encodeURIComponent(scanId)}/readers`);
    if (mine === seq) show(next);
  } catch (e) {
    if (!view && !(e instanceof ApiError && e.status === 401)) failed("the reports", e);
  }
  if (mine === seq) poll(); // one failed poll does not stop the polling
}

async function main() {
  wireLogout();
  scanId = new URLSearchParams(location.search).get("scan");
  if (!scanId) {
    $("reports-title").textContent = "No scan given. Open one from the Scans page.";
    return;
  }
  try {
    await get("/auth/me");
    const scan = await get(`/scans/${encodeURIComponent(scanId)}`);
    $("reports-title").textContent = scan.filename;
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 401)) failed("the scan", e);
    return;
  }
  $("reports-sub").textContent = SUB;
  window.addEventListener("focus", load);
  if (typeof BroadcastChannel !== "undefined") {
    new BroadcastChannel("radar-desk-readers").onmessage = (e) => {
      if (e.data?.scan_id === scanId && e.data.view) {
        seq++; // any fetch still in flight is now older than this view
        show(e.data.view);
        poll();
      }
    };
  }
  await load();
}

main();
