import "./styles.css";
import { ApiError, del, get, post } from "./api.js";
import { computePhrase, fromGpuStatus } from "./compute.js";
import { wireLogout } from "./login.js";
import { bytes, dims, el, elapsed, scanChip, shortId, spacing, usd, when } from "./format.js";

let scans = [];

// ---------- scans table ----------

function actionsFor(scan) {
  const cell = el("td", { class: "actions" });
  if (scan.state === "ready") {
    cell.append(el("a", { class: "button", href: `/workspace.html?scan=${encodeURIComponent(scan.id)}` }, "Open"));
    const job = scan.latest_job;
    if (!job || ["failed", "cancelled"].includes(job.state)) {
      cell.append(el("button", { type: "button", onclick: () => score(scan) }, "Score"));
    }
  }
  cell.append(el("button", { type: "button", class: "danger", onclick: (e) => remove(scan, e.currentTarget) }, "Delete"));
  return cell;
}

function stateCell(scan) {
  const chip = scanChip(scan);
  const td = el("td", {}, el("span", { class: `chip chip-${chip}` }, chip));
  const job = scan.latest_job;
  if (scan.state === "rejected" && scan.rejected_reason) td.append(el("div", { class: "muted small" }, scan.rejected_reason));
  if (job?.hold_reason) td.append(el("div", { class: "muted small" }, `held: ${job.hold_reason}`));
  return td;
}

function scanRow(scan) {
  const job = scan.latest_job;
  const positives = job?.state === "done" && job.positives_at_50 != null ? String(job.positives_at_50) : "–";
  return el(
    "tr",
    {},
    el("td", { class: "file", title: scan.sha256 || "" }, scan.filename),
    stateCell(scan),
    el("td", {}, dims(scan.header)),
    el("td", {}, spacing(scan.header)),
    el("td", {}, bytes(scan.size_bytes)),
    el("td", {}, scan.fixture_id ? el("span", { class: "tag" }, scan.fixture_id) : "–"),
    el("td", { class: "num" }, positives),
    el("td", {}, when(scan.created_at)),
    actionsFor(scan),
  );
}

async function loadScans() {
  const data = await get("/scans");
  scans = data.scans;
  const body = document.querySelector("#scans-table tbody");
  body.replaceChildren(...scans.map(scanRow));
  document.getElementById("scans-empty").hidden = scans.length > 0;
}

async function score(scan) {
  try {
    await post(`/scans/${scan.id}/jobs`);
  } catch (e) {
    alertError(e);
  }
  loadScans();
}

async function remove(scan, button) {
  // A second click confirms; no modal dialogs.
  if (!button.dataset.armed) {
    button.dataset.armed = "1";
    button.textContent = "Delete scan?";
    setTimeout(() => {
      delete button.dataset.armed;
      button.textContent = "Delete";
    }, 4000);
    return;
  }
  try {
    await del(`/scans/${scan.id}`);
  } catch (e) {
    alertError(e);
  }
  loadScans();
}

function alertError(e) {
  if (e instanceof ApiError && e.status === 401) return;
  setUploadText(e.message, true);
}

// ---------- GPU strip ----------

function scanName(scanId) {
  return scans.find((s) => s.id === scanId)?.filename || shortId(scanId);
}

function gpuPhrase(g) {
  const queued = scans.filter((s) => s.latest_job?.state === "queued").length;
  const price = g.price_per_hour_usd != null ? ` at $${g.price_per_hour_usd.toFixed(2)}/h` : "";
  const backend = { fake: "Fake GPU", modal: "Modal", serverless: "RunPod serverless" }[g.backend] || g.backend;
  const gpu = g.in_flight?.gpu_used || g.gpu_requested.join(" or ");
  const budgetHeld = g.held.find((j) => j.hold_reason === "budget");
  if (budgetHeld) return `Held: monthly GPU budget reached, raise or wait. ${g.held.length} job(s) held.`;
  if (g.in_flight) {
    const ahead = queued ? `, ${queued} more in queue` : "";
    return `${backend}: scoring ${scanName(g.in_flight.scan_id)}, ${elapsed(g.in_flight.submitted_at)}, ${gpu}${price}${ahead}`;
  }
  if (g.held.length) {
    const reasons = [...new Set(g.held.map((j) => j.hold_reason))].join(", ");
    return `Held: ${g.held.length} job(s) waiting (${reasons}), the poller retries.`;
  }
  if (queued) return `${backend}: ${queued} job(s) queued, nothing on the GPU yet`;
  return `GPU idle, nothing queued (${backend === "Modal" ? `Modal, ${gpu}` : "fake backend"})`;
}

async function pollGpu() {
  const strip = document.getElementById("gpu-strip");
  try {
    const g = await get("/gpu/status");
    const spend = `Spent ${usd(g.spend_month_usd)} of ${usd(g.budget_usd)} in ${g.month}.`;
    // Worker or serverless mode, or a pod still draining after a switch, uses the same phrase as the jobs page.
    const c = g.pod || ["worker", "runpod", "serverless"].includes(g.compute_mode) ? computePhrase(fromGpuStatus(g, scanName), Date.now()) : null;
    strip.replaceChildren(el("strong", {}, c ? c.text : gpuPhrase(g)), " ", el("span", { class: "muted" }, spend));
    strip.classList.toggle("busy", c ? c.tone === "busy" : Boolean(g.in_flight));
    strip.classList.toggle("held", c ? c.tone === "held" || c.tone === "problem" : g.held.length > 0);
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 401)) strip.textContent = `GPU status unavailable: ${e.message}`;
  }
}

// ---------- upload ----------

function setUploadText(text, isError = false) {
  const box = document.getElementById("upload-status");
  box.hidden = false;
  const span = document.getElementById("upload-text");
  span.textContent = text;
  span.classList.toggle("error", isError);
}

function putFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("Content-Type", file.name.endsWith(".gz") ? "application/gzip" : "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(`Upload failed: HTTP ${xhr.status}`)));
    xhr.onerror = () => reject(new Error("Upload failed: network error"));
    xhr.send(file);
  });
}

async function upload(file) {
  const ok = document.getElementById("research-ok");
  if (!ok.checked) {
    setUploadText("Tick the research-only confirmation first.", true);
    ok.focus();
    return;
  }
  if (!/\.nii(\.gz)?$/i.test(file.name)) {
    setUploadText("Only .nii or .nii.gz files.", true);
    return;
  }
  const bar = document.getElementById("upload-progress");
  try {
    setUploadText(`Requesting upload for ${file.name}…`);
    const slot = await post("/uploads", { filename: file.name, size_bytes: file.size, research_only_confirmed: true });
    loadScans();
    await putFile(slot.put_url, file, (f) => {
      bar.value = Math.round(f * 100);
      setUploadText(`Uploading ${file.name}: ${Math.round(f * 100)}% of ${bytes(file.size)}`);
    });
    setUploadText("Checking the NIfTI header…");
    // A rejected file answers 422 with the reason; the catch below shows it.
    const scan = await post(`/uploads/${slot.scan_id}/complete`);
    setUploadText("Accepted. Queuing scoring and opening the workspace…");
    await post(`/scans/${scan.id}/jobs`);
    location.href = `/workspace.html?scan=${encodeURIComponent(scan.id)}`;
  } catch (e) {
    if (e instanceof ApiError && e.status === 422) setUploadText(`Rejected: ${e.message}`, true);
    else alertError(e);
    loadScans();
  }
}

function wireDropzone() {
  const target = document.getElementById("drop-target");
  const input = document.getElementById("file-input");
  input.addEventListener("change", () => input.files[0] && upload(input.files[0]));
  target.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  target.addEventListener("dragover", (e) => {
    e.preventDefault();
    target.classList.add("over");
  });
  target.addEventListener("dragleave", () => target.classList.remove("over"));
  target.addEventListener("drop", (e) => {
    e.preventDefault();
    target.classList.remove("over");
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  });
}

async function main() {
  wireLogout();
  wireDropzone();
  try {
    await get("/auth/me");
  } catch {
    return;
  }
  await loadScans();
  pollGpu();
  setInterval(pollGpu, 5000);
  setInterval(() => {
    if (scans.some((s) => ["uploading", "queued", "scoring"].includes(scanChip(s)))) loadScans();
  }, 5000);
}

main();
