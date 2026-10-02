import "./styles.css";
import { ApiError, get, post } from "./api.js";
import { mountChat } from "./chat.js";
import { createFindings } from "./findings.js";
import { buildLookCard } from "./lookcard.js";
import { bytes, dims, el, elapsed, shortId, spacing, usd, when } from "./format.js";
import { wireLogout } from "./login.js";
import { registerPageTools } from "./pagetools.js";
import { organCss, UNSCORED_GREY } from "./palette.js";
import { createViewer, rawBytes, tooBig, WINDOWS } from "./viewer.js";

const TERMINAL = new Set(["done", "failed", "cancelled"]);
const WINDOW_KEYS = { 1: "soft_tissue", 2: "liver", 3: "bone", 4: "lung" };

const $ = (id) => document.getElementById(id);

const ws = {
  scan: null,
  job: null, // the latest job for the scan
  resultJob: null, // the done job whose result is on screen (can be older than job)
  result: null,
  labels: [],
  scoredOrgans: [],
  activeOrgan: null,
  activeFinding: null,
  boxOn: false,
  viewer: null,
  findings: null,
  catalog: [],
  chat: null,
  ctLoaded: false,
};

// ---------- notices ----------

function notice(message) {
  const box = $("viewer-notice");
  box.replaceChildren(el("p", {}, message));
  box.hidden = false;
}

const clearNotice = () => ($("viewer-notice").hidden = true);

// A question with a button. Kept apart from notice() so other messages cannot clear it.
function ask(message, label, run) {
  const box = $("viewer-prompt");
  const button = el("button", { type: "button", onclick: () => ((box.hidden = true), run()) }, label);
  box.replaceChildren(el("p", {}, message), button);
  box.hidden = false;
  button.focus();
}

function setJobStatus(text, kind = "") {
  const node = $("job-status");
  node.textContent = text;
  node.className = `job-status ${kind}`;
}

// ---------- actions shared by clicks, keys and page tools ----------

const labelOf = (organ) => ws.scoredOrgans.find((o) => o.organ === organ)?.label ?? null;

function jumpToOrgan(organ, { scroll = true } = {}) {
  if (!labelOf(organ)) throw new Error(`Unknown organ: ${organ}. Use one of the 18 scored organs.`);
  ws.activeOrgan = organ;
  if (scroll) ws.findings.scrollToOrgan(organ);
  if (!ws.result) return { ok: false, organ, reason: "No result yet for this scan." };
  const stats = ws.result.organ_stats?.[organ];
  if (!stats) {
    ws.viewer.isolateOrgan(null);
    ws.viewer.showBox(null); // the box of the previous organ would mislead here
    notice(`RADAR's segmentation found no ${organ.toLowerCase()} in this scan, so there is nothing to jump to.`);
    return { ok: false, organ, reason: "Organ not found by RADAR's segmentation; its findings were not scored." };
  }
  clearNotice();
  ws.viewer.jumpToMm(stats.centroid_mm);
  ws.viewer.isolateOrgan(labelOf(organ));
  if (ws.boxOn) showScoringBox(organ, true);
  const how = ws.result.organs_scored.find((o) => o.organ === organ);
  return { ok: true, organ, centroid_mm: stats.centroid_mm, ml: stats.ml, how: how?.how ?? null, window_index: how?.window_index ?? null };
}

function selectFinding(keyOrName) {
  const f = ws.findings.finding(keyOrName);
  if (!f) throw new Error(`Unknown finding: ${keyOrName}`);
  ws.activeFinding = f.key;
  const jumped = jumpToOrgan(f.organ, { scroll: false });
  ws.findings.setActive(f.key);
  return { ...jumped, key: f.key, finding: f.finding, prob: ws.findings.probOf(f.key) };
}

function setBoxOn(on) {
  ws.boxOn = on;
  $("box-toggle").setAttribute("aria-pressed", String(on));
}

function showScoringBox(organ, on) {
  if (!labelOf(organ)) throw new Error(`Unknown organ: ${organ}`);
  if (!on) {
    ws.viewer.showBox(null);
    setBoxOn(false);
    return { ok: true, organ, on: false };
  }
  const entry = ws.result?.organs_scored.find((o) => o.organ === organ);
  if (!entry) return { ok: false, organ, reason: "This organ was not scored, so there is no scoring box." };
  if (!ws.viewer.showBox(entry.box_mm, organ)) return { ok: false, organ, reason: "The box needs the mask, which is not loaded." };
  setBoxOn(true);
  return { ok: true, organ, on: true, how: entry.how, window_index: entry.window_index, box_mm: entry.box_mm };
}

function toggleMask(on) {
  const next = on ?? !ws.viewer.state().mask_on;
  ws.viewer.setMaskVisible(next);
  $("mask-toggle").setAttribute("aria-pressed", String(next));
  return { ok: true, mask_on: next, mask_loaded: ws.viewer.state().mask_loaded };
}

// Called by the viewer for presets and for right-drag ("custom").
function markWindow(preset) {
  for (const b of document.querySelectorAll("[data-window]")) b.setAttribute("aria-pressed", String(b.dataset.window === preset));
}

function setWindow(preset) {
  ws.viewer.setWindow(preset);
  return { ok: true, window_preset: preset, hu_range: [WINDOWS[preset].min, WINDOWS[preset].max] };
}

function setSlice(name) {
  ws.viewer.setSliceType(name);
  closeStaleCard();
  for (const b of document.querySelectorAll("[data-slice]")) b.setAttribute("aria-pressed", String(b.dataset.slice === name));
}

function setThreshold(percent) {
  ws.findings.setThreshold(percent);
  return { ok: true, threshold: ws.findings.state.threshold / 100, note: "Display only; RADAR ships no calibrated thresholds." };
}

function viewState() {
  const t = ws.findings.state.threshold;
  return {
    scan_id: ws.scan?.id ?? null,
    job_id: ws.resultJob?.id ?? ws.job?.id ?? null,
    job_state: (ws.resultJob ?? ws.job)?.state ?? null,
    latest_job_id: ws.job?.id ?? null,
    latest_job_state: ws.job?.state ?? null,
    active_organ: ws.activeOrgan,
    active_finding: ws.activeFinding,
    threshold: t / 100,
    threshold_pct: t,
    ...ws.viewer.state(),
    ...ws.viewer.sliceState(),
  };
}

// ---------- rendering ----------

function renderScanBar(scan) {
  $("scan-title").textContent = scan.filename;
  const parts = [dims(scan.header), `${spacing(scan.header)} mm`, bytes(scan.size_bytes)];
  if (scan.fixture_id) parts.push(`fixture ${scan.fixture_id}`);
  $("scan-facts").textContent = parts.join(" · ");
  document.title = `${scan.filename} · radar-desk`;
}

function renderReadout(loc) {
  const mm = loc.crosshair_mm ? loc.crosshair_mm.map((v) => v.toFixed(1)).join(", ") : "–";
  const hu = loc.hu == null ? "–" : loc.hu;
  $("readout").textContent = `${mm} mm · ${hu} HU · ${loc.label_name ?? "no mask"}`;
}

function renderLegend(present) {
  const scored = ws.scoredOrgans.filter((o) => present.includes(o.label));
  const unscored = ws.labels.filter((l) => l.label > 0 && present.includes(l.label) && !ws.scoredOrgans.some((o) => o.label === l.label));
  const items = scored.map((o) =>
    el("button", { type: "button", class: "legend-item", onclick: () => jumpToOrgan(o.organ) }, el("i", { style: `background:${organCss(o.organ)}` }), o.organ),
  );
  if (unscored.length) {
    items.push(
      el(
        "span",
        { class: "legend-item dim", title: unscored.map((l) => l.en).join(", ") },
        el("i", { style: `background:rgb(${UNSCORED_GREY.join(" ")})` }),
        `${unscored.length} segmented, not scored`,
      ),
    );
  }
  $("legend").replaceChildren(...items);
}

function renderVersions(job, result) {
  const v = result.versions || {};
  const t = job.timings;
  const secs = (x) => (typeof x === "number" ? `${x.toFixed(1)} s` : "–");
  const parts = [
    `Model ${job.model_version || "RADAR"}`,
    `checkpoint ${(v.checkpoint_sha256 || "–").slice(0, 12)}`,
    `vendored ${(v.code_commit || "–").slice(0, 7)}`,
    `image ${v.image_id || "–"}`,
    `GPU ${v.gpu || job.gpu_used || "–"}`,
  ];
  if (t) parts.push(`load ${secs(t.load_s)}, infer ${secs(t.infer_s)}, total ${secs(t.total_s)}`);
  if (job.cost_estimate_usd != null) parts.push(`est. ${usd(job.cost_estimate_usd)}`);
  if (v.torch) parts.push(`torch ${v.torch}${v.cuda ? `, CUDA ${v.cuda}` : ""}`);
  $("versions").textContent = parts.join(" · ");
  $("versions").title = v.checkpoint_sha256 ? `checkpoint sha256 ${v.checkpoint_sha256}` : "";
  $("fake-banner").hidden = v.gpu !== "fake";
}

function renderJob(job) {
  if (!job) {
    setJobStatus("Not scored yet");
    const button = el("button", { type: "button", onclick: startJob }, "Score this scan");
    $("job-status").append(" ", button);
    return;
  }
  const hold = job.hold_reason ? `, held: ${job.hold_reason}` : "";
  const text = {
    queued: `Queued${hold}`,
    submitted: `Scoring on ${job.gpu_used || job.gpu_requested?.join(" or ") || "GPU"}, ${elapsed(job.submitted_at)}${hold}`,
    done: `Scored in ${job.timings?.total_s?.toFixed(1) ?? "–"} s`,
    failed: `Failed: ${job.error?.class || "error"}${job.error?.message ? `, ${job.error.message}` : ""}`,
    cancelled: "Cancelled",
  }[job.state];
  setJobStatus(text || job.state, `state-${job.state}`);
  renderShownJob();
  if (job.state === "failed" || job.state === "cancelled") {
    $("job-status").append(" ", el("button", { type: "button", onclick: retryJob }, "Retry"));
  }
}

// Says which job is on screen when it is not the latest one.
function renderShownJob() {
  const node = $("shown-job");
  const shown = ws.resultJob;
  node.hidden = !shown || !ws.job || shown.id === ws.job.id;
  if (!node.hidden) node.textContent = `On screen: scores from job ${shortId(shown.id)}, finished ${when(shown.finished_at)}. The latest job is ${ws.job.state === "submitted" ? "scoring" : ws.job.state}.`;
}

// ---------- job lifecycle ----------

async function startJob() {
  try {
    trackJob(await post(`/scans/${ws.scan.id}/jobs`));
  } catch (e) {
    setJobStatus(`Could not queue: ${e.message}`, "state-failed");
  }
}

// The retry route re-queues failed or held jobs; a cancelled one needs a new job.
async function retryJob() {
  try {
    const path = ws.job.state === "cancelled" ? `/scans/${ws.scan.id}/jobs` : `/jobs/${ws.job.id}/retry`;
    trackJob(await post(path));
  } catch (e) {
    setJobStatus(`Could not retry: ${e.message}`, "state-failed");
  }
}

let ticker = null;
let source = null;

function onJob(job) {
  ws.job = job;
  renderJob(job);
  if (TERMINAL.has(job.state)) {
    source?.close();
    source = null;
    clearInterval(ticker);
    ticker = null;
    if (job.state === "done") loadResult(job);
  }
}

function pollJob(id) {
  ticker = setInterval(async () => {
    try {
      onJob(await get(`/jobs/${id}`));
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) clearInterval(ticker);
    }
  }, 3000);
}

function trackJob(job) {
  source?.close();
  clearInterval(ticker);
  onJob(job);
  if (TERMINAL.has(job.state)) return;
  if (!("EventSource" in window)) return pollJob(job.id);
  source = new EventSource(`/jobs/${job.id}/events`);
  source.addEventListener("state", (e) => onJob(JSON.parse(e.data)));
  source.onerror = () => {
    source?.close();
    source = null;
    if (!TERMINAL.has(ws.job.state)) pollJob(job.id);
  };
}

// Keep the elapsed time moving between job events.
setInterval(() => ws.job && !TERMINAL.has(ws.job.state) && renderJob(ws.job), 1000);

async function loadResult(job) {
  let result;
  try {
    result = await get(`/jobs/${job.id}/result`);
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 401)) setJobStatus(`Could not load the result: ${e.message}`, "state-failed");
    return;
  }
  ws.result = result;
  ws.resultJob = job;
  ws.findings.setCompare(null);
  ws.findings.setResult(result);
  renderVersions(job, result);
  renderShownJob();
  if (ws.scan.fixture_id) {
    get(`/jobs/${job.id}/compare?against=fixture`)
      .then((c) => ws.findings.setCompare(c))
      .catch(() => {});
  }
  await ws.viewer.ready;
  if (ws.ctLoaded) await loadMask(job);
}

let maskJobId = null;
async function loadMask(job) {
  if (maskJobId === job.id) return;
  maskJobId = job.id;
  try {
    const present = await ws.viewer.loadMask(`/jobs/${job.id}/mask.nii.gz`, ws.labels, ws.scoredOrgans);
    renderLegend(present);
    if (ws.boxOn && ws.activeOrgan) showScoringBox(ws.activeOrgan, true);
  } catch {
    notice("No mask for this job. Showing the CT only.");
  }
}

// ---------- CT ----------

async function loadCT() {
  try {
    await ws.viewer.loadCT(`/scans/${ws.scan.id}/source.nii.gz`);
    ws.ctLoaded = true;
    clearNotice();
    renderReadout(ws.viewer.location());
    if (ws.resultJob) await loadMask(ws.resultJob);
  } catch (e) {
    notice(`Could not load the CT: ${e.message}`);
  }
}

function offerCT() {
  if (tooBig(ws.scan)) {
    const mb = Math.round(rawBytes(ws.scan) / 1024 / 1024);
    ask(`This volume is ${mb} MB uncompressed, above 512 MB. Loading it may stall the browser.`, "Load anyway", () => {
      notice("Loading CT…");
      loadCT();
    });
    return;
  }
  notice("Loading CT…");
  loadCT();
}

// ---------- what am I looking at ----------

const LOOK_QUESTION = "What am I looking at?";

// Built from fresh view state each time; no model call, so it works without the chat.
function showLookCard() {
  const card = buildLookCard(viewState(), ws.result?.findings || [], ws.catalog);
  const box = $("look-card");
  const close = el("button", { type: "button", onclick: hideLookCard, "aria-label": "Close" }, "Close");
  const parts = [el("header", {}, el("h2", {}, LOOK_QUESTION), close)];
  if (card.empty) parts.push(el("p", {}, card.message));
  if (card.plane) parts.push(el("p", {}, el("strong", {}, card.plane)));
  if (card.organs.length) {
    parts.push(el("p", { class: "muted small" }, "RADAR outlined these on this slice, largest first."));
    parts.push(
      el(
        "ul",
        {},
        card.organs.map((o) =>
          el(
            "li",
            { class: "look-organ" },
            el("span", { class: "dot", style: `background:${o.scored ? organCss(o.organ) : `rgb(${UNSCORED_GREY.join(" ")})`}` }),
            " ",
            el("strong", {}, o.organ),
            el("span", { class: "muted" }, `${o.percent}% of the outlined area on this slice`),
            o.note ? el("div", { class: "muted small" }, o.note) : null,
            o.findings.map((f) =>
              el(
                "div",
                { class: `look-finding${f.positive ? " positive" : ""}`, title: f.positive ? "At or above the display line" : null },
                el("span", {}, f.finding, f.positive ? el("span", { class: "line-tag" }, "at or above line") : null),
                el("span", {}, f.score_text),
              ),
            ),
          ),
        ),
      ),
    );
  }
  if (card.note) parts.push(el("p", { class: "muted" }, card.note));
  if (card.point?.where) parts.push(el("p", {}, card.point.where));
  if (card.point) parts.push(el("p", {}, card.point.hu));
  parts.push(el("div", { class: "look-actions" }, el("button", { type: "button", onclick: askChat }, "Ask the chat")));
  parts.push(el("p", { class: "look-footer" }, card.footer));
  box.replaceChildren(...parts);
  box.hidden = false;
  box.focus();
  lookSlice = sliceKey();
}

function hideLookCard({ refocus = true } = {}) {
  $("look-card").hidden = true;
  if (refocus) $("look-button").focus();
}

// Which slice the card describes: the view type and the crosshair's fraction along the
// displayed plane's normal (NiiVue's RAS order: x, y, z). Cheap, no pass over the mask.
let lookSlice = null;
function sliceKey() {
  const type = ws.viewer.state().slice_type;
  const axis = { sagittal: 0, coronal: 1 }[type] ?? 2;
  return `${type}:${ws.viewer.nv.scene.crosshairPos[axis]}`;
}

// A card about another slice would be stale, so it closes; w reopens it.
function closeStaleCard() {
  if (!$("look-card").hidden && sliceKey() !== lookSlice) hideLookCard({ refocus: false });
}

// persona 4.25 controller: open() shows the panel, submitMessage(text) sends it as the user.
function askChat() {
  if (!ws.chat) return notice("The chat is not available.");
  ws.chat.open();
  if (!ws.chat.submitMessage(LOOK_QUESTION)) notice("The chat could not send right now; try again.");
}

// ---------- wiring ----------

function wireToolbar() {
  for (const b of document.querySelectorAll("[data-slice]")) b.addEventListener("click", () => setSlice(b.dataset.slice));
  for (const b of document.querySelectorAll("[data-window]")) b.addEventListener("click", () => setWindow(b.dataset.window));
  $("mask-toggle").addEventListener("click", () => toggleMask());
  $("mask-opacity").addEventListener("input", (e) => ws.viewer.setMaskOpacity(Number(e.target.value) / 100));
  $("outline-toggle").addEventListener("click", (e) => {
    const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
    e.currentTarget.setAttribute("aria-pressed", String(on));
    ws.viewer.setOutline(on);
  });
  $("box-toggle").addEventListener("click", () => {
    if (!ws.activeOrgan) return notice("Select a finding or an organ first.");
    showScoringBox(ws.activeOrgan, !ws.boxOn);
  });
  $("look-button").addEventListener("click", showLookCard);
  $("show-all").addEventListener("click", () => {
    ws.viewer.isolateOrgan(null);
    clearNotice();
  });
  // Clicking an organ in the image scrolls the list to it.
  $("viewer").addEventListener("pointerup", () => {
    const organ = ws.viewer.location().organ;
    if (organ) ws.findings.scrollToOrgan(organ);
  });
}

// Keys typed into a field or into the chat panel are not shortcuts.
function typing(target) {
  if (!target.closest) return false;
  if (target.closest("input, textarea, select, [contenteditable=''], [contenteditable='true']")) return true;
  return Boolean(target.closest("[data-persona-root]") && !target.closest("#workspace-main"));
}

function step(delta) {
  const keys = ws.findings.visibleKeys();
  if (!keys.length) return;
  const i = keys.indexOf(ws.activeFinding);
  const next = i < 0 ? (delta > 0 ? 0 : keys.length - 1) : Math.max(0, Math.min(keys.length - 1, i + delta));
  selectFinding(keys[next]);
}

function wireKeys() {
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
    const k = e.key;
    if (k === "j") step(1);
    else if (k === "k") step(-1);
    else if (k === " ") toggleMask();
    else if (WINDOW_KEYS[k]) setWindow(WINDOW_KEYS[k]);
    else if (k === "[") setThreshold(ws.findings.state.threshold - 5);
    else if (k === "]") setThreshold(ws.findings.state.threshold + 5);
    else if (k === "m") setSlice(ws.viewer.state().slice_type === "multiplanar" ? "axial" : "multiplanar");
    else if (k === "w") showLookCard();
    else if (k === "Escape" && !$("look-card").hidden) hideLookCard();
    else return;
    e.preventDefault();
  });
}

async function main() {
  wireLogout();
  const scanId = new URLSearchParams(location.search).get("scan");
  if (!scanId) {
    $("scan-title").textContent = "No scan given. Open one from the Scans page.";
    return;
  }
  try {
    await get("/auth/me");
  } catch {
    return;
  }
  const [catalog, labels, scan] = await Promise.all([get("/catalog/findings"), get("/catalog/labels"), get(`/scans/${encodeURIComponent(scanId)}`)]);
  ws.scan = scan;
  ws.labels = labels.labels;
  ws.scoredOrgans = labels.scored_organs;

  ws.findings = createFindings({
    onSelect: (key) => selectFinding(key),
    onOrgan: (organ) => jumpToOrgan(organ),
  });
  ws.findings.setCatalog(catalog.findings, labels.scored_organs);
  ws.catalog = catalog.findings;
  ws.viewer = createViewer($("viewer"), {
    onLocation: (loc) => {
      renderReadout(loc);
      closeStaleCard();
    },
    onWindow: markWindow,
  });

  renderScanBar(scan);
  wireToolbar();
  wireKeys();

  ws.chat = mountChat({ target: "#workspace-main", context: () => ({ scan_id: ws.scan.id, job_id: ws.job?.id ?? null }) });
  const tools = await registerPageTools({ viewState, jumpToOrgan, selectFinding, setWindow, setThreshold, showScoringBox, toggleMask });
  window.__radar = { viewState, selectFinding, jumpToOrgan, setWindow, setThreshold, showScoringBox, toggleMask, tools, viewer: ws.viewer };

  if (scan.state !== "ready") {
    setJobStatus(scan.state === "rejected" ? `Rejected: ${scan.rejected_reason}` : "Upload not finished", "state-failed");
    notice(scan.state === "rejected" ? "This scan was rejected and cannot be viewed." : "The upload has not been completed.");
    return;
  }
  offerCT();
  if (!scan.latest_job) return renderJob(null);
  // While the latest job is not done, show the newest done one (the chat falls back the same way).
  if (scan.latest_job.state !== "done") {
    const { jobs } = await get(`/jobs?scan_id=${encodeURIComponent(scan.id)}`);
    const done = jobs.find((j) => j.state === "done");
    if (done) loadResult(done);
  }
  trackJob(await get(`/jobs/${scan.latest_job.id}`));
}

main().catch((e) => {
  if (!(e instanceof ApiError && e.status === 401)) setJobStatus(`Error: ${e.message}`, "state-failed");
});
