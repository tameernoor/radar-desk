import "./styles.css";
import { ApiError, get, post } from "./api.js";
import { mountChat } from "./chat.js";
import { createFindings } from "./findings.js";
import { buildLookCard } from "./lookcard.js";
import { bytes, dims, el, elapsed, shortId, spacing, usd, when } from "./format.js";
import { COLORMAPS, rangeOf, sliderFromWidth, widthFromSlider } from "./light.js";
import { wireLogout } from "./login.js";
import { registerPageTools } from "./pagetools.js";
import { organCss, UNSCORED_GREY } from "./palette.js";
import { recipeFor } from "./recipes.js";
import { createViewer, rawBytes, tooBig, WINDOWS } from "./viewer.js";

const TERMINAL = new Set(["done", "failed", "cancelled"]);
const WINDOW_KEYS = { 1: "soft_tissue", 2: "liver", 3: "bone", 4: "lung", 5: "angio" };

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
  focus: false, // focus mode: the viewer over the whole window
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
  ws.findings.markCurrentOrgan(organ);
  syncControls();
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

// Back to all organs: no isolation, no active organ, no scoring box.
function clearOrgan() {
  ws.activeOrgan = null;
  ws.viewer.isolateOrgan(null);
  ws.viewer.showBox(null);
  setBoxOn(false);
  ws.findings.clearCurrentOrgan();
  clearNotice();
}

// A click on an organ chip or legend item: a second click on the active organ shows all organs
// again. Organ headings in the list and the chat's jump_to_organ always jump.
function toggleOrgan(organ) {
  if (ws.activeOrgan === organ) return clearOrgan();
  return jumpToOrgan(organ);
}

function selectFinding(keyOrName) {
  const f = ws.findings.finding(keyOrName);
  if (!f) throw new Error(`Unknown finding: ${keyOrName}`);
  ws.activeFinding = f.key;
  const jumped = jumpToOrgan(f.organ, { scroll: false });
  ws.findings.setActive(f.key);
  return { ...jumped, key: f.key, finding: f.finding, prob: ws.findings.probOf(f.key) };
}

// An organ or finding the way it is usually read: the jump, then the recipe's window, the organ in the
// middle of the view at the recipe's zoom, and the scoring box. A failed jump changes nothing else.
function setViewFor(target) {
  const r = recipeFor(target, (q) => ws.findings.finding(q));
  const jumped = r.key ? selectFinding(r.key) : jumpToOrgan(r.organ);
  if (!jumped.ok) return { ok: false, target, organ: r.organ, reason: jumped.reason };
  setWindow(r.preset);
  ws.viewer.centreOn(jumped.centroid_mm, r.zoom);
  const box = showScoringBox(r.organ, true);
  const v = ws.viewer.state();
  return {
    ok: true,
    target,
    organ: r.organ,
    finding: r.finding,
    key: r.key,
    window_preset: v.window_preset,
    window_hu: v.window_hu,
    zoom: v.zoom,
    scoring_box: box.ok ? { on: true, how: box.how, window_index: box.window_index } : { on: false, reason: box.reason },
    why: r.why,
    note: r.note,
  };
}

function setBoxOn(on) {
  ws.boxOn = on;
  syncControls();
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
  syncControls();
  return { ok: true, mask_on: next, mask_loaded: ws.viewer.state().mask_loaded };
}

function setOutline(on) {
  ws.viewer.setOutline(on ?? !ws.viewer.state().outline);
  syncControls();
}

// The viewer calls onLight (and so syncControls) after every light change, right-drag included.
function setWindow(preset) {
  ws.viewer.setWindow(preset);
  return { ok: true, window_preset: preset, hu_range: [WINDOWS[preset].min, WINDOWS[preset].max] };
}

function setWindowLevel({ width, level }) {
  if (!Number.isFinite(width) || !Number.isFinite(level)) throw new Error("width and level must be numbers in HU.");
  const { min, max } = rangeOf(width, level);
  ws.viewer.setWindowHU(min, max);
  const v = ws.viewer.state();
  return { ok: true, window_preset: v.window_preset, window_hu: v.window_hu };
}

// Back to the default look: soft tissue, gamma 1, no invert, gray, mask on at 0.45, no outline.
// Zoom and pan are left alone.
function resetLight() {
  ws.viewer.resetLight();
  ws.viewer.setMaskVisible(true);
  ws.viewer.setMaskOpacity(0.45);
  ws.viewer.setOutline(false);
  syncControls();
}

// Reset first, then gamma, invert and colour map; the reducer throws on bad values. The colour map is
// checked before anything applies, so a bad one changes nothing.
function setLight({ gamma, invert, colormap, reset } = {}) {
  if (gamma === undefined && invert === undefined && colormap === undefined && !reset) throw new Error("Give at least one of gamma, invert, colormap or reset.");
  if (colormap !== undefined && !COLORMAPS.includes(colormap)) throw new Error(`Unknown colour map: ${colormap}. Use one of ${COLORMAPS.join(", ")}.`);
  if (reset) ws.viewer.resetLight();
  if (gamma !== undefined) ws.viewer.setGamma(gamma);
  if (invert !== undefined) ws.viewer.setInvert(invert);
  if (colormap !== undefined) ws.viewer.setColormap(colormap);
  const v = ws.viewer.state();
  return { ok: true, gamma: v.gamma, invert: v.invert, colormap: v.colormap };
}

function setZoom(zoom) {
  ws.viewer.setZoom(zoom);
  return { ok: true, zoom: ws.viewer.state().zoom };
}

function toggleLight(on) {
  $("light-panel").hidden = !(on ?? $("light-panel").hidden);
  syncControls();
}

// Focus mode covers the page with the viewer and hides the chat. It touches no viewer state;
// NiiVue's own ResizeObserver resizes the canvas and the viewer's one redoes the multiplanar layout.
function toggleFocus(on) {
  const next = on ?? !ws.focus;
  if (next !== ws.focus) {
    ws.focus = next;
    document.body.classList.toggle("focus", next);
    // In multiplanar any right-side panel covers a reference view, so it waits for the Light button there.
    $("light-panel").hidden = !next || ws.viewer.state().slice_type === "multiplanar";
  }
  syncControls();
  return { ok: true, focus: ws.focus };
}

// Every toolbar and panel control from the viewer's state, so clicks, keys, page tools and
// right-drag all show the same thing. `from` is the input being dragged; it is left alone so
// the log-scaled width slider does not snap under the pointer.
function syncControls(from = null) {
  const v = ws.viewer.state();
  const press = (selector, on) => {
    for (const b of document.querySelectorAll(selector)) b.setAttribute("aria-pressed", String(typeof on === "function" ? on(b) : on));
  };
  const set = (id, value) => $(id) !== from && ($(id).value = String(value));
  press("[data-slice]", (b) => b.dataset.slice === v.slice_type || (v.slice_type === "multiplanar" && b.dataset.slice === v.main_plane));
  press("[data-window]", (b) => b.dataset.window === v.window_preset);
  press("[data-mask]", v.mask_on);
  press("[data-outline]", v.outline);
  for (const input of document.querySelectorAll("[data-mask-opacity]")) if (input !== from) input.value = String(Math.round(v.mask_opacity * 100));
  press("#box-toggle", ws.boxOn);
  const viewFor = $("view-for");
  viewFor.textContent = viewFor.title = `View for ${ws.activeOrgan ?? "organ"}`;
  viewFor.classList.toggle("muted", !ws.activeOrgan);
  press("#focus-toggle", ws.focus);
  press("#light-toggle", !$("light-panel").hidden);
  const hu = (x) => String(Math.round(x * 10) / 10);
  const w = v.window_hu;
  set("ww", sliderFromWidth(w.width));
  set("wl", Math.round(w.level));
  $("ww-value").textContent = hu(w.width);
  $("wl-value").textContent = hu(w.level);
  $("window-range").textContent = `${WINDOWS[v.window_preset]?.label ?? "Custom"}, ${hu(w.min)} to ${hu(w.max)} HU`;
  set("gamma", v.gamma);
  $("gamma-value").textContent = v.gamma.toFixed(2);
  press("#invert", v.invert);
  set("colormap", v.colormap);
}

// Multiplanar toggles; a plane button in multiplanar picks the main view.
function setSlice(name) {
  if (name === "multiplanar" && ws.viewer.state().slice_type === "multiplanar") ws.viewer.exitMultiplanar();
  else ws.viewer.setSliceType(name);
}

// Called by the viewer after any view change, including a swap from a reference tile.
function onViewChange() {
  syncControls();
  closeStaleCard();
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
    focus: ws.focus,
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
    el("button", { type: "button", class: "legend-item", onclick: () => toggleOrgan(o.organ) }, el("i", { style: `background:${organCss(o.organ)}` }), o.organ),
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
  const { slice_type: type, main_plane: main } = ws.viewer.state();
  const axis = { sagittal: 0, coronal: 1 }[main] ?? 2;
  return `${type}:${main}:${ws.viewer.nv.scene.crosshairPos[axis]}`;
}

// A card about another slice would be stale, so it closes; w reopens it.
function closeStaleCard() {
  if (!$("look-card").hidden && sliceKey() !== lookSlice) hideLookCard({ refocus: false });
}

// persona 4.25 controller: open() shows the panel, submitMessage(text) sends it as the user.
function askChat() {
  if (!ws.chat) return notice("The chat is not available.");
  toggleFocus(false); // the chat is hidden in focus mode
  ws.chat.open();
  if (!ws.chat.submitMessage(LOOK_QUESTION)) notice("The chat could not send right now; try again.");
}

// ---------- wiring ----------

let sliderSource = null; // the slider whose input event is being handled, see dragging()

function wireToolbar() {
  for (const b of document.querySelectorAll("[data-slice]")) b.addEventListener("click", () => setSlice(b.dataset.slice));
  for (const b of document.querySelectorAll("[data-window]")) b.addEventListener("click", () => setWindow(b.dataset.window));
  for (const b of document.querySelectorAll("[data-mask]")) b.addEventListener("click", () => toggleMask());
  for (const b of document.querySelectorAll("[data-outline]")) b.addEventListener("click", () => setOutline());
  for (const input of document.querySelectorAll("[data-mask-opacity]")) {
    input.addEventListener("input", (e) => {
      ws.viewer.setMaskOpacity(Number(e.target.value) / 100);
      syncControls(e.target);
    });
  }
  // Sliders apply on input; the viewer's onLight then updates the numbers.
  const dragging = (handler) => (e) => {
    sliderSource = e.target;
    try {
      handler(e);
    } finally {
      sliderSource = null;
    }
  };
  $("ww").addEventListener("input", dragging((e) => setWindowLevel({ width: widthFromSlider(Number(e.target.value)), level: ws.viewer.state().window_hu.level })));
  $("wl").addEventListener("input", dragging((e) => setWindowLevel({ width: ws.viewer.state().window_hu.width, level: Number(e.target.value) })));
  $("gamma").addEventListener("input", dragging((e) => ws.viewer.setGamma(Number(e.target.value))));
  $("invert").addEventListener("click", () => ws.viewer.setInvert(!ws.viewer.state().invert));
  $("colormap").addEventListener("change", (e) => {
    ws.viewer.setColormap(e.target.value);
    e.target.blur(); // a focused select swallows the shortcuts
  });
  $("light-reset").addEventListener("click", resetLight);
  $("light-toggle").addEventListener("click", () => toggleLight());
  $("light-close").addEventListener("click", () => toggleLight(false));
  $("focus-toggle").addEventListener("click", () => toggleFocus());
  $("zoom-in").addEventListener("click", () => ws.viewer.zoomBy(1));
  $("zoom-out").addEventListener("click", () => ws.viewer.zoomBy(-1));
  $("view-reset").addEventListener("click", () => ws.viewer.resetView());
  $("box-toggle").addEventListener("click", () => {
    if (!ws.activeOrgan) return notice("Select a finding or an organ first.");
    showScoringBox(ws.activeOrgan, !ws.boxOn);
  });
  $("view-for").addEventListener("click", () => {
    if (!ws.activeOrgan) return notice("Select a finding or an organ first.");
    setViewFor(ws.activeOrgan);
  });
  $("look-button").addEventListener("click", showLookCard);
  $("show-all").addEventListener("click", clearOrgan);
  // Clicking an organ in the image scrolls the list to it.
  $("viewer").addEventListener("pointerup", () => {
    const organ = ws.viewer.location().organ;
    if (organ) ws.findings.scrollToOrgan(organ);
  });
}

// Keys typed into a field or into the chat panel are not shortcuts. A slider is not a field.
function typing(target) {
  if (!target.closest) return false;
  if (target.closest("input:not([type=range]), textarea, select, [contenteditable=''], [contenteditable='true']")) return true;
  return Boolean(target.closest("[data-persona-root]") && !target.closest("#workspace-main"));
}

// In focus mode the toolbar and bottom bar float over the viewer; panels and notices start below
// the toolbar, so the pane carries both bars' heights.
function wireBars() {
  const pane = document.querySelector(".viewer-pane");
  const bottom = document.querySelector(".viewer-bottom");
  const measure = () => {
    pane.style.setProperty("--toolbar-h", `${$("viewer-toolbar").offsetHeight}px`);
    pane.style.setProperty("--bottom-h", `${bottom.offsetHeight}px`);
  };
  const observer = new ResizeObserver(measure);
  observer.observe($("viewer-toolbar"));
  observer.observe(bottom);
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
    else if (k === "m") setSlice("multiplanar");
    else if (k === "w") showLookCard();
    else if (k === "f") toggleFocus();
    else if (k === "+" || k === "=") ws.viewer.zoomBy(1);
    else if (k === "-") ws.viewer.zoomBy(-1);
    else if (k === "Escape" && !$("look-card").hidden) hideLookCard();
    else if (k === "Escape" && !ws.focus && !$("light-panel").hidden) {
      toggleLight(false);
      $("light-toggle").focus();
    }
    else if (k === "Escape" && ws.focus) toggleFocus(false);
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
    onOrgan: (organ, { toggle = true } = {}) => (toggle ? toggleOrgan(organ) : jumpToOrgan(organ)),
  });
  ws.findings.setCatalog(catalog.findings, labels.scored_organs);
  ws.catalog = catalog.findings;
  ws.viewer = createViewer($("viewer"), {
    onLocation: (loc) => {
      renderReadout(loc);
      closeStaleCard();
    },
    onLight: () => syncControls(sliderSource),
    onView: onViewChange,
  });

  renderScanBar(scan);
  wireToolbar();
  wireKeys();
  wireBars();
  syncControls();

  ws.chat = mountChat({ target: "#workspace-main", context: () => ({ scan_id: ws.scan.id, job_id: ws.job?.id ?? null }) });
  const actions = { viewState, jumpToOrgan, selectFinding, setWindow, setWindowLevel, setThreshold, showScoringBox, toggleMask, toggleFocus, setZoom, setViewFor, setLight };
  const tools = await registerPageTools(actions);
  window.__radar = { ...actions, toggleLight, resetLight, tools, viewer: ws.viewer };

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
