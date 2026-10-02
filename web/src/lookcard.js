// "What am I looking at?" card content, built from the view state with no model call.
// Pure: no DOM, no NiiVue. The workspace renders the structure this returns.
import { huGuess } from "./hu.js";

export const LOOK_FOOTER = "Outlines are RADAR's segmentation, not confirmed anatomy.";
const MAX_FINDINGS = 3;
const PLANE_NAMES = { axial: "Axial", coronal: "Coronal", sagittal: "Sagittal" };
const VIEW_NAMES = { axial: "Axial view", coronal: "Coronal view", sagittal: "Sagittal view", multiplanar: "Multiplanar view", render: "3D view" };

function planeLine(view) {
  const s = view.slice;
  const where = `${PLANE_NAMES[s.axis]} slice ${s.number} of ${s.count}`;
  if (view.plane === "multiplanar") return `Multiplanar view, ${where.toLowerCase()}`;
  if (view.plane === "render") return `3D view, ${where.toLowerCase()}`;
  return where;
}

// Highest scores first, so findings at or above the display line come before the rest.
// Without a result, an organ lists its first catalog findings with "no score".
function topFindings(organ, scores, catalog, thresholdPct) {
  const source = scores.length ? scores : catalog;
  const rows = source.filter((f) => f.organ === organ).map((f) => ({ key: f.key, finding: f.finding, prob: f.prob ?? null }));
  const scored = rows.filter((f) => f.prob != null).sort((a, b) => b.prob - a.prob);
  const picked = scored.length ? scored : rows;
  return picked.slice(0, MAX_FINDINGS).map((f) => ({
    ...f,
    positive: f.prob != null && f.prob * 100 >= thresholdPct,
    score_text: f.prob == null ? "no score" : `${(f.prob * 100).toFixed(1)}%`,
  }));
}

function pointLines(view) {
  const c = view.crosshair;
  let where;
  if (c.label == null) where = null; // no mask yet; the card's note says so
  else if (c.label > 0) where = `The crosshair is inside RADAR's ${c.organ} outline.`;
  else if (view.nearest_organ) where = `The crosshair is on background. The nearest outline is ${view.nearest_organ.organ}, ${view.nearest_organ.distance_mm} mm away on this slice.`;
  else where = "The crosshair is on background, and nothing is outlined on this slice.";
  const guess = huGuess(c.hu);
  return { where, hu: guess ? `${guess.text}.` : "No HU value under the crosshair." };
}

function organEntry(o, findings, catalog, thresholdPct) {
  const entry = { organ: o.organ, label: o.label, percent: o.percent_of_mask, scored: o.scored, findings: [], note: null };
  if (!o.scored) return { ...entry, note: "Segmented by RADAR but not one of the 18 scored organs." };
  if (!findings.length) return { ...entry, findings: topFindings(o.organ, [], catalog, thresholdPct), note: "No result loaded yet." };
  const top = topFindings(o.organ, findings, catalog, thresholdPct);
  if (top.every((f) => f.prob == null)) return { ...entry, note: "RADAR did not score this organ in this run." };
  return { ...entry, findings: top };
}

// view: get_view_state output. findings: the loaded result's findings ([] when none).
// catalog: the 146 catalog findings. Returns a plain object.
export function buildLookCard(view, findings = [], catalog = []) {
  if (!view?.crosshair) return { empty: true, message: "No CT is on screen.", footer: LOOK_FOOTER };
  const thresholdPct = view.threshold_pct ?? 50;
  const card = { empty: false, plane: null, organs: [], note: null, point: pointLines(view), footer: LOOK_FOOTER };
  if (!view.slice) {
    card.plane = VIEW_NAMES[view.plane] || null;
    card.note = "The CT is loaded, but RADAR's outlines are not there yet. The HU value under the crosshair is below.";
    return card;
  }
  card.plane = planeLine(view);
  card.organs = (view.organs_on_slice || []).map((o) => organEntry(o, findings, catalog, thresholdPct));
  if (!card.organs.length) card.note = "RADAR outlined nothing on this slice.";
  return card;
}
