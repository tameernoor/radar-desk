// The Readers lens as pure functions: a row's three cells at the current display line, the flag, the order and
// the fold. No DOM here, so it unit-tests like light.js. Thresholds are in percent, like the findings slider.

export const SYMBOL = { present: "●", absent: "○", not_mentioned: "–", outside: "◌", not_scored: "·", pending: "…", none: "" };
export const STATE_TEXT = {
  present: "present",
  absent: "absent",
  not_mentioned: "not mentioned",
  outside: "outside NV's view",
  not_scored: "not scored",
  pending: "waiting",
  none: "",
};
export const KEY = ["present", "absent", "not_mentioned", "outside", "not_scored"];

// RADAR's best score over the row's findings against the display line; · when RADAR has no score for the row.
export function radarCell(row, threshold) {
  const r = row.radar;
  if (!r?.scored || r.max_prob == null) return "not_scored";
  return r.max_prob * 100 >= threshold ? "present" : "absent";
}

export function reportCell(row, reportPending) {
  if (reportPending) return "pending";
  return row.report ? row.report.state : "none";
}

// ◌ replaces NV's read when less than half the organ lies inside its crop; the read still shows when expanded.
export function nvCell(row, nvPending) {
  if (nvPending) return "pending";
  if (!row.nv) return "none";
  if (row.nv.in_view === false) return "outside";
  return row.nv.state;
}

export function cellsFor(row, { threshold, nvPending = false, reportPending = false }) {
  return { report: reportCell(row, reportPending), radar: radarCell(row, threshold), nv: nvCell(row, nvPending) };
}

// ⚑ only when one reader says present and another says absent. Silence, outside and not scored never count.
export function disagrees(cells) {
  const states = Object.values(cells);
  return states.includes("present") && states.includes("absent");
}

// Flagged rows first, then rows where any reader says present, then rows still waiting on a reader (never folded,
// since the fold says nobody reports it), then the rest folded away.
export function arrange(rows, opts) {
  const flagged = [];
  const positive = [];
  const waiting = [];
  const folded = [];
  for (const row of rows) {
    const cells = cellsFor(row, opts);
    const item = { row, cells, flag: disagrees(cells) };
    if (item.flag) flagged.push(item);
    else if (Object.values(cells).includes("present")) positive.push(item);
    else if (Object.values(cells).includes("pending")) waiting.push(item);
    else folded.push(item);
  }
  return { flagged, positive, waiting, folded };
}

export function foldText(n) {
  return `${n} ${n === 1 ? "row" : "rows"}: nobody reports it`;
}

export function nvPendingOf(view) {
  const state = view.nv?.job?.state;
  return state === "queued" || state === "submitted" || view.nv?.read?.status === "pending";
}

// While true the tab polls the server again.
export function isBusy(view) {
  return nvPendingOf(view) || view.report?.status === "pending" || view.queries?.status === "pending" || Boolean(view.nv?.asking?.length);
}

// NV's report text split at its IMPRESSION: heading. With no heading the whole text is the impression.
export function splitReport(answer) {
  const text = String(answer ?? "").trim();
  const m = /^[\s*#]*IMPRESSIONS?\**\s*:\**/im.exec(text);
  if (!m) return { findings: "", impression: text };
  return { findings: text.slice(0, m.index).trim(), impression: text.slice(m.index + m[0].length).replace(/^[\s*#]+/, "").trim() };
}

// Every prepared question once, general ones first, then each row's, de-duplicated by id.
export function allQuestions(view) {
  const seen = new Set();
  const out = [];
  for (const q of [...(view.queries?.general || []), ...view.rows.flatMap((r) => r.queries || [])]) {
    if (seen.has(q.id)) continue;
    seen.add(q.id);
    out.push(q);
  }
  return out;
}

// The latest answer NV gave to a prepared question (matched by its words), or null.
export function answerFor(view, q) {
  const all = view.nv?.answers || [];
  for (let i = all.length - 1; i >= 0; i--) if (all[i].question === q.text) return all[i];
  return null;
}

// Why there are no questions to show yet.
export function noQuestionsText(view) {
  const nv = view.nv || {};
  if (!nv.job) return nv.unavailable ? `No questions yet. NV-Reason-CT: ${nv.unavailable}.` : "No questions yet. Run NV-Reason-CT first; its report is what the questions are written from.";
  if (nv.job.state !== "done") return "No questions yet. They are written once NV-Reason-CT has read the scan.";
  if (nv.read?.status === "no_llm" || view.queries?.status === "no_llm") return "No questions yet. Writing them needs a chat model.";
  if (nv.read?.status === "failed" || view.queries?.status === "failed") return "The questions could not be written. Retry from the reports above.";
  return "Writing the questions…";
}

// The radiologist's report shortened to its first lines, with whether anything was cut.
export const REPORT_LINES = 12;
export function clipLines(text, max = REPORT_LINES) {
  const lines = String(text ?? "").split("\n");
  return lines.length > max ? { text: lines.slice(0, max).join("\n"), cut: true } : { text: String(text ?? ""), cut: false };
}
