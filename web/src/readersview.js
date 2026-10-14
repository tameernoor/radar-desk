// The Readers tab: one row per label with the radiologist report, RADAR and NV-Reason-CT side by side.
// It never says who is right; it shows where the readers agree, disagree, are silent or cannot see.
// Every word from a report, a read or an answer goes in as text (el() uses text nodes), never as HTML.
import { el, pct } from "./format.js";
import { arrange, foldText, KEY, nvPendingOf, STATE_TEXT, SYMBOL } from "./readers.js";

const NAMES = { report: "Report", radar: "RADAR", nv: "NV" };
const HOLDS = { needs_modal: "needs Modal compute", budget: "over the monthly GPU budget", spawn_error: "could not start, trying again" };

export function createReadersView({ root, onOrgan, onRetryReads, onRunNv }) {
  const state = { view: null, threshold: 50, open: new Set(), foldOpen: false, othersOpen: false, error: null };

  const readers = () => (state.view?.has_report ? ["report", "radar", "nv"] : ["radar", "nv"]);
  const link = (label, run) => el("button", { type: "button", class: "link rretry", onclick: run }, label);
  const note = (text) => el("span", { class: "muted small" }, text);

  // row.report is null while the report is unread, so the column head says why.
  function reportHead(view) {
    const status = view.report.status;
    if (status === "pending") return note("Report: reading…");
    if (status === "failed") return link("Report: could not read it, retry", onRetryReads);
    if (status === "no_llm") return note("Report: needs a chat model");
    return null;
  }

  function nvHead(view) {
    const nv = view.nv;
    const job = nv.job;
    if (!job) return nv.unavailable ? note(`NV: ${nv.unavailable}`) : link("Run NV-Reason-CT", onRunNv);
    if (job.state === "failed" || job.state === "cancelled") return link("NV: failed, retry", onRunNv);
    if (job.state === "queued" && job.hold_reason) return note(`NV: waiting, ${HOLDS[job.hold_reason] || job.hold_reason}`);
    if (job.state === "queued" || job.state === "submitted") return note("NV: reading the scan…");
    if (nv.read.status === "pending") return note("NV: reading its report…");
    if (nv.read.status === "failed") return link("NV: could not read its report, retry", onRetryReads);
    if (nv.read.status === "no_llm") return note("NV: needs a chat model");
    const notes = [];
    if (nv.fake) notes.push("fake backend");
    if (nv.crop?.fallback) notes.push("NV placed its view without finding the abdomen");
    return notes.length ? note(`NV: ${notes.join(", ")}`) : null;
  }

  function header(view) {
    const notes = [view.has_report ? reportHead(view) : null, nvHead(view)].filter(Boolean);
    return [
      el("div", { class: "rhead", style: `--cols:${readers().length}` }, el("span"), el("span"), el("span"), readers().map((r) => el("span", { class: "rcol" }, NAMES[r]))),
      notes.length ? el("div", { class: "rnotes" }, notes) : null,
    ];
  }

  function cell(reader, value) {
    return el("span", { class: "rcell", "data-reader": reader, "data-state": value, title: `${NAMES[reader]}: ${STATE_TEXT[value] || "not read"}` }, SYMBOL[value]);
  }

  // A claim without a quote is shown as such, so it reads as unverified.
  function words(label, read, fallback) {
    if (!read) return el("p", {}, el("strong", {}, `${label}: `), fallback);
    if (read.quote) return el("p", {}, el("strong", {}, `${label}: `), `${STATE_TEXT[read.state]}, "${read.quote}"`);
    const claim = read.state === "present" || read.state === "absent";
    return el("p", {}, el("strong", {}, `${label}: `), claim ? `${STATE_TEXT[read.state]}, no quote` : STATE_TEXT[read.state]);
  }

  function details(row) {
    const parts = [];
    if (state.view.has_report) parts.push(words("Report", row.report, "not read yet"));
    const scores = row.radar.findings.filter((f) => f.prob != null);
    parts.push(el("p", {}, el("strong", {}, "RADAR: "), scores.length ? scores.map((f) => `${f.finding} (${f.organ}) ${pct(f.prob)}`).join(", ") : "no RADAR score for this row"));
    parts.push(words("NV-Reason-CT", row.nv, "not read yet"));
    if (row.nv?.in_view === false) parts.push(el("p", { class: "muted small" }, `Only ${Math.round(row.nv.coverage * 100)}% of the ${row.organ.toLowerCase()} is inside NV's view.`));
    for (const a of row.asked) {
      parts.push(
        el("p", { class: "rasked" }, el("strong", {}, "Asked NV: "), a.question, el("br"), `NV: ${a.answer}`, a.differs ? el("span", { class: "rdiffers" }, " (differs from NV's report)") : null),
      );
    }
    return el("div", { class: "rdetails" }, parts);
  }

  function toggleRow(id) {
    if (state.open.has(id)) state.open.delete(id);
    else state.open.add(id);
    render();
  }

  function rowNodes({ row, cells, flag }) {
    const open = state.open.has(row.id);
    const node = el(
      "div",
      {
        class: `rrow${flag ? " flagged" : ""}${row.organ ? " jumps" : ""}`,
        style: `--cols:${readers().length}`,
        "data-row": row.id,
        title: row.organ ? `Jump to the ${row.organ.toLowerCase()}` : null,
        onclick: () => row.organ && onOrgan(row.organ),
      },
      el("button", { type: "button", class: "rexpand", "aria-expanded": String(open), "aria-label": `${open ? "Hide" : "Show"} details for ${row.title}`, onclick: (e) => (e.stopPropagation(), toggleRow(row.id)) }, open ? "▾" : "▸"),
      el("span", { class: "rflag", title: flag ? "One reader says present, another says absent" : null }, flag ? "⚑" : ""),
      el("span", { class: "rtitle" }, row.title),
      readers().map((r) => cell(r, cells[r])),
    );
    return open ? [node, details(row)] : [node];
  }

  function fold(items) {
    if (!items.length) return [];
    const head = el("button", { type: "button", class: "rfold", "aria-expanded": String(state.foldOpen), onclick: () => ((state.foldOpen = !state.foldOpen), render()) }, `${state.foldOpen ? "▾" : "▸"} ${foldText(items.length)}`);
    return state.foldOpen ? [head, ...items.flatMap(rowNodes)] : [head];
  }

  function others(view) {
    const items = view.other_mentions || [];
    if (!items.length) return [];
    const head = el("button", { type: "button", class: "rfold", "aria-expanded": String(state.othersOpen), onclick: () => ((state.othersOpen = !state.othersOpen), render()) }, `${state.othersOpen ? "▾" : "▸"} Other mentions (${items.length})`);
    if (!state.othersOpen) return [head];
    return [head, el("ul", { class: "rothers" }, items.map((m) => el("li", {}, el("span", { class: "muted small" }, m.reader === "report" ? "Report: " : "NV: "), `"${m.quote}"`)))];
  }

  function legend() {
    return el("p", { class: "rkey muted small" }, [...KEY.map((s) => `${SYMBOL[s]} ${STATE_TEXT[s]}`), "⚑ disagree"].join("   "));
  }

  function render() {
    if (state.error) return root.replaceChildren(el("p", { class: "muted" }, state.error));
    const view = state.view;
    if (!view) return root.replaceChildren(el("p", { class: "muted" }, "Loading the readers…"));
    const { flagged, positive, waiting, folded } = arrange(view.rows, {
      threshold: state.threshold,
      nvPending: nvPendingOf(view),
      reportPending: view.has_report && view.report.status === "pending",
    });
    const intro = view.has_report
      ? "The radiologist report, RADAR and NV-Reason-CT on the same scan. None of them is treated as right."
      : "RADAR and NV-Reason-CT on the same scan. None of them is treated as right. Add a report to compare a third reader.";
    const parts = [el("p", { class: "rintro muted small" }, intro), ...header(view), ...[...flagged, ...positive, ...waiting].flatMap(rowNodes), ...fold(folded), ...others(view), legend()];
    root.replaceChildren(...parts.filter(Boolean));
    root.dataset.ready = "true";
  }

  return {
    setView(view) {
      state.view = view;
      state.error = null;
      render();
    },
    setError(message) {
      state.error = message;
      render();
    },
    setThreshold(percent) {
      state.threshold = percent;
      if (state.view) render();
    },
    get view() {
      return state.view;
    },
  };
}
