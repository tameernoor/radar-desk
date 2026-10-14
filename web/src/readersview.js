// The Readers tab: one row per label with the radiologist report, RADAR and NV-Reason-CT side by side.
// It never says who is right; it shows where the readers agree, disagree, are silent or cannot see.
// Every word from a report, a read or an answer goes in as text (el() uses text nodes), never as HTML.
import { el, pct } from "./format.js";
import { allQuestions, answerFor, arrange, clipLines, foldText, KEY, noQuestionsText, nvPendingOf, splitReport, STATE_TEXT, SYMBOL } from "./readers.js";

const NAMES = { report: "Report", radar: "RADAR", nv: "NV" };
const YES_NO_SUFFIX = "Answer only Yes or No."; // mirrors the Python constant; hidden in the display, still sent to NV
const shown = (text) => text.endsWith(YES_NO_SUFFIX) ? text.slice(0, -YES_NO_SUFFIX.length).trimEnd() : text;
const HOLDS = { needs_modal: "needs Modal compute", budget: "over the monthly GPU budget", spawn_error: "could not start, trying again" };

export function createReadersView({ root, onOrgan, onRetryReads, onRunNv, onAsk, onAddReport }) {
  const state = { view: null, threshold: 50, open: new Set(), foldOpen: false, othersOpen: false, error: null, asking: new Set(), askError: null, detailsOpen: new Set(), reportOpen: false };

  // No Report column without a report, no NV column when NV cannot run and never did.
  const readers = () => {
    const nv = state.view?.nv;
    return [state.view?.has_report ? "report" : null, "radar", nv?.unavailable && !nv.job ? null : "nv"].filter(Boolean);
  };
  const link = (label, run) => el("button", { type: "button", class: "link rretry", onclick: run }, label);
  const note = (text) => el("span", { class: "muted small" }, text);

  const section = (title, ...body) => el("section", { class: "rsec" }, el("h3", {}, title), ...body);

  // <details> that stay open across the re-renders a poll causes.
  function folded1(key, label, body) {
    return el("details", { class: "rfolded", open: state.detailsOpen.has(key), ontoggle: (e) => (e.target.open ? state.detailsOpen.add(key) : state.detailsOpen.delete(key)) }, el("summary", {}, label), body);
  }

  function nvCard(view) {
    const nv = view.nv;
    const job = nv.job;
    const body = [];
    if (!job) {
      body.push(nv.unavailable ? note(`Not available: ${nv.unavailable}.`) : el("p", { class: "muted" }, "Not run yet. ", link("Run NV-Reason-CT", onRunNv)));
    } else if (job.state === "failed" || job.state === "cancelled") {
      body.push(el("p", { class: "rretry" }, `It did not finish${job.error?.message ? `: ${job.error.message}` : "."} `, link("Retry", onRunNv)));
    } else if (job.state === "queued" && job.hold_reason) {
      body.push(note(`Waiting, ${HOLDS[job.hold_reason] || job.hold_reason}.`));
    } else if (job.state === "queued" || job.state === "submitted") {
      body.push(el("p", { class: "muted" }, "Reading the scan, about 2 minutes."));
    } else if (nv.report) {
      const { findings, impression } = splitReport(nv.report.answer);
      body.push(el("p", { class: "rtext", "data-part": "impression" }, impression || "NV wrote no impression."));
      if (findings) body.push(folded1("nv-findings", "Findings", el("p", { class: "rtext" }, findings)));
      if (nv.report.thinking) body.push(folded1("nv-thinking", "How it reasoned", el("p", { class: "rtext muted" }, nv.report.thinking)));
      const small = ["Research model. It saw a 384 mm cube of the scan."];
      if (nv.fake) small.push("This is the fake backend, not model output.");
      if (nv.crop?.fallback) small.push("It placed its view without finding the abdomen.");
      body.push(note(small.join(" ")));
    } else {
      body.push(note("Its report is not available."));
    }
    if (job?.state === "done") {
      if (nv.read.status === "pending") body.push(el("p", {}, note("Reading its report for the table below…")));
      if (nv.read.status === "failed") body.push(el("p", {}, link("Could not read its report for the table, retry", onRetryReads)));
      if (nv.read.status === "no_llm") body.push(el("p", {}, note("Reading its report for the table needs a chat model.")));
    }
    return el("div", { class: "rcard", "data-card": "nv" }, el("h4", {}, "NV-Reason-CT's report"), ...body);
  }

  function radiologistCard(view) {
    const body = [];
    if (!view.has_report) {
      body.push(el("p", { class: "muted" }, "No report attached."), el("button", { type: "button", class: "rquiet", onclick: onAddReport }, "Add report"));
    } else {
      const text = view.report.text || "";
      const clip = clipLines(text);
      const showAll = state.reportOpen || !clip.cut;
      body.push(el("p", { class: "rtext", "data-part": "radiologist" }, showAll ? text : `${clip.text}…`));
      if (clip.cut) body.push(link(state.reportOpen ? "Show less" : "Show all", () => ((state.reportOpen = !state.reportOpen), render())));
      const status = view.report.status;
      if (status === "pending") body.push(el("p", {}, note("Reading it for the table below…")));
      if (status === "failed") body.push(el("p", {}, link("Could not read it for the table, retry", onRetryReads)));
      if (status === "no_llm") body.push(el("p", {}, note("Reading it for the table needs a chat model.")));
    }
    return el("div", { class: "rcard", "data-card": "report" }, el("h4", {}, "Radiologist's report"), ...body);
  }

  // One question asked in place: its button, then its answer, failure or wait underneath.
  function question(q) {
    const nv = state.view.nv;
    const busy = state.asking.has(q.id) || Boolean(nv.asking?.includes(q.id));
    const inFlight = state.asking.size > 0 || Boolean(nv.asking?.length);
    const answer = answerFor(state.view, q);
    const failed = nv.ask_errors?.[q.id];
    const under = [];
    if (busy) under.push(el("p", { class: "muted small rwait" }, nv.warm ? "Asking…" : "Asking... the first question after a pause takes about 2 minutes."));
    else if (failed) under.push(el("p", { class: "rretry small" }, `${failed[0].toUpperCase()}${failed.slice(1)}.`));
    if (answer && !busy) {
      under.push(el("p", { class: "rqa" }, `NV: ${answer.answer || "no answer"}`, answer.differs ? el("span", { class: "rdiffers" }, `${String(answer.answer ?? "").endsWith(".") ? " " : ". "}NV's answer differs from its report`) : null));
    }
    return el("div", { class: "rq", "data-q": q.id }, el("button", { type: "button", class: "rchip", "data-query": q.id, disabled: inFlight, title: "Ask NV-Reason-CT this question", onclick: (e) => (e.stopPropagation(), ask(q)) }, shown(q.text)), ...under);
  }

  function askSection(view) {
    const items = allQuestions(view);
    return section(
      "Ask NV-Reason-CT",
      items.length ? el("div", { class: "rqs" }, items.map(question)) : el("p", { class: "muted" }, noQuestionsText(view)),
      state.askError ? el("p", { class: "rretry" }, state.askError) : null,
      el("p", { class: "muted small" }, "Or type your own question in the chat on the right."),
    );
  }

  function header() {
    return el("div", { class: "rhead", style: `--cols:${readers().length}` }, el("span"), el("span"), el("span"), readers().map((r) => el("span", { class: "rcol" }, NAMES[r])));
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

  async function ask(q) {
    state.asking.add(q.id);
    state.askError = null;
    render();
    try {
      await onAsk(q.id);
    } catch (e) {
      state.askError = `Could not ask NV-Reason-CT: ${e.message}`;
    } finally {
      state.asking.delete(q.id);
      render();
    }
  }

  function details(row) {
    const parts = [];
    if (state.view.has_report) parts.push(words("Report", row.report, "not read yet"));
    const scores = row.radar.findings.filter((f) => f.prob != null);
    parts.push(el("p", {}, el("strong", {}, "RADAR: "), scores.length ? scores.map((f) => `${f.finding} (${f.organ}) ${pct(f.prob)}`).join(", ") : "no RADAR score for this row"));
    const nvShown = readers().includes("nv");
    if (nvShown) parts.push(words("NV-Reason-CT", row.nv, "not read yet"));
    if (nvShown && row.nv?.in_view === false) parts.push(el("p", { class: "muted small" }, `Only ${Math.round(row.nv.coverage * 100)}% of the ${row.organ.toLowerCase()} is inside NV's view.`));
    const prepared = new Set((row.queries || []).map((q) => q.text));
    for (const a of row.asked.filter((x) => !prepared.has(x.question))) {
      parts.push(
        el("p", { class: "rasked" }, el("strong", {}, "Asked NV: "), shown(a.question), el("br"), `NV: ${a.answer ?? "no answer"}`, a.differs ? el("span", { class: "rdiffers" }, `${String(a.answer ?? "").endsWith(".") ? " " : ". "}NV's answer differs from its report`) : null),
      );
    }
    if (row.queries?.length) parts.push(el("div", { class: "rqs" }, row.queries.map(question)));
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
    const states = KEY.filter((s) => s !== "outside" || readers().includes("nv"));
    const items = states.map((s) => el("span", {}, el("span", { class: "rcell", "data-state": s }, SYMBOL[s]), ` ${STATE_TEXT[s]}`));
    return el("p", { class: "rkey muted small" }, [...items, el("span", {}, "⚑ disagree")]);
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
    const parts = [
      el("p", { class: "rintro muted" }, "Three readers of the same scan: the radiologist's report, RADAR's scores and NV-Reason-CT's own report. Nothing here says who is right."),
      section("Reports", el("div", { class: "rcards" }, radiologistCard(view), nvCard(view))),
      askSection(view),
      section(
        "Compare the readers",
        el("p", { class: "muted small" }, "Each row is a finding. ● says present, ○ says absent. ⚑ marks rows where the readers disagree. Click a row to see each reader's own words."),
        header(),
        ...[...flagged, ...positive, ...waiting].flatMap(rowNodes),
        ...fold(folded),
        ...others(view),
        legend(),
      ),
    ];
    root.replaceChildren(...parts.filter(Boolean));
    root.dataset.ready = "true";
  }

  return {
    setView(view) {
      state.view = view;
      state.error = null;
      state.askError = null;
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
