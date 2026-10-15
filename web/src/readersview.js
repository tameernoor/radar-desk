// The NV-Reason-CT findings tab as an overview: NV's status and impression, then the findings grouped by where the
// readers disagree, what NV found, what only the others found and what NV could not see. It never says who is right.
// Every word from a report, a read or an answer goes in as text (el() uses text nodes), never as HTML.
import { el, pct } from "./format.js";
import { allQuestions, answerFor, foldText, HOLDS, noQuestionsText, nvPendingOf, overview, runNotes, shown, splitReport, STATE_TEXT, WORD } from "./readers.js";

const NAMES = { report: "Report", radar: "RADAR", nv: "NV" };
const ANSWER_CHARS = 300; // longer answers are cut here in the narrow pane; the reports window has them in full

export function createReadersView({ root, onOrgan, onRetryReads, onRunNv, onAsk, onAddReport, onOpenReports }) {
  const state = { view: null, threshold: 50, open: new Set(), foldOpen: false, othersOpen: false, error: null, asking: new Set(), askError: null };

  // No Report column without a report, no NV column when NV cannot run and never did.
  const readers = () => {
    const nv = state.view?.nv;
    return [state.view?.has_report ? "report" : null, "radar", nv?.unavailable && !nv.job ? null : "nv"].filter(Boolean);
  };
  const link = (label, run) => el("button", { type: "button", class: "link rretry", onclick: run }, label);
  const note = (text) => el("span", { class: "muted" }, text);

  const section = (title, ...body) => el("section", { class: "rsec rgroup" }, el("h3", { class: "rgroup-head" }, title), ...body);

  // NV's status on one line, with the action it offers; the notes under it are muted lines.
  function nvStatus(view) {
    const nv = view.nv;
    const job = nv.job;
    if (!job) return nv.unavailable ? note(`not available: ${nv.unavailable}.`) : el("span", { class: "muted" }, "not run yet. ", link("Run NV-Reason-CT", onRunNv));
    if (job.state === "failed" || job.state === "cancelled") return el("span", { class: "rretry" }, `did not finish${job.error?.message ? `: ${job.error.message}` : "."} `, link("Retry", onRunNv));
    if (job.state === "queued" && job.hold_reason) return note(`waiting, ${HOLDS[job.hold_reason] || job.hold_reason}.`);
    if (job.state === "queued" || job.state === "submitted") return el("span", { class: "muted" }, "reading the scan, about 2 minutes.");
    return nv.report ? el("span", { class: "muted" }, "saw a 384 mm cube of the scan") : note("its report is not available.");
  }

  function headBlock(view) {
    const nv = view.nv;
    const done = nv.job?.state === "done";
    const parts = [el("div", { class: "ohead" }, el("p", { class: "otitle" }, el("strong", {}, "NV-Reason-CT"), " ", nvStatus(view)), el("button", { type: "button", class: "rquiet", onclick: onOpenReports }, "Open reports"))];
    if (done && nv.report) {
      const { impression } = splitReport(nv.report.answer);
      parts.push(el("p", { class: "rtext rclamp", "data-part": "impression" }, el("strong", {}, "Impression: "), impression || "NV wrote no impression."));
      const fine = runNotes(nv).join(" ");
      if (fine) parts.push(el("p", { class: "muted small" }, fine));
    }
    if (done) {
      if (nv.read.status === "pending") parts.push(el("p", { class: "muted small" }, "Reading its report for the findings below…"));
      if (nv.read.status === "failed") parts.push(el("p", { class: "small" }, link("Could not read its report for the findings, retry", onRetryReads)));
      if (nv.read.status === "no_llm") parts.push(el("p", { class: "muted small" }, "Reading its report for the findings needs a chat model."));
    }
    if (view.has_report) {
      parts.push(el("p", { class: "muted small" }, "Radiologist's report attached."));
      const status = view.report.status;
      if (status === "pending") parts.push(el("p", { class: "muted small" }, "Reading it for the findings below…"));
      if (status === "failed") parts.push(el("p", { class: "small" }, link("Could not read it for the findings, retry", onRetryReads)));
      if (status === "no_llm") parts.push(el("p", { class: "muted small" }, "Reading it for the findings needs a chat model."));
    } else {
      parts.push(el("p", { class: "muted small" }, "No radiologist's report. ", el("button", { type: "button", class: "rquiet", onclick: onAddReport }, "Add report")));
    }
    return el("div", { class: "rsec ohead-block" }, parts);
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
      const full = answer.answer || "no answer";
      const long = full.length > ANSWER_CHARS;
      under.push(el("p", { class: "rqa" }, `NV: ${long ? `${full.slice(0, ANSWER_CHARS)}…` : full}`, long ? [" ", link("Read in full", onOpenReports)] : null, answer.differs ? el("span", { class: "rdiffers" }, `${!long && String(answer.answer ?? "").endsWith(".") ? " " : ". "}NV's answer differs from its report`) : null));
    }
    return el("div", { class: "rq", "data-q": q.id }, el("button", { type: "button", class: "rchip", "data-query": q.id, disabled: inFlight, title: "Ask NV-Reason-CT this question", onclick: (e) => (e.stopPropagation(), ask(q)) }, shown(q.text)), ...under);
  }

  function askSection(view, noChat) {
    const items = allQuestions(view);
    return section(
      "Ask NV-Reason-CT",
      items.length ? el("div", { class: "rqs" }, items.map(question)) : el("p", { class: "muted" }, noQuestionsText(view)),
      state.askError ? el("p", { class: "rretry" }, state.askError) : null,
      noChat ? null : el("p", { class: "muted small" }, "Or ask the chat on the right to put your own question to NV-Reason-CT."),
    );
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

  function cell(reader, value, row) {
    const text = reader === "radar" ? (value === "not_scored" ? "—" : pct(row.radar.max_prob, 0)) : WORD[value];
    return el("span", { class: "rcell", "data-reader": reader, "data-state": value, title: `${NAMES[reader]}: ${STATE_TEXT[value] || "not read"}` }, `${NAMES[reader]} ${text}`);
  }

  function rowNodes({ row, cells, flag }) {
    const open = state.open.has(row.id);
    const shownReaders = readers();
    const slots = ["nv", "report", "radar"].filter((r) => shownReaders.includes(r)).map((r) => cell(r, cells[r], row));
    const node = el(
      "div",
      {
        class: `rrow${flag ? " flagged" : ""}`,
        "data-row": row.id,
        title: row.organ ? `Jump to the ${row.organ.toLowerCase()}` : null,
        onclick: () => {
          if (row.organ) onOrgan(row.organ);
          toggleRow(row.id);
        },
      },
      el("button", { type: "button", class: "rexpand", "aria-expanded": String(open), "aria-label": `${open ? "Hide" : "Show"} details for ${row.title}`, onclick: (e) => (e.stopPropagation(), toggleRow(row.id)) }, open ? "▾" : "▸"),
      el("span", { class: "rtitle" }, row.title),
      slots,
    );
    return open ? [node, details(row)] : [node];
  }

  function group(key, title, items) {
    if (!items.length) return null;
    return el("section", { class: "rgroup", "data-group": key }, el("h3", { class: "rgroup-head" }, title, " ", el("span", { class: "rcount" }, String(items.length))), ...items.flatMap(rowNodes));
  }

  function outsideGroup(items) {
    if (!items.length) return null;
    return el(
      "section",
      { class: "rgroup", "data-group": "outside" },
      el("h3", { class: "rgroup-head" }, "NV could not see ", el("span", { class: "rcount" }, String(items.length))),
      ...items.map(({ row }) => {
        const claim = row.nv.state === "present" || row.nv.state === "absent"; // silence is not something NV said
        const said = !claim ? null : row.nv.quote ? `NV said ${STATE_TEXT[row.nv.state]}, "${row.nv.quote}"` : `NV said ${STATE_TEXT[row.nv.state]}`;
        return el("p", { class: "rout", "data-out": row.id }, el("span", { class: "rtitle" }, row.title), el("span", { class: "muted small" }, ` · ${Math.round(row.nv.coverage * 100)}% inside NV's view${said ? ` · ${said}` : ""}`));
      }),
    );
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

  function render() {
    if (state.error) return root.replaceChildren(el("p", { class: "muted" }, state.error));
    const view = state.view;
    if (!view) return root.replaceChildren(el("p", { class: "muted" }, "Loading the readers…"));
    const noChat = [view.report?.status, view.nv?.read?.status, view.queries?.status].includes("no_llm");
    const g = overview(view.rows, { threshold: state.threshold, nvPending: nvPendingOf(view), reportPending: view.has_report && view.report.status === "pending" });
    const parts = [
      headBlock(view),
      el("p", { class: "muted small rline-note", "data-part": "display-line" }, `RADAR counts as present at or above the display line, now ${state.threshold}%. Set it on the RADAR findings tab.`),
      group("disagree", "Disagree", g.disagree),
      group("nv-found", "NV found", g.nvFound),
      group("others-found", "Found by the others", g.othersFound),
      group("waiting", "Waiting for a reader", g.waiting),
      outsideGroup(g.outside),
      ...fold(g.quiet),
      askSection(view, noChat),
      ...others(view),
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
      render();
    },
    get view() {
      return state.view;
    },
  };
}
