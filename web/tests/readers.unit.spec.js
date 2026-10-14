// Unit tests for src/readers.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import { arrange, cellsFor, disagrees, foldText, isBusy, KEY, nvCell, nvPendingOf, radarCell, reportCell, STATE_TEXT, SYMBOL } from "../src/readers.js";

const row = (id, { report = null, radar = { scored: true, max_prob: 0.3 }, nv = null } = {}) => ({
  id,
  title: id,
  organ: "Liver",
  report,
  radar: { findings: [], ...radar },
  nv,
  asked: [],
});

test("RADAR's dot follows the display line", () => {
  const r = row("a", { radar: { scored: true, max_prob: 0.62 } });
  expect(radarCell(r, 50)).toBe("present");
  expect(radarCell(r, 62)).toBe("present");
  expect(radarCell(r, 63)).toBe("absent");
  expect(radarCell(row("b", { radar: { scored: false, max_prob: null } }), 0)).toBe("not_scored");
});

test("report and NV cells", () => {
  expect(reportCell(row("a", { report: { state: "absent", quote: "x" } }), false)).toBe("absent");
  expect(reportCell(row("a"), false)).toBe("none");
  expect(reportCell(row("a"), true)).toBe("pending");
  const outside = row("b", { nv: { state: "present", quote: "x", in_view: false, coverage: 0.2 } });
  expect(nvCell(outside, false)).toBe("outside");
  expect(nvCell(outside, true)).toBe("pending");
  expect(nvCell(row("c", { nv: { state: "absent", in_view: null } }), false)).toBe("absent");
  expect(nvCell(row("d"), false)).toBe("none");
  expect(SYMBOL.outside).toBe("◌");
  expect(KEY.map((k) => `${SYMBOL[k]} ${STATE_TEXT[k]}`)).toEqual(["● present", "○ absent", "– not mentioned", "◌ outside NV's view", "· not scored"]);
});

test("⚑ only for present against absent", () => {
  expect(disagrees({ report: "present", radar: "absent", nv: "none" })).toBe(true);
  expect(disagrees({ report: "present", radar: "not_scored", nv: "not_mentioned" })).toBe(false);
  expect(disagrees({ report: "present", radar: "present", nv: "outside" })).toBe(false);
  expect(disagrees({ report: "absent", radar: "absent", nv: "absent" })).toBe(false);
});

test("an NV read outside its view never makes a flag", () => {
  const r = row("a", { report: { state: "absent" }, radar: { scored: false, max_prob: null }, nv: { state: "present", in_view: false } });
  expect(arrange([r], { threshold: 50 }).flagged).toEqual([]);
  expect(cellsFor(r, { threshold: 50 })).toEqual({ report: "absent", radar: "not_scored", nv: "outside" });
});

test("order: flags, then any present, then the fold", () => {
  const rows = [
    row("quiet", { radar: { scored: true, max_prob: 0.1 } }),
    row("pos", { radar: { scored: true, max_prob: 0.9 } }),
    row("flag", { report: { state: "absent" }, radar: { scored: true, max_prob: 0.9 } }),
    row("silent", { radar: { scored: false, max_prob: null }, nv: { state: "not_mentioned", in_view: true } }),
  ];
  const at50 = arrange(rows, { threshold: 50 });
  expect(at50.flagged.map((x) => x.row.id)).toEqual(["flag"]);
  expect(at50.positive.map((x) => x.row.id)).toEqual(["pos"]);
  expect(at50.folded.map((x) => x.row.id)).toEqual(["quiet", "silent"]);
  const at95 = arrange(rows, { threshold: 95 });
  expect(at95.flagged).toEqual([]);
  expect(at95.folded.map((x) => x.row.id)).toEqual(["quiet", "pos", "flag", "silent"]);
  expect(foldText(15)).toBe("15 rows: nobody reports it");
  expect(foldText(1)).toBe("1 row: nobody reports it");
});

test("busy while NV runs or something is being read", () => {
  const base = { report: { status: "ready" }, nv: { job: { state: "done" }, read: { status: "ready" } } };
  expect(isBusy(base)).toBe(false);
  expect(nvPendingOf({ ...base, nv: { job: { state: "submitted" }, read: { status: "none" } } })).toBe(true);
  expect(isBusy({ ...base, nv: { job: { state: "queued" }, read: { status: "none" } } })).toBe(true);
  expect(isBusy({ ...base, report: { status: "pending" } })).toBe(true);
  expect(isBusy({ ...base, queries: { status: "pending" } })).toBe(true);
  expect(isBusy({ ...base, nv: { ...base.nv, asking: ["q_1"] } })).toBe(true);
  expect(isBusy({ ...base, nv: { job: null, read: { status: "none" } } })).toBe(false);
});

test("a row waiting on a reader is never folded", () => {
  const rows = [
    row("wait", { report: { state: "absent" }, radar: { scored: true, max_prob: 0.1 } }),
    row("pos", { radar: { scored: true, max_prob: 0.9 } }),
  ];
  for (const opts of [{ threshold: 50, nvPending: true }, { threshold: 50, reportPending: true }]) {
    const out = arrange(rows, opts);
    expect(out.folded.map((x) => x.row.id)).toEqual([]);
    expect(out.positive.map((x) => x.row.id)).toEqual(["pos"]);
    expect(out.waiting.map((x) => x.row.id)).toEqual(["wait"]);
  }
});

test("pending and outside never make a flag", () => {
  expect(disagrees({ report: "pending", radar: "absent", nv: "none" })).toBe(false);
  expect(disagrees({ report: "outside", radar: "absent", nv: "none" })).toBe(false);
});

test("no rows gives empty buckets", () => {
  expect(arrange([], { threshold: 50 })).toEqual({ flagged: [], positive: [], waiting: [], folded: [] });
});

test("splitReport takes the text after IMPRESSION:", async () => {
  const { splitReport } = await import("../src/readers.js");
  expect(splitReport("FINDINGS:\nLiver: ok.\n\nIMPRESSION:\n1. A.\n2. B.")).toEqual({ findings: "FINDINGS:\nLiver: ok.", impression: "1. A.\n2. B." });
  expect(splitReport("Just text.")).toEqual({ findings: "", impression: "Just text." });
  expect(splitReport("Findings.\nImpression:\n1. A.")).toEqual({ findings: "Findings.", impression: "1. A." });
  expect(splitReport("Findings.\nIMPRESSIONS:\n1. A.").impression).toBe("1. A.");
  expect(splitReport("Findings.\n**IMPRESSION:**\n1. A.")).toEqual({ findings: "Findings.", impression: "1. A." });
  expect(splitReport("Findings.\n**IMPRESSION**: 1. A.").impression).toBe("1. A.");
  expect(splitReport("A cyst giving the impression of a mass.")).toEqual({ findings: "", impression: "A cyst giving the impression of a mass." });
  expect(splitReport(null)).toEqual({ findings: "", impression: "" });
});

test("allQuestions de-duplicates by id and answerFor takes the latest answer", async () => {
  const { allQuestions, answerFor, clipLines, noQuestionsText } = await import("../src/readers.js");
  const g = { id: "q_g", text: "General?" };
  const a = { id: "q_a", text: "A?" };
  const view = { queries: { general: [g] }, rows: [{ queries: [a, g] }, { queries: [a] }, { queries: undefined }], nv: { answers: [{ question: "A?", answer: "No." }, { question: "A?", answer: "Yes." }] } };
  expect(allQuestions(view).map((q) => q.id)).toEqual(["q_g", "q_a"]);
  expect(answerFor(view, a).answer).toBe("Yes.");
  expect(answerFor(view, g)).toBeNull();
  expect(clipLines("a\nb\nc", 2)).toEqual({ text: "a\nb", cut: true });
  expect(clipLines("a\nb", 2)).toEqual({ text: "a\nb", cut: false });
  expect(noQuestionsText({ nv: { job: null, unavailable: "needs Modal compute" } })).toContain("NV-Reason-CT: needs Modal compute");
  expect(noQuestionsText({ nv: { job: { state: "queued" } } })).toContain("once NV-Reason-CT has read");
});
