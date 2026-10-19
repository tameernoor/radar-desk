// Unit tests for src/readers.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import { cellsFor, disagrees, foldText, isBusy, nvCell, nvPendingOf, overview, radarCell, reportCell, WORD } from "../src/readers.js";

const row = (id, { report = null, radar = { scored: true, max_prob: 0.3 }, nv = null } = {}) => ({
  id,
  title: id,
  organ: "Liver",
  report,
  radar: { findings: [], ...radar },
  nv,
  asked: [],
});

test("RADAR's cell follows the display line", () => {
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
  expect(["present", "absent", "not_mentioned", "outside", "pending", "none"].map((k) => WORD[k])).toEqual(["present", "absent", "silent", "can't see", "…", "—"]);
});

test("A flag only for present against absent", () => {
  expect(disagrees({ report: "present", radar: "absent", nv: "none" })).toBe(true);
  expect(disagrees({ report: "present", radar: "not_scored", nv: "not_mentioned" })).toBe(false);
  expect(disagrees({ report: "present", radar: "present", nv: "outside" })).toBe(false);
  expect(disagrees({ report: "absent", radar: "absent", nv: "absent" })).toBe(false);
});

test("an NV read outside its view never makes a flag", () => {
  const r = row("a", { report: { state: "absent" }, radar: { scored: false, max_prob: null }, nv: { state: "present", in_view: false } });
  expect(overview([r], { threshold: 50 }).disagree).toEqual([]);
  expect(cellsFor(r, { threshold: 50 })).toEqual({ report: "absent", radar: "not_scored", nv: "outside" });
});

test("overview groups: disagree, NV found, found by the others, outside, quiet", () => {
  const rows = [
    row("quiet", { radar: { scored: true, max_prob: 0.1 } }),
    row("others", { radar: { scored: true, max_prob: 0.9 } }),
    row("flag", { report: { state: "absent" }, radar: { scored: true, max_prob: 0.9 } }),
    row("silent", { radar: { scored: false, max_prob: null }, nv: { state: "not_mentioned", in_view: true } }),
    row("nv", { radar: { scored: false, max_prob: null }, nv: { state: "present", in_view: true } }),
    row("out", { report: { state: "present" }, radar: { scored: false, max_prob: null }, nv: { state: "present", in_view: false, coverage: 0.38 } }),
  ];
  const ids = (xs) => xs.map((x) => x.row.id);
  const at50 = overview(rows, { threshold: 50 });
  expect(ids(at50.disagree)).toEqual(["flag"]);
  expect(ids(at50.nvFound)).toEqual(["nv"]);
  expect(ids(at50.othersFound)).toEqual(["others", "out"]);
  expect(ids(at50.outside)).toEqual(["out"]);
  expect(ids(at50.quiet)).toEqual(["quiet", "silent"]);
  expect(at50.waiting).toEqual([]);
  const at95 = overview(rows, { threshold: 95 });
  expect(at95.disagree).toEqual([]);
  expect(ids(at95.quiet)).toEqual(["quiet", "others", "flag", "silent"]);
  expect(ids(at95.nvFound)).toEqual(["nv"]);
  expect(foldText(15)).toBe("15 findings nobody reports");
  expect(foldText(1)).toBe("1 finding nobody reports");
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

test("a row waiting on a reader is never quiet", () => {
  const rows = [
    row("wait", { report: { state: "absent" }, radar: { scored: true, max_prob: 0.1 } }),
    row("pos", { radar: { scored: true, max_prob: 0.9 } }),
  ];
  for (const opts of [{ threshold: 50, nvPending: true }, { threshold: 50, reportPending: true }]) {
    const out = overview(rows, opts);
    expect(out.quiet).toEqual([]);
    expect(out.othersFound.map((x) => x.row.id)).toEqual(["pos"]);
    expect(out.waiting.map((x) => x.row.id)).toEqual(["wait"]);
  }
});

test("pending and outside never make a flag", () => {
  expect(disagrees({ report: "pending", radar: "absent", nv: "none" })).toBe(false);
  expect(disagrees({ report: "outside", radar: "absent", nv: "none" })).toBe(false);
});

test("no rows gives empty buckets", () => {
  expect(overview([], { threshold: 50 })).toEqual({ disagree: [], nvFound: [], othersFound: [], waiting: [], outside: [], quiet: [] });
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
  const { allQuestions, answerFor, noQuestionsText } = await import("../src/readers.js");
  const g = { id: "q_g", text: "General?" };
  const a = { id: "q_a", text: "A?" };
  const view = { queries: { general: [g] }, rows: [{ queries: [a, g] }, { queries: [a] }, { queries: undefined }], nv: { answers: [{ question: "A?", answer: "No." }, { question: "A?", answer: "Yes." }] } };
  expect(allQuestions(view).map((q) => q.id)).toEqual(["q_g", "q_a"]);
  expect(answerFor(view, a).answer).toBe("Yes.");
  expect(answerFor(view, g)).toBeNull();
  expect(noQuestionsText({ nv: { job: null, unavailable: "needs Modal compute" } })).toContain("NV-Reason-CT: needs Modal compute");
  expect(noQuestionsText({ nv: { job: { state: "queued" } } })).toContain("once NV-Reason-CT has read");
});

test("shown hides the yes/no suffix and nothing else", async () => {
  const { shown, YES_NO_SUFFIX } = await import("../src/readers.js");
  expect(shown(`Is there fluid? ${YES_NO_SUFFIX}`)).toBe("Is there fluid?");
  expect(shown("Where is the fluid?")).toBe("Where is the fluid?");
});
