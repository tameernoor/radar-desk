// Unit tests for src/lookcard.js and src/hu.js. Playwright runs these in Node; no browser is opened.
import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { HU_CAVEAT, HU_TABLE, huGuess } from "../src/hu.js";
import { buildLookCard, LOOK_FOOTER } from "../src/lookcard.js";

const f = (organ, finding, prob) => ({ key: `${organ}_${finding}`, organ, finding, prob });
const FINDINGS = [
  f("Liver", "Cyst", 0.2),
  f("Liver", "Mass", 0.9),
  f("Liver", "Fatty liver", 0.55),
  f("Liver", "Calcification", 0.05),
  f("Kidney", "Stone", 0.3),
  f("Kidney", "Cyst", null),
  f("Sacrum", "Fracture", null),
];
const CATALOG = FINDINGS.map(({ prob, ...rest }) => rest);

function view(over = {}) {
  return {
    plane: "axial",
    threshold_pct: 50,
    slice: { axis: "axial", index: 15, number: 16, count: 24 },
    organs_on_slice: [
      { organ: "Liver", label: 21, pixels: 60, percent_of_mask: 60, scored: true },
      { organ: "Kidney", label: 20, pixels: 30, percent_of_mask: 30, scored: true },
      { organ: "Erector spinae muscle", label: 3, pixels: 10, percent_of_mask: 10, scored: false },
    ],
    crosshair: { mm: [1, 2, 3], hu: 40, label: 21, organ: "Liver" },
    nearest_organ: null,
    ...over,
  };
}

test("organs largest first, at most three findings each, positives first", () => {
  const card = buildLookCard(view(), FINDINGS, CATALOG);
  expect(card.plane).toBe("Axial slice 16 of 24");
  expect(card.organs.map((o) => o.organ)).toEqual(["Liver", "Kidney", "Erector spinae muscle"]);
  const liver = card.organs[0];
  expect(liver.percent).toBe(60);
  expect(liver.findings.map((x) => x.finding)).toEqual(["Mass", "Fatty liver", "Cyst"]);
  expect(liver.findings.map((x) => x.positive)).toEqual([true, true, false]);
  expect(liver.findings[0].score_text).toBe("90.0%");
  expect(card.organs[2]).toMatchObject({ scored: false, findings: [] });
  expect(card.point.where).toBe("The crosshair is inside RADAR's Liver outline.");
  expect(card.footer).toBe(LOOK_FOOTER);
});

test("an organ RADAR did not score in this run gets a note, not no-score rows", () => {
  const card = buildLookCard(view({ organs_on_slice: [{ organ: "Sacrum", label: 27, pixels: 5, percent_of_mask: 100, scored: true }] }), FINDINGS, CATALOG);
  expect(card.organs[0].findings).toEqual([]);
  expect(card.organs[0].note).toBe("RADAR did not score this organ in this run.");
  // An organ with some scores lists the scored ones only.
  const kidney = buildLookCard(view(), FINDINGS, CATALOG).organs[1];
  expect(kidney.findings.map((x) => x.finding)).toEqual(["Stone"]);
  expect(kidney.note).toBeNull();
  // Without a result, catalog findings are listed with no score.
  const bare = buildLookCard(view(), [], CATALOG);
  expect(bare.organs[0].findings.every((x) => x.score_text === "no score")).toBe(true);
  expect(bare.organs[0].note).toBe("No result loaded yet.");
});

test("background crosshair names the nearest outline", () => {
  const card = buildLookCard(
    view({ crosshair: { mm: [0, 0, 0], hu: -100, label: 0, organ: null }, nearest_organ: { organ: "Kidney", label: 20, distance_mm: 2.8 } }),
    FINDINGS,
    CATALOG,
  );
  expect(card.point.where).toBe("The crosshair is on background. The nearest outline is Kidney, 2.8 mm away on this slice.");
  expect(card.point.hu).toBe(`-100 HU, in the Fat range (${HU_CAVEAT}).`);
  const empty = buildLookCard(view({ organs_on_slice: [], crosshair: { mm: [0, 0, 0], hu: 0, label: 0, organ: null } }), FINDINGS, CATALOG);
  expect(empty.point.where).toBe("The crosshair is on background, and nothing is outlined on this slice.");
  expect(empty.note).toBe("RADAR outlined nothing on this slice.");
});

test("HU guess rows", () => {
  expect(huGuess(-1000).rows).toEqual(["Air"]);
  expect(huGuess(-100).rows).toEqual(["Fat"]);
  expect(huGuess(0).rows).toEqual(["Water"]);
  expect(huGuess(350).rows).toEqual(["Cancellous bone"]);
  expect(huGuess(1200).rows).toEqual(["Cortical bone"]);
  expect(huGuess(40).rows).toEqual(["Blood", "Soft tissue, unenhanced"]);
  expect(huGuess(40).text).toBe(`40 HU, in the "Blood" or "Soft tissue, unenhanced" ranges (${HU_CAVEAT})`);
  expect(huGuess(-300)).toMatchObject({ rows: [], between: ["Lung parenchyma", "Fat"] });
  expect(huGuess(450).text).toBe(`450 HU, between Cancellous bone and Cortical bone (${HU_CAVEAT})`);
  expect(huGuess(3000).between).toEqual(["Cortical bone", null]);
  expect(huGuess(null)).toBeNull();
});

test("nothing loaded, and CT without a mask", () => {
  const nothing = buildLookCard({ crosshair: null, slice: null }, [], CATALOG);
  expect(nothing).toMatchObject({ empty: true, message: "No CT is on screen.", footer: LOOK_FOOTER });
  const noMask = buildLookCard(view({ slice: null, organs_on_slice: null, crosshair: { mm: [0, 0, 0], hu: 70, label: null, organ: null } }), [], CATALOG);
  expect(noMask.plane).toBe("Axial view");
  expect(
    buildLookCard(view({ slice_type: "multiplanar", plane: "coronal", slice: null, crosshair: { mm: [0, 0, 0], hu: 70, label: null, organ: null } }), [], CATALOG).plane,
  ).toBe("Coronal view, main view of three");
  expect(noMask.organs).toEqual([]);
  expect(noMask.note).toBe("The CT is loaded, but RADAR's outlines are not there yet. The HU value under the crosshair is below.");
  expect(noMask.point.where).toBeNull();
  expect(noMask.point.hu).toBe(`70 HU, in the Blood range (${HU_CAVEAT}).`);
});

test("multiplanar names the main view's slice", () => {
  const card = buildLookCard(view({ slice_type: "multiplanar", plane: "sagittal", slice: { axis: "sagittal", index: 3, number: 4, count: 64 } }), FINDINGS, CATALOG);
  expect(card.plane).toBe("Sagittal slice 4 of 64, main view of three");
  expect(buildLookCard(view({ plane: "render" }), FINDINGS, CATALOG).plane).toBe("3D view, axial slice 16 of 24");
});

test("HU table matches the chat prompt's table", () => {
  const prompt = fs.readFileSync(path.join(import.meta.dirname, "..", "..", "src", "radar_desk", "chat", "prompt.py"), "utf8");
  const block = prompt.slice(prompt.indexOf('HU_TABLE = """'), prompt.indexOf('"""', prompt.indexOf('HU_TABLE = """') + 14));
  const rows = block.split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2));
  expect(rows).toEqual(HU_TABLE.map((r) => `${r.name}: ${r.text}`));
});
