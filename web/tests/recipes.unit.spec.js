// Unit tests for src/recipes.js and its mirror in the chat prompt. Playwright runs these in Node; no browser is opened.
import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { windowOf, WINDOWS } from "../src/light.js";
import { ORGAN_COLOURS } from "../src/palette.js";
import { NOTES, RECIPES, recipeFor, WHY } from "../src/recipes.js";

// A small catalog and the same lookup rules as findings.finding().
const CATALOG = [
  { key: "主动脉_主动脉瘤", english: "Aorta_Aortic aneurysm", organ: "Aorta", finding: "Aortic aneurysm" },
  { key: "肝脏_肝囊肿", english: "Liver_Cyst", organ: "Liver", finding: "Cyst" },
  { key: "肾脏_肾囊肿", english: "Kidney_Cyst", organ: "Kidney", finding: "Cyst" },
  { key: "胰腺_胰腺炎", english: "Pancreas_Pancreatitis", organ: "Pancreas", finding: "Pancreatitis" },
];
function lookup(keyOrName) {
  const q = String(keyOrName).trim().toLowerCase();
  const exact = CATALOG.find((f) => f.key === keyOrName || f.english.toLowerCase() === q);
  if (exact) return exact;
  const byName = CATALOG.filter((f) => f.finding.toLowerCase() === q);
  return byName.length === 1 ? byName[0] : null;
}

test("every scored organ has a recipe whose preset is a window", () => {
  expect(Object.keys(RECIPES).sort()).toEqual(Object.keys(ORGAN_COLOURS).sort());
  for (const [organ, r] of Object.entries(RECIPES)) {
    expect(WINDOWS[r.preset], organ).toBeTruthy();
    expect(WHY[r.preset], organ).toBeTruthy();
    expect(r.zoom).toBeGreaterThanOrEqual(1);
  }
  expect(Object.keys(NOTES)).toEqual(["Pancreas"]);
});

test("recipeFor by organ, lower-case organ, key, Organ_Finding and finding name", () => {
  const aorta = recipeFor("Aorta", lookup);
  expect(aorta).toEqual({
    organ: "Aorta",
    finding: null,
    key: null,
    preset: "angio",
    window: { width: 600, level: 200 },
    zoom: 2.5,
    why: WHY.angio,
    note: null,
  });
  expect(recipeFor("adrenal gland", lookup)).toMatchObject({ organ: "Adrenal gland", preset: "soft_tissue", zoom: 3 });
  expect(recipeFor("主动脉_主动脉瘤", lookup)).toMatchObject({ organ: "Aorta", key: "主动脉_主动脉瘤", finding: "Aortic aneurysm", preset: "angio" });
  expect(recipeFor("Liver_Cyst", lookup)).toMatchObject({ organ: "Liver", key: "肝脏_肝囊肿", finding: "Cyst", preset: "liver", zoom: 1.5 });
  expect(recipeFor("pancreatitis", lookup)).toMatchObject({ organ: "Pancreas", preset: "soft_tissue", zoom: 2.5, note: NOTES.Pancreas });
});

test("recipeFor's window is the preset's window", () => {
  for (const organ of Object.keys(RECIPES)) {
    const r = recipeFor(organ, lookup);
    const w = WINDOWS[r.preset];
    expect(r.window).toEqual(windowOf(w.min, w.max));
  }
});

test("an unknown target throws with the organ list", () => {
  expect(() => recipeFor("brain", lookup)).toThrow(/Unknown organ or finding: brain/);
  expect(() => recipeFor("brain", lookup)).toThrow(new RegExp(Object.keys(RECIPES).join(", ")));
  // A finding name shared by two organs is not a unique target.
  expect(() => recipeFor("Cyst", lookup)).toThrow(/Unknown organ or finding/);
});

test("viewing recipes match the chat prompt's table", () => {
  const prompt = fs.readFileSync(path.join(import.meta.dirname, "..", "..", "src", "radar_desk", "chat", "prompt.py"), "utf8");
  const start = prompt.indexOf('VIEW_RECIPES = """');
  expect(start).toBeGreaterThan(-1);
  const block = prompt.slice(start, prompt.indexOf('"""', start + 18));
  const lines = block.split("\n");

  const windowsLine = lines.find((l) => l.startsWith("Windows: "));
  const windows = [...windowsLine.matchAll(/(\w+) W (-?\d+) L (-?\d+)/g)].map((m) => [m[1], Number(m[2]), Number(m[3])]);
  expect(windows).toEqual(Object.entries(WINDOWS).map(([name, w]) => [name, windowOf(w.min, w.max).width, windowOf(w.min, w.max).level]));

  const rows = lines
    .filter((l) => l.startsWith("- "))
    .map((l) => {
      const m = l.match(/^- ([^:]+): (\w+)\. ([^(]+?)(?: \((.+)\))?$/);
      expect(m, l).toBeTruthy();
      return { organ: m[1], preset: m[2], why: m[3], note: m[4] ?? null };
    });
  expect(rows).toEqual(Object.entries(RECIPES).map(([organ, r]) => ({ organ, preset: r.preset, why: WHY[r.preset], note: NOTES[organ] ?? null })));
});
