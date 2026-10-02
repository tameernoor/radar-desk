// Viewing recipes: the window and zoom that show each scored organ the way it is usually read. Pure.
// The windows are the cited ones in light.js. The why lines and the pancreas note mirror VIEW_RECIPES
// in src/radar_desk/chat/prompt.py; keep the two identical (a unit test compares them).
import { windowOf, WINDOWS } from "./light.js";

// zoom is the magnification about the organ centroid; 1 means the whole slice fits.
export const RECIPES = {
  Liver: { preset: "liver", zoom: 1.5 },
  Pancreas: { preset: "soft_tissue", zoom: 2.5 },
  Kidney: { preset: "soft_tissue", zoom: 2 },
  Gallbladder: { preset: "soft_tissue", zoom: 2.5 },
  Spleen: { preset: "soft_tissue", zoom: 2 },
  "Adrenal gland": { preset: "soft_tissue", zoom: 3 },
  Stomach: { preset: "soft_tissue", zoom: 1.5 },
  Duodenum: { preset: "soft_tissue", zoom: 2.5 },
  "Small bowel": { preset: "soft_tissue", zoom: 1.5 },
  "Large bowel": { preset: "soft_tissue", zoom: 1.25 },
  Bladder: { preset: "soft_tissue", zoom: 2 },
  Esophagus: { preset: "soft_tissue", zoom: 2.5 },
  Aorta: { preset: "angio", zoom: 2.5 },
  "Portal vein": { preset: "angio", zoom: 2.5 },
  Heart: { preset: "angio", zoom: 1.5 },
  Lung: { preset: "lung", zoom: 1.25 },
  Rib: { preset: "bone", zoom: 1.5 },
  Sacrum: { preset: "bone", zoom: 2 },
};

export const WHY = {
  soft_tissue: "abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels",
  liver: "narrow liver window; small density differences inside the parenchyma show",
  angio: "vascular window; the enhanced lumen, the wall and calcification stay apart instead of saturating",
  bone: "wide bone window; cortex and marrow show instead of saturating white",
  lung: "wide lung window centred on air; parenchyma and nodules show at the lung bases",
};

export const NOTES = {
  Pancreas: "the source gives no pancreas window; the abdomen soft tissue one is used",
};

// target: a scored organ (any case) or a finding, resolved by findingLookup (key, Organ_Finding or name).
export function recipeFor(target, findingLookup) {
  const q = String(target ?? "").trim().toLowerCase();
  const organ = Object.keys(RECIPES).find((o) => o.toLowerCase() === q);
  const f = organ ? null : findingLookup?.(String(target ?? "").trim());
  const name = organ ?? f?.organ;
  if (!RECIPES[name]) throw new Error(`Unknown organ or finding: ${target}. Use a finding or one of the scored organs: ${Object.keys(RECIPES).join(", ")}.`);
  const { preset, zoom } = RECIPES[name];
  const w = WINDOWS[preset];
  return { organ: name, finding: f?.finding ?? null, key: f?.key ?? null, preset, window: windowOf(w.min, w.max), zoom, why: WHY[preset], note: NOTES[name] ?? null };
}
