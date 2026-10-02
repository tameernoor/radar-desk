// Unit tests for src/light.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import {
  clampWindow,
  COLORMAPS,
  LIGHT_DEFAULTS,
  lightReducer,
  nextZoom,
  presetFor,
  rangeOf,
  sliderFromWidth,
  widthFromSlider,
  windowOf,
  WINDOWS,
  zoomAround,
} from "../src/light.js";

test("windowOf and rangeOf round trip", () => {
  for (const w of Object.values(WINDOWS)) {
    const { width, level } = windowOf(w.min, w.max);
    expect(rangeOf(width, level)).toEqual({ min: w.min, max: w.max });
  }
  expect(windowOf(-160, 240)).toEqual({ width: 400, level: 40 });
  expect(rangeOf(1, 0)).toEqual({ min: -0.5, max: 0.5 });
});

test("the presets take Radiopaedia's windows, angio included", () => {
  const wl = Object.fromEntries(Object.entries(WINDOWS).map(([k, w]) => [k, windowOf(w.min, w.max)]));
  expect(wl).toEqual({
    soft_tissue: { width: 400, level: 50 },
    liver: { width: 150, level: 30 },
    bone: { width: 1800, level: 400 },
    lung: { width: 1500, level: -600 },
    angio: { width: 600, level: 200 },
  });
  expect(WINDOWS.angio).toEqual({ label: "Angio", min: -100, max: 500 });
});

test("the width slider maps 0 to 1000 onto 1 to 4000 HU, log-scaled", () => {
  expect(widthFromSlider(0)).toBe(1);
  expect(widthFromSlider(1000)).toBe(4000);
  expect(sliderFromWidth(1)).toBe(0);
  expect(sliderFromWidth(4000)).toBe(1000);
  for (const w of [1, 400, 4000]) {
    // Back within one slider step, which is under 1% of the width on a log scale.
    const back = widthFromSlider(sliderFromWidth(w));
    expect(Math.abs(back - w)).toBeLessThanOrEqual(Math.max(1, w * 0.01));
  }
  let last = 0;
  for (let s = 0; s <= 1000; s++) {
    const w = widthFromSlider(s);
    expect(w).toBeGreaterThanOrEqual(last);
    last = w;
  }
  expect(sliderFromWidth(400)).toBeLessThan(sliderFromWidth(1500));
  expect(widthFromSlider(-5)).toBe(1);
  expect(widthFromSlider(5000)).toBe(4000);
});

test("presetFor names every preset and calls anything else custom", () => {
  for (const [name, w] of Object.entries(WINDOWS)) expect(presetFor(w.min, w.max)).toBe(name);
  expect(presetFor(-150, 251)).toBe("custom");
  expect(presetFor(0, 100)).toBe("custom");
});

test("clampWindow keeps width in 1 to 4000 and level in -1200 to 2000", () => {
  expect(clampWindow({ width: 0, level: 0 })).toEqual({ width: 1, level: 0 });
  expect(clampWindow({ width: 9000, level: 5000 })).toEqual({ width: 4000, level: 2000 });
  expect(clampWindow({ width: 400, level: -3000 })).toEqual({ width: 400, level: -1200 });
  expect(clampWindow({ width: 400, level: 40 })).toEqual({ width: 400, level: 40 });
});

test("reducer: preset, window, gamma, invert, colormap, reset", () => {
  let s = { ...LIGHT_DEFAULTS };
  s = lightReducer(s, { type: "preset", name: "liver" });
  expect([s.min, s.max]).toEqual([-45, 105]);
  expect(lightReducer(s, { type: "preset", name: "angio" })).toMatchObject({ min: -100, max: 500 });
  s = lightReducer(s, { type: "window", min: -100, max: 300 });
  expect([s.min, s.max]).toEqual([-100, 300]);
  s = lightReducer(s, { type: "window", min: -5000, max: 5000 }); // width clamped to 4000 around level 0
  expect([s.min, s.max]).toEqual([-2000, 2000]);
  s = lightReducer(s, { type: "gamma", value: 1.5 });
  expect(s.gamma).toBe(1.5);
  expect(lightReducer(s, { type: "gamma", value: 10 }).gamma).toBe(3);
  expect(lightReducer(s, { type: "gamma", value: 0 }).gamma).toBe(0.2);
  s = lightReducer(s, { type: "invert", value: true });
  expect(s.invert).toBe(true);
  for (const name of COLORMAPS) expect(lightReducer(s, { type: "colormap", name }).colormap).toBe(name);
  s = lightReducer(s, { type: "colormap", name: "hot" });
  expect(lightReducer(s, { type: "reset" })).toEqual(LIGHT_DEFAULTS);
  expect(LIGHT_DEFAULTS).toEqual({ min: -150, max: 250, gamma: 1, invert: false, colormap: "gray" });
});

test("reducer throws on an unknown preset, colour map or action", () => {
  expect(() => lightReducer(LIGHT_DEFAULTS, { type: "preset", name: "brain" })).toThrow(/Unknown window preset/);
  expect(() => lightReducer(LIGHT_DEFAULTS, { type: "colormap", name: "jet" })).toThrow(/Unknown colour map/);
  expect(() => lightReducer(LIGHT_DEFAULTS, { type: "window", min: NaN, max: 1 })).toThrow();
  expect(() => lightReducer(LIGHT_DEFAULTS, { type: "nope" })).toThrow(/Unknown light action/);
});

test("nextZoom steps by 1.25, rounds to two decimals and clamps", () => {
  expect(nextZoom(1, 1)).toBe(1.25);
  expect(nextZoom(1.25, 1)).toBe(1.56);
  expect(nextZoom(1, -1)).toBe(0.8);
  expect(nextZoom(1.25, -1)).toBe(1);
  expect(nextZoom(14, 1)).toBe(16);
  expect(nextZoom(16, 1)).toBe(16);
  expect(nextZoom(0.3, -1)).toBe(0.25);
  expect(nextZoom(0.25, -1)).toBe(0.25);
});

test("zoomAround keeps the crosshair in place", () => {
  const zoomed = zoomAround([0, 0, 0, 1], 2, [10, 20, 30]);
  expect(zoomed).toEqual([-10, -20, -30, 2]);
  expect(zoomAround(zoomed, 1, [10, 20, 30])).toEqual([0, 0, 0, 1]);
});
