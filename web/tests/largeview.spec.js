// Large view in the browser: focus mode, the light panel, zoom and pan.
import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { sliderFromWidth, widthFromSlider, zoomAround } from "../src/light.js";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";
const SHOTS = process.env.RADAR_SHOTS || path.join(import.meta.dirname, "..", ".smoke-out");

// Calls a registered page tool the way the chat does, through document.modelContext.
function callTool(page, name, args = {}) {
  return page.evaluate(
    async ({ name, args }) => {
      const mc = document.modelContext;
      const tool = (await mc.getTools()).find((t) => t.name === name);
      const raw = await mc.executeTool(tool, JSON.stringify(args));
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      return JSON.parse(parsed.content[0].text);
    },
    { name, args },
  );
}

// Centre of NiiVue tile i in CSS pixels: leftTopWidthHeight is in canvas pixels (CSS times uiData.dpr).
function tileCentre(page, i) {
  return page.evaluate((i) => {
    const nv = window.__radar.viewer.nv;
    const [l, t, w, h] = nv.screenSlices[i].leftTopWidthHeight;
    const dpr = nv.uiData.dpr || 1;
    const rect = nv.canvas.getBoundingClientRect();
    return { x: rect.left + (l + w / 2) / dpr, y: rect.top + (t + h / 2) / dpr };
  }, i);
}

async function login(page) {
  await page.goto("/");
  const res = await page.request.post("/auth/login", { data: { token: TOKEN } });
  expect(res.status(), "login").toBeLessThan(300);
}

async function scanWithDoneJob(page) {
  if (process.env.RADAR_SMOKE_SCAN) return process.env.RADAR_SMOKE_SCAN;
  const res = await page.request.get("/scans");
  const { scans } = await res.json();
  const scan = scans.find((s) => s.latest_job?.state === "done");
  expect(scan, "a scan with a done job (seed one with the fake backend)").toBeTruthy();
  return scan.id;
}

async function shot(page, name) {
  fs.mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS, name) });
}

async function ctrlWheel(page, x, y, deltaY) {
  await page.mouse.move(x, y);
  await page.keyboard.down("Control");
  await page.mouse.wheel(0, deltaY);
  await page.keyboard.up("Control");
}

// The canvas element under a point, so a wheel test cannot pass by landing on a panel.
function onCanvas(page, { x, y }) {
  return page.evaluate(({ x, y }) => document.elementFromPoint(x, y) === window.__radar.viewer.nv.canvas, { x, y });
}

test.use({ viewport: { width: 1600, height: 950 } });

test("large view: focus mode, light panel, zoom and pan", async ({ page }) => {
  await login(page);
  const scanId = await scanWithDoneJob(page);
  await page.goto(`/workspace.html?scan=${encodeURIComponent(scanId)}`);
  await page.waitForFunction(() => window.__radar?.viewState().mask_loaded === true, null, { timeout: 60_000 });
  const vs = () => page.evaluate(() => window.__radar.viewState());
  const canvasWidth = () => page.evaluate(() => document.getElementById("viewer").getBoundingClientRect().width);
  // The drawing buffer: it only follows the CSS size when NiiVue's own observer resizes it.
  const bufferWidth = () => page.evaluate(() => window.__radar.viewer.nv.canvas.width);
  const cal = () => page.evaluate(() => [window.__radar.viewer.nv.volumes[0].cal_min, window.__radar.viewer.nv.volumes[0].cal_max]);

  // A selected finding, so focus can be shown to keep it.
  await page.locator(".organ-group:not(.not-scored *) .finding").first().click();
  await expect.poll(async () => (await vs()).active_finding).not.toBeNull();
  await shot(page, "largeview-normal.png");

  // f enters focus: the viewer covers the page, the panel opens, the selection and crosshair stay.
  const before = await vs();
  expect(before.focus).toBe(false);
  const normalWidth = await canvasWidth();
  const normalBuffer = await bufferWidth();
  await page.locator("body").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("f");
  await expect(page.locator("body")).toHaveClass(/\bfocus\b/);
  await expect(page.locator("#focus-toggle")).toHaveAttribute("aria-pressed", "true");
  await expect.poll(canvasWidth).toBeGreaterThan(normalWidth + 300);
  await expect.poll(bufferWidth).toBeGreaterThan(normalBuffer + 300);
  await expect(page.locator("#light-panel")).toBeVisible();
  await expect(page.locator("#light-toggle")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("[data-persona-root]")).toBeHidden();
  // The findings pane is covered by the viewer.
  const covered = await page.evaluate(() => {
    const r = document.querySelector(".findings-pane").getBoundingClientRect();
    return document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2).closest(".viewer-pane") != null;
  });
  expect(covered).toBe(true);
  let now = await vs();
  expect(now.focus).toBe(true);
  expect(now.crosshair_mm).toEqual(before.crosshair_mm);
  expect(now.active_finding).toBe(before.active_finding);
  expect(now.isolated_label).toBe(before.isolated_label);
  expect(now.slice).toEqual(before.slice);

  // The width slider sets the window: cal_min/cal_max, the HU numbers and window_hu agree.
  await page.locator("#ww").fill("800");
  const width = widthFromSlider(800);
  now = await vs();
  expect(now.window_preset).toBe("custom");
  expect(now.window_hu.width).toBe(width);
  expect(now.window_hu.level).toBe(50);
  expect(await cal()).toEqual([now.window_hu.min, now.window_hu.max]);
  await expect(page.locator("#ww-value")).toHaveText(String(width));
  await expect(page.locator("#window-range")).toHaveText(`Custom, ${now.window_hu.min} to ${now.window_hu.max} HU`);
  await expect(page.locator("[data-window][aria-pressed=true]")).toHaveCount(0);

  // The toolbar's Liver preset moves the panel's sliders.
  await page.locator("#viewer-toolbar [data-window=liver]").click();
  await expect(page.locator("#ww-value")).toHaveText("150");
  await expect(page.locator("#wl-value")).toHaveText("30");
  await expect(page.locator("#ww")).toHaveValue(String(sliderFromWidth(150)));
  await expect(page.locator("#wl")).toHaveValue("30");
  await expect(page.locator("[data-window=liver][aria-pressed=true]")).toHaveCount(1);
  await expect(page.locator("#window-range")).toHaveText("Liver, -45 to 105 HU");
  expect(await cal()).toEqual([-45, 105]);

  // The level slider and the page tool.
  await page.locator("#wl").fill("100");
  expect((await vs()).window_hu).toEqual({ min: 25, max: 175, width: 150, level: 100 });
  // Right-drag sets the window from the dragged box (NiiVue's contrast mode); the panel follows.
  const img = await page.locator("#viewer").boundingBox();
  await page.mouse.move(img.x + img.width * 0.3, img.y + img.height * 0.3);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(img.x + img.width * 0.55, img.y + img.height * 0.7, { steps: 5 });
  await page.mouse.up({ button: "right" });
  await expect.poll(async () => (await vs()).window_hu.level).not.toBe(100);
  now = await vs();
  expect(now.window_preset).toBe("custom");
  const [lo, hi] = await cal();
  expect(lo).toBeCloseTo(now.window_hu.min, 0);
  expect(hi).toBeCloseTo(now.window_hu.max, 0);
  await expect(page.locator("#ww-value")).toHaveText(String(now.window_hu.width));
  await expect(page.locator("#wl-value")).toHaveText(String(now.window_hu.level));
  const wl = await callTool(page, "set_window_level", { width: 400, level: 50 });
  expect(wl).toEqual({ ok: true, window_preset: "soft_tissue", window_hu: { min: -150, max: 250, width: 400, level: 50 } });
  await expect(page.locator("#wl")).toHaveValue("50");

  // Zoom by key, button, ctrl+wheel and page tool; Reset view returns to 1. The crosshair stays.
  const crosshair = () => page.evaluate(() => Array.from(window.__radar.viewer.nv.scene.crosshairPos));
  const pos = await crosshair();
  await page.locator("body").click({ position: { x: 800, y: 3 } }); // on the toolbar's padding, off the canvas
  await page.keyboard.press("+");
  expect((await vs()).zoom).toBe(1.25);
  await page.locator("#zoom-in").click();
  expect((await vs()).zoom).toBe(1.56);
  await page.locator("#zoom-out").click();
  expect((await vs()).zoom).toBe(1.25);
  await page.keyboard.press("-");
  expect((await vs()).zoom).toBe(1);
  const box = await page.locator("#viewer").boundingBox();
  await ctrlWheel(page, box.x + box.width / 3, box.y + box.height / 2, -100);
  expect((await vs()).zoom).toBe(1.25);
  expect(await crosshair()).toEqual(pos); // a ctrl+wheel never reaches NiiVue's slice scroll
  expect(await callTool(page, "set_zoom", { zoom: 3 })).toEqual({ ok: true, zoom: 3 });
  // The pan moved so the crosshair stays where it was on screen.
  const { actual, expected } = await page.evaluate(() => {
    const nv = window.__radar.viewer.nv;
    return { actual: Array.from(nv.scene.pan2Dxyzmm), expected: Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3) };
  });
  zoomAround([0, 0, 0, 1], 3, expected).forEach((v, i) => expect(actual[i]).toBeCloseTo(v, 3));
  await page.locator("#view-reset").click();
  expect((await vs()).zoom).toBe(1);
  expect(await page.evaluate(() => Array.from(window.__radar.viewer.nv.scene.pan2Dxyzmm))).toEqual([0, 0, 0, 1]);

  // Shift-drag and middle-drag pan; neither moves the crosshair.
  const pan = () => page.evaluate(() => Array.from(window.__radar.viewer.nv.scene.pan2Dxyzmm));
  const cx = box.x + box.width / 3;
  const cy = box.y + box.height / 2;
  await page.keyboard.down("Shift");
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 80, cy + 40, { steps: 5 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  const shifted = await pan();
  expect(shifted.slice(0, 3)).not.toEqual([0, 0, 0]);
  await page.mouse.move(cx, cy);
  await page.mouse.down({ button: "middle" });
  await page.mouse.move(cx - 60, cy, { steps: 5 });
  await page.mouse.up({ button: "middle" });
  expect((await pan()).slice(0, 3)).not.toEqual(shifted.slice(0, 3));
  expect(await crosshair()).toEqual(pos);
  await page.locator("#view-reset").click();
  expect(await pan()).toEqual([0, 0, 0, 1]);

  // Gamma, invert, colour map, then Reset, which leaves the zoom alone.
  await page.locator("#gamma").fill("1.5");
  await expect(page.locator("#gamma-value")).toHaveText("1.50");
  const sceneGamma = () => page.evaluate(() => window.__radar.viewer.nv.scene.gamma);
  expect(await sceneGamma()).toBe(1.5);
  await page.locator("#invert").click();
  await page.locator("#colormap").selectOption("hot");
  await page.locator("#light-panel [data-outline]").click();
  await page.locator("#light-panel [data-mask]").click();
  now = await vs();
  expect([now.gamma, now.invert, now.colormap, now.outline, now.mask_on]).toEqual([1.5, true, "hot", true, false]);
  expect(await page.evaluate(() => [window.__radar.viewer.nv.volumes[0].colormap, window.__radar.viewer.nv.volumes[0].colormapInvert])).toEqual(["hot", true]);
  expect(await cal()).toEqual([-150, 250]); // the colour map change kept the window
  await expect(page.locator("#mask-toggle")).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#outline-toggle")).toHaveAttribute("aria-pressed", "true");
  await shot(page, "largeview-focus.png");
  await callTool(page, "set_zoom", { zoom: 2 });
  await page.locator("#light-reset").click();
  now = await vs();
  expect([now.window_preset, now.gamma, now.invert, now.colormap, now.mask_on, now.mask_opacity, now.outline, now.zoom]).toEqual([
    "soft_tissue", 1, false, "gray", true, 0.45, false, 2,
  ]);
  await expect(page.locator("#colormap")).toHaveValue("gray");
  await expect(page.locator("#invert")).toHaveAttribute("aria-pressed", "false");
  expect(await cal()).toEqual([-150, 250]);
  expect(await sceneGamma()).toBe(1);
  await page.locator("#view-reset").click();

  // Escape closes an open look card first and leaves focus only on the second press.
  await page.locator("body").click({ position: { x: 800, y: 3 } });
  await page.keyboard.press("w");
  await expect(page.locator("#look-card")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator("#look-card")).toBeHidden();
  await expect(page.locator("body")).toHaveClass(/\bfocus\b/);
  // Leaving focus: the canvas and its drawing buffer shrink back and the panel closes.
  await page.keyboard.press("Escape");
  await expect(page.locator("body")).not.toHaveClass(/\bfocus\b/);
  await expect.poll(canvasWidth).toBeCloseTo(normalWidth, 0);
  await expect.poll(bufferWidth).toBe(normalBuffer);
  await expect(page.locator("#light-panel")).toBeHidden();
  await expect(page.locator("[data-persona-root]").first()).toBeVisible();
  expect((await vs()).focus).toBe(false);

  // Multiplanar plus focus keeps the three tiles and the crosshair.
  await page.keyboard.press("m");
  const posMulti = await crosshair();
  await page.keyboard.press("f");
  await expect.poll(canvasWidth).toBeGreaterThan(normalWidth + 300);
  expect(await page.evaluate(() => window.__radar.viewer.nv.getCustomLayout())).toHaveLength(3);
  expect(await crosshair()).toEqual(posMulti);
  now = await vs();
  expect([now.slice_type, now.main_plane, now.focus]).toEqual(["multiplanar", "axial", true]);
  // In multiplanar the panel waits for the Light button, since it would cover a reference view.
  await expect(page.locator("#light-panel")).toBeHidden();
  await page.evaluate(() => window.__radar.toggleLight(false));
  // Ctrl+wheel is blocked on a reference tile and zooms on the main one. Wait until NiiVue has
  // resized the drawing buffer to the focus size and laid the references out in the right quarter
  // (leftTopWidthHeight is the drawn slice, fitted to its aspect, so only the left edge is fixed).
  await expect
    .poll(() =>
      page.evaluate(() => {
        const nv = window.__radar.viewer.nv;
        const settled = Math.abs(nv.canvas.width - nv.canvas.clientWidth * (nv.uiData.dpr || 1)) < 2;
        return settled && nv.screenSlices[1].leftTopWidthHeight[0] >= 0.75 * nv.canvas.width - 4;
      }),
    )
    .toBe(true);
  const ref = await tileCentre(page, 1);
  expect(await onCanvas(page, ref)).toBe(true);
  await ctrlWheel(page, ref.x, ref.y, -100);
  expect((await vs()).zoom).toBe(1);
  const main = await tileCentre(page, 0);
  expect(await onCanvas(page, main)).toBe(true);
  await ctrlWheel(page, main.x, main.y, -100);
  expect((await vs()).zoom).toBe(1.25);
  expect(await crosshair()).toEqual(posMulti);
  await shot(page, "largeview-multiplanar-focus.png");

  // The page tool toggles focus on and off.
  expect(await callTool(page, "toggle_focus", { on: false })).toEqual({ ok: true, focus: false });
  await expect(page.locator("body")).not.toHaveClass(/\bfocus\b/);
  expect(await callTool(page, "toggle_focus", {})).toEqual({ ok: true, focus: true });
  await expect(page.locator("body")).toHaveClass(/\bfocus\b/);
  const state = await callTool(page, "get_view_state");
  for (const key of ["focus", "window_hu", "gamma", "invert", "colormap", "zoom"]) expect(state).toHaveProperty(key);
  expect(await callTool(page, "toggle_focus", { on: false })).toEqual({ ok: true, focus: false });
});
