// Viewing recipes in the browser: set_view_for, set_light, the Angio preset and the View-for button.
import { expect, test } from "@playwright/test";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";

// Calls a registered page tool the way the chat does, through document.modelContext.
// Returns the parsed result, or {isError, text} for an error result.
function callTool(page, name, args = {}) {
  return page.evaluate(
    async ({ name, args }) => {
      const mc = document.modelContext;
      const tool = (await mc.getTools()).find((t) => t.name === name);
      const raw = await mc.executeTool(tool, JSON.stringify(args));
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (parsed.isError) return { isError: true, text: parsed.content[0].text };
      return JSON.parse(parsed.content[0].text);
    },
    { name, args },
  );
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

test.use({ viewport: { width: 1600, height: 950 } });

test("viewing recipes: set_view_for, set_light, Angio and View for", async ({ page }) => {
  await login(page);
  const scanId = await scanWithDoneJob(page);
  await page.goto(`/workspace.html?scan=${encodeURIComponent(scanId)}`);
  await page.waitForFunction(() => window.__radar?.viewState().mask_loaded === true, null, { timeout: 60_000 });
  const vs = () => page.evaluate(() => window.__radar.viewState());
  const cal = () => page.evaluate(() => [window.__radar.viewer.nv.volumes[0].cal_min, window.__radar.viewer.nv.volumes[0].cal_max]);
  const jobId = (await vs()).job_id;
  const stats = (await (await page.request.get(`/jobs/${jobId}/result`)).json()).organ_stats;
  expect(stats.Aorta, "the fake result outlines the aorta").toBeTruthy();
  const centroid = stats.Aorta.centroid_mm;

  // With nothing selected the button names no organ and asks for one.
  await expect(page.locator("#view-for")).toHaveText("View for organ");
  await expect(page.locator("#view-for")).toHaveClass(/\bmuted\b/);
  await page.locator("#view-for").click();
  await expect(page.locator("#viewer-notice")).toHaveText("Select a finding or an organ first.");

  // set_view_for with an organ: jump, the angio window, the recipe zoom, the box.
  const aorta = await callTool(page, "set_view_for", { target: "aorta" });
  expect(aorta).toMatchObject({
    ok: true,
    target: "aorta",
    organ: "Aorta",
    finding: null,
    key: null,
    window_preset: "angio",
    window_hu: { min: -100, max: 500, width: 600, level: 200 },
    zoom: 2.5,
    scoring_box: { on: true },
    note: null,
  });
  expect(aorta.why).toContain("vascular window");
  expect(await cal()).toEqual([-100, 500]);
  let now = await vs();
  now.crosshair_mm.forEach((v, i) => expect(Math.abs(v - centroid[i])).toBeLessThan(0.1));
  expect([now.zoom, now.scoring_box_organ, now.active_organ]).toEqual([2.5, "Aorta", "Aorta"]);
  expect(await page.evaluate(() => window.__radar.viewer.nv.volumes.find((v) => v.name === "scoring box")?.opacity)).toBe(1);
  await expect(page.locator('.organ-chip.current[data-organ="Aorta"]')).toHaveCount(1);
  await expect(page.locator("#box-toggle")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("[data-window=angio][aria-pressed=true]")).toHaveCount(1);
  await expect(page.locator("#window-range")).toHaveText("Angio, -100 to 500 HU");
  await expect(page.locator("#view-for")).toHaveText("View for Aorta");
  await expect(page.locator("#view-for")).toHaveAttribute("title", "View for Aorta");
  await expect(page.locator("#view-for")).not.toHaveClass(/\bmuted\b/);

  // The aorta is in the middle of the main view: with the crosshair moved off it (the pan stays),
  // a click at the canvas centre brings it back.
  await page.evaluate((c) => window.__radar.viewer.jumpToMm([c[0] + 10, c[1] + 10, c[2]]), centroid);
  const away = (await vs()).crosshair_mm;
  expect(Math.abs(away[0] - centroid[0])).toBeGreaterThan(5);
  const box = await page.locator("#viewer").boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await expect.poll(async () => (await vs()).crosshair_mm).not.toEqual(away);
  now = await vs();
  expect(Math.abs(now.crosshair_mm[0] - centroid[0])).toBeLessThan(3);
  expect(Math.abs(now.crosshair_mm[1] - centroid[1])).toBeLessThan(3);

  // set_view_for with a finding key selects that finding.
  const key = await page.locator('.finding[data-organ="Liver"]').first().getAttribute("data-key");
  const liver = await callTool(page, "set_view_for", { target: key });
  expect(liver).toMatchObject({ ok: true, organ: "Liver", key, window_preset: "liver", zoom: 1.5 });
  expect((await vs()).active_finding).toBe(key);
  await expect(page.locator(`.finding.active[data-key="${key}"]`)).toHaveCount(1);

  // An unknown target is an error naming the organs.
  const bad = await callTool(page, "set_view_for", { target: "brain" });
  expect(bad.isError).toBe(true);
  expect(bad.text).toContain("Adrenal gland");

  // set_light: gamma, then invert and colour map with the window kept, then reset.
  expect(await callTool(page, "set_light", { gamma: 1.5 })).toEqual({ ok: true, gamma: 1.5, invert: false, colormap: "gray" });
  expect(await page.evaluate(() => window.__radar.viewer.nv.scene.gamma)).toBe(1.5);
  const before = await cal();
  expect(await callTool(page, "set_light", { invert: true, colormap: "hot" })).toEqual({ ok: true, gamma: 1.5, invert: true, colormap: "hot" });
  expect(await page.evaluate(() => [window.__radar.viewer.nv.volumes[0].colormap, window.__radar.viewer.nv.volumes[0].colormapInvert])).toEqual(["hot", true]);
  expect(await cal()).toEqual(before);
  expect(await callTool(page, "set_light", { reset: true })).toEqual({ ok: true, gamma: 1, invert: false, colormap: "gray" });
  expect((await vs()).window_preset).toBe("soft_tissue");
  expect((await callTool(page, "set_light", { colormap: "jet" })).isError).toBe(true);
  // A bad colour map changes nothing, not even the gamma given with it.
  expect((await callTool(page, "set_light", { gamma: 2, colormap: "jet" })).isError).toBe(true);
  expect((await vs()).gamma).toBe(1);
  expect((await callTool(page, "set_light", {})).isError).toBe(true);

  // Key 5 and the toolbar's Angio button.
  await page.locator("body").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("5");
  expect((await vs()).window_preset).toBe("angio");
  expect(await cal()).toEqual([-100, 500]);
  await page.keyboard.press("1");
  expect((await vs()).window_preset).toBe("soft_tissue");
  await page.locator("#viewer-toolbar [data-window=angio]").click();
  expect((await vs()).window_preset).toBe("angio");

  // The View-for button does what the tool does for the active organ.
  await page.locator('.organ-chip[data-organ="Aorta"]').click();
  await expect.poll(async () => (await vs()).active_organ).toBe("Aorta");
  await page.locator("#view-reset").click();
  await page.keyboard.press("1");
  await page.locator("#box-toggle").click(); // box off, so the button has to turn it on
  expect((await vs()).scoring_box_organ).toBeNull();
  await page.locator("#view-for").click();
  now = await vs();
  expect([now.window_preset, now.zoom, now.scoring_box_organ]).toEqual(["angio", 2.5, "Aorta"]);
  now.crosshair_mm.forEach((v, i) => expect(Math.abs(v - centroid[i])).toBeLessThan(0.1));

  // The Light panel fits a 1280 by 800 window, and Escape closes it outside focus mode.
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.locator("#light-toggle").click();
  await expect(page.locator("#light-reset")).toBeInViewport();
  await page.keyboard.press("Escape");
  await expect(page.locator("#light-panel")).toBeHidden();
  await expect(page.locator("#light-toggle")).toBeFocused();
});
