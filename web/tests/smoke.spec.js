import { expect, test } from "@playwright/test";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";

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

test("workspace: findings, viewer, page tools, keys", async ({ page }) => {
  await login(page);
  const scanId = await scanWithDoneJob(page);
  await page.goto(`/workspace.html?scan=${encodeURIComponent(scanId)}`);

  // The three fixed sentences.
  await expect(page.getByText("RADAR ships no calibrated thresholds; this line is for display")).toBeVisible();
  await expect(
    page.getByText(
      "The Radar model is currently intended for research purposes only. Further improvements and prospective clinical studies are still required before it can be used directly for clinical deployment.",
    ),
  ).toBeVisible();
  await expect(page.getByText("Model: RADAR by Alibaba DAMO Academy, CC BY-NC-SA 4.0")).toBeVisible();

  // 146 findings once the result is in.
  await expect(page.locator('#findings[data-ready="true"] .finding')).toHaveCount(146, { timeout: 30_000 });
  await page.waitForFunction(() => window.__radar?.viewState().mask_loaded === true, null, { timeout: 60_000 });

  // Click a finding in a scored organ; the crosshair moves.
  const before = await page.evaluate(() => window.__radar.viewState().crosshair_mm);
  const fracBefore = await page.evaluate(() => Array.from(window.__radar.viewer.nv.scene.crosshairPos));
  const row = page.locator(".organ-group:not(.not-scored *) .finding").first();
  const organ = await row.getAttribute("data-organ");
  await row.click();
  await expect.poll(() => page.evaluate(() => window.__radar.viewState().crosshair_mm)).not.toEqual(before);
  // And NiiVue's own crosshair moved, not just our copy of it.
  await expect.poll(() => page.evaluate(() => Array.from(window.__radar.viewer.nv.scene.crosshairPos))).not.toEqual(fracBefore);

  // The registered page tool answers through document.modelContext.
  const state = await page.evaluate(async () => {
    const mc = document.modelContext;
    const tools = await mc.getTools();
    const tool = tools.find((t) => t.name === "get_view_state");
    const raw = await mc.executeTool(tool, "{}");
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    return JSON.parse(parsed.content[0].text);
  });
  expect(state.active_organ).toBe(organ);
  expect(state.scan_id).toBe(scanId);

  // Space toggles the mask.
  await page.locator("body").click({ position: { x: 5, y: 5 } });
  const maskBefore = await page.evaluate(() => window.__radar.viewState().mask_on);
  await page.keyboard.press("Space");
  expect(await page.evaluate(() => window.__radar.viewState().mask_on)).toBe(!maskBefore);

  // ] moves the display line up by 5.
  await page.keyboard.press("]");
  expect(await page.evaluate(() => window.__radar.viewState().threshold_pct)).toBe(55);
});
