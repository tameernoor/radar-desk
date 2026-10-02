import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";

// Calls the registered page tool the way the chat does, through document.modelContext.
function getViewState(page) {
  return page.evaluate(async () => {
    const mc = document.modelContext;
    const tools = await mc.getTools();
    const tool = tools.find((t) => t.name === "get_view_state");
    const raw = await mc.executeTool(tool, "{}");
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    return JSON.parse(parsed.content[0].text);
  });
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
  const state = await getViewState(page);
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

  // Scroll to another slice; the view state reports the new slice and what RADAR outlined on it.
  const sliceBefore = state.slice;
  expect(sliceBefore).toMatchObject({ axis: "axial" });
  const posBefore = await page.evaluate(() => Array.from(window.__radar.viewer.nv.scene.crosshairPos));
  const box = await page.locator("#viewer").boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, 100);
  await page.mouse.wheel(0, 100);
  await expect.poll(() => page.evaluate(() => Array.from(window.__radar.viewer.nv.scene.crosshairPos))).not.toEqual(posBefore);
  const scrolled = await getViewState(page);
  console.log("get_view_state after scrolling:\n" + JSON.stringify(scrolled, null, 2));
  // Outside test-results, which Playwright empties at the start of every run.
  const out = path.join(import.meta.dirname, "..", ".smoke-out");
  fs.mkdirSync(out, { recursive: true });
  fs.writeFileSync(path.join(out, "view-state.json"), JSON.stringify(scrolled, null, 2));
  expect(scrolled.slice.index).not.toBe(sliceBefore.index);
  expect(scrolled.slice.count).toBe(sliceBefore.count);
  expect(scrolled.slice.number).toBe(scrolled.slice.index + 1);
  expect(Array.isArray(scrolled.organs_on_slice) && scrolled.organs_on_slice.length > 0).toBe(true);
  for (const o of scrolled.organs_on_slice) {
    expect(typeof o.organ).toBe("string");
    expect(o.pixels).toBeGreaterThan(0);
    expect(typeof o.percent_of_mask).toBe("number");
  }

  // w opens the "What am I looking at?" card for this slice; Escape closes it.
  await page.keyboard.press("w");
  const card = page.locator("#look-card");
  await expect(card).toBeVisible();
  await expect(card).toContainText(`slice ${scrolled.slice.number} of ${scrolled.slice.count}`);
  await expect(card.locator(".look-organ").first()).toBeVisible();
  await expect(card).toContainText("Outlines are RADAR's segmentation, not confirmed anatomy.");
  await page.keyboard.press("Escape");
  await expect(card).toBeHidden();

  // A card about one slice closes when the slice changes, so it is never stale.
  await page.keyboard.press("w");
  await expect(card).toBeVisible();
  // Scroll just left of the card, which covers the right of the canvas; the image tile is centred,
  // so this point is on the image at this viewport size.
  const cardBox = await card.boundingBox();
  await page.mouse.move(cardBox.x - 40, box.y + box.height / 2);
  await page.mouse.wheel(0, 100);
  await expect(card).toBeHidden();
});
