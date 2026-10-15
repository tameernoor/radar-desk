// The Readers tab, against an isolated server seeded with scripts/seed_dev.py --readers (fake GPU, no chat model).
import { expect, test } from "@playwright/test";
import { foldText, overview } from "../src/readers.js";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";

async function login(page) {
  await page.goto("/");
  const res = await page.request.post("/auth/login", { data: { token: TOKEN } });
  expect(res.status(), "login").toBeLessThan(300);
}

// The seeded scan is the one with a report.
async function seededScan(page) {
  const { scans } = await (await page.request.get("/scans")).json();
  for (const s of scans) if ((await page.request.get(`/scans/${s.id}/report`)).ok()) return s;
  return null;
}

test.use({ viewport: { width: 1600, height: 950 } });

test("readers tab: groups, words, fold, display line, expand, jump, NV's view", async ({ page }) => {
  await login(page);
  const scan = await seededScan(page);
  expect(scan, "a scan seeded with scripts/seed_dev.py --readers").toBeTruthy();
  const view = await (await page.request.get(`/scans/${scan.id}/readers`)).json();
  expect(view.nv.job.state).toBe("done");
  expect(view.report.status).toBe("ready");
  expect(view.nv.read.status).toBe("ready");

  await page.goto(`/workspace.html?scan=${encodeURIComponent(scan.id)}`);
  await page.getByRole("tab", { name: "NV-Reason-CT findings" }).click();
  await expect(page.locator("#findings")).toBeHidden();
  const readers = page.locator('#readers[data-ready="true"]');
  await expect(readers).toBeVisible();

  // Gone: the intro, the Reports cards, the Compare section, the slider and the dot legend.
  await expect(readers.getByText("Three readers of the same scan")).toHaveCount(0);
  await expect(readers.getByText("Nothing here says who is right")).toHaveCount(0);
  await expect(readers.getByRole("slider")).toHaveCount(0);
  await expect(readers.getByRole("heading", { name: "Compare the readers" })).toHaveCount(0);
  await expect(readers.locator(".rkey")).toHaveCount(0);

  // The expected groups come from the same pure function the page uses, applied to what the server reports.
  const at50 = overview(view.rows, { threshold: 50 });
  const group = (key) => readers.locator(`[data-group="${key}"]`);
  await expect(group("disagree").locator(".rrow")).toHaveCount(at50.disagree.length);
  await expect(group("disagree").locator(".rrow.flagged")).toHaveCount(at50.disagree.length);
  const ascites = group("disagree").locator('.rrow[data-row="ascites"]');
  await expect(ascites).toHaveClass(/flagged/);
  await expect(ascites).toContainText("NV absent");
  await expect(ascites).toContainText("Report present");
  await expect(ascites.locator('[data-reader="radar"]')).toHaveAttribute("data-state", "not_scored");
  await expect(ascites).not.toContainText(/[●○◌]/);
  await expect(readers.getByText(foldText(at50.quiet.length))).toBeVisible();
  if (at50.outside.length) {
    await expect(group("outside").locator(".rout")).toHaveCount(at50.outside.length);
    await expect(group("outside").locator(".rout").first()).toContainText("inside NV's view");
    for (const { row } of at50.outside) {
      const line = group("outside").locator(`.rout[data-out="${row.id}"]`);
      if (row.nv.state === "present" || row.nv.state === "absent") await expect(line).toContainText("NV said");
      else await expect(line).not.toContainText("NV said");
    }
  }
  // Columns line up across groups.
  // Both measured in one evaluate, polled, since the tab re-renders while the scan loads.
  if (at50.disagree.length && at50.othersFound.length) {
    const xs = () => readers.evaluate((root) => ["disagree", "others-found"].map((g) => root.querySelector(`[data-group="${g}"] .rrow .rtitle`)?.getBoundingClientRect().x ?? -1));
    await expect.poll(async () => { const [a, b] = await xs(); return a > 0 && a === b; }).toBe(true);
  }

  await page.waitForFunction(() => Boolean(window.__radar?.setThreshold)); // the tab can render before the page tools are registered
  // RADAR's display line (on the RADAR tab) moves the Disagree count.
  await expect(readers.locator('[data-part="display-line"]')).toContainText("now 50%");
  await page.evaluate(() => window.__radar.setThreshold(1));
  await expect(readers.locator('[data-part="display-line"]')).toContainText("now 1%");
  const at1 = overview(view.rows, { threshold: 1 });
  if (at1.disagree.length) await expect(group("disagree").locator(".rrow")).toHaveCount(at1.disagree.length);
  else await expect(group("disagree")).toHaveCount(0);
  await page.evaluate(() => window.__radar.setThreshold(50));
  await expect(group("disagree").locator(".rrow")).toHaveCount(at50.disagree.length);

  // Each reader in its own words; the click on the row toggles it.
  await ascites.locator(".rtitle").click();
  await expect(readers.locator(".rdetails")).toContainText('"Small volume ascites in the pelvis."');
  await expect(readers.locator(".rdetails")).toContainText('"No ascites."');
  await readers.getByRole("button", { name: /Other mentions \(1\)/ }).click();
  await expect(readers.getByText('"Degenerative change of the lumbar spine."')).toBeVisible();

  // A click on a row also jumps the viewer to its organ.
  await page.waitForFunction(() => window.__radar?.viewState().mask_loaded === true, null, { timeout: 60_000 });
  await readers.locator('.rrow[data-row="liver_lesion"] .rtitle').click();
  expect(await page.evaluate(() => window.__radar.viewState().active_organ)).toBe("Liver");
  await ascites.getByRole("button", { name: /Hide details for/ }).click(); // keyboard path, collapses it
  await expect(readers.locator(".rdetails").filter({ hasText: '"No ascites."' })).toHaveCount(0);

  // NV's view draws the crop box.
  await page.getByRole("button", { name: "NV's view" }).click();
  await expect(page.getByRole("button", { name: "NV's view" })).toHaveAttribute("aria-pressed", "true");
  expect(await page.evaluate(() => window.__radar.viewState().nv_view)).toBe(true);

  // Back to RADAR's list.
  await page.getByRole("tab", { name: "RADAR findings" }).click();
  await expect(page.locator('#findings[data-ready="true"] .finding')).toHaveCount(146, { timeout: 30_000 });
});

test("readers tab: header, impression and questions are visible on open, a question is answered in place", async ({ page }) => {
  await login(page);
  const scan = await seededScan(page);
  const view = await (await page.request.get(`/scans/${scan.id}/readers`)).json();
  const q = view.rows.find((r) => r.id === "ascites").queries.find((x) => x.kind === "yes_no");
  expect(q, "a seeded yes/no question for ascites").toBeTruthy();
  expect(view.nv.report.answer).toContain("IMPRESSION:");

  await page.goto(`/workspace.html?scan=${encodeURIComponent(scan.id)}`);
  await page.getByRole("tab", { name: "NV-Reason-CT findings" }).click();
  const readers = page.locator('#readers[data-ready="true"]');
  await expect(readers.getByText("saw a 384 mm cube of the scan")).toBeVisible();
  await expect(readers.getByRole("button", { name: "Open reports" })).toBeVisible();
  await expect(readers.getByText("Radiologist's report attached.")).toBeVisible();

  // The impression is clamped, the full text lives in the reports window.
  const imp = readers.locator('[data-part="impression"]');
  await expect(imp).toContainText("Impression:");
  await expect(imp).toContainText("2. Left renal cyst.");
  await expect(imp).not.toContainText("<answer>");
  await expect(readers.getByText("How it reasoned")).toHaveCount(0);
  await expect(readers.locator('[data-part="radiologist"]')).toHaveCount(0);

  // Questions: every prepared question once, no answer yet.
  const ask = readers.locator("section", { has: page.getByRole("heading", { name: "Ask NV-Reason-CT" }) });
  const ids = [...new Set([...view.queries.general, ...view.rows.flatMap((r) => r.queries || [])].map((x) => x.id))];
  await expect(ask.locator(".rchip")).toHaveCount(ids.length);
  await expect(ask.getByText("Or ask the chat on the right to put your own question to NV-Reason-CT.")).toBeVisible();
  await expect(ask.locator(`.rchip[data-query="${q.id}"]`)).not.toContainText("Answer only Yes or No");

  await ask.locator(`.rchip[data-query="${q.id}"]`).click();
  const asked = ask.locator(`.rq[data-q="${q.id}"]`);
  await expect(asked.locator(".rqa")).toContainText("NV: Yes.", { timeout: 30_000 });
  await expect(asked.locator(".rqa")).not.toContainText("Yes..");
  await expect(asked.getByText("NV's answer differs from its report")).toBeVisible();

  // The same question in its row shows the same answer; the NV cell stays the report read.
  await readers.locator('.rrow[data-row="ascites"] .rexpand').click();
  await expect(readers.locator(`.rdetails .rq[data-q="${q.id}"] .rqa`)).toContainText("NV: Yes.");
  await expect(readers.locator('.rrow[data-row="ascites"] [data-reader="nv"]')).toHaveAttribute("data-state", "absent");
});

test("readers tab: Open reports pops out the full reports", async ({ page }) => {
  await login(page);
  const scan = await seededScan(page);
  await page.goto(`/workspace.html?scan=${encodeURIComponent(scan.id)}`);
  await page.getByRole("tab", { name: "NV-Reason-CT findings" }).click();
  const readers = page.locator('#readers[data-ready="true"]');
  await expect(readers).toBeVisible();

  const [popup] = await Promise.all([page.waitForEvent("popup"), readers.getByRole("button", { name: "Open reports" }).click()]);
  await expect(popup).toHaveURL(new RegExp(`/reports\\.html\\?scan=${scan.id}`));
  const rad = popup.locator('[data-col="radiologist"]');
  await expect(rad).toContainText("Peritoneum: Small volume ascites in the pelvis.");
  await expect(rad).toContainText("2. Small volume ascites.");
  const nv = popup.locator('[data-col="nv"]');
  await expect(nv.getByRole("heading", { name: "Impression" })).toBeVisible();
  await expect(nv).toContainText("2. Left renal cyst.");
  await expect(nv.getByRole("heading", { name: "Findings" })).toBeVisible();
  await expect(nv).toContainText("Liver: A small hypodense lesion");
  await expect(nv.locator("details", { hasText: "How it reasoned" })).toHaveCount(1);
  await expect(nv.locator("details")).not.toHaveAttribute("open", "");
});

test("readers tab: opening the reports again keeps the open window as it is", async ({ page }) => {
  await login(page);
  const scan = await seededScan(page);
  await page.goto(`/workspace.html?scan=${encodeURIComponent(scan.id)}`);
  await page.getByRole("tab", { name: "NV-Reason-CT findings" }).click();
  const readers = page.locator('#readers[data-ready="true"]');
  await expect(readers).toBeVisible();
  const open = readers.getByRole("button", { name: "Open reports" });
  const [popup] = await Promise.all([page.waitForEvent("popup"), open.click()]);
  await popup.locator("summary").click();
  await expect(popup.locator("details")).toHaveAttribute("open", "");
  await popup.evaluate(() => (document.body.dataset.marker = "same document"));
  await open.click();
  await page.waitForTimeout(500);
  await expect(popup.locator("body")).toHaveAttribute("data-marker", "same document"); // not reloaded
  await expect(popup.locator("details")).toHaveAttribute("open", "");
});
