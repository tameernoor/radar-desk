// The Readers tab, against an isolated server seeded with scripts/seed_dev.py --readers (fake GPU, no chat model).
import { expect, test } from "@playwright/test";
import { arrange, foldText } from "../src/readers.js";

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

test("readers tab: rows, flags, fold, display line, words, jump, NV's view", async ({ page }) => {
  await login(page);
  const scan = await seededScan(page);
  expect(scan, "a scan seeded with scripts/seed_dev.py --readers").toBeTruthy();
  const view = await (await page.request.get(`/scans/${scan.id}/readers`)).json();
  expect(view.nv.job.state).toBe("done");
  expect(view.report.status).toBe("ready");
  expect(view.nv.read.status).toBe("ready");

  await page.goto(`/workspace.html?scan=${encodeURIComponent(scan.id)}`);
  await page.getByRole("tab", { name: "Readers" }).click();
  await expect(page.locator("#findings")).toBeHidden();
  const readers = page.locator('#readers[data-ready="true"]');
  await expect(readers).toBeVisible();

  // The expected rows come from the same pure functions the page uses, applied to what the server reports.
  const at50 = arrange(view.rows, { threshold: 50 });
  await expect(readers.locator(".rrow.flagged")).toHaveCount(at50.flagged.length);
  await expect(readers.locator('.rrow[data-row="ascites"]')).toHaveClass(/flagged/);
  await expect(readers.locator('.rrow[data-row="ascites"] [data-reader="report"]')).toHaveAttribute("data-state", "present");
  await expect(readers.locator('.rrow[data-row="ascites"] [data-reader="nv"]')).toHaveAttribute("data-state", "absent");
  await expect(readers.locator('.rrow[data-row="ascites"] [data-reader="radar"]')).toHaveAttribute("data-state", "not_scored");
  await expect(readers.getByText(foldText(at50.folded.length))).toBeVisible();

  await page.waitForFunction(() => Boolean(window.__radar?.setThreshold)); // the tab can render before the page tools are registered
  // The display line moves RADAR's dots.
  await page.evaluate(() => window.__radar.setThreshold(1));
  await expect(readers.locator(".rrow.flagged")).toHaveCount(arrange(view.rows, { threshold: 1 }).flagged.length);
  await page.evaluate(() => window.__radar.setThreshold(50));
  // The display line sits in the Compare section, not pinned above the tab, and stays in step with the page tools.
  await expect(page.locator(".threshold")).toBeHidden();
  await readers.getByRole("slider", { name: "Display line" }).fill("1");
  await expect(readers.locator(".rrow.flagged")).toHaveCount(arrange(view.rows, { threshold: 1 }).flagged.length);
  expect(await page.evaluate(() => window.__radar.viewState().threshold_pct)).toBe(1);
  await page.evaluate(() => window.__radar.setThreshold(50));
  await expect(readers.getByRole("slider", { name: "Display line" })).toHaveValue("50");

  // Each reader in its own words.
  await readers.locator('.rrow[data-row="ascites"] .rexpand').click();
  await expect(readers.locator(".rdetails")).toContainText('"Small volume ascites in the pelvis."');
  await expect(readers.locator(".rdetails")).toContainText('"No ascites."');
  await readers.getByRole("button", { name: /Other mentions \(1\)/ }).click();
  await expect(readers.getByText('"Degenerative change of the lumbar spine."')).toBeVisible();

  // A row with an organ jumps the viewer.
  await page.waitForFunction(() => window.__radar?.viewState().mask_loaded === true, null, { timeout: 60_000 });
  await readers.locator('.rrow[data-row="liver_lesion"] .rtitle').click();
  expect(await page.evaluate(() => window.__radar.viewState().active_organ)).toBe("Liver");

  // NV's view draws the crop box.
  await page.getByRole("button", { name: "NV's view" }).click();
  await expect(page.getByRole("button", { name: "NV's view" })).toHaveAttribute("aria-pressed", "true");
  expect(await page.evaluate(() => window.__radar.viewState().nv_view)).toBe(true);

  // Legend items are separate, and the NV column and button exist because the seeded NV job is done.
  await expect(readers.locator(".rkey > span")).toHaveCount(6);
  await expect(readers.locator('.rrow [data-reader="nv"]').first()).toBeVisible();

  // Back to RADAR's list.
  await page.getByRole("tab", { name: "RADAR findings" }).click();
  await expect(page.locator('#findings[data-ready="true"] .finding')).toHaveCount(146, { timeout: 30_000 });
});

test("readers tab: reports and questions are visible on open, a question is answered in place", async ({ page }) => {
  await login(page);
  const scan = await seededScan(page);
  const view = await (await page.request.get(`/scans/${scan.id}/readers`)).json();
  const q = view.rows.find((r) => r.id === "ascites").queries.find((x) => x.kind === "yes_no");
  expect(q, "a seeded yes/no question for ascites").toBeTruthy();
  expect(view.nv.report.answer).toContain("IMPRESSION:");
  expect(view.report.text).toContain("Small volume ascites");

  await page.goto(`/workspace.html?scan=${encodeURIComponent(scan.id)}`);
  await page.getByRole("tab", { name: "Readers" }).click();
  const readers = page.locator('#readers[data-ready="true"]');
  await expect(readers.getByText("Three readers of the same scan")).toBeVisible();

  // Reports: NV's impression in full, its findings and reasoning folded, never the answer tag; the radiologist's text.
  const nv = readers.locator('[data-card="nv"]');
  await expect(nv.locator('[data-part="impression"]')).toContainText("2. Left renal cyst.");
  await expect(nv.locator('[data-part="impression"]')).not.toContainText("<answer>");
  await expect(nv).not.toContainText("answer>");
  await expect(nv.getByText("Liver: A small hypodense lesion")).toBeHidden();
  await nv.getByText("Findings", { exact: true }).click();
  await expect(nv.getByText("Liver: A small hypodense lesion")).toBeVisible();
  await expect(nv.getByText("Fake thinking.")).toBeHidden();
  await expect(nv).toContainText("Research model. It saw a 384 mm cube of the scan.");
  await expect(readers.locator('[data-part="radiologist"]')).toContainText("Small volume ascites in the pelvis.");

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

  // The same question in its row shows the same answer; the dot stays the report read.
  await readers.locator('.rrow[data-row="ascites"] .rexpand').click();
  await expect(readers.locator(`.rdetails .rq[data-q="${q.id}"] .rqa`)).toContainText("NV: Yes.");
  await expect(readers.locator('.rrow[data-row="ascites"] [data-reader="nv"]')).toHaveAttribute("data-state", "absent");
});
