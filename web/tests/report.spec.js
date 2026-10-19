// The report in the scan bar: add, replace from a .txt file, remove. Runs against an isolated server only.
import { expect, test } from "@playwright/test";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";

async function login(page) {
  await page.goto("/");
  const res = await page.request.post("/auth/login", { data: { token: TOKEN } });
  expect(res.status(), "login").toBeLessThan(300);
}

test("report: add, replace from a file, remove", async ({ page }) => {
  await login(page);
  const { scans } = await (await page.request.get("/scans")).json();
  const scan = scans.find((s) => s.state === "ready");
  expect(scan, "a ready scan (seed one with scripts/seed_dev.py)").toBeTruthy();
  const before = await page.request.get(`/scans/${scan.id}/report`);
  const original = before.ok() ? (await before.json()).text : null; // put back at the end for the readers spec
  await page.request.delete(`/scans/${scan.id}/report`);
  await page.goto(`/workspace.html?scan=${encodeURIComponent(scan.id)}`);

  await page.getByRole("button", { name: "Add report" }).click();
  await page.locator("#report-text").fill("FINDINGS: Synthetic test report. No ascites.");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("button", { name: "Report attached" })).toBeVisible();
  let stored = await (await page.request.get(`/scans/${scan.id}/report`)).json();
  expect(stored.text).toBe("FINDINGS: Synthetic test report. No ascites.");

  await page.getByRole("button", { name: "Report attached" }).click();
  await page.locator("#report-file").setInputFiles({ name: "report.txt", mimeType: "text/plain", buffer: Buffer.from("﻿Replaced report.\r\nSecond line.\r\n") });
  await expect(page.locator("#report-text")).toHaveValue(/Replaced report\./);
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("button", { name: "Report attached" })).toBeVisible();
  stored = await (await page.request.get(`/scans/${scan.id}/report`)).json();
  expect(stored.text).toBe("Replaced report.\nSecond line.");

  await page.getByRole("button", { name: "Report attached" }).click();
  await page.getByRole("button", { name: "Remove report" }).click();
  await expect(page.getByRole("button", { name: "Add report" })).toBeVisible();
  expect((await page.request.get(`/scans/${scan.id}/report`)).status()).toBe(404);
  if (original != null) await page.request.put(`/scans/${scan.id}/report`, { data: { text: original } });
});
