import { expect, test } from "@playwright/test";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";

async function login(page) {
  await page.goto("/");
  const res = await page.request.post("/auth/login", { data: { token: TOKEN } });
  expect(res.status(), "login").toBeLessThan(300);
}

test("workers: create a token, copy the run line, revoke", async ({ page }) => {
  await login(page);
  await page.goto("/jobs.html");
  const section = page.getByRole("region", { name: "Workers" });

  await section.getByLabel("Token name").fill("pod-1");
  await section.getByRole("button", { name: "Create token" }).click();
  await expect(section.locator("#token-plain")).toHaveText(/^rdw_/);
  await expect(section.getByText("Shown once. Store it now.")).toBeVisible();

  const runLine = section.locator("#run-line");
  await expect(runLine).toContainText("RADAR_WORKER_TOKEN=rdw_");
  await expect(runLine).toContainText("-v /workspace:/workspace");

  // Earlier runs may have left tokens named pod-1; the newest one is ours.
  const row = section.locator("#tokens-table tbody tr", { hasText: "pod-1" }).filter({ hasText: "active" }).last();
  await expect(row).toBeVisible();
  const id = await row.getAttribute("data-token");
  const ours = section.locator(`#tokens-table tr[data-token="${id}"]`);
  await ours.getByRole("button", { name: "Revoke" }).click();
  await ours.getByRole("button", { name: "Revoke token?" }).click();
  await expect(ours).toContainText("revoked");
  await expect(ours.getByRole("button")).toHaveCount(0);

  await expect(section.getByText("No worker has reported in yet.")).toBeVisible();
});
