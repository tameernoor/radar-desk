import { expect, test } from "@playwright/test";

const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";

async function login(page) {
  await page.goto("/");
  const res = await page.request.post("/auth/login", { data: { token: TOKEN } });
  expect(res.status(), "login").toBeLessThan(300);
}

test("compute: the fake backend cannot be switched", async ({ page }) => {
  await login(page);
  const res = await page.request.get("/compute");
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(body.mode).toBe("fake");
  expect(body.changeable).toBe(false);

  await page.goto("/jobs.html");
  const section = page.getByRole("region", { name: "Compute" });
  await expect(section.getByText("The fake backend is chosen at start-up.")).toBeVisible();
  const radios = section.getByRole("radio");
  await expect(radios).toHaveCount(3);
  await expect(radios.nth(0)).toBeDisabled();
  await expect(radios.nth(1)).toBeDisabled();
  await expect(radios.nth(2)).toBeDisabled();
  // The fake backend runs on local storage with no RunPod keys.
  await expect(section.locator("#compute-storage")).toHaveText("Storage: Local folder (used by all modes)");
  const labels = section.locator("label");
  await expect(labels.nth(0)).toHaveText("Modal (not on local storage)");
  await expect(labels.nth(0)).toHaveAttribute("title", "Modal cannot reach a local folder; this app stores scans under DATA_DIR");
  await expect(labels.nth(1)).toHaveText("Worker");
  await expect(labels.nth(1)).not.toHaveAttribute("title");
  await expect(labels.nth(2)).toHaveText("RunPod serverless (needs a shared bucket)");
  await expect(labels.nth(2)).toHaveAttribute(
    "title",
    "Serverless needs a shared bucket or the RunPod volume; this app stores scans in a local folder",
  );
  await expect(section.locator("#compute-line")).toHaveText(/^Fake backend/);
  await expect(section.getByRole("button", { name: "Start now" })).toBeHidden();
  await expect(section.getByRole("button", { name: "Stop now" })).toBeHidden();
});
