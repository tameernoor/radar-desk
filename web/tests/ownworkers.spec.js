import { expect, test } from "@playwright/test";

// The fake backend cannot switch modes, so GET /compute is answered here with a mode-worker or mode-runpod body.
const TOKEN = process.env.RADAR_OWNER_TOKEN || "dev-token";
const open = { available: true, reason: null, note: null };

const body = (mode, configured) => ({
  mode,
  changeable: true,
  changed_at: null,
  tunnel_mode: "managed",
  public_url: null,
  runpod: { configured, datacenter: null, max_pod_hours: 3, gpus: [], idle_delete_s: 600, app_lost_delete_s: 600 },
  serverless: { configured: false, endpoint_id: null, gpus: [], idle_s: 60, price_per_s: null, health: null, job: null },
  pod: null,
  in_flight: null,
  queued: 0,
  held: [],
  problem: null,
  last_event: null,
  spend_month_usd: 0,
  budget_usd: 10,
  month: "2026-10",
  storage: { backend: "local", name: "Local folder", modes: { modal: open, worker: open, runpod: open, serverless: open } },
});

async function openJobs(page, compute) {
  await page.goto("/");
  const res = await page.request.post("/auth/login", { data: { token: TOKEN } });
  expect(res.status(), "login").toBeLessThan(300);
  await page.route("**/compute", (route) =>
    route.request().method() === "GET" ? route.fulfill({ json: compute }) : route.continue(),
  );
  await page.goto("/jobs.html");
  return page.getByRole("region", { name: "Compute" });
}

// True when the Workers section comes before the Jobs section in the page.
const workersFirst = (page) =>
  page.evaluate(() => {
    const workers = document.querySelector('section[aria-labelledby="workers-title"]');
    const jobs = document.querySelector('section[aria-labelledby="jobs-title"]');
    return Boolean(workers.compareDocumentPosition(jobs) & Node.DOCUMENT_POSITION_FOLLOWING);
  });

test("own workers: the app waits for a worker and the Workers section moves up", async ({ page }) => {
  // RunPod configured and available, so a hidden Start now is down to the mode alone.
  const section = await openJobs(page, body("worker", true));
  const radio = section.getByRole("radio", { name: "Own GPU workers" });
  await expect(radio).toBeChecked();
  await expect(radio).toBeEnabled();
  await expect(section.locator("#compute-line")).toHaveText("Waiting for your workers; start one with the docker run line below");
  await expect(section.getByRole("button", { name: "Start now" })).toBeHidden();
  await expect(section.getByRole("button", { name: "Stop now" })).toBeHidden();
  await expect(page.locator("#workers-intro")).toBeVisible();
  expect(await workersFirst(page)).toBe(true);
});

test("runpod pod: Start now is offered and the Workers section stays below Jobs", async ({ page }) => {
  const section = await openJobs(page, body("runpod", true));
  await expect(section.getByRole("radio", { name: "RunPod pod" })).toBeChecked();
  await expect(section.getByRole("button", { name: "Start now" })).toBeVisible();
  await expect(section.locator("#compute-line")).toHaveText("RunPod pod, none running, starts when a job is queued");
  await expect(page.locator("#workers-intro")).toBeHidden();
  expect(await workersFirst(page)).toBe(false);
});
