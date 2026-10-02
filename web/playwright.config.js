import os from "node:os";
import path from "node:path";
import { defineConfig } from "@playwright/test";

// The smoke runs against the real API with the fake GPU backend. Without RADAR_BASE_URL, Playwright
// starts the server on port 8000 (or reuses one already listening there). The data folder sits under
// the OS temp dir, and the seed runs before the server starts, because the fake backend keeps its
// calls in memory and a running poller would fail a job seeded from another process.
// The server runs from web/, so the project's .env is not read.
const baseURL = process.env.RADAR_BASE_URL || "http://127.0.0.1:8000";
const dataDir = path.join(os.tmpdir(), "radar-desk-playwright");
const python = "../.venv/bin/python";

export default defineConfig({
  testDir: "tests",
  timeout: 90_000,
  use: {
    channel: "chrome",
    baseURL,
  },
  webServer: process.env.RADAR_BASE_URL
    ? undefined
    : {
        command: `${python} ../scripts/seed_dev.py --if-empty && ${python} -m radar_desk`,
        url: "http://127.0.0.1:8000/health",
        reuseExistingServer: true,
        timeout: 60_000,
        env: {
          OWNER_TOKEN: process.env.RADAR_OWNER_TOKEN || "dev-token",
          SESSION_SECRET: "dev-secret",
          DATA_DIR: dataDir,
          GPU_BACKEND: "fake",
          PUBLIC_BASE_URL: "http://127.0.0.1:8000",
          PORT: "8000",
        },
      },
});
