import { resolve } from "node:path";
import { defineConfig } from "vite";

const API = "http://127.0.0.1:8000";
const proxied = ["/auth", "/scans", "/jobs", "/uploads", "/catalog", "/fixtures", "/gpu", "/export", "/chat", "/_storage"];

export default defineConfig({
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 2500, // NiiVue alone is about 1.5 MB
    rollupOptions: {
      input: {
        index: resolve(import.meta.dirname, "index.html"),
        workspace: resolve(import.meta.dirname, "workspace.html"),
        jobs: resolve(import.meta.dirname, "jobs.html"),
      },
    },
  },
  server: {
    // Anchored patterns, so /jobs does not swallow /jobs.html.
    proxy: Object.fromEntries(proxied.map((p) => [`^${p}(?:[/?]|$)`, { target: API, changeOrigin: false }])),
  },
});
