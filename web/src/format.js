export function bytes(n) {
  if (n == null) return "–";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export const dims = (h) => (h ? h.dims.join(" × ") : "–");
export const spacing = (h) => (h ? h.spacing_mm.map((s) => s.toFixed(2)).join(" × ") : "–");
export const pct = (p, digits = 1) => (p == null ? "–" : `${(p * 100).toFixed(digits)}%`);
export const usd = (n) => (n == null ? "–" : `$${n.toFixed(n < 1 ? 3 : 2)}`);
export const shortId = (id) => (id ? String(id).slice(0, 8) : "–");

export function when(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });
}

export function elapsed(fromIso, toIso) {
  if (!fromIso) return "";
  const s = Math.max(0, Math.round(((toIso ? new Date(toIso) : new Date()) - new Date(fromIso)) / 1000));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "style") node.style.cssText = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

// The chip shown for a scan: scan state first, then the latest job.
export function scanChip(scan) {
  if (scan.state === "uploading") return "uploading";
  if (scan.state === "rejected") return "rejected";
  const job = scan.latest_job;
  if (!job) return "ready";
  if (job.state === "submitted") return "scoring";
  return job.state; // queued, done, failed, cancelled
}
