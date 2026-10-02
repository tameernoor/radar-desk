import { showLogin } from "./login.js";

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function detailOf(res) {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    return res.statusText;
  }
}

// JSON fetch against the same-origin API. A 401 shows the login screen.
export async function api(path, { method = "GET", body } = {}) {
  const init = { method, credentials: "same-origin", headers: { Accept: "application/json" } };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  if (res.status === 401) {
    showLogin();
    throw new ApiError(401, "Not logged in");
  }
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  if (res.status === 204) return null;
  const type = res.headers.get("Content-Type") || "";
  return type.includes("json") ? res.json() : res.text();
}

export const get = (path) => api(path);
export const post = (path, body) => api(path, { method: "POST", body: body ?? {} });
export const del = (path) => api(path, { method: "DELETE" });
