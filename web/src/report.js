// The radiologist report attached to a scan: add, replace or remove it from the scan bar.
import { api, ApiError, del, get } from "./api.js";
import { el } from "./format.js";

export const REPORT_MAX_CHARS = 100_000;

export function mountReport({ scanId, onChange }) {
  const $ = (id) => document.getElementById(id);
  const path = `/scans/${encodeURIComponent(scanId)}/report`;
  const dialog = $("report-dialog");
  const text = $("report-text");
  const error = $("report-error");
  let current = null;

  function renderBar() {
    $("report-bar").replaceChildren(el("button", { type: "button", class: "link report-open", onclick: openDialog }, current == null ? "Add report" : "Report attached"));
  }

  function openDialog() {
    text.value = current ?? "";
    error.textContent = "";
    $("report-remove").hidden = current == null;
    dialog.showModal();
    text.focus();
  }

  function changed(next) {
    current = next;
    dialog.close();
    renderBar();
    onChange?.();
  }

  $("report-file").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (file) text.value = await file.text();
    e.target.value = "";
  });
  $("report-save").addEventListener("click", async () => {
    const value = text.value.trim();
    if (!value) return (error.textContent = "The report is empty.");
    if (value.length > REPORT_MAX_CHARS) return (error.textContent = "The report is longer than 100,000 characters.");
    try {
      changed((await api(path, { method: "PUT", body: { text: value } })).text);
    } catch (e) {
      error.textContent = `Could not save: ${e.message}`;
    }
  });
  $("report-remove").addEventListener("click", async () => {
    try {
      await del(path);
      changed(null);
    } catch (e) {
      error.textContent = `Could not remove: ${e.message}`;
    }
  });
  $("report-cancel").addEventListener("click", () => dialog.close());

  return {
    open: openDialog,
    async load() {
      try {
        current = (await get(path)).text;
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 404)) throw e;
        current = null;
      }
      renderBar();
    },
    get text() {
      return current;
    },
  };
}
