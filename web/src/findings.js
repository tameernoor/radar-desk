// Left pane: organ strip, threshold, and the 146 findings in DAMO order.
import { el, pct } from "./format.js";
import { organCss } from "./palette.js";

const HOW_TEXT = { window: "scored in a window", centered_crop: "scored by centred crop" };

function howText(how) {
  if (!how) return "";
  if (how.how === "window" && how.window_index != null) return `scored in window ${how.window_index}`;
  return HOW_TEXT[how.how] || how.how;
}

export function createFindings({ onSelect, onOrgan }) {
  const listEl = document.getElementById("findings");
  const stripEl = document.getElementById("organ-strip");
  const sortEl = document.getElementById("sort");
  const positivesEl = document.getElementById("positives-only");
  const thresholdEl = document.getElementById("threshold");
  const thresholdOut = document.getElementById("threshold-value");

  const state = {
    catalog: [], // 146 {key, organ, finding, index}
    scoredOrgans: [], // 18 {organ, label, finding_count}
    result: null,
    compare: null,
    threshold: 50, // percent
    sort: "organ",
    positivesOnly: false,
    active: null, // finding key
  };

  const probOf = (key) => state.result?.findings.find((f) => f.key === key)?.prob ?? null;
  const isPositive = (p) => p != null && p * 100 >= state.threshold;
  const notFound = () => new Set(state.result?.organs_not_found || []);
  const howOf = (organ) => state.result?.organs_scored.find((o) => o.organ === organ) || null;

  function deltaCell(key) {
    const c = state.compare;
    if (!c || c.pending) return null;
    const d = c.deltas.find((x) => x.key === key);
    if (!d) return el("span", { class: "ref muted" }, "–");
    const over = d.delta != null && Math.abs(d.delta) > c.tolerance;
    const sign = d.delta > 0 ? "+" : "";
    return el(
      "span",
      { class: `ref${over ? " over" : ""}`, title: `Reference (${c.source}): ${pct(d.reference, 2)}, delta ${d.delta == null ? "–" : sign + d.delta.toFixed(4)}` },
      `${pct(d.reference)} `,
      el("small", {}, d.delta == null ? "" : `${sign}${(d.delta * 100).toFixed(2)} pt`),
    );
  }

  function row(f) {
    const p = probOf(f.key);
    const positive = isPositive(p);
    const node = el(
      "div",
      {
        class: `finding${positive ? " positive" : ""}${state.active === f.key ? " active" : ""}`,
        role: "button",
        tabindex: "-1",
        "data-key": f.key,
        "data-organ": f.organ,
        "data-prob": p == null ? "" : String(p),
        onclick: () => onSelect(f.key),
      },
      el("span", { class: "dot", style: `background:${organCss(f.organ)}` }),
      el("span", { class: "name", title: f.key }, f.finding),
      el("span", { class: "bar" }, el("span", { style: `width:${p == null ? 0 : (p * 100).toFixed(1)}%;background:${organCss(f.organ, 0.85)}` })),
      el("span", { class: "score" }, p == null ? "–" : pct(p)),
      deltaCell(f.key),
    );
    if (state.positivesOnly && !positive) node.hidden = true;
    return node;
  }

  function organHeader(organ, extra) {
    const how = howOf(organ);
    return el(
      "div",
      { class: "organ-head", "data-organ-head": organ, onclick: () => onOrgan(organ, { toggle: false }) },
      el("span", { class: "dot", style: `background:${organCss(organ)}` }),
      el("strong", {}, organ),
      el("span", { class: "muted small" }, extra || howText(how)),
    );
  }

  // With the positives filter on, an organ with no visible rows loses its header too.
  function hideIfEmpty(group) {
    group.hidden = !group.querySelector(".finding:not([hidden])");
  }

  function renderList() {
    const missing = notFound();
    const nodes = [];
    if (state.compare && !state.compare.pending) {
      const c = state.compare;
      nodes.push(
        el("p", { class: "compare-note muted small" }, `Reference column: ${c.source}, tolerance ${c.tolerance}. Max delta ${c.max_abs_delta?.toFixed(4) ?? "–"}, ${c.over_tolerance} over.`),
      );
    }
    if (state.sort === "score") {
      const scored = state.catalog.filter((f) => !missing.has(f.organ));
      scored.sort((a, b) => (probOf(b.key) ?? -1) - (probOf(a.key) ?? -1));
      nodes.push(...scored.map(row));
    } else {
      const organs = [...new Set(state.catalog.map((f) => f.organ))].filter((o) => !missing.has(o));
      for (const organ of organs) {
        const group = el("div", { class: "organ-group", "data-group": organ }, organHeader(organ));
        group.append(...state.catalog.filter((f) => f.organ === organ).map(row));
        hideIfEmpty(group);
        nodes.push(group);
      }
    }
    if (missing.size) {
      const section = el("div", { class: "not-scored" }, el("h3", {}, "Not scored"));
      for (const organ of missing) {
        const count = state.catalog.filter((f) => f.organ === organ).length;
        const group = el(
          "div",
          { class: "organ-group", "data-group": organ },
          organHeader(organ, `RADAR's segmentation found no ${organ.toLowerCase()} in this scan, so its ${count === 1 ? "finding has" : `${count} findings have`} no score.`),
        );
        group.append(...state.catalog.filter((f) => f.organ === organ).map(row));
        hideIfEmpty(group);
        section.append(group);
      }
      section.hidden = ![...section.querySelectorAll(".organ-group")].some((g) => !g.hidden);
      nodes.push(section);
    }
    listEl.replaceChildren(...nodes);
    listEl.dataset.ready = state.result ? "true" : "false";
  }

  function renderStrip() {
    const missing = notFound();
    stripEl.replaceChildren(
      ...state.scoredOrgans.map(({ organ }) => {
        const how = howOf(organ);
        const status = !state.result ? "pending" : missing.has(organ) ? "missing" : how?.how === "centered_crop" ? "crop" : "window";
        const label = { pending: "not scored yet", missing: "not found", crop: HOW_TEXT.centered_crop, window: HOW_TEXT.window }[status];
        return el(
          "button",
          { type: "button", class: `organ-chip status-${status}`, title: `${organ}: ${label}`, "data-organ": organ, onclick: () => onOrgan(organ) },
          el("span", { class: "dot", style: `background:${organCss(organ)}` }),
          organ,
        );
      }),
    );
  }

  function render() {
    renderStrip();
    renderList();
  }

  function setThreshold(percent) {
    state.threshold = Math.max(0, Math.min(100, Math.round(percent)));
    thresholdEl.value = String(state.threshold);
    thresholdOut.textContent = `${state.threshold}%`;
    renderList();
  }

  thresholdEl.addEventListener("input", () => setThreshold(Number(thresholdEl.value)));
  sortEl.addEventListener("change", () => {
    state.sort = sortEl.value;
    renderList();
  });
  positivesEl.addEventListener("change", () => {
    state.positivesOnly = positivesEl.checked;
    renderList();
  });

  return {
    state,
    setCatalog(findings, scoredOrgans) {
      state.catalog = findings;
      state.scoredOrgans = scoredOrgans;
      render();
    },
    setResult(result) {
      state.result = result;
      render();
    },
    setCompare(compare) {
      state.compare = compare;
      renderList();
    },
    setThreshold,
    setActive(key) {
      state.active = key;
      for (const node of listEl.querySelectorAll(".finding.active")) node.classList.remove("active");
      const node = key && listEl.querySelector(`.finding[data-key="${CSS.escape(key)}"]`);
      if (node) {
        node.hidden = false;
        for (let up = node.parentElement; up && up !== listEl; up = up.parentElement) up.hidden = false;
        node.classList.add("active");
        node.scrollIntoView({ block: "nearest" });
      }
    },
    markCurrentOrgan(organ) {
      for (const chip of stripEl.querySelectorAll(".organ-chip")) chip.classList.toggle("current", chip.dataset.organ === organ);
    },
    clearCurrentOrgan() {
      for (const chip of stripEl.querySelectorAll(".organ-chip.current")) chip.classList.remove("current");
    },
    scrollToOrgan(organ) {
      listEl.querySelector(`[data-group="${CSS.escape(organ)}"]`)?.scrollIntoView({ block: "start", behavior: "smooth" });
      for (const chip of stripEl.querySelectorAll(".organ-chip")) chip.classList.toggle("current", chip.dataset.organ === organ);
    },
    // Keys in the order they are shown, for j/k.
    visibleKeys() {
      return [...listEl.querySelectorAll(".finding:not([hidden])")].map((n) => n.dataset.key);
    },
    // By upstream key, by "Organ_Finding", or by finding name when that name is unique.
    finding(keyOrName) {
      const q = String(keyOrName).trim().toLowerCase();
      const exact = state.catalog.find((f) => f.key === keyOrName || f.english?.toLowerCase() === q);
      if (exact) return exact;
      const byName = state.catalog.filter((f) => f.finding.toLowerCase() === q);
      return byName.length === 1 ? byName[0] : null;
    },
    probOf,
  };
}
