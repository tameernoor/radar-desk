// CT light (window, gamma, invert, colour map) and zoom maths. Pure: no DOM, no NiiVue.

export const WINDOWS = {
  soft_tissue: { label: "Soft tissue", min: -160, max: 240 },
  liver: { label: "Liver", min: -20, max: 160 },
  bone: { label: "Bone", min: -450, max: 1050 },
  lung: { label: "Lung", min: -1350, max: 150 },
};

export const COLORMAPS = ["gray", "inferno", "viridis", "hot"];

export const LIGHT_DEFAULTS = Object.freeze({ min: WINDOWS.soft_tissue.min, max: WINDOWS.soft_tissue.max, gamma: 1, invert: false, colormap: "gray" });

const WIDTH_RANGE = [1, 4000];
const LEVEL_RANGE = [-1200, 2000];
const GAMMA_RANGE = [0.2, 3];
const SLIDER_MAX = 1000;
export const ZOOM_RANGE = [0.25, 16];
const ZOOM_STEP = 1.25;

const clamp = (v, [lo, hi]) => Math.max(lo, Math.min(hi, v));

export const windowOf = (min, max) => ({ width: max - min, level: (min + max) / 2 });
export const rangeOf = (width, level) => ({ min: level - width / 2, max: level + width / 2 });

export function clampWindow({ width, level }) {
  return { width: clamp(width, WIDTH_RANGE), level: clamp(level, LEVEL_RANGE) };
}

// The width slider is log-scaled so narrow windows get as much travel as wide ones.
const LOG_SPAN = Math.log(WIDTH_RANGE[1] / WIDTH_RANGE[0]);
export const widthFromSlider = (s) => Math.round(WIDTH_RANGE[0] * Math.exp((clamp(s, [0, SLIDER_MAX]) / SLIDER_MAX) * LOG_SPAN));
export const sliderFromWidth = (w) => Math.round((Math.log(clamp(w, WIDTH_RANGE) / WIDTH_RANGE[0]) / LOG_SPAN) * SLIDER_MAX);

// The preset whose range this is exactly, or "custom".
export function presetFor(min, max) {
  return Object.keys(WINDOWS).find((k) => WINDOWS[k].min === min && WINDOWS[k].max === max) ?? "custom";
}

// state: {min, max, gamma, invert, colormap}. Returns a new state; throws on bad input.
export function lightReducer(state, action) {
  switch (action.type) {
    case "preset": {
      const w = WINDOWS[action.name];
      if (!w) throw new Error(`Unknown window preset: ${action.name}`);
      return { ...state, min: w.min, max: w.max };
    }
    case "window": {
      if (!Number.isFinite(action.min) || !Number.isFinite(action.max)) throw new Error("A window needs finite min and max in HU.");
      const { width, level } = clampWindow(windowOf(action.min, action.max));
      return { ...state, ...rangeOf(width, level) };
    }
    case "gamma":
      if (!Number.isFinite(action.value)) throw new Error("Gamma must be a number.");
      return { ...state, gamma: clamp(action.value, GAMMA_RANGE) };
    case "invert":
      return { ...state, invert: Boolean(action.value) };
    case "colormap":
      if (!COLORMAPS.includes(action.name)) throw new Error(`Unknown colour map: ${action.name}. Use one of ${COLORMAPS.join(", ")}.`);
      return { ...state, colormap: action.name };
    case "reset":
      return { ...LIGHT_DEFAULTS };
    default:
      throw new Error(`Unknown light action: ${action.type}`);
  }
}

// One zoom step in or out (direction 1 or -1), rounded to two decimals and clamped.
export function nextZoom(zoom, direction) {
  const z = direction > 0 ? zoom * ZOOM_STEP : zoom / ZOOM_STEP;
  return clamp(Math.round(z * 100) / 100, ZOOM_RANGE);
}

// NiiVue's own wheel zoom keeps the crosshair in place this way (WheelController.ts 199-203).
// pan: [x, y, z, zoom] in mm; returns the new pan2Dxyzmm.
export function zoomAround(pan, newZoom, crosshairMm) {
  const change = pan[3] - newZoom;
  return [pan[0] + change * crosshairMm[0], pan[1] + change * crosshairMm[1], pan[2] + change * crosshairMm[2], newZoom];
}
