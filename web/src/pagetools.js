// The twelve WebMCP page tools. They only read or move the view.
import { WebMcpBridge } from "@runtypelabs/persona";
import { COLORMAPS, WINDOWS } from "./light.js";
import { ORGAN_COLOURS } from "./palette.js";

export const PAGE_TOOL_NAMES = [
  "get_view_state",
  "jump_to_organ",
  "select_finding",
  "set_window",
  "set_threshold",
  "show_scoring_box",
  "toggle_mask",
  "set_window_level",
  "set_zoom",
  "toggle_focus",
  "set_view_for",
  "set_light",
];

const ORGANS = Object.keys(ORGAN_COLOURS);
const text = (value) => ({ content: [{ type: "text", text: JSON.stringify(value) }] });

// ctx: the workspace's actions. Each returns plain data.
function definitions(ctx) {
  return [
    {
      name: "get_view_state",
      description:
        "Call this first for questions like 'what am I looking at?'. Returns the plane on screen and the slice as {axis, index, number, count}; " +
        "index counts from 0, so tell the user slice number of count. In multiplanar the plane and slice are the main (large) view's, main_plane names it and " +
        "reference_planes lists the two small views; the user can click a reference view to make it the main one. In the 3D view the slice is the axial one through the crosshair. " +
        "Also returns " +
        "the organs RADAR outlined on that slice with their share of the outlined pixels (largest first, and whether each is one of the 18 scored organs), " +
        "what is under the crosshair (mm, HU, mask label and organ), and the nearest outlined organ with its in-plane distance when the crosshair is on background. " +
        "Also scan_id, job_id, active organ and finding, display threshold, whether the mask is shown, the window (window_preset, or 'custom', " +
        "and window_hu {min, max, width, level} in HU), gamma, invert and colormap of the CT, zoom (1 means the whole slice fits) and focus " +
        "(true when the viewer fills the window and the chat is hidden).",
      inputSchema: { type: "object", properties: {} },
      execute: () => ctx.viewState(),
    },
    {
      name: "jump_to_organ",
      description: "Move the crosshair to the centroid of an organ's mask, show only that organ, and scroll the findings list to it.",
      inputSchema: {
        type: "object",
        properties: { organ: { type: "string", enum: ORGANS, description: "One of the 18 scored organs" } },
        required: ["organ"],
      },
      execute: ({ organ }) => ctx.jumpToOrgan(organ),
    },
    {
      name: "select_finding",
      description: "Select a finding as if the user clicked it in the list: jump to its organ and show it. Pass the upstream key, 'Organ_Finding', or the finding name.",
      inputSchema: { type: "object", properties: { key: { type: "string" } }, required: ["key"] },
      execute: ({ key }) => ctx.selectFinding(key),
    },
    {
      name: "set_window",
      description: "Set the CT window preset.",
      inputSchema: {
        type: "object",
        properties: { preset: { type: "string", enum: Object.keys(WINDOWS) } },
        required: ["preset"],
      },
      execute: ({ preset }) => ctx.setWindow(preset),
    },
    {
      name: "set_window_level",
      description: "Set the CT window by width and level in HU, for windows the five presets do not cover. Width 1 to 4000, level -1200 to 2000; values outside are clamped.",
      inputSchema: {
        type: "object",
        properties: {
          width: { type: "number", minimum: 1, maximum: 4000, description: "Window width in HU, for example 400" },
          level: { type: "number", minimum: -1200, maximum: 2000, description: "Window level (centre) in HU, for example 40" },
        },
        required: ["width", "level"],
      },
      execute: ({ width, level }) => ctx.setWindowLevel({ width, level }),
    },
    {
      name: "set_zoom",
      description:
        "Zoom the 2D views, keeping the crosshair where it is on screen. 1 means the whole slice fits; 2 is twice as large; 0.25 to 16. " +
        "To zoom in on an organ, call jump_to_organ first so the crosshair is on it, then set_zoom.",
      inputSchema: { type: "object", properties: { zoom: { type: "number", minimum: 0.25, maximum: 16 } }, required: ["zoom"] },
      execute: ({ zoom }) => ctx.setZoom(zoom),
    },
    {
      name: "toggle_focus",
      description: "Focus mode: the viewer fills the window and the findings list and this chat are hidden until the user presses Escape or f. Omit 'on' to toggle.",
      inputSchema: { type: "object", properties: { on: { type: "boolean" } } },
      execute: ({ on } = {}) => ctx.toggleFocus(on),
    },
    {
      name: "set_view_for",
      description:
        "Show an organ or a finding the way it is usually read: jump to it (a finding is also selected in the list), set the window from the viewing recipes table, " +
        "centre it at the recipe's zoom and show its scoring box. Pass one of the 18 scored organs, or a finding by upstream key, 'Organ_Finding' or name. " +
        "Returns the window set, why that window, and ok false with a reason when the organ was not found or there is no result yet.",
      inputSchema: {
        type: "object",
        properties: { target: { type: "string", description: "A scored organ such as 'Pancreas', or a finding" } },
        required: ["target"],
      },
      execute: ({ target }) => ctx.setViewFor(target),
    },
    {
      name: "set_light",
      description:
        "Change how the CT is drawn: gamma, invert and colour map (the mask keeps its colours). Give at least one field. reset goes back to the soft tissue window, " +
        "gamma 1, no invert and gray before the other fields apply. Colour maps are a spotting aid, not the reading standard.",
      inputSchema: {
        type: "object",
        properties: {
          gamma: { type: "number", minimum: 0.2, maximum: 3, description: "1 is linear; above 1 brightens mid-greys" },
          invert: { type: "boolean" },
          colormap: { type: "string", enum: COLORMAPS },
          reset: { type: "boolean" },
        },
      },
      execute: (input = {}) => ctx.setLight(input),
    },
    {
      name: "set_threshold",
      description: "Move the display line in the findings list. It is for display only; RADAR ships no calibrated thresholds.",
      inputSchema: {
        type: "object",
        properties: { prob: { type: "number", minimum: 0, maximum: 1, description: "Probability between 0 and 1, for example 0.8" } },
        required: ["prob"],
      },
      execute: ({ prob }) => ctx.setThreshold(prob * 100),
    },
    {
      name: "show_scoring_box",
      description: "Show or hide the outline of the window or centred crop in which RADAR scored an organ.",
      inputSchema: {
        type: "object",
        properties: { organ: { type: "string", enum: ORGANS }, on: { type: "boolean" } },
        required: ["organ", "on"],
      },
      execute: ({ organ, on }) => ctx.showScoringBox(organ, on),
    },
    {
      name: "toggle_mask",
      description: "Show or hide RADAR's organ mask overlay. Omit 'on' to toggle.",
      inputSchema: { type: "object", properties: { on: { type: "boolean" } } },
      execute: ({ on } = {}) => ctx.toggleMask(on),
    },
  ];
}

// Installs the WebMCP polyfill (persona's bridge does it on first snapshot) and
// registers the tools on document.modelContext. Returns name -> execute for tests.
export async function registerPageTools(ctx) {
  const tools = definitions(ctx).map((t) => ({
    ...t,
    execute: async (input) => {
      try {
        return text(await t.execute(input || {}));
      } catch (e) {
        return { isError: true, content: [{ type: "text", text: e.message }] };
      }
    },
  }));
  // This bridge exists only to install the polyfill before we register; the install is idempotent.
  await new WebMcpBridge({ enabled: true }).snapshotForDispatch();
  const mc = document.modelContext;
  if (mc?.registerTool) {
    for (const tool of tools) await mc.registerTool(tool);
  } else {
    console.warn("WebMCP is not available on this page; the chat cannot move the viewer.");
  }
  return Object.fromEntries(tools.map((t) => [t.name, t.execute]));
}
