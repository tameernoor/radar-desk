// The seven WebMCP page tools. They only read or move the view.
import { WebMcpBridge } from "@runtypelabs/persona";
import { ORGAN_COLOURS } from "./palette.js";

export const PAGE_TOOL_NAMES = [
  "get_view_state",
  "jump_to_organ",
  "select_finding",
  "set_window",
  "set_threshold",
  "show_scoring_box",
  "toggle_mask",
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
        "index counts from 0, so tell the user slice number of count. In multiplanar and 3D views the slice is the axial one through the crosshair. " +
        "Also returns " +
        "the organs RADAR outlined on that slice with their share of the outlined pixels (largest first, and whether each is one of the 18 scored organs), " +
        "what is under the crosshair (mm, HU, mask label and organ), and the nearest outlined organ with its in-plane distance when the crosshair is on background. " +
        "Also scan_id, job_id, active organ and finding, display threshold, window preset and whether the mask is shown.",
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
        properties: { preset: { type: "string", enum: ["soft_tissue", "liver", "bone", "lung"] } },
        required: ["preset"],
      },
      execute: ({ preset }) => ctx.setWindow(preset),
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
