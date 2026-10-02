// Multiplanar as one main view plus two small reference views. Pure: no DOM, no NiiVue.

export const PLANES = ["axial", "coronal", "sagittal"];

// The other two planes, in fixed order, are the references.
export function referencePlanes(mainPlane) {
  return PLANES.filter((p) => p !== mainPlane);
}

// Tile fractions [left, top, width, height] of the canvas, main first.
// Wide canvas: main on the left at 75% width and full height, references stacked on the right.
// Narrow canvas: main on top at 75% height and full width, references side by side below,
// so the main view keeps the full width of a tall window instead of shrinking to 75% of it.
export function multiplanarLayout(mainPlane, width, height) {
  const [first, second] = referencePlanes(mainPlane);
  if (width >= height) {
    return [
      { plane: mainPlane, position: [0, 0, 0.75, 1] },
      { plane: first, position: [0.75, 0, 0.25, 0.5] },
      { plane: second, position: [0.75, 0.5, 0.25, 0.5] },
    ];
  }
  return [
    { plane: mainPlane, position: [0, 0, 1, 0.75] },
    { plane: first, position: [0, 0.75, 0.5, 0.25] },
    { plane: second, position: [0.5, 0.75, 0.5, 0.25] },
  ];
}

const BLOCKED_ON_REFERENCE = new Set(["wheel", "mousedown", "mouseup", "contextmenu", "dblclick", "touchstart", "touchmove", "touchend"]);

// What to do with a pointer event before NiiVue sees it.
// "pass" lets NiiVue handle it, "block" stops it, "swap" makes the reference tile's plane the main view,
// "zoom" zooms the 2D views. A wheel with ctrl or cmd (withModifier; a trackpad pinch arrives as
// ctrl+wheel) zooms in every mode except over a reference tile, where it is blocked like a plain wheel.
// Outside multiplanar, over the main tile, or outside every tile (tileIndex -1) everything passes.
// Over a reference tile, scrolling, pressing, the context menu, double clicks and touch are blocked,
// and a plain left click swaps. The caller decides what counts as a plain click (press and
// release on the same tile without movement) and only then reports eventType "click".
// A release or touch move that ends over a reference tile still passes when the press began on
// the main tile (pressStartedOnMain), so NiiVue is never left in the middle of a drag.
export function routeEvent({ mode, mainPlane, tilePlane, tileIndex, eventType, button = 0, pressStartedOnMain = false, withModifier = false }) {
  const onReference = mode === "multiplanar" && tileIndex >= 0 && tilePlane != null && tilePlane !== mainPlane;
  if (eventType === "wheel" && withModifier) return onReference ? "block" : "zoom";
  if (mode !== "multiplanar" || tileIndex < 0 || tilePlane == null || tilePlane === mainPlane) return "pass";
  if (pressStartedOnMain && ["mouseup", "touchmove", "touchend"].includes(eventType)) return "pass";
  if (eventType === "click") return button === 0 ? "swap" : "block";
  if (BLOCKED_ON_REFERENCE.has(eventType)) return "block";
  return "pass";
}
