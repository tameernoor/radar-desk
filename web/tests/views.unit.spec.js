// Unit tests for src/views.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import { multiplanarLayout, referencePlanes, routeEvent } from "../src/views.js";

const base = { mode: "multiplanar", mainPlane: "axial", tilePlane: "coronal", tileIndex: 1, button: 0 };
const EVENTS = ["wheel", "mousedown", "mouseup", "contextmenu", "dblclick", "touchstart", "touchmove", "touchend", "click"];

test("outside multiplanar everything passes", () => {
  for (const mode of ["axial", "coronal", "sagittal", "render"]) {
    for (const eventType of EVENTS) expect(routeEvent({ ...base, mode, eventType })).toBe("pass");
  }
});

test("over the main tile everything passes", () => {
  for (const eventType of EVENTS) expect(routeEvent({ ...base, tilePlane: "axial", tileIndex: 0, eventType })).toBe("pass");
});

test("outside every tile (index -1) passes", () => {
  for (const eventType of EVENTS) expect(routeEvent({ ...base, tileIndex: -1, tilePlane: null, eventType })).toBe("pass");
});

test("over a reference tile scrolling, pressing, menus, double clicks and touch are blocked", () => {
  for (const eventType of ["wheel", "mousedown", "mouseup", "contextmenu", "dblclick", "touchstart", "touchmove", "touchend"]) {
    for (const button of [0, 1, 2]) expect(routeEvent({ ...base, eventType, button })).toBe("block");
  }
  expect(routeEvent({ ...base, eventType: "mousemove" })).toBe("pass");
});

test("only a plain left click on a reference tile swaps", () => {
  expect(routeEvent({ ...base, eventType: "click", button: 0 })).toBe("swap");
  expect(routeEvent({ ...base, eventType: "click", button: 1 })).toBe("block");
  expect(routeEvent({ ...base, eventType: "click", button: 2 })).toBe("block");
  expect(routeEvent({ ...base, tilePlane: "sagittal", tileIndex: 2, eventType: "click" })).toBe("swap");
});

test("a drag that began on the main tile may end over a reference", () => {
  for (const eventType of ["mouseup", "touchmove", "touchend"]) expect(routeEvent({ ...base, eventType, pressStartedOnMain: true })).toBe("pass");
  for (const eventType of ["wheel", "mousedown", "touchstart"]) expect(routeEvent({ ...base, eventType, pressStartedOnMain: true })).toBe("block");
});

test("a tile with no slice plane (tilePlane null) passes", () => {
  for (const eventType of EVENTS) expect(routeEvent({ ...base, tileIndex: 1, tilePlane: null, eventType })).toBe("pass");
});

function overlaps(a, b) {
  const [ax, ay, aw, ah] = a.position;
  const [bx, by, bw, bh] = b.position;
  return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}

function checkTiling(tiles) {
  for (let i = 0; i < tiles.length; i++) for (let j = i + 1; j < tiles.length; j++) expect(overlaps(tiles[i], tiles[j])).toBe(false);
  const area = tiles.reduce((sum, t) => sum + t.position[2] * t.position[3], 0);
  expect(area).toBeCloseTo(1, 9);
  for (const t of tiles) for (const v of t.position) expect(v).toBeGreaterThanOrEqual(0);
}

test("wide layout: main left 75% full height, references stacked right", () => {
  const tiles = multiplanarLayout("axial", 1200, 700);
  expect(tiles.map((t) => t.plane)).toEqual(["axial", "coronal", "sagittal"]);
  expect(tiles[0].position).toEqual([0, 0, 0.75, 1]);
  expect(tiles[1].position).toEqual([0.75, 0, 0.25, 0.5]);
  expect(tiles[2].position).toEqual([0.75, 0.5, 0.25, 0.5]);
  checkTiling(tiles);
  expect(multiplanarLayout("axial", 500, 500)[0].position).toEqual([0, 0, 0.75, 1]); // square counts as wide
});

test("narrow layout: main on top 75% full width, references side by side", () => {
  const tiles = multiplanarLayout("sagittal", 600, 900);
  expect(tiles.map((t) => t.plane)).toEqual(["sagittal", "axial", "coronal"]);
  expect(tiles[0].position).toEqual([0, 0, 1, 0.75]);
  expect(tiles[1].position).toEqual([0, 0.75, 0.5, 0.25]);
  expect(tiles[2].position).toEqual([0.5, 0.75, 0.5, 0.25]);
  checkTiling(tiles);
});

test("references are the other two planes", () => {
  expect(referencePlanes("coronal")).toEqual(["axial", "sagittal"]);
});
