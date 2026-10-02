// Unit tests for src/slice.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import { countSlice, invert4, mmToVoxel, nearestLabelled, organsOnSlice, planeAxis, voxelIndex, voxelSpacing } from "../src/slice.js";

const DIMS = [8, 8, 4];
const at = (i, j, k) => i + j * DIMS[0] + k * DIMS[0] * DIMS[1];

function volume(entries) {
  const img = new Uint8Array(DIMS[0] * DIMS[1] * DIMS[2]);
  for (const [i, j, k, label] of entries) img[at(i, j, k)] = label;
  return img;
}

const NAMES = { 2: "Aorta", 5: "Liver", 3: "Erector spinae muscle", 4: "Kidney" };
const nameOf = (l) => NAMES[l];
const isScored = (l) => l !== 3;

test("counts labels on an axial slice and ignores other slices", () => {
  const img = volume([
    [0, 0, 1, 5], [1, 0, 1, 5], [2, 0, 1, 5], [0, 1, 1, 5], [1, 1, 1, 5], [2, 1, 1, 5],
    [7, 7, 1, 2], [6, 7, 1, 2],
    [3, 3, 2, 5], // another slice
  ]);
  const counts = countSlice(img, DIMS, 2, 1);
  expect(Object.fromEntries(counts)).toEqual({ 5: 6, 2: 2 });
  expect(Object.fromEntries(countSlice(img, DIMS, 0, 7))).toEqual({ 2: 1 }); // sagittal slice i = 7
});

test("percent of mask and largest-first ordering", () => {
  const counts = new Map([[2, 2], [5, 6], [3, 1]]);
  const rows = organsOnSlice(counts, nameOf, isScored);
  expect(rows.map((r) => r.organ)).toEqual(["Liver", "Aorta", "Erector spinae muscle"]);
  expect(rows[0]).toEqual({ organ: "Liver", label: 5, pixels: 6, percent_of_mask: 66.7, scored: true });
  expect(rows[1].percent_of_mask).toBe(22.2);
  expect(rows[2]).toMatchObject({ percent_of_mask: 11.1, scored: false });
});

test("nearest organ uses the in-plane spacing", () => {
  const img = volume([[4, 0, 1, 3], [0, 3, 1, 4]]);
  // Along i the label is 4 voxels away, along j 3 voxels away.
  expect(nearestLabelled(img, DIMS, 2, 1, [0, 0, 1], [1, 2, 5])).toEqual({ label: 3, distance_mm: 4 });
  expect(nearestLabelled(img, DIMS, 2, 1, [0, 0, 1], [2, 1, 5])).toEqual({ label: 4, distance_mm: 3 });
  // Diagonal: sqrt(1.5^2 + 2^2) = 2.5.
  expect(nearestLabelled(volume([[1, 2, 0, 5]]), DIMS, 2, 0, [0, 0, 0], [1.5, 1, 3])).toEqual({ label: 5, distance_mm: 2.5 });
});

test("empty slice gives no organs and no nearest organ", () => {
  const img = volume([[1, 1, 3, 5]]);
  expect(organsOnSlice(countSlice(img, DIMS, 2, 0), nameOf, isScored)).toEqual([]);
  expect(nearestLabelled(img, DIMS, 2, 0, [4, 4, 0], [1, 1, 1])).toBeNull();
  expect(countSlice(img, DIMS, 2, 9).size).toBe(0); // out of range
  expect(nearestLabelled(img, DIMS, 2, -1, [4, 4, 0], [1, 1, 1])).toBeNull();
});

test("plane axis from an LAS affine", () => {
  const las = [[-0.8, 0, 0, 200], [0, 0.8, 0, -150], [0, 0, 2.5, -300], [0, 0, 0, 1]];
  expect(planeAxis(las, "axial")).toBe(2);
  expect(planeAxis(las, "coronal")).toBe(1);
  expect(planeAxis(las, "sagittal")).toBe(0);
  expect(voxelSpacing(las)).toEqual([0.8, 0.8, 2.5]);
});

test("plane axis from a permuted affine", () => {
  // Native i steps along world z, j along world x, k along world y.
  const permuted = [[0, -0.7, 0, 0], [0, 0, 0.7, 0], [3, 0, 0, 0], [0, 0, 0, 1]];
  expect(planeAxis(permuted, "axial")).toBe(0);
  expect(planeAxis(permuted, "coronal")).toBe(2);
  expect(planeAxis(permuted, "sagittal")).toBe(1);
  expect(voxelSpacing(permuted)).toEqual([3, 0.7, 0.7]);
});

test("world mm through a permuted affine to the slice index on the plane axis", () => {
  // Native i steps along world z (3 mm), j along world -x, k along world y (0.7 mm).
  const permuted = [[0, -0.7, 0, 100], [0, 0, 0.7, -50], [3, 0, 0, -30], [0, 0, 0, 1]];
  const voxel = mmToVoxel(invert4(permuted), [100 - 0.7 * 5, -50 + 0.7 * 6, -30 + 3 * 2]);
  voxel.forEach((v, n) => expect(v).toBeCloseTo([2, 5, 6][n], 9));
  const axial = planeAxis(permuted, "axial");
  expect(axial).toBe(0);
  expect(Math.round(voxel[axial])).toBe(2); // axial slice index 2, slice number 3
  expect(Math.round(voxel[planeAxis(permuted, "sagittal")])).toBe(5);
  expect(voxelIndex([8, 8, 4], 2, 5, 3)).toBe(2 + 5 * 8 + 3 * 64);
});
