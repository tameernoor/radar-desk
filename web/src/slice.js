// Pure slice geometry for the label mask. No DOM, no NiiVue: arrays, dims and affines only.
// Voxel data is in the file's native order: index = i + j*nx + k*nx*ny.

// World axis that is the normal of each displayed plane (row of the affine).
const PLANE_NORMAL = { axial: 2, coronal: 1, sagittal: 0 };

// The native voxel axis (0, 1 or 2) that steps along the plane's world normal: the column of
// the 3x3 part of the affine with the largest absolute component in that world row.
export function planeAxis(affine, plane) {
  const row = PLANE_NORMAL[plane];
  if (row == null) throw new Error(`Unknown plane: ${plane}`);
  let best = 0;
  for (let j = 1; j < 3; j++) if (Math.abs(affine[row][j]) > Math.abs(affine[row][best])) best = j;
  return best;
}

// Flat index of native voxel (i, j, k) in a volume of size dims.
export function voxelIndex(dims, i, j, k) {
  return i + j * dims[0] + k * dims[0] * dims[1];
}

// Voxel size along each native axis, from the length of each affine column.
export function voxelSpacing(affine) {
  return [0, 1, 2].map((j) => Math.hypot(affine[0][j], affine[1][j], affine[2][j]));
}

export function invert4(m) {
  const a = m.flat();
  const inv = new Array(16);
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  return [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => inv[r * 4 + c] / det));
}

// World mm to continuous native voxel coordinates, given the inverse affine.
export function mmToVoxel(inv, mm) {
  return [0, 1, 2].map((r) => inv[r][0] * mm[0] + inv[r][1] * mm[1] + inv[r][2] * mm[2] + inv[r][3]);
}

// The two in-plane axes and the strides for walking one slice.
function sliceWalk(dims, axis) {
  const stride = [1, dims[0], dims[0] * dims[1]];
  const [u, v] = [0, 1, 2].filter((a) => a !== axis);
  return { u, v, stride };
}

// Pixel count per non-zero label on one slice: Map(label -> pixels).
export function countSlice(img, dims, axis, index) {
  const counts = new Map();
  if (index < 0 || index >= dims[axis]) return counts;
  const { u, v, stride } = sliceWalk(dims, axis);
  const base = index * stride[axis];
  for (let b = 0; b < dims[v]; b++) {
    const row = base + b * stride[v];
    for (let a = 0; a < dims[u]; a++) {
      const label = img[row + a * stride[u]];
      if (label) counts.set(label, (counts.get(label) || 0) + 1);
    }
  }
  return counts;
}

// Counts to the list the page tools return, largest first. percent_of_mask is the share of all
// labelled pixels on the slice, one decimal. nameOf(label) and isScored(label) come from the catalog.
export function organsOnSlice(counts, nameOf, isScored) {
  let total = 0;
  for (const n of counts.values()) total += n;
  return [...counts.entries()]
    .sort((x, y) => y[1] - x[1] || x[0] - y[0])
    .map(([label, pixels]) => ({
      organ: nameOf(label),
      label,
      pixels,
      percent_of_mask: Math.round((pixels / total) * 1000) / 10,
      scored: Boolean(isScored(label)),
    }));
}

// Nearest labelled pixel on one slice to a point on that slice, by in-plane distance in mm.
// voxel is the point in continuous native voxel coordinates; spacing is per native axis.
// One pass over the slice is enough: it runs only when asked for, its cost is bounded by the
// slice size (262,144 pixels at 512 x 512), so a distance transform or index would buy nothing.
// Returns {label, distance_mm} or null when the slice has no labels.
export function nearestLabelled(img, dims, axis, index, voxel, spacing) {
  if (index < 0 || index >= dims[axis]) return null;
  const { u, v, stride } = sliceWalk(dims, axis);
  const base = index * stride[axis];
  const [cu, cv, su, sv] = [voxel[u], voxel[v], spacing[u], spacing[v]];
  let best = null;
  let bestD2 = Infinity;
  for (let b = 0; b < dims[v]; b++) {
    const row = base + b * stride[v];
    const dv = (b - cv) * sv;
    for (let a = 0; a < dims[u]; a++) {
      const label = img[row + a * stride[u]];
      if (!label) continue;
      const du = (a - cu) * su;
      const d2 = du * du + dv * dv;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = label;
      }
    }
  }
  return best == null ? null : { label: best, distance_mm: Math.round(Math.sqrt(bestD2) * 10) / 10 };
}
