// NiiVue wrapper. The four NiiVue calls the app relies on (mm2frac, frac2mm,
// setColormapLabel, onLocationChange) are wrapped here so an upgrade touches one file.
import { Niivue } from "@niivue/niivue";
import { organRgb, UNSCORED_GREY } from "./palette.js";

export const WINDOWS = {
  soft_tissue: { label: "Soft tissue", min: -160, max: 240 },
  liver: { label: "Liver", min: -20, max: 160 },
  bone: { label: "Bone", min: -450, max: 1050 },
  lung: { label: "Lung", min: -1350, max: 150 },
};

const MAX_RAW_BYTES = 512 * 1024 * 1024;
const DTYPE_BYTES = { uint8: 1, int8: 1, int16: 2, uint16: 2, int32: 4, uint32: 4, float32: 4, float64: 8 };

// ---------- the four wrapped NiiVue calls ----------
//
// Also pinned to NiiVue 0.69 internals; check these on a 1.0 upgrade:
// - nv.createOnLocationChange() is called to refresh the readout after a
//   programmatic jump (refreshLocation).
// - fillBox and labelsPresent read and write volume.img directly, in the file's
//   native voxel order, indexed with hdr.dims and hdr.affine (not NiiVue's RAS order).
// - jumpToMm writes nv.scene.crosshairPos (1.0 renames this to setCrosshairPos).

function mmToFrac(nv, mm) {
  return nv.mm2frac([mm[0], mm[1], mm[2]]);
}

function fracToMm(nv, frac) {
  const mm = nv.frac2mm(frac);
  return [mm[0], mm[1], mm[2]];
}

function applyLabelColours(volume, colormap) {
  volume.setColormapLabel(colormap);
}

function listenLocation(nv, handler) {
  nv.onLocationChange = (loc) => handler({ mm: [loc.mm[0], loc.mm[1], loc.mm[2]], values: loc.values.map((v) => v.value) });
}

// ---------- helpers ----------

// Raw size of a volume from its header. Falls back to decompressed_bytes.
export function rawBytes(scan) {
  const h = scan.header;
  if (!h) return scan.decompressed_bytes || 0;
  const per = DTYPE_BYTES[String(h.dtype).toLowerCase()] || 4;
  return h.dims.reduce((a, b) => a * b, 1) * per;
}

export const tooBig = (scan) => rawBytes(scan) > MAX_RAW_BYTES;

function invert4(m) {
  // General 4x4 inverse (row-major nested arrays).
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

function mmToVoxel(inv, mm) {
  return [0, 1, 2].map((r) => inv[r][0] * mm[0] + inv[r][1] * mm[1] + inv[r][2] * mm[2] + inv[r][3]);
}

// ---------- viewer ----------

export function createViewer(canvas, { onLocation, onWindow } = {}) {
  const nv = new Niivue({
    backColor: [0, 0, 0, 1],
    crosshairColor: [1, 0.85, 0.2, 0.9],
    isColorbar: false,
    show3Dcrosshair: true,
    isOrientCube: false,
    dragAndDropEnabled: false,
    loadingText: "Loading CT", // NiiVue's font has no ellipsis glyph
    viewModeHotKey: "", // the page owns the view type; NiiVue's own key would desync it
  });
  // Nearest neighbour so label edges stay exact. NiiVue 0.69 only has a global
  // setting, so the CT is drawn nearest neighbour too.
  nv.setInterpolation(true);

  const state = {
    ct: null,
    mask: null,
    box: null,
    labels: [], // index = label value, {label, en}
    organByLabel: new Map(), // label value -> scored organ name
    isolated: null, // label value or null
    maskOn: true,
    opacity: 0.45,
    outline: false,
    window: "soft_tissue",
    slice: "axial",
    crosshairMm: null,
    hu: null,
    labelValue: null,
    boxOrgan: null,
  };

  listenLocation(nv, ({ mm, values }) => {
    state.crosshairMm = mm;
    state.hu = values[0] ?? null;
    state.labelValue = state.mask ? Math.round(values[nv.volumes.indexOf(state.mask)] ?? 0) : null;
    onLocation?.(api.location());
  });

  // Right-drag window/level leaves the named presets behind.
  nv.onIntensityChange = () => {
    state.window = "custom";
    onWindow?.("custom");
  };

  const ready = nv.attachToCanvas(canvas);

  function volumeIndex(vol) {
    return nv.volumes.indexOf(vol);
  }

  function labelColormap(highlight) {
    const R = [], G = [], B = [], A = [], I = [], names = [];
    for (const { label, en } of state.labels) {
      const organ = state.organByLabel.get(label);
      const [r, g, b] = organ ? organRgb(organ) : UNSCORED_GREY;
      let a = label === 0 ? 0 : organ ? 255 : 150;
      if (highlight != null && label !== 0) a = label === highlight ? 255 : 0;
      R.push(r);
      G.push(g);
      B.push(b);
      A.push(a);
      I.push(label);
      names.push(organ || en);
    }
    return { R, G, B, A, I, labels: names };
  }

  function refreshMaskColours() {
    if (!state.mask) return;
    applyLabelColours(state.mask, labelColormap(state.isolated));
    nv.updateGLVolume();
  }

  function refreshLocation() {
    if (typeof nv.createOnLocationChange === "function") nv.createOnLocationChange();
  }

  const api = {
    nv,
    ready,

    async loadCT(url) {
      await ready;
      await nv.loadVolumes([{ url, colormap: "gray" }]);
      state.ct = nv.volumes[0];
      api.setWindow(state.window);
      api.setSliceType(state.slice);
      state.crosshairMm = fracToMm(nv, nv.scene.crosshairPos);
      refreshLocation();
    },

    // labels: [{label, en}], organLabels: [{organ, label}]
    async loadMask(url, labels, organLabels) {
      await ready;
      state.labels = [...labels].sort((a, b) => a.label - b.label);
      state.organByLabel = new Map(organLabels.map((o) => [o.label, o.organ]));
      // A newer job replaces the mask (and its scoring box) of an older one.
      for (const old of [state.box, state.mask]) if (old) nv.removeVolume(old);
      state.box = null;
      state.boxOrgan = null;
      state.mask = null;
      const mask = await nv.addVolumeFromUrl({ url, colormap: "gray", opacity: state.maskOn ? state.opacity : 0 });
      state.mask = mask;
      refreshMaskColours();
      nv.setOpacity(volumeIndex(mask), state.maskOn ? state.opacity : 0);
      refreshLocation();
      return api.labelsPresent();
    },

    // Label values that occur in the mask, read once from the voxel data.
    labelsPresent() {
      if (!state.mask?.img) return [];
      const seen = new Uint8Array(256);
      const img = state.mask.img;
      for (let i = 0; i < img.length; i++) seen[img[i] & 255] = 1;
      return [...seen.keys()].filter((v) => v > 0 && seen[v]);
    },

    setWindow(name) {
      const w = WINDOWS[name];
      if (!w) throw new Error(`Unknown window preset: ${name}`);
      state.window = name;
      onWindow?.(name);
      if (!state.ct) return;
      state.ct.cal_min = w.min;
      state.ct.cal_max = w.max;
      nv.updateGLVolume();
    },

    setSliceType(name) {
      const types = {
        axial: nv.sliceTypeAxial,
        coronal: nv.sliceTypeCoronal,
        sagittal: nv.sliceTypeSagittal,
        multiplanar: nv.sliceTypeMultiplanar,
        render: nv.sliceTypeRender,
      };
      if (!(name in types)) throw new Error(`Unknown view: ${name}`);
      state.slice = name;
      nv.setSliceType(types[name]);
    },

    jumpToMm(mm) {
      if (!state.ct) return;
      nv.scene.crosshairPos = mmToFrac(nv, mm);
      state.crosshairMm = [mm[0], mm[1], mm[2]];
      nv.updateGLVolume();
      nv.drawScene();
      refreshLocation();
    },

    // Each change re-uploads the volumes, so skip it when nothing changes.
    isolateOrgan(label) {
      if (state.isolated === label) return;
      state.isolated = label;
      refreshMaskColours();
    },

    setMaskVisible(on) {
      state.maskOn = on;
      if (state.mask) nv.setOpacity(volumeIndex(state.mask), on ? state.opacity : 0);
    },

    setMaskOpacity(a) {
      state.opacity = a;
      if (state.mask && state.maskOn) nv.setOpacity(volumeIndex(state.mask), a);
    },

    setOutline(on) {
      state.outline = on;
      nv.setAtlasOutline(on ? 1 : 0);
    },

    // Outline of the scoring box as a one-voxel shell in a second label volume.
    showBox(boxMm, organ) {
      if (!state.mask) return false;
      if (!boxMm) {
        state.boxOrgan = null;
        if (state.box) nv.setOpacity(volumeIndex(state.box), 0);
        return true;
      }
      if (!state.box) {
        state.box = state.mask.clone();
        state.box.name = "scoring box";
        applyLabelColours(state.box, { R: [0, 255], G: [0, 255], B: [0, 255], A: [0, 255], I: [0, 1], labels: ["", "scoring box"] });
        nv.addVolume(state.box);
      }
      fillBox(state.box, boxMm);
      state.boxOrgan = organ;
      nv.setOpacity(volumeIndex(state.box), 1);
      nv.updateGLVolume();
      return true;
    },

    location() {
      const labelName =
        state.labelValue == null
          ? null
          : state.labelValue === 0
            ? "background"
            : state.organByLabel.get(state.labelValue) || state.labels.find((l) => l.label === state.labelValue)?.en || String(state.labelValue);
      return {
        crosshair_mm: state.crosshairMm ? state.crosshairMm.map((v) => Math.round(v * 10) / 10) : null,
        hu: state.hu == null ? null : Math.round(state.hu),
        label: state.labelValue,
        label_name: labelName,
        organ: state.labelValue != null ? state.organByLabel.get(state.labelValue) || null : null,
      };
    },

    state() {
      return {
        window_preset: state.window,
        slice_type: state.slice,
        mask_loaded: Boolean(state.mask),
        mask_on: state.maskOn,
        mask_opacity: state.opacity,
        outline: state.outline,
        isolated_label: state.isolated,
        scoring_box_organ: state.boxOrgan,
        ...api.location(),
      };
    },
  };
  return api;
}

function fillBox(volume, boxMm) {
  const img = volume.img;
  img.fill(0);
  const [nx, ny, nz] = [volume.hdr.dims[1], volume.hdr.dims[2], volume.hdr.dims[3]];
  const inv = invert4(volume.hdr.affine);
  const [lo, hi] = boxMm;
  const corners = [];
  for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) corners.push(mmToVoxel(inv, [x, y, z]));
  const clamp = (v, n) => Math.max(0, Math.min(n - 1, Math.round(v)));
  const range = (axis, n) => {
    const vals = corners.map((c) => c[axis]);
    return [clamp(Math.min(...vals), n), clamp(Math.max(...vals), n)];
  };
  const [i0, i1] = range(0, nx);
  const [j0, j1] = range(1, ny);
  const [k0, k1] = range(2, nz);
  const set = (i, j, k) => (img[i + j * nx + k * nx * ny] = 1);
  for (let k = k0; k <= k1; k++)
    for (let j = j0; j <= j1; j++) {
      set(i0, j, k);
      set(i1, j, k);
    }
  for (let k = k0; k <= k1; k++)
    for (let i = i0; i <= i1; i++) {
      set(i, j0, k);
      set(i, j1, k);
    }
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      set(i, j, k0);
      set(i, j, k1);
    }
}
