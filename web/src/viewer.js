// NiiVue wrapper. The NiiVue calls the app relies on (mm2frac, frac2mm,
// setColormapLabel, onLocationChange) are wrapped here so an upgrade touches one file.
import { DRAG_MODE, Niivue } from "@niivue/niivue";
import { LIGHT_DEFAULTS, lightReducer, nextZoom, presetFor, windowOf, WINDOWS, ZOOM_RANGE, zoomAround } from "./light.js";
import { organRgb, UNSCORED_GREY } from "./palette.js";
import { multiplanarLayout, PLANES, referencePlanes, routeEvent } from "./views.js";
import { countSlice, invert4, mmToVoxel, nearestLabelled, organsOnSlice, planeAxis, voxelIndex, voxelSpacing } from "./slice.js";

export { WINDOWS };

const MAX_RAW_BYTES = 512 * 1024 * 1024;
const DTYPE_BYTES = { uint8: 1, int8: 1, int16: 2, uint16: 2, int32: 4, uint32: 4, float32: 4, float64: 8 };

// ---------- the four wrapped NiiVue calls ----------
//
// Also pinned to NiiVue 0.69 internals; check these on a 1.0 upgrade:
// - nv.createOnLocationChange() is called to refresh the readout after a
//   programmatic jump (refreshLocation).
// - fillBox and labelsPresent read and write volume.img directly, in the file's
//   native voxel order, indexed with hdr.dims and hdr.affine (not NiiVue's RAS order).
// - jumpToMm writes nv.scene.crosshairPos (1.0 renames this to setCrosshairPos), and
//   sliceState reads it back through frac2mm to find the slice on screen.
// - sliceState reads the CT value with NVImage.mm2vox and getValue, as NiiVue's own
//   location string does.
// - Multiplanar uses the public setCustomLayout, clearCustomLayout and getCustomLayout
//   (src/niivue/index.ts 3145, 3165, 3174).
// - Pointer routing (the capture listener in createViewer) relies on
//   - tileIndex(x, y), index.ts 7809, @internal, which tests screenSlices[i].leftTopWidthHeight;
//   - screenSlices, index.ts 344, an undocumented public field, and its axCorSag
//     (SLICE_TYPE values in src/nvdocument.ts 40);
//   - uiData.dpr, index.ts 301, set at 698-702 and again on resize at 1212;
//   - canvas pixels computed like NiiVue's own handlers, CSS offset from the canvas's top left
//     (src/niivue/interaction/EventController.ts 58-87) times uiData.dpr (index.ts 1271, 1285);
//   - NiiVue registering its pointer listeners on the canvas in the bubbling phase
//     (index.ts 2538-2560), so a capture listener on the canvas's parent runs first.
// - Light (applyLight) uses the public setGamma (index.ts 7543), which sets the global
//   cmapper.gamma; gamma only enters makeLut (colortables.ts 276-283), so label maps keep their
//   colours. It writes volume.colormap (setter, nvimage/index.ts 789, which also resets cal_min and
//   cal_max through calMinMax, ColormapManager.ts 23-29), volume.colormapInvert
//   (nvimage/index.ts 43) and cal_min/cal_max, then calls updateGLVolume.
// - Right-drag window/level: NiiVue sets cal_min/cal_max on the volume and then calls
//   nv.onIntensityChange(volume) (index.ts 441, fired at 1578).
// - Zoom and pan use the public setPan2Dxyzmm (index.ts 3855) and read nv.scene.pan2Dxyzmm back;
//   yoke3Dto2DZoom (nvdocument.ts 153) makes the 3D render follow. Mouse buttons are mapped with
//   the public setMouseEventConfig (index.ts 8746) and the DRAG_MODE enum (nvdocument.ts 75-86).
// - centreOn reads sceneExtentsMinMax(true), index.ts 9947, tagged @internal at 9945 (CoordinateTransform.ts 38-76),
//   whose first two entries are the scene's min and max in mm. draw2D (index.ts 9683-9692) shows
//   (extent - pan) / zoom on each in-plane axis, with pan swizzled to the plane, so the view centre is
//   (c - pan) / zoom with c the middle of the extents. With isSliceMM off (the default) draw2D takes
//   the CT's ortho extents (index.ts 9637), which equal the mm ones for a scan that is not oblique.
// - Focus mode resizes the canvas's parent; NiiVue's own ResizeObserver on it (index.ts 921-924)
//   resizes the canvas. resizeListener (index.ts 1188) is @internal and is not called.

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

// ---------- viewer ----------

export function createViewer(canvas, { onLocation, onLight, onView } = {}) {
  const nv = new Niivue({
    backColor: [0, 0, 0, 1],
    crosshairColor: [1, 0.85, 0.2, 0.9],
    isColorbar: false,
    show3Dcrosshair: true,
    isOrientCube: false,
    dragAndDropEnabled: false,
    loadingText: "Loading CT", // NiiVue's font has no ellipsis glyph
    viewModeHotKey: "", // the page owns the view type; NiiVue's own key would desync it
    yoke3Dto2DZoom: true, // the 3D render follows the 2D zoom
  });
  // Plain left moves the crosshair, shift-left and middle pan, right-drag sets the window.
  // Without this the middle button means contrast.
  nv.setMouseEventConfig({
    leftButton: { primary: DRAG_MODE.crosshair, withShift: DRAG_MODE.pan, withCtrl: DRAG_MODE.crosshair },
    rightButton: DRAG_MODE.contrast,
    centerButton: DRAG_MODE.pan,
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
    light: { ...LIGHT_DEFAULTS }, // CT window in HU, gamma, invert, colour map
    slice: "axial", // view type: axial, coronal, sagittal, multiplanar or render
    mainPlane: "axial", // in multiplanar, the big view; the other two planes are references
    layoutWide: null,
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

  // Right-drag window/level: NiiVue has set the range on the volume; take it through the reducer
  // so it is clamped like any other window.
  nv.onIntensityChange = (volume) => {
    if (volume !== state.ct) return;
    light({ type: "window", min: volume.cal_min, max: volume.cal_max });
  };

  // One reducer step, then one updateGLVolume (setGamma does its own).
  function light(action) {
    const before = state.light;
    state.light = lightReducer(before, action);
    applyLight(before);
    onLight?.();
  }

  function applyLight(before = null) {
    const ct = state.ct;
    if (!ct) return;
    const l = state.light;
    // The colormap setter recomputes cal_min/cal_max (ColormapManager.ts 23-29), so it goes first.
    if (ct.colormap !== l.colormap) ct.colormap = l.colormap;
    ct.cal_min = l.min;
    ct.cal_max = l.max;
    ct.colormapInvert = l.invert;
    if (!before || before.gamma !== l.gamma) nv.setGamma(l.gamma);
    else nv.updateGLVolume();
  }

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

  // Organ names as the rest of the UI uses them; unscored labels get their catalog name, capitalised.
  function nameOfLabel(label) {
    if (label === 0) return "background";
    const organ = state.organByLabel.get(label);
    if (organ) return organ;
    const en = state.labels.find((l) => l.label === label)?.en;
    return en ? en[0].toUpperCase() + en.slice(1) : String(label);
  }

  // The plane of the main view: the single plane, the multiplanar main, or axial for 3D.
  function mainPlaneOf() {
    if (state.slice === "multiplanar") return state.mainPlane;
    return PLANES.includes(state.slice) ? state.slice : "axial";
  }

  // What get_view_state calls plane: the main plane in multiplanar, otherwise the view type.
  function displayedPlane() {
    return state.slice === "multiplanar" ? state.mainPlane : state.slice;
  }

  const SLICE_TYPES = () => ({
    axial: nv.sliceTypeAxial,
    coronal: nv.sliceTypeCoronal,
    sagittal: nv.sliceTypeSagittal,
    multiplanar: nv.sliceTypeMultiplanar,
    render: nv.sliceTypeRender,
  });
  const planeOfSliceType = (t) => PLANES.find((p) => SLICE_TYPES()[p] === t) ?? null;

  // Main view plus two references, sized for the wrapper's shape.
  function applyLayout(force = false) {
    if (state.slice !== "multiplanar") return;
    const box = canvas.parentElement.getBoundingClientRect();
    const wide = box.width >= box.height;
    if (!force && wide === state.layoutWide) return;
    state.layoutWide = wide;
    const types = SLICE_TYPES();
    nv.setCustomLayout(multiplanarLayout(state.mainPlane, box.width, box.height).map((t) => ({ sliceType: types[t.plane], position: t.position })));
  }

  new ResizeObserver(() => applyLayout()).observe(canvas.parentElement);

  // Pointer routing in multiplanar. NiiVue's listeners sit on the canvas in the bubbling phase
  // and are private, so they cannot be wrapped or removed. A capture-phase listener on an
  // ancestor runs before them for every event aimed at the canvas, and stopping the event there
  // means NiiVue never sees it. The internals it reads are listed at the top of this file.
  function tileAt(e) {
    const point = e.touches?.[0] || e.changedTouches?.[0] || e;
    const rect = canvas.getBoundingClientRect();
    // Same canvas pixels NiiVue uses: CSS offset from the canvas's top left times uiData.dpr.
    const dpr = nv.uiData.dpr || 1;
    const index = nv.tileIndex((point.clientX - rect.left) * dpr, (point.clientY - rect.top) * dpr);
    const plane = index >= 0 ? planeOfSliceType(nv.screenSlices[index].axCorSag) : null;
    return { index, plane, x: point.clientX, y: point.clientY };
  }

  // A trackpad pinch sends many small ctrl+wheel events, so deltas add up to one step per notch.
  let wheelSum = 0;
  function wheelZoom(e) {
    const d = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY; // Firefox can report lines
    if (Math.sign(d) !== Math.sign(wheelSum)) wheelSum = 0;
    wheelSum += d;
    if (Math.abs(wheelSum) < 50) return;
    api.zoomBy(wheelSum < 0 ? 1 : -1);
    wheelSum = 0;
  }

  let press = null; // where the last press began, kept until the next one: {index, x, y, onMain}
  function route(e) {
    const withModifier = e.type === "wheel" && (e.ctrlKey || e.metaKey);
    if (e.target !== canvas || (state.slice !== "multiplanar" && !withModifier)) return;
    const tile = tileAt(e);
    const type = e.type;
    if (type === "mousedown" || type === "touchstart") press = { index: tile.index, x: tile.x, y: tile.y, onMain: tile.plane === state.mainPlane };
    let eventType = type;
    if (type === "click") {
      // e.detail > 1 is the second click of a double click, which would swap straight back.
      const plain = e.detail <= 1 && press && press.index === tile.index && Math.hypot(tile.x - press.x, tile.y - press.y) < 4;
      if (!plain) return;
    }
    const action = routeEvent({
      mode: state.slice,
      mainPlane: state.mainPlane,
      tilePlane: tile.plane,
      tileIndex: tile.index,
      eventType,
      button: e.button ?? 0,
      pressStartedOnMain: Boolean(press?.onMain),
      withModifier,
    });
    if (action === "pass") return;
    e.stopPropagation();
    if (type === "wheel" || type === "contextmenu") e.preventDefault();
    if (action === "swap") api.setMainPlane(tile.plane);
    if (action === "zoom") wheelZoom(e);
  }
  for (const type of ["mousedown", "mouseup", "click", "dblclick", "wheel", "contextmenu", "touchstart", "touchmove", "touchend"]) {
    canvas.parentElement.addEventListener(type, route, { capture: true, passive: false });
  }

  // Switch the view type. Entering multiplanar keeps the single plane as the main view.
  function showView(name) {
    const types = SLICE_TYPES();
    if (name === "multiplanar" && PLANES.includes(state.slice)) state.mainPlane = state.slice;
    state.slice = name;
    if (name === "multiplanar") {
      nv.setSliceType(types.multiplanar);
      applyLayout(true);
    } else {
      if (nv.getCustomLayout()) nv.clearCustomLayout();
      state.layoutWide = null;
      nv.setSliceType(types[name]);
    }
    onView?.();
  }

  // What is on screen, computed only when asked (get_view_state), never on scroll.
  function sliceState() {
    const empty = { plane: displayedPlane(), slice: null, organs_on_slice: null, crosshair: null, nearest_organ: null };
    if (!state.ct) return empty;
    const mm = fracToMm(nv, nv.scene.crosshairPos);
    const vox = state.ct.mm2vox(mm);
    const hu = state.ct.getValue(vox[0], vox[1], vox[2]);
    const crosshair = { mm: mm.map((v) => Math.round(v * 10) / 10), hu: Number.isFinite(hu) ? Math.round(hu) : null, label: null, organ: null };
    const mask = state.mask;
    if (!mask?.img || !mask.hdr?.affine) return { ...empty, crosshair };

    // Counted on the main view's plane; 3D has no slice plane, so it reports the axial one.
    const plane = mainPlaneOf();
    const dims = [mask.hdr.dims[1], mask.hdr.dims[2], mask.hdr.dims[3]];
    const axis = planeAxis(mask.hdr.affine, plane);
    const voxel = mmToVoxel(invert4(mask.hdr.affine), mm);
    // Clamp as fillBox does: at a crosshair fraction of exactly 1 the rounded voxel equals n.
    const clamp = (v, n) => Math.max(0, Math.min(n - 1, Math.round(v)));
    const index = clamp(voxel[axis], dims[axis]);
    const inside = voxel.every((v, i) => Math.round(v) >= 0 && Math.round(v) < dims[i]);
    const label = inside ? mask.img[voxelIndex(dims, ...voxel.map((v, i) => clamp(v, dims[i])))] : 0;
    crosshair.label = label;
    crosshair.organ = label ? nameOfLabel(label) : null;

    const counts = countSlice(mask.img, dims, axis, index);
    const isScored = (l) => state.organByLabel.has(l);
    let nearest = null;
    if (!label) {
      const hit = nearestLabelled(mask.img, dims, axis, index, voxel, voxelSpacing(mask.hdr.affine));
      if (hit) nearest = { organ: nameOfLabel(hit.label), label: hit.label, distance_mm: hit.distance_mm };
    }
    return {
      plane: displayedPlane(),
      slice: { axis: plane, index, number: index + 1, count: dims[axis] },
      organs_on_slice: organsOnSlice(counts, nameOfLabel, isScored),
      crosshair,
      nearest_organ: nearest,
    };
  }

  const api = {
    nv,
    ready,

    async loadCT(url) {
      await ready;
      await nv.loadVolumes([{ url, colormap: "gray" }]);
      state.ct = nv.volumes[0];
      applyLight();
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
      light({ type: "preset", name });
    },

    setWindowHU(min, max) {
      light({ type: "window", min, max });
    },

    setGamma(value) {
      light({ type: "gamma", value });
    },

    setInvert(on) {
      light({ type: "invert", value: on });
    },

    setColormap(name) {
      light({ type: "colormap", name });
    },

    resetLight() {
      light({ type: "reset" });
    },

    // Zoom keeps the crosshair where it is on screen, as NiiVue's own wheel zoom does.
    setZoom(zoom) {
      if (!Number.isFinite(zoom)) throw new Error("Zoom must be a number.");
      const z = Math.max(ZOOM_RANGE[0], Math.min(ZOOM_RANGE[1], zoom));
      nv.setPan2Dxyzmm(zoomAround(Array.from(nv.scene.pan2Dxyzmm), z, fracToMm(nv, nv.scene.crosshairPos)));
    },

    zoomBy(direction) {
      api.setZoom(nextZoom(nv.scene.pan2Dxyzmm[3], direction));
    },

    resetView() {
      nv.setPan2Dxyzmm([0, 0, 0, 1]);
    },

    // Put mm in the middle of every 2D view at the given zoom: pan = c - zoom * mm on each axis.
    centreOn(mm, zoom) {
      if (!Number.isFinite(zoom)) throw new Error("Zoom must be a number.");
      const z = Math.max(ZOOM_RANGE[0], Math.min(ZOOM_RANGE[1], zoom));
      const [mn, mx] = nv.sceneExtentsMinMax(true);
      nv.setPan2Dxyzmm([0, 1, 2].map((i) => (mn[i] + mx[i]) / 2 - z * mm[i]).concat(z));
    },

    // A plane while in multiplanar picks the main view; otherwise it is the single view.
    setSliceType(name) {
      const types = SLICE_TYPES();
      if (!(name in types)) throw new Error(`Unknown view: ${name}`);
      if (state.slice === "multiplanar" && PLANES.includes(name)) return api.setMainPlane(name);
      showView(name);
    },

    // Swap the main view; the crosshair is left where it is.
    setMainPlane(plane) {
      if (!PLANES.includes(plane)) throw new Error(`Unknown plane: ${plane}`);
      state.mainPlane = plane;
      applyLayout(true);
      onView?.();
    },

    // Back to a single view of the main plane.
    exitMultiplanar() {
      if (state.slice === "multiplanar") showView(state.mainPlane);
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
      const labelName = state.labelValue == null ? null : nameOfLabel(state.labelValue);
      return {
        crosshair_mm: state.crosshairMm ? state.crosshairMm.map((v) => Math.round(v * 10) / 10) : null,
        hu: state.hu == null ? null : Math.round(state.hu),
        label: state.labelValue,
        label_name: labelName,
        organ: state.labelValue != null ? state.organByLabel.get(state.labelValue) || null : null,
      };
    },

    // The slice on screen and what RADAR outlined on it. Walks one mask slice, so it runs only
    // for get_view_state and the look card, never for key presses or on scroll.
    sliceState() {
      const view = sliceState();
      const c = view.crosshair;
      // The flat fields follow the same fresh reading, so a scroll without a click cannot leave them stale.
      const fresh = c
        ? { crosshair_mm: c.mm, hu: c.hu, ...(c.label == null ? {} : { label: c.label, label_name: nameOfLabel(c.label), organ: state.organByLabel.get(c.label) || null }) }
        : {};
      return { ...fresh, ...view };
    },

    // Cheap: toggles and settings plus the last crosshair reading.
    state() {
      const l = state.light;
      const r1 = (v) => Math.round(v * 10) / 10;
      const { width, level } = windowOf(l.min, l.max);
      return {
        window_preset: presetFor(l.min, l.max),
        window_hu: { min: r1(l.min), max: r1(l.max), width: r1(width), level: r1(level) },
        gamma: l.gamma,
        invert: l.invert,
        colormap: l.colormap,
        zoom: nv.scene.pan2Dxyzmm[3],
        slice_type: state.slice,
        main_plane: state.slice === "render" ? null : mainPlaneOf(),
        reference_planes: state.slice === "multiplanar" ? referencePlanes(state.mainPlane) : [],
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
  const dims = [nx, ny, nz];
  const set = (i, j, k) => (img[voxelIndex(dims, i, j, k)] = 1);
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
