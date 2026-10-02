// One colour per scored organ, used in the overlay, legend, findings list and organ strip.
export const ORGAN_COLOURS = {
  Liver: [214, 120, 80],
  "Large bowel": [205, 170, 70],
  Kidney: [230, 90, 120],
  Gallbladder: [120, 200, 80],
  "Small bowel": [240, 200, 140],
  Lung: [110, 170, 230],
  Pancreas: [240, 150, 200],
  Spleen: [160, 110, 220],
  Heart: [220, 60, 60],
  "Adrenal gland": [250, 230, 90],
  Stomach: [90, 200, 190],
  Bladder: [80, 140, 240],
  Duodenum: [200, 130, 40],
  Esophagus: [170, 220, 140],
  Aorta: [255, 90, 40],
  Rib: [235, 235, 220],
  "Portal vein": [60, 110, 200],
  Sacrum: [190, 180, 150],
};

// Labels that are segmented but not scored.
export const UNSCORED_GREY = [96, 100, 108];

export function organRgb(organ) {
  return ORGAN_COLOURS[organ] || UNSCORED_GREY;
}

export function organCss(organ, alpha = 1) {
  const [r, g, b] = organRgb(organ);
  return alpha === 1 ? `rgb(${r} ${g} ${b})` : `rgb(${r} ${g} ${b} / ${alpha})`;
}
