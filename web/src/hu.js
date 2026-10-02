// HU reference for a density guess under the crosshair. Typical ranges on CT, for orientation only,
// not thresholds. Source: Wikipedia, Hounsfield scale table, read 2026-10-02.
// The text mirrors HU_TABLE in src/radar_desk/chat/prompt.py; keep the two identical.
// lo and hi are this file's reading of each row. Air is open below and ends 50 HU above -1000.
// The other "about" values are used as range ends. Water spans chyle to CSF (-30 to 15), and
// unenhanced soft tissue spans kidney's low end to liver (20 to 60). Where rows overlap, every
// matching row is named.
export const HU_TABLE = [
  { name: "Air", text: "about -1000", lo: -Infinity, hi: -950 },
  { name: "Lung parenchyma", text: "-700 to -600", lo: -700, hi: -600 },
  { name: "Fat", text: "-120 to -90", lo: -120, hi: -90 },
  { name: "Water", text: "0; urine and bile -5 to 15; CSF about 15; chyle about -30", lo: -30, hi: 15 },
  { name: "Blood", text: "unclotted 13 to 50, clotted 50 to 75", lo: 13, hi: 75 },
  { name: "Soft tissue, unenhanced", text: "kidney 20 to 45, muscle 35 to 55, liver about 60", lo: 20, hi: 60 },
  { name: "Soft tissue on contrast CT (enhanced vessel or organ)", text: "100 to 300, depends on phase", lo: 100, hi: 300 },
  { name: "Cancellous bone", text: "300 to 400", lo: 300, hi: 400 },
  { name: "Cortical bone", text: "500 to 1900", lo: 500, hi: 1900 },
];

export const HU_CAVEAT = "density guess, not a diagnosis";

// Rows whose range holds the value, or the two rows it falls between.
// Returns {hu, rows: [names], between: [below, above] | null, text}.
export function huGuess(hu) {
  if (hu == null || !Number.isFinite(hu)) return null;
  const rows = HU_TABLE.filter((r) => hu >= r.lo && hu <= r.hi).map((r) => r.name);
  // Several row names are quoted so "Soft tissue, unenhanced" does not read as two items.
  const named = rows.length > 1 ? `${rows.map((r) => `"${r}"`).join(" or ")} ranges` : `${rows[0]} range`;
  if (rows.length) return { hu, rows, between: null, text: `${hu} HU, in the ${named} (${HU_CAVEAT})` };
  const sorted = [...HU_TABLE].sort((a, b) => a.hi - b.hi);
  const below = sorted.filter((r) => r.hi < hu).at(-1);
  const above = sorted.find((r) => r.lo > hu);
  if (!above) return { hu, rows: [], between: [below.name, null], text: `${hu} HU, denser than ${below.name} in the table (${HU_CAVEAT})` };
  return { hu, rows: [], between: [below.name, above.name], text: `${hu} HU, between ${below.name} and ${above.name} (${HU_CAVEAT})` };
}
