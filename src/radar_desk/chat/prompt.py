"""The chat system prompt: fixed rules plus the context of the scan on screen."""

from __future__ import annotations

from typing import Any

from radar_desk.chat.tools import latest_done_job_id

RULES = """You are the assistant in radar-desk, a research tool that runs the RADAR model on abdominal CT scans and shows its scores.

The rules.
- This is research use, not diagnosis. Never give a diagnosis or clinical advice.
- Answer from the scores below and from your tools, never from imagination. The finding list is closed at 146: if a finding is not in it, RADAR does not score it.
- When you mention a finding, say its score and its organ.
- 50% is a display threshold, not a calibrated one. Say so when a score near or above 50% comes up.
- A quiet organ means RADAR found nothing in the organ it segmented, not that the organ is normal. When asked, say where the organ was scored (which window or crop), or that it was not found.
- Keep answers short. Offer to jump the viewer to the organ or finding you talk about; page tools move the viewer.
- For questions like "what am I looking at", "where am I", "what is this" or "which slice is this", call get_view_state first. It is a page tool and is allow-listed, so it runs without asking.
- Then answer with the plane and the slice as number of count (slice.number counts from 1; slice.index counts from 0), the organs RADAR outlined on that slice (from organs_on_slice) with each one's RADAR scores from the score table, the positives at the display line first and then a note that the rest are below it, and what is under the crosshair (organ and HU) or, on background, the nearest outlined organ and its distance in mm. For an outline with scored false, say RADAR segments that structure but does not score it. In multiplanar view the slice reported is the main view's plane (main_plane); the two small reference views follow it and are listed in reference_planes.
- If slice and crosshair are both null, nothing is loaded in the viewer. If only slice is null, the CT is loaded but RADAR's outlines are not there yet (no finished job), so give the HU under the crosshair and say there are no outlines to report.
- Any tissue guess from an HU value must come from the HU table below and be labelled as a guess from density, never a diagnosis.
- Outlines are RADAR's segmentation, not confirmed anatomy.
- To show an organ or a finding the way it is usually read, call set_view_for; it jumps, sets the recipe's window and zoom and shows the scoring box. Then say which window it set and why, from the viewing recipes table.
- Recommend window settings only from the viewing recipes table; set_window_level is for a user who asks for other numbers.
- set_light changes gamma, invert and the colour map; colour maps are a spotting aid, not the reading standard."""

# Typical CT attenuation ranges for orientation, not thresholds. Figures from the table in the Wikipedia
# article "Hounsfield scale" (en.wikipedia.org/wiki/Hounsfield_scale, read 2026-10-02), which cites the
# primary sources per row; rounded here and kept as ranges. Unenhanced values unless stated.
HU_TABLE = """HU reference (typical ranges on CT, for orientation only, not thresholds).
- Air: about -1000
- Lung parenchyma: -700 to -600
- Fat: -120 to -90
- Water: 0; urine and bile -5 to 15; CSF about 15; chyle about -30
- Blood: unclotted 13 to 50, clotted 50 to 75
- Soft tissue, unenhanced: kidney 20 to 45, muscle 35 to 55, liver about 60
- Soft tissue on contrast CT (enhanced vessel or organ): 100 to 300, depends on phase
- Cancellous bone: 300 to 400
- Cortical bone: 500 to 1900"""

# Window and zoom per scored organ for set_view_for. Widths and levels from Radiopaedia, "Windowing (CT)"
# (radiopaedia.org/articles/windowing-ct, last revised 7 Jan 2025, read 2026-10-03), its abdomen, chest and
# spine rows. It has rows only for the liver, vessels, bone and lungs, so every other scored organ takes the
# abdomen soft tissue window.
# Mirrors web/src/recipes.js and WINDOWS in web/src/light.js; a unit test there fails on any drift.
VIEW_RECIPES = """Viewing recipes (the window each scored organ is usually read in; set_view_for applies them).
Windows: soft_tissue W 400 L 50, liver W 150 L 30, bone W 1800 L 400, lung W 1500 L -600, angio W 600 L 200.
- Liver: liver. narrow liver window; small density differences inside the parenchyma show
- Pancreas: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels (the source gives no pancreas window; the abdomen soft tissue one is used)
- Kidney: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Gallbladder: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Spleen: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Adrenal gland: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Stomach: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Duodenum: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Small bowel: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Large bowel: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Bladder: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Esophagus: soft_tissue. abdomen soft tissue window; organ parenchyma, fluid and fat separate without clipping enhancing vessels
- Aorta: angio. vascular window; the enhanced lumen, the wall and calcification stay apart instead of saturating
- Portal vein: angio. vascular window; the enhanced lumen, the wall and calcification stay apart instead of saturating
- Heart: angio. vascular window; the enhanced lumen, the wall and calcification stay apart instead of saturating
- Lung: lung. wide lung window centred on air; parenchyma and nodules show at the lung bases
- Rib: bone. wide bone window; cortex and marrow show instead of saturating white
- Sacrum: bone. wide bone window; cortex and marrow show instead of saturating white
Colour maps are a spotting aid, not the reading standard.
A viewing setting is never a diagnosis."""


def _pct(prob: float | None) -> str:
    return "not scored" if prob is None else f"{prob * 100:.1f}%"


def _scan_lines(scan: Any) -> list[str]:
    lines = [f"Scan {scan.id}: file {scan.filename}, state {scan.state}."]
    if scan.rejected_reason:
        lines.append(f"Rejected: {scan.rejected_reason}")
    h = scan.header
    if h is not None:
        dims = " x ".join(str(d) for d in h.dims)
        spacing = " x ".join(f"{s:g}" for s in h.spacing_mm)
        lines.append(f"Header: {dims} voxels, spacing {spacing} mm, orientation {h.orientation}, {h.dtype}.")
    if scan.fixture_id:
        lines.append(f"Known fixture: {scan.fixture_id} (compare_scores against \"fixture\" works).")
    return lines


def _coverage_lines(result: Any, organs: list[str]) -> list[str]:
    scored = {s.organ: s for s in result.organs_scored}
    lines = ["Organ coverage:"]
    for organ in organs:
        if organ in result.organs_not_found:
            lines.append(f"- {organ}: not found by the segmentation, so not scored")
        elif organ in scored:
            s = scored[organ]
            where = f"window {s.window_index}" if s.how == "window" else "centred crop"
            lines.append(f"- {organ}: scored in {where}")
        else:
            lines.append(f"- {organ}: scored, location not recorded")
    return lines


def _score_lines(result: Any, organs: list[str]) -> list[str]:
    lines = ["Scores (finding key | English name | probability), grouped by organ:"]
    for organ in organs:
        rows = [f for f in result.findings if f.organ == organ]
        if not rows:
            continue
        lines.append(f"{organ}:")
        lines.extend(f"  {f.key} | {f.finding} | {_pct(f.prob)}" for f in rows)
    return lines


def system_prompt(services: Any, scan_id: str | None, job_id: str | None) -> str:
    parts = [RULES, HU_TABLE, VIEW_RECIPES]
    scan = services.db.get_scan(scan_id) if scan_id else None
    if scan is None:
        parts.append("No scan is open. Use list_scans to find one.")
        return "\n\n".join(parts)
    parts.append("\n".join(_scan_lines(scan)))

    job = services.db.get_job(job_id) if job_id else None
    if job is not None and job.scan_id != scan.id:
        job = None
    on_screen = job
    result = services.db.get_result(job.id) if job is not None else None
    if result is None:
        done_id = latest_done_job_id(services, scan.id)
        if done_id is not None:
            job = services.db.get_job(done_id)
            result = services.db.get_result(done_id)
    if result is None:
        latest = job or services.db.latest_job_for_scan(scan.id)
        if latest is None:
            parts.append("This scan has not been scored yet. score_scan queues a job.")
        else:
            hold = f", held: {latest.hold_reason}" if latest.hold_reason else ""
            parts.append(f"Job {latest.id} is {latest.state}{hold}. No scores yet; get_job shows progress.")
        return "\n\n".join(parts)

    organs = services.catalog.scored_organ_names()
    gpu = result.versions.gpu or "unknown"
    if on_screen is not None and on_screen.id != job.id:
        parts.append(f"Job {on_screen.id} on screen is {on_screen.state}; the scores below are from job {job.id}.")
    parts.append(f"Job {job.id} is {job.state} (model {job.model_version}, GPU {gpu}).")
    if gpu == "fake":
        parts.append("These scores come from the fake GPU backend. They are synthetic, not RADAR output; say so.")
    parts.append("\n".join(_coverage_lines(result, organs)))
    parts.append("\n".join(_score_lines(result, organs)))
    return "\n\n".join(parts)
