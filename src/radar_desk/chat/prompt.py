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
- Keep answers short. Offer to jump the viewer to the organ or finding you talk about; page tools move the viewer."""


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
    parts = [RULES]
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
