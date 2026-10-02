"""Server tools for the chat agent: thin calls into services, JSON in and JSON out.

Each tool has a name, a description, a JSON schema for its arguments and a `run(services, args)`.
`run_tool` checks the arguments against the schema (required keys and types only) and turns every
exception into `{"error": {"class", "message"}}`, so a failing tool never ends the chat run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from radar_desk.services.errors import ServiceError
from radar_desk.services.exports import ARTEFACT_URL_S

ARTEFACT_NAMES = ("mask", "scores_json", "scores_csv", "trace", "log")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    run: Callable[[Any, dict], dict]


def _schema(properties: dict | None = None, required: tuple[str, ...] = ()) -> dict:
    return {
        "type": "object",
        "properties": properties or {},
        "required": list(required),
        "additionalProperties": False,
    }


def _dump(model: Any) -> dict:
    return model.model_dump(mode="json")


def _scan_summary(services: Any, scan: Any) -> dict:
    header = scan.header
    return {
        "id": scan.id,
        "filename": scan.filename,
        "state": scan.state,
        "rejected_reason": scan.rejected_reason,
        "fixture_id": scan.fixture_id,
        "created_at": scan.created_at,
        "header": None
        if header is None
        else {
            "dims": header.dims,
            "spacing_mm": header.spacing_mm,
            "orientation": header.orientation,
            "dtype": header.dtype,
        },
        "latest_job": services.jobs.latest_summary(scan.id),
    }


def _job_summary(job: Any) -> dict:
    return {
        "job_id": job.id,
        "scan_id": job.scan_id,
        "state": job.state,
        "hold_reason": job.hold_reason,
        "gpu_requested": job.gpu_requested,
        "gpu_used": job.gpu_used,
        "model_version": job.model_version,
        "queued_at": job.queued_at,
        "submitted_at": job.submitted_at,
        "finished_at": job.finished_at,
        "error": _dump(job.error) if job.error else None,
        "timings": _dump(job.timings) if job.timings else None,
        "cost_estimate_usd": job.cost_estimate_usd,
    }


def latest_done_job_id(services: Any, scan_id: str) -> str | None:
    """The newest done job of a scan that has a stored result, or None."""
    for job in services.jobs.list(state="done", scan_id=scan_id, limit=20):
        if services.db.get_result(job.id) is not None:
            return job.id
    return None


# Tool bodies


def list_scans(services: Any, args: dict) -> dict:
    scans = services.scans.list(state=args.get("state"), limit=args.get("limit", 20))
    return {"scans": [_scan_summary(services, s) for s in scans]}


def get_scan(services: Any, args: dict) -> dict:
    scan = services.scans.get(args["scan_id"])
    jobs = services.jobs.list(scan_id=scan.id, limit=50)
    return {
        "scan": _scan_summary(services, scan),
        "jobs": [_job_summary(j) for j in jobs],
    }


def score_scan(services: Any, args: dict) -> dict:
    job = services.jobs.create(args["scan_id"])
    return {"job_id": job.id, "state": job.state, "hold_reason": job.hold_reason}


def get_job(services: Any, args: dict) -> dict:
    return _job_summary(services.jobs.get(args["job_id"]))


def cancel_job(services: Any, args: dict) -> dict:
    job = services.jobs.cancel(args["job_id"])
    return {"job_id": job.id, "state": job.state}


def get_scores(services: Any, args: dict) -> dict:
    job_id = args.get("job_id")
    if not job_id:
        scan_id = args.get("scan_id")
        if not scan_id:
            raise ServiceError(422, "give job_id or scan_id")
        services.scans.get(scan_id)
        job_id = latest_done_job_id(services, scan_id)
        if job_id is None:
            raise ServiceError(404, f"scan {scan_id} has no finished job with scores")
    result = services.results.get(job_id)
    findings = result.findings
    organ = args.get("organ")
    if organ:
        names = {o.lower(): o for o in services.catalog.scored_organ_names()}
        name = names.get(organ.strip().lower())
        if name is None:
            raise ServiceError(404, f"{organ!r} is not a scored organ")
        findings = [f for f in findings if f.organ == name]
    min_prob = args.get("min_prob")
    if min_prob is not None:
        findings = [f for f in findings if f.prob is not None and f.prob >= min_prob]
    return {
        "job_id": job_id,
        "findings": [_dump(f) for f in findings],
        "organs_not_found": result.organs_not_found,
        "versions": _dump(result.versions),
    }


def get_organ(services: Any, args: dict) -> dict:
    return services.results.get_organ(args["job_id"], args["organ"])


def compare_scores(services: Any, args: dict) -> dict:
    return services.results.compare(args["job_id"], args["against"])


def get_artifacts(services: Any, args: dict) -> dict:
    job_id = args["job_id"]
    services.results.get(job_id)
    urls, missing = {}, {}
    for name in ARTEFACT_NAMES:
        try:
            urls[name] = services.exports.artefact_url(job_id, name)
        except ServiceError as exc:
            missing[name] = exc.detail
    return {"job_id": job_id, "urls": urls, "missing": missing, "expires_in_s": ARTEFACT_URL_S}


def gpu_status(services: Any, args: dict) -> dict:
    return services.gpu_status()


_JOB_ID = {"job_id": {"type": "string", "description": "Job id such as job_ab12cd34ef56."}}
_SCAN_ID = {"scan_id": {"type": "string", "description": "Scan id such as scan_ab12cd34ef56."}}

TOOLS: list[ToolSpec] = [
    ToolSpec(
        "list_scans",
        "List uploaded scans, newest first, with header facts and the latest job summary.",
        _schema(
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Default 20."},
                "state": {"type": "string", "enum": ["uploading", "ready", "rejected"]},
            }
        ),
        list_scans,
    ),
    ToolSpec(
        "get_scan",
        "One scan with its header facts, all its scoring jobs and its fixture tag.",
        _schema(_SCAN_ID, ("scan_id",)),
        get_scan,
    ),
    ToolSpec(
        "score_scan",
        "Queue a scoring job for a ready scan and return its job id at once. Idempotent: an active "
        "job is returned instead of a new one. Scoring takes minutes; poll get_job.",
        _schema(_SCAN_ID, ("scan_id",)),
        score_scan,
    ),
    ToolSpec(
        "get_job",
        "A scoring job's state, hold reason, GPU type, timings and cost estimate.",
        _schema(_JOB_ID, ("job_id",)),
        get_job,
    ),
    ToolSpec(
        "cancel_job",
        "Cancel a queued or running scoring job.",
        _schema(_JOB_ID, ("job_id",)),
        cancel_job,
    ),
    ToolSpec(
        "get_scores",
        "Finding probabilities of a job, or of a scan's latest finished job. Filter by minimum "
        "probability (0 to 1) or by organ. Also returns organs not found and versions.",
        _schema(
            {
                **_JOB_ID,
                **_SCAN_ID,
                "min_prob": {"type": "number", "minimum": 0, "maximum": 1},
                "organ": {"type": "string", "description": "English organ name such as Liver."},
            }
        ),
        get_scores,
    ),
    ToolSpec(
        "get_organ",
        "One organ in a job: its findings, its segmentation stats, and how and where it was scored.",
        _schema({**_JOB_ID, "organ": {"type": "string"}}, ("job_id", "organ")),
        get_organ,
    ),
    ToolSpec(
        "compare_scores",
        "Per-finding deltas of a job against another job id or against \"fixture\", the published "
        "reference for a known scan. Returns max abs delta and the count over tolerance.",
        _schema(
            {**_JOB_ID, "against": {"type": "string", "description": "A job id or \"fixture\"."}},
            ("job_id", "against"),
        ),
        compare_scores,
    ),
    ToolSpec(
        "get_artifacts",
        "Short-lived download URLs for a job's mask, scores, trace and log.",
        _schema(_JOB_ID, ("job_id",)),
        get_artifacts,
    ),
    ToolSpec(
        "gpu_status",
        "The GPU backend: in-flight job, GPU type, spend this month, budget and held jobs.",
        _schema(),
        gpu_status,
    ),
]


def openai_tools(specs: list[ToolSpec]) -> list[dict]:
    """The specs in the OpenAI `tools` request format."""
    return [
        {"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.parameters}}
        for s in specs
    ]


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def validate_args(schema: dict, args: Any) -> str | None:
    """A message for the first problem found, or None. Checks required keys and value types only."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    if "_raw" in args:
        return "arguments were not valid JSON"
    for key in schema.get("required", []):
        if args.get(key) is None:
            return f"missing required argument {key!r}"
    props = schema.get("properties", {})
    for key, value in args.items():
        if value is None:
            continue
        if key not in props:
            if schema.get("additionalProperties") is False:
                return f"unknown argument {key!r}"
            continue
        expected = props[key].get("type")
        allowed = _TYPES.get(expected)
        if allowed is None:
            continue
        if isinstance(value, bool) and bool not in allowed:
            return f"argument {key!r} must be {expected}"
        if not isinstance(value, allowed):
            return f"argument {key!r} must be {expected}"
    return None


def is_error(result: Any) -> bool:
    """True for run_tool's error envelope, a dict whose only key is "error"."""
    return isinstance(result, dict) and set(result) == {"error"}


def _error(klass: str, message: str, **extra: Any) -> dict:
    return {"error": {"class": klass, "message": message, **extra}}


def run_tool(services: Any, name: str, args: Any, tools: list[ToolSpec] | None = None) -> dict:
    """Run one tool by name. Never raises; problems come back as {"error": {...}}."""
    spec = next((t for t in (TOOLS if tools is None else tools) if t.name == name), None)
    if spec is None:
        return _error("unknown_tool", f"no tool named {name!r}")
    problem = validate_args(spec.parameters, args)
    if problem:
        return _error("invalid_arguments", problem)
    clean = {k: v for k, v in args.items() if v is not None}
    try:
        return spec.run(services, clean)
    except ServiceError as exc:
        return _error("ServiceError", exc.detail, status=exc.status)
    except Exception as exc:  # noqa: BLE001 - a failing tool is data for the model, not a crash
        return _error(type(exc).__name__, str(exc) or type(exc).__name__)
