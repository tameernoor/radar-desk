"""Pydantic records shared by the database, the API, the poller and the chat agent.

Timestamps are ISO 8601 UTC strings with microseconds and a trailing Z, so they sort as text.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .radar.header import Header  # one Header for records and parsing

ScanState = Literal["uploading", "ready", "rejected"]
JobState = Literal["queued", "submitted", "done", "failed", "cancelled"]
TERMINAL_JOB_STATES: tuple[str, ...] = ("done", "failed", "cancelled")
ChatState = Literal["running", "awaiting", "done", "error"]
PodPhase = Literal["tunnel", "starting", "ready", "stopped"]

Box = list[list[float]]  # [[xmin, ymin, zmin], [xmax, ymax, zmax]] in world mm


def new_id(prefix: str) -> str:
    """A short random id such as scan_ab12cd34ef56."""
    return f"{prefix}_{secrets.token_hex(6)}"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Scan(BaseModel):
    id: str
    filename: str
    size_bytes: int
    sha256: str | None = None
    state: ScanState = "uploading"
    rejected_reason: str | None = None
    header: Header | None = None
    decompressed_bytes: int | None = None
    fixture_id: str | None = None
    research_only_confirmed_at: str | None = None
    created_at: str = Field(default_factory=now_iso)


class JobError(BaseModel):
    """Serialises as {class, message}; use `klass` in Python."""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, serialize_by_alias=True)

    klass: str = Field(alias="class")
    message: str


class Timings(BaseModel):
    load_s: float | None = None
    download_s: float | None = None
    infer_s: float | None = None
    postprocess_s: float | None = None
    upload_s: float | None = None
    total_s: float | None = None


class Lease(BaseModel):
    """A pull worker's hold on a submitted job. The lease id itself is the job's `modal_call_id`."""

    worker_id: str
    expires_at: str
    heartbeat_at: str
    progress: str | None = None


class Job(BaseModel):
    id: str
    scan_id: str
    state: JobState = "queued"
    hold_reason: str | None = None
    modal_call_id: str | None = None
    gpu_requested: list[str] = Field(default_factory=list)
    gpu_used: str | None = None
    model_version: str | None = None
    queued_at: str = Field(default_factory=now_iso)
    submitted_at: str | None = None
    finished_at: str | None = None
    error: JobError | None = None
    timings: Timings | None = None
    cost_estimate_usd: float | None = None
    lease: Lease | None = None
    lease_losses: int = 0
    backend: str | None = None  # modal, worker or fake, set at spawn and at claim


class PodRecord(BaseModel):
    """A RunPod worker pod the app started, from its tunnel to its deletion. `id` is local."""

    id: str = Field(default_factory=lambda: new_id("pod"))
    runpod_id: str | None = None
    name: str = "radar-worker"
    phase: PodPhase = "tunnel"
    gpu: str | None = None
    image: str | None = None
    cost_per_hr: float | None = None
    created_at: str = Field(default_factory=now_iso)
    started_at: str | None = None
    ready_at: str | None = None
    stopped_at: str | None = None
    token_id: str | None = None
    worker_id: str = Field(default_factory=lambda: f"runpod-{secrets.token_hex(3)}")
    tunnel_url: str | None = None
    tunnel: dict[str, Any] | None = None  # {pid, url, log, binary}, managed tunnel only
    cost_usd: float | None = None
    reason: str | None = None
    error: str | None = None


class WorkerToken(BaseModel):
    """A pull worker's credential. Only the sha256 of the plaintext is kept."""

    id: str
    name: str
    token_hash: str
    created_at: str = Field(default_factory=now_iso)
    revoked_at: str | None = None
    last_used_at: str | None = None


class Worker(BaseModel):
    """A pull worker as it last reported itself. The id is chosen by the worker."""

    id: str
    token_id: str
    hostname: str | None = None
    gpu_name: str | None = None
    device: str | None = None
    versions: dict[str, Any] = Field(default_factory=dict)
    first_seen_at: str = Field(default_factory=now_iso)
    last_seen_at: str = Field(default_factory=now_iso)
    job_id: str | None = None


class Finding(BaseModel):
    key: str
    organ: str
    finding: str
    prob: float | None = None


class OrganScored(BaseModel):
    organ: str
    label: int
    how: Literal["window", "centered_crop"]
    window_index: int | None = None
    box_mm: Box


class OrganStats(BaseModel):
    voxels: int
    ml: float
    centroid_mm: list[float]
    bbox_mm: Box


class Versions(BaseModel):
    code_commit: str | None = None
    checkpoint_sha256: str | None = None
    torch: str | None = None
    cuda: str | None = None
    gpu: str | None = None
    image_id: str | None = None


class Artefacts(BaseModel):
    """Bucket keys of the job's output objects."""

    scores_json: str | None = None
    scores_csv: str | None = None
    mask: str | None = None
    trace: str | None = None
    log: str | None = None


class Result(BaseModel):
    job_id: str
    findings: list[Finding]
    organs_scored: list[OrganScored] = Field(default_factory=list)
    organs_not_found: list[str] = Field(default_factory=list)
    organ_stats: dict[str, OrganStats] = Field(default_factory=dict)
    versions: Versions = Field(default_factory=Versions)
    artefacts: Artefacts = Field(default_factory=Artefacts)
    created_at: str = Field(default_factory=now_iso)


class ChatExecution(BaseModel):
    execution_id: str
    scan_id: str
    job_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    client_tools: list[dict[str, Any]] = Field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    iteration: int = 0
    state: ChatState = "running"
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
