"""The interface between the poller and whatever runs the scoring function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from radar_desk.records import Job

# Artefact name in the result -> file name under jobs/{job_id}/.
ARTEFACT_FILES: dict[str, str] = {
    "scores_json": "scores.json",
    "scores_csv": "scores.csv",
    "mask": "mask.nii.gz",
    "trace": "trace.json",
    "log": "worker.log",
}


def artefact_keys(job_id: str) -> dict[str, str]:
    """Bucket keys of a job's five artefacts."""
    return {name: f"jobs/{job_id}/{fname}" for name, fname in ARTEFACT_FILES.items()}


@dataclass(frozen=True)
class Pending:
    pass


@dataclass(frozen=True)
class Finished:
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Errored:
    klass: str
    message: str = ""


PollOutcome = Pending | Finished | Errored


class GpuBackend(Protocol):
    name: str

    def spawn(
        self,
        job: Job,
        source_url: str,
        artefact_urls: dict[str, str],
        artefact_keys: dict[str, str] | None = None,
    ) -> str:
        """Start scoring and return the call id. Raises when the call could not be started."""
        ...

    def poll(self, call_id: str) -> PollOutcome: ...

    def cancel(self, call_id: str) -> None: ...

    def logs(self, call_id: str, lines: int = 200) -> str: ...
