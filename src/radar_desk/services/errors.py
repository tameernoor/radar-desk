"""The one exception services raise for the caller to show. Routes map `status` 1:1."""

from __future__ import annotations


class ServiceError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail

    def __repr__(self) -> str:
        return f"ServiceError({self.status}, {self.detail!r})"
