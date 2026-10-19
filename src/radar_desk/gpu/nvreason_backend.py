"""NV-Reason-CT on the deployed Modal class radar-desk-nvreason / NvReason (worker/nvreason_app.py).

Polling, cancel and logs are the RADAR backend's (FunctionCall.from_id). A job spawns the class's `report` method;
`ask` spawns one answer and waits for it up to a timeout. `modal` is imported inside the methods.
"""

from __future__ import annotations

import builtins
import logging
from typing import Any

from radar_desk.gpu.modal_backend import ModalGpuBackend

log = logging.getLogger(__name__)

NV_APP_NAME = "radar-desk-nvreason"
NV_CLASS_NAME = "NvReason"


class NvAskTimeout(RuntimeError):
    """NV-Reason-CT did not answer in time; the call was cancelled."""


class ModalNvReasonBackend(ModalGpuBackend):
    name = "modal"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._instance: Any = None

    def spawn(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("the NV-Reason-CT backend never starts a RADAR run; use spawn_report")

    def _reader(self) -> Any:
        if self._instance is None:
            cls = self._modal().Cls.from_name(NV_APP_NAME, NV_CLASS_NAME, **self._kwargs())
            self._instance = cls()
        return self._instance

    def spawn_report(self, job_id: str, source_ref: str) -> str:
        return self._reader().report.spawn(job_id, source_ref).object_id

    def ask(self, source_ref: str, region: str, messages: list[dict], enable_thinking: bool,
            max_new_tokens: int, timeout_s: float) -> dict:
        call = self._reader().ask.spawn(source_ref, region, messages, enable_thinking, max_new_tokens)
        try:
            value = call.get(timeout=timeout_s)
        except builtins.TimeoutError:
            try:
                call.cancel()
            except Exception:  # the timeout is the error worth reporting; this one is only logged
                log.warning("could not cancel the timed-out NV-Reason-CT ask", exc_info=True)
            raise NvAskTimeout(f"NV-Reason-CT did not answer within {timeout_s:g} s") from None
        if not isinstance(value, dict):
            raise TypeError(f"NV-Reason-CT returned {type(value).__name__}, not a dict")
        return value
