"""Modal backend: spawn the deployed function and poll its FunctionCall.

`modal` is imported inside the methods, so the API starts without touching it when the fake
backend is chosen.
"""

from __future__ import annotations

import builtins
from typing import Any

from radar_desk.gpu.backend import Errored, Finished, Pending, PollOutcome
from radar_desk.gpu.backend import artefact_keys as default_artefact_keys
from radar_desk.records import Job


class ModalGpuBackend:
    name = "modal"

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._client: Any = None
        self._function: Any = None

    def _modal(self) -> Any:
        import modal

        return modal

    def _client_or_none(self) -> Any:
        """A client from the settings' token, or None so modal falls back to its own config."""
        if self._client is None:
            tid, tsecret = self.settings.modal_token_id, self.settings.modal_token_secret
            if tid is not None and tsecret is not None:
                self._client = self._modal().Client.from_credentials(
                    tid.get_secret_value(), tsecret.get_secret_value()
                )
        return self._client

    def _kwargs(self) -> dict[str, Any]:
        client = self._client_or_none()
        return {"client": client} if client is not None else {}

    def _call(self, call_id: str) -> Any:
        return self._modal().FunctionCall.from_id(call_id, **self._kwargs())

    def spawn(
        self,
        job: Job,
        source_url: str,
        artefact_urls: dict[str, str],
        artefact_keys: dict[str, str] | None = None,
    ) -> str:
        keys = artefact_keys or default_artefact_keys(job.id)
        if self._function is None:
            self._function = self._modal().Function.from_name(
                self.settings.modal_app_name, self.settings.modal_function_name, **self._kwargs()
            )
        call = self._function.spawn(job.id, source_url, artefact_urls, keys)
        return call.object_id

    def poll(self, call_id: str) -> PollOutcome:
        modal = self._modal()
        exc_mod = modal.exception
        try:
            value = self._call(call_id).get(timeout=0)
        except exc_mod.OutputExpiredError as exc:
            return Errored("expired", str(exc) or "the result expired before it was read")
        except exc_mod.FunctionTimeoutError as exc:
            return Errored("FunctionTimeoutError", str(exc) or "the function hit its timeout")
        except builtins.TimeoutError:
            return Pending()
        except self._transport_errors():
            raise  # Modal unreachable: the poller keeps the job submitted under the stuck rule
        except Exception as exc:  # noqa: BLE001 - the call's own exception after Modal's retry
            return Errored(type(exc).__name__, str(exc))
        if isinstance(value, dict):
            return Finished(value)
        return Errored("bad_result", f"the function returned {type(value).__name__}, not a dict")

    def _transport_errors(self) -> tuple[type[BaseException], ...]:
        """Errors that say nothing about the call itself, only that Modal could not be reached."""
        exc_mod = self._modal().exception
        names = ("ConnectionError", "AuthError", "InternalError", "ServiceError", "ClientClosed",
                 "_GRPCErrorWrapper")
        found: list[type[BaseException]] = [OSError]
        found += [getattr(exc_mod, n) for n in names if isinstance(getattr(exc_mod, n, None), type)]
        try:
            import grpclib
            from grpclib.exceptions import StreamTerminatedError

            found += [grpclib.GRPCError, StreamTerminatedError]
        except ImportError:
            pass
        return tuple(found)

    def cancel(self, call_id: str) -> None:
        self._call(call_id).cancel()

    def logs(self, call_id: str, lines: int = 200) -> str:
        entries = self._call(call_id).logs.tail(entries=lines)
        return "".join(str(getattr(entry, "message", entry)) for entry in entries)
