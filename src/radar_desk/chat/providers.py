"""Chat model providers: OpenRouter (default), a local Ollama, or any OpenAI-compatible endpoint.

All three speak the OpenAI chat completions API, so one client serves them all. A ProviderSpec
holds what differs: the default base URL, whether a key is needed, extra request headers, how to
read an error body, and a health check that the page shows through GET /chat/status.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from radar_desk.records import now_iso

CHECK_TIMEOUT_S = 5.0
STATUS_TTL_OK_S = 30.0
STATUS_TTL_FAIL_S = 5.0
MESSAGE_CAP = 300


@dataclass
class ProviderStatus:
    configured: bool
    provider: str
    model: str | None
    base_url_host: str
    ok: bool
    problem: str | None = None
    detail: dict = field(default_factory=dict)
    checked_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


Check = Callable[["ProviderSpec", Any, httpx.AsyncClient], Awaitable[ProviderStatus]]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    default_base_url: str
    key_required: bool
    extra_headers: dict[str, str]
    parse_error: Callable[[str], str]
    check: Check


# Error bodies


def message_of(err: Any, fallback: str) -> str:
    """err["message"] when it is a non-empty string, else the fallback text; capped at 300 characters."""
    if isinstance(err, dict) and isinstance(err.get("message"), str) and err["message"]:
        return err["message"][:MESSAGE_CAP]
    if isinstance(err, str) and err:
        return err[:MESSAGE_CAP]
    return fallback[:MESSAGE_CAP]


def _error_object(body: str) -> Any:
    try:
        data = json.loads(body)
    except ValueError:
        return None
    return data.get("error") if isinstance(data, dict) else None


def standard_error(body: str) -> str:
    """OpenAI's shape {"error": {"message", "type", "param", "code"}}, or a bare {"error": "text"}."""
    return message_of(_error_object(body), body)


def openrouter_error(body: str) -> str:
    """OpenRouter's shape {"error": {"code": int, "message", "metadata"?}} (openrouter.ai/docs/api-reference/errors)."""
    return message_of(_error_object(body), body)


def ollama_error(body: str) -> str:
    """Ollama's shape {"error": {"message", "type", "param", "code": str|null}}.

    Source github.com/ollama/ollama/blob/main/openai/openai.go, type Error and NewError: a 404 has
    type "not_found_error", which is what a model that is not pulled gives.
    """
    err = _error_object(body)
    message = message_of(err, body)
    if isinstance(err, dict) and err.get("type") == "not_found_error":
        return f"{message} (run `ollama pull` for the model)"
    return message


# Checks


def host_of(base_url: str) -> str:
    return urlsplit(base_url).netloc or base_url


def _status(spec: ProviderSpec, settings: Any, ok: bool, problem: str | None = None,
            detail: dict | None = None, configured: bool = True) -> ProviderStatus:
    return ProviderStatus(
        configured=configured,
        provider=spec.name,
        model=settings.chat_model,
        base_url_host=host_of(resolved_base_url(settings)),
        ok=ok,
        problem=problem,
        detail=detail or {},
    )


def _bearer(settings: Any) -> dict[str, str]:
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    return {"Authorization": f"Bearer {key}"} if key else {}


async def check_openrouter(spec: ProviderSpec, settings: Any, client: httpx.AsyncClient) -> ProviderStatus:
    """GET {base}/key with the inference key.

    Response {"data": {"label", "limit", "limit_reset", "limit_remaining", "usage", "is_free_tier", ...}}
    per openrouter.ai/docs/api-reference/limits. limit_remaining is the key's own credit cap, null when
    unlimited. GET /credits is no fallback: it needs a management key (openrouter.ai/docs/api-reference/get-credits).
    """
    base = resolved_base_url(settings).rstrip("/")
    resp = await client.get(f"{base}/key", headers=_bearer(settings))
    if resp.status_code == 401:
        return _status(spec, settings, False, "the OpenRouter key was rejected")
    if resp.status_code >= 300:
        return _status(spec, settings, False,
                       f"OpenRouter answered {resp.status_code}: {openrouter_error(resp.text)}")
    data = (resp.json() or {}).get("data") or {}
    # No "label": OpenRouter names a key by a masked prefix of the key itself by default, and nothing
    # that comes from the key may reach a response or a log.
    detail = {k: data[k] for k in ("limit", "limit_remaining", "usage", "is_free_tier") if k in data}
    remaining = data.get("limit_remaining")
    if isinstance(remaining, int | float) and remaining <= 0:
        return _status(spec, settings, False, "this key's OpenRouter credit limit is used up", detail)
    return _status(spec, settings, True, None, detail)


def ollama_origin(base_url: str) -> str:
    """The native API origin: the OpenAI-compatible base without its trailing /v1."""
    base = base_url.rstrip("/")
    return base.removesuffix("/v1")


async def check_ollama(spec: ProviderSpec, settings: Any, client: httpx.AsyncClient) -> ProviderStatus:
    """GET /api/tags lists pulled models as {"models": [{"name", "model", ...}]}; GET /api/version gives
    {"version"} (github.com/ollama/ollama/blob/main/docs/api.md). Ollama ignores the Authorization header."""
    base = resolved_base_url(settings)
    origin = ollama_origin(base)
    try:
        resp = await client.get(f"{origin}/api/tags")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return _status(spec, settings, False, f"Ollama is not running at {host_of(base)}")
    if resp.status_code >= 300:
        return _status(spec, settings, False, f"Ollama answered {resp.status_code}: {ollama_error(resp.text)}")
    try:
        models = (resp.json() or {}).get("models") or []
    except (ValueError, AttributeError):
        return _status(spec, settings, False, f"{host_of(base)} did not answer like Ollama (no model list)")
    names = {m.get(k) for m in models if isinstance(m, dict) for k in ("name", "model") if m.get(k)}
    detail: dict[str, Any] = {"models": len(models)}
    try:
        version = await client.get(f"{origin}/api/version")
        if version.status_code < 300:
            detail["version"] = (version.json() or {}).get("version")
    except (httpx.HTTPError, ValueError, AttributeError):
        pass  # the version is optional
    model = settings.chat_model or ""
    candidates = {model, f"{model}:latest", model.removesuffix(":latest")}
    if not candidates & names:
        return _status(spec, settings, False, f"model {model} is not pulled; run `ollama pull {model}`", detail)
    return _status(spec, settings, True, None, detail)


async def check_openai(spec: ProviderSpec, settings: Any, client: httpx.AsyncClient) -> ProviderStatus:
    """GET {base}/models with the key. The model need not be listed; some servers list only a few.

    Only a rejected key (401, 403) or a server error (5xx) fails. A 404 or 405 means the server does
    not list models at all, which is fine.
    """
    base = resolved_base_url(settings).rstrip("/")
    resp = await client.get(f"{base}/models", headers=_bearer(settings))
    if resp.status_code in (401, 403):
        return _status(spec, settings, False, f"the key was rejected by {host_of(base)}")
    if resp.status_code >= 500:
        return _status(spec, settings, False,
                       f"{host_of(base)} answered {resp.status_code}: {standard_error(resp.text)}")
    if resp.status_code in (404, 405):
        return _status(spec, settings, True, None, {"models_endpoint": "absent"})
    if resp.status_code >= 300:
        return _status(spec, settings, True, None, {"models_endpoint_status": resp.status_code})
    return _status(spec, settings, True)


PROVIDERS: dict[str, ProviderSpec] = {
    "openrouter": ProviderSpec(
        name="openrouter",
        label="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        key_required=True,
        extra_headers={"X-Title": "radar-desk"},
        parse_error=openrouter_error,
        check=check_openrouter,
    ),
    "ollama": ProviderSpec(
        name="ollama",
        label="Ollama",
        default_base_url="http://localhost:11434/v1",
        key_required=False,
        extra_headers={},
        parse_error=ollama_error,
        check=check_ollama,
    ),
    "openai": ProviderSpec(
        name="openai",
        label="an OpenAI-compatible endpoint",
        default_base_url="https://api.openai.com/v1",
        key_required=True,
        extra_headers={},
        parse_error=standard_error,
        check=check_openai,
    ),
}


def get_provider(name: str | None) -> ProviderSpec:
    try:
        return PROVIDERS[name or "openrouter"]
    except KeyError:
        raise KeyError(f"unknown LLM provider {name!r} (expected one of {sorted(PROVIDERS)})") from None


def resolved_provider_name(settings: Any) -> str:
    """LLM_PROVIDER when set. Otherwise inferred from LLM_BASE_URL, so setups from before LLM_PROVIDER
    existed keep working: openrouter.ai is openrouter, localhost or 127.0.0.1 on port 11434 is ollama,
    any other URL is a generic OpenAI-compatible endpoint, and no URL at all is openrouter."""
    if settings.llm_provider:
        return settings.llm_provider
    if not settings.llm_base_url:
        return "openrouter"
    parts = urlsplit(settings.llm_base_url)
    host = (parts.hostname or "").lower()
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return "openrouter"
    if host in ("localhost", "127.0.0.1", "::1") and parts.port == 11434:
        return "ollama"
    return "openai"


def provider_spec(settings: Any) -> ProviderSpec:
    return get_provider(resolved_provider_name(settings))


def resolved_base_url(settings: Any) -> str:
    """LLM_BASE_URL when set, else the provider's default."""
    return settings.llm_base_url or provider_spec(settings).default_base_url


def has_key(settings: Any) -> bool:
    return bool(settings.llm_api_key and settings.llm_api_key.get_secret_value())


def llm_configured(settings: Any) -> bool:
    spec = provider_spec(settings)
    return bool(settings.chat_model) and (has_key(settings) or not spec.key_required)


def unconfigured_message(spec: ProviderSpec) -> str:
    if spec.key_required:
        return f"Chat is not configured: set CHAT_MODEL and, for {spec.label}, LLM_API_KEY."
    return "Chat is not configured: set CHAT_MODEL."


async def check_provider(settings: Any, client: httpx.AsyncClient) -> ProviderStatus:
    """The provider's status. Never raises and never puts the key in a message."""
    spec = provider_spec(settings)
    if not llm_configured(settings):
        return _status(spec, settings, False, unconfigured_message(spec), configured=False)
    host = host_of(resolved_base_url(settings))
    try:
        return await spec.check(spec, settings, client)
    except httpx.TimeoutException:
        return _status(spec, settings, False, f"{host} did not answer within {CHECK_TIMEOUT_S:g} s")
    except httpx.HTTPError as exc:
        return _status(spec, settings, False, f"could not reach {host} ({type(exc).__name__})")
    except Exception as exc:  # noqa: BLE001 - the page shows the problem; the type name carries no secret
        return _status(spec, settings, False, f"could not check {host} ({type(exc).__name__})")


class StatusCache:
    """The last provider status, so the page can poll and each dispatch can check cheaply.

    A passing check is kept for 30 s, a failing one for 5 s, so a user who starts Ollama or fixes a
    key is not refused for long and a one-off timeout clears quickly.
    """

    def __init__(self, settings: Any, ttl_ok_s: float = STATUS_TTL_OK_S,
                 ttl_fail_s: float = STATUS_TTL_FAIL_S,
                 transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self.ttl_ok_s = ttl_ok_s
        self.ttl_fail_s = ttl_fail_s
        self.transport = transport
        self.clock = clock
        self._last: ProviderStatus | None = None
        self._at = 0.0
        self._lock = asyncio.Lock()

    def _fresh(self) -> ProviderStatus | None:
        if self._last is None:
            return None
        ttl = self.ttl_ok_s if self._last.ok else self.ttl_fail_s
        return self._last if self.clock() - self._at < ttl else None

    async def get(self) -> ProviderStatus:
        if (fresh := self._fresh()) is not None:
            return fresh
        async with self._lock:
            if (fresh := self._fresh()) is not None:
                return fresh
            async with httpx.AsyncClient(timeout=CHECK_TIMEOUT_S, transport=self.transport) as client:
                status = await check_provider(self.settings, client)
            self._last, self._at = status, self.clock()
            return status
