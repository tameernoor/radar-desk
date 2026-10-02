"""The chat loop behind POST /chat and POST /chat/resume (design.md, persona integration).

dispatch: store a ChatExecution, call the model with the server tools and the page tools, run
server tools in-process and loop. A page tool call ends the stream with `await` frames and the
execution waits in state "awaiting". resume: append the page's tool outputs and continue the same
execution at the next iteration. Executions live in SQLite, so a resume works after a restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from radar_desk.chat.llm import LLM, Done, LlmError, OpenAICompatibleLLM, TextDelta, ToolCall
from radar_desk.chat.prompt import system_prompt
from radar_desk.chat.providers import StatusCache, llm_configured, provider_spec, unconfigured_message
from radar_desk.chat.tools import TOOLS, ToolSpec, is_error, openai_tools, run_tool
from radar_desk.chat.wire import Emitter
from radar_desk.records import ChatExecution, new_id

log = logging.getLogger(__name__)

MAX_LLM_CALLS = 8
CAP_TEXT = "I stopped after too many tool steps in this conversation. Ask again to continue."


class ChatNotFound(LookupError):
    """No execution with this id is waiting for tool outputs. The route answers 404."""


def make_llm(settings: Any) -> LLM | None:
    """The client for LLM_PROVIDER, or None without CHAT_MODEL or without a key the provider needs."""
    if not llm_configured(settings):
        return None
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return OpenAICompatibleLLM(
        settings.resolved_llm_base_url, key, settings.chat_model, provider=provider_spec(settings)
    )


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def flatten_history(messages: Any) -> list[dict]:
    """persona messages to OpenAI user and assistant messages with plain text.

    Replayed page tool calls and results (other roles, or no text parts) are skipped, and so are
    client system messages: the server owns the system prompt.
    """
    out = []
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        text = _text_of(m.get("content"))
        if text.strip():
            out.append({"role": m["role"], "content": text})
    return out


def _client_tool_specs(client_tools: list[dict], reserved: set[str]) -> list[dict]:
    specs = []
    for t in client_tools:
        name = t.get("name")
        if not isinstance(name, str) or not name or name in reserved:
            continue
        params = t.get("parametersSchema") or {"type": "object", "properties": {}}
        specs.append(
            {
                "type": "function",
                "function": {"name": name, "description": t.get("description") or "", "parameters": params},
            }
        )
    return specs


def _llm_calls_so_far(messages: list[dict]) -> int:
    """Model calls made by this execution: each one that asked for tools left an assistant tool_calls message."""
    return sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))


def _encode_args(arguments: dict) -> str:
    if set(arguments) == {"_raw"}:
        return str(arguments["_raw"])
    return json.dumps(arguments, ensure_ascii=False)


def _tool_output_text(output: Any) -> tuple[str, bool]:
    if isinstance(output, str):
        return output, False
    if not isinstance(output, dict):
        return "", True
    return _text_of(output.get("content")), bool(output.get("isError"))


class ChatAgent:
    def __init__(
        self,
        services: Any,
        llm: LLM | None,
        tools: list[ToolSpec] = TOOLS,
        max_llm_calls: int = MAX_LLM_CALLS,
        emitter_options: dict | None = None,
        status: StatusCache | None = None,
    ) -> None:
        self.services = services
        self.status = status if status is not None else StatusCache(services.settings)
        self.llm = llm
        self.tools = tools
        self.max_llm_calls = max_llm_calls
        self.emitter_options = emitter_options or {}
        self._server_names = {t.name for t in tools}

    def _emitter(self, execution_id: str, iteration: int, announce_start: bool) -> Emitter:
        return Emitter(execution_id, iteration=iteration, announce_start=announce_start, **self.emitter_options)

    async def dispatch(self, body: dict) -> AsyncIterator[str]:
        """POST /chat. Yields wire frames for one new execution."""
        emitter = self._emitter(new_id("exec"), 1, True)
        if self.llm is None:
            yield emitter.start()
            spec = provider_spec(self.services.settings)
            yield emitter.error("chat_unconfigured", unconfigured_message(spec))
            return
        # One provider check per dispatch, cached for 30 s, so the page sees "Ollama is not running"
        # instead of a raw connection error. Skipped when the settings name no provider to check.
        status = await self.status.get()
        if status.configured and not status.ok:
            yield emitter.start()
            yield emitter.error("provider_unavailable", status.problem or "the chat model provider is unavailable")
            return
        ex: ChatExecution | None = None
        try:
            context = body.get("context") or {}
            scan_id = context.get("scan_id") or ""
            job_id = context.get("job_id") or None
            client_tools = [t for t in body.get("clientTools") or [] if isinstance(t, dict)]
            messages = [
                {"role": "system", "content": system_prompt(self.services, scan_id, job_id)},
                *flatten_history(body.get("messages")),
            ]
            ex = ChatExecution(
                execution_id=emitter.execution_id,
                scan_id=scan_id,
                job_id=job_id,
                messages=messages,
                client_tools=client_tools,
                iteration=1,
            )
            self.services.db.insert_execution(ex)
        except Exception as exc:  # noqa: BLE001 - reported to the page as execution_error
            yield emitter.start()
            yield emitter.error("agent_error", f"{type(exc).__name__}: {exc}")
            return
        yield emitter.start()
        async for frame in self._run(ex, emitter):
            yield frame

    def resume(self, body: dict) -> AsyncIterator[str]:
        """POST /chat/resume. Raises ChatNotFound at call time, before any frame, so the route can 404."""
        execution_id = body.get("executionId") if isinstance(body, dict) else None
        ex = self.services.db.get_execution(execution_id) if isinstance(execution_id, str) else None
        if ex is None:
            raise ChatNotFound(f"no chat execution {execution_id}")
        if ex.state != "awaiting":
            raise ChatNotFound(f"chat execution {execution_id} is {ex.state}, not waiting for tool outputs")
        # Claimed here, with no await between the check and the write, so a second resume sees "running".
        ex = self.services.db.update_execution(ex.execution_id, state="running")
        return self._resume(ex, body.get("toolOutputs") or {})

    async def _resume(self, ex: ChatExecution, outputs: dict) -> AsyncIterator[str]:
        iteration = ex.iteration + 1
        emitter = self._emitter(ex.execution_id, iteration, False)
        messages = list(ex.messages)
        for call in ex.pending_tool_calls:
            output = outputs.get(call["id"]) if isinstance(outputs, dict) else None
            if output is None:
                content = json.dumps(
                    {"error": {"class": "missing_output", "message": "the page sent no output for this call"}}
                )
            else:
                text, is_error = _tool_output_text(output)
                content = json.dumps({"error": {"class": "page_tool_error", "message": text}}) if is_error else text
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        ex = self.services.db.update_execution(
            ex.execution_id, messages=messages, pending_tool_calls=[], iteration=iteration, state="running"
        )
        async for frame in self._run(ex, emitter):
            yield frame

    async def _run(self, ex: ChatExecution, emitter: Emitter) -> AsyncIterator[str]:
        db = self.services.db
        messages = list(ex.messages)
        try:
            page_names = {
                t["name"] for t in ex.client_tools if isinstance(t.get("name"), str)
            } - self._server_names
            tool_specs = openai_tools(self.tools) + _client_tool_specs(ex.client_tools, self._server_names)
            while True:
                if _llm_calls_so_far(messages) >= self.max_llm_calls:
                    yield emitter.text_delta(CAP_TEXT)
                    messages.append({"role": "assistant", "content": CAP_TEXT})
                    break
                text, calls = [], []
                async for event in self.llm.stream_chat(messages, tool_specs):
                    if isinstance(event, TextDelta):
                        text.append(event.text)
                        yield emitter.text_delta(event.text)
                    elif isinstance(event, ToolCall):
                        calls.append(event)
                    elif isinstance(event, Done):
                        pass
                assistant: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
                if calls:
                    assistant["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": _encode_args(c.arguments)},
                        }
                        for c in calls
                    ]
                messages.append(assistant)
                if not calls:
                    break

                page_calls = []
                for call in calls:
                    if call.name in page_names:
                        page_calls.append(call)
                        continue
                    yield emitter.tool_start(call.id, call.name, call.arguments)
                    result = await asyncio.to_thread(
                        run_tool, self.services, call.name, call.arguments, self.tools
                    )
                    yield emitter.tool_complete(call.id, not is_error(result), result)
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
                    )

                if page_calls:
                    # Saved before the await frames, so a resume finds the execution waiting even if
                    # the stream is dropped or the resume arrives before this generator runs again.
                    db.update_execution(
                        ex.execution_id,
                        messages=messages,
                        pending_tool_calls=[
                            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in page_calls
                        ],
                        iteration=emitter.iteration,
                        state="awaiting",
                    )
                    for call in page_calls:
                        yield emitter.await_page_tool(call.id, call.name, call.arguments)
                    return
                db.update_execution(ex.execution_id, messages=messages)

            db.update_execution(ex.execution_id, messages=messages, iteration=emitter.iteration, state="done")
            yield emitter.complete()
        except Exception as exc:  # noqa: BLE001 - reported to the page as execution_error
            code = "llm_error" if isinstance(exc, LlmError) else "agent_error"
            message = str(exc) if isinstance(exc, LlmError) else f"{type(exc).__name__}: {exc}"
            try:
                db.update_execution(ex.execution_id, messages=messages, state="error")
            except Exception:  # the error frame matters more than the record
                log.exception("could not mark chat execution %s as error", ex.execution_id)
            yield emitter.error(code, message)
