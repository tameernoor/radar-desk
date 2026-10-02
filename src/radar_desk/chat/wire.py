"""persona wire emitter, a Python port of persona-wire/src/index.ts (MIT).

Each method returns the SSE frame text it produced, so the caller can yield it
into a streaming response. One emitter covers one HTTP stream: the first
dispatch, or one resume of the same execution.

A run on the wire:

    execution_start, turn_start, text_start, text_delta..., text_complete,
    tool_start, tool_complete, turn_complete, execution_complete

A page tool call ends the stream with an `await` frame and no completion
frames. The resume stream carries the same executionId, no execution_start,
and a fresh turn_start at the next iteration.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any


def _default_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"


def _default_clock() -> str:
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now % 1 * 1000):03d}Z"


class Emitter:
    def __init__(
        self,
        execution_id: str,
        iteration: int = 1,
        announce_start: bool = True,
        agent_id: str = "radar-desk",
        agent_name: str = "radar-desk",
        id_factory: Callable[[str], str] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.iteration = iteration
        self.announce_start = announce_start
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._id = id_factory or _default_id
        self._clock = clock or _default_clock
        self._seq = 0
        self._started = False
        self._turn_id: str | None = None
        self._text_id: str | None = None
        self._paused = False
        self._finished = False

    @property
    def paused(self) -> bool:
        return self._paused

    def _send(self, event: str, payload: dict[str, Any]) -> str:
        body = {"type": event, "executionId": self.execution_id, "seq": self._seq, **payload}
        self._seq += 1
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        return f"event: {event}\ndata: {data}\n\n"

    def _open_turn(self) -> str:
        if self._turn_id is not None:
            return ""
        self._turn_id = self._id("turn")
        return self._send(
            "turn_start", {"id": self._turn_id, "role": "assistant", "iteration": self.iteration}
        )

    def _close_text(self) -> str:
        if self._text_id is None:
            return ""
        out = self._send("text_complete", {"id": self._text_id})
        self._text_id = None
        return out

    def start(self) -> str:
        if not self.announce_start or self._started or self._finished:
            return ""
        self._started = True
        return self._send(
            "execution_start",
            {
                "kind": "agent",
                "agentId": self.agent_id,
                "agentName": self.agent_name,
                "startedAt": self._clock(),
            },
        )

    def text_delta(self, text: str) -> str:
        if self._finished:
            return ""
        out = self._open_turn()
        if self._text_id is None:
            self._text_id = self._id("text")
            out += self._send("text_start", {"id": self._text_id})
        return out + self._send("text_delta", {"id": self._text_id, "delta": text})

    def tool_start(self, tool_call_id: str, name: str, parameters: Any, tool_type: str = "local") -> str:
        if self._finished:
            return ""
        out = self._close_text() + self._open_turn()
        return out + self._send(
            "tool_start",
            {
                "toolCallId": tool_call_id,
                "toolName": name,
                "toolType": tool_type,
                "parameters": parameters,
                "iteration": self.iteration,
            },
        )

    def tool_complete(self, tool_call_id: str, success: bool, result: Any) -> str:
        if self._finished:
            return ""
        return self._send("tool_complete", {"toolCallId": tool_call_id, "success": success, "result": result})

    def tool_call(self, name: str, parameters: Any, result: Any, tool_call_id: str | None = None) -> str:
        call_id = tool_call_id or self._id("call")
        return self.tool_start(call_id, name, parameters) + self.tool_complete(
            call_id, True, {} if result is None else result
        )

    def await_page_tool(self, tool_call_id: str, name: str, parameters: Any) -> str:
        """Pause for a page tool. The stream should end after this; complete() then emits nothing."""
        if self._finished:
            return ""
        out = self._close_text() + self._open_turn()
        self._paused = True
        return out + self._send(
            "await",
            {
                "toolName": name,
                "origin": "webmcp",
                "toolId": tool_call_id,
                "toolCallId": tool_call_id,
                "parameters": parameters,
                "awaitedAt": self._clock(),
            },
        )

    def complete(self) -> str:
        if self._finished or self._paused:
            return ""
        self._finished = True
        out = self._close_text()
        if self._turn_id is not None:
            out += self._send(
                "turn_complete",
                {
                    "id": self._turn_id,
                    "role": "assistant",
                    "iteration": self.iteration,
                    "stopReason": "end_turn",
                    "completedAt": self._clock(),
                },
            )
            self._turn_id = None
        return out + self._send(
            "execution_complete", {"kind": "agent", "success": True, "completedAt": self._clock()}
        )

    def error(self, code: str, message: str) -> str:
        if self._finished:
            return ""
        self._finished = True
        return self._send("execution_error", {"kind": "agent", "error": {"code": code, "message": message}})


def parse_frames(text: str) -> list[dict[str, Any]]:
    """Split SSE text into [{event, data}] with data parsed as JSON."""
    frames = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        frames.append({"event": event, "data": data})
    return frames
