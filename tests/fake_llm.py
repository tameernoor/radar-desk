"""A scripted chat model for tests and the dev seed: each call answers with the next scripted reply."""

from __future__ import annotations

import copy

from radar_desk.chat.llm import Done, LlmError, TextDelta


class ScriptedLLM:
    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def stream_chat(self, messages, tools):
        self.calls.append(copy.deepcopy(messages))
        if not self.replies:
            raise LlmError("the scripted model has no reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        yield TextDelta(reply)
        yield Done("stop")
