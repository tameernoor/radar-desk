"""The reading radiologist's calls: one chat-model call per output, strict JSON, checked before it is kept.

A read keeps a label's state only when it is one of three; a quote is kept only when it appears in the text it
claims to quote (case and spacing aside). A missing label counts as not_mentioned; unknown ids are dropped. An
unusable answer is asked again once, then the read fails.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from radar_desk.chat.llm import LlmError, TextDelta
from radar_desk.readers.prompts import read_messages

STATES = ("present", "absent", "not_mentioned")
ATTEMPTS = 2


class ReadError(RuntimeError):
    """The model's answer could not be used; the message says why."""


async def _collect(llm: Any, messages: list[dict]) -> str:
    parts = []
    async for event in llm.stream_chat(messages, []):
        if isinstance(event, TextDelta):
            parts.append(event.text)
    return "".join(parts)


def complete(llm: Any, messages: list[dict]) -> str:
    """One call, the whole answer as text. Runs its own event loop, so call it from a worker thread."""
    return asyncio.run(_collect(llm, messages))


def parse_json(text: str) -> dict:
    """The first-to-last brace span of the answer as a JSON object (so code fences and chatter are ignored)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ReadError("the answer holds no JSON object")
    try:
        value = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise ReadError(f"the answer is not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ReadError("the answer is not a JSON object")
    return value


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def validate_read(data: dict, source: str, label_ids: list[str]) -> dict:
    labels = data.get("labels")
    if not isinstance(labels, dict):
        raise ReadError("no labels object")
    src = norm(source)
    out = {}
    for lid in label_ids:
        item = labels.get(lid, {"state": "not_mentioned"})
        if not isinstance(item, dict):
            raise ReadError(f"label {lid} is not an object")
        state = item.get("state")
        if state not in STATES:
            raise ReadError(f"label {lid} has state {state!r}")
        quote = item.get("quote")
        if state == "not_mentioned" or not isinstance(quote, str) or not quote.strip() or norm(quote) not in src:
            quote = None
        out[lid] = {"state": state, "quote": quote}
    others: list[str] = []
    raw_others = data.get("other_mentions")
    if raw_others is None:
        raw_others = []
    if not isinstance(raw_others, list) or not all(isinstance(q, str) for q in raw_others):
        raise ReadError("other_mentions is not a list of strings")
    for q in raw_others:
        if isinstance(q, str) and q.strip() and norm(q) in src and q.strip() not in others:
            others.append(q.strip())
    return {"labels": out, "other_mentions": others}


def read_text(llm: Any, text: str, labels: list[dict]) -> dict:
    """Read one report against the labels. Raises ReadError after ATTEMPTS unusable answers."""
    ids = [r["id"] for r in labels]
    messages = read_messages(text, labels)
    problem = "no answer"
    for _ in range(ATTEMPTS):
        try:
            return validate_read(parse_json(complete(llm, messages)), text, ids)
        except (ReadError, LlmError) as exc:
            problem = str(exc)
    raise ReadError(problem)
