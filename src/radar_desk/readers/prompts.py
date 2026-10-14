"""Every prompt of the reading radiologist, in one place.

One role, played by the chat model (CHAT_MODEL): a radiologist who reads reports and prepares questions. It reads
and asks; it never judges which reader is right.
"""

from __future__ import annotations

import hashlib

READ_SYSTEM = """You are a radiologist reading a CT report that someone else wrote. You record what the report says. You never judge whether it is right, and you never add findings of your own.

For each label in the list, decide what the report says about that finding:
- present: the report says the finding is there.
- absent: the report says the finding is not there, or calls the organ normal or unremarkable.
- not_mentioned: the report says nothing about it.

For present and absent, copy the shortest sentence or phrase from the report that says so, word for word. For not_mentioned, the quote is null.

Then list the other findings the report states as present that fit none of the labels, each as a word-for-word quote.

Answer with one JSON object and nothing else, in this shape:
{"labels": {"<label id>": {"state": "present", "quote": "<words from the report>"}}, "other_mentions": ["<words from the report>"]}
Use every label id from the list exactly once."""


def prompt_sha(*parts: str) -> str:
    return hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()


READ_PROMPT_SHA = prompt_sha(READ_SYSTEM)


def label_lines(labels: list[dict]) -> str:
    return "\n".join(f"- {r['id']}: {r['title']}" + (f" ({r['organ']})" if r["organ"] else "") for r in labels)


def read_messages(text: str, labels: list[dict]) -> list[dict]:
    """One read of one text: the radiologist report or NV-Reason-CT's report, the same way for both."""
    user = f"Labels:\n{label_lines(labels)}\n\nReport:\n<<<\n{text}\n>>>"
    return [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": user}]
