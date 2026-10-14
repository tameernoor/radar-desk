"""The reading radiologist's calls: one chat-model call per output, strict JSON, checked before it is kept.

A read keeps a label's state only when it is one of three; a quote is kept only when it appears in the text it
claims to quote (case and spacing aside). A missing label counts as not_mentioned; unknown ids are dropped. An
unusable answer is asked again once, then the read fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

from radar_desk.chat.llm import LlmError, TextDelta
from radar_desk.gpu.nvreason_fake import YES_NO_SUFFIX
from radar_desk.readers.prompts import QUERY_RETRY, query_messages, read_messages

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


KINDS = ("yes_no", "finding", "follow_up", "differential", "reasoning")
MAX_QUERY_CHARS = 300

# What a prepared question may not say: it would reveal another reader, or ask what NV-Reason-CT cannot answer.
_BANNED = [
    (re.compile(r"\breports?\b|\breported\b", re.IGNORECASE), "mentions a report"),
    (re.compile(r"\bradiologists?\b", re.IGNORECASE), "mentions the radiologist"),
    (re.compile(r"\bradar\b", re.IGNORECASE), "mentions RADAR"),
    (re.compile(r"\b(?:another|other|second)\s+(?:reader|observer|model|opinion)\b|\bsomeone\b"
                r"|\b(?:doctor|clinician|physician|ai|algorithm)\b|\bare you sure\b", re.IGNORECASE),
     "hints at what another reader said"),
    (re.compile(r"\bscor(?:e|es|ed|ing)\b|\bprobabilit|\blikelihood\b|\bconfiden|\b0?\.\d+\b", re.IGNORECASE),
     "mentions a score"),
    (re.compile(r"%|\bper ?cent", re.IGNORECASE), "mentions a percentage"),
    (re.compile(r"\bmeasur|\bdiameter|\bdimensions?\b|\bhow (?:big|large)\b|\bwhat size\b|\bsize of\b"
                r"|\d\s*(?:mm|cm)\b|\bmillimet|\bcentimet", re.IGNORECASE), "asks for a measurement"),
    (re.compile(r"\bprior\b|\bprevious\b|\binterval\b|\bcompared (?:to|with) (?:the )?(?:prior|previous|earlier|last)"
                r"|\bstable\b|\bnew\b|\bgrown\b|\bgrowth\b|\bsince\b|\bchanged?\b", re.IGNORECASE),
     "mentions a prior scan"),
]

# Words that name a scored organ, for the check against organs mostly outside NV's cube.
ORGAN_WORDS = {
    "Liver": ("liver", "hepatic"),
    "Kidney": ("kidney", "kidneys", "renal"),
    "Spleen": ("spleen", "splenic"),
    "Pancreas": ("pancreas", "pancreatic"),
    "Gallbladder": ("gallbladder",),
    "Adrenal gland": ("adrenal", "adrenals"),
    "Aorta": ("aorta", "aortic"),
    "Esophagus": ("esophagus", "esophageal", "oesophagus", "hiatal"),
    "Lung": ("lung", "lungs", "pulmonary", "pleural"),
    "Rib": ("rib", "ribs"),
    "Heart": ("heart", "cardiac", "pericardial"),
    "Stomach": ("stomach", "gastric"),
    "Bladder": ("bladder",),
    "Small bowel": ("small bowel", "small intestine", "jejunum", "ileum"),
    "Large bowel": ("large bowel", "colon", "colonic", "rectum", "rectal"),
    "Duodenum": ("duodenum", "duodenal"),
    "Portal vein": ("portal vein",),
    "Sacrum": ("sacrum", "sacral"),
}

# Stems of derived terms (hepatic, nephropathy, pneumonia ...), mapped to the organ names the coverage uses.
ORGAN_STEMS = {
    "Liver": ("hepat",),
    "Spleen": ("splen",),
    "Kidney": ("nephr", "hydroneph"),
    "Lung": ("pneum", "atelecta"),
    "Gallbladder": ("cholecyst",),
    "Heart": ("pericardi",),
    "Rib": ("costal",),
}


def validate_query(q: Any, label_organs: dict[str, str | None], outside: list[str]) -> str | None:
    """Why this prepared question breaks NV-Reason-CT's limits, or None when it keeps to them."""
    if not isinstance(q, dict):
        return "not an object"
    text, kind, label = q.get("text"), q.get("kind"), q.get("label")
    if not isinstance(text, str) or not text.strip():
        return "no text"
    text = text.strip()
    if len(text) > MAX_QUERY_CHARS:
        return f"longer than {MAX_QUERY_CHARS} characters"
    if kind not in KINDS:
        return f"unknown kind {kind!r}"
    if label is not None and (not isinstance(label, str) or label not in label_organs):
        return f"unknown label {label!r}"
    if kind == "yes_no" and not text.endswith(YES_NO_SUFFIX):
        return f'a yes_no question must end with "{YES_NO_SUFFIX}"'
    for pattern, why in _BANNED:
        if pattern.search(text):
            return why
    if label is not None and label_organs[label] in outside:
        return f"asks about {label_organs[label]}, mostly outside the cube"
    for organ in outside:
        for word in ORGAN_WORDS.get(organ, (organ.lower(),)):
            if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
                return f"names {organ}, mostly outside the cube"
        for stem in ORGAN_STEMS.get(organ, ()):
            if re.search(rf"\b{re.escape(stem)}\w*", text, re.IGNORECASE):
                return f"names {organ}, mostly outside the cube"
    return None


def _split(raw: Any, label_organs: dict, outside: list[str]) -> tuple[list[dict], list[tuple[Any, str]]]:
    good, bad = [], []
    for q in raw if isinstance(raw, list) else []:
        why = validate_query(q, label_organs, outside)
        if why is None:
            good.append({"label": q.get("label"), "kind": q["kind"], "text": q["text"].strip()})
        else:
            bad.append((q, why))
    return good, bad


def write_queries(llm: Any, ctx: dict) -> list[dict]:
    """The prepared questions. Rejected ones are sent back once with the reasons; still bad, they are dropped."""
    label_organs = {r["id"]: r["organ"] for r in ctx["labels"]}
    messages = query_messages(ctx)
    answer, raw, problem = "", None, "no answer"
    for _ in range(ATTEMPTS):
        try:
            answer = complete(llm, messages)
            raw = parse_json(answer).get("queries")
            if not isinstance(raw, list):
                raise ReadError("no queries list")
            break
        except (ReadError, LlmError) as exc:
            problem, raw = str(exc), None
    if raw is None:
        raise ReadError(problem)
    good, bad = _split(raw, label_organs, ctx["outside"])
    if bad:
        lines = "\n".join(f"- {q.get('text') if isinstance(q, dict) else q} ({why})" for q, why in bad)
        retry = [*messages, {"role": "assistant", "content": answer},
                 {"role": "user", "content": f"{QUERY_RETRY}\n{lines}"}]
        try:
            again, _ = _split(parse_json(complete(llm, retry)).get("queries"), label_organs, ctx["outside"])
            good += again
        except (ReadError, LlmError):
            pass  # the good ones stand; the rejected ones are dropped
    out, seen = [], set()
    for q in good:
        qid = "q_" + hashlib.sha256(f"{q['label']}|{q['text']}".encode()).hexdigest()[:10]
        if qid not in seen:
            seen.add(qid)
            out.append({"id": qid, **q})
    return out
