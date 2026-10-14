"""The reading radiologist's read: one call, strict JSON, checked against the label ids and the source text."""

from __future__ import annotations

import json

import pytest

from fake_llm import ScriptedLLM
from radar_desk.chat.llm import LlmError
from radar_desk.readers.labels import LABEL_IDS, LABELS
from radar_desk.readers.prompts import READ_SYSTEM, read_messages
from radar_desk.readers.reading import ReadError, parse_json, read_text, validate_read
from readers_seed import REPORT_READ, SYNTHETIC_REPORT, read_reply


def test_reads_a_report():
    llm = ScriptedLLM([REPORT_READ])
    read = read_text(llm, SYNTHETIC_REPORT, LABELS)
    assert set(read["labels"]) == set(LABEL_IDS)
    assert read["labels"]["ascites"] == {"state": "present", "quote": "Small volume ascites in the pelvis."}
    assert read["labels"]["renal_cyst"] == {"state": "absent", "quote": "The kidneys are unremarkable."}
    assert read["labels"]["gallstones"] == {"state": "not_mentioned", "quote": None}
    assert read["other_mentions"] == ["Degenerative change of the lumbar spine."]
    system, user = llm.calls[0]
    assert system == {"role": "system", "content": READ_SYSTEM}
    assert "- renal_cyst: Renal cyst (Kidney)" in user["content"] and "- ascites: Ascites" in user["content"]
    assert SYNTHETIC_REPORT in user["content"]


def test_fenced_json_and_unknown_or_missing_ids():
    data = json.loads(REPORT_READ)
    data["labels"]["made_up"] = {"state": "present", "quote": "x"}
    del data["labels"]["gallstones"]
    llm = ScriptedLLM(["Here you go:\n```json\n" + json.dumps(data) + "\n```"])
    read = read_text(llm, SYNTHETIC_REPORT, LABELS)
    assert "made_up" not in read["labels"]
    assert read["labels"]["gallstones"] == {"state": "not_mentioned", "quote": None}


def test_a_quote_that_is_not_in_the_text_is_dropped():
    reply = read_reply({"liver_lesion": ("present", "A large liver mass."),
                        "ascites": ("present", "small   volume ASCITES in the pelvis.")},
                       ["Not in the text at all."])
    read = read_text(ScriptedLLM([reply]), SYNTHETIC_REPORT, LABELS)
    assert read["labels"]["liver_lesion"] == {"state": "present", "quote": None}
    assert read["labels"]["ascites"]["quote"] == "small   volume ASCITES in the pelvis."
    assert read["other_mentions"] == []


def test_a_bad_answer_is_asked_again_once():
    bad = read_reply({"ascites": ("maybe", None)}, [])
    llm = ScriptedLLM([bad, REPORT_READ])
    assert read_text(llm, SYNTHETIC_REPORT, LABELS)["labels"]["ascites"]["state"] == "present"
    assert len(llm.calls) == 2


def test_two_bad_answers_fail():
    llm = ScriptedLLM(["no json here", LlmError("the model endpoint returned 503")])
    with pytest.raises(ReadError, match="503"):
        read_text(llm, SYNTHETIC_REPORT, LABELS)
    assert len(llm.calls) == 2


def test_parse_json_and_validate_errors():
    with pytest.raises(ReadError, match="no JSON object"):
        parse_json("nothing")
    with pytest.raises(ReadError, match="not valid JSON"):
        parse_json("{oops}")
    with pytest.raises(ReadError, match="no labels object"):
        validate_read({"labels": []}, SYNTHETIC_REPORT, LABEL_IDS)


def test_read_messages_shape():
    messages = read_messages("Some text.", LABELS[:1])
    assert messages[1]["content"] == "Labels:\n- liver_lesion: Liver lesion (Liver)\n\nReport:\n<<<\nSome text.\n>>>"
