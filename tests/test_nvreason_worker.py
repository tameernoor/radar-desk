"""The NV-Reason-CT worker: pure helpers here, the Modal app only as far as it can be checked without a GPU."""

from __future__ import annotations

import importlib
import socket
import sys
import warnings
from pathlib import Path

import pytest

from radar_worker.nvreason import (
    CROP,
    REGION,
    REPORT_MAX_NEW_TOKENS,
    REPORT_PROMPT,
    REVISION,
    crop_summary,
    split_output,
)

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "worker"


def test_split_output_with_thinking_and_answer_tag():
    decoded = "<think>Look at the liver.</think>\nFINDINGS:\nLiver: small cyst.\n<answer> Liver cyst </answer>\n"
    assert split_output(decoded) == {
        "thinking": "Look at the liver.",
        "text": "FINDINGS:\nLiver: small cyst.\n<answer> Liver cyst </answer>",
        "answer": "FINDINGS:\nLiver: small cyst.",
        "answer_tag": "Liver cyst",
    }


def test_split_output_without_thinking_or_tag():
    assert split_output("Yes") == {"thinking": "", "text": "Yes", "answer": "Yes", "answer_tag": None}


def test_split_output_uses_the_last_think_close():
    out = split_output("a</think>b</think> answer")
    assert out["thinking"] == "a</think>b" and out["answer"] == "answer"


def test_crop_summary_cuts_below_and_above():
    s = crop_summary([-190.0, -30.0, -470.0], [194.0, 354.0, -86.0], [18, 18, 14], nz=270, spacing_mm=2.0,
                     fallback=False, region="abdomen")
    assert s == {
        "region": "abdomen",
        "box_mm": [[-190.0, -30.0, -470.0], [194.0, 354.0, -86.0]],
        "crop_offset_vox": [18, 18, 14],
        "scan_length_mm": 540.0,
        "cut_below_mm": 28.0,
        "cut_above_mm": round((270 - 14 - CROP) * 2.0, 1),
        "fallback": False,
    }


def test_crop_summary_never_negative():
    s = crop_summary([0, 0, 0], [1, 1, 1], [0, 0, 0], nz=100, spacing_mm=2.0, fallback=True, region="abdomen")
    assert s["cut_below_mm"] == 0.0 and s["cut_above_mm"] == 0.0 and s["fallback"] is True


def test_the_job_prompt_is_the_proven_one():
    assert REPORT_PROMPT == "Write a structured abdominal CT report."
    assert REPORT_MAX_NEW_TOKENS == 2048 and REGION == "abdomen"
    assert REVISION == "386b93e034983f6c1fc841a43833a1b6a0cd9c13"


@pytest.fixture()
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"network access during import: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_nvreason_app_imports_without_side_effects(monkeypatch, no_network):
    modal = pytest.importorskip("modal")
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(WORKER))
    monkeypatch.setenv("MODAL_DATA_VOLUME", "radar-data-test")
    sys.modules.pop("nvreason_app", None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app_module = importlib.import_module("nvreason_app")
            assert isinstance(app_module.app, modal.App)
            assert app_module.app.name == "radar-desk-nvreason"
            assert "NvReason" in app_module.app.registered_classes
        options = app_module.CLASS_OPTIONS
        assert options["gpu"] == "L4" and options["scaledown_window"] == 300
        assert options["max_containers"] == 1 and options["retries"] == 0
        assert sorted(options["volumes"]) == ["/cache", "/data"]
        assert app_module.DATA_VOLUME == "radar-data-test"
    finally:
        sys.modules.pop("nvreason_app", None)


def test_nvreason_app_takes_the_volume_helpers_from_radar_worker():
    import ast

    tree = ast.parse((WORKER / "nvreason_app.py").read_text(encoding="utf-8"))
    imported = {a.name for node in tree.body if isinstance(node, ast.ImportFrom)
                and node.module == "radar_worker.volume" for a in node.names}
    assert {"VOLUME_SCHEME", "volume_path"} <= imported
