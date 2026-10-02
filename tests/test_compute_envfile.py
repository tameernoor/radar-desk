"""The compute command's `.env` reader."""

from __future__ import annotations

from radar_desk.compute.envfile import read_env


def test_read_env_quotes_export_and_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# c\n\nexport A=1\nB='two words'\nC=\"x # y\"\nD=plain # note\n#E=no\nA=last\nbad line\n")
    assert read_env(env) == {"A": "last", "B": "two words", "C": "x # y", "D": "plain"}
    assert read_env(tmp_path / "missing") == {}


def test_read_env_crlf_and_last_definition_wins(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"GPU_BACKEND=fake\r\nA=1\r\n  export GPU_BACKEND = modal\r\nLAST=no-newline")
    assert read_env(env) == {"GPU_BACKEND": "modal", "A": "1", "LAST": "no-newline"}
