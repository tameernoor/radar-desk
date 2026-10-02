"""The compute switch's `.env` reader and single-key editor."""

from __future__ import annotations

import os

from radar_desk.compute.envfile import read_env, set_key


def test_replace_keeps_every_other_byte(tmp_path):
    env = tmp_path / ".env"
    before = (b"# radar desk\r\n\r\nOWNER_TOKEN=abc\r\n#GPU_BACKEND=modal\r\n"
              b"GPU_BACKEND=fake\r\n\r\nexport X='y z'\r\nLAST=no-newline")
    env.write_bytes(before)
    os.chmod(env, 0o600)
    assert set_key(env, "GPU_BACKEND", "worker") is True
    assert env.read_bytes() == before.replace(b"GPU_BACKEND=fake\r\n", b"GPU_BACKEND=worker\r\n")
    assert env.stat().st_mode & 0o777 == 0o600
    assert set_key(env, "GPU_BACKEND", "worker") is False


def test_commented_line_is_ignored_and_missing_key_appended(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"#GPU_BACKEND=modal\nA=1")
    os.chmod(env, 0o640)
    assert set_key(env, "GPU_BACKEND", "worker") is True
    assert env.read_bytes() == b"#GPU_BACKEND=modal\nA=1\nGPU_BACKEND=worker\n"
    assert env.stat().st_mode & 0o777 == 0o640


def test_export_line_and_new_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("  export GPU_BACKEND = fake\n")
    set_key(env, "GPU_BACKEND", "modal")
    assert env.read_text() == "  export GPU_BACKEND = modal\n"
    new = tmp_path / "new.env"
    set_key(new, "GPU_BACKEND", "worker")
    assert new.read_text() == "GPU_BACKEND=worker\n"
    assert new.stat().st_mode & 0o777 == 0o600
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == [".env"]


def test_read_env_quotes_export_and_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# c\n\nexport A=1\nB='two words'\nC=\"x # y\"\nD=plain # note\n#E=no\nA=last\nbad line\n")
    assert read_env(env) == {"A": "last", "B": "two words", "C": "x # y", "D": "plain"}
    assert read_env(tmp_path / "missing") == {}


def test_duplicate_lines_are_all_rewritten(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GPU_BACKEND=fake\nA=1\nGPU_BACKEND=modal\n")
    set_key(env, "GPU_BACKEND", "worker")
    assert env.read_text() == "GPU_BACKEND=worker\nA=1\nGPU_BACKEND=worker\n"
    assert read_env(env)["GPU_BACKEND"] == "worker"
