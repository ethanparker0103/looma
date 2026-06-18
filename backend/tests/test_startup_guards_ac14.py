"""AC-14 startup-guard tests.

AC-14 requires the Looma app to validate its environment at startup:

1. **``ffmpeg`` (and ``ffprobe``) are on PATH** — both are
   required by the ingest stage. If either is missing, the process
   must exit with a clear, actionable error.

LLM key checks are no longer needed — LLM calls are made from
the frontend with user-provided keys (BYOK).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from app.main import _check_ffmpeg_or_exit  # noqa: E402

# Make ``app`` importable from the tests directory.
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def fake_no_ffmpeg(monkeypatch):
    """Patch :func:`shutil.which` so ``ffmpeg`` and ``ffprobe`` appear missing."""
    real_which = shutil.which

    def _fake_which(name: str) -> str | None:
        if name in ("ffmpeg", "ffprobe"):
            return None
        return real_which(name)

    monkeypatch.setattr(shutil, "which", _fake_which)
    yield monkeypatch


# --- _check_ffmpeg_or_exit: unit tests --------------------------------------


class TestCheckFfmpeg:
    def test_exits_with_code_1_when_missing(self, fake_no_ffmpeg, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _check_ffmpeg_or_exit()
        assert excinfo.value.code == 1

    def test_writes_to_stderr(self, fake_no_ffmpeg, capsys) -> None:
        with pytest.raises(SystemExit):
            _check_ffmpeg_or_exit()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "ffmpeg" in captured.err

    def test_message_is_actionable(self, fake_no_ffmpeg, capsys) -> None:
        with pytest.raises(SystemExit):
            _check_ffmpeg_or_exit()
        captured = capsys.readouterr()
        assert "ffmpeg" in captured.err.lower()
        assert "install" in captured.err.lower()

    def test_names_specific_missing_tool(self, capsys) -> None:
        real_which = shutil.which

        def _fake_which(name: str) -> str | None:
            if name == "ffmpeg":
                return None
            return real_which(name)

        with mock.patch("shutil.which", side_effect=_fake_which):
            with pytest.raises(SystemExit):
                _check_ffmpeg_or_exit()
        captured = capsys.readouterr()
        assert "ffmpeg" in captured.err

    def test_is_noop_when_present(self) -> None:
        _check_ffmpeg_or_exit()  # should not raise

    def test_message_says_startup_aborted(self, fake_no_ffmpeg, capsys) -> None:
        with pytest.raises(SystemExit):
            _check_ffmpeg_or_exit()
        captured = capsys.readouterr()
        assert "startup aborted" in captured.err.lower()


# --- Subprocess tests: the import-time guard actually fires -----------------


_IMPORT_SCRIPT = "import sys; sys.path.insert(0, '.'); from app.main import create_app"


def _run_subprocess(env: dict[str, str], *, path: str | None = None) -> subprocess.CompletedProcess:
    full_env = dict(env)
    if path is not None:
        full_env["PATH"] = path
    return subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=str(_BACKEND),
        timeout=30,
    )


def _env_with_minimum_required() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


@pytest.mark.slow
def test_subprocess_import_exits_zero_when_env_ok() -> None:
    result = _run_subprocess(_env_with_minimum_required())
    assert result.returncode == 0, f"stderr={result.stderr}"


@pytest.mark.slow
def test_subprocess_import_exits_1_when_no_ffmpeg() -> None:
    env = _env_with_minimum_required()
    result = _run_subprocess(env, path="/empty-no-such-directory")
    assert result.returncode == 1
    assert "ffmpeg" in result.stderr.lower()
    assert "install" in result.stderr.lower()
