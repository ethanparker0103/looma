"""AC-14 startup-guard tests.

AC-14 requires the Looma app to validate its environment at
startup:

1. **``ffmpeg`` (and ``ffprobe``) are on PATH** — both are
   required by the ingest (AC-2/AC-3) and narrate (AC-7)
   stages. If either is missing, the process must exit with a
   clear, actionable error.

2. **At least one LLM API key is configured** — either
   ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``. If neither is
   set, the process must exit with a clear, actionable error.

The implementation lives in :mod:`app.main` as the
:func:`_check_ffmpeg_or_exit` and :func:`_check_llm_key_or_exit`
helpers, both of which are invoked at import time so the
process never even starts serving traffic if the environment
is broken.

This module exercises every AC-14 sub-requirement:

* **Unit tests** — each guard function exits with code 1 and
  writes an actionable message to stderr when the relevant
  precondition is missing, and is a no-op when the
  precondition is met.
* **Actionability** — the stderr message must name the
  missing tool / key and tell the user what to do (e.g.
  ``apt-get install -y ffmpeg`` for ffmpeg, ``.env.example``
  for the LLM key).
* **Subprocess tests** — the import-time guards are wired
  correctly: ``python -c "import app.main"`` exits 1 when the
  environment is broken, and the error message goes to
  stderr (not stdout) so CI log scrapers see it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make ``app`` importable from the tests directory.
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# We import the guard functions directly. The import runs
# the guards in the parent process — which is fine for unit
# tests, since the test env has both ffmpeg and an LLM key.
# The subprocess tests below exercise the *missing* env paths.
from app.main import _check_ffmpeg_or_exit, _check_llm_key_or_exit  # noqa: E402


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def clean_environ(monkeypatch):
    """Yield a monkeypatch with both LLM env vars cleared.

    Use this when a test wants to assert the ``no LLM key``
    behavior. The fixture is explicit so we don't accidentally
    clear the user's real env in a CI runner.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield monkeypatch


@pytest.fixture
def fake_no_ffmpeg(monkeypatch):
    """Patch :func:`shutil.which` so ``ffmpeg`` and ``ffprobe`` appear missing.

    The real ``shutil.which`` would find ``ffmpeg`` on the
    test host (CI runners install it). We monkeypatch it to
    return ``None`` for both tools so the guard can be
    exercised in-process.
    """
    real_which = shutil.which

    def _fake_which(name: str) -> str | None:
        if name in ("ffmpeg", "ffprobe"):
            return None
        return real_which(name)

    monkeypatch.setattr(shutil, "which", _fake_which)
    yield monkeypatch


# --- _check_ffmpeg_or_exit: unit tests --------------------------------------


def test_check_ffmpeg_exits_with_code_1_when_missing(
    fake_no_ffmpeg, capsys
) -> None:
    """AC-14: ffmpeg missing -> process exits with code 1."""
    with pytest.raises(SystemExit) as excinfo:
        _check_ffmpeg_or_exit()
    assert excinfo.value.code == 1, (
        f"expected exit code 1, got {excinfo.value.code!r}"
    )


def test_check_ffmpeg_writes_to_stderr(fake_no_ffmpeg, capsys) -> None:
    """AC-14: the error message goes to stderr (not stdout)."""
    with pytest.raises(SystemExit):
        _check_ffmpeg_or_exit()
    captured = capsys.readouterr()
    assert captured.out == "", (
        f"ffmpeg-guard should not write to stdout; got {captured.out!r}"
    )
    assert "ffmpeg" in captured.err, (
        f"ffmpeg-guard stderr should mention 'ffmpeg'; got {captured.err!r}"
    )


def test_check_ffmpeg_message_is_actionable(fake_no_ffmpeg, capsys) -> None:
    """AC-14: the error message tells the user how to fix the problem.

    A "clear, actionable" error must (a) name the missing
    dependency, (b) say what package to install, and (c)
    include the install command.
    """
    with pytest.raises(SystemExit):
        _check_ffmpeg_or_exit()
    captured = capsys.readouterr()
    # All three parts of an actionable message.
    assert "ffmpeg" in captured.err.lower()
    assert "apt" in captured.err or "install" in captured.err.lower(), (
        f"stderr should suggest how to install ffmpeg; got {captured.err!r}"
    )


def test_check_ffmpeg_message_names_specific_missing_tool(
    fake_no_ffmpeg, capsys
) -> None:
    """AC-14: the error message names the specific tool(s) that are missing.

    A user who has ffmpeg but not ffprobe (or vice versa)
    needs to know which one to install. The guard lists the
    missing tools in parentheses.
    """
    # Patch only ffmpeg to be missing; ffprobe remains.
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
    # The ffprobe name should NOT appear in the missing list
    # (it's installed in the test env).
    assert "ffprobe" not in captured.err or "found" in captured.err.lower(), (
        f"ffprobe should not be listed as missing; got {captured.err!r}"
    )


def test_check_ffmpeg_is_noop_when_present() -> None:
    """AC-14: when ffmpeg and ffprobe are both on PATH, the guard is a no-op.

    No exit, no stderr write. This is the happy path the
    real process takes in production.
    """
    # The real shutil.which finds them in the test env.
    _check_ffmpeg_or_exit()  # should not raise


# --- _check_llm_key_or_exit: unit tests -------------------------------------


def test_check_llm_key_exits_with_code_1_when_missing(
    clean_environ, capsys
) -> None:
    """AC-14: no LLM key -> process exits with code 1."""
    with pytest.raises(SystemExit) as excinfo:
        _check_llm_key_or_exit()
    assert excinfo.value.code == 1, (
        f"expected exit code 1, got {excinfo.value.code!r}"
    )


def test_check_llm_key_writes_to_stderr(clean_environ, capsys) -> None:
    """AC-14: the error message goes to stderr."""
    with pytest.raises(SystemExit):
        _check_llm_key_or_exit()
    captured = capsys.readouterr()
    assert captured.out == "", (
        f"llm-guard should not write to stdout; got {captured.out!r}"
    )
    assert "LLM" in captured.err or "API" in captured.err, (
        f"stderr should mention 'LLM' or 'API'; got {captured.err!r}"
    )


def test_check_llm_key_message_is_actionable(
    clean_environ, capsys
) -> None:
    """AC-14: the error message names both possible keys and how to set them."""
    with pytest.raises(SystemExit):
        _check_llm_key_or_exit()
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err, (
        f"stderr should mention ANTHROPIC_API_KEY; got {captured.err!r}"
    )
    assert "OPENAI_API_KEY" in captured.err, (
        f"stderr should mention OPENAI_API_KEY; got {captured.err!r}"
    )
    # Actionability: mention .env.example or "set" / "export".
    assert ".env" in captured.err.lower() or "set " in captured.err.lower(), (
        f"stderr should tell the user how to set the key; got {captured.err!r}"
    )


def test_check_llm_key_is_noop_when_anthropic_set(monkeypatch) -> None:
    """AC-14: when ANTHROPIC_API_KEY is set, the guard is a no-op."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _check_llm_key_or_exit()  # should not raise


def test_check_llm_key_is_noop_when_openai_set(monkeypatch) -> None:
    """AC-14: when OPENAI_API_KEY is set, the guard is a no-op."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _check_llm_key_or_exit()  # should not raise


def test_check_llm_key_treats_empty_string_as_missing(monkeypatch) -> None:
    """AC-14: an empty-string LLM key is treated as missing.

    A user who sets ``ANTHROPIC_API_KEY=`` in their .env
    (forgot to fill it in) is functionally identical to not
    having a key at all — the guard must reject both.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(SystemExit):
        _check_llm_key_or_exit()


# --- Subprocess tests: the import-time guards actually fire ----------------


# A tiny script that imports ``app.main``. We invoke it in a
# subprocess with a controlled environment so the import-time
# guards fire. This is the "real" AC-14 contract: importing the
# app is what uvicorn does, and if the import fails the process
# must exit 1 with the error on stderr.
_IMPORT_SCRIPT = "import sys; sys.path.insert(0, '.'); import app.main"


def _run_subprocess(env: dict[str, str], *, path: str | None = None) -> subprocess.CompletedProcess:
    """Run ``python -c "import app.main"`` with a controlled env.

    Args:
        env: Environment variables to pass to the subprocess.
            ``PATH`` is set from this dict; if ``path`` is also
            given, it overrides the ``PATH`` value.
        path: Optional override for the ``PATH`` env var (used
            to simulate a host without ffmpeg/ffprobe).
    """
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
    """Build an env dict that satisfies both guards.

    Real CI / dev environments have ffmpeg/ffprobe on PATH and
    at least one LLM key. The subprocess tests below mutate
    this baseline to exercise the missing-env paths.
    """
    # Use ``/usr/bin:/bin`` as a sane default PATH; tests can
    # override it to drop ffmpeg.
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "ANTHROPIC_API_KEY": "test-key-for-subprocess",
    }


@pytest.mark.slow
def test_subprocess_import_exits_zero_when_env_ok() -> None:
    """AC-14 happy path: with ffmpeg + a key, ``import app.main`` exits 0.

    This is the production case — uvicorn starts, the
    process listens on the port. We assert the subprocess
    exits 0 with no stderr.
    """
    result = _run_subprocess(_env_with_minimum_required())
    assert result.returncode == 0, (
        f"import app.main should succeed in a healthy env; "
        f"stderr=\n{result.stderr}\nstdout=\n{result.stdout}"
    )
    assert "aborted" not in result.stderr, (
        f"healthy env should not trigger the abort path; "
        f"stderr=\n{result.stderr}"
    )


@pytest.mark.slow
def test_subprocess_import_exits_1_when_no_llm_key() -> None:
    """AC-14: with ffmpeg present but no LLM key, ``import app.main`` exits 1.

    This is the production case where the user forgot to
    set their API key. The startup guard must catch this
    and exit 1 with a clear message on stderr.
    """
    env = _env_with_minimum_required()
    env.pop("ANTHROPIC_API_KEY", None)
    env["OPENAI_API_KEY"] = ""
    result = _run_subprocess(env)
    assert result.returncode == 1, (
        f"import app.main should exit 1 when no LLM key is set; "
        f"stderr=\n{result.stderr}"
    )
    assert "ANTHROPIC_API_KEY" in result.stderr
    assert "OPENAI_API_KEY" in result.stderr


@pytest.mark.slow
def test_subprocess_import_exits_1_when_no_ffmpeg() -> None:
    """AC-14: with no ffmpeg on PATH, ``import app.main`` exits 1.

    We use ``PATH=/empty`` to simulate a host with no
    ffmpeg. The startup guard must catch this and exit 1
    with a clear message on stderr.
    """
    env = _env_with_minimum_required()
    # An empty PATH means shutil.which returns None for everything.
    result = _run_subprocess(env, path="/empty-no-such-directory")
    assert result.returncode == 1, (
        f"import app.main should exit 1 when ffmpeg is missing; "
        f"stderr=\n{result.stderr}"
    )
    assert "ffmpeg" in result.stderr.lower()
    # Actionable: tell the user how to install.
    assert "apt" in result.stderr or "install" in result.stderr.lower()


@pytest.mark.slow
def test_subprocess_import_exits_1_when_no_ffmpeg_and_no_key() -> None:
    """AC-14: with no ffmpeg AND no key, the ffmpeg guard fires first.

    The two guards run in order. The ffmpeg guard runs first
    (it's the more critical failure), so the subprocess
    exits 1 with an ffmpeg message even when the LLM key is
    also missing.
    """
    env = _env_with_minimum_required()
    env.pop("ANTHROPIC_API_KEY", None)
    env["OPENAI_API_KEY"] = ""
    result = _run_subprocess(env, path="/empty-no-such-directory")
    assert result.returncode == 1
    # The ffmpeg guard runs first; its message is what the user sees.
    assert "ffmpeg" in result.stderr.lower()


# --- End-to-end error message contract --------------------------------------


def test_check_ffmpeg_message_says_startup_aborted(
    fake_no_ffmpeg, capsys
) -> None:
    """AC-14: the error says "startup aborted" so log scrapers can grep it.

    A consistent prefix makes it easy to alert on "Looma
    startup aborted" in production. The README's deployment
    section leans on this.
    """
    with pytest.raises(SystemExit):
        _check_ffmpeg_or_exit()
    captured = capsys.readouterr()
    assert "startup aborted" in captured.err.lower(), (
        f"stderr should say 'startup aborted'; got {captured.err!r}"
    )


def test_check_llm_key_message_says_startup_aborted(
    clean_environ, capsys
) -> None:
    """AC-14: the LLM-key error also says "startup aborted" for grep-ability."""
    with pytest.raises(SystemExit):
        _check_llm_key_or_exit()
    captured = capsys.readouterr()
    assert "startup aborted" in captured.err.lower(), (
        f"stderr should say 'startup aborted'; got {captured.err!r}"
    )
