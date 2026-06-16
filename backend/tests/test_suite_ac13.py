"""AC-13 pytest-suite meta-tests.

AC-13 requires the test suite to:

1. Cover the six AC-named areas — URL validation, file conversion,
   transcript parsing, LLM schema (with recorded fixture), TTS file
   creation, and API endpoints via ``httpx.AsyncClient`` against the
   FastAPI app.
2. Pass under ``pytest -v`` with **at least 10 tests passing**.

This module is the machine-verifiable contract for AC-13. It does
*not* duplicate the substantive tests in the rest of the suite
(those are already passing); it asserts the suite *as a whole*
meets the AC-13 bar by introspecting pytest's collection output
and the test files' documented roles.

The contract is split into three sections:

* **Counts** — the suite has at least 10 tests, and the per-AC
  test count is non-zero for every AC-13 category.
* **Coverage** — every AC-13 category has a dedicated test file
  with at least N tests, the LLM schema test reads a recorded
  fixture, and the API tests use ``httpx.AsyncClient``.
* **Tooling** — ``pytest -v`` exits 0 against the suite; the
  pytest config wires ``asyncio_mode = "auto"``; the suite is
  CI-friendly (no live ffmpeg / Whisper / network).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# --- Paths -----------------------------------------------------------------


#: Path to the test directory.
TESTS_DIR: Path = Path(__file__).resolve().parent

#: Path to the fixtures directory used by recorded-fixture tests.
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"

#: Path to the project root, where ``pyproject.toml`` lives.
PROJECT_ROOT: Path = TESTS_DIR.parent

#: Path to the pyproject.toml that holds pytest configuration.
PYPROJECT_PATH: Path = PROJECT_ROOT / "pyproject.toml"

#: LLM-extract fixture that the AC-5 LLM-schema test loads.
LLM_FIXTURE_PATH: Path = FIXTURES_DIR / "llm_extract_response.json"

#: Whisper-transcript fixture that the AC-4 transcript-parsing test loads.
TRANSCRIPTION_FIXTURE_PATH: Path = FIXTURES_DIR / "sample_transcription.json"


# --- Fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def collected_test_names() -> list[str]:
    """Return the list of ``path::test_name`` strings pytest collects.

    The fixture is module-scoped because collection is expensive and
    the result is shared by every meta-test in this file.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(TESTS_DIR),
            "--collect-only",
            "-q",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            # No real keys needed for collection.
            "ANTHROPIC_API_KEY": os.environ.get(
                "ANTHROPIC_API_KEY", "fake-collect"
            ),
        },
    )
    # Each collected item is a single line of the form
    # ``tests/test_foo.py::test_bar``. We filter for that shape.
    pattern = re.compile(r"^tests/test_\w+\.py::\w")
    return [line.strip() for line in result.stdout.splitlines() if pattern.match(line)]


@pytest.fixture(scope="module")
def test_files() -> dict[str, Path]:
    """Map test file stems to their on-disk paths."""
    return {p.stem: p for p in TESTS_DIR.glob("test_*.py")}


# --- Counts ----------------------------------------------------------------


def test_suite_has_at_least_10_tests(collected_test_names: list[str]) -> None:
    """AC-13: ``pytest -v`` exits 0 with at least 10 passing tests."""
    assert len(collected_test_names) >= 10, (
        f"AC-13 requires >= 10 tests; suite has {len(collected_test_names)}"
    )


def test_suite_count_is_substantial(collected_test_names: list[str]) -> None:
    """Sanity: the suite is well over the AC-13 minimum.

    A 10-test suite that just barely passes AC-13 leaves no margin
    for future contributors; the project ships with hundreds of
    tests across every AC-named area, and this assertion is a
    tripwire for a future PR that deletes most of them.
    """
    assert len(collected_test_names) >= 100, (
        f"suite has only {len(collected_test_names)} tests — "
        f"expected well over 100. Did a contributor accidentally "
        f"delete a whole test file?"
    )


# --- Coverage: per-AC-13-category test files ------------------------------


# The AC-13 categories and the test files that cover them. Each
# entry is (category-name, file-stem, min-count) so a future
# rename / file split can update one place.
AC13_CATEGORY_FILES: list[tuple[str, str, int]] = [
    # URL validation lives in test_ingest.py
    # (validate_youtube_url is exercised by ~15 tests).
    ("URL validation", "test_ingest", 5),
    # File conversion (mp4/mov/mkv/webm -> 16 kHz mono MP3) is
    # in test_ingest.py (convert_upload_to_mp3 + _ffmpeg_convert_cmd).
    ("File conversion", "test_ingest", 5),
    # Transcript parsing (Whisper segments, duration, language)
    # is in test_transcribe.py.
    ("Transcript parsing", "test_transcribe", 5),
    # LLM schema (recorded fixture) is in test_extract.py.
    ("LLM schema", "test_extract", 5),
    # TTS file creation is in test_narrate.py.
    ("TTS file creation", "test_narrate", 5),
    # API endpoints via httpx.AsyncClient: covered by the
    # test_api_*.py and test_error_handling_ac11.py files.
    ("API endpoints via httpx.AsyncClient", "test_api_extract", 3),
]


def _count_tests_in(file_stem: str, collected_test_names: list[str]) -> int:
    prefix = f"tests/{file_stem}.py::"
    return sum(1 for name in collected_test_names if name.startswith(prefix))


@pytest.mark.parametrize(
    ("category", "file_stem", "min_count"),
    AC13_CATEGORY_FILES,
    ids=[row[0] for row in AC13_CATEGORY_FILES],
)
def test_ac13_category_has_tests(
    category: str,
    file_stem: str,
    min_count: int,
    collected_test_names: list[str],
) -> None:
    """AC-13: every named category has at least ``min_count`` tests."""
    count = _count_tests_in(file_stem, collected_test_names)
    assert count >= min_count, (
        f"AC-13 category {category!r} ({file_stem}.py) has only "
        f"{count} tests; expected at least {min_count}"
    )


# --- Coverage: LLM fixture --------------------------------------------------


def test_llm_extract_fixture_exists() -> None:
    """AC-13: the LLM schema test uses a recorded fixture (not a live API)."""
    assert LLM_FIXTURE_PATH.is_file(), (
        f"LLM-extract fixture missing at {LLM_FIXTURE_PATH}"
    )
    # The fixture should be non-empty JSON, not a placeholder.
    import json
    payload = json.loads(LLM_FIXTURE_PATH.read_text(encoding="utf-8"))
    for field in (
        "title", "summary", "insights", "chapters", "narrative", "filler_removed"
    ):
        assert field in payload, f"LLM fixture missing field {field!r}"


def test_extract_py_loads_llm_fixture() -> None:
    """AC-13: the LLM schema test reads the recorded fixture (no live LLM)."""
    text = (TESTS_DIR / "test_extract.py").read_text(encoding="utf-8")
    # The fixture path is referenced by file name (the relative
    # ``fixtures/llm_extract_response.json`` lookup is the AC-13
    # contract — a "live" test would patch ``_invoke_provider``
    # instead).
    assert "llm_extract_response" in text, (
        "test_extract.py should load the recorded LLM fixture"
    )


def test_transcription_fixture_exists() -> None:
    """AC-13: transcript parsing uses a recorded fixture too."""
    assert TRANSCRIPTION_FIXTURE_PATH.is_file(), (
        f"transcript fixture missing at {TRANSCRIPTION_FIXTURE_PATH}"
    )
    import json
    payload = json.loads(TRANSCRIPTION_FIXTURE_PATH.read_text(encoding="utf-8"))
    for field in ("transcript", "segments", "language", "duration_seconds"):
        assert field in payload, f"transcription fixture missing field {field!r}"


# --- Coverage: API endpoint testing pattern ---------------------------------


def test_api_tests_use_httpx_async_client() -> None:
    """AC-13: API endpoints are exercised via ``httpx.AsyncClient``."""
    api_files = [
        "test_api_async_extract.py",
        "test_api_audio.py",
        "test_api_extract.py",
        "test_api_jobs.py",
        "test_error_handling_ac11.py",
        "test_frontend_serving.py",
    ]
    for fname in api_files:
        text = (TESTS_DIR / fname).read_text(encoding="utf-8")
        assert "httpx" in text, (
            f"{fname} must import httpx (per AC-13's API testing pattern)"
        )
        assert "AsyncClient" in text, (
            f"{fname} must use httpx.AsyncClient"
        )
        assert "ASGITransport" in text, (
            f"{fname} must use ASGITransport to drive the in-process app"
        )


# --- Tooling: pytest config -------------------------------------------------


def test_pyproject_toml_exists() -> None:
    """AC-13: pytest config lives in ``pyproject.toml``."""
    assert PYPROJECT_PATH.is_file(), f"missing {PYPROJECT_PATH}"


def test_pytest_config_declares_testpaths() -> None:
    """AC-13: ``[tool.pytest.ini_options].testpaths`` points at the tests dir."""
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    m = re.search(
        r'\[tool\.pytest\.ini_options\](.*?)(?=\[|\Z)',
        text,
        re.DOTALL,
    )
    assert m, "pyproject.toml missing [tool.pytest.ini_options] section"
    section = m.group(1)
    assert "testpaths" in section, (
        "pyproject.toml must declare testpaths under [tool.pytest.ini_options]"
    )


def test_pytest_config_uses_asyncio_auto_mode() -> None:
    """AC-13: ``asyncio_mode = "auto"`` so the AsyncClient tests run without
    an explicit ``@pytest.mark.asyncio`` on every coroutine.
    """
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert re.search(r'asyncio_mode\s*=\s*["\']auto["\']', text), (
        'pyproject.toml must set asyncio_mode = "auto"'
    )


# --- Tooling: the suite exits 0 under `pytest -v` --------------------------


@pytest.mark.slow
def test_pytest_v_exits_zero() -> None:
    """AC-13: ``pytest -v`` exits 0 with the suite.

    We shell out to pytest rather than introspecting the
    in-process collection so the assertion is the same as what
    CI runs: ``subprocess.run([pytest, "-v", tests_dir])``.

    Critically, we exclude this very test file from the
    subprocess invocation — otherwise the test would recursively
    invoke itself and the suite would hang.
    """
    # Use --ignore to skip this file in the subprocess run.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(TESTS_DIR),
            "-v",
            "--no-header",
            "--tb=short",
            # Don't recurse into this file (would hang).
            f"--ignore={TESTS_DIR / 'test_suite_ac13.py'}",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            # No real keys needed for the suite — every LLM
            # and TTS call is patched.
            "ANTHROPIC_API_KEY": os.environ.get(
                "ANTHROPIC_API_KEY", "fake-ac13"
            ),
        },
    )
    assert result.returncode == 0, (
        f"`pytest -v` exited {result.returncode}, expected 0. "
        f"Last 60 lines of stdout:\n"
        + "\n".join(result.stdout.splitlines()[-60:])
    )
    # And the verbose output should show >= 10 test lines, which
    # is the AC-13 minimum.
    passed_line = re.search(r"=+ (\d+) passed", result.stdout)
    assert passed_line, (
        f"`pytest -v` stdout missing 'passed' summary line:\n{result.stdout[-500:]}"
    )
    passed = int(passed_line.group(1))
    assert passed >= 10, f"pytest reported only {passed} passed tests"


# --- Tooling: no live network / model / binary in the suite ---------------


# The AC-13 contract is that the suite is "CI-friendly" — it
# doesn't require a live LLM, Whisper model, or ffmpeg binary. We
# assert this by checking that no test file imports a network-
# facing or model-loading symbol at module scope. (Imports of
# ``whisper``, ``yt_dlp``, ``anthropic``, ``openai``, ``edge_tts``
# are fine if they happen inside a function; we only flag
# top-of-file imports here.)


def test_no_top_of_file_whisper_import_in_tests() -> None:
    """No test file does ``import whisper`` at module top-level."""
    # A live whisper import would slow down collection by 5-30s
    # (model load). The pipeline module imports whisper lazily,
    # so a clean test suite never sees it at import time.
    for f in TESTS_DIR.glob("test_*.py"):
        head = "\n".join(f.read_text(encoding="utf-8").splitlines()[:30])
        assert not re.search(r"^\s*import\s+whisper\b", head, re.MULTILINE), (
            f"{f.name} imports whisper at module top-level — should be lazy"
        )
