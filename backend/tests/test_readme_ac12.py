"""AC-12 README-documentation tests.

AC-12 requires ``README.md`` to document:

1. **Prerequisites** — Python 3.11+, ``ffmpeg``, and an LLM API key
   (``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``).
2. **Environment variables** — the env vars that affect runtime.
3. **Install** — how to set up a fresh checkout.
4. **Run** — how to launch the dev server.
5. **Demo screenshot path** — the literal string ``docs/demo.png`` must
   appear in the README (and ideally be a real file in the repo).

These tests parse the README as a single string and assert each
section is present. They are intentionally strict about the
demo-path requirement because AC-12 is the only AC that hinges on
that literal string being findable by a fresh user.

The README is loaded from the repo root (``../README.md`` relative
to this test file), so the tests work no matter where pytest is
invoked from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# --- Paths -----------------------------------------------------------------


#: Path to the repo root, computed from this test file's location.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
README_PATH: Path = REPO_ROOT / "README.md"
DEMO_PNG_PATH: Path = REPO_ROOT / "docs" / "demo.png"


@pytest.fixture(scope="module")
def readme() -> str:
    """Load the README once per test module.

    A module-scoped fixture keeps the I/O cheap across the 20+
    assertions in this file.
    """
    return README_PATH.read_text(encoding="utf-8")


# --- File exists -----------------------------------------------------------


def test_readme_exists() -> None:
    """AC-12: ``README.md`` must be present at the repo root."""
    assert README_PATH.is_file(), f"README not found at {README_PATH}"


def test_readme_is_non_trivially_long() -> None:
    """Sanity: the README is more than a stub (>= 1000 chars).

    A 1-line stub would pass ``test_readme_exists`` but wouldn't
    actually document anything. AC-12 is about documentation
    quality, not just file presence.
    """
    text = README_PATH.read_text(encoding="utf-8")
    assert len(text) >= 1000, (
        f"README is suspiciously short ({len(text)} chars) — "
        f"AC-12 expects real documentation."
    )


# --- Demo screenshot placeholder -------------------------------------------


def test_readme_mentions_docs_demo_png(readme: str) -> None:
    """AC-12: the literal ``docs/demo.png`` path is in the README."""
    assert "docs/demo.png" in readme, (
        "README must mention the docs/demo.png placeholder path"
    )


def test_readme_image_markup_uses_docs_demo_png(readme: str) -> None:
    """AC-12: the README references the demo image via Markdown markup.

    A bare mention in a comment is not enough — the README must
    *link* the path so a Markdown renderer would actually display
    the image. We accept either ``![](docs/demo.png)`` or
    ``![...](docs/demo.png)``.
    """
    pattern = r"!\[[^\]]*\]\(\.?/?docs/demo\.png\)"
    assert re.search(pattern, readme), (
        f"README must embed docs/demo.png via Markdown image syntax; "
        f"searched for {pattern!r}"
    )


def test_docs_demo_png_placeholder_file_exists() -> None:
    """AC-12: the ``docs/demo.png`` placeholder file is in the repo.

    AC-12 says the README *documents* the path. The path is only
    useful if the file actually exists in the repo (so the link
    resolves on a fresh clone). We check for any non-empty file
    at that path; a real demo screenshot will replace the
    placeholder whenever a maintainer has one to share.
    """
    assert DEMO_PNG_PATH.is_file(), f"placeholder missing at {DEMO_PNG_PATH}"
    assert DEMO_PNG_PATH.stat().st_size > 0, (
        f"{DEMO_PNG_PATH} is empty — drop a real demo image there"
    )


# --- Prerequisites ----------------------------------------------------------


def test_readme_mentions_python_3_11(readme: str) -> None:
    """AC-12: Python 3.11+ is listed as a prerequisite."""
    assert re.search(r"python\s*3\.11\+?", readme, re.IGNORECASE), (
        "README must document Python 3.11+ as a prerequisite"
    )


def test_readme_mentions_ffmpeg(readme: str) -> None:
    """AC-12: ``ffmpeg`` is listed as a prerequisite."""
    assert "ffmpeg" in readme.lower(), (
        "README must document ffmpeg as a prerequisite"
    )


def test_readme_mentions_anthropic_api_key(readme: str) -> None:
    """AC-12: ``ANTHROPIC_API_KEY`` is documented as a key option."""
    assert "ANTHROPIC_API_KEY" in readme, (
        "README must document ANTHROPIC_API_KEY"
    )


def test_readme_mentions_openai_api_key(readme: str) -> None:
    """AC-12: ``OPENAI_API_KEY`` is documented as a key option."""
    assert "OPENAI_API_KEY" in readme, (
        "README must document OPENAI_API_KEY"
    )


# --- Environment variables --------------------------------------------------


def test_readme_has_environment_variables_section(readme: str) -> None:
    """AC-12: there's a top-level "Environment Variables" section."""
    # Match a Markdown header that begins a section about env vars.
    pattern = r"^#{1,3}\s+.*[Ee]nviron.*[Vv]ar.*"
    assert re.search(pattern, readme, re.MULTILINE), (
        "README must have an Environment Variables section"
    )


def test_readme_documents_data_dir(readme: str) -> None:
    """The ``DATA_DIR`` env var is documented (commonly tweaked)."""
    assert "DATA_DIR" in readme, (
        "README must document the DATA_DIR env var"
    )


def test_readme_documents_max_video_seconds(readme: str) -> None:
    """``MAX_VIDEO_SECONDS`` is documented (controls the 90-min cap)."""
    assert "MAX_VIDEO_SECONDS" in readme, (
        "README must document MAX_VIDEO_SECONDS"
    )


def test_readme_documents_max_upload_mb(readme: str) -> None:
    """``MAX_UPLOAD_MB`` is documented (controls the 200-MB upload cap)."""
    assert "MAX_UPLOAD_MB" in readme, (
        "README must document MAX_UPLOAD_MB"
    )


def test_readme_documents_max_pipeline_seconds(readme: str) -> None:
    """AC-10 budget: ``MAX_PIPELINE_SECONDS`` is documented.

    AC-10 introduced the 5-minute pipeline budget as an env-overridable
    constant; the README should reflect it so operators can tighten
    the budget in CI / staging.
    """
    assert "MAX_PIPELINE_SECONDS" in readme, (
        "README must document the MAX_PIPELINE_SECONDS budget"
    )


# --- Install instructions --------------------------------------------------


def test_readme_has_install_section(readme: str) -> None:
    """AC-12: there's a top-level Install / Quick Start section."""
    pattern = r"^#{1,3}\s+.*(Install|Quick Start)"
    assert re.search(pattern, readme, re.MULTILINE | re.IGNORECASE), (
        "README must have an Install (or Quick Start) section"
    )


def test_readme_documents_pip_install_requirements(readme: str) -> None:
    """Install: a ``pip install -r requirements.txt`` line is shown."""
    assert "pip install -r requirements.txt" in readme, (
        "README must show `pip install -r requirements.txt`"
    )


def test_readme_documents_python_venv(readme: str) -> None:
    """Install: the ``python -m venv .venv`` flow is shown."""
    assert re.search(r"python\s*-m\s*venv\s+\.venv", readme), (
        "README must show `python -m venv .venv`"
    )


def test_readme_documents_env_template_copy(readme: str) -> None:
    """Install: a ``cp .env.example .env`` line is shown."""
    assert "cp .env.example .env" in readme, (
        "README must show `cp .env.example .env`"
    )


# --- Run instructions ------------------------------------------------------


def test_readme_has_run_section(readme: str) -> None:
    """AC-12: there's a top-level Run / Quick Start section."""
    pattern = r"^#{1,3}\s+.*\bRun\b"
    assert re.search(pattern, readme, re.MULTILINE), (
        "README must have a Run section"
    )


def test_readme_documents_run_sh(readme: str) -> None:
    """Run: ``bash run.sh`` is shown as the launch command."""
    assert "bash run.sh" in readme, (
        "README must show `bash run.sh` as a launch command"
    )


def test_readme_documents_uvicorn_invocation(readme: str) -> None:
    """Run: a direct ``uvicorn`` invocation is shown.

    Operators who don't use ``run.sh`` (e.g. system-D services,
    container CMD) need a one-liner they can copy.
    """
    assert re.search(r"uvicorn\s+app\.main:app", readme), (
        "README must show a direct `uvicorn app.main:app` invocation"
    )


# --- API reference ---------------------------------------------------------


def test_readme_documents_api_endpoints(readme: str) -> None:
    """Bonus (not strictly AC-12): the API endpoints are documented.

    Operators need to know about ``/api/extract`` and ``/api/jobs``
    to drive Looma from curl / a script. We assert the success-path
    verbs and paths are present.
    """
    assert "/api/extract" in readme, "README must document /api/extract"
    assert "/api/jobs" in readme, "README must document /api/jobs"


def test_readme_documents_error_shape(readme: str) -> None:
    """AC-11 is paired with AC-12: the canonical error shape is documented.

    The README's Error Shape section pins the
    ``{"error", "code"}`` body and the status-code allow-list so
    JS clients can rely on the contract.
    """
    assert '"error"' in readme or "'error'" in readme, (
        "README must document the 'error' field of the canonical body"
    )
    assert '"code"' in readme or "'code'" in readme, (
        "README must document the 'code' field of the canonical body"
    )
    # Status-code allow-list.
    for code in ("400", "404", "413", "415", "500"):
        assert code in readme, f"README must mention status code {code}"


# --- Performance section (AC-10) ------------------------------------------


def test_readme_documents_perf_budget(readme: str) -> None:
    """AC-10 is paired with AC-12: the 5-min budget is documented."""
    # Look for either "5 minutes" or "300 s" or "MAX_PIPELINE_SECONDS".
    assert (
        "5 minutes" in readme
        or "300" in readme
        or "MAX_PIPELINE_SECONDS" in readme
    ), "README must document the AC-10 5-minute pipeline budget"
