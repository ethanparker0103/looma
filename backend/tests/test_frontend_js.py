"""Unit tests for the frontend's pure helper functions (AC-8).

The frontend ``app.js`` exposes a tiny ``Loomafmt`` global with
three pure functions: ``formatTimestamp``, ``buildMarkdown``, and
``escapeHtml``. We drive them under Node.js (already installed in
the test environment) so the JS logic is verified end-to-end
without needing a headless browser.

The Node script returns a single JSON object on stdout, which the
test parses. This keeps the test self-contained — no extra Python
deps (no jsdom, no playwright) — and runs in well under a second.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
APP_JS_PATH = FRONTEND_DIR / "app.js"


# A loader script that we feed to node. It pulls in app.js, then
# invokes the requested helper on a few sample inputs and prints a
# single JSON object with the results. The harness expects a single
# line of JSON to keep parsing simple.
#
# We pass the input as a base64-encoded JSON blob (so quoting /
# escaping can't break the script) and read app.js from disk.
_LOADER_TEMPLATE = dedent(
    """
    // jsdom-lite stubs so app.js doesn't crash on import.
    globalThis.document = { addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [], readyState: 'loading' };
    globalThis.window = globalThis;
    globalThis.navigator = { clipboard: null };
    globalThis.fetch = () => Promise.reject(new Error('no fetch'));

    const fs = require('fs');
    const path = process.argv[1];
    const inputB64 = process.argv[2];
    const src = fs.readFileSync(path, 'utf8');
    eval(src);

    const fmt = globalThis.Loomafmt;
    const input = JSON.parse(Buffer.from(inputB64, 'base64').toString('utf8'));
    const op = input.op;
    const out = {};

    if (op === 'formatTimestamp') {
        // NaN / undefined / null / numbers all funnel through here.
        // The harness may pass "NaN" as a string to test the fallback.
        let v = input.value;
        if (v === '__NaN__') v = NaN;
        out.results = fmt.formatTimestamp(v);
    } else if (op === 'escapeHtml') {
        out.results = fmt.escapeHtml(input.value);
    } else if (op === 'buildMarkdown') {
        out.results = fmt.buildMarkdown(input.value);
    } else {
        out.error = 'unknown op ' + op;
    }
    process.stdout.write(JSON.stringify(out));
    """
).strip()


def _run_js(op: str, value):
    """Run ``app.js`` in Node and return the helper's result."""
    import base64

    payload = json.dumps({"op": op, "value": value})
    b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    completed = subprocess.run(
        ["node", "-e", _LOADER_TEMPLATE, str(APP_JS_PATH), b64],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, (
        f"node failed: stdout={completed.stdout!r} "
        f"stderr={completed.stderr!r}"
    )
    return json.loads(completed.stdout).get("results")


# --- formatTimestamp -------------------------------------------------------


def test_format_timestamp_zero() -> None:
    assert _run_js("formatTimestamp", 0) == "0:00"


def test_format_timestamp_seconds_only() -> None:
    assert _run_js("formatTimestamp", 7) == "0:07"
    assert _run_js("formatTimestamp", 59) == "0:59"


def test_format_timestamp_minutes() -> None:
    assert _run_js("formatTimestamp", 60) == "1:00"
    assert _run_js("formatTimestamp", 125) == "2:05"
    assert _run_js("formatTimestamp", 599) == "9:59"


def test_format_timestamp_hours() -> None:
    assert _run_js("formatTimestamp", 3600) == "1:00:00"
    assert _run_js("formatTimestamp", 3725) == "1:02:05"


def test_format_timestamp_fractional_seconds_floors() -> None:
    assert _run_js("formatTimestamp", 5.9) == "0:05"


def test_format_timestamp_negative_falls_back() -> None:
    assert _run_js("formatTimestamp", -1) == "0:00"


def test_format_timestamp_nan_falls_back() -> None:
    """``NaN`` doesn't survive JSON, so we use a sentinel value."""
    assert _run_js("formatTimestamp", "__NaN__") == "0:00"


def test_format_timestamp_non_number_falls_back() -> None:
    assert _run_js("formatTimestamp", "abc") == "0:00"


# --- escapeHtml -----------------------------------------------------------


def test_escape_html_basic() -> None:
    assert _run_js("escapeHtml", "<script>alert(1)</script>") == (
        "&lt;script&gt;alert(1)&lt;/script&gt;"
    )


def test_escape_html_quotes() -> None:
    assert _run_js("escapeHtml", 'a"b\'c') == "a&quot;b&#39;c"


def test_escape_html_ampersand() -> None:
    assert _run_js("escapeHtml", "A & B") == "A &amp; B"


def test_escape_html_none() -> None:
    assert _run_js("escapeHtml", None) == ""


# --- buildMarkdown --------------------------------------------------------


def _sample_result() -> dict:
    return {
        "title": "Sample",
        "knowledge": {
            "title": "Sample Title",
            "summary": "Sentence one. Sentence two. Sentence three.",
            "insights": [
                "First imperative insight.",
                "Second imperative insight.",
            ],
            "chapters": [
                {"start_seconds": 0.0, "end_seconds": 60.0, "title": "Intro"},
                {"start_seconds": 60.0, "end_seconds": 180.0, "title": "Body"},
            ],
            "narrative": "Hello world " * 50,  # 100 words
        },
        "audio_url": "/audio/abc-1.mp3",
    }


def test_build_markdown_contains_title() -> None:
    md = _run_js("buildMarkdown", _sample_result())
    assert "# Sample Title" in md


def test_build_markdown_contains_summary_heading() -> None:
    md = _run_js("buildMarkdown", _sample_result())
    assert "## Summary" in md
    assert "Sentence one." in md


def test_build_markdown_contains_insights_as_bullets() -> None:
    md = _run_js("buildMarkdown", _sample_result())
    assert "## Key insights" in md
    assert "- First imperative insight." in md
    assert "- Second imperative insight." in md


def test_build_markdown_contains_chapter_timestamps() -> None:
    md = _run_js("buildMarkdown", _sample_result())
    assert "## Chapters" in md
    # 0s -> 0:00, 60s -> 1:00.
    assert "[0:00] Intro" in md
    assert "[1:00] Body" in md


def test_build_markdown_contains_narrative_section() -> None:
    md = _run_js("buildMarkdown", _sample_result())
    assert "## Narration" in md
    assert "Hello world" in md


def test_build_markdown_contains_audio_link() -> None:
    md = _run_js("buildMarkdown", _sample_result())
    assert "## Audio" in md
    assert "[Listen](/audio/abc-1.mp3)" in md


def test_build_markdown_handles_missing_audio_url() -> None:
    r = _sample_result()
    r["audio_url"] = None
    md = _run_js("buildMarkdown", r)
    # No "## Audio" section.
    assert "## Audio" not in md


def test_build_markdown_handles_empty_result() -> None:
    """An empty input renders a placeholder title rather than crashing.

    The frontend uses this fallback so the user always sees *something*
    when the LLM extractor returned no content. The placeholder is
    a single ``# Untitled`` heading — useful for debugging, never
    thrown at the user because the orchestrator only calls this with
    a fully-validated :class:`LoomaResult`.
    """
    md = _run_js("buildMarkdown", {})
    assert "# Untitled" in md
    # And the optional sections are absent.
    assert "## Summary" not in md
    assert "## Key insights" not in md


# --- Frontend file structure ----------------------------------------------


def test_frontend_index_html_exists() -> None:
    """The single-page entry point must exist on disk."""
    assert (FRONTEND_DIR / "index.html").exists()


def test_frontend_styles_css_exists() -> None:
    assert (FRONTEND_DIR / "styles.css").exists()


def test_frontend_app_js_exists() -> None:
    assert APP_JS_PATH.exists()


def test_index_html_has_required_anchors() -> None:
    """AC-8 requires specific UI elements; the HTML must scaffold them."""
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    # Input tabs
    assert 'id="tab-youtube"' in html
    assert 'id="tab-upload"' in html
    assert 'data-tab="youtube"' in html
    assert 'data-tab="upload"' in html

    # Forms
    assert 'id="form-youtube"' in html
    assert 'id="form-upload"' in html
    assert 'name="youtube_url"' in html
    assert 'name="file"' in html

    # Progress stages (the three async-job lifecycle stages)
    # — the old sync design had four (extracting, narrating) but
    # the async pipeline folds those two into ``transcribing``
    # because they're sub-second sub-pipelines.
    for stage in ("downloading", "transcribing", "done"):
        assert f'data-stage="{stage}"' in html, f"missing stage: {stage}"

    # Five result sections
    assert 'id="result-title"' in html
    assert 'id="result-summary"' in html
    assert 'id="result-insights"' in html
    assert 'id="result-chapters"' in html
    assert 'id="result-narrative"' in html

    # Audio player
    assert 'id="audio-player"' in html
    assert "<audio" in html
    assert 'controls' in html

    # Copy as Markdown button
    assert 'id="copy-md-button"' in html
    assert "Copy as Markdown" in html

    # Error display
    assert 'id="error"' in html
    assert 'id="error-code"' in html

    # Script and stylesheet links (cache-busting ``?v=N`` suffix is OK)
    assert 'href="/styles.css' in html
    assert 'src="/app.js' in html


def test_app_js_loads_under_node() -> None:
    """``app.js`` parses cleanly under Node — no syntax errors."""
    completed = subprocess.run(
        ["node", "--check", str(APP_JS_PATH)],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, (
        f"node --check failed: stderr={completed.stderr!r}"
    )


def test_app_js_exposes_loomafmt_namespace() -> None:
    """After evaluating ``app.js`` the ``Loomafmt`` global is present."""
    script = dedent(
        """
        globalThis.document = { addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [], readyState: 'loading' };
        globalThis.window = globalThis;
        globalThis.navigator = { clipboard: null };
        globalThis.fetch = () => Promise.reject(new Error('no fetch'));
        const fs = require('fs');
        const path = process.argv[1];
        eval(fs.readFileSync(path, 'utf8'));
        process.stdout.write(JSON.stringify({
          has_fmt: typeof globalThis.Loomafmt,
          has_formatTimestamp: typeof globalThis.Loomafmt?.formatTimestamp,
          has_buildMarkdown: typeof globalThis.Loomafmt?.buildMarkdown,
          has_escapeHtml: typeof globalThis.Loomafmt?.escapeHtml,
        }));
        """
    ).strip()
    completed = subprocess.run(
        ["node", "-e", script, str(APP_JS_PATH)],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "has_fmt": "object",
        "has_formatTimestamp": "function",
        "has_buildMarkdown": "function",
        "has_escapeHtml": "function",
    }
