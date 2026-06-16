"""Unit tests for the ``GET /audio/{job_id}.mp3`` route (AC-7).

These tests exercise the FastAPI app via :class:`httpx.AsyncClient`
against the in-process ASGI transport — no live network, no live
TTS, no live ffmpeg outside of the stub provider that the test
fixture wires up.

The MP3 file the route serves is produced by
:func:`app.pipeline.narrate._synthesize_stub_tts` (a real
ffmpeg-backed sine wave), so we can assert on ``Content-Type`` and
``Content-Length`` and on the bytes the client receives.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import httpx
import pytest

from app import main as main_mod
from app.config import OUTPUTS_DIR
from app.models import CODE_NOT_FOUND
from app.pipeline import narrate as narrate_mod


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build an isolated FastAPI app whose OUTPUTS_DIR points at tmp_path."""
    # Redirect the API's OUTPUTS_DIR to a fresh tmp dir so tests don't
    # pollute the real on-disk outputs directory.
    monkeypatch.setattr(main_mod, "OUTPUTS_DIR", tmp_path)
    app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def seeded_output(tmp_path, monkeypatch) -> Path:
    """Write a real stub MP3 to ``tmp_path/<job>.mp3`` for the test to fetch."""
    monkeypatch.setattr(main_mod, "OUTPUTS_DIR", tmp_path)
    out = tmp_path / "abc-123.mp3"
    narrate_mod._synthesize_stub_tts("hello world", out)
    return out


# --- Happy path: route returns the MP3 -------------------------------------


@pytest.mark.asyncio
async def test_get_audio_returns_200_with_mp3(
    client, seeded_output: Path
) -> None:
    async with client as c:
        resp = await c.get("/audio/abc-123.mp3")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    body = resp.content
    assert len(body) == seeded_output.stat().st_size
    assert body == seeded_output.read_bytes()


@pytest.mark.asyncio
async def test_get_audio_content_disposition(client, seeded_output: Path) -> None:
    async with client as c:
        resp = await c.get("/audio/abc-123.mp3")
    # FileResponse sets Content-Disposition by default.
    assert "attachment" in resp.headers.get("content-disposition", "")


# --- 404 path: missing file -------------------------------------------------


@pytest.mark.asyncio
async def test_get_audio_returns_404_for_missing_job(client) -> None:
    async with client as c:
        resp = await c.get("/audio/does-not-exist.mp3")
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["code"] == CODE_NOT_FOUND == "NOT_FOUND"
    assert "error" in payload


# --- Path-traversal defense -------------------------------------------------


@pytest.mark.asyncio
async def test_get_audio_rejects_path_traversal(client) -> None:
    async with client as c:
        resp = await c.get("/audio/..%2F..%2Fetc%2Fpasswd.mp3")
    # FastAPI will normally 404 a non-matching path, but our handler
    # also rejects unsafe ids. The 400 path is taken if the path
    # actually reaches the route.
    assert resp.status_code in (400, 404)
    if resp.status_code == 400:
        payload = resp.json()
        assert "error" in payload
        assert "code" in payload


@pytest.mark.asyncio
async def test_get_audio_rejects_special_chars(client) -> None:
    """Job IDs with slashes or spaces are rejected by the safe-id filter."""
    # FastAPI will normalize ``/`` in the path before reaching the
    # route, but other special characters pass through and should be
    # rejected with 400.
    async with client as c:
        resp = await c.get("/audio/has spaces.mp3")
    assert resp.status_code in (400, 404)


@pytest.mark.asyncio
async def test_is_safe_job_id_unit() -> None:
    """Direct probe of the safe-id helper."""
    assert main_mod._is_safe_job_id("abc-123")
    assert main_mod._is_safe_job_id("a")
    assert main_mod._is_safe_job_id("a_b-c-1-2-3")
    assert not main_mod._is_safe_job_id("../etc/passwd")
    assert not main_mod._is_safe_job_id("has spaces")
    assert not main_mod._is_safe_job_id("with/slash")
    assert not main_mod._is_safe_job_id("with.dot")
    assert not main_mod._is_safe_job_id("")
    assert not main_mod._is_safe_job_id("a" * 200)


# --- Health check -----------------------------------------------------------


@pytest.mark.asyncio
async def test_healthz_returns_ok(client) -> None:
    async with client as c:
        resp = await c.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# --- OpenAPI / docs still mount --------------------------------------------


@pytest.mark.asyncio
async def test_openapi_schema_loads(client) -> None:
    async with client as c:
        resp = await c.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    # The audio route is documented under ``/audio/{job_id}.mp3``.
    paths = schema.get("paths", {})
    assert "/audio/{job_id}.mp3" in paths


# --- No-state contamination -------------------------------------------------


@pytest.mark.asyncio
async def test_get_audio_404_does_not_serve_unrelated_file(client) -> None:
    """Asking for a missing job_id must never serve a different file."""
    async with client as c:
        resp = await c.get("/audio/zzz-missing.mp3")
    assert resp.status_code == 404
