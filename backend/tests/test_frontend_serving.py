"""Unit tests for the static frontend serving (AC-8).

These tests confirm that the FastAPI app correctly serves
``frontend/index.html`` at ``/`` and the sibling assets
(``styles.css``, ``app.js``) at their relative paths. No
real browser is launched — we use :class:`httpx.AsyncClient` and
the in-process ASGI transport.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import httpx
import pytest

from app import config as cfg
from app import main as main_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build an isolated app; the FRONTEND_DIR env var points at the
    real on-disk frontend so StaticFiles can find it."""
    import os
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    monkeypatch.setattr(main_mod, "_FRONTEND_DIR", frontend)
    monkeypatch.setattr(cfg, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# --- Frontend entry point ---------------------------------------------------


@pytest.mark.asyncio
async def test_get_index_returns_html(client) -> None:
    async with client as c:
        resp = await c.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # Must reference the Looma brand and the input tabs.
    assert "Looma" in body
    assert 'id="tab-youtube"' in body
    assert 'id="tab-upload"' in body


@pytest.mark.asyncio
async def test_get_styles_css(client) -> None:
    async with client as c:
        resp = await c.get("/styles.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]
    assert len(resp.text) > 0


@pytest.mark.asyncio
async def test_get_app_js(client) -> None:
    async with client as c:
        resp = await c.get("/app.js")
    assert resp.status_code == 200
    body = resp.text
    # The JS is large enough that something is wrong if it's tiny.
    assert len(body) > 1000
    # Must expose the Loomafmt namespace the tests rely on.
    assert "Loomafmt" in body


# --- Fallback when the frontend dir is missing -----------------------------


@pytest.mark.asyncio
async def test_frontend_warning_when_dir_missing(tmp_path, monkeypatch, caplog) -> None:
    """If the FRONTEND_DIR is missing, GET / returns 404 and the app
    does not crash."""
    import logging
    missing = tmp_path / "no-such-frontend"
    monkeypatch.setattr(main_mod, "_FRONTEND_DIR", missing)
    monkeypatch.setattr(cfg, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    with caplog.at_level(logging.WARNING):
        app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/")
    # With no StaticFiles mounted, the only route is the API/audio
    # ones; GET / has nothing to match. FastAPI's default 404 is OK.
    assert resp.status_code == 404
